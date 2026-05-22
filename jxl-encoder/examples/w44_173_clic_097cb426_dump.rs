//! W44-173: per-region + per-strategy SSIM2 attribution on
//! clic_097cb426 (CLIC 2025 smooth photo) — diagnose the persistent
//! SSIM2 deficit found by W44-170 across BOTH EncoderStrategy::Zenjxl
//! and ::Libjxl at d=3/4/5 (deficit -3 to -4.5 SSIM2 vs cjxl).
//!
//! Read-only measurement chunk. NO production code change.
//!
//! Per W44-170 TSV, e7/e8/e9 cells are BYTE-IDENTICAL between Zen and
//! Libjxl strategies — meaning the existing zen content-aware gates
//! (W22-1 screenshot lift, W44-29 high-d photo, W44-91 photo widen,
//! W44-96 variant Z, etc.) DO NOT FIRE on this image at those efforts.
//! The SSIM2 deficit is therefore STRUCTURAL (shared cost-model /
//! recon-pipeline divergence), not a strategy-gate miss.
//!
//! e5/e6 cells DO differ between strategies (Libjxl is byte-worse,
//! SSIM2 similar). We measure both strategies at e6/e7/e8 to confirm
//! the W44-170 ledger numbers and to characterize the per-strategy
//! split at e6.
//!
//! Mirrors W44-103 terminal d=4 + W44-121 codec_wiki d=3 methodology:
//! 1. Encode at d=3, d=4, d=5 with both Zenjxl + Libjxl, threads=1
//! 2. Decode each via jxl-oxide (linear sRGB); cjxl as reference
//! 3. Global + per-region (3×3) SSIM2 + butteraugli
//! 4. Per-strategy AC tokenization dump (W44-76) for worst cell
//! 5. ZenanalyzeProxies (m3_colourfulness, fcbr, edge_density) +
//!    mask1x1 median/p25 estimate for cluster-pattern matching.
//!
//! Output: benchmarks/w44_173_clic_097cb426_dump_2026-05-21.tsv
//!         benchmarks/w44_173_clic_097cb426_dump_2026-05-21.meta
//!
//! Run:
//!   cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
//!     --example w44_173_clic_097cb426_dump
use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SRC_PNG: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png";
const SHORT_NAME: &str = "clic_097cb426";
const DISTANCES: &[f32] = &[3.0, 4.0, 5.0];
const EFFORTS: &[u8] = &[6, 7, 8];
const WORST_EFFORT_FOR_DUMP: u8 = 7; // Larger e=8 with buttloop = expensive
const WORST_DISTANCE_FOR_DUMP: f32 = 5.0; // Biggest SSIM2 deficit per W44-170

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
}

// sRGB ↔ linear ─────────────────────────────────────────────────────────────

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn rgb_to_linear_img(rgb: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let pixels: Vec<RGB<f32>> = rgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear_f32(c[0]),
                srgb_to_linear_f32(c[1]),
                srgb_to_linear_f32(c[2]),
            )
        })
        .collect();
    Img::new(pixels, w as usize, h as usize)
}

fn rgb_to_srgb_arr3(rgb: &[u8], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w as usize, h as usize)
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn extract_region_lin(
    full: &Img<Vec<RGB<f32>>>,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Img<Vec<RGB<f32>>> {
    let stride = full.width();
    let mut out = Vec::with_capacity(w * h);
    let buf = full.buf();
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            out.push(buf[y * stride + x]);
        }
    }
    Img::new(out, w, h)
}

fn extract_region_srgb(
    full: &Img<Vec<[u8; 3]>>,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Img<Vec<[u8; 3]>> {
    let stride = full.width();
    let mut out = Vec::with_capacity(w * h);
    let buf = full.buf();
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            out.push(buf[y * stride + x]);
        }
    }
    Img::new(out, w, h)
}

fn score_region(
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    dec_lin: &Img<Vec<RGB<f32>>>,
    dec_srgb: &Img<Vec<[u8; 3]>>,
) -> Option<(f64, f64)> {
    let bfly = butteraugli_linear(
        orig_lin.as_ref(),
        dec_lin.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn encode_ours(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    strategy: EncoderStrategy,
) -> Option<(Vec<u8>, f64)> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(strategy);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes, ms))
}

fn encode_cjxl(
    src_png: &Path,
    distance: f32,
    effort: u8,
    dump_dir: Option<&Path>,
) -> Option<(Vec<u8>, f64)> {
    let tmp = std::env::temp_dir().join(format!(
        "w44_173_cjxl_e{}_d{}.jxl",
        effort,
        distance.to_string().replace('.', "_")
    ));
    let bin = cjxl_bin();
    let mut cmd = Command::new(&bin);
    cmd.arg(src_png)
        .arg(&tmp)
        .args(["-e", &effort.to_string()])
        .args(["-d", &distance.to_string()])
        .args(["--num_threads", "1"])
        .arg("--quiet");
    if let Some(dir) = dump_dir {
        cmd.env("JXL_W44_76_PER_BLOCK_DUMP", dir);
    }
    let start = Instant::now();
    let status = cmd.status().ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some((bytes, ms))
}

struct CellResult {
    bytes: usize,
    encode_ms: f64,
    global_bfly: f64,
    global_ssim2: f64,
    region_ssim2: [[f64; 3]; 3],
    region_bfly: [[f64; 3]; 3],
}

fn measure(
    bytes: &[u8],
    encode_ms: f64,
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<CellResult> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);

    let (gb, gs) = score_region(orig_lin, orig_srgb, &dec_lin_img, &dec_srgb_img)?;

    // 3x3 regions, aligned to multiples of 32 (DCT32 block boundary)
    let region_w = (dw / 3) & !31;
    let region_h = (dh / 3) & !31;
    let mut region_ssim2 = [[0.0; 3]; 3];
    let mut region_bfly = [[0.0; 3]; 3];
    for ry in 0..3usize {
        for rx in 0..3usize {
            let x0 = rx * region_w;
            let y0 = ry * region_h;
            let rw = if rx == 2 { dw - x0 } else { region_w };
            let rh = if ry == 2 { dh - y0 } else { region_h };
            if rw < 64 || rh < 64 {
                continue;
            }
            let o_lin = extract_region_lin(orig_lin, x0, y0, rw, rh);
            let o_srgb = extract_region_srgb(orig_srgb, x0, y0, rw, rh);
            let d_lin = extract_region_lin(&dec_lin_img, x0, y0, rw, rh);
            let d_srgb = extract_region_srgb(&dec_srgb_img, x0, y0, rw, rh);
            if let Some((b, s)) = score_region(&o_lin, &o_srgb, &d_lin, &d_srgb) {
                region_bfly[ry][rx] = b;
                region_ssim2[ry][rx] = s;
            }
        }
    }

    Some(CellResult {
        bytes: bytes.len(),
        encode_ms,
        global_bfly: gb,
        global_ssim2: gs,
        region_ssim2,
        region_bfly,
    })
}

// ── ZenanalyzeProxies (local replica — read-only diagnostic) ─────────────
// Matches `vardct::encoder::ZenanalyzeProxies::compute_srgb_u8` exactly.

#[derive(Debug, Clone, Copy)]
struct LocalProxies {
    m3_colourfulness: f32,
    flat_color_block_ratio: f32,
    edge_density: f32,
}

fn compute_proxies(pixels: &[u8], width: usize, height: usize) -> LocalProxies {
    // bpp=3 RGB
    let n_pix = (width * height) as f64;
    if n_pix == 0.0 {
        return LocalProxies {
            m3_colourfulness: 0.0,
            flat_color_block_ratio: 0.0,
            edge_density: 0.0,
        };
    }
    // M3 colourfulness
    let mut rg_sum = 0.0f64;
    let mut rg_sq = 0.0f64;
    let mut yb_sum = 0.0f64;
    let mut yb_sq = 0.0f64;
    for y in 0..height {
        for x in 0..width {
            let off = (y * width + x) * 3;
            let r = pixels[off] as f64;
            let g = pixels[off + 1] as f64;
            let b = pixels[off + 2] as f64;
            let rg = r - g;
            let yb = 0.5 * (r + g) - b;
            rg_sum += rg;
            rg_sq += rg * rg;
            yb_sum += yb;
            yb_sq += yb * yb;
        }
    }
    let mu_rg = rg_sum / n_pix;
    let mu_yb = yb_sum / n_pix;
    let var_rg = (rg_sq / n_pix - mu_rg * mu_rg).max(0.0);
    let var_yb = (yb_sq / n_pix - mu_yb * mu_yb).max(0.0);
    let m3 = ((var_rg + var_yb).sqrt() + 0.3 * (mu_rg * mu_rg + mu_yb * mu_yb).sqrt()) as f32;

    // flat_color_block_ratio: 8x8 blocks
    let bx = width / 8;
    let by = height / 8;
    let mut flat = 0usize;
    let total = bx * by;
    for blk_y in 0..by {
        for blk_x in 0..bx {
            let mut rmin = 255u8;
            let mut rmax = 0u8;
            let mut gmin = 255u8;
            let mut gmax = 0u8;
            let mut bmin = 255u8;
            let mut bmax = 0u8;
            for dy in 0..8 {
                for dx in 0..8 {
                    let off = ((blk_y * 8 + dy) * width + (blk_x * 8 + dx)) * 3;
                    let r = pixels[off];
                    let g = pixels[off + 1];
                    let b = pixels[off + 2];
                    if r < rmin {
                        rmin = r;
                    }
                    if r > rmax {
                        rmax = r;
                    }
                    if g < gmin {
                        gmin = g;
                    }
                    if g > gmax {
                        gmax = g;
                    }
                    if b < bmin {
                        bmin = b;
                    }
                    if b > bmax {
                        bmax = b;
                    }
                }
            }
            if (rmax - rmin) <= 4 && (gmax - gmin) <= 4 && (bmax - bmin) <= 4 {
                flat += 1;
            }
        }
    }
    let fcbr = if total == 0 {
        0.0
    } else {
        flat as f32 / total as f32
    };

    // edge_density: Sobel luma gradient magnitude > 30 (BT.601)
    let mut high = 0usize;
    let mut interior = 0usize;
    let luma = |idx: usize| -> f32 {
        let r = pixels[idx] as f32;
        let g = pixels[idx + 1] as f32;
        let b = pixels[idx + 2] as f32;
        0.299 * r + 0.587 * g + 0.114 * b
    };
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let center = (y * width + x) * 3;
            let tl = ((y - 1) * width + (x - 1)) * 3;
            let t = ((y - 1) * width + x) * 3;
            let tr = ((y - 1) * width + (x + 1)) * 3;
            let l = (y * width + (x - 1)) * 3;
            let r = (y * width + (x + 1)) * 3;
            let bl = ((y + 1) * width + (x - 1)) * 3;
            let b = ((y + 1) * width + x) * 3;
            let br = ((y + 1) * width + (x + 1)) * 3;
            let _ = center; // not used; Sobel skips center
            let gx = -luma(tl) - 2.0 * luma(l) - luma(bl) + luma(tr) + 2.0 * luma(r) + luma(br);
            let gy = -luma(tl) - 2.0 * luma(t) - luma(tr) + luma(bl) + 2.0 * luma(b) + luma(br);
            let mag = (gx * gx + gy * gy).sqrt();
            if mag > 30.0 {
                high += 1;
            }
            interior += 1;
        }
    }
    let edge_density = if interior == 0 {
        0.0
    } else {
        high as f32 / interior as f32
    };

    LocalProxies {
        m3_colourfulness: m3,
        flat_color_block_ratio: fcbr,
        edge_density,
    }
}

fn main() {
    let out_dir = PathBuf::from("/home/lilith/work/zen/jxl-encoder/benchmarks");
    let tsv_path = out_dir.join("w44_173_clic_097cb426_dump_2026-05-21.tsv");
    let meta_path = out_dir.join("w44_173_clic_097cb426_dump_2026-05-21.meta");
    let proxies_path = out_dir.join("w44_173_clic_097cb426_proxies_2026-05-21.txt");
    let dump_root = PathBuf::from("/tmp/w44_173_dumps");
    std::fs::create_dir_all(&dump_root).expect("mkdir dumps");

    let src_path = PathBuf::from(SRC_PNG);
    let (rgb, w, h) = load_png(&src_path).expect("load source");
    eprintln!("Loaded {}: {}x{} ({} bytes)", SHORT_NAME, w, h, rgb.len());

    // ZenanalyzeProxies for cluster-pattern matching
    let proxies = compute_proxies(&rgb, w as usize, h as usize);
    eprintln!("ZenanalyzeProxies: {:?}", proxies);
    let mut pf = std::fs::File::create(&proxies_path).unwrap();
    writeln!(pf, "# W44-173 zenanalyze proxies for clic_097cb426").unwrap();
    writeln!(pf, "image\t{}", SHORT_NAME).unwrap();
    writeln!(pf, "src\t{}", SRC_PNG).unwrap();
    writeln!(pf, "width\t{}", w).unwrap();
    writeln!(pf, "height\t{}", h).unwrap();
    writeln!(pf, "m3_colourfulness\t{:.4}", proxies.m3_colourfulness).unwrap();
    writeln!(
        pf,
        "flat_color_block_ratio\t{:.6}",
        proxies.flat_color_block_ratio
    )
    .unwrap();
    writeln!(pf, "edge_density\t{:.6}", proxies.edge_density).unwrap();
    writeln!(pf, "\n# Discriminator gates (compare to thresholds):").unwrap();
    writeln!(pf, "# W44-91 m3>=80 + fcbr<0.01 → high-d photo widen").unwrap();
    writeln!(
        pf,
        "# W44-96 edge_density>=0.7 + fcbr<0.01 + mask<50 → variant Z"
    )
    .unwrap();
    writeln!(pf, "# W44-98 m3>=25 (within W44-96) → HC variant Z").unwrap();
    writeln!(pf, "# W44-99 m3<25 (within W44-96) → LC variant Z").unwrap();
    writeln!(
        pf,
        "# W44-124 m3<25 + edge_density<0.16 + mask_p25>=W44_124_MIN+distance gate"
    )
    .unwrap();
    writeln!(
        pf,
        "# W44-166 mask_p25>=85 + distance>=4.5 → variant Z admit"
    )
    .unwrap();
    writeln!(
        pf,
        "# W44-168 mask_p25>=85 → smooth-photo iter skip; edge_density>=0.5 → textured iter bump"
    )
    .unwrap();

    let orig_lin = rgb_to_linear_img(&rgb, w, h);
    let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);

    let mut tsv = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tsv_path)
        .expect("open tsv");
    writeln!(
        tsv,
        "encoder\tstrategy\teffort\tdistance\tbytes\tencode_ms\tglobal_bfly\tglobal_ssim2\t\
         r00_ssim2\tr01_ssim2\tr02_ssim2\tr10_ssim2\tr11_ssim2\tr12_ssim2\tr20_ssim2\tr21_ssim2\tr22_ssim2\t\
         r00_bfly\tr01_bfly\tr02_bfly\tr10_bfly\tr11_bfly\tr12_bfly\tr20_bfly\tr21_bfly\tr22_bfly"
    )
    .unwrap();

    for &effort in EFFORTS {
        for &distance in DISTANCES {
            for (strat_name, strat) in [
                ("zenjxl", EncoderStrategy::Zenjxl),
                ("libjxl", EncoderStrategy::Libjxl),
            ] {
                eprintln!("=== {} e{} d={} ===", strat_name, effort, distance);

                // OURS — with W44-76 dump only on the worst cell (e7 d=5) for each strategy
                let dump_this = effort == WORST_EFFORT_FOR_DUMP
                    && (distance - WORST_DISTANCE_FOR_DUMP).abs() < 0.01;
                let ours_dump = dump_root.join(format!(
                    "{}_e{}_d{}_ours",
                    strat_name, effort, distance as i32
                ));
                if dump_this {
                    std::fs::create_dir_all(&ours_dump).unwrap();
                    unsafe {
                        std::env::set_var("JXL_W44_76_PER_BLOCK_DUMP", &ours_dump);
                    }
                }
                let Some((ours_bytes, ours_ms)) =
                    encode_ours(&rgb, w, h, distance, effort, strat.clone())
                else {
                    eprintln!("  ours encode FAILED");
                    if dump_this {
                        unsafe {
                            std::env::remove_var("JXL_W44_76_PER_BLOCK_DUMP");
                        }
                    }
                    continue;
                };
                if dump_this {
                    unsafe {
                        std::env::remove_var("JXL_W44_76_PER_BLOCK_DUMP");
                    }
                }
                let Some(ours_r) = measure(&ours_bytes, ours_ms, &orig_lin, &orig_srgb, w, h)
                else {
                    eprintln!("  ours measure FAILED");
                    continue;
                };
                eprintln!(
                    "  ours: bytes={} bfly={:.4} ssim2={:.4} ({:.0}ms)",
                    ours_r.bytes, ours_r.global_bfly, ours_r.global_ssim2, ours_r.encode_ms
                );

                write!(
                    tsv,
                    "ours\t{}\t{}\t{}\t{}\t{:.2}\t{:.6}\t{:.4}",
                    strat_name,
                    effort,
                    distance,
                    ours_r.bytes,
                    ours_r.encode_ms,
                    ours_r.global_bfly,
                    ours_r.global_ssim2
                )
                .unwrap();
                for ry in 0..3 {
                    for rx in 0..3 {
                        write!(tsv, "\t{:.4}", ours_r.region_ssim2[ry][rx]).unwrap();
                    }
                }
                for ry in 0..3 {
                    for rx in 0..3 {
                        write!(tsv, "\t{:.6}", ours_r.region_bfly[ry][rx]).unwrap();
                    }
                }
                writeln!(tsv).unwrap();
            }

            // cjxl reference — only encode once per (effort, distance), it doesn't depend on strategy.
            eprintln!("=== cjxl e{} d={} ===", effort, distance);
            let dump_this = effort == WORST_EFFORT_FOR_DUMP
                && (distance - WORST_DISTANCE_FOR_DUMP).abs() < 0.01;
            let cjxl_dump = dump_root.join(format!("cjxl_e{}_d{}", effort, distance as i32));
            if dump_this {
                std::fs::create_dir_all(&cjxl_dump).unwrap();
            }
            let Some((cjxl_bytes, cjxl_ms)) = encode_cjxl(
                &src_path,
                distance,
                effort,
                if dump_this { Some(&cjxl_dump) } else { None },
            ) else {
                eprintln!("  cjxl encode FAILED");
                continue;
            };
            let Some(cjxl_r) = measure(&cjxl_bytes, cjxl_ms, &orig_lin, &orig_srgb, w, h) else {
                eprintln!("  cjxl measure FAILED");
                continue;
            };
            eprintln!(
                "  cjxl: bytes={} bfly={:.4} ssim2={:.4} ({:.0}ms)",
                cjxl_r.bytes, cjxl_r.global_bfly, cjxl_r.global_ssim2, cjxl_r.encode_ms
            );
            write!(
                tsv,
                "cjxl\tNA\t{}\t{}\t{}\t{:.2}\t{:.6}\t{:.4}",
                effort,
                distance,
                cjxl_r.bytes,
                cjxl_r.encode_ms,
                cjxl_r.global_bfly,
                cjxl_r.global_ssim2
            )
            .unwrap();
            for ry in 0..3 {
                for rx in 0..3 {
                    write!(tsv, "\t{:.4}", cjxl_r.region_ssim2[ry][rx]).unwrap();
                }
            }
            for ry in 0..3 {
                for rx in 0..3 {
                    write!(tsv, "\t{:.6}", cjxl_r.region_bfly[ry][rx]).unwrap();
                }
            }
            writeln!(tsv).unwrap();

            eprintln!("  per-region SSIM2 (cjxl):");
            for ry in 0..3 {
                let row_top = if ry == 0 {
                    "top"
                } else if ry == 1 {
                    "mid"
                } else {
                    "bot"
                };
                eprintln!(
                    "    {}: {:.3} {:.3} {:.3}",
                    row_top,
                    cjxl_r.region_ssim2[ry][0],
                    cjxl_r.region_ssim2[ry][1],
                    cjxl_r.region_ssim2[ry][2],
                );
            }
        }
    }

    eprintln!();
    eprintln!("Wrote {}", tsv_path.display());

    let meta = format!(
        "# W44-173 clic_097cb426 per-region + per-strategy SSIM2 attribution
# Bench: encode {short} ({src}) at d={{3,4,5}} with effort {{6,7,8}}, threads=1
# Strategies: Zenjxl + Libjxl; cjxl reference v0.12.0
# Dimensions: {w}x{h}
# Date: 2026-05-21
#
# Predecessor: W44-103 (terminal d=4), W44-121 (codec_wiki d=3).
# W44-170 finding: clic_097cb426 SSIM2 deficit -3 to -4.5 vs cjxl at d=3+
# is STRUCTURAL — affects BOTH EncoderStrategy::Zenjxl and ::Libjxl.
# At e7/e8/e9 the two strategies produce BYTE-IDENTICAL output (no zen
# content-aware gate fires on this smooth-photo content).
#
# 3x3 spatial region grid (region indices (ry, rx) where (0,0)=top-left):
#   top: r00 r01 r02
#   mid: r10 r11 r12
#   bot: r20 r21 r22
# Each region is `(w/3) & !31` x `(h/3) & !31`, aligned to DCT32 blocks.
#
# Per-block strategy dumps for the worst cell (e{worst_e} d={worst_d}) for
# each strategy + cjxl written to:
#   /tmp/w44_173_dumps/zenjxl_e{worst_e}_d{worst_d}_ours/per_block_ours.tsv
#   /tmp/w44_173_dumps/libjxl_e{worst_e}_d{worst_d}_ours/per_block_ours.tsv
#   /tmp/w44_173_dumps/cjxl_e{worst_e}_d{worst_d}/per_block_libjxl.tsv
#
# ZenanalyzeProxies (for cluster-pattern matching):
# {proxies_path}
#
# Reproducer:
#   cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \\
#     --example w44_173_clic_097cb426_dump
",
        short = SHORT_NAME,
        src = SRC_PNG,
        w = w,
        h = h,
        worst_e = WORST_EFFORT_FOR_DUMP,
        worst_d = WORST_DISTANCE_FOR_DUMP as i32,
        proxies_path = proxies_path.display(),
    );
    std::fs::write(&meta_path, meta).unwrap();
    eprintln!("Wrote {}", meta_path.display());
}
