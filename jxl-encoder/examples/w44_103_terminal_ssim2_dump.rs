//! W44-103: per-region + per-strategy SSIM2 attribution on terminal.png d=4
//!
//! Measurement-only chunk. NO production code change.
//!
//! Goal: identify WHERE the terminal SSIM2 cost vs cjxl comes from.
//!
//! Approach:
//! 1. Encode terminal.png at d=4 with effort 5..=9 using cjxl-rs AND cjxl
//! 2. Decode each, compute global + per-region SSIM2 (3x3 spatial grid)
//! 3. Compare per-region error maps between ours vs cjxl
//! 4. Cross-reference with per-strategy AC tokenization dump (W44-76 pattern)
//!
//! Output: benchmarks/w44_103_terminal_ssim2_dump_2026-05-19.tsv
//!
//! Run:
//!   cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
//!     --example w44_103_terminal_ssim2_dump
use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const TERMINAL: &str = "/home/lilith/work/codec-corpus/gb82-sc/terminal.png";
const DISTANCE: f32 = 4.0;
const EFFORTS: &[u8] = &[5, 6, 7, 8, 9];

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

// Extract a sub-region (x0,y0,w,h) from a full-image buffer.
// `src` is row-major RGB f32 (3 floats per pixel) for linear, or
// row-major [u8; 3] for sRGB.
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

// Score a (sub)image: returns (butteraugli, ssim2).
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

fn encode_ours(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> Option<(Vec<u8>, f64)> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1);
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
        "w44_103_cjxl_e{}_d{}.jxl",
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

// Generic "encode + decode + score in 3x3 grid" pipeline.
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

    // 3x3 regions
    // Make sure to round down to multiple of 32 to stay aligned with DCT32 blocks.
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
            // Region must be >= 8x8 for ssim2; sanity bound
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

fn main() {
    let out_dir = PathBuf::from("/home/lilith/work/zen/jxl-encoder/benchmarks");
    let tsv_path = out_dir.join("w44_103_terminal_ssim2_dump_2026-05-19.tsv");
    let meta_path = out_dir.join("w44_103_terminal_ssim2_dump_2026-05-19.meta");
    let dump_root = PathBuf::from("/tmp/w44_103_dumps");
    std::fs::create_dir_all(&dump_root).expect("mkdir dumps");

    let src_path = PathBuf::from(TERMINAL);
    let (rgb, w, h) = load_png(&src_path).expect("load terminal.png");
    eprintln!("Loaded terminal.png: {}x{}", w, h);

    let orig_lin = rgb_to_linear_img(&rgb, w, h);
    let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);

    // Open TSV
    let mut tsv = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tsv_path)
        .expect("open tsv");
    writeln!(
        tsv,
        "encoder\teffort\tdistance\tbytes\tencode_ms\tglobal_bfly\tglobal_ssim2\t\
         r00_ssim2\tr01_ssim2\tr02_ssim2\tr10_ssim2\tr11_ssim2\tr12_ssim2\tr20_ssim2\tr21_ssim2\tr22_ssim2\t\
         r00_bfly\tr01_bfly\tr02_bfly\tr10_bfly\tr11_bfly\tr12_bfly\tr20_bfly\tr21_bfly\tr22_bfly"
    )
    .unwrap();

    for &effort in EFFORTS {
        eprintln!("=== effort {} d={} ===", effort, DISTANCE);

        // OURS — with W44-76 dump
        let ours_dump = dump_root.join(format!("e{}_d{}_ours", effort, DISTANCE as i32));
        std::fs::create_dir_all(&ours_dump).unwrap();
        // SAFETY: setting env vars in a single-threaded harness for child encode.
        // unsafe gate per Rust 2024 edition.
        unsafe {
            std::env::set_var("JXL_W44_76_PER_BLOCK_DUMP", &ours_dump);
        }
        let Some((ours_bytes, ours_ms)) = encode_ours(&rgb, w, h, DISTANCE, effort) else {
            eprintln!("  ours encode FAILED");
            continue;
        };
        unsafe {
            std::env::remove_var("JXL_W44_76_PER_BLOCK_DUMP");
        }
        let Some(ours_r) = measure(&ours_bytes, ours_ms, &orig_lin, &orig_srgb, w, h) else {
            eprintln!("  ours measure FAILED");
            continue;
        };
        eprintln!(
            "  ours: bytes={} bfly={:.4} ssim2={:.4} ({:.0}ms)",
            ours_r.bytes, ours_r.global_bfly, ours_r.global_ssim2, ours_r.encode_ms
        );

        let cjxl_dump = dump_root.join(format!("e{}_d{}_cjxl", effort, DISTANCE as i32));
        std::fs::create_dir_all(&cjxl_dump).unwrap();
        let Some((cjxl_bytes, cjxl_ms)) =
            encode_cjxl(&src_path, DISTANCE, effort, Some(&cjxl_dump))
        else {
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
        eprintln!(
            "  delta: bytes={:+.2}%  ssim2={:+.4}  bfly={:+.2}%",
            100.0 * (ours_r.bytes as f64 - cjxl_r.bytes as f64) / cjxl_r.bytes as f64,
            ours_r.global_ssim2 - cjxl_r.global_ssim2,
            100.0 * (ours_r.global_bfly - cjxl_r.global_bfly) / cjxl_r.global_bfly,
        );

        for (label, r) in [("ours", &ours_r), ("cjxl", &cjxl_r)] {
            write!(
                tsv,
                "{}\t{}\t{}\t{}\t{:.2}\t{:.6}\t{:.4}",
                label, effort, DISTANCE, r.bytes, r.encode_ms, r.global_bfly, r.global_ssim2
            )
            .unwrap();
            for ry in 0..3 {
                for rx in 0..3 {
                    write!(tsv, "\t{:.4}", r.region_ssim2[ry][rx]).unwrap();
                }
            }
            for ry in 0..3 {
                for rx in 0..3 {
                    write!(tsv, "\t{:.6}", r.region_bfly[ry][rx]).unwrap();
                }
            }
            writeln!(tsv).unwrap();
        }

        // Print per-region SSIM2 delta table to stdout
        eprintln!("  per-region SSIM2 (ours - cjxl):");
        for ry in 0..3 {
            let row_top = if ry == 0 {
                "top"
            } else if ry == 1 {
                "mid"
            } else {
                "bot"
            };
            eprintln!(
                "    {}: {:+.3} {:+.3} {:+.3}",
                row_top,
                ours_r.region_ssim2[ry][0] - cjxl_r.region_ssim2[ry][0],
                ours_r.region_ssim2[ry][1] - cjxl_r.region_ssim2[ry][1],
                ours_r.region_ssim2[ry][2] - cjxl_r.region_ssim2[ry][2],
            );
        }
    }

    eprintln!();
    eprintln!("Wrote {}", tsv_path.display());

    // Brief meta footer
    let meta = format!(
        "# W44-103 terminal.png per-region SSIM2 attribution
# Bench: encode terminal.png at d=4 with effort {effort_list:?}
# Compare ours (cjxl-rs) vs cjxl (libjxl reference v0.12.0 with W44-76 dump patch)
# Dimensions: {w}x{h}; image: {src}
# Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
#
# 3x3 spatial region grid (region indices (ry, rx) where (0,0)=top-left):
#   top: r00 r01 r02
#   mid: r10 r11 r12
#   bot: r20 r21 r22
# Each region is `(w/3) & !31` x `(h/3) & !31` aligned to DCT32 blocks.
# Right column / bottom row get the remainder.
#
# Per-block strategy dumps for both encoders written to:
#   {dump_root}/e{{effort}}_d{distance}_ours/per_block_ours.tsv
#   {dump_root}/e{{effort}}_d{distance}_cjxl/per_block_libjxl.tsv
#
# Reproducer:
#   cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \\
#     --example w44_103_terminal_ssim2_dump
",
        effort_list = EFFORTS,
        w = w,
        h = h,
        src = TERMINAL,
        dump_root = dump_root.display(),
        distance = DISTANCE as i32
    );
    std::fs::write(&meta_path, meta).unwrap();
    eprintln!("Wrote {}", meta_path.display());
}
