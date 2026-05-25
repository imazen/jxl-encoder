//! W44-AUDIT-7 — wider-corpus cjxl-parity bench (Phase 1, ≤90min).
//!
//! Extends the W44-AUDIT-1 4-image bench to 20 images × 6 content classes,
//! covering screen-text, screen-graphics, photo-portrait, photo-landscape,
//! photo-smooth, and CLIC2025 web-photos. Sweeps efforts {5,7,9} × distances
//! {0.5, 1.0, 2.0, 4.0} × strategies {Zenjxl default, Libjxl strict-parity}
//! against cjxl 0.12.0 reference.
//!
//! Why: AUDIT-6 shipped M3-colourfulness-gated discriminators on the
//! W44-109/W44-105 screenshot qac seed scales, calibrated on 1 mixed-content
//! image (codec_wiki M3=145.73) vs 5 text screenshots (M3 ∈ [10,29]). A
//! wider corpus either CONFIRMS AUDIT-6 generalizes, FALSIFIES it, or
//! exposes new wedges.
//!
//! Output: a TSV with per-cell bytes/butteraugli/SSIM2 deltas for both
//! Zenjxl and Libjxl strategies vs cjxl, plus M3-colourfulness for each
//! image so Phase 2 can filter AUDIT-6 win/regression cells.
//!
//! Cells: 20 images × 3 efforts × 4 distances = 240 cells × 3 encodes = 720
//! encodes. Wall budget on a 7950X (release, parallel + butteraugli-loop +
//! ssim2-loop): ~40-70 minutes.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_7_wider_corpus -- \
//!     --output benchmarks/w44_audit_7_wider_corpus_2026-05-24.tsv

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

// ── Bench matrix ────────────────────────────────────────────────────────────

const EFFORTS: &[u8] = &[5, 7, 9];
const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 4.0];

struct Cell {
    image_id: &'static str,
    path: &'static str,
    class: &'static str,
}

// 20 images × 6 content classes. Sizes chosen to fit ~60 min budget.
//   - SCREEN_TEXT (3): terminal, codec_wiki, imac_g3 (large)
//   - SCREEN_GRAPHICS (3): graph, gui, windows95
//   - PHOTO_PORTRAIT (3): CID22 picks plausibly containing human faces
//   - PHOTO_LANDSCAPE (3): CID22 outdoor picks
//   - PHOTO_SMOOTH (3): low-edge-density CID22 incl W44-78 cluster cells
//   - CLIC2025_WEB (5): modern web photography at 1024² (mid-content)
const CELLS: &[Cell] = &[
    // ── SCREEN_TEXT (3) ─────────────────────────────────────────────────
    Cell {
        image_id: "terminal",
        path: "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        class: "SCREEN_TEXT",
    },
    Cell {
        image_id: "codec_wiki",
        path: "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        class: "SCREEN_TEXT",
    },
    Cell {
        image_id: "imac_g3",
        path: "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        class: "SCREEN_TEXT",
    },
    // ── SCREEN_GRAPHICS (3) ─────────────────────────────────────────────
    Cell {
        image_id: "graph",
        path: "/home/lilith/work/codec-corpus/gb82-sc/graph.png",
        class: "SCREEN_GRAPHICS",
    },
    Cell {
        image_id: "gui",
        path: "/home/lilith/work/codec-corpus/gb82-sc/gui.png",
        class: "SCREEN_GRAPHICS",
    },
    Cell {
        image_id: "windows95",
        path: "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
        class: "SCREEN_GRAPHICS",
    },
    // ── PHOTO_PORTRAIT (3) ──────────────────────────────────────────────
    Cell {
        image_id: "1025469",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
        class: "PHOTO_PORTRAIT",
    },
    Cell {
        image_id: "1418519",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        class: "PHOTO_PORTRAIT",
    },
    Cell {
        image_id: "1279330",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1279330.png",
        class: "PHOTO_PORTRAIT",
    },
    // ── PHOTO_LANDSCAPE (3) ─────────────────────────────────────────────
    Cell {
        image_id: "1189261",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
        class: "PHOTO_LANDSCAPE",
    },
    Cell {
        image_id: "1044329",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1044329.png",
        class: "PHOTO_LANDSCAPE",
    },
    Cell {
        image_id: "1475938",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png",
        class: "PHOTO_LANDSCAPE",
    },
    // ── PHOTO_SMOOTH (3) ────────────────────────────────────────────────
    Cell {
        image_id: "1531677",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        class: "PHOTO_SMOOTH",
    },
    Cell {
        image_id: "1420710",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        class: "PHOTO_SMOOTH",
    },
    Cell {
        image_id: "1544947",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1544947.png",
        class: "PHOTO_SMOOTH",
    },
    // ── CLIC2025_WEB (5) ────────────────────────────────────────────────
    // 1024×1024 modern web photos. Larger than CID22 (4× pixels).
    Cell {
        image_id: "clic_028092",
        path: "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
        class: "CLIC2025_WEB",
    },
    Cell {
        image_id: "clic_097cb4",
        path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png",
        class: "CLIC2025_WEB",
    },
    Cell {
        image_id: "clic_0c49a5",
        path: "/home/lilith/work/codec-corpus/clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png",
        class: "CLIC2025_WEB",
    },
    Cell {
        image_id: "clic_100a02",
        path: "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png",
        class: "CLIC2025_WEB",
    },
    Cell {
        image_id: "clic_22ea12",
        path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
        class: "CLIC2025_WEB",
    },
];

// ── Helpers ─────────────────────────────────────────────────────────────────

fn srgb_u8_to_linear_f32(x: u8) -> f32 {
    let x = x as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(x: f32) -> u8 {
    let x = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn make_imgs(pixels: &[u8], w: u32, h: u32) -> (Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>) {
    let lin: Vec<RGB<f32>> = pixels
        .chunks_exact(3)
        .map(|c| {
            RGB::new(
                srgb_u8_to_linear_f32(c[0]),
                srgb_u8_to_linear_f32(c[1]),
                srgb_u8_to_linear_f32(c[2]),
            )
        })
        .collect();
    let srgb: Vec<[u8; 3]> = pixels.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let lin_img = Img::new(lin, w as usize, h as usize);
    let srgb_img = Img::new(srgb, w as usize, h as usize);
    (lin_img, srgb_img)
}

/// Hasler-Süsstrunk M3 colourfulness over sRGB u8 pixels.
///
/// Bit-equivalent to `zenanalyze::tier1::m3_colourfulness` (which is itself a
/// port of `cv::saturation_colourfulness` from the Hasler-Süsstrunk 2003
/// paper). Used by W44-91 / W44-96 / W44-164 / W44-AUDIT-6 in-encoder.
/// Replicated here as a measurement aid — does NOT feed encoder dispatch.
///
/// Inputs: u8 sRGB pixels (3 per pixel).
/// Returns: M3 score (typically 0–200 range; CID22 photos: 5–90; high-colour
/// screens like codec_wiki/windows95: 100+; text-only screens: ~10–30).
fn compute_m3_colourfulness(pixels: &[u8]) -> f64 {
    let n = (pixels.len() / 3) as f64;
    if n < 1.0 {
        return 0.0;
    }
    let (mut sum_rg, mut sum_yb) = (0.0f64, 0.0f64);
    let (mut sum_rg2, mut sum_yb2) = (0.0f64, 0.0f64);
    for chunk in pixels.chunks_exact(3) {
        let r = chunk[0] as f64;
        let g = chunk[1] as f64;
        let b = chunk[2] as f64;
        let rg = r - g;
        let yb = 0.5 * (r + g) - b;
        sum_rg += rg;
        sum_yb += yb;
        sum_rg2 += rg * rg;
        sum_yb2 += yb * yb;
    }
    let mean_rg = sum_rg / n;
    let mean_yb = sum_yb / n;
    let var_rg = (sum_rg2 / n) - mean_rg * mean_rg;
    let var_yb = (sum_yb2 / n) - mean_yb * mean_yb;
    let sigma_rg_yb = (var_rg.max(0.0) + var_yb.max(0.0)).sqrt();
    let mu_rg_yb = (mean_rg * mean_rg + mean_yb * mean_yb).sqrt();
    sigma_rg_yb + 0.3 * mu_rg_yb
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

fn score_jxl(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    // Detect channel count via len/(dw*dh): cjxl-encoded source PNGs that have
    // alpha (e.g. gui.png in gb82-sc) round-trip as RGBA; we score as RGB
    // ignoring alpha. Three-channel naive `chunks_exact(3)` would read RGBA
    // pixels as if RGB triples, producing garbage metrics (SSIM2 = -204).
    let n_pixels = dw * dh;
    let channels = if n_pixels > 0 {
        dec_lin.len() / n_pixels
    } else {
        0
    };
    if channels < 3 {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks_exact(channels)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(
        orig_linear.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;

    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks_exact(channels)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;

    Some((bfly, ssim2))
}

fn encode_zenjxl(
    pixels: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    strategy: EncoderStrategy,
) -> Option<(Vec<u8>, u128)> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(strategy);
    let t0 = Instant::now();
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .ok()?;
    let ms = t0.elapsed().as_millis();
    Some((buf, ms))
}

fn encode_cjxl(src_path: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_audit7_{}_{}_{}_{}.jxl",
        src_path.file_stem()?.to_string_lossy(),
        effort,
        (distance * 1000.0) as u32,
        std::process::id()
    ));
    let t0 = Instant::now();
    let status = Command::new(cjxl)
        .arg(src_path)
        .arg(&tmp)
        .arg("-e")
        .arg(format!("{}", effort))
        .arg("-d")
        .arg(format!("{}", distance))
        .arg("--quiet")
        .status()
        .ok()?;
    let ms = t0.elapsed().as_millis();
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some((bytes, ms))
}

// ── Main ────────────────────────────────────────────────────────────────────

#[derive(Default, Debug, Clone)]
struct Row {
    image_id: String,
    class: String,
    width: u32,
    height: u32,
    m3_colourfulness: f64,
    effort: u8,
    distance: f32,
    cjxl_bytes: usize,
    cjxl_bfly: f64,
    cjxl_ssim2: f64,
    cjxl_encode_ms: u128,
    zenjxl_bytes: usize,
    zenjxl_bfly: f64,
    zenjxl_ssim2: f64,
    zenjxl_encode_ms: u128,
    libjxl_strat_bytes: usize,
    libjxl_strat_bfly: f64,
    libjxl_strat_ssim2: f64,
    libjxl_strat_encode_ms: u128,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out_path: PathBuf = PathBuf::from("benchmarks/w44_audit_7_wider_corpus_2026-05-24.tsv");
    let mut filter_image: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            out_path = PathBuf::from(&args[i + 1]);
            i += 2;
        } else if args[i] == "--only" && i + 1 < args.len() {
            filter_image = Some(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    eprintln!("[bench] output: {}", out_path.display());
    eprintln!(
        "[bench] {} images × {} efforts × {} distances = {} cells",
        CELLS.len(),
        EFFORTS.len(),
        DISTANCES.len(),
        CELLS.len() * EFFORTS.len() * DISTANCES.len()
    );

    let total_cells = CELLS.len() * EFFORTS.len() * DISTANCES.len();
    let mut rows: Vec<Row> = Vec::with_capacity(total_cells);
    let bench_start = Instant::now();
    let mut done = 0;

    let cells: Vec<&Cell> = if let Some(ref filt) = filter_image {
        CELLS
            .iter()
            .filter(|c| c.image_id == filt.as_str())
            .collect()
    } else {
        CELLS.iter().collect()
    };
    eprintln!("[bench] filtered cells: {}", cells.len());

    for cell in cells {
        let (pixels, w, h) = match load_png(Path::new(cell.path)) {
            Some(t) => t,
            None => {
                eprintln!("[bench] FAIL: load {}", cell.path);
                continue;
            }
        };
        let m3 = compute_m3_colourfulness(&pixels);
        eprintln!(
            "[bench] LOADED {} {}x{} class={} M3={:.2}",
            cell.image_id, w, h, cell.class, m3
        );
        let (lin_img, srgb_img) = make_imgs(&pixels, w, h);

        for &effort in EFFORTS {
            for &distance in DISTANCES {
                done += 1;
                let t_cell = Instant::now();
                eprint!(
                    "[bench] {}/{} {} e{} d={} ... ",
                    done, total_cells, cell.image_id, effort, distance
                );

                let mut row = Row {
                    image_id: cell.image_id.to_string(),
                    class: cell.class.to_string(),
                    width: w,
                    height: h,
                    m3_colourfulness: m3,
                    effort,
                    distance,
                    ..Default::default()
                };

                if let Some((c_bytes, c_ms)) = encode_cjxl(Path::new(cell.path), effort, distance) {
                    row.cjxl_bytes = c_bytes.len();
                    row.cjxl_encode_ms = c_ms;
                    if let Some((bfly, ss)) = score_jxl(&c_bytes, &lin_img, &srgb_img, w, h) {
                        row.cjxl_bfly = bfly;
                        row.cjxl_ssim2 = ss;
                    }
                }

                if let Some((z_bytes, z_ms)) =
                    encode_zenjxl(&pixels, w, h, effort, distance, EncoderStrategy::Zenjxl)
                {
                    row.zenjxl_bytes = z_bytes.len();
                    row.zenjxl_encode_ms = z_ms;
                    if let Some((bfly, ss)) = score_jxl(&z_bytes, &lin_img, &srgb_img, w, h) {
                        row.zenjxl_bfly = bfly;
                        row.zenjxl_ssim2 = ss;
                    }
                }

                if let Some((l_bytes, l_ms)) =
                    encode_zenjxl(&pixels, w, h, effort, distance, EncoderStrategy::Libjxl)
                {
                    row.libjxl_strat_bytes = l_bytes.len();
                    row.libjxl_strat_encode_ms = l_ms;
                    if let Some((bfly, ss)) = score_jxl(&l_bytes, &lin_img, &srgb_img, w, h) {
                        row.libjxl_strat_bfly = bfly;
                        row.libjxl_strat_ssim2 = ss;
                    }
                }

                let elapsed_ms = t_cell.elapsed().as_millis();
                eprintln!(
                    "z={}B/{:.3}/{:.2} l={}B/{:.3}/{:.2} c={}B/{:.3}/{:.2} ({}ms)",
                    row.zenjxl_bytes,
                    row.zenjxl_bfly,
                    row.zenjxl_ssim2,
                    row.libjxl_strat_bytes,
                    row.libjxl_strat_bfly,
                    row.libjxl_strat_ssim2,
                    row.cjxl_bytes,
                    row.cjxl_bfly,
                    row.cjxl_ssim2,
                    elapsed_ms
                );
                rows.push(row);

                // Incremental TSV flush every 10 cells so partial results survive
                // a wall-budget abort.
                if done % 10 == 0 {
                    write_tsv(&out_path, &rows);
                }
            }
        }
    }

    write_tsv(&out_path, &rows);
    eprintln!(
        "[bench] {} rows written to {} in {:.1}s",
        rows.len(),
        out_path.display(),
        bench_start.elapsed().as_secs_f64()
    );
}

fn write_tsv(out_path: &Path, rows: &[Row]) {
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = match File::create(out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[bench] FAIL create TSV: {}", e);
            return;
        }
    };
    writeln!(
        f,
        "image_id\tclass\twidth\theight\tm3_colourfulness\teffort\tdistance\t\
         cjxl_bytes\tcjxl_bfly\tcjxl_ssim2\tcjxl_encode_ms\t\
         zenjxl_bytes\tzenjxl_bfly\tzenjxl_ssim2\tzenjxl_encode_ms\t\
         libjxl_strat_bytes\tlibjxl_strat_bfly\tlibjxl_strat_ssim2\tlibjxl_strat_encode_ms\t\
         zenjxl_dBytes_pct\tzenjxl_dSsim2\tzenjxl_dBfly\t\
         libjxl_dBytes_pct\tlibjxl_dSsim2\tlibjxl_dBfly"
    )
    .unwrap();
    for r in rows {
        let z_dbytes = if r.cjxl_bytes > 0 {
            (r.zenjxl_bytes as f64 - r.cjxl_bytes as f64) / r.cjxl_bytes as f64 * 100.0
        } else {
            f64::NAN
        };
        let z_dssim2 = r.zenjxl_ssim2 - r.cjxl_ssim2;
        let z_dbfly = r.zenjxl_bfly - r.cjxl_bfly;
        let l_dbytes = if r.cjxl_bytes > 0 {
            (r.libjxl_strat_bytes as f64 - r.cjxl_bytes as f64) / r.cjxl_bytes as f64 * 100.0
        } else {
            f64::NAN
        };
        let l_dssim2 = r.libjxl_strat_ssim2 - r.cjxl_ssim2;
        let l_dbfly = r.libjxl_strat_bfly - r.cjxl_bfly;
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {:.3}\t{:.3}\t{:.4}\t\
             {:.3}\t{:.3}\t{:.4}",
            r.image_id,
            r.class,
            r.width,
            r.height,
            r.m3_colourfulness,
            r.effort,
            r.distance,
            r.cjxl_bytes,
            r.cjxl_bfly,
            r.cjxl_ssim2,
            r.cjxl_encode_ms,
            r.zenjxl_bytes,
            r.zenjxl_bfly,
            r.zenjxl_ssim2,
            r.zenjxl_encode_ms,
            r.libjxl_strat_bytes,
            r.libjxl_strat_bfly,
            r.libjxl_strat_ssim2,
            r.libjxl_strat_encode_ms,
            z_dbytes,
            z_dssim2,
            z_dbfly,
            l_dbytes,
            l_dssim2,
            l_dbfly
        )
        .unwrap();
    }
}
