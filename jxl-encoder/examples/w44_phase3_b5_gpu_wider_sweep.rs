//! W44-phase3-B5 — wider 30+ cell sweep of GPU butteraugli backend vs CPU baseline.
//!
//! Follow-on to W44-phase3-B4 (commit c121c08e) which measured 11-21% wall
//! reduction on 6 cells with zero regressions. B5 broadens to 38 cells across
//! multiple resolutions (0.26 MP photos → 5.62 MP screens), efforts (e8, e9),
//! distances (0.5, 1.0, 2.0, 5.0), and corpora (CID22, CLIC2025, gb82-sc) to
//! decide whether to flip `gpu-butteraugli` default ON for builds that have
//! the feature compiled in.
//!
//! ## Cell design (38 cells)
//!
//! - 5 CID22 photos (0.26 MP) × {0.5, 1.0, 2.0, 5.0} × e8 = 20
//! - 5 CID22 photos (0.26 MP) × {1.0, 2.0}       × e9 = 10
//! - 2 CLIC photos (1.05 MP) × {1.0, 2.0}        × e8 = 4
//! - terminal    (1.75 MP) × {1.0, 4.0}          × e8 = 2
//! - codec_wiki  (4.26 MP) × {2.0}                × e8 = 1
//! - imac_g3     (5.62 MP) × {2.0}                × e8 = 1
//!
//! ## Acceptance gates (per W44-phase3-B5 task spec)
//!
//! - (a) ZERO cells regress wall > 3% vs CPU baseline
//! - (b) Median wall speedup ≥ 1.05× across all cells
//! - (c) bytes_delta_pct within ±0.5% on every cell
//! - (d) ssim2 within ±0.5 absolute on every cell
//!
//! If ALL gates pass → ship default-flip (separate commit in jxl-encoder/src/api.rs).
//! If ANY gate fails → HONEST-STOP with per-cell offending data.
//!
//! ## Requires
//!
//! - `gpu-butteraugli` cargo feature (pulls in butteraugli-gpu + cubecl-cuda
//!   with `internals` enabled — see jxl-encoder/Cargo.toml)
//! - CUDA 13.2 at `/usr/local/cuda`
//! - `butteraugli-loop` + `ssim2-loop` + `parallel` features
//!
//! Single-thread (`with_threads(1)`) for like-for-like timing comparison.
//!
//! Usage:
//!   CUDA_PATH=/usr/local/cuda cargo run --release \
//!     --features 'gpu-butteraugli butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_phase3_b5_gpu_wider_sweep -- \
//!     --output benchmarks/w44_phase3_b5_gpu_wider_sweep_2026-05-23.tsv

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// W44-phase3-B5: 3 iters per cell × 2 modes × 38 cells = 228 encode runs total.
// At 4 iters wall would be ~33% higher with marginal variance reduction.
// Best-of-N timing reported.
const TIME_ITERS: u32 = 3;
const DEFAULT_OUTPUT: &str = "benchmarks/w44_phase3_b5_gpu_wider_sweep_2026-05-23.tsv";

fn default_corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

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

fn score(
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
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
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
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn encode_ours(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    gpu: bool,
) -> Option<(Vec<u8>, f64)> {
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_ms = f64::INFINITY;
    for _ in 0..TIME_ITERS {
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_gpu_butteraugli(gpu)
            .with_threads(1);
        let start = Instant::now();
        let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
            best_bytes = Some(bytes);
        }
    }
    best_bytes.map(|b| (b, best_ms))
}

struct Cell {
    name: &'static str,
    relpath: &'static str,
    effort: u8,
    distance: f32,
    role: &'static str,
}

const CELLS: &[Cell] = &[
    // ── 5 CID22 photos × {0.5, 1.0, 2.0, 5.0} × e8 (20 cells) ─────────────
    Cell {
        name: "cid22_1418519_e8_d0_5",
        relpath: "CID22/CID22-512/validation/1418519.png",
        effort: 8,
        distance: 0.5,
        role: "PHOTO_SMOOTH",
    },
    Cell {
        name: "cid22_1418519_e8_d1_0",
        relpath: "CID22/CID22-512/validation/1418519.png",
        effort: 8,
        distance: 1.0,
        role: "PHOTO_SMOOTH",
    },
    Cell {
        name: "cid22_1418519_e8_d2_0",
        relpath: "CID22/CID22-512/validation/1418519.png",
        effort: 8,
        distance: 2.0,
        role: "PHOTO_SMOOTH",
    },
    Cell {
        name: "cid22_1418519_e8_d5_0",
        relpath: "CID22/CID22-512/validation/1418519.png",
        effort: 8,
        distance: 5.0,
        role: "PHOTO_SMOOTH",
    },
    Cell {
        name: "cid22_1025469_e8_d0_5",
        relpath: "CID22/CID22-512/validation/1025469.png",
        effort: 8,
        distance: 0.5,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1025469_e8_d1_0",
        relpath: "CID22/CID22-512/validation/1025469.png",
        effort: 8,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1025469_e8_d2_0",
        relpath: "CID22/CID22-512/validation/1025469.png",
        effort: 8,
        distance: 2.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1025469_e8_d5_0",
        relpath: "CID22/CID22-512/validation/1025469.png",
        effort: 8,
        distance: 5.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1531677_e8_d0_5",
        relpath: "CID22/CID22-512/validation/1531677.png",
        effort: 8,
        distance: 0.5,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1531677_e8_d1_0",
        relpath: "CID22/CID22-512/validation/1531677.png",
        effort: 8,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1531677_e8_d2_0",
        relpath: "CID22/CID22-512/validation/1531677.png",
        effort: 8,
        distance: 2.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1531677_e8_d5_0",
        relpath: "CID22/CID22-512/validation/1531677.png",
        effort: 8,
        distance: 5.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1420710_e8_d0_5",
        relpath: "CID22/CID22-512/validation/1420710.png",
        effort: 8,
        distance: 0.5,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1420710_e8_d1_0",
        relpath: "CID22/CID22-512/validation/1420710.png",
        effort: 8,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1420710_e8_d2_0",
        relpath: "CID22/CID22-512/validation/1420710.png",
        effort: 8,
        distance: 2.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1420710_e8_d5_0",
        relpath: "CID22/CID22-512/validation/1420710.png",
        effort: 8,
        distance: 5.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_3637739_e8_d0_5",
        relpath: "CID22/CID22-512/validation/3637739.png",
        effort: 8,
        distance: 0.5,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_3637739_e8_d1_0",
        relpath: "CID22/CID22-512/validation/3637739.png",
        effort: 8,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_3637739_e8_d2_0",
        relpath: "CID22/CID22-512/validation/3637739.png",
        effort: 8,
        distance: 2.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_3637739_e8_d5_0",
        relpath: "CID22/CID22-512/validation/3637739.png",
        effort: 8,
        distance: 5.0,
        role: "PHOTO",
    },
    // ── 5 CID22 photos × {1.0, 2.0} × e9 (10 cells, high-effort coverage) ─
    Cell {
        name: "cid22_1418519_e9_d1_0",
        relpath: "CID22/CID22-512/validation/1418519.png",
        effort: 9,
        distance: 1.0,
        role: "PHOTO_SMOOTH",
    },
    Cell {
        name: "cid22_1418519_e9_d2_0",
        relpath: "CID22/CID22-512/validation/1418519.png",
        effort: 9,
        distance: 2.0,
        role: "PHOTO_SMOOTH",
    },
    Cell {
        name: "cid22_1025469_e9_d1_0",
        relpath: "CID22/CID22-512/validation/1025469.png",
        effort: 9,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1025469_e9_d2_0",
        relpath: "CID22/CID22-512/validation/1025469.png",
        effort: 9,
        distance: 2.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1531677_e9_d1_0",
        relpath: "CID22/CID22-512/validation/1531677.png",
        effort: 9,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1531677_e9_d2_0",
        relpath: "CID22/CID22-512/validation/1531677.png",
        effort: 9,
        distance: 2.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1420710_e9_d1_0",
        relpath: "CID22/CID22-512/validation/1420710.png",
        effort: 9,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_1420710_e9_d2_0",
        relpath: "CID22/CID22-512/validation/1420710.png",
        effort: 9,
        distance: 2.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_3637739_e9_d1_0",
        relpath: "CID22/CID22-512/validation/3637739.png",
        effort: 9,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "cid22_3637739_e9_d2_0",
        relpath: "CID22/CID22-512/validation/3637739.png",
        effort: 9,
        distance: 2.0,
        role: "PHOTO",
    },
    // ── 2 CLIC 1MP photos × {1.0, 2.0} × e8 (4 cells) ─────────────────────
    Cell {
        name: "clic_097cb426_e8_d1_0",
        relpath: "clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png",
        effort: 8,
        distance: 1.0,
        role: "PHOTO_SMOOTH",
    },
    Cell {
        name: "clic_097cb426_e8_d2_0",
        relpath: "clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png",
        effort: 8,
        distance: 2.0,
        role: "PHOTO_SMOOTH",
    },
    Cell {
        name: "clic_0369d229_e8_d1_0",
        relpath: "clic2025-1024/0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png",
        effort: 8,
        distance: 1.0,
        role: "PHOTO",
    },
    Cell {
        name: "clic_0369d229_e8_d2_0",
        relpath: "clic2025-1024/0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png",
        effort: 8,
        distance: 2.0,
        role: "PHOTO",
    },
    // ── Screen-class cells (4 cells) ──────────────────────────────────────
    Cell {
        name: "terminal_e8_d1_0",
        relpath: "gb82-sc/terminal.png",
        effort: 8,
        distance: 1.0,
        role: "SCREENSHOT",
    },
    Cell {
        name: "terminal_e8_d4_0",
        relpath: "gb82-sc/terminal.png",
        effort: 8,
        distance: 4.0,
        role: "SCREENSHOT",
    },
    Cell {
        name: "codec_wiki_e8_d2_0",
        relpath: "gb82-sc/codec_wiki.png",
        effort: 8,
        distance: 2.0,
        role: "SCREENSHOT",
    },
    Cell {
        name: "imac_g3_e8_d2_0",
        relpath: "gb82-sc/imac_g3.png",
        effort: 8,
        distance: 2.0,
        role: "SCREENSHOT",
    },
];

fn main() -> std::io::Result<()> {
    let mut output_path = PathBuf::from(DEFAULT_OUTPUT);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output" if i + 1 < args.len() => {
                output_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    let corpus_dir = default_corpus_dir();

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut out = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_path)?;

    writeln!(
        out,
        "name\trole\twidth\theight\tpixels_mp\teffort\tdistance\t\
         cpu_bytes\tgpu_bytes\tbytes_delta_pct\t\
         cpu_wall_ms\tgpu_wall_ms\tspeedup\t\
         cpu_bfly\tgpu_bfly\tbfly_delta_pct\t\
         cpu_ssim2\tgpu_ssim2\tssim2_delta\t\
         cpu_decode_ok\tgpu_decode_ok"
    )?;
    out.flush()?;

    let total = CELLS.len();
    eprintln!("W44-phase3-B5 wider sweep: {total} cells × 2 modes × {TIME_ITERS} iters\n");

    for (idx, cell) in CELLS.iter().enumerate() {
        let path = corpus_dir.join(cell.relpath);
        let (rgb, w, h) = match load_png(&path) {
            Some(t) => t,
            None => {
                eprintln!("SKIP {} (missing: {})", cell.name, path.display());
                continue;
            }
        };
        let pixels_mp = (w as f64 * h as f64) / 1.0e6;

        // Build orig_linear / orig_srgb for scoring.
        let orig_pixels_linear: Vec<RGB<f32>> = rgb
            .chunks(3)
            .map(|c| {
                RGB::new(
                    srgb_to_linear_f32(c[0]),
                    srgb_to_linear_f32(c[1]),
                    srgb_to_linear_f32(c[2]),
                )
            })
            .collect();
        let orig_linear: Img<Vec<RGB<f32>>> = Img::new(orig_pixels_linear, w as usize, h as usize);
        let orig_srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let orig_srgb: Img<Vec<[u8; 3]>> = Img::new(orig_srgb_pixels, w as usize, h as usize);

        eprintln!(
            "[{}/{}] {} ({}×{} = {:.2} MP) e{} d={} role={}",
            idx + 1,
            total,
            cell.name,
            w,
            h,
            pixels_mp,
            cell.effort,
            cell.distance,
            cell.role
        );
        let (cpu_bytes, cpu_ms) = match encode_ours(&rgb, w, h, cell.distance, cell.effort, false) {
            Some(t) => t,
            None => {
                eprintln!("  CPU encode FAILED");
                continue;
            }
        };
        let cpu_score = score(&cpu_bytes, &orig_linear, &orig_srgb, w, h);
        let (cpu_bfly, cpu_ssim2) = cpu_score.unwrap_or((f64::NAN, f64::NAN));
        let cpu_decode_ok = cpu_score.is_some();
        eprintln!(
            "  CPU: {} bytes, {:.1} ms, bfly={:.3}, ssim2={:.3}",
            cpu_bytes.len(),
            cpu_ms,
            cpu_bfly,
            cpu_ssim2,
        );

        let (gpu_bytes, gpu_ms) = match encode_ours(&rgb, w, h, cell.distance, cell.effort, true) {
            Some(t) => t,
            None => {
                eprintln!("  GPU encode FAILED");
                continue;
            }
        };
        let gpu_score = score(&gpu_bytes, &orig_linear, &orig_srgb, w, h);
        let (gpu_bfly, gpu_ssim2) = gpu_score.unwrap_or((f64::NAN, f64::NAN));
        let gpu_decode_ok = gpu_score.is_some();
        eprintln!(
            "  GPU: {} bytes, {:.1} ms, bfly={:.3}, ssim2={:.3}",
            gpu_bytes.len(),
            gpu_ms,
            gpu_bfly,
            gpu_ssim2,
        );

        let bytes_delta_pct =
            (gpu_bytes.len() as f64 - cpu_bytes.len() as f64) / cpu_bytes.len() as f64 * 100.0;
        let speedup = cpu_ms / gpu_ms;
        let bfly_delta_pct = if cpu_bfly > 0.0 {
            (gpu_bfly - cpu_bfly) / cpu_bfly * 100.0
        } else {
            0.0
        };
        let ssim2_delta = gpu_ssim2 - cpu_ssim2;

        eprintln!(
            "  Δbytes={:+.3}% speedup={:.3}× Δbfly={:+.3}% Δssim2={:+.4}",
            bytes_delta_pct, speedup, bfly_delta_pct, ssim2_delta,
        );

        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t\
             {}\t{}\t{:.3}\t\
             {:.1}\t{:.1}\t{:.3}\t\
             {:.4}\t{:.4}\t{:.3}\t\
             {:.4}\t{:.4}\t{:.4}\t\
             {}\t{}",
            cell.name,
            cell.role,
            w,
            h,
            pixels_mp,
            cell.effort,
            cell.distance,
            cpu_bytes.len(),
            gpu_bytes.len(),
            bytes_delta_pct,
            cpu_ms,
            gpu_ms,
            speedup,
            cpu_bfly,
            gpu_bfly,
            bfly_delta_pct,
            cpu_ssim2,
            gpu_ssim2,
            ssim2_delta,
            cpu_decode_ok,
            gpu_decode_ok,
        )?;
        out.flush()?;
    }

    eprintln!("\nResults written to {}", output_path.display());
    Ok(())
}
