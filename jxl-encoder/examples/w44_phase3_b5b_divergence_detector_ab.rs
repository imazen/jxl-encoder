//! W44-PHASE3-B5b — divergence-detector validation bench on the W44-PHASE3-B5
//! 38-cell wider sweep.
//!
//! Three encode modes per cell, paired best-of-N timing:
//!
//!   - `cpu`   — `LossyConfig::with_gpu_butteraugli(false)` (baseline)
//!   - `gpu`   — `LossyConfig::with_gpu_butteraugli(true)`, detector OFF
//!   - `gpu_d` — `LossyConfig::with_gpu_butteraugli(true)`, detector ON
//!     (env `JXL_W44_PHASE3_B5B_DETECTOR=1` set per-cell; counters reset
//!     between cells via [`perceptual_backend::b5b_counters::reset`])
//!
//! Acceptance gates (per W44-PHASE3-B5b task spec):
//!
//!   - (a) 36/38 B5-non-divergent cells: detector does NOT fire (no fallback,
//!         no bytes shift vs `gpu` mode)
//!   - (b) 2/38 B5-divergent cells: detector FIRES (fallback → CPU bytes)
//!   - (c) bytes within ±0.5% on all 38 vs `cpu` baseline for `gpu_d` mode
//!   - (d) wall overhead `gpu_d` vs `gpu`: ≤ +1.5% median
//!
//! Same 38 cells as `w44_phase3_b5_gpu_wider_sweep`. Re-uses the same
//! corpus-paths so re-running both produces directly-comparable TSVs.
//!
//! Output columns:
//!
//!   name role width height pixels_mp effort distance
//!   cpu_bytes gpu_bytes gpu_d_bytes
//!   gpu_vs_cpu_bytes_pct gpu_d_vs_cpu_bytes_pct gpu_d_vs_gpu_bytes_pct
//!   cpu_wall_ms gpu_wall_ms gpu_d_wall_ms
//!   gpu_speedup gpu_d_speedup gpu_d_overhead_pct
//!   detector_ran detector_tripped detector_divergence_pct
//!   cpu_bfly gpu_bfly gpu_d_bfly
//!   cpu_ssim2 gpu_ssim2 gpu_d_ssim2
//!   gpu_d_decode_ok
//!
//! Required features:
//!   `gpu-butteraugli butteraugli-loop ssim2-loop parallel __expert`
//!
//! Usage:
//!   CUDA_PATH=/usr/local/cuda cargo run --release \
//!     --features 'gpu-butteraugli butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_phase3_b5b_divergence_detector_ab -- \
//!     --output benchmarks/w44_phase3_b5b_divergence_detector_2026-05-24.tsv

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
#[cfg(feature = "gpu-butteraugli")]
use jxl_encoder::vardct::__b5b_counters as b5b_counters;
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const TIME_ITERS: u32 = 3;
const DEFAULT_OUTPUT: &str = "benchmarks/w44_phase3_b5b_divergence_detector_2026-05-24.tsv";

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

#[derive(Copy, Clone, Debug)]
enum Mode {
    Cpu,
    Gpu,
    GpuDetector,
}

fn encode_ours(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    mode: Mode,
) -> Option<(Vec<u8>, f64)> {
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_ms = f64::INFINITY;
    for _ in 0..TIME_ITERS {
        // Set / clear the detector env var per-mode so it only fires
        // for GpuDetector. The encoder reads this in `construct_backend`.
        match mode {
            Mode::GpuDetector => unsafe { std::env::set_var("JXL_W44_PHASE3_B5B_DETECTOR", "1") },
            _ => unsafe { std::env::remove_var("JXL_W44_PHASE3_B5B_DETECTOR") },
        }
        let gpu = matches!(mode, Mode::Gpu | Mode::GpuDetector);
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_perceptual_device(if gpu {
                jxl_encoder::api::PerceptualDevice::Gpu
            } else {
                jxl_encoder::api::PerceptualDevice::Cpu
            })
            .with_threads(1);
        let start = Instant::now();
        let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
            best_bytes = Some(bytes);
        }
    }
    // Detector counters are reset BEFORE the cell-level run (not per
    // TIME_ITERS) so the final snapshot reflects the cumulative state
    // of the last TIME_ITERS encodes. Mode::GpuDetector reset+snapshot
    // is done in `main` around the (single) encode.
    best_bytes.map(|b| (b, best_ms))
}

struct Cell {
    name: &'static str,
    relpath: &'static str,
    effort: u8,
    distance: f32,
    role: &'static str,
}

// Same 38 cells as W44-PHASE3-B5 wider sweep.
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
    // *** B5 divergent cell 1 ***
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
    // ── 5 CID22 photos × {1.0, 2.0} × e9 (10 cells) ───────────────────────
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
    // *** B5 divergent cell 2 ***
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
         cpu_bytes\tgpu_bytes\tgpu_d_bytes\t\
         gpu_vs_cpu_bytes_pct\tgpu_d_vs_cpu_bytes_pct\tgpu_d_vs_gpu_bytes_pct\t\
         cpu_wall_ms\tgpu_wall_ms\tgpu_d_wall_ms\t\
         gpu_speedup\tgpu_d_speedup\tgpu_d_overhead_pct\t\
         detector_ran\tdetector_tripped\tdetector_divergence_pct\t\
         cpu_bfly\tgpu_bfly\tgpu_d_bfly\t\
         cpu_ssim2\tgpu_ssim2\tgpu_d_ssim2\t\
         gpu_d_decode_ok"
    )?;
    out.flush()?;

    let total = CELLS.len();
    eprintln!(
        "W44-PHASE3-B5b divergence-detector bench: {total} cells × 3 modes × {TIME_ITERS} iters\n"
    );

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

        // Mode A: CPU baseline
        let (cpu_bytes, cpu_ms) =
            match encode_ours(&rgb, w, h, cell.distance, cell.effort, Mode::Cpu) {
                Some(t) => t,
                None => {
                    eprintln!("  CPU encode FAILED");
                    continue;
                }
            };
        let cpu_score = score(&cpu_bytes, &orig_linear, &orig_srgb, w, h);
        let (cpu_bfly, cpu_ssim2) = cpu_score.unwrap_or((f64::NAN, f64::NAN));
        eprintln!(
            "  CPU:   {} bytes, {:.1} ms, bfly={:.3}, ssim2={:.3}",
            cpu_bytes.len(),
            cpu_ms,
            cpu_bfly,
            cpu_ssim2,
        );

        // Mode B: GPU without detector
        let (gpu_bytes, gpu_ms) =
            match encode_ours(&rgb, w, h, cell.distance, cell.effort, Mode::Gpu) {
                Some(t) => t,
                None => {
                    eprintln!("  GPU encode FAILED");
                    continue;
                }
            };
        let gpu_score = score(&gpu_bytes, &orig_linear, &orig_srgb, w, h);
        let (gpu_bfly, gpu_ssim2) = gpu_score.unwrap_or((f64::NAN, f64::NAN));
        eprintln!(
            "  GPU:   {} bytes, {:.1} ms, bfly={:.3}, ssim2={:.3}",
            gpu_bytes.len(),
            gpu_ms,
            gpu_bfly,
            gpu_ssim2,
        );

        // Mode C: GPU with detector. Reset counters BEFORE the per-cell
        // run (3 iters share state) so the post-run snapshot reflects
        // this cell only.
        #[cfg(feature = "gpu-butteraugli")]
        b5b_counters::reset();
        let (gpu_d_bytes, gpu_d_ms) =
            match encode_ours(&rgb, w, h, cell.distance, cell.effort, Mode::GpuDetector) {
                Some(t) => t,
                None => {
                    eprintln!("  GPU_DETECTOR encode FAILED");
                    continue;
                }
            };
        let gpu_d_score = score(&gpu_d_bytes, &orig_linear, &orig_srgb, w, h);
        let (gpu_d_bfly, gpu_d_ssim2) = gpu_d_score.unwrap_or((f64::NAN, f64::NAN));
        let gpu_d_decode_ok = gpu_d_score.is_some();

        #[cfg(feature = "gpu-butteraugli")]
        let det_snap = b5b_counters::snapshot();
        #[cfg(not(feature = "gpu-butteraugli"))]
        let det_snap = b5b_counters::Snapshot {
            run_count: 0,
            fallback_count: 0,
            divergence_pct_sum: 0.0,
            divergence_pct_max: 0.0,
        };
        // run_count > 0 means the detector observed at least one
        // encode-iter's iter-0 (in the best-of-3 run); fallback_count
        // > 0 means at least one iter tripped.
        let detector_ran = det_snap.run_count > 0;
        let detector_tripped = det_snap.fallback_count > 0;
        // The PER-CELL divergence we report = the MAX divergence over
        // the TIME_ITERS encodes for this cell (these are deterministic
        // per input so the max == mean == any iter's value modulo
        // measurement noise from the GPU reduction-tree order).
        let detector_divergence_pct = det_snap.divergence_pct_max * 100.0;

        eprintln!(
            "  GPU_D: {} bytes, {:.1} ms, bfly={:.3}, ssim2={:.3} \
             (detector_run={} tripped={} div={:.4}%)",
            gpu_d_bytes.len(),
            gpu_d_ms,
            gpu_d_bfly,
            gpu_d_ssim2,
            detector_ran,
            detector_tripped,
            detector_divergence_pct,
        );

        let gpu_vs_cpu_bytes_pct =
            (gpu_bytes.len() as f64 - cpu_bytes.len() as f64) / cpu_bytes.len() as f64 * 100.0;
        let gpu_d_vs_cpu_bytes_pct =
            (gpu_d_bytes.len() as f64 - cpu_bytes.len() as f64) / cpu_bytes.len() as f64 * 100.0;
        let gpu_d_vs_gpu_bytes_pct =
            (gpu_d_bytes.len() as f64 - gpu_bytes.len() as f64) / gpu_bytes.len() as f64 * 100.0;
        let gpu_speedup = cpu_ms / gpu_ms;
        let gpu_d_speedup = cpu_ms / gpu_d_ms;
        let gpu_d_overhead_pct = (gpu_d_ms - gpu_ms) / gpu_ms * 100.0;

        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t\
             {}\t{}\t{}\t\
             {:.3}\t{:.3}\t{:.3}\t\
             {:.1}\t{:.1}\t{:.1}\t\
             {:.3}\t{:.3}\t{:.3}\t\
             {}\t{}\t{:.4}\t\
             {:.4}\t{:.4}\t{:.4}\t\
             {:.4}\t{:.4}\t{:.4}\t\
             {}",
            cell.name,
            cell.role,
            w,
            h,
            pixels_mp,
            cell.effort,
            cell.distance,
            cpu_bytes.len(),
            gpu_bytes.len(),
            gpu_d_bytes.len(),
            gpu_vs_cpu_bytes_pct,
            gpu_d_vs_cpu_bytes_pct,
            gpu_d_vs_gpu_bytes_pct,
            cpu_ms,
            gpu_ms,
            gpu_d_ms,
            gpu_speedup,
            gpu_d_speedup,
            gpu_d_overhead_pct,
            detector_ran,
            detector_tripped,
            detector_divergence_pct,
            cpu_bfly,
            gpu_bfly,
            gpu_d_bfly,
            cpu_ssim2,
            gpu_ssim2,
            gpu_d_ssim2,
            gpu_d_decode_ok,
        )?;
        out.flush()?;
    }

    eprintln!("\nResults written to {}", output_path.display());
    Ok(())
}
