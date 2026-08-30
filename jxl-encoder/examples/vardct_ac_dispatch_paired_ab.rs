//! Paired A/B bench for the VarDCT `adapt_to_image_lossy` per-image
//! dispatch (chunk-1 of the VarDCT speed push; pixel + distance gate
//! that drops `try_dct64=false` on small + low-d cells).
//!
//! Per-cell A/B (interleaved sample-major):
//!  A = dispatch ON (default; `LossyConfig` does
//!      `effective_profile_for_image -> adapt_to_image_lossy`, which
//!      drops `try_dct64=true→false` when
//!      `pixels < LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD (500_000)` AND
//!      `distance < LOSSY_LOW_DISTANCE_THRESHOLD (2.0)`).
//!  B = dispatch OFF (force `try_dct64=true` via `__expert`
//!      `LossyInternalParams` — the override-respect logic in
//!      `effective_profile_for_image` skips the adapter so the override
//!      survives).
//!
//! Interleaved sample-major so paired Δ measurements stay thermally
//! close (CLAUDE.md zenbench discipline). Mirrors the lossless
//! `bucket_dispatch_paired_ab` example (commit `488cc68d`).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features 'std parallel butteraugli-loop __expert' \
//!       --example vardct_ac_dispatch_paired_ab \
//!       > benchmarks/vardct_ac_dispatch_paired_2026-05-17.tsv
//!
//! Environment:
//!   SAMPLES=10            (default: 7; per-cell sample count, paired)
//!   THREADS=1             (default: 1; per CLAUDE.md zenbench discipline,
//!                          single-thread amplifies VarDCT AC search cost)
//!   DISTANCES=0.5,1.0,2.0 (default; brief asks d=0.5, d=1.0, d=2.0)
//!   EFFORT=7              (default; only effort 7 — DCT64 lives at e>=7)
//!
//! Acceptance gates (per task brief):
//!   - small + d=0.5: ≥3% wall-clock improvement
//!   - all gated cells: bytes within +0.5% of pre-dispatch baseline
//!   - non-gated cells (medium/large × any d, or any size × d>=2.0):
//!     bytes-identical (sha256 prefix equal sample-wise)

use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use sha2::Digest;
use std::path::PathBuf;
use std::time::Instant;

/// Profile images (label, full-resolution source path).
/// The 256×256 cell is generated via center-crop on the same CID22
/// source as the small cell (no separate corpus dependency).
const IMAGES: &[(&str, &str, Option<u32>)] = &[
    (
        "tiny_0.07MP",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png",
        Some(256),
    ),
    (
        "small_0.26MP",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png",
        None,
    ),
    (
        "medium_1.05MP",
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
        None,
    ),
    (
        "large_2.78MP",
        "/home/lilith/work/codec-corpus/clic2025/final-test/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
        None,
    ),
];

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_distances() -> Vec<f32> {
    std::env::var("DISTANCES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<f32>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<f32>| !v.is_empty())
        .unwrap_or_else(|| vec![0.5, 1.0, 2.0])
}

fn parse_effort() -> u8 {
    std::env::var("EFFORT")
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())
        .unwrap_or(7)
}

/// Load an RGB image, optionally center-cropping to a square.
fn load_rgb(path: &str, crop_to: Option<u32>) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    let (rgb, ow, oh) = match crop_to {
        Some(c) if c <= w && c <= h => {
            let x0 = (w - c) / 2;
            let y0 = (h - c) / 2;
            let mut out = Vec::with_capacity((c * c * 3) as usize);
            for y in y0..(y0 + c) {
                let row_start = ((y * w + x0) * 3) as usize;
                let row_end = ((y * w + x0 + c) * 3) as usize;
                out.extend_from_slice(&img.as_raw()[row_start..row_end]);
            }
            (out, c, c)
        }
        _ => (img.into_raw(), w, h),
    };
    Some((rgb, ow, oh))
}

/// Variant A: dispatch ON (the new default). Uses `LossyConfig`
/// straight — `effective_profile_for_image` does the per-image
/// `adapt_to_image_lossy` flip when small + low-d.
fn encode_dispatch_on(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    threads: usize,
) -> (usize, f64, u64) {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut h = sha2::Sha256::new();
    h.update(&bytes);
    let digest = h.finalize();
    let prefix = u64::from_be_bytes(digest[..8].try_into().unwrap());
    (bytes.len(), ms, prefix)
}

/// Variant B: dispatch OFF (pre-dispatch baseline). Pins
/// `try_dct64=Some(true)` via the `__expert` internal-params override,
/// which `effective_profile_for_image` honours by skipping the
/// per-image adapter (the `profile_override.is_none()` gate). At cells
/// where the adapter would not have fired anyway (medium/large size or
/// d >= 2.0), variants A and B produce byte-identical output.
fn encode_dispatch_off(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    threads: usize,
) -> (usize, f64, u64) {
    let mut params = LossyInternalParams::default();
    params.try_dct64 = Some(true);
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut h = sha2::Sha256::new();
    h.update(&bytes);
    let digest = h.finalize();
    let prefix = u64::from_be_bytes(digest[..8].try_into().unwrap());
    (bytes.len(), ms, prefix)
}

fn refresh_marker(activity: &str) {
    let repo_root = std::env::var("REPO_ROOT")
        .unwrap_or_else(|_| "/home/lilith/work/zen/jxl-encoder--vardct-ac-dispatch".to_string());
    let path = format!("{}/.workongoing", repo_root);
    let _ = std::fs::write(
        path,
        format!("{} claude-vardct-ac-dispatch {}\n", iso_now(), activity),
    );
}

fn main() {
    let samples = parse_usize("SAMPLES", 7);
    let threads = parse_usize("THREADS", 1);
    let distances = parse_distances();
    let effort = parse_effort();

    eprintln!(
        "# vardct_ac_dispatch paired A/B (chunk 1)\n\
         # distances: {:?}\n\
         # images: {:?}\n\
         # effort: {}\n\
         # samples per cell (paired): {}\n\
         # threads: {}",
        distances,
        IMAGES.iter().map(|(l, _, _)| *l).collect::<Vec<_>>(),
        effort,
        samples,
        threads
    );

    println!("# vardct_ac_dispatch paired A/B (chunk 1)");
    println!("# A = dispatch ON  (new default; small_<500k + d<2.0 → try_dct64=false)");
    println!("# B = dispatch OFF (force try_dct64=true via __expert override)");
    println!("# effort: {}", effort);
    println!("# samples per cell (paired): {}", samples);
    println!("# threads: {}", threads);
    println!(
        "image\tlabel\twidth\theight\tmegapixels\tdistance\teffort\tsample\tvariant\tbytes\tencode_ms\tsha256_prefix"
    );

    // Load images once.
    let images: Vec<(String, String, Vec<u8>, u32, u32)> = IMAGES
        .iter()
        .filter_map(|(label, path, crop)| {
            let (rgb, w, h) = load_rgb(path, *crop)?;
            Some((label.to_string(), path.to_string(), rgb, w, h))
        })
        .collect();
    if images.len() < IMAGES.len() {
        eprintln!(
            "WARNING: only {} of {} images loaded",
            images.len(),
            IMAGES.len()
        );
    }

    // Warm-up at the smallest A/B per (image, distance) so first-run
    // effects amortize (file mmap, rayon thread spin-up, archmage
    // token cache).
    for (label, _path, rgb, w, h) in &images {
        for &d in &distances {
            refresh_marker(&format!("warmup {} d{:.2}", label, d));
            let _ = encode_dispatch_on(rgb, *w, *h, d, effort, threads);
            let _ = encode_dispatch_off(rgb, *w, *h, d, effort, threads);
        }
    }

    // Sample-major interleave: each sample s walks every (image,
    // distance) cell, alternating A then B so paired Δ stays
    // thermally close.
    for s in 1..=samples {
        for (label, path, rgb, w, h) in &images {
            let mp = (*w as f64) * (*h as f64) / 1_000_000.0;
            let basename = PathBuf::from(path)
                .file_name()
                .map(|x| x.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());

            for &d in &distances {
                refresh_marker(&format!("sample {}/{}: {} d{:.2} A", s, samples, label, d));
                let (bytes_a, ms_a, sha_a) = encode_dispatch_on(rgb, *w, *h, d, effort, threads);
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{}\t{}\tA\t{}\t{:.2}\t{:016x}",
                    basename, label, *w, *h, mp, d, effort, s, bytes_a, ms_a, sha_a
                );

                refresh_marker(&format!("sample {}/{}: {} d{:.2} B", s, samples, label, d));
                let (bytes_b, ms_b, sha_b) = encode_dispatch_off(rgb, *w, *h, d, effort, threads);
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{}\t{}\tB\t{}\t{:.2}\t{:016x}",
                    basename, label, *w, *h, mp, d, effort, s, bytes_b, ms_b, sha_b
                );
            }
        }
    }

    refresh_marker("paired A/B complete");
}

// ── Local ISO timestamp helper (mirrors lossless examples) ──
fn iso_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs as i64;
    let days = s / 86400;
    let secs_of_day = s % 86400;
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;
    let (yy, mo, dd) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        yy, mo, dd, hh, mm, ss
    )
}

fn days_to_ymd(mut days: i64) -> (i32, u32, u32) {
    let mut year: i32 = 1970;
    loop {
        let leap = is_leap(year);
        let yd = if leap { 366 } else { 365 };
        if days < yd as i64 {
            break;
        }
        days -= yd as i64;
        year += 1;
    }
    let dim: [u32; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0;
    let mut d = days as u32;
    for (i, &md) in dim.iter().enumerate() {
        if d < md {
            month = i;
            break;
        }
        d -= md;
    }
    (year, month as u32 + 1, d + 1)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
