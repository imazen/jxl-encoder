//! Paired A/B bench for the `tree_max_buckets` per-image dispatch
//! (audit item #3 / commit `4572790` Pareto sweep).
//!
//! Per-cell A/B:
//!  A = dispatch ON (default; `LosslessConfig` does
//!      `effective_profile_for_image -> adapt_tree_max_buckets_for_image`,
//!      which drops `tree_max_buckets` 256→192 at large+e9 cells only).
//!  B = dispatch OFF (force `tree_max_buckets=256` via `__expert`
//!      `LosslessInternalParams` — sweep harness's pinned value
//!      survives the dispatch per `effective_profile_for_image`).
//!
//! Interleaved sample-major so paired Δ measurements stay thermally
//! close (CLAUDE.md zenbench discipline).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features 'parallel-tree-learning __expert' \
//!       --example bucket_dispatch_paired_ab \
//!       > benchmarks/bucket_dispatch_paired_ab_2026-05-17.tsv
//!
//! Environment:
//!   SAMPLES=10   (default: 7; per-cell sample count, paired)
//!   THREADS=8    (default: 8)
//!   EFFORTS=7,8,9   (default: 7,8,9)
//!
//! Acceptance gates (per task brief):
//!   - large+e9: ≥5% wall-clock improvement
//!   - large+e9: bytes Δ within +0.5% of pre-dispatch baseline
//!   - all other 8 cells: bytes-identical AND wall-clock within noise
//!     (the dispatch must not fire outside its gate)

use jxl_encoder::LosslessInternalParams;
use jxl_encoder::api::{LosslessConfig, PixelLayout};
use sha2::Digest;
use std::path::PathBuf;
use std::time::Instant;

const IMAGES: &[(&str, &str)] = &[
    (
        "small_0.26MP",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png",
    ),
    (
        "medium_1.05MP",
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    ),
    (
        "large_4.19MP",
        "/home/lilith/work/codec-corpus/clic2025/final-test/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png",
    ),
];

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_efforts() -> Vec<u8> {
    std::env::var("EFFORTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u8>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<u8>| !v.is_empty())
        .unwrap_or_else(|| vec![7, 8, 9])
}

fn load_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

/// Variant A: dispatch ON (the new default). Uses `LosslessConfig`
/// straight — `effective_profile_for_image` does the per-image
/// `adapt_tree_max_buckets_for_image` flip.
fn encode_dispatch_on(rgb: &[u8], w: u32, h: u32, effort: u8, threads: usize) -> (usize, f64, u64) {
    let cfg = LosslessConfig::new()
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

/// Variant B: dispatch OFF (pre-dispatch baseline). Forces
/// `tree_max_buckets=256` via the `__expert` internal-params override,
/// which `effective_profile_for_image` honours by skipping
/// `adapt_tree_max_buckets_for_image`.
fn encode_dispatch_off(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    threads: usize,
) -> (usize, f64, u64) {
    // Match `tree_max_buckets_for` exactly so OFF reproduces the pre-
    // dispatch effort default for every effort, not just e9.
    let baseline = match effort {
        0..=4 => 32,
        5 => 48,
        6 => 64,
        7 => 96,
        8 => 128,
        _ => 256,
    };
    let mut params = LosslessInternalParams::default();
    params.tree_max_buckets = Some(baseline);
    let cfg = LosslessConfig::new()
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
        .unwrap_or_else(|_| "/home/lilith/work/zen/jxl-encoder--bucket-dispatch".to_string());
    let path = format!("{}/.workongoing", repo_root);
    let _ = std::fs::write(
        path,
        format!("{} claude-bucket-dispatch {}\n", iso_now(), activity),
    );
}

fn main() {
    let samples = parse_usize("SAMPLES", 7);
    let threads = parse_usize("THREADS", 8);
    let efforts = parse_efforts();

    eprintln!(
        "# tree_max_buckets dispatch paired A/B (audit #3)\n# efforts: {:?}\n# images: {:?}\n# samples per cell (paired): {}\n# threads: {}",
        efforts,
        IMAGES.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
        samples,
        threads
    );

    println!("# tree_max_buckets dispatch paired A/B (audit #3)");
    println!("# A = dispatch ON  (new default; large+e9 → 192, else effort default)");
    println!("# B = dispatch OFF (force effort-default buckets via __expert override)");
    println!("# samples per cell (paired): {}", samples);
    println!("# threads: {}", threads);
    println!(
        "image\tlabel\twidth\theight\tmegapixels\teffort\tsample\tvariant\tbytes\tencode_ms\tsha256_prefix"
    );

    // Load images once.
    let images: Vec<(String, String, Vec<u8>, u32, u32)> = IMAGES
        .iter()
        .filter_map(|(label, path)| {
            let (rgb, w, h) = load_rgb(path)?;
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

    // Warm up once per (image, effort, variant) at the smallest A/B to
    // amortize first-run effects (cargo cache, file mmap, rayon thread
    // spin-up).
    for (label, _path, rgb, w, h) in &images {
        for &e in &efforts {
            refresh_marker(&format!("warmup {} e{}", label, e));
            let _ = encode_dispatch_on(rgb, *w, *h, e, threads);
            let _ = encode_dispatch_off(rgb, *w, *h, e, threads);
        }
    }

    // Sample-major interleave: for each sample s in 1..=N, walk every
    // (image, effort) cell, alternating A then B so the paired Δ
    // measurements are thermally close. Per CLAUDE.md zenbench
    // discipline (randomized round-robin > criterion-style blocks).
    for s in 1..=samples {
        for (label, path, rgb, w, h) in &images {
            let mp = (*w as f64) * (*h as f64) / 1_000_000.0;
            let basename = PathBuf::from(path)
                .file_name()
                .map(|x| x.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());

            for &e in &efforts {
                refresh_marker(&format!("sample {}/{}: {} e{} A", s, samples, label, e));
                let (bytes_a, ms_a, sha_a) = encode_dispatch_on(rgb, *w, *h, e, threads);
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\tA\t{}\t{:.2}\t{:016x}",
                    basename, label, *w, *h, mp, e, s, bytes_a, ms_a, sha_a
                );

                refresh_marker(&format!("sample {}/{}: {} e{} B", s, samples, label, e));
                let (bytes_b, ms_b, sha_b) = encode_dispatch_off(rgb, *w, *h, e, threads);
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\tB\t{}\t{:.2}\t{:016x}",
                    basename, label, *w, *h, mp, e, s, bytes_b, ms_b, sha_b
                );
            }
        }
    }

    refresh_marker("paired A/B complete");
}

// ── Local ISO timestamp helper (mirrors lossless_e9_buckets_sweep.rs) ──
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
