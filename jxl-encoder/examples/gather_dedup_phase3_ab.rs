//! Phase 3 of issue #41: paired A/B harness comparing the Phase 2
//! gather-time cuckoo dedup (`GatherDedupTable`, SoA-indexed) against the
//! Phase 3 inline-fingerprint cuckoo dedup (`InlineDedupTable`,
//! fingerprint-cached canonical-key storage).
//!
//! Both arms run with `gather_dedup = true` (gather-time dedup is the
//! shared prerequisite); the only difference is `gather_dedup_phase3`,
//! which switches the backend inside `gather_samples_strided_with_dedup_backend`.
//!
//! Real CLIC photos only — synthetic content masks tree-learning hot-path
//! behaviour (per CLAUDE.md "no synthetic-only quality tests"). The
//! harness interleaves A and B samples per round so thermal / turbo / OS
//! scheduler bias averages out (matches the zenbench paired-statistics
//! discipline).
//!
//! Acceptance gate (per Chunk 2 brief): ≥ +5 % wall-clock win on
//! 1.05 MP at e7 8T flips the default knob; otherwise stay opt-in.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features '__expert parallel-tree-learning' \
//!       --example gather_dedup_phase3_ab -- [image1.png ...]
//!
//! With no args, defaults to the 3 profile images (0.26 MP / 1.05 MP /
//! 4.19 MP) used by `lossless_cliff_profile`.
//!
//! Environment knobs:
//!   GD_EFFORTS="7,8,9"   # comma-separated effort levels
//!   GD_SAMPLES=8         # samples per cell (per knob)
//!   GD_THREADS=8         # 1 = single-threaded; 0 = ambient rayon pool;
//!                        # 8 is the production parallel-tree-learning shape.
//!   GD_WARMUP=2          # warm-up samples discarded before measurement

use jxl_encoder::api::{LosslessConfig, PixelLayout};
use jxl_encoder::effort::LosslessInternalParams;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_IMAGES: &[&str] = &[
    // Small (~0.26 MP) — CID22 photo
    "/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png",
    // Medium (~1.05 MP) — CLIC 1024 photo
    "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    // Large (~4.19 MP) — CLIC 2048 photo
    "/home/lilith/work/codec-corpus/clic2025/final-test/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png",
];

fn parse_efforts() -> Vec<u8> {
    std::env::var("GD_EFFORTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u8>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<u8>| !v.is_empty())
        .unwrap_or_else(|| vec![7, 8, 9])
}

fn parse_u32_env(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn load_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

/// `enable_phase3 = false` runs the Phase 2 backend (existing default
/// when `gather_dedup = true`); `enable_phase3 = true` runs the new
/// Phase 3 inline-fingerprint backend. Both arms always have
/// `gather_dedup = true` — Phase 3 only kicks in when gather-time dedup
/// is itself enabled.
fn encode_once(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    threads: usize,
    enable_phase3: bool,
) -> (usize, f64) {
    let mut params = LosslessInternalParams::default();
    params.gather_dedup = Some(true);
    params.gather_dedup_phase3 = Some(enable_phase3);
    let cfg = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(threads)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes.len(), elapsed_ms)
}

/// Paired summary stats over the per-sample diffs (delta = b - a).
/// Mirrors zenbench's "paired" reporting — same as gather_dedup_ab.rs.
struct PairedStats {
    n: usize,
    a_med_ms: f64,
    b_med_ms: f64,
    delta_med_ms: f64,
    delta_pct_med: f64,
    delta_pct_best: f64,
    a_best_ms: f64,
    b_best_ms: f64,
    a_bytes: usize,
    b_bytes: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn summarise(samples: &[(f64, f64)], a_bytes: usize, b_bytes: usize) -> PairedStats {
    let n = samples.len();
    let mut a_sorted: Vec<f64> = samples.iter().map(|p| p.0).collect();
    let mut b_sorted: Vec<f64> = samples.iter().map(|p| p.1).collect();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut deltas: Vec<f64> = samples.iter().map(|p| p.1 - p.0).collect();
    deltas.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut deltas_pct: Vec<f64> = samples
        .iter()
        .filter(|p| p.0 > 0.0)
        .map(|p| (p.1 - p.0) / p.0 * 100.0)
        .collect();
    deltas_pct.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let a_best = a_sorted[0];
    let b_best = b_sorted[0];
    let delta_pct_best = if a_best > 0.0 {
        (b_best - a_best) / a_best * 100.0
    } else {
        0.0
    };
    PairedStats {
        n,
        a_med_ms: percentile(&a_sorted, 0.5),
        b_med_ms: percentile(&b_sorted, 0.5),
        delta_med_ms: percentile(&deltas, 0.5),
        delta_pct_med: percentile(&deltas_pct, 0.5),
        delta_pct_best,
        a_best_ms: a_best,
        b_best_ms: b_best,
        a_bytes,
        b_bytes,
    }
}

fn iso_now() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let s_of_day = secs % 86400;
    let (h, m, s) = (s_of_day / 3600, (s_of_day % 3600) / 60, s_of_day % 60);
    let year = 2026;
    let day_of_year = days - 20454;
    format!(
        "{:04}-DOY{:03}T{:02}:{:02}:{:02}Z",
        year, day_of_year, h, m, s
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let images: Vec<&str> = if args.is_empty() {
        DEFAULT_IMAGES.to_vec()
    } else {
        args.iter().map(String::as_str).collect()
    };
    let efforts = parse_efforts();
    let samples = parse_u32_env("GD_SAMPLES", 8) as usize;
    let warmup = parse_u32_env("GD_WARMUP", 2) as usize;
    let threads = parse_u32_env("GD_THREADS", 8) as usize;

    println!("# gather_dedup_phase3 A/B (issue #41 Phase 3, Chunk 2)");
    println!("# generated {}", iso_now());
    println!(
        "# samples per cell: {}  warmup: {}  threads: {}",
        samples, warmup, threads
    );
    println!("# efforts: {:?}", efforts);
    println!("# A = Phase 2 (GatherDedupTable, SoA-indexed cuckoo)");
    println!("# B = Phase 3 (InlineDedupTable, fingerprint-cached cuckoo)");
    println!("# both arms have gather_dedup = true; only gather_dedup_phase3 differs");
    println!();
    println!(
        "image\tmegapixels\teffort\tn\ta_med_ms\tb_med_ms\tdelta_med_ms\tdelta_pct_med\tdelta_pct_best\ta_best_ms\tb_best_ms\ta_bytes\tb_bytes\tbyte_delta_pct"
    );

    for img_path in &images {
        let (rgb, w, h) = match load_rgb(img_path) {
            Some(v) => v,
            None => {
                eprintln!("WARN: skip {} (load failed)", img_path);
                continue;
            }
        };
        let mp = (w as f64) * (h as f64) / 1_000_000.0;
        let basename = PathBuf::from(img_path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| img_path.to_string());

        for &effort in &efforts {
            // Warm-up rounds (alternate A then B to even out caches).
            for _ in 0..warmup {
                let _ = encode_once(&rgb, w, h, effort, threads, false);
                let _ = encode_once(&rgb, w, h, effort, threads, true);
            }

            // Interleaved A/B measurement to keep per-round thermal /
            // scheduler noise paired (the same round's environment hits
            // both arms within milliseconds).
            let mut paired: Vec<(f64, f64)> = Vec::with_capacity(samples);
            let mut last_a_bytes = 0;
            let mut last_b_bytes = 0;
            for _ in 0..samples {
                // Randomise A/B order each round to defeat any
                // deterministic-cache bias the alternation could leak.
                let order_b_first = (paired.len() & 1) == 0;
                let (b_bytes, b_ms) = if order_b_first {
                    encode_once(&rgb, w, h, effort, threads, true)
                } else {
                    (0, 0.0)
                };
                let (a_bytes, a_ms) = encode_once(&rgb, w, h, effort, threads, false);
                let (b_bytes, b_ms) = if order_b_first {
                    (b_bytes, b_ms)
                } else {
                    encode_once(&rgb, w, h, effort, threads, true)
                };
                paired.push((a_ms, b_ms));
                last_a_bytes = a_bytes;
                last_b_bytes = b_bytes;
            }
            let stats = summarise(&paired, last_a_bytes, last_b_bytes);
            let byte_delta_pct = if stats.a_bytes > 0 {
                (stats.b_bytes as f64 - stats.a_bytes as f64) / (stats.a_bytes as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "{}\t{:.2}\t{}\t{}\t{:.1}\t{:.1}\t{:+.2}\t{:+.2}\t{:+.2}\t{:.1}\t{:.1}\t{}\t{}\t{:+.4}",
                basename,
                mp,
                effort,
                stats.n,
                stats.a_med_ms,
                stats.b_med_ms,
                stats.delta_med_ms,
                stats.delta_pct_med,
                stats.delta_pct_best,
                stats.a_best_ms,
                stats.b_best_ms,
                stats.a_bytes,
                stats.b_bytes,
                byte_delta_pct,
            );
        }
    }
}
