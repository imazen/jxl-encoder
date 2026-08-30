//! Phase 2 of issue #41: paired A/B harness comparing the
//! gather-time cuckoo dedup (`gather_dedup = true`) against the
//! production post-pass sort dedup (`gather_dedup = false`).
//!
//! Real CLIC photos only — synthetic content masks tree-learning
//! hot-path behaviour (per CLAUDE.md "no synthetic-only quality
//! tests"). The harness interleaves A and B samples per round so
//! thermal / turbo / OS scheduler bias averages out (matches the
//! zenbench paired-statistics discipline).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features __expert \
//!       --example gather_dedup_ab -- [image1.png ...]
//!
//! With no args, defaults to the 3 profile images
//! (0.26 MP / 1.05 MP / 4.19 MP) used by `lossless_cliff_profile`.
//!
//! Environment knobs:
//!   GD_EFFORTS="7,8,9"   # comma-separated effort levels
//!   GD_SAMPLES=5         # samples per cell (per knob)
//!   GD_THREADS=1         # 1 = single-threaded; 0 = ambient rayon pool
//!   GD_WARMUP=2          # warm-up samples discarded before measurement

#[cfg(feature = "profile-phases")]
use jxl_encoder::__test_exports::profile_phases;
use jxl_encoder::LosslessInternalParams;
use jxl_encoder::api::{LosslessConfig, PixelLayout};
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
        .unwrap_or_else(|| vec![7])
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

fn encode_once(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    threads: usize,
    gather_dedup: bool,
) -> (usize, f64) {
    // Build a config with the same defaults as encoder().with_effort(e)
    // except for the gather_dedup flag we want to toggle.
    // LosslessInternalParams is #[non_exhaustive], so build via Default
    // and mutate the one field we care about.
    let mut params = LosslessInternalParams::default();
    params.gather_dedup = Some(gather_dedup);
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
/// Mirrors zenbench's "paired" reporting: mean of diffs + per-sample
/// percentiles are cleaner than independent A/B means because the same
/// thermal/scheduling noise affects both arms.
struct PairedStats {
    n: usize,
    a_med_ms: f64,
    b_med_ms: f64,
    delta_med_ms: f64,
    delta_pct_med: f64,
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
    PairedStats {
        n,
        a_med_ms: percentile(&a_sorted, 0.5),
        b_med_ms: percentile(&b_sorted, 0.5),
        delta_med_ms: percentile(&deltas, 0.5),
        delta_pct_med: percentile(&deltas_pct, 0.5),
        a_best_ms: a_sorted[0],
        b_best_ms: b_sorted[0],
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
    // Crude ISO 8601 UTC without time zone library deps.
    let days = secs / 86400;
    let s_of_day = secs % 86400;
    let (h, m, s) = (s_of_day / 3600, (s_of_day % 3600) / 60, s_of_day % 60);
    // Treat 2026 as a normal year, accept rough date (the timestamp is
    // a human-readable provenance hint, not a precise field).
    let year = 2026;
    let day_of_year = days - 20454; // 2026-01-01 offset
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
    let samples = parse_u32_env("GD_SAMPLES", 5) as usize;
    let warmup = parse_u32_env("GD_WARMUP", 2) as usize;
    let threads = parse_u32_env("GD_THREADS", 1) as usize;

    println!("# gather_dedup A/B (issue #41 Phase 2)");
    println!("# generated {}", iso_now());
    println!(
        "# samples per cell: {}  warmup: {}  threads: {}",
        samples, warmup, threads
    );
    println!("# efforts: {:?}", efforts);
    println!();
    println!(
        "image\tmegapixels\teffort\tn\ta_med_ms\tb_med_ms\tdelta_med_ms\tdelta_pct_med\ta_best_ms\tb_best_ms\ta_bytes\tb_bytes\tbyte_delta_pct"
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
            #[cfg(feature = "profile-phases")]
            profile_phases::reset();

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
                "{}\t{:.2}\t{}\t{}\t{:.1}\t{:.1}\t{:+.2}\t{:+.2}\t{:.1}\t{:.1}\t{}\t{}\t{:+.4}",
                basename,
                mp,
                effort,
                stats.n,
                stats.a_med_ms,
                stats.b_med_ms,
                stats.delta_med_ms,
                stats.delta_pct_med,
                stats.a_best_ms,
                stats.b_best_ms,
                stats.a_bytes,
                stats.b_bytes,
                byte_delta_pct,
            );

            #[cfg(feature = "profile-phases")]
            {
                let snap = profile_phases::take_snapshot();
                // Both arms contribute to the snapshot; without per-arm
                // separation we just print a few hot phases.
                eprintln!("    [aggregated phases for {} @ e{}]", basename, effort);
                let mut entries: Vec<(&'static str, u128)> = snap.into_iter().collect();
                entries.sort_by_key(|b| core::cmp::Reverse(b.1));
                for (name, ns) in entries.iter().take(12) {
                    eprintln!("      {:42} {:>9.1} ms", name, (*ns as f64) / 1_000_000.0);
                }
            }
        }
    }
}
