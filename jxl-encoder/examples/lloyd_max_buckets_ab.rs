//! A/B benchmark for the EX-J5 reinterpretation:
//! **Lloyd-Max iterative clustering for MA-tree bucket boundaries on the
//! residual-energy proxy properties (4 = `|N|`, 5 = `|W|`, 15 = `wp_max_error`)**.
//!
//! For each (image, lloyd_max) cell, encodes N times taking the min wall-clock
//! to wash out load contamination. Reports bytes + encode_ms with the percent
//! delta of the Lloyd-Max variant vs the sort-quantile default.
//!
//! Bytes delta is the headline metric — the EX-J5 paper claims 0.5-1 % on
//! textured photos for the full 1024-context CALIC system. Our spec-legal
//! adaptation refines only the bucket boundaries of the existing 16 JXL
//! properties (we cannot add a 17th — JXL hard-codes `kNumNonrefProperties =
//! 16`), so we expect a fraction of the headline number: 0.2–0.5 % on the
//! same content.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features '__expert parallel-tree-learning' \
//!       --example lloyd_max_buckets_ab -- > bench.tsv
//!
//! Environment:
//!   SAMPLES=3   (default 3; bench uses min wall-clock across N samples)
//!   THREADS=8   (default 8)
//!   EFFORT=7    (default 7)

use jxl_encoder::api::{LosslessConfig, PixelLayout};
use jxl_encoder::effort::LosslessInternalParams;
use std::time::Instant;

const TEXTURED_PHOTOS: &[(&str, &str)] = &[
    (
        "clic_02809272_1.05MP",
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    ),
    (
        "clic_07b9f93f_1.05MP",
        "/home/lilith/work/codec-corpus/clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
    ),
    (
        "clic_0369d229_1.05MP",
        "/home/lilith/work/codec-corpus/clic2025-1024/0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png",
    ),
    (
        "clic_0d154749_1.05MP",
        "/home/lilith/work/codec-corpus/clic2025-1024/0d154749c7771f58e89ad343653ec4e20d6f037da829f47f5598e5d0a4ab61f0.png",
    ),
    (
        "cid22_1025469_0.26MP",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
    ),
];

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_u8(name: &str, default: u8) -> u8 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())
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
    lloyd_max: bool,
) -> (usize, f64) {
    let mut params = LosslessInternalParams::default();
    params.lloyd_max_buckets = Some(lloyd_max);
    let cfg = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(threads)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes.len(), ms)
}

fn bench_min(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    threads: usize,
    lloyd_max: bool,
    samples: usize,
) -> (usize, f64) {
    // Warmup encode (cold pool / branch predictor).
    let _ = encode_once(rgb, w, h, effort, threads, lloyd_max);

    let mut best_ms = f64::INFINITY;
    let mut bytes_observed = 0usize;
    for _ in 0..samples {
        let (b, ms) = encode_once(rgb, w, h, effort, threads, lloyd_max);
        if ms < best_ms {
            best_ms = ms;
        }
        bytes_observed = b;
    }
    (bytes_observed, best_ms)
}

fn main() {
    let samples = parse_usize("SAMPLES", 3);
    let threads = parse_usize("THREADS", 8);
    let effort = parse_u8("EFFORT", 7);

    eprintln!(
        "# Lloyd-Max bucket boundaries A/B at e{} ({} samples, {} threads)",
        effort, samples, threads
    );
    eprintln!("# baseline = sort-quantile (default); variant = Lloyd-Max on props 4/5/15");
    println!(
        "image\tbytes_a_sort\tms_a_sort\tbytes_b_lloyd\tms_b_lloyd\tbytes_delta_pct\tms_delta_pct"
    );

    let mut total_a = 0i64;
    let mut total_b = 0i64;

    for (label, path) in TEXTURED_PHOTOS {
        let (rgb, w, h) = match load_rgb(path) {
            Some(t) => t,
            None => {
                eprintln!("# SKIP {} (missing {})", label, path);
                continue;
            }
        };

        let (bytes_a, ms_a) = bench_min(&rgb, w, h, effort, threads, false, samples);
        let (bytes_b, ms_b) = bench_min(&rgb, w, h, effort, threads, true, samples);

        let bytes_delta = (bytes_b as f64 - bytes_a as f64) / bytes_a as f64 * 100.0;
        let ms_delta = (ms_b - ms_a) / ms_a * 100.0;

        println!(
            "{}\t{}\t{:.1}\t{}\t{:.1}\t{:+.3}\t{:+.2}",
            label, bytes_a, ms_a, bytes_b, ms_b, bytes_delta, ms_delta
        );

        total_a += bytes_a as i64;
        total_b += bytes_b as i64;
    }

    eprintln!();
    let agg = (total_b as f64 - total_a as f64) / total_a as f64 * 100.0;
    eprintln!(
        "# AGGREGATE: total_a = {} bytes, total_b = {} bytes ({:+.3}%)",
        total_a, total_b, agg
    );
}
