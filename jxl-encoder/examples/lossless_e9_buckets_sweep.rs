//! Sweep `tree_max_buckets` at effort 9 to validate the Pareto-optimal
//! default for chunk-4 of the e8/e9 lossless wall-clock cliff plan
//! (see `lossless_e8_e9_cliff_2026-05-16.md`).
//!
//! For each (image, buckets) cell, encodes N times (taking `min` to wash out
//! load contamination), records (bytes, encode_ms). All against a baseline
//! of 256 (the current e9 default per `EffortProfile::tree_max_buckets_for`).
//!
//! Acceptance gate: shipping a non-256 default is OK iff
//! `≤+0.5% bytes` and `≥5% wall-clock` win on at least 2 of 3 profile images.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features 'parallel-tree-learning __expert' \
//!       --example lossless_e9_buckets_sweep -- > sweep.tsv
//!
//! Environment:
//!   SWEEP_BUCKETS="64,96,128,192,224,256"   (defaults to that)
//!   SAMPLES=5                               (default: 5; bench uses min)
//!   THREADS=8                               (default: 8)

use jxl_encoder::api::{LosslessConfig, PixelLayout};
use jxl_encoder::effort::LosslessInternalParams;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_IMAGES: &[(&str, &str)] = &[
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

fn parse_buckets() -> Vec<u16> {
    std::env::var("SWEEP_BUCKETS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u16>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<u16>| !v.is_empty())
        .unwrap_or_else(|| vec![64, 96, 128, 192, 224, 256])
}

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn load_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

fn encode_with_buckets(rgb: &[u8], w: u32, h: u32, buckets: u16, threads: usize) -> (usize, f64) {
    let mut params = LosslessInternalParams::default();
    params.tree_max_buckets = Some(buckets);
    let cfg = LosslessConfig::new()
        .with_effort(9)
        .with_threads(threads)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes.len(), ms)
}

fn refresh_marker(activity: &str) {
    let repo_root = std::env::var("REPO_ROOT")
        .unwrap_or_else(|_| "/home/lilith/work/zen/jxl-encoder--bucket-sweep".to_string());
    let path = format!("{}/.workongoing", repo_root);
    let _ = std::fs::write(
        path,
        format!("{} claude-bucket-sweep {}\n", iso_now(), activity),
    );
}

fn main() {
    let buckets = parse_buckets();
    let samples = parse_usize("SAMPLES", 5);
    let threads = parse_usize("THREADS", 8);

    eprintln!(
        "# tree_max_buckets sweep at e9 (chunk-4 of e8/e9 cliff plan)\n# buckets: {:?}\n# samples per cell: {}\n# threads: {}\n# baseline = 256 (current e9 default)",
        buckets, samples, threads
    );

    println!("# tree_max_buckets sweep at e9 (chunk-4)");
    println!("# buckets: {:?}", buckets);
    println!("# samples per cell: {} (take min)", samples);
    println!("# threads: {}", threads);
    println!("# baseline = 256 (current e9 default)");
    println!("image\tlabel\twidth\theight\tmegapixels\tbuckets\tsample\tbytes\tencode_ms");

    // Build (label, path) once
    let images: Vec<(String, String, Vec<u8>, u32, u32)> = DEFAULT_IMAGES
        .iter()
        .filter_map(|(label, path)| {
            let (rgb, w, h) = load_rgb(path)?;
            Some((label.to_string(), path.to_string(), rgb, w, h))
        })
        .collect();

    // Warm up once per image at largest bucket to amortize first-run effects.
    for (label, _path, rgb, w, h) in &images {
        refresh_marker(&format!("warmup {} buckets=256", label));
        let _ = encode_with_buckets(rgb, *w, *h, 256, threads);
    }

    // Interleave (sample-major, then per-image cycle through bucket values)
    // so paired comparisons are thermally close across bucket values for
    // the SAME image. Per CLAUDE.md: paired A/B, not back-to-back blocks.
    for s in 1..=samples {
        for (label, _path, rgb, w, h) in &images {
            let mp = (*w as f64) * (*h as f64) / 1_000_000.0;
            let basename = PathBuf::from(_path)
                .file_name()
                .map(|x| x.to_string_lossy().to_string())
                .unwrap_or_else(|| _path.clone());

            for &b in &buckets {
                refresh_marker(&format!(
                    "sample {}/{}: {} buckets={}",
                    s, samples, label, b
                ));
                let (bytes, ms) = encode_with_buckets(rgb, *w, *h, b, threads);
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{:.2}",
                    basename, label, *w, *h, mp, b, s, bytes, ms
                );
            }
        }
    }

    refresh_marker("sweep complete");
}

// Tiny ISO timestamp (no chrono dep).
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
