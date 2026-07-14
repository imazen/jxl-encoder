//! Sweep `tree_sample_fraction` at effort 7 to validate the right default
//! for issue #23 (lossless e3->e7 cliff).
//!
//! For each (image, fraction) cell, encodes once, records
//! (bytes, encode_ms, fraction). Use real photos only.
//!
//! Default fractions: 0.10, 0.15, 0.20, 0.25, 0.35, 0.5 (current default).
//! Output goes to stdout as TSV; pipe to a file.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --example lossless_e7_fraction_sweep -- > sweep.tsv
//!
//! Environment:
//!   SWEEP_FRACTIONS="0.10,0.15,0.20,0.25,0.35,0.5"

use jxl_encoder::api::{LosslessConfig, PixelLayout};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_IMAGES: &[&str] = &[
    // Small photo
    "/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png",
    // Medium photos (3 from CLIC 1024)
    "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    "/home/lilith/work/codec-corpus/clic2025-1024/0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png",
    "/home/lilith/work/codec-corpus/clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
    // Large photo
    "/home/lilith/work/codec-corpus/clic2025/final-test/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png",
];

fn parse_fractions() -> Vec<f32> {
    std::env::var("SWEEP_FRACTIONS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<f32>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<f32>| !v.is_empty())
        .unwrap_or_else(|| vec![0.10, 0.15, 0.20, 0.25, 0.35, 0.5])
}

fn load_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

fn encode_with_fraction(rgb: &[u8], w: u32, h: u32, fraction: Option<f32>) -> (usize, f64) {
    let mut cfg = LosslessConfig::new().with_effort(7).with_threads(1);
    if let Some(f) = fraction {
        cfg = cfg.with_tree_learning_sample_fraction(f);
    }
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes.len(), ms)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let images: Vec<&str> = if args.is_empty() {
        DEFAULT_IMAGES.to_vec()
    } else {
        args.iter().map(String::as_str).collect()
    };
    let fractions = parse_fractions();

    println!("# lossless e7 sample-fraction sweep (issue #23)");
    println!("# fractions: {:?}", fractions);
    println!("# baseline = no override (uses effort-derived default of 0.5 at e7)");
    println!(
        "image\twidth\theight\tmegapixels\tfraction\tbytes\tencode_ms\tbytes_vs_baseline_pct\ttime_vs_baseline_pct"
    );

    for img_path in &images {
        let (rgb, w, h) = match load_rgb(img_path) {
            Some(v) => v,
            None => {
                eprintln!("WARN: skip {}", img_path);
                continue;
            }
        };
        let mp = (w as f64) * (h as f64) / 1_000_000.0;
        let basename = PathBuf::from(img_path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| img_path.to_string());

        // Baseline (no override = effort-derived default 0.5)
        let (base_bytes, base_ms) = encode_with_fraction(&rgb, w, h, None);
        println!(
            "{}\t{}\t{}\t{:.2}\tbaseline=0.5\t{}\t{:.1}\t+0.0\t+0.0",
            basename, w, h, mp, base_bytes, base_ms,
        );

        let _ = std::fs::write(
            ".workongoing",
            format!(
                "{} claude-session-issue23 sweep-running img={}\n",
                iso_now(),
                basename
            ),
        );

        for &f in &fractions {
            let (b, ms) = encode_with_fraction(&rgb, w, h, Some(f));
            let bytes_pct = 100.0 * (b as f64 - base_bytes as f64) / (base_bytes as f64);
            let time_pct = 100.0 * (ms - base_ms) / base_ms;
            println!(
                "{}\t{}\t{}\t{:.2}\t{:.2}\t{}\t{:.1}\t{:+.2}\t{:+.1}",
                basename, w, h, mp, f, b, ms, bytes_pct, time_pct,
            );
            let _ = std::fs::write(
                ".workongoing",
                format!(
                    "{} claude-session-issue23 sweep-running f={:.2} img={}\n",
                    iso_now(),
                    f,
                    basename
                ),
            );
        }
    }
}

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
