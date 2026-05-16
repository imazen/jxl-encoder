//! Phase-1 profile harness for issue #23 (lossless e3->e7 cliff).
//!
//! Encodes 3 representative images at e3, e5, e7 and prints a wall-clock
//! breakdown along with output bytes. Real photos only — synthetic content
//! hides issue #23-class hotspots (per CLAUDE.md "no synthetic-only quality
//! tests").
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features profile-phases \
//!       --example lossless_cliff_profile -- [image1.png ...]
//!
//! With no args, defaults to small (512x512) + medium (1024x1024) +
//! large (2048x2048) CLIC photos.
//!
//! Environment knobs:
//!   CLIFF_EFFORTS="3,5,7"        # comma-separated effort levels
//!   CLIFF_SAMPLES=3               # samples per cell
//!   CLIFF_THREADS=1               # threads (1 = single, 0 = ambient pool)

use jxl_encoder::api::{LosslessConfig, PixelLayout};
use jxl_encoder::profile_phases;
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
    std::env::var("CLIFF_EFFORTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u8>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<u8>| !v.is_empty())
        .unwrap_or_else(|| vec![3, 5, 7])
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

fn encode_once(rgb: &[u8], w: u32, h: u32, effort: u8, threads: usize) -> (usize, f64) {
    let cfg = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(threads);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes.len(), elapsed_ms)
}

fn fmt_phase_table(phases: &[(&'static str, u128)], total_ns: u128) -> String {
    let mut out = String::new();
    let mut sum: u128 = 0;
    for (k, ns) in phases {
        let pct = if total_ns > 0 {
            100.0 * (*ns as f64) / (total_ns as f64)
        } else {
            0.0
        };
        out.push_str(&format!(
            "    {:42}  {:>9.1} ms ({:>5.1}%)\n",
            k,
            (*ns as f64) / 1_000_000.0,
            pct,
        ));
        sum += ns;
    }
    let other_ns = total_ns.saturating_sub(sum);
    let other_pct = if total_ns > 0 {
        100.0 * (other_ns as f64) / (total_ns as f64)
    } else {
        0.0
    };
    out.push_str(&format!(
        "    {:42}  {:>9.1} ms ({:>5.1}%)\n",
        "[other / unaccounted]",
        (other_ns as f64) / 1_000_000.0,
        other_pct,
    ));
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let images: Vec<&str> = if args.is_empty() {
        DEFAULT_IMAGES.iter().copied().collect()
    } else {
        args.iter().map(String::as_str).collect()
    };
    let efforts = parse_efforts();
    let samples = parse_u32_env("CLIFF_SAMPLES", 3);
    let threads = parse_u32_env("CLIFF_THREADS", 1) as usize;

    println!("# lossless cliff profile (issue #23)");
    println!("# generated {}", iso_now());
    println!("# samples per cell: {}", samples);
    println!("# threads: {}", threads);
    println!("# efforts: {:?}", efforts);
    println!("# profile-phases feature: {}", cfg!(feature = "profile-phases"));
    println!();
    println!("## summary table");
    println!(
        "image\twidth\theight\tmegapixels\teffort\tbytes\tbest_ms\tmedian_ms\tworst_ms\tms_per_mp"
    );

    // Cache breakdown to print after the summary.
    let mut breakdowns: Vec<(String, f64, Vec<(&'static str, u128)>, u128)> = Vec::new();

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
            // Warm-up sample (don't measure phase data).
            let (_b, _ms) = encode_once(&rgb, w, h, effort, threads);
            profile_phases::reset();

            let mut samples_vec: Vec<(usize, f64)> = (0..samples)
                .map(|_| encode_once(&rgb, w, h, effort, threads))
                .collect();
            // Snapshot accumulated phases (sum across all `samples` runs).
            let phase_snapshot = profile_phases::take_snapshot();

            samples_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let bytes = samples_vec[0].0;
            let best = samples_vec[0].1;
            let median = samples_vec[samples_vec.len() / 2].1;
            let worst = samples_vec[samples_vec.len() - 1].1;
            let ms_per_mp = best / mp.max(1e-9);
            println!(
                "{}\t{}\t{}\t{:.2}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.0}",
                basename, w, h, mp, effort, bytes, best, median, worst, ms_per_mp,
            );

            // Per-sample average across the runs.
            let mean_ms: f64 =
                samples_vec.iter().map(|(_, ms)| ms).sum::<f64>() / samples_vec.len() as f64;
            let total_phase_ns: u128 = (mean_ms * 1_000_000.0) as u128 * samples_vec.len() as u128;

            breakdowns.push((
                format!("{} @ e{}", basename, effort),
                mean_ms,
                phase_snapshot,
                total_phase_ns,
            ));

            // refresh marker between long runs
            let _ = std::fs::write(
                ".workongoing",
                format!(
                    "{} claude-session-issue23 cliff-profile-running effort={} img={}\n",
                    iso_now(),
                    effort,
                    basename
                ),
            );
        }
    }

    // Print phase breakdown for each cell. Per-sample numbers (divide
    // accumulated ns by sample count).
    println!();
    println!("## phase breakdown (mean across samples)");
    for (label, mean_ms, phases, total_ns) in &breakdowns {
        println!(
            "\n{}: {:.1} ms wall-clock per encode\n  phases (summed across {} samples, divide by samples to get per-encode):",
            label, mean_ms, samples,
        );
        print!("{}", fmt_phase_table(phases, *total_ns));
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
