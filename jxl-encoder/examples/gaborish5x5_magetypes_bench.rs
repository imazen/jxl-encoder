//! Microbenchmark for the magetypes-consolidated `gaborish_5x5_channel` kernel.
//!
//! Measures wall-clock per call at 5 image sizes and writes paired TSV +
//! .meta to `/tmp/`. The runner caller then atomically `mv`s the TSV into
//! `benchmarks/` once the run is clean (avoids jj-vs-fd races per
//! `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/jj_concurrent_workspace_failure_mode_2026-05-18.md`).
//!
//! Run:
//!   cargo run -p jxl-encoder --release --example gaborish5x5_magetypes_bench
//!
//! Optional env knobs:
//!   GABOR_SAMPLES=15     # samples per cell (default 11)
//!   GABOR_WARMUP=2       # warm-up samples discarded (default 2)
//!   GABOR_SIZES="256,512,1024,2048,4096"   # comma-sep widths/heights

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn parse_sizes() -> Vec<usize> {
    env::var("GABOR_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![256, 512, 1024, 2048, 4096])
}

fn parse_u32_env(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Synthetic XYB-Y-ish plane: smooth ramp + medium-frequency texture, in
/// roughly the [-0.2, 0.4] range typical of XYB Y values.
fn synthetic_xyb_y(width: usize, height: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let ramp = (x as f32 / width as f32) * 0.4 - 0.2;
            let tex = ((x as f32 * 0.13).sin() + (y as f32 * 0.17).cos()) * 0.05;
            buf[y * width + x] = ramp + tex;
        }
    }
    buf
}

/// libjxl default gaborish weights (Y channel, normalized to sum=1).
fn default_weights() -> (f32, f32, f32, f32, f32, f32) {
    let wc = 1.0_f32;
    let wr = 0.115_416_72;
    let wd = 0.061_359_57;
    let w_big_r = 0.026_375_18;
    let wl = 0.005_125_56;
    let w_big_d = 0.001_660_99;
    let sum = wc + 4.0 * wr + 4.0 * wd + 4.0 * w_big_r + 8.0 * wl + 4.0 * w_big_d;
    (
        wc / sum,
        wr / sum,
        wd / sum,
        w_big_r / sum,
        wl / sum,
        w_big_d / sum,
    )
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

#[allow(clippy::too_many_arguments)]
fn run_dispatch(
    label: &str,
    width: usize,
    height: usize,
    samples: u32,
    warmup: u32,
) -> (f64, f64, f64) {
    let (wc, wr, wd, w_big_r, wl, w_big_d) = default_weights();
    let mut data = synthetic_xyb_y(width, height);
    let mut scratch = vec![0.0f32; width * height];

    for _ in 0..warmup {
        jxl_simd::gaborish_5x5_channel(
            &mut data,
            &mut scratch,
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );
        std::hint::black_box(&data);
    }

    let mut times_ms = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let start = Instant::now();
        jxl_simd::gaborish_5x5_channel(
            &mut data,
            &mut scratch,
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&data);
        times_ms.push(elapsed);
    }

    let med = median(times_ms.clone());
    let best = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    eprintln!(
        "  {label} {width}x{height}: median={med:.3} ms, best={best:.3} ms, mean={mean:.3} ms"
    );
    (med, best, mean)
}

#[allow(clippy::too_many_arguments)]
fn run_scalar(
    label: &str,
    width: usize,
    height: usize,
    samples: u32,
    warmup: u32,
) -> (f64, f64, f64) {
    let (wc, wr, wd, w_big_r, wl, w_big_d) = default_weights();
    let input = synthetic_xyb_y(width, height);
    let mut output = vec![0.0f32; width * height];

    for _ in 0..warmup {
        jxl_simd::gaborish_5x5_scalar(
            &mut output,
            &input,
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );
        std::hint::black_box(&output);
    }

    let mut times_ms = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let start = Instant::now();
        jxl_simd::gaborish_5x5_scalar(
            &mut output,
            &input,
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&output);
        times_ms.push(elapsed);
    }

    let med = median(times_ms.clone());
    let best = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    eprintln!(
        "  {label} {width}x{height}: median={med:.3} ms, best={best:.3} ms, mean={mean:.3} ms"
    );
    (med, best, mean)
}

fn main() {
    let sizes = parse_sizes();
    let samples = parse_u32_env("GABOR_SAMPLES", 11);
    let warmup = parse_u32_env("GABOR_WARMUP", 2);

    eprintln!("gaborish_5x5 magetypes bench: sizes={sizes:?} samples={samples} warmup={warmup}");

    let ts = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let host = env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());
    // Try git first; fall back to jj (jj workspaces lack a per-workspace .git).
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::process::Command::new("jj")
                .args(["log", "-r", "@", "--no-graph", "-T", "commit_id.short(8)"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unknown".into());

    let tsv_path = "/tmp/gaborish5x5_magetypes_bench.tsv";
    let meta_path = "/tmp/gaborish5x5_magetypes_bench.meta";

    let mut tsv = File::create(tsv_path).expect("create TSV");
    writeln!(
        tsv,
        "label\twidth\theight\tpixels\tmedian_ms\tbest_ms\tmean_ms"
    )
    .unwrap();

    for &n in &sizes {
        let w = n;
        let h = n;

        // Dispatcher (picks best SIMD tier at runtime)
        let (med, best, mean) = run_dispatch("dispatch  ", w, h, samples, warmup);
        writeln!(tsv, "dispatch\t{w}\t{h}\t{}\t{med}\t{best}\t{mean}", w * h).unwrap();

        // Scalar baseline (no SIMD)
        let (med_s, best_s, mean_s) = run_scalar("scalar    ", w, h, samples, warmup);
        writeln!(
            tsv,
            "scalar\t{w}\t{h}\t{}\t{med_s}\t{best_s}\t{mean_s}",
            w * h
        )
        .unwrap();

        let speedup = med_s / med;
        eprintln!("  -> {w}x{h} median speedup: {speedup:.2}x");
    }

    drop(tsv);

    let mut meta = File::create(meta_path).expect("create meta");
    writeln!(meta, "# gaborish_5x5 magetypes consolidation bench").unwrap();
    writeln!(meta, "timestamp: {ts}").unwrap();
    writeln!(meta, "hostname: {host}").unwrap();
    writeln!(meta, "git_commit: {commit}").unwrap();
    writeln!(meta, "sizes: {sizes:?}").unwrap();
    writeln!(meta, "samples_per_cell: {samples}").unwrap();
    writeln!(meta, "warmup: {warmup}").unwrap();
    writeln!(
        meta,
        "purpose: validates that the magetypes(define(f32x8), v4, v3, neon, wasm128, scalar) consolidation"
    )
    .unwrap();
    writeln!(
        meta,
        "         in jxl-encoder-simd/src/gaborish5x5.rs preserves wall-clock vs the prior 3-variant hand-written code"
    )
    .unwrap();
    writeln!(
        meta,
        "         (and adds a NEW wasm128 SIMD body where the previous crate fell through to scalar)"
    )
    .unwrap();
    writeln!(
        meta,
        "kernel:   gaborish_5x5_channel — 21-tap 5x5 weighted convolution (sharpening pre-filter)"
    )
    .unwrap();
    writeln!(
        meta,
        "input:    synthetic XYB-Y plane (smooth ramp + texture)"
    )
    .unwrap();
    writeln!(
        meta,
        "weights:  libjxl default Y-channel tuple, normalized to sum=1"
    )
    .unwrap();
    writeln!(meta, "tsv:      {tsv_path}").unwrap();
    drop(meta);

    eprintln!();
    eprintln!("TSV  written to {tsv_path}");
    eprintln!("META written to {meta_path}");
    eprintln!("Move into benchmarks/ once verified (atomic mv pattern per jj race notes).");
}
