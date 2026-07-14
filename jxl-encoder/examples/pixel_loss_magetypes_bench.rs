//! Microbenchmark for the magetypes-consolidated `pixel_domain_loss` kernel.
//!
//! Measures wall-clock per call at 5 image sizes (modelled as a sweep of
//! 8×8 blocks over an N×N XYB-Y-ish plane, matching how
//! `vardct::ac_strategy::estimate_entropy_full` calls the kernel — once
//! per candidate block per channel) and writes paired TSV + .meta to
//! `/tmp/`. The runner caller then atomically `mv`s the TSV into
//! `benchmarks/` once the run is clean (avoids jj-vs-fd races per
//! `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/jj_concurrent_workspace_failure_mode_2026-05-18.md`).
//!
//! Run:
//!   cargo run -p jxl-encoder --release --example pixel_loss_magetypes_bench
//!
//! Optional env knobs:
//!   PIXEL_LOSS_SAMPLES=15     # samples per cell (default 11)
//!   PIXEL_LOSS_WARMUP=2       # warm-up samples discarded (default 2)
//!   PIXEL_LOSS_SIZES="256,512,1024,2048,4096"   # comma-sep widths/heights

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const BLOCK_SIZE: usize = 8;
const MASK_OFFSET_Y: f32 = 0.0; // libjxl's [12.0, 0.0, 4.0] are pre-scaled; 0.0 is the Y channel.

fn parse_sizes() -> Vec<usize> {
    env::var("PIXEL_LOSS_SIZES")
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

/// Synthetic XYB-error plane: roughly the residual after dequant.
/// Values typically in [-0.5, 0.5] range.
fn synthetic_error(width: usize, height: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let tex = ((x as f32 * 0.13).sin()
                + (y as f32 * 0.17).cos()
                + (x as f32 * y as f32 * 0.001).sin())
                * 0.25;
            let sign = if (x ^ y) & 1 == 0 { 1.0 } else { -1.0 };
            buf[y * width + x] = tex * sign;
        }
    }
    buf
}

/// Synthetic mask1x1 field — strictly positive, low-frequency.
/// Values typically in [0.1, 1.0].
fn synthetic_mask(width: usize, height: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let v = 0.55 + 0.35 * ((x as f32 * 0.05).sin() + (y as f32 * 0.07).cos()) * 0.5;
            buf[y * width + x] = v.clamp(0.1, 1.0);
        }
    }
    buf
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

/// Iterate every 8×8 block over an N×N plane, calling `pixel_domain_loss`
/// once per block — same shape as the inner loop of
/// `estimate_entropy_full`. Per-block scratch is sized to BLOCK_SIZE².
#[allow(clippy::too_many_arguments)]
fn one_pass_dispatch(pixel_error: &[f32], mask: &[f32], width: usize, height: usize) -> f64 {
    let blocks_x = width / BLOCK_SIZE;
    let blocks_y = height / BLOCK_SIZE;
    let mut block_error = vec![0.0f32; BLOCK_SIZE * BLOCK_SIZE];
    let mut total = 0.0_f64;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            // Copy 8x8 error block out of the strided plane (mirrors
            // estimate_entropy_full's IDCT-into-scratch pattern).
            for py in 0..BLOCK_SIZE {
                let src = (by * BLOCK_SIZE + py) * width + bx * BLOCK_SIZE;
                let dst = py * BLOCK_SIZE;
                block_error[dst..dst + BLOCK_SIZE]
                    .copy_from_slice(&pixel_error[src..src + BLOCK_SIZE]);
            }
            let mask_base = (by * BLOCK_SIZE) * width + bx * BLOCK_SIZE;
            total += jxl_simd::pixel_domain_loss(
                &block_error,
                mask,
                mask_base,
                width,
                MASK_OFFSET_Y,
                BLOCK_SIZE,
                BLOCK_SIZE,
            );
        }
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn one_pass_scalar(pixel_error: &[f32], mask: &[f32], width: usize, height: usize) -> f64 {
    let blocks_x = width / BLOCK_SIZE;
    let blocks_y = height / BLOCK_SIZE;
    let mut block_error = vec![0.0f32; BLOCK_SIZE * BLOCK_SIZE];
    let mut total = 0.0_f64;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            for py in 0..BLOCK_SIZE {
                let src = (by * BLOCK_SIZE + py) * width + bx * BLOCK_SIZE;
                let dst = py * BLOCK_SIZE;
                block_error[dst..dst + BLOCK_SIZE]
                    .copy_from_slice(&pixel_error[src..src + BLOCK_SIZE]);
            }
            let mask_base = (by * BLOCK_SIZE) * width + bx * BLOCK_SIZE;
            total += jxl_simd::pixel_domain_loss_scalar(
                &block_error,
                mask,
                mask_base,
                width,
                MASK_OFFSET_Y,
                BLOCK_SIZE,
                BLOCK_SIZE,
            );
        }
    }
    total
}

fn run_dispatch(
    label: &str,
    width: usize,
    height: usize,
    samples: u32,
    warmup: u32,
) -> (f64, f64, f64) {
    let pixel_error = synthetic_error(width, height);
    let mask = synthetic_mask(width, height);

    for _ in 0..warmup {
        let t = one_pass_dispatch(&pixel_error, &mask, width, height);
        std::hint::black_box(t);
    }

    let mut times_ms = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let start = Instant::now();
        let t = one_pass_dispatch(&pixel_error, &mask, width, height);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(t);
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

fn run_scalar(
    label: &str,
    width: usize,
    height: usize,
    samples: u32,
    warmup: u32,
) -> (f64, f64, f64) {
    let pixel_error = synthetic_error(width, height);
    let mask = synthetic_mask(width, height);

    for _ in 0..warmup {
        let t = one_pass_scalar(&pixel_error, &mask, width, height);
        std::hint::black_box(t);
    }

    let mut times_ms = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let start = Instant::now();
        let t = one_pass_scalar(&pixel_error, &mask, width, height);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(t);
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
    let samples = parse_u32_env("PIXEL_LOSS_SAMPLES", 11);
    let warmup = parse_u32_env("PIXEL_LOSS_WARMUP", 2);

    eprintln!(
        "pixel_domain_loss magetypes bench: sizes={sizes:?} samples={samples} warmup={warmup} block=8x8"
    );

    let ts = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let host = env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());
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

    let tsv_path = "/tmp/pixel_loss_magetypes_bench.tsv";
    let meta_path = "/tmp/pixel_loss_magetypes_bench.meta";

    let mut tsv = File::create(tsv_path).expect("create TSV");
    writeln!(
        tsv,
        "label\twidth\theight\tpixels\tblocks\tmedian_ms\tbest_ms\tmean_ms"
    )
    .unwrap();

    for &n in &sizes {
        let w = n;
        let h = n;
        let blocks = (w / BLOCK_SIZE) * (h / BLOCK_SIZE);

        let (med, best, mean) = run_dispatch("dispatch  ", w, h, samples, warmup);
        writeln!(
            tsv,
            "dispatch\t{w}\t{h}\t{}\t{blocks}\t{med}\t{best}\t{mean}",
            w * h
        )
        .unwrap();

        let (med_s, best_s, mean_s) = run_scalar("scalar    ", w, h, samples, warmup);
        writeln!(
            tsv,
            "scalar\t{w}\t{h}\t{}\t{blocks}\t{med_s}\t{best_s}\t{mean_s}",
            w * h
        )
        .unwrap();

        let speedup = med_s / med;
        eprintln!("  -> {w}x{h} median speedup: {speedup:.2}x");
    }

    drop(tsv);

    let mut meta = File::create(meta_path).expect("create meta");
    writeln!(meta, "# pixel_domain_loss magetypes consolidation bench").unwrap();
    writeln!(meta, "timestamp: {ts}").unwrap();
    writeln!(meta, "hostname: {host}").unwrap();
    writeln!(meta, "git_commit: {commit}").unwrap();
    writeln!(meta, "sizes: {sizes:?}").unwrap();
    writeln!(meta, "samples_per_cell: {samples}").unwrap();
    writeln!(meta, "warmup: {warmup}").unwrap();
    writeln!(
        meta,
        "purpose: validates that the magetypes(define(f32x8, f64x4), v3, neon, wasm128, scalar)"
    )
    .unwrap();
    writeln!(
        meta,
        "         consolidation in jxl-encoder-simd/src/pixel_loss.rs preserves wall-clock vs"
    )
    .unwrap();
    writeln!(
        meta,
        "         the prior 3-variant hand-written code (AVX2 / NEON / WASM128) and the scalar fallback."
    )
    .unwrap();
    writeln!(
        meta,
        "kernel:   pixel_domain_loss — 8th-power of masked pixel errors (x^2 * x^2 * x^2 chain)"
    )
    .unwrap();
    writeln!(
        meta,
        "shape:    one 8x8 block per call, looped over the entire NxN plane to mirror"
    )
    .unwrap();
    writeln!(
        meta,
        "          estimate_entropy_full's per-block invocation pattern."
    )
    .unwrap();
    writeln!(
        meta,
        "input:    synthetic XYB-error plane + strictly-positive mask1x1-like plane."
    )
    .unwrap();
    writeln!(
        meta,
        "note:     v4 (AVX-512) tier is NOT emitted — magetypes 0.9.23 lacks F64x4Backend for"
    )
    .unwrap();
    writeln!(
        meta,
        "          X64V4Token (the natural f64 width on AVX-512 is f64x8). Ceiling is v3 (AVX2)."
    )
    .unwrap();
    writeln!(meta, "tsv:      {tsv_path}").unwrap();
    drop(meta);

    eprintln!();
    eprintln!("TSV  written to {tsv_path}");
    eprintln!("META written to {meta_path}");
    eprintln!("Move into benchmarks/ once verified (atomic mv pattern per jj race notes).");
}
