//! Microbenchmark for the magetypes-consolidated `linear_rgb_to_xyb_batch`
//! kernel (`forward_xyb_impl` in jxl-encoder-simd).
//!
//! Measures wall-clock per call at 5 image sizes — the kernel runs once per
//! encode over the full image (`vardct::xyb::linear_rgb_to_xyb` calls
//! `jxl_simd::linear_rgb_to_xyb_batch` on width×height f32 planes). Writes
//! paired TSV + .meta to `/tmp/`; the runner caller then atomically `mv`s
//! the TSV into `benchmarks/` once the run is clean (avoids jj-vs-fd races
//! per `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/jj_concurrent_workspace_failure_mode_2026-05-18.md`).
//!
//! Run:
//!   cargo run -p jxl-encoder --release --example forward_xyb_magetypes_bench
//!
//! Optional env knobs:
//!   FORWARD_XYB_SAMPLES=15    # samples per cell (default 11)
//!   FORWARD_XYB_WARMUP=2      # warm-up samples discarded (default 2)
//!   FORWARD_XYB_SIZES="256,512,1024,2048,4096"   # comma-sep widths/heights

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn parse_sizes() -> Vec<usize> {
    env::var("FORWARD_XYB_SIZES")
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

/// Synthetic linear-sRGB-ish plane. Values in [0.0, 1.0].
fn synthetic_plane(width: usize, height: usize, seed: u32) -> Vec<f32> {
    let mut buf = vec![0.0f32; width * height];
    let s = seed as f32;
    for y in 0..height {
        for x in 0..width {
            let v = 0.5
                + 0.4
                    * (((x as f32 * 0.013 + s).sin()
                        + (y as f32 * 0.017 + s * 0.3).cos()
                        + (x as f32 * y as f32 * 0.0001 + s * 0.7).sin())
                        / 3.0);
            buf[y * width + x] = v.clamp(0.0, 1.0);
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

fn one_pass_dispatch(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    jxl_simd::linear_rgb_to_xyb_batch(r, g, b, x_out, y_out, b_out);
}

fn one_pass_scalar(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let n = r.len();
    jxl_simd::forward_xyb_scalar(r, g, b, x_out, y_out, b_out, n);
}

fn run_dispatch(
    label: &str,
    width: usize,
    height: usize,
    samples: u32,
    warmup: u32,
) -> (f64, f64, f64) {
    let n = width * height;
    let r = synthetic_plane(width, height, 11);
    let g = synthetic_plane(width, height, 23);
    let b = synthetic_plane(width, height, 47);
    let mut x_out = vec![0.0f32; n];
    let mut y_out = vec![0.0f32; n];
    let mut b_out = vec![0.0f32; n];

    for _ in 0..warmup {
        one_pass_dispatch(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);
        std::hint::black_box(&x_out);
    }

    let mut times_ms = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let start = Instant::now();
        one_pass_dispatch(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&x_out);
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
    let n = width * height;
    let r = synthetic_plane(width, height, 11);
    let g = synthetic_plane(width, height, 23);
    let b = synthetic_plane(width, height, 47);
    let mut x_out = vec![0.0f32; n];
    let mut y_out = vec![0.0f32; n];
    let mut b_out = vec![0.0f32; n];

    for _ in 0..warmup {
        one_pass_scalar(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);
        std::hint::black_box(&x_out);
    }

    let mut times_ms = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let start = Instant::now();
        one_pass_scalar(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&x_out);
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
    let samples = parse_u32_env("FORWARD_XYB_SAMPLES", 11);
    let warmup = parse_u32_env("FORWARD_XYB_WARMUP", 2);

    eprintln!(
        "forward_xyb magetypes bench: sizes={sizes:?} samples={samples} warmup={warmup} planar f32"
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

    let tsv_path = "/tmp/forward_xyb_magetypes_bench.tsv";
    let meta_path = "/tmp/forward_xyb_magetypes_bench.meta";

    let mut tsv = File::create(tsv_path).expect("create TSV");
    writeln!(
        tsv,
        "label\twidth\theight\tpixels\tmedian_ms\tbest_ms\tmean_ms"
    )
    .unwrap();

    for &n in &sizes {
        let w = n;
        let h = n;

        let (med, best, mean) = run_dispatch("dispatch  ", w, h, samples, warmup);
        writeln!(tsv, "dispatch\t{w}\t{h}\t{}\t{med}\t{best}\t{mean}", w * h).unwrap();

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
    writeln!(
        meta,
        "# forward_xyb magetypes consolidation bench (W43-2 chunk-6)"
    )
    .unwrap();
    writeln!(meta, "timestamp: {ts}").unwrap();
    writeln!(meta, "hostname: {host}").unwrap();
    writeln!(meta, "git_commit: {commit}").unwrap();
    writeln!(meta, "sizes: {sizes:?}").unwrap();
    writeln!(meta, "samples_per_cell: {samples}").unwrap();
    writeln!(meta, "warmup: {warmup}").unwrap();
    writeln!(
        meta,
        "purpose: validates that the magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)"
    )
    .unwrap();
    writeln!(
        meta,
        "         consolidation in jxl-encoder-simd/src/xyb.rs preserves wall-clock vs the prior"
    )
    .unwrap();
    writeln!(
        meta,
        "         3-variant hand-written code (AVX2 / NEON / WASM128) and the scalar fallback."
    )
    .unwrap();
    writeln!(
        meta,
        "kernel:   forward_xyb — linear RGB → XYB (matrix mul + clamp + Newton-Raphson cbrt in f64 + mix)"
    )
    .unwrap();
    writeln!(
        meta,
        "shape:    one call per encode over the full width×height f32 plane (SoA: 3 in, 3 out)."
    )
    .unwrap();
    writeln!(
        meta,
        "input:    3 synthetic planes (R, G, B), each filled with low-freq sinusoids in [0,1]."
    )
    .unwrap();
    writeln!(
        meta,
        "note:     v4 (AVX-512) tier IS emitted (pure f32 kernel — no f64x4 inside the SIMD body)."
    )
    .unwrap();
    writeln!(
        meta,
        "          Contrast with W43-2 chunk-5 (pixel_loss) which required f64x4 and was capped at v3."
    )
    .unwrap();
    writeln!(
        meta,
        "          v4 path opt-in via the jxl-encoder-simd 'avx512' cargo feature."
    )
    .unwrap();
    writeln!(meta, "tsv:      {tsv_path}").unwrap();
    drop(meta);

    eprintln!();
    eprintln!("TSV  written to {tsv_path}");
    eprintln!("META written to {meta_path}");
    eprintln!("Move into benchmarks/ once verified (atomic mv pattern per jj race notes).");
}
