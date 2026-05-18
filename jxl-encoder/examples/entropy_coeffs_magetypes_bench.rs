//! Microbenchmark for the magetypes-consolidated `entropy_estimate_coeffs`
//! kernel (`entropy_coeffs_impl` in jxl-encoder-simd).
//!
//! Measures wall-clock per call at 5 image sizes. The kernel runs once per
//! candidate AC strategy per 8×8 block during `estimate_entropy_full` —
//! this is the single biggest VarDCT encoder hotspot (~7.5 % CPU at e7).
//! The bench drives many DCT8-shaped block evaluations (64 coeffs each) over
//! the image's block grid, alternating pixel-domain and coefficient-domain
//! modes to exercise both branches in the kernel body.
//!
//! Writes paired TSV + .meta to `/tmp/`; the runner caller then atomically
//! `mv`s the TSV into `benchmarks/` once the run is clean (avoids
//! jj-vs-fd races per
//! `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/jj_concurrent_workspace_failure_mode_2026-05-18.md`).
//!
//! Run:
//!   cargo run -p jxl-encoder --release --example entropy_coeffs_magetypes_bench
//!
//! Optional env knobs:
//!   ENTROPY_COEFFS_SAMPLES=15   # samples per cell (default 11)
//!   ENTROPY_COEFFS_WARMUP=2     # warm-up samples discarded (default 2)
//!   ENTROPY_COEFFS_SIZES="256,512,1024,2048,4096"   # comma-sep widths/heights

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use jxl_simd::{entropy_coeffs_scalar, entropy_estimate_coeffs};

const BLOCK_N: usize = 64; // DCT8 coefficient count

fn parse_sizes() -> Vec<usize> {
    env::var("ENTROPY_COEFFS_SIZES")
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

/// Synthetic per-block coefficient distribution. Values typical for VarDCT:
/// large DC term + decaying AC magnitudes with sign variation.
fn synthetic_blocks(num_blocks: usize) -> Vec<f32> {
    let n = num_blocks * BLOCK_N;
    let mut buf = vec![0.0f32; n];
    for b in 0..num_blocks {
        let base = b * BLOCK_N;
        let s = b as f32 * 0.013;
        for i in 0..BLOCK_N {
            // DC large, AC decays inversely with frequency-like index
            let mag = if i == 0 {
                32.0 + (s.sin() * 4.0)
            } else {
                12.0 / (1.0 + i as f32 * 0.5) * (1.0 + 0.3 * (s + i as f32 * 0.07).sin())
            };
            let sign = if (b + i) % 3 == 0 { -1.0 } else { 1.0 };
            buf[base + i] = mag * sign;
        }
    }
    buf
}

/// Synthetic per-block quantization weights, low-freq small (preserve), high-freq large (discard).
fn synthetic_weights(num_blocks: usize) -> Vec<f32> {
    let n = num_blocks * BLOCK_N;
    let mut buf = vec![0.0f32; n];
    for b in 0..num_blocks {
        let base = b * BLOCK_N;
        for i in 0..BLOCK_N {
            // weight grows from ~0.05 to ~2.0 across the 64 coeffs
            buf[base + i] = 0.05 + (i as f32) * 0.03;
        }
        let _ = b;
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

/// Drive `entropy_estimate_coeffs` over all blocks in the image, alternating
/// pixel-domain and coefficient-domain mode (50/50 split).
#[allow(clippy::too_many_arguments)]
fn one_pass_dispatch(
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    inv_weights: &[f32],
    num_blocks: usize,
    error_coeffs: &mut [f32],
) -> (f64, f64) {
    let cmap_factor = 0.15f32;
    let quant = 3.5f32;
    let k_cost_delta = 5.335f32;
    let k_cost2 = 4.463f32;

    let mut entropy_total = 0.0f64;
    let mut info_loss_total = 0.0f64;
    for b in 0..num_blocks {
        let base = b * BLOCK_N;
        let pixel_domain = (b & 1) == 0;
        let r = entropy_estimate_coeffs(
            &block_c[base..base + BLOCK_N],
            &block_y[base..base + BLOCK_N],
            &weights[base..base + BLOCK_N],
            &inv_weights[base..base + BLOCK_N],
            BLOCK_N,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            pixel_domain,
            &mut error_coeffs[base..base + BLOCK_N],
        );
        entropy_total += r.entropy_sum as f64;
        info_loss_total += r.info_loss_sum as f64;
    }
    (entropy_total, info_loss_total)
}

#[allow(clippy::too_many_arguments)]
fn one_pass_scalar(
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    inv_weights: &[f32],
    num_blocks: usize,
    error_coeffs: &mut [f32],
) -> (f64, f64) {
    let cmap_factor = 0.15f32;
    let quant = 3.5f32;
    let k_cost_delta = 5.335f32;
    let k_cost2 = 4.463f32;

    let mut entropy_total = 0.0f64;
    let mut info_loss_total = 0.0f64;
    for b in 0..num_blocks {
        let base = b * BLOCK_N;
        let pixel_domain = (b & 1) == 0;
        let r = entropy_coeffs_scalar(
            &block_c[base..base + BLOCK_N],
            &block_y[base..base + BLOCK_N],
            &weights[base..base + BLOCK_N],
            &inv_weights[base..base + BLOCK_N],
            BLOCK_N,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            pixel_domain,
            &mut error_coeffs[base..base + BLOCK_N],
        );
        entropy_total += r.entropy_sum as f64;
        info_loss_total += r.info_loss_sum as f64;
    }
    (entropy_total, info_loss_total)
}

fn run_dispatch(label: &str, width: usize, height: usize, samples: u32, warmup: u32) -> f64 {
    // One DCT8-shaped block per 8x8 pixel patch
    let num_blocks = (width / 8) * (height / 8);
    let n = num_blocks * BLOCK_N;
    let block_c = synthetic_blocks(num_blocks);
    let mut block_y = synthetic_blocks(num_blocks);
    // Perturb block_y so it's not identical to block_c
    for v in block_y.iter_mut() {
        *v *= 0.7;
    }
    let weights = synthetic_weights(num_blocks);
    let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();
    let mut error_coeffs = vec![0.0f32; n];

    for _ in 0..warmup {
        let r = one_pass_dispatch(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            num_blocks,
            &mut error_coeffs,
        );
        std::hint::black_box(r);
    }

    let mut times_ms = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let start = Instant::now();
        let r = one_pass_dispatch(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            num_blocks,
            &mut error_coeffs,
        );
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(r);
        times_ms.push(elapsed);
    }

    let med = median(times_ms.clone());
    let best = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    eprintln!(
        "  {label} {width}x{height} ({num_blocks} blocks): median={med:.3} ms, best={best:.3} ms, mean={mean:.3} ms"
    );
    med
}

fn run_scalar(label: &str, width: usize, height: usize, samples: u32, warmup: u32) -> f64 {
    let num_blocks = (width / 8) * (height / 8);
    let n = num_blocks * BLOCK_N;
    let block_c = synthetic_blocks(num_blocks);
    let mut block_y = synthetic_blocks(num_blocks);
    for v in block_y.iter_mut() {
        *v *= 0.7;
    }
    let weights = synthetic_weights(num_blocks);
    let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();
    let mut error_coeffs = vec![0.0f32; n];

    for _ in 0..warmup {
        let r = one_pass_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            num_blocks,
            &mut error_coeffs,
        );
        std::hint::black_box(r);
    }

    let mut times_ms = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let start = Instant::now();
        let r = one_pass_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            num_blocks,
            &mut error_coeffs,
        );
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(r);
        times_ms.push(elapsed);
    }

    let med = median(times_ms.clone());
    let best = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    eprintln!(
        "  {label} {width}x{height} ({num_blocks} blocks): median={med:.3} ms, best={best:.3} ms, mean={mean:.3} ms"
    );
    med
}

fn main() {
    let sizes = parse_sizes();
    let samples = parse_u32_env("ENTROPY_COEFFS_SAMPLES", 11);
    let warmup = parse_u32_env("ENTROPY_COEFFS_WARMUP", 2);

    eprintln!(
        "entropy_coeffs magetypes bench: sizes={sizes:?} samples={samples} warmup={warmup} 50/50 pixel/coeff-domain"
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

    let tsv_path = "/tmp/entropy_coeffs_magetypes_bench.tsv";
    let meta_path = "/tmp/entropy_coeffs_magetypes_bench.meta";

    let mut tsv = File::create(tsv_path).expect("create TSV");
    writeln!(
        tsv,
        "label\twidth\theight\tnum_blocks\tmedian_ms\tspeedup_vs_scalar"
    )
    .unwrap();

    for &n in &sizes {
        let w = n;
        let h = n;
        let num_blocks = (w / 8) * (h / 8);

        let med_disp = run_dispatch("dispatch", w, h, samples, warmup);
        let med_scal = run_scalar("scalar  ", w, h, samples, warmup);
        let speedup = med_scal / med_disp;
        writeln!(
            tsv,
            "dispatch\t{w}\t{h}\t{num_blocks}\t{med_disp}\t{speedup:.3}"
        )
        .unwrap();
        writeln!(tsv, "scalar\t{w}\t{h}\t{num_blocks}\t{med_scal}\t1.000").unwrap();
        eprintln!("  -> {w}x{h} median speedup: {speedup:.2}x");
    }

    drop(tsv);

    let mut meta = File::create(meta_path).expect("create meta");
    writeln!(
        meta,
        "# entropy_coeffs magetypes consolidation bench (W43-2 chunk-7, final)"
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
        "         consolidation in jxl-encoder-simd/src/entropy.rs preserves wall-clock vs the"
    )
    .unwrap();
    writeln!(
        meta,
        "         prior 3-variant hand-written code (AVX2 / NEON / WASM128) and the scalar fallback."
    )
    .unwrap();
    writeln!(
        meta,
        "kernel:  entropy_estimate_coeffs — per-coefficient entropy + nzeros + info_loss accumulation."
    )
    .unwrap();
    writeln!(
        meta,
        "         Called once per candidate AC strategy per 8x8 block during estimate_entropy_full;"
    )
    .unwrap();
    writeln!(
        meta,
        "         single biggest VarDCT encoder hotspot (~7.5 % CPU at e7)."
    )
    .unwrap();
    writeln!(
        meta,
        "shape:   (width/8) * (height/8) DCT8-shaped blocks (64 coeffs each), 50/50 pixel/coeff domain"
    )
    .unwrap();
    writeln!(
        meta,
        "         (alternating) to exercise both branches in the kernel body."
    )
    .unwrap();
    writeln!(
        meta,
        "input:   block_c / block_y synthesized as decaying-AC blocks (large DC, freq-decay AC),"
    )
    .unwrap();
    writeln!(
        meta,
        "         weights grow 0.05 -> ~2.0 across coeffs (low-freq preserve, high-freq discard)."
    )
    .unwrap();
    writeln!(
        meta,
        "note:    v4 (AVX-512) tier IS emitted (pure f32 kernel — all 5 accumulators are f32x8,"
    )
    .unwrap();
    writeln!(
        meta,
        "         no f64 inside the SIMD body). v4 opt-in via the jxl-encoder-simd 'avx512' feature."
    )
    .unwrap();
    writeln!(
        meta,
        "         Accumulator structure (entropy_acc / nzeros_acc / info_loss_acc / info_loss2_acc /"
    )
    .unwrap();
    writeln!(
        meta,
        "         cost2_acc) preserved bit-for-bit from the prior AVX2 / NEON / WASM bodies — FMA"
    )
    .unwrap();
    writeln!(
        meta,
        "         reduction order is load-bearing for W44-9 (DCT8 entropy FMA-order investigation)."
    )
    .unwrap();
    writeln!(meta, "tsv:     {tsv_path}").unwrap();
    drop(meta);

    eprintln!();
    eprintln!("TSV  written to {tsv_path}");
    eprintln!("META written to {meta_path}");
    eprintln!("Move into benchmarks/ once verified (atomic mv pattern per jj race notes).");
}
