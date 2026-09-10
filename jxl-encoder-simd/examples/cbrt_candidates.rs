// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
//! Which cube root should the forward XYB transform use?
//!
//! The cube root is 86-87 % of the forward XYB transform
//! (`jxl-encoder/benchmarks/e3_profile_2026-09-10.*`), and the transform is
//! 15.3 % of lossy e3 CPU, so this one kernel is worth measuring properly.
//!
//! Candidates, all evaluated at the platform's native vector width:
//!
//! * `ours_f64` — the shipped `cbrt_fast`: bit-hack guess, then TWO Newton
//!   iterations in f64, each with a DIVISION. Bit-exact with today's output.
//!   Needs `f64x4`, which is native on x86 and a doubled polyfill on NEON.
//! * `mt_lowp` / `mt_midp` — `magetypes`' own `cbrt_lowp` / `cbrt_midp`
//!   (Kahan bit-hack + 1 or 2 Halley steps; 1 resp. 2 divisions).
//!   `cbrt_lowp`'s own docs name perceptual colour (Oklab/XYB) as its target.
//! * `libjxl` — a transcription of libjxl `base/fast_math-inl.h`
//!   `CubeRootAndAdd`: Newton on the INVERSE cube root using only multiplies
//!   and FMAs, with the initial guess done in INTEGER VECTOR LANES. No
//!   divisions and no scalar round-trip anywhere.
//!
//! Accuracy is reported against `f64::cbrt` over the range the transform
//! actually produces, not over all of f32.
//!
//! Run: `cargo run --release -p jxl-encoder-simd --example cbrt_candidates`

use std::hint::black_box;
use std::time::Instant;

/// Bias added before the cube root in the opsin transform (libjxl
/// `cms/opsin_params.h`). The post-matrix values the cube root actually sees
/// are `>= this`, which is why accuracy is measured on `[bias, ~1.2]` and not
/// on all of f32.
const OPSIN_BIAS: f32 = 0.003_793_073_4;

/// The shipped scalar `cbrt_fast`, verbatim, as the accuracy/speed baseline.
#[inline(always)]
fn cbrt_ours_scalar(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let sign = ui & 0x8000_0000;
    let hx = ui & 0x7FFF_FFFF;
    let approx = hx / 3 + B1;
    let mut t = f64::from(f32::from_bits(sign | approx));
    let xf64 = f64::from(x);
    let r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    let r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);

    // Inputs in the range the opsin matrix actually produces: linear RGB in
    // [0, 1] through the matrix plus the bias, clamped at 0, so roughly
    // [bias, ~1.1]. A fraction are exactly the bias (black pixels), which is
    // the case the zero/denormal handling has to survive.
    let mut seed = 0x9E37_79B9u32;
    let src: Vec<f32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = (seed >> 8) as f32 / 16_777_215.0;
            if v < 0.03 {
                OPSIN_BIAS
            } else {
                v * 1.1 + OPSIN_BIAS
            }
        })
        .collect();
    let mut out = vec![0.0f32; n];

    let reference: Vec<f64> = src.iter().map(|&x| f64::from(x).cbrt()).collect();

    type Arm = (&'static str, fn(&[f32], &mut [f32]));
    let arms: [Arm; 5] = [
        ("ours_f64 (shipped cbrt_fast)", run_ours),
        ("magetypes cbrt_lowp", run_mt_lowp),
        ("magetypes cbrt_midp", run_mt_midp),
        ("libjxl CubeRootAndAdd", run_libjxl),
        ("std cbrtf (scalar)", run_std),
    ];

    let mut best = [f64::INFINITY; 5];
    for rep in 0..reps {
        for k in 0..arms.len() {
            let a = (k + rep) % arms.len();
            let t = Instant::now();
            arms[a].1(&src, &mut out);
            best[a] = best[a].min(t.elapsed().as_secs_f64() * 1000.0);
            black_box(&out);
        }
    }

    println!("{n} values, min of {reps}, single-threaded, input in [bias, ~1.1]:");
    println!(
        "{:<30} {:>9} {:>10} {:>12} {:>12}",
        "candidate", "ms", "ns/value", "max rel err", "max ULP"
    );
    for (a, (label, f)) in arms.iter().enumerate() {
        f(&src, &mut out);
        let mut max_rel = 0.0f64;
        let mut max_ulp = 0i64;
        for i in 0..n {
            let got = f64::from(out[i]);
            let want = reference[i];
            if want != 0.0 {
                max_rel = max_rel.max(((got - want) / want).abs());
            }
            let a_bits = out[i].to_bits() as i64;
            let b_bits = (want as f32).to_bits() as i64;
            max_ulp = max_ulp.max((a_bits - b_bits).abs());
        }
        println!(
            "{:<30} {:>9.2} {:>10.2} {:>12.3e} {:>12}",
            label,
            best[a],
            best[a] * 1e6 / n as f64,
            max_rel,
            max_ulp
        );
    }
}

fn run_ours(src: &[f32], out: &mut [f32]) {
    for (o, &x) in out.iter_mut().zip(src) {
        *o = cbrt_ours_scalar(x);
    }
}

fn run_std(src: &[f32], out: &mut [f32]) {
    for (o, &x) in out.iter_mut().zip(src) {
        *o = x.cbrt();
    }
}

// ── vector arms ────────────────────────────────────────────────────────────
//
// Dispatched through `incant!` exactly like production kernels, so each arm is
// measured in the same `#[target_feature]` context the encoder would use.

fn run_mt_lowp(src: &[f32], out: &mut [f32]) {
    jxl_encoder_simd::bench_cbrt::cbrt_lowp_batch(src, out);
}
fn run_mt_midp(src: &[f32], out: &mut [f32]) {
    jxl_encoder_simd::bench_cbrt::cbrt_midp_batch(src, out);
}
fn run_libjxl(src: &[f32], out: &mut [f32]) {
    jxl_encoder_simd::bench_cbrt::cbrt_libjxl_batch(src, out);
}
