// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
//! Direct timing harness for the DCT-32 / DCT-64 forward kernels.
//!
//! Used to A/B the FMA-ized B-transform butterfly (survey item #1) against the
//! pristine mul-then-add form. Build this example against each source variant
//! and compare wall-time; the kernel selection is identical, only the inner
//! `s[0] = sqrt2 * s[0] + s[1]` differs (mul+add vs fused mul_add).
//!
//! Methodology: warm up, then run many iterations summing a checksum so the
//! optimizer can't elide the work. Reports ns/call for each kernel size.

use std::hint::black_box;
use std::time::Instant;

fn fill(seed: &mut u64) -> f32 {
    // cheap xorshift -> f32 in [-128, 128)
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / 16777216.0 - 0.5) * 256.0
}

fn main() {
    let iters: u64 = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000_000);

    let mut seed = 0x9E3779B97F4A7C15u64;

    // 32x32 input
    let mut in32 = [0.0f32; 1024];
    for v in in32.iter_mut() {
        *v = fill(&mut seed);
    }
    // 64x64 input
    let mut in64 = [0.0f32; 4096];
    for v in in64.iter_mut() {
        *v = fill(&mut seed);
    }

    let mut out32 = [0.0f32; 1024];
    let mut out64 = [0.0f32; 4096];

    // ---- warmup ----
    for _ in 0..50_000 {
        jxl_encoder_simd::dct_32x32(black_box(&in32), black_box(&mut out32));
        black_box(&out32);
    }

    // ---- dct_32x32 ----
    let t = Instant::now();
    let mut acc32 = 0.0f32;
    for _ in 0..iters {
        jxl_encoder_simd::dct_32x32(black_box(&in32), black_box(&mut out32));
        acc32 += out32[7] + out32[513];
    }
    let d32 = t.elapsed();
    black_box(acc32);

    // ---- dct_64x64 (fewer iters: ~4x the work) ----
    let iters64 = (iters / 4).max(1);
    for _ in 0..20_000 {
        jxl_encoder_simd::dct_64x64(black_box(&in64), black_box(&mut out64));
        black_box(&out64);
    }
    let t = Instant::now();
    let mut acc64 = 0.0f32;
    for _ in 0..iters64 {
        jxl_encoder_simd::dct_64x64(black_box(&in64), black_box(&mut out64));
        acc64 += out64[15] + out64[2049];
    }
    let d64 = t.elapsed();
    black_box(acc64);

    println!(
        "dct_32x32: {:.3} ns/call  ({} iters)  checksum={}",
        d32.as_nanos() as f64 / iters as f64,
        iters,
        acc32
    );
    println!(
        "dct_64x64: {:.3} ns/call  ({} iters)  checksum={}",
        d64.as_nanos() as f64 / iters64 as f64,
        iters64,
        acc64
    );
}
