// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Batch cube-root wrappers used to SELECT the forward-XYB cube root.
//!
//! `#[doc(hidden)]` — these exist so `examples/cbrt_candidates.rs` can time the
//! candidates in the same `#[target_feature]` context production uses, on the
//! same dispatch path. Production calls the inlined forms in [`crate::xyb`].

use archmage::prelude::*;

/// libjxl `base/fast_math-inl.h::CubeRootAndAdd` (with `add == 0`).
///
/// Newton on the INVERSE cube root: multiplies and FMAs only, **no divisions**,
/// and — unlike every other candidate — the initial guess runs in INTEGER
/// VECTOR LANES instead of round-tripping through `to_array()`. On a polyfilled
/// target that round-trip is most of the cost.
///
/// `x == 0` is handled with a SELECT, matching the original's
/// `IfThenZeroElse`; an early return would put a branch in the inner loop and
/// block vectorisation outright.
#[magetypes(define(f32x8, i32x8), v3, neon, wasm128, scalar)]
pub fn cbrt_libjxl_batch_impl(token: Token, src: &[f32], out: &mut [f32], n: usize) {
    let k1_3 = f32x8::splat(token, 1.0 / 3.0);
    let k4_3 = f32x8::splat(token, 4.0 / 3.0);
    let exp_bias = i32x8::splat(token, 0x5480_0000);
    let exp_mul = i32x8::splat(token, 0x002A_AAAA);
    let izero = i32x8::splat(token, 0);

    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let x = f32x8::from_slice(token, &src[base..]);
        let xa_3 = k1_3 * x;

        let m1 = x.bitcast_i32x8();
        let m2 = exp_bias - m1.shr_arithmetic_const::<23>() * exp_mul;
        let m2 = i32x8::blend(m1.simd_eq(izero), izero, m2);
        let mut r = m2.bitcast_f32x8();

        for _ in 0..3 {
            let r2 = r * r;
            // NegMulAdd(xa_3, r^4, k4_3*r) == k4_3*r - xa_3*r^4, fused.
            r = (-xa_3).mul_add(r2 * r2, k4_3 * r);
        }
        let r2 = r * r;
        // MulAdd(k1_3, NegMulAdd(xa, r^4, r), r)
        r = k1_3.mul_add((-x).mul_add(r2 * r2, r), r);
        let r2 = r * r;
        let os: &mut [f32; 8] = (&mut out[base..base + 8]).try_into().unwrap();
        (r2 * x).store(os);
    }
    for i in (chunks * 8)..n {
        out[i] = crate::xyb::cbrt_libjxl_scalar(src[i]);
    }
}

/// `magetypes::cbrt_lowp` — Kahan bit-hack + ONE Halley step, one division.
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub fn cbrt_lowp_batch_impl(token: Token, src: &[f32], out: &mut [f32], n: usize) {
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let os: &mut [f32; 8] = (&mut out[base..base + 8]).try_into().unwrap();
        f32x8::from_slice(token, &src[base..]).cbrt_lowp().store(os);
    }
    for i in (chunks * 8)..n {
        out[i] = src[i].cbrt();
    }
}

/// `magetypes::cbrt_midp` — Kahan bit-hack + TWO Halley steps, two divisions.
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub fn cbrt_midp_batch_impl(token: Token, src: &[f32], out: &mut [f32], n: usize) {
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let os: &mut [f32; 8] = (&mut out[base..base + 8]).try_into().unwrap();
        f32x8::from_slice(token, &src[base..]).cbrt_midp().store(os);
    }
    for i in (chunks * 8)..n {
        out[i] = src[i].cbrt();
    }
}

#[doc(hidden)]
pub fn cbrt_libjxl_batch(src: &[f32], out: &mut [f32]) {
    let n = src.len().min(out.len());
    incant!(
        cbrt_libjxl_batch_impl(src, out, n),
        [v3, neon, wasm128, scalar]
    )
}

#[doc(hidden)]
pub fn cbrt_lowp_batch(src: &[f32], out: &mut [f32]) {
    let n = src.len().min(out.len());
    incant!(
        cbrt_lowp_batch_impl(src, out, n),
        [v3, neon, wasm128, scalar]
    )
}

#[doc(hidden)]
pub fn cbrt_midp_batch(src: &[f32], out: &mut [f32]) {
    let n = src.len().min(out.len());
    incant!(
        cbrt_midp_batch_impl(src, out, n),
        [v3, neon, wasm128, scalar]
    )
}
