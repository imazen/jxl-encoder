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

/// `cbrt_lowp` with the Kahan bit-hack guess computed in INTEGER VECTOR LANES.
///
/// `magetypes::cbrt_lowp` builds its initial approximation as
/// `f32::from_bits((x.to_bits() / 3) + MAGIC)` through `to_array()` /
/// `from_array()` — a store/load round trip plus one scalar integer division
/// per lane. This arm exists to price that round trip: everything after the
/// guess is identical, so the delta between this and `cbrt_lowp_batch` is the
/// cost of leaving the vector domain.
///
/// **MEASURED 2026-09-10, and it LOSES**: 0.36 ms/Mvalue against `cbrt_lowp`'s
/// 0.29 on aarch64, same 259 max ULP. LLVM already SLP-vectorises the eight
/// scalar `bits / 3` into `umull`/`shrn` across the `to_array()` window, and
/// that is cheaper than the fourteen vector ops `divu3` costs. Kept as the
/// evidence for that null — the round trip LOOKS like the obvious thing to
/// remove and is not.
///
/// The vector `/ 3` is Hacker's Delight `divu3` (shift-add series, then a
/// remainder correction). It is EXACT — verified against `n / 3` over the
/// whole `u32` range, which matters because the result feeds a bit pattern:
/// an off-by-one guess changes the Halley output and would break the
/// cross-tier bit-identity the hash locks depend on. Logical shifts throughout,
/// so `i32x8` carries the `u32` arithmetic unchanged (every magetypes integer
/// `add`/`sub`/`mul` is wrapping, including the scalar backend's).
#[magetypes(define(f32x8, i32x8), v3, neon, wasm128, scalar)]
pub fn cbrt_lowp_vecguess_batch_impl(token: Token, src: &[f32], out: &mut [f32], n: usize) {
    const MAGIC: i32 = 0x2a50_8c2d;
    let magic = i32x8::splat(token, MAGIC);
    let three = i32x8::splat(token, 3);
    let eleven = i32x8::splat(token, 11);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::splat(token, 0.0);
    let sign_mask = f32x8::splat(token, -0.0);

    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let x = f32x8::from_slice(token, &src[base..]);
        let sign = x & sign_mask;
        let abs_x = x.abs();

        // divu3, exact: q ≈ n/3 by shift-add series, then correct by 11*r/32.
        let bits = abs_x.bitcast_i32x8();
        let mut q = bits.shr_logical_const::<2>() + bits.shr_logical_const::<4>();
        q = q + q.shr_logical_const::<4>();
        q = q + q.shr_logical_const::<8>();
        q = q + q.shr_logical_const::<16>();
        let r = bits - q * three;
        let q = q + (r * eleven).shr_logical_const::<5>();

        let mut y = (q + magic).bitcast_f32x8();
        let y3 = y * y * y;
        y *= (y3 + two * abs_x) / (two * y3 + abs_x);

        let os: &mut [f32; 8] = (&mut out[base..base + 8]).try_into().unwrap();
        f32x8::blend(x.simd_eq(zero), x, y | sign).store(os);
    }
    for i in (chunks * 8)..n {
        out[i] = src[i].cbrt();
    }
}

#[doc(hidden)]
pub fn cbrt_lowp_vecguess_batch(src: &[f32], out: &mut [f32]) {
    let n = src.len().min(out.len());
    incant!(
        cbrt_lowp_vecguess_batch_impl(src, out, n),
        [v3, neon, wasm128, scalar]
    )
}
