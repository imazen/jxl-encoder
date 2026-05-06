// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Replace non-finite f32 values (NaN / ±Inf) with 0.0 in-place.
//!
//! Used at the XYB→downstream-pipeline boundary in the encoder. NaN
//! comparisons are always false, which silently bypasses clamps in the
//! butteraugli iteration loop and propagates non-finite values into
//! `f32 as i32` / `as usize` casts in patch detection and spline
//! rendering — the suspected upstream of the `0x40000000`-prefix index
//! corruption observed in production sweeps.
//!
//! Detection trick: `v * 0.0` evaluates to:
//! - `0.0` (or `-0.0`) for finite `v` — both compare equal to `0.0`
//! - `NaN` for ±Inf or NaN inputs — does not compare equal to anything,
//!   including `0.0`
//!
//! So `(v * 0.0).simd_eq(splat(0.0))` is the SIMD finite-mask. Blend
//! with `0.0` and store. Returns whether any replacement happened so
//! the caller can `debug_assert!` against legitimate inputs producing
//! non-finite XYB (which means upstream invariants regressed).

/// Replace any NaN or ±Inf in `plane` with `0.0`. Returns `true` if any
/// replacement happened.
///
/// Dispatches to the best available SIMD at runtime.
#[inline]
pub fn sanitize_finite(plane: &mut [f32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return sanitize_finite_avx2(token, plane);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return sanitize_finite_neon(token, plane);
        }
    }
    sanitize_finite_scalar(plane)
}

#[inline]
pub fn sanitize_finite_scalar(plane: &mut [f32]) -> bool {
    let mut replaced = false;
    for v in plane.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
            replaced = true;
        }
    }
    replaced
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
pub fn sanitize_finite_avx2(token: archmage::X64V3Token, plane: &mut [f32]) -> bool {
    use magetypes::simd::f32x8;
    let zero = f32x8::splat(token, 0.0);

    let n = plane.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    // Accumulator for "any non-finite seen" — we propagate non-finite
    // values through via floating-point addition (NaN + x = NaN, Inf + x
    // = Inf for finite x). At the end any non-zero or non-finite lane in
    // the accumulator signals a replacement.
    let mut nonfinite_acc = f32x8::splat(token, 0.0);

    for c in 0..chunks {
        let off = c * 8;
        let arr: &mut [f32; 8] = (&mut plane[off..off + 8]).try_into().unwrap();
        let v = f32x8::load(token, arr);
        // `v * 0` is `0.0` for finite v, `NaN` for ±Inf or NaN. Comparing
        // to `0.0` with `simd_eq` (which uses `_CMP_EQ_OQ` — *ordered*
        // equality) returns:
        //   - true  for finite v (0 == 0)
        //   - false for non-finite (NaN comparisons are unordered → false)
        // So this is the FINITE mask. Use it to keep finite lanes and
        // replace non-finite with zero.
        let finite_mask = (v * zero).simd_eq(zero);
        let cleaned = f32x8::blend(finite_mask, v, zero);
        cleaned.store(arr);
        // `v - cleaned`: 0 for finite (cleaned == v), NaN/Inf for
        // non-finite (cleaned == 0). Add to accumulator — any non-finite
        // propagates because NaN + finite = NaN, ±Inf + finite = ±Inf.
        nonfinite_acc += v - cleaned;
    }

    let mut acc = [0.0f32; 8];
    nonfinite_acc.store(&mut acc);
    let mut replaced = acc.iter().any(|&x| x != 0.0 || !x.is_finite());

    // Tail
    for v in plane[tail_start..].iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
            replaced = true;
        }
    }
    replaced
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
pub fn sanitize_finite_neon(token: archmage::NeonToken, plane: &mut [f32]) -> bool {
    use magetypes::simd::f32x4;
    let zero = f32x4::splat(token, 0.0);

    let n = plane.len();
    let chunks = n / 4;
    let tail_start = chunks * 4;

    let mut nonfinite_acc = f32x4::splat(token, 0.0);

    for c in 0..chunks {
        let off = c * 4;
        let arr: &mut [f32; 4] = (&mut plane[off..off + 4]).try_into().unwrap();
        let v = f32x4::load(token, arr);
        // See AVX2 path for the finite-mask trick.
        let finite_mask = (v * zero).simd_eq(zero);
        let cleaned = f32x4::blend(finite_mask, v, zero);
        cleaned.store(arr);
        nonfinite_acc += v - cleaned;
    }

    let mut acc = [0.0f32; 4];
    nonfinite_acc.store(&mut acc);
    let mut replaced = acc.iter().any(|&x| x != 0.0 || !x.is_finite());

    for v in plane[tail_start..].iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
            replaced = true;
        }
    }
    replaced
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn scalar_replaces_nan_inf() {
        let mut v = vec![
            1.0,
            f32::NAN,
            3.0,
            f32::INFINITY,
            -1.0,
            f32::NEG_INFINITY,
            0.0,
            5.0,
        ];
        let replaced = sanitize_finite_scalar(&mut v);
        assert!(replaced);
        assert_eq!(v, vec![1.0, 0.0, 3.0, 0.0, -1.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn scalar_preserves_finite() {
        let mut v = vec![1.0, -2.0, 3.5, -0.0, 1e30, -1e-30];
        let original = v.clone();
        let replaced = sanitize_finite_scalar(&mut v);
        assert!(!replaced);
        assert_eq!(v, original);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        use archmage::SimdToken;
        let Some(token) = archmage::X64V3Token::summon() else {
            return; // CPU without AVX2 — skip
        };
        // Mix of all categories, length not a multiple of 8 to exercise tail.
        let original: Vec<f32> = vec![
            1.0,
            f32::NAN,
            f32::INFINITY,
            -2.0,
            0.0,
            f32::NEG_INFINITY,
            -0.0,
            7.0,
            f32::NAN,
            8.5,
            -1e30,
            1e-30,
            13.0,
            14.0,
            15.0,
        ];
        let mut a = original.clone();
        let mut b = original.clone();
        let r_scalar = sanitize_finite_scalar(&mut a);
        let r_avx2 = sanitize_finite_avx2(token, &mut b);
        assert_eq!(r_scalar, r_avx2);
        assert_eq!(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_clean_input_returns_false() {
        use archmage::SimdToken;
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };
        let mut v: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let replaced = sanitize_finite_avx2(token, &mut v);
        assert!(!replaced, "no NaN in input — should not report replacement");
    }

    #[test]
    fn dispatch_returns_replaced_flag() {
        let mut v = vec![f32::NAN; 100];
        let replaced = sanitize_finite(&mut v);
        assert!(replaced);
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
