// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Detect / replace non-finite f32 values (NaN / ±Inf) in pixel planes.
//!
//! Used at the XYB→downstream-pipeline boundary in the encoder. NaN
//! comparisons are always false, which silently bypasses clamps in the
//! butteraugli iteration loop and propagates non-finite values into
//! `f32 as i32` / `as usize` casts in patch detection and spline
//! rendering — the suspected upstream of the `0x40000000`-prefix index
//! corruption observed in production sweeps.
//!
//! Two entry points:
//!
//! - [`is_finite_plane`] — read-only check. Returns `true` if every
//!   value in the plane is finite. Memory-bandwidth-bound; touches each
//!   byte once. Use when the caller wants to fail-fast with an error.
//!
//! - [`sanitize_finite`] — read-modify-write. Replaces non-finite
//!   values with `0.0` and returns whether any replacement happened.
//!   Memory-bandwidth-bound; touches each byte twice (load + store) on
//!   chunks where any replacement is needed. Use when the caller wants
//!   defense-in-depth on hostile input.
//!
//! Detection trick (shared between both kernels): `v * 0.0` evaluates to:
//! - `0.0` (or `-0.0`) for finite `v` — both compare equal to `0.0`
//! - `NaN` for ±Inf or NaN inputs — does not compare equal to anything,
//!   including `0.0`
//!
//! So `(v * 0.0).simd_eq(splat(0.0))` is the SIMD finite-mask. Blend
//! with `0.0` for sanitize; sum and check finite for the read-only check.

/// Read-only check: returns `true` if every value in `plane` is finite
/// (no NaN, no ±Inf). Faster than [`sanitize_finite`] — no buffer writes,
/// just a load + accumulate + reduce.
///
/// Dispatches to the best available SIMD at runtime.
#[inline]
pub fn is_finite_plane(plane: &[f32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return is_finite_plane_avx2(token, plane);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return is_finite_plane_neon(token, plane);
        }
    }
    is_finite_plane_scalar(plane)
}

#[inline]
pub fn is_finite_plane_scalar(plane: &[f32]) -> bool {
    plane.iter().all(|v| v.is_finite())
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
pub fn is_finite_plane_avx2(token: archmage::X64V3Token, plane: &[f32]) -> bool {
    use magetypes::simd::f32x8;
    let zero = f32x8::splat(token, 0.0);

    let n = plane.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    // Accumulator: `v * 0` is 0 for finite v, NaN for ±Inf or NaN.
    // Sum across chunks — NaN propagates through addition. At the end,
    // if any lane is non-zero or non-finite, some input lane was
    // non-finite.
    let mut acc = f32x8::splat(token, 0.0);

    for c in 0..chunks {
        let off = c * 8;
        let arr: &[f32; 8] = (&plane[off..off + 8]).try_into().unwrap();
        let v = f32x8::load(token, arr);
        acc += v * zero;
    }

    let mut a = [0.0f32; 8];
    acc.store(&mut a);
    if a.iter().any(|&x| x != 0.0 || !x.is_finite()) {
        return false;
    }

    // Tail
    plane[tail_start..].iter().all(|v| v.is_finite())
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
pub fn is_finite_plane_neon(token: archmage::NeonToken, plane: &[f32]) -> bool {
    use magetypes::simd::f32x4;
    let zero = f32x4::splat(token, 0.0);

    let n = plane.len();
    let chunks = n / 4;
    let tail_start = chunks * 4;

    let mut acc = f32x4::splat(token, 0.0);

    for c in 0..chunks {
        let off = c * 4;
        let arr: &[f32; 4] = (&plane[off..off + 4]).try_into().unwrap();
        let v = f32x4::load(token, arr);
        acc += v * zero;
    }

    let mut a = [0.0f32; 4];
    acc.store(&mut a);
    if a.iter().any(|&x| x != 0.0 || !x.is_finite()) {
        return false;
    }

    plane[tail_start..].iter().all(|v| v.is_finite())
}

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

    #[test]
    fn is_finite_plane_scalar_all_finite() {
        let v: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        assert!(is_finite_plane_scalar(&v));
    }

    #[test]
    fn is_finite_plane_scalar_detects_nan() {
        let mut v: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        v[100] = f32::NAN;
        assert!(!is_finite_plane_scalar(&v));
    }

    #[test]
    fn is_finite_plane_scalar_detects_inf() {
        let mut v: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        v[200] = f32::INFINITY;
        assert!(!is_finite_plane_scalar(&v));
        v[200] = 0.0;
        v[200] = f32::NEG_INFINITY;
        assert!(!is_finite_plane_scalar(&v));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn is_finite_plane_avx2_matches_scalar() {
        use archmage::SimdToken;
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };
        let cases: &[Vec<f32>] = &[
            (0..15).map(|i| i as f32).collect(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            {
                let mut v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
                v[500] = f32::NAN;
                v
            },
            {
                let mut v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
                v[7] = f32::INFINITY; // first chunk
                v
            },
            {
                let mut v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
                v[999] = f32::NEG_INFINITY; // tail
                v
            },
            vec![f32::NAN; 17], // all NaN, tail
            vec![],             // empty
        ];
        for (i, v) in cases.iter().enumerate() {
            assert_eq!(
                is_finite_plane_scalar(v),
                is_finite_plane_avx2(token, v),
                "mismatch on case {i}"
            );
        }
    }

    #[test]
    fn is_finite_plane_dispatch() {
        let v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        assert!(is_finite_plane(&v));
        let mut v_bad = v.clone();
        v_bad[500] = f32::NAN;
        assert!(!is_finite_plane(&v_bad));
    }

    // ========================================================================
    // for_each_token_permutation parity coverage
    // ========================================================================
    //
    // sanitize_finite and is_finite_plane are explicitly designed to accept
    // NaN/Inf input.  The output is a (bool, plane) pair where:
    //   - the bool result MUST match scalar bit-for-bit (it's a u8)
    //   - the post-sanitize plane MUST match scalar bit-for-bit (finite
    //     values preserved, non-finite replaced with +0.0)
    // is_finite_plane returns bool only and MUST match scalar exactly.

    use crate::test_helpers::*;
    use alloc::format;

    #[test]
    fn is_finite_plane_scalar_vs_dispatch_sizes() {
        for &n in edge_case_sizes() {
            for case in f32_nonfinite_battery(n) {
                let ref_out = is_finite_plane_scalar(&case.data);
                run_dispatch_parity(|perm| {
                    let act_out = is_finite_plane(&case.data);
                    assert_eq!(
                        ref_out, act_out,
                        "is_finite_plane divergence: scalar={ref_out} dispatch={act_out} \
                         perm={perm} ctx={}",
                        case.label
                    );
                });
            }
        }
    }

    #[test]
    fn sanitize_finite_scalar_vs_dispatch_sizes() {
        for &n in edge_case_sizes() {
            for case in f32_nonfinite_battery(n) {
                let mut ref_plane = case.data.clone();
                let ref_replaced = sanitize_finite_scalar(&mut ref_plane);
                run_dispatch_parity(|perm| {
                    let mut act_plane = case.data.clone();
                    let act_replaced = sanitize_finite(&mut act_plane);
                    assert_eq!(
                        ref_replaced, act_replaced,
                        "sanitize_finite replaced-flag divergence: scalar={ref_replaced} \
                         dispatch={act_replaced} perm={perm} ctx={}",
                        case.label
                    );
                    assert_f32_slice_bit_eq(&ref_plane, &act_plane, perm, &case.label);
                });
            }
        }
    }

    /// Tail-stress: a non-finite value ONLY in the scalar tail (past the
    /// SIMD chunked region).  This is the regression case the existing
    /// fixed-input tests couldn't hit at every size.
    #[test]
    fn sanitize_finite_tail_nan_only() {
        // For each size that has a tail (n % SIMD-width != 0), put a NaN
        // strictly past the last full SIMD chunk on f32x8 boundary.
        for &n in &[9_usize, 17, 33, 65, 129] {
            let mut input = alloc::vec![1.0_f32; n];
            input[n - 1] = f32::NAN; // tail position
            let mut ref_plane = input.clone();
            let ref_replaced = sanitize_finite_scalar(&mut ref_plane);
            assert!(ref_replaced);
            run_dispatch_parity(|perm| {
                let mut act_plane = input.clone();
                let act_replaced = sanitize_finite(&mut act_plane);
                assert!(act_replaced, "tail NaN missed at n={n} perm={perm}");
                assert_f32_slice_bit_eq(&ref_plane, &act_plane, perm, &format!("tail_nan(n={n})"));
            });
        }
    }
}
