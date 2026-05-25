// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Test infrastructure for SIMD scalar-vs-dispatch parity testing.
//!
//! Motivated by SA-G commit `7d383785`: the CfL Newton SIMD diverged from
//! libjxl despite "parity" claims because existing tests only checked
//! scalar-vs-dispatch on 1-2 fixed input cases, missing tail-loop boundary
//! sizes (e.g. SIMD-width + 1), edge values (zeros, denormals, +/-Inf, NaN),
//! and reduction-order divergence on inputs that stress the merge tree.
//!
//! This module provides:
//!
//! 1. [`edge_case_sizes`] — canonical battery of element-count cases that
//!    exercises tail loops at every SIMD width the crate ships (4/8/16 lanes).
//!
//! 2. [`f32_edge_battery`] — canonical battery of f32 input distributions
//!    (zeros, all-equal, small/large, denormals, sign mix). NaN/Inf cases are
//!    OPT-IN per kernel via [`f32_nonfinite_battery`] because most kernels
//!    cannot accept NaN without behavioral divergence (e.g. reductions).
//!
//! 3. [`gen_f32`] — deterministic seeded f32 generator (no `rand` dep).
//!
//! 4. [`assert_f32_bit_eq`] / [`assert_f32_close`] — assertion helpers that
//!    print the divergence location and tier label on mismatch.
//!
//! 5. [`run_dispatch_parity`] — the canonical "for each permutation, compare
//!    scalar-vs-dispatch" wrapper. Encapsulates the 20-line pattern from
//!    cfl.rs into a 3-line call.
//!
//! # Pattern
//!
//! ```rust,ignore
//! use crate::test_helpers::*;
//!
//! #[test]
//! fn my_kernel_scalar_vs_dispatch() {
//!     for &n in edge_case_sizes() {
//!         for case in f32_edge_battery(n) {
//!             let ref_out = my_kernel_scalar(&case);
//!             run_dispatch_parity(|perm| {
//!                 let test_out = my_kernel_dispatch(&case);
//!                 assert_f32_bit_eq(ref_out, test_out, perm, &format!("n={n}"));
//!             });
//!         }
//!     }
//! }
//! ```

#![cfg(test)]

extern crate std;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Canonical battery of element-count cases for SIMD tail-loop coverage.
///
/// Covers:
/// - 0 (empty)
/// - 1 (sub-lane)
/// - 3, 4, 5 (around f32x4 boundary — NEON, scalar)
/// - 7, 8, 9 (around f32x8 boundary — AVX2)
/// - 15, 16, 17 (around f32x16 boundary — AVX-512)
/// - 31, 32, 33 (2 full f32x16 chunks +/- 1)
/// - 63, 64, 65 (8x8 block coverage — DCT-relevant)
/// - 127, 128, 129 (large)
/// - 1023, 1024 (large enough to stress chunked reductions)
pub const fn edge_case_sizes() -> &'static [usize] {
    &[
        0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024,
    ]
}

/// Sizes ≥ 1 (skip empty for kernels that require non-empty input).
pub const fn edge_case_sizes_nonempty() -> &'static [usize] {
    &[
        1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024,
    ]
}

/// Deterministic seeded f32 generator (xorshift64*).
///
/// Output range: ~[-2.0, +2.0]. Reproducible across runs.
pub fn gen_f32(seed: u64, n: usize, range: f32) -> Vec<f32> {
    let mut s = seed | 1; // never zero
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32;
        // Map [0, 2^32) → [-range, range].
        let f = (u as f32 / u32::MAX as f32) * (2.0 * range) - range;
        v.push(f);
    }
    v
}

/// Deterministic seeded f32 generator in [0, 1].
pub fn gen_f32_unit(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32;
        v.push(u as f32 / u32::MAX as f32);
    }
    v
}

/// Labeled f32 input case (label printed on assertion failure).
#[derive(Clone)]
pub struct F32Case {
    pub label: String,
    pub data: Vec<f32>,
}

/// Battery of f32 input distributions of length `n` (excluding non-finite cases).
///
/// Use [`f32_nonfinite_battery`] for kernels that explicitly accept NaN/Inf.
pub fn f32_edge_battery(n: usize) -> Vec<F32Case> {
    let mut out = Vec::new();
    if n == 0 {
        out.push(F32Case {
            label: format!("empty(n=0)"),
            data: Vec::new(),
        });
        return out;
    }
    let mk = |label: &str, data: Vec<f32>| F32Case {
        label: format!("{label}(n={n})"),
        data,
    };
    out.push(mk("zeros", alloc::vec![0.0; n]));
    out.push(mk("ones", alloc::vec![1.0; n]));
    out.push(mk("neg_ones", alloc::vec![-1.0; n]));
    out.push(mk("small_pos", alloc::vec![1e-20_f32; n]));
    out.push(mk("large_pos", alloc::vec![1e20_f32; n]));
    out.push(mk(
        "alternating_sign",
        (0..n)
            .map(|i| if i & 1 == 0 { 1.0 } else { -1.0 })
            .collect(),
    ));
    out.push(mk(
        "ramp",
        (0..n).map(|i| (i as f32 - n as f32 * 0.5) * 0.1).collect(),
    ));
    out.push(mk("rand_a", gen_f32(0xA5A5_5A5A, n, 1.0)));
    out.push(mk("rand_b", gen_f32(0xC3C3_3C3C, n, 100.0)));
    // Subnormal/denormal — exercises FTZ behavior.
    out.push(mk("denormals", alloc::vec![f32::MIN_POSITIVE * 0.5; n]));
    out
}

/// Battery including ±Inf and NaN cases.  ONLY use on kernels documented to
/// handle non-finite input (sanitize_finite, is_finite_plane).
pub fn f32_nonfinite_battery(n: usize) -> Vec<F32Case> {
    let mut out = f32_edge_battery(n);
    if n == 0 {
        return out;
    }
    let mk = |label: &str, data: Vec<f32>| F32Case {
        label: format!("{label}(n={n})"),
        data,
    };
    out.push(mk("all_nan", alloc::vec![f32::NAN; n]));
    out.push(mk("all_pos_inf", alloc::vec![f32::INFINITY; n]));
    out.push(mk("all_neg_inf", alloc::vec![f32::NEG_INFINITY; n]));
    // Mixed: NaN at start, middle, end (exercises tail-loop scalar fallback).
    let mut mix = alloc::vec![1.0_f32; n];
    mix[0] = f32::NAN;
    if n > 1 {
        mix[n - 1] = f32::INFINITY;
    }
    if n > 2 {
        mix[n / 2] = f32::NEG_INFINITY;
    }
    out.push(mk("nan_inf_at_boundary", mix));
    out
}

/// Assert two f32 values are bit-equal (same NaN payload, +0/-0 distinguished).
///
/// On mismatch prints the diverging values, the permutation label, and the
/// caller-supplied context.
#[track_caller]
pub fn assert_f32_bit_eq(
    expected: f32,
    actual: f32,
    perm: &archmage::testing::TokenPermutation,
    ctx: &str,
) {
    if expected.to_bits() != actual.to_bits() {
        panic!(
            "scalar-vs-dispatch divergence: expected={expected:?} bits={:#010x} \
             actual={actual:?} bits={:#010x} perm={perm} ctx={ctx}",
            expected.to_bits(),
            actual.to_bits(),
        );
    }
}

/// Assert two f32 slices are bit-equal element-wise (NaN-aware via bits).
#[track_caller]
pub fn assert_f32_slice_bit_eq(
    expected: &[f32],
    actual: &[f32],
    perm: &archmage::testing::TokenPermutation,
    ctx: &str,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "length divergence: expected={} actual={} perm={perm} ctx={ctx}",
        expected.len(),
        actual.len()
    );
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e.to_bits() != a.to_bits() {
            panic!(
                "scalar-vs-dispatch divergence at [{i}/{}]: \
                 expected={e:?} bits={:#010x} actual={a:?} bits={:#010x} perm={perm} ctx={ctx}",
                expected.len(),
                e.to_bits(),
                a.to_bits(),
            );
        }
    }
}

/// Assert two f32 slices are close within `ulps_tol` ULPs (NaN-aware).
///
/// Use when the SIMD kernel cannot be bit-exact (different reduction-tree
/// order, FMA association). The tolerance must be documented per kernel.
///
/// For values near zero where ULP is meaningless, also accepts absolute
/// difference ≤ `abs_floor`.  Default `abs_floor = 1e-6` per call site.
#[track_caller]
pub fn assert_f32_slice_close_ulps_abs(
    expected: &[f32],
    actual: &[f32],
    ulps_tol: u32,
    abs_floor: f32,
    perm: &archmage::testing::TokenPermutation,
    ctx: &str,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "length divergence: expected={} actual={} perm={perm} ctx={ctx}",
        expected.len(),
        actual.len()
    );
    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e.is_nan() && a.is_nan() {
            continue;
        }
        let abs_diff = (e - a).abs();
        if abs_diff <= abs_floor {
            continue;
        }
        let eu = e.to_bits() as i32;
        let au = a.to_bits() as i32;
        let ulps = (eu - au).unsigned_abs();
        if ulps > ulps_tol {
            panic!(
                "scalar-vs-dispatch ULP+abs divergence at [{i}/{}]: \
                 expected={e:?} bits={:#010x} actual={a:?} bits={:#010x} \
                 ulps={ulps} > tol={ulps_tol} abs_diff={abs_diff:e} > floor={abs_floor:e} \
                 perm={perm} ctx={ctx}",
                expected.len(),
                e.to_bits(),
                a.to_bits(),
            );
        }
    }
}

/// Strict-ULP variant (no absolute floor).  Use when the kernel's output
/// range is far from zero.
#[track_caller]
pub fn assert_f32_slice_close_ulps(
    expected: &[f32],
    actual: &[f32],
    ulps_tol: u32,
    perm: &archmage::testing::TokenPermutation,
    ctx: &str,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "length divergence: expected={} actual={} perm={perm} ctx={ctx}",
        expected.len(),
        actual.len()
    );
    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e.is_nan() && a.is_nan() {
            continue;
        }
        let eu = e.to_bits() as i32;
        let au = a.to_bits() as i32;
        let ulps = (eu - au).unsigned_abs();
        if ulps > ulps_tol {
            panic!(
                "scalar-vs-dispatch ULP divergence at [{i}/{}]: \
                 expected={e:?} bits={:#010x} actual={a:?} bits={:#010x} \
                 ulps={ulps} > tol={ulps_tol} perm={perm} ctx={ctx}",
                expected.len(),
                e.to_bits(),
                a.to_bits(),
            );
        }
    }
}

/// Canonical scalar-vs-dispatch parity wrapper.
///
/// Equivalent to:
/// ```rust,ignore
/// let report = archmage::testing::for_each_token_permutation(
///     archmage::testing::CompileTimePolicy::Warn,
///     |perm| f(perm),
/// );
/// std::eprintln!("{report}");
/// ```
pub fn run_dispatch_parity<F: FnMut(&archmage::testing::TokenPermutation)>(mut f: F) {
    let report = archmage::testing::for_each_token_permutation(
        archmage::testing::CompileTimePolicy::Warn,
        |perm| f(perm),
    );
    std::eprintln!("{report}");
}

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn edge_case_sizes_covers_simd_boundaries() {
        let sizes = edge_case_sizes();
        // Must cover f32x4 (NEON), f32x8 (AVX2), f32x16 (AVX-512) boundaries.
        for &boundary in &[4_usize, 8, 16] {
            assert!(
                sizes.iter().any(|&s| s == boundary - 1),
                "missing {} (size - 1)",
                boundary
            );
            assert!(
                sizes.iter().any(|&s| s == boundary),
                "missing {}",
                boundary
            );
            assert!(
                sizes.iter().any(|&s| s == boundary + 1),
                "missing {} (size + 1)",
                boundary
            );
        }
    }

    #[test]
    fn gen_f32_is_deterministic() {
        let a = gen_f32(0xCAFE_BABE, 16, 1.0);
        let b = gen_f32(0xCAFE_BABE, 16, 1.0);
        assert_eq!(a, b);
        // Use a very different seed to avoid xorshift period-1 collisions
        // from neighboring seeds at small step counts.
        let c = gen_f32(0xFFFF_0000_AAAA_5555, 16, 1.0);
        assert_ne!(a, c);
    }

    #[test]
    fn gen_f32_range_respected() {
        let v = gen_f32(0xDEAD_BEEF, 1000, 5.0);
        assert!(v.iter().all(|&x| x.abs() <= 5.0 + 1e-6));
    }

    #[test]
    fn f32_edge_battery_n0_returns_empty_case() {
        let cases = f32_edge_battery(0);
        assert_eq!(cases.len(), 1);
        assert!(cases[0].data.is_empty());
    }

    #[test]
    fn f32_edge_battery_n_positive_includes_zeros_ones_ramp() {
        let cases = f32_edge_battery(8);
        let labels: Vec<&str> = cases.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.iter().any(|l| l.starts_with("zeros")));
        assert!(labels.iter().any(|l| l.starts_with("ones")));
        assert!(labels.iter().any(|l| l.starts_with("ramp")));
        for c in &cases {
            assert_eq!(c.data.len(), 8, "case {} has wrong length", c.label);
        }
    }

    #[test]
    fn f32_nonfinite_battery_adds_nan_inf_cases() {
        let cases = f32_nonfinite_battery(8);
        let labels: Vec<&str> = cases.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.iter().any(|l| l.starts_with("all_nan")));
        assert!(labels.iter().any(|l| l.starts_with("all_pos_inf")));
        assert!(labels.iter().any(|l| l.starts_with("all_neg_inf")));
    }
}
