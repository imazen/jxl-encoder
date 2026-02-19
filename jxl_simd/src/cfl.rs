// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated chroma-from-luma (CfL) dot product computation.
//!
//! `find_best_multiplier`: least-squares fitting of integer CfL coefficient.
//! Inner loop is a dual dot product (sum_aa, sum_ab) over up to 4096 elements.

const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;

/// Bias towards zero and quantize to i8 (libjxl enc_chroma_from_luma.cc:176-183).
///
/// Small CfL factors (within ±2.6 of zero) are snapped to zero to reduce
/// oscillations in the CfL map. Larger factors are shifted towards zero by 2.6.
#[inline(always)]
fn bias_and_quantize(x: f32) -> i8 {
    const TOWARDS_ZERO: f32 = 2.6;
    let biased = if x >= TOWARDS_ZERO {
        x - TOWARDS_ZERO
    } else if x <= -TOWARDS_ZERO {
        x + TOWARDS_ZERO
    } else {
        0.0
    };
    biased.round().clamp(-128.0, 127.0) as i8
}

/// Find the best integer CfL multiplier via regularized least-squares.
///
/// Computes: x = -sum_ab / (sum_aa + num * distance_mul * 0.5)
/// where sum_aa = sum(a_i^2), sum_ab = sum(a_i * b_i),
/// a_i = values_m[i] / 84, b_i = base * values_m[i] - values_s[i].
pub fn find_best_multiplier(
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return find_best_multiplier_avx2(token, values_m, values_s, num, base, distance_mul);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return find_best_multiplier_neon(token, values_m, values_s, num, base, distance_mul);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            return find_best_multiplier_wasm128(
                token,
                values_m,
                values_s,
                num,
                base,
                distance_mul,
            );
        }
    }

    find_best_multiplier_scalar(values_m, values_s, num, base, distance_mul)
}

pub fn find_best_multiplier_scalar(
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    if num == 0 {
        return 0;
    }
    let mut sum_aa = 0.0_f32;
    let mut sum_ab = 0.0_f32;
    for i in 0..num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
    }
    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    bias_and_quantize(x)
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn find_best_multiplier_avx2(
    token: archmage::X64V3Token,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    use magetypes::simd::f32x8;

    if num == 0 {
        return 0;
    }

    let inv_cf = f32x8::splat(token, K_INV_COLOR_FACTOR);
    let base_v = f32x8::splat(token, base);
    let mut acc_aa = f32x8::splat(token, 0.0);
    let mut acc_ab = f32x8::splat(token, 0.0);

    let simd_end = num & !7;
    let mut i = 0;
    while i < simd_end {
        let m = f32x8::from_slice(token, &values_m[i..]);
        let s = f32x8::from_slice(token, &values_s[i..]);
        let a = inv_cf * m;
        let b = base_v * m - s;
        acc_aa = a.mul_add(a, acc_aa);
        acc_ab = a.mul_add(b, acc_ab);
        i += 8;
    }

    // Horizontal reduction
    let aa_arr: [f32; 8] = acc_aa.into();
    let ab_arr: [f32; 8] = acc_ab.into();
    let mut sum_aa: f32 = aa_arr.iter().sum();
    let mut sum_ab: f32 = ab_arr.iter().sum();

    // Scalar remainder
    while i < num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
        i += 1;
    }

    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    bias_and_quantize(x)
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn find_best_multiplier_neon(
    token: archmage::NeonToken,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    use magetypes::simd::f32x4;

    if num == 0 {
        return 0;
    }

    let inv_cf = f32x4::splat(token, K_INV_COLOR_FACTOR);
    let base_v = f32x4::splat(token, base);
    let mut acc_aa = f32x4::splat(token, 0.0);
    let mut acc_ab = f32x4::splat(token, 0.0);

    let simd_end = num & !3;
    let mut i = 0;
    while i < simd_end {
        let m = f32x4::from_slice(token, &values_m[i..]);
        let s = f32x4::from_slice(token, &values_s[i..]);
        let a = inv_cf * m;
        let b = base_v * m - s;
        acc_aa = a.mul_add(a, acc_aa);
        acc_ab = a.mul_add(b, acc_ab);
        i += 4;
    }

    let aa_arr: [f32; 4] = acc_aa.into();
    let ab_arr: [f32; 4] = acc_ab.into();
    let mut sum_aa: f32 = aa_arr.iter().sum();
    let mut sum_ab: f32 = ab_arr.iter().sum();

    while i < num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
        i += 1;
    }

    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    bias_and_quantize(x)
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn find_best_multiplier_wasm128(
    token: archmage::Wasm128Token,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    use magetypes::simd::f32x4;

    if num == 0 {
        return 0;
    }

    let inv_cf = f32x4::splat(token, K_INV_COLOR_FACTOR);
    let base_v = f32x4::splat(token, base);
    let mut acc_aa = f32x4::splat(token, 0.0);
    let mut acc_ab = f32x4::splat(token, 0.0);

    let simd_end = num & !3;
    let mut i = 0;
    while i < simd_end {
        let m = f32x4::from_slice(token, &values_m[i..]);
        let s = f32x4::from_slice(token, &values_s[i..]);
        let a = inv_cf * m;
        let b = base_v * m - s;
        acc_aa = a.mul_add(a, acc_aa);
        acc_ab = a.mul_add(b, acc_ab);
        i += 4;
    }

    let aa_arr: [f32; 4] = acc_aa.into();
    let ab_arr: [f32; 4] = acc_ab.into();
    let mut sum_aa: f32 = aa_arr.iter().sum();
    let mut sum_ab: f32 = ab_arr.iter().sum();

    while i < num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
        i += 1;
    }

    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    bias_and_quantize(x)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn test_find_best_multiplier_scalar_vs_dispatch() {
        let num = 256;
        let values_m: alloc::vec::Vec<f32> = (0..num).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let values_s: alloc::vec::Vec<f32> =
            (0..num).map(|i| (i as f32 - 128.0) * 0.05 + 0.3).collect();

        let ref0 = find_best_multiplier_scalar(&values_m, &values_s, num, 0.0, 1e-3);
        let ref1 = find_best_multiplier_scalar(&values_m, &values_s, num, 1.0, 1e-3);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let test0 = find_best_multiplier(&values_m, &values_s, num, 0.0, 1e-3);
                assert_eq!(
                    ref0, test0,
                    "base=0.0: scalar={ref0} dispatch={test0} [{perm}]"
                );

                let test1 = find_best_multiplier(&values_m, &values_s, num, 1.0, 1e-3);
                assert_eq!(
                    ref1, test1,
                    "base=1.0: scalar={ref1} dispatch={test1} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_find_best_multiplier_empty() {
        assert_eq!(find_best_multiplier(&[], &[], 0, 0.0, 1e-3), 0);
    }

    #[test]
    fn test_find_best_multiplier_correlated() {
        let factor = 42.0_f32;
        let m: alloc::vec::Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 10.0).collect();
        let s: alloc::vec::Vec<f32> = m.iter().map(|&v| factor / 84.0 * v).collect();
        let result = find_best_multiplier(&m, &s, 64, 0.0, 1e-3);
        // Optimization produces ~42.0, towards_zero bias subtracts 2.6 → ~39.4 → rounds to 39
        let expected = (factor - 2.6).round() as i8;
        assert!(
            (result as f32 - expected as f32).abs() < 2.0,
            "Expected ~{expected} (factor {factor} - 2.6 bias), got {result}"
        );
    }
}
