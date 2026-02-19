// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated coefficient processing for entropy estimation.
//!
//! The inner coefficient loop of `estimate_entropy_full` is the single biggest
//! encoder hotspot (~7.5% CPU). This kernel vectorizes the per-coefficient math:
//!   val = (block_c[i] - block_y[i] * cmap_factor) / weights[i] * quant
//!   rval = round(val)
//!   entropy_sum += sqrt(|rval|) * cost_delta
//!   nzeros += (rval != 0)

/// Results from vectorized entropy coefficient processing.
#[derive(Debug, Clone, Copy)]
pub struct EntropyCoeffResult {
    /// Sum of sqrt(|round(val)|) * cost_delta for all coefficients.
    pub entropy_sum: f32,
    /// Count of non-zero quantized coefficients.
    pub nzeros_sum: f32,
    /// Sum of |val - round(val)| (coefficient-domain mode only).
    pub info_loss_sum: f32,
    /// Sum of (val - round(val))^2 (coefficient-domain mode only).
    pub info_loss2_sum: f32,
}

/// Vectorized entropy coefficient processing.
///
/// For each coefficient i in 0..n:
///   val = (block_c[i] - block_y[i] * cmap_factor) / weights[i] * quant
///   rval = round(val)
///   entropy_sum += sqrt(|rval|) * k_cost_delta
///   nzeros += (rval != 0)
///
/// In pixel-domain mode: writes `error_coeffs[i] = weights[i] * (val - rval)`
/// In coefficient-domain mode: accumulates info_loss stats and k_cost2 penalty.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn entropy_estimate_coeffs(
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    n: usize,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
    pixel_domain: bool,
    error_coeffs: &mut [f32],
) -> EntropyCoeffResult {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return entropy_coeffs_avx2(
                token,
                block_c,
                block_y,
                weights,
                n,
                cmap_factor,
                quant,
                k_cost_delta,
                k_cost2,
                pixel_domain,
                error_coeffs,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return entropy_coeffs_neon(
                token,
                block_c,
                block_y,
                weights,
                n,
                cmap_factor,
                quant,
                k_cost_delta,
                k_cost2,
                pixel_domain,
                error_coeffs,
            );
        }
    }

    entropy_coeffs_scalar(
        block_c,
        block_y,
        weights,
        n,
        cmap_factor,
        quant,
        k_cost_delta,
        k_cost2,
        pixel_domain,
        error_coeffs,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_scalar(
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    n: usize,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
    pixel_domain: bool,
    error_coeffs: &mut [f32],
) -> EntropyCoeffResult {
    let mut entropy_sum = 0.0f32;
    let mut nzeros_sum = 0.0f32;
    let mut info_loss_sum = 0.0f32;
    let mut info_loss2_sum = 0.0f32;

    for i in 0..n {
        let val_in = block_c[i];
        let val_y = block_y[i] * cmap_factor;
        let val = (val_in - val_y) * (1.0 / weights[i]) * quant;
        let rval = val.round();
        let diff = val - rval;

        if pixel_domain {
            error_coeffs[i] = weights[i] * diff;
        }

        let q = rval.abs();
        entropy_sum += q.sqrt() * k_cost_delta;
        if q != 0.0 {
            nzeros_sum += 1.0;
        }

        if !pixel_domain {
            let diff_abs = diff.abs();
            info_loss_sum += diff_abs;
            info_loss2_sum += diff_abs * diff_abs;
            if q >= 1.5 {
                entropy_sum += k_cost2;
            }
        }
    }

    EntropyCoeffResult {
        entropy_sum,
        nzeros_sum,
        info_loss_sum,
        info_loss2_sum,
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_avx2(
    token: archmage::X64V3Token,
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    n: usize,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
    pixel_domain: bool,
    error_coeffs: &mut [f32],
) -> EntropyCoeffResult {
    use magetypes::simd::f32x8;

    let cmap_v = f32x8::splat(token, cmap_factor);
    let quant_v = f32x8::splat(token, quant);
    let cost_delta_v = f32x8::splat(token, k_cost_delta);
    let cost2_v = f32x8::splat(token, k_cost2);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let thr_1_5 = f32x8::splat(token, 1.5);

    let mut entropy_acc = f32x8::zero(token);
    let mut nzeros_acc = f32x8::zero(token);
    let mut info_loss_acc = f32x8::zero(token);
    let mut info_loss2_acc = f32x8::zero(token);
    let mut cost2_acc = f32x8::zero(token);

    let chunks = n / 8;
    // Pre-slice to exact SIMD length so the compiler can prove
    // all from_slice loads are in bounds (base + 8 <= chunks * 8).
    let simd_n = chunks * 8;
    let block_c_s = &block_c[..simd_n];
    let block_y_s = &block_y[..simd_n];
    let weights_s = &weights[..simd_n];
    for chunk in 0..chunks {
        let base = chunk * 8;

        let bc = f32x8::from_slice(token, &block_c_s[base..]);
        let by_v = f32x8::from_slice(token, &block_y_s[base..]);
        let w = f32x8::from_slice(token, &weights_s[base..]);

        // val = (block_c - block_y * cmap_factor) / weights * quant
        let adjusted = bc - by_v * cmap_v;
        let val = adjusted / w * quant_v;

        let rval = val.round();
        let diff = val - rval;

        // Write error coefficients for pixel-domain loss
        if pixel_domain {
            let err = w * diff;
            let out: &mut [f32; 8] = (&mut error_coeffs[base..base + 8]).try_into().unwrap();
            err.store(out);
        }

        // Entropy accumulation: entropy += sqrt(|rval|) * cost_delta
        let q = rval.abs();
        entropy_acc = q.sqrt().mul_add(cost_delta_v, entropy_acc);

        // nzeros: count non-zero rounded values
        let nz_mask = q.simd_ne(zero);
        nzeros_acc += f32x8::blend(nz_mask, one, zero);

        // Coefficient-domain statistics
        if !pixel_domain {
            let diff_abs = diff.abs();
            info_loss_acc += diff_abs;
            info_loss2_acc = diff_abs.mul_add(diff_abs, info_loss2_acc);

            // q >= 1.5 penalty
            let ge_mask = q.simd_ge(thr_1_5);
            cost2_acc += f32x8::blend(ge_mask, cost2_v, zero);
        }
    }

    // Handle remainder with scalar fallback
    let start = chunks * 8;
    let remainder = entropy_coeffs_scalar(
        &block_c[start..n],
        &block_y[start..n],
        &weights[start..n],
        n - start,
        cmap_factor,
        quant,
        k_cost_delta,
        k_cost2,
        pixel_domain,
        &mut error_coeffs[start..n],
    );

    let mut entropy_sum = entropy_acc.reduce_add() + remainder.entropy_sum;
    if !pixel_domain {
        entropy_sum += cost2_acc.reduce_add();
    }

    EntropyCoeffResult {
        entropy_sum,
        nzeros_sum: nzeros_acc.reduce_add() + remainder.nzeros_sum,
        info_loss_sum: info_loss_acc.reduce_add() + remainder.info_loss_sum,
        info_loss2_sum: info_loss2_acc.reduce_add() + remainder.info_loss2_sum,
    }
}

// ============================================================================
// aarch64 NEON implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_neon(
    token: archmage::NeonToken,
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    n: usize,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
    pixel_domain: bool,
    error_coeffs: &mut [f32],
) -> EntropyCoeffResult {
    use magetypes::simd::f32x4;

    let cmap_v = f32x4::splat(token, cmap_factor);
    let quant_v = f32x4::splat(token, quant);
    let cost_delta_v = f32x4::splat(token, k_cost_delta);
    let cost2_v = f32x4::splat(token, k_cost2);
    let zero = f32x4::zero(token);
    let one = f32x4::splat(token, 1.0);
    let thr_1_5 = f32x4::splat(token, 1.5);

    let mut entropy_acc = f32x4::zero(token);
    let mut nzeros_acc = f32x4::zero(token);
    let mut info_loss_acc = f32x4::zero(token);
    let mut info_loss2_acc = f32x4::zero(token);
    let mut cost2_acc = f32x4::zero(token);

    let chunks = n / 4;
    let simd_n = chunks * 4;
    let block_c_s = &block_c[..simd_n];
    let block_y_s = &block_y[..simd_n];
    let weights_s = &weights[..simd_n];
    for chunk in 0..chunks {
        let base = chunk * 4;

        let bc = f32x4::from_slice(token, &block_c_s[base..]);
        let by_v = f32x4::from_slice(token, &block_y_s[base..]);
        let w = f32x4::from_slice(token, &weights_s[base..]);

        // val = (block_c - block_y * cmap_factor) / weights * quant
        let adjusted = bc - by_v * cmap_v;
        let val = adjusted / w * quant_v;

        let rval = val.round();
        let diff = val - rval;

        if pixel_domain {
            let err = w * diff;
            let out: &mut [f32; 4] = (&mut error_coeffs[base..base + 4]).try_into().unwrap();
            err.store(out);
        }

        let q = rval.abs();
        entropy_acc = q.sqrt().mul_add(cost_delta_v, entropy_acc);

        let nz_mask = q.simd_ne(zero);
        nzeros_acc += f32x4::blend(nz_mask, one, zero);

        if !pixel_domain {
            let diff_abs = diff.abs();
            info_loss_acc += diff_abs;
            info_loss2_acc = diff_abs.mul_add(diff_abs, info_loss2_acc);

            let ge_mask = q.simd_ge(thr_1_5);
            cost2_acc += f32x4::blend(ge_mask, cost2_v, zero);
        }
    }

    // Scalar remainder
    let start = chunks * 4;
    let remainder = entropy_coeffs_scalar(
        &block_c[start..n],
        &block_y[start..n],
        &weights[start..n],
        n - start,
        cmap_factor,
        quant,
        k_cost_delta,
        k_cost2,
        pixel_domain,
        &mut error_coeffs[start..n],
    );

    let mut entropy_sum = entropy_acc.reduce_add() + remainder.entropy_sum;
    if !pixel_domain {
        entropy_sum += cost2_acc.reduce_add();
    }

    EntropyCoeffResult {
        entropy_sum,
        nzeros_sum: nzeros_acc.reduce_add() + remainder.nzeros_sum,
        info_loss_sum: info_loss_acc.reduce_add() + remainder.info_loss_sum,
        info_loss2_sum: info_loss2_acc.reduce_add() + remainder.info_loss2_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    extern crate std;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Verify SIMD matches scalar for pixel-domain mode.
    #[test]
    fn test_entropy_coeffs_pixel_domain() {
        let n = 64;
        let block_c: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 - 20.0) * 0.1).collect();
        let block_y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 - 15.0) * 0.1).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.01 + (i as f32) * 0.005).collect();

        let cmap_factor = 0.15f32;
        let quant = 3.5f32;
        let k_cost_delta = 5.335f32;
        let k_cost2 = 4.463f32;

        // Reference: scalar
        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            n,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            true,
            &mut error_ref,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut error_simd = vec![0.0f32; n];
                let simd_result = entropy_estimate_coeffs(
                    &block_c,
                    &block_y,
                    &weights,
                    n,
                    cmap_factor,
                    quant,
                    k_cost_delta,
                    k_cost2,
                    true,
                    &mut error_simd,
                );
                let rel_eps = 0.005;
                let entropy_rel = (simd_result.entropy_sum - ref_result.entropy_sum).abs()
                    / ref_result.entropy_sum.abs();
                assert!(
                    entropy_rel < rel_eps,
                    "entropy_sum rel_err={entropy_rel:.4} [{perm}]"
                );
                let nz_rel = (simd_result.nzeros_sum - ref_result.nzeros_sum).abs()
                    / ref_result.nzeros_sum.abs().max(1.0);
                assert!(nz_rel < 0.05, "nzeros_sum rel_err={nz_rel:.4} [{perm}]");
                let mut max_err = 0.0f32;
                for i in 0..n {
                    max_err = max_err.max((error_simd[i] - error_ref[i]).abs());
                }
                assert!(
                    max_err < 0.5,
                    "Error coeffs max diff: {max_err:.2e} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    /// Verify SIMD matches scalar for coefficient-domain mode.
    #[test]
    fn test_entropy_coeffs_coeff_domain() {
        let n = 64;
        let block_c: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3 - 40.0) * 0.05).collect();
        let block_y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.9 - 30.0) * 0.05).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.02 + (i as f32) * 0.003).collect();

        let cmap_factor = 0.0f32;
        let quant = 5.0f32;
        let k_cost_delta = 5.335f32;
        let k_cost2 = 4.463f32;

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            n,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            false,
            &mut error_ref,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut error_simd = vec![0.0f32; n];
                let simd_result = entropy_estimate_coeffs(
                    &block_c,
                    &block_y,
                    &weights,
                    n,
                    cmap_factor,
                    quant,
                    k_cost_delta,
                    k_cost2,
                    false,
                    &mut error_simd,
                );
                let rel_eps = 0.005;
                let entropy_rel = (simd_result.entropy_sum - ref_result.entropy_sum).abs()
                    / ref_result.entropy_sum.abs();
                assert!(entropy_rel < rel_eps, "entropy_sum [{perm}]");
                let nz_rel = (simd_result.nzeros_sum - ref_result.nzeros_sum).abs()
                    / ref_result.nzeros_sum.abs().max(1.0);
                assert!(nz_rel < 0.05, "nzeros_sum [{perm}]");
                let il_rel = (simd_result.info_loss_sum - ref_result.info_loss_sum).abs()
                    / ref_result.info_loss_sum.abs().max(1.0);
                assert!(il_rel < rel_eps, "info_loss_sum [{perm}]");
                let il2_rel = (simd_result.info_loss2_sum - ref_result.info_loss2_sum).abs()
                    / ref_result.info_loss2_sum.abs().max(1.0);
                assert!(il2_rel < rel_eps, "info_loss2_sum [{perm}]");
            },
        );
        std::eprintln!("{report}");
    }

    /// Test with non-multiple-of-8 sizes (remainder handling).
    #[test]
    fn test_entropy_coeffs_remainder() {
        let n = 67;
        let block_c: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let block_y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.08 - 2.5).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.01 + (i as f32) * 0.002).collect();

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            n,
            0.2,
            4.0,
            5.335,
            4.463,
            true,
            &mut error_ref,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut error_simd = vec![0.0f32; n];
                let simd_result = entropy_estimate_coeffs(
                    &block_c,
                    &block_y,
                    &weights,
                    n,
                    0.2,
                    4.0,
                    5.335,
                    4.463,
                    true,
                    &mut error_simd,
                );
                let rel_eps = 0.005;
                let entropy_rel = (simd_result.entropy_sum - ref_result.entropy_sum).abs()
                    / ref_result.entropy_sum.abs().max(1.0);
                assert!(entropy_rel < rel_eps, "entropy_sum [{perm}]");
                let nz_rel = (simd_result.nzeros_sum - ref_result.nzeros_sum).abs()
                    / ref_result.nzeros_sum.abs().max(1.0);
                assert!(nz_rel < 0.05, "nzeros_sum [{perm}]");
                let max_err = error_simd
                    .iter()
                    .zip(error_ref.iter())
                    .take(n)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 0.01,
                    "Error coeffs max diff: {max_err:.2e} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    /// Test with large blocks (DCT64x64 = 4096 coefficients).
    #[test]
    fn test_entropy_coeffs_large_block() {
        let n = 4096;
        let block_c: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() * 5.0).collect();
        let block_y: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos() * 4.0).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.005 + (i as f32) * 0.001).collect();

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            n,
            0.1,
            2.0,
            5.335,
            4.463,
            true,
            &mut error_ref,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut error_simd = vec![0.0f32; n];
                let simd_result = entropy_estimate_coeffs(
                    &block_c,
                    &block_y,
                    &weights,
                    n,
                    0.1,
                    2.0,
                    5.335,
                    4.463,
                    true,
                    &mut error_simd,
                );

                // Large block: use relative tolerance
                let rel_eps = 0.005;
                let entropy_rel = (simd_result.entropy_sum - ref_result.entropy_sum).abs()
                    / ref_result.entropy_sum.abs();
                assert!(
                    entropy_rel < rel_eps,
                    "entropy_sum: SIMD={}, ref={}, rel_err={:.4}% [{perm}]",
                    simd_result.entropy_sum,
                    ref_result.entropy_sum,
                    entropy_rel * 100.0
                );

                let max_err = error_simd
                    .iter()
                    .zip(error_ref.iter())
                    .take(n)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 1e-3,
                    "Error coeffs max diff: {:.2e} [{perm}]",
                    max_err
                );
            },
        );
        std::eprintln!("{report}");
    }
}
