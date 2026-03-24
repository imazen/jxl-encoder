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

impl EntropyCoeffResult {
    pub const ZERO: Self = Self {
        entropy_sum: 0.0,
        nzeros_sum: 0.0,
        info_loss_sum: 0.0,
        info_loss2_sum: 0.0,
    };
}

/// Vectorized entropy coefficient processing.
///
/// For each coefficient `i` in 0..n:
///   `val = (block_c[i] - block_y[i] * cmap_factor) * inv_weights[i] * quant`
///   rval = round(val)
///   entropy_sum += sqrt(|rval|) * k_cost_delta
///   nzeros += (rval != 0)
///
/// `inv_weights` contains precomputed reciprocals (1/quant_weight) to replace
/// per-coefficient SIMD division with multiplication.
///
/// In pixel-domain mode: writes `error_coeffs[i] = weights[i] * (val - rval)`
/// In coefficient-domain mode: accumulates info_loss stats and k_cost2 penalty.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn entropy_estimate_coeffs(
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    inv_weights: &[f32],
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
                inv_weights,
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
                inv_weights,
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

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            return entropy_coeffs_wasm128(
                token,
                block_c,
                block_y,
                weights,
                inv_weights,
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
        inv_weights,
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
    inv_weights: &[f32],
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
        let val = (val_in - val_y) * inv_weights[i] * quant;
        let rval = val.round();
        let diff = val - rval;

        if pixel_domain {
            error_coeffs[i] = weights[i] * diff;
        }

        let q = rval.abs();
        entropy_sum = q.sqrt().mul_add(k_cost_delta, entropy_sum);
        if q != 0.0 {
            nzeros_sum += 1.0;
        }

        if !pixel_domain {
            let diff_abs = diff.abs();
            info_loss_sum += diff_abs;
            info_loss2_sum = diff_abs.mul_add(diff_abs, info_loss2_sum);
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
    inv_weights: &[f32],
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
    let simd_n = chunks * 8;
    let block_c_s = &block_c[..simd_n];
    let block_y_s = &block_y[..simd_n];
    let weights_s = &weights[..simd_n];
    let inv_weights_s = &inv_weights[..simd_n];
    let error_coeffs_s = &mut error_coeffs[..simd_n];
    for chunk in 0..chunks {
        let base = chunk * 8;

        let bc = crate::load_f32x8(token, block_c_s, base);
        let by_v = crate::load_f32x8(token, block_y_s, base);
        let w = crate::load_f32x8(token, weights_s, base);
        let iw = crate::load_f32x8(token, inv_weights_s, base);

        // val = (block_c - block_y * cmap_factor) * inv_weights * quant
        let adjusted = bc - by_v * cmap_v;
        let val = adjusted * iw * quant_v;

        let rval = val.round();
        let diff = val - rval;

        // Write error coefficients for pixel-domain loss
        if pixel_domain {
            let err = w * diff;
            crate::store_f32x8(error_coeffs_s, base, err);
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

    // Handle remainder with scalar fallback (skip when n is multiple of 8)
    let start = chunks * 8;
    let remainder = if start < n {
        entropy_coeffs_scalar(
            &block_c[start..n],
            &block_y[start..n],
            &weights[start..n],
            &inv_weights[start..n],
            n - start,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            pixel_domain,
            &mut error_coeffs[start..n],
        )
    } else {
        EntropyCoeffResult::ZERO
    };

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
    inv_weights: &[f32],
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
    let inv_weights_s = &inv_weights[..simd_n];
    for chunk in 0..chunks {
        let base = chunk * 4;

        let bc = f32x4::from_slice(token, &block_c_s[base..]);
        let by_v = f32x4::from_slice(token, &block_y_s[base..]);
        let w = f32x4::from_slice(token, &weights_s[base..]);
        let iw = f32x4::from_slice(token, &inv_weights_s[base..]);

        // val = (block_c - block_y * cmap_factor) * inv_weights * quant
        let adjusted = bc - by_v * cmap_v;
        let val = adjusted * iw * quant_v;

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
        &inv_weights[start..n],
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
// wasm32 SIMD128 implementation
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_wasm128(
    token: archmage::Wasm128Token,
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    inv_weights: &[f32],
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
    let inv_weights_s = &inv_weights[..simd_n];
    for chunk in 0..chunks {
        let base = chunk * 4;

        let bc = f32x4::from_slice(token, &block_c_s[base..]);
        let by_v = f32x4::from_slice(token, &block_y_s[base..]);
        let w = f32x4::from_slice(token, &weights_s[base..]);
        let iw = f32x4::from_slice(token, &inv_weights_s[base..]);

        // val = (block_c - block_y * cmap_factor) * inv_weights * quant
        let adjusted = bc - by_v * cmap_v;
        let val = adjusted * iw * quant_v;

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
        &inv_weights[start..n],
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
// Shannon entropy computation (P6: histogram entropy for clustering)
// ============================================================================

// fast_log2f polynomial coefficients (shared with mask1x1, adaptive_quant).
// Used by arch-gated Shannon entropy functions and scalar fallback.
const LOG2_P0: f32 = -1.850_383_3e-6;
const LOG2_P1: f32 = 1.428_716;
const LOG2_P2: f32 = 0.742_458_7;
const LOG2_Q0: f32 = 0.990_328_14;
const LOG2_Q1: f32 = 1.009_671_9;
const LOG2_Q2: f32 = 0.174_093_43;

/// Fast log2 approximation. Max relative error ~3e-7. Input must be > 0.
///
/// Uses integer bit manipulation on f32 with a Padé approximant for the
/// fractional part. Matches libjxl's `FastLog2f` from `fast_math-inl.h`.
#[inline(always)]
pub fn fast_log2f(x: f32) -> f32 {
    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits.wrapping_sub(0x3f2a_aaab_u32 as i32);
    let exp_shifted = exp_bits >> 23;
    let mantissa = f32::from_bits((x_bits.wrapping_sub(exp_shifted << 23)) as u32);
    let exp_val = exp_shifted as f32;
    let frac = mantissa - 1.0;
    let num = LOG2_P0 + frac * (LOG2_P1 + frac * LOG2_P2);
    let den = LOG2_Q0 + frac * (LOG2_Q1 + frac * LOG2_Q2);
    num / den + exp_val
}

/// Fast base-2 exponentiation. Max relative error ~3e-7.
///
/// Matches libjxl's `FastPow2f` from `fast_math-inl.h` (line 72).
/// Uses integer bit manipulation for the integer exponent part and a (3,3)
/// rational polynomial for the fractional part.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn fast_pow2f(x: f32) -> f32 {
    let floorx = x.floor();
    // Integer part → IEEE 754 exponent via bit shift
    let exp = f32::from_bits(((floorx as i32 + 127) << 23) as u32);
    let frac = x - floorx;
    // (3,3) rational polynomial for 2^frac, frac in [0, 1)
    // Coefficients from libjxl fast_math-inl.h — must match exactly.
    // Numerator: Horner form
    let mut num = frac + 1.01749063e+01;
    num = num * frac + 4.88687798e+01;
    num = num * frac + 9.85506591e+01;
    num *= exp;
    // Denominator: Horner form
    let mut den = frac * 2.10242958e-01 + (-2.22328856e-02);
    den = den * frac + (-1.94414990e+01);
    den = den * frac + 9.85506633e+01;
    num / den
}

/// Fast power function: `base^exponent`. Max relative error ~3e-5.
///
/// Matches libjxl's `FastPowf` from `fast_math-inl.h` (line 90).
/// Computes `2^(log2(base) * exponent)` using [`fast_log2f`] and [`fast_pow2f`].
/// Input `base` must be > 0.
#[inline(always)]
pub fn fast_powf(base: f32, exponent: f32) -> f32 {
    fast_pow2f(fast_log2f(base) * exponent)
}

/// Compute Shannon entropy of a histogram: -sum(count * log2(count / total)).
///
/// Returns total entropy in bits. Excludes zero counts and the case where
/// a single count equals total (entropy contribution = 0).
///
/// Uses fast_log2f approximation (~3e-7 relative error per log2 call).
#[inline]
pub fn shannon_entropy_bits(counts: &[i32], total_count: usize) -> f32 {
    if total_count == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return shannon_entropy_avx2(token, counts, total_count);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return shannon_entropy_neon(token, counts, total_count);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            return shannon_entropy_wasm128(token, counts, total_count);
        }
    }

    shannon_entropy_scalar(counts, total_count)
}

/// Scalar Shannon entropy using fast_log2f.
#[inline]
pub fn shannon_entropy_scalar(counts: &[i32], total_count: usize) -> f32 {
    let inv_total = 1.0 / total_count as f32;
    let total_f = total_count as f32;
    let mut entropy = 0.0f32;

    for &count in counts {
        if count > 0 {
            let c = count as f32;
            if c != total_f {
                entropy -= c * fast_log2f(c * inv_total);
            }
        }
    }

    entropy
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
pub fn shannon_entropy_avx2(
    token: archmage::X64V3Token,
    counts: &[i32],
    total_count: usize,
) -> f32 {
    use magetypes::simd::{f32x8, i32x8};

    let inv_total_v = f32x8::splat(token, 1.0 / total_count as f32);
    let total_f_v = f32x8::splat(token, total_count as f32);
    let zero_f = f32x8::zero(token);
    let mut acc = f32x8::zero(token);

    // fast_log2f constants
    let offset = i32x8::splat(token, 0x3f2a_aaab_u32 as i32);
    let one = f32x8::splat(token, 1.0);
    let p0 = f32x8::splat(token, LOG2_P0);
    let p1 = f32x8::splat(token, LOG2_P1);
    let p2 = f32x8::splat(token, LOG2_P2);
    let q0 = f32x8::splat(token, LOG2_Q0);
    let q1 = f32x8::splat(token, LOG2_Q1);
    let q2 = f32x8::splat(token, LOG2_Q2);

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn fast_log2f_x8(
        x: f32x8,
        offset: i32x8,
        p0: f32x8,
        p1: f32x8,
        p2: f32x8,
        q0: f32x8,
        q1: f32x8,
        q2: f32x8,
        one: f32x8,
    ) -> f32x8 {
        let x_bits: i32x8 = x.bitcast_i32x8();
        let exp_bits = x_bits - offset;
        let exp_shifted = exp_bits.shr_arithmetic::<23>();
        let mantissa_bits = x_bits - exp_shifted.shl::<23>();
        let mantissa = mantissa_bits.bitcast_f32x8();
        let exp_val = exp_shifted.to_f32x8();
        let frac = mantissa - one;
        let num = frac.mul_add(p2, p1).mul_add(frac, p0);
        let den = frac.mul_add(q2, q1).mul_add(frac, q0);
        num / den + exp_val
    }

    let chunks = counts.len() / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let c_i = i32x8::from_slice(token, &counts[base..]);
        let c_f = c_i.to_f32x8();

        // Compare in f32 space (blend needs f32 masks)
        let nonzero_mask = c_f.simd_gt(zero_f);
        let not_total_mask = c_f.simd_ne(total_f_v);
        // Combine masks: multiply 1.0/0.0 selects
        let nz_float = f32x8::blend(nonzero_mask, one, zero_f);
        let nt_float = f32x8::blend(not_total_mask, one, zero_f);
        let valid_mask = nz_float * nt_float; // 1.0 where valid, 0.0 where not

        // For log2, we need count > 0 to avoid log2(0). Use max(count, 1) for safe input.
        let safe_c = f32x8::blend(nonzero_mask, c_f, one);
        let prob = safe_c * inv_total_v;
        let log2_prob = fast_log2f_x8(prob, offset, p0, p1, p2, q0, q1, q2, one);

        // contribution = -count * log2(count/total), masked to 0 where invalid
        let contribution = c_f * log2_prob * valid_mask;
        acc -= contribution;
    }

    // Handle remainder (scalar)
    let mut scalar_sum = 0.0f32;
    let inv_total = 1.0 / total_count as f32;
    let total_f = total_count as f32;
    for &count in &counts[chunks * 8..] {
        if count > 0 {
            let c = count as f32;
            if c != total_f {
                scalar_sum -= c * fast_log2f(c * inv_total);
            }
        }
    }

    acc.reduce_add() + scalar_sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
pub fn shannon_entropy_neon(token: archmage::NeonToken, counts: &[i32], total_count: usize) -> f32 {
    use magetypes::simd::{f32x4, i32x4};

    let inv_total_v = f32x4::splat(token, 1.0 / total_count as f32);
    let total_f_v = f32x4::splat(token, total_count as f32);
    let zero_f = f32x4::zero(token);
    let mut acc = f32x4::zero(token);

    let offset = i32x4::splat(token, 0x3f2a_aaab_u32 as i32);
    let one = f32x4::splat(token, 1.0);
    let p0 = f32x4::splat(token, LOG2_P0);
    let p1 = f32x4::splat(token, LOG2_P1);
    let p2 = f32x4::splat(token, LOG2_P2);
    let q0 = f32x4::splat(token, LOG2_Q0);
    let q1 = f32x4::splat(token, LOG2_Q1);
    let q2 = f32x4::splat(token, LOG2_Q2);

    #[inline(always)]
    fn fast_log2f_x4(
        x: f32x4,
        offset: i32x4,
        p0: f32x4,
        p1: f32x4,
        p2: f32x4,
        q0: f32x4,
        q1: f32x4,
        q2: f32x4,
        one: f32x4,
    ) -> f32x4 {
        let x_bits: i32x4 = x.bitcast_i32x4();
        let exp_bits = x_bits - offset;
        let exp_shifted = exp_bits.shr_arithmetic::<23>();
        let mantissa_bits = x_bits - exp_shifted.shl::<23>();
        let mantissa = mantissa_bits.bitcast_f32x4();
        let exp_val = exp_shifted.to_f32x4();
        let frac = mantissa - one;
        let num = frac.mul_add(p2, p1).mul_add(frac, p0);
        let den = frac.mul_add(q2, q1).mul_add(frac, q0);
        num / den + exp_val
    }

    let chunks = counts.len() / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let c_i = i32x4::from_slice(token, &counts[base..]);
        let c_f = c_i.to_f32x4();

        // Compare in f32 space (blend needs f32 masks)
        let nonzero_mask = c_f.simd_gt(zero_f);
        let not_total_mask = c_f.simd_ne(total_f_v);
        let nz_float = f32x4::blend(nonzero_mask, one, zero_f);
        let nt_float = f32x4::blend(not_total_mask, one, zero_f);
        let valid_mask = nz_float * nt_float;

        let safe_c = f32x4::blend(nonzero_mask, c_f, one);
        let prob = safe_c * inv_total_v;
        let log2_prob = fast_log2f_x4(prob, offset, p0, p1, p2, q0, q1, q2, one);

        let contribution = c_f * log2_prob * valid_mask;
        acc -= contribution;
    }

    let mut scalar_sum = 0.0f32;
    let inv_total = 1.0 / total_count as f32;
    let total_f = total_count as f32;
    for &count in &counts[chunks * 4..] {
        if count > 0 {
            let c = count as f32;
            if c != total_f {
                scalar_sum -= c * fast_log2f(c * inv_total);
            }
        }
    }

    acc.reduce_add() + scalar_sum
}

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
pub fn shannon_entropy_wasm128(
    token: archmage::Wasm128Token,
    counts: &[i32],
    total_count: usize,
) -> f32 {
    use magetypes::simd::{f32x4, i32x4};

    let inv_total_v = f32x4::splat(token, 1.0 / total_count as f32);
    let total_f_v = f32x4::splat(token, total_count as f32);
    let zero_f = f32x4::zero(token);
    let mut acc = f32x4::zero(token);

    let offset = i32x4::splat(token, 0x3f2a_aaab_u32 as i32);
    let one = f32x4::splat(token, 1.0);
    let p0 = f32x4::splat(token, LOG2_P0);
    let p1 = f32x4::splat(token, LOG2_P1);
    let p2 = f32x4::splat(token, LOG2_P2);
    let q0 = f32x4::splat(token, LOG2_Q0);
    let q1 = f32x4::splat(token, LOG2_Q1);
    let q2 = f32x4::splat(token, LOG2_Q2);

    #[inline(always)]
    fn fast_log2f_x4(
        x: f32x4,
        offset: i32x4,
        p0: f32x4,
        p1: f32x4,
        p2: f32x4,
        q0: f32x4,
        q1: f32x4,
        q2: f32x4,
        one: f32x4,
    ) -> f32x4 {
        let x_bits: i32x4 = x.bitcast_i32x4();
        let exp_bits = x_bits - offset;
        let exp_shifted = exp_bits.shr_arithmetic::<23>();
        let mantissa_bits = x_bits - exp_shifted.shl::<23>();
        let mantissa = mantissa_bits.bitcast_f32x4();
        let exp_val = exp_shifted.to_f32x4();
        let frac = mantissa - one;
        let num = frac.mul_add(p2, p1).mul_add(frac, p0);
        let den = frac.mul_add(q2, q1).mul_add(frac, q0);
        num / den + exp_val
    }

    let chunks = counts.len() / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let c_i = i32x4::from_slice(token, &counts[base..]);
        let c_f = c_i.to_f32x4();

        // Compare in f32 space (blend needs f32 masks)
        let nonzero_mask = c_f.simd_gt(zero_f);
        let not_total_mask = c_f.simd_ne(total_f_v);
        let nz_float = f32x4::blend(nonzero_mask, one, zero_f);
        let nt_float = f32x4::blend(not_total_mask, one, zero_f);
        let valid_mask = nz_float * nt_float;

        let safe_c = f32x4::blend(nonzero_mask, c_f, one);
        let prob = safe_c * inv_total_v;
        let log2_prob = fast_log2f_x4(prob, offset, p0, p1, p2, q0, q1, q2, one);

        let contribution = c_f * log2_prob * valid_mask;
        acc -= contribution;
    }

    let mut scalar_sum = 0.0f32;
    let inv_total = 1.0 / total_count as f32;
    let total_f = total_count as f32;
    for &count in &counts[chunks * 4..] {
        if count > 0 {
            let c = count as f32;
            if c != total_f {
                scalar_sum -= c * fast_log2f(c * inv_total);
            }
        }
    }

    acc.reduce_add() + scalar_sum
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
        let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

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
            &inv_weights,
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
                    &inv_weights,
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
        let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

        let cmap_factor = 0.0f32;
        let quant = 5.0f32;
        let k_cost_delta = 5.335f32;
        let k_cost2 = 4.463f32;

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
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
                    &inv_weights,
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
        let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
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
                    &inv_weights,
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
        let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
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
                    &inv_weights,
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

    // =====================================================================
    // Shannon entropy tests
    // =====================================================================

    /// Reference Shannon entropy using f32::log2 (not fast_log2f).
    fn reference_shannon_entropy(counts: &[i32], total_count: usize) -> f32 {
        if total_count == 0 {
            return 0.0;
        }
        let inv_total = 1.0 / total_count as f32;
        let total_f = total_count as f32;
        let mut entropy = 0.0f32;
        for &count in counts {
            if count > 0 {
                let c = count as f32;
                if c != total_f {
                    entropy -= c * (c * inv_total).log2();
                }
            }
        }
        entropy
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution: entropy = n * log2(n) bits total
        let counts = [100i32, 100, 100, 100, 0, 0, 0, 0];
        let total = 400;
        let ref_ent = reference_shannon_entropy(&counts, total);
        let simd_ent = shannon_entropy_bits(&counts, total);
        let scalar_ent = shannon_entropy_scalar(&counts, total);

        // Expected: 400 * log2(4) = 800
        assert!((ref_ent - 800.0).abs() < 0.1, "ref = {ref_ent}");
        assert!(
            (simd_ent - ref_ent).abs() < 0.5,
            "simd={simd_ent} ref={ref_ent}"
        );
        assert!(
            (scalar_ent - ref_ent).abs() < 0.5,
            "scalar={scalar_ent} ref={ref_ent}"
        );
    }

    #[test]
    fn test_shannon_entropy_single_symbol() {
        // All counts in one symbol: entropy = 0
        let counts = [1000i32, 0, 0, 0, 0, 0, 0, 0];
        let total = 1000;
        let ent = shannon_entropy_bits(&counts, total);
        assert!(ent.abs() < 0.01, "entropy should be 0, got {ent}");
    }

    #[test]
    fn test_shannon_entropy_realistic_histogram() {
        // Realistic distribution like AC coefficient magnitudes
        let mut counts = alloc::vec![0i32; 64];
        counts[0] = 5000; // lots of zeros (but treated as symbol 0)
        counts[1] = 2000;
        counts[2] = 1000;
        counts[3] = 500;
        counts[4] = 200;
        counts[5] = 100;
        counts[6] = 50;
        counts[7] = 20;
        let total: usize = counts.iter().map(|&c| c as usize).sum();

        let ref_ent = reference_shannon_entropy(&counts, total);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let simd_ent = shannon_entropy_bits(&counts, total);
                let rel_err = (simd_ent - ref_ent).abs() / ref_ent.abs().max(1.0);
                assert!(
                    rel_err < 0.001,
                    "Shannon entropy: simd={simd_ent}, ref={ref_ent}, rel_err={rel_err:.4} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_shannon_entropy_large_alphabet() {
        // Large alphabet (256 symbols) with Zipf-like distribution
        let mut counts = alloc::vec![0i32; 256];
        let mut total = 0usize;
        for (i, count) in counts.iter_mut().enumerate() {
            *count = 10000 / (i as i32 + 1);
            total += *count as usize;
        }

        let ref_ent = reference_shannon_entropy(&counts, total);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let simd_ent = shannon_entropy_bits(&counts, total);
                let rel_err = (simd_ent - ref_ent).abs() / ref_ent.abs().max(1.0);
                assert!(
                    rel_err < 0.001,
                    "Large alphabet: simd={simd_ent}, ref={ref_ent}, rel_err={rel_err:.4} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_shannon_entropy_empty() {
        let counts = [0i32; 8];
        let ent = shannon_entropy_bits(&counts, 0);
        assert_eq!(ent, 0.0);
    }

    #[test]
    fn test_fast_pow2f_accuracy() {
        // Test exact powers of 2
        assert!((fast_pow2f(0.0) - 1.0).abs() < 1e-5);
        assert!((fast_pow2f(1.0) - 2.0).abs() < 1e-4);
        assert!((fast_pow2f(3.0) - 8.0).abs() < 1e-3);
        assert!((fast_pow2f(-1.0) - 0.5).abs() < 1e-5);

        // Test fractional exponents
        let val = fast_pow2f(0.5);
        let expected = core::f32::consts::SQRT_2;
        assert!(
            (val - expected).abs() / expected < 5e-7,
            "2^0.5: got {val}, expected {expected}"
        );
    }

    #[test]
    fn test_fast_powf_accuracy() {
        // Test basic powers
        let val = fast_powf(2.0, 3.0);
        assert!(
            (val - 8.0).abs() / 8.0 < 5e-5,
            "2^3: got {val}, expected 8.0"
        );

        // Test sRGB TF: (0.5)^2.4
        let base = 0.5f32;
        let exact = base.powf(2.4);
        let fast = fast_powf(base, 2.4);
        assert!(
            (fast - exact).abs() / exact < 5e-5,
            "0.5^2.4: got {fast}, expected {exact}"
        );

        // Test ratio^K_POW (the compute_scaled_constants case)
        let ratio = 1.5f32;
        let exact = ratio.powf(0.337);
        let fast = fast_powf(ratio, 0.337);
        assert!(
            (fast - exact).abs() / exact < 5e-5,
            "1.5^0.337: got {fast}, expected {exact}"
        );
    }
}
