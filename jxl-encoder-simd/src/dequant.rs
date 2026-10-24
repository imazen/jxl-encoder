// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated AC coefficient dequantization for DCT8 blocks.
//!
//! The reconstruct_xyb inner dequant loop is ~7% of encoder CPU. For DCT8
//! (the most common strategy), we process 64 coefficients in one pass:
//! dequantize + adjust_quant_bias + CfL restore.

/// Dequantize a DCT8 block and apply CfL (chroma-from-luma) in one pass.
///
/// For each channel `c` and coefficient `i` (except DC at index 0):
///   `biased = adjust_quant_bias(quant_ac[c][i], c)`
///   `dequant[c][i] = biased * weights[c][i] / (qac * qm_mul[c])`
///
/// Then CfL restore (AC positions only):
///   `dequant[X][i] += x_factor * dequant[Y][i]`
///   `dequant[B][i] += b_factor * dequant[Y][i]`
///
/// DC (index 0) is left as-is in the output (caller must restore LLF from DC).
///
/// # Parameters
/// - `quant_ac_x/y/b`: Quantized AC coefficients per channel, [i32; 64]
/// - `weights_x/y/b`: Dequantization weights per channel, [f32; 64]
/// - `qac_qm`: Per-channel `qac * qm_mul` values [x, y, b]
/// - `x_factor`: CfL ytox ratio for this tile
/// - `b_factor`: CfL ytob ratio for this tile
/// - `output_x/y/b`: Output dequantized coefficients per channel, [f32; 64]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dequant_block_dct8(
    quant_ac_x: &[i32; 64],
    quant_ac_y: &[i32; 64],
    quant_ac_b: &[i32; 64],
    weights_x: &[f32; 64],
    weights_y: &[f32; 64],
    weights_b: &[f32; 64],
    qac_qm: [f32; 3], // [x, y, b]
    x_factor: f32,
    b_factor: f32,
    output_x: &mut [f32; 64],
    output_y: &mut [f32; 64],
    output_b: &mut [f32; 64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dequant_dct8_avx2(
                token, quant_ac_x, quant_ac_y, quant_ac_b, weights_x, weights_y, weights_b, qac_qm,
                x_factor, b_factor, output_x, output_y, output_b,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            dequant_dct8_neon(
                token, quant_ac_x, quant_ac_y, quant_ac_b, weights_x, weights_y, weights_b, qac_qm,
                x_factor, b_factor, output_x, output_y, output_b,
            );
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            dequant_dct8_wasm128(
                token, quant_ac_x, quant_ac_y, quant_ac_b, weights_x, weights_y, weights_b, qac_qm,
                x_factor, b_factor, output_x, output_y, output_b,
            );
            return;
        }
    }

    dequant_dct8_scalar(
        quant_ac_x, quant_ac_y, quant_ac_b, weights_x, weights_y, weights_b, qac_qm, x_factor,
        b_factor, output_x, output_y, output_b,
    );
}

// AdjustQuantBias constants (pre-computed to avoid clippy lossy_float_literal)
const BIAS_X: f32 = 0.945_349_93; // 1.0 - 0.054_650_073
const BIAS_Y: f32 = 0.929_945_5; // 1.0 - 0.070_054_499
const BIAS_B: f32 = 0.950_064_9; // 1.0 - 0.049_935_103
const BIAS_RECIP: f32 = 0.145;

#[inline(always)]
fn adjust_quant_bias_scalar(q_int: i32, channel_bias: f32) -> f32 {
    if q_int == 0 {
        return 0.0;
    }
    let q = q_int as f32;
    if q.abs() < 1.125 {
        q.signum() * channel_bias
    } else {
        q - BIAS_RECIP / q
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dequant_dct8_scalar(
    quant_ac_x: &[i32; 64],
    quant_ac_y: &[i32; 64],
    quant_ac_b: &[i32; 64],
    weights_x: &[f32; 64],
    weights_y: &[f32; 64],
    weights_b: &[f32; 64],
    qac_qm: [f32; 3],
    x_factor: f32,
    b_factor: f32,
    output_x: &mut [f32; 64],
    output_y: &mut [f32; 64],
    output_b: &mut [f32; 64],
) {
    let inv_qac_x = 1.0 / qac_qm[0];
    let inv_qac_y = 1.0 / qac_qm[1];
    let inv_qac_b = 1.0 / qac_qm[2];

    // DC (index 0) must be zeroed — it's restored from DC separately
    output_x[0] = 0.0;
    output_y[0] = 0.0;
    output_b[0] = 0.0;

    for i in 1..64 {
        // Dequantize each channel
        let biased_x = adjust_quant_bias_scalar(quant_ac_x[i], BIAS_X);
        let biased_y = adjust_quant_bias_scalar(quant_ac_y[i], BIAS_Y);
        let biased_b = adjust_quant_bias_scalar(quant_ac_b[i], BIAS_B);

        let dq_y = biased_y * weights_y[i] * inv_qac_y;
        output_y[i] = dq_y;

        // CfL restore: X += ytox * Y, B += ytob * Y
        output_x[i] = biased_x * weights_x[i] * inv_qac_x + x_factor * dq_y;
        output_b[i] = biased_b * weights_b[i] * inv_qac_b + b_factor * dq_y;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn dequant_dct8_avx2(
    token: archmage::X64V3Token,
    quant_ac_x: &[i32; 64],
    quant_ac_y: &[i32; 64],
    quant_ac_b: &[i32; 64],
    weights_x: &[f32; 64],
    weights_y: &[f32; 64],
    weights_b: &[f32; 64],
    qac_qm: [f32; 3],
    x_factor: f32,
    b_factor: f32,
    output_x: &mut [f32; 64],
    output_y: &mut [f32; 64],
    output_b: &mut [f32; 64],
) {
    use magetypes::simd::{f32x8, i32x8};

    let inv_qac_x_v = f32x8::splat(token, 1.0 / qac_qm[0]);
    let inv_qac_y_v = f32x8::splat(token, 1.0 / qac_qm[1]);
    let inv_qac_b_v = f32x8::splat(token, 1.0 / qac_qm[2]);
    let x_factor_v = f32x8::splat(token, x_factor);
    let b_factor_v = f32x8::splat(token, b_factor);
    let zero_f = f32x8::zero(token);
    let zero_i = i32x8::zero(token);
    let one_f = f32x8::splat(token, 1.0);
    let neg_one_f = f32x8::splat(token, -1.0);
    let threshold = f32x8::splat(token, 1.125);
    let bias_recip_v = f32x8::splat(token, BIAS_RECIP);
    let bias_x_v = f32x8::splat(token, BIAS_X);
    let bias_y_v = f32x8::splat(token, BIAS_Y);
    let bias_b_v = f32x8::splat(token, BIAS_B);

    // Process 8 chunks of 8 coefficients
    for chunk in 0..8 {
        let base = chunk * 8;

        // --- Channel Y ---
        let q_i_y = i32x8::from_slice(token, &quant_ac_y[base..]);
        let dq_y = dequant_8_avx2(
            token,
            q_i_y,
            bias_y_v,
            bias_recip_v,
            threshold,
            zero_i,
            zero_f,
            one_f,
            neg_one_f,
            &weights_y[base..],
            inv_qac_y_v,
        );
        dq_y.store((&mut output_y[base..base + 8]).try_into().unwrap());

        // --- Channel X + CfL ---
        let q_i_x = i32x8::from_slice(token, &quant_ac_x[base..]);
        let dq_x_raw = dequant_8_avx2(
            token,
            q_i_x,
            bias_x_v,
            bias_recip_v,
            threshold,
            zero_i,
            zero_f,
            one_f,
            neg_one_f,
            &weights_x[base..],
            inv_qac_x_v,
        );
        let dq_x = dq_x_raw + x_factor_v * dq_y;
        dq_x.store((&mut output_x[base..base + 8]).try_into().unwrap());

        // --- Channel B + CfL ---
        let q_i_b = i32x8::from_slice(token, &quant_ac_b[base..]);
        let dq_b_raw = dequant_8_avx2(
            token,
            q_i_b,
            bias_b_v,
            bias_recip_v,
            threshold,
            zero_i,
            zero_f,
            one_f,
            neg_one_f,
            &weights_b[base..],
            inv_qac_b_v,
        );
        let dq_b = dq_b_raw + b_factor_v * dq_y;
        dq_b.store((&mut output_b[base..base + 8]).try_into().unwrap());
    }

    // DC (index 0) must be zeroed — it's restored from DC separately
    output_x[0] = 0.0;
    output_y[0] = 0.0;
    output_b[0] = 0.0;
}

/// Dequantize 8 coefficients with adjust_quant_bias, branchless SIMD.
///
/// For each element:
///   if q == 0: result = 0
///   elif |q| == 1: result = sign(q) * channel_bias
///   else: result = q - 0.145/q
///   output = result * weight / (qac * qm_mul)
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dequant_8_avx2(
    token: archmage::X64V3Token,
    q_int: magetypes::simd::i32x8,
    channel_bias: magetypes::simd::f32x8,
    bias_recip: magetypes::simd::f32x8,
    threshold: magetypes::simd::f32x8,
    _zero_i: magetypes::simd::i32x8,
    zero_f: magetypes::simd::f32x8,
    one_f: magetypes::simd::f32x8,
    neg_one_f: magetypes::simd::f32x8,
    weights: &[f32],
    inv_qac_qm: magetypes::simd::f32x8,
) -> magetypes::simd::f32x8 {
    use magetypes::simd::f32x8;

    // Convert q to f32
    let q_f = q_int.to_f32x8();
    let abs_q = q_f.abs();

    // Compute sign: 1.0 for positive, -1.0 for negative (0 handled by zero_mask)
    let sign = f32x8::blend(q_f.simd_ge(zero_f), one_f, neg_one_f);

    // Case 1: |q| < 1.125 (i.e., q == ±1 for integer inputs)
    // Result: sign * channel_bias
    let case_one = sign * channel_bias;

    // Case 2: |q| >= 1.125 (i.e., |q| >= 2 for integer inputs)
    // Result: q - 0.145 / q
    let case_large = q_f - bias_recip / q_f;

    // Select: if |q| < 1.125 use case_one, else case_large
    let is_large = abs_q.simd_ge(threshold);
    let biased = f32x8::blend(is_large, case_large, case_one);

    // Zero out where q == 0 (compare in f32 space since blend needs f32 mask)
    let is_nonzero = abs_q.simd_ge(f32x8::splat(token, 0.5)); // integers: |q|>=0.5 means nonzero
    let biased = f32x8::blend(is_nonzero, biased, zero_f);

    // Multiply by dequant weight and inverse qac_qm
    let w = f32x8::from_slice(token, weights);
    biased * w * inv_qac_qm
}

// --- aarch64 NEON implementation ---

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn dequant_dct8_neon(
    token: archmage::NeonToken,
    quant_ac_x: &[i32; 64],
    quant_ac_y: &[i32; 64],
    quant_ac_b: &[i32; 64],
    weights_x: &[f32; 64],
    weights_y: &[f32; 64],
    weights_b: &[f32; 64],
    qac_qm: [f32; 3],
    x_factor: f32,
    b_factor: f32,
    output_x: &mut [f32; 64],
    output_y: &mut [f32; 64],
    output_b: &mut [f32; 64],
) {
    use magetypes::simd::{f32x4, i32x4};

    let inv_qac_x_v = f32x4::splat(token, 1.0 / qac_qm[0]);
    let inv_qac_y_v = f32x4::splat(token, 1.0 / qac_qm[1]);
    let inv_qac_b_v = f32x4::splat(token, 1.0 / qac_qm[2]);
    let x_factor_v = f32x4::splat(token, x_factor);
    let b_factor_v = f32x4::splat(token, b_factor);
    let zero_f = f32x4::zero(token);
    let one_f = f32x4::splat(token, 1.0);
    let neg_one_f = f32x4::splat(token, -1.0);
    let threshold = f32x4::splat(token, 1.125);
    let bias_recip_v = f32x4::splat(token, BIAS_RECIP);
    let bias_x_v = f32x4::splat(token, BIAS_X);
    let bias_y_v = f32x4::splat(token, BIAS_Y);
    let bias_b_v = f32x4::splat(token, BIAS_B);
    let half_v = f32x4::splat(token, 0.5);

    // Process 16 chunks of 4 coefficients
    for chunk in 0..16 {
        let base = chunk * 4;

        // Channel Y
        let q_i_y = i32x4::from_slice(token, &quant_ac_y[base..]);
        let dq_y = neon_dequant_4(
            token,
            q_i_y,
            bias_y_v,
            bias_recip_v,
            threshold,
            zero_f,
            one_f,
            neg_one_f,
            half_v,
            &weights_y[base..],
            inv_qac_y_v,
        );
        dq_y.store((&mut output_y[base..base + 4]).try_into().unwrap());

        // Channel X + CfL
        let q_i_x = i32x4::from_slice(token, &quant_ac_x[base..]);
        let dq_x_raw = neon_dequant_4(
            token,
            q_i_x,
            bias_x_v,
            bias_recip_v,
            threshold,
            zero_f,
            one_f,
            neg_one_f,
            half_v,
            &weights_x[base..],
            inv_qac_x_v,
        );
        let dq_x = dq_x_raw + x_factor_v * dq_y;
        dq_x.store((&mut output_x[base..base + 4]).try_into().unwrap());

        // Channel B + CfL
        let q_i_b = i32x4::from_slice(token, &quant_ac_b[base..]);
        let dq_b_raw = neon_dequant_4(
            token,
            q_i_b,
            bias_b_v,
            bias_recip_v,
            threshold,
            zero_f,
            one_f,
            neg_one_f,
            half_v,
            &weights_b[base..],
            inv_qac_b_v,
        );
        let dq_b = dq_b_raw + b_factor_v * dq_y;
        dq_b.store((&mut output_b[base..base + 4]).try_into().unwrap());
    }

    output_x[0] = 0.0;
    output_y[0] = 0.0;
    output_b[0] = 0.0;
}

/// Dequantize 4 coefficients with adjust_quant_bias, branchless NEON.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
#[allow(clippy::too_many_arguments)]
fn neon_dequant_4(
    token: archmage::NeonToken,
    q_int: magetypes::simd::i32x4,
    channel_bias: magetypes::simd::f32x4,
    bias_recip: magetypes::simd::f32x4,
    threshold: magetypes::simd::f32x4,
    zero_f: magetypes::simd::f32x4,
    one_f: magetypes::simd::f32x4,
    neg_one_f: magetypes::simd::f32x4,
    half_v: magetypes::simd::f32x4,
    weights: &[f32],
    inv_qac_qm: magetypes::simd::f32x4,
) -> magetypes::simd::f32x4 {
    use magetypes::simd::f32x4;

    let q_f = f32x4::from_i32x4(token, q_int);
    let abs_q = q_f.abs();

    let sign = f32x4::blend(q_f.simd_ge(zero_f), one_f, neg_one_f);

    // Case 1: |q| < 1.125 → sign * channel_bias
    let case_one = sign * channel_bias;

    // Case 2: |q| >= 1.125 → q - 0.145/q
    let case_large = q_f - bias_recip / q_f;

    let is_large = abs_q.simd_ge(threshold);
    let biased = f32x4::blend(is_large, case_large, case_one);

    // Zero out where q == 0
    let is_nonzero = abs_q.simd_ge(half_v);
    let biased = f32x4::blend(is_nonzero, biased, zero_f);

    let w = f32x4::from_slice(token, weights);
    biased * w * inv_qac_qm
}

// --- wasm32 SIMD128 implementation ---

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn dequant_dct8_wasm128(
    token: archmage::Wasm128Token,
    quant_ac_x: &[i32; 64],
    quant_ac_y: &[i32; 64],
    quant_ac_b: &[i32; 64],
    weights_x: &[f32; 64],
    weights_y: &[f32; 64],
    weights_b: &[f32; 64],
    qac_qm: [f32; 3],
    x_factor: f32,
    b_factor: f32,
    output_x: &mut [f32; 64],
    output_y: &mut [f32; 64],
    output_b: &mut [f32; 64],
) {
    use magetypes::simd::{f32x4, i32x4};

    let inv_qac_x_v = f32x4::splat(token, 1.0 / qac_qm[0]);
    let inv_qac_y_v = f32x4::splat(token, 1.0 / qac_qm[1]);
    let inv_qac_b_v = f32x4::splat(token, 1.0 / qac_qm[2]);
    let x_factor_v = f32x4::splat(token, x_factor);
    let b_factor_v = f32x4::splat(token, b_factor);
    let zero_f = f32x4::zero(token);
    let one_f = f32x4::splat(token, 1.0);
    let neg_one_f = f32x4::splat(token, -1.0);
    let threshold = f32x4::splat(token, 1.125);
    let bias_recip_v = f32x4::splat(token, BIAS_RECIP);
    let bias_x_v = f32x4::splat(token, BIAS_X);
    let bias_y_v = f32x4::splat(token, BIAS_Y);
    let bias_b_v = f32x4::splat(token, BIAS_B);
    let half_v = f32x4::splat(token, 0.5);

    // Process 16 chunks of 4 coefficients
    for chunk in 0..16 {
        let base = chunk * 4;

        // Channel Y
        let q_i_y = i32x4::from_slice(token, &quant_ac_y[base..]);
        let dq_y = wasm128_dequant_4(
            token,
            q_i_y,
            bias_y_v,
            bias_recip_v,
            threshold,
            zero_f,
            one_f,
            neg_one_f,
            half_v,
            &weights_y[base..],
            inv_qac_y_v,
        );
        dq_y.store((&mut output_y[base..base + 4]).try_into().unwrap());

        // Channel X + CfL
        let q_i_x = i32x4::from_slice(token, &quant_ac_x[base..]);
        let dq_x_raw = wasm128_dequant_4(
            token,
            q_i_x,
            bias_x_v,
            bias_recip_v,
            threshold,
            zero_f,
            one_f,
            neg_one_f,
            half_v,
            &weights_x[base..],
            inv_qac_x_v,
        );
        let dq_x = dq_x_raw + x_factor_v * dq_y;
        dq_x.store((&mut output_x[base..base + 4]).try_into().unwrap());

        // Channel B + CfL
        let q_i_b = i32x4::from_slice(token, &quant_ac_b[base..]);
        let dq_b_raw = wasm128_dequant_4(
            token,
            q_i_b,
            bias_b_v,
            bias_recip_v,
            threshold,
            zero_f,
            one_f,
            neg_one_f,
            half_v,
            &weights_b[base..],
            inv_qac_b_v,
        );
        let dq_b = dq_b_raw + b_factor_v * dq_y;
        dq_b.store((&mut output_b[base..base + 4]).try_into().unwrap());
    }

    output_x[0] = 0.0;
    output_y[0] = 0.0;
    output_b[0] = 0.0;
}

/// Dequantize 4 coefficients with adjust_quant_bias, branchless WASM128.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
#[allow(clippy::too_many_arguments)]
fn wasm128_dequant_4(
    token: archmage::Wasm128Token,
    q_int: magetypes::simd::i32x4,
    channel_bias: magetypes::simd::f32x4,
    bias_recip: magetypes::simd::f32x4,
    threshold: magetypes::simd::f32x4,
    zero_f: magetypes::simd::f32x4,
    one_f: magetypes::simd::f32x4,
    neg_one_f: magetypes::simd::f32x4,
    half_v: magetypes::simd::f32x4,
    weights: &[f32],
    inv_qac_qm: magetypes::simd::f32x4,
) -> magetypes::simd::f32x4 {
    use magetypes::simd::f32x4;

    let q_f = f32x4::from_i32x4(token, q_int);
    let abs_q = q_f.abs();

    let sign = f32x4::blend(q_f.simd_ge(zero_f), one_f, neg_one_f);

    // Case 1: |q| < 1.125 → sign * channel_bias
    let case_one = sign * channel_bias;

    // Case 2: |q| >= 1.125 → q - 0.145/q
    let case_large = q_f - bias_recip / q_f;

    let is_large = abs_q.simd_ge(threshold);
    let biased = f32x4::blend(is_large, case_large, case_one);

    // Zero out where q == 0
    let is_nonzero = abs_q.simd_ge(half_v);
    let biased = f32x4::blend(is_nonzero, biased, zero_f);

    let w = f32x4::from_slice(token, weights);
    biased * w * inv_qac_qm
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    extern crate std;

    #[test]
    fn test_dequant_dct8_matches_scalar() {
        let mut quant_x = [0i32; 64];
        let mut quant_y = [0i32; 64];
        let mut quant_b = [0i32; 64];
        let mut weights_x = [0.01f32; 64];
        let mut weights_y = [0.01f32; 64];
        let mut weights_b = [0.01f32; 64];

        // Fill with varied values: 0, ±1, and larger
        for i in 0..64 {
            let v = (i as i32) - 32; // -32..+31
            quant_x[i] = v;
            quant_y[i] = v / 2;
            quant_b[i] = -v;
            weights_x[i] = 0.01 + i as f32 * 0.001;
            weights_y[i] = 0.02 + i as f32 * 0.0005;
            weights_b[i] = 0.015 + i as f32 * 0.0008;
        }

        let qac_qm = [3.5f32, 4.0, 3.2];
        let x_factor = 0.15f32;
        let b_factor = 1.05f32;

        let mut ref_x = [0.0f32; 64];
        let mut ref_y = [0.0f32; 64];
        let mut ref_b = [0.0f32; 64];
        dequant_dct8_scalar(
            &quant_x, &quant_y, &quant_b, &weights_x, &weights_y, &weights_b, qac_qm, x_factor,
            b_factor, &mut ref_x, &mut ref_y, &mut ref_b,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut out_x = [0.0f32; 64];
                let mut out_y = [0.0f32; 64];
                let mut out_b = [0.0f32; 64];
                dequant_block_dct8(
                    &quant_x, &quant_y, &quant_b, &weights_x, &weights_y, &weights_b, qac_qm,
                    x_factor, b_factor, &mut out_x, &mut out_y, &mut out_b,
                );
                let eps = 1e-5;
                for i in 0..64 {
                    assert!((out_x[i] - ref_x[i]).abs() < eps, "X[{i}] [{perm}]");
                    assert!((out_y[i] - ref_y[i]).abs() < eps, "Y[{i}] [{perm}]");
                    assert!((out_b[i] - ref_b[i]).abs() < eps, "B[{i}] [{perm}]");
                }
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_dequant_dct8_all_zeros() {
        let quant = [0i32; 64];
        let weights = [1.0f32; 64];
        let qac_qm = [1.0f32; 3];

        let mut out_x = [99.0f32; 64];
        let mut out_y = [99.0f32; 64];
        let mut out_b = [99.0f32; 64];
        dequant_block_dct8(
            &quant, &quant, &quant, &weights, &weights, &weights, qac_qm, 0.1, 1.0, &mut out_x,
            &mut out_y, &mut out_b,
        );

        // DC (index 0) stays as-is from SIMD but scalar sets to 0 since it starts from i=1
        // Actually the SIMD path zeros DC explicitly
        for i in 0..64 {
            assert_eq!(out_x[i], 0.0, "X[{}] should be 0 for zero input", i);
            assert_eq!(out_y[i], 0.0, "Y[{}] should be 0 for zero input", i);
            assert_eq!(out_b[i], 0.0, "B[{}] should be 0 for zero input", i);
        }
    }

    #[test]
    fn test_dequant_dct8_unit_values() {
        // All ±1 values — tests the channel bias path
        let mut quant = [0i32; 64];
        for (i, q) in quant.iter_mut().enumerate().skip(1) {
            *q = if i % 2 == 0 { 1 } else { -1 };
        }
        let weights = [1.0f32; 64];
        let qac_qm = [1.0f32, 1.0, 1.0];

        let mut ref_x = [0.0f32; 64];
        let mut ref_y = [0.0f32; 64];
        let mut ref_b = [0.0f32; 64];
        dequant_dct8_scalar(
            &quant, &quant, &quant, &weights, &weights, &weights, qac_qm, 0.0, 0.0, &mut ref_x,
            &mut ref_y, &mut ref_b,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut out_x = [0.0f32; 64];
                let mut out_y = [0.0f32; 64];
                let mut out_b = [0.0f32; 64];
                dequant_block_dct8(
                    &quant, &quant, &quant, &weights, &weights, &weights, qac_qm, 0.0, 0.0,
                    &mut out_x, &mut out_y, &mut out_b,
                );
                let eps = 1e-6;
                for i in 1..64 {
                    assert!((out_y[i] - ref_y[i]).abs() < eps, "Y[{i}] [{perm}]");
                }
            },
        );
        std::eprintln!("{report}");
    }
}
