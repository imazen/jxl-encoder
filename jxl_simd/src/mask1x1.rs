// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated per-pixel masking field computation (compute_mask1x1).
//!
//! Computes a perceptual masking field from the XYB Y channel.
//! For each pixel: 4-neighbor average → ratio_of_derivatives → |diff| → log1p → reciprocal.
//! The fast_log2f approximation uses integer bit manipulation on f32x8 via bitcast.

// --- Constants ---

const MATCH_GAMMA_OFFSET: f32 = 0.019;
const K_MUL: f32 = 1.0;
const K_OFFSET: f32 = 0.01;

// SimpleGamma constants (from libjxl enc_adaptive_quantization.cc)
#[allow(clippy::excessive_precision)]
const SG_MUL: f32 = 226.77216153508914;
#[allow(clippy::excessive_precision)]
const SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_INV_LOG2E: f32 = core::f32::consts::LN_2;
#[allow(clippy::excessive_precision)]
const SG_RET_MUL: f32 = SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
#[allow(clippy::excessive_precision)]
const SG_V_OFFSET: f32 = 7.7825991679894591;

// Precomputed constants for ratio_of_derivatives
const EPSILON: f32 = 1e-2;
const K_NUM_MUL: f32 = SG_RET_MUL * 3.0 * SG_MUL;
const K_V_OFFSET: f32 = SG_V_OFFSET * K_INV_LOG2E + EPSILON;
const K_DEN_MUL: f32 = K_INV_LOG2E * SG_MUL;

// ln(x) = log2(x) * ln(2)
const LN2: f32 = K_INV_LOG2E;

// fast_log2f polynomial coefficients
const LOG2_P0: f32 = -1.850_383_3e-6;
const LOG2_P1: f32 = 1.428_716;
const LOG2_P2: f32 = 0.742_458_7;
const LOG2_Q0: f32 = 0.990_328_14;
const LOG2_Q1: f32 = 1.009_671_9;
const LOG2_Q2: f32 = 0.174_093_43;

/// Fast log2 approximation using integer bit tricks.
/// Max relative error ~3e-7. Input must be > 0.
#[inline(always)]
fn fast_log2f(x: f32) -> f32 {
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

/// Ratio of derivatives (den/num, for invert=false).
/// Maps from opsin (cubic root) space to butteraugli's log-gamma space.
#[inline(always)]
fn ratio_of_derivatives_scalar(v: f32) -> f32 {
    let v = v.max(0.0);
    let v2 = v * v;
    let num = K_NUM_MUL * v2 + EPSILON;
    let den = K_DEN_MUL * v * v2 + K_V_OFFSET;
    den / num
}

/// Compute per-pixel masking field from XYB Y channel.
///
/// For each pixel: neighbor average → gamma derivative ratio → |diff| → log1p → reciprocal.
///
/// `xyb_y`: Y channel, row-major, `width * height` elements.
/// `output`: output buffer, same size.
#[inline]
pub fn compute_mask1x1(xyb_y: &[f32], width: usize, height: usize, output: &mut [f32]) {
    debug_assert!(xyb_y.len() >= width * height);
    debug_assert!(output.len() >= width * height);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            compute_mask1x1_avx2(token, xyb_y, width, height, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            compute_mask1x1_neon(token, xyb_y, width, height, output);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            compute_mask1x1_wasm128(token, xyb_y, width, height, output);
            return;
        }
    }

    compute_mask1x1_scalar(xyb_y, width, height, output);
}

#[inline]
pub fn compute_mask1x1_scalar(xyb_y: &[f32], width: usize, height: usize, output: &mut [f32]) {
    for y in 0..height {
        let y1 = y.saturating_sub(1);
        let y2 = (y + 1).min(height - 1);

        for x in 0..width {
            let x1 = x.saturating_sub(1);
            let x2 = (x + 1).min(width - 1);

            let base = 0.25
                * (xyb_y[y1 * width + x]
                    + xyb_y[y2 * width + x]
                    + xyb_y[y * width + x1]
                    + xyb_y[y * width + x2]);

            let pixel_val = xyb_y[y * width + x];
            let gammac = ratio_of_derivatives_scalar(pixel_val + MATCH_GAMMA_OFFSET);

            let diff = (gammac * (pixel_val - base)).abs();
            let diff = fast_log2f(1.0 + diff) * LN2; // ln(1 + diff)

            output[y * width + x] = K_MUL / (diff + K_OFFSET);
        }
    }
}

// ============================================================================
// x86_64 AVX2+FMA implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
pub fn compute_mask1x1_avx2(
    token: archmage::X64V3Token,
    xyb_y: &[f32],
    width: usize,
    height: usize,
    output: &mut [f32],
) {
    use magetypes::simd::{f32x8, i32x8};

    // Images too small for SIMD interior
    if width < 10 || height < 3 {
        compute_mask1x1_scalar(xyb_y, width, height, output);
        return;
    }

    let quarter = f32x8::splat(token, 0.25);
    let gamma_off = f32x8::splat(token, MATCH_GAMMA_OFFSET);
    let zero = f32x8::splat(token, 0.0);
    let eps_v = f32x8::splat(token, EPSILON);
    let k_num_mul_v = f32x8::splat(token, K_NUM_MUL);
    let k_den_mul_v = f32x8::splat(token, K_DEN_MUL);
    let k_v_offset_v = f32x8::splat(token, K_V_OFFSET);
    let one_v = f32x8::splat(token, 1.0);
    let ln2_v = f32x8::splat(token, LN2);
    let k_offset_v = f32x8::splat(token, K_OFFSET);
    let k_mul_v = f32x8::splat(token, K_MUL);

    // fast_log2f constants
    let log2_offset = i32x8::splat(token, 0x3f2a_aaab_u32 as i32);
    let log2_p0_v = f32x8::splat(token, LOG2_P0);
    let log2_p1_v = f32x8::splat(token, LOG2_P1);
    let log2_p2_v = f32x8::splat(token, LOG2_P2);
    let log2_q0_v = f32x8::splat(token, LOG2_Q0);
    let log2_q1_v = f32x8::splat(token, LOG2_Q1);
    let log2_q2_v = f32x8::splat(token, LOG2_Q2);

    // SIMD fast_log2f: log2(x) via integer bit manipulation
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
        // Rational polynomial: (p0 + frac*(p1 + frac*p2)) / (q0 + frac*(q1 + frac*q2))
        let num = frac.mul_add(p2, p1).mul_add(frac, p0);
        let den = frac.mul_add(q2, q1).mul_add(frac, q0);
        num / den + exp_val
    }

    // Process rows
    for y in 0..height {
        let y1 = y.saturating_sub(1);
        let y2 = (y + 1).min(height - 1);
        let r_top = y1 * width;
        let r_cur = y * width;
        let r_bot = y2 * width;

        // Scalar for first pixel (x=0)
        {
            let base = 0.25 * (xyb_y[r_top] + xyb_y[r_bot] + xyb_y[r_cur] + xyb_y[r_cur + 1]);
            let pv = xyb_y[r_cur];
            let gammac = ratio_of_derivatives_scalar(pv + MATCH_GAMMA_OFFSET);
            let diff = (gammac * (pv - base)).abs();
            let diff = fast_log2f(1.0 + diff) * LN2;
            output[r_cur] = K_MUL / (diff + K_OFFSET);
        }

        // SIMD interior: x = 1 .. width-1, in chunks of 8
        // Need left (x-1) and right (x+1), so x-1 >= 0 and x+8 < width
        let simd_end = if width > 9 { width - 1 - 8 + 1 } else { 1 };
        let mut x = 1;

        while x < simd_end {
            // Load neighbors
            let top = crate::load_f32x8(token, xyb_y, r_top + x);
            let bot = crate::load_f32x8(token, xyb_y, r_bot + x);
            let left = crate::load_f32x8(token, xyb_y, r_cur + x - 1);
            let right = crate::load_f32x8(token, xyb_y, r_cur + x + 1);
            let center = crate::load_f32x8(token, xyb_y, r_cur + x);

            // base = 0.25 * (top + bot + left + right)
            let base = quarter * (top + bot + left + right);

            // v = max(center + gamma_offset, 0)
            let v = (center + gamma_off).max(zero);

            // ratio_of_derivatives(v) = den/num
            let v2 = v * v;
            let v3 = v2 * v;
            let num = k_num_mul_v.mul_add(v2, eps_v);
            let den = k_den_mul_v.mul_add(v3, k_v_offset_v);
            let gammac = den / num;

            // diff = |gammac * (center - base)|
            let diff = (gammac * (center - base)).abs();

            // ln(1 + diff) = log2(1 + diff) * ln(2)
            let arg = one_v + diff;
            let log2_val = fast_log2f_x8(
                arg,
                log2_offset,
                log2_p0_v,
                log2_p1_v,
                log2_p2_v,
                log2_q0_v,
                log2_q1_v,
                log2_q2_v,
                one_v,
            );
            let ln_val = log2_val * ln2_v;

            // output = K_MUL / (ln_val + K_OFFSET)
            let result = k_mul_v / (ln_val + k_offset_v);

            crate::store_f32x8(output, r_cur + x, result);

            x += 8;
        }

        // Scalar remainder (right edge)
        while x < width {
            let x1 = x.saturating_sub(1);
            let x2 = (x + 1).min(width - 1);
            let base = 0.25
                * (xyb_y[r_top + x] + xyb_y[r_bot + x] + xyb_y[r_cur + x1] + xyb_y[r_cur + x2]);
            let pv = xyb_y[r_cur + x];
            let gammac = ratio_of_derivatives_scalar(pv + MATCH_GAMMA_OFFSET);
            let diff = (gammac * (pv - base)).abs();
            let diff = fast_log2f(1.0 + diff) * LN2;
            output[r_cur + x] = K_MUL / (diff + K_OFFSET);
            x += 1;
        }
    }
}

// ============================================================================
// aarch64 NEON implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
pub fn compute_mask1x1_neon(
    token: archmage::NeonToken,
    xyb_y: &[f32],
    width: usize,
    height: usize,
    output: &mut [f32],
) {
    use magetypes::simd::{f32x4, i32x4};

    // Images too small for SIMD interior
    if width < 6 || height < 3 {
        compute_mask1x1_scalar(xyb_y, width, height, output);
        return;
    }

    let quarter = f32x4::splat(token, 0.25);
    let gamma_off = f32x4::splat(token, MATCH_GAMMA_OFFSET);
    let zero = f32x4::splat(token, 0.0);
    let eps_v = f32x4::splat(token, EPSILON);
    let k_num_mul_v = f32x4::splat(token, K_NUM_MUL);
    let k_den_mul_v = f32x4::splat(token, K_DEN_MUL);
    let k_v_offset_v = f32x4::splat(token, K_V_OFFSET);
    let one_v = f32x4::splat(token, 1.0);
    let ln2_v = f32x4::splat(token, LN2);
    let k_offset_v = f32x4::splat(token, K_OFFSET);
    let k_mul_v = f32x4::splat(token, K_MUL);

    // fast_log2f constants
    let log2_offset = i32x4::splat(token, 0x3f2a_aaab_u32 as i32);
    let log2_p0_v = f32x4::splat(token, LOG2_P0);
    let log2_p1_v = f32x4::splat(token, LOG2_P1);
    let log2_p2_v = f32x4::splat(token, LOG2_P2);
    let log2_q0_v = f32x4::splat(token, LOG2_Q0);
    let log2_q1_v = f32x4::splat(token, LOG2_Q1);
    let log2_q2_v = f32x4::splat(token, LOG2_Q2);

    // SIMD fast_log2f via integer bit manipulation (f32x4 version)
    #[allow(clippy::too_many_arguments)]
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

    for y in 0..height {
        let y1 = y.saturating_sub(1);
        let y2 = (y + 1).min(height - 1);
        let r_top = y1 * width;
        let r_cur = y * width;
        let r_bot = y2 * width;

        // Scalar for first pixel (x=0)
        {
            let base = 0.25 * (xyb_y[r_top] + xyb_y[r_bot] + xyb_y[r_cur] + xyb_y[r_cur + 1]);
            let pv = xyb_y[r_cur];
            let gammac = ratio_of_derivatives_scalar(pv + MATCH_GAMMA_OFFSET);
            let diff = (gammac * (pv - base)).abs();
            let diff = fast_log2f(1.0 + diff) * LN2;
            output[r_cur] = K_MUL / (diff + K_OFFSET);
        }

        // SIMD interior: need left (x-1) and right (x+1)
        let simd_end = if width > 5 { width - 1 - 4 + 1 } else { 1 };
        let mut x = 1;

        while x < simd_end {
            let top = f32x4::from_slice(token, &xyb_y[r_top + x..]);
            let bot = f32x4::from_slice(token, &xyb_y[r_bot + x..]);
            let left = f32x4::from_slice(token, &xyb_y[r_cur + x - 1..]);
            let right = f32x4::from_slice(token, &xyb_y[r_cur + x + 1..]);
            let center = f32x4::from_slice(token, &xyb_y[r_cur + x..]);

            let base = quarter * (top + bot + left + right);

            let v = (center + gamma_off).max(zero);
            let v2 = v * v;
            let v3 = v2 * v;
            let num = k_num_mul_v.mul_add(v2, eps_v);
            let den = k_den_mul_v.mul_add(v3, k_v_offset_v);
            let gammac = den / num;

            let diff = (gammac * (center - base)).abs();

            let arg = one_v + diff;
            let log2_val = fast_log2f_x4(
                arg,
                log2_offset,
                log2_p0_v,
                log2_p1_v,
                log2_p2_v,
                log2_q0_v,
                log2_q1_v,
                log2_q2_v,
                one_v,
            );
            let ln_val = log2_val * ln2_v;

            let result = k_mul_v / (ln_val + k_offset_v);

            let out_arr: &mut [f32; 4] =
                (&mut output[r_cur + x..r_cur + x + 4]).try_into().unwrap();
            result.store(out_arr);

            x += 4;
        }

        // Scalar remainder (right edge)
        while x < width {
            let x1 = x.saturating_sub(1);
            let x2 = (x + 1).min(width - 1);
            let base = 0.25
                * (xyb_y[r_top + x] + xyb_y[r_bot + x] + xyb_y[r_cur + x1] + xyb_y[r_cur + x2]);
            let pv = xyb_y[r_cur + x];
            let gammac = ratio_of_derivatives_scalar(pv + MATCH_GAMMA_OFFSET);
            let diff = (gammac * (pv - base)).abs();
            let diff = fast_log2f(1.0 + diff) * LN2;
            output[r_cur + x] = K_MUL / (diff + K_OFFSET);
            x += 1;
        }
    }
}

// ============================================================================
// wasm32 SIMD128 implementation
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
pub fn compute_mask1x1_wasm128(
    token: archmage::Wasm128Token,
    xyb_y: &[f32],
    width: usize,
    height: usize,
    output: &mut [f32],
) {
    use magetypes::simd::{f32x4, i32x4};

    // Images too small for SIMD interior
    if width < 6 || height < 3 {
        compute_mask1x1_scalar(xyb_y, width, height, output);
        return;
    }

    let quarter = f32x4::splat(token, 0.25);
    let gamma_off = f32x4::splat(token, MATCH_GAMMA_OFFSET);
    let zero = f32x4::splat(token, 0.0);
    let eps_v = f32x4::splat(token, EPSILON);
    let k_num_mul_v = f32x4::splat(token, K_NUM_MUL);
    let k_den_mul_v = f32x4::splat(token, K_DEN_MUL);
    let k_v_offset_v = f32x4::splat(token, K_V_OFFSET);
    let one_v = f32x4::splat(token, 1.0);
    let ln2_v = f32x4::splat(token, LN2);
    let k_offset_v = f32x4::splat(token, K_OFFSET);
    let k_mul_v = f32x4::splat(token, K_MUL);

    // fast_log2f constants
    let log2_offset = i32x4::splat(token, 0x3f2a_aaab_u32 as i32);
    let log2_p0_v = f32x4::splat(token, LOG2_P0);
    let log2_p1_v = f32x4::splat(token, LOG2_P1);
    let log2_p2_v = f32x4::splat(token, LOG2_P2);
    let log2_q0_v = f32x4::splat(token, LOG2_Q0);
    let log2_q1_v = f32x4::splat(token, LOG2_Q1);
    let log2_q2_v = f32x4::splat(token, LOG2_Q2);

    // SIMD fast_log2f via integer bit manipulation (f32x4 version)
    #[allow(clippy::too_many_arguments)]
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

    for y in 0..height {
        let y1 = y.saturating_sub(1);
        let y2 = (y + 1).min(height - 1);
        let r_top = y1 * width;
        let r_cur = y * width;
        let r_bot = y2 * width;

        // Scalar for first pixel (x=0)
        {
            let base = 0.25 * (xyb_y[r_top] + xyb_y[r_bot] + xyb_y[r_cur] + xyb_y[r_cur + 1]);
            let pv = xyb_y[r_cur];
            let gammac = ratio_of_derivatives_scalar(pv + MATCH_GAMMA_OFFSET);
            let diff = (gammac * (pv - base)).abs();
            let diff = fast_log2f(1.0 + diff) * LN2;
            output[r_cur] = K_MUL / (diff + K_OFFSET);
        }

        // SIMD interior: need left (x-1) and right (x+1)
        let simd_end = if width > 5 { width - 1 - 4 + 1 } else { 1 };
        let mut x = 1;

        while x < simd_end {
            let top = f32x4::from_slice(token, &xyb_y[r_top + x..]);
            let bot = f32x4::from_slice(token, &xyb_y[r_bot + x..]);
            let left = f32x4::from_slice(token, &xyb_y[r_cur + x - 1..]);
            let right = f32x4::from_slice(token, &xyb_y[r_cur + x + 1..]);
            let center = f32x4::from_slice(token, &xyb_y[r_cur + x..]);

            let base = quarter * (top + bot + left + right);

            let v = (center + gamma_off).max(zero);
            let v2 = v * v;
            let v3 = v2 * v;
            let num = k_num_mul_v.mul_add(v2, eps_v);
            let den = k_den_mul_v.mul_add(v3, k_v_offset_v);
            let gammac = den / num;

            let diff = (gammac * (center - base)).abs();

            let arg = one_v + diff;
            let log2_val = fast_log2f_x4(
                arg,
                log2_offset,
                log2_p0_v,
                log2_p1_v,
                log2_p2_v,
                log2_q0_v,
                log2_q1_v,
                log2_q2_v,
                one_v,
            );
            let ln_val = log2_val * ln2_v;

            let result = k_mul_v / (ln_val + k_offset_v);

            let out_arr: &mut [f32; 4] =
                (&mut output[r_cur + x..r_cur + x + 4]).try_into().unwrap();
            result.store(out_arr);

            x += 4;
        }

        // Scalar remainder (right edge)
        while x < width {
            let x1 = x.saturating_sub(1);
            let x2 = (x + 1).min(width - 1);
            let base = 0.25
                * (xyb_y[r_top + x] + xyb_y[r_bot + x] + xyb_y[r_cur + x1] + xyb_y[r_cur + x2]);
            let pv = xyb_y[r_cur + x];
            let gammac = ratio_of_derivatives_scalar(pv + MATCH_GAMMA_OFFSET);
            let diff = (gammac * (pv - base)).abs();
            let diff = fast_log2f(1.0 + diff) * LN2;
            output[r_cur + x] = K_MUL / (diff + K_OFFSET);
            x += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    extern crate std;
    use alloc::vec;

    #[test]
    fn test_mask1x1_matches_reference() {
        // Create a test image with varied values
        let width = 128;
        let height = 128;
        let mut xyb_y = vec![0.0f32; width * height];

        // Fill with semi-random pattern using deterministic formula
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 7 + y * 13 + x * y) % 1000) as f32 / 1000.0;
                xyb_y[y * width + x] = v * 0.5; // Keep in typical XYB Y range
            }
        }

        // Reference: use scalar with standard ln
        let mut reference = vec![0.0f32; width * height];
        for y in 0..height {
            let y1 = y.saturating_sub(1);
            let y2 = (y + 1).min(height - 1);
            for x in 0..width {
                let x1 = x.saturating_sub(1);
                let x2 = (x + 1).min(width - 1);
                let base = 0.25
                    * (xyb_y[y1 * width + x]
                        + xyb_y[y2 * width + x]
                        + xyb_y[y * width + x1]
                        + xyb_y[y * width + x2]);
                let pv = xyb_y[y * width + x];
                let v = (pv + MATCH_GAMMA_OFFSET).max(0.0);
                let v2 = v * v;
                let num = K_NUM_MUL * v2 + EPSILON;
                let den = K_DEN_MUL * v * v2 + K_V_OFFSET;
                let gammac = den / num;
                let diff = (gammac * (pv - base)).abs();
                // Use standard ln for reference
                let ln_diff = (1.0 + diff).ln();
                reference[y * width + x] = K_MUL / (ln_diff + K_OFFSET);
            }
        }

        let mut result = vec![0.0f32; width * height];
        compute_mask1x1(&xyb_y, width, height, &mut result);

        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..width * height {
            let diff = (result[i] - reference[i]).abs();
            let rel = if reference[i].abs() > 1e-6 {
                diff / reference[i].abs()
            } else {
                diff
            };
            max_diff = max_diff.max(diff);
            max_rel = max_rel.max(rel);
        }

        // fast_log2f has ~3e-7 max relative error, which cascades to small output errors
        assert!(max_rel < 0.01, "max_rel = {max_rel}, max_diff = {max_diff}");
    }

    #[test]
    fn test_mask1x1_small_images() {
        // Test edge cases: very small images
        for (w, h) in [(8, 8), (9, 9), (1, 1), (2, 2), (3, 3)] {
            let xyb_y = vec![0.1f32; w * h];
            let mut output = vec![0.0f32; w * h];
            compute_mask1x1(&xyb_y, w, h, &mut output);
            // Should not panic, all values should be finite and positive
            for &v in &output {
                assert!(v.is_finite() && v > 0.0, "w={w}, h={h}: got {v}");
            }
        }
    }

    #[test]
    fn test_mask1x1_non_multiple_of_8() {
        // Width not a multiple of 8 — exercises scalar remainder
        let width = 37;
        let height = 19;
        let mut xyb_y = vec![0.0f32; width * height];
        for (i, val) in xyb_y.iter_mut().enumerate() {
            *val = (i as f32 * 0.001).sin().abs() * 0.3;
        }

        // Scalar reference
        let mut scalar_out = vec![0.0f32; width * height];
        compute_mask1x1_scalar(&xyb_y, width, height, &mut scalar_out);

        // Dispatch — test all token permutations
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut output = vec![0.0f32; width * height];
                compute_mask1x1(&xyb_y, width, height, &mut output);

                let mut max_diff = 0.0f32;
                for i in 0..width * height {
                    let diff = (output[i] - scalar_out[i]).abs();
                    max_diff = max_diff.max(diff);
                }

                // SIMD and scalar use same fast_log2f, should be very close
                assert!(
                    max_diff < 1e-4,
                    "SIMD vs scalar max_diff = {max_diff} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }
}
