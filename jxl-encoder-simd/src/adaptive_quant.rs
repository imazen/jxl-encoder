// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated adaptive quantization kernels.
//!
//! Two kernels:
//! - `compute_pre_erosion`: per-pixel stencil (4-neighbor avg, ratio_of_derivatives, masking_sqrt)
//!   with 4× downsampling. Processes every pixel of the image.
//! - `per_block_modulations`: per-8×8-block modulations (ComputeMask, GammaModulation,
//!   HfModulation, BlueModulation, exp2). Processes 64 pixels per block.

// Ported float constants from C++ — exact values are intentional for parity.
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

// ============================================================================
// Constants (shared across all implementations)
// ============================================================================

// SimpleGamma / ratio_of_derivatives constants
#[allow(clippy::excessive_precision)]
const SG_MUL: f32 = 226.77216153508914;
#[allow(clippy::excessive_precision)]
const SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_INV_LOG2E: f32 = core::f32::consts::LN_2;
#[allow(clippy::excessive_precision)]
const SG_RET_MUL: f32 = SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
#[allow(clippy::excessive_precision)]
const SG_V_OFFSET: f32 = 7.7825991679894591;

const EPSILON: f32 = 1e-2;
const K_NUM_MUL: f32 = SG_RET_MUL * 3.0 * SG_MUL;
const K_V_OFFSET: f32 = SG_V_OFFSET * K_INV_LOG2E + EPSILON;
const K_DEN_MUL: f32 = K_INV_LOG2E * SG_MUL;

// Pre-erosion constants
const MATCH_GAMMA_OFFSET: f32 = 0.019;
const LIMIT: f32 = 0.2;

// MaskingSqrt constants
#[allow(clippy::excessive_precision)]
const MASKING_K_LOG_OFFSET: f32 = 27.505837037000106;
#[allow(clippy::excessive_precision)]
const MASKING_K_MUL: f32 = 211.66567973503678;

// ComputeMask constants
const CM_K_BASE: f32 = -0.7647;
#[allow(clippy::excessive_precision)]
const CM_K_MUL4: f32 = 9.4708735624378946;
#[allow(clippy::excessive_precision)]
const CM_K_MUL2: f32 = 17.35036561631863;
#[allow(clippy::excessive_precision)]
const CM_K_OFFSET2: f32 = 302.59587815579727;
#[allow(clippy::excessive_precision)]
const CM_K_MUL3: f32 = 6.7943250517376494;
#[allow(clippy::excessive_precision)]
const CM_K_OFFSET3: f32 = 3.7179635626140772;
const CM_K_OFFSET4: f32 = 0.25 * CM_K_OFFSET3;
#[allow(clippy::excessive_precision)]
const CM_K_MUL0: f32 = 0.80061762862741759;

// GammaModulation constants
const GM_K_BIAS: f32 = 0.16;
#[allow(clippy::excessive_precision)]
const GM_K_GAMMA: f32 = 0.1005613337192697;

// HfModulation constants
const HF_VALMIN_Y: f32 = 0.0206;
const HF_K_MUL_Y: f32 = -0.38;
const HF_K_OFFSET: f32 = 0.42;

// BlueModulation constants
#[allow(clippy::excessive_precision)]
const BM_K_LIMIT: f32 = 0.010474084867598155;
#[allow(clippy::excessive_precision)]
const BM_K_OFFSET: f32 = 0.0031994768654636393;
#[allow(clippy::excessive_precision)]
const BM_K_MAX_LIMIT: f32 = 15.463398341612438;
#[allow(clippy::excessive_precision)]
const BM_K_MUL: f32 = 0.90590804735610064;

// fast_log2f polynomial coefficients
const LOG2_P0: f32 = -1.850_383_3e-6;
const LOG2_P1: f32 = 1.428_716;
const LOG2_P2: f32 = 0.742_458_7;
const LOG2_Q0: f32 = 0.990_328_14;
const LOG2_Q1: f32 = 1.009_671_9;
const LOG2_Q2: f32 = 0.174_093_43;

// ============================================================================
// Scalar helper functions
// ============================================================================

#[inline(always)]
fn fast_log2f(x: f32) -> f32 {
    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits.wrapping_sub(0x3f2a_aaab_u32 as i32);
    let exp_shifted = exp_bits >> 23;
    let mantissa = f32::from_bits((x_bits.wrapping_sub(exp_shifted << 23)) as u32);
    let frac = mantissa - 1.0;
    let num = LOG2_P0 + frac * (LOG2_P1 + frac * LOG2_P2);
    let den = LOG2_Q0 + frac * (LOG2_Q1 + frac * LOG2_Q2);
    num / den + exp_shifted as f32
}

#[inline(always)]
fn fast_pow2f(x: f32) -> f32 {
    let floorx = x.floor();
    let exp = f32::from_bits(((floorx as i32 + 127) << 23) as u32);
    let frac = x - floorx;
    let num = ((frac + 1.017_490_63e+01) * frac + 4.886_877_98e+01) * frac + 9.855_065_91e+01;
    let num = num * exp;
    let den = ((frac * 2.102_429_58e-01 + (-2.223_288_56e-02)) * frac + (-1.944_149_9e+01)) * frac
        + 9.855_066_33e+01;
    num / den
}

/// ratio_of_derivatives(v, invert=false): den/num
#[inline(always)]
fn ratio_of_deriv_normal(v: f32) -> f32 {
    let v = v.max(0.0);
    let v2 = v * v;
    let num = K_NUM_MUL * v2 + EPSILON;
    let den = K_DEN_MUL * v * v2 + K_V_OFFSET;
    den / num
}

/// ratio_of_derivatives(v, invert=true): num/den
#[inline(always)]
fn ratio_of_deriv_inverted(v: f32) -> f32 {
    let v = v.max(0.0);
    let v2 = v * v;
    let num = K_NUM_MUL * v2 + EPSILON;
    let den = K_DEN_MUL * v * v2 + K_V_OFFSET;
    num / den
}

#[inline(always)]
fn masking_sqrt(v: f32) -> f32 {
    let mul_v = MASKING_K_MUL * 1e8;
    0.25 * (v * mul_v.sqrt() + MASKING_K_LOG_OFFSET).sqrt()
}

// ============================================================================
// Dispatch functions
// ============================================================================

/// Compute pre-erosion map from Y channel: per-pixel stencil with 4× downsampling.
///
/// Returns `(pre_erosion_vec, width, height)` of the downsampled output.
pub fn compute_pre_erosion(
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
) -> (Vec<f32>, usize, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return compute_pre_erosion_avx2(
                token, xyb_y, width, height, tile_x0, tile_y0, tile_x1, tile_y1,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return compute_pre_erosion_neon(
                token, xyb_y, width, height, tile_x0, tile_y0, tile_x1, tile_y1,
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            return compute_pre_erosion_wasm128(
                token, xyb_y, width, height, tile_x0, tile_y0, tile_x1, tile_y1,
            );
        }
    }

    compute_pre_erosion_scalar(xyb_y, width, height, tile_x0, tile_y0, tile_x1, tile_y1)
}

/// Apply per-block modulations to aq_map in-place.
///
/// Full libjxl order: ComputeMask → GammaModulation → min(HfModulation, BlueModulation) → exp2.
#[allow(clippy::too_many_arguments)]
pub fn per_block_modulations(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    scale: f32,
    rect_x0_blocks: usize,
    rect_y0_blocks: usize,
    rect_w_blocks: usize,
    rect_h_blocks: usize,
    aq_map: &mut [f32],
    aq_map_stride: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            per_block_modulations_avx2(
                token,
                xyb_x,
                xyb_y,
                xyb_b,
                stride,
                butteraugli_target,
                scale,
                rect_x0_blocks,
                rect_y0_blocks,
                rect_w_blocks,
                rect_h_blocks,
                aq_map,
                aq_map_stride,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            per_block_modulations_neon(
                token,
                xyb_x,
                xyb_y,
                xyb_b,
                stride,
                butteraugli_target,
                scale,
                rect_x0_blocks,
                rect_y0_blocks,
                rect_w_blocks,
                rect_h_blocks,
                aq_map,
                aq_map_stride,
            );
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            per_block_modulations_wasm128(
                token,
                xyb_x,
                xyb_y,
                xyb_b,
                stride,
                butteraugli_target,
                scale,
                rect_x0_blocks,
                rect_y0_blocks,
                rect_w_blocks,
                rect_h_blocks,
                aq_map,
                aq_map_stride,
            );
            return;
        }
    }

    per_block_modulations_scalar(
        xyb_x,
        xyb_y,
        xyb_b,
        stride,
        butteraugli_target,
        scale,
        rect_x0_blocks,
        rect_y0_blocks,
        rect_w_blocks,
        rect_h_blocks,
        aq_map,
        aq_map_stride,
    );
}

// ============================================================================
// Scalar implementations
// ============================================================================

pub fn compute_pre_erosion_scalar(
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
) -> (Vec<f32>, usize, usize) {
    let x0 = tile_x0.saturating_sub(4);
    let x1 = if tile_x1 < width {
        tile_x1 + 4
    } else {
        tile_x1
    };
    let y_start = tile_y0.saturating_sub(4);
    let y_end = if tile_y1 < height {
        tile_y1 + 4
    } else {
        tile_y1
    };

    let diff_width = x1 - x0;
    let pre_erosion_w = diff_width / 4;
    let pre_erosion_h = (y_end - y_start) / 4;

    let mut diff_buffer = vec![0.0_f32; diff_width];
    let mut pre_erosion = vec![0.0_f32; pre_erosion_w * pre_erosion_h];

    let max_x = width - 1;
    let max_y = height - 1;

    for y in y_start..y_end {
        let yc = y.min(max_y);
        let y2 = (y + 1).min(max_y);
        let y1 = if y > 0 { (y - 1).min(max_y) } else { 0 };

        for x in x0..x1 {
            let xc = x.min(max_x);
            let x2 = (x + 1).min(max_x);
            let x1_local = if x > 0 { (x - 1).min(max_x) } else { 0 };

            let base = 0.25
                * (xyb_y[y2 * width + xc]
                    + xyb_y[y1 * width + xc]
                    + xyb_y[yc * width + x1_local]
                    + xyb_y[yc * width + x2]);

            let gammac = ratio_of_deriv_normal(xyb_y[yc * width + xc] + MATCH_GAMMA_OFFSET);

            let mut diff = gammac * (xyb_y[yc * width + xc] - base);
            diff *= diff;
            if diff >= LIMIT {
                diff = LIMIT;
            }
            diff = masking_sqrt(diff);

            let local_x = x - x0;
            if (y - y_start) % 4 != 0 {
                diff_buffer[local_x] += diff;
            } else {
                diff_buffer[local_x] = diff;
            }
        }

        if (y - y_start) % 4 == 3 {
            let row_y = (y - y_start) / 4;
            for bx in 0..pre_erosion_w {
                let sum = diff_buffer[bx * 4]
                    + diff_buffer[bx * 4 + 1]
                    + diff_buffer[bx * 4 + 2]
                    + diff_buffer[bx * 4 + 3];
                pre_erosion[row_y * pre_erosion_w + bx] = sum * 0.25;
            }
        }
    }

    (pre_erosion, pre_erosion_w, pre_erosion_h)
}

#[allow(clippy::too_many_arguments)]
pub fn per_block_modulations_scalar(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    scale: f32,
    rect_x0_blocks: usize,
    rect_y0_blocks: usize,
    rect_w_blocks: usize,
    rect_h_blocks: usize,
    aq_map: &mut [f32],
    aq_map_stride: usize,
) {
    let base_level = 0.48 * scale;
    let k_dampen_ramp_start = 2.0_f32;
    let k_dampen_ramp_end = 14.0_f32;
    let mut dampen = 1.0_f32;
    if butteraugli_target >= k_dampen_ramp_start {
        dampen = 1.0
            - ((butteraugli_target - k_dampen_ramp_start)
                / (k_dampen_ramp_end - k_dampen_ramp_start));
        if dampen < 0.0 {
            dampen = 0.0;
        }
    }
    let mul = scale * dampen;
    let add = (1.0 - dampen) * base_level;

    for iy in 0..rect_h_blocks {
        let block_iy = rect_y0_blocks + iy;
        let py = block_iy * 8;
        for ix in 0..rect_w_blocks {
            let block_ix = rect_x0_blocks + ix;
            let px = block_ix * 8;

            let mut out_val = aq_map[iy * aq_map_stride + ix];

            // ComputeMask
            out_val = compute_mask_scalar(out_val);

            // GammaModulation
            out_val = gamma_modulation_scalar(px, py, xyb_x, xyb_y, stride, out_val);

            // min(HfModulation, BlueModulation)
            let mask_val = out_val;
            let after_hf = hf_modulation_scalar(px, py, xyb_y, stride, mask_val);
            let after_blue = blue_modulation_scalar(px, py, xyb_x, xyb_y, xyb_b, stride, mask_val);
            out_val = after_hf.min(after_blue);

            aq_map[iy * aq_map_stride + ix] = fast_pow2f(out_val * 1.442_695) * mul + add;
        }
    }
}

#[inline(always)]
fn compute_mask_scalar(out_val: f32) -> f32 {
    let v1 = (out_val * CM_K_MUL0).max(1e-3);
    let v2 = 1.0 / (v1 + CM_K_OFFSET2);
    let v3 = 1.0 / (v1 * v1 + CM_K_OFFSET3);
    let v4 = 1.0 / (v1 * v1 + CM_K_OFFSET4);
    CM_K_BASE + CM_K_MUL4 * v4 + CM_K_MUL2 * v2 + CM_K_MUL3 * v3
}

#[inline(always)]
fn gamma_modulation_scalar(
    x: usize,
    y: usize,
    xyb_x: &[f32],
    xyb_y: &[f32],
    stride: usize,
    out_val: f32,
) -> f32 {
    let mut overall_ratio = 0.0_f32;
    for dy in 0..8 {
        let py = y + dy;
        for dx in 0..8 {
            let px = x + dx;
            let idx = py * stride + px;
            let iny = xyb_y[idx] + GM_K_BIAS;
            let inx = xyb_x[idx];
            overall_ratio +=
                ratio_of_deriv_inverted(iny - inx) + ratio_of_deriv_inverted(iny + inx);
        }
    }
    overall_ratio *= 0.5 / 64.0;
    out_val + GM_K_GAMMA * fast_log2f(overall_ratio)
}

#[inline(always)]
fn hf_modulation_scalar(x: usize, y: usize, xyb_y: &[f32], stride: usize, out_val: f32) -> f32 {
    let mut sum_y = 0.0_f32;
    for dy in 0..8 {
        let py = y + dy;
        let py_next = if dy == 7 { py } else { py + 1 };
        for dx in 0..8 {
            let px = x + dx;
            let p_y = xyb_y[py * stride + px];
            if dx < 7 {
                sum_y += (p_y - xyb_y[py * stride + px + 1]).abs().min(HF_VALMIN_Y);
            }
            sum_y += (p_y - xyb_y[py_next * stride + px]).abs().min(HF_VALMIN_Y);
        }
    }
    out_val + sum_y * HF_K_MUL_Y + HF_K_OFFSET
}

#[inline(always)]
fn blue_modulation_scalar(
    x: usize,
    y: usize,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    out_val: f32,
) -> f32 {
    let mut sum = 0.0_f32;
    for dy in 0..8 {
        let py = y + dy;
        for dx in 0..8 {
            let px = x + dx;
            let idx = py * stride + px;
            let p_x = xyb_x[idx];
            let p_b = xyb_b[idx];
            let p_y_effective = xyb_y[idx] + BM_K_OFFSET + p_x.abs();
            if p_b > p_y_effective {
                sum += (p_b - p_y_effective).min(BM_K_LIMIT);
            }
        }
    }
    if sum >= 32.0 * BM_K_LIMIT {
        sum = 64.0 * BM_K_LIMIT - sum;
    }
    if sum >= BM_K_MAX_LIMIT * BM_K_LIMIT {
        sum = BM_K_MAX_LIMIT * BM_K_LIMIT;
    }
    sum *= BM_K_MUL;
    out_val + sum
}

// ============================================================================
// x86_64 AVX2+FMA implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn compute_pre_erosion_avx2(
    token: archmage::X64V3Token,
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
) -> (Vec<f32>, usize, usize) {
    use magetypes::simd::f32x8;

    let x0 = tile_x0.saturating_sub(4);
    let x1 = if tile_x1 < width {
        tile_x1 + 4
    } else {
        tile_x1
    };
    let y_start = tile_y0.saturating_sub(4);
    let y_end = if tile_y1 < height {
        tile_y1 + 4
    } else {
        tile_y1
    };

    let diff_width = x1 - x0;
    let pre_erosion_w = diff_width / 4;
    let pre_erosion_h = (y_end - y_start) / 4;

    // Pad diff_buffer to next multiple of 8 for AVX2 loads/stores
    let diff_buffer_len = (diff_width + 7) & !7;
    let mut diff_buffer = vec![0.0_f32; diff_buffer_len];
    let mut pre_erosion = vec![0.0_f32; pre_erosion_w * pre_erosion_h];

    let max_x = width - 1;
    let max_y = height - 1;

    // SIMD constants
    let quarter = f32x8::splat(token, 0.25);
    let gamma_off = f32x8::splat(token, MATCH_GAMMA_OFFSET);
    let zero = f32x8::splat(token, 0.0);
    let eps_v = f32x8::splat(token, EPSILON);
    let k_num_mul_v = f32x8::splat(token, K_NUM_MUL);
    let k_den_mul_v = f32x8::splat(token, K_DEN_MUL);
    let k_v_offset_v = f32x8::splat(token, K_V_OFFSET);
    let limit_v = f32x8::splat(token, LIMIT);
    let masking_mul_v = f32x8::splat(token, (MASKING_K_MUL * 1e8_f32).sqrt());
    let masking_offset_v = f32x8::splat(token, MASKING_K_LOG_OFFSET);
    let masking_scale = f32x8::splat(token, 0.25);

    for y in y_start..y_end {
        let yc = y.min(max_y);
        let y2 = (y + 1).min(max_y);
        let y1 = if y > 0 { (y - 1).min(max_y) } else { 0 };

        // Interior pixels: SIMD when both left and right neighbors are in bounds
        // Need: x >= 1 and x+1 <= max_x, and within tile x0..x1
        // The tile coordinates are the interesting range; edges use scalar
        let interior_start = if x0 == 0 { 1 } else { 0 };
        let interior_end = if x1 > width {
            diff_width.saturating_sub(1)
        } else {
            diff_width
        };

        // Process scalar edges at start
        for local_x in 0..interior_start.min(diff_width) {
            let x = x0 + local_x;
            let xc = x.min(max_x);
            let x2 = (x + 1).min(max_x);
            let x1_local = if x > 0 { (x - 1).min(max_x) } else { 0 };
            let val = pre_erosion_pixel(xyb_y, width, yc, y1, y2, xc, x1_local, x2);
            if (y - y_start) % 4 != 0 {
                diff_buffer[local_x] += val;
            } else {
                diff_buffer[local_x] = val;
            }
        }

        // SIMD interior
        let mut local_x = interior_start;
        let simd_end = if interior_end > 8 {
            interior_end - 7
        } else {
            interior_start
        };

        while local_x < simd_end {
            let x = x0 + local_x;
            let xc = x.min(max_x); // For interior, x == xc

            // Load 4 neighbors + center
            let top = crate::load_f32x8(token, xyb_y, y1 * width + xc);
            let bot = crate::load_f32x8(token, xyb_y, y2 * width + xc);
            let left = crate::load_f32x8(token, xyb_y, yc * width + xc - 1);
            let right = crate::load_f32x8(token, xyb_y, yc * width + xc + 1);
            let center = crate::load_f32x8(token, xyb_y, yc * width + xc);

            let base = quarter * (top + bot + left + right);

            // ratio_of_derivatives(center + offset, invert=false) = den/num
            let v = (center + gamma_off).max(zero);
            let v2 = v * v;
            let v3 = v2 * v;
            let num = k_num_mul_v.mul_add(v2, eps_v);
            let den = k_den_mul_v.mul_add(v3, k_v_offset_v);
            let gammac = den / num;

            // diff = (gammac * (center - base))²
            let raw_diff = gammac * (center - base);
            let diff_sq = raw_diff * raw_diff;

            // clamp to LIMIT
            let clamped = diff_sq.min(limit_v);

            // masking_sqrt: 0.25 * sqrt(v * sqrt(K_MUL*1e8) + K_LOG_OFFSET)
            let result = masking_scale * (clamped.mul_add(masking_mul_v, masking_offset_v)).sqrt();

            // Accumulate into diff_buffer
            if (y - y_start) % 4 != 0 {
                let existing = f32x8::from_slice(token, &diff_buffer[local_x..]);
                let sum = existing + result;
                let out: &mut [f32; 8] =
                    (&mut diff_buffer[local_x..local_x + 8]).try_into().unwrap();
                sum.store(out);
            } else {
                let out: &mut [f32; 8] =
                    (&mut diff_buffer[local_x..local_x + 8]).try_into().unwrap();
                result.store(out);
            }

            local_x += 8;
        }

        // Scalar remainder
        while local_x < diff_width {
            let x = x0 + local_x;
            let xc = x.min(max_x);
            let x2 = (x + 1).min(max_x);
            let x1_local = if x > 0 { (x - 1).min(max_x) } else { 0 };
            let val = pre_erosion_pixel(xyb_y, width, yc, y1, y2, xc, x1_local, x2);
            if (y - y_start) % 4 != 0 {
                diff_buffer[local_x] += val;
            } else {
                diff_buffer[local_x] = val;
            }
            local_x += 1;
        }

        // 4× vertical downsample
        if (y - y_start) % 4 == 3 {
            let row_y = (y - y_start) / 4;
            for bx in 0..pre_erosion_w {
                let sum = diff_buffer[bx * 4]
                    + diff_buffer[bx * 4 + 1]
                    + diff_buffer[bx * 4 + 2]
                    + diff_buffer[bx * 4 + 3];
                pre_erosion[row_y * pre_erosion_w + bx] = sum * 0.25;
            }
        }
    }

    (pre_erosion, pre_erosion_w, pre_erosion_h)
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[archmage::arcane]
pub fn per_block_modulations_avx2(
    token: archmage::X64V3Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    scale: f32,
    rect_x0_blocks: usize,
    rect_y0_blocks: usize,
    rect_w_blocks: usize,
    rect_h_blocks: usize,
    aq_map: &mut [f32],
    aq_map_stride: usize,
) {
    use magetypes::simd::f32x8;

    let base_level = 0.48 * scale;
    let k_dampen_ramp_start = 2.0_f32;
    let k_dampen_ramp_end = 14.0_f32;
    let mut dampen = 1.0_f32;
    if butteraugli_target >= k_dampen_ramp_start {
        dampen = 1.0
            - ((butteraugli_target - k_dampen_ramp_start)
                / (k_dampen_ramp_end - k_dampen_ramp_start));
        if dampen < 0.0 {
            dampen = 0.0;
        }
    }
    let mul = scale * dampen;
    let add = (1.0 - dampen) * base_level;

    // SIMD constants for gamma modulation
    let bias_v = f32x8::splat(token, GM_K_BIAS);
    let zero_v = f32x8::splat(token, 0.0);
    let eps_v = f32x8::splat(token, EPSILON);
    let k_num_v = f32x8::splat(token, K_NUM_MUL);
    let k_den_v = f32x8::splat(token, K_DEN_MUL);
    let k_voff_v = f32x8::splat(token, K_V_OFFSET);

    // HF modulation constants
    let valmin_v = f32x8::splat(token, HF_VALMIN_Y);

    // Blue modulation constants
    let bm_offset_v = f32x8::splat(token, BM_K_OFFSET);
    let bm_limit_v = f32x8::splat(token, BM_K_LIMIT);

    for iy in 0..rect_h_blocks {
        let block_iy = rect_y0_blocks + iy;
        let py = block_iy * 8;
        for ix in 0..rect_w_blocks {
            let block_ix = rect_x0_blocks + ix;
            let px = block_ix * 8;

            let mut out_val = aq_map[iy * aq_map_stride + ix];

            // ComputeMask (scalar — single value, not vectorizable)
            out_val = compute_mask_scalar(out_val);

            // GammaModulation: accumulate over 8×8 = 64 pixels
            // Process 8 pixels at a time (one row)
            let mut ratio_acc = f32x8::splat(token, 0.0);
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                let y_vals = crate::load_f32x8(token, xyb_y, row_off);
                let x_vals = crate::load_f32x8(token, xyb_x, row_off);
                let iny = y_vals + bias_v;
                let r = (iny - x_vals).max(zero_v);
                let g = (iny + x_vals).max(zero_v);
                // ratio_of_deriv_inverted(r): num/den
                let r2 = r * r;
                let r3 = r2 * r;
                let r_num = k_num_v.mul_add(r2, eps_v);
                let r_den = k_den_v.mul_add(r3, k_voff_v);
                let g2 = g * g;
                let g3 = g2 * g;
                let g_num = k_num_v.mul_add(g2, eps_v);
                let g_den = k_den_v.mul_add(g3, k_voff_v);
                ratio_acc = ratio_acc + r_num / r_den + g_num / g_den;
            }
            // Horizontal sum of ratio_acc (8 partial sums → 1 total)
            let ratio_arr: [f32; 8] = ratio_acc.into();
            let overall_ratio: f32 = ratio_arr.iter().sum::<f32>() * (0.5 / 64.0);
            out_val += GM_K_GAMMA * fast_log2f(overall_ratio);

            // HfModulation: accumulate abs-diff with right and below neighbors
            let mask_val = out_val;
            let mut hf_acc = f32x8::splat(token, 0.0);
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                let next_row_off = if dy == 7 {
                    row_off
                } else {
                    (py + dy + 1) * stride + px
                };
                let cur = crate::load_f32x8(token, xyb_y, row_off);
                let below = crate::load_f32x8(token, xyb_y, next_row_off);
                // Below neighbor: abs(cur - below).min(valmin)
                hf_acc = hf_acc + (cur - below).abs().min(valmin_v);
                // Right neighbor: shift cur left by 1, duplicate last element.
                // Position 7 (dx=7) has no right neighbor, so it contributes 0.
                // Duplicating cur[7] at position 7 makes diff=0 → abs(0).min(valmin)=0.
                let cur_arr: [f32; 8] = cur.into();
                let right = f32x8::from_array(
                    token,
                    [
                        cur_arr[1], cur_arr[2], cur_arr[3], cur_arr[4], cur_arr[5], cur_arr[6],
                        cur_arr[7], cur_arr[7],
                    ],
                );
                hf_acc = hf_acc + (cur - right).abs().min(valmin_v);
            }
            let hf_arr: [f32; 8] = hf_acc.into();
            let sum_y: f32 = hf_arr.iter().sum();
            let after_hf = mask_val + sum_y * HF_K_MUL_Y + HF_K_OFFSET;

            // BlueModulation: conditional accumulate
            let mut blue_acc = f32x8::splat(token, 0.0);
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                let x_vals = crate::load_f32x8(token, xyb_x, row_off);
                let y_vals = crate::load_f32x8(token, xyb_y, row_off);
                let b_vals = crate::load_f32x8(token, xyb_b, row_off);
                let y_eff = y_vals + bm_offset_v + x_vals.abs();
                let excess = (b_vals - y_eff).min(bm_limit_v).max(zero_v);
                blue_acc = blue_acc + excess;
            }
            let blue_arr: [f32; 8] = blue_acc.into();
            let mut sum: f32 = blue_arr.iter().sum();
            if sum >= 32.0 * BM_K_LIMIT {
                sum = 64.0 * BM_K_LIMIT - sum;
            }
            if sum >= BM_K_MAX_LIMIT * BM_K_LIMIT {
                sum = BM_K_MAX_LIMIT * BM_K_LIMIT;
            }
            sum *= BM_K_MUL;
            let after_blue = mask_val + sum;

            out_val = after_hf.min(after_blue);
            aq_map[iy * aq_map_stride + ix] = fast_pow2f(out_val * 1.442_695) * mul + add;
        }
    }
}

// ============================================================================
// aarch64 NEON implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn compute_pre_erosion_neon(
    token: archmage::NeonToken,
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
) -> (Vec<f32>, usize, usize) {
    use magetypes::simd::f32x4;

    let x0 = tile_x0.saturating_sub(4);
    let x1 = if tile_x1 < width {
        tile_x1 + 4
    } else {
        tile_x1
    };
    let y_start = tile_y0.saturating_sub(4);
    let y_end = if tile_y1 < height {
        tile_y1 + 4
    } else {
        tile_y1
    };

    let diff_width = x1 - x0;
    let pre_erosion_w = diff_width / 4;
    let pre_erosion_h = (y_end - y_start) / 4;

    // Pad diff_buffer to next multiple of 4 for NEON loads/stores
    let diff_buffer_len = (diff_width + 3) & !3;
    let mut diff_buffer = vec![0.0_f32; diff_buffer_len];
    let mut pre_erosion = vec![0.0_f32; pre_erosion_w * pre_erosion_h];

    let max_x = width - 1;
    let max_y = height - 1;

    let quarter = f32x4::splat(token, 0.25);
    let gamma_off = f32x4::splat(token, MATCH_GAMMA_OFFSET);
    let zero = f32x4::splat(token, 0.0);
    let eps_v = f32x4::splat(token, EPSILON);
    let k_num_mul_v = f32x4::splat(token, K_NUM_MUL);
    let k_den_mul_v = f32x4::splat(token, K_DEN_MUL);
    let k_v_offset_v = f32x4::splat(token, K_V_OFFSET);
    let limit_v = f32x4::splat(token, LIMIT);
    let masking_mul_v = f32x4::splat(token, (MASKING_K_MUL * 1e8_f32).sqrt());
    let masking_offset_v = f32x4::splat(token, MASKING_K_LOG_OFFSET);
    let masking_scale = f32x4::splat(token, 0.25);

    for y in y_start..y_end {
        let yc = y.min(max_y);
        let y2 = (y + 1).min(max_y);
        let y1 = if y > 0 { (y - 1).min(max_y) } else { 0 };

        let interior_start = if x0 == 0 { 1 } else { 0 };

        // Scalar edges
        for local_x in 0..interior_start.min(diff_width) {
            let x = x0 + local_x;
            let xc = x.min(max_x);
            let x2 = (x + 1).min(max_x);
            let x1_local = if x > 0 { (x - 1).min(max_x) } else { 0 };
            let val = pre_erosion_pixel(xyb_y, width, yc, y1, y2, xc, x1_local, x2);
            if (y - y_start) % 4 != 0 {
                diff_buffer[local_x] += val;
            } else {
                diff_buffer[local_x] = val;
            }
        }

        let mut local_x = interior_start;
        let simd_end = if diff_width > 4 + interior_start {
            diff_width - 3
        } else {
            interior_start
        };

        while local_x < simd_end {
            let x = x0 + local_x;
            let xc = x.min(max_x);

            let top = f32x4::from_slice(token, &xyb_y[y1 * width + xc..]);
            let bot = f32x4::from_slice(token, &xyb_y[y2 * width + xc..]);
            let left = f32x4::from_slice(token, &xyb_y[yc * width + xc - 1..]);
            let right = f32x4::from_slice(token, &xyb_y[yc * width + xc + 1..]);
            let center = f32x4::from_slice(token, &xyb_y[yc * width + xc..]);

            let base = quarter * (top + bot + left + right);

            let v = (center + gamma_off).max(zero);
            let v2 = v * v;
            let v3 = v2 * v;
            let num = k_num_mul_v.mul_add(v2, eps_v);
            let den = k_den_mul_v.mul_add(v3, k_v_offset_v);
            let gammac = den / num;

            let raw_diff = gammac * (center - base);
            let diff_sq = raw_diff * raw_diff;
            let clamped = diff_sq.min(limit_v);
            let result = masking_scale * (clamped.mul_add(masking_mul_v, masking_offset_v)).sqrt();

            if (y - y_start) % 4 != 0 {
                let existing = f32x4::from_slice(token, &diff_buffer[local_x..]);
                let sum = existing + result;
                let out: &mut [f32; 4] =
                    (&mut diff_buffer[local_x..local_x + 4]).try_into().unwrap();
                sum.store(out);
            } else {
                let out: &mut [f32; 4] =
                    (&mut diff_buffer[local_x..local_x + 4]).try_into().unwrap();
                result.store(out);
            }

            local_x += 4;
        }

        // Scalar remainder
        while local_x < diff_width {
            let x = x0 + local_x;
            let xc = x.min(max_x);
            let x2 = (x + 1).min(max_x);
            let x1_local = if x > 0 { (x - 1).min(max_x) } else { 0 };
            let val = pre_erosion_pixel(xyb_y, width, yc, y1, y2, xc, x1_local, x2);
            if (y - y_start) % 4 != 0 {
                diff_buffer[local_x] += val;
            } else {
                diff_buffer[local_x] = val;
            }
            local_x += 1;
        }

        if (y - y_start) % 4 == 3 {
            let row_y = (y - y_start) / 4;
            for bx in 0..pre_erosion_w {
                let sum = diff_buffer[bx * 4]
                    + diff_buffer[bx * 4 + 1]
                    + diff_buffer[bx * 4 + 2]
                    + diff_buffer[bx * 4 + 3];
                pre_erosion[row_y * pre_erosion_w + bx] = sum * 0.25;
            }
        }
    }

    (pre_erosion, pre_erosion_w, pre_erosion_h)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[archmage::arcane]
pub fn per_block_modulations_neon(
    token: archmage::NeonToken,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    scale: f32,
    rect_x0_blocks: usize,
    rect_y0_blocks: usize,
    rect_w_blocks: usize,
    rect_h_blocks: usize,
    aq_map: &mut [f32],
    aq_map_stride: usize,
) {
    use magetypes::simd::f32x4;

    let base_level = 0.48 * scale;
    let k_dampen_ramp_start = 2.0_f32;
    let k_dampen_ramp_end = 14.0_f32;
    let mut dampen = 1.0_f32;
    if butteraugli_target >= k_dampen_ramp_start {
        dampen = 1.0
            - ((butteraugli_target - k_dampen_ramp_start)
                / (k_dampen_ramp_end - k_dampen_ramp_start));
        if dampen < 0.0 {
            dampen = 0.0;
        }
    }
    let mul = scale * dampen;
    let add = (1.0 - dampen) * base_level;

    let bias_v = f32x4::splat(token, GM_K_BIAS);
    let zero_v = f32x4::splat(token, 0.0);
    let eps_v = f32x4::splat(token, EPSILON);
    let k_num_v = f32x4::splat(token, K_NUM_MUL);
    let k_den_v = f32x4::splat(token, K_DEN_MUL);
    let k_voff_v = f32x4::splat(token, K_V_OFFSET);
    let valmin_v = f32x4::splat(token, HF_VALMIN_Y);
    let bm_offset_v = f32x4::splat(token, BM_K_OFFSET);
    let bm_limit_v = f32x4::splat(token, BM_K_LIMIT);

    for iy in 0..rect_h_blocks {
        let block_iy = rect_y0_blocks + iy;
        let py = block_iy * 8;
        for ix in 0..rect_w_blocks {
            let block_ix = rect_x0_blocks + ix;
            let px = block_ix * 8;

            let mut out_val = aq_map[iy * aq_map_stride + ix];
            out_val = compute_mask_scalar(out_val);

            // GammaModulation with f32x4: two 4-wide loads per row (8 pixels = 2 NEON loads)
            let mut ratio_total = 0.0_f32;
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                for half in 0..2 {
                    let off = row_off + half * 4;
                    let y_vals = f32x4::from_slice(token, &xyb_y[off..]);
                    let x_vals = f32x4::from_slice(token, &xyb_x[off..]);
                    let iny = y_vals + bias_v;
                    let r = (iny - x_vals).max(zero_v);
                    let g = (iny + x_vals).max(zero_v);
                    let r2 = r * r;
                    let g2 = g * g;
                    let r_num = k_num_v.mul_add(r2, eps_v);
                    let r_den = k_den_v.mul_add(r2 * r, k_voff_v);
                    let g_num = k_num_v.mul_add(g2, eps_v);
                    let g_den = k_den_v.mul_add(g2 * g, k_voff_v);
                    let ratios = r_num / r_den + g_num / g_den;
                    let arr: [f32; 4] = ratios.into();
                    ratio_total += arr[0] + arr[1] + arr[2] + arr[3];
                }
            }
            let overall_ratio = ratio_total * (0.5 / 64.0);
            out_val += GM_K_GAMMA * fast_log2f(overall_ratio);

            // HfModulation with f32x4
            let mask_val = out_val;
            let mut hf_total = 0.0_f32;
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                let next_row_off = if dy == 7 {
                    row_off
                } else {
                    (py + dy + 1) * stride + px
                };
                for half in 0..2 {
                    let off = row_off + half * 4;
                    let cur = f32x4::from_slice(token, &xyb_y[off..]);
                    let below = f32x4::from_slice(token, &xyb_y[next_row_off + half * 4..]);
                    let bd = (cur - below).abs().min(valmin_v);
                    let bd_arr: [f32; 4] = bd.into();
                    hf_total += bd_arr[0] + bd_arr[1] + bd_arr[2] + bd_arr[3];

                    // Right neighbor: shift cur left by 1 to avoid OOB reads.
                    // Last position in each half-group doesn't contribute right-diff:
                    // half=0: all 4 have right neighbor (dx=0..3, right=1..4)
                    // half=1: dx=7 has no right neighbor (duplicated → diff=0)
                    let cur_arr: [f32; 4] = cur.into();
                    let right = if half == 0 {
                        f32x4::from_slice(token, &xyb_y[off + 1..])
                    } else {
                        // Shift left, duplicate last element (diff=0 for dx=7)
                        f32x4::from_array(token, [cur_arr[1], cur_arr[2], cur_arr[3], cur_arr[3]])
                    };
                    let rd = (cur - right).abs().min(valmin_v);
                    let rd_arr: [f32; 4] = rd.into();
                    hf_total += rd_arr[0] + rd_arr[1] + rd_arr[2];
                    if half == 0 {
                        hf_total += rd_arr[3]; // dx=3 has right neighbor (dx=4)
                    }
                    // half=1: rd_arr[3] = 0 (duplicated element) — skip it
                }
            }
            let after_hf = mask_val + hf_total * HF_K_MUL_Y + HF_K_OFFSET;

            // BlueModulation with f32x4
            let mut blue_total = 0.0_f32;
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                for half in 0..2 {
                    let off = row_off + half * 4;
                    let x_vals = f32x4::from_slice(token, &xyb_x[off..]);
                    let y_vals = f32x4::from_slice(token, &xyb_y[off..]);
                    let b_vals = f32x4::from_slice(token, &xyb_b[off..]);
                    let y_eff = y_vals + bm_offset_v + x_vals.abs();
                    let excess = (b_vals - y_eff).min(bm_limit_v).max(zero_v);
                    let arr: [f32; 4] = excess.into();
                    blue_total += arr[0] + arr[1] + arr[2] + arr[3];
                }
            }
            let mut sum = blue_total;
            if sum >= 32.0 * BM_K_LIMIT {
                sum = 64.0 * BM_K_LIMIT - sum;
            }
            if sum >= BM_K_MAX_LIMIT * BM_K_LIMIT {
                sum = BM_K_MAX_LIMIT * BM_K_LIMIT;
            }
            sum *= BM_K_MUL;
            let after_blue = mask_val + sum;

            out_val = after_hf.min(after_blue);
            aq_map[iy * aq_map_stride + ix] = fast_pow2f(out_val * 1.442_695) * mul + add;
        }
    }
}

// ============================================================================
// wasm32 SIMD128 implementation
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn compute_pre_erosion_wasm128(
    token: archmage::Wasm128Token,
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
) -> (Vec<f32>, usize, usize) {
    use magetypes::simd::f32x4;

    let x0 = tile_x0.saturating_sub(4);
    let x1 = if tile_x1 < width {
        tile_x1 + 4
    } else {
        tile_x1
    };
    let y_start = tile_y0.saturating_sub(4);
    let y_end = if tile_y1 < height {
        tile_y1 + 4
    } else {
        tile_y1
    };

    let diff_width = x1 - x0;
    let pre_erosion_w = diff_width / 4;
    let pre_erosion_h = (y_end - y_start) / 4;

    // Pad diff_buffer to next multiple of 4 for SIMD loads/stores
    let diff_buffer_len = (diff_width + 3) & !3;
    let mut diff_buffer = vec![0.0_f32; diff_buffer_len];
    let mut pre_erosion = vec![0.0_f32; pre_erosion_w * pre_erosion_h];

    let max_x = width - 1;
    let max_y = height - 1;

    let quarter = f32x4::splat(token, 0.25);
    let gamma_off = f32x4::splat(token, MATCH_GAMMA_OFFSET);
    let zero = f32x4::splat(token, 0.0);
    let eps_v = f32x4::splat(token, EPSILON);
    let k_num_mul_v = f32x4::splat(token, K_NUM_MUL);
    let k_den_mul_v = f32x4::splat(token, K_DEN_MUL);
    let k_v_offset_v = f32x4::splat(token, K_V_OFFSET);
    let limit_v = f32x4::splat(token, LIMIT);
    let masking_mul_v = f32x4::splat(token, (MASKING_K_MUL * 1e8_f32).sqrt());
    let masking_offset_v = f32x4::splat(token, MASKING_K_LOG_OFFSET);
    let masking_scale = f32x4::splat(token, 0.25);

    for y in y_start..y_end {
        let yc = y.min(max_y);
        let y2 = (y + 1).min(max_y);
        let y1 = if y > 0 { (y - 1).min(max_y) } else { 0 };

        let interior_start = if x0 == 0 { 1 } else { 0 };

        // Scalar edges
        for local_x in 0..interior_start.min(diff_width) {
            let x = x0 + local_x;
            let xc = x.min(max_x);
            let x2 = (x + 1).min(max_x);
            let x1_local = if x > 0 { (x - 1).min(max_x) } else { 0 };
            let val = pre_erosion_pixel(xyb_y, width, yc, y1, y2, xc, x1_local, x2);
            if (y - y_start) % 4 != 0 {
                diff_buffer[local_x] += val;
            } else {
                diff_buffer[local_x] = val;
            }
        }

        let mut local_x = interior_start;
        let simd_end = if diff_width > 4 + interior_start {
            diff_width - 3
        } else {
            interior_start
        };

        while local_x < simd_end {
            let x = x0 + local_x;
            let xc = x.min(max_x);

            let top = f32x4::from_slice(token, &xyb_y[y1 * width + xc..]);
            let bot = f32x4::from_slice(token, &xyb_y[y2 * width + xc..]);
            let left = f32x4::from_slice(token, &xyb_y[yc * width + xc - 1..]);
            let right = f32x4::from_slice(token, &xyb_y[yc * width + xc + 1..]);
            let center = f32x4::from_slice(token, &xyb_y[yc * width + xc..]);

            let base = quarter * (top + bot + left + right);

            let v = (center + gamma_off).max(zero);
            let v2 = v * v;
            let v3 = v2 * v;
            let num = k_num_mul_v.mul_add(v2, eps_v);
            let den = k_den_mul_v.mul_add(v3, k_v_offset_v);
            let gammac = den / num;

            let raw_diff = gammac * (center - base);
            let diff_sq = raw_diff * raw_diff;
            let clamped = diff_sq.min(limit_v);
            let result = masking_scale * (clamped.mul_add(masking_mul_v, masking_offset_v)).sqrt();

            if (y - y_start) % 4 != 0 {
                let existing = f32x4::from_slice(token, &diff_buffer[local_x..]);
                let sum = existing + result;
                let out: &mut [f32; 4] =
                    (&mut diff_buffer[local_x..local_x + 4]).try_into().unwrap();
                sum.store(out);
            } else {
                let out: &mut [f32; 4] =
                    (&mut diff_buffer[local_x..local_x + 4]).try_into().unwrap();
                result.store(out);
            }

            local_x += 4;
        }

        // Scalar remainder
        while local_x < diff_width {
            let x = x0 + local_x;
            let xc = x.min(max_x);
            let x2 = (x + 1).min(max_x);
            let x1_local = if x > 0 { (x - 1).min(max_x) } else { 0 };
            let val = pre_erosion_pixel(xyb_y, width, yc, y1, y2, xc, x1_local, x2);
            if (y - y_start) % 4 != 0 {
                diff_buffer[local_x] += val;
            } else {
                diff_buffer[local_x] = val;
            }
            local_x += 1;
        }

        if (y - y_start) % 4 == 3 {
            let row_y = (y - y_start) / 4;
            for bx in 0..pre_erosion_w {
                let sum = diff_buffer[bx * 4]
                    + diff_buffer[bx * 4 + 1]
                    + diff_buffer[bx * 4 + 2]
                    + diff_buffer[bx * 4 + 3];
                pre_erosion[row_y * pre_erosion_w + bx] = sum * 0.25;
            }
        }
    }

    (pre_erosion, pre_erosion_w, pre_erosion_h)
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
#[archmage::arcane]
pub fn per_block_modulations_wasm128(
    token: archmage::Wasm128Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    scale: f32,
    rect_x0_blocks: usize,
    rect_y0_blocks: usize,
    rect_w_blocks: usize,
    rect_h_blocks: usize,
    aq_map: &mut [f32],
    aq_map_stride: usize,
) {
    use magetypes::simd::f32x4;

    let base_level = 0.48 * scale;
    let k_dampen_ramp_start = 2.0_f32;
    let k_dampen_ramp_end = 14.0_f32;
    let mut dampen = 1.0_f32;
    if butteraugli_target >= k_dampen_ramp_start {
        dampen = 1.0
            - ((butteraugli_target - k_dampen_ramp_start)
                / (k_dampen_ramp_end - k_dampen_ramp_start));
        if dampen < 0.0 {
            dampen = 0.0;
        }
    }
    let mul = scale * dampen;
    let add = (1.0 - dampen) * base_level;

    let bias_v = f32x4::splat(token, GM_K_BIAS);
    let zero_v = f32x4::splat(token, 0.0);
    let eps_v = f32x4::splat(token, EPSILON);
    let k_num_v = f32x4::splat(token, K_NUM_MUL);
    let k_den_v = f32x4::splat(token, K_DEN_MUL);
    let k_voff_v = f32x4::splat(token, K_V_OFFSET);
    let valmin_v = f32x4::splat(token, HF_VALMIN_Y);
    let bm_offset_v = f32x4::splat(token, BM_K_OFFSET);
    let bm_limit_v = f32x4::splat(token, BM_K_LIMIT);

    for iy in 0..rect_h_blocks {
        let block_iy = rect_y0_blocks + iy;
        let py = block_iy * 8;
        for ix in 0..rect_w_blocks {
            let block_ix = rect_x0_blocks + ix;
            let px = block_ix * 8;

            let mut out_val = aq_map[iy * aq_map_stride + ix];
            out_val = compute_mask_scalar(out_val);

            // GammaModulation with f32x4: two 4-wide loads per row (8 pixels = 2 SIMD loads)
            let mut ratio_total = 0.0_f32;
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                for half in 0..2 {
                    let off = row_off + half * 4;
                    let y_vals = f32x4::from_slice(token, &xyb_y[off..]);
                    let x_vals = f32x4::from_slice(token, &xyb_x[off..]);
                    let iny = y_vals + bias_v;
                    let r = (iny - x_vals).max(zero_v);
                    let g = (iny + x_vals).max(zero_v);
                    let r2 = r * r;
                    let g2 = g * g;
                    let r_num = k_num_v.mul_add(r2, eps_v);
                    let r_den = k_den_v.mul_add(r2 * r, k_voff_v);
                    let g_num = k_num_v.mul_add(g2, eps_v);
                    let g_den = k_den_v.mul_add(g2 * g, k_voff_v);
                    let ratios = r_num / r_den + g_num / g_den;
                    let arr: [f32; 4] = ratios.into();
                    ratio_total += arr[0] + arr[1] + arr[2] + arr[3];
                }
            }
            let overall_ratio = ratio_total * (0.5 / 64.0);
            out_val += GM_K_GAMMA * fast_log2f(overall_ratio);

            // HfModulation with f32x4
            let mask_val = out_val;
            let mut hf_total = 0.0_f32;
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                let next_row_off = if dy == 7 {
                    row_off
                } else {
                    (py + dy + 1) * stride + px
                };
                for half in 0..2 {
                    let off = row_off + half * 4;
                    let cur = f32x4::from_slice(token, &xyb_y[off..]);
                    let below = f32x4::from_slice(token, &xyb_y[next_row_off + half * 4..]);
                    let bd = (cur - below).abs().min(valmin_v);
                    let bd_arr: [f32; 4] = bd.into();
                    hf_total += bd_arr[0] + bd_arr[1] + bd_arr[2] + bd_arr[3];

                    let cur_arr: [f32; 4] = cur.into();
                    let right = if half == 0 {
                        f32x4::from_slice(token, &xyb_y[off + 1..])
                    } else {
                        // Shift left, duplicate last element (diff=0 for dx=7)
                        f32x4::from_array(token, [cur_arr[1], cur_arr[2], cur_arr[3], cur_arr[3]])
                    };
                    let rd = (cur - right).abs().min(valmin_v);
                    let rd_arr: [f32; 4] = rd.into();
                    hf_total += rd_arr[0] + rd_arr[1] + rd_arr[2];
                    if half == 0 {
                        hf_total += rd_arr[3]; // dx=3 has right neighbor (dx=4)
                    }
                    // half=1: rd_arr[3] = 0 (duplicated element) — skip it
                }
            }
            let after_hf = mask_val + hf_total * HF_K_MUL_Y + HF_K_OFFSET;

            // BlueModulation with f32x4
            let mut blue_total = 0.0_f32;
            for dy in 0..8 {
                let row_off = (py + dy) * stride + px;
                for half in 0..2 {
                    let off = row_off + half * 4;
                    let x_vals = f32x4::from_slice(token, &xyb_x[off..]);
                    let y_vals = f32x4::from_slice(token, &xyb_y[off..]);
                    let b_vals = f32x4::from_slice(token, &xyb_b[off..]);
                    let y_eff = y_vals + bm_offset_v + x_vals.abs();
                    let excess = (b_vals - y_eff).min(bm_limit_v).max(zero_v);
                    let arr: [f32; 4] = excess.into();
                    blue_total += arr[0] + arr[1] + arr[2] + arr[3];
                }
            }
            let mut sum = blue_total;
            if sum >= 32.0 * BM_K_LIMIT {
                sum = 64.0 * BM_K_LIMIT - sum;
            }
            if sum >= BM_K_MAX_LIMIT * BM_K_LIMIT {
                sum = BM_K_MAX_LIMIT * BM_K_LIMIT;
            }
            sum *= BM_K_MUL;
            let after_blue = mask_val + sum;

            out_val = after_hf.min(after_blue);
            aq_map[iy * aq_map_stride + ix] = fast_pow2f(out_val * 1.442_695) * mul + add;
        }
    }
}

// ============================================================================
// Shared scalar helper for pre-erosion
// ============================================================================

#[inline(always)]
fn pre_erosion_pixel(
    xyb_y: &[f32],
    width: usize,
    yc: usize,
    y1: usize,
    y2: usize,
    xc: usize,
    x1: usize,
    x2: usize,
) -> f32 {
    let base = 0.25
        * (xyb_y[y2 * width + xc]
            + xyb_y[y1 * width + xc]
            + xyb_y[yc * width + x1]
            + xyb_y[yc * width + x2]);
    let gammac = ratio_of_deriv_normal(xyb_y[yc * width + xc] + MATCH_GAMMA_OFFSET);
    let mut diff = gammac * (xyb_y[yc * width + xc] - base);
    diff *= diff;
    if diff >= LIMIT {
        diff = LIMIT;
    }
    masking_sqrt(diff)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn test_pre_erosion_scalar_vs_dispatch() {
        let w = 64;
        let h = 64;
        let xyb_y: Vec<f32> = (0..w * h).map(|i| 0.1 + 0.001 * (i % 37) as f32).collect();

        let (ref_out, ref_w, ref_h) = compute_pre_erosion_scalar(&xyb_y, w, h, 0, 0, w, h);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let (test_out, test_w, test_h) = compute_pre_erosion(&xyb_y, w, h, 0, 0, w, h);
                assert_eq!(ref_w, test_w, "[{perm}]");
                assert_eq!(ref_h, test_h, "[{perm}]");
                for i in 0..ref_out.len() {
                    let diff = (ref_out[i] - test_out[i]).abs();
                    assert!(
                        diff < 1e-4,
                        "pre_erosion mismatch at {i}: scalar={} dispatch={} [{perm}]",
                        ref_out[i],
                        test_out[i]
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_per_block_modulations_scalar_vs_dispatch() {
        let w = 32;
        let h = 32;
        let stride = w;
        let n = w * h;
        let xyb_x: Vec<f32> = (0..n)
            .map(|i| 0.01 * ((i * 7) % 13) as f32 - 0.05)
            .collect();
        let xyb_y: Vec<f32> = (0..n).map(|i| 0.2 + 0.01 * ((i * 3) % 17) as f32).collect();
        let xyb_b: Vec<f32> = (0..n)
            .map(|i| 0.15 + 0.01 * ((i * 11) % 19) as f32)
            .collect();

        let xb = w / 8;
        let yb = h / 8;

        let mut aq_ref = vec![0.5_f32; xb * yb];
        per_block_modulations_scalar(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            stride,
            1.0,
            0.765,
            0,
            0,
            xb,
            yb,
            &mut aq_ref,
            xb,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut aq_test = vec![0.5_f32; xb * yb];
                per_block_modulations(
                    &xyb_x,
                    &xyb_y,
                    &xyb_b,
                    stride,
                    1.0,
                    0.765,
                    0,
                    0,
                    xb,
                    yb,
                    &mut aq_test,
                    xb,
                );
                for i in 0..aq_ref.len() {
                    let diff = (aq_ref[i] - aq_test[i]).abs();
                    assert!(
                        diff < 1e-3,
                        "modulations mismatch at {i}: scalar={} dispatch={} [{perm}]",
                        aq_ref[i],
                        aq_test[i]
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }
}
