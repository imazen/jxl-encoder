// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Adaptive quantization field computation.
//!
//! Ported from libjxl-tiny `enc_adaptive_quantization.cc`.

// Ported float constants from C++ - exact values are intentional for parity.
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
//! Computes per-block quantization values based on perceptual masking.
//!
//! Pipeline:
//! 1. `compute_pre_erosion()` — Y + kXMul×X channel diffs, gamma ratio, masking sqrt, 4x downsample
//! 2. `fuzzy_erosion()` — 3×3 min-4 weighted sum, 2x downsample
//! 3. `per_block_modulations()` — ComputeMask + HfModulation + ColorModulation + GammaModulation + exp2
//! 4. Convert: `raw_quant = clamp(round(quant_field * inv_scale + 0.5), 1, 255)`

use super::common::clamp;

// --- Fast math approximations (ported from fast_math-inl.h) ---

/// Fast base-2 logarithm approximation. L1 error ~3.9E-6.
///
/// Uses rational polynomial approximation of log1p(x)/log(2).
fn fast_log2f(x: f32) -> f32 {
    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits.wrapping_sub(0x3f2aaaab_u32 as i32); // subtract 2/3
    let exp_shifted = exp_bits >> 23;
    let mantissa = f32::from_bits((x_bits.wrapping_sub(exp_shifted << 23)) as u32);
    let exp_val = exp_shifted as f32;

    let frac = mantissa - 1.0;

    // Rational polynomial coefficients (degree 2/2)
    let p0 = -1.850_383_3e-6_f32;
    let p1 = 1.428_716_f32;
    let p2 = 0.742_458_7_f32;

    let q0 = 0.990_328_14_f32;
    let q1 = 1.009_671_9_f32;
    let q2 = 0.174_093_43_f32;

    let num = p0 + frac * (p1 + frac * p2);
    let den = q0 + frac * (q1 + frac * q2);

    num / den + exp_val
}

/// Fast base-2 power approximation. Max relative error ~3e-7.
fn fast_pow2f(x: f32) -> f32 {
    let floorx = x.floor();
    let exp = f32::from_bits(((floorx as i32 + 127) << 23) as u32);
    let frac = x - floorx;

    let num = frac + 1.017_490_63e+01;
    let num = num * frac + 4.886_877_98e+01;
    let num = num * frac + 9.855_065_91e+01;
    let num = num * exp;

    let den = frac * 2.102_429_58e-01 + (-2.223_288_56e-02);
    let den = den * frac + (-1.944_149_9e+01);
    let den = den * frac + 9.855_066_33e+01;

    num / den
}

// --- SimpleGamma constants ---

const SG_MUL: f32 = 226.048_045_f32;
const SG_MUL2: f32 = 1.0 / 73.377_13_f32;
const K_LOG2: f32 = 0.693_147_2_f32;
const SG_RET_MUL: f32 = SG_MUL2 * 18.658_093_f32 * K_LOG2;
const SG_V_OFFSET: f32 = 7.146_724_7_f32;

/// Ratio of derivatives of cubic root to simple gamma.
///
/// Maps from opsin (cubic root of photons) space to butteraugli's
/// log-gamma psychovisual space.
///
/// When `invert` is false: returns den/num (used in pre-erosion).
/// When `invert` is true: returns num/den (used in gamma modulation).
fn ratio_of_derivatives(v: f32, invert: bool) -> f32 {
    let epsilon = 1e-2_f32;
    let v = v.max(0.0);

    let k_num_mul = SG_RET_MUL * 3.0 * SG_MUL;
    let k_v_offset = SG_V_OFFSET * K_LOG2 + epsilon;
    let k_den_mul = K_LOG2 * SG_MUL;

    let v2 = v * v;

    let num = k_num_mul * v2 + epsilon;
    let den = k_den_mul * v * v2 + k_v_offset;

    if invert { num / den } else { den / num }
}

// --- Masking ---

/// MaskingSqrt: converts accumulated diff values through masking function.
fn masking_sqrt(v: f32) -> f32 {
    const K_LOG_OFFSET: f32 = 26.481_47_f32;
    const K_MUL: f32 = 211.507_6_f32;
    let mul_v = K_MUL * 1e8;
    0.25 * (v * mul_v.sqrt() + K_LOG_OFFSET).sqrt()
}

/// Insert `v` into the smallest-4 tracking variables if it's smaller than `min3`.
#[inline(always)]
fn store_min4(v: f32, min0: &mut f32, min1: &mut f32, min2: &mut f32, min3: &mut f32) {
    if v < *min3 {
        if v < *min0 {
            *min3 = *min2;
            *min2 = *min1;
            *min1 = *min0;
            *min0 = v;
        } else if v < *min1 {
            *min3 = *min2;
            *min2 = *min1;
            *min1 = v;
        } else if v < *min2 {
            *min3 = *min2;
            *min2 = v;
        } else {
            *min3 = v;
        }
    }
}

/// ComputeMask: modulates exponent based on out_val.
fn compute_mask(out_val: f32) -> f32 {
    const K_BASE: f32 = -0.741_749_93_f32;
    const K_MUL4: f32 = 3.235_325_7_f32;
    const K_MUL2: f32 = 12.906_028_f32;
    const K_OFFSET2: f32 = 305.040_36_f32;
    const K_MUL3: f32 = 5.022_031_3_f32;
    const K_OFFSET3: f32 = 2.192_574_f32;
    const K_OFFSET4: f32 = 0.25 * K_OFFSET3;
    const K_MUL0: f32 = 0.747_604_22_f32;

    // Avoid division by zero
    let v1 = (out_val * K_MUL0).max(1e-3);
    let v2 = 1.0 / (v1 + K_OFFSET2);
    let v3 = 1.0 / (v1 * v1 + K_OFFSET3);
    let v4 = 1.0 / (v1 * v1 + K_OFFSET4);

    K_BASE + K_MUL4 * v4 + K_MUL2 * v2 + K_MUL3 * v3
}

/// HfModulation: adjust quantization based on high-frequency content.
///
/// Computes sum of absolute pixel differences (right neighbor + below neighbor)
/// in the Y channel over an 8×8 block.
///
/// The buffer must be padded to at least (y+8) rows and (x+8) columns with
/// edge-replicated values, so no bounds checking is needed.
fn hf_modulation(x: usize, y: usize, xyb_y: &[f32], stride: usize, out_val: f32) -> f32 {
    let mut sum = 0.0_f32;

    for dy in 0..8 {
        let py = y + dy;
        // For dy < 7, the below-neighbor is py+1 which is within the 8-row block.
        // For dy == 7, we use py itself (no below-neighbor at block edge).
        let py_next = if dy == 7 { py } else { py + 1 };

        for dx in 0..8 {
            let px = x + dx;
            let p = xyb_y[py * stride + px];

            // Right neighbor difference (skip last column)
            if dx < 7 {
                sum += (p - xyb_y[py * stride + px + 1]).abs();
            }

            // Below neighbor difference
            let pd = xyb_y[py_next * stride + px];
            sum += (p - pd).abs();
        }
    }

    // -2.0052193233688884 / 112 ≈ -0.017903
    out_val + sum * (-2.005_219_3_f32 / 112.0)
}

/// ColorModulation: adjust quantization based on color content (red/blue coverage).
///
/// The buffer must be padded to at least (y+8) rows and (x+8) columns with
/// edge-replicated values, so no bounds checking is needed.
#[allow(clippy::too_many_arguments)]
fn color_modulation(
    x: usize,
    y: usize,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    out_val: f32,
) -> f32 {
    const K_STRENGTH_MUL: f32 = 2.177_823_4_f32;
    const K_RED_RAMP_START: f32 = 0.007_320_014_f32;
    const K_RED_RAMP_LENGTH: f32 = 0.019_421_556_f32;
    const K_BLUE_RAMP_LENGTH: f32 = 0.086_890_61_f32;
    const K_BLUE_RAMP_START: f32 = 0.269_734_2_f32;

    let strength = K_STRENGTH_MUL * (1.0 - 0.25 * butteraugli_target);
    if strength < 0.0 {
        return out_val;
    }

    let red_strength = strength * 5.992_297_8_f32;
    let blue_strength = strength;

    // Offset: reduce bits from areas not blue or red
    let offset = strength * -0.009_174_542_f32;
    let result = out_val + offset;

    let mut blue_coverage = 0.0_f32;
    let mut red_coverage = 0.0_f32;

    for dy in 0..8 {
        let py = y + dy;
        for dx in 0..8 {
            let px = x + dx;
            let idx = py * stride + px;
            let pixel_x = (xyb_x[idx] - K_RED_RAMP_START).max(0.0);
            let pixel_y = xyb_y[idx];
            let pixel_b = (xyb_b[idx] - pixel_y - K_BLUE_RAMP_START).max(0.0);

            let blue_slope = pixel_b.min(K_BLUE_RAMP_LENGTH);
            let red_slope = pixel_x.min(K_RED_RAMP_LENGTH);
            red_coverage += red_slope;
            blue_coverage += blue_slope;
        }
    }

    const RATIO: f32 = 30.610_615_f32; // out of 64 pixels

    let overall_red = red_coverage.min(RATIO * K_RED_RAMP_LENGTH) * (red_strength / RATIO);
    let overall_blue = blue_coverage.min(RATIO * K_BLUE_RAMP_LENGTH) * (blue_strength / RATIO);

    result + overall_red + overall_blue
}

/// GammaModulation: adjust quantization based on gamma approximation.
///
/// The buffer must be padded to at least (y+8) rows and (x+8) columns with
/// edge-replicated values, so no bounds checking is needed.
fn gamma_modulation(
    x: usize,
    y: usize,
    xyb_x: &[f32],
    xyb_y: &[f32],
    stride: usize,
    out_val: f32,
) -> f32 {
    const K_BIAS: f32 = 0.16;
    let mut overall_ratio = 0.0_f32;

    for dy in 0..8 {
        let py = y + dy;
        for dx in 0..8 {
            let px = x + dx;
            let idx = py * stride + px;
            let iny = xyb_y[idx] + K_BIAS;
            let inx = xyb_x[idx];
            let r = iny - inx;
            let g = iny + inx;
            let ratio_r = ratio_of_derivatives(r, true);
            let ratio_g = ratio_of_derivatives(g, true);
            overall_ratio += 0.5 * (ratio_r + ratio_g);
        }
    }

    overall_ratio /= 64.0;

    // ln(2) constant folded in because we want std::log but have fast_log2f
    const K_GAM: f32 = -0.155_268_78_f32 * 0.693_147_2_f32;
    out_val + K_GAM * fast_log2f(overall_ratio)
}

/// Compute pre-erosion map from XYB planes.
///
/// For each pixel, computes local differences in Y (with kXMul×X contribution),
/// applies gamma ratio and masking sqrt, then downsamples 4× in each direction.
///
/// The tile may be padded by 4 pixels on each side for border handling.
/// Output dimensions: ceil(tile_pixel_w / 4) × ceil(tile_pixel_h / 4).
#[allow(clippy::too_many_arguments)]
fn compute_pre_erosion(
    xyb_x: &[f32],
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
) -> (Vec<f32>, usize, usize) {
    const MATCH_GAMMA_OFFSET: f32 = 0.019;
    const K_X_MUL: f32 = 23.426_803_f32;

    // Extend tile region by 4 pixels for border handling.
    // tile_x1/tile_y1 may exceed image dimensions (padded to block boundary);
    // pixel accesses below are clamped to simulate edge replication.
    let x0 = if tile_x0 > 0 { tile_x0 - 4 } else { 0 };
    let x1 = if tile_x1 < width {
        tile_x1 + 4
    } else {
        tile_x1
    };
    let y_start = if tile_y0 > 0 { tile_y0 - 4 } else { 0 };
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

    // max_x / max_y: clamp coordinates to actual image bounds (edge replication)
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

            // Y channel base (average of 4 neighbors)
            let base = 0.25
                * (xyb_y[y2 * width + xc]
                    + xyb_y[y1 * width + xc]
                    + xyb_y[yc * width + x1_local]
                    + xyb_y[yc * width + x2]);

            let gammac = ratio_of_derivatives(xyb_y[yc * width + xc] + MATCH_GAMMA_OFFSET, false);

            let mut diff = gammac * (xyb_y[yc * width + xc] - base);
            diff *= diff;

            // X channel base
            let base_x = 0.25
                * (xyb_x[y2 * width + xc]
                    + xyb_x[y1 * width + xc]
                    + xyb_x[yc * width + x1_local]
                    + xyb_x[yc * width + x2]);

            let mut diff_x = gammac * (xyb_x[yc * width + xc] - base_x);
            diff_x *= diff_x;
            diff += K_X_MUL * diff_x;
            diff = masking_sqrt(diff);

            let local_x = x - x0;
            if (y - y_start) % 4 != 0 {
                diff_buffer[local_x] += diff;
            } else {
                diff_buffer[local_x] = diff;
            }
        }

        // At every 4th row (y%4 == 3), downsample horizontally by 4
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

/// FuzzyErosion: 3×3 min-4 weighted sum, then 2x downsample.
///
/// For each pixel, finds the 4 smallest values in its 3×3 neighborhood,
/// then computes a weighted sum. Downsamples by 2 in both dimensions.
fn fuzzy_erosion(
    from: &[f32],
    from_w: usize,
    from_h: usize,
    from_x0: usize,
    from_y0: usize,
    region_w: usize,
    region_h: usize,
) -> (Vec<f32>, usize, usize) {
    let out_w = region_w / 2;
    let out_h = region_h / 2;
    let mut out = vec![0.0_f32; out_w * out_h];

    for fy in 0..region_h {
        let y = fy + from_y0;
        let ym1 = if y >= 1 { y - 1 } else { y };
        let yp1 = if y + 1 < from_h { y + 1 } else { y };

        for fx in 0..region_w {
            let x = fx + from_x0;
            let xm1 = if x >= 1 { x - 1 } else { x };
            let xp1 = if x + 1 < from_w { x + 1 } else { x };

            // Get all 9 neighbors
            let center = from[y * from_w + x];
            let left = from[y * from_w + xm1];
            let right = from[y * from_w + xp1];
            let top_left = from[ym1 * from_w + xm1];
            let top = from[ym1 * from_w + x];
            let top_right = from[ym1 * from_w + xp1];
            let bot_left = from[yp1 * from_w + xm1];
            let bot = from[yp1 * from_w + x];
            let bot_right = from[yp1 * from_w + xp1];

            // Find smallest 4 from 9 values
            // Start with first 4, sort them
            let mut min0 = center;
            let mut min1 = left;
            let mut min2 = right;
            let mut min3 = top_left;

            // Sort first 4
            if min0 > min1 {
                core::mem::swap(&mut min0, &mut min1);
            }
            if min0 > min2 {
                core::mem::swap(&mut min0, &mut min2);
            }
            if min0 > min3 {
                core::mem::swap(&mut min0, &mut min3);
            }
            if min1 > min2 {
                core::mem::swap(&mut min1, &mut min2);
            }
            if min1 > min3 {
                core::mem::swap(&mut min1, &mut min3);
            }
            if min2 > min3 {
                core::mem::swap(&mut min2, &mut min3);
            }

            // Insert remaining 5 values
            store_min4(top, &mut min0, &mut min1, &mut min2, &mut min3);
            store_min4(top_right, &mut min0, &mut min1, &mut min2, &mut min3);
            store_min4(bot_left, &mut min0, &mut min1, &mut min2, &mut min3);
            store_min4(bot, &mut min0, &mut min1, &mut min2, &mut min3);
            store_min4(bot_right, &mut min0, &mut min1, &mut min2, &mut min3);

            // Uniform weights (libjxl-tiny uses all 0.05)
            const K_MUL_C: f32 = 0.05;
            const K_MUL0: f32 = 0.05;
            const K_MUL1: f32 = 0.05;
            const K_MUL2: f32 = 0.05;
            const K_MUL3: f32 = 0.05;
            let v =
                K_MUL_C * center + K_MUL0 * min0 + K_MUL1 * min1 + K_MUL2 * min2 + K_MUL3 * min3;

            let ox = fx / 2;
            let oy = fy / 2;
            if fx % 2 == 0 && fy % 2 == 0 {
                out[oy * out_w + ox] = v;
            } else {
                out[oy * out_w + ox] += v;
            }
        }
    }

    (out, out_w, out_h)
}

/// ComputeMaskForAcStrategyUse: simple masking hack.
fn compute_mask_for_ac_strategy_use(out_val: f32) -> f32 {
    const K_MUL: f32 = 1.0;
    const K_OFFSET: f32 = 0.001;
    K_MUL / (out_val + K_OFFSET)
}

/// PerBlockModulations: apply all modulations and convert exponent to multiplier.
///
/// For each block, applies ComputeMask, HfModulation, ColorModulation,
/// GammaModulation, then converts from exponent space to multiplicative
/// quant field via exp2.
///
/// `stride` is the row stride (padded width) of the XYB buffers.
#[allow(clippy::too_many_arguments)]
fn per_block_modulations(
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
    aq_map_w: usize,
) {
    let base_level = 0.5 * scale;
    let k_dampen_ramp_start = 7.0_f32;
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

            let mut out_val = aq_map[iy * aq_map_w + ix];
            out_val = compute_mask(out_val);
            out_val = hf_modulation(px, py, xyb_y, stride, out_val);
            out_val = color_modulation(
                px,
                py,
                xyb_x,
                xyb_y,
                xyb_b,
                stride,
                butteraugli_target,
                out_val,
            );
            out_val = gamma_modulation(px, py, xyb_x, xyb_y, stride, out_val);

            // Convert from exponent to multiplicative field: exp2(out_val * log2(e)) * mul + add
            // C++ uses: FastPow2f(out_val * 1.442695041f) * mul + add
            aq_map[iy * aq_map_w + ix] = fast_pow2f(out_val * 1.442_695_f32) * mul + add;
        }
    }
}

/// Compute the adaptive quantization field for the entire image.
///
/// Compute the float quant field and masking without converting to u8.
///
/// Returns `(quant_field_float, masking)`:
/// - `quant_field_float`: Per-block float quant values for content-adaptive global_scale
/// - `masking`: Per-block masking values for AC strategy selection
///
/// Use `quantize_quant_field()` to convert float field to u8 raw_quant after
/// computing global_scale from the float field statistics.
///
/// # Arguments
/// * `xyb_x`, `xyb_y`, `xyb_b` - XYB color planes, flat row-major `[y * width + x]`
/// * `width`, `height` - image dimensions in pixels
/// * `xsize_blocks`, `ysize_blocks` - image dimensions in 8×8 blocks
/// * `distance` - butteraugli target distance
#[allow(clippy::too_many_arguments)]
pub fn compute_quant_field_float(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
) -> (Vec<f32>, Vec<f32>) {
    const K_AC_QUANT: f32 = 0.8294;
    let scale = K_AC_QUANT / distance;

    // Process the entire image as one tile.
    let tile_x0_pixels = 0;
    let tile_y0_pixels = 0;

    // Step 1: Compute pre-erosion (4x downsample of local differences)
    let (pre_erosion, pre_erosion_w, pre_erosion_h) = compute_pre_erosion(
        xyb_x,
        xyb_y,
        width,
        height,
        tile_x0_pixels,
        tile_y0_pixels,
        width,
        height,
    );

    // Step 2: Fuzzy erosion (3×3 min-4 weighted sum, 2x downsample)
    let from_x0 = if tile_x0_pixels > 0 { 1 } else { 0 };
    let from_y0 = if tile_y0_pixels > 0 { 1 } else { 0 };
    let erosion_region_w = (xsize_blocks * 2).min(pre_erosion_w.saturating_sub(from_x0));
    let erosion_region_h = (ysize_blocks * 2).min(pre_erosion_h.saturating_sub(from_y0));

    let (mut aq_map, aq_map_w, _aq_map_h) = fuzzy_erosion(
        &pre_erosion,
        pre_erosion_w,
        pre_erosion_h,
        from_x0,
        from_y0,
        erosion_region_w,
        erosion_region_h,
    );

    // Step 2.5: Compute masking field for AC strategy use (snapshot before modulations)
    let mut masking = vec![0.0f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            masking[by * xsize_blocks + bx] =
                compute_mask_for_ac_strategy_use(aq_map[by * aq_map_w + bx]);
        }
    }

    // Step 3: Per-block modulations
    per_block_modulations(
        xyb_x,
        xyb_y,
        xyb_b,
        width,
        distance,
        scale,
        0,
        0,
        xsize_blocks,
        ysize_blocks,
        &mut aq_map,
        aq_map_w,
    );

    // Step 4: Extract compact float quant field
    let mut quant_field_float = vec![0.0f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            quant_field_float[by * xsize_blocks + bx] = aq_map[by * aq_map_w + bx];
        }
    }

    (quant_field_float, masking)
}

/// Convert float quant field to u8 raw_quant values.
///
/// raw_quant = clamp(round(quant_field * inv_scale + 0.5), 1, 255)
pub fn quantize_quant_field(quant_field_float: &[f32], inv_scale: f32) -> Vec<u8> {
    quant_field_float
        .iter()
        .map(|&qf| {
            let val = (qf * inv_scale + 0.5).round() as i32;
            clamp(val, 1, 255) as u8
        })
        .collect()
}

/// Returns a flat buffer of `u8` values, indexed as `[by * xsize_blocks + bx]`.
/// Each value is the per-block raw_quant in range [1, 255].
///
/// This is a convenience wrapper that calls `compute_quant_field_float()` then
/// `quantize_quant_field()`. For content-adaptive global_scale, use those two
/// functions separately.
///
/// # Arguments
/// * `xyb_x`, `xyb_y`, `xyb_b` - XYB color planes, flat row-major `[y * width + x]`
/// * `width`, `height` - image dimensions in pixels
/// * `xsize_blocks`, `ysize_blocks` - image dimensions in 8×8 blocks
/// * `distance` - butteraugli target distance
/// * `inv_scale` - 1.0 / (global_scale / 65536)
#[allow(clippy::too_many_arguments)]
pub fn compute_adaptive_quant_field(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    inv_scale: f32,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (quant_field_float, masking) = compute_quant_field_float(
        xyb_x,
        xyb_y,
        xyb_b,
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        distance,
    );
    let raw_quant_field = quantize_quant_field(&quant_field_float, inv_scale);
    (raw_quant_field, masking, quant_field_float)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_log2f() {
        // log2(1.0) = 0.0
        let v = fast_log2f(1.0);
        assert!((v - 0.0).abs() < 0.001, "fast_log2f(1.0) = {}", v);

        // log2(2.0) = 1.0
        let v = fast_log2f(2.0);
        assert!((v - 1.0).abs() < 0.001, "fast_log2f(2.0) = {}", v);

        // log2(4.0) = 2.0
        let v = fast_log2f(4.0);
        assert!((v - 2.0).abs() < 0.001, "fast_log2f(4.0) = {}", v);

        // log2(0.5) = -1.0
        let v = fast_log2f(0.5);
        assert!((v - (-1.0)).abs() < 0.001, "fast_log2f(0.5) = {}", v);
    }

    #[test]
    fn test_fast_pow2f() {
        // 2^0 = 1.0
        let v = fast_pow2f(0.0);
        assert!((v - 1.0).abs() < 0.001, "fast_pow2f(0.0) = {}", v);

        // 2^1 = 2.0
        let v = fast_pow2f(1.0);
        assert!((v - 2.0).abs() < 0.01, "fast_pow2f(1.0) = {}", v);

        // 2^(-1) = 0.5
        let v = fast_pow2f(-1.0);
        assert!((v - 0.5).abs() < 0.001, "fast_pow2f(-1.0) = {}", v);

        // 2^10 = 1024
        let v = fast_pow2f(10.0);
        assert!((v - 1024.0).abs() < 1.0, "fast_pow2f(10.0) = {}", v);
    }

    #[test]
    fn test_masking_sqrt() {
        // Should produce reasonable positive values
        let v = masking_sqrt(0.0);
        assert!(v > 0.0, "masking_sqrt(0.0) = {}", v);

        // Monotonically increasing
        let v1 = masking_sqrt(1.0);
        let v2 = masking_sqrt(10.0);
        assert!(v2 > v1, "masking_sqrt should be monotonically increasing");
    }

    #[test]
    fn test_store_min4() {
        let mut min0 = 5.0_f32;
        let mut min1 = 6.0;
        let mut min2 = 7.0;
        let mut min3 = 8.0;

        store_min4(3.0, &mut min0, &mut min1, &mut min2, &mut min3);
        assert_eq!(min0, 3.0);
        assert_eq!(min1, 5.0);
        assert_eq!(min2, 6.0);
        assert_eq!(min3, 7.0);

        store_min4(100.0, &mut min0, &mut min1, &mut min2, &mut min3);
        // 100 > min3, so nothing changes
        assert_eq!(min3, 7.0);
    }

    #[test]
    fn test_compute_mask() {
        // Should produce finite values for reasonable inputs
        let v = compute_mask(1.0);
        assert!(v.is_finite(), "compute_mask(1.0) = {}", v);

        let v = compute_mask(0.0);
        assert!(v.is_finite(), "compute_mask(0.0) = {}", v);
    }

    #[test]
    fn test_ratio_of_derivatives() {
        // Should produce finite positive values for positive inputs
        let v = ratio_of_derivatives(1.0, false);
        assert!(v > 0.0 && v.is_finite(), "ratio(1.0, false) = {}", v);

        let v = ratio_of_derivatives(1.0, true);
        assert!(v > 0.0 && v.is_finite(), "ratio(1.0, true) = {}", v);

        // Zero input should not crash (clamped internally)
        let v = ratio_of_derivatives(0.0, false);
        assert!(v.is_finite(), "ratio(0.0, false) = {}", v);
    }

    #[test]
    fn test_adaptive_quant_field_uniform() {
        // A uniform gray image should produce roughly uniform quant field
        let w = 16;
        let h = 16;
        let n = w * h;
        let xyb_x = vec![0.0_f32; n];
        let xyb_y = vec![0.5_f32; n];
        let xyb_b = vec![0.5_f32; n];

        let xb = w / 8;
        let yb = h / 8;

        let (result, masking, _quant_float) =
            compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, w, h, xb, yb, 1.0, 8.93);

        assert_eq!(result.len(), xb * yb);
        assert_eq!(masking.len(), xb * yb);
        // All values should be in valid range
        for &v in &result {
            assert!(v >= 1, "quant value {} out of range", v);
        }
        // For uniform image, all blocks should have the same value
        let first = result[0];
        for &v in &result {
            assert_eq!(v, first, "uniform image should produce uniform quant field");
        }
    }

    #[test]
    fn test_adaptive_quant_field_varying() {
        // An image with varying content should produce varying quant values
        let w = 32;
        let h = 32;
        let n = w * h;
        let mut xyb_x = vec![0.0_f32; n];
        let mut xyb_y = vec![0.0_f32; n];
        let mut xyb_b = vec![0.0_f32; n];

        // Left half: smooth (low values)
        // Right half: high-frequency pattern
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if x < w / 2 {
                    xyb_y[idx] = 0.5;
                    xyb_b[idx] = 0.5;
                } else {
                    // Checkerboard pattern
                    xyb_y[idx] = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
                    xyb_b[idx] = xyb_y[idx];
                    xyb_x[idx] = if x % 2 == 0 { 0.1 } else { -0.1 };
                }
            }
        }

        let xb = w / 8;
        let yb = h / 8;

        let (result, _masking, _quant_float) =
            compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, w, h, xb, yb, 1.0, 8.93);

        assert_eq!(result.len(), xb * yb);
        // All values should be in valid range
        for &v in &result {
            assert!(v >= 1, "quant value {} out of range", v);
        }
        // Smooth and textured regions should differ
        // (left column blocks vs right column blocks)
        let left_avg: f32 = (0..yb).map(|by| result[by * xb] as f32).sum::<f32>() / yb as f32;
        let right_avg: f32 = (0..yb)
            .map(|by| result[by * xb + xb - 1] as f32)
            .sum::<f32>()
            / yb as f32;
        // They should be different (adaptive quant is doing something)
        assert!(
            (left_avg - right_avg).abs() > 0.01,
            "smooth vs textured should differ: left={}, right={}",
            left_avg,
            right_avg
        );
    }

    #[test]
    fn test_adaptive_quant_field_non_multiple_of_8() {
        // Regression test: dimensions not multiples of 8 caused OOB panic
        // because pre-erosion dimensions were too small for the block count.
        //
        // The caller (encoder.rs) pads XYB buffers to block boundaries with
        // edge replication. This test simulates that by allocating padded buffers.
        for &(w, h) in &[
            (300usize, 300usize),
            (301, 301),
            (100, 100),
            (17, 17),
            (9, 9),
            (15, 33),
            (257, 129),
        ] {
            let xb = w.div_ceil(8);
            let yb = h.div_ceil(8);
            let pw = xb * 8; // padded width
            let ph = yb * 8; // padded height
            let n = pw * ph;
            let xyb_x = vec![0.0_f32; n];
            let xyb_y = vec![0.5_f32; n];
            let xyb_b = vec![0.5_f32; n];

            let (result, _masking, _quant_float) =
                compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, pw, ph, xb, yb, 1.0, 8.93);

            assert_eq!(
                result.len(),
                xb * yb,
                "wrong length for {}x{}: got {}, expected {}",
                w,
                h,
                result.len(),
                xb * yb
            );
            for &v in &result {
                assert!(v >= 1, "quant value {} out of range for {}x{}", v, w, h);
            }
        }
    }
}
