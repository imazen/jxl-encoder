// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! AC coefficient quantization: thresholding, bias correction, error diffusion.
//!
//! Contains the dead-zone quantization logic, AdjustQuantBlockAC heuristics,
//! and AdjustQuantBias dequantization correction. These are SIMD-friendly
//! per-coefficient operations.

use super::ac_strategy::{
    RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT8X4,
    RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT32X32,
    RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
};
use super::common::{BLOCK_DIM, DCT_BLOCK_SIZE};
use super::encoder::VarDctEncoder;

/// Apply AdjustQuantBias to a quantized value for dequantization.
///
/// Ported from libjxl-tiny's AdjustQuantBias. For +/-1 values, returns a
/// channel-specific biased value. For larger values, applies a small
/// reciprocal correction: `q - 0.145 / q`.
#[allow(clippy::excessive_precision)]
#[inline]
pub(super) fn adjust_quant_bias(quantized: i32, channel: usize) -> f32 {
    // kDefaultQuantBias from libjxl-tiny enc_group.cc
    // [0..2] = channel-specific bias for +/-1 values
    // [3] = reciprocal correction factor for |q| >= 2
    const BIAS: [f32; 4] = [
        1.0 - 0.05465007330715401,  // [0] X channel +/-1 -> 0.945349
        1.0 - 0.07005449891748593,  // [1] Y channel +/-1 -> 0.929946
        1.0 - 0.049935103337343655, // [2] B channel +/-1 -> 0.950065
        0.145,                      // [3] reciprocal correction
    ];

    if quantized == 0 {
        return 0.0;
    }

    let q = quantized as f32;

    // C++ uses abs(float) < 1.125 to detect +/-1 (since q is integer)
    if q.abs() < 1.125 {
        // +/-1: return +/-BIAS[channel]
        q.signum() * BIAS[channel]
    } else {
        // |q| >= 2: return q - BIAS[3] / q
        q - BIAS[3] / q
    }
}

impl VarDctEncoder {
    /// Compute default dead-zone thresholds for a given channel and coverage.
    ///
    /// Returns [f32; 4] thresholds for the 4 quadrants of a block.
    /// Matches full libjxl enc_group.cc:58-72 (> kHare speed tier).
    pub(crate) fn default_thresholds(c: usize, covered_x: usize, covered_y: usize) -> [f32; 4] {
        // Full libjxl values (enc_group.cc:58-65, > kHare speed):
        //   Y (c=1): {0.56, 0.62, 0.62, 0.62}
        //   X (c=0): {0.58, 0.62, 0.62, 0.62}
        //   B (c=2): {0.58, 0.62, 0.62, 0.62}
        let mut thres = if c == 1 {
            [0.56f32, 0.62, 0.62, 0.62]
        } else {
            [0.58f32, 0.62, 0.62, 0.62]
        };
        // X/B multi-block threshold reduction (enc_group.cc:66-72)
        // For c != 1 (X and B channels) with coverage >= 4 blocks
        if c != 1 && covered_x * covered_y >= 4 {
            let adj = 0.00744 * (covered_x * covered_y) as f32;
            for t in thres.iter_mut() {
                *t -= adj;
                if *t < 0.5 {
                    *t = 0.5;
                }
            }
        }
        thres
    }

    /// Quantize a single AC coefficient with thresholding.
    ///
    /// Ported from libjxl-tiny QuantizeBlockAC. Small coefficients below a
    /// threshold are zeroed out. The threshold depends on:
    /// - Quadrant position within the block (4 quadrants)
    ///
    /// `thresholds` are the pre-computed dead-zone thresholds for the 4 quadrants.
    /// `qm_multiplier` is typically 1.0, but for X channel it's `x_qm_mul`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quantize_coeff_ac(
        coef: f32,
        inv_weight: f32, // 1/weight (InvMatrix in C++)
        qac: f32,        // scale * quant_ac
        qm_multiplier: f32,
        thresholds: &[f32; 4],
        y_in_block: usize,
        x_in_block: usize,
        block_height: usize,
        block_width: usize,
    ) -> i32 {
        // Quadrant selection: which of the 4 quadrants does this coeff fall in
        let y_half = if y_in_block >= block_height / 2 { 2 } else { 0 };
        let x_half = if x_in_block >= block_width / 2 { 1 } else { 0 };
        let thr = thresholds[y_half + x_half];

        let val = inv_weight * qac * qm_multiplier * coef;
        if val.abs() < thr {
            0
        } else {
            val.round() as i32
        }
    }

    /// Adjust per-block quantization and thresholds based on coefficient analysis.
    ///
    /// Ported from libjxl enc_group.cc:104-328. Only applies to DCT8+ strategies
    /// (skips IDENTITY, DCT2X2, DCT4X4, DCT4X8, DCT8X4). Implements 6 heuristics:
    ///
    /// 1. Threshold reduction for multi-block transforms
    /// 2. Sparse block Y-channel quant boost + threshold adjustment (B)
    /// 3. High-frequency corner quant increase (C)
    /// 4. DCT8 flatness detection quant boost (D)
    /// 5. Large transform error correction (E)
    /// 6. Activity-based quant reduction + threshold adjustment (F)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adjust_quant_block_ac(
        block_coeffs: &[f32],
        weights: &[f32],
        qac: f32,
        qm_multiplier: f32,
        c: usize,
        raw_strategy: u8,
        block_width: usize,
        block_height: usize,
        xsize: usize, // cx (8x8 blocks in x)
        ysize: usize, // cy (8x8 blocks in y)
        thresholds: &mut [f32; 4],
        quant: &mut i32,
    ) {
        const QUANT_MAX: i32 = 256;

        // Skip partial block kinds (small transforms)
        match raw_strategy {
            RAW_STRATEGY_IDENTITY
            | RAW_STRATEGY_DCT2X2
            | RAW_STRATEGY_DCT4X4
            | RAW_STRATEGY_DCT4X8
            | RAW_STRATEGY_DCT8X4 => return,
            _ => {}
        }

        // (1) Threshold reduction for large transforms
        if xsize > 1 || ysize > 1 {
            let adj = (0.003 * (xsize * ysize) as f32).clamp(0.0, 0.08);
            for t in thresholds.iter_mut() {
                *t -= adj;
                if *t < 0.54 {
                    *t = 0.54;
                }
            }
        }

        // Pre-scan: compute statistics over non-LLF coefficients
        let mut sum_of_highest_freq: f32 = 0.0;
        let mut sum_of_error: f32 = 0.0;
        let mut sum_of_vals: f32 = 0.0;
        let mut hf_nonzeros = [0.0f32; 4];
        let mut hf_max_error = [0.0f32; 4];

        for y in 0..block_height {
            for x in 0..block_width {
                let pos = y * block_width + x;
                // Skip LLF positions
                if x < xsize && y < ysize {
                    continue;
                }
                let hfix = (if y >= block_height / 2 { 2 } else { 0 })
                    + (if x >= block_width / 2 { 1 } else { 0 });

                // Match our quantize_coeff_ac formula: val = (1/weight) * qac * qm_mul * coef
                let inv_w = 1.0 / weights[pos];
                let val = block_coeffs[pos] * inv_w * qac * qm_multiplier;
                let v = if val.abs() < thresholds[hfix] {
                    0.0
                } else {
                    val.round()
                };
                let error = (val - v).abs();
                sum_of_error += error;
                sum_of_vals += v.abs();

                if c == 1 && v == 0.0 && hf_max_error[hfix] < error {
                    hf_max_error[hfix] = error;
                }
                if v != 0.0 {
                    hf_nonzeros[hfix] += v.abs();
                    let in_corner = y >= 7 * ysize && x >= 7 * xsize;
                    let on_border = y == block_height - 1 || x == block_width - 1;
                    let in_larger_corner = x >= 4 * xsize && y >= 4 * ysize;
                    if in_corner || (on_border && in_larger_corner) {
                        sum_of_highest_freq += val.abs();
                    }
                }
            }
        }

        // (2) Sparse block Y-channel handling (B heuristic)
        if c == 1 && (sum_of_vals * 8.0) < (xsize * ysize) as f32 {
            const K_LIMIT: [f64; 4] = [0.46, 0.46, 0.46, 0.46];
            const K_MUL: [f64; 4] = [0.9999, 0.9999, 0.9999, 0.9999];

            let orig_quant = *quant;
            let mut new_quant = *quant;
            for i in 1..4 {
                if hf_nonzeros[i] == 0.0 && (hf_max_error[i] as f64) > K_LIMIT[i] {
                    new_quant = orig_quant + 1;
                    break;
                }
            }
            *quant = new_quant;

            if hf_nonzeros[3] == 0.0 && (hf_max_error[3] as f64) > K_LIMIT[3] {
                thresholds[3] = (K_MUL[3] * hf_max_error[3] as f64 * new_quant as f64
                    / orig_quant as f64) as f32;
            } else if (hf_nonzeros[1] == 0.0 && (hf_max_error[1] as f64) > K_LIMIT[1])
                || (hf_nonzeros[2] == 0.0 && (hf_max_error[2] as f64) > K_LIMIT[2])
            {
                let max_err = hf_max_error[1].max(hf_max_error[2]);
                thresholds[1] =
                    (K_MUL[1] * max_err as f64 * new_quant as f64 / orig_quant as f64) as f32;
                thresholds[2] = thresholds[1];
            } else if hf_nonzeros[0] == 0.0 && (hf_max_error[0] as f64) > K_LIMIT[0] {
                thresholds[0] = (K_MUL[0] * hf_max_error[0] as f64 * new_quant as f64
                    / orig_quant as f64) as f32;
            }
        }

        // (3) High-frequency corner penalty (C heuristic)
        {
            let all = hf_nonzeros[0] + hf_nonzeros[1] + hf_nonzeros[2] + hf_nonzeros[3] + 1.0;
            let mul = [70.0f32, 30.0, 60.0];
            if mul[c] * sum_of_highest_freq >= all {
                *quant += (mul[c] * sum_of_highest_freq / all) as i32;
                if *quant >= QUANT_MAX {
                    *quant = QUANT_MAX - 1;
                }
            }
        }

        // (4) DCT8 flatness detection (D heuristic)
        if raw_strategy == 0 {
            // DCT8: if block is very flat (few nonzeros), increase quant to reduce blocking
            if hf_nonzeros[0] + hf_nonzeros[1] + hf_nonzeros[2] + hf_nonzeros[3] < 11.0 {
                *quant += 1;
                if *quant >= QUANT_MAX {
                    *quant = QUANT_MAX - 1;
                }
            }
        }

        // (5) Large transform error correction (E heuristic)
        {
            #[allow(clippy::excessive_precision)]
            const K_MUL1: [[f64; 3]; 4] = [
                [
                    0.22080615753848404,
                    0.45797479824262011,
                    0.29859235095977965,
                ],
                [
                    0.70109486510286834,
                    0.16185281305512639,
                    0.14387691730035473,
                ],
                [
                    0.114985964456218638,
                    0.44656840441027695,
                    0.10587658215149048,
                ],
                [
                    0.46849665264409396,
                    0.41239077937781954,
                    0.088667407767185444,
                ],
            ];
            #[allow(clippy::excessive_precision)]
            const K_MUL2: [[f64; 3]; 4] = [
                [0.27450281941822197, 1.1255766549984996, 0.98950459134128388],
                [0.4652168675598285, 0.40945807983455818, 0.36581899811751367],
                [0.28034972424715715, 0.9182653201929738, 1.5581531543057416],
                [0.26873118114033728, 0.68863712390392484, 1.2082185408666786],
            ];
            const K_QUANT_NORMALIZER: f64 = 2.294_270_834_328_472;

            // Only applies to DCT16X16 and larger
            let is_large = matches!(
                raw_strategy,
                RAW_STRATEGY_DCT16X16
                    | RAW_STRATEGY_DCT32X32
                    | RAW_STRATEGY_DCT16X8
                    | RAW_STRATEGY_DCT8X16
                    | RAW_STRATEGY_DCT64X64
                    | RAW_STRATEGY_DCT64X32
                    | RAW_STRATEGY_DCT32X64
            );
            if is_large {
                // Map strategy to table index
                let ix = match raw_strategy {
                    RAW_STRATEGY_DCT16X16 => 0,
                    RAW_STRATEGY_DCT32X32 => 2,
                    // DCT16X8 and DCT8X16 use default index 3
                    _ => 3,
                };

                let norm_error = sum_of_error as f64 * K_QUANT_NORMALIZER;
                let norm_vals = sum_of_vals as f64 * K_QUANT_NORMALIZER;
                let area = (xsize * ysize * BLOCK_DIM * BLOCK_DIM) as f64;
                let threshold = K_MUL1[ix][c] * area + K_MUL2[ix][c] * norm_vals;

                if norm_error > threshold {
                    let step = (norm_error / threshold) as i32;
                    let step = step.clamp(0, 2);
                    *quant += step;
                    if *quant >= QUANT_MAX {
                        *quant = QUANT_MAX - 1;
                    }
                }
            }
        }

        // (6) Activity-based quant reduction (F heuristic)
        {
            let div = (xsize * ysize) as i32;
            let mut activity = (hf_nonzeros[0] as i32 + div / 2) / div;
            let orig_qp_limit = (*quant / 2).max(4);
            for hf_nz in &hf_nonzeros[1..4] {
                activity = activity.min((*hf_nz as i32 + div / 2) / div);
            }
            if activity >= 15 {
                activity = 15;
            }
            let mut qp = *quant - activity;
            if c == 1 {
                for t in thresholds[1..4].iter_mut() {
                    *t += 0.01 * activity as f32;
                }
            }
            if qp < orig_qp_limit {
                qp = orig_qp_limit;
            }
            *quant = qp;
        }
    }

    /// Quantize AC coefficients with thresholding and store in quant_ac slots.
    /// When error_diffusion is true, processes coefficients in zigzag order
    /// and propagates quantization error to subsequent coefficients.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quantize_ac_block(
        dct_coeffs: &[f32],
        weights: &[f32],
        qac: f32,
        qm_multiplier: f32,
        thresholds: &[f32; 4],
        _block_width: usize,
        _block_height: usize,
        covered_x: usize,
        covered_y: usize,
        _covered_blocks: usize,
        size: usize,
        _raw_strategy: u8,
        bx: usize,
        by: usize,
        quant_ac: &mut [Vec<[i32; DCT_BLOCK_SIZE]>],
        error_diffusion: bool,
        zigzag_order: Option<&[u32]>,
        error_scratch: Option<&mut Vec<f32>>,
    ) {
        // C++ QuantizeBlockAC uses post-swap (cx, cy) for the coefficient grid:
        // stride = cx * 8 (block_width), height = cy * 8 (block_height).
        // After swap, cx >= cy. Both DCT16x8 and DCT8x16 have grid_width=16.
        let grid_width = _block_width;
        let grid_height = _block_height;
        let cx = _block_width / BLOCK_DIM;
        let cy = _block_height / BLOCK_DIM;

        // For rectangular transforms like DCT16x8, the coefficient layout (16x8) differs
        // from physical block coverage (1x2). We need to transpose the slot mapping when
        // the physical coverage is "tall" (covered_y > covered_x) but coefficient layout
        // is "wide" (cx > cy).
        let transpose_slots = covered_y > covered_x;

        if !error_diffusion {
            // Standard quantization without error diffusion
            #[cfg(feature = "debug-tokens")]
            let mut debug_nonzero_count = 0usize;
            for idx in 0..size {
                // LLF positions are at (y, x) where y < cy and x < cx in the grid.
                // For DCT8 this is just index 0.
                // For DCT16x16 (cx=cy=2, stride=16) this is {0, 1, 16, 17}.
                // For DCT16x8 (cx=2, cy=1, stride=16) this is {0, 1}.
                let is_llf = (idx / grid_width) < cy && (idx % grid_width) < cx;
                let qval = if is_llf {
                    0 // LLF handled separately
                } else {
                    let y = idx / grid_width;
                    let x = idx % grid_width;
                    Self::quantize_coeff_ac(
                        dct_coeffs[idx],
                        1.0 / weights[idx],
                        qac,
                        qm_multiplier,
                        thresholds,
                        y,
                        x,
                        grid_height,
                        grid_width,
                    )
                };

                #[cfg(feature = "debug-tokens")]
                if qval != 0 {
                    debug_nonzero_count += 1;
                }

                // Store in flat layout: idx = y * grid_width + x in the transform grid.
                // Map to 8x8 block slots for storage.
                let y = idx / grid_width;
                let x = idx % grid_width;
                let coef_slot_y = y / BLOCK_DIM;
                let coef_slot_x = x / BLOCK_DIM;
                let pos_y = y % BLOCK_DIM;
                let pos_x = x % BLOCK_DIM;
                let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;

                // Map coefficient slot to physical block offset.
                // For DCT16x8: coefficient layout is 16x8 (2 cols x 1 row of slots)
                //              physical coverage is 1x2 (1 col x 2 rows of blocks)
                // So coef_slot_x maps to physical row offset, coef_slot_y to col offset.
                let (phys_row_off, phys_col_off) = if transpose_slots {
                    (coef_slot_x, coef_slot_y)
                } else {
                    (coef_slot_y, coef_slot_x)
                };
                quant_ac[by + phys_row_off][bx + phys_col_off][pos_in_8x8] = qval;
            }
            #[cfg(feature = "debug-tokens")]
            if _raw_strategy == 4 && bx == 0 && by == 0 {
                eprintln!(
                    "[DCT32x32 quantize debug] Y at (0,0): {} nonzero AC coeffs stored (qac={:.4})",
                    debug_nonzero_count, qac
                );
                // Show first few AC coefficients and their quantized values
                let mut shown = 0;
                for idx in 16..size {
                    if shown >= 5 {
                        break;
                    }
                    let is_llf = (idx / grid_width) < cy && (idx % grid_width) < cx;
                    if !is_llf {
                        let coef = dct_coeffs[idx];
                        let w = weights[idx];
                        let inv_w = 1.0 / w;
                        let val = inv_w * qac * qm_multiplier * coef;
                        eprintln!(
                            "  [{}] coef={:.6}, weight={:.6}, inv_w={:.4}, val={:.4}",
                            idx, coef, w, inv_w, val
                        );
                        shown += 1;
                    }
                }
            }
        } else {
            // Error diffusion: process in zigzag order, propagate error to next coefficient
            let zigzag = zigzag_order.unwrap_or_else(|| {
                panic!("zigzag_order must be provided when error_diffusion is true")
            });

            // Accumulated error to add to next coefficient (in zigzag order)
            // Using separate accumulators for different frequency bands
            let mut accumulated_error: f32 = 0.0;
            const ERROR_DIFFUSION_FACTOR: f32 = 0.25; // Propagate 1/4 of error

            // Use pre-allocated scratch buffer or allocate if not provided
            let scratch =
                error_scratch.expect("error_scratch must be provided when error_diffusion is true");
            if scratch.len() < size {
                scratch.resize(size, 0.0);
            }
            scratch[..size].copy_from_slice(&dct_coeffs[..size]);
            let corrected_coeffs = &mut scratch[..size];

            for (zigzag_pos, &flat_idx) in zigzag.iter().enumerate() {
                let idx = flat_idx as usize;
                if idx >= size {
                    continue;
                }

                let is_llf = (idx / grid_width) < cy && (idx % grid_width) < cx;

                if is_llf {
                    // LLF handled separately, no error diffusion
                    // Use flat layout mapping
                    let y = idx / grid_width;
                    let x = idx % grid_width;
                    let coef_slot_y = y / BLOCK_DIM;
                    let coef_slot_x = x / BLOCK_DIM;
                    let pos_y = y % BLOCK_DIM;
                    let pos_x = x % BLOCK_DIM;
                    let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                    let (phys_row_off, phys_col_off) = if transpose_slots {
                        (coef_slot_x, coef_slot_y)
                    } else {
                        (coef_slot_y, coef_slot_x)
                    };
                    quant_ac[by + phys_row_off][bx + phys_col_off][pos_in_8x8] = 0;
                    continue;
                }

                // Add accumulated error to this coefficient
                corrected_coeffs[idx] += accumulated_error * weights[idx];

                let y = idx / grid_width;
                let x = idx % grid_width;
                let inv_weight = 1.0 / weights[idx];
                let scaled_coeff = corrected_coeffs[idx] * inv_weight * qac * qm_multiplier;

                // Quantize
                let qval = Self::quantize_coeff_ac(
                    corrected_coeffs[idx],
                    inv_weight,
                    qac,
                    qm_multiplier,
                    thresholds,
                    y,
                    x,
                    grid_height,
                    grid_width,
                );

                // Compute quantization error
                // error = (original_scaled - quantized) / (qac * qm_multiplier)
                // This error is in the normalized coefficient domain
                let dequant_val = qval as f32;
                let error = (scaled_coeff - dequant_val) / (qac * qm_multiplier);

                // Accumulate error for next coefficient (only if not at the end)
                if zigzag_pos + 1 < zigzag.len() {
                    accumulated_error = error * ERROR_DIFFUSION_FACTOR;
                }

                // Store in flat layout: y, x already computed above
                let coef_slot_y = y / BLOCK_DIM;
                let coef_slot_x = x / BLOCK_DIM;
                let pos_y = y % BLOCK_DIM;
                let pos_x = x % BLOCK_DIM;
                let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                let (phys_row_off, phys_col_off) = if transpose_slots {
                    (coef_slot_x, coef_slot_y)
                } else {
                    (coef_slot_y, coef_slot_x)
                };
                quant_ac[by + phys_row_off][bx + phys_col_off][pos_in_8x8] = qval;
            }
        }
    }
}
