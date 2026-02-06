// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Transform and quantization pipeline: XYB conversion, DCT, quantization, CfL.

use super::ac_group::{num_nonzero_8x8_except_dc, num_nonzero_except_llf};
use super::ac_strategy::{
    AcStrategyMap, RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
    RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT8X4,
    RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT16X32,
    RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32,
    RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
};
use super::afv::{afv_transform_from_pixels, dc_from_afv};
use super::chroma_from_luma::{CflMap, ytob_ratio, ytox_ratio};
use super::coeff_order::natural_coeff_order;
use super::common::*;
use super::dct::{
    dc_from_dct_4x4_full, dc_from_dct_4x8_full, dc_from_dct_8x4_full, dc_from_dct_8x16,
    dc_from_dct_16x8, dc_from_dct_16x16, dc_from_dct_16x32, dc_from_dct_32x16, dc_from_dct_32x32,
    dc_from_dct_32x64, dc_from_dct_64x32, dc_from_dct_64x64, dct_4x4_full, dct_4x8_full,
    dct_8x4_full, dct_8x8, dct_8x16, dct_16x8, dct_16x16, dct_16x32, dct_32x16, dct_32x32,
    dct_32x64, dct_64x32, dct_64x64, dct2x2_transform, identity_transform,
};
use super::encoder::TinyEncoder;
use super::frame::DistanceParams;
use super::quant::INV_DC_QUANT;
use crate::color::xyb::linear_rgb_to_xyb;

impl TinyEncoder {
    /// Convert linear RGB to XYB color space with padding to block boundaries.
    ///
    /// Returns (xyb_x, xyb_y, xyb_b) arrays padded to `padded_width × padded_height`
    /// using edge replication (last pixel value extended to the boundary).
    /// This allows SIMD code to process full blocks without bounds checking.
    pub(crate) fn convert_to_xyb_padded(
        &self,
        width: usize,
        height: usize,
        padded_width: usize,
        padded_height: usize,
        linear_rgb: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let padded_n = padded_width * padded_height;
        let mut xyb_x = vec![0.0f32; padded_n];
        let mut xyb_y = vec![0.0f32; padded_n];
        let mut xyb_b = vec![0.0f32; padded_n];

        // Convert the actual image pixels
        for y in 0..height {
            for x in 0..width {
                let src_idx = y * width + x;
                let dst_idx = y * padded_width + x;
                let r = linear_rgb[src_idx * 3];
                let g = linear_rgb[src_idx * 3 + 1];
                let b = linear_rgb[src_idx * 3 + 2];
                let (xv, yv, bv) = linear_rgb_to_xyb(r, g, b);
                #[cfg(feature = "debug-dc")]
                if x == 0 && y == 0 {
                    eprintln!(
                        "XYB[0,0]: linear_rgb=({:.6},{:.6},{:.6}) -> XYB=({:.6},{:.6},{:.6})",
                        r, g, b, xv, yv, bv
                    );
                }
                xyb_x[dst_idx] = xv;
                xyb_y[dst_idx] = yv;
                xyb_b[dst_idx] = bv;
            }

            // Pad right edge with last pixel value (edge replication)
            if padded_width > width {
                let last_x_idx = y * padded_width + (width - 1);
                let last_x = xyb_x[last_x_idx];
                let last_y = xyb_y[last_x_idx];
                let last_b = xyb_b[last_x_idx];
                for x in width..padded_width {
                    let dst_idx = y * padded_width + x;
                    xyb_x[dst_idx] = last_x;
                    xyb_y[dst_idx] = last_y;
                    xyb_b[dst_idx] = last_b;
                }
            }
        }

        // Pad bottom rows by copying the last row
        if padded_height > height {
            let last_row_start = (height - 1) * padded_width;
            for y in height..padded_height {
                let dst_row_start = y * padded_width;
                for x in 0..padded_width {
                    xyb_x[dst_row_start + x] = xyb_x[last_row_start + x];
                    xyb_y[dst_row_start + x] = xyb_y[last_row_start + x];
                    xyb_b[dst_row_start + x] = xyb_b[last_row_start + x];
                }
            }
        }

        (xyb_x, xyb_y, xyb_b)
    }

    /// Compute default dead-zone thresholds for a given channel and coverage.
    ///
    /// Returns [f32; 4] thresholds for the 4 quadrants of a block.
    /// Matches full libjxl enc_group.cc:58-72 (> kHare speed tier).
    #[inline]
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

    /// Apply AdjustQuantBias to a quantized value for dequantization.
    ///
    /// Ported from libjxl-tiny's AdjustQuantBias. For ±1 values, returns a
    /// channel-specific biased value. For larger values, applies a small
    /// reciprocal correction: `q - 0.145 / q`.
    #[allow(clippy::excessive_precision)]
    #[inline]
    pub(crate) fn adjust_quant_bias(quantized: i32, channel: usize) -> f32 {
        // kDefaultQuantBias from libjxl-tiny enc_group.cc
        // [0..2] = channel-specific bias for ±1 values
        // [3] = reciprocal correction factor for |q| >= 2
        const BIAS: [f32; 4] = [
            1.0 - 0.05465007330715401,  // [0] X channel ±1 → 0.945349
            1.0 - 0.07005449891748593,  // [1] Y channel ±1 → 0.929946
            1.0 - 0.049935103337343655, // [2] B channel ±1 → 0.950065
            0.145,                      // [3] reciprocal correction
        ];

        if quantized == 0 {
            return 0.0;
        }

        let q = quantized as f32;

        // C++ uses abs(float) < 1.125 to detect ±1 (since q is integer)
        if q.abs() < 1.125 {
            // ±1: return ±BIAS[channel]
            q.signum() * BIAS[channel]
        } else {
            // |q| >= 2: return q - BIAS[3] / q
            q - BIAS[3] / q
        }
    }

    /// Apply DCT to a single channel at block position (bx, by).
    ///
    /// The `channel_data` must be padded to block boundaries (stride = padded_width).
    /// No bounds checking is performed - caller must ensure data is properly padded.
    pub(crate) fn apply_dct(
        channel_data: &[f32],
        stride: usize, // padded_width (row stride)
        bx: usize,
        by: usize,
        raw_strategy: u8,
        output: &mut [f32],
    ) {
        match raw_strategy {
            0 => {
                let mut block = [0.0f32; 64];
                for dy in 0..8 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..8 {
                        block[dy * 8 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 64];
                dct_8x8(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT16X8 => {
                let mut block = [0.0f32; 128];
                for dy in 0..16 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..8 {
                        block[dy * 8 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 128];
                dct_16x8(&block, &mut dct_out);
                output[..128].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT8X16 => {
                let mut block = [0.0f32; 128];
                for dy in 0..8 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..16 {
                        block[dy * 16 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 128];
                dct_8x16(&block, &mut dct_out);
                output[..128].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT16X16 => {
                let mut block = [0.0f32; 256];
                for dy in 0..16 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..16 {
                        block[dy * 16 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 256];
                dct_16x16(&block, &mut dct_out);
                output[..256].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT32X32 => {
                let mut block = [0.0f32; 1024];
                for dy in 0..32 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..32 {
                        block[dy * 32 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 1024];
                dct_32x32(&block, &mut dct_out);
                output[..1024].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT4X8 => {
                // DCT4X8 full: two 4x8 transforms covering 8x8 pixels
                let mut block = [0.0f32; 64];
                for dy in 0..8 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..8 {
                        block[dy * 8 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 64];
                dct_4x8_full(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT8X4 => {
                // DCT8X4 full: two 8x4 transforms covering 8x8 pixels
                let mut block = [0.0f32; 64];
                for dy in 0..8 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..8 {
                        block[dy * 8 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 64];
                dct_8x4_full(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT4X4 => {
                // DCT4X4 full: four 4x4 transforms covering 8x8 pixels
                let mut block = [0.0f32; 64];
                for dy in 0..8 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..8 {
                        block[dy * 8 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 64];
                dct_4x4_full(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_IDENTITY => {
                // IDENTITY: pixel differences from reference pixel per 4x4 sub-block
                let pixel_offset = by * BLOCK_DIM * stride + bx * BLOCK_DIM;
                identity_transform(&channel_data[pixel_offset..], stride, &mut output[..64]);
            }
            RAW_STRATEGY_DCT2X2 => {
                // DCT2X2: hierarchical 2x2 DCT
                let pixel_offset = by * BLOCK_DIM * stride + bx * BLOCK_DIM;
                dct2x2_transform(&channel_data[pixel_offset..], stride, &mut output[..64]);
            }
            RAW_STRATEGY_DCT32X16 => {
                // DCT32X16: 32x16 transform (4 rows × 2 cols of 8x8 blocks = 32 rows × 16 cols)
                let mut block = [0.0f32; 512];
                for dy in 0..32 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..16 {
                        block[dy * 16 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 512];
                dct_32x16(&block, &mut dct_out);
                output[..512].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT16X32 => {
                // DCT16X32: 16x32 transform (2 rows × 4 cols of 8x8 blocks = 16 rows × 32 cols)
                let mut block = [0.0f32; 512];
                for dy in 0..16 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..32 {
                        block[dy * 32 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 512];
                dct_16x32(&block, &mut dct_out);
                output[..512].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT64X64 => {
                // DCT64X64: 64x64 transform (8 rows × 8 cols of 8x8 blocks)
                let mut block = [0.0f32; 4096];
                for dy in 0..64 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..64 {
                        block[dy * 64 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 4096];
                dct_64x64(&block, &mut dct_out);
                output[..4096].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT64X32 => {
                // DCT64X32: 64x32 transform (8 rows × 4 cols of 8x8 blocks = 64 rows × 32 cols)
                let mut block = [0.0f32; 2048];
                for dy in 0..64 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..32 {
                        block[dy * 32 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 2048];
                dct_64x32(&block, &mut dct_out);
                output[..2048].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT32X64 => {
                // DCT32X64: 32x64 transform (4 rows × 8 cols of 8x8 blocks = 32 rows × 64 cols)
                let mut block = [0.0f32; 2048];
                for dy in 0..32 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..64 {
                        block[dy * 64 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 2048];
                dct_32x64(&block, &mut dct_out);
                output[..2048].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
                // AFV: Adaptive Frequency Variable (hybrid transform for corners)
                // Extract 8x8 pixels and compute AFV transform
                let mut pixels = [0.0f32; 64];
                for dy in 0..8 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..8 {
                        pixels[dy * 8 + dx] = channel_data[row_offset + dx];
                    }
                }
                let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
                let mut dct_out = [0.0f32; 64];
                afv_transform_from_pixels(&pixels, afv_kind, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            _ => unreachable!(),
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
    ) {
        // C++ QuantizeBlockAC uses post-swap (cx, cy) for the coefficient grid:
        // stride = cx * 8 (block_width), height = cy * 8 (block_height).
        // After swap, cx >= cy. Both DCT16x8 and DCT8x16 have grid_width=16.
        let grid_width = _block_width;
        let grid_height = _block_height;
        let cx = _block_width / BLOCK_DIM;
        let cy = _block_height / BLOCK_DIM;

        // For rectangular transforms like DCT16x8, the coefficient layout (16×8) differs
        // from physical block coverage (1×2). We need to transpose the slot mapping when
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
                // For DCT16x8: coefficient layout is 16×8 (2 cols × 1 row of slots)
                //              physical coverage is 1×2 (1 col × 2 rows of blocks)
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
            let zigzag = natural_coeff_order(cx, cy);

            // Accumulated error to add to next coefficient (in zigzag order)
            // Using separate accumulators for different frequency bands
            let mut accumulated_error: f32 = 0.0;
            const ERROR_DIFFUSION_FACTOR: f32 = 0.25; // Propagate 1/4 of error

            // Create a mutable copy of coefficients to apply error correction
            let mut corrected_coeffs = dct_coeffs.to_vec();

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

    /// Perform DCT and quantization on all blocks.
    ///
    /// Supports DCT8, DCT16X8, and DCT8X16 transforms based on ac_strategy.
    /// For multi-block transforms, only first blocks are processed; the second
    /// block's quant_ac slot stores the second half of the 128 coefficients.
    ///
    /// Processing order matches C++ WriteACGroup:
    /// 1. DCT Y → extract Y DC → quantize Y AC (with thresholding)
    /// 2. Dequantize Y AC back (AdjustQuantBias) → roundtripped Y
    /// 3. DCT X, B → apply CfL using roundtripped Y → extract X/B DC
    /// 4. Quantize X/B AC (with thresholding + x_qm_mul for X)
    ///
    /// Returns (quantized_dc, quantized_ac, nzeros)
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn transform_and_quantize(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize, // stride for padded XYB data
        xsize_blocks: usize,
        ysize_blocks: usize,
        params: &DistanceParams,
        quant_field: &mut [u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
    ) -> (
        [Vec<Vec<i16>>; 3],                   // quant_dc
        [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3], // quant_ac
        [Vec<Vec<u8>>; 3],                    // nzeros (shifted, for prediction)
        [Vec<Vec<u16>>; 3],                   // raw_nzeros (unshifted, for bitstream)
    ) {
        // Initialize output arrays
        let mut quant_dc: [Vec<Vec<i16>>; 3] = [
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
        ];

        let mut quant_ac: [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3] = [
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
        ];

        // Shifted nzeros for neighbor prediction (nzeros / covered_blocks)
        let mut nzeros: [Vec<Vec<u8>>; 3] = [
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
        ];
        // Raw (unshifted) nzeros for bitstream writing — stored at first-block positions
        let mut raw_nzeros: [Vec<Vec<u16>>; 3] = [
            vec![vec![0u16; xsize_blocks]; ysize_blocks],
            vec![vec![0u16; xsize_blocks]; ysize_blocks],
            vec![vec![0u16; xsize_blocks]; ysize_blocks],
        ];

        let channels = [xyb_x, xyb_y, xyb_b];

        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                // Skip non-first blocks of multi-block transforms
                if !ac_strategy.is_first(bx, by) {
                    continue;
                }

                let raw_strategy = ac_strategy.raw_strategy(bx, by);
                #[cfg(feature = "debug-dc")]
                eprintln!(
                    "Block (by={}, bx={}): raw_strategy={}",
                    by, bx, raw_strategy
                );
                let covered_x = ac_strategy.covered_blocks_x(bx, by);
                let covered_y = ac_strategy.covered_blocks_y(bx, by);
                let covered_blocks = covered_x * covered_y;
                let size = covered_blocks * DCT_BLOCK_SIZE;

                // CfL factors for this tile
                let tx = bx / TILE_DIM_IN_BLOCKS;
                let ty_cfl = by / TILE_DIM_IN_BLOCKS;
                let x_factor = ytox_ratio(cfl_map.ytox_at(tx, ty_cfl));
                let b_factor = ytob_ratio(cfl_map.ytob_at(tx, ty_cfl));

                // Coefficient layout: after C++ swap(cx,cy) so cx >= cy,
                // stride = cx * 8. Both DCT16X8 and DCT8X16 produce 8×16 layout.
                let (cx, cy) = if covered_y > covered_x {
                    (covered_y, covered_x)
                } else {
                    (covered_x, covered_y)
                };
                let block_width = cx * BLOCK_DIM;
                let block_height = cy * BLOCK_DIM;

                let x_qm_mul = 1.25f32.powf(params.x_qm_scale as f32 - 2.0);
                let b_qm_mul = 1.25f32.powf(params.b_qm_scale as f32 - 2.0);

                let mut dct_coeffs: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0f32; size]);

                // ── Step 1: DCT Y channel ──────────────────────────────────
                Self::apply_dct(
                    channels[1],
                    padded_width,
                    bx,
                    by,
                    raw_strategy,
                    &mut dct_coeffs[1],
                );

                // ── Step 2: Extract Y DC (before roundtrip quantization) ───
                // Inlined instead of using extract_dc to avoid borrow conflict.
                {
                    let inv_factor = INV_DC_QUANT[1] * params.scale_dc;
                    match raw_strategy {
                        0 => {
                            #[cfg(feature = "debug-dc")]
                            eprintln!(
                                "DCT8 Y DC: dct[0]={:.6}, inv_factor={:.4}, scale_dc={:.6}, quant_dc={}",
                                dct_coeffs[1][0],
                                inv_factor,
                                params.scale_dc,
                                (dct_coeffs[1][0] * inv_factor).round() as i16
                            );
                            quant_dc[1][by][bx] = (dct_coeffs[1][0] * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT16X8 => {
                            let coeffs_arr: [f32; 128] = dct_coeffs[1][..128]
                                .try_into()
                                .expect("128 coefficients for DCT16x8");
                            let dcs = dc_from_dct_16x8(&coeffs_arr);
                            for iy in 0..2 {
                                quant_dc[1][by + iy][bx] = (dcs[iy] * inv_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT8X16 => {
                            let coeffs_arr: [f32; 128] = dct_coeffs[1][..128]
                                .try_into()
                                .expect("128 coefficients for DCT8x16");
                            let dcs = dc_from_dct_8x16(&coeffs_arr);
                            for ix in 0..2 {
                                quant_dc[1][by][bx + ix] = (dcs[ix] * inv_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT16X16 => {
                            let coeffs_arr: [f32; 256] = dct_coeffs[1][..256]
                                .try_into()
                                .expect("256 coefficients for DCT16x16");
                            let dcs = dc_from_dct_16x16(&coeffs_arr);
                            // dcs = [dc00, dc01, dc10, dc11] in row-major 2x2
                            #[cfg(feature = "debug-dc")]
                            eprintln!(
                                "DCT16x16 block (by={}, bx={}): dcs=[{:.4}, {:.4}, {:.4}, {:.4}], LLF=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                by,
                                bx,
                                dcs[0],
                                dcs[1],
                                dcs[2],
                                dcs[3],
                                coeffs_arr[0],
                                coeffs_arr[1],
                                coeffs_arr[16],
                                coeffs_arr[17]
                            );
                            for iy in 0..2 {
                                for ix in 0..2 {
                                    let qdc = (dcs[iy * 2 + ix] * inv_factor).round() as i16;
                                    #[cfg(feature = "debug-dc")]
                                    eprintln!(
                                        "  quant_dc[1][{}][{}] = {} (raw dc={:.4}, inv_factor={:.4})",
                                        by + iy,
                                        bx + ix,
                                        qdc,
                                        dcs[iy * 2 + ix],
                                        inv_factor
                                    );
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X32 => {
                            let coeffs_arr: [f32; 1024] = dct_coeffs[1][..1024]
                                .try_into()
                                .expect("1024 coefficients for DCT32x32");
                            let dcs = dc_from_dct_32x32(&coeffs_arr);
                            #[cfg(feature = "debug-dc")]
                            eprintln!(
                                "DCT32x32 block (by={}, bx={}): dcs[0..4]=[{:.4}, {:.4}, {:.4}, {:.4}], LLF=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                by,
                                bx,
                                dcs[0],
                                dcs[1],
                                dcs[2],
                                dcs[3],
                                coeffs_arr[0],
                                coeffs_arr[1],
                                coeffs_arr[32],
                                coeffs_arr[33]
                            );
                            // dcs = 16 DC values in row-major 4x4
                            for iy in 0..4 {
                                for ix in 0..4 {
                                    let qdc = (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                    #[cfg(feature = "debug-dc")]
                                    eprintln!(
                                        "  quant_dc[1][{}][{}] = {} (raw dc={:.4})",
                                        by + iy,
                                        bx + ix,
                                        qdc,
                                        dcs[iy * 4 + ix]
                                    );
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X16 => {
                            // DCT32X16: 4×2 blocks, returns 8 DC values in row-major 4x2
                            let coeffs_arr: [f32; 512] = dct_coeffs[1][..512]
                                .try_into()
                                .expect("512 coefficients for DCT32x16");
                            let dcs = dc_from_dct_32x16(&coeffs_arr);
                            for iy in 0..4 {
                                for ix in 0..2 {
                                    let qdc = (dcs[iy * 2 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT16X32 => {
                            // DCT16X32: 2×4 blocks, returns 8 DC values in row-major 2x4
                            let coeffs_arr: [f32; 512] = dct_coeffs[1][..512]
                                .try_into()
                                .expect("512 coefficients for DCT16x32");
                            let dcs = dc_from_dct_16x32(&coeffs_arr);
                            for iy in 0..2 {
                                for ix in 0..4 {
                                    let qdc = (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X64 => {
                            // DCT64X64: 8×8 blocks, returns 64 DC values in row-major 8x8
                            let coeffs_arr: [f32; 4096] = dct_coeffs[1][..4096]
                                .try_into()
                                .expect("4096 coefficients for DCT64x64");
                            let dcs = dc_from_dct_64x64(&coeffs_arr);
                            for iy in 0..8 {
                                for ix in 0..8 {
                                    let qdc = (dcs[iy * 8 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X32 => {
                            // DCT64X32: 8×4 blocks, returns 32 DC values in row-major 8x4
                            let coeffs_arr: [f32; 2048] = dct_coeffs[1][..2048]
                                .try_into()
                                .expect("2048 coefficients for DCT64x32");
                            let dcs = dc_from_dct_64x32(&coeffs_arr);
                            for iy in 0..8 {
                                for ix in 0..4 {
                                    let qdc = (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X64 => {
                            // DCT32X64: 4×8 blocks, returns 32 DC values in row-major 4x8
                            let coeffs_arr: [f32; 2048] = dct_coeffs[1][..2048]
                                .try_into()
                                .expect("2048 coefficients for DCT32x64");
                            let dcs = dc_from_dct_32x64(&coeffs_arr);
                            for iy in 0..4 {
                                for ix in 0..8 {
                                    let qdc = (dcs[iy * 8 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT4X8 => {
                            // DCT4X8 full covers 1×1 blocks, returns single DC
                            let coeffs_arr: [f32; 64] = dct_coeffs[1][..64]
                                .try_into()
                                .expect("64 coefficients for DCT4X8");
                            let dc = dc_from_dct_4x8_full(&coeffs_arr);
                            quant_dc[1][by][bx] = (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT8X4 => {
                            // DCT8X4 full covers 1×1 blocks, returns single DC
                            let coeffs_arr: [f32; 64] = dct_coeffs[1][..64]
                                .try_into()
                                .expect("64 coefficients for DCT8X4");
                            let dc = dc_from_dct_8x4_full(&coeffs_arr);
                            quant_dc[1][by][bx] = (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT4X4 => {
                            // DCT4X4 full covers 1×1 blocks, returns single DC
                            let coeffs_arr: [f32; 64] = dct_coeffs[1][..64]
                                .try_into()
                                .expect("64 coefficients for DCT4X4");
                            let dc = dc_from_dct_4x4_full(&coeffs_arr);
                            quant_dc[1][by][bx] = (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                        | RAW_STRATEGY_AFV3 => {
                            // AFV covers 1×1 blocks, returns single DC
                            let coeffs_arr: [f32; 64] = dct_coeffs[1][..64]
                                .try_into()
                                .expect("64 coefficients for AFV");
                            let dc = dc_from_afv(&coeffs_arr);
                            quant_dc[1][by][bx] = (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_IDENTITY | RAW_STRATEGY_DCT2X2 => {
                            // IDENTITY/DCT2X2: 1×1 coverage, DC at position [0]
                            quant_dc[1][by][bx] = (dct_coeffs[1][0] * inv_factor).round() as i16;
                        }
                        _ => unreachable!(),
                    }
                }

                // ── Step 2b: DCT X and B channels (before AdjustQuantBlockAC) ──
                // libjxl DCTs all 3 channels before running AdjustQuantBlockAC.
                // X/B coefficients here are pre-CfL (CfL subtraction happens later in Step 6).
                for &c in &[0usize, 2] {
                    Self::apply_dct(
                        channels[c],
                        padded_width,
                        bx,
                        by,
                        raw_strategy,
                        &mut dct_coeffs[c],
                    );
                }

                // ── Step 2c: AdjustQuantBlockAC ──────────────────────────────
                // Ported from libjxl enc_group.cc. Adjusts per-block quant and
                // Y thresholds based on coefficient statistics across all 3 channels.
                // Takes max quant adjustment across channels, saves Y thresholds.
                let mut thresholds_y;
                let qac;
                {
                    let quant_idx = by * xsize_blocks + bx;
                    let mut quant_int = quant_field[quant_idx] as i32;
                    let orig_qac = params.scale * quant_int as f32;
                    thresholds_y = [0.58f32, 0.64, 0.64, 0.64];
                    let mut max_quant = quant_int;
                    for &c in &[1usize, 0, 2] {
                        let mut thres = [0.58f32, 0.64, 0.64, 0.64];
                        let mut quant_c = quant_int;
                        let qm_mul = if c == 0 {
                            x_qm_mul
                        } else if c == 2 {
                            b_qm_mul
                        } else {
                            1.0
                        };
                        let weights_c = super::quant::quant_weights(raw_strategy as usize, c);
                        Self::adjust_quant_block_ac(
                            &dct_coeffs[c],
                            weights_c,
                            orig_qac,
                            qm_mul,
                            c,
                            raw_strategy,
                            block_width,
                            block_height,
                            cx,
                            cy,
                            &mut thres,
                            &mut quant_c,
                        );
                        if c == 1 {
                            thresholds_y = thres;
                        }
                        max_quant = max_quant.max(quant_c);
                    }
                    quant_int = max_quant;
                    // Write adjusted quant back (decoder sees this in AC metadata)
                    quant_field[quant_idx] = quant_int.clamp(1, 255) as u8;
                    qac = params.scale * quant_int as f32;
                }

                // ── Step 3: Quantize Y AC with thresholding ────────────────
                {
                    let c = 1;
                    let weights = super::quant::quant_weights(raw_strategy as usize, c);
                    Self::quantize_ac_block(
                        &dct_coeffs[c],
                        weights,
                        qac,
                        1.0, // no x_qm_mul for Y
                        &thresholds_y,
                        block_width,
                        block_height,
                        covered_x,
                        covered_y,
                        covered_blocks,
                        size,
                        raw_strategy,
                        bx,
                        by,
                        &mut quant_ac[c],
                        self.error_diffusion,
                    );
                }

                // ── Step 4: Dequantize Y back (AdjustQuantBias roundtrip) ──
                // C++ QuantizeRoundtripYBlockAC: quantize all → dequantize all.
                // We already quantized AC; now also quantize LLF (temporarily)
                // and dequantize everything back into dct_coeffs[1].
                {
                    let weights = super::quant::quant_weights(raw_strategy as usize, 1);
                    let inv_qac = 1.0 / qac;
                    // Use post-swap dimensions for grid (matches C++ and quantize_ac_block)
                    for idx in 0..size {
                        // LLF positions: (y, x) where y < cy and x < cx in the grid
                        let is_llf = (idx / block_width) < cy && (idx % block_width) < cx;
                        let q = if is_llf {
                            // LLF: not stored in quant_ac, compute inline
                            // C++ QuantizeBlockAC quantizes all positions including LLF
                            let y = idx / block_width;
                            let x = idx % block_width;
                            Self::quantize_coeff_ac(
                                dct_coeffs[1][idx],
                                1.0 / weights[idx],
                                qac,
                                1.0,
                                &thresholds_y,
                                y,
                                x,
                                block_height,
                                block_width,
                            )
                        } else {
                            // Use flat layout: idx indexes into a grid of block_width x block_height
                            let y = idx / block_width;
                            let x = idx % block_width;
                            let coef_slot_y = y / BLOCK_DIM;
                            let coef_slot_x = x / BLOCK_DIM;
                            let pos_y = y % BLOCK_DIM;
                            let pos_x = x % BLOCK_DIM;
                            let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                            // Same transpose_slots logic as quantize_ac_block
                            let transpose_slots = covered_y > covered_x;
                            let (phys_row_off, phys_col_off) = if transpose_slots {
                                (coef_slot_x, coef_slot_y)
                            } else {
                                (coef_slot_y, coef_slot_x)
                            };
                            quant_ac[1][by + phys_row_off][bx + phys_col_off][pos_in_8x8]
                        };
                        let adj = Self::adjust_quant_bias(q, 1);
                        dct_coeffs[1][idx] = adj * weights[idx] * inv_qac;
                    }
                }

                // ── Step 5: CfL on AC coefficients using roundtripped Y ───
                // X/B DCTs were done in Step 2b (before AdjustQuantBlockAC).
                // C++ applies CfL to ALL positions (0..size) including DC/LLF,
                // but the decoder's DequantBlock calls LowestFrequenciesFromDC
                // AFTER DequantLane, overwriting LLF positions with DC-derived
                // values. So coefficient-level CfL on LLF is discarded by the
                // decoder. We skip LLF here; DC CfL uses dc_cfl_factor instead.
                #[allow(clippy::needless_range_loop)]
                // k used for LLF check and indexing two arrays
                for k in 0..size {
                    let is_llf = (k / block_width) < cy && (k % block_width) < cx;
                    if !is_llf {
                        dct_coeffs[0][k] -= x_factor * dct_coeffs[1][k];
                        dct_coeffs[2][k] -= b_factor * dct_coeffs[1][k];
                    }
                }

                // ── Step 7: Extract X/B DC + quantize X/B AC ───────────────
                for &c in &[0usize, 2] {
                    let dc_cfl_factor = if c == 2 { 0.5f32 } else { 0.0f32 };
                    let inv_factor = INV_DC_QUANT[c] * params.scale_dc;
                    let qm_multiplier = if c == 0 {
                        x_qm_mul
                    } else if c == 2 {
                        b_qm_mul
                    } else {
                        1.0
                    };

                    // Extract DC from CfL-adjusted coefficients.
                    // Read Y DC into temporaries to avoid borrow conflict
                    // (can't have &quant_dc[1] and &mut quant_dc[c] simultaneously).
                    match raw_strategy {
                        0 => {
                            let dc = dct_coeffs[c][0];
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT16X8 => {
                            let coeffs_arr: [f32; 128] = dct_coeffs[c][..128]
                                .try_into()
                                .expect("128 coefficients for DCT16x8");
                            let dcs = dc_from_dct_16x8(&coeffs_arr);
                            for iy in 0..2 {
                                let y_dc = quant_dc[1][by + iy][bx] as f32;
                                quant_dc[c][by + iy][bx] =
                                    (dcs[iy] * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT8X16 => {
                            let coeffs_arr: [f32; 128] = dct_coeffs[c][..128]
                                .try_into()
                                .expect("128 coefficients for DCT8x16");
                            let dcs = dc_from_dct_8x16(&coeffs_arr);
                            for ix in 0..2 {
                                let y_dc = quant_dc[1][by][bx + ix] as f32;
                                quant_dc[c][by][bx + ix] =
                                    (dcs[ix] * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT16X16 => {
                            let coeffs_arr: [f32; 256] = dct_coeffs[c][..256]
                                .try_into()
                                .expect("256 coefficients for DCT16x16");
                            let dcs = dc_from_dct_16x16(&coeffs_arr);
                            for iy in 0..2 {
                                for ix in 0..2 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 2 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X32 => {
                            let coeffs_arr: [f32; 1024] = dct_coeffs[c][..1024]
                                .try_into()
                                .expect("1024 coefficients for DCT32x32");
                            let dcs = dc_from_dct_32x32(&coeffs_arr);
                            for iy in 0..4 {
                                for ix in 0..4 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X16 => {
                            // DCT32X16: 2 cols × 4 rows coverage
                            let coeffs_arr: [f32; 512] = dct_coeffs[c][..512]
                                .try_into()
                                .expect("512 coefficients for DCT32x16");
                            let dcs = dc_from_dct_32x16(&coeffs_arr);
                            for iy in 0..4 {
                                for ix in 0..2 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 2 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT16X32 => {
                            // DCT16X32: 4 cols × 2 rows coverage
                            let coeffs_arr: [f32; 512] = dct_coeffs[c][..512]
                                .try_into()
                                .expect("512 coefficients for DCT16x32");
                            let dcs = dc_from_dct_16x32(&coeffs_arr);
                            for iy in 0..2 {
                                for ix in 0..4 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X64 => {
                            let coeffs_arr: [f32; 4096] = dct_coeffs[c][..4096]
                                .try_into()
                                .expect("4096 coefficients for DCT64x64");
                            let dcs = dc_from_dct_64x64(&coeffs_arr);
                            for iy in 0..8 {
                                for ix in 0..8 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 8 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X32 => {
                            let coeffs_arr: [f32; 2048] = dct_coeffs[c][..2048]
                                .try_into()
                                .expect("2048 coefficients for DCT64x32");
                            let dcs = dc_from_dct_64x32(&coeffs_arr);
                            for iy in 0..8 {
                                for ix in 0..4 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X64 => {
                            let coeffs_arr: [f32; 2048] = dct_coeffs[c][..2048]
                                .try_into()
                                .expect("2048 coefficients for DCT32x64");
                            let dcs = dc_from_dct_32x64(&coeffs_arr);
                            for iy in 0..4 {
                                for ix in 0..8 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 8 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT4X8 => {
                            let coeffs_arr: [f32; 64] = dct_coeffs[c][..64]
                                .try_into()
                                .expect("64 coefficients for DCT4X8");
                            let dc = dc_from_dct_4x8_full(&coeffs_arr);
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT8X4 => {
                            let coeffs_arr: [f32; 64] = dct_coeffs[c][..64]
                                .try_into()
                                .expect("64 coefficients for DCT8X4");
                            let dc = dc_from_dct_8x4_full(&coeffs_arr);
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT4X4 => {
                            let coeffs_arr: [f32; 64] = dct_coeffs[c][..64]
                                .try_into()
                                .expect("64 coefficients for DCT4X4");
                            let dc = dc_from_dct_4x4_full(&coeffs_arr);
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                        | RAW_STRATEGY_AFV3 => {
                            // AFV covers 1×1 blocks, returns single DC
                            let coeffs_arr: [f32; 64] = dct_coeffs[c][..64]
                                .try_into()
                                .expect("64 coefficients for AFV");
                            let dc = dc_from_afv(&coeffs_arr);
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_IDENTITY | RAW_STRATEGY_DCT2X2 => {
                            // IDENTITY/DCT2X2: 1×1 coverage, DC at position [0]
                            let dc = dct_coeffs[c][0];
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        _ => unreachable!(),
                    }

                    // Quantize AC with thresholding
                    // libjxl uses [0.58, 0.62, 0.62, 0.62] for X/B channels
                    // (different from libjxl-tiny's per-channel adjustments)
                    let thresholds_xb = Self::default_thresholds(c, covered_x, covered_y);
                    let weights = super::quant::quant_weights(raw_strategy as usize, c);
                    Self::quantize_ac_block(
                        &dct_coeffs[c],
                        weights,
                        qac,
                        qm_multiplier,
                        &thresholds_xb,
                        block_width,
                        block_height,
                        covered_x,
                        covered_y,
                        covered_blocks,
                        size,
                        raw_strategy,
                        bx,
                        by,
                        &mut quant_ac[c],
                        self.error_diffusion,
                    );
                }

                // ── Step 8: Count non-zeros for all 3 channels ─────────────
                let transpose_slots = covered_y > covered_x;
                for c in 0..3 {
                    if covered_blocks == 1 {
                        num_nonzero_8x8_except_dc(&quant_ac[c][by][bx], &mut nzeros[c][by][bx]);
                        raw_nzeros[c][by][bx] = nzeros[c][by][bx] as u16;
                    } else {
                        // Build flat block in cx*8 × cy*8 layout (stride = cx*8).
                        // num_nonzero_except_llf expects block[y * stride + x] for y,x in 0..cy*8, 0..cx*8.
                        // The 8x8 block storage uses quant_ac[slot_by][slot_bx][pos_in_8x8].
                        let stride = cx * BLOCK_DIM;
                        let full_block: Vec<i32> = (0..size)
                            .map(|idx| {
                                // idx = y * stride + x in the flat layout
                                let y = idx / stride;
                                let x = idx % stride;
                                // Which 8x8 block slot in coefficient space
                                let coef_slot_y = y / BLOCK_DIM;
                                let coef_slot_x = x / BLOCK_DIM;
                                // Position within the 8x8 block
                                let pos_y = y % BLOCK_DIM;
                                let pos_x = x % BLOCK_DIM;
                                let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                                // Map to physical block offset
                                let (phys_row_off, phys_col_off) = if transpose_slots {
                                    (coef_slot_x, coef_slot_y)
                                } else {
                                    (coef_slot_y, coef_slot_x)
                                };
                                quant_ac[c][by + phys_row_off][bx + phys_col_off][pos_in_8x8]
                            })
                            .collect();
                        let flat_len = (covered_y - 1) * xsize_blocks + covered_x;
                        let mut flat_nz = vec![0u8; flat_len];
                        let raw_nz = num_nonzero_except_llf(
                            cx,
                            cy,
                            &full_block,
                            xsize_blocks,
                            &mut flat_nz,
                            covered_x,
                            covered_y,
                        );
                        for dy in 0..covered_y {
                            for dx in 0..covered_x {
                                nzeros[c][by + dy][bx + dx] = flat_nz[dx + dy * xsize_blocks];
                            }
                        }
                        raw_nzeros[c][by][bx] = raw_nz;
                    }
                }
            }
        }

        (quant_dc, quant_ac, nzeros, raw_nzeros)
    }
}
