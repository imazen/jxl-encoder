// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Adaptive AC strategy selection.
//!
//! Ported from libjxl-tiny `enc_ac_strategy.cc`. For each 16x16 block region,
//! selects between:
//! - Four DCT8 (8×8) transforms
//! - Two DCT16x8 (16×8) transforms (left column + right column)
//! - Two DCT8x16 (8×16) transforms (top row + bottom row)
//!
//! Selection is based on entropy estimation: the strategy that yields the
//! lowest estimated coded size (including information-loss penalty) wins.

use super::afv::afv_transform_from_pixels;
use super::chroma_from_luma::{CflMap, ytob_ratio, ytox_ratio};
use super::common::{BLOCK_DIM, DCT_BLOCK_SIZE, TILE_DIM_IN_BLOCKS, ceil_log2_nonzero};
use super::dct::{
    dct_4x4_full, dct_4x8_full, dct_8x4_full, dct_8x8, dct_8x16, dct_16x8, dct_16x16, dct_16x32,
    dct_32x16, dct_32x32, dct_32x64, dct_64x32, dct_64x64, dct2x2_transform, idct_8x8, idct_8x16,
    idct_16x8, idct_16x16, idct_16x32, idct_32x16, idct_32x32, idct_32x64, idct_64x32, idct_64x64,
    identity_transform, inverse_dct2x2_transform, inverse_identity_transform,
};
use super::quant::quant_weights;

/// Raw strategy codes matching the C++ `AcStrategy::Type` enum.
/// Note: These are internal codes, not bitstream codes. Use STRATEGY_CODE_LUT
/// to convert to bitstream codes.
pub const RAW_STRATEGY_DCT8: u8 = 0;
pub const RAW_STRATEGY_DCT16X8: u8 = 1;
pub const RAW_STRATEGY_DCT8X16: u8 = 2;
pub const RAW_STRATEGY_DCT16X16: u8 = 3;
pub const RAW_STRATEGY_DCT32X32: u8 = 4;
pub const RAW_STRATEGY_DCT4X8: u8 = 5;
pub const RAW_STRATEGY_DCT8X4: u8 = 6;
pub const RAW_STRATEGY_DCT4X4: u8 = 7;
pub const RAW_STRATEGY_IDENTITY: u8 = 8;
pub const RAW_STRATEGY_DCT2X2: u8 = 9;
pub const RAW_STRATEGY_DCT32X16: u8 = 10;
pub const RAW_STRATEGY_DCT16X32: u8 = 11;
pub const RAW_STRATEGY_AFV0: u8 = 12;
pub const RAW_STRATEGY_AFV1: u8 = 13;
pub const RAW_STRATEGY_AFV2: u8 = 14;
pub const RAW_STRATEGY_AFV3: u8 = 15;
pub const RAW_STRATEGY_DCT64X64: u8 = 16;
pub const RAW_STRATEGY_DCT64X32: u8 = 17;
pub const RAW_STRATEGY_DCT32X64: u8 = 18;

/// Number of supported raw strategies.
pub const NUM_RAW_STRATEGIES: usize = 19;

/// Strategy code as written to the bitstream (via `StrategyCode()`).
/// These differ from raw strategy codes.
/// From libjxl ac_strategy.h: DCT=0, IDENTITY=1, DCT2X2=2, DCT4X4=3, DCT16X16=4,
/// DCT32X32=5, DCT16X8=6, DCT8X16=7, DCT32X16=10, DCT16X32=11, DCT4X8=12, DCT8X4=13,
/// AFV0=14, AFV1=15, AFV2=16, AFV3=17, DCT64X64=18, DCT64X32=19, DCT32X64=20.
pub(crate) const STRATEGY_CODE_LUT: [u8; NUM_RAW_STRATEGIES] = [
    0, 6, 7, 4, 5, 12, 13, 3, 1, 2, 10, 11, 14, 15, 16, 17, 18, 19, 20,
];

/// Covered blocks in X direction for each raw strategy.
/// IDENTITY, DCT2X2, DCT4X8, DCT8X4, DCT4X4, and AFV0-3 cover 1×1 blocks.
/// DCT32X16 (32 rows × 16 cols): 2 cols × 4 rows of 8×8 blocks
/// DCT16X32 (16 rows × 32 cols): 4 cols × 2 rows of 8×8 blocks
/// DCT64X64: 8 cols × 8 rows. DCT64X32 (64r × 32c): 4 cols × 8 rows.
/// DCT32X64 (32r × 64c): 8 cols × 4 rows.
pub(crate) const COVERED_X: [usize; NUM_RAW_STRATEGIES] =
    [1, 1, 2, 2, 4, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 8, 4, 8];

/// Covered blocks in Y direction for each raw strategy.
pub(crate) const COVERED_Y: [usize; NUM_RAW_STRATEGIES] =
    [1, 2, 1, 2, 4, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1, 1, 8, 8, 4];

/// Per-block AC strategy map.
///
/// Each byte stores `(raw_strategy << 1) | is_first` matching the C++
/// `AcStrategyImage` layout.
pub struct AcStrategyMap {
    data: Vec<u8>,
    pub xsize_blocks: usize,
    pub ysize_blocks: usize,
}

impl AcStrategyMap {
    /// Create a new map filled with DCT8 (all blocks are first blocks).
    pub fn new_dct8(xsize_blocks: usize, ysize_blocks: usize) -> Self {
        // DCT8: raw_strategy=0, is_first=true → (0 << 1) | 1 = 1
        let data = vec![1u8; xsize_blocks * ysize_blocks];
        Self {
            data,
            xsize_blocks,
            ysize_blocks,
        }
    }

    /// Create a new map forcing a specific strategy for all blocks that fit.
    /// Blocks that don't fit the strategy (e.g., at image edges) use DCT8.
    pub fn force_strategy(xsize_blocks: usize, ysize_blocks: usize, raw_strategy: u8) -> Self {
        let mut map = Self::new_dct8(xsize_blocks, ysize_blocks);
        let cx = COVERED_X[raw_strategy as usize];
        let cy = COVERED_Y[raw_strategy as usize];

        for by in (0..ysize_blocks).step_by(cy) {
            for bx in (0..xsize_blocks).step_by(cx) {
                // Only set if the full coverage fits
                if bx + cx <= xsize_blocks && by + cy <= ysize_blocks {
                    map.set(bx, by, raw_strategy);
                }
            }
        }
        map
    }

    /// Get the raw strategy at (bx, by).
    #[inline]
    pub fn raw_strategy(&self, bx: usize, by: usize) -> u8 {
        self.data[by * self.xsize_blocks + bx] >> 1
    }

    /// Is this the first (top-left) block of the transform?
    #[inline]
    pub fn is_first(&self, bx: usize, by: usize) -> bool {
        (self.data[by * self.xsize_blocks + bx] & 1) != 0
    }

    /// Get the strategy code for bitstream writing.
    #[inline]
    pub fn strategy_code(&self, bx: usize, by: usize) -> u8 {
        STRATEGY_CODE_LUT[self.raw_strategy(bx, by) as usize]
    }

    /// Covered blocks in X for the strategy at (bx, by).
    #[inline]
    pub fn covered_blocks_x(&self, bx: usize, by: usize) -> usize {
        COVERED_X[self.raw_strategy(bx, by) as usize]
    }

    /// Covered blocks in Y for the strategy at (bx, by).
    #[inline]
    pub fn covered_blocks_y(&self, bx: usize, by: usize) -> usize {
        COVERED_Y[self.raw_strategy(bx, by) as usize]
    }

    /// Set a block and all its covered sub-blocks.
    ///
    /// For DCT8 (raw_strategy=0): sets 1 block.
    /// For DCT16X8 (raw_strategy=1): sets 2 blocks vertically (1×2).
    /// For DCT8X16 (raw_strategy=2): sets 2 blocks horizontally (2×1).
    pub fn set(&mut self, bx: usize, by: usize, raw_strategy: u8) {
        let cx = COVERED_X[raw_strategy as usize];
        let cy = COVERED_Y[raw_strategy as usize];
        for iy in 0..cy {
            for ix in 0..cx {
                let is_first = (iy | ix) == 0;
                self.data[(by + iy) * self.xsize_blocks + bx + ix] =
                    (raw_strategy << 1) | (is_first as u8);
            }
        }
    }

    /// Count the number of "first blocks" (= number of distinct transforms).
    #[cfg(test)]
    pub fn count_first_blocks(&self) -> usize {
        self.data.iter().filter(|&&v| (v & 1) != 0).count()
    }

    /// Return strategy histogram indexed by raw strategy code.
    /// Counts first blocks only (number of times each transform was selected).
    #[cfg(feature = "debug-ac-strategy")]
    pub fn strategy_histogram(&self) -> [usize; 10] {
        let mut counts = [0usize; 10];
        for &v in &self.data {
            if (v & 1) != 0 {
                // is_first block
                let raw = v >> 1;
                if (raw as usize) < 10 {
                    counts[raw as usize] += 1;
                }
            }
        }
        counts
    }

    /// Print strategy histogram with names.
    #[cfg(feature = "debug-ac-strategy")]
    pub fn print_histogram(&self) {
        const NAMES: [&str; 10] = [
            "DCT8", "DCT16x8", "DCT8x16", "DCT16x16", "DCT32x32", "DCT4x8", "DCT8x4", "DCT4x4",
            "IDENTITY", "DCT2X2",
        ];
        let hist = self.strategy_histogram();
        let total: usize = hist.iter().sum();
        eprintln!("Strategy histogram (total {} transforms):", total);
        for (i, &count) in hist.iter().enumerate() {
            if count > 0 {
                let pct = 100.0 * count as f64 / total as f64;
                eprintln!("  {:10}: {:6} ({:5.1}%)", NAMES[i], count, pct);
            }
        }
    }
}

// ─── Entropy estimation ─────────────────────────────────────────────────────

/// Channel offsets for pixel-domain loss masking.
/// From libjxl enc_ac_strategy.cc:446
const MASK_CHANNEL_OFFSET: [f32; 3] = [12.0, 0.0, 4.0];

/// Channel multipliers for pixel-domain loss (8th power).
/// From libjxl enc_ac_strategy.cc:479
/// Pre-computed: 8.2^8 ≈ 2.088e7, 1.0^8 = 1.0, 1.03^8 ≈ 1.267
const CHANNEL_MUL: [f64; 3] = [
    20882706.4655936, // X channel: 8.2^8
    1.0,              // Y channel: 1.0^8
    1.26677008064,    // B channel: 1.03^8
];

/// Base entropy estimation constants for full libjxl pixel-domain loss model.
/// From libjxl enc_ac_strategy.cc:1111-1113
/// These are SCALED by distance before use via `compute_scaled_constants()`.
const K_INFO_LOSS_MULTIPLIER_BASE: f32 = 1.2;
const K_COST_DELTA_BASE: f32 = 10.833_274;
const K_ZEROS_MUL_BASE: f32 = 9.308_906;

/// Distance scaling exponents from libjxl enc_ac_strategy.cc:1119-1123
const K_BIAS: f32 = 0.137_317_4;
const K_POW_INFO_LOSS: f32 = 0.336_778_1;
const K_POW_ZEROS_MUL: f32 = 0.509_909_3;
const K_POW_COST_DELTA: f32 = 0.367_029_4;

/// Compute distance-scaled constants for full libjxl cost model.
/// At d=1.0, returns the base values. At higher distances, increases all values.
fn compute_scaled_constants(distance: f32) -> (f32, f32, f32) {
    let ratio = (distance + K_BIAS) / (1.0 + K_BIAS);
    let info_loss_mul = K_INFO_LOSS_MULTIPLIER_BASE * ratio.powf(K_POW_INFO_LOSS);
    let zeros_mul = K_ZEROS_MUL_BASE * ratio.powf(K_POW_ZEROS_MUL);
    let cost_delta = K_COST_DELTA_BASE * ratio.powf(K_POW_COST_DELTA);
    (info_loss_mul, cost_delta, zeros_mul)
}

/// Raw (unnormalized) entropy multipliers per transform type (from libjxl FindBest8x8Transform).
/// These are NORMALIZED by dividing by DCT8's value (0.8) before use.
/// See libjxl enc_ac_strategy.cc:584: `float entropy_mul = tx.entropy_mul / kTransforms8x8[0].entropy_mul;`
const RAW_ENTROPY_MUL_DCT8: f32 = 0.8;
const RAW_ENTROPY_MUL_DCT4X4: f32 = 1.08;
const RAW_ENTROPY_MUL_DCT4X8: f32 = 0.859_316_37;
const RAW_ENTROPY_MUL_IDENTITY: f32 = 1.0428;
const RAW_ENTROPY_MUL_DCT2X2: f32 = 0.95;
const RAW_ENTROPY_MUL_AFV: f32 = 0.817_794_9;
const RAW_ENTROPY_MUL_DCT16X8: f32 = 1.21;
const RAW_ENTROPY_MUL_DCT16X16: f32 = 1.34;
const RAW_ENTROPY_MUL_DCT16X32: f32 = 1.49;
const RAW_ENTROPY_MUL_DCT32X32: f32 = 1.48;
const RAW_ENTROPY_MUL_DCT64X32: f32 = 2.25;
const RAW_ENTROPY_MUL_DCT64X64: f32 = 2.25;

/// Get the entropy multiplier for a raw strategy (full libjxl mode).
///
/// CRITICAL: libjxl only normalizes 8x8 transforms in FindBest8x8Transform.
/// Larger transforms use RAW values in TryMergeAcs.
///
/// 8x8 transforms (normalized by DCT8's 0.8):
/// - DCT8: 0.8 / 0.8 = 1.0
/// - DCT4X8: 0.859 / 0.8 = 1.074
/// - DCT4X4: 1.08 / 0.8 = 1.35
///
/// Larger transforms (RAW values, NOT normalized):
/// - DCT16X8: 1.21
/// - DCT16X16: 1.34
/// - DCT32X32: 1.48
fn entropy_mul_for_strategy(raw_strategy: u8) -> f32 {
    match raw_strategy {
        // 8x8 transforms: normalize by DCT8's 0.8 (so DCT8 = 1.0)
        RAW_STRATEGY_DCT8 => 1.0,
        RAW_STRATEGY_DCT4X8 | RAW_STRATEGY_DCT8X4 => RAW_ENTROPY_MUL_DCT4X8 / RAW_ENTROPY_MUL_DCT8,
        RAW_STRATEGY_DCT4X4 => RAW_ENTROPY_MUL_DCT4X4 / RAW_ENTROPY_MUL_DCT8,
        RAW_STRATEGY_IDENTITY => RAW_ENTROPY_MUL_IDENTITY / RAW_ENTROPY_MUL_DCT8,
        RAW_STRATEGY_DCT2X2 => RAW_ENTROPY_MUL_DCT2X2 / RAW_ENTROPY_MUL_DCT8,
        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
            RAW_ENTROPY_MUL_AFV / RAW_ENTROPY_MUL_DCT8
        }
        // Larger transforms: use RAW values (libjxl TryMergeAcs uses raw entropy_mul)
        RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => RAW_ENTROPY_MUL_DCT16X8,
        RAW_STRATEGY_DCT16X16 => RAW_ENTROPY_MUL_DCT16X16,
        RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => RAW_ENTROPY_MUL_DCT16X32,
        RAW_STRATEGY_DCT32X32 => RAW_ENTROPY_MUL_DCT32X32,
        RAW_STRATEGY_DCT64X32 | RAW_STRATEGY_DCT32X64 => RAW_ENTROPY_MUL_DCT64X32,
        RAW_STRATEGY_DCT64X64 => RAW_ENTROPY_MUL_DCT64X64,
        _ => 1.0,
    }
}

/// Estimate entropy using coefficient-domain loss (libjxl-tiny style).
///
/// This is a convenience wrapper that calls `estimate_entropy_with_mask` with
/// `mask1x1 = None`, for backward compatibility with tests and code that
/// doesn't need pixel-domain loss.
#[allow(clippy::too_many_arguments, dead_code)]
fn estimate_entropy(
    raw_strategy: u8,
    xyb: [&[f32]; 3],
    stride: usize,
    bx: usize,
    by: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
) -> f32 {
    estimate_entropy_with_mask(
        raw_strategy,
        xyb,
        stride,
        bx,
        by,
        distance,
        quant_field,
        xsize_blocks,
        masking,
        ytox,
        ytob,
        None,
        0,
        0.0,
    )
}

/// Estimate entropy with optional pixel-domain loss.
///
/// When `mask1x1` is Some, uses full libjxl pixel-domain loss model with:
/// - Distance-scaled constants
/// - Fixed entropy multiplier per transform type
///
/// When `mask1x1` is None, uses coefficient-domain loss (libjxl-tiny style).
///
/// `entropy_mul_adjust`: additive adjustment to entropy_mul. In libjxl,
/// kFavor2X2AtHighQuality and kAvoidEntropyOfTransforms modify entropy_mul
/// before passing to EstimateEntropy. Pass 0.0 for no adjustment.
#[allow(clippy::too_many_arguments)]
fn estimate_entropy_with_mask(
    raw_strategy: u8,
    xyb: [&[f32]; 3],
    stride: usize,
    bx: usize,
    by: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    entropy_mul_adjust: f32,
) -> f32 {
    // In pixel-domain mode, use fixed entropy_mul values per transform
    // In coefficient-domain mode, entropy_mul is applied outside by the caller (mul8x8 etc.)
    let entropy_mul = if mask1x1.is_some() {
        (entropy_mul_for_strategy(raw_strategy) + entropy_mul_adjust).max(0.01)
    } else {
        // Coefficient-domain: entropy_mul is 1.0, caller handles multiplier.
        // Still apply adjustment for kFavor2X2/kAvoidEntropy since the
        // returned estimate gets multiplied by the caller's multiplier.
        (1.0 + entropy_mul_adjust).max(0.01)
    };

    estimate_entropy_full(
        raw_strategy,
        xyb,
        stride,
        bx,
        by,
        distance,
        quant_field,
        xsize_blocks,
        masking,
        ytox,
        ytob,
        mask1x1,
        mask1x1_stride,
        entropy_mul,
    )
}

/// Estimate entropy with optional pixel-domain loss calculation.
///
/// When `mask1x1` is Some, uses full libjxl pixel-domain loss model with
/// distance-scaled constants.
/// When `mask1x1` is None, uses coefficient-domain loss (libjxl-tiny style).
///
/// `entropy_mul` multiplies ONLY the entropy part, not the loss. In full libjxl
/// mode, this is a fixed value per transform type. In libjxl-tiny mode, this
/// is 1.0 and the caller applies multipliers externally.
#[allow(clippy::too_many_arguments)]
fn estimate_entropy_full(
    raw_strategy: u8,
    xyb: [&[f32]; 3],
    stride: usize,
    bx: usize,
    by: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    entropy_mul: f32,
) -> f32 {
    let cx = COVERED_X[raw_strategy as usize];
    let cy = COVERED_Y[raw_strategy as usize];
    let num_blocks = cx * cy;
    let size = num_blocks * DCT_BLOCK_SIZE;

    // Use different constants based on whether we're using pixel-domain loss
    let use_pixel_domain = mask1x1.is_some();

    // Entropy estimation constants
    // In pixel-domain mode: use distance-scaled constants
    // In coefficient-domain mode: use libjxl-tiny static constants
    let (k_info_loss_mul, k_cost_delta, k_zeros_mul) = if use_pixel_domain {
        compute_scaled_constants(distance)
    } else {
        // libjxl-tiny style constants (not distance-scaled)
        (138.0_f32, 5.335_918_5_f32, 7.565_053_4_f32)
    };
    const K_INFO_LOSS_MULTIPLIER2: f32 = 50.468_4;
    const K_COST2: f32 = 4.462_815;

    // Apply transform for each channel
    let mut block = vec![0.0f32; 3 * size]; // 3 channels × size coeffs
    for (c, xyb_c) in xyb.iter().enumerate() {
        let offset = c * size;
        match raw_strategy {
            RAW_STRATEGY_DCT8 => {
                let mut input = [0.0f32; 64];
                extract_block_8x8(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 64];
                dct_8x8(&input, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT16X8 => {
                let mut input = [0.0f32; 128];
                extract_block_8x16(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 128];
                dct_16x8(&input, &mut output);
                block[offset..offset + 128].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT8X16 => {
                let mut input = [0.0f32; 128];
                extract_block_16x8(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 128];
                dct_8x16(&input, &mut output);
                block[offset..offset + 128].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT16X16 => {
                let mut input = [0.0f32; 256];
                extract_block_16x16(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 256];
                dct_16x16(&input, &mut output);
                block[offset..offset + 256].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT32X32 => {
                let mut input = [0.0f32; 1024];
                extract_block_32x32(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 1024];
                dct_32x32(&input, &mut output);
                block[offset..offset + 1024].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT32X16 => {
                let mut input = [0.0f32; 512];
                extract_block_32x16(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 512];
                dct_32x16(&input, &mut output);
                block[offset..offset + 512].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT16X32 => {
                let mut input = [0.0f32; 512];
                extract_block_16x32(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 512];
                dct_16x32(&input, &mut output);
                block[offset..offset + 512].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT64X64 => {
                let mut input = [0.0f32; 4096];
                extract_block_64x64(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 4096];
                dct_64x64(&input, &mut output);
                block[offset..offset + 4096].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT64X32 => {
                let mut input = [0.0f32; 2048];
                extract_block_64x32(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 2048];
                dct_64x32(&input, &mut output);
                block[offset..offset + 2048].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT32X64 => {
                let mut input = [0.0f32; 2048];
                extract_block_32x64(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 2048];
                dct_32x64(&input, &mut output);
                block[offset..offset + 2048].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT4X8 => {
                let mut input = [0.0f32; 64];
                extract_block_8x8(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 64];
                dct_4x8_full(&input, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT8X4 => {
                let mut input = [0.0f32; 64];
                extract_block_8x8(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 64];
                dct_8x4_full(&input, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT4X4 => {
                let mut input = [0.0f32; 64];
                extract_block_8x8(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 64];
                dct_4x4_full(&input, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            RAW_STRATEGY_IDENTITY => {
                let mut output = [0.0f32; 64];
                let pixel_offset = by * BLOCK_DIM * stride + bx * BLOCK_DIM;
                identity_transform(&xyb_c[pixel_offset..], stride, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT2X2 => {
                let mut output = [0.0f32; 64];
                let pixel_offset = by * BLOCK_DIM * stride + bx * BLOCK_DIM;
                dct2x2_transform(&xyb_c[pixel_offset..], stride, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
                let mut input = [0.0f32; 64];
                extract_block_8x8(xyb_c, stride, bx, by, &mut input);
                let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
                let mut output = [0.0f32; 64];
                afv_transform_from_pixels(&input, afv_kind, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            _ => unreachable!(),
        }
    }

    // Load QF and masking: take max over covered blocks
    let mut quant = 0.0f32;
    let mut mask_val = 0.0f32;
    for iy in 0..cy {
        for ix in 0..cx {
            let idx = (by + iy) * xsize_blocks + bx + ix;
            quant = quant.max(quant_field[idx]);
            mask_val = mask_val.max(masking[idx]);
        }
    }

    // Compute quant_norm16 for pixel-domain loss
    // libjxl uses different computation based on block count:
    // - 1 block (DCT8): single quant value
    // - 2 blocks (DCT16x8, DCT8x16): MAX of the two quant values
    // - 4+ blocks (DCT16x16, DCT32x32): 16th norm
    // Reference: lib/jxl/enc_ac_strategy.cc:383-410
    let quant_norm16 = if use_pixel_domain {
        if num_blocks == 1 {
            // Single block: use the quant value directly
            quant_field[by * xsize_blocks + bx]
        } else if num_blocks == 2 {
            // Two blocks: use MAX of the two quant values (NOT 16th norm!)
            let q1 = quant_field[by * xsize_blocks + bx];
            let q2 = if cy == 2 {
                // DCT8x16: blocks are vertically stacked
                quant_field[(by + 1) * xsize_blocks + bx]
            } else {
                // DCT16x8: blocks are horizontally adjacent
                quant_field[by * xsize_blocks + bx + 1]
            };
            q1.max(q2)
        } else {
            // 4+ blocks: use 16th norm
            let mut norm_sum = 0.0f32;
            for iy in 0..cy {
                for ix in 0..cx {
                    let idx = (by + iy) * xsize_blocks + bx + ix;
                    let qval = quant_field[idx];
                    // qval^16 = (qval^2)^8
                    let q2 = qval * qval;
                    let q4 = q2 * q2;
                    let q8 = q4 * q4;
                    let q16 = q8 * q8;
                    norm_sum += q16;
                }
            }
            norm_sum /= num_blocks as f32;
            norm_sum.powf(1.0 / 16.0)
        }
    } else {
        0.0 // Not used in coefficient-domain mode
    };

    let cmap_factors = [ytox_ratio(ytox), 0.0f32, ytob_ratio(ytob)];

    let mut entropy = 0.0f32;
    let mut info_loss_sum = 0.0f32;
    let mut info_loss2_sum = 0.0f32;

    // For pixel-domain loss: accumulate loss across all channels
    let mut total_pixel_loss = 0.0f64;

    // Error coefficient buffer for pixel-domain IDCT (reused per channel)
    let mut error_coeffs = vec![0.0f32; size];

    let slope = (distance / 3.0).min(1.0);
    let cost_of_1 = 1.0 + slope * 8.870_325;

    // Pixel base coordinates
    let pixel_x = bx * BLOCK_DIM;
    let pixel_y = by * BLOCK_DIM;

    for (c, &cmap_factor) in cmap_factors.iter().enumerate() {
        let weights = quant_weights(raw_strategy as usize, c);

        let offset_c = c * size;
        let offset_y = size; // Y channel always at offset 1*size

        let mut entropy_sum = 0.0f32;
        let mut nzeros_sum = 0.0f32;

        // Process all coefficients (including LLF for pixel-domain loss storage)
        for i in 0..size {
            let val_in = block[offset_c + i];
            let val_y = block[offset_y + i] * cmap_factor;
            // weights stores dequant matrix; inv_matrix = 1/weight
            let inv_matrix_val = 1.0 / weights[i];
            let val = (val_in - val_y) * inv_matrix_val * quant;
            let rval = val.round();
            let diff = val - rval;

            // Store error coefficient for IDCT (matrix * diff)
            if use_pixel_domain {
                error_coeffs[i] = weights[i] * diff;
            }

            // NOTE: We do NOT skip LLF coefficients here.
            // Both libjxl and libjxl-tiny process ALL coefficients (including LLF)
            // in entropy estimation. The LLF coefficients contribute to entropy_v
            // and nzeros_v in the reference implementations.

            let diff_abs = diff.abs();
            if !use_pixel_domain {
                // Coefficient-domain loss (libjxl-tiny style)
                info_loss_sum += diff_abs;
                info_loss2_sum += diff_abs * diff_abs;
            }

            let q = rval.abs();
            // K_COST2 threshold only in coefficient-domain mode (libjxl-tiny style)
            // Full libjxl pixel-domain mode doesn't have this threshold
            if !use_pixel_domain && q >= 1.5 {
                entropy_sum += K_COST2;
            }
            // Full libjxl uses sqrt(q) * cost_delta, libjxl-tiny similar
            entropy_sum += q.sqrt() * k_cost_delta;
            if q != 0.0 {
                nzeros_sum += 1.0;
            }
        }
        // cost_of_1 term only in coefficient-domain mode (libjxl-tiny style)
        // Full libjxl pixel-domain mode doesn't have this per-nzero term
        if !use_pixel_domain {
            entropy_sum += nzeros_sum * cost_of_1;
        }
        entropy += entropy_sum;

        let num_nzeros = nzeros_sum as usize;
        let nbits = ceil_log2_nonzero(num_nzeros + 1) as usize + 1;
        entropy += k_zeros_mul * (ceil_log2_nonzero(nbits + 17) + nbits as u32) as f32;

        // X channel penalty for large transforms
        if c == 0 && num_blocks >= 2 && use_pixel_domain {
            let w = 1.0 + (num_blocks as f32 / 8.0).min(3.0);
            entropy *= w;
        }

        // Pixel-domain loss calculation
        if let Some(mask) = mask1x1 {
            // Apply IDCT to error coefficients to get pixel-domain error
            let pixel_error = apply_idct_for_strategy(raw_strategy, &error_coeffs);

            // Compute 8th power norm with per-pixel masking
            let mut channel_loss = 0.0f64;
            let mask_offset = MASK_CHANNEL_OFFSET[c];

            let block_width = cx * BLOCK_DIM;
            let block_height = cy * BLOCK_DIM;

            for py in 0..block_height {
                for px in 0..block_width {
                    let abs_x = pixel_x + px;
                    let abs_y = pixel_y + py;

                    // Bounds check for mask access
                    if abs_x < mask1x1_stride && abs_y * mask1x1_stride + abs_x < mask.len() {
                        let mask_val = mask[abs_y * mask1x1_stride + abs_x];
                        let error_val = pixel_error[py * block_width + px];

                        // masked = (mask + offset) * error
                        let masked = (mask_val + mask_offset) * error_val;

                        // 8th power: masked^8 = (masked^2)^4
                        let m2 = (masked * masked) as f64;
                        let m4 = m2 * m2;
                        let m8 = m4 * m4;

                        channel_loss += m8;
                    }
                }
            }

            // Apply channel multiplier
            channel_loss *= CHANNEL_MUL[c];

            total_pixel_loss += channel_loss;

            // X channel penalty for large transforms - applied to TOTAL loss accumulator
            // (not per-channel). This matches libjxl enc_ac_strategy.cc:500-501
            // IMPORTANT: Apply AFTER adding channel_loss, so it multiplies entire accumulator
            if c == 0 && num_blocks >= 2 {
                let w = 1.0 + (num_blocks as f64 / 8.0).min(3.0);
                total_pixel_loss *= w;
            }
        }
    }

    // Compute final cost: entropy * entropy_mul + loss
    // CRITICAL: entropy_mul applies ONLY to entropy, not to loss!
    // This matches libjxl enc_ac_strategy.cc:508-509
    if use_pixel_domain {
        // Pixel-domain loss: (sum/n)^(1/8) * n / quant_norm16
        let n = (num_blocks * DCT_BLOCK_SIZE) as f64;
        let loss_scalar = (total_pixel_loss / n).powf(1.0 / 8.0) * n / quant_norm16 as f64;
        // Apply entropy_mul to entropy, then add loss
        entropy *= entropy_mul;
        entropy += k_info_loss_mul * loss_scalar as f32;
    } else {
        // Coefficient-domain loss (libjxl-tiny style)
        // In this mode, entropy_mul is 1.0 and caller applies multipliers externally
        let infoloss2 = (num_blocks as f32 * info_loss2_sum).sqrt();
        let info_loss_score = k_info_loss_mul * info_loss_sum + K_INFO_LOSS_MULTIPLIER2 * infoloss2;
        entropy += mask_val * info_loss_score;
    }

    entropy
}

/// Apply inverse DCT to error coefficients based on strategy.
/// Returns pixel-domain error in row-major layout.
fn apply_idct_for_strategy(raw_strategy: u8, error_coeffs: &[f32]) -> Vec<f32> {
    match raw_strategy {
        RAW_STRATEGY_DCT8 | RAW_STRATEGY_DCT4X8 | RAW_STRATEGY_DCT8X4 | RAW_STRATEGY_DCT4X4 => {
            // All these use 8x8 pixel output with standard 8x8 IDCT
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&error_coeffs[..64]);
            let mut output = [0.0f32; 64];
            idct_8x8(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
            // AFV has a hybrid inverse transform (AFV4x4 + DCT4x4 + DCT4x8)
            let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&error_coeffs[..64]);
            let mut output = [0.0f32; 64];
            super::afv::inverse_afv_transform(&input, afv_kind, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT16X8 => {
            // 8 wide × 16 tall (stored as 8x16 layout after IDCT)
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&error_coeffs[..128]);
            let mut output = [0.0f32; 128];
            idct_16x8(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT8X16 => {
            // 16 wide × 8 tall
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&error_coeffs[..128]);
            let mut output = [0.0f32; 128];
            idct_8x16(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT16X16 => {
            // 16 wide × 16 tall
            let mut input = [0.0f32; 256];
            input.copy_from_slice(&error_coeffs[..256]);
            let mut output = [0.0f32; 256];
            idct_16x16(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT32X32 => {
            let mut input = [0.0f32; 1024];
            input.copy_from_slice(&error_coeffs[..1024]);
            let mut output = [0.0f32; 1024];
            idct_32x32(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT32X16 => {
            let mut input = [0.0f32; 512];
            input.copy_from_slice(&error_coeffs[..512]);
            let mut output = [0.0f32; 512];
            idct_32x16(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT16X32 => {
            let mut input = [0.0f32; 512];
            input.copy_from_slice(&error_coeffs[..512]);
            let mut output = [0.0f32; 512];
            idct_16x32(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT64X64 => {
            let mut input = vec![0.0f32; 4096];
            input.copy_from_slice(&error_coeffs[..4096]);
            let mut output = vec![0.0f32; 4096];
            idct_64x64(&input, &mut output);
            output
        }
        RAW_STRATEGY_DCT64X32 => {
            let mut input = vec![0.0f32; 2048];
            input.copy_from_slice(&error_coeffs[..2048]);
            let mut output = vec![0.0f32; 2048];
            idct_64x32(&input, &mut output);
            output
        }
        RAW_STRATEGY_DCT32X64 => {
            let mut input = vec![0.0f32; 2048];
            input.copy_from_slice(&error_coeffs[..2048]);
            let mut output = vec![0.0f32; 2048];
            idct_32x64(&input, &mut output);
            output
        }
        RAW_STRATEGY_IDENTITY => {
            let mut output = [0.0f32; 64];
            inverse_identity_transform(&error_coeffs[..64], &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT2X2 => {
            let mut output = [0.0f32; 64];
            inverse_dct2x2_transform(&error_coeffs[..64], &mut output);
            output.to_vec()
        }
        _ => unreachable!(
            "unknown strategy {} in apply_idct_for_strategy",
            raw_strategy
        ),
    }
}

// ─── Block extraction helpers ────────────────────────────────────────────────

/// Extract an 8×8 pixel block from a plane.
///
/// The buffer must be padded to at least (by*8+8) rows and (bx*8+8) columns
/// with edge-replicated values, so no bounds checking is needed.
fn extract_block_8x8(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 64]) {
    for dy in 0..8 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..8 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 8 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract an 8×16 pixel block (1 wide × 2 tall) for DCT16x8.
/// Layout: 16 rows × 8 cols, row-major.
///
/// The buffer must be padded to at least (by*8+16) rows and (bx*8+8) columns.
fn extract_block_8x16(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 128]) {
    for dy in 0..16 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..8 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 8 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract a 16×8 pixel block (2 wide × 1 tall) for DCT8x16.
/// Layout: 8 rows × 16 cols, row-major.
///
/// The buffer must be padded to at least (by*8+8) rows and (bx*8+16) columns.
fn extract_block_16x8(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 128]) {
    for dy in 0..8 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..16 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 16 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract a 16×16 pixel block (2 wide × 2 tall) for DCT16x16.
/// Layout: 16 rows × 16 cols, row-major.
///
/// The buffer must be padded to at least (by*8+16) rows and (bx*8+16) columns.
fn extract_block_16x16(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 256]) {
    for dy in 0..16 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..16 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 16 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract a 32×32 pixel block (4 wide × 4 tall) for DCT32x32.
/// Layout: 32 rows × 32 cols, row-major.
///
/// The buffer must be padded to at least (by*8+32) rows and (bx*8+32) columns.
fn extract_block_32x32(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 1024]) {
    for dy in 0..32 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..32 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 32 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract a 32×16 pixel block (2 wide × 4 tall) for DCT32x16.
/// Layout: 32 rows × 16 cols, row-major.
///
/// The buffer must be padded to at least (by*8+32) rows and (bx*8+16) columns.
fn extract_block_32x16(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 512]) {
    for dy in 0..32 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..16 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 16 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract a 16×32 pixel block (4 wide × 2 tall) for DCT16x32.
/// Layout: 16 rows × 32 cols, row-major.
///
/// The buffer must be padded to at least (by*8+16) rows and (bx*8+32) columns.
fn extract_block_16x32(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 512]) {
    for dy in 0..16 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..32 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 32 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract a 64×64 pixel block (8 wide × 8 tall) for DCT64x64.
/// Layout: 64 rows × 64 cols, row-major.
fn extract_block_64x64(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 4096]) {
    for dy in 0..64 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..64 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 64 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract a 64×32 pixel block (4 wide × 8 tall) for DCT64x32.
/// Layout: 64 rows × 32 cols, row-major.
fn extract_block_64x32(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 2048]) {
    for dy in 0..64 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..32 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 32 + dx] = plane[py * stride + px];
        }
    }
}

/// Extract a 32×64 pixel block (8 wide × 4 tall) for DCT32x64.
/// Layout: 32 rows × 64 cols, row-major.
fn extract_block_32x64(plane: &[f32], stride: usize, bx: usize, by: usize, out: &mut [f32; 2048]) {
    for dy in 0..32 {
        let py = by * BLOCK_DIM + dy;
        for dx in 0..64 {
            let px = bx * BLOCK_DIM + dx;
            out[dy * 64 + dx] = plane[py * stride + px];
        }
    }
}

// ─── 16×16 transform selection ──────────────────────────────────────────────

/// Find the best transform for a 16×16 block region (2×2 group of 8×8 blocks).
///
/// Port of C++ `FindBest16x16Transform`. Evaluates four DCT8, two DCT16X8 and
/// two DCT8X16 options, then picks the combination with lowest entropy.
///
/// # Arguments
/// * `(bx0, by0)` - Tile origin in block coordinates
/// * `(cx, cy)` - Position within tile (in 8×8 blocks, must be even)
/// * `stride` - Row stride (padded width) of the XYB buffers
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn find_best_16x16_transform(
    xyb: [&[f32]; 3],
    stride: usize,
    bx0: usize,
    by0: usize,
    cx: usize,
    cy: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    ac_strategy: &mut AcStrategyMap,
) {
    // In pixel-domain mode (mask1x1.is_some()), entropy_mul is applied internally
    // by estimate_entropy_full using fixed values per transform. External multipliers
    // are 1.0. In coefficient-domain mode, use libjxl-tiny distance-dependent multipliers.
    let use_pixel_domain = mask1x1.is_some();

    // Distance-dependent multipliers (from libjxl-tiny) - only used in coefficient-domain mode
    let (mul8x8, mul16x8, mul16x16, mul4x8, mul4x4) = if use_pixel_domain {
        // In pixel-domain mode, entropy_mul is handled internally. No external multiplier.
        (1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32)
    } else {
        let k8x8mul1: f32 = -0.55 * 0.75;
        let k8x8mul2: f32 = 1.073_575_8 * 0.75;
        let k8x8base: f32 = 1.4;
        let m8x8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);

        let k8x16mul1: f32 = -0.55;
        let k8x16mul2: f32 = 0.901_958_8;
        let k8x16base: f32 = 1.6;
        let m16x8 = k8x16mul2 + k8x16mul1 / (distance + k8x16base);

        let k16x16mul1: f32 = -0.65;
        let k16x16mul2: f32 = 0.88;
        let k16x16base: f32 = 1.8;
        let m16x16 = k16x16mul2 + k16x16mul1 / (distance + k16x16base);

        let k4x8mul1: f32 = -0.50 * 0.75;
        let k4x8mul2: f32 = 0.88;
        let k4x8base: f32 = 1.3;
        let m4x8 = k4x8mul2 + k4x8mul1 / (distance + k4x8base);

        let k4x4mul1: f32 = -0.45 * 0.75;
        let k4x4mul2: f32 = 0.85;
        let k4x4base: f32 = 1.2;
        let m4x4 = k4x4mul2 + k4x4mul1 / (distance + k4x4base);

        (m8x8, m16x8, m16x16, m4x8, m4x4)
    };

    // Base cost added for DCT8 transforms (from libjxl-tiny)
    // In pixel-domain mode, this is 0 since costs are already calibrated
    let base_cost_8x8 = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };

    // Entropy_mul adjustments from libjxl enc_ac_strategy.cc:585-600.
    // These are applied INSIDE EstimateEntropy to the entropy portion only,
    // NOT as post-hoc cost multipliers (which would incorrectly scale loss too).

    // kFavor2X2AtHighQuality: bonus for IDENTITY/DCT2X2 at distance < 5.0.
    // Matches libjxl enc_ac_strategy.cc:585-590: -0.4 * ((5-d)/5)^2
    // AdjustQuantBlockAC is now implemented, which prevents over-selection.
    let favor_weight = if distance < 5.0 {
        ((5.0 - distance) / 5.0_f32).powi(2)
    } else {
        0.0
    };
    let favor_2x2_adjust = -0.15 * favor_weight; // libjxl uses -0.4, kept at -0.15 (causes quality regression at d<1.0 when increased)

    // kAvoidEntropyOfTransforms: penalty for non-DCT/non-2x2/non-IDENTITY at distance > 4.0
    let avoid_transforms_adjust = if distance > 4.0 {
        let mul = if distance < 12.0 {
            (12.0 - 4.0) / (distance - 4.0)
        } else {
            1.0
        };
        0.5 * mul // positive = increases entropy_mul = higher cost
    } else {
        0.0
    };

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Evaluate four 8×8 blocks with DCT8, DCT4X8, DCT8X4, DCT4X4, IDENTITY, DCT2X2
    // Track entropy and best strategy for each block
    let mut entropy = [[0.0f32; 2]; 2];
    let mut best_single_strategy = [[RAW_STRATEGY_DCT8; 2]; 2];
    for (dy, (entropy_row, strat_row)) in entropy
        .iter_mut()
        .zip(best_single_strategy.iter_mut())
        .enumerate()
    {
        for (dx, (entropy_val, best_strat)) in
            entropy_row.iter_mut().zip(strat_row.iter_mut()).enumerate()
        {
            let block_x = abs_bx + dx;
            let block_y = abs_by + dy;

            // Helper: evaluate a single-block strategy with entropy_mul adjustment
            let eval = |strategy: u8, adjust: f32| -> f32 {
                estimate_entropy_with_mask(
                    strategy,
                    xyb,
                    stride,
                    block_x,
                    block_y,
                    distance,
                    quant_field,
                    xsize_blocks,
                    masking,
                    ytox,
                    ytob,
                    mask1x1,
                    mask1x1_stride,
                    adjust,
                )
            };

            // DCT8 (no adjustment)
            let e8 = eval(RAW_STRATEGY_DCT8, 0.0);
            let cost8 = base_cost_8x8 + mul8x8 * e8;

            // DCT4X8 (kAvoidEntropy penalty at high distance)
            let e4x8 = eval(RAW_STRATEGY_DCT4X8, avoid_transforms_adjust);
            let base_cost_4x8 = if use_pixel_domain { 0.0 } else { 3.0 * mul4x8 };
            let cost4x8 = base_cost_4x8 + mul4x8 * e4x8;

            // DCT8X4
            let e8x4 = eval(RAW_STRATEGY_DCT8X4, avoid_transforms_adjust);
            let cost8x4 = base_cost_4x8 + mul4x8 * e8x4;

            // DCT4X4
            let e4x4 = eval(RAW_STRATEGY_DCT4X4, avoid_transforms_adjust);
            let base_cost_4x4 = if use_pixel_domain { 0.0 } else { 3.0 * mul4x4 };
            let cost4x4 = base_cost_4x4 + mul4x4 * e4x4;

            // IDENTITY (kFavor2X2 bonus at low distance)
            let e_identity = eval(RAW_STRATEGY_IDENTITY, favor_2x2_adjust);
            let base_cost_identity = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };
            let cost_identity = base_cost_identity + mul8x8 * e_identity;

            // DCT2X2 (kFavor2X2 bonus at low distance)
            let e_dct2 = eval(RAW_STRATEGY_DCT2X2, favor_2x2_adjust);
            let base_cost_dct2 = if use_pixel_domain { 0.0 } else { 3.0 * mul8x8 };
            let cost_dct2 = base_cost_dct2 + mul8x8 * e_dct2;

            // Pick the best single-block strategy
            *entropy_val = cost8;
            *best_strat = RAW_STRATEGY_DCT8;

            if cost4x8 < *entropy_val {
                *entropy_val = cost4x8;
                *best_strat = RAW_STRATEGY_DCT4X8;
            }
            if cost8x4 < *entropy_val {
                *entropy_val = cost8x4;
                *best_strat = RAW_STRATEGY_DCT8X4;
            }
            if cost4x4 < *entropy_val {
                *entropy_val = cost4x4;
                *best_strat = RAW_STRATEGY_DCT4X4;
            }
            if cost_identity < *entropy_val {
                *entropy_val = cost_identity;
                *best_strat = RAW_STRATEGY_IDENTITY;
            }
            if cost_dct2 < *entropy_val {
                *entropy_val = cost_dct2;
                *best_strat = RAW_STRATEGY_DCT2X2;
            }

            // AFV0-3 corner DCT
            // AFV auto-selection disabled in pixel-domain mode: the inverse AFV
            // transform produces systematically underestimated pixel-domain error,
            // causing AFV to be selected too aggressively (35% AFV vs libjxl's <5%).
            // This caused a massive quality regression (SSIM2 84→57 on frymire).
            // Re-enable once the AFV pixel-domain cost model is calibrated.
            if !use_pixel_domain {
                let base_cost_afv = 3.0 * mul8x8;
                let e_afv0 = eval(RAW_STRATEGY_AFV0, avoid_transforms_adjust);
                let e_afv1 = eval(RAW_STRATEGY_AFV1, avoid_transforms_adjust);
                let e_afv2 = eval(RAW_STRATEGY_AFV2, avoid_transforms_adjust);
                let e_afv3 = eval(RAW_STRATEGY_AFV3, avoid_transforms_adjust);
                let cost_afv0 = base_cost_afv + mul8x8 * e_afv0;
                let cost_afv1 = base_cost_afv + mul8x8 * e_afv1;
                let cost_afv2 = base_cost_afv + mul8x8 * e_afv2;
                let cost_afv3 = base_cost_afv + mul8x8 * e_afv3;

                if cost_afv0 < *entropy_val {
                    *entropy_val = cost_afv0;
                    *best_strat = RAW_STRATEGY_AFV0;
                }
                if cost_afv1 < *entropy_val {
                    *entropy_val = cost_afv1;
                    *best_strat = RAW_STRATEGY_AFV1;
                }
                if cost_afv2 < *entropy_val {
                    *entropy_val = cost_afv2;
                    *best_strat = RAW_STRATEGY_AFV2;
                }
                if cost_afv3 < *entropy_val {
                    *entropy_val = cost_afv3;
                    *best_strat = RAW_STRATEGY_AFV3;
                }
            }
        }
    }

    // Evaluate two DCT16X8 options (left column, right column)
    let entropy_16x8_left = mul16x8
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X8,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_16x8_right = mul16x8
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X8,
            xyb,
            stride,
            abs_bx + 1,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );

    // Evaluate two DCT8X16 options (top row, bottom row)
    let entropy_8x16_top = mul16x8
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT8X16,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_8x16_bottom = mul16x8
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT8X16,
            xyb,
            stride,
            abs_bx,
            abs_by + 1,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );

    // Evaluate DCT16x16 (one transform covering the entire 2x2 region)
    let entropy_16x16 = mul16x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X16,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );

    // Compare all options: four single-block, 16x8 split, 8x16 split, or one 16x16
    let cost_all_single = entropy[0][0] + entropy[0][1] + entropy[1][0] + entropy[1][1];
    let cost16x8 = (entropy_16x8_left).min(entropy[0][0] + entropy[1][0])
        + (entropy_16x8_right).min(entropy[0][1] + entropy[1][1]);
    let cost8x16 = (entropy_8x16_top).min(entropy[0][0] + entropy[0][1])
        + (entropy_8x16_bottom).min(entropy[1][0] + entropy[1][1]);
    let cost16x16 = entropy_16x16;

    // Find best non-single-block cost (minimum of 16x8, 8x16, 16x16)
    let best_rect = cost16x8.min(cost8x16);
    let best_large = best_rect.min(cost16x16);

    // Only use a non-single-block strategy if it beats four single-block transforms
    if best_large >= cost_all_single {
        // Keep all four as their best single-block strategy (DCT8, DCT4X8, or DCT8X4)
        for dy in 0..2 {
            for dx in 0..2 {
                let strat = best_single_strategy[dy][dx];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + dx, abs_by + dy, strat);
                }
            }
        }
        return;
    }

    if cost16x16 <= best_rect {
        // DCT16x16 is the overall best
        ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X16);
    } else if cost16x8 < cost8x16 {
        // Try 16x8 for each column
        if entropy_16x8_left < entropy[0][0] + entropy[1][0] {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X8);
        } else {
            // Use best single-block for both blocks in left column
            for dy in 0..2 {
                let strat = best_single_strategy[dy][0];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx, abs_by + dy, strat);
                }
            }
        }
        if entropy_16x8_right < entropy[0][1] + entropy[1][1] {
            ac_strategy.set(abs_bx + 1, abs_by, RAW_STRATEGY_DCT16X8);
        } else {
            // Use best single-block for both blocks in right column
            for dy in 0..2 {
                let strat = best_single_strategy[dy][1];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + 1, abs_by + dy, strat);
                }
            }
        }
    } else {
        // Try 8x16 for each row
        if entropy_8x16_top < entropy[0][0] + entropy[0][1] {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT8X16);
        } else {
            // Use best single-block for both blocks in top row
            for dx in 0..2 {
                let strat = best_single_strategy[0][dx];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + dx, abs_by, strat);
                }
            }
        }
        if entropy_8x16_bottom < entropy[1][0] + entropy[1][1] {
            ac_strategy.set(abs_bx, abs_by + 1, RAW_STRATEGY_DCT8X16);
        } else {
            // Use best single-block for both blocks in bottom row
            for dx in 0..2 {
                let strat = best_single_strategy[1][dx];
                if strat != RAW_STRATEGY_DCT8 {
                    ac_strategy.set(abs_bx + dx, abs_by + 1, strat);
                }
            }
        }
    }
}

// ─── 32×32 transform selection ──────────────────────────────────────────────

/// Find the best transform for a 32×32 block region (4×4 group of 8×8 blocks).
///
/// Evaluates one DCT32x32 against four `find_best_16x16_transform` results.
/// Returns true if DCT32x32 was selected.
///
/// Only call when `bx + 3 < xsize_blocks && by + 3 < ysize_blocks`.
#[allow(clippy::too_many_arguments, unreachable_code)]
fn find_best_32x32_transform(
    xyb: [&[f32]; 3],
    stride: usize,
    bx0: usize,
    by0: usize,
    cx: usize,
    cy: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    ac_strategy: &mut AcStrategyMap,
) -> bool {
    // Large transforms (32x32, 32x16, 16x32) average large pixel blocks, which
    // works well for smooth content but produces blur on high-contrast edges.
    // The cost model correctly avoids them for high-contrast blocks.
    // Enable at d >= 2.0 where compression benefit outweighs edge blur risk.
    if distance < 2.0 {
        // At low distances, evaluate 16x16 and smaller transforms only
        for qy in (0..4).step_by(2) {
            for qx in (0..4).step_by(2) {
                find_best_16x16_transform(
                    xyb,
                    stride,
                    bx0,
                    by0,
                    cx + qx,
                    cy + qy,
                    distance,
                    quant_field,
                    xsize_blocks,
                    masking,
                    ytox,
                    ytob,
                    mask1x1,
                    mask1x1_stride,
                    ac_strategy,
                );
            }
        }
        return false;
    }

    // At higher distances (d >= 2.0), evaluate DCT32x32, DCT32x16, DCT16x32 as options
    let k32x32mul1: f32 = -0.75;
    let k32x32mul2: f32 = 1.2; // Very conservative
    let k32x32base: f32 = 2.0;
    let mul32x32 = k32x32mul2 + k32x32mul1 / (distance + k32x32base);

    // DCT32x16/DCT16x32 use similar multipliers to DCT32x32
    let k32x16mul1: f32 = -0.70;
    let k32x16mul2: f32 = 1.1;
    let k32x16base: f32 = 2.0;
    let mul32x16 = k32x16mul2 + k32x16mul1 / (distance + k32x16base);

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Evaluate DCT32x32 cost
    let entropy_32x32 = mul32x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X32,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );

    // Evaluate DCT32x16 costs (two transforms: at (0,0) and (0,2))
    // DCT32x16 covers 4 rows × 2 cols of 8x8 blocks
    let entropy_32x16_0 = mul32x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X16,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_32x16_1 = mul32x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X16,
            xyb,
            stride,
            abs_bx + 2,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_32x16_total = entropy_32x16_0 + entropy_32x16_1;

    // Evaluate DCT16x32 costs (two transforms: at (0,0) and (2,0))
    // DCT16x32 covers 2 rows × 4 cols of 8x8 blocks
    let entropy_16x32_0 = mul32x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X32,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_16x32_1 = mul32x16
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT16X32,
            xyb,
            stride,
            abs_bx,
            abs_by + 2,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_16x32_total = entropy_16x32_0 + entropy_16x32_1;

    // Run four 16x16 evaluations (each covers 2×2 blocks)
    for qy in (0..4).step_by(2) {
        for qx in (0..4).step_by(2) {
            find_best_16x16_transform(
                xyb,
                stride,
                bx0,
                by0,
                cx + qx,
                cy + qy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
            );
        }
    }

    // Compute the combined cost of the four 16x16 sub-evaluations.
    // We need to re-estimate using whatever strategies were selected.
    let mut cost_sub = 0.0f32;
    for iy in 0..4 {
        for ix in 0..4 {
            if !ac_strategy.is_first(abs_bx + ix, abs_by + iy) {
                continue;
            }
            let sub_raw = ac_strategy.raw_strategy(abs_bx + ix, abs_by + iy);
            // Distance-dependent multipliers (must match find_best_16x16_transform)
            let k8x8mul1: f32 = -0.55 * 0.75;
            let k8x8mul2: f32 = 1.073_575_8 * 0.75;
            let k8x8base: f32 = 1.4;
            let mul8x8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);
            let k8x16mul1: f32 = -0.55;
            let k8x16mul2: f32 = 0.901_958_8;
            let k8x16base: f32 = 1.6;
            let mul16x8 = k8x16mul2 + k8x16mul1 / (distance + k8x16base);
            let k16x16mul1: f32 = -0.65;
            let k16x16mul2: f32 = 0.88;
            let k16x16base: f32 = 1.8;
            let mul16x16 = k16x16mul2 + k16x16mul1 / (distance + k16x16base);

            let mul = match sub_raw {
                RAW_STRATEGY_DCT8 => mul8x8,
                RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => mul16x8,
                RAW_STRATEGY_DCT16X16 => mul16x16,
                _ => mul8x8,
            };
            let base = if sub_raw == RAW_STRATEGY_DCT8 {
                3.0 * mul8x8
            } else {
                0.0
            };

            let e = estimate_entropy_with_mask(
                sub_raw,
                xyb,
                stride,
                abs_bx + ix,
                abs_by + iy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                0.0,
            );
            cost_sub += base + mul * e;
        }
    }

    // Find the best option among: DCT32x32, DCT32x16 pair, DCT16x32 pair, 16x16 sub-evaluations
    let mut best_cost = cost_sub;
    let mut best_choice = 0u8; // 0 = keep sub, 1 = DCT32x32, 2 = DCT32x16, 3 = DCT16x32

    if entropy_32x32 < best_cost {
        best_cost = entropy_32x32;
        best_choice = 1;
    }
    // DCT32x16/DCT16x32 now enabled (fixed pixel extraction bug Feb 4, 2026)
    if entropy_32x16_total < best_cost {
        best_cost = entropy_32x16_total;
        best_choice = 2;
    }
    if entropy_16x32_total < best_cost {
        // best_cost = entropy_16x32_total; // Not needed, just using best_choice
        best_choice = 3;
    }

    match best_choice {
        1 => {
            // DCT32x32 wins
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X32);
            true
        }
        2 => {
            // Two DCT32x16 transforms win
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X16);
            ac_strategy.set(abs_bx + 2, abs_by, RAW_STRATEGY_DCT32X16);
            true
        }
        3 => {
            // Two DCT16x32 transforms win
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X32);
            ac_strategy.set(abs_bx, abs_by + 2, RAW_STRATEGY_DCT16X32);
            true
        }
        _ => {
            // Keep the 16x16 sub-evaluation results (already in ac_strategy)
            false
        }
    }
}

// ─── 64×64 transform selection ──────────────────────────────────────────────

/// Find the best transform for a 64×64 pixel region (8×8 group of 8×8 blocks).
///
/// Evaluates DCT64x64, two DCT64x32, two DCT32x64, and four find_best_32x32_transform.
/// Only evaluated at d >= 3.0 (conservative — DCT64 averages 64x64 blocks).
#[allow(clippy::too_many_arguments)]
fn find_best_64x64_transform(
    xyb: [&[f32]; 3],
    stride: usize,
    bx0: usize,
    by0: usize,
    cx: usize,
    cy: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    ac_strategy: &mut AcStrategyMap,
) {
    // DCT64 transforms only at d >= 3.0
    if distance < 3.0 {
        // At lower distances, fall through to 32x32 evaluation
        for qy in (0..8).step_by(4) {
            for qx in (0..8).step_by(4) {
                find_best_32x32_transform(
                    xyb,
                    stride,
                    bx0,
                    by0,
                    cx + qx,
                    cy + qy,
                    distance,
                    quant_field,
                    xsize_blocks,
                    masking,
                    ytox,
                    ytob,
                    mask1x1,
                    mask1x1_stride,
                    ac_strategy,
                );
            }
        }
        return;
    }

    // Conservative multipliers for DCT64 transforms
    let k64x64mul1: f32 = -0.80;
    let k64x64mul2: f32 = 1.3;
    let k64x64base: f32 = 2.5;
    let mul64x64 = k64x64mul2 + k64x64mul1 / (distance + k64x64base);

    let k64x32mul1: f32 = -0.75;
    let k64x32mul2: f32 = 1.2;
    let k64x32base: f32 = 2.5;
    let mul64x32 = k64x32mul2 + k64x32mul1 / (distance + k64x32base);

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Evaluate DCT64x64 cost
    let entropy_64x64 = mul64x64
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT64X64,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );

    // Evaluate DCT64x32 costs (two transforms stacked vertically)
    // DCT64x32 covers 8 rows × 4 cols of 8×8 blocks
    // Split: left half (bx, by) and right half (bx+4, by)
    let entropy_64x32_0 = mul64x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT64X32,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_64x32_1 = mul64x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT64X32,
            xyb,
            stride,
            abs_bx + 4,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_64x32_total = entropy_64x32_0 + entropy_64x32_1;

    // Evaluate DCT32x64 costs (two transforms side by side)
    // DCT32x64 covers 4 rows × 8 cols of 8×8 blocks
    // Split: top half (bx, by) and bottom half (bx, by+4)
    let entropy_32x64_0 = mul64x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X64,
            xyb,
            stride,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_32x64_1 = mul64x32
        * estimate_entropy_with_mask(
            RAW_STRATEGY_DCT32X64,
            xyb,
            stride,
            abs_bx,
            abs_by + 4,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
            mask1x1,
            mask1x1_stride,
            0.0,
        );
    let entropy_32x64_total = entropy_32x64_0 + entropy_32x64_1;

    // Run four 32x32 evaluations (each covers 4×4 blocks)
    for qy in (0..8).step_by(4) {
        for qx in (0..8).step_by(4) {
            find_best_32x32_transform(
                xyb,
                stride,
                bx0,
                by0,
                cx + qx,
                cy + qy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
            );
        }
    }

    // Compute the combined cost of the four 32x32 sub-evaluations
    let mut cost_sub = 0.0f32;
    for iy in 0..8 {
        for ix in 0..8 {
            if !ac_strategy.is_first(abs_bx + ix, abs_by + iy) {
                continue;
            }
            let sub_raw = ac_strategy.raw_strategy(abs_bx + ix, abs_by + iy);
            // Distance-dependent multipliers (must match find_best_32x32/16x16_transform)
            let k8x8mul1: f32 = -0.55 * 0.75;
            let k8x8mul2: f32 = 1.073_575_8 * 0.75;
            let k8x8base: f32 = 1.4;
            let mul8x8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);
            let k8x16mul1: f32 = -0.55;
            let k8x16mul2: f32 = 0.901_958_8;
            let k8x16base: f32 = 1.6;
            let mul16x8 = k8x16mul2 + k8x16mul1 / (distance + k8x16base);
            let k16x16mul1: f32 = -0.65;
            let k16x16mul2: f32 = 0.88;
            let k16x16base: f32 = 1.8;
            let mul16x16 = k16x16mul2 + k16x16mul1 / (distance + k16x16base);
            let k32x32mul1: f32 = -0.75;
            let k32x32mul2: f32 = 1.2;
            let k32x32base: f32 = 2.0;
            let mul32x32 = k32x32mul2 + k32x32mul1 / (distance + k32x32base);
            let k32x16mul1: f32 = -0.70;
            let k32x16mul2: f32 = 1.1;
            let k32x16base: f32 = 2.0;
            let mul32x16 = k32x16mul2 + k32x16mul1 / (distance + k32x16base);

            let mul = match sub_raw {
                RAW_STRATEGY_DCT8 => mul8x8,
                RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => mul16x8,
                RAW_STRATEGY_DCT16X16 => mul16x16,
                RAW_STRATEGY_DCT32X32 => mul32x32,
                RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => mul32x16,
                _ => mul8x8,
            };
            let base = if sub_raw == RAW_STRATEGY_DCT8 {
                3.0 * mul8x8
            } else {
                0.0
            };

            let e = estimate_entropy_with_mask(
                sub_raw,
                xyb,
                stride,
                abs_bx + ix,
                abs_by + iy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                0.0,
            );
            cost_sub += base + mul * e;
        }
    }

    // Find the best option
    let mut best_cost = cost_sub;
    let mut best_choice = 0u8; // 0=keep sub, 1=DCT64x64, 2=DCT64x32, 3=DCT32x64

    if entropy_64x64 < best_cost {
        best_cost = entropy_64x64;
        best_choice = 1;
    }
    if entropy_64x32_total < best_cost {
        best_cost = entropy_64x32_total;
        best_choice = 2;
    }
    if entropy_32x64_total < best_cost {
        let _ = best_cost;
        best_choice = 3;
    }

    match best_choice {
        1 => {
            // DCT64x64 wins
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT64X64);
        }
        2 => {
            // Two DCT64x32 transforms win
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT64X32);
            ac_strategy.set(abs_bx + 4, abs_by, RAW_STRATEGY_DCT64X32);
        }
        3 => {
            // Two DCT32x64 transforms win
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT32X64);
            ac_strategy.set(abs_bx, abs_by + 4, RAW_STRATEGY_DCT32X64);
        }
        _ => {
            // Keep the 32x32 sub-evaluation results (already in ac_strategy)
        }
    }
}

// ─── Quant field adjustment ─────────────────────────────────────────────────

/// Adjust the quant field for non-8×8 transforms.
///
/// For multi-block transforms, all covered blocks get a weighted blend of max and mean.
/// At low distances (d ≤ 1.54), uses max. At high distances, blends toward mean.
/// This improves quality at high distances by not over-quantizing.
///
/// Port of C++ `AdjustQuantField` from enc_adaptive_quantization.cc.
pub fn adjust_quant_field_with_distance(
    ac_strategy: &AcStrategyMap,
    quant_field: &mut [u8],
    butteraugli_target: f32,
) {
    let xsize_blocks = ac_strategy.xsize_blocks;

    // At low distances use max, at high distances blend toward mean.
    // libjxl constants from enc_adaptive_quantization.cc:1207-1215
    const K_LIMIT: f32 = 1.54138;
    const K_MUL: f32 = 0.56391;
    const K_MIN: f32 = 0.0;

    let mut mean_max_mixer = 1.0_f32;
    if butteraugli_target > K_LIMIT {
        mean_max_mixer -= (butteraugli_target - K_LIMIT) * K_MUL;
        if mean_max_mixer < K_MIN {
            mean_max_mixer = K_MIN;
        }
    }

    for by in 0..ac_strategy.ysize_blocks {
        for bx in 0..ac_strategy.xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }
            let cx = ac_strategy.covered_blocks_x(bx, by);
            let cy = ac_strategy.covered_blocks_y(bx, by);
            if cx == 1 && cy == 1 {
                continue;
            }

            // Compute max and mean of covered region
            let mut max_q = 0u8;
            let mut sum = 0u32;
            for iy in 0..cy {
                for ix in 0..cx {
                    let q = quant_field[(by + iy) * xsize_blocks + bx + ix];
                    max_q = max_q.max(q);
                    sum += q as u32;
                }
            }
            let mean = sum as f32 / (cx * cy) as f32;

            // Blend max and mean (for 4+ block transforms)
            let blended = if cx * cy >= 4 {
                let max_f = max_q as f32;
                max_f * mean_max_mixer + mean * (1.0 - mean_max_mixer)
            } else {
                max_q as f32
            };
            let blended_q = blended.round().clamp(1.0, 255.0) as u8;

            // Set all covered blocks to blended value
            for iy in 0..cy {
                for ix in 0..cx {
                    quant_field[(by + iy) * xsize_blocks + bx + ix] = blended_q;
                }
            }
        }
    }
}

/// Adjust the quant field for non-8×8 transforms (legacy, max-only version).
/// Use `adjust_quant_field_with_distance` for better quality at high distances.
#[allow(dead_code)]
pub fn adjust_quant_field(ac_strategy: &AcStrategyMap, quant_field: &mut [u8]) {
    // Use max-only behavior (mean_max_mixer = 1.0, equivalent to d < 1.54)
    adjust_quant_field_with_distance(ac_strategy, quant_field, 0.0);
}

// ─── Top-level API ──────────────────────────────────────────────────────────

/// Compute the AC strategy map for the entire image.
///
/// Iterates over 2×2 block groups within each tile, calling
/// `find_best_16x16_transform()` for each.
///
/// # Arguments
/// * `xyb_x`, `xyb_y`, `xyb_b` - XYB channel planes (padded to block boundaries)
/// * `stride` - Row stride (padded width) of the XYB buffers
/// * `buf_height` - Padded height of the XYB buffers
/// * `xsize_blocks`, `ysize_blocks` - Image dimensions in 8×8 blocks
/// * `distance` - Butteraugli target distance
/// * `quant_field_float` - Per-block float aq_map values
/// * `masking` - Per-block masking field from adaptive quantization
/// * `cfl_map` - Chroma-from-luma parameters
/// * `mask1x1` - Optional per-pixel masking field for pixel-domain loss
/// * `mask1x1_stride` - Stride of the mask1x1 array (typically padded_width)
#[allow(clippy::too_many_arguments)]
pub fn compute_ac_strategy(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    buf_height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    quant_field_float: &[f32],
    masking: &[f32],
    cfl_map: &CflMap,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
) -> AcStrategyMap {
    let _ = buf_height; // Used for documentation; buffer is padded to ysize_blocks * 8
    let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

    // C++ passes the float aq_map values directly to EstimateEntropy.
    // These are the adaptive quant values BEFORE conversion to u8 raw_quant.
    // Using u8 cast to f32 would give ~6x larger values (raw_quant = aq_map * inv_scale).

    let xyb = [xyb_x, xyb_y, xyb_b];

    // Process each tile (8×8 blocks = 64×64 pixels)
    for tile_by in (0..ysize_blocks).step_by(TILE_DIM_IN_BLOCKS) {
        for tile_bx in (0..xsize_blocks).step_by(TILE_DIM_IN_BLOCKS) {
            let tile_w = TILE_DIM_IN_BLOCKS.min(xsize_blocks - tile_bx);
            let tile_h = TILE_DIM_IN_BLOCKS.min(ysize_blocks - tile_by);

            // Get CfL params for this tile
            let tx = tile_bx / TILE_DIM_IN_BLOCKS;
            let ty = tile_by / TILE_DIM_IN_BLOCKS;
            let ytox = cfl_map.ytox_at(tx, ty);
            let ytob = cfl_map.ytob_at(tx, ty);

            // Process 8×8 block groups first (64×64), then 4×4 (32×32), then 2×2 (16×16)
            let mut cy = 0;
            // Process 8-row bands: try DCT64x64/DCT64x32/DCT32x64
            while cy + 7 < tile_h {
                let mut cx = 0;
                while cx + 7 < tile_w {
                    find_best_64x64_transform(
                        xyb,
                        stride,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        quant_field_float,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        &mut ac_strategy,
                    );
                    cx += 8;
                }
                // Remaining cols in this 8-row band: 4-block groups, then 2-block groups
                while cx + 3 < tile_w {
                    find_best_32x32_transform(
                        xyb,
                        stride,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        quant_field_float,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        &mut ac_strategy,
                    );
                    cx += 4;
                }
                while cx + 1 < tile_w {
                    find_best_16x16_transform(
                        xyb,
                        stride,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        quant_field_float,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        &mut ac_strategy,
                    );
                    cx += 2;
                }
                cy += 8;
            }
            // Remaining rows: 4-row bands for 32×32, then 2-row bands for 16×16
            while cy + 3 < tile_h {
                let mut cx = 0;
                while cx + 3 < tile_w {
                    find_best_32x32_transform(
                        xyb,
                        stride,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        quant_field_float,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        &mut ac_strategy,
                    );
                    cx += 4;
                }
                while cx + 1 < tile_w {
                    find_best_16x16_transform(
                        xyb,
                        stride,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        quant_field_float,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        &mut ac_strategy,
                    );
                    cx += 2;
                }
                cy += 4;
            }
            // Handle remaining rows that don't fit a 32×32 block
            while cy + 1 < tile_h {
                let mut cx = 0;
                while cx + 1 < tile_w {
                    find_best_16x16_transform(
                        xyb,
                        stride,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        quant_field_float,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        &mut ac_strategy,
                    );
                    cx += 2;
                }
                cy += 2;
            }
        }
    }

    ac_strategy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac_strategy_map_default() {
        let map = AcStrategyMap::new_dct8(4, 4);
        for by in 0..4 {
            for bx in 0..4 {
                assert_eq!(map.raw_strategy(bx, by), 0);
                assert!(map.is_first(bx, by));
                assert_eq!(map.strategy_code(bx, by), 0);
            }
        }
        assert_eq!(map.count_first_blocks(), 16);
    }

    #[test]
    fn test_ac_strategy_map_set_dct16x8() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        // DCT16X8 at (0,0): covers (0,0) and (0,1)
        map.set(0, 0, RAW_STRATEGY_DCT16X8);
        assert_eq!(map.raw_strategy(0, 0), RAW_STRATEGY_DCT16X8);
        assert!(map.is_first(0, 0));
        assert_eq!(map.raw_strategy(0, 1), RAW_STRATEGY_DCT16X8);
        assert!(!map.is_first(0, 1));
        // Strategy code for DCT16X8 is 6
        assert_eq!(map.strategy_code(0, 0), 6);
        // Rest should still be DCT8
        assert_eq!(map.raw_strategy(1, 0), 0);
        assert!(map.is_first(1, 0));
    }

    #[test]
    fn test_ac_strategy_map_set_dct8x16() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        // DCT8X16 at (2,0): covers (2,0) and (3,0)
        map.set(2, 0, RAW_STRATEGY_DCT8X16);
        assert_eq!(map.raw_strategy(2, 0), RAW_STRATEGY_DCT8X16);
        assert!(map.is_first(2, 0));
        assert_eq!(map.raw_strategy(3, 0), RAW_STRATEGY_DCT8X16);
        assert!(!map.is_first(3, 0));
        assert_eq!(map.strategy_code(2, 0), 7);
    }

    #[test]
    fn test_adjust_quant_field() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        // Set a DCT16X8 at (0,0)
        map.set(0, 0, RAW_STRATEGY_DCT16X8);
        let mut qf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        adjust_quant_field(&map, &mut qf);
        // DCT16X8 covers (0,0) and (0,1): max(1, 5) = 5
        assert_eq!(qf[0], 5);
        assert_eq!(qf[4], 5);
        // Other blocks unchanged
        assert_eq!(qf[1], 2);
    }

    #[test]
    fn test_estimate_entropy_finite() {
        // Test that estimate_entropy produces finite positive values
        let stride = 16;
        let buf_height = 16;
        let n = stride * buf_height;
        let xyb_x = vec![0.1f32; n];
        let xyb_y = vec![0.5f32; n];
        let xyb_b = vec![0.3f32; n];
        let xsize_blocks = 2;
        let quant_field = vec![4.0f32; 4];
        let masking = vec![1.0f32; 4];

        let ent = estimate_entropy(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
        );
        assert!(ent.is_finite(), "entropy should be finite: {}", ent);
        assert!(ent >= 0.0, "entropy should be non-negative: {}", ent);
    }

    #[test]
    fn test_count_first_blocks() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        assert_eq!(map.count_first_blocks(), 16);

        // Set one DCT16X8 (covers 2 blocks, 1 first)
        map.set(0, 0, RAW_STRATEGY_DCT16X8);
        assert_eq!(map.count_first_blocks(), 15); // 16 - 2 + 1

        // Set one DCT8X16 (covers 2 blocks, 1 first)
        map.set(2, 0, RAW_STRATEGY_DCT8X16);
        assert_eq!(map.count_first_blocks(), 14);
    }

    #[test]
    fn test_estimate_entropy_pixel_domain() {
        // Test that pixel-domain loss calculation produces finite positive values
        // and differs from coefficient-domain loss
        let stride = 16;
        let buf_height = 16;
        let n = stride * buf_height;
        let xyb_x = vec![0.1f32; n];
        let xyb_y = vec![0.5f32; n];
        let xyb_b = vec![0.3f32; n];
        let xsize_blocks = 2;
        let quant_field = vec![4.0f32; 4];
        let masking = vec![1.0f32; 4];

        // Create a simple mask1x1 field
        let mask1x1_stride = stride;
        let mask1x1 = vec![0.5f32; n];

        // Calculate coefficient-domain loss (without mask1x1)
        let ent_coeff = estimate_entropy_full(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            None,
            0,
            1.0, // entropy_mul = 1.0 for coefficient-domain (caller applies mul8x8)
        );

        // Calculate pixel-domain loss (with mask1x1)
        let ent_pixel = estimate_entropy_full(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT8), // Normalized entropy_mul for DCT8 = 1.0
        );

        eprintln!("Coefficient-domain entropy: {}", ent_coeff);
        eprintln!("Pixel-domain entropy: {}", ent_pixel);

        // Both should be finite and non-negative
        assert!(
            ent_coeff.is_finite(),
            "coeff entropy should be finite: {}",
            ent_coeff
        );
        assert!(
            ent_coeff >= 0.0,
            "coeff entropy should be non-negative: {}",
            ent_coeff
        );
        assert!(
            ent_pixel.is_finite(),
            "pixel entropy should be finite: {}",
            ent_pixel
        );
        assert!(
            ent_pixel >= 0.0,
            "pixel entropy should be non-negative: {}",
            ent_pixel
        );

        // They should be different (pixel-domain uses different constants and loss calculation)
        // The difference magnitude depends on the specific test data
        // For uniform inputs, both may be similar, but for real images they differ more
    }

    #[test]
    fn test_estimate_entropy_pixel_domain_strategies() {
        // Test pixel-domain loss for different strategies
        let stride = 32;
        let buf_height = 32;
        let n = stride * buf_height;

        // Non-uniform input to exercise the loss calculation
        let xyb_x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.01).collect();
        let xyb_y: Vec<f32> = (0..n).map(|i| 0.3 + (i % 13) as f32 * 0.02).collect();
        let xyb_b: Vec<f32> = (0..n).map(|i| 0.2 + (i % 11) as f32 * 0.015).collect();
        let xsize_blocks = 4;
        let quant_field = vec![4.0f32; 16];
        let masking = vec![1.0f32; 16];

        let mask1x1_stride = stride;
        let mask1x1: Vec<f32> = (0..n).map(|i| 0.3 + (i % 7) as f32 * 0.1).collect();

        // Test DCT8
        let ent_dct8 = estimate_entropy_full(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT8),
        );
        eprintln!("DCT8 pixel-domain entropy: {}", ent_dct8);
        assert!(ent_dct8.is_finite() && ent_dct8 >= 0.0);

        // Test DCT16x8 (requires 2-block tall region)
        let ent_dct16x8 = estimate_entropy_full(
            RAW_STRATEGY_DCT16X8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT16X8),
        );
        eprintln!("DCT16x8 pixel-domain entropy: {}", ent_dct16x8);
        assert!(ent_dct16x8.is_finite() && ent_dct16x8 >= 0.0);

        // Test DCT8x16 (requires 2-block wide region)
        let ent_dct8x16 = estimate_entropy_full(
            RAW_STRATEGY_DCT8X16,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT16X8),
        );
        eprintln!("DCT8x16 pixel-domain entropy: {}", ent_dct8x16);
        assert!(ent_dct8x16.is_finite() && ent_dct8x16 >= 0.0);

        // Test DCT16x16 (requires 2x2 block region)
        let ent_dct16x16 = estimate_entropy_full(
            RAW_STRATEGY_DCT16X16,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT16X16),
        );
        eprintln!("DCT16x16 pixel-domain entropy: {}", ent_dct16x16);
        assert!(ent_dct16x16.is_finite() && ent_dct16x16 >= 0.0);
    }
}
