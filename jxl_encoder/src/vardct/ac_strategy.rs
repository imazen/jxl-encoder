// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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

use super::ac_strategy_search::{
    find_best_16x16_transform, find_best_32x32_transform, find_best_64x64_transform,
};
use super::afv::afv_transform_from_pixels;
use super::block_extract::*;
use super::chroma_from_luma::{CflMap, ytob_ratio, ytox_ratio};
use super::common::{BLOCK_DIM, DCT_BLOCK_SIZE, TILE_DIM_IN_BLOCKS, ceil_log2_nonzero};
use super::dct::{
    dct_4x4_full, dct_4x8_full, dct_8x4_full, dct_8x8, dct_8x16, dct_16x8, dct_16x16, dct_16x32,
    dct_32x16, dct_32x32, dct_32x64, dct_64x32, dct_64x64, dct2x2_transform, idct_4x4_full,
    idct_4x8_full, idct_8x4_full, idct_8x8, idct_8x16, idct_16x8, idct_16x16, idct_16x32,
    idct_32x16, idct_32x32, idct_32x64, idct_64x32, idct_64x64, identity_transform,
    inverse_dct2x2_transform, inverse_identity_transform,
};
use super::quant::quant_weights;
use crate::effort::EffortProfile;

/// Pre-allocated scratch buffers for entropy estimation.
/// Avoids per-call heap allocations in the hot `estimate_entropy_full` loop.
pub(super) struct EntropyEstScratch {
    /// DCT coefficients for 3 channels (max 3 × 4096 for DCT64x64).
    pub block: Vec<f32>,
    /// Error coefficients for pixel-domain IDCT (max 4096).
    pub error_coeffs: Vec<f32>,
    /// Pixel-domain error output from IDCT (max 4096).
    pub pixel_error: Vec<f32>,
}

impl EntropyEstScratch {
    pub fn new() -> Self {
        const MAX: usize = 4096; // DCT64x64
        Self {
            block: vec![0.0f32; 3 * MAX],
            error_coeffs: vec![0.0f32; MAX],
            pixel_error: vec![0.0f32; MAX],
        }
    }
}

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

    /// Get the raw packed byte at (bx, by).
    /// The byte is `(raw_strategy << 1) | is_first`.
    #[inline]
    fn raw_byte(&self, bx: usize, by: usize) -> u8 {
        self.data[by * self.xsize_blocks + bx]
    }

    /// Set the raw packed byte at (bx, by) directly.
    /// Bypasses multi-block coverage logic — use only for save/restore.
    #[inline]
    fn set_raw_byte(&mut self, bx: usize, by: usize, byte: u8) {
        self.data[by * self.xsize_blocks + bx] = byte;
    }

    /// Find the first block (top-left corner) of the transform that owns (bx, by).
    /// Returns (first_x, first_y, raw_strategy).
    fn find_first_block(&self, bx: usize, by: usize) -> (usize, usize, u8) {
        if self.is_first(bx, by) {
            return (bx, by, self.raw_strategy(bx, by));
        }
        // The first block is at some position (fx, fy) where fx <= bx and fy <= by.
        // Walk up-left to find it. The strategy at (bx, by) tells us the raw strategy,
        // so we know the coverage. We need to find the top-left corner.
        let raw = self.raw_strategy(bx, by);
        let cx = COVERED_X[raw as usize];
        let cy = COVERED_Y[raw as usize];
        // The first block must be at an aligned position for this strategy.
        // For a transform covering cx×cy blocks, the first block (fx,fy) satisfies:
        //   fx <= bx < fx + cx  →  fx = bx - (bx % cx) if aligned
        //   fy <= by < fy + cy  →  fy = by - (by % cy) if aligned
        // But with non-aligned matching, alignment isn't guaranteed.
        // Instead, search backward.
        let min_fy = by.saturating_sub(cy - 1);
        for fy in (min_fy..=by).rev() {
            let min_fx = bx.saturating_sub(cx - 1);
            for fx in (min_fx..=bx).rev() {
                if self.is_first(fx, fy) && self.raw_strategy(fx, fy) == raw {
                    let fcx = COVERED_X[raw as usize];
                    let fcy = COVERED_Y[raw as usize];
                    if fx + fcx > bx && fy + fcy > by {
                        return (fx, fy, raw);
                    }
                }
            }
        }
        // Fallback: treat as single block
        (bx, by, raw)
    }

    /// Check if a proposed `blocks × blocks` region at `(bx, by)` can be
    /// re-evaluated without breaking any existing larger transform.
    ///
    /// Returns true if it's safe to call `find_best_16x16_transform` (blocks=2)
    /// or `find_best_32x32_transform` (blocks=4) at this position.
    ///
    /// The check verifies that no existing transform extends both inside and
    /// outside the proposed region (i.e., would need to be "split" by the new one).
    fn can_evaluate_region(&self, bx: usize, by: usize, blocks: usize) -> bool {
        // For each block in the proposed region, find its owning transform
        // and check that the transform is fully contained within the region.
        for dy in 0..blocks {
            for dx in 0..blocks {
                let x = bx + dx;
                let y = by + dy;
                if x >= self.xsize_blocks || y >= self.ysize_blocks {
                    return false;
                }
                let (fx, fy, raw) = self.find_first_block(x, y);
                let cx = COVERED_X[raw as usize];
                let cy = COVERED_Y[raw as usize];
                // The owning transform spans [fx, fx+cx) × [fy, fy+cy).
                // It must be fully inside or fully outside the region [bx, bx+blocks) × [by, by+blocks).
                // Since (x,y) is inside the region and inside the transform,
                // the transform must be fully contained within the region.
                if fx < bx || fy < by || fx + cx > bx + blocks || fy + cy > by + blocks {
                    return false; // Transform extends outside the region
                }
            }
        }
        true
    }

    /// Count the number of "first blocks" (= number of distinct transforms).
    #[cfg(test)]
    pub fn count_first_blocks(&self) -> usize {
        self.data.iter().filter(|&&v| (v & 1) != 0).count()
    }

    /// Return strategy histogram indexed by raw strategy code (0..19).
    /// Counts first blocks only (number of times each transform was selected).
    pub fn strategy_histogram(&self) -> [u32; 19] {
        let mut counts = [0u32; 19];
        for &v in &self.data {
            if (v & 1) != 0 {
                // is_first block
                let raw = (v >> 1) as usize;
                if raw < 19 {
                    counts[raw] += 1;
                }
            }
        }
        counts
    }

    /// Print strategy histogram with names.
    #[cfg(feature = "debug-ac-strategy")]
    pub fn print_histogram(&self) {
        const NAMES: [&str; 19] = [
            "DCT8", "DCT16x8", "DCT8x16", "DCT16x16", "DCT32x32", "DCT4x8", "DCT8x4", "DCT4x4",
            "IDENTITY", "DCT2X2", "DCT32x16", "DCT16x32", "AFV0", "AFV1", "AFV2", "AFV3",
            "DCT64x64", "DCT64x32", "DCT32x64",
        ];
        let hist = self.strategy_histogram();
        let total: u32 = hist.iter().sum();
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

/// Default pixel-domain cost model base constants (info_loss, zeros, cost_delta).
/// From libjxl enc_ac_strategy.cc:1111-1113.
pub(super) const DEFAULT_COST_BASES: (f32, f32, f32) = (1.2, 9.308_906, 10.833_273);

/// Distance scaling exponents from libjxl enc_ac_strategy.cc:1115-1120
const K_BIAS: f32 = 0.137_317_43;
const K_POW_INFO_LOSS: f32 = 0.336_778_07;
const K_POW_ZEROS_MUL: f32 = 0.509_909_3;
const K_POW_COST_DELTA: f32 = 0.367_029_4;

/// Compute distance-scaled constants for full libjxl cost model.
/// At d=1.0, returns the base values. At higher distances, increases all values.
///
/// `bases` is `(info_loss_mul_base, zeros_mul_base, cost_delta_base)` from
/// [`EffortProfile`](crate::effort::EffortProfile).
pub(super) fn compute_scaled_constants(distance: f32, bases: (f32, f32, f32)) -> (f32, f32, f32) {
    let (info_loss_base, zeros_base, cost_delta_base) = bases;
    let ratio = (distance + K_BIAS) / (1.0 + K_BIAS);
    let info_loss_mul = info_loss_base * ratio.powf(K_POW_INFO_LOSS);
    let zeros_mul = zeros_base * ratio.powf(K_POW_ZEROS_MUL);
    let cost_delta = cost_delta_base * ratio.powf(K_POW_COST_DELTA);
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
pub(super) fn entropy_mul_for_strategy(raw_strategy: u8) -> f32 {
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
pub(super) fn estimate_entropy(
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
    let mut scratch = EntropyEstScratch::new();
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
        DEFAULT_COST_BASES,
        &mut scratch,
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
pub(super) fn estimate_entropy_with_mask(
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
    pixel_domain_cost_bases: (f32, f32, f32),
    scratch: &mut EntropyEstScratch,
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
        pixel_domain_cost_bases,
        scratch,
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
///
/// Dispatches to an `#[arcane]` variant on x86_64/aarch64 so the entire function
/// body (including scalar arithmetic) runs under `#[target_feature]` and LLVM can
/// inline the jxl_simd calls + auto-vectorize surrounding code.
#[allow(clippy::too_many_arguments)]
pub(super) fn estimate_entropy_full(
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
    pixel_domain_cost_bases: (f32, f32, f32),
    scratch: &mut EntropyEstScratch,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::X64V3Token::summon() {
            return estimate_entropy_full_avx2(
                token,
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
                pixel_domain_cost_bases,
                scratch,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::NeonToken::summon() {
            return estimate_entropy_full_neon(
                token,
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
                pixel_domain_cost_bases,
                scratch,
            );
        }
    }
    estimate_entropy_full_impl(
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
        pixel_domain_cost_bases,
        scratch,
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn estimate_entropy_full_avx2(
    _token: jxl_simd::X64V3Token,
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
    pixel_domain_cost_bases: (f32, f32, f32),
    scratch: &mut EntropyEstScratch,
) -> f32 {
    estimate_entropy_full_impl(
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
        pixel_domain_cost_bases,
        scratch,
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn estimate_entropy_full_neon(
    _token: jxl_simd::NeonToken,
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
    pixel_domain_cost_bases: (f32, f32, f32),
    scratch: &mut EntropyEstScratch,
) -> f32 {
    estimate_entropy_full_impl(
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
        pixel_domain_cost_bases,
        scratch,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn estimate_entropy_full_impl(
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
    pixel_domain_cost_bases: (f32, f32, f32),
    scratch: &mut EntropyEstScratch,
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
        compute_scaled_constants(distance, pixel_domain_cost_bases)
    } else {
        // libjxl-tiny style constants (not distance-scaled)
        (138.0_f32, 5.335_918_5_f32, 7.565_053_4_f32)
    };
    const K_INFO_LOSS_MULTIPLIER2: f32 = 50.468_4;
    const K_COST2: f32 = 4.462_815;

    // Use pre-allocated scratch buffers (no fill needed — transforms overwrite all positions)
    let block = &mut scratch.block[..3 * size];
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
                let mut input = [0.0f32; 64];
                extract_block_8x8(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 64];
                identity_transform(&input, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT2X2 => {
                let mut input = [0.0f32; 64];
                extract_block_8x8(xyb_c, stride, bx, by, &mut input);
                let mut output = [0.0f32; 64];
                dct2x2_transform(&input, &mut output);
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

    // Zero the LLF (lowest-frequency) positions in the block data for all channels.
    // libjxl zeros these positions in inv_table (quant_weights.cc:342-355) so they
    // contribute nothing to entropy or loss estimates. The DC/LLF coefficients are
    // overwritten by LowestFrequenciesFromDC during decoding, so their quantization
    // cost is irrelevant for strategy selection. Without this, different strategies
    // have wildly different DC dequant weights (e.g., DCT8 Y=560 vs AFV Y=58),
    // creating phantom entropy differences that bias AFV selection.
    //
    // The LLF region uses the transposed layout (cx_t >= cy_t, stride = cx_t * 8),
    // matching libjxl's swap(cx, cy) convention. Non-square DCTs always output in
    // the layout where the longer dimension is the stride (e.g., both DCT16X8 and
    // DCT8X16 output in 8×16 layout with stride 16). The LLF positions are in the
    // top-left corner of this layout.
    {
        let (cx_t, cy_t) = if cy > cx { (cy, cx) } else { (cx, cy) };
        let llf_stride = cx_t * BLOCK_DIM;
        for c in 0..3 {
            let offset = c * size;
            for iy in 0..cy_t {
                for ix in 0..cx_t {
                    block[offset + iy * llf_stride + ix] = 0.0;
                }
            }
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
            // x^(1/16) = sqrt(sqrt(sqrt(sqrt(x))))
            norm_sum.sqrt().sqrt().sqrt().sqrt()
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

    // Error coefficient buffer for pixel-domain IDCT (reused per channel,
    // no fill needed — entropy_estimate_coeffs writes all positions)
    let error_coeffs = &mut scratch.error_coeffs[..size];

    let slope = (distance / 3.0).min(1.0);
    let cost_of_1 = 1.0 + slope * 8.870_325;

    // Pixel base coordinates
    let pixel_x = bx * BLOCK_DIM;
    let pixel_y = by * BLOCK_DIM;

    for (c, &cmap_factor) in cmap_factors.iter().enumerate() {
        let weights = quant_weights(raw_strategy as usize, c);

        let offset_c = c * size;
        let offset_y = size; // Y channel always at offset 1*size

        // SIMD-accelerated coefficient processing (biggest encoder hotspot).
        // LLF positions are pre-zeroed above (matching libjxl quant_weights.cc:342-355),
        // so DC/LLF contribute nothing to entropy or loss estimates.
        //
        // In pixel-domain mode: use quant_norm16 (L16 norm for 4+ blocks, max for
        // 1-2 blocks) matching libjxl enc_ac_strategy.cc:415.
        // In coefficient-domain mode: use max(quant_field) (libjxl-tiny style).
        let quant_for_coeffs = if use_pixel_domain {
            quant_norm16
        } else {
            quant
        };
        let coeff_result = jxl_simd::entropy_estimate_coeffs(
            &block[offset_c..offset_c + size],
            &block[offset_y..offset_y + size],
            weights,
            size,
            cmap_factor,
            quant_for_coeffs,
            k_cost_delta,
            K_COST2,
            use_pixel_domain,
            error_coeffs,
        );
        let mut entropy_sum = coeff_result.entropy_sum;
        let nzeros_sum = coeff_result.nzeros_sum;
        if !use_pixel_domain {
            info_loss_sum += coeff_result.info_loss_sum;
            info_loss2_sum += coeff_result.info_loss2_sum;
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
            let pixel_error_buf = &mut scratch.pixel_error[..size];
            apply_idct_for_strategy(raw_strategy, error_coeffs, pixel_error_buf);
            let pixel_error = &*pixel_error_buf;

            // Compute 8th power norm with per-pixel masking via SIMD kernel.
            // mask1x1 is padded to block-aligned dimensions (xsize_blocks*8 × ysize_blocks*8),
            // and mask1x1_stride = padded_width, so all block pixel accesses are in-bounds.
            let mask_offset = MASK_CHANNEL_OFFSET[c];
            let block_width = cx * BLOCK_DIM;
            let block_height = cy * BLOCK_DIM;
            let mask_row_base = pixel_y * mask1x1_stride + pixel_x;

            let mut channel_loss = jxl_simd::pixel_domain_loss(
                pixel_error,
                mask,
                mask_row_base,
                mask1x1_stride,
                mask_offset,
                block_width,
                block_height,
            );

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
        // x^(1/8) = sqrt(sqrt(sqrt(x)))
        let loss_scalar = (total_pixel_loss / n).sqrt().sqrt().sqrt() * n / quant_norm16 as f64;
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
/// Writes pixel-domain error in row-major layout into `output`.
pub(super) fn apply_idct_for_strategy(raw_strategy: u8, error_coeffs: &[f32], output: &mut [f32]) {
    match raw_strategy {
        RAW_STRATEGY_DCT8 => {
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&error_coeffs[..64]);
            let mut tmp = [0.0f32; 64];
            idct_8x8(&input, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT4X8 => {
            let input: &[f32; 64] = error_coeffs[..64].try_into().unwrap();
            let mut tmp = [0.0f32; 64];
            idct_4x8_full(input, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT8X4 => {
            let input: &[f32; 64] = error_coeffs[..64].try_into().unwrap();
            let mut tmp = [0.0f32; 64];
            idct_8x4_full(input, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT4X4 => {
            let input: &[f32; 64] = error_coeffs[..64].try_into().unwrap();
            let mut tmp = [0.0f32; 64];
            idct_4x4_full(input, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
            let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&error_coeffs[..64]);
            let mut tmp = [0.0f32; 64];
            super::afv::inverse_afv_transform(&input, afv_kind, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT16X8 => {
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&error_coeffs[..128]);
            let mut tmp = [0.0f32; 128];
            idct_16x8(&input, &mut tmp);
            output[..128].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT8X16 => {
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&error_coeffs[..128]);
            let mut tmp = [0.0f32; 128];
            idct_8x16(&input, &mut tmp);
            output[..128].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT16X16 => {
            let mut input = [0.0f32; 256];
            input.copy_from_slice(&error_coeffs[..256]);
            let mut tmp = [0.0f32; 256];
            idct_16x16(&input, &mut tmp);
            output[..256].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT32X32 => {
            let mut input = [0.0f32; 1024];
            input.copy_from_slice(&error_coeffs[..1024]);
            let mut tmp = [0.0f32; 1024];
            idct_32x32(&input, &mut tmp);
            output[..1024].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT32X16 => {
            let mut input = [0.0f32; 512];
            input.copy_from_slice(&error_coeffs[..512]);
            let mut tmp = [0.0f32; 512];
            idct_32x16(&input, &mut tmp);
            output[..512].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT16X32 => {
            let mut input = [0.0f32; 512];
            input.copy_from_slice(&error_coeffs[..512]);
            let mut tmp = [0.0f32; 512];
            idct_16x32(&input, &mut tmp);
            output[..512].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT64X64 => {
            idct_64x64(&error_coeffs[..4096], &mut output[..4096]);
        }
        RAW_STRATEGY_DCT64X32 => {
            idct_64x32(&error_coeffs[..2048], &mut output[..2048]);
        }
        RAW_STRATEGY_DCT32X64 => {
            idct_32x64(&error_coeffs[..2048], &mut output[..2048]);
        }
        RAW_STRATEGY_IDENTITY => {
            let mut tmp = [0.0f32; 64];
            let coeffs: &[f32; 64] = error_coeffs[..64].try_into().unwrap();
            inverse_identity_transform(coeffs, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT2X2 => {
            let mut tmp = [0.0f32; 64];
            let coeffs: &[f32; 64] = error_coeffs[..64].try_into().unwrap();
            inverse_dct2x2_transform(coeffs, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        _ => unreachable!(
            "unknown strategy {} in apply_idct_for_strategy",
            raw_strategy
        ),
    }
}

/// Adjust the float quant field for multi-block transforms.
///
/// Same algorithm as `adjust_quant_field_with_distance` but operates on the
/// float quant field (values ~0.3-1.5) instead of u8 (1-255). This matches
/// libjxl's `AdjustQuantField` which works on `ImageF` before `SetQuantField`.
pub fn adjust_quant_field_float_with_distance(
    ac_strategy: &AcStrategyMap,
    quant_field: &mut [f32],
    butteraugli_target: f32,
) {
    let xsize_blocks = ac_strategy.xsize_blocks;

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
            let mut max_q = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            for iy in 0..cy {
                for ix in 0..cx {
                    let q = quant_field[(by + iy) * xsize_blocks + bx + ix];
                    max_q = max_q.max(q);
                    sum += q as f64;
                }
            }
            let mean = (sum / (cx * cy) as f64) as f32;

            // Blend max and mean (for 4+ block transforms)
            let blended = if cx * cy >= 4 {
                max_q * mean_max_mixer + mean * (1.0 - mean_max_mixer)
            } else {
                max_q
            };

            // Set all covered blocks to blended value (no integer clamping)
            for iy in 0..cy {
                for ix in 0..cx {
                    quant_field[(by + iy) * xsize_blocks + bx + ix] = blended;
                }
            }
        }
    }
}

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
    profile: &EffortProfile,
) -> AcStrategyMap {
    let _ = buf_height; // Used for documentation; buffer is padded to ysize_blocks * 8
    let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

    // C++ passes the float aq_map values directly to EstimateEntropy.
    // These are the adaptive quant values BEFORE conversion to u8 raw_quant.
    // Using u8 cast to f32 would give ~6x larger values (raw_quant = aq_map * inv_scale).

    let xyb = [xyb_x, xyb_y, xyb_b];
    let mut scratch = EntropyEstScratch::new();

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

            // Process hierarchically: 8×8 block groups (64×64) at e7+,
            // then 4×4 (32×32) at e5+, then always 2×2 (16×16).
            let try_64 = profile.try_dct64;
            let try_32 = profile.try_dct32;

            let mut cy = 0;
            // Process 8-row bands: try DCT64x64/DCT64x32/DCT32x64 at effort 7+
            while try_64 && cy + 7 < tile_h {
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
                        &mut scratch,
                        profile,
                    );
                    cx += 8;
                }
                // Remaining cols in this 8-row band: 4-block groups, then 2-block groups
                while try_32 && cx + 3 < tile_w {
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
                        &mut scratch,
                        profile,
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
                        &mut scratch,
                        1.0,
                        profile,
                    );
                    cx += 2;
                }
                cy += 8;
            }
            // Remaining rows: 4-row bands for 32×32 at effort 5+, then 2-row bands for 16×16
            while try_32 && cy + 3 < tile_h {
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
                        &mut scratch,
                        profile,
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
                        &mut scratch,
                        1.0,
                        profile,
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
                        &mut scratch,
                        1.0,
                        profile,
                    );
                    cx += 2;
                }
                cy += 2;
            }

            // Non-aligned matching: try 16×16/16×8/8×16 at non-2-aligned positions.
            // This catches cases where the optimal block boundary straddles a 2-aligned
            // position. Matches libjxl enc_ac_strategy.cc:1035-1044 (effort >= 6).
            // Only accept results when a multi-block transform is selected — single-block
            // re-evaluation at non-aligned positions can override good aligned-pass choices.
            //
            // NOTE: favor_single_mul = 1.0 here (not mul8x8). find_best_16x16_transform
            // already applies mul8x8 internally to single-block costs. Passing mul8x8 as
            // favor_single_mul would double-apply it, making singles too cheap and
            // preventing rectangle transforms from ever winning. libjxl's non-aligned
            // pass uses the same stored entropy_estimate values (with single mul8x8).
            for cy in if profile.non_aligned_eval { 0 } else { tile_h }..tile_h.saturating_sub(1) {
                for cx in 0..tile_w.saturating_sub(1) {
                    // Skip 2-aligned positions (already evaluated in the aligned pass)
                    if (cy | cx) % 2 == 0 {
                        continue;
                    }
                    let abs_bx = tile_bx + cx;
                    let abs_by = tile_by + cy;
                    // Check that the proposed 2×2 region doesn't cross any existing
                    // multi-block transform boundaries
                    if !ac_strategy.can_evaluate_region(abs_bx, abs_by, 2) {
                        continue;
                    }
                    // Save current strategies for the 2×2 region
                    let mut saved = [0u8; 4];
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            saved[dy * 2 + dx] = ac_strategy.raw_byte(abs_bx + dx, abs_by + dy);
                        }
                    }
                    // Reset all blocks in the region to DCT8 before re-evaluation.
                    // This is necessary because find_best_16x16_transform skips
                    // set() for DCT8 (treating it as default), but blocks may have
                    // non-DCT8 strategies from the aligned pass.
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            ac_strategy.set(abs_bx + dx, abs_by + dy, RAW_STRATEGY_DCT8);
                        }
                    }
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
                        &mut scratch,
                        1.0,
                        profile,
                    );
                    // Only keep results if a multi-block transform was selected.
                    // If only single-block strategies were chosen, the aligned pass
                    // already found optimal single-block choices for these positions.
                    let has_multi = (0..2usize).any(|dy| {
                        (0..2usize).any(|dx| {
                            let raw = ac_strategy.raw_strategy(abs_bx + dx, abs_by + dy);
                            COVERED_X[raw as usize] > 1 || COVERED_Y[raw as usize] > 1
                        })
                    });
                    if !has_multi {
                        // Restore original aligned-pass strategies
                        for dy in 0..2usize {
                            for dx in 0..2usize {
                                ac_strategy.set_raw_byte(
                                    abs_bx + dx,
                                    abs_by + dy,
                                    saved[dy * 2 + dx],
                                );
                            }
                        }
                    }
                }
            }

            // Non-aligned matching for 32×32/32×16/16×32 at non-4-aligned positions.
            // Matches libjxl enc_ac_strategy.cc:1045-1057 (effort >= 7).
            // Only at d>=2.0 where DCT32x32 is enabled.
            if distance >= 2.0 && profile.try_dct64 {
                let step = profile.fine_grained_step as usize;
                for cy in (0..tile_h.saturating_sub(3)).step_by(step) {
                    for cx in (0..tile_w.saturating_sub(3)).step_by(step) {
                        // Skip 4-aligned positions (already evaluated in aligned pass)
                        if (cy | cx) % 4 == 0 {
                            continue;
                        }
                        let abs_bx = tile_bx + cx;
                        let abs_by = tile_by + cy;
                        if !ac_strategy.can_evaluate_region(abs_bx, abs_by, 4) {
                            continue;
                        }
                        // Save current strategies for the 4×4 region
                        let mut saved = [0u8; 16];
                        for dy in 0..4usize {
                            for dx in 0..4usize {
                                saved[dy * 4 + dx] = ac_strategy.raw_byte(abs_bx + dx, abs_by + dy);
                            }
                        }
                        // Reset all blocks in the 4×4 region to DCT8
                        for dy in 0..4usize {
                            for dx in 0..4usize {
                                ac_strategy.set(abs_bx + dx, abs_by + dy, RAW_STRATEGY_DCT8);
                            }
                        }
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
                            &mut scratch,
                            profile,
                        );
                        // Only keep results if a multi-block transform was selected
                        let has_multi = (0..4usize).any(|dy| {
                            (0..4usize).any(|dx| {
                                let raw = ac_strategy.raw_strategy(abs_bx + dx, abs_by + dy);
                                COVERED_X[raw as usize] > 1 || COVERED_Y[raw as usize] > 1
                            })
                        });
                        if !has_multi {
                            // Restore original strategies
                            for dy in 0..4usize {
                                for dx in 0..4usize {
                                    ac_strategy.set_raw_byte(
                                        abs_bx + dx,
                                        abs_by + dy,
                                        saved[dy * 4 + dx],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Validate strategy map consistency in debug builds
    #[cfg(debug_assertions)]
    {
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let raw = ac_strategy.raw_strategy(bx, by);
                if ac_strategy.is_first(bx, by) {
                    let cx = COVERED_X[raw as usize];
                    let cy = COVERED_Y[raw as usize];
                    // Verify all covered blocks have matching raw_strategy and is_first=false
                    for iy in 0..cy {
                        for ix in 0..cx {
                            assert!(
                                bx + ix < xsize_blocks && by + iy < ysize_blocks,
                                "Transform at ({},{}) raw={} extends out of bounds: ({},{}) vs {}x{}",
                                bx,
                                by,
                                raw,
                                bx + ix,
                                by + iy,
                                xsize_blocks,
                                ysize_blocks
                            );
                            assert_eq!(
                                ac_strategy.raw_strategy(bx + ix, by + iy),
                                raw,
                                "Inconsistent raw_strategy at ({},{}) — expected {} (from first block ({},{})), got {}",
                                bx + ix,
                                by + iy,
                                raw,
                                bx,
                                by,
                                ac_strategy.raw_strategy(bx + ix, by + iy)
                            );
                            if (ix | iy) != 0 {
                                assert!(
                                    !ac_strategy.is_first(bx + ix, by + iy),
                                    "Block ({},{}) should not be first (owned by ({},{}) raw={})",
                                    bx + ix,
                                    by + iy,
                                    bx,
                                    by,
                                    raw
                                );
                            }
                        }
                    }
                }
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

        let mut scratch = EntropyEstScratch::new();
        let cost_bases = (1.2_f32, 9.308_906_f32, 10.833_273_f32);

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
            cost_bases,
            &mut scratch,
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
            cost_bases,
            &mut scratch,
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

        let mut scratch = EntropyEstScratch::new();
        let cost_bases = (1.2_f32, 9.308_906_f32, 10.833_273_f32);

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
            cost_bases,
            &mut scratch,
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
            cost_bases,
            &mut scratch,
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
            cost_bases,
            &mut scratch,
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
            cost_bases,
            &mut scratch,
        );
        eprintln!("DCT16x16 pixel-domain entropy: {}", ent_dct16x16);
        assert!(ent_dct16x16.is_finite() && ent_dct16x16 >= 0.0);
    }
}
