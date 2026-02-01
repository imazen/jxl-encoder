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

use super::chroma_from_luma::{CflMap, ytob_ratio, ytox_ratio};
use super::common::{BLOCK_DIM, DCT_BLOCK_SIZE, TILE_DIM_IN_BLOCKS, ceil_log2_nonzero};
use super::dct::{dct_8x8, dct_8x16, dct_16x8};
use super::quant::quant_weights;

/// Raw strategy codes matching the C++ `AcStrategy::Type` enum.
pub const RAW_STRATEGY_DCT8: u8 = 0;
pub const RAW_STRATEGY_DCT16X8: u8 = 1;
pub const RAW_STRATEGY_DCT8X16: u8 = 2;

/// Strategy code as written to the bitstream (via `StrategyCode()`).
/// These differ from raw strategy codes.
const STRATEGY_CODE_LUT: [u8; 3] = [0, 6, 7];

/// Covered blocks in X direction for each raw strategy.
const COVERED_X: [usize; 3] = [1, 1, 2];

/// Covered blocks in Y direction for each raw strategy.
const COVERED_Y: [usize; 3] = [1, 2, 1];

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
}

// ─── Entropy estimation ─────────────────────────────────────────────────────

/// Estimate the coded entropy of a block under a given transform strategy.
///
/// Port of C++ `EstimateEntropy`. Returns a cost that combines:
/// - Estimated bits for coding the quantized coefficients
/// - Information loss penalty weighted by masking
///
/// # Arguments
/// * `raw_strategy` - 0=DCT8, 1=DCT16X8, 2=DCT8X16
/// * `xyb` - The three XYB channel planes
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `bx`, `by` - Block coordinates of the top-left 8×8 block
/// * `distance` - Butteraugli target distance
/// * `quant_field` - Per-block quant values (flat, indexed by*xblocks+bx)
/// * `xsize_blocks` - Image width in blocks
/// * `masking` - Per-block masking field (flat, indexed by*xblocks+bx)
/// * `ytox`, `ytob` - CfL parameters for this tile
#[allow(clippy::too_many_arguments)]
fn estimate_entropy(
    raw_strategy: u8,
    xyb: [&[f32]; 3],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
) -> f32 {
    let cx = COVERED_X[raw_strategy as usize];
    let cy = COVERED_Y[raw_strategy as usize];
    let num_blocks = cx * cy;
    let size = num_blocks * DCT_BLOCK_SIZE;

    // Apply transform for each channel
    let mut block = [0.0f32; 3 * 2 * DCT_BLOCK_SIZE]; // max 3 channels × 128 coeffs
    for (c, xyb_c) in xyb.iter().enumerate() {
        let offset = c * size;
        match raw_strategy {
            RAW_STRATEGY_DCT8 => {
                let mut input = [0.0f32; 64];
                extract_block_8x8(xyb_c, width, height, bx, by, &mut input);
                let mut output = [0.0f32; 64];
                dct_8x8(&input, &mut output);
                block[offset..offset + 64].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT16X8 => {
                // 1 wide × 2 tall: extract 8×16 pixel region
                let mut input = [0.0f32; 128];
                extract_block_8x16(xyb_c, width, height, bx, by, &mut input);
                let mut output = [0.0f32; 128];
                dct_16x8(&input, &mut output);
                block[offset..offset + 128].copy_from_slice(&output);
            }
            RAW_STRATEGY_DCT8X16 => {
                // 2 wide × 1 tall: extract 16×8 pixel region
                let mut input = [0.0f32; 128];
                extract_block_16x8(xyb_c, width, height, bx, by, &mut input);
                let mut output = [0.0f32; 128];
                dct_8x16(&input, &mut output);
                block[offset..offset + 128].copy_from_slice(&output);
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

    // Entropy estimation constants (from C++)
    const K_INFO_LOSS_MULTIPLIER: f32 = 138.0;
    const K_INFO_LOSS_MULTIPLIER2: f32 = 50.468_4;
    const K_COST2: f32 = 4.462_815;
    const K_COST_DELTA: f32 = 5.335_918_5;
    const K_ZEROS_MUL: f32 = 7.565_053_4;

    let cmap_factors = [ytox_ratio(ytox), 0.0f32, ytob_ratio(ytob)];

    let mut entropy = 0.0f32;
    let mut info_loss_sum = 0.0f32;
    let mut info_loss2_sum = 0.0f32;

    let slope = (distance / 3.0).min(1.0);
    let cost_of_1 = 1.0 + slope * 8.870_325;

    for (c, &cmap_factor) in cmap_factors.iter().enumerate() {
        // For the strategy, use the raw_strategy for weight lookup:
        // C++ uses acs.RawStrategy() which is 0/1/2
        let inv_matrix = quant_weights(raw_strategy as usize, c);

        let offset_c = c * size;
        let offset_y = size; // Y channel always at offset 1*size

        let mut entropy_sum = 0.0f32;
        let mut nzeros_sum = 0.0f32;

        // Skip LLF coefficients (positions 0..num_blocks).
        // C++ zeroes these positions in InvMatrix so they contribute nothing.
        // LLF/DC coefficients are handled by the DC path, not AC entropy.
        for i in num_blocks..size {
            let val_in = block[offset_c + i];
            let val_y = block[offset_y + i] * cmap_factor;
            // inv_matrix stores weights; C++ InvMatrix = 1/weight
            let im = 1.0 / inv_matrix[i];
            let val = (val_in - val_y) * im * quant;
            let rval = val.round();
            let diff = (val - rval).abs();
            info_loss_sum += diff;
            info_loss2_sum += diff * diff;
            let q = rval.abs();
            if q >= 1.5 {
                entropy_sum += K_COST2;
            }
            entropy_sum += q.sqrt() * K_COST_DELTA;
            if q != 0.0 {
                nzeros_sum += 1.0;
            }
        }
        entropy_sum += nzeros_sum * cost_of_1;
        entropy += entropy_sum;

        let num_nzeros = nzeros_sum as usize;
        let nbits = ceil_log2_nonzero(num_nzeros + 1) as usize + 1;
        entropy += K_ZEROS_MUL * (ceil_log2_nonzero(nbits + 17) + nbits as u32) as f32;
    }

    let infoloss2 = (num_blocks as f32 * info_loss2_sum).sqrt();
    let info_loss_score =
        K_INFO_LOSS_MULTIPLIER * info_loss_sum + K_INFO_LOSS_MULTIPLIER2 * infoloss2;
    entropy + mask_val * info_loss_score
}

// ─── Block extraction helpers ────────────────────────────────────────────────

/// Extract an 8×8 pixel block from a plane with edge clamping.
fn extract_block_8x8(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 64],
) {
    for dy in 0..8 {
        for dx in 0..8 {
            let py = (by * BLOCK_DIM + dy).min(height - 1);
            let px = (bx * BLOCK_DIM + dx).min(width - 1);
            out[dy * 8 + dx] = plane[py * width + px];
        }
    }
}

/// Extract an 8×16 pixel block (1 wide × 2 tall) for DCT16x8.
/// Layout: 16 rows × 8 cols, row-major.
fn extract_block_8x16(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 128],
) {
    for dy in 0..16 {
        for dx in 0..8 {
            let py = (by * BLOCK_DIM + dy).min(height - 1);
            let px = (bx * BLOCK_DIM + dx).min(width - 1);
            out[dy * 8 + dx] = plane[py * width + px];
        }
    }
}

/// Extract a 16×8 pixel block (2 wide × 1 tall) for DCT8x16.
/// Layout: 8 rows × 16 cols, row-major.
fn extract_block_16x8(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 128],
) {
    for dy in 0..8 {
        for dx in 0..16 {
            let py = (by * BLOCK_DIM + dy).min(height - 1);
            let px = (bx * BLOCK_DIM + dx).min(width - 1);
            out[dy * 16 + dx] = plane[py * width + px];
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
#[allow(clippy::too_many_arguments)]
fn find_best_16x16_transform(
    xyb: [&[f32]; 3],
    width: usize,
    height: usize,
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
    ac_strategy: &mut AcStrategyMap,
) {
    // Distance-dependent multipliers (from C++)
    let k8x8mul1: f32 = -0.55 * 0.75;
    let k8x8mul2: f32 = 1.073_575_8 * 0.75;
    let k8x8base: f32 = 1.4;
    let mul8x8 = k8x8mul2 + k8x8mul1 / (distance + k8x8base);

    let k8x16mul1: f32 = -0.55;
    let k8x16mul2: f32 = 0.901_958_8;
    let k8x16base: f32 = 1.6;
    let mul16x8 = k8x16mul2 + k8x16mul1 / (distance + k8x16base);

    let abs_bx = bx0 + cx;
    let abs_by = by0 + cy;

    // Evaluate four 8×8 blocks
    let mut entropy = [[0.0f32; 2]; 2];
    for (dy, entropy_row) in entropy.iter_mut().enumerate() {
        for (dx, entropy_val) in entropy_row.iter_mut().enumerate() {
            let e = estimate_entropy(
                RAW_STRATEGY_DCT8,
                xyb,
                width,
                height,
                abs_bx + dx,
                abs_by + dy,
                distance,
                quant_field,
                xsize_blocks,
                masking,
                ytox,
                ytob,
            );
            *entropy_val = 3.0 * mul8x8 + mul8x8 * e;
        }
    }

    // Evaluate two DCT16X8 options (left column, right column)
    let entropy_16x8_left = mul16x8
        * estimate_entropy(
            RAW_STRATEGY_DCT16X8,
            xyb,
            width,
            height,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
        );
    let entropy_16x8_right = mul16x8
        * estimate_entropy(
            RAW_STRATEGY_DCT16X8,
            xyb,
            width,
            height,
            abs_bx + 1,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
        );

    // Evaluate two DCT8X16 options (top row, bottom row)
    let entropy_8x16_top = mul16x8
        * estimate_entropy(
            RAW_STRATEGY_DCT8X16,
            xyb,
            width,
            height,
            abs_bx,
            abs_by,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
        );
    let entropy_8x16_bottom = mul16x8
        * estimate_entropy(
            RAW_STRATEGY_DCT8X16,
            xyb,
            width,
            height,
            abs_bx,
            abs_by + 1,
            distance,
            quant_field,
            xsize_blocks,
            masking,
            ytox,
            ytob,
        );

    // Compare 16x8 split vs 8x16 split
    let cost16x8 = (entropy_16x8_left).min(entropy[0][0] + entropy[1][0])
        + (entropy_16x8_right).min(entropy[0][1] + entropy[1][1]);
    let cost8x16 = (entropy_8x16_top).min(entropy[0][0] + entropy[0][1])
        + (entropy_8x16_bottom).min(entropy[1][0] + entropy[1][1]);

    if cost16x8 < cost8x16 {
        // Try 16x8 for each column
        if entropy_16x8_left < entropy[0][0] + entropy[1][0] {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT16X8);
        }
        if entropy_16x8_right < entropy[0][1] + entropy[1][1] {
            ac_strategy.set(abs_bx + 1, abs_by, RAW_STRATEGY_DCT16X8);
        }
    } else {
        // Try 8x16 for each row
        if entropy_8x16_top < entropy[0][0] + entropy[0][1] {
            ac_strategy.set(abs_bx, abs_by, RAW_STRATEGY_DCT8X16);
        }
        if entropy_8x16_bottom < entropy[1][0] + entropy[1][1] {
            ac_strategy.set(abs_bx, abs_by + 1, RAW_STRATEGY_DCT8X16);
        }
    }
}

// ─── Quant field adjustment ─────────────────────────────────────────────────

/// Adjust the quant field for non-8×8 transforms.
///
/// For multi-block transforms, all covered blocks get the maximum quant value.
/// Port of C++ `AdjustQuantField`.
pub fn adjust_quant_field(ac_strategy: &AcStrategyMap, quant_field: &mut [u8]) {
    let xsize_blocks = ac_strategy.xsize_blocks;
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
            // Find max quant in covered region
            let mut max_q = 0u8;
            for iy in 0..cy {
                for ix in 0..cx {
                    max_q = max_q.max(quant_field[(by + iy) * xsize_blocks + bx + ix]);
                }
            }
            // Set all covered blocks to max
            for iy in 0..cy {
                for ix in 0..cx {
                    quant_field[(by + iy) * xsize_blocks + bx + ix] = max_q;
                }
            }
        }
    }
}

// ─── Top-level API ──────────────────────────────────────────────────────────

/// Compute the AC strategy map for the entire image.
///
/// Iterates over 2×2 block groups within each tile, calling
/// `find_best_16x16_transform()` for each.
///
/// # Arguments
/// * `xyb_x`, `xyb_y`, `xyb_b` - XYB channel planes
/// * `width`, `height` - Image dimensions in pixels
/// * `xsize_blocks`, `ysize_blocks` - Image dimensions in 8×8 blocks
/// * `distance` - Butteraugli target distance
/// * `quant_field_u8` - Per-block raw_quant in [1, 255]
/// * `masking` - Per-block masking field from adaptive quantization
/// * `cfl_map` - Chroma-from-luma parameters
#[allow(clippy::too_many_arguments)]
pub fn compute_ac_strategy(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    quant_field_u8: &[u8],
    masking: &[f32],
    cfl_map: &CflMap,
) -> AcStrategyMap {
    let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

    // Convert u8 quant field to f32 for entropy estimation
    // (C++ uses ImageF quant_field directly)
    let quant_field_f32: Vec<f32> = quant_field_u8.iter().map(|&q| q as f32).collect();

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

            // Process 2×2 block groups within this tile
            let mut cy = 0;
            while cy + 1 < tile_h {
                let mut cx = 0;
                while cx + 1 < tile_w {
                    find_best_16x16_transform(
                        xyb,
                        width,
                        height,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        &quant_field_f32,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
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
        let width = 16;
        let height = 16;
        let n = width * height;
        let xyb_x = vec![0.1f32; n];
        let xyb_y = vec![0.5f32; n];
        let xyb_b = vec![0.3f32; n];
        let xsize_blocks = 2;
        let quant_field = vec![4.0f32; 4];
        let masking = vec![1.0f32; 4];

        let ent = estimate_entropy(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            width,
            height,
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
}
