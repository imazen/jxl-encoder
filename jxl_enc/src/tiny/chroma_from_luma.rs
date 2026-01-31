// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Chroma-from-Luma (CfL) computation.
//!
//! Determines per-tile linear models for the X and B channels from the Y channel.
//! Ported from libjxl-tiny's `enc_chroma_from_luma.cc`.

use super::common::*;
use super::dct::dct_8x8;
use super::quant;

/// Inverse of the color factor used in CfL ratio conversion.
/// `ytox_ratio(x) = x * K_INV_COLOR_FACTOR`
/// `ytob_ratio(b) = 1.0 + b * K_INV_COLOR_FACTOR`
const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;

/// Regularization multiplier for AC coefficient fitting.
const K_DISTANCE_MULTIPLIER_AC: f32 = 1e-3;

/// Convert a ytox i8 value to the ratio used for CfL subtraction.
#[inline]
pub fn ytox_ratio(x: i8) -> f32 {
    x as f32 * K_INV_COLOR_FACTOR
}

/// Convert a ytob i8 value to the ratio used for CfL subtraction.
#[inline]
pub fn ytob_ratio(b: i8) -> f32 {
    1.0 + b as f32 * K_INV_COLOR_FACTOR
}

/// Per-tile chroma-from-luma map.
pub struct CflMap {
    /// YtoX values per tile, row-major.
    pub ytox: Vec<i8>,
    /// YtoB values per tile, row-major.
    pub ytob: Vec<i8>,
    /// Number of tiles in x direction.
    pub xsize_tiles: usize,
    /// Number of tiles in y direction.
    #[allow(dead_code)]
    pub ysize_tiles: usize,
}

impl CflMap {
    /// Create a CfL map with all zeros (no chroma decorrelation).
    pub fn zeros(xsize_tiles: usize, ysize_tiles: usize) -> Self {
        let n = xsize_tiles * ysize_tiles;
        Self {
            ytox: vec![0i8; n],
            ytob: vec![0i8; n],
            xsize_tiles,
            ysize_tiles,
        }
    }

    /// Get the ytox value for a tile at (tx, ty).
    #[inline]
    pub fn ytox_at(&self, tx: usize, ty: usize) -> i8 {
        self.ytox[ty * self.xsize_tiles + tx]
    }

    /// Get the ytob value for a tile at (tx, ty).
    #[inline]
    pub fn ytob_at(&self, tx: usize, ty: usize) -> i8 {
        self.ytob[ty * self.xsize_tiles + tx]
    }
}

/// Find the best integer multiplier for a chroma-from-luma linear model.
///
/// Minimizes `sum_i (base * values_m[i] - values_s[i] + x/84 * values_m[i])^2 + distance_mul * x^2`
/// via least-squares with L2 regularization.
///
/// Ported from libjxl-tiny's `FindBestMultiplier`.
fn find_best_multiplier(
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    if num == 0 {
        return 0;
    }
    let mut sum_aa = 0.0f32;
    let mut sum_ab = 0.0f32;
    for i in 0..num {
        // color residual = a*x + b, where a = values_m[i] / 84, b = base * values_m[i] - values_s[i]
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
    }
    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    x.round().clamp(-128.0, 127.0) as i8
}

/// Compute the CfL map for an entire image.
///
/// For each 64x64-pixel tile (8x8 blocks), computes optimal ytox and ytob
/// values by DCT-transforming each block, weighting coefficients by inverse
/// quantization matrices, and fitting a least-squares linear model.
///
/// Ported from libjxl-tiny's `ComputeCmapTile`.
pub fn compute_cfl_map(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> CflMap {
    let xsize_tiles = div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS);
    let ysize_tiles = div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS);
    let num_tiles = xsize_tiles * ysize_tiles;

    let mut ytox = vec![0i8; num_tiles];
    let mut ytob = vec![0i8; num_tiles];

    // Pre-fetch inverse quant weights for X and B channels (DCT8 strategy).
    // C++ uses dequant.InvMatrix(strategy, channel) which is 1/kQuantWeights.
    let qw_x = quant::quant_weights(0, 0); // DCT8, X channel — small values
    let qw_b = quant::quant_weights(0, 2); // DCT8, B channel — small values

    // Max coefficients per tile: 8*8 blocks * 64 coefficients = 4096
    let max_coeffs_per_tile = TILE_DIM_IN_BLOCKS * TILE_DIM_IN_BLOCKS * DCT_BLOCK_SIZE;
    let mut coeffs_yx = vec![0.0f32; max_coeffs_per_tile];
    let mut coeffs_x = vec![0.0f32; max_coeffs_per_tile];
    let mut coeffs_yb = vec![0.0f32; max_coeffs_per_tile];
    let mut coeffs_b = vec![0.0f32; max_coeffs_per_tile];

    for ty in 0..ysize_tiles {
        for tx in 0..xsize_tiles {
            let tile_bx0 = tx * TILE_DIM_IN_BLOCKS;
            let tile_by0 = ty * TILE_DIM_IN_BLOCKS;
            let tile_bx1 = (tile_bx0 + TILE_DIM_IN_BLOCKS).min(xsize_blocks);
            let tile_by1 = (tile_by0 + TILE_DIM_IN_BLOCKS).min(ysize_blocks);

            let mut num_ac = 0usize;

            for by in tile_by0..tile_by1 {
                for bx in tile_bx0..tile_bx1 {
                    // Extract and DCT each channel for this block
                    let mut block_y = [0.0f32; DCT_BLOCK_SIZE];
                    let mut block_x = [0.0f32; DCT_BLOCK_SIZE];
                    let mut block_b = [0.0f32; DCT_BLOCK_SIZE];

                    for dy in 0..BLOCK_DIM {
                        for dx in 0..BLOCK_DIM {
                            let py = (by * BLOCK_DIM + dy).min(height - 1);
                            let px = (bx * BLOCK_DIM + dx).min(width - 1);
                            let idx = py * width + px;
                            block_y[dy * BLOCK_DIM + dx] = xyb_y[idx];
                            block_x[dy * BLOCK_DIM + dx] = xyb_x[idx];
                            block_b[dy * BLOCK_DIM + dx] = xyb_b[idx];
                        }
                    }

                    let mut dct_y = [0.0f32; DCT_BLOCK_SIZE];
                    let mut dct_x = [0.0f32; DCT_BLOCK_SIZE];
                    let mut dct_b = [0.0f32; DCT_BLOCK_SIZE];
                    dct_8x8(&block_y, &mut dct_y);
                    dct_8x8(&block_x, &mut dct_x);
                    dct_8x8(&block_b, &mut dct_b);

                    // Zero out DC so it doesn't affect the AC-only fitting.
                    // C++ does this explicitly: block_y[0] = block_x[0] = block_b[0] = 0
                    dct_y[0] = 0.0;
                    dct_x[0] = 0.0;
                    dct_b[0] = 0.0;

                    // Multiply by inverse quant weights (1/kQuantWeights = InvMatrix)
                    // and accumulate into coefficient arrays.
                    for i in 0..DCT_BLOCK_SIZE {
                        let inv_qm_x = 1.0 / qw_x[i]; // InvMatrix for X channel
                        let inv_qm_b = 1.0 / qw_b[i]; // InvMatrix for B channel
                        coeffs_yx[num_ac + i] = dct_y[i] * inv_qm_x;
                        coeffs_x[num_ac + i] = dct_x[i] * inv_qm_x;
                        coeffs_yb[num_ac + i] = dct_y[i] * inv_qm_b;
                        coeffs_b[num_ac + i] = dct_b[i] * inv_qm_b;
                    }
                    num_ac += DCT_BLOCK_SIZE;
                }
            }

            let tile_idx = ty * xsize_tiles + tx;
            ytox[tile_idx] =
                find_best_multiplier(&coeffs_yx, &coeffs_x, num_ac, 0.0, K_DISTANCE_MULTIPLIER_AC);
            ytob[tile_idx] =
                find_best_multiplier(&coeffs_yb, &coeffs_b, num_ac, 1.0, K_DISTANCE_MULTIPLIER_AC);
        }
    }

    CflMap {
        ytox,
        ytob,
        xsize_tiles,
        ysize_tiles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ytox_ratio() {
        assert_eq!(ytox_ratio(0), 0.0);
        assert!((ytox_ratio(84) - 1.0).abs() < 1e-6);
        assert!((ytox_ratio(-84) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ytob_ratio() {
        assert_eq!(ytob_ratio(0), 1.0);
        assert!((ytob_ratio(84) - 2.0).abs() < 1e-6);
        assert!((ytob_ratio(-84) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_best_multiplier_zero_input() {
        assert_eq!(find_best_multiplier(&[], &[], 0, 0.0, 1e-3), 0);
    }

    #[test]
    fn test_find_best_multiplier_uncorrelated() {
        // When values_m and values_s are uncorrelated, the multiplier should be near 0
        let m = [1.0, 0.0, -1.0, 0.0];
        let s = [0.0, 1.0, 0.0, -1.0];
        let result = find_best_multiplier(&m, &s, 4, 0.0, 1e-3);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_find_best_multiplier_correlated() {
        // When s = base*m + factor/84*m, the multiplier should recover factor
        // (with regularization pulling toward 0).
        // Use large values to make regularization negligible.
        let factor = 42.0;
        let base = 0.0;
        let m: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 10.0).collect();
        let s: Vec<f32> = m.iter().map(|&v| base * v + factor / 84.0 * v).collect();
        let result = find_best_multiplier(&m, &s, 64, base, 1e-3);
        assert!(
            (result as f32 - factor).abs() < 2.0,
            "Expected ~{}, got {}",
            factor,
            result
        );
    }

    #[test]
    fn test_cfl_map_uniform_gray() {
        // Uniform gray image: all channels identical after XYB transform
        // means X≈0, B≈Y, so CfL should produce ytox≈0, ytob≈0
        use crate::color::xyb::linear_rgb_to_xyb;

        let width = 16;
        let height = 16;
        let n = width * height;
        let mut xyb_x = vec![0.0f32; n];
        let mut xyb_y = vec![0.0f32; n];
        let mut xyb_b = vec![0.0f32; n];

        for i in 0..n {
            let (x, y, b) = linear_rgb_to_xyb(0.5, 0.5, 0.5);
            xyb_x[i] = x;
            xyb_y[i] = y;
            xyb_b[i] = b;
        }

        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let cfl = compute_cfl_map(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            width,
            height,
            xsize_blocks,
            ysize_blocks,
        );

        // Uniform image: all AC coefficients are 0 except DC,
        // and DC is zeroed out before fitting. So all values should be 0.
        assert_eq!(cfl.ytox_at(0, 0), 0);
        assert_eq!(cfl.ytob_at(0, 0), 0);
    }
}
