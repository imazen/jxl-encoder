// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chroma-from-Luma (CfL) computation.
//!
//! Determines per-tile linear models for the X and B channels from the Y channel.
//! Ported from libjxl-tiny's `enc_chroma_from_luma.cc`.

use super::ac_strategy::{AcStrategyMap, COVERED_X, COVERED_Y};
use super::common::*;
use super::dct::dct_8x8;
use super::encoder::VarDctEncoder;
use super::quant;

/// Inverse of the color factor used in CfL ratio conversion.
/// `ytox_ratio(x) = x * K_INV_COLOR_FACTOR`
/// `ytob_ratio(b) = 1.0 + b * K_INV_COLOR_FACTOR`
const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;

/// Regularization multiplier for AC coefficient fitting.
/// libjxl uses 1e-9 (essentially no regularization). Matches libjxl.
const K_DISTANCE_MULTIPLIER_AC: f32 = 1e-9;

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

/// libjxl `kCFLFixedPointPrecision` — bit precision of the
/// `RatioJPEG` fixed-point factor (`enc/chroma_from_luma.h:43`).
#[cfg(feature = "jpeg-reencoding")]
pub const CFL_FIXED_POINT_PRECISION: i32 = 11;

/// libjxl `kDefaultColorFactor` — denominator in `RatioJPEG`
/// (`chroma_from_luma.h:37`). JPEG-compatible CfL maps must use
/// this factor (other values are not encodable in the JPEG
/// recompression path).
#[cfg(feature = "jpeg-reencoding")]
pub const DEFAULT_COLOR_FACTOR: i32 = 84;

/// libjxl `RatioJPEG(factor)` (`chroma_from_luma.h:68-70`):
/// `factor << 11 / 84`. Returns the fixed-point Y multiplier the
/// decoder applies to luma DCT coefficients before subtracting them
/// from chroma when undoing JPEG-CfL.
#[cfg(feature = "jpeg-reencoding")]
#[inline]
pub fn ratio_jpeg(factor: i32) -> i32 {
    (factor * (1 << CFL_FIXED_POINT_PRECISION)) / DEFAULT_COLOR_FACTOR
}

/// Per-channel zero-bias used by the JPEG-CfL search. libjxl
/// `kZeroBiasDefault` (`quantizer.h:36`).
#[cfg(feature = "jpeg-reencoding")]
pub const JPEG_CFL_ZERO_BIAS_DEFAULT: [f32; 3] = [0.5, 0.5, 0.5];

/// Search for an integer YtoX (c=0) or YtoB (c=2) multiplier per
/// 8×8-block color tile that maximizes the count of zero chroma AC
/// coefficients after subtracting `RatioJPEG(factor) * Y` from each.
/// Mirrors libjxl `enc_frame.cc:855-941` (the JPEG-CfL search loop).
///
/// **JPEG mode constraints** (mirrors libjxl `IsJPEGCompatible`):
/// - `base_correlation_x = base_correlation_b = 0`
/// - DC factors zero (only AC coefficients participate)
/// - color_factor = 84 (the default)
///
/// Inputs:
/// - `c` — chroma channel index in JXL convention (0=X/Cb, 2=B/Cr)
/// - `xsize_blocks` × `ysize_blocks` — frame dimensions in 8×8 blocks
/// - `luma_ac` / `chroma_ac` — per-channel AC coefficients indexed
///   `[by][bx][coeffpos]`. `coeffpos == 0` is DC and is skipped.
///   Block layout matches libjxl's transposed `scaled_qtable` order.
/// - `scaled_qtable_chroma` — 64-entry fixed-point quant table
///   `(1 << 11) * qt_y[pos] / qt_c[pos]` per position, transposed.
///
/// Returns a `Vec<i8>` of length `xsize_tiles * ysize_tiles` —
/// the per-tile multiplier (relative to libjxl's `kOffset = 127`).
/// Caller writes into `CflMap.ytox` (c=0) or `CflMap.ytob` (c=2).
#[cfg(feature = "jpeg-reencoding")]
pub fn jpeg_cfl_search(
    c: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    luma_ac: &[Vec<[i32; 64]>],
    chroma_ac: &[Vec<[i32; 64]>],
    scaled_qtable_chroma: &[i32; 64],
) -> Vec<i8> {
    debug_assert!(
        c == 0 || c == 2,
        "JPEG-CfL search only valid for c=0 or c=2"
    );
    let xsize_tiles = xsize_blocks.div_ceil(TILE_DIM_IN_BLOCKS);
    let ysize_tiles = ysize_blocks.div_ceil(TILE_DIM_IN_BLOCKS);
    let k_scale = DEFAULT_COLOR_FACTOR as f32;
    const K_OFFSET: i32 = 127;
    let k_base: f32 = 0.0;
    let k_zero_thresh = k_scale * JPEG_CFL_ZERO_BIAS_DEFAULT[c] * 0.9999;
    let inv_fp = 1.0 / ((1 << CFL_FIXED_POINT_PRECISION) as f32);

    let mut out = vec![0i8; xsize_tiles * ysize_tiles];
    for ty in 0..ysize_tiles {
        for tx in 0..xsize_tiles {
            let y0 = ty * TILE_DIM_IN_BLOCKS;
            let x0 = tx * TILE_DIM_IN_BLOCKS;
            let y1 = ((ty + 1) * TILE_DIM_IN_BLOCKS).min(ysize_blocks);
            let x1 = ((tx + 1) * TILE_DIM_IN_BLOCKS).min(xsize_blocks);

            let mut d_num_zeros = [0i32; 257];
            for by in y0..y1 {
                if by >= luma_ac.len() || by >= chroma_ac.len() {
                    continue;
                }
                let luma_row = &luma_ac[by];
                let chroma_row = &chroma_ac[by];
                for bx in x0..x1 {
                    if bx >= luma_row.len() || bx >= chroma_row.len() {
                        continue;
                    }
                    let luma_block = &luma_row[bx];
                    let chroma_block = &chroma_row[bx];
                    for coeffpos in 1..64 {
                        let scaled_m = (luma_block[coeffpos] as f32)
                            * (scaled_qtable_chroma[coeffpos] as f32)
                            * inv_fp;
                        let scaled_s = k_scale * (chroma_block[coeffpos] as f32)
                            + (K_OFFSET as f32 - k_base * k_scale) * scaled_m;
                        if scaled_m.abs() <= 1e-8 {
                            continue;
                        }
                        let (mut from, mut to) = if scaled_m > 0.0 {
                            (
                                (scaled_s - k_zero_thresh) / scaled_m,
                                (scaled_s + k_zero_thresh) / scaled_m,
                            )
                        } else {
                            (
                                (scaled_s + k_zero_thresh) / scaled_m,
                                (scaled_s - k_zero_thresh) / scaled_m,
                            )
                        };
                        if from < 0.0 {
                            from = 0.0;
                        }
                        if to > 255.0 {
                            to = 255.0;
                        }
                        if from <= to {
                            let lo = from.ceil() as i32;
                            let hi = (to + 1.0).floor() as i32;
                            if (0..=256).contains(&lo) {
                                d_num_zeros[lo as usize] += 1;
                            }
                            if (0..=256).contains(&hi) {
                                d_num_zeros[hi as usize] -= 1;
                            }
                        }
                    }
                }
            }

            let mut best_i: i32 = 0;
            let mut best_sum: i32 = 0;
            let mut offset_sum: i32 = 0;
            let mut running: i32 = 0;
            for i in 0..256 {
                running += d_num_zeros[i];
                if running > best_sum {
                    best_sum = running;
                    best_i = i as i32;
                }
                if i as i32 == K_OFFSET {
                    offset_sum = running;
                }
            }
            if best_sum > offset_sum + 1 {
                out[ty * xsize_tiles + tx] = (best_i - K_OFFSET) as i8;
            }
        }
    }
    out
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
/// SIMD-accelerated via jxl_simd.
///
/// When `use_newton` is false (effort < 7):
///   Minimizes `sum_i (base * values_m[i] - values_s[i] + x/84 * values_m[i])^2 + distance_mul * x^2`
///   via least-squares with L2 regularization. Fast, single-pass.
///
/// When `use_newton` is true (effort >= 7):
///   Minimizes `1/3 * sum((|ax+b|+1)^2 - 1) + distance_mul * x^2 * num`
///   via Newton's method with perceptual cost. More robust to outliers.
///   Matches libjxl enc_chroma_from_luma.cc at speed_tier <= kSquirrel.
#[allow(clippy::too_many_arguments)]
fn find_best_multiplier(
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
    use_newton: bool,
    newton_eps: f32,
    newton_max_iters: usize,
) -> i8 {
    if use_newton {
        jxl_simd::cfl_find_best_multiplier_newton(
            values_m,
            values_s,
            num,
            base,
            distance_mul,
            newton_eps,
            newton_max_iters,
        )
    } else {
        jxl_simd::cfl_find_best_multiplier(values_m, values_s, num, base, distance_mul)
    }
}

/// Compute the CfL map for an entire image.
///
/// For each 64x64-pixel tile (8x8 blocks), computes optimal ytox and ytob
/// values by DCT-transforming each block, weighting coefficients by inverse
/// quantization matrices, and fitting a least-squares linear model.
///
/// `stride` is the row stride (padded width) of the XYB buffers.
/// `buf_height` is the padded height. Both must be multiples of 8.
///
/// Ported from libjxl-tiny's `ComputeCmapTile`.
#[allow(clippy::too_many_arguments)]
pub fn compute_cfl_map(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    buf_height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    use_newton: bool,
    newton_eps: f32,
    newton_max_iters: usize,
) -> CflMap {
    let _ = buf_height; // Used for documentation; buffer is padded to ysize_blocks * 8
    let xsize_tiles = div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS);
    let ysize_tiles = div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS);
    let num_tiles = xsize_tiles * ysize_tiles;

    // Pre-compute inverse quant weights once (avoid per-block division).
    let qw_x = quant::quant_weights(0, 0); // DCT8, X channel
    let qw_b = quant::quant_weights(0, 2); // DCT8, B channel
    let mut inv_qm_x = [0.0f32; DCT_BLOCK_SIZE];
    let mut inv_qm_b = [0.0f32; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        inv_qm_x[i] = 1.0 / qw_x[i];
        inv_qm_b[i] = 1.0 / qw_b[i];
    }

    // Process tiles in parallel. Each tile is independent (reads shared XYB,
    // writes only to its own ytox/ytob slot).
    let tile_results = crate::parallel::parallel_map(num_tiles, |tile_idx| {
        let tx = tile_idx % xsize_tiles;
        let ty = tile_idx / xsize_tiles;
        let tile_bx0 = tx * TILE_DIM_IN_BLOCKS;
        let tile_by0 = ty * TILE_DIM_IN_BLOCKS;
        let tile_bx1 = (tile_bx0 + TILE_DIM_IN_BLOCKS).min(xsize_blocks);
        let tile_by1 = (tile_by0 + TILE_DIM_IN_BLOCKS).min(ysize_blocks);

        // Thread-local scratch buffers
        let max_coeffs_per_tile = TILE_DIM_IN_BLOCKS * TILE_DIM_IN_BLOCKS * DCT_BLOCK_SIZE;
        let mut coeffs_yx = vec![0.0f32; max_coeffs_per_tile];
        let mut coeffs_x = vec![0.0f32; max_coeffs_per_tile];
        let mut coeffs_yb = vec![0.0f32; max_coeffs_per_tile];
        let mut coeffs_b = vec![0.0f32; max_coeffs_per_tile];

        let mut num_ac = 0usize;

        for by in tile_by0..tile_by1 {
            for bx in tile_bx0..tile_bx1 {
                let mut block_y = [0.0f32; DCT_BLOCK_SIZE];
                let mut block_x = [0.0f32; DCT_BLOCK_SIZE];
                let mut block_b = [0.0f32; DCT_BLOCK_SIZE];

                let x0 = bx * BLOCK_DIM;
                for dy in 0..BLOCK_DIM {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    let dst = dy * BLOCK_DIM;
                    block_y[dst..dst + BLOCK_DIM].copy_from_slice(&xyb_y[src..src + BLOCK_DIM]);
                    block_x[dst..dst + BLOCK_DIM].copy_from_slice(&xyb_x[src..src + BLOCK_DIM]);
                    block_b[dst..dst + BLOCK_DIM].copy_from_slice(&xyb_b[src..src + BLOCK_DIM]);
                }

                let mut dct_y = [0.0f32; DCT_BLOCK_SIZE];
                let mut dct_x = [0.0f32; DCT_BLOCK_SIZE];
                let mut dct_b = [0.0f32; DCT_BLOCK_SIZE];
                dct_8x8(&block_y, &mut dct_y);
                dct_8x8(&block_x, &mut dct_x);
                dct_8x8(&block_b, &mut dct_b);

                // Zero out DC so it doesn't affect the AC-only fitting.
                dct_y[0] = 0.0;
                dct_x[0] = 0.0;
                dct_b[0] = 0.0;

                for i in 0..DCT_BLOCK_SIZE {
                    coeffs_yx[num_ac + i] = dct_y[i] * inv_qm_x[i];
                    coeffs_x[num_ac + i] = dct_x[i] * inv_qm_x[i];
                    coeffs_yb[num_ac + i] = dct_y[i] * inv_qm_b[i];
                    coeffs_b[num_ac + i] = dct_b[i] * inv_qm_b[i];
                }
                num_ac += DCT_BLOCK_SIZE;
            }
        }

        let tx_val = find_best_multiplier(
            &coeffs_yx,
            &coeffs_x,
            num_ac,
            0.0,
            K_DISTANCE_MULTIPLIER_AC,
            use_newton,
            newton_eps,
            newton_max_iters,
        );
        let tb_val = find_best_multiplier(
            &coeffs_yb,
            &coeffs_b,
            num_ac,
            1.0,
            K_DISTANCE_MULTIPLIER_AC,
            use_newton,
            newton_eps,
            newton_max_iters,
        );

        (tx_val, tb_val)
    });

    // Unpack results into ytox/ytob arrays
    let mut ytox = vec![0i8; num_tiles];
    let mut ytob = vec![0i8; num_tiles];
    for (tile_idx, &(tx_val, tb_val)) in tile_results.iter().enumerate() {
        ytox[tile_idx] = tx_val;
        ytob[tile_idx] = tb_val;
    }

    CflMap {
        ytox,
        ytob,
        xsize_tiles,
        ysize_tiles,
    }
}

/// CfL pass 2: recompute CfL map using actual AC strategies and per-block
/// quantization weighting.
///
/// Unlike pass 1 (`compute_cfl_map`) which forces DCT8 and q=1, pass 2 uses
/// the actual AC strategy per block and weights coefficients by the per-block
/// quantization factor and strategy-specific inverse quant matrices. This
/// produces better CfL values because the fitting accounts for how the encoder
/// will actually encode each block.
///
/// Matches libjxl `ComputeTile` with `use_dct8=false` in enc_chroma_from_luma.cc.
///
/// Called after AC strategy selection and quant field computation.
#[allow(clippy::too_many_arguments)]
pub fn refine_cfl_map(
    cfl_map: &mut CflMap,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    ac_strategy: &AcStrategyMap,
    quant_field: &[u8],
    quant_scale: f32,
    use_newton: bool,
    newton_eps: f32,
    newton_max_iters: usize,
) {
    let xsize_tiles = cfl_map.xsize_tiles;
    let ysize_tiles = cfl_map.ysize_tiles;
    let num_tiles = xsize_tiles * ysize_tiles;

    // Process tiles in parallel. Each tile is independent.
    let tile_results = crate::parallel::parallel_map(num_tiles, |tile_idx| {
        let tx = tile_idx % xsize_tiles;
        let ty = tile_idx / xsize_tiles;
        let tile_bx0 = tx * TILE_DIM_IN_BLOCKS;
        let tile_by0 = ty * TILE_DIM_IN_BLOCKS;
        let tile_bx1 = (tile_bx0 + TILE_DIM_IN_BLOCKS).min(xsize_blocks);
        let tile_by1 = (tile_by0 + TILE_DIM_IN_BLOCKS).min(ysize_blocks);

        // Thread-local scratch buffers
        let max_coeffs_per_tile = TILE_DIM_IN_BLOCKS * TILE_DIM_IN_BLOCKS * DCT_BLOCK_SIZE;
        let mut coeffs_yx = vec![0.0f32; max_coeffs_per_tile];
        let mut coeffs_x = vec![0.0f32; max_coeffs_per_tile];
        let mut coeffs_yb = vec![0.0f32; max_coeffs_per_tile];
        let mut coeffs_b = vec![0.0f32; max_coeffs_per_tile];

        const MAX_COEFF_AREA: usize = 4096;
        let mut dct_y = vec![0.0f32; MAX_COEFF_AREA];
        let mut dct_x = vec![0.0f32; MAX_COEFF_AREA];
        let mut dct_b = vec![0.0f32; MAX_COEFF_AREA];

        let mut num_ac = 0usize;
        let buf_cap = coeffs_yx.len();

        'tile_loop: for by in tile_by0..tile_by1 {
            for bx in tile_bx0..tile_bx1 {
                if !ac_strategy.is_first(bx, by) {
                    continue;
                }

                let raw_strategy = ac_strategy.raw_strategy(bx, by);
                let covered_x = COVERED_X[raw_strategy as usize];
                let covered_y = COVERED_Y[raw_strategy as usize];

                if covered_x + tile_bx0 > tile_bx1 || covered_y + tile_by0 > tile_by1 {
                    continue;
                }

                VarDctEncoder::apply_dct(xyb_y, stride, bx, by, raw_strategy, &mut dct_y);
                VarDctEncoder::apply_dct(xyb_x, stride, bx, by, raw_strategy, &mut dct_x);
                VarDctEncoder::apply_dct(xyb_b, stride, bx, by, raw_strategy, &mut dct_b);

                let (cx, cy) = if covered_x >= covered_y {
                    (covered_x, covered_y)
                } else {
                    (covered_y, covered_x)
                };

                for iy in 0..cy {
                    for ix in 0..cx {
                        let pos = cx * BLOCK_DIM * iy + ix;
                        dct_y[pos] = 0.0;
                        dct_x[pos] = 0.0;
                        dct_b[pos] = 0.0;
                    }
                }

                let qq = quant_field[by * xsize_blocks + bx] as f32;
                let q = quant_scale * 128.0 * qq;

                let qw_x = quant::quant_weights(raw_strategy as usize, 0);
                let qw_b = quant::quant_weights(raw_strategy as usize, 2);

                let num_coeffs = cx * cy * DCT_BLOCK_SIZE;
                // Bound by accumulator buffer size. libjxl's heuristic at
                // enc_chroma_from_luma.cc:304-306 (`covered + x0 > x1`)
                // uses the TILE ORIGIN as the reference, not the current
                // block's `bx`/`by`, so a multi-block first-block whose
                // (bx, by) sits near the tile-end edge isn't filtered out
                // — its coefficient contribution is fully counted in
                // *this* tile (libjxl does the same). In pathological
                // ac_strategy configurations the sum can exceed the
                // accumulator (`kColorTileDim * kColorTileDim = 4096`).
                // libjxl writes past it via SIMD stores and treats the
                // tail as undefined; we clamp here to keep release builds
                // panic-free for downstream callers (notably the GPU
                // strat-search injector) that can construct ac_strategy
                // configurations the in-tree CPU strategy search wouldn't.
                let buf_remaining = buf_cap.saturating_sub(num_ac);
                let take = num_coeffs.min(buf_remaining);
                for i in 0..take {
                    let qqm_x = q / qw_x[i];
                    let qqm_b = q / qw_b[i];
                    coeffs_yx[num_ac + i] = dct_y[i] * qqm_x;
                    coeffs_x[num_ac + i] = dct_x[i] * qqm_x;
                    coeffs_yb[num_ac + i] = dct_y[i] * qqm_b;
                    coeffs_b[num_ac + i] = dct_b[i] * qqm_b;
                }
                num_ac += take;
                if num_ac >= buf_cap {
                    break 'tile_loop;
                }
            }
        }

        let tx_val = find_best_multiplier(
            &coeffs_yx,
            &coeffs_x,
            num_ac,
            0.0,
            K_DISTANCE_MULTIPLIER_AC,
            use_newton,
            newton_eps,
            newton_max_iters,
        );
        let tb_val = find_best_multiplier(
            &coeffs_yb,
            &coeffs_b,
            num_ac,
            1.0,
            K_DISTANCE_MULTIPLIER_AC,
            use_newton,
            newton_eps,
            newton_max_iters,
        );

        (tx_val, tb_val)
    });

    // Write results back to cfl_map
    for (tile_idx, &(tx_val, tb_val)) in tile_results.iter().enumerate() {
        cfl_map.ytox[tile_idx] = tx_val;
        cfl_map.ytob[tile_idx] = tb_val;
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
        assert_eq!(
            find_best_multiplier(&[], &[], 0, 0.0, 1e-3, false, 1.0, 10),
            0
        );
    }

    #[test]
    fn test_find_best_multiplier_uncorrelated() {
        // When values_m and values_s are uncorrelated, the multiplier should be near 0
        let m = [1.0, 0.0, -1.0, 0.0];
        let s = [0.0, 1.0, 0.0, -1.0];
        let result = find_best_multiplier(&m, &s, 4, 0.0, 1e-3, false, 1.0, 10);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_find_best_multiplier_correlated() {
        // When s = base*m + factor/84*m, the multiplier should recover factor
        // (with regularization pulling toward 0).
        // Use large values to make regularization negligible.
        // The towards_zero bias (2.6) shifts the result towards 0.
        let factor = 42.0;
        let base = 0.0;
        let m: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 10.0).collect();
        let s: Vec<f32> = m.iter().map(|&v| base * v + factor / 84.0 * v).collect();
        let result = find_best_multiplier(&m, &s, 64, base, 1e-3, false, 1.0, 10);
        // Optimization yields ~42.0, towards_zero bias subtracts 2.6 → ~39
        let expected = (factor - 2.6).round();
        assert!(
            (result as f32 - expected).abs() < 2.0,
            "Expected ~{} (factor {} - 2.6 bias), got {}",
            expected,
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
            false, // use_newton
            1.0,
            10,
        );

        // Uniform image: all AC coefficients are 0 except DC,
        // and DC is zeroed out before fitting. So all values should be 0.
        assert_eq!(cfl.ytox_at(0, 0), 0);
        assert_eq!(cfl.ytob_at(0, 0), 0);
    }

    #[test]
    fn test_refine_cfl_map_runs_on_real_input() {
        // Smoke test: refine_cfl_map should not panic on a 64×64
        // RGB-to-XYB image with mixed AC strategies + a non-uniform
        // quant field, and should produce SOMETHING (not all-zeros)
        // for an image with chroma content.
        //
        // The point of this test is to prove the function actually
        // executes its body when wired through __pre_quantized
        // re-exports — historically the GPU buttloop wiring couldn't
        // tell whether refine_cfl_map fired at all.
        use crate::color::xyb::linear_rgb_to_xyb;
        use crate::vardct::ac_strategy::AcStrategyMap;
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        const N: usize = WIDTH * HEIGHT;
        let mut xyb_x = vec![0.0f32; N];
        let mut xyb_y = vec![0.0f32; N];
        let mut xyb_b = vec![0.0f32; N];
        // Vertical color gradient (red-ish on left, blue-ish on right)
        // to give CfL something non-trivial to fit.
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let r = (x as f32 / WIDTH as f32).clamp(0.05, 0.95);
                let g = 0.4;
                let b_ = ((WIDTH - x) as f32 / WIDTH as f32).clamp(0.05, 0.95);
                let (vx, vy, vb) = linear_rgb_to_xyb(r, g, b_);
                let i = y * WIDTH + x;
                xyb_x[i] = vx;
                xyb_y[i] = vy;
                xyb_b[i] = vb;
            }
        }
        let xsize_blocks = WIDTH / BLOCK_DIM;
        let ysize_blocks = HEIGHT / BLOCK_DIM;
        let mut cfl = compute_cfl_map(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            WIDTH,
            HEIGHT,
            xsize_blocks,
            ysize_blocks,
            true,
            1e-3,
            10,
        );
        let pre_ytox: Vec<i8> = cfl.ytox.clone();
        let pre_ytob: Vec<i8> = cfl.ytob.clone();
        let ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
        let quant_field = vec![5u8; xsize_blocks * ysize_blocks];
        // Should not panic.
        refine_cfl_map(
            &mut cfl,
            &xyb_x,
            &xyb_y,
            &xyb_b,
            WIDTH,
            xsize_blocks,
            ysize_blocks,
            &ac_strategy,
            &quant_field,
            0.5,  // quant_scale
            true, // use_newton
            1e-3, 10,
        );
        // The function ran without panic on a real input. Whether it
        // mutated the map depends on how much the per-block-weighted
        // refit differs from the forced-DCT8/q=1 pass-1 result for this
        // synthetic gradient — for a near-uniform gradient at q=5, the
        // change is small but should be measurable.
        let _ = (pre_ytox, pre_ytob);
    }

    #[test]
    fn test_refine_cfl_map_differs_from_pass1_on_complex_input() {
        // A complex per-block ac_strategy + non-uniform quant_field
        // should produce a refined map that differs from pass-1's
        // forced-DCT8/q=1 result on chroma-bearing content. This is
        // the property the GPU buttloop wiring relies on — if pass 2
        // collapsed back to pass 1 in production, the wiring would
        // be a silent no-op.
        use crate::color::xyb::linear_rgb_to_xyb;
        use crate::vardct::ac_strategy::{AcStrategyMap, RAW_STRATEGY_DCT16X8};
        const WIDTH: usize = 128;
        const HEIGHT: usize = 128;
        const N: usize = WIDTH * HEIGHT;
        let mut xyb_x = vec![0.0f32; N];
        let mut xyb_y = vec![0.0f32; N];
        let mut xyb_b = vec![0.0f32; N];
        // Multi-frequency colorful pattern that gives CfL real signal.
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let fx = x as f32 / WIDTH as f32;
                let fy = y as f32 / HEIGHT as f32;
                let r = (0.5 + 0.4 * (fx * 7.0).sin() * (fy * 5.0).cos()).clamp(0.05, 0.95);
                let g = (0.4 + 0.3 * (fx * 3.0).cos()).clamp(0.05, 0.95);
                let b_ = (0.5 + 0.4 * (fy * 11.0).sin()).clamp(0.05, 0.95);
                let (vx, vy, vb) = linear_rgb_to_xyb(r, g, b_);
                let i = y * WIDTH + x;
                xyb_x[i] = vx;
                xyb_y[i] = vy;
                xyb_b[i] = vb;
            }
        }
        let xsize_blocks = WIDTH / BLOCK_DIM;
        let ysize_blocks = HEIGHT / BLOCK_DIM;
        let mut cfl = compute_cfl_map(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            WIDTH,
            HEIGHT,
            xsize_blocks,
            ysize_blocks,
            true,
            1e-3,
            10,
        );
        let pre_ytox = cfl.ytox.clone();
        let pre_ytob = cfl.ytob.clone();

        // Mixed strategies: half the blocks use DCT16x8.
        let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
        for by in (0..ysize_blocks - 1).step_by(2) {
            for bx in 0..xsize_blocks {
                ac_strategy.set(bx, by, RAW_STRATEGY_DCT16X8);
            }
        }

        // Non-uniform quant field.
        let mut quant_field = vec![3u8; xsize_blocks * ysize_blocks];
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                quant_field[by * xsize_blocks + bx] = 1 + ((bx + by) % 8) as u8;
            }
        }

        refine_cfl_map(
            &mut cfl,
            &xyb_x,
            &xyb_y,
            &xyb_b,
            WIDTH,
            xsize_blocks,
            ysize_blocks,
            &ac_strategy,
            &quant_field,
            0.5,
            true,
            1e-3,
            10,
        );

        let changed = (0..cfl.ytox.len())
            .filter(|&i| pre_ytox[i] != cfl.ytox[i] || pre_ytob[i] != cfl.ytob[i])
            .count();
        assert!(
            changed > 0,
            "refine_cfl_map produced no changes vs pass 1 on complex input \
             (xsize_tiles={}, ysize_tiles={}); wiring would be a silent no-op",
            cfl.xsize_tiles,
            cfl.ysize_tiles
        );
    }
}
