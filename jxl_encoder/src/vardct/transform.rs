// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Transform pipeline: DCT dispatch and block-level transform + quantize.

use super::ac_group::{num_nonzero_8x8_except_dc, num_nonzero_except_llf};
use super::ac_strategy::{
    AcStrategyMap, RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
    RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT8X4,
    RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT16X32,
    RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32,
    RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
};
use super::afv::{afv_transform_from_pixels, dc_from_afv};
use super::block_extract::extract_block_8x8;
use super::chroma_from_luma::{CflMap, ytob_ratio, ytox_ratio};
use super::common::*;
use super::dct::{
    dc_from_dct_4x4_full, dc_from_dct_4x8_full, dc_from_dct_8x4_full, dc_from_dct_8x16,
    dc_from_dct_16x8, dc_from_dct_16x16, dc_from_dct_16x32, dc_from_dct_32x16, dc_from_dct_32x32,
    dc_from_dct_32x64, dc_from_dct_64x32, dc_from_dct_64x64, dct_4x4_full, dct_4x8_full,
    dct_8x4_full, dct_8x8, dct_8x16, dct_16x8, dct_16x16, dct_16x32, dct_32x16, dct_32x32,
    dct_32x64, dct_64x32, dct_64x64, dct2x2_transform, identity_transform,
};
use super::encoder::VarDctEncoder;
use super::frame::DistanceParams;
use super::quant::INV_DC_QUANT;
use super::quantize::adjust_quant_bias;
use crate::debug_rect;

/// Pre-allocated output buffers for `transform_and_quantize`.
///
/// Reuse across butteraugli iterations to avoid re-allocating Vec<Vec<>> arrays.
pub(crate) struct TransformOutput {
    pub quant_dc: [Vec<Vec<i16>>; 3],
    pub quant_ac: [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    pub nzeros: [Vec<Vec<u8>>; 3],
    pub raw_nzeros: [Vec<Vec<u16>>; 3],
    /// Raw (pre-CfL, pre-quantization) float DC values from dc_from_dct_NxN.
    /// These are the correct per-8×8-block DC values that account for multi-block
    /// transform structure (e.g., for DCT16, these come from the 16×16 DCT's LLF
    /// via inverse reinterpreting DCT, NOT from simple 8×8 sub-block pixel averages).
    /// Layout: `[channel][by * xsize_blocks + bx]` in XYB channel order.
    pub float_dc: [Vec<f32>; 3],
}

impl TransformOutput {
    pub fn new(xsize_blocks: usize, ysize_blocks: usize) -> Self {
        let n = xsize_blocks * ysize_blocks;
        Self {
            quant_dc: core::array::from_fn(|_| vec![vec![0i16; xsize_blocks]; ysize_blocks]),
            quant_ac: core::array::from_fn(|_| {
                vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks]
            }),
            nzeros: core::array::from_fn(|_| vec![vec![0u8; xsize_blocks]; ysize_blocks]),
            raw_nzeros: core::array::from_fn(|_| vec![vec![0u16; xsize_blocks]; ysize_blocks]),
            float_dc: core::array::from_fn(|_| vec![0.0f32; n]),
        }
    }
}

impl VarDctEncoder {
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
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                let mut dct_out = [0.0f32; 64];
                dct_8x8(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT16X8 => {
                let mut block = [0.0f32; 128];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 8..dy * 8 + 8].copy_from_slice(&channel_data[src..src + 8]);
                }
                let mut dct_out = [0.0f32; 128];
                dct_16x8(&block, &mut dct_out);
                output[..128].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT8X16 => {
                let mut block = [0.0f32; 128];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..8 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                let mut dct_out = [0.0f32; 128];
                dct_8x16(&block, &mut dct_out);
                output[..128].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT16X16 => {
                let mut block = [0.0f32; 256];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                let mut dct_out = [0.0f32; 256];
                dct_16x16(&block, &mut dct_out);
                output[..256].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT32X32 => {
                let mut block = [0.0f32; 1024];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                let mut dct_out = [0.0f32; 1024];
                dct_32x32(&block, &mut dct_out);
                output[..1024].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT4X8 => {
                // DCT4X8 full: two 4x8 transforms covering 8x8 pixels
                let mut block = [0.0f32; 64];
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                let mut dct_out = [0.0f32; 64];
                dct_4x8_full(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT8X4 => {
                // DCT8X4 full: two 8x4 transforms covering 8x8 pixels
                let mut block = [0.0f32; 64];
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                let mut dct_out = [0.0f32; 64];
                dct_8x4_full(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT4X4 => {
                // DCT4X4 full: four 4x4 transforms covering 8x8 pixels
                let mut block = [0.0f32; 64];
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                let mut dct_out = [0.0f32; 64];
                dct_4x4_full(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_IDENTITY => {
                // IDENTITY: pixel differences from reference pixel per 4x4 sub-block
                let mut input = [0.0f32; 64];
                extract_block_8x8(channel_data, stride, bx, by, &mut input);
                let mut out64 = [0.0f32; 64];
                identity_transform(&input, &mut out64);
                output[..64].copy_from_slice(&out64);
            }
            RAW_STRATEGY_DCT2X2 => {
                // DCT2X2: hierarchical 2x2 DCT
                let mut input = [0.0f32; 64];
                extract_block_8x8(channel_data, stride, bx, by, &mut input);
                let mut out64 = [0.0f32; 64];
                dct2x2_transform(&input, &mut out64);
                output[..64].copy_from_slice(&out64);
            }
            RAW_STRATEGY_DCT32X16 => {
                // DCT32X16: 32x16 transform (4 rows × 2 cols of 8x8 blocks = 32 rows × 16 cols)
                let mut block = [0.0f32; 512];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                let mut dct_out = [0.0f32; 512];
                dct_32x16(&block, &mut dct_out);
                output[..512].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT16X32 => {
                // DCT16X32: 16x32 transform (2 rows × 4 cols of 8x8 blocks = 16 rows × 32 cols)
                let mut block = [0.0f32; 512];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                let mut dct_out = [0.0f32; 512];
                dct_16x32(&block, &mut dct_out);
                output[..512].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT64X64 => {
                // DCT64X64: 64x64 transform (8 rows × 8 cols of 8x8 blocks)
                let mut block = [0.0f32; 4096];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..64 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 64..dy * 64 + 64].copy_from_slice(&channel_data[src..src + 64]);
                }
                let mut dct_out = [0.0f32; 4096];
                dct_64x64(&block, &mut dct_out);
                output[..4096].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT64X32 => {
                // DCT64X32: 64x32 transform (8 rows × 4 cols of 8x8 blocks = 64 rows × 32 cols)
                let mut block = [0.0f32; 2048];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..64 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                let mut dct_out = [0.0f32; 2048];
                dct_64x32(&block, &mut dct_out);
                output[..2048].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT32X64 => {
                // DCT32X64: 32x64 transform (4 rows × 8 cols of 8x8 blocks = 32 rows × 64 cols)
                let mut block = [0.0f32; 2048];
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 64..dy * 64 + 64].copy_from_slice(&channel_data[src..src + 64]);
                }
                let mut dct_out = [0.0f32; 2048];
                dct_32x64(&block, &mut dct_out);
                output[..2048].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
                // AFV: Adaptive Frequency Variable (hybrid transform for corners)
                // Extract 8x8 pixels and compute AFV transform
                let mut pixels = [0.0f32; 64];
                extract_block_8x8(channel_data, stride, bx, by, &mut pixels);
                let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
                let mut dct_out = [0.0f32; 64];
                afv_transform_from_pixels(&pixels, afv_kind, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            _ => unreachable!(),
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
    /// Allocates output buffers and calls `transform_and_quantize_into`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn transform_and_quantize(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        params: &DistanceParams,
        quant_field: &mut [u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
    ) -> TransformOutput {
        let mut out = TransformOutput::new(xsize_blocks, ysize_blocks);
        self.transform_and_quantize_into(
            xyb_x,
            xyb_y,
            xyb_b,
            padded_width,
            xsize_blocks,
            ysize_blocks,
            params,
            quant_field,
            cfl_map,
            ac_strategy,
            &mut out,
        );
        out
    }

    /// Fill pre-allocated `TransformOutput` buffers.
    ///
    /// All positions are written by the block loop (every first-block position
    /// gets its quant_dc, quant_ac, nzeros, raw_nzeros, and float_dc filled).
    /// No pre-clearing is needed since `is_first()` blocks cover the entire grid.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transform_and_quantize_into(
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
        out: &mut TransformOutput,
    ) {
        let quant_dc = &mut out.quant_dc;
        let quant_ac = &mut out.quant_ac;
        let nzeros = &mut out.nzeros;
        let raw_nzeros = &mut out.raw_nzeros;
        let float_dc = &mut out.float_dc;

        let channels = [xyb_x, xyb_y, xyb_b];

        // Hoist constant computations out of the block loop
        let x_qm_mul = jxl_simd::fast_powf(1.25, params.x_qm_scale as f32 - 2.0);
        let b_qm_mul = jxl_simd::fast_powf(1.25, params.b_qm_scale as f32 - 2.0);

        // Pre-allocate scratch buffers for DCT coefficients (max DCT64x64 = 4096)
        const MAX_BLOCK_SIZE: usize = 4096;
        let mut dct_scratch: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0f32; MAX_BLOCK_SIZE]);

        // Pre-compute zigzag orders for error diffusion (avoids per-block Vec allocation).
        // Index by (cx, cy) pair. Only 7 distinct pairs across all strategies.
        use super::coeff_order::natural_coeff_order;
        let zigzag_cache: Vec<(usize, usize, Vec<u32>)> = if self.error_diffusion {
            [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4), (8, 4), (8, 8)]
                .iter()
                .map(|&(cx, cy)| (cx, cy, natural_coeff_order(cx, cy)))
                .collect()
        } else {
            Vec::new()
        };
        // Scratch buffer for error diffusion corrected coefficients (reused per block)
        let mut error_scratch = if self.error_diffusion {
            vec![0.0f32; MAX_BLOCK_SIZE]
        } else {
            Vec::new()
        };

        // Scratch buffer for large-block SIMD quantization flat output (reused per block)
        let mut quant_flat_scratch = vec![0i32; MAX_BLOCK_SIZE];
        // Scratch buffers for multi-block nzeros counting (reused per block)
        let mut nz_full_block_scratch = vec![0i32; MAX_BLOCK_SIZE];
        // Max flat_nz size: for DCT64x64, covered = 8×8, flat_len = 7*xsize_blocks+8
        let mut nz_flat_scratch = vec![0u8; 7 * xsize_blocks + 8];

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

                // No fill needed — apply_dct writes all output positions
                // Alias for readability — dct_coeffs[c] is dct_scratch[c][..size]
                let dct_coeffs = &mut dct_scratch;

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
                            float_dc[1][by * xsize_blocks + bx] = dct_coeffs[1][0];
                            quant_dc[1][by][bx] = (dct_coeffs[1][0] * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT16X8 => {
                            let dcs = dc_from_dct_16x8(as_array_ref::<128>(&dct_coeffs[1], 0));
                            for iy in 0..2 {
                                float_dc[1][(by + iy) * xsize_blocks + bx] = dcs[iy];
                                quant_dc[1][by + iy][bx] = (dcs[iy] * inv_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT8X16 => {
                            let dcs = dc_from_dct_8x16(as_array_ref::<128>(&dct_coeffs[1], 0));
                            for ix in 0..2 {
                                float_dc[1][by * xsize_blocks + bx + ix] = dcs[ix];
                                quant_dc[1][by][bx + ix] = (dcs[ix] * inv_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT16X16 => {
                            let dcs = dc_from_dct_16x16(as_array_ref::<256>(&dct_coeffs[1], 0));
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
                                    float_dc[1][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 2 + ix];
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
                            let dcs = dc_from_dct_32x32(as_array_ref::<1024>(&dct_coeffs[1], 0));
                            #[cfg(feature = "debug-dc")]
                            eprintln!(
                                "DCT32x32 block (by={}, bx={}): dcs[0..4]=[{:.4}, {:.4}, {:.4}, {:.4}], LLF=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                by,
                                bx,
                                dcs[0],
                                dcs[1],
                                dcs[2],
                                dcs[3],
                                dct_coeffs[1][0],
                                dct_coeffs[1][1],
                                dct_coeffs[1][32],
                                dct_coeffs[1][33]
                            );
                            // dcs = 16 DC values in row-major 4x4
                            for iy in 0..4 {
                                for ix in 0..4 {
                                    float_dc[1][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 4 + ix];
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
                            // DCT32X16: 4 block rows × 2 block cols, returns 8 DC values in 4×2 order
                            let dcs = dc_from_dct_32x16(as_array_ref::<512>(&dct_coeffs[1], 0));
                            for iy in 0..4 {
                                for ix in 0..2 {
                                    float_dc[1][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 2 + ix];
                                    let qdc = (dcs[iy * 2 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT16X32 => {
                            // DCT16X32: 2×4 blocks, returns 8 DC values in row-major 2x4
                            let dcs = dc_from_dct_16x32(as_array_ref::<512>(&dct_coeffs[1], 0));
                            for iy in 0..2 {
                                for ix in 0..4 {
                                    float_dc[1][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 4 + ix];
                                    let qdc = (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X64 => {
                            // DCT64X64: 8×8 blocks, returns 64 DC values in row-major 8x8
                            let dcs = dc_from_dct_64x64(&dct_coeffs[1]);
                            for iy in 0..8 {
                                for ix in 0..8 {
                                    float_dc[1][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 8 + ix];
                                    let qdc = (dcs[iy * 8 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X32 => {
                            // DCT64X32: 8 block rows × 4 block cols, returns 32 DC values in 8×4 order
                            let dcs = dc_from_dct_64x32(&dct_coeffs[1]);
                            for iy in 0..8 {
                                for ix in 0..4 {
                                    float_dc[1][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 4 + ix];
                                    let qdc = (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X64 => {
                            // DCT32X64: 4×8 blocks, returns 32 DC values in row-major 4x8
                            let dcs = dc_from_dct_32x64(&dct_coeffs[1]);
                            for iy in 0..4 {
                                for ix in 0..8 {
                                    float_dc[1][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 8 + ix];
                                    let qdc = (dcs[iy * 8 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT4X8 => {
                            let dc = dc_from_dct_4x8_full(as_array_ref::<64>(&dct_coeffs[1], 0));
                            float_dc[1][by * xsize_blocks + bx] = dc;
                            quant_dc[1][by][bx] = (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT8X4 => {
                            let dc = dc_from_dct_8x4_full(as_array_ref::<64>(&dct_coeffs[1], 0));
                            float_dc[1][by * xsize_blocks + bx] = dc;
                            quant_dc[1][by][bx] = (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT4X4 => {
                            let dc = dc_from_dct_4x4_full(as_array_ref::<64>(&dct_coeffs[1], 0));
                            float_dc[1][by * xsize_blocks + bx] = dc;
                            quant_dc[1][by][bx] = (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                        | RAW_STRATEGY_AFV3 => {
                            let dc = dc_from_afv(as_array_ref::<64>(&dct_coeffs[1], 0));
                            float_dc[1][by * xsize_blocks + bx] = dc;
                            quant_dc[1][by][bx] = (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_IDENTITY | RAW_STRATEGY_DCT2X2 => {
                            // IDENTITY/DCT2X2: 1×1 coverage, DC at position [0]
                            float_dc[1][by * xsize_blocks + bx] = dct_coeffs[1][0];
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
                // Ported from libjxl enc_group.cc QuantizeRoundtripYBlockAC.
                // libjxl gates on speed_tier <= kHare (effort >= 5):
                // adjusts per-block quant and Y thresholds based on coefficient
                // statistics across all 3 channels.
                // At effort < 5: uses fixed thresholds, no per-block adjustment.
                let mut thresholds_y;
                let qac;
                {
                    let quant_idx = by * xsize_blocks + bx;
                    let mut quant_int = quant_field[quant_idx] as i32;
                    if self.profile.adjust_quant_ac {
                        // effort >= Hare: run AdjustQuantBlockAC for all 3 channels
                        let orig_qac = params.scale * quant_int as f32;
                        thresholds_y = self.profile.adjust_thresholds;
                        let mut max_quant = quant_int;
                        for &c in &[1usize, 0, 2] {
                            let mut thres = self.profile.adjust_thresholds;
                            let mut quant_c = quant_int;
                            let qm_mul = if c == 0 {
                                x_qm_mul
                            } else if c == 2 {
                                b_qm_mul
                            } else {
                                1.0
                            };
                            let weights_c = super::quant::quant_weights(raw_strategy as usize, c);
                            let (hflags, vals, err, activity) = Self::adjust_quant_block_ac(
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
                                debug_rect!(
                                    "quant/heur",
                                    bx * 8,
                                    by * 8,
                                    cx * 8,
                                    cy * 8,
                                    "c=Y flags={:06b} vals={:.0} err={:.1} act={} q={}→{}",
                                    hflags,
                                    vals,
                                    err,
                                    activity,
                                    quant_int,
                                    quant_c
                                );
                            }
                            max_quant = max_quant.max(quant_c);
                        }
                        let quant_before = quant_field[quant_idx];
                        quant_int = max_quant;
                        quant_field[quant_idx] = quant_int.clamp(1, 255) as u8;
                        debug_rect!(
                            "quant/adjust",
                            bx * 8,
                            by * 8,
                            cx * 8,
                            cy * 8,
                            "strat={} q={}→{} (e>=5 AdjustQuantBlockAC)",
                            raw_strategy,
                            quant_before,
                            quant_field[quant_idx]
                        );
                    } else {
                        // effort < Hare: fixed thresholds, no per-block adjustment
                        // (enc_group.cc:358-363)
                        thresholds_y = self.profile.fixed_thresholds_y;
                    }
                    qac = params.scale * quant_int as f32;
                }

                // ── Step 3: Quantize Y AC with thresholding ────────────────
                {
                    let c = 1;
                    let weights = super::quant::quant_weights(raw_strategy as usize, c);
                    let zigzag = if self.error_diffusion {
                        zigzag_cache
                            .iter()
                            .find(|(cx2, cy2, _)| *cx2 == cx && *cy2 == cy)
                            .map(|(_, _, v)| v.as_slice())
                    } else {
                        None
                    };
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
                        zigzag,
                        if self.error_diffusion {
                            Some(&mut error_scratch)
                        } else {
                            None
                        },
                        &mut quant_flat_scratch,
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
                        let adj = adjust_quant_bias(q, 1);
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
                            float_dc[c][by * xsize_blocks + bx] = dc;
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT16X8 => {
                            let dcs = dc_from_dct_16x8(as_array_ref::<128>(&dct_coeffs[c], 0));
                            for iy in 0..2 {
                                float_dc[c][(by + iy) * xsize_blocks + bx] = dcs[iy];
                                let y_dc = quant_dc[1][by + iy][bx] as f32;
                                quant_dc[c][by + iy][bx] =
                                    (dcs[iy] * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT8X16 => {
                            let dcs = dc_from_dct_8x16(as_array_ref::<128>(&dct_coeffs[c], 0));
                            for ix in 0..2 {
                                float_dc[c][by * xsize_blocks + bx + ix] = dcs[ix];
                                let y_dc = quant_dc[1][by][bx + ix] as f32;
                                quant_dc[c][by][bx + ix] =
                                    (dcs[ix] * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT16X16 => {
                            let dcs = dc_from_dct_16x16(as_array_ref::<256>(&dct_coeffs[c], 0));
                            for iy in 0..2 {
                                for ix in 0..2 {
                                    float_dc[c][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 2 + ix];
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 2 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X32 => {
                            let dcs = dc_from_dct_32x32(as_array_ref::<1024>(&dct_coeffs[c], 0));
                            for iy in 0..4 {
                                for ix in 0..4 {
                                    float_dc[c][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 4 + ix];
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X16 => {
                            let dcs = dc_from_dct_32x16(as_array_ref::<512>(&dct_coeffs[c], 0));
                            for iy in 0..4 {
                                for ix in 0..2 {
                                    float_dc[c][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 2 + ix];
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 2 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT16X32 => {
                            let dcs = dc_from_dct_16x32(as_array_ref::<512>(&dct_coeffs[c], 0));
                            for iy in 0..2 {
                                for ix in 0..4 {
                                    float_dc[c][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 4 + ix];
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X64 => {
                            let dcs = dc_from_dct_64x64(&dct_coeffs[c]);
                            for iy in 0..8 {
                                for ix in 0..8 {
                                    float_dc[c][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 8 + ix];
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 8 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X32 => {
                            let dcs = dc_from_dct_64x32(&dct_coeffs[c]);
                            for iy in 0..8 {
                                for ix in 0..4 {
                                    float_dc[c][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 4 + ix];
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X64 => {
                            let dcs = dc_from_dct_32x64(&dct_coeffs[c]);
                            for iy in 0..4 {
                                for ix in 0..8 {
                                    float_dc[c][(by + iy) * xsize_blocks + (bx + ix)] =
                                        dcs[iy * 8 + ix];
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 8 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT4X8 => {
                            let dc = dc_from_dct_4x8_full(as_array_ref::<64>(&dct_coeffs[c], 0));
                            float_dc[c][by * xsize_blocks + bx] = dc;
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT8X4 => {
                            let dc = dc_from_dct_8x4_full(as_array_ref::<64>(&dct_coeffs[c], 0));
                            float_dc[c][by * xsize_blocks + bx] = dc;
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT4X4 => {
                            let dc = dc_from_dct_4x4_full(as_array_ref::<64>(&dct_coeffs[c], 0));
                            float_dc[c][by * xsize_blocks + bx] = dc;
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                        | RAW_STRATEGY_AFV3 => {
                            let dc = dc_from_afv(as_array_ref::<64>(&dct_coeffs[c], 0));
                            float_dc[c][by * xsize_blocks + bx] = dc;
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_IDENTITY | RAW_STRATEGY_DCT2X2 => {
                            // IDENTITY/DCT2X2: 1×1 coverage, DC at position [0]
                            let dc = dct_coeffs[c][0];
                            float_dc[c][by * xsize_blocks + bx] = dc;
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
                    let zigzag = if self.error_diffusion {
                        zigzag_cache
                            .iter()
                            .find(|(cx2, cy2, _)| *cx2 == cx && *cy2 == cy)
                            .map(|(_, _, v)| v.as_slice())
                    } else {
                        None
                    };
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
                        zigzag,
                        if self.error_diffusion {
                            Some(&mut error_scratch)
                        } else {
                            None
                        },
                        &mut quant_flat_scratch,
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
                        let full_block = &mut nz_full_block_scratch[..size];
                        #[allow(clippy::needless_range_loop)]
                        for idx in 0..size {
                            let y = idx / stride;
                            let x = idx % stride;
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
                            full_block[idx] =
                                quant_ac[c][by + phys_row_off][bx + phys_col_off][pos_in_8x8];
                        }
                        let flat_len = (covered_y - 1) * xsize_blocks + covered_x;
                        let flat_nz = &mut nz_flat_scratch[..flat_len];
                        flat_nz.fill(0);
                        let raw_nz = num_nonzero_except_llf(
                            cx,
                            cy,
                            full_block,
                            xsize_blocks,
                            flat_nz,
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
    }
}
