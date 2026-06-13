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
use crate::budget::MemoryBudget;
use crate::debug_rect;
use crate::error::{Error, Result};
use alloc::sync::Arc;

/// Pre-allocated output buffers for `transform_and_quantize`.
///
/// Reuse across butteraugli iterations to avoid re-allocating Vec<Vec<>> arrays.
//
// Visibility: `pub` so the `__pre_quantized` test path can call
// `transform_and_quantize_for_test` and inspect/forward the
// produced data. The constructor (`new`) is `pub(crate)` because
// it takes a `pub(crate) MemoryBudget`; downstream callers obtain
// instances from `transform_and_quantize_for_test`.
pub struct TransformOutput {
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
    /// RAII budget reservation released when this `TransformOutput`
    /// drops. The base-encode `TransformOutput` and the perceptual
    /// loop's scratch `TransformOutput` (`perceptual_loop.rs`,
    /// `ssim2_loop.rs`, `zensim_loop.rs`) never overlap in real memory —
    /// the loop's is dropped before the base one is built — so reserving
    /// permanently double-counted ~149 MB at 12 MP against the budget.
    /// Holding the guard makes the budget high-water mark track real
    /// peak RSS. Same fix pattern as W44-AUDIT-2 in `epf.rs`.
    _budget_guard: crate::budget::BudgetGuard,
}

impl TransformOutput {
    pub(crate) fn new(
        xsize_blocks: usize,
        ysize_blocks: usize,
        budget: Option<&Arc<MemoryBudget>>,
    ) -> Result<Self> {
        let n = xsize_blocks
            .checked_mul(ysize_blocks)
            .ok_or(Error::DimensionOverflow {
                width: xsize_blocks,
                height: ysize_blocks,
                channels: 3,
            })?;
        // Bytes per channel:
        //   quant_dc:    n * sizeof(i16)              = 2n
        //   quant_ac:    n * 64 * sizeof(i32)         = 256n
        //   nzeros:      n * sizeof(u8)               = n
        //   raw_nzeros:  n * sizeof(u16)              = 2n
        //   float_dc:    n * sizeof(f32)              = 4n
        // Per channel: 265n; three channels: 795n.
        let bytes = (n as u64).saturating_mul(265 * 3);
        // RAII reservation (not permanent): released on drop so the
        // perceptual loop's scratch copy doesn't double-count against
        // the budget after it is freed. See the `_budget_guard` field doc.
        let _budget_guard = MemoryBudget::reserve_opt(budget, bytes)?;
        Ok(Self {
            quant_dc: core::array::from_fn(|_| vec![vec![0i16; xsize_blocks]; ysize_blocks]),
            quant_ac: core::array::from_fn(|_| {
                vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks]
            }),
            nzeros: core::array::from_fn(|_| vec![vec![0u8; xsize_blocks]; ysize_blocks]),
            raw_nzeros: core::array::from_fn(|_| vec![vec![0u16; xsize_blocks]; ysize_blocks]),
            float_dc: core::array::from_fn(|_| vec![0.0f32; n]),
            _budget_guard,
        })
    }
}

/// Per-group transform results with locally-indexed output arrays.
///
/// Storage: flat `Box<[T]>` per channel per field — `width * height`
/// entries indexed as `[ly * width + lx]`. Single allocation per
/// channel-field instead of `(1 + height)` allocations as the prior
/// `Vec<Vec<T>>` shape required.
///
/// At 12 MP this drops per-group allocations from ~400 (one outer Vec
/// plus 32 inner Vecs × 5 fields × 3 channels) to 16 (one Box per field
/// per channel + one quant_adjustments Vec). Across 204 groups that's
/// ~80,000 → ~3,300 allocations per encode — a meaningful win on
/// allocator-sensitive platforms (Windows in particular).
///
/// Layout note: outer indexing stays per-channel, but the inner row
/// is now flat: `quant_dc[c][ly * width + lx]` instead of the prior
/// `quant_dc[c][ly][lx]`.
pub(crate) struct GroupTransformResult {
    pub start_bx: usize,
    pub start_by: usize,
    pub width: usize,
    pub height: usize,
    pub quant_dc: [Box<[i16]>; 3],
    pub quant_ac: [Box<[[i32; DCT_BLOCK_SIZE]]>; 3],
    pub nzeros: [Box<[u8]>; 3],
    pub raw_nzeros: [Box<[u16]>; 3],
    pub float_dc: [Box<[f32]>; 3],
    /// Quant field adjustments: (global_index, new_value).
    pub quant_adjustments: Vec<(usize, u8)>,
}

impl GroupTransformResult {
    pub fn new(start_bx: usize, start_by: usize, width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            start_bx,
            start_by,
            width,
            height,
            quant_dc: core::array::from_fn(|_| vec![0i16; n].into_boxed_slice()),
            quant_ac: core::array::from_fn(|_| vec![[0i32; DCT_BLOCK_SIZE]; n].into_boxed_slice()),
            nzeros: core::array::from_fn(|_| vec![0u8; n].into_boxed_slice()),
            raw_nzeros: core::array::from_fn(|_| vec![0u16; n].into_boxed_slice()),
            float_dc: core::array::from_fn(|_| vec![0.0f32; n].into_boxed_slice()),
            quant_adjustments: Vec::new(),
        }
    }

    /// Copy this group's results into the global TransformOutput.
    pub fn scatter_into(self, out: &mut TransformOutput, xsize_blocks: usize) {
        let width = self.width;
        for c in 0..3 {
            for ly in 0..self.height {
                let gy = self.start_by + ly;
                let row_off = ly * width;
                for lx in 0..width {
                    let gx = self.start_bx + lx;
                    let i = row_off + lx;
                    out.quant_dc[c][gy][gx] = self.quant_dc[c][i];
                    out.quant_ac[c][gy][gx] = self.quant_ac[c][i];
                    out.nzeros[c][gy][gx] = self.nzeros[c][i];
                    out.raw_nzeros[c][gy][gx] = self.raw_nzeros[c][i];
                    out.float_dc[c][gy * xsize_blocks + gx] = self.float_dc[c][i];
                }
            }
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
        use super::common::{as_array_mut, uninit_buf};

        match raw_strategy {
            0 => {
                let mut block = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                dct_8x8(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT16X8 => {
                let mut block = uninit_buf::<128>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 8..dy * 8 + 8].copy_from_slice(&channel_data[src..src + 8]);
                }
                dct_16x8(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT8X16 => {
                let mut block = uninit_buf::<128>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..8 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                dct_8x16(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT16X16 => {
                let mut block = uninit_buf::<256>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                dct_16x16(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT32X32 => {
                let mut block = uninit_buf::<1024>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                dct_32x32(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT4X8 => {
                let mut block = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                dct_4x8_full(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT8X4 => {
                let mut block = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                dct_8x4_full(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT4X4 => {
                let mut block = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                dct_4x4_full(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_IDENTITY => {
                let mut input = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut input);
                identity_transform(&input, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT2X2 => {
                let mut input = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut input);
                dct2x2_transform(&input, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT32X16 => {
                let mut block = uninit_buf::<512>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                dct_32x16(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT16X32 => {
                let mut block = uninit_buf::<512>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                dct_16x32(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT64X64 => {
                let mut block = uninit_buf::<4096>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..64 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 64..dy * 64 + 64].copy_from_slice(&channel_data[src..src + 64]);
                }
                dct_64x64(&block, &mut output[..4096]);
            }
            RAW_STRATEGY_DCT64X32 => {
                let mut block = uninit_buf::<2048>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..64 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                dct_64x32(&block, &mut output[..2048]);
            }
            RAW_STRATEGY_DCT32X64 => {
                let mut block = uninit_buf::<2048>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 64..dy * 64 + 64].copy_from_slice(&channel_data[src..src + 64]);
                }
                dct_32x64(&block, &mut output[..2048]);
            }
            RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
                let mut pixels = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut pixels);
                let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
                afv_transform_from_pixels(&pixels, afv_kind, as_array_mut(output, 0));
            }
            _ => unreachable!(),
        }
    }

    /// Process a RANGE of blocks, writing to a `GroupTransformResult` with local coordinates.
    ///
    /// This is the same algorithm as `transform_and_quantize_into` but operates on a
    /// sub-rectangle `[start_by..end_by, start_bx..end_bx]` and writes results into
    /// locally-indexed arrays in `result`. The `quant_field` is read-only; any quant
    /// adjustments are recorded in `result.quant_adjustments` for later application.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transform_blocks_into(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        xsize_blocks: usize,
        params: &DistanceParams,
        quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        start_by: usize,
        end_by: usize,
        start_bx: usize,
        end_bx: usize,
        result: &mut GroupTransformResult,
    ) {
        let yoff = start_by;
        let xoff = start_bx;
        let width = result.width;

        let quant_dc = &mut result.quant_dc;
        let quant_ac = &mut result.quant_ac;
        let nzeros = &mut result.nzeros;
        let raw_nzeros = &mut result.raw_nzeros;
        let float_dc = &mut result.float_dc;

        let channels = [xyb_x, xyb_y, xyb_b];

        // Hoist constant computations out of the block loop
        let x_qm_mul = jxl_simd::fast_powf(1.25, params.x_qm_scale as f32 - 2.0);
        let b_qm_mul = jxl_simd::fast_powf(1.25, params.b_qm_scale as f32 - 2.0);

        // Pre-allocate scratch buffers for DCT coefficients (max DCT64x64 = 4096)
        const MAX_BLOCK_SIZE: usize = 4096;
        let mut dct_scratch: [Vec<f32>; 3] =
            core::array::from_fn(|_| jxl_simd::vec_f32_dirty(MAX_BLOCK_SIZE));

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
        // Max flat_nz size: for DCT64x64, covered = 8×8, flat_len = 7*width+8
        let mut nz_flat_scratch = vec![0u8; 7 * width + 8];

        for by in start_by..end_by {
            for bx in start_bx..end_bx {
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
                // W44-AUDIT-8 Phase 5: multiply inv_factor by
                // `dc_mul = 1 << extra_dc_precision` (= 1 at e>=8, 2 at e<=7)
                // to match libjxl's `nl_dc` gate (enc_modular.cc:1580
                // `mul = 1 << extra_dc_precision`). Symmetric with
                // `reconstruct.rs` so encoder-side buttloop dequant cancels.
                {
                    let dc_mul = (1u32 << params.extra_dc_precision) as f32;
                    let inv_factor = INV_DC_QUANT[1] * params.scale_dc * dc_mul;
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
                            float_dc[1][(by - yoff) * width + (bx - xoff)] = dct_coeffs[1][0];
                            quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                (dct_coeffs[1][0] * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT16X8 => {
                            let dcs = dc_from_dct_16x8(as_array_ref::<128>(&dct_coeffs[1], 0));
                            for iy in 0..2 {
                                float_dc[1][(by - yoff + iy) * width + (bx - xoff)] = dcs[iy];
                                quant_dc[1][(by - yoff + iy) * width + (bx - xoff)] =
                                    (dcs[iy] * inv_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT8X16 => {
                            let dcs = dc_from_dct_8x16(as_array_ref::<128>(&dct_coeffs[1], 0));
                            for ix in 0..2 {
                                float_dc[1][(by - yoff) * width + (bx - xoff + ix)] = dcs[ix];
                                quant_dc[1][(by - yoff) * width + (bx - xoff + ix)] =
                                    (dcs[ix] * inv_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT16X16 => {
                            let dcs = dc_from_dct_16x16(as_array_ref::<256>(&dct_coeffs[1], 0));
                            #[cfg(feature = "debug-dc")]
                            eprintln!(
                                "DCT16x16 block (by={}, bx={}): dcs=[{:.4}, {:.4}, {:.4}, {:.4}], LLF=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                by,
                                bx,
                                dcs[0],
                                dcs[1],
                                dcs[2],
                                dcs[3],
                                dct_coeffs[1][0],
                                dct_coeffs[1][1],
                                dct_coeffs[1][16],
                                dct_coeffs[1][17]
                            );
                            for iy in 0..2 {
                                for ix in 0..2 {
                                    float_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] =
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
                                    quant_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] = qdc;
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
                                    float_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] =
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
                                    quant_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X16 => {
                            // DCT32X16: 4 block rows × 2 block cols, returns 8 DC values in 4×2 order
                            let dcs = dc_from_dct_32x16(as_array_ref::<512>(&dct_coeffs[1], 0));
                            for iy in 0..4 {
                                for ix in 0..2 {
                                    float_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 2 + ix];
                                    let qdc = (dcs[iy * 2 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT16X32 => {
                            // DCT16X32: 2×4 blocks, returns 8 DC values in row-major 2x4
                            let dcs = dc_from_dct_16x32(as_array_ref::<512>(&dct_coeffs[1], 0));
                            for iy in 0..2 {
                                for ix in 0..4 {
                                    float_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 4 + ix];
                                    let qdc = (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X64 => {
                            // DCT64X64: 8×8 blocks, returns 64 DC values in row-major 8x8
                            let dcs = dc_from_dct_64x64(&dct_coeffs[1]);
                            for iy in 0..8 {
                                for ix in 0..8 {
                                    float_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 8 + ix];
                                    let qdc = (dcs[iy * 8 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X32 => {
                            // DCT64X32: 8 block rows × 4 block cols, returns 32 DC values in 8×4 order
                            let dcs = dc_from_dct_64x32(&dct_coeffs[1]);
                            for iy in 0..8 {
                                for ix in 0..4 {
                                    float_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 4 + ix];
                                    let qdc = (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X64 => {
                            // DCT32X64: 4×8 blocks, returns 32 DC values in row-major 4x8
                            let dcs = dc_from_dct_32x64(&dct_coeffs[1]);
                            for iy in 0..4 {
                                for ix in 0..8 {
                                    float_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 8 + ix];
                                    let qdc = (dcs[iy * 8 + ix] * inv_factor).round() as i16;
                                    quant_dc[1][(by - yoff + iy) * width + (bx - xoff + ix)] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT4X8 => {
                            let dc = dc_from_dct_4x8_full(as_array_ref::<64>(&dct_coeffs[1], 0));
                            float_dc[1][(by - yoff) * width + (bx - xoff)] = dc;
                            quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT8X4 => {
                            let dc = dc_from_dct_8x4_full(as_array_ref::<64>(&dct_coeffs[1], 0));
                            float_dc[1][(by - yoff) * width + (bx - xoff)] = dc;
                            quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT4X4 => {
                            let dc = dc_from_dct_4x4_full(as_array_ref::<64>(&dct_coeffs[1], 0));
                            float_dc[1][(by - yoff) * width + (bx - xoff)] = dc;
                            quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                        | RAW_STRATEGY_AFV3 => {
                            let dc = dc_from_afv(as_array_ref::<64>(&dct_coeffs[1], 0));
                            float_dc[1][(by - yoff) * width + (bx - xoff)] = dc;
                            quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_IDENTITY | RAW_STRATEGY_DCT2X2 => {
                            // IDENTITY/DCT2X2: 1×1 coverage, DC at position [0]
                            float_dc[1][(by - yoff) * width + (bx - xoff)] = dct_coeffs[1][0];
                            quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                (dct_coeffs[1][0] * inv_factor).round() as i16;
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
                            #[cfg(feature = "investigate-adjust-quant-block-ac")]
                            let _orig_quant_c = quant_c;
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
                            #[cfg(feature = "investigate-adjust-quant-block-ac")]
                            super::aqba_diag::record(
                                raw_strategy,
                                c,
                                hflags,
                                _orig_quant_c,
                                quant_c,
                            );
                            // SA-A (2026-05-24): per-block dump when
                            // JXL_AQBA_PERBLOCK_TSV env var is set.
                            #[cfg(feature = "investigate-adjust-quant-block-ac")]
                            super::aqba_diag::record_perblock(
                                bx as u32,
                                by as u32,
                                c as u8,
                                raw_strategy,
                                _orig_quant_c,
                                quant_c,
                                hflags,
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
                        quant_int = max_quant;
                        let new_quant = quant_int.clamp(1, 255) as u8;
                        result.quant_adjustments.push((quant_idx, new_quant));
                        debug_rect!(
                            "quant/adjust",
                            bx * 8,
                            by * 8,
                            cx * 8,
                            cy * 8,
                            "strat={} q={}→{} (e>=5 AdjustQuantBlockAC)",
                            raw_strategy,
                            quant_field[quant_idx],
                            new_quant
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
                        bx - xoff,
                        by - yoff,
                        &mut quant_ac[c],
                        width,
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
                    let transpose_slots = covered_y > covered_x;
                    // Use post-swap dimensions for grid (matches C++ and quantize_ac_block).
                    // Nested loops eliminate per-element integer divisions.
                    // Pre-slice weights and dct_coeffs rows to eliminate inner bounds checks.
                    for coef_slot_y in 0..cy {
                        for pos_y in 0..BLOCK_DIM {
                            let y = coef_slot_y * BLOCK_DIM + pos_y;
                            let is_llf_row = y < cy;
                            let row_off = y * block_width;
                            let w_row = &weights[row_off..row_off + block_width];
                            let coeff_row = &mut dct_coeffs[1][row_off..row_off + block_width];
                            for coef_slot_x in 0..cx {
                                for pos_x in 0..BLOCK_DIM {
                                    let x = coef_slot_x * BLOCK_DIM + pos_x;
                                    let is_llf = is_llf_row && x < cx;
                                    let q = if is_llf {
                                        Self::quantize_coeff_ac(
                                            coeff_row[x],
                                            1.0 / w_row[x],
                                            qac,
                                            1.0,
                                            &thresholds_y,
                                            y,
                                            x,
                                            block_height,
                                            block_width,
                                        )
                                    } else {
                                        let (phys_row_off, phys_col_off) = if transpose_slots {
                                            (coef_slot_x, coef_slot_y)
                                        } else {
                                            (coef_slot_y, coef_slot_x)
                                        };
                                        let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                                        quant_ac[1][(by - yoff + phys_row_off) * width
                                            + (bx - xoff + phys_col_off)][pos_in_8x8]
                                    };
                                    let adj = adjust_quant_bias(q, 1);
                                    coeff_row[x] = adj * w_row[x] * inv_qac;
                                }
                            }
                        }
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
                // Nested loops eliminate per-element div/mod; split_at_mut for disjoint refs;
                // pre-slice rows to eliminate inner bounds checks.
                {
                    let (dc_x, rest) = dct_coeffs.split_at_mut(1);
                    let (dc_y, dc_b) = rest.split_at_mut(1);
                    for y in 0..block_height {
                        let x_start = if y < cy { cx } else { 0 };
                        let row_off = y * block_width;
                        let yr = &dc_y[0][row_off..row_off + block_width];
                        let xr = &mut dc_x[0][row_off..row_off + block_width];
                        let br = &mut dc_b[0][row_off..row_off + block_width];
                        for x in x_start..block_width {
                            xr[x] -= x_factor * yr[x];
                            br[x] -= b_factor * yr[x];
                        }
                    }
                }

                // ── Step 7: Extract X/B DC + quantize X/B AC ───────────────
                // W44-AUDIT-8 Phase 5: multiply inv_factor by
                // `dc_mul = 1 << extra_dc_precision` to match libjxl
                // `nl_dc` gate (see Step 2 comment for full ref).
                let dc_mul = (1u32 << params.extra_dc_precision) as f32;
                for &c in &[0usize, 2] {
                    let dc_cfl_factor = if c == 2 { 0.5f32 } else { 0.0f32 };
                    let inv_factor = INV_DC_QUANT[c] * params.scale_dc * dc_mul;
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
                            float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                            let y_dc = quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                            quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            // W44-181 read-only probe (DCT8 raw_strategy=0).
                            #[cfg(feature = "std")]
                            super::w44_181_dump::dump_dc(
                                bx,
                                by,
                                c,
                                0,
                                dc,
                                y_dc,
                                inv_factor,
                                dc_cfl_factor,
                            );
                        }
                        RAW_STRATEGY_DCT16X8 => {
                            let dcs = dc_from_dct_16x8(as_array_ref::<128>(&dct_coeffs[c], 0));
                            for iy in 0..2 {
                                float_dc[c][(by - yoff + iy) * width + (bx - xoff)] = dcs[iy];
                                let y_dc =
                                    quant_dc[1][(by - yoff + iy) * width + (bx - xoff)] as f32;
                                quant_dc[c][(by - yoff + iy) * width + (bx - xoff)] =
                                    (dcs[iy] * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT8X16 => {
                            let dcs = dc_from_dct_8x16(as_array_ref::<128>(&dct_coeffs[c], 0));
                            for ix in 0..2 {
                                float_dc[c][(by - yoff) * width + (bx - xoff + ix)] = dcs[ix];
                                let y_dc =
                                    quant_dc[1][(by - yoff) * width + (bx - xoff + ix)] as f32;
                                quant_dc[c][(by - yoff) * width + (bx - xoff + ix)] =
                                    (dcs[ix] * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT16X16 => {
                            let dcs = dc_from_dct_16x16(as_array_ref::<256>(&dct_coeffs[c], 0));
                            for iy in 0..2 {
                                for ix in 0..2 {
                                    float_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 2 + ix];
                                    let y_dc = quant_dc[1]
                                        [(by - yoff + iy) * width + (bx - xoff + ix)]
                                        as f32;
                                    quant_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        (dcs[iy * 2 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X32 => {
                            let dcs = dc_from_dct_32x32(as_array_ref::<1024>(&dct_coeffs[c], 0));
                            for iy in 0..4 {
                                for ix in 0..4 {
                                    float_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 4 + ix];
                                    let y_dc = quant_dc[1]
                                        [(by - yoff + iy) * width + (bx - xoff + ix)]
                                        as f32;
                                    quant_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                    // W44-181 read-only probe.
                                    #[cfg(feature = "std")]
                                    super::w44_181_dump::dump_dc(
                                        bx + ix,
                                        by + iy,
                                        c,
                                        RAW_STRATEGY_DCT32X32,
                                        dcs[iy * 4 + ix],
                                        y_dc,
                                        inv_factor,
                                        dc_cfl_factor,
                                    );
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X16 => {
                            let dcs = dc_from_dct_32x16(as_array_ref::<512>(&dct_coeffs[c], 0));
                            for iy in 0..4 {
                                for ix in 0..2 {
                                    float_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 2 + ix];
                                    let y_dc = quant_dc[1]
                                        [(by - yoff + iy) * width + (bx - xoff + ix)]
                                        as f32;
                                    quant_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        (dcs[iy * 2 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT16X32 => {
                            let dcs = dc_from_dct_16x32(as_array_ref::<512>(&dct_coeffs[c], 0));
                            for iy in 0..2 {
                                for ix in 0..4 {
                                    float_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 4 + ix];
                                    let y_dc = quant_dc[1]
                                        [(by - yoff + iy) * width + (bx - xoff + ix)]
                                        as f32;
                                    quant_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X64 => {
                            let dcs = dc_from_dct_64x64(&dct_coeffs[c]);
                            for iy in 0..8 {
                                for ix in 0..8 {
                                    float_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 8 + ix];
                                    let y_dc = quant_dc[1]
                                        [(by - yoff + iy) * width + (bx - xoff + ix)]
                                        as f32;
                                    quant_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        (dcs[iy * 8 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                    // W44-181 read-only probe: dump DC quant inputs to
                                    // measure f32 evaluation-order divergence vs libjxl.
                                    // Zero overhead when JXL_W44_181_DUMP_DC env unset.
                                    #[cfg(feature = "std")]
                                    super::w44_181_dump::dump_dc(
                                        bx + ix,
                                        by + iy,
                                        c,
                                        RAW_STRATEGY_DCT64X64,
                                        dcs[iy * 8 + ix],
                                        y_dc,
                                        inv_factor,
                                        dc_cfl_factor,
                                    );
                                }
                            }
                        }
                        RAW_STRATEGY_DCT64X32 => {
                            let dcs = dc_from_dct_64x32(&dct_coeffs[c]);
                            for iy in 0..8 {
                                for ix in 0..4 {
                                    float_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 4 + ix];
                                    let y_dc = quant_dc[1]
                                        [(by - yoff + iy) * width + (bx - xoff + ix)]
                                        as f32;
                                    quant_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X64 => {
                            let dcs = dc_from_dct_32x64(&dct_coeffs[c]);
                            for iy in 0..4 {
                                for ix in 0..8 {
                                    float_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        dcs[iy * 8 + ix];
                                    let y_dc = quant_dc[1]
                                        [(by - yoff + iy) * width + (bx - xoff + ix)]
                                        as f32;
                                    quant_dc[c][(by - yoff + iy) * width + (bx - xoff + ix)] =
                                        (dcs[iy * 8 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT4X8 => {
                            let dc = dc_from_dct_4x8_full(as_array_ref::<64>(&dct_coeffs[c], 0));
                            float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                            let y_dc = quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                            quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT8X4 => {
                            let dc = dc_from_dct_8x4_full(as_array_ref::<64>(&dct_coeffs[c], 0));
                            float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                            let y_dc = quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                            quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT4X4 => {
                            let dc = dc_from_dct_4x4_full(as_array_ref::<64>(&dct_coeffs[c], 0));
                            float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                            let y_dc = quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                            quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                        | RAW_STRATEGY_AFV3 => {
                            let dc = dc_from_afv(as_array_ref::<64>(&dct_coeffs[c], 0));
                            float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                            let y_dc = quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                            quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_IDENTITY | RAW_STRATEGY_DCT2X2 => {
                            // IDENTITY/DCT2X2: 1×1 coverage, DC at position [0]
                            let dc = dct_coeffs[c][0];
                            float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                            let y_dc = quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                            quant_dc[c][(by - yoff) * width + (bx - xoff)] =
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
                        bx - xoff,
                        by - yoff,
                        &mut quant_ac[c],
                        width,
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

                // W44-AUDIT-8 Phase 4 read-only DC probe (env-gated).
                // After Step 7 completes the DC has been populated in
                // float_dc + quant_dc for ALL channels and ALL strategies via
                // the per-strategy `dc_from_dct_*` calls; the iteration shape
                // (covered_y × covered_x) matches every match-arm above.
                // Coordinates are absolute block indices (after rect origin).
                // Zero overhead when JXL_W44_AUDIT_8_P4_DUMP env unset.
                #[cfg(feature = "std")]
                for iy in 0..covered_y {
                    for ix in 0..covered_x {
                        let dc_idx = (by - yoff + iy) * width + (bx - xoff + ix);
                        for c in 0..3 {
                            super::w44_audit_8_p4_dump::dump_dc(
                                bx + ix,
                                by + iy,
                                c,
                                raw_strategy,
                                float_dc[c][dc_idx],
                                quant_dc[c][dc_idx],
                            );
                        }
                    }
                }

                // ── Step 8: Count non-zeros for all 3 channels ─────────────
                let transpose_slots = covered_y > covered_x;
                for c in 0..3 {
                    if covered_blocks == 1 {
                        num_nonzero_8x8_except_dc(
                            &quant_ac[c][(by - yoff) * width + (bx - xoff)],
                            &mut nzeros[c][(by - yoff) * width + (bx - xoff)],
                        );
                        raw_nzeros[c][(by - yoff) * width + (bx - xoff)] =
                            nzeros[c][(by - yoff) * width + (bx - xoff)] as u16;
                    } else {
                        // Build flat block in cx*8 × cy*8 layout (stride = cx*8).
                        // num_nonzero_except_llf expects block[y * stride + x] for y,x in 0..cy*8, 0..cx*8.
                        // The 8x8 block storage uses quant_ac[slot_by][slot_bx][pos_in_8x8].
                        let stride = cx * BLOCK_DIM;
                        let full_block = &mut nz_full_block_scratch[..size];
                        // Nested loops eliminate per-element integer divisions.
                        // Pre-slice full_block rows to eliminate inner bounds checks.
                        for coef_slot_y in 0..cy {
                            for pos_y in 0..BLOCK_DIM {
                                let y = coef_slot_y * BLOCK_DIM + pos_y;
                                let fb_row = &mut full_block[y * stride..y * stride + stride];
                                for coef_slot_x in 0..cx {
                                    let (phys_row_off, phys_col_off) = if transpose_slots {
                                        (coef_slot_x, coef_slot_y)
                                    } else {
                                        (coef_slot_y, coef_slot_x)
                                    };
                                    let row = &quant_ac[c][(by - yoff + phys_row_off) * width
                                        + (bx - xoff + phys_col_off)];
                                    for pos_x in 0..BLOCK_DIM {
                                        let x = coef_slot_x * BLOCK_DIM + pos_x;
                                        fb_row[x] = row[pos_y * BLOCK_DIM + pos_x];
                                    }
                                }
                            }
                        }
                        let flat_len = (covered_y - 1) * width + covered_x;
                        let flat_nz = &mut nz_flat_scratch[..flat_len];
                        flat_nz.fill(0);
                        let raw_nz = num_nonzero_except_llf(
                            cx, cy, full_block, width, flat_nz, covered_x, covered_y,
                        );
                        for dy in 0..covered_y {
                            for dx in 0..covered_x {
                                nzeros[c][(by - yoff + dy) * width + (bx - xoff + dx)] =
                                    flat_nz[dx + dy * width];
                            }
                        }
                        raw_nzeros[c][(by - yoff) * width + (bx - xoff)] = raw_nz;
                    }
                }
            }
        }
    }

    /// Perform DCT and quantization on all blocks (parallel over groups).
    ///
    /// Supports all AC strategies. For multi-block transforms, only first blocks
    /// are processed; sub-block slots store their portion of the coefficients.
    ///
    /// Processing order per block matches C++ WriteACGroup:
    /// 1. DCT Y → extract Y DC → quantize Y AC (with thresholding)
    /// 2. Dequantize Y AC back (AdjustQuantBias) → roundtripped Y
    /// 3. DCT X, B → apply CfL using roundtripped Y → extract X/B DC
    /// 4. Quantize X/B AC (with thresholding + x_qm_mul for X)
    ///
    /// Groups are processed in parallel; results are scattered into the output.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    /// Diagnostic-only wrapper around `transform_and_quantize` that
    /// returns the raw `TransformOutput`. Used by the
    /// `__pre_quantized` test path to feed CPU-produced
    /// transform output into `encode_from_pre_quantized_ac` and
    /// isolate "is the entry point itself correct?" from "is the
    /// GPU producer correct?".
    #[cfg(feature = "__pre_quantized")]
    pub fn transform_and_quantize_for_test(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &mut [u8],
        params: &DistanceParams,
    ) -> Result<TransformOutput> {
        self.transform_and_quantize(
            &precomputed.xyb_x,
            &precomputed.xyb_y,
            &precomputed.xyb_b,
            precomputed.padded_width,
            precomputed.xsize_blocks,
            precomputed.ysize_blocks,
            params,
            quant_field,
            &precomputed.cfl_map,
            &precomputed.ac_strategy,
        )
    }

    /// Pull-style variant of [`Self::transform_and_quantize`] that
    /// reads XYB data through a [`super::region_source::XybRegionSource`].
    ///
    /// **Streaming refactor chunk 8b (#11)**: this is the seam that
    /// lets the encoder pull XYB data per-region instead of holding
    /// three whole-image plane borrows for the lifetime of the call.
    /// Chunk-8b ships only the *seam* — the body still calls
    /// `xyb_full()` once and delegates to the existing whole-image
    /// implementation, so every byte produced is identical to the
    /// pre-refactor path (verified by `hash_lock_features.rs` 36/36).
    ///
    /// The trait abstraction unlocks two future wins:
    /// 1. **Per-DC-group source materialisation (chunk 8c)** — a
    ///    streaming source can materialise one DC group's region at
    ///    a time, dropping it before the next region is pulled.
    /// 2. **Region-aware AC-group fan-out (chunk 8c)** — the per-tile
    ///    `transform_blocks_into` loop body can be lifted out into a
    ///    per-DC-group orchestrator that pulls + releases per region.
    ///
    /// See [`super::region_source`] module docs for the chunk-8b
    /// scope and the list of remaining whole-image consumers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transform_and_quantize_with_source(
        &self,
        source: &dyn super::region_source::XybRegionSource,
        xsize_blocks: usize,
        ysize_blocks: usize,
        params: &DistanceParams,
        quant_field: &mut [u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
    ) -> Result<TransformOutput> {
        let padded_width = source.padded_width();
        let (xyb_x, xyb_y, xyb_b) = source.xyb_full();
        let out = self.transform_and_quantize(
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
        )?;
        // W44-112 Layer-1.5 hook: capture the post-production quant_field
        // (after AdjustQuantBlockAC second-application) + the `DistanceParams`
        // used by production. Compared against the buttloop's internal
        // post-final-iter `quant_field` to discriminate the W44-111 candidates.
        // Zero cost when feature off; cheap atomic load + early-exit when on
        // but capture disabled.
        #[cfg(feature = "__internal_recon_hook")]
        if super::butteraugli_loop::recon_hook::production_qf_capture_enabled() {
            super::butteraugli_loop::recon_hook::store_production_qf(
                super::butteraugli_loop::recon_hook::ProductionQf {
                    xsize_blocks,
                    ysize_blocks,
                    quant_field_u8: quant_field.to_vec(),
                    global_scale: params.global_scale,
                    scale: params.scale,
                    inv_scale: params.inv_scale,
                },
            );
        }
        Ok(out)
    }

    // Internal hot-path entry: factoring these into a struct
    // would force per-call packing/unpacking on the per-group
    // parallel reduce. All call sites are within this crate; the
    // signature is exercised by butteraugli/ssim2/zensim loops.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<TransformOutput> {
        let mut out = TransformOutput::new(xsize_blocks, ysize_blocks, self.budget.as_ref())?;

        let xsize_groups = div_ceil(xsize_blocks, GROUP_DIM_IN_BLOCKS);
        let ysize_groups = div_ceil(ysize_blocks, GROUP_DIM_IN_BLOCKS);
        let num_groups = xsize_groups * ysize_groups;

        // W44-89 HONEST-STOP: a parallel-vs-serial guard was investigated
        // here per the W44-88 follow-on task. Paired interleaved A/B
        // benchmark (5 trials × 12 cells, baseline-binary vs guard-binary,
        // `benchmarks/parallel_xform_guard_AB_2026-05-19.tsv`) showed the
        // W44-88 "terminal e3 regression" is NOT reproducible on stable
        // measurements — 8T parallel reliably beats 1T serial by 1.62-1.72×
        // even at 35 AC groups (terminal e3 8T median 21.8 ms vs 1T 37.6 ms
        // across 5 interleaved trials). W44-88's high-variance non-paired
        // measurement was likely contaminated by cold rayon thread-pool
        // wake-up or thermal noise on trial 1. Forcing serial at any group
        // count INTRODUCED a +16 ms regression on terminal e3. Guard NOT
        // shipped. The helper `parallel_map_min` was added to `parallel.rs`
        // for future use if a real regression appears.
        let group_results = crate::parallel::parallel_map(num_groups, |group_idx| {
            let gy = group_idx / xsize_groups;
            let gx = group_idx % xsize_groups;
            let start_bx = gx * GROUP_DIM_IN_BLOCKS;
            let start_by = gy * GROUP_DIM_IN_BLOCKS;
            let end_bx = (start_bx + GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
            let end_by = (start_by + GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
            let width = end_bx - start_bx;
            let height = end_by - start_by;

            let mut result = GroupTransformResult::new(start_bx, start_by, width, height);
            self.transform_blocks_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                params,
                quant_field,
                cfl_map,
                ac_strategy,
                start_by,
                end_by,
                start_bx,
                end_bx,
                &mut result,
            );
            // W44-27: flush worker thread's AdjustQuantBlockAC firing
            // counts to the global aggregate before this rayon task
            // returns. Without this, the main-thread emit_and_reset
            // sees only the main thread's contributions.
            #[cfg(feature = "investigate-adjust-quant-block-ac")]
            super::aqba_diag::flush_tl_to_global();
            // SA-A (2026-05-24): also flush per-block dump TLS to global.
            #[cfg(feature = "investigate-adjust-quant-block-ac")]
            super::aqba_diag::flush_tl_perblock_public();
            result
        });

        for result in group_results {
            for &(idx, val) in &result.quant_adjustments {
                quant_field[idx] = val;
            }
            result.scatter_into(&mut out, xsize_blocks);
        }

        Ok(out)
    }

    /// Fill pre-allocated `TransformOutput` buffers (parallel across groups,
    /// for buttloop / ssim2-loop / zensim-loop iterations).
    ///
    /// Processes groups via `transform_blocks_into` and scatters results into
    /// the pre-allocated `out`. Used by every per-iter quantization loop that
    /// reuses the same `TransformOutput` across iterations.
    ///
    /// # W44-175: per-group parallelism (rayon)
    ///
    /// Pre-W44-175 this loop was a sequential `for gy { for gx { … } }`. The
    /// sibling [`Self::transform_and_quantize`] (called once per encode for the
    /// AC-strategy-search pass) ALREADY ran parallel across groups via
    /// [`crate::parallel::parallel_map`] (W44-89, `e0178b550c38`), but the
    /// `_into` variant — invoked once per buttloop iter (typically 2-4× per
    /// encode at e8+) — was left sequential. On large screenshots that ate
    /// 50-200 ms of buttloop wall per encode.
    ///
    /// ## Why the parallelization is safe
    ///
    /// `transform_blocks_into` takes `quant_field` by `&[u8]` (read-only) and
    /// writes its per-block quant adjustments into
    /// `GroupTransformResult.quant_adjustments`. The "Apply quant adjustments
    /// immediately so later groups see them" comment in the pre-W44-175 code
    /// was misleading: groups process DISJOINT block ranges (a group covers
    /// `[start_by..end_by, start_bx..end_bx]`), so the `quant_field[idx]`
    /// reads in one group cannot observe writes from another. Sequencing the
    /// writes within a single iter changes nothing about the per-block
    /// output. The merge phase (after the parallel fan-out) applies every
    /// group's adjustments to `quant_field` before the NEXT buttloop iter
    /// reads it — same observation order as the sequential code.
    ///
    /// ## Output ordering
    ///
    /// [`crate::parallel::parallel_map`] uses `into_par_iter().map(f).collect()`,
    /// which preserves index order in the returned `Vec`. We then iterate
    /// the results in order to merge into `out` and `quant_field`, so the
    /// final state is byte-identical to the sequential version.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn transform_and_quantize_into(
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
        out: &mut TransformOutput,
    ) {
        let xsize_groups = div_ceil(xsize_blocks, GROUP_DIM_IN_BLOCKS);
        let ysize_groups = div_ceil(ysize_blocks, GROUP_DIM_IN_BLOCKS);
        let num_groups = xsize_groups * ysize_groups;

        // W44-175: parallel per-group fan-out. Mirrors `transform_and_quantize`
        // (the AC-strategy-search variant). `quant_field` is captured as
        // `&[u8]` inside the closure — `transform_blocks_into` only reads it;
        // writes are recorded into `GroupTransformResult.quant_adjustments`
        // and applied serially after collection so the merge order is
        // deterministic.
        let quant_field_ro: &[u8] = quant_field;
        let group_results = crate::parallel::parallel_map(num_groups, |group_idx| {
            let gy = group_idx / xsize_groups;
            let gx = group_idx % xsize_groups;
            let start_bx = gx * GROUP_DIM_IN_BLOCKS;
            let start_by = gy * GROUP_DIM_IN_BLOCKS;
            let end_bx = (start_bx + GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
            let end_by = (start_by + GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
            let width = end_bx - start_bx;
            let height = end_by - start_by;

            let mut result = GroupTransformResult::new(start_bx, start_by, width, height);
            self.transform_blocks_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                params,
                quant_field_ro,
                cfl_map,
                ac_strategy,
                start_by,
                end_by,
                start_bx,
                end_bx,
                &mut result,
            );
            // W44-27 parity: flush worker thread's AdjustQuantBlockAC firing
            // counts to the global aggregate before this rayon task returns,
            // matching the sibling `transform_and_quantize` invariant. Without
            // this, the main-thread emit_and_reset would see only the main
            // thread's contributions on rayon-parallel callers.
            #[cfg(feature = "investigate-adjust-quant-block-ac")]
            super::aqba_diag::flush_tl_to_global();
            // SA-A (2026-05-24): also flush per-block dump TLS to global.
            #[cfg(feature = "investigate-adjust-quant-block-ac")]
            super::aqba_diag::flush_tl_perblock_public();
            result
        });

        // Merge phase: apply quant_adjustments serially (in group order) and
        // scatter each group's output into `out`. This preserves the exact
        // post-merge state of `quant_field` and `out` from the sequential
        // version — see the function doc-comment for the disjoint-indices
        // argument.
        for result in group_results {
            for &(idx, val) in &result.quant_adjustments {
                quant_field[idx] = val;
            }
            result.scatter_into(out, xsize_blocks);
        }
    }
}

#[cfg(test)]
mod budget_tests {
    use super::TransformOutput;
    use crate::budget::MemoryBudget;

    // Per-channel bytes formula mirrored from `TransformOutput::new`.
    fn one_transform_output_bytes(xb: usize, yb: usize) -> u64 {
        (xb as u64) * (yb as u64) * (265 * 3)
    }

    /// The perceptual loop (`perceptual_loop.rs` / `ssim2_loop.rs` /
    /// `zensim_loop.rs`) allocates a scratch `TransformOutput` that is
    /// dropped before the base-encode `TransformOutput` is built — the two
    /// never coexist in real memory. Before the RAII fix, `new` reserved
    /// the ~149 MB (at 12 MP) **permanently**, so the second
    /// `TransformOutput` was charged against a budget the first one's
    /// already-freed bytes still occupied — a chunk of the 12 MP HDR
    /// 2 GiB-cap overrun. This pins the release-on-drop contract: a budget
    /// sized for one-and-a-half `TransformOutput`s must admit a second one
    /// once the first drops.
    #[test]
    fn transform_output_reservation_released_on_drop() {
        let (xb, yb) = (64usize, 64usize); // 4096 blocks
        let one = one_transform_output_bytes(xb, yb);
        let budget = MemoryBudget::new(one + one / 2); // fits one, not two
        {
            let _t1 = TransformOutput::new(xb, yb, Some(&budget)).expect("first fits");
            assert!(budget.used() >= one, "first reservation must be charged");
        }
        // _t1 dropped: its RAII guard releases the reservation.
        assert_eq!(budget.used(), 0, "drop must release the reservation");
        let _t2 = TransformOutput::new(xb, yb, Some(&budget))
            .expect("second TransformOutput must fit after the first is dropped");
    }

    /// Inverse guard: two *simultaneously live* `TransformOutput`s still
    /// exceed a 1.5× cap — proves the guard tracks real peak (the fix did
    /// not simply stop charging the reservation).
    #[test]
    fn transform_output_two_live_still_exceed_cap() {
        let (xb, yb) = (64usize, 64usize);
        let one = one_transform_output_bytes(xb, yb);
        let budget = MemoryBudget::new(one + one / 2);
        let _t1 = TransformOutput::new(xb, yb, Some(&budget)).expect("first fits");
        let r2 = TransformOutput::new(xb, yb, Some(&budget));
        assert!(
            r2.is_err(),
            "two concurrently-live TransformOutputs must exceed a 1.5x cap"
        );
    }
}
