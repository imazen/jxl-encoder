// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

// archmage::arcane macro expansion doesn't propagate function-level allows.
#![allow(clippy::too_many_arguments)]

//! Encoder-side reconstruction pipeline.
//!
//! Simulates what the decoder produces from quantized coefficients, enabling:
//! - EPF sharpness selection (compare reconstruction vs original)
//! - Butteraugli quantization loop (iterative quality refinement)
//!
//! The pipeline: dequantize -> CfL restore -> LLF from DC -> IDCT -> [gab smooth] -> [EPF]

use super::ac_strategy::{
    AcStrategyMap, RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
    RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT8,
    RAW_STRATEGY_DCT8X4, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
    RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
    RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
};
use super::chroma_from_luma::{CflMap, ytob_ratio, ytox_ratio};
use super::common::*;
use super::dct::*;
use super::frame::DistanceParams;
use super::quant::{INV_DC_QUANT, quant_weights};
use super::quantize::adjust_quant_bias;

/// Reconstruct XYB pixel planes from quantized coefficients.
///
/// This simulates the decoder's output BEFORE gaborish smooth and EPF.
/// Returns `(xyb_x, xyb_y, xyb_b)` as flat arrays of size `padded_width * padded_height`.
///
/// # Arguments
/// * `quant_dc` - Quantized DC per channel `[Vec<Vec<i16>>; 3]`
/// * `quant_ac` - Quantized AC per channel `[Vec<Vec<[i32; 64]>>; 3]`
/// * `params` - Distance parameters (scale, qm_scale, etc.)
/// * `quant_field` - Per-block raw quantization values (u8)
/// * `cfl_map` - Chroma-from-luma tile map
/// * `ac_strategy` - Per-block AC strategy map
/// * `xsize_blocks` - Image width in 8x8 blocks
/// * `ysize_blocks` - Image height in 8x8 blocks
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_xyb(
    quant_dc: &[Vec<Vec<i16>>; 3],
    quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    params: &DistanceParams,
    quant_field: &[u8],
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> [Vec<f32>; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::X64V3Token::summon() {
            return reconstruct_xyb_avx2(
                token,
                quant_dc,
                quant_ac,
                params,
                quant_field,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::NeonToken::summon() {
            return reconstruct_xyb_neon(
                token,
                quant_dc,
                quant_ac,
                params,
                quant_field,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use jxl_simd::SimdToken;
        if let Some(token) = jxl_simd::Wasm128Token::summon() {
            return reconstruct_xyb_wasm128(
                token,
                quant_dc,
                quant_ac,
                params,
                quant_field,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );
        }
    }
    reconstruct_xyb_impl(
        quant_dc,
        quant_ac,
        params,
        quant_field,
        cfl_map,
        ac_strategy,
        xsize_blocks,
        ysize_blocks,
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn reconstruct_xyb_avx2(
    _token: jxl_simd::X64V3Token,
    quant_dc: &[Vec<Vec<i16>>; 3],
    quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    params: &DistanceParams,
    quant_field: &[u8],
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> [Vec<f32>; 3] {
    reconstruct_xyb_impl(
        quant_dc,
        quant_ac,
        params,
        quant_field,
        cfl_map,
        ac_strategy,
        xsize_blocks,
        ysize_blocks,
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn reconstruct_xyb_neon(
    _token: jxl_simd::NeonToken,
    quant_dc: &[Vec<Vec<i16>>; 3],
    quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    params: &DistanceParams,
    quant_field: &[u8],
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> [Vec<f32>; 3] {
    reconstruct_xyb_impl(
        quant_dc,
        quant_ac,
        params,
        quant_field,
        cfl_map,
        ac_strategy,
        xsize_blocks,
        ysize_blocks,
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn reconstruct_xyb_wasm128(
    _token: jxl_simd::Wasm128Token,
    quant_dc: &[Vec<Vec<i16>>; 3],
    quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    params: &DistanceParams,
    quant_field: &[u8],
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> [Vec<f32>; 3] {
    reconstruct_xyb_impl(
        quant_dc,
        quant_ac,
        params,
        quant_field,
        cfl_map,
        ac_strategy,
        xsize_blocks,
        ysize_blocks,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn reconstruct_xyb_impl(
    quant_dc: &[Vec<Vec<i16>>; 3],
    quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    params: &DistanceParams,
    quant_field: &[u8],
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> [Vec<f32>; 3] {
    let padded_width = xsize_blocks * BLOCK_DIM;
    let padded_height = ysize_blocks * BLOCK_DIM;
    let num_pixels = padded_width * padded_height;

    let x_qm_mul = jxl_simd::fast_powf(1.25, params.x_qm_scale as f32 - 2.0);
    let b_qm_mul = jxl_simd::fast_powf(1.25, params.b_qm_scale as f32 - 2.0);

    // Step 1: Dequantize all coefficients into floating-point DCT domain.
    // For each first-block of each transform, reconstruct the full coefficient block.
    // Output: per-channel float coefficient planes in pixel layout after IDCT.
    let mut planes = [
        jxl_simd::vec_f32_dirty(num_pixels),
        jxl_simd::vec_f32_dirty(num_pixels),
        jxl_simd::vec_f32_dirty(num_pixels),
    ];

    // Pre-allocate scratch buffers to avoid per-block heap allocations.
    // Max block size is 4096 (DCT64x64 = 64x64 coefficients).
    const MAX_BLOCK_SIZE: usize = 4096;
    let mut dequant_scratch = [
        jxl_simd::vec_f32_dirty(MAX_BLOCK_SIZE),
        jxl_simd::vec_f32_dirty(MAX_BLOCK_SIZE),
        jxl_simd::vec_f32_dirty(MAX_BLOCK_SIZE),
    ];
    let mut transpose_scratch = jxl_simd::vec_f32_dirty(MAX_BLOCK_SIZE);
    let mut idct_scratch = jxl_simd::vec_f32_dirty(MAX_BLOCK_SIZE);

    // Process all first-blocks
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }

            let raw_strategy = ac_strategy.raw_strategy(bx, by);
            let covered_x = ac_strategy.covered_blocks_x(bx, by);
            let covered_y = ac_strategy.covered_blocks_y(bx, by);
            // Use PHYSICAL coverage for coefficient iteration and pixel output.
            // The IDCT expects coefficients in natural (pre-swap) layout.
            // Match the encoder's coefficient layout: swap cx/cy so cx >= cy.
            // This gives the same stride and block mapping as the encoder's
            // quantize_ac_block. After dequantizing, we transpose back to the
            // IDCT's expected (natural) layout.
            let transpose_slots = covered_y > covered_x;
            let (cx, cy) = if transpose_slots {
                (covered_y, covered_x)
            } else {
                (covered_x, covered_y)
            };
            let block_width = cx * BLOCK_DIM;
            let block_height = cy * BLOCK_DIM;
            let size = block_width * block_height;

            // CfL factors for this tile
            let tx = bx / TILE_DIM_IN_BLOCKS;
            let ty = by / TILE_DIM_IN_BLOCKS;
            let x_factor = ytox_ratio(cfl_map.ytox_at(tx, ty));
            let b_factor = ytob_ratio(cfl_map.ytob_at(tx, ty));

            // DCT8 fast path: SIMD dequant + CfL + IDCT in one optimized pass
            if raw_strategy == RAW_STRATEGY_DCT8 {
                let qac = params.scale * quant_field[by * xsize_blocks + bx] as f32;
                let qac_qm = [qac * x_qm_mul, qac, qac * b_qm_mul];
                let weights_x: &[f32; 64] =
                    as_array_ref(quant_weights(RAW_STRATEGY_DCT8 as usize, 0), 0);
                let weights_y: &[f32; 64] =
                    as_array_ref(quant_weights(RAW_STRATEGY_DCT8 as usize, 1), 0);
                let weights_b: &[f32; 64] =
                    as_array_ref(quant_weights(RAW_STRATEGY_DCT8 as usize, 2), 0);

                let mut dq_x = uninit_buf::<64>();
                let mut dq_y = uninit_buf::<64>();
                let mut dq_b = uninit_buf::<64>();

                jxl_simd::dequant_block_dct8(
                    &quant_ac[0][by][bx],
                    &quant_ac[1][by][bx],
                    &quant_ac[2][by][bx],
                    weights_x,
                    weights_y,
                    weights_b,
                    qac_qm,
                    x_factor,
                    b_factor,
                    &mut dq_x,
                    &mut dq_y,
                    &mut dq_b,
                );

                // Restore LLF (DC) for each channel
                // DC CfL uses dc_cfl_factor (0.5 for B channel, 0 for X/Y) —
                // this is separate from the AC-level CfL (x_factor, b_factor)
                // which is already applied by the SIMD kernel for AC positions.
                // DC is NOT subject to tile-level CfL (the generic path skips LLF
                // positions in the CfL loop).
                let dc_cfl_factor_b: f32 = 0.5;
                // W44-AUDIT-8 Phase 5: symmetric inverse — multiply
                // inv_factor by `dc_mul = 1 << extra_dc_precision` so the
                // encoder-side reconstruction matches the decoder
                // (`mul = 1.0 / dc_mul` applied to stored quant_dc).
                let dc_mul = (1u32 << params.extra_dc_precision) as f32;
                let inv_factor = [
                    INV_DC_QUANT[0] * params.scale_dc * dc_mul,
                    INV_DC_QUANT[1] * params.scale_dc * dc_mul,
                    INV_DC_QUANT[2] * params.scale_dc * dc_mul,
                ];
                let dc_stored = [
                    quant_dc[0][by][bx] as f32,
                    quant_dc[1][by][bx] as f32,
                    quant_dc[2][by][bx] as f32,
                ];

                dq_y[0] = dc_stored[1] / inv_factor[1];
                dq_x[0] = dc_stored[0] / inv_factor[0];
                dq_b[0] = (dc_stored[2] + dc_stored[1] * dc_cfl_factor_b) / inv_factor[2];

                // IDCT + write to output planes
                let pixel_x = bx * BLOCK_DIM;
                let pixel_y = by * BLOCK_DIM;
                let out_base = pixel_y * padded_width + pixel_x;

                let mut px = [0.0f32; 64];
                idct_8x8(&dq_x, &mut px);
                for row in 0..BLOCK_DIM {
                    let src = row * BLOCK_DIM;
                    let dst = out_base + row * padded_width;
                    planes[0][dst..dst + BLOCK_DIM].copy_from_slice(&px[src..src + BLOCK_DIM]);
                }
                idct_8x8(&dq_y, &mut px);
                for row in 0..BLOCK_DIM {
                    let src = row * BLOCK_DIM;
                    let dst = out_base + row * padded_width;
                    planes[1][dst..dst + BLOCK_DIM].copy_from_slice(&px[src..src + BLOCK_DIM]);
                }
                idct_8x8(&dq_b, &mut px);
                for row in 0..BLOCK_DIM {
                    let src = row * BLOCK_DIM;
                    let dst = out_base + row * padded_width;
                    planes[2][dst..dst + BLOCK_DIM].copy_from_slice(&px[src..src + BLOCK_DIM]);
                }
                continue;
            }

            // Generic path for non-DCT8 strategies
            // Zero the scratch regions we'll use
            for ch in &mut dequant_scratch {
                ch[..size].fill(0.0);
            }

            for c in 0..3usize {
                let qm_mul = match c {
                    0 => x_qm_mul,
                    2 => b_qm_mul,
                    _ => 1.0,
                };

                // Dequantize AC coefficients
                let qac = params.scale * quant_field[by * xsize_blocks + bx] as f32;
                let weights = quant_weights(raw_strategy as usize, c);

                // Nested loops eliminate per-element integer divisions.
                // Pre-slice weights and dequant rows to eliminate inner bounds checks.
                let inv_qac_qm = 1.0 / (qac * qm_mul);
                for coef_slot_y in 0..cy {
                    for pos_y in 0..BLOCK_DIM {
                        let y = coef_slot_y * BLOCK_DIM + pos_y;
                        let is_llf_row = y < cy;
                        let row_off = y * block_width;
                        let w_row = &weights[row_off..row_off + block_width];
                        let dq_row = &mut dequant_scratch[c][row_off..row_off + block_width];
                        for coef_slot_x in 0..cx {
                            let (phys_row_off, phys_col_off) = if transpose_slots {
                                (coef_slot_x, coef_slot_y)
                            } else {
                                (coef_slot_y, coef_slot_x)
                            };
                            let row = &quant_ac[c][by + phys_row_off][bx + phys_col_off];
                            for pos_x in 0..BLOCK_DIM {
                                let x = coef_slot_x * BLOCK_DIM + pos_x;
                                if is_llf_row && x < cx {
                                    continue;
                                }
                                let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                                let q_int = row[pos_in_8x8];
                                if q_int != 0 {
                                    let biased = adjust_quant_bias(q_int, c);
                                    dq_row[x] = biased * w_row[x] * inv_qac_qm;
                                }
                            }
                        }
                    }
                }

                // Restore LLF from DC
                restore_llf_from_dc(
                    &mut dequant_scratch[c][..size],
                    &quant_dc[c],
                    &quant_dc[1], // Y channel DC for CfL
                    c,
                    params,
                    raw_strategy,
                    bx,
                    by,
                    cx,
                    cy,
                    block_width,
                );
            }

            // Step 2: Restore CfL (AC positions only, not LLF)
            // The decoder applies: X[k] += x_factor * Y[k], B[k] += b_factor * Y[k]
            // split_at_mut to get disjoint &mut references, then pre-slice rows.
            {
                let (dq_x, rest) = dequant_scratch.split_at_mut(1);
                let (dq_y, dq_b) = rest.split_at_mut(1);
                for y in 0..block_height {
                    let x_start = if y < cy { cx } else { 0 };
                    let row_off = y * block_width;
                    let yr = &dq_y[0][row_off..row_off + block_width];
                    let xr = &mut dq_x[0][row_off..row_off + block_width];
                    let br = &mut dq_b[0][row_off..row_off + block_width];
                    for x in x_start..block_width {
                        xr[x] += x_factor * yr[x];
                        br[x] += b_factor * yr[x];
                    }
                }
            }

            // Step 3: Transpose coefficients if needed, then IDCT.
            // The encoder stores coefficients in post-swap layout (cx >= cy, stride = cx*8).
            //
            // Most IDCT functions expect the post-swap layout directly (matching the
            // forward DCT output). But idct_16x8 uses a gather/scatter implementation
            // that expects the natural pixel layout (16×8, stride 8), so DCT16X8 needs
            // a transpose from the post-swap 8×16 layout.
            //
            // idct_32x16 and idct_64x32 use explicit internal transposes and expect
            // the post-swap layout — no transpose needed for those.
            let needs_transpose = raw_strategy == RAW_STRATEGY_DCT16X8;
            for c in 0..3usize {
                let idct_input = if needs_transpose {
                    // Transpose from post-swap to natural layout using scratch
                    for y in 0..block_height {
                        let src_row =
                            &dequant_scratch[c][y * block_width..y * block_width + block_width];
                        for x in 0..block_width {
                            transpose_scratch[x * block_height + y] = src_row[x];
                        }
                    }
                    &transpose_scratch[..size]
                } else {
                    &dequant_scratch[c][..size]
                };
                idct_for_strategy(raw_strategy, idct_input, &mut idct_scratch[..size]);

                // Write pixels to output plane using physical coverage dimensions
                let pixel_x = bx * BLOCK_DIM;
                let pixel_y = by * BLOCK_DIM;
                let pix_w = covered_x * BLOCK_DIM;
                let pix_h = covered_y * BLOCK_DIM;

                for py in 0..pix_h {
                    let out_start = (pixel_y + py) * padded_width + pixel_x;
                    let in_start = py * pix_w;
                    planes[c][out_start..out_start + pix_w]
                        .copy_from_slice(&idct_scratch[in_start..in_start + pix_w]);
                }
            }
        }
    }

    planes
}

/// Restore LLF coefficients from quantized DC values.
///
/// The decoder's `LowestFrequenciesFromDC` takes the DC grid values,
/// applies a small forward DCT, scales by resample factors, and writes
/// into the LLF positions of the coefficient block.
#[allow(clippy::too_many_arguments)]
fn restore_llf_from_dc(
    coeffs: &mut [f32],
    quant_dc_ch: &[Vec<i16>],
    quant_dc_y: &[Vec<i16>], // Y channel DC for CfL on B channel
    channel: usize,
    params: &DistanceParams,
    raw_strategy: u8,
    bx: usize,
    by: usize,
    _cx: usize,
    _cy: usize,
    _block_width: usize,
) {
    let dc_cfl_factor: f32 = if channel == 2 { 0.5 } else { 0.0 };
    // W44-AUDIT-8 Phase 5: symmetric inverse — see other reconstruct.rs sites.
    let dc_mul = (1u32 << params.extra_dc_precision) as f32;
    let inv_factor = INV_DC_QUANT[channel] * params.scale_dc * dc_mul;

    // Helper: dequantize a DC value with CfL correction
    let dequant_dc = |iy: usize, ix: usize| -> f32 {
        let stored = quant_dc_ch[by + iy][bx + ix] as f32;
        let y_stored = quant_dc_y[by + iy][bx + ix] as f32;
        (stored + y_stored * dc_cfl_factor) / inv_factor
    };

    // Collect DC values and dequantize them
    // DC was stored as: quant_dc[c][by+iy][bx+ix] = (dc * inv_factor - y_dc * dc_cfl_factor).round()
    // Dequant: dc_float = (quant_dc + y_dc * dc_cfl_factor) / inv_factor
    // But we need Y channel DC for CfL restoration on X and B.
    // Actually, the LLF restoration happens BEFORE CfL restore in the decoder.
    // The decoder dequantizes DC → LowestFrequenciesFromDC → DequantBlock → (CfL is implicit in the prediction).
    // Wait - let me re-read the decoder flow more carefully.
    //
    // Decoder flow for each block:
    // 1. Read DC values (already dequantized via DC prediction + inverse quant)
    // 2. LowestFrequenciesFromDC: fill LLF positions from DC grid
    // 3. DequantBlock: multiply each AC coefficient by weight / (qac * qm_mul)
    //    (for non-LLF positions)
    // 4. After dequant, CfL is applied: X += ytox * Y, B += ytob * Y
    //
    // The DC values stored in the bitstream are:
    //   dc_stored = round(dc_float * inv_factor - y_dc_stored * dc_cfl_factor)
    // Where y_dc_stored is the Y channel's stored DC.
    //
    // The decoder reconstructs: dc_float = (dc_stored + y_dc_stored * dc_cfl_factor) / inv_factor
    //
    // For Y channel (dc_cfl_factor=0): dc_float = dc_stored / inv_factor
    // For X channel (dc_cfl_factor=0): dc_float = dc_stored / inv_factor
    // For B channel (dc_cfl_factor=0.5): dc_float = (dc_stored + y_dc * 0.5) / inv_factor

    match raw_strategy {
        RAW_STRATEGY_DCT8
        | RAW_STRATEGY_DCT4X4
        | RAW_STRATEGY_DCT4X8
        | RAW_STRATEGY_DCT8X4
        | RAW_STRATEGY_IDENTITY
        | RAW_STRATEGY_DCT2X2
        | RAW_STRATEGY_AFV0
        | RAW_STRATEGY_AFV1
        | RAW_STRATEGY_AFV2
        | RAW_STRATEGY_AFV3 => {
            // Single-block: LLF is just DC at position [0]
            coeffs[0] = dequant_dc(0, 0);
        }

        RAW_STRATEGY_DCT16X8 => {
            // 2 DC values in column (by, by+1)
            let dc0 = dequant_dc(0, 0);
            let dc1 = dequant_dc(1, 0);

            // Inverse of dc_from_dct_16x8:
            // Forward: dc[0] = llf0*s0 + llf1*s1, dc[1] = llf0*s0 - llf1*s1
            // Inverse: llf0 = (dc0+dc1) / (2*s0), llf1 = (dc0-dc1) / (2*s1)
            // Note: 2-point Hadamard H*H = 2*I, so inverse = H/2
            let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
            let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
            coeffs[0] = (dc0 + dc1) / (2.0 * s0);
            coeffs[1] = (dc0 - dc1) / (2.0 * s1);
        }

        RAW_STRATEGY_DCT8X16 => {
            // 2 DC values in row (bx, bx+1)
            let dc0 = dequant_dc(0, 0);
            let dc1 = dequant_dc(0, 1);

            let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
            let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
            coeffs[0] = (dc0 + dc1) / (2.0 * s0);
            coeffs[1] = (dc0 - dc1) / (2.0 * s1);
        }

        RAW_STRATEGY_DCT16X16 => {
            // 2x2 DC values
            let mut dc_grid = [0.0f32; 4];
            for iy in 0..2 {
                for ix in 0..2 {
                    dc_grid[iy * 2 + ix] = dequant_dc(iy, ix);
                }
            }

            // Inverse of dc_from_dct_16x16:
            // dc_from_dct_16x16 extracts LLF positions, scales by SCALE_16_TO_2, then
            // applies 2x2 Hadamard (dct1d_2 on rows, transpose, dct1d_2 on rows).
            // Hadamard is self-inverse: H*H = 4*I for 2x2.
            // So: coeffs_llf = H(dc_grid) / (4 * scale)
            let h00 = dc_grid[0] + dc_grid[1] + dc_grid[2] + dc_grid[3];
            let h01 = dc_grid[0] + dc_grid[1] - dc_grid[2] - dc_grid[3];
            let h10 = dc_grid[0] - dc_grid[1] + dc_grid[2] - dc_grid[3];
            let h11 = dc_grid[0] - dc_grid[1] - dc_grid[2] + dc_grid[3];

            let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
            let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];

            coeffs[0] = h00 / (4.0 * s0 * s0);
            coeffs[1] = h01 / (4.0 * s0 * s1);
            coeffs[16] = h10 / (4.0 * s1 * s0);
            coeffs[17] = h11 / (4.0 * s1 * s1);
        }

        RAW_STRATEGY_DCT32X32 => {
            // 4x4 DC values
            let mut dc_grid = [0.0f32; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    dc_grid[iy * 4 + ix] = dequant_dc(iy, ix);
                }
            }

            // dc_from_dct_32x32 applies:
            //   block[iy*4+ix] = coeffs[iy*32+ix] * SCALE_32_TO_4[iy] * SCALE_32_TO_4[ix] * 16.0
            //   then matched 4x4 IDCT (idct1d_4 on rows, transpose, idct1d_4 on rows)
            //
            // Inverse: forward 4x4 DCT of dc_grid, then divide by (SCALE * 16)

            let mut block = dc_grid;
            // Forward 4pt DCT on rows
            dct1d_4(&mut block[0..4]);
            dct1d_4(&mut block[4..8]);
            dct1d_4(&mut block[8..12]);
            dct1d_4(&mut block[12..16]);
            // Transpose 4x4
            let mut transposed = [0.0f32; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    transposed[ix * 4 + iy] = block[iy * 4 + ix];
                }
            }
            // Forward 4pt DCT on rows
            dct1d_4(&mut transposed[0..4]);
            dct1d_4(&mut transposed[4..8]);
            dct1d_4(&mut transposed[8..12]);
            dct1d_4(&mut transposed[12..16]);

            // Write to LLF positions
            for iy in 0..4 {
                for ix in 0..4 {
                    let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
                    coeffs[iy * 32 + ix] = transposed[iy * 4 + ix] / (scale * 16.0);
                }
            }
        }

        RAW_STRATEGY_DCT32X16 => {
            // 4x2 DC values (4 rows, 2 cols) — physical block layout
            let mut dc_grid = [0.0f32; 8];
            for iy in 0..4 {
                for ix in 0..2 {
                    dc_grid[iy * 2 + ix] = dequant_dc(iy, ix);
                }
            }

            // Inverse of dc_from_dct_32x16:
            //   Extraction reads 2×4 LLF at coeffs[iy*32+ix] for iy 0..2, ix 0..4
            //   Scales: SCALE_16_TO_2[iy] * SCALE_32_TO_4[ix] * 4.0
            //   IDCT: 4pt rows → transpose 2×4→4×2 → 2pt rows → 4×2 spatial output
            //
            // Reconstruction: forward DCT (inverse of IDCT), write to same LLF positions.

            // Forward 2pt DCT on rows (4 rows of 2)
            let mut block = dc_grid;
            for iy in 0..4 {
                dct1d_2(&mut block[iy * 2..(iy + 1) * 2]);
            }
            // Transpose 4x2 -> 2x4
            let mut transposed = [0.0f32; 8];
            for iy in 0..4 {
                for ix in 0..2 {
                    transposed[ix * 4 + iy] = block[iy * 2 + ix];
                }
            }
            // Forward 4pt DCT on rows (2 rows of 4)
            dct1d_4(&mut transposed[0..4]);
            dct1d_4(&mut transposed[4..8]);

            // Write to 2×4 LLF positions (rows 0..2, cols 0..4, stride 32)
            // Forward DCT gain: dct1d_2(2) × dct1d_4(4) = 8 → divide by 8
            for iy in 0..2 {
                for ix in 0..4 {
                    let scale = DCT_RESAMPLE_SCALE_16_TO_2[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
                    coeffs[iy * 32 + ix] = transposed[iy * 4 + ix] / (scale * 8.0);
                }
            }
        }

        RAW_STRATEGY_DCT16X32 => {
            // 2x4 DC values (2 rows, 4 cols)
            let mut dc_grid = [0.0f32; 8];
            for iy in 0..2 {
                for ix in 0..4 {
                    dc_grid[iy * 4 + ix] = dequant_dc(iy, ix);
                }
            }

            // Inverse of dc_from_dct_16x32 (ROWS<COLS branch):
            //   Extraction reads 2×4 LLF at coeffs[iy*32+ix] for iy 0..2, ix 0..4
            //   Scales: SCALE_16_TO_2[iy] * SCALE_32_TO_4[ix] * 4.0 (IDCT gain 1/4)
            //   IDCT: 4pt rows → transpose 2×4→4×2 → 2pt rows → transpose 4×2→2×4 → 2×4 spatial output
            //
            // Reconstruction: forward DCT gain = dct1d_4(4) × dct1d_2(2) = 8 → divide by 8
            let mut block = dc_grid;
            // Forward 4pt DCT on rows (2 rows of 4)
            dct1d_4(&mut block[0..4]);
            dct1d_4(&mut block[4..8]);
            // Transpose 2x4 -> 4x2
            let mut transposed = [0.0f32; 8];
            for iy in 0..2 {
                for ix in 0..4 {
                    transposed[ix * 2 + iy] = block[iy * 4 + ix];
                }
            }
            // Forward 2pt DCT on rows (4 rows of 2)
            for iy in 0..4 {
                dct1d_2(&mut transposed[iy * 2..(iy + 1) * 2]);
            }
            // Transpose back 4x2 -> 2x4
            let mut result = [0.0f32; 8];
            for iy in 0..4 {
                for ix in 0..2 {
                    result[ix * 4 + iy] = transposed[iy * 2 + ix];
                }
            }

            // Write to LLF positions (stride 32)
            for iy in 0..2 {
                for ix in 0..4 {
                    let scale = DCT_RESAMPLE_SCALE_16_TO_2[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
                    coeffs[iy * 32 + ix] = result[iy * 4 + ix] / (scale * 8.0);
                }
            }
        }

        RAW_STRATEGY_DCT64X64 => {
            // 8x8 DC values
            let mut dc_grid = [0.0f32; 64];
            for iy in 0..8 {
                for ix in 0..8 {
                    dc_grid[iy * 8 + ix] = dequant_dc(iy, ix);
                }
            }

            // dc_from_dct_64x64:
            //   block[iy*8+ix] = coeffs[iy*64+ix] * SCALE_64_TO_8[iy] * SCALE_64_TO_8[ix]
            //   8x8 IDCT
            //
            // Inverse: 8x8 forward DCT then divide by scale
            let mut output = [0.0f32; 64];
            dct_8x8(&dc_grid, &mut output);

            for iy in 0..8 {
                for ix in 0..8 {
                    let scale = DCT_RESAMPLE_SCALE_64_TO_8[iy] * DCT_RESAMPLE_SCALE_64_TO_8[ix];
                    coeffs[iy * 64 + ix] = output[iy * 8 + ix] / scale;
                }
            }
        }

        RAW_STRATEGY_DCT64X32 => {
            // 8x4 DC values (8 rows, 4 cols) — physical block layout
            let mut dc_grid = [0.0f32; 32];
            for iy in 0..8 {
                for ix in 0..4 {
                    dc_grid[iy * 4 + ix] = dequant_dc(iy, ix);
                }
            }

            // Inverse of dc_from_dct_64x32:
            //   Extraction reads 4×8 LLF at coeffs[iy*64+ix] for iy 0..4, ix 0..8
            //   Scales: SCALE_32_TO_4[iy] * SCALE_64_TO_8[ix] * 4.0
            //   IDCT: 8pt rows → transpose 4×8→8×4 → 4pt rows → 8×4 spatial output
            //
            // Reconstruction: forward DCT, write to same LLF positions.

            let mut block = dc_grid;
            // Forward 4pt DCT on rows (8 rows of 4)
            for iy in 0..8 {
                dct1d_4(&mut block[iy * 4..(iy + 1) * 4]);
            }
            // Transpose 8x4 -> 4x8
            let mut transposed = [0.0f32; 32];
            for iy in 0..8 {
                for ix in 0..4 {
                    transposed[ix * 8 + iy] = block[iy * 4 + ix];
                }
            }
            // Forward 8pt DCT on rows (4 rows of 8), compensate dct1d_8 gain
            for iy in 0..4 {
                let s = iy * 8;
                dct1d_8(&mut transposed[s..s + 8]);
                for i in 0..8 {
                    transposed[s + i] *= 1.0 / 8.0;
                }
            }

            // Write to 4×8 LLF positions (rows 0..4, cols 0..8, stride 64)
            // Forward DCT gain: dct1d_4(4) × dct1d_8/8(1) = 4 → divide by 4
            for iy in 0..4 {
                for ix in 0..8 {
                    let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_64_TO_8[ix];
                    coeffs[iy * 64 + ix] = transposed[iy * 8 + ix] / (scale * 4.0);
                }
            }
        }

        RAW_STRATEGY_DCT32X64 => {
            // 4x8 DC values (4 rows, 8 cols)
            let mut dc_grid = [0.0f32; 32];
            for iy in 0..4 {
                for ix in 0..8 {
                    dc_grid[iy * 8 + ix] = dequant_dc(iy, ix);
                }
            }

            // dc_from_dct_32x64 (ROWS<COLS branch):
            //   block[iy*8+ix] = coeffs[iy*64+ix] * SCALE_32_TO_4[iy] * SCALE_64_TO_8[ix] * 4.0
            //   4x8 IDCT: idct on 8-element rows, transpose 4x8->8x4, idct on 4-element rows, transpose back

            let mut block = dc_grid;
            // Forward 8pt DCT on rows (4 rows of 8)
            for iy in 0..4 {
                let s = iy * 8;
                dct1d_8(&mut block[s..s + 8]);
                for i in 0..8 {
                    block[s + i] *= 1.0 / 8.0;
                }
            }
            // Transpose 4x8 -> 8x4
            let mut transposed = [0.0f32; 32];
            for iy in 0..4 {
                for ix in 0..8 {
                    transposed[ix * 4 + iy] = block[iy * 8 + ix];
                }
            }
            // Forward 4pt DCT on rows (8 rows of 4)
            for iy in 0..8 {
                dct1d_4(&mut transposed[iy * 4..(iy + 1) * 4]);
            }
            // Transpose back 8x4 -> 4x8
            let mut result = [0.0f32; 32];
            for iy in 0..8 {
                for ix in 0..4 {
                    result[ix * 8 + iy] = transposed[iy * 4 + ix];
                }
            }

            for iy in 0..4 {
                for ix in 0..8 {
                    let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_64_TO_8[ix];
                    coeffs[iy * 64 + ix] = result[iy * 8 + ix] / (scale * 4.0);
                }
            }
        }

        _ => {
            // Unknown strategy — shouldn't happen
        }
    }
}

/// Apply IDCT for a given strategy, producing pixel-domain output.
fn idct_for_strategy(raw_strategy: u8, coeffs: &[f32], output: &mut [f32]) {
    match raw_strategy {
        RAW_STRATEGY_DCT8 => {
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);
            let mut tmp = [0.0f32; 64];
            idct_8x8(&input, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT4X4 => {
            // Inverse of dct_4x4_full: undo DC combining, de-interleave, apply idct_4x4
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);

            // Undo 2x2 DC combining (inverse of 2x2 Hadamard * 0.25)
            let dc00 = input[0];
            let dc01 = input[1];
            let dc10 = input[8];
            let dc11 = input[9];
            input[0] = dc00 + dc01 + dc10 + dc11;
            input[1] = dc00 + dc01 - dc10 - dc11;
            input[8] = dc00 - dc01 + dc10 + dc11;
            input[9] = dc00 - dc01 - dc10 + dc11;

            // All 64 positions written: 4 sub-blocks × 4×4 = 64 pixels
            for y in 0..2 {
                for x in 0..2 {
                    // De-interleave sub-block coefficients
                    let mut sub = [0.0f32; 16];
                    for iy in 0..4 {
                        for ix in 0..4 {
                            sub[iy * 4 + ix] = input[(y + iy * 2) * 8 + x + ix * 2];
                        }
                    }
                    // Apply base 4x4 IDCT
                    let mut pixels = [0.0f32; 16];
                    idct_4x4(&sub, &mut pixels);
                    // Place into output
                    for iy in 0..4 {
                        for ix in 0..4 {
                            output[(y * 4 + iy) * 8 + (x * 4 + ix)] = pixels[iy * 4 + ix];
                        }
                    }
                }
            }
        }
        RAW_STRATEGY_DCT4X8 => {
            // Inverse of dct_4x8_full: undo DC combining, de-interleave, apply idct_4x8
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);

            // Undo 2-point DC combining (inverse of Hadamard * 0.5)
            let dc0 = input[0];
            let dc1 = input[8];
            input[0] = dc0 + dc1;
            input[8] = dc0 - dc1;

            // All 64 positions written: 2 sub-blocks × 4×8 = 64 pixels
            for y in 0..2 {
                // De-interleave sub-block coefficients
                let mut sub = [0.0f32; 32];
                for iy in 0..4 {
                    for ix in 0..8 {
                        sub[iy * 8 + ix] = input[(y + iy * 2) * 8 + ix];
                    }
                }
                // Apply base 4x8 IDCT
                let mut pixels = [0.0f32; 32];
                idct_4x8(&sub, &mut pixels);
                // Place into output (4 rows, 8 cols)
                for iy in 0..4 {
                    for ix in 0..8 {
                        output[(y * 4 + iy) * 8 + ix] = pixels[iy * 8 + ix];
                    }
                }
            }
        }
        RAW_STRATEGY_DCT8X4 => {
            // Inverse of dct_8x4_full: undo DC combining, de-interleave, apply idct_8x4
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);

            // Undo 2-point DC combining (inverse of Hadamard * 0.5)
            let dc0 = input[0];
            let dc1 = input[8];
            input[0] = dc0 + dc1;
            input[8] = dc0 - dc1;

            // All 64 positions written: 2 sub-blocks × 8×4 = 64 pixels
            for x in 0..2 {
                // De-interleave sub-block coefficients
                let mut sub = [0.0f32; 32];
                for iy in 0..4 {
                    for ix in 0..8 {
                        sub[iy * 8 + ix] = input[(x + iy * 2) * 8 + ix];
                    }
                }
                // Apply base 8x4 IDCT
                let mut pixels = [0.0f32; 32];
                idct_8x4(&sub, &mut pixels);
                // Place into output (8 rows, 4 cols)
                for iy in 0..8 {
                    for ix in 0..4 {
                        output[iy * 8 + (x * 4 + ix)] = pixels[iy * 4 + ix];
                    }
                }
            }
        }
        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
            let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);
            let mut tmp = [0.0f32; 64];
            super::afv::inverse_afv_transform(&input, afv_kind, &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT16X8 => {
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&coeffs[..128]);
            let mut tmp = [0.0f32; 128];
            idct_16x8(&input, &mut tmp);
            output[..128].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT8X16 => {
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&coeffs[..128]);
            let mut tmp = [0.0f32; 128];
            idct_8x16(&input, &mut tmp);
            output[..128].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT16X16 => {
            let mut input = [0.0f32; 256];
            input.copy_from_slice(&coeffs[..256]);
            let mut tmp = [0.0f32; 256];
            idct_16x16(&input, &mut tmp);
            output[..256].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT32X32 => {
            let mut input = [0.0f32; 1024];
            input.copy_from_slice(&coeffs[..1024]);
            let mut tmp = [0.0f32; 1024];
            idct_32x32(&input, &mut tmp);
            output[..1024].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT32X16 => {
            let mut input = [0.0f32; 512];
            input.copy_from_slice(&coeffs[..512]);
            let mut tmp = [0.0f32; 512];
            idct_32x16(&input, &mut tmp);
            output[..512].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT16X32 => {
            let mut input = [0.0f32; 512];
            input.copy_from_slice(&coeffs[..512]);
            let mut tmp = [0.0f32; 512];
            idct_16x32(&input, &mut tmp);
            output[..512].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT64X64 => {
            // DCT64 uses stack arrays via the output parameter
            idct_64x64(&coeffs[..4096], &mut output[..4096]);
        }
        RAW_STRATEGY_DCT64X32 => {
            idct_64x32(&coeffs[..2048], &mut output[..2048]);
        }
        RAW_STRATEGY_DCT32X64 => {
            idct_32x64(&coeffs[..2048], &mut output[..2048]);
        }
        RAW_STRATEGY_IDENTITY => {
            let mut tmp = [0.0f32; 64];
            inverse_identity_transform(as_array_ref(coeffs, 0), &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        RAW_STRATEGY_DCT2X2 => {
            let mut tmp = [0.0f32; 64];
            inverse_dct2x2_transform(as_array_ref(coeffs, 0), &mut tmp);
            output[..64].copy_from_slice(&tmp);
        }
        _ => {
            // Unknown strategy: output zeros
            output[..64].fill(0.0);
        }
    }
}

/// Apply decoder-side gaborish smooth (3x3 weighted blur).
///
/// This is the decoder's 3x3 convolution that compensates for the encoder's
/// 5x5 sharpening pre-filter. Applied per-channel independently.
///
/// Default gab weights (all channels same):
/// ```text
///   w2  w1  w2
///   w1  c   w1
///   w2  w1  w2
/// ```
/// where w1 = 0.115170, w2 = 0.061249, c = 1.0, normalized by 1/(1 + 4*(w1+w2)).
pub(crate) fn gab_smooth(planes: &mut [Vec<f32>; 3], width: usize, height: usize) {
    // Gab weights from libjxl epf.cc / loop_filter.h
    let w1_base = 0.104_699_57_f32 * 1.1;
    let w2_base = 0.055_680_54_f32 * 1.1;
    let div = 1.0 + 4.0 * (w1_base + w2_base);
    let w_center = 1.0 / div;
    let w1 = w1_base / div;
    let w2 = w2_base / div;

    // Reuse a single scratch buffer across all 3 channels to avoid 3x cloning
    let num_pixels = width * height;
    let mut scratch = vec![0.0f32; num_pixels];

    for plane in planes.iter_mut() {
        jxl_simd::gab_smooth_channel(plane, &mut scratch, width, height, w_center, w1, w2);
    }
}

/// Convert XYB pixel planes to interleaved linear RGB.
///
/// Implements the inverse of the XYB color transform:
/// 1. Unmix: L = Y + X, M = Y - X, S = B
/// 2. Undo gamma: add cbrt(bias), then cube, then subtract bias
/// 3. Apply inverse opsin matrix to get linear RGB
///
/// Output: interleaved [R, G, B, R, G, B, ...] in linear light (0.0-1.0 range).
/// Values are NOT clamped — caller should clamp if needed.
#[cfg(any(test, feature = "butteraugli-loop"))]
#[allow(dead_code)] // Used by `test_xyb_to_linear_rgb_roundtrip` in this module.
pub(crate) fn xyb_to_linear_rgb(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let num_pixels = width * height;
    let mut linear_rgb = vec![0.0f32; num_pixels * 3];
    jxl_simd::xyb_to_linear_rgb_batch(xyb_x, xyb_y, xyb_b, &mut linear_rgb, num_pixels);
    linear_rgb
}

/// Converts XYB planes to planar linear RGB (3 separate channel buffers).
///
/// Avoids the interleave step when the consumer needs planar data (e.g., butteraugli).
/// Only consumers under `butteraugli-loop` / `ssim2-loop` / `zensim-loop` use
/// this; gated on the same set so non-loop builds don't trip dead-code warnings.
#[cfg(any(
    feature = "butteraugli-loop",
    feature = "ssim2-loop",
    feature = "zensim-loop"
))]
pub(crate) fn xyb_to_linear_rgb_planar(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    num_pixels: usize,
) {
    jxl_simd::xyb_to_linear_rgb_planar(xyb_x, xyb_y, xyb_b, out_r, out_g, out_b, num_pixels);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that LLF restoration is the inverse of DC extraction for DCT16x16.
    #[test]
    fn test_llf_roundtrip_16x16() {
        // Create a test 16x16 block with known DCT coefficients
        let input: [f32; 256] = core::array::from_fn(|i| ((i as f32 * 0.7).sin()) * 50.0);
        let mut coeffs = [0.0f32; 256];
        dct_16x16(&input, &mut coeffs);

        // Extract DC values using the forward path
        let dcs = dc_from_dct_16x16(&coeffs);

        // Verify the Hadamard inverse recovers the original LLF coefficients
        let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
        let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];

        // Original LLF
        let orig_llf = [coeffs[0], coeffs[1], coeffs[16], coeffs[17]];

        // DC values from extraction
        let dc_grid = dcs;

        // Hadamard of dc_grid
        let h00 = dc_grid[0] + dc_grid[1] + dc_grid[2] + dc_grid[3];
        let h01 = dc_grid[0] + dc_grid[1] - dc_grid[2] - dc_grid[3];
        let h10 = dc_grid[0] - dc_grid[1] + dc_grid[2] - dc_grid[3];
        let h11 = dc_grid[0] - dc_grid[1] - dc_grid[2] + dc_grid[3];

        let restored_llf = [
            h00 / (4.0 * s0 * s0),
            h01 / (4.0 * s0 * s1),
            h10 / (4.0 * s1 * s0),
            h11 / (4.0 * s1 * s1),
        ];

        for i in 0..4 {
            let err = (orig_llf[i] - restored_llf[i]).abs();
            assert!(
                err < 1e-3,
                "LLF16x16[{}]: orig={}, restored={}, err={}",
                i,
                orig_llf[i],
                restored_llf[i],
                err
            );
        }
    }

    /// Test that LLF restoration is the inverse of DC extraction for DCT32x32.
    #[test]
    fn test_llf_roundtrip_32x32() {
        let input: [f32; 1024] = core::array::from_fn(|i| ((i as f32 * 0.3).sin()) * 30.0);
        let mut coeffs = [0.0f32; 1024];
        dct_32x32(&input, &mut coeffs);

        let dcs = dc_from_dct_32x32(&coeffs);

        // Original LLF
        let mut orig_llf = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                orig_llf[iy * 4 + ix] = coeffs[iy * 32 + ix];
            }
        }

        // Restore: forward 4x4 DCT of dc_grid, divide by (scale * 16)
        let mut dc_grid = dcs;
        dct1d_4(&mut dc_grid[0..4]);
        dct1d_4(&mut dc_grid[4..8]);
        dct1d_4(&mut dc_grid[8..12]);
        dct1d_4(&mut dc_grid[12..16]);
        let mut transposed = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                transposed[ix * 4 + iy] = dc_grid[iy * 4 + ix];
            }
        }
        dct1d_4(&mut transposed[0..4]);
        dct1d_4(&mut transposed[4..8]);
        dct1d_4(&mut transposed[8..12]);
        dct1d_4(&mut transposed[12..16]);

        let mut restored_llf = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
                restored_llf[iy * 4 + ix] = transposed[iy * 4 + ix] / (scale * 16.0);
            }
        }

        for i in 0..16 {
            let err = (orig_llf[i] - restored_llf[i]).abs();
            assert!(
                err < 1e-2,
                "LLF32x32[{}]: orig={}, restored={}, err={}",
                i,
                orig_llf[i],
                restored_llf[i],
                err
            );
        }
    }

    /// Test gab_smooth produces reasonable output (no NaN, preserves constant).
    #[test]
    fn test_gab_smooth_constant() {
        let w = 16;
        let h = 16;
        let val = 42.0f32;
        let mut planes = [vec![val; w * h], vec![val; w * h], vec![val; w * h]];
        gab_smooth(&mut planes, w, h);

        // Constant input should produce constant output
        for (c, plane) in planes.iter().enumerate() {
            for (i, &v) in plane.iter().enumerate() {
                let err = (v - val).abs();
                assert!(
                    err < 1e-5,
                    "gab_smooth constant: c={} i={} got {} expected {}",
                    c,
                    i,
                    v,
                    val
                );
            }
        }
    }

    /// Test that XYB → linear RGB inverse is the inverse of linear RGB → XYB forward.
    #[cfg(feature = "butteraugli-loop")]
    #[test]
    fn test_xyb_to_linear_rgb_roundtrip() {
        use crate::color::xyb::linear_rgb_to_xyb;

        // Test several colors
        let test_colors: &[(f32, f32, f32)] = &[
            (1.0, 0.0, 0.0),    // red
            (0.0, 1.0, 0.0),    // green
            (0.0, 0.0, 1.0),    // blue
            (1.0, 1.0, 1.0),    // white
            (0.0, 0.0, 0.0),    // black
            (0.5, 0.3, 0.7),    // arbitrary
            (0.18, 0.18, 0.18), // mid-gray
        ];

        for &(r, g, b) in test_colors {
            let (x, y, b_xyb) = linear_rgb_to_xyb(r, g, b);

            // Inverse via xyb_to_linear_rgb
            let xyb_x = [x];
            let xyb_y = [y];
            let xyb_b = [b_xyb];
            let linear = xyb_to_linear_rgb(&xyb_x, &xyb_y, &xyb_b, 1, 1);

            let r2 = linear[0];
            let g2 = linear[1];
            let b2 = linear[2];

            let err_r = (r - r2).abs();
            let err_g = (g - g2).abs();
            let err_b = (b - b2).abs();

            assert!(
                err_r < 1e-5 && err_g < 1e-5 && err_b < 1e-5,
                "XYB roundtrip failed for ({}, {}, {}): got ({}, {}, {}), err=({}, {}, {})",
                r,
                g,
                b,
                r2,
                g2,
                b2,
                err_r,
                err_g,
                err_b
            );
        }
    }

    /// Test that full quantize→dequant→IDCT roundtrip works for DCT16x16.
    /// This isolates whether the reconstruction formula matches the encoder formula.
    #[test]
    fn test_full_roundtrip_dct16x16() {
        use super::super::dct::{dc_from_dct_16x16, dct_16x16};
        use super::super::frame::DistanceParams;
        use super::super::quant::quant_weights;

        // Create a 16x16 pixel block with varied content
        let pixels: [f32; 256] = core::array::from_fn(|i| {
            let x = (i % 16) as f32;
            let y = (i / 16) as f32;
            // A mix of low and high frequency content
            0.5 + 0.2 * (x * 0.5).sin() + 0.1 * (y * 0.3).cos() + 0.05 * ((x + y) * 0.7).sin()
        });

        // Forward DCT
        let mut coeffs = [0.0f32; 256];
        dct_16x16(&pixels, &mut coeffs);

        // Quantize: val = coeff * inv_w * qac * qm_mul, quantized = round(val)
        let strategy = 3; // DCT16x16
        let channel = 1; // Y channel (qm_mul = 1.0)
        let qf = 6u8;
        let params = DistanceParams::compute(1.0);
        let qac = params.scale * qf as f32;
        let weights = quant_weights(strategy, channel);
        let qm_mul = 1.0f32; // Y channel

        let mut quantized = [0i32; 256];
        // Skip LLF positions
        let cx = 2usize;
        let cy = 2usize;
        for idx in 0..256 {
            let y = idx / 16;
            let x = idx % 16;
            let slot_y = y / 8;
            let slot_x = x / 8;
            let is_llf = slot_y < cy && slot_x < cx && (y % 8) == 0 && (x % 8) == 0;
            if is_llf {
                continue;
            }
            let inv_w = 1.0 / weights[idx];
            let val = coeffs[idx] * inv_w * qac * qm_mul;
            quantized[idx] = val.round() as i32;
        }

        // Extract DC values (forward path)
        let dcs = dc_from_dct_16x16(&coeffs);
        let inv_factor = super::super::quant::INV_DC_QUANT[channel] * params.scale_dc;
        let quant_dc: Vec<i16> = dcs
            .iter()
            .map(|&dc| (dc * inv_factor).round() as i16)
            .collect();

        // Now reconstruct: dequant AC + restore LLF from DC + IDCT
        let mut dequant = [0.0f32; 256];

        // Dequant AC
        for idx in 0..256 {
            let y = idx / 16;
            let x = idx % 16;
            let slot_y = y / 8;
            let slot_x = x / 8;
            let is_llf = slot_y < cy && slot_x < cx && (y % 8) == 0 && (x % 8) == 0;
            if is_llf {
                continue;
            }
            if quantized[idx] != 0 {
                let biased = adjust_quant_bias(quantized[idx], channel);
                let weight = weights[idx];
                dequant[idx] = biased * weight / (qac * qm_mul);
            }
        }

        // Restore LLF from DC (same as in restore_llf_from_dc for DCT16x16)
        let dc_grid: Vec<f32> = quant_dc.iter().map(|&v| v as f32 / inv_factor).collect();

        let s0 = super::super::dct::DCT_RESAMPLE_SCALE_16_TO_2[0];
        let s1 = super::super::dct::DCT_RESAMPLE_SCALE_16_TO_2[1];

        let h00 = dc_grid[0] + dc_grid[1] + dc_grid[2] + dc_grid[3];
        let h01 = dc_grid[0] + dc_grid[1] - dc_grid[2] - dc_grid[3];
        let h10 = dc_grid[0] - dc_grid[1] + dc_grid[2] - dc_grid[3];
        let h11 = dc_grid[0] - dc_grid[1] - dc_grid[2] + dc_grid[3];

        dequant[0] = h00 / (4.0 * s0 * s0);
        dequant[1] = h01 / (4.0 * s0 * s1);
        dequant[16] = h10 / (4.0 * s1 * s0);
        dequant[17] = h11 / (4.0 * s1 * s1);

        // IDCT
        let mut recon_pixels = [0.0f32; 256];
        super::super::dct::idct_16x16(&dequant, &mut recon_pixels);

        // Compare
        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f32;
        for i in 0..256 {
            let err = (pixels[i] - recon_pixels[i]).abs();
            if err > max_err {
                max_err = err;
            }
            sum_err += err;
        }
        let mean_err = sum_err / 256.0;

        println!(
            "DCT16x16 roundtrip: mean_err={:.6}, max_err={:.6}, pixel range=[{:.3}, {:.3}]",
            mean_err,
            max_err,
            pixels.iter().cloned().fold(f32::INFINITY, f32::min),
            pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // For lossy quantization at d=1.0, expect reasonable error
        assert!(
            max_err < 0.5,
            "DCT16x16 roundtrip max error too large: {} (mean: {})",
            max_err,
            mean_err
        );
    }
}
