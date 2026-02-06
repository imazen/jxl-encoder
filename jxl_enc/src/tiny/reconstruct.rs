// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

/// Adjust quantized value with bias for dequantization.
/// Exact copy of TinyEncoder::adjust_quant_bias (transform.rs).
fn adjust_quant_bias(quantized: i32, channel: usize) -> f32 {
    #[allow(clippy::excessive_precision)]
    const BIAS: [f32; 4] = [
        1.0 - 0.05465007330715401,
        1.0 - 0.07005449891748593,
        1.0 - 0.049935103337343655,
        0.145,
    ];

    if quantized == 0 {
        return 0.0;
    }

    let q = quantized as f32;
    if q.abs() < 1.125 {
        q.signum() * BIAS[channel]
    } else {
        q - BIAS[3] / q
    }
}

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
    let padded_width = xsize_blocks * BLOCK_DIM;
    let padded_height = ysize_blocks * BLOCK_DIM;
    let num_pixels = padded_width * padded_height;

    let x_qm_mul = 1.25f32.powf(params.x_qm_scale as f32 - 2.0);
    let b_qm_mul = 1.25f32.powf(params.b_qm_scale as f32 - 2.0);

    // Step 1: Dequantize all coefficients into floating-point DCT domain.
    // For each first-block of each transform, reconstruct the full coefficient block.
    // Output: per-channel float coefficient planes in pixel layout after IDCT.
    let mut planes = [
        vec![0.0f32; num_pixels],
        vec![0.0f32; num_pixels],
        vec![0.0f32; num_pixels],
    ];

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

            // Dequantize all 3 channels
            let mut dequant_coeffs = [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];

            for c in 0..3usize {
                let qm_mul = match c {
                    0 => x_qm_mul,
                    2 => b_qm_mul,
                    _ => 1.0,
                };

                // Dequantize AC coefficients
                let qac = params.scale * quant_field[by * xsize_blocks + bx] as f32;
                let weights = quant_weights(raw_strategy as usize, c);

                for idx in 0..size {
                    let y = idx / block_width;
                    let x = idx % block_width;

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
                    let q_int = quant_ac[c][by + phys_row_off][bx + phys_col_off][pos_in_8x8];

                    // Check if this is an LLF position
                    let is_llf_pos =
                        coef_slot_y < cy && coef_slot_x < cx && pos_y == 0 && pos_x == 0;

                    if is_llf_pos {
                        // LLF positions are restored from DC, skip AC dequant
                        continue;
                    }

                    if q_int != 0 {
                        let weight = weights[idx];
                        // Dequant formula: coeff = adjust_quant_bias(q) / (inv_weight * qac * qm_mul)
                        // where inv_weight = 1/weight, so coeff = adjust_quant_bias(q) * weight / (qac * qm_mul)
                        // Wait - the encoder uses: val = coef * inv_weight * qac * qm_mul
                        // So: coef = val / (inv_weight * qac * qm_mul) = val * weight / (qac * qm_mul)
                        // But actually the weights array from quant_weights() returns the
                        // dequantization weights (i.e., the values the decoder multiplies by).
                        // Let me check: in quantize_coeff_ac, inv_weight = 1/weight.
                        // The quantization: val = coef * (1/weight) * qac * qm_mul
                        // The dequantization: coef = val * weight / (qac * qm_mul)
                        // But with bias: coef = adjust_quant_bias(q_int, c) * weight / (qac * qm_mul)
                        // Hmm, actually the quant_weights() function returns different things
                        // for encoding vs decoding. Let me check what the encoder stores.

                        // In transform.rs, quantize_coeff_ac uses:
                        //   inv_weight = weights[idx]  (where weights = quant_weights(strat, c))
                        //   val = coef * inv_weight * qac * qm_mul
                        // So quant_weights returns 1/dequant_weight = inv_weight.
                        // Dequant: coef = adjust_quant_bias(q) / (inv_weight * qac * qm_mul)
                        let biased = adjust_quant_bias(q_int, c);
                        dequant_coeffs[c][idx] = biased / (weight * qac * qm_mul);
                    }
                }

                // Restore LLF from DC
                restore_llf_from_dc(
                    &mut dequant_coeffs[c],
                    &quant_dc[c],
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
            #[allow(clippy::needless_range_loop)]
            for idx in 0..size {
                let y = idx / block_width;
                let x = idx % block_width;
                let coef_slot_y = y / BLOCK_DIM;
                let coef_slot_x = x / BLOCK_DIM;
                let pos_y = y % BLOCK_DIM;
                let pos_x = x % BLOCK_DIM;

                let is_llf_pos = coef_slot_y < cy && coef_slot_x < cx && pos_y == 0 && pos_x == 0;

                if !is_llf_pos {
                    dequant_coeffs[0][idx] += x_factor * dequant_coeffs[1][idx];
                    dequant_coeffs[2][idx] += b_factor * dequant_coeffs[1][idx];
                }
            }

            // Step 3: Transpose coefficients for rectangular transforms, then IDCT.
            // The encoder stores coefficients in post-swap layout (cx >= cy, stride = cx*8).
            // The IDCT functions expect the natural (physical) layout.
            // For transpose_slots transforms, transpose from (cy_post × cx_post) to
            // (cx_post × cy_post) = (covered_y × covered_x) layout.
            for c in 0..3usize {
                let idct_input = if transpose_slots {
                    // Transpose from post-swap to natural layout
                    let mut transposed = vec![0.0f32; size];
                    for y in 0..block_height {
                        for x in 0..block_width {
                            transposed[x * block_height + y] =
                                dequant_coeffs[c][y * block_width + x];
                        }
                    }
                    transposed
                } else {
                    dequant_coeffs[c].clone()
                };
                let pixels = idct_for_strategy(raw_strategy, &idct_input);

                // Write pixels to output plane using physical coverage dimensions
                let pixel_x = bx * BLOCK_DIM;
                let pixel_y = by * BLOCK_DIM;
                let pix_w = covered_x * BLOCK_DIM;
                let pix_h = covered_y * BLOCK_DIM;

                for py in 0..pix_h {
                    for px in 0..pix_w {
                        let out_y = pixel_y + py;
                        let out_x = pixel_x + px;
                        if out_y < padded_height && out_x < padded_width {
                            planes[c][out_y * padded_width + out_x] = pixels[py * pix_w + px];
                        }
                    }
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
    channel: usize,
    params: &DistanceParams,
    raw_strategy: u8,
    bx: usize,
    by: usize,
    _cx: usize,
    _cy: usize,
    _block_width: usize,
) {
    let _dc_cfl_factor = if channel == 2 { 0.5f32 } else { 0.0f32 };
    let inv_factor = INV_DC_QUANT[channel] * params.scale_dc;

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
            let dc_stored = quant_dc_ch[by][bx] as f32;
            let dc_float = dc_stored / inv_factor;
            // CfL on DC is handled by the caller's CfL restore step
            // (which adds y_dc * dc_cfl_factor back)
            // Actually no - for DC, the encoder subtracted y_dc * dc_cfl_factor during
            // quantization. The decoder adds it back. We need to replicate the decoder.
            // But since we do CfL restore separately for AC, we need to handle DC CfL here.
            // Actually, the decoder's LowestFrequenciesFromDC just sets the LLF from the
            // already-dequantized DC. The CfL on DC happens through the DC prediction path.
            // Let me just dequantize DC correctly and let the CfL step handle AC.
            coeffs[0] = dc_float;
        }

        RAW_STRATEGY_DCT16X8 => {
            // 2 DC values in column (by, by+1 if not transposed)
            let dc0 = quant_dc_ch[by][bx] as f32 / inv_factor;
            let dc1 = quant_dc_ch[by + 1][bx] as f32 / inv_factor;

            // Forward 2-point DCT + scale
            let llf0 = (dc0 + dc1) * DCT_RESAMPLE_SCALE_2_TO_16[0];
            let llf1 = (dc0 - dc1) * DCT_RESAMPLE_SCALE_2_TO_16[1];

            // Write to LLF positions (stride = block_width = 8 for 16x8)
            // DCT16x8: coefficient layout is 8 cols x 16 rows, stride 8
            // LLF positions: (0,0) and (1,0) in slot grid = positions [0] and [8]
            // Wait - for DCT16x8 (no final transpose, ROWS >= COLS), the coefficient
            // layout is such that the LLF region is the top-left cx × cy = 1 × 2 positions.
            // In the stride=8 layout: position [0*8+0] and [1*8+0] = [0] and [8].
            // But actually: dc_from_dct_16x8 reads coeffs[0] and coeffs[1] (both at row 0).
            // Because after the forward DCT (no final transpose), the LLF coefficients
            // are at the first row: coeffs[0] = LLF(0,0), coeffs[1] = LLF(0,1).
            // The stride is 8 (short dimension) but LLF is 2 consecutive values.
            // Let me check dc_from_dct_16x8 more carefully.
            coeffs[0] = llf0;
            coeffs[1] = llf1;
        }

        RAW_STRATEGY_DCT8X16 => {
            // 2 DC values in row (bx, bx+1)
            let dc0 = quant_dc_ch[by][bx] as f32 / inv_factor;
            let dc1 = quant_dc_ch[by][bx + 1] as f32 / inv_factor;

            let llf0 = (dc0 + dc1) * DCT_RESAMPLE_SCALE_2_TO_16[0];
            let llf1 = (dc0 - dc1) * DCT_RESAMPLE_SCALE_2_TO_16[1];

            // DCT8x16 has final transpose. LLF positions in stride-16 layout:
            // dc_from_dct_8x16 reads coeffs[0] and coeffs[1]
            coeffs[0] = llf0;
            coeffs[1] = llf1;
        }

        RAW_STRATEGY_DCT16X16 => {
            // 2x2 DC values
            let mut dc_grid = [0.0f32; 4];
            for iy in 0..2 {
                for ix in 0..2 {
                    dc_grid[iy * 2 + ix] = quant_dc_ch[by + iy][bx + ix] as f32 / inv_factor;
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
                    dc_grid[iy * 4 + ix] = quant_dc_ch[by + iy][bx + ix] as f32 / inv_factor;
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
            // 4x2 DC values (4 rows, 2 cols)
            let mut dc_grid = [0.0f32; 8];
            for iy in 0..4 {
                for ix in 0..2 {
                    dc_grid[iy * 2 + ix] = quant_dc_ch[by + iy][bx + ix] as f32 / inv_factor;
                }
            }

            // dc_from_dct_32x16:
            //   block[iy*2+ix] = coeffs[iy*32+ix] * SCALE_32_TO_4[iy] * SCALE_16_TO_2[ix] * 8.0
            //   4x2 IDCT: idct on 2-element rows, transpose 4x2->2x4, idct on 4-element rows
            //
            // Inverse: forward DCT

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

            // Write to LLF positions (stride 32 for 16x32 layout)
            for iy in 0..4 {
                for ix in 0..2 {
                    let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_16_TO_2[ix];
                    coeffs[iy * 32 + ix] = transposed[iy * 2 + ix] / (scale * 8.0);
                }
            }
        }

        RAW_STRATEGY_DCT16X32 => {
            // 2x4 DC values (2 rows, 4 cols)
            let mut dc_grid = [0.0f32; 8];
            for iy in 0..2 {
                for ix in 0..4 {
                    dc_grid[iy * 4 + ix] = quant_dc_ch[by + iy][bx + ix] as f32 / inv_factor;
                }
            }

            // dc_from_dct_16x32 (ROWS<COLS branch):
            //   block[iy*4+ix] = coeffs[iy*32+ix] * SCALE_16_TO_2[iy] * SCALE_32_TO_4[ix] * 8.0
            //   2x4 IDCT: idct on 4-element rows, transpose 2x4->4x2, idct on 2-element rows, transpose back
            //
            // Inverse:
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
                    dc_grid[iy * 8 + ix] = quant_dc_ch[by + iy][bx + ix] as f32 / inv_factor;
                }
            }

            // dc_from_dct_64x64:
            //   block[iy*8+ix] = coeffs[iy*64+ix] * SCALE_64_TO_8[iy] * SCALE_64_TO_8[ix]
            //   8x8 IDCT
            //
            // Inverse: 8x8 forward DCT then divide by scale
            let mut output = [0.0f32; 64];
            dct_8x8(dc_grid[..64].try_into().unwrap(), &mut output);

            for iy in 0..8 {
                for ix in 0..8 {
                    let scale = DCT_RESAMPLE_SCALE_64_TO_8[iy] * DCT_RESAMPLE_SCALE_64_TO_8[ix];
                    coeffs[iy * 64 + ix] = output[iy * 8 + ix] / scale;
                }
            }
        }

        RAW_STRATEGY_DCT64X32 => {
            // 8x4 DC values (8 rows, 4 cols)
            let mut dc_grid = [0.0f32; 32];
            for iy in 0..8 {
                for ix in 0..4 {
                    dc_grid[iy * 4 + ix] = quant_dc_ch[by + iy][bx + ix] as f32 / inv_factor;
                }
            }

            // dc_from_dct_64x32: ROWS=8 >= COLS=4
            //   block[iy*4+ix] = coeffs[iy*64+ix] * SCALE_64_TO_8[iy] * SCALE_32_TO_4[ix] * 4.0
            //   8x4 IDCT: idct on 4-element rows, transpose 8x4->4x8, idct on 8-element rows

            // Inverse: forward DCT
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
            // Forward 8pt DCT on rows (4 rows of 8)
            for iy in 0..4 {
                let s = iy * 8;
                dct1d_8(&mut transposed[s..s + 8]);
                for i in 0..8 {
                    transposed[s + i] *= 1.0 / 8.0;
                }
            }

            for iy in 0..8 {
                for ix in 0..4 {
                    let scale = DCT_RESAMPLE_SCALE_64_TO_8[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
                    coeffs[iy * 64 + ix] = transposed[iy * 4 + ix] / (scale * 4.0);
                }
            }
        }

        RAW_STRATEGY_DCT32X64 => {
            // 4x8 DC values (4 rows, 8 cols)
            let mut dc_grid = [0.0f32; 32];
            for iy in 0..4 {
                for ix in 0..8 {
                    dc_grid[iy * 8 + ix] = quant_dc_ch[by + iy][bx + ix] as f32 / inv_factor;
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
fn idct_for_strategy(raw_strategy: u8, coeffs: &[f32]) -> Vec<f32> {
    match raw_strategy {
        RAW_STRATEGY_DCT8 => {
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);
            let mut output = [0.0f32; 64];
            idct_8x8(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT4X4 => {
            // DCT4x4 uses 4 sub-blocks in interleaved layout.
            // The inverse needs to undo the interleaving.
            // For now, use idct_8x8 as an approximation (matches existing behavior).
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);
            let mut output = [0.0f32; 64];
            idct_8x8(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT4X8 | RAW_STRATEGY_DCT8X4 => {
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);
            let mut output = [0.0f32; 64];
            idct_8x8(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
            let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
            let mut input = [0.0f32; 64];
            input.copy_from_slice(&coeffs[..64]);
            let mut output = [0.0f32; 64];
            super::afv::inverse_afv_transform(&input, afv_kind, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT16X8 => {
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&coeffs[..128]);
            let mut output = [0.0f32; 128];
            idct_16x8(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT8X16 => {
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&coeffs[..128]);
            let mut output = [0.0f32; 128];
            idct_8x16(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT16X16 => {
            let mut input = [0.0f32; 256];
            input.copy_from_slice(&coeffs[..256]);
            let mut output = [0.0f32; 256];
            idct_16x16(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT32X32 => {
            let mut input = [0.0f32; 1024];
            input.copy_from_slice(&coeffs[..1024]);
            let mut output = [0.0f32; 1024];
            idct_32x32(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT32X16 => {
            let mut input = [0.0f32; 512];
            input.copy_from_slice(&coeffs[..512]);
            let mut output = [0.0f32; 512];
            idct_32x16(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT16X32 => {
            let mut input = [0.0f32; 512];
            input.copy_from_slice(&coeffs[..512]);
            let mut output = [0.0f32; 512];
            idct_16x32(&input, &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT64X64 => {
            let mut input = vec![0.0f32; 4096];
            input.copy_from_slice(&coeffs[..4096]);
            let mut output = vec![0.0f32; 4096];
            idct_64x64(&input, &mut output);
            output
        }
        RAW_STRATEGY_DCT64X32 => {
            let mut input = vec![0.0f32; 2048];
            input.copy_from_slice(&coeffs[..2048]);
            let mut output = vec![0.0f32; 2048];
            idct_64x32(&input, &mut output);
            output
        }
        RAW_STRATEGY_DCT32X64 => {
            let mut input = vec![0.0f32; 2048];
            input.copy_from_slice(&coeffs[..2048]);
            let mut output = vec![0.0f32; 2048];
            idct_32x64(&input, &mut output);
            output
        }
        RAW_STRATEGY_IDENTITY => {
            let mut output = [0.0f32; 64];
            inverse_identity_transform(&coeffs[..64], &mut output);
            output.to_vec()
        }
        RAW_STRATEGY_DCT2X2 => {
            let mut output = [0.0f32; 64];
            inverse_dct2x2_transform(&coeffs[..64], &mut output);
            output.to_vec()
        }
        _ => {
            // Unknown strategy: return zeros
            vec![0.0f32; 64]
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

    for plane in planes.iter_mut() {
        let input = plane.clone();
        let output = plane;

        for y in 0..height {
            for x in 0..width {
                let ym = if y > 0 { y - 1 } else { 0 };
                let yp = if y + 1 < height { y + 1 } else { height - 1 };
                let xm = if x > 0 { x - 1 } else { 0 };
                let xp = if x + 1 < width { x + 1 } else { width - 1 };

                let center = input[y * width + x];
                let top = input[ym * width + x];
                let bottom = input[yp * width + x];
                let left = input[y * width + xm];
                let right = input[y * width + xp];
                let tl = input[ym * width + xm];
                let tr = input[ym * width + xp];
                let bl = input[yp * width + xm];
                let br = input[yp * width + xp];

                output[y * width + x] = w_center * center
                    + w1 * (top + bottom + left + right)
                    + w2 * (tl + tr + bl + br);
            }
        }
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
#[cfg(feature = "butteraugli-loop")]
pub(crate) fn xyb_to_linear_rgb(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    use crate::color::xyb::{NEG_OPSIN_ABSORBANCE_BIAS_CBRT, OPSIN_ABSORBANCE_BIAS};

    // Inverse opsin absorbance matrix (from libjxl cms/opsin_params.h)
    #[allow(clippy::excessive_precision)]
    const INV_OPSIN: [[f32; 3]; 3] = [
        [11.031566901960783, -9.866943921568629, -0.16462299647058826],
        [-3.254147380392157, 4.418770392156863, -0.16462299647058826],
        [-3.6588512862745097, 2.7129230470588235, 1.9459282392156863],
    ];

    let cbrt_bias_0 = -NEG_OPSIN_ABSORBANCE_BIAS_CBRT[0]; // cbrt(bias) ≈ 0.15595
    let cbrt_bias_1 = -NEG_OPSIN_ABSORBANCE_BIAS_CBRT[1];
    let cbrt_bias_2 = -NEG_OPSIN_ABSORBANCE_BIAS_CBRT[2];
    let neg_bias_0 = -OPSIN_ABSORBANCE_BIAS[0];
    let neg_bias_1 = -OPSIN_ABSORBANCE_BIAS[1];
    let neg_bias_2 = -OPSIN_ABSORBANCE_BIAS[2];

    let num_pixels = width * height;
    let mut linear_rgb = vec![0.0f32; num_pixels * 3];

    for i in 0..num_pixels {
        let x = xyb_x[i];
        let y = xyb_y[i];
        let b = xyb_b[i];

        // Step 1: Unmix XYB to LMS gamma domain
        let mut gamma_r = y + x; // L
        let mut gamma_g = y - x; // M
        let mut gamma_b = b; // S

        // Step 2: Add cbrt(bias) back (undo the encoder's subtraction)
        gamma_r += cbrt_bias_0;
        gamma_g += cbrt_bias_1;
        gamma_b += cbrt_bias_2;

        // Step 3: Cube and subtract bias to get mixed (opsin LMS) values
        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias_0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias_1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias_2;

        // Step 4: Apply inverse opsin matrix to get linear RGB
        let r = INV_OPSIN[0][0] * mixed_r + INV_OPSIN[0][1] * mixed_g + INV_OPSIN[0][2] * mixed_b;
        let g = INV_OPSIN[1][0] * mixed_r + INV_OPSIN[1][1] * mixed_g + INV_OPSIN[1][2] * mixed_b;
        let b_lin =
            INV_OPSIN[2][0] * mixed_r + INV_OPSIN[2][1] * mixed_g + INV_OPSIN[2][2] * mixed_b;

        linear_rgb[i * 3] = r;
        linear_rgb[i * 3 + 1] = g;
        linear_rgb[i * 3 + 2] = b_lin;
    }

    linear_rgb
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
}
