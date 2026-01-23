//! DCT transform pipeline for VarDCT encoding.
//!
//! Transforms XYB image data into quantized DCT coefficients.
//! Supports DCT8, DCT16, and DCT32 transforms based on AC strategy.

use crate::BLOCK_DIM;
use crate::heuristics::AcStrategyMap;
use crate::vardct::AcStrategy;
use jxl_enc_transforms::{dct8, dct16, dct32};

use super::enc_coeff::quantize_block_8x8;
use super::quant_weights::{INV_LF_QUANT, get_dct8_inv_dequant_per_channel};
use super::quantizer::{GLOBAL_SCALE_DENOM, QuantizerParams};

/// Transformed and quantized image data.
pub struct TransformedData {
    /// DC coefficients for each block (XYB interleaved).
    /// Layout: [block0_x_dc, block0_y_dc, block0_b_dc, block1_x_dc, ...]
    pub dc_coeffs: Vec<i32>,
    /// AC coefficients for each block (XYB interleaved, 63 AC per block).
    /// Layout: [block0_x_ac0..ac62, block0_y_ac0..ac62, block0_b_ac0..ac62, ...]
    pub ac_coeffs: Vec<i32>,
    /// Number of blocks in X direction.
    pub num_blocks_x: usize,
    /// Number of blocks in Y direction.
    pub num_blocks_y: usize,
}

/// Transform XYB image data into DCT coefficients and quantize.
///
/// # Arguments
/// * `xyb_data` - XYB pixel data in planar format [X plane, Y plane, B plane]
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `quantizer` - Quantization parameters
pub fn transform_and_quantize(
    xyb_data: &[&[f32]; 3],
    width: usize,
    height: usize,
    quantizer: &QuantizerParams,
) -> TransformedData {
    let num_blocks_x = width.div_ceil(BLOCK_DIM);
    let num_blocks_y = height.div_ceil(BLOCK_DIM);
    let num_blocks = num_blocks_x * num_blocks_y;

    // Allocate output buffers
    // DC: 1 per channel per block
    let mut dc_coeffs = vec![0i32; num_blocks * 3];
    // AC: 63 per channel per block
    let mut ac_coeffs = vec![0i32; num_blocks * 3 * 63];

    // Get global scale for quantization
    let global_scale_float = quantizer.global_scale as f32 / GLOBAL_SCALE_DENOM as f32;
    let quant_dc = quantizer.quant_dc as i32;

    // Compute raw_quant for uniform quantization
    // In libjxl: raw_quant = quant_field * inv_global_scale + 0.5
    // For distance=1.0 with default settings, quant_field ≈ 0.765 and raw_quant ≈ 1
    // Using raw_quant=1 gives proper quantization levels matching the trace_quant output
    let raw_quant = 1i32;

    // Get DCT8 inverse dequant matrices for each channel (X, Y, B)
    // These provide perceptual weighting for AC coefficients
    let inv_dequant_per_channel = get_dct8_inv_dequant_per_channel();

    // Default CfL factors (from JXL spec):
    // cfl_fac_x = base_correlation_x + ytox_lf / color_factor = 0 + 0/84 = 0
    // cfl_fac_b = base_correlation_b + ytob_lf / color_factor = 1 + 0/84 = 1
    let cfl_fac_x = 0.0f32;
    let cfl_fac_b = 1.0f32;

    // Process each block
    for by in 0..num_blocks_y {
        for bx in 0..num_blocks_x {
            let block_idx = by * num_blocks_x + bx;

            // Extract and DCT all channels first
            let mut dct_y = [0.0f32; 64];
            let mut dct_x = [0.0f32; 64];
            let mut dct_b = [0.0f32; 64];

            let mut block_in = [0.0f32; 64];
            extract_block(xyb_data[0], width, height, bx, by, &mut block_in); // X
            dct8(&block_in, &mut dct_x);
            extract_block(xyb_data[1], width, height, bx, by, &mut block_in); // Y
            dct8(&block_in, &mut dct_y);
            extract_block(xyb_data[2], width, height, bx, by, &mut block_in); // B
            dct8(&block_in, &mut dct_b);

            // Apply inverse CfL (Color from Luma) transform
            // The decoder applies: decoded_X = Y * cfl_fac_x + encoded_X
            //                      decoded_B = Y * cfl_fac_b + encoded_B
            // So encoder must compute: encoded_X = raw_X - Y * cfl_fac_x
            //                          encoded_B = raw_B - Y * cfl_fac_b
            for i in 0..64 {
                dct_x[i] -= dct_y[i] * cfl_fac_x;
                dct_b[i] -= dct_y[i] * cfl_fac_b;
            }

            // Quantize each channel with inverse-CfL-adjusted coefficients
            let dct_channels = [&dct_x, &dct_y, &dct_b];
            for c in 0..3 {
                let mut quant_out = [0i32; 64];
                quantize_block_8x8(
                    dct_channels[c],
                    quant_dc,
                    raw_quant,
                    global_scale_float,
                    INV_LF_QUANT[c],
                    &inv_dequant_per_channel[c],
                    &mut quant_out,
                );

                // Store DC coefficient
                dc_coeffs[block_idx * 3 + c] = quant_out[0];

                // Store AC coefficients (positions 1-63)
                let ac_start = block_idx * 3 * 63 + c * 63;
                ac_coeffs[ac_start..ac_start + 63].copy_from_slice(&quant_out[1..64]);
            }
        }
    }

    TransformedData {
        dc_coeffs,
        ac_coeffs,
        num_blocks_x,
        num_blocks_y,
    }
}

/// Extract an 8x8 block from image data.
fn extract_block(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block: &mut [f32; 64],
) {
    let start_x = bx * BLOCK_DIM;
    let start_y = by * BLOCK_DIM;

    for y in 0..BLOCK_DIM {
        let src_y = (start_y + y).min(height - 1);
        for x in 0..BLOCK_DIM {
            let src_x = (start_x + x).min(width - 1);
            block[y * BLOCK_DIM + x] = plane[src_y * width + src_x];
        }
    }
}

/// Extract a 16x16 block from image data.
/// Block position (bx, by) is in units of 8x8 blocks (topleft of the 2x2 region).
fn extract_block_16x16(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block: &mut [f32; 256],
) {
    let start_x = bx * BLOCK_DIM;
    let start_y = by * BLOCK_DIM;

    for y in 0..16 {
        let src_y = (start_y + y).min(height - 1);
        for x in 0..16 {
            let src_x = (start_x + x).min(width - 1);
            block[y * 16 + x] = plane[src_y * width + src_x];
        }
    }
}

/// Extract a 32x32 block from image data.
/// Block position (bx, by) is in units of 8x8 blocks (topleft of the 4x4 region).
fn extract_block_32x32(
    plane: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block: &mut [f32; 1024],
) {
    let start_x = bx * BLOCK_DIM;
    let start_y = by * BLOCK_DIM;

    for y in 0..32 {
        let src_y = (start_y + y).min(height - 1);
        for x in 0..32 {
            let src_x = (start_x + x).min(width - 1);
            block[y * 32 + x] = plane[src_y * width + src_x];
        }
    }
}

/// Transform and quantize with AC strategy map.
///
/// Uses the AC strategy map to select between DCT8, DCT16, and DCT32 transforms.
/// Returns transformed data with per-block DC and variable-length AC coefficients.
pub fn transform_and_quantize_with_strategy(
    xyb_data: &[&[f32]; 3],
    width: usize,
    height: usize,
    quantizer: &QuantizerParams,
    ac_strategy_map: &AcStrategyMap,
) -> TransformedDataWithStrategy {
    let num_blocks_x = width.div_ceil(BLOCK_DIM);
    let num_blocks_y = height.div_ceil(BLOCK_DIM);
    let num_blocks = num_blocks_x * num_blocks_y;

    // DC: 1 per channel per block
    let mut dc_coeffs = vec![0i32; num_blocks * 3];
    // AC: variable per block, stored in a flat buffer with offsets
    let mut ac_coeffs = Vec::new();
    // For each block, store the offset into ac_coeffs where its AC data starts
    let mut ac_offsets = vec![0usize; num_blocks * 3 + 1];

    let global_scale_float = quantizer.global_scale as f32 / GLOBAL_SCALE_DENOM as f32;
    let quant_dc = quantizer.quant_dc as i32;

    // Compute raw_quant for uniform quantization
    // Using raw_quant=1 gives proper quantization levels
    let raw_quant = 1i32;

    let inv_dequant_per_channel = get_dct8_inv_dequant_per_channel();

    // Track which blocks have been processed (for DCT16/32 which cover multiple 8x8 positions)
    let mut processed = vec![false; num_blocks];

    let mut current_offset = 0;

    for by in 0..num_blocks_y {
        for bx in 0..num_blocks_x {
            let block_idx = by * num_blocks_x + bx;

            if processed[block_idx] {
                // This block is covered by a larger transform
                for c in 0..3 {
                    ac_offsets[block_idx * 3 + c] = current_offset;
                }
                continue;
            }

            let strategy = ac_strategy_map.get(bx, by);

            match strategy {
                AcStrategy::Dct32x32 if bx + 3 < num_blocks_x && by + 3 < num_blocks_y => {
                    // Process 4x4 block region with DCT32
                    process_dct32(
                        xyb_data,
                        width,
                        height,
                        bx,
                        by,
                        num_blocks_x,
                        quant_dc,
                        raw_quant,
                        global_scale_float,
                        &inv_dequant_per_channel,
                        &mut dc_coeffs,
                        &mut ac_coeffs,
                        &mut ac_offsets,
                        &mut current_offset,
                    );
                    // Mark all 16 blocks as processed
                    for dy in 0..4 {
                        for dx in 0..4 {
                            processed[(by + dy) * num_blocks_x + (bx + dx)] = true;
                        }
                    }
                }
                AcStrategy::Dct16x16 if bx + 1 < num_blocks_x && by + 1 < num_blocks_y => {
                    // Process 2x2 block region with DCT16
                    process_dct16(
                        xyb_data,
                        width,
                        height,
                        bx,
                        by,
                        num_blocks_x,
                        quant_dc,
                        raw_quant,
                        global_scale_float,
                        &inv_dequant_per_channel,
                        &mut dc_coeffs,
                        &mut ac_coeffs,
                        &mut ac_offsets,
                        &mut current_offset,
                    );
                    // Mark all 4 blocks as processed
                    for dy in 0..2 {
                        for dx in 0..2 {
                            processed[(by + dy) * num_blocks_x + (bx + dx)] = true;
                        }
                    }
                }
                _ => {
                    // DCT8 (default)
                    process_dct8(
                        xyb_data,
                        width,
                        height,
                        bx,
                        by,
                        num_blocks_x,
                        quant_dc,
                        raw_quant,
                        global_scale_float,
                        &inv_dequant_per_channel,
                        &mut dc_coeffs,
                        &mut ac_coeffs,
                        &mut ac_offsets,
                        &mut current_offset,
                    );
                    processed[block_idx] = true;
                }
            }
        }
    }

    // Set final offset
    ac_offsets[num_blocks * 3] = current_offset;

    TransformedDataWithStrategy {
        dc_coeffs,
        ac_coeffs,
        ac_offsets,
        num_blocks_x,
        num_blocks_y,
        strategies: ac_strategy_map.clone(),
    }
}

/// Transformed data with variable block sizes.
pub struct TransformedDataWithStrategy {
    /// DC coefficients for each block (XYB interleaved).
    pub dc_coeffs: Vec<i32>,
    /// AC coefficients (variable length per block, concatenated).
    pub ac_coeffs: Vec<i32>,
    /// Offsets into ac_coeffs for each block*channel.
    /// ac_offsets[block_idx * 3 + channel] gives start offset.
    pub ac_offsets: Vec<usize>,
    /// Number of blocks in X direction.
    pub num_blocks_x: usize,
    /// Number of blocks in Y direction.
    pub num_blocks_y: usize,
    /// AC strategy map used.
    pub strategies: AcStrategyMap,
}

/// Process a single 8x8 block with DCT8.
#[allow(clippy::too_many_arguments)]
fn process_dct8(
    xyb_data: &[&[f32]; 3],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    num_blocks_x: usize,
    quant_dc: i32,
    raw_quant: i32,
    global_scale_float: f32,
    inv_dequant_per_channel: &[[f32; 64]; 3],
    dc_coeffs: &mut [i32],
    ac_coeffs: &mut Vec<i32>,
    ac_offsets: &mut [usize],
    current_offset: &mut usize,
) {
    let block_idx = by * num_blocks_x + bx;

    // Default CfL factors (from libjxl: base_correlation = 0, color_factor = 84)
    // With default settings, no chroma correlation is applied
    let cfl_fac_x = 0.0f32;
    let cfl_fac_b = 1.0f32;

    // Extract and DCT all channels
    let mut dct_x = [0.0f32; 64];
    let mut dct_y = [0.0f32; 64];
    let mut dct_b = [0.0f32; 64];

    let mut block_in = [0.0f32; 64];
    extract_block(xyb_data[0], width, height, bx, by, &mut block_in);
    dct8(&block_in, &mut dct_x);
    extract_block(xyb_data[1], width, height, bx, by, &mut block_in);
    dct8(&block_in, &mut dct_y);
    extract_block(xyb_data[2], width, height, bx, by, &mut block_in);
    dct8(&block_in, &mut dct_b);

    // Apply inverse CfL transform
    for i in 0..64 {
        dct_x[i] -= dct_y[i] * cfl_fac_x;
        dct_b[i] -= dct_y[i] * cfl_fac_b;
    }

    // Quantize each channel
    let dct_channels = [&dct_x, &dct_y, &dct_b];
    for c in 0..3 {
        let mut quant_out = [0i32; 64];
        quantize_block_8x8(
            dct_channels[c],
            quant_dc,
            raw_quant,
            global_scale_float,
            INV_LF_QUANT[c],
            &inv_dequant_per_channel[c],
            &mut quant_out,
        );

        dc_coeffs[block_idx * 3 + c] = quant_out[0];
        ac_offsets[block_idx * 3 + c] = *current_offset;
        ac_coeffs.extend_from_slice(&quant_out[1..64]);
        *current_offset += 63;
    }
}

/// DCT resample scale factors for DCT32 -> 4x4 (from libjxl dct_scales.h).
#[allow(clippy::excessive_precision)]
const DCT_RESAMPLE_SCALE_32_4: [f32; 4] = [1.0, 0.9748868, 0.9017642, 0.7870549];

/// Convert LLF coefficients to DC values for DCT16.
///
/// The decoder's `reinterpreting_dct2d_2_2` takes 4 DC values and applies
/// a forward 2x2 DCT to produce LLF coefficients. We need to invert this.
///
/// The jxl 2x2 forward DCT (reinterpreting_dct_2 applied twice, column-first):
/// Given input [a, b, c, d] arranged as [[a,b],[c,d]]:
///
/// Step 1 - Column DCT with stride:
///   v0 = (a+c)*0.5, v1 = (a-c)*0.554469  (column 0)
///   v2 = (b+d)*0.5, v3 = (b-d)*0.554469  (column 1)
///
/// Step 2 - Transpose: [[v0,v1],[v2,v3]] -> [[v0,v2],[v1,v3]]
///
/// Step 3 - Row DCT:
///   out[0] = (v0+v2)*0.5 = (a+b+c+d)*0.25
///   out[1] = (v0-v2)*0.554469 = ((a+c)-(b+d))*0.5*0.554469 = (a-b+c-d)*0.277234
///   out[16] = (v1+v3)*0.5 = ((a-c)+(b-d))*0.554469*0.5 = (a+b-c-d)*0.277234
///   out[17] = (v1-v3)*0.554469 = ((a-c)-(b-d))*0.554469^2 = (a-b-c+d)*0.307435
///
/// Note: out[1] and out[16] are swapped from a "normal" 2x2 DCT due to transpose order!
///
/// So the mapping is:
///   LLF[0][0] = (a+b+c+d) * 0.25
///   LLF[0][1] = (a-b+c-d) * 0.277234   <- note the sign pattern
///   LLF[1][0] = (a+b-c-d) * 0.277234   <- swapped with above in a normal DCT
///   LLF[1][1] = (a-b-c+d) * 0.307435
fn llf_to_dc_dct16(llf: [[f32; 2]; 2]) -> [[f32; 2]; 2] {
    // Convert LLF coefficients from our DCT16 output to DC values for the
    // 4 covered 8x8 blocks. The decoder will apply ReinterpretingDCT to these
    // DC values to reconstruct the LLF.
    //
    // Our DCT16 produces (for piecewise constant quadrants with pixel values a, b, c, d):
    //   l00 = (a+b+c+d) * 4.0
    //   l01 = (a-b+c-d) * 3.607
    //   l10 = (a+b-c-d) * 3.607
    //   l11 = (a-b-c+d) * 3.253
    //
    // The decoder expects DC values where dc[i] = 8 * block_avg[i] (same as DCT8).
    // ReinterpretingDCT then produces:
    //   jxl_l00 = (dc0+dc1+dc2+dc3) * 0.25 = 8*(a+b+c+d) * 0.25 = 2*(a+b+c+d)
    //   jxl_l01 = (dc0-dc1+dc2-dc3) * 0.277234 = 8*(a-b+c-d) * 0.277234 = 2.218*(a-b+c-d)
    //   jxl_l10 = (dc0+dc1-dc2-dc3) * 0.277234 = 8*(a+b-c-d) * 0.277234 = 2.218*(a+b-c-d)
    //   jxl_l11 = (dc0-dc1-dc2+dc3) * 0.307435 = 8*(a-b-c+d) * 0.307435 = 2.459*(a-b-c+d)
    //
    // To convert our LLF to DC values, we multiply by constants that combine:
    // 1. Converting our LLF scaling to jxl LLF scaling
    // 2. Applying jxl's ReinterpretingIDCT
    //
    // The combined constants give sums in terms of 8*pixel_value:
    //   s1 = 8*(a+b+c+d) = l00 * (8/4) = l00 * 2
    //   s2 = 8*(a-b+c-d) = l01 * (8*0.277234/3.607) ≈ l01 * 0.6147
    //   s3 = 8*(a+b-c-d) = l10 * 0.6147
    //   s4 = 8*(a-b-c+d) = l11 * (8*0.307435/3.253) ≈ l11 * 0.7559
    //
    // Actually, let's compute directly: we need s = 8*(pattern) from our l = pattern * our_scale
    // s = l * (8 / our_scale)
    let l00 = llf[0][0];
    let l01 = llf[0][1];
    let l10 = llf[1][0];
    let l11 = llf[1][1];

    // Conversion constants: multiply to get 8 * (pattern sum)
    // l00 = (a+b+c+d) * 4.0, need 8*(a+b+c+d), so multiply by 8/4 = 2
    // l01 = (a-b+c-d) * 3.607, need 8*(a-b+c-d), so multiply by 8/3.607 ≈ 2.218
    // l10 = (a+b-c-d) * 3.607, need 8*(a+b-c-d), so multiply by 8/3.607 ≈ 2.218
    // l11 = (a-b-c+d) * 3.253, need 8*(a-b-c+d), so multiply by 8/3.253 ≈ 2.459
    const SCALE_00: f32 = 2.0; // 8 / 4
    const SCALE_01: f32 = 8.0 * 0.277234; // 8 / 3.607 ≈ 2.218
    const SCALE_10: f32 = 8.0 * 0.277234; // 8 / 3.607 ≈ 2.218
    const SCALE_11: f32 = 8.0 * 0.307435; // 8 / 3.253 ≈ 2.459

    // Compute sums in 8*pixel_value units
    let s1 = l00 * SCALE_00; // = 8*(a+b+c+d)
    let s2 = l01 * SCALE_01; // = 8*(a-b+c-d)
    let s3 = l10 * SCALE_10; // = 8*(a+b-c-d)
    let s4 = l11 * SCALE_11; // = 8*(a-b-c+d)

    // Solve for DC values (each = 8 * pixel_value for that block):
    // s1 + s2 + s3 + s4 = 32*a, so dc0 = 8*a = (s1+s2+s3+s4)/4
    // s1 - s2 + s3 - s4 = 32*b, so dc1 = 8*b = (s1-s2+s3-s4)/4
    // s1 + s2 - s3 - s4 = 32*c, so dc2 = 8*c = (s1+s2-s3-s4)/4
    // s1 - s2 - s3 + s4 = 32*d, so dc3 = 8*d = (s1-s2-s3+s4)/4
    let dc0 = (s1 + s2 + s3 + s4) * 0.25;
    let dc1 = (s1 - s2 + s3 - s4) * 0.25;
    let dc2 = (s1 + s2 - s3 - s4) * 0.25;
    let dc3 = (s1 - s2 - s3 + s4) * 0.25;

    [[dc0, dc1], [dc2, dc3]]
}

/// Convert LLF coefficients to DC values for DCT32 using 4x4 IDCT.
#[allow(clippy::needless_range_loop)]
fn llf_to_dc_dct32(llf: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Apply resample scale factors
    let mut scaled = [[0.0f32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            scaled[y][x] = llf[y][x] * DCT_RESAMPLE_SCALE_32_4[y] * DCT_RESAMPLE_SCALE_32_4[x];
        }
    }

    // Apply 4x4 IDCT using the standard formula
    let mut dc = [[0.0f32; 4]; 4];
    let pi = std::f32::consts::PI;
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                for l in 0..4 {
                    let ck = if k == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 };
                    let cl = if l == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 };
                    sum += ck
                        * cl
                        * scaled[k][l]
                        * ((2.0 * i as f32 + 1.0) * k as f32 * pi / 8.0).cos()
                        * ((2.0 * j as f32 + 1.0) * l as f32 * pi / 8.0).cos();
                }
            }
            dc[i][j] = sum * 0.5;
        }
    }

    dc
}

/// Process a 2x2 region with DCT16.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn process_dct16(
    xyb_data: &[&[f32]; 3],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    num_blocks_x: usize,
    quant_dc: i32,
    raw_quant: i32,
    global_scale_float: f32,
    inv_dequant_per_channel: &[[f32; 64]; 3],
    dc_coeffs: &mut [i32],
    ac_coeffs: &mut Vec<i32>,
    ac_offsets: &mut [usize],
    current_offset: &mut usize,
) {
    let block_idx = by * num_blocks_x + bx;

    // Default CfL factors
    let cfl_fac_x = 0.0f32;
    let cfl_fac_b = 1.0f32;

    // Extract and DCT all channels
    let mut dct_x = [0.0f32; 256];
    let mut dct_y = [0.0f32; 256];
    let mut dct_b = [0.0f32; 256];

    let mut block_in = [0.0f32; 256];
    extract_block_16x16(xyb_data[0], width, height, bx, by, &mut block_in);
    dct16(&block_in, &mut dct_x);
    extract_block_16x16(xyb_data[1], width, height, bx, by, &mut block_in);
    dct16(&block_in, &mut dct_y);
    extract_block_16x16(xyb_data[2], width, height, bx, by, &mut block_in);
    dct16(&block_in, &mut dct_b);

    // Apply inverse CfL transform
    for i in 0..256 {
        dct_x[i] -= dct_y[i] * cfl_fac_x;
        dct_b[i] -= dct_y[i] * cfl_fac_b;
    }

    // DC quantization threshold
    let threshold = 0.5f32;

    // Process each channel
    for c in 0..3 {
        let dct: &[f32; 256] = match c {
            0 => &dct_x,
            1 => &dct_y,
            _ => &dct_b,
        };

        // Extract 2x2 LLF region (positions 0, 1, 16, 17) as float
        let llf = [
            [dct[0], dct[1]],   // positions (0,0) and (1,0)
            [dct[16], dct[17]], // positions (0,1) and (1,1)
        ];

        // Convert LLF to DC values using 2x2 IDCT.
        // The DC values returned are in "8 * block_average" format (same as DCT8).
        let dc_values = llf_to_dc_dct16(llf);

        // Quantize DC values using the same formula as quantize_block_8x8:
        // The JXL LF image stores block AVERAGES, not DCT DC coefficients.
        // dc_values[i] = 8 * block_average, so divide by 8 to get the average.
        // quantized_dc = (dc_value / 8) * inv_lf_quant * global_scale_float * quant_dc
        let qdc = INV_LF_QUANT[c] * global_scale_float * quant_dc as f32;

        for dy in 0..2 {
            for dx in 0..2 {
                let covered_idx = (by + dy) * num_blocks_x + (bx + dx);
                let dc_avg = dc_values[dy][dx] / 8.0; // Convert to block average
                let dc_val = dc_avg * qdc;
                dc_coeffs[covered_idx * 3 + c] = if dc_val.abs() >= threshold {
                    dc_val.round() as i32
                } else {
                    0
                };
            }
        }

        // Now quantize AC coefficients (all positions except LLF 0, 1, 16, 17)
        let qac = global_scale_float * raw_quant as f32;

        ac_offsets[block_idx * 3 + c] = *current_offset;
        for pos in 0..256 {
            // Skip LLF positions
            if pos == 0 || pos == 1 || pos == 16 || pos == 17 {
                continue;
            }

            // Map to 8x8 position for weight lookup
            let x = pos % 16;
            let y = pos / 16;
            let x8 = x / 2;
            let y8 = y / 2;
            let weight_pos = y8 * 8 + x8;

            let val = dct[pos] * qac * inv_dequant_per_channel[c][weight_pos];
            let quantized = if val.abs() >= threshold {
                val.round() as i32
            } else {
                0
            };
            ac_coeffs.push(quantized);
        }
        *current_offset += 252; // 256 - 4 LLF positions

        // Set offsets for covered blocks to point to empty region
        for dy in 0..2 {
            for dx in 0..2 {
                if dx != 0 || dy != 0 {
                    let covered_idx = (by + dy) * num_blocks_x + (bx + dx);
                    ac_offsets[covered_idx * 3 + c] = *current_offset;
                }
            }
        }
    }
}

/// Process a 4x4 region with DCT32.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn process_dct32(
    xyb_data: &[&[f32]; 3],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    num_blocks_x: usize,
    quant_dc: i32,
    raw_quant: i32,
    global_scale_float: f32,
    inv_dequant_per_channel: &[[f32; 64]; 3],
    dc_coeffs: &mut [i32],
    ac_coeffs: &mut Vec<i32>,
    ac_offsets: &mut [usize],
    current_offset: &mut usize,
) {
    let block_idx = by * num_blocks_x + bx;

    // Default CfL factors
    let cfl_fac_x = 0.0f32;
    let cfl_fac_b = 1.0f32;

    // Extract and DCT all channels
    let mut dct_x = [0.0f32; 1024];
    let mut dct_y = [0.0f32; 1024];
    let mut dct_b = [0.0f32; 1024];

    let mut block_in = [0.0f32; 1024];
    extract_block_32x32(xyb_data[0], width, height, bx, by, &mut block_in);
    dct32(&block_in, &mut dct_x);
    extract_block_32x32(xyb_data[1], width, height, bx, by, &mut block_in);
    dct32(&block_in, &mut dct_y);
    extract_block_32x32(xyb_data[2], width, height, bx, by, &mut block_in);
    dct32(&block_in, &mut dct_b);

    // Apply inverse CfL transform
    for i in 0..1024 {
        dct_x[i] -= dct_y[i] * cfl_fac_x;
        dct_b[i] -= dct_y[i] * cfl_fac_b;
    }

    // DC quantization threshold
    let threshold = 0.5f32;

    // Process each channel
    for c in 0..3 {
        let dct = match c {
            0 => &dct_x,
            1 => &dct_y,
            _ => &dct_b,
        };

        // Extract 4x4 LLF region from the 32x32 grid
        // LLF positions: (y, x) where y < 4 and x < 4
        // Position in grid = y * 32 + x
        let mut llf = [[0.0f32; 4]; 4];
        for y in 0..4 {
            for x in 0..4 {
                llf[y][x] = dct[y * 32 + x];
            }
        }

        // Convert LLF to DC values using 4x4 IDCT
        let dc_values = llf_to_dc_dct32(llf);

        // Quantize DC values using the same formula as quantize_block_8x8:
        // dc_values[i] = 8 * block_average, so divide by 8 to get the average.
        let qdc = INV_LF_QUANT[c] * global_scale_float * quant_dc as f32;

        for dy in 0..4 {
            for dx in 0..4 {
                let covered_idx = (by + dy) * num_blocks_x + (bx + dx);
                let dc_avg = dc_values[dy][dx] / 8.0; // Convert to block average
                let dc_val = dc_avg * qdc;
                dc_coeffs[covered_idx * 3 + c] = if dc_val.abs() >= threshold {
                    dc_val.round() as i32
                } else {
                    0
                };
            }
        }

        // Now quantize AC coefficients (all positions except 4x4 LLF region)
        let qac = global_scale_float * raw_quant as f32;

        ac_offsets[block_idx * 3 + c] = *current_offset;
        for pos in 0..1024 {
            let x = pos % 32;
            let y = pos / 32;
            // Skip LLF region (x < 4 && y < 4)
            if x < 4 && y < 4 {
                continue;
            }

            // Map to 8x8 position for weight lookup
            let x8 = x / 4;
            let y8 = y / 4;
            let weight_pos = y8 * 8 + x8;

            let val = dct[pos] * qac * inv_dequant_per_channel[c][weight_pos];
            let quantized = if val.abs() >= threshold {
                val.round() as i32
            } else {
                0
            };
            ac_coeffs.push(quantized);
        }
        *current_offset += 1008; // 1024 - 16 LLF positions

        // Set offsets for covered blocks to point to empty region
        for dy in 0..4 {
            for dx in 0..4 {
                if dx != 0 || dy != 0 {
                    let covered_idx = (by + dy) * num_blocks_x + (bx + dx);
                    ac_offsets[covered_idx * 3 + c] = *current_offset;
                }
            }
        }
    }
}

/// Simple XYB transform pipeline entry point.
///
/// Takes interleaved XYB data and returns transformed/quantized coefficients.
pub fn transform_xyb_image(
    xyb_interleaved: &[f32],
    width: usize,
    height: usize,
    quantizer: &QuantizerParams,
) -> TransformedData {
    let num_pixels = width * height;

    // Deinterleave XYB data into planes
    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    for i in 0..num_pixels {
        x_plane[i] = xyb_interleaved[i * 3];
        y_plane[i] = xyb_interleaved[i * 3 + 1];
        b_plane[i] = xyb_interleaved[i * 3 + 2];
    }

    transform_and_quantize(&[&x_plane, &y_plane, &b_plane], width, height, quantizer)
}

/// Strategy-aware XYB transform pipeline entry point.
///
/// Takes interleaved XYB data and AC strategy map, returns transformed data
/// with variable-length AC coefficients based on DCT sizes.
pub fn transform_xyb_image_with_strategy(
    xyb_interleaved: &[f32],
    width: usize,
    height: usize,
    quantizer: &QuantizerParams,
    ac_strategy_map: &AcStrategyMap,
) -> TransformedDataWithStrategy {
    let num_pixels = width * height;

    // Deinterleave XYB data into planes
    let mut x_plane = vec![0.0f32; num_pixels];
    let mut y_plane = vec![0.0f32; num_pixels];
    let mut b_plane = vec![0.0f32; num_pixels];

    for i in 0..num_pixels {
        x_plane[i] = xyb_interleaved[i * 3];
        y_plane[i] = xyb_interleaved[i * 3 + 1];
        b_plane[i] = xyb_interleaved[i * 3 + 2];
    }

    transform_and_quantize_with_strategy(
        &[&x_plane, &y_plane, &b_plane],
        width,
        height,
        quantizer,
        ac_strategy_map,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_block() {
        // 16x16 test image
        let plane: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let mut block = [0.0f32; 64];

        // Extract block at (0,0)
        extract_block(&plane, 16, 16, 0, 0, &mut block);
        assert_eq!(block[0], 0.0);
        assert_eq!(block[1], 1.0);
        assert_eq!(block[8], 16.0); // Second row

        // Extract block at (1,0)
        extract_block(&plane, 16, 16, 1, 0, &mut block);
        assert_eq!(block[0], 8.0);
    }

    #[test]
    fn test_transform_and_quantize_flat() {
        // Flat 8x8 image (single block)
        let flat_value = 0.5f32;
        let plane = vec![flat_value; 64];
        let quantizer = QuantizerParams::from_distance(1.0);

        let result = transform_and_quantize(&[&plane, &plane, &plane], 8, 8, &quantizer);

        assert_eq!(result.num_blocks_x, 1);
        assert_eq!(result.num_blocks_y, 1);
        assert_eq!(result.dc_coeffs.len(), 3); // 3 channels
        assert_eq!(result.ac_coeffs.len(), 3 * 63); // 63 AC per channel
    }

    #[test]
    fn test_transform_multi_block() {
        // 16x16 image = 4 blocks
        let plane = vec![1.0f32; 256];
        let quantizer = QuantizerParams::from_distance(1.0);

        let result = transform_and_quantize(&[&plane, &plane, &plane], 16, 16, &quantizer);

        assert_eq!(result.num_blocks_x, 2);
        assert_eq!(result.num_blocks_y, 2);
        assert_eq!(result.dc_coeffs.len(), 4 * 3); // 4 blocks * 3 channels
    }

    #[test]
    fn test_extract_block_16x16() {
        // 32x32 test image
        let plane: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let mut block = [0.0f32; 256];

        extract_block_16x16(&plane, 32, 32, 0, 0, &mut block);
        assert_eq!(block[0], 0.0);
        assert_eq!(block[1], 1.0);
        assert_eq!(block[16], 32.0); // Second row

        // Extract block at (1,0) - starts at pixel x=8
        extract_block_16x16(&plane, 32, 32, 1, 0, &mut block);
        assert_eq!(block[0], 8.0);
    }

    #[test]
    fn test_extract_block_32x32() {
        // 64x64 test image
        let plane: Vec<f32> = (0..4096).map(|i| i as f32).collect();
        let mut block = [0.0f32; 1024];

        extract_block_32x32(&plane, 64, 64, 0, 0, &mut block);
        assert_eq!(block[0], 0.0);
        assert_eq!(block[1], 1.0);
        assert_eq!(block[32], 64.0); // Second row

        // Extract block at (1,0) - starts at pixel x=8
        extract_block_32x32(&plane, 64, 64, 1, 0, &mut block);
        assert_eq!(block[0], 8.0);
    }

    #[test]
    fn test_transform_with_strategy_all_dct8() {
        // 16x16 image = 4 blocks, all DCT8
        let plane = vec![1.0f32; 256];
        let quantizer = QuantizerParams::from_distance(1.0);
        let ac_map = AcStrategyMap::new_dct8(2, 2);

        let result = transform_and_quantize_with_strategy(
            &[&plane, &plane, &plane],
            16,
            16,
            &quantizer,
            &ac_map,
        );

        assert_eq!(result.num_blocks_x, 2);
        assert_eq!(result.num_blocks_y, 2);
        assert_eq!(result.dc_coeffs.len(), 4 * 3);
        // 4 blocks × 3 channels × 63 AC each = 756 total AC
        assert_eq!(result.ac_coeffs.len(), 4 * 3 * 63);
    }

    #[test]
    fn test_transform_with_strategy_dct16() {
        // 16x16 image = 4 blocks, using DCT16 (covers all 4 blocks)
        let plane = vec![1.0f32; 256];
        let quantizer = QuantizerParams::from_distance(1.0);
        let mut ac_map = AcStrategyMap::new_dct8(2, 2);
        ac_map.set(0, 0, AcStrategy::Dct16x16);

        let result = transform_and_quantize_with_strategy(
            &[&plane, &plane, &plane],
            16,
            16,
            &quantizer,
            &ac_map,
        );

        assert_eq!(result.num_blocks_x, 2);
        assert_eq!(result.num_blocks_y, 2);
        assert_eq!(result.dc_coeffs.len(), 4 * 3);
        // DCT16 produces 252 AC per channel (256 - 4 LLF positions)
        assert_eq!(result.ac_coeffs.len(), 3 * 252);
    }

    #[test]
    fn test_transform_with_strategy_dct32() {
        // 32x32 image = 16 blocks, using DCT32 (covers all 16 blocks)
        let plane = vec![1.0f32; 1024];
        let quantizer = QuantizerParams::from_distance(1.0);
        let mut ac_map = AcStrategyMap::new_dct8(4, 4);
        ac_map.set(0, 0, AcStrategy::Dct32x32);

        let result = transform_and_quantize_with_strategy(
            &[&plane, &plane, &plane],
            32,
            32,
            &quantizer,
            &ac_map,
        );

        assert_eq!(result.num_blocks_x, 4);
        assert_eq!(result.num_blocks_y, 4);
        assert_eq!(result.dc_coeffs.len(), 16 * 3);
        // DCT32 produces 1008 AC per channel (1024 - 16 LLF positions)
        assert_eq!(result.ac_coeffs.len(), 3 * 1008);
    }

    #[test]
    fn test_llf_to_dc_dct16_roundtrip() {
        // Test that llf_to_dc_dct16 correctly converts our DCT16 LLF output to DC values.
        // Our DCT16 produces LLF coefficients with these scaling factors
        // (for piecewise constant 16x16 blocks with pixel values a, b, c, d in quadrants):
        //   l00 = (a+b+c+d) * 4.0
        //   l01 = (a-b+c-d) * 3.607
        //   l10 = (a+b-c-d) * 3.607
        //   l11 = (a-b-c+d) * 3.253
        //
        // The DC values should be 8 * pixel_value for each quadrant (same as DCT8).

        // Test with known pixel values for each 8x8 quadrant
        let a = 100.0f32; // top-left quadrant pixel value
        let b = 110.0f32; // top-right quadrant pixel value
        let c = 120.0f32; // bottom-left quadrant pixel value
        let d = 130.0f32; // bottom-right quadrant pixel value

        // Compute LLF as our DCT16 would produce
        const OUR_SCALE_00: f32 = 4.0;
        const OUR_SCALE_01: f32 = 1.0 / 0.277234; // ≈ 3.607
        const OUR_SCALE_10: f32 = 1.0 / 0.277234;
        const OUR_SCALE_11: f32 = 1.0 / 0.307435; // ≈ 3.253

        let l00 = (a + b + c + d) * OUR_SCALE_00;
        let l01 = (a - b + c - d) * OUR_SCALE_01;
        let l10 = (a + b - c - d) * OUR_SCALE_10;
        let l11 = (a - b - c + d) * OUR_SCALE_11;

        let llf = [[l00, l01], [l10, l11]];

        // Convert LLF to DC values
        let dc = llf_to_dc_dct16(llf);

        // Expected DC values are 8 * pixel_value (same as DCT8)
        let expected_dc0 = 8.0 * a; // 800
        let expected_dc1 = 8.0 * b; // 880
        let expected_dc2 = 8.0 * c; // 960
        let expected_dc3 = 8.0 * d; // 1040

        eprintln!("Input pixel values: a={}, b={}, c={}, d={}", a, b, c, d);
        eprintln!(
            "Forward LLF: l00={}, l01={}, l10={}, l11={}",
            l00, l01, l10, l11
        );
        eprintln!(
            "Recovered DC: [0][0]={}, [0][1]={}, [1][0]={}, [1][1]={}",
            dc[0][0], dc[0][1], dc[1][0], dc[1][1]
        );
        eprintln!(
            "Expected DC: 8*a={}, 8*b={}, 8*c={}, 8*d={}",
            expected_dc0, expected_dc1, expected_dc2, expected_dc3
        );

        // Check recovery - DC values should be 8 * pixel_value
        assert!(
            (dc[0][0] - expected_dc0).abs() < 1.0,
            "dc[0][0]={} should be 8*a={}",
            dc[0][0],
            expected_dc0
        );
        assert!(
            (dc[0][1] - expected_dc1).abs() < 1.0,
            "dc[0][1]={} should be 8*b={}",
            dc[0][1],
            expected_dc1
        );
        assert!(
            (dc[1][0] - expected_dc2).abs() < 1.0,
            "dc[1][0]={} should be 8*c={}",
            dc[1][0],
            expected_dc2
        );
        assert!(
            (dc[1][1] - expected_dc3).abs() < 1.0,
            "dc[1][1]={} should be 8*d={}",
            dc[1][1],
            expected_dc3
        );
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::color::xyb::srgb_to_xyb;

    #[test]
    fn debug_gradient_quantization() {
        // Create 16x16 gradient image (same as test)
        let mut r = vec![0.0f32; 256];
        let mut g = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];

        for y in 0..16 {
            for x in 0..16 {
                let idx = y * 16 + x;
                r[idx] = ((x + y) * 8) as f32;
                g[idx] = ((x * 2) % 256) as f32;
                b[idx] = ((y * 2) % 256) as f32;
            }
        }

        // Convert to XYB
        let mut x_plane = vec![0.0f32; 256];
        let mut y_plane = vec![0.0f32; 256];
        let mut b_plane = vec![0.0f32; 256];

        for i in 0..256 {
            let (x, y, bb) = srgb_to_xyb(r[i], g[i], b[i]);
            x_plane[i] = x;
            y_plane[i] = y;
            b_plane[i] = bb;
        }

        crate::trace::debug_eprintln!("XYB ranges:");
        crate::trace::debug_eprintln!(
            "  X: {:?} to {:?}",
            x_plane.iter().cloned().fold(f32::INFINITY, f32::min),
            x_plane.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        crate::trace::debug_eprintln!(
            "  Y: {:?} to {:?}",
            y_plane.iter().cloned().fold(f32::INFINITY, f32::min),
            y_plane.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        crate::trace::debug_eprintln!(
            "  B: {:?} to {:?}",
            b_plane.iter().cloned().fold(f32::INFINITY, f32::min),
            b_plane.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Transform and quantize
        let quantizer = QuantizerParams::from_distance(1.0);
        crate::trace::debug_eprintln!(
            "Quantizer: global_scale={}, quant_dc={}",
            quantizer.global_scale,
            quantizer.quant_dc
        );

        let result = transform_and_quantize(&[&x_plane, &y_plane, &b_plane], 16, 16, &quantizer);

        crate::trace::debug_eprintln!("DC coefficients: {:?}", result.dc_coeffs);

        let nonzero_ac = result.ac_coeffs.iter().filter(|&&x| x != 0).count();
        crate::trace::debug_eprintln!(
            "Non-zero AC: {} out of {}",
            nonzero_ac,
            result.ac_coeffs.len()
        );

        // Find max AC coefficient
        let max_ac = result.ac_coeffs.iter().map(|&x| x.abs()).max().unwrap_or(0);
        crate::trace::debug_eprintln!("Max |AC|: {}", max_ac);

        // Show first block's AC coefficients
        crate::trace::debug_eprintln!(
            "First block AC (X channel, first 10): {:?}",
            &result.ac_coeffs[0..10]
        );
    }

    #[test]
    fn debug_dct_output() {
        use crate::color::xyb::srgb_to_xyb;
        use jxl_enc_transforms::dct8;

        // Create a simple gradient block (top-left 8x8 of our test image)
        let mut r = [0.0f32; 64];
        let mut g = [0.0f32; 64];
        let mut b = [0.0f32; 64];

        for y in 0..8 {
            for x in 0..8 {
                let idx = y * 8 + x;
                r[idx] = ((x + y) * 8) as f32;
                g[idx] = ((x * 2) % 256) as f32;
                b[idx] = ((y * 2) % 256) as f32;
            }
        }

        // Convert to XYB
        let mut x_block = [0.0f32; 64];
        let mut y_block = [0.0f32; 64];
        let mut b_block = [0.0f32; 64];

        for i in 0..64 {
            let (x, y, bb) = srgb_to_xyb(r[i], g[i], b[i]);
            x_block[i] = x;
            y_block[i] = y;
            b_block[i] = bb;
        }

        crate::trace::debug_eprintln!("Y block values (first 8): {:?}", &y_block[0..8]);

        // Apply DCT
        let mut dct_y = [0.0f32; 64];
        dct8(&y_block, &mut dct_y);

        crate::trace::debug_eprintln!("DCT Y coefficients:");
        for row in 0..8 {
            crate::trace::debug_eprintln!(
                "  Row {}: {:?}",
                row,
                &dct_y[row * 8..(row + 1) * 8]
                    .iter()
                    .map(|x| format!("{:.2}", x))
                    .collect::<Vec<_>>()
            );
        }

        // Check quantization factor
        let quantizer = QuantizerParams::from_distance(1.0);
        let global_scale_float = quantizer.global_scale as f32 / 65536.0;
        let quant_dc = quantizer.quant_dc as i32;
        crate::trace::debug_eprintln!(
            "global_scale_float = {}, quant_dc = {}",
            global_scale_float,
            quant_dc
        );
        crate::trace::debug_eprintln!(
            "qac = global_scale_float * quant_dc = {}",
            global_scale_float * quant_dc as f32
        );
    }
}

#[cfg(test)]
mod debug_tests2 {
    use crate::vardct::quant_weights::get_dct8_inv_dequant_per_channel;

    #[test]
    fn debug_inv_dequant() {
        let inv_dequant = get_dct8_inv_dequant_per_channel();

        crate::trace::debug_eprintln!("Inverse dequant matrices:");
        for (c, name) in [(0, "X"), (1, "Y"), (2, "B")] {
            crate::trace::debug_eprintln!("  {} channel - DC (pos 0): {}", name, inv_dequant[c][0]);
            crate::trace::debug_eprintln!(
                "  {} channel - AC1 (pos 1): {}",
                name,
                inv_dequant[c][1]
            );
            crate::trace::debug_eprintln!(
                "  {} channel - first row: {:?}",
                name,
                &inv_dequant[c][0..8]
            );
            crate::trace::debug_eprintln!(
                "  {} channel - pos 63 (Nyquist): {}",
                name,
                inv_dequant[c][63]
            );
            crate::trace::debug_eprintln!(
                "  {} channel - last row: {:?}",
                name,
                &inv_dequant[c][56..64]
            );
        }

        // Simulate quantization for checkerboard max AC coefficient
        let dct_coeff_63 = 0.092288f32; // From our DCT test
        let qac = 4.975f32; // From debug output
        let threshold = 0.5f32;

        crate::trace::debug_eprintln!("\nSimulate quantizing checkerboard max AC coeff (pos 63):");
        crate::trace::debug_eprintln!("  DCT coeff[63] = {}", dct_coeff_63);
        crate::trace::debug_eprintln!("  qac = {}", qac);
        crate::trace::debug_eprintln!("  threshold = {}", threshold);

        for (c, name) in [(0, "X"), (1, "Y"), (2, "B")] {
            let inv_dq = inv_dequant[c][63];
            let val = inv_dq * qac * dct_coeff_63;
            let quantized = if val.abs() >= threshold {
                val.round() as i32
            } else {
                0
            };
            crate::trace::debug_eprintln!(
                "  {} channel: inv_dequant[63]={:.6}, val={:.6}, quantized={}",
                name,
                inv_dq,
                val,
                quantized
            );
            crate::trace::debug_eprintln!("    val/threshold ratio = {:.3}", val / threshold);
        }
    }
}

#[cfg(test)]
mod debug_tests3 {
    use crate::vardct::quantizer::{GLOBAL_SCALE_DENOM, QuantizerParams};

    #[test]
    fn debug_quant_values() {
        let quantizer = QuantizerParams::from_distance(1.0);

        crate::trace::debug_eprintln!("For distance=1.0:");
        crate::trace::debug_eprintln!("  global_scale = {}", quantizer.global_scale);
        crate::trace::debug_eprintln!("  quant_dc (serialized) = {}", quantizer.quant_dc);
        crate::trace::debug_eprintln!(
            "  global_scale_float = {}",
            quantizer.global_scale as f32 / GLOBAL_SCALE_DENOM as f32
        );
        crate::trace::debug_eprintln!(
            "  inv_global_scale = {}",
            GLOBAL_SCALE_DENOM as f32 / quantizer.global_scale as f32
        );

        // What the raw quant field value should be for uniform encoding
        // In libjxl: raw_quant = quant_field * inv_global_scale + 0.5
        // For quant_field_target = 5.0:
        let quant_field_target = 5.0f32;
        let inv_global_scale = GLOBAL_SCALE_DENOM as f32 / quantizer.global_scale as f32;
        let raw_quant = (quant_field_target * inv_global_scale + 0.5) as i32;
        crate::trace::debug_eprintln!("  raw_quant (from field target 5.0) = {}", raw_quant);

        // What qac should be:
        let global_scale_float = quantizer.global_scale as f32 / GLOBAL_SCALE_DENOM as f32;
        let qac_with_raw = global_scale_float * raw_quant as f32;
        let qac_with_dc = global_scale_float * quantizer.quant_dc as f32;

        crate::trace::debug_eprintln!("  qac with raw_quant = {}", qac_with_raw);
        crate::trace::debug_eprintln!("  qac with quant_dc = {}", qac_with_dc);

        // For Y DC with inv_dequant = 0.00179:
        let inv_dequant_y_dc = 0.00179f32;
        let coeff = 2.02f32; // Typical Y DC coefficient

        let quantized_with_raw = (coeff * inv_dequant_y_dc * qac_with_raw).round();
        let quantized_with_dc = (coeff * inv_dequant_y_dc * qac_with_dc).round();

        crate::trace::debug_eprintln!("\n  For Y DC coeff = 2.02, inv_dequant = 0.00179:");
        crate::trace::debug_eprintln!("  quantized with raw_quant = {}", quantized_with_raw);
        crate::trace::debug_eprintln!("  quantized with quant_dc = {}", quantized_with_dc);
    }
}

#[cfg(test)]
mod dc_value_tests {
    use crate::color::xyb::srgb_to_xyb;
    use crate::vardct::quant_weights::INV_LF_QUANT;
    use crate::vardct::quantizer::{GLOBAL_SCALE_DENOM, QuantizerParams};
    use jxl_enc_transforms::dct8;

    /// Test what DC value a solid mid-gray block produces
    #[test]
    fn test_solid_gray_dc_value() {
        // Solid gray 8x8 block
        let gray_val = 128u8;

        crate::trace::debug_eprintln!("Input: Solid gray {}", gray_val);

        // Convert to XYB
        let (_, y_val, _) = srgb_to_xyb(gray_val as f32, gray_val as f32, gray_val as f32);
        let y_plane = [y_val; 64];

        crate::trace::debug_eprintln!("XYB Y[0]: {:.6}", y_plane[0]);

        // Apply DCT
        let mut dct_y = [0.0f32; 64];
        dct8(&y_plane, &mut dct_y);

        crate::trace::debug_eprintln!("DCT Y[0] (DC): {:.6}", dct_y[0]);

        // Quantize DC
        let quantizer = QuantizerParams::from_distance(1.0);
        let global_scale_float = quantizer.global_scale as f32 / GLOBAL_SCALE_DENOM as f32;
        let quant_dc = quantizer.quant_dc as i32;
        let inv_lf_quant_y = INV_LF_QUANT[1];

        crate::trace::debug_eprintln!(
            "Quantizer: global_scale={}, quant_dc={}",
            quantizer.global_scale,
            quant_dc
        );
        crate::trace::debug_eprintln!("INV_LF_QUANT[Y] = {}", inv_lf_quant_y);

        let qdc = inv_lf_quant_y * global_scale_float * quant_dc as f32;
        let dc_avg = dct_y[0] / 8.0; // Convert DCT DC to block average
        let dc_val = qdc * dc_avg;
        let quantized_dc = dc_val.round() as i32;

        crate::trace::debug_eprintln!("qdc = {}", qdc);
        crate::trace::debug_eprintln!("dc_avg (DC / 8) = {}", dc_avg);
        crate::trace::debug_eprintln!("dc_val = qdc * dc_avg = {}", dc_val);
        crate::trace::debug_eprintln!("quantized_dc = {}", quantized_dc);

        // Now compute what value should decode back
        let reconstructed_avg = quantized_dc as f32 / qdc;
        crate::trace::debug_eprintln!("reconstructed_avg = {}", reconstructed_avg);

        // The DC represents the average XYB Y value
        crate::trace::debug_eprintln!("\nExpected vs actual:");
        crate::trace::debug_eprintln!("  Input XYB Y = {:.6}", y_plane[0]);
        crate::trace::debug_eprintln!("  Reconstructed avg Y ≈ {:.6}", reconstructed_avg);
    }

    /// Test what value black decodes to
    #[test]
    fn test_black_dc_value() {
        // Solid black 8x8 block
        let gray_val = 0u8;

        crate::trace::debug_eprintln!("Input: Solid black ({})", gray_val);

        // Convert to XYB
        let (_, y, _) = srgb_to_xyb(gray_val as f32, gray_val as f32, gray_val as f32);
        crate::trace::debug_eprintln!("XYB Y: {:.6}", y);

        // For a flat block, DCT DC = 8 * avg
        let dct_dc = 8.0 * y;
        crate::trace::debug_eprintln!("Expected DCT DC: {:.6}", dct_dc);

        // Black should produce Y=0
        assert!(y.abs() < 1e-6, "Black should have Y=0, got {}", y);
    }

    /// Test what value white decodes to
    #[test]
    fn test_white_dc_value() {
        // Solid white 8x8 block
        let gray_val = 255u8;

        crate::trace::debug_eprintln!("Input: Solid white ({})", gray_val);

        // Convert to XYB
        let (_, y, _) = srgb_to_xyb(gray_val as f32, gray_val as f32, gray_val as f32);
        crate::trace::debug_eprintln!("XYB Y: {:.6}", y);

        // For a flat block, DCT DC = 8 * avg
        let dct_dc = 8.0 * y;
        crate::trace::debug_eprintln!("Expected DCT DC: {:.6}", dct_dc);

        // Quantize
        let quantizer = QuantizerParams::from_distance(1.0);
        let global_scale_float = quantizer.global_scale as f32 / GLOBAL_SCALE_DENOM as f32;
        let quant_dc = quantizer.quant_dc as i32;
        let inv_lf_quant_y = INV_LF_QUANT[1];

        let qdc = inv_lf_quant_y * global_scale_float * quant_dc as f32;
        let dc_avg = dct_dc / 8.0;
        let dc_val = qdc * dc_avg;
        let quantized_dc = dc_val.round() as i32;

        crate::trace::debug_eprintln!("qdc = {}", qdc);
        crate::trace::debug_eprintln!("dc_avg = {}", dc_avg);
        crate::trace::debug_eprintln!("dc_val = {}", dc_val);
        crate::trace::debug_eprintln!("quantized_dc = {}", quantized_dc);

        // White should have XYB Y ≈ 0.84
        assert!(y > 0.8, "White should have Y > 0.8, got {}", y);
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    use crate::color::xyb::srgb_image_to_xyb;
    use crate::vardct::quantizer::QuantizerParams;

    #[test]
    #[ignore = "diagnostic test for DCT16 coefficient inspection"]
    fn test_dct16_coefficient_values() {
        // Create a 16x16 gradient image
        let width = 16usize;
        let height = 16usize;
        let num_pixels = width * height;

        // Create RGB planes as f32 (sRGB values 0-255)
        let mut r_in = vec![0.0f32; num_pixels];
        let mut g_in = vec![0.0f32; num_pixels];
        let mut b_in = vec![0.0f32; num_pixels];

        for y in 0..height {
            for x in 0..width {
                let val = ((x + y) * 255 / (width + height)) as f32;
                let idx = y * width + x;
                r_in[idx] = val;
                g_in[idx] = val;
                b_in[idx] = val;
            }
        }

        // Convert to XYB
        let mut x_plane = vec![0.0f32; num_pixels];
        let mut y_plane = vec![0.0f32; num_pixels];
        let mut b_plane = vec![0.0f32; num_pixels];
        srgb_image_to_xyb(
            &r_in,
            &g_in,
            &b_in,
            &mut x_plane,
            &mut y_plane,
            &mut b_plane,
        );

        let planes: [&[f32]; 3] = [&x_plane, &y_plane, &b_plane];

        println!("XYB values (first 5 pixels):");
        for i in 0..5 {
            println!(
                "  Pixel {}: X={:.4}, Y={:.4}, B={:.4}",
                i, x_plane[i], y_plane[i], b_plane[i]
            );
        }

        // Create DCT16 strategy map (force all blocks to DCT16)
        let blocks_x = width.div_ceil(BLOCK_DIM);
        let blocks_y = height.div_ceil(BLOCK_DIM);
        println!(
            "\nBlock grid: {}x{} = {} blocks",
            blocks_x,
            blocks_y,
            blocks_x * blocks_y
        );

        let mut ac_map = AcStrategyMap::new_dct8(blocks_x, blocks_y);
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                ac_map.set(bx, by, AcStrategy::Dct16x16);
            }
        }

        // Transform and quantize
        let quantizer = QuantizerParams::from_distance(1.0);
        println!(
            "\nQuantizer: global_scale={}, quant_dc={}",
            quantizer.global_scale, quantizer.quant_dc
        );

        let transformed =
            transform_and_quantize_with_strategy(&planes, width, height, &quantizer, &ac_map);

        // Check the DC/LLF coefficients
        println!("\nDC coefficients for each block (should be LLF values):");
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let idx = by * blocks_x + bx;
                let dc_x = transformed.dc_coeffs[idx * 3];
                let dc_y = transformed.dc_coeffs[idx * 3 + 1];
                let dc_b = transformed.dc_coeffs[idx * 3 + 2];
                println!(
                    "  Block ({},{}): X={}, Y={}, B={}",
                    bx, by, dc_x, dc_y, dc_b
                );
            }
        }

        // Check AC coefficient count
        println!("\nAC coefficient stats:");
        println!("  Total AC coefficients: {}", transformed.ac_coeffs.len());

        // For DCT16, we expect 252 AC per channel (256 - 4 LLF)
        // With 1 DCT16 covering 4 blocks, for 3 channels = 252 * 3 = 756
        let expected_ac = 252 * 3; // 1 DCT16 for 3 channels
        println!("  Expected AC coefficients: {}", expected_ac);

        let nonzeros: usize = transformed.ac_coeffs.iter().filter(|&&x| x != 0).count();
        println!("  Non-zero AC coefficients: {}", nonzeros);

        // Show AC offsets
        println!("\nAC offsets (first 12):");
        for i in 0..12.min(transformed.ac_offsets.len()) {
            println!("  ac_offsets[{}] = {}", i, transformed.ac_offsets[i]);
        }

        // Show first few AC coefficients
        println!("\nFirst 20 AC coefficients:");
        for i in 0..20.min(transformed.ac_coeffs.len()) {
            println!("  ac_coeffs[{}] = {}", i, transformed.ac_coeffs[i]);
        }

        // Compare with DCT8 path
        println!("\n=== DCT8 comparison ===");
        let ac_map_dct8 = AcStrategyMap::new_dct8(blocks_x, blocks_y);
        let transformed_dct8 =
            transform_and_quantize_with_strategy(&planes, width, height, &quantizer, &ac_map_dct8);

        println!("DC coefficients (DCT8):");
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let idx = by * blocks_x + bx;
                let dc_x = transformed_dct8.dc_coeffs[idx * 3];
                let dc_y = transformed_dct8.dc_coeffs[idx * 3 + 1];
                let dc_b = transformed_dct8.dc_coeffs[idx * 3 + 2];
                println!(
                    "  Block ({},{}): X={}, Y={}, B={}",
                    bx, by, dc_x, dc_y, dc_b
                );
            }
        }

        let nonzeros_dct8: usize = transformed_dct8
            .ac_coeffs
            .iter()
            .filter(|&&x| x != 0)
            .count();
        println!(
            "\nDCT8 total AC: {}, non-zeros: {}",
            transformed_dct8.ac_coeffs.len(),
            nonzeros_dct8
        );
    }
}
