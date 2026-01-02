//! DCT transform pipeline for VarDCT encoding.
//!
//! Transforms XYB image data into quantized DCT coefficients.
//! Supports DCT8, DCT16, and DCT32 transforms based on AC strategy.

use crate::BLOCK_DIM;
use crate::heuristics::AcStrategyMap;
use crate::vardct::AcStrategy;
use jxl_enc_transforms::{dct8, dct16, dct32};

use super::enc_coeff::{quantize_block_8x8, quantize_block_16x16, quantize_block_32x32};
use super::quant_weights::get_dct8_inv_dequant_per_channel;
use super::quantizer::QuantizerParams;

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
    // Note: quantize_block_8x8 expects global_scale_float (= global_scale / GLOBAL_SCALE_DENOM)
    // which gives: qac = global_scale_float * quant = (global_scale / 65536) * quant
    // Higher distance → lower global_scale → smaller qac → more quantization (more zeros)
    let global_scale_float = quantizer.global_scale as f32 / 65536.0;
    let quant_dc = quantizer.quant_dc as i32;

    // Get DCT8 inverse dequant matrices for each channel (X, Y, B)
    // These provide perceptual weighting - higher weights = more quantization (less precision)
    // Y channel (luma) gets finest precision, X and B (chroma) are quantized more coarsely
    let inv_dequant_per_channel = get_dct8_inv_dequant_per_channel();

    // Process each block
    for by in 0..num_blocks_y {
        for bx in 0..num_blocks_x {
            let block_idx = by * num_blocks_x + bx;

            // Process each channel (X, Y, B)
            for c in 0..3 {
                // Extract 8x8 block from planar data
                let mut block_in = [0.0f32; 64];
                extract_block(xyb_data[c], width, height, bx, by, &mut block_in);

                // Apply DCT
                let mut dct_out = [0.0f32; 64];
                dct8(&block_in, &mut dct_out);

                // Quantize with per-channel perceptual weights
                let mut quant_out = [0i32; 64];
                quantize_block_8x8(
                    &dct_out,
                    quant_dc,
                    global_scale_float,
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

    let global_scale_float = quantizer.global_scale as f32 / 65536.0;
    let quant_dc = quantizer.quant_dc as i32;
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
    global_scale_float: f32,
    inv_dequant_per_channel: &[[f32; 64]; 3],
    dc_coeffs: &mut [i32],
    ac_coeffs: &mut Vec<i32>,
    ac_offsets: &mut [usize],
    current_offset: &mut usize,
) {
    let block_idx = by * num_blocks_x + bx;

    for c in 0..3 {
        let mut block_in = [0.0f32; 64];
        extract_block(xyb_data[c], width, height, bx, by, &mut block_in);

        let mut dct_out = [0.0f32; 64];
        dct8(&block_in, &mut dct_out);

        let mut quant_out = [0i32; 64];
        quantize_block_8x8(
            &dct_out,
            quant_dc,
            global_scale_float,
            &inv_dequant_per_channel[c],
            &mut quant_out,
        );

        dc_coeffs[block_idx * 3 + c] = quant_out[0];
        ac_offsets[block_idx * 3 + c] = *current_offset;
        ac_coeffs.extend_from_slice(&quant_out[1..64]);
        *current_offset += 63;
    }
}

/// Process a 2x2 region with DCT16.
#[allow(clippy::too_many_arguments)]
fn process_dct16(
    xyb_data: &[&[f32]; 3],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    num_blocks_x: usize,
    quant_dc: i32,
    global_scale_float: f32,
    inv_dequant_per_channel: &[[f32; 64]; 3],
    dc_coeffs: &mut [i32],
    ac_coeffs: &mut Vec<i32>,
    ac_offsets: &mut [usize],
    current_offset: &mut usize,
) {
    let block_idx = by * num_blocks_x + bx;

    for c in 0..3 {
        let mut block_in = [0.0f32; 256];
        extract_block_16x16(xyb_data[c], width, height, bx, by, &mut block_in);

        let mut dct_out = [0.0f32; 256];
        dct16(&block_in, &mut dct_out);

        let mut quant_out = [0i32; 256];
        quantize_block_16x16(
            &dct_out,
            quant_dc,
            global_scale_float,
            &inv_dequant_per_channel[c],
            &mut quant_out,
        );

        // DC goes at top-left block position only
        dc_coeffs[block_idx * 3 + c] = quant_out[0];
        // Zero DC for covered blocks (they get the shared DC from top-left)
        for dy in 0..2 {
            for dx in 0..2 {
                if dx != 0 || dy != 0 {
                    let covered_idx = (by + dy) * num_blocks_x + (bx + dx);
                    dc_coeffs[covered_idx * 3 + c] = 0;
                }
            }
        }

        // AC coefficients (255 of them for DCT16)
        ac_offsets[block_idx * 3 + c] = *current_offset;
        ac_coeffs.extend_from_slice(&quant_out[1..256]);
        *current_offset += 255;

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
fn process_dct32(
    xyb_data: &[&[f32]; 3],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    num_blocks_x: usize,
    quant_dc: i32,
    global_scale_float: f32,
    inv_dequant_per_channel: &[[f32; 64]; 3],
    dc_coeffs: &mut [i32],
    ac_coeffs: &mut Vec<i32>,
    ac_offsets: &mut [usize],
    current_offset: &mut usize,
) {
    let block_idx = by * num_blocks_x + bx;

    for c in 0..3 {
        let mut block_in = [0.0f32; 1024];
        extract_block_32x32(xyb_data[c], width, height, bx, by, &mut block_in);

        let mut dct_out = [0.0f32; 1024];
        dct32(&block_in, &mut dct_out);

        let mut quant_out = [0i32; 1024];
        quantize_block_32x32(
            &dct_out,
            quant_dc,
            global_scale_float,
            &inv_dequant_per_channel[c],
            &mut quant_out,
        );

        // DC goes at top-left block position only
        dc_coeffs[block_idx * 3 + c] = quant_out[0];
        // Zero DC for covered blocks
        for dy in 0..4 {
            for dx in 0..4 {
                if dx != 0 || dy != 0 {
                    let covered_idx = (by + dy) * num_blocks_x + (bx + dx);
                    dc_coeffs[covered_idx * 3 + c] = 0;
                }
            }
        }

        // AC coefficients (1023 of them for DCT32)
        ac_offsets[block_idx * 3 + c] = *current_offset;
        ac_coeffs.extend_from_slice(&quant_out[1..1024]);
        *current_offset += 1023;

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
        // DCT16 produces 255 AC per channel for the block
        assert_eq!(result.ac_coeffs.len(), 3 * 255);
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
        // DCT32 produces 1023 AC per channel for the block
        assert_eq!(result.ac_coeffs.len(), 3 * 1023);
    }
}
