//! DCT transform pipeline for VarDCT encoding.
//!
//! Transforms XYB image data into quantized DCT coefficients.

use crate::BLOCK_DIM;
use jxl_enc_transforms::dct8;

use super::enc_coeff::quantize_block_8x8;
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
    let inv_global_scale = quantizer.inv_global_scale();
    let quant_dc = quantizer.quant_dc as i32;

    // Simple inverse dequant matrix (flat for now - real implementation uses the tables)
    let inv_dequant = [1.0f32; 64];

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

                // Quantize
                let mut quant_out = [0i32; 64];
                quantize_block_8x8(
                    &dct_out,
                    quant_dc,
                    inv_global_scale,
                    &inv_dequant,
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
}
