//! Coefficient encoding for VarDCT.
//!
//! This module handles quantization and encoding of DCT coefficients.

use crate::BLOCK_SIZE;

/// Threshold for zeroing small coefficients.
/// Coefficients with |quantized_value| < threshold are set to zero.
const DEFAULT_THRESHOLD: f32 = 0.5;

/// Quantization thresholds per quadrant (top-left, top-right, bottom-left, bottom-right).
#[derive(Clone, Copy, Debug)]
pub struct QuantThresholds {
    pub thresholds: [f32; 4],
}

impl Default for QuantThresholds {
    fn default() -> Self {
        Self {
            thresholds: [0.58, 0.62, 0.62, 0.62],
        }
    }
}

impl QuantThresholds {
    /// Create thresholds for Y channel.
    pub fn for_y() -> Self {
        Self {
            thresholds: [0.56, 0.62, 0.62, 0.62],
        }
    }

    /// Create thresholds for X/B channels.
    pub fn for_xb() -> Self {
        Self {
            thresholds: [0.58, 0.62, 0.62, 0.62],
        }
    }

    /// Adjust thresholds based on block size (larger blocks need more zeroing).
    pub fn adjust_for_block_size(&mut self, xblocks: usize, yblocks: usize) {
        let size = xblocks * yblocks;
        if size >= 4 {
            for t in &mut self.thresholds {
                *t -= 0.00744 * size as f32;
                if *t < 0.5 {
                    *t = 0.5;
                }
            }
        }
    }
}

/// Quantize a block of AC coefficients.
///
/// # Arguments
/// * `block_in` - Input DCT coefficients (float)
/// * `quant` - Per-block quantization value
/// * `global_scale_float` - Global scale (global_scale / GLOBAL_SCALE_DENOM)
/// * `inv_dequant_matrix` - Inverse dequantization matrix for this transform type
/// * `qm_multiplier` - Quantization matrix multiplier (usually 1.0)
/// * `thresholds` - Thresholds per quadrant for zeroing
/// * `xblocks` - Number of 8x8 blocks in X direction
/// * `yblocks` - Number of 8x8 blocks in Y direction
/// * `block_out` - Output quantized coefficients (i32)
#[allow(clippy::too_many_arguments)]
pub fn quantize_block_ac(
    block_in: &[f32],
    quant: i32,
    global_scale_float: f32,
    inv_dequant_matrix: &[f32],
    qm_multiplier: f32,
    thresholds: &QuantThresholds,
    xblocks: usize,
    yblocks: usize,
    block_out: &mut [i32],
) {
    let total_size = xblocks * yblocks * BLOCK_SIZE;
    assert!(block_in.len() >= total_size);
    assert!(inv_dequant_matrix.len() >= total_size);
    assert!(block_out.len() >= total_size);

    // qac = global_scale_float * quant
    let qac = global_scale_float * quant as f32 * qm_multiplier;

    let half_x = xblocks * 4; // Half point in X
    let half_y = yblocks * 4; // Half point in Y
    let stride = xblocks * 8;

    for y in 0..yblocks * 8 {
        let yfix = if y >= half_y { 2 } else { 0 };
        let off = y * stride;

        for x in 0..xblocks * 8 {
            let xfix = if x >= half_x { 1 } else { 0 };
            let threshold = thresholds.thresholds[yfix + xfix];

            let pos = off + x;
            let q = inv_dequant_matrix[pos] * qac;
            let val = q * block_in[pos];

            // Zero small coefficients
            block_out[pos] = if val.abs() >= threshold {
                val.round() as i32
            } else {
                0
            };
        }
    }
}

/// Quantize a single 8x8 block (DCT8x8).
///
/// # Arguments
/// * `block_in` - Input DCT coefficients (float)
/// * `quant_dc` - DC quantization value (from QuantizerParams.quant_dc)
/// * `raw_quant` - Per-block raw quant field value for AC coefficients
/// * `global_scale_float` - Global scale (global_scale / GLOBAL_SCALE_DENOM)
/// * `inv_lf_quant` - Inverse LF quant factor for DC (INV_LF_QUANT[channel])
/// * `inv_dequant_matrix` - Inverse dequantization matrix for AC coefficients
/// * `block_out` - Output quantized coefficients (i32)
pub fn quantize_block_8x8(
    block_in: &[f32; BLOCK_SIZE],
    quant_dc: i32,
    raw_quant: i32,
    global_scale_float: f32,
    inv_lf_quant: f32,
    inv_dequant_matrix: &[f32; BLOCK_SIZE],
    block_out: &mut [i32; BLOCK_SIZE],
) {
    let threshold = DEFAULT_THRESHOLD;

    // DC coefficient uses different quantization: inv_lf_quant * global_scale_float * quant_dc
    let qdc = inv_lf_quant * global_scale_float * quant_dc as f32;
    let dc_val = qdc * block_in[0];
    block_out[0] = if dc_val.abs() >= threshold {
        dc_val.round() as i32
    } else {
        0
    };

    // AC coefficients use: inv_dequant_matrix[i] * global_scale_float * raw_quant
    let qac = global_scale_float * raw_quant as f32;
    for i in 1..BLOCK_SIZE {
        let q = inv_dequant_matrix[i] * qac;
        let val = q * block_in[i];
        block_out[i] = if val.abs() >= threshold {
            val.round() as i32
        } else {
            0
        };
    }
}

/// Quantize a 16x16 block (DCT16x16).
///
/// Uses scaled DCT8 weights for the larger block.
pub fn quantize_block_16x16(
    block_in: &[f32; 256],
    quant_dc: i32,
    raw_quant: i32,
    global_scale_float: f32,
    inv_lf_quant: f32,
    inv_dequant_8x8: &[f32; 64],
    block_out: &mut [i32; 256],
) {
    let threshold = DEFAULT_THRESHOLD;

    // DC coefficient (position 0) uses separate quantization
    let qdc = inv_lf_quant * global_scale_float * quant_dc as f32;
    let dc_val = qdc * block_in[0];
    block_out[0] = if dc_val.abs() >= threshold {
        dc_val.round() as i32
    } else {
        0
    };

    // AC coefficients use: inv_dequant * global_scale_float * raw_quant
    let qac = global_scale_float * raw_quant as f32;

    // Scale DCT8 weights to DCT16 by interpolation
    // For position (x, y) in 16x16, map to (x/2, y/2) in 8x8 weights
    for y in 0..16 {
        for x in 0..16 {
            let pos = y * 16 + x;
            if pos == 0 {
                continue; // DC already handled
            }
            // Map to 8x8 position with interpolation
            let x8 = x / 2;
            let y8 = y / 2;
            let weight_pos = y8 * 8 + x8;

            let q = inv_dequant_8x8[weight_pos] * qac;
            let val = q * block_in[pos];
            block_out[pos] = if val.abs() >= threshold {
                val.round() as i32
            } else {
                0
            };
        }
    }
}

/// Quantize a 32x32 block (DCT32x32).
///
/// Uses scaled DCT8 weights for the larger block.
pub fn quantize_block_32x32(
    block_in: &[f32; 1024],
    quant_dc: i32,
    raw_quant: i32,
    global_scale_float: f32,
    inv_lf_quant: f32,
    inv_dequant_8x8: &[f32; 64],
    block_out: &mut [i32; 1024],
) {
    let threshold = DEFAULT_THRESHOLD;

    // DC coefficient (position 0) uses separate quantization
    let qdc = inv_lf_quant * global_scale_float * quant_dc as f32;
    let dc_val = qdc * block_in[0];
    block_out[0] = if dc_val.abs() >= threshold {
        dc_val.round() as i32
    } else {
        0
    };

    // AC coefficients use: inv_dequant * global_scale_float * raw_quant
    let qac = global_scale_float * raw_quant as f32;

    // Scale DCT8 weights to DCT32 by interpolation
    // For position (x, y) in 32x32, map to (x/4, y/4) in 8x8 weights
    for y in 0..32 {
        for x in 0..32 {
            let pos = y * 32 + x;
            if pos == 0 {
                continue; // DC already handled
            }
            // Map to 8x8 position with interpolation
            let x8 = x / 4;
            let y8 = y / 4;
            let weight_pos = y8 * 8 + x8;

            let q = inv_dequant_8x8[weight_pos] * qac;
            let val = q * block_in[pos];
            block_out[pos] = if val.abs() >= threshold {
                val.round() as i32
            } else {
                0
            };
        }
    }
}

/// Simple quantization without dequant matrix (for testing).
pub fn quantize_simple(block_in: &[f32], quant: f32, block_out: &mut [i32]) {
    assert_eq!(block_in.len(), block_out.len());
    for (i, &coeff) in block_in.iter().enumerate() {
        let val = coeff * quant;
        block_out[i] = if val.abs() >= DEFAULT_THRESHOLD {
            val.round() as i32
        } else {
            0
        };
    }
}

/// Count non-zero coefficients in a block.
pub fn count_nonzeros(block: &[i32]) -> usize {
    block.iter().filter(|&&x| x != 0).count()
}

/// Pack a signed coefficient value using zigzag encoding.
/// This maps signed integers to unsigned integers:
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, etc.
#[inline]
pub fn pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) * 2
    } else {
        ((-value) as u32) * 2 - 1
    }
}

/// Unpack a zigzag-encoded value back to signed.
#[inline]
pub fn unpack_signed(value: u32) -> i32 {
    if value & 1 == 0 {
        (value / 2) as i32
    } else {
        -((value / 2 + 1) as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_signed() {
        assert_eq!(pack_signed(0), 0);
        assert_eq!(pack_signed(1), 2);
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(2), 4);
        assert_eq!(pack_signed(-2), 3);
        assert_eq!(pack_signed(100), 200);
        assert_eq!(pack_signed(-100), 199);
    }

    #[test]
    fn test_unpack_signed() {
        for i in -1000..1000 {
            assert_eq!(unpack_signed(pack_signed(i)), i);
        }
    }

    #[test]
    fn test_quantize_simple() {
        let input = [1.0, 2.0, 0.1, -0.1, -1.0, -2.0, 0.6, -0.6];
        let mut output = [0i32; 8];

        quantize_simple(&input, 1.0, &mut output);

        assert_eq!(output[0], 1); // 1.0 -> 1
        assert_eq!(output[1], 2); // 2.0 -> 2
        assert_eq!(output[2], 0); // 0.1 < 0.5 -> 0
        assert_eq!(output[3], 0); // -0.1 -> 0
        assert_eq!(output[4], -1); // -1.0 -> -1
        assert_eq!(output[5], -2); // -2.0 -> -2
        assert_eq!(output[6], 1); // 0.6 -> 1
        assert_eq!(output[7], -1); // -0.6 -> -1
    }

    #[test]
    fn test_count_nonzeros() {
        let block = [0, 1, 0, -1, 0, 0, 2, 0];
        assert_eq!(count_nonzeros(&block), 3);

        let all_zeros = [0i32; 64];
        assert_eq!(count_nonzeros(&all_zeros), 0);
    }

    #[test]
    fn test_quantize_block_8x8() {
        let mut block_in = [0.0f32; 64];
        block_in[0] = 100.0; // DC coefficient
        block_in[1] = 10.0; // AC coefficient
        block_in[63] = 1.0; // High frequency

        let inv_dequant = [1.0f32; 64]; // Identity matrix for simplicity
        let mut block_out = [0i32; 64];

        // quant_dc=1, raw_quant=1, global_scale_float=1.0, inv_lf_quant=1.0 means no scaling
        quantize_block_8x8(&block_in, 1, 1, 1.0, 1.0, &inv_dequant, &mut block_out);

        assert_eq!(block_out[0], 100);
        assert_eq!(block_out[1], 10);
        assert_eq!(block_out[63], 1);
    }

    #[test]
    fn test_thresholds_default() {
        let t = QuantThresholds::default();
        assert_eq!(t.thresholds[0], 0.58);
    }

    #[test]
    fn test_thresholds_adjust() {
        let mut t = QuantThresholds::default();
        t.adjust_for_block_size(2, 2); // 4 blocks

        // Each threshold reduced by 0.00744 * 4 = 0.02976
        assert!(t.thresholds[0] < 0.58);
        assert!(t.thresholds[0] >= 0.5); // But not below 0.5
    }
}
