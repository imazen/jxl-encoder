// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Forward DCT transforms for sizes up to 16x16.

use super::constants::*;

/// In-place 1D DCT for N=2
#[inline]
pub fn dct1d_2(mem: &mut [f32]) {
    let in1 = mem[0];
    let in2 = mem[1];
    mem[0] = in1 + in2;
    mem[1] = in1 - in2;
}

/// In-place 1D DCT for N=4
#[inline]
pub fn dct1d_4(mem: &mut [f32]) {
    // AddReverse: tmp[i] = mem[i] + mem[N-1-i] for first half
    // SubReverse: tmp[N/2+i] = mem[i] - mem[N-1-i] for second half
    let mut tmp = [0.0f32; 4];
    tmp[0] = mem[0] + mem[3];
    tmp[1] = mem[1] + mem[2];
    tmp[2] = mem[0] - mem[3];
    tmp[3] = mem[1] - mem[2];

    // DCT on first half
    dct1d_2(&mut tmp[0..2]);

    // Multiply second half by WcMultipliers
    tmp[2] *= WC_MULTIPLIERS_4[0];
    tmp[3] *= WC_MULTIPLIERS_4[1];

    // DCT on second half
    dct1d_2(&mut tmp[2..4]);

    // B transform on second half
    // B: tmp[0] = sqrt2 * tmp[0] + tmp[1], then tmp[i] = tmp[i] + tmp[i+1] for rest
    tmp[2] = SQRT2.mul_add(tmp[2], tmp[3]);
    // (no more elements for N/2=2)

    // InverseEvenOdd: interleave even and odd
    mem[0] = tmp[0];
    mem[2] = tmp[1];
    mem[1] = tmp[2];
    mem[3] = tmp[3];
}

/// In-place 1D DCT for N=8
pub fn dct1d_8(mem: &mut [f32]) {
    let mut tmp = [0.0f32; 8];

    // AddReverse for first half
    for i in 0..4 {
        tmp[i] = mem[i] + mem[7 - i];
    }
    // SubReverse for second half
    for i in 0..4 {
        tmp[4 + i] = mem[i] - mem[7 - i];
    }

    // DCT on first half
    dct1d_4(&mut tmp[0..4]);

    // Multiply second half by WcMultipliers
    for i in 0..4 {
        tmp[4 + i] *= WC_MULTIPLIERS_8[i];
    }

    // DCT on second half
    dct1d_4(&mut tmp[4..8]);

    // B transform on second half
    tmp[4] = SQRT2.mul_add(tmp[4], tmp[5]);
    tmp[5] += tmp[6];
    tmp[6] += tmp[7];

    // InverseEvenOdd: interleave
    for i in 0..4 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[4 + i];
    }
}

/// In-place 1D DCT for N=16
pub fn dct1d_16(mem: &mut [f32]) {
    let mut tmp = [0.0f32; 16];

    // AddReverse for first half
    for i in 0..8 {
        tmp[i] = mem[i] + mem[15 - i];
    }
    // SubReverse for second half
    for i in 0..8 {
        tmp[8 + i] = mem[i] - mem[15 - i];
    }

    // DCT on first half
    dct1d_8(&mut tmp[0..8]);

    // Multiply second half by WcMultipliers
    for i in 0..8 {
        tmp[8 + i] *= WC_MULTIPLIERS_16[i];
    }

    // DCT on second half
    dct1d_8(&mut tmp[8..16]);

    // B transform on second half
    tmp[8] = SQRT2.mul_add(tmp[8], tmp[9]);
    for i in 1..7 {
        tmp[8 + i] += tmp[8 + i + 1];
    }

    // InverseEvenOdd: interleave
    for i in 0..8 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[8 + i];
    }
}

/// Compute scaled 8x8 DCT.
///
/// Input: 8x8 block in row-major order
/// Output: 8x8 DCT coefficients in **transposed** layout
///
/// IMPORTANT: libjxl-tiny's ComputeScaledDCT does NOT transpose back for square blocks.
/// The decoder expects coefficients in this transposed layout. For 8x8 blocks,
/// output[cx * 8 + cy] contains the coefficient for frequency (cy, cx) where
/// cy is the vertical frequency and cx is the horizontal frequency.
#[inline]
pub fn dct_8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::dct_8x8(input, output);
}

/// Compute base 4x8 DCT (4 rows, 8 columns).
///
/// This is the primitive transform for a single 4x8 sub-block.
/// Input: 4x8 = 32 floats in row-major order
/// Output: 32 DCT coefficients
///
/// Based on libjxl's ComputeScaledDCT<4, 8>. Since ROWS < COLS,
/// the transform includes a final transpose.
#[inline]
pub fn dct_4x8(input: &[f32; 32], output: &mut [f32; 32]) {
    let mut tmp = [0.0f32; 32];

    // Transform rows (8 columns each) with 8-point DCT
    for row in 0..4 {
        let row_start = row * 8;
        tmp[row_start..row_start + 8].copy_from_slice(&input[row_start..row_start + 8]);
        dct1d_8(&mut tmp[row_start..row_start + 8]);
        for i in 0..8 {
            tmp[row_start + i] *= 1.0 / 8.0;
        }
    }

    // Transpose 4x8 -> 8x4
    let mut transposed = [0.0f32; 32];
    for row in 0..4 {
        for col in 0..8 {
            transposed[col * 4 + row] = tmp[row * 8 + col];
        }
    }

    // Transform columns (now 4 elements each) with 4-point DCT
    for row in 0..8 {
        let row_start = row * 4;
        dct1d_4(&mut transposed[row_start..row_start + 4]);
        for i in 0..4 {
            transposed[row_start + i] *= 1.0 / 4.0;
        }
    }

    // Final transpose 8x4 -> 4x8 (ROWS < COLS branch in libjxl)
    for row in 0..8 {
        for col in 0..4 {
            output[col * 8 + row] = transposed[row * 4 + col];
        }
    }
}

/// Compute base 8x4 DCT (8 rows, 4 columns).
///
/// This is the primitive transform for a single 8x4 sub-block.
/// Input: 8x4 = 32 floats in row-major order
/// Output: 32 DCT coefficients
///
/// Based on libjxl's ComputeScaledDCT<8, 4>. Since ROWS >= COLS,
/// there is NO final transpose.
#[inline]
pub fn dct_8x4(input: &[f32; 32], output: &mut [f32; 32]) {
    let mut tmp = [0.0f32; 32];

    // Transform rows (4 columns each) with 4-point DCT
    for row in 0..8 {
        let row_start = row * 4;
        tmp[row_start..row_start + 4].copy_from_slice(&input[row_start..row_start + 4]);
        dct1d_4(&mut tmp[row_start..row_start + 4]);
        for i in 0..4 {
            tmp[row_start + i] *= 1.0 / 4.0;
        }
    }

    // Transpose 8x4 -> 4x8
    let mut transposed = [0.0f32; 32];
    for row in 0..8 {
        for col in 0..4 {
            transposed[col * 8 + row] = tmp[row * 4 + col];
        }
    }

    // Transform columns (now 8 elements each) with 8-point DCT
    for row in 0..4 {
        let row_start = row * 8;
        dct1d_8(&mut transposed[row_start..row_start + 8]);
        for i in 0..8 {
            transposed[row_start + i] *= 1.0 / 8.0;
        }
    }

    // NO final transpose for ROWS >= COLS (matches dct_8x8 behavior)
    output.copy_from_slice(&transposed);
}

/// Compute full DCT4X8 transform for 8x8 pixel block.
///
/// This covers an 8x8 pixel region using TWO vertically-stacked 4x8 sub-blocks.
/// The DC values of the two sub-blocks are combined with a 2-point transform.
///
/// Input: 8x8 = 64 floats in row-major order (stride 8)
/// Output: 64 DCT coefficients in interleaved layout
///
/// Matches libjxl's Type::DCT4X8 case in enc_transforms-inl.h
#[inline]
pub fn dct_4x8_full(input: &[f32; 64], output: &mut [f32; 64]) {
    // Process two 4x8 sub-blocks (top and bottom halves)
    for y in 0..2 {
        // Extract 4x8 sub-block
        let mut block = [0.0f32; 32];
        for iy in 0..4 {
            for ix in 0..8 {
                block[iy * 8 + ix] = input[(y * 4 + iy) * 8 + ix];
            }
        }

        // Apply base 4x8 DCT
        let mut coeffs = [0.0f32; 32];
        dct_4x8(&block, &mut coeffs);

        // Interleave into output: coefficients[(y + iy * 2) * 8 + ix]
        for iy in 0..4 {
            for ix in 0..8 {
                output[(y + iy * 2) * 8 + ix] = coeffs[iy * 8 + ix];
            }
        }
    }

    // Combine DC values of the two sub-blocks with 2-point transform
    let block0_dc = output[0];
    let block1_dc = output[8];
    output[0] = (block0_dc + block1_dc) * 0.5;
    output[8] = (block0_dc - block1_dc) * 0.5;
}

/// Compute full DCT8X4 transform for 8x8 pixel block.
///
/// This covers an 8x8 pixel region using TWO horizontally-adjacent 8x4 sub-blocks.
/// The DC values of the two sub-blocks are combined with a 2-point transform.
///
/// Input: 8x8 = 64 floats in row-major order (stride 8)
/// Output: 64 DCT coefficients in interleaved layout
///
/// Matches libjxl's Type::DCT8X4 case in enc_transforms-inl.h
#[inline]
pub fn dct_8x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    // Process two 8x4 sub-blocks (left and right halves)
    for x in 0..2 {
        // Extract 8x4 sub-block
        let mut block = [0.0f32; 32];
        for iy in 0..8 {
            for ix in 0..4 {
                block[iy * 4 + ix] = input[iy * 8 + (x * 4 + ix)];
            }
        }

        // Apply base 8x4 DCT
        let mut coeffs = [0.0f32; 32];
        dct_8x4(&block, &mut coeffs);

        // Interleave into output: coefficients[(x + iy * 2) * 8 + ix]
        // Note: the 8x4 output is in 4x8 layout (stride 8) after the transform
        for iy in 0..4 {
            for ix in 0..8 {
                output[(x + iy * 2) * 8 + ix] = coeffs[iy * 8 + ix];
            }
        }
    }

    // Combine DC values of the two sub-blocks with 2-point transform
    let block0_dc = output[0];
    let block1_dc = output[8];
    output[0] = (block0_dc + block1_dc) * 0.5;
    output[8] = (block0_dc - block1_dc) * 0.5;
}

/// Extract DC value from DCT4X8 full transform coefficients.
///
/// For DCT4X8 (and DCT8X4), the 8x8 block is covered by a single 1x1 DC region.
/// The DC combining step already produced the DC at position [0].
#[inline]
pub fn dc_from_dct_4x8_full(coeffs: &[f32; 64]) -> f32 {
    coeffs[0]
}

/// Extract DC value from DCT8X4 full transform coefficients.
///
/// Same as DCT4X8 - single DC at position [0].
#[inline]
pub fn dc_from_dct_8x4_full(coeffs: &[f32; 64]) -> f32 {
    coeffs[0]
}

/// Compute base 4x4 DCT.
///
/// Input: 4x4 = 16 floats in row-major order (stride 4)
/// Output: 16 DCT coefficients
///
/// Based on libjxl's ComputeScaledDCT<4, 4>. Since ROWS == COLS (square),
/// there is NO final transpose.
#[inline]
pub fn dct_4x4(input: &[f32; 16], output: &mut [f32; 16]) {
    let mut tmp = [0.0f32; 16];

    // Transform rows with 4-point DCT
    for row in 0..4 {
        let row_start = row * 4;
        tmp[row_start..row_start + 4].copy_from_slice(&input[row_start..row_start + 4]);
        dct1d_4(&mut tmp[row_start..row_start + 4]);
        for i in 0..4 {
            tmp[row_start + i] *= 1.0 / 4.0;
        }
    }

    // Transpose 4x4
    let mut transposed = [0.0f32; 16];
    for row in 0..4 {
        for col in 0..4 {
            transposed[col * 4 + row] = tmp[row * 4 + col];
        }
    }

    // Transform columns (now rows after transpose) with 4-point DCT
    for row in 0..4 {
        let row_start = row * 4;
        dct1d_4(&mut transposed[row_start..row_start + 4]);
        for i in 0..4 {
            transposed[row_start + i] *= 1.0 / 4.0;
        }
    }

    // No final transpose for square blocks (ROWS >= COLS in libjxl)
    // Output is in transposed layout
    output.copy_from_slice(&transposed);
}

/// Compute full DCT4X4 transform for 8x8 pixel block.
///
/// This covers an 8x8 pixel region using FOUR 4x4 sub-blocks arranged in a 2x2 grid.
/// The DC values of the four sub-blocks are combined with a 2x2 DCT.
///
/// Input: 8x8 = 64 floats in row-major order (stride 8)
/// Output: 64 DCT coefficients in interleaved layout
///
/// Matches libjxl's Type::DCT4X4 case in enc_transforms-inl.h
#[inline]
pub fn dct_4x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    // Process four 4x4 sub-blocks in 2x2 grid
    for y in 0..2 {
        for x in 0..2 {
            // Extract 4x4 sub-block
            let mut block = [0.0f32; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    block[iy * 4 + ix] = input[(y * 4 + iy) * 8 + (x * 4 + ix)];
                }
            }

            // Apply base 4x4 DCT
            let mut coeffs = [0.0f32; 16];
            dct_4x4(&block, &mut coeffs);

            // Interleave into output: coefficients[(y + iy * 2) * 8 + x + ix * 2]
            for iy in 0..4 {
                for ix in 0..4 {
                    output[(y + iy * 2) * 8 + x + ix * 2] = coeffs[iy * 4 + ix];
                }
            }
        }
    }

    // Combine DC values of the four sub-blocks with 2x2 DCT
    // Sub-block DCs are at positions: (0,0)->0, (0,1)->1, (1,0)->8, (1,1)->9
    let block00 = output[0];
    let block01 = output[1];
    let block10 = output[8];
    let block11 = output[9];

    // 2x2 DCT: same as libjxl's DC combining
    output[0] = (block00 + block01 + block10 + block11) * 0.25;
    output[1] = (block00 + block01 - block10 - block11) * 0.25;
    output[8] = (block00 - block01 + block10 - block11) * 0.25;
    output[9] = (block00 - block01 - block10 + block11) * 0.25;
}

/// Extract DC value from DCT4X4 full transform coefficients.
///
/// For DCT4X4, the 8x8 block has a 2x2 LLF region at positions [0,1,8,9].
/// The DC (average) is at position [0].
#[inline]
pub fn dc_from_dct_4x4_full(coeffs: &[f32; 64]) -> f32 {
    coeffs[0]
}

/// Compute scaled 16x8 DCT (16 rows, 8 columns).
///
/// Input: 16x8 block in row-major order (128 floats)
/// Output: 16x8 DCT coefficients
#[inline]
pub fn dct_16x8(input: &[f32; 128], output: &mut [f32; 128]) {
    jxl_simd::dct_16x8(input, output);
}

/// Compute scaled 8x16 DCT (8 rows, 16 columns).
///
/// Input: 8x16 block in row-major order (128 floats)
/// Output: 8x16 DCT coefficients
#[inline]
pub fn dct_8x16(input: &[f32; 128], output: &mut [f32; 128]) {
    jxl_simd::dct_8x16(input, output);
}

/// Compute scaled 16x16 DCT (16 rows, 16 columns).
///
/// Input: 16x16 block in row-major order (256 floats)
/// Output: 16x16 DCT coefficients
///
/// Like `dct_8x8()`, there is NO final transpose for square blocks.
/// C++ `ComputeScaledDCT<16,16>` takes the ROWS >= COLS branch (no final transpose).
#[inline]
pub fn dct_16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    jxl_simd::dct_16x16(input, output);
}

/// Extract DC values from 16x16 DCT coefficients.
/// Returns 4 DC values in spatial order: `[top-left, top-right, bottom-left, bottom-right]`.
///
/// The caller stores `dcs[iy * 2 + ix]` at position `(by + iy, bx + ix)`, so:
///   dcs[0] → (by, bx), dcs[1] → (by, bx+1), dcs[2] → (by+1, bx), dcs[3] → (by+1, bx+1).
///
/// The LLF region is 2x2 coefficients at positions [0, 1, 16, 17] in the 16x16 layout
/// (stride 16). We apply `DCTTotalResampleScale<16, 2>` to each dimension, then a
/// 2x2 IDCT to get the 4 DC values.
///
/// C++ uses `ReinterpretingIDCT<16, 16, 2, 2, 2, 2>` → `ComputeScaledIDCT<2, 2>`.
/// The IDCT steps (ROWS >= COLS branch): IDCT rows → transpose → IDCT rows.
/// The transpose between steps swaps off-diagonal elements.
pub fn dc_from_dct_16x16(coeffs: &[f32; 256]) -> [f32; 4] {
    let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0]; // 1.0
    let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1]; // 0.9018...

    // Read LLF 2x2 from positions [0, 1, 16, 17] and apply resample scales.
    // C++ ROWS >= COLS: block[y * ROWS + x] = input[y * stride + x] * scale_col[y] * scale_row[x]
    let b00 = coeffs[0] * s0 * s0;
    let b01 = coeffs[1] * s0 * s1;
    let b10 = coeffs[16] * s1 * s0;
    let b11 = coeffs[17] * s1 * s1;

    // 2x2 IDCT (ComputeScaledIDCT<2,2>, ROWS >= COLS):
    // Step 1 — IDCT rows (length 2): [a, b] → [a+b, a-b]
    //   Row 0: [b00+b01, b00-b01]
    //   Row 1: [b10+b11, b10-b11]
    // Step 2 — Transpose 2×2:
    //   [b00+b01, b10+b11]
    //   [b00-b01, b10-b11]
    // Step 3 — IDCT rows (length 2):
    //   out[0,0] = (b00+b01) + (b10+b11)
    //   out[0,1] = (b00+b01) - (b10+b11)
    //   out[1,0] = (b00-b01) + (b10-b11)
    //   out[1,1] = (b00-b01) - (b10-b11)
    let out00 = (b00 + b01) + (b10 + b11); // top-left
    let out01 = (b00 + b01) - (b10 + b11); // top-right
    let out10 = (b00 - b01) + (b10 - b11); // bottom-left
    let out11 = (b00 - b01) - (b10 - b11); // bottom-right

    [out00, out01, out10, out11]
}
