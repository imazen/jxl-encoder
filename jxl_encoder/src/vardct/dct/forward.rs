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

/// Value-returning 1D DCT for N=4. Returns [out0, out1, out2, out3].
#[inline(always)]
fn dct1d_4_val(a: f32, b: f32, c: f32, d: f32) -> [f32; 4] {
    // AddReverse/SubReverse
    let t0 = a + d;
    let t1 = b + c;
    let t2 = a - d;
    let t3 = b - c;
    // DCT on first half (2-point)
    let u0 = t0 + t1;
    let u1 = t0 - t1;
    // Wc multiply + DCT on second half (2-point)
    let v0 = t2 * WC_MULTIPLIERS_4[0];
    let v1 = t3 * WC_MULTIPLIERS_4[1];
    let w0 = v0 + v1;
    let w1 = v0 - v1;
    // B transform
    let b0 = SQRT2 * w0 + w1;
    // InverseEvenOdd
    [u0, b0, u1, w1]
}

/// In-place 1D DCT for N=4
#[inline(always)]
pub fn dct1d_4(mem: &mut [f32]) {
    let r = dct1d_4_val(mem[0], mem[1], mem[2], mem[3]);
    mem[0] = r[0];
    mem[1] = r[1];
    mem[2] = r[2];
    mem[3] = r[3];
}

/// Value-returning 1D DCT for N=8. Returns [out0..out7].
#[inline(always)]
fn dct1d_8_val(m: [f32; 8]) -> [f32; 8] {
    // AddReverse/SubReverse
    let t0 = m[0] + m[7];
    let t1 = m[1] + m[6];
    let t2 = m[2] + m[5];
    let t3 = m[3] + m[4];
    let t4 = m[0] - m[7];
    let t5 = m[1] - m[6];
    let t6 = m[2] - m[5];
    let t7 = m[3] - m[4];
    // DCT-4 on first half
    let r0 = dct1d_4_val(t0, t1, t2, t3);
    // Wc multiply on second half
    let w4 = t4 * WC_MULTIPLIERS_8[0];
    let w5 = t5 * WC_MULTIPLIERS_8[1];
    let w6 = t6 * WC_MULTIPLIERS_8[2];
    let w7 = t7 * WC_MULTIPLIERS_8[3];
    // DCT-4 on second half
    let r1 = dct1d_4_val(w4, w5, w6, w7);
    // B transform: sqrt2*r1[0]+r1[1], r1[1]+r1[2], r1[2]+r1[3], r1[3]
    let b0 = SQRT2 * r1[0] + r1[1];
    let b1 = r1[1] + r1[2];
    let b2 = r1[2] + r1[3];
    let b3 = r1[3];
    // InverseEvenOdd
    [r0[0], b0, r0[1], b1, r0[2], b2, r0[3], b3]
}

/// In-place 1D DCT for N=8
pub fn dct1d_8(mem: &mut [f32]) {
    let r = dct1d_8_val([
        mem[0], mem[1], mem[2], mem[3], mem[4], mem[5], mem[6], mem[7],
    ]);
    mem[..8].copy_from_slice(&r);
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
    tmp[8] = SQRT2 * tmp[8] + tmp[9];
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
#[inline(always)]
pub fn dct_4x8(input: &[f32; 32], output: &mut [f32; 32]) {
    // Pass 1: row DCTs (8-point) + transpose + scale into temp
    // Row DCTs produce 4 rows × 8 cols; we store transposed: 8 rows × 4 cols
    let mut temp = [0.0f32; 32];
    for row in 0..4 {
        let s = row * 8;
        let r = dct1d_8_val([
            input[s],
            input[s + 1],
            input[s + 2],
            input[s + 3],
            input[s + 4],
            input[s + 5],
            input[s + 6],
            input[s + 7],
        ]);
        // Store transposed (col-major) with 1/8 scale
        for col in 0..8 {
            temp[col * 4 + row] = r[col] * (1.0 / 8.0);
        }
    }

    // Pass 2: column DCTs (4-point) + final transpose + scale
    // After pass 1, temp is 8×4. DCT each row of 4, then store transposed → 4×8 output.
    for row in 0..8 {
        let s = row * 4;
        let r = dct1d_4_val(temp[s], temp[s + 1], temp[s + 2], temp[s + 3]);
        // Final transpose (ROWS < COLS): store as col-major in output
        for col in 0..4 {
            output[col * 8 + row] = r[col] * (1.0 / 4.0);
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
#[inline(always)]
pub fn dct_8x4(input: &[f32; 32], output: &mut [f32; 32]) {
    // Pass 1: row DCTs (4-point) + transpose + scale into temp
    // Row DCTs produce 8 rows × 4 cols; we store transposed: 4 rows × 8 cols
    let mut temp = [0.0f32; 32];
    for row in 0..8 {
        let s = row * 4;
        let r = dct1d_4_val(input[s], input[s + 1], input[s + 2], input[s + 3]);
        // Store transposed with 1/4 scale
        for col in 0..4 {
            temp[col * 8 + row] = r[col] * (1.0 / 4.0);
        }
    }

    // Pass 2: column DCTs (8-point) + scale → output (no final transpose for ROWS >= COLS)
    for row in 0..4 {
        let s = row * 8;
        let r = dct1d_8_val([
            temp[s],
            temp[s + 1],
            temp[s + 2],
            temp[s + 3],
            temp[s + 4],
            temp[s + 5],
            temp[s + 6],
            temp[s + 7],
        ]);
        for col in 0..8 {
            output[row * 8 + col] = r[col] * (1.0 / 8.0);
        }
    }
}

/// Compute full DCT4X8 transform for 8x8 pixel block.
///
/// This covers an 8x8 pixel region using TWO vertically-stacked 4x8 sub-blocks.
/// The DC values of the two sub-blocks are combined with a 2-point transform.
///
/// Matches libjxl's Type::DCT4X8 case in enc_transforms-inl.h
#[inline(always)]
pub fn dct_4x8_full(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::dct_4x8_full(input, output);
}

/// Compute full DCT8X4 transform for 8x8 pixel block.
///
/// This covers an 8x8 pixel region using TWO horizontally-adjacent 8x4 sub-blocks.
/// The DC values of the two sub-blocks are combined with a 2-point transform.
///
/// Matches libjxl's Type::DCT8X4 case in enc_transforms-inl.h
#[inline(always)]
pub fn dct_8x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::dct_8x4_full(input, output);
}

/// Extract DC value from DCT4X8 full transform coefficients.
///
/// For DCT4X8 (and DCT8X4), the 8x8 block is covered by a single 1x1 DC region.
/// The DC combining step already produced the DC at position `[0]`.
#[inline]
pub fn dc_from_dct_4x8_full(coeffs: &[f32; 64]) -> f32 {
    coeffs[0]
}

/// Extract DC value from DCT8X4 full transform coefficients.
///
/// Same as DCT4X8 - single DC at position `[0]`.
#[inline]
pub fn dc_from_dct_8x4_full(coeffs: &[f32; 64]) -> f32 {
    coeffs[0]
}

/// Compute base 4x4 DCT.
///
/// Input: 4x4 = 16 floats in row-major order (stride 4)
/// Output: 16 DCT coefficients in transposed layout (no final transpose for square)
///
/// Based on libjxl's ComputeScaledDCT<4, 4>.
#[inline(always)]
pub fn dct_4x4(input: &[f32; 16], output: &mut [f32; 16]) {
    // Pass 1: row DCTs + transpose + scale into temp
    let mut temp = [0.0f32; 16];
    for row in 0..4 {
        let s = row * 4;
        let r = dct1d_4_val(input[s], input[s + 1], input[s + 2], input[s + 3]);
        // Store transposed with 1/4 scale
        for col in 0..4 {
            temp[col * 4 + row] = r[col] * (1.0 / 4.0);
        }
    }

    // Pass 2: column DCTs + scale → output (no final transpose for square)
    for row in 0..4 {
        let s = row * 4;
        let r = dct1d_4_val(temp[s], temp[s + 1], temp[s + 2], temp[s + 3]);
        for col in 0..4 {
            output[row * 4 + col] = r[col] * (1.0 / 4.0);
        }
    }
}

/// Compute full DCT4X4 transform for 8x8 pixel block.
///
/// This covers an 8x8 pixel region using FOUR 4x4 sub-blocks arranged in a 2x2 grid.
/// The DC values of the four sub-blocks are combined with a 2x2 DCT.
///
/// Matches libjxl's Type::DCT4X4 case in enc_transforms-inl.h
#[inline(always)]
pub fn dct_4x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::dct_4x4_full(input, output);
}

/// Extract DC value from DCT4X4 full transform coefficients.
///
/// For DCT4X4, the 8x8 block has a 2x2 LLF region at positions `[0,1,8,9]`.
/// The DC (average) is at position `[0]`.
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
///   `dcs[0]` → (by, bx), `dcs[1]` → (by, bx+1), `dcs[2]` → (by+1, bx), `dcs[3]` → (by+1, bx+1).
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
