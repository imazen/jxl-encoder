// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Inverse DCT transforms for all sizes.

use super::constants::*;

// ============================================================================
// Inverse DCT (IDCT) implementations for pixel-domain loss calculation
// ============================================================================
//
// NOTE: These IDCT functions use a simple reference implementation. The
// roundtrip (DCT → IDCT) is not perfectly accurate for all inputs due to
// scaling differences with the optimized forward DCT. For the pixel-domain
// loss in EstimateEntropy, relative magnitudes matter more than exact values.
// If exact roundtrip is needed, the scaling factor should be calibrated
// empirically or a matched IDCT algorithm should be implemented.

/// Fast 1D IDCT for N=2 (exactly reverses dct1d_2).
/// Forward: [a, b] → [a+b, a-b]
/// Inverse: [x, y] → [(x+y)/2, (x-y)/2]
#[inline]
pub fn idct1d_2(mem: &mut [f32]) {
    let x = mem[0];
    let y = mem[1];
    mem[0] = (x + y) * 0.5;
    mem[1] = (x - y) * 0.5;
}

/// Fast 1D IDCT for N=4 (exactly reverses dct1d_4).
pub fn idct1d_4(mem: &mut [f32]) {
    // Reverse step 7 (interleave): tmp = [mem[0], mem[2], mem[1], mem[3]]
    let mut tmp = [mem[0], mem[2], mem[1], mem[3]];

    // Reverse step 6 (B transform): original was tmp[2] = sqrt2*tmp[2] + tmp[3]
    tmp[2] = (tmp[2] - tmp[3]) / SQRT2;

    // Reverse step 5: idct on second half
    idct1d_2(&mut tmp[2..4]);

    // Reverse step 4: divide by WcMultipliers
    tmp[2] /= WC_MULTIPLIERS_4[0];
    tmp[3] /= WC_MULTIPLIERS_4[1];

    // Reverse step 3: idct on first half
    idct1d_2(&mut tmp[0..2]);

    // Reverse steps 1-2: combine even/odd
    // Forward: e0 = a+d, e1 = b+c, o0 = a-d, o1 = b-c
    // Inverse: a = (e0+o0)/2, d = (e0-o0)/2, b = (e1+o1)/2, c = (e1-o1)/2
    let e0 = tmp[0];
    let e1 = tmp[1];
    let o0 = tmp[2];
    let o1 = tmp[3];
    mem[0] = (e0 + o0) * 0.5;
    mem[3] = (e0 - o0) * 0.5;
    mem[1] = (e1 + o1) * 0.5;
    mem[2] = (e1 - o1) * 0.5;
}

/// Core 1D IDCT for N=8 without the N scaling factor.
///
/// This reverses the dct1d_8 butterfly operations only, without compensating
/// for the 1/N scaling applied by the 2D wrapper (dct_8x8). Used internally
/// by idct1d_16 which applies its own scaling.
fn idct1d_8_core(mem: &mut [f32]) {
    // Reverse step 7 (interleave)
    let mut tmp = [0.0f32; 8];
    for i in 0..4 {
        tmp[i] = mem[2 * i];
        tmp[4 + i] = mem[2 * i + 1];
    }

    // Reverse step 6 (B transform)
    tmp[6] -= tmp[7];
    tmp[5] -= tmp[6];
    tmp[4] = (tmp[4] - tmp[5]) / SQRT2;

    // Reverse step 5: idct on second half
    idct1d_4(&mut tmp[4..8]);

    // Reverse step 4: divide by WcMultipliers
    for i in 0..4 {
        tmp[4 + i] /= WC_MULTIPLIERS_8[i];
    }

    // Reverse step 3: idct on first half
    idct1d_4(&mut tmp[0..4]);

    // Reverse steps 1-2: combine even/odd
    for i in 0..4 {
        mem[i] = (tmp[i] + tmp[4 + i]) * 0.5;
        mem[7 - i] = (tmp[i] - tmp[4 + i]) * 0.5;
    }
}

/// Fast 1D IDCT for N=8 (exactly reverses dct1d_8).
///
/// Includes the *= 8 scaling to compensate for the 1/8 applied by dct_8x8.
pub fn idct1d_8(mem: &mut [f32]) {
    // Scale by N to compensate for 1/N in forward transform
    for x in mem.iter_mut().take(8) {
        *x *= 8.0;
    }
    idct1d_8_core(mem);
}

/// Fast 1D IDCT for N=16 (exactly reverses dct1d_16).
///
/// Includes *= 16 scaling to compensate for the 1/16 applied by dct_16x16.
/// Uses idct1d_8_core (without 8x scaling) for the recursive sub-transforms
/// since the scaling is handled at this level.
pub fn idct1d_16(mem: &mut [f32]) {
    // Scale by N to compensate for 1/N in forward transform
    for x in mem.iter_mut().take(16) {
        *x *= 16.0;
    }

    // Reverse step 7 (interleave): deinterleave
    let mut tmp = [0.0f32; 16];
    for i in 0..8 {
        tmp[i] = mem[2 * i];
        tmp[8 + i] = mem[2 * i + 1];
    }

    // Reverse step 6 (B transform):
    // Forward: tmp[8] = sqrt2*tmp[8] + tmp[9]; tmp[8+i] += tmp[8+i+1] for i in 1..7
    // Reverse: tmp[8+i] -= tmp[8+i+1] for i in (1..7).rev(); tmp[8] = (tmp[8] - tmp[9]) / sqrt2
    for i in (1..7).rev() {
        tmp[8 + i] -= tmp[8 + i + 1];
    }
    tmp[8] = (tmp[8] - tmp[9]) / SQRT2;

    // Reverse step 5: idct on second half (use core without 8x scaling)
    idct1d_8_core(&mut tmp[8..16]);

    // Reverse step 4: divide by WcMultipliers
    for i in 0..8 {
        tmp[8 + i] /= WC_MULTIPLIERS_16[i];
    }

    // Reverse step 3: idct on first half (use core without 8x scaling)
    idct1d_8_core(&mut tmp[0..8]);

    // Reverse steps 1-2: combine AddReverse/SubReverse
    // Forward: even[i] = mem[i] + mem[15-i], odd[i] = mem[i] - mem[15-i]
    // Inverse: mem[i] = (even[i] + odd[i])/2, mem[15-i] = (even[i] - odd[i])/2
    for i in 0..8 {
        mem[i] = (tmp[i] + tmp[8 + i]) * 0.5;
        mem[15 - i] = (tmp[i] - tmp[8 + i]) * 0.5;
    }
}

/// Reference 8-point 1D IDCT (formula-based, for use in larger IDCTs).
/// Input and output are separate arrays.
#[allow(clippy::needless_range_loop)]
fn idct1d_8_ref(input: &[f32], output: &mut [f32]) {
    let n = 8usize;
    let pi = core::f32::consts::PI;

    for k in 0..n {
        let mut sum = 0.5 * input[0];
        for j in 1..n {
            let angle = pi * (j as f32) * ((2 * k + 1) as f32) / (2.0 * n as f32);
            sum += input[j] * angle.cos();
        }
        output[k] = sum;
    }
}

/// Compute 8x8 inverse DCT (exactly reverses dct_8x8).
///
/// This uses the fast matched IDCT that exactly reverses our forward DCT algorithm.
/// Roundtrip error is essentially zero (floating point precision only).
pub fn idct_8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::idct_8x8(input, output);
}

/// Compute 16x16 inverse DCT (exactly reverses dct_16x16).
pub fn idct_16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    jxl_simd::idct_16x16(input, output);
}

/// Compute 16x8 inverse DCT (16 rows x 8 cols, exactly reverses dct_16x8).
pub fn idct_16x8(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut tmp = [0.0f32; 128];

    // Apply 8-point IDCT to each row
    for row in 0..16 {
        let row_start = row * 8;
        tmp[row_start..row_start + 8].copy_from_slice(&input[row_start..row_start + 8]);
        idct1d_8(&mut tmp[row_start..row_start + 8]);
    }

    // Apply 16-point IDCT to each column (in-place via temporary column buffer)
    for col in 0..8 {
        let mut col_buf = [0.0f32; 16];
        for row in 0..16 {
            col_buf[row] = tmp[row * 8 + col];
        }
        idct1d_16(&mut col_buf);
        for row in 0..16 {
            output[row * 8 + col] = col_buf[row];
        }
    }
}

/// Compute 8x16 inverse DCT (8 rows x 16 cols, exactly reverses dct_8x16).
pub fn idct_8x16(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut tmp = [0.0f32; 128];

    // Apply 16-point IDCT to each row
    for row in 0..8 {
        let row_start = row * 16;
        tmp[row_start..row_start + 16].copy_from_slice(&input[row_start..row_start + 16]);
        idct1d_16(&mut tmp[row_start..row_start + 16]);
    }

    // Apply 8-point IDCT to each column (in-place via temporary column buffer)
    for col in 0..16 {
        let mut col_buf = [0.0f32; 8];
        for row in 0..8 {
            col_buf[row] = tmp[row * 16 + col];
        }
        idct1d_8(&mut col_buf);
        for row in 0..8 {
            output[row * 16 + col] = col_buf[row];
        }
    }
}

/// Compute 4x4 inverse DCT (exactly reverses dct_4x4).
/// Input layout: 4 rows x 4 cols, stride 4.
pub fn idct_4x4(input: &[f32; 16], output: &mut [f32; 16]) {
    let mut tmp = [0.0f32; 16];

    // Apply 4-point IDCT to each row
    // Scale by 4 to compensate for the 1/4 scaling in forward transform
    for row in 0..4 {
        let row_start = row * 4;
        for i in 0..4 {
            tmp[row_start + i] = input[row_start + i] * 4.0;
        }
        idct1d_4(&mut tmp[row_start..row_start + 4]);
    }

    // Transpose
    for row in 0..4 {
        for col in 0..4 {
            output[col * 4 + row] = tmp[row * 4 + col];
        }
    }

    // Apply 4-point IDCT to each row of transposed (now columns)
    // Scale by 4 to compensate for the 1/4 scaling in forward transform
    for row in 0..4 {
        let row_start = row * 4;
        for i in 0..4 {
            output[row_start + i] *= 4.0;
        }
        idct1d_4(&mut output[row_start..row_start + 4]);
    }
}

/// Compute 4x8 inverse DCT (exactly reverses dct_4x8).
/// Input layout: 4 rows x 8 cols, stride 8.
///
/// dct_4x8 does:
///   1. 8-point DCT on rows, then *= 1/8
///   2. Transpose 4x8 -> 8x4
///   3. 4-point DCT on rows of transposed, then *= 1/4
///   4. Transpose 8x4 -> 4x8
///
/// So idct_4x8 must reverse these steps:
///   1. Transpose 4x8 -> 8x4
///   2. *= 4, then 4-point IDCT on rows
///   3. Transpose 8x4 -> 4x8
///   4. 8-point IDCT on rows (includes internal *= 8)
pub fn idct_4x8(input: &[f32; 32], output: &mut [f32; 32]) {
    // Step 1: Transpose 4x8 -> 8x4
    let mut transposed = [0.0f32; 32];
    for row in 0..4 {
        for col in 0..8 {
            transposed[col * 4 + row] = input[row * 8 + col];
        }
    }

    // Step 2: *= 4, then 4-point IDCT on each row
    for row in 0..8 {
        let row_start = row * 4;
        for i in 0..4 {
            transposed[row_start + i] *= 4.0;
        }
        idct1d_4(&mut transposed[row_start..row_start + 4]);
    }

    // Step 3: Transpose 8x4 -> 4x8
    let mut tmp = [0.0f32; 32];
    for row in 0..8 {
        for col in 0..4 {
            tmp[col * 8 + row] = transposed[row * 4 + col];
        }
    }

    // Step 4: 8-point IDCT on each row (includes internal *= 8)
    for row in 0..4 {
        let row_start = row * 8;
        output[row_start..row_start + 8].copy_from_slice(&tmp[row_start..row_start + 8]);
        idct1d_8(&mut output[row_start..row_start + 8]);
    }
}

/// Compute 8x4 inverse DCT (exactly reverses dct_8x4).
/// Input layout: 4 rows x 8 cols, stride 8 (output of dct_8x4 which has no final transpose).
///
/// dct_8x4 (ROWS=8 >= COLS=4, no final transpose):
///
///   1. 4pt DCT on rows (8 rows of 4), *= 1/4
///   2. Transpose 8x4 -> 4x8
///   3. 8pt DCT on rows (4 rows of 8), *= 1/8
///
/// No final transpose. Output is 4x8 (stride 8).
pub fn idct_8x4(input: &[f32; 32], output: &mut [f32; 32]) {
    let mut tmp = [0.0f32; 32];

    // Step 1: 8pt IDCT on each of 4 rows (stride 8)
    for row in 0..4 {
        let s = row * 8;
        tmp[s..s + 8].copy_from_slice(&input[s..s + 8]);
        idct1d_8(&mut tmp[s..s + 8]);
    }

    // Step 2: Transpose 4x8 -> 8x4
    let mut transposed = [0.0f32; 32];
    for row in 0..4 {
        for col in 0..8 {
            transposed[col * 4 + row] = tmp[row * 8 + col];
        }
    }

    // Step 3: *= 4, then 4pt IDCT on each of 8 rows (stride 4)
    for row in 0..8 {
        let s = row * 4;
        for i in 0..4 {
            transposed[s + i] *= 4.0;
        }
        idct1d_4(&mut transposed[s..s + 4]);
    }

    output.copy_from_slice(&transposed);
}

/// Core 1D IDCT for N=16 without the N scaling factor.
/// Used internally by idct1d_32 which applies its own scaling.
fn idct1d_16_core(mem: &mut [f32]) {
    // Reverse step 7 (interleave): deinterleave
    let mut tmp = [0.0f32; 16];
    for i in 0..8 {
        tmp[i] = mem[2 * i];
        tmp[8 + i] = mem[2 * i + 1];
    }

    // Reverse step 6 (B transform)
    for i in (1..7).rev() {
        tmp[8 + i] -= tmp[8 + i + 1];
    }
    tmp[8] = (tmp[8] - tmp[9]) / SQRT2;

    // Reverse step 5: idct on second half
    idct1d_8_core(&mut tmp[8..16]);

    // Reverse step 4: divide by WcMultipliers
    for i in 0..8 {
        tmp[8 + i] /= WC_MULTIPLIERS_16[i];
    }

    // Reverse step 3: idct on first half
    idct1d_8_core(&mut tmp[0..8]);

    // Reverse steps 1-2: combine
    for i in 0..8 {
        mem[i] = (tmp[i] + tmp[8 + i]) * 0.5;
        mem[15 - i] = (tmp[i] - tmp[8 + i]) * 0.5;
    }
}

/// Fast 1D IDCT for N=32 (exactly reverses dct1d_32).
///
/// Includes *= 32 scaling to compensate for the 1/32 applied by dct_32x32.
fn idct1d_32(mem: &mut [f32]) {
    for x in mem.iter_mut().take(32) {
        *x *= 32.0;
    }
    idct1d_32_core(mem);
}

/// Core 1D IDCT for N=32 without the N scaling factor.
fn idct1d_32_core(mem: &mut [f32]) {
    let mut tmp = [0.0f32; 32];
    for i in 0..16 {
        tmp[i] = mem[2 * i];
        tmp[16 + i] = mem[2 * i + 1];
    }

    // Reverse B transform
    for i in (1..15).rev() {
        tmp[16 + i] -= tmp[16 + i + 1];
    }
    tmp[16] = (tmp[16] - tmp[17]) / SQRT2;

    // IDCT on second half
    idct1d_16_core(&mut tmp[16..32]);

    // Divide by WcMultipliers
    for i in 0..16 {
        tmp[16 + i] /= WC_MULTIPLIERS_32[i];
    }

    // IDCT on first half
    idct1d_16_core(&mut tmp[0..16]);

    // Combine
    for i in 0..16 {
        mem[i] = (tmp[i] + tmp[16 + i]) * 0.5;
        mem[31 - i] = (tmp[i] - tmp[16 + i]) * 0.5;
    }
}

/// Compute 32x32 inverse DCT (exactly reverses dct_32x32).
pub fn idct_32x32(input: &[f32; 1024], output: &mut [f32; 1024]) {
    let mut tmp = [0.0f32; 1024];

    for row in 0..32 {
        let s = row * 32;
        tmp[s..s + 32].copy_from_slice(&input[s..s + 32]);
        idct1d_32(&mut tmp[s..s + 32]);
    }

    let mut transposed = [0.0f32; 1024];
    transpose::<32, 32>(&tmp, &mut transposed);

    for row in 0..32 {
        let s = row * 32;
        output[s..s + 32].copy_from_slice(&transposed[s..s + 32]);
        idct1d_32(&mut output[s..s + 32]);
    }
}

/// Compute 32x16 inverse DCT (exactly reverses dct_32x16).
///
/// dct_32x16 (ROWS=32 >= COLS=16, no final transpose):
///
///   1. 16pt DCT on rows (32 rows of 16), *= 1/16
///   2. Transpose 32x16 -> 16x32
///   3. 32pt DCT on rows (16 rows of 32), *= 1/32
///
/// Output: 16x32 (stride 32).
pub fn idct_32x16(input: &[f32; 512], output: &mut [f32; 512]) {
    let mut tmp = [0.0f32; 512];

    // 32pt IDCT on each of 16 rows (stride 32)
    for row in 0..16 {
        let s = row * 32;
        tmp[s..s + 32].copy_from_slice(&input[s..s + 32]);
        idct1d_32(&mut tmp[s..s + 32]);
    }

    // Transpose 16x32 -> 32x16
    let mut transposed = [0.0f32; 512];
    for row in 0..16 {
        for col in 0..32 {
            transposed[col * 16 + row] = tmp[row * 32 + col];
        }
    }

    // 16pt IDCT on each of 32 rows (stride 16)
    for row in 0..32 {
        let s = row * 16;
        output[s..s + 16].copy_from_slice(&transposed[s..s + 16]);
        idct1d_16(&mut output[s..s + 16]);
    }
}

/// Compute 16x32 inverse DCT (exactly reverses dct_16x32).
///
/// dct_16x32 (ROWS=16 < COLS=32, WITH final transpose):
///   1. 32pt DCT on rows, *= 1/32
///   2. Transpose 16x32 -> 32x16
///   3. 16pt DCT on rows, *= 1/16
///   4. Transpose 32x16 -> 16x32
pub fn idct_16x32(input: &[f32; 512], output: &mut [f32; 512]) {
    // Undo final transpose: 16x32 -> 32x16
    let mut transposed = [0.0f32; 512];
    for row in 0..16 {
        for col in 0..32 {
            transposed[col * 16 + row] = input[row * 32 + col];
        }
    }

    // 16pt IDCT on each of 32 rows (stride 16)
    let mut tmp = [0.0f32; 512];
    for row in 0..32 {
        let s = row * 16;
        tmp[s..s + 16].copy_from_slice(&transposed[s..s + 16]);
        idct1d_16(&mut tmp[s..s + 16]);
    }

    // Transpose 32x16 -> 16x32
    let mut transposed2 = [0.0f32; 512];
    for row in 0..32 {
        for col in 0..16 {
            transposed2[col * 32 + row] = tmp[row * 16 + col];
        }
    }

    // 32pt IDCT on each of 16 rows (stride 32)
    for row in 0..16 {
        let s = row * 32;
        output[s..s + 32].copy_from_slice(&transposed2[s..s + 32]);
        idct1d_32(&mut output[s..s + 32]);
    }
}

/// Fast 1D IDCT for N=64 (exactly reverses dct1d_64).
fn idct1d_64(mem: &mut [f32]) {
    for x in mem.iter_mut().take(64) {
        *x *= 64.0;
    }
    idct1d_64_core(mem);
}

/// Core 1D IDCT for N=64 without the N scaling factor.
fn idct1d_64_core(mem: &mut [f32]) {
    let mut tmp = [0.0f32; 64];
    for i in 0..32 {
        tmp[i] = mem[2 * i];
        tmp[32 + i] = mem[2 * i + 1];
    }

    // Reverse B transform
    for i in (1..31).rev() {
        tmp[32 + i] -= tmp[32 + i + 1];
    }
    tmp[32] = (tmp[32] - tmp[33]) / SQRT2;

    // IDCT on second half
    idct1d_32_core(&mut tmp[32..64]);

    // Divide by WcMultipliers
    for i in 0..32 {
        tmp[32 + i] /= WC_MULTIPLIERS_64[i];
    }

    // IDCT on first half
    idct1d_32_core(&mut tmp[0..32]);

    // Combine
    for i in 0..32 {
        mem[i] = (tmp[i] + tmp[32 + i]) * 0.5;
        mem[63 - i] = (tmp[i] - tmp[32 + i]) * 0.5;
    }
}

/// Compute 64x64 inverse DCT (exactly reverses dct_64x64).
pub fn idct_64x64(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 4096);
    debug_assert!(output.len() >= 4096);

    let mut tmp = [0.0f32; 4096];

    for row in 0..64 {
        let s = row * 64;
        tmp[s..s + 64].copy_from_slice(&input[s..s + 64]);
        idct1d_64(&mut tmp[s..s + 64]);
    }

    let mut transposed = [0.0f32; 4096];
    transpose::<64, 64>(&tmp, &mut transposed);

    for row in 0..64 {
        let s = row * 64;
        output[s..s + 64].copy_from_slice(&transposed[s..s + 64]);
        idct1d_64(&mut output[s..s + 64]);
    }
}

/// Compute 64x32 inverse DCT (exactly reverses dct_64x32).
///
/// dct_64x32 (ROWS=64 >= COLS=32, no final transpose):
///   Output: 32x64 (stride 64).
pub fn idct_64x32(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 2048);
    debug_assert!(output.len() >= 2048);

    let mut tmp = [0.0f32; 2048];

    // 64pt IDCT on each of 32 rows (stride 64)
    for row in 0..32 {
        let s = row * 64;
        tmp[s..s + 64].copy_from_slice(&input[s..s + 64]);
        idct1d_64(&mut tmp[s..s + 64]);
    }

    // Transpose 32x64 -> 64x32
    let mut transposed = [0.0f32; 2048];
    for row in 0..32 {
        for col in 0..64 {
            transposed[col * 32 + row] = tmp[row * 64 + col];
        }
    }

    // 32pt IDCT on each of 64 rows (stride 32)
    for row in 0..64 {
        let s = row * 32;
        output[s..s + 32].copy_from_slice(&transposed[s..s + 32]);
        idct1d_32(&mut output[s..s + 32]);
    }
}

/// Compute 32x64 inverse DCT (exactly reverses dct_32x64).
///
/// dct_32x64 (ROWS=32 < COLS=64, WITH final transpose).
pub fn idct_32x64(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 2048);
    debug_assert!(output.len() >= 2048);

    // Undo final transpose: 32x64 -> 64x32
    let mut transposed = [0.0f32; 2048];
    for row in 0..32 {
        for col in 0..64 {
            transposed[col * 32 + row] = input[row * 64 + col];
        }
    }

    // 32pt IDCT on each of 64 rows (stride 32)
    let mut tmp = [0.0f32; 2048];
    for row in 0..64 {
        let s = row * 32;
        tmp[s..s + 32].copy_from_slice(&transposed[s..s + 32]);
        idct1d_32(&mut tmp[s..s + 32]);
    }

    // Transpose 64x32 -> 32x64
    let mut transposed2 = [0.0f32; 2048];
    for row in 0..64 {
        for col in 0..32 {
            transposed2[col * 64 + row] = tmp[row * 32 + col];
        }
    }

    // 64pt IDCT on each of 32 rows (stride 64)
    for row in 0..32 {
        let s = row * 64;
        output[s..s + 64].copy_from_slice(&transposed2[s..s + 64]);
        idct1d_64(&mut output[s..s + 64]);
    }
}

/// Generic N-point 1D IDCT reference implementation.
#[allow(clippy::needless_range_loop)]
fn idct1d_n_ref(input: &[f32], output: &mut [f32], n: usize) {
    let pi = core::f32::consts::PI;

    // Explicit indices for mathematical clarity (k, j are frequency/position indices)
    for k in 0..n {
        let mut sum = 0.5 * input[0];
        for j in 1..n {
            let angle = pi * (j as f32) * ((2 * k + 1) as f32) / (2.0 * n as f32);
            sum += input[j] * angle.cos();
        }
        output[k] = sum;
    }
}

/// Extract DC value from 8x8 DCT coefficients.
/// For DCT8, DC is just the [0,0] coefficient.
#[inline]
pub fn dc_from_dct_8x8(coeffs: &[f32; 64]) -> f32 {
    coeffs[0]
}

/// Extract DC values from 16x8 DCT coefficients.
/// Returns 2 DC values (for the 2 covered 8x8 blocks).
///
/// Uses ReinterpretingIDCT to convert LF coefficients to DC.
pub fn dc_from_dct_16x8(coeffs: &[f32; 128]) -> [f32; 2] {
    // For 16x8, the LF region is 2x1 coefficients (2 rows, 1 col in freq domain)
    // In the 8×16 output layout (stride 16), both LLF coefficients are at indices 0 and 1.
    //
    // C++ DCFromLowestFrequencies uses DCTTotalResampleScale<16, 2> (forward direction:
    // FROM 16-point DCT TO 2-point domain). Must use 16_TO_2 scales, NOT 2_TO_16.
    let lf0 = coeffs[0] * DCT_RESAMPLE_SCALE_16_TO_2[0];
    let lf1 = coeffs[1] * DCT_RESAMPLE_SCALE_16_TO_2[1];

    // 2-point IDCT: [a+b, a-b]
    [lf0 + lf1, lf0 - lf1]
}

/// Extract DC values from 8x16 DCT coefficients.
/// Returns 2 DC values (for the 2 covered 8x8 blocks).
pub fn dc_from_dct_8x16(coeffs: &[f32; 128]) -> [f32; 2] {
    // For 8x16, the LF region is 1x2 coefficients
    // Uses 16_TO_2 direction (FROM 16-point DCT TO 2-point domain).
    let lf0 = coeffs[0] * DCT_RESAMPLE_SCALE_16_TO_2[0];
    let lf1 = coeffs[1] * DCT_RESAMPLE_SCALE_16_TO_2[1];

    // 2-point IDCT: [a+b, a-b]
    [lf0 + lf1, lf0 - lf1]
}

fn idct1d_4_ref(input: &[f32; 4], output: &mut [f32; 4]) {
    // The unnormalized type-III DCT of length 4:
    // X[k] = x[0] + sum_{n=1..3} x[n] * 2 * cos(pi * n * (2k+1) / 8) for k=0..3
    //
    // We use the butterfly decomposition matching libjxl's ComputeScaledIDCT:
    // Stage 1: B-transform (reverse of forward B-transform)
    // Stage 2: EvenOdd separation
    // Stage 3: WC multiply
    // Stage 4: IDCT on halves
    // Stage 5: AddSubReverse

    // For 4-point: direct computation is clearest.
    // cos(pi/8) = cos(22.5°), cos(3pi/8) = cos(67.5°)
    let c1 = core::f32::consts::FRAC_PI_8.cos(); // cos(pi/8) ≈ 0.9239
    let c3 = (3.0 * core::f32::consts::FRAC_PI_8).cos(); // cos(3pi/8) ≈ 0.3827

    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    // IDCT-III formula: out[k] = x[0] + 2 * sum_{n=1..N-1} x[n] * cos(pi*n*(2k+1)/(2N))
    // For N=4:
    output[0] = x0
        + 2.0
            * (x1 * (core::f32::consts::PI * 1.0 / 8.0).cos()
                + x2 * (core::f32::consts::PI * 2.0 / 8.0).cos()
                + x3 * (core::f32::consts::PI * 3.0 / 8.0).cos());
    output[1] = x0
        + 2.0
            * (x1 * (core::f32::consts::PI * 3.0 / 8.0).cos()
                + x2 * (core::f32::consts::PI * 6.0 / 8.0).cos()
                + x3 * (core::f32::consts::PI * 9.0 / 8.0).cos());
    output[2] = x0
        + 2.0
            * (x1 * (core::f32::consts::PI * 5.0 / 8.0).cos()
                + x2 * (core::f32::consts::PI * 10.0 / 8.0).cos()
                + x3 * (core::f32::consts::PI * 15.0 / 8.0).cos());
    output[3] = x0
        + 2.0
            * (x1 * (core::f32::consts::PI * 7.0 / 8.0).cos()
                + x2 * (core::f32::consts::PI * 14.0 / 8.0).cos()
                + x3 * (core::f32::consts::PI * 21.0 / 8.0).cos());

    // Suppress unused variable warning
    let _ = (c1, c3);
}
