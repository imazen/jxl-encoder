// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD-accelerated 16x16 forward DCT.
//!
//! Processes 8 independent 16-point DCTs in parallel using AVX2 f32x8 vectors.
//! Each f32x8 lane holds one row's element at a given column position, so the
//! butterfly operates across registers (cross-position) while SIMD parallelism
//! handles multiple rows simultaneously.
//!
//! This is the forward counterpart of `idct16.rs`. The forward DCT uses:
//! - AddReverse/SubReverse butterfly (not de-interleave)
//! - Direct WC_MULTIPLIERS multiplication (not division by inverse)
//! - B transform AFTER inner DCT (not reverse B before)
//! - InverseEvenOdd interleave at END (not de-interleave at start)

// Constants matching jxl_encoder/src/vardct/dct/constants.rs
const SQRT2: f32 = core::f32::consts::SQRT_2;
const WC_MULTIPLIERS_4: [f32; 2] = [0.541_196_1, 1.306_563];
const WC_MULTIPLIERS_8: [f32; 4] = [0.509_795_6, 0.601_344_9, 0.899_976_2, 2.562_915_5];
const WC_MULTIPLIERS_16: [f32; 8] = [
    0.502_419_3,
    0.522_498_6,
    0.566_944_06,
    0.646_821_8,
    0.788_154_65,
    1.060_677_7,
    1.722_447_1,
    5.101_148_6,
];

/// Compute 16x16 forward DCT with SIMD acceleration.
///
/// Input: 256 f32 in row-major order (spatial domain).
/// Output: 256 f32 in transposed layout (coefficient domain).
/// No final transpose for square blocks, matching libjxl convention.
/// Dispatches to AVX2 when available; falls back to scalar otherwise.
pub fn dct_16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_16x16_avx2(token, input, output);
            return;
        }
    }

    dct_16x16_scalar(input, output);
}

// ============================================================================
// Scalar fallback — matches jxl_encoder/src/vardct/dct/forward.rs exactly
// ============================================================================

fn dct_16x16_scalar(input: &[f32; 256], output: &mut [f32; 256]) {
    let mut tmp = [0.0f32; 256];

    // Forward DCT on each row
    for row in 0..16 {
        let s = row * 16;
        tmp[s..s + 16].copy_from_slice(&input[s..s + 16]);
        dct1d_16_scalar(&mut tmp[s..s + 16]);
        // Scale by 1/16
        for i in 0..16 {
            tmp[s + i] *= 1.0 / 16.0;
        }
    }

    // Transpose 16x16
    let mut transposed = [0.0f32; 256];
    for r in 0..16 {
        for c in 0..16 {
            transposed[c * 16 + r] = tmp[r * 16 + c];
        }
    }

    // Forward DCT on each row of transposed (columns of original)
    for row in 0..16 {
        let s = row * 16;
        dct1d_16_scalar(&mut transposed[s..s + 16]);
        // Scale by 1/16
        for i in 0..16 {
            transposed[s + i] *= 1.0 / 16.0;
        }
    }

    // No final transpose for square blocks
    output.copy_from_slice(&transposed);
}

#[inline]
fn dct1d_2_scalar(mem: &mut [f32]) {
    let a = mem[0];
    let b = mem[1];
    mem[0] = a + b;
    mem[1] = a - b;
}

fn dct1d_4_scalar(mem: &mut [f32]) {
    let mut tmp = [0.0f32; 4];
    tmp[0] = mem[0] + mem[3];
    tmp[1] = mem[1] + mem[2];
    tmp[2] = mem[0] - mem[3];
    tmp[3] = mem[1] - mem[2];

    // DCT-2 on first half
    dct1d_2_scalar(&mut tmp[0..2]);

    // Multiply second half by WcMultipliers_4
    tmp[2] *= WC_MULTIPLIERS_4[0];
    tmp[3] *= WC_MULTIPLIERS_4[1];

    // DCT-2 on second half
    dct1d_2_scalar(&mut tmp[2..4]);

    // B transform on second half
    tmp[2] = SQRT2 * tmp[2] + tmp[3];

    // InverseEvenOdd interleave
    mem[0] = tmp[0];
    mem[2] = tmp[1];
    mem[1] = tmp[2];
    mem[3] = tmp[3];
}

fn dct1d_8_scalar(mem: &mut [f32]) {
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
    dct1d_4_scalar(&mut tmp[0..4]);

    // Multiply second half by WcMultipliers_8
    for i in 0..4 {
        tmp[4 + i] *= WC_MULTIPLIERS_8[i];
    }

    // DCT on second half
    dct1d_4_scalar(&mut tmp[4..8]);

    // B transform on second half
    tmp[4] = SQRT2 * tmp[4] + tmp[5];
    tmp[5] += tmp[6];
    tmp[6] += tmp[7];

    // InverseEvenOdd interleave
    for i in 0..4 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[4 + i];
    }
}

/// 16-point forward DCT (no scaling — caller applies 1/16).
fn dct1d_16_scalar(mem: &mut [f32]) {
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
    dct1d_8_scalar(&mut tmp[0..8]);

    // Multiply second half by WcMultipliers_16
    for i in 0..8 {
        tmp[8 + i] *= WC_MULTIPLIERS_16[i];
    }

    // DCT on second half
    dct1d_8_scalar(&mut tmp[8..16]);

    // B transform on second half
    tmp[8] = SQRT2 * tmp[8] + tmp[9];
    for i in 1..7 {
        tmp[8 + i] += tmp[8 + i + 1];
    }

    // InverseEvenOdd interleave
    for i in 0..8 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[8 + i];
    }
}

// ============================================================================
// x86_64 AVX2 implementation — batched 16-point forward DCT, 8 rows at a time
// ============================================================================

/// Load column `j` from 8 consecutive rows starting at `base_row` in `data` (stride 16).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
fn gather_col(
    token: archmage::X64V3Token,
    data: &[f32],
    base_row: usize,
    j: usize,
) -> magetypes::simd::f32x8 {
    magetypes::simd::f32x8::from_array(
        token,
        [
            data[base_row * 16 + j],
            data[(base_row + 1) * 16 + j],
            data[(base_row + 2) * 16 + j],
            data[(base_row + 3) * 16 + j],
            data[(base_row + 4) * 16 + j],
            data[(base_row + 5) * 16 + j],
            data[(base_row + 6) * 16 + j],
            data[(base_row + 7) * 16 + j],
        ],
    )
}

/// Store f32x8 lanes back to column `j` of 8 consecutive rows starting at `base_row`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scatter_col(v: magetypes::simd::f32x8, data: &mut [f32], base_row: usize, j: usize) {
    let mut lane = [0.0f32; 8];
    v.store(&mut lane);
    for r in 0..8 {
        data[(base_row + r) * 16 + j] = lane[r];
    }
}

/// AVX2 batched 4-point forward DCT.
///
/// `v` holds [v0, v1, v2, v3] representing positions 0-3 across 8 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
fn dct1d_4_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 4]) {
    use magetypes::simd::f32x8;

    let sqrt2 = f32x8::splat(token, SQRT2);
    let wc4_0 = f32x8::splat(token, WC_MULTIPLIERS_4[0]);
    let wc4_1 = f32x8::splat(token, WC_MULTIPLIERS_4[1]);

    // AddReverse / SubReverse
    let a0 = v[0] + v[3];
    let a1 = v[1] + v[2];
    let s0 = v[0] - v[3];
    let s1 = v[1] - v[2];

    // DCT-2 on first half {a0, a1}
    let fh0 = a0 + a1;
    let fh1 = a0 - a1;

    // Multiply second half by WcMultipliers_4
    let s0 = s0 * wc4_0;
    let s1 = s1 * wc4_1;

    // DCT-2 on second half {s0, s1}
    let sh0 = s0 + s1;
    let sh1 = s0 - s1;

    // B transform: sh0 = sqrt2 * sh0 + sh1
    let sh0 = sqrt2.mul_add(sh0, sh1);

    // InverseEvenOdd interleave: [fh0, sh0, fh1, sh1]
    v[0] = fh0;
    v[1] = sh0;
    v[2] = fh1;
    v[3] = sh1;
}

/// AVX2 batched 8-point forward DCT.
///
/// `v` holds [v0..v7] representing positions 0-7 across 8 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
fn dct1d_8_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 8]) {
    use magetypes::simd::f32x8;

    let sqrt2 = f32x8::splat(token, SQRT2);

    // AddReverse for first half, SubReverse for second half
    let a0 = v[0] + v[7];
    let a1 = v[1] + v[6];
    let a2 = v[2] + v[5];
    let a3 = v[3] + v[4];
    let s0 = v[0] - v[7];
    let s1 = v[1] - v[6];
    let s2 = v[2] - v[5];
    let s3 = v[3] - v[4];

    // DCT-4 on first half {a0, a1, a2, a3}
    let mut first_half = [a0, a1, a2, a3];
    dct1d_4_batch(token, &mut first_half);

    // Multiply second half by WcMultipliers_8
    let s0 = s0 * f32x8::splat(token, WC_MULTIPLIERS_8[0]);
    let s1 = s1 * f32x8::splat(token, WC_MULTIPLIERS_8[1]);
    let s2 = s2 * f32x8::splat(token, WC_MULTIPLIERS_8[2]);
    let s3 = s3 * f32x8::splat(token, WC_MULTIPLIERS_8[3]);

    // DCT-4 on second half {s0, s1, s2, s3}
    let mut second_half = [s0, s1, s2, s3];
    dct1d_4_batch(token, &mut second_half);

    // B transform on second half
    // sh[0] = sqrt2 * sh[0] + sh[1]; sh[1] += sh[2]; sh[2] += sh[3]
    second_half[0] = sqrt2.mul_add(second_half[0], second_half[1]);
    second_half[1] = second_half[1] + second_half[2];
    second_half[2] = second_half[2] + second_half[3];

    // InverseEvenOdd interleave
    v[0] = first_half[0];
    v[1] = second_half[0];
    v[2] = first_half[1];
    v[3] = second_half[1];
    v[4] = first_half[2];
    v[5] = second_half[2];
    v[6] = first_half[3];
    v[7] = second_half[3];
}

/// AVX2 batched 16-point forward DCT (no scaling — caller applies 1/16).
///
/// `v` holds [v0..v15] representing positions 0-15 across 8 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dct1d_16_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 16]) {
    use magetypes::simd::f32x8;

    let sqrt2 = f32x8::splat(token, SQRT2);

    // AddReverse for first half, SubReverse for second half
    let a0 = v[0] + v[15];
    let a1 = v[1] + v[14];
    let a2 = v[2] + v[13];
    let a3 = v[3] + v[12];
    let a4 = v[4] + v[11];
    let a5 = v[5] + v[10];
    let a6 = v[6] + v[9];
    let a7 = v[7] + v[8];
    let s0 = v[0] - v[15];
    let s1 = v[1] - v[14];
    let s2 = v[2] - v[13];
    let s3 = v[3] - v[12];
    let s4 = v[4] - v[11];
    let s5 = v[5] - v[10];
    let s6 = v[6] - v[9];
    let s7 = v[7] - v[8];

    // DCT-8 on first half {a0..a7}
    let mut first_half = [a0, a1, a2, a3, a4, a5, a6, a7];
    dct1d_8_batch(token, &mut first_half);

    // Multiply second half by WcMultipliers_16
    let s0 = s0 * f32x8::splat(token, WC_MULTIPLIERS_16[0]);
    let s1 = s1 * f32x8::splat(token, WC_MULTIPLIERS_16[1]);
    let s2 = s2 * f32x8::splat(token, WC_MULTIPLIERS_16[2]);
    let s3 = s3 * f32x8::splat(token, WC_MULTIPLIERS_16[3]);
    let s4 = s4 * f32x8::splat(token, WC_MULTIPLIERS_16[4]);
    let s5 = s5 * f32x8::splat(token, WC_MULTIPLIERS_16[5]);
    let s6 = s6 * f32x8::splat(token, WC_MULTIPLIERS_16[6]);
    let s7 = s7 * f32x8::splat(token, WC_MULTIPLIERS_16[7]);

    // DCT-8 on second half {s0..s7}
    let mut second_half = [s0, s1, s2, s3, s4, s5, s6, s7];
    dct1d_8_batch(token, &mut second_half);

    // B transform on second half
    // sh[0] = sqrt2 * sh[0] + sh[1]
    // sh[i] += sh[i+1] for i = 1..7
    second_half[0] = sqrt2.mul_add(second_half[0], second_half[1]);
    second_half[1] = second_half[1] + second_half[2];
    second_half[2] = second_half[2] + second_half[3];
    second_half[3] = second_half[3] + second_half[4];
    second_half[4] = second_half[4] + second_half[5];
    second_half[5] = second_half[5] + second_half[6];
    second_half[6] = second_half[6] + second_half[7];

    // InverseEvenOdd interleave
    v[0] = first_half[0];
    v[1] = second_half[0];
    v[2] = first_half[1];
    v[3] = second_half[1];
    v[4] = first_half[2];
    v[5] = second_half[2];
    v[6] = first_half[3];
    v[7] = second_half[3];
    v[8] = first_half[4];
    v[9] = second_half[4];
    v[10] = first_half[5];
    v[11] = second_half[5];
    v[12] = first_half[6];
    v[13] = second_half[6];
    v[14] = first_half[7];
    v[15] = second_half[7];
}

/// AVX2 16x16 forward DCT: process 8 rows at a time via batched 16-point DCT.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
fn dct_16x16_avx2(token: archmage::X64V3Token, input: &[f32; 256], output: &mut [f32; 256]) {
    use magetypes::simd::f32x8;

    let scale = f32x8::splat(token, 1.0 / 16.0);
    let mut tmp = [0.0f32; 256];

    // --- Pass 1: Forward DCT on rows ---
    // Process rows 0-7 (first batch of 8)
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, input, 0, j);
        }
        dct1d_16_batch(token, &mut v);
        for j in 0..16 {
            v[j] = v[j] * scale;
        }
        for j in 0..16 {
            scatter_col(v[j], &mut tmp, 0, j);
        }
    }

    // Process rows 8-15 (second batch of 8)
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, input, 8, j);
        }
        dct1d_16_batch(token, &mut v);
        for j in 0..16 {
            v[j] = v[j] * scale;
        }
        for j in 0..16 {
            scatter_col(v[j], &mut tmp, 8, j);
        }
    }

    // --- 16x16 scalar transpose ---
    let mut transposed = [0.0f32; 256];
    for r in 0..16 {
        for c in 0..16 {
            transposed[c * 16 + r] = tmp[r * 16 + c];
        }
    }

    // --- Pass 2: Forward DCT on columns (now rows of transposed) ---
    // Process rows 0-7
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, &transposed, 0, j);
        }
        dct1d_16_batch(token, &mut v);
        for j in 0..16 {
            v[j] = v[j] * scale;
        }
        for j in 0..16 {
            scatter_col(v[j], output, 0, j);
        }
    }

    // Process rows 8-15
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, &transposed, 8, j);
        }
        dct1d_16_batch(token, &mut v);
        for j in 0..16 {
            v[j] = v[j] * scale;
        }
        for j in 0..16 {
            scatter_col(v[j], output, 8, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_16x16_simd_matches_scalar() {
        // Sequential values 0..256 as input
        let mut input = [0.0f32; 256];
        for (i, val) in input.iter_mut().enumerate() {
            *val = i as f32;
        }

        let mut scalar_out = [0.0f32; 256];
        let mut simd_out = [0.0f32; 256];

        dct_16x16_scalar(&input, &mut scalar_out);
        dct_16x16(&input, &mut simd_out);

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        for i in 0..256 {
            let diff = (scalar_out[i] - simd_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }

        assert!(
            max_diff < 1e-2,
            "DCT16x16 SIMD vs scalar max diff = {} at index {} (scalar={}, simd={})",
            max_diff,
            max_idx,
            scalar_out[max_idx],
            simd_out[max_idx],
        );
    }

    #[test]
    fn test_dct_16x16_dc_only() {
        // All-same input: should produce nonzero only at DC (position 0)
        let input = [42.0f32; 256];

        let mut output = [0.0f32; 256];
        dct_16x16(&input, &mut output);

        // DC should be nonzero
        assert!(
            output[0].abs() > 1.0,
            "DC coefficient should be nonzero, got {}",
            output[0],
        );

        // All AC coefficients should be near zero
        let mut max_ac = 0.0f32;
        let mut max_ac_idx = 0;
        for i in 1..256 {
            let val = output[i].abs();
            if val > max_ac {
                max_ac = val;
                max_ac_idx = i;
            }
        }

        assert!(
            max_ac < 1e-3,
            "AC coefficients should be near zero, max = {} at index {}",
            max_ac,
            max_ac_idx,
        );
    }

    #[test]
    fn test_dct_16x16_roundtrip() {
        // Forward DCT then inverse DCT should recover original data
        let mut input = [0.0f32; 256];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.37 + 1.5).cos() * 100.0;
        }

        let mut dct_out = [0.0f32; 256];
        let mut roundtrip = [0.0f32; 256];

        dct_16x16(&input, &mut dct_out);
        super::super::idct16::idct_16x16(&dct_out, &mut roundtrip);

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        for i in 0..256 {
            let diff = (input[i] - roundtrip[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }

        assert!(
            max_diff < 1e-2,
            "DCT16x16 roundtrip max diff = {} at index {} (input={}, roundtrip={})",
            max_diff,
            max_idx,
            input[max_idx],
            roundtrip[max_idx],
        );
    }
}
