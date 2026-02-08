// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD-accelerated 16x16 inverse DCT.
//!
//! Processes 8 independent 16-point IDCTs in parallel using AVX2 f32x8 vectors.
//! Each f32x8 lane holds one row's element at a given column position, so the
//! butterfly operates across registers (cross-position) while SIMD parallelism
//! handles multiple rows simultaneously.

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

// Pre-computed reciprocals to replace division with multiplication.
const INV_WC4: [f32; 2] = [1.0 / 0.541_196_1, 1.0 / 1.306_563];

const INV_WC8: [f32; 4] = [
    1.0 / 0.509_795_6,
    1.0 / 0.601_344_9,
    1.0 / 0.899_976_2,
    1.0 / 2.562_915_5,
];

const INV_WC16: [f32; 8] = [
    1.0 / 0.502_419_3,
    1.0 / 0.522_498_6,
    1.0 / 0.566_944_06,
    1.0 / 0.646_821_8,
    1.0 / 0.788_154_65,
    1.0 / 1.060_677_7,
    1.0 / 1.722_447_1,
    1.0 / 5.101_148_6,
];

/// Compute 16x16 inverse DCT with SIMD acceleration.
///
/// Input: 256 f32 in row-major order (coefficient domain).
/// Output: 256 f32 in row-major order (spatial domain).
/// Dispatches to AVX2 when available; falls back to scalar otherwise.
pub fn idct_16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_16x16_avx2(token, input, output);
            return;
        }
    }

    idct_16x16_scalar(input, output);
}

// ============================================================================
// Scalar fallback — matches jxl_encoder/src/vardct/dct/inverse.rs exactly
// ============================================================================

fn idct_16x16_scalar(input: &[f32; 256], output: &mut [f32; 256]) {
    let mut tmp = [0.0f32; 256];

    // IDCT on each row
    for row in 0..16 {
        let s = row * 16;
        tmp[s..s + 16].copy_from_slice(&input[s..s + 16]);
        idct1d_16_scalar(&mut tmp[s..s + 16]);
    }

    // Transpose 16x16
    let mut transposed = [0.0f32; 256];
    for r in 0..16 {
        for c in 0..16 {
            transposed[c * 16 + r] = tmp[r * 16 + c];
        }
    }

    // IDCT on each row of transposed (columns of original)
    for row in 0..16 {
        let s = row * 16;
        output[s..s + 16].copy_from_slice(&transposed[s..s + 16]);
        idct1d_16_scalar(&mut output[s..s + 16]);
    }
}

#[inline]
fn idct1d_2_scalar(mem: &mut [f32]) {
    let x = mem[0];
    let y = mem[1];
    mem[0] = (x + y) * 0.5;
    mem[1] = (x - y) * 0.5;
}

fn idct1d_4_scalar(mem: &mut [f32]) {
    let mut tmp = [mem[0], mem[2], mem[1], mem[3]];

    // Reverse B transform
    tmp[2] = (tmp[2] - tmp[3]) / SQRT2;

    // IDCT-2 on second half
    idct1d_2_scalar(&mut tmp[2..4]);

    // Divide by WcMultipliers
    tmp[2] /= WC_MULTIPLIERS_4[0];
    tmp[3] /= WC_MULTIPLIERS_4[1];

    // IDCT-2 on first half
    idct1d_2_scalar(&mut tmp[0..2]);

    // Combine
    mem[0] = (tmp[0] + tmp[2]) * 0.5;
    mem[3] = (tmp[0] - tmp[2]) * 0.5;
    mem[1] = (tmp[1] + tmp[3]) * 0.5;
    mem[2] = (tmp[1] - tmp[3]) * 0.5;
}

/// Core 8-point IDCT without the N scaling factor.
fn idct1d_8_core_scalar(mem: &mut [f32]) {
    let mut tmp = [0.0f32; 8];
    for i in 0..4 {
        tmp[i] = mem[2 * i];
        tmp[4 + i] = mem[2 * i + 1];
    }

    // Reverse B transform
    tmp[6] -= tmp[7];
    tmp[5] -= tmp[6];
    tmp[4] = (tmp[4] - tmp[5]) / SQRT2;

    // IDCT-4 on second half
    idct1d_4_scalar(&mut tmp[4..8]);

    // Divide by WcMultipliers
    for i in 0..4 {
        tmp[4 + i] /= WC_MULTIPLIERS_8[i];
    }

    // IDCT-4 on first half
    idct1d_4_scalar(&mut tmp[0..4]);

    // Combine
    for i in 0..4 {
        mem[i] = (tmp[i] + tmp[4 + i]) * 0.5;
        mem[7 - i] = (tmp[i] - tmp[4 + i]) * 0.5;
    }
}

/// 16-point IDCT with *= 16 scaling.
fn idct1d_16_scalar(mem: &mut [f32]) {
    for x in mem.iter_mut().take(16) {
        *x *= 16.0;
    }

    let mut tmp = [0.0f32; 16];
    for i in 0..8 {
        tmp[i] = mem[2 * i];
        tmp[8 + i] = mem[2 * i + 1];
    }

    // Reverse B transform
    for i in (1..7).rev() {
        tmp[8 + i] -= tmp[8 + i + 1];
    }
    tmp[8] = (tmp[8] - tmp[9]) / SQRT2;

    // IDCT-8 core on second half
    idct1d_8_core_scalar(&mut tmp[8..16]);

    // Divide by WcMultipliers
    for i in 0..8 {
        tmp[8 + i] /= WC_MULTIPLIERS_16[i];
    }

    // IDCT-8 core on first half
    idct1d_8_core_scalar(&mut tmp[0..8]);

    // Combine
    for i in 0..8 {
        mem[i] = (tmp[i] + tmp[8 + i]) * 0.5;
        mem[15 - i] = (tmp[i] - tmp[8 + i]) * 0.5;
    }
}

// ============================================================================
// x86_64 AVX2 implementation — batched 16-point IDCT, 8 rows at a time
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

/// AVX2 batched 4-point IDCT.
///
/// `v` holds [v0, v1, v2, v3] representing positions 0-3 across 8 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
fn idct1d_4_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 4]) {
    use magetypes::simd::f32x8;

    let half = f32x8::splat(token, 0.5);
    let inv_sqrt2 = f32x8::splat(token, 1.0 / SQRT2);
    let inv_wc4_0 = f32x8::splat(token, INV_WC4[0]);
    let inv_wc4_1 = f32x8::splat(token, INV_WC4[1]);

    // De-interleave: even positions -> first half, odd positions -> second half
    // Input: [v0, v1, v2, v3] -> tmp = [v0, v2, v1, v3]
    let t0 = v[0];
    let t1 = v[2];
    let t2 = v[1];
    let t3 = v[3];

    // Reverse B transform on second half: t2 = (t2 - t3) / sqrt2
    let t2 = (t2 - t3) * inv_sqrt2;

    // IDCT-2 on second half [t2, t3]
    let s2 = (t2 + t3) * half;
    let s3 = (t2 - t3) * half;

    // Divide by WcMultipliers_4
    let s2 = s2 * inv_wc4_0;
    let s3 = s3 * inv_wc4_1;

    // IDCT-2 on first half [t0, t1]
    let s0 = (t0 + t1) * half;
    let s1 = (t0 - t1) * half;

    // Combine
    v[0] = (s0 + s2) * half;
    v[3] = (s0 - s2) * half;
    v[1] = (s1 + s3) * half;
    v[2] = (s1 - s3) * half;
}

/// AVX2 batched 8-point IDCT core (no N scaling).
///
/// `v` holds [v0..v7] representing positions 0-7 across 8 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
fn idct1d_8_core_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 8]) {
    use magetypes::simd::f32x8;

    let half = f32x8::splat(token, 0.5);
    let inv_sqrt2 = f32x8::splat(token, 1.0 / SQRT2);

    // De-interleave: even -> first_half, odd -> second_half
    let mut first_half = [v[0], v[2], v[4], v[6]];
    let mut second_half = [v[1], v[3], v[5], v[7]];

    // Reverse B transform on second half
    second_half[2] -= second_half[3];
    second_half[1] -= second_half[2];
    second_half[0] = (second_half[0] - second_half[1]) * inv_sqrt2;

    // IDCT-4 on second half
    idct1d_4_batch(token, &mut second_half);

    // Divide by WcMultipliers_8
    second_half[0] *= f32x8::splat(token, INV_WC8[0]);
    second_half[1] *= f32x8::splat(token, INV_WC8[1]);
    second_half[2] *= f32x8::splat(token, INV_WC8[2]);
    second_half[3] *= f32x8::splat(token, INV_WC8[3]);

    // IDCT-4 on first half
    idct1d_4_batch(token, &mut first_half);

    // Combine: out[i] = (first[i] + second[i]) * 0.5
    //          out[7-i] = (first[i] - second[i]) * 0.5
    v[0] = (first_half[0] + second_half[0]) * half;
    v[7] = (first_half[0] - second_half[0]) * half;
    v[1] = (first_half[1] + second_half[1]) * half;
    v[6] = (first_half[1] - second_half[1]) * half;
    v[2] = (first_half[2] + second_half[2]) * half;
    v[5] = (first_half[2] - second_half[2]) * half;
    v[3] = (first_half[3] + second_half[3]) * half;
    v[4] = (first_half[3] - second_half[3]) * half;
}

/// AVX2 batched 16-point IDCT with *= 16 scaling.
///
/// `v` holds [v0..v15] representing positions 0-15 across 8 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
fn idct1d_16_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 16]) {
    use magetypes::simd::f32x8;

    let scale16 = f32x8::splat(token, 16.0);
    let half = f32x8::splat(token, 0.5);
    let inv_sqrt2 = f32x8::splat(token, 1.0 / SQRT2);

    // Scale by 16 to compensate for 1/16 in forward transform
    for vi in v.iter_mut() {
        *vi *= scale16;
    }

    // De-interleave: even -> first_half[0..8], odd -> second_half[0..8]
    let mut first_half = [v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]];
    let mut second_half = [v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]];

    // Reverse B transform on second half
    // Forward: sh[0] = sqrt2*sh[0] + sh[1]; sh[i] += sh[i+1] for i=1..7
    // Reverse: sh[i] -= sh[i+1] for i in (1..7).rev(); sh[0] = (sh[0] - sh[1]) / sqrt2
    second_half[6] -= second_half[7];
    second_half[5] -= second_half[6];
    second_half[4] -= second_half[5];
    second_half[3] -= second_half[4];
    second_half[2] -= second_half[3];
    second_half[1] -= second_half[2];
    second_half[0] = (second_half[0] - second_half[1]) * inv_sqrt2;

    // IDCT-8 core on second half
    idct1d_8_core_batch(token, &mut second_half);

    // Divide by WcMultipliers_16
    second_half[0] *= f32x8::splat(token, INV_WC16[0]);
    second_half[1] *= f32x8::splat(token, INV_WC16[1]);
    second_half[2] *= f32x8::splat(token, INV_WC16[2]);
    second_half[3] *= f32x8::splat(token, INV_WC16[3]);
    second_half[4] *= f32x8::splat(token, INV_WC16[4]);
    second_half[5] *= f32x8::splat(token, INV_WC16[5]);
    second_half[6] *= f32x8::splat(token, INV_WC16[6]);
    second_half[7] *= f32x8::splat(token, INV_WC16[7]);

    // IDCT-8 core on first half
    idct1d_8_core_batch(token, &mut first_half);

    // Combine: out[i] = (first[i] + second[i]) * 0.5
    //          out[15-i] = (first[i] - second[i]) * 0.5
    v[0] = (first_half[0] + second_half[0]) * half;
    v[15] = (first_half[0] - second_half[0]) * half;
    v[1] = (first_half[1] + second_half[1]) * half;
    v[14] = (first_half[1] - second_half[1]) * half;
    v[2] = (first_half[2] + second_half[2]) * half;
    v[13] = (first_half[2] - second_half[2]) * half;
    v[3] = (first_half[3] + second_half[3]) * half;
    v[12] = (first_half[3] - second_half[3]) * half;
    v[4] = (first_half[4] + second_half[4]) * half;
    v[11] = (first_half[4] - second_half[4]) * half;
    v[5] = (first_half[5] + second_half[5]) * half;
    v[10] = (first_half[5] - second_half[5]) * half;
    v[6] = (first_half[6] + second_half[6]) * half;
    v[9] = (first_half[6] - second_half[6]) * half;
    v[7] = (first_half[7] + second_half[7]) * half;
    v[8] = (first_half[7] - second_half[7]) * half;
}

/// AVX2 16x16 IDCT: process 8 rows at a time via batched 16-point IDCT.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
fn idct_16x16_avx2(token: archmage::X64V3Token, input: &[f32; 256], output: &mut [f32; 256]) {
    use magetypes::simd::f32x8;

    let mut tmp = [0.0f32; 256];

    // --- Pass 1: IDCT on rows ---
    // Process rows 0-7 (first batch of 8)
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, input, 0, j);
        }
        idct1d_16_batch(token, &mut v);
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
        idct1d_16_batch(token, &mut v);
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

    // --- Pass 2: IDCT on columns (now rows of transposed) ---
    // Process rows 0-7
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, &transposed, 0, j);
        }
        idct1d_16_batch(token, &mut v);
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
        idct1d_16_batch(token, &mut v);
        for j in 0..16 {
            scatter_col(v[j], output, 8, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idct_16x16_simd_matches_scalar() {
        // Sequential values 0..256 as input
        let mut input = [0.0f32; 256];
        for (i, val) in input.iter_mut().enumerate() {
            *val = i as f32;
        }

        let mut scalar_out = [0.0f32; 256];
        let mut simd_out = [0.0f32; 256];

        idct_16x16_scalar(&input, &mut scalar_out);
        idct_16x16(&input, &mut simd_out);

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        for i in 0..256 {
            let diff = (scalar_out[i] - simd_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }

        // SIMD uses pre-computed reciprocals instead of division, so there's a small
        // difference due to extra rounding. 0.003 on a value of ~192 is relative error ~1.5e-5.
        assert!(
            max_diff < 1e-2,
            "IDCT16x16 SIMD vs scalar max diff = {} at index {} (scalar={}, simd={})",
            max_diff,
            max_idx,
            scalar_out[max_idx],
            simd_out[max_idx],
        );
    }

    #[test]
    fn test_idct_16x16_simd_matches_scalar_cosine_input() {
        // More varied input using cosine values
        let mut input = [0.0f32; 256];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.37 + 1.5).cos() * 100.0;
        }

        let mut scalar_out = [0.0f32; 256];
        let mut simd_out = [0.0f32; 256];

        idct_16x16_scalar(&input, &mut scalar_out);
        idct_16x16(&input, &mut simd_out);

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
            "IDCT16x16 cosine input max diff = {} at index {} (scalar={}, simd={})",
            max_diff,
            max_idx,
            scalar_out[max_idx],
            simd_out[max_idx],
        );
    }

    #[test]
    fn test_idct_16x16_dc_only() {
        // DC-only input: only [0,0] is nonzero
        let mut input = [0.0f32; 256];
        input[0] = 128.0;

        let mut scalar_out = [0.0f32; 256];
        let mut simd_out = [0.0f32; 256];

        idct_16x16_scalar(&input, &mut scalar_out);
        idct_16x16(&input, &mut simd_out);

        let mut max_diff = 0.0f32;
        for i in 0..256 {
            let diff = (scalar_out[i] - simd_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        assert!(max_diff < 1e-3, "IDCT16x16 DC-only max diff = {}", max_diff);

        // All output values should be the same (flat DC block)
        let dc_val = simd_out[0];
        for (i, &val) in simd_out.iter().enumerate().skip(1) {
            assert!(
                (val - dc_val).abs() < 1e-3,
                "DC-only output not uniform: [0]={}, [{}]={}",
                dc_val,
                i,
                val,
            );
        }
    }

    #[test]
    fn test_idct_16x16_single_ac_coefficient() {
        // Single AC coefficient to test frequency response
        let mut input = [0.0f32; 256];
        input[1] = 50.0; // frequency (0,1)

        let mut scalar_out = [0.0f32; 256];
        let mut simd_out = [0.0f32; 256];

        idct_16x16_scalar(&input, &mut scalar_out);
        idct_16x16(&input, &mut simd_out);

        let mut max_diff = 0.0f32;
        for i in 0..256 {
            let diff = (scalar_out[i] - simd_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        assert!(
            max_diff < 1e-3,
            "IDCT16x16 single AC max diff = {}",
            max_diff,
        );
    }
}
