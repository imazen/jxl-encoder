// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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
#[inline]
pub fn dct_16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_16x16_avx2(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            dct_16x16_neon(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            dct_16x16_wasm128(token, input, output);
            return;
        }
    }

    dct_16x16_scalar(input, output);
}

// ============================================================================
// Scalar fallback — matches jxl_encoder/src/vardct/dct/forward.rs exactly
// ============================================================================

#[inline]
pub fn dct_16x16_scalar(input: &[f32; 256], output: &mut [f32; 256]) {
    let mut tmp = crate::scratch_buf::<256>();

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
    let mut transposed = crate::scratch_buf::<256>();
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
    second_half[1] += second_half[2];
    second_half[2] += second_half[3];

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
pub(crate) fn dct1d_16_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 16]) {
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
    second_half[1] += second_half[2];
    second_half[2] += second_half[3];
    second_half[3] += second_half[4];
    second_half[4] += second_half[5];
    second_half[5] += second_half[6];
    second_half[6] += second_half[7];

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
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_16x16_avx2(token: archmage::X64V3Token, input: &[f32; 256], output: &mut [f32; 256]) {
    use magetypes::simd::f32x8;

    let scale = f32x8::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<256>();

    // --- Pass 1: Forward DCT on rows ---
    // Process rows 0-7 (first batch of 8)
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, input, 0, j);
        }
        dct1d_16_batch(token, &mut v);
        for j in 0..16 {
            v[j] *= scale;
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
            v[j] *= scale;
        }
        for j in 0..16 {
            scatter_col(v[j], &mut tmp, 8, j);
        }
    }

    // --- 16x16 scalar transpose ---
    let mut transposed = crate::scratch_buf::<256>();
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
            v[j] *= scale;
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
            v[j] *= scale;
        }
        for j in 0..16 {
            scatter_col(v[j], output, 8, j);
        }
    }
}

// ============================================================================
// 16x8 forward DCT (16 rows, 8 cols)
// ============================================================================

/// Compute scaled 16x8 forward DCT with SIMD acceleration.
///
/// Input: 128 f32 in row-major order (16 rows x 8 cols, stride 8).
/// Output: 128 f32 in 8x16 layout (stride 16) — no final transpose (ROWS >= COLS).
/// Dispatches to AVX2 when available; falls back to scalar otherwise.
#[inline]
pub fn dct_16x8(input: &[f32; 128], output: &mut [f32; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_16x8_avx2(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            dct_16x8_neon(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            dct_16x8_wasm128(token, input, output);
            return;
        }
    }

    dct_16x8_scalar(input, output);
}

#[inline]
pub fn dct_16x8_scalar(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut tmp = crate::scratch_buf::<128>();

    // Transform rows (8 columns each) with 8-point DCT
    for row in 0..16 {
        let s = row * 8;
        tmp[s..s + 8].copy_from_slice(&input[s..s + 8]);
        dct1d_8_scalar(&mut tmp[s..s + 8]);
        for i in 0..8 {
            tmp[s + i] *= 1.0 / 8.0;
        }
    }

    // Transpose 16x8 -> 8x16
    let mut transposed = crate::scratch_buf::<128>();
    for row in 0..16 {
        for col in 0..8 {
            transposed[col * 16 + row] = tmp[row * 8 + col];
        }
    }

    // Transform columns (now 16 elements each) with 16-point DCT
    for row in 0..8 {
        let s = row * 16;
        dct1d_16_scalar(&mut transposed[s..s + 16]);
        for i in 0..16 {
            transposed[s + i] *= 1.0 / 16.0;
        }
    }

    // No final transpose — ROWS >= COLS (same as dct_8x8)
    output.copy_from_slice(&transposed);
}

/// Load column `j` from 8 consecutive rows starting at `base_row` in `data` (stride 8).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
fn gather_col_s8(
    token: archmage::X64V3Token,
    data: &[f32],
    base_row: usize,
    j: usize,
) -> magetypes::simd::f32x8 {
    magetypes::simd::f32x8::from_array(
        token,
        [
            data[base_row * 8 + j],
            data[(base_row + 1) * 8 + j],
            data[(base_row + 2) * 8 + j],
            data[(base_row + 3) * 8 + j],
            data[(base_row + 4) * 8 + j],
            data[(base_row + 5) * 8 + j],
            data[(base_row + 6) * 8 + j],
            data[(base_row + 7) * 8 + j],
        ],
    )
}

/// Store f32x8 lanes back to column `j` of 8 consecutive rows starting at `base_row` (stride 8).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scatter_col_s8(v: magetypes::simd::f32x8, data: &mut [f32], base_row: usize, j: usize) {
    let mut lane = [0.0f32; 8];
    v.store(&mut lane);
    for r in 0..8 {
        data[(base_row + r) * 8 + j] = lane[r];
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_16x8_avx2(token: archmage::X64V3Token, input: &[f32; 128], output: &mut [f32; 128]) {
    use magetypes::simd::f32x8;

    let scale8 = f32x8::splat(token, 1.0 / 8.0);
    let scale16 = f32x8::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<128>();

    // --- Pass 1: 8-point forward DCT on each of 16 rows (stride 8) ---
    // Process 8 rows at a time. Each row has 8 elements.
    // Gather column j from 8 rows -> f32x8, apply dct1d_8_batch, scatter back.
    // Batch 1: rows 0-7
    {
        let mut v = [f32x8::zero(token); 8];
        for j in 0..8 {
            v[j] = gather_col_s8(token, input, 0, j);
        }
        dct1d_8_batch(token, &mut v);
        for j in 0..8 {
            v[j] *= scale8;
        }
        for j in 0..8 {
            scatter_col_s8(v[j], &mut tmp, 0, j);
        }
    }
    // Batch 2: rows 8-15
    {
        let mut v = [f32x8::zero(token); 8];
        for j in 0..8 {
            v[j] = gather_col_s8(token, input, 8, j);
        }
        dct1d_8_batch(token, &mut v);
        for j in 0..8 {
            v[j] *= scale8;
        }
        for j in 0..8 {
            scatter_col_s8(v[j], &mut tmp, 8, j);
        }
    }

    // --- Transpose 16x8 -> 8x16 (scalar) ---
    let mut transposed = crate::scratch_buf::<128>();
    for row in 0..16 {
        for col in 0..8 {
            transposed[col * 16 + row] = tmp[row * 8 + col];
        }
    }

    // --- Pass 2: 16-point forward DCT on each of 8 rows (stride 16) ---
    // Each row has 16 elements. Gather column j from 8 rows (all 8 rows fit in one batch).
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, &transposed, 0, j);
        }
        dct1d_16_batch(token, &mut v);
        for j in 0..16 {
            v[j] *= scale16;
        }
        for j in 0..16 {
            scatter_col(v[j], output, 0, j);
        }
    }

    // No final transpose (ROWS >= COLS)
}

// ============================================================================
// 8x16 forward DCT (8 rows, 16 cols)
// ============================================================================

/// Compute scaled 8x16 forward DCT with SIMD acceleration.
///
/// Input: 128 f32 in row-major order (8 rows x 16 cols, stride 16).
/// Output: 128 f32 in 8x16 layout (stride 16) — includes final transpose (ROWS < COLS).
/// Dispatches to AVX2 when available; falls back to scalar otherwise.
#[inline]
pub fn dct_8x16(input: &[f32; 128], output: &mut [f32; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_8x16_avx2(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            dct_8x16_neon(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            dct_8x16_wasm128(token, input, output);
            return;
        }
    }

    dct_8x16_scalar(input, output);
}

#[inline]
pub fn dct_8x16_scalar(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut tmp = crate::scratch_buf::<128>();

    // Transform rows (16 columns each) with 16-point DCT
    for row in 0..8 {
        let s = row * 16;
        tmp[s..s + 16].copy_from_slice(&input[s..s + 16]);
        dct1d_16_scalar(&mut tmp[s..s + 16]);
        for i in 0..16 {
            tmp[s + i] *= 1.0 / 16.0;
        }
    }

    // Transpose 8x16 -> 16x8
    let mut transposed = crate::scratch_buf::<128>();
    for row in 0..8 {
        for col in 0..16 {
            transposed[col * 8 + row] = tmp[row * 16 + col];
        }
    }

    // Transform columns (now 8 elements each) with 8-point DCT
    for row in 0..16 {
        let s = row * 8;
        dct1d_8_scalar(&mut transposed[s..s + 8]);
        for i in 0..8 {
            transposed[s + i] *= 1.0 / 8.0;
        }
    }

    // Final transpose 16x8 -> 8x16 (ROWS < COLS branch in libjxl)
    for row in 0..16 {
        for col in 0..8 {
            output[col * 16 + row] = transposed[row * 8 + col];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_8x16_avx2(token: archmage::X64V3Token, input: &[f32; 128], output: &mut [f32; 128]) {
    use magetypes::simd::f32x8;

    let scale8 = f32x8::splat(token, 1.0 / 8.0);
    let scale16 = f32x8::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<128>();

    // --- Pass 1: 16-point forward DCT on each of 8 rows (stride 16) ---
    // All 8 rows fit in one batch. Gather column j from 8 rows.
    {
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, input, 0, j);
        }
        dct1d_16_batch(token, &mut v);
        for j in 0..16 {
            v[j] *= scale16;
        }
        for j in 0..16 {
            scatter_col(v[j], &mut tmp, 0, j);
        }
    }

    // --- Transpose 8x16 -> 16x8 (scalar) ---
    let mut transposed = crate::scratch_buf::<128>();
    for row in 0..8 {
        for col in 0..16 {
            transposed[col * 8 + row] = tmp[row * 16 + col];
        }
    }

    // --- Pass 2: 8-point forward DCT on each of 16 rows (stride 8) ---
    // Process 8 rows at a time (2 batches for 16 rows).
    // Batch 1: rows 0-7
    {
        let mut v = [f32x8::zero(token); 8];
        for j in 0..8 {
            v[j] = gather_col_s8(token, &transposed, 0, j);
        }
        dct1d_8_batch(token, &mut v);
        for j in 0..8 {
            v[j] *= scale8;
        }
        for j in 0..8 {
            scatter_col_s8(v[j], &mut transposed, 0, j);
        }
    }
    // Batch 2: rows 8-15
    {
        let mut v = [f32x8::zero(token); 8];
        for j in 0..8 {
            v[j] = gather_col_s8(token, &transposed, 8, j);
        }
        dct1d_8_batch(token, &mut v);
        for j in 0..8 {
            v[j] *= scale8;
        }
        for j in 0..8 {
            scatter_col_s8(v[j], &mut transposed, 8, j);
        }
    }

    // --- Final transpose 16x8 -> 8x16 (ROWS < COLS) ---
    for row in 0..16 {
        for col in 0..8 {
            output[col * 16 + row] = transposed[row * 8 + col];
        }
    }
}

// ============================================================================
// aarch64 NEON implementations
// ============================================================================

/// Gather column `j` from 4 consecutive rows starting at `base_row` (stride `s`).
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn gather_col_neon(
    token: archmage::NeonToken,
    data: &[f32],
    base_row: usize,
    j: usize,
    s: usize,
) -> magetypes::simd::f32x4 {
    magetypes::simd::f32x4::from_array(
        token,
        [
            data[base_row * s + j],
            data[(base_row + 1) * s + j],
            data[(base_row + 2) * s + j],
            data[(base_row + 3) * s + j],
        ],
    )
}

/// Scatter f32x4 lanes back to column `j` of 4 consecutive rows (stride `s`).
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn scatter_col_neon(
    _token: archmage::NeonToken,
    v: magetypes::simd::f32x4,
    data: &mut [f32],
    base_row: usize,
    j: usize,
    s: usize,
) {
    let mut lane = [0.0f32; 4];
    v.store(&mut lane);
    for r in 0..4 {
        data[(base_row + r) * s + j] = lane[r];
    }
}

/// NEON batched 4-point forward DCT on f32x4 arrays.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn dct1d_4_batch_neon(token: archmage::NeonToken, v: &mut [magetypes::simd::f32x4; 4]) {
    use magetypes::simd::f32x4;

    let sqrt2 = f32x4::splat(token, SQRT2);
    let wc4_0 = f32x4::splat(token, WC_MULTIPLIERS_4[0]);
    let wc4_1 = f32x4::splat(token, WC_MULTIPLIERS_4[1]);

    let a0 = v[0] + v[3];
    let a1 = v[1] + v[2];
    let s0 = v[0] - v[3];
    let s1 = v[1] - v[2];

    let fh0 = a0 + a1;
    let fh1 = a0 - a1;

    let s0 = s0 * wc4_0;
    let s1 = s1 * wc4_1;
    let sh0 = s0 + s1;
    let sh1 = s0 - s1;
    let sh0 = sqrt2.mul_add(sh0, sh1);

    v[0] = fh0;
    v[1] = sh0;
    v[2] = fh1;
    v[3] = sh1;
}

/// NEON batched 8-point forward DCT on f32x4 arrays.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn dct1d_8_batch_neon(token: archmage::NeonToken, v: &mut [magetypes::simd::f32x4; 8]) {
    use magetypes::simd::f32x4;

    let sqrt2 = f32x4::splat(token, SQRT2);

    let a0 = v[0] + v[7];
    let a1 = v[1] + v[6];
    let a2 = v[2] + v[5];
    let a3 = v[3] + v[4];
    let s0 = v[0] - v[7];
    let s1 = v[1] - v[6];
    let s2 = v[2] - v[5];
    let s3 = v[3] - v[4];

    let mut first_half = [a0, a1, a2, a3];
    dct1d_4_batch_neon(token, &mut first_half);

    let s0 = s0 * f32x4::splat(token, WC_MULTIPLIERS_8[0]);
    let s1 = s1 * f32x4::splat(token, WC_MULTIPLIERS_8[1]);
    let s2 = s2 * f32x4::splat(token, WC_MULTIPLIERS_8[2]);
    let s3 = s3 * f32x4::splat(token, WC_MULTIPLIERS_8[3]);
    let mut second_half = [s0, s1, s2, s3];
    dct1d_4_batch_neon(token, &mut second_half);

    second_half[0] = sqrt2.mul_add(second_half[0], second_half[1]);
    second_half[1] += second_half[2];
    second_half[2] += second_half[3];

    v[0] = first_half[0];
    v[1] = second_half[0];
    v[2] = first_half[1];
    v[3] = second_half[1];
    v[4] = first_half[2];
    v[5] = second_half[2];
    v[6] = first_half[3];
    v[7] = second_half[3];
}

/// NEON batched 16-point forward DCT on f32x4 arrays.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
pub(crate) fn dct1d_16_batch_neon(
    token: archmage::NeonToken,
    v: &mut [magetypes::simd::f32x4; 16],
) {
    use magetypes::simd::f32x4;

    let sqrt2 = f32x4::splat(token, SQRT2);

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

    let mut first_half = [a0, a1, a2, a3, a4, a5, a6, a7];
    dct1d_8_batch_neon(token, &mut first_half);

    let s0 = s0 * f32x4::splat(token, WC_MULTIPLIERS_16[0]);
    let s1 = s1 * f32x4::splat(token, WC_MULTIPLIERS_16[1]);
    let s2 = s2 * f32x4::splat(token, WC_MULTIPLIERS_16[2]);
    let s3 = s3 * f32x4::splat(token, WC_MULTIPLIERS_16[3]);
    let s4 = s4 * f32x4::splat(token, WC_MULTIPLIERS_16[4]);
    let s5 = s5 * f32x4::splat(token, WC_MULTIPLIERS_16[5]);
    let s6 = s6 * f32x4::splat(token, WC_MULTIPLIERS_16[6]);
    let s7 = s7 * f32x4::splat(token, WC_MULTIPLIERS_16[7]);
    let mut second_half = [s0, s1, s2, s3, s4, s5, s6, s7];
    dct1d_8_batch_neon(token, &mut second_half);

    second_half[0] = sqrt2.mul_add(second_half[0], second_half[1]);
    second_half[1] += second_half[2];
    second_half[2] += second_half[3];
    second_half[3] += second_half[4];
    second_half[4] += second_half[5];
    second_half[5] += second_half[6];
    second_half[6] += second_half[7];

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

/// Process a batch of 4 rows through gather → 8-point DCT → scale → scatter.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
#[allow(clippy::needless_range_loop)]
fn neon_dct8_batch(
    token: archmage::NeonToken,
    data_in: &[f32],
    data_out: &mut [f32],
    base_row: usize,
    stride: usize,
    scale: magetypes::simd::f32x4,
) {
    let mut v = [magetypes::simd::f32x4::zero(token); 8];
    for j in 0..8 {
        v[j] = gather_col_neon(token, data_in, base_row, j, stride);
    }
    dct1d_8_batch_neon(token, &mut v);
    for j in 0..8 {
        v[j] *= scale;
    }
    for j in 0..8 {
        scatter_col_neon(token, v[j], data_out, base_row, j, stride);
    }
}

/// Process a batch of 4 rows through gather → 16-point DCT → scale → scatter.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
#[allow(clippy::needless_range_loop)]
fn neon_dct16_batch(
    token: archmage::NeonToken,
    data_in: &[f32],
    data_out: &mut [f32],
    base_row: usize,
    stride: usize,
    scale: magetypes::simd::f32x4,
) {
    let mut v = [magetypes::simd::f32x4::zero(token); 16];
    for j in 0..16 {
        v[j] = gather_col_neon(token, data_in, base_row, j, stride);
    }
    dct1d_16_batch_neon(token, &mut v);
    for j in 0..16 {
        v[j] *= scale;
    }
    for j in 0..16 {
        scatter_col_neon(token, v[j], data_out, base_row, j, stride);
    }
}

/// NEON 16x16 forward DCT: process 4 rows at a time.
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_16x16_neon(token: archmage::NeonToken, input: &[f32; 256], output: &mut [f32; 256]) {
    use magetypes::simd::f32x4;
    let scale = f32x4::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<256>();

    // Pass 1: Forward DCT on rows (4 batches of 4 rows)
    for batch in 0..4 {
        neon_dct16_batch(token, input, &mut tmp, batch * 4, 16, scale);
    }

    // Transpose 16x16
    let mut transposed = crate::scratch_buf::<256>();
    for r in 0..16 {
        for c in 0..16 {
            transposed[c * 16 + r] = tmp[r * 16 + c];
        }
    }

    // Pass 2: Forward DCT on columns (4 batches of 4 rows)
    for batch in 0..4 {
        neon_dct16_batch(token, &transposed, output, batch * 4, 16, scale);
    }
}

/// NEON 16x8 forward DCT.
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_16x8_neon(token: archmage::NeonToken, input: &[f32; 128], output: &mut [f32; 128]) {
    use magetypes::simd::f32x4;
    let scale8 = f32x4::splat(token, 1.0 / 8.0);
    let scale16 = f32x4::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<128>();

    // Pass 1: 8-point DCT on 16 rows (stride 8), 4 batches of 4 rows
    for batch in 0..4 {
        neon_dct8_batch(token, input, &mut tmp, batch * 4, 8, scale8);
    }

    // Transpose 16x8 -> 8x16
    let mut transposed = crate::scratch_buf::<128>();
    for row in 0..16 {
        for col in 0..8 {
            transposed[col * 16 + row] = tmp[row * 8 + col];
        }
    }

    // Pass 2: 16-point DCT on 8 rows (stride 16), 2 batches of 4 rows
    for batch in 0..2 {
        neon_dct16_batch(token, &transposed, output, batch * 4, 16, scale16);
    }
}

/// NEON 8x16 forward DCT.
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_8x16_neon(token: archmage::NeonToken, input: &[f32; 128], output: &mut [f32; 128]) {
    use magetypes::simd::f32x4;
    let scale8 = f32x4::splat(token, 1.0 / 8.0);
    let scale16 = f32x4::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<128>();

    // Pass 1: 16-point DCT on 8 rows (stride 16), 2 batches of 4 rows
    for batch in 0..2 {
        neon_dct16_batch(token, input, &mut tmp, batch * 4, 16, scale16);
    }

    // Transpose 8x16 -> 16x8
    let mut transposed = crate::scratch_buf::<128>();
    for row in 0..8 {
        for col in 0..16 {
            transposed[col * 8 + row] = tmp[row * 16 + col];
        }
    }

    // Pass 2: 8-point DCT on 16 rows (stride 8), 4 batches of 4 rows
    let mut pass2_out = crate::scratch_buf::<128>();
    for batch in 0..4 {
        neon_dct8_batch(token, &transposed, &mut pass2_out, batch * 4, 8, scale8);
    }

    // Final transpose 16x8 -> 8x16 (ROWS < COLS)
    for row in 0..16 {
        for col in 0..8 {
            output[col * 16 + row] = pass2_out[row * 8 + col];
        }
    }
}

// ============================================================================
// wasm32 SIMD128 implementations
// ============================================================================

/// Gather column `j` from 4 consecutive rows starting at `base_row` (stride `s`).
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn gather_col_wasm128(
    token: archmage::Wasm128Token,
    data: &[f32],
    base_row: usize,
    j: usize,
    s: usize,
) -> magetypes::simd::f32x4 {
    magetypes::simd::f32x4::from_array(
        token,
        [
            data[base_row * s + j],
            data[(base_row + 1) * s + j],
            data[(base_row + 2) * s + j],
            data[(base_row + 3) * s + j],
        ],
    )
}

/// Scatter f32x4 lanes back to column `j` of 4 consecutive rows (stride `s`).
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn scatter_col_wasm128(
    _token: archmage::Wasm128Token,
    v: magetypes::simd::f32x4,
    data: &mut [f32],
    base_row: usize,
    j: usize,
    s: usize,
) {
    let mut lane = [0.0f32; 4];
    v.store(&mut lane);
    for r in 0..4 {
        data[(base_row + r) * s + j] = lane[r];
    }
}

/// WASM128 batched 4-point forward DCT on f32x4 arrays.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn dct1d_4_batch_wasm128(token: archmage::Wasm128Token, v: &mut [magetypes::simd::f32x4; 4]) {
    use magetypes::simd::f32x4;

    let sqrt2 = f32x4::splat(token, SQRT2);
    let wc4_0 = f32x4::splat(token, WC_MULTIPLIERS_4[0]);
    let wc4_1 = f32x4::splat(token, WC_MULTIPLIERS_4[1]);

    let a0 = v[0] + v[3];
    let a1 = v[1] + v[2];
    let s0 = v[0] - v[3];
    let s1 = v[1] - v[2];

    let fh0 = a0 + a1;
    let fh1 = a0 - a1;

    let s0 = s0 * wc4_0;
    let s1 = s1 * wc4_1;
    let sh0 = s0 + s1;
    let sh1 = s0 - s1;
    let sh0 = sqrt2.mul_add(sh0, sh1);

    v[0] = fh0;
    v[1] = sh0;
    v[2] = fh1;
    v[3] = sh1;
}

/// WASM128 batched 8-point forward DCT on f32x4 arrays.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn dct1d_8_batch_wasm128(token: archmage::Wasm128Token, v: &mut [magetypes::simd::f32x4; 8]) {
    use magetypes::simd::f32x4;

    let sqrt2 = f32x4::splat(token, SQRT2);

    let a0 = v[0] + v[7];
    let a1 = v[1] + v[6];
    let a2 = v[2] + v[5];
    let a3 = v[3] + v[4];
    let s0 = v[0] - v[7];
    let s1 = v[1] - v[6];
    let s2 = v[2] - v[5];
    let s3 = v[3] - v[4];

    let mut first_half = [a0, a1, a2, a3];
    dct1d_4_batch_wasm128(token, &mut first_half);

    let s0 = s0 * f32x4::splat(token, WC_MULTIPLIERS_8[0]);
    let s1 = s1 * f32x4::splat(token, WC_MULTIPLIERS_8[1]);
    let s2 = s2 * f32x4::splat(token, WC_MULTIPLIERS_8[2]);
    let s3 = s3 * f32x4::splat(token, WC_MULTIPLIERS_8[3]);
    let mut second_half = [s0, s1, s2, s3];
    dct1d_4_batch_wasm128(token, &mut second_half);

    second_half[0] = sqrt2.mul_add(second_half[0], second_half[1]);
    second_half[1] += second_half[2];
    second_half[2] += second_half[3];

    v[0] = first_half[0];
    v[1] = second_half[0];
    v[2] = first_half[1];
    v[3] = second_half[1];
    v[4] = first_half[2];
    v[5] = second_half[2];
    v[6] = first_half[3];
    v[7] = second_half[3];
}

/// WASM128 batched 16-point forward DCT on f32x4 arrays.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
pub(crate) fn dct1d_16_batch_wasm128(
    token: archmage::Wasm128Token,
    v: &mut [magetypes::simd::f32x4; 16],
) {
    use magetypes::simd::f32x4;

    let sqrt2 = f32x4::splat(token, SQRT2);

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

    let mut first_half = [a0, a1, a2, a3, a4, a5, a6, a7];
    dct1d_8_batch_wasm128(token, &mut first_half);

    let s0 = s0 * f32x4::splat(token, WC_MULTIPLIERS_16[0]);
    let s1 = s1 * f32x4::splat(token, WC_MULTIPLIERS_16[1]);
    let s2 = s2 * f32x4::splat(token, WC_MULTIPLIERS_16[2]);
    let s3 = s3 * f32x4::splat(token, WC_MULTIPLIERS_16[3]);
    let s4 = s4 * f32x4::splat(token, WC_MULTIPLIERS_16[4]);
    let s5 = s5 * f32x4::splat(token, WC_MULTIPLIERS_16[5]);
    let s6 = s6 * f32x4::splat(token, WC_MULTIPLIERS_16[6]);
    let s7 = s7 * f32x4::splat(token, WC_MULTIPLIERS_16[7]);
    let mut second_half = [s0, s1, s2, s3, s4, s5, s6, s7];
    dct1d_8_batch_wasm128(token, &mut second_half);

    second_half[0] = sqrt2.mul_add(second_half[0], second_half[1]);
    second_half[1] += second_half[2];
    second_half[2] += second_half[3];
    second_half[3] += second_half[4];
    second_half[4] += second_half[5];
    second_half[5] += second_half[6];
    second_half[6] += second_half[7];

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

/// Process a batch of 4 rows through gather -> 8-point DCT -> scale -> scatter.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
#[allow(clippy::needless_range_loop)]
fn wasm128_dct8_batch(
    token: archmage::Wasm128Token,
    data_in: &[f32],
    data_out: &mut [f32],
    base_row: usize,
    stride: usize,
    scale: magetypes::simd::f32x4,
) {
    let mut v = [magetypes::simd::f32x4::zero(token); 8];
    for j in 0..8 {
        v[j] = gather_col_wasm128(token, data_in, base_row, j, stride);
    }
    dct1d_8_batch_wasm128(token, &mut v);
    for j in 0..8 {
        v[j] *= scale;
    }
    for j in 0..8 {
        scatter_col_wasm128(token, v[j], data_out, base_row, j, stride);
    }
}

/// Process a batch of 4 rows through gather -> 16-point DCT -> scale -> scatter.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
#[allow(clippy::needless_range_loop)]
fn wasm128_dct16_batch(
    token: archmage::Wasm128Token,
    data_in: &[f32],
    data_out: &mut [f32],
    base_row: usize,
    stride: usize,
    scale: magetypes::simd::f32x4,
) {
    let mut v = [magetypes::simd::f32x4::zero(token); 16];
    for j in 0..16 {
        v[j] = gather_col_wasm128(token, data_in, base_row, j, stride);
    }
    dct1d_16_batch_wasm128(token, &mut v);
    for j in 0..16 {
        v[j] *= scale;
    }
    for j in 0..16 {
        scatter_col_wasm128(token, v[j], data_out, base_row, j, stride);
    }
}

/// WASM128 16x16 forward DCT: process 4 rows at a time.
#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_16x16_wasm128(
    token: archmage::Wasm128Token,
    input: &[f32; 256],
    output: &mut [f32; 256],
) {
    use magetypes::simd::f32x4;
    let scale = f32x4::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<256>();

    // Pass 1: Forward DCT on rows (4 batches of 4 rows)
    for batch in 0..4 {
        wasm128_dct16_batch(token, input, &mut tmp, batch * 4, 16, scale);
    }

    // Transpose 16x16
    let mut transposed = crate::scratch_buf::<256>();
    for r in 0..16 {
        for c in 0..16 {
            transposed[c * 16 + r] = tmp[r * 16 + c];
        }
    }

    // Pass 2: Forward DCT on columns (4 batches of 4 rows)
    for batch in 0..4 {
        wasm128_dct16_batch(token, &transposed, output, batch * 4, 16, scale);
    }
}

/// WASM128 16x8 forward DCT.
#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_16x8_wasm128(
    token: archmage::Wasm128Token,
    input: &[f32; 128],
    output: &mut [f32; 128],
) {
    use magetypes::simd::f32x4;
    let scale8 = f32x4::splat(token, 1.0 / 8.0);
    let scale16 = f32x4::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<128>();

    // Pass 1: 8-point DCT on 16 rows (stride 8), 4 batches of 4 rows
    for batch in 0..4 {
        wasm128_dct8_batch(token, input, &mut tmp, batch * 4, 8, scale8);
    }

    // Transpose 16x8 -> 8x16
    let mut transposed = crate::scratch_buf::<128>();
    for row in 0..16 {
        for col in 0..8 {
            transposed[col * 16 + row] = tmp[row * 8 + col];
        }
    }

    // Pass 2: 16-point DCT on 8 rows (stride 16), 2 batches of 4 rows
    for batch in 0..2 {
        wasm128_dct16_batch(token, &transposed, output, batch * 4, 16, scale16);
    }
}

/// WASM128 8x16 forward DCT.
#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_8x16_wasm128(
    token: archmage::Wasm128Token,
    input: &[f32; 128],
    output: &mut [f32; 128],
) {
    use magetypes::simd::f32x4;
    let scale8 = f32x4::splat(token, 1.0 / 8.0);
    let scale16 = f32x4::splat(token, 1.0 / 16.0);
    let mut tmp = crate::scratch_buf::<128>();

    // Pass 1: 16-point DCT on 8 rows (stride 16), 2 batches of 4 rows
    for batch in 0..2 {
        wasm128_dct16_batch(token, input, &mut tmp, batch * 4, 16, scale16);
    }

    // Transpose 8x16 -> 16x8
    let mut transposed = crate::scratch_buf::<128>();
    for row in 0..8 {
        for col in 0..16 {
            transposed[col * 8 + row] = tmp[row * 16 + col];
        }
    }

    // Pass 2: 8-point DCT on 16 rows (stride 8), 4 batches of 4 rows
    let mut pass2_out = crate::scratch_buf::<128>();
    for batch in 0..4 {
        wasm128_dct8_batch(token, &transposed, &mut pass2_out, batch * 4, 8, scale8);
    }

    // Final transpose 16x8 -> 8x16 (ROWS < COLS)
    for row in 0..16 {
        for col in 0..8 {
            output[col * 16 + row] = pass2_out[row * 8 + col];
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn test_dct_16x16_simd_matches_scalar() {
        let mut input = [0.0f32; 256];
        for (i, val) in input.iter_mut().enumerate() {
            *val = i as f32;
        }
        let mut scalar_out = [0.0f32; 256];
        dct_16x16_scalar(&input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 256];
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
                    "DCT16x16 max diff = {max_diff} at {max_idx} (scalar={}, simd={}) [{perm}]",
                    scalar_out[max_idx],
                    simd_out[max_idx],
                );
            },
        );
        std::eprintln!("{report}");
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
        for (i, &coeff) in output.iter().enumerate().skip(1) {
            let val = coeff.abs();
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

    #[test]
    fn test_dct_16x8_simd_matches_scalar() {
        let mut input = [0.0f32; 128];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.43 + 2.1).cos() * 80.0;
        }
        let mut scalar_out = [0.0f32; 128];
        dct_16x8_scalar(&input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 128];
                dct_16x8(&input, &mut simd_out);
                let mut max_diff = 0.0f32;
                let mut max_idx = 0;
                for i in 0..128 {
                    let diff = (scalar_out[i] - simd_out[i]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = i;
                    }
                }
                assert!(
                    max_diff < 1e-2,
                    "DCT16x8 max diff = {max_diff} at {max_idx} (scalar={}, simd={}) [{perm}]",
                    scalar_out[max_idx],
                    simd_out[max_idx],
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_dct_8x16_simd_matches_scalar() {
        let mut input = [0.0f32; 128];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.29 + 0.7).sin() * 120.0;
        }
        let mut scalar_out = [0.0f32; 128];
        dct_8x16_scalar(&input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 128];
                dct_8x16(&input, &mut simd_out);
                let mut max_diff = 0.0f32;
                let mut max_idx = 0;
                for i in 0..128 {
                    let diff = (scalar_out[i] - simd_out[i]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = i;
                    }
                }
                assert!(
                    max_diff < 1e-2,
                    "DCT8x16 max diff = {max_diff} at {max_idx} (scalar={}, simd={}) [{perm}]",
                    scalar_out[max_idx],
                    simd_out[max_idx],
                );
            },
        );
        std::eprintln!("{report}");
    }
}
