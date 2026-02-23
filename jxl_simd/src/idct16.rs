// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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
#[inline]
pub fn idct_16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_16x16_avx2(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            idct_16x16_neon(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            idct_16x16_wasm128(token, input, output);
            return;
        }
    }

    idct_16x16_scalar(input, output);
}

// ============================================================================
// Scalar fallback — matches jxl_encoder/src/vardct/dct/inverse.rs exactly
// ============================================================================

#[inline]
pub fn idct_16x16_scalar(input: &[f32; 256], output: &mut [f32; 256]) {
    let mut tmp = crate::scratch_buf::<256>();

    // IDCT on each row
    for row in 0..16 {
        let s = row * 16;
        tmp[s..s + 16].copy_from_slice(&input[s..s + 16]);
        idct1d_16_scalar(&mut tmp[s..s + 16]);
    }

    // Transpose 16x16
    let mut transposed = crate::scratch_buf::<256>();
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
pub(crate) fn idct1d_16_core_batch(
    token: archmage::X64V3Token,
    v: &mut [magetypes::simd::f32x8; 16],
) {
    use magetypes::simd::f32x8;

    let half = f32x8::splat(token, 0.5);
    let inv_sqrt2 = f32x8::splat(token, 1.0 / SQRT2);

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

/// AVX2 batched 16-point IDCT with *= 16 scaling.
///
/// `v` holds [v0..v15] representing positions 0-15 across 8 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
pub(crate) fn idct1d_16_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 16]) {
    use magetypes::simd::f32x8;

    let scale16 = f32x8::splat(token, 16.0);
    for vi in v.iter_mut() {
        *vi *= scale16;
    }
    idct1d_16_core_batch(token, v);
}

/// AVX2 16x16 IDCT: process 8 rows at a time via batched 16-point IDCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_16x16_avx2(token: archmage::X64V3Token, input: &[f32; 256], output: &mut [f32; 256]) {
    use magetypes::simd::f32x8;

    let mut tmp = crate::scratch_buf::<256>();

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
    let mut transposed = crate::scratch_buf::<256>();
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

// ============================================================================
// 16x8 inverse DCT (16 rows, 8 cols)
// ============================================================================

/// Compute 16x8 inverse DCT with SIMD acceleration.
///
/// Input: 128 f32 in 8x16 layout (stride 16, coefficient domain — output of dct_16x8).
/// Output: 128 f32 in 16x8 row-major order (stride 8, spatial domain).
/// Dispatches to AVX2 when available; falls back to scalar otherwise.
#[inline]
pub fn idct_16x8(input: &[f32; 128], output: &mut [f32; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_16x8_avx2(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            idct_16x8_neon(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            idct_16x8_wasm128(token, input, output);
            return;
        }
    }

    idct_16x8_scalar(input, output);
}

#[inline]
pub fn idct_16x8_scalar(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut tmp = crate::scratch_buf::<128>();

    // Apply 8-point IDCT (with x8 scaling) to each of 16 rows (stride 8)
    for row in 0..16 {
        let s = row * 8;
        tmp[s..s + 8].copy_from_slice(&input[s..s + 8]);
        idct1d_8_scalar(&mut tmp[s..s + 8]);
    }

    // Apply 16-point IDCT (with x16 scaling) to each of 8 columns
    for col in 0..8 {
        let mut col_buf = [0.0f32; 16];
        for row in 0..16 {
            col_buf[row] = tmp[row * 8 + col];
        }
        idct1d_16_scalar(&mut col_buf);
        for row in 0..16 {
            output[row * 8 + col] = col_buf[row];
        }
    }
}

/// 8-point IDCT with *= 8 scaling (scalar).
fn idct1d_8_scalar(mem: &mut [f32]) {
    for x in mem.iter_mut().take(8) {
        *x *= 8.0;
    }
    idct1d_8_core_scalar(mem);
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

/// AVX2 batched 8-point IDCT with *= 8 scaling.
///
/// `v` holds [v0..v7] representing positions 0-7 across 8 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
fn idct1d_8_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 8]) {
    use magetypes::simd::f32x8;

    let scale8 = f32x8::splat(token, 8.0);
    for vi in v.iter_mut() {
        *vi *= scale8;
    }
    idct1d_8_core_batch(token, v);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_16x8_avx2(token: archmage::X64V3Token, input: &[f32; 128], output: &mut [f32; 128]) {
    use magetypes::simd::f32x8;

    let mut tmp = crate::scratch_buf::<128>();

    // --- Pass 1: 8-point IDCT (with x8 scaling) on each of 16 rows (stride 8) ---
    // Process 8 rows at a time. Gather column j from 8 rows -> f32x8.
    // Batch 1: rows 0-7
    {
        let mut v = [f32x8::zero(token); 8];
        for j in 0..8 {
            v[j] = gather_col_s8(token, input, 0, j);
        }
        idct1d_8_batch(token, &mut v);
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
        idct1d_8_batch(token, &mut v);
        for j in 0..8 {
            scatter_col_s8(v[j], &mut tmp, 8, j);
        }
    }

    // --- Pass 2: 16-point IDCT (with x16 scaling) on each of 8 columns ---
    // Load row i as contiguous f32x8 from tmp[i*8..i*8+8]. Each lane k = column k's i-th element.
    {
        let mut v = [f32x8::zero(token); 16];
        for i in 0..16 {
            v[i] = f32x8::from_slice(token, &tmp[i * 8..i * 8 + 8]);
        }
        idct1d_16_batch(token, &mut v);
        for i in 0..16 {
            v[i].store((&mut output[i * 8..i * 8 + 8]).try_into().unwrap());
        }
    }
}

// ============================================================================
// 8x16 inverse DCT (8 rows, 16 cols)
// ============================================================================

/// Compute 8x16 inverse DCT with SIMD acceleration.
///
/// Input: 128 f32 in 8x16 layout (stride 16, coefficient domain — output of dct_8x16).
/// Output: 128 f32 in 8x16 row-major order (stride 16, spatial domain).
/// Dispatches to AVX2 when available; falls back to scalar otherwise.
#[inline]
pub fn idct_8x16(input: &[f32; 128], output: &mut [f32; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_8x16_avx2(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            idct_8x16_neon(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            idct_8x16_wasm128(token, input, output);
            return;
        }
    }

    idct_8x16_scalar(input, output);
}

#[inline]
pub fn idct_8x16_scalar(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut tmp = crate::scratch_buf::<128>();

    // Apply 16-point IDCT (with x16 scaling) to each of 8 rows (stride 16)
    for row in 0..8 {
        let s = row * 16;
        tmp[s..s + 16].copy_from_slice(&input[s..s + 16]);
        idct1d_16_scalar(&mut tmp[s..s + 16]);
    }

    // Apply 8-point IDCT (with x8 scaling) to each of 16 columns
    for col in 0..16 {
        let mut col_buf = [0.0f32; 8];
        for row in 0..8 {
            col_buf[row] = tmp[row * 16 + col];
        }
        idct1d_8_scalar(&mut col_buf);
        for row in 0..8 {
            output[row * 16 + col] = col_buf[row];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_8x16_avx2(token: archmage::X64V3Token, input: &[f32; 128], output: &mut [f32; 128]) {
    use magetypes::simd::f32x8;

    let mut tmp = crate::scratch_buf::<128>();

    // --- Pass 1: 16-point IDCT (with x16 scaling) on each of 8 rows (stride 16) ---
    // All 8 rows fit in one batch. Gather column j from 8 rows at stride 16.
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

    // --- Pass 2: 8-point IDCT (with x8 scaling) on each of 16 columns ---
    // Process 8 columns at a time via contiguous loads (2 batches for 16 columns).
    // Batch 1: columns 0-7
    {
        let mut v = [f32x8::zero(token); 8];
        for i in 0..8 {
            v[i] = f32x8::from_slice(token, &tmp[i * 16..i * 16 + 8]);
        }
        idct1d_8_batch(token, &mut v);
        for i in 0..8 {
            v[i].store((&mut output[i * 16..i * 16 + 8]).try_into().unwrap());
        }
    }
    // Batch 2: columns 8-15
    {
        let mut v = [f32x8::zero(token); 8];
        for i in 0..8 {
            v[i] = f32x8::from_slice(token, &tmp[i * 16 + 8..i * 16 + 16]);
        }
        idct1d_8_batch(token, &mut v);
        for i in 0..8 {
            v[i].store((&mut output[i * 16 + 8..i * 16 + 16]).try_into().unwrap());
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

/// NEON batched 4-point inverse DCT on f32x4 arrays.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn idct1d_4_batch_neon(token: archmage::NeonToken, v: &mut [magetypes::simd::f32x4; 4]) {
    use magetypes::simd::f32x4;

    let half = f32x4::splat(token, 0.5);
    let inv_sqrt2 = f32x4::splat(token, 1.0 / SQRT2);
    let inv_wc4_0 = f32x4::splat(token, INV_WC4[0]);
    let inv_wc4_1 = f32x4::splat(token, INV_WC4[1]);

    // De-interleave: [v0, v1, v2, v3] -> [v0, v2, v1, v3]
    let t0 = v[0];
    let t1 = v[2];
    let t2 = v[1];
    let t3 = v[3];

    // Reverse B transform: t2 = (t2 - t3) / sqrt2
    let t2 = (t2 - t3) * inv_sqrt2;

    // IDCT-2 on second half
    let s2 = (t2 + t3) * half;
    let s3 = (t2 - t3) * half;

    // Divide by WcMultipliers_4
    let s2 = s2 * inv_wc4_0;
    let s3 = s3 * inv_wc4_1;

    // IDCT-2 on first half
    let s0 = (t0 + t1) * half;
    let s1 = (t0 - t1) * half;

    // Combine
    v[0] = (s0 + s2) * half;
    v[3] = (s0 - s2) * half;
    v[1] = (s1 + s3) * half;
    v[2] = (s1 - s3) * half;
}

/// NEON batched 8-point IDCT core (no N scaling) on f32x4 arrays.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn idct1d_8_core_batch_neon(token: archmage::NeonToken, v: &mut [magetypes::simd::f32x4; 8]) {
    use magetypes::simd::f32x4;

    let half = f32x4::splat(token, 0.5);
    let inv_sqrt2 = f32x4::splat(token, 1.0 / SQRT2);

    // De-interleave: even -> first_half, odd -> second_half
    let mut first_half = [v[0], v[2], v[4], v[6]];
    let mut second_half = [v[1], v[3], v[5], v[7]];

    // Reverse B transform on second half
    second_half[2] -= second_half[3];
    second_half[1] -= second_half[2];
    second_half[0] = (second_half[0] - second_half[1]) * inv_sqrt2;

    // IDCT-4 on second half
    idct1d_4_batch_neon(token, &mut second_half);

    // Divide by WcMultipliers_8
    second_half[0] *= f32x4::splat(token, INV_WC8[0]);
    second_half[1] *= f32x4::splat(token, INV_WC8[1]);
    second_half[2] *= f32x4::splat(token, INV_WC8[2]);
    second_half[3] *= f32x4::splat(token, INV_WC8[3]);

    // IDCT-4 on first half
    idct1d_4_batch_neon(token, &mut first_half);

    // Combine
    v[0] = (first_half[0] + second_half[0]) * half;
    v[7] = (first_half[0] - second_half[0]) * half;
    v[1] = (first_half[1] + second_half[1]) * half;
    v[6] = (first_half[1] - second_half[1]) * half;
    v[2] = (first_half[2] + second_half[2]) * half;
    v[5] = (first_half[2] - second_half[2]) * half;
    v[3] = (first_half[3] + second_half[3]) * half;
    v[4] = (first_half[3] - second_half[3]) * half;
}

/// NEON batched 8-point IDCT with *= 8 scaling.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn idct1d_8_batch_neon(token: archmage::NeonToken, v: &mut [magetypes::simd::f32x4; 8]) {
    use magetypes::simd::f32x4;

    let scale8 = f32x4::splat(token, 8.0);
    for vi in v.iter_mut() {
        *vi *= scale8;
    }
    idct1d_8_core_batch_neon(token, v);
}

/// NEON batched 16-point IDCT core (no scaling).
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
pub(crate) fn idct1d_16_core_batch_neon(
    token: archmage::NeonToken,
    v: &mut [magetypes::simd::f32x4; 16],
) {
    use magetypes::simd::f32x4;

    let half = f32x4::splat(token, 0.5);
    let inv_sqrt2 = f32x4::splat(token, 1.0 / SQRT2);

    // De-interleave: even -> first_half[0..8], odd -> second_half[0..8]
    let mut first_half = [v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]];
    let mut second_half = [v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]];

    // Reverse B transform on second half
    second_half[6] -= second_half[7];
    second_half[5] -= second_half[6];
    second_half[4] -= second_half[5];
    second_half[3] -= second_half[4];
    second_half[2] -= second_half[3];
    second_half[1] -= second_half[2];
    second_half[0] = (second_half[0] - second_half[1]) * inv_sqrt2;

    // IDCT-8 core on second half
    idct1d_8_core_batch_neon(token, &mut second_half);

    // Divide by WcMultipliers_16
    second_half[0] *= f32x4::splat(token, INV_WC16[0]);
    second_half[1] *= f32x4::splat(token, INV_WC16[1]);
    second_half[2] *= f32x4::splat(token, INV_WC16[2]);
    second_half[3] *= f32x4::splat(token, INV_WC16[3]);
    second_half[4] *= f32x4::splat(token, INV_WC16[4]);
    second_half[5] *= f32x4::splat(token, INV_WC16[5]);
    second_half[6] *= f32x4::splat(token, INV_WC16[6]);
    second_half[7] *= f32x4::splat(token, INV_WC16[7]);

    // IDCT-8 core on first half
    idct1d_8_core_batch_neon(token, &mut first_half);

    // Combine
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

/// NEON batched 16-point IDCT with *= 16 scaling.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
pub(crate) fn idct1d_16_batch_neon(
    token: archmage::NeonToken,
    v: &mut [magetypes::simd::f32x4; 16],
) {
    use magetypes::simd::f32x4;

    let scale16 = f32x4::splat(token, 16.0);
    for vi in v.iter_mut() {
        *vi *= scale16;
    }
    idct1d_16_core_batch_neon(token, v);
}

/// Process a batch of 4 rows through gather → 8-point IDCT → scatter.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
#[allow(clippy::needless_range_loop)]
fn neon_idct8_batch(
    token: archmage::NeonToken,
    data_in: &[f32],
    data_out: &mut [f32],
    base_row: usize,
    stride: usize,
) {
    let mut v = [magetypes::simd::f32x4::zero(token); 8];
    for j in 0..8 {
        v[j] = gather_col_neon(token, data_in, base_row, j, stride);
    }
    idct1d_8_batch_neon(token, &mut v);
    for j in 0..8 {
        scatter_col_neon(token, v[j], data_out, base_row, j, stride);
    }
}

/// Process a batch of 4 rows through gather → 16-point IDCT → scatter.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
#[allow(clippy::needless_range_loop)]
fn neon_idct16_batch(
    token: archmage::NeonToken,
    data_in: &[f32],
    data_out: &mut [f32],
    base_row: usize,
    stride: usize,
) {
    let mut v = [magetypes::simd::f32x4::zero(token); 16];
    for j in 0..16 {
        v[j] = gather_col_neon(token, data_in, base_row, j, stride);
    }
    idct1d_16_batch_neon(token, &mut v);
    for j in 0..16 {
        scatter_col_neon(token, v[j], data_out, base_row, j, stride);
    }
}

/// NEON 16x16 inverse DCT: process 4 rows at a time.
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_16x16_neon(token: archmage::NeonToken, input: &[f32; 256], output: &mut [f32; 256]) {
    let mut tmp = crate::scratch_buf::<256>();

    // Pass 1: IDCT on rows (4 batches of 4 rows)
    for batch in 0..4 {
        neon_idct16_batch(token, input, &mut tmp, batch * 4, 16);
    }

    // Transpose 16x16
    let mut transposed = crate::scratch_buf::<256>();
    for r in 0..16 {
        for c in 0..16 {
            transposed[c * 16 + r] = tmp[r * 16 + c];
        }
    }

    // Pass 2: IDCT on columns (4 batches of 4 rows)
    for batch in 0..4 {
        neon_idct16_batch(token, &transposed, output, batch * 4, 16);
    }
}

/// NEON 16x8 inverse DCT.
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_16x8_neon(token: archmage::NeonToken, input: &[f32; 128], output: &mut [f32; 128]) {
    let mut tmp = crate::scratch_buf::<128>();

    // Pass 1: 8-point IDCT on 16 rows (stride 8), 4 batches of 4 rows
    for batch in 0..4 {
        neon_idct8_batch(token, input, &mut tmp, batch * 4, 8);
    }

    // Pass 2: 16-point IDCT on 8 columns
    // Gather column data and apply IDCT via gather/scatter with stride 8
    // We need to process columns, so extract each column, apply 16-point IDCT, write back.
    // With NEON (4-wide), we process 4 columns at a time.
    for col_base in (0..8).step_by(4) {
        // Gather 16 values from 4 columns into f32x4 vectors
        let mut v = [magetypes::simd::f32x4::zero(token); 16];
        for row in 0..16 {
            v[row] = magetypes::simd::f32x4::from_array(
                token,
                [
                    tmp[row * 8 + col_base],
                    tmp[row * 8 + col_base + 1],
                    tmp[row * 8 + col_base + 2],
                    tmp[row * 8 + col_base + 3],
                ],
            );
        }
        idct1d_16_batch_neon(token, &mut v);
        for row in 0..16 {
            let mut lane = [0.0f32; 4];
            v[row].store(&mut lane);
            output[row * 8 + col_base] = lane[0];
            output[row * 8 + col_base + 1] = lane[1];
            output[row * 8 + col_base + 2] = lane[2];
            output[row * 8 + col_base + 3] = lane[3];
        }
    }
}

/// NEON 8x16 inverse DCT.
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_8x16_neon(token: archmage::NeonToken, input: &[f32; 128], output: &mut [f32; 128]) {
    let mut tmp = crate::scratch_buf::<128>();

    // Pass 1: 16-point IDCT on 8 rows (stride 16), 2 batches of 4 rows
    for batch in 0..2 {
        neon_idct16_batch(token, input, &mut tmp, batch * 4, 16);
    }

    // Pass 2: 8-point IDCT on 16 columns
    // Process 4 columns at a time (4 batches for 16 columns)
    for col_base in (0..16).step_by(4) {
        // Gather 8 values from 4 columns into f32x4 vectors
        let mut v = [magetypes::simd::f32x4::zero(token); 8];
        for row in 0..8 {
            v[row] = magetypes::simd::f32x4::from_array(
                token,
                [
                    tmp[row * 16 + col_base],
                    tmp[row * 16 + col_base + 1],
                    tmp[row * 16 + col_base + 2],
                    tmp[row * 16 + col_base + 3],
                ],
            );
        }
        idct1d_8_batch_neon(token, &mut v);
        for row in 0..8 {
            let mut lane = [0.0f32; 4];
            v[row].store(&mut lane);
            output[row * 16 + col_base] = lane[0];
            output[row * 16 + col_base + 1] = lane[1];
            output[row * 16 + col_base + 2] = lane[2];
            output[row * 16 + col_base + 3] = lane[3];
        }
    }
}

// ============================================================================
// wasm32 WASM SIMD128 implementations
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

/// WASM128 batched 4-point inverse DCT on f32x4 arrays.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn idct1d_4_batch_wasm128(token: archmage::Wasm128Token, v: &mut [magetypes::simd::f32x4; 4]) {
    use magetypes::simd::f32x4;

    let half = f32x4::splat(token, 0.5);
    let inv_sqrt2 = f32x4::splat(token, 1.0 / SQRT2);
    let inv_wc4_0 = f32x4::splat(token, INV_WC4[0]);
    let inv_wc4_1 = f32x4::splat(token, INV_WC4[1]);

    // De-interleave: [v0, v1, v2, v3] -> [v0, v2, v1, v3]
    let t0 = v[0];
    let t1 = v[2];
    let t2 = v[1];
    let t3 = v[3];

    // Reverse B transform: t2 = (t2 - t3) / sqrt2
    let t2 = (t2 - t3) * inv_sqrt2;

    // IDCT-2 on second half
    let s2 = (t2 + t3) * half;
    let s3 = (t2 - t3) * half;

    // Divide by WcMultipliers_4
    let s2 = s2 * inv_wc4_0;
    let s3 = s3 * inv_wc4_1;

    // IDCT-2 on first half
    let s0 = (t0 + t1) * half;
    let s1 = (t0 - t1) * half;

    // Combine
    v[0] = (s0 + s2) * half;
    v[3] = (s0 - s2) * half;
    v[1] = (s1 + s3) * half;
    v[2] = (s1 - s3) * half;
}

/// WASM128 batched 8-point IDCT core (no N scaling) on f32x4 arrays.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn idct1d_8_core_batch_wasm128(token: archmage::Wasm128Token, v: &mut [magetypes::simd::f32x4; 8]) {
    use magetypes::simd::f32x4;

    let half = f32x4::splat(token, 0.5);
    let inv_sqrt2 = f32x4::splat(token, 1.0 / SQRT2);

    // De-interleave: even -> first_half, odd -> second_half
    let mut first_half = [v[0], v[2], v[4], v[6]];
    let mut second_half = [v[1], v[3], v[5], v[7]];

    // Reverse B transform on second half
    second_half[2] -= second_half[3];
    second_half[1] -= second_half[2];
    second_half[0] = (second_half[0] - second_half[1]) * inv_sqrt2;

    // IDCT-4 on second half
    idct1d_4_batch_wasm128(token, &mut second_half);

    // Divide by WcMultipliers_8
    second_half[0] *= f32x4::splat(token, INV_WC8[0]);
    second_half[1] *= f32x4::splat(token, INV_WC8[1]);
    second_half[2] *= f32x4::splat(token, INV_WC8[2]);
    second_half[3] *= f32x4::splat(token, INV_WC8[3]);

    // IDCT-4 on first half
    idct1d_4_batch_wasm128(token, &mut first_half);

    // Combine
    v[0] = (first_half[0] + second_half[0]) * half;
    v[7] = (first_half[0] - second_half[0]) * half;
    v[1] = (first_half[1] + second_half[1]) * half;
    v[6] = (first_half[1] - second_half[1]) * half;
    v[2] = (first_half[2] + second_half[2]) * half;
    v[5] = (first_half[2] - second_half[2]) * half;
    v[3] = (first_half[3] + second_half[3]) * half;
    v[4] = (first_half[3] - second_half[3]) * half;
}

/// WASM128 batched 8-point IDCT with *= 8 scaling.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn idct1d_8_batch_wasm128(token: archmage::Wasm128Token, v: &mut [magetypes::simd::f32x4; 8]) {
    use magetypes::simd::f32x4;

    let scale8 = f32x4::splat(token, 8.0);
    for vi in v.iter_mut() {
        *vi *= scale8;
    }
    idct1d_8_core_batch_wasm128(token, v);
}

/// WASM128 batched 16-point IDCT core (no scaling).
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
pub(crate) fn idct1d_16_core_batch_wasm128(
    token: archmage::Wasm128Token,
    v: &mut [magetypes::simd::f32x4; 16],
) {
    use magetypes::simd::f32x4;

    let half = f32x4::splat(token, 0.5);
    let inv_sqrt2 = f32x4::splat(token, 1.0 / SQRT2);

    // De-interleave: even -> first_half[0..8], odd -> second_half[0..8]
    let mut first_half = [v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]];
    let mut second_half = [v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]];

    // Reverse B transform on second half
    second_half[6] -= second_half[7];
    second_half[5] -= second_half[6];
    second_half[4] -= second_half[5];
    second_half[3] -= second_half[4];
    second_half[2] -= second_half[3];
    second_half[1] -= second_half[2];
    second_half[0] = (second_half[0] - second_half[1]) * inv_sqrt2;

    // IDCT-8 core on second half
    idct1d_8_core_batch_wasm128(token, &mut second_half);

    // Divide by WcMultipliers_16
    second_half[0] *= f32x4::splat(token, INV_WC16[0]);
    second_half[1] *= f32x4::splat(token, INV_WC16[1]);
    second_half[2] *= f32x4::splat(token, INV_WC16[2]);
    second_half[3] *= f32x4::splat(token, INV_WC16[3]);
    second_half[4] *= f32x4::splat(token, INV_WC16[4]);
    second_half[5] *= f32x4::splat(token, INV_WC16[5]);
    second_half[6] *= f32x4::splat(token, INV_WC16[6]);
    second_half[7] *= f32x4::splat(token, INV_WC16[7]);

    // IDCT-8 core on first half
    idct1d_8_core_batch_wasm128(token, &mut first_half);

    // Combine
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

/// WASM128 batched 16-point IDCT with *= 16 scaling.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
pub(crate) fn idct1d_16_batch_wasm128(
    token: archmage::Wasm128Token,
    v: &mut [magetypes::simd::f32x4; 16],
) {
    use magetypes::simd::f32x4;

    let scale16 = f32x4::splat(token, 16.0);
    for vi in v.iter_mut() {
        *vi *= scale16;
    }
    idct1d_16_core_batch_wasm128(token, v);
}

/// Process a batch of 4 rows through gather -> 8-point IDCT -> scatter.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
#[allow(clippy::needless_range_loop)]
fn wasm128_idct8_batch(
    token: archmage::Wasm128Token,
    data_in: &[f32],
    data_out: &mut [f32],
    base_row: usize,
    stride: usize,
) {
    let mut v = [magetypes::simd::f32x4::zero(token); 8];
    for j in 0..8 {
        v[j] = gather_col_wasm128(token, data_in, base_row, j, stride);
    }
    idct1d_8_batch_wasm128(token, &mut v);
    for j in 0..8 {
        scatter_col_wasm128(token, v[j], data_out, base_row, j, stride);
    }
}

/// Process a batch of 4 rows through gather -> 16-point IDCT -> scatter.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
#[allow(clippy::needless_range_loop)]
fn wasm128_idct16_batch(
    token: archmage::Wasm128Token,
    data_in: &[f32],
    data_out: &mut [f32],
    base_row: usize,
    stride: usize,
) {
    let mut v = [magetypes::simd::f32x4::zero(token); 16];
    for j in 0..16 {
        v[j] = gather_col_wasm128(token, data_in, base_row, j, stride);
    }
    idct1d_16_batch_wasm128(token, &mut v);
    for j in 0..16 {
        scatter_col_wasm128(token, v[j], data_out, base_row, j, stride);
    }
}

/// WASM128 16x16 inverse DCT: process 4 rows at a time.
#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_16x16_wasm128(
    token: archmage::Wasm128Token,
    input: &[f32; 256],
    output: &mut [f32; 256],
) {
    let mut tmp = crate::scratch_buf::<256>();

    // Pass 1: IDCT on rows (4 batches of 4 rows)
    for batch in 0..4 {
        wasm128_idct16_batch(token, input, &mut tmp, batch * 4, 16);
    }

    // Transpose 16x16
    let mut transposed = crate::scratch_buf::<256>();
    for r in 0..16 {
        for c in 0..16 {
            transposed[c * 16 + r] = tmp[r * 16 + c];
        }
    }

    // Pass 2: IDCT on columns (4 batches of 4 rows)
    for batch in 0..4 {
        wasm128_idct16_batch(token, &transposed, output, batch * 4, 16);
    }
}

/// WASM128 16x8 inverse DCT.
#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_16x8_wasm128(
    token: archmage::Wasm128Token,
    input: &[f32; 128],
    output: &mut [f32; 128],
) {
    let mut tmp = crate::scratch_buf::<128>();

    // Pass 1: 8-point IDCT on 16 rows (stride 8), 4 batches of 4 rows
    for batch in 0..4 {
        wasm128_idct8_batch(token, input, &mut tmp, batch * 4, 8);
    }

    // Pass 2: 16-point IDCT on 8 columns
    // Gather column data and apply IDCT via gather/scatter with stride 8
    // We need to process columns, so extract each column, apply 16-point IDCT, write back.
    // With WASM128 (4-wide), we process 4 columns at a time.
    for col_base in (0..8).step_by(4) {
        // Gather 16 values from 4 columns into f32x4 vectors
        let mut v = [magetypes::simd::f32x4::zero(token); 16];
        for row in 0..16 {
            v[row] = magetypes::simd::f32x4::from_array(
                token,
                [
                    tmp[row * 8 + col_base],
                    tmp[row * 8 + col_base + 1],
                    tmp[row * 8 + col_base + 2],
                    tmp[row * 8 + col_base + 3],
                ],
            );
        }
        idct1d_16_batch_wasm128(token, &mut v);
        for row in 0..16 {
            let mut lane = [0.0f32; 4];
            v[row].store(&mut lane);
            output[row * 8 + col_base] = lane[0];
            output[row * 8 + col_base + 1] = lane[1];
            output[row * 8 + col_base + 2] = lane[2];
            output[row * 8 + col_base + 3] = lane[3];
        }
    }
}

/// WASM128 8x16 inverse DCT.
#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_8x16_wasm128(
    token: archmage::Wasm128Token,
    input: &[f32; 128],
    output: &mut [f32; 128],
) {
    let mut tmp = crate::scratch_buf::<128>();

    // Pass 1: 16-point IDCT on 8 rows (stride 16), 2 batches of 4 rows
    for batch in 0..2 {
        wasm128_idct16_batch(token, input, &mut tmp, batch * 4, 16);
    }

    // Pass 2: 8-point IDCT on 16 columns
    // Process 4 columns at a time (4 batches for 16 columns)
    for col_base in (0..16).step_by(4) {
        // Gather 8 values from 4 columns into f32x4 vectors
        let mut v = [magetypes::simd::f32x4::zero(token); 8];
        for row in 0..8 {
            v[row] = magetypes::simd::f32x4::from_array(
                token,
                [
                    tmp[row * 16 + col_base],
                    tmp[row * 16 + col_base + 1],
                    tmp[row * 16 + col_base + 2],
                    tmp[row * 16 + col_base + 3],
                ],
            );
        }
        idct1d_8_batch_wasm128(token, &mut v);
        for row in 0..8 {
            let mut lane = [0.0f32; 4];
            v[row].store(&mut lane);
            output[row * 16 + col_base] = lane[0];
            output[row * 16 + col_base + 1] = lane[1];
            output[row * 16 + col_base + 2] = lane[2];
            output[row * 16 + col_base + 3] = lane[3];
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    fn assert_simd_matches_scalar_256(
        scalar_fn: fn(&[f32; 256], &mut [f32; 256]),
        dispatch_fn: fn(&[f32; 256], &mut [f32; 256]),
        input: &[f32; 256],
        label: &str,
    ) {
        let mut scalar_out = [0.0f32; 256];
        scalar_fn(input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 256];
                dispatch_fn(input, &mut simd_out);
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
                    "{label} max diff = {max_diff} at {max_idx} (scalar={}, simd={}) [{perm}]",
                    scalar_out[max_idx],
                    simd_out[max_idx],
                );
            },
        );
        std::eprintln!("{label}: {report}");
    }

    fn assert_simd_matches_scalar_128(
        scalar_fn: fn(&[f32; 128], &mut [f32; 128]),
        dispatch_fn: fn(&[f32; 128], &mut [f32; 128]),
        input: &[f32; 128],
        label: &str,
    ) {
        let mut scalar_out = [0.0f32; 128];
        scalar_fn(input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 128];
                dispatch_fn(input, &mut simd_out);
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
                    "{label} max diff = {max_diff} at {max_idx} (scalar={}, simd={}) [{perm}]",
                    scalar_out[max_idx],
                    simd_out[max_idx],
                );
            },
        );
        std::eprintln!("{label}: {report}");
    }

    #[test]
    fn test_idct_16x16_simd_matches_scalar() {
        let mut input = [0.0f32; 256];
        for (i, val) in input.iter_mut().enumerate() {
            *val = i as f32;
        }
        assert_simd_matches_scalar_256(idct_16x16_scalar, idct_16x16, &input, "IDCT16x16 seq");
    }

    #[test]
    fn test_idct_16x16_simd_matches_scalar_cosine_input() {
        let mut input = [0.0f32; 256];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.37 + 1.5).cos() * 100.0;
        }
        assert_simd_matches_scalar_256(idct_16x16_scalar, idct_16x16, &input, "IDCT16x16 cos");
    }

    #[test]
    fn test_idct_16x16_dc_only() {
        let mut input = [0.0f32; 256];
        input[0] = 128.0;
        assert_simd_matches_scalar_256(idct_16x16_scalar, idct_16x16, &input, "IDCT16x16 DC");
    }

    #[test]
    fn test_idct_16x16_single_ac_coefficient() {
        let mut input = [0.0f32; 256];
        input[1] = 50.0;
        assert_simd_matches_scalar_256(idct_16x16_scalar, idct_16x16, &input, "IDCT16x16 AC");
    }

    #[test]
    fn test_idct_16x8_simd_matches_scalar() {
        let mut input = [0.0f32; 128];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.43 + 2.1).cos() * 80.0;
        }
        assert_simd_matches_scalar_128(idct_16x8_scalar, idct_16x8, &input, "IDCT16x8");
    }

    #[test]
    fn test_idct_8x16_simd_matches_scalar() {
        let mut input = [0.0f32; 128];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.29 + 0.7).sin() * 120.0;
        }
        assert_simd_matches_scalar_128(idct_8x16_scalar, idct_8x16, &input, "IDCT8x16");
    }
}
