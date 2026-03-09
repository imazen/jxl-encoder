// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated 64×64/64×32/32×64 inverse DCT.
//!
//! Processes 8 independent 64-point IDCTs in parallel using AVX2 f32x8 vectors.
//! The 64-point batch IDCT recursively calls the 32-point core IDCT from `idct32.rs`.

// Constants matching jxl_encoder/src/vardct/dct/constants.rs
// Full f64 precision — Rust truncates to nearest f32 at compile time.
#[allow(clippy::excessive_precision)]
const SQRT2: f32 = core::f32::consts::SQRT_2;
#[allow(clippy::excessive_precision)]
const WC_MULTIPLIERS_4: [f32; 2] = [0.541196100146197, 1.3065629648763764];
#[allow(clippy::excessive_precision)]
const WC_MULTIPLIERS_8: [f32; 4] = [
    0.5097955791041592,
    0.6013448869350453,
    0.8999762231364156,
    2.5629154477415055,
];
#[allow(clippy::excessive_precision)]
const WC_MULTIPLIERS_16: [f32; 8] = [
    0.5024192861881557,
    0.5224986149396889,
    0.5669440348163577,
    0.6468217833599901,
    0.7881546234512502,
    1.060677685990347,
    1.7224470982383342,
    5.101148618689155,
];
#[allow(clippy::excessive_precision)]
const WC_MULTIPLIERS_32: [f32; 16] = [
    0.5006029982351963,
    0.5054709598975436,
    0.5154473099226246,
    0.5310425910897841,
    0.5531038960344445,
    0.5829349682061339,
    0.6225041230356648,
    0.6748083414550057,
    0.7445362710022986,
    0.8393496454155268,
    0.9725682378619608,
    1.1694399334328847,
    1.4841646163141662,
    2.057781009953411,
    3.407608418468719,
    10.190008123548033,
];
#[allow(clippy::excessive_precision)]
const WC_MULTIPLIERS_64: [f32; 32] = [
    0.500150636020651,
    0.5013584524464084,
    0.5037887256810443,
    0.5074711720725553,
    0.5124514794082247,
    0.5187927131053328,
    0.52657731515427,
    0.535909816907992,
    0.5469204379855088,
    0.5597698129470802,
    0.57465518403266,
    0.5918185358574165,
    0.6115573478825099,
    0.6342389366884031,
    0.6603198078137061,
    0.6903721282002123,
    0.7251205223771985,
    0.7654941649730891,
    0.8127020908144905,
    0.8683447152233481,
    0.9345835970364075,
    1.0144082649970547,
    1.1120716205797176,
    1.233832737976571,
    1.3892939586328277,
    1.5939722833856311,
    1.8746759800084078,
    2.282050068005162,
    2.924628428158216,
    4.084611078129248,
    6.796750711673633,
    20.373878167231453,
];

// Pre-computed reciprocals to replace division with multiplication.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::excessive_precision)]
const INV_WC64: [f32; 32] = [
    1.0 / 0.500150636020651,
    1.0 / 0.5013584524464084,
    1.0 / 0.5037887256810443,
    1.0 / 0.5074711720725553,
    1.0 / 0.5124514794082247,
    1.0 / 0.5187927131053328,
    1.0 / 0.52657731515427,
    1.0 / 0.535909816907992,
    1.0 / 0.5469204379855088,
    1.0 / 0.5597698129470802,
    1.0 / 0.57465518403266,
    1.0 / 0.5918185358574165,
    1.0 / 0.6115573478825099,
    1.0 / 0.6342389366884031,
    1.0 / 0.6603198078137061,
    1.0 / 0.6903721282002123,
    1.0 / 0.7251205223771985,
    1.0 / 0.7654941649730891,
    1.0 / 0.8127020908144905,
    1.0 / 0.8683447152233481,
    1.0 / 0.9345835970364075,
    1.0 / 1.0144082649970547,
    1.0 / 1.1120716205797176,
    1.0 / 1.233832737976571,
    1.0 / 1.3892939586328277,
    1.0 / 1.5939722833856311,
    1.0 / 1.8746759800084078,
    1.0 / 2.282050068005162,
    1.0 / 2.924628428158216,
    1.0 / 4.084611078129248,
    1.0 / 6.796750711673633,
    1.0 / 20.373878167231453,
];

// ============================================================================
// Scalar fallback — self-contained 64-point IDCT chain
// ============================================================================

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
    let mut tmp = crate::scratch_buf::<8>();
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

/// Core 16-point IDCT without the N scaling factor.
fn idct1d_16_core_scalar(mem: &mut [f32]) {
    let mut tmp = crate::scratch_buf::<16>();
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

/// Core 32-point IDCT without the N scaling factor.
fn idct1d_32_core_scalar(mem: &mut [f32]) {
    let mut tmp = crate::scratch_buf::<32>();
    for i in 0..16 {
        tmp[i] = mem[2 * i];
        tmp[16 + i] = mem[2 * i + 1];
    }

    // Reverse B transform
    for i in (1..15).rev() {
        tmp[16 + i] -= tmp[16 + i + 1];
    }
    tmp[16] = (tmp[16] - tmp[17]) / SQRT2;

    // IDCT-16 core on second half
    idct1d_16_core_scalar(&mut tmp[16..32]);

    // Divide by WcMultipliers
    for i in 0..16 {
        tmp[16 + i] /= WC_MULTIPLIERS_32[i];
    }

    // IDCT-16 core on first half
    idct1d_16_core_scalar(&mut tmp[0..16]);

    // Combine
    for i in 0..16 {
        mem[i] = (tmp[i] + tmp[16 + i]) * 0.5;
        mem[31 - i] = (tmp[i] - tmp[16 + i]) * 0.5;
    }
}

/// 32-point IDCT with *= 32 scaling.
fn idct1d_32_scalar(mem: &mut [f32]) {
    for x in mem.iter_mut().take(32) {
        *x *= 32.0;
    }
    idct1d_32_core_scalar(mem);
}

/// Core 64-point IDCT without the N scaling factor.
fn idct1d_64_core_scalar(mem: &mut [f32]) {
    let mut tmp = crate::scratch_buf::<64>();
    for i in 0..32 {
        tmp[i] = mem[2 * i];
        tmp[32 + i] = mem[2 * i + 1];
    }

    // Reverse B transform
    for i in (1..31).rev() {
        tmp[32 + i] -= tmp[32 + i + 1];
    }
    tmp[32] = (tmp[32] - tmp[33]) / SQRT2;

    // IDCT-32 core on second half
    idct1d_32_core_scalar(&mut tmp[32..64]);

    // Divide by WcMultipliers
    for i in 0..32 {
        tmp[32 + i] /= WC_MULTIPLIERS_64[i];
    }

    // IDCT-32 core on first half
    idct1d_32_core_scalar(&mut tmp[0..32]);

    // Combine
    for i in 0..32 {
        mem[i] = (tmp[i] + tmp[32 + i]) * 0.5;
        mem[63 - i] = (tmp[i] - tmp[32 + i]) * 0.5;
    }
}

/// 64-point IDCT with *= 64 scaling.
fn idct1d_64_scalar(mem: &mut [f32]) {
    for x in mem.iter_mut().take(64) {
        *x *= 64.0;
    }
    idct1d_64_core_scalar(mem);
}

/// Scalar 64×64 inverse DCT.
#[inline]
pub fn idct_64x64_scalar(input: &[f32; 4096], output: &mut [f32; 4096]) {
    let mut tmp = crate::scratch_buf::<4096>();

    // IDCT on each row
    for row in 0..64 {
        let s = row * 64;
        tmp[s..s + 64].copy_from_slice(&input[s..s + 64]);
        idct1d_64_scalar(&mut tmp[s..s + 64]);
    }

    // Transpose 64×64
    let mut transposed = crate::scratch_buf::<4096>();
    for r in 0..64 {
        for c in 0..64 {
            transposed[c * 64 + r] = tmp[r * 64 + c];
        }
    }

    // IDCT on each column (now rows of transposed)
    for row in 0..64 {
        let s = row * 64;
        output[s..s + 64].copy_from_slice(&transposed[s..s + 64]);
        idct1d_64_scalar(&mut output[s..s + 64]);
    }
}

/// Scalar 64×32 inverse DCT.
///
/// Reverses dct_64x32: input in 32×64 layout (stride 64).
/// Output in 64×32 layout (stride 32, spatial domain).
#[inline]
pub fn idct_64x32_scalar(input: &[f32; 2048], output: &mut [f32; 2048]) {
    let mut tmp = crate::scratch_buf::<2048>();

    // IDCT-64 on each of 32 rows (stride 64)
    for row in 0..32 {
        let s = row * 64;
        tmp[s..s + 64].copy_from_slice(&input[s..s + 64]);
        idct1d_64_scalar(&mut tmp[s..s + 64]);
    }

    // Transpose 32×64 → 64×32
    let mut transposed = crate::scratch_buf::<2048>();
    for r in 0..32 {
        for c in 0..64 {
            transposed[c * 32 + r] = tmp[r * 64 + c];
        }
    }

    // IDCT-32 on each of 64 rows (stride 32)
    for row in 0..64 {
        let s = row * 32;
        output[s..s + 32].copy_from_slice(&transposed[s..s + 32]);
        idct1d_32_scalar(&mut output[s..s + 32]);
    }
}

/// Scalar 32×64 inverse DCT.
///
/// Reverses dct_32x64: input in 32×64 layout (stride 64).
/// Output in 32×64 layout (stride 64, spatial domain).
#[inline]
pub fn idct_32x64_scalar(input: &[f32; 2048], output: &mut [f32; 2048]) {
    // Un-transpose: 32×64 → 64×32
    let mut transposed = crate::scratch_buf::<2048>();
    for r in 0..32 {
        for c in 0..64 {
            transposed[c * 32 + r] = input[r * 64 + c];
        }
    }

    // IDCT-32 on each of 64 rows (stride 32)
    let mut tmp = crate::scratch_buf::<2048>();
    for row in 0..64 {
        let s = row * 32;
        tmp[s..s + 32].copy_from_slice(&transposed[s..s + 32]);
        idct1d_32_scalar(&mut tmp[s..s + 32]);
    }

    // Transpose 64×32 → 32×64
    let mut transposed2 = crate::scratch_buf::<2048>();
    for r in 0..64 {
        for c in 0..32 {
            transposed2[c * 64 + r] = tmp[r * 32 + c];
        }
    }

    // IDCT-64 on each of 32 rows (stride 64)
    for row in 0..32 {
        let s = row * 64;
        output[s..s + 64].copy_from_slice(&transposed2[s..s + 64]);
        idct1d_64_scalar(&mut output[s..s + 64]);
    }
}

// ============================================================================
// x86_64 AVX2 implementation
// ============================================================================

/// Load column `j` from 8 consecutive rows starting at `base_row` with given stride.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn gather_col(
    token: archmage::X64V3Token,
    data: &[f32],
    base_row: usize,
    j: usize,
    stride: usize,
) -> magetypes::simd::f32x8 {
    crate::gather_col_strided(token, data, base_row, j, stride)
}

/// Store f32x8 lanes back to column `j` of 8 consecutive rows with given stride.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scatter_col(
    v: magetypes::simd::f32x8,
    data: &mut [f32],
    base_row: usize,
    j: usize,
    stride: usize,
) {
    crate::scatter_col_strided(v, data, base_row, j, stride)
}

/// AVX2 batched 64-point inverse DCT with *= 64 scaling.
///
/// `v[0..64]` holds positions 0-63 across 8 independent 1D transforms.
/// Recursively calls `idct1d_32_core_batch` from `idct32.rs` for the two halves.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
pub(crate) fn idct1d_64_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 64]) {
    use magetypes::simd::f32x8;

    let scale64 = f32x8::splat(token, 64.0);

    // Scale by 64
    for vi in v.iter_mut() {
        *vi *= scale64;
    }

    idct1d_64_core_batch(token, v);
}

/// AVX2 batched 64-point core inverse DCT WITHOUT scaling.
///
/// `v[0..64]` holds positions 0-63 across 8 independent 1D transforms.
/// Does NOT apply the *= 64 scaling factor.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
pub(crate) fn idct1d_64_core_batch(
    token: archmage::X64V3Token,
    v: &mut [magetypes::simd::f32x8; 64],
) {
    use magetypes::simd::f32x8;

    let half = f32x8::splat(token, 0.5);
    let inv_sqrt2 = f32x8::splat(token, 1.0 / SQRT2);

    // De-interleave: even → first_half, odd → second_half
    let mut first = [f32x8::zero(token); 32];
    let mut second = [f32x8::zero(token); 32];
    for i in 0..32 {
        first[i] = v[2 * i];
        second[i] = v[2 * i + 1];
    }

    // Reverse B transform on second half
    for i in (1..31).rev() {
        second[i] -= second[i + 1];
    }
    second[0] = (second[0] - second[1]) * inv_sqrt2;

    // IDCT-32 core on second half (no scaling)
    crate::idct32::idct1d_32_core_batch(token, &mut second);

    // Divide by WcMultipliers_64 (multiply by inverse)
    for i in 0..32 {
        second[i] *= f32x8::splat(token, INV_WC64[i]);
    }

    // IDCT-32 core on first half (no scaling)
    crate::idct32::idct1d_32_core_batch(token, &mut first);

    // Combine
    for i in 0..32 {
        v[i] = (first[i] + second[i]) * half;
        v[63 - i] = (first[i] - second[i]) * half;
    }
}

/// AVX2 64×64 inverse DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_64x64_avx2(token: archmage::X64V3Token, input: &[f32; 4096], output: &mut [f32; 4096]) {
    use magetypes::simd::f32x8;

    let mut tmp = crate::scratch_buf::<4096>();

    // Pass 1: IDCT-64 on rows, 8 batches of 8 rows
    for batch in 0..8 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 64];
        for j in 0..64 {
            v[j] = gather_col(token, input, base, j, 64);
        }
        idct1d_64_batch(token, &mut v);
        for j in 0..64 {
            scatter_col(v[j], &mut tmp, base, j, 64);
        }
    }

    // Transpose 64×64
    let mut transposed = crate::scratch_buf::<4096>();
    for r in 0..64 {
        for c in 0..64 {
            transposed[c * 64 + r] = tmp[r * 64 + c];
        }
    }

    // Pass 2: IDCT-64 on columns (now rows), 8 batches of 8 rows
    for batch in 0..8 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 64];
        for j in 0..64 {
            v[j] = gather_col(token, &transposed, base, j, 64);
        }
        idct1d_64_batch(token, &mut v);
        for j in 0..64 {
            scatter_col(v[j], output, base, j, 64);
        }
    }
}

/// AVX2 64×32 inverse DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_64x32_avx2(token: archmage::X64V3Token, input: &[f32; 2048], output: &mut [f32; 2048]) {
    use magetypes::simd::f32x8;

    let mut tmp = crate::scratch_buf::<2048>();

    // Pass 1: IDCT-64 on 32 rows (stride 64), 4 batches of 8
    for batch in 0..4 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 64];
        for j in 0..64 {
            v[j] = gather_col(token, input, base, j, 64);
        }
        idct1d_64_batch(token, &mut v);
        for j in 0..64 {
            scatter_col(v[j], &mut tmp, base, j, 64);
        }
    }

    // Transpose 32×64 → 64×32
    let mut transposed = crate::scratch_buf::<2048>();
    for r in 0..32 {
        for c in 0..64 {
            transposed[c * 32 + r] = tmp[r * 64 + c];
        }
    }

    // Pass 2: IDCT-32 on 64 rows (stride 32), 8 batches of 8
    for batch in 0..8 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 32];
        for j in 0..32 {
            v[j] = gather_col(token, &transposed, base, j, 32);
        }
        crate::idct32::idct1d_32_batch(token, &mut v);
        for j in 0..32 {
            scatter_col(v[j], output, base, j, 32);
        }
    }
}

/// AVX2 32×64 inverse DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_32x64_avx2(token: archmage::X64V3Token, input: &[f32; 2048], output: &mut [f32; 2048]) {
    use magetypes::simd::f32x8;

    // Un-transpose: 32×64 → 64×32
    let mut transposed = crate::scratch_buf::<2048>();
    for r in 0..32 {
        for c in 0..64 {
            transposed[c * 32 + r] = input[r * 64 + c];
        }
    }

    // Pass 1: IDCT-32 on 64 rows (stride 32), 8 batches of 8
    let mut tmp = crate::scratch_buf::<2048>();
    for batch in 0..8 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 32];
        for j in 0..32 {
            v[j] = gather_col(token, &transposed, base, j, 32);
        }
        crate::idct32::idct1d_32_batch(token, &mut v);
        for j in 0..32 {
            scatter_col(v[j], &mut tmp, base, j, 32);
        }
    }

    // Transpose 64×32 → 32×64
    let mut transposed2 = crate::scratch_buf::<2048>();
    for r in 0..64 {
        for c in 0..32 {
            transposed2[c * 64 + r] = tmp[r * 32 + c];
        }
    }

    // Pass 2: IDCT-64 on 32 rows (stride 64), 4 batches of 8
    for batch in 0..4 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 64];
        for j in 0..64 {
            v[j] = gather_col(token, &transposed2, base, j, 64);
        }
        idct1d_64_batch(token, &mut v);
        for j in 0..64 {
            scatter_col(v[j], output, base, j, 64);
        }
    }
}

// ============================================================================
// Dispatchers
// ============================================================================

/// Compute 64×64 inverse DCT with SIMD acceleration.
///
/// Input: 4096 f32 DCT coefficients.
/// Output: 4096 f32 in row-major order (spatial domain).
#[inline]
pub fn idct_64x64(input: &[f32; 4096], output: &mut [f32; 4096]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_64x64_avx2(token, input, output);
            return;
        }
    }
    idct_64x64_scalar(input, output);
}

/// Compute 64×32 inverse DCT with SIMD acceleration.
///
/// Input: 2048 f32 DCT coefficients in 32×64 layout (stride 64).
/// Output: 2048 f32 in 64×32 row-major order (stride 32, spatial domain).
#[inline]
pub fn idct_64x32(input: &[f32; 2048], output: &mut [f32; 2048]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_64x32_avx2(token, input, output);
            return;
        }
    }
    idct_64x32_scalar(input, output);
}

/// Compute 32×64 inverse DCT with SIMD acceleration.
///
/// Input: 2048 f32 DCT coefficients in 32×64 layout (stride 64).
/// Output: 2048 f32 in 32×64 row-major order (stride 64, spatial domain).
#[inline]
pub fn idct_32x64(input: &[f32; 2048], output: &mut [f32; 2048]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_32x64_avx2(token, input, output);
            return;
        }
    }
    idct_32x64_scalar(input, output);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    fn assert_simd_matches_scalar_4096(
        scalar_fn: fn(&[f32; 4096], &mut [f32; 4096]),
        dispatch_fn: fn(&[f32; 4096], &mut [f32; 4096]),
        input: &[f32; 4096],
        label: &str,
    ) {
        let mut scalar_out = [0.0f32; 4096];
        scalar_fn(input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 4096];
                dispatch_fn(input, &mut simd_out);
                let mut max_diff = 0.0f32;
                let mut max_idx = 0;
                for i in 0..4096 {
                    let diff = (scalar_out[i] - simd_out[i]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = i;
                    }
                }
                // Relative tolerance: 1e-4 of max magnitude, floor 1e-2
                let max_mag = scalar_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let tol = (max_mag * 1e-4).max(1e-2);
                assert!(
                    max_diff < tol,
                    "{label} max diff = {max_diff} at {max_idx} (scalar={}, simd={}, tol={tol}) [{perm}]",
                    scalar_out[max_idx],
                    simd_out[max_idx],
                );
            },
        );
        std::eprintln!("{label}: {report}");
    }

    fn assert_simd_matches_scalar_2048(
        scalar_fn: fn(&[f32; 2048], &mut [f32; 2048]),
        dispatch_fn: fn(&[f32; 2048], &mut [f32; 2048]),
        input: &[f32; 2048],
        label: &str,
    ) {
        let mut scalar_out = [0.0f32; 2048];
        scalar_fn(input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 2048];
                dispatch_fn(input, &mut simd_out);
                let mut max_diff = 0.0f32;
                let mut max_idx = 0;
                for i in 0..2048 {
                    let diff = (scalar_out[i] - simd_out[i]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = i;
                    }
                }
                let max_mag = scalar_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let tol = (max_mag * 1e-4).max(1e-2);
                assert!(
                    max_diff < tol,
                    "{label} max diff = {max_diff} at {max_idx} (scalar={}, simd={}, tol={tol}) [{perm}]",
                    scalar_out[max_idx],
                    simd_out[max_idx],
                );
            },
        );
        std::eprintln!("{label}: {report}");
    }

    #[test]
    fn test_idct_64x64_simd_matches_scalar() {
        let mut input = [0.0f32; 4096];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.31 + 1.7).cos() * 100.0;
        }
        assert_simd_matches_scalar_4096(idct_64x64_scalar, idct_64x64, &input, "IDCT64x64 cos");
    }

    #[test]
    fn test_idct_64x64_dc_only() {
        let mut input = [0.0f32; 4096];
        input[0] = 128.0;
        assert_simd_matches_scalar_4096(idct_64x64_scalar, idct_64x64, &input, "IDCT64x64 DC");
    }

    #[test]
    fn test_idct_64x64_sequential() {
        let mut input = [0.0f32; 4096];
        for (i, val) in input.iter_mut().enumerate() {
            *val = i as f32;
        }
        assert_simd_matches_scalar_4096(idct_64x64_scalar, idct_64x64, &input, "IDCT64x64 seq");
    }

    #[test]
    fn test_idct_64x32_simd_matches_scalar() {
        let mut input = [0.0f32; 2048];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.43 + 2.1).cos() * 80.0;
        }
        assert_simd_matches_scalar_2048(idct_64x32_scalar, idct_64x32, &input, "IDCT64x32");
    }

    #[test]
    fn test_idct_32x64_simd_matches_scalar() {
        let mut input = [0.0f32; 2048];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.29 + 0.7).sin() * 120.0;
        }
        assert_simd_matches_scalar_2048(idct_32x64_scalar, idct_32x64, &input, "IDCT32x64");
    }

    /// Verify DCT64 → IDCT64 roundtrip is near-identity.
    #[test]
    fn test_dct64_idct64_roundtrip() {
        let mut input = [0.0f32; 4096];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.17 + 3.2).sin() * 50.0;
        }

        let mut coeffs = [0.0f32; 4096];
        crate::dct64::dct_64x64(&input, &mut coeffs);

        let mut output = [0.0f32; 4096];
        idct_64x64(&coeffs, &mut output);

        let mut max_diff = 0.0f32;
        for i in 0..4096 {
            let diff = (input[i] - output[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1.0,
            "DCT64→IDCT64 roundtrip max error = {max_diff}"
        );
    }

    /// Verify DCT64x32 → IDCT64x32 roundtrip.
    #[test]
    fn test_dct64x32_idct64x32_roundtrip() {
        let mut input = [0.0f32; 2048];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.23 + 1.1).cos() * 60.0;
        }

        let mut coeffs = [0.0f32; 2048];
        crate::dct64::dct_64x32(&input, &mut coeffs);

        let mut output = [0.0f32; 2048];
        idct_64x32(&coeffs, &mut output);

        let mut max_diff = 0.0f32;
        for i in 0..2048 {
            let diff = (input[i] - output[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1.0,
            "DCT64x32→IDCT64x32 roundtrip max error = {max_diff}"
        );
    }

    /// Verify DCT32x64 → IDCT32x64 roundtrip.
    #[test]
    fn test_dct32x64_idct32x64_roundtrip() {
        let mut input = [0.0f32; 2048];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.37 + 2.5).sin() * 40.0;
        }

        let mut coeffs = [0.0f32; 2048];
        crate::dct64::dct_32x64(&input, &mut coeffs);

        let mut output = [0.0f32; 2048];
        idct_32x64(&coeffs, &mut output);

        let mut max_diff = 0.0f32;
        for i in 0..2048 {
            let diff = (input[i] - output[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1.0,
            "DCT32x64→IDCT32x64 roundtrip max error = {max_diff}"
        );
    }
}
