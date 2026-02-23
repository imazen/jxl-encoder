// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated 64×64/64×32/32×64 forward DCT.
//!
//! Processes 8 independent 64-point DCTs in parallel using AVX2 f32x8 vectors.
//! Each f32x8 lane holds one row's element at a given column position, so the
//! butterfly operates across registers (cross-position) while SIMD parallelism
//! handles multiple rows simultaneously.
//!
//! The 64-point batch DCT recursively calls the 32-point batch DCT from `dct32.rs`.

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

// ============================================================================
// Scalar fallback — self-contained 64-point DCT chain
// ============================================================================

#[inline]
fn dct1d_2_scalar(mem: &mut [f32]) {
    let x = mem[0] + mem[1];
    let y = mem[0] - mem[1];
    mem[0] = x;
    mem[1] = y;
}

fn dct1d_4_scalar(mem: &mut [f32]) {
    let mut tmp = crate::scratch_buf::<4>();
    tmp[0] = mem[0] + mem[3];
    tmp[1] = mem[1] + mem[2];
    tmp[2] = mem[0] - mem[3];
    tmp[3] = mem[1] - mem[2];

    dct1d_2_scalar(&mut tmp[0..2]);

    tmp[2] *= WC_MULTIPLIERS_4[0];
    tmp[3] *= WC_MULTIPLIERS_4[1];

    dct1d_2_scalar(&mut tmp[2..4]);

    tmp[2] = SQRT2 * tmp[2] + tmp[3];

    mem[0] = tmp[0];
    mem[1] = tmp[2];
    mem[2] = tmp[1];
    mem[3] = tmp[3];
}

fn dct1d_8_scalar(mem: &mut [f32]) {
    let mut tmp = crate::scratch_buf::<8>();
    for i in 0..4 {
        tmp[i] = mem[i] + mem[7 - i];
        tmp[4 + i] = mem[i] - mem[7 - i];
    }

    dct1d_4_scalar(&mut tmp[0..4]);

    for i in 0..4 {
        tmp[4 + i] *= WC_MULTIPLIERS_8[i];
    }

    dct1d_4_scalar(&mut tmp[4..8]);

    tmp[4] = SQRT2 * tmp[4] + tmp[5];
    for i in 1..3 {
        tmp[4 + i] += tmp[4 + i + 1];
    }

    for i in 0..4 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[4 + i];
    }
}

fn dct1d_16_scalar(mem: &mut [f32]) {
    let mut tmp = crate::scratch_buf::<16>();
    for i in 0..8 {
        tmp[i] = mem[i] + mem[15 - i];
        tmp[8 + i] = mem[i] - mem[15 - i];
    }

    dct1d_8_scalar(&mut tmp[0..8]);

    for i in 0..8 {
        tmp[8 + i] *= WC_MULTIPLIERS_16[i];
    }

    dct1d_8_scalar(&mut tmp[8..16]);

    tmp[8] = SQRT2 * tmp[8] + tmp[9];
    for i in 1..7 {
        tmp[8 + i] += tmp[8 + i + 1];
    }

    for i in 0..8 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[8 + i];
    }
}

fn dct1d_32_scalar(mem: &mut [f32]) {
    let mut tmp = crate::scratch_buf::<32>();
    for i in 0..16 {
        tmp[i] = mem[i] + mem[31 - i];
        tmp[16 + i] = mem[i] - mem[31 - i];
    }

    dct1d_16_scalar(&mut tmp[0..16]);

    for i in 0..16 {
        tmp[16 + i] *= WC_MULTIPLIERS_32[i];
    }

    dct1d_16_scalar(&mut tmp[16..32]);

    tmp[16] = SQRT2 * tmp[16] + tmp[17];
    for i in 1..15 {
        tmp[16 + i] += tmp[16 + i + 1];
    }

    for i in 0..16 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[16 + i];
    }
}

fn dct1d_64_scalar(mem: &mut [f32]) {
    let mut tmp = crate::scratch_buf::<64>();
    for i in 0..32 {
        tmp[i] = mem[i] + mem[63 - i];
        tmp[32 + i] = mem[i] - mem[63 - i];
    }

    dct1d_32_scalar(&mut tmp[0..32]);

    for i in 0..32 {
        tmp[32 + i] *= WC_MULTIPLIERS_64[i];
    }

    dct1d_32_scalar(&mut tmp[32..64]);

    tmp[32] = SQRT2 * tmp[32] + tmp[33];
    for i in 1..31 {
        tmp[32 + i] += tmp[32 + i + 1];
    }

    for i in 0..32 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[32 + i];
    }
}

/// Scalar 64×64 forward DCT.
///
/// No final transpose for square blocks (ROWS ≥ COLS branch).
#[inline]
pub fn dct_64x64_scalar(input: &[f32; 4096], output: &mut [f32; 4096]) {
    let mut tmp = crate::scratch_buf::<4096>();

    for row in 0..64 {
        let s = row * 64;
        tmp[s..s + 64].copy_from_slice(&input[s..s + 64]);
        dct1d_64_scalar(&mut tmp[s..s + 64]);
        for v in tmp[s..s + 64].iter_mut() {
            *v *= 1.0 / 64.0;
        }
    }

    let mut transposed = crate::scratch_buf::<4096>();
    for r in 0..64 {
        for c in 0..64 {
            transposed[c * 64 + r] = tmp[r * 64 + c];
        }
    }

    for row in 0..64 {
        let s = row * 64;
        dct1d_64_scalar(&mut transposed[s..s + 64]);
        for v in transposed[s..s + 64].iter_mut() {
            *v *= 1.0 / 64.0;
        }
    }

    output.copy_from_slice(&transposed);
}

/// Scalar 64×32 forward DCT.
///
/// Output in 32×64 layout (stride 64). No final transpose (ROWS ≥ COLS).
#[inline]
pub fn dct_64x32_scalar(input: &[f32; 2048], output: &mut [f32; 2048]) {
    let mut tmp = crate::scratch_buf::<2048>();

    for row in 0..64 {
        let s = row * 32;
        tmp[s..s + 32].copy_from_slice(&input[s..s + 32]);
        dct1d_32_scalar(&mut tmp[s..s + 32]);
        for v in tmp[s..s + 32].iter_mut() {
            *v *= 1.0 / 32.0;
        }
    }

    let mut transposed = crate::scratch_buf::<2048>();
    for r in 0..64 {
        for c in 0..32 {
            transposed[c * 64 + r] = tmp[r * 32 + c];
        }
    }

    for row in 0..32 {
        let s = row * 64;
        dct1d_64_scalar(&mut transposed[s..s + 64]);
        for v in transposed[s..s + 64].iter_mut() {
            *v *= 1.0 / 64.0;
        }
    }

    output.copy_from_slice(&transposed);
}

/// Scalar 32×64 forward DCT.
///
/// Output in 32×64 layout (stride 64). Final transpose (ROWS < COLS).
#[inline]
pub fn dct_32x64_scalar(input: &[f32; 2048], output: &mut [f32; 2048]) {
    let mut tmp = crate::scratch_buf::<2048>();

    for row in 0..32 {
        let s = row * 64;
        tmp[s..s + 64].copy_from_slice(&input[s..s + 64]);
        dct1d_64_scalar(&mut tmp[s..s + 64]);
        for v in tmp[s..s + 64].iter_mut() {
            *v *= 1.0 / 64.0;
        }
    }

    let mut transposed = crate::scratch_buf::<2048>();
    for r in 0..32 {
        for c in 0..64 {
            transposed[c * 32 + r] = tmp[r * 64 + c];
        }
    }

    for row in 0..64 {
        let s = row * 32;
        dct1d_32_scalar(&mut transposed[s..s + 32]);
        for v in transposed[s..s + 32].iter_mut() {
            *v *= 1.0 / 32.0;
        }
    }

    // Final transpose 64×32 → 32×64 (ROWS < COLS branch)
    for r in 0..64 {
        for c in 0..32 {
            output[c * 64 + r] = transposed[r * 32 + c];
        }
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

/// AVX2 batched 64-point forward DCT.
///
/// `v[0..64]` holds positions 0-63 across 8 independent 1D transforms.
/// Recursively calls `dct1d_32_batch` from `dct32.rs` for the two halves.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
pub(crate) fn dct1d_64_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 64]) {
    use magetypes::simd::f32x8;

    let sqrt2 = f32x8::splat(token, SQRT2);

    // AddReverse + SubReverse
    let mut a = [f32x8::zero(token); 32];
    let mut s = [f32x8::zero(token); 32];
    for i in 0..32 {
        a[i] = v[i] + v[63 - i];
        s[i] = v[i] - v[63 - i];
    }

    // DCT-32 on first half
    crate::dct32::dct1d_32_batch(token, &mut a);

    // Multiply second half by WcMultipliers_64
    for i in 0..32 {
        s[i] *= f32x8::splat(token, WC_MULTIPLIERS_64[i]);
    }

    // DCT-32 on second half
    crate::dct32::dct1d_32_batch(token, &mut s);

    // B transform on second half
    s[0] = sqrt2 * s[0] + s[1];
    for i in 1..31 {
        s[i] += s[i + 1];
    }

    // InverseEvenOdd interleave
    for i in 0..32 {
        v[2 * i] = a[i];
        v[2 * i + 1] = s[i];
    }
}

/// AVX2 64×64 forward DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_64x64_avx2(token: archmage::X64V3Token, input: &[f32; 4096], output: &mut [f32; 4096]) {
    use magetypes::simd::f32x8;

    let inv64 = f32x8::splat(token, 1.0 / 64.0);
    let mut tmp = crate::scratch_buf::<4096>();

    // Pass 1: DCT-64 on rows, 8 batches of 8 rows
    for batch in 0..8 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 64];
        for j in 0..64 {
            v[j] = gather_col(token, input, base, j, 64);
        }
        dct1d_64_batch(token, &mut v);
        for j in 0..64 {
            v[j] *= inv64;
        }
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

    // Pass 2: DCT-64 on columns (now rows of transposed), 8 batches of 8 rows
    for batch in 0..8 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 64];
        for j in 0..64 {
            v[j] = gather_col(token, &transposed, base, j, 64);
        }
        dct1d_64_batch(token, &mut v);
        for j in 0..64 {
            v[j] *= inv64;
        }
        for j in 0..64 {
            scatter_col(v[j], output, base, j, 64);
        }
    }
}

/// AVX2 64×32 forward DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_64x32_avx2(token: archmage::X64V3Token, input: &[f32; 2048], output: &mut [f32; 2048]) {
    use magetypes::simd::f32x8;

    let inv32 = f32x8::splat(token, 1.0 / 32.0);
    let inv64 = f32x8::splat(token, 1.0 / 64.0);
    let mut tmp = crate::scratch_buf::<2048>();

    // Pass 1: DCT-32 on 64 rows (stride 32), 8 batches of 8
    for batch in 0..8 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 32];
        for j in 0..32 {
            v[j] = gather_col(token, input, base, j, 32);
        }
        crate::dct32::dct1d_32_batch(token, &mut v);
        for j in 0..32 {
            v[j] *= inv32;
        }
        for j in 0..32 {
            scatter_col(v[j], &mut tmp, base, j, 32);
        }
    }

    // Transpose 64×32 → 32×64
    let mut transposed = crate::scratch_buf::<2048>();
    for r in 0..64 {
        for c in 0..32 {
            transposed[c * 64 + r] = tmp[r * 32 + c];
        }
    }

    // Pass 2: DCT-64 on 32 rows (stride 64), 4 batches of 8
    for batch in 0..4 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 64];
        for j in 0..64 {
            v[j] = gather_col(token, &transposed, base, j, 64);
        }
        dct1d_64_batch(token, &mut v);
        for j in 0..64 {
            v[j] *= inv64;
        }
        for j in 0..64 {
            scatter_col(v[j], output, base, j, 64);
        }
    }
}

/// AVX2 32×64 forward DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn dct_32x64_avx2(token: archmage::X64V3Token, input: &[f32; 2048], output: &mut [f32; 2048]) {
    use magetypes::simd::f32x8;

    let inv32 = f32x8::splat(token, 1.0 / 32.0);
    let inv64 = f32x8::splat(token, 1.0 / 64.0);
    let mut tmp = crate::scratch_buf::<2048>();

    // Pass 1: DCT-64 on 32 rows (stride 64), 4 batches of 8
    for batch in 0..4 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 64];
        for j in 0..64 {
            v[j] = gather_col(token, input, base, j, 64);
        }
        dct1d_64_batch(token, &mut v);
        for j in 0..64 {
            v[j] *= inv64;
        }
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

    // Pass 2: DCT-32 on 64 rows (stride 32), 8 batches of 8
    for batch in 0..8 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 32];
        for j in 0..32 {
            v[j] = gather_col(token, &transposed, base, j, 32);
        }
        crate::dct32::dct1d_32_batch(token, &mut v);
        for j in 0..32 {
            v[j] *= inv32;
        }
        for j in 0..32 {
            scatter_col(v[j], &mut transposed, base, j, 32);
        }
    }

    // Final transpose 64×32 → 32×64 (ROWS < COLS branch)
    for r in 0..64 {
        for c in 0..32 {
            output[c * 64 + r] = transposed[r * 32 + c];
        }
    }
}

// ============================================================================
// Dispatchers
// ============================================================================

/// Compute 64×64 forward DCT with SIMD acceleration.
///
/// Input: 4096 f32 in row-major order (spatial domain).
/// Output: 4096 f32 DCT coefficients.
/// No final transpose for square blocks.
#[inline]
pub fn dct_64x64(input: &[f32; 4096], output: &mut [f32; 4096]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_64x64_avx2(token, input, output);
            return;
        }
    }
    dct_64x64_scalar(input, output);
}

/// Compute 64×32 forward DCT with SIMD acceleration.
///
/// Input: 2048 f32 in 64×32 row-major order (stride 32).
/// Output: 2048 f32 DCT coefficients in 32×64 layout (stride 64).
#[inline]
pub fn dct_64x32(input: &[f32; 2048], output: &mut [f32; 2048]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_64x32_avx2(token, input, output);
            return;
        }
    }
    dct_64x32_scalar(input, output);
}

/// Compute 32×64 forward DCT with SIMD acceleration.
///
/// Input: 2048 f32 in 32×64 row-major order (stride 64).
/// Output: 2048 f32 DCT coefficients in 32×64 layout (stride 64).
#[inline]
pub fn dct_32x64(input: &[f32; 2048], output: &mut [f32; 2048]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_32x64_avx2(token, input, output);
            return;
        }
    }
    dct_32x64_scalar(input, output);
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
                // DCT64 accumulates more FP error than DCT32 due to deeper recursion
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
    fn test_dct_64x64_simd_matches_scalar() {
        let mut input = [0.0f32; 4096];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.31 + 1.7).cos() * 100.0;
        }
        assert_simd_matches_scalar_4096(dct_64x64_scalar, dct_64x64, &input, "DCT64x64 cos");
    }

    #[test]
    fn test_dct_64x64_dc_only() {
        let mut input = [0.0f32; 4096];
        input[0] = 128.0;
        assert_simd_matches_scalar_4096(dct_64x64_scalar, dct_64x64, &input, "DCT64x64 DC");
    }

    #[test]
    fn test_dct_64x64_sequential() {
        let mut input = [0.0f32; 4096];
        for (i, val) in input.iter_mut().enumerate() {
            *val = i as f32;
        }
        assert_simd_matches_scalar_4096(dct_64x64_scalar, dct_64x64, &input, "DCT64x64 seq");
    }

    #[test]
    fn test_dct_64x32_simd_matches_scalar() {
        let mut input = [0.0f32; 2048];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.43 + 2.1).cos() * 80.0;
        }
        assert_simd_matches_scalar_2048(dct_64x32_scalar, dct_64x32, &input, "DCT64x32");
    }

    #[test]
    fn test_dct_32x64_simd_matches_scalar() {
        let mut input = [0.0f32; 2048];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.29 + 0.7).sin() * 120.0;
        }
        assert_simd_matches_scalar_2048(dct_32x64_scalar, dct_32x64, &input, "DCT32x64");
    }

    /// Verify the 1D scalar chain is correct by checking energy conservation.
    /// Parseval: sum(X²)*N = sum(x²)/N for our 1/N scaled DCT.
    #[test]
    fn test_dct1d_64_energy() {
        let mut input = [0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.7 + 0.3).sin() * 50.0;
        }
        let input_energy: f64 = input.iter().map(|x| (*x as f64) * (*x as f64)).sum();

        let mut output = input;
        dct1d_64_scalar(&mut output);
        // Scale by 1/64 as the 2D functions do
        for v in output.iter_mut() {
            *v *= 1.0 / 64.0;
        }

        let output_energy: f64 = output.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        let ratio = output_energy / input_energy;
        // Should be close to 1/64 = 0.015625
        assert!(
            (ratio - 1.0 / 64.0).abs() < 0.001,
            "Energy ratio {ratio:.6} far from 1/64 = 0.015625"
        );
    }
}
