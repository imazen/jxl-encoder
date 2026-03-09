// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated 32×32/32×16/16×32 inverse DCT.
//!
//! Processes 8 independent 32-point IDCTs in parallel using AVX2 f32x8 vectors.
//! The 32-point batch IDCT recursively calls the 16-point core IDCT from `idct16.rs`.

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

// Pre-computed reciprocals to replace division with multiplication.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::excessive_precision)]
const INV_WC32: [f32; 16] = [
    1.0 / 0.5006029982351963,
    1.0 / 0.5054709598975436,
    1.0 / 0.5154473099226246,
    1.0 / 0.5310425910897841,
    1.0 / 0.5531038960344445,
    1.0 / 0.5829349682061339,
    1.0 / 0.6225041230356648,
    1.0 / 0.6748083414550057,
    1.0 / 0.7445362710022986,
    1.0 / 0.8393496454155268,
    1.0 / 0.9725682378619608,
    1.0 / 1.1694399334328847,
    1.0 / 1.4841646163141662,
    1.0 / 2.057781009953411,
    1.0 / 3.407608418468719,
    1.0 / 10.190008123548033,
];

// ============================================================================
// Scalar fallback — self-contained 32-point IDCT chain
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

/// 16-point IDCT with *= 16 scaling.
fn idct1d_16_scalar(mem: &mut [f32]) {
    for x in mem.iter_mut().take(16) {
        *x *= 16.0;
    }
    idct1d_16_core_scalar(mem);
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

/// Scalar 32×32 inverse DCT.
#[inline]
pub fn idct_32x32_scalar(input: &[f32; 1024], output: &mut [f32; 1024]) {
    let mut tmp = crate::scratch_buf::<1024>();

    // IDCT on each row
    for row in 0..32 {
        let s = row * 32;
        tmp[s..s + 32].copy_from_slice(&input[s..s + 32]);
        idct1d_32_scalar(&mut tmp[s..s + 32]);
    }

    // Transpose 32×32
    let mut transposed = crate::scratch_buf::<1024>();
    for r in 0..32 {
        for c in 0..32 {
            transposed[c * 32 + r] = tmp[r * 32 + c];
        }
    }

    // IDCT on each column (now rows of transposed)
    for row in 0..32 {
        let s = row * 32;
        output[s..s + 32].copy_from_slice(&transposed[s..s + 32]);
        idct1d_32_scalar(&mut output[s..s + 32]);
    }
}

/// Scalar 32×16 inverse DCT.
///
/// Reverses dct_32x16: input in 16×32 layout (stride 32).
/// Output in 32×16 layout (stride 16, spatial domain).
#[inline]
pub fn idct_32x16_scalar(input: &[f32; 512], output: &mut [f32; 512]) {
    let mut tmp = crate::scratch_buf::<512>();

    // IDCT-32 on each of 16 rows (stride 32)
    for row in 0..16 {
        let s = row * 32;
        tmp[s..s + 32].copy_from_slice(&input[s..s + 32]);
        idct1d_32_scalar(&mut tmp[s..s + 32]);
    }

    // Transpose 16×32 → 32×16
    let mut transposed = crate::scratch_buf::<512>();
    for r in 0..16 {
        for c in 0..32 {
            transposed[c * 16 + r] = tmp[r * 32 + c];
        }
    }

    // IDCT-16 on each of 32 rows (stride 16)
    for row in 0..32 {
        let s = row * 16;
        output[s..s + 16].copy_from_slice(&transposed[s..s + 16]);
        idct1d_16_scalar(&mut output[s..s + 16]);
    }
}

/// Scalar 16×32 inverse DCT.
///
/// Reverses dct_16x32: input in 16×32 layout (stride 32).
/// Output in 16×32 layout (stride 32, spatial domain).
#[inline]
pub fn idct_16x32_scalar(input: &[f32; 512], output: &mut [f32; 512]) {
    // Un-transpose: 16×32 → 32×16
    let mut transposed = crate::scratch_buf::<512>();
    for r in 0..16 {
        for c in 0..32 {
            transposed[c * 16 + r] = input[r * 32 + c];
        }
    }

    // IDCT-16 on each of 32 rows (stride 16)
    let mut tmp = crate::scratch_buf::<512>();
    for row in 0..32 {
        let s = row * 16;
        tmp[s..s + 16].copy_from_slice(&transposed[s..s + 16]);
        idct1d_16_scalar(&mut tmp[s..s + 16]);
    }

    // Transpose 32×16 → 16×32
    let mut transposed2 = crate::scratch_buf::<512>();
    for r in 0..32 {
        for c in 0..16 {
            transposed2[c * 32 + r] = tmp[r * 16 + c];
        }
    }

    // IDCT-32 on each of 16 rows (stride 32)
    for row in 0..16 {
        let s = row * 32;
        output[s..s + 32].copy_from_slice(&transposed2[s..s + 32]);
        idct1d_32_scalar(&mut output[s..s + 32]);
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

/// AVX2 batched 32-point core inverse DCT WITHOUT scaling.
///
/// `v[0..32]` holds positions 0-31 across 8 independent 1D transforms.
/// Recursively calls `idct1d_16_core_batch` from `idct16.rs` for the two halves.
/// Does NOT apply the *= 32 scaling factor — use `idct1d_32_batch` for the scaled version.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
pub(crate) fn idct1d_32_core_batch(
    token: archmage::X64V3Token,
    v: &mut [magetypes::simd::f32x8; 32],
) {
    use magetypes::simd::f32x8;

    let half = f32x8::splat(token, 0.5);
    let inv_sqrt2 = f32x8::splat(token, 1.0 / SQRT2);

    // De-interleave: even → first_half, odd → second_half
    let mut first = [f32x8::zero(token); 16];
    let mut second = [f32x8::zero(token); 16];
    for i in 0..16 {
        first[i] = v[2 * i];
        second[i] = v[2 * i + 1];
    }

    // Reverse B transform on second half
    for i in (1..15).rev() {
        second[i] -= second[i + 1];
    }
    second[0] = (second[0] - second[1]) * inv_sqrt2;

    // IDCT-16 core on second half (no scaling)
    crate::idct16::idct1d_16_core_batch(token, &mut second);

    // Divide by WcMultipliers_32 (multiply by inverse)
    for i in 0..16 {
        second[i] *= f32x8::splat(token, INV_WC32[i]);
    }

    // IDCT-16 core on first half (no scaling)
    crate::idct16::idct1d_16_core_batch(token, &mut first);

    // Combine
    for i in 0..16 {
        v[i] = (first[i] + second[i]) * half;
        v[31 - i] = (first[i] - second[i]) * half;
    }
}

/// AVX2 batched 32-point inverse DCT with *= 32 scaling.
///
/// `v[0..32]` holds positions 0-31 across 8 independent 1D transforms.
/// Applies *= 32 scaling then delegates to `idct1d_32_core_batch`.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(always)]
pub(crate) fn idct1d_32_batch(token: archmage::X64V3Token, v: &mut [magetypes::simd::f32x8; 32]) {
    use magetypes::simd::f32x8;

    let scale32 = f32x8::splat(token, 32.0);

    // Scale by 32
    for vi in v.iter_mut() {
        *vi *= scale32;
    }

    idct1d_32_core_batch(token, v);
}

/// AVX2 32×32 inverse DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_32x32_avx2(token: archmage::X64V3Token, input: &[f32; 1024], output: &mut [f32; 1024]) {
    use magetypes::simd::f32x8;

    let mut tmp = crate::scratch_buf::<1024>();

    // Pass 1: IDCT-32 on rows, 4 batches of 8 rows
    for batch in 0..4 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 32];
        for j in 0..32 {
            v[j] = gather_col(token, input, base, j, 32);
        }
        idct1d_32_batch(token, &mut v);
        for j in 0..32 {
            scatter_col(v[j], &mut tmp, base, j, 32);
        }
    }

    // Transpose 32×32
    let mut transposed = crate::scratch_buf::<1024>();
    for r in 0..32 {
        for c in 0..32 {
            transposed[c * 32 + r] = tmp[r * 32 + c];
        }
    }

    // Pass 2: IDCT-32 on columns (now rows), 4 batches of 8 rows
    for batch in 0..4 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 32];
        for j in 0..32 {
            v[j] = gather_col(token, &transposed, base, j, 32);
        }
        idct1d_32_batch(token, &mut v);
        for j in 0..32 {
            scatter_col(v[j], output, base, j, 32);
        }
    }
}

/// AVX2 32×16 inverse DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_32x16_avx2(token: archmage::X64V3Token, input: &[f32; 512], output: &mut [f32; 512]) {
    use magetypes::simd::f32x8;

    let mut tmp = crate::scratch_buf::<512>();

    // Pass 1: IDCT-32 on 16 rows (stride 32), 2 batches of 8
    for batch in 0..2 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 32];
        for j in 0..32 {
            v[j] = gather_col(token, input, base, j, 32);
        }
        idct1d_32_batch(token, &mut v);
        for j in 0..32 {
            scatter_col(v[j], &mut tmp, base, j, 32);
        }
    }

    // Transpose 16×32 → 32×16
    let mut transposed = crate::scratch_buf::<512>();
    for r in 0..16 {
        for c in 0..32 {
            transposed[c * 16 + r] = tmp[r * 32 + c];
        }
    }

    // Pass 2: IDCT-16 on 32 rows (stride 16), 4 batches of 8
    for batch in 0..4 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, &transposed, base, j, 16);
        }
        crate::idct16::idct1d_16_batch(token, &mut v);
        for j in 0..16 {
            scatter_col(v[j], output, base, j, 16);
        }
    }
}

/// AVX2 16×32 inverse DCT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::needless_range_loop)]
pub fn idct_16x32_avx2(token: archmage::X64V3Token, input: &[f32; 512], output: &mut [f32; 512]) {
    use magetypes::simd::f32x8;

    // Un-transpose: 16×32 → 32×16
    let mut transposed = crate::scratch_buf::<512>();
    for r in 0..16 {
        for c in 0..32 {
            transposed[c * 16 + r] = input[r * 32 + c];
        }
    }

    // Pass 1: IDCT-16 on 32 rows (stride 16), 4 batches of 8
    let mut tmp = crate::scratch_buf::<512>();
    for batch in 0..4 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 16];
        for j in 0..16 {
            v[j] = gather_col(token, &transposed, base, j, 16);
        }
        crate::idct16::idct1d_16_batch(token, &mut v);
        for j in 0..16 {
            scatter_col(v[j], &mut tmp, base, j, 16);
        }
    }

    // Transpose 32×16 → 16×32
    let mut transposed2 = crate::scratch_buf::<512>();
    for r in 0..32 {
        for c in 0..16 {
            transposed2[c * 32 + r] = tmp[r * 16 + c];
        }
    }

    // Pass 2: IDCT-32 on 16 rows (stride 32), 2 batches of 8
    for batch in 0..2 {
        let base = batch * 8;
        let mut v = [f32x8::zero(token); 32];
        for j in 0..32 {
            v[j] = gather_col(token, &transposed2, base, j, 32);
        }
        idct1d_32_batch(token, &mut v);
        for j in 0..32 {
            scatter_col(v[j], output, base, j, 32);
        }
    }
}

// ============================================================================
// Dispatchers
// ============================================================================

/// Compute 32×32 inverse DCT with SIMD acceleration.
///
/// Input: 1024 f32 DCT coefficients.
/// Output: 1024 f32 in row-major order (spatial domain).
#[inline]
pub fn idct_32x32(input: &[f32; 1024], output: &mut [f32; 1024]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_32x32_avx2(token, input, output);
            return;
        }
    }
    idct_32x32_scalar(input, output);
}

/// Compute 32×16 inverse DCT with SIMD acceleration.
///
/// Input: 512 f32 DCT coefficients in 16×32 layout (stride 32).
/// Output: 512 f32 in 32×16 row-major order (stride 16, spatial domain).
#[inline]
pub fn idct_32x16(input: &[f32; 512], output: &mut [f32; 512]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_32x16_avx2(token, input, output);
            return;
        }
    }
    idct_32x16_scalar(input, output);
}

/// Compute 16×32 inverse DCT with SIMD acceleration.
///
/// Input: 512 f32 DCT coefficients in 16×32 layout (stride 32).
/// Output: 512 f32 in 16×32 row-major order (stride 32, spatial domain).
#[inline]
pub fn idct_16x32(input: &[f32; 512], output: &mut [f32; 512]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_16x32_avx2(token, input, output);
            return;
        }
    }
    idct_16x32_scalar(input, output);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    fn assert_simd_matches_scalar_1024(
        scalar_fn: fn(&[f32; 1024], &mut [f32; 1024]),
        dispatch_fn: fn(&[f32; 1024], &mut [f32; 1024]),
        input: &[f32; 1024],
        label: &str,
    ) {
        let mut scalar_out = [0.0f32; 1024];
        scalar_fn(input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 1024];
                dispatch_fn(input, &mut simd_out);
                let mut max_diff = 0.0f32;
                let mut max_idx = 0;
                for i in 0..1024 {
                    let diff = (scalar_out[i] - simd_out[i]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = i;
                    }
                }
                // Relative tolerance: 1e-5 of max magnitude, floor 1e-2
                let max_mag = scalar_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let tol = (max_mag * 1e-5).max(1e-2);
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

    fn assert_simd_matches_scalar_512(
        scalar_fn: fn(&[f32; 512], &mut [f32; 512]),
        dispatch_fn: fn(&[f32; 512], &mut [f32; 512]),
        input: &[f32; 512],
        label: &str,
    ) {
        let mut scalar_out = [0.0f32; 512];
        scalar_fn(input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 512];
                dispatch_fn(input, &mut simd_out);
                let mut max_diff = 0.0f32;
                let mut max_idx = 0;
                for i in 0..512 {
                    let diff = (scalar_out[i] - simd_out[i]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = i;
                    }
                }
                let max_mag = scalar_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let tol = (max_mag * 1e-5).max(1e-2);
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
    fn test_idct_32x32_simd_matches_scalar() {
        let mut input = [0.0f32; 1024];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.31 + 1.7).cos() * 100.0;
        }
        assert_simd_matches_scalar_1024(idct_32x32_scalar, idct_32x32, &input, "IDCT32x32 cos");
    }

    #[test]
    fn test_idct_32x32_dc_only() {
        let mut input = [0.0f32; 1024];
        input[0] = 128.0;
        assert_simd_matches_scalar_1024(idct_32x32_scalar, idct_32x32, &input, "IDCT32x32 DC");
    }

    #[test]
    fn test_idct_32x32_sequential() {
        let mut input = [0.0f32; 1024];
        for (i, val) in input.iter_mut().enumerate() {
            *val = i as f32;
        }
        assert_simd_matches_scalar_1024(idct_32x32_scalar, idct_32x32, &input, "IDCT32x32 seq");
    }

    #[test]
    fn test_idct_32x16_simd_matches_scalar() {
        let mut input = [0.0f32; 512];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.43 + 2.1).cos() * 80.0;
        }
        assert_simd_matches_scalar_512(idct_32x16_scalar, idct_32x16, &input, "IDCT32x16");
    }

    #[test]
    fn test_idct_16x32_simd_matches_scalar() {
        let mut input = [0.0f32; 512];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.29 + 0.7).sin() * 120.0;
        }
        assert_simd_matches_scalar_512(idct_16x32_scalar, idct_16x32, &input, "IDCT16x32");
    }

    /// Verify DCT32 → IDCT32 roundtrip is near-identity.
    #[test]
    fn test_dct32_idct32_roundtrip() {
        let mut input = [0.0f32; 1024];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.17 + 3.2).sin() * 50.0;
        }

        let mut coeffs = [0.0f32; 1024];
        crate::dct32::dct_32x32(&input, &mut coeffs);

        let mut output = [0.0f32; 1024];
        idct_32x32(&coeffs, &mut output);

        let mut max_diff = 0.0f32;
        for i in 0..1024 {
            let diff = (input[i] - output[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 0.1,
            "DCT32→IDCT32 roundtrip max error = {max_diff}"
        );
    }

    /// Verify DCT32x16 → IDCT32x16 roundtrip.
    #[test]
    fn test_dct32x16_idct32x16_roundtrip() {
        let mut input = [0.0f32; 512];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.23 + 1.1).cos() * 60.0;
        }

        let mut coeffs = [0.0f32; 512];
        crate::dct32::dct_32x16(&input, &mut coeffs);

        let mut output = [0.0f32; 512];
        idct_32x16(&coeffs, &mut output);

        let mut max_diff = 0.0f32;
        for i in 0..512 {
            let diff = (input[i] - output[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 0.1,
            "DCT32x16→IDCT32x16 roundtrip max error = {max_diff}"
        );
    }

    /// Verify DCT16x32 → IDCT16x32 roundtrip.
    #[test]
    fn test_dct16x32_idct16x32_roundtrip() {
        let mut input = [0.0f32; 512];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.37 + 2.5).sin() * 40.0;
        }

        let mut coeffs = [0.0f32; 512];
        crate::dct32::dct_16x32(&input, &mut coeffs);

        let mut output = [0.0f32; 512];
        idct_16x32(&coeffs, &mut output);

        let mut max_diff = 0.0f32;
        for i in 0..512 {
            let diff = (input[i] - output[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 0.1,
            "DCT16x32→IDCT16x32 roundtrip max error = {max_diff}"
        );
    }
}
