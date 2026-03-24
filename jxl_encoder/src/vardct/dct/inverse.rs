// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Inverse DCT transforms for all sizes.

use super::constants::*;
use crate::vardct::common::{as_array_mut, as_array_ref};

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

/// Value-returning 1D IDCT for N=4. Returns [out0, out1, out2, out3].
/// Parameters: (even0, even1, odd0, odd1) — caller de-interleaves before calling.
/// Uses reciprocal multiplications instead of divisions.
#[inline(always)]
fn idct1d_4_val(a: f32, b: f32, c: f32, d: f32) -> [f32; 4] {
    // a=even[0], b=even[1], c=odd[0], d=odd[1]
    // Reverse B transform on odd half
    let odd0 = (c - d) * INV_SQRT2;
    // Reverse idct1d_2 on odd half
    let o0_pre = (odd0 + d) * 0.5;
    let o1_pre = (odd0 - d) * 0.5;
    // Reverse WcMultipliers (multiply by reciprocal instead of divide)
    let o0 = o0_pre * INV_WC_MULTIPLIERS_4[0];
    let o1 = o1_pre * INV_WC_MULTIPLIERS_4[1];
    // Reverse idct1d_2 on even half
    let e0 = (a + b) * 0.5;
    let e1 = (a - b) * 0.5;
    // Combine even/odd
    [
        (e0 + o0) * 0.5,
        (e1 + o1) * 0.5,
        (e1 - o1) * 0.5,
        (e0 - o0) * 0.5,
    ]
}

/// Fast 1D IDCT for N=4 (exactly reverses dct1d_4).
#[inline(always)]
pub fn idct1d_4(mem: &mut [f32]) {
    let r = idct1d_4_val(mem[0], mem[2], mem[1], mem[3]);
    mem[..4].copy_from_slice(&r);
}

/// Value-returning core 1D IDCT for N=8 without the N scaling factor.
/// Uses reciprocal multiplications instead of divisions.
#[inline(always)]
fn idct1d_8_core_val(m: [f32; 8]) -> [f32; 8] {
    // De-interleave: even = [m[0], m[2], m[4], m[6]], odd = [m[1], m[3], m[5], m[7]]
    let (e0, e1, e2, e3) = (m[0], m[2], m[4], m[6]);
    let (mut o0, mut o1, mut o2, o3) = (m[1], m[3], m[5], m[7]);

    // Reverse B transform
    o2 -= o3;
    o1 -= o2;
    o0 = (o0 - o1) * INV_SQRT2;

    // Reverse idct1d_4 on odd half, then multiply by reciprocal WcMultipliers
    let odd = idct1d_4_val(o0, o2, o1, o3);
    let o = [
        odd[0] * INV_WC_MULTIPLIERS_8[0],
        odd[1] * INV_WC_MULTIPLIERS_8[1],
        odd[2] * INV_WC_MULTIPLIERS_8[2],
        odd[3] * INV_WC_MULTIPLIERS_8[3],
    ];

    // Reverse idct1d_4 on even half
    let e = idct1d_4_val(e0, e2, e1, e3);

    // Combine even/odd
    [
        (e[0] + o[0]) * 0.5,
        (e[1] + o[1]) * 0.5,
        (e[2] + o[2]) * 0.5,
        (e[3] + o[3]) * 0.5,
        (e[3] - o[3]) * 0.5,
        (e[2] - o[2]) * 0.5,
        (e[1] - o[1]) * 0.5,
        (e[0] - o[0]) * 0.5,
    ]
}

/// Core 1D IDCT for N=8 without the N scaling factor.
///
/// This reverses the dct1d_8 butterfly operations only, without compensating
/// for the 1/N scaling applied by the 2D wrapper (dct_8x8). Used internally
/// by idct1d_16 which applies its own scaling.
fn idct1d_8_core(mem: &mut [f32]) {
    let mut m = [0.0f32; 8];
    m.copy_from_slice(&mem[..8]);
    let r = idct1d_8_core_val(m);
    mem[..8].copy_from_slice(&r);
}

/// Fast 1D IDCT for N=8 (exactly reverses dct1d_8).
///
/// Includes the *= 8 scaling to compensate for the 1/8 applied by dct_8x8.
pub fn idct1d_8(mem: &mut [f32]) {
    let m = [
        mem[0] * 8.0,
        mem[1] * 8.0,
        mem[2] * 8.0,
        mem[3] * 8.0,
        mem[4] * 8.0,
        mem[5] * 8.0,
        mem[6] * 8.0,
        mem[7] * 8.0,
    ];
    let r = idct1d_8_core_val(m);
    mem[..8].copy_from_slice(&r);
}

/// Fast 1D IDCT for N=16 (exactly reverses dct1d_16).
///
/// Includes *= 16 scaling to compensate for the 1/16 applied by dct_16x16.
/// Uses idct1d_8_core_val (without 8x scaling) for the recursive sub-transforms
/// since the scaling is handled at this level.
pub fn idct1d_16(mem: &mut [f32]) {
    // De-interleave with scaling: even = scaled[0,2,4,...], odd = scaled[1,3,5,...]
    let mut even = [0.0f32; 8];
    let mut odd = [0.0f32; 8];
    for i in 0..8 {
        even[i] = mem[2 * i] * 16.0;
        odd[i] = mem[2 * i + 1] * 16.0;
    }

    // Reverse B transform
    for i in (1..7).rev() {
        odd[i] -= odd[i + 1];
    }
    odd[0] = (odd[0] - odd[1]) * INV_SQRT2;

    // Reverse idct on odd half, then multiply by reciprocal WcMultipliers
    let odd_r = idct1d_8_core_val(odd);
    let mut odd_scaled = [0.0f32; 8];
    for i in 0..8 {
        odd_scaled[i] = odd_r[i] * INV_WC_MULTIPLIERS_16[i];
    }

    // Reverse idct on even half
    let even_r = idct1d_8_core_val(even);

    // Combine
    for i in 0..8 {
        mem[i] = (even_r[i] + odd_scaled[i]) * 0.5;
        mem[15 - i] = (even_r[i] - odd_scaled[i]) * 0.5;
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
#[inline]
pub fn idct_8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::idct_8x8(input, output);
}

/// Compute 16x16 inverse DCT (exactly reverses dct_16x16).
#[inline]
pub fn idct_16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    jxl_simd::idct_16x16(input, output);
}

/// Compute 16x8 inverse DCT (16 rows x 8 cols, exactly reverses dct_16x8).
#[inline]
pub fn idct_16x8(input: &[f32; 128], output: &mut [f32; 128]) {
    jxl_simd::idct_16x8(input, output);
}

/// Compute 8x16 inverse DCT (8 rows x 16 cols, exactly reverses dct_8x16).
#[inline]
pub fn idct_8x16(input: &[f32; 128], output: &mut [f32; 128]) {
    jxl_simd::idct_8x16(input, output);
}

/// Compute 4x4 inverse DCT (exactly reverses dct_4x4).
/// Input layout: 4 rows x 4 cols, stride 4.
#[inline(always)]
pub fn idct_4x4(input: &[f32; 16], output: &mut [f32; 16]) {
    // Pass 1: 4-point IDCT on each row (scaled by 4), store transposed
    let mut temp = [0.0f32; 16];
    for row in 0..4 {
        let s = row * 4;
        let r = idct1d_4_val(
            input[s] * 4.0,
            input[s + 2] * 4.0,
            input[s + 1] * 4.0,
            input[s + 3] * 4.0,
        );
        // Store transposed: row → column
        for col in 0..4 {
            temp[col * 4 + row] = r[col];
        }
    }

    // Pass 2: 4-point IDCT on each row of transposed (scaled by 4), write to output
    for row in 0..4 {
        let s = row * 4;
        let r = idct1d_4_val(
            temp[s] * 4.0,
            temp[s + 2] * 4.0,
            temp[s + 1] * 4.0,
            temp[s + 3] * 4.0,
        );
        output[s..s + 4].copy_from_slice(&r);
    }
}

/// Compute 4x8 inverse DCT (exactly reverses dct_4x8).
/// Fused: transpose → 4pt IDCT → transpose → 8pt IDCT, using value-returning butterflies.
#[inline(always)]
pub fn idct_4x8(input: &[f32; 32], output: &mut [f32; 32]) {
    // Pass 1: Transpose 4x8 → 8x4, then 4pt IDCT (scaled by 4) on each of 8 rows,
    // store transposed back to 4x8 layout.
    let mut temp = [0.0f32; 32];
    for col in 0..8 {
        // Gather column from 4x8 input (becomes row in transposed)
        let a = input[col] * 4.0;
        let b = input[8 + col] * 4.0;
        let c = input[16 + col] * 4.0;
        let d = input[24 + col] * 4.0;
        // idct1d_4_val expects de-interleaved: [even0, even1, odd0, odd1] = [a, c, b, d]
        let r = idct1d_4_val(a, c, b, d);
        // Store transposed: row col → col*8 + row index in 4x8 layout
        for row in 0..4 {
            temp[row * 8 + col] = r[row];
        }
    }

    // Pass 2: 8pt IDCT (scaled by 8) on each of 4 rows
    for row in 0..4 {
        let s = row * 8;
        let m = [
            temp[s] * 8.0,
            temp[s + 1] * 8.0,
            temp[s + 2] * 8.0,
            temp[s + 3] * 8.0,
            temp[s + 4] * 8.0,
            temp[s + 5] * 8.0,
            temp[s + 6] * 8.0,
            temp[s + 7] * 8.0,
        ];
        let r = idct1d_8_core_val(m);
        output[s..s + 8].copy_from_slice(&r);
    }
}

/// Compute 8x4 inverse DCT (exactly reverses dct_8x4).
/// Fused: 8pt IDCT → transpose → 4pt IDCT, using value-returning butterflies.
#[inline(always)]
pub fn idct_8x4(input: &[f32; 32], output: &mut [f32; 32]) {
    // Pass 1: 8pt IDCT (scaled by 8) on each of 4 rows, store transposed to 8x4 layout
    let mut temp = [0.0f32; 32];
    for row in 0..4 {
        let s = row * 8;
        let m = [
            input[s] * 8.0,
            input[s + 1] * 8.0,
            input[s + 2] * 8.0,
            input[s + 3] * 8.0,
            input[s + 4] * 8.0,
            input[s + 5] * 8.0,
            input[s + 6] * 8.0,
            input[s + 7] * 8.0,
        ];
        let r = idct1d_8_core_val(m);
        // Store transposed: 4x8 row → 8x4 column
        for col in 0..8 {
            temp[col * 4 + row] = r[col];
        }
    }

    // Pass 2: 4pt IDCT (scaled by 4) on each of 8 rows
    for row in 0..8 {
        let s = row * 4;
        let r = idct1d_4_val(
            temp[s] * 4.0,
            temp[s + 2] * 4.0,
            temp[s + 1] * 4.0,
            temp[s + 3] * 4.0,
        );
        output[s..s + 4].copy_from_slice(&r);
    }
}

/// Core 1D IDCT for N=16 without the N scaling factor.
/// Used internally by idct1d_32 which applies its own scaling.
fn idct1d_16_core(mem: &mut [f32]) {
    // De-interleave
    let mut even = [0.0f32; 8];
    let mut odd = [0.0f32; 8];
    for i in 0..8 {
        even[i] = mem[2 * i];
        odd[i] = mem[2 * i + 1];
    }

    // Reverse B transform
    for i in (1..7).rev() {
        odd[i] -= odd[i + 1];
    }
    odd[0] = (odd[0] - odd[1]) * INV_SQRT2;

    // Reverse idct on odd half, then multiply by reciprocal WcMultipliers
    let odd_r = idct1d_8_core_val(odd);
    let mut odd_scaled = [0.0f32; 8];
    for i in 0..8 {
        odd_scaled[i] = odd_r[i] * INV_WC_MULTIPLIERS_16[i];
    }

    // Reverse idct on even half
    let even_r = idct1d_8_core_val(even);

    // Combine
    for i in 0..8 {
        mem[i] = (even_r[i] + odd_scaled[i]) * 0.5;
        mem[15 - i] = (even_r[i] - odd_scaled[i]) * 0.5;
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
    let mut even = [0.0f32; 16];
    let mut odd = [0.0f32; 16];
    for i in 0..16 {
        even[i] = mem[2 * i];
        odd[i] = mem[2 * i + 1];
    }

    // Reverse B transform
    for i in (1..15).rev() {
        odd[i] -= odd[i + 1];
    }
    odd[0] = (odd[0] - odd[1]) * INV_SQRT2;

    // IDCT on odd half, then multiply by reciprocal WcMultipliers
    idct1d_16_core(&mut odd);
    for i in 0..16 {
        odd[i] *= INV_WC_MULTIPLIERS_32[i];
    }

    // IDCT on even half
    idct1d_16_core(&mut even);

    // Combine
    for i in 0..16 {
        mem[i] = (even[i] + odd[i]) * 0.5;
        mem[31 - i] = (even[i] - odd[i]) * 0.5;
    }
}

/// Compute 32x32 inverse DCT (exactly reverses dct_32x32).
pub fn idct_32x32(input: &[f32; 1024], output: &mut [f32; 1024]) {
    jxl_simd::idct_32x32(input, output);
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
    jxl_simd::idct_32x16(input, output);
}

/// Compute 16x32 inverse DCT (exactly reverses dct_16x32).
///
/// dct_16x32 (ROWS=16 < COLS=32, WITH final transpose):
///   1. 32pt DCT on rows, *= 1/32
///   2. Transpose 16x32 -> 32x16
///   3. 16pt DCT on rows, *= 1/16
///   4. Transpose 32x16 -> 16x32
pub fn idct_16x32(input: &[f32; 512], output: &mut [f32; 512]) {
    jxl_simd::idct_16x32(input, output);
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
    tmp[32] = (tmp[32] - tmp[33]) * INV_SQRT2;

    // IDCT on second half
    idct1d_32_core(&mut tmp[32..64]);

    // Multiply by reciprocal WcMultipliers
    for i in 0..32 {
        tmp[32 + i] *= INV_WC_MULTIPLIERS_64[i];
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

    jxl_simd::idct_64x64(as_array_ref(input, 0), as_array_mut(output, 0));
}

/// Compute 64x32 inverse DCT (exactly reverses dct_64x32).
///
/// dct_64x32 (ROWS=64 >= COLS=32, no final transpose):
///   Output: 32x64 (stride 64).
pub fn idct_64x32(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 2048);
    debug_assert!(output.len() >= 2048);

    jxl_simd::idct_64x32(as_array_ref(input, 0), as_array_mut(output, 0));
}

/// Compute 32x64 inverse DCT (exactly reverses dct_32x64).
///
/// dct_32x64 (ROWS=32 < COLS=64, WITH final transpose).
pub fn idct_32x64(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 2048);
    debug_assert!(output.len() >= 2048);

    jxl_simd::idct_32x64(as_array_ref(input, 0), as_array_mut(output, 0));
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
/// For DCT8, DC is just the `[0,0]` coefficient.
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

/// Compute full inverse DCT4X8 transform for 8x8 coefficient block.
///
/// This exactly reverses `dct_4x8_full`.
///
/// Input: 64 DCT coefficients in interleaved layout
/// Output: 8x8 = 64 floats in row-major pixel order (stride 8)
#[inline(always)]
pub fn idct_4x8_full(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::idct_4x8_full(input, output);
}

/// Compute full inverse DCT8X4 transform for 8x8 coefficient block.
///
/// This exactly reverses `dct_8x4_full`.
///
/// Input: 64 DCT coefficients in interleaved layout
/// Output: 8x8 = 64 floats in row-major pixel order (stride 8)
#[inline(always)]
pub fn idct_8x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::idct_8x4_full(input, output);
}

/// Compute full inverse DCT4X4 transform for 8x8 coefficient block.
///
/// This exactly reverses `dct_4x4_full`.
///
/// Input: 64 DCT coefficients in interleaved layout
/// Output: 8x8 = 64 floats in row-major pixel order (stride 8)
#[inline(always)]
pub fn idct_4x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_simd::idct_4x4_full(input, output);
}
