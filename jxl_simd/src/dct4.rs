// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated DCT4-based full transforms (4x4, 4x8, 8x4) and their inverses.
//!
//! These transforms operate on 8x8 pixel blocks decomposed into sub-blocks.
//! The SIMD versions pack two sub-blocks per f32x8 register (one per 128-bit lane).

const SQRT2: f32 = core::f32::consts::SQRT_2;
const INV_SQRT2: f32 = 0.707_106_77; // 1/sqrt(2)
const WC_MULTIPLIERS_4: [f32; 2] = [0.541_196_1, 1.306_563];
const INV_WC_MULTIPLIERS_4: [f32; 2] = [1.0 / 0.541_196_1, 1.0 / 1.306_563];
const WC_MULTIPLIERS_8: [f32; 4] = [0.509_795_6, 0.601_344_9, 0.899_976_2, 2.562_915_5];
const INV_WC_MULTIPLIERS_8: [f32; 4] = [
    1.0 / 0.509_795_6,
    1.0 / 0.601_344_9,
    1.0 / 0.899_976_2,
    1.0 / 2.562_915_5,
];

// =============================================================================
// Forward DCT4x4 full transform
// =============================================================================

/// Compute full DCT4X4 transform for 8x8 pixel block.
///
/// Covers an 8x8 pixel region using FOUR 4x4 sub-blocks in a 2x2 grid.
/// DCs of the four sub-blocks are combined with a 2x2 Hadamard.
#[inline]
pub fn dct_4x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_4x4_full_avx2(token, input, output);
            return;
        }
    }

    dct_4x4_full_scalar(input, output);
}

/// Compute full inverse DCT4X4 transform for 8x8 coefficient block.
#[inline]
pub fn idct_4x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_4x4_full_avx2(token, input, output);
            return;
        }
    }

    idct_4x4_full_scalar(input, output);
}

/// Compute full DCT4X8 transform for 8x8 pixel block.
///
/// Covers an 8x8 pixel region using TWO vertically-stacked 4x8 sub-blocks.
#[inline]
pub fn dct_4x8_full(input: &[f32; 64], output: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_4x8_full_avx2(token, input, output);
            return;
        }
    }

    dct_4x8_full_scalar(input, output);
}

/// Compute full inverse DCT4X8 transform for 8x8 coefficient block.
#[inline]
pub fn idct_4x8_full(input: &[f32; 64], output: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_4x8_full_avx2(token, input, output);
            return;
        }
    }

    idct_4x8_full_scalar(input, output);
}

/// Compute full DCT8X4 transform for 8x8 pixel block.
///
/// Covers an 8x8 pixel region using TWO horizontally-adjacent 8x4 sub-blocks.
#[inline]
pub fn dct_8x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct_8x4_full_avx2(token, input, output);
            return;
        }
    }

    dct_8x4_full_scalar(input, output);
}

/// Compute full inverse DCT8X4 transform for 8x8 coefficient block.
#[inline]
pub fn idct_8x4_full(input: &[f32; 64], output: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            idct_8x4_full_avx2(token, input, output);
            return;
        }
    }

    idct_8x4_full_scalar(input, output);
}

// =============================================================================
// Scalar implementations
// =============================================================================

/// Value-returning scalar 4-point DCT.
#[inline(always)]
fn dct1d_4_val(a: f32, b: f32, c: f32, d: f32) -> [f32; 4] {
    let t0 = a + d;
    let t1 = b + c;
    let t2 = a - d;
    let t3 = b - c;
    let u0 = t0 + t1;
    let u1 = t0 - t1;
    let v0 = t2 * WC_MULTIPLIERS_4[0];
    let v1 = t3 * WC_MULTIPLIERS_4[1];
    let w0 = v0 + v1;
    let w1 = v0 - v1;
    let b0 = SQRT2 * w0 + w1;
    [u0, b0, u1, w1]
}

/// Value-returning scalar 8-point DCT.
#[inline(always)]
fn dct1d_8_val(m: [f32; 8]) -> [f32; 8] {
    let t0 = m[0] + m[7];
    let t1 = m[1] + m[6];
    let t2 = m[2] + m[5];
    let t3 = m[3] + m[4];
    let t4 = m[0] - m[7];
    let t5 = m[1] - m[6];
    let t6 = m[2] - m[5];
    let t7 = m[3] - m[4];
    let r0 = dct1d_4_val(t0, t1, t2, t3);
    let w4 = t4 * WC_MULTIPLIERS_8[0];
    let w5 = t5 * WC_MULTIPLIERS_8[1];
    let w6 = t6 * WC_MULTIPLIERS_8[2];
    let w7 = t7 * WC_MULTIPLIERS_8[3];
    let r1 = dct1d_4_val(w4, w5, w6, w7);
    let b0 = SQRT2 * r1[0] + r1[1];
    let b1 = r1[1] + r1[2];
    let b2 = r1[2] + r1[3];
    let b3 = r1[3];
    [r0[0], b0, r0[1], b1, r0[2], b2, r0[3], b3]
}

/// Value-returning scalar 4-point IDCT.
/// Input: (even0, even1, odd0, odd1) — caller de-interleaves.
#[inline(always)]
fn idct1d_4_val(a: f32, b: f32, c: f32, d: f32) -> [f32; 4] {
    let odd0 = (c - d) * INV_SQRT2;
    let o0_pre = (odd0 + d) * 0.5;
    let o1_pre = (odd0 - d) * 0.5;
    let o0 = o0_pre * INV_WC_MULTIPLIERS_4[0];
    let o1 = o1_pre * INV_WC_MULTIPLIERS_4[1];
    let e0 = (a + b) * 0.5;
    let e1 = (a - b) * 0.5;
    [
        (e0 + o0) * 0.5,
        (e1 + o1) * 0.5,
        (e1 - o1) * 0.5,
        (e0 - o0) * 0.5,
    ]
}

/// Value-returning scalar 8-point IDCT core (no N scaling).
#[inline(always)]
fn idct1d_8_core_val(m: [f32; 8]) -> [f32; 8] {
    let (e0, e1, e2, e3) = (m[0], m[2], m[4], m[6]);
    let (mut o0, mut o1, mut o2, o3) = (m[1], m[3], m[5], m[7]);
    o2 -= o3;
    o1 -= o2;
    o0 = (o0 - o1) * INV_SQRT2;
    let odd = idct1d_4_val(o0, o2, o1, o3);
    let o = [
        odd[0] * INV_WC_MULTIPLIERS_8[0],
        odd[1] * INV_WC_MULTIPLIERS_8[1],
        odd[2] * INV_WC_MULTIPLIERS_8[2],
        odd[3] * INV_WC_MULTIPLIERS_8[3],
    ];
    let e = idct1d_4_val(e0, e2, e1, e3);
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

pub fn dct_4x4_full_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    for y in 0..2 {
        for x in 0..2 {
            let mut temp = [0.0f32; 16];
            for iy in 0..4 {
                let base = (y * 4 + iy) * 8 + x * 4;
                let r = dct1d_4_val(
                    input[base],
                    input[base + 1],
                    input[base + 2],
                    input[base + 3],
                );
                for col in 0..4 {
                    temp[col * 4 + iy] = r[col] * 0.25;
                }
            }
            for col in 0..4 {
                let s = col * 4;
                let r = dct1d_4_val(temp[s], temp[s + 1], temp[s + 2], temp[s + 3]);
                for ix in 0..4 {
                    output[(y + col * 2) * 8 + x + ix * 2] = r[ix] * 0.25;
                }
            }
        }
    }
    let block00 = output[0];
    let block01 = output[1];
    let block10 = output[8];
    let block11 = output[9];
    output[0] = (block00 + block01 + block10 + block11) * 0.25;
    output[1] = (block00 + block01 - block10 - block11) * 0.25;
    output[8] = (block00 - block01 + block10 - block11) * 0.25;
    output[9] = (block00 - block01 - block10 + block11) * 0.25;
}

pub fn dct_4x8_full_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    for y in 0..2 {
        let mut temp = [0.0f32; 32];
        for iy in 0..4 {
            let base = (y * 4 + iy) * 8;
            let r = dct1d_8_val([
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ]);
            for col in 0..8 {
                temp[col * 4 + iy] = r[col] * 0.125;
            }
        }
        for col in 0..8 {
            let s = col * 4;
            let r = dct1d_4_val(temp[s], temp[s + 1], temp[s + 2], temp[s + 3]);
            for iy in 0..4 {
                output[(y + iy * 2) * 8 + col] = r[iy] * 0.25;
            }
        }
    }
    let block0_dc = output[0];
    let block1_dc = output[8];
    output[0] = (block0_dc + block1_dc) * 0.5;
    output[8] = (block0_dc - block1_dc) * 0.5;
}

pub fn dct_8x4_full_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    for x in 0..2 {
        let mut temp = [0.0f32; 32];
        for iy in 0..8 {
            let base = iy * 8 + x * 4;
            let r = dct1d_4_val(
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            );
            for col in 0..4 {
                temp[col * 8 + iy] = r[col] * 0.25;
            }
        }
        for col in 0..4 {
            let s = col * 8;
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
            for ix in 0..8 {
                output[(x + col * 2) * 8 + ix] = r[ix] * 0.125;
            }
        }
    }
    let block0_dc = output[0];
    let block1_dc = output[8];
    output[0] = (block0_dc + block1_dc) * 0.5;
    output[8] = (block0_dc - block1_dc) * 0.5;
}

pub fn idct_4x4_full_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    let mut coeffs = *input;
    // Reverse DC combining (2x2 inverse)
    let a = coeffs[0];
    let b = coeffs[1];
    let c = coeffs[8];
    let d = coeffs[9];
    coeffs[0] = a + b + c + d;
    coeffs[1] = a + b - c - d;
    coeffs[8] = a - b + c - d;
    coeffs[9] = a - b - c + d;

    for y in 0..2 {
        for x in 0..2 {
            let mut block = [0.0f32; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    block[iy * 4 + ix] = coeffs[(y + iy * 2) * 8 + (x + ix * 2)];
                }
            }
            // idct_4x4
            let mut temp = [0.0f32; 16];
            for row in 0..4 {
                let s = row * 4;
                let r = idct1d_4_val(
                    block[s] * 4.0,
                    block[s + 2] * 4.0,
                    block[s + 1] * 4.0,
                    block[s + 3] * 4.0,
                );
                for col in 0..4 {
                    temp[col * 4 + row] = r[col];
                }
            }
            for row in 0..4 {
                let s = row * 4;
                let r = idct1d_4_val(
                    temp[s] * 4.0,
                    temp[s + 2] * 4.0,
                    temp[s + 1] * 4.0,
                    temp[s + 3] * 4.0,
                );
                for ix in 0..4 {
                    output[(y * 4 + row) * 8 + (x * 4 + ix)] = r[ix];
                }
            }
        }
    }
}

pub fn idct_4x8_full_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    let mut coeffs = *input;
    let combined_dc = coeffs[0];
    let combined_ac = coeffs[8];
    coeffs[0] = combined_dc + combined_ac;
    coeffs[8] = combined_dc - combined_ac;

    for y in 0..2 {
        let mut block = [0.0f32; 32];
        for iy in 0..4 {
            for ix in 0..8 {
                block[iy * 8 + ix] = coeffs[(y + iy * 2) * 8 + ix];
            }
        }
        // idct_4x8
        let mut temp = [0.0f32; 32];
        for col in 0..8 {
            let a = block[col] * 4.0;
            let b = block[8 + col] * 4.0;
            let c = block[16 + col] * 4.0;
            let d = block[24 + col] * 4.0;
            let r = idct1d_4_val(a, c, b, d);
            for row in 0..4 {
                temp[row * 8 + col] = r[row];
            }
        }
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
            for ix in 0..8 {
                output[(y * 4 + row) * 8 + ix] = r[ix];
            }
        }
    }
}

pub fn idct_8x4_full_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    let mut coeffs = *input;
    let combined_dc = coeffs[0];
    let combined_ac = coeffs[8];
    coeffs[0] = combined_dc + combined_ac;
    coeffs[8] = combined_dc - combined_ac;

    for x in 0..2 {
        let mut block = [0.0f32; 32];
        for iy in 0..4 {
            for ix in 0..8 {
                block[iy * 8 + ix] = coeffs[(x + iy * 2) * 8 + ix];
            }
        }
        // idct_8x4
        let mut temp = [0.0f32; 32];
        for row in 0..4 {
            let s = row * 8;
            let m = [
                block[s] * 8.0,
                block[s + 1] * 8.0,
                block[s + 2] * 8.0,
                block[s + 3] * 8.0,
                block[s + 4] * 8.0,
                block[s + 5] * 8.0,
                block[s + 6] * 8.0,
                block[s + 7] * 8.0,
            ];
            let r = idct1d_8_core_val(m);
            for col in 0..8 {
                temp[col * 4 + row] = r[col];
            }
        }
        for row in 0..8 {
            let s = row * 4;
            let r = idct1d_4_val(
                temp[s] * 4.0,
                temp[s + 2] * 4.0,
                temp[s + 1] * 4.0,
                temp[s + 3] * 4.0,
            );
            for ix in 0..4 {
                output[row * 8 + (x * 4 + ix)] = r[ix];
            }
        }
    }
}

// =============================================================================
// x86_64 AVX2 implementations
// =============================================================================

/// Vectorized 4-point forward DCT butterfly.
///
/// Each f32x8 register holds one "position" across 8 independent DCTs.
/// The butterfly operates across registers, processing all 8 DCTs simultaneously.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
#[allow(clippy::type_complexity)]
fn vectorized_dct1d_4(
    token: archmage::X64V3Token,
    r0: magetypes::simd::f32x8,
    r1: magetypes::simd::f32x8,
    r2: magetypes::simd::f32x8,
    r3: magetypes::simd::f32x8,
) -> (
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
) {
    use magetypes::simd::f32x8;

    let sqrt2 = f32x8::splat(token, SQRT2);

    // AddReverse / SubReverse
    let t0 = r0 + r3;
    let t1 = r1 + r2;
    let t2 = r0 - r3;
    let t3 = r1 - r2;

    // DCT-2 on first half
    let u0 = t0 + t1;
    let u1 = t0 - t1;

    // WcMultipliers_4 on second half
    let v0 = t2 * f32x8::splat(token, WC_MULTIPLIERS_4[0]);
    let v1 = t3 * f32x8::splat(token, WC_MULTIPLIERS_4[1]);

    // DCT-2 on second half
    let w0 = v0 + v1;
    let w1 = v0 - v1;

    // B transform
    let b0 = sqrt2.mul_add(w0, w1);

    // InverseEvenOdd: [u0, b0, u1, w1]
    (u0, b0, u1, w1)
}

/// Vectorized 4-point inverse DCT butterfly.
///
/// Input positions: [even0, odd0, even1, odd1] (InverseEvenOdd layout from forward).
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
#[allow(clippy::type_complexity)]
fn vectorized_idct1d_4(
    token: archmage::X64V3Token,
    r0: magetypes::simd::f32x8,
    r1: magetypes::simd::f32x8,
    r2: magetypes::simd::f32x8,
    r3: magetypes::simd::f32x8,
) -> (
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
) {
    use magetypes::simd::f32x8;

    let inv_sqrt2 = f32x8::splat(token, INV_SQRT2);
    let half = f32x8::splat(token, 0.5);

    // De-interleave: even0=r0, even1=r2, odd0=r1, odd1=r3
    let even0 = r0;
    let odd0 = r1;
    let even1 = r2;
    let odd1 = r3;

    // Reverse B transform on odd half
    let c = (odd0 - odd1) * inv_sqrt2;

    // Reverse IDCT-2 on odd half
    let o0_pre = (c + odd1) * half;
    let o1_pre = (c - odd1) * half;

    // Reverse WcMultipliers
    let o0 = o0_pre * f32x8::splat(token, INV_WC_MULTIPLIERS_4[0]);
    let o1 = o1_pre * f32x8::splat(token, INV_WC_MULTIPLIERS_4[1]);

    // Reverse IDCT-2 on even half
    let e0 = (even0 + even1) * half;
    let e1 = (even0 - even1) * half;

    // Combine
    let out0 = (e0 + o0) * half;
    let out1 = (e1 + o1) * half;
    let out2 = (e1 - o1) * half;
    let out3 = (e0 - o0) * half;

    (out0, out1, out2, out3)
}

/// Per-lane 4x4 transpose using AVX2 in-lane operations.
///
/// Operates independently on the low and high 128-bit lanes.
/// Each lane transposes a 4x4 matrix.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
#[allow(clippy::type_complexity)]
fn transpose_4x4_per_lane(
    token: archmage::X64V3Token,
    r0: magetypes::simd::f32x8,
    r1: magetypes::simd::f32x8,
    r2: magetypes::simd::f32x8,
    r3: magetypes::simd::f32x8,
) -> (
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
    magetypes::simd::f32x8,
) {
    use core::arch::x86_64::*;
    use magetypes::simd::f32x8;

    let r0 = r0.raw();
    let r1 = r1.raw();
    let r2 = r2.raw();
    let r3 = r3.raw();

    // Stage 1: interleave pairs (operates per 128-bit lane)
    let t0 = _mm256_unpacklo_ps(r0, r1); // [a0,b0,a1,b1 | e0,f0,e1,f1]
    let t1 = _mm256_unpackhi_ps(r0, r1); // [a2,b2,a3,b3 | e2,f2,e3,f3]
    let t2 = _mm256_unpacklo_ps(r2, r3); // [c0,d0,c1,d1 | g0,h0,g1,h1]
    let t3 = _mm256_unpackhi_ps(r2, r3); // [c2,d2,c3,d3 | g2,h2,g3,h3]

    // Stage 2: form 4-element groups (per-lane shuffle)
    let s0 = _mm256_shuffle_ps::<0x44>(t0, t2); // [a0,b0,c0,d0 | e0,f0,g0,h0]
    let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2); // [a1,b1,c1,d1 | e1,f1,g1,h1]
    let s2 = _mm256_shuffle_ps::<0x44>(t1, t3); // [a2,b2,c2,d2 | e2,f2,g2,h2]
    let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3); // [a3,b3,c3,d3 | e3,f3,g3,h3]

    (
        f32x8::from_m256(token, s0),
        f32x8::from_m256(token, s1),
        f32x8::from_m256(token, s2),
        f32x8::from_m256(token, s3),
    )
}

/// Interleave low and high 128-bit lanes: [a0,a1,a2,a3|b0,b1,b2,b3] → [a0,b0,a1,b1,a2,b2,a3,b3]
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn interleave_lanes(
    token: archmage::X64V3Token,
    v: magetypes::simd::f32x8,
) -> magetypes::simd::f32x8 {
    use core::arch::x86_64::*;
    use magetypes::simd::f32x8;

    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    f32x8::from_m256(token, _mm256_permutevar8x32_ps(v.raw(), perm))
}

/// De-interleave: [a0,b0,a1,b1,a2,b2,a3,b3] → [a0,a1,a2,a3|b0,b1,b2,b3]
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn deinterleave_lanes(
    token: archmage::X64V3Token,
    v: magetypes::simd::f32x8,
) -> magetypes::simd::f32x8 {
    use core::arch::x86_64::*;
    use magetypes::simd::f32x8;

    let perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    f32x8::from_m256(token, _mm256_permutevar8x32_ps(v.raw(), perm))
}

/// AVX2 DCT4x4 full transform.
///
/// Packs two sub-blocks per f32x8 (one per 128-bit lane). Each half-register
/// holds one 4×4 sub-block's data. Column DCTs, per-lane transpose, row DCTs,
/// then interleave via vpermps.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn dct_4x4_full_avx2(token: archmage::X64V3Token, input: &[f32; 64], output: &mut [f32; 64]) {
    use magetypes::simd::f32x8;

    let scale = f32x8::splat(token, 0.25);

    // Load 8 rows. Each row naturally packs left/right sub-blocks:
    // r0 = [sub(0,0)_row0 | sub(0,1)_row0], etc.
    let r0 = f32x8::from_slice(token, &input[0..]);
    let r1 = f32x8::from_slice(token, &input[8..]);
    let r2 = f32x8::from_slice(token, &input[16..]);
    let r3 = f32x8::from_slice(token, &input[24..]);
    let r4 = f32x8::from_slice(token, &input[32..]);
    let r5 = f32x8::from_slice(token, &input[40..]);
    let r6 = f32x8::from_slice(token, &input[48..]);
    let r7 = f32x8::from_slice(token, &input[56..]);

    // Column DCTs on top pair (y=0 sub-blocks: rows 0-3)
    let (t0, t1, t2, t3) = vectorized_dct1d_4(token, r0, r1, r2, r3);
    // Column DCTs on bottom pair (y=1 sub-blocks: rows 4-7)
    let (b0, b1, b2, b3) = vectorized_dct1d_4(token, r4, r5, r6, r7);

    // Scale by 1/4
    let t0 = t0 * scale;
    let t1 = t1 * scale;
    let t2 = t2 * scale;
    let t3 = t3 * scale;
    let b0 = b0 * scale;
    let b1 = b1 * scale;
    let b2 = b2 * scale;
    let b3 = b3 * scale;

    // Per-lane 4x4 transpose (each 128-bit lane transposes independently)
    let (t0, t1, t2, t3) = transpose_4x4_per_lane(token, t0, t1, t2, t3);
    let (b0, b1, b2, b3) = transpose_4x4_per_lane(token, b0, b1, b2, b3);

    // Row DCTs
    let (t0, t1, t2, t3) = vectorized_dct1d_4(token, t0, t1, t2, t3);
    let (b0, b1, b2, b3) = vectorized_dct1d_4(token, b0, b1, b2, b3);

    // Scale by 1/4
    let t0 = t0 * scale;
    let t1 = t1 * scale;
    let t2 = t2 * scale;
    let t3 = t3 * scale;
    let b0 = b0 * scale;
    let b1 = b1 * scale;
    let b2 = b2 * scale;
    let b3 = b3 * scale;

    // Interleave: [sub_left[0..4] | sub_right[0..4]] → [L0,R0,L1,R1,L2,R2,L3,R3]
    let t0 = interleave_lanes(token, t0);
    let t1 = interleave_lanes(token, t1);
    let t2 = interleave_lanes(token, t2);
    let t3 = interleave_lanes(token, t3);
    let b0 = interleave_lanes(token, b0);
    let b1 = interleave_lanes(token, b1);
    let b2 = interleave_lanes(token, b2);
    let b3 = interleave_lanes(token, b3);

    // Store: alternating top/bottom rows
    // top col_freq 0 → row 0, bottom col_freq 0 → row 1, etc.
    t0.store((&mut output[0..8]).try_into().unwrap());
    b0.store((&mut output[8..16]).try_into().unwrap());
    t1.store((&mut output[16..24]).try_into().unwrap());
    b1.store((&mut output[24..32]).try_into().unwrap());
    t2.store((&mut output[32..40]).try_into().unwrap());
    b2.store((&mut output[40..48]).try_into().unwrap());
    t3.store((&mut output[48..56]).try_into().unwrap());
    b3.store((&mut output[56..64]).try_into().unwrap());

    // DC combine: 2x2 Hadamard at positions [0], [1], [8], [9]
    let dc00 = output[0];
    let dc01 = output[1];
    let dc10 = output[8];
    let dc11 = output[9];
    output[0] = (dc00 + dc01 + dc10 + dc11) * 0.25;
    output[1] = (dc00 + dc01 - dc10 - dc11) * 0.25;
    output[8] = (dc00 - dc01 + dc10 - dc11) * 0.25;
    output[9] = (dc00 - dc01 - dc10 + dc11) * 0.25;
}

/// AVX2 inverse DCT4x4 full transform.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn idct_4x4_full_avx2(token: archmage::X64V3Token, input: &[f32; 64], output: &mut [f32; 64]) {
    use magetypes::simd::f32x8;

    // Copy and reverse DC combining first
    let mut coeffs = *input;
    let a = coeffs[0];
    let b = coeffs[1];
    let c = coeffs[8];
    let d = coeffs[9];
    coeffs[0] = a + b + c + d;
    coeffs[1] = a + b - c - d;
    coeffs[8] = a - b + c - d;
    coeffs[9] = a - b - c + d;

    // De-interleave rows and load with 4x scaling
    // Input is interleaved: output[(y + col*2)*8 + x + ix*2]
    // We need to de-interleave back to [sub_left | sub_right] per row.
    let scale4 = f32x8::splat(token, 4.0);

    // Load interleaved rows and de-interleave
    let row0 = deinterleave_lanes(token, f32x8::from_slice(token, &coeffs[0..]));
    let row1 = deinterleave_lanes(token, f32x8::from_slice(token, &coeffs[8..]));
    let row2 = deinterleave_lanes(token, f32x8::from_slice(token, &coeffs[16..]));
    let row3 = deinterleave_lanes(token, f32x8::from_slice(token, &coeffs[24..]));
    let row4 = deinterleave_lanes(token, f32x8::from_slice(token, &coeffs[32..]));
    let row5 = deinterleave_lanes(token, f32x8::from_slice(token, &coeffs[40..]));
    let row6 = deinterleave_lanes(token, f32x8::from_slice(token, &coeffs[48..]));
    let row7 = deinterleave_lanes(token, f32x8::from_slice(token, &coeffs[56..]));

    // Top pair (y=0): rows 0,2,4,6 contain col_freq 0,1,2,3
    // Bottom pair (y=1): rows 1,3,5,7 contain col_freq 0,1,2,3
    // After de-interleave: each register has [sub(y,0)_data | sub(y,1)_data]

    // For IDCT: need de-interleaved inputs in InverseEvenOdd order
    // Forward output was [u0, b0, u1, w1] at positions 0,1,2,3
    // IDCT input: positions 0=even0, 1=odd0, 2=even1, 3=odd1
    // Rows map: row 0→pos0, row 2→pos1, row 4→pos2, row 6→pos3 for top pair
    let t0 = row0 * scale4;
    let t1 = row2 * scale4;
    let t2 = row4 * scale4;
    let t3 = row6 * scale4;
    let b0 = row1 * scale4;
    let b1 = row3 * scale4;
    let b2 = row5 * scale4;
    let b3 = row7 * scale4;

    // Inverse row DCTs (operates on col-freq dimension)
    let (t0, t1, t2, t3) = vectorized_idct1d_4(token, t0, t1, t2, t3);
    let (b0, b1, b2, b3) = vectorized_idct1d_4(token, b0, b1, b2, b3);

    // Per-lane 4x4 transpose
    let (t0, t1, t2, t3) = transpose_4x4_per_lane(token, t0, t1, t2, t3);
    let (b0, b1, b2, b3) = transpose_4x4_per_lane(token, b0, b1, b2, b3);

    // Scale for column IDCT
    let (t0, t1, t2, t3) = (t0 * scale4, t1 * scale4, t2 * scale4, t3 * scale4);
    let (b0, b1, b2, b3) = (b0 * scale4, b1 * scale4, b2 * scale4, b3 * scale4);

    // Inverse column DCTs
    let (t0, t1, t2, t3) = vectorized_idct1d_4(token, t0, t1, t2, t3);
    let (b0, b1, b2, b3) = vectorized_idct1d_4(token, b0, b1, b2, b3);

    // Store: top sub-blocks in rows 0-3, bottom in rows 4-7
    // Top: t0=row0, t1=row1, t2=row2, t3=row3
    // Bottom: b0=row4, b1=row5, b2=row6, b3=row7
    t0.store((&mut output[0..8]).try_into().unwrap());
    t1.store((&mut output[8..16]).try_into().unwrap());
    t2.store((&mut output[16..24]).try_into().unwrap());
    t3.store((&mut output[24..32]).try_into().unwrap());
    b0.store((&mut output[32..40]).try_into().unwrap());
    b1.store((&mut output[40..48]).try_into().unwrap());
    b2.store((&mut output[48..56]).try_into().unwrap());
    b3.store((&mut output[56..64]).try_into().unwrap());
}

/// AVX2 DCT4x8 full transform.
///
/// Two vertically-stacked 4x8 sub-blocks. Uses 4-point column DCTs (vectorized)
/// then 8×8 transpose + 8-point row DCTs (reusing dct8 infrastructure).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn dct_4x8_full_avx2(token: archmage::X64V3Token, input: &[f32; 64], output: &mut [f32; 64]) {
    use crate::dct8::{transpose_8x8_regs, vectorized_dct1d_8};
    use magetypes::simd::f32x8;

    let scale4 = f32x8::splat(token, 0.25);
    let scale8 = f32x8::splat(token, 0.125);

    // Load 8 rows
    let r0 = f32x8::from_slice(token, &input[0..]);
    let r1 = f32x8::from_slice(token, &input[8..]);
    let r2 = f32x8::from_slice(token, &input[16..]);
    let r3 = f32x8::from_slice(token, &input[24..]);
    let r4 = f32x8::from_slice(token, &input[32..]);
    let r5 = f32x8::from_slice(token, &input[40..]);
    let r6 = f32x8::from_slice(token, &input[48..]);
    let r7 = f32x8::from_slice(token, &input[56..]);

    // 4-point column DCTs on top half (rows 0-3) and bottom half (rows 4-7)
    // Each register holds a full row (8 elements = 8 independent column DCTs)
    let (ct0, ct1, ct2, ct3) = vectorized_dct1d_4(token, r0, r1, r2, r3);
    let (cb0, cb1, cb2, cb3) = vectorized_dct1d_4(token, r4, r5, r6, r7);

    // Scale by 1/4
    let ct0 = ct0 * scale4;
    let ct1 = ct1 * scale4;
    let ct2 = ct2 * scale4;
    let ct3 = ct3 * scale4;
    let cb0 = cb0 * scale4;
    let cb1 = cb1 * scale4;
    let cb2 = cb2 * scale4;
    let cb3 = cb3 * scale4;

    // Now we have 8 registers (4 top col-freq bands + 4 bottom col-freq bands).
    // Each register holds 8 values = one full row of the intermediate.
    // To do 8-point row DCTs, we need to transpose so that each register holds
    // one position across 8 independent DCTs.
    let (p0, p1, p2, p3, p4, p5, p6, p7) =
        transpose_8x8_regs(token, ct0, ct1, ct2, ct3, cb0, cb1, cb2, cb3);

    // 8-point row DCTs (8 independent DCTs in parallel)
    let (rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7) =
        vectorized_dct1d_8(token, p0, p1, p2, p3, p4, p5, p6, p7);

    // Scale by 1/8
    let rd0 = rd0 * scale8;
    let rd1 = rd1 * scale8;
    let rd2 = rd2 * scale8;
    let rd3 = rd3 * scale8;
    let rd4 = rd4 * scale8;
    let rd5 = rd5 * scale8;
    let rd6 = rd6 * scale8;
    let rd7 = rd7 * scale8;

    // Transpose back to get each DCT's full output in one register
    let (o0, o1, o2, o3, o4, o5, o6, o7) =
        transpose_8x8_regs(token, rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7);

    // o0..o3 = top sub-block col-freq 0..3 full row-freq outputs
    // o4..o7 = bottom sub-block col-freq 0..3 full row-freq outputs
    // Store interleaved: row 0 = top cfreq0, row 1 = bot cfreq0, etc.
    o0.store((&mut output[0..8]).try_into().unwrap()); // row 0
    o4.store((&mut output[8..16]).try_into().unwrap()); // row 1
    o1.store((&mut output[16..24]).try_into().unwrap()); // row 2
    o5.store((&mut output[24..32]).try_into().unwrap()); // row 3
    o2.store((&mut output[32..40]).try_into().unwrap()); // row 4
    o6.store((&mut output[40..48]).try_into().unwrap()); // row 5
    o3.store((&mut output[48..56]).try_into().unwrap()); // row 6
    o7.store((&mut output[56..64]).try_into().unwrap()); // row 7

    // DC combine
    let block0_dc = output[0];
    let block1_dc = output[8];
    output[0] = (block0_dc + block1_dc) * 0.5;
    output[8] = (block0_dc - block1_dc) * 0.5;
}

/// AVX2 inverse DCT4x8 full transform.
///
/// Reverses the forward DCT4x8: DC uncombine → de-interleave rows →
/// transpose → 8-pt inverse row DCT → transpose → 4-pt inverse column DCT.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn idct_4x8_full_avx2(token: archmage::X64V3Token, input: &[f32; 64], output: &mut [f32; 64]) {
    use crate::dct8::{transpose_8x8_regs, vectorized_idct1d_8};
    use magetypes::simd::f32x8;

    // DC uncombine (reverse of forward's DC combine)
    let mut coeffs = *input;
    let combined_dc = coeffs[0];
    let combined_ac = coeffs[8];
    coeffs[0] = combined_dc + combined_ac;
    coeffs[8] = combined_dc - combined_ac;

    let scale4 = f32x8::splat(token, 4.0);

    // Load de-interleaved: forward stored [top_cf0, bot_cf0, top_cf1, bot_cf1, ...]
    // Top sub-block column-freq bands from even rows, bottom from odd rows
    let ct0 = f32x8::from_slice(token, &coeffs[0..]); // row 0: top col-freq 0
    let cb0 = f32x8::from_slice(token, &coeffs[8..]); // row 1: bot col-freq 0
    let ct1 = f32x8::from_slice(token, &coeffs[16..]); // row 2: top col-freq 1
    let cb1 = f32x8::from_slice(token, &coeffs[24..]); // row 3: bot col-freq 1
    let ct2 = f32x8::from_slice(token, &coeffs[32..]); // row 4: top col-freq 2
    let cb2 = f32x8::from_slice(token, &coeffs[40..]); // row 5: bot col-freq 2
    let ct3 = f32x8::from_slice(token, &coeffs[48..]); // row 6: top col-freq 3
    let cb3 = f32x8::from_slice(token, &coeffs[56..]); // row 7: bot col-freq 3

    // Transpose: each register becomes one row-freq position across all 8 col-freq bands
    let (p0, p1, p2, p3, p4, p5, p6, p7) =
        transpose_8x8_regs(token, ct0, ct1, ct2, ct3, cb0, cb1, cb2, cb3);

    // Inverse 8-pt row DCTs (vectorized_idct1d_8 has no internal halvings,
    // so no pre-scaling needed — it directly inverts the 1/8-scaled forward)
    let (r0, r1, r2, r3, r4, r5, r6, r7) =
        vectorized_idct1d_8(token, p0, p1, p2, p3, p4, p5, p6, p7);

    // Transpose back: each register = one col-freq band's spatial columns
    let (ct0, ct1, ct2, ct3, cb0, cb1, cb2, cb3) =
        transpose_8x8_regs(token, r0, r1, r2, r3, r4, r5, r6, r7);

    // Scale by 4 (undo forward's 1/4) and inverse 4-pt column DCTs
    let (t0, t1, t2, t3) = vectorized_idct1d_4(
        token,
        ct0 * scale4,
        ct1 * scale4,
        ct2 * scale4,
        ct3 * scale4,
    );
    let (b0, b1, b2, b3) = vectorized_idct1d_4(
        token,
        cb0 * scale4,
        cb1 * scale4,
        cb2 * scale4,
        cb3 * scale4,
    );

    // Store: top sub-block in rows 0-3, bottom in rows 4-7
    t0.store((&mut output[0..8]).try_into().unwrap());
    t1.store((&mut output[8..16]).try_into().unwrap());
    t2.store((&mut output[16..24]).try_into().unwrap());
    t3.store((&mut output[24..32]).try_into().unwrap());
    b0.store((&mut output[32..40]).try_into().unwrap());
    b1.store((&mut output[40..48]).try_into().unwrap());
    b2.store((&mut output[48..56]).try_into().unwrap());
    b3.store((&mut output[56..64]).try_into().unwrap());
}

/// AVX2 DCT8x4 full transform.
///
/// Two horizontally-adjacent 8x4 sub-blocks. Uses 8-point column DCTs,
/// 8x8 transpose, then 4-point row DCTs (separate per sub-block).
///
/// Output layout: output[(x + rfreq*2)*8 + cfreq] where x=sub-block (0,1),
/// rfreq=row-frequency (0-3), cfreq=column-frequency (0-7).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn dct_8x4_full_avx2(token: archmage::X64V3Token, input: &[f32; 64], output: &mut [f32; 64]) {
    use crate::dct8::{transpose_8x8_regs, vectorized_dct1d_8};
    use magetypes::simd::f32x8;

    let scale4 = f32x8::splat(token, 0.25);
    let scale8 = f32x8::splat(token, 0.125);

    // Load 8 rows. Each row has [sub0_cols(0-3) | sub1_cols(0-3)]
    let r0 = f32x8::from_slice(token, &input[0..]);
    let r1 = f32x8::from_slice(token, &input[8..]);
    let r2 = f32x8::from_slice(token, &input[16..]);
    let r3 = f32x8::from_slice(token, &input[24..]);
    let r4 = f32x8::from_slice(token, &input[32..]);
    let r5 = f32x8::from_slice(token, &input[40..]);
    let r6 = f32x8::from_slice(token, &input[48..]);
    let r7 = f32x8::from_slice(token, &input[56..]);

    // 8-point column DCTs: processes 8 independent column DCTs
    // (4 columns per sub-block × 2 sub-blocks = 8)
    let (c0, c1, c2, c3, c4, c5, c6, c7) =
        vectorized_dct1d_8(token, r0, r1, r2, r3, r4, r5, r6, r7);

    // Scale by 1/8
    let c0 = c0 * scale8;
    let c1 = c1 * scale8;
    let c2 = c2 * scale8;
    let c3 = c3 * scale8;
    let c4 = c4 * scale8;
    let c5 = c5 * scale8;
    let c6 = c6 * scale8;
    let c7 = c7 * scale8;

    // 8x8 transpose: after this, each register t_k holds all 8 column-freq
    // values for spatial column k. Sub0 cols are t0-t3, sub1 cols are t4-t7.
    let (t0, t1, t2, t3, t4, t5, t6, t7) =
        transpose_8x8_regs(token, c0, c1, c2, c3, c4, c5, c6, c7);

    // 4-point row DCTs for sub0 (columns 0-3)
    let (s0_0, s0_1, s0_2, s0_3) = vectorized_dct1d_4(token, t0, t1, t2, t3);
    // 4-point row DCTs for sub1 (columns 4-7)
    let (s1_0, s1_1, s1_2, s1_3) = vectorized_dct1d_4(token, t4, t5, t6, t7);

    // Scale by 1/4
    let s0_0 = s0_0 * scale4;
    let s0_1 = s0_1 * scale4;
    let s0_2 = s0_2 * scale4;
    let s0_3 = s0_3 * scale4;
    let s1_0 = s1_0 * scale4;
    let s1_1 = s1_1 * scale4;
    let s1_2 = s1_2 * scale4;
    let s1_3 = s1_3 * scale4;

    // Store interleaved: output[(x + rfreq*2)*8 + cfreq]
    // s0_k = sub0's rfreq k: all 8 cfreq values
    // s1_k = sub1's rfreq k: all 8 cfreq values
    s0_0.store((&mut output[0..8]).try_into().unwrap()); // row 0: x=0, rfreq=0
    s1_0.store((&mut output[8..16]).try_into().unwrap()); // row 1: x=1, rfreq=0
    s0_1.store((&mut output[16..24]).try_into().unwrap()); // row 2: x=0, rfreq=1
    s1_1.store((&mut output[24..32]).try_into().unwrap()); // row 3: x=1, rfreq=1
    s0_2.store((&mut output[32..40]).try_into().unwrap()); // row 4: x=0, rfreq=2
    s1_2.store((&mut output[40..48]).try_into().unwrap()); // row 5: x=1, rfreq=2
    s0_3.store((&mut output[48..56]).try_into().unwrap()); // row 6: x=0, rfreq=3
    s1_3.store((&mut output[56..64]).try_into().unwrap()); // row 7: x=1, rfreq=3

    // DC combine
    let block0_dc = output[0];
    let block1_dc = output[8];
    output[0] = (block0_dc + block1_dc) * 0.5;
    output[8] = (block0_dc - block1_dc) * 0.5;
}

/// AVX2 inverse DCT8x4 full transform.
///
/// Reverses the forward DCT8x4: DC uncombine → de-interleave rows →
/// 4-pt inverse row DCT → transpose → 8-pt inverse column DCT → store.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn idct_8x4_full_avx2(token: archmage::X64V3Token, input: &[f32; 64], output: &mut [f32; 64]) {
    use crate::dct8::{transpose_8x8_regs, vectorized_idct1d_8};
    use magetypes::simd::f32x8;

    // DC uncombine (reverse of forward's DC combine)
    let mut coeffs = *input;
    let combined_dc = coeffs[0];
    let combined_ac = coeffs[8];
    coeffs[0] = combined_dc + combined_ac;
    coeffs[8] = combined_dc - combined_ac;

    let scale4 = f32x8::splat(token, 4.0);

    // Load de-interleaved: forward stored [sub0_rf0, sub1_rf0, sub0_rf1, sub1_rf1, ...]
    // Sub0 row-freq bands from even rows, sub1 from odd rows
    let s0_0 = f32x8::from_slice(token, &coeffs[0..]); // row 0: sub0 rfreq 0
    let s1_0 = f32x8::from_slice(token, &coeffs[8..]); // row 1: sub1 rfreq 0
    let s0_1 = f32x8::from_slice(token, &coeffs[16..]); // row 2: sub0 rfreq 1
    let s1_1 = f32x8::from_slice(token, &coeffs[24..]); // row 3: sub1 rfreq 1
    let s0_2 = f32x8::from_slice(token, &coeffs[32..]); // row 4: sub0 rfreq 2
    let s1_2 = f32x8::from_slice(token, &coeffs[40..]); // row 5: sub1 rfreq 2
    let s0_3 = f32x8::from_slice(token, &coeffs[48..]); // row 6: sub0 rfreq 3
    let s1_3 = f32x8::from_slice(token, &coeffs[56..]); // row 7: sub1 rfreq 3

    // Inverse 4-pt row DCTs with 4x pre-scaling (vectorized_idct1d_4 has internal halvings)
    let (s0_0, s0_1, s0_2, s0_3) = vectorized_idct1d_4(
        token,
        s0_0 * scale4,
        s0_1 * scale4,
        s0_2 * scale4,
        s0_3 * scale4,
    );
    let (s1_0, s1_1, s1_2, s1_3) = vectorized_idct1d_4(
        token,
        s1_0 * scale4,
        s1_1 * scale4,
        s1_2 * scale4,
        s1_3 * scale4,
    );

    // Transpose: packs sub0 and sub1 spatial columns into positional layout
    let (p0, p1, p2, p3, p4, p5, p6, p7) =
        transpose_8x8_regs(token, s0_0, s0_1, s0_2, s0_3, s1_0, s1_1, s1_2, s1_3);

    // Inverse 8-pt column DCTs (vectorized_idct1d_8 has no internal halvings,
    // so no pre-scaling needed). After IDCT, each register r_k holds spatial
    // row k across all 8 columns — no second transpose needed.
    let (r0, r1, r2, r3, r4, r5, r6, r7) =
        vectorized_idct1d_8(token, p0, p1, p2, p3, p4, p5, p6, p7);

    // Store 8 rows. Each row has [sub0_cols(0-3) | sub1_cols(0-3)]
    r0.store((&mut output[0..8]).try_into().unwrap());
    r1.store((&mut output[8..16]).try_into().unwrap());
    r2.store((&mut output[16..24]).try_into().unwrap());
    r3.store((&mut output[24..32]).try_into().unwrap());
    r4.store((&mut output[32..40]).try_into().unwrap());
    r5.store((&mut output[40..48]).try_into().unwrap());
    r6.store((&mut output[48..56]).try_into().unwrap());
    r7.store((&mut output[56..64]).try_into().unwrap());
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    fn make_test_input() -> [f32; 64] {
        let mut input = [0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.37 + 1.5).cos();
        }
        input
    }

    #[test]
    fn test_dct_4x4_full_simd_matches_scalar() {
        let input = make_test_input();
        let mut scalar_out = [0.0f32; 64];
        dct_4x4_full_scalar(&input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 64];
                dct_4x4_full(&input, &mut simd_out);
                for i in 0..64 {
                    assert!(
                        (scalar_out[i] - simd_out[i]).abs() < 1e-5,
                        "DCT4x4 mismatch at {i}: scalar={} simd={} [{perm}]",
                        scalar_out[i],
                        simd_out[i]
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_dct_4x4_full_roundtrip() {
        let input = make_test_input();

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut coeffs = [0.0f32; 64];
                let mut pixels = [0.0f32; 64];
                dct_4x4_full(&input, &mut coeffs);
                idct_4x4_full(&coeffs, &mut pixels);
                let max_err = input
                    .iter()
                    .zip(pixels.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 1e-4,
                    "DCT4x4 roundtrip max error {max_err} [{perm}]",
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_dct_4x8_full_simd_matches_scalar() {
        let input = make_test_input();
        let mut scalar_out = [0.0f32; 64];
        dct_4x8_full_scalar(&input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 64];
                dct_4x8_full(&input, &mut simd_out);
                for i in 0..64 {
                    assert!(
                        (scalar_out[i] - simd_out[i]).abs() < 1e-5,
                        "DCT4x8 mismatch at {i}: scalar={} simd={} [{perm}]",
                        scalar_out[i],
                        simd_out[i]
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_dct_4x8_full_roundtrip() {
        let input = make_test_input();

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut coeffs = [0.0f32; 64];
                let mut pixels = [0.0f32; 64];
                dct_4x8_full(&input, &mut coeffs);
                idct_4x8_full(&coeffs, &mut pixels);
                let max_err = input
                    .iter()
                    .zip(pixels.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 1e-4,
                    "DCT4x8 roundtrip max error {max_err} [{perm}]",
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_dct_8x4_full_simd_matches_scalar() {
        let input = make_test_input();
        let mut scalar_out = [0.0f32; 64];
        dct_8x4_full_scalar(&input, &mut scalar_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0.0f32; 64];
                dct_8x4_full(&input, &mut simd_out);
                for i in 0..64 {
                    assert!(
                        (scalar_out[i] - simd_out[i]).abs() < 1e-5,
                        "DCT8x4 mismatch at {i}: scalar={} simd={} [{perm}]",
                        scalar_out[i],
                        simd_out[i]
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_dct_8x4_full_roundtrip() {
        let input = make_test_input();

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut coeffs = [0.0f32; 64];
                let mut pixels = [0.0f32; 64];
                dct_8x4_full(&input, &mut coeffs);
                idct_8x4_full(&coeffs, &mut pixels);
                let max_err = input
                    .iter()
                    .zip(pixels.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 1e-4,
                    "DCT8x4 roundtrip max error {max_err} [{perm}]",
                );
            },
        );
        std::eprintln!("{report}");
    }
}
