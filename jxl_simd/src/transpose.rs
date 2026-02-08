// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD-accelerated matrix transpose.
//!
//! Provides a fast 8x8 f32 transpose using AVX2 shuffle/permute instructions.
//! This is pure data movement — guaranteed bit-exact with the scalar version.

/// Transpose an 8x8 f32 matrix.
///
/// `input` and `output` must each be at least 64 elements.
/// Reads `input[row*8 + col]`, writes `output[col*8 + row]`.
///
/// Dispatches to SIMD when available; falls back to scalar otherwise.
pub fn transpose_8x8(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 64);
    debug_assert!(output.len() >= 64);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            transpose_8x8_avx2(token, input, output);
            return;
        }
    }

    // Scalar fallback
    for row in 0..8 {
        for col in 0..8 {
            output[col * 8 + row] = input[row * 8 + col];
        }
    }
}

/// AVX2 8x8 transpose using unpack/shuffle/permute instructions.
///
/// All operations are pure data movement — no arithmetic, bit-exact with scalar.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn transpose_8x8_avx2(token: archmage::X64V3Token, input: &[f32], output: &mut [f32]) {
    use magetypes::simd::f32x8;

    // Load 8 rows
    let r0 = f32x8::from_slice(token, &input[0..]);
    let r1 = f32x8::from_slice(token, &input[8..]);
    let r2 = f32x8::from_slice(token, &input[16..]);
    let r3 = f32x8::from_slice(token, &input[24..]);
    let r4 = f32x8::from_slice(token, &input[32..]);
    let r5 = f32x8::from_slice(token, &input[40..]);
    let r6 = f32x8::from_slice(token, &input[48..]);
    let r7 = f32x8::from_slice(token, &input[56..]);

    // 3-stage AVX2 8x8 transpose:
    // Stage 1: unpacklo/hi pairs within 128-bit lanes
    // Stage 2: shuffle to get 4-element groups
    // Stage 3: permute2f128 to exchange 128-bit halves
    use core::arch::x86_64::*;

    let r0 = r0.raw();
    let r1 = r1.raw();
    let r2 = r2.raw();
    let r3 = r3.raw();
    let r4 = r4.raw();
    let r5 = r5.raw();
    let r6 = r6.raw();
    let r7 = r7.raw();

    // Stage 1: interleave pairs
    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    let t6 = _mm256_unpacklo_ps(r6, r7);
    let t7 = _mm256_unpackhi_ps(r6, r7);

    // Stage 2: shuffle to form 4-element groups
    let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);

    // Stage 3: exchange 128-bit halves to complete transpose
    let c0 = _mm256_permute2f128_ps::<0x20>(s0, s4);
    let c1 = _mm256_permute2f128_ps::<0x20>(s1, s5);
    let c2 = _mm256_permute2f128_ps::<0x20>(s2, s6);
    let c3 = _mm256_permute2f128_ps::<0x20>(s3, s7);
    let c4 = _mm256_permute2f128_ps::<0x31>(s0, s4);
    let c5 = _mm256_permute2f128_ps::<0x31>(s1, s5);
    let c6 = _mm256_permute2f128_ps::<0x31>(s2, s6);
    let c7 = _mm256_permute2f128_ps::<0x31>(s3, s7);

    // Store results
    // SAFETY: from_raw is safe inside #[arcane] because token proves AVX2 support
    unsafe {
        f32x8::from_raw(c0).store((&mut output[0..8]).try_into().unwrap());
        f32x8::from_raw(c1).store((&mut output[8..16]).try_into().unwrap());
        f32x8::from_raw(c2).store((&mut output[16..24]).try_into().unwrap());
        f32x8::from_raw(c3).store((&mut output[24..32]).try_into().unwrap());
        f32x8::from_raw(c4).store((&mut output[32..40]).try_into().unwrap());
        f32x8::from_raw(c5).store((&mut output[40..48]).try_into().unwrap());
        f32x8::from_raw(c6).store((&mut output[48..56]).try_into().unwrap());
        f32x8::from_raw(c7).store((&mut output[56..64]).try_into().unwrap());
    }
}
