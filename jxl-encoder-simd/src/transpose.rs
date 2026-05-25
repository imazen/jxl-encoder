// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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
#[inline]
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

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            transpose_8x8_neon(token, input, output);
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
#[inline]
#[archmage::arcane]
pub fn transpose_8x8_avx2(token: archmage::X64V3Token, input: &[f32], output: &mut [f32]) {
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

    // Store results — from_m256 is token-gated safe
    f32x8::from_m256(token, c0).store((&mut output[0..8]).try_into().unwrap());
    f32x8::from_m256(token, c1).store((&mut output[8..16]).try_into().unwrap());
    f32x8::from_m256(token, c2).store((&mut output[16..24]).try_into().unwrap());
    f32x8::from_m256(token, c3).store((&mut output[24..32]).try_into().unwrap());
    f32x8::from_m256(token, c4).store((&mut output[32..40]).try_into().unwrap());
    f32x8::from_m256(token, c5).store((&mut output[40..48]).try_into().unwrap());
    f32x8::from_m256(token, c6).store((&mut output[48..56]).try_into().unwrap());
    f32x8::from_m256(token, c7).store((&mut output[56..64]).try_into().unwrap());
}

// ============================================================================
// aarch64 NEON implementation
// ============================================================================

/// NEON 8x8 transpose using four 4x4 sub-transposes.
///
/// Decomposes the 8x8 matrix into four 4x4 quadrants:
/// ```text
///   [A B]       [A^T C^T]
///   [C D]  -->  [B^T D^T]
/// ```
/// Each 4x4 transpose uses vtrn + 64-bit lane swap (2 stages, 4 instructions).
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
pub fn transpose_8x8_neon(token: archmage::NeonToken, input: &[f32], output: &mut [f32]) {
    use magetypes::simd::f32x4;

    // Load 8 rows as pairs of f32x4 (lo = cols 0-3, hi = cols 4-7)
    let r0_lo = f32x4::from_slice(token, &input[0..]).raw();
    let r0_hi = f32x4::from_slice(token, &input[4..]).raw();
    let r1_lo = f32x4::from_slice(token, &input[8..]).raw();
    let r1_hi = f32x4::from_slice(token, &input[12..]).raw();
    let r2_lo = f32x4::from_slice(token, &input[16..]).raw();
    let r2_hi = f32x4::from_slice(token, &input[20..]).raw();
    let r3_lo = f32x4::from_slice(token, &input[24..]).raw();
    let r3_hi = f32x4::from_slice(token, &input[28..]).raw();
    let r4_lo = f32x4::from_slice(token, &input[32..]).raw();
    let r4_hi = f32x4::from_slice(token, &input[36..]).raw();
    let r5_lo = f32x4::from_slice(token, &input[40..]).raw();
    let r5_hi = f32x4::from_slice(token, &input[44..]).raw();
    let r6_lo = f32x4::from_slice(token, &input[48..]).raw();
    let r6_hi = f32x4::from_slice(token, &input[52..]).raw();
    let r7_lo = f32x4::from_slice(token, &input[56..]).raw();
    let r7_hi = f32x4::from_slice(token, &input[60..]).raw();

    // Transpose quadrant A (rows 0-3, cols 0-3) → output rows 0-3, cols 0-3
    let (a0, a1, a2, a3) = transpose_4x4_neon(token, r0_lo, r1_lo, r2_lo, r3_lo);
    // Transpose quadrant B (rows 0-3, cols 4-7) → output rows 4-7, cols 0-3
    let (b0, b1, b2, b3) = transpose_4x4_neon(token, r0_hi, r1_hi, r2_hi, r3_hi);
    // Transpose quadrant C (rows 4-7, cols 0-3) → output rows 0-3, cols 4-7
    let (c0, c1, c2, c3) = transpose_4x4_neon(token, r4_lo, r5_lo, r6_lo, r7_lo);
    // Transpose quadrant D (rows 4-7, cols 4-7) → output rows 4-7, cols 4-7
    let (d0, d1, d2, d3) = transpose_4x4_neon(token, r4_hi, r5_hi, r6_hi, r7_hi);

    // Store: output row i = [A^T row i | C^T row i] for i=0..3
    //        output row i = [B^T row (i-4) | D^T row (i-4)] for i=4..7
    f32x4::from_float32x4_t(token, a0).store((&mut output[0..4]).try_into().unwrap());
    f32x4::from_float32x4_t(token, c0).store((&mut output[4..8]).try_into().unwrap());
    f32x4::from_float32x4_t(token, a1).store((&mut output[8..12]).try_into().unwrap());
    f32x4::from_float32x4_t(token, c1).store((&mut output[12..16]).try_into().unwrap());
    f32x4::from_float32x4_t(token, a2).store((&mut output[16..20]).try_into().unwrap());
    f32x4::from_float32x4_t(token, c2).store((&mut output[20..24]).try_into().unwrap());
    f32x4::from_float32x4_t(token, a3).store((&mut output[24..28]).try_into().unwrap());
    f32x4::from_float32x4_t(token, c3).store((&mut output[28..32]).try_into().unwrap());
    f32x4::from_float32x4_t(token, b0).store((&mut output[32..36]).try_into().unwrap());
    f32x4::from_float32x4_t(token, d0).store((&mut output[36..40]).try_into().unwrap());
    f32x4::from_float32x4_t(token, b1).store((&mut output[40..44]).try_into().unwrap());
    f32x4::from_float32x4_t(token, d1).store((&mut output[44..48]).try_into().unwrap());
    f32x4::from_float32x4_t(token, b2).store((&mut output[48..52]).try_into().unwrap());
    f32x4::from_float32x4_t(token, d2).store((&mut output[52..56]).try_into().unwrap());
    f32x4::from_float32x4_t(token, b3).store((&mut output[56..60]).try_into().unwrap());
    f32x4::from_float32x4_t(token, d3).store((&mut output[60..64]).try_into().unwrap());
}

/// NEON 4x4 transpose using vtrn + 64-bit lane swap.
///
/// Stage 1: vtrn1/vtrn2 interleave pairs of 32-bit elements
/// Stage 2: Reinterpret as f64x2 and vtrn to swap 64-bit halves
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
#[allow(clippy::type_complexity)]
fn transpose_4x4_neon(
    _token: archmage::NeonToken,
    r0: core::arch::aarch64::float32x4_t,
    r1: core::arch::aarch64::float32x4_t,
    r2: core::arch::aarch64::float32x4_t,
    r3: core::arch::aarch64::float32x4_t,
) -> (
    core::arch::aarch64::float32x4_t,
    core::arch::aarch64::float32x4_t,
    core::arch::aarch64::float32x4_t,
    core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::*;

    // Stage 1: interleave 32-bit elements pairwise
    // vtrn1: [a0,b0, a2,b2], vtrn2: [a1,b1, a3,b3]
    let t01_lo = vtrn1q_f32(r0, r1);
    let t01_hi = vtrn2q_f32(r0, r1);
    let t23_lo = vtrn1q_f32(r2, r3);
    let t23_hi = vtrn2q_f32(r2, r3);

    // Stage 2: swap 64-bit halves via reinterpret as f64
    let lo0 = vreinterpretq_f64_f32(t01_lo);
    let lo1 = vreinterpretq_f64_f32(t23_lo);
    let hi0 = vreinterpretq_f64_f32(t01_hi);
    let hi1 = vreinterpretq_f64_f32(t23_hi);

    let out0 = vreinterpretq_f32_f64(vtrn1q_f64(lo0, lo1));
    let out1 = vreinterpretq_f32_f64(vtrn1q_f64(hi0, hi1));
    let out2 = vreinterpretq_f32_f64(vtrn2q_f64(lo0, lo1));
    let out3 = vreinterpretq_f32_f64(vtrn2q_f64(hi0, hi1));

    (out0, out1, out2, out3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use alloc::format;
    use alloc::vec::Vec;

    /// Scalar reference: brute-force 8x8 transpose.
    fn transpose_8x8_scalar_ref(input: &[f32], output: &mut [f32]) {
        for row in 0..8 {
            for col in 0..8 {
                output[col * 8 + row] = input[row * 8 + col];
            }
        }
    }

    /// Pure data movement — guaranteed BIT-EXACT against scalar.  Failure
    /// here indicates a real shuffle-mask bug, not float-precision noise.
    /// Sweeps every f32_edge_battery distribution at n=64 (exact 8x8 block).
    #[test]
    fn transpose_8x8_scalar_vs_dispatch_edge_values() {
        for case in f32_edge_battery(64) {
            if case.data.is_empty() {
                continue;
            }
            let mut ref_out = alloc::vec![0.0_f32; 64];
            transpose_8x8_scalar_ref(&case.data, &mut ref_out);

            run_dispatch_parity(|perm| {
                let mut act_out = alloc::vec![0.0_f32; 64];
                transpose_8x8(&case.data, &mut act_out);
                assert_f32_slice_bit_eq(&ref_out, &act_out, perm, &case.label);
            });
        }
    }

    /// Distinguished-position test: each input element is unique so any
    /// shuffle bug surfaces as a swap rather than a 0↔0 silent equality.
    #[test]
    fn transpose_8x8_unique_positions() {
        // Use bit patterns so we can detect lane swaps unambiguously even
        // if some FP optimization collapsed values to 0.
        let input: Vec<f32> = (0..64)
            .map(|i| f32::from_bits(0x4000_0000 | (i as u32))) // ~2.0 + tag
            .collect();
        let mut ref_out = alloc::vec![0.0_f32; 64];
        transpose_8x8_scalar_ref(&input, &mut ref_out);

        run_dispatch_parity(|perm| {
            let mut act_out = alloc::vec![0.0_f32; 64];
            transpose_8x8(&input, &mut act_out);
            assert_f32_slice_bit_eq(&ref_out, &act_out, perm, "unique_positions");
        });
    }

    /// Non-finite preservation: NaN, +/-Inf at every position must survive
    /// transposition bit-identically (no arithmetic, so bit-preservation
    /// is required).
    #[test]
    fn transpose_8x8_nonfinite_preserved() {
        // Mix NaN, +Inf, -Inf into 64-element block.
        let mut input = alloc::vec![1.0_f32; 64];
        input[0] = f32::NAN;
        input[7] = f32::INFINITY;
        input[8] = f32::NEG_INFINITY;
        input[15] = f32::NAN;
        input[56] = f32::INFINITY; // bottom row
        input[63] = f32::NEG_INFINITY; // bottom-right corner

        let mut ref_out = alloc::vec![0.0_f32; 64];
        transpose_8x8_scalar_ref(&input, &mut ref_out);

        run_dispatch_parity(|perm| {
            let mut act_out = alloc::vec![0.0_f32; 64];
            transpose_8x8(&input, &mut act_out);
            assert_f32_slice_bit_eq(&ref_out, &act_out, perm, "nonfinite_preserved");
        });
    }

    /// Idempotence: transpose(transpose(x)) == x.
    #[test]
    fn transpose_8x8_self_inverse() {
        let input: Vec<f32> = gen_f32(0xDEAD_BEEF_CAFE_BABE, 64, 100.0);

        run_dispatch_parity(|perm| {
            let mut once = alloc::vec![0.0_f32; 64];
            let mut twice = alloc::vec![0.0_f32; 64];
            transpose_8x8(&input, &mut once);
            transpose_8x8(&once, &mut twice);
            assert_f32_slice_bit_eq(&input, &twice, perm, "self_inverse");
        });
    }
}
