// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Vectorized IDENTITY and DCT2X2 special transforms (8x8, forward + inverse).
//!
//! These are the 8x8-class strategies the AC-strategy search evaluates per
//! block at every lossy effort (IDENTITY/DCT2X2 from e3, joined by
//! DCT4X8/AFV at e6+), so the four functions here run ~260K+ times per 4K
//! encode — they were the scalar remainder of the 23%-self
//! `find_best_16x16` profile bucket (benchmarks/jxl_wall_parity_2026-08-16.md
//! round 4).
//!
//! BIT-IDENTITY CONTRACT (both per-arch and cross-arch): every output value
//! is computed with the EXACT operand association of the scalar reference
//! (`*_scalar` in this file, verbatim copies of the historical
//! `vardct/dct/special.rs` bodies). The vector forms only ROUTE operands
//! with shuffles — element-mapped, no accumulation trees — and perform the
//! adds/subs in the scalar's left-associated order, so the SIMD paths
//! produce bit-identical results to the scalar reference on every arch,
//! and hash-locks are unaffected. Sequential-chain outputs that cannot be
//! vectorized without reassociation (the IDENTITY 16-pixel DC sums and
//! 15-coefficient residual sums, the 4-value Hadamard corners, the final
//! 2x2 DCT pass) STAY SCALAR inside the vector bodies.
//!
//! The `run_dispatch_parity` battery asserts dispatch == scalar bit-exactly
//! on edge-value distributions; any divergence is a bug here, never a
//! tolerance.

/// IDENTITY forward transform — dispatcher.
#[inline]
pub fn identity_from_pixels(pixels: &[f32; 64], coefficients: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            identity_from_pixels_avx2(token, pixels, coefficients);
            return;
        }
    }
    // aarch64: the scalar reference is already competitive here (LLVM
    // auto-vectorizes the fixed-array bodies with zip/uzp); real NEON lane
    // work is queued behind the x64 target (dct4.rs precedent).
    identity_from_pixels_scalar(pixels, coefficients);
}

/// IDENTITY inverse transform — dispatcher.
#[inline]
pub fn identity_to_pixels(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            identity_to_pixels_avx2(token, coefficients, pixels);
            return;
        }
    }
    // aarch64: the scalar reference is already competitive here (LLVM
    // auto-vectorizes the fixed-array bodies with zip/uzp); real NEON lane
    // work is queued behind the x64 target (dct4.rs precedent).
    identity_to_pixels_scalar(coefficients, pixels);
}

/// DCT2X2 forward transform — dispatcher.
#[inline]
pub fn dct2x2_from_pixels(pixels: &[f32; 64], coefficients: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct2x2_from_pixels_avx2(token, pixels, coefficients);
            return;
        }
    }
    // aarch64: the scalar reference is already competitive here (LLVM
    // auto-vectorizes the fixed-array bodies with zip/uzp); real NEON lane
    // work is queued behind the x64 target (dct4.rs precedent).
    dct2x2_from_pixels_scalar(pixels, coefficients);
}

/// DCT2X2 inverse transform — dispatcher.
#[inline]
pub fn dct2x2_to_pixels(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            dct2x2_to_pixels_avx2(token, coefficients, pixels);
            return;
        }
    }
    // aarch64: the scalar reference is already competitive here (LLVM
    // auto-vectorizes the fixed-array bodies with zip/uzp); real NEON lane
    // work is queued behind the x64 target (dct4.rs precedent).
    dct2x2_to_pixels_scalar(coefficients, pixels);
}

// ============================================================================
// Scalar references — verbatim ports of the historical special.rs bodies.
// These define the bit pattern every SIMD variant must reproduce exactly.
// ============================================================================

/// IDENTITY forward, scalar reference.
#[inline]
pub fn identity_from_pixels_scalar(pixels: &[f32; 64], coefficients: &mut [f32; 64]) {
    for y in 0..2usize {
        for x in 0..2usize {
            let mut block_dc = 0.0f32;
            for iy in 0..4 {
                for ix in 0..4 {
                    block_dc += pixels[(y * 4 + iy) * 8 + x * 4 + ix];
                }
            }
            block_dc *= 1.0 / 16.0;

            let ref_pixel = pixels[(y * 4 + 1) * 8 + x * 4 + 1];

            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 1 && iy == 1 {
                        continue;
                    }
                    coefficients[(y + iy * 2) * 8 + x + ix * 2] =
                        pixels[(y * 4 + iy) * 8 + x * 4 + ix] - ref_pixel;
                }
            }

            coefficients[(y + 2) * 8 + x + 2] = coefficients[y * 8 + x];
            coefficients[y * 8 + x] = block_dc;
        }
    }

    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];
    let block11 = coefficients[9];
    coefficients[0] = (block00 + block01 + block10 + block11) * 0.25;
    coefficients[1] = (block00 + block01 - block10 - block11) * 0.25;
    coefficients[8] = (block00 - block01 + block10 - block11) * 0.25;
    coefficients[9] = (block00 - block01 - block10 + block11) * 0.25;
}

/// IDENTITY inverse, scalar reference.
#[inline]
pub fn identity_to_pixels_scalar(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];
    let block11 = coefficients[9];
    let dcs = [
        block00 + block01 + block10 + block11,
        block00 + block01 - block10 - block11,
        block00 - block01 + block10 - block11,
        block00 - block01 - block10 + block11,
    ];

    for y in 0..2usize {
        for x in 0..2usize {
            let block_dc = dcs[y * 2 + x];

            let mut residual_sum = 0.0f32;
            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 0 && iy == 0 {
                        continue;
                    }
                    residual_sum += coefficients[(y + iy * 2) * 8 + x + ix * 2];
                }
            }

            let ref_pixel = block_dc - residual_sum * (1.0 / 16.0);
            pixels[(4 * y + 1) * 8 + 4 * x + 1] = ref_pixel;

            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 1 && iy == 1 {
                        continue;
                    }
                    pixels[(y * 4 + iy) * 8 + x * 4 + ix] =
                        coefficients[(y + iy * 2) * 8 + x + ix * 2] + ref_pixel;
                }
            }

            pixels[y * 4 * 8 + x * 4] = coefficients[(y + 2) * 8 + x + 2] + ref_pixel;
        }
    }
}

#[inline(always)]
fn dct2_first_scalar<const S: usize>(block: &[f32; 64], out: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = block[y * 2 * 8 + x * 2];
            let c01 = block[y * 2 * 8 + x * 2 + 1];
            let c10 = block[(y * 2 + 1) * 8 + x * 2];
            let c11 = block[(y * 2 + 1) * 8 + x * 2 + 1];

            let r00 = (c00 + c01 + c10 + c11) * 0.25;
            let r01 = (c00 + c01 - c10 - c11) * 0.25;
            let r10 = (c00 - c01 + c10 - c11) * 0.25;
            let r11 = (c00 - c01 - c10 + c11) * 0.25;

            out[y * 8 + x] = r00;
            out[y * 8 + num_2x2 + x] = r01;
            out[(y + num_2x2) * 8 + x] = r10;
            out[(y + num_2x2) * 8 + num_2x2 + x] = r11;
        }
    }
}

#[inline(always)]
fn dct2_inplace_scalar<const S: usize>(data: &mut [f32; 64]) {
    let snap = *data;
    dct2_first_scalar::<S>(&snap, data);
}

/// DCT2X2 forward, scalar reference (three hierarchical passes).
#[inline]
pub fn dct2x2_from_pixels_scalar(pixels: &[f32; 64], coefficients: &mut [f32; 64]) {
    dct2_first_scalar::<8>(pixels, coefficients);
    dct2_inplace_scalar::<4>(coefficients);
    dct2_inplace_scalar::<2>(coefficients);
}

#[inline(always)]
fn idct2_inplace_scalar<const S: usize>(data: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    let snap = *data;
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = snap[y * 8 + x];
            let c01 = snap[y * 8 + num_2x2 + x];
            let c10 = snap[(y + num_2x2) * 8 + x];
            let c11 = snap[(y + num_2x2) * 8 + num_2x2 + x];

            let r00 = c00 + c01 + c10 + c11;
            let r01 = c00 + c01 - c10 - c11;
            let r10 = c00 - c01 + c10 - c11;
            let r11 = c00 - c01 - c10 + c11;

            data[y * 2 * 8 + x * 2] = r00;
            data[y * 2 * 8 + x * 2 + 1] = r01;
            data[(y * 2 + 1) * 8 + x * 2] = r10;
            data[(y * 2 + 1) * 8 + x * 2 + 1] = r11;
        }
    }
}

/// DCT2X2 inverse, scalar reference (three inverse passes).
#[inline]
pub fn dct2x2_to_pixels_scalar(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    *pixels = *coefficients;
    idct2_inplace_scalar::<2>(pixels);
    idct2_inplace_scalar::<4>(pixels);
    idct2_inplace_scalar::<8>(pixels);
}

// ============================================================================
// AVX2 bodies. Bridge pattern per dct4.rs: magetypes f32x8 for safe
// load/store/arithmetic, `.raw()` + value-only shuffle intrinsics (safe in
// the #[arcane] target-feature context), `from_m256` back.
// ============================================================================

/// IDENTITY forward, AVX2. Bulk = one permute + one subtract per row
/// (60 of 64 outputs); DC chains / corner / Hadamard stay scalar for
/// bit-exact association with the scalar reference.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn identity_from_pixels_avx2(
    token: archmage::X64V3Token,
    pixels: &[f32; 64],
    coefficients: &mut [f32; 64],
) {
    use core::arch::x86_64::*;
    use magetypes::simd::f32x8;

    // Output row r takes input row (r&1)*4 + (r>>1), lane j takes input
    // lane (j&1)*4 + (j>>1)  ==  the [0,4,1,5,2,6,3,7] interleave.
    let perm_in = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    // Refs: lane j subtracts the ref of sub-block x = j&1, i.e.
    // [ref0, ref1, ref0, ref1, ...] — ref lanes 1 and 5 of input row
    // (y*4 + 1).
    let perm_ref = _mm256_setr_epi32(1, 5, 1, 5, 1, 5, 1, 5);

    let p_row = |r: usize| f32x8::from_slice(token, &pixels[r * 8..]);
    let ref0 = _mm256_permutevar8x32_ps(p_row(1).raw(), perm_ref);
    let ref1 = _mm256_permutevar8x32_ps(p_row(5).raw(), perm_ref);

    for r in 0..8usize {
        let y = r & 1;
        let iy = r >> 1;
        let src = p_row(y * 4 + iy).raw();
        let permuted = _mm256_permutevar8x32_ps(src, perm_in);
        let refs = if y == 0 { ref0 } else { ref1 };
        let out = f32x8::from_m256(token, _mm256_sub_ps(permuted, refs));
        out.store(sub_array_mut(coefficients, r * 8));
    }

    // Scalar fixups, in the scalar reference's sub-block order.
    for y in 0..2usize {
        for x in 0..2usize {
            let mut block_dc = 0.0f32;
            for iy in 0..4 {
                for ix in 0..4 {
                    block_dc += pixels[(y * 4 + iy) * 8 + x * 4 + ix];
                }
            }
            block_dc *= 1.0 / 16.0;
            coefficients[(y + 2) * 8 + x + 2] = coefficients[y * 8 + x];
            coefficients[y * 8 + x] = block_dc;
        }
    }
    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];
    let block11 = coefficients[9];
    coefficients[0] = (block00 + block01 + block10 + block11) * 0.25;
    coefficients[1] = (block00 + block01 - block10 - block11) * 0.25;
    coefficients[8] = (block00 - block01 + block10 - block11) * 0.25;
    coefficients[9] = (block00 - block01 - block10 + block11) * 0.25;
}

/// IDENTITY inverse, AVX2. Bulk = one permute + one add per row; Hadamard,
/// residual chains, and the two per-sub-block position fixups stay scalar.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn identity_to_pixels_avx2(
    token: archmage::X64V3Token,
    coefficients: &[f32; 64],
    pixels: &mut [f32; 64],
) {
    use core::arch::x86_64::*;
    use magetypes::simd::f32x8;

    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];
    let block11 = coefficients[9];
    let dcs = [
        block00 + block01 + block10 + block11,
        block00 + block01 - block10 - block11,
        block00 - block01 + block10 - block11,
        block00 - block01 - block10 + block11,
    ];

    // Refs per sub-block, scalar residual chains (exact order).
    let mut refs = [0.0f32; 4];
    for y in 0..2usize {
        for x in 0..2usize {
            let mut residual_sum = 0.0f32;
            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 0 && iy == 0 {
                        continue;
                    }
                    residual_sum += coefficients[(y + iy * 2) * 8 + x + ix * 2];
                }
            }
            refs[y * 2 + x] = dcs[y * 2 + x] - residual_sum * (1.0 / 16.0);
        }
    }

    // Pixel row pr reads coefficient row (pr>>2)&1 ... : y = pr/4, iy = pr%4,
    // source row y + iy*2; lane px reads coefficient lane
    // (px>>2) + (px&3)*2  ==  the [0,2,4,6,1,3,5,7] deinterleave.
    let perm_out = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let c_row = |r: usize| f32x8::from_slice(token, &coefficients[r * 8..]);
    // Ref vector for pixel rows of sub-block row y: [ref(y,0) x4 | ref(y,1) x4].
    let refs_y0 = _mm256_setr_ps(
        refs[0], refs[0], refs[0], refs[0], refs[1], refs[1], refs[1], refs[1],
    );
    let refs_y1 = _mm256_setr_ps(
        refs[2], refs[2], refs[2], refs[2], refs[3], refs[3], refs[3], refs[3],
    );

    for pr in 0..8usize {
        let y = pr >> 2;
        let iy = pr & 3;
        let src = c_row(y + iy * 2).raw();
        let permuted = _mm256_permutevar8x32_ps(src, perm_out);
        let rv = if y == 0 { refs_y0 } else { refs_y1 };
        let out = f32x8::from_m256(token, _mm256_add_ps(permuted, rv));
        out.store(sub_array_mut(pixels, pr * 8));
    }

    // Scalar position fixups, in the scalar reference's sub-block order.
    for y in 0..2usize {
        for x in 0..2usize {
            let ref_pixel = refs[y * 2 + x];
            pixels[(4 * y + 1) * 8 + 4 * x + 1] = ref_pixel;
            pixels[y * 4 * 8 + x * 4] = coefficients[(y + 2) * 8 + x + 2] + ref_pixel;
        }
    }
}

/// DCT2X2 forward, AVX2: passes S=8 and S=4 vectorized (butterflies in the
/// scalar reference's left-associated order via even/odd permutes), the
/// final S=2 pass scalar on 4 values.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn dct2x2_from_pixels_avx2(
    token: archmage::X64V3Token,
    pixels: &[f32; 64],
    coefficients: &mut [f32; 64],
) {
    use core::arch::x86_64::*;
    use magetypes::simd::f32x8;

    let quarter = _mm256_set1_ps(0.25);
    let perm_even = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    let perm_odd = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);

    // ── Pass S=8: pixels -> coefficients ──
    let mut rows_04: [__m256; 4] = [_mm256_setzero_ps(); 4]; // out rows 0..3
    for y in 0..4usize {
        let a = f32x8::from_slice(token, &pixels[(2 * y) * 8..]).raw();
        let b = f32x8::from_slice(token, &pixels[(2 * y + 1) * 8..]).raw();
        let ea = _mm256_permutevar8x32_ps(a, perm_even);
        let oa = _mm256_permutevar8x32_ps(a, perm_odd);
        let eb = _mm256_permutevar8x32_ps(b, perm_even);
        let ob = _mm256_permutevar8x32_ps(b, perm_odd);
        // Scalar association: ((c00 + c01) + c10) + c11, etc.
        let ha = _mm256_add_ps(ea, oa); // c00+c01
        let hd = _mm256_sub_ps(ea, oa); // c00-c01
        let r00 = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(ha, eb), ob), quarter);
        let r01 = _mm256_mul_ps(_mm256_sub_ps(_mm256_sub_ps(ha, eb), ob), quarter);
        let r10 = _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(hd, eb), ob), quarter);
        let r11 = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(hd, eb), ob), quarter);
        // out row y = [r00 lanes 0..3 | r01 lanes 0..3] — both perms
        // duplicated their selection into the high half, so taking
        // r00.lo128 | r01.hi128 yields exactly that.
        rows_04[y] = _mm256_permute2f128_ps::<0x30>(r00, r01);
        let lower = f32x8::from_m256(token, _mm256_permute2f128_ps::<0x30>(r10, r11));
        lower.store(sub_array_mut(coefficients, (y + 4) * 8));
    }

    // ── Pass S=4: rows 0..3 lanes 0..3 in registers; lanes 4..7 preserved ──
    // Cells read row pairs (2y', 2y'+1) at lane pairs (2x', 2x'+1), x' in 0..2.
    let perm_even4 = _mm256_setr_epi32(0, 2, 0, 2, 0, 2, 0, 2);
    let perm_odd4 = _mm256_setr_epi32(1, 3, 1, 3, 1, 3, 1, 3);
    let mut out_r: [__m256; 4] = rows_04;
    for yp in 0..2usize {
        let a = rows_04[2 * yp];
        let b = rows_04[2 * yp + 1];
        let ea = _mm256_permutevar8x32_ps(a, perm_even4);
        let oa = _mm256_permutevar8x32_ps(a, perm_odd4);
        let eb = _mm256_permutevar8x32_ps(b, perm_even4);
        let ob = _mm256_permutevar8x32_ps(b, perm_odd4);
        let ha = _mm256_add_ps(ea, oa);
        let hd = _mm256_sub_ps(ea, oa);
        let r00 = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(ha, eb), ob), quarter);
        let r01 = _mm256_mul_ps(_mm256_sub_ps(_mm256_sub_ps(ha, eb), ob), quarter);
        let r10 = _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(hd, eb), ob), quarter);
        let r11 = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(hd, eb), ob), quarter);
        // Combined row: lanes [r00_0, r00_1, r01_0, r01_1] then preserve 4..7.
        // shuffle_ps imm 0x44 picks [a0,a1,b0,b1] per 128-lane.
        let top = _mm256_shuffle_ps::<0x44>(r00, r01);
        let bot = _mm256_shuffle_ps::<0x44>(r10, r11);
        // Preserve lanes 4..7 from the pass-1 rows (blend mask 0xF0 keeps b).
        out_r[yp] = _mm256_blend_ps::<0xF0>(top, rows_04[yp]);
        out_r[yp + 2] = _mm256_blend_ps::<0xF0>(bot, rows_04[yp + 2]);
    }
    for (y, row) in out_r.iter().enumerate() {
        f32x8::from_m256(token, *row).store(sub_array_mut(coefficients, y * 8));
    }

    // ── Pass S=2: scalar on the stored array (exact reference order) ──
    let c00 = coefficients[0];
    let c01 = coefficients[1];
    let c10 = coefficients[8];
    let c11 = coefficients[9];
    coefficients[0] = (c00 + c01 + c10 + c11) * 0.25;
    coefficients[1] = (c00 + c01 - c10 - c11) * 0.25;
    coefficients[8] = (c00 - c01 + c10 - c11) * 0.25;
    coefficients[9] = (c00 - c01 - c10 + c11) * 0.25;
}

/// DCT2X2 inverse, AVX2: pass S=2 scalar, S=4 and S=8 vectorized (no 0.25
/// scale on the inverse butterflies; interleave scatter via unpacks).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn dct2x2_to_pixels_avx2(
    token: archmage::X64V3Token,
    coefficients: &[f32; 64],
    pixels: &mut [f32; 64],
) {
    use core::arch::x86_64::*;
    use magetypes::simd::f32x8;

    *pixels = *coefficients;

    // ── Pass S=2: scalar (exact reference order) ──
    let c00 = pixels[0];
    let c01 = pixels[1];
    let c10 = pixels[8];
    let c11 = pixels[9];
    pixels[0] = c00 + c01 + c10 + c11;
    pixels[1] = c00 + c01 - c10 - c11;
    pixels[8] = c00 - c01 + c10 - c11;
    pixels[9] = c00 - c01 - c10 + c11;

    // ── Pass S=4: rows 0..1 x quadrant lanes {x} / {x+2}, write rows 0..3
    // lanes 0..3, preserve lanes 4..7 ──
    {
        let r0 = f32x8::from_slice(token, &pixels[0..]).raw();
        let r1 = f32x8::from_slice(token, &pixels[8..]).raw();
        let r2 = f32x8::from_slice(token, &pixels[16..]).raw();
        let r3 = f32x8::from_slice(token, &pixels[24..]).raw();
        // Swap lane pairs (0,1)<->(2,3) within each 128 half: imm 0b01001110.
        let swap2 = |v: __m256| _mm256_permute_ps::<0b0100_1110>(v);
        let rows_in = [r0, r1];
        let rows_lo = [r2, r3];
        let mut out = [r0, r1, r2, r3];
        for y in 0..2usize {
            let a = rows_in[y];
            let b = rows_lo[y];
            let ha = swap2(a); // lane x holds a[x+2]
            let hb = swap2(b);
            // ((c00 + c01) + c10) + c11 in scalar order:
            let s = _mm256_add_ps(a, ha); // c00+c01 (lanes 0..1 valid)
            let d = _mm256_sub_ps(a, ha); // c00-c01
            let r00 = _mm256_add_ps(_mm256_add_ps(s, b), hb);
            let r01 = _mm256_sub_ps(_mm256_sub_ps(s, b), hb);
            let r10 = _mm256_sub_ps(_mm256_add_ps(d, b), hb);
            let r11 = _mm256_add_ps(_mm256_sub_ps(d, b), hb);
            // Interleave r00/r01 lanes 0..1 -> [00_0, 01_0, 00_1, 01_1]:
            let top = _mm256_unpacklo_ps(r00, r01);
            let bot = _mm256_unpacklo_ps(r10, r11);
            out[2 * y] = _mm256_blend_ps::<0xF0>(top, out[2 * y]);
            out[2 * y + 1] = _mm256_blend_ps::<0xF0>(bot, out[2 * y + 1]);
        }
        for (i, row) in out.iter().enumerate() {
            f32x8::from_m256(token, *row).store(sub_array_mut(pixels, i * 8));
        }
    }

    // ── Pass S=8: quadrant halves are 128-bit register halves ──
    {
        let mut rows: [__m256; 8] = [_mm256_setzero_ps(); 8];
        for (i, row) in rows.iter_mut().enumerate() {
            *row = f32x8::from_slice(token, &pixels[i * 8..]).raw();
        }
        for y in 0..4usize {
            let a = rows[y];
            let b = rows[y + 4];
            let ha = _mm256_permute2f128_ps::<0x01>(a, a); // halves swapped
            let hb = _mm256_permute2f128_ps::<0x01>(b, b);
            let s = _mm256_add_ps(a, ha);
            let d = _mm256_sub_ps(a, ha);
            let r00 = _mm256_add_ps(_mm256_add_ps(s, b), hb);
            let r01 = _mm256_sub_ps(_mm256_sub_ps(s, b), hb);
            let r10 = _mm256_sub_ps(_mm256_add_ps(d, b), hb);
            let r11 = _mm256_add_ps(_mm256_sub_ps(d, b), hb);
            // Interleave lanes 0..3 of r00/r01 across the full 8:
            let il = _mm256_unpacklo_ps(r00, r01); // [00_0,01_0,00_1,01_1 | ..hi..]
            let ih = _mm256_unpackhi_ps(r00, r01); // [00_2,01_2,00_3,01_3 | ..hi..]
            let top = _mm256_permute2f128_ps::<0x20>(il, ih); // [il.lo | ih.lo]
            let il2 = _mm256_unpacklo_ps(r10, r11);
            let ih2 = _mm256_unpackhi_ps(r10, r11);
            let bot = _mm256_permute2f128_ps::<0x20>(il2, ih2);
            f32x8::from_m256(token, top).store(sub_array_mut(pixels, (2 * y) * 8));
            f32x8::from_m256(token, bot).store(sub_array_mut(pixels, (2 * y + 1) * 8));
        }
    }
}

/// Borrow output row `r*8..r*8+8` as a fixed-size array (bounds-check-free
/// store target; `base + 8 <= 64` for every caller).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn sub_array_mut(buf: &mut [f32; 64], base: usize) -> &mut [f32; 8] {
    (&mut buf[base..base + 8]).try_into().unwrap()
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn test_blocks() -> Vec<[f32; 64]> {
        // Edge-value battery + deterministic pseudo-random blocks.
        let mut blocks: Vec<[f32; 64]> = Vec::new();
        blocks.push([0.0f32; 64]);
        blocks.push([1.0f32; 64]);
        blocks.push(core::array::from_fn(|i| i as f32 - 32.0));
        blocks.push(core::array::from_fn(|i| {
            if i % 2 == 0 { 1e-30 } else { -3.5e4 }
        }));
        let mut state: u32 = 0x1234_5678;
        for _ in 0..64 {
            let mut b = [0.0f32; 64];
            for v in b.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                // Spread across magnitudes incl. negatives and denormal-ish.
                let m = (state >> 9) as f32 / (1 << 23) as f32; // [0,1)
                let e = ((state >> 27) as i32) - 8; // 2^-8 .. 2^7
                *v = (m - 0.5) * (2.0f32).powi(e) * 300.0;
            }
            blocks.push(b);
        }
        blocks
    }

    #[test]
    fn identity_fwd_dispatch_bit_exact_vs_scalar() {
        for (bi, b) in test_blocks().iter().enumerate() {
            let mut want = [0.0f32; 64];
            let mut got = [7.0f32; 64];
            identity_from_pixels_scalar(b, &mut want);
            identity_from_pixels(b, &mut got);
            for i in 0..64 {
                assert_eq!(
                    want[i].to_bits(),
                    got[i].to_bits(),
                    "identity fwd block {bi} coeff {i}: {} vs {}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn identity_inv_dispatch_bit_exact_vs_scalar() {
        for (bi, b) in test_blocks().iter().enumerate() {
            let mut want = [0.0f32; 64];
            let mut got = [7.0f32; 64];
            identity_to_pixels_scalar(b, &mut want);
            identity_to_pixels(b, &mut got);
            for i in 0..64 {
                assert_eq!(
                    want[i].to_bits(),
                    got[i].to_bits(),
                    "identity inv block {bi} px {i}: {} vs {}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn dct2x2_fwd_dispatch_bit_exact_vs_scalar() {
        for (bi, b) in test_blocks().iter().enumerate() {
            let mut want = [0.0f32; 64];
            let mut got = [7.0f32; 64];
            dct2x2_from_pixels_scalar(b, &mut want);
            dct2x2_from_pixels(b, &mut got);
            for i in 0..64 {
                assert_eq!(
                    want[i].to_bits(),
                    got[i].to_bits(),
                    "dct2x2 fwd block {bi} coeff {i}: {} vs {}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn dct2x2_inv_dispatch_bit_exact_vs_scalar() {
        for (bi, b) in test_blocks().iter().enumerate() {
            let mut want = [0.0f32; 64];
            let mut got = [7.0f32; 64];
            dct2x2_to_pixels_scalar(b, &mut want);
            dct2x2_to_pixels(b, &mut got);
            for i in 0..64 {
                assert_eq!(
                    want[i].to_bits(),
                    got[i].to_bits(),
                    "dct2x2 inv block {bi} px {i}: {} vs {}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    /// Fwd→inv roundtrip must reconstruct pixels (the transforms are exact
    /// inverses up to float rounding; assert tight closeness as a sanity
    /// check that the vector forms implement the RIGHT transform, not just
    /// a self-consistent one).
    #[test]
    fn special_transforms_roundtrip_sanity() {
        for b in test_blocks().iter() {
            let mut c = [0.0f32; 64];
            let mut p = [0.0f32; 64];
            identity_from_pixels(b, &mut c);
            identity_to_pixels(&c, &mut p);
            // Reconstruction error scales with the BLOCK's magnitude (ref
            // pixels and DC averages mix all 64 values), not per-pixel.
            let block_scale = b.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            for i in 0..64 {
                assert!(
                    (b[i] - p[i]).abs() <= block_scale * 1e-4,
                    "identity roundtrip px {i}: {} vs {}",
                    b[i],
                    p[i]
                );
            }
        }
    }
}
