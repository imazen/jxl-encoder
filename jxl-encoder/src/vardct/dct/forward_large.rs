// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Forward DCT transforms for sizes 32x32 and larger.

use super::constants::*;
use super::forward::dct1d_16;
use super::inverse::{idct1d_4, idct1d_8};
use crate::vardct::common::{as_array_mut, as_array_ref};

pub fn dct1d_32(mem: &mut [f32]) {
    let mut tmp = [0.0f32; 32];

    // AddReverse for first half
    for i in 0..16 {
        tmp[i] = mem[i] + mem[31 - i];
    }
    // SubReverse for second half
    for i in 0..16 {
        tmp[16 + i] = mem[i] - mem[31 - i];
    }

    // DCT on first half
    dct1d_16(&mut tmp[0..16]);

    // Multiply second half by WcMultipliers
    for i in 0..16 {
        tmp[16 + i] *= WC_MULTIPLIERS_32[i];
    }

    // DCT on second half
    dct1d_16(&mut tmp[16..32]);

    // B transform on second half
    tmp[16] = SQRT2 * tmp[16] + tmp[17];
    for i in 1..15 {
        tmp[16 + i] += tmp[16 + i + 1];
    }

    // InverseEvenOdd: interleave
    for i in 0..16 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[16 + i];
    }
}

/// Compute scaled 32x32 DCT (32 rows, 32 columns).
///
/// Input: 32x32 block in row-major order (1024 floats)
/// Output: 32x32 DCT coefficients
///
/// Like `dct_8x8()` and `dct_16x16()`, there is NO final transpose for square blocks.
/// C++ `ComputeScaledDCT<32,32>` takes the ROWS >= COLS branch (no final transpose).
pub fn dct_32x32(input: &[f32; 1024], output: &mut [f32; 1024]) {
    jxl_simd::dct_32x32(input, output);
}

/// Extract DC values from 32x32 DCT coefficients.
/// Returns 16 DC values (for the 16 covered 8x8 blocks) in row-major 4x4 order.
///
/// The LLF region is 4x4 coefficients at positions `[r*32+c]` for r,c in 0..4
/// in the 32x32 layout (stride 32). We apply `DCTTotalResampleScale<32, 4>` to
/// each dimension, then a 4x4 IDCT to get the 16 DC values.
///
/// C++ uses `ReinterpretingIDCT<32, 32, 4, 4, 4, 4>` with the ROWS >= COLS branch
/// (since ROWS=COLS=4).
pub fn dc_from_dct_32x32(coeffs: &[f32; 1024]) -> [f32; 16] {
    // Step 1: Extract 4x4 LLF and apply resample scales.
    // The forward DCT32x32 scaled by 1/1024. The 4x4 IDCT will apply 4*4=16 scaling,
    // so we need an additional 1024/16 = 64 to get back to spatial values.
    let mut block = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            block[iy * 4 + ix] = coeffs[iy * 32 + ix]
                * DCT_RESAMPLE_SCALE_32_TO_4[iy]
                * DCT_RESAMPLE_SCALE_32_TO_4[ix]
                * 16.0; // Compensate for forward/inverse scaling mismatch
        }
    }

    // Step 2: 4x4 IDCT matching C++ ComputeScaledIDCT<4,4> (ROWS >= COLS):
    //   IDCT rows → transpose → IDCT rows.
    // Using matched idct1d_4 that exactly reverses our forward dct1d_4.

    // IDCT rows (in-place)
    for iy in 0..4 {
        idct1d_4(&mut block[iy * 4..(iy + 1) * 4]);
    }

    // Transpose 4x4
    let mut transposed = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            transposed[ix * 4 + iy] = block[iy * 4 + ix];
        }
    }

    // IDCT rows (on transposed data = columns of original)
    for iy in 0..4 {
        idct1d_4(&mut transposed[iy * 4..(iy + 1) * 4]);
    }

    transposed
}

// =============================================================================
// DCT32x16 and DCT16x32 support
// =============================================================================

/// Compute scaled 32x16 DCT (32 rows, 16 columns).
///
/// Input: 32x16 block in row-major order (512 floats)
/// Output: 32x16 DCT coefficients in 16×32 layout (stride 32)
///
/// C++ `ComputeScaledDCT<32,16>` takes the ROWS >= COLS branch (no final transpose).
pub fn dct_32x16(input: &[f32; 512], output: &mut [f32; 512]) {
    jxl_simd::dct_32x16(input, output);
}

/// Compute scaled 16x32 DCT (16 rows, 32 columns).
///
/// Input: 16x32 block in row-major order (512 floats)
/// Output: 16x32 DCT coefficients
///
/// C++ `ComputeScaledDCT<16,32>` takes the ROWS < COLS branch (includes final transpose).
pub fn dct_16x32(input: &[f32; 512], output: &mut [f32; 512]) {
    jxl_simd::dct_16x32(input, output);
}

/// Extract DC values from 32x16 DCT coefficients.
/// Returns 8 DC values (for the 8 covered 8x8 blocks) in row-major 4x2 order.
///
/// DCT32x16 output is 16×32 (stride 32, no final transpose):
///   row = horiz freq (16-dim, 2 blocks), col = vert freq (32-dim, 4 blocks)
/// LLF region is 2×4 at positions `[r*32+c]` for r in 0..2, c in 0..4.
/// We apply `DCTTotalResampleScale<16, 2>` to rows and `<32, 4>` to columns,
/// then a 2×4 IDCT (skipping final transpose to produce 4×2 spatial output).
pub fn dc_from_dct_32x16(coeffs: &[f32; 512]) -> [f32; 8] {
    // Extract 2×4 LLF and apply resample scales.
    // Compensation factor = 4.0: idct1d_4 has DC gain 1/4, inline 2-point has DC gain 1.
    // Total LLF IDCT DC gain = 1/4, so we compensate with ×4.
    let mut block = [0.0f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            block[iy * 4 + ix] = coeffs[iy * 32 + ix]
                * DCT_RESAMPLE_SCALE_16_TO_2[iy]
                * DCT_RESAMPLE_SCALE_32_TO_4[ix]
                * 4.0;
        }
    }

    // 2×4 IDCT (ROWS=2 < COLS=4): IDCT rows → transpose → IDCT rows.
    // Skip final transpose: the 2×4 LLF has rows=horiz freq, cols=vert freq.
    // After IDCT without final transpose, result is 4×2 = [block_row][block_col].

    // IDCT on 4-element rows (2 rows) — converts vert freq → spatial block row
    idct1d_4(&mut block[0..4]);
    idct1d_4(&mut block[4..8]);

    // Transpose 2×4 → 4×2
    let mut transposed = [0.0f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            transposed[ix * 2 + iy] = block[iy * 4 + ix];
        }
    }

    // IDCT on 2-element rows (4 rows) — converts horiz freq → spatial block col
    for iy in 0..4 {
        let a = transposed[iy * 2];
        let b = transposed[iy * 2 + 1];
        transposed[iy * 2] = a + b;
        transposed[iy * 2 + 1] = a - b;
    }

    // Result is 4×2 = [block_row * 2 + block_col], matching caller
    transposed
}

/// Extract DC values from 16x32 DCT coefficients.
/// Returns 8 DC values (for the 8 covered 8x8 blocks) in row-major 2x4 order.
///
/// The LLF region is 2x4 coefficients. We apply `DCTTotalResampleScale<16, 2>` to rows
/// and `DCTTotalResampleScale<32, 4>` to columns, then a 2x4 IDCT.
pub fn dc_from_dct_16x32(coeffs: &[f32; 512]) -> [f32; 8] {
    // Extract 2×4 LLF and apply resample scales.
    // Compensation factor = 4.0: idct1d_4 has DC gain 1/4, inline 2-point has DC gain 1.
    // Total LLF IDCT DC gain = 1/4, so we compensate with ×4.
    let mut block = [0.0f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            block[iy * 4 + ix] = coeffs[iy * 32 + ix]
                * DCT_RESAMPLE_SCALE_16_TO_2[iy]
                * DCT_RESAMPLE_SCALE_32_TO_4[ix]
                * 4.0;
        }
    }

    // 2×4 IDCT: Since ROWS=2 < COLS=4, this is ROWS < COLS branch
    // IDCT rows -> transpose -> IDCT rows -> transpose back

    // IDCT on 4-element rows (2 rows)
    idct1d_4(&mut block[0..4]);
    idct1d_4(&mut block[4..8]);

    // Transpose 2x4 -> 4x2
    let mut transposed = [0.0f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            transposed[ix * 2 + iy] = block[iy * 4 + ix];
        }
    }

    // IDCT on 2-element rows (4 rows)
    for iy in 0..4 {
        let a = transposed[iy * 2];
        let b = transposed[iy * 2 + 1];
        transposed[iy * 2] = a + b;
        transposed[iy * 2 + 1] = a - b;
    }

    // Transpose back 4x2 -> 2x4
    let mut result = [0.0f32; 8];
    for iy in 0..4 {
        for ix in 0..2 {
            result[ix * 4 + iy] = transposed[iy * 2 + ix];
        }
    }

    result
}

pub fn dct1d_64(mem: &mut [f32]) {
    let mut tmp = [0.0f32; 64];

    // AddReverse for first half
    for i in 0..32 {
        tmp[i] = mem[i] + mem[63 - i];
    }
    // SubReverse for second half
    for i in 0..32 {
        tmp[32 + i] = mem[i] - mem[63 - i];
    }

    // DCT on first half
    dct1d_32(&mut tmp[0..32]);

    // Multiply second half by WcMultipliers
    for i in 0..32 {
        tmp[32 + i] *= WC_MULTIPLIERS_64[i];
    }

    // DCT on second half
    dct1d_32(&mut tmp[32..64]);

    // B transform on second half
    tmp[32] = SQRT2 * tmp[32] + tmp[33];
    for i in 1..31 {
        tmp[32 + i] += tmp[32 + i + 1];
    }

    // InverseEvenOdd: interleave
    for i in 0..32 {
        mem[2 * i] = tmp[i];
        mem[2 * i + 1] = tmp[32 + i];
    }
}

/// Compute scaled 64x64 DCT (64 rows, 64 columns).
///
/// Input: 64x64 block in row-major order (4096 floats)
/// Output: 64x64 DCT coefficients
///
/// NO final transpose for square blocks (ROWS >= COLS branch).
pub fn dct_64x64(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 4096);
    debug_assert!(output.len() >= 4096);

    jxl_simd::dct_64x64(as_array_ref(input, 0), as_array_mut(output, 0));
}

/// Compute scaled 64x32 DCT (64 rows, 32 columns).
///
/// Input: 64x32 block in row-major order (2048 floats)
/// Output: DCT coefficients in 32x64 layout (stride 64)
///
/// C++ `ComputeScaledDCT<64,32>` takes the ROWS >= COLS branch (no final transpose).
pub fn dct_64x32(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 2048);
    debug_assert!(output.len() >= 2048);

    jxl_simd::dct_64x32(as_array_ref(input, 0), as_array_mut(output, 0));
}

/// Compute scaled 32x64 DCT (32 rows, 64 columns).
///
/// Input: 32x64 block in row-major order (2048 floats)
/// Output: DCT coefficients
///
/// C++ `ComputeScaledDCT<32,64>` takes the ROWS < COLS branch (WITH final transpose).
pub fn dct_32x64(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 2048);
    debug_assert!(output.len() >= 2048);

    jxl_simd::dct_32x64(as_array_ref(input, 0), as_array_mut(output, 0));
}

/// Extract DC values from 64x64 DCT coefficients.
/// Returns 64 DC values (for the 64 covered 8x8 blocks) in row-major 8x8 order.
///
/// The LLF region is 8x8 coefficients at positions `[r*64+c]` for r,c in 0..8
/// in the 64x64 layout (stride 64). We apply `DCTResampleScale<64, 8>` to
/// each dimension, then an 8x8 IDCT.
pub fn dc_from_dct_64x64(coeffs: &[f32]) -> [f32; 64] {
    debug_assert!(coeffs.len() >= 4096);

    // Step 1: Extract 8x8 LLF and apply resample scales.
    // Forward DCT64x64 scaled by 1/4096. The 8x8 IDCT will apply 8*8=64 scaling,
    // so we need 4096/64 = 64 factor.
    let mut block = [0.0f32; 64];
    for iy in 0..8 {
        for ix in 0..8 {
            block[iy * 8 + ix] = coeffs[iy * 64 + ix]
                * DCT_RESAMPLE_SCALE_64_TO_8[iy]
                * DCT_RESAMPLE_SCALE_64_TO_8[ix];
        }
    }

    // Step 2: 8x8 IDCT matching ComputeScaledIDCT<8,8> (ROWS >= COLS):
    //   IDCT rows → transpose → IDCT rows.

    // IDCT rows
    for iy in 0..8 {
        idct1d_8(&mut block[iy * 8..(iy + 1) * 8]);
    }

    // Transpose 8x8
    let mut transposed = [0.0f32; 64];
    for iy in 0..8 {
        for ix in 0..8 {
            transposed[ix * 8 + iy] = block[iy * 8 + ix];
        }
    }

    // IDCT rows
    for iy in 0..8 {
        idct1d_8(&mut transposed[iy * 8..(iy + 1) * 8]);
    }

    transposed
}

/// Extract DC values from 64x32 DCT coefficients.
/// Returns 32 DC values (for the 32 covered 8x8 blocks) in row-major 8x4 order.
///
/// DCT64x32 output is 32×64 (stride 64, no final transpose):
///   row = horiz freq (32-dim, 4 blocks), col = vert freq (64-dim, 8 blocks)
/// LLF region is 4×8 at positions `[r*64+c]` for r in 0..4, c in 0..8.
/// We apply `DCTTotalResampleScale<32, 4>` to rows and `<64, 8>` to columns,
/// then a 4×8 IDCT (skipping final transpose to produce 8×4 spatial output).
///
/// Coverage: 8 block rows × 4 block cols. DC output is 8 rows × 4 cols.
pub fn dc_from_dct_64x32(coeffs: &[f32]) -> [f32; 32] {
    debug_assert!(coeffs.len() >= 2048);

    // Extract 4×8 LLF from the 32×64 layout (stride 64)
    let mut block = [0.0f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            block[iy * 8 + ix] = coeffs[iy * 64 + ix]
                * DCT_RESAMPLE_SCALE_32_TO_4[iy]
                * DCT_RESAMPLE_SCALE_64_TO_8[ix]
                * 4.0;
        }
    }

    // 4×8 IDCT (ROWS=4 < COLS=8): IDCT rows → transpose → IDCT rows.
    // Skip final transpose: the 4×8 LLF has rows=horiz freq, cols=vert freq.
    // After IDCT without final transpose, result is 8×4 = [block_row][block_col].

    // IDCT on 8-element rows (4 rows) — converts vert freq → spatial block row
    for iy in 0..4 {
        idct1d_8(&mut block[iy * 8..(iy + 1) * 8]);
    }

    // Transpose 4×8 → 8×4
    let mut transposed = [0.0f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            transposed[ix * 4 + iy] = block[iy * 8 + ix];
        }
    }

    // IDCT on 4-element rows (8 rows) — converts horiz freq → spatial block col
    for iy in 0..8 {
        idct1d_4(&mut transposed[iy * 4..(iy + 1) * 4]);
    }

    // Result is 8×4 = [block_row * 4 + block_col], matching caller
    transposed
}

/// Extract DC values from 32x64 DCT coefficients.
/// Returns 32 DC values (for the 32 covered 8x8 blocks) in row-major 8x4 order.
///
/// Coverage: 8 cols × 4 rows of 8x8 blocks. DC output is 4 rows × 8 cols.
/// After dct_32x64's final transpose, coefficients are in stride-64 layout.
/// CoefficientLayout: cx=8 >= cy=4, so stride = cx*8 = 64.
pub fn dc_from_dct_32x64(coeffs: &[f32]) -> [f32; 32] {
    debug_assert!(coeffs.len() >= 2048);

    // Extract 4x8 LLF from stride-64 layout
    let mut block = [0.0f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            block[iy * 8 + ix] = coeffs[iy * 64 + ix]
                * DCT_RESAMPLE_SCALE_32_TO_4[iy]
                * DCT_RESAMPLE_SCALE_64_TO_8[ix]
                * 4.0;
        }
    }

    // 4x8 IDCT: ROWS=4 < COLS=8, so ROWS < COLS branch:
    // IDCT rows -> transpose -> IDCT rows -> transpose back

    // IDCT on 8-element rows (4 rows)
    for iy in 0..4 {
        idct1d_8(&mut block[iy * 8..(iy + 1) * 8]);
    }

    // Transpose 4x8 -> 8x4
    let mut transposed = [0.0f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            transposed[ix * 4 + iy] = block[iy * 8 + ix];
        }
    }

    // IDCT on 4-element rows (8 rows)
    for iy in 0..8 {
        idct1d_4(&mut transposed[iy * 4..(iy + 1) * 4]);
    }

    // Transpose back 8x4 -> 4x8
    let mut result = [0.0f32; 32];
    for iy in 0..8 {
        for ix in 0..4 {
            result[ix * 8 + iy] = transposed[iy * 4 + ix];
        }
    }

    result
}

// =============================================================================
// libjxl pass-order variants (W45-RECON part 15)
//
// Same construction as `forward.rs`'s `*_lj` wrappers: libjxl
// `ComputeScaledDCT<R, C>` transforms the storage-row direction first.
// Rectangular shapes call the transposed-shape sibling on the transposed
// input (which yields the identical output layout); square shapes wrap as
// transpose-in → kernel → transpose-out.
// =============================================================================

/// libjxl `IDCT1DImpl<4>` on one SIMD lane group (dct-inl.h:219-232).
///
/// `ForwardEvenOdd` de-interleaves `(f0, f1, f2, f3)` to
/// `(f0, f2 | f1, f3)`, `IDCT1DImpl<2>` runs on the even half,
/// `BTranspose<2>` then `IDCT1DImpl<2>` on the odd half, and
/// `MultiplyAndAdd<4>` emits `MulAdd`/`NegMulAdd` fused ops — single
/// rounding, so `f32::mul_add` is required for bit parity.
#[inline(always)]
fn idct1d_4_lj_lane(f0: f32, f1: f32, f2: f32, f3: f32) -> [f32; 4] {
    let te0 = f0 + f2;
    let te1 = f0 - f2;
    // BTranspose<2>: coeff[1] += coeff[0], then coeff[0] *= sqrt2.
    let t3 = f3 + f1;
    let t2 = f1 * SQRT2;
    let to0 = t2 + t3;
    let to1 = t2 - t3;
    // MultiplyAndAdd<4>: out[i] = MulAdd(mul[i], odd[i], even[i]),
    // out[3-i] = NegMulAdd(mul[i], odd[i], even[i]).
    [
        WC_MULTIPLIERS_4[0].mul_add(to0, te0),
        WC_MULTIPLIERS_4[1].mul_add(to1, te1),
        (-WC_MULTIPLIERS_4[1]).mul_add(to1, te1),
        (-WC_MULTIPLIERS_4[0]).mul_add(to0, te0),
    ]
}

/// Strict libjxl-parity `dc_from_dct_32x32` (W45-RECON part 20).
///
/// Ports `ReinterpretingIDCT<32, 32, 4, 4, 4, 4>` +
/// `ComputeScaledIDCT<4, 4>` in libjxl's SIMD-lane evaluation order:
/// the ROWS >= COLS branch transforms down each *column* of the scaled
/// LLF block (columns are the SIMD lanes), transposes, then transforms
/// down each column of the transpose — i.e. each *row* of the
/// first-pass intermediate is written into an output *column*.
///
/// Mathematically identical to [`dc_from_dct_32x32`] but a different
/// float operation order (~1-ulp differences that flip DC quantization
/// boundary cases upstream of `QuantizeWP`). Strict-only; verified
/// bit-exact against the instrumented v0.12 reference.
pub fn dc_from_dct_32x32_lj(coeffs: &[f32; 1024]) -> [f32; 16] {
    // libjxl `ReinterpretingIDCT`: `block = in * S(y) * S(x)` — no
    // scale compensation, because `IDCT1DImpl` is unscaled (the `*16`
    // in `dc_from_dct_32x32` cancels our `idct1d_4`'s 1/16 gain).
    let mut block = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            block[iy * 4 + ix] = coeffs[iy * 32 + ix]
                * DCT_RESAMPLE_SCALE_32_TO_4[iy]
                * DCT_RESAMPLE_SCALE_32_TO_4[ix];
        }
    }

    // Pass 1: `IDCT1D<4, 4>` down each column of `block` (SIMD lanes).
    let mut s1 = [0.0f32; 16];
    for l in 0..4 {
        let r = idct1d_4_lj_lane(block[l], block[4 + l], block[8 + l], block[12 + l]);
        for (k, v) in r.iter().enumerate() {
            s1[k * 4 + l] = *v;
        }
    }
    // Transpose is implicit: pass 2 transforms each *row* of `s1` and
    // writes the result down an output *column*.
    let mut out = [0.0f32; 16];
    for l in 0..4 {
        let r = idct1d_4_lj_lane(s1[l * 4], s1[l * 4 + 1], s1[l * 4 + 2], s1[l * 4 + 3]);
        for (k, v) in r.iter().enumerate() {
            out[k * 4 + l] = *v;
        }
    }
    out
}

/// libjxl-order `ComputeScaledDCT<32, 32>`: transpose-in → kernel →
/// transpose-out.
#[inline]
pub fn dct_32x32_lj(input: &[f32; 1024], output: &mut [f32; 1024]) {
    let mut t = [0.0f32; 1024];
    crate::vardct::common::transpose_block::<32, 32>(input, &mut t);
    let mut u = [0.0f32; 1024];
    dct_32x32(&t, &mut u);
    crate::vardct::common::transpose_block::<32, 32>(&u, output);
}

/// libjxl-order `ComputeScaledDCT<32, 16>`: vertical-32 first via
/// `dct_16x32` on the transposed input.
#[inline]
pub fn dct_32x16_lj(input: &[f32; 512], output: &mut [f32; 512]) {
    let mut t = [0.0f32; 512];
    crate::vardct::common::transpose_block::<32, 16>(input, &mut t);
    dct_16x32(&t, output);
}

/// libjxl-order `ComputeScaledDCT<16, 32>`: vertical-16 first via
/// `dct_32x16` on the transposed input.
#[inline]
pub fn dct_16x32_lj(input: &[f32; 512], output: &mut [f32; 512]) {
    let mut t = [0.0f32; 512];
    crate::vardct::common::transpose_block::<16, 32>(input, &mut t);
    dct_32x16(&t, output);
}

/// libjxl-order `ComputeScaledDCT<64, 64>`: transpose-in → kernel →
/// transpose-out.
#[inline]
pub fn dct_64x64_lj(input: &[f32], output: &mut [f32]) {
    let mut t = [0.0f32; 4096];
    crate::vardct::common::transpose_block::<64, 64>(&input[..4096], &mut t);
    let mut u = [0.0f32; 4096];
    dct_64x64(&t, &mut u);
    crate::vardct::common::transpose_block::<64, 64>(&u, &mut output[..4096]);
}

/// libjxl-order `ComputeScaledDCT<64, 32>`: vertical-64 first via
/// `dct_32x64` on the transposed input.
#[inline]
pub fn dct_64x32_lj(input: &[f32], output: &mut [f32]) {
    let mut t = [0.0f32; 2048];
    crate::vardct::common::transpose_block::<64, 32>(&input[..2048], &mut t);
    dct_32x64(&t, output);
}

/// libjxl-order `ComputeScaledDCT<32, 64>`: vertical-32 first via
/// `dct_64x32` on the transposed input.
#[inline]
pub fn dct_32x64_lj(input: &[f32], output: &mut [f32]) {
    let mut t = [0.0f32; 2048];
    crate::vardct::common::transpose_block::<32, 64>(&input[..2048], &mut t);
    dct_64x32(&t, output);
}
