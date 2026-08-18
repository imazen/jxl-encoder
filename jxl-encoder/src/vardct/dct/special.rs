// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! IDENTITY and DCT2X2 special transforms from full libjxl (enc_transforms-inl.h).
//! Uses fixed-size arrays to eliminate bounds checks.

// =============================================================================
// IDENTITY transform (libjxl enc_transforms-inl.h:464-494)
// =============================================================================

/// IDENTITY forward transform: stores pixel differences relative to reference
/// pixel (1,1) in each 4x4 sub-block, with DC averaging.
///
/// The 8x8 block is divided into four 4x4 sub-blocks. For each sub-block:
/// 1. Compute block_dc = average of 16 pixels
/// 2. Store AC: `pixel[iy][ix] - pixel[1][1]` (reference pixel)
/// 3. Merge the four sub-block DCs with 2x2 Hadamard (x0.25)
///
/// Input: `pixels` is 8x8 in stride-8 layout.
/// Output: `coefficients` in stride-8 layout.
#[inline(always)]
pub fn identity_transform(pixels: &[f32; 64], coefficients: &mut [f32; 64]) {
    jxl_simd::identity_from_pixels(pixels, coefficients);
}

// =============================================================================
// DCT2X2 transform (libjxl enc_transforms-inl.h:556-560)
// =============================================================================

/// DCT2TopBlock: hierarchical 2x2 DCT at scale S (first pass).
///
/// Reads from `block` with stride 8, writes to `out`.
/// Processes S/2 x S/2 pairs of 2x2 values, applies Hadamard transform (x0.25),
/// and stores results in four quadrants. Only the SxS region of `out` is written.
#[inline(always)]
fn dct2_top_block_first<const S: usize>(block: &[f32; 64], out: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    // `block` and `out` are distinct buffers, so butterflies write straight
    // to `out` — no intermediate temp / copy-back pass. Identical values in
    // identical order; only positions inside SxS are written (the caller
    // relies on out-of-region preservation for multi-pass composition).
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

/// DCT2TopBlock in-place: hierarchical 2x2 DCT at scale S.
///
/// Reads interleaved 2x2 values from `data`, writes quadrant layout back to `data`.
/// Only the SxS region is modified; positions outside SxS are preserved.
#[inline(always)]
fn dct2_top_block_inplace<const S: usize>(data: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    // Read-snapshot instead of temp-write + copy-back: reads and writes
    // overlap within the SxS region, so butterflies read the snapshot and
    // write `data` directly — one 256 B copy replaces the temp fill AND
    // the copy-back pass. Values and write order unchanged; positions
    // outside SxS are never written (multi-pass composition contract).
    let snap = *data;
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = snap[y * 2 * 8 + x * 2];
            let c01 = snap[y * 2 * 8 + x * 2 + 1];
            let c10 = snap[(y * 2 + 1) * 8 + x * 2];
            let c11 = snap[(y * 2 + 1) * 8 + x * 2 + 1];

            let r00 = (c00 + c01 + c10 + c11) * 0.25;
            let r01 = (c00 + c01 - c10 - c11) * 0.25;
            let r10 = (c00 - c01 + c10 - c11) * 0.25;
            let r11 = (c00 - c01 - c10 + c11) * 0.25;

            data[y * 8 + x] = r00;
            data[y * 8 + num_2x2 + x] = r01;
            data[(y + num_2x2) * 8 + x] = r10;
            data[(y + num_2x2) * 8 + num_2x2 + x] = r11;
        }
    }
}

/// DCT2X2 forward transform: hierarchical 2x2 DCT applied three times.
///
/// Input: `pixels` is 8x8 in stride-8 layout.
/// Output: `coefficients` in stride-8 layout.
#[inline(always)]
pub fn dct2x2_transform(pixels: &[f32; 64], coefficients: &mut [f32; 64]) {
    jxl_simd::dct2x2_from_pixels(pixels, coefficients);
}

// =============================================================================
// Inverse IDENTITY transform (libjxl dec_transforms-inl.h:463-498)
// =============================================================================

/// Inverse of `identity_transform`. Reconstructs 8x8 pixels from coefficients.
///
/// 1. Inverse Hadamard on DC positions `[0],[1],[8],[9]` (no x0.25 — full sum)
/// 2. For each 4x4 sub-block: compute residual_sum, derive ref_pixel = dc - residual_sum/16
/// 3. Reconstruct: pixel = coefficient + ref_pixel; corner from coefficients[(y+2)*8+x+2]
///
/// Input/Output: stride-8 layout.
#[inline(always)]
pub fn inverse_identity_transform(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    jxl_simd::identity_to_pixels(coefficients, pixels);
}

// =============================================================================
// Inverse DCT2X2 transform (libjxl dec_transforms-inl.h:569-581)
// =============================================================================

/// IDCT2TopBlock: inverse of `dct2_top_block`. Reads from quadrants, writes to
/// interleaved 2x2 positions. No x0.25 scaling (forward has it, inverse doesn't).
///
/// Operates in-place on stride-8 layout within the SxS region.
/// Positions outside SxS are preserved (critical for multi-pass composition).
#[inline(always)]
fn idct2_top_block_inplace<const S: usize>(data: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    // Read-snapshot + direct writes (see `dct2_top_block_inplace`).
    let snap = *data;
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            // Read from quadrant positions
            let c00 = snap[y * 8 + x];
            let c01 = snap[y * 8 + num_2x2 + x];
            let c10 = snap[(y + num_2x2) * 8 + x];
            let c11 = snap[(y + num_2x2) * 8 + num_2x2 + x];

            // Inverse Hadamard (no x0.25)
            let r00 = c00 + c01 + c10 + c11;
            let r01 = c00 + c01 - c10 - c11;
            let r10 = c00 - c01 + c10 - c11;
            let r11 = c00 - c01 - c10 + c11;

            // Write to interleaved 2x2 positions
            data[y * 2 * 8 + x * 2] = r00;
            data[y * 2 * 8 + x * 2 + 1] = r01;
            data[(y * 2 + 1) * 8 + x * 2] = r10;
            data[(y * 2 + 1) * 8 + x * 2 + 1] = r11;
        }
    }
}

/// Inverse of `dct2x2_transform`. Three passes of inverse hierarchical 2x2 DCT.
///
/// Input/Output: stride-8 layout.
#[inline(always)]
pub fn inverse_dct2x2_transform(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    jxl_simd::dct2x2_to_pixels(coefficients, pixels);
}
