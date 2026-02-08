// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! IDENTITY and DCT2X2 special transforms from full libjxl (enc_transforms-inl.h).

// =============================================================================
// IDENTITY transform (libjxl enc_transforms-inl.h:464-494)
// =============================================================================

/// IDENTITY forward transform: stores pixel differences relative to reference
/// pixel (1,1) in each 4x4 sub-block, with DC averaging.
///
/// The 8x8 block is divided into four 4x4 sub-blocks. For each sub-block:
/// 1. Compute block_dc = average of 16 pixels
/// 2. Store AC: pixel[iy][ix] - pixel[1][1] (reference pixel)
/// 3. Merge the four sub-block DCs with 2x2 Hadamard (×0.25)
///
/// Input: `pixels` is 8x8 with given `stride`.
/// Output: `coefficients[0..64]` in stride-8 layout.
pub fn identity_transform(pixels: &[f32], stride: usize, coefficients: &mut [f32]) {
    debug_assert!(coefficients.len() >= 64);

    // Process 2x2 grid of 4x4 sub-blocks
    for y in 0..2usize {
        for x in 0..2usize {
            // Compute block DC (average of 16 pixels)
            let mut block_dc = 0.0f32;
            for iy in 0..4 {
                for ix in 0..4 {
                    block_dc += pixels[(y * 4 + iy) * stride + x * 4 + ix];
                }
            }
            block_dc *= 1.0 / 16.0;

            // Reference pixel: (1,1) in each 4x4 sub-block
            let ref_pixel = pixels[(y * 4 + 1) * stride + x * 4 + 1];

            // Store AC coefficients: pixel - reference pixel
            // Coefficient layout: interleaved at (y + iy*2, x + ix*2) positions
            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 1 && iy == 1 {
                        continue; // Skip ref pixel position
                    }
                    coefficients[(y + iy * 2) * 8 + x + ix * 2] =
                        pixels[(y * 4 + iy) * stride + x * 4 + ix] - ref_pixel;
                }
            }

            // Copy corner coefficient, then store DC at (y, x)
            coefficients[(y + 2) * 8 + x + 2] = coefficients[y * 8 + x];
            coefficients[y * 8 + x] = block_dc;
        }
    }

    // Merge 2x2 block DCs with Hadamard transform (×0.25)
    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];
    let block11 = coefficients[9];
    coefficients[0] = (block00 + block01 + block10 + block11) * 0.25;
    coefficients[1] = (block00 + block01 - block10 - block11) * 0.25;
    coefficients[8] = (block00 - block01 + block10 - block11) * 0.25;
    coefficients[9] = (block00 - block01 - block10 + block11) * 0.25;
}

// =============================================================================
// DCT2X2 transform (libjxl enc_transforms-inl.h:556-560)
// =============================================================================

/// DCT2TopBlock: hierarchical 2x2 DCT at scale S (first pass, separate source).
///
/// Reads from `block` with stride `src_stride`, writes to `out` with stride 8.
/// Processes S/2 × S/2 pairs of 2x2 values, applies Hadamard transform (×0.25),
/// and stores results in four quadrants. Only the S×S region of `out` is written.
fn dct2_top_block_first<const S: usize>(block: &[f32], src_stride: usize, out: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    let mut temp = [0.0f32; 64];

    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = block[y * 2 * src_stride + x * 2];
            let c01 = block[y * 2 * src_stride + x * 2 + 1];
            let c10 = block[(y * 2 + 1) * src_stride + x * 2];
            let c11 = block[(y * 2 + 1) * src_stride + x * 2 + 1];

            let r00 = (c00 + c01 + c10 + c11) * 0.25;
            let r01 = (c00 + c01 - c10 - c11) * 0.25;
            let r10 = (c00 - c01 + c10 - c11) * 0.25;
            let r11 = (c00 - c01 - c10 + c11) * 0.25;

            temp[y * 8 + x] = r00;
            temp[y * 8 + num_2x2 + x] = r01;
            temp[(y + num_2x2) * 8 + x] = r10;
            temp[(y + num_2x2) * 8 + num_2x2 + x] = r11;
        }
    }

    // Copy S×S region from temp to output
    for y in 0..S {
        for x in 0..S {
            out[y * 8 + x] = temp[y * 8 + x];
        }
    }
}

/// DCT2TopBlock in-place: hierarchical 2x2 DCT at scale S.
///
/// Reads interleaved 2x2 values from `data`, writes quadrant layout back to `data`.
/// Only the S×S region is modified; positions outside S×S are preserved.
/// Matches libjxl's `DCT2TopBlock<S>(coefficients, kBlockDim, coefficients)` pattern.
fn dct2_top_block_inplace<const S: usize>(data: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    let mut temp = [0.0f32; 64];

    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = data[y * 2 * 8 + x * 2];
            let c01 = data[y * 2 * 8 + x * 2 + 1];
            let c10 = data[(y * 2 + 1) * 8 + x * 2];
            let c11 = data[(y * 2 + 1) * 8 + x * 2 + 1];

            let r00 = (c00 + c01 + c10 + c11) * 0.25;
            let r01 = (c00 + c01 - c10 - c11) * 0.25;
            let r10 = (c00 - c01 + c10 - c11) * 0.25;
            let r11 = (c00 - c01 - c10 + c11) * 0.25;

            temp[y * 8 + x] = r00;
            temp[y * 8 + num_2x2 + x] = r01;
            temp[(y + num_2x2) * 8 + x] = r10;
            temp[(y + num_2x2) * 8 + num_2x2 + x] = r11;
        }
    }

    // Copy only S×S region back (preserving positions outside S×S)
    for y in 0..S {
        for x in 0..S {
            data[y * 8 + x] = temp[y * 8 + x];
        }
    }
}

/// DCT2X2 forward transform: hierarchical 2x2 DCT applied three times.
///
/// Matches libjxl enc_transforms-inl.h:556-559:
///   DCT2TopBlock<8>(pixels, pixels_stride, coefficients);
///   DCT2TopBlock<4>(coefficients, kBlockDim, coefficients);
///   DCT2TopBlock<2>(coefficients, kBlockDim, coefficients);
///
/// First pass reads from `pixels` (with `stride`), writes to coefficients.
/// Passes 2 and 3 operate in-place on coefficients, preserving data outside
/// their S×S region (critical for correct coefficient layout).
///
/// Input: `pixels` is 8x8 with given `stride`.
/// Output: `coefficients[0..64]` in stride-8 layout.
pub fn dct2x2_transform(pixels: &[f32], stride: usize, coefficients: &mut [f32]) {
    debug_assert!(coefficients.len() >= 64);

    // Pass 1: read from pixels with source stride, write to coefficients
    let mut buf = [0.0f32; 64];
    dct2_top_block_first::<8>(pixels, stride, &mut buf);

    // Passes 2 and 3: in-place on buf (matching C++ pattern)
    dct2_top_block_inplace::<4>(&mut buf);
    dct2_top_block_inplace::<2>(&mut buf);

    coefficients[..64].copy_from_slice(&buf);
}

// =============================================================================
// Inverse IDENTITY transform (libjxl dec_transforms-inl.h:463-498)
// =============================================================================

/// Inverse of `identity_transform`. Reconstructs 8x8 pixels from coefficients.
///
/// 1. Inverse Hadamard on DC positions [0],[1],[8],[9] (no ×0.25 — full sum)
/// 2. For each 4x4 sub-block: compute residual_sum, derive ref_pixel = dc - residual_sum/16
/// 3. Reconstruct: pixel = coefficient + ref_pixel; corner from coefficients[(y+2)*8+x+2]
///
/// Input: `coefficients[0..64]` in stride-8 layout.
/// Output: `pixels[0..64]` in stride-8 layout.
pub fn inverse_identity_transform(coefficients: &[f32], pixels: &mut [f32]) {
    debug_assert!(coefficients.len() >= 64);
    debug_assert!(pixels.len() >= 64);

    // Inverse Hadamard on DC positions (no ×0.25 scaling — this is the inverse)
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

            // Sum all residual coefficients (skip [0][0] which is DC)
            let mut residual_sum = 0.0f32;
            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 0 && iy == 0 {
                        continue;
                    }
                    residual_sum += coefficients[(y + iy * 2) * 8 + x + ix * 2];
                }
            }

            // Derive reference pixel: dc - residual_sum/16
            let ref_pixel = block_dc - residual_sum * (1.0 / 16.0);
            pixels[(4 * y + 1) * 8 + 4 * x + 1] = ref_pixel;

            // Reconstruct all other pixels: coefficient + ref_pixel
            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 1 && iy == 1 {
                        continue;
                    }
                    pixels[(y * 4 + iy) * 8 + x * 4 + ix] =
                        coefficients[(y + iy * 2) * 8 + x + ix * 2] + ref_pixel;
                }
            }

            // Corner pixel comes from the saved position
            pixels[y * 4 * 8 + x * 4] = coefficients[(y + 2) * 8 + x + 2] + ref_pixel;
        }
    }
}

// =============================================================================
// Inverse DCT2X2 transform (libjxl dec_transforms-inl.h:569-581)
// =============================================================================

/// IDCT2TopBlock: inverse of `dct2_top_block`. Reads from quadrants, writes to
/// interleaved 2x2 positions. No ×0.25 scaling (forward has it, inverse doesn't).
///
/// Operates in-place on stride-8 layout within the S×S region, matching libjxl's
/// `IDCT2TopBlock(coeffs, kBlockDim, coeffs)` pattern. Positions outside S×S
/// are preserved (critical for multi-pass composition).
fn idct2_top_block_inplace<const S: usize>(data: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    let mut temp = [0.0f32; 64];

    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            // Read from quadrant positions
            let c00 = data[y * 8 + x];
            let c01 = data[y * 8 + num_2x2 + x];
            let c10 = data[(y + num_2x2) * 8 + x];
            let c11 = data[(y + num_2x2) * 8 + num_2x2 + x];

            // Inverse Hadamard (no ×0.25)
            let r00 = c00 + c01 + c10 + c11;
            let r01 = c00 + c01 - c10 - c11;
            let r10 = c00 - c01 + c10 - c11;
            let r11 = c00 - c01 - c10 + c11;

            // Write to interleaved 2x2 positions in temp
            temp[y * 2 * 8 + x * 2] = r00;
            temp[y * 2 * 8 + x * 2 + 1] = r01;
            temp[(y * 2 + 1) * 8 + x * 2] = r10;
            temp[(y * 2 + 1) * 8 + x * 2 + 1] = r11;
        }
    }

    // Copy only S×S region back to data (preserving positions outside S×S)
    for y in 0..S {
        for x in 0..S {
            data[y * 8 + x] = temp[y * 8 + x];
        }
    }
}

/// Inverse of `dct2x2_transform`. Three passes of inverse hierarchical 2x2 DCT.
///
/// Input: `coefficients[0..64]` in stride-8 layout.
/// Output: `pixels[0..64]` in stride-8 layout.
pub fn inverse_dct2x2_transform(coefficients: &[f32], pixels: &mut [f32]) {
    debug_assert!(coefficients.len() >= 64);
    debug_assert!(pixels.len() >= 64);

    // Work in-place on a copy, matching libjxl's pattern:
    //   IDCT2TopBlock<2>(coeffs, kBlockDim, coeffs);
    //   IDCT2TopBlock<4>(coeffs, kBlockDim, coeffs);
    //   IDCT2TopBlock<8>(coeffs, kBlockDim, coeffs);
    let mut coeffs = [0.0f32; 64];
    coeffs.copy_from_slice(&coefficients[..64]);

    idct2_top_block_inplace::<2>(&mut coeffs);
    idct2_top_block_inplace::<4>(&mut coeffs);
    idct2_top_block_inplace::<8>(&mut coeffs);

    pixels[..64].copy_from_slice(&coeffs);
}
