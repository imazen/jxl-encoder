// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated per-block masked weighted L2 error computation.
//!
//! For each 8x8 block, computes:
//!   error = sum over pixels of: mask[px]^2 * sum_c(weight[c] * (orig[c][px] - recon[c][px])^2)
//!
//! Used by EPF sharpness selection to compare reconstruction quality.

use alloc::vec;
use alloc::vec::Vec;

/// Channel weights for L2 error: X=12.34, Y=1.0, B=0.2
const CHANNEL_WEIGHTS: [f32; 3] = [12.339_445, 1.0, 0.2];

/// Compute per-block masked weighted L2 error between original and reconstructed XYB planes.
///
/// Each block's error = sum over 8x8 pixels of:
///   `mask[px]^2 * (w_x * dx^2 + w_y * dy^2 + w_b * db^2)`
///
/// All planes and mask have stride = `xsize_blocks * 8`.
#[inline]
pub fn compute_block_l2_errors(
    original: [&[f32]; 3],
    reconstructed: [&[f32]; 3],
    mask1x1: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<f32> {
    let padded_width = xsize_blocks * 8;
    let nblocks = xsize_blocks * ysize_blocks;

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return compute_block_l2_errors_avx2(
                token,
                original,
                reconstructed,
                mask1x1,
                xsize_blocks,
                ysize_blocks,
                padded_width,
                nblocks,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return compute_block_l2_errors_neon(
                token,
                original,
                reconstructed,
                mask1x1,
                xsize_blocks,
                ysize_blocks,
                padded_width,
                nblocks,
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            return compute_block_l2_errors_wasm128(
                token,
                original,
                reconstructed,
                mask1x1,
                xsize_blocks,
                ysize_blocks,
                padded_width,
                nblocks,
            );
        }
    }

    compute_block_l2_errors_scalar(
        original,
        reconstructed,
        mask1x1,
        xsize_blocks,
        ysize_blocks,
        padded_width,
        nblocks,
    )
}

#[inline]
pub fn compute_block_l2_errors_scalar(
    original: [&[f32]; 3],
    reconstructed: [&[f32]; 3],
    mask1x1: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
    padded_width: usize,
    nblocks: usize,
) -> Vec<f32> {
    let mut errors = vec![0.0f32; nblocks];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;
            let mut total_err = 0.0f32;

            for py in 0..8 {
                for px in 0..8 {
                    let y = by * 8 + py;
                    let x = bx * 8 + px;
                    let pixel_idx = y * padded_width + x;
                    let mask = mask1x1[pixel_idx];
                    let mask_sq = mask * mask;

                    for c in 0..3 {
                        let diff = original[c][pixel_idx] - reconstructed[c][pixel_idx];
                        total_err += CHANNEL_WEIGHTS[c] * mask_sq * diff * diff;
                    }
                }
            }

            errors[block_idx] = total_err;
        }
    }

    errors
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn compute_block_l2_errors_avx2(
    token: archmage::X64V3Token,
    original: [&[f32]; 3],
    reconstructed: [&[f32]; 3],
    mask1x1: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
    padded_width: usize,
    nblocks: usize,
) -> Vec<f32> {
    use magetypes::simd::f32x8;

    let w_x = f32x8::splat(token, CHANNEL_WEIGHTS[0]);
    // w_y = 1.0, multiplication skipped in inner loop
    let w_b = f32x8::splat(token, CHANNEL_WEIGHTS[2]);

    let mut errors = vec![0.0f32; nblocks];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;
            let mut acc = f32x8::zero(token);

            for py in 0..8 {
                let row_start = (by * 8 + py) * padded_width + bx * 8;

                // Load 8 mask values and square them
                let mask_v = crate::load_f32x8(token, mask1x1, row_start);
                let mask_sq = mask_v * mask_v;

                // X channel: w_x * mask_sq * (orig_x - recon_x)^2
                let orig_x = crate::load_f32x8(token, original[0], row_start);
                let recon_x = crate::load_f32x8(token, reconstructed[0], row_start);
                let diff_x = orig_x - recon_x;
                acc += w_x * mask_sq * diff_x * diff_x;

                // Y channel: w_y * mask_sq * (orig_y - recon_y)^2
                let orig_y = crate::load_f32x8(token, original[1], row_start);
                let recon_y = crate::load_f32x8(token, reconstructed[1], row_start);
                let diff_y = orig_y - recon_y;
                // w_y = 1.0, so skip the multiply
                acc += mask_sq * diff_y * diff_y;

                // B channel: w_b * mask_sq * (orig_b - recon_b)^2
                let orig_b = crate::load_f32x8(token, original[2], row_start);
                let recon_b = crate::load_f32x8(token, reconstructed[2], row_start);
                let diff_b = orig_b - recon_b;
                acc += w_b * mask_sq * diff_b * diff_b;
            }

            // Horizontal sum of the 8-lane accumulator
            errors[block_idx] = acc.reduce_add();
        }
    }

    errors
}

// ============================================================================
// aarch64 NEON implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn compute_block_l2_errors_neon(
    token: archmage::NeonToken,
    original: [&[f32]; 3],
    reconstructed: [&[f32]; 3],
    mask1x1: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
    padded_width: usize,
    nblocks: usize,
) -> Vec<f32> {
    use magetypes::simd::f32x4;

    let w_x = f32x4::splat(token, CHANNEL_WEIGHTS[0]);
    let w_b = f32x4::splat(token, CHANNEL_WEIGHTS[2]);

    let mut errors = vec![0.0f32; nblocks];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;
            let mut acc = f32x4::zero(token);

            for py in 0..8 {
                let row_start = (by * 8 + py) * padded_width + bx * 8;

                // Process 8 pixels as two f32x4 chunks
                for half in 0..2usize {
                    let off = row_start + half * 4;

                    let mask_v = f32x4::from_slice(token, &mask1x1[off..]);
                    let mask_sq = mask_v * mask_v;

                    let orig_x = f32x4::from_slice(token, &original[0][off..]);
                    let recon_x = f32x4::from_slice(token, &reconstructed[0][off..]);
                    let diff_x = orig_x - recon_x;
                    acc += w_x * mask_sq * diff_x * diff_x;

                    let orig_y = f32x4::from_slice(token, &original[1][off..]);
                    let recon_y = f32x4::from_slice(token, &reconstructed[1][off..]);
                    let diff_y = orig_y - recon_y;
                    acc += mask_sq * diff_y * diff_y;

                    let orig_b = f32x4::from_slice(token, &original[2][off..]);
                    let recon_b = f32x4::from_slice(token, &reconstructed[2][off..]);
                    let diff_b = orig_b - recon_b;
                    acc += w_b * mask_sq * diff_b * diff_b;
                }
            }

            errors[block_idx] = acc.reduce_add();
        }
    }

    errors
}

// ============================================================================
// wasm32 SIMD128 implementation
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn compute_block_l2_errors_wasm128(
    token: archmage::Wasm128Token,
    original: [&[f32]; 3],
    reconstructed: [&[f32]; 3],
    mask1x1: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
    padded_width: usize,
    nblocks: usize,
) -> Vec<f32> {
    use magetypes::simd::f32x4;

    let w_x = f32x4::splat(token, CHANNEL_WEIGHTS[0]);
    let w_b = f32x4::splat(token, CHANNEL_WEIGHTS[2]);

    let mut errors = vec![0.0f32; nblocks];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;
            let mut acc = f32x4::zero(token);

            for py in 0..8 {
                let row_start = (by * 8 + py) * padded_width + bx * 8;

                // Process 8 pixels as two f32x4 chunks
                for half in 0..2usize {
                    let off = row_start + half * 4;

                    let mask_v = f32x4::from_slice(token, &mask1x1[off..]);
                    let mask_sq = mask_v * mask_v;

                    let orig_x = f32x4::from_slice(token, &original[0][off..]);
                    let recon_x = f32x4::from_slice(token, &reconstructed[0][off..]);
                    let diff_x = orig_x - recon_x;
                    acc += w_x * mask_sq * diff_x * diff_x;

                    let orig_y = f32x4::from_slice(token, &original[1][off..]);
                    let recon_y = f32x4::from_slice(token, &reconstructed[1][off..]);
                    let diff_y = orig_y - recon_y;
                    acc += mask_sq * diff_y * diff_y;

                    let orig_b = f32x4::from_slice(token, &original[2][off..]);
                    let recon_b = f32x4::from_slice(token, &reconstructed[2][off..]);
                    let diff_b = orig_b - recon_b;
                    acc += w_b * mask_sq * diff_b * diff_b;
                }
            }

            errors[block_idx] = acc.reduce_add();
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use alloc::vec;

    #[test]
    fn test_block_l2_errors_uniform() {
        let xsize_blocks = 2;
        let ysize_blocks = 2;
        let padded_width = xsize_blocks * 8;
        let n = padded_width * ysize_blocks * 8;

        // Uniform original, zero reconstructed → diff = original
        let original = [vec![1.0f32; n], vec![1.0f32; n], vec![1.0f32; n]];
        let reconstructed = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
        let mask = vec![1.0f32; n];

        let errors = compute_block_l2_errors(
            [&original[0], &original[1], &original[2]],
            [&reconstructed[0], &reconstructed[1], &reconstructed[2]],
            &mask,
            xsize_blocks,
            ysize_blocks,
        );

        // Each pixel: mask^2 * (w_x * 1^2 + w_y * 1^2 + w_b * 1^2)
        //           = 1.0 * (12.339445 + 1.0 + 0.2) = 13.539445
        // 64 pixels per block: 64 * 13.539445 = 866.52448
        let expected = 64.0 * (CHANNEL_WEIGHTS[0] + CHANNEL_WEIGHTS[1] + CHANNEL_WEIGHTS[2]);
        for (i, &err) in errors.iter().enumerate() {
            assert!(
                (err - expected).abs() < 0.1,
                "Block {} error {} != expected {}",
                i,
                err,
                expected
            );
        }
    }

    #[test]
    fn test_block_l2_errors_matches_scalar() {
        let xsize_blocks = 4;
        let ysize_blocks = 4;
        let padded_width = xsize_blocks * 8;
        let n = padded_width * ysize_blocks * 8;

        let mut orig0 = vec![0.0f32; n];
        let mut orig1 = vec![0.0f32; n];
        let mut orig2 = vec![0.0f32; n];
        let mut recon0 = vec![0.0f32; n];
        let mut recon1 = vec![0.0f32; n];
        let mut recon2 = vec![0.0f32; n];
        let mut mask = vec![0.0f32; n];

        for i in 0..n {
            let f = i as f32;
            orig0[i] = (f * 0.013).sin() * 0.5;
            orig1[i] = (f * 0.017).cos() * 0.8;
            orig2[i] = (f * 0.023).sin() * 0.3;
            recon0[i] = orig0[i] + (f * 0.031).sin() * 0.1;
            recon1[i] = orig1[i] + (f * 0.037).cos() * 0.05;
            recon2[i] = orig2[i] + (f * 0.041).sin() * 0.02;
            mask[i] = 0.5 + (f * 0.007).sin().abs() * 0.5;
        }

        let scalar_result = compute_block_l2_errors_scalar(
            [&orig0, &orig1, &orig2],
            [&recon0, &recon1, &recon2],
            &mask,
            xsize_blocks,
            ysize_blocks,
            padded_width,
            xsize_blocks * ysize_blocks,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let simd_result = compute_block_l2_errors(
                    [&orig0, &orig1, &orig2],
                    [&recon0, &recon1, &recon2],
                    &mask,
                    xsize_blocks,
                    ysize_blocks,
                );
                assert_eq!(simd_result.len(), scalar_result.len(), "[{perm}]");
                for (i, (&s, &sc)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
                    let rel_err = if sc.abs() > 1e-10 {
                        ((s - sc) / sc).abs()
                    } else {
                        (s - sc).abs()
                    };
                    assert!(
                        rel_err < 1e-5,
                        "Block {i} SIMD {s} vs scalar {sc} rel_err {rel_err} [{perm}]",
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }
}
