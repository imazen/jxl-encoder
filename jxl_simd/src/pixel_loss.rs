// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD-accelerated pixel-domain loss computation for AC strategy estimation.
//!
//! Computes the 8th-power-norm of masked pixel errors:
//!   channel_loss = sum_over_pixels( ((mask[px] + offset) * error[px])^8 )
//!
//! The squaring is done in f64 for precision (matching the scalar code):
//!   m2 = (masked * masked) as f64
//!   m4 = m2 * m2
//!   m8 = m4 * m4

/// Compute pixel-domain loss for one channel of a block.
///
/// For each pixel: channel_loss += ((mask_val + mask_offset) * error_val)^8
///
/// The inner multiply is in f32, then squared three times in f64 for precision.
///
/// `pixel_error`: error values, row-major, `block_width * block_height` elements
/// `mask`: full mask1x1 buffer (stride = `mask_stride`)
/// `mask_row_base`: `pixel_y * mask_stride + pixel_x` (start of this block in mask)
/// `mask_offset`: channel-specific offset added to mask values
/// `block_width`: pixels per row (always multiple of 8)
/// `block_height`: number of rows
///
/// Returns the channel loss as f64.
pub fn pixel_domain_loss(
    pixel_error: &[f32],
    mask: &[f32],
    mask_row_base: usize,
    mask_stride: usize,
    mask_offset: f32,
    block_width: usize,
    block_height: usize,
) -> f64 {
    debug_assert!(
        block_width.is_multiple_of(8),
        "block_width must be multiple of 8"
    );
    debug_assert!(pixel_error.len() >= block_width * block_height);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return pixel_domain_loss_avx2(
                token,
                pixel_error,
                mask,
                mask_row_base,
                mask_stride,
                mask_offset,
                block_width,
                block_height,
            );
        }
    }

    // Scalar fallback
    pixel_domain_loss_scalar(
        pixel_error,
        mask,
        mask_row_base,
        mask_stride,
        mask_offset,
        block_width,
        block_height,
    )
}

fn pixel_domain_loss_scalar(
    pixel_error: &[f32],
    mask: &[f32],
    mask_row_base: usize,
    mask_stride: usize,
    mask_offset: f32,
    block_width: usize,
    block_height: usize,
) -> f64 {
    let mut channel_loss = 0.0f64;
    for py in 0..block_height {
        let mask_row_start = mask_row_base + py * mask_stride;
        let error_row_start = py * block_width;
        for px in 0..block_width {
            let mask_val = mask[mask_row_start + px];
            let error_val = pixel_error[error_row_start + px];
            let masked = (mask_val + mask_offset) * error_val;
            let m2 = (masked * masked) as f64;
            let m4 = m2 * m2;
            let m8 = m4 * m4;
            channel_loss += m8;
        }
    }
    channel_loss
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn pixel_domain_loss_avx2(
    token: archmage::X64V3Token,
    pixel_error: &[f32],
    mask: &[f32],
    mask_row_base: usize,
    mask_stride: usize,
    mask_offset: f32,
    block_width: usize,
    block_height: usize,
) -> f64 {
    use core::arch::x86_64::*;
    use magetypes::simd::{f32x8, f64x4};

    let offset_v = f32x8::splat(token, mask_offset);
    let mut acc_lo = f64x4::zero(token);
    let mut acc_hi = f64x4::zero(token);

    for py in 0..block_height {
        let mask_row_start = mask_row_base + py * mask_stride;
        let error_row_start = py * block_width;
        // Pre-slice rows so compiler can prove SIMD loads are in-bounds
        let mask_row = &mask[mask_row_start..mask_row_start + block_width];
        let error_row = &pixel_error[error_row_start..error_row_start + block_width];

        let mut px = 0;
        while px < block_width {
            // Load 8 mask values and 8 error values
            let mask_v = f32x8::from_slice(token, &mask_row[px..]);
            let error_v = f32x8::from_slice(token, &error_row[px..]);

            // masked = (mask + offset) * error (in f32)
            let masked_v = (mask_v + offset_v) * error_v;

            // m2 = masked * masked (in f32)
            let m2_v = masked_v * masked_v;

            // Convert f32x8 m2 to two f64x4 vectors via raw intrinsics
            // SAFETY: from_raw is safe inside #[arcane] because token proves AVX2 support
            let (m2_lo, m2_hi) = unsafe {
                // Lower 4 floats → f64x4
                let m2_lo_128 = _mm256_castps256_ps128(m2_v.raw());
                let m2_lo = f64x4::from_raw(_mm256_cvtps_pd(m2_lo_128));

                // Upper 4 floats → f64x4
                let m2_hi_128 = _mm256_extractf128_ps::<1>(m2_v.raw());
                let m2_hi = f64x4::from_raw(_mm256_cvtps_pd(m2_hi_128));
                (m2_lo, m2_hi)
            };

            // m4 = m2 * m2 (in f64)
            let m4_lo = m2_lo * m2_lo;
            let m4_hi = m2_hi * m2_hi;

            // m8 = m4 * m4 (in f64)
            let m8_lo = m4_lo * m4_lo;
            let m8_hi = m4_hi * m4_hi;

            // Accumulate
            acc_lo += m8_lo;
            acc_hi += m8_hi;

            px += 8;
        }
    }

    // Horizontal sum
    acc_lo.reduce_add() + acc_hi.reduce_add()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_pixel_domain_loss_uniform() {
        let block_width = 8;
        let block_height = 8;
        let mask_stride = 16;

        let pixel_error = vec![1.0f32; block_width * block_height];
        let mask = vec![0.5f32; mask_stride * 16];
        let mask_offset = 0.5f32;

        let result = pixel_domain_loss(
            &pixel_error,
            &mask,
            0,
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );

        // masked = (0.5 + 0.5) * 1.0 = 1.0
        // m2 = 1.0, m4 = 1.0, m8 = 1.0
        // 64 pixels × 1.0 = 64.0
        assert!(
            (result - 64.0).abs() < 1e-6,
            "Expected 64.0, got {}",
            result
        );
    }

    #[test]
    fn test_pixel_domain_loss_matches_scalar() {
        let block_width = 16;
        let block_height = 8;
        let mask_stride = 32;

        // Use varied data
        let mut pixel_error = vec![0.0f32; block_width * block_height];
        let mut mask = vec![0.0f32; mask_stride * 16];
        for i in 0..pixel_error.len() {
            pixel_error[i] = (i as f32 * 0.1 + 0.5) * if i % 3 == 0 { -1.0 } else { 1.0 };
        }
        for i in 0..mask.len() {
            mask[i] = (i as f32 * 0.01 + 0.3).sin().abs();
        }
        let mask_offset = 0.7f32;

        let simd_result = pixel_domain_loss(
            &pixel_error,
            &mask,
            0,
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );

        let scalar_result = pixel_domain_loss_scalar(
            &pixel_error,
            &mask,
            0,
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );

        let rel_err = ((simd_result - scalar_result) / scalar_result.max(1e-20)).abs();
        assert!(
            rel_err < 1e-6,
            "SIMD ({}) vs scalar ({}) relative error {} too large",
            simd_result,
            scalar_result,
            rel_err
        );
    }

    #[test]
    fn test_pixel_domain_loss_16x16() {
        let block_width = 16;
        let block_height = 16;
        let mask_stride = 32;

        let pixel_error = vec![0.5f32; block_width * block_height];
        let mask = vec![1.0f32; mask_stride * 32];
        let mask_offset = 0.0f32;

        let result = pixel_domain_loss(
            &pixel_error,
            &mask,
            0,
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );

        // masked = 1.0 * 0.5 = 0.5
        // m8 = 0.5^8 = 1/256 = 0.00390625
        // 256 pixels × 0.00390625 = 1.0
        assert!((result - 1.0).abs() < 1e-6, "Expected 1.0, got {}", result);
    }
}
