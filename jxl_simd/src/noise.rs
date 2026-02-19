// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated Wiener denoising channel filter.
//!
//! `denoise_channel`: per-pixel Wiener filter with 5x5 local statistics.
//! For each pixel, estimates local signal variance, subtracts noise variance
//! (from LUT interpolation on Y channel), and applies adaptive filtering.
//!
//! The 5x5 window accumulation is the hot inner loop: 25 loads + FMAs per pixel.
//! AVX2 processes 8 pixels and NEON 4 pixels in parallel.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assign_op_pattern)]

const NUM_NOISE_POINTS: usize = 8;
const RADIUS: usize = 2;
const EPS: f32 = 1e-10;

/// Compute index and fractional part for noise LUT interpolation.
#[inline]
fn index_and_frac(x: f32) -> (usize, f32) {
    let k_scale_numerator = (NUM_NOISE_POINTS - 2) as f32;
    let scaled_x = (x * k_scale_numerator).max(0.0);
    let floor_x = scaled_x.floor();
    let frac_x = scaled_x - floor_x;
    if scaled_x >= k_scale_numerator + 1.0 {
        (k_scale_numerator as usize, 1.0)
    } else {
        (floor_x as usize, frac_x)
    }
}

/// Interpolate the noise LUT at a given intensity value.
#[inline]
fn interpolate_noise_lut(noise_lut: &[f32; NUM_NOISE_POINTS], intensity: f32) -> f32 {
    let (idx, frac) = index_and_frac(intensity);
    if idx >= NUM_NOISE_POINTS - 1 {
        return noise_lut[NUM_NOISE_POINTS - 1];
    }
    noise_lut[idx] * (1.0 - frac) + noise_lut[idx + 1] * frac
}

/// Apply Wiener filter to a single pixel (scalar helper for border pixels).
#[inline(always)]
fn denoise_pixel(
    dest: &mut [f32],
    orig: &[f32],
    y_channel: &[f32],
    width: usize,
    height: usize,
    noise_lut: &[f32; NUM_NOISE_POINTS],
    denoise_scale: f32,
    px: usize,
    py: usize,
) {
    let idx = py * width + px;
    let y_val = y_channel[idx];
    let sigma = interpolate_noise_lut(noise_lut, y_val.abs()) * denoise_scale;
    let noise_var = sigma * sigma;
    if noise_var < EPS {
        return;
    }

    let y_start = py.saturating_sub(RADIUS);
    let y_end = (py + RADIUS + 1).min(height);
    let x_start = px.saturating_sub(RADIUS);
    let x_end = (px + RADIUS + 1).min(width);

    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0.0f32;
    for ny in y_start..y_end {
        for nx in x_start..x_end {
            let v = orig[ny * width + nx];
            sum += v;
            sum_sq += v * v;
            count += 1.0;
        }
    }

    let mean = sum / count;
    let variance = ((sum_sq / count) - mean * mean).max(0.0);
    let signal_var = (variance - noise_var).max(0.0);
    let wiener = signal_var / (signal_var + noise_var);
    dest[idx] = mean + (orig[idx] - mean) * wiener;
}

/// Wiener filter for a single channel (runtime dispatch).
///
/// `noise_lut` is the 8-point noise LUT mapping intensity to noise level.
/// `denoise_scale` combines the denoising fraction with quality coefficient.
pub fn denoise_channel(
    dest: &mut [f32],
    orig: &[f32],
    y_channel: &[f32],
    width: usize,
    height: usize,
    noise_lut: &[f32; NUM_NOISE_POINTS],
    denoise_scale: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return denoise_channel_avx2(
                token,
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return denoise_channel_neon(
                token,
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            return denoise_channel_wasm128(
                token,
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
            );
        }
    }

    denoise_channel_scalar(
        dest,
        orig,
        y_channel,
        width,
        height,
        noise_lut,
        denoise_scale,
    );
}

/// Wiener filter for a single channel (scalar).
pub fn denoise_channel_scalar(
    dest: &mut [f32],
    orig: &[f32],
    y_channel: &[f32],
    width: usize,
    height: usize,
    noise_lut: &[f32; NUM_NOISE_POINTS],
    denoise_scale: f32,
) {
    for py in 0..height {
        for px in 0..width {
            denoise_pixel(
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
                px,
                py,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn denoise_channel_avx2(
    token: archmage::X64V3Token,
    dest: &mut [f32],
    orig: &[f32],
    y_channel: &[f32],
    width: usize,
    height: usize,
    noise_lut: &[f32; NUM_NOISE_POINTS],
    denoise_scale: f32,
) {
    use magetypes::simd::f32x8;

    let zero_v = f32x8::splat(token, 0.0);
    let count_inv_v = f32x8::splat(token, 1.0 / 25.0);
    let eps_v = f32x8::splat(token, EPS);
    let half_v = f32x8::splat(token, 0.5);

    for py in 0..height {
        // Border rows: full scalar
        if py < RADIUS || py + RADIUS >= height {
            for px in 0..width {
                denoise_pixel(
                    dest,
                    orig,
                    y_channel,
                    width,
                    height,
                    noise_lut,
                    denoise_scale,
                    px,
                    py,
                );
            }
            continue;
        }

        // Left border pixels
        for px in 0..RADIUS {
            denoise_pixel(
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
                px,
                py,
            );
        }

        // SIMD interior: 8 pixels at a time.
        // For 8 pixels at px..px+7, the 5x5 window accesses [px-2..px+9].
        // Need px >= RADIUS and px + 9 < width, i.e., px + 10 <= width.
        let mut px = RADIUS;
        while px + 10 <= width {
            let idx_base = py * width + px;

            // Scalar LUT interpolation for 8 y_channel values
            let y_arr: [f32; 8] = f32x8::from_slice(token, &y_channel[idx_base..]).into();
            let mut sigma_arr = [0.0f32; 8];
            for j in 0..8 {
                sigma_arr[j] = interpolate_noise_lut(noise_lut, y_arr[j].abs()) * denoise_scale;
            }
            let noise_var = {
                let sigma = f32x8::from_array(token, sigma_arr);
                sigma * sigma
            };

            // Accumulate sum and sum-of-squares over the 5x5 window.
            // For each window row dy and column offset dx, load 8 consecutive
            // values from orig starting at (py+dy, px+dx). These correspond to
            // the window column dx for all 8 output pixels simultaneously.
            let mut sum_v = zero_v;
            let mut sum_sq_v = zero_v;
            for dy in -(RADIUS as i32)..=(RADIUS as i32) {
                let row_start = ((py as i32 + dy) as usize) * width;
                for dx in -(RADIUS as i32)..=(RADIUS as i32) {
                    let off = row_start + ((px as i32 + dx) as usize);
                    let v = f32x8::from_slice(token, &orig[off..]);
                    sum_v = sum_v + v;
                    sum_sq_v = v.mul_add(v, sum_sq_v);
                }
            }

            // Wiener filter: denoised = mean + (pixel - mean) * signal_var / (signal_var + noise_var)
            let mean = sum_v * count_inv_v;
            // max(x, 0) via (x + |x|) * 0.5 — avoids needing a comparison/blend op
            let raw_variance = sum_sq_v * count_inv_v - mean * mean;
            let variance = (raw_variance + raw_variance.abs()) * half_v;
            let raw_signal = variance - noise_var;
            let signal_var = (raw_signal + raw_signal.abs()) * half_v;
            // Add EPS to denominator to avoid NaN when both signal_var and noise_var are tiny.
            // When noise_var < EPS, wiener approaches 1.0 and result approaches orig.
            let wiener = signal_var / (signal_var + noise_var + eps_v);

            let orig_vals = f32x8::from_slice(token, &orig[idx_base..]);
            let result = mean + (orig_vals - mean) * wiener;
            result.store((&mut dest[idx_base..idx_base + 8]).try_into().unwrap());

            px += 8;
        }

        // Scalar remainder + right border
        while px < width {
            denoise_pixel(
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
                px,
                py,
            );
            px += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn denoise_channel_neon(
    token: archmage::NeonToken,
    dest: &mut [f32],
    orig: &[f32],
    y_channel: &[f32],
    width: usize,
    height: usize,
    noise_lut: &[f32; NUM_NOISE_POINTS],
    denoise_scale: f32,
) {
    use magetypes::simd::f32x4;

    let zero_v = f32x4::splat(token, 0.0);
    let count_inv_v = f32x4::splat(token, 1.0 / 25.0);
    let eps_v = f32x4::splat(token, EPS);
    let half_v = f32x4::splat(token, 0.5);

    for py in 0..height {
        if py < RADIUS || py + RADIUS >= height {
            for px in 0..width {
                denoise_pixel(
                    dest,
                    orig,
                    y_channel,
                    width,
                    height,
                    noise_lut,
                    denoise_scale,
                    px,
                    py,
                );
            }
            continue;
        }

        for px in 0..RADIUS {
            denoise_pixel(
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
                px,
                py,
            );
        }

        // SIMD interior: 4 pixels at a time.
        // For 4 pixels at px..px+3, window accesses [px-2..px+5].
        // Need px >= RADIUS and px + 5 < width, i.e., px + 6 <= width.
        let mut px = RADIUS;
        while px + 6 <= width {
            let idx_base = py * width + px;

            let y_arr: [f32; 4] = f32x4::from_slice(token, &y_channel[idx_base..]).into();
            let mut sigma_arr = [0.0f32; 4];
            for j in 0..4 {
                sigma_arr[j] = interpolate_noise_lut(noise_lut, y_arr[j].abs()) * denoise_scale;
            }
            let noise_var = {
                let sigma = f32x4::from_array(token, sigma_arr);
                sigma * sigma
            };

            let mut sum_v = zero_v;
            let mut sum_sq_v = zero_v;
            for dy in -(RADIUS as i32)..=(RADIUS as i32) {
                let row_start = ((py as i32 + dy) as usize) * width;
                for dx in -(RADIUS as i32)..=(RADIUS as i32) {
                    let off = row_start + ((px as i32 + dx) as usize);
                    let v = f32x4::from_slice(token, &orig[off..]);
                    sum_v = sum_v + v;
                    sum_sq_v = v.mul_add(v, sum_sq_v);
                }
            }

            let mean = sum_v * count_inv_v;
            let raw_variance = sum_sq_v * count_inv_v - mean * mean;
            let variance = (raw_variance + raw_variance.abs()) * half_v;
            let raw_signal = variance - noise_var;
            let signal_var = (raw_signal + raw_signal.abs()) * half_v;
            let wiener = signal_var / (signal_var + noise_var + eps_v);

            let orig_vals = f32x4::from_slice(token, &orig[idx_base..]);
            let result = mean + (orig_vals - mean) * wiener;
            result.store((&mut dest[idx_base..idx_base + 4]).try_into().unwrap());

            px += 4;
        }

        while px < width {
            denoise_pixel(
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
                px,
                py,
            );
            px += 1;
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn denoise_channel_wasm128(
    token: archmage::Wasm128Token,
    dest: &mut [f32],
    orig: &[f32],
    y_channel: &[f32],
    width: usize,
    height: usize,
    noise_lut: &[f32; NUM_NOISE_POINTS],
    denoise_scale: f32,
) {
    use magetypes::simd::f32x4;

    let zero_v = f32x4::splat(token, 0.0);
    let count_inv_v = f32x4::splat(token, 1.0 / 25.0);
    let eps_v = f32x4::splat(token, EPS);
    let half_v = f32x4::splat(token, 0.5);

    for py in 0..height {
        if py < RADIUS || py + RADIUS >= height {
            for px in 0..width {
                denoise_pixel(
                    dest,
                    orig,
                    y_channel,
                    width,
                    height,
                    noise_lut,
                    denoise_scale,
                    px,
                    py,
                );
            }
            continue;
        }

        for px in 0..RADIUS {
            denoise_pixel(
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
                px,
                py,
            );
        }

        // SIMD interior: 4 pixels at a time.
        // For 4 pixels at px..px+3, window accesses [px-2..px+5].
        // Need px >= RADIUS and px + 5 < width, i.e., px + 6 <= width.
        let mut px = RADIUS;
        while px + 6 <= width {
            let idx_base = py * width + px;

            let y_arr: [f32; 4] = f32x4::from_slice(token, &y_channel[idx_base..]).into();
            let mut sigma_arr = [0.0f32; 4];
            for j in 0..4 {
                sigma_arr[j] = interpolate_noise_lut(noise_lut, y_arr[j].abs()) * denoise_scale;
            }
            let noise_var = {
                let sigma = f32x4::from_array(token, sigma_arr);
                sigma * sigma
            };

            let mut sum_v = zero_v;
            let mut sum_sq_v = zero_v;
            for dy in -(RADIUS as i32)..=(RADIUS as i32) {
                let row_start = ((py as i32 + dy) as usize) * width;
                for dx in -(RADIUS as i32)..=(RADIUS as i32) {
                    let off = row_start + ((px as i32 + dx) as usize);
                    let v = f32x4::from_slice(token, &orig[off..]);
                    sum_v = sum_v + v;
                    sum_sq_v = v.mul_add(v, sum_sq_v);
                }
            }

            let mean = sum_v * count_inv_v;
            let raw_variance = sum_sq_v * count_inv_v - mean * mean;
            let variance = (raw_variance + raw_variance.abs()) * half_v;
            let raw_signal = variance - noise_var;
            let signal_var = (raw_signal + raw_signal.abs()) * half_v;
            let wiener = signal_var / (signal_var + noise_var + eps_v);

            let orig_vals = f32x4::from_slice(token, &orig[idx_base..]);
            let result = mean + (orig_vals - mean) * wiener;
            result.store((&mut dest[idx_base..idx_base + 4]).try_into().unwrap());

            px += 4;
        }

        while px < width {
            denoise_pixel(
                dest,
                orig,
                y_channel,
                width,
                height,
                noise_lut,
                denoise_scale,
                px,
                py,
            );
            px += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    fn make_test_data(width: usize, height: usize) -> (alloc::vec::Vec<f32>, alloc::vec::Vec<f32>) {
        let n = width * height;
        let orig: alloc::vec::Vec<f32> = (0..n)
            .map(|i| {
                let base = 0.3 + 0.4 * ((i % width) as f32 / width as f32);
                let noise = ((i * 7919 + 1234) % 1000) as f32 / 10000.0 - 0.05;
                base + noise
            })
            .collect();
        let y_channel = orig.clone();
        (orig, y_channel)
    }

    #[test]
    fn test_denoise_scalar_vs_dispatch() {
        let width = 64;
        let height = 64;
        let noise_lut = [0.05f32; NUM_NOISE_POINTS];
        let denoise_scale = 0.25;

        let (orig, y_channel) = make_test_data(width, height);

        let mut dest_scalar = orig.clone();
        denoise_channel_scalar(
            &mut dest_scalar,
            &orig,
            &y_channel,
            width,
            height,
            &noise_lut,
            denoise_scale,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut dest_dispatch = orig.clone();
                denoise_channel(
                    &mut dest_dispatch,
                    &orig,
                    &y_channel,
                    width,
                    height,
                    &noise_lut,
                    denoise_scale,
                );
                for i in 0..orig.len() {
                    let diff = (dest_scalar[i] - dest_dispatch[i]).abs();
                    assert!(
                        diff < 1e-4,
                        "Mismatch at pixel {i}: scalar={} dispatch={} diff={diff} [{perm}]",
                        dest_scalar[i],
                        dest_dispatch[i],
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_denoise_zero_noise() {
        let noise_lut = [0.0f32; NUM_NOISE_POINTS];
        let mut dest = alloc::vec![0.5; 16];
        let orig = alloc::vec![0.5; 16];
        let y = alloc::vec![0.5; 16];
        // Zero noise LUT → noise_var = 0 < EPS → no pixels modified
        denoise_channel(&mut dest, &orig, &y, 4, 4, &noise_lut, 1.0);
        for (i, (&d, &o)) in dest.iter().zip(orig.iter()).enumerate() {
            assert!((d - o).abs() < 1e-6, "Pixel {} changed: {} -> {}", i, o, d,);
        }
    }

    #[test]
    fn test_denoise_reduces_noise() {
        let width = 32;
        let height = 32;
        let n = width * height;
        let clean_val = 0.5f32;
        let noise_lut = [0.1f32; NUM_NOISE_POINTS];
        let orig: alloc::vec::Vec<f32> = (0..n)
            .map(|i| {
                let noise = ((i * 7919 + 1234) % 1000) as f32 / 1000.0 - 0.5;
                clean_val + noise * 0.05
            })
            .collect();
        let y_channel = orig.clone();
        let mut dest = orig.clone();

        let before_rmse: f32 =
            (orig.iter().map(|&v| (v - clean_val).powi(2)).sum::<f32>() / n as f32).sqrt();

        denoise_channel(
            &mut dest, &orig, &y_channel, width, height, &noise_lut, 0.25,
        );

        let after_rmse: f32 =
            (dest.iter().map(|&v| (v - clean_val).powi(2)).sum::<f32>() / n as f32).sqrt();

        assert!(
            after_rmse < before_rmse,
            "Denoising should reduce RMSE: before={}, after={}",
            before_rmse,
            after_rmse,
        );
    }

    #[test]
    fn test_denoise_small_image() {
        let width = 12;
        let height = 8;
        let noise_lut = [0.05f32; NUM_NOISE_POINTS];
        let denoise_scale = 0.25;

        let (orig, y_channel) = make_test_data(width, height);
        let mut dest_scalar = orig.clone();
        denoise_channel_scalar(
            &mut dest_scalar,
            &orig,
            &y_channel,
            width,
            height,
            &noise_lut,
            denoise_scale,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut dest_dispatch = orig.clone();
                denoise_channel(
                    &mut dest_dispatch,
                    &orig,
                    &y_channel,
                    width,
                    height,
                    &noise_lut,
                    denoise_scale,
                );
                for i in 0..orig.len() {
                    let diff = (dest_scalar[i] - dest_dispatch[i]).abs();
                    assert!(
                        diff < 1e-4,
                        "Mismatch at pixel {i} ({width}x{height}): scalar={} dispatch={} diff={diff} [{perm}]",
                        dest_scalar[i],
                        dest_dispatch[i],
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }
}
