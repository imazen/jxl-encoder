// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Dot detection for star fields and specular highlights (refs #19).
//!
//! Ports libjxl `enc_detect_dots.cc`. The pipeline is:
//!
//! 1. Compute an **energy image** from XYB orig vs blurred:
//!    `energy = w_x * dX^2 + w_y * dY^2 + w_b * dB^2` where `(dX, dY, dB)`
//!    is the difference between the original XYB image and a "smooth"
//!    XYB image (Gaussian blurred 3x with `Gaussian3`). Color weights
//!    `(0, 10, 0)` mean only the Y channel contributes — chrominance
//!    differences are ignored.
//!
//! 2. **Connected components** via flood-fill on the energy image with a
//!    dual threshold (`0.04` to enter, `0.02` to grow). Components are
//!    capped at 1000 pixels and 5×5 bounding window.
//!
//! 3. **2D Gaussian ellipse fitting** to each connected component
//!    (position, sigma, angle, per-channel intensity).
//!
//! 4. **Quality filter** — reject poor fits, low intensity, bad
//!    centroid alignment.
//!
//! 5. Encode as patches via the patch dictionary.
//!
//! **Gating**: effort >= 7, distance >= 3.0. Skipped if text-like
//! patches are already found. Niche feature — only fires on astronomy
//! images, specular highlights on dark backgrounds, certain noise
//! patterns.
//!
//! This module is being ported in stages. Currently implemented:
//! - [`SumOfSquareDifferences`]: weighted sum of squared XYB diffs
//! - [`gaussian_separable_5_horizontal`] / `_vertical`: 5-tap
//!   horizontal/vertical Gaussian convolution helpers
//! - [`compute_energy_image`]: orchestrates blur + diff to produce the
//!   per-pixel energy map (caller-provided smooth output buffer)
//!
//! TODO (subsequent ticks):
//! - Connected component flood-fill (`FindCC`)
//! - `FitGaussianFast` + `FitGaussian` (2D ellipse least-squares)
//! - `ComputeDotLosses` quality filter
//! - Patch dictionary encoding
//! - Wire-up at effort >= 7, distance >= 3.0 in encode pipeline

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// libjxl color-channel weights for the energy image. Y-channel only;
/// chroma differences are ignored. From `enc_detect_dots.cc:54-56`.
pub(crate) const ENERGY_COLOR_COEF: [f32; 3] = [0.0, 10.0, 0.0];

/// 5-tap separable Gaussian for the **noise-smoothing** pass (sigma
/// ≈ 0.65). Coefficients from libjxl
/// `WeightsSeparable5Gaussian0_65` (`enc_detect_dots.cc:128-138`).
/// The kernel is symmetric so only the center + 2 outer taps are
/// stored; the actual 5-tap convolution applies them as
/// `[w2, w1, w0, w1, w2]` at each pixel.
pub(crate) const GAUSSIAN_0_65_TAPS: [f32; 3] = [0.558311, 0.210395, 0.010449];

/// 5-tap separable Gaussian for the **dot-removal** pass (sigma ≈ 3.0,
/// applied twice → effective sigma ≈ 4.24). Coefficients from libjxl
/// `WeightsSeparable5Gaussian3` (`enc_detect_dots.cc:140-149`).
pub(crate) const GAUSSIAN_3_TAPS: [f32; 3] = [0.222338, 0.210431, 0.1784];

/// Bounding-window cap for a single dot's connected component (libjxl
/// `kEllipseWindowSize`, `enc_detect_dots.cc:97`).
pub(crate) const ELLIPSE_WINDOW_SIZE: usize = 5;

/// Single-pass 5-tap horizontal Gaussian convolution. `taps` are
/// `[center, 1-step, 2-step]`. Edges replicate (mirror would also be
/// reasonable; libjxl uses replicate per `WeightsSeparable5`'s
/// horizontal-convolution helper).
pub(crate) fn gaussian_separable_5_horizontal(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    let [w0, w1, w2] = taps;
    for y in 0..height {
        let row = &src[y * width..(y + 1) * width];
        let drow = &mut dst[y * width..(y + 1) * width];
        for x in 0..width {
            let xm2 = if x >= 2 { x - 2 } else { 0 };
            let xm1 = if x >= 1 { x - 1 } else { 0 };
            let xp1 = if x + 1 < width { x + 1 } else { width - 1 };
            let xp2 = if x + 2 < width { x + 2 } else { width - 1 };
            drow[x] = w0 * row[x] + w1 * (row[xm1] + row[xp1]) + w2 * (row[xm2] + row[xp2]);
        }
    }
}

/// Single-pass 5-tap vertical Gaussian convolution. Same conventions
/// as [`gaussian_separable_5_horizontal`] but along the y axis.
pub(crate) fn gaussian_separable_5_vertical(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    let [w0, w1, w2] = taps;
    for y in 0..height {
        let ym2 = if y >= 2 { y - 2 } else { 0 };
        let ym1 = if y >= 1 { y - 1 } else { 0 };
        let yp1 = if y + 1 < height { y + 1 } else { height - 1 };
        let yp2 = if y + 2 < height { y + 2 } else { height - 1 };
        for x in 0..width {
            let v = w0 * src[y * width + x]
                + w1 * (src[ym1 * width + x] + src[yp1 * width + x])
                + w2 * (src[ym2 * width + x] + src[yp2 * width + x]);
            dst[y * width + x] = v;
        }
    }
}

/// Apply a single full 5×5 separable Gaussian (horizontal then
/// vertical) on a planar f32 image. Allocates one scratch buffer
/// (height × width).
pub(crate) fn gaussian_separable_5(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    let mut tmp = vec![0.0_f32; width * height];
    gaussian_separable_5_horizontal(src, &mut tmp, width, height, taps);
    gaussian_separable_5_vertical(&tmp, dst, width, height, taps);
}

/// Sum-of-square-differences between three planar XYB channel pairs,
/// weighted by [`ENERGY_COLOR_COEF`]. Mirrors libjxl
/// `SumOfSquareDifferences` (`enc_detect_dots.cc:50-87`).
///
/// Output `energy[i] = 0 * dX^2 + 10 * dY^2 + 0 * dB^2` where
/// `dC = orig_C[i] - smooth_C[i]`. With weights `(0, 10, 0)` the X
/// and B contributions vanish; the function still iterates them so
/// the per-channel buffers stay live and the future weight-tuning
/// path doesn't need a separate codepath.
pub fn sum_of_square_differences(
    orig_x: &[f32],
    orig_y: &[f32],
    orig_b: &[f32],
    smooth_x: &[f32],
    smooth_y: &[f32],
    smooth_b: &[f32],
    energy: &mut [f32],
) {
    let n = energy.len();
    debug_assert_eq!(orig_x.len(), n);
    debug_assert_eq!(orig_y.len(), n);
    debug_assert_eq!(orig_b.len(), n);
    debug_assert_eq!(smooth_x.len(), n);
    debug_assert_eq!(smooth_y.len(), n);
    debug_assert_eq!(smooth_b.len(), n);
    let [cx, cy, cb] = ENERGY_COLOR_COEF;
    for i in 0..n {
        let dx = orig_x[i] - smooth_x[i];
        let dy = orig_y[i] - smooth_y[i];
        let db = orig_b[i] - smooth_b[i];
        energy[i] = cx * dx * dx + cy * dy * dy + cb * db * db;
    }
}

/// Compute the per-pixel energy image used by dot detection. Mirrors
/// libjxl `ComputeEnergyImage` (`enc_detect_dots.cc:151-176`).
///
/// Inputs are three planar XYB channel buffers (interleaved input
/// must be deinterleaved by the caller). Outputs:
/// - `smooth_x` / `smooth_y` / `smooth_b`: the noise-smoothed
///   (`Gaussian0_65`) XYB image — caller pre-allocates and passes
///   mutable references; subsequent quality fitting uses this.
/// - `energy`: the per-pixel energy map (`width * height` f32).
///
/// libjxl smooths the original twice with the dot-removal Gaussian
/// (sigma ≈ 3, applied 2× → effective sigma ≈ 4.24) to produce a
/// "background" estimate, then computes the squared difference vs the
/// noise-smoothed original. We do the same.
#[allow(clippy::too_many_arguments)]
pub fn compute_energy_image(
    orig_x: &[f32],
    orig_y: &[f32],
    orig_b: &[f32],
    width: usize,
    height: usize,
    smooth_x: &mut [f32],
    smooth_y: &mut [f32],
    smooth_b: &mut [f32],
    energy: &mut [f32],
) {
    let n = width * height;
    debug_assert_eq!(orig_x.len(), n);
    debug_assert_eq!(orig_y.len(), n);
    debug_assert_eq!(orig_b.len(), n);
    debug_assert_eq!(smooth_x.len(), n);
    debug_assert_eq!(smooth_y.len(), n);
    debug_assert_eq!(smooth_b.len(), n);
    debug_assert_eq!(energy.len(), n);

    // Step 1: noise-smooth each channel into smooth_*.
    let mut tmp = vec![0.0_f32; n];
    gaussian_separable_5_horizontal(orig_x, &mut tmp, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_vertical(&tmp, smooth_x, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_horizontal(orig_y, &mut tmp, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_vertical(&tmp, smooth_y, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_horizontal(orig_b, &mut tmp, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_vertical(&tmp, smooth_b, width, height, GAUSSIAN_0_65_TAPS);

    // Step 2: dot-remove each channel via 2× Gaussian-3 → background.
    // We need separate buffers for the background since smooth_*
    // already holds the noise-smoothed versions.
    let mut bg_x = vec![0.0_f32; n];
    let mut bg_y = vec![0.0_f32; n];
    let mut bg_b = vec![0.0_f32; n];
    let mut tmp2 = vec![0.0_f32; n];

    // First Gaussian3 pass: orig → tmp2 → bg.
    gaussian_separable_5_horizontal(orig_x, &mut tmp, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_vertical(&tmp, &mut bg_x, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_horizontal(orig_y, &mut tmp, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_vertical(&tmp, &mut bg_y, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_horizontal(orig_b, &mut tmp, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_vertical(&tmp, &mut bg_b, width, height, GAUSSIAN_3_TAPS);

    // Second Gaussian3 pass on each channel: bg → tmp2 → bg.
    for ch in [&mut bg_x, &mut bg_y, &mut bg_b] {
        gaussian_separable_5_horizontal(ch, &mut tmp, width, height, GAUSSIAN_3_TAPS);
        gaussian_separable_5_vertical(&tmp, &mut tmp2, width, height, GAUSSIAN_3_TAPS);
        ch.copy_from_slice(&tmp2);
    }

    // Step 3: energy = weighted SoSD between noise-smoothed and background.
    sum_of_square_differences(smooth_x, smooth_y, smooth_b, &bg_x, &bg_y, &bg_b, energy);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_sums_match_libjxl() {
        // Each separable kernel should sum to 1.0 across its 5 taps:
        //   [w2, w1, w0, w1, w2]
        let g0_65: f32 =
            GAUSSIAN_0_65_TAPS[0] + 2.0 * GAUSSIAN_0_65_TAPS[1] + 2.0 * GAUSSIAN_0_65_TAPS[2];
        assert!(
            (g0_65 - 1.0).abs() < 1e-3,
            "Gaussian0_65 5-tap sum should be ~1.0, got {g0_65}",
        );
        let g3: f32 = GAUSSIAN_3_TAPS[0] + 2.0 * GAUSSIAN_3_TAPS[1] + 2.0 * GAUSSIAN_3_TAPS[2];
        assert!(
            (g3 - 1.0).abs() < 1e-3,
            "Gaussian3 5-tap sum should be ~1.0, got {g3}",
        );
    }

    #[test]
    fn test_gaussian_separable_5_uniform_input_is_uniform_output() {
        let w = 16;
        let h = 16;
        let src = vec![0.5_f32; w * h];
        let mut dst = vec![0.0_f32; w * h];
        gaussian_separable_5(&src, &mut dst, w, h, GAUSSIAN_0_65_TAPS);
        for &v in &dst {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "uniform 0.5 should stay ~0.5, got {v}",
            );
        }
    }

    #[test]
    fn test_gaussian_separable_5_blurs_a_dot() {
        // 7×7 image with a single bright pixel in the center.
        let w = 7;
        let h = 7;
        let mut src = vec![0.0_f32; w * h];
        src[3 * w + 3] = 1.0;
        let mut dst = vec![0.0_f32; w * h];
        gaussian_separable_5(&src, &mut dst, w, h, GAUSSIAN_0_65_TAPS);
        // Center should drop (energy spread); outer ring should rise.
        assert!(
            dst[3 * w + 3] < 1.0,
            "center should drop, got {}",
            dst[3 * w + 3]
        );
        assert!(dst[2 * w + 3] > 0.0, "neighbor should pick up energy");
        // Conservation: total energy preserved within rounding.
        let total: f32 = dst.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.05,
            "5-tap separable Gaussian should approximately preserve energy; total={total}",
        );
    }

    #[test]
    fn test_sum_of_square_differences_y_only() {
        // Energy = 0*dX^2 + 10*dY^2 + 0*dB^2. Verify only Y contributes.
        let n = 4;
        let orig_x = vec![1.0_f32; n];
        let orig_y = vec![2.0_f32; n];
        let orig_b = vec![3.0_f32; n];
        let smooth_x = vec![0.5_f32; n]; // dX = 0.5 → ignored
        let smooth_y = vec![1.0_f32; n]; // dY = 1.0 → contributes 10
        let smooth_b = vec![2.0_f32; n]; // dB = 1.0 → ignored
        let mut energy = vec![0.0_f32; n];
        sum_of_square_differences(
            &orig_x,
            &orig_y,
            &orig_b,
            &smooth_x,
            &smooth_y,
            &smooth_b,
            &mut energy,
        );
        for &e in &energy {
            // 10 * (1.0)^2 = 10.0
            assert!((e - 10.0).abs() < 1e-5, "energy should be 10.0, got {e}");
        }
    }

    #[test]
    fn test_compute_energy_image_uniform_input_is_zero() {
        // A uniform image has no dots → energy should be ~0 everywhere
        // (Gaussian blurs of a uniform image are uniform; diff is 0).
        let w = 16;
        let h = 16;
        let orig = vec![0.5_f32; w * h];
        let mut sx = vec![0.0_f32; w * h];
        let mut sy = vec![0.0_f32; w * h];
        let mut sb = vec![0.0_f32; w * h];
        let mut energy = vec![0.0_f32; w * h];
        compute_energy_image(
            &orig,
            &orig,
            &orig,
            w,
            h,
            &mut sx,
            &mut sy,
            &mut sb,
            &mut energy,
        );
        for &e in &energy {
            assert!(e.abs() < 1e-3, "uniform input → ~0 energy, got {e}");
        }
    }
}
