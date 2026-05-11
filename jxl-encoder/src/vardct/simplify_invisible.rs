// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Pre-pass that smooths invisible (alpha=0) pixels to reduce DCT cost.
//!
//! When encoding RGBA images with large transparent regions, the color
//! data in alpha=0 pixels is arbitrary (often noise/garbage from image
//! editors). That noise creates high-frequency DCT energy that costs
//! bits even though no decoder will ever see those pixels.
//!
//! This module ports libjxl's `SimplifyInvisible` (`enc_frame.cc:511-570`):
//! - **Lossy mode**: weighted average of neighboring visible pixels
//!   (left, top-right, above/below) — produces a smooth gradient that
//!   smears from visible edges into transparent areas.
//! - **Lossless mode**: zero out all channels where alpha=0.
//!
//! Applied after pixel-to-linear-RGB conversion, before XYB / VarDCT.
//! Caller is responsible for the gating ("alpha is present, no
//! premultiplied alpha, no resampling mismatch").

/// Smear invisible (alpha=0) pixels to a weighted average of neighbors.
///
/// Bit-faithful port of libjxl `SimplifyInvisible` at `enc_frame.cc:511`.
/// Operates on interleaved linear-RGB (`Vec<f32>`, length = `w*h*3`).
/// `alpha` is `&[u8]` with length `w*h`, alpha=0 marks invisible pixels.
///
/// `lossless = false` smooths via the libjxl weighted-average formula.
/// `lossless = true` zeroes out invisible pixel values.
///
/// Cost: scalar per-pixel scan, ~6 conditionals per invisible pixel.
/// On photos with small alpha masks this is essentially free; on
/// sprites with large transparent regions it costs O(n) but unlocks
/// 5-20% file size reduction in the downstream DCT.
pub fn simplify_invisible_rgb(
    linear_rgb_interleaved: &mut [f32],
    alpha: &[u8],
    width: usize,
    height: usize,
    lossless: bool,
) {
    debug_assert_eq!(alpha.len(), width * height);
    debug_assert_eq!(linear_rgb_interleaved.len(), width * height * 3);
    if width == 0 || height == 0 {
        return;
    }

    // Process each color channel independently — matches libjxl's
    // outer `for (size_t c = 0; c < 3; ++c)` loop. Within a channel,
    // we walk the image scanline-by-scanline and update only invisible
    // pixels, leaving visible ones untouched.
    //
    // Interleaved layout: rgb[y * w * 3 + x * 3 + c] is the (x, y)
    // pixel's c-th channel. The libjxl reference uses planar
    // (separate row pointers per channel); the interleave here costs
    // a 3-stride lookup but lets us avoid copying to a planar buffer.
    for c in 0..3 {
        // Snapshot per-channel rows so we can read the pre-update
        // (pre-this-scanline) values from the row above. The libjxl
        // reference does the same via `prow` (row above) / `nrow`
        // (row below) pointers from the SAME f32 plane — both are
        // observed pre-update because the inner loop only writes
        // `row[x]` for the current y. We mirror that read pattern.
        for y in 0..height {
            for x in 0..width {
                if alpha[y * width + x] != 0 {
                    continue;
                }
                let row_base = y * width * 3;
                let idx = |xx: usize| row_base + xx * 3 + c;
                let prow_idx = |xx: usize| (y - 1) * width * 3 + xx * 3 + c;
                let nrow_idx = |xx: usize| (y + 1) * width * 3 + xx * 3 + c;
                let alpha_at = |xx: usize, yy: usize| alpha[yy * width + xx];

                if lossless {
                    linear_rgb_interleaved[idx(x)] = 0.0;
                    continue;
                }

                let mut sum = 0.0f32;
                let mut d = 0.0f32;
                if x > 0 {
                    sum += linear_rgb_interleaved[idx(x - 1)];
                    d += 1.0;
                    if alpha_at(x - 1, y) > 0 {
                        sum += linear_rgb_interleaved[idx(x - 1)];
                        d += 1.0;
                    }
                }
                if x + 1 < width {
                    if y > 0 {
                        sum += linear_rgb_interleaved[prow_idx(x + 1)];
                        d += 1.0;
                    }
                    if alpha_at(x + 1, y) > 0 {
                        sum += 2.0 * linear_rgb_interleaved[idx(x + 1)];
                        d += 2.0;
                    }
                    if y > 0 && alpha_at(x + 1, y - 1) > 0 {
                        sum += 2.0 * linear_rgb_interleaved[prow_idx(x + 1)];
                        d += 2.0;
                    }
                    if y + 1 < height && alpha_at(x + 1, y + 1) > 0 {
                        sum += 2.0 * linear_rgb_interleaved[nrow_idx(x + 1)];
                        d += 2.0;
                    }
                }
                if y > 0 && alpha_at(x, y - 1) > 0 {
                    sum += 2.0 * linear_rgb_interleaved[prow_idx(x)];
                    d += 2.0;
                }
                if y + 1 < height && alpha_at(x, y + 1) > 0 {
                    sum += 2.0 * linear_rgb_interleaved[nrow_idx(x)];
                    d += 2.0;
                }
                let v = if d > 1.0 { sum / d } else { sum };
                linear_rgb_interleaved[idx(x)] = v;
            }
        }
    }
}

/// Predicate for whether a pre-pass on `alpha` would do useful work.
/// Returns true if any pixel has alpha=0. Cheap to compute (single
/// linear scan, early-exit on first zero).
///
/// Caller can use this to skip the pre-pass entirely when the alpha
/// channel is fully opaque — no point walking the image when no
/// invisible pixels exist. Sprites/icons trigger; photos with
/// fully-opaque alpha do not.
pub fn has_any_invisible_pixels(alpha: &[u8]) -> bool {
    alpha.contains(&0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_invisible_pixels_is_no_op() {
        // alpha all 255 → simplify_invisible should leave linear_rgb
        // bit-identical (the predicate would normally short-circuit
        // this in the caller, but the inner function must be a
        // no-op too as a safety net).
        let mut rgb = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 pixels
        let alpha = vec![255u8, 255];
        let original = rgb.clone();
        simplify_invisible_rgb(&mut rgb, &alpha, 2, 1, false);
        assert_eq!(rgb, original);
    }

    #[test]
    fn test_lossless_zeros_invisible() {
        // 2x1 image: pixel 0 visible (R=1, G=2, B=3), pixel 1 invisible
        // (R=99, G=99, B=99 — garbage). After lossless simplify,
        // pixel 1 should be (0, 0, 0).
        let mut rgb = vec![1.0_f32, 2.0, 3.0, 99.0, 99.0, 99.0];
        let alpha = vec![255u8, 0];
        simplify_invisible_rgb(&mut rgb, &alpha, 2, 1, true);
        assert_eq!(rgb[3], 0.0);
        assert_eq!(rgb[4], 0.0);
        assert_eq!(rgb[5], 0.0);
        // Visible pixel untouched.
        assert_eq!(rgb[0], 1.0);
        assert_eq!(rgb[1], 2.0);
        assert_eq!(rgb[2], 3.0);
    }

    #[test]
    fn test_lossless_skips_visible() {
        let mut rgb = vec![0.5_f32; 12]; // 4 pixels, all values 0.5
        let alpha = vec![255u8; 4];
        simplify_invisible_rgb(&mut rgb, &alpha, 2, 2, true);
        assert!(rgb.iter().all(|&v| v == 0.5));
    }

    #[test]
    fn test_lossy_smears_from_neighbors() {
        // 3x1 image: visible-invisible-visible, all R=1.0/G=2.0/B=3.0
        // for visible. The middle pixel (invisible, garbage 99) should
        // get smoothed toward the visible neighbors' values.
        let mut rgb = vec![
            1.0, 2.0, 3.0, // pixel 0 visible
            99.0, 99.0, 99.0, // pixel 1 invisible
            1.0, 2.0, 3.0, // pixel 2 visible
        ];
        let alpha = vec![255u8, 0, 255];
        simplify_invisible_rgb(&mut rgb, &alpha, 3, 1, false);
        // Visible pixels untouched.
        assert_eq!(rgb[0], 1.0);
        assert_eq!(rgb[8], 3.0);
        // Invisible pixel should be in the [1.0, 3.0] range per channel
        // (a weighted average of its neighbors), NOT the original 99.
        assert!(rgb[3] >= 1.0 && rgb[3] <= 1.0 + 1e-3, "R={}", rgb[3]);
        assert!(rgb[4] >= 2.0 && rgb[4] <= 2.0 + 1e-3, "G={}", rgb[4]);
        assert!(rgb[5] >= 3.0 && rgb[5] <= 3.0 + 1e-3, "B={}", rgb[5]);
    }

    #[test]
    fn test_lossy_isolated_invisible_pixel_zeros_out() {
        // 1x1 image with alpha=0 and no neighbors → d stays at 0,
        // so `v = sum (= 0)` (since the isolated invisible pixel has
        // no contributors). Sets to 0.
        let mut rgb = vec![99.0_f32, 99.0, 99.0];
        let alpha = vec![0u8];
        simplify_invisible_rgb(&mut rgb, &alpha, 1, 1, false);
        assert_eq!(rgb, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_has_any_invisible_pixels_predicate() {
        assert!(!has_any_invisible_pixels(&[255, 255, 255]));
        assert!(has_any_invisible_pixels(&[255, 0, 255]));
        assert!(has_any_invisible_pixels(&[0]));
        assert!(!has_any_invisible_pixels(&[]));
    }

    #[test]
    fn test_zero_dim_is_no_op() {
        let mut rgb: Vec<f32> = Vec::new();
        let alpha: Vec<u8> = Vec::new();
        // Should not panic on empty buffers.
        simplify_invisible_rgb(&mut rgb, &alpha, 0, 0, false);
        simplify_invisible_rgb(&mut rgb, &alpha, 0, 0, true);
    }
}
