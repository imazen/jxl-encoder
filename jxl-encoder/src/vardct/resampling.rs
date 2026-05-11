// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Input downsampling for high-distance encoding (refs #12).
//!
//! At high distances (libjxl auto-selects at d ≥ 10), per-pixel
//! fidelity matters less than overall structure. Downsampling the
//! input by a small integer factor (2/4/8) before encoding +
//! signaling the decoder to upsample after decoding can dramatically
//! cut file size at the same perceived quality.
//!
//! This module ports the simple box-filter half of libjxl's
//! `image_ops.cc:44-98` (used for 4× and 8×). The 2× case in
//! libjxl uses a 12×12 sharper kernel for better quality; we keep
//! both cases on the simple box filter for now (foundation only —
//! the sharper 2× kernel + the auto-select-at-d=10 logic + the
//! frame-header `upsampling` wire-up are TBD).
//!
//! libjxl reference: `enc_heuristics.cc:279-405` (sharper 12×12)
//! and `image_ops.cc:44-98` (simple box).

extern crate alloc;
use alloc::vec::Vec;

/// Box-filter downsample interleaved RGB (3 channels per pixel) by
/// an integer factor. Output dimensions are
/// `(width.div_ceil(factor), height.div_ceil(factor))` and each
/// output sample is the unweighted mean of the up to `factor × factor`
/// source samples that fall inside its footprint (clipped at the
/// right / bottom edges).
///
/// `factor` must be one of `1`, `2`, `4`, `8` — the JPEG XL spec's
/// allowed `upsampling` values. Returns `(downsampled_rgb,
/// out_width, out_height)`.
///
/// The factor=1 case is a clone (no downsampling); included so
/// callers can dispatch without a special-case.
pub fn box_downsample_rgb(
    rgb_interleaved: &[f32],
    width: usize,
    height: usize,
    factor: u32,
) -> (Vec<f32>, u32, u32) {
    debug_assert!(matches!(factor, 1 | 2 | 4 | 8), "factor must be 1/2/4/8");
    debug_assert_eq!(rgb_interleaved.len(), width * height * 3);
    if factor == 1 {
        return (rgb_interleaved.to_vec(), width as u32, height as u32);
    }
    let f = factor as usize;
    let out_w = width.div_ceil(f);
    let out_h = height.div_ceil(f);
    let mut out = Vec::with_capacity(out_w * out_h * 3);
    for oy in 0..out_h {
        let y0 = oy * f;
        let y1 = (y0 + f).min(height);
        for ox in 0..out_w {
            let x0 = ox * f;
            let x1 = (x0 + f).min(width);
            let mut sum = [0.0f32; 3];
            let mut count = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * width + x) * 3;
                    sum[0] += rgb_interleaved[idx];
                    sum[1] += rgb_interleaved[idx + 1];
                    sum[2] += rgb_interleaved[idx + 2];
                    count += 1;
                }
            }
            let inv = 1.0 / count as f32;
            out.push(sum[0] * inv);
            out.push(sum[1] * inv);
            out.push(sum[2] * inv);
        }
    }
    (out, out_w as u32, out_h as u32)
}

/// Box-filter downsample a single-channel u8 buffer (alpha) by an
/// integer factor. Same semantics as [`box_downsample_rgb`] but for
/// 1 byte/pixel inputs. Output values are rounded.
pub fn box_downsample_alpha_u8(
    alpha: &[u8],
    width: usize,
    height: usize,
    factor: u32,
) -> (Vec<u8>, u32, u32) {
    debug_assert!(matches!(factor, 1 | 2 | 4 | 8), "factor must be 1/2/4/8");
    debug_assert_eq!(alpha.len(), width * height);
    if factor == 1 {
        return (alpha.to_vec(), width as u32, height as u32);
    }
    let f = factor as usize;
    let out_w = width.div_ceil(f);
    let out_h = height.div_ceil(f);
    let mut out = Vec::with_capacity(out_w * out_h);
    for oy in 0..out_h {
        let y0 = oy * f;
        let y1 = (y0 + f).min(height);
        for ox in 0..out_w {
            let x0 = ox * f;
            let x1 = (x0 + f).min(width);
            let mut sum = 0u32;
            let mut count = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += alpha[y * width + x] as u32;
                    count += 1;
                }
            }
            // Round-to-nearest-even isn't strictly needed; (sum + count/2) / count
            // is fine for the alpha-quantization use case.
            out.push(((sum + count / 2) / count) as u8);
        }
    }
    (out, out_w as u32, out_h as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_1_is_clone() {
        let rgb = vec![0.5_f32, 0.25, 0.75, 0.1, 0.2, 0.3]; // 2 px
        let (out, w, h) = box_downsample_rgb(&rgb, 2, 1, 1);
        assert_eq!(out, rgb);
        assert_eq!(w, 2);
        assert_eq!(h, 1);
    }

    #[test]
    fn test_factor_2_uniform_input_yields_uniform_output() {
        // 4×4 RGB image filled with (0.5, 0.25, 0.75); 2× downsample → 2×2
        // each output pixel = mean of 4 identical inputs = same value.
        let rgb = vec![0.5_f32, 0.25, 0.75].into_iter().cycle().take(4 * 4 * 3).collect::<Vec<_>>();
        let (out, w, h) = box_downsample_rgb(&rgb, 4, 4, 2);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(out.len(), 2 * 2 * 3);
        for chunk in out.chunks_exact(3) {
            assert!((chunk[0] - 0.5).abs() < 1e-6);
            assert!((chunk[1] - 0.25).abs() < 1e-6);
            assert!((chunk[2] - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn test_factor_2_averages_2x2_block() {
        // 2×2 RGB image with distinct corner values (R channel only):
        // [1, 0, 0,  2, 0, 0,
        //  3, 0, 0,  4, 0, 0]
        // 2× downsample → 1×1 with R = (1+2+3+4)/4 = 2.5.
        let rgb = vec![
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0,
            3.0, 0.0, 0.0,
            4.0, 0.0, 0.0,
        ];
        let (out, w, h) = box_downsample_rgb(&rgb, 2, 2, 2);
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        assert!((out[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_factor_4_handles_partial_edge() {
        // 5×3 image, factor 4 → 2×1 output: first cell averages cols 0..4 rows 0..3
        // (12 cells), second cell averages just col 4, rows 0..3 (3 cells).
        let mut rgb = Vec::new();
        for y in 0..3u32 {
            for x in 0..5u32 {
                let v = (x + y * 5) as f32;
                rgb.push(v);
                rgb.push(v);
                rgb.push(v);
            }
        }
        let (out, w, h) = box_downsample_rgb(&rgb, 5, 3, 4);
        assert_eq!(w, 2);
        assert_eq!(h, 1);
        // Cell 0: averages cols 0..4, rows 0..3 → 12 cells with values
        // {0..3, 5..8, 10..13}. Mean = (0+1+2+3 + 5+6+7+8 + 10+11+12+13) / 12
        // = (6 + 26 + 46) / 12 = 78 / 12 = 6.5.
        assert!((out[0] - 6.5).abs() < 1e-4);
        // Cell 1: averages col 4, rows 0..3 → values 4, 9, 14. Mean = 27/3 = 9.0.
        assert!((out[3] - 9.0).abs() < 1e-4);
    }

    #[test]
    fn test_factor_8() {
        // 8×8 uniform RGB image, factor 8 → 1×1 output equal to input value.
        let rgb = vec![0.42_f32; 8 * 8 * 3];
        let (out, w, h) = box_downsample_rgb(&rgb, 8, 8, 8);
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        assert!((out[0] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_factor_2_averages() {
        // 2×2 alpha: [255, 0, 0, 255]. Factor 2 → 1×1 with mean ≈ 128.
        let alpha = vec![255u8, 0, 0, 255];
        let (out, w, h) = box_downsample_alpha_u8(&alpha, 2, 2, 2);
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        // (255 + 0 + 0 + 255 + 2) / 4 = 512 / 4 = 128.
        assert_eq!(out[0], 128);
    }

    #[test]
    fn test_alpha_factor_1_is_clone() {
        let alpha = vec![1u8, 2, 3, 4, 5];
        let (out, w, h) = box_downsample_alpha_u8(&alpha, 5, 1, 1);
        assert_eq!(out, alpha);
        assert_eq!(w, 5);
        assert_eq!(h, 1);
    }
}
