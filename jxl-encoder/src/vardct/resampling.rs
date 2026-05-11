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

/// 12×12 weighted kernel used by [`sharper_downsample_2x_rgb`]. Ported
/// verbatim from libjxl `enc_heuristics.cc:286-345` (`kernel[144]`).
/// Symmetric in both axes; sums to ~1.0; emphasizes the central 2×2
/// (≈ 0.41 each) and balances negative outer lobes for sharpening.
const SHARPER_KERNEL_2X: [f32; 144] = [
    -0.000314256996835,
    -0.000314256996835,
    -0.000897597057705,
    -0.000562751488849,
    -0.000176807273646,
    0.001864627368902,
    0.001864627368902,
    -0.000176807273646,
    -0.000562751488849,
    -0.000897597057705,
    -0.000314256996835,
    -0.000314256996835,
    -0.000314256996835,
    -0.001527942804748,
    -0.000121760530512,
    0.000191123989093,
    0.010193185932466,
    0.058637519197110,
    0.058637519197110,
    0.010193185932466,
    0.000191123989093,
    -0.000121760530512,
    -0.001527942804748,
    -0.000314256996835,
    -0.000897597057705,
    -0.000121760530512,
    0.000946363683751,
    0.007113577630288,
    0.000437956841058,
    -0.000372823835211,
    -0.000372823835211,
    0.000437956841058,
    0.007113577630288,
    0.000946363683751,
    -0.000121760530512,
    -0.000897597057705,
    -0.000562751488849,
    0.000191123989093,
    0.007113577630288,
    0.044592622228814,
    0.000222278879007,
    -0.162864473015945,
    -0.162864473015945,
    0.000222278879007,
    0.044592622228814,
    0.007113577630288,
    0.000191123989093,
    -0.000562751488849,
    -0.000176807273646,
    0.010193185932466,
    0.000437956841058,
    0.000222278879007,
    -0.000913092543974,
    -0.017071696107902,
    -0.017071696107902,
    -0.000913092543974,
    0.000222278879007,
    0.000437956841058,
    0.010193185932466,
    -0.000176807273646,
    0.001864627368902,
    0.058637519197110,
    -0.000372823835211,
    -0.162864473015945,
    -0.017071696107902,
    0.414660099370354,
    0.414660099370354,
    -0.017071696107902,
    -0.162864473015945,
    -0.000372823835211,
    0.058637519197110,
    0.001864627368902,
    0.001864627368902,
    0.058637519197110,
    -0.000372823835211,
    -0.162864473015945,
    -0.017071696107902,
    0.414660099370354,
    0.414660099370354,
    -0.017071696107902,
    -0.162864473015945,
    -0.000372823835211,
    0.058637519197110,
    0.001864627368902,
    -0.000176807273646,
    0.010193185932466,
    0.000437956841058,
    0.000222278879007,
    -0.000913092543974,
    -0.017071696107902,
    -0.017071696107902,
    -0.000913092543974,
    0.000222278879007,
    0.000437956841058,
    0.010193185932466,
    -0.000176807273646,
    -0.000562751488849,
    0.000191123989093,
    0.007113577630288,
    0.044592622228814,
    0.000222278879007,
    -0.162864473015945,
    -0.162864473015945,
    0.000222278879007,
    0.044592622228814,
    0.007113577630288,
    0.000191123989093,
    -0.000562751488849,
    -0.000897597057705,
    -0.000121760530512,
    0.000946363683751,
    0.007113577630288,
    0.000437956841058,
    -0.000372823835211,
    -0.000372823835211,
    0.000437956841058,
    0.007113577630288,
    0.000946363683751,
    -0.000121760530512,
    -0.000897597057705,
    -0.000314256996835,
    -0.001527942804748,
    -0.000121760530512,
    0.000191123989093,
    0.010193185932466,
    0.058637519197110,
    0.058637519197110,
    0.010193185932466,
    0.000191123989093,
    -0.000121760530512,
    -0.001527942804748,
    -0.000314256996835,
    -0.000314256996835,
    -0.000314256996835,
    -0.000897597057705,
    -0.000562751488849,
    -0.000176807273646,
    0.001864627368902,
    0.001864627368902,
    -0.000176807273646,
    -0.000562751488849,
    -0.000897597057705,
    -0.000314256996835,
    -0.000314256996835,
];

const SHARPER_KERNEL_DIM: usize = 12;
const SHARPER_KERNEL_BOUND_R: usize = 5;

#[inline(always)]
fn store_min2(v: f32, min1: &mut f32, min2: &mut f32) {
    // Mirrors libjxl `StoreMin2` (enc_heuristics.cc:234-243).
    if v < *min2 {
        if v < *min1 {
            *min2 = *min1;
            *min1 = v;
        } else {
            *min2 = v;
        }
    }
}

/// Build the per-output-pixel mask used to clamp ringing in
/// [`sharper_downsample_2x_rgb`]. Each entry is the second-smallest of
/// the absolute differences between the (already box-downsampled)
/// pixel and its 4-neighbors. Mirrors libjxl
/// `enc_heuristics.cc:CreateMask` (lines 245-271).
fn create_ringing_mask(box_image: &[f32], width: usize, height: usize, out: &mut [f32]) {
    debug_assert_eq!(box_image.len(), width * height);
    debug_assert_eq!(out.len(), width * height);
    for y in 0..height {
        let row_n = if y > 0 { y - 1 } else { y };
        let row_s = if y + 1 < height { y + 1 } else { y };
        for x in 0..width {
            let c = box_image[y * width + x];
            let w = if x > 0 {
                box_image[y * width + x - 1]
            } else {
                c
            };
            let e = if x + 1 < width {
                box_image[y * width + x + 1]
            } else {
                c
            };
            let n = box_image[row_n * width + x];
            let s = box_image[row_s * width + x];
            let dw = (c - w).abs();
            let de = (c - e).abs();
            let dn = (c - n).abs();
            let ds = (c - s).abs();
            let mut min = f32::MAX;
            let mut min2 = f32::MAX;
            store_min2(dw, &mut min, &mut min2);
            store_min2(de, &mut min, &mut min2);
            store_min2(dn, &mut min, &mut min2);
            store_min2(ds, &mut min, &mut min2);
            out[y * width + x] = min2;
        }
    }
}

#[inline(always)]
fn clamp_idx(i: i64, max_exclusive: i64) -> usize {
    if i < 0 {
        0
    } else if i >= max_exclusive {
        (max_exclusive - 1) as usize
    } else {
        i as usize
    }
}

/// Per-channel scalar 12×12 sharper-2× kernel. `plane.len() == width *
/// height`. Output `out.len() == out_w * out_h` where `out_w =
/// div_ceil(width, 2)` and `out_h = div_ceil(height, 2)`. Edge clamps
/// indices and re-clamps output to the input's central-2×2 min/max
/// extended by `mask` to suppress ringing in smooth regions while
/// preserving edges. Mirrors libjxl `DownsampleImage2_Sharper` (single
/// channel) at `enc_heuristics.cc:279-405`.
fn sharper_downsample_2x_plane(
    plane: &[f32],
    width: usize,
    height: usize,
    out: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    debug_assert_eq!(plane.len(), width * height);
    debug_assert_eq!(out.len(), out_w * out_h);
    debug_assert_eq!(out_w, width.div_ceil(2));
    debug_assert_eq!(out_h, height.div_ceil(2));

    // Box-downsample first to seed the mask. libjxl uses its own
    // `DownsampleImage(_, 2)` (a 2×2 average) here.
    let mut box_plane = alloc::vec![0.0_f32; out_w * out_h];
    for oy in 0..out_h {
        let y0 = oy * 2;
        let y1 = (y0 + 2).min(height);
        for ox in 0..out_w {
            let x0 = ox * 2;
            let x1 = (x0 + 2).min(width);
            let mut sum = 0.0_f32;
            let mut count = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += plane[y * width + x];
                    count += 1;
                }
            }
            box_plane[oy * out_w + ox] = sum / count as f32;
        }
    }

    let mut mask = alloc::vec![0.0_f32; out_w * out_h];
    create_ringing_mask(&box_plane, out_w, out_h, &mut mask);

    let kernel_dim = SHARPER_KERNEL_DIM as i64;
    let xsize = width as i64;
    let ysize = height as i64;
    let half = (kernel_dim - 1) / 2; // 5

    for oy in 0..out_h {
        let row_mask_off = oy * out_w;
        for ox in 0..out_w {
            // Bound the output to the central R..kernely-R input
            // window's min/max — that's the 2×2 input footprint
            // directly under the output pixel (matches libjxl's R=5
            // restriction).
            let mut mn = f32::MAX;
            let mut mx = f32::MIN;
            for ky in SHARPER_KERNEL_BOUND_R as i64..(kernel_dim - SHARPER_KERNEL_BOUND_R as i64) {
                let iy = clamp_idx(oy as i64 * 2 + ky - half, ysize);
                let row = iy * width;
                for kx in
                    SHARPER_KERNEL_BOUND_R as i64..(kernel_dim - SHARPER_KERNEL_BOUND_R as i64)
                {
                    let ix = clamp_idx(ox as i64 * 2 + kx - half, xsize);
                    let v = plane[row + ix];
                    if v < mn {
                        mn = v;
                    }
                    if v > mx {
                        mx = v;
                    }
                }
            }

            // Apply full 12×12 kernel.
            let mut sum = 0.0_f32;
            for ky in 0..kernel_dim {
                let iy = clamp_idx(oy as i64 * 2 + ky - half, ysize);
                let row = iy * width;
                let kernel_row_off = (ky as usize) * SHARPER_KERNEL_DIM;
                for kx in 0..kernel_dim {
                    let ix = clamp_idx(ox as i64 * 2 + kx - half, xsize);
                    sum += plane[row + ix] * SHARPER_KERNEL_2X[kernel_row_off + kx as usize];
                }
            }

            let m = mask[row_mask_off + ox]; // mask_multiplier=1 in libjxl
            let lo = mn - m;
            let hi = mx + m;
            out[row_mask_off + ox] = sum.clamp(lo, hi);
        }
    }
}

/// Sharper 12×12 kernel-based 2× downsample (mirrors libjxl
/// `DownsampleImage2_Sharper` for the 3-channel image case). Operates
/// on **interleaved** RGB f32 input. Output dims are
/// `(width.div_ceil(2), height.div_ceil(2))`. Each channel runs
/// [`sharper_downsample_2x_plane`] independently — the kernel + mask
/// are per-channel in libjxl as well (it operates on `Image3F` plane
/// by plane via `DownsampleImage2_Sharper(opsin)`).
///
/// Compared with [`box_downsample_rgb`] at factor 2, this preserves
/// edge detail and reduces blocking artifacts in the upsampled
/// reconstruction; the cost is ~36× more arithmetic per output pixel
/// (144-tap vs 4-tap) and one auxiliary mask plane per channel.
pub fn sharper_downsample_2x_rgb(
    rgb_interleaved: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, u32, u32) {
    debug_assert_eq!(rgb_interleaved.len(), width * height * 3);
    let out_w = width.div_ceil(2);
    let out_h = height.div_ceil(2);

    // Deinterleave into 3 planes.
    let mut plane_r = alloc::vec![0.0_f32; width * height];
    let mut plane_g = alloc::vec![0.0_f32; width * height];
    let mut plane_b = alloc::vec![0.0_f32; width * height];
    for i in 0..(width * height) {
        plane_r[i] = rgb_interleaved[i * 3];
        plane_g[i] = rgb_interleaved[i * 3 + 1];
        plane_b[i] = rgb_interleaved[i * 3 + 2];
    }

    let mut out_r = alloc::vec![0.0_f32; out_w * out_h];
    let mut out_g = alloc::vec![0.0_f32; out_w * out_h];
    let mut out_b = alloc::vec![0.0_f32; out_w * out_h];
    sharper_downsample_2x_plane(&plane_r, width, height, &mut out_r, out_w, out_h);
    sharper_downsample_2x_plane(&plane_g, width, height, &mut out_g, out_w, out_h);
    sharper_downsample_2x_plane(&plane_b, width, height, &mut out_b, out_w, out_h);

    // Re-interleave.
    let mut out = Vec::with_capacity(out_w * out_h * 3);
    for i in 0..(out_w * out_h) {
        out.push(out_r[i]);
        out.push(out_g[i]);
        out.push(out_b[i]);
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
        let rgb = vec![0.5_f32, 0.25, 0.75]
            .into_iter()
            .cycle()
            .take(4 * 4 * 3)
            .collect::<Vec<_>>();
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
        let rgb = vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0];
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

    #[test]
    fn test_sharper_2x_dimensions() {
        // 64×64 → 32×32; 65×64 → 33×32 (div_ceil).
        let rgb = vec![0.5_f32; 64 * 64 * 3];
        let (out, w, h) = sharper_downsample_2x_rgb(&rgb, 64, 64);
        assert_eq!(w, 32);
        assert_eq!(h, 32);
        assert_eq!(out.len(), 32 * 32 * 3);
        let rgb = vec![0.5_f32; 65 * 64 * 3];
        let (_, w, h) = sharper_downsample_2x_rgb(&rgb, 65, 64);
        assert_eq!(w, 33);
        assert_eq!(h, 32);
    }

    #[test]
    fn test_sharper_2x_uniform_input_is_uniform() {
        // Uniform input → kernel sum × value. The kernel sums to ~1.0
        // by construction; confirm output is approximately the input.
        let rgb = vec![0.5_f32; 32 * 32 * 3];
        let (out, w, h) = sharper_downsample_2x_rgb(&rgb, 32, 32);
        assert_eq!(w, 16);
        assert_eq!(h, 16);
        for &v in &out {
            // Allow some kernel-sum drift, but expect within 1 %.
            assert!((v - 0.5).abs() < 0.005, "uniform 0.5 → {v}");
        }
    }

    #[test]
    fn test_sharper_2x_clamps_to_local_extrema() {
        // Synthetic "spike" image: one bright pixel surrounded by dark.
        // The clamp logic bounds the output to the central 2×2 input
        // region's [min - mask, max + mask]; for a uniform-dark
        // background away from the spike, output stays in the dark range.
        let mut rgb = vec![0.0_f32; 16 * 16 * 3];
        rgb[((8 * 16) + 8) * 3] = 1.0; // R-channel spike at (8,8)
        let (out, _, _) = sharper_downsample_2x_rgb(&rgb, 16, 16);
        // Far-corner output (0,0) sees no spike in its 2×2 input window
        // (covers input (0..2, 0..2)) — must be clamped to ~0.
        assert!(
            out[0] < 0.01,
            "far corner R should clamp to ~0; got {}",
            out[0]
        );
    }

    #[test]
    fn test_sharper_kernel_sum_close_to_1() {
        let s: f32 = SHARPER_KERNEL_2X.iter().sum();
        // libjxl's kernel sums to within rounding of 1.0; verify our
        // copy matches.
        assert!((s - 1.0).abs() < 0.01, "kernel sum {s} should be ~1.0");
    }
}
