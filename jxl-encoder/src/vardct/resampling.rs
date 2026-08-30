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
//! This module ports libjxl's three downsamplers: the simple box filter
//! (`image_ops.cc:44-98`, used for 4× and 8×), the 12×12 sharper 2×
//! kernel (`enc_heuristics.cc:279-405`, our 2× default at effort ≤ 9),
//! and the iterative 2× refinement (`enc_heuristics.cc:425-780`,
//! `DownsampleImage2_Iterative` — used at effort ≥ 10 to mirror libjxl's
//! `speed_tier <= kGlacier` gate at `enc_frame.cc:752`; issue #45).

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
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<(Vec<f32>, u32, u32)> {
    debug_assert!(matches!(factor, 1 | 2 | 4 | 8), "factor must be 1/2/4/8");
    debug_assert_eq!(rgb_interleaved.len(), width * height * 3);
    if factor == 1 {
        return Ok((rgb_interleaved.to_vec(), width as u32, height as u32));
    }
    let f = factor as usize;
    let out_w = width.div_ceil(f);
    let out_h = height.div_ceil(f);
    // Dimension-driven output buffer — honor the runtime fallible-alloc policy;
    // byte-identical when infallible.
    let mut out = crate::budget::vec_with_capacity_fallible(
        budget.is_some_and(|b| b.is_fallible()),
        out_w * out_h * 3,
    )?;
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
    Ok((out, out_w as u32, out_h as u32))
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
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<(Vec<f32>, u32, u32)> {
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

    // Re-interleave. Dimension-driven output buffer — honor the runtime
    // fallible-alloc policy; byte-identical when infallible.
    let mut out = crate::budget::vec_with_capacity_fallible(
        budget.is_some_and(|b| b.is_fallible()),
        out_w * out_h * 3,
    )?;
    for i in 0..(out_w * out_h) {
        out.push(out_r[i]);
        out.push(out_g[i]);
        out.push(out_b[i]);
    }
    Ok((out, out_w as u32, out_h as u32))
}

// ── Iterative 2× downsampler (libjxl `DownsampleImage2_Iterative`) ──────
//
// Ported from libjxl `enc_heuristics.cc:425-780` (@ d089091). libjxl uses
// this at `speed_tier <= kGlacier` (effort ≥ 10) when `upsampling == 2`:
// it refines the sharper-kernel result so that the DECODER's default 2×
// upsampler reproduces the original as closely as possible (3 rounds of
// gradient correction through the upsampler's adjoint), then clamps
// ringing against the box-downsampled neighborhood. libjxl applies it to
// the opsin (XYB) planes; we apply it to linear RGB pre-XYB, matching the
// existing sharper-2× wiring (documented divergence — see
// docs/LIBJXL_DIVERGENCES.md).
//
// The libjxl `Image3F` wrapper also builds a butteraugli mask it never
// uses (`mask` / `mask_fuzzy` in `DownsampleImage2_Iterative(Image3F*)`
// are dead stores upstream); we port only the live per-plane algorithm.

/// The decoder's default 2× upsampling kernels (`dec_upsample.cc` default
/// `CustomTransformData`), transcribed from libjxl
/// `enc_heuristics.cc:429-460`. One 5×5 kernel per output-pixel phase
/// (even/odd x × even/odd y).
const UPSAMPLE2_KERNEL_00: [f32; 25] = [
    -0.01716200,
    -0.03452303,
    -0.04022174,
    -0.02921014,
    -0.00624645, //
    -0.03452303,
    0.14111091,
    0.28896755,
    0.00278718,
    -0.01610267, //
    -0.04022174,
    0.28896755,
    0.56661550,
    0.03777607,
    -0.01986694, //
    -0.02921014,
    0.00278718,
    0.03777607,
    -0.03144731,
    -0.01185068, //
    -0.00624645,
    -0.01610267,
    -0.01986694,
    -0.01185068,
    -0.00213539,
];
const UPSAMPLE2_KERNEL_01: [f32; 25] = [
    -0.00624645,
    -0.01610267,
    -0.01986694,
    -0.01185068,
    -0.00213539, //
    -0.02921014,
    0.00278718,
    0.03777607,
    -0.03144731,
    -0.01185068, //
    -0.04022174,
    0.28896755,
    0.56661550,
    0.03777607,
    -0.01986694, //
    -0.03452303,
    0.14111091,
    0.28896755,
    0.00278718,
    -0.01610267, //
    -0.01716200,
    -0.03452303,
    -0.04022174,
    -0.02921014,
    -0.00624645,
];
const UPSAMPLE2_KERNEL_10: [f32; 25] = [
    -0.00624645,
    -0.02921014,
    -0.04022174,
    -0.03452303,
    -0.01716200, //
    -0.01610267,
    0.00278718,
    0.28896755,
    0.14111091,
    -0.03452303, //
    -0.01986694,
    0.03777607,
    0.56661550,
    0.28896755,
    -0.04022174, //
    -0.01185068,
    -0.03144731,
    0.03777607,
    0.00278718,
    -0.02921014, //
    -0.00213539,
    -0.01185068,
    -0.01986694,
    -0.01610267,
    -0.00624645,
];
const UPSAMPLE2_KERNEL_11: [f32; 25] = [
    -0.00213539,
    -0.01185068,
    -0.01986694,
    -0.01610267,
    -0.00624645, //
    -0.01185068,
    -0.03144731,
    0.03777607,
    0.00278718,
    -0.02921014, //
    -0.01986694,
    0.03777607,
    0.56661550,
    0.28896755,
    -0.04022174, //
    -0.01610267,
    0.00278718,
    0.28896755,
    0.14111091,
    -0.03452303, //
    -0.00624645,
    -0.02921014,
    -0.04022174,
    -0.03452303,
    -0.01716200,
];

/// 5×5 kernel side (libjxl `kSize`).
const UPSAMPLE2_KSIZE: i64 = 5;

#[inline]
fn upsample2_kernel(x: i64, y: i64) -> &'static [f32; 25] {
    match ((x & 1) != 0, (y & 1) != 0) {
        (true, true) => &UPSAMPLE2_KERNEL_11,
        (true, false) => &UPSAMPLE2_KERNEL_10,
        (false, true) => &UPSAMPLE2_KERNEL_01,
        (false, false) => &UPSAMPLE2_KERNEL_00,
    }
}

/// The decoder's default 2× upsampler on one plane (libjxl
/// `enc_heuristics.cc:UpsampleImage`, itself a mirror of `dec_upsample`
/// with default `CustomTransformData`): 5×5 kernel per phase with
/// edge-clamped taps, output clamped to the support's min/max.
/// `input` is `in_w × in_h`; `out` is `out_w × out_h` (the full-res
/// dims, `out_w.div_ceil(2) == in_w`).
fn upsample2_plane(
    input: &[f32],
    in_w: usize,
    in_h: usize,
    out: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    debug_assert_eq!(input.len(), in_w * in_h);
    debug_assert_eq!(out.len(), out_w * out_h);
    let (xsize, ysize) = (in_w as i64, in_h as i64);
    for y in 0..out_h as i64 {
        for x in 0..out_w as i64 {
            let kernel = upsample2_kernel(x, y);
            let (x2, y2) = (x / 2, y / 2);
            let mut sum = 0.0f32;
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for ky in 0..UPSAMPLE2_KSIZE {
                let yi = clamp_idx(y2 - UPSAMPLE2_KSIZE / 2 + ky, ysize);
                let row = yi * in_w;
                for kx in 0..UPSAMPLE2_KSIZE {
                    let xi = clamp_idx(x2 - UPSAMPLE2_KSIZE / 2 + kx, xsize);
                    let v = input[row + xi];
                    min = min.min(v);
                    max = max.max(v);
                    sum += v * kernel[(ky * UPSAMPLE2_KSIZE + kx) as usize];
                }
            }
            out[(y as usize) * out_w + x as usize] = sum.clamp(min, max);
        }
    }
}

/// Derivative of the 2× upsampler with respect to input pixel `(x2, y2)`
/// at output pixel `(x, y)`, ignoring the clamp (libjxl `UpsamplerDeriv`).
#[inline]
fn upsample2_deriv(x2: i64, y2: i64, x: i64, y: i64) -> f32 {
    let kernel = upsample2_kernel(x, y);
    let kx = x2 - x / 2 + UPSAMPLE2_KSIZE / 2;
    let ky = y2 - y / 2 + UPSAMPLE2_KSIZE / 2;
    if !(0..UPSAMPLE2_KSIZE).contains(&kx) || !(0..UPSAMPLE2_KSIZE).contains(&ky) {
        return 0.0;
    }
    kernel[(ky * UPSAMPLE2_KSIZE + kx) as usize]
}

/// Adjoint of the 2× upsampler: accumulates each full-res pixel back into
/// the half-res grid weighted by the upsampler derivative (libjxl
/// `AntiUpsample`). `input` is full-res (`in_w × in_h`), `out` is
/// half-res (`out_w × out_h`). Accumulation matches libjxl's mixed
/// float/double arithmetic (`double deriv` × float input into a float
/// accumulator).
fn anti_upsample2(
    input: &[f32],
    in_w: usize,
    in_h: usize,
    out: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    debug_assert_eq!(input.len(), in_w * in_h);
    debug_assert_eq!(out.len(), out_w * out_h);
    let (xsize, ysize) = (in_w as i64, in_h as i64);
    let k0 = UPSAMPLE2_KSIZE - 1;
    let k1 = UPSAMPLE2_KSIZE;
    for y2 in 0..out_h as i64 {
        for x2 in 0..out_w as i64 {
            let x0 = (x2 * 2 - k0).max(0);
            let x1 = (x2 * 2 + k1 + 1).min(xsize);
            let y0 = (y2 * 2 - k0).max(0);
            let y1 = (y2 * 2 + k1 + 1).min(ysize);
            let mut sum = 0.0f32;
            for y in y0..y1 {
                let row = (y as usize) * in_w;
                for x in x0..x1 {
                    let deriv = f64::from(upsample2_deriv(x2, y2, x, y));
                    sum = ((f64::from(sum)) + deriv * f64::from(input[row + x as usize])) as f32;
                }
            }
            out[(y2 as usize) * out_w + x2 as usize] = sum;
        }
    }
}

/// Ringing clamp on the iteratively-refined result (libjxl
/// `ReduceRinging`): bound each output pixel to the 3×3 neighborhood
/// min/max of the `initial` (sharper) result, widened by
/// `mask × 2` (libjxl `mask_multiplier = 2`).
fn reduce_ringing_2x(initial: &[f32], mask: &[f32], down: &mut [f32], w: usize, h: usize) {
    debug_assert_eq!(initial.len(), w * h);
    debug_assert_eq!(mask.len(), w * h);
    debug_assert_eq!(down.len(), w * h);
    const MASK_MULTIPLIER: f32 = 2.0;
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let idx = (y as usize) * w + x as usize;
            let mut min = initial[idx];
            let mut max = initial[idx];
            for yi in -1..2i64 {
                for xi in -1..2i64 {
                    let (x2, y2) = (x + xi, y + yi);
                    if x2 < 0 || y2 < 0 || x2 >= w as i64 || y2 >= h as i64 {
                        continue;
                    }
                    let v = initial[(y2 as usize) * w + x2 as usize];
                    min = min.min(v);
                    max = max.max(v);
                }
            }
            let a = mask[idx] * MASK_MULTIPLIER;
            down[idx] = down[idx].clamp(min - a, max + a);
        }
    }
}

/// 2×2 box downsample of one plane (libjxl `DownsampleImage(_, 2)`):
/// unweighted mean over the up-to-2×2 footprint, edge-clipped.
fn box_downsample_2x_plane(plane: &[f32], w: usize, h: usize, out: &mut [f32], out_w: usize) {
    for oy in 0..h.div_ceil(2) {
        let y0 = oy * 2;
        let y1 = (y0 + 2).min(h);
        for ox in 0..out_w {
            let x0 = ox * 2;
            let x1 = (x0 + 2).min(w);
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += plane[y * w + x];
                    count += 1;
                }
            }
            out[oy * out_w + ox] = sum / count as f32;
        }
    }
}

/// Iterative 2× downsample of one plane (libjxl
/// `DownsampleImage2_Iterative(const ImageF&, ImageF*)`,
/// `enc_heuristics.cc:658-739`): start from the sharper-kernel result,
/// then run 3 rounds of `down += AntiUpsample(orig − Upsample(down)) ÷
/// AntiUpsample(1)` so the decoder's default upsampler reconstructs the
/// original with minimal error, and finally clamp ringing against the
/// box-downsampled neighborhood mask.
fn iterative_downsample_2x_plane(
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

    // Box-downsample → ringing mask (same CreateMask as the sharper
    // path, but computed on the box result per libjxl).
    let mut box_plane = alloc::vec![0.0_f32; out_w * out_h];
    box_downsample_2x_plane(plane, width, height, &mut box_plane, out_w);
    let mut mask = alloc::vec![0.0_f32; out_w * out_h];
    create_ringing_mask(&box_plane, out_w, out_h, &mut mask);

    // Initial result: the sharper 12×12 kernel.
    let mut initial = alloc::vec![0.0_f32; out_w * out_h];
    sharper_downsample_2x_plane(plane, width, height, &mut initial, out_w, out_h);
    let mut down = initial.clone();

    // weights ≡ 1 full-res (libjxl leaves the anti-ringing weights field
    // at 1 — see the TODO in enc_heuristics.cc:704-709); its adjoint
    // normalizes the correction, differing from a constant only at the
    // image borders. `corr ×= weights` is skipped (× 1.0 is exact).
    let weights = alloc::vec![1.0_f32; width * height];
    let mut weights2 = alloc::vec![0.0_f32; out_w * out_h];
    anti_upsample2(&weights, width, height, &mut weights2, out_w, out_h);

    let mut up = alloc::vec![0.0_f32; width * height];
    let mut corr = alloc::vec![0.0_f32; width * height];
    let mut corr2 = alloc::vec![0.0_f32; out_w * out_h];
    const NUM_IT: usize = 3;
    for _ in 0..NUM_IT {
        upsample2_plane(&down, out_w, out_h, &mut up, width, height);
        for (c, (o, u)) in corr.iter_mut().zip(plane.iter().zip(up.iter())) {
            *c = o - u;
        }
        anti_upsample2(&corr, width, height, &mut corr2, out_w, out_h);
        for (d, (c2, w2)) in down.iter_mut().zip(corr2.iter().zip(weights2.iter())) {
            *d += c2 / w2;
        }
    }

    reduce_ringing_2x(&initial, &mask, &mut down, out_w, out_h);
    out.copy_from_slice(&down);
}

/// Iterative 2× downsample (libjxl `DownsampleImage2_Iterative`, used at
/// `speed_tier <= kGlacier` — our effort ≥ 10 — when `resampling == 2`).
/// Takes and returns **interleaved linear RGB** f32 like
/// [`sharper_downsample_2x_rgb`], but internally optimizes in the **XYB
/// (opsin) domain**: the decoder's 2× upsampling stage runs on the XYB
/// planes BEFORE the inverse color transform (libjxl `dec_cache.cc` adds
/// `GetUpsamplingStage` ahead of `GetXYBStage`; libjxl's encoder
/// correspondingly downsamples the opsin image — `enc_frame.cc:742`
/// `DownsampleColorChannels(..., Image3F* opsin)`). Optimizing the
/// adjoint rounds in linear RGB would target the wrong roundtrip and
/// measurably WORSENS quality (butteraugli 30.8 vs 19.7 on the first
/// CID22-512 differential cell). Conversion uses the unit-intensity
/// opsin pair (`linear_rgb_to_xyb_batch` / `xyb_to_linear_rgb_batch`)
/// — exact for the SDR default (`intensity_target = 255` ⇒
/// `intensity_mul = 1.0`); HDR intensity targets optimize in a
/// slightly-rescaled domain (bounded, encoder-side only).
///
/// Output dims are `(width.div_ceil(2), height.div_ceil(2))`. Compared
/// with the sharper kernel alone this minimizes the decoder-side
/// reconstruction error `‖orig − Upsample2×(down)‖` in the upsampler's
/// actual domain (3 adjoint-gradient rounds), at ~4× the sharper path's
/// arithmetic (libjxl documents ~80 % whole-encode slowdown,
/// `enc_frame.cc:753-755`).
pub fn iterative_downsample_2x_rgb(
    rgb_interleaved: &[f32],
    width: usize,
    height: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<(Vec<f32>, u32, u32)> {
    debug_assert_eq!(rgb_interleaved.len(), width * height * 3);
    let out_w = width.div_ceil(2);
    let out_h = height.div_ceil(2);
    let n = width * height;
    let n_out = out_w * out_h;

    // Deinterleave linear RGB into planes, then forward-opsin them.
    let mut plane_r = alloc::vec![0.0_f32; n];
    let mut plane_g = alloc::vec![0.0_f32; n];
    let mut plane_b = alloc::vec![0.0_f32; n];
    for i in 0..n {
        plane_r[i] = rgb_interleaved[i * 3];
        plane_g[i] = rgb_interleaved[i * 3 + 1];
        plane_b[i] = rgb_interleaved[i * 3 + 2];
    }
    let mut xyb_x = alloc::vec![0.0_f32; n];
    let mut xyb_y = alloc::vec![0.0_f32; n];
    let mut xyb_b = alloc::vec![0.0_f32; n];
    jxl_simd::linear_rgb_to_xyb_batch(
        &plane_r, &plane_g, &plane_b, &mut xyb_x, &mut xyb_y, &mut xyb_b,
    );
    drop(plane_r);
    drop(plane_g);
    drop(plane_b);

    // Iterative refinement per opsin plane (the decoder-upsampler's
    // actual input domain).
    let mut down_x = alloc::vec![0.0_f32; n_out];
    let mut down_y = alloc::vec![0.0_f32; n_out];
    let mut down_b = alloc::vec![0.0_f32; n_out];
    iterative_downsample_2x_plane(&xyb_x, width, height, &mut down_x, out_w, out_h);
    iterative_downsample_2x_plane(&xyb_y, width, height, &mut down_y, out_w, out_h);
    iterative_downsample_2x_plane(&xyb_b, width, height, &mut down_b, out_w, out_h);

    // Back to interleaved linear RGB for the (unchanged) encode pipeline,
    // which re-runs its own forward opsin on the half-res image.
    let mut out = crate::budget::vec_with_capacity_fallible(
        budget.is_some_and(|b| b.is_fallible()),
        n_out * 3,
    )?;
    out.resize(n_out * 3, 0.0);
    jxl_simd::xyb_to_linear_rgb_batch(&down_x, &down_y, &down_b, &mut out, n_out);
    Ok((out, out_w as u32, out_h as u32))
}

/// Box-filter downsample a single-channel u8 buffer (alpha) by an
/// integer factor. Same semantics as [`box_downsample_rgb`] but for
/// 1 byte/pixel inputs. Output values are rounded.
pub fn box_downsample_alpha_u8(
    alpha: &[u8],
    width: usize,
    height: usize,
    factor: u32,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<(Vec<u8>, u32, u32)> {
    debug_assert!(matches!(factor, 1 | 2 | 4 | 8), "factor must be 1/2/4/8");
    debug_assert_eq!(alpha.len(), width * height);
    if factor == 1 {
        return Ok((alpha.to_vec(), width as u32, height as u32));
    }
    let f = factor as usize;
    let out_w = width.div_ceil(f);
    let out_h = height.div_ceil(f);
    // Dimension-driven output buffer — honor the runtime fallible-alloc policy;
    // byte-identical when infallible.
    let mut out = crate::budget::vec_with_capacity_fallible(
        budget.is_some_and(|b| b.is_fallible()),
        out_w * out_h,
    )?;
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
    Ok((out, out_w as u32, out_h as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_1_is_clone() {
        let rgb = vec![0.5_f32, 0.25, 0.75, 0.1, 0.2, 0.3]; // 2 px
        let (out, w, h) = box_downsample_rgb(&rgb, 2, 1, 1, None).unwrap();
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
        let (out, w, h) = box_downsample_rgb(&rgb, 4, 4, 2, None).unwrap();
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(out.len(), 2 * 2 * 3);
        for chunk in out.as_chunks::<3>().0 {
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
        let (out, w, h) = box_downsample_rgb(&rgb, 2, 2, 2, None).unwrap();
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
        let (out, w, h) = box_downsample_rgb(&rgb, 5, 3, 4, None).unwrap();
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
        let (out, w, h) = box_downsample_rgb(&rgb, 8, 8, 8, None).unwrap();
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        assert!((out[0] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_factor_2_averages() {
        // 2×2 alpha: [255, 0, 0, 255]. Factor 2 → 1×1 with mean ≈ 128.
        let alpha = vec![255u8, 0, 0, 255];
        let (out, w, h) = box_downsample_alpha_u8(&alpha, 2, 2, 2, None).unwrap();
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        // (255 + 0 + 0 + 255 + 2) / 4 = 512 / 4 = 128.
        assert_eq!(out[0], 128);
    }

    #[test]
    fn test_alpha_factor_1_is_clone() {
        let alpha = vec![1u8, 2, 3, 4, 5];
        let (out, w, h) = box_downsample_alpha_u8(&alpha, 5, 1, 1, None).unwrap();
        assert_eq!(out, alpha);
        assert_eq!(w, 5);
        assert_eq!(h, 1);
    }

    #[test]
    fn test_sharper_2x_dimensions() {
        // 64×64 → 32×32; 65×64 → 33×32 (div_ceil).
        let rgb = vec![0.5_f32; 64 * 64 * 3];
        let (out, w, h) = sharper_downsample_2x_rgb(&rgb, 64, 64, None).unwrap();
        assert_eq!(w, 32);
        assert_eq!(h, 32);
        assert_eq!(out.len(), 32 * 32 * 3);
        let rgb = vec![0.5_f32; 65 * 64 * 3];
        let (_, w, h) = sharper_downsample_2x_rgb(&rgb, 65, 64, None).unwrap();
        assert_eq!(w, 33);
        assert_eq!(h, 32);
    }

    #[test]
    fn test_sharper_2x_uniform_input_is_uniform() {
        // Uniform input → kernel sum × value. The kernel sums to ~1.0
        // by construction; confirm output is approximately the input.
        let rgb = vec![0.5_f32; 32 * 32 * 3];
        let (out, w, h) = sharper_downsample_2x_rgb(&rgb, 32, 32, None).unwrap();
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
        let (out, _, _) = sharper_downsample_2x_rgb(&rgb, 16, 16, None).unwrap();
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

    // ── Iterative 2× (DownsampleImage2_Iterative port) ──────────────────

    #[test]
    fn test_upsample2_kernels_sum_to_1_and_mirror() {
        // Each decoder-upsampler phase kernel preserves DC (sums ≈ 1),
        // and the four phases are mirror images of each other — both
        // properties hold in the libjxl table this was transcribed from.
        for (name, k) in [
            ("00", &UPSAMPLE2_KERNEL_00),
            ("01", &UPSAMPLE2_KERNEL_01),
            ("10", &UPSAMPLE2_KERNEL_10),
            ("11", &UPSAMPLE2_KERNEL_11),
        ] {
            let s: f32 = k.iter().sum();
            assert!(
                (s - 1.0).abs() < 0.01,
                "kernel{name} sum {s} should be ~1.0"
            );
        }
        for ky in 0..5 {
            for kx in 0..5 {
                // 01 = vertical mirror of 00; 10 = horizontal mirror;
                // 11 = both.
                assert_eq!(
                    UPSAMPLE2_KERNEL_00[ky * 5 + kx],
                    UPSAMPLE2_KERNEL_01[(4 - ky) * 5 + kx]
                );
                assert_eq!(
                    UPSAMPLE2_KERNEL_00[ky * 5 + kx],
                    UPSAMPLE2_KERNEL_10[ky * 5 + (4 - kx)]
                );
                assert_eq!(
                    UPSAMPLE2_KERNEL_00[ky * 5 + kx],
                    UPSAMPLE2_KERNEL_11[(4 - ky) * 5 + (4 - kx)]
                );
            }
        }
    }

    #[test]
    fn test_iterative_2x_dimensions_and_uniform() {
        // Uniform input stays uniform (kernels preserve DC; the
        // correction rounds see zero residual), even/odd dims both work.
        for (w, h) in [(64usize, 64usize), (65, 64), (33, 47)] {
            let rgb = vec![0.5_f32; w * h * 3];
            let (out, ow, oh) = iterative_downsample_2x_rgb(&rgb, w, h, None).unwrap();
            assert_eq!(ow as usize, w.div_ceil(2));
            assert_eq!(oh as usize, h.div_ceil(2));
            assert_eq!(out.len(), (ow * oh * 3) as usize);
            for &v in &out {
                assert!((v - 0.5).abs() < 0.01, "uniform 0.5 → {v} at {w}x{h}");
            }
        }
    }

    /// Sum of squared reconstruction error after running the DECODER's
    /// default 2× upsampler on a half-res plane.
    fn roundtrip_l2(orig: &[f32], down: &[f32], w: usize, h: usize) -> f64 {
        let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
        let mut up = vec![0.0_f32; w * h];
        upsample2_plane(down, ow, oh, &mut up, w, h);
        orig.iter()
            .zip(up.iter())
            .map(|(o, u)| {
                let d = f64::from(o - u);
                d * d
            })
            .sum()
    }

    #[test]
    fn test_iterative_2x_beats_sharper_on_decoder_roundtrip() {
        // The whole point of the iterative refinement (libjxl
        // `DownsampleImage2_Iterative`) is to minimize the error the
        // DECODER's default upsampler reconstructs with. On structured
        // content the refined plane must beat the sharper-kernel plane
        // on that metric. Content: a deterministic mix of gradient +
        // edges + texture (structure is what the adjoint iterations
        // exploit; a uniform plane would tie).
        let (w, h) = (96usize, 80usize);
        let mut plane = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let gradient = x as f32 / w as f32;
                let edge = if (x / 16 + y / 12) % 2 == 0 { 0.6 } else { 0.0 };
                let texture = 0.08 * (((x * 7 + y * 13) % 11) as f32 / 11.0);
                plane[y * w + x] = (gradient * 0.3 + edge + texture).min(1.0);
            }
        }
        let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
        let mut sharper = vec![0.0_f32; ow * oh];
        sharper_downsample_2x_plane(&plane, w, h, &mut sharper, ow, oh);
        let mut iterative = vec![0.0_f32; ow * oh];
        iterative_downsample_2x_plane(&plane, w, h, &mut iterative, ow, oh);

        let err_sharper = roundtrip_l2(&plane, &sharper, w, h);
        let err_iterative = roundtrip_l2(&plane, &iterative, w, h);
        assert!(
            err_iterative < err_sharper,
            "iterative must reduce decoder-upsampler roundtrip error: \
             iterative {err_iterative:.6} vs sharper {err_sharper:.6}"
        );
    }

    #[test]
    fn test_anti_upsample2_weights_positive_and_interior_4() {
        // AntiUpsample(1) is the per-input-pixel kernel mass: ≈ 4.0 in
        // the interior (4 output phases, each kernel sums ≈ 1) and
        // strictly positive everywhere (no division-by-zero in the
        // correction normalizer).
        let (w, h) = (20usize, 14usize);
        let ones = vec![1.0_f32; w * h];
        let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
        let mut w2 = vec![0.0_f32; ow * oh];
        anti_upsample2(&ones, w, h, &mut w2, ow, oh);
        for (i, &v) in w2.iter().enumerate() {
            assert!(v > 0.5, "weights2[{i}] = {v} must be positive");
        }
        // A fully-interior cell.
        let center = w2[(oh / 2) * ow + ow / 2];
        assert!(
            (center - 4.0).abs() < 0.05,
            "interior anti-upsampled weight ≈ 4.0, got {center}"
        );
    }
}
