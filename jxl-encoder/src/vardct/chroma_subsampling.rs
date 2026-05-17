// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chroma-subsampled YCbCr pipeline primitives for VarDCT encoding.
//!
//! Follow-on to chunk 1 (`8eb7ea4f`, PR #47), which shipped the
//! [`crate::api::ChromaSubsampling`] enum + builder + `InvalidConfig`
//! rejection in the lossy path. This module lands the foundational
//! building blocks the eventual end-to-end Sub420 / Sub422 / Sub440
//! pipeline needs:
//!
//! 1. Forward `RGB → YCbCr` colour-transform — full-range BT.601 / JFIF
//!    Clause 7, matching the libjxl decoder's `kYCbCrStage`
//!    (`render_pipeline/stage_ycbcr.cc:24-39`). Y is centred on 0 (in
//!    `[-128/255, 127/255]`), Cb/Cr also on 0.
//! 2. Box-filter chroma downsample for 4:2:0 / 4:2:2 / 4:4:0.
//! 3. Builder for a non-JPEG YCbCr [`crate::headers::frame_header::FrameHeader`]
//!    that mirrors the JPEG-transcode path
//!    (`jxl-encoder/src/jpeg/encode.rs:636-652`).
//!
//! Wiring these helpers into [`crate::vardct::encoder::VarDctEncoder::encode_inner`]
//! is the next chunk. Until then [`crate::api::ChromaSubsampling::Sub420`]
//! / `Sub422` / `Sub440` continue to return `EncodeError::InvalidConfig`
//! at the lossy entry points — see `api.rs:5222` and `api.rs:6521`.
//!
//! # libjxl references
//!
//! - Forward YCbCr matrix: inverse of `render_pipeline/stage_ycbcr.cc:24-39`.
//! - Frame-header layout: `frame_header.cc:244-254`
//!   (`do_ycbcr` bit + per-channel `jpeg_upsampling`).
//! - Chroma-subsampling-only-with-kYCbCr rule: `enc_frame.cc:381-387`.

// chunk-2 helpers are not called from the encoder pipeline yet — chunk 3
// wires them into `encoder::encode_inner`. The integration is
// blocked only on the colour-pipeline branch in `encode_inner`, not on
// these helpers' correctness (each is unit-tested below). Allow the
// dead_code lints in the interim so the rest of the crate stays warning-
// free.
#![allow(dead_code)]

use crate::headers::frame_header::{Encoding, FrameHeader};

use super::common::{JPEG_UPSAMPLING_H_SHIFT, JPEG_UPSAMPLING_V_SHIFT, div_ceil};

/// Forward `RGB → YCbCr` colour transform, full-range BT.601 / JFIF Clause 7.
///
/// Inputs are linear sRGB in `[0.0, 1.0]` (out-of-gamut values are
/// allowed — they map linearly). Outputs are centred-on-0 YCbCr:
///
/// - `Y  = 0.299 R + 0.587 G + 0.114 B − 128/255` (range `[-128/255, 127/255]`)
/// - `Cb = (B − Y′) / 1.772`             (range `[-0.5, 0.5]`)
/// - `Cr = (R − Y′) / 1.402`             (range `[-0.5, 0.5]`)
///
/// where `Y′ = Y + 128/255` is the un-centred luma.
///
/// This is the exact inverse of libjxl's `kYCbCrStage::ProcessRow`
/// (`render_pipeline/stage_ycbcr.cc:24-39`), so a
/// `rgb_to_ycbcr_pixel → ycbcr_to_rgb_pixel` round-trip is the identity
/// modulo floating-point precision (verified by
/// `test_ycbcr_roundtrip_identity`).
///
/// **Why centred Y**: libjxl's decoder adds `128/255` back during
/// reconstruction. The VarDCT entropy coder benefits when DC values
/// cluster around 0, so the encoder pre-subtracts the offset.
#[inline]
pub fn rgb_to_ycbcr_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // BT.601 luma coefficients.
    let y_unbiased = 0.299_f32 * r + 0.587_f32 * g + 0.114_f32 * b;
    let y = y_unbiased - (128.0_f32 / 255.0_f32);
    // Centred chroma. The libjxl decoder formula
    //     R = Y + 1.402 * Cr
    //     B = Y + 1.772 * Cb
    // inverts to
    //     Cr = (R − Y′) / 1.402
    //     Cb = (B − Y′) / 1.772
    let cb = (b - y_unbiased) / 1.772_f32;
    let cr = (r - y_unbiased) / 1.402_f32;
    (y, cb, cr)
}

/// Inverse of [`rgb_to_ycbcr_pixel`], matching libjxl's `kYCbCrStage`
/// (`render_pipeline/stage_ycbcr.cc:24-39`) exactly. Kept here so the
/// forward-then-inverse round-trip can be unit-tested without dragging
/// in a full decoder pipeline.
#[inline]
pub fn ycbcr_to_rgb_pixel(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    // Per stage_ycbcr.cc:
    //   const auto c128 = Set(df, 128.0f / 255);
    //   r = y + c128 + 1.402 * cr
    //   g = y + c128 - 0.114*1.772/0.587 * cb - 0.299*1.402/0.587 * cr
    //   b = y + c128 + 1.772 * cb
    let y_biased = y + (128.0_f32 / 255.0_f32);
    let r = y_biased + 1.402_f32 * cr;
    let g = y_biased
        + (-0.114_f32 * 1.772_f32 / 0.587_f32) * cb
        + (-0.299_f32 * 1.402_f32 / 0.587_f32) * cr;
    let b = y_biased + 1.772_f32 * cb;
    (r, g, b)
}

/// Convert a tightly-packed interleaved linear sRGB plane (`R, G, B,
/// R, G, B, …`, `width * height * 3` floats) into three separate planar
/// YCbCr buffers. Y is stored centred on 0; Cb/Cr likewise.
///
/// The output planes have length `width * height` each and use natural
/// row-major layout (`y * width + x`). Caller-side padding to block /
/// chroma-shift boundaries happens later in the pipeline (see
/// [`box_downsample_2x_horizontal`] / [`box_downsample_2x_vertical`] /
/// [`box_downsample_2x_both`]).
///
/// # Panics
///
/// In debug builds, panics if `interleaved_rgb.len() != width * height * 3`.
/// Release builds skip the check; the higher-level `encode_inner`
/// validates input dimensions before calling here.
pub fn rgb_to_ycbcr_planar(
    width: usize,
    height: usize,
    interleaved_rgb: &[f32],
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
) {
    let n = width * height;
    debug_assert_eq!(interleaved_rgb.len(), n * 3);
    debug_assert_eq!(y_plane.len(), n);
    debug_assert_eq!(cb_plane.len(), n);
    debug_assert_eq!(cr_plane.len(), n);
    for i in 0..n {
        let r = interleaved_rgb[i * 3];
        let g = interleaved_rgb[i * 3 + 1];
        let b = interleaved_rgb[i * 3 + 2];
        let (y, cb, cr) = rgb_to_ycbcr_pixel(r, g, b);
        y_plane[i] = y;
        cb_plane[i] = cb;
        cr_plane[i] = cr;
    }
}

/// 2× box-filter downsample on both axes. Each output pixel is the
/// arithmetic mean of a 2×2 input neighbourhood. Used for 4:2:0 chroma.
///
/// Output dimensions are `div_ceil(width, 2) × div_ceil(height, 2)`.
/// Odd input edges are handled by edge-replication: a 2×2 box at the
/// right / bottom edge averages whatever in-bounds samples it sees,
/// dividing by the actual count (1, 2, or 4). This matches the libjxl
/// JPEG-transcode subsampling rule of padding the original to even
/// dimensions, then averaging.
///
/// # Panics
///
/// In debug builds, panics if buffer lengths disagree with the
/// width / height arguments.
pub fn box_downsample_2x_both(
    width: usize,
    height: usize,
    src: &[f32],
    dst_width: usize,
    dst_height: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst_width, div_ceil(width, 2));
    debug_assert_eq!(dst_height, div_ceil(height, 2));
    debug_assert_eq!(dst.len(), dst_width * dst_height);
    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let sx0 = dx * 2;
            let sy0 = dy * 2;
            let sx1 = (sx0 + 1).min(width - 1);
            let sy1 = (sy0 + 1).min(height - 1);
            // Count the number of *distinct* in-bounds samples. When
            // we reach the right / bottom edge of an odd-sized plane,
            // sx1 == sx0 (or sy1 == sy0); use straight 1×1 / 1×2 / 2×1
            // / 2×2 averages so the edge value isn't biased by
            // duplicating itself.
            let xs: [usize; 2] = [sx0, sx1];
            let ys: [usize; 2] = [sy0, sy1];
            let unique_x = if sx0 == sx1 { 1 } else { 2 };
            let unique_y = if sy0 == sy1 { 1 } else { 2 };
            let mut sum = 0.0_f32;
            for j in 0..unique_y {
                for i in 0..unique_x {
                    sum += src[ys[j] * width + xs[i]];
                }
            }
            let count = (unique_x * unique_y) as f32;
            dst[dy * dst_width + dx] = sum / count;
        }
    }
}

/// 2× box-filter downsample on the horizontal axis only (used for
/// 4:2:2). Each output pixel is the mean of two adjacent input
/// samples; odd right-edge takes the single in-bounds sample.
pub fn box_downsample_2x_horizontal(
    width: usize,
    height: usize,
    src: &[f32],
    dst_width: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst_width, div_ceil(width, 2));
    debug_assert_eq!(dst.len(), dst_width * height);
    for y in 0..height {
        for dx in 0..dst_width {
            let sx0 = dx * 2;
            let sx1 = (sx0 + 1).min(width - 1);
            if sx0 == sx1 {
                dst[y * dst_width + dx] = src[y * width + sx0];
            } else {
                let a = src[y * width + sx0];
                let b = src[y * width + sx1];
                dst[y * dst_width + dx] = 0.5_f32 * (a + b);
            }
        }
    }
}

/// 2× box-filter downsample on the vertical axis only (used for
/// 4:4:0). Each output pixel is the mean of two adjacent rows; odd
/// bottom-edge takes the single in-bounds row.
pub fn box_downsample_2x_vertical(
    width: usize,
    height: usize,
    src: &[f32],
    dst_height: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst_height, div_ceil(height, 2));
    debug_assert_eq!(dst.len(), width * dst_height);
    for dy in 0..dst_height {
        let sy0 = dy * 2;
        let sy1 = (sy0 + 1).min(height - 1);
        for x in 0..width {
            if sy0 == sy1 {
                dst[dy * width + x] = src[sy0 * width + x];
            } else {
                let a = src[sy0 * width + x];
                let b = src[sy1 * width + x];
                dst[dy * width + x] = 0.5_f32 * (a + b);
            }
        }
    }
}

/// Compute the libjxl-style `jpeg_upsampling` triple for a given
/// chroma-subsampling mode, in JXL channel order (`[Cb, Y, Cr]`).
///
/// **Subtle**: the value at each slot is **not** the per-channel
/// downsampling shift. It is libjxl's `channel_mode_[c]` — an index
/// into the `kHShift` / `kVShift` lookup tables
/// (`frame_header.cc:30-31`):
///
/// | Mode index | `kHShift` | `kVShift` | Encoded *sampling factor* |
/// |------------|-----------|-----------|---------------------------|
/// | 0          | 0         | 0         | 1×1 (subsampled chroma)   |
/// | 1          | 1         | 1         | 2×2 (Y in 4:2:0)          |
/// | 2          | 1         | 0         | 2×1 (Y in 4:2:2)          |
/// | 3          | 0         | 1         | 1×2 (Y in 4:4:0)          |
///
/// The actual downsampling shift of a channel is then
/// `HShift(c) = max(kHShift[…]) − kHShift[channel_mode_[c]]`
/// (`frame_header.h:84`). For 4:2:0:
///
/// - `jpeg_upsampling = [0, 1, 0]`  (Cb=mode0, Y=mode1, Cr=mode0)
/// - `max_kHShift = 1`, `max_kVShift = 1`
/// - `HShift(Cb) = 1 − 0 = 1` (Cb is half-resolution horizontally) ✓
/// - `HShift(Y)  = 1 − 1 = 0` (Y is full-resolution)                 ✓
///
/// This matches what `compute_jpeg_upsampling` in
/// `jxl-encoder/src/jpeg/encode.rs:616-633` computes for a 4:2:0 JPEG
/// (where the JPEG sampling factors are H=2/V=2 for Y and H=1/V=1 for
/// chroma).
///
/// **Not the same as [`crate::api::ChromaSubsampling::h_shifts`]**:
/// the public API getter returns the *post-max actual shift*
/// (`[1, 0, 1]` for Sub420) — useful for sizing per-channel buffers.
/// `jpeg_upsampling_for` returns the *raw mode index* (`[0, 1, 0]` for
/// Sub420) — useful for stamping `FrameHeader::jpeg_upsampling`. They
/// are related but not interchangeable; chunk-3 wiring uses both.
pub fn jpeg_upsampling_for(mode: crate::api::ChromaSubsampling) -> [u8; 3] {
    use crate::api::ChromaSubsampling;
    match mode {
        // Every channel at full resolution: every mode index 0 →
        // kHShift = kVShift = 0 everywhere, max = 0, all HShift = 0.
        ChromaSubsampling::Full444 => [0, 0, 0],
        // Y at mode 1 (kHShift=1, kVShift=1 → "biggest"); chroma at mode 0.
        // → HShift(Y) = 1-1 = 0, HShift(Cb) = HShift(Cr) = 1-0 = 1.
        ChromaSubsampling::Sub420 => [0, 1, 0],
        // Y at mode 2 (kHShift=1, kVShift=0); chroma at mode 0.
        // → HShift(Y) = 0, HShift(Cb/Cr) = 1; VShift(*) = 0.
        ChromaSubsampling::Sub422 => [0, 2, 0],
        // Y at mode 3 (kHShift=0, kVShift=1); chroma at mode 0.
        // → HShift(*) = 0; VShift(Y) = 0, VShift(Cb/Cr) = 1.
        ChromaSubsampling::Sub440 => [0, 3, 0],
    }
}

/// Build a [`FrameHeader`] for a non-JPEG VarDCT encode that uses
/// `ColorTransform::kYCbCr` and the given [`crate::api::ChromaSubsampling`]
/// mode.
///
/// Mirrors `build_jpeg_frame_header` in
/// `jxl-encoder/src/jpeg/encode.rs:636-652` but for the regular
/// (non-pre-quantized) encode path:
///
/// - `xyb_encoded = false` (decoder uses kYCbCr for inverse colour-transform)
/// - `do_ycbcr = true`
/// - `jpeg_upsampling = [Cb_mode, 0, Cr_mode]` from [`jpeg_upsampling_for`]
/// - `encoding = VarDct`
/// - Other fields are left at [`FrameHeader::default`]; downstream
///   pipeline code (gaborish, EPF, x/b_qm_scale) overwrites the
///   per-encode-specific values before serialisation.
pub fn build_ycbcr_vardct_frame_header(mode: crate::api::ChromaSubsampling) -> FrameHeader {
    FrameHeader {
        encoding: Encoding::VarDct,
        xyb_encoded: false,
        do_ycbcr: true,
        jpeg_upsampling: jpeg_upsampling_for(mode),
        ..FrameHeader::default()
    }
}

/// Per-channel `(h_shift, v_shift)` triple for a given mode, in JXL
/// channel order (`[Cb, Y, Cr]`).
///
/// Useful for sizing per-channel block dims:
/// `xsize_blocks_c = div_ceil(width, 8 << h_shift_c)`.
pub fn channel_shifts_for(mode: crate::api::ChromaSubsampling) -> [(usize, usize); 3] {
    let up = jpeg_upsampling_for(mode);
    // Mirror the libjxl convention used in `jpeg/encode.rs:95-108`:
    // shift = max_raw - per-channel raw. For these 4 chroma modes Y is
    // always full-resolution (mode 0 → raw shift 0), so max_raw_hs ==
    // raw h_shift of the chroma mode and the Cb/Cr shift is exactly
    // that value.
    let max_raw_hs = up
        .iter()
        .map(|&u| JPEG_UPSAMPLING_H_SHIFT[u as usize])
        .max()
        .unwrap_or(0);
    let max_raw_vs = up
        .iter()
        .map(|&u| JPEG_UPSAMPLING_V_SHIFT[u as usize])
        .max()
        .unwrap_or(0);
    let mut out = [(0usize, 0usize); 3];
    for c in 0..3 {
        let raw_hs = JPEG_UPSAMPLING_H_SHIFT[up[c] as usize];
        let raw_vs = JPEG_UPSAMPLING_V_SHIFT[up[c] as usize];
        out[c] = (max_raw_hs - raw_hs, max_raw_vs - raw_vs);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ChromaSubsampling;

    /// Layer 1: per-pixel `rgb → ycbcr → rgb` is identity to within
    /// f32 precision. This is the foundational correctness check — if
    /// the matrix is wrong, every chunk-3 wiring effort is doomed.
    #[test]
    fn test_ycbcr_roundtrip_identity() {
        // Sweep a representative set of in-gamut and out-of-gamut
        // values, including the corners of the unit cube and a couple
        // of negative / >1 values that decoders see from XYB-style
        // pipelines.
        let probes: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.75, 0.10],
            [0.9, 0.1, 0.4],
            // mild out-of-gamut (XYB→linear can produce these)
            [-0.05, 0.5, 1.05],
            [1.2, -0.1, 0.3],
        ];
        for &[r, g, b] in probes {
            let (y, cb, cr) = rgb_to_ycbcr_pixel(r, g, b);
            let (r2, g2, b2) = ycbcr_to_rgb_pixel(y, cb, cr);
            let tol = 1.0e-6_f32;
            assert!(
                (r - r2).abs() < tol,
                "R mismatch: rgb=({r},{g},{b}) → round-trip {r2}, diff {}",
                (r - r2).abs()
            );
            assert!(
                (g - g2).abs() < tol,
                "G mismatch: rgb=({r},{g},{b}) → round-trip {g2}, diff {}",
                (g - g2).abs()
            );
            assert!(
                (b - b2).abs() < tol,
                "B mismatch: rgb=({r},{g},{b}) → round-trip {b2}, diff {}",
                (b - b2).abs()
            );
        }
    }

    /// Y for pure colours should match the BT.601 luma weights minus
    /// the 128/255 bias.
    #[test]
    fn test_ycbcr_luma_weights_bt601() {
        let bias = 128.0_f32 / 255.0_f32;
        // Pure red → Y = 0.299 − bias
        let (y, _, _) = rgb_to_ycbcr_pixel(1.0, 0.0, 0.0);
        assert!((y - (0.299_f32 - bias)).abs() < 1.0e-6);
        // Pure green → Y = 0.587 − bias
        let (y, _, _) = rgb_to_ycbcr_pixel(0.0, 1.0, 0.0);
        assert!((y - (0.587_f32 - bias)).abs() < 1.0e-6);
        // Pure blue → Y = 0.114 − bias
        let (y, _, _) = rgb_to_ycbcr_pixel(0.0, 0.0, 1.0);
        assert!((y - (0.114_f32 - bias)).abs() < 1.0e-6);
        // Gray → Cb = Cr = 0
        let (_, cb, cr) = rgb_to_ycbcr_pixel(0.5, 0.5, 0.5);
        assert!(cb.abs() < 1.0e-6);
        assert!(cr.abs() < 1.0e-6);
    }

    /// Planar conversion preserves the per-pixel result (and writes
    /// every output cell).
    #[test]
    fn test_rgb_to_ycbcr_planar_matches_per_pixel() {
        let width = 4;
        let height = 3;
        let mut interleaved = Vec::with_capacity(width * height * 3);
        for i in 0..(width * height) {
            // Generate a varied but deterministic pattern.
            let r = (i as f32) / 20.0;
            let g = 1.0 - (i as f32) / 30.0;
            let b = ((i * 7) % 11) as f32 / 11.0;
            interleaved.push(r);
            interleaved.push(g);
            interleaved.push(b);
        }
        let mut y = vec![0.0_f32; width * height];
        let mut cb = vec![0.0_f32; width * height];
        let mut cr = vec![0.0_f32; width * height];
        rgb_to_ycbcr_planar(width, height, &interleaved, &mut y, &mut cb, &mut cr);
        for i in 0..(width * height) {
            let r = interleaved[i * 3];
            let g = interleaved[i * 3 + 1];
            let b = interleaved[i * 3 + 2];
            let (ey, ecb, ecr) = rgb_to_ycbcr_pixel(r, g, b);
            assert!((y[i] - ey).abs() < 1.0e-7);
            assert!((cb[i] - ecb).abs() < 1.0e-7);
            assert!((cr[i] - ecr).abs() < 1.0e-7);
        }
    }

    /// Box-downsample 2×2 returns the mean of each 2×2 patch. Edge
    /// cases at odd widths / heights average only the in-bounds
    /// samples.
    #[test]
    fn test_box_downsample_2x_both_basic() {
        // 4×4 plane → 2×2 plane. Each output is the mean of a
        // disjoint 2×2 block.
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0_f32; 4];
        box_downsample_2x_both(4, 4, &src, 2, 2, &mut dst);
        // Block (0,0): {0,1,4,5} → 2.5
        // Block (1,0): {2,3,6,7} → 4.5
        // Block (0,1): {8,9,12,13} → 10.5
        // Block (1,1): {10,11,14,15} → 12.5
        assert_eq!(dst[0], 2.5);
        assert_eq!(dst[1], 4.5);
        assert_eq!(dst[2], 10.5);
        assert_eq!(dst[3], 12.5);
    }

    #[test]
    fn test_box_downsample_2x_both_odd_width() {
        // 3×2 plane → 2×1 plane. Right column has only one in-bounds
        // sample per row, so the output is a vertical-pair mean.
        let src = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut dst = vec![0.0_f32; 2];
        box_downsample_2x_both(3, 2, &src, 2, 1, &mut dst);
        // dst[0]: {10,20,40,50} / 4 = 30
        // dst[1]: {30,60} / 2 = 45 (right edge: single column averaged
        //                            with row below)
        assert_eq!(dst[0], 30.0);
        assert_eq!(dst[1], 45.0);
    }

    #[test]
    fn test_box_downsample_2x_both_odd_height() {
        // 2×3 plane → 1×2 plane. Bottom row has only one in-bounds row.
        let src = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut dst = vec![0.0_f32; 2];
        box_downsample_2x_both(2, 3, &src, 1, 2, &mut dst);
        // dst[0]: {10,20,30,40} / 4 = 25
        // dst[1]: {50,60} / 2 = 55 (bottom edge: single row averaged
        //                            across two columns)
        assert_eq!(dst[0], 25.0);
        assert_eq!(dst[1], 55.0);
    }

    #[test]
    fn test_box_downsample_2x_both_single_pixel() {
        // 1×1 plane → 1×1 plane (identity).
        let src = [42.0_f32];
        let mut dst = vec![0.0_f32; 1];
        box_downsample_2x_both(1, 1, &src, 1, 1, &mut dst);
        assert_eq!(dst[0], 42.0);
    }

    #[test]
    fn test_box_downsample_2x_horizontal_basic() {
        // 4×2 → 2×2: each output is the horizontal-pair mean.
        let src = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mut dst = vec![0.0_f32; 4];
        box_downsample_2x_horizontal(4, 2, &src, 2, &mut dst);
        assert_eq!(dst[0], 15.0);
        assert_eq!(dst[1], 35.0);
        assert_eq!(dst[2], 55.0);
        assert_eq!(dst[3], 75.0);
    }

    #[test]
    fn test_box_downsample_2x_vertical_basic() {
        // 2×4 → 2×2: each output is the vertical-pair mean.
        let src = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mut dst = vec![0.0_f32; 4];
        box_downsample_2x_vertical(2, 4, &src, 2, &mut dst);
        // Rows are: [10,20], [30,40], [50,60], [70,80]
        // dst row 0: ([10,20] + [30,40]) / 2 = [20,30]
        // dst row 1: ([50,60] + [70,80]) / 2 = [60,70]
        assert_eq!(dst[0], 20.0);
        assert_eq!(dst[1], 30.0);
        assert_eq!(dst[2], 60.0);
        assert_eq!(dst[3], 70.0);
    }

    /// `jpeg_upsampling_for` returns libjxl `channel_mode_[c]` indices,
    /// **not** the per-channel actual H/V shift. Y always gets the
    /// "biggest" mode (1/2/3 — the one with maximum kHShift/kVShift);
    /// chroma always gets mode 0. This matches what
    /// `compute_jpeg_upsampling` in `jpeg/encode.rs:616-633` writes for
    /// a same-subsampling JPEG.
    #[test]
    fn test_jpeg_upsampling_for_all_modes() {
        // [Cb, Y, Cr] layout, mode indices (not shifts).
        assert_eq!(jpeg_upsampling_for(ChromaSubsampling::Full444), [0, 0, 0]);
        assert_eq!(jpeg_upsampling_for(ChromaSubsampling::Sub420), [0, 1, 0]);
        assert_eq!(jpeg_upsampling_for(ChromaSubsampling::Sub422), [0, 2, 0]);
        assert_eq!(jpeg_upsampling_for(ChromaSubsampling::Sub440), [0, 3, 0]);
    }

    /// The libjxl `HShift(c) = maxhs - kHShift[mode_c]` /
    /// `VShift(c) = maxvs - kVShift[mode_c]` formulae applied to our
    /// `jpeg_upsampling_for` output recover the per-channel shift
    /// values that `ChromaSubsampling::h_shifts` / `v_shifts` advertise
    /// in the chunk-1 public API. This is the structural proof that
    /// the two getters return *consistent* views of the same physical
    /// subsampling.
    #[test]
    fn test_jpeg_upsampling_round_trips_to_h_v_shifts() {
        for &mode in &[
            ChromaSubsampling::Full444,
            ChromaSubsampling::Sub420,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub440,
        ] {
            let up = jpeg_upsampling_for(mode);
            let max_hs = up
                .iter()
                .map(|&u| JPEG_UPSAMPLING_H_SHIFT[u as usize])
                .max()
                .unwrap();
            let max_vs = up
                .iter()
                .map(|&u| JPEG_UPSAMPLING_V_SHIFT[u as usize])
                .max()
                .unwrap();
            let want_h = mode.h_shifts();
            let want_v = mode.v_shifts();
            for c in 0..3 {
                let got_h = max_hs - JPEG_UPSAMPLING_H_SHIFT[up[c] as usize];
                let got_v = max_vs - JPEG_UPSAMPLING_V_SHIFT[up[c] as usize];
                assert_eq!(
                    got_h as u8, want_h[c],
                    "h_shift mismatch for {mode:?} c={c}: jpeg_upsampling={up:?}"
                );
                assert_eq!(
                    got_v as u8, want_v[c],
                    "v_shift mismatch for {mode:?} c={c}: jpeg_upsampling={up:?}"
                );
            }
        }
    }

    /// `channel_shifts_for` agrees with the (h_shift, v_shift)
    /// accessors on [`ChromaSubsampling`] for the chroma channels.
    /// Layout is `[Cb, Y, Cr]` (libjxl order).
    #[test]
    fn test_channel_shifts_for_all_modes() {
        // Full444: every channel at full resolution.
        assert_eq!(
            channel_shifts_for(ChromaSubsampling::Full444),
            [(0, 0), (0, 0), (0, 0)]
        );
        // Sub420: Cb/Cr halved on both axes; Y still full-res.
        assert_eq!(
            channel_shifts_for(ChromaSubsampling::Sub420),
            [(1, 1), (0, 0), (1, 1)]
        );
        // Sub422: Cb/Cr halved horizontally only.
        assert_eq!(
            channel_shifts_for(ChromaSubsampling::Sub422),
            [(1, 0), (0, 0), (1, 0)]
        );
        // Sub440: Cb/Cr halved vertically only.
        assert_eq!(
            channel_shifts_for(ChromaSubsampling::Sub440),
            [(0, 1), (0, 0), (0, 1)]
        );
    }

    /// `build_ycbcr_vardct_frame_header` emits the correct
    /// `do_ycbcr` / `xyb_encoded` / `jpeg_upsampling` triple for each
    /// mode. The Y / Full444 case is the sanity check; Sub420 / Sub422
    /// / Sub440 cover the three subsampling modes that chunk 3 will
    /// wire into encode_inner.
    #[test]
    fn test_build_ycbcr_vardct_frame_header_all_modes() {
        for (mode, want_up) in [
            (ChromaSubsampling::Full444, [0, 0, 0]),
            (ChromaSubsampling::Sub420, [0, 1, 0]),
            (ChromaSubsampling::Sub422, [0, 2, 0]),
            (ChromaSubsampling::Sub440, [0, 3, 0]),
        ] {
            let fh = build_ycbcr_vardct_frame_header(mode);
            assert_eq!(fh.encoding, Encoding::VarDct, "mode {mode:?}");
            assert!(!fh.xyb_encoded, "mode {mode:?}: xyb_encoded must be false");
            assert!(fh.do_ycbcr, "mode {mode:?}: do_ycbcr must be true");
            assert_eq!(fh.jpeg_upsampling, want_up, "mode {mode:?}");
        }
    }

    /// Builder consistency: the frame-header `jpeg_upsampling` field
    /// is derived from `jpeg_upsampling_for`. Cb and Cr must share the
    /// same `channel_mode_` (libjxl invariant — `frame_header.h:122-144`
    /// `Is444/Is420/Is422/Is440` all assume `Cb == Cr` shifts).
    #[test]
    fn test_header_and_shifts_are_consistent() {
        for &mode in &[
            ChromaSubsampling::Full444,
            ChromaSubsampling::Sub420,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub440,
        ] {
            let fh = build_ycbcr_vardct_frame_header(mode);
            let up = jpeg_upsampling_for(mode);
            assert_eq!(fh.jpeg_upsampling, up);
            // Cb and Cr must share the same channel_mode_ value (the
            // libjxl `Is420/422/440` predicates require it).
            assert_eq!(
                up[0], up[2],
                "Cb and Cr channel_mode_ must match for {mode:?}"
            );
        }
    }

    /// Full444 (default) is the "no-op" mode: forward conversion to
    /// YCbCr, then 4:4:4-equivalent (= no) downsampling, then inverse
    /// conversion, recovers the original RGB pixel-for-pixel. This
    /// proves the helpers compose cleanly for the chunk-3 wiring.
    #[test]
    fn test_full444_pipeline_roundtrip() {
        let width = 8;
        let height = 4;
        let mut rgb = Vec::with_capacity(width * height * 3);
        for i in 0..(width * height) {
            rgb.push(((i * 13) % 17) as f32 / 17.0);
            rgb.push(((i * 7) % 11) as f32 / 11.0);
            rgb.push(((i * 19) % 23) as f32 / 23.0);
        }
        let mut y = vec![0.0_f32; width * height];
        let mut cb = vec![0.0_f32; width * height];
        let mut cr = vec![0.0_f32; width * height];
        rgb_to_ycbcr_planar(width, height, &rgb, &mut y, &mut cb, &mut cr);
        // Full444: no downsample. Convert straight back.
        for i in 0..(width * height) {
            let (r2, g2, b2) = ycbcr_to_rgb_pixel(y[i], cb[i], cr[i]);
            assert!((rgb[i * 3] - r2).abs() < 1.0e-6);
            assert!((rgb[i * 3 + 1] - g2).abs() < 1.0e-6);
            assert!((rgb[i * 3 + 2] - b2).abs() < 1.0e-6);
        }
    }

    /// Sub420 pipeline on a constant-colour input: downsampled chroma
    /// equals the constant, and the chroma-upsample step (replicate
    /// each chroma sample 2×2 back into a full-res plane) reconstructs
    /// the original colour exactly. This is the *easy* sanity check
    /// before chunk 3 wires in real strategy search.
    #[test]
    fn test_sub420_constant_color_no_loss() {
        let width = 8;
        let height = 4;
        // Constant blue-ish patch.
        let r = 0.2_f32;
        let g = 0.4_f32;
        let b = 0.8_f32;
        let rgb: Vec<f32> = (0..(width * height)).flat_map(|_| [r, g, b]).collect();
        let mut y_full = vec![0.0_f32; width * height];
        let mut cb_full = vec![0.0_f32; width * height];
        let mut cr_full = vec![0.0_f32; width * height];
        rgb_to_ycbcr_planar(width, height, &rgb, &mut y_full, &mut cb_full, &mut cr_full);
        // Downsample chroma 2x2 → constant patch stays the same value.
        let dst_w = div_ceil(width, 2);
        let dst_h = div_ceil(height, 2);
        let mut cb_sub = vec![0.0_f32; dst_w * dst_h];
        let mut cr_sub = vec![0.0_f32; dst_w * dst_h];
        box_downsample_2x_both(width, height, &cb_full, dst_w, dst_h, &mut cb_sub);
        box_downsample_2x_both(width, height, &cr_full, dst_w, dst_h, &mut cr_sub);
        // Verify every chroma sample equals the original.
        let (_, ecb, ecr) = rgb_to_ycbcr_pixel(r, g, b);
        for &v in &cb_sub {
            assert!((v - ecb).abs() < 1.0e-6);
        }
        for &v in &cr_sub {
            assert!((v - ecr).abs() < 1.0e-6);
        }
        // Y stays full-res, every sample equals the original Y.
        let (ey, _, _) = rgb_to_ycbcr_pixel(r, g, b);
        for &v in &y_full {
            assert!((v - ey).abs() < 1.0e-6);
        }
        // Reconstruct: upsample chroma by 2×2 replication and convert
        // back. For a constant patch the result must round-trip
        // exactly (within f32 noise).
        for j in 0..height {
            for i in 0..width {
                let cbv = cb_sub[(j / 2) * dst_w + (i / 2)];
                let crv = cr_sub[(j / 2) * dst_w + (i / 2)];
                let (r2, g2, b2) = ycbcr_to_rgb_pixel(y_full[j * width + i], cbv, crv);
                assert!((r - r2).abs() < 1.0e-6);
                assert!((g - g2).abs() < 1.0e-6);
                assert!((b - b2).abs() < 1.0e-6);
            }
        }
    }
}
