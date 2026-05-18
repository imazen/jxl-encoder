// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chroma subsampling conversion helpers (issue #47 chunk 3).
//!
//! Foundational building blocks for the eventual end-to-end Sub420 /
//! Sub422 / Sub440 lossy pipeline (chunk 4). All conversion is
//! delegated to the production [`zenyuv`] crate so we inherit its
//! AVX2 / NEON / WASM SIMD dispatch (archmage tokens) and its
//! L2-optimal Newton-step Sharp YUV implementation.
//!
//! # Status
//!
//! - **Chunk 3 (this commit)**: helpers ship + are unit-tested in
//!   isolation. The public [`crate::api::LossyConfig`] still rejects
//!   any non-`Full444` [`crate::api::ChromaSubsampling`] with
//!   [`crate::api::EncodeError::InvalidConfig`] — the conversion
//!   functions here are decoupled from that gate so callers can use
//!   them today (e.g. for offline content analysis / sharp-YUV
//!   quality comparisons).
//!
//! - **Chunk 4 (queued)**: wire these helpers through the VarDCT
//!   encode pipeline. The smallest viable path is the JPEG
//!   transcode-shaped pipeline ([`crate::jpeg::encode`]), which
//!   already supports `do_ycbcr=true` + `jpeg_upsampling=[1,0,1]` +
//!   per-channel block grids — feed it RGB → YCbCr+420 from
//!   [`rgb_to_yuv420_sharp`] / [`rgb_to_yuv420_box`] instead of the
//!   parsed JPEG payload.
//!
//! # Why zenyuv vs a homegrown helper
//!
//! Earlier scratch work (PR #48, superseded) included homegrown
//! `rgb_to_ycbcr_planar` + `box_downsample_2x_both` helpers. zenyuv
//! 0.1.3 ships:
//!
//! - AVX2 / NEON / WASM SIMD dispatch (32 px/iter on AVX2; 16 on
//!   NEON / WASM).
//! - BT.601 / BT.709 / BT.2020 matrices in both Full and Limited
//!   range, byte-identical across SIMD tiers.
//! - Sharp YUV (L2-optimal Newton step Cb/Cr refinement) for 4:2:0 —
//!   measurably better visual quality on high-contrast edges than
//!   box-filter downsampling, with `#[forbid(unsafe_code)]` + SIMD.
//! - `#[forbid(unsafe_code)]` + `no_std + alloc` — same constraints
//!   as this crate.
//!
//! Reusing a battle-tested SIMD kernel is the right call.

use alloc::vec;
use alloc::vec::Vec;

use zenyuv::{Matrix, Range, SharpYuvConfig, YuvContext};

use crate::api::ChromaSubsampling;
use crate::headers::frame_header::{Encoding, FrameHeader};

/// Three planar YCbCr u8 buffers as `[Y, Cb, Cr]`.
///
/// Channel order is `[Cb, Y, Cr]` in JXL's `jpeg_upsampling` and
/// `do_ycbcr` plane layout (libjxl `frame_header.h:81`); the JPEG
/// reencoding path remaps to that order at emit time
/// (`jxl-encoder/src/jpeg/encode.rs:58-64`). This struct uses the
/// natural source order `(Y, Cb, Cr)` to match zenyuv's plane
/// arguments and avoid one source of confusion; the JXL-side reorder
/// is the consumer's responsibility (chunk 4 will do it inside the
/// pipeline glue, matching the JPEG path).
///
/// For [`ChromaSubsampling::Full444`] all three planes have length
/// `width * height`. For [`ChromaSubsampling::Sub420`] Y has length
/// `width * height` and Cb/Cr have length
/// `ceil(width / 2) * ceil(height / 2)`. Sub422 / Sub440 are not
/// produced by this module (Sharp YUV is 4:2:0-only in zenyuv 0.1.3;
/// 4:2:2 / 4:4:0 will be added in chunk 4 alongside the wire-up).
#[derive(Debug, Clone)]
pub struct YCbCrPlanes {
    /// Y plane (luma). Length `width * height`.
    pub y: Vec<u8>,
    /// Cb plane (chroma blue). Length depends on subsampling mode.
    pub cb: Vec<u8>,
    /// Cr plane (chroma red). Length depends on subsampling mode.
    pub cr: Vec<u8>,
    /// Chroma plane width (`width` for 4:4:4 / 4:4:0,
    /// `ceil(width / 2)` for 4:2:0 / 4:2:2).
    pub chroma_width: usize,
    /// Chroma plane height (`height` for 4:4:4 / 4:2:2,
    /// `ceil(height / 2)` for 4:2:0 / 4:4:0).
    pub chroma_height: usize,
}

/// Convert interleaved RGB to 4:4:4 planar YCbCr via BT.601 Full
/// range (JFIF Clause 7 — the libjxl `kYCbCrStage::ProcessRow` decode
/// inverse, `lib/jxl/render_pipeline/stage_ycbcr.cc:24-39`).
///
/// `rgb` must be `width * height * 3` bytes in row-major RGBRGBRGB
/// order. Output Y/Cb/Cr planes are each `width * height` bytes.
///
/// Delegates to [`YuvContext::encode_444_u8`].
pub fn rgb_to_ycbcr_444(rgb: &[u8], width: usize, height: usize) -> YCbCrPlanes {
    assert!(
        rgb.len() >= width * height * 3,
        "rgb buffer too small: {} < {}",
        rgb.len(),
        width * height * 3
    );
    let n = width * height;
    let mut y = vec![0u8; n];
    let mut cb = vec![0u8; n];
    let mut cr = vec![0u8; n];
    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
    ctx.encode_444_u8(rgb, &mut y, &mut cb, &mut cr, width, height);
    YCbCrPlanes {
        y,
        cb,
        cr,
        chroma_width: width,
        chroma_height: height,
    }
}

/// Convert interleaved RGB to 4:2:0 planar YCbCr via box-filter
/// chroma downsample (BT.601 Full range).
///
/// Y is computed at full resolution; Cb/Cr are 2x2-block averaged
/// chroma at half resolution in each dimension. Use this when speed
/// matters and the input has gentle chroma transitions (photos with
/// soft colour gradients).
///
/// For sharper output on screenshots / UI / high-contrast edges, use
/// [`rgb_to_yuv420_sharp`] — it costs more CPU (one extra refinement
/// pass per 2x2 block) but reduces chroma reconstruction error
/// measurably.
///
/// Delegates to [`YuvContext::encode_420_u8`].
pub fn rgb_to_yuv420_box(rgb: &[u8], width: usize, height: usize) -> YCbCrPlanes {
    assert!(
        rgb.len() >= width * height * 3,
        "rgb buffer too small: {} < {}",
        rgb.len(),
        width * height * 3
    );
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut y = vec![0u8; width * height];
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
    ctx.encode_420_u8(rgb, &mut y, &mut cb, &mut cr, width, height);
    YCbCrPlanes {
        y,
        cb,
        cr,
        chroma_width: cw,
        chroma_height: ch,
    }
}

/// Convert interleaved RGB to 4:2:0 planar YCbCr with Sharp YUV
/// chroma refinement (BT.601 Full range).
///
/// Y is computed at full resolution via the SIMD kernel; Cb/Cr are
/// refined via zenyuv's L2-optimal Newton step (matches libwebp's
/// `SharpYuvConvert` algorithm; 25× faster than the original scalar
/// implementation per zenyuv's bench data, with better quality
/// thanks to the correct Jacobian vs hand-tuned damping constants).
///
/// A Y-refinement pass (`SharpYuvConfig::refine_y`, on by default)
/// adjusts the full-res Y to compensate for the luma error
/// introduced by 4:2:0 chroma subsampling.
///
/// Use this for any content where chroma fidelity matters
/// (screenshots, UI, line art, anything with high-contrast colour
/// edges). For photos with gentle chroma transitions the cheaper
/// box-filter [`rgb_to_yuv420_box`] is usually visually equivalent.
///
/// Delegates to [`zenyuv::sharp::rgb_to_yuv420_sharp`].
pub fn rgb_to_yuv420_sharp(rgb: &[u8], width: usize, height: usize) -> YCbCrPlanes {
    assert!(
        rgb.len() >= width * height * 3,
        "rgb buffer too small: {} < {}",
        rgb.len(),
        width * height * 3
    );
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut y = vec![0u8; width * height];
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    let config = SharpYuvConfig::default();
    zenyuv::sharp::rgb_to_yuv420_sharp(
        rgb,
        &mut y,
        &mut cb,
        &mut cr,
        width,
        height,
        Range::Full,
        Matrix::Bt601,
        &config,
    );
    YCbCrPlanes {
        y,
        cb,
        cr,
        chroma_width: cw,
        chroma_height: ch,
    }
}

/// Per-channel `jpeg_upsampling` mode index (the RAW value the JXL
/// codestream stores in `FrameHeader.jpeg_upsampling[c]`, NOT the
/// per-channel actual shift).
///
/// Channel order is `[Cb, Y, Cr]` to match
/// [`FrameHeader::jpeg_upsampling`]. The actual decoder shift is
/// `HShift(c) = max(JPEG_UPSAMPLING_H_SHIFT[mode_c'])
///   − JPEG_UPSAMPLING_H_SHIFT[mode_c]` (see
/// `jxl-encoder/src/jpeg/encode.rs:85-107` for the JPEG-side
/// computation). The four [`ChromaSubsampling`] modes round-trip to
/// these raw triples:
///
/// | Mode    | jpeg_upsampling[Cb] | [Y] | [Cr] | Decoder Cb/Cr shift |
/// |---------|---------------------|-----|------|---------------------|
/// | Full444 | 0                   | 0   | 0    | (0,0)               |
/// | Sub422  | 0                   | 2   | 0    | (1,0)               |
/// | Sub420  | 0                   | 1   | 0    | (1,1)               |
/// | Sub440  | 0                   | 3   | 0    | (0,1)               |
///
/// Y carries the full-resolution mode tag (1=2×2, 2=2×1, 3=1×2,
/// 0=1×1) and Cb/Cr both stay at 0 (factor 1×1 — the actual chroma
/// "downsample" emerges from `max − Cb_mode_shift`).
///
/// This matches libjxl's `YCbCrChromaSubsampling::Set(modes)` —
/// callers pass the *mode index* of each channel, not the shift.
pub const fn jpeg_upsampling_for(mode: ChromaSubsampling) -> [u8; 3] {
    match mode {
        ChromaSubsampling::Full444 => [0, 0, 0],
        ChromaSubsampling::Sub422 => [0, 2, 0],
        ChromaSubsampling::Sub420 => [0, 1, 0],
        ChromaSubsampling::Sub440 => [0, 3, 0],
    }
}

/// Build a minimal VarDCT [`FrameHeader`] for non-JPEG-transcode
/// YCbCr encoding.
///
/// Mirrors [`crate::jpeg::encode::build_jpeg_frame_header`] but
/// without the JPEG-specific `flags = 0x80` (SKIP_ADAPTIVE_LF_SMOOTHING)
/// / `gaborish = false` / `epf_iters = 0` / `x_qm_scale = 2`
/// overrides — those exist because JPEG-quantised coefficients
/// already lock in the smoothing/filtering decisions, and the
/// transcode path needs to preserve byte-exact decode of the source
/// JPEG. A fresh YCbCr VarDCT encode keeps the standard defaults so
/// the encoder can still tune gaborish / EPF for the actual content.
///
/// **Not wired into any encode path yet** (chunk 4). Provided so the
/// chunk-3 helpers form a coherent building block — anything that
/// runs zenyuv → DCT8 → quantize → bitstream for a chunk-4 demo can
/// stamp the frame header via this and stay parity with the JPEG
/// path's signaling.
pub fn build_ycbcr_vardct_frame_header(mode: ChromaSubsampling) -> FrameHeader {
    FrameHeader {
        encoding: Encoding::VarDct,
        xyb_encoded: false,
        do_ycbcr: true,
        jpeg_upsampling: jpeg_upsampling_for(mode),
        ..FrameHeader::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth gradient — exercises the SIMD encode kernels without
    /// the high-frequency content that would inflate roundtrip
    /// error.
    fn make_gradient_rgb(width: usize, height: usize) -> Vec<u8> {
        let mut out = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                let denom_x = width.max(1);
                let denom_y = height.max(1);
                let denom_xy = (width + height).max(1);
                out[i] = ((x * 255) / denom_x) as u8;
                out[i + 1] = ((y * 255) / denom_y) as u8;
                out[i + 2] = (((x + y) * 255) / denom_xy) as u8;
            }
        }
        out
    }

    #[test]
    fn rgb_to_ycbcr_444_full_resolution() {
        let (w, h) = (32, 16);
        let rgb = make_gradient_rgb(w, h);
        let planes = rgb_to_ycbcr_444(&rgb, w, h);
        assert_eq!(planes.y.len(), w * h);
        assert_eq!(planes.cb.len(), w * h);
        assert_eq!(planes.cr.len(), w * h);
        assert_eq!(planes.chroma_width, w);
        assert_eq!(planes.chroma_height, h);
    }

    #[test]
    fn rgb_to_yuv420_box_halves_chroma() {
        let (w, h) = (64, 48);
        let rgb = make_gradient_rgb(w, h);
        let planes = rgb_to_yuv420_box(&rgb, w, h);
        assert_eq!(planes.y.len(), w * h);
        assert_eq!(planes.cb.len(), (w / 2) * (h / 2));
        assert_eq!(planes.cr.len(), (w / 2) * (h / 2));
        assert_eq!(planes.chroma_width, w / 2);
        assert_eq!(planes.chroma_height, h / 2);
    }

    #[test]
    fn rgb_to_yuv420_box_odd_dimensions_round_up() {
        // 33×17 → chroma 17×9 (libwebp / libjxl convention).
        let (w, h) = (33, 17);
        let rgb = make_gradient_rgb(w, h);
        let planes = rgb_to_yuv420_box(&rgb, w, h);
        assert_eq!(planes.chroma_width, 17);
        assert_eq!(planes.chroma_height, 9);
        assert_eq!(planes.cb.len(), 17 * 9);
        assert_eq!(planes.cr.len(), 17 * 9);
    }

    #[test]
    fn rgb_to_yuv420_sharp_runs_and_returns_half_res_chroma() {
        let (w, h) = (64, 64);
        let rgb = make_gradient_rgb(w, h);
        let sharp = rgb_to_yuv420_sharp(&rgb, w, h);
        let boxd = rgb_to_yuv420_box(&rgb, w, h);

        // Both produce same plane sizes.
        assert_eq!(sharp.y.len(), boxd.y.len());
        assert_eq!(sharp.cb.len(), boxd.cb.len());
        assert_eq!(sharp.cr.len(), boxd.cr.len());

        // Sharp Y may differ from box Y by ±1 (libwebp `SharpYuvUpdateY`
        // refinement) but never by more — pin the loose upper bound.
        let max_y_drift = sharp
            .y
            .iter()
            .zip(boxd.y.iter())
            .map(|(s, b)| s.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_y_drift <= 4,
            "sharp Y drift {} > 4 (expected refinement to stay tight)",
            max_y_drift
        );

        // Sharp chroma MUST differ from box chroma on a gradient (the
        // whole point of iterative refinement). If both are byte-identical
        // the sharp kernel didn't fire.
        let chroma_diff_sum: u32 = sharp
            .cb
            .iter()
            .zip(boxd.cb.iter())
            .chain(sharp.cr.iter().zip(boxd.cr.iter()))
            .map(|(s, b)| s.abs_diff(*b) as u32)
            .sum();
        assert!(
            chroma_diff_sum > 0,
            "sharp chroma identical to box — refinement no-op"
        );
    }

    #[test]
    fn jpeg_upsampling_for_full444_is_zeros() {
        assert_eq!(jpeg_upsampling_for(ChromaSubsampling::Full444), [0, 0, 0]);
    }

    #[test]
    fn jpeg_upsampling_for_round_trips_to_h_v_shifts() {
        // For each mode, derive (h_shift, v_shift) from the raw
        // jpeg_upsampling triple the same way the decoder does
        // (`HShift(c) = maxhs - kHShift[mode_c]`) and compare to the
        // public `ChromaSubsampling::h_shifts()` / `v_shifts()` API.
        // This pins the chunk-3 helpers consistent with the chunk-1
        // API surface so a refactor of either can't drift them apart
        // silently.
        //
        // JPEG_UPSAMPLING_H_SHIFT / V_SHIFT in
        // `jxl-encoder/src/jpeg/data.rs`:
        //   mode 0 (1×1): (0, 0)
        //   mode 1 (2×2): (1, 1)
        //   mode 2 (2×1): (1, 0)
        //   mode 3 (1×2): (0, 1)
        const JPEG_H: [u8; 4] = [0, 1, 1, 0];
        const JPEG_V: [u8; 4] = [0, 1, 0, 1];
        for mode in [
            ChromaSubsampling::Full444,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub420,
            ChromaSubsampling::Sub440,
        ] {
            let raw = jpeg_upsampling_for(mode);
            let max_h = raw.iter().map(|&u| JPEG_H[u as usize]).max().unwrap();
            let max_v = raw.iter().map(|&u| JPEG_V[u as usize]).max().unwrap();
            let derived_h = [
                max_h - JPEG_H[raw[0] as usize],
                max_h - JPEG_H[raw[1] as usize],
                max_h - JPEG_H[raw[2] as usize],
            ];
            let derived_v = [
                max_v - JPEG_V[raw[0] as usize],
                max_v - JPEG_V[raw[1] as usize],
                max_v - JPEG_V[raw[2] as usize],
            ];
            assert_eq!(
                derived_h,
                mode.h_shifts(),
                "{:?} h_shifts mismatch (raw={raw:?}, derived={derived_h:?})",
                mode
            );
            assert_eq!(
                derived_v,
                mode.v_shifts(),
                "{:?} v_shifts mismatch (raw={raw:?}, derived={derived_v:?})",
                mode
            );
        }
    }

    #[test]
    fn build_ycbcr_vardct_frame_header_signals_correctly() {
        let header = build_ycbcr_vardct_frame_header(ChromaSubsampling::Sub420);
        assert!(header.do_ycbcr, "do_ycbcr must be true for YCbCr encoding");
        assert!(
            !header.xyb_encoded,
            "xyb_encoded must be false when do_ycbcr is true"
        );
        assert_eq!(header.encoding, Encoding::VarDct);
        assert_eq!(header.jpeg_upsampling, [0, 1, 0]);
    }

    #[test]
    fn build_ycbcr_vardct_frame_header_full444_jpeg_upsampling_is_zero() {
        let header = build_ycbcr_vardct_frame_header(ChromaSubsampling::Full444);
        assert!(header.do_ycbcr);
        assert_eq!(header.jpeg_upsampling, [0, 0, 0]);
    }

    /// White / black pixels must round through 4:4:4 with chroma = 128.
    /// Pins the canonical YCbCr identity (R=G=B → Cb=Cr=128) for the
    /// zenyuv kernel via our wrapper.
    #[test]
    fn rgb_white_and_black_have_chroma_128_in_444() {
        let rgb: Vec<u8> = [[0u8, 0, 0], [255, 255, 255]]
            .iter()
            .flatten()
            .copied()
            .collect();
        let planes = rgb_to_ycbcr_444(&rgb, 2, 1);
        assert_eq!(planes.y, [0, 255]);
        assert_eq!(planes.cb, [128, 128]);
        assert_eq!(planes.cr, [128, 128]);
    }
}
