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

// ── chunk 4 — Sub420 end-to-end via `jpeg::encode` ──────────────────────────
//
// The chunk-4 strategy (see module-level doc): we don't refactor the
// VarDCT pipeline to grow per-channel block grids. Instead, we convert
// RGB → YCbCr+420 with zenyuv, run a standard 8×8 forward DCT +
// integer quantization on every block of each plane (Y at full res,
// Cb/Cr at half res), pack the quantized coefficients into a
// [`JpegData`] payload, and hand it to
// [`crate::jpeg::encode::encode_jpeg_to_jxl`]. That encoder already
// supports `do_ycbcr=true` + `jpeg_upsampling=[0,1,0]` +
// per-component block dimensions, so we get a decoder-roundtrippable
// 4:2:0 bitstream without touching the standard VarDCT pipeline.
//
// The chunk-4 path is intentionally **lossy** (quantization is not
// bit-exact to any specific cjxl output) but the bitstream IS valid
// JXL and decodes to YCbCr pixels with the requested 4:2:0 chroma
// shape. RD parity with cjxl is chunk-5+ territory.

#[cfg(feature = "jpeg-reencoding")]
use crate::jpeg::JpegData;

/// JPEG-style standard luma quantization table (Annex K Table K.1).
///
/// Stored in natural row-major order. Scaled by the encoder's
/// quality-derived factor (see [`distance_to_jpeg_quality`]) to
/// produce per-distance Y quant tables.
#[cfg(feature = "jpeg-reencoding")]
const STD_LUMA_QT: [i32; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// JPEG-style standard chroma quantization table (Annex K Table K.2).
#[cfg(feature = "jpeg-reencoding")]
const STD_CHROMA_QT: [i32; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// Approximate Butteraugli distance → libjpeg-style 1..100 quality
/// factor. Used only to scale the standard luma/chroma tables for the
/// chunk-4 Sub420 path. Not a calibrated mapping (chunk 5+ will tune
/// per-distance quant matrices against cjxl) — just a monotonic
/// "smaller distance → higher quality, finer quant" curve that keeps
/// quant values >= 1 across the supported `distance` range.
#[cfg(feature = "jpeg-reencoding")]
fn distance_to_jpeg_quality(distance: f32) -> i32 {
    // Anchor points (rough libjxl distance vs subjective JPEG quality):
    //   d=0.5 ↔ q≈92, d=1.0 ↔ q≈85, d=2.0 ↔ q≈75, d=4.0 ↔ q≈60.
    // Linear interpolation in log(distance) space is good enough for
    // the chunk-4 acceptance test (valid roundtrip; no RD parity claim).
    let d = distance.clamp(0.05, 25.0);
    let q = 95.0_f32 - 15.0_f32 * d.log2().max(-2.0);
    q.round().clamp(1.0, 100.0) as i32
}

/// Scale a standard JPEG quant table by a libjpeg-style quality
/// factor `1..100`. Mirrors libjpeg's `jpeg_quality_scaling` formula.
#[cfg(feature = "jpeg-reencoding")]
fn scale_qt(base: &[i32; 64], quality: i32) -> [i32; 64] {
    // libjpeg-style scale: q < 50 -> 5000/q, else 200 - 2*q.
    let scale = if quality < 50 {
        5000 / quality
    } else {
        200 - 2 * quality
    };
    let mut out = [0i32; 64];
    for i in 0..64 {
        let v = (base[i] * scale + 50) / 100;
        out[i] = v.clamp(1, 255);
    }
    out
}

/// Reference 8×8 forward DCT-II that produces JPEG natural-order
/// coefficients (row-major, frequency `(u, v)` at `coeffs[u*8+v]`).
///
/// Implements the standard separable 2D DCT-II with libjpeg
/// normalisation: `C[u,v] = (1/4) * a(u) * a(v) * sum_{x,y}
/// input[y,x] * cos(...) * cos(...)`. Not the fastest possible
/// implementation but trivially correct — the chunk-4 path runs
/// per-block, not per-row-strip, so the speed difference is
/// invisible at our target sizes. Reuse via the jxl `dct_8x8` would
/// require an additional transpose (jxl emits coeffs in transposed
/// layout — see `vardct/dct/forward.rs:128`) and would entangle the
/// jpeg-shaped path with the VarDCT DCT module's scaling convention;
/// a small standalone implementation is easier to audit.
#[cfg(feature = "jpeg-reencoding")]
fn forward_dct_8x8_natural(input: &[f32; 64], output: &mut [f32; 64]) {
    // Per-row 1D DCT-II into a temp.
    let mut tmp = [0.0_f32; 64];
    for y in 0..8 {
        dct1d_8_natural(
            &[
                input[y * 8],
                input[y * 8 + 1],
                input[y * 8 + 2],
                input[y * 8 + 3],
                input[y * 8 + 4],
                input[y * 8 + 5],
                input[y * 8 + 6],
                input[y * 8 + 7],
            ],
            &mut tmp[y * 8..y * 8 + 8],
        );
    }
    // Per-column 1D DCT-II into output.
    for x in 0..8 {
        let col = [
            tmp[x],
            tmp[8 + x],
            tmp[16 + x],
            tmp[24 + x],
            tmp[32 + x],
            tmp[40 + x],
            tmp[48 + x],
            tmp[56 + x],
        ];
        let mut col_out = [0.0_f32; 8];
        dct1d_8_natural(&col, &mut col_out);
        for u in 0..8 {
            output[u * 8 + x] = col_out[u];
        }
    }
}

/// 1D DCT-II of length 8 with libjpeg orthonormalisation
/// (`a(0) = 1/sqrt(2)`, `a(k) = 1` for k > 0, overall `1/2`).
///
/// The combined 2D scaling (`(1/2)^2 = 1/4` plus per-axis `a(u)*a(v)`)
/// is split across the two passes so the per-pass output is
/// orthonormal. Specifically each 1D pass multiplies by `sqrt(2/N) =
/// 1/2` and the DC term gets the additional `1/sqrt(2)` factor.
#[cfg(feature = "jpeg-reencoding")]
fn dct1d_8_natural(input: &[f32; 8], output: &mut [f32]) {
    let pi_over_16 = core::f32::consts::PI / 16.0;
    for (u, out_u) in output.iter_mut().enumerate().take(8) {
        let mut s = 0.0_f32;
        for (x, &in_x) in input.iter().enumerate() {
            // (2x + 1) * u * pi / 16
            let theta = ((2 * x + 1) as f32) * (u as f32) * pi_over_16;
            s += in_x * theta.cos();
        }
        // 1D normalisation: sqrt(2/N) = 1/2, with extra 1/sqrt(2) for DC.
        let a = if u == 0 {
            1.0 / core::f32::consts::SQRT_2
        } else {
            1.0
        };
        *out_u = 0.5 * a * s;
    }
}

/// Forward-DCT + quantize one 8×8 pixel block (u8 input, level-shifted
/// by −128 per JPEG convention) into 64 i16 natural-order quantized
/// coefficients.
#[cfg(feature = "jpeg-reencoding")]
fn fdct_quantize_block(pixels: &[u8; 64], qtable: &[i32; 64], out: &mut [i16; 64]) {
    let mut f = [0.0_f32; 64];
    for i in 0..64 {
        f[i] = pixels[i] as f32 - 128.0;
    }
    let mut coeffs = [0.0_f32; 64];
    forward_dct_8x8_natural(&f, &mut coeffs);
    for i in 0..64 {
        let q = qtable[i].max(1) as f32;
        // Symmetric rounding away from zero matches libjpeg's
        // `(coeff + sign * q/2) / q` integer divide convention.
        let v = (coeffs[i] / q).round() as i32;
        out[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Extract an 8×8 pixel block from a planar buffer, with right/bottom
/// edge replication for out-of-bounds samples (standard JPEG
/// boundary-pad convention).
#[cfg(feature = "jpeg-reencoding")]
fn extract_block(
    plane: &[u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    out: &mut [u8; 64],
) {
    let x0 = block_x * 8;
    let y0 = block_y * 8;
    for j in 0..8 {
        let y = (y0 + j).min(height.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(width.saturating_sub(1));
            out[j * 8 + i] = plane[y * width + x];
        }
    }
}

/// Forward-DCT-quantize every 8×8 block of a planar u8 buffer and
/// return the concatenated natural-order quantized coefficients
/// (block-by-block in raster order).
#[cfg(feature = "jpeg-reencoding")]
fn fdct_quantize_plane(
    plane: &[u8],
    width: usize,
    height: usize,
    qtable: &[i32; 64],
) -> (Vec<i16>, u32, u32) {
    let w_blocks = width.div_ceil(8);
    let h_blocks = height.div_ceil(8);
    let n = w_blocks * h_blocks;
    let mut coeffs = vec![0_i16; n * 64];
    let mut pixels = [0_u8; 64];
    let mut block = [0_i16; 64];
    for by in 0..h_blocks {
        for bx in 0..w_blocks {
            extract_block(plane, width, height, bx, by, &mut pixels);
            fdct_quantize_block(&pixels, qtable, &mut block);
            let dst = (by * w_blocks + bx) * 64;
            coeffs[dst..dst + 64].copy_from_slice(&block);
        }
    }
    (coeffs, w_blocks as u32, h_blocks as u32)
}

/// Synthesise a [`JpegData`] payload from an interleaved RGB buffer
/// for the chunk-4 Sub420 path.
///
/// Pipeline:
/// 1. RGB → YCbCr 4:2:0 via [`rgb_to_yuv420_sharp`] (zenyuv Sharp YUV).
/// 2. Forward 8×8 DCT-II + integer quantisation on every block of Y
///    (full res) and Cb / Cr (half res).
/// 3. Pack the resulting i16 coefficient arrays into [`JpegData`]
///    fields so [`crate::jpeg::encode::encode_jpeg_to_jxl`] can
///    consume it.
///
/// The synthetic [`JpegData`] omits scan_info / marker_order /
/// inter_marker_data / huffman_code / app_data — none of those are
/// read by the JXL-emit half of the JPEG path (see
/// `jpeg/encode.rs:51-456`). JBRD reconstruction is NOT possible
/// from this output (the source bytes are RGB pixels, not a JPEG);
/// the synthesised payload is only valid as input to
/// `encode_jpeg_to_jxl` (codestream-only).
#[cfg(feature = "jpeg-reencoding")]
fn synth_jpeg_data_from_rgb8_sub420(
    rgb: &[u8],
    width: usize,
    height: usize,
    distance: f32,
) -> Result<JpegData, crate::error::Error> {
    if width == 0 || height == 0 {
        return Err(crate::error::Error::InvalidInput(format!(
            "synth_jpeg_data_from_rgb8_sub420 requires non-zero dimensions, got {width}x{height}"
        )));
    }
    if rgb.len() < width * height * 3 {
        return Err(crate::error::Error::InvalidInput(format!(
            "synth_jpeg_data_from_rgb8_sub420 rgb buffer too small: {} < {}",
            rgb.len(),
            width * height * 3
        )));
    }

    let quality = distance_to_jpeg_quality(distance);
    let luma_qt = scale_qt(&STD_LUMA_QT, quality);
    let chroma_qt = scale_qt(&STD_CHROMA_QT, quality);

    let planes = rgb_to_yuv420_sharp(rgb, width, height);

    // Y plane: full resolution.
    let (y_coeffs, y_w_blocks, y_h_blocks) =
        fdct_quantize_plane(&planes.y, width, height, &luma_qt);
    // Cb / Cr planes: chroma_width × chroma_height (half-res in each axis).
    let (cb_coeffs, cb_w_blocks, cb_h_blocks) = fdct_quantize_plane(
        &planes.cb,
        planes.chroma_width,
        planes.chroma_height,
        &chroma_qt,
    );
    let (cr_coeffs, cr_w_blocks, cr_h_blocks) = fdct_quantize_plane(
        &planes.cr,
        planes.chroma_width,
        planes.chroma_height,
        &chroma_qt,
    );
    debug_assert_eq!((cb_w_blocks, cb_h_blocks), (cr_w_blocks, cr_h_blocks));

    // Build component list. Channel order is JPEG-native (Y, Cb, Cr);
    // the JXL encode side remaps via `jpeg_c_map = [1, 0, 2]` per the
    // libjxl convention (`jpeg/encode.rs:62`).
    let comp_y = jpeg_component(1, 2, 2, 0, y_w_blocks, y_h_blocks, y_coeffs);
    let comp_cb = jpeg_component(2, 1, 1, 1, cb_w_blocks, cb_h_blocks, cb_coeffs);
    let comp_cr = jpeg_component(3, 1, 1, 1, cr_w_blocks, cr_h_blocks, cr_coeffs);

    let quant_y = jpeg_quant_table(&luma_qt, 0, false);
    let quant_c = jpeg_quant_table(&chroma_qt, 1, true);

    Ok(JpegData {
        width: width as u32,
        height: height as u32,
        restart_interval: 0,
        app_data: Vec::new(),
        app_marker_type: Vec::new(),
        com_data: Vec::new(),
        quant: alloc::vec![quant_y, quant_c],
        huffman_code: Vec::new(),
        components: alloc::vec![comp_y, comp_cb, comp_cr],
        scan_info: Vec::new(),
        marker_order: Vec::new(),
        inter_marker_data: Vec::new(),
        tail_data: Vec::new(),
        has_zero_padding_bit: false,
        padding_bits: Vec::new(),
        component_type: crate::jpeg::JpegComponentType::YCbCr,
    })
}

#[cfg(feature = "jpeg-reencoding")]
fn jpeg_component(
    id: u32,
    h_samp: u32,
    v_samp: u32,
    quant_idx: u32,
    width_in_blocks: u32,
    height_in_blocks: u32,
    coeffs: Vec<i16>,
) -> crate::jpeg::JpegComponent {
    crate::jpeg::JpegComponent {
        id,
        h_samp_factor: h_samp,
        v_samp_factor: v_samp,
        quant_idx,
        width_in_blocks,
        height_in_blocks,
        coeffs,
    }
}

#[cfg(feature = "jpeg-reencoding")]
fn jpeg_quant_table(values: &[i32; 64], index: u32, is_last: bool) -> crate::jpeg::JpegQuantTable {
    crate::jpeg::JpegQuantTable {
        values: *values,
        precision: 0,
        index,
        is_last,
    }
}

/// Encode an interleaved RGB8 image as a JXL codestream with
/// chroma subsampling 4:2:0. Public entry point for the chunk-4
/// Sub420 path; called from the `LossyConfig` lossy encoder when the
/// caller picks [`ChromaSubsampling::Sub420`].
///
/// Output is a bare JXL codestream (not a container) — same shape as
/// what [`crate::jpeg::encode::encode_jpeg_to_jxl`] returns. The
/// `distance` parameter only controls the JPEG quant table scaling
/// (see [`distance_to_jpeg_quality`]); it does not run a butteraugli
/// loop or per-block adaptive quant. RD parity with cjxl is chunk-5+
/// territory; this entry just produces a decoder-roundtrippable
/// Sub420 bitstream.
#[cfg(feature = "jpeg-reencoding")]
pub fn encode_rgb8_sub420_via_jpeg_path(
    rgb: &[u8],
    width: usize,
    height: usize,
    distance: f32,
) -> Result<Vec<u8>, crate::error::Error> {
    let jpeg = synth_jpeg_data_from_rgb8_sub420(rgb, width, height, distance)?;
    crate::jpeg::encode_jpeg_to_jxl(&jpeg)
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
