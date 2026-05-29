// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy JPEG → JXL recompression to a perceptual-quality **target**.
//!
//! Productizes the **PreserveJxl** coefficient-domain path (from `jxl-encoder`,
//! `encode_jpeg_recompress_auto_codestream`) into a closed loop that hits a
//! chosen quality target by bisecting the coarsening scale, scoring each
//! candidate **in-process** (encode → decode → score). This is the natural home
//! for the loop because `zenjxl` already deps both the encoder *and* the decoder.
//!
//! The loop is **metric-agnostic**: the caller supplies a [`RelativeScorer`]
//! over decoded RGB8 pixels, so the same driver hits a zensim-A / cvvdp /
//! butteraugli (or any) target. This matches the "caller selects the metric"
//! design used elsewhere in the codec stack.
//!
//! Targets here are **relative** (generation loss vs the source's own decoded
//! pixels): the reference is the lossless transcode (scale 1.0). Inferred
//! (absolute-vs-original) targeting layers on top via the source-quality floor
//! calibration — see `jxl-encoder/docs/JPEG_LOSSY_RECOMPRESSION.md`.
//!
//! Requires the `jpeg-lossy` cargo feature.

use alloc::vec::Vec;

use jxl_encoder::jpeg::encode_jpeg_recompress_auto_codestream;
use zenpixels::PixelDescriptor;

use crate::decode::decode;
use crate::error::JxlError;

type At<E> = whereat::At<E>;

/// A scorer over `(reference_rgb8, distorted_rgb8, width, height) -> score`.
///
/// Both slices are tightly-packed RGB8 (3 bytes/pixel, `width*height*3` bytes),
/// the reference being the source JPEG's own decoded pixels (lossless
/// transcode). The score's *direction* is described by `higher_is_better` on
/// [`recompress_jpeg_lossy_relative`] (zensim-A / cvvdp: higher better;
/// butteraugli: lower better).
pub type RelativeScorer<'a> = dyn Fn(&[u8], &[u8], u32, u32) -> f32 + 'a;

/// Decode a bare JXL codestream to tightly-packed RGB8 + dimensions.
fn decode_rgb8(codestream: &[u8]) -> Result<(Vec<u8>, u32, u32), At<JxlError>> {
    let out = decode(codestream, None, &[PixelDescriptor::RGB8])?;
    let w = out.info.width;
    let h = out.info.height;
    Ok((out.pixels.into_vec(), w, h))
}

/// Recompress a JPEG to the coarsest PreserveJxl setting still meeting a
/// **relative** quality target, returning the bare JXL codestream.
///
/// `target` is the score level in `scorer`'s units; `higher_is_better` gives the
/// metric direction. The loop bisects the coarsening scale (the bundled
/// deadzone + chroma policy is applied internally), encoding and decoding each
/// candidate and scoring it against the lossless-transcode reference. If the
/// target is unreachable by coarsening (too strict), the lossless transcode is
/// returned (the quality floor — never larger, never worse).
///
/// Cost: ~10–14 encode+decode+score probes. Requires the `jpeg-lossy` feature.
pub fn recompress_jpeg_lossy_relative(
    jpeg_bytes: &[u8],
    target: f32,
    higher_is_better: bool,
    scorer: &RelativeScorer<'_>,
    effort: u8,
) -> Result<Vec<u8>, At<JxlError>> {
    // Reference = the source's own decoded pixels (lossless transcode @ 1.0).
    let lossless = encode_coarsen(jpeg_bytes, 1.0, effort)?;
    let (ref_px, w, h) = decode_rgb8(&lossless)?;

    let meets = |score: f32| {
        if higher_is_better {
            score >= target
        } else {
            score <= target
        }
    };

    // Encode+decode+score one candidate scale.
    let probe = |scale: f32| -> Result<(Vec<u8>, f32), At<JxlError>> {
        let cs = encode_coarsen(jpeg_bytes, scale, effort)?;
        let (dist_px, dw, dh) = decode_rgb8(&cs)?;
        let score = if dw == w && dh == h && dist_px.len() == ref_px.len() {
            scorer(&ref_px, &dist_px, w, h)
        } else {
            // Dimensions must match the reference; treat a mismatch as worst.
            if higher_is_better { f32::MIN } else { f32::MAX }
        };
        Ok((cs, score))
    };

    // `best` is the coarsest codestream meeting the target; floor = lossless.
    let mut best = lossless;
    let mut a = 1.0f32; // lo: lossless always "meets" (ref vs ref)
    let mut b = 6.0f32;

    // Extend the upper bound until coarsening fails to meet the target.
    let mut bracketed = false;
    for _ in 0..6 {
        let (cs, sc) = probe(b)?;
        if meets(sc) {
            best = cs;
            a = b;
            b *= 1.8;
        } else {
            bracketed = true;
            break;
        }
    }
    if !bracketed {
        return Ok(best); // never failed up to the extended cap — coarsest wins
    }

    // Bisect [a (meets), b (fails)] for the coarsest meeting scale.
    for _ in 0..8 {
        let mid = 0.5 * (a + b);
        let (cs, sc) = probe(mid)?;
        if meets(sc) {
            best = cs;
            a = mid;
        } else {
            b = mid;
        }
    }
    Ok(best)
}

/// Thin wrapper: PreserveJxl coarsen at an explicit `scale` (no quality loop),
/// applying the bundled deadzone + chroma policy. `scale <= 1.0` is the lossless
/// transcode. Exposed so callers that already know the scale (or drive their own
/// loop) can reach the coefficient-domain path through the codec wrapper.
pub fn recompress_jpeg_coarsen(
    jpeg_bytes: &[u8],
    scale: f32,
    effort: u8,
) -> Result<Vec<u8>, At<JxlError>> {
    encode_coarsen(jpeg_bytes, scale, effort)
}

#[inline]
fn encode_coarsen(jpeg_bytes: &[u8], scale: f32, effort: u8) -> Result<Vec<u8>, At<JxlError>> {
    encode_jpeg_recompress_auto_codestream(jpeg_bytes, scale, effort)
        .map_err(|e| whereat::at!(JxlError::Encode(jxl_encoder::EncodeError::from(e))))
}
