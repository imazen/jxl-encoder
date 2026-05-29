// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JPEG parsing and lossless reencoding into JPEG XL.
//!
//! This module provides a complete JPEG parser that extracts quantized DCT
//! coefficients, quantization/Huffman tables, and all metadata needed for
//! bit-exact JPEG reconstruction from a JPEG XL container.

mod data;
mod encode;
mod jbrd;
mod lossy;
mod parse;

pub use data::JpegData;
pub use encode::{
    encode_jpeg_to_jxl, encode_jpeg_to_jxl_container, encode_jpeg_to_jxl_container_with_effort,
    encode_jpeg_to_jxl_with_effort,
};
pub use lossy::{coarsen_coefficients, coarsen_coefficients_dz, coarsen_coefficients_planar};
pub use parse::{JpegError, read_jpeg};

/// PreserveJxl: coefficient-domain lossy JPEG → bare JXL codestream.
///
/// Parses `jpeg_bytes`, coarsens its quantized DCT coefficients in the DCT
/// domain by `scale` (> 1.0; near-uniform scale of the source's own quant
/// tables) with AC deadzone widening `dz` in `[0.0, 0.5]` (see
/// [`coarsen_coefficients_dz`]), then losslessly transcodes the coarsened
/// coefficients to a YCbCr JXL codestream (no JBRD). The output decodes to
/// the coarsened image; `scale <= 1.0` is identical to a lossless transcode.
///
/// The result is **guaranteed ≤ the lossless transcode size**: if coarsening
/// does not shrink the codestream (e.g. very gentle settings on an already-
/// sparse source), the lossless transcode is returned instead — never a
/// larger, quality-degraded file.
///
/// Requires the `jpeg-reencoding` feature.
pub fn encode_jpeg_recompress_codestream(
    jpeg_bytes: &[u8],
    scale: f32,
    dz: f32,
    effort: u8,
) -> Result<alloc::vec::Vec<u8>, crate::error::Error> {
    let jpeg = read_jpeg(jpeg_bytes)
        .map_err(|e| crate::error::Error::InvalidInput(alloc::format!("JPEG parse: {e:?}")))?;
    // Lossless transcode is the "do no harm" floor.
    let lossless = encode_jpeg_to_jxl_with_effort(&jpeg, effort)?;
    if !(scale > 1.0) {
        return Ok(lossless);
    }
    let mut coarsened = jpeg;
    coarsen_coefficients_dz(&mut coarsened, scale, dz);
    let lossy = encode_jpeg_to_jxl_with_effort(&coarsened, effort)?;
    // No-size-regression guard (RECOMPRESSION_COMPENDIUM §10.6): never ship a
    // larger, quality-degraded file than the lossless transcode.
    if lossy.len() < lossless.len() {
        Ok(lossy)
    } else {
        Ok(lossless)
    }
}

/// PreserveJxl with **separate luma/chroma** coarsening (see
/// [`coarsen_coefficients_planar`]). Same no-size-regression guard as
/// [`encode_jpeg_recompress_codestream`].
///
/// Requires the `jpeg-reencoding` feature.
pub fn encode_jpeg_recompress_planar_codestream(
    jpeg_bytes: &[u8],
    luma_scale: f32,
    luma_dz: f32,
    chroma_scale: f32,
    chroma_dz: f32,
    effort: u8,
) -> Result<alloc::vec::Vec<u8>, crate::error::Error> {
    let jpeg = read_jpeg(jpeg_bytes)
        .map_err(|e| crate::error::Error::InvalidInput(alloc::format!("JPEG parse: {e:?}")))?;
    let lossless = encode_jpeg_to_jxl_with_effort(&jpeg, effort)?;
    if !(luma_scale > 1.0 || chroma_scale > 1.0) {
        return Ok(lossless);
    }
    let mut coarsened = jpeg;
    coarsen_coefficients_planar(&mut coarsened, luma_scale, luma_dz, chroma_scale, chroma_dz);
    let lossy = encode_jpeg_to_jxl_with_effort(&coarsened, effort)?;
    if lossy.len() < lossless.len() {
        Ok(lossy)
    } else {
        Ok(lossless)
    }
}

// Re-export for tests that need direct JBRD access.
#[doc(hidden)]
pub use jbrd::encode_jbrd;

// Re-exports for the chunk-4 chroma-subsampling Sub420 path
// (`vardct::chroma_subsampling::encode_rgb8_sub420_via_jpeg_path`),
// which synthesises a [`JpegData`] from raw RGB pixels and hands it
// to [`encode_jpeg_to_jxl`]. The synthesised payload needs the
// component / quant-table / component-type types — these were
// already `pub` on their parent module; the re-export just makes
// them reachable through `crate::jpeg::*` without exposing the
// `data` submodule itself.
#[doc(hidden)]
pub use data::{JpegComponent, JpegComponentType, JpegQuantTable};

/// Fast check: do the supplied bytes look like a JPEG file?
///
/// Returns `true` if `bytes` starts with the JPEG SOI (Start Of Image)
/// marker `0xFF 0xD8` followed by another `0xFF` marker byte (any
/// well-formed JPEG follows SOI immediately with another marker, no
/// padding). Returns `false` for shorter inputs.
///
/// This is a lightweight signature sniff for routing decisions
/// (e.g., "did the caller hand us JPEG bytes, route to transcode?").
/// It does not validate the JPEG structure — use
/// [`read_jpeg`] for full parsing.
#[inline]
pub fn is_jpeg_signature(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}
