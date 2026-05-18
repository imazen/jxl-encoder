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
mod parse;

pub use data::JpegData;
pub use encode::{encode_jpeg_to_jxl, encode_jpeg_to_jxl_container};
pub use parse::{JpegError, read_jpeg};

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
