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
