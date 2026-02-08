// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG parsing and lossless reencoding into JPEG XL.
//!
//! This module provides a complete JPEG parser that extracts quantized DCT
//! coefficients, quantization/Huffman tables, and all metadata needed for
//! bit-exact JPEG reconstruction from a JPEG XL container.

mod data;
mod encode;
mod jbrd;
mod parse;

pub use data::*;
pub use encode::{encode_jpeg_to_jxl, encode_jpeg_to_jxl_container};
pub use jbrd::{encode_jbrd, extract_exif, extract_xmp};
pub use parse::read_jpeg;
