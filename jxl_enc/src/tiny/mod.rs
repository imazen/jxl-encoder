// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! "Tiny" JPEG XL encoder - a simplified VarDCT encoder.
//!
//! This is a port of [libjxl-tiny](https://github.com/libjxl/libjxl-tiny), a simplified
//! JPEG XL encoder aimed at photographic images. It uses a subset of encoding tools:
//!
//! - Only DCT8, DCT8x16, and DCT16x8 transforms
//! - Only Huffman entropy coding (no ANS)
//! - Default zig-zag coefficient order
//! - Fixed context tree for DC coding
//! - No LZ77 backward references
//!
//! This provides a simpler encoding path that's easier to get correct while still
//! producing valid JPEG XL bitstreams.

mod ac_context;
mod ac_group;
mod cluster;
mod common;
mod context_tree;
mod dc_coding;
mod dct;
mod encoder;
mod entropy_code;
mod frame;
mod quant;
mod static_codes;
mod token;

pub use encoder::TinyEncoder;

#[cfg(test)]
mod tests;
