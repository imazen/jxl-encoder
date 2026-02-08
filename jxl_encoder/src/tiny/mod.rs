// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! "Tiny" JPEG XL encoder - a simplified VarDCT encoder.
//!
//! This started as a port of [libjxl-tiny](https://github.com/libjxl/libjxl-tiny), a simplified
//! JPEG XL encoder aimed at photographic images. It now includes additional features:
//!
//! - DCT8, DCT4x4, DCT4x8, DCT8x4, DCT8x16, DCT16x8, DCT16x16, DCT32x32 transforms
//! - Huffman or ANS entropy coding (`use_ans` flag)
//! - Custom or default zig-zag coefficient order (`custom_orders` flag)
//! - Fixed context tree for DC coding
//! - LZ77 backward references with RLE or hash chain matching (`enable_lz77`, `lz77_method`)
//!
//! This provides a simpler encoding path that's easier to get correct while still
//! producing valid JPEG XL bitstreams.

mod ac_context;
mod ac_group;
mod ac_strategy;
mod adaptive_quant;
mod afv;
mod bitstream;
mod chroma_from_luma;
pub(crate) mod cluster;
mod coeff_order;
pub(crate) mod common;
mod context_tree;
mod dc_coding;
mod dc_tree_learn;
pub mod dct;
pub mod debug_log;
mod encoder;
#[allow(dead_code)] // Used in upcoming EPF sharpness selection
pub(crate) mod epf;
mod frame;
mod gaborish;
pub(crate) mod noise;
#[cfg(feature = "rate-control")]
mod precomputed;
#[cfg(feature = "rate-control")]
pub mod rate_control;
#[cfg(feature = "rate-control")]
mod tile_distmap;

mod quant;
#[allow(dead_code)] // Functions used in upcoming phases (EPF, butteraugli)
pub(crate) mod reconstruct;
mod static_codes;
mod transform;

pub use encoder::TinyEncoder;
#[cfg(feature = "rate-control")]
pub use precomputed::EncoderPrecomputed;
#[cfg(feature = "rate-control")]
pub use rate_control::RateControlConfig;

#[cfg(test)]
mod tests;
