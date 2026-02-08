// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! VarDCT (lossy) encoder for JPEG XL.
//!
//! Variable-DCT encoding transforms image blocks using DCT of various sizes,
//! quantizes coefficients with perceptual weighting, and entropy codes the result.
//!
//! Supports 19 of 27 DCT strategies (all that libjxl evaluates through effort 9),
//! Huffman or ANS entropy coding, custom coefficient ordering, LZ77 backward
//! references, adaptive quantization, chroma-from-luma, gaborish inverse,
//! noise synthesis, and butteraugli-guided rate control.

mod ac_context;
mod ac_group;
mod ac_strategy;
mod ac_strategy_search;
mod adaptive_quant;
mod afv;
mod bitstream;
mod block_extract;
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
mod quantize;
#[allow(dead_code)] // Functions used in upcoming phases (EPF, butteraugli)
pub(crate) mod reconstruct;
mod static_codes;
mod transform;
mod xyb;

pub use encoder::VarDctEncoder;
#[cfg(feature = "rate-control")]
pub use precomputed::EncoderPrecomputed;
#[cfg(feature = "rate-control")]
pub use rate_control::RateControlConfig;

#[cfg(test)]
mod tests;
