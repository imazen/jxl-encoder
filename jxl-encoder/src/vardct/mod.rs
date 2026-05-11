// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! VarDCT (lossy) encoder for JPEG XL.
//!
//! Variable-DCT encoding transforms image blocks using DCT of various sizes,
//! quantizes coefficients with perceptual weighting, and entropy codes the result.
//!
//! Supports 19 of 27 DCT strategies (all that libjxl evaluates through effort 9),
//! Huffman or ANS entropy coding, custom coefficient ordering, LZ77 backward
//! references, adaptive quantization, chroma-from-luma, gaborish inverse,
//! noise synthesis, and butteraugli-guided rate control.

pub(crate) mod ac_context;
pub(crate) mod ac_group;
pub(crate) mod ac_strategy;
mod ac_strategy_search;
pub(crate) mod adaptive_quant;
mod afv;
mod bitstream;
mod block_extract;
#[cfg(feature = "butteraugli-loop")]
mod butteraugli_loop;
pub(crate) mod chroma_from_luma;
pub(crate) mod cluster;
mod coeff_order;
pub(crate) mod common;
pub(crate) mod context_tree;
pub(crate) mod dc_coding;
mod dc_tree_learn;
pub mod dct;
pub(crate) mod debug_log;
// libjxl enc_detect_dots.cc port (refs #19). Wired into encoder.rs at
// effort >= 7, distance >= 3.0; dots get promoted to a fresh
// PatchesData via from_dots() and travel through the regular patch
// subtract + decode pipeline. `ConnectedComponent::pixels` is
// retained for future tuning even though only the bounds field is
// consumed today.
#[allow(dead_code)]
pub(crate) mod dot_detection;
pub(crate) mod encoder;
pub(crate) mod entropy_code;
pub(crate) mod epf;
pub(crate) mod frame;
mod gaborish;
pub(crate) mod lf_frame;
pub(crate) mod noise;
pub(crate) mod patches;
#[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
pub(crate) mod precomputed;
#[cfg(feature = "rate-control")]
pub mod rate_control;
pub(crate) mod resampling;
pub(crate) mod simplify_invisible;
pub(crate) mod splines;
#[cfg(feature = "ssim2-loop")]
mod ssim2_loop;
#[cfg(feature = "rate-control")]
mod tile_distmap;
#[cfg(feature = "zensim-loop")]
mod zensim_loop;

pub(crate) mod quant;
pub(crate) mod quantize;
pub(crate) mod reconstruct;
mod static_codes;
mod transform;
pub(crate) mod xyb;

pub use encoder::{VarDctEncoder, VarDctOutput};
#[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
pub use precomputed::EncoderPrecomputed;
#[cfg(feature = "rate-control")]
pub use rate_control::RateControlConfig;

#[cfg(test)]
mod tests;
