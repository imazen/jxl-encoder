// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Entropy coding for JPEG XL encoder.
//!
//! This module provides ANS (Asymmetric Numeral Systems) and Huffman
//! encoding implementations for compressing symbols in the JXL bitstream.

pub(crate) mod ans;
pub(crate) mod ans_decode;
pub(crate) mod cluster;
pub(crate) mod context_map;
pub(crate) mod encode;
pub(crate) mod encode_ans;
mod encode_huffman;
pub(crate) mod histogram;
pub(crate) mod huffman_tree;
pub(crate) mod hybrid_uint;
pub(crate) mod lz77;
pub(crate) mod token;

// #76 (0.4.0): the supported entropy surface is exactly the two
// config-visible enums below (both also re-exported at the crate root).
// Everything else — the ANS coder, histogram machinery, clustering,
// Huffman trees, the context map — is implementation detail, kept
// reachable crate-internally via `pub(crate) use`.
pub use ans::ANSHistogramStrategy;
pub use lz77::Lz77Method;

// In-crate consumers import from the submodules directly; the only
// re-export still routed through this module is the MTF helper.
pub(crate) use context_map::move_to_front_transform;
