// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Entropy coding for JPEG XL encoder.
//!
//! This module provides ANS (Asymmetric Numeral Systems) and Huffman
//! encoding implementations for compressing symbols in the JXL bitstream.

pub mod ans;
pub mod ans_decode;
pub mod cluster;
pub mod context_map;
pub mod encode;
pub mod encode_ans;
pub mod encode_huffman;
pub mod histogram;
pub mod huffman_tree;
pub mod hybrid_uint;
pub mod lz77;
pub mod token;

// #76 (0.4.0): the supported entropy surface is exactly the two
// config-visible enums below (both also re-exported at the crate root).
// Everything else — the ANS coder, histogram machinery, clustering,
// Huffman trees, the context map — is implementation detail, kept
// reachable crate-internally via `pub use`.
pub use ans::ANSHistogramStrategy;
pub use lz77::Lz77Method;

// In-crate consumers import from the submodules directly; the only
// re-export still routed through this module is the MTF helper.
pub use context_map::move_to_front_transform;
