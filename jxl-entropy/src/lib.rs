// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Entropy-coding primitives extracted from `jxl-encoder`: bit writer,
//! tokens, hybrid-uint, Huffman/ANS writers, context-map clustering,
//! LZ77 match finding. `jxl-encoder` re-exports this crate's modules
//! so `crate::entropy_coding::…` paths are unchanged; downstream users
//! can also depend on this crate directly.
//!
//! The useful surface is at the crate root: [`BitWriter`],
//! [`Token`]/[`Lz77Method`], [`HybridUintConfig`], [`EntropyCode`],
//! and the ANS/Huffman writers under [`entropy_coding`].

#![forbid(unsafe_code)]
extern crate alloc;

pub mod bit_writer;
pub mod budget;
pub mod clock;
pub mod debug_log;
pub mod entropy_coding;
pub mod error;
pub mod parallel;
pub mod profile_phases;
pub mod tiny_cluster;
pub mod trace;

// Root-level ergonomic surface for downstream codec work — the module
// tree is the full internal API; these are the pieces a downstream
// consumer most likely wants by name.
pub use bit_writer::BitWriter;
pub use budget::MemoryBudget;
pub use entropy_coding::encode::{EntropyCode, PrefixCode, UintConfigMethod};
pub use entropy_coding::hybrid_uint::HybridUintConfig;
pub use entropy_coding::lz77::{Lz77Method, Lz77Params};
pub use entropy_coding::token::{Lz77UintCoder, Token};
pub use error::{Error, Result};
