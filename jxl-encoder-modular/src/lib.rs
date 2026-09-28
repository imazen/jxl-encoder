// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Modular (lossless) encoder half of `jxl-encoder`: channel/palette
//! transforms, MA tree learning, sectioned local trees, headers,
//! effort schedule. `jxl-encoder` re-exports these modules so
//! `crate::modular::…` paths are unchanged; for the public API use
//! `jxl-encoder` — this crate is mostly internal plumbing.
//!
//! The api-shape pieces are reachable at `jxl_encoder_modular::api::*`
//! (SectionedTrees, PixelLayout, EncoderMode, strategy enums) and the
//! effort/config types at `effort::*` / `api_bits::*`.

#![forbid(unsafe_code)]
extern crate alloc;

// Re-export the leaf crate's modules so internal `crate::X` paths keep
// working exactly as they did pre-split.
pub use jxl_entropy::{
    bit_writer, budget, clock, debug_log, entropy_coding, error, parallel, profile_phases,
    tiny_cluster, trace,
};
#[allow(unused_imports)]
pub use jxl_entropy::{
    debug_eprintln, debug_log as debug_log_macro, debug_log_flush, trace_section, trace_write,
};

#[allow(unused_imports)]
pub use jxl_entropy::profile_time;
pub mod api_bits;
pub mod api {
    //! Shim presenting the extracted api-side config types at
    //! `crate::api::*` — moved code keeps `crate::api::SectionedTrees`
    //! etc. unchanged.
    pub use crate::api_bits::*;
    pub use crate::gate_registry::CustomEncoderImprovements;
    pub use crate::pixel_layout::*;
    pub use crate::strategy::*;
}
pub mod common;
pub mod consts;
pub mod dot_detection;

pub mod debug_rect;
pub mod effort;
pub mod f16;
pub mod gate_registry;
pub mod headers;
pub mod heuristics;
pub mod modular;
pub mod patches;
pub mod pixel_layout;
pub mod strategy;
pub mod validation;

// Ergonomic root re-exports — the names a downstream consumer most
// likely reaches for first.
pub use api_bits::SectionedTrees;
pub use effort::EffortProfile;
pub use pixel_layout::PixelLayout;
pub use strategy::EncoderMode;
