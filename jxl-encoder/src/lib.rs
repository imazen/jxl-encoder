// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JPEG XL encoder in pure Rust.
//!
//! This crate provides a complete JPEG XL encoder implementation, supporting
//! both lossless (modular) and lossy (VarDCT) encoding modes.

#![cfg_attr(not(feature = "unsafe-performance"), forbid(unsafe_code))]
#![cfg_attr(feature = "unsafe-performance", deny(unsafe_code))]

extern crate alloc;

pub mod api;
pub mod bit_writer;
pub mod color;
pub mod container;
pub mod debug_rect;
// `effort` carries internal effort-derived knobs. Kept `pub` for
// backwards-compatibility with 0.3.0 (which re-exported `EffortProfile`
// at the crate root). The actual sweep / picker escape-hatch entry point
// (`LosslessConfig::with_effort_profile_override` / its lossy twin) is
// gated behind the `__expert` feature.
pub mod effort;
pub mod entropy_coding;
pub mod error;
#[allow(dead_code)] // Used by upcoming lf_frame module
pub(crate) mod f16;
pub mod headers;
pub(crate) mod icc;
pub mod image;
#[cfg(feature = "jpeg-reencoding")]
pub mod jpeg;
pub mod modular;
pub(crate) mod parallel;
pub mod trace;
pub mod validation;
#[cfg(test)]
mod validation_tests;
pub mod vardct;

#[cfg(feature = "convenience")]
pub mod convenience;

// Re-export new API as primary
pub use api::{
    AnimationFrame, AnimationParams, At, EncodeError, EncodeMode, EncodeRequest, EncodeResult,
    EncodeStats, EncoderMode, ImageMetadata, Limits, LosslessConfig, LosslessEncoder, LossyConfig,
    LossyEncoder, Lz77Method, PixelLayout, ProgressiveMode, Quality, ResultAtExt, Stop,
    Unstoppable, at, calibrated_jxl_quality, quality_to_distance,
};
// `EffortProfile` was re-exported at the crate root in 0.3.0; it is now an
// **internal** type that drives the encoder's effort-derived decisions.
// The public picker / sweep escape hatch is the segmented
// `LossyInternalParams` / `LosslessInternalParams` pair, applied via
// `LossyConfig::with_internal_params` / `LosslessConfig::with_internal_params`
// (gated behind `__expert`). `EntropyMulTable` remains reachable because
// `LossyInternalParams::entropy_mul_table` carries it. The `EffortProfile`
// re-export is `#[doc(hidden)]` to discourage new use; existing callers
// that still reference it keep working.
#[doc(hidden)]
pub use effort::EffortProfile;
pub use effort::EntropyMulTable;
#[cfg(feature = "__expert")]
pub use effort::{LosslessInternalParams, LossyInternalParams};
pub use headers::color_encoding::{
    CIExy, ColorEncoding, ColorSpace, CustomPrimaries, Primaries, RenderingIntent,
    TransferFunction, WhitePoint,
};
pub use validation::ValidationError;
pub use vardct::splines::{Spline, SplinePoint};

#[cfg(feature = "convenience")]
pub use convenience::{
    encode_bgra8, encode_bgra8_lossless, encode_gray8, encode_gray8_lossless, encode_rgb8,
    encode_rgb8_lossless, encode_rgba8, encode_rgba8_lossless,
};

/// Group dimension in pixels (256x256 groups).
pub const GROUP_DIM: usize = 256;

/// DCT block dimension (8x8 blocks).
pub const BLOCK_DIM: usize = 8;

/// Size of a single DCT block (64 coefficients).
pub const BLOCK_SIZE: usize = BLOCK_DIM * BLOCK_DIM;

/// JXL signature bytes.
pub const JXL_SIGNATURE: [u8; 2] = [0xFF, 0x0A];

/// Test path helpers for integration tests and examples.
///
/// Provides configurable paths via environment variables for corpus directories,
/// tool binaries, and output directories. Not part of the public API.
#[doc(hidden)]
pub mod test_helpers;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "api_tests.rs"]
mod api_tests;

#[cfg(all(test, feature = "__expert"))]
#[path = "effort_expert_tests.rs"]
mod effort_expert_tests;
