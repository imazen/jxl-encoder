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
pub mod vardct;

// Re-export new API as primary
pub use api::{
    AnimationFrame, AnimationParams, At, EncodeError, EncodeMode, EncodeRequest, EncodeResult,
    EncodeStats, EncoderMode, ImageMetadata, Limits, LosslessConfig, LosslessEncoder, LossyConfig,
    LossyEncoder, Lz77Method, PixelLayout, ProgressiveMode, Quality, ResultAtExt, Stop,
    Unstoppable, at,
};
pub use effort::EffortProfile;
pub use headers::color_encoding::{
    ColorEncoding, ColorSpace, Primaries, RenderingIntent, TransferFunction, WhitePoint,
};
pub use vardct::splines::{Spline, SplinePoint};

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
