// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG XL encoder in pure Rust.
//!
//! This crate provides a complete JPEG XL encoder implementation, supporting
//! both lossless (modular) and lossy (VarDCT) encoding modes.

#![forbid(unsafe_code)]

pub mod api;
pub mod bit_writer;
pub mod color;
pub mod container;
pub mod entropy_coding;
pub mod error;
pub mod headers;
pub(crate) mod icc;
pub mod image;
#[cfg(feature = "jpeg-reencoding")]
pub mod jpeg;
pub mod modular;
pub mod trace;
pub mod vardct;

// Re-export new API as primary
pub use api::{
    At, EncodeError, EncodeRequest, ImageMetadata, Limits, LosslessConfig, LossyConfig, Lz77Method,
    PixelLayout, Quality, ResultAtExt, Stop, Unstoppable, at,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "api_tests.rs"]
mod api_tests;

#[cfg(test)]
pub mod test_helpers;

/// Group dimension in pixels (256x256 groups).
pub const GROUP_DIM: usize = 256;

/// DCT block dimension (8x8 blocks).
pub const BLOCK_DIM: usize = 8;

/// Size of a single DCT block (64 coefficients).
pub const BLOCK_SIZE: usize = BLOCK_DIM * BLOCK_DIM;

/// JXL signature bytes.
pub const JXL_SIGNATURE: [u8; 2] = [0xFF, 0x0A];
