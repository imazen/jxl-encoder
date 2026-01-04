// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG XL encoder in pure Rust.
//!
//! This crate provides a complete JPEG XL encoder implementation, supporting
//! both lossless (modular) and lossy (VarDCT) encoding modes.

#![deny(unsafe_code)]

pub mod bit_writer;
pub mod color;
pub mod encoder;
pub mod entropy_coding;
pub mod error;
pub mod frame;
pub mod headers;
pub mod heuristics;
pub mod image;
pub mod modular;
pub mod trace;
pub mod vardct;

// Re-export main API
pub use encoder::{Encoder, EncoderOptions, encode_rgb8, encode_rgba8};

#[cfg(test)]
mod tests;

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
