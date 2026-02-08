// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD-accelerated primitives for jxl_encoder.
//!
//! This crate wraps platform-specific SIMD intrinsics behind safe public functions.
//! The main encoder crate (`jxl_encoder`) maintains `#![forbid(unsafe_code)]` and
//! calls into these safe wrappers.
//!
//! Uses [archmage](https://docs.rs/archmage) for token-based SIMD dispatch
//! and [magetypes](https://docs.rs/magetypes) for cross-platform vector types.

#![no_std]
extern crate alloc;

mod dct8;
mod dct16;
mod dequant;
mod entropy;
mod epf;
mod gab;
mod gaborish5x5;
mod idct16;
mod quantize;
mod transpose;
mod xyb;

pub use dct8::dct_8x8;
pub use dct8::idct_8x8;
pub use dct16::dct_16x16;
pub use dequant::dequant_block_dct8;
pub use entropy::{EntropyCoeffResult, entropy_estimate_coeffs};
pub use epf::{epf_step1, epf_step2};
pub use gab::gab_smooth_channel;
pub use gaborish5x5::gaborish_5x5_channel;
pub use idct16::idct_16x16;
pub use quantize::quantize_block_dct8;
pub use transpose::transpose_8x8;
pub use xyb::linear_rgb_to_xyb_batch;
pub use xyb::xyb_to_linear_rgb_batch;
pub use xyb::xyb_to_linear_rgb_planar;
