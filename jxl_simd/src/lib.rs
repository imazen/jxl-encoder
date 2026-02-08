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

mod gab;
mod transpose;

pub use gab::gab_smooth_channel;
pub use transpose::transpose_8x8;
