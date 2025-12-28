// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Forward DCT transforms for JPEG XL encoding.
//!
//! This crate provides forward DCT (Discrete Cosine Transform) implementations
//! for various block sizes used in JPEG XL VarDCT encoding.

#![deny(unsafe_code)]

pub mod dct;

pub use dct::{dct2, dct4, dct8, dct16, dct32};
