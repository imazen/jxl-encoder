// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! DCT transforms for JPEG XL encoding.
//!
//! This crate provides forward and inverse DCT (Discrete Cosine Transform)
//! implementations for various block sizes used in JPEG XL VarDCT encoding.
//!
//! ## Forward DCT (DCT-II)
//! - Square: `dct4`, `dct8`, `dct16`, `dct32`
//! - Rectangular: `dct_4x8`, `dct_8x4`, `dct_16x8`, `dct_8x16`
//!
//! ## Inverse DCT (DCT-III)
//! - Square: `idct4`, `idct8`, `idct16`
//! - Rectangular: `idct_4x8`, `idct_8x4`, `idct_16x8`, `idct_8x16`

#![deny(unsafe_code)]

pub mod dct;

// Forward DCT exports
pub use dct::{dct_4x8, dct_8x4, dct_8x16, dct_16x8};
pub use dct::{dct2, dct4, dct8, dct16, dct32};

// Inverse DCT exports
pub use dct::{idct_4x8, idct_8x4, idct_8x16, idct_16x8};
pub use dct::{idct4, idct8, idct16};
