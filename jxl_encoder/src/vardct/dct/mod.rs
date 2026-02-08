// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! DCT transforms ported from libjxl-tiny and libjxl.
//!
//! Implements the "Lowest Complexity Self Recursive Radix-2 DCT II/III
//! Algorithms" by Siriani M. Perera and Jianhua Liu.
//!
//! Also includes IDENTITY and DCT2X2 transforms from full libjxl
//! (enc_transforms-inl.h).

// Ported float constants from C++ - exact values are intentional for parity.
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(dead_code)]

mod constants;
mod forward;
mod forward_large;
mod inverse;
mod special;

pub use constants::*;
pub use forward::*;
pub use forward_large::*;
pub use inverse::*;
pub use special::*;

#[cfg(test)]
mod tests;
