// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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
