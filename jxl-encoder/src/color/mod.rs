// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Color space transforms for JPEG XL encoding.
//!
//! This module provides forward color transforms for VarDCT (lossy) encoding,
//! including sRGB to linear and linear RGB to XYB.

pub mod xyb;

pub use xyb::{linear_rgb_to_xyb, srgb_to_linear, srgb_to_xyb};
