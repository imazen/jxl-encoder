//! Color space transforms for JPEG XL encoding.
//!
//! This module provides forward color transforms for VarDCT (lossy) encoding,
//! including sRGB to linear and linear RGB to XYB.

pub mod xyb;

pub use xyb::{linear_rgb_to_xyb, srgb_to_linear, srgb_to_xyb};
