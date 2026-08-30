// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Color space transforms for JPEG XL encoding.
//!
//! This module provides forward color transforms for VarDCT (lossy) encoding,
//! including sRGB to linear and linear RGB to XYB.

pub mod xyb;

// #76: no mod-level re-exports — consumers import from `xyb` directly
// (externally via the doc-hidden `__test_exports::xyb` seam).
