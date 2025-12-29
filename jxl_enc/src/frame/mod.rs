// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Frame encoding for JPEG XL.
//!
//! This module handles the assembly of complete JXL frames, including
//! headers, modular data, and group encoding.

mod frame_encoder;

pub use frame_encoder::{FrameEncoder, FrameEncoderOptions};
