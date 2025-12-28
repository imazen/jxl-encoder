// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Header structures and serialization for JPEG XL.
//!
//! This module contains the data structures for JXL file and frame headers,
//! along with methods to serialize them to the bitstream.

pub mod color_encoding;
pub mod extra_channels;
pub mod file_header;
pub mod frame_header;
pub mod size;

pub use color_encoding::ColorEncoding;
pub use extra_channels::ExtraChannelInfo;
pub use file_header::FileHeader;
pub use frame_header::FrameHeader;
