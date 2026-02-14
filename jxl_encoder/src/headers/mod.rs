// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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
pub use frame_header::{BlendMode, Encoding, FrameHeader, FrameType};
