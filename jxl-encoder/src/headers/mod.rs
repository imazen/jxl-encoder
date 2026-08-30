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

// #76 (0.4.0): this module is `pub(crate)`; these re-exports are the
// crate-internal convenience paths still in use (`modular::frame`,
// `tests`). The public routes are the crate-root re-exports of the
// `color_encoding` types and `api`'s `BlendMode`.
pub use color_encoding::ColorEncoding;
pub use file_header::FileHeader;
