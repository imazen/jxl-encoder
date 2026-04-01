// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Error types for the JPEG XL encoder.

use alloc::collections::TryReserveError;
#[cfg(feature = "std")]
use std::io;
use thiserror::Error;

/// Result type alias using the encoder's Error type.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Encoder error types.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    // BitWriter errors
    #[error("BitWriter buffer overflow: tried to write {attempted} bits, capacity is {capacity}")]
    BitWriterOverflow { attempted: usize, capacity: usize },

    #[error("Too many bits per write call: {0}, max is 56")]
    TooManyBitsPerCall(usize),

    #[error("BitWriter not byte-aligned: {0} bits written")]
    NotByteAligned(usize),

    // Image errors
    #[error("Invalid image dimensions: {0}x{1}")]
    InvalidImageDimensions(usize, usize),

    #[error("Image too large: {0}x{1}, max is {2}x{3}")]
    ImageTooLarge(usize, usize, usize, usize),

    #[error("Invalid bit depth: {0}")]
    InvalidBitDepth(u32),

    #[error("Invalid number of channels: {0}")]
    InvalidChannelCount(usize),

    #[error("Dimension overflow: {width}x{height}x{channels} exceeds usize")]
    DimensionOverflow {
        width: usize,
        height: usize,
        channels: usize,
    },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    // Entropy coding errors
    #[error("Invalid histogram: {0}")]
    InvalidHistogram(String),

    #[error("ANS encoding error: {0}")]
    AnsEncodingError(String),

    #[error("Bitstream error: {0}")]
    Bitstream(String),

    #[error("Too many unique symbols: found {found}, max {max} (minimal encoder limit)")]
    TooManySymbols { found: usize, max: usize },

    // Header errors
    #[error("Invalid color encoding")]
    InvalidColorEncoding,

    #[error("Invalid extra channel configuration")]
    InvalidExtraChannel,

    #[error("Invalid frame header")]
    InvalidFrameHeader,

    // Transform errors
    #[error("Invalid DCT size: {0}")]
    InvalidDctSize(usize),

    #[error("Transform coefficient overflow")]
    TransformOverflow,

    // General errors
    #[error("Out of memory")]
    OutOfMemory(#[from] TryReserveError),

    #[cfg(feature = "std")]
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),

    #[error("Encoding cancelled")]
    Cancelled,

    #[error("Feature not yet implemented: {0}")]
    NotImplemented(String),
}
