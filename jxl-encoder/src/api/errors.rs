// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Public encode error type (`EncodeError`), its `From` conversions, the
//! `Result<T>` alias, and the `at_from` helper.

use super::*;

/// Encode error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum EncodeError {
    /// Input validation failed (wrong buffer size, zero dimensions, etc.).
    InvalidInput { message: String },
    /// Config validation failed (contradictory options, out-of-range values).
    InvalidConfig { message: String },
    /// Pixel layout not supported for this config/mode.
    UnsupportedPixelLayout(PixelLayout),
    /// A configured limit was exceeded.
    LimitExceeded { message: String },
    /// Encoding was cancelled via [`Stop`].
    Cancelled,
    /// Allocation failure.
    Oom(std::collections::TryReserveError),
    /// I/O error.
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// Internal encoder error (should not happen — file a bug).
    Internal { message: String },
    /// JPEG input could not be parsed for lossless transcoding.
    ///
    /// Raised by [`LosslessConfig::encode_jpeg_transcode`] and the bare
    /// [`LosslessConfig::encode_jpeg_transcode_codestream`] variant when
    /// the supplied bytes are not a valid baseline-sequential JPEG, when
    /// the JPEG uses a feature unsupported by the transcoder (e.g.,
    /// arithmetic coding), or when the embedded decoder fails to extract
    /// quantized DCT coefficients. The message is plain-text and safe to
    /// surface to end users.
    ///
    /// Only constructible when the `jpeg-reencoding` cargo feature is
    /// enabled.
    JpegParse { message: String },
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { message } => write!(f, "invalid input: {message}"),
            Self::InvalidConfig { message } => write!(f, "invalid config: {message}"),
            Self::UnsupportedPixelLayout(layout) => {
                write!(f, "unsupported pixel layout: {layout:?}")
            }
            Self::LimitExceeded { message } => write!(f, "limit exceeded: {message}"),
            Self::Cancelled => write!(f, "encoding cancelled"),
            Self::Oom(e) => write!(f, "out of memory: {e}"),
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Internal { message } => write!(f, "internal error: {message}"),
            Self::JpegParse { message } => write!(f, "JPEG parse error: {message}"),
        }
    }
}

impl core::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Oom(e) => Some(e),
            #[cfg(feature = "std")]
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::error::Error> for EncodeError {
    fn from(e: crate::error::Error) -> Self {
        match e {
            crate::error::Error::InvalidImageDimensions(w, h) => Self::InvalidInput {
                message: format!("invalid dimensions: {w}x{h}"),
            },
            crate::error::Error::ImageTooLarge(w, h, mw, mh) => Self::LimitExceeded {
                message: format!("image {w}x{h} exceeds max {mw}x{mh}"),
            },
            crate::error::Error::DimensionOverflow {
                width,
                height,
                channels,
            } => Self::InvalidInput {
                message: format!("dimension overflow: {width}x{height}x{channels} exceeds usize"),
            },
            crate::error::Error::InvalidInput(msg) => Self::InvalidInput { message: msg },
            crate::error::Error::AllocationLimit {
                requested,
                used,
                cap,
            } => Self::LimitExceeded {
                message: format!(
                    "memory budget exceeded: requested {requested} bytes on top of {used} (cap {cap})"
                ),
            },
            crate::error::Error::OutOfMemory(e) => Self::Oom(e),
            #[cfg(feature = "std")]
            crate::error::Error::IoError(e) => Self::Io(e),
            crate::error::Error::Cancelled => Self::Cancelled,
            other => Self::Internal {
                message: format!("{other}"),
            },
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for EncodeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<crate::validation::ValidationError> for EncodeError {
    fn from(e: crate::validation::ValidationError) -> Self {
        EncodeError::InvalidInput {
            message: format!("{e}"),
        }
    }
}

impl From<enough::StopReason> for EncodeError {
    fn from(_: enough::StopReason) -> Self {
        Self::Cancelled
    }
}

#[cfg(feature = "jpeg-reencoding")]
impl From<crate::jpeg::JpegError> for EncodeError {
    fn from(e: crate::jpeg::JpegError) -> Self {
        match e {
            // Cooperative cancellation surfaces as the dedicated variant, not
            // a parse error (mirrors the VarDCT path's `Error::Cancelled`).
            crate::jpeg::JpegError::Cancelled => Self::Cancelled,
            // A coefficient-buffer reservation that overran the memory budget
            // surfaces as a limit error, not a parse error (mirrors the pixel
            // path's `AllocationLimit` → `LimitExceeded` mapping).
            crate::jpeg::JpegError::ResourceLimit(message) => Self::LimitExceeded { message },
            other => Self::JpegParse {
                message: format!("{other}"),
            },
        }
    }
}

/// Result type for encoding operations.
///
/// Errors carry location traces via [`whereat::At`] for lightweight
/// production-safe error tracking without debuginfo or backtraces.
pub type Result<T> = core::result::Result<T, At<EncodeError>>;

/// Convert any error that maps to [`EncodeError`] into an [`At<EncodeError>`],
/// capturing **this call site** as the origin frame (with crate info, so the
/// trace carries a clickable GitHub link).
///
/// This is the bridge for the `?` boundary where an internal
/// [`crate::error::Result`] (carrying the bare [`crate::error::Error`]) crosses
/// into the public [`Result`] alias: `inner_call().map_err(at_from)?`. Without
/// it, `?` cannot apply (`From<Error>` exists for `EncodeError` but not for
/// `At<EncodeError>`), and the location would otherwise be stamped at the
/// outermost API boundary rather than at the conversion site near the failure.
///
/// `#[track_caller]` makes the captured location the *caller's*, not this
/// helper's.
#[track_caller]
#[inline]
pub(crate) fn at_from<E: Into<EncodeError>>(e: E) -> At<EncodeError> {
    At::wrap(e.into())
        .set_crate_info(crate::at_crate_info())
        .at()
}

// ── Limit aliases ──────────────────────────────────────────────────────────
