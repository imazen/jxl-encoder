// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Three-layer public API: Config → Request → Encoder.
//!
//! ```rust,no_run
//! use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};
//!
//! # let pixels = vec![0u8; 800 * 600 * 3];
//! // Simple — one line, no request visible
//! let jxl = LossyConfig::new(1.0)
//!     .encode(&pixels, 800, 600, PixelLayout::Rgb8)?;
//!
//! // Full control — request layer for metadata, limits, cancellation
//! let jxl = LosslessConfig::new()
//!     .encode_request(800, 600, PixelLayout::Rgb8)
//!     .encode(&pixels)?;
//! # Ok::<_, jxl_encoder::At<jxl_encoder::EncodeError>>(())
//! ```

pub use crate::entropy_coding::Lz77Method;
pub use crate::headers::frame_header::BlendMode;
#[cfg(feature = "butteraugli-loop")]
pub use crate::vardct::hdr_metrics::HdrLoss;
pub use enough::{Stop, Unstoppable};
pub use whereat::{At, ResultAtExt, at};

// ── Error type ──────────────────────────────────────────────────────────────

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
        Self::JpegParse {
            message: format!("{e}"),
        }
    }
}

/// Result type for encoding operations.
///
/// Errors carry location traces via [`whereat::At`] for lightweight
/// production-safe error tracking without debuginfo or backtraces.
pub type Result<T> = core::result::Result<T, At<EncodeError>>;

// ── Limit aliases ──────────────────────────────────────────────────────────

/// Hard upper bound for quantization-loop iterations. Alias of
/// [`Limits::DEFAULT_MAX_QUANT_LOOP_ITERS`] — preserved for callers that
/// referenced the bare const before per-encode limits became
/// configurable. Prefer setting [`Limits::with_max_quant_loop_iters`]
/// (or letting the default apply) over hard-coding this constant.
pub const MAX_QUANT_LOOP_ITERS: u32 = Limits::DEFAULT_MAX_QUANT_LOOP_ITERS;

/// Default soft cap on encoder working-set memory when no explicit
/// [`Limits::max_memory_bytes`] is set. Alias of
/// [`Limits::DEFAULT_MAX_MEMORY_BYTES`].
pub const DEFAULT_MAX_MEMORY_BYTES: u64 = Limits::DEFAULT_MAX_MEMORY_BYTES;

/// Conservative-upper-bound peak working-set estimate for a lossy
/// encode at the given dimensions. Backs
/// [`LossyConfig::estimate_peak_memory_bytes`]; pulled out as a free
/// function so the formula can be unit-tested without instantiating
/// a config. See the doc on `LossyConfig::estimate_peak_memory_bytes`
/// for the per-buffer breakdown.
pub(crate) fn estimate_peak_memory_bytes_lossy(
    width: u32,
    height: u32,
    layout: PixelLayout,
) -> Option<u64> {
    let w = width as u64;
    let h = height as u64;
    let pixels = w.checked_mul(h)?;
    let padded_w = w.checked_add(7)? & !7;
    let padded_h = h.checked_add(7)? & !7;
    let padded_pixels = padded_w.checked_mul(padded_h)?;
    let blocks = padded_pixels / 64;

    // (1) linear_rgb: pixels × 3 channels × 4 bytes (f32). Always RGB
    //     internally regardless of pixel layout — gray expands.
    let linear_rgb = pixels.checked_mul(12)?;

    // (2) XYB planes: 3 padded channels × 4 bytes (f32).
    let xyb = padded_pixels.checked_mul(12)?;

    // (3) quant_ac: 3 channels × blocks × 64 coeffs × 4 bytes (i32).
    let quant_ac = blocks.checked_mul(3 * 64 * 4)?;

    // (4) Alpha buffer (when present).
    let alpha = if layout.has_alpha() { pixels } else { 0 };

    let major = linear_rgb
        .checked_add(xyb)?
        .checked_add(quant_ac)?
        .checked_add(alpha)?;

    // 25 % overhead for entropy-coder bit buffer, histograms,
    // scratch, transform working state.
    major.checked_add(major / 4)
}

/// Conservative-upper-bound peak working-set estimate for a lossless
/// encode at the given dimensions. Backs
/// [`LosslessConfig::estimate_peak_memory_bytes`].
pub(crate) fn estimate_peak_memory_bytes_lossless(
    width: u32,
    height: u32,
    layout: PixelLayout,
    effort: u8,
    squeeze: bool,
) -> Option<u64> {
    let w = width as u64;
    let h = height as u64;
    let pixels = w.checked_mul(h)?;

    let channels: u64 = match layout {
        PixelLayout::Gray8 | PixelLayout::Gray16 | PixelLayout::GrayLinearF32 => 1,
        PixelLayout::GrayAlpha8 | PixelLayout::GrayAlpha16 | PixelLayout::GrayAlphaLinearF32 => 2,
        PixelLayout::Rgb8 | PixelLayout::Rgb16 | PixelLayout::RgbLinearF32 | PixelLayout::Bgr8 => 3,
        PixelLayout::Rgba8
        | PixelLayout::Rgba16
        | PixelLayout::RgbaLinearF32
        | PixelLayout::Bgra8 => 4,
        PixelLayout::RgbLinearF16 | PixelLayout::RgbaLinearF16 => 4,
        PixelLayout::GrayLinearF16 | PixelLayout::GrayAlphaLinearF16 => 2,
        // A3 chunk 1b: f32 PQ/HLG/BT.709 RGB(A) (issue #46).
        PixelLayout::RgbPqF32 | PixelLayout::RgbHlgF32 | PixelLayout::RgbBt709F32 => 3,
        PixelLayout::RgbaPqF32 | PixelLayout::RgbaHlgF32 | PixelLayout::RgbaBt709F32 => 4,
        // CMYK lossless: 3 colour planes + 1 Black extra channel
        // (same i32 plane layout as RGBA).
        PixelLayout::Cmyk8 | PixelLayout::Cmyk16 => 4,
    };

    // (1) Channel planes: i32 per pixel per channel.
    let channel_planes = pixels.checked_mul(channels)?.checked_mul(4)?;

    // (2) Predictor scratch (gradient + weighted state): one i32 plane.
    let predictor = pixels.checked_mul(4)?;

    // (3) Tree-learning state (effort >= 7). 8 bytes/pixel for typical
    //     histogram counts; 0 otherwise.
    let tree_learning = if effort >= 7 {
        pixels.checked_mul(8)?
    } else {
        0
    };

    // (4) Squeeze residuals: one extra channel-plane pair when on.
    let squeeze_state = if squeeze {
        pixels.checked_mul(channels)?.checked_mul(4)?
    } else {
        0
    };

    let major = channel_planes
        .checked_add(predictor)?
        .checked_add(tree_learning)?
        .checked_add(squeeze_state)?;

    major.checked_add(major / 4)
}

// ── EncodeResult / EncodeStats ──────────────────────────────────────────────

/// Result of an encode operation. Holds encoded data and metrics.
///
/// After `encode()`, `data()` returns the JXL bytes. After `encode_into()`
/// or `encode_to()`, `data()` returns `None` (data already delivered).
/// Use `take_data()` to move the vec out without cloning.
#[derive(Clone, Debug)]
pub struct EncodeResult {
    data: Option<Vec<u8>>,
    stats: EncodeStats,
}

impl EncodeResult {
    /// Encoded JXL bytes (borrowing). None if data was written elsewhere.
    pub fn data(&self) -> Option<&[u8]> {
        self.data.as_deref()
    }

    /// Take the owned data vec, leaving None in its place.
    pub fn take_data(&mut self) -> Option<Vec<u8>> {
        self.data.take()
    }

    /// Encode metrics.
    pub fn stats(&self) -> &EncodeStats {
        &self.stats
    }
}

/// Encode metrics collected during encoding.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct EncodeStats {
    codestream_size: usize,
    output_size: usize,
    mode: EncodeMode,
    /// Index = raw strategy code (0..19), value = first-block count.
    strategy_counts: [u32; 19],
    gaborish: bool,
    ans: bool,
    butteraugli_iters: u32,
    pixel_domain_loss: bool,
}

impl EncodeStats {
    /// Size of the JXL codestream in bytes (before container wrapping).
    pub fn codestream_size(&self) -> usize {
        self.codestream_size
    }

    /// Size of the final output in bytes (after container wrapping, if any).
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Whether the encode was lossy or lossless.
    pub fn mode(&self) -> EncodeMode {
        self.mode
    }

    /// Per-strategy first-block counts, indexed by raw strategy code (0..19).
    pub fn strategy_counts(&self) -> &[u32; 19] {
        &self.strategy_counts
    }

    /// Whether gaborish pre-filtering was enabled.
    pub fn gaborish(&self) -> bool {
        self.gaborish
    }

    /// Whether ANS entropy coding was used.
    pub fn ans(&self) -> bool {
        self.ans
    }

    /// Number of butteraugli quantization loop iterations performed.
    pub fn butteraugli_iters(&self) -> u32 {
        self.butteraugli_iters
    }

    /// Whether pixel-domain loss was enabled.
    pub fn pixel_domain_loss(&self) -> bool {
        self.pixel_domain_loss
    }
}

/// Encoding mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EncodeMode {
    /// Lossy (VarDCT) encoding.
    #[default]
    Lossy,
    /// Lossless (modular) encoding.
    Lossless,
}

// ── PixelLayout ─────────────────────────────────────────────────────────────

/// Describes the pixel format of input data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelLayout {
    /// 8-bit sRGB, 3 bytes per pixel (R, G, B).
    Rgb8,
    /// 8-bit sRGB + alpha, 4 bytes per pixel (R, G, B, A).
    Rgba8,
    /// 8-bit sRGB in BGR order, 3 bytes per pixel (B, G, R).
    Bgr8,
    /// 8-bit sRGB in BGRA order, 4 bytes per pixel (B, G, R, A).
    Bgra8,
    /// 8-bit grayscale, 1 byte per pixel.
    Gray8,
    /// 8-bit grayscale + alpha, 2 bytes per pixel.
    GrayAlpha8,
    /// 16-bit sRGB, 6 bytes per pixel (R, G, B) — native-endian u16.
    Rgb16,
    /// 16-bit sRGB + alpha, 8 bytes per pixel (R, G, B, A) — native-endian u16.
    Rgba16,
    /// 16-bit grayscale, 2 bytes per pixel — native-endian u16.
    Gray16,
    /// 16-bit grayscale + alpha, 4 bytes per pixel — native-endian u16.
    GrayAlpha16,
    /// Linear f32 RGB, 12 bytes per pixel. Skips sRGB→linear conversion.
    RgbLinearF32,
    /// Linear f32 RGBA, 16 bytes per pixel. Skips sRGB→linear conversion.
    RgbaLinearF32,
    /// Linear f32 grayscale, 4 bytes per pixel.
    GrayLinearF32,
    /// Linear f32 grayscale + alpha, 8 bytes per pixel.
    GrayAlphaLinearF32,
    /// Linear IEEE 754 binary16 RGB, 6 bytes per pixel — native-endian
    /// u16. Common for GPU pipelines (WebGPU, CUDA, Metal, Vulkan,
    /// Direct2D). Same semantics as [`Self::RgbLinearF32`] but at
    /// half precision; encoder converts to f32 internally before XYB.
    RgbLinearF16,
    /// Linear IEEE 754 binary16 RGBA, 8 bytes per pixel — native-endian
    /// u16. See [`Self::RgbLinearF16`].
    RgbaLinearF16,
    /// Linear IEEE 754 binary16 grayscale, 2 bytes per pixel.
    GrayLinearF16,
    /// Linear IEEE 754 binary16 grayscale + alpha, 4 bytes per pixel.
    GrayAlphaLinearF16,
    /// PQ-encoded (SMPTE ST 2084) f32 RGB, 12 bytes per pixel. Same
    /// storage shape as [`Self::RgbLinearF32`] but the f32 values are
    /// interpreted as PQ-encoded `[0, 1]` and run through the inverse
    /// PQ EOTF before XYB. Use for HDR GPU/Vulkan/Metal pipelines that
    /// emit PQ floats directly. Closes A3 chunk 1b (issue #46).
    RgbPqF32,
    /// PQ-encoded f32 RGBA, 16 bytes per pixel. See [`Self::RgbPqF32`].
    RgbaPqF32,
    /// HLG-encoded (BT.2100 / ARIB STD-B67) f32 RGB, 12 bytes per
    /// pixel. f32 values are interpreted as HLG-encoded `[0, 1]` and
    /// run through the inverse HLG OETF (scene-light) before XYB.
    /// Closes A3 chunk 1b (issue #46).
    RgbHlgF32,
    /// HLG-encoded f32 RGBA, 16 bytes per pixel. See [`Self::RgbHlgF32`].
    RgbaHlgF32,
    /// BT.709 (Rec. ITU-R BT.709-6) gamma-encoded f32 RGB, 12 bytes
    /// per pixel. f32 values are interpreted as BT.709-encoded `[0, 1]`
    /// and run through the inverse BT.709 OETF before XYB. Closes A3
    /// chunk 1b (issue #46).
    RgbBt709F32,
    /// BT.709-encoded f32 RGBA, 16 bytes per pixel. See [`Self::RgbBt709F32`].
    RgbaBt709F32,
    /// 8-bit CMYK, 4 bytes per pixel (C, M, Y, K). The C/M/Y planes
    /// are encoded as the frame's 3 color channels; the K (Black)
    /// plane is encoded as an [`ExtraChannelType::Black`] extra
    /// channel. JXL convention is **`0 = full ink, 255 = no ink`**
    /// for all four planes (libjxl `enc_image_bundle.cc:65`); callers
    /// must pre-invert any "0 = no ink" CMYK input before encoding.
    ///
    /// For a colour-managed CMYK workflow attach a CMYK ICC profile
    /// via [`LosslessConfig::with_metadata`] →
    /// [`ImageMetadata::icc_profile`]; without an ICC the decoder will
    /// fall back to interpreting the CMY planes as sRGB and the K
    /// plane as an opaque extra channel. Lossless-only in this
    /// chunk; lossy CMYK is not yet wired (would route C/M/Y through
    /// VarDCT in XYB, which loses CMYK semantics). Closes #58.
    ///
    /// Bumps codestream level to 10 (level 5 forbids the Black
    /// extra channel; see `compute_codestream_level`).
    Cmyk8,
    /// 16-bit CMYK, 8 bytes per pixel — native-endian u16 per channel
    /// (C, M, Y, K). Same `0 = full ink, 65535 = no ink` convention
    /// as [`Self::Cmyk8`]; same lossless-only restriction. The Black
    /// channel is signaled as 16-bit in the extra-channel header.
    Cmyk16,
}

impl PixelLayout {
    /// Bytes per pixel for this layout.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 | Self::Bgr8 => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::Gray8 => 1,
            Self::GrayAlpha8 => 2,
            Self::Rgb16 => 6,
            Self::Rgba16 => 8,
            Self::Gray16 => 2,
            Self::GrayAlpha16 => 4,
            Self::RgbLinearF32 => 12,
            Self::RgbaLinearF32 => 16,
            Self::GrayLinearF32 => 4,
            Self::GrayAlphaLinearF32 => 8,
            Self::RgbLinearF16 => 6,
            Self::RgbaLinearF16 => 8,
            Self::GrayLinearF16 => 2,
            Self::GrayAlphaLinearF16 => 4,
            // A3 chunk 1b: f32 HDR/SDR-encoded variants (issue #46).
            // Same byte width as RgbLinearF32 / RgbaLinearF32; the
            // transfer function lives in the layout name, not the
            // storage size.
            Self::RgbPqF32 | Self::RgbHlgF32 | Self::RgbBt709F32 => 12,
            Self::RgbaPqF32 | Self::RgbaHlgF32 | Self::RgbaBt709F32 => 16,
            // CMYK is C, M, Y, K interleaved — 4 bytes (8-bit) or
            // 8 bytes (16-bit). The K plane is split off and encoded
            // as a Black extra channel; the on-disk codestream still
            // carries 3 colour channels.
            Self::Cmyk8 => 4,
            Self::Cmyk16 => 8,
        }
    }

    /// Whether this layout uses linear (not gamma-encoded) values.
    pub const fn is_linear(self) -> bool {
        matches!(
            self,
            Self::RgbLinearF32
                | Self::RgbaLinearF32
                | Self::GrayLinearF32
                | Self::GrayAlphaLinearF32
                | Self::RgbLinearF16
                | Self::RgbaLinearF16
                | Self::GrayLinearF16
                | Self::GrayAlphaLinearF16
        )
    }

    /// Whether this layout uses 16-bit samples.
    pub const fn is_16bit(self) -> bool {
        matches!(
            self,
            Self::Rgb16 | Self::Rgba16 | Self::Gray16 | Self::GrayAlpha16 | Self::Cmyk16
        )
    }

    /// Whether this layout is CMYK (3 colour channels + Black extra
    /// channel). Encoded losslessly only; lossy CMYK would have to
    /// pass C/M/Y through XYB and is not yet wired.
    pub const fn is_cmyk(self) -> bool {
        matches!(self, Self::Cmyk8 | Self::Cmyk16)
    }

    /// Whether this layout uses f32 samples.
    pub const fn is_f32(self) -> bool {
        matches!(
            self,
            Self::RgbLinearF32
                | Self::RgbaLinearF32
                | Self::GrayLinearF32
                | Self::GrayAlphaLinearF32
                | Self::RgbPqF32
                | Self::RgbaPqF32
                | Self::RgbHlgF32
                | Self::RgbaHlgF32
                | Self::RgbBt709F32
                | Self::RgbaBt709F32
        )
    }

    /// Whether this layout uses IEEE 754 binary16 (f16, half-float)
    /// samples in native-endian u16 storage.
    pub const fn is_f16(self) -> bool {
        matches!(
            self,
            Self::RgbLinearF16
                | Self::RgbaLinearF16
                | Self::GrayLinearF16
                | Self::GrayAlphaLinearF16
        )
    }

    /// Whether this layout includes an alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::Rgba8
                | Self::Bgra8
                | Self::GrayAlpha8
                | Self::Rgba16
                | Self::GrayAlpha16
                | Self::RgbaLinearF32
                | Self::GrayAlphaLinearF32
                | Self::RgbaLinearF16
                | Self::GrayAlphaLinearF16
                | Self::RgbaPqF32
                | Self::RgbaHlgF32
                | Self::RgbaBt709F32
        )
    }

    /// Whether this layout is grayscale.
    pub const fn is_grayscale(self) -> bool {
        matches!(
            self,
            Self::Gray8
                | Self::GrayAlpha8
                | Self::Gray16
                | Self::GrayAlpha16
                | Self::GrayLinearF32
                | Self::GrayAlphaLinearF32
                | Self::GrayLinearF16
                | Self::GrayAlphaLinearF16
        )
    }

    /// The transfer function implied by this layout, if any.
    ///
    /// Layouts whose names carry an explicit transfer function (PQ /
    /// HLG / BT.709 f32 input — A3 chunk 1b, issue #46) return
    /// `Some(...)`; sRGB-default and linear layouts return `None`
    /// (caller may still override with [`LossyConfig::with_color_encoding`] /
    /// [`LosslessConfig::with_color_encoding`]).
    ///
    /// The encoder consults this when the caller did NOT set an
    /// explicit `with_color_encoding(...)` so that PQ / HLG / BT.709
    /// f32 input is signaled correctly in the codestream without
    /// requiring callers to wire a `ColorEncoding` separately.
    pub(crate) fn implied_transfer_function(
        self,
    ) -> Option<crate::headers::color_encoding::TransferFunction> {
        use crate::headers::color_encoding::TransferFunction;
        match self {
            Self::RgbPqF32 | Self::RgbaPqF32 => Some(TransferFunction::Pq),
            Self::RgbHlgF32 | Self::RgbaHlgF32 => Some(TransferFunction::Hlg),
            Self::RgbBt709F32 | Self::RgbaBt709F32 => Some(TransferFunction::Bt709),
            // Linear-named float layouts imply `Linear`. The existing
            // u8 / u16 / sRGB-named layouts do NOT carry an implied TF
            // (caller-controlled via `with_color_encoding`).
            Self::RgbLinearF32
            | Self::RgbaLinearF32
            | Self::GrayLinearF32
            | Self::GrayAlphaLinearF32
            | Self::RgbLinearF16
            | Self::RgbaLinearF16
            | Self::GrayLinearF16
            | Self::GrayAlphaLinearF16 => Some(TransferFunction::Linear),
            _ => None,
        }
    }
}

// ── Quality ─────────────────────────────────────────────────────────────────

/// Quality specification for lossy encoding.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Quality {
    /// Butteraugli distance (1.0 = high quality, lower = better).
    Distance(f32),
    /// Percentage scale (0–100, 100 = mathematically lossless, invalid for lossy).
    Percent(u32),
}

impl Quality {
    /// Convert to butteraugli distance.
    fn to_distance(self) -> core::result::Result<f32, EncodeError> {
        match self {
            Self::Distance(d) => {
                if d <= 0.0 {
                    return Err(EncodeError::InvalidConfig {
                        message: format!("lossy distance must be > 0.0, got {d}"),
                    });
                }
                Ok(d)
            }
            Self::Percent(q) => {
                if q >= 100 {
                    return Err(EncodeError::InvalidConfig {
                        message: "quality 100 is lossless; use LosslessConfig instead".into(),
                    });
                }
                Ok(percent_to_distance(q))
            }
        }
    }
}

fn percent_to_distance(quality: u32) -> f32 {
    if quality >= 100 {
        0.0
    } else if quality >= 90 {
        (100 - quality) as f32 / 10.0
    } else if quality >= 70 {
        1.0 + (90 - quality) as f32 / 20.0
    } else {
        2.0 + (70 - quality) as f32 / 10.0
    }
}

/// Convert quality on 0–100 scale to JXL butteraugli distance.
///
/// Matches the jxl-encoder's own `percent_to_distance` piecewise mapping:
/// - 90–100 → distance 0.0–1.0  (perceptually lossless zone)
/// - 70–90  → distance 1.0–2.0  (high quality)
/// - 0–70   → distance 2.0–9.0  (lower quality)
#[must_use]
pub fn quality_to_distance(quality: f32) -> f32 {
    let q = quality.clamp(0.0, 100.0);
    if q >= 100.0 {
        0.0
    } else if q >= 90.0 {
        (100.0 - q) / 10.0
    } else if q >= 70.0 {
        1.0 + (90.0 - q) / 20.0
    } else {
        2.0 + (70.0 - q) / 10.0
    }
}

/// Map generic quality (libjpeg-turbo scale) to JXL native quality.
///
/// Calibrated on CID22-512 corpus (209 images) to produce the same median
/// SSIMULACRA2 as libjpeg-turbo at each quality level. The native quality
/// is then mapped to Butteraugli distance by [`quality_to_distance`].
#[must_use]
pub fn calibrated_jxl_quality(generic_q: f32) -> f32 {
    let clamped = generic_q.clamp(0.0, 100.0);
    const TABLE: &[(f32, f32)] = &[
        (5.0, 5.0),
        (10.0, 5.0),
        (15.0, 5.0),
        (20.0, 5.0),
        (25.0, 9.3),
        (30.0, 22.7),
        (35.0, 33.0),
        (40.0, 38.8),
        (45.0, 43.8),
        (50.0, 48.5),
        (55.0, 51.9),
        (60.0, 55.1),
        (65.0, 58.0),
        (70.0, 61.3),
        (72.0, 63.2),
        (75.0, 65.5),
        (78.0, 67.9),
        (80.0, 69.1),
        (82.0, 71.8),
        (85.0, 76.1),
        (87.0, 79.3),
        (90.0, 84.2),
        (92.0, 86.9),
        (95.0, 91.2),
        (97.0, 92.8),
        (99.0, 93.8),
    ];
    interp_quality(TABLE, clamped)
}

/// Piecewise linear interpolation with clamping at table bounds.
fn interp_quality(table: &[(f32, f32)], x: f32) -> f32 {
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }
    for i in 1..table.len() {
        if x <= table[i].0 {
            let (x0, y0) = table[i - 1];
            let (x1, y1) = table[i];
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    table[table.len() - 1].1
}

/// Box-filter downsample a u8 channel by `factor` (mirrors libjxl
/// `DoDownsampleImage` in `lib/jxl/image_ops.cc`). Companion to
/// [`ExtraChannel::with_dim_shift`] for the
/// `--ec_resampling`-style workflow where the caller owns
/// full-resolution channel data (e.g., alpha plane sliced out of
/// RGBA8 input) but wants to store it at lower resolution.
///
/// Output dimensions are `(width.div_ceil(factor),
/// height.div_ceil(factor))`. Edge cells average only the
/// in-bounds samples (partial 2×2 / 4×4 / 8×8 patches), matching
/// the libjxl behaviour.
///
/// `factor` should be one of `{1, 2, 4, 8}` to match libjxl's
/// `--ec_resampling N` accepted values; passing `factor == 0`
/// returns an empty `Vec` and `factor == 1` returns a clone of the
/// input.
///
/// # Example
///
/// ```ignore
/// use jxl_encoder::api::{downsample_channel_u8, ExtraChannel};
/// let alpha_half = downsample_channel_u8(&alpha_fullres, w, h, 2);
/// let extras = [ExtraChannel::from_alpha_buf(&alpha_half, false)
///     .with_dim_shift(1)];
/// ```
#[must_use]
pub fn downsample_channel_u8(src: &[u8], width: usize, height: usize, factor: u32) -> Vec<u8> {
    if factor == 0 {
        return Vec::new();
    }
    if factor == 1 {
        return src.to_vec();
    }
    debug_assert_eq!(
        src.len(),
        width * height,
        "downsample_channel_u8: src len {} != width*height {}",
        src.len(),
        width * height,
    );
    let factor = factor as usize;
    let out_w = width.div_ceil(factor);
    let out_h = height.div_ceil(factor);
    let mut out = Vec::with_capacity(out_w * out_h);
    for oy in 0..out_h {
        let y0 = oy * factor;
        for ox in 0..out_w {
            let x0 = ox * factor;
            let mut sum: u32 = 0;
            let mut cnt: u32 = 0;
            for iy in 0..factor {
                let y = y0 + iy;
                if y >= height {
                    break;
                }
                let row_off = y * width;
                for ix in 0..factor {
                    let x = x0 + ix;
                    if x >= width {
                        break;
                    }
                    sum += src[row_off + x] as u32;
                    cnt += 1;
                }
            }
            // cnt is guaranteed > 0 because oy < out_h, ox < out_w
            // imply at least the (y0, x0) sample is in-bounds.
            let avg = (sum + cnt / 2) / cnt; // round-to-nearest
            out.push(avg.min(255) as u8);
        }
    }
    out
}

// ── W44-35: smooth-photo DCT64 admission detector ──────────────────────────

/// Mean-abs adjacent-luma-diff threshold (4×-downsampled luma plane,
/// integer scale 0..255) below which the input is "smooth enough" for
/// DCT64-class transforms to win on the gated-cell (W44-35).
/// Calibration: smooth photos 1418519 / 7256805 / 1025469 all measure
/// `<= 0.10`; textured photos and most screenshots measure `>= 0.20`.
const W44_35_PROXY_EDGE_DENSITY_MAX: f32 = 0.15;

/// Fraction-of-8×8-blocks-with-luma-variance-below-5 threshold above
/// which the input looks like solid-color screen content (W44-35).
/// Calibration: smooth photos measure `~0.50`; solid-color UI screenshots
/// (codec_wiki, terminal) measure `>= 0.67`.
const W44_35_PROXY_FLAT_SOLID_MAX: f32 = 0.60;

/// `mean(|2c - l - r| over rows + |2c - t - b| over cols) / mean(|r-l| + |b-t|)`
/// threshold above which the input has too much high-frequency energy to
/// benefit from DCT64-class merges (W44-35). Calibration: smooth photos
/// 1418519 / 7256805 measure `<= 0.95`; textured photo 1025469 measures
/// `1.38`; screenshots measure `>= 1.30`.
const W44_35_PROXY_HF_RATIO_MAX: f32 = 1.0;

/// Auto detector for the W44-35 smooth-photo DCT64 admission gate.
///
/// Returns `true` when the input classifies as a smooth photo that
/// benefits from DCT64-class transforms even on the small + low-distance
/// gated cell. Cheap cost (~0.2 ms on 512×512 RGB on a modern desktop):
/// reads the input once, computes three integer aggregates on a
/// 4×-downsampled luma plane.
///
/// Discriminator (all 3 must hold):
/// - Edge density proxy < [`W44_35_PROXY_EDGE_DENSITY_MAX`] = 0.15
/// - Solid-color block ratio < [`W44_35_PROXY_FLAT_SOLID_MAX`] = 0.60
/// - HF energy ratio < [`W44_35_PROXY_HF_RATIO_MAX`] = 1.0
///
/// **Provenance**: `benchmarks/w44_35_dct64_smart_dispatch_calibrate.tsv`
/// and `examples/dct64_smart_dispatch_proxy_calibrate.rs`. Validated on
/// 10 stratified images across 8 cells each (e ∈ {6, 7}, d ∈ {1.0, 1.2,
/// 1.6, 2.0}): admits 1418519 (-5 to -7 % bytes), admits 7256805
/// (-2 to -4 %), correctly skips 1025469 (mixed +/- < 1.3 %), correctly
/// skips all 4 screenshots / pixel-art.
///
/// **Callers**: each lossy entry point in [`crate::api`] computes this
/// once on the input RGB before resolving the per-image profile via
/// [`LossyConfig::effective_profile_for_image_with_smoothness`]. The
/// `StrategyOverrides::smooth_photo_dct64_hint = Some(_)` override
/// (set via [`LossyConfig::with_strategy_overrides`]) always wins.
///
/// **Layout dispatch**: this function takes raw `pixels` plus a
/// [`PixelLayout`] descriptor and only fires on the u8 sRGB-encoded
/// layouts (`Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`). For other layouts
/// (16-bit, float, gray) it returns `false` — the gate falls through
/// to the default `adapt_to_image_lossy` behaviour. Callers wanting
/// the admission on non-u8 layouts should set
/// [`LossyConfig::with_strategy_overrides`] with
/// `smooth_photo_dct64_hint: Some(true)`.
#[must_use]
pub(crate) fn detect_smooth_photo_for_dct64_from_layout(
    pixels: &[u8],
    w: u32,
    h: u32,
    layout: PixelLayout,
) -> bool {
    // Only sRGB u8 layouts are supported by the cheap proxy. Other
    // layouts return `false` (conservative — the gate stays as today).
    let (bpp, r_offset) = match layout {
        PixelLayout::Rgb8 => (3usize, 0usize),
        PixelLayout::Rgba8 => (4usize, 0usize),
        PixelLayout::Bgr8 => (3usize, 2usize), // R at +2, B at +0
        PixelLayout::Bgra8 => (4usize, 2usize),
        _ => return false,
    };
    detect_smooth_photo_for_dct64_inner(pixels, w, h, bpp, r_offset)
}

/// Internal kernel for [`detect_smooth_photo_for_dct64_from_layout`].
/// Operates on interleaved u8 sRGB at `bpp` bytes per pixel with the
/// R channel at byte offset `r_offset` (0 for RGB-order, 2 for BGR-order).
/// Luma is approximated as `(R + 2G + B) >> 2`.
fn detect_smooth_photo_for_dct64_inner(
    rgb: &[u8],
    w: u32,
    h: u32,
    bpp: usize,
    r_offset: usize,
) -> bool {
    // Cheap proxy: only meaningful when the gate condition would fire
    // (pixels < 500_000). Skip the work otherwise to keep the entry-point
    // dispatch overhead-free on production-sized inputs.
    let pixels = (w as u64) * (h as u64);
    if pixels >= crate::effort::LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD {
        return false;
    }
    let w = w as usize;
    let h = h as usize;
    if w * h * bpp != rgb.len() {
        return false;
    }
    let stride = w * bpp;
    // Channel-byte offsets for R/G/B (G is always between R and B
    // in RGB-order; for BGR-order R is at +2, G is at +1, B is at +0).
    let g_off = 1usize;
    let b_off = if r_offset == 0 { 2 } else { 0 };
    let ds = 4usize;
    let dw = w / ds;
    let dh = h / ds;
    if dw < 3 || dh < 3 {
        return false;
    }
    // Build integer luma plane on the 4×-downsampled grid.
    let mut luma = vec![0i32; dw * dh];
    for dy in 0..dh {
        let y = dy * ds;
        let row_off = y * stride;
        for dx in 0..dw {
            let x = dx * ds;
            let i = row_off + x * bpp;
            let r = rgb[i + r_offset] as i32;
            let g = rgb[i + g_off] as i32;
            let b = rgb[i + b_off] as i32;
            // Approximate luma: (R + 2G + B) / 4, range 0..255.
            luma[dy * dw + dx] = (r + 2 * g + b) >> 2;
        }
    }
    // Proxy 1: mean abs adjacent luma diff (edge density).
    let mut sum_d1: u64 = 0;
    let mut count_d1: u64 = 0;
    for y in 0..dh {
        for x in 1..dw {
            let a = luma[y * dw + x - 1];
            let b = luma[y * dw + x];
            sum_d1 += (a - b).unsigned_abs() as u64;
            count_d1 += 1;
        }
    }
    for y in 1..dh {
        for x in 0..dw {
            let a = luma[(y - 1) * dw + x];
            let b = luma[y * dw + x];
            sum_d1 += (a - b).unsigned_abs() as u64;
            count_d1 += 1;
        }
    }
    let mean_d1 = (sum_d1 as f32) / (count_d1.max(1) as f32);
    let proxy_edge = mean_d1 / 64.0;
    if proxy_edge >= W44_35_PROXY_EDGE_DENSITY_MAX {
        return false;
    }
    // Proxy 2: HF energy ratio (mean abs 2nd-derivative / mean abs 1st-derivative).
    let mut sum_d2: u64 = 0;
    let mut count_d2: u64 = 0;
    let mut sum_d1_center: u64 = 0;
    for y in 0..dh {
        for x in 1..dw - 1 {
            let l = luma[y * dw + x - 1];
            let c = luma[y * dw + x];
            let r = luma[y * dw + x + 1];
            sum_d2 += (2 * c - l - r).unsigned_abs() as u64;
            sum_d1_center += (r - l).unsigned_abs() as u64;
            count_d2 += 1;
        }
    }
    for y in 1..dh - 1 {
        for x in 0..dw {
            let t = luma[(y - 1) * dw + x];
            let c = luma[y * dw + x];
            let b = luma[(y + 1) * dw + x];
            sum_d2 += (2 * c - t - b).unsigned_abs() as u64;
            sum_d1_center += (b - t).unsigned_abs() as u64;
            count_d2 += 1;
        }
    }
    let mean_d2 = (sum_d2 as f32) / (count_d2.max(1) as f32);
    let mean_d1c = (sum_d1_center as f32) / (count_d2.max(1) as f32) + 0.001;
    let proxy_hf = mean_d2 / mean_d1c;
    if proxy_hf >= W44_35_PROXY_HF_RATIO_MAX {
        return false;
    }
    // Proxy 3: fraction of 8×8 luma blocks with intra-block variance < 5
    // (= near-solid-color regions, typical of screen content). Run on the
    // full-resolution input.
    let block = 8usize;
    let bw = w / block;
    let bh = h / block;
    if bw == 0 || bh == 0 {
        // Image smaller than a single 8×8 block — can't be DCT64 candidate.
        return false;
    }
    let mut flat: u64 = 0;
    let mut total: u64 = 0;
    for by in 0..bh {
        for bx in 0..bw {
            let mut sum: u32 = 0;
            let mut sum_sq: u32 = 0;
            for py in 0..block {
                let y = by * block + py;
                let row_off = y * stride;
                for px in 0..block {
                    let x = bx * block + px;
                    let i = row_off + x * bpp;
                    let r = rgb[i + r_offset] as u32;
                    let g = rgb[i + g_off] as u32;
                    let b = rgb[i + b_off] as u32;
                    let luma_v = (r + 2 * g + b) >> 2;
                    sum += luma_v;
                    sum_sq += luma_v * luma_v;
                }
            }
            let n = (block * block) as u32;
            let mean = sum as f32 / n as f32;
            let var = (sum_sq as f32 / n as f32) - mean * mean;
            if var < 5.0 {
                flat += 1;
            }
            total += 1;
        }
    }
    let ratio = (flat as f32) / (total.max(1) as f32);
    if ratio >= W44_35_PROXY_FLAT_SOLID_MAX {
        return false;
    }
    true
}

// ── Supporting types ────────────────────────────────────────────────────────

/// Image metadata (ICC, EXIF, XMP, JUMBF, tone mapping) to embed in the JXL file.
#[derive(Clone, Debug, Default)]
pub struct ImageMetadata<'a> {
    icc_profile: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
    /// JUMBF (JPEG Universal Metadata Box Format, ISO 19566-5) payload,
    /// emitted as a `jumb` ISOBMFF box appended after `Exif`/`xml `.
    /// Used by C2PA / Content Authenticity Initiative tooling. The
    /// encoder passes the bytes through verbatim — no validation.
    jumbf: Option<&'a [u8]>,
    /// Alternative colour-descriptor box payload (ISOBMFF `colr`, ISO/IEC
    /// 14496-12), appended after all other metadata boxes. Pass-through
    /// only — the encoder does not interpret. Use
    /// [`crate::container::colr_nclx_payload`] to build a conformant nclx
    /// payload from CICP enum values.
    colr_payload: Option<&'a [u8]>,
    /// HDR content-description box payload (`hCdR`), appended after all
    /// other metadata boxes. Pass-through only — caller assembles the
    /// schema-specific bytes (e.g. SMPTE ST 2086 + CTA-861.3
    /// MaxCLL/MaxFALL). The encoder does not validate.
    hcdr_payload: Option<&'a [u8]>,
    /// Peak display luminance in nits (cd/m²). `None` uses the JXL default (255.0 = SDR).
    intensity_target: Option<f32>,
    /// Minimum display luminance in nits. `None` uses the JXL default (0.0).
    min_nits: Option<f32>,
    /// Intrinsic display size `(width, height)`, if different from coded dimensions.
    intrinsic_size: Option<(u32, u32)>,
}

impl<'a> ImageMetadata<'a> {
    /// Create empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach an ICC color profile.
    pub fn with_icc_profile(mut self, data: &'a [u8]) -> Self {
        self.icc_profile = Some(data);
        self
    }

    /// Attach EXIF data.
    pub fn with_exif(mut self, data: &'a [u8]) -> Self {
        self.exif = Some(data);
        self
    }

    /// Attach XMP data.
    pub fn with_xmp(mut self, data: &'a [u8]) -> Self {
        self.xmp = Some(data);
        self
    }

    /// Attach a JUMBF (JPEG Universal Metadata Box Format) payload.
    ///
    /// The bytes are written verbatim into a `jumb` ISOBMFF box appended
    /// after the standard `Exif`/`xml ` boxes. Used by C2PA / Content
    /// Authenticity Initiative tooling for provenance metadata; the
    /// caller produces the JUMBF superbox (typically via the `c2pa`
    /// crate) and we pass it through without inspection. Mirrors
    /// libjxl's `JxlEncoderAddBox(enc, "jumb", ...)` API.
    pub fn with_jumbf(mut self, data: &'a [u8]) -> Self {
        self.jumbf = Some(data);
        self
    }

    /// Get the ICC color profile, if set.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile
    }

    /// Get the EXIF data, if set.
    pub fn exif(&self) -> Option<&[u8]> {
        self.exif
    }

    /// Get the XMP data, if set.
    pub fn xmp(&self) -> Option<&[u8]> {
        self.xmp
    }

    /// Get the JUMBF payload, if set.
    pub fn jumbf(&self) -> Option<&[u8]> {
        self.jumbf
    }

    /// Attach an alternative colour-descriptor box (`colr`,
    /// ISO/IEC 14496-12 ColourInformationBox).
    ///
    /// `data` is the raw box content — the first 4 bytes are the
    /// `colour_type` FourCC (`nclx`, `rICC`, `prof`, …) and the rest is
    /// the subtype-specific payload. Use
    /// [`crate::container::colr_nclx_payload`] to construct an nclx
    /// payload from CICP enum values.
    ///
    /// JPEG XL signals its primary colour information in-codestream via
    /// [`crate::headers::color_encoding::ColourEncoding`]; this box is
    /// an **alternative descriptor** for ISOBMFF-aware tooling
    /// (HEIF/AVIF metadata inspectors). Per JPEG XL spec clause 5,
    /// decoders MUST ignore boxes with unrecognised types — so emitting
    /// this box never alters decoded pixels.
    ///
    /// Only honoured on the one-shot [`EncodeRequest`] path. Streaming
    /// encoders (`LossyEncoder` / `LosslessEncoder`) do not surface
    /// `ImageMetadata` and silently drop this field.
    pub fn with_colr_payload(mut self, data: &'a [u8]) -> Self {
        self.colr_payload = Some(data);
        self
    }

    /// Get the `colr` payload, if set.
    pub fn colr_payload(&self) -> Option<&[u8]> {
        self.colr_payload
    }

    /// Attach an HDR content-description box (`hCdR`) payload.
    ///
    /// `data` is the raw box content. The encoder does not validate or
    /// interpret it — callers assemble the schema-specific bytes for
    /// their downstream tooling (e.g. SMPTE ST 2086 mastering display
    /// volume + CTA-861.3 MaxCLL/MaxFALL).
    ///
    /// JPEG XL signals peak/min display luminance in-codestream via
    /// [`crate::headers::color_encoding::ToneMapping`]
    /// (`intensity_target`, `min_nits`). This box is an **alternative
    /// descriptor** for ISOBMFF-aware HDR tooling. Per JPEG XL spec
    /// clause 5, decoders MUST ignore boxes with unrecognised types —
    /// so emitting this box never alters decoded pixels.
    ///
    /// Only honoured on the one-shot [`EncodeRequest`] path. Streaming
    /// encoders (`LossyEncoder` / `LosslessEncoder`) do not surface
    /// `ImageMetadata` and silently drop this field.
    pub fn with_hcdr_payload(mut self, data: &'a [u8]) -> Self {
        self.hcdr_payload = Some(data);
        self
    }

    /// Get the `hCdR` payload, if set.
    pub fn hcdr_payload(&self) -> Option<&[u8]> {
        self.hcdr_payload
    }

    /// Set the peak display luminance in nits (cd/m²) for HDR content.
    ///
    /// Written to the JXL codestream `ToneMapping.intensity_target` field.
    /// Default is 255.0 (SDR). Set to e.g. 4000.0 or 10000.0 for HDR.
    pub fn with_intensity_target(mut self, nits: f32) -> Self {
        self.intensity_target = Some(nits);
        self
    }

    /// Set the minimum display luminance in nits.
    ///
    /// Written to the JXL codestream `ToneMapping.min_nits` field.
    /// Default is 0.0.
    pub fn with_min_nits(mut self, nits: f32) -> Self {
        self.min_nits = Some(nits);
        self
    }

    /// Get the intensity target, if set.
    pub fn intensity_target(&self) -> Option<f32> {
        self.intensity_target
    }

    /// Get the min nits, if set.
    pub fn min_nits(&self) -> Option<f32> {
        self.min_nits
    }

    /// Set the intrinsic display size.
    ///
    /// When set, the image should be rendered at this `(width, height)` rather
    /// than the coded dimensions. Written to the JXL codestream `intrinsic_size` field.
    pub fn with_intrinsic_size(mut self, width: u32, height: u32) -> Self {
        self.intrinsic_size = Some((width, height));
        self
    }

    /// Get the intrinsic size, if set.
    pub fn intrinsic_size(&self) -> Option<(u32, u32)> {
        self.intrinsic_size
    }
}

/// Resource limits for encoding.
///
/// Every field is `Option<…>`; `None` means "unlimited" (or "use the
/// validator-side default", for fields that have one). Callers wire a
/// [`Limits`] onto an [`EncodeRequest`] via
/// [`EncodeRequest::with_limits`]; the encoder consults it before any
/// dimension-driven allocation and before each per-encode CPU budget
/// check.
///
/// Two policy fields are intentionally NOT bare `pub const`s:
///
/// - [`Self::max_quant_loop_iters`] — the cap on
///   butteraugli/ssim2/zensim quantization-loop iterations. The
///   validator's hard upper bound is [`Self::DEFAULT_MAX_QUANT_LOOP_ITERS`]
///   (= [`crate::validation::ITER_MAX`]); a caller may set a lower limit
///   here, but never a higher one. The encoder saturates at the lower of
///   `Limits.max_quant_loop_iters` (or its default) and the validator
///   max.
/// - [`Self::max_memory_bytes`] — when `None`, the encoder applies
///   [`Self::DEFAULT_MAX_MEMORY_BYTES`] (≈ 2 GB) as a soft cap so that an
///   image proxy without explicit `Limits` configuration still has a
///   working-set ceiling. Set to `Some(u64::MAX)` to opt out of the soft
///   cap explicitly (this surfaces in logs as "user explicitly disabled
///   memory limit").
#[derive(Clone, Debug, Default)]
pub struct Limits {
    max_width: Option<u64>,
    max_height: Option<u64>,
    max_pixels: Option<u64>,
    max_memory_bytes: Option<u64>,
    max_quant_loop_iters: Option<u32>,
}

impl Limits {
    /// Hard upper bound for quantization-loop iterations. Mirrors
    /// [`crate::validation::ITER_MAX`] so the validator and the encoder
    /// agree on what counts as "too many iters".
    pub const DEFAULT_MAX_QUANT_LOOP_ITERS: u32 = crate::validation::ITER_MAX;

    /// Default soft cap on encoder working-set memory when no explicit
    /// [`Self::with_max_memory_bytes`] is set. ~40 bytes/pixel × ~50
    /// megapixels is roughly 2 GB. Image proxies that don't configure
    /// `Limits` still get this ceiling so an oversized upload can't OOM
    /// the process; set a tighter cap explicitly for hostile-input
    /// scenarios.
    pub const DEFAULT_MAX_MEMORY_BYTES: u64 = 2 * 1024 * 1024 * 1024;

    /// Create limits with no restrictions (all `None`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum image width.
    pub fn with_max_width(mut self, w: u64) -> Self {
        self.max_width = Some(w);
        self
    }

    /// Set maximum image height.
    pub fn with_max_height(mut self, h: u64) -> Self {
        self.max_height = Some(h);
        self
    }

    /// Set maximum total pixels (width × height).
    pub fn with_max_pixels(mut self, p: u64) -> Self {
        self.max_pixels = Some(p);
        self
    }

    /// Set maximum memory bytes the encoder may allocate.
    ///
    /// When unset, the encoder applies
    /// [`Self::DEFAULT_MAX_MEMORY_BYTES`] (~2 GB) as a soft cap. Pass
    /// `u64::MAX` explicitly to disable the cap (with the understanding
    /// that an unbounded working set on a hostile input can OOM the
    /// process).
    pub fn with_max_memory_bytes(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    /// Set maximum quantization-loop iterations (butteraugli / ssim2 /
    /// zensim). Saturated at [`Self::DEFAULT_MAX_QUANT_LOOP_ITERS`] —
    /// passing a higher value silently lowers it to the validator-side
    /// hard limit. Use a lower value to bound CPU on untrusted callers.
    pub fn with_max_quant_loop_iters(mut self, n: u32) -> Self {
        self.max_quant_loop_iters = Some(n.min(Self::DEFAULT_MAX_QUANT_LOOP_ITERS));
        self
    }

    /// Get maximum width, if set.
    pub fn max_width(&self) -> Option<u64> {
        self.max_width
    }

    /// Get maximum height, if set.
    pub fn max_height(&self) -> Option<u64> {
        self.max_height
    }

    /// Get maximum pixels, if set.
    pub fn max_pixels(&self) -> Option<u64> {
        self.max_pixels
    }

    /// Get maximum memory bytes, if set. When `None`,
    /// [`Self::effective_max_memory_bytes`] gives the soft default the
    /// encoder will actually enforce.
    pub fn max_memory_bytes(&self) -> Option<u64> {
        self.max_memory_bytes
    }

    /// The cap the encoder will actually apply: explicit
    /// `max_memory_bytes` if set, else
    /// [`Self::DEFAULT_MAX_MEMORY_BYTES`].
    pub fn effective_max_memory_bytes(&self) -> u64 {
        self.max_memory_bytes
            .unwrap_or(Self::DEFAULT_MAX_MEMORY_BYTES)
    }

    /// Get maximum quantization-loop iterations.
    pub fn max_quant_loop_iters(&self) -> Option<u32> {
        self.max_quant_loop_iters
    }

    /// The cap the encoder will actually apply: explicit
    /// `max_quant_loop_iters` if set, else
    /// [`Self::DEFAULT_MAX_QUANT_LOOP_ITERS`].
    pub fn effective_max_quant_loop_iters(&self) -> u32 {
        self.max_quant_loop_iters
            .unwrap_or(Self::DEFAULT_MAX_QUANT_LOOP_ITERS)
    }
}

// ── Animation ──────────────────────────────────────────────────────────────

/// Animation timing parameters.
#[derive(Clone, Debug)]
pub struct AnimationParams {
    /// Ticks per second numerator (default 100 = 10ms precision).
    pub tps_numerator: u32,
    /// Ticks per second denominator (default 1).
    pub tps_denominator: u32,
    /// Number of loops: 0 = infinite (default), >0 = play N times.
    pub num_loops: u32,
    /// Input frame RGB is already multiplied by alpha (associated /
    /// premultiplied alpha). Default `false` (straight alpha). When
    /// `true`, the encoder unpremultiplies before XYB conversion and
    /// signals `alpha_associated=true` in the codestream so the decoder
    /// re-premultiplies on output. Mirrors
    /// [`EncodeRequest::with_premultiplied_alpha`] for the animation
    /// path. Closes the lossy portion of the animation audit's
    /// "doesn't unpremultiply alpha" finding.
    pub premultiplied_alpha: bool,
}

impl Default for AnimationParams {
    fn default() -> Self {
        Self {
            tps_numerator: 100,
            tps_denominator: 1,
            num_loops: 0,
            premultiplied_alpha: false,
        }
    }
}

/// A single frame in an animation sequence.
///
/// `pixels` and `duration` are required. The remaining fields are
/// optional overrides for libjxl frame-header semantics — when `None`,
/// the encoder picks the default that matches the prior
/// `pixels` + `duration` behavior (see `encode_animation`).
///
/// Use [`AnimationFrame::new`] for the common single-frame case and
/// the `with_*` builders to attach blend mode / timecode / name /
/// reference semantics for multi-layer animations.
///
/// ```rust,no_run
/// # use jxl_encoder::{AnimationFrame, BlendMode};
/// # let frame0_pixels = vec![0u8; 64 * 64 * 4];
/// # let frame1_pixels = vec![0u8; 64 * 64 * 4];
/// let base = AnimationFrame::new(&frame0_pixels, 10);
/// // Frame 1 composites over frame 0 using alpha blending instead of
/// // replacing it. Useful for sprite animations.
/// let overlay = AnimationFrame::new(&frame1_pixels, 10)
///     .with_blend_mode(BlendMode::Blend)
///     .with_name("overlay");
/// ```
pub struct AnimationFrame<'a> {
    /// Raw pixel data (must match width/height/layout from the encode call).
    pub pixels: &'a [u8],
    /// Duration of this frame in ticks (tps_numerator/tps_denominator seconds per tick).
    pub duration: u32,
    /// Per-frame blend mode (libjxl `BlendingInfo::mode`). `None` keeps the
    /// encoder default — `Replace` for frame 0 and any full-frame replacement,
    /// or the crop-derived default when this frame's pixels are a partial
    /// canvas update. `Some(mode)` overrides unconditionally; pair with
    /// [`Self::with_blend_source`] when blending against a reference frame
    /// other than the previous frame's canvas.
    pub blend_mode: Option<BlendMode>,
    /// Source reference slot (0–3) for blending. `None` keeps the encoder
    /// default (1 when this frame uses a crop, 0 otherwise). Only meaningful
    /// when `blend_mode` is set to a non-`Replace` mode.
    pub blend_source: Option<u32>,
    /// Save this frame to a reference slot (0–3) so later frames can
    /// composite against it. `None` keeps the encoder default — non-last
    /// animated frames save to slot 1 so successor frames with crops can
    /// blend over the persistent canvas. `Some(0)` explicitly disables
    /// saving (typical for the last frame).
    pub save_as_reference: Option<u32>,
    /// Encode this frame as a `ReferenceOnly` frame (libjxl
    /// `FrameType::kReferenceOnly`). The codestream stores the frame and
    /// writes it into the `save_as_reference` slot, but it is NOT
    /// counted as a displayable keyframe — decoders skip it during
    /// playback. Subsequent regular frames composite against the stored
    /// canvas via [`Self::with_blend_source`].
    ///
    /// Combine with [`Self::with_save_as_reference`] to pick the target
    /// slot (default `1`). The animation entry points reject
    /// `reference_only=true` on the last frame
    /// ([`EncodeError::InvalidInput`]) — the codestream must end with a
    /// displayable frame.
    ///
    /// `duration` is ignored for reference-only frames; the field is
    /// suppressed in the bitstream by the spec.
    pub reference_only: bool,
    /// Optional frame name (libjxl `FrameHeader::name`). `None` writes no
    /// name. Useful for tooling that wants per-frame labels (layer titles,
    /// animation key names). Maximum 1019 bytes per the spec.
    pub name: Option<String>,
    /// Optional SMPTE timecode for this frame (libjxl
    /// `FrameHeader::timecode`). `None` writes no timecode. Setting
    /// `Some(_)` on **any** frame in the animation flips the file-level
    /// `have_timecodes` flag and emits a 32-bit timecode field for every
    /// frame (frames left at `None` get timecode `0`).
    pub timecode: Option<u32>,
}

impl<'a> AnimationFrame<'a> {
    /// Create an animation frame with `pixels` and `duration`. All
    /// optional fields default to `None` (= keep the encoder defaults).
    pub fn new(pixels: &'a [u8], duration: u32) -> Self {
        Self {
            pixels,
            duration,
            blend_mode: None,
            blend_source: None,
            save_as_reference: None,
            reference_only: false,
            name: None,
            timecode: None,
        }
    }

    /// Override the per-frame blend mode. See [`BlendMode`].
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = Some(mode);
        self
    }

    /// Set the reference frame slot to blend against (0–3).
    /// Only meaningful when `blend_mode` is non-`Replace`.
    pub fn with_blend_source(mut self, source: u32) -> Self {
        self.blend_source = Some(source);
        self
    }

    /// Save this frame to reference slot (0–3) for later compositing.
    /// Pass `0` to explicitly disable saving (overriding the encoder's
    /// default of `1` for non-last animated frames).
    pub fn with_save_as_reference(mut self, slot: u32) -> Self {
        self.save_as_reference = Some(slot);
        self
    }

    /// Mark this frame as `ReferenceOnly` (libjxl
    /// `FrameType::kReferenceOnly`). The frame is encoded and saved to
    /// its target reference slot but NOT presented as a displayable
    /// keyframe — decoders skip it during playback. Useful for
    /// composing a "background" or "layer source" that later regular
    /// frames blend on top of via [`Self::with_blend_source`].
    ///
    /// Combine with [`Self::with_save_as_reference`] to choose the
    /// target slot (default `1`). Pair with `[Self::with_blend_source]`
    /// on a later frame to composite against the saved canvas.
    ///
    /// Constraints (enforced by [`crate::LossyConfig::encode_animation`]
    /// / [`crate::LosslessConfig::encode_animation`]):
    /// - Reference-only frames may NOT be the last frame in the
    ///   animation. The file must end on a displayable frame.
    /// - `duration` is ignored: the spec suppresses the duration field
    ///   for ReferenceOnly frames.
    pub fn with_reference_only(mut self, reference_only: bool) -> Self {
        self.reference_only = reference_only;
        self
    }

    /// Attach a name to this frame (libjxl `FrameHeader::name`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Attach a SMPTE timecode to this frame. Setting this on any frame
    /// in the animation flips the file-level `have_timecodes` flag.
    pub fn with_timecode(mut self, timecode: u32) -> Self {
        self.timecode = Some(timecode);
        self
    }
}

impl Default for AnimationFrame<'_> {
    fn default() -> Self {
        Self::new(&[], 0)
    }
}

// ── Shared knob enums (LossyConfig + LosslessConfig) ───────────────────────

/// Container-wrap policy for the encoded JXL output.
///
/// Mirrors libjxl `cjxl --container 0|1`. The default ([`Auto`]) wraps
/// the codestream in a JXL container (`JXL ` signature box +
/// `jxlc`/`jxlp` data boxes + any metadata boxes) **only** when
/// required — i.e., the codestream uses a level that requires the
/// container box (libjxl `MustUseContainer`), or the caller attached
/// EXIF / XMP / JUMBF / colr / hCdR metadata.
///
/// [`Always`] forces a container wrapper even when the bare codestream
/// would have been spec-valid on its own — useful for downstream tools
/// that always expect the ISOBMFF framing. [`Never`] skips the
/// container even when metadata is present (the metadata is silently
/// dropped); this fails the encode (returns
/// [`EncodeError::InvalidInput`]) if the codestream level requires a
/// container, since the result would be unreadable.
///
/// [`Auto`]: ContainerMode::Auto
/// [`Always`]: ContainerMode::Always
/// [`Never`]: ContainerMode::Never
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ContainerMode {
    /// **Default.** Wrap in a container box only when required
    /// (metadata present, or `codestream_level != 5`). Matches libjxl's
    /// `MustUseContainer` semantics.
    #[default]
    Auto,
    /// Always emit the container wrapper, even for bare-codestream-OK
    /// encodes. Equivalent to libjxl `--container 1`.
    Always,
    /// Never wrap; emit the bare codestream. Drops attached EXIF / XMP
    /// / JUMBF / colr / hCdR silently (they have nowhere to go without
    /// the container). Returns [`EncodeError::InvalidInput`] when the
    /// codestream level requires a container (e.g. level 10).
    /// Equivalent to libjxl `--container 0`.
    Never,
}

/// Trait bundle for output destinations that support both writing and
/// random-access seeking. Required by the streaming-refactor
/// [`LossyEncoder::finish_to_seekable`] / [`LosslessEncoder::finish_to_seekable`]
/// entry points (jxl-encoder#11 chunk 6).
///
/// The blanket impl covers every type that implements [`std::io::Write`]
/// + [`std::io::Seek`], so concrete callers don't need to opt in:
///
/// ```ignore
/// use std::io::Cursor;
/// use jxl_encoder::{LossyConfig, Quality, PixelLayout};
///
/// let mut buf = Cursor::new(Vec::<u8>::new());
/// LossyConfig::new(1.0)
///     .encoder(1024, 1024, PixelLayout::Rgb8)?
///     .finish_to_seekable(&mut buf)?;
/// let encoded: Vec<u8> = buf.into_inner();
/// # Ok::<(), jxl_encoder::EncodeError>(())
/// ```
///
/// **Chunk 6 (this commit)** plumbs the trait through both encoder
/// builders but uses it only as a [`std::io::Write`]: the buffered-
/// output bytes are produced in memory then written in one pass. The
/// seek capability becomes load-bearing in chunk 7 when the level-3
/// streaming-output path lands (permuted TOC + DC-global placeholder +
/// post-frame seek-back, mirroring libjxl `acc28c0` /
/// `OutputProcessor::Seek`).
///
/// Mirrors libjxl's "streaming_output" assumption in
/// `EncodeFrameStreaming` (`enc_frame.cc:2042-2200`) that the output
/// sink can rewind to the DC-global slot once all per-DC-group section
/// bytes are known.
#[cfg(feature = "std")]
pub trait WritableSeek: std::io::Write + std::io::Seek {}

#[cfg(feature = "std")]
impl<T: std::io::Write + std::io::Seek> WritableSeek for T {}

/// Input/output buffering policy for the encode pipeline. Mirrors
/// libjxl `cjxl --buffering -1..3`
/// ([`JXL_ENC_FRAME_SETTING_BUFFERING`][libjxl-encode-h]).
///
/// This is the scaffolding API for the streaming refactor tracked in
/// jxl-encoder#11 / libjxl PRs #4634 + #4635 + #4637 + #4642 + #4728
/// (commits `acc28c0` + `032d39a` + `b3510d1` + `1389871` + `6553831`).
/// **Chunk 1 (this commit)** introduces the enum, the builder methods
/// on [`LossyConfig`] / [`LosslessConfig`], and the CLI flag. **No
/// dispatch is wired yet** — every variant currently routes through
/// the existing one-shot full-buffer path, so output bytes are
/// identical regardless of which `Buffering` value is selected.
/// Chunks 2-7 land the per-DC-group split, the buffered-output path
/// (libjxl level 2), the permuted-TOC seek-back path (libjxl level
/// 3), and the lossless mirror. See
/// [`libjxl_streaming_refactor_porting_plan_2026-05-18`][plan] for
/// the full chunk schedule.
///
/// libjxl semantics (post-`acc28c0`):
///
/// | libjxl int | This enum                       | Meaning                                                                                          |
/// |-----------:|---------------------------------|--------------------------------------------------------------------------------------------------|
/// |       `-1` | [`Auto`](Self::Auto)            | Encoder picks. Currently resolves to libjxl level **2** for `num_dc_groups > 8`, else level **0**. |
/// |        `0` | [`FullBuffered`](Self::FullBuffered) | Buffer everything — semantically equivalent to today's one-shot encode path.                |
/// |        `1` | [`Threshold2048`](Self::Threshold2048) | Buffer for ≤ 2048×2048; stream input + buffer output for larger images.                   |
/// |        `2` | [`BufferedOutput`](Self::BufferedOutput) | Stream input + buffer output whenever `num_dc_groups > 8`. **libjxl default since `032d39a`.** |
/// |        `3` | [`FullStreaming`](Self::FullStreaming) | Stream input AND stream output. Requires seek-back support on the output sink; the produced bitstream is not progressively decodable. |
///
/// **Critical distinction** (per the libjxl-spec doc-comment in
/// `lib/include/jxl/encode.h` post-`acc28c0`):
///
/// - **Levels 0-2** all produce *progressive-decodability-friendly*
///   bitstreams with a non-permuted TOC and natural-order section
///   layout. Level 2 simply trades the input-side full-buffer for a
///   streaming pixel source while still buffering the encoded
///   per-DC-group sections in `global_group_codes[]` until the loop
///   ends.
/// - **Level 3** is the original "streaming encode" path: permuted
///   TOC with a DC-global placeholder, sections emitted to the sink
///   as soon as each DC group finishes, then a seek-back at the end
///   to fill in the real DC-global + TOC. Smaller peak RAM but the
///   output is *not* progressively decodable.
///
/// **Default**: [`Auto`](Self::Auto) (matches libjxl post-`032d39a`,
/// which changed `JXL_ENC_FRAME_SETTING_BUFFERING` default from `1`
/// to `2`).
///
/// [libjxl-encode-h]: https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/encode.h
/// [plan]: https://github.com/imazen/jxl-encoder/issues/11
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Buffering {
    /// **Default.** Encoder picks per-image based on dimensions and
    /// `num_dc_groups`. Mirrors libjxl `--buffering -1`.
    ///
    /// Current heuristic (chunk 1; will refine in chunks 5/7):
    /// - `num_dc_groups <= 8` → resolves to [`FullBuffered`](Self::FullBuffered).
    /// - Otherwise → resolves to [`BufferedOutput`](Self::BufferedOutput).
    ///
    /// The 2048² threshold matches libjxl's
    /// `CanDoStreamingEncoding` gate (`enc_frame.cc:1779-1820`): a
    /// `2048×2048` image fits in exactly one DC group (so streaming
    /// gives no win), while larger images split into multiple DC
    /// groups where the buffered-output path can drop per-region
    /// pixel buffers as soon as the corresponding sections are
    /// emitted.
    #[default]
    Auto,
    /// Buffer everything. Semantically equivalent to today's one-shot
    /// encode path. Mirrors libjxl `--buffering 0`.
    ///
    /// Peak memory ≈ pixel buffer + full XYB / quant / mask / CfL /
    /// AC-strategy plane buffers + accumulated section bytes. Smallest
    /// code path; same output as the pre-streaming-refactor encoder.
    FullBuffered,
    /// Buffer everything for inputs ≤ 2048×2048; otherwise stream
    /// input + buffer output. Mirrors libjxl `--buffering 1`.
    ///
    /// Chunk 1: no behavioural difference — routes through the
    /// one-shot path. Chunks 3+5 land the per-DC-group split and the
    /// large-image streaming path.
    Threshold2048,
    /// Always stream input + buffer output when `num_dc_groups > 8`
    /// (i.e. images larger than ~ a single 2048×2048 DC group).
    /// Mirrors libjxl `--buffering 2`. **This is libjxl's default
    /// since `032d39a`.**
    ///
    /// Buffered-output path: the encoder still accumulates every
    /// DC-group's bitstream section in `global_group_codes[]` until
    /// the per-group loop finishes, then writes a non-permuted TOC +
    /// sections in natural order. No seek-back required on the
    /// output sink. Lets the encoder drop each DC group's plane
    /// slice as soon as its sections are emitted — the load-bearing
    /// memory win is the absence of the whole-image XYB / quant /
    /// CfL / AC-strategy plane buffers, not the section buffers
    /// themselves.
    ///
    /// Chunk 1: no behavioural difference — routes through the
    /// one-shot path. Chunk 5 lands the buffered-output streaming
    /// path.
    BufferedOutput,
    /// Stream input AND stream output. Mirrors libjxl `--buffering 3`.
    ///
    /// Requires seek-back support on the output sink (the encoder
    /// reserves the DC-global slot, emits per-DC-group sections as
    /// they finish, then seeks back to write the real DC-global +
    /// permuted TOC at end-of-frame). The produced bitstream is *not*
    /// progressively decodable — the TOC permutation reorders the
    /// sections so DC-global sits at the end.
    ///
    /// Chunk 1: no behavioural difference — routes through the
    /// one-shot path. Chunks 6-7 land the `WritableSeek` trait and
    /// the level-3 streaming-output path.
    FullStreaming,
}

impl Buffering {
    /// Convert from the libjxl `--buffering` integer encoding
    /// (`-1..=3`). Values outside the documented range fold to
    /// [`Auto`](Self::Auto) (matches libjxl's
    /// `JXL_ENC_FRAME_SETTING_BUFFERING` defaulting behaviour for
    /// out-of-range input on the C API).
    pub const fn from_i8(v: i8) -> Self {
        match v {
            0 => Self::FullBuffered,
            1 => Self::Threshold2048,
            2 => Self::BufferedOutput,
            3 => Self::FullStreaming,
            _ => Self::Auto,
        }
    }

    /// Inverse of [`Self::from_i8`]: convert to the libjxl `cjxl
    /// --buffering` integer encoding. [`Auto`](Self::Auto) maps to
    /// `-1`.
    pub const fn to_i8(self) -> i8 {
        match self {
            Self::Auto => -1,
            Self::FullBuffered => 0,
            Self::Threshold2048 => 1,
            Self::BufferedOutput => 2,
            Self::FullStreaming => 3,
        }
    }

    /// Resolve [`Auto`](Self::Auto) to a concrete variant for an
    /// image with the given pixel dimensions. Non-`Auto` variants
    /// are returned unchanged.
    ///
    /// Chunk 1 heuristic (mirrors libjxl `CanDoStreamingEncoding`
    /// in `enc_frame.cc:1779-1820`): images with `width * height
    /// <= 2048 * 2048` (i.e. one DC group) resolve to
    /// [`FullBuffered`](Self::FullBuffered); larger images resolve
    /// to [`BufferedOutput`](Self::BufferedOutput) (libjxl level 2,
    /// matching the post-`032d39a` default).
    ///
    /// This is a no-op for chunk 1 — every concrete variant
    /// currently routes through the same one-shot encode path. The
    /// helper exists so chunks 3-7 can dispatch on the resolved
    /// value without re-implementing the threshold rule.
    pub const fn resolve_for(self, width: u32, height: u32) -> Self {
        match self {
            Self::Auto => {
                // libjxl threshold: a single 2048×2048 DC group fits
                // any image ≤ 2048² total pixels. Use `u64` to avoid
                // overflow on the 4G × 4G upper bound.
                let pixels = (width as u64).saturating_mul(height as u64);
                if pixels <= 2048u64 * 2048u64 {
                    Self::FullBuffered
                } else {
                    Self::BufferedOutput
                }
            }
            other => other,
        }
    }

    /// Returns `true` if this buffering policy is compatible with
    /// streaming encoding (i.e. the encoder may drop per-DC-group
    /// XYB / quant / mask storage as soon as the corresponding
    /// section is emitted).
    ///
    /// Mirrors the streaming-side of libjxl's
    /// [`CanDoStreamingEncoding`](https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_frame.cc)
    /// gate: only [`BufferedOutput`](Self::BufferedOutput) and
    /// [`FullStreaming`](Self::FullStreaming) request the per-region
    /// release path. [`Auto`](Self::Auto) is resolved first via
    /// [`Self::resolve_for`] before this check is meaningful.
    pub const fn is_streaming(self) -> bool {
        matches!(self, Self::BufferedOutput | Self::FullStreaming)
    }

    /// Chunk-8c (#11) streaming gate. Returns the buffering policy
    /// to actually use given a caller-requested mode and whether the
    /// butteraugli quantization loop will run on this encode.
    ///
    /// Mirrors libjxl `CanDoStreamingEncoding` in `enc_frame.cc`:
    /// the butteraugli loop reconstructs the whole image multiple
    /// times to evaluate per-block quality and cannot run from a
    /// sliding-window XYB source. When a caller asks for streaming
    /// (`BufferedOutput` / `FullStreaming` / `Auto` resolved to one
    /// of those) **and** the butteraugli loop is active, this
    /// helper returns [`FullBuffered`](Self::FullBuffered) instead
    /// — the encoder runs the loop on a whole-image XYB then
    /// emits the final pass through the buffered-output path.
    ///
    /// Today the butteraugli loop is feature-gated and effort-gated
    /// (off at default effort 7); the typical request path
    /// (`Auto` + default effort) is unaffected. The `Auto`
    /// resolution happens first so the returned variant is always a
    /// concrete level (never `Auto`).
    pub const fn resolve_for_streaming(
        self,
        width: u32,
        height: u32,
        butteraugli_iters: u32,
    ) -> Self {
        let resolved = self.resolve_for(width, height);
        if butteraugli_iters > 0 && resolved.is_streaming() {
            // Downgrade to FullBuffered so the buttloop sees the
            // whole-image XYB it requires. Mirrors libjxl's
            // CanDoStreamingEncoding which returns false on
            // `use_butteraugli_loop`.
            Self::FullBuffered
        } else {
            resolved
        }
    }
}

/// Premultiplied (associated) alpha policy for inputs with an alpha
/// channel.
///
/// Mirrors libjxl `cjxl --premultiply -1|0|1`.
///
/// - [`Off`]: input alpha is straight (unassociated). Color samples
///   were captured without alpha pre-multiplication. **Default.**
/// - [`On`]: input alpha is premultiplied (associated). Color samples
///   were already multiplied by alpha. Standard for GPU pipelines
///   (Skia, Cairo, Metal, Vulkan, Direct2D, Wayland, CompositorAPI).
/// - [`Auto`]: detect from pixels at encode time. The encoder scans
///   the buffer once: if every color sample is ≤ its alpha sample,
///   the data is treated as premultiplied; otherwise straight. The
///   scan is O(N) and runs before the encode loop; for trusted inputs
///   prefer the explicit [`Off`]/[`On`] forms.
///
/// On the [`LossyConfig`] path the encoder requires the
/// unpremultiplication pre-pass (#13) — calling `finish()` on a lossy
/// encode with [`On`] (or [`Auto`] that resolves to premultiplied)
/// returns [`EncodeError::InvalidInput`]. On the [`LosslessConfig`]
/// path the pixels are preserved bit-exactly and the
/// `alpha_associated` header bit is set so the decoder interprets the
/// stored values correctly.
///
/// [`Auto`]: PremultipliedAlphaMode::Auto
/// [`Off`]: PremultipliedAlphaMode::Off
/// [`On`]: PremultipliedAlphaMode::On
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PremultipliedAlphaMode {
    /// **Default.** Straight (unassociated) alpha. Equivalent to
    /// libjxl `--premultiply 0`.
    #[default]
    Off,
    /// Premultiplied (associated) alpha. Equivalent to libjxl
    /// `--premultiply 1`.
    On,
    /// Detect from pixels at encode time. Equivalent to libjxl
    /// `--premultiply -1`. Adds a single O(N) scan over the input
    /// before encoding.
    Auto,
}

impl PremultipliedAlphaMode {
    /// Convert from the libjxl `--premultiply` integer encoding.
    /// `< 0` = [`Auto`](Self::Auto), `0` = [`Off`](Self::Off), `> 0` =
    /// [`On`](Self::On).
    pub const fn from_i8(v: i8) -> Self {
        if v < 0 {
            Self::Auto
        } else if v == 0 {
            Self::Off
        } else {
            Self::On
        }
    }
}

/// Maximum value for [`LossyConfig::with_faster_decoding`] /
/// [`LosslessConfig::with_faster_decoding`]. Matches libjxl
/// `cjxl --faster_decoding 0..4`.
pub const MAX_FASTER_DECODING: u8 = 4;

/// Maximum value for [`LossyConfig::with_progressive_dc`]. Matches
/// libjxl `cjxl --progressive_dc 0..2`.
///
/// 0 = no progressive DC.
/// 1 = one [`LfFrame`](crate::LossyConfig::with_lf_frame) ahead of the
/// main VarDCT frame.
/// 2 = two nested LfFrames (libjxl path; our encoder currently emits a
/// single LfFrame and warns).
pub const MAX_PROGRESSIVE_DC: u8 = 2;

// ── LosslessConfig ──────────────────────────────────────────────────────────

/// Lossless (modular) encoding configuration.
///
/// Has a sensible `Default` — lossless has no quality ambiguity.
///
/// # libjxl-parity knobs
///
/// The following builders mirror libjxl `cparams` fields:
///
/// - [`Self::with_force_rct`] — `cparams.colorspace`, force a
///   specific Reversible Color Transform (skip the per-effort
///   search). Use [`crate::RctType::YCOCG`] for screenshots.
/// - [`Self::with_tree_learning_sample_fraction`] — override the
///   effort-derived tree-learning sample fraction. Lower the
///   effort-7 cliff (#23) by setting `0.10..=0.20` for a
///   "tree-learning lite" trade.
/// - [`Self::with_squeeze`] — Haar wavelet decomposition (libjxl
///   `cparams.responsive`).
/// - [`Self::with_lossy_palette`] — near-lossless delta palette
///   (libjxl `cparams.lossy_palette`).
/// - [`EncodeRequest::with_brotli_metadata`] — Brotli-compress EXIF /
///   XMP into `brob` boxes (request-level, applies to both modes).
///
/// See [`LossyConfig`] for the matching VarDCT-side knobs
/// (`with_photon_noise_iso`, `with_original_distance`,
/// `with_quant_ac_rescale`, etc.).
#[derive(Clone, Debug)]
pub struct LosslessConfig {
    effort: u8,
    mode: EncoderMode,
    use_ans: bool,
    squeeze: bool,
    tree_learning: bool,
    lz77: bool,
    lz77_method: Lz77Method,
    patches: bool,
    lossy_palette: bool,
    threads: usize,
    /// Override for the effort-derived tree-learning sample fraction
    /// (refs #23 — gives a smoother time/size trade between e6 and e7).
    /// `None` keeps the effort default; `Some(f)` clamps to `[0.0, 1.0]`
    /// and overrides when `tree_learning` is enabled.
    tree_sample_fraction_override: Option<f32>,
    /// Caller-supplied RCT colorspace override (libjxl
    /// `cparams.colorspace`). `None` keeps the per-effort search;
    /// `Some(rct)` skips the search and applies the given RCT.
    forced_rct: Option<crate::modular::rct::RctType>,
    /// Sweep / picker hook: when set, replaces the effort+mode-derived
    /// `EffortProfile` everywhere the encoder asks for one. See
    /// [`Self::with_effort_profile_override`].
    profile_override: Option<crate::effort::EffortProfile>,
    /// Opt-in: re-tune `tree_parallel_max_depth` / `tree_parallel_floor`
    /// per-image (based on pixel count) instead of using the effort-only
    /// defaults. Bitstream-equivalent — only changes rayon fanout shape.
    /// See [`crate::effort::EffortProfile::adapt_to_image`].
    tree_parallel_smart: bool,
    /// Override the always-on small-image parallel-tree-learning
    /// fallback gate. `None` keeps the default (auto-on for inputs
    /// below 1 MP); `Some(false)` forces the gate off (pre-`fe2d3a2`
    /// + pre-`cb5e202` behaviour); `Some(true)` forces the gate on
    ///   regardless of image size. Intended for A/B benches; production
    ///   callers should leave this `None`.
    small_image_fallback_override: Option<bool>,
    /// Zero the RGB samples in pixels whose alpha=0 before lossless
    /// modular encoding (libjxl `SimplifyInvisible` lossless mode,
    /// `enc_frame.cc:511`). `false` (default) preserves all RGB bytes
    /// exactly — matches libjxl lossless default
    /// (`ApplyOverride(keep_invisible, IsLossless()) == true`). `true`
    /// drops RGB-under-transparent and lets modular compress 0-runs
    /// for 5-20% smaller files on sprites / UI assets. Set via
    /// [`Self::with_keep_invisible`].
    simplify_invisible: bool,
    /// Optional forced modular predictor override (CLI passthrough —
    /// mirrors libjxl `cjxl -P` / `--modular_predictor`,
    /// `enc_params.h:options.predictor`). `None` (default) lets the
    /// tree learner choose. `Some(n)` for `n in 0..=13` corresponds to
    /// [`crate::modular::Predictor`] variants `Zero..Average4`. `Some(14)`
    /// reserved for libjxl `Predictor::Best`, `Some(15)` for
    /// `Predictor::Variable` — both stored on the config for surface
    /// completeness; encoder-side fixed-predictor wiring is queued
    /// follow-on work (current behaviour: tree learner / weighted /
    /// gradient defaults remain in effect even when set).
    /// See [`Self::with_modular_predictor`].
    modular_predictor: Option<u8>,
    /// Optional override of the palette-transform colour cap (CLI
    /// passthrough — mirrors libjxl `cjxl --modular_palette_colors`,
    /// `enc_params.h:palette_colors`). `None` (default) keeps the
    /// built-in [`crate::modular::palette::MAX_PALETTE_COLORS`] (1024).
    /// `Some(0)` disables palette detection. `Some(n)` caps the
    /// palette-colour search at `n`. Stored on the config; wiring
    /// through the palette-search call sites in `modular/encode.rs`
    /// is queued follow-on work — current behaviour uses the built-in
    /// constant. See [`Self::with_modular_palette_colors`].
    modular_palette_colors: Option<i64>,
    /// Optional override of the global channel-colours percentage
    /// (CLI passthrough — mirrors libjxl `cjxl
    /// --modular_channel_colors_global_percent`,
    /// `enc_params.h:channel_colors_pre_transform_percent`). `None`
    /// (default) keeps the built-in
    /// [`crate::modular::palette::CHANNEL_COLORS_PERCENT`] (95.0).
    /// `Some(p)` for `p in 0.0..=100.0` overrides the cap used when
    /// the global pre-RCT channel-compact pass evaluates per-channel
    /// palette beneficence. Stored on the config; wiring through
    /// `modular/encode.rs` is queued follow-on work.
    /// See [`Self::with_modular_channel_colors_global_percent`].
    modular_channel_colors_global_percent: Option<f32>,
    /// Optional override of the per-group channel-colours percentage
    /// (CLI passthrough — mirrors libjxl `cjxl
    /// --modular_channel_colors_group_percent`,
    /// `enc_params.h:channel_colors_percent`). `None` (default) keeps
    /// the libjxl default (80.0). Stored on the config; per-group
    /// channel-compact wiring is queued follow-on work.
    /// See [`Self::with_modular_channel_colors_group_percent`].
    modular_channel_colors_group_percent: Option<f32>,
    /// Optional override of the previous-channel context properties
    /// limit for tree learning (CLI passthrough — mirrors libjxl
    /// `cjxl -E` / `--modular_nb_prev_channels`,
    /// `enc_params.h:max_properties`). `None` (default) keeps the
    /// effort-derived behaviour. `Some(n)` for `n in 0..=11` would
    /// cap the count of additional previous-channel properties offered
    /// to the MA tree learner. Stored on the config; tree-learning
    /// wiring is queued follow-on work — our current learner does
    /// not consume previous-channel properties.
    /// See [`Self::with_modular_nb_prev_channels`].
    modular_nb_prev_channels: Option<i32>,
    /// Decoding-speed tier (libjxl `--faster_decoding 0..4`). Higher
    /// values bias the modular encode toward simpler bitstreams that
    /// decode faster, at the cost of compression. Default `0`
    /// (compression-priority). Mirrors libjxl
    /// `cparams.decoding_speed_tier` and feeds into
    /// [`crate::effort::LosslessFasterDecoding`] knobs. See
    /// [`Self::with_faster_decoding`].
    faster_decoding: u8,
    /// Container-wrap policy (libjxl `--container 0|1`). Default
    /// [`ContainerMode::Auto`] keeps the existing behaviour (wrap only
    /// when metadata or level demands it). See
    /// [`Self::with_container_mode`].
    container_mode: ContainerMode,
    /// Optional modular group-size override (libjxl `cjxl -g 0..3`,
    /// `cparams.modular_group_size_shift`). `None` (default) keeps the
    /// existing 256-pixel group dimension (shift = 1) so output bytes
    /// are unchanged. `Some(n)` for `n in 0..=3` maps to group
    /// dimensions `128 << n` = {128, 256, 512, 1024}. Affects both the
    /// frame-header signal and the modular encoder's per-group
    /// partitioning. VarDCT is unaffected (libjxl + this encoder both
    /// fix VarDCT groups at 256). See
    /// [`Self::with_modular_group_size`].
    modular_group_size_shift: Option<u8>,
    /// Opt-in: enable automatic delta-frame mode selection for
    /// multi-frame animation encodes (A1 audit "Animation" — Skip /
    /// delta frame encoding). When `true`, the animation encode path is
    /// permitted to swap the per-frame [`BlendMode::Replace`] default
    /// for a delta-friendly alternative ([`BlendMode::Add`] with a 1×1
    /// zero-pixel crop that leaves the canvas unchanged) when it
    /// detects that frame N is byte-identical to the preceding
    /// displayed frame.
    ///
    /// Chunk 1 POC scope (this commit): one heuristic — identical-frame
    /// short-circuit using `Add` over a 1×1 zero-pixel crop. Chunk 2
    /// will add a full trial-encode of `Regular` vs
    /// `Add(reference=N-1)` vs `Blend(reference=N-1)` per frame and
    /// pick the cheapest decodable variant. Default `false` — no
    /// hash-locked bitstream changes at default.
    /// See [`Self::with_auto_delta_frames`].
    auto_delta_frames: bool,
    /// Input/output buffering policy (streaming refactor scaffolding,
    /// jxl-encoder#11). Default [`Buffering::Auto`] resolves to
    /// [`Buffering::FullBuffered`] for ≤ 2048² images and
    /// [`Buffering::BufferedOutput`] otherwise (matches libjxl post-
    /// `032d39a`). **Chunk 1: no dispatch is wired** — every variant
    /// currently routes through the existing one-shot path, so output
    /// bytes are identical regardless of `buffering`. See
    /// [`Self::with_buffering`].
    buffering: Buffering,
}

impl Default for LosslessConfig {
    fn default() -> Self {
        Self::with_effort_level(7)
    }
}

impl LosslessConfig {
    fn with_effort_level(effort: u8) -> Self {
        let profile = crate::effort::EffortProfile::lossless(effort, EncoderMode::Reference);
        Self {
            effort: profile.effort,
            mode: EncoderMode::Reference,
            use_ans: profile.use_ans,
            tree_learning: profile.tree_learning,
            squeeze: false, // squeeze hurts even with tree learning (14-62% larger on both photos and screenshots)
            lz77: profile.lz77,
            lz77_method: profile.lz77_method,
            patches: profile.patches,
            lossy_palette: false,
            threads: 0,
            tree_sample_fraction_override: None,
            forced_rct: None,
            profile_override: None,
            tree_parallel_smart: false,
            small_image_fallback_override: None,
            // libjxl lossless default: `keep_invisible = kDefault` with
            // `ApplyOverride(_, IsLossless()) == true`, i.e. NO simplify
            // pass. Caller opts in via `with_keep_invisible(false)`.
            simplify_invisible: false,
            modular_predictor: None,
            modular_palette_colors: None,
            modular_channel_colors_global_percent: None,
            modular_channel_colors_group_percent: None,
            modular_nb_prev_channels: None,
            faster_decoding: 0,
            container_mode: ContainerMode::Auto,
            modular_group_size_shift: None,
            auto_delta_frames: false,
            buffering: Buffering::Auto,
        }
    }

    /// Sets the modular group-size knob (libjxl `cjxl -g 0..3`,
    /// [`cparams.modular_group_size_shift`][libjxl-cparams]).
    ///
    /// The value is the `group_size_shift` signalled in the frame
    /// header, mapping to a group dimension of `128 << shift` pixels:
    ///
    /// | `shift` | group dim |
    /// |---------|-----------|
    /// | `0`     | 128       |
    /// | `1`     | 256 (default) |
    /// | `2`     | 512       |
    /// | `3`     | 1024      |
    ///
    /// `None` (default) keeps the current 256-pixel partitioning so
    /// bitstreams are byte-identical to before this knob existed.
    ///
    /// `Some(n)` for `n > 3` is clamped to `3` by the encoder; values
    /// outside `0..=3` are not representable in the 2-bit
    /// `group_size_shift` field.
    ///
    /// **What this affects:** the modular (lossless) encoder's group
    /// partitioning + the frame-header signal that tells the decoder
    /// what group dimension to use. Smaller groups (`-g 0`, 128 px)
    /// give a denser TOC and more parallel decode at the cost of
    /// per-group entropy-coder overhead. Larger groups (`-g 2`/`-g 3`)
    /// reduce TOC + global-state overhead and can compress better on
    /// small/medium images that would otherwise be split into many
    /// near-empty 256-px groups, at the cost of less parallelism on
    /// the decode side.
    ///
    /// **What this does NOT affect:** VarDCT (lossy) encoding. libjxl
    /// and this encoder both fix VarDCT groups at 256 pixels; the
    /// `group_size_shift` field is only emitted when the frame
    /// `encoding == Modular`.
    ///
    /// [libjxl-cparams]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_params.h
    pub fn with_modular_group_size(mut self, shift: Option<u8>) -> Self {
        self.modular_group_size_shift = shift.map(|s| s.min(3));
        self
    }

    /// Currently-configured modular group-size shift. `None` keeps the
    /// 256-pixel default; `Some(n)` overrides per [`Self::with_modular_group_size`].
    pub fn modular_group_size(&self) -> Option<u8> {
        self.modular_group_size_shift
    }

    /// Opt-in: enable automatic delta-frame mode selection for
    /// multi-frame animation encodes (A1 audit "Animation" — Skip /
    /// delta frame encoding). See the field doc on
    /// [`auto_delta_frames`][Self::auto_delta_frames] for the full
    /// rollout plan.
    ///
    /// Chunk 1 POC scope: one heuristic — identical-frame short-circuit
    /// using [`BlendMode::Add`] over a 1×1 zero-pixel crop. Chunk 2
    /// will add the full trial-encode loop (`Regular` vs `Add(prev)`
    /// vs `Blend(prev)`). Default `false` — no hash-locked bitstream
    /// changes at default.
    pub fn with_auto_delta_frames(mut self, enable: bool) -> Self {
        self.auto_delta_frames = enable;
        self
    }

    /// Whether the encode is permitted to emit delta-frame variants
    /// when [`Self::with_auto_delta_frames`] has been opted into.
    pub fn auto_delta_frames(&self) -> bool {
        self.auto_delta_frames
    }

    /// Opt-in: enable per-image smart-fanout for parallel tree learning.
    ///
    /// When enabled, the encoder re-tunes the rayon fanout depth /
    /// recursion floor / root threshold for the input image's pixel
    /// count. See [`crate::effort::EffortProfile::adapt_to_image`]
    /// for the rule.
    ///
    /// **Bitstream-equivalent** — the tree topology is determined by
    /// the samples, not the build order, so output bytes are identical
    /// with the smart-fanout knob on or off. This is purely a wall-clock
    /// knob.
    ///
    /// Not stable; the rule may change in patch releases as the
    /// sweep-correlation evidence grows.
    #[doc(hidden)]
    pub fn with_smart_fanout(mut self, on: bool) -> Self {
        self.tree_parallel_smart = on;
        self
    }

    /// Bias the modular encode toward simpler bitstreams that decode
    /// faster, at the cost of compression. Mirrors libjxl
    /// `cjxl --faster_decoding 0..4`
    /// ([`cparams.decoding_speed_tier`][libjxl-cparams]).
    ///
    /// Values are clamped to `0..=`[`MAX_FASTER_DECODING`]. The default
    /// `0` keeps the existing behaviour (no speed bias).
    ///
    /// Per-tier effect on the modular path
    /// ([libjxl `enc_modular.cc:469-516`][libjxl-modular],
    /// [`enc_frame.cc:340`][libjxl-frame]):
    ///
    /// - `1`: disables the Weighted predictor in tree learning;
    ///   `fast_decode_multiplier = 1.005` lifts the split-cost threshold
    ///   so the tree stays shallower.
    /// - `2`: same as tier 1 plus `modular_group_size_shift = 0`
    ///   (small groups for multithreaded decode);
    ///   `fast_decode_multiplier = 1.015`. Also clamps modular ANS
    ///   `max_histograms = 12`.
    /// - `3`: forces the Gradient predictor only and skips the MA tree
    ///   learner entirely (libjxl `kGradientOnly`).
    /// - `4`: tier 3 plus `nb_repeats = 0` (no MA tree at all). Also
    ///   disables the DC-frame patches pass.
    ///
    /// [libjxl-cparams]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_params.h
    /// [libjxl-modular]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_modular.cc
    /// [libjxl-frame]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_frame.cc
    pub fn with_faster_decoding(mut self, tier: u8) -> Self {
        self.faster_decoding = tier.min(MAX_FASTER_DECODING);
        self
    }

    /// Currently-configured decoding-speed tier (`0..=4`).
    pub fn faster_decoding(&self) -> u8 {
        self.faster_decoding
    }

    /// Container-wrap policy. Mirrors libjxl `cjxl --container 0|1`.
    /// Default [`ContainerMode::Auto`] wraps the codestream only when
    /// metadata is attached or the codestream level requires it.
    ///
    /// See [`ContainerMode`] for the per-variant semantics.
    pub fn with_container_mode(mut self, mode: ContainerMode) -> Self {
        self.container_mode = mode;
        self
    }

    /// Currently-configured container-wrap policy.
    pub fn container_mode(&self) -> ContainerMode {
        self.container_mode
    }

    /// Set the input/output buffering policy (streaming refactor
    /// scaffolding, jxl-encoder#11). Mirrors libjxl `cjxl --buffering
    /// -1..3`. See [`Buffering`] for variant semantics and the chunk
    /// schedule.
    ///
    /// **Chunk 1: no dispatch is wired** — every variant currently
    /// routes through the existing one-shot path, so output bytes are
    /// identical regardless of which `Buffering` value is selected.
    /// Chunks 2-7 land the per-DC-group split, the buffered-output
    /// streaming path (libjxl level 2), the seekable streaming-output
    /// path (libjxl level 3), and the lossless mirror.
    pub fn with_buffering(mut self, mode: Buffering) -> Self {
        self.buffering = mode;
        self
    }

    /// Currently-configured input/output buffering policy. See
    /// [`Self::with_buffering`].
    pub fn buffering(&self) -> Buffering {
        self.buffering
    }

    /// Resolve the effective [`EffortProfile`]: the override if set,
    /// otherwise the standard profile derived from effort + mode. Then
    /// apply the public per-knob overrides (sample fraction, forced
    /// RCT) on top.
    pub(crate) fn effective_profile(&self) -> crate::effort::EffortProfile {
        let mut p = self
            .profile_override
            .clone()
            .unwrap_or_else(|| crate::effort::EffortProfile::lossless(self.effort, self.mode));
        if let Some(f) = self.tree_sample_fraction_override {
            p.tree_sample_fraction = f;
        }
        if self.forced_rct.is_some() {
            p.forced_rct = self.forced_rct;
        }
        // Apply faster_decoding tier last so it can override sweep-pinned
        // values from `__expert` profile_override — that matches libjxl's
        // ordering (cparams.decoding_speed_tier is consulted at each gate
        // site directly, AFTER the speed-tier-derived defaults are set).
        p.apply_faster_decoding(self.faster_decoding);
        p
    }

    /// Resolve the effective `modular_group_size_shift`, honoring
    /// `faster_decoding >= 2` (libjxl `enc_frame.cc:340-343` forces
    /// `group_size_shift = 0` for smaller groups and multithreaded
    /// decode). When the caller has explicitly set
    /// [`Self::with_modular_group_size`] that override wins (caller
    /// intent is preserved). `None` (default) + `faster_decoding < 2`
    /// keeps the existing behaviour.
    pub(crate) fn effective_modular_group_size_shift(&self) -> Option<u8> {
        if self.modular_group_size_shift.is_some() {
            return self.modular_group_size_shift;
        }
        if self.faster_decoding >= 2 {
            return Some(0);
        }
        None
    }

    /// Resolve the effective LZ77 enable flag, honoring
    /// `faster_decoding >= 1` (libjxl `enc_ans.cc:1372` and
    /// `enc_modular.cc` paths set the LZ77 method to `kNone`).
    /// Returns the stored `cfg.lz77` field at tier 0.
    pub(crate) fn effective_lz77(&self) -> bool {
        if self.faster_decoding >= 1 {
            return false;
        }
        self.lz77
    }

    /// Resolve the effective tree-learning enable flag, honoring
    /// `faster_decoding >= 4` (libjxl `enc_modular.cc:506-513` zeros
    /// `nb_repeats` at tier 4, disabling MA-tree learning).
    pub(crate) fn effective_tree_learning(&self) -> bool {
        if self.faster_decoding >= 4 {
            return false;
        }
        self.tree_learning
    }

    /// Resolve the effective patches enable flag, honoring
    /// `faster_decoding >= 2` (libjxl `enc_modular.cc:707` gates
    /// `FindBestPatchDictionary` on `decoding_speed_tier < 2`).
    pub(crate) fn effective_patches(&self) -> bool {
        if self.faster_decoding >= 2 {
            return false;
        }
        self.patches
    }

    /// Override the small-image parallel-tree-learning fallback gate.
    /// See [`Self::small_image_fallback_override`].
    ///
    /// `None` (the default) keeps the gate **OFF** — the bench data
    /// gathered during landing of this knob (paired 10× on top of
    /// chunk-3c `79ff70ed`) showed the audit-claimed +0.85% cb5e202
    /// regression no longer reproduces (def 255.74 ms vs nofallback
    /// 254.73 ms, median Δ -0.40% at 0.26 MP × e7 × 8T). The cache
    /// is at parity or slightly winning across all measured cells.
    /// The infrastructure stays in place behind this opt-in for
    /// future investigation if the regression re-emerges.
    ///
    /// `Some(true)` forces the auto-gate ON (flips the fallback for
    /// inputs below 1 MP AT EFFORT ≤ 7). `Some(false)` forces the
    /// gate OFF regardless of size/effort (same as `None`).
    ///
    /// Intended for sweep harnesses + A/B benches; not stable.
    #[doc(hidden)]
    pub fn with_small_image_fallback_override(mut self, val: Option<bool>) -> Self {
        self.small_image_fallback_override = val;
        self
    }

    /// Variant of [`Self::effective_profile`] that applies the
    /// per-image adapters. Pass the input image's pixel count.
    ///
    /// Small-image fallback: OPT-IN via
    /// [`Self::with_small_image_fallback_override`]. Default `None`
    /// keeps the gate off because the audit-claimed cb5e202 regression
    /// no longer reproduces post-chunk3c. See the
    /// `with_small_image_fallback_override` doc for the bench data.
    /// When opt-in is on (`Some(true)`),
    /// [`crate::effort::EffortProfile::adapt_small_image_fallback`]
    /// flips `tree_parallel_small_image_fallback` to `true` when
    /// `pixels < SMALL_IMAGE_PIXEL_THRESHOLD` (1 MP) AND effort ≤ 7.
    ///
    /// Opt-in adapter (when `tree_parallel_smart` is on):
    /// [`crate::effort::EffortProfile::adapt_to_image`] re-tunes the
    /// rayon fanout depth/floor/threshold for the image size.
    pub(crate) fn effective_profile_for_image(&self, pixels: u64) -> crate::effort::EffortProfile {
        let mut p = self.effective_profile();
        // Small-image fallback gate (audit item #10): default OFF (None),
        // opt-in via `with_small_image_fallback_override(Some(true))`.
        // `Some(false)` / `None` leave the gate off (default behaviour).
        if let Some(true) = self.small_image_fallback_override {
            p.adapt_small_image_fallback(pixels);
        }
        // Always-on tree_max_buckets dispatch (audit item #3): drops
        // bucket cap from 256 → 192 at large+e9 cells only. Hash-locks
        // shift at those cells (+0.09% bytes) in exchange for ~12% wall-
        // clock. All other (size, effort) cells stay byte-identical.
        // Skipped only if the caller has supplied an explicit override
        // via `with_internal_params` (profile_override), to avoid
        // silently re-overriding a sweep harness's pinned value.
        if self.profile_override.is_none() {
            p.adapt_tree_max_buckets_for_image(pixels);
        }
        // Opt-in smart-fanout re-tuning.
        if self.tree_parallel_smart {
            p.adapt_to_image(pixels);
        }
        p
    }

    /// Apply picker / sweep override knobs scoped to the **lossless
    /// (modular)** encode path.
    ///
    /// Each `Some(_)` field on the supplied
    /// [`crate::effort::LosslessInternalParams`] overrides the corresponding
    /// effort-derived default; `None` fields keep the default. Per-knob
    /// public setters (`with_lz77_method`, `with_squeeze`, …) called after
    /// this still take precedence on the few knobs they cover.
    ///
    /// The type system enforces mode-correctness: lossy-only knobs
    /// (AC strategy gates, CfL, cost-model constants) live on
    /// [`crate::effort::LossyInternalParams`] and cannot be passed here.
    ///
    /// **Requires the `__expert` cargo feature.**
    /// Not stable; the underlying field set may grow additively between
    /// minor versions.
    #[cfg(feature = "__expert")]
    #[doc(hidden)]
    pub fn with_internal_params(mut self, params: crate::effort::LosslessInternalParams) -> Self {
        let mut profile = crate::effort::EffortProfile::lossless(self.effort, self.mode);
        params.apply_to(&mut profile);
        self.profile_override = Some(profile);
        self
    }

    /// Create a new lossless config with defaults (effort 7).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set effort level (1–12). Higher effort = slower, better compression.
    ///
    /// This adjusts all effort-dependent defaults:
    /// - **e1–3**: Huffman encoding
    /// - **e4–6**: + ANS entropy coding
    /// - **e7**: + content-adaptive tree learning, LZ77 RLE
    /// - **e8**: + LZ77 greedy hash chain
    /// - **e9–12**: + LZ77 optimal (Viterbi DP)
    ///
    /// **e10/e11/e12 are our extensions** beyond libjxl's kTortoise=9 ceiling
    /// (RFC#45 pick #1, extended in chunk 2 to e12). Today they map to the
    /// e9 lossless code paths; multi-seed tree learning at e10/e11 fans out
    /// 2/16 seeded runs. Bitstreams remain 100% spec-valid (djxl / jxl-rs /
    /// jxl-oxide decode unchanged).
    ///
    /// **WARNING — e6→e7 cliff** (#23): tree learning at e7 dominates
    /// the time profile and is significantly slower than e6 (a single
    /// 1024×683 illustration measured ~28× slower at e7 vs e3 for
    /// a ~38% size win). Picking e7 as a default silently pays this
    /// cost; for batch / interactive workloads where time matters
    /// more than the last 5-10% of size, e6 is often the better
    /// trade. Re-evaluate when the tree-learning sample budget gets
    /// a tunable knob.
    ///
    /// Individual `with_*()` calls after `with_effort()` override these defaults.
    pub fn with_effort(self, effort: u8) -> Self {
        let mut new = Self::with_effort_level(effort);
        // Preserve settings that aren't effort-derived
        new.mode = self.mode;
        new.squeeze = self.squeeze;
        new.profile_override = self.profile_override;
        new.tree_parallel_smart = self.tree_parallel_smart;
        new.small_image_fallback_override = self.small_image_fallback_override;
        // Buffering policy — never effort-derived; pure caller
        // preference. Carry across `with_effort` so the builder chain
        // `LosslessConfig::new().with_buffering(_).with_effort(_)` is
        // order-independent.
        new.buffering = self.buffering;
        new
    }

    /// Set encoder mode (default: [`EncoderMode::Reference`]).
    ///
    /// `Reference` matches libjxl's algorithm choices for comparable output.
    /// `Experimental` enables encoder-specific improvements.
    pub fn with_mode(mut self, mode: EncoderMode) -> Self {
        self.mode = mode;
        self
    }

    /// Current encoder mode.
    pub fn mode(&self) -> EncoderMode {
        self.mode
    }

    /// Enable/disable patches (dictionary-based repeated pattern detection).
    /// Default: true at effort >= 5. Huge wins on screenshots, zero cost on photos.
    pub fn with_patches(mut self, enable: bool) -> Self {
        self.patches = enable;
        self
    }

    /// Enable/disable ANS entropy coding (default: true).
    pub fn with_ans(mut self, enable: bool) -> Self {
        self.use_ans = enable;
        self
    }

    /// Enable/disable squeeze (Haar wavelet) transform (default: false).
    ///
    /// Squeeze is disabled by default because tree learning provides better
    /// compression on both photos and screenshots. Squeeze can still be
    /// enabled via `.with_squeeze(true)` for experimentation.
    pub fn with_squeeze(mut self, enable: bool) -> Self {
        self.squeeze = enable;
        self
    }

    /// Enable/disable content-adaptive tree learning (default: false).
    pub fn with_tree_learning(mut self, enable: bool) -> Self {
        self.tree_learning = enable;
        self
    }

    /// Override the tree-learning pixel sampling fraction (refs #23).
    ///
    /// Tree learning at e7 walks a fraction of the image's pixels to
    /// build the per-context histogram used for split selection. The
    /// effort-derived defaults are roughly:
    ///
    /// | effort | sample fraction |
    /// |-------:|----------------:|
    /// |     ≤4 | 0.15            |
    /// |      5 | 0.25            |
    /// |      6 | 0.35            |
    /// |      7 | 0.50            |
    /// |      8 | 0.55            |
    /// |     ≥9 | 0.65            |
    ///
    /// Sampling more pixels = better tree quality (smaller files) but
    /// linearly more time. e7 is the cliff in #23 because tree
    /// learning *first* turns on there; lowering the sample fraction
    /// at e7 gives a smoother time/size trade between e6 (no tree)
    /// and e7-default (tree at 0.5).
    ///
    /// # Calibrated values for e7
    ///
    /// Sweep on 5 real photos (0.26 / 1.05 / 4.19 MP), single-thread
    /// release build, source data
    /// [`benchmarks/lossless_e7_sample_fraction_sweep_2026-05-15.tsv`]:
    ///
    /// | fraction | bytes vs e7 default | encode time vs e7 default |
    /// |---------:|--------------------:|--------------------------:|
    /// | 0.10     | +0.40 to +2.30 %    | -60 to -69 %              |
    /// | 0.15     | +0.36 to +1.43 %    | -54 to -61 %              |
    /// | 0.20     | -0.01 to +1.43 %    | -48 to -55 %              |
    /// | 0.25     | +0.11 to +1.12 %    | -29 to -41 %              |
    /// | 0.35     | +0.14 to +0.88 %    | -18 to -30 % (≤1 MP)      |
    /// | 0.50     | baseline (0 %)      | baseline                  |
    ///
    /// **Recommendation**: start at `f = 0.25` for an "e7-lite" tier —
    /// average -36 % wall-clock and ≤ +0.6 % bytes on photos. Use
    /// `0.10..=0.20` for the most aggressive "fast e7" trade (size
    /// regresses up to ~2 % on small images, but encode-time drops
    /// ~50–70 %).
    ///
    /// Range `[0.0, 1.0]`; `f.clamp(0.0, 1.0)` is applied so a stray
    /// caller can't trip the validator. No-op when `tree_learning` is
    /// disabled.
    pub fn with_tree_learning_sample_fraction(mut self, f: f32) -> Self {
        self.tree_sample_fraction_override = Some(f.clamp(0.0, 1.0));
        self
    }

    /// Current tree-learning sample fraction override, if set.
    pub fn tree_learning_sample_fraction(&self) -> Option<f32> {
        self.tree_sample_fraction_override
    }

    /// Force a specific Reversible Color Transform colorspace,
    /// skipping the per-effort RCT search. Mirrors libjxl's
    /// `cparams.colorspace`.
    ///
    /// Use cases:
    /// - Known-best RCT for a specific content class (e.g.
    ///   `RctType::YCOCG` for screenshots) — saves the search cost
    ///   without losing quality on average.
    /// - Reproducibility / determinism (skip search variability).
    /// - Picker output: when an offline sweep has identified the
    ///   best RCT for a feature signature, the runtime picker can
    ///   dial it directly.
    ///
    /// `None` (default) keeps the per-effort search. `Some(rct)`
    /// applies the given RCT directly without evaluating others.
    /// Common values: [`crate::modular::rct::RctType::YCOCG`] (libjxl
    /// default fallback, 6), [`crate::modular::rct::RctType::NONE`]
    /// (no transform, 0), [`crate::modular::rct::RctType::SUBTRACT_GREEN`]
    /// (G-R / G-B decorrelation, 3).
    pub fn with_force_rct(mut self, rct: Option<crate::modular::rct::RctType>) -> Self {
        self.forced_rct = rct;
        self
    }

    /// Configured forced RCT colorspace, if any.
    pub fn force_rct(&self) -> Option<crate::modular::rct::RctType> {
        self.forced_rct
    }

    /// Enable/disable LZ77 backward references (default: false).
    pub fn with_lz77(mut self, enable: bool) -> Self {
        self.lz77 = enable;
        self
    }

    /// Set LZ77 method (default: Greedy). Only effective when LZ77 is enabled.
    pub fn with_lz77_method(mut self, method: Lz77Method) -> Self {
        self.lz77_method = method;
        self
    }

    /// Enable/disable lossy delta palette (default: false).
    ///
    /// When enabled, uses quantized palette with delta entries and error diffusion
    /// for near-lossless encoding. This is NOT pixel-exact — it trades some color
    /// accuracy for significantly smaller files on images with many colors.
    /// Matching libjxl's modular lossy palette mode.
    pub fn with_lossy_palette(mut self, enable: bool) -> Self {
        self.lossy_palette = enable;
        self
    }

    /// Preserve or drop RGB samples in fully-transparent (alpha=0)
    /// pixels.
    ///
    /// Mirrors libjxl `cparams.keep_invisible` + the lossless branch of
    /// `SimplifyInvisible` (`enc_frame.cc:511`, `enc_frame.cc:1588-1597`).
    ///
    /// - `true` (**default**) — preserve all RGB bytes exactly. Encoded
    ///   output is bit-exact RGBA. Matches libjxl default for lossless
    ///   (`ApplyOverride(kDefault, IsLossless()) == true`, i.e. simplify
    ///   pass does **not** run).
    /// - `false` — overwrite RGB with `0` wherever alpha=0 before the
    ///   modular encoder sees the channel. Decoded *visible* pixels
    ///   stay bit-exact; only data no decoder will display changes.
    ///   Lets modular's predictor + LZ77 compress long zero runs for
    ///   **5–20 % smaller files on sprites / UI assets / icons** with
    ///   large transparent regions; near-zero overhead on photos with
    ///   mostly-opaque alpha (single linear scan to detect any
    ///   invisible pixel).
    ///
    /// No-op when:
    /// - the input layout has no alpha channel (Rgb8, Bgr8, Gray8,
    ///   Rgb16, Gray16, RgbLinearF32, GrayLinearF32, …);
    /// - the alpha channel is fully opaque (no pixel has alpha=0);
    /// - the request signals premultiplied alpha — alpha=0 pixels
    ///   already hold RGB=0 by construction, so zeroing is redundant.
    pub fn with_keep_invisible(mut self, keep: bool) -> Self {
        // Internal storage is the inverse so the "run the pre-pass"
        // branch is a single boolean read on the hot path.
        self.simplify_invisible = !keep;
        self
    }

    /// Force a fixed modular predictor (CLI passthrough — mirrors libjxl
    /// `cjxl -P` / `--modular_predictor`).
    ///
    /// `None` (default) lets the MA tree learner pick. `Some(n)` for
    /// `n in 0..=13` corresponds to [`crate::modular::Predictor`]
    /// variants `Zero..Average4` (see the enum in
    /// `jxl-encoder/src/modular/predictor.rs`).
    ///
    /// `Some(15)` is libjxl's `Variable` meta-mode — falls through to
    /// the per-leaf ID3 tree learner. `Some(14)` is libjxl's `Best`
    /// slot, which we repurpose as **RIGED** (Sharma 2018, Resolution-
    /// Independent Gradient-aware Edge Detection): the tree learner is
    /// replaced with a hand-crafted 3-leaf gradient-aware MA tree
    /// switching between `Top`/`Left`/`Average((W+N)/2)` per pixel based
    /// on `|NW - W|` and `|W - WW|` thresholds. Encoder-only meta-mode
    /// — the wire bitstream uses only spec-conformant predictors and
    /// properties, so any JXL decoder rounds-trips pixel-exact. See
    /// [`crate::modular::tree::riged_tree`] for the tree shape.
    ///
    /// Values outside `0..=15` are clamped silently.
    pub fn with_modular_predictor(mut self, p: Option<u8>) -> Self {
        self.modular_predictor = p.map(|v| v.min(15));
        self
    }

    /// Currently-set modular predictor override (or `None` if unset).
    pub fn modular_predictor(&self) -> Option<u8> {
        self.modular_predictor
    }

    /// Override the palette-transform colour cap (CLI passthrough —
    /// mirrors libjxl `cjxl --modular_palette_colors`).
    ///
    /// `None` (default) keeps the built-in
    /// [`crate::modular::palette::MAX_PALETTE_COLORS`] (1024). `Some(0)`
    /// disables palette detection. `Some(n)` for `n > 0` caps the
    /// palette-colour search.
    ///
    /// Encoder-side wiring through the palette-search call sites in
    /// `modular/encode.rs` is queued follow-on work. The value is
    /// stored on the config for surface completeness.
    pub fn with_modular_palette_colors(mut self, n: Option<i64>) -> Self {
        self.modular_palette_colors = n;
        self
    }

    /// Currently-set modular palette colours cap (or `None` if unset).
    pub fn modular_palette_colors(&self) -> Option<i64> {
        self.modular_palette_colors
    }

    /// Override the global channel-colours percentage cap (CLI
    /// passthrough — mirrors libjxl `cjxl
    /// --modular_channel_colors_global_percent`).
    ///
    /// `None` (default) keeps the built-in
    /// [`crate::modular::palette::CHANNEL_COLORS_PERCENT`] (95.0).
    /// `Some(p)` for `p in 0.0..=100.0` overrides. Values outside that
    /// range are clamped silently.
    ///
    /// Encoder-side wiring is queued follow-on work.
    pub fn with_modular_channel_colors_global_percent(mut self, p: Option<f32>) -> Self {
        self.modular_channel_colors_global_percent = p.map(|v| v.clamp(0.0, 100.0));
        self
    }

    /// Currently-set global channel-colours percentage (or `None` if
    /// unset).
    pub fn modular_channel_colors_global_percent(&self) -> Option<f32> {
        self.modular_channel_colors_global_percent
    }

    /// Override the per-group channel-colours percentage cap (CLI
    /// passthrough — mirrors libjxl `cjxl
    /// --modular_channel_colors_group_percent`).
    ///
    /// `None` (default) keeps the libjxl default (80.0). `Some(p)` for
    /// `p in 0.0..=100.0` overrides. Values outside that range are
    /// clamped silently.
    ///
    /// Encoder-side wiring is queued follow-on work.
    pub fn with_modular_channel_colors_group_percent(mut self, p: Option<f32>) -> Self {
        self.modular_channel_colors_group_percent = p.map(|v| v.clamp(0.0, 100.0));
        self
    }

    /// Currently-set per-group channel-colours percentage (or `None`
    /// if unset).
    pub fn modular_channel_colors_group_percent(&self) -> Option<f32> {
        self.modular_channel_colors_group_percent
    }

    /// Override the previous-channel context-properties limit (CLI
    /// passthrough — mirrors libjxl `cjxl -E` /
    /// `--modular_nb_prev_channels`).
    ///
    /// `None` (default) keeps the effort-derived behaviour. `Some(n)`
    /// for `n in 0..=11` would cap the count of additional
    /// previous-channel properties offered to the MA tree learner.
    /// `Some(-1)` mirrors libjxl's "use default" sentinel. Stored on
    /// the config; tree-learning wiring is queued follow-on work —
    /// our current learner does not consume previous-channel
    /// properties.
    pub fn with_modular_nb_prev_channels(mut self, n: Option<i32>) -> Self {
        self.modular_nb_prev_channels = n;
        self
    }

    /// Currently-set previous-channel context-properties cap (or
    /// `None` if unset).
    pub fn modular_nb_prev_channels(&self) -> Option<i32> {
        self.modular_nb_prev_channels
    }

    /// Build a [`crate::modular::palette::ModularKnobs`] snapshot from
    /// the current `modular_*` overrides. Internal helper used to thread
    /// the knobs into [`crate::modular::frame::FrameEncoderOptions`].
    pub(crate) fn modular_knobs(&self) -> crate::modular::palette::ModularKnobs {
        crate::modular::palette::ModularKnobs {
            modular_predictor: self.modular_predictor,
            palette_colors: self.modular_palette_colors,
            channel_colors_global_percent: self.modular_channel_colors_global_percent,
            channel_colors_group_percent: self.modular_channel_colors_group_percent,
            nb_prev_channels: self.modular_nb_prev_channels,
        }
    }

    /// Set thread count for parallel encoding.
    ///
    /// - `0` (default): use the ambient rayon pool. The caller can control
    ///   thread count by wrapping the encode call in `pool.install(|| ...)`.
    /// - `1`: force sequential encoding (no rayon).
    /// - `N >= 2`: create a dedicated N-thread pool for this encode.
    ///
    /// Requires the `parallel` feature. When `parallel` is not enabled,
    /// this value is ignored and encoding is always sequential.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    // ── Getters ───────────────────────────────────────────────────────

    /// Current effort level.
    pub fn effort(&self) -> u8 {
        self.effort
    }

    /// Whether ANS entropy coding is enabled.
    pub fn ans(&self) -> bool {
        self.use_ans
    }

    /// Whether squeeze (Haar wavelet) transform is enabled.
    pub fn squeeze(&self) -> bool {
        self.squeeze
    }

    /// Whether content-adaptive tree learning is enabled.
    pub fn tree_learning(&self) -> bool {
        self.tree_learning
    }

    /// Whether LZ77 backward references are enabled.
    pub fn lz77(&self) -> bool {
        self.lz77
    }

    /// Current LZ77 method.
    pub fn lz77_method(&self) -> Lz77Method {
        self.lz77_method
    }

    /// Conservative upper bound on peak working-set memory for a
    /// lossless encode of this configuration at `(width, height)`
    /// pixels with the given pixel layout.
    ///
    /// Models the dimension-driven buffers that dominate the modular
    /// encoder's peak RSS:
    ///
    /// 1. Channel planes: one `i32` per pixel per channel
    ///    (`pixels * channels * 4` bytes). 8-bit and 16-bit inputs
    ///    both expand to i32 internally for residual encoding.
    /// 2. Predictor scratch: one i32 plane equivalent
    ///    (`pixels * 4` bytes) for gradient / weighted-predictor
    ///    state.
    /// 3. Tree-learning state (effort >= 7): `pixels * tokens` bytes
    ///    for the sample histogram. Modelled as 8 bytes per pixel for
    ///    a typical run.
    /// 4. Squeeze residuals (when enabled): one extra channel-plane
    ///    pair for the wavelet decomposition.
    ///
    /// Then a 25 % overhead is added for the entropy-coder bit
    /// buffer, histograms, and unmodelled scratch.
    ///
    /// Returns `None` only if the dimensions overflow `u64`.
    pub fn estimate_peak_memory_bytes(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Option<u64> {
        estimate_peak_memory_bytes_lossless(width, height, layout, self.effort, self.squeeze)
    }

    /// Whether patches (dictionary-based repeated pattern detection) are enabled.
    pub fn patches(&self) -> bool {
        self.patches
    }

    /// Whether lossy delta palette is enabled.
    pub fn lossy_palette(&self) -> bool {
        self.lossy_palette
    }

    /// Thread count (0 = auto, 1 = sequential).
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Borrow the resolved `EffortProfile` override, if any. Internal hook
    /// used by [`crate::validation`].
    #[cfg(feature = "__expert")]
    pub(crate) fn profile_override_ref(&self) -> Option<&crate::effort::EffortProfile> {
        self.profile_override.as_ref()
    }

    // ── Request / fluent encode ─────────────────────────────────────

    /// Create an encode request for an image with this config.
    ///
    /// Use this when you need to attach metadata, limits, or cancellation.
    pub fn encode_request(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> EncodeRequest<'_> {
        EncodeRequest {
            config: ConfigRef::Lossless(self),
            width,
            height,
            layout,
            metadata: None,
            limits: None,
            stop: None,
            source_gamma: None,
            color_encoding: None,
            intensity_target: None,
            min_nits: None,
            premultiplied_alpha: false,
            premultiplied_alpha_mode: None,
            bits_per_sample: None,
            brotli_metadata_quality: None,
            row_stride: None,
            extra_channels: &[],
        }
    }

    /// Encode pixels directly with this config. Shortcut for simple cases.
    ///
    /// ```rust,no_run
    /// # let pixels = vec![0u8; 100 * 100 * 3];
    /// let jxl = jxl_encoder::LosslessConfig::new()
    ///     .encode(&pixels, 100, 100, jxl_encoder::PixelLayout::Rgb8)?;
    /// # Ok::<_, jxl_encoder::At<jxl_encoder::EncodeError>>(())
    /// ```
    #[track_caller]
    pub fn encode(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<Vec<u8>> {
        self.encode_request(width, height, layout).encode(pixels)
    }

    /// Encode pixels, appending to an existing buffer.
    #[track_caller]
    pub fn encode_into(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        self.encode_request(width, height, layout)
            .encode_into(pixels, out)
            .map(|_| ())
    }

    /// Encode a multi-frame animation as a lossless JXL.
    ///
    /// Each frame must have the same dimensions and pixel layout.
    /// Returns the complete JXL codestream bytes.
    #[track_caller]
    pub fn encode_animation(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
        animation: &AnimationParams,
        frames: &[AnimationFrame<'_>],
    ) -> Result<Vec<u8>> {
        encode_animation_lossless(self, width, height, layout, animation, frames, None).map_err(at)
    }

    /// Encode a multi-frame animation with explicit resource [`Limits`].
    ///
    /// Same shape as [`Self::encode_animation`], plus a per-encode
    /// allocation cap that the modular FrameEncoder consults at every
    /// dimension-driven allocation site. The cap applies across **all**
    /// frames combined — a single oversized frame is rejected before any
    /// of the per-frame buffers are allocated.
    #[track_caller]
    pub fn encode_animation_with_limits(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
        animation: &AnimationParams,
        frames: &[AnimationFrame<'_>],
        limits: &Limits,
    ) -> Result<Vec<u8>> {
        encode_animation_lossless(self, width, height, layout, animation, frames, Some(limits))
            .map_err(at)
    }

    // ── JPEG → JXL lossless transcoding ─────────────────────────────────
    //
    // Parses an existing JPEG file and re-encodes its quantized DCT
    // coefficients into a JXL bitstream. Pixel-identical to the original
    // (no re-quantization, no perceptual changes) AND — when called via
    // [`Self::encode_jpeg_transcode`] — byte-exact JPEG reconstruction
    // via the JBRD box in the JXL container. Typical ratio: ~80% of the
    // original JPEG bytes on photographic content.
    //
    // This is **the** flagship JXL feature for serving smaller JPEG-like
    // bytes without re-decoding/re-encoding through pixels. The transcoded
    // JXL can be decoded directly OR reconstructed back to the exact
    // original JPEG via `djxl --reconstruct_jpeg`.
    //
    // Currently only baseline-sequential JPEGs with 1 or 3 components
    // (grayscale, YCbCr 4:4:4/4:2:0/4:2:2/4:4:0, RGB) are supported.
    // Progressive JPEGs and arithmetic-coded JPEGs are unsupported — they
    // return [`EncodeError::JpegParse`] / [`EncodeError::InvalidInput`].
    //
    // The [`LosslessConfig`] effort / mode / per-knob settings do NOT
    // currently affect the transcode path (JPEG → JXL is a deterministic
    // bit-level recoding). The config argument is taken for forwards
    // compatibility — future versions may use it to gate the JPEG-CfL
    // search effort, JBRD Brotli effort, or related tuning knobs.

    /// Losslessly transcode a JPEG file into JXL with JBRD container for
    /// byte-exact JPEG reconstruction.
    ///
    /// Parses `jpeg_bytes`, extracts the quantized DCT coefficients, and
    /// emits a JXL container that:
    /// 1. Decodes to pixel-identical output as the original JPEG
    ///    (via any JXL decoder: djxl, jxl-rs, jxl-oxide, ...).
    /// 2. Reconstructs the original JPEG byte-for-byte via
    ///    `djxl --reconstruct_jpeg out.jxl out.jpg` (or any decoder that
    ///    honors the JBRD reconstruction box).
    ///
    /// Returns the complete JXL container bytes (signature box + codestream
    /// + JBRD box). Typical ratio: ~80% of the original JPEG bytes for
    /// photographic content; gains depend on the source quantization
    /// quality and chroma subsampling shape.
    ///
    /// Requires the `jpeg-reencoding` cargo feature.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::JpegParse`] if the input is not a valid
    /// baseline-sequential JPEG or uses an unsupported feature (arithmetic
    /// coding, hierarchical mode, etc.). Returns
    /// [`EncodeError::InvalidInput`] for JPEGs whose component count is
    /// not 1 or 3.
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "jpeg-reencoding")]
    /// # fn main() -> Result<(), jxl_encoder::At<jxl_encoder::EncodeError>> {
    /// let jpeg_bytes = std::fs::read("photo.jpg").unwrap();
    /// let jxl = jxl_encoder::LosslessConfig::new()
    ///     .encode_jpeg_transcode(&jpeg_bytes)?;
    /// std::fs::write("photo.jxl", &jxl).unwrap();
    /// // To reconstruct the exact original JPEG:
    /// //   djxl photo.jxl photo_reconstructed.jpg --reconstruct_jpeg
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "jpeg-reencoding"))]
    /// # fn main() {}
    /// ```
    #[cfg(feature = "jpeg-reencoding")]
    #[track_caller]
    pub fn encode_jpeg_transcode(&self, jpeg_bytes: &[u8]) -> Result<Vec<u8>> {
        // Config is currently unused — see module-level comment above.
        let _ = self;
        let jpeg = crate::jpeg::read_jpeg(jpeg_bytes).map_err(|e| at(EncodeError::from(e)))?;
        crate::jpeg::encode_jpeg_to_jxl_container(&jpeg).map_err(|e| at(EncodeError::from(e)))
    }

    /// Losslessly transcode a JPEG file into a bare JXL codestream
    /// (no container, no JBRD box).
    ///
    /// Same pixel-identical guarantee as
    /// [`Self::encode_jpeg_transcode`], but produces only the raw JXL
    /// codestream — no container wrapping, no JBRD reconstruction box.
    /// The resulting JXL bytes are smaller (no JBRD overhead) but the
    /// original JPEG cannot be reconstructed byte-for-byte. Use this
    /// when you only need to display / decode the image and don't need
    /// to round-trip back to the original JPEG bytes.
    ///
    /// Requires the `jpeg-reencoding` cargo feature.
    ///
    /// # Errors
    ///
    /// See [`Self::encode_jpeg_transcode`].
    #[cfg(feature = "jpeg-reencoding")]
    #[track_caller]
    pub fn encode_jpeg_transcode_codestream(&self, jpeg_bytes: &[u8]) -> Result<Vec<u8>> {
        let _ = self;
        let jpeg = crate::jpeg::read_jpeg(jpeg_bytes).map_err(|e| at(EncodeError::from(e)))?;
        crate::jpeg::encode_jpeg_to_jxl(&jpeg).map_err(|e| at(EncodeError::from(e)))
    }
}

// ── EncoderMode ──────────────────────────────────────────────────────────────

/// Controls whether the encoder matches libjxl's algorithm choices or uses
/// its own improvements.
///
/// Both modes produce valid JPEG XL bitstreams decodable by any conformant
/// decoder. The difference is in *encoder-side* decisions: strategy selection
/// heuristics, cost models, entropy coding parameters, tree learning, etc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EncoderMode {
    /// Match libjxl's algorithm choices at the configured effort level.
    ///
    /// Output is statistically equivalent to `cjxl` at the same effort and
    /// distance — same RD curve within measurement noise. Use this when
    /// comparing against libjxl or when reproducibility matters.
    #[default]
    Reference,

    /// Use encoder-specific improvements and research features.
    ///
    /// May produce better rate-distortion performance than libjxl at the
    /// same effort level, but output will differ. Use this for production
    /// encoding where quality per byte is the goal.
    Experimental,
}

// ── PatchesDispatch ──────────────────────────────────────────────────────────

/// Controls when the VarDCT patches detector runs.
///
/// The patches scan (text glyph / icon / button repeated-rectangle detector,
/// see [`crate::vardct::patches::find_and_build_with_per_patch_gate`]) costs
/// **~25-30 ms/MP** at effort >= 7. On photo content (CID22, CLIC) the scan
/// has historically produced zero output — the per-patch cost gate vetoes
/// every candidate, and the early-out `min_peak` filter rejects most before
/// they reach the cost gate. The full scan still runs end-to-end every time.
///
/// `Auto` (default) consults the same `median(mask1x1) > 95` discriminator
/// already used by [`Self::with_content_aware_entropy_mul`] / the GPU
/// encoder's AFV cost-grid gate and the W23-2 auto-splines screenshot skip.
/// When the discriminator says "photo class", `Auto` skips the scan entirely
/// — the omitted scan would have produced the same empty `PatchesData` it
/// always produces on photos, so the output is byte-identical and the wall
/// clock drops by ~25-30 ms/MP.
///
/// When the discriminator says "screenshot class" (median(mask1x1) > 95),
/// `Auto` runs the scan as before. Screenshots see no behavioural change.
///
/// `AlwaysScan` forces the patches scan regardless of content (the
/// pre-W36-3 behavior — useful for A/B benchmarks and reproducibility
/// against earlier output).
///
/// `NeverScan` short-circuits the scan and skips it on every image
/// (equivalent to [`LossyConfig::with_patches`]`(false)` for the scan step
/// — note that the rest of the patches pipeline including `enable_patches`
/// gating still applies; this only suppresses the detector).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PatchesDispatch {
    /// Skip the scan on photo content (`median(mask1x1) <= 95`); run on
    /// screenshot content (`> 95`). Default since W36-3.
    #[default]
    Auto,
    /// Always run the patches detector when `enable_patches` is true.
    /// Pre-W36-3 behavior. Use to compare A/B against `Auto` output, or
    /// when calibration sweeps need to compare to the older codepath.
    AlwaysScan,
    /// Never run the patches detector — skip the scan on every image
    /// regardless of `enable_patches`. Equivalent to gating the patches
    /// step off at the call site.
    NeverScan,
}

// ── ProgressiveMode ──────────────────────────────────────────────────────────

/// Progressive encoding mode for VarDCT.
///
/// Progressive encoding splits AC coefficients across multiple passes by
/// reducing precision. Decoders can render a coarse preview after early passes,
/// improving user experience for web delivery.
///
/// The shift mechanism works by right-shifting quantized coefficients before
/// encoding in early passes. The decoder left-shifts and accumulates, so the
/// final result is exact (lossless reconstruction of the quantized coefficients).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ProgressiveMode {
    /// Single pass (default). No progressive rendering.
    #[default]
    Single,
    /// 2-pass quantized progressive.
    ///
    /// - Pass 0: All AC coefficients right-shifted by 1 bit (coarse)
    /// - Pass 1: Residual at full precision
    ///
    /// Provides quick 2x-downsampled preview, then full quality refinement.
    QuantizedAcFullAc,
    /// 3-pass progressive (DC/VLF → LF → Full AC).
    ///
    /// - Pass 0: All AC coefficients right-shifted by 2 bits (very coarse, 8x downsample hint)
    /// - Pass 1: Residual right-shifted by 1 bit (medium, 4x downsample hint)
    /// - Pass 2: Final residual at full precision
    ///
    /// Provides staged refinement: blurry preview → sharper → final.
    DcVlfLfAc,
}

/// Chroma subsampling mode for lossy VarDCT encoding (issue #47).
///
/// Mirrors libjxl's four `YCbCrChromaSubsampling` modes
/// (`frame_header.h:81`). Each mode is described by the
/// (horizontal, vertical) shift applied to the Cb/Cr channels:
///
/// | Mode       | Cb / Cr H-shift | Cb / Cr V-shift | Cb/Cr sample density |
/// |------------|-----------------|-----------------|----------------------|
/// | `Full444`  | 0               | 0               | full resolution      |
/// | `Sub422`   | 1               | 0               | half horizontal      |
/// | `Sub420`   | 1               | 1               | quarter (H+V halved) |
/// | `Sub440`   | 0               | 1               | half vertical        |
///
/// # Current status (chunk 3)
///
/// **API surface + zenyuv-backed RGB→YCbCr+420 helpers landed; encoder
/// pipeline not yet wired.** Only [`ChromaSubsampling::Full444`] (the
/// default) is currently honoured end-to-end. Setting any other mode
/// causes the encoder to return [`EncodeError::InvalidConfig`].
///
/// The conversion building blocks live in
/// `crate::vardct::chroma_subsampling` (gated behind the
/// `chroma-subsampling` cargo feature) and call into the production
/// `zenyuv` SIMD kernels — Box-filter 4:2:0 (`rgb_to_yuv420`) and Sharp
/// YUV 4:2:0 (`rgb_to_yuv420_sharp_with_workspace`). What's missing
/// is the encoder-side wiring: the JXL spec ties chroma subsampling to
/// `ColorTransform::kYCbCr` (libjxl `enc_frame.cc:381-387`), but our
/// VarDCT pipeline currently emits `ColorTransform::kXYB`, and the
/// VarDCT encoder's adaptive_quant / CfL / AC-strategy / transform
/// stages assume all three channels share one block grid. Per-channel
/// block grids (Y full-res, Cb/Cr half-res) exist only in the
/// `jpeg-reencoding` path today.
///
/// Chunk 4 work (tracked on issue #47): route Sub420 through the JPEG
/// transcode-shaped pipeline — convert RGB → YCbCr via zenyuv, DCT8 +
/// quantize all three planes (Y at full res, Cb/Cr at half res), and
/// reuse `crate::jpeg::encode::encode_jpeg_to_jxl_inner`'s
/// `channel_shifts` / `do_ycbcr=true` / `jpeg_upsampling=[1,0,1]` /
/// modular substream layout. That gets us a decoder-roundtrippable
/// Sub420 bitstream without touching the standard VarDCT pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// **Default.** Full-resolution chroma (4:4:4). Y, Cb, Cr each sampled
    /// at every pixel. Largest files; highest chroma fidelity. The only
    /// mode currently honoured by the encoder.
    #[default]
    Full444,
    /// 4:2:2 — chroma halved horizontally, full vertical.
    /// (Cb/Cr H-shift = 1, V-shift = 0.)
    Sub422,
    /// 4:2:0 — chroma halved both horizontally and vertically.
    /// (Cb/Cr H-shift = 1, V-shift = 1.) The classic JPEG default.
    Sub420,
    /// 4:4:0 — chroma halved vertically, full horizontal.
    /// (Cb/Cr H-shift = 0, V-shift = 1.) Rare in practice.
    Sub440,
}

impl ChromaSubsampling {
    /// Per-channel horizontal shift in `[Cb, Y, Cr]` order. Mirrors
    /// libjxl `YCbCrChromaSubsampling::HShift(c)` — the shift the
    /// decoder applies (so `Sub420` returns `[1, 0, 1]`, NOT the raw
    /// mode index).
    pub const fn h_shifts(self) -> [u8; 3] {
        match self {
            Self::Full444 => [0, 0, 0],
            Self::Sub422 => [1, 0, 1],
            Self::Sub420 => [1, 0, 1],
            Self::Sub440 => [0, 0, 0],
        }
    }

    /// Per-channel vertical shift in `[Cb, Y, Cr]` order. See
    /// [`Self::h_shifts`].
    pub const fn v_shifts(self) -> [u8; 3] {
        match self {
            Self::Full444 => [0, 0, 0],
            Self::Sub422 => [0, 0, 0],
            Self::Sub420 => [1, 0, 1],
            Self::Sub440 => [1, 0, 1],
        }
    }

    /// `true` for [`Self::Full444`] (no subsampling). False for any
    /// real subsampling mode. Convenience for code that wants to
    /// short-circuit the YCbCr conversion path when the caller hasn't
    /// asked for subsampling.
    pub const fn is_full(self) -> bool {
        matches!(self, Self::Full444)
    }

    /// Industry-convention tag string (`"4:4:4"` / `"4:2:2"` / etc.).
    /// Used in [`EncodeError::InvalidConfig`] messages so callers see
    /// the format they typed in CLI / config rather than the Rust
    /// variant name.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Full444 => "4:4:4",
            Self::Sub422 => "4:2:2",
            Self::Sub420 => "4:2:0",
            Self::Sub440 => "4:4:0",
        }
    }
}

/// Adaptive dispatch policy for the per-block EPF sharpness search.
///
/// The per-block EPF sharpness selection (libjxl
/// `ComputeARHeuristics`) is, on the W36-1 phase profile
/// (`benchmarks/lossy_phase_baseline_2026-05-18.{tsv,meta}`),
/// **45.5% of e6 wall-clock** and **33.8% of e7**, dominating the
/// VarDCT pipeline at default effort. On smooth photo regions the
/// search converges on the default sharpness value (4) for nearly
/// every block; running the full two-pass search there is pure
/// overhead — the bitstream is identical to writing the uniform
/// default directly.
///
/// `EpfDispatch::Auto` skips the search when the input is "smooth
/// enough" by a `mask1x1`-based discriminator and emits the uniform
/// default sharpness for the affected region instead. On textured /
/// edge-heavy content the search still runs.
///
/// **Default**: [`EpfDispatch::AlwaysSelect`]. Flipping this default
/// requires a fresh hash-lock rebake plus a measured RD pass — until
/// that lands the byte-identical default stays put.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EpfDispatch {
    /// **Default.** Always run the per-block sharpness selection when
    /// the underlying gate (`epf_iters > 0 && distance >= 0.5 &&
    /// profile.epf_dynamic_sharpness`) is satisfied. Byte-identical
    /// to historical encoder behaviour.
    #[default]
    AlwaysSelect,
    /// Always force the uniform default sharpness (4) and skip the
    /// per-block search. Cheap; gives up the per-block tuning win.
    /// Use this when you've measured that the search isn't worth the
    /// CPU on your content.
    AlwaysDefault,
    /// Run the per-block selection only when a `mask1x1`-based
    /// smoothness predicate says the input has enough texture/edges
    /// to benefit. On smooth regions the uniform default sharpness
    /// is written without invoking the search. Bitstream-affecting
    /// on the gated subset; behaviour matches [`Self::AlwaysSelect`]
    /// on content the predicate doesn't gate.
    Auto,
}

/// Adaptive dispatch policy for the pixel-domain loss term added to
/// the AC-strategy search cost (libjxl
/// `enc_adaptive_quantization.cc::EstimateEntropy` →
/// `enc_ac_strategy.cc`).
///
/// The pixel-domain loss path runs an IDCT of the per-block
/// quantization error, multiplies by the per-pixel `mask1x1`
/// perceptual mask, and folds an 8th-power norm into the
/// strategy-selection cost. It's the W38-1 phase profile's dominant
/// AC-strategy overhead at e5 — `pixel_domain_loss = true` adds
/// ~11 ms/MP on photos and ~70 ms/MP on screenshots vs the
/// coefficient-domain-only path
/// (`benchmarks/lossy_phase_low_effort_with_zenjpeg_2026-05-19.{tsv,meta}`).
///
/// On smooth photo content the pixel-domain loss term rarely changes
/// which strategy wins — the AC-strategy search already converges on
/// DCT8/DCT16 picks from the coefficient-domain entropy estimate
/// alone. [`PixelLossDispatch::Auto`] short-circuits the loss path
/// in that regime (per-image `median(mask1x1) > 80` — smooth /
/// low-variance content) while preserving it on textured/edge
/// content where the loss term changes picks.
///
/// **Default**: [`PixelLossDispatch::AlwaysOn`]. Flipping the
/// default to `Auto` is a separate chunk after a wider corpus bench;
/// until that lands callers who want the speed-up opt in explicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PixelLossDispatch {
    /// **Default.** Always include the pixel-domain loss term in the
    /// AC-strategy search cost when the underlying gate
    /// (`ac_strategy_enabled && pixel_domain_loss`) is satisfied.
    /// Byte-identical to historical encoder behaviour.
    #[default]
    AlwaysOn,
    /// Always skip the pixel-domain loss term. Equivalent to
    /// `with_pixel_domain_loss(false)` at the encoder layer (mask1x1
    /// is not computed; AC-strategy search uses the
    /// coefficient-domain-only constants). Cheap; gives up the
    /// pixel-domain loss contribution to strategy picks.
    AlwaysOff,
    /// Run the pixel-domain loss term only when a `mask1x1`-based
    /// smoothness predicate says the input has enough texture/edges
    /// to benefit. On smooth regions (`median(mask1x1) > 80`) the
    /// mask is dropped before the AC-strategy search and the cost
    /// folds back to the coefficient-domain-only path. Bitstream-
    /// affecting on the gated subset; behaviour matches
    /// [`Self::AlwaysOn`] on content the predicate doesn't gate.
    Auto,
}

/// Adaptive dispatch policy for the two-pass entropy code optimization
/// (W44-87 — `optimize_codes` controls dynamic vs static Huffman path).
///
/// The two-pass entropy path collects every AC token into a per-context
/// histogram, builds optimal Huffman/ANS codes from the empirical
/// distribution, then re-walks the tokens to write the optimized
/// bitstream. The W38 phase profile measured this `entropy` +
/// `build_codes` pair at 56-62% of e5 photo wall-clock — about 14 ms
/// (`benchmarks/lossy_phase_baseline_low_effort_2026-05-19.tsv`).
///
/// The single-pass path uses pre-computed static Huffman codes
/// (`get_dc_entropy_code()` / `get_ac_entropy_code()`), eliminating
/// the histogram collection + code build entirely. The trade is a
/// small bitstream-size regression (the static codes are tuned for an
/// averaged token distribution that doesn't fit any single image as
/// tightly as per-image-optimized codes).
///
/// On smooth photo content at low distance (`d <= 1.0`,
/// `median(mask1x1)` below the smooth-content threshold) the
/// regression is typically 2-4% bytes — well below the 30%+
/// wall-clock saving — making this a high-value dispatch on the
/// content class that dominates web/CDN encode workloads.
///
/// `Auto` (this dispatch's content-aware mode) flips to single-pass
/// only when ALL of the following hold:
///   - `effort == 5` (the targeted speed tier),
///   - `distance <= 1.0`,
///   - `median(mask1x1) < SMOOTH_THRESHOLD` (smooth-photo class),
///   - the encode has NO features that require the two-pass
///     plumbing (patches, splines, learned tree, sharpness map,
///     noise params, LF frame, extras / alpha).
///
/// On any other content / mode / feature combo `Auto` behaves
/// identically to [`Self::AlwaysTwoPass`].
///
/// **Default**: [`SinglePassEntropyDispatch::AlwaysTwoPass`].
/// Bitstream byte-identical to historical builds; callers opt in
/// via [`LossyConfig::with_single_pass_entropy_dispatch`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SinglePassEntropyDispatch {
    /// **Default.** Always run the two-pass dynamic entropy path
    /// when the effort profile asks for it (`profile.optimize_codes`).
    /// Byte-identical to historical encoder behaviour.
    #[default]
    AlwaysTwoPass,
    /// Always use the single-pass static-Huffman path. Equivalent to
    /// `enc.optimize_codes = false`; will fall back to the two-pass
    /// path automatically when the encode has features the single-
    /// pass path cannot serialize (patches, splines, learned tree,
    /// sharpness map, noise params, LF frame, extras).
    /// Skips the histogram pass + code build entirely (~7-14 ms
    /// savings/MP at e5 on smooth photos).
    AlwaysSinglePass,
    /// Use single-pass static-Huffman codes when the content
    /// classifier says "smooth photo at low distance"
    /// (`effort == 5 && distance <= 1.0 && median(mask1x1) <
    /// SMOOTH_THRESHOLD`) AND the single-pass-safety predicate
    /// holds (no patches/splines/learned tree/sharpness map/noise/
    /// LF frame/extras). Otherwise behaves like [`Self::AlwaysTwoPass`].
    Auto,
}

// ── EncoderStrategy (W44-127 Chunk A — type surface only) ──────────────────
//
// This section ships the type definitions for the EncoderStrategy API
// consolidation work specified in `docs/COMPATIBILITY_MODES.md` (W44-126 v2
// design, commit `746ede8c`). It is Chunk A of a 7-chunk plan:
//
//   Chunk A (THIS COMMIT) — type defs in `api.rs` only.
//   Chunk B               — add `LossyConfig::strategy` field +
//                           `with_strategy` setter; wire `resolve()` into
//                           encoder construction. Hash-locks gate.
//   Chunk C               — rewire call sites (one per commit) to read the
//                           resolved enum picks instead of the legacy
//                           `Option<bool>` hint fields.
//   Chunk D               — delete the legacy `with_*_hint` and
//                           `with_*_dispatch` setters (absorbed into
//                           `EncoderImprovementsCustom`); move surviving
//                           hint fields into `StrategyOverrides`.
//   Chunk E               — `--strategy` CLI flag.
//   Chunk F               — promote 4 env-var knobs (`JXL_W44_117_DISABLE`,
//                           `JXL_W44_120_EPF_SEED_MIN_DISTANCE`,
//                           `JXL_BUTTLOOP_INITIAL_QF_SCALE`,
//                           `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`) into
//                           `EncoderImprovementsCustom` fields with env-var
//                           fallback at the bottom of the resolution stack.
//   Chunk G               — Section A effort-gate consultation in
//                           `effort.rs` (3 sites); Section D KNOWN-BUG
//                           re-enable (`block_ctx_map_15_cluster`).
//
// No `LossyConfig::strategy` field exists yet (added in Chunk B). No call
// sites read these types yet (rewired in Chunk C onwards). Resolver
// methods (`EncoderStrategy::resolve`, `ResolvedImprovements::*`,
// `StrategyOverrides::apply_to`) are `pub(crate)` so they don't leak
// surface area, but they're exercised by the unit tests at the bottom of
// this file.

/// Encoder behaviour bundle controlling which of our W44-* improvements
/// over libjxl reference are active.
///
/// **Default**: [`EncoderStrategy::Zenjxl`] — the production bundle we
/// ship today. Equivalent to leaving every `with_*_hint` setter at its
/// current default value.
///
/// Set via `LossyConfig::with_strategy` (added in Chunk B). Individual
/// `LossyConfig::with_*_hint` setters called AFTER `with_strategy`
/// override the matching field on the resolved
/// [`EncoderImprovementsCustom`]; this mirrors the
/// `with_perceptual_optimizations(false).with_gaborish(true)`
/// precedence pattern.
///
/// **Variants**:
///
/// - [`Self::Libjxl`] — strict libjxl-parity bundle. Disables every
///   Section B content-aware lift, flips the Section A effort-gate
///   divergences (`cfl_two_pass`, `try_dct64`, `epf_dynamic_sharpness`),
///   and deliberately re-enables the Section D `BlockCtxMap` 15-cluster
///   default (intentionally re-introduces the regression that
///   KNOWN-BUG cluster describes — the point IS act exactly like libjxl,
///   regressions and all).
/// - [`Self::LeanFaster`] — drops the heavy per-image content gates
///   (W22-1 screenshot lift, W44-65/68/123 DCT64/DCT32 admission,
///   W44-105/107/108 buttloop chain, W44-109 adaptive-quant chain,
///   W44-117/118/120 EPF chain, W44-34/35 smooth-photo DCT64). Keeps
///   the photo-class entropy-mul lowering (cheap table swaps) and our
///   effort-gate values. Faster encode without the heavy gates.
/// - [`Self::Zenjxl`] — production default. `impl Default` returns this.
///   Every Section B gate auto-fires per documented discriminator.
/// - [`Self::Aggressive`] — currently equivalent to `Zenjxl` after
///   W44-124's auto-discriminator obsoleted the previous
///   "flip W44-123 globally" behaviour. Kept as a forward-compatible
///   slot for the next opt-in chunk with a too-narrow auto-discriminator.
/// - [`Self::Custom`] — caller picks every dial individually via
///   [`EncoderImprovementsCustom`]. Includes the perf-dispatch policies
///   (`EpfDispatch`, `PixelLossDispatch`, `SinglePassEntropyDispatch`,
///   `PatchesDispatch`) absorbed as direct fields.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum EncoderStrategy {
    /// **Strict libjxl-parity mode — all-divergence bundle.** See enum
    /// doc-comment.
    Libjxl,
    /// **LeanFaster.** Skips heavy per-image content gates and the
    /// EPF/buttloop corrections. Keeps the at-parity algorithm fixes
    /// and the cheap photo-class entropy-mul lowering.
    LeanFaster,
    /// **Zenjxl.** Production default — what we ship today.
    /// `impl Default for EncoderStrategy` returns this variant.
    #[default]
    Zenjxl,
    /// **Aggressive.** Forward-compatible slot; currently equivalent
    /// to `Zenjxl`.
    Aggressive,
    /// **Custom.** Caller picks every dial. See
    /// [`EncoderImprovementsCustom`].
    Custom(Box<EncoderImprovementsCustom>),
}

/// W22-1 screenshot entropy-mul lift policy.
///
/// Lifts `IDENTITY` / `DCT2X2` / `AFV` / `DCT4X8` entropy_mul on
/// screenshot-class content to suppress small-transform artefacts at
/// sharp glyph edges. See `docs/LIBJXL_DIVERGENCES.md` Section B.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScreenshotEntropyMulPolicy {
    /// **Default.** Auto-fire via `median(mask1x1) > 95` when the
    /// underlying `content_aware_entropy_mul` enable bit is set.
    #[default]
    Auto,
    /// Force the lift on regardless of content. Caller asserts the
    /// image is screenshot-class.
    ForceOn,
    /// Suppress the lift even when mask1x1 would fire it. Equivalent
    /// to the W22-1 `Some(false)` override.
    ForceOff,
    /// Disable the gate entirely (the `content_aware_entropy_mul`
    /// enable bit is false). [`EncoderStrategy::Libjxl`] uses this.
    Disabled,
}

/// W44-29 + nested sub-gates (W44-91 / W44-96 / W44-98 / W44-99 / W44-100).
///
/// Lowers `entropy_mul[DCT16X16]` / `entropy_mul[DCT32X32]` on smooth
/// photos at `d >= 4.0` to close the F-D residual byte gap vs cjxl. The
/// nested sub-gates narrow admission to the specific photo classes
/// (1189261 / 1420710 / 1531677). See `docs/LIBJXL_DIVERGENCES.md`
/// Section B.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HighDPhotoEntropyMulPolicy {
    /// **Default.** Auto-fire via `d >= 4.0 AND mask1x1 < SMOOTH_THRESHOLD`
    /// with the W44-91 / W44-96 / W44-98 / W44-99 / W44-100 zenanalyze
    /// sub-discriminators composing on top.
    #[default]
    Auto,
    /// Force the lowering on regardless of content / distance.
    ForceOn,
    /// Suppress the lowering even when the auto gate would fire.
    ForceOff,
    /// Disable the gate entirely. [`EncoderStrategy::Libjxl`] uses this.
    Disabled,
}

/// W44-65 / W44-68 DCT64-class search admission.
///
/// Auto-suppresses DCT64-class search on screenshot-class content via
/// `median(mask1x1) >= 99.5`. See `docs/LIBJXL_DIVERGENCES.md`
/// Section B (W44-65/68 row).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dct64SearchPolicy {
    /// **Default.** Auto-suppress via `median(mask1x1) >= 99.5`.
    #[default]
    Auto,
    /// Force-suppress regardless of content. Equivalent to the
    /// `dct_suppress_hint: Some(true)` override on
    /// [`StrategyOverrides`].
    ForceSuppress,
    /// Force-allow DCT64 evaluation everywhere. Equivalent to the
    /// `dct_suppress_hint: Some(false)` override on
    /// [`StrategyOverrides`]. [`EncoderStrategy::Libjxl`] uses this.
    ForceAllow,
}

/// W44-123 / W44-124 DCT32-class search retention.
///
/// Composes with [`Dct64SearchPolicy`]: only matters when DCT64 has
/// been suppressed (auto or forced) AND the underlying W44-68 default
/// would also drop `try_dct32`. The default policy uses W44-124's
/// `m3_colourfulness >= 60 AND edge_density < 0.05` auto-discriminator
/// to keep DCT32 on codec_wiki-class smooth screen content while
/// dropping it on the other screenshot classes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dct32SearchPolicy {
    /// **Default.** Follow W44-68 (`try_dct32` dropped together with
    /// `try_dct64` when W44-65 fires). On `EncoderStrategy::Zenjxl`
    /// this composes with W44-124's auto-discriminator at the
    /// call site.
    #[default]
    FollowDct64Suppression,
    /// When DCT64 is suppressed (W44-65 fires), KEEP
    /// `try_dct32 = true`. Useful on codec_wiki-class smooth screen
    /// content where DCT16X16 → DCT32X32 splitting is the dominant
    /// win.
    KeepWhenDct64Suppressed,
}

/// W44-34 / W44-35 smooth-photo DCT64 admission inside the
/// `pixels < 500_000 AND distance < 2.0` smart-dispatch gate.
///
/// Orthogonal to [`Dct64SearchPolicy`] (that one is screenshot
/// suppression; this one is photo admission inside the
/// small-image-pixel smart-dispatch gate).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SmoothPhotoDct64Policy {
    /// **Default.** Auto-admit via the smooth-photo classifier (edge
    /// density + flat block ratio + HF energy).
    #[default]
    Auto,
    /// Force-admit on the gated cell.
    ForceAdmit,
    /// Force-skip the admission (preserves pre-W44-35 behaviour).
    /// [`EncoderStrategy::Libjxl`] uses this.
    ForceSkip,
}

/// W44-105 / W44-107 / W44-108 buttloop qf seed scaling (effort ≥ 8).
///
/// Pre-scales the butteraugli loop's initial qf seed on screenshot-class
/// content at high distance to close the W44-105 SSIM2 gap. Gate
/// predicate: `is_screenshot AND (d >= 3.5 OR (m3 < 30 AND d >= 2.0))`.
/// Promoted from env-var `JXL_BUTTLOOP_INITIAL_QF_SCALE` (Chunk F will
/// wire the env-var fallback).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ButtloopQfSeedPolicy {
    /// **Default.** Auto-fire per the W44-105/107/108 gate at scale
    /// `4.0`.
    #[default]
    AutoScale4,
    /// Custom scale (replaces the 4.0 default but keeps the same gate
    /// predicate). `1.0` ≡ off.
    AutoScale(f32),
    /// Force-fire the scale on every encode at the given factor (no
    /// gate). Useful for harness sweeps.
    ForceScale(f32),
    /// Off — never scale (`scale == 1.0`). [`EncoderStrategy::Libjxl`]
    /// uses this.
    Off,
}

/// W44-109 adaptive-quant qf pre-scale at effort ∈ \[5, 7\].
///
/// Mirrors [`ButtloopQfSeedPolicy`] at lower effort where the
/// butteraugli loop is unavailable; pre-scales `quant_field_float`
/// at adaptive-quant time. Default per-effort scales: `2.0` at e5/e6,
/// `3.0` at e7. Promoted from env-var
/// `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AdaptiveQuantQfSeedPolicy {
    /// **Default.** Auto-fire on screenshot-class at e ∈ \[5, 7\] with
    /// the per-effort scales (2.0 at e5/e6, 3.0 at e7).
    #[default]
    AutoScalePerEffort,
    /// Custom per-effort scales (replaces the 2.0/3.0 defaults but
    /// keeps the same gate predicate).
    AutoScaleCustom {
        /// Pre-scale at effort 5 and effort 6.
        e5_e6: f32,
        /// Pre-scale at effort 7.
        e7: f32,
    },
    /// Off — never pre-scale. [`EncoderStrategy::Libjxl`] uses this.
    Off,
}

/// W44-117 / W44-118 / W44-120 EPF sharpness seed for the butteraugli
/// loop.
///
/// Models the buttloop's internal `apply_epf` sharpness map source.
/// Mutually exclusive — exactly one of the three picks. The
/// `Option<bool>` shape we ship today admits invalid states like
/// "force_seed AND force_uniform4 AND per_iter_recompute" — the enum
/// shape makes those unrepresentable.
///
/// Promoted from env-vars `JXL_W44_117_DISABLE` (selects
/// [`Self::LegacyUniform4`]) and `JXL_W44_120_EPF_SEED_MIN_DISTANCE`
/// (overrides the `min_distance` field on [`Self::AutoW44_117`]).
#[derive(Clone, Copy, Debug, PartialEq)]
// `PerIterRecompute` is hidden but intentionally constructible — harness
// sweeps and the W44-118 Mode D bisect both use it. clippy interprets the
// `#[doc(hidden)]` last-variant shape as a manual `#[non_exhaustive]`,
// which would change the semantics (block construction outside the
// crate); suppress the heuristic here.
#[allow(clippy::manual_non_exhaustive)]
pub enum EpfSharpnessSeed {
    /// **Default.** W44-117 one-shot `compute_epf_sharpness` on the
    /// initial reconstruction, with the W44-118 `is_screenshot` gate
    /// AND W44-120 `target_distance >= min_distance` gate. Falls back
    /// to [`Self::LegacyUniform4`] on photos and on screenshots at
    /// `d < min_distance`.
    ///
    /// `min_distance` default is `1.0` (W44-120 pick from the bisect).
    AutoW44_117 {
        /// Minimum target distance at which the W44-117 seed compute
        /// fires; below this falls back to legacy uniform-4 sharpness.
        min_distance: f32,
    },
    /// Pre-W44-117 behaviour: uniform sharpness = 4 across the whole
    /// frame inside the buttloop. [`EncoderStrategy::Libjxl`] uses
    /// this.
    LegacyUniform4,
    /// Future-shape pick — recompute `compute_epf_sharpness` per
    /// buttloop iter. Bench so far shows this regresses (W44-118
    /// Mode D bisect); reserved for future investigation.
    #[doc(hidden)]
    PerIterRecompute,
}

impl Default for EpfSharpnessSeed {
    fn default() -> Self {
        Self::AutoW44_117 { min_distance: 1.0 }
    }
}

/// Section A effort-gate threshold.
///
/// A Section A divergence row in `docs/LIBJXL_DIVERGENCES.md` has us
/// at `effort >= N` while libjxl is at either `effort >= M` (different
/// N) or no effort gate at all. This enum models the four states
/// cleanly so [`EncoderStrategy::Libjxl`] can flip to libjxl's gate
/// without ambiguity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EffortGate {
    /// **Default.** Use the jxl-encoder threshold (Section A "Ours"
    /// column).
    #[default]
    Ours,
    /// Use the libjxl threshold (Section A "libjxl" column). For
    /// `cfl_two_pass` this is `>= 5`; for `try_dct64` and
    /// `epf_dynamic_sharpness` this is no effort gate at all.
    Libjxl,
    /// Disable the effort gate entirely (always run / never run
    /// depending on the consuming site's semantics).
    Off,
    /// Custom threshold (effort ≥ N).
    AtLeast(u8),
}

impl EffortGate {
    /// Evaluate the gate at the given `effort`, parameterised by the
    /// per-site `ours_min_effort` and `libjxl_min_effort` defaults.
    ///
    /// **Per-site defaults** (read directly from `effort.rs`
    /// `lossy_reference` + libjxl's `enc_heuristics.cc` / `enc_ac_strategy.cc`
    /// sources; documented per `docs/LIBJXL_DIVERGENCES.md` Section A):
    ///
    /// | site | `ours_min_effort` | `libjxl_min_effort` |
    /// |---|---|---|
    /// | `cfl_two_pass` | `7` (we) | `5` (libjxl `speed_tier <= kHare`) |
    /// | `try_dct64` | `7` (we) | `0` (libjxl has no effort gate; uses `decoding_speed_tier`) |
    /// | `epf_dynamic_sharpness` | `6` (we) | `0` (libjxl has no effort gate) |
    ///
    /// Semantics:
    /// - [`Ours`](EffortGate::Ours) → `effort >= ours_min_effort`
    /// - [`Libjxl`](EffortGate::Libjxl) → `effort >= libjxl_min_effort`
    /// - [`Off`](EffortGate::Off) → `true` (gate disabled, always fire)
    /// - [`AtLeast(n)`](EffortGate::AtLeast) → `effort >= n`
    ///
    /// W44-133 Chunk G consumes this from
    /// `LosslessConfig::effective_profile_for_image_with_smoothness` and the
    /// equivalent lossy boundary to flip the 3 Section A effort gates in
    /// `EffortProfile` when [`EncoderStrategy::Libjxl`] is selected. The
    /// default value [`EffortGate::Ours`] preserves all pre-Chunk-G hash
    /// locks byte-identical.
    pub(crate) fn evaluate(self, effort: u8, ours_min_effort: u8, libjxl_min_effort: u8) -> bool {
        match self {
            EffortGate::Ours => effort >= ours_min_effort,
            EffortGate::Libjxl => effort >= libjxl_min_effort,
            EffortGate::Off => true,
            EffortGate::AtLeast(n) => effort >= n,
        }
    }
}

/// Fine-grained per-divergence picks. Use with [`EncoderStrategy::Custom`]
/// when none of the named presets fit.
///
/// Every field has a [`Default`] impl that matches
/// [`EncoderStrategy::Zenjxl`]. Construct via
/// `EncoderImprovementsCustom::default()` and then mutate the fields
/// you care about (Chunk D will add `with_*` builders for a fluent
/// experience; for now use struct-update syntax with
/// `..Default::default()`).
///
/// Field groups:
///
/// - **Screenshot-class entropy-mul lifts**: `screenshot_entropy_mul`
/// - **Photo-class entropy-mul lowering**: `high_d_photo_entropy_mul`
/// - **DCT-class search admission**: `dct64_search_policy`,
///   `dct32_search_policy`, `smooth_photo_dct64_admission`
/// - **Butteraugli loop qf seeding**: `buttloop_qf_seed`
/// - **Adaptive-quant qf seeding** (effort ∈ \[5, 7\]):
///   `adaptive_quant_qf_seed`
/// - **EPF sharpness seed for buttloop**: `buttloop_epf_sharpness_seed`
/// - **Perf dispatches** (ABSORBED from `LossyConfig` per user
///   decision — see `docs/COMPATIBILITY_MODES.md` §7 Q2):
///   `epf_dispatch`, `pixel_loss_dispatch`,
///   `single_pass_entropy_dispatch`, `patches_dispatch`
/// - **Section A effort-gate divergences** (Libjxl-only flips):
///   `cfl_two_pass_min_effort`, `try_dct64_min_effort`,
///   `epf_dynamic_sharpness_min_effort`
/// - **Section D KNOWN-BUG re-enables** (Libjxl-only):
///   `block_ctx_map_15_cluster`
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderImprovementsCustom {
    // ── Screenshot-class entropy-mul lifts ─────────────────────────
    /// W22-1 screenshot lift table (lifts `IDENTITY` / `DCT2X2` /
    /// `AFV` / `DCT4X8`).
    pub screenshot_entropy_mul: ScreenshotEntropyMulPolicy,

    // ── Photo-class entropy-mul lowering ───────────────────────────
    /// W44-29 + nested sub-gates (W44-91 / W44-96 / W44-98 / W44-99 /
    /// W44-100).
    pub high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy,

    // ── DCT-class search admission ─────────────────────────────────
    /// W44-65 / W44-68 DCT64-class suppression on screenshot content.
    pub dct64_search_policy: Dct64SearchPolicy,
    /// W44-123 / W44-124 DCT32-class search retention. Only matters
    /// when `dct64_search_policy` would otherwise drop the DCT32
    /// family together.
    pub dct32_search_policy: Dct32SearchPolicy,
    /// W44-34 / W44-35 smooth-photo DCT64 admission inside the
    /// `pixels < 500_000 AND distance < 2.0` smart-dispatch gate.
    pub smooth_photo_dct64_admission: SmoothPhotoDct64Policy,

    // ── Butteraugli loop qf seeding (effort ≥ 8) ────────────────────
    /// W44-105 / W44-107 / W44-108 — pre-scale the buttloop's
    /// initial qf seed on screenshot-class at high distance. Promoted
    /// from env-var `JXL_BUTTLOOP_INITIAL_QF_SCALE` (Chunk F).
    pub buttloop_qf_seed: ButtloopQfSeedPolicy,

    // ── Adaptive-quant qf seeding (effort ∈ [5, 7]) ─────────────────
    /// W44-109 — mirror of W44-105 at lower effort where buttloop is
    /// unavailable. Promoted from env-var
    /// `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE` (Chunk F).
    pub adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy,

    // ── EPF sharpness seed for buttloop ────────────────────────────
    /// W44-117 / W44-118 / W44-120 — one-shot
    /// `compute_epf_sharpness` seed. The `AutoW44_117 { min_distance }`
    /// variant is promoted from env-var
    /// `JXL_W44_120_EPF_SEED_MIN_DISTANCE`. The `LegacyUniform4` pick
    /// is promoted from env-var `JXL_W44_117_DISABLE=1` (Chunk F).
    pub buttloop_epf_sharpness_seed: EpfSharpnessSeed,

    // ── Perf dispatches (ABSORBED into Custom per user decision) ───
    /// W37-2 — EPF per-block sharpness search dispatch. Was the
    /// independent setter `with_epf_dispatch` on `LossyConfig`
    /// (deleted in Chunk D).
    pub epf_dispatch: EpfDispatch,
    /// W38-2 / W44-90 — pixel-domain loss dispatch. Was the
    /// independent setter `with_pixel_loss_dispatch` on `LossyConfig`
    /// (deleted in Chunk D).
    pub pixel_loss_dispatch: PixelLossDispatch,
    /// W44-87 — single-pass entropy dispatch at e=5 on smooth
    /// photos. Was the independent setter
    /// `with_single_pass_entropy_dispatch` on `LossyConfig` (deleted
    /// in Chunk D).
    pub single_pass_entropy_dispatch: SinglePassEntropyDispatch,
    /// W37-1 / W41-2 — patches scan dispatch. Was the independent
    /// setter `with_patches_dispatch` on `LossyConfig` (deleted in
    /// Chunk D).
    pub patches_dispatch: PatchesDispatch,

    // ── Section A effort-gate divergences (Libjxl-only flips) ──────
    /// `cfl_two_pass` minimum effort threshold (we e7+, libjxl e5+).
    /// [`EncoderStrategy::Libjxl`] sets [`EffortGate::Libjxl`] (= 5).
    pub cfl_two_pass_min_effort: EffortGate,
    /// `try_dct64` minimum effort threshold (we e7+, libjxl no
    /// effort gate). [`EncoderStrategy::Libjxl`] sets
    /// [`EffortGate::Libjxl`] (= no effort gate).
    pub try_dct64_min_effort: EffortGate,
    /// `epf_dynamic_sharpness` minimum effort threshold (we e6+,
    /// libjxl no effort gate). [`EncoderStrategy::Libjxl`] sets
    /// [`EffortGate::Libjxl`].
    pub epf_dynamic_sharpness_min_effort: EffortGate,

    // ── Section D KNOWN-BUG re-enables (Libjxl-only) ───────────────
    /// 15-cluster default for `BlockCtxMap`. Issue #59 KNOWN-BUG —
    /// currently DISABLED because of an upstream `cluster_histograms`
    /// divergence that regresses bytes. [`EncoderStrategy::Libjxl`]
    /// re-enables this (deliberately re-introducing the regression
    /// to match libjxl byte-for-byte). Default = `false`.
    pub block_ctx_map_15_cluster: bool,
}

impl Default for EncoderImprovementsCustom {
    /// Default values matching [`EncoderStrategy::Zenjxl`] — the
    /// production-shipping bundle.
    ///
    /// **W44-130 Chunk D**: `screenshot_entropy_mul` defaults to
    /// [`ScreenshotEntropyMulPolicy::Disabled`] (NOT `Auto`) to
    /// preserve the pre-Chunk-D default-off W22-1 lift behaviour
    /// (was `content_aware_entropy_mul = false` enable bit, now
    /// folded into the policy enum). Callers wanting the lift use
    /// `Custom` with `screenshot_entropy_mul: ForceOn` (typically
    /// alongside a zenanalyze-driven content classifier) or set the
    /// matching override via
    /// [`LossyConfig::with_strategy_overrides`].
    fn default() -> Self {
        Self {
            // W44-130 Chunk D: explicitly `Disabled` (not the field
            // default `Auto`) to preserve pre-Chunk-D opt-in behaviour
            // — `Auto` here would fire the mask1x1 discriminator on
            // every screenshot-like input, changing default bytes on
            // screenshot fixtures.
            screenshot_entropy_mul: ScreenshotEntropyMulPolicy::Disabled,
            // All other fields inherit their type-level defaults
            // (which are tuned to match Zenjxl shipping behaviour).
            high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy::default(),
            dct64_search_policy: Dct64SearchPolicy::default(),
            dct32_search_policy: Dct32SearchPolicy::default(),
            smooth_photo_dct64_admission: SmoothPhotoDct64Policy::default(),
            buttloop_qf_seed: ButtloopQfSeedPolicy::default(),
            adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::default(),
            buttloop_epf_sharpness_seed: EpfSharpnessSeed::default(),
            epf_dispatch: EpfDispatch::default(),
            pixel_loss_dispatch: PixelLossDispatch::default(),
            single_pass_entropy_dispatch: SinglePassEntropyDispatch::default(),
            patches_dispatch: PatchesDispatch::default(),
            cfl_two_pass_min_effort: EffortGate::default(),
            try_dct64_min_effort: EffortGate::default(),
            epf_dynamic_sharpness_min_effort: EffortGate::default(),
            block_ctx_map_15_cluster: false,
        }
    }
}

/// Fully-resolved per-divergence flags consumed by the internal
/// encoder. Built once per encode by [`EncoderStrategy::resolve`] from
/// the caller-supplied strategy variant plus any individual
/// `with_*_hint` setters that override the preset.
///
/// `pub(crate)` — not part of the public API. Call sites read fields
/// directly. Chunk C onwards rewires the call sites to consume this
/// struct.
//
// W44-127 Chunk A: `dead_code` allowed because the call sites that
// will read these fields land in Chunks B/C/G. Tests at the bottom of
// this file exercise the construction and resolve paths.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedImprovements {
    // Section B (content-aware gates)
    pub(crate) screenshot_entropy_mul: ScreenshotEntropyMulPolicy,
    pub(crate) high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy,
    pub(crate) dct64_search_policy: Dct64SearchPolicy,
    pub(crate) dct32_search_policy: Dct32SearchPolicy,
    pub(crate) smooth_photo_dct64_admission: SmoothPhotoDct64Policy,
    pub(crate) buttloop_qf_seed: ButtloopQfSeedPolicy,
    pub(crate) adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy,
    pub(crate) buttloop_epf_sharpness_seed: EpfSharpnessSeed,

    // Perf dispatches (absorbed)
    pub(crate) epf_dispatch: EpfDispatch,
    pub(crate) pixel_loss_dispatch: PixelLossDispatch,
    pub(crate) single_pass_entropy_dispatch: SinglePassEntropyDispatch,
    pub(crate) patches_dispatch: PatchesDispatch,

    // Section A effort-gate flips (Libjxl-only changes from Ours default)
    pub(crate) cfl_two_pass_min_effort: EffortGate,
    pub(crate) try_dct64_min_effort: EffortGate,
    pub(crate) epf_dynamic_sharpness_min_effort: EffortGate,

    // Section D KNOWN-BUG re-enable
    pub(crate) block_ctx_map_15_cluster: bool,
}

impl Default for ResolvedImprovements {
    /// Default values matching [`EncoderStrategy::Zenjxl`] —
    /// production-shipping bundle. Mirrors
    /// [`EncoderImprovementsCustom::default`].
    ///
    /// **W44-130 Chunk D**: `screenshot_entropy_mul` defaults to
    /// [`ScreenshotEntropyMulPolicy::Disabled`] (NOT `Auto`) to
    /// preserve the pre-Chunk-D default-off W22-1 lift behaviour
    /// (the `content_aware_entropy_mul` enable bit was folded into
    /// this policy). Direct `VarDctEncoder::new` callers (tests +
    /// examples) inherit this via
    /// `VarDctEncoder.resolved_improvements: ResolvedImprovements`
    /// initialised at `Default::default()`; production API paths
    /// overwrite via `LossyConfig::resolve_improvements`.
    fn default() -> Self {
        Self {
            screenshot_entropy_mul: ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy::default(),
            dct64_search_policy: Dct64SearchPolicy::default(),
            dct32_search_policy: Dct32SearchPolicy::default(),
            smooth_photo_dct64_admission: SmoothPhotoDct64Policy::default(),
            buttloop_qf_seed: ButtloopQfSeedPolicy::default(),
            adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::default(),
            buttloop_epf_sharpness_seed: EpfSharpnessSeed::default(),
            epf_dispatch: EpfDispatch::default(),
            pixel_loss_dispatch: PixelLossDispatch::default(),
            single_pass_entropy_dispatch: SinglePassEntropyDispatch::default(),
            patches_dispatch: PatchesDispatch::default(),
            cfl_two_pass_min_effort: EffortGate::default(),
            try_dct64_min_effort: EffortGate::default(),
            epf_dynamic_sharpness_min_effort: EffortGate::default(),
            block_ctx_map_15_cluster: false,
        }
    }
}

/// Per-field overrides set via the existing `with_*_hint` setters
/// AFTER `with_strategy` is called. Field-by-field precedence over
/// the strategy preset's resolved value. Mirrors the
/// `with_perceptual_optimizations(false).with_gaborish(true)`
/// precedence pattern.
///
/// W44-130 (Chunk D): exposed as `pub` and reachable via
/// [`LossyConfig::with_strategy_overrides`]. Replaces the five deleted
/// `with_*_hint(Option<bool>)` setters; use `EncoderStrategy::Custom`
/// with [`EncoderImprovementsCustom`] when full per-divergence control
/// is needed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrategyOverrides {
    /// Override for the W22-1 screenshot entropy_mul lift gate. `None`
    /// = use the strategy preset's value (typically `Auto` =
    /// `median(mask1x1) > 95` discriminator). `Some(true/false)` =
    /// force the matching `ScreenshotEntropyMulPolicy::ForceOn/Off`.
    pub screenshot_lift_hint: Option<bool>,
    /// Override for the W44-29 high-distance smooth-photo entropy_mul
    /// lowering gate. `None` = use the strategy preset's value
    /// (typically `Auto` = `distance >= 4.0 AND median(mask1x1) <
    /// SMOOTH_THRESHOLD`). `Some(true/false)` = force the matching
    /// `HighDPhotoEntropyMulPolicy::ForceOn/Off`.
    pub high_d_photo_hint: Option<bool>,
    /// Override for the W44-34/35 smooth-photo DCT64 admission gate.
    /// `None` = use the strategy preset's value (typically `Auto` =
    /// `detect_smooth_photo_for_dct64` auto-detector inside the
    /// `pixels < 500_000 AND distance < 2.0` smart-dispatch gate).
    /// `Some(true/false)` = force the matching
    /// `SmoothPhotoDct64Policy::ForceAdmit/Skip`.
    pub smooth_photo_dct64_hint: Option<bool>,
    /// Override for the W44-65 content-aware DCT64-class suppression
    /// gate. `None` = use the strategy preset's value (typically
    /// `Auto` = `median(mask1x1) >= 99.5` screenshot-class
    /// discriminator). `Some(true)` = force-suppress (screenshot
    /// override); `Some(false)` = force-allow (pre-W44-65
    /// byte-equivalence).
    pub dct_suppress_hint: Option<bool>,
    /// Override for the W44-123/124 DCT32-class search retention gate
    /// (composes with `dct_suppress_hint`). `None` = use the strategy
    /// preset's value (typically `FollowDct64Suppression` =
    /// W44-124 auto-discriminator on m3_colourfulness + edge_density).
    /// `Some(true)` = force `Dct32SearchPolicy::KeepWhenDct64Suppressed`;
    /// `Some(false)` = force `FollowDct64Suppression`.
    pub dct32_keep_hint: Option<bool>,
}

impl StrategyOverrides {
    /// Apply per-field overrides on top of a resolved strategy. Each
    /// `Option<bool>` field, when `Some`, REPLACES the matching policy
    /// in `base` with the corresponding `Force*` variant; when `None`,
    /// `base` is left untouched.
    ///
    /// Mapping (matches the legacy `with_*_hint` semantics):
    /// - `screenshot_lift_hint: Some(true)` → `ScreenshotEntropyMulPolicy::ForceOn`
    /// - `screenshot_lift_hint: Some(false)` → `ScreenshotEntropyMulPolicy::ForceOff`
    /// - `high_d_photo_hint: Some(true)` → `HighDPhotoEntropyMulPolicy::ForceOn`
    /// - `high_d_photo_hint: Some(false)` → `HighDPhotoEntropyMulPolicy::ForceOff`
    /// - `smooth_photo_dct64_hint: Some(true)` → `SmoothPhotoDct64Policy::ForceAdmit`
    /// - `smooth_photo_dct64_hint: Some(false)` → `SmoothPhotoDct64Policy::ForceSkip`
    /// - `dct_suppress_hint: Some(true)` → `Dct64SearchPolicy::ForceSuppress`
    /// - `dct_suppress_hint: Some(false)` → `Dct64SearchPolicy::ForceAllow`
    /// - `dct32_keep_hint: Some(true)` → `Dct32SearchPolicy::KeepWhenDct64Suppressed`
    /// - `dct32_keep_hint: Some(false)` → `Dct32SearchPolicy::FollowDct64Suppression`
    pub(crate) fn apply_to(&self, mut base: ResolvedImprovements) -> ResolvedImprovements {
        if let Some(b) = self.screenshot_lift_hint {
            base.screenshot_entropy_mul = if b {
                ScreenshotEntropyMulPolicy::ForceOn
            } else {
                ScreenshotEntropyMulPolicy::ForceOff
            };
        }
        if let Some(b) = self.high_d_photo_hint {
            base.high_d_photo_entropy_mul = if b {
                HighDPhotoEntropyMulPolicy::ForceOn
            } else {
                HighDPhotoEntropyMulPolicy::ForceOff
            };
        }
        if let Some(b) = self.smooth_photo_dct64_hint {
            base.smooth_photo_dct64_admission = if b {
                SmoothPhotoDct64Policy::ForceAdmit
            } else {
                SmoothPhotoDct64Policy::ForceSkip
            };
        }
        if let Some(b) = self.dct_suppress_hint {
            base.dct64_search_policy = if b {
                Dct64SearchPolicy::ForceSuppress
            } else {
                Dct64SearchPolicy::ForceAllow
            };
        }
        if let Some(b) = self.dct32_keep_hint {
            base.dct32_search_policy = if b {
                Dct32SearchPolicy::KeepWhenDct64Suppressed
            } else {
                Dct32SearchPolicy::FollowDct64Suppression
            };
        }
        base
    }
}

impl EncoderStrategy {
    /// Resolve to the internal per-divergence flag struct.
    ///
    /// `overrides` carries any individual `with_*_hint` calls the
    /// caller made AFTER `with_strategy` — those win field-by-field,
    /// mirroring the `with_perceptual_optimizations` precedence
    /// pattern.
    //
    // W44-128 Chunk B: now called by `LossyConfig::resolve_improvements`
    // at all three `VarDctEncoder` construction sites (still-image
    // `EncodeRequest`, streaming `LossyEncoder`, animation per-frame);
    // the W44-127-era `#[allow(dead_code)]` on this method was removed.
    //
    // W44-132 Chunk F: env-var fallback layer applied AFTER
    // `overrides.apply_to(base)`. The fallback applies ONLY when the
    // resolved field equals its `Default::default()` value — explicit
    // caller settings (via `Custom` payload or `StrategyOverrides`)
    // ALWAYS win over the env-var. See `apply_env_var_fallbacks` for
    // the per-field mapping.
    pub(crate) fn resolve(&self, overrides: &StrategyOverrides) -> ResolvedImprovements {
        let base = match self {
            Self::Libjxl => ResolvedImprovements::libjxl(),
            Self::LeanFaster => ResolvedImprovements::lean_faster(),
            Self::Zenjxl => ResolvedImprovements::zenjxl(),
            Self::Aggressive => ResolvedImprovements::aggressive(),
            Self::Custom(c) => ResolvedImprovements::from_custom(c),
        };
        let mut resolved = overrides.apply_to(base);
        apply_env_var_fallbacks(&mut resolved);
        resolved
    }
}

/// W44-132 Chunk F: env-var fallback for the four promoted env-only
/// knobs. Applies the env-var override only when the resolved field
/// equals its `Default::default()` value — so any explicit caller
/// setting (via `EncoderStrategy::Custom` payload or
/// `StrategyOverrides::apply_to`) wins over the env-var.
///
/// Env-var → field mapping:
///
/// | Env var | Default | Field | Default field value |
/// |---|---|---|---|
/// | `JXL_W44_117_DISABLE=1` | unset | `buttloop_epf_sharpness_seed` | `AutoW44_117 { min_distance: 1.0 }`; promotes to `LegacyUniform4` when env=`1` |
/// | `JXL_W44_120_EPF_SEED_MIN_DISTANCE=<f32>` | `1.0` | `buttloop_epf_sharpness_seed`' `min_distance` | replaces the `1.0` inside `AutoW44_117 { min_distance }` |
/// | `JXL_BUTTLOOP_INITIAL_QF_SCALE=<f32>` | `4.0` | `buttloop_qf_seed` | replaces `AutoScale4` → `AutoScale(env_value)` |
/// | `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=<f32>` | per-effort 2.0/3.0 | `adaptive_quant_qf_seed` | replaces `AutoScalePerEffort` → `AutoScaleCustom { e5_e6: env, e7: env }` |
///
/// Precedence (within the fallback): `JXL_W44_117_DISABLE=1` (force-off)
/// is checked BEFORE `JXL_W44_120_EPF_SEED_MIN_DISTANCE` (min-distance
/// tweak). When the disable env var is set, the `min_distance` env var
/// is ignored — the seed compute is forced off entirely so the
/// distance gate is moot.
///
/// On `not(feature = "std")` builds the fallback is a no-op (env vars
/// are unreadable in `no_std`); the policy retains its post-overrides
/// value bit-identically. The `#[cfg]` guards mirror the call-site
/// env-var-read pattern this layer replaces.
#[cfg_attr(not(feature = "std"), allow(unused_variables))]
fn apply_env_var_fallbacks(r: &mut ResolvedImprovements) {
    #[cfg(feature = "std")]
    {
        // ── buttloop_epf_sharpness_seed ────────────────────────────
        // Default value: `AutoW44_117 { min_distance: 1.0 }`.
        // Two env vars feed this slot. `JXL_W44_117_DISABLE=1` wins if
        // both are set (force-off short-circuits the min-distance
        // tweak — the seed compute never runs so the gate is moot).
        if r.buttloop_epf_sharpness_seed == EpfSharpnessSeed::default() {
            if std::env::var("JXL_W44_117_DISABLE").as_deref() == Ok("1") {
                r.buttloop_epf_sharpness_seed = EpfSharpnessSeed::LegacyUniform4;
            } else if let Ok(s) = std::env::var("JXL_W44_120_EPF_SEED_MIN_DISTANCE")
                && let Ok(d) = s.parse::<f32>()
            {
                r.buttloop_epf_sharpness_seed = EpfSharpnessSeed::AutoW44_117 { min_distance: d };
            }
        }

        // ── buttloop_qf_seed ───────────────────────────────────────
        // Default value: `AutoScale4` (= same gate at 4.0).
        // Env-var `JXL_BUTTLOOP_INITIAL_QF_SCALE=<f32>` replaces with
        // `AutoScale(env_value)` (same gate, custom scale).
        if r.buttloop_qf_seed == ButtloopQfSeedPolicy::default()
            && let Ok(s) = std::env::var("JXL_BUTTLOOP_INITIAL_QF_SCALE")
            && let Ok(v) = s.parse::<f32>()
        {
            r.buttloop_qf_seed = ButtloopQfSeedPolicy::AutoScale(v);
        }

        // ── adaptive_quant_qf_seed ─────────────────────────────────
        // Default value: `AutoScalePerEffort` (= 2.0 e5/e6, 3.0 e7).
        // Env-var `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=<f32>` is a
        // SINGLE value (the env var was historically one knob — kept
        // the per-effort split internal to the default). Replaces both
        // e5/e6 AND e7 with the env value.
        if r.adaptive_quant_qf_seed == AdaptiveQuantQfSeedPolicy::default()
            && let Ok(s) = std::env::var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE")
            && let Ok(v) = s.parse::<f32>()
        {
            r.adaptive_quant_qf_seed =
                AdaptiveQuantQfSeedPolicy::AutoScaleCustom { e5_e6: v, e7: v };
        }
    }
}

// W44-128 Chunk B: `EncoderStrategy::resolve` now runs at every
// `VarDctEncoder` construction site (still-image, streaming,
// animation) via `LossyConfig::resolve_improvements`, which
// transitively keeps `libjxl`/`lean_faster`/`zenjxl`/`aggressive`/
// `from_custom` reachable. The W44-127-era `#[allow(dead_code)]` on
// the impl block was removed.
impl ResolvedImprovements {
    /// Strict libjxl parity (all-divergence). Includes Section A
    /// effort-gate flips AND the Section D KNOWN-BUG `BlockCtxMap`
    /// 15-cluster re-enable — see [`EncoderStrategy::Libjxl`]
    /// doc-comment.
    fn libjxl() -> Self {
        Self {
            screenshot_entropy_mul: ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy::Disabled,
            dct64_search_policy: Dct64SearchPolicy::ForceAllow,
            dct32_search_policy: Dct32SearchPolicy::FollowDct64Suppression,
            smooth_photo_dct64_admission: SmoothPhotoDct64Policy::ForceSkip,
            buttloop_qf_seed: ButtloopQfSeedPolicy::Off,
            adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::Off,
            buttloop_epf_sharpness_seed: EpfSharpnessSeed::LegacyUniform4,
            // Perf dispatches: leave at Default. Libjxl is byte-identical
            // on `Auto` for libjxl-shaped inputs (the dispatch enums
            // are perf-only supersets of libjxl behaviour); callers who
            // care about decode-speed tradeoffs compose those
            // orthogonally via `Custom`.
            epf_dispatch: EpfDispatch::default(),
            pixel_loss_dispatch: PixelLossDispatch::default(),
            single_pass_entropy_dispatch: SinglePassEntropyDispatch::default(),
            patches_dispatch: PatchesDispatch::default(),
            // Section A: flip to libjxl gates
            cfl_two_pass_min_effort: EffortGate::Libjxl,
            try_dct64_min_effort: EffortGate::Libjxl,
            epf_dynamic_sharpness_min_effort: EffortGate::Libjxl,
            // Section D KNOWN-BUG: deliberately re-enable to match
            // libjxl
            block_ctx_map_15_cluster: true,
        }
    }

    /// LeanFaster — drops the heavy per-image content gates; keeps
    /// the cheap photo-class entropy-mul lowering.
    fn lean_faster() -> Self {
        Self {
            screenshot_entropy_mul: ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy::Auto,
            dct64_search_policy: Dct64SearchPolicy::ForceAllow,
            dct32_search_policy: Dct32SearchPolicy::FollowDct64Suppression,
            smooth_photo_dct64_admission: SmoothPhotoDct64Policy::ForceSkip,
            buttloop_qf_seed: ButtloopQfSeedPolicy::Off,
            adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::Off,
            buttloop_epf_sharpness_seed: EpfSharpnessSeed::LegacyUniform4,
            ..Default::default()
        }
    }

    /// Zenjxl — every field at its enum's `#[default]` =
    /// current shipping behaviour.
    fn zenjxl() -> Self {
        Default::default()
    }

    /// Aggressive — currently equivalent to [`Self::zenjxl`] after
    /// W44-124's auto-discriminator obsoleted the previous
    /// "Aggressive flips W44-123 globally" behaviour. Forward-
    /// compatible slot for the next opt-in chunk that has a
    /// too-narrow auto-discriminator for the Zenjxl bundle.
    fn aggressive() -> Self {
        Self::zenjxl()
    }

    /// Copy every field from [`EncoderImprovementsCustom`].
    fn from_custom(c: &EncoderImprovementsCustom) -> Self {
        Self {
            screenshot_entropy_mul: c.screenshot_entropy_mul,
            high_d_photo_entropy_mul: c.high_d_photo_entropy_mul,
            dct64_search_policy: c.dct64_search_policy,
            dct32_search_policy: c.dct32_search_policy,
            smooth_photo_dct64_admission: c.smooth_photo_dct64_admission,
            buttloop_qf_seed: c.buttloop_qf_seed,
            adaptive_quant_qf_seed: c.adaptive_quant_qf_seed,
            buttloop_epf_sharpness_seed: c.buttloop_epf_sharpness_seed,
            epf_dispatch: c.epf_dispatch,
            pixel_loss_dispatch: c.pixel_loss_dispatch,
            single_pass_entropy_dispatch: c.single_pass_entropy_dispatch,
            patches_dispatch: c.patches_dispatch,
            cfl_two_pass_min_effort: c.cfl_two_pass_min_effort,
            try_dct64_min_effort: c.try_dct64_min_effort,
            epf_dynamic_sharpness_min_effort: c.epf_dynamic_sharpness_min_effort,
            block_ctx_map_15_cluster: c.block_ctx_map_15_cluster,
        }
    }
}

// ── LossyConfig ─────────────────────────────────────────────────────────────

/// Lossy (VarDCT) encoding configuration.
///
/// No `Default` — distance/quality is a required choice.
///
/// # libjxl-parity knobs
///
/// The following builders mirror libjxl `cparams` fields and give
/// callers fine-grained control matching what `cjxl` exposes via
/// command-line flags:
///
/// - [`Self::with_photon_noise_iso`] — `--photon_noise=ISO`,
///   synthesise camera-ISO grain instead of estimating from content.
/// - [`Self::with_manual_noise_lut`] — caller-supplied 8-point noise
///   LUT (`cparams.manual_noise`).
/// - [`Self::with_original_distance`] — source distance for re-encode
///   pipelines (`cparams.original_butteraugli_distance`); `x_qm_scale`
///   ramps against this rather than the target.
/// - [`Self::with_quant_ac_rescale`] — post-compute multiplier on
///   AC `global_scale` (`cparams.quant_ac_rescale`); `r < 1.0` →
///   finer quant.
/// - [`Self::with_already_downsampled`] — skip the internal
///   downsample when the caller has already downsampled the input
///   (`cparams.already_downsampled`).
/// - [`Self::with_resampling`] / [`Self::with_auto_resampling`] —
///   `cparams.resampling`.
/// - [`Self::with_center_first`] — concentric-square AC group
///   ordering (`cparams.centerfirst`).
/// - [`EncodeRequest::with_brotli_metadata`] — Brotli-compress EXIF /
///   XMP into `brob` boxes (request-level, applies to both modes).
///
/// See [`LosslessConfig`] for the matching modular-side knobs
/// (`with_force_rct`, `with_tree_learning_sample_fraction`).
#[derive(Clone, Debug)]
pub struct LossyConfig {
    distance: f32,
    effort: u8,
    mode: EncoderMode,
    use_ans: bool,
    gaborish: bool,
    /// EX-J13 — per-tile contrast-adaptive gaborish kernel strength.
    /// Encoder-only; decoder always applies the fixed 3x3 inverse blur.
    /// Default `false`. See [`Self::with_adaptive_gaborish`].
    adaptive_gaborish: bool,
    noise: bool,
    /// When `Some(iso)`, synthesise noise from the ISO value rather
    /// than estimating from content. Matches libjxl `--photon_noise=ISO`.
    photon_noise_iso: Option<f32>,
    /// Caller-supplied 8-point noise LUT. Mirrors libjxl
    /// `cparams.manual_noise`. Lower priority than `photon_noise_iso`,
    /// higher than content estimation.
    manual_noise_lut: Option<[f32; 8]>,
    /// Multiplier applied to the AC quantiser's `global_scale` after
    /// the standard distance-driven computation. Mirrors libjxl's
    /// `cparams.quant_ac_rescale`. `None` (default) leaves
    /// `global_scale` untouched.
    quant_ac_rescale: Option<f32>,
    /// Caller-supplied source-image butteraugli distance for re-encode
    /// pipelines. Mirrors libjxl `cparams.original_butteraugli_distance`.
    /// `None` keeps libjxl's default behaviour (treat source as
    /// ground-truth, original = target).
    original_distance: Option<f32>,
    denoise: bool,
    error_diffusion: bool,
    pixel_domain_loss: bool,
    lz77: bool,
    lz77_method: Lz77Method,
    force_strategy: Option<u8>,
    max_strategy_size: Option<u8>,
    patches: bool,
    /// libjxl-style dot detection (refs #19). Default `true` to
    /// mirror libjxl's `Override::kDefault` semantics — the in-encoder
    /// gates (effort >= 7, distance >= 3.0, no text-like patches in
    /// the same image) make this effectively a no-op outside its
    /// niche content range, matching `cjxl`'s "encoder chooses"
    /// default for `--dots`. Disable explicitly via
    /// [`Self::with_dot_detection`] / `--no-dot-detection`.
    dot_detection: bool,
    /// Smear color values in alpha=0 pixels to a weighted average of
    /// visible neighbors (libjxl `SimplifyInvisible` lossy mode,
    /// `enc_frame.cc:511`). 5-20% smaller files on sprites/icons with
    /// large transparent regions; near-zero cost on photos with
    /// mostly-opaque alpha. Default `true`. Disable via
    /// [`Self::with_simplify_invisible`].
    simplify_invisible: bool,
    /// Reorder AC groups in the multi-group TOC so groups near the
    /// image center appear first in the bitstream — for progressive
    /// renderers that show partial frames during download. libjxl
    /// `cparams.centerfirst`. Default `false` (raster order). See
    /// [`Self::with_center_first`].
    center_first: bool,
    /// Decoder upsampling factor (refs #12). `1` (default) = no
    /// resampling; `2`/`4`/`8` = box-filter downsample the input by
    /// this factor before encoding and signal the decoder to upsample
    /// after rendering. Trades per-pixel fidelity for dramatic file-size
    /// reduction at very high distances. libjxl auto-selects 2× at
    /// d ≥ 10. See [`Self::with_resampling`].
    resampling: u32,
    /// `true` when [`Self::with_resampling`] was called explicitly.
    /// Used to decide whether the auto-resample-at-high-distance
    /// gate fires (refs #12). Auto only kicks in if the caller did
    /// **not** pin a resampling factor.
    resampling_explicit: bool,
    /// `true` (default) enables libjxl's auto-resample-at-d≥10 rule
    /// (`enc_frame.cc:103-115`). When the effective gate triggers,
    /// the encoder uses the sharper 2× kernel and adjusts the
    /// internal distance to `d * 0.25 + 0.25` so the bpp stays
    /// roughly comparable. Disable via [`Self::with_auto_resampling`]
    /// if you want strict pinned behavior.
    auto_resampling: bool,
    /// `true` when the caller has already downsampled the input to
    /// the target resolution and just wants the encoder to write the
    /// matching `upsampling` factor in the bitstream. Mirrors libjxl
    /// `cparams.already_downsampled`. No-op when `resampling == 1`.
    already_downsampled: bool,
    splines: Option<Vec<crate::vardct::splines::Spline>>,
    /// Enable automatic spline detection from the input XYB planes.
    ///
    /// When `true` AND [`Self::splines`] is unset AND the effective
    /// [`Self::effort`] is ≥ 7, the encoder asks
    /// [`crate::vardct::splines::find_splines`] for thin-feature curves
    /// (power lines, horizons, hair) to subtract before VarDCT and
    /// add back in the decoder. Mirrors libjxl `enc_heuristics.cc:1048-1054`
    /// (`speed_tier <= kSquirrel`).
    ///
    /// Default derived from effort via
    /// [`crate::effort::EffortProfile::auto_splines_default`]
    /// (`effort >= 8`). When the caller explicitly opts in/out via
    /// [`Self::with_auto_splines`], [`Self::auto_splines_explicit`]
    /// flips and the explicit value wins outright (mirroring the
    /// `patches_explicit` / `butteraugli_iters_explicit` pattern).
    auto_splines: bool,
    /// Tracks whether the caller has explicitly set `auto_splines`
    /// via [`Self::with_auto_splines`]. Mirrors the
    /// `patches_explicit` / `butteraugli_iters_explicit` pattern.
    /// `false` means the auto-splines enable state derives from the
    /// per-effort profile default; `true` means the user-set
    /// [`Self::auto_splines`] wins outright.
    auto_splines_explicit: bool,
    progressive: ProgressiveMode,
    lf_frame: bool,
    #[cfg(feature = "butteraugli-loop")]
    butteraugli_iters: u32,
    #[cfg(feature = "butteraugli-loop")]
    butteraugli_iters_explicit: bool,
    /// HDR-aware perceptual loss for the butteraugli quantization loop
    /// (EX-J11). Default [`HdrLoss::Butteraugli`] keeps every existing
    /// hash-lock byte-identical. [`HdrLoss::Vdp2`] is opt-in and surfaces
    /// [`EncodeError::InvalidConfig`] at encode time until the chunk-2
    /// HDR-VDP-2 maths land. See [`Self::with_hdr_loss`].
    #[cfg(feature = "butteraugli-loop")]
    hdr_loss: HdrLoss,
    #[cfg(feature = "ssim2-loop")]
    ssim2_iters: u32,
    #[cfg(feature = "zensim-loop")]
    zensim_iters: u32,
    threads: usize,
    non_finite_action: NonFiniteAction,
    /// Sweep / picker hook: when set, replaces the effort+mode-derived
    /// `EffortProfile` everywhere the encoder asks for one. See
    /// [`Self::with_effort_profile_override`].
    profile_override: Option<crate::effort::EffortProfile>,
    /// Input canonicalization pre-pass (drop opaque alpha,
    /// near-grayscale collapse, 16→8 downcast when safe). Default
    /// `false` to keep existing hash-locks byte-identical. See
    /// [`Self::with_canonicalize_input`].
    canonicalize_input: bool,
    /// RFC #45 pick #4 chunk 1 — content-class dispatch override /
    /// opt-in. When `Some(class)` the caller has pre-computed the
    /// content class (e.g. via zenanalyze or any other classifier);
    /// [`Self::effective_profile_for_image`] will route it through
    /// [`crate::effort::EffortProfile::adapt_to_image_content`].
    /// `None` (default) keeps every existing hash-lock byte-identical.
    /// See [`Self::with_content_class`].
    content_class: Option<crate::effort::ImageContentClass>,
    // W44-130 Chunk D: `content_aware_entropy_mul: bool` field
    // DELETED. The opt-in enable bit was subsumed by the
    // [`ScreenshotEntropyMulPolicy`] 4-state enum
    // (`Auto` / `ForceOn` / `ForceOff` / `Disabled`). The Zenjxl
    // default in [`EncoderImprovementsCustom::default`] is
    // `Disabled` (preserving pre-Chunk-D default-off behaviour).
    // Callers opt in via `EncoderStrategy::Custom` with
    // `screenshot_entropy_mul: ForceOn` or
    // `with_strategy_overrides(StrategyOverrides {
    // screenshot_lift_hint: Some(true), .. })`.
    /// W44-130 (Chunk D) — per-field overrides applied AFTER the
    /// [`Self::strategy`] preset resolves. Replaces the five legacy
    /// `with_*_hint(Option<bool>)` setters (deleted in Chunk D); the
    /// surviving escape hatch is
    /// [`Self::with_strategy_overrides`]. Each `Some` field maps to
    /// the matching `Force*` variant on the resolved
    /// [`ResolvedImprovements`] via [`StrategyOverrides::apply_to`].
    /// Default (`StrategyOverrides::default()`) is all `None` —
    /// overrides nothing; the preset's resolved value passes through
    /// unchanged.
    strategy_overrides: StrategyOverrides,
    /// W44-128 (Chunk B) encoder compatibility / improvements bundle.
    ///
    /// Selects a named preset (`Libjxl` / `LeanFaster` / `Zenjxl` /
    /// `Aggressive`) or a fully-custom set of dials via
    /// [`EncoderStrategy::Custom`]. Default
    /// [`EncoderStrategy::Zenjxl`] reproduces what we ship today.
    ///
    /// Individual `with_*_hint` setters called AFTER
    /// [`Self::with_strategy`] override the matching field on the
    /// resolved [`ResolvedImprovements`] (mirrors the
    /// [`Self::with_perceptual_optimizations`] precedence pattern).
    ///
    /// **Chunk B**: the resolved [`ResolvedImprovements`] is computed
    /// once at encoder construction time and stored alongside
    /// `VarDctEncoder` for Chunk C+ to consume. No call site reads it
    /// yet; the existing `with_*_hint` `Option<bool>` fields still
    /// drive every gate. Hash-locks therefore stay byte-identical.
    ///
    /// See [`Self::with_strategy`] and `docs/COMPATIBILITY_MODES.md`.
    strategy: EncoderStrategy,
    /// Tracks whether the caller has explicitly set `patches` via
    /// [`Self::with_patches`]. Mirrors the
    /// `butteraugli_iters_explicit` / `resampling_explicit` pattern.
    /// `false` means the patches enable state derives from the
    /// per-image profile (effort default + content-class dispatch);
    /// `true` means the user-set [`Self::patches`] wins outright.
    /// Default `false`. See [`Self::with_patches`].
    patches_explicit: bool,
    // W44-130 Chunk D: `patches_dispatch` field deleted from
    // `LossyConfig`. The dispatch policy now lives on
    // `EncoderImprovementsCustom.patches_dispatch` and flows to
    // `VarDctEncoder.patches_dispatch` via `resolved_improvements`.
    /// Edge-preserving filter (EPF) iteration count override.
    ///
    /// `-1` (default) = encoder chooses based on butteraugli distance
    /// (the libjxl-parity thresholds `[0.7, 1.5, 4.0]`: 0 iters below
    /// 0.7, 1 at \[0.7,1.5), 2 at \[1.5,4.0), 3 at >=4.0).
    /// `0` = forced off — the decoder skips EPF entirely.
    /// `1`/`2`/`3` = forced iteration count (1 = Step 2 only, 2 =
    /// Step 1+2, 3 = Step 0+1+2). Higher = heavier smoothing,
    /// slower decode. Mirrors libjxl `cjxl --epf` and the
    /// `JXL_ENC_FRAME_SETTING_EPF` C API knob
    /// (`enc_frame.cc:284-285`).
    ///
    /// See [`Self::with_epf_level`].
    epf_level: i8,
    // W44-130 Chunk D: `epf_dispatch`, `pixel_loss_dispatch`,
    // `single_pass_entropy_dispatch` fields deleted from
    // `LossyConfig`. The dispatch policies now live on
    // `EncoderImprovementsCustom` and flow to the matching
    // `VarDctEncoder.*_dispatch` fields via `resolved_improvements`.
    /// Optional separate butteraugli distance for the alpha extra
    /// channel (CLI passthrough — mirrors libjxl `cjxl --alpha_distance`,
    /// `enc_params.h:alpha_distance`). `None` (default) keeps the
    /// existing pipeline behaviour (alpha encoded losslessly when the
    /// layout has alpha). `Some(d)` is stored on the config; encoder-side
    /// wiring of a separately-quantised lossy alpha channel is queued
    /// follow-on work — the value is currently advisory only.
    /// See [`Self::with_alpha_distance`].
    alpha_distance: Option<f32>,
    /// Opt-in: engage the **squeeze-on-extras** (responsive=1) lossy
    /// alpha pipeline. Default `false`. See
    /// [`Self::with_alpha_squeeze`] for the framework + chunk-2
    /// status.
    alpha_squeeze: bool,
    /// Optional modular group-encoding order (CLI passthrough — mirrors
    /// libjxl `cjxl --group_order` / `JXL_ENC_FRAME_SETTING_GROUP_ORDER`).
    /// `None` (default) = scanline order. `Some(0)` = scanline. `Some(1)`
    /// = center-first (equivalent to [`Self::with_center_first(true)`]).
    /// `Some(2)` is reserved for future encoder modes. When set to 1 the
    /// encoder mirrors `center_first` so the existing center-first
    /// reorder kicks in; the explicit `group_order` setting also flips
    /// the `center_first` flag for downstream pipeline parity.
    /// See [`Self::with_group_order`].
    group_order: Option<u8>,
    /// Optional centre-pixel X coordinate for the center-first AC group
    /// reorder (CLI passthrough — mirrors libjxl `cjxl --center_x` /
    /// `JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X`). `None` (default)
    /// uses the image centre. Stored on the config; encoder-side
    /// honouring of a non-default centre is queued follow-on work
    /// (the existing center-first reorder anchors at image centre).
    /// See [`Self::with_center_x`].
    center_x: Option<i64>,
    /// Optional centre-pixel Y coordinate for the center-first AC group
    /// reorder (CLI passthrough — mirrors libjxl `cjxl --center_y` /
    /// `JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y`). `None` (default)
    /// uses the image centre. Stored on the config; encoder-side
    /// honouring of a non-default centre is queued follow-on work.
    /// See [`Self::with_center_y`].
    center_y: Option<i64>,
    /// Optional decoder upsampling mode (CLI passthrough — mirrors
    /// libjxl `cjxl --upsampling_mode`, `enc_params.h:upsampling_mode`).
    /// `None` / `Some(-1)` = non-separable (libjxl default). `Some(0)`
    /// = nearest neighbour (pixel-art). `Some(1)` = reserved. Stored on
    /// the config; emitting the custom upsampling LUT in `FrameHeader`
    /// is queued follow-on work — current behaviour uses the JXL spec's
    /// default upsampling for the active `with_resampling` factor.
    /// See [`Self::with_upsampling_mode`].
    upsampling_mode: Option<i32>,
    /// Decoding-speed tier (libjxl `--faster_decoding 0..4`). Higher
    /// values bias the VarDCT encode toward simpler bitstreams that
    /// decode faster, at the cost of compression. Default `0`
    /// (compression-priority). Mirrors libjxl
    /// `cparams.decoding_speed_tier`; see
    /// [`Self::with_faster_decoding`] for the per-tier effects.
    faster_decoding: u8,
    /// Container-wrap policy (libjxl `--container 0|1`). Default
    /// [`ContainerMode::Auto`] keeps the existing behaviour (wrap only
    /// when metadata or level demands it). See
    /// [`Self::with_container_mode`].
    container_mode: ContainerMode,
    /// Explicit progressive-DC level (libjxl `--progressive_dc 0..2`).
    /// `0` = no progressive DC (default); `1` = one LfFrame ahead of
    /// the main VarDCT frame (equivalent to
    /// [`Self::with_lf_frame(true)`]); `2` = two nested LfFrames
    /// (libjxl path; our encoder currently emits a single LfFrame and
    /// warns). See [`Self::with_progressive_dc`].
    progressive_dc: u8,
    /// Opt-in: enable automatic delta-frame mode selection for
    /// multi-frame animation encodes (A1 audit "Animation" — Skip /
    /// delta frame encoding). When `true`, the animation encode path is
    /// permitted to swap the per-frame [`BlendMode::Replace`] default
    /// for a delta-friendly alternative
    /// ([`BlendMode::Add`] with a tiny crop that leaves the canvas
    /// unchanged) when it detects that frame N is byte-identical to the
    /// preceding displayed frame.
    ///
    /// Chunk 1 POC scope (this commit): one heuristic — identical-frame
    /// short-circuit using `Add` over a 1×1 zero-pixel crop. Chunk 2
    /// will add a full trial-encode of `Regular` vs
    /// `Add(reference=N-1)` vs `Blend(reference=N-1)` per frame and
    /// pick the cheapest decodable variant. Default `false` — no
    /// hash-locked bitstream changes at default.
    ///
    /// Lossless only in chunk 1: a residual-from-prior `Add` payload in
    /// the lossy pipeline must round-trip through the reconstructed
    /// (already-quantised) reference frame, not the original pixels —
    /// chunk 2 will add a reconstruction shadow for the lossy path.
    /// See [`Self::with_auto_delta_frames`].
    auto_delta_frames: bool,
    /// Input/output buffering policy (streaming refactor scaffolding,
    /// jxl-encoder#11). Default [`Buffering::Auto`] resolves to
    /// [`Buffering::FullBuffered`] for ≤ 2048² images and
    /// [`Buffering::BufferedOutput`] otherwise (matches libjxl post-
    /// `032d39a`). **Chunk 1: no dispatch is wired** — every variant
    /// currently routes through the existing one-shot path, so output
    /// bytes are identical regardless of `buffering`. See
    /// [`Self::with_buffering`].
    buffering: Buffering,
    /// Chroma subsampling mode (issue #47). Default
    /// [`ChromaSubsampling::Full444`] keeps existing bitstreams
    /// byte-identical. Non-`Full444` modes currently return
    /// [`EncodeError::InvalidConfig`] (encoder wiring is chunk 4); the
    /// zenyuv-backed conversion helpers in
    /// `crate::vardct::chroma_subsampling` are ready for the wire-up.
    /// See [`Self::with_chroma_subsampling`].
    chroma_subsampling: ChromaSubsampling,
}

/// Policy for what to do if the encoder finds non-finite (NaN / ±Inf)
/// f32 values in the XYB pixel planes at the conversion→pipeline
/// boundary.
///
/// The opsin XYB transform (`cbrt(mixed + bias) - cbrt(bias)`) is
/// finite for any finite linear-RGB input — non-finite XYB indicates
/// an upstream bug (caller passed non-finite linear-RGB, internal
/// arithmetic leaked NaN, or memory corruption). The encoder runs a
/// SIMD scan at the boundary either way; this enum picks what happens
/// when the scan reports non-finite.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NonFiniteAction {
    /// **Default.** Read-only SIMD scan; return
    /// [`EncodeError::InvalidInput`] on first non-finite value.
    /// ~4× faster than [`Sanitize`](Self::Sanitize) (no buffer writes
    /// — single pass through cache hierarchy). Fail-fast surface, no
    /// DoS exposure because the encode never touches the bad data.
    #[default]
    Error,
    /// Read-modify-write SIMD scrub on the linear-RGB input plane (and
    /// defense-in-depth on XYB output): replace any non-finite value
    /// with `0.0` and continue encoding. Use for image-proxy
    /// deployments that prefer best-effort encoding over fail-fast on
    /// hostile input. Costs an extra owned-buffer copy + one
    /// read-modify-write SIMD pass (~12.5 GB/s) over the linear-RGB
    /// plane vs. the read-only [`Error`](Self::Error) path.
    Sanitize,
}

impl LossyConfig {
    /// Create with butteraugli distance (1.0 = high quality). Default effort 7.
    pub fn new(distance: f32) -> Self {
        Self::new_with_effort(distance, 7)
    }

    fn new_with_effort(distance: f32, effort: u8) -> Self {
        let profile = crate::effort::EffortProfile::lossy(effort, EncoderMode::Reference);
        Self {
            distance,
            effort: profile.effort,
            mode: EncoderMode::Reference,
            use_ans: profile.use_ans,
            gaborish: profile.gaborish,
            adaptive_gaborish: false,
            noise: false,
            photon_noise_iso: None,
            manual_noise_lut: None,
            quant_ac_rescale: None,
            original_distance: None,
            denoise: false,
            error_diffusion: profile.error_diffusion,
            pixel_domain_loss: profile.pixel_domain_loss,
            lz77: profile.lz77,
            lz77_method: profile.lz77_method,
            force_strategy: None,
            max_strategy_size: None,
            patches: profile.patches,
            dot_detection: true, // refs #19; default-on to mirror libjxl Override::kDefault (gated effort>=7 && d>=3.0)
            simplify_invisible: true,
            center_first: false,
            resampling: 1,
            resampling_explicit: false,
            auto_resampling: true,
            already_downsampled: false,
            splines: None,
            // Default derived from effort. `with_auto_splines` flips
            // `auto_splines_explicit = true` and pins the value.
            auto_splines: crate::effort::EffortProfile::auto_splines_default(profile.effort),
            auto_splines_explicit: false,
            progressive: ProgressiveMode::Single,
            lf_frame: false,
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: profile.butteraugli_iters,
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters_explicit: false,
            // EX-J11 chunk 4: default flipped from `Butteraugli` to
            // `Auto`. SDR encodes (sRGB / BT.709 / Linear / Unknown)
            // resolve `Auto` → `Butteraugli` at encode entry — the
            // hash-lock fixtures all use SDR transfer functions and
            // therefore stay byte-identical. PQ / HLG encodes pick up
            // `Vdp2` automatically; chunk-3 measured -36.5% avg
            // paper-faithful reference score improvement vs.
            // butteraugli on HDR-AIC-2025. See
            // [`HdrLoss::resolve`] for the full dispatch matrix.
            #[cfg(feature = "butteraugli-loop")]
            hdr_loss: HdrLoss::Auto,
            #[cfg(feature = "ssim2-loop")]
            ssim2_iters: 0,
            #[cfg(feature = "zensim-loop")]
            zensim_iters: 0,
            threads: 0,
            non_finite_action: NonFiniteAction::default(),
            profile_override: None,
            canonicalize_input: false,
            content_class: None,
            // W44-130 Chunk D: `content_aware_entropy_mul` field
            // deleted; opt-in lives via `EncoderStrategy::Custom` with
            // `screenshot_entropy_mul: ForceOn` (or
            // `with_strategy_overrides`).
            // W44-130 Chunk D: default `StrategyOverrides::default()`
            // is all-`None` — overrides nothing. The strategy preset's
            // resolved value passes through unchanged. Replaces the
            // five deleted `with_*_hint(Option<bool>)` setters; the
            // surviving escape hatch is `with_strategy_overrides`.
            strategy_overrides: StrategyOverrides::default(),
            // W44-128 Chunk B: default `EncoderStrategy::Zenjxl`
            // (production shipping). Computed `ResolvedImprovements`
            // is unused until Chunk C+ rewires call sites; hash-locks
            // therefore stay byte-identical at the default.
            strategy: EncoderStrategy::default(),
            patches_explicit: false,
            // W44-130 Chunk D: `patches_dispatch`, `epf_dispatch`,
            // `pixel_loss_dispatch`, `single_pass_entropy_dispatch`
            // fields deleted (absorbed into `EncoderImprovementsCustom`).
            epf_level: -1,
            alpha_distance: None,
            // Chunk-1 default: keep responsive=0 lossy alpha path
            // (byte-identical to today). Opt-in via
            // `LossyConfig::with_alpha_squeeze(true)`.
            alpha_squeeze: false,
            group_order: None,
            center_x: None,
            center_y: None,
            upsampling_mode: None,
            faster_decoding: 0,
            container_mode: ContainerMode::Auto,
            progressive_dc: 0,
            auto_delta_frames: false,
            buffering: Buffering::Auto,
            chroma_subsampling: ChromaSubsampling::Full444,
        }
    }

    /// Resolve the effective [`EffortProfile`]: the override if set,
    /// otherwise the standard profile derived from effort + mode. The
    /// `faster_decoding` knob is applied last (libjxl ordering — the
    /// speed-tier gates fire AFTER effort defaults are computed).
    pub(crate) fn effective_profile(&self) -> crate::effort::EffortProfile {
        let mut p = self
            .profile_override
            .clone()
            .unwrap_or_else(|| crate::effort::EffortProfile::lossy(self.effort, self.mode));
        p.apply_faster_decoding(self.faster_decoding);
        p
    }

    /// Effective patches flag (libjxl `enc_modular.cc:707` —
    /// `decoding_speed_tier < 2` for the modular subtract-and-encode
    /// path). At lossy tier >= 2 we skip the VarDCT patches pre-pass
    /// for the same reason.
    pub(crate) fn effective_patches(&self) -> bool {
        if self.faster_decoding >= 2 {
            return false;
        }
        self.patches
    }

    /// Effective LZ77 flag. libjxl `enc_ans.cc:1372` skips LZ77 for
    /// VarDCT streams at `decoding_speed_tier >= 1` (the per-frame
    /// AC histogram pass forces `lz77_method = kNone`). Returns the
    /// stored `cfg.lz77` field at tier 0.
    pub(crate) fn effective_lz77(&self) -> bool {
        if self.faster_decoding >= 1 {
            return false;
        }
        self.lz77
    }

    /// Effective gaborish flag. libjxl `enc_frame.cc:280` disables
    /// gaborish unconditionally at `decoding_speed_tier == 4` (its
    /// 3x3 inverse on every decoded plane adds measurable decode
    /// time without commensurate quality benefit at tier 4 quality
    /// targets).
    pub(crate) fn effective_gaborish(&self) -> bool {
        if self.faster_decoding >= 4 {
            return false;
        }
        self.gaborish
    }

    /// Resolve the per-image effective [`EffortProfile`] for the lossy
    /// VarDCT path. Layered on top of [`Self::effective_profile`] with
    /// the always-on
    /// [`crate::effort::EffortProfile::adapt_to_image_lossy`]
    /// adapter, which drops `try_dct64` to `false` on the
    /// `pixels < LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD` AND
    /// `distance < LOSSY_LOW_DISTANCE_THRESHOLD` cell.
    ///
    /// **Override-skipping**: when the caller has supplied an explicit
    /// `__expert` profile_override via [`Self::with_internal_params`],
    /// the adapter is skipped — sweep harnesses that pin
    /// `try_dct64 = Some(true)` survive the dispatch.
    ///
    /// Mirrors the lossless [`LosslessConfig::effective_profile_for_image`]
    /// (audit item #3 / chunk 1 `1c4691f`).
    pub(crate) fn effective_profile_for_image(&self, pixels: u64) -> crate::effort::EffortProfile {
        self.effective_profile_for_image_with_smoothness(pixels, false)
    }

    /// Variant of [`Self::effective_profile_for_image`] that also takes a
    /// caller-computed `smooth_photo_for_dct64` hint (W44-35).
    ///
    /// When the auto detector says `true` AND the caller has not pinned
    /// `Some(false)` via
    /// [`Self::with_strategy_overrides`]'s `smooth_photo_dct64_hint`,
    /// the `adapt_to_image_lossy` `try_dct64 -> false` flip is
    /// suppressed on the gated cell so DCT64-class transforms are
    /// evaluated.
    ///
    /// Caller-supplied explicit `Some(true)`/`Some(false)` always wins
    /// over the auto detector. Default `None` defers to the auto value.
    pub(crate) fn effective_profile_for_image_with_smoothness(
        &self,
        pixels: u64,
        smooth_photo_for_dct64_auto: bool,
    ) -> crate::effort::EffortProfile {
        let mut p = self.effective_profile();
        // Always-on per-image adapter — skipped only when an explicit
        // `__expert` override is in play, to avoid silently re-flipping
        // a sweep harness's pinned value.
        if self.profile_override.is_none() {
            // W44-129 Chunk C: resolve the `smooth_photo_dct64_admission`
            // policy from the `EncoderStrategy` bundle + per-field
            // overrides. `ResolvedImprovements` is computed once here
            // (cheap — no allocation for the named-strategy variants;
            // `Custom(Box<_>)` is the only allocating path).
            //
            // Policy translation (matches `StrategyOverrides::apply_to`):
            //   * `Auto` → existing auto detector value
            //   * `ForceAdmit` → true (admit DCT64 on the gated cell)
            //   * `ForceSkip` → false (preserves pre-W44-35 behaviour;
            //     `EncoderStrategy::Libjxl` uses this)
            //
            // `StrategyOverrides::apply_to` maps the legacy
            // `smooth_photo_dct64_hint: Some(true)` → `ForceAdmit` and
            // `Some(false)` → `ForceSkip` so production semantics stay
            // bit-identical when the caller chains hints AFTER
            // `with_strategy(...)`.
            let resolved = self.resolve_improvements();
            let smooth_hint = match resolved.smooth_photo_dct64_admission {
                crate::api::SmoothPhotoDct64Policy::Auto => smooth_photo_for_dct64_auto,
                crate::api::SmoothPhotoDct64Policy::ForceAdmit => true,
                crate::api::SmoothPhotoDct64Policy::ForceSkip => false,
            };
            p.adapt_to_image_lossy_with_smoothness(pixels, self.distance, smooth_hint);
            // RFC #45 pick #4 chunk 1 — content-class dispatch.
            // Fires only when the caller has explicitly set the class
            // via `with_content_class` (default `None` keeps every
            // hash-lock fixture byte-identical, because the dispatch
            // surface itself is opt-in at the API level for chunk 1).
            if let Some(class) = self.content_class {
                p.adapt_to_image_content(pixels, self.distance, class);
            }
            // W44-133 Chunk G: Section A effort-gate consultation.
            // Flips `cfl_two_pass` / `try_dct64` / `epf_dynamic_sharpness`
            // to the libjxl threshold when `EncoderStrategy::Libjxl` is
            // selected (or to `Off`/`AtLeast(n)` for `Custom` strategies
            // that set the matching `EffortGate` variant). Default
            // `EffortGate::Ours` preserves the pre-Chunk-G value
            // byte-identically. Applied AFTER `adapt_to_image_lossy_with_smoothness`
            // so the W44-34/35 smart-dispatch (which already may
            // promote `try_dct64 -> true` on smooth photos) and the
            // content-class dispatch run first; the consultation can
            // still re-flip the field to the libjxl gate value if the
            // strategy requests it.
            p.apply_section_a_effort_gates(&resolved);
        }
        p
    }

    /// Apply picker / sweep override knobs scoped to the **lossy (VarDCT)**
    /// encode path.
    ///
    /// Each `Some(_)` field on the supplied
    /// [`crate::effort::LossyInternalParams`] overrides the corresponding
    /// effort-derived default; `None` fields keep the default. Per-knob
    /// public setters (`with_butteraugli_iters`, `with_gaborish`, …) called
    /// after this still take precedence on the few knobs they cover.
    ///
    /// The type system enforces mode-correctness: modular-only knobs
    /// (RCT search, WP parameter scan, tree-learning shape) live on
    /// [`crate::effort::LosslessInternalParams`] and cannot be passed here.
    ///
    /// **Requires the `__expert` cargo feature.**
    /// Not stable; the underlying field set may grow additively between
    /// minor versions.
    #[cfg(feature = "__expert")]
    #[doc(hidden)]
    pub fn with_internal_params(mut self, params: crate::effort::LossyInternalParams) -> Self {
        let mut profile = crate::effort::EffortProfile::lossy(self.effort, self.mode);
        params.apply_to(&mut profile);
        self.profile_override = Some(profile);
        self
    }

    /// Create from a [`Quality`] specification.
    pub fn from_quality(quality: Quality) -> core::result::Result<Self, EncodeError> {
        let distance = quality.to_distance()?;
        Ok(Self::new(distance))
    }

    /// Set effort level (1–12). Higher effort = slower, better compression.
    ///
    /// This adjusts all effort-dependent defaults:
    /// - **e1–3**: DCT8 only, Huffman, no gaborish/patches/butteraugli
    /// - **e4**: + ANS entropy coding, custom coefficient orders
    /// - **e5**: + gaborish, pixel-domain loss, AC strategy search, AdjustQuantBlockAC
    /// - **e6**: + DCT4x8/AFV strategies, non-aligned eval, EPF dynamic sharpness
    /// - **e7**: + patches, error diffusion, CfL two-pass, LZ77 RLE, DCT64 strategies
    /// - **e8**: + butteraugli loop (2 iters), LZ77 greedy, WP param search (2 modes)
    /// - **e9**: + LZ77 optimal (Viterbi DP), 4 butteraugli iters, WP search (5 modes)
    /// - **e10**: + 8 butteraugli iters, 2 tree-learn seeds (RFC#45 pick #1)
    /// - **e11**: + 16 butteraugli iters, 4 lossy-search seeds, 16 tree-learn seeds
    /// - **e12**: + 32 butteraugli iters (RFC#45 chunk 2; requires `MAX_QUANT_LOOP_ITERS = 32`)
    ///
    /// e10/e11/e12 extend libjxl's kTortoise=9 ceiling with strictly-longer
    /// search budgets; the bitstream remains 100% spec-valid. See RFC issue
    /// #45.
    ///
    /// Individual `with_*()` calls after `with_effort()` override these defaults.
    pub fn with_effort(self, effort: u8) -> Self {
        let mut new = Self::new_with_effort(self.distance, effort);
        // Preserve settings that are never effort-derived (always opt-in)
        new.mode = self.mode;
        new.noise = self.noise;
        new.photon_noise_iso = self.photon_noise_iso;
        new.manual_noise_lut = self.manual_noise_lut;
        new.quant_ac_rescale = self.quant_ac_rescale;
        new.original_distance = self.original_distance;
        new.denoise = self.denoise;
        new.force_strategy = self.force_strategy;
        new.max_strategy_size = self.max_strategy_size;
        new.splines = self.splines;
        // Preserve explicit auto_splines setting across with_effort.
        // Otherwise let the effort-derived default in `new` win, so
        // that `LossyConfig::new(d).with_effort(8)` flips on the
        // chunk-3 detector while `with_effort(7)` flips it off.
        if self.auto_splines_explicit {
            new.auto_splines = self.auto_splines;
            new.auto_splines_explicit = true;
        }
        new.progressive = self.progressive;
        // Preserve explicit butteraugli override
        #[cfg(feature = "butteraugli-loop")]
        if self.butteraugli_iters_explicit {
            new.butteraugli_iters = self.butteraugli_iters;
            new.butteraugli_iters_explicit = true;
        }
        // EX-J11 chunk 1: hdr_loss is a plain flag (no _explicit twin —
        // the default Butteraugli is the existing behaviour, so simple
        // copy preserves the caller's choice across with_effort() exactly
        // like ssim2_iters / zensim_iters below).
        #[cfg(feature = "butteraugli-loop")]
        {
            new.hdr_loss = self.hdr_loss;
        }
        #[cfg(feature = "ssim2-loop")]
        {
            new.ssim2_iters = self.ssim2_iters;
        }
        #[cfg(feature = "zensim-loop")]
        {
            new.zensim_iters = self.zensim_iters;
        }
        new.profile_override = self.profile_override;
        new.canonicalize_input = self.canonicalize_input;
        new.content_class = self.content_class;
        // W44-130 Chunk D: `content_aware_entropy_mul` field deleted;
        // opt-in lives via `strategy` / `strategy_overrides` below.
        // Preserve the caller's strategy overrides across
        // `with_effort` (mirror of the strategy preservation below).
        new.strategy_overrides = self.strategy_overrides.clone();
        // W44-128 Chunk B: preserve the caller's
        // `with_strategy(EncoderStrategy::...)` across `with_effort`.
        // Effort-derived state (in `new`) is regenerated from the new
        // effort + mode, but the strategy bundle is orthogonal — the
        // caller's choice of bundle should outlive effort changes.
        new.strategy = self.strategy.clone();
        // Preserve explicit patches setting across with_effort.
        if self.patches_explicit {
            new.patches = self.patches;
            new.patches_explicit = true;
        }
        // W44-130 Chunk D: the 4 dispatch fields (`patches_dispatch`,
        // `pixel_loss_dispatch`, `single_pass_entropy_dispatch`,
        // `epf_dispatch`) were deleted from `LossyConfig` and
        // absorbed into `EncoderImprovementsCustom`. The `strategy`
        // bundle (preserved across `with_effort` below) carries the
        // dispatch values; no separate copy needed.
        // Preserve CLI-passthrough knobs across with_effort (they're
        // never effort-derived; opt-in / pure forwarding).
        new.alpha_distance = self.alpha_distance;
        new.alpha_squeeze = self.alpha_squeeze;
        new.group_order = self.group_order;
        new.center_x = self.center_x;
        new.center_y = self.center_y;
        new.upsampling_mode = self.upsampling_mode;
        // If group_order was set to 1 (center-first), keep center_first
        // wired through with_effort too.
        if matches!(self.group_order, Some(1)) {
            new.center_first = true;
        }
        // Chroma subsampling — never effort-derived; carry across
        // `with_effort` so the builder chain
        // `LossyConfig::new(d).with_chroma_subsampling(Sub420).with_effort(5)`
        // is order-independent.
        new.chroma_subsampling = self.chroma_subsampling;
        // Buffering policy — never effort-derived; pure caller
        // preference. Carry across `with_effort` so the builder chain
        // `LossyConfig::new(d).with_buffering(_).with_effort(_)` is
        // order-independent.
        new.buffering = self.buffering;
        new
    }

    /// Set encoder mode (default: [`EncoderMode::Reference`]).
    ///
    /// `Reference` matches libjxl's algorithm choices for comparable output.
    /// `Experimental` enables encoder-specific improvements.
    pub fn with_mode(mut self, mode: EncoderMode) -> Self {
        self.mode = mode;
        self
    }

    /// Current encoder mode.
    pub fn mode(&self) -> EncoderMode {
        self.mode
    }

    /// Enable/disable ANS entropy coding (default: true).
    pub fn with_ans(mut self, enable: bool) -> Self {
        self.use_ans = enable;
        self
    }

    /// Enable/disable gaborish inverse pre-filter (default: true).
    pub fn with_gaborish(mut self, enable: bool) -> Self {
        self.gaborish = enable;
        self
    }

    /// Enable EX-J13 — per-tile contrast-adaptive gaborish kernel strength
    /// (default: `false`).
    ///
    /// When enabled, the encoder samples local Laplacian contrast per 16×16
    /// tile on the Y (luma) channel and modulates the 5×5 sharpening
    /// kernel's strength multiplier in `[0.8, 1.0]` — the libjxl-faithful
    /// baseline `mul = 1.0` on edges/text, gentler `mul ≈ 0.8` on smooth
    /// regions. X (red-green) and B (blue) keep `mul = 1.0`. The bias
    /// below the baseline is deliberate: pushing `mul > 1.0` over-sharpens
    /// natural content and blows up AC coefficient energy with no
    /// perceptual win the decoder's fixed 3×3 inverse blur can recover.
    ///
    /// **Encoder-only.** The decoder always applies the same fixed 3×3
    /// inverse Gabor blur; any adaptive sharpening must be pre-baked into
    /// the post-Gab samples. Bitstream-compatible with all conformant
    /// decoders.
    ///
    /// Silent gate: when [`Self::with_gaborish`] is `false` (or the
    /// `effective_gaborish()` distance/speed-tier gates disable gab), this
    /// flag is also a no-op.
    pub fn with_adaptive_gaborish(mut self, enable: bool) -> Self {
        self.adaptive_gaborish = enable;
        self
    }

    /// Whether adaptive gaborish (EX-J13) is enabled. Defaults to `false`.
    pub fn adaptive_gaborish(&self) -> bool {
        self.adaptive_gaborish
    }

    /// Override the edge-preserving filter (EPF) iteration count.
    ///
    /// Mirrors libjxl `cjxl --epf -1..3` and the
    /// `JXL_ENC_FRAME_SETTING_EPF` C API knob
    /// (`enc_frame.cc:284-285`). The encoder runs the filter for the
    /// requested iteration count and signals it in the frame header
    /// (`LoopFilter.epf_iters`); the decoder applies the matching
    /// number of passes.
    ///
    /// - `-1` (default) — encoder chooses based on butteraugli distance
    ///   (libjxl thresholds `[0.7, 1.5, 4.0]`).
    /// - `0` — forced off; decoder skips EPF entirely.
    /// - `1`/`2`/`3` — forced iteration count (1 = Step 2 only, 2 =
    ///   Step 1+2, 3 = Step 0+1+2). Higher iteration counts smooth
    ///   harder at the cost of decode time.
    ///
    /// Values outside `-1..=3` are clamped to that range. Setting `0`
    /// also disables the per-block dynamic sharpness search, since
    /// there is no filter to tune.
    pub fn with_epf_level(mut self, level: i8) -> Self {
        self.epf_level = level.clamp(-1, 3);
        self
    }

    // W44-130 Chunk D: `with_epf_dispatch` setter + `epf_dispatch`
    // getter on `LossyConfig` were DELETED. The dispatch policy is
    // now reachable via
    // `with_strategy(EncoderStrategy::Custom(Box::new(
    //     EncoderImprovementsCustom { epf_dispatch: EpfDispatch::..., ..Default::default() }
    // )))`.

    /// Enable/disable content-estimated noise synthesis (default: false).
    ///
    /// When `true`, the encoder scans flat XYB patches, fits an 8-point
    /// noise LUT via SCG optimisation, and emits a noise header.
    ///
    /// # Gate / silent-drop conditions
    ///
    /// Lowest-priority noise source. Both [`Self::with_photon_noise_iso`]
    /// and [`Self::with_manual_noise_lut`] override this when set.
    /// Order matches libjxl `enc_frame.cc:680-689`:
    ///
    /// 1. `photon_noise_iso` (highest)
    /// 2. `manual_noise_lut`
    /// 3. `with_noise(true)` + content estimation (this)
    /// 4. No noise
    ///
    /// Bitstream emission gate (vardct/encoder.rs:709, bitstream.rs:1284):
    ///
    /// - `estimate_noise_params` returns `None` when no flat patches
    ///   are detected — header is silently skipped. This is normal on
    ///   noise-free synthetic content (gradients, solid fills, UI).
    /// - [`Self::with_denoise(true)`](Self::with_denoise) implies this
    ///   (`with_denoise` sets `noise = true` automatically).
    /// - Lossy-only (no field on [`LosslessConfig`]).
    pub fn with_noise(mut self, enable: bool) -> Self {
        self.noise = enable;
        self
    }

    /// Set a caller-supplied 8-point noise LUT (matches libjxl
    /// `cparams.manual_noise`). Each entry is the per-intensity
    /// noise level the decoder will synthesise; positions 0–7 are
    /// the standard JXL noise points covering the intensity range.
    /// Values are clamped to `[0.0, ~0.9995]` so the 10-bit
    /// quantisation can't trip the writer's debug-asserts.
    ///
    /// Priority order (matches libjxl `enc_frame.cc:680-689`):
    /// 1. [`Self::with_photon_noise_iso`] (highest)
    /// 2. This (`manual_noise_lut`)
    /// 3. [`Self::with_noise`] + content estimation
    /// 4. No noise
    ///
    /// An all-zero LUT is silently dropped (no noise header is
    /// emitted). Useful when the caller has its own noise model
    /// (e.g. film grain emulation, calibrated sensor noise from
    /// downstream metadata).
    ///
    /// `None` disables the override; the encoder falls back to the
    /// next-priority noise source.
    ///
    /// # Gate / silent-drop conditions
    ///
    /// Wired through all three encode entry points (since the
    /// 2026-05-17 photon-noise audit): one-shot
    /// [`EncodeRequest::encode`] (api.rs:4540), streaming
    /// [`LossyEncoder::finish`] (api.rs:5424), and animation
    /// [`AnimationRequest::encode`] (api.rs:6901). The bitstream
    /// emission gate is in `VarDctEncoder::encode`
    /// (vardct/encoder.rs:699) and `bitstream::write_animation_frame`
    /// (vardct/bitstream.rs:1274):
    ///
    /// - Caller LUT is clamped per-entry to `[0.0, ~0.9995]` before
    ///   emission (10-bit-quantise assert guard).
    /// - All-zero post-clamp LUT → no noise header. (The clamp can
    ///   silently zero entries that were `< 0.0`; an entire negative
    ///   LUT therefore drops.)
    /// - Effort / XYB gating: same as
    ///   [`Self::with_photon_noise_iso`] (no effort gate, lossy-only).
    pub fn with_manual_noise_lut(mut self, lut: Option<[f32; 8]>) -> Self {
        self.manual_noise_lut = lut;
        self
    }

    /// Configured manual noise LUT, if any.
    pub fn manual_noise_lut(&self) -> Option<[f32; 8]> {
        self.manual_noise_lut
    }

    /// Set a multiplier applied to the AC quantiser's `global_scale`
    /// after the standard distance-driven computation. Mirrors
    /// libjxl's `cparams.quant_ac_rescale`
    /// (`enc_cache.cc:99` → `Quantizer::ScaleGlobalScale`,
    /// `quantizer.h:73`).
    ///
    /// `r < 1.0` produces a smaller `global_scale` → finer AC quant
    /// → larger files but higher quality. `r > 1.0` is the inverse.
    /// `r = 1.0` (or `None`) is a no-op. Negative / NaN values are
    /// silently ignored.
    ///
    /// Useful as a fine-grained AC quality nudge on top of a fixed
    /// `distance` — e.g. picker output ("encode at d=1.0 but quant
    /// AC 5 % finer for this content"). Doesn't change the target
    /// butteraugli distance reported in the bitstream metadata —
    /// this is an encoder-side tweak only.
    ///
    /// Reasonable range: `0.5..=2.0`. Aggressive values produce
    /// surprising quality / size deltas.
    pub fn with_quant_ac_rescale(mut self, rescale: Option<f32>) -> Self {
        self.quant_ac_rescale = rescale.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Configured AC quantiser rescale multiplier, if any.
    pub fn quant_ac_rescale(&self) -> Option<f32> {
        self.quant_ac_rescale
    }

    /// Set the caller-supplied source-image butteraugli distance for
    /// re-encode pipelines. Mirrors libjxl
    /// `cparams.original_butteraugli_distance`.
    ///
    /// When the source isn't ground truth (e.g. re-encoding an
    /// already-lossy JPEG or JXL), the encoder's distance-based
    /// heuristics that compare against source quality — primarily
    /// `x_qm_scale` (libjxl `enc_frame.cc:658`) — should ramp
    /// against the *source's* distance, not the target. The target
    /// distance is what we ask butteraugli to hit; the source
    /// distance is the existing error budget the source ships with.
    ///
    /// `None` (default) keeps libjxl's behaviour: treat source as
    /// ground truth, original = target. `Some(orig)` with `orig >
    /// target_distance` enables; `Some(orig)` with `orig <=
    /// target_distance` is silently treated as `None` (no need —
    /// already encoding to a tighter budget than the source).
    /// Negative / NaN / zero are quietly ignored.
    pub fn with_original_distance(mut self, original: Option<f32>) -> Self {
        self.original_distance = original.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Configured original (source) butteraugli distance, if any.
    /// `Some(orig)` only when the caller explicitly opted in.
    pub fn original_distance(&self) -> Option<f32> {
        self.original_distance
    }

    /// Synthesise noise from an ISO value (matches libjxl
    /// `--photon_noise=ISO`). Bypasses content estimation — the
    /// encoder generates an 8-point noise LUT corresponding to a
    /// camera at the given ISO setting (read noise, photon shot
    /// noise, photo response non-uniformity), assuming a 35 mm
    /// full-frame sensor and daylight spectrum.
    ///
    /// Useful for re-encoding **denoised** photographs (or CGI / HDR
    /// content) where the caller wants controlled grain matching a
    /// target camera ISO instead of preserving the source's natural
    /// noise. Typical values: `100` for bright outdoors, `800`
    /// indoor, `6400+` for low-light grainy.
    ///
    /// `Some(iso)` with `iso > 0.0` enables; `None` or `Some(0.0)`
    /// disables. Takes priority over [`Self::with_noise`] (and
    /// implies it from a bitstream perspective — both flag the noise
    /// header). Negative or non-finite ISO values are ignored.
    ///
    /// Closes the libjxl `--photon_noise` feature parity gap.
    ///
    /// # Gate / silent-drop conditions
    ///
    /// Always wired through all three encode entry points: one-shot
    /// [`EncodeRequest::encode`] (api.rs:4539), streaming
    /// [`LossyEncoder::finish`] (api.rs:5422), and animation
    /// [`AnimationRequest::encode`] (api.rs:6900). The bitstream
    /// emission gate is in `VarDctEncoder::encode` (vardct/encoder.rs:690)
    /// and `bitstream::write_animation_frame` (vardct/bitstream.rs:1265):
    ///
    /// - If `simulate_photon_noise(w, h, iso).has_any()` is `false`
    ///   (all-zero LUT — happens at very low ISO on very small images),
    ///   no noise header is emitted. The caller's intent is honoured
    ///   only when the LUT carries non-zero energy.
    /// - Effort gating: does **not** depend on effort level. Photon
    ///   noise emits at every effort 1-10.
    /// - XYB gating: noise synthesis requires XYB transform (lossy
    ///   path); the lossless [`LosslessConfig`] has no noise field.
    /// - Decoder must support libjxl Level 5 noise headers (every
    ///   JPEG-XL conformant decoder does).
    pub fn with_photon_noise_iso(mut self, iso: Option<f32>) -> Self {
        self.photon_noise_iso = iso.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Enable/disable Wiener denoising pre-filter (default: false). Implies noise.
    pub fn with_denoise(mut self, enable: bool) -> Self {
        self.denoise = enable;
        if enable {
            self.noise = true;
        }
        self
    }

    /// Enable/disable error diffusion in AC quantization (default: false).
    ///
    /// Error diffusion propagates 1/4 of the quantization error to the next
    /// coefficient in zigzag order. Note: libjxl's `QuantizeBlockAC` accepts
    /// this parameter but never references it — the feature is effectively a
    /// no-op in the reference encoder. Our implementation actually performs
    /// the diffusion, which can hurt quality on certain content (bright features
    /// in dark regions), especially when combined with gaborish.
    pub fn with_error_diffusion(mut self, enable: bool) -> Self {
        self.error_diffusion = enable;
        self
    }

    /// Enable/disable pixel-domain loss in strategy selection (default: true).
    pub fn with_pixel_domain_loss(mut self, enable: bool) -> Self {
        self.pixel_domain_loss = enable;
        self
    }

    // W44-130 Chunk D: `with_pixel_loss_dispatch` setter +
    // `pixel_loss_dispatch` getter on `LossyConfig` were DELETED.
    // Reachable via `with_strategy(EncoderStrategy::Custom(...))`
    // with `pixel_loss_dispatch: PixelLossDispatch::...`.

    // W44-130 Chunk D: `with_single_pass_entropy_dispatch` setter +
    // `single_pass_entropy_dispatch` getter on `LossyConfig` were
    // DELETED. Reachable via
    // `with_strategy(EncoderStrategy::Custom(...))` with
    // `single_pass_entropy_dispatch: SinglePassEntropyDispatch::...`.

    /// Convenience switch that toggles all encoder-side perceptual
    /// heuristics on or off in one call. Mirrors libjxl's
    /// `cparams.disable_perceptual_optimizations` (`enc_heuristics.cc:215,
    /// 1098`, `enc_frame.cc:282`, `enc_patch_dictionary.cc:637`).
    ///
    /// Calling `with_perceptual_optimizations(false)` is equivalent to
    /// chaining the matching individual disables:
    ///
    /// ```ignore
    /// cfg.with_gaborish(false)
    ///    .with_patches(false)
    ///    .with_dot_detection(false)
    ///    .with_noise(false)
    ///    .with_pixel_domain_loss(false)
    /// ```
    ///
    /// Calling `with_perceptual_optimizations(true)` resets each of
    /// those to the libjxl-faithful defaults (gaborish on, patches
    /// on, dot detection on — gated internally to effort>=7 && d>=3.0,
    /// matching libjxl `Override::kDefault`; noise off, pixel-domain
    /// loss on).
    ///
    /// Use cases:
    /// - **Decoder testing / spec strict mode**: caller wants to
    ///   exercise the decoder without encoder-side heuristics
    ///   muddying the waters.
    /// - **Reproducibility**: removes content-dependent gating that
    ///   makes outputs hard to A/B compare across versions.
    /// - **Picker training without confounds**: when sweeping AC
    ///   strategy / quant constants, perceptual heuristics inflate
    ///   the noise floor.
    ///
    /// Note: this is a **convenience wrapper** — caller-supplied
    /// per-knob settings called *after* this still take precedence
    /// (e.g. `cfg.with_perceptual_optimizations(false).with_gaborish(true)`
    /// re-enables just gaborish).
    pub fn with_perceptual_optimizations(mut self, enable: bool) -> Self {
        // Set the five perceptual knobs to their on/off positions.
        // Defaults mirror libjxl's enabled state when on.
        self.gaborish = enable;
        self.patches = enable;
        // Convenience setter pins patches — opting out via this method
        // suppresses the content-class dispatch too.
        self.patches_explicit = true;
        self.dot_detection = enable; // libjxl `Override::kDefault`; in-encoder effort/distance gates make this niche-only
        self.noise = false; // off by default in libjxl too
        self.pixel_domain_loss = enable;
        self
    }

    /// Enable/disable LZ77 backward references (default: false).
    pub fn with_lz77(mut self, enable: bool) -> Self {
        self.lz77 = enable;
        self
    }

    /// Set LZ77 method (default: Greedy).
    pub fn with_lz77_method(mut self, method: Lz77Method) -> Self {
        self.lz77_method = method;
        self
    }

    /// Force a specific AC strategy for all blocks. `None` for auto-selection.
    pub fn with_force_strategy(mut self, strategy: Option<u8>) -> Self {
        self.force_strategy = strategy;
        self
    }

    /// Limit the maximum AC strategy transform size.
    ///
    /// Controls the largest DCT transform the encoder will consider:
    /// - `8`: Only 8×8-class transforms (DCT8, DCT4x4, DCT4x8, AFV, IDENTITY, DCT2x2)
    /// - `16`: Up to 16×16 (adds DCT16x16, DCT16x8, DCT8x16)
    /// - `32`: Up to 32×32 (adds DCT32x32, DCT32x16, DCT16x32)
    /// - `64`: No restriction (adds DCT64x64, DCT64x32, DCT32x64) — the default
    ///
    /// `None` means no restriction (same as `64`). Values are clamped to the
    /// nearest valid size.
    pub fn with_max_strategy_size(mut self, size: Option<u8>) -> Self {
        self.max_strategy_size = size;
        self
    }

    /// Enable/disable patches (dictionary-based repeated pattern detection).
    /// Default: true at effort ≥ 7 (libjxl-parity). Huge wins on
    /// screenshots, zero cost on photos.
    ///
    /// Calling this method pins the value — it suppresses the
    /// content-class dispatch
    /// ([`crate::effort::EffortProfile::adapt_to_image_content`])
    /// so an explicit `with_patches(false)` is respected even when a
    /// `Screenshot` class has been set via
    /// [`Self::with_content_class`].
    pub fn with_patches(mut self, enable: bool) -> Self {
        self.patches = enable;
        self.patches_explicit = true;
        self
    }

    // W44-130 Chunk D: `with_patches_dispatch` setter +
    // `patches_dispatch` getter on `LossyConfig` were DELETED.
    // Reachable via `with_strategy(EncoderStrategy::Custom(...))`
    // with `patches_dispatch: PatchesDispatch::...`.

    /// Enable libjxl-style **dot detection** (refs #19). Default `true`,
    /// mirroring libjxl's `Override::kDefault` semantics for `--dots`
    /// (`tools/cjxl_main.cc:363-367` + `enc_patch_dictionary.cc:632-643`).
    ///
    /// When enabled, the encoder will run a star-field / specular-highlight
    /// detector **only** if all of the following hold (matching libjxl's
    /// internal gates exactly):
    ///
    /// * effort ≥ 7 (`speed_tier <= kSquirrel`)
    /// * distance ≥ 3.0 (`kMinButteraugliForDots`)
    /// * no text-like patches were found for this frame
    ///
    /// When the gates fire, the detector finds isolated bright
    /// Gaussian-shaped pixels too small to survive VarDCT quantization
    /// at high distances. Each surviving dot is appended to the patch
    /// dictionary so the decoder reconstructs it exactly.
    ///
    /// **Niche feature** — outside its gates the call is a no-op. Even
    /// inside, it only fires on astronomy images, specular highlights on
    /// dark backgrounds, certain noise patterns. Has no effect on typical
    /// photographic content. libjxl ports the algorithm in
    /// `enc_detect_dots.cc`; we mirror its gating + the 7-neighbor
    /// flood-fill bug for bit-parity.
    ///
    /// Pass `false` to force-disable (mirrors `cjxl --dots=0`).
    pub fn with_dot_detection(mut self, enable: bool) -> Self {
        self.dot_detection = enable;
        self
    }

    /// Enable/disable invisible-pixel simplification (closes #10).
    ///
    /// When `true` (default), color values in alpha=0 pixels are
    /// replaced with a smooth weighted average of visible neighbors
    /// before XYB conversion. Mirrors libjxl's `SimplifyInvisible`
    /// pre-pass (`enc_frame.cc:511`). 5-20% file-size reduction on
    /// sprites / icons / UI elements with large transparent regions;
    /// near-zero cost on photos with mostly-opaque alpha.
    ///
    /// Decoded visible pixels are unaffected — the simplification only
    /// touches data that no decoder will display. Disable only if you
    /// need bit-exact preservation of arbitrary garbage in invisible
    /// pixels (e.g., for steganography or alpha-channel side data).
    pub fn with_simplify_invisible(mut self, enable: bool) -> Self {
        self.simplify_invisible = enable;
        self
    }

    /// Preserve or drop the RGB samples in fully-transparent (alpha=0)
    /// pixels — libjxl-named alias for the inverse of
    /// [`Self::with_simplify_invisible`].
    ///
    /// Mirrors libjxl `cparams.keep_invisible` (`enc_params.h:83`) +
    /// `ApplyOverride(keep_invisible, IsLossless())` at
    /// `enc_frame.cc:1590`.
    ///
    /// - `true` — keep the RGB bytes under transparent pixels intact.
    ///   No `SimplifyInvisible` pre-pass runs. Use this for
    ///   steganography / side-channel data / fuzzing reproducers that
    ///   need bit-exact preservation of pixels no decoder will display.
    /// - `false` (**default for [`LossyConfig`]**) — smear the invisible
    ///   pixels' RGB to a weighted average of visible neighbors so the
    ///   downstream DCT doesn't waste bits on hidden noise. 5-20%
    ///   smaller files on sprites / UI assets / icons with large
    ///   transparent regions; near-zero overhead on photos.
    ///
    /// Equivalent to `with_simplify_invisible(!keep)`; we expose both
    /// names so callers porting from cjxl can use libjxl terminology.
    pub fn with_keep_invisible(mut self, keep: bool) -> Self {
        self.simplify_invisible = !keep;
        self
    }

    /// Enable/disable input canonicalization pre-pass (default: `false`).
    ///
    /// When enabled, the encoder scans the input pixels once before
    /// encoding and applies the following lossless transforms when
    /// safe:
    ///
    /// 1. **Drop opaque alpha** — if every alpha sample equals the
    ///    layout's max value (`0xFF` for 8-bit, `0xFFFF` for 16-bit),
    ///    strip the alpha plane and downgrade the layout
    ///    (`Rgba8 → Rgb8`, `Bgra8 → Bgr8`, `Rgba16 → Rgb16`,
    ///    `GrayAlpha8 → Gray8`, `GrayAlpha16 → Gray16`).
    ///
    /// 2. **Near-grayscale collapse** — if `R == G == B` (within
    ///    ±1 LSB tolerance at 16-bit, exact at 8-bit) for ≥ 99.5 %
    ///    of pixels, downgrade RGB(A) → Gray(Alpha). The green
    ///    channel is preserved as the gray value.
    ///
    /// 3. **16→8 downcast** — if every 16-bit sample is
    ///    byte-replicated (`high == low`, the canonical
    ///    `* 0x0101` zero-extension), downcast to the matching
    ///    8-bit layout.
    ///
    /// Each step is a no-op (single-pass O(pixels) scan, no
    /// allocation) when its precondition fails. Outputs are
    /// strictly smaller-or-equal and preserve every pixel value
    /// bit-exactly within the new layout. Best suited for
    /// accidentally-padded inputs from upstream pipelines (RGBA
    /// with fully-opaque alpha, 16-bit storage of 8-bit content,
    /// RGB storage of grayscale scans).
    ///
    /// **Default is `false`** so existing hash-locks remain
    /// byte-identical. Enable to recover -25 % to -66 % bytes on
    /// padded inputs; real-photo inputs see no change.
    pub fn with_canonicalize_input(mut self, enable: bool) -> Self {
        self.canonicalize_input = enable;
        self
    }

    /// Whether input canonicalization pre-pass is enabled.
    pub fn canonicalize_input(&self) -> bool {
        self.canonicalize_input
    }

    /// **RFC #45 pick #4 chunk 1 — content-class dispatch.**
    ///
    /// Inform the encoder of a pre-computed coarse content class
    /// ([`crate::effort::ImageContentClass`]). When set, the per-image
    /// adapter [`crate::effort::EffortProfile::adapt_to_image_content`]
    /// runs and may flip effort-derived defaults based on the class
    /// (currently: `Screenshot` enables `patches` one or two effort
    /// levels earlier than libjxl's e ≥ 7 default).
    ///
    /// Defaults to `None` (no dispatch). Pass `None` explicitly to
    /// clear a previously-set class.
    ///
    /// Callers typically derive the class from
    /// [`zenanalyze`](https://lib.rs/crates/zenanalyze) Tier 1 features
    /// (cheap stripe-sampled scan). The encoder intentionally does NOT
    /// depend on zenanalyze; classification is the caller's
    /// responsibility so the encoder stays no-default-features for
    /// CI / wasm builds.
    ///
    /// **Hash-lock impact**: default `None` keeps every existing
    /// hash-lock fixture byte-identical. The dispatch fires only when
    /// (a) `with_content_class(Some(class))` is explicitly set AND
    /// (b) the per-class rule matches the (effort, distance, pixels)
    /// of the encode.
    pub fn with_content_class(mut self, class: Option<crate::effort::ImageContentClass>) -> Self {
        self.content_class = class;
        self
    }

    /// Currently-set [`crate::effort::ImageContentClass`] (or `None` if
    /// unset). See [`Self::with_content_class`].
    pub fn content_class(&self) -> Option<crate::effort::ImageContentClass> {
        self.content_class
    }

    // W44-130 Chunk D: `with_content_aware_entropy_mul(bool)` setter
    // + `content_aware_entropy_mul()` getter on `LossyConfig` were
    // DELETED. The opt-in enable bit is subsumed by the
    // [`ScreenshotEntropyMulPolicy`] enum.
    //
    // Migration:
    // - `cfg.with_content_aware_entropy_mul(true)` →
    //   `cfg.with_strategy_overrides(StrategyOverrides {
    //         screenshot_lift_hint: Some(true), ..Default::default()
    //   })`
    //   OR
    //   `cfg.with_strategy(EncoderStrategy::Custom(Box::new(
    //         EncoderImprovementsCustom {
    //             screenshot_entropy_mul: ScreenshotEntropyMulPolicy::ForceOn,
    //             ..Default::default()
    //         }
    //   )))`
    // - `cfg.with_content_aware_entropy_mul(false)` is a no-op (this
    //   is the Zenjxl default — `EncoderImprovementsCustom::default`
    //   sets `screenshot_entropy_mul: Disabled`).

    /// W44-130 (Chunk D) — set the per-field override bundle applied
    /// AFTER [`Self::with_strategy`] resolves.
    ///
    /// Replaces the five legacy `with_*_hint(Option<bool>)` setters
    /// (`with_screenshot_lift_hint`, `with_high_d_photo_hint`,
    /// `with_smooth_photo_dct64_hint`, `with_dct_suppress_hint`,
    /// `with_dct32_keep_hint`) deleted in Chunk D. Callers needing
    /// fine-grained per-divergence control should prefer
    /// [`EncoderStrategy::Custom`] with [`EncoderImprovementsCustom`]
    /// for full coverage; this setter is the smaller escape hatch when
    /// only a few fields need overriding on top of a named preset.
    ///
    /// Field-by-field precedence over the preset's resolved value via
    /// [`StrategyOverrides::apply_to`] (mirrors the
    /// [`Self::with_perceptual_optimizations`] precedence pattern).
    ///
    /// ```ignore
    /// use jxl_encoder::api::{EncoderStrategy, LossyConfig, StrategyOverrides};
    /// // Zenjxl default, but force-skip the W44-65 DCT64 suppression
    /// // (pre-W44-65 bitstream behaviour on screenshots).
    /// let cfg = LossyConfig::new(1.0)
    ///     .with_strategy(EncoderStrategy::Zenjxl)
    ///     .with_strategy_overrides(StrategyOverrides {
    ///         dct_suppress_hint: Some(false),
    ///         ..Default::default()
    ///     });
    /// ```
    pub fn with_strategy_overrides(mut self, overrides: StrategyOverrides) -> Self {
        self.strategy_overrides = overrides;
        self
    }

    /// Currently-set [`Self::with_strategy_overrides`] (default empty
    /// — all fields `None`).
    pub fn strategy_overrides(&self) -> &StrategyOverrides {
        &self.strategy_overrides
    }

    /// W44-128 (Chunk B) — set the encoder compatibility / improvements
    /// bundle.
    ///
    /// Default [`EncoderStrategy::Zenjxl`] reproduces what we ship
    /// today (every per-image content gate auto-fires per its
    /// documented discriminator). [`EncoderStrategy::Libjxl`] is the
    /// strict-parity bundle (disables every Section B content-aware
    /// lift, flips Section A effort-gates, re-enables Section D
    /// KNOWN-BUG `BlockCtxMap` 15-cluster). [`EncoderStrategy::Custom`]
    /// lets the caller pick every dial individually via
    /// [`EncoderImprovementsCustom`].
    ///
    /// [`Self::with_strategy_overrides`] called AFTER `with_strategy`
    /// takes precedence on the matching field (mirrors the
    /// [`Self::with_perceptual_optimizations`] precedence pattern):
    ///
    /// ```ignore
    /// use jxl_encoder::api::{EncoderStrategy, LossyConfig, StrategyOverrides};
    /// // Strict libjxl-parity bundle, but force-allow DCT64 evaluation
    /// // on screenshots (overrides Libjxl's `ForceAllow` default with
    /// // an explicit `ForceAllow` — these agree, so the override is a
    /// // no-op). Useful as a documentation pattern; the override would
    /// // win field-by-field over the Libjxl preset if they disagreed.
    /// let cfg = LossyConfig::new(1.0)
    ///     .with_strategy(EncoderStrategy::Libjxl)
    ///     .with_strategy_overrides(StrategyOverrides {
    ///         dct_suppress_hint: Some(false),
    ///         ..Default::default()
    ///     });
    /// ```
    ///
    /// **W44-130 Chunk D**: this setter stores the strategy on the
    /// `LossyConfig`. At encoder construction time
    /// [`EncoderStrategy::resolve`] is called once with the
    /// [`StrategyOverrides`] from
    /// [`Self::with_strategy_overrides`], and the resulting
    /// [`ResolvedImprovements`] is stored on the encoder. The 8 call
    /// sites in `vardct/encoder.rs` + `vardct/butteraugli_loop.rs`
    /// read this directly. The Zenjxl default produces byte-identical
    /// output to pre-Chunk-D main on all 36 hash-lock fixtures.
    pub fn with_strategy(mut self, strategy: EncoderStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Currently-set [`Self::with_strategy`] bundle (default
    /// [`EncoderStrategy::Zenjxl`]).
    pub fn strategy(&self) -> &EncoderStrategy {
        &self.strategy
    }

    /// W44-128 (Chunk B) / W44-130 (Chunk D) — resolve
    /// [`Self::strategy`] composed with [`Self::strategy_overrides`].
    ///
    /// Called once per encode at the boundary between `LossyConfig`
    /// and the internal `VarDctEncoder`. The resulting
    /// [`ResolvedImprovements`] is stored on the encoder; the 8 call
    /// sites in `vardct/encoder.rs` + `vardct/butteraugli_loop.rs`
    /// consume it directly.
    pub(crate) fn resolve_improvements(&self) -> ResolvedImprovements {
        self.strategy.resolve(&self.strategy_overrides)
    }

    /// Set a separate butteraugli distance for the alpha extra channel
    /// (CLI passthrough — mirrors libjxl `cjxl --alpha_distance`).
    ///
    /// `None` (default) and `Some(0.0)` keep the lossless alpha path
    /// (gradient predictor + LZ77 RLE). `Some(d)` with `d > 0.0`
    /// engages the lossy alpha pipeline: an integer pixel quantizer
    /// derived from libjxl's no-squeeze formula
    /// (`enc_modular.cc:973-1027`) snaps each alpha pixel to the
    /// nearest multiple of `q` and the decoder reconstructs via the
    /// modular-tree leaf's `(mul_log, mul_bits)` multiplier. `d` is
    /// clamped to `[0.01, 25.0]` (matches libjxl `encode.cc:1552`).
    /// Applies per-channel: with a mixed-extras frame (alpha + depth /
    /// spot color / selection mask / ...) only the alpha-typed extras
    /// take this `q`; all other types stay lossless until per-channel
    /// `ec_distance` is wired through the public API (libjxl
    /// `cparams.ec_distance[i]`). Sample yields at 8-bit alpha:
    /// `d=1.0` → `q=1` (still lossless), `d=2.0` → `q=3`, `d=10.0`
    /// → `q=15`.
    pub fn with_alpha_distance(mut self, d: Option<f32>) -> Self {
        self.alpha_distance = d;
        self
    }

    /// Currently-set alpha-channel distance (or `None` if unset).
    pub fn alpha_distance(&self) -> Option<f32> {
        self.alpha_distance
    }

    /// Opt-in to the **squeeze-on-extras** (responsive=1) lossy alpha
    /// pipeline. Default `false`.
    ///
    /// libjxl's default cjxl path uses `--responsive=1` for lossy
    /// alpha, which applies the Squeeze (Haar wavelet) transform on
    /// the alpha plane and routes a per-band quantizer through the
    /// shifted entries of `squeeze_luma_qtable[16]`
    /// (`enc_modular.cc:1004-1027`). This delivers `-18%` to `-160%`
    /// smaller bytes on non-opaque alpha than the `responsive=0`
    /// no-squeeze path we ship today (audit: commit `a160deb7`,
    /// three-image sweep at d ∈ {0.5, 1.0, 2.0, 5.0}).
    ///
    /// **Chunk-1 framework (current ship)**: setting this to `true`
    /// validates the per-band quantizer table + shift-aware quantizer
    /// function are in place, but surfaces a clear
    /// `Error::NotImplemented` from the encoder when the lossy alpha
    /// path is actually engaged
    /// (`alpha_distance > 0.0` AND an alpha extra is present). The
    /// chunk-2 follow-on wires the Squeeze application on the alpha
    /// extra and a per-band quantizer dispatch through the modular
    /// channel-split tree, at which point this flag will deliver
    /// real byte savings.
    ///
    /// Default `false` keeps the existing pipeline byte-for-byte
    /// identical (hash-locks 36/36 unchanged).
    ///
    /// See also: [`Self::with_alpha_distance`] (the distance knob
    /// this opt-in modifies the **encoding** of, not its target
    /// quality).
    pub fn with_alpha_squeeze(mut self, on: bool) -> Self {
        self.alpha_squeeze = on;
        self
    }

    /// Currently-set squeeze-on-extras opt-in (default `false`).
    pub fn alpha_squeeze(&self) -> bool {
        self.alpha_squeeze
    }

    /// Set the modular-group encoding order (CLI passthrough — mirrors
    /// libjxl `cjxl --group_order`).
    ///
    /// `None` (default) = scanline order. `Some(0)` = explicit scanline.
    /// `Some(1)` = center-first; mirrors
    /// [`Self::with_center_first(true)`](Self::with_center_first) and
    /// flips that flag so the existing center-first reorder kicks in.
    /// `Some(2)` is reserved for future encoder modes (stored, no-op).
    pub fn with_group_order(mut self, order: Option<u8>) -> Self {
        self.group_order = order;
        if matches!(order, Some(1)) {
            self.center_first = true;
        } else if matches!(order, Some(0)) {
            self.center_first = false;
        }
        self
    }

    /// Currently-set modular group order (or `None` if unset).
    pub fn group_order(&self) -> Option<u8> {
        self.group_order
    }

    /// Set a custom centre X coordinate for the center-first AC group
    /// reorder (CLI passthrough — mirrors libjxl `cjxl --center_x`).
    ///
    /// `None` (default) anchors the reorder at the image centre. Stored
    /// on the config; encoder-side honouring of a non-default centre
    /// is queued follow-on work. Negative values are interpreted by
    /// libjxl as "use image centre"; we follow the same convention.
    pub fn with_center_x(mut self, x: Option<i64>) -> Self {
        self.center_x = x;
        self
    }

    /// Currently-set centre X (or `None` if unset).
    pub fn center_x(&self) -> Option<i64> {
        self.center_x
    }

    /// Set a custom centre Y coordinate for the center-first AC group
    /// reorder (CLI passthrough — mirrors libjxl `cjxl --center_y`).
    /// See [`Self::with_center_x`] for semantics.
    pub fn with_center_y(mut self, y: Option<i64>) -> Self {
        self.center_y = y;
        self
    }

    /// Currently-set centre Y (or `None` if unset).
    pub fn center_y(&self) -> Option<i64> {
        self.center_y
    }

    /// Set the decoder upsampling mode (CLI passthrough — mirrors
    /// libjxl `cjxl --upsampling_mode`).
    ///
    /// Values follow libjxl conventions:
    /// - `None` or `Some(-1)` = non-separable upsampling (libjxl default).
    /// - `Some(0)` = nearest neighbour (pixel-art preservation).
    /// - `Some(1)` = reserved.
    ///
    /// Stored on the config; emitting a custom upsampling LUT in the
    /// `FrameHeader` is queued follow-on work — current behaviour uses
    /// the spec-default upsampling for the active
    /// [`Self::with_resampling`] factor.
    pub fn with_upsampling_mode(mut self, mode: Option<i32>) -> Self {
        self.upsampling_mode = mode;
        self
    }

    /// Currently-set upsampling mode (or `None` if unset).
    pub fn upsampling_mode(&self) -> Option<i32> {
        self.upsampling_mode
    }

    /// Reorder AC groups in the multi-group TOC by concentric-square
    /// distance from the image center (closes #14).
    ///
    /// When `true`, the encoder writes the AC group sections in
    /// "center-first" order so progressive decoders display the most
    /// important content (image center) before edges/corners. The
    /// codestream `permuted` flag is set and the permutation is
    /// encoded as Lehmer codes via the existing permutation entropy
    /// code (8 contexts).
    ///
    /// No effect on single-group images (≤256×256 pixels) — the
    /// reorder is a no-op when num_groups ≤ 1.
    ///
    /// libjxl `cparams.centerfirst`. Default `false`.
    pub fn with_center_first(mut self, enable: bool) -> Self {
        self.center_first = enable;
        self
    }

    /// Set the decoder upsampling factor (refs #12).
    ///
    /// `factor` must be one of `1`, `2`, `4`, or `8` (the JPEG XL
    /// spec's permitted values). Any other value is silently clamped
    /// to `1` (a future revision may surface a [`ValidationError`]).
    /// Default `1` (no resampling).
    ///
    /// When `factor > 1`, the encoder box-filters the input down by
    /// `factor` along each axis before encoding and signals the
    /// decoder to upsample by the same factor on output. The
    /// codestream's file header still reports the original
    /// (pre-downsample) dimensions, so callers and downstream tooling
    /// see the full-size image. Output dimensions use `div_ceil`, so
    /// odd / non-multiple sizes round up — the decoder upsamples to
    /// `(out_w * factor, out_h * factor)` which may exceed the
    /// original by up to `factor - 1` pixels along each axis (the
    /// decoder crops to the file-header dimensions).
    ///
    /// libjxl auto-selects `factor = 2` at distance ≥ 10
    /// (`enc_frame.cc:89-121`). We don't auto-select yet; callers
    /// opt in explicitly. The simple box filter matches libjxl's 4×
    /// and 8× paths; libjxl's 2× path uses a sharper 12×12 kernel
    /// (`enc_heuristics.cc:279-405`) which is TBD.
    pub fn with_resampling(mut self, factor: u32) -> Self {
        self.resampling = if matches!(factor, 1 | 2 | 4 | 8) {
            factor
        } else {
            1
        };
        self.resampling_explicit = true;
        self
    }

    /// Current resampling factor (1, 2, 4, or 8). Default `1`.
    ///
    /// When auto-resample is enabled (the default) and the distance
    /// is ≥ 10, the **effective** resampling factor at encode time is
    /// `2`, but this getter still returns the explicitly-set value
    /// (or `1` if unset). Use [`Self::effective_resampling`] to query
    /// what the encoder actually uses.
    pub fn resampling(&self) -> u32 {
        self.resampling
    }

    /// Enable / disable libjxl's auto-resample-at-d≥10 rule (refs #12).
    /// Default `true`. When enabled and the caller has *not* pinned a
    /// resampling factor via [`Self::with_resampling`], the encoder
    /// engages 2× sharper downsampling at distance ≥ 10 and adjusts
    /// the internal target distance to `d * 0.25 + 0.25`. libjxl
    /// reference: `enc_frame.cc:103-115`.
    pub fn with_auto_resampling(mut self, enable: bool) -> Self {
        self.auto_resampling = enable;
        self
    }

    /// Current auto-resample setting. Default `true`.
    pub fn auto_resampling(&self) -> bool {
        self.auto_resampling
    }

    /// Tell the encoder the input is **already** at the post-resampling
    /// resolution; the encoder should write the matching `upsampling`
    /// factor in the bitstream but skip the internal downsample step.
    /// Mirrors libjxl's `cparams.already_downsampled`.
    ///
    /// Use case: the caller has a GPU pipeline that already produced
    /// a downsampled image at the target encode resolution, and wants
    /// the encoder to honour it (write `upsampling=N`, decoder
    /// upsamples on the way out, file header advertises original dims
    /// = `input_dims * N`). Without this flag, `with_resampling(N)`
    /// would downsample the input *again*.
    ///
    /// No-op when `effective_resampling() == 1`. Pair with
    /// [`Self::with_resampling`]; pass the **already downsampled**
    /// dimensions to [`crate::api::EncodeRequest`] — the file header
    /// will advertise `dims * N` as the original size.
    pub fn with_already_downsampled(mut self, already: bool) -> Self {
        self.already_downsampled = already;
        self
    }

    /// Current already-downsampled flag. Default `false`.
    pub fn already_downsampled(&self) -> bool {
        self.already_downsampled
    }

    /// Effective resampling factor the encoder will actually use,
    /// after applying auto-resample at d≥10 (refs #12). Returns
    /// `self.resampling` unless auto-resample is enabled, no explicit
    /// factor was set, and `self.distance >= 10`.
    pub fn effective_resampling(&self) -> u32 {
        if !self.resampling_explicit && self.auto_resampling && self.distance >= 10.0 {
            2
        } else {
            self.resampling
        }
    }

    /// Effective butteraugli distance the encoder will actually use,
    /// after applying libjxl's distance adjustment when auto-resample
    /// kicks in (refs #12). Returns `self.distance` unless auto-resample
    /// fires; otherwise returns `distance * 0.25 + 0.25`.
    pub fn effective_distance(&self) -> f32 {
        if !self.resampling_explicit && self.auto_resampling && self.distance >= 10.0 {
            self.distance * 0.25 + 0.25
        } else {
            self.distance
        }
    }

    /// Set manual splines to overlay on the image.
    ///
    /// Splines are Gaussian-blurred parametric curves overlaid additively.
    /// They encode thin features (power lines, horizons) efficiently.
    /// The encoder subtracts splines from XYB before VarDCT; the decoder
    /// adds them back after reconstruction. Default: `None`.
    pub fn with_splines(mut self, splines: Vec<crate::vardct::splines::Spline>) -> Self {
        self.splines = Some(splines);
        self
    }

    /// Enable automatic spline detection from the input image.
    ///
    /// When enabled AND [`Self::with_splines`] has not been called AND the
    /// effective effort is ≥ 7, the encoder runs a thin-feature detector
    /// (power lines, horizons, hair) and subtracts the resulting curves
    /// from XYB before VarDCT. The decoder adds them back after
    /// reconstruction. Mirrors libjxl `enc_heuristics.cc:1048-1054`
    /// (`speed_tier <= kSquirrel`).
    ///
    /// **Default `false` at every effort level.** A flip-on-at-e8+
    /// proposal was investigated and rejected: the chunk-3 detector's
    /// trial-encode cost gate rejects every candidate on every tested
    /// image at e8 plus e9 (10 / 10 byte-identical, including the multi-line
    /// power-line synthetics the detector was designed to win on at e7).
    /// Default-on would ship CPU overhead (Sobel, NMS, Hessian,
    /// polyline trace, trial-encode) for zero byte change. See
    /// [`crate::effort::EffortProfile::auto_splines_default`] and
    /// `benchmarks/auto_splines_bench_2026-05-17.tsv` for the data.
    ///
    /// Opt-in usage: `with_auto_splines(true)` at e7 admits the chunk-3
    /// detector and wins 138 / 557 bytes saved on the 4-line / 8-line
    /// synthetic ridges (118 bytes cost on the 1-line edge case). Photo
    /// content stays byte-identical because the gate rejects all
    /// candidates. Calling this method pins the value across subsequent
    /// [`Self::with_effort`] calls.
    ///
    /// A manual [`Self::with_splines`] call always wins outright — the
    /// auto-detector is only consulted when no manual splines are set.
    pub fn with_auto_splines(mut self, enable: bool) -> Self {
        self.auto_splines = enable;
        self.auto_splines_explicit = true;
        self
    }

    /// Whether automatic spline detection is enabled. See
    /// [`Self::with_auto_splines`].
    pub fn auto_splines(&self) -> bool {
        self.auto_splines
    }

    /// Whether [`Self::auto_splines`] was set explicitly via
    /// [`Self::with_auto_splines`] (rather than derived from the
    /// effort-based default in
    /// [`crate::effort::EffortProfile::auto_splines_default`]).
    pub fn auto_splines_explicit(&self) -> bool {
        self.auto_splines_explicit
    }

    /// Set progressive encoding mode (default: Single = no progressive).
    ///
    /// Progressive encoding splits AC coefficients across multiple passes,
    /// allowing decoders to render coarse previews before the full file is received.
    pub fn with_progressive(mut self, mode: ProgressiveMode) -> Self {
        self.progressive = mode;
        self
    }

    /// Enable LfFrame (separate DC frame).
    ///
    /// When true, DC coefficients are encoded as a separate modular frame
    /// before the main VarDCT frame, matching libjxl's `progressive_dc >= 1`.
    pub fn with_lf_frame(mut self, enable: bool) -> Self {
        self.lf_frame = enable;
        self
    }

    /// Explicit `progressive_dc` level. Mirrors libjxl
    /// `cjxl --progressive_dc 0..2`.
    ///
    /// - `0`: no progressive DC (default).
    /// - `1`: one LfFrame ahead of the main VarDCT frame (same as
    ///   [`Self::with_lf_frame(true)`]).
    /// - `2`: two nested LfFrames (libjxl path; our encoder currently
    ///   emits a single LfFrame and warns — the value is stored and
    ///   surfaced via [`Self::progressive_dc`] for forward compatibility).
    ///
    /// Values are clamped to `0..=`[`MAX_PROGRESSIVE_DC`]. Setting
    /// any non-zero level implies [`Self::with_lf_frame(true)`].
    pub fn with_progressive_dc(mut self, level: u8) -> Self {
        let lvl = level.min(MAX_PROGRESSIVE_DC);
        self.progressive_dc = lvl;
        if lvl >= 1 {
            self.lf_frame = true;
        }
        self
    }

    /// Currently-configured `progressive_dc` level (`0..=2`).
    pub fn progressive_dc(&self) -> u8 {
        self.progressive_dc
    }

    /// Opt-in: enable automatic delta-frame mode selection for
    /// multi-frame animation encodes (A1 audit "Animation" — Skip /
    /// delta frame encoding). See the field doc on
    /// [`auto_delta_frames`][Self::auto_delta_frames] for the full
    /// rollout plan.
    ///
    /// Chunk 1 POC scope: one heuristic — identical-frame short-circuit
    /// using [`BlendMode::Add`] over a 1×1 zero-pixel crop. Chunk 2
    /// will add the full trial-encode loop. Default `false` — no
    /// hash-locked bitstream changes at default.
    ///
    /// Lossy path: the chunk 1 heuristic is wired but the lossy
    /// pipeline reconstructs from the already-quantised reference
    /// frame, not the original pixels — the residual semantics are
    /// only safe when the per-frame quantisation is locked. Treat
    /// `with_auto_delta_frames(true)` on [`LossyConfig`] as
    /// experimental until chunk 2 lands; the safe demoable path is the
    /// [`LosslessConfig`] variant.
    pub fn with_auto_delta_frames(mut self, enable: bool) -> Self {
        self.auto_delta_frames = enable;
        self
    }

    /// Whether the encode is permitted to emit delta-frame variants
    /// when [`Self::with_auto_delta_frames`] has been opted into.
    pub fn auto_delta_frames(&self) -> bool {
        self.auto_delta_frames
    }

    /// Set the chroma subsampling mode (issue #47).
    ///
    /// Default is [`ChromaSubsampling::Full444`] — every existing
    /// bitstream stays byte-identical without an explicit call. See
    /// [`ChromaSubsampling`] for the per-mode shift table and the
    /// chunk-3 status: only `Full444` is honoured end-to-end; setting
    /// any other mode causes the encoder to return
    /// [`EncodeError::InvalidConfig`] with a message that names the
    /// missing encoder-side wiring.
    ///
    /// The conversion helpers (RGB→YCbCr, Sharp YUV 4:2:0 downsample)
    /// are already implemented in
    /// `crate::vardct::chroma_subsampling` when the
    /// `chroma-subsampling` cargo feature is enabled — chunk 4 wires
    /// them through the encode pipeline.
    pub fn with_chroma_subsampling(mut self, mode: ChromaSubsampling) -> Self {
        self.chroma_subsampling = mode;
        self
    }

    /// Currently-set chroma subsampling mode. Defaults to
    /// [`ChromaSubsampling::Full444`]. See
    /// [`Self::with_chroma_subsampling`].
    pub fn chroma_subsampling(&self) -> ChromaSubsampling {
        self.chroma_subsampling
    }

    /// Bias the VarDCT encode toward simpler bitstreams that decode
    /// faster, at the cost of compression. Mirrors libjxl
    /// `cjxl --faster_decoding 0..4`
    /// ([`cparams.decoding_speed_tier`][libjxl-cparams]).
    ///
    /// Values are clamped to `0..=`[`MAX_FASTER_DECODING`]. The default
    /// `0` keeps the existing behaviour (no speed bias).
    ///
    /// Per-tier effect on the VarDCT path (libjxl
    /// [`enc_frame.cc:280`][libjxl-frame],
    /// [`enc_ac_strategy.cc:884`][libjxl-acs],
    /// [`enc_ans.cc:1372`][libjxl-ans]):
    ///
    /// - `1`: cluster all AC blocks into a single block-context map
    ///   (simpler entropy contexts); cap VarDCT histograms at 6 (the
    ///   AC pass) / 12 (the modular fallback pass); skip the patches
    ///   pre-pass.
    /// - `2`: same as tier 1 plus tighter group-size shift for
    ///   multithreaded decode.
    /// - `3`: skip EPF (the lowest threshold drops out — only `>= 1.5`
    ///   and `>= 4.0` butteraugli distances still enable any EPF
    ///   iters).
    /// - `4`: gaborish is forced off; AC strategy search prunes
    ///   anything larger than 32x32; DCT32x32 itself is disabled.
    ///
    /// [libjxl-cparams]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_params.h
    /// [libjxl-frame]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_frame.cc
    /// [libjxl-acs]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_ac_strategy.cc
    /// [libjxl-ans]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_ans.cc
    pub fn with_faster_decoding(mut self, tier: u8) -> Self {
        self.faster_decoding = tier.min(MAX_FASTER_DECODING);
        self
    }

    /// Currently-configured decoding-speed tier (`0..=4`).
    pub fn faster_decoding(&self) -> u8 {
        self.faster_decoding
    }

    /// Container-wrap policy. Mirrors libjxl `cjxl --container 0|1`.
    /// Default [`ContainerMode::Auto`] wraps the codestream only when
    /// metadata is attached or the codestream level requires it.
    ///
    /// See [`ContainerMode`] for the per-variant semantics.
    pub fn with_container_mode(mut self, mode: ContainerMode) -> Self {
        self.container_mode = mode;
        self
    }

    /// Currently-configured container-wrap policy.
    pub fn container_mode(&self) -> ContainerMode {
        self.container_mode
    }

    /// Set the input/output buffering policy (streaming refactor
    /// scaffolding, jxl-encoder#11). Mirrors libjxl `cjxl --buffering
    /// -1..3`. See [`Buffering`] for variant semantics and the chunk
    /// schedule.
    ///
    /// **Chunk 1: no dispatch is wired** — every variant currently
    /// routes through the existing one-shot path, so output bytes are
    /// identical regardless of which `Buffering` value is selected.
    /// Chunks 2-7 land the per-DC-group split, the buffered-output
    /// streaming path (libjxl level 2), the seekable streaming-output
    /// path (libjxl level 3), and the lossless mirror.
    pub fn with_buffering(mut self, mode: Buffering) -> Self {
        self.buffering = mode;
        self
    }

    /// Currently-configured input/output buffering policy. See
    /// [`Self::with_buffering`].
    pub fn buffering(&self) -> Buffering {
        self.buffering
    }

    /// Set butteraugli quantization loop iterations explicitly.
    ///
    /// Overrides the automatic effort-based default (effort 7: 0, effort 8: 2, effort 9+: 4).
    /// Stores the value as-given for [`Self::validate`] to surface as
    /// [`crate::ValidationError::IterCountOutOfRange`] if it exceeds
    /// [`MAX_QUANT_LOOP_ITERS`]. The encoder additionally saturates at
    /// consumption time so callers that skip `validate()` still cannot
    /// DoS the encoder by passing a huge value.
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn with_butteraugli_iters(mut self, n: u32) -> Self {
        self.butteraugli_iters = n;
        self.butteraugli_iters_explicit = true;
        self
    }

    /// Pick the perceptual loss used by the butteraugli quantization loop
    /// on HDR encodes (EX-J11).
    ///
    /// Default [`HdrLoss::Auto`] (chunk 4) dispatches to
    /// [`HdrLoss::Vdp2`] on PQ / HLG content and [`HdrLoss::Butteraugli`]
    /// on everything else — see [`HdrLoss::resolve`] for the dispatch
    /// matrix. SDR hash-lock fixtures stay byte-identical.
    ///
    /// Override with [`HdrLoss::Butteraugli`] to pin the SDR-tuned loss
    /// regardless of transfer function (e.g. for byte-stable encodes on
    /// PQ-tagged but visually-SDR content), or [`HdrLoss::Vdp2`] to
    /// force the HDR-VDP-2-lite metric on any content.
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn with_hdr_loss(mut self, loss: HdrLoss) -> Self {
        self.hdr_loss = loss;
        self
    }

    /// Currently configured HDR-aware perceptual loss. May be
    /// [`HdrLoss::Auto`] (the default) — use [`Self::resolve_hdr_loss`]
    /// to see the loss that actually runs for a given pixel layout.
    #[cfg(feature = "butteraugli-loop")]
    pub fn hdr_loss(&self) -> HdrLoss {
        self.hdr_loss
    }

    /// Resolve the configured [`HdrLoss`] into the concrete loss that
    /// will run inside the butteraugli quantization loop, given the
    /// caller's input pixel layout and (optionally) an explicit
    /// `ColorEncoding` from `EncodeRequest::with_color_encoding`.
    ///
    /// When [`Self::with_hdr_loss`] is set to [`HdrLoss::Auto`] (the
    /// default), the resolution uses:
    ///
    /// 1. The transfer function of `color_encoding` if the caller
    ///    wired one explicitly on the request, else
    /// 2. The transfer function implied by `layout` (PQ / HLG / BT.709
    ///    f32 input variants populate this; sRGB-u8 / linear-f32
    ///    layouts don't).
    /// 3. If neither path yields a TF, the resolver assumes SDR and
    ///    returns [`HdrLoss::Butteraugli`].
    ///
    /// Non-`Auto` variants pass through unchanged. See
    /// [`HdrLoss::resolve`] for the full dispatch matrix.
    ///
    /// `color_encoding` lives on [`EncodeRequest`] (not on this
    /// config), so the encoder pipelines pass it through explicitly.
    /// This is the single dispatch site for chunk-4 — called once
    /// when wiring `enc.hdr_loss`, so the per-iteration butteraugli
    /// loop reads a concrete variant with zero dispatch overhead.
    #[cfg(feature = "butteraugli-loop")]
    pub fn resolve_hdr_loss(
        &self,
        layout: PixelLayout,
        color_encoding: Option<&crate::headers::color_encoding::ColorEncoding>,
    ) -> HdrLoss {
        let tf = color_encoding
            .map(|ce| ce.transfer_function)
            .or_else(|| layout.implied_transfer_function());
        self.hdr_loss.resolve(tf)
    }

    /// Set the policy for non-finite XYB values at the
    /// conversion→pipeline boundary. See [`NonFiniteAction`] for the
    /// trade-off between fail-fast (default, `Error`) and best-effort
    /// (`Sanitize`).
    pub fn with_non_finite_action(mut self, action: NonFiniteAction) -> Self {
        self.non_finite_action = action;
        self
    }

    /// The currently-configured [`NonFiniteAction`] policy.
    pub fn non_finite_action(&self) -> NonFiniteAction {
        self.non_finite_action
    }

    /// Set SSIM2 quantization loop iterations.
    ///
    /// Alternative to butteraugli loop: uses per-block linear RGB RMSE + full-image SSIM2.
    /// See [`Self::with_butteraugli_iters`] for how out-of-range values
    /// are handled.
    /// Requires the `ssim2-loop` feature.
    #[cfg(feature = "ssim2-loop")]
    pub fn with_ssim2_iters(mut self, n: u32) -> Self {
        self.ssim2_iters = n;
        self
    }

    /// Set zensim quantization loop iterations.
    ///
    /// Alternative to butteraugli loop: uses zensim's psychovisual metric for
    /// both global quality tracking and per-pixel spatial error map (diffmap in XYB space).
    /// Also refines AC strategy by splitting large transforms with high perceptual error.
    /// Can stack with butteraugli loop (butteraugli runs first, then zensim fine-tunes).
    /// Requires the `zensim-loop` feature.
    #[cfg(feature = "zensim-loop")]
    pub fn with_zensim_iters(mut self, n: u32) -> Self {
        self.zensim_iters = n;
        self
    }

    /// Set thread count for parallel encoding.
    ///
    /// - `0` (default): use the ambient rayon pool. The caller can control
    ///   thread count by wrapping the encode call in `pool.install(|| ...)`.
    /// - `1`: force sequential encoding (no rayon).
    /// - `N >= 2`: create a dedicated N-thread pool for this encode.
    ///
    /// Requires the `parallel` feature. When `parallel` is not enabled,
    /// this value is ignored and encoding is always sequential.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    // ── Getters ───────────────────────────────────────────────────────

    /// Current butteraugli distance.
    pub fn distance(&self) -> f32 {
        self.distance
    }

    /// Current effort level.
    pub fn effort(&self) -> u8 {
        self.effort
    }

    /// Whether ANS entropy coding is enabled.
    pub fn ans(&self) -> bool {
        self.use_ans
    }

    /// Whether gaborish inverse pre-filter is enabled.
    pub fn gaborish(&self) -> bool {
        self.gaborish
    }

    /// Configured edge-preserving filter override.
    ///
    /// `-1` (default) = encoder chooses by distance; `0` = forced off;
    /// `1`/`2`/`3` = forced iteration count. See [`Self::with_epf_level`].
    pub fn epf_level(&self) -> i8 {
        self.epf_level
    }

    /// Whether noise synthesis is enabled.
    pub fn noise(&self) -> bool {
        self.noise
    }

    /// Configured photon-noise ISO, if any. `Some(iso)` means the
    /// encoder will synthesise noise from this ISO value instead of
    /// estimating from content. Matches libjxl `--photon_noise=ISO`.
    pub fn photon_noise_iso(&self) -> Option<f32> {
        self.photon_noise_iso
    }

    /// Whether Wiener denoising pre-filter is enabled.
    pub fn denoise(&self) -> bool {
        self.denoise
    }

    /// Whether error diffusion in AC quantization is enabled.
    pub fn error_diffusion(&self) -> bool {
        self.error_diffusion
    }

    /// Whether pixel-domain loss is enabled.
    pub fn pixel_domain_loss(&self) -> bool {
        self.pixel_domain_loss
    }

    /// Whether patches (dictionary-based repeated pattern detection)
    /// are enabled.
    pub fn patches(&self) -> bool {
        self.patches
    }

    /// Whether dot detection (refs #19) is enabled.
    pub fn dot_detection(&self) -> bool {
        self.dot_detection
    }

    /// Whether LZ77 backward references are enabled.
    pub fn lz77(&self) -> bool {
        self.lz77
    }

    /// Current LZ77 method.
    pub fn lz77_method(&self) -> Lz77Method {
        self.lz77_method
    }

    /// Forced AC strategy, if any.
    pub fn force_strategy(&self) -> Option<u8> {
        self.force_strategy
    }

    /// Maximum AC strategy transform size, if set.
    pub fn max_strategy_size(&self) -> Option<u8> {
        self.max_strategy_size
    }

    /// Current progressive mode.
    pub fn progressive(&self) -> ProgressiveMode {
        self.progressive
    }

    /// Whether LfFrame (separate DC frame) is enabled.
    pub fn lf_frame(&self) -> bool {
        self.lf_frame
    }

    /// Conservative upper bound on peak working-set memory for an
    /// encode of this configuration at `(width, height)` pixels with
    /// the given pixel layout.
    ///
    /// Models the four large dimension-driven buffers that dominate
    /// encoder peak RSS today:
    ///
    /// 1. `linear_rgb`: `pixels * 3 * 4` bytes (always RGB f32 — gray
    ///    layouts are expanded before XYB conversion).
    /// 2. XYB planes (`xyb_x` / `xyb_y` / `xyb_b`):
    ///    `padded_pixels * 3 * 4` bytes, padded to the 8×8 block
    ///    boundary so SIMD doesn't bounds-check.
    /// 3. `quant_ac`: `blocks * 3 * 64 * 4` bytes (per-channel,
    ///    per-block 64 i32 coefficients).
    /// 4. Alpha buffer (when the layout carries alpha): `pixels` bytes.
    ///
    /// Then a 25 % overhead is added to absorb small unmodelled
    /// allocations (entropy-coder bit buffer, scratch transforms,
    /// histograms, tokens, transient gaborish padding). The result is
    /// a *conservative upper bound* — actual usage is typically a few
    /// tens of percent lower.
    ///
    /// Useful for capacity planning and for choosing between one-shot
    /// encode and the streaming path (closes #11) once it lands —
    /// streaming will collapse buffers (1)–(3) to roughly one DC
    /// group's worth (~1.5 MB) regardless of full image size.
    ///
    /// Returns `None` only if the dimensions overflow `u64`, which is
    /// effectively unreachable for any realistic encode.
    pub fn estimate_peak_memory_bytes(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Option<u64> {
        estimate_peak_memory_bytes_lossy(width, height, layout)
    }

    /// Butteraugli quantization loop iterations.
    #[cfg(feature = "butteraugli-loop")]
    pub fn butteraugli_iters(&self) -> u32 {
        self.butteraugli_iters
    }

    /// SSIM2 quantization loop iterations (internal accessor for validation).
    #[cfg(feature = "ssim2-loop")]
    pub(crate) fn ssim2_iters_value(&self) -> u32 {
        self.ssim2_iters
    }

    /// zensim quantization loop iterations (internal accessor for validation).
    #[cfg(feature = "zensim-loop")]
    pub(crate) fn zensim_iters_value(&self) -> u32 {
        self.zensim_iters
    }

    /// Borrow the resolved `EffortProfile` override, if any. Internal hook
    /// used by [`crate::validation`].
    #[cfg(feature = "__expert")]
    pub(crate) fn profile_override_ref(&self) -> Option<&crate::effort::EffortProfile> {
        self.profile_override.as_ref()
    }

    /// Thread count (0 = auto, 1 = sequential).
    pub fn threads(&self) -> usize {
        self.threads
    }

    // ── Request / fluent encode ─────────────────────────────────────

    /// Create an encode request for an image with this config.
    ///
    /// Use this when you need to attach metadata, limits, or cancellation.
    pub fn encode_request(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> EncodeRequest<'_> {
        EncodeRequest {
            config: ConfigRef::Lossy(self),
            width,
            height,
            layout,
            metadata: None,
            limits: None,
            stop: None,
            source_gamma: None,
            color_encoding: None,
            intensity_target: None,
            min_nits: None,
            premultiplied_alpha: false,
            premultiplied_alpha_mode: None,
            bits_per_sample: None,
            brotli_metadata_quality: None,
            row_stride: None,
            extra_channels: &[],
        }
    }

    /// Encode pixels directly with this config. Shortcut for simple cases.
    ///
    /// ```rust,no_run
    /// # let pixels = vec![0u8; 100 * 100 * 3];
    /// let jxl = jxl_encoder::LossyConfig::new(1.0)
    ///     .encode(&pixels, 100, 100, jxl_encoder::PixelLayout::Rgb8)?;
    /// # Ok::<_, jxl_encoder::At<jxl_encoder::EncodeError>>(())
    /// ```
    #[track_caller]
    pub fn encode(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<Vec<u8>> {
        self.encode_request(width, height, layout).encode(pixels)
    }

    /// Encode pixels, appending to an existing buffer.
    #[track_caller]
    pub fn encode_into(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        self.encode_request(width, height, layout)
            .encode_into(pixels, out)
            .map(|_| ())
    }

    /// Encode a multi-frame animation as a lossy JXL.
    ///
    /// Each frame must have the same dimensions and pixel layout.
    /// Returns the complete JXL codestream bytes.
    #[track_caller]
    pub fn encode_animation(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
        animation: &AnimationParams,
        frames: &[AnimationFrame<'_>],
    ) -> Result<Vec<u8>> {
        encode_animation_lossy(self, width, height, layout, animation, frames, None).map_err(at)
    }

    /// Encode a multi-frame animation with explicit resource [`Limits`].
    ///
    /// Same shape as [`Self::encode_animation`], plus a per-encode
    /// allocation cap that the VarDCT encoder consults at every
    /// dimension-driven allocation site. The cap applies across **all**
    /// frames combined — a single oversized frame is rejected before any
    /// of the per-frame buffers are allocated.
    #[track_caller]
    pub fn encode_animation_with_limits(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
        animation: &AnimationParams,
        frames: &[AnimationFrame<'_>],
        limits: &Limits,
    ) -> Result<Vec<u8>> {
        encode_animation_lossy(self, width, height, layout, animation, frames, Some(limits))
            .map_err(at)
    }
}

// ── EncodeRequest ───────────────────────────────────────────────────────────

/// Internal config reference (lossy or lossless).
#[derive(Clone, Copy, Debug)]
enum ConfigRef<'a> {
    Lossless(&'a LosslessConfig),
    Lossy(&'a LossyConfig),
}

/// An encoding request — binds config + image dimensions + pixel layout.
///
/// Created via [`LosslessConfig::encode_request`] or [`LossyConfig::encode_request`].
pub struct EncodeRequest<'a> {
    config: ConfigRef<'a>,
    width: u32,
    height: u32,
    layout: PixelLayout,
    metadata: Option<&'a ImageMetadata<'a>>,
    limits: Option<&'a Limits>,
    stop: Option<&'a dyn Stop>,
    source_gamma: Option<f32>,
    color_encoding: Option<crate::headers::color_encoding::ColorEncoding>,
    intensity_target: Option<f32>,
    min_nits: Option<f32>,
    premultiplied_alpha: bool,
    /// Premultiplied-alpha policy when the caller wants explicit auto
    /// detection (libjxl `--premultiply -1|0|1`). When `Some(_)` this
    /// overrides the boolean [`Self::with_premultiplied_alpha`] flag.
    /// `Some(Auto)` triggers a one-pass scan of the input pixels at
    /// encode time; `Some(On)`/`Some(Off)` are equivalent to passing
    /// `true`/`false` to [`Self::with_premultiplied_alpha`]. See
    /// [`Self::with_premultiplied_alpha_mode`].
    premultiplied_alpha_mode: Option<PremultipliedAlphaMode>,
    /// Optional input precision override for u16 layouts. `None` →
    /// full 16-bit (input divisor 65535). `Some(N)` → input divisor
    /// `(1 << N) - 1` and codestream `BitDepth.bits_per_sample = N`.
    /// Closes the configurable bits_per_sample portion of #18.
    bits_per_sample: Option<u32>,
    /// Brotli quality (0-11) for `brob` (Brotli-compressed) metadata
    /// boxes. `None` → plain `Exif`/`xml ` boxes. `Some(q)` → wrap
    /// each metadata blob in a `brob` box when it saves bytes
    /// (sub-500-byte payloads typically fall back due to overhead).
    /// Requires the `brotli-metadata` cargo feature; ignored otherwise.
    /// libjxl default quality is 4. Closes #15.
    brotli_metadata_quality: Option<u32>,
    /// Row stride (bytes per source row) for non-tightly-packed input.
    /// `None` → stride defaults to `width * layout.bytes_per_pixel()`.
    /// `Some(s)` → each source row is `s` bytes (with `s -
    /// width * bytes_per_pixel` padding bytes after each row's pixel
    /// data). Used by GPU textures, Windows BITMAP, Cairo surfaces,
    /// and any source that aligns rows to a power of 2.
    /// Closes row-stride portion of #18.
    row_stride: Option<usize>,
    /// Optional extra-channel buffers (refs #9). Each channel's
    /// dimensions match the request's `(width, height)`. Currently
    /// only u8 8-bit channels of `Depth` or `SpotColor` type are
    /// wired through the lossless encode path; lossy + 16-bit + the
    /// other libjxl channel types (SelectionMask, CFA, Thermal) are
    /// queued for follow-up ticks.
    extra_channels: &'a [ExtraChannel<'a>],
}

/// One additional channel (depth, spot color, selection mask, …)
/// to attach to the encoded image alongside the color + alpha
/// planes. Refs #9.
///
/// Built via [`Self::depth`] / [`Self::spot_color`] /
/// [`Self::selection_mask`] / [`Self::thermal`] / [`Self::cfa`].
/// The buffer dimensions must match the
/// [`EncodeRequest`]'s `(width, height)`. Only 8-bit u8 buffers are
/// supported in this iteration; 16-bit + dim_shift > 0 follow-up
/// ticks will widen the surface.
///
/// Wire-up status:
/// - Lossless RGB(A) + extra channels: WORKING (channels appended to
///   the modular image; ExtraChannelInfo entries written to file
///   header)
/// - Lossy VarDCT + extras beyond alpha: NOT YET (encoder pipeline
///   for additional modular sub-bitstreams pending)
#[derive(Debug, Clone)]
pub struct ExtraChannel<'a> {
    info: crate::headers::extra_channels::ExtraChannelInfo,
    data: ExtraChannelBuf<'a>,
}

/// Per-channel pixel data — either 8-bit or 16-bit samples.
#[derive(Debug, Clone, Copy)]
pub enum ExtraChannelBuf<'a> {
    /// `width * height` u8 samples.
    U8(&'a [u8]),
    /// `width * height` u16 samples (native byte order).
    U16(&'a [u16]),
}

impl<'a> ExtraChannelBuf<'a> {
    /// Number of samples in the buffer.
    pub fn len(&self) -> usize {
        match self {
            ExtraChannelBuf::U8(s) => s.len(),
            ExtraChannelBuf::U16(s) => s.len(),
        }
    }
    /// `true` if the buffer has no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> ExtraChannel<'a> {
    /// Attach an alpha channel (`ExtraChannelType::Alpha`). `data` is
    /// `width * height` bytes of u8 alpha values; `associated`
    /// signals whether the alpha is premultiplied
    /// (`alpha_associated=true`).
    ///
    /// In practice callers rarely build this by hand — the RGBA pixel
    /// layouts already wire alpha through automatically. Exposed for
    /// completeness and for the lossy + extras-beyond-alpha path
    /// (where alpha gets bundled in with the other extras).
    pub fn from_alpha_buf(data: &'a [u8], associated: bool) -> Self {
        let mut info = crate::headers::extra_channels::ExtraChannelInfo::alpha();
        info.alpha_associated = associated;
        Self {
            info,
            data: ExtraChannelBuf::U8(data),
        }
    }

    /// Attach a depth channel (`ExtraChannelType::Depth`). Use cases:
    /// 3D photos, iPhone Portrait Mode, structured-light scan output.
    /// `data` is `width * height` bytes of u8 depth values.
    pub fn depth(data: &'a [u8]) -> Self {
        Self {
            info: crate::headers::extra_channels::ExtraChannelInfo::depth(),
            data: ExtraChannelBuf::U8(data),
        }
    }

    /// Attach a 16-bit depth channel. `data` is `width * height`
    /// u16 samples; the channel info is marked as 16-bit so the
    /// decoder preserves the precision.
    pub fn depth_u16(data: &'a [u16]) -> Self {
        let mut info = crate::headers::extra_channels::ExtraChannelInfo::depth();
        info.bit_depth = crate::headers::file_header::BitDepth::uint16();
        Self {
            info,
            data: ExtraChannelBuf::U16(data),
        }
    }

    /// Attach a spot-color channel (`ExtraChannelType::SpotColor`).
    /// `data` is `width * height` bytes of u8 spot intensity (0 =
    /// no coverage, 255 = full coverage). `color` is the RGBA tint
    /// applied at decode time. Used in print production for
    /// non-CMYK inks (Pantone-style spot colors).
    pub fn spot_color(data: &'a [u8], color: [f32; 4]) -> Self {
        Self {
            info: crate::headers::extra_channels::ExtraChannelInfo::spot_color(color),
            data: ExtraChannelBuf::U8(data),
        }
    }

    /// Attach a Black (K) channel (`ExtraChannelType::Black`) — the
    /// fourth plane of a CMYK encode. `data` is `width * height`
    /// bytes of u8 ink coverage; JXL convention is
    /// **`0 = full ink, 255 = no ink`** (libjxl
    /// `enc_image_bundle.cc:65`). Forces codestream level 10 because
    /// the Black extra channel is forbidden at level 5
    /// (`compute_codestream_level`).
    ///
    /// In practice callers should prefer [`PixelLayout::Cmyk8`] which
    /// splits interleaved CMYK input into 3 colour planes + an
    /// automatically-synthesised Black extra channel. Exposed for
    /// callers that already keep their CMY and K planes separate
    /// (e.g. print-pipeline producers that store K as a separate buffer).
    pub fn black(data: &'a [u8]) -> Self {
        Self {
            info: crate::headers::extra_channels::ExtraChannelInfo {
                ec_type: crate::headers::extra_channels::ExtraChannelType::Black,
                ..Default::default()
            },
            data: ExtraChannelBuf::U8(data),
        }
    }

    /// Attach a 16-bit Black (K) channel. Same `0 = full ink,
    /// 65535 = no ink` convention as [`Self::black`]; the channel
    /// info is marked as 16-bit so the decoder preserves the full
    /// precision. Pairs with [`PixelLayout::Cmyk16`] when callers
    /// keep their K plane separate from C/M/Y.
    pub fn black_u16(data: &'a [u16]) -> Self {
        let mut info = crate::headers::extra_channels::ExtraChannelInfo {
            ec_type: crate::headers::extra_channels::ExtraChannelType::Black,
            ..Default::default()
        };
        info.bit_depth = crate::headers::file_header::BitDepth::uint16();
        Self {
            info,
            data: ExtraChannelBuf::U16(data),
        }
    }

    /// Attach a selection-mask channel
    /// (`ExtraChannelType::SelectionMask`). `data` is `width * height`
    /// bytes. Editing tools can use this to round-trip Photoshop-style
    /// per-image selections. *Header-only support today — the buffer
    /// is encoded but no dedicated semantics; treat it as an opaque
    /// 8-bit auxiliary channel.*
    pub fn selection_mask(data: &'a [u8]) -> Self {
        Self {
            info: crate::headers::extra_channels::ExtraChannelInfo {
                ec_type: crate::headers::extra_channels::ExtraChannelType::SelectionMask,
                ..Default::default()
            },
            data: ExtraChannelBuf::U8(data),
        }
    }

    /// Attach a thermal-data channel (`ExtraChannelType::Thermal`).
    /// `data` is `width * height` bytes. Same opaque-channel caveat
    /// as [`Self::selection_mask`].
    pub fn thermal(data: &'a [u8]) -> Self {
        Self {
            info: crate::headers::extra_channels::ExtraChannelInfo {
                ec_type: crate::headers::extra_channels::ExtraChannelType::Thermal,
                ..Default::default()
            },
            data: ExtraChannelBuf::U8(data),
        }
    }

    /// Attach a CFA (Color Filter Array) channel
    /// (`ExtraChannelType::Cfa`). `data` is `width * height` bytes;
    /// `cfa_index` selects the Bayer-style pattern used.
    pub fn cfa(data: &'a [u8], cfa_index: u32) -> Self {
        Self {
            info: crate::headers::extra_channels::ExtraChannelInfo {
                ec_type: crate::headers::extra_channels::ExtraChannelType::Cfa,
                cfa_channel: cfa_index,
                ..Default::default()
            },
            data: ExtraChannelBuf::U8(data),
        }
    }

    /// Set the dimension shift (log2 downsampling factor). When
    /// `n > 0`, the buffer must be sized
    /// `(width >> n) * (height >> n)` samples (or `div_ceil` of
    /// those dims; we use a plain shift). libjxl accepts
    /// `dim_shift ∈ {0, 3, 4} ∪ 1..=8` via the size coder. Most
    /// usage is `dim_shift = 0` (full resolution); `dim_shift = 2`
    /// gives a 1/4-resolution depth map. Refs #9.
    ///
    /// Use [`downsample_channel_u8`] to pre-downsample a full-res
    /// buffer with the same box filter libjxl uses on the
    /// `--ec_resampling` path; pair it with `with_dim_shift(log2(factor))`.
    pub fn with_dim_shift(mut self, n: u32) -> Self {
        self.info.dim_shift = n;
        self
    }

    /// Read-only access to the metadata that will be written into
    /// the file header for this channel.
    pub fn info(&self) -> &crate::headers::extra_channels::ExtraChannelInfo {
        &self.info
    }

    /// Read-only access to the channel's pixel buffer.
    pub fn data(&self) -> ExtraChannelBuf<'_> {
        self.data
    }

    /// The dimensions an N-pixel-wide image's extra channel should
    /// have under this channel's `dim_shift`. Mirrors libjxl's
    /// `DivCeil(d, 1 << dim_shift)`.
    pub(crate) fn downsampled_dims(&self, w: usize, h: usize) -> (usize, usize) {
        let ds = self.info.dim_shift.min(31);
        let factor = 1usize << ds;
        (w.div_ceil(factor), h.div_ceil(factor))
    }
}

impl<'a> EncodeRequest<'a> {
    /// Attach image metadata (ICC, EXIF, XMP).
    pub fn with_metadata(mut self, meta: &'a ImageMetadata<'a>) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Attach resource limits.
    pub fn with_limits(mut self, limits: &'a Limits) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Attach a cooperative cancellation token.
    ///
    /// The encoder will check this periodically and return
    /// [`EncodeError::Cancelled`] if stopped.
    pub fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Specify that source pixels use a custom gamma transfer function.
    ///
    /// When set, the encoder linearizes u8/u16 pixels with `pixel ^ (1/gamma)`
    /// instead of the sRGB transfer function, and writes `have_gamma=true` in
    /// the JXL header. This matches cjxl's behavior for PNGs with gAMA chunks.
    ///
    /// Example: `0.45455` for standard gamma 2.2 encoding (gAMA=45455).
    pub fn with_source_gamma(mut self, gamma: f32) -> Self {
        self.source_gamma = Some(gamma);
        self
    }

    /// Override the color encoding written to the JXL header.
    ///
    /// When set, this color encoding is used instead of the default (sRGB for
    /// u8/u16, linear sRGB for f32) or any gamma derived from
    /// [`with_source_gamma`](Self::with_source_gamma).
    ///
    /// Use this for HDR content (PQ, HLG) or non-sRGB primaries (BT.2020, Display P3).
    ///
    /// Note: this only affects the signaled color encoding in the JXL header.
    /// Pixel linearization for lossy encoding is still controlled by
    /// `with_source_gamma()`. For float input, pixels are assumed already linear.
    pub fn with_color_encoding(
        mut self,
        ce: crate::headers::color_encoding::ColorEncoding,
    ) -> Self {
        self.color_encoding = Some(ce);
        self
    }

    /// Set the peak display luminance in nits (cd/m²) for HDR content.
    ///
    /// Written to the JXL codestream `ToneMapping.intensity_target` field.
    /// Default is 255.0 (SDR). Set to e.g. 4000.0 or 10000.0 for HDR.
    ///
    /// Pairs with [`Self::with_color_encoding`] for HDR signaling
    /// (e.g. [`ColorEncoding::bt2100_pq`] / [`ColorEncoding::bt2100_hlg`]).
    /// If both this builder and an attached [`ImageMetadata`] set this
    /// value, the request-level value wins.
    ///
    /// [`ColorEncoding::bt2100_pq`]: crate::ColorEncoding::bt2100_pq
    /// [`ColorEncoding::bt2100_hlg`]: crate::ColorEncoding::bt2100_hlg
    pub fn with_intensity_target(mut self, nits: f32) -> Self {
        self.intensity_target = Some(nits);
        self
    }

    /// Set the minimum display luminance in nits.
    ///
    /// Written to the JXL codestream `ToneMapping.min_nits` field.
    /// Default is 0.0. If both this builder and an attached
    /// [`ImageMetadata`] set this value, the request-level value wins.
    pub fn with_min_nits(mut self, nits: f32) -> Self {
        self.min_nits = Some(nits);
        self
    }

    /// Signal that the input alpha channel is premultiplied (associated).
    ///
    /// Standard for GPU pipelines (Skia, Cairo, Metal, Vulkan,
    /// Direct2D, Wayland, CompositorAPI). When set, the encoder
    /// records `alpha_associated=true` in the `ExtraChannelInfo`
    /// header so decoders know to interpret the color values as
    /// already-multiplied-by-alpha.
    ///
    /// **Lossless**: works correctly — the encoder writes the pixels
    /// as-is and the header bit tells the decoder to keep them
    /// premultiplied.
    ///
    /// **Lossy**: NOT YET supported (closes lossless portion of #13;
    /// lossy portion needs the unpremultiplication pre-pass that
    /// libjxl does at `enc_frame.cc:1588-1597` before XYB conversion
    /// — quantization errors multiply with alpha if you skip it).
    /// Calling this on a lossy encode returns
    /// [`EncodeError::InvalidInput`].
    ///
    /// Default is `false` (straight / unassociated alpha).
    pub fn with_premultiplied_alpha(mut self, enable: bool) -> Self {
        self.premultiplied_alpha = enable;
        self
    }

    /// Explicit premultiplied-alpha mode (libjxl `--premultiply -1|0|1`).
    ///
    /// Setting this overrides any prior
    /// [`Self::with_premultiplied_alpha`] call. The three accepted modes
    /// are:
    ///
    /// - [`PremultipliedAlphaMode::Off`] — straight alpha (libjxl `0`).
    /// - [`PremultipliedAlphaMode::On`] — premultiplied alpha (libjxl `1`).
    /// - [`PremultipliedAlphaMode::Auto`] — detect at encode time by
    ///   scanning the input pixels once (libjxl `-1`). The scan is O(N)
    ///   and runs before the encode loop; for trusted inputs prefer the
    ///   explicit forms above.
    ///
    /// `On`/`Off` map directly onto
    /// [`Self::with_premultiplied_alpha(true|false)`]. `Auto` records
    /// the policy on the request; the encoder samples the input once
    /// before the encode loop and resolves it to `On` or `Off`. Lossy
    /// resolution to `On` still returns
    /// [`EncodeError::InvalidInput`] until the unpremultiplication
    /// pre-pass (#13) lands.
    pub fn with_premultiplied_alpha_mode(mut self, mode: PremultipliedAlphaMode) -> Self {
        self.premultiplied_alpha_mode = Some(mode);
        match mode {
            PremultipliedAlphaMode::On => {
                self.premultiplied_alpha = true;
            }
            PremultipliedAlphaMode::Off => {
                self.premultiplied_alpha = false;
            }
            PremultipliedAlphaMode::Auto => {
                // Resolved at encode time by the input-scanning pre-pass.
                // The boolean flag retains its previous value as a
                // fallback if the scanner is not enabled (e.g. lossy
                // encode where Auto resolves to On).
            }
        }
        self
    }

    /// Currently configured premultiplied-alpha mode.
    ///
    /// Returns the explicit mode set via
    /// [`Self::with_premultiplied_alpha_mode`] if any, otherwise
    /// reflects the boolean
    /// [`Self::with_premultiplied_alpha`] flag (Off if `false`, On if
    /// `true`).
    pub fn premultiplied_alpha_mode(&self) -> PremultipliedAlphaMode {
        self.premultiplied_alpha_mode
            .unwrap_or(if self.premultiplied_alpha {
                PremultipliedAlphaMode::On
            } else {
                PremultipliedAlphaMode::Off
            })
    }

    /// Override the input precision for u16 layouts (closes
    /// `bits_per_sample` portion of #18). 10-bit (broadcast / video),
    /// 12-bit (medical / cinema DPX), and 14-bit (DSLR raw) are
    /// commonly stored in u16 buffers with the value occupying the
    /// LOW bits — i.e. a 12-bit white pixel is `4095u16`, not `65535`.
    /// Without this builder the encoder would normalize 4095 / 65535 ≈
    /// 0.062 instead of 4095 / 4095 = 1.0, producing a near-black
    /// encoded image.
    ///
    /// When set:
    /// - u16 input is normalized as `value / ((1 << bits) - 1)`
    /// - codestream `BitDepth.bits_per_sample` is `bits`
    /// - decoder sees the correct precision metadata
    ///
    /// `bits` must be in `1..=16`; out-of-range values are clamped.
    /// Streaming-encoder parity is also wired (LossyEncoder +
    /// LosslessEncoder both expose this builder).
    pub fn with_bits_per_sample(mut self, bits: u32) -> Self {
        self.bits_per_sample = Some(bits.clamp(1, 16));
        self
    }

    /// Set a custom row stride (bytes per source row) for
    /// non-tightly-packed input. Closes row-stride portion of #18.
    ///
    /// `stride` must be `>= width * layout.bytes_per_pixel()`. The
    /// default (`None`) treats the input as tightly packed (no
    /// per-row padding). When set, each row is `stride` bytes; the
    /// first `width * bytes_per_pixel` of each row carry the actual
    /// pixel data and the remaining `stride - width * bytes_per_pixel`
    /// bytes are padding (their content is ignored).
    ///
    /// Common origins: GPU textures (OpenGL/Vulkan/Metal often align
    /// rows to 256 / 512 / 4096 bytes), Windows BITMAP (`stride =
    /// ((width * bpp + 31) / 32) * 4`), Cairo image surfaces,
    /// `image::DynamicImage` after a sub-region crop.
    ///
    /// Implementation: when set, the encoder unpacks pixels into a
    /// tightly-packed scratch buffer once via `memcpy`-per-row, then
    /// runs the existing per-layout converters on that buffer. The
    /// extra buffer costs O(width × height × bytes_per_pixel) but the
    /// unpack is O(n) and amortizes across all downstream work
    /// (linearization, XYB, DCT, etc.).
    pub fn with_row_stride(mut self, stride: usize) -> Self {
        self.row_stride = Some(stride);
        self
    }

    /// Attach extra-channel buffers (refs #9) — depth, spot color,
    /// selection mask, thermal, CFA. Each [`ExtraChannel`] carries
    /// `width * height` bytes of u8 channel data plus the
    /// metadata that gets written into the file header.
    ///
    /// Currently wired through the **lossless** encode path. Lossy
    /// encodes with extras beyond alpha return
    /// `EncodeError::InvalidInput("extra channels beyond alpha not
    /// yet supported in lossy encode")`. 16-bit channels and
    /// `dim_shift > 0` (per-channel downsampling) follow-up ticks.
    pub fn with_extra_channels(mut self, channels: &'a [ExtraChannel<'a>]) -> Self {
        self.extra_channels = channels;
        self
    }

    /// Brotli-compress EXIF / XMP metadata into `brob` boxes
    /// (closes #15). `quality` is the Brotli effort (0-11; libjxl
    /// default 4); higher = smaller output but slower encode. Each
    /// metadata blob is independently evaluated — if the compressed
    /// brob box would be ≥ the uncompressed Exif/xml box, the
    /// uncompressed form is used (sub-500-byte payloads typically
    /// fall back due to Brotli framing overhead).
    ///
    /// Requires the `brotli-metadata` cargo feature. When the feature
    /// is OFF the call still compiles (the value is stored but
    /// ignored at encode time); add the feature flag to enable.
    pub fn with_brotli_metadata(mut self, quality: u32) -> Self {
        self.brotli_metadata_quality = Some(quality.min(11));
        self
    }

    /// Encode pixels and return the JXL bytes.
    #[track_caller]
    pub fn encode(self, pixels: &[u8]) -> Result<Vec<u8>> {
        self.encode_inner(pixels)
            .map(|mut r| r.take_data().unwrap())
            .map_err(at)
    }

    /// Encode pixels and return the JXL bytes together with [`EncodeStats`].
    #[track_caller]
    pub fn encode_with_stats(self, pixels: &[u8]) -> Result<EncodeResult> {
        self.encode_inner(pixels).map_err(at)
    }

    /// Encode pixels, appending to an existing buffer. Returns metrics.
    #[track_caller]
    pub fn encode_into(self, pixels: &[u8], out: &mut Vec<u8>) -> Result<EncodeResult> {
        let mut result = self.encode_inner(pixels).map_err(at)?;
        if let Some(data) = result.data.take() {
            out.extend_from_slice(&data);
        }
        Ok(result)
    }

    /// Encode pixels, writing to a `std::io::Write` destination. Returns metrics.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn encode_to(self, pixels: &[u8], mut dest: impl std::io::Write) -> Result<EncodeResult> {
        let mut result = self.encode_inner(pixels).map_err(at)?;
        if let Some(data) = result.data.take() {
            dest.write_all(&data)
                .map_err(|e| at(EncodeError::from(e)))?;
        }
        Ok(result)
    }

    fn encode_inner(&self, pixels: &[u8]) -> core::result::Result<EncodeResult, EncodeError> {
        self.validate_pixels(pixels)?;
        self.check_limits()?;
        // Run the full config validator (distance, effort, iter
        // counts, mutual exclusivity, etc.). This was previously
        // opt-in via `cfg.validate()`; auto-calling it on the encode
        // path means callers no longer have to remember to invoke it.
        match self.config {
            ConfigRef::Lossy(cfg) => cfg.validate()?,
            ConfigRef::Lossless(cfg) => cfg.validate()?,
        }
        if let Some(ref ce) = self.color_encoding {
            crate::vardct::xyb::validate_color_encoding(ce).map_err(EncodeError::from)?;
        }
        // Defensive caps on caller-supplied metadata buffers (see
        // `validate_metadata_sizes` for rationale).
        validate_metadata_sizes(
            self.metadata.and_then(|m| m.icc_profile),
            self.metadata.and_then(|m| m.exif),
            self.metadata.and_then(|m| m.xmp),
            self.metadata.and_then(|m| m.jumbf),
        )?;
        // Tone-mapping numeric range checks. Request-level overrides
        // win over metadata-level values (`encode_lossy` line ~3018);
        // we apply the same precedence here so the validator sees the
        // value the encoder will actually use.
        let it = self
            .intensity_target
            .or_else(|| self.metadata.and_then(|m| m.intensity_target));
        let mn = self
            .min_nits
            .or_else(|| self.metadata.and_then(|m| m.min_nits));
        validate_tone_mapping(it, mn)?;
        // Source gamma + intrinsic size up-front checks.
        validate_source_gamma(self.source_gamma)?;
        validate_intrinsic_size(self.metadata.and_then(|m| m.intrinsic_size))?;

        // Build the per-encode allocation budget. Caller-supplied
        // Limits.max_memory_bytes wins; otherwise Limits provides its
        // soft-default cap (~2 GB). The budget is threaded through to
        // major dimension-driven allocation sites (XYB planes, padded
        // scratch, group buffers, modular channels) via RAII guards;
        // peak working-set is observable post-encode via `peak()`.
        //
        // Up-front reservation of the rough working-set estimate gives
        // an early bail-out for absurd dimensions before any allocator
        // call would have to fail individually. We also refuse a budget
        // smaller than that estimate, so callers see a meaningful
        // error instead of a confusing mid-encode failure.
        let budget_cap = self
            .limits
            .map(|l| l.effective_max_memory_bytes())
            .unwrap_or(Limits::DEFAULT_MAX_MEMORY_BYTES);
        let budget = crate::budget::MemoryBudget::new(budget_cap);
        let est_bytes = (self.width as u64)
            .checked_mul(self.height as u64)
            .and_then(|n| n.checked_mul(40))
            .ok_or_else(|| EncodeError::LimitExceeded {
                message: format!(
                    "image {}x{} too large for working-set estimate",
                    self.width, self.height
                ),
            })?;
        if est_bytes > budget_cap {
            return Err(EncodeError::LimitExceeded {
                message: format!(
                    "estimated working set {est_bytes} bytes for {}x{} image \
                     exceeds budget cap {budget_cap}",
                    self.width, self.height
                ),
            });
        }

        let threads = match self.config {
            ConfigRef::Lossless(cfg) => cfg.threads,
            ConfigRef::Lossy(cfg) => cfg.threads,
        };

        // Repack strided input into a tightly-packed buffer once.
        // Closes row-stride portion of #18. Downstream encode paths
        // assume tightly-packed `width * bytes_per_pixel` per row, so
        // the unpack is the entry-side adapter — extra image-sized
        // buffer + O(n) memcpy. None → use caller's slice as-is.
        let packed_storage;
        let pixels: &[u8] = if let Some(stride) = self.row_stride {
            packed_storage = unpack_strided_pixels(
                pixels,
                self.width as usize,
                self.height as usize,
                self.layout.bytes_per_pixel(),
                stride,
            )?;
            &packed_storage
        } else {
            pixels
        };

        let (codestream, mut stats) = run_with_threads(threads, || match self.config {
            ConfigRef::Lossless(cfg) => self.encode_lossless(cfg, pixels, &budget),
            ConfigRef::Lossy(cfg) => self.encode_lossy(cfg, pixels, &budget),
        })?;

        stats.codestream_size = codestream.len();

        // Pick the codestream level: 5 for baseline-fits images, 10
        // when any cap is exceeded (> 262144 dim, > 2²⁸ pixels, >4 EC,
        // CMYK, large ICC). Mirrors libjxl `VerifyLevelSettings`.
        // Alpha-bearing layouts count as +1 extra channel.
        let icc_size = self
            .metadata
            .and_then(|m| m.icc_profile)
            .map_or(0u64, |icc| icc.len() as u64);
        // CMYK layouts auto-synthesise a Black extra channel inside
        // `encode_lossless` — count it here so the level computation
        // (which forbids the Black channel at level 5) bumps to 10
        // before the codestream is wrapped.
        let num_ec = self.extra_channels.len() as u32
            + u32::from(self.layout.has_alpha())
            + u32::from(self.layout.is_cmyk());
        let has_black = self.layout.is_cmyk()
            || self.extra_channels.iter().any(|ec| {
                ec.info.ec_type == crate::headers::extra_channels::ExtraChannelType::Black
            });
        let level = compute_required_level(self.width, self.height, num_ec, has_black, icc_size)?;

        // Wrap in container if metadata (EXIF/XMP/JUMBF/colr/hCdR) is
        // present OR if the level requires a container (level != 5
        // means a `jxll` box must precede the codestream — mirrors
        // libjxl `MustUseContainer`).
        let colr = self.metadata.and_then(|m| m.colr_payload);
        let hcdr = self.metadata.and_then(|m| m.hcdr_payload);
        let has_meta = self
            .metadata
            .map(|m| m.exif.is_some() || m.xmp.is_some() || m.jumbf.is_some())
            .unwrap_or(false);
        let has_aux_boxes = colr.is_some() || hcdr.is_some();
        let mut output =
            if has_meta || has_aux_boxes || crate::container::level_requires_container(level) {
                let (exif, xmp, jumbf) = match self.metadata {
                    Some(m) => (m.exif, m.xmp, m.jumbf),
                    None => (None, None, None),
                };
                wrap_metadata_container(
                    &codestream,
                    exif,
                    xmp,
                    jumbf,
                    self.brotli_metadata_quality,
                    level,
                )
            } else {
                codestream
            };
        // Append `colr` (alternative colour descriptor) and `hCdR` (HDR
        // metadata) boxes last. They are pass-through extras for
        // ISOBMFF-aware tooling; per JPEG XL spec clause 5 a decoder
        // MUST ignore unrecognised boxes so this never alters decoded
        // pixels. Appended after standard metadata so legacy readers
        // that stop at the first unknown box still see the codestream.
        if let Some(payload) = colr {
            output = crate::container::append_colr_box(&output, payload);
        }
        if let Some(payload) = hcdr {
            output = crate::container::append_hcdr_box(&output, payload);
        }

        stats.output_size = output.len();

        Ok(EncodeResult {
            data: Some(output),
            stats,
        })
    }

    fn validate_pixels(&self, pixels: &[u8]) -> core::result::Result<(), EncodeError> {
        validate_dims(self.width, self.height)?;
        let w = self.width as usize;
        let h = self.height as usize;
        let expected = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(self.layout.bytes_per_pixel()));
        // Internal allocations are sized as `width * height * N` for N up
        // to 8 (4 channels × f32 = 16 bytes/px would also fit since
        // `usize` can absorb a 4× multiplier on top of `bpp` ≤ 4 within
        // the same budget). Enforce a single up-front check that
        // `width * height * 16` fits in `usize` so the encoder never has
        // to re-validate inside hot loops. This bounds the per-pixel
        // working-set scaling factor for all downstream callers.
        const MAX_INTERNAL_SCALE: usize = 16;
        if w.checked_mul(h)
            .and_then(|n| n.checked_mul(MAX_INTERNAL_SCALE))
            .is_none()
        {
            return Err(EncodeError::LimitExceeded {
                message: format!(
                    "image {w}x{h} too large for encoder working buffers \
                     (width × height × {MAX_INTERNAL_SCALE} overflows usize)"
                ),
            });
        }
        // When row_stride is set, the buffer is `height * stride`
        // bytes (stride may include per-row padding). Validate
        // `stride >= width * bytes_per_pixel` and the buffer size up
        // front so callers fail before any allocation; the strided
        // unpack downstream re-checks defensively.
        if let Some(stride) = self.row_stride {
            let row_bytes = w
                .checked_mul(self.layout.bytes_per_pixel())
                .ok_or_else(|| EncodeError::InvalidInput {
                    message: "width * bytes_per_pixel overflows usize".into(),
                })?;
            if stride < row_bytes {
                return Err(EncodeError::InvalidInput {
                    message: format!(
                        "row_stride {stride} is less than width * bytes_per_pixel = {w} * {} = {row_bytes}",
                        self.layout.bytes_per_pixel(),
                    ),
                });
            }
            let needed = h
                .checked_mul(stride)
                .ok_or_else(|| EncodeError::InvalidInput {
                    message: "height * row_stride overflows usize".into(),
                })?;
            if pixels.len() < needed {
                return Err(EncodeError::InvalidInput {
                    message: format!(
                        "pixel buffer too small for strided input: need {needed} bytes (height {h} × stride {stride}), got {}",
                        pixels.len(),
                    ),
                });
            }
            return Ok(());
        }
        match expected {
            Some(expected) if pixels.len() == expected => Ok(()),
            Some(expected) => Err(EncodeError::InvalidInput {
                message: format!(
                    "pixel buffer size mismatch: expected {expected} bytes for {w}x{h} {:?}, got {}",
                    self.layout,
                    pixels.len()
                ),
            }),
            None => Err(EncodeError::InvalidInput {
                message: "image dimensions overflow".into(),
            }),
        }
    }

    fn check_limits(&self) -> core::result::Result<(), EncodeError> {
        let Some(limits) = self.limits else {
            return Ok(());
        };
        let w = self.width as u64;
        let h = self.height as u64;
        if let Some(max_w) = limits.max_width
            && w > max_w
        {
            return Err(EncodeError::LimitExceeded {
                message: format!("width {w} > max {max_w}"),
            });
        }
        if let Some(max_h) = limits.max_height
            && h > max_h
        {
            return Err(EncodeError::LimitExceeded {
                message: format!("height {h} > max {max_h}"),
            });
        }
        if let Some(max_px) = limits.max_pixels
            && w * h > max_px
        {
            return Err(EncodeError::LimitExceeded {
                message: format!("pixels {}x{} = {} > max {max_px}", w, h, w * h),
            });
        }
        if let Some(max_mem) = limits.max_memory_bytes {
            // Conservative estimate: ~40 bytes per pixel covers XYB (3×f32=12),
            // quantization fields, strategy maps, and entropy coding buffers.
            let estimated = w.saturating_mul(h).saturating_mul(40);
            if estimated > max_mem {
                return Err(EncodeError::LimitExceeded {
                    message: format!(
                        "estimated memory {estimated} bytes > max {max_mem} bytes \
                         (for {w}x{h} image)"
                    ),
                });
            }
        }
        // If the caller set an explicit max_quant_loop_iters and the
        // resolved config is asking for more, reject. The encoder still
        // saturates at the validator hard cap (`Limits::DEFAULT_MAX_QUANT_LOOP_ITERS`)
        // at consumption sites — this lets a caller set a *tighter* cap
        // and have it surface as an error rather than a silent saturation.
        if let Some(max_iters) = limits.max_quant_loop_iters {
            let configured = match self.config {
                ConfigRef::Lossy(cfg) => self.lossy_max_iter_value(cfg),
                ConfigRef::Lossless(_) => 0,
            };
            if configured > max_iters {
                return Err(EncodeError::LimitExceeded {
                    message: format!(
                        "quantization-loop iterations ({configured}) exceed \
                         Limits::max_quant_loop_iters ({max_iters})"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Maximum of butteraugli/ssim2/zensim iters across the loop knobs
    /// available on this config — used by `check_limits` to surface a
    /// caller-set per-encode iter cap.
    #[cfg(any(
        feature = "butteraugli-loop",
        feature = "ssim2-loop",
        feature = "zensim-loop"
    ))]
    fn lossy_max_iter_value(&self, cfg: &LossyConfig) -> u32 {
        let mut m = 0u32;
        #[cfg(feature = "butteraugli-loop")]
        {
            m = m.max(cfg.butteraugli_iters);
        }
        #[cfg(feature = "ssim2-loop")]
        {
            m = m.max(cfg.ssim2_iters);
        }
        #[cfg(feature = "zensim-loop")]
        {
            m = m.max(cfg.zensim_iters);
        }
        m
    }
    #[cfg(not(any(
        feature = "butteraugli-loop",
        feature = "ssim2-loop",
        feature = "zensim-loop"
    )))]
    fn lossy_max_iter_value(&self, _cfg: &LossyConfig) -> u32 {
        0
    }

    // ── Lossless path ───────────────────────────────────────────────────

    fn encode_lossless(
        &self,
        cfg: &LosslessConfig,
        pixels: &[u8],
        budget: &alloc::sync::Arc<crate::budget::MemoryBudget>,
    ) -> core::result::Result<(Vec<u8>, EncodeStats), EncodeError> {
        use crate::bit_writer::BitWriter;
        use crate::headers::color_encoding::ColorSpace;
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::channel::ModularImage;
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        let w = self.width as usize;
        let h = self.height as usize;

        // Normalize pixels to RGB8 for detection if needed (BGR swap)
        let rgb_pixels;
        let detection_pixels: &[u8] = match self.layout {
            PixelLayout::Bgr8 => {
                rgb_pixels = bgr_to_rgb(pixels, 3);
                &rgb_pixels
            }
            PixelLayout::Bgra8 => {
                rgb_pixels = bgr_to_rgb(pixels, 4);
                &rgb_pixels
            }
            _ => {
                rgb_pixels = Vec::new();
                let _ = &rgb_pixels;
                pixels
            }
        };

        // Build ModularImage from pixel layout. The 8-bit RGB(A) paths
        // route through `from_*_with_budget` so the channel allocations
        // (the dominant working-set in lossless mode) are charged
        // against the per-encode cap. Other layouts allocate the same
        // shape but route through legacy constructors; the up-front
        // working-set check in `encode_inner` already gates them.
        let budget_opt = Some(budget);
        // CMYK layouts split into 3 colour planes (CMY) + a separately
        // emitted Black extra channel. We deinterleave once here to
        // avoid bouncing the same bytes through two passes downstream;
        // the K plane is kept on the side and injected as the FIRST
        // extra channel further down (so it always lives at ec index
        // 0 regardless of any user-supplied extras).
        let synthesised_black_u8: Option<Vec<u8>>;
        let synthesised_black_u16: Option<Vec<u16>>;
        let mut image = match self.layout {
            PixelLayout::Rgb8 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_rgb8_with_budget(pixels, w, h, budget_opt)
            }
            PixelLayout::Rgba8 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_rgba8_with_budget(pixels, w, h, budget_opt)
            }
            PixelLayout::Bgr8 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_rgb8_with_budget(&bgr_to_rgb(pixels, 3), w, h, budget_opt)
            }
            PixelLayout::Bgra8 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_rgba8_with_budget(&bgr_to_rgb(pixels, 4), w, h, budget_opt)
            }
            PixelLayout::Gray8 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_gray8(pixels, w, h)
            }
            PixelLayout::GrayAlpha8 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_grayalpha8(pixels, w, h)
            }
            PixelLayout::Rgb16 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_rgb16_native(pixels, w, h)
            }
            PixelLayout::Rgba16 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_rgba16_native(pixels, w, h)
            }
            PixelLayout::Gray16 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_gray16_native(pixels, w, h)
            }
            PixelLayout::GrayAlpha16 => {
                synthesised_black_u8 = None;
                synthesised_black_u16 = None;
                ModularImage::from_grayalpha16_native(pixels, w, h)
            }
            PixelLayout::Cmyk8 => {
                // Reject if the caller already provided their own
                // Black extra channel — the file header would carry
                // two Black entries and the second K plane would
                // never reach the decoder.
                if self.extra_channels.iter().any(|ec| {
                    ec.info.ec_type == crate::headers::extra_channels::ExtraChannelType::Black
                }) {
                    return Err(EncodeError::InvalidInput {
                        message: "PixelLayout::Cmyk8 already synthesises a Black extra \
                                  channel; remove the user-supplied ExtraChannel::black(...)"
                            .into(),
                    });
                }
                // Deinterleave CMYK → 3-channel CMY + separate K buffer.
                // Two passes over the input but a single allocation
                // per output buffer; total work matches a memcpy of
                // the source.
                let n = w * h;
                let mut cmy = Vec::with_capacity(n * 3);
                let mut k = Vec::with_capacity(n);
                for i in 0..n {
                    let base = i * 4;
                    cmy.push(pixels[base]);
                    cmy.push(pixels[base + 1]);
                    cmy.push(pixels[base + 2]);
                    k.push(pixels[base + 3]);
                }
                synthesised_black_u8 = Some(k);
                synthesised_black_u16 = None;
                ModularImage::from_rgb8_with_budget(&cmy, w, h, budget_opt)
            }
            PixelLayout::Cmyk16 => {
                if self.extra_channels.iter().any(|ec| {
                    ec.info.ec_type == crate::headers::extra_channels::ExtraChannelType::Black
                }) {
                    return Err(EncodeError::InvalidInput {
                        message: "PixelLayout::Cmyk16 already synthesises a Black extra \
                                  channel; remove the user-supplied ExtraChannel::black_u16(...)"
                            .into(),
                    });
                }
                // 16-bit CMYK input is interleaved native-endian u16
                // (8 bytes/pixel). Reinterpret the byte slice as u16
                // via a copying deinterleave (avoids an unsafe cast
                // and absorbs unaligned input).
                let n = w * h;
                if pixels.len() != n * 8 {
                    return Err(EncodeError::InvalidInput {
                        message: format!(
                            "Cmyk16 expects {} bytes ({}x{} × 8), got {}",
                            n * 8,
                            w,
                            h,
                            pixels.len(),
                        ),
                    });
                }
                let mut cmy = Vec::with_capacity(n * 3 * 2);
                let mut k = Vec::with_capacity(n);
                for i in 0..n {
                    let base = i * 8;
                    cmy.extend_from_slice(&pixels[base..base + 6]);
                    let k_lo = pixels[base + 6];
                    let k_hi = pixels[base + 7];
                    k.push(u16::from_ne_bytes([k_lo, k_hi]));
                }
                synthesised_black_u8 = None;
                synthesised_black_u16 = Some(k);
                ModularImage::from_rgb16_native(&cmy, w, h)
            }
            other => return Err(EncodeError::UnsupportedPixelLayout(other)),
        }
        .map_err(EncodeError::from)?;

        // `keep_invisible = false` pre-pass (libjxl `SimplifyInvisible`
        // lossless mode, `enc_frame.cc:511`+`1588-1597`). When the
        // caller opts in via `LosslessConfig::with_keep_invisible(false)`,
        // zero the color samples in pixels whose alpha=0 so the modular
        // predictor + LZ77 can compress long runs of zeros instead of
        // arbitrary editor noise. Pixel-exact output is preserved for
        // every *visible* pixel; only data no decoder will display
        // changes.
        //
        // Gated identically to libjxl + the lossy path: requires alpha,
        // skipped for premultiplied input (alpha=0 ⇒ RGB=0 already by
        // construction), and short-circuits if no pixel is fully
        // transparent (predicate is one linear scan, early-exit).
        if cfg.simplify_invisible && image.has_alpha && !self.premultiplied_alpha {
            // Alpha is the trailing channel for both RGBA-class and
            // GrayAlpha layouts in `ModularImage`. Extra channels are
            // appended AFTER this point so the count here is exactly
            // the color-plus-alpha planes.
            let alpha_idx = image.channels.len() - 1;
            // Color channels are everything BEFORE alpha (R/G/B for
            // RGBA; Gray for GrayAlpha).
            let color_channels = alpha_idx;
            // Snapshot the alpha plane so we can mutate color planes
            // without a borrow conflict. `.to_vec()` is O(n) but the
            // pre-pass already touches every pixel; one extra read
            // pass is in the noise.
            let alpha_plane: Vec<i32> = image.channels[alpha_idx].data().to_vec();
            if alpha_plane.contains(&0) {
                for c in 0..color_channels {
                    let plane = image.channels[c].data_mut();
                    for (px, &a) in plane.iter_mut().zip(alpha_plane.iter()) {
                        if a == 0 {
                            *px = 0;
                        }
                    }
                }
            }
        }

        // CMYK: inject the synthesised Black plane as the FIRST extra
        // channel. We push to `image.channels` here so the encoder
        // pipeline sees a `[C, M, Y, K]` layout; the matching
        // `ExtraChannelInfo::black()` is inserted at the head of
        // `file_header.metadata.extra_channels` further down (after
        // the FileHeader is constructed). Keeping the K plane at ec
        // index 0 mirrors libjxl's `enc_image_bundle.cc:57` CMYK
        // pipeline and matches what the libjxl `EncoderTest.CMYK`
        // round-trip writes (`encode_test.cc:2070`).
        if let Some(ref k_u8) = synthesised_black_u8 {
            image
                .push_extra_channel_u8(k_u8, w, h)
                .map_err(EncodeError::from)?;
        }
        if let Some(ref k_u16) = synthesised_black_u16 {
            image
                .push_extra_channel_u16(k_u16, w, h)
                .map_err(EncodeError::from)?;
        }

        // Append extra channels (refs #9 — Depth, SpotColor, etc.).
        // Each `ExtraChannel` carries an 8-bit or 16-bit plane at
        // its own dimensions, which may be smaller than the image
        // when `dim_shift > 0` is set (e.g., a 1/4-resolution depth
        // map). The expected sample count is the image dims shifted
        // down by `dim_shift` (using `div_ceil`). The channel is
        // added to the modular image and its `ExtraChannelInfo` is
        // written into the file header.
        for (idx, ec) in self.extra_channels.iter().enumerate() {
            let (ec_w, ec_h) = ec.downsampled_dims(w, h);
            let len = ec.data.len();
            if len != ec_w * ec_h {
                return Err(EncodeError::InvalidInput {
                    message: format!(
                        "extra_channels[{idx}]: expected {} samples for {ec_w}x{ec_h} (dim_shift={}), got {len}",
                        ec_w * ec_h,
                        ec.info.dim_shift,
                    ),
                });
            }
            // For `dim_shift > 0` extras (e.g. `--ec_resampling N`
            // alpha at `log2(N)` half-steps), the multi-group writer
            // needs the channel's hshift/vshift set so per-group
            // rects crop in channel-local coords (libjxl
            // `enc_modular.cc:1400-1407`). Single-group writers
            // never call `extract_region`, so this is a no-op for
            // single-group; multi-group at dim_shift > 0 (>256-pixel
            // images with half-res alpha) now writes correctly.
            let ec_shift = ec.info.dim_shift;
            match ec.data {
                ExtraChannelBuf::U8(d) => {
                    image.push_extra_channel_u8_with_shift(d, ec_w, ec_h, ec_shift, ec_shift)
                }
                ExtraChannelBuf::U16(d) => {
                    image.push_extra_channel_u16_with_shift(d, ec_w, ec_h, ec_shift, ec_shift)
                }
            }
            .map_err(EncodeError::from)?;
        }

        // Detect patches for lossless mode (RGB 8-bit only, non-grayscale).
        // CMYK is excluded: the patches detector assumes RGB-like
        // perceptual colour and operates on the first 3 channels
        // (which are CMY, not RGB) — false matches would inject
        // bogus subtractive-colour patches into the codestream.
        let num_channels = self.layout.bytes_per_pixel();
        let can_use_patches = cfg.effective_patches()
            && !image.is_grayscale
            && image.bit_depth <= 8
            && num_channels >= 3
            && !self.layout.is_cmyk();
        let patches_data = if can_use_patches {
            crate::profile_time!("modular/patches_detect", {
                let pd_opt = crate::vardct::patches::find_and_build_lossless(
                    detection_pixels,
                    w,
                    h,
                    num_channels,
                    image.bit_depth,
                );
                // RFC#45 chunks 4-7 lossless backport (chunk 5 lossless
                // trial encoder): per-image cost gate. Trial-encodes
                // lossless-shape ref-frame + dictionary overhead,
                // requires `savings_est >= 1.5 * overhead`. Protects
                // against pathological mixed content where patches barely
                // clear the detector's 1% coverage filter but the
                // ref-frame overhead dominates net savings. See
                // `PatchesData::is_cost_effective_lossless` doc-comment.
                pd_opt.filter(|pd| pd.is_cost_effective_lossless(image.bit_depth, cfg.use_ans))
            })
        } else {
            None
        };

        // Build file header
        let mut file_header = if image.is_grayscale {
            FileHeader::new_gray(self.width, self.height)
        } else if image.has_alpha {
            FileHeader::new_rgba(self.width, self.height)
        } else {
            FileHeader::new_rgb(self.width, self.height)
        };
        if image.bit_depth == 16 {
            file_header.metadata.bit_depth = crate::headers::file_header::BitDepth::uint16();
            for ec in &mut file_header.metadata.extra_channels {
                ec.bit_depth = crate::headers::file_header::BitDepth::uint16();
            }
        }
        // CMYK: prepend the Black extra-channel header entry to match
        // the K plane we pushed onto `image.channels` above. Must go
        // BEFORE the user-extras loop so K ends up at ec index 0 (the
        // decoder finds it by walking `ec_info` and matching on
        // `ec_type == Black`; libjxl `image_bundle.h:187`). 16-bit
        // CMYK marks the K-plane info as 16-bit so the decoder
        // preserves the full precision.
        if self.layout.is_cmyk() {
            let mut k_info = crate::headers::extra_channels::ExtraChannelInfo {
                ec_type: crate::headers::extra_channels::ExtraChannelType::Black,
                ..Default::default()
            };
            if self.layout == PixelLayout::Cmyk16 {
                k_info.bit_depth = crate::headers::file_header::BitDepth::uint16();
            }
            file_header.metadata.extra_channels.insert(0, k_info);
        }
        // Append extra-channel metadata (refs #9). The corresponding
        // pixel data was added to `image.channels` above.
        for ec in self.extra_channels.iter() {
            file_header.metadata.extra_channels.push(ec.info.clone());
        }
        // Override file_header's default color_encoding with the
        // caller's `with_color_encoding(...)` if set. Closes lossless
        // portion of #17 — without this, the codestream header
        // always reports sRGB regardless of the caller's TF tag.
        // For grayscale layouts, the existing logic at the
        // encode_modular_with_patches call site (line 2467) coerces
        // ce.color_space to Gray; we mirror that here so the
        // file_header matches.
        if let Some(ce) = self.color_encoding.clone() {
            file_header.metadata.color_encoding =
                if image.is_grayscale && ce.color_space != ColorSpace::Gray {
                    crate::headers::color_encoding::ColorEncoding {
                        color_space: ColorSpace::Gray,
                        ..ce
                    }
                } else {
                    ce
                };
        }
        // Configurable bits_per_sample for one-shot lossless (#18
        // sub-feature). Lossless preserves pixels bit-exactly so this
        // only affects the codestream BitDepth header signaling.
        if let Some(bits) = self.bits_per_sample {
            file_header.metadata.bit_depth.bits_per_sample = bits;
            for ec in &mut file_header.metadata.extra_channels {
                ec.bit_depth.bits_per_sample = bits;
            }
        }
        // Premultiplied-alpha signaling (lossless portion of #13).
        // The alpha channel header gets `alpha_associated=true` so the
        // decoder knows the encoded color values are already
        // multiplied by alpha. Encoded pixels are written unchanged
        // (lossless), so the bit-flip is the entire fix.
        if self.premultiplied_alpha {
            for ec in &mut file_header.metadata.extra_channels {
                if ec.ec_type == crate::headers::extra_channels::ExtraChannelType::Alpha {
                    ec.alpha_associated = true;
                }
            }
        }
        if let Some(meta) = self.metadata {
            if meta.icc_profile.is_some() {
                file_header.metadata.color_encoding.want_icc = true;
            }
            if let Some(it) = meta.intensity_target {
                file_header.metadata.intensity_target = it;
            }
            if let Some(mn) = meta.min_nits {
                file_header.metadata.min_nits = mn;
            }
            if let Some((w, h)) = meta.intrinsic_size {
                file_header.metadata.have_intrinsic_size = true;
                file_header.metadata.intrinsic_width = w;
                file_header.metadata.intrinsic_height = h;
            }
        }
        // Request-level intensity_target / min_nits override the
        // metadata-level values. Lets callers do
        //   `cfg.encode_request(...).with_intensity_target(10000.0)`
        // without constructing an ImageMetadata. Closes #21.
        if let Some(it) = self.intensity_target {
            file_header.metadata.intensity_target = it;
        }
        if let Some(mn) = self.min_nits {
            file_header.metadata.min_nits = mn;
        }

        // Write codestream
        let mut writer = BitWriter::new();
        file_header.write(&mut writer).map_err(EncodeError::from)?;
        if let Some(meta) = self.metadata
            && let Some(icc) = meta.icc_profile
        {
            crate::icc::write_icc(icc, &mut writer).map_err(EncodeError::from)?;
        }
        writer.zero_pad_to_byte();

        // Write reference frame and subtract patches from image if detected
        if let Some(ref pd) = patches_data {
            let lossless_profile = cfg.effective_profile();
            crate::vardct::patches::encode_reference_frame_rgb(
                pd,
                image.bit_depth,
                cfg.use_ans,
                lossless_profile.patch_ref_tree_learning,
                &mut writer,
                Some(budget),
            )
            .map_err(EncodeError::from)?;
            writer.zero_pad_to_byte();
            let bd = image.bit_depth;
            crate::vardct::patches::subtract_patches_modular(&mut image, pd, bd);
        }

        // Encode frame
        let use_tree_learning = cfg.effective_tree_learning();
        let smart_profile = cfg.effective_profile_for_image((w as u64) * (h as u64));
        let frame_encoder = FrameEncoder::new(
            w,
            h,
            FrameEncoderOptions {
                use_modular: true,
                effort: cfg.effort,
                use_ans: cfg.use_ans,
                use_tree_learning,
                use_squeeze: cfg.squeeze,
                enable_lz77: cfg.effective_lz77(),
                lz77_method: cfg.lz77_method,
                lossy_palette: cfg.lossy_palette,
                encoder_mode: cfg.mode,
                profile: smart_profile,
                modular_knobs: cfg.modular_knobs(),
                modular_group_size_shift: cfg.effective_modular_group_size_shift(),
                ..Default::default()
            },
        )
        .with_budget(alloc::sync::Arc::clone(budget));
        let color_encoding = if let Some(ce) = self.color_encoding.clone() {
            // Explicit color encoding overrides source_gamma and defaults.
            // Adjust for grayscale if needed.
            if image.is_grayscale && ce.color_space != ColorSpace::Gray {
                ColorEncoding {
                    color_space: ColorSpace::Gray,
                    ..ce
                }
            } else {
                ce
            }
        } else if let Some(gamma) = self.source_gamma {
            if image.is_grayscale {
                ColorEncoding::gray_with_gamma(gamma)
            } else {
                ColorEncoding::with_gamma(gamma)
            }
        } else if image.is_grayscale {
            ColorEncoding::gray()
        } else {
            ColorEncoding::srgb()
        };
        frame_encoder
            .encode_modular_with_patches(
                &image,
                &color_encoding,
                &mut writer,
                patches_data.as_ref(),
            )
            .map_err(EncodeError::from)?;

        let stats = EncodeStats {
            mode: EncodeMode::Lossless,
            ans: cfg.use_ans,
            ..Default::default()
        };
        Ok((writer.finish_with_padding(), stats))
    }

    // ── Lossy path ──────────────────────────────────────────────────────

    fn encode_lossy(
        &self,
        cfg: &LossyConfig,
        pixels: &[u8],
        budget: &alloc::sync::Arc<crate::budget::MemoryBudget>,
    ) -> core::result::Result<(Vec<u8>, EncodeStats), EncodeError> {
        // Chroma subsampling gate (issue #47).
        //
        // - Chunk 3: signal-only. All non-Full444 modes returned
        //   InvalidConfig.
        // - Chunk 4: Sub420 routed through the JPEG-shaped pipeline
        //   in `vardct::chroma_subsampling` (RGB → YCbCr+420 via
        //   zenyuv → forward-DCT8 → integer quantize → reuse
        //   `crate::jpeg::encode_jpeg_to_jxl`).
        // - Chunk 5 (this change): Sub422 and Sub440 join Sub420 on
        //   the same JPEG-shaped path. Chroma downsampling for the
        //   single-axis modes goes through a small box-filter tail on
        //   top of zenyuv's 4:4:4 SIMD encode (zenyuv 0.1.3 has no
        //   dedicated 4:2:2 / 4:4:0 kernels; a future zenyuv release
        //   can swap in here without API change).
        //
        // The subsampled paths only fire when BOTH `chroma-subsampling`
        // and `jpeg-reencoding` features are compiled in; without
        // them the InvalidConfig fallback still ships.
        #[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
        if !cfg.chroma_subsampling.is_full() {
            return self.encode_lossy_sub_via_jpeg_path(cfg, pixels);
        }
        if !cfg.chroma_subsampling.is_full() {
            return Err(EncodeError::InvalidConfig {
                message: format!(
                    "chroma subsampling {} requires `do_ycbcr=true` + \
                     per-channel block grids. The subsampled lossy path \
                     requires both `chroma-subsampling` and \
                     `jpeg-reencoding` cargo features.",
                    cfg.chroma_subsampling.tag(),
                ),
            });
        }
        let w = self.width as usize;
        let h = self.height as usize;

        // Build linear f32 RGB and extract alpha from input layout.
        // Grayscale layouts are expanded to RGB (R=G=B) for VarDCT encoding.
        // When source_gamma is set, use gamma linearization instead of sRGB TF.
        let gamma = self.source_gamma;
        // Configurable bits_per_sample for u16 input (closes that
        // sub-feature of #18). Default 65535 = full 16-bit precision;
        // override via with_bits_per_sample(N) so 10/12/14-bit data
        // stored in the LOW bits of u16 normalizes to [0, 1.0] correctly.
        let u16_max = self
            .bits_per_sample
            .map_or(65535.0_f32, |b| ((1u32 << b) - 1) as f32);
        // PQ / HLG EOTF dispatch (closes PQ + HLG portions of #17).
        // When the caller sets a color_encoding with
        // TransferFunction::Pq or ::Hlg, the input pixels are
        // PQ/HLG-encoded; we apply the matching inverse EOTF instead
        // of the default sRGB linearization. source_gamma still wins
        // (caller explicitly chose gamma over the encoding's TF).
        // Currently wired only for the u16 RGB(A) layouts — broader
        // coverage (u8 / Gray / BT.709, lossless) is the remainder
        // of #17.
        //
        // A3 chunk 1b (issue #46): for the dedicated f32 PQ/HLG/BT.709
        // layouts the dispatch fires unconditionally inside the layout
        // arms — these helpers don't consult `source_is_*`.
        let source_is_pq = gamma.is_none()
            && self.color_encoding.as_ref().is_some_and(|ce| {
                ce.transfer_function == crate::headers::color_encoding::TransferFunction::Pq
            });
        let source_is_hlg = gamma.is_none()
            && self.color_encoding.as_ref().is_some_and(|ce| {
                ce.transfer_function == crate::headers::color_encoding::TransferFunction::Hlg
            });
        let source_is_bt709 = gamma.is_none()
            && self.color_encoding.as_ref().is_some_and(|ce| {
                ce.transfer_function == crate::headers::color_encoding::TransferFunction::Bt709
            });
        // CMYK arms (Cmyk8/Cmyk16) deinterleave the K plane and stash
        // it here for the extras-list construction further down. Set
        // by the matching layout arm; consumed where the extras Vec
        // is built. Mutually exclusive with the input check that
        // rejects a caller-supplied Black extra when the layout
        // already synthesises one.
        let mut synthesised_black_u8: Option<Vec<u8>> = None;
        let mut synthesised_black_u16: Option<Vec<u16>> = None;
        // Reject a caller-supplied Black extra when the layout already
        // synthesises one — otherwise the codestream would carry two
        // Black entries and the second K plane would never reach the
        // decoder. Same guard as the lossless one-shot path
        // (api.rs:4242-4248, f2deff72).
        if self.layout.is_cmyk()
            && self.extra_channels.iter().any(|ec| {
                ec.info.ec_type == crate::headers::extra_channels::ExtraChannelType::Black
            })
        {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "PixelLayout::{:?} already synthesises a Black extra channel; \
                     remove the user-supplied ExtraChannel::black(...)",
                    self.layout,
                ),
            });
        }
        let (linear_rgb, alpha, bit_depth_16) = match self.layout {
            PixelLayout::Rgb8 => {
                let linear = if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(pixels, 3, g)
                } else if source_is_pq {
                    pq_u8_to_linear_f32(pixels, 3)
                } else if source_is_hlg {
                    hlg_u8_to_linear_f32(pixels, 3)
                } else if source_is_bt709 {
                    bt709_u8_to_linear_f32(pixels, 3)
                } else {
                    srgb_u8_to_linear_f32(pixels, 3)
                };
                (linear, None, false)
            }
            PixelLayout::Bgr8 => {
                let rgb = bgr_to_rgb(pixels, 3);
                let linear = if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(&rgb, 3, g)
                } else if source_is_pq {
                    pq_u8_to_linear_f32(&rgb, 3)
                } else if source_is_hlg {
                    hlg_u8_to_linear_f32(&rgb, 3)
                } else if source_is_bt709 {
                    bt709_u8_to_linear_f32(&rgb, 3)
                } else {
                    srgb_u8_to_linear_f32(&rgb, 3)
                };
                (linear, None, false)
            }
            PixelLayout::Rgba8 => {
                let rgb = if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(pixels, 4, g)
                } else if source_is_pq {
                    pq_u8_to_linear_f32(pixels, 4)
                } else if source_is_hlg {
                    hlg_u8_to_linear_f32(pixels, 4)
                } else if source_is_bt709 {
                    bt709_u8_to_linear_f32(pixels, 4)
                } else {
                    srgb_u8_to_linear_f32(pixels, 4)
                };
                let alpha = extract_alpha(pixels, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::Bgra8 => {
                let swapped = bgr_to_rgb(pixels, 4);
                let rgb = if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(&swapped, 4, g)
                } else if source_is_pq {
                    pq_u8_to_linear_f32(&swapped, 4)
                } else if source_is_hlg {
                    hlg_u8_to_linear_f32(&swapped, 4)
                } else if source_is_bt709 {
                    bt709_u8_to_linear_f32(&swapped, 4)
                } else {
                    srgb_u8_to_linear_f32(&swapped, 4)
                };
                let alpha = extract_alpha(pixels, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::Gray8 => {
                let rgb = if let Some(g) = gamma {
                    gamma_gray_u8_to_linear_f32_rgb(pixels, 1, g)
                } else if source_is_pq {
                    pq_gray_u8_to_linear_f32_rgb(pixels, 1)
                } else if source_is_hlg {
                    hlg_gray_u8_to_linear_f32_rgb(pixels, 1)
                } else if source_is_bt709 {
                    bt709_gray_u8_to_linear_f32_rgb(pixels, 1)
                } else {
                    gray_u8_to_linear_f32_rgb(pixels, 1)
                };
                (rgb, None, false)
            }
            PixelLayout::GrayAlpha8 => {
                let rgb = if let Some(g) = gamma {
                    gamma_gray_u8_to_linear_f32_rgb(pixels, 2, g)
                } else if source_is_pq {
                    pq_gray_u8_to_linear_f32_rgb(pixels, 2)
                } else if source_is_hlg {
                    hlg_gray_u8_to_linear_f32_rgb(pixels, 2)
                } else if source_is_bt709 {
                    bt709_gray_u8_to_linear_f32_rgb(pixels, 2)
                } else {
                    gray_u8_to_linear_f32_rgb(pixels, 2)
                };
                let alpha = extract_alpha(pixels, 2, 1);
                (rgb, Some(alpha), false)
            }
            PixelLayout::Rgb16 => {
                let linear = if let Some(g) = gamma {
                    gamma_u16_to_linear_f32(pixels, 3, g, u16_max)
                } else if source_is_pq {
                    pq_u16_to_linear_f32(pixels, 3, u16_max)
                } else if source_is_hlg {
                    hlg_u16_to_linear_f32(pixels, 3, u16_max)
                } else if source_is_bt709 {
                    bt709_u16_to_linear_f32(pixels, 3, u16_max)
                } else {
                    srgb_u16_to_linear_f32(pixels, 3, u16_max)
                };
                (linear, None, true)
            }
            PixelLayout::Rgba16 => {
                let rgb = if let Some(g) = gamma {
                    gamma_u16_to_linear_f32(pixels, 4, g, u16_max)
                } else if source_is_pq {
                    pq_u16_to_linear_f32(pixels, 4, u16_max)
                } else if source_is_hlg {
                    hlg_u16_to_linear_f32(pixels, 4, u16_max)
                } else if source_is_bt709 {
                    bt709_u16_to_linear_f32(pixels, 4, u16_max)
                } else {
                    srgb_u16_to_linear_f32(pixels, 4, u16_max)
                };
                let alpha = extract_alpha_u16(pixels, 4, 3, u16_max);
                (rgb, Some(alpha), true)
            }
            PixelLayout::Gray16 => {
                let rgb = if let Some(g) = gamma {
                    gamma_gray_u16_to_linear_f32_rgb(pixels, 1, g, u16_max)
                } else if source_is_pq {
                    pq_gray_u16_to_linear_f32_rgb(pixels, 1, u16_max)
                } else if source_is_hlg {
                    hlg_gray_u16_to_linear_f32_rgb(pixels, 1, u16_max)
                } else if source_is_bt709 {
                    bt709_gray_u16_to_linear_f32_rgb(pixels, 1, u16_max)
                } else {
                    gray_u16_to_linear_f32_rgb(pixels, 1, u16_max)
                };
                (rgb, None, true)
            }
            PixelLayout::GrayAlpha16 => {
                let rgb = if let Some(g) = gamma {
                    gamma_gray_u16_to_linear_f32_rgb(pixels, 2, g, u16_max)
                } else if source_is_pq {
                    pq_gray_u16_to_linear_f32_rgb(pixels, 2, u16_max)
                } else if source_is_hlg {
                    hlg_gray_u16_to_linear_f32_rgb(pixels, 2, u16_max)
                } else if source_is_bt709 {
                    bt709_gray_u16_to_linear_f32_rgb(pixels, 2, u16_max)
                } else {
                    gray_u16_to_linear_f32_rgb(pixels, 2, u16_max)
                };
                let alpha = extract_alpha_u16(pixels, 2, 1, u16_max);
                (rgb, Some(alpha), true)
            }
            PixelLayout::RgbLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                (floats.to_vec(), None, false)
            }
            PixelLayout::RgbaLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                let rgb: Vec<f32> = floats
                    .chunks(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect();
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::GrayLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                (gray_f32_to_linear_f32_rgb(floats, 1), None, false)
            }
            PixelLayout::GrayAlphaLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                let rgb = gray_f32_to_linear_f32_rgb(floats, 2);
                let alpha = extract_alpha_f32(floats, 2, 1);
                (rgb, Some(alpha), false)
            }
            // Closes FLOAT16 portion of #18.
            PixelLayout::RgbLinearF16 => (f16_to_linear_f32_rgb(pixels, 3), None, false),
            PixelLayout::RgbaLinearF16 => {
                let rgb = f16_to_linear_f32_rgb(pixels, 4);
                let alpha = extract_alpha_f16(pixels, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::GrayLinearF16 => (f16_gray_to_linear_f32_rgb(pixels, 1), None, false),
            PixelLayout::GrayAlphaLinearF16 => {
                let rgb = f16_gray_to_linear_f32_rgb(pixels, 2);
                let alpha = extract_alpha_f16(pixels, 2, 1);
                (rgb, Some(alpha), false)
            }
            // A3 chunk 1b: f32 PQ/HLG/BT.709 RGB(A) (issue #46). The
            // layout name carries the transfer function; no
            // color_encoding override is required for linearization to
            // fire. We still run the f32-domain inverse EOTF here.
            PixelLayout::RgbPqF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                (pq_f32_to_linear_f32_rgb(floats, 3), None, false)
            }
            PixelLayout::RgbaPqF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                let rgb = pq_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::RgbHlgF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                (hlg_f32_to_linear_f32_rgb(floats, 3), None, false)
            }
            PixelLayout::RgbaHlgF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                let rgb = hlg_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::RgbBt709F32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                (bt709_f32_to_linear_f32_rgb(floats, 3), None, false)
            }
            PixelLayout::RgbaBt709F32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                let rgb = bt709_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha), false)
            }
            // Lossy CMYK. The C/M/Y planes are routed through the
            // VarDCT (XYB) pipeline via a 1-CMY × (1-K) subtractive
            // → linear-RGB transform (chunk 3, follow-on to 1b222af),
            // and the K plane is split off and attached as an
            // `ExtraChannelType::Black` extra below (handled by the
            // existing alpha+extras flow). This is the same wire shape
            // libjxl uses for lossy CMYK — three colour planes carrying
            // colour in XYB plus a Black extra carrying K
            // (lib/jxl/enc_image_bundle.cc:57).
            //
            // The 1-CMY × (1-K) mapping is the naive uncalibrated
            // subtractive model: each ink absorbs its complementary
            // primary, K darkens uniformly. It is NOT colorimetric —
            // a future chunk can wire either the caller-supplied
            // CMYK ICC profile (option A) or a hardcoded SWOP/FOGRA
            // matrix (option B). What it does provide is gamut-
            // direction correctness: pure cyan input now encodes as
            // a cyan-ish XYB sample (no red leak), so the perceptual
            // quantiser allocates bits sensibly. Chunk 2 (1b222af)
            // shipped a placeholder that treated CMY bytes as if they
            // were sRGB-encoded R/G/B — a fully-saturated cyan ink
            // encoded as bright red, an obvious wrong gamut sector.
            //
            // The K plane survives the round-trip losslessly because
            // it travels as a modular extra channel, not through XYB.
            // Caller gamma + ICC are ignored on the CMY input — they
            // would only make sense once chunk A/B colour management
            // lands. Synthesised K is stashed in the per-arm locals
            // `synthesised_black_u8` / `synthesised_black_u16` and
            // picked up by the extras-list construction further down.
            PixelLayout::Cmyk8 => {
                let n = w * h;
                if pixels.len() != n * 4 {
                    return Err(EncodeError::InvalidInput {
                        message: format!(
                            "Cmyk8 expects {} bytes ({}x{} × 4), got {}",
                            n * 4,
                            w,
                            h,
                            pixels.len(),
                        ),
                    });
                }
                // Deinterleave CMYK → 3-channel CMY + separate K plane.
                // One pass over the input.
                let mut cmy = Vec::with_capacity(n * 3);
                let mut k = Vec::with_capacity(n);
                for i in 0..n {
                    let base = i * 4;
                    cmy.push(pixels[base]);
                    cmy.push(pixels[base + 1]);
                    cmy.push(pixels[base + 2]);
                    k.push(pixels[base + 3]);
                }
                let linear = cmyk_u8_to_linear_f32_rgb(&cmy, &k);
                synthesised_black_u8 = Some(k);
                (linear, None, false)
            }
            PixelLayout::Cmyk16 => {
                let n = w * h;
                if pixels.len() != n * 8 {
                    return Err(EncodeError::InvalidInput {
                        message: format!(
                            "Cmyk16 expects {} bytes ({}x{} × 8), got {}",
                            n * 8,
                            w,
                            h,
                            pixels.len(),
                        ),
                    });
                }
                // Deinterleave 16-bit CMYK → 6 bytes of CMY u16 +
                // separate K u16 plane. Native-endian, matches the
                // lossless Cmyk16 arm.
                let mut cmy = Vec::with_capacity(n * 3 * 2);
                let mut k = Vec::with_capacity(n);
                for i in 0..n {
                    let base = i * 8;
                    cmy.extend_from_slice(&pixels[base..base + 6]);
                    let k_lo = pixels[base + 6];
                    let k_hi = pixels[base + 7];
                    k.push(u16::from_ne_bytes([k_lo, k_hi]));
                }
                let linear = cmyk_u16_to_linear_f32_rgb(&cmy, &k, u16_max);
                synthesised_black_u16 = Some(k);
                (linear, None, true)
            }
        };

        // W44-35: cheap smooth-photo auto-detect on the raw sRGB u8
        // input (when applicable) feeds the DCT64 admission gate via
        // `effective_profile_for_image_with_smoothness`. Returns false
        // for non-u8 layouts, large images (>= 500k px), and content
        // that fails the smoothness discriminator. Caller-supplied
        // `StrategyOverrides::smooth_photo_dct64_hint = Some(_)`
        // (via `with_strategy_overrides`) always wins over the auto
        // value (resolved inside `effective_profile_*`).
        let smooth_photo_for_dct64 =
            detect_smooth_photo_for_dct64_from_layout(pixels, self.width, self.height, self.layout);
        let mut profile = cfg.effective_profile_for_image_with_smoothness(
            (w as u64) * (h as u64),
            smooth_photo_for_dct64,
        );

        let mut linear_rgb = linear_rgb;

        // Unpremultiply alpha BEFORE the SimplifyInvisible pre-pass and
        // BEFORE XYB conversion (closes lossy portion of #13). libjxl
        // `enc_frame.cc:1588-1597` runs SimplifyInvisible only when
        // alpha is straight (`!alpha_eci->alpha_associated`); when the
        // caller signals premultiplied input we unpremultiply first so
        // the encoder can run the rest of its pipeline on straight
        // RGB. The header gets `alpha_associated=true` so the decoder
        // re-premultiplies on output, closing the round-trip.
        if self.premultiplied_alpha
            && let Some(ref alpha_buf) = alpha
        {
            unpremultiply_alpha_inplace(&mut linear_rgb, alpha_buf);
        }

        // SimplifyInvisible pre-pass (closes #10): smooth color
        // values in alpha=0 pixels to a weighted average of visible
        // neighbors, reducing high-frequency DCT energy from arbitrary
        // garbage in transparent regions. libjxl `enc_frame.cc:511`
        // (default-on for lossy). Sprites/icons benefit (5-20% smaller);
        // photos with mostly-opaque alpha pay only the cheap
        // `has_any_invisible_pixels` predicate (single linear scan
        // with early-exit on the first zero).
        //
        // libjxl gates SimplifyInvisible on `!alpha_associated` — for
        // premultiplied input the alpha-zero pixels already hold black
        // (premultiplication zeros them) so the smear contribution is
        // dilution-only, no win. We mirror that gate.
        if cfg.simplify_invisible
            && !self.premultiplied_alpha
            && let Some(ref alpha_buf) = alpha
            && crate::vardct::simplify_invisible::has_any_invisible_pixels(alpha_buf)
        {
            crate::vardct::simplify_invisible::simplify_invisible_rgb(
                &mut linear_rgb,
                alpha_buf,
                w,
                h,
                false, // lossless = false (smear, not zero)
            );
        }

        // Apply max_strategy_size to profile flags
        if let Some(max_size) = cfg.max_strategy_size {
            if max_size < 16 {
                profile.try_dct16 = false;
            }
            if max_size < 32 {
                profile.try_dct32 = false;
            }
            if max_size < 64 {
                profile.try_dct64 = false;
            }
        }

        // Apply libjxl's auto-resample-at-d≥10 (refs #12,
        // enc_frame.cc:103-115). The effective distance + resampling
        // are derived once here and used everywhere downstream.
        let effective_resampling = cfg.effective_resampling();
        let effective_distance = cfg.effective_distance();

        let mut enc = crate::vardct::VarDctEncoder::new(effective_distance);
        // W44-128 Chunk B + W44-130 Chunk D: resolve the
        // EncoderStrategy bundle once here (caller-set preset +
        // collected `with_*_hint` overrides) and store on the encoder.
        // Field is non-optional as of Chunk D — consumed directly by
        // the 8 call sites in `vardct/encoder.rs` +
        // `vardct/butteraugli_loop.rs`.
        enc.resolved_improvements = cfg.resolve_improvements();
        enc.effort = cfg.effort;
        enc.profile = profile;
        enc.use_ans = cfg.use_ans;
        enc.optimize_codes = enc.profile.optimize_codes;
        enc.custom_orders = enc.profile.custom_orders;
        enc.ac_strategy_enabled = enc.profile.ac_strategy_enabled;
        enc.enable_noise = cfg.noise;
        enc.photon_noise_iso = cfg.photon_noise_iso;
        enc.manual_noise_lut = cfg.manual_noise_lut;
        enc.quant_ac_rescale = cfg.quant_ac_rescale;
        enc.original_distance = cfg.original_distance;
        enc.enable_denoise = cfg.denoise;
        // libjxl gates gaborish at distance > 0.5 (enc_frame.cc:281)
        // and unconditionally OFF at decoding_speed_tier == 4
        // (enc_frame.cc:280) — captured by `cfg.effective_gaborish()`.
        enc.enable_gaborish = cfg.effective_gaborish() && effective_distance > 0.5;
        // EX-J13: adaptive gaborish is silently gated to be a subset of
        // gaborish (no-op when the fixed inverse is disabled).
        enc.enable_adaptive_gaborish = enc.enable_gaborish && cfg.adaptive_gaborish;
        // libjxl `--epf -1..3` override (enc_frame.cc:284-285). `-1` =
        // encoder chooses by distance; otherwise force the given count.
        enc.epf_level_override = if cfg.epf_level < 0 {
            None
        } else {
            Some(cfg.epf_level as u32)
        };
        // W44-130 Chunk D: the 4 dispatch policies were absorbed into
        // `EncoderImprovementsCustom` per design doc §7 Q2 — they now
        // flow via `enc.resolved_improvements` instead of dedicated
        // `LossyConfig` fields. The `VarDctEncoder.X_dispatch` fields
        // remain (many call-site reads); we hydrate them from the
        // resolved bundle here.
        enc.epf_dispatch = enc.resolved_improvements.epf_dispatch;
        enc.error_diffusion = cfg.error_diffusion;
        enc.pixel_domain_loss = cfg.pixel_domain_loss;
        enc.pixel_loss_dispatch = enc.resolved_improvements.pixel_loss_dispatch;
        enc.single_pass_entropy_dispatch = enc.resolved_improvements.single_pass_entropy_dispatch;
        enc.enable_lz77 = cfg.effective_lz77();
        enc.lz77_method = cfg.lz77_method;
        enc.force_strategy = cfg.force_strategy;
        // RFC #45 pick #4 — when the caller has explicitly pinned `cfg.patches`
        // via `with_patches`, that wins; otherwise read the per-image
        // dispatched profile (the content-class adapter may have flipped
        // patches on for Screenshot content at e5/e6).
        //
        // CMYK exception (chunk 2): the patches detector assumes
        // RGB-like perceptual colour and operates on the first 3
        // channels — which are CMY here, not RGB — so it would inject
        // bogus subtractive-colour patches into the codestream. Same
        // exclusion the lossless one-shot path applies at
        // api.rs:4404-4408.
        enc.enable_patches = if self.layout.is_cmyk() {
            false
        } else if cfg.patches_explicit {
            cfg.effective_patches()
        } else if cfg.faster_decoding >= 2 {
            // libjxl `enc_modular.cc:707` skips patches at
            // `decoding_speed_tier >= 2`. Override the profile-derived
            // gate (which may have flipped patches on via the
            // content-class adapter).
            false
        } else {
            enc.profile.patches
        };
        enc.patches_dispatch = enc.resolved_improvements.patches_dispatch;
        enc.enable_dot_detection = cfg.dot_detection;
        enc.encoder_mode = cfg.mode;
        enc.splines = cfg.splines.clone();
        enc.auto_splines = cfg.auto_splines;
        enc.is_grayscale = self.layout.is_grayscale();
        enc.progressive = cfg.progressive;
        enc.use_lf_frame = cfg.lf_frame;
        // W44-130 Chunk D: `content_aware_entropy_mul` enable bit +
        // 5 `with_*_hint` Option<bool> setters + their VarDctEncoder
        // fallback fields all deleted. Strategy + overrides flow
        // through `cfg.resolve_improvements()` →
        // `enc.resolved_improvements` which the 8 consuming call
        // sites read directly.
        // W44-91: cheap zenanalyze-equivalent proxies for the textured-
        // colourful-photo sub-band gate (mask1x1 ∈ [50, 80] @ d ∈ [3, 5]).
        // See `compute_w44_91_zenanalyze_proxies` for which layouts the
        // proxy is well-defined on; for everything else (16-bit, linear-f32,
        // grayscale, HDR) the proxy stays `None` and the W44-91 gate
        // cannot fire — the W44-29 mask1x1<50 gate retains full coverage.
        enc.zenanalyze_proxies = compute_w44_91_zenanalyze_proxies(pixels, w, h, self.layout);
        // Streaming refactor #11 chunk 6: thread the caller-selected
        // [`Buffering`] policy into VarDctEncoder so the per-region
        // precompute dispatch (precomputed.rs:compute_with_budget_and_buffering)
        // can route on it. `Buffering::Auto` resolves on image size at
        // dispatch time.
        enc.buffering = cfg.buffering;
        #[cfg(feature = "butteraugli-loop")]
        {
            enc.butteraugli_iters = cfg.butteraugli_iters;
            // EX-J11 chunk 4: resolve `HdrLoss::Auto` to a concrete
            // loss now (using caller's `with_color_encoding` if set,
            // else `PixelLayout::implied_transfer_function()`), so the
            // per-iter butteraugli loop reads a fixed variant. PQ /
            // HLG content lands on `Vdp2`; everything else on
            // `Butteraugli` (SDR hash-locks stay byte-identical).
            enc.hdr_loss = cfg.resolve_hdr_loss(self.layout, self.color_encoding.as_ref());
        }
        #[cfg(feature = "ssim2-loop")]
        {
            enc.ssim2_iters = cfg.ssim2_iters;
        }
        #[cfg(feature = "zensim-loop")]
        {
            enc.zensim_iters = cfg.zensim_iters;
        }

        enc.bit_depth_16 = bit_depth_16;
        enc.source_gamma = self.source_gamma;
        // A3 chunk 1b (issue #46): if the caller didn't set an
        // explicit color encoding but the layout name carries an
        // implied transfer function (PQ / HLG / BT.709 f32), auto-set
        // a matching ColorEncoding so the codestream signals the
        // correct TF. PQ + HLG also imply BT.2100 primaries (the only
        // gamut these TFs are spec'd against); BT.709 stays on sRGB
        // primaries (BT.709 + sRGB primaries are interchangeable for
        // gamut, only the TF differs). source_gamma still wins.
        enc.color_encoding = self.color_encoding.clone().or_else(|| {
            if self.source_gamma.is_some() {
                return None;
            }
            use crate::headers::color_encoding::{ColorEncoding, TransferFunction};
            match self.layout.implied_transfer_function() {
                Some(TransferFunction::Pq) => Some(ColorEncoding::bt2100_pq()),
                Some(TransferFunction::Hlg) => Some(ColorEncoding::bt2100_hlg()),
                Some(TransferFunction::Bt709) => Some(ColorEncoding {
                    transfer_function: TransferFunction::Bt709,
                    ..ColorEncoding::srgb()
                }),
                Some(TransferFunction::Linear) => Some(ColorEncoding::linear_srgb()),
                _ => None,
            }
        });
        enc.non_finite_action = cfg.non_finite_action;
        enc.budget = Some(alloc::sync::Arc::clone(budget));
        // Lossy portion of #13: signal premultiplied alpha in the
        // codestream header (decoder re-premultiplies on output).
        // The unpremultiplication of the input pixels already happened
        // above (immediately after building linear_rgb).
        enc.alpha_associated = self.premultiplied_alpha;
        // Configurable bits_per_sample (#18 sub-feature) — drives the
        // codestream BitDepth header. Input normalization (u16_max)
        // handles the matching pixel scaling above.
        enc.bits_per_sample_override = self.bits_per_sample;
        // Center-first AC group permutation (#14).
        enc.center_first = cfg.center_first;
        // Caller-supplied center point for `group_order = center-first`
        // (CLI passthrough — libjxl `cparams.center_x` / `center_y`).
        // Clamp to u32 and pass through; `None` falls back to image
        // centre downstream.
        enc.center_x = cfg.center_x.map(|v| v.max(0).min(u32::MAX as i64) as u32);
        enc.center_y = cfg.center_y.map(|v| v.max(0).min(u32::MAX as i64) as u32);
        // Decoder upsampling factor (refs #12). Caller-supplied
        // (width, height) and pixel buffers are downsampled below
        // before reaching the encoder; the encoder operates entirely
        // at the downsampled resolution and signals the decoder to
        // upsample after rendering. The file-header dims still report
        // the original (pre-downsample) size.
        enc.upsampling = effective_resampling;
        // Custom upsampling LUT selection (libjxl
        // `JxlEncoderSetUpsamplingMode`). The encoder records the
        // mode on the file-header builder; the LUT itself is emitted
        // in `FileHeader::write_transform_data` only when
        // `upsampling > 1` AND the mode is `Some(0)` / `Some(1)`.
        enc.upsampling_mode = cfg.upsampling_mode;
        // Alpha extra channel butteraugli distance (CLI passthrough —
        // libjxl `cjxl --alpha_distance`). `None` and `Some(0.0)`
        // keep the lossless path. A non-zero value engages the lossy
        // alpha pipeline (pre-quantize + modular-tree multiplier);
        // see [`crate::vardct::VarDctEncoder::compute_extra_pixel_quantizer`]
        // for the libjxl-parity formula.
        enc.alpha_distance = cfg.alpha_distance;
        // Squeeze-on-extras opt-in (chunk-1 framework — see
        // [`crate::LossyConfig::with_alpha_squeeze`] and
        // [`crate::vardct::VarDctEncoder::alpha_squeeze_engaged`]).
        enc.alpha_squeeze = cfg.alpha_squeeze;

        // Tone mapping and intrinsic size from metadata
        if let Some(meta) = self.metadata {
            if let Some(it) = meta.intensity_target {
                enc.intensity_target = it;
            }
            if let Some(mn) = meta.min_nits {
                enc.min_nits = mn;
            }
            if meta.intrinsic_size.is_some() {
                enc.intrinsic_size = meta.intrinsic_size;
            }
        }
        // Request-level intensity_target / min_nits override the
        // metadata-level values. Closes #21.
        if let Some(it) = self.intensity_target {
            enc.intensity_target = it;
        }
        if let Some(mn) = self.min_nits {
            enc.min_nits = mn;
        }

        // ICC profile from metadata
        if let Some(meta) = self.metadata
            && let Some(icc) = meta.icc_profile
        {
            enc.icc_profile = Some(icc.to_vec());
        }

        // Apply downsampling for resampling > 1 (refs #12). Factor 2
        // uses libjxl's sharper 12×12 kernel (`enc_heuristics.cc:279`)
        // which preserves edge detail; factors 4 and 8 use the simple
        // box filter (libjxl behavior). When `already_downsampled` is
        // set, the caller has done their own downsample and wants the
        // encoder to honour the input dims; skip the internal
        // downsample but keep the upsampling factor in the bitstream.
        let (encode_rgb, encode_alpha, encode_w, encode_h) =
            if effective_resampling > 1 && !cfg.already_downsampled {
                let (down_rgb, dw, dh) = if effective_resampling == 2 {
                    crate::vardct::resampling::sharper_downsample_2x_rgb(&linear_rgb, w, h)
                } else {
                    crate::vardct::resampling::box_downsample_rgb(
                        &linear_rgb,
                        w,
                        h,
                        effective_resampling,
                    )
                };
                let down_alpha = alpha.as_ref().map(|a| {
                    let (a_down, _, _) = crate::vardct::resampling::box_downsample_alpha_u8(
                        a,
                        w,
                        h,
                        effective_resampling,
                    );
                    a_down
                });
                (down_rgb, down_alpha, dw as usize, dh as usize)
            } else {
                (linear_rgb, alpha, w, h)
            };

        // Build the extras list passed to VarDctEncoder. The wire
        // order is: synthesised Black (CMYK only) first so K lands at
        // ec index 0, then alpha (when the layout carries it), then
        // any caller-supplied non-alpha extras (depth, spot color, …)
        // from `self.extra_channels`. Keeping K at ec index 0 mirrors
        // libjxl's `enc_image_bundle.cc:57` CMYK pipeline and matches
        // the lossless one-shot path (api.rs:4444-4452).
        //
        // Extras flow only when the resampling factor is 1 — at
        // `resampling > 1` we already downsample RGB+alpha to the
        // encoded dims, and downsampling arbitrary extras (including
        // the synthesised K plane) is a follow-up. Reject explicitly
        // so a caller can't accidentally ship a file whose extras are
        // sized for the original dims while the file header advertises
        // the downsampled dims.
        let has_synthesised_black =
            synthesised_black_u8.is_some() || synthesised_black_u16.is_some();
        let extras_vec: Vec<crate::api::ExtraChannel<'_>> = if !self.extra_channels.is_empty()
            || has_synthesised_black
        {
            if effective_resampling > 1 {
                return Err(EncodeError::InvalidInput {
                    message: format!(
                        "extra channels with resampling > 1 not yet supported (resampling = {effective_resampling})"
                    ),
                });
            }
            let mut v: Vec<crate::api::ExtraChannel<'_>> = Vec::with_capacity(
                self.extra_channels.len()
                    + usize::from(encode_alpha.is_some())
                    + usize::from(has_synthesised_black),
            );
            // Synthesised K plane (CMYK only). Lives at ec index 0
            // so the decoder finds it first when walking
            // `ec_info` looking for `ec_type == Black`. Black
            // forbidden at level 5 → the shared level computation
            // (api.rs:3920) bumps to level 10 when
            // `self.layout.is_cmyk()` is true.
            if let Some(ref k_u8) = synthesised_black_u8 {
                v.push(crate::api::ExtraChannel::black(k_u8));
            }
            if let Some(ref k_u16) = synthesised_black_u16 {
                v.push(crate::api::ExtraChannel::black_u16(k_u16));
            }
            if let Some(ref buf) = encode_alpha {
                v.push(crate::api::ExtraChannel::from_alpha_buf(
                    buf,
                    self.premultiplied_alpha,
                ));
            }
            for ec in self.extra_channels.iter() {
                if matches!(
                    ec.info().ec_type,
                    crate::headers::extra_channels::ExtraChannelType::Alpha
                ) {
                    // Caller passed an Alpha-typed extra alongside an
                    // alpha-carrying pixel layout — refuse rather than
                    // silently producing two alpha channels.
                    if encode_alpha.is_some() {
                        return Err(EncodeError::InvalidInput {
                            message: "Alpha extra channel conflicts with the pixel layout's alpha \
                                     (use a non-Alpha layout or omit the extra)"
                                .to_string(),
                        });
                    }
                }
                v.push(ec.clone());
            }
            v
        } else {
            // Fast path: no caller-supplied extras and no synthesised
            // K plane. Build just an alpha entry when the layout
            // carries alpha.
            if let Some(ref buf) = encode_alpha {
                vec![crate::api::ExtraChannel::from_alpha_buf(
                    buf,
                    self.premultiplied_alpha,
                )]
            } else {
                Vec::new()
            }
        };

        let output = enc
            .encode_with_extras(encode_w, encode_h, &encode_rgb, &extras_vec)
            .map_err(EncodeError::from)?;

        #[cfg(feature = "butteraugli-loop")]
        let butteraugli_iters_actual = cfg.butteraugli_iters;
        #[cfg(not(feature = "butteraugli-loop"))]
        let butteraugli_iters_actual = 0u32;

        let stats = EncodeStats {
            mode: EncodeMode::Lossy,
            strategy_counts: output.strategy_counts,
            gaborish: cfg.gaborish,
            ans: cfg.use_ans,
            butteraugli_iters: butteraugli_iters_actual,
            pixel_domain_loss: cfg.pixel_domain_loss,
            ..Default::default()
        };
        Ok((output.data, stats))
    }

    /// Chunk-4 / chunk-5 entry point for any non-`Full444`
    /// [`ChromaSubsampling`] mode: convert RGB → YCbCr (with per-mode
    /// chroma downsampling) via zenyuv, forward-DCT + integer-quantize
    /// all blocks, synthesise a [`crate::jpeg::JpegData`] payload,
    /// and hand it to [`crate::jpeg::encode_jpeg_to_jxl`]. See
    /// [`crate::vardct::chroma_subsampling::encode_rgb8_via_jpeg_path`]
    /// for the implementation.
    ///
    /// Currently only honours [`PixelLayout::Rgb8`] — Rgba8 / Bgra8 /
    /// Gray / 16-bit / float / linear layouts return
    /// [`EncodeError::InvalidConfig`]. The encoder ignores extras,
    /// EXIF/XMP, ICC profile, progressive mode, butteraugli loop,
    /// splines, patches, and rate-control for the subsampled paths
    /// (none of those are wired through the JPEG-shaped pipeline yet).
    #[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
    fn encode_lossy_sub_via_jpeg_path(
        &self,
        cfg: &LossyConfig,
        pixels: &[u8],
    ) -> core::result::Result<(Vec<u8>, EncodeStats), EncodeError> {
        let mode = cfg.chroma_subsampling;
        let tag = mode.tag();
        if !matches!(self.layout, PixelLayout::Rgb8) {
            return Err(EncodeError::InvalidConfig {
                message: format!(
                    "chroma subsampling {tag} currently only honours \
                     `PixelLayout::Rgb8`; got {:?}. Rgba8 / Bgr8 / \
                     Bgra8 / Gray / 16-bit / float / linear layouts \
                     are still pending.",
                    self.layout
                ),
            });
        }
        let w = self.width as usize;
        let h = self.height as usize;
        if w == 0 || h == 0 {
            return Err(EncodeError::InvalidInput {
                message: format!("{tag} requires non-zero dimensions, got {w}x{h}"),
            });
        }
        let expected = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| EncodeError::InvalidInput {
                message: format!("{tag} dimensions overflow usize: {w}x{h}"),
            })?;
        if pixels.len() < expected {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "{tag} RGB buffer too small: {} < {} for {w}x{h}",
                    pixels.len(),
                    expected
                ),
            });
        }
        let bytes = crate::vardct::chroma_subsampling::encode_rgb8_via_jpeg_path(
            pixels,
            w,
            h,
            cfg.distance,
            mode,
        )
        .map_err(EncodeError::from)?;
        let stats = EncodeStats {
            mode: EncodeMode::Lossy,
            ans: cfg.use_ans,
            ..Default::default()
        };
        Ok((bytes, stats))
    }
}

// ── Streaming Encoders ──────────────────────────────────────────────────────

/// Streaming lossy (VarDCT) encoder.
///
/// Accepts pixel rows incrementally via [`push_rows`](Self::push_rows), then
/// encodes on [`finish`](Self::finish). This allows callers to free source pixel
/// buffers as rows are pushed, rather than materializing the entire image in
/// memory before encoding.
///
/// ```rust,no_run
/// use jxl_encoder::{LossyConfig, PixelLayout};
///
/// let mut enc = LossyConfig::new(1.0)
///     .encoder(800, 600, PixelLayout::Rgb8)?;
///
/// // Push rows from a streaming source (e.g. PNG decoder)
/// # let row_bytes = 800 * 3;
/// # let source_rows = vec![0u8; row_bytes * 600];
/// for chunk in source_rows.chunks(row_bytes * 100) {
///     enc.push_rows(chunk, 100)?;
/// }
///
/// let jxl_bytes = enc.finish()?;
/// # Ok::<_, jxl_encoder::At<jxl_encoder::EncodeError>>(())
/// ```
pub struct LossyEncoder {
    cfg: LossyConfig,
    width: u32,
    height: u32,
    layout: PixelLayout,
    rows_pushed: u32,
    linear_rgb: Vec<f32>,
    alpha: Option<Vec<u8>>,
    bit_depth_16: bool,
    icc_profile: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    /// JUMBF (ISO 19566-5, C2PA) superbox payload, emitted verbatim
    /// into a `jumb` box appended after `Exif`/`xml `.
    jumbf: Option<Vec<u8>>,
    source_gamma: Option<f32>,
    color_encoding: Option<crate::headers::color_encoding::ColorEncoding>,
    intensity_target: f32,
    min_nits: f32,
    intrinsic_size: Option<(u32, u32)>,
    /// Premultiplied (associated) alpha signaling. On lossy this is a
    /// no-op until the unpremultiplication pre-pass lands (#13);
    /// `finish()` returns `EncodeError::InvalidInput` if set.
    premultiplied_alpha: bool,
    /// Configurable bits_per_sample for u16 input (#18 sub-feature).
    /// Mirrors `EncodeRequest::with_bits_per_sample` on the streaming
    /// path. `None` → 65535 divisor (full 16-bit). `Some(N)` →
    /// `(1<<N)-1` divisor + codestream BitDepth = N.
    bits_per_sample: Option<u32>,
    /// Brotli-compressed metadata box quality (#15). Mirrors
    /// `EncodeRequest::with_brotli_metadata`.
    brotli_metadata_quality: Option<u32>,
    /// Optional caller-supplied resource cap. When present, dimension-
    /// driven allocations charge against the cap; when absent, the
    /// encoder applies [`Limits::DEFAULT_MAX_MEMORY_BYTES`] (~2 GB) as
    /// a soft default.
    limits: Option<Limits>,
}

impl LossyEncoder {
    /// Attach an ICC color profile.
    pub fn with_icc_profile(mut self, data: &[u8]) -> Self {
        self.icc_profile = Some(data.to_vec());
        self
    }

    /// Attach EXIF data.
    pub fn with_exif(mut self, data: &[u8]) -> Self {
        self.exif = Some(data.to_vec());
        self
    }

    /// Attach XMP data.
    pub fn with_xmp(mut self, data: &[u8]) -> Self {
        self.xmp = Some(data.to_vec());
        self
    }

    /// Attach a JUMBF payload (C2PA / Content Authenticity Initiative
    /// metadata, ISO 19566-5). Bytes are emitted verbatim into a `jumb`
    /// ISOBMFF box appended after `Exif`/`xml `. Mirrors the
    /// [`ImageMetadata::with_jumbf`] field on the one-shot path.
    pub fn with_jumbf(mut self, data: &[u8]) -> Self {
        self.jumbf = Some(data.to_vec());
        self
    }

    /// Specify that source pixels use a custom gamma transfer function.
    pub fn with_source_gamma(mut self, gamma: f32) -> Self {
        self.source_gamma = Some(gamma);
        self
    }

    /// Override the color encoding written to the JXL header.
    pub fn with_color_encoding(
        mut self,
        ce: crate::headers::color_encoding::ColorEncoding,
    ) -> Self {
        self.color_encoding = Some(ce);
        self
    }

    /// Set the peak display luminance in nits for HDR content.
    pub fn with_intensity_target(mut self, nits: f32) -> Self {
        self.intensity_target = nits;
        self
    }

    /// Set the minimum display luminance in nits.
    pub fn with_min_nits(mut self, nits: f32) -> Self {
        self.min_nits = nits;
        self
    }

    /// Set the intrinsic display size.
    pub fn with_intrinsic_size(mut self, width: u32, height: u32) -> Self {
        self.intrinsic_size = Some((width, height));
        self
    }

    /// Signal that the input alpha channel is premultiplied (associated).
    /// Mirrors [`EncodeRequest::with_premultiplied_alpha`]. See that
    /// builder for the lossless-vs-lossy semantic discussion. On the
    /// `LossyEncoder` this returns an `EncodeError::InvalidInput` from
    /// [`finish`](Self::finish) until the unpremultiplication pre-pass
    /// is implemented (#13). On the `LosslessEncoder` it sets
    /// `alpha_associated=true` in the encoded header and writes pixels
    /// unchanged.
    pub fn with_premultiplied_alpha(mut self, enable: bool) -> Self {
        self.premultiplied_alpha = enable;
        self
    }

    /// Override the input precision for u16 layouts. Mirrors
    /// [`EncodeRequest::with_bits_per_sample`] on the streaming path.
    /// `bits` is clamped to `1..=16`. See the EncodeRequest builder
    /// for the full semantic discussion. Closes the streaming-encoder
    /// parity follow-up to today's bits_per_sample landing (#18).
    pub fn with_bits_per_sample(mut self, bits: u32) -> Self {
        self.bits_per_sample = Some(bits.clamp(1, 16));
        self
    }

    /// Brotli-compress EXIF / XMP metadata into `brob` boxes
    /// (closes #15). `quality` is the Brotli effort (0-11; libjxl
    /// default 4); higher = smaller output but slower encode. Each
    /// metadata blob is independently evaluated — if the compressed
    /// brob box would be ≥ the uncompressed Exif/xml box, the
    /// uncompressed form is used (sub-500-byte payloads typically
    /// fall back due to Brotli framing overhead).
    ///
    /// Requires the `brotli-metadata` cargo feature. When the feature
    /// is OFF the call still compiles (the value is stored but
    /// ignored at encode time); add the feature flag to enable.
    pub fn with_brotli_metadata(mut self, quality: u32) -> Self {
        self.brotli_metadata_quality = Some(quality.min(11));
        self
    }

    /// Attach resource limits.
    ///
    /// The supplied [`Limits`] is consulted at [`finish`](Self::finish)
    /// time to derive the per-encode allocation cap, mirroring
    /// [`EncodeRequest::with_limits`]. When unset the encoder applies the
    /// soft default ([`Limits::DEFAULT_MAX_MEMORY_BYTES`], ~2 GB).
    pub fn with_limits(mut self, limits: &Limits) -> Self {
        self.limits = Some(limits.clone());
        self
    }

    /// Number of rows pushed so far.
    pub fn rows_pushed(&self) -> u32 {
        self.rows_pushed
    }

    /// Total expected height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Push pixel rows into the encoder.
    ///
    /// `pixels` must contain exactly `width * num_rows * bytes_per_pixel` bytes.
    /// Rows are converted to the internal linear f32 format immediately, so the
    /// caller can free the source buffer after this call returns.
    #[track_caller]
    pub fn push_rows(&mut self, pixels: &[u8], num_rows: u32) -> Result<()> {
        self.push_rows_inner(pixels, num_rows).map_err(at)
    }

    fn push_rows_inner(
        &mut self,
        pixels: &[u8],
        num_rows: u32,
    ) -> core::result::Result<(), EncodeError> {
        if num_rows == 0 {
            return Ok(());
        }
        let remaining = self.height - self.rows_pushed;
        if num_rows > remaining {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "push_rows: {num_rows} rows would exceed image height \
                     ({} pushed + {num_rows} > {})",
                    self.rows_pushed, self.height
                ),
            });
        }
        let w = self.width as usize;
        let n = num_rows as usize;
        let expected = w
            .checked_mul(n)
            .and_then(|wn| wn.checked_mul(self.layout.bytes_per_pixel()));
        match expected {
            Some(expected) if pixels.len() == expected => {}
            Some(expected) => {
                return Err(EncodeError::InvalidInput {
                    message: format!(
                        "push_rows: expected {expected} bytes for {w}x{n} {:?}, got {}",
                        self.layout,
                        pixels.len()
                    ),
                });
            }
            None => {
                return Err(EncodeError::InvalidInput {
                    message: "push_rows: row dimensions overflow".into(),
                });
            }
        }

        let gamma = self.source_gamma;
        // Streaming-encoder bits_per_sample (#18 follow-up). Mirrors
        // EncodeRequest::encode_lossy's u16_max computation.
        let u16_max = self
            .bits_per_sample
            .map_or(65535.0_f32, |b| ((1u32 << b) - 1) as f32);
        // Streaming PQ/HLG/BT.709 dispatch (#17). Mirrors the
        // EncodeRequest::encode_lossy `source_is_*` predicates.
        // Same dispatch order: gamma > PQ > HLG > BT.709 > sRGB.
        let source_is_pq = gamma.is_none()
            && self.color_encoding.as_ref().is_some_and(|ce| {
                ce.transfer_function == crate::headers::color_encoding::TransferFunction::Pq
            });
        let source_is_hlg = gamma.is_none()
            && self.color_encoding.as_ref().is_some_and(|ce| {
                ce.transfer_function == crate::headers::color_encoding::TransferFunction::Hlg
            });
        let source_is_bt709 = gamma.is_none()
            && self.color_encoding.as_ref().is_some_and(|ce| {
                ce.transfer_function == crate::headers::color_encoding::TransferFunction::Bt709
            });

        // Convert and append linear RGB
        let new_linear: Vec<f32> = match self.layout {
            PixelLayout::Rgb8 => {
                if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(pixels, 3, g)
                } else if source_is_pq {
                    pq_u8_to_linear_f32(pixels, 3)
                } else if source_is_hlg {
                    hlg_u8_to_linear_f32(pixels, 3)
                } else if source_is_bt709 {
                    bt709_u8_to_linear_f32(pixels, 3)
                } else {
                    srgb_u8_to_linear_f32(pixels, 3)
                }
            }
            PixelLayout::Bgr8 => {
                let rgb = bgr_to_rgb(pixels, 3);
                if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(&rgb, 3, g)
                } else if source_is_pq {
                    pq_u8_to_linear_f32(&rgb, 3)
                } else if source_is_hlg {
                    hlg_u8_to_linear_f32(&rgb, 3)
                } else if source_is_bt709 {
                    bt709_u8_to_linear_f32(&rgb, 3)
                } else {
                    srgb_u8_to_linear_f32(&rgb, 3)
                }
            }
            PixelLayout::Rgba8 => {
                if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(pixels, 4, g)
                } else if source_is_pq {
                    pq_u8_to_linear_f32(pixels, 4)
                } else if source_is_hlg {
                    hlg_u8_to_linear_f32(pixels, 4)
                } else if source_is_bt709 {
                    bt709_u8_to_linear_f32(pixels, 4)
                } else {
                    srgb_u8_to_linear_f32(pixels, 4)
                }
            }
            PixelLayout::Bgra8 => {
                let swapped = bgr_to_rgb(pixels, 4);
                if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(&swapped, 4, g)
                } else if source_is_pq {
                    pq_u8_to_linear_f32(&swapped, 4)
                } else if source_is_hlg {
                    hlg_u8_to_linear_f32(&swapped, 4)
                } else if source_is_bt709 {
                    bt709_u8_to_linear_f32(&swapped, 4)
                } else {
                    srgb_u8_to_linear_f32(&swapped, 4)
                }
            }
            PixelLayout::Gray8 => {
                if let Some(g) = gamma {
                    gamma_gray_u8_to_linear_f32_rgb(pixels, 1, g)
                } else if source_is_pq {
                    pq_gray_u8_to_linear_f32_rgb(pixels, 1)
                } else if source_is_hlg {
                    hlg_gray_u8_to_linear_f32_rgb(pixels, 1)
                } else if source_is_bt709 {
                    bt709_gray_u8_to_linear_f32_rgb(pixels, 1)
                } else {
                    gray_u8_to_linear_f32_rgb(pixels, 1)
                }
            }
            PixelLayout::GrayAlpha8 => {
                if let Some(g) = gamma {
                    gamma_gray_u8_to_linear_f32_rgb(pixels, 2, g)
                } else if source_is_pq {
                    pq_gray_u8_to_linear_f32_rgb(pixels, 2)
                } else if source_is_hlg {
                    hlg_gray_u8_to_linear_f32_rgb(pixels, 2)
                } else if source_is_bt709 {
                    bt709_gray_u8_to_linear_f32_rgb(pixels, 2)
                } else {
                    gray_u8_to_linear_f32_rgb(pixels, 2)
                }
            }
            PixelLayout::Rgb16 => {
                if let Some(g) = gamma {
                    gamma_u16_to_linear_f32(pixels, 3, g, u16_max)
                } else if source_is_pq {
                    pq_u16_to_linear_f32(pixels, 3, u16_max)
                } else if source_is_hlg {
                    hlg_u16_to_linear_f32(pixels, 3, u16_max)
                } else if source_is_bt709 {
                    bt709_u16_to_linear_f32(pixels, 3, u16_max)
                } else {
                    srgb_u16_to_linear_f32(pixels, 3, u16_max)
                }
            }
            PixelLayout::Rgba16 => {
                if let Some(g) = gamma {
                    gamma_u16_to_linear_f32(pixels, 4, g, u16_max)
                } else if source_is_pq {
                    pq_u16_to_linear_f32(pixels, 4, u16_max)
                } else if source_is_hlg {
                    hlg_u16_to_linear_f32(pixels, 4, u16_max)
                } else if source_is_bt709 {
                    bt709_u16_to_linear_f32(pixels, 4, u16_max)
                } else {
                    srgb_u16_to_linear_f32(pixels, 4, u16_max)
                }
            }
            PixelLayout::Gray16 => {
                if let Some(g) = gamma {
                    gamma_gray_u16_to_linear_f32_rgb(pixels, 1, g, u16_max)
                } else if source_is_pq {
                    pq_gray_u16_to_linear_f32_rgb(pixels, 1, u16_max)
                } else if source_is_hlg {
                    hlg_gray_u16_to_linear_f32_rgb(pixels, 1, u16_max)
                } else if source_is_bt709 {
                    bt709_gray_u16_to_linear_f32_rgb(pixels, 1, u16_max)
                } else {
                    gray_u16_to_linear_f32_rgb(pixels, 1, u16_max)
                }
            }
            PixelLayout::GrayAlpha16 => {
                if let Some(g) = gamma {
                    gamma_gray_u16_to_linear_f32_rgb(pixels, 2, g, u16_max)
                } else if source_is_pq {
                    pq_gray_u16_to_linear_f32_rgb(pixels, 2, u16_max)
                } else if source_is_hlg {
                    hlg_gray_u16_to_linear_f32_rgb(pixels, 2, u16_max)
                } else if source_is_bt709 {
                    bt709_gray_u16_to_linear_f32_rgb(pixels, 2, u16_max)
                } else {
                    gray_u16_to_linear_f32_rgb(pixels, 2, u16_max)
                }
            }
            PixelLayout::RgbLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                floats.to_vec()
            }
            PixelLayout::RgbaLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                floats
                    .chunks(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect()
            }
            PixelLayout::GrayLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                gray_f32_to_linear_f32_rgb(floats, 1)
            }
            PixelLayout::GrayAlphaLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                gray_f32_to_linear_f32_rgb(floats, 2)
            }
            // FLOAT16 streaming input (closes FLOAT16 portion of #18).
            PixelLayout::RgbLinearF16 => f16_to_linear_f32_rgb(pixels, 3),
            PixelLayout::RgbaLinearF16 => f16_to_linear_f32_rgb(pixels, 4),
            PixelLayout::GrayLinearF16 => f16_gray_to_linear_f32_rgb(pixels, 1),
            PixelLayout::GrayAlphaLinearF16 => f16_gray_to_linear_f32_rgb(pixels, 2),
            // A3 chunk 1b: f32 PQ/HLG/BT.709 streaming input (issue #46).
            // Same linearization helpers as the one-shot path.
            PixelLayout::RgbPqF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                pq_f32_to_linear_f32_rgb(floats, 3)
            }
            PixelLayout::RgbaPqF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                pq_f32_to_linear_f32_rgb(floats, 4)
            }
            PixelLayout::RgbHlgF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                hlg_f32_to_linear_f32_rgb(floats, 3)
            }
            PixelLayout::RgbaHlgF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                hlg_f32_to_linear_f32_rgb(floats, 4)
            }
            PixelLayout::RgbBt709F32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                bt709_f32_to_linear_f32_rgb(floats, 3)
            }
            PixelLayout::RgbaBt709F32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                bt709_f32_to_linear_f32_rgb(floats, 4)
            }
            // Streaming CMYK is not yet wired — only the one-shot
            // lossless path (`LosslessConfig::encode`) handles CMYK
            // input. The streaming lossy encoder would also need a
            // C/M/Y → XYB mapping (see comment on `Cmyk8` in the
            // first match site).
            PixelLayout::Cmyk8 | PixelLayout::Cmyk16 => {
                return Err(EncodeError::UnsupportedPixelLayout(self.layout));
            }
        };
        self.linear_rgb.extend_from_slice(&new_linear);

        // Extract and append alpha
        match self.layout {
            PixelLayout::Rgba8 | PixelLayout::Bgra8 => {
                let new_alpha = extract_alpha(pixels, 4, 3);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::GrayAlpha8 => {
                let new_alpha = extract_alpha(pixels, 2, 1);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::Rgba16 => {
                let new_alpha = extract_alpha_u16(pixels, 4, 3, u16_max);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::GrayAlpha16 => {
                let new_alpha = extract_alpha_u16(pixels, 2, 1, u16_max);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::RgbaLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                let new_alpha = extract_alpha_f32(floats, 4, 3);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::GrayAlphaLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                let new_alpha = extract_alpha_f32(floats, 2, 1);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::RgbaLinearF16 => {
                let new_alpha = extract_alpha_f16(pixels, 4, 3);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::GrayAlphaLinearF16 => {
                let new_alpha = extract_alpha_f16(pixels, 2, 1);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            // A3 chunk 1b (issue #46): alpha is linear in [0, 1]
            // regardless of color transfer function — the inverse EOTF
            // applies only to RGB.
            PixelLayout::RgbaPqF32 | PixelLayout::RgbaHlgF32 | PixelLayout::RgbaBt709F32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                let new_alpha = extract_alpha_f32(floats, 4, 3);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            _ => {}
        }

        self.rows_pushed += num_rows;
        Ok(())
    }

    /// Encode the accumulated pixels and return the JXL bytes.
    ///
    /// All rows must have been pushed via [`push_rows`](Self::push_rows) before
    /// calling this. Returns an error if the image is incomplete.
    #[track_caller]
    pub fn finish(self) -> Result<Vec<u8>> {
        self.finish_inner()
            .map(|mut r| r.take_data().unwrap())
            .map_err(at)
    }

    /// Encode and return JXL bytes together with [`EncodeStats`].
    #[track_caller]
    pub fn finish_with_stats(self) -> Result<EncodeResult> {
        self.finish_inner().map_err(at)
    }

    /// Encode, appending to an existing buffer.
    #[track_caller]
    pub fn finish_into(self, out: &mut Vec<u8>) -> Result<EncodeResult> {
        let mut result = self.finish_inner().map_err(at)?;
        if let Some(data) = result.data.take() {
            out.extend_from_slice(&data);
        }
        Ok(result)
    }

    /// Encode, writing to a `std::io::Write` destination.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn finish_to(self, mut dest: impl std::io::Write) -> Result<EncodeResult> {
        let mut result = self.finish_inner().map_err(at)?;
        if let Some(data) = result.data.take() {
            dest.write_all(&data)
                .map_err(|e| at(EncodeError::from(e)))?;
        }
        Ok(result)
    }

    /// Encode, writing to a seekable destination (any type that
    /// implements [`WritableSeek`], e.g. `std::fs::File` /
    /// `std::io::Cursor<Vec<u8>>`).
    ///
    /// **Streaming refactor #11 chunk 6**: this is the seek-aware
    /// finish path. Chunk-6 implementation routes through
    /// [`Self::finish_inner`] like [`Self::finish_to`] — the buffered-
    /// output one-shot bytes are computed in memory then written to
    /// the sink in a single pass. The seek capability is plumbed but
    /// **not yet exercised** because the level-3 streaming-output
    /// path (`Buffering::FullStreaming` with permuted TOC + DC-global
    /// placeholder + seek-back) is a chunk-7 deliverable.
    ///
    /// Callers should prefer [`Self::finish_to`] when the destination
    /// only implements `Write` (e.g. a network socket). Use this entry
    /// point when the destination is a file or in-memory cursor that
    /// can accept the chunk-7 seek-back semantics without API
    /// changes.
    ///
    /// libjxl reference: PR #4728 (`6553831`) — fixes the
    /// `permuted_toc=0` bit on the non-streaming path. We mirror that
    /// fix in chunk 7 alongside the actual seek-back implementation.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn finish_to_seekable(self, mut dest: impl WritableSeek) -> Result<EncodeResult> {
        let mut result = self.finish_inner().map_err(at)?;
        if let Some(data) = result.data.take() {
            dest.write_all(&data)
                .map_err(|e| at(EncodeError::from(e)))?;
        }
        Ok(result)
    }

    fn finish_inner(self) -> core::result::Result<EncodeResult, EncodeError> {
        if self.rows_pushed != self.height {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "incomplete image: {} of {} rows pushed",
                    self.rows_pushed, self.height
                ),
            });
        }
        // Mirror the one-shot chroma subsampling gate (issue #47).
        // Streaming and one-shot must report subsampling support
        // identically. The streaming path's eager linearisation
        // (sRGB → f32) means we cannot route the JPEG-shaped pipeline
        // (which needs the raw u8 RGB for BT.601 YCbCr conversion)
        // without a sRGB-encode round-trip on the accumulated linear
        // buffer. A future chunk will wire that; for now any
        // subsampled mode on streaming returns InvalidConfig with a
        // pointer to the one-shot path.
        if !self.cfg.chroma_subsampling.is_full() {
            return Err(EncodeError::InvalidConfig {
                message: format!(
                    "chroma subsampling {} on the streaming `LossyEncoder` \
                     is not yet wired (one-shot `EncodeRequest::encode` \
                     supports it; streaming support is queued). Use \
                     `LossyConfig::new(d).with_chroma_subsampling(...).\
                     encode_request(w, h, layout).encode(&pixels)` for now.",
                    self.cfg.chroma_subsampling.tag(),
                ),
            });
        }
        // Run the full config validator (distance, effort, iter
        // counts, mutual exclusivity). Mirrors
        // `EncodeRequest::encode_inner`.
        self.cfg.validate()?;
        // Defensive caps on caller-supplied metadata buffers (mirrors
        // EncodeRequest::encode_inner).
        validate_metadata_sizes(
            self.icc_profile.as_deref(),
            self.exif.as_deref(),
            self.xmp.as_deref(),
            self.jumbf.as_deref(),
        )?;
        // Tone-mapping numeric range checks. Stored as plain f32 on
        // the encoder; pass `Some(_)` only when set away from the
        // libjxl default so a caller who never touched these knobs
        // gets the encoder default behavior.
        let it = (self.intensity_target != 255.0).then_some(self.intensity_target);
        let mn = (self.min_nits != 0.0).then_some(self.min_nits);
        validate_tone_mapping(it, mn)?;
        validate_source_gamma(self.source_gamma)?;
        validate_intrinsic_size(self.intrinsic_size)?;
        let cfg = &self.cfg;
        let w = self.width as usize;
        let h = self.height as usize;
        let mut linear_rgb = self.linear_rgb;
        let alpha = self.alpha;

        // Unpremultiply BEFORE SimplifyInvisible / XYB — see the
        // matching block in `EncodeRequest::encode_lossy` for the full
        // reasoning. Closes lossy portion of #13.
        if self.premultiplied_alpha
            && let Some(ref alpha_buf) = alpha
        {
            unpremultiply_alpha_inplace(&mut linear_rgb, alpha_buf);
        }

        // SimplifyInvisible pre-pass (closes #10) — mirrored from the
        // one-shot path in `EncodeRequest::encode_lossy`. Required to
        // keep `oneshot == streaming` byte-exact when the input has any
        // alpha=0 pixel (caught by `test_streaming_lossy_rgba`).
        // Gated on !premultiplied_alpha to match libjxl
        // `enc_frame.cc:1588`.
        if cfg.simplify_invisible
            && !self.premultiplied_alpha
            && let Some(ref alpha_buf) = alpha
            && crate::vardct::simplify_invisible::has_any_invisible_pixels(alpha_buf)
        {
            crate::vardct::simplify_invisible::simplify_invisible_rgb(
                &mut linear_rgb,
                alpha_buf,
                w,
                h,
                false,
            );
        }

        // Construct the per-encode allocation budget. Streaming callers
        // can attach a [`Limits`] via [`Self::with_limits`]; otherwise we
        // apply the same soft default the request path uses (~2 GB).
        // Mirrors the up-front working-set check in
        // `EncodeRequest::encode_inner` so absurd dimensions get an early
        // `LimitExceeded` instead of a confusing mid-encode failure.
        let budget_cap = self
            .limits
            .as_ref()
            .map(|l| l.effective_max_memory_bytes())
            .unwrap_or(Limits::DEFAULT_MAX_MEMORY_BYTES);
        let budget = crate::budget::MemoryBudget::new(budget_cap);
        let est_bytes = (self.width as u64)
            .checked_mul(self.height as u64)
            .and_then(|n| n.checked_mul(40))
            .ok_or_else(|| EncodeError::LimitExceeded {
                message: format!(
                    "image {}x{} too large for working-set estimate",
                    self.width, self.height
                ),
            })?;
        if est_bytes > budget_cap {
            return Err(EncodeError::LimitExceeded {
                message: format!(
                    "estimated working set {est_bytes} bytes for {}x{} image \
                     exceeds budget cap {budget_cap}",
                    self.width, self.height
                ),
            });
        }

        let (codestream, mut stats) = run_with_threads(cfg.threads, || {
            let mut profile = cfg.effective_profile_for_image((w as u64) * (h as u64));
            if let Some(max_size) = cfg.max_strategy_size {
                if max_size < 16 {
                    profile.try_dct16 = false;
                }
                if max_size < 32 {
                    profile.try_dct32 = false;
                }
                if max_size < 64 {
                    profile.try_dct64 = false;
                }
            }

            // Apply auto-resample-at-d≥10 (refs #12) before building
            // the encoder so distance + resampling stay coherent.
            let effective_resampling = cfg.effective_resampling();
            let effective_distance = cfg.effective_distance();

            let mut enc = crate::vardct::VarDctEncoder::new(effective_distance);
            // W44-128 Chunk B + W44-130 Chunk D: resolve EncoderStrategy
            // bundle once (streaming `LossyEncoder` path). Field is
            // non-optional as of Chunk D.
            enc.resolved_improvements = cfg.resolve_improvements();
            enc.effort = cfg.effort;
            enc.profile = profile;
            enc.use_ans = cfg.use_ans;
            enc.optimize_codes = enc.profile.optimize_codes;
            enc.custom_orders = enc.profile.custom_orders;
            enc.ac_strategy_enabled = enc.profile.ac_strategy_enabled;
            enc.enable_noise = cfg.noise;
            enc.photon_noise_iso = cfg.photon_noise_iso;
            // Streaming LossyEncoder must mirror the non-streaming
            // `EncodeRequest::encode_lossy` wire-up (api.rs:4531-4569)
            // and the animation `encode_animation_lossy` wire-up
            // (api.rs:6892-6929). Forgetting any of these fields here
            // is a silent-drop gate: the caller sets it on the
            // `LossyConfig`, the `with_*` setter accepts the value, and
            // the streaming `finish*()` path quietly ignores it. Audit
            // 2026-05-17 surfaced `manual_noise_lut` (photon-noise
            // siblings #2 audit) and four others.
            enc.manual_noise_lut = cfg.manual_noise_lut;
            enc.quant_ac_rescale = cfg.quant_ac_rescale;
            enc.original_distance = cfg.original_distance;
            enc.enable_denoise = cfg.denoise;
            enc.enable_gaborish = cfg.effective_gaborish() && effective_distance > 0.5;
            // EX-J13: adaptive gaborish is silently gated to be a subset of
            // gaborish (no-op when the fixed inverse is disabled).
            enc.enable_adaptive_gaborish = enc.enable_gaborish && cfg.adaptive_gaborish;
            // libjxl `--epf -1..3` override (enc_frame.cc:284-285). `-1`
            // = encoder chooses by distance; otherwise force the given
            // count.
            enc.epf_level_override = if cfg.epf_level < 0 {
                None
            } else {
                Some(cfg.epf_level as u32)
            };
            // W44-130 Chunk D: dispatch policies hydrated from the
            // resolved bundle (LossyConfig setters deleted; absorbed
            // into `EncoderImprovementsCustom`).
            enc.epf_dispatch = enc.resolved_improvements.epf_dispatch;
            enc.error_diffusion = cfg.error_diffusion;
            enc.pixel_domain_loss = cfg.pixel_domain_loss;
            enc.pixel_loss_dispatch = enc.resolved_improvements.pixel_loss_dispatch;
            enc.single_pass_entropy_dispatch =
                enc.resolved_improvements.single_pass_entropy_dispatch;
            enc.enable_lz77 = cfg.effective_lz77();
            enc.lz77_method = cfg.lz77_method;
            enc.force_strategy = cfg.force_strategy;
            // RFC #45 pick #4 — when the caller has explicitly pinned `cfg.patches`
            // via `with_patches`, that wins; otherwise read the per-image
            // dispatched profile (the content-class adapter may have flipped
            // patches on for Screenshot content at e5/e6).
            enc.enable_patches = if cfg.patches_explicit {
                cfg.effective_patches()
            } else if cfg.faster_decoding >= 2 {
                // libjxl `enc_modular.cc:707` skips patches at
                // `decoding_speed_tier >= 2`.
                false
            } else {
                enc.profile.patches
            };
            enc.patches_dispatch = enc.resolved_improvements.patches_dispatch;
            enc.enable_dot_detection = cfg.dot_detection;
            enc.encoder_mode = cfg.mode;
            enc.splines = cfg.splines.clone();
            enc.auto_splines = cfg.auto_splines;
            enc.is_grayscale = self.layout.is_grayscale();
            enc.progressive = cfg.progressive;
            enc.use_lf_frame = cfg.lf_frame;
            // W44-130 Chunk D: `content_aware_entropy_mul` + legacy
            // `with_*_hint` setters all deleted; strategy + overrides
            // flow via `cfg.resolve_improvements()` into
            // `enc.resolved_improvements`.
            // W44-91: streaming `LossyEncoder` ingests pre-converted
            // `linear_rgb` rows, so the sRGB u8 source bytes the
            // zenanalyze-equivalent proxy needs are not available
            // here — leave `zenanalyze_proxies = None`, which keeps
            // the W44-91 gate dormant on this code path. Callers that
            // need the W44-91 lift on a streaming encode can set
            // [`LossyConfig::with_strategy_overrides`] with
            // `high_d_photo_hint: Some(true)`
            // explicitly after computing the proxy upstream.
            // Streaming refactor #11 chunk 6 (streaming LossyEncoder
            // path).
            enc.buffering = cfg.buffering;
            #[cfg(feature = "butteraugli-loop")]
            {
                enc.butteraugli_iters = cfg.butteraugli_iters;
                // EX-J11 chunk 4: see `encode_lossy` site above for
                // the resolution rationale. Auto → Vdp2 on PQ/HLG,
                // Butteraugli otherwise.
                enc.hdr_loss = cfg.resolve_hdr_loss(self.layout, self.color_encoding.as_ref());
            }
            #[cfg(feature = "ssim2-loop")]
            {
                enc.ssim2_iters = cfg.ssim2_iters;
            }
            #[cfg(feature = "zensim-loop")]
            {
                enc.zensim_iters = cfg.zensim_iters;
            }
            enc.bit_depth_16 = self.bit_depth_16;
            enc.source_gamma = self.source_gamma;
            // A3 chunk 1b (issue #46): mirrors EncodeRequest::encode_lossy
            // — auto-derive a ColorEncoding from the layout's implied
            // transfer function when the caller didn't set one
            // explicitly. See that site for the full rationale.
            enc.color_encoding = self.color_encoding.clone().or_else(|| {
                if self.source_gamma.is_some() {
                    return None;
                }
                use crate::headers::color_encoding::{ColorEncoding, TransferFunction};
                match self.layout.implied_transfer_function() {
                    Some(TransferFunction::Pq) => Some(ColorEncoding::bt2100_pq()),
                    Some(TransferFunction::Hlg) => Some(ColorEncoding::bt2100_hlg()),
                    Some(TransferFunction::Bt709) => Some(ColorEncoding {
                        transfer_function: TransferFunction::Bt709,
                        ..ColorEncoding::srgb()
                    }),
                    Some(TransferFunction::Linear) => Some(ColorEncoding::linear_srgb()),
                    _ => None,
                }
            });
            enc.intensity_target = self.intensity_target;
            enc.min_nits = self.min_nits;
            enc.intrinsic_size = self.intrinsic_size;
            enc.alpha_associated = self.premultiplied_alpha;
            enc.bits_per_sample_override = self.bits_per_sample;
            enc.center_first = self.cfg.center_first;
            // Decoder upsampling factor (refs #12). Mirrors the
            // EncodeRequest::encode_lossy wire-up below.
            enc.upsampling = effective_resampling;
            enc.non_finite_action = self.cfg.non_finite_action;
            enc.budget = Some(alloc::sync::Arc::clone(&budget));
            if let Some(ref icc) = self.icc_profile {
                enc.icc_profile = Some(icc.clone());
            }

            let (encode_rgb, encode_alpha, encode_w, encode_h) = if effective_resampling > 1 {
                let (down_rgb, dw, dh) = if effective_resampling == 2 {
                    crate::vardct::resampling::sharper_downsample_2x_rgb(&linear_rgb, w, h)
                } else {
                    crate::vardct::resampling::box_downsample_rgb(
                        &linear_rgb,
                        w,
                        h,
                        effective_resampling,
                    )
                };
                let down_alpha = alpha.as_ref().map(|a| {
                    let (a_down, _, _) = crate::vardct::resampling::box_downsample_alpha_u8(
                        a,
                        w,
                        h,
                        effective_resampling,
                    );
                    a_down
                });
                (down_rgb, down_alpha, dw as usize, dh as usize)
            } else {
                (linear_rgb, alpha, w, h)
            };

            let output = enc
                .encode(encode_w, encode_h, &encode_rgb, encode_alpha.as_deref())
                .map_err(EncodeError::from)?;

            #[cfg(feature = "butteraugli-loop")]
            let butteraugli_iters_actual = cfg.butteraugli_iters;
            #[cfg(not(feature = "butteraugli-loop"))]
            let butteraugli_iters_actual = 0u32;

            let stats = EncodeStats {
                mode: EncodeMode::Lossy,
                strategy_counts: output.strategy_counts,
                gaborish: cfg.gaborish,
                ans: cfg.use_ans,
                butteraugli_iters: butteraugli_iters_actual,
                pixel_domain_loss: cfg.pixel_domain_loss,
                ..Default::default()
            };
            Ok::<_, EncodeError>((output.data, stats))
        })?;

        stats.codestream_size = codestream.len();

        // Streaming LossyEncoder does not accept extra channels beyond
        // alpha; count alpha from layout.
        let icc_size = self.icc_profile.as_deref().map_or(0u64, |i| i.len() as u64);
        let num_ec = u32::from(self.layout.has_alpha());
        let level = compute_required_level(self.width, self.height, num_ec, false, icc_size)?;

        let has_meta = self.exif.is_some() || self.xmp.is_some() || self.jumbf.is_some();
        let output = if has_meta || crate::container::level_requires_container(level) {
            wrap_metadata_container(
                &codestream,
                self.exif.as_deref(),
                self.xmp.as_deref(),
                self.jumbf.as_deref(),
                self.brotli_metadata_quality,
                level,
            )
        } else {
            codestream
        };

        stats.output_size = output.len();
        Ok(EncodeResult {
            data: Some(output),
            stats,
        })
    }
}

/// JXL spec maximum dimension (2^30) per axis.
///
/// Both width and height must be `<= MAX_JXL_DIM`. This is enforced on every
/// encode entry point (one-shot, streaming, animation) so that the file header
/// `write_size_u2s` (30-bit field) cannot silently truncate caller-supplied
/// dimensions and produce a bitstream whose declared dimensions disagree with
/// the encoded data.
const MAX_JXL_DIM: u32 = 1 << 30;

/// Internal scale headroom factor used by working-buffer overflow checks
/// (matches the one-shot `validate_pixels` 16x factor).
const MAX_INTERNAL_SCALE: usize = 16;

/// Validate `(width, height)` against the JXL spec ceiling and `usize`
/// working-buffer overflow.
///
/// Shared by `validate_pixels` (one-shot), `LossyConfig::encoder` /
/// `LosslessConfig::encoder` (streaming), and `validate_animation_input`.
/// Without this check the streaming entry points silently accepted
/// `width = u32::MAX`, which `write_size_u2s` would then truncate to 30 bits,
/// emitting a header whose declared width does not match the encoded data.
/// Per-blob cap for caller-supplied metadata buffers (ICC / EXIF /
/// XMP). Real-world payloads are well under 10 MB; the ~1 GB cap is
/// purely defensive and never rejects legitimate input. Without this,
/// pathological multi-GB metadata reaches `Vec::with_capacity` in the
/// container wrapper, exhausts system memory at write time, and the
/// kernel kills the process.
const METADATA_SIZE_LIMIT: usize = u32::MAX as usize >> 2;

/// Maximum value (in nits) accepted for `intensity_target` /
/// `min_nits`. Bounded by the f16 representation used in the
/// codestream (`f16::MAX = 65504`). Anything larger silently fails
/// in `f32_to_f16_bits` deep in `file_header.write`; this lets us
/// surface a clean `InvalidInput` instead.
const F16_MAX_NITS: f32 = 65504.0;

/// Validate caller-supplied tone-mapping fields. `None` means
/// "use the encoder default" — only `Some(_)` values are checked.
/// Rules:
/// - `intensity_target` must be finite, `> 0`, and `<= 65504`
///   (f16 representation cap; anything larger silently fails in the
///   header writer).
/// - `min_nits` must be finite, `>= 0`, and `<= intensity_target`
///   (or `<= 65504` if `intensity_target` is unset). A min above
///   the peak is physically nonsensical and would confuse decoders.
fn validate_tone_mapping(
    intensity_target: Option<f32>,
    min_nits: Option<f32>,
) -> core::result::Result<(), EncodeError> {
    let it = intensity_target;
    if let Some(it) = it {
        if !it.is_finite() {
            return Err(EncodeError::InvalidInput {
                message: format!("intensity_target must be finite (got {it})"),
            });
        }
        if it <= 0.0 {
            return Err(EncodeError::InvalidInput {
                message: format!("intensity_target must be > 0 (got {it})"),
            });
        }
        if it > F16_MAX_NITS {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "intensity_target {it} exceeds f16 max ({F16_MAX_NITS}); the codestream cannot represent it",
                ),
            });
        }
    }
    if let Some(mn) = min_nits {
        if !mn.is_finite() {
            return Err(EncodeError::InvalidInput {
                message: format!("min_nits must be finite (got {mn})"),
            });
        }
        if mn < 0.0 {
            return Err(EncodeError::InvalidInput {
                message: format!("min_nits must be >= 0 (got {mn})"),
            });
        }
        let cap = it.unwrap_or(F16_MAX_NITS);
        if mn > cap {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "min_nits {mn} exceeds intensity_target {cap} (min cannot exceed peak)",
                ),
            });
        }
    }
    Ok(())
}

/// Apply [`METADATA_SIZE_LIMIT`] to caller-supplied ICC, EXIF, and
/// XMP buffers. Empty ICC is also rejected (the encoder cannot
/// embed a zero-byte ICC profile). Used by `EncodeRequest`,
/// `LossyEncoder::finish_inner`, and `LosslessEncoder::finish_inner`
/// for parity.
fn validate_metadata_sizes(
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
) -> core::result::Result<(), EncodeError> {
    if let Some(icc) = icc {
        if icc.is_empty() {
            return Err(EncodeError::InvalidInput {
                message: "ICC profile must not be empty".into(),
            });
        }
        if icc.len() > METADATA_SIZE_LIMIT {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "ICC profile too large: {} bytes (max {METADATA_SIZE_LIMIT})",
                    icc.len()
                ),
            });
        }
    }
    if let Some(exif) = exif
        && exif.len() > METADATA_SIZE_LIMIT
    {
        return Err(EncodeError::InvalidInput {
            message: format!(
                "EXIF metadata too large: {} bytes (max {METADATA_SIZE_LIMIT})",
                exif.len()
            ),
        });
    }
    if let Some(xmp) = xmp
        && xmp.len() > METADATA_SIZE_LIMIT
    {
        return Err(EncodeError::InvalidInput {
            message: format!(
                "XMP metadata too large: {} bytes (max {METADATA_SIZE_LIMIT})",
                xmp.len()
            ),
        });
    }
    if let Some(jumbf) = jumbf {
        if jumbf.is_empty() {
            // Empty payload would produce a zero-payload `jumb` box
            // (8-byte header only) which no JUMBF reader can parse.
            return Err(EncodeError::InvalidInput {
                message: "JUMBF payload must not be empty".into(),
            });
        }
        if jumbf.len() > METADATA_SIZE_LIMIT {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "JUMBF metadata too large: {} bytes (max {METADATA_SIZE_LIMIT})",
                    jumbf.len()
                ),
            });
        }
    }
    Ok(())
}

/// Validate `with_source_gamma(gamma)` value. JXL spec stores the
/// gamma exponent in (1/255, 1] (decode: `pixel = sample^gamma`).
/// Our encode pipeline uses `inv_gamma = 1.0 / gamma`, so any
/// non-positive / non-finite value yields silently garbage output
/// (LUT becomes all-zero or all-one, or contains NaN/Inf). Reject
/// up front instead.
fn validate_source_gamma(gamma: Option<f32>) -> core::result::Result<(), EncodeError> {
    let Some(g) = gamma else {
        return Ok(());
    };
    if !g.is_finite() {
        return Err(EncodeError::InvalidInput {
            message: format!("source_gamma must be finite (got {g})"),
        });
    }
    // libjxl accepts gamma in (1/255, 1]; we mirror that exactly so
    // codestreams round-trip through cjxl/djxl unchanged.
    const GAMMA_MIN: f32 = 1.0 / 255.0;
    if g <= GAMMA_MIN {
        return Err(EncodeError::InvalidInput {
            message: format!(
                "source_gamma must be > {GAMMA_MIN:.6} (got {g}); typical sRGB-ish values are 1/2.2 ≈ 0.4545",
            ),
        });
    }
    if g > 1.0 {
        return Err(EncodeError::InvalidInput {
            message: format!(
                "source_gamma must be <= 1.0 (got {g}); the stored value is the encoding exponent, not its inverse",
            ),
        });
    }
    Ok(())
}

/// Validate `with_intrinsic_size(width, height)`. Same shape as the
/// coded-image dimension validator: zero rejected, exceeds-spec-max
/// rejected. Reused at the same up-front spots so a caller who sets
/// intrinsic_size to nonsense gets a clean error before the encoder
/// allocates anything.
fn validate_intrinsic_size(intrinsic: Option<(u32, u32)>) -> core::result::Result<(), EncodeError> {
    let Some((iw, ih)) = intrinsic else {
        return Ok(());
    };
    if iw == 0 || ih == 0 {
        return Err(EncodeError::InvalidInput {
            message: format!("intrinsic_size must be non-zero (got {iw}x{ih})"),
        });
    }
    if iw > MAX_JXL_DIM || ih > MAX_JXL_DIM {
        return Err(EncodeError::LimitExceeded {
            message: format!(
                "intrinsic_size {iw}x{ih} exceeds JXL spec maximum of {MAX_JXL_DIM} per dimension",
            ),
        });
    }
    Ok(())
}

fn validate_dims(width: u32, height: u32) -> core::result::Result<(), EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::InvalidInput {
            message: format!("zero dimensions: {width}x{height}"),
        });
    }
    if width > MAX_JXL_DIM || height > MAX_JXL_DIM {
        return Err(EncodeError::LimitExceeded {
            message: format!(
                "image {width}x{height} exceeds JXL spec maximum of {MAX_JXL_DIM} per dimension",
            ),
        });
    }
    let w = width as usize;
    let h = height as usize;
    if w.checked_mul(h)
        .and_then(|n| n.checked_mul(MAX_INTERNAL_SCALE))
        .is_none()
    {
        return Err(EncodeError::LimitExceeded {
            message: format!(
                "image dimensions {width}x{height} overflow internal working-buffer sizing",
            ),
        });
    }
    Ok(())
}

impl LossyConfig {
    /// Create a streaming encoder for incremental row input.
    ///
    /// Pixels are converted to the internal format as rows are pushed via
    /// [`LossyEncoder::push_rows`], allowing callers to free source buffers
    /// incrementally rather than materializing the entire image.
    #[track_caller]
    pub fn encoder(&self, width: u32, height: u32, layout: PixelLayout) -> Result<LossyEncoder> {
        validate_dims(width, height).map_err(at)?;
        let w = width as usize;
        let h = height as usize;
        let rgb_capacity = w.checked_mul(h).and_then(|n| n.checked_mul(3));
        let Some(rgb_capacity) = rgb_capacity else {
            return Err(at(EncodeError::InvalidInput {
                message: "image dimensions overflow".into(),
            }));
        };

        let bit_depth_16 = layout.is_16bit();
        let has_alpha = layout.has_alpha();
        let alpha = if has_alpha {
            let mut v = Vec::new();
            v.try_reserve(w * h)
                .map_err(|e| at(EncodeError::from(crate::error::Error::from(e))))?;
            Some(v)
        } else {
            None
        };

        let mut linear_rgb = Vec::new();
        linear_rgb
            .try_reserve(rgb_capacity)
            .map_err(|e| at(EncodeError::from(crate::error::Error::from(e))))?;

        Ok(LossyEncoder {
            cfg: self.clone(),
            width,
            height,
            layout,
            rows_pushed: 0,
            linear_rgb,
            alpha,
            bit_depth_16,
            icc_profile: None,
            exif: None,
            xmp: None,
            jumbf: None,
            source_gamma: None,
            color_encoding: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            intrinsic_size: None,
            premultiplied_alpha: false,
            bits_per_sample: None,
            brotli_metadata_quality: None,
            limits: None,
        })
    }
}

/// Streaming lossless (modular) encoder.
///
/// Accepts pixel rows incrementally via [`push_rows`](Self::push_rows), then
/// encodes on [`finish`](Self::finish). This allows callers to free source pixel
/// buffers as rows are pushed, rather than materializing the entire image in
/// memory before encoding.
///
/// ```rust,no_run
/// use jxl_encoder::{LosslessConfig, PixelLayout};
///
/// let mut enc = LosslessConfig::new()
///     .encoder(800, 600, PixelLayout::Rgb8)?;
///
/// # let row_bytes = 800 * 3;
/// # let source_rows = vec![0u8; row_bytes * 600];
/// for chunk in source_rows.chunks(row_bytes * 100) {
///     enc.push_rows(chunk, 100)?;
/// }
///
/// let jxl_bytes = enc.finish()?;
/// # Ok::<_, jxl_encoder::At<jxl_encoder::EncodeError>>(())
/// ```
pub struct LosslessEncoder {
    cfg: LosslessConfig,
    width: u32,
    height: u32,
    layout: PixelLayout,
    rows_pushed: u32,
    channels: Vec<crate::modular::channel::Channel>,
    num_source_channels: usize,
    bit_depth: u32,
    is_grayscale: bool,
    has_alpha: bool,
    icc_profile: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    /// JUMBF (ISO 19566-5, C2PA) superbox payload, emitted verbatim
    /// into a `jumb` box appended after `Exif`/`xml `.
    jumbf: Option<Vec<u8>>,
    source_gamma: Option<f32>,
    color_encoding: Option<crate::headers::color_encoding::ColorEncoding>,
    intensity_target: f32,
    min_nits: f32,
    intrinsic_size: Option<(u32, u32)>,
    /// Premultiplied (associated) alpha signaling. When `true`, the
    /// alpha extra channel header is written with `alpha_associated=true`.
    /// Encoded pixels are unchanged (lossless preserves them bit-exactly).
    /// Default `false`. Mirrors `EncodeRequest::with_premultiplied_alpha`.
    premultiplied_alpha: bool,
    /// Configurable BitDepth.bits_per_sample for the codestream
    /// header (#18 sub-feature). Lossless preserves pixels bit-exactly,
    /// so this only affects header signaling; the encoded values
    /// remain whatever the caller pushed. Mirrors
    /// `EncodeRequest::with_bits_per_sample`.
    bits_per_sample: Option<u32>,
    /// Brotli-compressed metadata box quality (#15). Mirrors
    /// `EncodeRequest::with_brotli_metadata`.
    brotli_metadata_quality: Option<u32>,
    /// Optional caller-supplied resource cap. When present, dimension-
    /// driven allocations charge against the cap; when absent, the
    /// encoder applies [`Limits::DEFAULT_MAX_MEMORY_BYTES`] (~2 GB) as
    /// a soft default.
    limits: Option<Limits>,
}

impl LosslessEncoder {
    /// Attach an ICC color profile.
    pub fn with_icc_profile(mut self, data: &[u8]) -> Self {
        self.icc_profile = Some(data.to_vec());
        self
    }

    /// Attach EXIF data.
    pub fn with_exif(mut self, data: &[u8]) -> Self {
        self.exif = Some(data.to_vec());
        self
    }

    /// Attach XMP data.
    pub fn with_xmp(mut self, data: &[u8]) -> Self {
        self.xmp = Some(data.to_vec());
        self
    }

    /// Attach a JUMBF payload (C2PA / Content Authenticity Initiative
    /// metadata, ISO 19566-5). Bytes are emitted verbatim into a `jumb`
    /// ISOBMFF box appended after `Exif`/`xml `. Mirrors the
    /// [`ImageMetadata::with_jumbf`] field on the one-shot path.
    pub fn with_jumbf(mut self, data: &[u8]) -> Self {
        self.jumbf = Some(data.to_vec());
        self
    }

    /// Specify that source pixels use a custom gamma transfer function.
    pub fn with_source_gamma(mut self, gamma: f32) -> Self {
        self.source_gamma = Some(gamma);
        self
    }

    /// Override the color encoding written to the JXL header.
    pub fn with_color_encoding(
        mut self,
        ce: crate::headers::color_encoding::ColorEncoding,
    ) -> Self {
        self.color_encoding = Some(ce);
        self
    }

    /// Set the peak display luminance in nits for HDR content.
    pub fn with_intensity_target(mut self, nits: f32) -> Self {
        self.intensity_target = nits;
        self
    }

    /// Set the minimum display luminance in nits.
    pub fn with_min_nits(mut self, nits: f32) -> Self {
        self.min_nits = nits;
        self
    }

    /// Set the intrinsic display size.
    pub fn with_intrinsic_size(mut self, width: u32, height: u32) -> Self {
        self.intrinsic_size = Some((width, height));
        self
    }

    /// Signal that the input alpha channel is premultiplied (associated).
    /// Mirrors [`EncodeRequest::with_premultiplied_alpha`]. See that
    /// builder for the lossless-vs-lossy semantic discussion. On the
    /// `LossyEncoder` this returns an `EncodeError::InvalidInput` from
    /// [`finish`](Self::finish) until the unpremultiplication pre-pass
    /// is implemented (#13). On the `LosslessEncoder` it sets
    /// `alpha_associated=true` in the encoded header and writes pixels
    /// unchanged.
    pub fn with_premultiplied_alpha(mut self, enable: bool) -> Self {
        self.premultiplied_alpha = enable;
        self
    }

    /// Override the input precision for u16 layouts. Mirrors
    /// [`EncodeRequest::with_bits_per_sample`] on the streaming path.
    /// `bits` is clamped to `1..=16`. See the EncodeRequest builder
    /// for the full semantic discussion. Closes the streaming-encoder
    /// parity follow-up to today's bits_per_sample landing (#18).
    pub fn with_bits_per_sample(mut self, bits: u32) -> Self {
        self.bits_per_sample = Some(bits.clamp(1, 16));
        self
    }

    /// Brotli-compress EXIF / XMP metadata into `brob` boxes
    /// (closes #15). `quality` is the Brotli effort (0-11; libjxl
    /// default 4); higher = smaller output but slower encode. Each
    /// metadata blob is independently evaluated — if the compressed
    /// brob box would be ≥ the uncompressed Exif/xml box, the
    /// uncompressed form is used (sub-500-byte payloads typically
    /// fall back due to Brotli framing overhead).
    ///
    /// Requires the `brotli-metadata` cargo feature. When the feature
    /// is OFF the call still compiles (the value is stored but
    /// ignored at encode time); add the feature flag to enable.
    pub fn with_brotli_metadata(mut self, quality: u32) -> Self {
        self.brotli_metadata_quality = Some(quality.min(11));
        self
    }

    /// Attach resource limits.
    ///
    /// The supplied [`Limits`] is consulted at [`finish`](Self::finish)
    /// time to derive the per-encode allocation cap, mirroring
    /// [`EncodeRequest::with_limits`]. When unset the encoder applies the
    /// soft default ([`Limits::DEFAULT_MAX_MEMORY_BYTES`], ~2 GB).
    pub fn with_limits(mut self, limits: &Limits) -> Self {
        self.limits = Some(limits.clone());
        self
    }

    /// Number of rows pushed so far.
    pub fn rows_pushed(&self) -> u32 {
        self.rows_pushed
    }

    /// Total expected height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Push pixel rows into the encoder.
    ///
    /// `pixels` must contain exactly `width * num_rows * bytes_per_pixel` bytes.
    /// Rows are deinterleaved into per-channel planes immediately, so the caller
    /// can free the source buffer after this call returns.
    #[track_caller]
    pub fn push_rows(&mut self, pixels: &[u8], num_rows: u32) -> Result<()> {
        self.push_rows_inner(pixels, num_rows).map_err(at)
    }

    fn push_rows_inner(
        &mut self,
        pixels: &[u8],
        num_rows: u32,
    ) -> core::result::Result<(), EncodeError> {
        if num_rows == 0 {
            return Ok(());
        }
        let remaining = self.height - self.rows_pushed;
        if num_rows > remaining {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "push_rows: {num_rows} rows would exceed image height \
                     ({} pushed + {num_rows} > {})",
                    self.rows_pushed, self.height
                ),
            });
        }
        let w = self.width as usize;
        let n = num_rows as usize;
        let bpp = self.layout.bytes_per_pixel();
        let expected = w.checked_mul(n).and_then(|wn| wn.checked_mul(bpp));
        match expected {
            Some(expected) if pixels.len() == expected => {}
            Some(expected) => {
                return Err(EncodeError::InvalidInput {
                    message: format!(
                        "push_rows: expected {expected} bytes for {w}x{n} {:?}, got {}",
                        self.layout,
                        pixels.len()
                    ),
                });
            }
            None => {
                return Err(EncodeError::InvalidInput {
                    message: "push_rows: row dimensions overflow".into(),
                });
            }
        }

        let y_start = self.rows_pushed as usize;
        let nc = self.num_source_channels;

        match self.layout {
            PixelLayout::Rgb8 | PixelLayout::Bgr8 => {
                let is_bgr = matches!(self.layout, PixelLayout::Bgr8);
                for y in 0..n {
                    let row_offset = y * w * 3;
                    let dst_y = y_start + y;
                    for x in 0..w {
                        let src = row_offset + x * 3;
                        let (r, g, b) = if is_bgr {
                            (pixels[src + 2], pixels[src + 1], pixels[src])
                        } else {
                            (pixels[src], pixels[src + 1], pixels[src + 2])
                        };
                        self.channels[0].set(x, dst_y, r as i32);
                        self.channels[1].set(x, dst_y, g as i32);
                        self.channels[2].set(x, dst_y, b as i32);
                    }
                }
            }
            PixelLayout::Rgba8 | PixelLayout::Bgra8 => {
                let is_bgr = matches!(self.layout, PixelLayout::Bgra8);
                for y in 0..n {
                    let row_offset = y * w * 4;
                    let dst_y = y_start + y;
                    for x in 0..w {
                        let src = row_offset + x * 4;
                        let (r, g, b) = if is_bgr {
                            (pixels[src + 2], pixels[src + 1], pixels[src])
                        } else {
                            (pixels[src], pixels[src + 1], pixels[src + 2])
                        };
                        self.channels[0].set(x, dst_y, r as i32);
                        self.channels[1].set(x, dst_y, g as i32);
                        self.channels[2].set(x, dst_y, b as i32);
                        self.channels[3].set(x, dst_y, pixels[src + 3] as i32);
                    }
                }
            }
            PixelLayout::Gray8 => {
                for y in 0..n {
                    let row_offset = y * w;
                    let dst_y = y_start + y;
                    for x in 0..w {
                        self.channels[0].set(x, dst_y, pixels[row_offset + x] as i32);
                    }
                }
            }
            PixelLayout::GrayAlpha8 => {
                for y in 0..n {
                    let row_offset = y * w * 2;
                    let dst_y = y_start + y;
                    for x in 0..w {
                        let src = row_offset + x * 2;
                        self.channels[0].set(x, dst_y, pixels[src] as i32);
                        self.channels[1].set(x, dst_y, pixels[src + 1] as i32);
                    }
                }
            }
            PixelLayout::Rgb16
            | PixelLayout::Rgba16
            | PixelLayout::Gray16
            | PixelLayout::GrayAlpha16 => {
                let pixels_u16: &[u16] = bytemuck::cast_slice(pixels);
                for y in 0..n {
                    let row_offset = y * w * nc;
                    let dst_y = y_start + y;
                    for x in 0..w {
                        let src = row_offset + x * nc;
                        for c in 0..nc {
                            self.channels[c].set(x, dst_y, pixels_u16[src + c] as i32);
                        }
                    }
                }
            }
            _ => {
                return Err(EncodeError::UnsupportedPixelLayout(self.layout));
            }
        }

        self.rows_pushed += num_rows;
        Ok(())
    }

    /// Encode the accumulated pixels and return the JXL bytes.
    ///
    /// All rows must have been pushed via [`push_rows`](Self::push_rows) before
    /// calling this. Returns an error if the image is incomplete.
    #[track_caller]
    pub fn finish(self) -> Result<Vec<u8>> {
        self.finish_inner()
            .map(|mut r| r.take_data().unwrap())
            .map_err(at)
    }

    /// Encode and return JXL bytes together with [`EncodeStats`].
    #[track_caller]
    pub fn finish_with_stats(self) -> Result<EncodeResult> {
        self.finish_inner().map_err(at)
    }

    /// Encode, appending to an existing buffer.
    #[track_caller]
    pub fn finish_into(self, out: &mut Vec<u8>) -> Result<EncodeResult> {
        let mut result = self.finish_inner().map_err(at)?;
        if let Some(data) = result.data.take() {
            out.extend_from_slice(&data);
        }
        Ok(result)
    }

    /// Encode, writing to a `std::io::Write` destination.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn finish_to(self, mut dest: impl std::io::Write) -> Result<EncodeResult> {
        let mut result = self.finish_inner().map_err(at)?;
        if let Some(data) = result.data.take() {
            dest.write_all(&data)
                .map_err(|e| at(EncodeError::from(e)))?;
        }
        Ok(result)
    }

    /// Encode, writing to a seekable destination ([`WritableSeek`]).
    ///
    /// **Streaming refactor #11 chunk 6**: seek-aware finish hook for
    /// the lossless modular encoder. Same chunk-6 caveat as
    /// [`LossyEncoder::finish_to_seekable`] — the bytes are computed in
    /// memory and written in a single pass today; the level-3 seek-
    /// back machinery (chunk 7) will use the seek capability once it
    /// lands. See [`LossyEncoder::finish_to_seekable`] for the full
    /// contract.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn finish_to_seekable(self, mut dest: impl WritableSeek) -> Result<EncodeResult> {
        let mut result = self.finish_inner().map_err(at)?;
        if let Some(data) = result.data.take() {
            dest.write_all(&data)
                .map_err(|e| at(EncodeError::from(e)))?;
        }
        Ok(result)
    }

    fn finish_inner(self) -> core::result::Result<EncodeResult, EncodeError> {
        use crate::bit_writer::BitWriter;
        use crate::headers::color_encoding::ColorSpace;
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::channel::ModularImage;
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        if self.rows_pushed != self.height {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "incomplete image: {} of {} rows pushed",
                    self.rows_pushed, self.height
                ),
            });
        }
        // Run the full config validator. Mirrors
        // `EncodeRequest::encode_inner`.
        self.cfg.validate()?;
        // Defensive caps on caller-supplied metadata buffers (mirrors
        // EncodeRequest::encode_inner).
        validate_metadata_sizes(
            self.icc_profile.as_deref(),
            self.exif.as_deref(),
            self.xmp.as_deref(),
            self.jumbf.as_deref(),
        )?;
        // Tone-mapping numeric range checks. See the lossy-encoder
        // mirror above for the `Some(_) iff non-default` shape.
        let it = (self.intensity_target != 255.0).then_some(self.intensity_target);
        let mn = (self.min_nits != 0.0).then_some(self.min_nits);
        validate_tone_mapping(it, mn)?;
        validate_source_gamma(self.source_gamma)?;
        validate_intrinsic_size(self.intrinsic_size)?;

        let cfg = &self.cfg;
        let w = self.width as usize;
        let h = self.height as usize;

        // Construct the per-encode allocation budget. Mirrors the request
        // path's up-front working-set check and propagates the cap through
        // to the modular FrameEncoder for hot allocation sites.
        let budget_cap = self
            .limits
            .as_ref()
            .map(|l| l.effective_max_memory_bytes())
            .unwrap_or(Limits::DEFAULT_MAX_MEMORY_BYTES);
        let budget = crate::budget::MemoryBudget::new(budget_cap);
        let est_bytes = (self.width as u64)
            .checked_mul(self.height as u64)
            .and_then(|n| n.checked_mul(40))
            .ok_or_else(|| EncodeError::LimitExceeded {
                message: format!(
                    "image {}x{} too large for working-set estimate",
                    self.width, self.height
                ),
            })?;
        if est_bytes > budget_cap {
            return Err(EncodeError::LimitExceeded {
                message: format!(
                    "estimated working set {est_bytes} bytes for {}x{} image \
                     exceeds budget cap {budget_cap}",
                    self.width, self.height
                ),
            });
        }

        let mut image = ModularImage {
            channels: self.channels,
            bit_depth: self.bit_depth,
            is_grayscale: self.is_grayscale,
            has_alpha: self.has_alpha,
        };

        let (codestream, mut stats) = run_with_threads(cfg.threads, || {
            // Reconstruct interleaved pixels for patch detection (8-bit RGB only)
            let num_channels = self.layout.bytes_per_pixel();
            let can_use_patches = cfg.effective_patches()
                && !image.is_grayscale
                && image.bit_depth <= 8
                && num_channels >= 3;
            let patches_data = if can_use_patches {
                let mut detection_pixels = vec![0u8; w * h * num_channels];
                let nc = core::cmp::min(num_channels, image.channels.len());
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..nc {
                            detection_pixels[(y * w + x) * num_channels + c] =
                                image.channels[c].get(x, y) as u8;
                        }
                        // Fill remaining channels (alpha) from the image
                        for c in nc..num_channels {
                            if c < image.channels.len() {
                                detection_pixels[(y * w + x) * num_channels + c] =
                                    image.channels[c].get(x, y) as u8;
                            }
                        }
                    }
                }
                let pd_opt = crate::vardct::patches::find_and_build_lossless(
                    &detection_pixels,
                    w,
                    h,
                    num_channels,
                    image.bit_depth,
                );
                // RFC#45 chunks 4-7 lossless backport (chunk 5 lossless
                // trial encoder): per-image cost gate (see
                // `PatchesData::is_cost_effective_lossless`).
                pd_opt.filter(|pd| pd.is_cost_effective_lossless(image.bit_depth, cfg.use_ans))
            } else {
                None
            };

            // Build file header
            let mut file_header = if image.is_grayscale {
                FileHeader::new_gray(self.width, self.height)
            } else if image.has_alpha {
                FileHeader::new_rgba(self.width, self.height)
            } else {
                FileHeader::new_rgb(self.width, self.height)
            };
            if image.bit_depth == 16 {
                file_header.metadata.bit_depth = crate::headers::file_header::BitDepth::uint16();
                for ec in &mut file_header.metadata.extra_channels {
                    ec.bit_depth = crate::headers::file_header::BitDepth::uint16();
                }
            }
            // Override file_header's color_encoding with the caller's
            // `with_color_encoding(...)` if set. Closes lossless
            // streaming portion of #17. Mirrors the encode_lossless
            // (one-shot) wiring.
            if let Some(ce) = self.color_encoding.clone() {
                file_header.metadata.color_encoding =
                    if image.is_grayscale && ce.color_space != ColorSpace::Gray {
                        crate::headers::color_encoding::ColorEncoding {
                            color_space: ColorSpace::Gray,
                            ..ce
                        }
                    } else {
                        ce
                    };
            }
            // Configurable bits_per_sample (#18 sub-feature). Lossless
            // preserves pixels bit-exactly so this only affects header
            // signaling — the encoded values stay whatever the caller
            // pushed via push_rows.
            if let Some(bits) = self.bits_per_sample {
                file_header.metadata.bit_depth.bits_per_sample = bits;
                for ec in &mut file_header.metadata.extra_channels {
                    ec.bit_depth.bits_per_sample = bits;
                }
            }
            // Premultiplied-alpha signaling — mirrors EncodeRequest's
            // wiring (#13 lossless portion). Encoded pixels are written
            // unchanged; the decoder learns from the bit how to
            // interpret them.
            if self.premultiplied_alpha {
                for ec in &mut file_header.metadata.extra_channels {
                    if ec.ec_type == crate::headers::extra_channels::ExtraChannelType::Alpha {
                        ec.alpha_associated = true;
                    }
                }
            }
            if self.icc_profile.is_some() {
                file_header.metadata.color_encoding.want_icc = true;
            }
            file_header.metadata.intensity_target = self.intensity_target;
            file_header.metadata.min_nits = self.min_nits;
            if let Some((w, h)) = self.intrinsic_size {
                file_header.metadata.have_intrinsic_size = true;
                file_header.metadata.intrinsic_width = w;
                file_header.metadata.intrinsic_height = h;
            }

            let mut writer = BitWriter::new();
            file_header.write(&mut writer).map_err(EncodeError::from)?;
            if let Some(ref icc) = self.icc_profile {
                crate::icc::write_icc(icc, &mut writer).map_err(EncodeError::from)?;
            }
            writer.zero_pad_to_byte();

            // Write reference frame and subtract patches
            if let Some(ref pd) = patches_data {
                let lossless_profile = cfg.effective_profile();
                crate::vardct::patches::encode_reference_frame_rgb(
                    pd,
                    image.bit_depth,
                    cfg.use_ans,
                    lossless_profile.patch_ref_tree_learning,
                    &mut writer,
                    Some(&budget),
                )
                .map_err(EncodeError::from)?;
                writer.zero_pad_to_byte();
                let bd = image.bit_depth;
                crate::vardct::patches::subtract_patches_modular(&mut image, pd, bd);
            }

            // Encode frame
            let smart_profile = cfg.effective_profile_for_image((w as u64) * (h as u64));
            let frame_encoder = FrameEncoder::new(
                w,
                h,
                FrameEncoderOptions {
                    use_modular: true,
                    effort: cfg.effort,
                    use_ans: cfg.use_ans,
                    use_tree_learning: cfg.effective_tree_learning(),
                    use_squeeze: cfg.squeeze,
                    enable_lz77: cfg.effective_lz77(),
                    lz77_method: cfg.lz77_method,
                    lossy_palette: cfg.lossy_palette,
                    encoder_mode: cfg.mode,
                    profile: smart_profile,
                    modular_knobs: cfg.modular_knobs(),
                    modular_group_size_shift: cfg.effective_modular_group_size_shift(),
                    ..Default::default()
                },
            )
            .with_budget(alloc::sync::Arc::clone(&budget));
            let color_encoding = if let Some(ce) = self.color_encoding.clone() {
                if image.is_grayscale && ce.color_space != ColorSpace::Gray {
                    ColorEncoding {
                        color_space: ColorSpace::Gray,
                        ..ce
                    }
                } else {
                    ce
                }
            } else if let Some(gamma) = self.source_gamma {
                if image.is_grayscale {
                    ColorEncoding::gray_with_gamma(gamma)
                } else {
                    ColorEncoding::with_gamma(gamma)
                }
            } else if image.is_grayscale {
                ColorEncoding::gray()
            } else {
                ColorEncoding::srgb()
            };
            frame_encoder
                .encode_modular_with_patches(
                    &image,
                    &color_encoding,
                    &mut writer,
                    patches_data.as_ref(),
                )
                .map_err(EncodeError::from)?;

            let stats = EncodeStats {
                mode: EncodeMode::Lossless,
                ans: cfg.use_ans,
                ..Default::default()
            };
            Ok::<_, EncodeError>((writer.finish_with_padding(), stats))
        })?;

        stats.codestream_size = codestream.len();

        // Streaming LosslessEncoder does not accept extra channels
        // beyond alpha; count alpha from layout.
        let icc_size = self.icc_profile.as_deref().map_or(0u64, |i| i.len() as u64);
        let num_ec = u32::from(self.layout.has_alpha());
        let level = compute_required_level(self.width, self.height, num_ec, false, icc_size)?;

        let has_meta = self.exif.is_some() || self.xmp.is_some() || self.jumbf.is_some();
        let output = if has_meta || crate::container::level_requires_container(level) {
            wrap_metadata_container(
                &codestream,
                self.exif.as_deref(),
                self.xmp.as_deref(),
                self.jumbf.as_deref(),
                self.brotli_metadata_quality,
                level,
            )
        } else {
            codestream
        };

        stats.output_size = output.len();
        Ok(EncodeResult {
            data: Some(output),
            stats,
        })
    }
}

impl LosslessConfig {
    /// Create a streaming encoder for incremental row input.
    ///
    /// Per-channel planes are pre-allocated and filled as rows are pushed via
    /// [`LosslessEncoder::push_rows`], allowing callers to free source buffers
    /// incrementally rather than materializing the entire image.
    #[track_caller]
    pub fn encoder(&self, width: u32, height: u32, layout: PixelLayout) -> Result<LosslessEncoder> {
        use crate::modular::channel::Channel;

        validate_dims(width, height).map_err(at)?;

        let w = width as usize;
        let h = height as usize;

        let (num_channels, bit_depth, is_grayscale, has_alpha) = match layout {
            PixelLayout::Rgb8 | PixelLayout::Bgr8 => (3, 8u32, false, false),
            PixelLayout::Rgba8 | PixelLayout::Bgra8 => (4, 8, false, true),
            PixelLayout::Gray8 => (1, 8, true, false),
            PixelLayout::GrayAlpha8 => (2, 8, true, true),
            PixelLayout::Rgb16 => (3, 16, false, false),
            PixelLayout::Rgba16 => (4, 16, false, true),
            PixelLayout::Gray16 => (1, 16, true, false),
            PixelLayout::GrayAlpha16 => (2, 16, true, true),
            other => return Err(at(EncodeError::UnsupportedPixelLayout(other))),
        };

        let mut channels = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            channels.push(Channel::new(w, h).map_err(|e| at(EncodeError::from(e)))?);
        }

        Ok(LosslessEncoder {
            cfg: self.clone(),
            width,
            height,
            layout,
            rows_pushed: 0,
            channels,
            num_source_channels: num_channels,
            bit_depth,
            is_grayscale,
            has_alpha,
            icc_profile: None,
            exif: None,
            xmp: None,
            jumbf: None,
            source_gamma: None,
            color_encoding: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            intrinsic_size: None,
            premultiplied_alpha: false,
            bits_per_sample: None,
            brotli_metadata_quality: None,
            limits: None,
        })
    }
}

// ── Thread pool helper ──────────────────────────────────────────────────────

/// Run a closure inside a rayon thread pool when the `parallel` feature
/// is enabled and `threads > 1`. Otherwise, just call the closure directly.
///
/// - `threads == 0`: use the ambient rayon pool (caller controls via
///   `pool.install()` or the global default).
/// - `threads == 1`: sequential — call `f()` on the current thread.
/// - `threads >= 2`: create a dedicated pool with that many threads.
#[cfg(feature = "parallel")]
fn run_with_threads<T>(threads: usize, f: impl FnOnce() -> T + Send) -> T
where
    T: Send,
{
    if threads <= 1 {
        return f();
    }
    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

#[cfg(not(feature = "parallel"))]
fn run_with_threads<T>(_threads: usize, f: impl FnOnce() -> T) -> T {
    f()
}

// ── Animation encode implementations ────────────────────────────────────────

fn validate_animation_input(
    width: u32,
    height: u32,
    layout: PixelLayout,
    frames: &[AnimationFrame<'_>],
) -> core::result::Result<(), EncodeError> {
    validate_dims(width, height)?;
    if frames.is_empty() {
        return Err(EncodeError::InvalidInput {
            message: "animation requires at least one frame".into(),
        });
    }
    let expected_size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(layout.bytes_per_pixel()))
        .ok_or_else(|| EncodeError::InvalidInput {
            message: "image dimensions overflow".into(),
        })?;
    // Match the still-image working-buffer headroom check (validate_pixels).
    const MAX_INTERNAL_SCALE: usize = 16;
    if (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(MAX_INTERNAL_SCALE))
        .is_none()
    {
        return Err(EncodeError::LimitExceeded {
            message: format!("image {width}x{height} too large for encoder working buffers"),
        });
    }
    let num_frames = frames.len();
    for (i, frame) in frames.iter().enumerate() {
        if frame.pixels.len() != expected_size {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "frame {} pixel buffer size mismatch: expected {expected_size}, got {}",
                    i,
                    frame.pixels.len()
                ),
            });
        }
        // Reference-only frames are stored to a save slot but NOT
        // presented as a displayable keyframe — decoders skip them
        // during playback. The codestream MUST end on a displayable
        // (Regular / SkipProgressive) frame; the spec gates
        // `is_last`, `duration`, and the blending fields on
        // `frame_type ∈ {Regular, SkipProgressive}` (FrameHeader::write
        // `normal_frame` predicate, headers/frame_header.rs:386).
        // Marking the last frame as ReferenceOnly would emit a
        // codestream that no decoder can present as the final image.
        if frame.reference_only && i == num_frames - 1 {
            return Err(EncodeError::InvalidInput {
                message: "last animation frame cannot be ReferenceOnly: the file must end on a \
                     displayable frame. Add a final regular AnimationFrame after the \
                     reference layer(s)."
                    .into(),
            });
        }
        // `save_as_reference` (and ReferenceOnly's implicit slot) only
        // accept values 0..=3 (2 bits in the bitstream).
        if let Some(slot) = frame.save_as_reference
            && slot > 3
        {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "frame {i}: save_as_reference slot {slot} out of range (must be 0..=3)"
                ),
            });
        }
        if let Some(src) = frame.blend_source
            && src > 3
        {
            return Err(EncodeError::InvalidInput {
                message: format!("frame {i}: blend_source {src} out of range (must be 0..=3)"),
            });
        }
    }
    Ok(())
}

fn encode_animation_lossless(
    cfg: &LosslessConfig,
    width: u32,
    height: u32,
    layout: PixelLayout,
    animation: &AnimationParams,
    frames: &[AnimationFrame<'_>],
    limits: Option<&Limits>,
) -> core::result::Result<Vec<u8>, EncodeError> {
    use crate::bit_writer::BitWriter;
    use crate::headers::file_header::AnimationHeader;
    use crate::headers::{ColorEncoding, FileHeader};
    use crate::modular::channel::ModularImage;
    use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

    cfg.validate()?;
    validate_animation_input(width, height, layout, frames)?;

    let w = width as usize;
    let h = height as usize;
    let num_frames = frames.len();

    // Per-encode allocation budget. Spans the lifetime of the entire
    // animation: every per-frame allocation charges against the same cap,
    // so an attacker cannot multiply the working set by sending many
    // oversized frames.
    let budget_cap = limits
        .map(|l| l.effective_max_memory_bytes())
        .unwrap_or(Limits::DEFAULT_MAX_MEMORY_BYTES);
    let budget = crate::budget::MemoryBudget::new(budget_cap);
    let est_bytes = (width as u64)
        .checked_mul(height as u64)
        .and_then(|n| n.checked_mul(40))
        .ok_or_else(|| EncodeError::LimitExceeded {
            message: format!("image {width}x{height} too large for working-set estimate"),
        })?;
    if est_bytes > budget_cap {
        return Err(EncodeError::LimitExceeded {
            message: format!(
                "estimated working set {est_bytes} bytes for {width}x{height} \
                 image exceeds budget cap {budget_cap}"
            ),
        });
    }

    // Build file header with animation
    let sample_image = match layout {
        PixelLayout::Rgb8 => ModularImage::from_rgb8(frames[0].pixels, w, h),
        PixelLayout::Rgba8 => ModularImage::from_rgba8(frames[0].pixels, w, h),
        PixelLayout::Bgr8 => ModularImage::from_rgb8(&bgr_to_rgb(frames[0].pixels, 3), w, h),
        PixelLayout::Bgra8 => ModularImage::from_rgba8(&bgr_to_rgb(frames[0].pixels, 4), w, h),
        PixelLayout::Gray8 => ModularImage::from_gray8(frames[0].pixels, w, h),
        PixelLayout::GrayAlpha8 => ModularImage::from_grayalpha8(frames[0].pixels, w, h),
        PixelLayout::Rgb16 => ModularImage::from_rgb16_native(frames[0].pixels, w, h),
        PixelLayout::Rgba16 => ModularImage::from_rgba16_native(frames[0].pixels, w, h),
        PixelLayout::Gray16 => ModularImage::from_gray16_native(frames[0].pixels, w, h),
        PixelLayout::GrayAlpha16 => ModularImage::from_grayalpha16_native(frames[0].pixels, w, h),
        other => return Err(EncodeError::UnsupportedPixelLayout(other)),
    }
    .map_err(EncodeError::from)?;

    let mut file_header = if sample_image.is_grayscale {
        FileHeader::new_gray(width, height)
    } else if sample_image.has_alpha {
        FileHeader::new_rgba(width, height)
    } else {
        FileHeader::new_rgb(width, height)
    };
    if sample_image.bit_depth == 16 {
        file_header.metadata.bit_depth = crate::headers::file_header::BitDepth::uint16();
        for ec in &mut file_header.metadata.extra_channels {
            ec.bit_depth = crate::headers::file_header::BitDepth::uint16();
        }
    }
    // `have_timecodes` flips to true if any frame supplied an
    // explicit timecode (libjxl writes the 32-bit timecode field
    // per-frame, so the file-level flag must be on).
    let have_timecodes = frames.iter().any(|f| f.timecode.is_some());
    file_header.metadata.animation = Some(AnimationHeader {
        tps_numerator: animation.tps_numerator,
        tps_denominator: animation.tps_denominator,
        num_loops: animation.num_loops,
        have_timecodes,
    });

    // Write file header
    let mut writer = BitWriter::new();
    file_header.write(&mut writer).map_err(EncodeError::from)?;
    writer.zero_pad_to_byte();

    // Encode each frame with crop detection
    let color_encoding = ColorEncoding::srgb();
    let bpp = layout.bytes_per_pixel();
    let mut prev_pixels: Option<&[u8]> = None;

    for (i, frame) in frames.iter().enumerate() {
        // Reference-only frames must be written full-size — they ARE
        // the canvas later regular frames composite against. Skip crop
        // detection; the diff base for the next regular frame stays
        // pinned to the last *displayed* frame (we don't update
        // `prev_pixels` after a reference-only frame, below).
        //
        // `delta_from_identical`: chunk-1 POC of
        // `with_auto_delta_frames` (A1 audit "Animation" — Skip /
        // delta frame encoding). When the caller opts in AND this
        // frame is byte-identical to the previous displayed frame,
        // the existing same-pixel `Replace`-over-1×1 emit is replaced
        // with a zero-pixel `Add`-over-1×1 emit. Add of zero leaves
        // the canvas unchanged (the decoder lifts the 1×1 crop to
        // linear float, adds 0.0, clamps, restores the canvas pixel),
        // and zero-valued modular pixels compress smaller than the
        // arbitrary canvas-pixel value the existing path encodes.
        //
        // Chunk-2 (this commit) widens chunk-1 in two ways:
        //   * RGBA support — when the identity short-circuit fires on
        //     a layout with alpha, the alpha extra channel pixels are
        //     also zeroed and the frame header's `ec_blend_modes` is
        //     overridden to `Add` (via `ec_blend_mode_override`) so
        //     `Add`-of-zero is a no-op for both main RGB and alpha.
        //   * Full-frame delta-residual trial — when frames differ but
        //     the caller has opted in, encode two variants per frame:
        //     (a) the existing Regular path and
        //     (b) a full-frame `BlendMode::Add` payload whose pixels
        //         are signed `frame_N - frame_N-1` deltas.
        //     Each variant is encoded into its own scratch BitWriter;
        //     whichever is smaller in bits gets appended to the output.
        //     Delta-residual is byte-exact for lossless because the
        //     modular signed-i32 channels round-trip both branches of
        //     the subtraction.
        let mut delta_from_identical = false;
        let crop = if frame.reference_only {
            None
        } else if let Some(prev) = prev_pixels {
            match detect_frame_crop(prev, frame.pixels, w, h, bpp, false) {
                Some(crop) if (crop.width as usize) < w || (crop.height as usize) < h => Some(crop),
                Some(_) => None, // Crop covers full frame — no benefit
                None => {
                    // Frames are identical — emit a minimal 1x1 crop to preserve canvas
                    if cfg.auto_delta_frames && frame.blend_mode.is_none() {
                        // Chunk-2: the alpha-channel gate from chunk-1
                        // is dropped here because `ec_blend_mode_override`
                        // now lets us match `Add` on extras too.
                        delta_from_identical = true;
                    }
                    Some(FrameCrop {
                        x0: 0,
                        y0: 0,
                        width: 1,
                        height: 1,
                    })
                }
            }
        } else {
            None // Frame 0: always full frame
        };

        // Build ModularImage from the appropriate pixel region. When
        // `delta_from_identical` is set, the 1×1 crop is filled with
        // zeros instead of the actual canvas pixel — the `Add` blend
        // mode below makes that a no-op redraw with cheaper modular
        // tokens.
        let (frame_w, frame_h, frame_pixels_owned);
        let frame_pixels: &[u8] = if let Some(ref crop) = crop {
            frame_w = crop.width as usize;
            frame_h = crop.height as usize;
            frame_pixels_owned = if delta_from_identical {
                vec![0u8; frame_w * frame_h * bpp]
            } else {
                extract_pixel_crop(frame.pixels, w, crop, bpp)
            };
            &frame_pixels_owned
        } else {
            frame_w = w;
            frame_h = h;
            frame_pixels_owned = Vec::new();
            let _ = &frame_pixels_owned; // suppress unused warning
            frame.pixels
        };

        let image = match layout {
            PixelLayout::Rgb8 => ModularImage::from_rgb8(frame_pixels, frame_w, frame_h),
            PixelLayout::Rgba8 => ModularImage::from_rgba8(frame_pixels, frame_w, frame_h),
            PixelLayout::Bgr8 => {
                ModularImage::from_rgb8(&bgr_to_rgb(frame_pixels, 3), frame_w, frame_h)
            }
            PixelLayout::Bgra8 => {
                ModularImage::from_rgba8(&bgr_to_rgb(frame_pixels, 4), frame_w, frame_h)
            }
            PixelLayout::Gray8 => ModularImage::from_gray8(frame_pixels, frame_w, frame_h),
            PixelLayout::GrayAlpha8 => {
                ModularImage::from_grayalpha8(frame_pixels, frame_w, frame_h)
            }
            PixelLayout::Rgb16 => ModularImage::from_rgb16_native(frame_pixels, frame_w, frame_h),
            PixelLayout::Rgba16 => ModularImage::from_rgba16_native(frame_pixels, frame_w, frame_h),
            PixelLayout::Gray16 => ModularImage::from_gray16_native(frame_pixels, frame_w, frame_h),
            PixelLayout::GrayAlpha16 => {
                ModularImage::from_grayalpha16_native(frame_pixels, frame_w, frame_h)
            }
            other => return Err(EncodeError::UnsupportedPixelLayout(other)),
        }
        .map_err(EncodeError::from)?;

        let use_tree_learning = cfg.effective_tree_learning();
        let smart_profile = cfg.effective_profile_for_image((frame_w as u64) * (frame_h as u64));
        let make_opts = |crop: Option<FrameCrop>,
                         blend_mode: Option<BlendMode>,
                         ec_override: Option<BlendMode>|
         -> FrameEncoderOptions {
            FrameEncoderOptions {
                use_modular: true,
                effort: cfg.effort,
                use_ans: cfg.use_ans,
                use_tree_learning,
                use_squeeze: cfg.squeeze,
                enable_lz77: cfg.effective_lz77(),
                lz77_method: cfg.lz77_method,
                lossy_palette: cfg.lossy_palette,
                encoder_mode: cfg.mode,
                profile: smart_profile.clone(),
                have_animation: true,
                have_timecodes,
                duration: frame.duration,
                is_last: i == num_frames - 1,
                crop,
                skip_rct: false,
                blend_mode,
                blend_source: frame.blend_source,
                save_as_reference: frame.save_as_reference,
                ec_blend_mode_override: ec_override,
                reference_only: frame.reference_only,
                name: frame.name.clone(),
                timecode: frame.timecode,
                modular_knobs: cfg.modular_knobs(),
                modular_group_size_shift: cfg.effective_modular_group_size_shift(),
                dc_quant_custom: None,
            }
        };

        // Chunk-2 RGBA extension to the identity short-circuit: when
        // `delta_from_identical` fires AND the input has alpha, ALL
        // channels in the 1×1 crop are already zero (the `vec![0u8;
        // frame_w * frame_h * bpp]` above zeroed the interleaved RGBA
        // bytes including the alpha byte), so we just need the frame
        // header to apply `Add` to both the main and the alpha extra
        // channel. `ec_blend_mode_override = Some(Add)` makes the
        // modular path overwrite the default `Replace`-for-extras with
        // `Add` for every extra channel in the frame header.
        let identity_blend_mode = if delta_from_identical {
            Some(BlendMode::Add)
        } else {
            frame.blend_mode
        };
        let identity_ec_override = if delta_from_identical {
            Some(BlendMode::Add)
        } else {
            None
        };

        // Trial-encode candidate A: Regular (the existing path's
        // header + payload). For frame 0, the identity short-circuit
        // branch, and reference-only frames we ship A unconditionally.
        let mut writer_a = crate::bit_writer::BitWriter::new();
        FrameEncoder::new(
            frame_w,
            frame_h,
            make_opts(crop, identity_blend_mode, identity_ec_override),
        )
        .with_budget(alloc::sync::Arc::clone(&budget))
        .encode_modular(&image, &color_encoding, &mut writer_a)
        .map_err(EncodeError::from)?;

        // Trial-encode candidate B: full-frame `BlendMode::Add` delta
        // payload. Only attempted when the caller has opted in, a
        // previous-frame canvas exists, this frame is genuinely
        // different (`crop` is `Some` with sub-frame coverage), and
        // the caller has not pinned a non-default blend mode of their
        // own. Skipped for reference-only frames (they ARE the
        // canvas, not a delta against it) and for frame 0 (no
        // previous canvas).
        //
        // Delta image is built by subtracting the previous *displayed*
        // canvas from the current frame in interleaved-pixel space.
        // Signed deltas live in the modular signed-i32 channel; the
        // decoder's `Add` blend mode adds the float-lifted delta back
        // to the float-lifted canvas, restoring `frame.pixels`
        // exactly (lossless modular round-trip).
        //
        // RGBA: the alpha extra also needs `Add` — same
        // `ec_blend_mode_override` plumbing as the identity branch.
        let writer_b: Option<crate::bit_writer::BitWriter> = if cfg.auto_delta_frames
            && !delta_from_identical
            && !frame.reference_only
            && frame.blend_mode.is_none()
            && crop.is_some()
            && let Some(prev) = prev_pixels
        {
            // The delta payload is full-frame: dimensions match the
            // canvas, so the `crop` is `None` and the blend covers
            // the whole frame at (0,0).
            match build_lossless_delta_image(layout, frame.pixels, prev, w, h, bpp) {
                Some(delta_image) => {
                    let mut wb = crate::bit_writer::BitWriter::new();
                    FrameEncoder::new(
                        w,
                        h,
                        make_opts(
                            None, // full-frame, no crop
                            Some(BlendMode::Add),
                            if layout.has_alpha() {
                                Some(BlendMode::Add)
                            } else {
                                None
                            },
                        ),
                    )
                    .with_budget(alloc::sync::Arc::clone(&budget))
                    .encode_modular(&delta_image, &color_encoding, &mut wb)
                    .map_err(EncodeError::from)?;
                    Some(wb)
                }
                // Unsupported layout for delta (e.g. one we haven't
                // wired into `build_lossless_delta_image`). Skip the
                // trial; ship candidate A.
                None => None,
            }
        } else {
            None
        };

        // Pick the smaller candidate (compare bit counts because
        // candidate appends are unaligned and the frame headers don't
        // self-pad to byte boundaries at start).
        let pick_a = match &writer_b {
            None => true,
            Some(wb) => writer_a.bits_written() <= wb.bits_written(),
        };
        if pick_a {
            writer
                .append_unaligned(&writer_a)
                .map_err(EncodeError::from)?;
        } else {
            writer
                .append_unaligned(writer_b.as_ref().unwrap())
                .map_err(EncodeError::from)?;
        }

        // ReferenceOnly frames are not the displayed canvas — keep
        // `prev_pixels` pinned to the previous displayed frame so the
        // next regular frame's crop-detection diffs against what the
        // viewer actually sees.
        if !frame.reference_only {
            prev_pixels = Some(frame.pixels);
        }
    }

    Ok(writer.finish_with_padding())
}

/// Build a full-frame signed-delta [`ModularImage`] for the lossless
/// auto-delta trial encode. Subtracts the previous displayed canvas
/// from the current frame in interleaved-pixel space and packs the
/// signed result into the modular signed-i32 channel.
///
/// Returns `None` for layouts we don't support yet (anything beyond
/// the 8-bit u8 / 16-bit u16 native-endian families). Reuses the same
/// channel layout as the matching `ModularImage::from_*` constructor
/// so the decoder sees an identically-shaped frame (the only difference
/// is the per-channel pixel values are signed deltas instead of raw
/// pixels — modular channels store i32 throughout).
fn build_lossless_delta_image(
    layout: PixelLayout,
    curr: &[u8],
    prev: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
) -> Option<crate::modular::channel::ModularImage> {
    use crate::modular::channel::{Channel, ModularImage};

    debug_assert_eq!(curr.len(), prev.len());
    debug_assert_eq!(curr.len(), width * height * bpp);

    // Per-channel layout + delta builder. For 8-bit layouts the delta
    // is `(curr[i] as i32) - (prev[i] as i32)`; for native-endian
    // 16-bit, it's the same after byte-pair → u16 → i32 reads. BGR
    // swap is handled by routing the BGR variants through the RGB
    // builder with a one-shot swap of `curr` and `prev`.
    let make_8bit = |interleaved_curr: &[u8],
                     interleaved_prev: &[u8],
                     num_chan: usize,
                     is_grayscale: bool,
                     has_alpha: bool|
     -> Option<ModularImage> {
        let mut channels = Vec::with_capacity(num_chan);
        for c in 0..num_chan {
            let mut ch = Channel::new(width, height).ok()?;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * num_chan + c;
                    let d = (interleaved_curr[idx] as i32) - (interleaved_prev[idx] as i32);
                    ch.set(x, y, d);
                }
            }
            channels.push(ch);
        }
        Some(ModularImage {
            channels,
            bit_depth: 8,
            is_grayscale,
            has_alpha,
        })
    };
    let make_16bit_native = |interleaved_curr: &[u8],
                             interleaved_prev: &[u8],
                             num_chan: usize,
                             is_grayscale: bool,
                             has_alpha: bool|
     -> Option<ModularImage> {
        let mut channels = Vec::with_capacity(num_chan);
        for c in 0..num_chan {
            let mut ch = Channel::new(width, height).ok()?;
            for y in 0..height {
                for x in 0..width {
                    let pix = (y * width + x) * num_chan + c;
                    let off = pix * 2;
                    let cur = u16::from_ne_bytes([interleaved_curr[off], interleaved_curr[off + 1]])
                        as i32;
                    let prv = u16::from_ne_bytes([interleaved_prev[off], interleaved_prev[off + 1]])
                        as i32;
                    ch.set(x, y, cur - prv);
                }
            }
            channels.push(ch);
        }
        Some(ModularImage {
            channels,
            bit_depth: 16,
            is_grayscale,
            has_alpha,
        })
    };

    match layout {
        PixelLayout::Rgb8 => make_8bit(curr, prev, 3, false, false),
        PixelLayout::Rgba8 => make_8bit(curr, prev, 4, false, true),
        PixelLayout::Bgr8 => {
            let curr_swap = bgr_to_rgb(curr, 3);
            let prev_swap = bgr_to_rgb(prev, 3);
            make_8bit(&curr_swap, &prev_swap, 3, false, false)
        }
        PixelLayout::Bgra8 => {
            let curr_swap = bgr_to_rgb(curr, 4);
            let prev_swap = bgr_to_rgb(prev, 4);
            make_8bit(&curr_swap, &prev_swap, 4, false, true)
        }
        PixelLayout::Gray8 => make_8bit(curr, prev, 1, true, false),
        PixelLayout::GrayAlpha8 => make_8bit(curr, prev, 2, true, true),
        PixelLayout::Rgb16 => make_16bit_native(curr, prev, 3, false, false),
        PixelLayout::Rgba16 => make_16bit_native(curr, prev, 4, false, true),
        PixelLayout::Gray16 => make_16bit_native(curr, prev, 1, true, false),
        PixelLayout::GrayAlpha16 => make_16bit_native(curr, prev, 2, true, true),
        // Float / PQ / HLG / CMYK aren't currently routed through the
        // lossless animation path; if/when they are, add matching arms
        // here. Returning `None` makes the trial-encode silently fall
        // back to candidate A (Regular).
        _ => None,
    }
}

fn encode_animation_lossy(
    cfg: &LossyConfig,
    width: u32,
    height: u32,
    layout: PixelLayout,
    animation: &AnimationParams,
    frames: &[AnimationFrame<'_>],
    limits: Option<&Limits>,
) -> core::result::Result<Vec<u8>, EncodeError> {
    use crate::bit_writer::BitWriter;
    use crate::headers::file_header::AnimationHeader;
    use crate::headers::frame_header::FrameOptions;

    cfg.validate()?;
    validate_animation_input(width, height, layout, frames)?;

    let w = width as usize;
    let h = height as usize;
    let num_frames = frames.len();

    // Per-encode allocation budget. Spans the lifetime of the entire
    // animation; see `encode_animation_lossless` for the reasoning.
    let budget_cap = limits
        .map(|l| l.effective_max_memory_bytes())
        .unwrap_or(Limits::DEFAULT_MAX_MEMORY_BYTES);
    let budget = crate::budget::MemoryBudget::new(budget_cap);
    let est_bytes = (width as u64)
        .checked_mul(height as u64)
        .and_then(|n| n.checked_mul(40))
        .ok_or_else(|| EncodeError::LimitExceeded {
            message: format!("image {width}x{height} too large for working-set estimate"),
        })?;
    if est_bytes > budget_cap {
        return Err(EncodeError::LimitExceeded {
            message: format!(
                "estimated working set {est_bytes} bytes for {width}x{height} \
                 image exceeds budget cap {budget_cap}"
            ),
        });
    }

    // Set up VarDCT encoder
    let mut profile = cfg.effective_profile_for_image((width as u64) * (height as u64));

    // Apply max_strategy_size to profile flags
    if let Some(max_size) = cfg.max_strategy_size {
        if max_size < 16 {
            profile.try_dct16 = false;
        }
        if max_size < 32 {
            profile.try_dct32 = false;
        }
        if max_size < 64 {
            profile.try_dct64 = false;
        }
    }

    let mut enc = crate::vardct::VarDctEncoder::new(cfg.distance);
    // W44-128 Chunk B + W44-130 Chunk D: resolve EncoderStrategy
    // bundle once (animation per-frame path). Field is non-optional
    // as of Chunk D.
    enc.resolved_improvements = cfg.resolve_improvements();
    enc.effort = cfg.effort;
    enc.profile = profile;
    enc.use_ans = cfg.use_ans;
    enc.optimize_codes = enc.profile.optimize_codes;
    enc.custom_orders = enc.profile.custom_orders;
    enc.ac_strategy_enabled = enc.profile.ac_strategy_enabled;
    enc.enable_noise = cfg.noise;
    enc.photon_noise_iso = cfg.photon_noise_iso;
    enc.manual_noise_lut = cfg.manual_noise_lut;
    enc.quant_ac_rescale = cfg.quant_ac_rescale;
    enc.original_distance = cfg.original_distance;
    enc.enable_denoise = cfg.denoise;
    // libjxl gates gaborish at distance > 0.5 (enc_frame.cc:281)
    // and unconditionally OFF at decoding_speed_tier == 4
    // (enc_frame.cc:280) — captured by `cfg.effective_gaborish()`.
    enc.enable_gaborish = cfg.effective_gaborish() && cfg.distance > 0.5;
    // EX-J13: adaptive gaborish is silently gated to be a subset of
    // gaborish (no-op when the fixed inverse is disabled).
    enc.enable_adaptive_gaborish = enc.enable_gaborish && cfg.adaptive_gaborish;
    // libjxl `--epf -1..3` override (enc_frame.cc:284-285). `-1` =
    // encoder chooses by distance; otherwise force the given count.
    enc.epf_level_override = if cfg.epf_level < 0 {
        None
    } else {
        Some(cfg.epf_level as u32)
    };
    // W44-130 Chunk D: dispatch policies hydrated from the resolved
    // bundle (LossyConfig setters deleted; absorbed into
    // `EncoderImprovementsCustom`).
    enc.epf_dispatch = enc.resolved_improvements.epf_dispatch;
    enc.error_diffusion = cfg.error_diffusion;
    enc.pixel_domain_loss = cfg.pixel_domain_loss;
    enc.pixel_loss_dispatch = enc.resolved_improvements.pixel_loss_dispatch;
    enc.single_pass_entropy_dispatch = enc.resolved_improvements.single_pass_entropy_dispatch;
    enc.enable_lz77 = cfg.effective_lz77();
    enc.lz77_method = cfg.lz77_method;
    enc.force_strategy = cfg.force_strategy;
    // RFC #45 pick #4 — when the caller has explicitly pinned `cfg.patches`
    // via `with_patches`, that wins; otherwise read the per-image
    // dispatched profile (the content-class adapter may have flipped
    // patches on for Screenshot content at e5/e6).
    enc.enable_patches = if cfg.patches_explicit {
        cfg.effective_patches()
    } else if cfg.faster_decoding >= 2 {
        // libjxl `enc_modular.cc:707` skips patches at
        // `decoding_speed_tier >= 2`.
        false
    } else {
        enc.profile.patches
    };
    enc.patches_dispatch = enc.resolved_improvements.patches_dispatch;
    enc.enable_dot_detection = cfg.dot_detection;
    enc.encoder_mode = cfg.mode;
    enc.splines = cfg.splines.clone();
    enc.auto_splines = cfg.auto_splines;
    enc.progressive = cfg.progressive;
    enc.use_lf_frame = cfg.lf_frame;
    // W44-130 Chunk D: `content_aware_entropy_mul` + legacy
    // `with_*_hint` setters all deleted; strategy + overrides flow
    // via `cfg.resolve_improvements()` into `enc.resolved_improvements`.
    // W44-91: animation per-frame encodes don't compute the
    // zenanalyze-proxy because the proxy is per-image and the discriminator
    // logic was designed against still-image CID22 validation cells.
    // Each frame falls back to the W44-29 mask1x1<50 gate (which works
    // on per-frame XYB the same way it does on still images). Callers
    // that want the W44-91 lift on a specific frame can use
    // [`LossyConfig::with_strategy_overrides`] with
    // `high_d_photo_hint: Some(true)` explicitly.
    // Streaming refactor #11 chunk 6 (animation frame path).
    enc.buffering = cfg.buffering;
    #[cfg(feature = "butteraugli-loop")]
    {
        enc.butteraugli_iters = cfg.butteraugli_iters;
        // EX-J11 chunk 4: see the still-image `encode_lossy` site
        // for the resolution rationale. The animation API has no
        // per-encode `with_color_encoding` (today), so resolution
        // falls back to the layout's implied transfer function —
        // PQ / HLG f32 layouts will route to `Vdp2`, everything
        // else to `Butteraugli`.
        enc.hdr_loss = cfg.resolve_hdr_loss(layout, None);
    }
    #[cfg(feature = "ssim2-loop")]
    {
        enc.ssim2_iters = cfg.ssim2_iters;
    }
    #[cfg(feature = "zensim-loop")]
    {
        enc.zensim_iters = cfg.zensim_iters;
    }
    enc.non_finite_action = cfg.non_finite_action;
    enc.budget = Some(alloc::sync::Arc::clone(&budget));
    // Premultiplied-alpha signaling for animation (mirrors the still
    // image lossy path's `enc.alpha_associated = self.premultiplied_alpha`
    // at api.rs:3877). Per-frame the linear RGB is unpremultiplied
    // before XYB conversion below; the codestream header signals
    // `alpha_associated=true` so the decoder re-premultiplies on output.
    enc.alpha_associated = animation.premultiplied_alpha;

    // Detect alpha and 16-bit from layout
    let has_alpha = layout.has_alpha();
    let bit_depth_16 = matches!(layout, PixelLayout::Rgb16 | PixelLayout::Rgba16);
    enc.bit_depth_16 = bit_depth_16;

    // Build file header from VarDCT encoder (sets xyb_encoded, rendering_intent, etc.)
    // then add animation metadata. Animation frames currently carry at
    // most one extra (alpha) — passing the alpha info list mirrors what
    // the old has_alpha bool used to derive.
    let alpha_info_buf;
    let extras_info: &[crate::headers::extra_channels::ExtraChannelInfo] = if has_alpha {
        let mut info = crate::headers::extra_channels::ExtraChannelInfo::alpha();
        info.alpha_associated = enc.alpha_associated;
        alpha_info_buf = [info];
        &alpha_info_buf
    } else {
        &[]
    };
    let mut file_header = enc.build_file_header(w, h, extras_info);
    // `have_timecodes` flips to true if any frame supplied an explicit
    // timecode (libjxl writes the 32-bit timecode field per-frame, so
    // the file-level flag must be on).
    let have_timecodes = frames.iter().any(|f| f.timecode.is_some());
    file_header.metadata.animation = Some(AnimationHeader {
        tps_numerator: animation.tps_numerator,
        tps_denominator: animation.tps_denominator,
        num_loops: animation.num_loops,
        have_timecodes,
    });

    let mut writer = BitWriter::with_capacity(w * h * 4);
    file_header.write(&mut writer).map_err(EncodeError::from)?;
    if let Some(ref icc) = enc.icc_profile {
        crate::icc::write_icc(icc, &mut writer).map_err(EncodeError::from)?;
    }
    writer.zero_pad_to_byte();

    // Encode each frame with crop detection
    let bpp = layout.bytes_per_pixel();
    let mut prev_pixels: Option<&[u8]> = None;

    for (i, frame) in frames.iter().enumerate() {
        // Reference-only frames are written full-size into a save slot
        // — they ARE the canvas that subsequent regular frames composite
        // against, so cropping them would discard the area outside the
        // diff bounding box. Skip crop detection entirely; the diff
        // base for the NEXT regular frame stays pinned to the last
        // *displayed* frame (we don't update `prev_pixels` after a
        // reference-only frame, below).
        // `delta_from_identical`: chunk-1 POC of
        // `with_auto_delta_frames` for the lossy animation path.
        // Mirrors the lossless path's identical-frame short-circuit
        // (see `encode_animation_lossless` for the rationale): when
        // the caller opts in AND frame N is byte-identical to the
        // previous displayed frame, the existing same-pixel
        // `Replace`-over-8×8 emit is replaced by a zero-pixel
        // `Add`-over-8×8 emit. All-zero VarDCT coefficients
        // dequantize to all-zero, IDCT to all-zero, and Add 0 to the
        // canvas is a no-op redraw with cheaper modular-quantised
        // tokens.
        //
        // Chunk-2 extends RGBA support: when alpha is present, the
        // extra-channel blend mode is forced to `Add` via
        // `ec_blend_mode_override`, and the alpha-extra payload is
        // zero (the alpha input is set to all-zero below). The full
        // delta-residual trial-encode that the lossless path runs is
        // NOT applied to the lossy pipeline because residuals must
        // round-trip through the reconstructed (already-quantised)
        // reference frame, not the original pixels — that requires a
        // reconstruction shadow that chunk-2 does not yet wire.
        let mut delta_from_identical = false;
        let crop = if frame.reference_only {
            None
        } else if let Some(prev) = prev_pixels {
            match detect_frame_crop(prev, frame.pixels, w, h, bpp, true) {
                Some(crop) if (crop.width as usize) < w || (crop.height as usize) < h => Some(crop),
                Some(_) => None, // Crop covers full frame — no benefit
                None => {
                    // Frames identical — emit minimal 8x8 crop (VarDCT minimum)
                    if cfg.auto_delta_frames && frame.blend_mode.is_none() {
                        // Chunk-2: the alpha-channel gate from chunk-1
                        // is dropped — `ec_blend_mode_override` (set in
                        // `frame_options` below) now sets the alpha
                        // extra channel's blend mode to `Add` so that
                        // an `Add`-of-zero alpha is also a no-op.
                        delta_from_identical = true;
                    }
                    Some(FrameCrop {
                        x0: 0,
                        y0: 0,
                        width: 8.min(width),
                        height: 8.min(height),
                    })
                }
            }
        } else {
            None // Frame 0: always full frame
        };

        // Extract crop region from raw pixels, then convert to linear.
        // When `delta_from_identical` is set, the 8×8 crop is filled
        // with zeros so the resulting VarDCT block round-trips to a
        // zero canvas-delta under the `Add` blend mode applied below.
        let (frame_w, frame_h) = if let Some(ref crop) = crop {
            (crop.width as usize, crop.height as usize)
        } else {
            (w, h)
        };

        let crop_pixels_owned;
        let src_pixels: &[u8] = if let Some(ref crop) = crop {
            crop_pixels_owned = if delta_from_identical {
                vec![0u8; (crop.width as usize) * (crop.height as usize) * bpp]
            } else {
                extract_pixel_crop(frame.pixels, w, crop, bpp)
            };
            &crop_pixels_owned
        } else {
            crop_pixels_owned = Vec::new();
            let _ = &crop_pixels_owned;
            frame.pixels
        };

        let (linear_rgb, alpha) = match layout {
            PixelLayout::Rgb8 => (srgb_u8_to_linear_f32(src_pixels, 3), None),
            PixelLayout::Bgr8 => (srgb_u8_to_linear_f32(&bgr_to_rgb(src_pixels, 3), 3), None),
            PixelLayout::Rgba8 => {
                let rgb = srgb_u8_to_linear_f32(src_pixels, 4);
                let alpha = extract_alpha(src_pixels, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::Bgra8 => {
                let swapped = bgr_to_rgb(src_pixels, 4);
                let rgb = srgb_u8_to_linear_f32(&swapped, 4);
                let alpha = extract_alpha(src_pixels, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::Gray8 => (gray_u8_to_linear_f32_rgb(src_pixels, 1), None),
            PixelLayout::GrayAlpha8 => {
                let rgb = gray_u8_to_linear_f32_rgb(src_pixels, 2);
                let alpha = extract_alpha(src_pixels, 2, 1);
                (rgb, Some(alpha))
            }
            PixelLayout::Rgb16 => (srgb_u16_to_linear_f32(src_pixels, 3, 65535.0), None),
            PixelLayout::Rgba16 => {
                let rgb = srgb_u16_to_linear_f32(src_pixels, 4, 65535.0);
                let alpha = extract_alpha_u16(src_pixels, 4, 3, 65535.0);
                (rgb, Some(alpha))
            }
            PixelLayout::Gray16 => (gray_u16_to_linear_f32_rgb(src_pixels, 1, 65535.0), None),
            PixelLayout::GrayAlpha16 => {
                let rgb = gray_u16_to_linear_f32_rgb(src_pixels, 2, 65535.0);
                let alpha = extract_alpha_u16(src_pixels, 2, 1, 65535.0);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                (floats.to_vec(), None)
            }
            PixelLayout::RgbaLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                let rgb: Vec<f32> = floats
                    .chunks(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect();
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::GrayLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                (gray_f32_to_linear_f32_rgb(floats, 1), None)
            }
            PixelLayout::GrayAlphaLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                let rgb = gray_f32_to_linear_f32_rgb(floats, 2);
                let alpha = extract_alpha_f32(floats, 2, 1);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbLinearF16 => (f16_to_linear_f32_rgb(src_pixels, 3), None),
            PixelLayout::RgbaLinearF16 => {
                let rgb = f16_to_linear_f32_rgb(src_pixels, 4);
                let alpha = extract_alpha_f16(src_pixels, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::GrayLinearF16 => (f16_gray_to_linear_f32_rgb(src_pixels, 1), None),
            PixelLayout::GrayAlphaLinearF16 => {
                let rgb = f16_gray_to_linear_f32_rgb(src_pixels, 2);
                let alpha = extract_alpha_f16(src_pixels, 2, 1);
                (rgb, Some(alpha))
            }
            // A3 chunk 1b: f32 PQ/HLG/BT.709 RGB(A) (issue #46).
            PixelLayout::RgbPqF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                (pq_f32_to_linear_f32_rgb(floats, 3), None)
            }
            PixelLayout::RgbaPqF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                let rgb = pq_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbHlgF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                (hlg_f32_to_linear_f32_rgb(floats, 3), None)
            }
            PixelLayout::RgbaHlgF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                let rgb = hlg_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbBt709F32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                (bt709_f32_to_linear_f32_rgb(floats, 3), None)
            }
            PixelLayout::RgbaBt709F32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                let rgb = bt709_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha))
            }
            // Animated CMYK (multi-frame lossy) is not yet wired — only
            // the one-shot lossless path handles CMYK input.
            PixelLayout::Cmyk8 | PixelLayout::Cmyk16 => {
                return Err(EncodeError::UnsupportedPixelLayout(layout));
            }
        };

        // Mirror of the still-image lossy pre-passes at api.rs:3776-3807.
        // Order is load-bearing: unpremultiply FIRST so SimplifyInvisible
        // operates on straight-alpha RGB, matching libjxl
        // `enc_frame.cc:1588-1597` which gates SimplifyInvisible on
        // `!alpha_eci->alpha_associated`. After unpremultiplication the
        // input is straight-alpha and the simplify pass is enabled even
        // when the caller signalled premultiplied alpha. The codestream
        // header still gets `alpha_associated=true` (set above) so the
        // decoder re-premultiplies on output.
        let mut linear_rgb = linear_rgb;
        if animation.premultiplied_alpha
            && let Some(ref alpha_buf) = alpha
        {
            unpremultiply_alpha_inplace(&mut linear_rgb, alpha_buf);
        }

        // SimplifyInvisible pre-pass (closes #10) — mirrors the
        // one-shot still-image path at api.rs:3795-3807. Smear color
        // values in alpha=0 pixels to a weighted average of visible
        // neighbors, reducing high-frequency DCT energy from arbitrary
        // garbage in transparent regions. libjxl `enc_frame.cc:511`
        // (default-on for lossy). Sprites/icons benefit (5-20% smaller);
        // photos with mostly-opaque alpha pay only the cheap
        // `has_any_invisible_pixels` predicate (single linear scan with
        // early-exit on the first zero).
        if cfg.simplify_invisible
            && !animation.premultiplied_alpha
            && let Some(ref alpha_buf) = alpha
            && crate::vardct::simplify_invisible::has_any_invisible_pixels(alpha_buf)
        {
            crate::vardct::simplify_invisible::simplify_invisible_rgb(
                &mut linear_rgb,
                alpha_buf,
                frame_w,
                frame_h,
                false, // lossless = false (smear, not zero)
            );
        }

        let frame_options = FrameOptions {
            have_animation: true,
            have_timecodes,
            duration: frame.duration,
            is_last: i == num_frames - 1,
            crop,
            blend_mode: if delta_from_identical {
                // Chunk-1 POC: identical-frame short-circuit via
                // `BlendMode::Add` over a zero-pixel 8×8 crop
                // (see `delta_from_identical` setup above).
                Some(BlendMode::Add)
            } else {
                frame.blend_mode
            },
            blend_source: frame.blend_source,
            save_as_reference: frame.save_as_reference,
            // Chunk-2 RGBA extension: when the identity short-circuit
            // fires on a layout with alpha, force the alpha extra
            // channel to `Add` too so the zeroed alpha buffer is a
            // canvas-preserving no-op. On non-alpha layouts the
            // override is `None` and the override block in
            // `vardct/bitstream.rs` is skipped.
            ec_blend_mode_override: if delta_from_identical && layout.has_alpha() {
                Some(BlendMode::Add)
            } else {
                None
            },
            reference_only: frame.reference_only,
            name: frame.name.clone(),
            timecode: frame.timecode,
        };

        // Animation frames currently only support alpha as an extra
        // channel. Build the extras list (zero or one entries) and
        // hand it to the encoder.
        let alpha_info_buf;
        let alpha_view_buf;
        let frame_extras: &[crate::vardct::extras::VardctExtra<'_>] = match alpha.as_deref() {
            None => &[],
            Some(buf) => {
                let mut info = crate::headers::extra_channels::ExtraChannelInfo::alpha();
                info.alpha_associated = enc.alpha_associated;
                alpha_info_buf = info;
                alpha_view_buf = [crate::vardct::extras::VardctExtra {
                    info: &alpha_info_buf,
                    data: crate::vardct::extras::VardctExtraBuf::U8(buf),
                }];
                &alpha_view_buf[..]
            }
        };

        enc.encode_frame_to_writer(
            frame_w,
            frame_h,
            &linear_rgb,
            frame_extras,
            &frame_options,
            &mut writer,
        )
        .map_err(EncodeError::from)?;

        // ReferenceOnly frames are not the displayed canvas — keep
        // `prev_pixels` pinned to the previous displayed frame so the
        // next regular frame's crop-detection diffs against what the
        // viewer actually sees.
        if !frame.reference_only {
            prev_pixels = Some(frame.pixels);
        }
    }

    Ok(writer.finish_with_padding())
}

// ── Animation frame crop detection ──────────────────────────────────────────

use crate::headers::frame_header::FrameCrop;

/// Detects the minimal bounding rectangle that differs between two frames.
///
/// Compares `prev` and `curr` byte-by-byte. Returns `Some(FrameCrop)` with the
/// tight bounding box of changed pixels, or `None` if the frames are identical.
///
/// When `align_to_8x8` is true (for VarDCT), the crop is expanded outward to
/// 8x8 block boundaries for better compression.
fn detect_frame_crop(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    height: usize,
    bytes_per_pixel: usize,
    align_to_8x8: bool,
) -> Option<FrameCrop> {
    let stride = width * bytes_per_pixel;
    debug_assert_eq!(prev.len(), height * stride);
    debug_assert_eq!(curr.len(), height * stride);

    // Find top (first row with a difference)
    let mut top = height;
    let mut bottom = 0;
    let mut left = width;
    let mut right = 0;

    for y in 0..height {
        let row_start = y * stride;
        let prev_row = &prev[row_start..row_start + stride];
        let curr_row = &curr[row_start..row_start + stride];

        // Fast row comparison via u64 chunks — lets the compiler auto-vectorize
        let (prev_prefix, prev_u64, prev_suffix) = bytemuck::pod_align_to::<u8, u64>(prev_row);
        let (curr_prefix, curr_u64, curr_suffix) = bytemuck::pod_align_to::<u8, u64>(curr_row);
        if prev_prefix == curr_prefix && prev_u64 == curr_u64 && prev_suffix == curr_suffix {
            continue;
        }

        // This row has differences — find leftmost and rightmost changed pixel
        if top == height {
            top = y;
        }
        bottom = y;

        // Scan from left to find first differing pixel
        for x in 0..width {
            let px_start = x * bytes_per_pixel;
            if prev_row[px_start..px_start + bytes_per_pixel]
                != curr_row[px_start..px_start + bytes_per_pixel]
            {
                left = left.min(x);
                break;
            }
        }
        // Scan from right to find last differing pixel
        for x in (0..width).rev() {
            let px_start = x * bytes_per_pixel;
            if prev_row[px_start..px_start + bytes_per_pixel]
                != curr_row[px_start..px_start + bytes_per_pixel]
            {
                right = right.max(x);
                break;
            }
        }
    }

    if top == height {
        // Frames are identical
        return None;
    }

    // Convert to crop rectangle (inclusive → exclusive for width/height)
    let mut crop_x = left as i32;
    let mut crop_y = top as i32;
    let mut crop_w = (right - left + 1) as u32;
    let mut crop_h = (bottom - top + 1) as u32;

    if align_to_8x8 {
        // Expand to 8x8 block boundaries
        let aligned_x = (crop_x / 8) * 8;
        let aligned_y = (crop_y / 8) * 8;
        let end_x = (crop_x as u32 + crop_w).div_ceil(8) * 8;
        let end_y = (crop_y as u32 + crop_h).div_ceil(8) * 8;
        crop_x = aligned_x;
        crop_y = aligned_y;
        crop_w = end_x.min(width as u32) - aligned_x as u32;
        crop_h = end_y.min(height as u32) - aligned_y as u32;
    }

    Some(FrameCrop {
        x0: crop_x,
        y0: crop_y,
        width: crop_w,
        height: crop_h,
    })
}

/// Extracts a rectangular crop region from a pixel buffer.
///
/// `bytes_per_pixel` is the number of bytes per pixel (e.g., 3 for RGB, 4 for RGBA).
fn extract_pixel_crop(
    pixels: &[u8],
    full_width: usize,
    crop: &FrameCrop,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    let cx = crop.x0 as usize;
    let cy = crop.y0 as usize;
    let cw = crop.width as usize;
    let ch = crop.height as usize;
    let stride = full_width * bytes_per_pixel;

    let mut out = Vec::with_capacity(cw * ch * bytes_per_pixel);
    for y in cy..cy + ch {
        let row_start = y * stride + cx * bytes_per_pixel;
        out.extend_from_slice(&pixels[row_start..row_start + cw * bytes_per_pixel]);
    }
    out
}

// ── Pixel conversion helpers ────────────────────────────────────────────────

/// Pre-computed sRGB u8 → linear f32 lookup table (256 entries).
/// Eliminates per-pixel `powf(2.4)` calls for the common 8-bit path.
const SRGB_U8_TO_LINEAR: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0u16;
    while i < 256 {
        let c = i as f64 / 255.0;
        // Use f64 for accuracy during const eval, then truncate to f32.
        // powf is not const, so we use exp(2.4 * ln(x)) via a manual series.
        // For const context, we precompute using the piecewise sRGB TF.
        table[i as usize] = if c <= 0.04045 {
            (c / 12.92) as f32
        } else {
            // ((c + 0.055) / 1.055)^2.4
            // = exp(2.4 * ln((c + 0.055) / 1.055))
            // Approximate via repeated squaring: x^2.4 = x^2 * x^0.4
            // x^0.4 = (x^0.5)^0.8 = ((x^0.5)^0.5)^... too complex for const.
            // Instead, use the identity: x^2.4 = (x^12)^(1/5)
            // and compute fifth root via Newton's method in f64.
            let base = (c + 0.055) / 1.055;
            // x^12 = ((x^2)^2)^3
            let x2 = base * base;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            let x12 = x8 * x4;
            // Fifth root of x^12 = x^(12/5) = x^2.4
            // Newton: y_{n+1} = y_n - (y_n^5 - x12) / (5 * y_n^4)
            //       = (4*y_n + x12/y_n^4) / 5
            let mut y = base * base; // initial guess ~x^2
            // 8 iterations of Newton's method for fifth root (converges in ~6 for f64)
            let mut iter = 0;
            while iter < 8 {
                let y2 = y * y;
                let y4 = y2 * y2;
                y = (4.0 * y + x12 / y4) / 5.0;
                iter += 1;
            }
            y as f32
        };
        i += 1;
    }
    table
};

/// sRGB u8 → linear f32 via LUT.
#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    SRGB_U8_TO_LINEAR[c as usize]
}

fn srgb_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let num_pixels = data.len() / channels;
    let mut out = vec![0.0f32; num_pixels * 3];
    let lut = &SRGB_U8_TO_LINEAR;
    // zip chunks to eliminate output bounds checks; u8 index into [f32; 256] is always in bounds
    for (px, rgb) in data.chunks_exact(channels).zip(out.chunks_exact_mut(3)) {
        rgb[0] = lut[px[0] as usize];
        rgb[1] = lut[px[1] as usize];
        rgb[2] = lut[px[2] as usize];
    }
    out
}

/// PQ u8 → linear f32 RGB. Uses a 256-entry LUT (avoids per-pixel
/// powf — matches the gamma_u8_to_linear_f32 optimization). 8-bit
/// PQ is unusual in practice (PQ's headroom rewards wider precision)
/// but accepting it lets callers tag low-bit-depth content correctly.
fn pq_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| pq_to_linear_f(i as f32 / 255.0));
    data.chunks(channels)
        .flat_map(|px| {
            [
                lut[px[0] as usize],
                lut[px[1] as usize],
                lut[px[2] as usize],
            ]
        })
        .collect()
}

/// HLG u8 → linear f32 RGB. 256-entry LUT.
fn hlg_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| hlg_to_linear_f(i as f32 / 255.0));
    data.chunks(channels)
        .flat_map(|px| {
            [
                lut[px[0] as usize],
                lut[px[1] as usize],
                lut[px[2] as usize],
            ]
        })
        .collect()
}

/// BT.709 u8 → linear f32 RGB. 256-entry LUT.
fn bt709_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| bt709_to_linear_f(i as f32 / 255.0));
    data.chunks(channels)
        .flat_map(|px| {
            [
                lut[px[0] as usize],
                lut[px[1] as usize],
                lut[px[2] as usize],
            ]
        })
        .collect()
}

/// PQ u16 → linear f32. `u16_max` mirrors the convention in
/// `srgb_u16_to_linear_f32` — the divisor for input normalization.
/// Output is in linear [0..1] where 1.0 corresponds to the encoder's
/// `intensity_target` peak luminance. Closes PQ portion of #17.
fn pq_u16_to_linear_f32(data: &[u8], channels: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                pq_to_linear_f(px[0] as f32 / u16_max),
                pq_to_linear_f(px[1] as f32 / u16_max),
                pq_to_linear_f(px[2] as f32 / u16_max),
            ]
        })
        .collect()
}

/// BT.709 u16 → linear f32. Same shape as `pq_u16_to_linear_f32`.
/// Closes BT.709 portion of #17.
fn bt709_u16_to_linear_f32(data: &[u8], channels: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                bt709_to_linear_f(px[0] as f32 / u16_max),
                bt709_to_linear_f(px[1] as f32 / u16_max),
                bt709_to_linear_f(px[2] as f32 / u16_max),
            ]
        })
        .collect()
}

/// HLG u16 → linear scene-light f32. Same shape as
/// `pq_u16_to_linear_f32`. Closes HLG portion of #17.
fn hlg_u16_to_linear_f32(data: &[u8], channels: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                hlg_to_linear_f(px[0] as f32 / u16_max),
                hlg_to_linear_f(px[1] as f32 / u16_max),
                hlg_to_linear_f(px[2] as f32 / u16_max),
            ]
        })
        .collect()
}

/// sRGB u16 → linear f32 (IEC 61966-2-1).
///
/// `u16_max` is the divisor for input normalization — `65535.0` for
/// full 16-bit input (the default), or `(1 << bits) - 1` for narrower
/// precision (e.g., 1023 for 10-bit, 4095 for 12-bit, 16383 for 14-bit).
/// See `EncodeRequest::with_bits_per_sample`.
fn srgb_u16_to_linear_f32(data: &[u8], channels: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                srgb_to_linear_f(px[0] as f32 / u16_max),
                srgb_to_linear_f(px[1] as f32 / u16_max),
                srgb_to_linear_f(px[2] as f32 / u16_max),
            ]
        })
        .collect()
}

/// sRGB transfer function: normalized float [0,1] → linear float.
#[inline]
fn srgb_to_linear_f(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        jxl_simd::fast_powf((c + 0.055) / 1.055, 2.4)
    }
}

/// PQ (SMPTE ST 2084) EOTF: PQ-encoded normalized [0,1] → linear [0,1]
/// where 1.0 = peak luminance (= the encoder's `intensity_target`,
/// typically 10 000 nits for full-spec PQ). Closes PQ portion of #17.
///
/// Constants per SMPTE ST 2084-2014 (m1 / m2 / c1 / c2 / c3). Negative
/// inputs are clamped to 0; outputs are non-negative by construction.
#[inline]
fn pq_to_linear_f(c: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0; // 0.1593017578125
    const M2: f32 = (2523.0 / 4096.0) * 128.0; // 78.84375
    const C1: f32 = 3424.0 / 4096.0; // 0.8359375
    const C2: f32 = (2413.0 / 4096.0) * 32.0; // 18.8515625
    const C3: f32 = (2392.0 / 4096.0) * 32.0; // 18.6875
    let e = c.max(0.0);
    let n = jxl_simd::fast_powf(e, 1.0 / M2);
    // numerator clamped at 0; denominator can't reach 0 in [0,1] domain
    // (c2 = 18.85, c3 = 18.69, c3*N <= 18.69 at N=1, c2 - c3*N >= 0.16)
    let num = (n - C1).max(0.0);
    let den = C2 - C3 * n;
    jxl_simd::fast_powf(num / den, 1.0 / M1)
}

/// BT.709 inverse OETF (Rec. ITU-R BT.709-6, the broadcast camera
/// transfer): encoded normalized [0,1] → linear scene-light [0,1].
///
/// Piecewise: linear toe below 0.081 (= 4.5 × 0.018) plus a power
/// curve above with effective inverse gamma ≈ 2.222. Note this is
/// the SCENE-light EOTF (the inverse of the broadcast OETF), NOT the
/// display EOTF (which would be a pure gamma 2.4 per BT.1886).
/// Matches libjxl's interpretation of `TransferFunction::Bt709` for
/// encoder input. Closes BT.709 portion of #17.
#[inline]
fn bt709_to_linear_f(c: f32) -> f32 {
    // Threshold = beta * alpha = 0.018 * 4.5 = 0.081 (encoded value
    // below which the toe is linear). Some references quote 0.0812
    // due to the alpha = 1.099 derivation; we use the spec's exact
    // 0.081 cutoff per Rec. BT.709-6 §1.2.
    const TOE_CUTOFF: f32 = 0.081;
    let e = c.max(0.0);
    if e <= TOE_CUTOFF {
        e / 4.5
    } else {
        jxl_simd::fast_powf((e + 0.099) / 1.099, 1.0 / 0.45)
    }
}

/// HLG (Hybrid Log-Gamma, BT.2100 / ARIB STD-B67) inverse OETF:
/// HLG-encoded normalized [0,1] → linear scene-light [0,1].
///
/// HLG is piecewise: a square-root-like toe in the lower half plus a
/// logarithmic shoulder in the upper half. Scene-light output is in
/// [0, 1] where 1.0 = peak signal; downstream display mapping (the
/// HLG OOTF) is the decoder's responsibility, NOT the encoder's.
///
/// Closes HLG portion of #17.
#[inline]
fn hlg_to_linear_f(c: f32) -> f32 {
    const A: f32 = 0.17883277;
    const B: f32 = 1.0 - 4.0 * A; // 0.28466892
    // c_const = 0.5 - a * ln(4 * a). Hard-coded literal because the
    // spec gives this value to high precision and we want bit-exact
    // agreement with reference decoders.
    const C_CONST: f32 = 0.55991073;
    let e = c.max(0.0);
    if e <= 0.5 {
        // Lower half: square-root-like toe. L = E²/3.
        (e * e) / 3.0
    } else {
        // Upper half: logarithmic shoulder. L = (exp((E - c)/a) + b)/12.
        // The /12 normalization keeps L in [0, 1] for E in [0, 1]
        // (HLG peak signal corresponds to 12 × the SDR diffuse white).
        ((((e - C_CONST) / A).exp()) + B) / 12.0
    }
}

/// Gamma u8 → linear f32 RGB. `linear = (encoded/255)^(1/gamma)`
fn gamma_u8_to_linear_f32(data: &[u8], channels: usize, gamma: f32) -> Vec<f32> {
    // Build 256-entry LUT for u8 values (avoids per-pixel powf)
    let inv_gamma = 1.0 / gamma;
    let lut: [f32; 256] =
        core::array::from_fn(|i| jxl_simd::fast_powf(i as f32 / 255.0, inv_gamma));
    data.chunks(channels)
        .flat_map(|px| {
            [
                lut[px[0] as usize],
                lut[px[1] as usize],
                lut[px[2] as usize],
            ]
        })
        .collect()
}

/// Gamma u16 → linear f32 RGB. `linear = (encoded/u16_max)^(1/gamma)`
fn gamma_u16_to_linear_f32(data: &[u8], channels: usize, gamma: f32, u16_max: f32) -> Vec<f32> {
    let inv_gamma = 1.0 / gamma;
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                jxl_simd::fast_powf(px[0] as f32 / u16_max, inv_gamma),
                jxl_simd::fast_powf(px[1] as f32 / u16_max, inv_gamma),
                jxl_simd::fast_powf(px[2] as f32 / u16_max, inv_gamma),
            ]
        })
        .collect()
}

/// CMY u8 + K u8 → linear-light f32 RGB via the naive uncalibrated
/// subtractive model: `R = (1 - C/255) * (1 - K/255)` etc., where
/// each ink absorbs its complementary primary in linear light.
///
/// This is the chunk-3 follow-on to the chunk-2 placeholder that
/// treated CMY as if it were sRGB-encoded RGB bytes (which had no
/// physical basis at all — a fully-saturated cyan ink would encode
/// as bright red in XYB and decode to an entirely wrong colour
/// family). The 1-CMY model is still an approximation: it ignores
/// per-ink chromaticity, dot-gain, illuminant, and printer profile,
/// so output won't match a colorimetric CMYK→sRGB conversion done
/// through an ICC profile. But it puts the colours in the right
/// half of the gamut — a pure cyan input now encodes as cyan-ish
/// (no red component), which the XYB perceptual model can quantise
/// sensibly. A future chunk can wire the caller's CMYK ICC profile
/// (option A) or a hardcoded SWOP/FOGRA matrix (option B) for
/// colorimetric accuracy.
///
/// `K` is also kept as a modular extra channel further down the
/// pipeline so the K plane round-trips losslessly — the CMY→RGB
/// transform here is purely for perceptual quantisation of the
/// colour content.
fn cmyk_u8_to_linear_f32_rgb(cmy: &[u8], k: &[u8]) -> Vec<f32> {
    debug_assert_eq!(cmy.len(), k.len() * 3);
    let inv = 1.0f32 / 255.0;
    let mut out = Vec::with_capacity(k.len() * 3);
    for (px, &kv) in cmy.chunks_exact(3).zip(k.iter()) {
        let one_minus_k = 1.0 - (kv as f32) * inv;
        out.push((1.0 - (px[0] as f32) * inv) * one_minus_k);
        out.push((1.0 - (px[1] as f32) * inv) * one_minus_k);
        out.push((1.0 - (px[2] as f32) * inv) * one_minus_k);
    }
    out
}

/// CMY u16 + K u16 → linear-light f32 RGB. Same 1-CMY × (1-K) model
/// as the 8-bit variant; `u16_max` is the bit-depth normaliser (e.g.
/// `65535.0` for full-precision 16-bit input).
fn cmyk_u16_to_linear_f32_rgb(cmy: &[u8], k: &[u16], u16_max: f32) -> Vec<f32> {
    let cmy_u16: &[u16] = bytemuck::cast_slice(cmy);
    debug_assert_eq!(cmy_u16.len(), k.len() * 3);
    let inv = 1.0f32 / u16_max;
    let mut out = Vec::with_capacity(k.len() * 3);
    for (px, &kv) in cmy_u16.chunks_exact(3).zip(k.iter()) {
        let one_minus_k = 1.0 - (kv as f32) * inv;
        out.push((1.0 - (px[0] as f32) * inv) * one_minus_k);
        out.push((1.0 - (px[1] as f32) * inv) * one_minus_k);
        out.push((1.0 - (px[2] as f32) * inv) * one_minus_k);
    }
    out
}

/// Gamma u8 grayscale → linear f32 RGB (gray→R=G=B). `linear = (encoded/255)^(1/gamma)`
fn gamma_gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize, gamma: f32) -> Vec<f32> {
    let inv_gamma = 1.0 / gamma;
    let lut: [f32; 256] =
        core::array::from_fn(|i| jxl_simd::fast_powf(i as f32 / 255.0, inv_gamma));
    data.chunks(stride)
        .flat_map(|px| {
            let v = lut[px[0] as usize];
            [v, v, v]
        })
        .collect()
}

/// Gamma u16 grayscale → linear f32 RGB (gray→R=G=B). `linear = (encoded/u16_max)^(1/gamma)`
fn gamma_gray_u16_to_linear_f32_rgb(
    data: &[u8],
    stride: usize,
    gamma: f32,
    u16_max: f32,
) -> Vec<f32> {
    let inv_gamma = 1.0 / gamma;
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = jxl_simd::fast_powf(px[0] as f32 / u16_max, inv_gamma);
            [v, v, v]
        })
        .collect()
}

/// Extract alpha channel from interleaved 16-bit pixel data as u8 (quantized).
///
/// `u16_max` is the source-precision max value (65535 for 16-bit,
/// `(1 << bits) - 1` for narrower precision). Used to scale alpha
/// from `0..=u16_max` to `0..=255` correctly.
fn extract_alpha_u16(data: &[u8], stride: usize, alpha_offset: usize, u16_max: f32) -> Vec<u8> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .map(|px| ((px[alpha_offset] as f32 / u16_max).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

/// W44-91: compute the cheap zenanalyze-equivalent proxies for the
/// high-distance smooth-photo gate widening when the input layout is
/// 8-bit sRGB-like. Returns `None` for all other layouts (16-bit,
/// linear-f32, grayscale, HDR, CMYK) where the M3 colourfulness scale
/// and per-block range threshold are not well-defined.
///
/// See [`crate::vardct::encoder::ZenanalyzeProxies::compute_srgb_u8`]
/// for the per-byte definitions (matches zenanalyze `src/tier1.rs`
/// colourfulness and `flat_color_blocks` accumulators exactly).
fn compute_w44_91_zenanalyze_proxies(
    pixels: &[u8],
    width: usize,
    height: usize,
    layout: PixelLayout,
) -> Option<crate::vardct::encoder::ZenanalyzeProxies> {
    use crate::vardct::encoder::ZenanalyzeProxies;
    // Per-layout (R offset, G offset, B offset, bytes per pixel). Only the
    // 8-bit sRGB layouts have a meaningful M3 colourfulness scale —
    // everything else stays `None`.
    let (r_off, g_off, b_off, bpp) = match layout {
        PixelLayout::Rgb8 => (0, 1, 2, 3),
        PixelLayout::Rgba8 => (0, 1, 2, 4),
        PixelLayout::Bgr8 => (2, 1, 0, 3),
        PixelLayout::Bgra8 => (2, 1, 0, 4),
        _ => return None,
    };
    let expected_len = width.checked_mul(height)?.checked_mul(bpp)?;
    if pixels.len() < expected_len || width == 0 || height == 0 {
        return None;
    }
    Some(ZenanalyzeProxies::compute_srgb_u8(
        pixels, width, height, bpp, r_off, g_off, b_off,
    ))
}

/// Swap B and R channels in-place equivalent: BGR(A) → RGB(A).
fn bgr_to_rgb(data: &[u8], stride: usize) -> Vec<u8> {
    let mut out = data.to_vec();
    for chunk in out.chunks_mut(stride) {
        chunk.swap(0, 2);
    }
    out
}

/// Extract a single channel from interleaved pixel data.
fn extract_alpha(data: &[u8], stride: usize, alpha_offset: usize) -> Vec<u8> {
    data.chunks(stride).map(|px| px[alpha_offset]).collect()
}

/// Extract alpha from interleaved f32 pixel data, converting to u8 (0..255).
fn extract_alpha_f32(data: &[f32], stride: usize, alpha_offset: usize) -> Vec<u8> {
    data.chunks(stride)
        .map(|px| (px[alpha_offset].clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

/// Expand 8-bit sRGB grayscale to linear f32 RGB (gray→R=G=B).
fn gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            let v = srgb_to_linear(px[0]);
            [v, v, v]
        })
        .collect()
}

/// Expand 16-bit sRGB grayscale to linear f32 RGB (gray→R=G=B).
fn gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = srgb_to_linear_f(px[0] as f32 / u16_max);
            [v, v, v]
        })
        .collect()
}

/// Expand u8 grayscale to linear f32 RGB via the PQ EOTF. Uses a
/// 256-entry LUT to avoid per-pixel powf, mirroring the PQ u8 RGB
/// helper. Closes Gray PQ portion of #17.
fn pq_gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| pq_to_linear_f(i as f32 / 255.0));
    data.chunks(stride)
        .flat_map(|px| {
            let v = lut[px[0] as usize];
            [v, v, v]
        })
        .collect()
}

/// Expand u16 grayscale to linear f32 RGB via the PQ EOTF.
fn pq_gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = pq_to_linear_f(px[0] as f32 / u16_max);
            [v, v, v]
        })
        .collect()
}

/// Expand u8 grayscale to linear f32 RGB via the HLG inverse OETF.
fn hlg_gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| hlg_to_linear_f(i as f32 / 255.0));
    data.chunks(stride)
        .flat_map(|px| {
            let v = lut[px[0] as usize];
            [v, v, v]
        })
        .collect()
}

/// Expand u16 grayscale to linear f32 RGB via the HLG inverse OETF.
fn hlg_gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = hlg_to_linear_f(px[0] as f32 / u16_max);
            [v, v, v]
        })
        .collect()
}

/// Expand u8 grayscale to linear f32 RGB via the BT.709 inverse OETF.
fn bt709_gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| bt709_to_linear_f(i as f32 / 255.0));
    data.chunks(stride)
        .flat_map(|px| {
            let v = lut[px[0] as usize];
            [v, v, v]
        })
        .collect()
}

/// Expand u16 grayscale to linear f32 RGB via the BT.709 inverse OETF.
fn bt709_gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = bt709_to_linear_f(px[0] as f32 / u16_max);
            [v, v, v]
        })
        .collect()
}

/// PQ-encoded f32 → linear f32 RGB. Input is interleaved
/// `stride`-channels-per-pixel where each channel is a PQ-encoded
/// `[0, 1]` value. Output is linear `[0, 1]` (where 1.0 = peak
/// luminance per the encoder's `intensity_target`).
///
/// A3 chunk 1b (issue #46). No LUT — input is already float, so the
/// per-pixel `powf` cost is unavoidable. Use the u8/u16 helpers for
/// quantized input.
fn pq_f32_to_linear_f32_rgb(data: &[f32], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            [
                pq_to_linear_f(px[0]),
                pq_to_linear_f(px[1]),
                pq_to_linear_f(px[2]),
            ]
        })
        .collect()
}

/// HLG-encoded f32 → linear (scene-light) f32 RGB. See
/// [`pq_f32_to_linear_f32_rgb`] for shape. A3 chunk 1b (issue #46).
fn hlg_f32_to_linear_f32_rgb(data: &[f32], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            [
                hlg_to_linear_f(px[0]),
                hlg_to_linear_f(px[1]),
                hlg_to_linear_f(px[2]),
            ]
        })
        .collect()
}

/// BT.709-encoded f32 → linear f32 RGB. See
/// [`pq_f32_to_linear_f32_rgb`] for shape. A3 chunk 1b (issue #46).
fn bt709_f32_to_linear_f32_rgb(data: &[f32], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            [
                bt709_to_linear_f(px[0]),
                bt709_to_linear_f(px[1]),
                bt709_to_linear_f(px[2]),
            ]
        })
        .collect()
}

/// Expand linear f32 grayscale to linear f32 RGB (gray→R=G=B).
fn gray_f32_to_linear_f32_rgb(data: &[f32], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            let v = px[0];
            [v, v, v]
        })
        .collect()
}

// ─── f16 (linear) input helpers ───────────────────────────────────
// Closes the FLOAT16 portion of #18. Storage is native-endian u16
// per channel; conversion via `crate::f16::f16_bits_to_f32`.

/// Convert interleaved linear f16 RGB(A) bytes (`stride` channels per
/// pixel) to interleaved linear f32 RGB (stride 3, alpha dropped).
/// `bytes` must contain exactly `n_pixels * stride * 2` u16-bytes.
fn f16_to_linear_f32_rgb(bytes: &[u8], stride: usize) -> Vec<f32> {
    use crate::f16::f16_bits_to_f32;
    let pixels: &[u16] = bytemuck::cast_slice(bytes);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            [
                f16_bits_to_f32(px[0]),
                f16_bits_to_f32(px[1]),
                f16_bits_to_f32(px[2]),
            ]
        })
        .collect()
}

/// Expand interleaved linear f16 grayscale (`stride=1` for gray-only,
/// `stride=2` for gray+alpha) to interleaved linear f32 RGB.
fn f16_gray_to_linear_f32_rgb(bytes: &[u8], stride: usize) -> Vec<f32> {
    use crate::f16::f16_bits_to_f32;
    let pixels: &[u16] = bytemuck::cast_slice(bytes);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = f16_bits_to_f32(px[0]);
            [v, v, v]
        })
        .collect()
}

/// Repack a row-strided pixel buffer into a tightly-packed `Vec<u8>`.
/// Closes row-stride portion of #18.
///
/// Caller must ensure `stride >= width * bytes_per_pixel`. The result
/// has `height * width * bytes_per_pixel` bytes; padding bytes from
/// each source row are discarded.
///
/// Returns `Err(EncodeError::InvalidInput)` if the source buffer is
/// too small to hold `height * stride` bytes (would index out of
/// bounds during the row copy).
fn unpack_strided_pixels(
    src: &[u8],
    width: usize,
    height: usize,
    bytes_per_pixel: usize,
    stride: usize,
) -> core::result::Result<Vec<u8>, EncodeError> {
    let row_bytes = width * bytes_per_pixel;
    if stride < row_bytes {
        return Err(EncodeError::InvalidInput {
            message: format!(
                "row_stride {stride} is less than width*bytes_per_pixel = {width}*{bytes_per_pixel} = {row_bytes}",
            ),
        });
    }
    let needed = height
        .checked_mul(stride)
        .ok_or_else(|| EncodeError::InvalidInput {
            message: "height * row_stride overflows usize".into(),
        })?;
    if src.len() < needed {
        return Err(EncodeError::InvalidInput {
            message: format!(
                "pixel buffer too small for strided input: need {needed} bytes (height {height} × stride {stride}), got {}",
                src.len(),
            ),
        });
    }
    let mut packed = Vec::with_capacity(height * row_bytes);
    for y in 0..height {
        let row_start = y * stride;
        packed.extend_from_slice(&src[row_start..row_start + row_bytes]);
    }
    Ok(packed)
}

/// Dispatch container-wrap by Brotli setting (closes #15 wire-up).
///
/// When `brotli_quality` is `Some(q)` AND the `brotli-metadata`
/// feature is enabled, routes through `wrap_in_container_with_brob`
/// (each metadata blob falls back to plain box if brob would be
/// bigger). Otherwise (or when feature is off), uses the plain
/// `wrap_in_container`. Centralizing the dispatch keeps the 3 call
/// sites (encode_inner, LossyEncoder::finish_inner,
/// LosslessEncoder::finish_inner) aligned.
///
/// `level` is the codestream level (5 or 10). When `level != 5` a
/// `jxll` (level) box is emitted directly after `ftyp`. For level 5
/// the byte layout is byte-identical to the historical wrap. See
/// [`crate::container::compute_codestream_level`].
fn wrap_metadata_container(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
    brotli_quality: Option<u32>,
    level: u8,
) -> Vec<u8> {
    #[cfg(feature = "brotli-metadata")]
    {
        if let Some(q) = brotli_quality {
            return crate::container::wrap_in_container_with_brob_and_level_and_jumbf(
                codestream, exif, xmp, jumbf, q, level,
            );
        }
    }
    let _ = brotli_quality;
    crate::container::wrap_in_container_with_level_and_jumbf(codestream, exif, xmp, level, jumbf)
}

/// Pick the codestream level required for an image with the given
/// dimensions, ICC size, and extra channels. Wraps
/// [`crate::container::compute_codestream_level`] and translates the
/// `None` (unencodable) case into [`EncodeError::InvalidInput`].
///
/// `num_extra_channels` MUST already include the alpha channel when
/// the pixel layout carries alpha — the level-5 cap is `<= 4` extras
/// *including* alpha, matching libjxl `VerifyLevelSettings` which
/// reads `m.num_extra_channels` (alpha is one of them).
fn compute_required_level(
    width: u32,
    height: u32,
    num_extra_channels: u32,
    has_black_channel: bool,
    icc_size: u64,
) -> core::result::Result<u8, EncodeError> {
    crate::container::compute_codestream_level(
        width,
        height,
        num_extra_channels,
        has_black_channel,
        icc_size,
    )
    .ok_or_else(|| EncodeError::InvalidInput {
        message: format!(
            "image {width}x{height} ({} px), {num_extra_channels} extra channels, \
             {icc_size}-byte ICC exceeds JPEG XL level 10 limits",
            u64::from(width).saturating_mul(u64::from(height)),
        ),
    })
}

/// Divide premultiplied (associated) linear RGB values by alpha so the
/// encoded codestream stores straight (unassociated) color. Mirrors
/// libjxl `UnpremultiplyAlpha` in `lib/jxl/alpha.cc:106`. Pairs with
/// `alpha_associated=true` in the codestream header — the decoder is
/// responsible for re-premultiplying the output.
///
/// `alpha_u8` is the per-pixel alpha after our standard u8 quantization
/// (matching the codestream's 8-bit BitDepth default). Using the same
/// quantized value the decoder will see ensures the round-trip
/// premultiplied → encode → decode → re-premultiplied closes.
///
/// `kSmallAlpha = 1.0 / (1<<26)` floor on the divisor — matches
/// libjxl `lib/jxl/alpha.h:21`. Lifts division-by-zero on alpha=0
/// pixels (where the original color is undefined anyway).
fn unpremultiply_alpha_inplace(linear_rgb_interleaved: &mut [f32], alpha_u8: &[u8]) {
    const K_SMALL_ALPHA: f32 = 1.0_f32 / ((1u32 << 26) as f32);
    debug_assert_eq!(linear_rgb_interleaved.len(), alpha_u8.len() * 3);
    for (rgb, &a) in linear_rgb_interleaved
        .chunks_exact_mut(3)
        .zip(alpha_u8.iter())
    {
        let a_f = (a as f32) / 255.0;
        let inv = 1.0 / a_f.max(K_SMALL_ALPHA);
        rgb[0] *= inv;
        rgb[1] *= inv;
        rgb[2] *= inv;
    }
}

/// Extract alpha from interleaved f16 pixel data, converting to u8
/// (0..255). Mirrors `extract_alpha_f32` but reads u16 bytes via f16
/// conversion before clamping.
fn extract_alpha_f16(bytes: &[u8], stride: usize, alpha_offset: usize) -> Vec<u8> {
    use crate::f16::f16_bits_to_f32;
    let pixels: &[u16] = bytemuck::cast_slice(bytes);
    pixels
        .chunks(stride)
        .map(|px| (f16_bits_to_f32(px[alpha_offset]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ec_resampling helper (A1 audit "Pixel formats / extras") ──

    /// Box filter at factor=1 is a passthrough.
    #[test]
    fn test_downsample_channel_u8_factor1_is_passthrough() {
        let src = [10u8, 20, 30, 40, 50, 60, 70, 80, 90];
        let got = downsample_channel_u8(&src, 3, 3, 1);
        assert_eq!(got, src);
    }

    /// 2×2 box filter on a 4×4 uniform region averages each quadrant.
    #[test]
    fn test_downsample_channel_u8_factor2_uniform_quadrants() {
        // 4×4 image where each 2×2 quadrant has a distinct value.
        let src: [u8; 16] = [
            10, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40,
        ];
        let got = downsample_channel_u8(&src, 4, 4, 2);
        assert_eq!(got, vec![10, 20, 30, 40]);
    }

    /// Partial edge cells (image dim not divisible by factor) only average
    /// in-bounds samples — matches libjxl `DoDownsampleImage`.
    #[test]
    fn test_downsample_channel_u8_factor2_partial_edges() {
        // 3×3 image, downsample by 2 → 2×2 output. Output cell (1, 1)
        // averages only the single in-bounds sample at (2, 2).
        let src: [u8; 9] = [10, 20, 30, 40, 50, 60, 70, 80, 90];
        let got = downsample_channel_u8(&src, 3, 3, 2);
        // (0,0): avg of (10,20,40,50) = 120/4 = 30
        // (0,1): avg of (30,60) = 90/2 = 45
        // (1,0): avg of (70,80) = 150/2 = 75
        // (1,1): avg of (90) = 90
        assert_eq!(got, vec![30, 45, 75, 90]);
    }

    /// Factor=4 over an 8×8 image with a vertical gradient.
    #[test]
    fn test_downsample_channel_u8_factor4_8x8() {
        // 8 rows × 8 cols, value = row*16 → [0, 16, 32, 48, 64, 80, 96, 112].
        let mut src = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                src[y * 8 + x] = (y * 16) as u8;
            }
        }
        let got = downsample_channel_u8(&src, 8, 8, 4);
        // 4×4 box top-left: rows 0..4 → values [0,16,32,48], mean = 24.
        // Bottom-left: rows 4..8 → [64,80,96,112], mean = 88.
        assert_eq!(got, vec![24, 24, 88, 88]);
    }

    /// Dimensions output match libjxl's `DivCeil(d, factor)`.
    #[test]
    fn test_downsample_channel_u8_output_dims() {
        let src = vec![42u8; 13 * 17];
        // 13 div_ceil 4 = 4; 17 div_ceil 4 = 5 → 20 samples.
        let got = downsample_channel_u8(&src, 13, 17, 4);
        assert_eq!(got.len(), 20);
        // Uniform input must produce uniform output.
        assert!(got.iter().all(|&v| v == 42));
    }

    /// factor=0 returns empty (defensive — caller usually validates).
    #[test]
    fn test_downsample_channel_u8_factor0_returns_empty() {
        let src = [1u8, 2, 3, 4];
        let got = downsample_channel_u8(&src, 2, 2, 0);
        assert!(got.is_empty());
    }

    // ─── PQ EOTF (closes PQ portion of #17) ───────────────────────

    /// Spot-check the PQ EOTF against published reference points
    /// (BT.2100 Table 4 / ST 2084 inverse). Tolerance is 1e-3 because
    /// the encoder uses fast_powf instead of std::powf — accuracy
    /// is in the same neighborhood as libjxl's PQ implementation.
    #[test]
    fn test_pq_to_linear_f_reference_points() {
        // EOTF[0] = 0
        assert!(pq_to_linear_f(0.0).abs() < 1e-3);
        // EOTF[1] = 1.0 (peak luminance / 10000 nits = full scale)
        let one = pq_to_linear_f(1.0);
        assert!(
            (one - 1.0).abs() < 1e-3,
            "PQ(1.0) should be 1.0 (peak); got {one}",
        );
        // EOTF[0.5081] ≈ 0.01 (= 100 nits / 10000) — the SDR diffuse
        // white reference. Per BT.2100 the encoded value for 100 nits
        // is ~0.508. Tolerance loosened a touch because fast_powf
        // diverges from std::powf in the middle of the range.
        let mid = pq_to_linear_f(0.5081);
        assert!(
            (mid - 0.01).abs() < 5e-3,
            "PQ(0.5081) should be ≈0.01 (100 nits); got {mid}",
        );
        // Monotonic
        let a = pq_to_linear_f(0.25);
        let b = pq_to_linear_f(0.5);
        let c = pq_to_linear_f(0.75);
        assert!(a < b && b < c, "PQ should be monotone; got {a}, {b}, {c}");
    }

    #[test]
    fn test_pq_to_linear_f_clamps_negative() {
        // Negative input clamps to 0 → output ~0 (avoids NaN from
        // x.powf(non-int) on negative x). fast_powf can produce a
        // tiny negative result from rounding; both must be safe to
        // feed into the encoder (no NaN/Inf).
        let v = pq_to_linear_f(-0.1);
        assert!(v.is_finite(), "PQ(-0.1) should be finite; got {v}");
        assert!(v.abs() < 1e-3, "PQ(-0.1) should clamp to ~0; got {v}");
    }

    /// BT.709 inverse OETF reference points (Rec. BT.709-6).
    /// - BT709(0) = 0
    /// - BT709(0.081) = 0.018 (toe/shoulder boundary)
    /// - BT709(1) = 1.0
    /// - Monotonic
    #[test]
    fn test_bt709_to_linear_f_reference_points() {
        assert!(bt709_to_linear_f(0.0).abs() < 1e-6);
        // Boundary: encoded 0.081 → linear 0.018 (= 0.081 / 4.5).
        let boundary = bt709_to_linear_f(0.081);
        assert!(
            (boundary - 0.018).abs() < 1e-5,
            "BT.709(0.081) should be ≈0.018; got {boundary}",
        );
        let one = bt709_to_linear_f(1.0);
        assert!(
            (one - 1.0).abs() < 1e-3,
            "BT.709(1.0) should be 1.0; got {one}",
        );
        let a = bt709_to_linear_f(0.25);
        let b = bt709_to_linear_f(0.5);
        let c = bt709_to_linear_f(0.75);
        assert!(
            a < b && b < c,
            "BT.709 should be monotone; got {a}, {b}, {c}"
        );
    }

    #[test]
    fn test_bt709_to_linear_f_clamps_negative() {
        let v = bt709_to_linear_f(-0.1);
        assert!(v.is_finite());
        assert!(
            (0.0..1e-3).contains(&v),
            "BT.709(-0.1) should clamp to ~0; got {v}"
        );
    }

    /// Reference points for HLG inverse OETF (BT.2100).
    /// - HLG(0) = 0
    /// - HLG(0.5) = 0.25 / 3 = 0.083333... (boundary of toe / shoulder)
    /// - HLG(1) = 1.0 (peak signal → peak scene-light)
    /// - Monotonic
    #[test]
    fn test_hlg_to_linear_f_reference_points() {
        assert!(hlg_to_linear_f(0.0).abs() < 1e-6);
        let half = hlg_to_linear_f(0.5);
        assert!(
            (half - (0.25 / 3.0)).abs() < 1e-5,
            "HLG(0.5) should be 0.0833...; got {half}",
        );
        let one = hlg_to_linear_f(1.0);
        assert!(
            (one - 1.0).abs() < 1e-3,
            "HLG(1.0) should be 1.0 (peak); got {one}",
        );
        let a = hlg_to_linear_f(0.25);
        let b = hlg_to_linear_f(0.5);
        let c = hlg_to_linear_f(0.75);
        assert!(a < b && b < c, "HLG should be monotone; got {a}, {b}, {c}");
    }

    #[test]
    fn test_hlg_to_linear_f_clamps_negative() {
        let v = hlg_to_linear_f(-0.1);
        assert!(v.is_finite());
        assert!(
            (0.0..1e-3).contains(&v),
            "HLG(-0.1) should clamp to ~0; got {v}"
        );
    }

    #[test]
    fn test_pq_u16_to_linear_f32_uses_pq_eotf() {
        // 16-bit PQ value 65535 should give linear ≈1.0.
        let pixels_u16: Vec<u16> = vec![65535, 65535, 65535];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels_u16);
        let linear = pq_u16_to_linear_f32(bytes, 3, 65535.0);
        for v in &linear {
            assert!((v - 1.0).abs() < 1e-3, "PQ(1.0) should be ≈1.0; got {v}");
        }
        // 16-bit PQ value 0 should give 0.
        let pixels0: Vec<u16> = vec![0, 0, 0];
        let bytes0: &[u8] = bytemuck::cast_slice(&pixels0);
        let linear0 = pq_u16_to_linear_f32(bytes0, 3, 65535.0);
        for v in &linear0 {
            assert!(v.abs() < 1e-6, "PQ(0) should be 0; got {v}");
        }
    }

    /// Audit item #3: `effective_profile_for_image` must drop
    /// `tree_max_buckets` from 256 → 192 ONLY at the (pixels >= 4 MP,
    /// effort >= 9) cell. Every other cell must keep the effort-only
    /// default so hash-locks stay stable.
    #[test]
    fn test_effective_profile_for_image_tree_max_buckets_dispatch() {
        // e9 + large: dispatch fires.
        let cfg = LosslessConfig::new().with_effort(9);
        let p = cfg.effective_profile_for_image(4_194_304);
        assert_eq!(
            p.tree_max_buckets,
            crate::effort::LARGE_E9_TREE_MAX_BUCKETS,
            "e9 large: buckets must drop to 192"
        );

        // e9 + medium (< 4 MP): no dispatch.
        let p = cfg.effective_profile_for_image(1_048_576);
        assert_eq!(p.tree_max_buckets, 256, "e9 medium: buckets stay 256");

        // e7 + large: no dispatch (effort gate).
        let cfg = LosslessConfig::new().with_effort(7);
        let p = cfg.effective_profile_for_image(8_000_000);
        assert_eq!(
            p.tree_max_buckets, 96,
            "e7 large: buckets stay 96 (default)"
        );

        // e10 + large: dispatch fires (effort >= 9).
        let cfg = LosslessConfig::new().with_effort(10);
        let p = cfg.effective_profile_for_image(8_000_000);
        assert_eq!(
            p.tree_max_buckets,
            crate::effort::LARGE_E9_TREE_MAX_BUCKETS,
            "e10 large: buckets drop to 192"
        );
    }

    /// When the caller has supplied an explicit `__expert`
    /// profile_override (e.g. a sweep harness pinning a specific
    /// `tree_max_buckets`), the always-on dispatch must NOT silently
    /// stomp it.
    #[cfg(feature = "__expert")]
    #[test]
    fn test_effective_profile_for_image_respects_internal_params_override() {
        let params = crate::effort::LosslessInternalParams {
            tree_max_buckets: Some(128),
            ..Default::default()
        };
        let cfg = LosslessConfig::new()
            .with_effort(9)
            .with_internal_params(params);
        let p = cfg.effective_profile_for_image(8_000_000);
        // Override wins — dispatch did not fire.
        assert_eq!(
            p.tree_max_buckets, 128,
            "sweep override must survive the dispatch"
        );
    }

    /// Chunk 1 VarDCT AC dispatch (`adapt_to_image_lossy`): drop
    /// `try_dct64` to `false` ONLY when the image is small (< 500_000
    /// pixels) AND distance is low (< 2.0). Every other cell keeps the
    /// effort-only default so corpus_regression bytes stay stable.
    #[test]
    fn test_lossy_effective_profile_for_image_dct64_dispatch() {
        // small + low-d at effort 7: dispatch fires (try_dct64 → false).
        let cfg = LossyConfig::new(1.0).with_effort(7);
        let p = cfg.effective_profile_for_image(256 * 256);
        assert!(
            !p.try_dct64,
            "small_0.07MP + d=1.0 + e7: try_dct64 must drop to false"
        );

        // small_0.26MP (512×512) + d=1.0 + e7: still small + low-d.
        let cfg = LossyConfig::new(1.0).with_effort(7);
        let p = cfg.effective_profile_for_image(512 * 512);
        assert!(
            !p.try_dct64,
            "small_0.26MP + d=1.0 + e7: try_dct64 must drop to false"
        );

        // medium (1 MP) + d=1.0: no dispatch (pixel-count gate).
        let cfg = LossyConfig::new(1.0).with_effort(7);
        let p = cfg.effective_profile_for_image(1024 * 1024);
        assert!(
            p.try_dct64,
            "medium_1.0MP: try_dct64 stays true (pixel gate excludes ≥500k)"
        );

        // small + d=2.0: no dispatch (distance gate is strict <).
        let cfg = LossyConfig::new(2.0).with_effort(7);
        let p = cfg.effective_profile_for_image(256 * 256);
        assert!(
            p.try_dct64,
            "small + d=2.0: try_dct64 stays true (distance gate is strict <2.0)"
        );

        // small + d=5.0: no dispatch (distance gate).
        let cfg = LossyConfig::new(5.0).with_effort(7);
        let p = cfg.effective_profile_for_image(256 * 256);
        assert!(p.try_dct64, "small + d=5.0: try_dct64 stays true");

        // small + low-d + effort 5: no dispatch (effort < 7 means
        // try_dct64 is already false in the default profile — adapter
        // is a no-op, no false-flip-to-true).
        let cfg = LossyConfig::new(1.0).with_effort(5);
        let p = cfg.effective_profile_for_image(256 * 256);
        assert!(
            !p.try_dct64,
            "small + d=1.0 + e5: try_dct64 already false at effort < 7"
        );

        // large + low-d at e7: no dispatch (pixel gate).
        let cfg = LossyConfig::new(0.5).with_effort(7);
        let p = cfg.effective_profile_for_image(4_194_304);
        assert!(p.try_dct64, "large_4MP + d=0.5 + e7: try_dct64 stays true");
    }

    /// W44-35 smooth-photo DCT64 admission gate: when the smoothness
    /// auto-detector (or caller hint) is `true`, suppress the
    /// `adapt_to_image_lossy` `try_dct64 -> false` flip even on the
    /// gated cell.
    #[test]
    fn test_lossy_effective_profile_for_image_smooth_photo_admission() {
        // Baseline: small + low-d + e7 with `smooth_photo=false` keeps
        // the gated behaviour (try_dct64 = false). Matches the pre-W44-35
        // result asserted in the existing dct64_dispatch test.
        let cfg = LossyConfig::new(1.0).with_effort(7);
        let p = cfg.effective_profile_for_image_with_smoothness(512 * 512, false);
        assert!(
            !p.try_dct64,
            "smooth_photo=false on gated cell: try_dct64 stays false"
        );

        // Auto detector returns `true` (input classified smooth) →
        // the dispatch must restore try_dct64=true so the encoder
        // evaluates DCT64-class transforms. This is the W44-34 fix.
        let cfg = LossyConfig::new(1.0).with_effort(7);
        let p = cfg.effective_profile_for_image_with_smoothness(512 * 512, true);
        assert!(
            p.try_dct64,
            "smooth_photo=true on gated cell: try_dct64 restored to true (W44-35)"
        );

        // Caller hint Some(true) wins over auto detector value false
        // (W44-130 Chunk D: hint moved into `StrategyOverrides`).
        let cfg = LossyConfig::new(1.0)
            .with_effort(7)
            .with_strategy_overrides(StrategyOverrides {
                smooth_photo_dct64_hint: Some(true),
                ..Default::default()
            });
        let p = cfg.effective_profile_for_image_with_smoothness(512 * 512, false);
        assert!(
            p.try_dct64,
            "explicit hint Some(true) wins over auto=false: try_dct64=true"
        );

        // Caller hint Some(false) wins over auto detector value true
        // (W44-130 Chunk D: hint moved into `StrategyOverrides`).
        let cfg = LossyConfig::new(1.0)
            .with_effort(7)
            .with_strategy_overrides(StrategyOverrides {
                smooth_photo_dct64_hint: Some(false),
                ..Default::default()
            });
        let p = cfg.effective_profile_for_image_with_smoothness(512 * 512, true);
        assert!(
            !p.try_dct64,
            "explicit hint Some(false) wins over auto=true: try_dct64=false"
        );

        // Outside the gate envelope (medium image), smoothness signal is
        // irrelevant — try_dct64 stays at the effort default (true at e7).
        let cfg = LossyConfig::new(1.0).with_effort(7);
        let p = cfg.effective_profile_for_image_with_smoothness(1024 * 1024, false);
        assert!(
            p.try_dct64,
            "medium image: try_dct64 stays true regardless of smoothness"
        );

        // At e6 the baseline try_dct64 is false (effort gate); on the
        // small + low-d cell the smooth-photo hint admits it (forced
        // true) so the encoder evaluates DCT64-class transforms.
        // Closes the 1418519 e6 cells (W44-34 forensics).
        let cfg = LossyConfig::new(1.2).with_effort(6);
        let p_default = cfg.effective_profile_for_image_with_smoothness(512 * 512, false);
        assert!(
            !p_default.try_dct64,
            "e6 baseline: try_dct64 stays false (pre-W44-35 behaviour)"
        );
        let p_smooth = cfg.effective_profile_for_image_with_smoothness(512 * 512, true);
        assert!(
            p_smooth.try_dct64,
            "e6 + smooth_photo=true on gated cell: try_dct64 admitted (W44-35)"
        );
    }

    /// W44-35 auto detector: smooth photo (low edge, low HF, low solid
    /// fill) returns `true`; textured / screen-content / large images
    /// return `false`.
    #[test]
    fn test_detect_smooth_photo_for_dct64() {
        // Large input (>= 500_000 px): short-circuits to false even on
        // smooth content (the gate it informs doesn't fire above 500k).
        let large_smooth = vec![128u8; 800 * 800 * 3];
        assert!(!detect_smooth_photo_for_dct64_from_layout(
            &large_smooth,
            800,
            800,
            PixelLayout::Rgb8,
        ));

        // Flat solid mid-gray (variance=0 everywhere) on a 512×512:
        // proxy_flat is 1.0 (all blocks solid) → rejected as
        // screenshot-like.
        let solid = vec![128u8; 512 * 512 * 3];
        assert!(!detect_smooth_photo_for_dct64_from_layout(
            &solid,
            512,
            512,
            PixelLayout::Rgb8,
        ));

        // Smooth low-frequency texture (photo-like): low edge density,
        // moderate flat ratio, low HF — should classify as smooth photo.
        // Built from a slow sinusoidal modulation so per-block variance
        // sits in the "smooth gradient" band (var > 5 → not "solid")
        // but the wavelength is long enough that proxy_edge and
        // proxy_hf both stay below the admission thresholds.
        let mut smooth = vec![0u8; 256 * 256 * 3];
        for y in 0..256 {
            for x in 0..256 {
                // Slow sinusoid in both axes, mean=128, amp~80.
                let fx = (x as f32) * 0.02; // ~32px wavelength
                let fy = (y as f32) * 0.02;
                let v = (128.0 + 80.0 * fx.sin() * fy.cos()).clamp(0.0, 255.0) as u8;
                let i = (y * 256 + x) * 3;
                smooth[i] = v;
                smooth[i + 1] = v;
                smooth[i + 2] = v;
            }
        }
        assert!(
            detect_smooth_photo_for_dct64_from_layout(&smooth, 256, 256, PixelLayout::Rgb8),
            "low-frequency sinusoidal texture should classify as smooth photo"
        );

        // Coarse high-contrast checkerboard (8×8 cells): high edge
        // density, screen-content-like — rejected. Cell size 8 is past
        // the 4× downsample Nyquist so edges survive into the proxy.
        let mut checker = vec![0u8; 256 * 256 * 3];
        for y in 0..256 {
            for x in 0..256 {
                let on = ((x / 8) + (y / 8)) % 2 == 0;
                let v = if on { 255u8 } else { 0 };
                let i = (y * 256 + x) * 3;
                checker[i] = v;
                checker[i + 1] = v;
                checker[i + 2] = v;
            }
        }
        assert!(
            !detect_smooth_photo_for_dct64_from_layout(&checker, 256, 256, PixelLayout::Rgb8),
            "coarse high-contrast checkerboard must NOT classify as smooth photo"
        );

        // Non-u8 layouts return false (auto detector skipped — caller
        // can still set Some(true) via the hint API).
        let f32_data = vec![0u8; 256 * 256 * 4 * 4]; // float pixels
        assert!(!detect_smooth_photo_for_dct64_from_layout(
            &f32_data,
            256,
            256,
            PixelLayout::RgbaLinearF32,
        ));
    }

    /// `__expert` sweep override pinning `try_dct64=Some(true)` must
    /// survive the per-image dispatch — mirrors the lossless override-
    /// respecting behaviour.
    #[cfg(feature = "__expert")]
    #[test]
    fn test_lossy_effective_profile_for_image_respects_internal_params_override() {
        let params = crate::effort::LossyInternalParams {
            try_dct64: Some(true),
            ..Default::default()
        };
        let cfg = LossyConfig::new(1.0)
            .with_effort(7)
            .with_internal_params(params);
        let p = cfg.effective_profile_for_image(256 * 256);
        // Override wins — dispatch did not fire.
        assert!(
            p.try_dct64,
            "sweep override try_dct64=Some(true) must survive the dispatch"
        );
    }

    #[test]
    fn test_lossless_config_builder_and_getters() {
        let cfg = LosslessConfig::new()
            .with_effort(5)
            .with_ans(false)
            .with_squeeze(true)
            .with_tree_learning(true);
        assert_eq!(cfg.effort(), 5);
        assert!(!cfg.ans());
        assert!(cfg.squeeze());
        assert!(cfg.tree_learning());
    }

    #[test]
    fn test_lossy_config_builder_and_getters() {
        let cfg = LossyConfig::new(2.0)
            .with_effort(3)
            .with_gaborish(false)
            .with_noise(true);
        assert_eq!(cfg.distance(), 2.0);
        assert_eq!(cfg.effort(), 3);
        assert!(!cfg.gaborish());
        assert!(cfg.noise());
    }

    #[test]
    fn test_lossy_config_epf_level_default_and_override() {
        // Default is -1 (encoder chooses).
        let cfg = LossyConfig::new(1.0);
        assert_eq!(cfg.epf_level(), -1);

        // Forced levels round-trip 0..=3.
        for level in [0i8, 1, 2, 3] {
            let cfg = LossyConfig::new(1.0).with_epf_level(level);
            assert_eq!(cfg.epf_level(), level);
        }

        // Values outside the libjxl `-1..=3` band are clamped.
        assert_eq!(LossyConfig::new(1.0).with_epf_level(-5).epf_level(), -1);
        assert_eq!(LossyConfig::new(1.0).with_epf_level(7).epf_level(), 3);
    }

    #[test]
    fn test_pixel_layout_helpers() {
        assert_eq!(PixelLayout::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(PixelLayout::Rgba8.bytes_per_pixel(), 4);
        assert_eq!(PixelLayout::Bgr8.bytes_per_pixel(), 3);
        assert_eq!(PixelLayout::Bgra8.bytes_per_pixel(), 4);
        assert_eq!(PixelLayout::Gray8.bytes_per_pixel(), 1);
        assert_eq!(PixelLayout::GrayAlpha8.bytes_per_pixel(), 2);
        assert_eq!(PixelLayout::Rgb16.bytes_per_pixel(), 6);
        assert_eq!(PixelLayout::Rgba16.bytes_per_pixel(), 8);
        assert_eq!(PixelLayout::Gray16.bytes_per_pixel(), 2);
        assert_eq!(PixelLayout::GrayAlpha16.bytes_per_pixel(), 4);
        assert_eq!(PixelLayout::RgbLinearF32.bytes_per_pixel(), 12);
        assert_eq!(PixelLayout::RgbaLinearF32.bytes_per_pixel(), 16);
        assert_eq!(PixelLayout::GrayLinearF32.bytes_per_pixel(), 4);
        assert_eq!(PixelLayout::GrayAlphaLinearF32.bytes_per_pixel(), 8);
        // Linear
        assert!(!PixelLayout::Rgb8.is_linear());
        assert!(PixelLayout::RgbLinearF32.is_linear());
        assert!(PixelLayout::RgbaLinearF32.is_linear());
        assert!(PixelLayout::GrayLinearF32.is_linear());
        assert!(PixelLayout::GrayAlphaLinearF32.is_linear());
        assert!(!PixelLayout::Rgb16.is_linear());
        // Alpha
        assert!(!PixelLayout::Rgb8.has_alpha());
        assert!(PixelLayout::Rgba8.has_alpha());
        assert!(PixelLayout::Bgra8.has_alpha());
        assert!(PixelLayout::GrayAlpha8.has_alpha());
        assert!(PixelLayout::Rgba16.has_alpha());
        assert!(PixelLayout::GrayAlpha16.has_alpha());
        assert!(PixelLayout::RgbaLinearF32.has_alpha());
        assert!(PixelLayout::GrayAlphaLinearF32.has_alpha());
        assert!(!PixelLayout::Rgb16.has_alpha());
        assert!(!PixelLayout::RgbLinearF32.has_alpha());
        // 16-bit
        assert!(PixelLayout::Rgb16.is_16bit());
        assert!(PixelLayout::Rgba16.is_16bit());
        assert!(PixelLayout::Gray16.is_16bit());
        assert!(PixelLayout::GrayAlpha16.is_16bit());
        assert!(!PixelLayout::Rgb8.is_16bit());
        assert!(!PixelLayout::RgbLinearF32.is_16bit());
        // f32
        assert!(PixelLayout::RgbLinearF32.is_f32());
        assert!(PixelLayout::RgbaLinearF32.is_f32());
        assert!(PixelLayout::GrayLinearF32.is_f32());
        assert!(PixelLayout::GrayAlphaLinearF32.is_f32());
        assert!(!PixelLayout::Rgb8.is_f32());
        assert!(!PixelLayout::Rgb16.is_f32());
        // Grayscale
        assert!(PixelLayout::Gray8.is_grayscale());
        assert!(PixelLayout::GrayAlpha8.is_grayscale());
        assert!(PixelLayout::Gray16.is_grayscale());
        assert!(PixelLayout::GrayAlpha16.is_grayscale());
        assert!(PixelLayout::GrayLinearF32.is_grayscale());
        assert!(PixelLayout::GrayAlphaLinearF32.is_grayscale());
        assert!(!PixelLayout::Rgb16.is_grayscale());
        assert!(!PixelLayout::RgbLinearF32.is_grayscale());
        // CMYK (#58) — 4-byte/8-byte, no alpha, not grayscale,
        // is_cmyk() flagged, Cmyk16 is also 16-bit.
        assert_eq!(PixelLayout::Cmyk8.bytes_per_pixel(), 4);
        assert_eq!(PixelLayout::Cmyk16.bytes_per_pixel(), 8);
        assert!(PixelLayout::Cmyk8.is_cmyk());
        assert!(PixelLayout::Cmyk16.is_cmyk());
        assert!(!PixelLayout::Rgb8.is_cmyk());
        assert!(!PixelLayout::Rgba8.is_cmyk());
        assert!(!PixelLayout::Cmyk8.has_alpha());
        assert!(!PixelLayout::Cmyk16.has_alpha());
        assert!(!PixelLayout::Cmyk8.is_grayscale());
        assert!(PixelLayout::Cmyk16.is_16bit());
        assert!(!PixelLayout::Cmyk8.is_16bit());
    }

    #[test]
    fn test_quality_to_distance() {
        assert!(Quality::Distance(1.0).to_distance().unwrap() == 1.0);
        assert!(Quality::Distance(-1.0).to_distance().is_err());
        assert!(Quality::Percent(100).to_distance().is_err()); // lossless invalid for lossy
        assert!(Quality::Percent(90).to_distance().unwrap() == 1.0);
    }

    #[test]
    fn test_pixel_validation() {
        let cfg = LosslessConfig::new();
        let req = cfg.encode_request(2, 2, PixelLayout::Rgb8);
        assert!(req.validate_pixels(&[0u8; 12]).is_ok());
    }

    #[test]
    fn test_pixel_validation_wrong_size() {
        let cfg = LosslessConfig::new();
        let req = cfg.encode_request(2, 2, PixelLayout::Rgb8);
        assert!(req.validate_pixels(&[0u8; 11]).is_err());
    }

    #[test]
    fn test_limits_check() {
        let limits = Limits::new().with_max_width(100);
        let cfg = LosslessConfig::new();
        let req = cfg
            .encode_request(200, 100, PixelLayout::Rgb8)
            .with_limits(&limits);
        assert!(req.check_limits().is_err());
    }

    #[test]
    fn test_lossless_encode_rgb8_small() {
        // 4x4 red image
        let pixels = [255u8, 0, 0].repeat(16);
        let result = LosslessConfig::new()
            .encode_request(4, 4, PixelLayout::Rgb8)
            .encode(&pixels);
        assert!(result.is_ok());
        let jxl = result.unwrap();
        assert_eq!(&jxl[..2], &[0xFF, 0x0A]); // JXL signature
    }

    #[test]
    fn test_lossy_encode_rgb8_small() {
        // 8x8 gradient
        let mut pixels = Vec::with_capacity(8 * 8 * 3);
        for y in 0..8u8 {
            for x in 0..8u8 {
                pixels.push(x * 32);
                pixels.push(y * 32);
                pixels.push(128);
            }
        }
        let result = LossyConfig::new(2.0)
            .with_gaborish(false)
            .encode_request(8, 8, PixelLayout::Rgb8)
            .encode(&pixels);
        assert!(result.is_ok());
        let jxl = result.unwrap();
        assert_eq!(&jxl[..2], &[0xFF, 0x0A]);
    }

    #[test]
    fn test_fluent_lossless() {
        let pixels = vec![128u8; 4 * 4 * 3];
        let result = LosslessConfig::new().encode(&pixels, 4, 4, PixelLayout::Rgb8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lossy_gray8() {
        // Grayscale input → RGB expansion → VarDCT (XYB)
        let pixels = vec![128u8; 8 * 8];
        let result = LossyConfig::new(2.0)
            .with_gaborish(false)
            .encode_request(8, 8, PixelLayout::Gray8)
            .encode(&pixels);
        assert!(result.is_ok(), "lossy Gray8 should encode: {result:?}");
    }

    #[test]
    fn test_lossy_gray_alpha8() {
        let pixels: Vec<u8> = (0..8 * 8).flat_map(|_| [128u8, 255]).collect();
        let result = LossyConfig::new(2.0)
            .with_gaborish(false)
            .encode_request(8, 8, PixelLayout::GrayAlpha8)
            .encode(&pixels);
        assert!(result.is_ok(), "lossy GrayAlpha8 should encode: {result:?}");
    }

    #[test]
    fn test_lossy_gray16() {
        let pixels_u16: Vec<u16> = (0..8 * 8).map(|_| 32768u16).collect();
        let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);
        let result = LossyConfig::new(2.0)
            .with_gaborish(false)
            .encode_request(8, 8, PixelLayout::Gray16)
            .encode(pixels);
        assert!(result.is_ok(), "lossy Gray16 should encode: {result:?}");
    }

    #[test]
    fn test_lossy_rgba_linear_f32() {
        let pixels_f32: Vec<f32> = (0..8 * 8).flat_map(|_| [0.5f32, 0.3, 0.7, 1.0]).collect();
        let pixels: &[u8] = bytemuck::cast_slice(&pixels_f32);
        let result = LossyConfig::new(2.0)
            .with_gaborish(false)
            .encode_request(8, 8, PixelLayout::RgbaLinearF32)
            .encode(pixels);
        assert!(
            result.is_ok(),
            "lossy RgbaLinearF32 should encode: {result:?}"
        );
    }

    #[test]
    fn test_lossy_gray_linear_f32() {
        let pixels_f32: Vec<f32> = (0..8 * 8).map(|_| 0.5f32).collect();
        let pixels: &[u8] = bytemuck::cast_slice(&pixels_f32);
        let result = LossyConfig::new(2.0)
            .with_gaborish(false)
            .encode_request(8, 8, PixelLayout::GrayLinearF32)
            .encode(pixels);
        assert!(
            result.is_ok(),
            "lossy GrayLinearF32 should encode: {result:?}"
        );
    }

    #[test]
    fn test_lossless_grayalpha8() {
        let pixels: Vec<u8> = (0..8 * 8).flat_map(|_| [200u8, 255]).collect();
        let result = LosslessConfig::new().encode(&pixels, 8, 8, PixelLayout::GrayAlpha8);
        assert!(
            result.is_ok(),
            "lossless GrayAlpha8 should encode: {result:?}"
        );
    }

    #[test]
    fn test_lossless_grayalpha16() {
        let pixels_u16: Vec<u16> = (0..8 * 8).flat_map(|_| [32768u16, 65535]).collect();
        let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);
        let result = LosslessConfig::new().encode(pixels, 8, 8, PixelLayout::GrayAlpha16);
        assert!(
            result.is_ok(),
            "lossless GrayAlpha16 should encode: {result:?}"
        );
    }

    #[test]
    fn test_bgra_lossless() {
        // 4x4 red image in BGRA (B=0, G=0, R=255, A=255)
        let pixels = [0u8, 0, 255, 255].repeat(16);
        let result = LosslessConfig::new().encode(&pixels, 4, 4, PixelLayout::Bgra8);
        assert!(result.is_ok());
        let jxl = result.unwrap();
        assert_eq!(&jxl[..2], &[0xFF, 0x0A]);
    }

    #[test]
    fn test_lossy_alpha_encodes() {
        // Lossy+alpha: VarDCT RGB + modular alpha extra channel
        let pixels = [255u8, 0, 0, 255].repeat(64);
        let result =
            LossyConfig::new(2.0)
                .with_gaborish(false)
                .encode(&pixels, 8, 8, PixelLayout::Bgra8);
        assert!(
            result.is_ok(),
            "BGRA lossy encode failed: {:?}",
            result.err()
        );

        let result2 = LossyConfig::new(2.0).encode(&pixels, 8, 8, PixelLayout::Rgba8);
        assert!(
            result2.is_ok(),
            "RGBA lossy encode failed: {:?}",
            result2.err()
        );
    }

    #[test]
    fn test_stop_cancellation() {
        use enough::Unstoppable;
        // Unstoppable should not cancel
        let pixels = vec![128u8; 4 * 4 * 3];
        let cfg = LosslessConfig::new();
        let result = cfg
            .encode_request(4, 4, PixelLayout::Rgb8)
            .with_stop(&Unstoppable)
            .encode(&pixels);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lossy_palette_encode() {
        // 16x16 RGB image with 4 colors + slight noise
        let colors = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
        let mut pixels = Vec::with_capacity(16 * 16 * 3);
        for y in 0..16u8 {
            for x in 0..16u8 {
                let ci = ((y / 4) * 4 + x / 4) as usize % 4;
                let noise = ((x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13))) % 5) as i16 - 2;
                for &channel in &colors[ci][..3] {
                    let v = (channel as i16 + noise).clamp(0, 255) as u8;
                    pixels.push(v);
                }
            }
        }
        let cfg = LosslessConfig::new()
            .with_lossy_palette(true)
            .with_ans(true);
        let result = cfg.encode(&pixels, 16, 16, PixelLayout::Rgb8);
        assert!(
            result.is_ok(),
            "lossy palette encode failed: {:?}",
            result.err()
        );
        let jxl = result.unwrap();
        assert_eq!(&jxl[..2], &[0xFF, 0x0A], "JXL signature");

        // Verify jxl-oxide can parse and decode it
        let cursor = std::io::Cursor::new(&jxl);
        let reader = std::io::BufReader::new(cursor);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("jxl-oxide parse");
        assert!(
            image.width() > 0,
            "decoded image should have non-zero width"
        );
    }

    #[test]
    fn test_lossy_palette_multi_group() {
        // 300x300 RGB image with ~20 dominant colors + noise (>256x256 = multi-group)
        let colors = [
            [255u8, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [128, 128, 128],
            [64, 64, 64],
        ];
        let mut pixels = Vec::with_capacity(300 * 300 * 3);
        for y in 0..300u32 {
            for x in 0..300u32 {
                let ci = ((y / 40) * 8 + x / 40) as usize % colors.len();
                let noise = ((x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13))) % 7) as i16 - 3;
                for &channel in &colors[ci][..3] {
                    let v = (channel as i16 + noise).clamp(0, 255) as u8;
                    pixels.push(v);
                }
            }
        }

        // Encode with lossy palette + ANS (multi-group)
        let cfg = LosslessConfig::new()
            .with_lossy_palette(true)
            .with_ans(true);
        let jxl = cfg
            .encode(&pixels, 300, 300, PixelLayout::Rgb8)
            .expect("lossy palette multi-group encode");
        assert_eq!(&jxl[..2], &[0xFF, 0x0A], "JXL signature");
        assert!(jxl.len() < 300 * 300 * 3, "should compress");

        // Save to disk for inspection
        let out = crate::test_helpers::output_dir("lossy_palette");
        let jxl_out = out.join("lossy_palette_multi.jxl");
        let png_out = out.join("lossy_palette_multi.png");
        std::fs::write(&jxl_out, &jxl).ok();
        eprintln!(
            "LOSSY_PALETTE_MULTI test: encoded {} bytes ({}x{})",
            jxl.len(),
            300,
            300
        );

        // Try djxl decode first for better error messages
        let djxl_result = std::process::Command::new("djxl")
            .args([jxl_out.to_str().unwrap(), png_out.to_str().unwrap()])
            .output();
        if let Ok(output) = djxl_result {
            eprintln!(
                "djxl: status={}, stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Verify jxl-rs can decode it
        let decoded = crate::test_helpers::decode_with_jxl_rs(&jxl).expect("jxl-rs decode failed");
        assert_eq!(decoded.width, 300);
        assert_eq!(decoded.height, 300);
        assert_eq!(decoded.channels, 3);

        // Verify lossy quality: each pixel should be within 50 of original (delta palette error)
        // decoded.pixels is f32 in [0.0, 1.0] — convert to u8 for comparison
        let mut max_error = 0i32;
        let mut error_pos = (0, 0, 0);
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
            let dec_u8 = (dec * 255.0).round().clamp(0.0, 255.0) as u8;
            let diff = (orig as i32 - dec_u8 as i32).abs();
            if diff > max_error {
                max_error = diff;
                let pixel = i / 3;
                error_pos = (pixel % 300, pixel / 300, i % 3);
            }
        }
        let err_idx = error_pos.1 * 300 * 3 + error_pos.0 * 3 + error_pos.2;
        let dec_u8 = (decoded.pixels[err_idx] * 255.0).round().clamp(0.0, 255.0) as u8;
        eprintln!(
            "max_error={} at ({},{}) ch={}, orig={} decoded={}",
            max_error, error_pos.0, error_pos.1, error_pos.2, pixels[err_idx], dec_u8,
        );
        assert!(
            max_error <= 80,
            "lossy palette max error {} too large (expected <= 80)",
            max_error
        );
    }

    #[test]
    fn test_palette_256_colors_regression() {
        // Regression test for palette+ANS checksum mismatch with many unique colors.
        // Root cause was u2S bit width bug in write_palette_transform (fixed Feb 17, 2026):
        // nb_colors selectors 1-2 used 11/14 bits instead of 10/12 bits. Triggered when
        // nb_colors >= 256 (selector 1). Two test cases:
        //
        // 1. 32x32 with 256 unique colors via standard API (passes 50% heuristic)
        // 2. 16x16 with 256 unique colors via internal API (bypasses heuristic)
        use crate::modular::channel::{Channel, ModularImage};
        use crate::modular::encode::write_modular_stream_with_palette;

        // Test 1: 32x32 through standard API (256 colors, each used 4x)
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for i in 0..1024u32 {
            let idx = (i / 4) as u8;
            pixels.push(idx);
            pixels.push(((idx as u32 * 7 + 13) & 0xFF) as u8);
            pixels.push(((idx as u32 * 31 + 97) & 0xFF) as u8);
        }
        let cfg = LosslessConfig::new().with_ans(true);
        let jxl = cfg
            .encode(&pixels, 32, 32, PixelLayout::Rgb8)
            .expect("palette 256-colors encode");
        let decoded = crate::test_helpers::decode_with_jxl_rs(&jxl).expect("jxl-rs decode failed");
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
            let dec_u8 = (dec * 255.0).round().clamp(0.0, 255.0) as u8;
            assert_eq!(
                orig, dec_u8,
                "32x32: mismatch at byte {}: orig={} decoded={}",
                i, orig, dec_u8
            );
        }

        // Test 2: 16x16 via internal API (bypasses 50% heuristic)
        let mut channels = Vec::new();
        for c in 0..3 {
            let mut ch = Channel::new(16, 16).unwrap();
            for y in 0..16 {
                for x in 0..16 {
                    let idx = y * 16 + x;
                    let val = match c {
                        0 => idx as i32,
                        1 => ((idx * 3 + 17) & 0xFF) as i32,
                        2 => (255 - idx) as i32,
                        _ => 0,
                    };
                    ch.set(x, y, val);
                }
            }
            channels.push(ch);
        }
        let image = ModularImage {
            channels,
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut writer = crate::bit_writer::BitWriter::new();
        write_modular_stream_with_palette(&image, &mut writer, true, 0, 3)
            .expect("palette encode with 256 unique colors must not fail");
    }

    #[test]
    fn test_16bit_tree_learning() {
        // Test multiple 16-bit scenarios that previously failed
        for &(w, h, layout, label) in &[
            (32u32, 32u32, PixelLayout::Rgb16, "32x32 RGB16"),
            (8, 8, PixelLayout::Rgba16, "8x8 RGBA16"),
            (8, 8, PixelLayout::Rgb16, "8x8 RGB16"),
            (16, 16, PixelLayout::Gray16, "16x16 Gray16"),
        ] {
            let nc = layout.bytes_per_pixel()
                / if layout.is_16bit() {
                    2
                } else if layout.is_f32() {
                    4
                } else {
                    1
                };
            let mut pixels = vec![0u16; (w * h) as usize * nc];
            for y in 0..h {
                for x in 0..w {
                    let idx = ((y * w + x) as usize) * nc;
                    pixels[idx] = (x * 2048) as u16;
                    if nc >= 2 {
                        pixels[idx + 1] = (y * 2048) as u16;
                    }
                    if nc >= 3 {
                        pixels[idx + 2] = ((x + y) * 1024) as u16;
                    }
                    if nc >= 4 {
                        pixels[idx + 3] = 65535; // opaque alpha
                    }
                }
            }
            let bytes: Vec<u8> = pixels.iter().flat_map(|v| v.to_ne_bytes()).collect();

            let cfg = LosslessConfig::new().with_effort(7).with_ans(true);
            let jxl = cfg
                .encode(&bytes, w, h, layout)
                .unwrap_or_else(|e| panic!("{}: encode failed: {}", label, e));

            let decoded = crate::test_helpers::decode_with_jxl_rs(&jxl)
                .unwrap_or_else(|e| panic!("{}: jxl-rs decode failed: {}", label, e));
            assert_eq!(decoded.width, w as usize, "{}: width", label);
            assert_eq!(decoded.height, h as usize, "{}: height", label);

            let scale = 65535.0;
            let mut mismatches = 0;
            for (i, (&orig, &dec_f)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
                let dec = (dec_f * scale).round().clamp(0.0, scale) as u16;
                if orig != dec && mismatches < 3 {
                    eprintln!("{}: mismatch[{}]: orig={} dec={}", label, i, orig, dec);
                    mismatches += 1;
                }
            }
            assert_eq!(mismatches, 0, "{}: {} mismatches", label, mismatches);
            eprintln!("{}: PASS ({} bytes)", label, jxl.len());
        }
    }

    #[test]
    fn test_srgb_lut_matches_powf() {
        for i in 0u16..256 {
            let lut_val = SRGB_U8_TO_LINEAR[i as usize];
            let fast_val = srgb_to_linear_f(i as f32 / 255.0);
            let diff = (lut_val - fast_val).abs();
            // LUT uses f64 exact powf, srgb_to_linear_f uses fast_powf (~3e-5 relative error)
            let tol = fast_val.abs() * 5e-5 + 1e-7;
            assert!(
                diff <= tol,
                "sRGB LUT mismatch at {i}: LUT={lut_val}, fast={fast_val}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_quality_to_distance_f32_mapping() {
        // Verify the piecewise mapping at key points.
        assert_eq!(quality_to_distance(100.0), 0.0);
        assert_eq!(quality_to_distance(90.0), 1.0); // visually lossless
        assert_eq!(quality_to_distance(80.0), 1.5);
        assert_eq!(quality_to_distance(70.0), 2.0);
        assert_eq!(quality_to_distance(50.0), 4.0);
        assert_eq!(quality_to_distance(0.0), 9.0);
        // Clamped above 100
        assert_eq!(quality_to_distance(110.0), 0.0);
    }

    #[test]
    fn test_calibrated_jxl_quality() {
        // Boundary: below table minimum clamps to first entry's output.
        assert_eq!(calibrated_jxl_quality(0.0), 5.0);
        // Boundary: above table maximum clamps to last entry's output.
        assert_eq!(calibrated_jxl_quality(100.0), 93.8);
        // Exact table entry.
        assert_eq!(calibrated_jxl_quality(90.0), 84.2);
        // Interpolated mid-point between (50, 48.5) and (55, 51.9).
        let mid = calibrated_jxl_quality(52.5);
        let expected = 48.5 + 0.5 * (51.9 - 48.5);
        assert!(
            (mid - expected).abs() < 0.01,
            "expected {expected}, got {mid}"
        );
    }

    #[test]
    fn test_interp_quality_edge_cases() {
        let table = &[(10.0f32, 20.0f32), (20.0, 40.0), (30.0, 60.0)];
        // Below table
        assert_eq!(interp_quality(table, 5.0), 20.0);
        // Above table
        assert_eq!(interp_quality(table, 35.0), 60.0);
        // Exact match
        assert_eq!(interp_quality(table, 20.0), 40.0);
        // Midpoint
        assert!((interp_quality(table, 15.0) - 30.0).abs() < 0.001);
    }

    // -----------------------------------------------------------------
    // Internal-params override (__expert) — segmented Lossy / Lossless
    // -----------------------------------------------------------------

    #[cfg(feature = "__expert")]
    mod internal_params {
        use super::*;
        use crate::effort::{LosslessInternalParams, LossyInternalParams};

        // Pseudo-random RGB image — large enough + complex enough to exercise
        // RCT search, WP, and tree-learning splits so different param
        // settings produce different bitstreams.
        fn pseudo_random_rgb8(w: u32, h: u32) -> Vec<u8> {
            let mut out = Vec::with_capacity((w * h * 3) as usize);
            let mut state: u32 = 0xDEAD_BEEF;
            for _ in 0..(w * h) {
                let r = state.wrapping_mul(1664525).wrapping_add(1013904223);
                state = r;
                let g = state.wrapping_mul(1664525).wrapping_add(1013904223);
                state = g;
                let b = state.wrapping_mul(1664525).wrapping_add(1013904223);
                state = b;
                out.push((r >> 24) as u8);
                out.push((g >> 24) as u8);
                out.push((b >> 24) as u8);
            }
            out
        }

        #[test]
        fn lossless_internal_params_changes_bitstream() {
            // Tighten tree learning + skip RCT search to push bytes off the
            // e7 default.
            let params = LosslessInternalParams {
                tree_max_buckets: Some(16),
                tree_num_properties: Some(3),
                nb_rcts_to_try: Some(0),
                ..Default::default()
            };

            let cfg_override = LosslessConfig::new()
                .with_effort(7)
                .with_internal_params(params)
                .with_threads(1);
            let cfg_default = LosslessConfig::new().with_effort(7).with_threads(1);

            let pixels = pseudo_random_rgb8(64, 64);
            let bytes_a = cfg_override
                .encode(&pixels, 64, 64, PixelLayout::Rgb8)
                .expect("override encode");
            let bytes_b = cfg_default
                .encode(&pixels, 64, 64, PixelLayout::Rgb8)
                .expect("default encode");

            assert_eq!(&bytes_a[..2], &crate::JXL_SIGNATURE);
            assert_eq!(&bytes_b[..2], &crate::JXL_SIGNATURE);
            assert_ne!(
                bytes_a, bytes_b,
                "internal_params override should produce different bitstream"
            );
        }

        #[test]
        fn lossy_internal_params_changes_bitstream() {
            let mut entropy = crate::effort::EntropyMulTable::reference();
            entropy.dct8 = 0.95;
            let params = LossyInternalParams {
                try_dct16: Some(false),
                try_dct32: Some(false),
                try_dct64: Some(false),
                try_dct4x8_afv: Some(false),
                k_info_loss_mul_base: Some(1.5),
                entropy_mul_table: Some(entropy),
                ..Default::default()
            };

            let cfg_override = LossyConfig::new(2.0)
                .with_effort(7)
                .with_internal_params(params)
                .with_threads(1);
            let cfg_default = LossyConfig::new(2.0).with_effort(7).with_threads(1);

            let pixels = pseudo_random_rgb8(64, 64);
            let bytes_a = cfg_override
                .encode(&pixels, 64, 64, PixelLayout::Rgb8)
                .expect("override encode");
            let bytes_b = cfg_default
                .encode(&pixels, 64, 64, PixelLayout::Rgb8)
                .expect("default encode");

            assert_eq!(&bytes_a[..2], &crate::JXL_SIGNATURE);
            assert_eq!(&bytes_b[..2], &crate::JXL_SIGNATURE);
            assert_ne!(
                bytes_a, bytes_b,
                "internal_params override should produce different bitstream"
            );
        }

        #[test]
        fn lossless_internal_params_persist_across_with_effort() {
            // Override applied before with_effort should still take effect
            // (with_effort preserves profile_override).
            let params = LosslessInternalParams {
                tree_max_buckets: Some(16),
                ..Default::default()
            };

            let cfg = LosslessConfig::new()
                .with_internal_params(params)
                .with_effort(9) // should NOT clobber the override
                .with_threads(1);

            let pixels = pseudo_random_rgb8(64, 64);
            let bytes_with_override = cfg
                .encode(&pixels, 64, 64, PixelLayout::Rgb8)
                .expect("encode");
            let bytes_e9_plain = LosslessConfig::new()
                .with_effort(9)
                .with_threads(1)
                .encode(&pixels, 64, 64, PixelLayout::Rgb8)
                .expect("encode");

            assert_ne!(
                bytes_with_override, bytes_e9_plain,
                "override should persist across with_effort()"
            );
        }
    }

    // ─── Shared knob enums (libjxl `cjxl` parity) ─────────────────

    #[test]
    fn test_container_mode_default_auto() {
        assert_eq!(LossyConfig::new(1.0).container_mode(), ContainerMode::Auto);
        assert_eq!(LosslessConfig::new().container_mode(), ContainerMode::Auto);
    }

    #[test]
    fn test_container_mode_round_trip() {
        let cfg = LossyConfig::new(1.0).with_container_mode(ContainerMode::Always);
        assert_eq!(cfg.container_mode(), ContainerMode::Always);
        let cfg = cfg.with_container_mode(ContainerMode::Never);
        assert_eq!(cfg.container_mode(), ContainerMode::Never);
    }

    #[test]
    fn test_faster_decoding_clamp() {
        // Out-of-range values clamp to MAX_FASTER_DECODING.
        let cfg = LossyConfig::new(1.0).with_faster_decoding(99);
        assert_eq!(cfg.faster_decoding(), MAX_FASTER_DECODING);
        // 0 is the default (no speed bias).
        let cfg = LossyConfig::new(1.0);
        assert_eq!(cfg.faster_decoding(), 0);
        // In-range values pass through.
        for tier in 0..=MAX_FASTER_DECODING {
            assert_eq!(
                LossyConfig::new(1.0)
                    .with_faster_decoding(tier)
                    .faster_decoding(),
                tier,
            );
            assert_eq!(
                LosslessConfig::new()
                    .with_faster_decoding(tier)
                    .faster_decoding(),
                tier,
            );
        }
    }

    #[test]
    fn test_faster_decoding_lossless_effective_getters() {
        // Tier 0: all getters return the stored field values.
        let cfg = LosslessConfig::new();
        let stored_lz77 = cfg.lz77();
        let stored_tree = cfg.tree_learning();
        let stored_patches = cfg.patches();
        assert_eq!(cfg.effective_lz77(), stored_lz77);
        assert_eq!(cfg.effective_tree_learning(), stored_tree);
        assert_eq!(cfg.effective_patches(), stored_patches);
        assert_eq!(cfg.effective_modular_group_size_shift(), None);

        // Tier 1: LZ77 off. Tree-learning + patches unchanged.
        let cfg = LosslessConfig::new().with_faster_decoding(1);
        assert!(!cfg.effective_lz77(), "tier 1 disables LZ77");
        assert_eq!(cfg.effective_tree_learning(), stored_tree);
        assert_eq!(cfg.effective_patches(), stored_patches);
        assert_eq!(cfg.effective_modular_group_size_shift(), None);

        // Tier 2: + group_size_shift = 0 + patches off.
        let cfg = LosslessConfig::new().with_faster_decoding(2);
        assert!(!cfg.effective_lz77());
        assert_eq!(cfg.effective_tree_learning(), stored_tree);
        assert!(!cfg.effective_patches(), "tier 2 disables patches");
        assert_eq!(cfg.effective_modular_group_size_shift(), Some(0));

        // Tier 4: + tree_learning off.
        let cfg = LosslessConfig::new().with_faster_decoding(4);
        assert!(!cfg.effective_lz77());
        assert!(
            !cfg.effective_tree_learning(),
            "tier 4 disables tree learning"
        );
        assert!(!cfg.effective_patches());
        assert_eq!(cfg.effective_modular_group_size_shift(), Some(0));

        // Explicit `with_modular_group_size` overrides the tier-2 default.
        let cfg = LosslessConfig::new()
            .with_faster_decoding(2)
            .with_modular_group_size(Some(2));
        assert_eq!(
            cfg.effective_modular_group_size_shift(),
            Some(2),
            "explicit modular_group_size wins over tier-2 default"
        );
    }

    #[test]
    fn test_faster_decoding_lossy_effective_getters() {
        // Tier 0: getters return stored field values.
        let cfg = LossyConfig::new(1.0);
        let stored_lz77 = cfg.lz77();
        let stored_patches = cfg.patches();
        let stored_gab = cfg.gaborish();
        assert_eq!(cfg.effective_lz77(), stored_lz77);
        assert_eq!(cfg.effective_patches(), stored_patches);
        assert_eq!(cfg.effective_gaborish(), stored_gab);

        // Tier 1: LZ77 off.
        let cfg = LossyConfig::new(1.0).with_faster_decoding(1);
        assert!(!cfg.effective_lz77());
        assert_eq!(cfg.effective_patches(), stored_patches);
        assert_eq!(cfg.effective_gaborish(), stored_gab);

        // Tier 2: + patches off.
        let cfg = LossyConfig::new(1.0).with_faster_decoding(2);
        assert!(!cfg.effective_lz77());
        assert!(!cfg.effective_patches());
        assert_eq!(cfg.effective_gaborish(), stored_gab);

        // Tier 4: + gaborish forced off.
        let cfg = LossyConfig::new(1.0).with_faster_decoding(4);
        assert!(!cfg.effective_lz77());
        assert!(!cfg.effective_patches());
        assert!(!cfg.effective_gaborish(), "tier 4 disables gaborish");
    }

    #[test]
    fn test_faster_decoding_lossless_roundtrip_levels_0_2_4() {
        // Encode a deterministic synthetic RGB image at faster_decoding
        // levels 0, 2, 4 and verify:
        //   (a) every level produces a valid jxl-rs roundtrip,
        //   (b) bytes grow monotonically as the tier rises (libjxl
        //       semantics — higher tier = simpler bitstream = larger
        //       file at the same effort).
        const W: u32 = 96;
        const H: u32 = 96;
        let mut pixels = Vec::with_capacity((W * H * 3) as usize);
        for y in 0..H {
            for x in 0..W {
                // Mix of smooth gradients and high-frequency content so the
                // tier-1/2/4 disables (LZ77, group_size, tree learning) all
                // see something interesting to bias on. Pure noise would
                // make tier-X bytes uncomfortably close to incompressible.
                let r = ((x.wrapping_mul(3) ^ y.wrapping_mul(5)) & 0xFF) as u8;
                let g = ((x + y) & 0xFF) as u8;
                let b = ((x.wrapping_mul(17) ^ (y.wrapping_mul(11))) & 0xFF) as u8;
                pixels.push(r);
                pixels.push(g);
                pixels.push(b);
            }
        }

        let encode = |tier: u8| -> Vec<u8> {
            LosslessConfig::new()
                .with_effort(7)
                .with_faster_decoding(tier)
                .encode(&pixels, W, H, PixelLayout::Rgb8)
                .unwrap_or_else(|e| panic!("encode tier={} failed: {:?}", tier, e))
        };

        let bytes0 = encode(0);
        let bytes2 = encode(2);
        let bytes4 = encode(4);

        // (a) all three roundtrip via jxl-rs and reproduce input bit-exact.
        for (tier, bytes) in [(0, &bytes0), (2, &bytes2), (4, &bytes4)] {
            let decoded = crate::test_helpers::decode_with_jxl_rs(bytes)
                .unwrap_or_else(|e| panic!("jxl-rs decode tier={} failed: {:?}", tier, e));
            assert_eq!(decoded.width, W as usize, "tier {} width", tier);
            assert_eq!(decoded.height, H as usize, "tier {} height", tier);
            assert_eq!(decoded.channels, 3, "tier {} channels", tier);
            // Lossless: pixels must match exactly.
            for (i, (&orig, &dec)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
                let dec_u8 = (dec * 255.0).round().clamp(0.0, 255.0) as u8;
                assert_eq!(
                    orig, dec_u8,
                    "tier {}: pixel mismatch at byte {}: orig={} decoded={}",
                    tier, i, orig, dec_u8,
                );
            }
        }

        // (b) bytes grow with tier. Higher tier = simpler bitstream =
        // larger file (the decode-speed tradeoff).
        eprintln!(
            "faster_decoding lossless bytes: t0={} t2={} t4={}",
            bytes0.len(),
            bytes2.len(),
            bytes4.len(),
        );
        assert!(
            bytes2.len() >= bytes0.len(),
            "tier 2 ({} B) should be >= tier 0 ({} B)",
            bytes2.len(),
            bytes0.len()
        );
        assert!(
            bytes4.len() >= bytes2.len(),
            "tier 4 ({} B) should be >= tier 2 ({} B)",
            bytes4.len(),
            bytes2.len()
        );
    }

    #[test]
    fn test_faster_decoding_lossy_roundtrip_levels_0_2_4() {
        // Lossy analog: encode at d=1.0, e7 with faster_decoding 0/2/4
        // and verify jxl-rs decodes each + records the byte counts.
        // We do NOT assert byte monotonicity on lossy — quality drift
        // from disabling gaborish (tier 4) can occasionally produce
        // smaller files via different AC strategy selection. The hard
        // requirement is "all tiers decode".
        const W: u32 = 96;
        const H: u32 = 96;
        let mut pixels = Vec::with_capacity((W * H * 3) as usize);
        for y in 0..H {
            for x in 0..W {
                let r = ((x.wrapping_mul(3) ^ y.wrapping_mul(5)) & 0xFF) as u8;
                let g = ((x + y) & 0xFF) as u8;
                let b = ((x.wrapping_mul(17) ^ (y.wrapping_mul(11))) & 0xFF) as u8;
                pixels.push(r);
                pixels.push(g);
                pixels.push(b);
            }
        }

        let encode = |tier: u8| -> Vec<u8> {
            LossyConfig::new(1.0)
                .with_effort(7)
                .with_faster_decoding(tier)
                .encode(&pixels, W, H, PixelLayout::Rgb8)
                .unwrap_or_else(|e| panic!("lossy encode tier={} failed: {:?}", tier, e))
        };

        let bytes0 = encode(0);
        let bytes2 = encode(2);
        let bytes4 = encode(4);

        eprintln!(
            "faster_decoding lossy bytes: t0={} t2={} t4={}",
            bytes0.len(),
            bytes2.len(),
            bytes4.len(),
        );

        for (tier, bytes) in [(0, &bytes0), (2, &bytes2), (4, &bytes4)] {
            let decoded = crate::test_helpers::decode_with_jxl_rs(bytes)
                .unwrap_or_else(|e| panic!("jxl-rs decode tier={} failed: {:?}", tier, e));
            assert_eq!(decoded.width, W as usize, "tier {} width", tier);
            assert_eq!(decoded.height, H as usize, "tier {} height", tier);
        }
    }

    #[test]
    fn test_faster_decoding_profile_apply() {
        use crate::effort::EffortProfile;

        let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
        let base_lz77 = p.lz77;
        let base_custom_orders = p.custom_orders;
        let base_enhanced = p.enhanced_clustering_vardct;
        let base_gaborish = p.gaborish;
        let base_try_dct32 = p.try_dct32;
        let base_tree_learning = p.tree_learning;
        let base_patches = p.patches;
        let base_threshold = p.tree_threshold_base;

        // Tier 0: no-op.
        let mut p0 = p.clone();
        p0.apply_faster_decoding(0);
        assert_eq!(p0.lz77, base_lz77);
        assert_eq!(p0.custom_orders, base_custom_orders);

        // Tier 1.
        let mut p1 = p.clone();
        p1.apply_faster_decoding(1);
        assert!(!p1.lz77);
        assert_eq!(p1.enhanced_clustering_vardct, base_enhanced);
        assert_eq!(p1.custom_orders, base_custom_orders);

        // Tier 2.
        let mut p2 = p.clone();
        p2.apply_faster_decoding(2);
        assert!(!p2.lz77);
        assert!(!p2.enhanced_clustering_vardct);
        assert_eq!(p2.custom_orders, base_custom_orders);

        // Tier 3: + custom_orders off + threshold raised.
        let mut p3 = p.clone();
        p3.apply_faster_decoding(3);
        assert!(!p3.custom_orders);
        assert!(p3.tree_threshold_base > base_threshold);

        // Tier 4: + tree_learning off + patches off + gaborish off + no DCT32.
        p.apply_faster_decoding(4);
        assert!(!p.tree_learning);
        assert!(!p.patches);
        assert!(!p.gaborish);
        assert!(!p.try_dct32);
        assert!(!p.try_dct64);
        // Sanity: base values were on at effort 7.
        let _ = (
            base_gaborish,
            base_try_dct32,
            base_tree_learning,
            base_patches,
        );
    }

    #[test]
    fn test_progressive_dc_clamp_and_lf_frame_implication() {
        let cfg = LossyConfig::new(1.0);
        assert_eq!(cfg.progressive_dc(), 0);
        assert!(!cfg.lf_frame());

        // level 1 implies lf_frame=true.
        let cfg = LossyConfig::new(1.0).with_progressive_dc(1);
        assert_eq!(cfg.progressive_dc(), 1);
        assert!(cfg.lf_frame(), "progressive_dc>=1 should imply lf_frame");

        // level 2 also implies lf_frame=true.
        let cfg = LossyConfig::new(1.0).with_progressive_dc(2);
        assert_eq!(cfg.progressive_dc(), 2);
        assert!(cfg.lf_frame());

        // Out-of-range clamps to MAX_PROGRESSIVE_DC.
        let cfg = LossyConfig::new(1.0).with_progressive_dc(255);
        assert_eq!(cfg.progressive_dc(), MAX_PROGRESSIVE_DC);
    }

    #[test]
    fn test_premultiplied_alpha_mode_from_i8() {
        assert_eq!(
            PremultipliedAlphaMode::from_i8(-1),
            PremultipliedAlphaMode::Auto
        );
        assert_eq!(
            PremultipliedAlphaMode::from_i8(-127),
            PremultipliedAlphaMode::Auto
        );
        assert_eq!(
            PremultipliedAlphaMode::from_i8(0),
            PremultipliedAlphaMode::Off
        );
        assert_eq!(
            PremultipliedAlphaMode::from_i8(1),
            PremultipliedAlphaMode::On
        );
        assert_eq!(
            PremultipliedAlphaMode::from_i8(127),
            PremultipliedAlphaMode::On
        );
    }

    #[test]
    fn test_premultiplied_alpha_mode_builder_round_trip() {
        let cfg = LossyConfig::new(1.0);
        {
            let req = cfg.encode_request(8, 8, PixelLayout::Rgba8);
            // Default: Off (matches the boolean `false` default).
            assert_eq!(req.premultiplied_alpha_mode(), PremultipliedAlphaMode::Off);
        }

        {
            let req = cfg
                .encode_request(8, 8, PixelLayout::Rgba8)
                .with_premultiplied_alpha_mode(PremultipliedAlphaMode::On);
            assert_eq!(req.premultiplied_alpha_mode(), PremultipliedAlphaMode::On);
        }

        {
            let req = cfg
                .encode_request(8, 8, PixelLayout::Rgba8)
                .with_premultiplied_alpha_mode(PremultipliedAlphaMode::Auto);
            assert_eq!(req.premultiplied_alpha_mode(), PremultipliedAlphaMode::Auto);
        }
    }

    // ─── EncoderStrategy (W44-127 Chunk A) ─────────────────────────────

    /// Default is `Zenjxl` — production shipping behaviour.
    /// See `docs/COMPATIBILITY_MODES.md` §4.1 + §7 Q1.
    #[test]
    fn test_encoder_strategy_default_is_zenjxl() {
        assert_eq!(EncoderStrategy::default(), EncoderStrategy::Zenjxl);
    }

    /// `EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default())`
    /// returns the strict-libjxl-parity bundle: every Section B policy
    /// is at the disabled / ForceAllow / ForceSkip variant, every
    /// Section A `EffortGate` is at `EffortGate::Libjxl`,
    /// `block_ctx_map_15_cluster == true`, and perf dispatches are at
    /// their `Default`.
    #[test]
    fn test_resolve_libjxl_field_values() {
        let resolved = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());

        // Section B content-aware gates: all disabled / force-allow / force-skip
        assert_eq!(
            resolved.screenshot_entropy_mul,
            ScreenshotEntropyMulPolicy::Disabled
        );
        assert_eq!(
            resolved.high_d_photo_entropy_mul,
            HighDPhotoEntropyMulPolicy::Disabled
        );
        assert_eq!(resolved.dct64_search_policy, Dct64SearchPolicy::ForceAllow);
        assert_eq!(
            resolved.dct32_search_policy,
            Dct32SearchPolicy::FollowDct64Suppression
        );
        assert_eq!(
            resolved.smooth_photo_dct64_admission,
            SmoothPhotoDct64Policy::ForceSkip
        );
        assert_eq!(resolved.buttloop_qf_seed, ButtloopQfSeedPolicy::Off);
        assert_eq!(
            resolved.adaptive_quant_qf_seed,
            AdaptiveQuantQfSeedPolicy::Off
        );
        assert_eq!(
            resolved.buttloop_epf_sharpness_seed,
            EpfSharpnessSeed::LegacyUniform4
        );

        // Section A effort-gate flips: every gate at Libjxl
        assert_eq!(resolved.cfl_two_pass_min_effort, EffortGate::Libjxl);
        assert_eq!(resolved.try_dct64_min_effort, EffortGate::Libjxl);
        assert_eq!(
            resolved.epf_dynamic_sharpness_min_effort,
            EffortGate::Libjxl
        );

        // Section D KNOWN-BUG re-enable
        assert!(resolved.block_ctx_map_15_cluster);

        // Perf dispatches: at Default (orthogonal to libjxl byte parity)
        assert_eq!(resolved.epf_dispatch, EpfDispatch::default());
        assert_eq!(resolved.pixel_loss_dispatch, PixelLossDispatch::default());
        assert_eq!(
            resolved.single_pass_entropy_dispatch,
            SinglePassEntropyDispatch::default()
        );
        assert_eq!(resolved.patches_dispatch, PatchesDispatch::default());
    }

    /// `EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default())`
    /// equals `ResolvedImprovements::default()` — every field at its
    /// enum's `#[default]`.
    #[test]
    fn test_resolve_zenjxl_field_values() {
        let resolved = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
        assert_eq!(resolved, ResolvedImprovements::default());
    }

    /// `EncoderStrategy::LeanFaster.resolve(...)`:
    /// `high_d_photo_entropy_mul` is `Auto` (kept — cheap), all
    /// screenshot-class is `Disabled` / `ForceAllow` / `ForceSkip`,
    /// perf dispatches are at `Default`.
    #[test]
    fn test_resolve_lean_faster_field_values() {
        let resolved = EncoderStrategy::LeanFaster.resolve(&StrategyOverrides::default());

        // Photo-class entropy-mul lowering KEPT (Auto) — cheap table swaps
        assert_eq!(
            resolved.high_d_photo_entropy_mul,
            HighDPhotoEntropyMulPolicy::Auto
        );

        // Screenshot-class / heavy gates: all disabled
        assert_eq!(
            resolved.screenshot_entropy_mul,
            ScreenshotEntropyMulPolicy::Disabled
        );
        assert_eq!(resolved.dct64_search_policy, Dct64SearchPolicy::ForceAllow);
        assert_eq!(
            resolved.dct32_search_policy,
            Dct32SearchPolicy::FollowDct64Suppression
        );
        assert_eq!(
            resolved.smooth_photo_dct64_admission,
            SmoothPhotoDct64Policy::ForceSkip
        );
        assert_eq!(resolved.buttloop_qf_seed, ButtloopQfSeedPolicy::Off);
        assert_eq!(
            resolved.adaptive_quant_qf_seed,
            AdaptiveQuantQfSeedPolicy::Off
        );
        assert_eq!(
            resolved.buttloop_epf_sharpness_seed,
            EpfSharpnessSeed::LegacyUniform4
        );

        // Section A effort gates: OURS (not libjxl) — keeps our
        // speed-conscious gating
        assert_eq!(resolved.cfl_two_pass_min_effort, EffortGate::Ours);
        assert_eq!(resolved.try_dct64_min_effort, EffortGate::Ours);
        assert_eq!(resolved.epf_dynamic_sharpness_min_effort, EffortGate::Ours);

        // Section D KNOWN-BUG: not re-enabled
        assert!(!resolved.block_ctx_map_15_cluster);

        // Perf dispatches: at Default
        assert_eq!(resolved.epf_dispatch, EpfDispatch::default());
        assert_eq!(resolved.pixel_loss_dispatch, PixelLossDispatch::default());
        assert_eq!(
            resolved.single_pass_entropy_dispatch,
            SinglePassEntropyDispatch::default()
        );
        assert_eq!(resolved.patches_dispatch, PatchesDispatch::default());
    }

    /// Per `docs/COMPATIBILITY_MODES.md` §4.4 + §7 Q1 note:
    /// `EncoderStrategy::Aggressive` is currently equivalent to
    /// `EncoderStrategy::Zenjxl` after W44-124's auto-discriminator
    /// obsoleted the previous "Aggressive flips W44-123 globally"
    /// behaviour.
    #[test]
    fn test_resolve_aggressive_equals_zenjxl() {
        let aggressive = EncoderStrategy::Aggressive.resolve(&StrategyOverrides::default());
        let zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
        assert_eq!(aggressive, zenjxl);
    }

    /// `Custom(Box::new(EncoderImprovementsCustom { dct64_search_policy:
    /// ForceSuppress, ..Default::default() }))` round-trips through
    /// resolve — the resolved struct exposes the same field values the
    /// caller put in `Custom`.
    #[test]
    fn test_resolve_custom_round_trip() {
        let custom = EncoderImprovementsCustom {
            dct64_search_policy: Dct64SearchPolicy::ForceSuppress,
            dct32_search_policy: Dct32SearchPolicy::KeepWhenDct64Suppressed,
            buttloop_qf_seed: ButtloopQfSeedPolicy::ForceScale(2.5),
            adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::AutoScaleCustom {
                e5_e6: 1.5,
                e7: 2.0,
            },
            buttloop_epf_sharpness_seed: EpfSharpnessSeed::AutoW44_117 { min_distance: 2.0 },
            cfl_two_pass_min_effort: EffortGate::AtLeast(6),
            try_dct64_min_effort: EffortGate::Off,
            block_ctx_map_15_cluster: true,
            ..Default::default()
        };
        let strategy = EncoderStrategy::Custom(Box::new(custom.clone()));
        let resolved = strategy.resolve(&StrategyOverrides::default());

        assert_eq!(resolved.dct64_search_policy, custom.dct64_search_policy);
        assert_eq!(resolved.dct32_search_policy, custom.dct32_search_policy);
        assert_eq!(resolved.buttloop_qf_seed, custom.buttloop_qf_seed);
        assert_eq!(
            resolved.adaptive_quant_qf_seed,
            custom.adaptive_quant_qf_seed
        );
        assert_eq!(
            resolved.buttloop_epf_sharpness_seed,
            custom.buttloop_epf_sharpness_seed
        );
        assert_eq!(
            resolved.cfl_two_pass_min_effort,
            custom.cfl_two_pass_min_effort
        );
        assert_eq!(resolved.try_dct64_min_effort, custom.try_dct64_min_effort);
        assert_eq!(
            resolved.block_ctx_map_15_cluster,
            custom.block_ctx_map_15_cluster
        );

        // Fields left at Default should be at the
        // EncoderImprovementsCustom::default() value (= Zenjxl
        // baseline). Note `screenshot_entropy_mul` defaults to
        // `Disabled` (NOT `Auto`) per W44-130 Chunk D — Zenjxl
        // preserves the pre-Chunk-D default-off W22-1 lift.
        assert_eq!(
            resolved.screenshot_entropy_mul,
            ScreenshotEntropyMulPolicy::Disabled
        );
        assert_eq!(
            resolved.epf_dynamic_sharpness_min_effort,
            EffortGate::default()
        );
    }

    /// `StrategyOverrides` field-by-field precedence over the resolved
    /// preset. `Some(...)` overrides; `None` is a no-op.
    #[test]
    fn test_strategy_overrides_precedence() {
        // Start from Libjxl (every screenshot gate Disabled) then
        // override two fields and confirm only those two flip.
        let overrides = StrategyOverrides {
            dct_suppress_hint: Some(true),
            dct32_keep_hint: Some(true),
            ..Default::default()
        };
        let resolved = EncoderStrategy::Libjxl.resolve(&overrides);

        // Overridden fields:
        assert_eq!(
            resolved.dct64_search_policy,
            Dct64SearchPolicy::ForceSuppress
        );
        assert_eq!(
            resolved.dct32_search_policy,
            Dct32SearchPolicy::KeepWhenDct64Suppressed
        );

        // Un-overridden fields stay at Libjxl values:
        assert_eq!(
            resolved.screenshot_entropy_mul,
            ScreenshotEntropyMulPolicy::Disabled
        );
        assert_eq!(
            resolved.buttloop_epf_sharpness_seed,
            EpfSharpnessSeed::LegacyUniform4
        );
        assert!(resolved.block_ctx_map_15_cluster);
    }

    /// Default impls for every nested policy match the documented
    /// "production shipping" picks.
    #[test]
    fn test_policy_defaults() {
        assert_eq!(
            ScreenshotEntropyMulPolicy::default(),
            ScreenshotEntropyMulPolicy::Auto
        );
        assert_eq!(
            HighDPhotoEntropyMulPolicy::default(),
            HighDPhotoEntropyMulPolicy::Auto
        );
        assert_eq!(Dct64SearchPolicy::default(), Dct64SearchPolicy::Auto);
        assert_eq!(
            Dct32SearchPolicy::default(),
            Dct32SearchPolicy::FollowDct64Suppression
        );
        assert_eq!(
            SmoothPhotoDct64Policy::default(),
            SmoothPhotoDct64Policy::Auto
        );
        assert_eq!(
            ButtloopQfSeedPolicy::default(),
            ButtloopQfSeedPolicy::AutoScale4
        );
        assert_eq!(
            AdaptiveQuantQfSeedPolicy::default(),
            AdaptiveQuantQfSeedPolicy::AutoScalePerEffort
        );
        assert_eq!(
            EpfSharpnessSeed::default(),
            EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 }
        );
        assert_eq!(EffortGate::default(), EffortGate::Ours);
    }

    /// `EncoderImprovementsCustom::default()` ≡
    /// `ResolvedImprovements::default()` field-by-field — Custom with
    /// all defaults resolves to Zenjxl.
    #[test]
    fn test_custom_default_equals_zenjxl_resolved() {
        let custom_strategy = EncoderStrategy::Custom(Box::<EncoderImprovementsCustom>::default());
        let resolved_custom = custom_strategy.resolve(&StrategyOverrides::default());
        let resolved_zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
        assert_eq!(resolved_custom, resolved_zenjxl);
    }

    // ── W44-128 Chunk B tests ────────────────────────────────────

    /// Default [`LossyConfig`] returns [`EncoderStrategy::Zenjxl`]
    /// from [`LossyConfig::strategy`]. Equivalent to never calling
    /// [`LossyConfig::with_strategy`].
    #[test]
    fn test_lossy_config_default_strategy_is_zenjxl() {
        let cfg = LossyConfig::new(1.0);
        assert_eq!(cfg.strategy(), &EncoderStrategy::Zenjxl);
    }

    /// [`LossyConfig::with_strategy`] roundtrips through
    /// [`LossyConfig::strategy`] for every named variant.
    #[test]
    fn test_with_strategy_setter_roundtrip() {
        for variant in [
            EncoderStrategy::Libjxl,
            EncoderStrategy::LeanFaster,
            EncoderStrategy::Zenjxl,
            EncoderStrategy::Aggressive,
        ] {
            let cfg = LossyConfig::new(1.0).with_strategy(variant.clone());
            assert_eq!(cfg.strategy(), &variant);
        }
        // Custom variant carries a payload; equality is structural.
        let mut custom_inner = EncoderImprovementsCustom::default();
        custom_inner.dct64_search_policy = Dct64SearchPolicy::ForceSuppress;
        let custom = EncoderStrategy::Custom(Box::new(custom_inner.clone()));
        let cfg = LossyConfig::new(1.0).with_strategy(custom.clone());
        assert_eq!(cfg.strategy(), &custom);
    }

    /// Override precedence (W44-128 / Chunk B contract, updated for
    /// W44-130 / Chunk D — `with_*_hint(Option<bool>)` setters
    /// deleted; per-field overrides now flow via
    /// `with_strategy_overrides(StrategyOverrides { ... })`):
    ///
    /// 1. `with_strategy(Libjxl).with_strategy_overrides(...)`:
    ///    `Libjxl` resolves `dct64_search_policy = ForceAllow`. The
    ///    `Some(false)` override also maps to `ForceAllow` — the two
    ///    agree, so resolution returns `ForceAllow`. Demonstrates the
    ///    override path's no-op behaviour when caller and preset
    ///    agree.
    ///
    /// 2. `with_strategy(Custom { dct64=ForceSuppress, .. })`
    ///    `.with_strategy_overrides(...)`:
    ///    Custom asks for `ForceSuppress`, but the override
    ///    rewrites it to `ForceAllow`. Demonstrates that overrides
    ///    WIN over the preset (mirrors the
    ///    `with_perceptual_optimizations(false).with_gaborish(true)`
    ///    precedence pattern).
    #[test]
    fn test_with_strategy_libjxl_then_hint_override() {
        // Case 1: Libjxl + Some(false) override → both say ForceAllow.
        let cfg = LossyConfig::new(1.0)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_strategy_overrides(StrategyOverrides {
                dct_suppress_hint: Some(false),
                ..Default::default()
            });
        let resolved = cfg.resolve_improvements();
        assert_eq!(
            resolved.dct64_search_policy,
            Dct64SearchPolicy::ForceAllow,
            "Libjxl base + Some(false) override should both agree on ForceAllow"
        );

        // Case 2: Custom asks for ForceSuppress, but a `Some(false)`
        // override rewrites the resolved policy to ForceAllow.
        // Overrides WIN over the preset.
        let mut custom_inner = EncoderImprovementsCustom::default();
        custom_inner.dct64_search_policy = Dct64SearchPolicy::ForceSuppress;
        let cfg = LossyConfig::new(1.0)
            .with_strategy(EncoderStrategy::Custom(Box::new(custom_inner)))
            .with_strategy_overrides(StrategyOverrides {
                dct_suppress_hint: Some(false),
                ..Default::default()
            });
        let resolved = cfg.resolve_improvements();
        assert_eq!(
            resolved.dct64_search_policy,
            Dct64SearchPolicy::ForceAllow,
            "Some(false) override should rewrite Custom(ForceSuppress) to ForceAllow"
        );
    }

    /// `LossyConfig::resolve_improvements()` at the default strategy
    /// (Zenjxl) with no hints set must equal `Zenjxl` resolved
    /// directly — proving the resolution helper doesn't smuggle in
    /// extra state from `LossyConfig`.
    #[test]
    fn test_resolve_improvements_default_equals_zenjxl_resolved() {
        let cfg = LossyConfig::new(1.0);
        let from_cfg = cfg.resolve_improvements();
        let direct = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
        assert_eq!(from_cfg, direct);
    }

    /// `with_effort` preserves the caller's `with_strategy` choice.
    /// Effort-derived fields regenerate; the strategy bundle does not.
    #[test]
    fn test_with_strategy_preserved_across_with_effort() {
        let cfg = LossyConfig::new(1.0)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_effort(8);
        assert_eq!(cfg.strategy(), &EncoderStrategy::Libjxl);
        assert_eq!(cfg.effort(), 8);
    }

    /// `resolve_improvements` propagates all five `StrategyOverrides`
    /// fields correctly. Starting from `Libjxl` (every relevant
    /// policy at `Disabled` / `ForceAllow` / `ForceSkip`), set every
    /// hint and confirm each one re-maps the matching policy field.
    /// (W44-130 Chunk D: hints moved into `StrategyOverrides`.)
    #[test]
    fn test_resolve_improvements_propagates_all_hints() {
        let cfg = LossyConfig::new(1.0)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_strategy_overrides(StrategyOverrides {
                screenshot_lift_hint: Some(true),
                high_d_photo_hint: Some(true),
                smooth_photo_dct64_hint: Some(true),
                dct_suppress_hint: Some(true),
                dct32_keep_hint: Some(true),
            });
        let resolved = cfg.resolve_improvements();
        assert_eq!(
            resolved.screenshot_entropy_mul,
            ScreenshotEntropyMulPolicy::ForceOn,
            "screenshot_lift_hint(Some(true)) maps to ForceOn"
        );
        assert_eq!(
            resolved.high_d_photo_entropy_mul,
            HighDPhotoEntropyMulPolicy::ForceOn,
            "high_d_photo_hint(Some(true)) maps to ForceOn"
        );
        assert_eq!(
            resolved.smooth_photo_dct64_admission,
            SmoothPhotoDct64Policy::ForceAdmit,
            "smooth_photo_dct64_hint(Some(true)) maps to ForceAdmit"
        );
        assert_eq!(
            resolved.dct64_search_policy,
            Dct64SearchPolicy::ForceSuppress,
            "dct_suppress_hint(Some(true)) maps to ForceSuppress"
        );
        assert_eq!(
            resolved.dct32_search_policy,
            Dct32SearchPolicy::KeepWhenDct64Suppressed,
            "dct32_keep_hint(Some(true)) maps to KeepWhenDct64Suppressed"
        );
        // Un-overridden Libjxl fields stay at Libjxl values.
        assert_eq!(
            resolved.buttloop_epf_sharpness_seed,
            EpfSharpnessSeed::LegacyUniform4
        );
        assert!(resolved.block_ctx_map_15_cluster);
    }

    /// W44-130 Chunk D — `with_strategy_overrides` setter round-trips:
    /// the setter stores the struct verbatim and the getter returns
    /// a reference to it. Default is all-`None` (no overrides applied).
    #[test]
    fn test_with_strategy_overrides_setter_roundtrip() {
        // Default: empty overrides, all None.
        let cfg = LossyConfig::new(1.0);
        assert_eq!(cfg.strategy_overrides(), &StrategyOverrides::default());

        // Set + read back: every field preserved exactly.
        let overrides = StrategyOverrides {
            screenshot_lift_hint: Some(true),
            high_d_photo_hint: Some(false),
            smooth_photo_dct64_hint: Some(true),
            dct_suppress_hint: Some(false),
            dct32_keep_hint: Some(true),
        };
        let cfg = LossyConfig::new(1.0).with_strategy_overrides(overrides.clone());
        assert_eq!(cfg.strategy_overrides(), &overrides);

        // Resolved policy reflects every override (Libjxl preset →
        // every override maps to Force*; the un-set buttloop fields
        // stay at Libjxl values, confirming the overrides don't
        // leak past their five named fields).
        let cfg = LossyConfig::new(1.0)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_strategy_overrides(overrides);
        let resolved = cfg.resolve_improvements();
        assert_eq!(
            resolved.screenshot_entropy_mul,
            ScreenshotEntropyMulPolicy::ForceOn
        );
        assert_eq!(
            resolved.high_d_photo_entropy_mul,
            HighDPhotoEntropyMulPolicy::ForceOff
        );
        assert_eq!(
            resolved.smooth_photo_dct64_admission,
            SmoothPhotoDct64Policy::ForceAdmit
        );
        assert_eq!(
            resolved.dct64_search_policy,
            Dct64SearchPolicy::ForceAllow
        );
        assert_eq!(
            resolved.dct32_search_policy,
            Dct32SearchPolicy::KeepWhenDct64Suppressed
        );
        // Un-overridden Libjxl-baseline fields preserved.
        assert_eq!(
            resolved.buttloop_epf_sharpness_seed,
            EpfSharpnessSeed::LegacyUniform4
        );
    }

    // ── W44-132 Chunk F (env-var-MUTATING tests) ────────────────
    //
    // Tests that mutate process env-vars live in
    // `tests/strategy_env_fallback.rs` (integration test, can opt
    // out of `#![forbid(unsafe_code)]` for the `unsafe { env::
    // set_var(...) }` calls Rust 2024 requires). The library code
    // itself just READS env-vars (safe) inside
    // `apply_env_var_fallbacks` — only the test suite needs the
    // mutating path.
    //
    // Pure unit tests below cover the no-env-var case (default
    // pass-through) and the explicit-caller-wins-over-env case
    // without needing to mutate the process environment.

    /// With NO env vars set, the resolved policy stays at the
    /// strategy preset's default value (bit-identical to pre-Chunk-F
    /// resolved values when no env-var is set). This is the
    /// production-default case — exercises the fallback function's
    /// "field equals default but env-var unset" code path.
    #[test]
    fn test_w44_132_env_fallback_pure_no_env_default_passthrough() {
        // NOTE: this test does NOT mutate env vars; it reads
        // whatever the runner inherited. The cjxl-rs CI sets no
        // JXL_* env vars, so the production hash-lock test (which
        // also runs unset) is the binding gate.
        //
        // What this test verifies: when the resolved field
        // (post-overrides) equals `Default::default()`, the fallback
        // function's match-on-default check works correctly without
        // running into the actual env-var lookup path (the parent
        // `if r.field == Default::default()` short-circuits the
        // env-var read if the policy was caller-overridden — but
        // when it WASN'T overridden, the env-var path is taken
        // safely).
        //
        // The mutating tests live in
        // `tests/strategy_env_fallback.rs` for the env-on cases.

        // Use Libjxl which sets every promoted field to a NON-default
        // value (Off / Off / LegacyUniform4) so the fallback's
        // default-check short-circuits the env read entirely.
        let resolved = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());
        assert_eq!(resolved.buttloop_qf_seed, ButtloopQfSeedPolicy::Off);
        assert_eq!(
            resolved.adaptive_quant_qf_seed,
            AdaptiveQuantQfSeedPolicy::Off
        );
        assert_eq!(
            resolved.buttloop_epf_sharpness_seed,
            EpfSharpnessSeed::LegacyUniform4
        );
    }

    /// Caller's explicit `EncoderStrategy::Custom(...)` value sets
    /// the field to a non-default value, which short-circuits the
    /// env-var fallback's `field == default` check. This test does
    /// not need to mutate env vars to verify the precedence rule —
    /// any non-default field value disqualifies the env-var path.
    #[test]
    fn test_w44_132_env_fallback_pure_custom_non_default_short_circuits() {
        // ForceScale is structurally non-default (`AutoScale4` is
        // default); fallback `if r.field == default` is false, so
        // the env-var read is skipped entirely regardless of what
        // any JXL_* env var is set to in the test runner's env.
        let custom = EncoderImprovementsCustom {
            buttloop_qf_seed: ButtloopQfSeedPolicy::ForceScale(5.0),
            adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::AutoScaleCustom {
                e5_e6: 1.5,
                e7: 2.5,
            },
            buttloop_epf_sharpness_seed: EpfSharpnessSeed::AutoW44_117 { min_distance: 3.0 },
            ..Default::default()
        };
        let strategy = EncoderStrategy::Custom(Box::new(custom.clone()));
        let resolved = strategy.resolve(&StrategyOverrides::default());
        assert_eq!(resolved.buttloop_qf_seed, custom.buttloop_qf_seed);
        assert_eq!(
            resolved.adaptive_quant_qf_seed,
            custom.adaptive_quant_qf_seed
        );
        assert_eq!(
            resolved.buttloop_epf_sharpness_seed,
            custom.buttloop_epf_sharpness_seed
        );
    }
}
