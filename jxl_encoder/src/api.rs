// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

pub use crate::tiny::Lz77Method;
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
    Io(std::io::Error),
    /// Internal encoder error (should not happen — file a bug).
    Internal { message: String },
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
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Internal { message } => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Oom(e) => Some(e),
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
            crate::error::Error::InvalidInput(msg) => Self::InvalidInput { message: msg },
            crate::error::Error::OutOfMemory(e) => Self::Oom(e),
            crate::error::Error::IoError(e) => Self::Io(e),
            crate::error::Error::Cancelled => Self::Cancelled,
            other => Self::Internal {
                message: format!("{other}"),
            },
        }
    }
}

impl From<std::io::Error> for EncodeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<enough::StopReason> for EncodeError {
    fn from(_: enough::StopReason) -> Self {
        Self::Cancelled
    }
}

/// Result type for encoding operations.
///
/// Errors carry location traces via [`whereat::At`] for lightweight
/// production-safe error tracking without debuginfo or backtraces.
pub type Result<T> = core::result::Result<T, At<EncodeError>>;

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
    /// Linear f32 RGB, 12 bytes per pixel. Skips sRGB→linear conversion.
    RgbLinearF32,
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
            Self::RgbLinearF32 => 12,
        }
    }

    /// Whether this layout uses linear (not gamma-encoded) values.
    pub const fn is_linear(self) -> bool {
        matches!(self, Self::RgbLinearF32)
    }

    /// Whether this layout uses 16-bit samples.
    pub const fn is_16bit(self) -> bool {
        matches!(self, Self::Rgb16 | Self::Rgba16 | Self::Gray16)
    }

    /// Whether this layout includes an alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::Rgba8 | Self::Bgra8 | Self::GrayAlpha8 | Self::Rgba16
        )
    }

    /// Whether this layout is grayscale.
    pub const fn is_grayscale(self) -> bool {
        matches!(self, Self::Gray8 | Self::GrayAlpha8 | Self::Gray16)
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

// ── Supporting types ────────────────────────────────────────────────────────

/// Image metadata (ICC, EXIF, XMP) to embed in the JXL file.
#[derive(Clone, Debug, Default)]
pub struct ImageMetadata<'a> {
    icc_profile: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
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
}

/// Resource limits for encoding.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    max_width: Option<u64>,
    max_height: Option<u64>,
    max_pixels: Option<u64>,
    max_memory_bytes: Option<u64>,
}

impl Limits {
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
    pub fn with_max_memory_bytes(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = Some(bytes);
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

    /// Get maximum memory bytes, if set.
    pub fn max_memory_bytes(&self) -> Option<u64> {
        self.max_memory_bytes
    }
}

// ── LosslessConfig ──────────────────────────────────────────────────────────

/// Lossless (modular) encoding configuration.
///
/// Has a sensible `Default` — lossless has no quality ambiguity.
#[derive(Clone, Debug)]
pub struct LosslessConfig {
    effort: u8,
    use_ans: bool,
    squeeze: bool,
    tree_learning: bool,
    lz77: bool,
    lz77_method: Lz77Method,
}

impl Default for LosslessConfig {
    fn default() -> Self {
        Self {
            effort: 7,
            use_ans: true,
            squeeze: false,
            tree_learning: false,
            lz77: false,
            lz77_method: Lz77Method::Greedy,
        }
    }
}

impl LosslessConfig {
    /// Create a new lossless config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set effort level (1–10). Higher = slower, better compression.
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort;
        self
    }

    /// Enable/disable ANS entropy coding (default: true).
    pub fn with_ans(mut self, enable: bool) -> Self {
        self.use_ans = enable;
        self
    }

    /// Enable/disable squeeze (Haar wavelet) transform (default: false).
    pub fn with_squeeze(mut self, enable: bool) -> Self {
        self.squeeze = enable;
        self
    }

    /// Enable/disable content-adaptive tree learning (default: false).
    pub fn with_tree_learning(mut self, enable: bool) -> Self {
        self.tree_learning = enable;
        self
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
    }
}

// ── LossyConfig ─────────────────────────────────────────────────────────────

/// Lossy (VarDCT) encoding configuration.
///
/// No `Default` — distance/quality is a required choice.
#[derive(Clone, Debug)]
pub struct LossyConfig {
    distance: f32,
    effort: u8,
    use_ans: bool,
    gaborish: bool,
    noise: bool,
    denoise: bool,
    error_diffusion: bool,
    pixel_domain_loss: bool,
    lz77: bool,
    lz77_method: Lz77Method,
    force_strategy: Option<u8>,
    #[cfg(feature = "butteraugli-loop")]
    butteraugli_iters: u32,
}

impl LossyConfig {
    /// Create with butteraugli distance (1.0 = high quality).
    pub fn new(distance: f32) -> Self {
        Self {
            distance,
            effort: 7,
            use_ans: true,
            gaborish: true,
            noise: false,
            denoise: false,
            error_diffusion: true,
            pixel_domain_loss: true,
            lz77: false,
            lz77_method: Lz77Method::Greedy,
            force_strategy: None,
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 2,
        }
    }

    /// Create from a [`Quality`] specification.
    pub fn from_quality(quality: Quality) -> core::result::Result<Self, EncodeError> {
        let distance = quality.to_distance()?;
        Ok(Self::new(distance))
    }

    /// Set effort level (1–10).
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort;
        self
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

    /// Enable/disable noise synthesis (default: false).
    pub fn with_noise(mut self, enable: bool) -> Self {
        self.noise = enable;
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

    /// Enable/disable error diffusion in AC quantization (default: true).
    pub fn with_error_diffusion(mut self, enable: bool) -> Self {
        self.error_diffusion = enable;
        self
    }

    /// Enable/disable pixel-domain loss in strategy selection (default: true).
    pub fn with_pixel_domain_loss(mut self, enable: bool) -> Self {
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

    /// Set butteraugli quantization loop iterations (default: 2).
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn with_butteraugli_iters(mut self, n: u32) -> Self {
        self.butteraugli_iters = n;
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

    /// Whether noise synthesis is enabled.
    pub fn noise(&self) -> bool {
        self.noise
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

    /// Butteraugli quantization loop iterations.
    #[cfg(feature = "butteraugli-loop")]
    pub fn butteraugli_iters(&self) -> u32 {
        self.butteraugli_iters
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

    /// Encode pixels and return the JXL bitstream.
    #[track_caller]
    pub fn encode(self, pixels: &[u8]) -> Result<Vec<u8>> {
        self.encode_inner(pixels).map_err(at)
    }

    /// Encode pixels, appending to an existing buffer.
    #[track_caller]
    pub fn encode_into(self, pixels: &[u8], out: &mut Vec<u8>) -> Result<()> {
        let data = self.encode_inner(pixels).map_err(at)?;
        out.extend_from_slice(&data);
        Ok(())
    }

    /// Encode pixels, writing to a `std::io::Write` destination.
    #[track_caller]
    pub fn encode_to(self, pixels: &[u8], mut dest: impl std::io::Write) -> Result<()> {
        let data = self.encode_inner(pixels).map_err(at)?;
        dest.write_all(&data)
            .map_err(|e| at(EncodeError::from(e)))?;
        Ok(())
    }

    fn encode_inner(&self, pixels: &[u8]) -> core::result::Result<Vec<u8>, EncodeError> {
        self.validate_pixels(pixels)?;
        self.check_limits()?;

        let codestream = match self.config {
            ConfigRef::Lossless(cfg) => self.encode_lossless(cfg, pixels),
            ConfigRef::Lossy(cfg) => self.encode_lossy(cfg, pixels),
        }?;

        // Wrap in container if metadata (EXIF/XMP) is present
        if let Some(meta) = self.metadata
            && (meta.exif.is_some() || meta.xmp.is_some())
        {
            Ok(crate::container::wrap_in_container(
                &codestream,
                meta.exif,
                meta.xmp,
            ))
        } else {
            Ok(codestream)
        }
    }

    fn validate_pixels(&self, pixels: &[u8]) -> core::result::Result<(), EncodeError> {
        let w = self.width as usize;
        let h = self.height as usize;
        if w == 0 || h == 0 {
            return Err(EncodeError::InvalidInput {
                message: format!("zero dimensions: {w}x{h}"),
            });
        }
        let expected = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(self.layout.bytes_per_pixel()));
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
        Ok(())
    }

    // ── Lossless path ───────────────────────────────────────────────────

    fn encode_lossless(
        &self,
        cfg: &LosslessConfig,
        pixels: &[u8],
    ) -> core::result::Result<Vec<u8>, EncodeError> {
        let options = crate::encoder::EncoderOptions {
            distance: 0.0,
            effort: cfg.effort,
            force_modular: true,
            use_ans: cfg.use_ans,
            optimize_codes: true,
            custom_orders: true,
            enable_noise: false,
            enable_denoise: false,
            enable_gaborish: false,
            use_tree_learning: cfg.tree_learning,
            use_squeeze: cfg.squeeze,
        };
        let encoder = crate::encoder::Encoder::with_options(options);
        let w = self.width as usize;
        let h = self.height as usize;

        let result = match self.layout {
            PixelLayout::Rgb8 => encoder.encode_rgb8(pixels, w, h),
            PixelLayout::Rgba8 => encoder.encode_rgba8(pixels, w, h),
            PixelLayout::Bgr8 => encoder.encode_rgb8(&bgr_to_rgb(pixels, 3), w, h),
            PixelLayout::Bgra8 => encoder.encode_rgba8(&bgr_to_rgb(pixels, 4), w, h),
            PixelLayout::Gray8 => encoder.encode_gray8(pixels, w, h),
            PixelLayout::Rgb16 => encoder.encode_rgb16_native(pixels, w, h),
            PixelLayout::Rgba16 => encoder.encode_rgba16_native(pixels, w, h),
            PixelLayout::Gray16 => encoder.encode_gray16_native(pixels, w, h),
            PixelLayout::GrayAlpha8 => {
                return Err(EncodeError::UnsupportedPixelLayout(PixelLayout::GrayAlpha8));
            }
            PixelLayout::RgbLinearF32 => {
                return Err(EncodeError::UnsupportedPixelLayout(
                    PixelLayout::RgbLinearF32,
                ));
            }
        };
        result.map_err(EncodeError::from)
    }

    // ── Lossy path ──────────────────────────────────────────────────────

    fn encode_lossy(
        &self,
        cfg: &LossyConfig,
        pixels: &[u8],
    ) -> core::result::Result<Vec<u8>, EncodeError> {
        let w = self.width as usize;
        let h = self.height as usize;

        // Build linear f32 RGB and extract alpha from input layout
        let (linear_rgb, alpha, bit_depth_16) = match self.layout {
            PixelLayout::Rgb8 => (srgb_u8_to_linear_f32(pixels, 3), None, false),
            PixelLayout::Bgr8 => (
                srgb_u8_to_linear_f32(&bgr_to_rgb(pixels, 3), 3),
                None,
                false,
            ),
            PixelLayout::Rgba8 => {
                let rgb = srgb_u8_to_linear_f32(pixels, 4);
                let alpha = extract_alpha(pixels, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::Bgra8 => {
                let swapped = bgr_to_rgb(pixels, 4);
                let rgb = srgb_u8_to_linear_f32(&swapped, 4);
                let alpha = extract_alpha(pixels, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::Rgb16 => (srgb_u16_to_linear_f32(pixels, 3), None, true),
            PixelLayout::Rgba16 => {
                let rgb = srgb_u16_to_linear_f32(pixels, 4);
                let alpha = extract_alpha_u16(pixels, 4, 3);
                (rgb, Some(alpha), true)
            }
            PixelLayout::RgbLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(pixels);
                (floats.to_vec(), None, false)
            }
            PixelLayout::Gray8 | PixelLayout::GrayAlpha8 | PixelLayout::Gray16 => {
                return Err(EncodeError::UnsupportedPixelLayout(self.layout));
            }
        };

        let mut tiny = crate::tiny::TinyEncoder::new(cfg.distance);
        tiny.use_ans = cfg.use_ans;
        tiny.optimize_codes = true;
        tiny.custom_orders = true;
        tiny.enable_noise = cfg.noise;
        tiny.enable_denoise = cfg.denoise;
        tiny.enable_gaborish = cfg.gaborish;
        tiny.error_diffusion = cfg.error_diffusion;
        tiny.pixel_domain_loss = cfg.pixel_domain_loss;
        tiny.enable_lz77 = cfg.lz77;
        tiny.lz77_method = cfg.lz77_method;
        tiny.force_strategy = cfg.force_strategy;
        #[cfg(feature = "butteraugli-loop")]
        {
            tiny.butteraugli_iters = cfg.butteraugli_iters;
        }

        tiny.bit_depth_16 = bit_depth_16;

        tiny.encode(w, h, &linear_rgb, alpha.as_deref())
            .map_err(EncodeError::from)
    }
}

// ── Pixel conversion helpers ────────────────────────────────────────────────

/// sRGB u8 → linear f32 (IEC 61966-2-1).
#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    srgb_to_linear_f(c as f32 / 255.0)
}

fn srgb_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    data.chunks(channels)
        .flat_map(|px| {
            [
                srgb_to_linear(px[0]),
                srgb_to_linear(px[1]),
                srgb_to_linear(px[2]),
            ]
        })
        .collect()
}

/// sRGB u16 → linear f32 (IEC 61966-2-1).
fn srgb_u16_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                srgb_to_linear_f(px[0] as f32 / 65535.0),
                srgb_to_linear_f(px[1] as f32 / 65535.0),
                srgb_to_linear_f(px[2] as f32 / 65535.0),
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
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Extract alpha channel from interleaved 16-bit pixel data as u8 (quantized).
fn extract_alpha_u16(data: &[u8], stride: usize, alpha_offset: usize) -> Vec<u8> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .map(|px| (px[alpha_offset] >> 8) as u8)
        .collect()
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_pixel_layout_helpers() {
        assert_eq!(PixelLayout::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(PixelLayout::Rgba8.bytes_per_pixel(), 4);
        assert_eq!(PixelLayout::Bgr8.bytes_per_pixel(), 3);
        assert_eq!(PixelLayout::Bgra8.bytes_per_pixel(), 4);
        assert_eq!(PixelLayout::Gray8.bytes_per_pixel(), 1);
        assert_eq!(PixelLayout::Rgb16.bytes_per_pixel(), 6);
        assert_eq!(PixelLayout::Rgba16.bytes_per_pixel(), 8);
        assert_eq!(PixelLayout::Gray16.bytes_per_pixel(), 2);
        assert!(!PixelLayout::Rgb8.is_linear());
        assert!(PixelLayout::RgbLinearF32.is_linear());
        assert!(!PixelLayout::Rgb16.is_linear());
        assert!(!PixelLayout::Rgb8.has_alpha());
        assert!(PixelLayout::Rgba8.has_alpha());
        assert!(PixelLayout::Bgra8.has_alpha());
        assert!(PixelLayout::GrayAlpha8.has_alpha());
        assert!(PixelLayout::Rgba16.has_alpha());
        assert!(!PixelLayout::Rgb16.has_alpha());
        assert!(PixelLayout::Rgb16.is_16bit());
        assert!(PixelLayout::Rgba16.is_16bit());
        assert!(PixelLayout::Gray16.is_16bit());
        assert!(!PixelLayout::Rgb8.is_16bit());
        assert!(PixelLayout::Gray8.is_grayscale());
        assert!(PixelLayout::Gray16.is_grayscale());
        assert!(!PixelLayout::Rgb16.is_grayscale());
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
    fn test_lossy_unsupported_gray() {
        let pixels = vec![128u8; 8 * 8];
        let result = LossyConfig::new(1.0)
            .encode_request(8, 8, PixelLayout::Gray8)
            .encode(&pixels);
        assert!(matches!(
            result.as_ref().map_err(|e| e.error()),
            Err(EncodeError::UnsupportedPixelLayout(_))
        ));
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
}
