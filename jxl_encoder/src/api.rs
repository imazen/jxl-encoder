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
            crate::error::Error::InvalidInput(msg) => Self::InvalidInput { message: msg },
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
}

impl Default for AnimationParams {
    fn default() -> Self {
        Self {
            tps_numerator: 100,
            tps_denominator: 1,
            num_loops: 0,
        }
    }
}

/// A single frame in an animation sequence.
pub struct AnimationFrame<'a> {
    /// Raw pixel data (must match width/height/layout from the encode call).
    pub pixels: &'a [u8],
    /// Duration of this frame in ticks (tps_numerator/tps_denominator seconds per tick).
    pub duration: u32,
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
        Self::with_effort_level(7)
    }
}

impl LosslessConfig {
    fn with_effort_level(effort: u8) -> Self {
        let effort = effort.clamp(1, 10);
        Self {
            effort,
            use_ans: effort >= 4,
            tree_learning: effort >= 7,
            squeeze: false, // squeeze hurts without tree-learned predictors
            lz77: effort >= 9,
            lz77_method: Lz77Method::Greedy,
        }
    }

    /// Create a new lossless config with defaults (effort 7).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set effort level (1–10). Higher effort = slower, better compression.
    ///
    /// This adjusts all effort-dependent defaults:
    /// - **e1–3**: Huffman encoding
    /// - **e4–6**: + ANS entropy coding
    /// - **e7**: + content-adaptive tree learning
    /// - **e9–10**: + LZ77 backward references
    ///
    /// Individual `with_*()` calls after `with_effort()` override these defaults.
    pub fn with_effort(self, effort: u8) -> Self {
        let mut new = Self::with_effort_level(effort);
        // Preserve settings that aren't effort-derived
        new.lz77_method = self.lz77_method;
        new.squeeze = self.squeeze;
        new
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
        encode_animation_lossless(self, width, height, layout, animation, frames).map_err(at)
    }
}

// ── LossyConfig ─────────────────────────────────────────────────────────────

#[cfg(feature = "butteraugli-loop")]
fn butteraugli_iters_for_effort(effort: u8) -> u32 {
    // libjxl runs FindBestQuantization (butteraugli loop) at all efforts <= kKitten (e8).
    // Default is 2 iterations, tortoise (e9+) gets 4.
    match effort {
        0..=4 => 0,
        5..=8 => 2,
        _ => 4,
    }
}

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
    patches: bool,
    #[cfg(feature = "butteraugli-loop")]
    butteraugli_iters: u32,
    #[cfg(feature = "butteraugli-loop")]
    butteraugli_iters_explicit: bool,
}

impl LossyConfig {
    /// Create with butteraugli distance (1.0 = high quality). Default effort 7.
    pub fn new(distance: f32) -> Self {
        Self::new_with_effort(distance, 7)
    }

    fn new_with_effort(distance: f32, effort: u8) -> Self {
        let effort = effort.clamp(1, 10);
        Self {
            distance,
            effort,
            use_ans: effort >= 4,
            gaborish: effort >= 3,
            noise: false,
            denoise: false,
            error_diffusion: effort >= 3,
            pixel_domain_loss: effort >= 5,
            lz77: effort >= 9,
            lz77_method: Lz77Method::Greedy,
            force_strategy: None,
            patches: effort >= 5,
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: butteraugli_iters_for_effort(effort),
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters_explicit: false,
        }
    }

    /// Create from a [`Quality`] specification.
    pub fn from_quality(quality: Quality) -> core::result::Result<Self, EncodeError> {
        let distance = quality.to_distance()?;
        Ok(Self::new(distance))
    }

    /// Set effort level (1–10). Higher effort = slower, better compression.
    ///
    /// This adjusts all effort-dependent defaults:
    /// - **e1–2**: DCT8 only, Huffman, no gaborish/patches/butteraugli
    /// - **e3**: + gaborish, error diffusion, Huffman
    /// - **e4**: + ANS entropy coding, multi-block AC strategies
    /// - **e5–7**: + patches, pixel-domain loss, butteraugli loop (2 iters)
    /// - **e8**: same as e7, reserved for future cost model refinements
    /// - **e9–10**: + LZ77 backward references, 4 butteraugli iterations
    ///
    /// Individual `with_*()` calls after `with_effort()` override these defaults.
    pub fn with_effort(self, effort: u8) -> Self {
        let mut new = Self::new_with_effort(self.distance, effort);
        // Preserve settings that are never effort-derived (always opt-in)
        new.noise = self.noise;
        new.denoise = self.denoise;
        new.force_strategy = self.force_strategy;
        new.lz77_method = self.lz77_method;
        // Preserve explicit butteraugli override
        #[cfg(feature = "butteraugli-loop")]
        if self.butteraugli_iters_explicit {
            new.butteraugli_iters = self.butteraugli_iters;
            new.butteraugli_iters_explicit = true;
        }
        new
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

    /// Enable/disable patches (dictionary-based repeated pattern detection).
    /// Default: true. Huge wins on screenshots, zero cost on photos.
    pub fn with_patches(mut self, enable: bool) -> Self {
        self.patches = enable;
        self
    }

    /// Set butteraugli quantization loop iterations explicitly.
    ///
    /// Overrides the automatic effort-based default (effort 7: 0, effort 8: 2, effort 9+: 4).
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn with_butteraugli_iters(mut self, n: u32) -> Self {
        self.butteraugli_iters = n;
        self.butteraugli_iters_explicit = true;
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
        encode_animation_lossy(self, width, height, layout, animation, frames).map_err(at)
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

        let (codestream, mut stats) = match self.config {
            ConfigRef::Lossless(cfg) => self.encode_lossless(cfg, pixels),
            ConfigRef::Lossy(cfg) => self.encode_lossy(cfg, pixels),
        }?;

        stats.codestream_size = codestream.len();

        // Wrap in container if metadata (EXIF/XMP) is present
        let output = if let Some(meta) = self.metadata
            && (meta.exif.is_some() || meta.xmp.is_some())
        {
            crate::container::wrap_in_container(&codestream, meta.exif, meta.xmp)
        } else {
            codestream
        };

        stats.output_size = output.len();

        Ok(EncodeResult {
            data: Some(output),
            stats,
        })
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
    ) -> core::result::Result<(Vec<u8>, EncodeStats), EncodeError> {
        use crate::bit_writer::BitWriter;
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::channel::ModularImage;
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        let w = self.width as usize;
        let h = self.height as usize;

        // Build ModularImage from pixel layout
        let image = match self.layout {
            PixelLayout::Rgb8 => ModularImage::from_rgb8(pixels, w, h),
            PixelLayout::Rgba8 => ModularImage::from_rgba8(pixels, w, h),
            PixelLayout::Bgr8 => ModularImage::from_rgb8(&bgr_to_rgb(pixels, 3), w, h),
            PixelLayout::Bgra8 => ModularImage::from_rgba8(&bgr_to_rgb(pixels, 4), w, h),
            PixelLayout::Gray8 => ModularImage::from_gray8(pixels, w, h),
            PixelLayout::Rgb16 => ModularImage::from_rgb16_native(pixels, w, h),
            PixelLayout::Rgba16 => ModularImage::from_rgba16_native(pixels, w, h),
            PixelLayout::Gray16 => ModularImage::from_gray16_native(pixels, w, h),
            other => return Err(EncodeError::UnsupportedPixelLayout(other)),
        }
        .map_err(EncodeError::from)?;

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
        if let Some(meta) = self.metadata
            && meta.icc_profile.is_some()
        {
            file_header.metadata.color_encoding.want_icc = true;
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

        // Encode frame
        // Tree learning is only validated for 8-bit images; disable for 16-bit.
        let use_tree_learning = cfg.tree_learning && image.bit_depth <= 8;
        let frame_encoder = FrameEncoder::new(
            w,
            h,
            FrameEncoderOptions {
                use_modular: true,
                effort: cfg.effort,
                use_ans: cfg.use_ans,
                use_tree_learning,
                use_squeeze: cfg.squeeze,
                have_animation: false,
                duration: 0,
                is_last: true,
                crop: None,
            },
        );
        let color_encoding = ColorEncoding::srgb();
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer)
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
    ) -> core::result::Result<(Vec<u8>, EncodeStats), EncodeError> {
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

        let mut tiny = crate::vardct::VarDctEncoder::new(cfg.distance);
        tiny.effort = cfg.effort;
        tiny.use_ans = cfg.use_ans;
        tiny.optimize_codes = cfg.effort >= 2;
        tiny.custom_orders = cfg.effort >= 3;
        tiny.ac_strategy_enabled = cfg.effort >= 3;
        tiny.enable_noise = cfg.noise;
        tiny.enable_denoise = cfg.denoise;
        tiny.enable_gaborish = cfg.gaborish;
        tiny.error_diffusion = cfg.error_diffusion;
        tiny.pixel_domain_loss = cfg.pixel_domain_loss;
        tiny.enable_lz77 = cfg.lz77;
        tiny.lz77_method = cfg.lz77_method;
        tiny.force_strategy = cfg.force_strategy;
        tiny.enable_patches = cfg.patches;
        #[cfg(feature = "butteraugli-loop")]
        {
            tiny.butteraugli_iters = cfg.butteraugli_iters;
        }

        tiny.bit_depth_16 = bit_depth_16;

        // ICC profile from metadata
        if let Some(meta) = self.metadata
            && let Some(icc) = meta.icc_profile
        {
            tiny.icc_profile = Some(icc.to_vec());
        }

        let output = tiny
            .encode(w, h, &linear_rgb, alpha.as_deref())
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
}

// ── Animation encode implementations ────────────────────────────────────────

fn validate_animation_input(
    width: u32,
    height: u32,
    layout: PixelLayout,
    frames: &[AnimationFrame<'_>],
) -> core::result::Result<(), EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::InvalidInput {
            message: format!("zero dimensions: {width}x{height}"),
        });
    }
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
) -> core::result::Result<Vec<u8>, EncodeError> {
    use crate::bit_writer::BitWriter;
    use crate::headers::file_header::AnimationHeader;
    use crate::headers::{ColorEncoding, FileHeader};
    use crate::modular::channel::ModularImage;
    use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

    validate_animation_input(width, height, layout, frames)?;

    let w = width as usize;
    let h = height as usize;
    let num_frames = frames.len();

    // Build file header with animation
    let sample_image = match layout {
        PixelLayout::Rgb8 => ModularImage::from_rgb8(frames[0].pixels, w, h),
        PixelLayout::Rgba8 => ModularImage::from_rgba8(frames[0].pixels, w, h),
        PixelLayout::Bgr8 => ModularImage::from_rgb8(&bgr_to_rgb(frames[0].pixels, 3), w, h),
        PixelLayout::Bgra8 => ModularImage::from_rgba8(&bgr_to_rgb(frames[0].pixels, 4), w, h),
        PixelLayout::Gray8 => ModularImage::from_gray8(frames[0].pixels, w, h),
        PixelLayout::Rgb16 => ModularImage::from_rgb16_native(frames[0].pixels, w, h),
        PixelLayout::Rgba16 => ModularImage::from_rgba16_native(frames[0].pixels, w, h),
        PixelLayout::Gray16 => ModularImage::from_gray16_native(frames[0].pixels, w, h),
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
    file_header.metadata.animation = Some(AnimationHeader {
        tps_numerator: animation.tps_numerator,
        tps_denominator: animation.tps_denominator,
        num_loops: animation.num_loops,
        have_timecodes: false,
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
        // Detect crop: compare current frame against previous.
        // Only use crop when it's smaller than the full frame.
        let crop = if let Some(prev) = prev_pixels {
            match detect_frame_crop(prev, frame.pixels, w, h, bpp, false) {
                Some(crop) if (crop.width as usize) < w || (crop.height as usize) < h => Some(crop),
                Some(_) => None, // Crop covers full frame — no benefit
                None => {
                    // Frames are identical — emit a minimal 1x1 crop to preserve canvas
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

        // Build ModularImage from the appropriate pixel region
        let (frame_w, frame_h, frame_pixels_owned);
        let frame_pixels: &[u8] = if let Some(ref crop) = crop {
            frame_w = crop.width as usize;
            frame_h = crop.height as usize;
            frame_pixels_owned = extract_pixel_crop(frame.pixels, w, crop, bpp);
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
            PixelLayout::Rgb16 => ModularImage::from_rgb16_native(frame_pixels, frame_w, frame_h),
            PixelLayout::Rgba16 => ModularImage::from_rgba16_native(frame_pixels, frame_w, frame_h),
            PixelLayout::Gray16 => ModularImage::from_gray16_native(frame_pixels, frame_w, frame_h),
            other => return Err(EncodeError::UnsupportedPixelLayout(other)),
        }
        .map_err(EncodeError::from)?;

        // Tree learning is only validated for 8-bit images; disable for 16-bit.
        let use_tree_learning = cfg.tree_learning && image.bit_depth <= 8;
        let frame_encoder = FrameEncoder::new(
            frame_w,
            frame_h,
            FrameEncoderOptions {
                use_modular: true,
                effort: cfg.effort,
                use_ans: cfg.use_ans,
                use_tree_learning,
                use_squeeze: cfg.squeeze,
                have_animation: true,
                duration: frame.duration,
                is_last: i == num_frames - 1,
                crop,
            },
        );
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .map_err(EncodeError::from)?;

        prev_pixels = Some(frame.pixels);
    }

    Ok(writer.finish_with_padding())
}

fn encode_animation_lossy(
    cfg: &LossyConfig,
    width: u32,
    height: u32,
    layout: PixelLayout,
    animation: &AnimationParams,
    frames: &[AnimationFrame<'_>],
) -> core::result::Result<Vec<u8>, EncodeError> {
    use crate::bit_writer::BitWriter;
    use crate::headers::file_header::AnimationHeader;
    use crate::headers::frame_header::FrameOptions;

    validate_animation_input(width, height, layout, frames)?;

    let w = width as usize;
    let h = height as usize;
    let num_frames = frames.len();

    // Set up VarDCT encoder
    let mut tiny = crate::vardct::VarDctEncoder::new(cfg.distance);
    tiny.effort = cfg.effort;
    tiny.use_ans = cfg.use_ans;
    tiny.optimize_codes = cfg.effort >= 2;
    tiny.custom_orders = cfg.effort >= 3;
    tiny.ac_strategy_enabled = cfg.effort >= 3;
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

    // Detect alpha and 16-bit from layout
    let has_alpha = layout.has_alpha();
    let bit_depth_16 = matches!(layout, PixelLayout::Rgb16 | PixelLayout::Rgba16);
    tiny.bit_depth_16 = bit_depth_16;

    // Build file header from VarDCT encoder (sets xyb_encoded, rendering_intent, etc.)
    // then add animation metadata
    let mut file_header = tiny.build_file_header(w, h, has_alpha);
    file_header.metadata.animation = Some(AnimationHeader {
        tps_numerator: animation.tps_numerator,
        tps_denominator: animation.tps_denominator,
        num_loops: animation.num_loops,
        have_timecodes: false,
    });

    let mut writer = BitWriter::with_capacity(w * h * 4);
    file_header.write(&mut writer).map_err(EncodeError::from)?;
    if let Some(ref icc) = tiny.icc_profile {
        crate::icc::write_icc(icc, &mut writer).map_err(EncodeError::from)?;
    }
    writer.zero_pad_to_byte();

    // Encode each frame with crop detection
    let bpp = layout.bytes_per_pixel();
    let mut prev_pixels: Option<&[u8]> = None;

    for (i, frame) in frames.iter().enumerate() {
        // Detect crop on raw input pixels (before linear conversion).
        // Only use crop when it's smaller than the full frame.
        let crop = if let Some(prev) = prev_pixels {
            match detect_frame_crop(prev, frame.pixels, w, h, bpp, true) {
                Some(crop) if (crop.width as usize) < w || (crop.height as usize) < h => Some(crop),
                Some(_) => None, // Crop covers full frame — no benefit
                None => {
                    // Frames identical — emit minimal 8x8 crop (VarDCT minimum)
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

        // Extract crop region from raw pixels, then convert to linear
        let (frame_w, frame_h) = if let Some(ref crop) = crop {
            (crop.width as usize, crop.height as usize)
        } else {
            (w, h)
        };

        let crop_pixels_owned;
        let src_pixels: &[u8] = if let Some(ref crop) = crop {
            crop_pixels_owned = extract_pixel_crop(frame.pixels, w, crop, bpp);
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
            PixelLayout::Rgb16 => (srgb_u16_to_linear_f32(src_pixels, 3), None),
            PixelLayout::Rgba16 => {
                let rgb = srgb_u16_to_linear_f32(src_pixels, 4);
                let alpha = extract_alpha_u16(src_pixels, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbLinearF32 => {
                let floats: &[f32] = bytemuck::cast_slice(src_pixels);
                (floats.to_vec(), None)
            }
            PixelLayout::Gray8 | PixelLayout::GrayAlpha8 | PixelLayout::Gray16 => {
                return Err(EncodeError::UnsupportedPixelLayout(layout));
            }
        };

        let frame_options = FrameOptions {
            have_animation: true,
            have_timecodes: false,
            duration: frame.duration,
            is_last: i == num_frames - 1,
            crop,
        };

        tiny.encode_frame_to_writer(
            frame_w,
            frame_h,
            &linear_rgb,
            alpha.as_deref(),
            &frame_options,
            &mut writer,
        )
        .map_err(EncodeError::from)?;

        prev_pixels = Some(frame.pixels);
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
