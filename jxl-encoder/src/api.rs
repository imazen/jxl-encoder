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
            crate::error::Error::DimensionOverflow {
                width,
                height,
                channels,
            } => Self::InvalidInput {
                message: format!("dimension overflow: {width}x{height}x{channels} exceeds usize"),
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
        )
    }

    /// Whether this layout uses 16-bit samples.
    pub const fn is_16bit(self) -> bool {
        matches!(
            self,
            Self::Rgb16 | Self::Rgba16 | Self::Gray16 | Self::GrayAlpha16
        )
    }

    /// Whether this layout uses f32 samples.
    pub const fn is_f32(self) -> bool {
        matches!(
            self,
            Self::RgbLinearF32
                | Self::RgbaLinearF32
                | Self::GrayLinearF32
                | Self::GrayAlphaLinearF32
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
        )
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

// ── Supporting types ────────────────────────────────────────────────────────

/// Image metadata (ICC, EXIF, XMP, tone mapping) to embed in the JXL file.
#[derive(Clone, Debug, Default)]
pub struct ImageMetadata<'a> {
    icc_profile: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
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
    mode: EncoderMode,
    use_ans: bool,
    squeeze: bool,
    tree_learning: bool,
    lz77: bool,
    lz77_method: Lz77Method,
    patches: bool,
    lossy_palette: bool,
    threads: usize,
    /// Sweep / picker hook: when set, replaces the effort+mode-derived
    /// `EffortProfile` everywhere the encoder asks for one. See
    /// [`Self::with_effort_profile_override`].
    profile_override: Option<crate::effort::EffortProfile>,
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
            profile_override: None,
        }
    }

    /// Resolve the effective [`EffortProfile`]: the override if set,
    /// otherwise the standard profile derived from effort + mode.
    pub(crate) fn effective_profile(&self) -> crate::effort::EffortProfile {
        self.profile_override
            .clone()
            .unwrap_or_else(|| crate::effort::EffortProfile::lossless(self.effort, self.mode))
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

    /// Set effort level (1–10). Higher effort = slower, better compression.
    ///
    /// This adjusts all effort-dependent defaults:
    /// - **e1–3**: Huffman encoding
    /// - **e4–6**: + ANS entropy coding
    /// - **e7**: + content-adaptive tree learning, LZ77 RLE
    /// - **e8**: + LZ77 greedy hash chain
    /// - **e9–10**: + LZ77 optimal (Viterbi DP)
    ///
    /// Individual `with_*()` calls after `with_effort()` override these defaults.
    pub fn with_effort(self, effort: u8) -> Self {
        let mut new = Self::with_effort_level(effort);
        // Preserve settings that aren't effort-derived
        new.mode = self.mode;
        new.squeeze = self.squeeze;
        new.profile_override = self.profile_override;
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

// ── LossyConfig ─────────────────────────────────────────────────────────────

/// Lossy (VarDCT) encoding configuration.
///
/// No `Default` — distance/quality is a required choice.
#[derive(Clone, Debug)]
pub struct LossyConfig {
    distance: f32,
    effort: u8,
    mode: EncoderMode,
    use_ans: bool,
    gaborish: bool,
    noise: bool,
    denoise: bool,
    error_diffusion: bool,
    pixel_domain_loss: bool,
    lz77: bool,
    lz77_method: Lz77Method,
    force_strategy: Option<u8>,
    max_strategy_size: Option<u8>,
    patches: bool,
    splines: Option<Vec<crate::vardct::splines::Spline>>,
    progressive: ProgressiveMode,
    lf_frame: bool,
    #[cfg(feature = "butteraugli-loop")]
    butteraugli_iters: u32,
    #[cfg(feature = "butteraugli-loop")]
    butteraugli_iters_explicit: bool,
    #[cfg(feature = "ssim2-loop")]
    ssim2_iters: u32,
    #[cfg(feature = "zensim-loop")]
    zensim_iters: u32,
    threads: usize,
    /// Sweep / picker hook: when set, replaces the effort+mode-derived
    /// `EffortProfile` everywhere the encoder asks for one. See
    /// [`Self::with_effort_profile_override`].
    profile_override: Option<crate::effort::EffortProfile>,
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
            noise: false,
            denoise: false,
            error_diffusion: profile.error_diffusion,
            pixel_domain_loss: profile.pixel_domain_loss,
            lz77: profile.lz77,
            lz77_method: profile.lz77_method,
            force_strategy: None,
            max_strategy_size: None,
            patches: profile.patches,
            splines: None,
            progressive: ProgressiveMode::Single,
            lf_frame: false,
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: profile.butteraugli_iters,
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters_explicit: false,
            #[cfg(feature = "ssim2-loop")]
            ssim2_iters: 0,
            #[cfg(feature = "zensim-loop")]
            zensim_iters: 0,
            threads: 0,
            profile_override: None,
        }
    }

    /// Resolve the effective [`EffortProfile`]: the override if set,
    /// otherwise the standard profile derived from effort + mode.
    pub(crate) fn effective_profile(&self) -> crate::effort::EffortProfile {
        self.profile_override
            .clone()
            .unwrap_or_else(|| crate::effort::EffortProfile::lossy(self.effort, self.mode))
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

    /// Set effort level (1–10). Higher effort = slower, better compression.
    ///
    /// This adjusts all effort-dependent defaults:
    /// - **e1–3**: DCT8 only, Huffman, no gaborish/patches/butteraugli
    /// - **e4**: + ANS entropy coding, custom coefficient orders
    /// - **e5**: + gaborish, pixel-domain loss, AC strategy search, AdjustQuantBlockAC
    /// - **e6**: + DCT4x8/AFV strategies, non-aligned eval, EPF dynamic sharpness
    /// - **e7**: + patches, error diffusion, CfL two-pass, LZ77 RLE, DCT64 strategies
    /// - **e8**: + butteraugli loop (2 iters), LZ77 greedy, WP param search (2 modes)
    /// - **e9–10**: + LZ77 optimal (Viterbi DP), 4 butteraugli iters, WP search (5 modes)
    ///
    /// Individual `with_*()` calls after `with_effort()` override these defaults.
    pub fn with_effort(self, effort: u8) -> Self {
        let mut new = Self::new_with_effort(self.distance, effort);
        // Preserve settings that are never effort-derived (always opt-in)
        new.mode = self.mode;
        new.noise = self.noise;
        new.denoise = self.denoise;
        new.force_strategy = self.force_strategy;
        new.max_strategy_size = self.max_strategy_size;
        new.splines = self.splines;
        new.progressive = self.progressive;
        // Preserve explicit butteraugli override
        #[cfg(feature = "butteraugli-loop")]
        if self.butteraugli_iters_explicit {
            new.butteraugli_iters = self.butteraugli_iters;
            new.butteraugli_iters_explicit = true;
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
    /// Default: true. Huge wins on screenshots, zero cost on photos.
    pub fn with_patches(mut self, enable: bool) -> Self {
        self.patches = enable;
        self
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

    /// Set SSIM2 quantization loop iterations.
    ///
    /// Alternative to butteraugli loop: uses per-block linear RGB RMSE + full-image SSIM2.
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
    source_gamma: Option<f32>,
    color_encoding: Option<crate::headers::color_encoding::ColorEncoding>,
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

        let threads = match self.config {
            ConfigRef::Lossless(cfg) => cfg.threads,
            ConfigRef::Lossy(cfg) => cfg.threads,
        };

        let (codestream, mut stats) = run_with_threads(threads, || match self.config {
            ConfigRef::Lossless(cfg) => self.encode_lossless(cfg, pixels),
            ConfigRef::Lossy(cfg) => self.encode_lossy(cfg, pixels),
        })?;

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
        validate_dims(self.width, self.height)?;
        let w = self.width as usize;
        let h = self.height as usize;
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
        Ok(())
    }

    // ── Lossless path ───────────────────────────────────────────────────

    fn encode_lossless(
        &self,
        cfg: &LosslessConfig,
        pixels: &[u8],
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

        // Build ModularImage from pixel layout
        let mut image = match self.layout {
            PixelLayout::Rgb8 => ModularImage::from_rgb8(pixels, w, h),
            PixelLayout::Rgba8 => ModularImage::from_rgba8(pixels, w, h),
            PixelLayout::Bgr8 => ModularImage::from_rgb8(&bgr_to_rgb(pixels, 3), w, h),
            PixelLayout::Bgra8 => ModularImage::from_rgba8(&bgr_to_rgb(pixels, 4), w, h),
            PixelLayout::Gray8 => ModularImage::from_gray8(pixels, w, h),
            PixelLayout::GrayAlpha8 => ModularImage::from_grayalpha8(pixels, w, h),
            PixelLayout::Rgb16 => ModularImage::from_rgb16_native(pixels, w, h),
            PixelLayout::Rgba16 => ModularImage::from_rgba16_native(pixels, w, h),
            PixelLayout::Gray16 => ModularImage::from_gray16_native(pixels, w, h),
            PixelLayout::GrayAlpha16 => ModularImage::from_grayalpha16_native(pixels, w, h),
            other => return Err(EncodeError::UnsupportedPixelLayout(other)),
        }
        .map_err(EncodeError::from)?;

        // Detect patches for lossless mode (RGB 8-bit only, non-grayscale)
        let num_channels = self.layout.bytes_per_pixel();
        let can_use_patches =
            cfg.patches && !image.is_grayscale && image.bit_depth <= 8 && num_channels >= 3;
        let patches_data = if can_use_patches {
            crate::vardct::patches::find_and_build_lossless(
                detection_pixels,
                w,
                h,
                num_channels,
                image.bit_depth,
            )
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
            )
            .map_err(EncodeError::from)?;
            writer.zero_pad_to_byte();
            let bd = image.bit_depth;
            crate::vardct::patches::subtract_patches_modular(&mut image, pd, bd);
        }

        // Encode frame
        let use_tree_learning = cfg.tree_learning;
        let frame_encoder = FrameEncoder::new(
            w,
            h,
            FrameEncoderOptions {
                use_modular: true,
                effort: cfg.effort,
                use_ans: cfg.use_ans,
                use_tree_learning,
                use_squeeze: cfg.squeeze,
                enable_lz77: cfg.lz77,
                lz77_method: cfg.lz77_method,
                lossy_palette: cfg.lossy_palette,
                encoder_mode: cfg.mode,
                profile: cfg.effective_profile(),
                have_animation: false,
                duration: 0,
                is_last: true,
                crop: None,
                skip_rct: false,
            },
        );
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
    ) -> core::result::Result<(Vec<u8>, EncodeStats), EncodeError> {
        let w = self.width as usize;
        let h = self.height as usize;

        // Build linear f32 RGB and extract alpha from input layout.
        // Grayscale layouts are expanded to RGB (R=G=B) for VarDCT encoding.
        // When source_gamma is set, use gamma linearization instead of sRGB TF.
        let gamma = self.source_gamma;
        let (linear_rgb, alpha, bit_depth_16) = match self.layout {
            PixelLayout::Rgb8 => {
                let linear = if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(pixels, 3, g)
                } else {
                    srgb_u8_to_linear_f32(pixels, 3)
                };
                (linear, None, false)
            }
            PixelLayout::Bgr8 => {
                let rgb = bgr_to_rgb(pixels, 3);
                let linear = if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(&rgb, 3, g)
                } else {
                    srgb_u8_to_linear_f32(&rgb, 3)
                };
                (linear, None, false)
            }
            PixelLayout::Rgba8 => {
                let rgb = if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(pixels, 4, g)
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
                } else {
                    srgb_u8_to_linear_f32(&swapped, 4)
                };
                let alpha = extract_alpha(pixels, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::Gray8 => {
                let rgb = if let Some(g) = gamma {
                    gamma_gray_u8_to_linear_f32_rgb(pixels, 1, g)
                } else {
                    gray_u8_to_linear_f32_rgb(pixels, 1)
                };
                (rgb, None, false)
            }
            PixelLayout::GrayAlpha8 => {
                let rgb = if let Some(g) = gamma {
                    gamma_gray_u8_to_linear_f32_rgb(pixels, 2, g)
                } else {
                    gray_u8_to_linear_f32_rgb(pixels, 2)
                };
                let alpha = extract_alpha(pixels, 2, 1);
                (rgb, Some(alpha), false)
            }
            PixelLayout::Rgb16 => {
                let linear = if let Some(g) = gamma {
                    gamma_u16_to_linear_f32(pixels, 3, g)
                } else {
                    srgb_u16_to_linear_f32(pixels, 3)
                };
                (linear, None, true)
            }
            PixelLayout::Rgba16 => {
                let rgb = if let Some(g) = gamma {
                    gamma_u16_to_linear_f32(pixels, 4, g)
                } else {
                    srgb_u16_to_linear_f32(pixels, 4)
                };
                let alpha = extract_alpha_u16(pixels, 4, 3);
                (rgb, Some(alpha), true)
            }
            PixelLayout::Gray16 => {
                let rgb = if let Some(g) = gamma {
                    gamma_gray_u16_to_linear_f32_rgb(pixels, 1, g)
                } else {
                    gray_u16_to_linear_f32_rgb(pixels, 1)
                };
                (rgb, None, true)
            }
            PixelLayout::GrayAlpha16 => {
                let rgb = if let Some(g) = gamma {
                    gamma_gray_u16_to_linear_f32_rgb(pixels, 2, g)
                } else {
                    gray_u16_to_linear_f32_rgb(pixels, 2)
                };
                let alpha = extract_alpha_u16(pixels, 2, 1);
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
        };

        let mut profile = cfg.effective_profile();

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
        enc.effort = cfg.effort;
        enc.profile = profile;
        enc.use_ans = cfg.use_ans;
        enc.optimize_codes = enc.profile.optimize_codes;
        enc.custom_orders = enc.profile.custom_orders;
        enc.ac_strategy_enabled = enc.profile.ac_strategy_enabled;
        enc.enable_noise = cfg.noise;
        enc.enable_denoise = cfg.denoise;
        // libjxl gates gaborish at distance > 0.5 (enc_frame.cc:281)
        enc.enable_gaborish = cfg.gaborish && cfg.distance > 0.5;
        enc.error_diffusion = cfg.error_diffusion;
        enc.pixel_domain_loss = cfg.pixel_domain_loss;
        enc.enable_lz77 = cfg.lz77;
        enc.lz77_method = cfg.lz77_method;
        enc.force_strategy = cfg.force_strategy;
        enc.enable_patches = cfg.patches;
        enc.encoder_mode = cfg.mode;
        enc.splines = cfg.splines.clone();
        enc.is_grayscale = self.layout.is_grayscale();
        enc.progressive = cfg.progressive;
        enc.use_lf_frame = cfg.lf_frame;
        #[cfg(feature = "butteraugli-loop")]
        {
            enc.butteraugli_iters = cfg.butteraugli_iters;
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
        enc.color_encoding = self.color_encoding.clone();

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

        // ICC profile from metadata
        if let Some(meta) = self.metadata
            && let Some(icc) = meta.icc_profile
        {
            enc.icc_profile = Some(icc.to_vec());
        }

        let output = enc
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
    source_gamma: Option<f32>,
    color_encoding: Option<crate::headers::color_encoding::ColorEncoding>,
    intensity_target: f32,
    min_nits: f32,
    intrinsic_size: Option<(u32, u32)>,
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

        // Convert and append linear RGB
        let new_linear: Vec<f32> = match self.layout {
            PixelLayout::Rgb8 => {
                if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(pixels, 3, g)
                } else {
                    srgb_u8_to_linear_f32(pixels, 3)
                }
            }
            PixelLayout::Bgr8 => {
                let rgb = bgr_to_rgb(pixels, 3);
                if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(&rgb, 3, g)
                } else {
                    srgb_u8_to_linear_f32(&rgb, 3)
                }
            }
            PixelLayout::Rgba8 => {
                if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(pixels, 4, g)
                } else {
                    srgb_u8_to_linear_f32(pixels, 4)
                }
            }
            PixelLayout::Bgra8 => {
                let swapped = bgr_to_rgb(pixels, 4);
                if let Some(g) = gamma {
                    gamma_u8_to_linear_f32(&swapped, 4, g)
                } else {
                    srgb_u8_to_linear_f32(&swapped, 4)
                }
            }
            PixelLayout::Gray8 => {
                if let Some(g) = gamma {
                    gamma_gray_u8_to_linear_f32_rgb(pixels, 1, g)
                } else {
                    gray_u8_to_linear_f32_rgb(pixels, 1)
                }
            }
            PixelLayout::GrayAlpha8 => {
                if let Some(g) = gamma {
                    gamma_gray_u8_to_linear_f32_rgb(pixels, 2, g)
                } else {
                    gray_u8_to_linear_f32_rgb(pixels, 2)
                }
            }
            PixelLayout::Rgb16 => {
                if let Some(g) = gamma {
                    gamma_u16_to_linear_f32(pixels, 3, g)
                } else {
                    srgb_u16_to_linear_f32(pixels, 3)
                }
            }
            PixelLayout::Rgba16 => {
                if let Some(g) = gamma {
                    gamma_u16_to_linear_f32(pixels, 4, g)
                } else {
                    srgb_u16_to_linear_f32(pixels, 4)
                }
            }
            PixelLayout::Gray16 => {
                if let Some(g) = gamma {
                    gamma_gray_u16_to_linear_f32_rgb(pixels, 1, g)
                } else {
                    gray_u16_to_linear_f32_rgb(pixels, 1)
                }
            }
            PixelLayout::GrayAlpha16 => {
                if let Some(g) = gamma {
                    gamma_gray_u16_to_linear_f32_rgb(pixels, 2, g)
                } else {
                    gray_u16_to_linear_f32_rgb(pixels, 2)
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
                let new_alpha = extract_alpha_u16(pixels, 4, 3);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::GrayAlpha16 => {
                let new_alpha = extract_alpha_u16(pixels, 2, 1);
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

    fn finish_inner(self) -> core::result::Result<EncodeResult, EncodeError> {
        if self.rows_pushed != self.height {
            return Err(EncodeError::InvalidInput {
                message: format!(
                    "incomplete image: {} of {} rows pushed",
                    self.rows_pushed, self.height
                ),
            });
        }

        let cfg = &self.cfg;
        let w = self.width as usize;
        let h = self.height as usize;
        let linear_rgb = self.linear_rgb;
        let alpha = self.alpha;

        let (codestream, mut stats) = run_with_threads(cfg.threads, || {
            let mut profile = cfg.effective_profile();
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
            enc.effort = cfg.effort;
            enc.profile = profile;
            enc.use_ans = cfg.use_ans;
            enc.optimize_codes = enc.profile.optimize_codes;
            enc.custom_orders = enc.profile.custom_orders;
            enc.ac_strategy_enabled = enc.profile.ac_strategy_enabled;
            enc.enable_noise = cfg.noise;
            enc.enable_denoise = cfg.denoise;
            enc.enable_gaborish = cfg.gaborish && cfg.distance > 0.5;
            enc.error_diffusion = cfg.error_diffusion;
            enc.pixel_domain_loss = cfg.pixel_domain_loss;
            enc.enable_lz77 = cfg.lz77;
            enc.lz77_method = cfg.lz77_method;
            enc.force_strategy = cfg.force_strategy;
            enc.enable_patches = cfg.patches;
            enc.encoder_mode = cfg.mode;
            enc.splines = cfg.splines.clone();
            enc.is_grayscale = self.layout.is_grayscale();
            enc.progressive = cfg.progressive;
            enc.use_lf_frame = cfg.lf_frame;
            #[cfg(feature = "butteraugli-loop")]
            {
                enc.butteraugli_iters = cfg.butteraugli_iters;
            }
            enc.bit_depth_16 = self.bit_depth_16;
            enc.source_gamma = self.source_gamma;
            enc.color_encoding = self.color_encoding.clone();
            enc.intensity_target = self.intensity_target;
            enc.min_nits = self.min_nits;
            enc.intrinsic_size = self.intrinsic_size;
            if let Some(ref icc) = self.icc_profile {
                enc.icc_profile = Some(icc.clone());
            }

            let output = enc
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
            Ok::<_, EncodeError>((output.data, stats))
        })?;

        stats.codestream_size = codestream.len();

        let output = if self.exif.is_some() || self.xmp.is_some() {
            crate::container::wrap_in_container(
                &codestream,
                self.exif.as_deref(),
                self.xmp.as_deref(),
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
    if w
        .checked_mul(h)
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
            source_gamma: None,
            color_encoding: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            intrinsic_size: None,
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
    source_gamma: Option<f32>,
    color_encoding: Option<crate::headers::color_encoding::ColorEncoding>,
    intensity_target: f32,
    min_nits: f32,
    intrinsic_size: Option<(u32, u32)>,
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

        let cfg = &self.cfg;
        let w = self.width as usize;
        let h = self.height as usize;

        let mut image = ModularImage {
            channels: self.channels,
            bit_depth: self.bit_depth,
            is_grayscale: self.is_grayscale,
            has_alpha: self.has_alpha,
        };

        let (codestream, mut stats) = run_with_threads(cfg.threads, || {
            // Reconstruct interleaved pixels for patch detection (8-bit RGB only)
            let num_channels = self.layout.bytes_per_pixel();
            let can_use_patches =
                cfg.patches && !image.is_grayscale && image.bit_depth <= 8 && num_channels >= 3;
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
                crate::vardct::patches::find_and_build_lossless(
                    &detection_pixels,
                    w,
                    h,
                    num_channels,
                    image.bit_depth,
                )
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
                )
                .map_err(EncodeError::from)?;
                writer.zero_pad_to_byte();
                let bd = image.bit_depth;
                crate::vardct::patches::subtract_patches_modular(&mut image, pd, bd);
            }

            // Encode frame
            let frame_encoder = FrameEncoder::new(
                w,
                h,
                FrameEncoderOptions {
                    use_modular: true,
                    effort: cfg.effort,
                    use_ans: cfg.use_ans,
                    use_tree_learning: cfg.tree_learning,
                    use_squeeze: cfg.squeeze,
                    enable_lz77: cfg.lz77,
                    lz77_method: cfg.lz77_method,
                    lossy_palette: cfg.lossy_palette,
                    encoder_mode: cfg.mode,
                    profile: cfg.effective_profile(),
                    have_animation: false,
                    duration: 0,
                    is_last: true,
                    crop: None,
                    skip_rct: false,
                },
            );
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

        let output = if self.exif.is_some() || self.xmp.is_some() {
            crate::container::wrap_in_container(
                &codestream,
                self.exif.as_deref(),
                self.xmp.as_deref(),
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
            source_gamma: None,
            color_encoding: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            intrinsic_size: None,
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

        let use_tree_learning = cfg.tree_learning;
        let frame_encoder = FrameEncoder::new(
            frame_w,
            frame_h,
            FrameEncoderOptions {
                use_modular: true,
                effort: cfg.effort,
                use_ans: cfg.use_ans,
                use_tree_learning,
                use_squeeze: cfg.squeeze,
                enable_lz77: cfg.lz77,
                lz77_method: cfg.lz77_method,
                lossy_palette: cfg.lossy_palette,
                encoder_mode: cfg.mode,
                profile: cfg.effective_profile(),
                have_animation: true,
                duration: frame.duration,
                is_last: i == num_frames - 1,
                crop,
                skip_rct: false,
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
    let mut profile = cfg.effective_profile();

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
    enc.effort = cfg.effort;
    enc.profile = profile;
    enc.use_ans = cfg.use_ans;
    enc.optimize_codes = enc.profile.optimize_codes;
    enc.custom_orders = enc.profile.custom_orders;
    enc.ac_strategy_enabled = enc.profile.ac_strategy_enabled;
    enc.enable_noise = cfg.noise;
    enc.enable_denoise = cfg.denoise;
    // libjxl gates gaborish at distance > 0.5 (enc_frame.cc:281)
    enc.enable_gaborish = cfg.gaborish && cfg.distance > 0.5;
    enc.error_diffusion = cfg.error_diffusion;
    enc.pixel_domain_loss = cfg.pixel_domain_loss;
    enc.enable_lz77 = cfg.lz77;
    enc.lz77_method = cfg.lz77_method;
    enc.force_strategy = cfg.force_strategy;
    enc.progressive = cfg.progressive;
    enc.use_lf_frame = cfg.lf_frame;
    #[cfg(feature = "butteraugli-loop")]
    {
        enc.butteraugli_iters = cfg.butteraugli_iters;
    }
    #[cfg(feature = "ssim2-loop")]
    {
        enc.ssim2_iters = cfg.ssim2_iters;
    }
    #[cfg(feature = "zensim-loop")]
    {
        enc.zensim_iters = cfg.zensim_iters;
    }

    // Detect alpha and 16-bit from layout
    let has_alpha = layout.has_alpha();
    let bit_depth_16 = matches!(layout, PixelLayout::Rgb16 | PixelLayout::Rgba16);
    enc.bit_depth_16 = bit_depth_16;

    // Build file header from VarDCT encoder (sets xyb_encoded, rendering_intent, etc.)
    // then add animation metadata
    let mut file_header = enc.build_file_header(w, h, has_alpha);
    file_header.metadata.animation = Some(AnimationHeader {
        tps_numerator: animation.tps_numerator,
        tps_denominator: animation.tps_denominator,
        num_loops: animation.num_loops,
        have_timecodes: false,
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
            PixelLayout::Gray8 => (gray_u8_to_linear_f32_rgb(src_pixels, 1), None),
            PixelLayout::GrayAlpha8 => {
                let rgb = gray_u8_to_linear_f32_rgb(src_pixels, 2);
                let alpha = extract_alpha(src_pixels, 2, 1);
                (rgb, Some(alpha))
            }
            PixelLayout::Rgb16 => (srgb_u16_to_linear_f32(src_pixels, 3), None),
            PixelLayout::Rgba16 => {
                let rgb = srgb_u16_to_linear_f32(src_pixels, 4);
                let alpha = extract_alpha_u16(src_pixels, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::Gray16 => (gray_u16_to_linear_f32_rgb(src_pixels, 1), None),
            PixelLayout::GrayAlpha16 => {
                let rgb = gray_u16_to_linear_f32_rgb(src_pixels, 2);
                let alpha = extract_alpha_u16(src_pixels, 2, 1);
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
        };

        let frame_options = FrameOptions {
            have_animation: true,
            have_timecodes: false,
            duration: frame.duration,
            is_last: i == num_frames - 1,
            crop,
        };

        enc.encode_frame_to_writer(
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
        jxl_simd::fast_powf((c + 0.055) / 1.055, 2.4)
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

/// Gamma u16 → linear f32 RGB. `linear = (encoded/65535)^(1/gamma)`
fn gamma_u16_to_linear_f32(data: &[u8], channels: usize, gamma: f32) -> Vec<f32> {
    let inv_gamma = 1.0 / gamma;
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                jxl_simd::fast_powf(px[0] as f32 / 65535.0, inv_gamma),
                jxl_simd::fast_powf(px[1] as f32 / 65535.0, inv_gamma),
                jxl_simd::fast_powf(px[2] as f32 / 65535.0, inv_gamma),
            ]
        })
        .collect()
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

/// Gamma u16 grayscale → linear f32 RGB (gray→R=G=B). `linear = (encoded/65535)^(1/gamma)`
fn gamma_gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize, gamma: f32) -> Vec<f32> {
    let inv_gamma = 1.0 / gamma;
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = jxl_simd::fast_powf(px[0] as f32 / 65535.0, inv_gamma);
            [v, v, v]
        })
        .collect()
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
fn gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    let pixels: &[u16] = bytemuck::cast_slice(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = srgb_to_linear_f(px[0] as f32 / 65535.0);
            [v, v, v]
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
}
