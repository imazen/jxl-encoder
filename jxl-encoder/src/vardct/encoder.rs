// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Main tiny encoder implementation.

use super::ac_strategy::{
    AcStrategyMap, adjust_quant_field_float_with_distance, adjust_quant_field_with_distance,
    compute_ac_strategy,
};
use super::adaptive_quant::quantize_quant_field;
use super::chroma_from_luma::{CflMap, compute_cfl_map};
use super::common::*;
use super::frame::{DistanceParams, write_toc, write_toc_with_permutation};
use super::gaborish::gaborish_inverse;
use super::noise::{denoise_xyb, estimate_noise_params, noise_quality_coef};
use super::static_codes::{get_ac_entropy_code, get_dc_entropy_code};
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::debug_rect;
use crate::error::{Error, Result};
use crate::headers::frame_header::FrameHeader;

// Re-export types from entropy_code sub-module.
pub(crate) use super::entropy_code::{BuiltEntropyCode, force_strategy_map};

/// Validate the three XYB planes against caller-configured policy at
/// the conversion→pipeline boundary.
///
/// The opsin transform (`color::xyb::linear_rgb_to_xyb`) is
/// `cbrt(mixed + bias) - cbrt(bias)` per channel — bias is positive, so
/// the cube-root argument is always strictly positive for any finite
/// linear-RGB input, and the output is finite. Non-finite XYB at this
/// boundary indicates an upstream bug:
///
/// 1. The caller passed non-finite linear-RGB (the `LinearF32` pixel
///    layouts allow this; we should validate at intake but currently
///    don't).
/// 2. An upstream computation (butteraugli-loop reconstruction, EPF,
///    gaborish) leaked NaN into XYB — fix-the-upstream bug.
/// 3. Memory corruption (the original v09/v11 sweep cause, mitigated
///    by removing `unsafe-performance` in PR #34).
///
/// Behavior:
///
/// - [`NonFiniteAction::Error`] (default): runs the **read-only**
///   `is_finite_plane` SIMD scan (~55 GB/s) and returns
///   [`crate::error::Error::InvalidInput`] on first non-finite plane.
///   Nothing downstream ever sees the bad data.
/// - [`NonFiniteAction::Sanitize`]: runs the read-modify-write
///   `sanitize_finite` SIMD kernel (~12.5 GB/s) and replaces non-finite
///   values with `0.0`. Encoding continues.
pub(crate) fn validate_xyb_planes(
    action: crate::api::NonFiniteAction,
    x: &mut [f32],
    y: &mut [f32],
    b: &mut [f32],
) -> crate::error::Result<()> {
    match action {
        crate::api::NonFiniteAction::Error => {
            if !(jxl_simd::is_finite_plane(x)
                && jxl_simd::is_finite_plane(y)
                && jxl_simd::is_finite_plane(b))
            {
                return Err(crate::error::Error::InvalidInput(
                    "non-finite (NaN / ±Inf) value detected in XYB pixel planes. \
                     This is an upstream bug. Common causes: caller passed \
                     non-finite linear-RGB, butteraugli-loop reconstruction \
                     polluted XYB, memory corruption. To accept and silently \
                     replace with 0.0 instead of erroring, use \
                     LossyConfig::with_non_finite_action(NonFiniteAction::Sanitize)."
                        .into(),
                ));
            }
        }
        crate::api::NonFiniteAction::Sanitize => {
            // Always run all three so each plane gets cleaned even if
            // an earlier one was clean.
            let _ = jxl_simd::sanitize_finite(x);
            let _ = jxl_simd::sanitize_finite(y);
            let _ = jxl_simd::sanitize_finite(b);
        }
    }
    Ok(())
}

/// Output of a VarDCT encode operation.
pub struct VarDctOutput {
    /// Encoded JXL codestream bytes.
    pub data: Vec<u8>,
    /// Per-strategy first-block counts, indexed by raw strategy code (0..19).
    pub strategy_counts: [u32; 19],
}

// ── Squeeze-on-extras quantization constants (libjxl parity) ────────────────
//
// Mirrors `enc_modular.cc:82-103`. Used by both the existing no-squeeze
// extras quantizer ([`VarDctEncoder::compute_extra_pixel_quantizer`])
// and the chunk-1 framework function
// ([`VarDctEncoder::compute_extra_pixel_quantizer_shifted`]) that will
// drive the responsive=1 alpha pipeline once the squeeze application
// for extras lands (chunk 2 — see CHANGELOG entry on the
// `with_alpha_squeeze` flag).

// Chunk-1 status note (CLAUDE.md "Investigation Notes"): the four
// items below are framework-only. They become live the moment
// `compute_extra_pixel_quantizer_shifted` is routed from the extras
// subbitstream writer (chunk 2). Unit-test coverage in the
// `tests` module below keeps the constants and the shift-aware
// quantizer fn exercised so a chunk-2 wiring change can't silently
// regress the libjxl-parity formula. Allow dead_code so production
// builds stay clean while the wire-up is staged.

/// Squeeze quality factor (libjxl `enc_modular.cc:82`).
/// "Decrease this number for higher quality."
#[allow(dead_code)]
const SQUEEZE_QUALITY_FACTOR_CONST: f32 = 0.35;

/// Squeeze luma factor (libjxl `enc_modular.cc:85`).
/// "Decrease this number for higher quality luma."
#[allow(dead_code)]
const SQUEEZE_LUMA_FACTOR_CONST: f32 = 1.1;

/// Length of the per-shift quantization table.
#[allow(dead_code)]
const SQUEEZE_LUMA_QTABLE_LEN: usize = 16;

/// Per-shift quantization multipliers for the luma (and "anything
/// non-chroma" — e.g. alpha) channels in the responsive=1 modular
/// pipeline. Mirrors libjxl `enc_modular.cc:101-103`.
///
/// Index 0 (= squeeze depth 0) corresponds to the lowest-frequency
/// average channel; higher indices correspond to deeper HF residual
/// bands. The values halve roughly every step from 163.84 down to
/// 0.005, so residual HF bands collapse to `q = 1` (lossless within
/// integer rounding) quickly — that is where the responsive=1
/// compression win on alpha comes from.
#[allow(dead_code)]
const SQUEEZE_LUMA_QTABLE: [f32; SQUEEZE_LUMA_QTABLE_LEN] = [
    163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01,
    0.005,
];

/// Tiny JPEG XL encoder.
///
/// This is a simplified VarDCT encoder based on libjxl-tiny that uses:
/// - Only DCT8, DCT8x16, DCT16x8 transforms
/// - Huffman or ANS entropy coding
/// - Default zig-zag coefficient order
/// - Fixed context tree for DC
pub struct VarDctEncoder {
    /// Target distance (quality). 1.0 = visually lossless.
    pub distance: f32,
    /// Effort level (1–11). Controls AC strategy gating and search depth.
    /// e10/e11 extends libjxl kTortoise=9 via extended search budgets.
    pub effort: u8,
    /// Centralized effort-derived decisions. All effort-gated constants and
    /// thresholds are read from this profile instead of inline `if effort >= N`.
    pub profile: crate::effort::EffortProfile,
    /// Use dynamic Huffman codes built from actual token frequencies.
    /// When true (default), uses a two-pass mode: collect tokens first, build optimal codes, then write.
    /// When false, uses pre-computed static codes (streaming, single-pass).
    pub optimize_codes: bool,
    /// Use enhanced histogram clustering with pair merge refinement.
    /// Only effective when `optimize_codes` is true.
    ///
    /// Note: The enhanced clustering algorithm was designed for ANS entropy coding
    /// and may not provide benefits (or may slightly increase size) when used with
    /// Huffman coding. This option is experimental.
    pub enhanced_clustering: bool,
    /// Use ANS entropy coding instead of Huffman.
    /// Only effective when `optimize_codes` is true (requires two-pass mode).
    /// ANS typically produces 5-10% smaller files than Huffman.
    pub use_ans: bool,
    /// Enable chroma-from-luma (CfL) optimization.
    /// When true (default), computes per-tile ytox/ytob values via least-squares fitting.
    /// When false, uses ytox=0, ytob=0 (no chroma decorrelation).
    pub cfl_enabled: bool,
    /// Enable adaptive AC strategy selection (DCT8/DCT16x8/DCT8x16).
    /// When true (default), selects the best transform size per 16x16 block region.
    /// When false, uses DCT8 for all blocks.
    pub ac_strategy_enabled: bool,
    /// Enable custom coefficient ordering.
    /// When true (default when optimize_codes is true), reorders AC coefficients
    /// so frequently-zero positions appear last, reducing bitstream size.
    /// Only effective when `optimize_codes` is true (requires two-pass mode).
    pub custom_orders: bool,
    /// Force a specific AC strategy for all blocks (for testing).
    /// When Some(strategy), uses that raw strategy code for all blocks that fit.
    /// None (default) uses normal strategy selection based on `ac_strategy_enabled`.
    pub force_strategy: Option<u8>,
    /// Enable noise synthesis.
    /// When true, estimates noise parameters from the image and encodes them
    /// in the frame header. The decoder regenerates noise during rendering.
    /// Off by default (matching libjxl's default).
    pub enable_noise: bool,
    /// When set, synthesises noise parameters from the given ISO value
    /// instead of estimating from the image. Matches libjxl's
    /// `--photon_noise=ISO` flag and bypasses `enable_noise`. Useful
    /// for re-encoding denoised content where the caller wants to
    /// inject controlled grain matching a target camera ISO.
    pub photon_noise_iso: Option<f32>,
    /// Caller-supplied 8-point noise LUT. Overrides content
    /// estimation when set. Mirrors libjxl's `cparams.manual_noise`.
    /// Lower priority than `photon_noise_iso`, higher than
    /// `enable_noise`. Each entry should be in `[0.0, ~1.0]` —
    /// values are written as 10-bit-quantised samples into the
    /// frame-header noise block.
    pub manual_noise_lut: Option<[f32; 8]>,
    /// Caller-supplied AC quantiser rescale. When `Some(r)` and `r != 1.0`,
    /// `global_scale` is multiplied by `r` (and `scale` / `inv_scale`
    /// / `scale_dc` recomputed) after the standard
    /// `DistanceParams::compute_for_profile` step. Mirrors libjxl's
    /// `cparams.quant_ac_rescale`. `r < 1.0` → finer AC quant
    /// (larger files, higher quality); `r > 1.0` → coarser (smaller
    /// files, lower quality). Default behaviour (`None`) leaves
    /// `global_scale` untouched.
    pub quant_ac_rescale: Option<f32>,
    /// Caller-supplied source-image butteraugli distance for re-encode
    /// pipelines. When set, x_qm_scale (and other distance-based
    /// heuristics that compare against source quality, not target
    /// quality) ramp against this value instead of `distance`.
    /// Mirrors libjxl's `cparams.original_butteraugli_distance`.
    /// `None` = use `distance` (ground-truth source).
    pub original_distance: Option<f32>,
    /// Enable Wiener denoising pre-filter (requires `enable_noise`).
    /// When true, applies a conservative Wiener filter to remove estimated noise
    /// before encoding. The decoder re-adds noise from the encoded parameters.
    /// Provides 1-8% file size savings with near-zero Butteraugli quality impact.
    /// Off by default (libjxl does not have a denoising pre-filter).
    pub enable_denoise: bool,
    /// Enable gaborish inverse pre-filter.
    /// When true (default), applies a 5x5 sharpening kernel to XYB before DCT
    /// and signals gab=1 in the frame header. The decoder applies a 3x3 blur
    /// to compensate, reducing blocking artifacts.
    /// Matches the libjxl VarDCT encoder default.
    pub enable_gaborish: bool,
    /// Override the edge-preserving filter (EPF) iteration count.
    ///
    /// `None` (default) = use the distance-derived `epf_iters` from
    /// [`DistanceParams`] (libjxl thresholds `[0.7, 1.5, 4.0]`).
    /// `Some(0..=3)` = force the given count; `0` disables EPF and
    /// skips the dynamic sharpness search. Mirrors libjxl
    /// `cparams.epf` (`enc_frame.cc:284-285`).
    pub epf_level_override: Option<u32>,
    /// Enable error diffusion in AC quantization.
    /// When true, spreads quantization error to neighboring coefficients in
    /// zigzag order, helping preserve smooth gradients at high compression.
    /// Off by default (modest quality improvement, slight performance cost).
    pub error_diffusion: bool,
    /// Enable pixel-domain loss calculation in AC strategy selection.
    /// When true, uses full libjxl's pixel-domain loss model (IDCT error,
    /// per-pixel masking, 8th power norm). This provides better distance
    /// calibration matching cjxl's output.
    /// When false (default), uses coefficient-domain loss (libjxl-tiny style).
    /// Note: Requires `ac_strategy_enabled` to have any effect.
    pub pixel_domain_loss: bool,
    /// Enable LZ77 backward references in entropy coding.
    /// When true, compresses token streams using LZ77 length+distance tokens.
    /// Only effective with two-pass mode (optimize_codes=true) and ANS (use_ans=true).
    /// Off by default — works for most cases but has known interactions with certain
    /// forced strategy combinations (DCT2x2, IDENTITY) that cause InvalidAnsStream.
    pub enable_lz77: bool,
    /// LZ77 method to use when enable_lz77 is true.
    ///
    /// - `Rle`: Only matches consecutive identical values (fast, limited on photos)
    /// - `Greedy`: Hash chain backward references (slower, 1-3% better on photos)
    ///
    /// Default: `Greedy` (best compression)
    pub lz77_method: crate::entropy_coding::lz77::Lz77Method,
    /// Enable DC tree learning.
    /// When true, learns an optimal context tree for DC coding from image content
    /// instead of using the fixed GRADIENT_CONTEXT_LUT.
    /// **DISABLED/BROKEN**: The learned tree doesn't correctly route AC metadata
    /// samples to contexts 0-10. Fixing requires parsing the static tree structure
    /// and splicing in the learned DC subtree while preserving AC metadata routing.
    /// Expected gain (~1.2% overall) doesn't justify the complexity. See CLAUDE.md.
    pub dc_tree_learning: bool,
    /// Number of butteraugli quantization loop iterations.
    /// When > 0, iteratively refines the per-block quant field using butteraugli
    /// perceptual distance feedback. Each iteration: encode → reconstruct → measure
    /// → adjust quant_field. AC strategy is kept fixed; only quant_field changes.
    ///
    /// libjxl uses 2 iterations at effort 8, 4 at effort 9.
    /// Requires the `butteraugli-loop` feature.
    ///
    /// Default: 0 (disabled)
    #[cfg(feature = "butteraugli-loop")]
    pub butteraugli_iters: u32,
    /// Number of SSIM2 quantization loop iterations.
    /// Alternative to butteraugli loop: uses per-block linear RGB RMSE + full-image SSIM2.
    /// Requires the `ssim2-loop` feature.
    ///
    /// Default: 0 (disabled)
    #[cfg(feature = "ssim2-loop")]
    pub ssim2_iters: u32,
    /// Number of zensim quantization loop iterations.
    /// Alternative to butteraugli loop: uses zensim's psychovisual metric for both
    /// global quality tracking and per-pixel spatial error map (diffmap in XYB space).
    /// Also refines AC strategy by splitting large transforms with high perceptual error.
    /// Requires the `zensim-loop` feature.
    ///
    /// Default: 0 (disabled)
    #[cfg(feature = "zensim-loop")]
    pub zensim_iters: u32,
    /// Whether the input has 16-bit samples. When true, the file header signals
    /// bit_depth=16 instead of 8. The actual VarDCT encoding is the same (XYB
    /// is always f32 internally), but the decoder uses this to reconstruct at
    /// the correct output bit depth.
    pub bit_depth_16: bool,
    /// ICC profile to embed in the codestream.
    /// When Some, writes has_icc=1 and encodes the profile after the file header.
    pub icc_profile: Option<Vec<u8>>,
    /// Enable patches (dictionary-based repeated pattern detection).
    /// When true, detects repeated rectangular elements (text glyphs, buttons, icons)
    /// and stores unique patterns once in a reference frame. Huge wins on screenshots.
    /// On by default for lossy encoding.
    pub enable_patches: bool,
    /// Enable libjxl-style **dot detection** (refs #19). When true and
    /// effort >= 7 and distance >= 3.0 and no text-like patches were
    /// found, the encoder runs the star-field / specular-highlight
    /// detector ([`super::dot_detection::detect_gaussian_ellipses`])
    /// and folds the detected Gaussian ellipses into the patch
    /// dictionary via [`super::patches::PatchesData::from_dots`].
    /// Default on internally is `false` so existing callers that
    /// build a `VarDctEncoder` directly preserve their behaviour; the
    /// public [`crate::LossyConfig`] defaults this to `true` to
    /// match libjxl's `Override::kDefault` (`enc_patch_dictionary.cc:632`).
    pub enable_dot_detection: bool,
    /// Encoder mode: Reference (match libjxl) or Experimental (own improvements).
    pub encoder_mode: crate::api::EncoderMode,
    /// Manual splines to overlay on the image (opt-in, None by default).
    pub splines: Option<Vec<crate::vardct::splines::Spline>>,
    /// Enable automatic spline detection from the input XYB planes.
    ///
    /// When `true` **and** [`Self::splines`] is `None` **and** [`Self::effort`]
    /// is ≥ 7, the encoder calls [`crate::vardct::splines::find_splines`]
    /// after the patches-subtract step (mirroring libjxl
    /// `enc_heuristics.cc:1048-1054`: `speed_tier <= kSquirrel`).
    ///
    /// Chunk 1 (this commit) ships the API + wiring with a stub detector
    /// that returns an empty vec — produces byte-identical output to the
    /// default path. Chunk 2 lands the real ridge-following detector;
    /// see [`crate::vardct::splines::find_splines`] for the algorithm sketch.
    ///
    /// A user-supplied non-empty [`Self::splines`] always wins over
    /// auto-detection — the manual API stays the source of truth when
    /// both are set.
    pub auto_splines: bool,
    /// Whether the input is grayscale. When true, the file header signals
    /// ColorSpace::Gray instead of RGB. VarDCT still operates in XYB (3 channels)
    /// internally — this only affects the output colorspace the decoder targets.
    pub is_grayscale: bool,
    /// Progressive encoding mode (Single, QuantizedAcFullAc, DcVlfLfAc).
    /// When not Single, AC coefficients are split across multiple passes with
    /// shift-based precision reduction for early preview rendering.
    pub progressive: crate::api::ProgressiveMode,
    /// Enable LfFrame (separate DC frame).
    /// When true, DC coefficients are encoded as a separate modular frame
    /// (frame_type=1, dc_level=1) before the main VarDCT frame, with
    /// distance-scaled quantization factors matching libjxl's progressive_dc >= 1.
    pub use_lf_frame: bool,
    /// Custom gamma (encoding exponent) from source image.
    /// When Some, writes have_gamma=true in the JXL header and uses gamma
    /// linearization instead of sRGB TF. Example: 0.45455 for gamma 2.2.
    pub source_gamma: Option<f32>,
    /// Explicit color encoding override for the JXL header.
    /// When Some, this is used instead of deriving from source_gamma / defaults.
    /// Allows signaling HDR (PQ, HLG) or non-sRGB primaries (BT.2020, P3).
    pub color_encoding: Option<crate::headers::color_encoding::ColorEncoding>,
    /// Peak display luminance in nits for ToneMapping. Default 255.0 (SDR).
    pub intensity_target: f32,
    /// Minimum display luminance in nits for ToneMapping. Default 0.0.
    pub min_nits: f32,
    /// Intrinsic display size `(width, height)`, if different from coded dimensions.
    pub intrinsic_size: Option<(u32, u32)>,
    /// When `true`, the alpha extra-channel header is emitted with
    /// `alpha_associated=true` (premultiplied / associated alpha).
    /// The encoder DOES NOT premultiply or unpremultiply pixels here —
    /// it's the caller's responsibility to feed straight color and
    /// declare the encoded form. The high-level `EncodeRequest` /
    /// `LossyEncoder` API does the unpremultiply pre-pass before
    /// handing us the linear RGB.
    pub alpha_associated: bool,
    /// Override `BitDepth.bits_per_sample` in the codestream header.
    /// `None` → derives from `bit_depth_16` (8 or 16). `Some(N)` →
    /// emits `N`. Closes configurable bits_per_sample sub-feature
    /// of #18 — lets callers signal 10/12/14-bit input precision.
    pub bits_per_sample_override: Option<u32>,
    /// When `true`, AC groups in the multi-group TOC are emitted in
    /// concentric-square order from the image center (libjxl
    /// `cparams.centerfirst`). The TOC `permuted` flag is set and
    /// the permutation is encoded as Lehmer codes via
    /// `coeff_order::tokenize_permutation` /
    /// `build_and_write_coeff_orders`. Closes #14.
    pub center_first: bool,
    /// Decoder upsampling factor for the main frame. `1` (default) =
    /// no upsampling; `2`/`4`/`8` = the decoder upsamples its decoded
    /// pixel buffer by this factor along each axis after rendering.
    ///
    /// When > 1, the caller must:
    /// - pass the **downsampled** dimensions and pixel buffers to
    ///   [`Self::encode`] (use [`super::resampling::box_downsample_rgb`]
    ///   etc.); the encoder operates entirely at the downsampled
    ///   resolution.
    /// - The codestream's file header still reports the **original**
    ///   (pre-downsample) dimensions — the encoder multiplies
    ///   the supplied (width, height) by `upsampling` when building
    ///   the file header, and writes `upsampling` into the frame
    ///   header. The decoder upsamples to that target.
    ///
    /// Used by [`crate::api::LossyConfig::with_resampling`] (refs #12).
    pub upsampling: u32,
    /// Decoder upsampling mode (libjxl
    /// `JxlEncoderSetUpsamplingMode(enc, factor, mode)`,
    /// `enc_modular.cc` etc.). Only emitted when [`Self::upsampling`] > 1
    /// (the spec ties the LUTs to the upsampling factor).
    ///
    /// - `None` (default) / `Some(-1)` — fancy default upsampling
    ///   (`custom_weight_mask=0`, no custom LUT in the file header).
    /// - `Some(0)` — nearest-neighbour upsampling for the active
    ///   `upsampling` factor: emits a zeroed LUT with a single 1.0
    ///   impulse, matching libjxl `JxlEncoderSetUpsamplingMode(_, _, 0)`.
    /// - `Some(1)` — "pixel dots" upsampling (nearest with cut corners).
    ///   Only meaningful for `upsampling == 4` or `upsampling == 8`;
    ///   silently behaves as nearest for `upsampling == 2` (matches
    ///   libjxl's per-factor table).
    pub upsampling_mode: Option<i32>,
    /// Center X for [`Self::center_first`] AC group reorder. `None`
    /// (default) uses `width / 2`. Mirrors libjxl
    /// `cparams.center_x` (`JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X`).
    pub center_x: Option<u32>,
    /// Center Y for [`Self::center_first`] AC group reorder. `None`
    /// (default) uses `height / 2`. Mirrors libjxl
    /// `cparams.center_y` (`JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y`).
    pub center_y: Option<u32>,
    /// Optional separate butteraugli distance for the alpha extra
    /// channel (CLI passthrough — libjxl `cjxl --alpha_distance`,
    /// `enc_params.h:alpha_distance`). `None` (default) and
    /// `Some(0.0)` keep the lossless alpha path (gradient predictor
    /// + LZ77 RLE).
    ///
    /// `Some(d)` with `d > 0.0` engages the lossy alpha pipeline:
    /// computes an integer pixel quantizer via
    /// [`Self::compute_extra_pixel_quantizer`] (libjxl no-squeeze
    /// formula, `enc_modular.cc:973-1027`), pre-quantizes each alpha
    /// pixel to the nearest multiple of `q`, and emits the modular
    /// extras tree leaf with `(mul_log, mul_bits)` carrying that
    /// multiplier so the decoder reconstructs
    /// `pixel = prediction + val * q`
    /// (`modular/encoding/encoding.cc:186-191`). Applied per channel:
    /// mixed-extras inputs route an alpha-typed leaf through this
    /// quantizer while non-alpha extras (depth, spot color, ...) stay
    /// lossless (`q = 1`) on the same frame.
    pub alpha_distance: Option<f32>,
    /// Opt-in: engage the **squeeze-on-extras** (responsive=1) lossy
    /// alpha pipeline rather than the default no-squeeze pixel
    /// quantizer (`enc_modular.cc:973-1027`). Default `false`.
    ///
    /// libjxl's default is `responsive=1` for lossy alpha; cjxl
    /// out-of-the-box gets `-18%` to `-160%` smaller bytes on
    /// non-opaque alpha than our `responsive=0` path (audit:
    /// `a160deb7`, three-image sweep at d ∈ {0.5, 1.0, 2.0, 5.0}).
    ///
    /// **Chunk-1 status (this flag)**: ungates a framework path that
    /// validates the per-band quantizer table
    /// ([`SQUEEZE_LUMA_QTABLE`]) and the [`Self::alpha_squeeze_engaged`]
    /// predicate. The actual squeeze-application + per-band quantizer
    /// routing through [`Self::compute_extra_pixel_quantizer_shifted`]
    /// is queued for chunk 2 — until then the extras subbitstream
    /// writer surfaces an explicit `Error::Unsupported` when this
    /// flag is `true` AND a non-trivial lossy alpha case is
    /// requested. The default `false` keeps the existing
    /// no-squeeze path byte-for-byte identical (hash-locks 36/36).
    ///
    /// Set via [`crate::LossyConfig::with_alpha_squeeze`].
    pub alpha_squeeze: bool,
    /// Policy for non-finite XYB values at the conversion→pipeline
    /// boundary. See [`crate::api::NonFiniteAction`].
    pub non_finite_action: crate::api::NonFiniteAction,
    /// Per-encode allocation budget. When `Some`, dimension-driven
    /// buffers (XYB planes, padded scratch) reserve their byte count
    /// before allocating and surface
    /// [`crate::error::Error::AllocationLimit`] if the cap would be
    /// exceeded. When `None` (the test/library default), allocation
    /// proceeds unbounded.
    pub(crate) budget: Option<alloc::sync::Arc<crate::budget::MemoryBudget>>,
}

impl Default for VarDctEncoder {
    fn default() -> Self {
        Self {
            distance: 1.0,
            effort: 7,
            profile: crate::effort::EffortProfile::lossy(7, crate::api::EncoderMode::Reference),
            optimize_codes: true,
            enhanced_clustering: true, // Profile-driven: e9+ for Best, Fast otherwise
            use_ans: true,             // ANS produces 4-10% smaller files than Huffman
            cfl_enabled: true,
            ac_strategy_enabled: true,
            custom_orders: true,
            force_strategy: None,
            enable_noise: false,
            photon_noise_iso: None,
            manual_noise_lut: None,
            quant_ac_rescale: None,
            original_distance: None,
            enable_denoise: false,
            enable_gaborish: true,
            epf_level_override: None,
            error_diffusion: false, // libjxl accepts param but never uses it in QuantizeBlockAC
            pixel_domain_loss: true, // Full libjxl pixel-domain loss: +0.2-1.9 SSIM2 at all distances
            enable_lz77: false,      // LZ77 has known interactions with DCT2x2/IDENTITY strategies
            lz77_method: crate::entropy_coding::lz77::Lz77Method::Greedy, // Best compression
            dc_tree_learning: false, // DC tree learning (experimental)
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 0, // Effort-gated: default off (effort 7). Set via LossyConfig.
            #[cfg(feature = "ssim2-loop")]
            ssim2_iters: 0, // Off by default. Set via LossyConfig.
            #[cfg(feature = "zensim-loop")]
            zensim_iters: 0, // Off by default. Set via LossyConfig.
            bit_depth_16: false,
            icc_profile: None,
            enable_patches: true, // Patches: huge wins on screenshots, zero cost on photos
            enable_dot_detection: false, // refs #19; LossyConfig flips this to true by default
            encoder_mode: crate::api::EncoderMode::Reference,
            splines: None,
            auto_splines: false, // Opt-in (chunk 1 ships stub; default-off keeps hash-locks)
            is_grayscale: false,
            progressive: crate::api::ProgressiveMode::Single,
            use_lf_frame: false,
            source_gamma: None,
            color_encoding: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            intrinsic_size: None,
            alpha_associated: false,
            bits_per_sample_override: None,
            center_first: false,
            upsampling: 1,
            upsampling_mode: None,
            center_x: None,
            center_y: None,
            alpha_distance: None,
            // Chunk-1 default: keep existing no-squeeze alpha pipeline.
            // Setting `true` ungates the chunk-1 framework path which
            // surfaces `Error::Unsupported` from the extras writer until
            // chunk-2 wires the per-band squeeze application.
            alpha_squeeze: false,
            non_finite_action: crate::api::NonFiniteAction::default(),
            budget: None,
        }
    }
}

impl VarDctEncoder {
    /// Create a new tiny encoder with the given distance.
    pub fn new(distance: f32) -> Self {
        Self {
            distance,
            effort: 7,
            profile: crate::effort::EffortProfile::lossy(7, crate::api::EncoderMode::Reference),
            optimize_codes: true,
            enhanced_clustering: true, // Profile-driven: e9+ for Best, Fast otherwise
            use_ans: true,             // ANS produces 4-10% smaller files than Huffman
            cfl_enabled: true,
            ac_strategy_enabled: true,
            custom_orders: true,
            force_strategy: None,
            enable_noise: false,
            photon_noise_iso: None,
            manual_noise_lut: None,
            quant_ac_rescale: None,
            original_distance: None,
            enable_denoise: false,
            enable_gaborish: true,
            epf_level_override: None,
            error_diffusion: false, // libjxl accepts param but never uses it in QuantizeBlockAC
            pixel_domain_loss: true, // Full libjxl pixel-domain loss: +0.2-1.9 SSIM2
            enable_lz77: false,     // LZ77 has known interactions with DCT2x2/IDENTITY strategies
            lz77_method: crate::entropy_coding::lz77::Lz77Method::Greedy, // Best compression
            dc_tree_learning: false, // DC tree learning (experimental)
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 0, // Effort-gated: default off (effort 7). Set via LossyConfig.
            #[cfg(feature = "ssim2-loop")]
            ssim2_iters: 0, // Off by default. Set via LossyConfig.
            #[cfg(feature = "zensim-loop")]
            zensim_iters: 0, // Off by default. Set via LossyConfig.
            bit_depth_16: false,
            icc_profile: None,
            enable_patches: true, // Patches: huge wins on screenshots, zero cost on photos
            enable_dot_detection: false, // refs #19; LossyConfig flips this to true by default
            encoder_mode: crate::api::EncoderMode::Reference,
            splines: None,
            auto_splines: false, // Opt-in (chunk 1 ships stub; default-off keeps hash-locks)
            is_grayscale: false,
            progressive: crate::api::ProgressiveMode::Single,
            use_lf_frame: false,
            source_gamma: None,
            color_encoding: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            intrinsic_size: None,
            alpha_associated: false,
            bits_per_sample_override: None,
            center_first: false,
            upsampling: 1,
            upsampling_mode: None,
            center_x: None,
            center_y: None,
            alpha_distance: None,
            // Chunk-1 default: keep existing no-squeeze alpha pipeline.
            // Setting `true` ungates the chunk-1 framework path which
            // surfaces `Error::Unsupported` from the extras writer until
            // chunk-2 wires the per-band squeeze application.
            alpha_squeeze: false,
            non_finite_action: crate::api::NonFiniteAction::default(),
            budget: None,
        }
    }

    /// Attach a per-encode allocation budget. Internal-only; the public
    /// API plumbs this from [`crate::api::Limits::max_memory_bytes`].
    ///
    /// Currently the API path sets [`Self::budget`] directly; this
    /// builder is here for future call sites (e.g., the streaming
    /// encoder, precomputed entry points) that don't have field
    /// access.
    #[doc(hidden)]
    #[allow(dead_code)]
    pub(crate) fn with_budget(
        mut self,
        budget: alloc::sync::Arc<crate::budget::MemoryBudget>,
    ) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Apply [`Self::epf_level_override`] to a freshly computed
    /// [`DistanceParams`], if the caller pinned one. Matches libjxl
    /// `enc_frame.cc:284-285`: any non-default value overrides the
    /// distance-derived `epf_iters` (including `0`, which forces EPF
    /// off and skips the dynamic sharpness search downstream).
    #[inline]
    pub(crate) fn apply_epf_level_override(&self, params: &mut DistanceParams) {
        if let Some(level) = self.epf_level_override {
            params.epf_iters = level;
        }
    }

    /// Compute the integer pixel quantizer `q` to apply to a single
    /// extra channel from [`Self::alpha_distance`] and the channel's
    /// bit depth + type, mirroring libjxl's lossy-modular
    /// `quantizers[component] * squeeze_quality_factor *
    /// squeeze_luma_factor * squeeze_luma_qtable[shift]` at
    /// `shift = 0` and `responsive = false`
    /// (`enc_modular.cc:973-1027`):
    ///
    /// - `quantizer = 0.25 * 0.1 = 0.025` (no-squeeze, "just color quantization")
    /// - `bitdepth_correction = (2^bits - 1) / 255`
    /// - `q_float = quantizer * dist * bitdepth_correction
    ///            * squeeze_quality_factor (0.35) * squeeze_luma_factor (1.1)
    ///            * squeeze_luma_qtable[0] (163.84)`
    /// - `q = max(1, floor(q_float))`
    ///
    /// Returns `1` (lossless) for any extra channel that does not carry
    /// its own distance knob: today only alpha-typed channels use
    /// `alpha_distance`. Non-alpha extras (depth, spot, selection mask,
    /// thermal, CFA, ...) stay lossless because we do not currently
    /// expose per-channel `ec_distance`. Also returns `1` when
    /// `alpha_distance` is `None`, `Some(d <= 0.0)`, or any input that
    /// rounds below `1` after libjxl's `floor`.
    pub(crate) fn compute_extra_pixel_quantizer(
        &self,
        extras_bits: u32,
        ec_type: crate::headers::extra_channels::ExtraChannelType,
    ) -> u32 {
        let dist = match ec_type {
            crate::headers::extra_channels::ExtraChannelType::Alpha => match self.alpha_distance {
                Some(d) if d > 0.0 => d,
                _ => return 1,
            },
            // Non-alpha extras have no per-channel distance knob today.
            // libjxl supports `cparams.ec_distance[i]` per extra channel;
            // when that is wired through the public API, dispatch here
            // by `ec_type` / channel index.
            _ => return 1,
        };
        // libjxl clamps non-zero alpha_distance to [0.01, 25.0] at the
        // public API boundary (encode.cc:1558). Mirror the lower clamp
        // here so callers writing `Some(0.001)` don't silently regress
        // to lossless.
        let dist = dist.clamp(0.01, 25.0);
        let ec_maxval = if extras_bits >= 32 {
            // i32::MAX would overflow the (1<<bits)-1 calc; libjxl
            // treats it as 0 (no correction). Treat the same way.
            0u64
        } else {
            (1u64 << extras_bits) - 1
        };
        let bitdepth_correction = if ec_maxval == 0 {
            1.0
        } else {
            (ec_maxval as f32) / 255.0
        };
        const QUANTIZER: f32 = 0.025; // 0.25 * 0.1 (no responsive/squeeze)
        const SQUEEZE_QUALITY_FACTOR: f32 = 0.35;
        const SQUEEZE_LUMA_FACTOR: f32 = 1.1;
        const SQUEEZE_LUMA_QTABLE_0: f32 = 163.84;
        let q_float = QUANTIZER
            * dist
            * bitdepth_correction
            * SQUEEZE_QUALITY_FACTOR
            * SQUEEZE_LUMA_FACTOR
            * SQUEEZE_LUMA_QTABLE_0;
        // libjxl truncates float→int via implicit conversion (C++
        // `int q = ...`), so we match with `floor`. Anything < 1
        // clamps to 1 (lossless).
        let q = q_float.floor() as i64;
        if q < 1 { 1 } else { q as u32 }
    }

    /// Build the per-channel quantizer vector for an extras slice.
    ///
    /// Returns `Vec<u32>` of length `extras.len()`. Each entry is
    /// computed via [`Self::compute_extra_pixel_quantizer`] from that
    /// channel's bit depth + [`ExtraChannelType`]. `extras.is_empty()`
    /// returns an empty vec.
    ///
    /// Mixed-extras invariant: the existing single-extra (`extras.len()
    /// == 1`) path delegates here, so a single alpha extra at index 0
    /// continues to produce the same `q` value as before.
    pub(crate) fn compute_extras_pixel_quantizers(
        &self,
        extras: &[super::extras::VardctExtra<'_>],
    ) -> alloc::vec::Vec<u32> {
        extras
            .iter()
            .map(|ec| {
                self.compute_extra_pixel_quantizer(
                    ec.info.bit_depth.bits_per_sample,
                    ec.info.ec_type,
                )
            })
            .collect()
    }

    /// Compute the integer pixel quantizer `q` for one
    /// **squeeze-shifted** sub-channel of an extras lossy plane,
    /// mirroring the responsive=1 branch of libjxl
    /// `enc_modular.cc:973-1027`:
    ///
    /// ```text
    /// quantizer       = 0.25            // base; NO `* 0.1` because responsive=1
    /// q_float         = quantizer
    ///                  * dist
    ///                  * bitdepth_correction
    ///                  * squeeze_quality_factor      (0.35)
    ///                  * squeeze_luma_factor         (1.1)
    ///                  * squeeze_luma_qtable[shift]
    /// q               = max(1, floor(q_float))
    /// ```
    ///
    /// Here `shift` is libjxl's per-channel `hshift + vshift` decremented
    /// by 1 (`enc_modular.cc:1006-1008`) — i.e. the depth of squeeze
    /// halvings already applied to this sub-channel — then clamped to
    /// `[0, 15]` (`SQUEEZE_LUMA_QTABLE_LEN - 1`).
    ///
    /// The `shift == 0` path **deliberately diverges** from
    /// [`Self::compute_extra_pixel_quantizer`] (the no-squeeze formula),
    /// which folds in an extra `* 0.1` factor (`enc_modular.cc:976-981`,
    /// "lossy compression without Squeeze transform is just color
    /// quantization"). At `shift == 0` and the same `dist`/`bits`,
    /// this function returns `~10×` the q value of the no-squeeze
    /// path because squeezed averages need much coarser quantization
    /// than raw pixels would. The HF residual sub-channels (shift > 0)
    /// then drop very quickly toward `q = 1` via the qtable.
    ///
    /// **Chunk-1 status**: this function is wired into the
    /// `with_alpha_squeeze` opt-in flag but **not yet routed into the
    /// extras subbitstream** — see
    /// [`Self::alpha_squeeze`] / chunk-2 plan in CHANGELOG. Today it
    /// is unit-testable and ready for chunk-2 to consume.
    ///
    /// Like [`Self::compute_extra_pixel_quantizer`] this returns `1`
    /// for non-alpha extras and for `alpha_distance` of `None` /
    /// `Some(0.0)` / `Some(d <= 0)` — preserves the lossless
    /// contract end-to-end.
    #[allow(dead_code)] // chunk-1 framework — chunk 2 wires this in.
    pub(crate) fn compute_extra_pixel_quantizer_shifted(
        &self,
        extras_bits: u32,
        ec_type: crate::headers::extra_channels::ExtraChannelType,
        shift: u32,
    ) -> u32 {
        let dist = match ec_type {
            crate::headers::extra_channels::ExtraChannelType::Alpha => match self.alpha_distance {
                Some(d) if d > 0.0 => d,
                _ => return 1,
            },
            _ => return 1,
        };
        let dist = dist.clamp(0.01, 25.0);
        let ec_maxval = if extras_bits >= 32 {
            0u64
        } else {
            (1u64 << extras_bits) - 1
        };
        let bitdepth_correction = if ec_maxval == 0 {
            1.0
        } else {
            (ec_maxval as f32) / 255.0
        };
        // responsive=1: base quantizer is 0.25 (no `* 0.1` folded in).
        const QUANTIZER_RESPONSIVE: f32 = 0.25;
        const SQUEEZE_QUALITY_FACTOR: f32 = SQUEEZE_QUALITY_FACTOR_CONST;
        const SQUEEZE_LUMA_FACTOR: f32 = SQUEEZE_LUMA_FACTOR_CONST;
        let shift_idx = shift.min(SQUEEZE_LUMA_QTABLE_LEN as u32 - 1) as usize;
        let qtable_val = SQUEEZE_LUMA_QTABLE[shift_idx];
        let q_float = QUANTIZER_RESPONSIVE
            * dist
            * bitdepth_correction
            * SQUEEZE_QUALITY_FACTOR
            * SQUEEZE_LUMA_FACTOR
            * qtable_val;
        let q = q_float.floor() as i64;
        if q < 1 { 1 } else { q as u32 }
    }

    /// Whether the squeeze-on-extras lossy alpha pipeline is engaged
    /// for this encode (chunk-1 framework opt-in, not yet wired into
    /// the bitstream).
    ///
    /// Returns `true` only when **both** of the following hold:
    /// - [`Self::alpha_squeeze`] was set via
    ///   [`crate::LossyConfig::with_alpha_squeeze`].
    /// - [`Self::alpha_distance`] is `Some(d > 0.0)` (lossy alpha
    ///   path actually engaged).
    ///
    /// When this returns `true`, the extras writer is expected to
    /// route through the squeeze + per-band quantizer pipeline
    /// (libjxl `enc_modular.cc:937-1027`, `responsive=1`). Chunk 1
    /// surfaces an `Error::NotImplemented` from the writer to keep
    /// the wire format honest while the per-band quantizer routing
    /// lands.
    ///
    /// When this returns `false`, the existing no-squeeze
    /// [`Self::compute_extra_pixel_quantizer`] path is used — fully
    /// backwards-compatible (hash-lock byte-identical).
    pub(crate) fn alpha_squeeze_engaged(&self) -> bool {
        self.alpha_squeeze && matches!(self.alpha_distance, Some(d) if d > 0.0)
    }

    /// Encode an image in linear sRGB format, optionally with an alpha channel.
    ///
    /// Input should be 3 channels (RGB) of f32 values in [0, 1] range.
    /// Values outside [0, 1] are allowed for out-of-gamut colors.
    ///
    /// If `alpha` is provided, it must be `width * height` bytes of u8 alpha values.
    /// Alpha is encoded as a modular extra channel alongside the VarDCT RGB data.
    ///
    /// For more than alpha — depth, spot color, selection mask, thermal,
    /// CFA — see [`Self::encode_with_extras`].
    pub fn encode(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        alpha: Option<&[u8]>,
    ) -> Result<VarDctOutput> {
        match alpha {
            None => self.encode_with_extras(width, height, linear_rgb, &[]),
            Some(buf) => {
                // Build a default-alpha ExtraChannel view borrowing the caller's
                // buffer. `with_alpha` honours the encoder's `alpha_associated`
                // setting; the file-header builder forwards it.
                let mut info = crate::headers::extra_channels::ExtraChannelInfo::alpha();
                info.alpha_associated = self.alpha_associated;
                let ec = super::extras::VardctExtra {
                    info: &info,
                    data: super::extras::VardctExtraBuf::U8(buf),
                };
                // Inline the dimension check here so the error message
                // points at `alpha` not at an abstract "extras[0]".
                let expected_alpha = width.checked_mul(height).ok_or(Error::DimensionOverflow {
                    width,
                    height,
                    channels: 1,
                })?;
                if buf.len() != expected_alpha {
                    return Err(Error::InvalidInput(format!(
                        "alpha length {} != expected {}",
                        buf.len(),
                        expected_alpha
                    )));
                }
                self.encode_inner(width, height, linear_rgb, &[ec])
            }
        }
    }

    /// Encode RGB plus an arbitrary list of extra channels (alpha,
    /// depth, spot color, selection mask, thermal, CFA, …).
    ///
    /// Each [`crate::api::ExtraChannel`] carries its own
    /// [`crate::headers::extra_channels::ExtraChannelInfo`] which goes
    /// into the file-header metadata, plus the pixel buffer (u8 or u16).
    ///
    /// **Current scope (refs #9)**:
    /// - `dim_shift` must be `0` on every extra (full-resolution channels).
    /// - Single-group (≤256×256) supports any number of extras.
    /// - Multi-group supports any number of extras at `dim_shift = 0`.
    ///
    /// Other combinations return `Error::Unsupported` so the wire format
    /// stays correct as those paths are filled in.
    pub fn encode_with_extras(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        extras: &[crate::api::ExtraChannel<'_>],
    ) -> Result<VarDctOutput> {
        // Materialize the internal `VardctExtra` views, validating
        // dimensions + bit depth + dim_shift up-front before any work.
        let mut views: Vec<super::extras::VardctExtra<'_>> = Vec::with_capacity(extras.len());
        for (idx, ec) in extras.iter().enumerate() {
            if ec.info().dim_shift != 0 {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: dim_shift = {} not yet supported in lossy encode (dim_shift > 0 \
                     for VarDCT extras is a follow-up)",
                    ec.info().dim_shift
                )));
            }
            let expected = width.checked_mul(height).ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 1,
            })?;
            let got = ec.data().len();
            if got != expected {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: expected {expected} samples for {width}x{height}, got {got}"
                )));
            }
            views.push(super::extras::VardctExtra::from_api(ec));
        }

        self.check_alpha_squeeze_supported(&views, Some((width, height)))?;
        self.encode_inner(width, height, linear_rgb, &views)
    }

    /// Chunk-2.b gate (multi-group + dim_shift consolidation).
    ///
    /// When [`Self::alpha_squeeze`] is `true` and the lossy alpha
    /// pipeline is engaged (`alpha_distance > 0.0`) AND there is
    /// exactly one alpha extra, the squeeze pipeline is selected at
    /// the bitstream-emit site — for any image size (single-group
    /// and multi-group both supported as of chunk-2.b).
    ///
    /// Multi-extra (alpha + depth, alpha + spot, …) still returns
    /// `Error::NotImplemented`; that case needs per-coded-channel
    /// tree routing to interleave the squeezed alpha sub-channels
    /// with the non-alpha extras and stays queued (not in chunk-2.b
    /// scope).
    ///
    /// `dim_shift > 0` is rejected upstream in every lossy VarDCT
    /// entry point (`encode_with_extras`, `encode_from_precomputed_with_extras`,
    /// the pre-quantized variant — see `encoder.rs:927`, `2497`, `2901`)
    /// with `Error::InvalidInput`. The squeeze pipeline therefore
    /// inherits that rule rather than carrying its own redundant
    /// `dim_shift` check — the constraint is a property of VarDCT
    /// lossy extras, not of squeeze in particular. If the upstream
    /// `dim_shift > 0` gate ever moves (e.g. via libjxl-parity wiring
    /// of pre-subsampled extras), the squeeze pipeline's
    /// `build_alpha_squeeze_pipeline` already materializes the alpha
    /// channel at its native `width >> dim_shift × height >> dim_shift`
    /// resolution; the chunk-2.b writers would still need a Channel
    /// `hshift = dim_shift, vshift = dim_shift` seed to keep the
    /// decoder-side shift bracket classification consistent with
    /// `dec_modular.cc:354`.
    ///
    /// `dims = Some((w, h))` plumbed for future shape-aware gates
    /// (e.g. very wide / very tall edge cases); unused as of
    /// chunk-2.b — the writer handles arbitrary multi-group sizes.
    ///
    /// When alpha_squeeze is `false` (default), returns `Ok(())` and
    /// the existing extras pipeline runs unchanged — preserves the
    /// byte-for-byte hash-lock baseline (36/36 hashes match).
    fn check_alpha_squeeze_supported(
        &self,
        extras: &[super::extras::VardctExtra<'_>],
        _dims: Option<(usize, usize)>,
    ) -> Result<()> {
        if !self.alpha_squeeze_engaged() {
            return Ok(());
        }
        // Find the alpha extra index, if any.
        let alpha_idx = extras.iter().position(|ec| {
            matches!(
                ec.info.ec_type,
                crate::headers::extra_channels::ExtraChannelType::Alpha
            )
        });
        let Some(_alpha_idx) = alpha_idx else {
            // No alpha extra present → squeeze-on-extras is a no-op
            // even when the flag is set. Don't surprise the caller.
            return Ok(());
        };
        // Chunk-2 scope (still): exactly one extra (the alpha).
        // Multi-extra (alpha + depth, alpha + spot, …) needs per-
        // coded-channel tree routing to interleave the squeezed alpha
        // sub-channels with the non-alpha extras — not in chunk-2.b
        // scope.
        if extras.len() > 1 {
            return Err(Error::NotImplemented(
                "with_alpha_squeeze(true) currently supports only a single alpha extra. \
                 Multi-extra (alpha + depth, alpha + spot, …) routing through the per-band \
                 quantizer is queued. Leave alpha_squeeze at its default \
                 (false) for now to keep mixed-extras encoding working."
                    .into(),
            ));
        }
        // dim_shift > 0 is enforced upstream by every lossy VarDCT
        // entry-point validator with `Error::InvalidInput` (see
        // `encoder.rs:927`, `2497`, `2901`). The squeeze pipeline
        // inherits that rule unmodified; we deliberately don't shadow
        // it with a squeeze-flag-specific message because the
        // restriction isn't a property of the squeeze flag.
        Ok(())
    }

    /// Try to build the squeeze pipeline for the chunk-2 / chunk-2.b
    /// alpha path. Returns `Some(pipeline)` when
    /// [`Self::alpha_squeeze_engaged`] is `true` AND the extras shape
    /// matches the supported scope (single alpha extra). Returns
    /// `Ok(None)` for the flag-off path or shapes that fall through
    /// to the existing raw-pixel writer (no-alpha-extra case,
    /// multi-extra fallback).
    ///
    /// `width` and `height` are the full image dims. The pipeline is
    /// built once per image and partitioned by section at the
    /// bitstream-emit site (see
    /// [`super::extras::AlphaSqueezePipeline::partition`]).
    pub(crate) fn maybe_build_alpha_squeeze_pipeline(
        &self,
        extras: &[super::extras::VardctExtra<'_>],
        width: usize,
        height: usize,
    ) -> Result<Option<super::extras::AlphaSqueezePipeline>> {
        if !self.alpha_squeeze_engaged() {
            return Ok(None);
        }
        if extras.len() != 1 {
            return Ok(None);
        }
        let alpha = &extras[0];
        if !matches!(
            alpha.info.ec_type,
            crate::headers::extra_channels::ExtraChannelType::Alpha
        ) {
            return Ok(None);
        }
        // dim_shift > 0 is enforced upstream as `Error::InvalidInput`;
        // assert here so a future caller skipping the upstream
        // validator trips the bug visibly.
        debug_assert_eq!(
            alpha.info.dim_shift, 0,
            "alpha squeeze pipeline expects dim_shift=0 alpha (upstream \
             VarDCT lossy entry-point validator enforces this)"
        );
        // Chunk-3 heuristic: skip squeeze when alpha is a single
        // constant value over the full image. The W14-1 ChannelCompact
        // path (`e97e5bb7`, `write_modular_extras_subbitstream` →
        // `kPalette(num_c=1, nb_colors=1)`) collapses constant extras
        // to ~76 bytes regardless of `alpha_distance`. Routing those
        // through the squeeze pipeline costs +0.6 to +0.8% on the
        // `red_night_opaque` W16-2 audit baseline (`191801a1`) because
        // squeeze adds a GroupHeader + per-band tree leaves on top of
        // an already-minimal payload; every squeeze sub-channel is
        // itself constant so the residuals carry no extra information.
        //
        // ChannelCompact wins for constant alpha; squeeze wins for
        // varying alpha (W14-4 / chunk-2 / chunk-2.b shows -30% to
        // -56% on the photo-mask + UI-gradient audit images).
        if alpha.is_constant_full_image(width, height) {
            return Ok(None);
        }
        let bits = alpha.info.bit_depth.bits_per_sample;
        let shift0_q = self.compute_extra_pixel_quantizer_shifted(bits, alpha.info.ec_type, 0);
        let shifted_q = |shift: u32| -> u32 {
            self.compute_extra_pixel_quantizer_shifted(bits, alpha.info.ec_type, shift)
        };
        let pipeline =
            super::extras::build_alpha_squeeze_pipeline(alpha, width, height, shift0_q, shifted_q)?;
        Ok(Some(pipeline))
    }

    /// Shared implementation backing both [`Self::encode`] and
    /// [`Self::encode_with_extras`]. Takes already-validated extras as
    /// the internal `VardctExtra` view; performs the RGB-shape check
    /// then drives the full pipeline.
    fn encode_inner(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        extras: &[super::extras::VardctExtra<'_>],
    ) -> Result<VarDctOutput> {
        let expected_rgb = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 3,
            })?;
        if linear_rgb.len() != expected_rgb {
            return Err(Error::InvalidInput(format!(
                "linear_rgb length {} != expected {}",
                linear_rgb.len(),
                expected_rgb
            )));
        }

        crate::debug_rect::clear();

        // Calculate dimensions
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;

        // Number of sections: DC global + DC groups + AC global + AC groups
        let num_sections = 2 + num_dc_groups + num_groups;

        // Pad to block boundary dimensions
        let padded_width = xsize_blocks * BLOCK_DIM;
        let padded_height = ysize_blocks * BLOCK_DIM;

        // Validate linear-RGB at intake. The `forward_xyb` SIMD kernel
        // uses `mixed.max(0.0)` per channel, which silently coerces NaN
        // to `0.0` (IEEE-754 ordered max returns the non-NaN operand).
        // That means a caller-supplied NaN linear-RGB never reaches the
        // XYB output — the post-XYB check would not fire either. To
        // surface caller bugs (Error mode) or actively scrub them
        // (Sanitize mode), we must check / fix here, before forward_xyb
        // runs. For 8-bit / 16-bit pixel layouts the linear-RGB
        // conversion is total (no non-finite possible) and the
        // is_finite_plane scan is a fast read-only no-op (~55 GB/s).
        //
        // Sanitize mode used to skip the input check entirely, relying
        // on forward_xyb's silent NaN→0 max to mask non-finite values.
        // That left the advertised "SIMD scrub" never running on the
        // linear-RGB plane and made Error/Sanitize behavior diverge.
        // Now Sanitize actively rewrites non-finite values to 0.0
        // (~12.5 GB/s), then runs the rest of the pipeline on a clean
        // buffer; the downstream XYB scan stays as defense-in-depth.
        let sanitized_linear_rgb_storage: Option<alloc::vec::Vec<f32>> =
            match self.non_finite_action {
                crate::api::NonFiniteAction::Error => {
                    if !jxl_simd::is_finite_plane(linear_rgb) {
                        return Err(crate::error::Error::InvalidInput(
                            "non-finite (NaN / ±Inf) value detected in linear-RGB input. \
                             Use LossyConfig::with_non_finite_action(NonFiniteAction::Sanitize) \
                             to silently scrub non-finite values to 0.0 instead."
                                .into(),
                        ));
                    }
                    None
                }
                crate::api::NonFiniteAction::Sanitize => {
                    // sanitize_finite needs &mut, but the caller-supplied
                    // buffer is borrowed. Clone-and-sanitize when (and
                    // only when) Sanitize mode is selected. For 8-bit /
                    // 16-bit pixel layouts there are never non-finite
                    // values, so most callers stay on the Error fast path.
                    let mut owned: alloc::vec::Vec<f32> = linear_rgb.to_vec();
                    let _ = jxl_simd::sanitize_finite(&mut owned);
                    Some(owned)
                }
            };
        let linear_rgb: &[f32] = sanitized_linear_rgb_storage
            .as_deref()
            .unwrap_or(linear_rgb);

        // Convert to XYB with edge-replicated padding to block boundaries.
        // This allows SIMD to process full blocks without bounds checking.
        let (mut xyb_x, mut xyb_y, mut xyb_b) =
            self.convert_to_xyb_padded(width, height, padded_width, padded_height, linear_rgb)?;

        // Defense-in-depth XYB scan. Catches downstream-bug non-finite
        // (memory corruption, butteraugli-loop reconstruction polluting
        // XYB) — should never fire on the encode-fresh path because
        // forward_xyb is finite-output-for-finite-input.
        validate_xyb_planes(self.non_finite_action, &mut xyb_x, &mut xyb_y, &mut xyb_b)?;

        // Noise parameters. Four sources, in priority order
        // (matches libjxl enc_frame.cc:680-689):
        // 1. `photon_noise_iso`: caller-supplied ISO value, bypasses
        //    content estimation. Matches libjxl --photon_noise. Useful
        //    for re-encoding denoised content with controlled grain.
        // 2. `manual_noise_lut`: caller-supplied 8-point LUT. Bypasses
        //    everything else. Matches libjxl cparams.manual_noise.
        // 3. `enable_noise` + content estimation: scan flat patches,
        //    fit an 8-point LUT via SCG optimisation.
        // 4. None of the above: no noise synthesis.
        let noise_params = if let Some(iso) = self.photon_noise_iso
            && iso > 0.0
        {
            // The decoder regenerates noise during rendering from the
            // 10-bit-per-point LUT; the encoder just emits the LUT
            // here. No content estimation, no denoise pre-filter — we
            // *add* synthetic grain, not preserve real noise.
            let params = super::noise::simulate_photon_noise(width, height, iso);
            if params.has_any() { Some(params) } else { None }
        } else if let Some(lut) = self.manual_noise_lut {
            // Caller-supplied LUT — clamp to the encodable range
            // [0, NOISE_LUT_MAX ≈ 0.9995] so write_noise_params can't
            // panic on the 10-bit-quantise debug_assert. has_any()
            // gates trivial all-zero LUTs out.
            let mut params = super::noise::NoiseParams::default();
            for (dst, src) in params.lut.iter_mut().zip(lut.iter()) {
                *dst = src.clamp(0.0, 0.9995);
            }
            if params.has_any() { Some(params) } else { None }
        } else if self.enable_noise {
            let quality_coef = noise_quality_coef(self.distance);
            let params = estimate_noise_params(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                quality_coef,
            );

            // Apply denoising pre-filter if enabled and noise was detected.
            // Removes estimated noise before encoding so the encoder spends fewer
            // bits on noise; the decoder re-adds it from the encoded parameters.
            if self.enable_denoise
                && let Some(ref p) = params
            {
                denoise_xyb(
                    &mut xyb_x,
                    &mut xyb_y,
                    &mut xyb_b,
                    padded_width,
                    padded_height,
                    p,
                    quality_coef,
                );
            }

            params
        } else {
            None
        };

        // Detect and subtract patches (before gaborish, after noise).
        // Patches work in the XYB domain: detect repeated rectangular elements,
        // store unique patterns in a reference frame, subtract from image.
        //
        // Distance-aware kMinPeak: below d=1.0 we revert to libjxl's
        // `kMinPeak = 2` because the W2-5 chunk 1 relaxation (commit
        // 7b8c06e) admits low-magnitude text patches that do not
        // amortize their ref-frame overhead at low distance — measured
        // `windows95.png @ d=0.5` regressed by +465 bytes (+0.96 %)
        // before this gate. At d>=1.0 the chunk 1 relaxation pays off
        // (`windows95 @ d=1.0`: -53 B; `@ d=2.0`: -43 B), so above the
        // threshold we keep the looser detector.
        // Distance-aware kMinPeak (W3-1 / commit 4fb0f52): libjxl
        // parity (=2) below d=1.0, W2-5 chunk 1 relaxation (=1) at
        // d>=1.0. RFC#45 chunk 3 layered a per-patch cost gate on top
        // (inside `find_and_build_with_per_patch_gate`) but did NOT
        // change `min_peak`: attempting `min_peak=1` at d<1.0 brought
        // back the +465 B windows95 regression that W3-1 closed. The
        // per-patch gate is a refinement, not a replacement for the
        // detector-side `min_peak` threshold — it captures patches
        // already detected, not ones the detector rejected.
        let min_peak = if self.distance < 1.0 { 2 } else { 1 };
        let mut patches_data = if self.enable_patches {
            super::patches::find_and_build_with_per_patch_gate(
                [&xyb_x, &xyb_y, &xyb_b],
                width,
                height,
                padded_width,
                min_peak,
                Some(self.distance),
                self.use_ans,
            )
        } else {
            None
        };
        // Cost-benefit gating for experimental mode only.
        // libjxl uses patches unconditionally when detected (no cost check),
        // so reference mode skips this to match.
        if matches!(self.encoder_mode, crate::api::EncoderMode::Experimental)
            && let Some(ref pd) = patches_data
            && !pd.is_cost_effective(self.distance, self.use_ans)
        {
            patches_data = None;
        }
        // Quantize ref_image so subtract/add use the same values the decoder will reconstruct.
        if let Some(ref mut pd) = patches_data {
            pd.quantize_ref_image();
        }
        if let Some(ref pd) = patches_data {
            let mut xyb = [
                core::mem::take(&mut xyb_x),
                core::mem::take(&mut xyb_y),
                core::mem::take(&mut xyb_b),
            ];
            super::patches::subtract_patches(&mut xyb, padded_width, pd);
            let [x, y, b] = xyb;
            xyb_x = x;
            xyb_y = y;
            xyb_b = b;
        }

        // Dot detection (refs #19). libjxl gates at speed_tier <= kSquirrel
        // (effort >= 7) AND distance >= 3.0, AND only when text-like
        // patches haven't been found. We mirror the gating exactly.
        // Feature is off by default — niche (astronomy / specular highlights);
        // enable via `LossyConfig::with_dot_detection(true)`.
        if self.enable_dot_detection
            && self.effort >= 7
            && self.distance >= 3.0
            && patches_data.is_none()
        {
            let dots = super::dot_detection::detect_gaussian_ellipses(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                &super::dot_detection::GaussianDetectParams::default(),
            );
            #[cfg(feature = "debug-tokens")]
            crate::debug_log!("dot detection: found {} candidate dots", dots.len());
            // Promote the detected dots into a fresh PatchesData. The
            // subsequent quantize_ref_image + subtract_patches block
            // below treats them identically to text-like patches.
            if !dots.is_empty()
                && let Some(mut pd) = super::patches::PatchesData::from_dots(&dots)
            {
                pd.quantize_ref_image();
                let mut xyb = [
                    core::mem::take(&mut xyb_x),
                    core::mem::take(&mut xyb_y),
                    core::mem::take(&mut xyb_b),
                ];
                super::patches::subtract_patches(&mut xyb, padded_width, &pd);
                let [x, y, b] = xyb;
                xyb_x = x;
                xyb_y = y;
                xyb_b = b;
                patches_data = Some(pd);
            }
        }

        // Build and subtract splines (after patches, before gaborish).
        // Splines are additive overlays: encoder subtracts, decoder adds back.
        // Uses default DC CfL params (y_to_x=0.0, y_to_b=1.0) since we write default DC cmap.
        //
        // Resolution order, mirroring libjxl `enc_heuristics.cc:1044-1055`:
        //   1. Manual non-empty `self.splines` always wins (libjxl
        //      `cparams.custom_splines.HasAny()`).
        //   2. Else, when `self.auto_splines` is set AND effort ≥ 7
        //      (libjxl `speed_tier <= kSquirrel`), call `find_splines()`
        //      on the post-patches XYB planes.
        //   3. Else, no splines.
        //
        // Chunk 1: `find_splines` is a stub that returns `vec![]`,
        // matching libjxl's `enc_splines.cc:104-107` TODO. The empty
        // branch below short-circuits to `None`, leaving every
        // default-config encode byte-identical.
        let effective_splines = if let Some(ref splines) = self.splines {
            // Manual override always wins, even if empty (caller may
            // explicitly disable by passing `vec![]`).
            Some(splines.clone())
        } else if self.auto_splines && self.effort >= 7 {
            // Chunk-5 content discriminator: skip auto-splines detection
            // on screenshot-like content. The bbox-area-linear
            // energy-drop proxy used by the per-spline cost-benefit gate
            // structurally over-claims VarDCT byte savings on long
            // bright ridges (table borders, wallpaper edges), regressing
            // real encodes by ~3% on `codec_wiki.png` and `imac_g3.png`
            // — see chunk-4 bench notes (`benchmarks/
            // auto_splines_bench_2026-05-17_chunk4.meta`) and the
            // `effort.rs::auto_splines_default` doc-comment for the
            // structural-proxy explanation.
            //
            // Uses the same `median(mask1x1) > 95.0` discriminator that
            // the GPU encoder's W7-3 AFV cost-grid gate uses
            // (`jxl-encoder-gpu/src/lossy_encoder.rs::
            // SCREENSHOT_MEDIAN_MASK_THRESHOLD`). On the chunk-5 corpus
            // bench (`benchmarks/auto_splines_bench_2026-05-17_chunk5.tsv`)
            // this correctly classifies every screenshot (`terminal`,
            // `codec_wiki`, `imac_g3`) and every flat-background line
            // synthetic as screenshot-class, while admitting all 5
            // CLIC2025-1024 photos (photo median ~56 vs screenshot/synth
            // median 100).
            if super::splines::looks_like_screenshot(&xyb_y, width, height, padded_width) {
                None
            } else {
                // Pass distance through so the per-spline cost-benefit gate
                // can scale the "VarDCT bytes saved per spline pixel"
                // estimate. The gate model assumes ~5 bits/pixel of AC
                // residual at d=1.0 and clamps the divisor at 1.0 — see
                // `spline_passes_cost_gate` in `vardct/splines.rs`.
                Some(super::splines::find_splines_at_distance(
                    &xyb_x,
                    &xyb_y,
                    &xyb_b,
                    width,
                    height,
                    padded_width,
                    self.distance,
                ))
            }
        } else {
            None
        };
        let splines_data = if let Some(ref splines) = effective_splines {
            if !splines.is_empty() {
                let sd = super::splines::SplinesData::from_splines(
                    splines.clone(),
                    0,   // quantization_adjustment
                    0.0, // y_to_x (default DC CfL)
                    1.0, // y_to_b (default DC CfL)
                    width,
                    height,
                );
                {
                    let mut xyb = [
                        core::mem::take(&mut xyb_x),
                        core::mem::take(&mut xyb_y),
                        core::mem::take(&mut xyb_b),
                    ];
                    super::splines::subtract_splines(&mut xyb, padded_width, width, height, &sd);
                    let [x, y, b] = xyb;
                    xyb_x = x;
                    xyb_y = y;
                    xyb_b = b;
                }
                Some(sd)
            } else {
                None
            }
        } else {
            None
        };

        // Compute pixel chromacity stats BEFORE gaborish (matching libjxl pipeline).
        // Gaborish sharpening inflates gradients, producing overly aggressive adjustment.
        // Gated at effort >= 7 to skip the full-image gradient scan at low effort.
        let (chromacity_x, chromacity_b) = if self.profile.chromacity_adjustment {
            let pixel_stats = super::frame::PixelStatsForChromacityAdjustment::calc(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
            );
            (
                pixel_stats.how_much_is_x_channel_pixelized(),
                pixel_stats.how_much_is_b_channel_pixelized(),
            )
        } else {
            (0, 0)
        };

        // Compute adaptive per-block quantization field and masking on ORIGINAL
        // (pre-gaborish) XYB. libjxl computes InitialQuantField before GaborishInverse
        // (enc_heuristics.cc:1117-1142, comment: "relies on pre-gaborish values").
        // When gaborish is off, scale distance by 0.62 for the quant field only
        // (not global_scale/quant_dc). This matches libjxl enc_heuristics.cc:1119.
        let distance_for_iqf = if self.enable_gaborish {
            self.distance
        } else {
            self.distance * 0.62
        };

        // Step 1: Compute float quant field on pre-gaborish XYB.
        //
        // libjxl effort gating (enc_heuristics.cc:1097-1128):
        // - effort < 5 (speed_tier > kHare): flat quant field = q_numerator/distance
        // - effort >= 5 (speed_tier <= kHare): adaptive via InitialQuantField
        let (mut quant_field_float, masking) = if self.profile.use_adaptive_quant {
            super::adaptive_quant::compute_quant_field_float_with_budget(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                distance_for_iqf,
                self.profile.k_ac_quant,
                self.budget.as_ref(),
            )?
        } else {
            // Flat quant field for low effort (matches libjxl enc_heuristics.cc:1105-1106).
            // Account both nblocks-sized f32 buffers against the budget.
            let nblocks = xsize_blocks.checked_mul(ysize_blocks).ok_or(
                crate::error::Error::DimensionOverflow {
                    width: xsize_blocks,
                    height: ysize_blocks,
                    channels: 1,
                },
            )?;
            crate::budget::MemoryBudget::reserve_permanent_opt(
                self.budget.as_ref(),
                (nblocks as u64).saturating_mul(4 * 2),
            )?;
            let q = self.profile.initial_q_numerator / self.distance;
            let flat_qf = vec![q; nblocks];
            let masking_val = 1.0 / (q + 0.001);
            let flat_masking = vec![masking_val; nblocks];
            (flat_qf, flat_masking)
        };

        // Step 2: Compute distance params with effort-matched global_scale.
        //
        // Uses profile.initial_q_numerator for q = numerator / distance.
        // The adaptive median/MAD formula is only used inside the butteraugli
        // loop (effort >= 8).
        let mut params = match self.original_distance {
            Some(orig) if orig > self.distance => {
                DistanceParams::compute_for_profile_with_original(
                    self.distance,
                    orig,
                    &self.profile,
                )
            }
            _ => DistanceParams::compute_for_profile(self.distance, &self.profile),
        };
        if let Some(rescale) = self.quant_ac_rescale {
            params.apply_quant_ac_rescale(rescale);
        }
        // libjxl --epf override: when the caller pinned a level, it wins
        // over the distance-derived `epf_iters` (enc_frame.cc:284-285).
        self.apply_epf_level_override(&mut params);

        // Apply pixel-level chromacity adjustments using pre-gaborish stats
        // Gated at effort >= 7 (speed_tier <= kSquirrel) matching libjxl
        if self.profile.chromacity_adjustment {
            params.apply_chromacity_adjustment(chromacity_x, chromacity_b);
        }

        debug_rect!(
            "enc/params",
            0,
            0,
            width,
            height,
            "global_scale={} quant_dc={} scale={:.4} inv_scale={:.4} epf_iters={} chrom_x={:.3} chrom_b={:.3}",
            params.global_scale,
            params.quant_dc,
            params.scale,
            params.inv_scale,
            params.epf_iters,
            chromacity_x,
            chromacity_b
        );

        // Step 3: Quantize float quant field to raw u8 with adaptive inv_scale
        let mut quant_field = quantize_quant_field(&quant_field_float, params.inv_scale);

        // Compute per-pixel mask on PRE-GABORISH image (matches libjxl:
        // initial_quant_masking1x1 is computed in InitialQuantField before GaborishInverse)
        let mask1x1 = if self.ac_strategy_enabled && self.pixel_domain_loss {
            Some(super::adaptive_quant::compute_mask1x1_with_budget(
                &xyb_y,
                padded_width,
                padded_height,
                self.budget.as_ref(),
            )?)
        } else {
            None
        };

        // Apply gaborish inverse (5x5 sharpening) AFTER quant field and mask1x1
        // but BEFORE CfL and AC strategy. This matches libjxl enc_heuristics.cc:
        //   line 1124: InitialQuantField (pre-gaborish)
        //   line 1142: GaborishInverse
        //   line 1150-1174: CfL (post-gaborish)
        //   line 1179: AC strategy (post-gaborish)
        if self.enable_gaborish {
            gaborish_inverse(
                &mut xyb_x,
                &mut xyb_y,
                &mut xyb_b,
                padded_width,
                padded_height,
                self.budget.as_ref(),
            )?;
        }

        // Float DC for LfFrame is now extracted from the transform pipeline
        // (TransformOutput.float_dc) using dc_from_dct_NxN, which produces correct
        // DC values for multi-block transforms (DCT16+). The old compute_float_dc
        // used simple 8x8 pixel averages which diverge from dc_from_dct_NxN for
        // blocks with spatial structure, causing catastrophic LfFrame quality for
        // DCT16+ (up to 31% error on gradient content, butteraugli 13-20 vs ~2.5).

        // Compute per-tile chroma-from-luma map on GABORISHED XYB
        // Pass 1 always uses LS (use_newton=false): with distance_mul=1e-9, the
        // perceptual cost function collapses to LS, so Newton adds no value.
        // Newton is only useful in pass 2 where actual quant weighting matters.
        let mut cfl_map = if self.cfl_enabled {
            compute_cfl_map(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                false,
                self.profile.cfl_newton_eps,
                self.profile.cfl_newton_max_iters,
            )
        } else {
            CflMap::zeros(
                div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS),
                div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS),
            )
        };

        debug_rect!(
            "enc/config",
            0,
            0,
            width,
            height,
            "d={:.2} gab={} cfl={} pixel_loss={} patches={} bfly_iters={} noise={} denoise={} ac_strat={} err_diff={}",
            self.distance,
            self.enable_gaborish,
            self.cfl_enabled,
            self.pixel_domain_loss,
            self.enable_patches,
            self.profile.butteraugli_iters,
            self.enable_noise,
            self.enable_denoise,
            self.ac_strategy_enabled,
            self.error_diffusion
        );

        // Compute adaptive AC strategy (DCT8/DCT16x8/DCT8x16/DCT16x16/DCT32x32)
        #[allow(unused_mut)]
        let mut ac_strategy = if let Some(forced) = self.force_strategy {
            // Force a specific strategy for all blocks that fit
            force_strategy_map(xsize_blocks, ysize_blocks, forced)
        } else if !self.ac_strategy_enabled {
            AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks)
        } else {
            compute_ac_strategy(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                self.distance,
                &quant_field_float,
                &masking,
                &cfl_map,
                mask1x1.as_deref(),
                padded_width,
                &self.profile,
            )
        };

        // Debug: print strategy histogram if enabled
        #[cfg(feature = "debug-ac-strategy")]
        {
            eprintln!(
                "AC strategy mode: {}",
                if mask1x1.is_some() {
                    "pixel-domain"
                } else {
                    "coefficient-domain"
                }
            );
            ac_strategy.print_histogram();
        }

        // Log AC strategy distribution
        {
            let mut counts = [0u32; 27];
            for by in 0..ysize_blocks {
                for bx in 0..xsize_blocks {
                    if ac_strategy.is_first(bx, by) {
                        let s = ac_strategy.raw_strategy(bx, by) as usize;
                        if s < counts.len() {
                            counts[s] += 1;
                        }
                    }
                }
            }
            let total: u32 = counts.iter().sum();
            // Format top strategies
            // Names indexed by RAW_STRATEGY_* internal codes (NOT bitstream order)
            let names = [
                "DCT8",     // 0 = RAW_STRATEGY_DCT8
                "DCT16x8",  // 1 = RAW_STRATEGY_DCT16X8
                "DCT8x16",  // 2 = RAW_STRATEGY_DCT8X16
                "DCT16x16", // 3 = RAW_STRATEGY_DCT16X16
                "DCT32x32", // 4 = RAW_STRATEGY_DCT32X32
                "DCT4x8",   // 5 = RAW_STRATEGY_DCT4X8
                "DCT8x4",   // 6 = RAW_STRATEGY_DCT8X4
                "DCT4x4",   // 7 = RAW_STRATEGY_DCT4X4
                "IDENTITY", // 8 = RAW_STRATEGY_IDENTITY
                "DCT2x2",   // 9 = RAW_STRATEGY_DCT2X2
                "DCT32x16", // 10 = RAW_STRATEGY_DCT32X16
                "DCT16x32", // 11 = RAW_STRATEGY_DCT16X32
                "AFV0",     // 12 = RAW_STRATEGY_AFV0
                "AFV1",     // 13 = RAW_STRATEGY_AFV1
                "AFV2",     // 14 = RAW_STRATEGY_AFV2
                "AFV3",     // 15 = RAW_STRATEGY_AFV3
                "DCT64x64", // 16 = RAW_STRATEGY_DCT64X64
                "DCT64x32", // 17 = RAW_STRATEGY_DCT64X32
                "DCT32x64", // 18 = RAW_STRATEGY_DCT32X64
            ];
            let mut parts = alloc::string::String::new();
            for (i, &c) in counts.iter().enumerate() {
                if c > 0 {
                    if !parts.is_empty() {
                        parts.push(' ');
                    }
                    let name = names.get(i).copied().unwrap_or("?");
                    let pct = c as f32 / total.max(1) as f32 * 100.0;
                    parts.push_str(&alloc::format!("{}={:.0}%", name, pct));
                }
            }
            debug_rect!(
                "enc/ac_strategy",
                0,
                0,
                width,
                height,
                "total={} {}",
                total,
                parts
            );
        }

        // Free masking — no longer needed after AC strategy selection.
        drop(masking);

        // Adjust quant field for multi-block transforms.
        // At low distances uses max, at high distances blends toward mean for better quality.
        // Adjust BOTH u8 and float fields (libjxl adjusts float before SetQuantField).
        adjust_quant_field_with_distance(&ac_strategy, &mut quant_field, self.distance);
        adjust_quant_field_float_with_distance(&ac_strategy, &mut quant_field_float, self.distance);

        // CfL pass 2: recompute CfL map using actual AC strategies and per-block
        // quantization weighting. Uses the same FindBestMultiplier as pass 1 but
        // with strategy-specific DCTs and quant-weighted coefficients.
        // Gated at effort >= 7 (speed_tier <= kSquirrel) matching libjxl.
        //
        // ORDERING: CfL pass 2 must run BEFORE the butteraugli loop so the
        // loop's internal recon and the shipped bitstream both see the same
        // post-pass-2 cfl_map. Mirrors libjxl enc_heuristics.cc:1190-1193 (CfL2)
        // → :1250-1252 (FindBestQuantizer/buttloop). See drift investigation
        // 2026-05-15 (chunk-3) — running CfL pass 2 AFTER the buttloop caused
        // the buttloop to converge on a target the decoder never delivered.
        if self.profile.cfl_two_pass && self.cfl_enabled {
            super::chroma_from_luma::refine_cfl_map(
                &mut cfl_map,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                &ac_strategy,
                &quant_field,
                params.scale,
                self.profile.cfl_newton,
                self.profile.cfl_newton_eps,
                self.profile.cfl_newton_max_iters,
            );
        }

        // Quantization loops: iteratively refine quant_field using perceptual
        // distance feedback. Butteraugli and zensim loops can stack: butteraugli
        // handles global convergence, zensim adds SSIM-aware spatial fine-tuning.
        // Works in float quant field domain with per-iteration global_scale
        // recomputation (matching libjxl FindBestQuantization).
        #[cfg(feature = "butteraugli-loop")]
        if self.butteraugli_iters > 0 {
            let initial_qf_float = quant_field_float.clone();
            params = self.butteraugli_refine_quant_field(
                linear_rgb,
                width,
                height,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                &params,
                &mut quant_field,
                &mut quant_field_float,
                &initial_qf_float,
                &cfl_map,
                &ac_strategy,
                patches_data.as_ref(),
                splines_data.as_ref(),
            )?;
        }

        // SSIM2 quantization loop: alternative to butteraugli using SSIM2 + per-block RMSE.
        #[cfg(feature = "ssim2-loop")]
        if self.ssim2_iters > 0 {
            let initial_qf_float = quant_field_float.clone();
            params = self.ssim2_refine_quant_field(
                linear_rgb,
                width,
                height,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                &params,
                &mut quant_field,
                &mut quant_field_float,
                &initial_qf_float,
                &cfl_map,
                &ac_strategy,
                patches_data.as_ref(),
                splines_data.as_ref(),
            )?;
        }

        // Zensim quantization loop: uses zensim psychovisual metric + per-pixel diffmap.
        // Also refines AC strategy by splitting large transforms with high perceptual error.
        #[cfg(feature = "zensim-loop")]
        if self.zensim_iters > 0 {
            let initial_qf_float = quant_field_float.clone();
            params = self.zensim_refine_quant_field(
                linear_rgb,
                width,
                height,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                &params,
                &mut quant_field,
                &mut quant_field_float,
                &initial_qf_float,
                &cfl_map,
                &mut ac_strategy,
                patches_data.as_ref(),
                splines_data.as_ref(),
            )?;
        }

        // Free float quant field — no longer needed after loop refinement.
        drop(quant_field_float);

        // Log quant field statistics after all adjustments
        {
            let qf = &quant_field;
            let sum: u64 = qf.iter().map(|&v| v as u64).sum();
            let avg = sum as f32 / qf.len() as f32;
            let min = qf.iter().copied().min().unwrap_or(0);
            let max = qf.iter().copied().max().unwrap_or(0);
            debug_rect!(
                "enc/quant_field",
                0,
                0,
                width,
                height,
                "final avg={:.1} min={} max={} blocks={}",
                avg,
                min,
                max,
                qf.len()
            );
        }

        // Dump AC strategy and quant field maps for comparison with libjxl.
        // Set JXL_DUMP_MAPS=/tmp/prefix to enable. Maps are written as CSV.
        #[cfg(feature = "debug-rect")]
        if let Ok(prefix) = std::env::var("JXL_DUMP_MAPS") {
            use std::io::Write;
            // AC strategy map
            if let Ok(mut f) = std::fs::File::create(format!("{prefix}_acs.csv")) {
                for by in 0..ysize_blocks {
                    for bx in 0..xsize_blocks {
                        if bx > 0 {
                            let _ = write!(f, ",");
                        }
                        let _ = write!(f, "{}", ac_strategy.raw_strategy(bx, by));
                    }
                    let _ = writeln!(f);
                }
                eprintln!("DIAG: wrote {prefix}_acs.csv ({xsize_blocks}x{ysize_blocks})");
            }
            // Quant field map
            if let Ok(mut f) = std::fs::File::create(format!("{prefix}_qf.csv")) {
                for by in 0..ysize_blocks {
                    for bx in 0..xsize_blocks {
                        if bx > 0 {
                            let _ = write!(f, ",");
                        }
                        let _ = write!(f, "{}", quant_field[by * xsize_blocks + bx]);
                    }
                    let _ = writeln!(f);
                }
                eprintln!("DIAG: wrote {prefix}_qf.csv ({xsize_blocks}x{ysize_blocks})");
            }
        }

        // CfL pass 2 was moved up to BEFORE the butteraugli loop (drift-fix
        // chunk-3, 2026-05-15). See the moved block above and the comment
        // there for ordering rationale.

        // Perform DCT and quantization (XYB data is padded to block boundaries)
        let transform_out = self.transform_and_quantize(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            xsize_blocks,
            ysize_blocks,
            &params,
            &mut quant_field,
            &cfl_map,
            &ac_strategy,
        )?;
        let quant_dc = &transform_out.quant_dc;
        let quant_ac = &transform_out.quant_ac;
        let nzeros = &transform_out.nzeros;
        let raw_nzeros = &transform_out.raw_nzeros;

        // Compute per-block EPF sharpness map when EPF is active
        // Dynamic sharpness gated at effort >= 6 (speed_tier <= kWombat) matching libjxl
        let sharpness_map =
            if params.epf_iters > 0 && self.distance >= 0.5 && self.profile.epf_dynamic_sharpness {
                let mask_fallback;
                let mask: &[f32] = match &mask1x1 {
                    Some(m) => m,
                    None => {
                        mask_fallback = super::adaptive_quant::compute_mask1x1_with_budget(
                            &xyb_y,
                            padded_width,
                            padded_height,
                            self.budget.as_ref(),
                        )?;
                        &mask_fallback
                    }
                };
                Some(super::epf::compute_epf_sharpness(
                    [&xyb_x, &xyb_y, &xyb_b],
                    quant_dc,
                    quant_ac,
                    &quant_field,
                    mask,
                    &params,
                    &cfl_map,
                    &ac_strategy,
                    self.enable_gaborish,
                    xsize_blocks,
                    ysize_blocks,
                    self.budget.as_ref(),
                )?)
            } else {
                None
            };

        // Free XYB planes — no longer needed after EPF sharpness computation.
        // At 4K (6720×4480), this frees ~339 MB (3 channels × padded_pixels × f32).
        drop(xyb_x);
        drop(xyb_y);
        drop(xyb_b);
        // Free mask1x1 — up to ~115 MB at 4K (padded_pixels × f32).
        drop(mask1x1);

        // Two-pass mode: collect tokens, build optimal codes, write bitstream
        if self.optimize_codes {
            let strategy_counts = ac_strategy.strategy_histogram();
            let data = self.encode_two_pass(
                width,
                height,
                &params,
                xsize_blocks,
                ysize_blocks,
                xsize_groups,
                ysize_groups,
                xsize_dc_groups,
                ysize_dc_groups,
                num_groups,
                num_dc_groups,
                num_sections,
                quant_dc,
                quant_ac,
                nzeros,
                raw_nzeros,
                &quant_field,
                &cfl_map,
                &ac_strategy,
                &noise_params,
                sharpness_map.as_deref(),
                extras,
                patches_data.as_ref(),
                splines_data.as_ref(),
                if self.use_lf_frame {
                    Some(&transform_out.float_dc)
                } else {
                    None
                },
            )?;
            crate::debug_rect::flush("");
            return Ok(VarDctOutput {
                data,
                strategy_counts,
            });
        }

        // Get static entropy codes (wrapped in BuiltEntropyCode for uniform handling)
        let dc_code = BuiltEntropyCode::StaticHuffman(get_dc_entropy_code());
        let ac_code = BuiltEntropyCode::StaticHuffman(get_ac_entropy_code());

        // Create main writer. The capacity is a heuristic upper bound on the
        // bitstream size — actual usage is much smaller in compression but
        // we budget the full reservation since BitWriter eagerly allocates.
        let main_cap = (width as u64)
            .saturating_mul(height as u64)
            .saturating_mul(4);
        crate::budget::MemoryBudget::reserve_permanent_opt(self.budget.as_ref(), main_cap)?;
        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write file header (includes JXL signature, ICC, and byte padding).
        // The streaming/one-pass static-Huffman path doesn't carry any
        // extras — they require the two-pass dynamic-entropy plumbing.
        self.write_file_header_and_pad(width, height, &[], &mut writer)?;
        #[cfg(feature = "debug-tokens")]
        debug_log!(
            "After file header: bit {} (byte {})",
            writer.bits_written(),
            writer.bits_written() / 8
        );

        // Write frame header
        {
            let mut fh = FrameHeader::lossy();
            fh.x_qm_scale = params.x_qm_scale;
            fh.b_qm_scale = params.b_qm_scale;
            fh.epf_iters = params.epf_iters;
            fh.gaborish = self.enable_gaborish;
            fh.upsampling = self.upsampling;
            if noise_params.is_some() {
                fh.flags |= 0x01; // ENABLE_NOISE
            }
            // streaming path: no extra channels
            fh.write(&mut writer)?;
        }
        #[cfg(feature = "debug-tokens")]
        debug_log!(
            "After frame header: bit {} (byte {})",
            writer.bits_written(),
            writer.bits_written() / 8
        );

        // For single-group images, combine all sections at the bit level
        // (no byte padding between sections, only at the end)
        if num_sections == 4 {
            // Write sections to individual BitWriters (no padding)
            let block_ctx_map = super::ac_context::BlockCtxMap::default();
            let num_blocks = xsize_blocks * ysize_blocks;
            let mut dc_global = BitWriter::with_capacity(4096);
            self.write_dc_global(
                &params,
                num_dc_groups,
                &dc_code,
                &noise_params,
                None,
                &block_ctx_map,
                None, // No learned tree in single-pass mode
                None, // No patches in streaming mode
                None, // No splines in streaming mode
                None, // No custom dc_quant in single-pass mode
                &mut dc_global,
            )?;

            // Get borrowed Huffman codes for streaming token writing
            let dc_huffman = dc_code.as_huffman();
            let ac_huffman = ac_code.as_huffman();

            // dc_group + ac_group BitWriters are sized proportional to the
            // image (10 bytes/block DC + 100 bytes/block AC heuristic).
            // Account both up front against the budget.
            crate::budget::MemoryBudget::reserve_permanent_opt(
                self.budget.as_ref(),
                (num_blocks as u64).saturating_mul(110),
            )?;
            let mut dc_group = BitWriter::with_capacity(num_blocks * 10);
            self.write_dc_group(
                0,
                quant_dc,
                xsize_blocks,
                ysize_blocks,
                xsize_dc_groups,
                &quant_field,
                &cfl_map,
                &ac_strategy,
                None, // no sharpness map in single-pass mode
                &dc_huffman,
                &mut dc_group,
            )?;

            let mut ac_global = BitWriter::with_capacity(4096);
            self.write_ac_global(
                num_groups,
                core::slice::from_ref(&ac_code),
                0,
                None,
                &[None],
                &mut ac_global,
            )?;

            let mut ac_group_writer = BitWriter::with_capacity(num_blocks * 100);
            self.write_ac_group(
                0,
                quant_ac,
                nzeros,
                raw_nzeros,
                xsize_blocks,
                ysize_blocks,
                xsize_groups,
                &quant_field,
                &ac_strategy,
                &block_ctx_map,
                &ac_huffman,
                &mut ac_group_writer,
            )?;

            #[cfg(feature = "debug-tokens")]
            {
                debug_log!(
                    "Section bit counts: DC_global={}, DC_group={}, AC_global={}, AC_group={}",
                    dc_global.bits_written(),
                    dc_group.bits_written(),
                    ac_global.bits_written(),
                    ac_group_writer.bits_written()
                );
            }

            // Combine at bit level
            let mut combined = dc_global;
            #[cfg(feature = "debug-tokens")]
            debug_log!("After DC_global: {} bits", combined.bits_written());
            combined.append_unaligned(&dc_group)?;
            #[cfg(feature = "debug-tokens")]
            debug_log!("After DC_group: {} bits", combined.bits_written());
            combined.append_unaligned(&ac_global)?;
            #[cfg(feature = "debug-tokens")]
            debug_log!("After AC_global: {} bits", combined.bits_written());
            combined.append_unaligned(&ac_group_writer)?;
            #[cfg(feature = "debug-tokens")]
            debug_log!("After AC_group: {} bits", combined.bits_written());
            combined.zero_pad_to_byte();
            let combined_bytes = combined.finish();

            #[cfg(feature = "debug-tokens")]
            {
                debug_log!("Combined section size: {} bytes", combined_bytes.len());
                debug_log!(
                    "Before TOC: bit {} (byte {})",
                    writer.bits_written(),
                    writer.bits_written() / 8
                );
            }
            write_toc(&[combined_bytes.len()], &mut writer)?;
            #[cfg(feature = "debug-tokens")]
            debug_log!(
                "After TOC: bit {} (byte {})",
                writer.bits_written(),
                writer.bits_written() / 8
            );
            writer.append_bytes(&combined_bytes)?;
        } else {
            // Multi-group: use byte-aligned sections.
            // Section bytes accumulate across groups; the heuristic capacity
            // is num_groups * blocks_per_group * 100 + num_dc_groups * 10240
            // ≈ num_blocks * 100 in total. Account up front.
            let total_groups_bytes = (xsize_blocks as u64)
                .saturating_mul(ysize_blocks as u64)
                .saturating_mul(110);
            crate::budget::MemoryBudget::reserve_permanent_opt(
                self.budget.as_ref(),
                total_groups_bytes,
            )?;
            let mut sections: Vec<Vec<u8>> = Vec::with_capacity(num_sections);
            let dc_huffman = dc_code.as_huffman();
            let ac_huffman = ac_code.as_huffman();

            // DC Global section
            let block_ctx_map = super::ac_context::BlockCtxMap::default();
            let mut dc_global = BitWriter::with_capacity(4096);
            self.write_dc_global(
                &params,
                num_dc_groups,
                &dc_code,
                &noise_params,
                None,
                &block_ctx_map,
                None, // No learned tree in single-pass mode
                None, // No patches in streaming mode
                None, // No splines in streaming mode
                None, // No custom dc_quant in single-pass mode
                &mut dc_global,
            )?;
            dc_global.zero_pad_to_byte();
            sections.push(dc_global.finish());

            // DC group sections
            let blocks_per_dc_group = (256 / 8) * (256 / 8); // 1024 blocks per DC group
            for dc_group_idx in 0..num_dc_groups {
                let mut dc_group = BitWriter::with_capacity(blocks_per_dc_group * 10);
                self.write_dc_group(
                    dc_group_idx,
                    quant_dc,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_dc_groups,
                    &quant_field,
                    &cfl_map,
                    &ac_strategy,
                    None, // no sharpness map in single-pass mode
                    &dc_huffman,
                    &mut dc_group,
                )?;
                dc_group.zero_pad_to_byte();
                sections.push(dc_group.finish());
            }

            // AC Global section
            let mut ac_global = BitWriter::with_capacity(4096);
            self.write_ac_global(
                num_groups,
                core::slice::from_ref(&ac_code),
                0,
                None,
                &[None],
                &mut ac_global,
            )?;
            ac_global.zero_pad_to_byte();
            sections.push(ac_global.finish());

            // AC group sections
            let blocks_per_ac_group = (256 / 8) * (256 / 8); // 1024 blocks per AC group
            for group_idx in 0..num_groups {
                let mut ac_group_writer = BitWriter::with_capacity(blocks_per_ac_group * 100);
                self.write_ac_group(
                    group_idx,
                    quant_ac,
                    nzeros,
                    raw_nzeros,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_groups,
                    &quant_field,
                    &ac_strategy,
                    &block_ctx_map,
                    &ac_huffman,
                    &mut ac_group_writer,
                )?;
                ac_group_writer.zero_pad_to_byte();
                sections.push(ac_group_writer.finish());
            }

            // Center-first AC group reordering (closes #14). Identity
            // prefix for [DC global, DC groups..., AC global], then
            // AC groups permuted by concentric-square distance from
            // image center. libjxl-faithful PermuteGroups algorithm.
            //
            // Single-pass only — multi-pass progressive interaction
            // is a future extension. num_groups <= 1 → no-op (nothing
            // to reorder).
            if self.center_first && num_groups > 1 {
                use crate::vardct::coeff_order::compute_center_first_ac_permutation;
                // Caller-supplied center_x / center_y (libjxl
                // `cparams.center_x` / `center_y`); fall back to image
                // centre when unset, matching libjxl's
                // `size_t(-1) → width/2` behaviour at enc_frame.cc.
                let cx = self
                    .center_x
                    .map(|x| x.min(width.saturating_sub(1) as u32))
                    .unwrap_or((width as u32) / 2);
                let cy = self
                    .center_y
                    .map(|y| y.min(height.saturating_sub(1) as u32))
                    .unwrap_or((height as u32) / 2);
                let ac_group_order =
                    compute_center_first_ac_permutation(xsize_groups, ysize_groups, cx, cy);
                // Build inverse mapping: inv[orig_idx] = on_disk_pos.
                let mut inv_ac = vec![0u32; num_groups];
                for (on_disk_pos, &orig_idx) in ac_group_order.iter().enumerate() {
                    inv_ac[orig_idx as usize] = on_disk_pos as u32;
                }
                // libjxl permutation array: identity prefix +
                // inv_ac_group_order offset by prefix length.
                let prefix_len = 2 + num_dc_groups;
                let total = prefix_len + num_groups;
                let mut permutation = Vec::with_capacity(total);
                for i in 0..prefix_len {
                    permutation.push(i as u32);
                }
                let prefix_u32 = prefix_len as u32;
                for &val in &inv_ac[..num_groups] {
                    permutation.push(prefix_u32 + val);
                }
                // Reorder sections: new[permutation[i]] = sections[i].
                let mut new_sections: Vec<Vec<u8>> = (0..total).map(|_| Vec::new()).collect();
                for (logical_idx, section_data) in sections.into_iter().enumerate() {
                    let on_disk = permutation[logical_idx] as usize;
                    new_sections[on_disk] = section_data;
                }
                let section_sizes: Vec<usize> = new_sections.iter().map(|s| s.len()).collect();
                write_toc_with_permutation(
                    &section_sizes,
                    &permutation,
                    self.use_ans,
                    &mut writer,
                )?;
                for section in new_sections {
                    writer.append_bytes(&section)?;
                }
            } else {
                let section_sizes: Vec<usize> = sections.iter().map(|s| s.len()).collect();
                write_toc(&section_sizes, &mut writer)?;
                for section in sections {
                    writer.append_bytes(&section)?;
                }
            }
        }

        let strategy_counts = ac_strategy.strategy_histogram();
        crate::debug_rect::flush("");
        Ok(VarDctOutput {
            data: writer.finish_with_padding(),
            strategy_counts,
        })
    }

    /// Encode with iterative rate control for improved distance targeting.
    ///
    /// This method:
    /// 1. Computes precomputed state (XYB, CfL, masking, AC strategy) once
    /// 2. Loops: encode → decode → butteraugli → adjust quant field
    /// 3. Returns when converged (within 5% of target) or max iterations reached
    ///
    /// Typically converges in 2-4 iterations. Each iteration costs ~50% of a
    /// full encode since XYB conversion, CfL, masking, and AC strategy are reused.
    ///
    /// Returns the encoded bytes. Use `encode_with_rate_control_config` for
    /// iteration count and custom configuration.
    ///
    /// Requires the `rate-control` feature.
    #[cfg(feature = "rate-control")]
    pub fn encode_with_rate_control(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
    ) -> Result<Vec<u8>> {
        let config = super::rate_control::RateControlConfig::default();
        let (encoded, _iters) =
            self.encode_with_rate_control_config(width, height, linear_rgb, &config)?;
        Ok(encoded)
    }

    /// Encode with iterative rate control and custom configuration.
    ///
    /// Returns `(encoded_bytes, iteration_count)`.
    ///
    /// Requires the `rate-control` feature.
    #[cfg(feature = "rate-control")]
    pub fn encode_with_rate_control_config(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        config: &super::rate_control::RateControlConfig,
    ) -> Result<(Vec<u8>, usize)> {
        // Compute precomputed state — patches detected and subtracted
        // BEFORE quant_field / mask / gaborish / CfL / AC strategy so
        // that all subsequent precomputation sees the patches-subtracted
        // XYB. Matches libjxl pipeline order
        // (`enc_heuristics.cc:1057-1194`).
        let precomputed = super::precomputed::EncoderPrecomputed::compute_with_budget(
            width,
            height,
            linear_rgb,
            self.distance,
            self.cfl_enabled,
            self.ac_strategy_enabled,
            self.pixel_domain_loss,
            self.enable_noise,
            self.enable_denoise,
            self.enable_gaborish,
            self.enable_patches,
            self.use_ans,
            self.encoder_mode,
            self.force_strategy,
            &self.profile,
            self.color_encoding.as_ref(),
            self.budget.as_ref(),
        )?;

        // Run rate control loop
        super::rate_control::encode_with_rate_control(self, &precomputed, config)
    }

    /// Encode from precomputed state with a specific quant field.
    ///
    /// This is the core encoding function used by rate control iterations.
    /// It skips XYB conversion, CfL, masking, and AC strategy computation,
    /// using the values from `precomputed` instead.
    ///
    /// Color-only entry point (no extra channels). For RGBA / depth /
    /// spot color / selection mask / thermal / CFA see
    /// [`Self::encode_from_precomputed_with_extras`].
    ///
    /// Requires the `rate-control` OR `__pre_quantized` feature. The
    /// latter is the unstable downstream-callable path used by
    /// jxl-encoder-gpu (or any other pre-quantized caller) to feed
    /// already-prepared XYB / AC strategy / CfL / quant field directly
    /// to the bitstream emit path, skipping XYB / CfL / masking /
    /// strat-search / butteraugli refinement that the encoder would
    /// otherwise re-do.
    #[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
    pub fn encode_from_precomputed(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
    ) -> Result<Vec<u8>> {
        self.encode_from_precomputed_with_extras(precomputed, quant_field, &[])
    }

    /// Encode from precomputed state with a specific quant field plus
    /// an arbitrary list of extra channels (alpha, depth, spot color,
    /// selection mask, thermal, CFA, …).
    ///
    /// Same fast path as [`Self::encode_from_precomputed`] (skips XYB
    /// conversion, CfL, masking, AC strategy search, and butteraugli
    /// refinement) but additionally writes a modular sub-bitstream
    /// carrying each extra channel alongside the VarDCT color data.
    /// Each extra's [`crate::headers::extra_channels::ExtraChannelInfo`]
    /// goes into the file-header metadata; the writer pulls
    /// `channel_width * channel_height` samples per channel out of the
    /// shared sub-bitstream in the same order they were passed in.
    ///
    /// **Current scope** (mirrors [`Self::encode_with_extras`]):
    /// - `dim_shift` must be `0` on every extra (full-resolution
    ///   channels).
    /// - Single-group (≤256×256) supports any number of extras.
    /// - Multi-group supports any number of extras at `dim_shift = 0`.
    ///
    /// Other combinations return [`crate::Error::InvalidInput`] so the
    /// wire format stays correct as those paths are filled in.
    ///
    /// **Butteraugli loop integration**: the buttloop only consumes
    /// the three color planes — extras are passed through unchanged to
    /// the final bitstream emit. Since the buttloop only refines
    /// `quant_field` (not extras), extras are forwarded after loop
    /// convergence; the GPU encoder / rate-control path produces the
    /// final `quant_field` first, then this entry threads the extras
    /// into the bitstream.
    ///
    /// Requires the `rate-control` OR `__pre_quantized` feature.
    #[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
    pub fn encode_from_precomputed_with_extras(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
        extras: &[crate::api::ExtraChannel<'_>],
    ) -> Result<Vec<u8>> {
        // Validate extras at the entry boundary (dim_shift = 0,
        // sample-count = width * height) before any encoding work runs.
        // Mirrors `encode_with_extras` so error messages and behavior
        // stay consistent across the two entry points. We need the
        // validated `VardctExtra` views for the final `encode_two_pass`
        // call below, so build them here once and reuse.
        let width = precomputed.width;
        let height = precomputed.height;
        let mut extras_views: Vec<super::extras::VardctExtra<'_>> =
            Vec::with_capacity(extras.len());
        for (idx, ec) in extras.iter().enumerate() {
            if ec.info().dim_shift != 0 {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: dim_shift = {} not yet supported in lossy encode (dim_shift > 0 \
                     for VarDCT extras is a follow-up)",
                    ec.info().dim_shift
                )));
            }
            let expected = width.checked_mul(height).ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 1,
            })?;
            let got = ec.data().len();
            if got != expected {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: expected {expected} samples for {width}x{height}, got {got}"
                )));
            }
            extras_views.push(super::extras::VardctExtra::from_api(ec));
        }
        self.check_alpha_squeeze_supported(&extras_views, Some((width, height)))?;
        self.encode_from_precomputed_inner(precomputed, quant_field, &extras_views)
    }

    /// Shared body for [`Self::encode_from_precomputed`] and
    /// [`Self::encode_from_precomputed_with_extras`]. Takes already-
    /// validated `extras` as the internal `VardctExtra` view.
    #[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
    fn encode_from_precomputed_inner(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
        extras: &[super::extras::VardctExtra<'_>],
    ) -> Result<Vec<u8>> {
        let width = precomputed.width;
        let height = precomputed.height;
        let xsize_blocks = precomputed.xsize_blocks;
        let ysize_blocks = precomputed.ysize_blocks;
        let padded_width = precomputed.padded_width;

        // Calculate group dimensions
        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
        let num_sections = 2 + num_dc_groups + num_groups;

        // Copy and adjust quant field for multi-block transforms
        let mut quant_field = quant_field.to_vec();
        adjust_quant_field_with_distance(&precomputed.ac_strategy, &mut quant_field, self.distance);

        // Compute distance params from effort profile
        let mut params = match self.original_distance {
            Some(orig) if orig > self.distance => {
                DistanceParams::compute_for_profile_with_original(
                    self.distance,
                    orig,
                    &self.profile,
                )
            }
            _ => DistanceParams::compute_for_profile(self.distance, &self.profile),
        };
        if let Some(rescale) = self.quant_ac_rescale {
            params.apply_quant_ac_rescale(rescale);
        }
        // libjxl --epf override: when the caller pinned a level, it wins
        // over the distance-derived `epf_iters` (enc_frame.cc:284-285).
        self.apply_epf_level_override(&mut params);

        // Apply pixel-level chromacity adjustments using pre-gaborish stats
        if self.profile.chromacity_adjustment {
            params.apply_chromacity_adjustment(
                precomputed.chromacity_x_pixelized,
                precomputed.chromacity_b_pixelized,
            );
        }

        // Patches handling.
        //
        // Two cases:
        //
        // (1) `precomputed.patches_data` is `Some(pd)`. The upstream
        // `compute_with_budget` already detected, quantized, and
        // subtracted patches from the XYB BEFORE running quant_field /
        // mask / gaborish / CfL / AC strategy — so `precomputed.xyb_*`
        // is the post-gaborish PATCHES-SUBTRACTED XYB and all the
        // precomputed state matches what the decoder will dequantize
        // against. We just write the patches frame to the bitstream.
        //
        // (2) `precomputed.patches_data` is `None` but
        // `precomputed.xyb_pre_gaborish` is `Some`. Legacy path used by
        // jxl-encoder-gpu callers that build via `from_parts` and
        // attach pre-gab XYB. We detect patches here, subtract from
        // pre-gab, re-apply gaborish, and use the result for DCT — but
        // the precomputed CfL / AC strategy / quant_field were computed
        // on UN-patched XYB so this case has the libjxl-parity gap
        // documented in the GPU encoder. Closing that requires GPU-side
        // patches detection before the GPU pipeline runs.
        //
        // (3) Both `None`: no patches.
        let _t_patches = std::time::Instant::now();
        let mut patches_data: Option<super::patches::PatchesData> =
            if precomputed.patches_data.is_some() {
                precomputed.patches_data.clone()
            } else if self.enable_patches
                && let Some(pre_gab) = precomputed.xyb_pre_gaborish.as_ref()
            {
                // Same distance-aware kMinPeak as `encode_inner` (~line 745):
                // libjxl parity at d<1.0, W2-5 chunk 1 relaxation at d>=1.0.
                // RFC#45 pick #5 chunk 3 per-patch cost gate — mirrors
                // `encode_inner` ~line 745 (see comment there).
                let min_peak = if self.distance < 1.0 { 2 } else { 1 };
                let mut pd = super::patches::find_and_build_with_per_patch_gate(
                    [&pre_gab[0], &pre_gab[1], &pre_gab[2]],
                    width,
                    height,
                    padded_width,
                    min_peak,
                    Some(self.distance),
                    self.use_ans,
                );
                if matches!(self.encoder_mode, crate::api::EncoderMode::Experimental)
                    && let Some(ref p) = pd
                    && !p.is_cost_effective(self.distance, self.use_ans)
                {
                    pd = None;
                }
                if let Some(ref mut p) = pd {
                    p.quantize_ref_image();
                }
                pd
            } else {
                None
            };
        // Materialize a gaborish_inverse'd patched XYB triple ONLY for
        // case (2) above — case (1) already has patches subtracted in
        // `precomputed.xyb_*`.
        let patched_xyb: Option<[Vec<f32>; 3]> = if precomputed.patches_data.is_none()
            && let (Some(pd), Some(pre_gab)) = (&patches_data, &precomputed.xyb_pre_gaborish)
        {
            let mut xyb = [pre_gab[0].clone(), pre_gab[1].clone(), pre_gab[2].clone()];
            super::patches::subtract_patches(&mut xyb, padded_width, pd);
            if precomputed.gaborish_enabled {
                let [mut x, mut y, mut b] = xyb;
                super::gaborish::gaborish_inverse(
                    &mut x,
                    &mut y,
                    &mut b,
                    padded_width,
                    precomputed.padded_height,
                    self.budget.as_ref(),
                )?;
                Some([x, y, b])
            } else {
                Some(xyb)
            }
        } else {
            None
        };
        let (xyb_x_for_dct, xyb_y_for_dct, xyb_b_for_dct): (&[f32], &[f32], &[f32]) =
            match &patched_xyb {
                Some([x, y, b]) => (x, y, b),
                None => (&precomputed.xyb_x, &precomputed.xyb_y, &precomputed.xyb_b),
            };
        // Suppress unused-mut warning when patches_data is left
        // unmodified after construction (case 1 + case 3).
        let _ = &mut patches_data;
        let _ms_patches = _t_patches.elapsed().as_secs_f64() * 1000.0;

        // CfL handling. For case (1) — patches detected and subtracted
        // upstream by `compute_with_budget` — `precomputed.cfl_map` was
        // already fitted to the patches-subtracted XYB and we use it
        // unchanged. For case (2) — patches detected here on a legacy
        // `from_parts + with_xyb_pre_gaborish` precomputed —
        // `precomputed.cfl_map` was fitted to the un-patched XYB; we
        // recompute pass 1 on the patches-subtracted XYB to avoid the
        // libjxl-parity gap on screenshot content.
        //
        // Pass 2 (refine_cfl_map) intentionally skipped in case (2):
        // it takes `ac_strategy` as input, and `precomputed.ac_strategy`
        // was computed on un-patched XYB — running pass 2 against a
        // mismatched strategy + patched XYB regressed file size by
        // 0.5-2% on gb82-sc screenshots in measurements 2026-05-15.
        let _t_cfl = std::time::Instant::now();
        let cfl_map_patched: Option<CflMap> = if patched_xyb.is_some() && self.cfl_enabled {
            Some(compute_cfl_map(
                xyb_x_for_dct,
                xyb_y_for_dct,
                xyb_b_for_dct,
                padded_width,
                precomputed.padded_height,
                xsize_blocks,
                ysize_blocks,
                false,
                self.profile.cfl_newton_eps,
                self.profile.cfl_newton_max_iters,
            ))
        } else {
            None
        };
        let cfl_map_for_encode: &CflMap = cfl_map_patched.as_ref().unwrap_or(&precomputed.cfl_map);
        let _ms_cfl = _t_cfl.elapsed().as_secs_f64() * 1000.0;

        // Perform DCT and quantization using precomputed XYB data
        let _t_xform = std::time::Instant::now();
        let transform_out = self.transform_and_quantize(
            xyb_x_for_dct,
            xyb_y_for_dct,
            xyb_b_for_dct,
            padded_width,
            xsize_blocks,
            ysize_blocks,
            &params,
            &mut quant_field,
            cfl_map_for_encode,
            &precomputed.ac_strategy,
        )?;
        let _ms_xform = _t_xform.elapsed().as_secs_f64() * 1000.0;
        let quant_dc = &transform_out.quant_dc;
        let quant_ac = &transform_out.quant_ac;
        let nzeros = &transform_out.nzeros;
        let raw_nzeros = &transform_out.raw_nzeros;

        // Compute per-block EPF sharpness map when EPF is active.
        // Mirrors the CPU path at `encode_image_lossy` (vardct/encoder.rs:1329-1362).
        // Dynamic sharpness gated at effort >= 6 (speed_tier <= kWombat) matching libjxl.
        // Without this the bitstream emits uniform sharpness=4, costing bytes on
        // content that benefits from per-block tuning.
        let _t_sharp = std::time::Instant::now();
        let padded_height = precomputed.padded_height;
        let sharpness_map =
            if params.epf_iters > 0 && self.distance >= 0.5 && self.profile.epf_dynamic_sharpness {
                let mask_fallback;
                let mask: &[f32] = match &precomputed.mask1x1 {
                    Some(m) => m,
                    None => {
                        mask_fallback = super::adaptive_quant::compute_mask1x1_with_budget(
                            &precomputed.xyb_y,
                            padded_width,
                            padded_height,
                            self.budget.as_ref(),
                        )?;
                        &mask_fallback
                    }
                };
                Some(super::epf::compute_epf_sharpness(
                    [xyb_x_for_dct, xyb_y_for_dct, xyb_b_for_dct],
                    quant_dc,
                    quant_ac,
                    &quant_field,
                    mask,
                    &params,
                    cfl_map_for_encode,
                    &precomputed.ac_strategy,
                    self.enable_gaborish,
                    xsize_blocks,
                    ysize_blocks,
                    self.budget.as_ref(),
                )?)
            } else {
                None
            };
        let _ms_sharp = _t_sharp.elapsed().as_secs_f64() * 1000.0;

        // Use two-pass mode for rate control (required for ANS)
        let _t_two = std::time::Instant::now();
        let res = self.encode_two_pass(
            width,
            height,
            &params,
            xsize_blocks,
            ysize_blocks,
            xsize_groups,
            ysize_groups,
            xsize_dc_groups,
            ysize_dc_groups,
            num_groups,
            num_dc_groups,
            num_sections,
            quant_dc,
            quant_ac,
            nzeros,
            raw_nzeros,
            &quant_field,
            cfl_map_for_encode,
            &precomputed.ac_strategy,
            &precomputed.noise_params,
            sharpness_map.as_deref(),
            extras,
            patches_data.as_ref(),
            None, // splines
            None, // float_dc
        );
        let _ms_two = _t_two.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some() {
            eprintln!(
                "encode_from_precomputed: patches={_ms_patches:.1} ms, cfl={_ms_cfl:.1} ms, transform_and_quantize={_ms_xform:.1} ms, sharpness={_ms_sharp:.1} ms, encode_two_pass={_ms_two:.1} ms",
            );
        }
        res
    }

    /// Pre-quantized AC entry point. Skips `transform_and_quantize`
    /// (forward DCT + quantize + nzeros) and goes straight to
    /// `encode_two_pass` (entropy coding). The caller is responsible
    /// for producing per-channel `quant_dc` / `quant_ac` / `nzeros` /
    /// `raw_nzeros` / `float_dc` matching the shape `transform_and_quantize`
    /// would have emitted on this `precomputed` input.
    ///
    /// Designed for the GPU encoder fast path where DCT + quantize
    /// run on the GPU and only the small per-block coefficient buffers
    /// need to cross the wire. Saves ~50 ms at 12 MP / d=1.0 on
    /// rayon-parallel CPU vs running `transform_and_quantize` again.
    ///
    /// Quant field adjustments (multi-block transform `adjust_quant_field_with_distance`)
    /// are applied internally — caller passes the **un-adjusted** per-block
    /// `u8` quant field.
    ///
    /// Color-only entry point (no extra channels). For RGBA / depth /
    /// spot color / selection mask / thermal / CFA see
    /// [`Self::encode_from_pre_quantized_ac_with_extras`].
    #[cfg(feature = "__pre_quantized")]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_from_pre_quantized_ac(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
        quant_dc: &[alloc::vec::Vec<alloc::vec::Vec<i16>>; 3],
        quant_ac: &[alloc::vec::Vec<alloc::vec::Vec<[i32; super::common::DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[alloc::vec::Vec<alloc::vec::Vec<u8>>; 3],
        raw_nzeros: &[alloc::vec::Vec<alloc::vec::Vec<u16>>; 3],
    ) -> Result<Vec<u8>> {
        self.encode_from_pre_quantized_ac_with_extras(
            precomputed,
            quant_field,
            quant_dc,
            quant_ac,
            nzeros,
            raw_nzeros,
            &[],
        )
    }

    /// Pre-quantized AC entry point that additionally writes a list of
    /// extra channels (alpha, depth, spot color, selection mask, thermal,
    /// CFA, …) as a modular sub-bitstream alongside the VarDCT color
    /// data.
    ///
    /// Same fast path as [`Self::encode_from_pre_quantized_ac`] (skips
    /// `transform_and_quantize` and goes straight to `encode_two_pass`)
    /// but threads `extras` through to the bitstream emit so the file
    /// header carries each extra's
    /// [`crate::headers::extra_channels::ExtraChannelInfo`] and the
    /// per-section modular sub-bitstream pulls
    /// `channel_width * channel_height` samples per channel in the
    /// order they were passed in.
    ///
    /// **Current scope** (mirrors
    /// [`Self::encode_from_precomputed_with_extras`]):
    /// - `dim_shift` must be `0` on every extra (full-resolution
    ///   channels).
    /// - Single-group (≤256×256) supports any number of extras.
    /// - Multi-group supports any number of extras at `dim_shift = 0`.
    ///
    /// Other combinations return [`crate::Error::InvalidInput`] so the
    /// wire format stays correct as those paths are filled in.
    ///
    /// **Pre-quantized AC contract**: the GPU encoder runs DCT + quant
    /// on the **color** planes only. Extras are not part of the
    /// pre-quantized AC pipeline — they are passed through unchanged to
    /// the final bitstream emit, the same way
    /// [`Self::encode_from_precomputed_with_extras`] threads them past
    /// the buttloop. Without this, an RGBA encode that goes through the
    /// GPU pre-quantized fast path would silently shed its alpha
    /// channel.
    #[cfg(feature = "__pre_quantized")]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_from_pre_quantized_ac_with_extras(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
        quant_dc: &[alloc::vec::Vec<alloc::vec::Vec<i16>>; 3],
        quant_ac: &[alloc::vec::Vec<alloc::vec::Vec<[i32; super::common::DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[alloc::vec::Vec<alloc::vec::Vec<u8>>; 3],
        raw_nzeros: &[alloc::vec::Vec<alloc::vec::Vec<u16>>; 3],
        extras: &[crate::api::ExtraChannel<'_>],
    ) -> Result<Vec<u8>> {
        let width = precomputed.width;
        let height = precomputed.height;
        let xsize_blocks = precomputed.xsize_blocks;
        let ysize_blocks = precomputed.ysize_blocks;
        let _ = xsize_blocks;
        let _ = ysize_blocks;

        // Validate extras at the entry boundary (dim_shift = 0,
        // sample-count = width * height) before any encoding work runs.
        // Mirrors `encode_from_precomputed_with_extras` so error
        // messages and behavior stay consistent across the two
        // pre-quantized entry points.
        let mut extras_views: Vec<super::extras::VardctExtra<'_>> =
            Vec::with_capacity(extras.len());
        for (idx, ec) in extras.iter().enumerate() {
            if ec.info().dim_shift != 0 {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: dim_shift = {} not yet supported in lossy encode (dim_shift > 0 \
                     for VarDCT extras is a follow-up)",
                    ec.info().dim_shift
                )));
            }
            let expected = width.checked_mul(height).ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 1,
            })?;
            let got = ec.data().len();
            if got != expected {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: expected {expected} samples for {width}x{height}, got {got}"
                )));
            }
            extras_views.push(super::extras::VardctExtra::from_api(ec));
        }
        self.check_alpha_squeeze_supported(&extras_views, Some((width, height)))?;

        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
        let num_sections = 2 + num_dc_groups + num_groups;

        // Apply multi-block quant field adjustment. Same step
        // `transform_and_quantize` would have applied internally before
        // calling encode_two_pass.
        let mut quant_field = quant_field.to_vec();
        adjust_quant_field_with_distance(&precomputed.ac_strategy, &mut quant_field, self.distance);

        let mut params = match self.original_distance {
            Some(orig) if orig > self.distance => {
                DistanceParams::compute_for_profile_with_original(
                    self.distance,
                    orig,
                    &self.profile,
                )
            }
            _ => DistanceParams::compute_for_profile(self.distance, &self.profile),
        };
        if let Some(rescale) = self.quant_ac_rescale {
            params.apply_quant_ac_rescale(rescale);
        }
        // libjxl --epf override: when the caller pinned a level, it wins
        // over the distance-derived `epf_iters` (enc_frame.cc:284-285).
        self.apply_epf_level_override(&mut params);
        if self.profile.chromacity_adjustment {
            params.apply_chromacity_adjustment(
                precomputed.chromacity_x_pixelized,
                precomputed.chromacity_b_pixelized,
            );
        }

        self.encode_two_pass(
            width,
            height,
            &params,
            precomputed.xsize_blocks,
            precomputed.ysize_blocks,
            xsize_groups,
            ysize_groups,
            xsize_dc_groups,
            ysize_dc_groups,
            num_groups,
            num_dc_groups,
            num_sections,
            quant_dc,
            quant_ac,
            nzeros,
            raw_nzeros,
            &quant_field,
            &precomputed.cfl_map,
            &precomputed.ac_strategy,
            &precomputed.noise_params,
            None,
            &extras_views,
            None,
            None,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = VarDctEncoder::new(1.0);
        assert_eq!(encoder.distance, 1.0);

        let encoder_default = VarDctEncoder::default();
        assert_eq!(encoder_default.distance, 1.0);
    }

    #[test]
    fn test_encode_small_image() {
        let encoder = VarDctEncoder::new(1.0);

        // Create a simple 8x8 red image
        let width = 8;
        let height = 8;
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = 1.0; // R
                linear_rgb[idx + 1] = 0.0; // G
                linear_rgb[idx + 2] = 0.0; // B
            }
        }

        // This should at least not panic - full encoding not yet implemented
        let result = encoder.encode(width, height, &linear_rgb, None);
        // For now, just check it produces some output
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.data.len() > 2);
        assert_eq!(output.data[0], 0xFF);
        assert_eq!(output.data[1], 0x0A);
    }

    #[test]
    fn test_convert_to_xyb_padded() {
        let encoder = VarDctEncoder::new(1.0);

        // Gray pixel (1x1 image -> padded to 8x8)
        let linear_rgb = vec![0.5, 0.5, 0.5];
        let (x, y, b) = encoder
            .convert_to_xyb_padded(1, 1, 8, 8, &linear_rgb)
            .unwrap();

        // Padded to 8x8 = 64 pixels
        assert_eq!(x.len(), 64);
        assert_eq!(y.len(), 64);
        assert_eq!(b.len(), 64);

        // Gray should have X ≈ 0 (equal L and M)
        assert!(x[0].abs() < 0.01, "X should be near zero for gray");
        assert!(y[0] > 0.0, "Y should be positive");
        assert!(b[0] > 0.0, "B should be positive");

        // Edge replication: all padded pixels should match the corner
        for i in 0..64 {
            assert!((x[i] - x[0]).abs() < 1e-6, "All padded X should match");
            assert!((y[i] - y[0]).abs() < 1e-6, "All padded Y should match");
            assert!((b[i] - b[0]).abs() < 1e-6, "All padded B should match");
        }
    }

    #[test]
    fn test_encode_16x16_red_image() {
        // Test a 16x16 pixel image (2x2 blocks) to compare with libjxl-tiny
        let encoder = VarDctEncoder::new(1.0);

        let width = 16;
        let height = 16;
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = 1.0; // R
                linear_rgb[idx + 1] = 0.0; // G
                linear_rgb[idx + 2] = 0.0; // B
            }
        }

        let result = encoder.encode(width, height, &linear_rgb, None);
        assert!(result.is_ok());
        let output = result.unwrap();

        eprintln!("Output file size: {} bytes", output.data.len());
        eprintln!(
            "First 32 bytes: {:02x?}",
            &output.data[..32.min(output.data.len())]
        );

        // Write output to file for comparison
        std::fs::write(std::env::temp_dir().join("our_16x16.jxl"), &output.data).unwrap();

        // libjxl-tiny produces:
        // DC_group: 106 bits (14 bytes)
        // Total combined: 1086 bytes
        // Total file: 1104 bytes
        //
        // Our encoder should match these sizes

        // Check signature
        assert_eq!(output.data[0], 0xFF);
        assert_eq!(output.data[1], 0x0A);
    }

    /// Compute a simple hash of a byte slice for output locking.
    fn hash_bytes(bytes: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash-locked test for 8x8 gradient image.
    /// This test ensures the encoder output doesn't change unexpectedly.
    /// x86_64 only: FP rounding differs on other architectures and 32-bit.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_lock_8x8_gradient() {
        let encoder = VarDctEncoder::new(1.0);
        let width = 8;
        let height = 8;
        let mut linear_rgb = vec![0.0f32; width * height * 3];

        // Simple gradient: R increases with x, G with y
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = x as f32 / 7.0; // R
                linear_rgb[idx + 1] = y as f32 / 7.0; // G
                linear_rgb[idx + 2] = 0.5; // B
            }
        }

        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let hash = hash_bytes(&bytes);

        // Lock the hash - if this changes, the encoding has changed
        // Updated: fix multi-DC-group context tree splitval
        const EXPECTED_HASH: u64 = 0xfde7b582460edebc;
        assert_eq!(
            hash,
            EXPECTED_HASH,
            "8x8 gradient hash mismatch: got {:#x}, expected {:#x}. \
             Output size: {} bytes. If intentional, update EXPECTED_HASH.",
            hash,
            EXPECTED_HASH,
            bytes.len()
        );
    }

    /// Hash-locked test for 16x16 solid color image.
    /// x86_64 only: FP rounding differs on other architectures and 32-bit.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_lock_16x16_solid() {
        let encoder = VarDctEncoder::new(1.0);
        let width = 16;
        let height = 16;
        let linear_rgb = vec![0.3f32; width * height * 3]; // gray

        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let hash = hash_bytes(&bytes);

        // Updated: fix multi-DC-group context tree splitval
        const EXPECTED_HASH: u64 = 0xb71172a676faf64d;
        assert_eq!(
            hash,
            EXPECTED_HASH,
            "16x16 solid hash mismatch: got {:#x}, expected {:#x}. \
             Output size: {} bytes. If intentional, update EXPECTED_HASH.",
            hash,
            EXPECTED_HASH,
            bytes.len()
        );
    }

    /// Hash-locked test for 64x64 checkerboard pattern.
    /// x86_64 only: FP rounding differs on other architectures and 32-bit.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_lock_64x64_checkerboard() {
        let encoder = VarDctEncoder::new(1.0);
        let width = 64;
        let height = 64;
        let mut linear_rgb = vec![0.0f32; width * height * 3];

        // 8x8 checkerboard pattern
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let checker = ((x / 8) + (y / 8)) % 2 == 0;
                let val = if checker { 0.8 } else { 0.2 };
                linear_rgb[idx] = val;
                linear_rgb[idx + 1] = val;
                linear_rgb[idx + 2] = val;
            }
        }

        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let hash = hash_bytes(&bytes);

        // Updated: fix multi-DC-group context tree splitval
        const EXPECTED_HASH: u64 = 0xeb729ad9e2766dd7;
        assert_eq!(
            hash,
            EXPECTED_HASH,
            "64x64 checkerboard hash mismatch: got {:#x}, expected {:#x}. \
             Output size: {} bytes. If intentional, update EXPECTED_HASH.",
            hash,
            EXPECTED_HASH,
            bytes.len()
        );
    }

    /// Hash-locked test for non-power-of-two size (tests padding).
    /// x86_64 only: FP rounding differs on other architectures and 32-bit.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_lock_13x17_noise() {
        let encoder = VarDctEncoder::new(1.0);
        let width = 13;
        let height = 17;
        let mut linear_rgb = vec![0.0f32; width * height * 3];

        // Deterministic pseudo-random pattern
        let mut seed = 12345u64;
        for val in &mut linear_rgb {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = ((seed >> 32) as f32) / (u32::MAX as f32);
        }

        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let hash = hash_bytes(&bytes);

        // Updated: fix multi-DC-group context tree splitval
        const EXPECTED_HASH: u64 = 0x8a3db6460320e743;
        assert_eq!(
            hash,
            EXPECTED_HASH,
            "13x17 noise hash mismatch: got {:#x}, expected {:#x}. \
             Output size: {} bytes. If intentional, update EXPECTED_HASH.",
            hash,
            EXPECTED_HASH,
            bytes.len()
        );
    }

    /// Roundtrip quality test for non-8-aligned dimensions.
    ///
    /// Encodes a 100x75 gradient, decodes with jxl-oxide, and verifies:
    /// 1. Dimensions match
    /// 2. Output is a valid JXL file (correct signature, decodable)
    ///
    /// This catches stride mismatch bugs where padded XYB buffers have
    /// stride != width, which corrupts adaptive quant, CfL, and AC strategy.
    #[test]
    fn test_roundtrip_non_8_aligned() {
        for &(w, h) in &[(100, 75), (13, 17), (33, 49), (7, 9)] {
            let mut linear_rgb = vec![0.0f32; w * h * 3];

            // Smooth gradient (linear RGB)
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    linear_rgb[idx] = x as f32 / w.max(1) as f32;
                    linear_rgb[idx + 1] = y as f32 / h.max(1) as f32;
                    linear_rgb[idx + 2] = 0.3;
                }
            }

            let encoder = VarDctEncoder::new(1.0);
            let bytes = encoder
                .encode(w, h, &linear_rgb, None)
                .unwrap_or_else(|e| panic!("encode {}x{} failed: {}", w, h, e))
                .data;

            // Verify JXL signature
            assert_eq!(bytes[0], 0xFF, "{}x{}: bad signature byte 0", w, h);
            assert_eq!(bytes[1], 0x0A, "{}x{}: bad signature byte 1", w, h);

            // Decode with jxl-oxide and verify dimensions
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap_or_else(|e| panic!("jxl-oxide decode {}x{} failed: {}", w, h, e));
            assert_eq!(
                image.width(),
                w as u32,
                "{}x{}: decoded width mismatch",
                w,
                h
            );
            assert_eq!(
                image.height(),
                h as u32,
                "{}x{}: decoded height mismatch",
                w,
                h
            );

            // Render to verify pixel data is valid
            let render = image
                .render_frame(0)
                .unwrap_or_else(|e| panic!("jxl-oxide render {}x{} failed: {}", w, h, e));
            let _pixels = render.image_all_channels();
        }
    }

    /// Test DC tree learning produces valid output.
    #[test]
    fn test_dc_tree_learning() {
        let width = 64;
        let height = 64;

        // Create a gradient image
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = x as f32 / width as f32;
                linear_rgb[idx + 1] = y as f32 / height as f32;
                linear_rgb[idx + 2] = 0.5;
            }
        }

        // Encode WITHOUT DC tree learning (baseline) — use ANS
        let mut encoder_baseline = VarDctEncoder::new(1.0);
        encoder_baseline.dc_tree_learning = false;
        let bytes_baseline = encoder_baseline
            .encode(width, height, &linear_rgb, None)
            .expect("baseline encode failed")
            .data;

        // Encode WITH DC tree learning — also use ANS
        let mut encoder_learned = VarDctEncoder::new(1.0);
        encoder_learned.dc_tree_learning = true;
        std::fs::write(
            std::env::temp_dir().join("dc_baseline_test.jxl"),
            &bytes_baseline,
        )
        .unwrap();
        let bytes_learned = encoder_learned
            .encode(width, height, &linear_rgb, None)
            .expect("learned encode failed")
            .data;
        std::fs::write(
            std::env::temp_dir().join("dc_learned_test.jxl"),
            &bytes_learned,
        )
        .unwrap();

        eprintln!(
            "DC tree learning: baseline={} bytes, learned={} bytes (delta={:.2}%)",
            bytes_baseline.len(),
            bytes_learned.len(),
            (bytes_learned.len() as f64 / bytes_baseline.len() as f64 - 1.0) * 100.0
        );

        // Verify both produce valid JXL signature
        assert_eq!(bytes_baseline[0], 0xFF);
        assert_eq!(bytes_baseline[1], 0x0A);
        assert_eq!(bytes_learned[0], 0xFF);
        assert_eq!(bytes_learned[1], 0x0A);

        // Verify baseline decodes (sanity check)
        {
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes_baseline))
                .expect("jxl-oxide parse of baseline failed");
            let render = image
                .render_frame(0)
                .expect("jxl-oxide render of baseline failed");
            let _pixels = render.image_all_channels();
            eprintln!("Baseline ANS decodes OK ({} bytes)", bytes_baseline.len());
        }

        // Decode the learned version with jxl-oxide to verify it's valid
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes_learned))
            .expect("jxl-oxide decode of learned version failed");
        assert_eq!(image.width(), width as u32);
        assert_eq!(image.height(), height as u32);

        // Render to verify pixel data is valid
        let render = image
            .render_frame(0)
            .expect("jxl-oxide render of learned version failed");
        let _pixels = render.image_all_channels();
        eprintln!("Learned ANS decodes OK ({} bytes)", bytes_learned.len());

        // Also verify with djxl
        std::fs::write(
            std::env::temp_dir().join("dc_learned_test.jxl"),
            &bytes_learned,
        )
        .unwrap();
    }

    /// Test that the butteraugli quantization loop produces valid output.
    #[cfg(feature = "butteraugli-loop")]
    #[test]
    fn test_butteraugli_loop_basic() {
        // Create a 64x64 test image with some variation
        let width = 64;
        let height = 64;
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;
                linear_rgb[idx] = fx * 0.8; // R
                linear_rgb[idx + 1] = fy * 0.6; // G
                linear_rgb[idx + 2] = (1.0 - fx) * 0.4; // B
            }
        }

        // Encode without butteraugli loop
        let mut encoder_baseline = VarDctEncoder::new(2.0);
        encoder_baseline.butteraugli_iters = 0;
        let bytes_baseline = encoder_baseline
            .encode(width, height, &linear_rgb, None)
            .expect("baseline encode failed")
            .data;

        // Encode with 2 butteraugli loop iterations
        let mut encoder_loop = VarDctEncoder::new(2.0);
        encoder_loop.butteraugli_iters = 2;
        let bytes_loop = encoder_loop
            .encode(width, height, &linear_rgb, None)
            .expect("butteraugli loop encode failed")
            .data;

        // Both should produce valid JXL
        assert_eq!(bytes_baseline[0], 0xFF);
        assert_eq!(bytes_baseline[1], 0x0A);
        assert_eq!(bytes_loop[0], 0xFF);
        assert_eq!(bytes_loop[1], 0x0A);

        // File sizes should differ (butteraugli loop changes quant field)
        eprintln!(
            "Baseline: {} bytes, Butteraugli loop (2 iters): {} bytes",
            bytes_baseline.len(),
            bytes_loop.len()
        );

        // Verify the butteraugli-loop output decodes correctly
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes_loop))
            .expect("jxl-oxide decode of butteraugli loop output failed");
        assert_eq!(image.width(), width as u32);
        assert_eq!(image.height(), height as u32);

        let render = image
            .render_frame(0)
            .expect("jxl-oxide render of butteraugli loop output failed");
        let _pixels = render.image_all_channels();
        eprintln!("Butteraugli loop output decodes OK");
    }

    // ── Chunk-1 framework tests for `with_alpha_squeeze` ──────────────────
    //
    // These tests exercise the new constants + quantizer fn so the
    // framework is provably reachable. They don't yet drive a real
    // squeeze-on-extras encode (that's chunk 2) — they verify the
    // chunk-1 contract: (a) constants are libjxl-parity, (b) the
    // shifted quantizer behaves correctly across the table, (c) the
    // opt-in flag surfaces NotImplemented at the right boundary AND
    // stays at byte-identical lossless when alpha_distance is unset.

    #[test]
    fn squeeze_luma_qtable_matches_libjxl_constants() {
        // Mirrors `lib/jxl/enc_modular.cc:101-103` exactly.
        let expected: [f32; 16] = [
            163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04,
            0.02, 0.01, 0.005,
        ];
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(
                SQUEEZE_LUMA_QTABLE[i], v,
                "SQUEEZE_LUMA_QTABLE[{i}] mismatch vs libjxl enc_modular.cc:101-103"
            );
        }
        assert_eq!(SQUEEZE_LUMA_QTABLE_LEN, 16);
        assert_eq!(SQUEEZE_QUALITY_FACTOR_CONST, 0.35);
        assert_eq!(SQUEEZE_LUMA_FACTOR_CONST, 1.1);
    }

    #[test]
    fn compute_extra_pixel_quantizer_shifted_alpha_d2_shift0_matches_responsive1() {
        // At shift = 0, 8-bit alpha, d = 2.0:
        // q_float = 0.25 * 2.0 * 1.0 * 0.35 * 1.1 * 163.84 = 31.5392
        // floor → 31. This is ~10x the responsive=0 q value (q=3 at
        // d=2.0 in compute_extra_pixel_quantizer), confirming the
        // responsive=1 vs responsive=0 base divergence (`* 0.1` skip).
        let mut enc = VarDctEncoder::new(1.0);
        enc.alpha_distance = Some(2.0);
        let q = enc.compute_extra_pixel_quantizer_shifted(
            8,
            crate::headers::extra_channels::ExtraChannelType::Alpha,
            0,
        );
        assert_eq!(q, 31, "shift=0 d=2.0 8-bit alpha should yield q=31");

        // Sanity: the responsive=0 path at d=2.0 yields q=3 (10x smaller).
        let q_r0 = enc.compute_extra_pixel_quantizer(
            8,
            crate::headers::extra_channels::ExtraChannelType::Alpha,
        );
        assert_eq!(q_r0, 3, "responsive=0 path unchanged");
    }

    #[test]
    fn compute_extra_pixel_quantizer_shifted_drops_to_1_at_deep_shifts() {
        // Deeper shifts → smaller qtable_val → q < 1 → clamp to 1
        // (lossless). At shift 11+ (qtable[11] = 0.08) with d=2.0,
        // 8-bit: q_float = 0.25 * 2.0 * 1.0 * 0.35 * 1.1 * 0.08 =
        // 0.0154 → floor 0 → clamp to 1.
        let mut enc = VarDctEncoder::new(1.0);
        enc.alpha_distance = Some(2.0);
        for shift in 11..=15 {
            let q = enc.compute_extra_pixel_quantizer_shifted(
                8,
                crate::headers::extra_channels::ExtraChannelType::Alpha,
                shift,
            );
            assert_eq!(q, 1, "deep shift={shift} should clamp to q=1 (lossless)");
        }
    }

    #[test]
    fn compute_extra_pixel_quantizer_shifted_clamps_oversized_shift() {
        // shift > 15 must clamp to 15 (the table last entry), never panic.
        let mut enc = VarDctEncoder::new(1.0);
        enc.alpha_distance = Some(5.0);
        let q_at_15 = enc.compute_extra_pixel_quantizer_shifted(
            8,
            crate::headers::extra_channels::ExtraChannelType::Alpha,
            15,
        );
        let q_at_100 = enc.compute_extra_pixel_quantizer_shifted(
            8,
            crate::headers::extra_channels::ExtraChannelType::Alpha,
            100,
        );
        assert_eq!(q_at_100, q_at_15, "shift > 15 must clamp, not panic");
    }

    #[test]
    fn compute_extra_pixel_quantizer_shifted_returns_1_for_non_alpha() {
        // Non-alpha extras stay lossless at every shift — only alpha
        // has the per-channel distance knob wired today.
        let mut enc = VarDctEncoder::new(1.0);
        enc.alpha_distance = Some(2.0);
        for ec_type in [
            crate::headers::extra_channels::ExtraChannelType::Depth,
            crate::headers::extra_channels::ExtraChannelType::SpotColor,
            crate::headers::extra_channels::ExtraChannelType::Thermal,
        ] {
            for shift in 0..=15 {
                let q = enc.compute_extra_pixel_quantizer_shifted(8, ec_type, shift);
                assert_eq!(
                    q, 1,
                    "non-alpha ec_type={ec_type:?} shift={shift} must stay lossless"
                );
            }
        }
    }

    #[test]
    fn alpha_squeeze_engaged_predicate() {
        // Default: flag off → never engaged.
        let mut enc = VarDctEncoder::new(1.0);
        assert!(!enc.alpha_squeeze_engaged(), "default off");

        // Flag on but no alpha distance → not engaged.
        enc.alpha_squeeze = true;
        assert!(
            !enc.alpha_squeeze_engaged(),
            "no alpha_distance → not engaged"
        );

        // Flag on but distance=0 → not engaged (lossless contract).
        enc.alpha_distance = Some(0.0);
        assert!(
            !enc.alpha_squeeze_engaged(),
            "alpha_distance=0 → not engaged"
        );

        // Flag on + non-zero distance → engaged.
        enc.alpha_distance = Some(2.0);
        assert!(enc.alpha_squeeze_engaged(), "true + d=2.0 → engaged");

        // Flag off + non-zero distance → not engaged
        // (responsive=0 path still runs).
        enc.alpha_squeeze = false;
        assert!(!enc.alpha_squeeze_engaged(), "flag off wins");
    }
}
