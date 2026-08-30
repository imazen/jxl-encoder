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
//!
//! # Module organization
//!
//! The independent pieces of the former `api.rs` monolith live in `api/`
//! submodules, re-exported here so public paths are unchanged:
//! `pixel_layout`, `quality`, `metadata`, `limits`, `animation`,
//! `container` (leaf types); `strategy` (encoder-mode / strategy enums);
//! `errors` (`EncodeError`); `ingest` (pixel-unpack + transfer-function
//! helpers); `content_detect` (smart-dispatch discriminators);
//! `validate` (encode-entry guards); `animate` (multi-frame paths);
//! `tests`.
//!
//! What deliberately stays in this file is the tightly-coupled encode
//! core: `LossyConfig`, `LosslessConfig`, `EncodeRequest`, `ExtraChannel`,
//! the streaming `LossyEncoder` / `LosslessEncoder`, and the
//! `EncodeResult` / `EncodeStats` result types. These share direct
//! private-field access across each other (e.g. `EncodeRequest::encode_lossy`
//! reads ~40 `LossyConfig` fields; the encode path mutates `ExtraChannel`
//! internals in place). Splitting any of them into a separate module would
//! force dozens of fields to `pub(crate)` — an encapsulation loss, not a
//! win — so they are kept together by design. See CLAUDE.md's api-narrowing
//! notes before relocating them; the clean cut is at the leaf/helper
//! boundary above, not through the core.

pub use crate::entropy_coding::Lz77Method;
pub use crate::headers::frame_header::BlendMode;
#[cfg(feature = "butteraugli-loop")]
pub use crate::vardct::hdr_metrics::HdrLoss;
pub use enough::{Stop, Unstoppable};
pub use whereat::{At, ResultAtExt, at};

// ── Pixel byte-slice casting ────────────────────────────────────────────────

/// Reinterpret raw pixel bytes as `T` lanes, tolerating any caller alignment.
///
/// `bytemuck::cast_slice` panics when the byte slice is not aligned for `T`,
/// and a caller-supplied `&[u8]` is only guaranteed 1-byte alignment (e.g. a
/// sub-slice at an odd offset into a larger file buffer) — so every 16-bit /
/// f32 pixel layout could be made to panic by input *placement* alone. The
/// aligned case — practically every buffer that was allocated as pixels —
/// stays zero-copy; misaligned input is copied into an owned, aligned buffer
/// instead of panicking.
///
/// `pixels.len()` must be a multiple of `size_of::<T>()`; `validate_pixels`
/// guarantees this before any conversion path runs.
pub(crate) fn cast_pixel_lanes<T: bytemuck::AnyBitPattern>(
    pixels: &[u8],
) -> alloc::borrow::Cow<'_, [T]> {
    debug_assert_eq!(pixels.len() % core::mem::size_of::<T>(), 0);
    match bytemuck::try_cast_slice::<u8, T>(pixels) {
        Ok(lanes) => alloc::borrow::Cow::Borrowed(lanes),
        Err(_) => alloc::borrow::Cow::Owned(
            // `as_chunks::<{size_of::<T>()}>()` needs unstable
            // generic_const_exprs; the lint's suggestion can't apply here.
            #[allow(clippy::chunks_exact_to_as_chunks)]
            pixels
                .chunks_exact(core::mem::size_of::<T>())
                .map(bytemuck::pod_read_unaligned::<T>)
                .collect(),
        ),
    }
}

// ── Error type ──────────────────────────────────────────────────────────────

mod errors;
pub(crate) use errors::at_from;
pub use errors::*;
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
    budget_peak_bytes: u64,
    threads_used: u32,
    estimated_peak_bytes: u64,
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

    /// Highest cumulative bytes reserved on the encoder's internal
    /// [`MemoryBudget`](crate::budget) during this encode.
    ///
    /// Tracks only the guarded dimension-driven allocation sites (XYB
    /// planes, group buffers, modular channels, butteraugli precompute…),
    /// so it is a lower bound on the process working set — the delta to
    /// real peak RSS is the unguarded allocation mass (see
    /// `benchmarks/jxl_encode_mem_threads_2026-08-01.tsv`). 0 for paths
    /// that don't populate it (animation).
    pub fn budget_peak_bytes(&self) -> u64 {
        self.budget_peak_bytes
    }

    /// Worker thread count the encode actually ran with, after the
    /// pre-flight walked the configured/ambient count down to fit the
    /// memory budget (see [`crate::heuristics::estimate_encode_threaded`]).
    /// 0 = ambient rayon pool (the pre-flight found no reduction needed
    /// and the caller requested 0 = ambient).
    pub fn threads_used(&self) -> u32 {
        self.threads_used
    }

    /// Pre-flight estimated peak memory (bytes) at [`Self::threads_used`],
    /// from the calibrated [`crate::heuristics::estimate_encode_threaded`]
    /// model. This is the estimate the budget admission decision used.
    pub fn estimated_peak_bytes(&self) -> u64 {
        self.estimated_peak_bytes
    }
}

/// The **compression kind**: lossy (VarDCT) vs lossless (modular).
///
/// This is the *what-format* axis. Do **not** confuse it with
/// [`EncoderMode`] (`Reference` / `Experimental`), which is a different,
/// orthogonal axis — *how libjxl-faithful* the encoder's algorithm choices
/// are. The two names differ by one letter but mean unrelated things:
/// `EncodeMode` = lossy/lossless; `EncoderMode` = reference/experimental
/// (itself largely subsumed by [`EncoderStrategy`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EncodeMode {
    /// Lossy (VarDCT) encoding.
    #[default]
    Lossy,
    /// Lossless (modular) encoding.
    Lossless,
}

// ── PixelLayout ─────────────────────────────────────────────────────────────

mod pixel_layout;
pub use pixel_layout::*;
mod quality;
pub use quality::*;
// interp_quality is a pub(crate) helper (not public API); re-export at the
// api level so api/tests.rs reaches it by bare name via `use super::*`.
#[cfg(test)]
pub(crate) use quality::interp_quality;
mod content_detect;
pub(crate) use content_detect::detect_smooth_photo_for_dct64_from_layout;
pub use content_detect::downsample_channel_u8;
// `classify_from_proxies` is exercised only by api/tests.rs; re-export it at
// the api level under cfg(test) so those tests reach it via `use super::*`.
#[cfg(test)]
pub(crate) use content_detect::classify_from_proxies;

// ── Supporting types ────────────────────────────────────────────────────────

mod metadata;
pub use metadata::*;
mod limits;
pub use limits::*;
// ── Animation ──────────────────────────────────────────────────────────────

mod animation;
pub use animation::*;

// ── Shared knob enums (LossyConfig + LosslessConfig) ───────────────────────

mod container;
pub use container::*;

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
/// Sectioned local-tree lossless encoding (imazen/jxl-encoder#96): learn one
/// MA tree PER 256x256 GROUP instead of one whole-image tree. Peak encode
/// memory drops to roughly the C-encoder level (4K photo e7: 834 -> 469 MB
/// measured) because the whole-image sample accumulator never exists, with
/// byte cost measured between -2.0% (photo e7) and +0.6% (photo e9).
///
/// `Auto` (the default) engages when the encode's memory budget
/// (`Limits::max_memory_bytes`, or the built-in lossless cap) cannot fit the
/// whole-image estimate — budget-capped and very large encodes transparently
/// switch instead of failing allocation — and, since 2026-08-19, at effort
/// <= 7 whenever the encode runs with more than one worker thread (measured
/// median-byte-neutral for -40 %+ wall on the 13-pick corpus study,
/// `benchmarks/lossless_sectioned_vs_global_x64_2026-08-18.*`). Output at
/// lossless e <= 7 therefore depends on the thread configuration by design;
/// pin `On` / `Off` for thread-invariant bytes. Scope: tree-learning ANS
/// encodes, including palette / ChannelCompact content (the meta channels
/// are coded in the global stream with their own tiny tree) and the
/// lossless patches dictionary; only custom-DC-quant (lossy-modular) and
/// the non-tree / non-ANS modes keep the whole-image tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SectionedTrees {
    /// Engage when the memory budget requires it (default).
    #[default]
    Auto,
    /// Never sectioned: always the whole-image global tree.
    Off,
    /// Always sectioned (tree-learning ANS encodes; see the type docs).
    On,
    /// Learn BOTH the global tree and per-group trees, and write each
    /// group with whichever is smaller (per-group `use_global_tree`
    /// choice — measured −2.25% (e7) / −0.25% (e9) vs the global tree on
    /// the 4K photo cell, ≥ global on every content class by
    /// construction). Uses global-mode memory; the per-group learns ride
    /// the gather waves in parallel.
    Hybrid,
}

#[derive(Clone, Debug)]
pub struct LosslessConfig {
    effort: u8,
    /// Sectioned local-tree mode selection. See [`SectionedTrees`].
    sectioned_trees: SectionedTrees,
    mode: EncoderMode,
    /// Effort-derived knob: `None` inherits the effort schedule's
    /// `use_ans`; `Some(v)` is a caller override. The `Option` *is* the
    /// touched-bit the pre-#80 `use_ans_explicit` flag tracked, so
    /// [`Self::with_effort`] is a pure setter (order-independent). See
    /// [`LossyConfig::use_ans`].
    use_ans: Option<bool>,
    squeeze: bool,
    /// Effort-derived knob (`None` inherits the schedule). `is_some()`
    /// replaces the pre-#80 `tree_learning_user_set` flag and gates the
    /// issue-#72 16-bit e5/e6 budgeted-tree lift.
    tree_learning: Option<bool>,
    /// Effort-derived knob (`None` inherits the schedule).
    lz77: Option<bool>,
    /// Effort-derived knob (`None` inherits the schedule).
    lz77_method: Option<Lz77Method>,
    /// Effort-derived knob (`None` inherits the per-image profile's
    /// `patches`). `is_some()` replaces the pre-#80 `patches_explicit`.
    patches: Option<bool>,
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
    /// Sweep / picker hook (`__expert`): sparse internal-param overrides
    /// stored as the params themselves (NOT an eagerly-resolved profile)
    /// and applied lazily in [`Self::effective_profile`] against the
    /// CURRENT effort — so `with_internal_params(_).with_effort(_)`
    /// resolves correctly regardless of builder order (issue #80).
    #[cfg(feature = "__expert")]
    internal_overrides: Option<crate::effort::LosslessInternalParams>,
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
    /// Resource [`Limits`] consulted by the untrusted JPEG-transcode path
    /// ([`Self::encode_jpeg_transcode`] /
    /// [`Self::encode_jpeg_transcode_codestream`]). Currently the transcode
    /// parser reads [`Limits::max_pixels`] as the pre-flight SOF pixel cap
    /// (default [`Limits::DEFAULT_MAX_JPEG_TRANSCODE_PIXELS`] = 120 MP when
    /// `None`). Set via [`Self::with_limits`]. (The pixel / [`EncodeRequest`]
    /// lossless path takes its limits from [`EncodeRequest::with_limits`]
    /// instead.)
    #[cfg(feature = "jpeg-reencoding")]
    limits: Option<Limits>,
}

impl Default for LosslessConfig {
    fn default() -> Self {
        Self::with_effort_level(7)
    }
}

impl LosslessConfig {
    /// Sectioned local-tree mode selection — see [`SectionedTrees`].
    /// Additive knob (imazen/jxl-encoder#96); `Auto` is the default and
    /// keeps ordinary encodes byte-identical.
    #[must_use]
    pub fn with_sectioned_trees(mut self, mode: SectionedTrees) -> Self {
        self.sectioned_trees = mode;
        self
    }

    /// Current [`SectionedTrees`] selection.
    pub fn sectioned_trees(&self) -> SectionedTrees {
        self.sectioned_trees
    }

    fn with_effort_level(effort: u8) -> Self {
        let profile = crate::effort::EffortProfile::lossless(effort, EncoderMode::Reference);
        Self {
            effort: profile.effort,
            sectioned_trees: SectionedTrees::Auto,
            mode: EncoderMode::Reference,
            use_ans: None,
            tree_learning: None,
            squeeze: false, // squeeze hurts even with tree learning (14-62% larger on both photos and screenshots)
            lz77: None,
            lz77_method: None,
            patches: None,
            lossy_palette: false,
            threads: 0,
            tree_sample_fraction_override: None,
            forced_rct: None,
            #[cfg(feature = "__expert")]
            internal_overrides: None,
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
            #[cfg(feature = "jpeg-reencoding")]
            limits: None,
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
    /// Base effort+mode schedule used to resolve the effort-derived
    /// `Option` knobs (issue #80). See [`LossyConfig::effort_schedule`].
    fn effort_schedule(&self) -> crate::effort::EffortProfile {
        crate::effort::EffortProfile::lossless(self.effort, self.mode)
    }

    pub(crate) fn effective_profile(&self) -> crate::effort::EffortProfile {
        let mut p = crate::effort::EffortProfile::lossless(self.effort, self.mode);
        // Sweep/picker internal-param overrides (issue #80): applied
        // lazily against the CURRENT effort.
        #[cfg(feature = "__expert")]
        if let Some(ip) = self.internal_overrides.clone() {
            ip.apply_to(&mut p);
        }
        // Sparse-override resolution (issue #80): apply the effort-derived
        // knob overrides on top of the schedule.
        if let Some(v) = self.use_ans {
            p.use_ans = v;
        }
        if let Some(v) = self.tree_learning {
            p.tree_learning = v;
        }
        if let Some(v) = self.lz77 {
            p.lz77 = v;
        }
        if let Some(v) = self.lz77_method {
            p.lz77_method = v;
        }
        if let Some(v) = self.patches {
            p.patches = v;
        }
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

    /// Resolved effort-profile (effort schedule + `__expert`/sparse
    /// overrides + `faster_decoding`), exposed for sweep
    /// encode-fingerprinting: the resolved byte-affecting state two
    /// knobsets must share to be byte-identical (see zenjxl
    /// `encode_fingerprint`). Public wrapper over the crate-internal
    /// resolver so sweep tooling needn't duplicate the override mapping.
    ///
    /// `#[doc(hidden)]` (#76): returns the internal [`EffortProfile`];
    /// reachable-but-unsupported, like the type itself.
    #[doc(hidden)]
    pub fn resolved_profile(&self) -> crate::effort::EffortProfile {
        self.effective_profile()
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
    /// Returns the stored `cfg.lz77()` field at tier 0.
    pub(crate) fn effective_lz77(&self) -> bool {
        if self.faster_decoding >= 1 {
            return false;
        }
        self.lz77()
    }

    /// Resolve the effective tree-learning enable flag, honoring
    /// `faster_decoding >= 4` (libjxl `enc_modular.cc:506-513` zeros
    /// `nb_repeats` at tier 4, disabling MA-tree learning).
    pub(crate) fn effective_tree_learning(&self) -> bool {
        if self.faster_decoding >= 4 {
            return false;
        }
        self.tree_learning()
    }

    /// Issue #72: 16-bit lossless at e5/e6 — budgeted tree learning.
    ///
    /// cjxl learns MA trees at EVERY effort (`enc_modular.cc` tier
    /// table: 3 props / x0.3 samples at the bottom, scaling up); our
    /// schedule was binary off-until-e7, which left 16-bit (HDR PQ)
    /// content +29..+49 % vs cjxl at e5/e6 (mean over 76 imazen-26 HDR
    /// PNGs, `benchmarks/hdr_png_ab_2026-06-11.meta`). This lift enables
    /// tree learning for integer 16-bit RGB(A) input at e5/e6 with hard
    /// budget caps (sample fraction 0.05, 4 properties, 32 buckets, 1
    /// RCT trial, no WP param search) — measured -22 % bytes on the
    /// 512^2 HDR-crop corpus at ~4x the (very cheap) non-tree e5 wall,
    /// landing in cjxl's own e5/e6 wall ballpark. 8-bit input is
    /// untouched. Opt out by calling [`Self::with_tree_learning`]
    /// explicitly, or via `JXL_NO_16BIT_TREE_LIFT=1` (runtime behaviour
    /// hook, A/B harness contract).
    ///
    /// **Extended to 8-bit integer layouts (GOAL_BEAT_CJXL wedge 1,
    /// 2026-06-12)**: the scoreboard's first full run measured lossless
    /// e5 on 8-bit graphics at +52..+557 % vs cjxl (plots: ours
    /// 59,804 B vs cjxl 9,096 B) — same mechanism, our off-until-e7
    /// schedule vs cjxl learning at every effort. Full-tree ceiling
    /// probe on the worst cells: plots −86.5 % (8,086 B, BEATS cjxl at
    /// the same wall), patents −70 %, web-screenshots −46 % at ~2.2×
    /// the (cheap) non-tree e5 wall. The 8-bit lift reuses the
    /// 16-bit-calibrated per-effort budgets and carries its own
    /// `JXL_NO_8BIT_TREE_LIFT` opt-out so the two lifts A/B
    /// independently.
    ///
    /// Returns `true` (and applies the budget caps to `profile`) when
    /// the lift fires.
    pub(crate) fn lift_integer_tree_learning(
        &self,
        layout: PixelLayout,
        pixels: u64,
        profile: &mut crate::effort::EffortProfile,
    ) -> bool {
        let is_int16_rgb = matches!(layout, PixelLayout::Rgb16 | PixelLayout::Rgba16);
        let is_int8 = matches!(
            layout,
            PixelLayout::Rgb8
                | PixelLayout::Rgba8
                | PixelLayout::Bgr8
                | PixelLayout::Bgra8
                | PixelLayout::Gray8
                | PixelLayout::GrayAlpha8
        );
        if !(is_int16_rgb || is_int8)
            || !(5..=6).contains(&self.effort)
            || self.tree_learning.is_some()
            || self.tree_learning()
            || self.faster_decoding >= 4
        {
            return false;
        }
        #[cfg(feature = "std")]
        if is_int16_rgb && std::env::var_os("JXL_NO_16BIT_TREE_LIFT").is_some() {
            return false;
        }
        #[cfg(feature = "std")]
        if is_int8 && std::env::var_os("JXL_NO_8BIT_TREE_LIFT").is_some() {
            return false;
        }
        // Per-effort budgets, shipped EXACTLY as measured on the 76-crop
        // sweep (/tmp-era data promoted to
        // benchmarks/hdr16_tree_lift_sweep_2026-06-12.tsv):
        // e5 "cheap": mean +6.4 % vs cjxl-e5 (was +17.1 %), tail capped
        //   +43 % (was +443 %), ~2.4x cjxl-e5 wall.
        // e6 "mid": mean -5.5 % vs cjxl-e5 — BEATS the reference — worse
        //   on only 6/76, tail +12 %, cjxl-e8-class wall.
        if self.effort == 5 {
            profile.tree_sample_fraction = 0.05;
            profile.tree_num_properties = 4;
            profile.tree_max_buckets = 32;
            // 8-bit keeps the default RCT trial count: capping to 1
            // regressed photos-png e5 by +19 % vs the no-tree path
            // (the off path's best-of-7 RCT is where photo bytes
            // live); 16-bit keeps the measured #72 cap.
            if is_int16_rgb {
                profile.nb_rcts_to_try = 1;
            }
            profile.wp_num_param_sets = 0;
        } else {
            // e6: size-adaptive sampling. Full mid-quality fraction
            // (0.25 — the config that BEAT cjxl-e5 by 5.5 % mean on the
            // 512^2 crop corpus) up to ~1.2M sampled pixels, decaying
            // hyperbolically above so 12 MP lands at frac 0.1 — the
            // fixed-budget mid config broke wall monotonicity there
            // (61-107 s > our e7). Props/buckets sit between the crop
            // winner (16/256) and the 12 MP wall point (6/64).
            let frac = (1_200_000.0 / pixels as f32).clamp(0.05, 0.25);
            profile.tree_sample_fraction = frac;
            profile.tree_num_properties = 10;
            profile.tree_max_buckets = 128;
            profile.nb_rcts_to_try = 3;
            profile.wp_num_param_sets = 1;
        }
        true
    }

    /// Resolve the effective patches enable flag, honoring
    /// `faster_decoding >= 2` (libjxl `enc_modular.cc:707` gates
    /// `FindBestPatchDictionary` on `decoding_speed_tier < 2`).
    pub(crate) fn effective_patches(&self) -> bool {
        if self.faster_decoding >= 2 {
            return false;
        }
        self.patches()
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
        if !self.has_internal_overrides() {
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
        // Store the sparse params (issue #80); resolved lazily in
        // `effective_profile` so the final effort wins regardless of
        // builder order.
        self.internal_overrides = Some(params);
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
    /// - **e9–13**: + LZ77 optimal (Viterbi DP)
    ///
    /// **e10** aligns with libjxl e10 (kGlacier): the MA-tree split
    /// threshold drops 89 → 75, admitting more splits (its other
    /// modular additions — global MA tree, per-leaf predictor search,
    /// no chunked encoding — our e9 already does). **e11/e12/e13 are
    /// our extensions** beyond libjxl (RFC#45 pick #1, renumbered +1 by
    /// the 2026-08-29 ladder shift): multi-seed tree learning fans out
    /// 2/16/16 seeded runs, and e11+ supersets libjxl e11 with the
    /// TectonicPlate per-image config trial — the whole frame is
    /// re-encoded under ~22 modular header/transform configurations
    /// (palette, channel-compact, group size, predictor, patches,
    /// sample density) at e10 search effort, the winner re-encoded at
    /// the full tier profile, smallest stream wins. Caller-explicit
    /// knobs are pinned across trials; streaming
    /// ([`LosslessEncoder`]) skips the trial (whole-image only), and
    /// explicit [`SectionedTrees::On`]/`Hybrid` skips it too.
    /// Bitstreams remain 100% spec-valid (djxl / jxl-rs / jxl-oxide
    /// decode unchanged).
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
    pub fn with_effort(mut self, effort: u8) -> Self {
        // Pure setter (issue #80): effort-derived knobs are `Option`
        // (resolved against `self.effort` in `effective_profile`) and
        // every other profile field is resolved from `self.effort`, so
        // changing the effort needs no rebuild and cannot clobber a
        // caller override regardless of call order.
        self.effort = effort;
        self
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
    /// Default per the effort schedule: true at effort >= 5.
    ///
    /// **Not yet consumed by the lossless modular path** (jxl-encoder#69):
    /// the patch-dictionary search currently runs only on the VarDCT
    /// pipeline, so this flag is stored for API-surface parity and takes
    /// effect the day lossless patch detection lands. Toggling it does
    /// not change lossless output bytes today.
    #[doc(hidden)]
    pub fn with_patches(mut self, enable: bool) -> Self {
        self.patches = Some(enable);
        self
    }

    /// Enable/disable ANS entropy coding (default: true).
    #[doc(hidden)]
    pub fn with_ans(mut self, enable: bool) -> Self {
        self.use_ans = Some(enable);
        self
    }

    /// Enable/disable squeeze (Haar wavelet) transform (default: false).
    ///
    /// Squeeze is disabled by default because tree learning provides better
    /// compression on both photos and screenshots. Squeeze can still be
    /// enabled via `.with_squeeze(true)` for experimentation.
    ///
    /// **MEASURED: ALWAYS LARGER OUTPUT** — squeeze+tree vs tree-only:
    /// +14.7 % on 1024² CLIC photos, +62 % on screenshots (imac_dark).
    /// Its purpose is progressive decoding, not compression; enable only
    /// when progressive lossless decode is the requirement.
    pub fn with_squeeze(mut self, enable: bool) -> Self {
        self.squeeze = enable;
        self
    }

    /// Enable/disable content-adaptive tree learning (default: from the
    /// effort profile — on at e7+). Explicitly calling this (either way)
    /// also opts out of the automatic 16-bit e5/e6 budgeted-tree lift
    /// (issue #72).
    #[doc(hidden)]
    pub fn with_tree_learning(mut self, enable: bool) -> Self {
        self.tree_learning = Some(enable);
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
    ///
    /// Effective sampling is **stride-quantized**: the gather walks
    /// every k-th pixel with `k = ceil(1 / f)`, so `f` rounds down to
    /// the nearest `1/k` — 0.65 and 0.55 sample 1-in-2 exactly like
    /// 0.5, and 0.4 samples 1-in-3 like 1/3. Every fraction in the
    /// table above lands on a distinct stride; overrides in
    /// `(0.5, 1.0)` are byte-identical to 0.5 (jxl-encoder#69).
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

    /// Enable/disable LZ77 backward references on modular token streams.
    /// Default follows the effort schedule (on at effort >= 7).
    ///
    /// **Currently inert on the lossless modular path** (jxl-encoder#69):
    /// the global-tree + per-group-section design cannot apply LZ77 to
    /// the combined stream without a histogram mismatch — the global ANS
    /// code would include LZ77 symbols the per-group sections don't emit
    /// (see the deliberate drop in `modular/section.rs`) — so the section
    /// writer ignores the flag. ICC payload compression and the squeeze
    /// multi-group path do honor LZ77. Toggling this does not change
    /// lossless output bytes today.
    #[doc(hidden)]
    pub fn with_lz77(mut self, enable: bool) -> Self {
        self.lz77 = Some(enable);
        self
    }

    /// Set the LZ77 match-search method. Default follows the effort
    /// schedule (Rle at effort <= 7, Greedy at e8, Optimal at e9+).
    /// Only meaningful where LZ77 is actually applied — see
    /// [`Self::with_lz77`] for the lossless-path caveat
    /// (jxl-encoder#69): on that path this is stored but unused today.
    #[doc(hidden)]
    pub fn with_lz77_method(mut self, method: Lz77Method) -> Self {
        self.lz77_method = Some(method);
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
    /// Consumed on BOTH lossless paths (issue #69 item 2): the
    /// single-group palette writer and the multi-group full-image
    /// palette gate in `modular/frame.rs` (nc >= 2; single-channel
    /// palette is ChannelCompact's domain and is governed by the
    /// channel-colours knobs instead).
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
            .unwrap_or_else(|| self.effort_schedule().use_ans)
    }

    /// Whether squeeze (Haar wavelet) transform is enabled.
    pub fn squeeze(&self) -> bool {
        self.squeeze
    }

    /// Whether content-adaptive tree learning is enabled.
    pub fn tree_learning(&self) -> bool {
        self.tree_learning
            .unwrap_or_else(|| self.effort_schedule().tree_learning)
    }

    /// Whether LZ77 backward references are enabled.
    pub fn lz77(&self) -> bool {
        self.lz77.unwrap_or_else(|| self.effort_schedule().lz77)
    }

    /// Current LZ77 method.
    pub fn lz77_method(&self) -> Lz77Method {
        self.lz77_method
            .unwrap_or_else(|| self.effort_schedule().lz77_method)
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
        // Conservative upper bound = the calibrated `max`. The previous
        // term-by-term model under-reported ~14x at e7+ (it modelled the
        // MA tree-learning working set as 8 B/px; measured ~440 in
        // 2026-06, ~95-140 after the August 2026 reductions — the band is
        // re-anchored per `heuristics.rs`). See [`Self::estimate_encode`].
        crate::heuristics::estimate_encode(
            width,
            height,
            layout.bytes_per_pixel() as u8,
            layout.has_alpha(),
            true,
            self.effort,
        )
        .map(|e| e.peak_memory_bytes_max)
    }

    /// Full calibrated resource estimate (min / typical / max peak
    /// memory, plus coarse time and output size) for a lossless encode at
    /// these settings. Mirrors the zen per-codec pattern
    /// ([`crate::heuristics::EncodeEstimate`]). `None` only on dimension
    /// overflow.
    ///
    /// This is the WHOLE-IMAGE tree-learning band, thread-independent. An
    /// encode that runs the sectioned local-tree mode
    /// ([`SectionedTrees`] `On`, or `Auto` under memory pressure / at
    /// effort <= 7 with several workers) peaks well below it (measured
    /// 2026-08-27, `benchmarks/jxl_sectioned_mem_2026-08-27.tsv`); the
    /// pre-flight and [`EncodeStats::estimated_peak_bytes`] use that
    /// sectioned estimate, so a cap sized from this method is
    /// conservative for the default path.
    pub fn estimate_encode(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Option<crate::heuristics::EncodeEstimate> {
        crate::heuristics::estimate_encode(
            width,
            height,
            layout.bytes_per_pixel() as u8,
            layout.has_alpha(),
            true,
            self.effort,
        )
    }

    /// Whether patches (dictionary-based repeated pattern detection) are enabled.
    pub fn patches(&self) -> bool {
        self.patches
            .unwrap_or_else(|| self.effort_schedule().patches)
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
    /// True when a sweep/picker has pinned `__expert` internal-param
    /// overrides (issue #80) — used to skip the per-image auto-adapter
    /// so the pinned value survives.
    #[cfg(feature = "__expert")]
    fn has_internal_overrides(&self) -> bool {
        self.internal_overrides.is_some()
    }

    #[cfg(not(feature = "__expert"))]
    fn has_internal_overrides(&self) -> bool {
        false
    }

    /// The resolved override profile (schedule + internal-param overrides)
    /// when a sweep has pinned `__expert` params; `None` otherwise. Used by
    /// `validate` to range-check pinned values.
    #[cfg(feature = "__expert")]
    pub(crate) fn overridden_profile(&self) -> Option<crate::effort::EffortProfile> {
        self.internal_overrides
            .as_ref()
            .map(|_| self.effective_profile())
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
            relative_to_max_display: None,
            linear_below: None,
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
        encode_animation_lossless(self, width, height, layout, animation, frames, None).at()
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
        encode_animation_lossless(self, width, height, layout, animation, frames, Some(limits)).at()
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
    // The [`LosslessConfig::effort`] level is honoured on the JPEG transcode
    // path: at `effort >= 9` (libjxl `speed_tier <= kTortoise`) the AC code
    // uses kBest pair-merge histogram clustering (-0.27 % vs default-effort
    // on a 10-file 2026-05-28 corpus). Effort 0-8 produces byte-identical
    // output to the pre-2026-05-28 default-effort transcode. libjxl also
    // enables kBest uint-method + RLE LZ77 at e9; both are currently
    // DEFAULT-OFF on our path (uint_method regresses by +0.5 % due to a
    // divergence in `optimize_uint_configs_best_from_freqs`, LZ77 global
    // savings threshold doesn't pass on JPEG AC streams). Env hooks
    // `JPEG_E9_FORCE_UINT_OPT=1` / `JPEG_E9_FORCE_LZ77=1` re-enable each for
    // future investigation. Other `LosslessConfig` settings (mode, patches,
    // lossy_palette, etc.) do not affect the transcode path.

    /// Attach resource [`Limits`] consulted by the JPEG-transcode path
    /// ([`Self::encode_jpeg_transcode`] /
    /// [`Self::encode_jpeg_transcode_codestream`]).
    ///
    /// The transcode parser reads [`Limits::max_pixels`] as the pre-flight
    /// `width × height` cap applied to the untrusted SOF dimensions before
    /// any coefficient buffer is allocated. When unset (or no `Limits` is
    /// attached), the secure default
    /// [`Limits::DEFAULT_MAX_JPEG_TRANSCODE_PIXELS`] (120 MP) applies; a
    /// trusted batch caller can raise it (or pass
    /// [`Limits::with_max_pixels`]`(u64::MAX)` to opt out), and a
    /// hostile-input proxy can tighten it.
    ///
    /// Only the `max_pixels` field is consulted today; the rest of the
    /// transcode-path [`Limits`] wiring (a full `MemoryBudget`) is tracked in
    /// issue #77.
    ///
    /// Requires the `jpeg-reencoding` cargo feature.
    #[cfg(feature = "jpeg-reencoding")]
    pub fn with_limits(mut self, limits: &Limits) -> Self {
        self.limits = Some(limits.clone());
        self
    }

    /// The [`Limits`] attached via [`Self::with_limits`], if any.
    ///
    /// Requires the `jpeg-reencoding` cargo feature.
    #[cfg(feature = "jpeg-reencoding")]
    pub fn limits(&self) -> Option<&Limits> {
        self.limits.as_ref()
    }

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
    /// Returns the complete JXL container bytes (signature box, codestream,
    /// JBRD box). Typical ratio: ~80% of the original JPEG bytes for
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
        // 2026-05-28: effort gates kBest pair-merge clustering at e>=8 and LZ77
        // (greedy at e=8, optimal at e>=9) on the AC code. See
        // `crate::jpeg::encode_jpeg_to_jxl_container_with_effort`.
        // Pre-flight SOF pixel cap from the attached `Limits::max_pixels`
        // (default 120 MP — `Limits::DEFAULT_MAX_JPEG_TRANSCODE_PIXELS` —
        // when unset; see `Self::with_limits`).
        // Pre-flight SOF pixel cap + per-encode memory budget from the attached
        // `Limits` (default 120 MP / the lossless 8 GiB default when unset; see
        // `Self::with_limits`). DoS protection is default-on for this untrusted-
        // bytes path, like the pixel path (#77 item 1).
        let max_pixels = self.limits.as_ref().and_then(|l| l.max_pixels());
        let budget = self.transcode_budget();
        let jpeg = crate::jpeg::read_jpeg_with_stop(jpeg_bytes, max_pixels, None, Some(&budget))
            .map_err(|e| at(EncodeError::from(e)))?;
        crate::jpeg::encode_jpeg_to_jxl_container_with_effort_stop(
            &jpeg,
            self.effort,
            None,
            Some(&budget),
        )
        .map_err(|e| at(EncodeError::from(e)))
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
        // 2026-05-28: see `encode_jpeg_transcode` for the effort gating.
        // Pre-flight SOF pixel cap from the attached `Limits::max_pixels`
        // (default 120 MP — `Limits::DEFAULT_MAX_JPEG_TRANSCODE_PIXELS` —
        // when unset; see `Self::with_limits`).
        let max_pixels = self.limits.as_ref().and_then(|l| l.max_pixels());
        let budget = self.transcode_budget();
        let jpeg = crate::jpeg::read_jpeg_with_stop(jpeg_bytes, max_pixels, None, Some(&budget))
            .map_err(|e| at(EncodeError::from(e)))?;
        crate::jpeg::encode_jpeg_to_jxl_with_effort_stop(&jpeg, self.effort, None, Some(&budget))
            .map_err(|e| at(EncodeError::from(e)))
    }

    /// Cancellable variant of [`Self::encode_jpeg_transcode`].
    ///
    /// Polls `stop` (an [`enough::Stop`] token) at coarse boundaries — entry,
    /// the zenjpeg coefficient decode, and per-group / pre-entropy during JXL
    /// encoding — returning [`EncodeError::Cancelled`] if cancellation is
    /// requested. Output is byte-identical to [`Self::encode_jpeg_transcode`]
    /// when the token never fires (e.g. an [`Unstoppable`] token); each poll
    /// is a cheap check skipped entirely on the non-stop path.
    ///
    /// This is the untrusted-input cancellation hook from issue #77 item 2 —
    /// a server can abort a slow transcode of an oversized JPEG.
    ///
    /// Requires the `jpeg-reencoding` cargo feature.
    #[cfg(feature = "jpeg-reencoding")]
    #[track_caller]
    pub fn encode_jpeg_transcode_with_stop(
        &self,
        jpeg_bytes: &[u8],
        stop: &dyn Stop,
    ) -> Result<Vec<u8>> {
        let max_pixels = self.limits.as_ref().and_then(|l| l.max_pixels());
        let budget = self.transcode_budget();
        let jpeg =
            crate::jpeg::read_jpeg_with_stop(jpeg_bytes, max_pixels, Some(stop), Some(&budget))
                .map_err(|e| at(EncodeError::from(e)))?;
        crate::jpeg::encode_jpeg_to_jxl_container_with_effort_stop(
            &jpeg,
            self.effort,
            Some(stop),
            Some(&budget),
        )
        .map_err(|e| at(EncodeError::from(e)))
    }

    /// Cancellable variant of [`Self::encode_jpeg_transcode_codestream`].
    ///
    /// See [`Self::encode_jpeg_transcode_with_stop`] for the polling contract
    /// and byte-identity guarantee.
    ///
    /// Requires the `jpeg-reencoding` cargo feature.
    #[cfg(feature = "jpeg-reencoding")]
    #[track_caller]
    pub fn encode_jpeg_transcode_codestream_with_stop(
        &self,
        jpeg_bytes: &[u8],
        stop: &dyn Stop,
    ) -> Result<Vec<u8>> {
        let max_pixels = self.limits.as_ref().and_then(|l| l.max_pixels());
        let budget = self.transcode_budget();
        let jpeg =
            crate::jpeg::read_jpeg_with_stop(jpeg_bytes, max_pixels, Some(stop), Some(&budget))
                .map_err(|e| at(EncodeError::from(e)))?;
        crate::jpeg::encode_jpeg_to_jxl_with_effort_stop(
            &jpeg,
            self.effort,
            Some(stop),
            Some(&budget),
        )
        .map_err(|e| at(EncodeError::from(e)))
    }

    /// Build the per-encode [`crate::budget::MemoryBudget`] for the
    /// JPEG-transcode path from the attached [`Limits`]: the explicit
    /// [`Limits::max_memory_bytes`] cap if set, else the lossless default
    /// [`Limits::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS`] (transcode emits a lossless
    /// bitstream); plus the [`Limits::fallible_alloc`] allocation policy.
    #[cfg(feature = "jpeg-reencoding")]
    fn transcode_budget(&self) -> alloc::sync::Arc<crate::budget::MemoryBudget> {
        let cap = self
            .limits
            .as_ref()
            .and_then(|l| l.max_memory_bytes())
            .unwrap_or(Limits::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS);
        let fallible = self.limits.as_ref().is_some_and(|l| l.fallible_alloc());
        crate::budget::MemoryBudget::with_alloc_policy(cap, fallible)
    }
}

// ── EncoderMode ──────────────────────────────────────────────────────────────

mod strategy;
pub use strategy::*;
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
    /// Effort-derived knob: `None` inherits the effort schedule's
    /// `use_ans`; `Some(v)` is a caller override (set via
    /// [`Self::with_ans`]). Resolved in [`Self::effective_profile`] /
    /// [`Self::ans`]. The `Option` *is* the touched-bit that the
    /// pre-#80 `use_ans_explicit` flag tracked by hand — so
    /// [`Self::with_effort`] is a pure setter and the builder chain is
    /// order-independent by construction.
    use_ans: Option<bool>,
    /// Effort-derived knob (`None` inherits the schedule's `gaborish`).
    /// See [`Self::use_ans`] for the Option-as-touched-bit pattern.
    gaborish: Option<bool>,
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
    /// Effort-derived knob (`None` inherits the schedule's
    /// `error_diffusion`). See [`Self::use_ans`] for the pattern.
    error_diffusion: Option<bool>,
    /// Effort-derived knob (`None` inherits the schedule's
    /// `pixel_domain_loss`). See [`Self::use_ans`] for the pattern.
    pixel_domain_loss: Option<bool>,
    /// Effort-derived knob (`None` inherits the schedule's `lz77`).
    /// See [`Self::use_ans`] for the pattern.
    lz77: Option<bool>,
    /// Effort-derived knob (`None` inherits the schedule's
    /// `lz77_method`). See [`Self::use_ans`] for the pattern.
    lz77_method: Option<Lz77Method>,
    force_strategy: Option<u8>,
    max_strategy_size: Option<u8>,
    /// Effort-derived knob (`None` inherits the per-image profile's
    /// `patches`; `Some(v)` is a caller pin via [`Self::with_patches`]).
    /// `patches.is_some()` replaces the pre-#80 `patches_explicit`
    /// flag. See [`Self::use_ans`] for the pattern and
    /// [`Self::effective_patches`] for resolution.
    patches: Option<bool>,
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
    /// Effort-derived knob (`None` inherits
    /// [`crate::effort::EffortProfile::auto_splines_default`], i.e.
    /// `effort >= 8`; `Some(v)` is a caller override via
    /// [`Self::with_auto_splines`]). `auto_splines.is_some()` replaces
    /// the pre-#80 `auto_splines_explicit` flag. See [`Self::use_ans`].
    auto_splines: Option<bool>,
    progressive: ProgressiveMode,
    lf_frame: bool,
    /// Effort-derived knob (`None` inherits the schedule's
    /// `butteraugli_iters`; `Some(n)` is a caller override via
    /// [`Self::with_butteraugli_iters`]). `is_some()` replaces the
    /// pre-#80 `butteraugli_iters_explicit` flag. See [`Self::use_ans`].
    #[cfg(feature = "butteraugli-loop")]
    butteraugli_iters: Option<u32>,
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
    /// Sweep / picker hook (`__expert`): sparse internal-param overrides
    /// stored as the params themselves (NOT an eagerly-resolved profile)
    /// and applied lazily in [`Self::effective_profile`] against the
    /// CURRENT effort — so `with_internal_params(_).with_effort(_)`
    /// resolves correctly regardless of builder order (issue #80).
    #[cfg(feature = "__expert")]
    internal_overrides: Option<crate::effort::LossyInternalParams>,
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
    /// W44-222 Tier-2 knobs (5 high-level interpretable knobs that
    /// expand to a full 6-param [`crate::tuning::runtime::RuntimeTuning`]
    /// override). When `Some`, [`Self::encode`] / `encode_inner` calls
    /// [`crate::tuning::runtime::install_or_check_idempotent`] with the
    /// expanded `RuntimeTuning` before encoding starts. Default `None`
    /// → no override is installed → the production code path stays
    /// const-fold-friendly and every existing hash-lock fixture
    /// remains byte-identical.
    ///
    /// Gated on the `tuning-override` cargo feature; the underlying
    /// types only exist under that feature.
    ///
    /// See [`Self::with_knobs`].
    #[cfg(feature = "tuning-override")]
    tier2_knobs: Option<crate::tuning::coupling::Tier2Knobs>,
    /// Multi-metric Phase 0 (RFC #3, 2026-05-25): which perceptual metric
    /// drives the buttloop's per-iter compare. See [`PerceptualMetric`]
    /// for the variants and `docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`
    /// for the full design.
    ///
    /// Default: [`PerceptualMetric::Butteraugli`]. The default produces
    /// byte-identical output to the pre-Phase-0
    /// `gpu_butteraugli`/`cvvdp_loop = None` shape on every existing
    /// hash-lock fixture.
    ///
    /// See [`Self::with_perceptual_metric`] +
    /// [`Self::resolve_perceptual_metric`].
    #[cfg(feature = "butteraugli-loop")]
    perceptual_metric: PerceptualMetric,
    /// Multi-metric Phase 0 (RFC #3, 2026-05-25): compute-device
    /// preference for the active perceptual metric. See
    /// [`PerceptualDevice`] for the variants.
    ///
    /// Default: [`PerceptualDevice::Auto`]. With the
    /// [`PerceptualMetric::Butteraugli`] default, `Auto` resolves to GPU
    /// when the `gpu-butteraugli` cargo feature is compiled in (matches
    /// the W44-PHASE3-B5-flip butteraugli default) and CPU otherwise.
    /// With [`PerceptualMetric::Cvvdp`], `Auto` resolves to GPU when
    /// `cvvdp-loop` is compiled (and CUDA inits), CPU when only
    /// `cvvdp-loop-cpu` is compiled, butteraugli fallback otherwise.
    ///
    /// See [`Self::with_perceptual_device`] +
    /// [`Self::resolve_perceptual_device`].
    #[cfg(feature = "butteraugli-loop")]
    perceptual_device: PerceptualDevice,
    /// Multi-metric Phase 0 (RFC #3, 2026-05-25): caller override for the
    /// per-distance target table.
    ///
    /// When `None` (default), the metric's built-in calibration table
    /// (`vardct/cvvdp_targets.rs` for cvvdp; identity pass-through for
    /// butteraugli) drives the buttloop's per-iter convergence target.
    /// When `Some(score)`, the loop targets that score directly in the
    /// metric's score-direction (smaller=better).
    ///
    /// Use for calibrating against a non-standard quality requirement
    /// (e.g. matching a specific reference encoder's output). Default
    /// `None` is the right choice for ~all production callers.
    #[cfg(feature = "butteraugli-loop")]
    perceptual_target_score: Option<f32>,

    /// cvvdp-fork Phase 8d (2026-05-25, RFC
    /// `docs/RFC_CVVDP_PHASE8_PARETO_TARGETING.md` §3.3 Intervention C):
    /// post-convergence bytes-tighten exit pass on the cvvdp seed loop.
    ///
    /// After the inner seed loop converges quant_field to satisfy the
    /// cvvdp metric target, run a batched multiplicative bump pass that
    /// LOOSENS qac while the score still satisfies `target * (1 + ε)`.
    /// Gives back bytes the converged state had headroom for.
    ///
    /// Tri-state:
    /// - `None` (default): "on when both the `cvvdp-loop-tighten` cargo
    ///   feature is compiled AND [`Self::cvvdp_loop`] resolves to true".
    ///   When the feature is OFF or the cvvdp loop is OFF, this resolves
    ///   to `false` (pass is skipped, byte-identical to pre-Phase-8d).
    /// - `Some(true)`: explicit opt-in. Same effective behaviour as `None`
    ///   when both the feature and cvvdp_loop are on; explicit kept for
    ///   symmetry with [`Self::cvvdp_loop`] and for documentation in
    ///   caller code.
    /// - `Some(false)`: explicit opt-out. Skips the tighten pass even when
    ///   the feature and cvvdp_loop are both on. Use for measuring the
    ///   pre-Phase-8d byte-vs-quality tradeoff in benches.
    ///
    /// **NEVER fires on the butteraugli loop.** The butteraugli per-block
    /// reducer is already calibrated to the W44 cost-model gates;
    /// loosening it post-convergence over-tightens the bytes/quality
    /// tradeoff. The accept-bound + mean_qf seed-picker already encodes
    /// the natural "biggest qf among qualifying seeds" preference for
    /// butteraugli.
    ///
    /// **Field always present** so hash-lock fixtures don't depend on
    /// the `cvvdp-loop-tighten` cargo feature; consulted only inside
    /// the cvvdp seed loop's post-convergence section, gated on both the
    /// cargo feature AND [`Self::resolve_perceptual_metric`] returning
    /// [`PerceptualMetric::Cvvdp`].
    ///
    /// See [`Self::with_cvvdp_bytes_tighten`] and
    /// [`Self::resolve_cvvdp_bytes_tighten`] for the dispatch matrix.
    #[cfg(feature = "butteraugli-loop")]
    cvvdp_bytes_tighten: Option<bool>,

    /// Phase 1 display-config backfill (RFC `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`,
    /// 2026-05-25): target display config for cvvdp scoring. See
    /// [`DisplayConfig`] for variants + the Phase 1 geometry caveat.
    ///
    /// Default [`DisplayConfig::WebSdr80`] keeps every existing hash-lock
    /// fixture byte-identical (the variant maps to
    /// `cvvdp_gpu::params::DisplayModel::STANDARD_4K`, which is what
    /// `CvvdpParams::default()` already used pre-Phase-1).
    ///
    /// **Has no effect** when the resolved [`PerceptualMetric`] is
    /// [`PerceptualMetric::Butteraugli`] (default) or
    /// [`PerceptualMetric::Zensim`] — display config only routes through
    /// the cvvdp scoring path. The field is always present so callers
    /// can set it ahead of switching the metric without needing a
    /// re-construction step.
    ///
    /// **Strict cjxl-parity invariant**: when
    /// [`Self::strategy`] is [`EncoderStrategy::Libjxl`], the resolved
    /// target display is forced to `WebSdr80` regardless of this field
    /// (matches the W44-126 pattern for `with_perceptual_metric`). See
    /// [`Self::resolve_target_display`].
    ///
    /// Field always present so hash-lock fixtures don't depend on the
    /// cvvdp cargo features.
    target_display: DisplayConfig,
}

/// Multi-metric Phase 0 (RFC #3, 2026-05-25): which perceptual metric
/// drives the iterative quantization loop when
/// [`LossyConfig::butteraugli_iters`] > 0.
///
/// All metrics share the
/// [`PerceptualBackend`](crate::vardct::perceptual_backend::PerceptualBackend)
/// trait surface — a per-cell `set_reference` followed by per-iter
/// `compare_with_reference` calls that return a scalar score + per-pixel
/// diffmap. The metric choice is fixed for the full encode; per-iter
/// switching is not supported.
///
/// **Default**: [`Self::Butteraugli`]. The non-default metrics are
/// opt-in per their respective cargo features ([`Self::Cvvdp`] requires
/// `cvvdp-loop` and/or `cvvdp-loop-cpu`; [`Self::Zensim`] requires
/// `zensim-loop` and/or `zensim-loop-gpu`).
///
/// **`EncoderStrategy::Libjxl` invariant**: when the active strategy is
/// [`EncoderStrategy::Libjxl`], the resolved metric is forced to
/// [`Self::Butteraugli`] regardless of this field (W44-126 strict
/// cjxl-parity invariant). See [`LossyConfig::resolve_perceptual_metric`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PerceptualMetric {
    /// Butteraugli (max-norm score; smaller=better; calibrated against
    /// libjxl reference encoder). The default. CPU always available via
    /// the `butteraugli-loop` cargo feature (default on). GPU
    /// acceleration via the `gpu-butteraugli` cargo feature.
    #[default]
    Butteraugli,

    /// CVVDP (Mantiuk et al. 2024; JOD-direction normalized to
    /// butteraugli-direction at trait boundary). Opt-in via the
    /// `cvvdp-loop` (GPU) or `cvvdp-loop-cpu` (CPU) cargo features.
    /// See `docs/RFC_CVVDP_FORK.md`.
    Cvvdp,

    /// zensim (multi-scale XYB SSIM + edge + HF + trained per-codec
    /// affine; native score lives in [0, 100] with 100 = identical, so
    /// the backend's trait boundary maps it to butteraugli-direction
    /// via `(100.0 - score).clamp(0.0, 100.0)`). Opt-in via the
    /// `zensim-loop` (CPU) or `zensim-loop-gpu` (GPU) cargo features.
    /// See `docs/RFC_ZENSIM_FORK_PLAN.md` + `docs/RFC_ZENSIM_BUTTLOOP_AUDIT.md`.
    ///
    /// **Calibration status:** the per-distance target table
    /// (`vardct/zensim_targets.rs`) is wired in, but the per-block reducer
    /// premultiplier (`K_TILE_NORM`) still carries the butteraugli-parity
    /// placeholder — the zensim-specific refit ("Phase 8-zensim") is
    /// pending, and the 2026-05-24 tracking sweep measured this calibration
    /// as over-loose (65.3 % Pareto-front vs butteraugli's 67.7 %). Opting
    /// in works end-to-end but is not yet Pareto-tuned.
    Zensim,
}

/// Multi-metric Phase 0 (RFC #3, 2026-05-25): compute-device preference
/// for the active perceptual metric.
///
/// Both `Cpu` and `Gpu` are opt-in subject to their respective cargo
/// features; `Auto` (default) prefers GPU when both backends are
/// compiled in AND the CUDA runtime initialises successfully, falling
/// back to CPU otherwise.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PerceptualDevice {
    /// "Prefer GPU when available; fall back to CPU otherwise." Matches
    /// the W44-PHASE3-B5-flip butteraugli default (GPU when
    /// `gpu-butteraugli` is compiled, else CPU).
    #[default]
    Auto,

    /// Force CPU. Required for reproducibility (CPU paths have no GPU
    /// reduction-order variance — see W44-RECON-DEEP/A7).
    Cpu,

    /// Force GPU. Silently falls back to CPU (per the metric's cargo
    /// features) if CUDA is unavailable — the encoder never panics on
    /// missing CUDA driver.
    Gpu,
}

/// Target display configuration for CVVDP scoring (Phase 1 display-config
/// backfill, RFC `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`, 2026-05-25).
///
/// Different displays cause different perceived quality at the same
/// encoded bytes — HDR TVs with high peak luminance amplify dark-region
/// artifacts vs SDR monitors; phones with EDR boost render SDR content
/// at higher absolute luminance than the encoder's default
/// 200 cd/m² STANDARD_4K assumption.
///
/// When [`PerceptualMetric::Cvvdp`] is active AND a non-default
/// `target_display` is set, the cvvdp backend constructs the matching
/// [`cvvdp_gpu::params::DisplayModel`](https://docs.rs/cvvdp-gpu) for
/// scoring AND the per-distance calibration table at
/// `vardct/cvvdp_targets.rs` switches to the per-display row. Default
/// behaviour ([`Self::WebSdr80`]) preserves backwards compatibility
/// byte-for-byte on every existing hash-lock fixture.
///
/// **Phase 2 (2026-05-26)** wires viewing geometry too. Upstream
/// `cvvdp_gpu` master `a4994bb4`+ exposes `CvvdpOpaque::new_with_geometry`,
/// so the GPU backend now constructs the underlying scorer with both
/// `DisplayModel` AND `DisplayGeometry`. `Phone` drives PPD ≈ 95
/// (handheld 30 cm); `Tv` drives `DisplayGeometry::LG_OLED_2026`;
/// `WebSdr80` drives `DisplayGeometry::STANDARD_4K` (75.4 PPD —
/// byte-identical to the pre-Phase-2 dispatch because that was already
/// the upstream default). The Phase 2 per-distance target tables at
/// `vardct/cvvdp_targets.rs` were re-measured against the 1,134-cell
/// tracking corpus under each `DisplayConfig`; Phone+Tv now ship
/// measured per-distance arrays rather than Phase 1's uniform
/// multipliers.
///
/// **Has no effect** when [`PerceptualMetric::Butteraugli`] (default)
/// or [`PerceptualMetric::Zensim`] is the active metric — display config
/// only routes through the cvvdp scoring path.
///
/// **Strict cjxl-parity invariant**: when the active strategy is
/// [`EncoderStrategy::Libjxl`], the resolved target display is forced
/// to [`Self::WebSdr80`] regardless of this field (matches the W44-126
/// pattern for `with_perceptual_metric`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DisplayConfig {
    /// SDR web content on a typical 4K desktop monitor (200 cd/m², sRGB
    /// EOTF, BT.709 primaries, 250 lux ambient, 75 PPD at 30″ / 0.75 m).
    /// Matches the pre-Phase-1 single-table baseline
    /// (`cvvdp_gpu::params::DisplayModel::STANDARD_4K`). Used when no
    /// explicit target is set.
    ///
    /// The variant name keeps the "SDR + 80-cd-target" framing used in
    /// internet codec discussions even though the underlying upstream
    /// preset is `STANDARD_4K` at 200 cd/m² — the perceptual model
    /// effectively targets ~80 cd/m² mean signal level on diffuse
    /// surfaces at this peak.
    #[default]
    WebSdr80,

    /// 2025-flagship phone class viewing SDR content with EDR / HBM
    /// auto-brightness boost (1000 cd/m² effective sustained peak,
    /// sRGB EOTF since the signal is still SDR, Display-P3 primaries,
    /// 200 lux indoor bright ambient).
    ///
    /// Captures the typical "user looking at a JXL photo on iPhone /
    /// Samsung Galaxy / Pixel in a bright environment" case. The
    /// custom 1000 cd/m² peak is below the panel's 1590-2000 nit HDR
    /// peak because EDR sustained output for SDR content peaks around
    /// 1000 nits, not at HDR-mode peak. Custom-built via
    /// [`cvvdp_gpu::params::DisplayModel::compute_y_refl`] because the
    /// upstream `IPHONE_14_PRO_HDR` preset uses HLG/BT.2020 (wrong shape
    /// for SDR-on-HDR content).
    Phone,

    /// 2026 flagship HDR TV viewing HDR PQ content
    /// ([`cvvdp_gpu::params::DisplayModel::LG_OLED_2026_HDR_PQ`]:
    /// 3000 cd/m², PQ EOTF, BT.2020 primaries, OLED contrast,
    /// 5 lux dim viewing). Represents LG G5/C5-class and Sony A95L-class
    /// panels with native HDR content.
    Tv,
}

impl DisplayConfig {
    /// Construct the matching [`cvvdp_gpu::params::DisplayModel`] for
    /// this display config. Used by the cvvdp backend to populate
    /// `CvvdpParams.display` at construction.
    ///
    /// Requires the `cvvdp-loop` cargo feature.
    #[cfg(feature = "cvvdp-loop")]
    #[must_use]
    pub fn display_model(self) -> cvvdp_gpu::params::DisplayModel {
        use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries};
        match self {
            DisplayConfig::WebSdr80 => DisplayModel::STANDARD_4K,
            DisplayConfig::Phone => DisplayModel {
                y_peak: 1000.0,
                // OLED contrast — matches the IPHONE_14_PRO_HDR minimum
                // (0.0004) bumped slightly for the sustained-EDR regime.
                y_black: 0.0005,
                // 200 lux ambient × 0.005 reflectivity / π — handheld
                // bright-indoor (corridor / cafe / outdoor shade).
                y_refl: DisplayModel::compute_y_refl(200.0, 0.005),
                // SDR signal even though the panel is HDR-capable: phone
                // OS tone-maps SDR content into the HDR pipeline.
                eotf: Eotf::Srgb,
                primaries: Primaries::DisplayP3,
                e_ambient_lux: 200.0,
                k_refl: 0.005,
            },
            DisplayConfig::Tv => DisplayModel::LG_OLED_2026_HDR_PQ,
        }
    }

    /// Construct the matching [`cvvdp_gpu::params::DisplayGeometry`]
    /// for this display config. Used by both the CPU cvvdp backend
    /// ([`cvvdp_cpu::Cvvdp::with_geometry`]) and, as of Phase 2
    /// (2026-05-26), the GPU cvvdp backend via
    /// `cvvdp_gpu::CvvdpOpaque::new_with_geometry` (upstream
    /// `cvvdp_gpu` master `a4994bb4`+).
    ///
    /// Requires the `cvvdp-loop` cargo feature (transitively pulls in
    /// `cvvdp_gpu::params::DisplayGeometry`).
    #[cfg(feature = "cvvdp-loop")]
    #[must_use]
    pub fn display_geometry(self) -> cvvdp_gpu::params::DisplayGeometry {
        use cvvdp_gpu::params::DisplayGeometry;
        match self {
            DisplayConfig::WebSdr80 => DisplayGeometry::STANDARD_4K,
            DisplayConfig::Phone => {
                // iPhone 16 Pro Max class (2868×1320), 30 cm handheld
                // (~11.8″ viewing distance), 6.9″ diagonal. PPD ≈ 95
                // at this geometry — represents the "user holding phone
                // at arm's reach" case, NOT the upstream IPHONE_14_PRO
                // 20″ viewing distance (more generous than typical).
                DisplayGeometry::from_inches(2868, 1320, 11.8, 6.9)
            }
            DisplayConfig::Tv => DisplayGeometry::LG_OLED_2026,
        }
    }
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
            use_ans: None,
            gaborish: None,
            adaptive_gaborish: false,
            noise: false,
            photon_noise_iso: None,
            manual_noise_lut: None,
            quant_ac_rescale: None,
            original_distance: None,
            denoise: false,
            error_diffusion: None,
            pixel_domain_loss: None,
            lz77: None,
            lz77_method: None,
            force_strategy: None,
            max_strategy_size: None,
            patches: None,
            dot_detection: true, // refs #19; default-on to mirror libjxl Override::kDefault (gated effort>=7 && d>=3.0)
            simplify_invisible: true,
            center_first: false,
            resampling: 1,
            resampling_explicit: false,
            auto_resampling: true,
            already_downsampled: false,
            splines: None,
            // `None` inherits the effort-derived auto-splines default
            // (resolved in `effective_profile`); `with_auto_splines`
            // pins `Some(v)`.
            auto_splines: None,
            progressive: ProgressiveMode::Single,
            lf_frame: false,
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: None,
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
            #[cfg(feature = "__expert")]
            internal_overrides: None,
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
            #[cfg(feature = "tuning-override")]
            tier2_knobs: None,
            // Multi-metric Phase 0 (RFC #3, 2026-05-25): default metric
            // is butteraugli (RFC §5.2 — calibration depth, Pareto
            // position, predictability cascade across the imageflow
            // ecosystem). EncoderStrategy::Libjxl forces this at the
            // resolver layer regardless of caller override (W44-126
            // strict cjxl-parity invariant).
            #[cfg(feature = "butteraugli-loop")]
            perceptual_metric: PerceptualMetric::default(),
            // Multi-metric Phase 0 (RFC #3, 2026-05-25): default device
            // is Auto. For butteraugli, `Auto` resolves to GPU when the
            // `gpu-butteraugli` cargo feature is compiled in
            // (W44-PHASE3-B5-flip parity — measured median 1.107× wall
            // speedup at byte-parity on the 38-cell sweep), CPU
            // otherwise. For cvvdp, `Auto` follows the metric-internal
            // dispatch matrix (GPU first when `cvvdp-loop` is compiled
            // and CUDA inits, CPU when only `cvvdp-loop-cpu` is
            // compiled, silent butteraugli fallback otherwise).
            //
            // Every existing hash-lock fixture stays byte-identical
            // because the default feature set does NOT include
            // `gpu-butteraugli` — so `Auto` resolves to CPU
            // butteraugli, identical to the pre-Phase-0
            // `gpu_butteraugli = false` resolution.
            #[cfg(feature = "butteraugli-loop")]
            perceptual_device: PerceptualDevice::default(),
            // Multi-metric Phase 0 (RFC #3 §2.2): default `None` ≡
            // "use the metric's built-in target table". For butteraugli
            // that's the identity pass-through of `target_distance`; for
            // cvvdp that's `cvvdp_target_score_for_distance`.
            #[cfg(feature = "butteraugli-loop")]
            perceptual_target_score: None,
            // cvvdp-fork Phase 8d (2026-05-25): default `None` ≡ "on
            // when both `cvvdp-loop-tighten` cargo feature is compiled
            // AND `resolve_perceptual_metric()` returns Cvvdp". When the
            // feature is OFF or cvvdp is not the active metric, this
            // resolves to false and hash-locks stay byte-identical.
            #[cfg(feature = "butteraugli-loop")]
            cvvdp_bytes_tighten: None,
            // Phase 1 display-config backfill (2026-05-25): default
            // `WebSdr80` maps to `cvvdp_gpu::params::DisplayModel::STANDARD_4K`
            // — bit-identical to the pre-Phase-1 cvvdp scoring shape.
            // Every existing hash-lock fixture stays byte-identical.
            target_display: DisplayConfig::default(),
        }
    }

    /// Resolve the effective [`EffortProfile`]: the override if set,
    /// otherwise the standard profile derived from effort + mode. The
    /// `faster_decoding` knob is applied last (libjxl ordering — the
    /// speed-tier gates fire AFTER effort defaults are computed).
    /// Base effort+mode schedule used to resolve the effort-derived
    /// `Option` knobs (issue #80). `None` knobs inherit this; `Some(v)`
    /// overrides win. Does NOT apply `faster_decoding` or per-image
    /// adaptation — it is the bare "what would the schedule pick"
    /// baseline the getters read.
    fn effort_schedule(&self) -> crate::effort::EffortProfile {
        crate::effort::EffortProfile::lossy(self.effort, self.mode)
    }

    pub(crate) fn effective_profile(&self) -> crate::effort::EffortProfile {
        let mut p = crate::effort::EffortProfile::lossy(self.effort, self.mode);
        // Sweep/picker internal-param overrides (issue #80): applied
        // lazily against the CURRENT effort.
        #[cfg(feature = "__expert")]
        if let Some(ip) = self.internal_overrides.clone() {
            ip.apply_to(&mut p);
        }
        // Sparse-override resolution (issue #80): apply the
        // effort-derived knob overrides on top of the schedule. `None`
        // inherits, `Some(v)` overrides. This is the single resolution
        // point; getters + the encoder read through it (or the matching
        // getter), so the value cannot diverge from `with_effort` call
        // order. `auto_splines` is resolved in its getter (it is
        // `auto_splines_default(effort)`-derived, not an `EffortProfile`
        // field).
        if let Some(v) = self.use_ans {
            p.use_ans = v;
        }
        if let Some(v) = self.gaborish {
            p.gaborish = v;
        }
        if let Some(v) = self.error_diffusion {
            p.error_diffusion = v;
        }
        if let Some(v) = self.pixel_domain_loss {
            p.pixel_domain_loss = v;
        }
        if let Some(v) = self.lz77 {
            p.lz77 = v;
        }
        if let Some(v) = self.lz77_method {
            p.lz77_method = v;
        }
        if let Some(v) = self.patches {
            p.patches = v;
        }
        #[cfg(feature = "butteraugli-loop")]
        if let Some(v) = self.butteraugli_iters {
            p.butteraugli_iters = v;
        }
        p.apply_faster_decoding(self.faster_decoding);
        p
    }

    /// Resolved effort-profile (effort schedule + `__expert`/sparse
    /// overrides + `faster_decoding`), exposed for sweep
    /// encode-fingerprinting: the resolved byte-affecting state two
    /// knobsets must share to be byte-identical (see zenjxl
    /// `encode_fingerprint`). Public wrapper over the crate-internal
    /// resolver so sweep tooling needn't duplicate the override mapping.
    ///
    /// `#[doc(hidden)]` (#76): returns the internal [`EffortProfile`];
    /// reachable-but-unsupported, like the type itself.
    #[doc(hidden)]
    pub fn resolved_profile(&self) -> crate::effort::EffortProfile {
        self.effective_profile()
    }

    /// Effective patches flag (libjxl `enc_modular.cc:707` —
    /// `decoding_speed_tier < 2` for the modular subtract-and-encode
    /// path). At lossy tier >= 2 we skip the VarDCT patches pre-pass
    /// for the same reason.
    pub(crate) fn effective_patches(&self) -> bool {
        if self.faster_decoding >= 2 {
            return false;
        }
        self.patches()
    }

    /// Effective LZ77 flag. libjxl `enc_ans.cc:1372` skips LZ77 for
    /// VarDCT streams at `decoding_speed_tier >= 1` (the per-frame
    /// AC histogram pass forces `lz77_method = kNone`). Returns the
    /// stored `cfg.lz77()` field at tier 0.
    pub(crate) fn effective_lz77(&self) -> bool {
        if self.faster_decoding >= 1 {
            return false;
        }
        self.lz77()
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
        self.gaborish()
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
        // Delegate to the broader variant with no auto-classified
        // content (callers that need the W44-164 dispatch use the
        // `_and_class` variant directly).
        self.effective_profile_for_image_with_smoothness_and_class(
            pixels,
            smooth_photo_for_dct64_auto,
            None,
        )
    }

    /// Variant of [`Self::effective_profile_for_image_with_smoothness`]
    /// that also takes a caller-computed
    /// [`crate::effort::ImageContentClass`] from the W44-164 auto-
    /// classifier (`classify_from_proxies`).
    ///
    /// Precedence (highest → lowest):
    /// 1. `self.content_class` (caller-set via `with_content_class`)
    /// 2. `auto_classified_content` IF
    ///    `resolved.content_class_auto_classify` is `true`
    ///    (default on `EncoderStrategy::Zenjxl` / `Aggressive`;
    ///    `false` on `Libjxl` / `LeanFaster`)
    /// 3. No class adapter fires
    pub(crate) fn effective_profile_for_image_with_smoothness_and_class(
        &self,
        pixels: u64,
        smooth_photo_for_dct64_auto: bool,
        auto_classified_content: Option<crate::effort::ImageContentClass>,
    ) -> crate::effort::EffortProfile {
        let mut p = self.effective_profile();
        // Always-on per-image adapter — skipped only when an explicit
        // `__expert` override is in play, to avoid silently re-flipping
        // a sweep harness's pinned value.
        if !self.has_internal_overrides() {
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
            // W44-164 Smart-Zenjxl chunk 1 — content-class dispatch.
            //
            // Precedence:
            //   * `self.content_class` (caller-set via
            //     `with_content_class`) — ALWAYS wins.
            //   * Auto-classified via
            //     `auto_classify_content_class_from_layout` ONLY when
            //     `resolved.content_class_auto_classify == true`
            //     (default on `EncoderStrategy::Zenjxl` /
            //     `EncoderStrategy::Aggressive`; off on `Libjxl` and
            //     `LeanFaster`).
            //   * Neither set → no class adapter fires.
            //
            // The auto-classifier only computes on 8-bit sRGB layouts
            // with `pixels >= CONTENT_CLASS_MIN_PIXELS` (= 65,536), so
            // every existing hash-lock fixture (largest = 48×48 =
            // 2,304 px) and every non-sRGB-u8 layout (16-bit, linear-
            // f32, grayscale, HDR) stays byte-identical to pre-W44-164.
            //
            // The `Unknown` variant short-circuits inside
            // `adapt_to_image_content` so a deadband classification is
            // also byte-identical.
            //
            // RFC #45 pick #4 chunk 1 prior wiring kept the explicit
            // path; W44-164 layers the auto-path on top.
            let auto_class_for_resolve = if resolved.content_class_auto_classify {
                auto_classified_content
            } else {
                None
            };
            let effective_class = self.content_class.or(auto_class_for_resolve);
            if let Some(class) = effective_class {
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
            // W44-184: Section C CfL Newton libjxl-parity flip. Sets
            // `cfl_newton_libjxl_parity = true` when the resolved field
            // is true (i.e. under `EncoderStrategy::Libjxl`). The W44-183
            // honest-stop demonstrated that flipping this in isolation
            // regresses 25/27 photo cells — only safe when the rest of
            // the cost-model calibration (Section A/B/D divergences) is
            // ALSO flipped to libjxl-parity, which `EncoderStrategy::Libjxl`
            // does. Default (`false`) preserves byte-identical hash-locks.
            p.apply_section_c_cfl_newton_libjxl_parity(&resolved);
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
        // Store the sparse params (issue #80); resolved lazily in
        // `effective_profile` so the final effort wins regardless of
        // builder order.
        self.internal_overrides = Some(params);
        self
    }

    /// W44-222: install a Tier-2 knob set for this encode.
    ///
    /// The knobs are expanded to a full 6-param
    /// [`crate::tuning::runtime::RuntimeTuning`] at encode start and
    /// installed via [`crate::tuning::runtime::install_or_check_idempotent`].
    /// At [`crate::tuning::coupling::Tier2Knobs::default()`] the expander
    /// returns `RuntimeTuning::default()` byte-for-byte → no install is
    /// attempted → the production code path stays unaffected and every
    /// existing hash-lock fixture remains byte-identical.
    ///
    /// **Single-shot semantics** (W44-222 known limitation; see W44-223+):
    /// the underlying `runtime::install` uses a `OnceLock`, so a process
    /// can only install ONE distinct `RuntimeTuning`. Subsequent encodes
    /// with the SAME knobs no-op (idempotent); subsequent encodes with
    /// DIFFERENT knobs return [`EncodeError::InvalidConfig`]. Production
    /// callers should set knobs at process start; sweep runners install
    /// once per worker. A thread-local override is queued as W44-227.
    ///
    /// Pass `None` (or do not call this method) to keep the default
    /// behaviour (no override installed).
    ///
    /// Gated on `tuning-override`; the underlying types only exist
    /// under that feature.
    ///
    /// **MEASURED DANGER REGION**: raw per-stratum sweep optima with
    /// `k1 < 0.5` or `k2 < 1.0` on screen/{very_high,high} strata
    /// re-incur the W44-105 SHIP-cell catastrophe (−4.9 to −5.1 SSIM2;
    /// W44-228c1 validation). Knob sets must be Pareto-validated on
    /// SHIP cells (bytes AND SSIM2) before install — see the
    /// "Tier-2 knobs / sweeps" binding constraints in CLAUDE.md.
    #[cfg(feature = "tuning-override")]
    pub fn with_knobs(mut self, knobs: crate::tuning::coupling::Tier2Knobs) -> Self {
        self.tier2_knobs = Some(knobs);
        self
    }

    /// W44-222: get the currently-set Tier-2 knobs (if any).
    #[cfg(feature = "tuning-override")]
    pub fn knobs(&self) -> Option<crate::tuning::coupling::Tier2Knobs> {
        self.tier2_knobs
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
    /// - **e10**: + finer non-aligned AC-strategy step (libjxl kGlacier
    ///   parity)
    /// - **e11**: + 8 butteraugli iters, 2 buttloop seeds, 2 tree-learn seeds
    /// - **e12**: + 16 butteraugli iters, 4 lossy-search seeds, 16 tree-learn seeds
    /// - **e13**: + 32 butteraugli iters (requires `MAX_QUANT_LOOP_ITERS = 32`)
    ///
    /// e10 supersets libjxl e10; e11/e12/e13 extend past libjxl with
    /// strictly-longer search budgets (RFC#45 pick #1, renumbered +1 by
    /// the 2026-08-29 ladder shift); the bitstream remains 100%
    /// spec-valid. See RFC issue #45.
    ///
    /// Individual `with_*()` calls after `with_effort()` override these defaults.
    pub fn with_effort(mut self, effort: u8) -> Self {
        // Pure setter (issue #80). Effort-derived knobs are `Option`
        // (resolved against `self.effort` in `effective_profile`), and
        // every other profile field is resolved from `self.effort` too —
        // nothing on the config caches an effort-derived value — so
        // changing the effort needs no field rebuild and *cannot* clobber
        // a caller override regardless of call order. This replaces the
        // ~40-line field-by-field preserve block + the `*_explicit`
        // touched-bits that the rebuild-from-`new_with_effort` design
        // required.
        self.effort = effort;
        self
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
    #[doc(hidden)]
    pub fn with_ans(mut self, enable: bool) -> Self {
        self.use_ans = Some(enable);
        self
    }

    /// Enable/disable gaborish inverse pre-filter (default: true).
    #[doc(hidden)]
    pub fn with_gaborish(mut self, enable: bool) -> Self {
        self.gaborish = Some(enable);
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
    pub fn with_error_diffusion(mut self, enable: bool) -> Self {
        self.error_diffusion = Some(enable);
        self
    }

    /// Enable/disable pixel-domain loss in strategy selection (default: true).
    #[doc(hidden)]
    pub fn with_pixel_domain_loss(mut self, enable: bool) -> Self {
        self.pixel_domain_loss = Some(enable);
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
    #[doc(hidden)]
    pub fn with_perceptual_optimizations(mut self, enable: bool) -> Self {
        // Set the five perceptual knobs to their on/off positions.
        // Defaults mirror libjxl's enabled state when on.
        self.gaborish = Some(enable);
        // Convenience setter pins patches (the `Some` *is* the pin) —
        // opting out via this method suppresses the content-class
        // dispatch too. Issue #80: pinning via `Some` makes a following
        // `with_effort` preserve all three knobs automatically.
        self.patches = Some(enable);
        self.dot_detection = enable; // libjxl `Override::kDefault`; in-encoder effort/distance gates make this niche-only
        self.noise = false; // off by default in libjxl too
        self.pixel_domain_loss = Some(enable);
        self
    }

    /// Enable/disable LZ77 backward references (default: false).
    #[doc(hidden)]
    pub fn with_lz77(mut self, enable: bool) -> Self {
        self.lz77 = Some(enable);
        self
    }

    /// Set LZ77 method (default: Greedy).
    #[doc(hidden)]
    pub fn with_lz77_method(mut self, method: Lz77Method) -> Self {
        self.lz77_method = Some(method);
        self
    }

    /// Force a specific AC strategy for all blocks. `None` for auto-selection.
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
    pub fn with_patches(mut self, enable: bool) -> Self {
        self.patches = Some(enable);
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
    #[doc(hidden)]
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
    #[doc(hidden)]
    pub fn with_auto_splines(mut self, enable: bool) -> Self {
        self.auto_splines = Some(enable);
        self
    }

    /// Whether automatic spline detection is enabled. See
    /// [`Self::with_auto_splines`].
    pub fn auto_splines(&self) -> bool {
        self.auto_splines
            .unwrap_or_else(|| crate::effort::EffortProfile::auto_splines_default(self.effort))
    }

    /// Whether [`Self::auto_splines`] was set explicitly via
    /// [`Self::with_auto_splines`] (rather than derived from the
    /// effort-based default in
    /// [`crate::effort::EffortProfile::auto_splines_default`]).
    pub fn auto_splines_explicit(&self) -> bool {
        self.auto_splines.is_some()
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
    #[doc(hidden)]
    pub fn with_butteraugli_iters(mut self, n: u32) -> Self {
        self.butteraugli_iters = Some(n);
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

    /// Multi-metric Phase 0 (RFC #3, 2026-05-25): set which perceptual
    /// metric drives the buttloop's iterative quantization loop. See
    /// [`PerceptualMetric`].
    ///
    /// Default: [`PerceptualMetric::Butteraugli`].
    ///
    /// Choosing a non-default metric requires the corresponding cargo
    /// feature to be compiled in. Without the feature, the dispatch
    /// silently falls back to butteraugli (one-shot `eprintln!` warning
    /// the first time the caller picks an unbuilt metric).
    ///
    /// **Strategy-level override**: [`EncoderStrategy::Libjxl`] FORCES
    /// the resolved metric back to [`PerceptualMetric::Butteraugli`]
    /// regardless of this field (W44-126 strict cjxl-parity invariant
    /// + RFC #1 §7.3). See [`Self::resolve_perceptual_metric`].
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn with_perceptual_metric(mut self, metric: PerceptualMetric) -> Self {
        self.perceptual_metric = metric;
        self
    }

    /// Currently configured perceptual metric (Phase 0). May differ
    /// from the runtime-resolved value — call
    /// [`Self::resolve_perceptual_metric`] to see what will actually
    /// drive the loop after accounting for the Libjxl strict-parity
    /// invariant and per-metric cargo-feature gates.
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn perceptual_metric(&self) -> PerceptualMetric {
        self.perceptual_metric
    }

    /// Multi-metric Phase 0 (RFC #3, 2026-05-25): set the compute-device
    /// preference for the active perceptual metric. See
    /// [`PerceptualDevice`].
    ///
    /// Default: [`PerceptualDevice::Auto`].
    ///
    /// `Auto` resolves per-metric: for butteraugli, GPU when the
    /// `gpu-butteraugli` cargo feature is compiled in (W44-PHASE3-B5-flip
    /// parity), CPU otherwise. For cvvdp, GPU first when `cvvdp-loop` is
    /// compiled and CUDA inits; CPU when only `cvvdp-loop-cpu` is
    /// compiled; butteraugli fallback if neither.
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn with_perceptual_device(mut self, device: PerceptualDevice) -> Self {
        self.perceptual_device = device;
        self
    }

    /// Currently configured perceptual device (Phase 0).
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn perceptual_device(&self) -> PerceptualDevice {
        self.perceptual_device
    }

    /// Multi-metric Phase 0 (RFC #3, 2026-05-25) + RFC
    /// `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md` Phase 1 (2026-05-26):
    /// override the metric's per-distance target table with an
    /// explicit per-image score target.
    ///
    /// When `None` (default), the metric's built-in calibration table
    /// drives the buttloop's effective convergence target via the
    /// caller-supplied [`Self::distance`]. When `Some(score)`, the
    /// per-metric inverse dispatch fires:
    ///
    /// - **butteraugli**: caller passes a butter-direction score
    ///   (smaller=better). The buttloop converts via
    ///   `vardct/butteraugli_targets.rs` (Phase 1 corpus-median table,
    ///   seeded from
    ///   `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`,
    ///   n=162 cells per distance band) to an `effective_distance`
    ///   value, and the loop's
    ///   `accept_bound = K_BUTTERAUGLI_ACCEPT_FACTOR × effective_distance`
    ///   drives convergence to approximately the requested butter
    ///   score (corpus-median precision; per-image variance ±30-50%
    ///   per RFC §1.3).
    /// - **cvvdp**: caller passes a cvvdp butter-direction score
    ///   (`10 - JOD`, smaller=better). The buttloop bypasses the
    ///   forward distance-table and uses the caller's score directly
    ///   as the metric-native convergence target.
    /// - **zensim**: caller passes a zensim butter-direction score
    ///   (`100 - native`, smaller=better). Same shape as cvvdp —
    ///   used directly as convergence target.
    ///
    /// **EncoderStrategy::Libjxl invariant**:
    /// [`EncoderStrategy::Libjxl`] FORCES the resolved target_score
    /// back to `None` regardless of this field (W44-126 strict
    /// cjxl-parity, enforced by
    /// `tests/strategy_libjxl_byte_lock.rs`). See
    /// [`Self::resolve_perceptual_target_score`].
    ///
    /// **Non-finite / non-positive guard**: `Some(f32::NAN)`,
    /// `Some(f32::INFINITY)`, `Some(0.0)`, `Some(-1.0)` etc. are
    /// silently dropped to `None` in the resolver. The metric-side
    /// dispatch lookups also guard, so the value cannot propagate
    /// into the loop arithmetic.
    ///
    /// Use for calibrating against a non-standard quality requirement
    /// (e.g. matching a specific reference encoder's output). Default
    /// `None` is the right choice for ~all production callers.
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    #[doc(hidden)]
    pub fn with_perceptual_target_score(mut self, score: Option<f32>) -> Self {
        self.perceptual_target_score = score;
        self
    }

    /// Currently configured per-distance target override (Phase 0).
    /// May be `None` (default — use the metric's built-in calibration
    /// table).
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn perceptual_target_score(&self) -> Option<f32> {
        self.perceptual_target_score
    }

    /// Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
    /// (2026-05-26): resolve the effective per-distance target-score
    /// override, honouring the
    /// [`EncoderStrategy::Libjxl`] strict-parity invariant.
    ///
    /// Returns the override that will ACTUALLY drive the buttloop,
    /// which differs from [`Self::perceptual_target_score`] when:
    ///
    /// - The active strategy is [`EncoderStrategy::Libjxl`] → ALWAYS
    ///   `None` (W44-126 byte-lock invariant: target_score is silently
    ///   dropped on Libjxl-strategy encodes regardless of caller
    ///   input).
    /// - Future strategies may add additional short-circuits here.
    ///
    /// Pre-Phase-1 (commit `23da77b1` onwards) the
    /// `LossyConfig::with_perceptual_target_score(Some(_))` setter was
    /// a phantom no-op — the field stored on the config but the
    /// `vardct::perceptual_loop::run_buttloop` dispatch never
    /// consulted it. Phase 1 closes the wiring through
    /// [`Self::resolve_perceptual_metric_selection`] →
    /// [`crate::vardct::perceptual_backend::propagate_resolved_metric_to_encoder`]
    /// → [`crate::vardct::VarDctEncoder::perceptual_target_score`].
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub(crate) fn resolve_perceptual_target_score(&self) -> Option<f32> {
        if matches!(self.strategy, EncoderStrategy::Libjxl) {
            // Strict cjxl-parity (W44-126): EncoderStrategy::Libjxl
            // forces target_score to None regardless of caller field.
            // Same shape as `resolve_perceptual_metric` /
            // `resolve_target_display` / `resolve_cvvdp_bytes_tighten`
            // — the per-cell SHA256 byte-lock at
            // `tests/strategy_libjxl_byte_lock.rs` enforces this.
            return None;
        }
        // NaN/Inf/non-positive sanitation: a caller setting
        // `with_perceptual_target_score(Some(f32::NAN))` should NOT
        // propagate the NaN into the loop. Drop to `None` to fall
        // back to the metric's forward calibration arm. The
        // butteraugli inverse table and the cvvdp/zensim direct-use
        // paths in the buttloop dispatch ALSO guard against these
        // values, so this is defense-in-depth.
        match self.perceptual_target_score {
            Some(s) if !s.is_finite() || s <= 0.0 => None,
            other => other,
        }
    }

    /// Multi-metric Phase 0 (RFC #3, 2026-05-25): resolve the effective
    /// perceptual metric, honouring the strict-parity invariant and the
    /// per-metric cargo-feature gates.
    ///
    /// Returns the metric that will ACTUALLY drive the loop, which may
    /// differ from [`Self::perceptual_metric`] when:
    ///
    /// - The active strategy is [`EncoderStrategy::Libjxl`] → ALWAYS
    ///   `Butteraugli` (W44-126 strict parity).
    /// - The configured metric's cargo feature is not compiled in →
    ///   falls back to `Butteraugli` (silent fallback; the construct-
    ///   backend dispatch emits a one-shot warning on the first
    ///   silent downgrade).
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub(crate) fn resolve_perceptual_metric(&self) -> PerceptualMetric {
        if matches!(self.strategy, EncoderStrategy::Libjxl) {
            return PerceptualMetric::Butteraugli;
        }
        match self.perceptual_metric {
            PerceptualMetric::Butteraugli => PerceptualMetric::Butteraugli,
            PerceptualMetric::Cvvdp => {
                #[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
                {
                    PerceptualMetric::Cvvdp
                }
                #[cfg(not(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu")))]
                {
                    PerceptualMetric::Butteraugli
                }
            }
            PerceptualMetric::Zensim => {
                // zensim-fork Phase 3 (RFC #3 / RFC_ZENSIM_FORK_PLAN.md §5,
                // 2026-05-25): silent fallback to Butteraugli when neither
                // zensim-loop nor zensim-loop-gpu was compiled in. The
                // construct-backend dispatch emits a one-shot warning on
                // the first silent downgrade so users can spot a missing
                // cargo feature without breaking the encode.
                #[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
                {
                    PerceptualMetric::Zensim
                }
                #[cfg(not(any(feature = "zensim-loop", feature = "zensim-loop-gpu")))]
                {
                    PerceptualMetric::Butteraugli
                }
            }
        }
    }

    /// Multi-metric Phase 0 (RFC #3, 2026-05-25): resolve the effective
    /// device preference (currently a pass-through — the construct-
    /// backend dispatch consumes the field directly via
    /// [`Self::resolve_perceptual_metric_selection`]).
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub(crate) fn resolve_perceptual_device(&self) -> PerceptualDevice {
        self.perceptual_device
    }

    /// Multi-metric Phase 0 (RFC #3 §4, 2026-05-25): bundle the
    /// resolved metric + device into a [`MetricSelection`] for
    /// downstream construct-backend dispatch.
    ///
    /// The Libjxl strict-parity short-circuit has already been applied
    /// — `metric == Butteraugli` in the returned struct unconditionally
    /// reflects "the encode will use butteraugli", regardless of what
    /// the caller passed in.
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub(crate) fn resolve_perceptual_metric_selection(
        &self,
    ) -> crate::vardct::perceptual_backend::MetricSelection {
        crate::vardct::perceptual_backend::MetricSelection {
            metric: self.resolve_perceptual_metric(),
            device: self.resolve_perceptual_device(),
            // Phase 1 of RFC
            // `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md` (2026-05-26):
            // route through the resolver so the
            // EncoderStrategy::Libjxl strict-parity short-circuit
            // fires before the value reaches the buttloop. Pre-Phase-1
            // this read `self.perceptual_target_score` directly,
            // bypassing the short-circuit — that was safe at the time
            // ONLY because the dispatch below
            // (`construct_backend`'s `let _ = selection.target_score`)
            // discarded the field. Now that the field is consumed,
            // the resolver MUST gate it.
            target_score: self.resolve_perceptual_target_score(),
            // Phase 1 display-config backfill (2026-05-25): bundle the
            // resolved display config into the selection struct so it
            // travels alongside the metric + device through every
            // downstream construct_backend / propagate site.
            target_display: self.resolve_target_display(),
        }
    }

    /// cvvdp-fork Phase 8d (2026-05-25): caller-supplied preference for
    /// the post-convergence bytes-tighten exit pass on the cvvdp seed
    /// loop. See [`Self::cvvdp_bytes_tighten`] field doc for the full
    /// semantics. Brief:
    ///
    /// - `None` (default): "on when both the `cvvdp-loop-tighten` cargo
    ///   feature is compiled AND
    ///   [`Self::resolve_perceptual_metric`] returns
    ///   [`PerceptualMetric::Cvvdp`]".
    /// - `Some(true)`: explicit opt-in. Same behaviour as `None` inside
    ///   the feature gate.
    /// - `Some(false)`: explicit opt-out. Skips the tighten pass even
    ///   when the cargo feature is compiled and cvvdp is the active
    ///   metric.
    ///
    /// The tighten pass NEVER fires on the butteraugli loop regardless
    /// of this setting — see [`Self::resolve_cvvdp_bytes_tighten`].
    ///
    /// **Multi-metric Phase 0 rename**: this is now the LAST surviving
    /// cvvdp-specific setter. The metric-selection setters
    /// (`with_cvvdp_loop` / `with_cvvdp_use_cpu` / `with_gpu_butteraugli`)
    /// were collapsed into [`Self::with_perceptual_metric`] +
    /// [`Self::with_perceptual_device`]; this knob stays because it
    /// tunes cvvdp's post-convergence pass, not the metric choice.
    ///
    /// Requires the `butteraugli-loop` feature (the underlying buttloop
    /// dispatch surface). The tighten pass itself further requires the
    /// `cvvdp-loop-tighten` cargo feature (which transitively requires
    /// `cvvdp-loop`).
    #[cfg(feature = "butteraugli-loop")]
    #[doc(hidden)]
    pub fn with_cvvdp_bytes_tighten(mut self, enable: Option<bool>) -> Self {
        self.cvvdp_bytes_tighten = enable;
        self
    }

    /// Currently configured CVVDP bytes-tighten opt-in (cvvdp-fork
    /// Phase 8d). May be `None` (default — auto-on inside the feature
    /// gate when cvvdp is the active metric) — use
    /// [`Self::resolve_cvvdp_bytes_tighten`] to see the effective bool
    /// that actually drives the dispatch.
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub fn cvvdp_bytes_tighten(&self) -> Option<bool> {
        self.cvvdp_bytes_tighten
    }

    /// cvvdp-fork Phase 8d: resolve the effective CVVDP bytes-tighten
    /// preference. Returns `true` (run the tighten pass) iff ALL of:
    ///
    /// 1. [`Self::resolve_perceptual_metric`] returns
    ///    [`PerceptualMetric::Cvvdp`] (cvvdp is the active backend —
    ///    the tighten pass NEVER fires on the butteraugli loop; see
    ///    field doc for rationale).
    /// 2. The `cvvdp-loop-tighten` cargo feature is compiled in.
    /// 3. [`Self::cvvdp_bytes_tighten`] is `Some(true)` OR `None`
    ///    (default-on inside the feature gate).
    ///
    /// Returns `false` (skip the tighten pass, byte-identical to
    /// pre-Phase-8d) in every other case.
    ///
    /// Requires the `butteraugli-loop` feature.
    #[cfg(feature = "butteraugli-loop")]
    pub(crate) fn resolve_cvvdp_bytes_tighten(&self) -> bool {
        // Outer gate: cvvdp must be the active backend (resolver
        // already applies the Libjxl strict-parity short-circuit).
        if !matches!(self.resolve_perceptual_metric(), PerceptualMetric::Cvvdp) {
            return false;
        }
        // Feature gate: cargo feature must be compiled.
        #[cfg(not(feature = "cvvdp-loop-tighten"))]
        {
            false
        }
        #[cfg(feature = "cvvdp-loop-tighten")]
        {
            // Field gate: explicit None → default-on inside the feature
            // gate. Some(true) is the explicit opt-in form (same effect
            // as None). Some(false) is the explicit opt-out.
            self.cvvdp_bytes_tighten.unwrap_or(true)
        }
    }

    /// Phase 1 display-config backfill (RFC
    /// `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`, 2026-05-25): set the
    /// target display config for cvvdp scoring.
    ///
    /// See [`DisplayConfig`] for variants + the Phase 1 geometry caveat.
    /// Default [`DisplayConfig::WebSdr80`] keeps every existing hash-lock
    /// fixture byte-identical.
    ///
    /// Has no effect when the resolved [`PerceptualMetric`] is NOT
    /// [`PerceptualMetric::Cvvdp`] — display config only routes through
    /// the cvvdp scoring path. The field is always present so callers
    /// can set it ahead of switching the metric.
    ///
    /// The setter is unconditional (no feature gate) because
    /// [`DisplayConfig`] itself is feature-independent — only the
    /// `display_model()` / `display_geometry()` conversion methods are
    /// gated on `cvvdp-loop`.
    pub fn with_target_display(mut self, display: DisplayConfig) -> Self {
        self.target_display = display;
        self
    }

    /// Currently configured target display (Phase 1). May differ from
    /// the runtime-resolved value when [`Self::strategy`] is
    /// [`EncoderStrategy::Libjxl`] (forces `WebSdr80` for strict
    /// cjxl-parity) — call [`Self::resolve_target_display`] for the
    /// effective value.
    pub fn target_display(&self) -> DisplayConfig {
        self.target_display
    }

    /// Phase 1 display-config backfill: resolve the effective target
    /// display config, honouring the strict-parity invariant.
    ///
    /// Returns the display that will ACTUALLY drive the cvvdp scoring,
    /// which may differ from [`Self::target_display`] when the active
    /// strategy is [`EncoderStrategy::Libjxl`] (forces `WebSdr80` —
    /// matches the W44-126 pattern for `with_perceptual_metric`).
    pub(crate) fn resolve_target_display(&self) -> DisplayConfig {
        if matches!(self.strategy, EncoderStrategy::Libjxl) {
            return DisplayConfig::WebSdr80;
        }
        self.target_display
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

    /// `true` when the content is HDR PQ or HLG (by explicit
    /// `with_color_encoding`, else the layout's implied transfer function).
    ///
    /// #74/#11 (2026-07-15): the encoder pipelines DISABLE the perceptual
    /// quantization loop on HDR content — see the call sites for the measured
    /// rationale and [`docs/LIBJXL_DIVERGENCES.md`]. Uses the SAME transfer
    /// resolution as [`Self::resolve_hdr_loss`] so the two stay consistent.
    #[cfg(feature = "butteraugli-loop")]
    pub(crate) fn is_hdr_pq_hlg(
        &self,
        layout: PixelLayout,
        color_encoding: Option<&crate::headers::color_encoding::ColorEncoding>,
    ) -> bool {
        use crate::headers::color_encoding::TransferFunction;
        let tf = color_encoding
            .map(|ce| ce.transfer_function)
            .or_else(|| layout.implied_transfer_function());
        matches!(tf, Some(TransferFunction::Pq) | Some(TransferFunction::Hlg))
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
            .unwrap_or_else(|| self.effort_schedule().use_ans)
    }

    /// Whether gaborish inverse pre-filter is enabled.
    pub fn gaborish(&self) -> bool {
        self.gaborish
            .unwrap_or_else(|| self.effort_schedule().gaborish)
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
            .unwrap_or_else(|| self.effort_schedule().error_diffusion)
    }

    /// Whether pixel-domain loss is enabled.
    pub fn pixel_domain_loss(&self) -> bool {
        self.pixel_domain_loss
            .unwrap_or_else(|| self.effort_schedule().pixel_domain_loss)
    }

    /// Whether patches (dictionary-based repeated pattern detection)
    /// are enabled.
    pub fn patches(&self) -> bool {
        self.patches
            .unwrap_or_else(|| self.effort_schedule().patches)
    }

    /// Whether dot detection (refs #19) is enabled.
    pub fn dot_detection(&self) -> bool {
        self.dot_detection
    }

    /// Whether LZ77 backward references are enabled.
    pub fn lz77(&self) -> bool {
        self.lz77.unwrap_or_else(|| self.effort_schedule().lz77)
    }

    /// Current LZ77 method.
    pub fn lz77_method(&self) -> Lz77Method {
        self.lz77_method
            .unwrap_or_else(|| self.effort_schedule().lz77_method)
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
        // Conservative upper bound (the historical contract of this
        // method) = the calibrated `max`. See [`Self::estimate_encode`]
        // for the full min/typical/max breakdown.
        crate::heuristics::estimate_encode(
            width,
            height,
            layout.bytes_per_pixel() as u8,
            layout.has_alpha(),
            false,
            self.effort,
        )
        .map(|e| e.peak_memory_bytes_max)
    }

    /// Full calibrated resource estimate (min / typical / max peak
    /// memory, plus coarse time and output size) for a lossy encode at
    /// these settings. Mirrors the zen per-codec pattern
    /// ([`crate::heuristics::EncodeEstimate`], cf. `zenwebp`). Use
    /// `typical` for capacity planning and `max` to size a
    /// [`Limits`] cap. `None` only on dimension overflow.
    pub fn estimate_encode(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Option<crate::heuristics::EncodeEstimate> {
        crate::heuristics::estimate_encode(
            width,
            height,
            layout.bytes_per_pixel() as u8,
            layout.has_alpha(),
            false,
            self.effort,
        )
    }

    /// Butteraugli quantization loop iterations.
    #[cfg(feature = "butteraugli-loop")]
    pub fn butteraugli_iters(&self) -> u32 {
        self.butteraugli_iters
            .unwrap_or_else(|| self.effort_schedule().butteraugli_iters)
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
    /// True when a sweep/picker has pinned `__expert` internal-param
    /// overrides (issue #80) — used to skip the per-image auto-adapter
    /// so the pinned value survives.
    #[cfg(feature = "__expert")]
    fn has_internal_overrides(&self) -> bool {
        self.internal_overrides.is_some()
    }

    #[cfg(not(feature = "__expert"))]
    fn has_internal_overrides(&self) -> bool {
        false
    }

    /// The resolved override profile (schedule + internal-param overrides)
    /// when a sweep has pinned `__expert` params; `None` otherwise. Used by
    /// `validate` to range-check pinned values.
    #[cfg(feature = "__expert")]
    pub(crate) fn overridden_profile(&self) -> Option<crate::effort::EffortProfile> {
        self.internal_overrides
            .as_ref()
            .map(|_| self.effective_profile())
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
            relative_to_max_display: None,
            linear_below: None,
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
        encode_animation_lossy(self, width, height, layout, animation, frames, None).at()
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
        encode_animation_lossy(self, width, height, layout, animation, frames, Some(limits)).at()
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
    /// `ToneMapping.relative_to_max_display` override. `None` falls
    /// back to the metadata-level value (or the JXL default `false`).
    /// Issue #46 chunk 1a.
    relative_to_max_display: Option<bool>,
    /// `ToneMapping.linear_below` override. `None` falls back to the
    /// metadata-level value (or the JXL default `0.0`). Issue #46
    /// chunk 1a.
    linear_below: Option<f32>,
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
    /// Besides signaling, the color encoding drives lossy pixel linearization
    /// for integer (u8/u16, RGB(A)/Gray) input since #17: `TransferFunction::Pq`,
    /// `::Hlg`, and `::Bt709` apply the matching inverse EOTF instead of the
    /// default sRGB curve. [`with_source_gamma`](Self::with_source_gamma) still
    /// wins when set (an explicit gamma overrides the encoding's TF), and the
    /// dedicated f32 PQ/HLG/BT.709 layouts dispatch unconditionally. For the
    /// plain linear f32 layouts, pixels are assumed already linear.
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

    /// Set `ToneMapping.relative_to_max_display`.
    ///
    /// When `true`, [`Self::with_linear_below`] is interpreted as a
    /// ratio in `[0, 1]` of the maximum display brightness rather
    /// than an absolute nit value. Default is `false`. If both this
    /// builder and an attached [`ImageMetadata`] set this value, the
    /// request-level value wins. Closes issue #46 chunk 1a.
    pub fn with_relative_to_max_display(mut self, relative: bool) -> Self {
        self.relative_to_max_display = Some(relative);
        self
    }

    /// Set `ToneMapping.linear_below`.
    ///
    /// Tone mapping leaves pixels strictly below this value
    /// unchanged. Interpretation depends on
    /// [`Self::with_relative_to_max_display`] — ratio in `[0, 1]`
    /// when `true`, absolute nits when `false`. Default is `0.0`. If
    /// both this builder and an attached [`ImageMetadata`] set this
    /// value, the request-level value wins. Closes issue #46 chunk 1a.
    pub fn with_linear_below(mut self, value: f32) -> Self {
        self.linear_below = Some(value);
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
            .at()
    }

    /// Encode pixels and return the JXL bytes together with [`EncodeStats`].
    #[track_caller]
    pub fn encode_with_stats(self, pixels: &[u8]) -> Result<EncodeResult> {
        self.encode_inner(pixels).at()
    }

    /// Encode pixels, appending to an existing buffer. Returns metrics.
    #[track_caller]
    pub fn encode_into(self, pixels: &[u8], out: &mut Vec<u8>) -> Result<EncodeResult> {
        let mut result = self.encode_inner(pixels).at()?;
        if let Some(data) = result.data.take() {
            out.extend_from_slice(&data);
        }
        Ok(result)
    }

    /// Encode pixels, writing to a `std::io::Write` destination. Returns metrics.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn encode_to(self, pixels: &[u8], mut dest: impl std::io::Write) -> Result<EncodeResult> {
        let mut result = self.encode_inner(pixels).at()?;
        if let Some(data) = result.data.take() {
            dest.write_all(&data).map_err(at_from)?;
        }
        Ok(result)
    }

    fn encode_inner(&self, pixels: &[u8]) -> Result<EncodeResult> {
        self.validate_pixels(pixels)?;
        self.check_limits()?;
        // Run the full config validator (distance, effort, iter
        // counts, mutual exclusivity, etc.). This was previously
        // opt-in via `cfg.validate()`; auto-calling it on the encode
        // path means callers no longer have to remember to invoke it.
        match self.config {
            ConfigRef::Lossy(cfg) => cfg.validate().map_err(at_from)?,
            ConfigRef::Lossless(cfg) => cfg.validate().map_err(at_from)?,
        }
        // W44-222: install Tier-2 knobs (if any) into the runtime-tuning
        // override before encoding starts. Default knobs (the round-trip
        // case) are detected by comparing against `Tier2Knobs::default()`
        // and skipped — keeps the no-override-installed fast path so
        // every existing hash-lock fixture stays byte-identical.
        #[cfg(feature = "tuning-override")]
        if let ConfigRef::Lossy(cfg) = self.config
            && let Some(knobs) = cfg.tier2_knobs
            && knobs != crate::tuning::coupling::Tier2Knobs::default()
        {
            let rt = knobs.expand_to_runtime_tuning();
            crate::tuning::runtime::install_or_check_idempotent(rt).map_err(|_existing| {
                at!(EncodeError::InvalidConfig {
                    message: "with_knobs: a different RuntimeTuning is already \
                              installed in this process; the runtime override \
                              is single-shot (see W44-222 known limitation, \
                              W44-227+ for thread-local follow-on)"
                        .into(),
                })
            })?;
        }
        if let Some(ref ce) = self.color_encoding {
            crate::vardct::xyb::validate_color_encoding(ce).map_err(at_from)?;
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
        let rtmd = self
            .relative_to_max_display
            .or_else(|| self.metadata.and_then(|m| m.relative_to_max_display));
        let lb = self
            .linear_below
            .or_else(|| self.metadata.and_then(|m| m.linear_below));
        validate_tone_mapping_full(it, mn, rtmd, lb)?;
        // Source gamma + intrinsic size up-front checks.
        validate_source_gamma(self.source_gamma)?;
        validate_intrinsic_size(self.metadata.and_then(|m| m.intrinsic_size))?;

        // Build the per-encode allocation budget + choose the worker
        // thread count. Caller-supplied Limits.max_memory_bytes wins;
        // otherwise Limits provides its path-aware soft-default cap. The
        // budget is threaded through to major dimension-driven allocation
        // sites (XYB planes, padded scratch, group buffers, modular
        // channels) via RAII guards; peak working-set is observable
        // post-encode via `EncodeStats::budget_peak_bytes`.
        //
        // The up-front check uses the calibrated path/effort/thread-aware
        // estimate (`heuristics::estimate_encode_threaded`): threads are
        // walked down until the estimate fits the cap (and the detected
        // available RAM × 0.8), and the encode is rejected only when even
        // the single-threaded estimate exceeds the cap — an early,
        // meaningful bail-out for absurd dimensions instead of a
        // confusing mid-encode failure (or a kernel OOM kill).
        let (is_lossless, effort, requested_threads) = match self.config {
            ConfigRef::Lossless(cfg) => (true, cfg.effort, cfg.threads),
            ConfigRef::Lossy(cfg) => (false, cfg.effort, cfg.threads),
        };
        // Sectioned local trees (imazen/jxl-encoder#96): a lossless config
        // whose knob can engage per-group trees is admitted and estimated
        // on the sectioned band where the frame encoder's gate will pick
        // it (see `encode_preflight_with_sectioned`).
        let sectioned = match self.config {
            ConfigRef::Lossless(cfg) => cfg.sectioned_trees(),
            ConfigRef::Lossy(_) => SectionedTrees::Off,
        };
        let preflight = encode_preflight_with_sectioned(
            self.width,
            self.height,
            self.layout.bytes_per_pixel() as u8,
            self.layout.has_alpha(),
            is_lossless,
            effort,
            requested_threads,
            false,
            self.limits,
            sectioned,
        )?;
        let EncodePreflight {
            budget,
            threads,
            estimated_peak_bytes,
        } = preflight;

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
        })
        .map_err(at_from)?;

        stats.codestream_size = codestream.len();
        stats.budget_peak_bytes = budget.peak();
        stats.threads_used = threads as u32;
        stats.estimated_peak_bytes = estimated_peak_bytes;

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

    fn validate_pixels(&self, pixels: &[u8]) -> Result<()> {
        validate_dims(self.width, self.height).at()?;
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
            return Err(at!(EncodeError::LimitExceeded {
                message: format!(
                    "image {w}x{h} too large for encoder working buffers \
                     (width × height × {MAX_INTERNAL_SCALE} overflows usize)"
                ),
            }));
        }
        // When row_stride is set, the buffer is `height * stride`
        // bytes (stride may include per-row padding). Validate
        // `stride >= width * bytes_per_pixel` and the buffer size up
        // front so callers fail before any allocation; the strided
        // unpack downstream re-checks defensively.
        if let Some(stride) = self.row_stride {
            let row_bytes = w
                .checked_mul(self.layout.bytes_per_pixel())
                .ok_or_else(|| {
                    at!(EncodeError::InvalidInput {
                        message: "width * bytes_per_pixel overflows usize".into(),
                    })
                })?;
            if stride < row_bytes {
                return Err(at!(EncodeError::InvalidInput {
                    message: format!(
                        "row_stride {stride} is less than width * bytes_per_pixel = {w} * {} = {row_bytes}",
                        self.layout.bytes_per_pixel(),
                    ),
                }));
            }
            let needed = h.checked_mul(stride).ok_or_else(|| {
                at!(EncodeError::InvalidInput {
                    message: "height * row_stride overflows usize".into(),
                })
            })?;
            if pixels.len() < needed {
                return Err(at!(EncodeError::InvalidInput {
                    message: format!(
                        "pixel buffer too small for strided input: need {needed} bytes (height {h} × stride {stride}), got {}",
                        pixels.len(),
                    ),
                }));
            }
            return Ok(());
        }
        match expected {
            Some(expected) if pixels.len() == expected => Ok(()),
            Some(expected) => Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "pixel buffer size mismatch: expected {expected} bytes for {w}x{h} {:?}, got {}",
                    self.layout,
                    pixels.len()
                ),
            })),
            None => Err(at!(EncodeError::InvalidInput {
                message: "image dimensions overflow".into(),
            })),
        }
    }

    fn check_limits(&self) -> Result<()> {
        let Some(limits) = self.limits else {
            return Ok(());
        };
        let w = self.width as u64;
        let h = self.height as u64;
        if let Some(max_w) = limits.max_width
            && w > max_w
        {
            return Err(at!(EncodeError::LimitExceeded {
                message: format!("width {w} > max {max_w}"),
            }));
        }
        if let Some(max_h) = limits.max_height
            && h > max_h
        {
            return Err(at!(EncodeError::LimitExceeded {
                message: format!("height {h} > max {max_h}"),
            }));
        }
        if let Some(max_px) = limits.max_pixels
            && w * h > max_px
        {
            return Err(at!(EncodeError::LimitExceeded {
                message: format!("pixels {}x{} = {} > max {max_px}", w, h, w * h),
            }));
        }
        // NOTE: max_memory_bytes admission is NOT checked here. The old
        // flat 40 B/px screen this method used both under-estimated
        // (admitted 108 MP encodes that peaked ≥ 44 GiB pre-2026-08-01)
        // and duplicated the real check — `encode_preflight` performs the
        // calibrated path/effort/thread-aware admission (strictly
        // stronger: every band's β exceeds 40 B/px) for every entry
        // point, including this request path.
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
                return Err(at!(EncodeError::LimitExceeded {
                    message: format!(
                        "quantization-loop iterations ({configured}) exceed \
                         Limits::max_quant_loop_iters ({max_iters})"
                    ),
                }));
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
            m = m.max(cfg.butteraugli_iters());
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

    /// Lossless dispatch: at effort ≥ 11 the TectonicPlate per-image
    /// config trial (issue #45, libjxl e11 superset) wraps the single
    /// encode; below that (and for every probe/trial/final encode the
    /// schedule issues) [`Self::encode_lossless_single`] runs directly.
    fn encode_lossless(
        &self,
        cfg: &LosslessConfig,
        pixels: &[u8],
        budget: &alloc::sync::Arc<crate::budget::MemoryBudget>,
    ) -> core::result::Result<(Vec<u8>, EncodeStats), EncodeError> {
        // Explicit sectioned/hybrid memory modes are whole-encode
        // commitments the trial schedule would contradict (and e ≥ 10
        // forces the global tree on `Auto` anyway) — honour them by
        // skipping the trials.
        if cfg.effort >= 11
            && !matches!(
                cfg.sectioned_trees(),
                SectionedTrees::On | SectionedTrees::Hybrid
            )
        {
            return self.encode_lossless_tectonic(cfg, pixels, budget);
        }
        self.encode_lossless_single(cfg, pixels, budget)
    }

    /// The e11+ lossless TectonicPlate schedule (issue #45; libjxl
    /// `enc_frame.cc:2576-2643` structure):
    ///
    /// 1. Encode the two probe configs at trial effort (e10 — the
    ///    kGlacier analogue; libjxl trials also run at kGlacier).
    /// 2. Branch on the probe sizes: palette-hostile → the 24-config
    ///    `LessPalette` list, palette-friendly → the 20-config
    ///    `MorePalette` list (`sweep::tectonic_*`).
    /// 3. Encode every unique config (dedup on the resolved profile
    ///    fingerprint + modular-knob tuple; wp-only and
    ///    fraction-saturated siblings collapse), sequentially, keeping
    ///    only the smallest stream seen (sequential trials keep the
    ///    peak working set at one trial's, so the e11 memory band stays
    ///    the multi-seed envelope — see `heuristics.rs`).
    /// 4. Re-encode the winning config at the ambient tier's full
    ///    profile (e11: 2-seed tree learn; e12/e13: 16-seed) and keep
    ///    the smaller of that and the best trial — every tier stays a
    ///    structural superset of the one below.
    ///
    /// Knobs the caller set explicitly are pinned across all trials
    /// (caller intent wins over the schedule, unlike libjxl which
    /// clobbers); pinned axes shrink the unique set further.
    fn encode_lossless_tectonic(
        &self,
        cfg: &LosslessConfig,
        pixels: &[u8],
        budget: &alloc::sync::Arc<crate::budget::MemoryBudget>,
    ) -> core::result::Result<(Vec<u8>, EncodeStats), EncodeError> {
        use crate::sweep::{
            TectonicConfig, dedup_tectonic, tectonic_less_palette, tectonic_more_palette,
            tectonic_probe_pair,
        };

        // Apply one trial config to a clone of the caller's config at
        // `effort`, pinning caller-explicit knobs.
        let apply = |tc: &TectonicConfig, effort: u8| -> LosslessConfig {
            let mut c = cfg.clone().with_effort(effort);
            if cfg.modular_palette_colors().is_none() {
                c = c.with_modular_palette_colors(Some(tc.palette_colors));
            }
            if cfg.modular_channel_colors_group_percent().is_none() {
                c = c.with_modular_channel_colors_group_percent(Some(
                    tc.channel_colors_group_percent,
                ));
            }
            if cfg.modular_channel_colors_global_percent().is_none() {
                c = c.with_modular_channel_colors_global_percent(Some(
                    tc.channel_colors_global_percent,
                ));
            }
            if cfg.modular_group_size().is_none() {
                c = c.with_modular_group_size(Some(tc.group_size_shift));
            }
            if cfg.modular_predictor().is_none()
                && let Some(p) = tc.predictor
            {
                c = c.with_modular_predictor(Some(p));
            }
            if cfg.patches.is_none() {
                c = c.with_patches(tc.patches);
            }
            if cfg.tree_learning_sample_fraction().is_none() {
                c = c.with_tree_learning_sample_fraction(tc.tree_sample_fraction());
            }
            if cfg.modular_nb_prev_channels().is_none() {
                c = c.with_modular_nb_prev_channels(Some(tc.nb_prev_channels));
            }
            c
        };
        // Dedup key for an APPLIED trial config: the resolved effort
        // profile's fingerprint (captures the fraction override and any
        // pinned effort-level knobs) + the modular-knob tuple the
        // profile doesn't see.
        let key_of = |c: &LosslessConfig| -> (u64, i64, u32, u32, u8, Option<u8>, bool, i32) {
            (
                c.effective_profile().fingerprint_impl(),
                c.modular_palette_colors().unwrap_or(i64::MIN),
                c.modular_channel_colors_group_percent()
                    .unwrap_or(-1.0)
                    .to_bits(),
                c.modular_channel_colors_global_percent()
                    .unwrap_or(-1.0)
                    .to_bits(),
                c.modular_group_size().unwrap_or(u8::MAX),
                c.modular_predictor(),
                c.effective_patches(),
                c.modular_nb_prev_channels().unwrap_or(-1),
            )
        };

        const TRIAL_EFFORT: u8 = 10;
        let mut seen = alloc::collections::BTreeSet::new();
        let mut best: Option<(Vec<u8>, EncodeStats, TectonicConfig)> = None;

        // Probe pair → branch (libjxl: `size_test[0] <= size_test[1]`
        // picks LessPalette).
        let probes = tectonic_probe_pair();
        let mut probe_bytes = [0usize; 2];
        for (i, tc) in probes.iter().enumerate() {
            let c = apply(tc, TRIAL_EFFORT);
            seen.insert(key_of(&c));
            let (bytes, stats) = self.encode_lossless_single(&c, pixels, budget)?;
            probe_bytes[i] = bytes.len();
            if best.as_ref().is_none_or(|(b, _, _)| bytes.len() < b.len()) {
                best = Some((bytes, stats, *tc));
            }
        }
        let list = if probe_bytes[0] <= probe_bytes[1] {
            tectonic_less_palette()
        } else {
            tectonic_more_palette()
        };

        for tc in dedup_tectonic(list) {
            let c = apply(&tc, TRIAL_EFFORT);
            if !seen.insert(key_of(&c)) {
                continue; // collapsed with a probe or an earlier trial
            }
            let (bytes, stats) = self.encode_lossless_single(&c, pixels, budget)?;
            if best.as_ref().is_none_or(|(b, _, _)| bytes.len() < b.len()) {
                best = Some((bytes, stats, tc));
            }
        }
        let (trial_bytes, trial_stats, winner) =
            best.expect("tectonic schedule always encodes the probe pair");

        // Final pass: the winning config at the ambient tier's full
        // profile (multi-seed extras). Skip when it resolves to the same
        // encode as the winning trial (fingerprint match ⇒ byte-identical).
        let final_cfg = apply(&winner, cfg.effort);
        if key_of(&final_cfg) == key_of(&apply(&winner, TRIAL_EFFORT)) {
            return Ok((trial_bytes, trial_stats));
        }
        let (final_bytes, final_stats) = self.encode_lossless_single(&final_cfg, pixels, budget)?;
        if final_bytes.len() < trial_bytes.len() {
            Ok((final_bytes, final_stats))
        } else {
            Ok((trial_bytes, trial_stats))
        }
    }

    fn encode_lossless_single(
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

        // Detect patches BEFORE the ModularImage is built (#96
        // patches-phase lifetime, 2026-08-30). Detection reads only
        // `detection_pixels` (the raw input, BGR-swapped if needed) —
        // never the ModularImage — so running it first means its
        // working set (the u8→f32 conversion planes, the background /
        // flood-fill planes and the BFS frontier) overlaps just the
        // input buffer instead of input + the whole-image i32
        // channels. On screen content that working set previously SAT
        // AT the sectioned-mode encode peak: the `MEM_PROBE_PATCHES`
        // A/B measured +76 MiB (imac_dark 5.6 MP) / +138 MiB
        // (reddit 10.5 MP) at every thread count, and the alloc-sites
        // probe placed the peak instant inside the detection phase
        // (`benchmarks/jxl_sectioned_patches_lifetime_2026-08-30.tsv`).
        //
        // The gate is layout-derived: for every layout the match below
        // ADMITS (integer RGB/gray/CMYK), `PixelLayout::is_16bit()` /
        // `is_grayscale()` equal the built image's `bit_depth` /
        // `is_grayscale` exactly (float layouts error out below before
        // the old image-based gate could have seen them), so hoisting
        // cannot change which encodes detect patches. CMYK is excluded:
        // the detector assumes RGB-like perceptual colour and would
        // match on CMY planes. `bytes_per_pixel` counts BYTES; patches
        // detection wants the CHANNEL count (and, for 16-bit layouts,
        // reads u16 samples — see `find_and_build_lossless`). 16-bit
        // enabled by issue #72: the tiled-pool HDR class lost 39-63 %
        // to cjxl at e5/e6 purely because this gate kept the detector
        // off.
        let layout_bit_depth: u32 = if self.layout.is_16bit() { 16 } else { 8 };
        let bytes_per_sample = if self.layout.is_16bit() { 2 } else { 1 };
        let num_channels = self.layout.bytes_per_pixel() / bytes_per_sample;
        let can_use_patches = cfg.effective_patches()
            && !self.layout.is_grayscale()
            && num_channels >= 3
            && !self.layout.is_cmyk();
        let patches_data = if can_use_patches {
            crate::profile_time!("modular/patches_detect", {
                let pd_opt = crate::vardct::patches::find_and_build_lossless(
                    detection_pixels,
                    w,
                    h,
                    num_channels,
                    layout_bit_depth,
                    Some(budget),
                )
                .map_err(EncodeError::from)?;
                // RFC#45 chunks 4-7 lossless backport (chunk 5 lossless
                // trial encoder): per-image cost gate. Trial-encodes
                // lossless-shape ref-frame + dictionary overhead,
                // requires `savings_est >= 1.5 * overhead`. Protects
                // against pathological mixed content where patches barely
                // clear the detector's 1% coverage filter but the
                // ref-frame overhead dominates net savings. See
                // `PatchesData::is_cost_effective_lossless` doc-comment.
                pd_opt.filter(|pd| pd.is_cost_effective_lossless(layout_bit_depth, cfg.ans()))
            })
        } else {
            None
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
            // dim_shift > 30 cannot correspond to a real channel (the max
            // image dimension is 2^30) and would overflow the modular
            // writer's per-group region shifts; reject it up front instead
            // of relying on the `.min(31)` cap inside `downsampled_dims`
            // (which only protects the length check below).
            if ec.info.dim_shift > 30 {
                return Err(EncodeError::InvalidInput {
                    message: format!(
                        "extra_channels[{idx}]: dim_shift {} exceeds the maximum of 30",
                        ec.info.dim_shift
                    ),
                });
            }
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

        // (Patches detection ran ABOVE, before the ModularImage build —
        // see the #96 patches-phase-lifetime comment at the
        // `detection_pixels` block.)

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
            if let Some(r) = meta.relative_to_max_display {
                file_header.metadata.relative_to_max_display = r;
            }
            if let Some(lb) = meta.linear_below {
                file_header.metadata.linear_below = lb;
            }
            if let Some((w, h)) = meta.intrinsic_size {
                file_header.metadata.have_intrinsic_size = true;
                file_header.metadata.intrinsic_width = w;
                file_header.metadata.intrinsic_height = h;
            }
        }
        // Request-level overrides win over metadata-level values. Lets
        // callers do
        //   `cfg.encode_request(...).with_intensity_target(10000.0)`
        // without constructing an ImageMetadata. Closes #21 (intensity
        // pair) + issue #46 chunk 1a (ToneMapping rest).
        if let Some(it) = self.intensity_target {
            file_header.metadata.intensity_target = it;
        }
        if let Some(mn) = self.min_nits {
            file_header.metadata.min_nits = mn;
        }
        if let Some(r) = self.relative_to_max_display {
            file_header.metadata.relative_to_max_display = r;
        }
        if let Some(lb) = self.linear_below {
            file_header.metadata.linear_below = lb;
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
                cfg.ans(),
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
        let mut use_tree_learning = cfg.effective_tree_learning();
        let mut smart_profile = cfg.effective_profile_for_image((w as u64) * (h as u64));
        // Issue #72: budgeted tree learning for 16-bit RGB(A) at e5/e6.
        use_tree_learning |= cfg.lift_integer_tree_learning(
            self.layout,
            (w as u64) * (h as u64),
            &mut smart_profile,
        );
        let frame_encoder = FrameEncoder::new(
            w,
            h,
            FrameEncoderOptions {
                use_modular: true,
                effort: cfg.effort,
                use_ans: cfg.ans(),
                use_tree_learning,
                use_squeeze: cfg.squeeze,
                enable_lz77: cfg.effective_lz77(),
                lz77_method: cfg.lz77_method(),
                lossy_palette: cfg.lossy_palette,
                encoder_mode: cfg.mode,
                profile: smart_profile,
                sectioned_trees: cfg.sectioned_trees,
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
            .encode_modular_with_patches_src(
                // Ownership lets the multi-group path free the pre-transform
                // image after its step-0 transforms - one full-image i32 copy
                // off the tree-learning peak. Nothing here reads `image`
                // after this call.
                crate::modular::frame::ImageSource::Owned(image),
                &color_encoding,
                &mut writer,
                patches_data.as_ref(),
                self.stop,
            )
            .map_err(EncodeError::from)?;

        let stats = EncodeStats {
            mode: EncodeMode::Lossless,
            ans: cfg.ans(),
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
        // PQ / HLG / BT.709 EOTF dispatch (#17, closed). When the caller
        // sets a color_encoding with TransferFunction::Pq / ::Hlg /
        // ::Bt709, the input pixels are coded in that transfer; we apply
        // the matching inverse EOTF instead of the default sRGB
        // linearization. source_gamma still wins (caller explicitly chose
        // gamma over the encoding's TF). Wired for every integer layout
        // arm below (u8/u16 × RGB(A)/Gray); the streaming push_rows path
        // mirrors these predicates. Lossless needs no linearization —
        // modular stores the original samples and the encoding is
        // signaling-only there.
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
        #[cfg(feature = "__env_var_diagnostics")]
        let _t_conv = std::time::Instant::now();
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
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                (floats.to_vec(), None, false)
            }
            PixelLayout::RgbaLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                let rgb: Vec<f32> = floats
                    .chunks(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect();
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::GrayLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                (gray_f32_to_linear_f32_rgb(floats, 1), None, false)
            }
            PixelLayout::GrayAlphaLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
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
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                (pq_f32_to_linear_f32_rgb(floats, 3), None, false)
            }
            PixelLayout::RgbaPqF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                let rgb = pq_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::RgbHlgF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                (hlg_f32_to_linear_f32_rgb(floats, 3), None, false)
            }
            PixelLayout::RgbaHlgF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                let rgb = hlg_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha), false)
            }
            PixelLayout::RgbBt709F32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                (bt709_f32_to_linear_f32_rgb(floats, 3), None, false)
            }
            PixelLayout::RgbaBt709F32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
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
        #[cfg(feature = "__env_var_diagnostics")]
        if std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some() {
            eprintln!("encode_lossy: conversion={:?}", _t_conv.elapsed());
        }

        // HLG forward OOTF (issue #73 follow-up — libjxl ApplyHlgOotf
        // parity). hlg_*_to_linear produce SCENE light (inverse OETF
        // only, identity-OOTF TF class); every decoder's linear->HLG
        // output conversion applies the INVERSE OOTF (jxl_cms.cc:1175
        // fires whenever exactly one side is HLG). Without the matching
        // forward pass here the roundtrip lands at scene^(1/gamma) — a
        // constant, distance-flat ~22 dB wedge. Resolution order for
        // the gamma's display peak mirrors the enc.intensity_target
        // assignment below: explicit request > metadata > the HLG
        // 1,000-nit default.
        let is_hlg_input = source_is_hlg
            || matches!(
                self.layout,
                PixelLayout::RgbHlgF32 | PixelLayout::RgbaHlgF32
            );
        let mut linear_rgb = linear_rgb;
        if is_hlg_input {
            let it = self
                .intensity_target
                .or_else(|| self.metadata.as_ref().and_then(|m| m.intensity_target))
                .unwrap_or(1_000.0);
            if let Some(g) = hlg_ootf_gamma(it) {
                let primaries = self
                    .color_encoding
                    .as_ref()
                    .map(|c| c.primaries)
                    .unwrap_or(crate::headers::color_encoding::Primaries::Bt2100);
                apply_hlg_forward_ootf(&mut linear_rgb, hlg_ootf_luminances(primaries), g);
            }
        }

        // W44-35: cheap smooth-photo auto-detect on the raw sRGB u8
        // input (when applicable) feeds the DCT64 admission gate via
        // `effective_profile_for_image_with_smoothness`. Returns false
        // for non-u8 layouts, large images (>= 500k px), and content
        // that fails the smoothness discriminator. Caller-supplied
        // `StrategyOverrides::smooth_photo_dct64_hint = Some(_)`
        // (via `with_strategy_overrides`) always wins over the auto
        // value (resolved inside `effective_profile_*`).
        #[cfg(feature = "__env_var_diagnostics")]
        let _t_an = std::time::Instant::now();
        let smooth_photo_for_dct64 =
            detect_smooth_photo_for_dct64_from_layout(pixels, self.width, self.height, self.layout);
        #[cfg(feature = "__env_var_diagnostics")]
        if std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some() {
            eprintln!("encode_lossy: smooth_detect={:?}", _t_an.elapsed());
        }
        #[cfg(feature = "__env_var_diagnostics")]
        let _t_cc = std::time::Instant::now();
        // W44-164 Smart-Zenjxl chunk 1: cheap zenanalyze-proxy-based
        // ImageContentClass auto-classifier. Only computes on 8-bit sRGB
        // layouts and images >= CONTENT_CLASS_MIN_PIXELS (= 65,536 px).
        // The dispatch fires only when `EncoderStrategy::Zenjxl` /
        // `Aggressive` is selected (resolved via
        // `content_class_auto_classify`) AND the caller hasn't set
        // `with_content_class(Some(...))` explicitly. See
        // `auto_classify_content_class_from_layout` for the
        // discriminator definition.
        // BAND THE COMPUTATION to its single consumer's gates
        // (`EffortProfile::adapt_to_image_content`, the only reader of
        // this value): the class adapter fires ONLY at effort 5-6 with
        // auto-classification enabled by the strategy, no caller-set
        // class, no expert overrides, and a lossy distance. Everywhere
        // else the classifier's full-image analysis (78 ms at 4K — more
        // than the whole e3 encode core) was computed and discarded —
        // the exact failure the lossy-low hygiene rule names. Skipping
        // under precisely the adapter's own gates is byte-identical by
        // construction.
        let class_consumed = {
            let eff = cfg.effort();
            (eff == 5 || eff == 6)
                && cfg.content_class.is_none()
                && !cfg.has_internal_overrides()
                && cfg.resolve_improvements().content_class_auto_classify
        };
        // ONE shared zenanalyze-proxy sweep for BOTH consumers (the
        // classifier here and `enc.zenanalyze_proxies` below): the
        // classifier is `compute_w44_91_zenanalyze_proxies` + a pure
        // threshold (`classify_from_proxies`), and the encoder computed
        // the identical full-image pass again ~60 lines later — 2 x
        // ~78 ms at 4K e5. The band is the union of both consumers'
        // gates (class band eff 5-6 is a subset of the proxies band).
        let shared_proxies = if cfg.effort() >= 5 || cfg.distance >= 2.0 {
            compute_w44_91_zenanalyze_proxies(pixels, w, h, self.layout)
        } else {
            None
        };
        // W44-231: learned sub-band lift admission (confident-BAD model,
        // vardct::learned_admission). Only consulted by the d < 3.5
        // qf-seed band, so compute inside the same band as the proxies
        // and only when the image is proxy-eligible (sRGB-u8 layouts —
        // reuse the proxies presence as the eligibility signal).
        #[cfg(feature = "learned-admission")]
        let learned_subband_bad = if shared_proxies.is_some()
            && cfg.resolve_improvements().learned_subband_exclude
            && cfg.distance >= 2.0
        {
            crate::vardct::learned_admission::extract_rgb8_verdict(pixels, w, h, self.layout)
        } else {
            None
        };
        #[cfg(not(feature = "learned-admission"))]
        let learned_subband_bad: Option<bool> = None;
        #[cfg(feature = "__env_var_diagnostics")]
        if std::env::var_os("JXL_PROXY_DEBUG").is_some() {
            if let Some(p) = shared_proxies.as_ref() {
                eprintln!(
                    "[proxies] m3={:.6} fcbr={:.6} ed={:.6} lv={:.3}",
                    p.m3_colourfulness, p.flat_color_block_ratio, p.edge_density, p.luma_var
                );
            }
        }
        let auto_content_class = if class_consumed
            && (w as u64) * (h as u64) >= crate::api::content_detect::W44_164_MIN_PIXELS
        {
            shared_proxies
                .as_ref()
                .map(crate::api::content_detect::classify_from_proxies)
        } else {
            None
        };
        #[cfg(feature = "__env_var_diagnostics")]
        if std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some() {
            eprintln!("encode_lossy: content_class={:?}", _t_cc.elapsed());
        }
        #[cfg(feature = "__env_var_diagnostics")]
        let _t_prof = std::time::Instant::now();
        let mut profile = cfg.effective_profile_for_image_with_smoothness_and_class(
            (w as u64) * (h as u64),
            smooth_photo_for_dct64,
            auto_content_class,
        );

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
        enc.use_ans = cfg.ans();
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
        enc.error_diffusion = cfg.error_diffusion();
        enc.pixel_domain_loss = cfg.pixel_domain_loss();
        enc.pixel_loss_dispatch = enc.resolved_improvements.pixel_loss_dispatch;
        enc.single_pass_entropy_dispatch = enc.resolved_improvements.single_pass_entropy_dispatch;
        enc.enable_lz77 = cfg.effective_lz77();
        enc.lz77_method = cfg.lz77_method();
        enc.force_strategy = cfg.force_strategy;
        // RFC #45 pick #4 — when the caller has explicitly pinned `cfg.patches()`
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
        } else if cfg.patches.is_some() {
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
        enc.auto_splines = cfg.auto_splines();
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
        //
        // Perf (/goal hunt 2026-06-10): every proxy consumer is banded to
        // effort >= 5 (the W44-164 classifier / 2c dispatch paths) or
        // distance >= 2.0 (the W44-91/96/98/124 discriminators), so at
        // e3/e4 with d < 2 the full-resolution classifier sweep
        // (`compute_srgb_u8`, 24 % of e3 d=1 CPU) was computed and
        // discarded. Skip it there — bytes A/B-verified identical across
        // e3/e4/e5 × d1.0/d3.0. If a future consumer fires below this
        // band, widen the predicate (the gate registry rows carry the
        // bands).
        enc.zenanalyze_proxies = shared_proxies;
        enc.learned_subband_bad = learned_subband_bad;
        // Streaming refactor #11 chunk 6: thread the caller-selected
        // [`Buffering`] policy into VarDctEncoder so the per-region
        // precompute dispatch (precomputed.rs:compute_with_budget_and_buffering)
        // can route on it. `Buffering::Auto` resolves on image size at
        // dispatch time.
        enc.buffering = cfg.buffering;
        #[cfg(feature = "butteraugli-loop")]
        {
            enc.butteraugli_iters = cfg.butteraugli_iters();
            // EX-J11 chunk 4: resolve `HdrLoss::Auto` to a concrete
            // loss now (using caller's `with_color_encoding` if set,
            // else `PixelLayout::implied_transfer_function()`), so the
            // per-iter butteraugli loop reads a fixed variant. PQ /
            // HLG content lands on `Vdp2`; everything else on
            // `Butteraugli` (SDR hash-locks stay byte-identical).
            enc.hdr_loss = cfg.resolve_hdr_loss(self.layout, self.color_encoding.as_ref());
            // #74/#11 (2026-07-15): DISABLE the perceptual quantization loop on the
            // DEFAULT HDR path (PQ/HLG content, which `HdrLoss::Auto` routes to
            // `Vdp2`). MEASURED (benchmarks/hdr_buttloop_blowup_2026-07-15.* +
            // hdr_loss_ab): at e8/e9 the loop OVER-REFINES HDR catastrophically
            // (+100..500 % bytes vs cjxl across d1..d4). Both butteraugli AND
            // VDP2-lite read per-block tile distances ~2× too high at HDR luminance
            // (td_median ~8 vs target 4, bad_rate ~0.94), so the loop cranks the
            // quant field far past the requested distance (VDP2 ~1.2 at 2-6× the
            // bytes). The no-loop base already MATCHES/BEATS cjxl e9 on bytes AND
            // VDP2 quality on every measured crop, so the loop is pure harm here.
            // Zeroing `butteraugli_iters` reproduces the `--no-butteraugli` base
            // EXACTLY (same field state → identical downstream W44-168/169 dispatch).
            //
            // Gate = HDR transfer AND resolved-loss `Vdp2` (the default routing):
            //  - SDR (transfer != PQ/HLG) → untouched, byte-identical (hash-locks).
            //  - explicit `with_hdr_loss(Butteraugli)` on HDR is a deliberate escape
            //    hatch → KEEPS the loop (still over-refines; documented caveat), so
            //    the `explicit_butteraugli_overrides_pq_layout` contract holds.
            //  - explicit `with_hdr_loss(Vdp2)` on SDR (odd but legal) is guarded by
            //    the transfer check → keeps the loop.
            // See docs/LIBJXL_DIVERGENCES.md.
            if cfg.is_hdr_pq_hlg(self.layout, self.color_encoding.as_ref())
                && matches!(enc.hdr_loss, crate::vardct::hdr_metrics::HdrLoss::Vdp2)
            {
                enc.butteraugli_iters = 0;
            }
            // Multi-metric Phase 0 (RFC #3, 2026-05-25): propagate the
            // resolved perceptual-metric selection. The
            // [`crate::vardct::perceptual_backend::propagate_resolved_metric_to_encoder`]
            // helper does the (metric, device) → (gpu_butteraugli,
            // cvvdp_loop, cvvdp_use_cpu) legacy-field translation; the
            // Libjxl strict-parity short-circuit already fired inside
            // `resolve_perceptual_metric` (still-image path).
            crate::vardct::perceptual_backend::propagate_resolved_metric_to_encoder(
                cfg.resolve_perceptual_metric_selection(),
                &mut enc,
            );
            // cvvdp-fork Phase 8d (2026-05-25): propagate bytes-tighten
            // opt-in (still-image path). `resolve_cvvdp_bytes_tighten`
            // returns true ONLY when the resolved metric is Cvvdp AND
            // the `cvvdp-loop-tighten` cargo feature is compiled AND
            // the field is `None` or `Some(true)`. Defaults to false in
            // every other case → hash-locks byte-identical.
            enc.cvvdp_bytes_tighten = cfg.resolve_cvvdp_bytes_tighten();
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

        // HDR intensity_target default — libjxl parity
        // (`luminance.cc:SetIntensityTarget`): PQ peaks at 10,000 nits
        // (SMPTE ST 2084), HLG's nominal display peak is 1,000 nits
        // (Rec. BT.2100-2). Without this, PQ input was linearized to
        // 1.0 = 10,000 nits (`pq_*_to_linear_f32`) while the header
        // kept intensity_target = 255 — the decoder then interpreted
        // the XYB data on a 255-nit scale, destroying the image
        // (issue #73: butteraugli ~170 at every distance, bytes
        // collapsed). Explicit metadata / request values below still
        // override, exactly like libjxl's explicit-set path.
        {
            use crate::headers::color_encoding::TransferFunction;
            if let Some(ce) = enc.color_encoding.as_ref() {
                match ce.transfer_function {
                    TransferFunction::Pq => enc.intensity_target = 10_000.0,
                    TransferFunction::Hlg => enc.intensity_target = 1_000.0,
                    _ => {}
                }
                // HDR QuantizeWP dispatch (#74 wedge, 2026-06-12): the
                // W44-AUDIT-8 Phase 7 default-flip was reverted because
                // the W44-202 per-cell SSIM2 gate failed on 4 SDR photo
                // cells — but every measured WP win is PQ/HLG content
                // (hdr_quantize_wp_ab_2026-06-12: medians +1.8..+7.9 %
                // -> +1.2..+4.7 % vs cjxl; smooth-sky e7 d4 -22 %
                // bytes; LfGroup section-diff attributes ~2/3 of the
                // remaining smooth-sky gap to the missing DC shaping).
                // Dispatch on the RESOLVED transfer function — a
                // layout-level predicate (same as the intensity
                // dispatch above), not a pixel discriminator, so there
                // is no content cliff class. SDR (every W44-202 cell)
                // is structurally unchanged. EncoderStrategy::Libjxl
                // keeps its byte-locked behaviour (strategy resolution
                // already pinned the field).
                if matches!(
                    ce.transfer_function,
                    TransferFunction::Pq | TransferFunction::Hlg
                ) && cfg.effort <= 7
                    && !matches!(cfg.strategy(), crate::api::EncoderStrategy::Libjxl)
                {
                    enc.profile.use_libjxl_wp_dc_quant = true;
                }
            }
        }
        // Tone mapping and intrinsic size from metadata
        if let Some(meta) = self.metadata {
            if let Some(it) = meta.intensity_target {
                enc.intensity_target = it;
            }
            if let Some(mn) = meta.min_nits {
                enc.min_nits = mn;
            }
            if let Some(r) = meta.relative_to_max_display {
                enc.relative_to_max_display = r;
            }
            if let Some(lb) = meta.linear_below {
                enc.linear_below = lb;
            }
            if meta.intrinsic_size.is_some() {
                enc.intrinsic_size = meta.intrinsic_size;
            }
        }
        // Request-level overrides win over metadata-level values.
        // Closes #21 (intensity pair) + issue #46 chunk 1a
        // (ToneMapping rest).
        if let Some(it) = self.intensity_target {
            enc.intensity_target = it;
        }
        if let Some(mn) = self.min_nits {
            enc.min_nits = mn;
        }
        if let Some(r) = self.relative_to_max_display {
            enc.relative_to_max_display = r;
        }
        if let Some(lb) = self.linear_below {
            enc.linear_below = lb;
        }

        // ICC profile from metadata
        if let Some(meta) = self.metadata
            && let Some(icc) = meta.icc_profile
        {
            enc.icc_profile = Some(icc.to_vec());
        }

        // Apply downsampling for resampling > 1 (refs #12). Factor 2
        // uses libjxl's sharper 12×12 kernel (`enc_heuristics.cc:279`)
        // at effort ≤ 9 and the iterative refinement
        // (`DownsampleImage2_Iterative`, decoder-upsampler adjoint) at
        // effort ≥ 10 — mirroring libjxl's `speed_tier <= kGlacier` gate
        // (`enc_frame.cc:752`, issue #45 ladder shift). Factors 4 and 8
        // use the simple box filter (libjxl behavior). When
        // `already_downsampled` is set, the caller has done their own
        // downsample and wants the encoder to honour the input dims;
        // skip the internal downsample but keep the upsampling factor
        // in the bitstream.
        let (encode_rgb, encode_alpha, encode_w, encode_h) =
            if effective_resampling > 1 && !cfg.already_downsampled {
                let (down_rgb, dw, dh) = if effective_resampling == 2 && cfg.effort >= 10 {
                    crate::vardct::resampling::iterative_downsample_2x_rgb(
                        &linear_rgb,
                        w,
                        h,
                        Some(budget),
                    )?
                } else if effective_resampling == 2 {
                    crate::vardct::resampling::sharper_downsample_2x_rgb(
                        &linear_rgb,
                        w,
                        h,
                        Some(budget),
                    )?
                } else {
                    crate::vardct::resampling::box_downsample_rgb(
                        &linear_rgb,
                        w,
                        h,
                        effective_resampling,
                        Some(budget),
                    )?
                };
                let down_alpha = match alpha.as_ref() {
                    Some(a) => {
                        let (a_down, _, _) = crate::vardct::resampling::box_downsample_alpha_u8(
                            a,
                            w,
                            h,
                            effective_resampling,
                            Some(budget),
                        )?;
                        Some(a_down)
                    }
                    None => None,
                };
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

        #[cfg(feature = "__env_var_diagnostics")]
        if std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some() {
            eprintln!("encode_lossy: post-class-setup={:?}", _t_prof.elapsed());
        }
        #[cfg(feature = "__env_var_diagnostics")]
        let _t_pre = std::time::Instant::now();
        #[cfg(feature = "__env_var_diagnostics")]
        if std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some() {
            eprintln!("encode_lossy: conv+setup={:?}", _t_conv.elapsed());
        }
        let output = enc
            .encode_with_extras_stop_src(
                encode_w,
                encode_h,
                // Ownership lets the encoder free the linear buffer after
                // the XYB conversion on loop-free efforts; nothing here
                // reads `encode_rgb` after this call.
                crate::vardct::encoder::LinearSource::Owned(encode_rgb),
                &extras_vec,
                self.stop,
            )
            .map_err(EncodeError::from)?;

        #[cfg(feature = "__env_var_diagnostics")]
        if std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some() {
            eprintln!("encode_lossy: inner-call={:?}", _t_pre.elapsed());
        }
        #[cfg(feature = "butteraugli-loop")]
        let butteraugli_iters_actual = cfg.butteraugli_iters();
        #[cfg(not(feature = "butteraugli-loop"))]
        let butteraugli_iters_actual = 0u32;

        let stats = EncodeStats {
            mode: EncodeMode::Lossy,
            strategy_counts: output.strategy_counts,
            gaborish: cfg.gaborish(),
            ans: cfg.ans(),
            butteraugli_iters: butteraugli_iters_actual,
            pixel_domain_loss: cfg.pixel_domain_loss(),
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
            ans: cfg.ans(),
            ..Default::default()
        };
        Ok((bytes, stats))
    }
}

// ── Streaming Encoders ──────────────────────────────────────────────────────

/// Streaming lossy (VarDCT) encoder.
///
/// Accepts pixel rows incrementally via [`push_rows`](Self::push_rows), then
/// encodes on [`finish`](Self::finish). Rows are converted to the internal
/// linear-RGB f32 representation as they arrive, so callers can free the
/// source pixel buffer incrementally. The converted full-image planes
/// (12 bytes/pixel for RGB) are still held in memory until `finish` —
/// streaming input bounds the caller's copy, not the encoder's peak memory.
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
    /// `ToneMapping.relative_to_max_display` (default `false`). When
    /// `true`, [`Self::linear_below`] is interpreted as a ratio in
    /// `[0, 1]` of the maximum display brightness. Issue #46 chunk 1a.
    relative_to_max_display: bool,
    /// `ToneMapping.linear_below` (default `0.0`). Issue #46 chunk 1a.
    linear_below: f32,
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
    /// encoder applies [`Limits::DEFAULT_MAX_MEMORY_BYTES`] (~4 GB) as
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

    /// Set `ToneMapping.relative_to_max_display`. When `true`,
    /// [`Self::with_linear_below`] is a ratio in `[0, 1]` rather than
    /// absolute nits. Closes issue #46 chunk 1a.
    pub fn with_relative_to_max_display(mut self, relative: bool) -> Self {
        self.relative_to_max_display = relative;
        self
    }

    /// Set `ToneMapping.linear_below`. Tone mapping leaves pixels
    /// strictly below this value unchanged. Closes issue #46 chunk 1a.
    pub fn with_linear_below(mut self, value: f32) -> Self {
        self.linear_below = value;
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
    /// soft default ([`Limits::DEFAULT_MAX_MEMORY_BYTES`], ~4 GB).
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
        self.push_rows_inner(pixels, num_rows).at()
    }

    fn push_rows_inner(&mut self, pixels: &[u8], num_rows: u32) -> Result<()> {
        if num_rows == 0 {
            return Ok(());
        }
        let remaining = self.height - self.rows_pushed;
        if num_rows > remaining {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "push_rows: {num_rows} rows would exceed image height \
                     ({} pushed + {num_rows} > {})",
                    self.rows_pushed, self.height
                ),
            }));
        }
        let w = self.width as usize;
        let n = num_rows as usize;
        let expected = w
            .checked_mul(n)
            .and_then(|wn| wn.checked_mul(self.layout.bytes_per_pixel()));
        match expected {
            Some(expected) if pixels.len() == expected => {}
            Some(expected) => {
                return Err(at!(EncodeError::InvalidInput {
                    message: format!(
                        "push_rows: expected {expected} bytes for {w}x{n} {:?}, got {}",
                        self.layout,
                        pixels.len()
                    ),
                }));
            }
            None => {
                return Err(at!(EncodeError::InvalidInput {
                    message: "push_rows: row dimensions overflow".into(),
                }));
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
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                floats.to_vec()
            }
            PixelLayout::RgbaLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                floats
                    .chunks(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect()
            }
            PixelLayout::GrayLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                gray_f32_to_linear_f32_rgb(floats, 1)
            }
            PixelLayout::GrayAlphaLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
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
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                pq_f32_to_linear_f32_rgb(floats, 3)
            }
            PixelLayout::RgbaPqF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                pq_f32_to_linear_f32_rgb(floats, 4)
            }
            PixelLayout::RgbHlgF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                hlg_f32_to_linear_f32_rgb(floats, 3)
            }
            PixelLayout::RgbaHlgF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                hlg_f32_to_linear_f32_rgb(floats, 4)
            }
            PixelLayout::RgbBt709F32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                bt709_f32_to_linear_f32_rgb(floats, 3)
            }
            PixelLayout::RgbaBt709F32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                bt709_f32_to_linear_f32_rgb(floats, 4)
            }
            // Streaming CMYK is not yet wired — only the one-shot
            // lossless path (`LosslessConfig::encode`) handles CMYK
            // input. The streaming lossy encoder would also need a
            // C/M/Y → XYB mapping (see comment on `Cmyk8` in the
            // first match site).
            PixelLayout::Cmyk8 | PixelLayout::Cmyk16 => {
                return Err(at!(EncodeError::UnsupportedPixelLayout(self.layout)));
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
                let floats: &[f32] = &cast_pixel_lanes(pixels);
                let new_alpha = extract_alpha_f32(floats, 4, 3);
                self.alpha
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&new_alpha);
            }
            PixelLayout::GrayAlphaLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(pixels);
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
                let floats: &[f32] = &cast_pixel_lanes(pixels);
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
        self.finish_inner().map(|mut r| r.take_data().unwrap()).at()
    }

    /// Encode and return JXL bytes together with [`EncodeStats`].
    #[track_caller]
    pub fn finish_with_stats(self) -> Result<EncodeResult> {
        self.finish_inner().at()
    }

    /// Encode, appending to an existing buffer.
    #[track_caller]
    pub fn finish_into(self, out: &mut Vec<u8>) -> Result<EncodeResult> {
        let mut result = self.finish_inner().at()?;
        if let Some(data) = result.data.take() {
            out.extend_from_slice(&data);
        }
        Ok(result)
    }

    /// Encode, writing to a `std::io::Write` destination.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn finish_to(self, mut dest: impl std::io::Write) -> Result<EncodeResult> {
        let mut result = self.finish_inner().at()?;
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
        let mut result = self.finish_inner().at()?;
        if let Some(data) = result.data.take() {
            dest.write_all(&data)
                .map_err(|e| at(EncodeError::from(e)))?;
        }
        Ok(result)
    }

    fn finish_inner(self) -> Result<EncodeResult> {
        if self.rows_pushed != self.height {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "incomplete image: {} of {} rows pushed",
                    self.rows_pushed, self.height
                ),
            }));
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
            return Err(at!(EncodeError::InvalidConfig {
                message: format!(
                    "chroma subsampling {} on the streaming `LossyEncoder` \
                     is not yet wired (one-shot `EncodeRequest::encode` \
                     supports it; streaming support is queued). Use \
                     `LossyConfig::new(d).with_chroma_subsampling(...).\
                     encode_request(w, h, layout).encode(&pixels)` for now.",
                    self.cfg.chroma_subsampling.tag(),
                ),
            }));
        }
        // Run the full config validator (distance, effort, iter
        // counts, mutual exclusivity). Mirrors
        // `EncodeRequest::encode_inner`.
        self.cfg.validate().map_err(at_from)?;
        // Defensive caps on caller-supplied metadata buffers (mirrors
        // EncodeRequest::encode_inner).
        validate_metadata_sizes(
            self.icc_profile.as_deref(),
            self.exif.as_deref(),
            self.xmp.as_deref(),
            self.jumbf.as_deref(),
        )?;
        // Tone-mapping numeric range checks. Stored as plain f32 / bool
        // on the encoder; pass `Some(_)` only when set away from the
        // libjxl default so a caller who never touched these knobs
        // gets the encoder default behavior. Issue #46 chunk 1a adds
        // `relative_to_max_display` and `linear_below` to the bundle.
        let it = (self.intensity_target != 255.0).then_some(self.intensity_target);
        let mn = (self.min_nits != 0.0).then_some(self.min_nits);
        let rtmd = self.relative_to_max_display.then_some(true);
        let lb = (self.linear_below != 0.0).then_some(self.linear_below);
        validate_tone_mapping_full(it, mn, rtmd, lb)?;
        validate_source_gamma(self.source_gamma)?;
        validate_intrinsic_size(self.intrinsic_size)?;
        let cfg = &self.cfg;
        let w = self.width as usize;
        let h = self.height as usize;
        let mut linear_rgb = self.linear_rgb;
        let alpha = self.alpha;

        // HLG forward OOTF — mirrors the block in
        // `EncodeRequest::encode_lossy` (same position: after
        // linearization, before unpremultiply) so `oneshot ==
        // streaming` stays byte-exact (caught by
        // `test_streaming_lossy_hlg_matches_oneshot`). Applied at
        // finish (not push) so a `with_color_encoding` /
        // `with_intensity_target` call after the first push still
        // resolves identically to the one-shot path. push_rows expands
        // gray layouts to interleaved RGB, so the 3-channel OOTF is
        // always well-formed here.
        let is_hlg_input = (self.source_gamma.is_none()
            && self.color_encoding.as_ref().is_some_and(|ce| {
                ce.transfer_function == crate::headers::color_encoding::TransferFunction::Hlg
            }))
            || matches!(
                self.layout,
                PixelLayout::RgbHlgF32 | PixelLayout::RgbaHlgF32
            );
        if is_hlg_input {
            // Streaming has no ImageMetadata channel; 255.0 is the
            // untouched SDR default (same sentinel the intensity
            // dispatch below uses), so: explicit > HLG 1,000-nit
            // default.
            let it = if self.intensity_target != 255.0 {
                self.intensity_target
            } else {
                1_000.0
            };
            if let Some(g) = hlg_ootf_gamma(it) {
                let primaries = self
                    .color_encoding
                    .as_ref()
                    .map(|c| c.primaries)
                    .unwrap_or(crate::headers::color_encoding::Primaries::Bt2100);
                apply_hlg_forward_ootf(&mut linear_rgb, hlg_ootf_luminances(primaries), g);
            }
        }

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

        // Construct the per-encode allocation budget + thread choice.
        // Streaming callers can attach a [`Limits`] via
        // [`Self::with_limits`]; otherwise the path-aware soft default
        // applies. Mirrors `EncodeRequest::encode_inner`: calibrated
        // path/effort/thread-aware estimate, thread walk-down, rejection
        // only when even the single-threaded estimate exceeds the cap.
        let preflight = encode_preflight(
            self.width,
            self.height,
            self.layout.bytes_per_pixel() as u8,
            self.layout.has_alpha(),
            false,
            cfg.effort,
            cfg.threads,
            false,
            self.limits.as_ref(),
        )?;
        let EncodePreflight {
            budget,
            threads,
            estimated_peak_bytes,
        } = preflight;

        let (codestream, mut stats) = run_with_threads(threads, || {
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
            enc.use_ans = cfg.ans();
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
            enc.error_diffusion = cfg.error_diffusion();
            enc.pixel_domain_loss = cfg.pixel_domain_loss();
            enc.pixel_loss_dispatch = enc.resolved_improvements.pixel_loss_dispatch;
            enc.single_pass_entropy_dispatch =
                enc.resolved_improvements.single_pass_entropy_dispatch;
            enc.enable_lz77 = cfg.effective_lz77();
            enc.lz77_method = cfg.lz77_method();
            enc.force_strategy = cfg.force_strategy;
            // RFC #45 pick #4 — when the caller has explicitly pinned `cfg.patches()`
            // via `with_patches`, that wins; otherwise read the per-image
            // dispatched profile (the content-class adapter may have flipped
            // patches on for Screenshot content at e5/e6).
            enc.enable_patches = if cfg.patches.is_some() {
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
            enc.auto_splines = cfg.auto_splines();
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
                enc.butteraugli_iters = cfg.butteraugli_iters();
                // EX-J11 chunk 4: see `encode_lossy` site above for
                // the resolution rationale. Auto → Vdp2 on PQ/HLG,
                // Butteraugli otherwise.
                enc.hdr_loss = cfg.resolve_hdr_loss(self.layout, self.color_encoding.as_ref());
                // #74/#11 (2026-07-15): DISABLE the perceptual quantization loop on
                // the DEFAULT HDR path (PQ/HLG → `Vdp2`) — see the `encode_lossy`
                // site above for the measured rationale + the exact gate semantics
                // (loop over-refines HDR +100..500 %; the no-loop base beats cjxl).
                // Zeroing `butteraugli_iters` reproduces `--no-butteraugli` exactly.
                // SDR + explicit-Butteraugli-on-HDR untouched (byte-identical).
                if cfg.is_hdr_pq_hlg(self.layout, self.color_encoding.as_ref())
                    && matches!(enc.hdr_loss, crate::vardct::hdr_metrics::HdrLoss::Vdp2)
                {
                    enc.butteraugli_iters = 0;
                }
                // Multi-metric Phase 0 (RFC #3, 2026-05-25): propagate
                // the resolved perceptual-metric selection (streaming
                // LossyEncoder path). Same semantics as the still-image
                // site above.
                crate::vardct::perceptual_backend::propagate_resolved_metric_to_encoder(
                    cfg.resolve_perceptual_metric_selection(),
                    &mut enc,
                );
                // cvvdp-fork Phase 8d (2026-05-25): propagate
                // bytes-tighten opt-in (streaming LossyEncoder path).
                // Same semantics as the still-image site above.
                enc.cvvdp_bytes_tighten = cfg.resolve_cvvdp_bytes_tighten();
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
            // HDR default — libjxl SetIntensityTarget parity (issue #73):
            // PQ -> 10,000 nits, HLG -> 1,000, unless the caller moved
            // intensity_target off the 255.0 SDR default explicitly.
            if self.intensity_target == 255.0 {
                use crate::headers::color_encoding::TransferFunction;
                if let Some(ce) = enc.color_encoding.as_ref() {
                    match ce.transfer_function {
                        TransferFunction::Pq => enc.intensity_target = 10_000.0,
                        TransferFunction::Hlg => enc.intensity_target = 1_000.0,
                        _ => {}
                    }
                    // HDR QuantizeWP dispatch — mirrors the one-shot
                    // site (see EncodeRequest::encode_lossy) so
                    // `oneshot == streaming` holds on PQ/HLG input.
                    if matches!(
                        ce.transfer_function,
                        TransferFunction::Pq | TransferFunction::Hlg
                    ) && self.cfg.effort <= 7
                        && !matches!(self.cfg.strategy(), crate::api::EncoderStrategy::Libjxl)
                    {
                        enc.profile.use_libjxl_wp_dc_quant = true;
                    }
                }
            }
            enc.min_nits = self.min_nits;
            enc.relative_to_max_display = self.relative_to_max_display;
            enc.linear_below = self.linear_below;
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
                // Factor-2 kernel choice mirrors the one-shot path (and
                // libjxl `enc_frame.cc:752`): iterative at effort ≥ 10,
                // sharper below.
                let (down_rgb, dw, dh) = if effective_resampling == 2 && self.cfg.effort >= 10 {
                    crate::vardct::resampling::iterative_downsample_2x_rgb(
                        &linear_rgb,
                        w,
                        h,
                        Some(&budget),
                    )?
                } else if effective_resampling == 2 {
                    crate::vardct::resampling::sharper_downsample_2x_rgb(
                        &linear_rgb,
                        w,
                        h,
                        Some(&budget),
                    )?
                } else {
                    crate::vardct::resampling::box_downsample_rgb(
                        &linear_rgb,
                        w,
                        h,
                        effective_resampling,
                        Some(&budget),
                    )?
                };
                let down_alpha = match alpha.as_ref() {
                    Some(a) => {
                        let (a_down, _, _) = crate::vardct::resampling::box_downsample_alpha_u8(
                            a,
                            w,
                            h,
                            effective_resampling,
                            Some(&budget),
                        )?;
                        Some(a_down)
                    }
                    None => None,
                };
                (down_rgb, down_alpha, dw as usize, dh as usize)
            } else {
                (linear_rgb, alpha, w, h)
            };

            let output = enc
                .encode(encode_w, encode_h, &encode_rgb, encode_alpha.as_deref())
                .map_err(EncodeError::from)?;

            #[cfg(feature = "butteraugli-loop")]
            let butteraugli_iters_actual = cfg.butteraugli_iters();
            #[cfg(not(feature = "butteraugli-loop"))]
            let butteraugli_iters_actual = 0u32;

            let stats = EncodeStats {
                mode: EncodeMode::Lossy,
                strategy_counts: output.strategy_counts,
                gaborish: cfg.gaborish(),
                ans: cfg.ans(),
                butteraugli_iters: butteraugli_iters_actual,
                pixel_domain_loss: cfg.pixel_domain_loss(),
                ..Default::default()
            };
            Ok::<_, EncodeError>((output.data, stats))
        })
        .map_err(at_from)?;

        stats.codestream_size = codestream.len();
        stats.budget_peak_bytes = budget.peak();
        stats.threads_used = threads as u32;
        stats.estimated_peak_bytes = estimated_peak_bytes;

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

mod validate;
use validate::{
    validate_dims, validate_intrinsic_size, validate_metadata_sizes, validate_source_gamma,
    validate_tone_mapping_full,
};

impl LossyConfig {
    /// Create a streaming encoder for incremental row input.
    ///
    /// Pixels are converted to the internal linear-f32 format as rows are
    /// pushed via [`LossyEncoder::push_rows`], so callers can free source
    /// buffers incrementally. The converted whole-image planes stay in
    /// memory until [`LossyEncoder::finish`] — input streaming does not
    /// bound peak encoder memory.
    #[track_caller]
    pub fn encoder(&self, width: u32, height: u32, layout: PixelLayout) -> Result<LossyEncoder> {
        validate_dims(width, height).at()?;
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
            relative_to_max_display: false,
            linear_below: 0.0,
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
/// encodes on [`finish`](Self::finish). Rows are copied into pre-allocated
/// per-channel planes as they arrive, so callers can free the source pixel
/// buffer incrementally. The full-image planes are still held in memory
/// until `finish` — streaming input bounds the caller's copy, not the
/// encoder's peak memory.
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
    /// `ToneMapping.relative_to_max_display` (default `false`). Issue
    /// #46 chunk 1a.
    relative_to_max_display: bool,
    /// `ToneMapping.linear_below` (default `0.0`). Issue #46 chunk 1a.
    linear_below: f32,
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
    /// encoder applies [`Limits::DEFAULT_MAX_MEMORY_BYTES`] (~4 GB) as
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

    /// Set `ToneMapping.relative_to_max_display`. When `true`,
    /// [`Self::with_linear_below`] is a ratio in `[0, 1]` rather than
    /// absolute nits. Closes issue #46 chunk 1a.
    pub fn with_relative_to_max_display(mut self, relative: bool) -> Self {
        self.relative_to_max_display = relative;
        self
    }

    /// Set `ToneMapping.linear_below`. Tone mapping leaves pixels
    /// strictly below this value unchanged. Closes issue #46 chunk 1a.
    pub fn with_linear_below(mut self, value: f32) -> Self {
        self.linear_below = value;
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
    /// soft default ([`Limits::DEFAULT_MAX_MEMORY_BYTES`], ~4 GB).
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
        self.push_rows_inner(pixels, num_rows).at()
    }

    fn push_rows_inner(&mut self, pixels: &[u8], num_rows: u32) -> Result<()> {
        if num_rows == 0 {
            return Ok(());
        }
        let remaining = self.height - self.rows_pushed;
        if num_rows > remaining {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "push_rows: {num_rows} rows would exceed image height \
                     ({} pushed + {num_rows} > {})",
                    self.rows_pushed, self.height
                ),
            }));
        }
        let w = self.width as usize;
        let n = num_rows as usize;
        let bpp = self.layout.bytes_per_pixel();
        let expected = w.checked_mul(n).and_then(|wn| wn.checked_mul(bpp));
        match expected {
            Some(expected) if pixels.len() == expected => {}
            Some(expected) => {
                return Err(at!(EncodeError::InvalidInput {
                    message: format!(
                        "push_rows: expected {expected} bytes for {w}x{n} {:?}, got {}",
                        self.layout,
                        pixels.len()
                    ),
                }));
            }
            None => {
                return Err(at!(EncodeError::InvalidInput {
                    message: "push_rows: row dimensions overflow".into(),
                }));
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
                let pixels_u16: &[u16] = &cast_pixel_lanes(pixels);
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
                return Err(at!(EncodeError::UnsupportedPixelLayout(self.layout)));
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
        self.finish_inner().map(|mut r| r.take_data().unwrap()).at()
    }

    /// Encode and return JXL bytes together with [`EncodeStats`].
    #[track_caller]
    pub fn finish_with_stats(self) -> Result<EncodeResult> {
        self.finish_inner().at()
    }

    /// Encode, appending to an existing buffer.
    #[track_caller]
    pub fn finish_into(self, out: &mut Vec<u8>) -> Result<EncodeResult> {
        let mut result = self.finish_inner().at()?;
        if let Some(data) = result.data.take() {
            out.extend_from_slice(&data);
        }
        Ok(result)
    }

    /// Encode, writing to a `std::io::Write` destination.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn finish_to(self, mut dest: impl std::io::Write) -> Result<EncodeResult> {
        let mut result = self.finish_inner().at()?;
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
        let mut result = self.finish_inner().at()?;
        if let Some(data) = result.data.take() {
            dest.write_all(&data)
                .map_err(|e| at(EncodeError::from(e)))?;
        }
        Ok(result)
    }

    fn finish_inner(self) -> Result<EncodeResult> {
        use crate::bit_writer::BitWriter;
        use crate::headers::color_encoding::ColorSpace;
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::channel::ModularImage;
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        if self.rows_pushed != self.height {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "incomplete image: {} of {} rows pushed",
                    self.rows_pushed, self.height
                ),
            }));
        }
        // Run the full config validator. Mirrors
        // `EncodeRequest::encode_inner`.
        self.cfg.validate().map_err(at_from)?;
        // Defensive caps on caller-supplied metadata buffers (mirrors
        // EncodeRequest::encode_inner).
        validate_metadata_sizes(
            self.icc_profile.as_deref(),
            self.exif.as_deref(),
            self.xmp.as_deref(),
            self.jumbf.as_deref(),
        )?;
        // Tone-mapping numeric range checks. See the lossy-encoder
        // mirror above for the `Some(_) iff non-default` shape. Issue
        // #46 chunk 1a extends the validator to cover
        // `relative_to_max_display` + `linear_below`.
        let it = (self.intensity_target != 255.0).then_some(self.intensity_target);
        let mn = (self.min_nits != 0.0).then_some(self.min_nits);
        let rtmd = self.relative_to_max_display.then_some(true);
        let lb = (self.linear_below != 0.0).then_some(self.linear_below);
        validate_tone_mapping_full(it, mn, rtmd, lb)?;
        validate_source_gamma(self.source_gamma)?;
        validate_intrinsic_size(self.intrinsic_size)?;

        let cfg = &self.cfg;
        let w = self.width as usize;
        let h = self.height as usize;

        // Construct the per-encode allocation budget + thread choice.
        // Mirrors the request path's calibrated pre-flight and propagates
        // the cap through to the modular FrameEncoder for hot allocation
        // sites.
        let preflight = encode_preflight_with_sectioned(
            self.width,
            self.height,
            self.layout.bytes_per_pixel() as u8,
            self.layout.has_alpha(),
            true,
            cfg.effort,
            cfg.threads,
            false,
            self.limits.as_ref(),
            cfg.sectioned_trees(),
        )?;
        let EncodePreflight {
            budget,
            threads,
            estimated_peak_bytes,
        } = preflight;

        let mut image = ModularImage {
            channels: self.channels,
            bit_depth: self.bit_depth,
            is_grayscale: self.is_grayscale,
            has_alpha: self.has_alpha,
        };

        let (codestream, mut stats) = run_with_threads(threads, || {
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
                    Some(&budget),
                )
                .map_err(EncodeError::from)?;
                // RFC#45 chunks 4-7 lossless backport (chunk 5 lossless
                // trial encoder): per-image cost gate (see
                // `PatchesData::is_cost_effective_lossless`).
                pd_opt.filter(|pd| pd.is_cost_effective_lossless(image.bit_depth, cfg.ans()))
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
            file_header.metadata.relative_to_max_display = self.relative_to_max_display;
            file_header.metadata.linear_below = self.linear_below;
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
                    cfg.ans(),
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
            let mut use_tree_learning_l = cfg.effective_tree_learning();
            let mut smart_profile = cfg.effective_profile_for_image((w as u64) * (h as u64));
            // Issue #72: budgeted tree learning for 16-bit RGB(A) at e5/e6.
            use_tree_learning_l |= cfg.lift_integer_tree_learning(
                self.layout,
                (w as u64) * (h as u64),
                &mut smart_profile,
            );
            let frame_encoder = FrameEncoder::new(
                w,
                h,
                FrameEncoderOptions {
                    use_modular: true,
                    effort: cfg.effort,
                    use_ans: cfg.ans(),
                    use_tree_learning: use_tree_learning_l,
                    use_squeeze: cfg.squeeze,
                    enable_lz77: cfg.effective_lz77(),
                    lz77_method: cfg.lz77_method(),
                    lossy_palette: cfg.lossy_palette,
                    encoder_mode: cfg.mode,
                    profile: smart_profile,
                    // imazen/jxl-encoder#96: the streaming path honours the
                    // same sectioned-trees knob as the one-shot request
                    // (it defaulted to `Auto` here regardless of the config
                    // before 2026-08-27).
                    sectioned_trees: cfg.sectioned_trees(),
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
                    None,
                )
                .map_err(EncodeError::from)?;

            let stats = EncodeStats {
                mode: EncodeMode::Lossless,
                ans: cfg.ans(),
                ..Default::default()
            };
            Ok::<_, EncodeError>((writer.finish_with_padding(), stats))
        })
        .map_err(at_from)?;

        stats.codestream_size = codestream.len();
        stats.budget_peak_bytes = budget.peak();
        stats.threads_used = threads as u32;
        stats.estimated_peak_bytes = estimated_peak_bytes;

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
    /// [`LosslessEncoder::push_rows`], so callers can free source buffers
    /// incrementally. The full-image planes stay in memory until
    /// [`LosslessEncoder::finish`] — input streaming does not bound peak
    /// encoder memory.
    #[track_caller]
    pub fn encoder(&self, width: u32, height: u32, layout: PixelLayout) -> Result<LosslessEncoder> {
        use crate::modular::channel::Channel;

        validate_dims(width, height).at()?;

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
            relative_to_max_display: false,
            linear_below: 0.0,
            intrinsic_size: None,
            premultiplied_alpha: false,
            bits_per_sample: None,
            brotli_metadata_quality: None,
            limits: None,
        })
    }
}

// ── Memory-budget pre-flight (shared by all encode entry points) ────────────

/// Outcome of [`encode_preflight`]: the per-encode budget plus the
/// thread count the encode should run with and the estimate that
/// admitted it. `pub(crate)` for the api_tests walk-down unit tests.
pub(crate) struct EncodePreflight {
    pub(crate) budget: alloc::sync::Arc<crate::budget::MemoryBudget>,
    /// Thread count to hand to `run_with_threads`. Equal to the caller's
    /// request when the estimate fits at that width; walked down toward 1
    /// when it doesn't. 0 = ambient pool (kept only when the ambient
    /// width's estimate fits — otherwise a concrete reduced count).
    pub(crate) threads: usize,
    /// Calibrated peak estimate (bytes) at the chosen thread count.
    pub(crate) estimated_peak_bytes: u64,
}

/// Detected available system RAM in bytes (Linux `/proc/meminfo`
/// `MemAvailable`). `None` on other platforms, in `no_std` builds, or
/// when the read/parse fails — callers must treat `None` as "unknown",
/// never as zero.
#[cfg(all(feature = "std", target_os = "linux"))]
fn available_ram_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

#[cfg(not(all(feature = "std", target_os = "linux")))]
fn available_ram_bytes() -> Option<u64> {
    None
}

/// Ambient worker-thread count for `threads == 0` (the current rayon
/// pool's width — the pool `run_with_threads(0, …)` will execute on).
#[cfg(feature = "parallel")]
fn ambient_thread_count() -> usize {
    rayon::current_num_threads().max(1)
}

#[cfg(not(feature = "parallel"))]
fn ambient_thread_count() -> usize {
    1
}

/// Shared memory pre-flight for every encode entry point (one-shot
/// request, streaming lossy/lossless finish, animation). Replaces the
/// former flat `width × height × 40` estimate (effort-, path- and
/// thread-blind — it admitted 108 MP lossy encodes that peak ≥ 30 GiB,
/// measured `benchmarks/jxl_encode_mem_threads_2026-08-01.tsv`) with the
/// calibrated [`crate::heuristics::estimate_encode_threaded`] model.
///
/// Semantics:
/// - The budget cap is the caller's `Limits::max_memory_bytes` if set,
///   else the path-aware default ([`Limits::default_max_memory_bytes`]).
/// - The estimate is evaluated at the requested thread count
///   (`requested_threads`, 0 = the ambient pool width). When it doesn't
///   fit the cap — or the detected-available-RAM soft ceiling (×0.8, so
///   an unset-limits caller on a busy box degrades threads instead of
///   getting the box OOM-killed) — the thread count walks down until the
///   estimate fits, flooring at 1.
/// - **Rejection is budget-driven only**: the encode is refused
///   (`LimitExceeded`) exactly when even the 1-thread estimate exceeds
///   the budget cap. Availability only reduces threads, never rejects.
/// - `admission_only`: entry points that cannot control their pool
///   (animation frames run on the ambient pool) pass `true` — the
///   1-thread admission check still applies but no thread walk-down is
///   attempted and `threads` echoes the request.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_preflight(
    width: u32,
    height: u32,
    input_bpp: u8,
    has_alpha: bool,
    is_lossless: bool,
    effort: u8,
    requested_threads: usize,
    admission_only: bool,
    limits: Option<&Limits>,
) -> Result<EncodePreflight> {
    encode_preflight_with_sectioned(
        width,
        height,
        input_bpp,
        has_alpha,
        is_lossless,
        effort,
        requested_threads,
        admission_only,
        limits,
        SectionedTrees::Off,
    )
}

/// [`encode_preflight`] aware of the lossless sectioned local-tree mode
/// (imazen/jxl-encoder#96). `sectioned` is the config's
/// [`SectionedTrees`] knob; the estimate consulted for admission, the
/// thread walk-down and `EncodeStats::estimated_peak_bytes` is the one for
/// the tree mode the modular frame encoder will ACTUALLY run at that
/// thread count (a mirror of its gate — see `path_est_at` below), so:
///
/// - `On` / `Auto`-under-pressure / `Auto`-e≤7-multithreaded → the
///   calibrated sectioned estimate
///   ([`crate::heuristics::estimate_encode_sectioned`]: image-copy +
///   pre-tree-phase floor plus one group's tree-learn working set per
///   worker), which is far below the whole-image band — a 21 MP e7
///   lossless encode is admitted under the 8 GiB default cap because its
///   sectioned peak fits, not because admission was switched off.
/// - Everything else (`Off`, `Hybrid`, non-lossless, efforts outside the
///   calibrated 7–9 band, `Auto` where the gate keeps the global tree) →
///   the whole-image estimate, exactly as [`encode_preflight`].
///
/// Rejection stays budget-driven: the encode is refused only when the
/// 1-thread estimate of the path that would run exceeds the cap. The
/// runtime `MemoryBudget` still enforces the cap allocation-by-allocation
/// on the paths the sectioned writer does not cover (custom DC quant,
/// non-tree / non-ANS modes — the estimate is then an under-prediction,
/// and the encode fails cleanly mid-way rather than exceeding the cap).
/// Palette / ChannelCompact / patches content has run sectioned since
/// 2026-08-28 and is inside the calibrated band.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_preflight_with_sectioned(
    width: u32,
    height: u32,
    input_bpp: u8,
    has_alpha: bool,
    is_lossless: bool,
    effort: u8,
    requested_threads: usize,
    admission_only: bool,
    limits: Option<&Limits>,
    sectioned: SectionedTrees,
) -> Result<EncodePreflight> {
    let budget_cap = limits
        .and_then(|l| l.max_memory_bytes())
        .unwrap_or(Limits::default_max_memory_bytes(is_lossless));
    let fallible = limits.is_some_and(|l| l.fallible_alloc());

    let overflow = || {
        at!(EncodeError::LimitExceeded {
            message: format!("image {width}x{height} too large for working-set estimate"),
        })
    };
    let whole_at = |threads: usize| -> Result<u64> {
        crate::heuristics::estimate_encode_threaded(
            width,
            height,
            input_bpp,
            has_alpha,
            is_lossless,
            effort,
            threads,
        )
        .map(|e| e.peak_memory_bytes)
        .ok_or_else(overflow)
    };

    // The sectioned arm exists for lossless tree-learning encodes in the
    // calibrated effort band, with a knob that can engage it. `Hybrid`
    // learns the global tree too (global-mode memory), so it keeps the
    // whole-image estimate.
    let sectioned_arm = is_lossless
        && crate::heuristics::sectioned_estimate_available(effort)
        && matches!(sectioned, SectionedTrees::On | SectionedTrees::Auto);
    // Mirror of the frame encoder's `Auto` gate
    // (`modular::frame::auto_tree_mode` + its `memory_pressure` input,
    // which estimates at the same 3-byte input term regardless of layout):
    // sectioned when the whole-image estimate does not fit the cap, or at
    // effort <= 7 with more than one worker (2026-08-19 policy). Without
    // the `parallel` feature the encoder's effective thread count is
    // always 1, so the thread arm cannot fire there.
    let auto_pressure = sectioned_arm
        && matches!(sectioned, SectionedTrees::Auto)
        && crate::heuristics::estimate_encode(width, height, 3, has_alpha, true, effort)
            .is_some_and(|e| e.peak_memory_bytes > budget_cap);
    let sectioned_engages = |threads: usize| -> bool {
        sectioned_arm
            && match sectioned {
                SectionedTrees::On => true,
                SectionedTrees::Auto => {
                    auto_pressure
                        || (cfg!(feature = "parallel") && effort <= 7 && threads.max(1) > 1)
                }
                SectionedTrees::Off | SectionedTrees::Hybrid => false,
            }
    };
    // Estimate for the tree mode that will run at `threads`.
    let path_est_at = |threads: usize| -> Result<(u64, bool)> {
        if sectioned_engages(threads) {
            crate::heuristics::estimate_encode_sectioned(
                width, height, input_bpp, has_alpha, effort, threads,
            )
            .map(|e| (e.peak_memory_bytes, true))
            .ok_or_else(overflow)
        } else {
            whole_at(threads).map(|e| (e, false))
        }
    };

    let start = if requested_threads == 0 {
        ambient_thread_count()
    } else {
        requested_threads
    }
    .max(1);

    // Budget-driven admission floor: the smallest estimate the walk-down
    // can reach must fit. Taken as the smaller of the 1- and 2-worker
    // estimates when the request allows ≥ 2 workers: the sectioned band
    // once measured a single-worker excess two workers did not have
    // (2026-08-27: 12 MP e7 855 vs 584 MiB — the patches detector's DFS
    // stack, removed 2026-08-28, both arms now carry one floor), and the
    // min form keeps the walk-down below from ever stepping into a larger
    // estimate should an arm diverge again.
    let (est_t1, t1_sectioned) = path_est_at(1)?;
    let (est_floor, floor_sectioned) = if !admission_only && start >= 2 {
        let (est_t2, t2_sectioned) = path_est_at(2)?;
        if est_t2 < est_t1 {
            (est_t2, t2_sectioned)
        } else {
            (est_t1, t1_sectioned)
        }
    } else {
        (est_t1, t1_sectioned)
    };
    if est_floor > budget_cap {
        return Err(at!(EncodeError::LimitExceeded {
            message: format!(
                "estimated peak working set {est_floor} bytes for {width}x{height} \
                 {}{} effort {effort} (minimum-thread floor) exceeds memory \
                 budget cap {budget_cap}; raise the cap via \
                 Limits::with_max_memory_bytes if this encode is intended",
                if is_lossless { "lossless" } else { "lossy" },
                if floor_sectioned {
                    " (sectioned local trees)"
                } else {
                    ""
                },
            ),
        }));
    }

    let (threads, estimated_peak_bytes) = if admission_only {
        (requested_threads, est_t1)
    } else {
        // Thread choice target: the budget cap, additionally soft-capped
        // by detected available RAM × 0.8 (availability shapes the thread
        // count only — the rejection above is already done).
        let thread_target = match available_ram_bytes() {
            Some(avail) => budget_cap.min(avail.saturating_mul(4) / 5),
            None => budget_cap,
        };
        let mut t = start;
        let mut est = path_est_at(t)?.0;
        while t > 1 && est > thread_target {
            // Walking down only helps while the estimate falls: a
            // thread-invariant band (lossless global, γ = 0) or the
            // sectioned 1-worker excess would otherwise trade wall time
            // for no memory (or for MORE), and could step an admitted
            // request into an estimate above the cap.
            let next = path_est_at(t - 1)?.0;
            if next >= est {
                break;
            }
            t -= 1;
            est = next;
        }
        if t == start {
            // No reduction needed: preserve the caller's exact request
            // (notably 0 = ambient pool, which `run_with_threads` treats
            // specially — byte-and-pool-identical to the pre-existing
            // behaviour).
            (requested_threads, est)
        } else {
            (t, est)
        }
    };

    Ok(EncodePreflight {
        budget: crate::budget::MemoryBudget::with_alloc_policy(budget_cap, fallible),
        threads,
        estimated_peak_bytes,
    })
}

// ── Thread pool helper ──────────────────────────────────────────────────────

/// Run a closure inside a rayon thread pool when the `parallel` feature
/// is enabled and `threads > 1`. Otherwise, just call the closure directly.
///
/// - `threads == 0`: use the ambient rayon pool (caller controls via
///   `pool.install()` or the global default).
/// - `threads == 1`: direct call — note the encode body's internal
///   `parallel_map` / `par_sort` still target the AMBIENT pool, so `1`
///   does NOT mean sequential internals (historic behaviour; output is
///   thread-count byte-invariant — 4-way sha256 check, 2026-06-10).
///   Two alternatives were measured and REJECTED
///   (`benchmarks/perf_pool1t_2026-06-10.meta`): a true one-worker pool
///   (+70..+195 % wall — the ambient-width parallelism dwarfs the
///   per-call cold-bridge toll) and a `rayon::join` warm entry
///   (+1.7..+3.7 % on 3 of 4 cells). The `Registry::in_worker_cold` /
///   `LockLatch` frames in profiles are bridge FRAMES carrying the
///   enclosed work's samples, not recoverable overhead — don't respawn
///   this without new structure.
/// - `threads >= 2`: create a dedicated pool with that many threads.
#[cfg(feature = "parallel")]
fn run_with_threads<T>(threads: usize, f: impl FnOnce() -> T + Send) -> T
where
    T: Send,
{
    if threads == 0 {
        // Documented contract: ambient rayon pool (caller controls via
        // pool.install around the encode call).
        return f();
    }
    // threads >= 1: dedicated pool of exactly that size. `threads == 1`
    // MUST install a 1-thread pool — the pre-2026-06-12 early-return ran
    // the closure on the ambient GLOBAL pool, so `with_threads(1)`
    // silently used every core for the rayon stages (violating the
    // documented "force sequential" contract and producing bogus
    // 1T wall benchmarks — #74 wall-grid postmortem).
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

mod animate;
use animate::{encode_animation_lossless, encode_animation_lossy};

// ── Pixel conversion helpers ────────────────────────────────────────────────

/// Pre-computed sRGB u8 → linear f32 lookup table (256 entries).
/// Eliminates per-pixel `powf(2.4)` calls for the common 8-bit path.
mod ingest;
use ingest::*;

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
