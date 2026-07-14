// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Encoder strategy presets and the gate/dispatch policy enum surface:
//! `EncoderMode`, `EncoderStrategy`, `StrategyOverrides`, the per-feature
//! dispatch policies, `EpfSharpnessSeed`, and `EffortGate`. Gate resolution
//! delegates to `crate::gate_registry`.

/// The **divergence-policy** axis: whether the encoder matches libjxl's
/// algorithm choices (`Reference`) or uses its own improvements
/// (`Experimental`).
///
/// Both modes produce valid JPEG XL bitstreams decodable by any conformant
/// decoder. The difference is in *encoder-side* decisions: strategy selection
/// heuristics, cost models, entropy coding parameters, tree learning, etc.
///
/// Do **not** confuse with [`EncodeMode`] (`Lossy` / `Lossless`) — that is the
/// orthogonal *compression-kind* axis (the one-letter-apart names are an
/// unfortunate historical clash). This coarse two-way policy is also largely
/// subsumed by the finer [`EncoderStrategy`] bundle (`Reference` ≈
/// [`EncoderStrategy::Libjxl`], `Experimental` ≈ [`EncoderStrategy::Zenjxl`]);
/// prefer `EncoderStrategy` for new code.
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
/// **Default**: [`EpfDispatch::Auto`] (flipped 2026-06-12, #74 wedge 5;
/// was `AlwaysSelect`). The measured RD pass the previous default's
/// doc demanded: on smooth content the per-block search was actively
/// counterproductive — the HDR smooth-sky cell paid ~400 B coding a
/// noisy sharpness map (29/64 varblocks nonzero vs cjxl's 7/64) AND
/// scored WORSE than the uniform default (PQ-butteraugli 1.819 vs
/// 1.778; bytes 1,487 → 1,091, beating cjxl's 1,230). Textured
/// content keeps the search (the mask-mean discriminator only fires
/// on smooth); CID22 GRAND + W44-202 photo cells unchanged in the
/// flip validation. Hash-locks re-baked per the protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EpfDispatch {
    /// Always run the per-block sharpness selection when
    /// the underlying gate (`epf_iters > 0 && distance >= 0.5 &&
    /// profile.epf_dynamic_sharpness`) is satisfied. Byte-identical
    /// to pre-2026-06-12 encoder behaviour.
    AlwaysSelect,
    /// Always force the uniform default sharpness (4) and skip the
    /// per-block search. Cheap; gives up the per-block tuning win.
    /// Use this when you've measured that the search isn't worth the
    /// CPU on your content.
    AlwaysDefault,
    /// **Default.** Run the per-block selection only when a `mask1x1`-based
    /// smoothness predicate says the input has enough texture/edges
    /// to benefit. On smooth regions the uniform default sharpness
    /// is written without invoking the search. Bitstream-affecting
    /// on the gated subset; behaviour matches [`Self::AlwaysSelect`]
    /// on content the predicate doesn't gate.
    #[default]
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
///
///
/// **W44-193 (2026-05-22)**: this struct is now generated by the
/// [`jxl_encoder_macros::strategy_def!`] proc-macro invocation in
/// [`crate::gate_registry`] (production big-bang migration per the
/// W44-190 RFC + W44-192 prototype + user 2026-05-22 signoff on the
/// single-PR approach). The struct lives in `crate::gate_registry`
/// as `CustomEncoderImprovements` and is re-exported here under its
/// historical public-API name. Field-by-field layout (and the
/// `..Default::default()` struct-update idiom) is preserved byte-for-
/// byte — every existing call site continues to work unchanged. Tests
/// in `crate::gate_registry::tests` and in `tests::test_w44_*` below
/// pin the default field values + per-strategy ctor output byte-for-
/// byte against the pre-W44-193 hand-written defaults.
///
/// See [`crate::gate_registry`] for the macro invocation +
/// per-gate divergence-table metadata (consumed by the W44-194
/// build-script that will auto-generate `docs/LIBJXL_DIVERGENCES.md`)
/// + the W44-120 dual-env-var supplemental fallback.
pub use crate::gate_registry::CustomEncoderImprovements as EncoderImprovementsCustom;

/// `pub(crate)` re-export of the macro-generated
/// `CustomResolvedImprovements`. Not part of the public API. Call
/// sites in `crate::vardct` read fields directly via this alias.
pub(crate) use crate::gate_registry::CustomResolvedImprovements as ResolvedImprovements;

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
/// W44-193: env-var fallback for the four promoted env-only knobs.
///
/// Delegates to [`crate::gate_registry::apply_env_var_fallbacks`]
/// which composes the macro-generated single-env fallback layer
/// (for `JXL_W44_117_DISABLE`, `JXL_BUTTLOOP_INITIAL_QF_SCALE`,
/// `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`,
/// `JXL_W44_184_FORCE_LIBJXL_NEWTON`) with the hand-written W44-120
/// supplement (for `JXL_W44_120_EPF_SEED_MIN_DISTANCE`, which feeds
/// the same gate as `JXL_W44_117_DISABLE` — the macro syntax
/// supports only one env hook per gate). Precedence preserved
/// byte-identically vs the pre-W44-193 hand-written fn (the W44-117
/// disable runs first and sets the field to a non-default value,
/// which short-circuits the W44-120 supplement's `field == default`
/// check — identical to the pre-W44-193 if/else-if pattern).
fn apply_env_var_fallbacks(r: &mut ResolvedImprovements) {
    crate::gate_registry::apply_env_var_fallbacks(r);
}

// W44-128 Chunk B: `EncoderStrategy::resolve` now runs at every
// `VarDctEncoder` construction site (still-image, streaming,
// animation) via `LossyConfig::resolve_improvements`, which
// transitively keeps `libjxl`/`lean_faster`/`zenjxl`/`aggressive`/
// `from_custom` reachable. The W44-127-era `#[allow(dead_code)]` on
// the impl block was removed.

// ── LossyConfig ─────────────────────────────────────────────────────────────
