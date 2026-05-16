// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Centralized effort-derived encoder decisions.
//!
//! Every effort-gated decision in the encoder reads from an [`EffortProfile`]
//! instead of checking `if effort >= N` inline. Construct once from
//! `(effort, mode)`, then pass to all subsystems.

use crate::api::EncoderMode;
use crate::entropy_coding::lz77::Lz77Method;

/// Per-strategy raw entropy multipliers for the AC strategy cost model.
///
/// These control the relative preference for each transform type in AC strategy
/// selection. Higher values penalize a strategy (making it less likely to be chosen);
/// lower values favor it. The 8x8-class values are normalized by DCT8's value before
/// use, so DCT8 always evaluates at 1.0. Larger transforms use raw values directly.
///
/// Default values match libjxl `enc_ac_strategy.cc:584` (`kTransforms8x8[i].entropy_mul`).
/// Experimental values from libjxl PR #4506 (Jon Sneyers, VarDCT cost tuning).
///
/// `#[non_exhaustive]` so future libjxl-side strategy additions can land
/// without a breaking change. Construct via [`Self::reference`] or
/// [`Self::experimental`] and mutate fields as needed.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EntropyMulTable {
    /// DCT8 base value. All 8x8-class transforms are normalized by this.
    /// Reference: 0.8 (libjxl `enc_ac_strategy.cc:357`, `kTransforms8x8[0].entropy_mul`).
    pub dct8: f32,

    /// DCT4x4 (four 4x4 sub-blocks per 8x8 block).
    /// Reference: 1.08. Experimental: 0.88 (PR #4506, ~19% reduction).
    /// Lowering favors DCT4x4 for textured/detailed regions (screenshots, text).
    pub dct4x4: f32,

    /// DCT4x8 / DCT8x4 (half-block transforms for edges/detail).
    /// Reference: 0.859316 (libjxl `enc_ac_strategy.cc`).
    pub dct4x8: f32,

    /// Identity (pixel copy, no transform).
    /// Reference: 1.0428. Experimental: 0.88 (PR #4506, ~16% reduction).
    /// Lowering favors identity blocks for flat/noisy regions.
    pub identity: f32,

    /// DCT2x2 (2x2 Hadamard-like transform).
    /// Reference: 0.95 (libjxl `enc_ac_strategy.cc`).
    pub dct2x2: f32,

    /// AFV (Adaptive Frequency Variable, corner DCT).
    /// Reference: 0.818. Experimental: 0.75 (PR #4506, ~8% reduction).
    /// Lowering favors AFV for edge blocks with mixed content.
    pub afv: f32,

    /// DCT16x8 / DCT8x16 (larger transforms use raw values, not normalized by DCT8).
    /// Reference: 1.21 (libjxl `enc_ac_strategy.cc`).
    pub dct16x8: f32,

    /// DCT16x16.
    /// Reference: 1.34 (libjxl `enc_ac_strategy.cc`).
    pub dct16x16: f32,

    /// DCT16x32 / DCT32x16.
    /// Reference: 1.49 (libjxl `enc_ac_strategy.cc`).
    pub dct16x32: f32,

    /// DCT32x32.
    /// Reference: 1.48 (libjxl `enc_ac_strategy.cc`).
    pub dct32x32: f32,

    /// DCT64x32 / DCT32x64.
    /// Reference: 2.25 (libjxl `enc_ac_strategy.cc`).
    pub dct64x32: f32,

    /// DCT64x64.
    /// Reference: 2.25 (libjxl `enc_ac_strategy.cc`).
    pub dct64x64: f32,
}

impl EntropyMulTable {
    /// Default values matching libjxl `enc_ac_strategy.cc:584`.
    pub fn reference() -> Self {
        Self {
            dct8: 0.8,
            dct4x4: 1.08,
            dct4x8: 0.859_316_37,
            identity: 1.0428,
            dct2x2: 0.95,
            afv: 0.817_794_9,
            dct16x8: 1.21,
            dct16x16: 1.34,
            dct16x32: 1.49,
            dct32x32: 1.48,
            dct64x32: 2.25,
            dct64x64: 2.25,
        }
    }

    /// Experimental values from libjxl PR #4506 (Jon Sneyers, VarDCT cost tuning).
    ///
    /// Changes vs reference:
    /// - dct4x4: 1.08 → 0.88 (~19% reduction) — favor detail-preserving 4x4 sub-blocks
    /// - identity: 1.0428 → 0.88 (~16% reduction) — favor pixel-copy for flat regions
    /// - afv: 0.818 → 0.75 (~8% reduction) — favor corner DCT for edge blocks
    pub fn experimental() -> Self {
        Self {
            dct4x4: 0.88,
            identity: 0.88,
            afv: 0.75,
            ..Self::reference()
        }
    }
}

/// All effort-derived encoder decisions, centralized.
///
/// Replaces scattered `if effort >= N` checks throughout the codebase.
/// Construct once from (effort, mode, encoding path), pass to all subsystems.
///
/// **Field categories**:
/// - **Effort-derived**: changes value across effort levels (e.g., `nb_rcts_to_try`,
///   `tree_max_buckets`, `butteraugli_iters`).
/// - **Tuning constants**: same value at every effort in the reference profile,
///   mode-dependent in experimental (e.g., `k_favor_2x2`, `k_info_loss_mul_base`,
///   `entropy_mul_table`, `k8x8` etc.). The picker can dial these independently
///   of effort.
///
/// `#[non_exhaustive]` so we can grow the field set as the picker discovers new
/// useful knobs without breaking external `EffortProfile { ... }` constructions.
/// Construct via [`Self::lossy`] or [`Self::lossless`] and mutate fields as needed.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EffortProfile {
    /// The raw effort level (1–10).
    pub effort: u8,

    // ─── Feature flags ───────────────────────────────────────────────────
    /// Use ANS entropy coding instead of Huffman.
    pub use_ans: bool,
    /// Use two-pass mode with optimized entropy codes.
    pub optimize_codes: bool,
    /// Use custom coefficient ordering (AC scan order from statistics).
    pub custom_orders: bool,
    /// Enable gaborish inverse pre-filter.
    pub gaborish: bool,
    /// Enable pixel-domain loss in AC strategy selection.
    pub pixel_domain_loss: bool,
    /// Enable error diffusion in AC quantization.
    pub error_diffusion: bool,
    /// Enable patches/dictionary detection.
    pub patches: bool,
    /// Enable content-adaptive MA tree learning (modular path).
    pub tree_learning: bool,
    /// Enable LZ77 backward references in entropy coding.
    pub lz77: bool,
    /// LZ77 method when lz77 is enabled.
    pub lz77_method: Lz77Method,
    /// Number of butteraugli quantization loop iterations.
    pub butteraugli_iters: u32,

    // ─── AC strategy search ──────────────────────────────────────────────
    /// Enable adaptive AC strategy selection (multi-block transforms).
    pub ac_strategy_enabled: bool,
    /// Try DCT16x16/DCT16x8/DCT8x16 transforms (multi-block 16x16 merges).
    pub try_dct16: bool,
    /// Try DCT32x32/DCT32x16/DCT16x32 transforms.
    pub try_dct32: bool,
    /// Try DCT64x64/DCT64x32/DCT32x64 transforms.
    pub try_dct64: bool,
    /// Try DCT4x8/DCT8x4/DCT4x4/AFV transforms (effort >= 6 in libjxl).
    pub try_dct4x8_afv: bool,
    /// Enable non-aligned evaluation pass (odd-aligned 16x16 regions).
    pub non_aligned_eval: bool,
    /// Step size for fine-grained AC strategy search on 32x32+ blocks.
    /// 1 = every position (effort 9+), 2 = every other (default).
    pub fine_grained_step: u8,

    // ─── VarDCT pipeline options ──────────────────────────────────────────
    /// Apply pixel-level chromacity adjustments (effort >= 7 in libjxl).
    pub chromacity_adjustment: bool,
    /// Use pair-merge clustering for VarDCT entropy codes (effort >= 9 in libjxl).
    /// When false, uses fast k-means-only clustering.
    pub enhanced_clustering_vardct: bool,
    /// Optimize per-histogram HybridUint configs for VarDCT entropy codes.
    /// libjxl uses uint_method=kNone (no optimization, default {4,2,0}) at effort < 9.
    /// The fast optimization picks non-default configs whose signaling overhead
    /// exceeds their coding benefit on VarDCT token distributions.
    pub optimize_uint_configs_vardct: bool,
    /// Compute per-block dynamic EPF sharpness (effort >= 6 in libjxl).
    pub epf_dynamic_sharpness: bool,
    /// Recompute CfL map after initial quantization for better estimates (effort >= 7 in libjxl).
    pub cfl_two_pass: bool,
    /// Use Newton's method (perceptual cost model) for CfL fitting (effort >= 7 in libjxl).
    /// When false, uses fast least-squares fitting (quadratic cost, single-pass).
    pub cfl_newton: bool,
    /// Newton finite-difference epsilon for CfL fitting.
    /// Controls second-derivative accuracy. Default 1.0 (libjxl uses 100.0, which oscillates).
    pub cfl_newton_eps: f32,
    /// Maximum Newton iterations for CfL fitting. Default 10 (libjxl uses 20).
    pub cfl_newton_max_iters: usize,

    // ─── Quantization ────────────────────────────────────────────────────
    /// Use adaptive (content-dependent) quant field via InitialQuantField.
    /// When false (effort < 5), uses flat quant field = 0.79/distance.
    /// Matches libjxl enc_heuristics.cc:1097-1128.
    pub use_adaptive_quant: bool,
    /// Enable per-block AdjustQuantBlockAC (effort >= 5 in libjxl).
    pub adjust_quant_ac: bool,
    /// Numerator for the effort-fixed q parameter used in global_scale computation.
    /// libjxl: 0.39 at effort >= 5, 0.79 at effort < 5.
    /// global_scale = 65536 * (initial_q_numerator / distance) / 5.0
    pub initial_q_numerator: f32,
    /// Fixed quantization thresholds applied per-coefficient on the Y channel
    /// when [`Self::adjust_quant_ac`] is `false`.
    ///
    /// Pipeline stage: VarDCT post-DCT quantization (`vardct/transform.rs`).
    /// The four entries gate progressively higher coefficient bands; values
    /// below the threshold round to zero.
    /// From libjxl `enc_group.cc:358` (`kThresholdMul` constants for low-effort path).
    /// Lowering the entries preserves more high-frequency Y detail at the cost
    /// of bitrate; raising flattens texture. Override when an asset class needs
    /// different texture-vs-bitrate balance than the libjxl defaults give.
    pub fixed_thresholds_y: [f32; 4],
    /// Initial quantization thresholds used when [`Self::adjust_quant_ac`] is
    /// `true` (effort >= 5). Per-block adjustment iterates from these.
    /// From libjxl `enc_group.cc:390`.
    /// Pipeline stage: VarDCT post-DCT quantization, prior to the
    /// `AdjustQuantBlockAC` per-block tweak. Useful as a starting point for
    /// pickers exploring the threshold-vs-rate frontier per content class.
    pub adjust_thresholds: [f32; 4],

    // ─── Cost model constants ────────────────────────────────────────────
    // All five `k_*` constants below feed `vardct/ac_strategy_search.rs`
    // (the per-8×8 cost evaluator that picks DCT8 vs DCT4x4 vs IDENTITY vs
    // larger merges). Default values come from libjxl's reference encoder
    // and are *the same at every effort level* — they describe the cost
    // model itself, not the search depth. The picker / sweep harness uses
    // them to retune the model per content class without touching effort.
    /// kFavor2X2AtHighQuality weight (-0.4 in libjxl,
    /// `enc_ac_strategy.cc::kFavor2X2AtHighQuality`).
    /// Applied as `k_favor_2x2 * ((5-distance)/5)^2` to IDENTITY/DCT2X2
    /// entropy at distance < 5. More-negative values aggressively favor
    /// pixel-copy / 2×2 blocks at low distances; useful for screenshots /
    /// pixel art where the default photo-tuned bias under-uses IDENTITY.
    pub k_favor_2x2: f32,
    /// Base penalty added to every non-DCT8 strategy's cost
    /// (libjxl `kAvoidEntropyOfTransforms = 0.5`,
    /// `enc_ac_strategy.cc::EvalAcStrategy`). Higher values discourage the
    /// AC strategy search from leaving DCT8; lower values let it spread to
    /// IDENTITY / DCT4x4 / DCT16x16 more freely.
    pub k_avoid_transforms_base: f32,
    /// Base multiplier on the IDCT-domain (pixel-domain) error term in
    /// `EstimateEntropy` (libjxl 1.2, `enc_ac_strategy.cc`).
    /// PR #4506 raised this to 1.3 for the experimental profile — heavier
    /// weight on visible artifacts vs coefficient-domain entropy.
    pub k_info_loss_mul_base: f32,
    /// Base multiplier on the zero-coefficient cost term (libjxl 9.309,
    /// `enc_ac_strategy.cc`). Increasing rewards strategies that leave
    /// many coefficients exactly zero (boosts large-DCT use on smooth
    /// regions). Lowering lets non-zero residuals stay cheaper.
    pub k_zeros_mul_base: f32,
    /// Base delta added inside the cost-model interpolation (libjxl 10.833,
    /// `enc_ac_strategy.cc`). Acts as an "exchange rate" between rate
    /// (entropy proxy) and distortion (info-loss term); rarely retuned
    /// outside picker/sweep work.
    pub k_cost_delta_base: f32,
    /// Quantization-cost constant used when materializing the initial
    /// quant field (libjxl 0.765, `enc_adaptive_quantization.cc`). Read by
    /// `vardct/precomputed.rs` and `vardct/encoder.rs`. Lower values
    /// produce a coarser initial field (less rate, more distortion);
    /// higher refines.
    pub k_ac_quant: f32,

    // ─── Coefficient-domain multiplier constants ─────────────────────────
    // Each tuple is `(mul1, mul2, base)` for the EstimateEntropy /
    // info-loss formula in `vardct/ac_strategy_search.rs`. `mul1` weights
    // the negative log-rate term, `mul2` weights the AC magnitude term,
    // and `base` is added unconditionally. Defaults come from libjxl's
    // `enc_ac_strategy.cc`. Mode-/effort-independent in both reference
    // and experimental — cost-model knobs the picker can dial.
    /// DCT8x8 coefficient-domain multiplier `(mul1, mul2, base)`.
    /// Note: stored values include libjxl's 0.75 factor on `mul1`/`mul2`
    /// (applied at `enc_ac_strategy.cc:790` for 8×8-class transforms).
    pub k8x8: (f32, f32, f32),
    /// DCT16x8 / DCT8x16 coefficient-domain multiplier `(mul1, mul2, base)`.
    /// Larger transforms skip the 0.75 factor and use the libjxl raw values.
    pub k16x8: (f32, f32, f32),
    /// DCT16x16 coefficient-domain multiplier `(mul1, mul2, base)`.
    pub k16x16: (f32, f32, f32),
    /// DCT4x8 / DCT8x4 coefficient-domain multiplier `(mul1, mul2, base)`.
    /// 4×N strategies share the 0.75 factor with 8×8.
    pub k4x8: (f32, f32, f32),
    /// DCT4x4 coefficient-domain multiplier `(mul1, mul2, base)`.
    /// 4×4 strategies share the 0.75 factor with 8×8.
    pub k4x4: (f32, f32, f32),

    // ─── Entropy multiplier table ──────────────────────────────────────────
    /// Per-strategy entropy multipliers for AC strategy cost model.
    /// Controls relative preference for each transform type.
    pub entropy_mul_table: EntropyMulTable,

    // ─── Patch encoding ────────────────────────────────────────────────────
    /// Use tree learning for patch reference frame encoding.
    /// When true AND ref frame is large enough (>= 128×128), enables adaptive
    /// prediction in the modular encoder for patch ref frames.
    /// Reference: false (libjxl uses simple Gradient predictor).
    /// Experimental: true at effort >= 7 (PR #4533 style improvement).
    pub patch_ref_tree_learning: bool,

    // ─── RCT selection ───────────────────────────────────────────────────
    /// Number of Reversible Color Transform variants to evaluate before
    /// committing to one (0 = skip search, use YCoCg unconditionally).
    ///
    /// Pipeline stage: modular pre-transform, before predictor + tree
    /// learning (`modular/encode.rs::select_best_rct`,
    /// `modular/frame.rs::select_best_rct_at`). Each candidate runs a
    /// cost estimate; the cheapest wins.
    /// Effort interaction: 0 at e<5, 4 at e5, 5 at e6, 7 at e7, 9 at e8,
    /// 19 at e9+ (libjxl `kSquirrel`/`kKitten`/`kTortoise` schedule).
    /// Override when a specific content class (e.g., film stills) has a
    /// known-best RCT and the search is wasted compute, or when sweeping
    /// to discover content-specific defaults.
    pub nb_rcts_to_try: u8,

    /// Caller-supplied RCT colorspace override. When `Some(rct)`,
    /// `select_best_rct(_at)` skips the search and applies the given
    /// RCT directly. Mirrors libjxl's `cparams.colorspace`. Default
    /// `None` (use the per-effort `nb_rcts_to_try` search).
    pub forced_rct: Option<crate::modular::rct::RctType>,

    // ─── WP parameter search ───────────────────────────────────────────────
    /// Number of weighted-predictor parameter sets to try when tuning the
    /// modular WP per channel (0 = use the libjxl default parameters
    /// without searching).
    ///
    /// Pipeline stage: modular predictor selection
    /// (`modular/predictor.rs::find_best_wp_params`, called from
    /// `modular/section.rs`, `modular/frame.rs`, `modular/encode.rs`).
    /// Effort interaction: 0 at e<8, 2 at e8, 5 at e9+. The search is
    /// expensive (each candidate runs a cost estimate over all WP-eligible
    /// channels), which is why libjxl gates it behind `kKitten`/`kTortoise`.
    /// Override to force the search on at lower effort (e.g., when a picker
    /// wants e6-quality bytes with WP-fitted parameters), or off at e9 for
    /// faster sweeps.
    pub wp_num_param_sets: u8,

    // ─── Tree learning parameters ────────────────────────────────────────
    // Read by `modular/tree_learn.rs::TreeLearningParams::from_profile`.
    // These describe the *shape* of the MA tree — wider trees split on
    // more properties / finer buckets, deeper trees use lower thresholds,
    // and the sampling caps trade tree-learning compute for accuracy.
    /// Number of MA-tree decision properties to evaluate per split.
    /// Capped to the order length defined in `modular/tree_learn.rs`
    /// (15 without `group_id`, 16 with).
    /// Effort interaction: 3 at e<=4, 4 at e5, 5 at e6, 7 at e7, 10 at e8,
    /// 16 at e9+. More properties = better trees but quadratic cost in
    /// `LearnTree`. Override to retune the speed/quality knee per content.
    pub tree_num_properties: u8,
    /// Maximum number of quantization buckets per property when building
    /// the histogram for tree splits. Matches libjxl
    /// `enc_modular.cc:556-590` `max_property_values` per speed tier.
    /// Effort interaction: 32 at e<=4, 48 at e5, 64 at e6, 96 at e7,
    /// 128 at e8, 256 at e9+. Higher = finer thresholds at higher learning
    /// cost. Override when a corpus benefits from coarser/finer splits
    /// than the libjxl tier table predicts.
    pub tree_max_buckets: u16,
    /// Base entropy-cost threshold a candidate split must beat to be
    /// accepted (libjxl `75 + 14 * speed_tier` in
    /// `enc_modular.cc::LearnTreeHeuristics`).
    /// Effort interaction: 173 at e<=1 (speed_tier=9), 117 at e5 (5),
    /// 75 at e9+ (1). Lower threshold = more splits = larger tree. Override
    /// to bias the tree shallower (cheaper decode) or deeper (better fit).
    pub tree_threshold_base: f32,
    /// Hard cap on samples drawn for tree learning when set; `0` defers
    /// to [`Self::tree_sample_fraction`].
    /// Read by `modular/tree_learn.rs::sample_count_for_profile`.
    /// Effort interaction: 65,000 at e<=4 (cheap, fixed budget), 0 at e>=5
    /// (let the fraction-based path scale with image size). Override to
    /// fix the tree-learning compute regardless of input pixels.
    pub tree_max_samples_fixed: u32,
    /// Fraction of total pixels to sample for tree learning when
    /// [`Self::tree_max_samples_fixed`] is `0`. Floor of 65,536 samples.
    /// Read by `modular/tree_learn.rs::sample_count_for_profile`.
    /// Effort interaction: 0.15 at e<=4, 0.25 at e5, 0.35 at e6, 0.5 at e7,
    /// 0.55 at e8, 0.65 at e9+ (libjxl PR #4236). Higher fractions improve
    /// tree fit (especially on large images) at proportional cost. Override
    /// to densify sampling on large images at moderate effort, or thin
    /// sampling for fast sweeps at high effort.
    pub tree_sample_fraction: f32,
    /// Use the streaming two-hash cuckoo dedup (libjxl `AddSample` parity,
    /// `enc_ma.cc:602-655`) instead of the default packed-key sort during
    /// tree-sample deduplication.
    ///
    /// Default `false` at every effort. The streaming path **regresses**
    /// end-to-end wall-clock by +3 % to +8 % at e7 on CLIC photos because
    /// `pack_sample_key` random-accesses parallel SoA arrays per sample
    /// (no cache locality) and the sort path exploits spatial coherence
    /// the hash path cannot. Retained as an opt-in for experimentation
    /// toward issue #41 Phase 2 — integrating dedup into the gather pass
    /// itself, where libjxl gets its actual win because keys land once
    /// during ingest.
    pub use_streaming_dedup: bool,
    /// Integrate the two-hash cuckoo dedup into the gather loop itself
    /// (libjxl `AddSample` parity, `enc_ma.cc:711`). This is Phase 2 of
    /// issue #41 — see
    /// [`crate::modular::tree_learn::TreeLearningParams::gather_dedup`].
    ///
    /// Default `false` at every effort: output is **not** byte-identical
    /// to the sort path because gather-time dedup hashes on raw i32
    /// property values rather than post-quantization bucket indices, so
    /// the unique set is a strict superset (the post-`pre_quantize` sort
    /// pass would have collapsed bucket-equivalent rows that gather-time
    /// kept separate). Callers opt in via the `__expert` lossless
    /// override; sweep harnesses re-bake hash-locks when they do.
    pub gather_dedup: bool,
}

impl EffortProfile {
    /// Create an effort profile for lossy (VarDCT) encoding.
    pub fn lossy(effort: u8, mode: EncoderMode) -> Self {
        let effort = effort.clamp(1, 10);
        match mode {
            EncoderMode::Reference => Self::lossy_reference(effort),
            EncoderMode::Experimental => Self::lossy_experimental(effort),
        }
    }

    /// Create an effort profile for lossless (modular) encoding.
    pub fn lossless(effort: u8, mode: EncoderMode) -> Self {
        let effort = effort.clamp(1, 10);
        match mode {
            EncoderMode::Reference => Self::lossless_reference(effort),
            EncoderMode::Experimental => Self::lossless_experimental(effort),
        }
    }

    fn lossy_reference(effort: u8) -> Self {
        let speed_tier = 10u8.saturating_sub(effort);

        Self {
            effort,

            // ── Feature flags ──
            use_ans: effort >= 3,
            optimize_codes: effort >= 3,
            custom_orders: effort >= 4,
            gaborish: effort >= 5,
            pixel_domain_loss: effort >= 5,
            error_diffusion: false, // libjxl accepts param but never uses it
            patches: effort >= 7,
            tree_learning: effort >= 7,
            // libjxl does NOT use LZ77 for VarDCT DC or AC at effort < 9.
            // DC: ForModular() → lz77_method = kNone (modular_mode=false).
            // AC: HistogramParams(kSquirrel, num_ctx) → lz77_method = kNone
            //     (enc_frame.cc overrides since tier > kTortoise).
            // Only kTortoise (effort 9+) enables LZ77 for VarDCT streams.
            lz77: effort >= 9,
            // **Lz77Method::Optimal at e9+ is deliberate** (issue #29).
            // libjxl uses Lz77Method::Rle for ALL VarDCT encodes regardless
            // of tier; we use Optimal because v07 RD analysis shows ~5×
            // size regression on synthetic gradients with RLE
            // (498B → 2,417B on 1024×1024 gradients), bit-identical
            // quality, while photographic content (~98% of inputs) is
            // byte-identical RLE-vs-Optimal.
            //
            // Caveat: Optimal trips a latent bug in jxl-rs's VarDCT AC
            // decoder path (libjxl/jxl-rs#765, our tracker #29). Affected
            // pipelines: anything that round-trips through zenjxl-decoder
            // (which forks jxl-rs unchanged). djxl + jxl-oxide decode
            // these bitstreams cleanly. DO NOT flip the default to RLE
            // to "fix" the decoder — that'd silently degrade gradient
            // encodes 5×. Wait for the upstream jxl-rs fix.
            lz77_method: match effort {
                0..=8 => Lz77Method::Rle,
                _ => Lz77Method::Optimal,
            },
            butteraugli_iters: match effort {
                // libjxl runs FindBestQuantization unconditionally for lossy
                // encoding. Gated at speed_tier <= kKitten (effort >= 8) in libjxl
                // (enc_adaptive_quantization.cc:1282). kDefaultButteraugliIters=2,
                // kMaxButteraugliIters=4 for kTortoise (effort 9+).
                0..=7 => 0,
                8 => 2,
                _ => 4,
            },

            // ── AC strategy search ──
            ac_strategy_enabled: effort >= 5,
            try_dct16: effort >= 5,
            try_dct32: effort >= 5,
            try_dct64: effort >= 7,
            try_dct4x8_afv: effort >= 6,
            non_aligned_eval: effort >= 6,
            fine_grained_step: if effort >= 9 { 1 } else { 2 },

            // ── VarDCT pipeline ──
            chromacity_adjustment: effort >= 7,
            enhanced_clustering_vardct: effort >= 9,
            optimize_uint_configs_vardct: effort >= 9,
            epf_dynamic_sharpness: effort >= 6,
            cfl_two_pass: effort >= 7,
            cfl_newton: effort >= 7,
            cfl_newton_eps: jxl_simd::NEWTON_EPS_DEFAULT,
            cfl_newton_max_iters: jxl_simd::NEWTON_MAX_ITERS_DEFAULT,

            // ── Quantization ──
            use_adaptive_quant: effort >= 5,
            adjust_quant_ac: effort >= 5,
            initial_q_numerator: if effort >= 5 { 0.39 } else { 0.79 },
            fixed_thresholds_y: [0.56, 0.62, 0.62, 0.62],
            adjust_thresholds: [0.58, 0.64, 0.64, 0.64],

            // ── Cost model constants (from libjxl) ──
            k_favor_2x2: -0.4,
            k_avoid_transforms_base: 0.5,
            k_info_loss_mul_base: 1.2,
            k_zeros_mul_base: 9.308_906,
            k_cost_delta_base: 10.833_273,
            k_ac_quant: 0.765,

            // ── Coefficient-domain multipliers ──
            // Note: k8x8 mul1 has 0.75 factor applied (libjxl enc_ac_strategy.cc:790)
            k8x8: (-0.55 * 0.75, 1.073_575_8 * 0.75, 1.4),
            k16x8: (-0.55, 0.901_958_8, 1.6),
            k16x16: (-0.65, 0.88, 1.8),
            k4x8: (-0.50 * 0.75, 0.88, 1.3),
            k4x4: (-0.45 * 0.75, 0.85, 1.2),

            // ── Entropy multiplier table ──
            entropy_mul_table: EntropyMulTable::reference(),

            // ── Patch encoding ──
            patch_ref_tree_learning: false,

            // ── RCT selection ──
            nb_rcts_to_try: match effort {
                0..=4 => 0,
                5 => 4,
                6 => 5,
                7 => 7,
                8 => 9,
                _ => 19,
            },
            forced_rct: None,

            // ── WP parameter search ──
            wp_num_param_sets: match effort {
                0..=7 => 0,
                8 => 2,
                _ => 5,
            },

            // ── Tree learning ──
            tree_num_properties: Self::tree_num_properties_for(effort),
            tree_max_buckets: Self::tree_max_buckets_for(effort),
            tree_threshold_base: 75.0 + 14.0 * speed_tier as f32,
            tree_max_samples_fixed: if effort <= 4 { 65_000 } else { 0 },
            // Effort-scaled nb_repeats matching libjxl PR #4236
            tree_sample_fraction: Self::tree_sample_fraction_for(effort),
            // Default OFF: streaming dedup regresses end-to-end wall-clock
            // on real photos (issue #41) in our post-gather pipeline.
            use_streaming_dedup: false,
            // Default OFF: gather-time dedup ships bytes that don't match
            // the sort-path hash-locks (raw vs bucket-quantized property
            // hashing — see TreeLearningParams::gather_dedup). Opt-in
            // via the __expert lossless override when sweep harnesses
            // are ready to re-bake hash_lock sidecars.
            gather_dedup: false,
        }
    }

    fn lossless_reference(effort: u8) -> Self {
        let speed_tier = 10u8.saturating_sub(effort);

        Self {
            effort,

            // ── Feature flags ──
            use_ans: effort >= 3,
            optimize_codes: effort >= 2,
            custom_orders: effort >= 3,
            gaborish: false,          // N/A for lossless
            pixel_domain_loss: false, // N/A for lossless
            error_diffusion: false,   // N/A for lossless
            patches: effort >= 5,
            tree_learning: effort >= 7,
            lz77: effort >= 7,
            lz77_method: match effort {
                0..=7 => Lz77Method::Rle,
                8 => Lz77Method::Greedy,
                _ => Lz77Method::Optimal,
            },
            butteraugli_iters: 0, // N/A for lossless

            // ── AC strategy (N/A for lossless) ──
            ac_strategy_enabled: false,
            try_dct16: false,
            try_dct32: false,
            try_dct64: false,
            try_dct4x8_afv: false,
            non_aligned_eval: false,
            fine_grained_step: 2,

            // ── VarDCT pipeline (N/A for lossless) ──
            chromacity_adjustment: false,
            enhanced_clustering_vardct: false,
            optimize_uint_configs_vardct: false, // N/A for lossless
            epf_dynamic_sharpness: false,
            cfl_two_pass: false,
            cfl_newton: false,
            cfl_newton_eps: jxl_simd::NEWTON_EPS_DEFAULT,
            cfl_newton_max_iters: jxl_simd::NEWTON_MAX_ITERS_DEFAULT,

            // ── Quantization (N/A for lossless) ──
            use_adaptive_quant: false,
            adjust_quant_ac: false,
            initial_q_numerator: 0.39,
            fixed_thresholds_y: [0.56, 0.62, 0.62, 0.62],
            adjust_thresholds: [0.58, 0.64, 0.64, 0.64],

            // ── Cost model constants (used for tree learning cost estimates) ──
            k_favor_2x2: -0.4,
            k_avoid_transforms_base: 0.5,
            k_info_loss_mul_base: 1.2,
            k_zeros_mul_base: 9.308_906,
            k_cost_delta_base: 10.833_273,
            k_ac_quant: 0.765,

            // ── Coefficient-domain multipliers (N/A for lossless) ──
            k8x8: (-0.55 * 0.75, 1.073_575_8 * 0.75, 1.4),
            k16x8: (-0.55, 0.901_958_8, 1.6),
            k16x16: (-0.65, 0.88, 1.8),
            k4x8: (-0.50 * 0.75, 0.88, 1.3),
            k4x4: (-0.45 * 0.75, 0.85, 1.2),

            // ── Entropy multiplier table (N/A for lossless, but struct requires it) ──
            entropy_mul_table: EntropyMulTable::reference(),

            // ── Patch encoding ──
            patch_ref_tree_learning: false,

            // ── RCT selection ──
            nb_rcts_to_try: match effort {
                0..=4 => 0,
                5 => 4,
                6 => 5,
                7 => 7,
                8 => 9,
                _ => 19,
            },
            forced_rct: None,

            // ── WP parameter search ──
            wp_num_param_sets: match effort {
                0..=7 => 0,
                8 => 2,
                _ => 5,
            },

            // ── Tree learning ──
            tree_num_properties: Self::tree_num_properties_for(effort),
            tree_max_buckets: Self::tree_max_buckets_for(effort),
            tree_threshold_base: 75.0 + 14.0 * speed_tier as f32,
            tree_max_samples_fixed: if effort <= 4 { 65_000 } else { 0 },
            // Effort-scaled nb_repeats matching libjxl PR #4236
            tree_sample_fraction: Self::tree_sample_fraction_for(effort),
            // Default OFF: streaming dedup regresses end-to-end wall-clock
            // on real photos (issue #41) in our post-gather pipeline.
            use_streaming_dedup: false,
            // Default OFF: gather-time dedup ships bytes that don't match
            // the sort-path hash-locks (raw vs bucket-quantized property
            // hashing — see TreeLearningParams::gather_dedup). Opt-in
            // via the __expert lossless override when sweep harnesses
            // are ready to re-bake hash_lock sidecars.
            gather_dedup: false,
        }
    }

    /// Experimental lossy profile with tuning from libjxl PRs and our own improvements.
    ///
    /// Divergences from reference (documented per-field):
    /// - `k_info_loss_mul_base`: 1.2 → 1.3 (PR #4506, +8% pixel-domain loss weight)
    /// - `entropy_mul_table`: PR #4506 values (favor DCT4x4, Identity, AFV)
    /// - `enhanced_clustering_vardct`: enabled at effort >= 7 (was e9+)
    /// - `patch_ref_tree_learning`: true at effort >= 7 (tree learning for patch ref frames)
    fn lossy_experimental(effort: u8) -> Self {
        let mut p = Self::lossy_reference(effort);

        // PR #4506 (Jon Sneyers): +8% weight on pixel-domain loss improves visual quality
        // on detailed content. The info_loss_mul scales the IDCT-domain error term in
        // EstimateEntropy, making the cost model more sensitive to visible artifacts.
        // Reference: 1.2 (libjxl enc_ac_strategy.cc). Experimental: 1.3.
        p.k_info_loss_mul_base = 1.3;

        // PR #4506 entropy multiplier rebalancing: favor small/detail-preserving transforms.
        p.entropy_mul_table = EntropyMulTable::experimental();

        // Pair-merge histogram clustering helps VarDCT at effort 7+ (not just e9+).
        // The ANS header cost savings from merging similar distributions outweigh the
        // slight data cost increase from sharing code tables across contexts.
        if effort >= 7 {
            p.enhanced_clustering_vardct = true;
        }

        // Tree learning for patch reference frames: adapts prediction to packed glyphs
        // instead of using fixed Gradient predictor. Significant on large ref frames
        // (screenshots with many unique patterns). Gated at effort >= 7.
        if effort >= 7 {
            p.patch_ref_tree_learning = true;
        }

        p
    }

    fn lossless_experimental(effort: u8) -> Self {
        Self::lossless_reference(effort)
    }

    fn tree_num_properties_for(effort: u8) -> u8 {
        match effort {
            0..=4 => 3,
            5 => 4,
            6 => 5,
            7 => 7,
            8 => 10,
            // 16 = all properties including group_id.
            // Non-squeeze array has 15 elements, so .min(15) caps correctly.
            // Squeeze array has 16 elements (group_id always included).
            _ => 16,
        }
    }

    /// Effort-scaled pixel sampling fraction for tree learning (libjxl PR #4236).
    fn tree_sample_fraction_for(effort: u8) -> f32 {
        match effort {
            0..=4 => 0.15,
            5 => 0.25,
            6 => 0.35,
            7 => 0.5,
            8 => 0.55,
            _ => 0.65,
        }
    }

    fn tree_max_buckets_for(effort: u8) -> u16 {
        // Matches libjxl enc_modular.cc:556-590 max_property_values by speed_tier.
        match effort {
            0..=4 => 32, // <=Cheetah
            5 => 48,     // Hare
            6 => 64,     // Wombat
            7 => 96,     // Squirrel
            8 => 128,    // Kitten
            _ => 256,    // Tortoise
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Public expert surface — segmented Lossy / Lossless internal-param structs
// ─────────────────────────────────────────────────────────────────────────
//
// `LossyInternalParams` and `LosslessInternalParams` are the public picker /
// sweep escape hatch (gated behind `__expert`). They split the internal
// [`EffortProfile`] into two type-disjoint surfaces — one per encode mode —
// so callers cannot accidentally hand the lossy encoder a knob that only
// affects modular output, and vice-versa. The type system enforces
// mode-correctness instead of relying on documentation.
//
// Each `Some(_)` field overrides the corresponding `EffortProfile` field
// the lossy / lossless code path actually reads. Fields left at `None` keep
// the (effort, mode)-derived default. This matches the segmented
// `InternalParams` pattern used by zenavif / zenwebp / zenravif.

/// Picker / sweep override knobs for the **lossy (VarDCT)** encode path.
///
/// Apply via [`crate::api::LossyConfig::with_internal_params`]. Fields are
/// optional: `Some(value)` overrides the corresponding effort-derived
/// default; `None` keeps the default. `#[non_exhaustive]` so additional
/// knobs can land additively without a breaking change.
///
/// The fields here are the lossy-side knobs that flow through `profile.X`
/// at lossy encode time (verified against `vardct/encoder.rs`,
/// `vardct/ac_strategy_search.rs`, `vardct/transform.rs`,
/// `vardct/precomputed.rs`, and `vardct/bitstream.rs`). Modular-only knobs
/// (RCT search, WP parameter scan, tree-learning shape) live on
/// [`LosslessInternalParams`] — VarDCT's DC frame uses a fixed Gradient
/// predictor, so those knobs do not affect lossy bytes.
#[cfg(feature = "__expert")]
#[non_exhaustive]
#[derive(Default, Clone, Debug)]
pub struct LossyInternalParams {
    /// Try DCT16x16 / DCT16x8 / DCT8x16 transforms in AC strategy search.
    /// Default at effort 7: `true`. Disabling forces no 16×16-class merges.
    pub try_dct16: Option<bool>,

    /// Try DCT32x32 / DCT32x16 / DCT16x32 transforms.
    /// Default at effort 7: `true`. Disabling forces no 32×32-class merges.
    pub try_dct32: Option<bool>,

    /// Try DCT64x64 / DCT64x32 / DCT32x64 transforms.
    /// Default at effort 7: `true`. Disabling forces no 64×64-class merges.
    pub try_dct64: Option<bool>,

    /// Try DCT4x8 / DCT8x4 / DCT4x4 / AFV transforms.
    /// Default at effort 6+: `true`. Disabling forces 8×8-or-larger only.
    pub try_dct4x8_afv: Option<bool>,

    /// Step size for fine-grained AC strategy search on 32×32+ blocks.
    /// `1` evaluates every position (effort 9+), `2` every other (default).
    pub fine_grained_step: Option<u8>,

    /// Base multiplier on the IDCT-domain (pixel-domain) error term in
    /// `EstimateEntropy`. Reference: 1.2 (libjxl). Experimental: 1.3
    /// (PR #4506). Higher values weight visible artifacts more heavily
    /// vs coefficient-domain entropy.
    pub k_info_loss_mul_base: Option<f32>,

    /// Per-strategy entropy multipliers for AC strategy cost model.
    /// Controls relative preference for each transform type.
    pub entropy_mul_table: Option<EntropyMulTable>,

    /// Recompute CfL map after initial quantization for better estimates.
    /// Default at effort 7+: `true`.
    pub cfl_two_pass: Option<bool>,

    /// Apply pixel-level chromacity adjustments. Default at effort 7+:
    /// `true`. Disabling skips per-pixel chromacity nudges.
    pub chromacity_adjustment: Option<bool>,

    /// Use tree learning for patch reference frame encoding instead of the
    /// fixed Gradient predictor. Reference: `false`. Experimental at
    /// effort 7+: `true`. Significant on screenshots / packed glyph patches.
    pub patch_ref_tree_learning: Option<bool>,

    /// Enable non-aligned evaluation pass (odd-aligned 16×16 regions) in
    /// AC strategy search. Default at effort 6+: `true`. Disabling halves
    /// the search depth.
    pub non_aligned_eval: Option<bool>,

    /// Use pair-merge clustering for VarDCT entropy codes. Reference at
    /// effort 9+: `true`; experimental at effort 7+: `true`. When `false`,
    /// uses fast k-means-only clustering (cheaper, slightly larger codes).
    pub enhanced_clustering_vardct: Option<bool>,

    /// Quantization-cost constant used when materializing the initial
    /// quant field (libjxl 0.765, `enc_adaptive_quantization.cc`). Lower
    /// values produce a coarser initial field (less rate, more distortion);
    /// higher values refine.
    pub k_ac_quant: Option<f32>,
}

/// Picker / sweep override knobs for the **lossless (modular)** encode path.
///
/// Apply via [`crate::api::LosslessConfig::with_internal_params`]. Fields
/// are optional: `Some(value)` overrides the corresponding effort-derived
/// default; `None` keeps the default. `#[non_exhaustive]` so additional
/// knobs can land additively without a breaking change.
///
/// The fields here are the modular-path knobs that flow through `profile.X`
/// in `modular/encode.rs`, `modular/frame.rs`, `modular/section.rs`,
/// `modular/predictor.rs`, and `modular/tree_learn.rs`. AC-strategy and
/// CfL knobs live on [`LossyInternalParams`].
#[cfg(feature = "__expert")]
#[non_exhaustive]
#[derive(Default, Clone, Debug)]
pub struct LosslessInternalParams {
    /// Number of Reversible Color Transform variants to evaluate before
    /// committing (0 = skip search, use YCoCg unconditionally).
    /// Effort interaction: 0 at e<5, 4 at e5, 5 at e6, 7 at e7, 9 at e8,
    /// 19 at e9+ (libjxl `kSquirrel`/`kKitten`/`kTortoise` schedule).
    pub nb_rcts_to_try: Option<u8>,

    /// Force a specific RCT colorspace; when `Some(rct)`,
    /// `select_best_rct(_at)` skips the search entirely.
    /// Mirrors libjxl's `cparams.colorspace`. `None` keeps the
    /// per-effort search behaviour.
    pub forced_rct: Option<crate::modular::rct::RctType>,

    /// Number of weighted-predictor parameter sets to try per WP-eligible
    /// channel (0 = use libjxl defaults without searching).
    /// Effort interaction: 0 at e<8, 2 at e8, 5 at e9+.
    pub wp_num_param_sets: Option<u8>,

    /// Maximum quantization buckets per property when building the
    /// histogram for tree splits.
    /// Effort interaction: 32 at e<=4, 48 at e5, 64 at e6, 96 at e7,
    /// 128 at e8, 256 at e9+. Higher = finer thresholds at higher cost.
    pub tree_max_buckets: Option<u16>,

    /// Number of MA-tree decision properties to evaluate per split.
    /// Effort interaction: 3 at e<=4, 4 at e5, 5 at e6, 7 at e7, 10 at e8,
    /// 16 at e9+.
    pub tree_num_properties: Option<u8>,

    /// Base entropy-cost threshold a candidate split must beat to be
    /// accepted (libjxl `75 + 14 * speed_tier`). Lower = more splits =
    /// larger tree.
    pub tree_threshold_base: Option<f32>,

    /// Fraction of total pixels to sample for tree learning (when
    /// `tree_max_samples_fixed` is `0`). Floor of 65,536 samples.
    /// Effort interaction: 0.15 at e<=4 ramping to 0.65 at e9+
    /// (libjxl PR #4236).
    pub tree_sample_fraction: Option<f32>,

    /// Hard cap on samples drawn for tree learning when set; `0` defers
    /// to [`Self::tree_sample_fraction`].
    /// Effort interaction: 65,000 at e<=4, 0 at e>=5.
    pub tree_max_samples_fixed: Option<u32>,

    /// Switch the tree-sample dedup backend.
    ///
    /// `Some(true)` enables the streaming two-hash cuckoo path
    /// (`dedup_samples_streaming`, libjxl `AddSample` parity). `Some(false)`
    /// keeps the default packed-key sort path
    /// (`dedup_samples_packed_sort`). `None` leaves the effort profile
    /// default (always `false` today; see [`EffortProfile::use_streaming_dedup`]).
    ///
    /// The streaming path **regresses** wall-clock by +3 % to +8 % at e7
    /// on real CLIC photos (issue #41 measurement, 2026-05-16). Retained
    /// for experimentation toward issue #41 Phase 2 (gather-integrated
    /// dedup); not recommended for production sweeps.
    pub use_streaming_dedup: Option<bool>,

    /// Enable libjxl-parity gather-time dedup (Phase 2 of issue #41).
    ///
    /// `Some(true)` runs each gathered sample through a two-hash cuckoo
    /// table inside `gather_channel_samples`, merging duplicates *during*
    /// the gather pass. The post-gather `dedup_samples_packed_sort` then
    /// operates on a much smaller surviving set. `Some(false)` keeps the
    /// existing post-pass dedup-only flow. `None` leaves the
    /// effort-profile default (always `false` today; see
    /// [`EffortProfile::gather_dedup`]).
    ///
    /// **Bytes are not byte-identical to the sort-only path.** Gather-time
    /// dedup hashes on raw i32 property values (pre-quantization runs
    /// later), so the surviving unique set is a strict superset of the
    /// bucket-equivalence set the sort path collapses to. Hash-locks must
    /// be re-baked when sweep harnesses enable this.
    pub gather_dedup: Option<bool>,
}

#[cfg(feature = "__expert")]
impl LossyInternalParams {
    /// Apply each `Some(_)` field on top of `profile`.
    pub(crate) fn apply_to(self, profile: &mut EffortProfile) {
        let LossyInternalParams {
            try_dct16,
            try_dct32,
            try_dct64,
            try_dct4x8_afv,
            fine_grained_step,
            k_info_loss_mul_base,
            entropy_mul_table,
            cfl_two_pass,
            chromacity_adjustment,
            patch_ref_tree_learning,
            non_aligned_eval,
            enhanced_clustering_vardct,
            k_ac_quant,
        } = self;
        if let Some(v) = try_dct16 {
            profile.try_dct16 = v;
        }
        if let Some(v) = try_dct32 {
            profile.try_dct32 = v;
        }
        if let Some(v) = try_dct64 {
            profile.try_dct64 = v;
        }
        if let Some(v) = try_dct4x8_afv {
            profile.try_dct4x8_afv = v;
        }
        if let Some(v) = fine_grained_step {
            profile.fine_grained_step = v;
        }
        if let Some(v) = k_info_loss_mul_base {
            profile.k_info_loss_mul_base = v;
        }
        if let Some(v) = entropy_mul_table {
            profile.entropy_mul_table = v;
        }
        if let Some(v) = cfl_two_pass {
            profile.cfl_two_pass = v;
        }
        if let Some(v) = chromacity_adjustment {
            profile.chromacity_adjustment = v;
        }
        if let Some(v) = patch_ref_tree_learning {
            profile.patch_ref_tree_learning = v;
        }
        if let Some(v) = non_aligned_eval {
            profile.non_aligned_eval = v;
        }
        if let Some(v) = enhanced_clustering_vardct {
            profile.enhanced_clustering_vardct = v;
        }
        if let Some(v) = k_ac_quant {
            profile.k_ac_quant = v;
        }
    }
}

#[cfg(feature = "__expert")]
impl LosslessInternalParams {
    /// Apply each `Some(_)` field on top of `profile`.
    pub(crate) fn apply_to(self, profile: &mut EffortProfile) {
        let LosslessInternalParams {
            nb_rcts_to_try,
            forced_rct,
            wp_num_param_sets,
            tree_max_buckets,
            tree_num_properties,
            tree_threshold_base,
            tree_sample_fraction,
            tree_max_samples_fixed,
            use_streaming_dedup,
            gather_dedup,
        } = self;
        if let Some(v) = nb_rcts_to_try {
            profile.nb_rcts_to_try = v;
        }
        if forced_rct.is_some() {
            profile.forced_rct = forced_rct;
        }
        if let Some(v) = wp_num_param_sets {
            profile.wp_num_param_sets = v;
        }
        if let Some(v) = tree_max_buckets {
            profile.tree_max_buckets = v;
        }
        if let Some(v) = tree_num_properties {
            profile.tree_num_properties = v;
        }
        if let Some(v) = tree_threshold_base {
            profile.tree_threshold_base = v;
        }
        if let Some(v) = tree_sample_fraction {
            profile.tree_sample_fraction = v;
        }
        if let Some(v) = tree_max_samples_fixed {
            profile.tree_max_samples_fixed = v;
        }
        if let Some(v) = use_streaming_dedup {
            profile.use_streaming_dedup = v;
        }
        if let Some(v) = gather_dedup {
            profile.gather_dedup = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lossy_reference_e7() {
        let p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert_eq!(p.effort, 7);
        assert!(p.use_ans);
        assert!(p.optimize_codes);
        assert!(p.custom_orders);
        assert!(p.gaborish);
        assert!(p.pixel_domain_loss);
        assert!(!p.error_diffusion);
        assert!(p.patches);
        assert!(!p.lz77); // libjxl only enables LZ77 for VarDCT at e9+ (kTortoise)
        assert_eq!(p.butteraugli_iters, 0); // libjxl gates at speed_tier <= kKitten (e8+)
        assert!(p.ac_strategy_enabled);
        assert!(p.try_dct32);
        assert!(p.try_dct64);
        assert!(p.try_dct4x8_afv); // e6+
        assert!(p.non_aligned_eval);
        assert_eq!(p.fine_grained_step, 2);
        assert!(p.chromacity_adjustment); // e7+
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert!(!p.optimize_uint_configs_vardct); // e9+ (libjxl kNone at e<9)
        assert!(p.epf_dynamic_sharpness); // e6+
        assert!(p.cfl_two_pass); // e7+
        assert!(p.cfl_newton); // e7+ with pass 2
        assert!(p.use_adaptive_quant);
        assert!(p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.39);
        assert_eq!(p.k_favor_2x2, -0.4);
        assert_eq!(p.k_ac_quant, 0.765);
        assert_eq!(p.nb_rcts_to_try, 7);
        assert_eq!(p.wp_num_param_sets, 0); // e8+
        assert_eq!(p.tree_num_properties, 7);
        assert_eq!(p.tree_max_buckets, 96);
    }

    #[test]
    fn test_lossy_reference_e5() {
        let p = EffortProfile::lossy(5, EncoderMode::Reference);
        assert_eq!(p.effort, 5);
        assert!(p.use_ans);
        assert!(p.gaborish);
        assert!(p.pixel_domain_loss);
        assert!(!p.error_diffusion); // e7+
        assert!(!p.patches); // e7+
        assert!(!p.lz77); // e9+ for VarDCT
        assert!(p.ac_strategy_enabled);
        assert!(p.try_dct32);
        assert!(!p.try_dct64); // e7+
        assert!(!p.try_dct4x8_afv); // e6+
        assert!(!p.non_aligned_eval); // e6+
        assert!(!p.chromacity_adjustment); // e7+
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert!(!p.optimize_uint_configs_vardct); // e9+
        assert!(!p.epf_dynamic_sharpness); // e6+
        assert!(!p.cfl_two_pass); // e7+
        assert!(!p.cfl_newton); // e7+
        assert!(p.use_adaptive_quant);
        assert!(p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.39);
        assert_eq!(p.butteraugli_iters, 0); // libjxl gates at speed_tier <= kKitten (e8+)
        assert_eq!(p.nb_rcts_to_try, 4);
        assert_eq!(p.wp_num_param_sets, 0); // e8+
    }

    #[test]
    fn test_lossy_reference_e9() {
        let p = EffortProfile::lossy(9, EncoderMode::Reference);
        assert!(p.lz77); // VarDCT LZ77 enabled at e9+ (kTortoise)
        assert_eq!(p.lz77_method, Lz77Method::Optimal);
        assert_eq!(p.butteraugli_iters, 4);
        assert_eq!(p.fine_grained_step, 1);
        assert!(p.enhanced_clustering_vardct); // e9+
        assert!(p.optimize_uint_configs_vardct); // e9+
        assert_eq!(p.nb_rcts_to_try, 19);
        assert_eq!(p.wp_num_param_sets, 5); // e9+
        assert_eq!(p.tree_num_properties, 16);
        assert_eq!(p.tree_max_buckets, 256);
    }

    #[test]
    fn test_lossy_reference_e8() {
        let p = EffortProfile::lossy(8, EncoderMode::Reference);
        assert!(!p.lz77); // libjxl only enables LZ77 for VarDCT at e9+
        assert_eq!(p.lz77_method, Lz77Method::Rle);
        assert_eq!(p.butteraugli_iters, 2);
        assert_eq!(p.fine_grained_step, 2);
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert!(!p.optimize_uint_configs_vardct); // e9+
        assert_eq!(p.wp_num_param_sets, 2); // e8
    }

    #[test]
    fn test_lossy_reference_e3() {
        let p = EffortProfile::lossy(3, EncoderMode::Reference);
        assert!(p.use_ans);
        assert!(p.optimize_codes);
        assert!(!p.gaborish);
        assert!(!p.ac_strategy_enabled);
        assert!(!p.use_adaptive_quant);
        assert!(!p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.79);
    }

    #[test]
    fn test_lossless_reference_e7() {
        let p = EffortProfile::lossless(7, EncoderMode::Reference);
        assert!(p.use_ans);
        assert!(p.tree_learning);
        assert!(p.lz77);
        assert_eq!(p.lz77_method, Lz77Method::Rle);
        assert!(p.patches);
        assert!(!p.gaborish); // N/A
        assert!(!p.pixel_domain_loss); // N/A
        assert!(!p.ac_strategy_enabled); // N/A
    }

    #[test]
    fn test_lossless_reference_e4() {
        let p = EffortProfile::lossless(4, EncoderMode::Reference);
        assert!(p.use_ans);
        assert!(!p.tree_learning); // e7+
        assert!(!p.lz77); // e7+
        assert!(!p.patches); // e5+
    }

    #[test]
    fn test_effort_clamp() {
        let p = EffortProfile::lossy(0, EncoderMode::Reference);
        assert_eq!(p.effort, 1);
        let p = EffortProfile::lossy(99, EncoderMode::Reference);
        assert_eq!(p.effort, 10);
    }

    #[test]
    fn test_experimental_diverges_from_reference() {
        // Experimental should share effort/feature-flag structure with reference
        for effort in 1..=10 {
            let r = EffortProfile::lossy(effort, EncoderMode::Reference);
            let e = EffortProfile::lossy(effort, EncoderMode::Experimental);
            assert_eq!(r.effort, e.effort);
            assert_eq!(r.use_ans, e.use_ans);
            assert_eq!(r.k_favor_2x2, e.k_favor_2x2);
            assert_eq!(r.butteraugli_iters, e.butteraugli_iters);
            assert_eq!(r.nb_rcts_to_try, e.nb_rcts_to_try);
        }

        // Verify specific divergences at effort 7
        let r = EffortProfile::lossy(7, EncoderMode::Reference);
        let e = EffortProfile::lossy(7, EncoderMode::Experimental);

        // k_info_loss_mul_base: 1.2 → 1.3 (PR #4506)
        assert_eq!(r.k_info_loss_mul_base, 1.2);
        assert_eq!(e.k_info_loss_mul_base, 1.3);

        // entropy_mul_table: PR #4506 rebalancing
        assert_eq!(r.entropy_mul_table.dct4x4, 1.08);
        assert_eq!(e.entropy_mul_table.dct4x4, 0.88);
        assert_eq!(r.entropy_mul_table.identity, 1.0428);
        assert_eq!(e.entropy_mul_table.identity, 0.88);
        assert_eq!(r.entropy_mul_table.afv, 0.817_794_9);
        assert_eq!(e.entropy_mul_table.afv, 0.75);
        // Unchanged values should match
        assert_eq!(r.entropy_mul_table.dct8, e.entropy_mul_table.dct8);
        assert_eq!(r.entropy_mul_table.dct16x8, e.entropy_mul_table.dct16x8);
        assert_eq!(r.entropy_mul_table.dct32x32, e.entropy_mul_table.dct32x32);

        // enhanced_clustering_vardct: e9+ → e7+ in experimental
        assert!(!r.enhanced_clustering_vardct); // reference e7: off
        assert!(e.enhanced_clustering_vardct); // experimental e7: on

        // patch_ref_tree_learning: false → true at e7+
        assert!(!r.patch_ref_tree_learning);
        assert!(e.patch_ref_tree_learning);

        // At effort 5, experimental should NOT enable the e7+ features
        let e5 = EffortProfile::lossy(5, EncoderMode::Experimental);
        assert!(!e5.enhanced_clustering_vardct);
        assert!(!e5.patch_ref_tree_learning);
        // But should still have the entropy_mul and info_loss_mul changes
        assert_eq!(e5.k_info_loss_mul_base, 1.3);
        assert_eq!(e5.entropy_mul_table.dct4x4, 0.88);
    }

    #[test]
    fn test_entropy_mul_table_reference_values() {
        // Verify all reference values match libjxl enc_ac_strategy.cc:584
        let t = EntropyMulTable::reference();
        assert_eq!(t.dct8, 0.8);
        assert_eq!(t.dct4x4, 1.08);
        assert_eq!(t.dct4x8, 0.859_316_37);
        assert_eq!(t.identity, 1.0428);
        assert_eq!(t.dct2x2, 0.95);
        assert_eq!(t.afv, 0.817_794_9);
        assert_eq!(t.dct16x8, 1.21);
        assert_eq!(t.dct16x16, 1.34);
        assert_eq!(t.dct16x32, 1.49);
        assert_eq!(t.dct32x32, 1.48);
        assert_eq!(t.dct64x32, 2.25);
        assert_eq!(t.dct64x64, 2.25);
    }

    #[test]
    fn test_entropy_mul_table_experimental_values() {
        // Verify PR #4506 changes and that unchanged values are preserved
        let t = EntropyMulTable::experimental();
        let r = EntropyMulTable::reference();

        // Changed values (PR #4506)
        assert_eq!(t.dct4x4, 0.88); // was 1.08
        assert_eq!(t.identity, 0.88); // was 1.0428
        assert_eq!(t.afv, 0.75); // was 0.818

        // Unchanged values
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct16x16, r.dct16x16);
        assert_eq!(t.dct16x32, r.dct16x32);
        assert_eq!(t.dct32x32, r.dct32x32);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_lossless_experimental_matches_reference() {
        // Lossless experimental is currently identical to reference
        for effort in 1..=10 {
            let r = EffortProfile::lossless(effort, EncoderMode::Reference);
            let e = EffortProfile::lossless(effort, EncoderMode::Experimental);
            assert_eq!(r.effort, e.effort);
            assert_eq!(r.use_ans, e.use_ans);
            assert_eq!(r.tree_learning, e.tree_learning);
            assert_eq!(r.lz77, e.lz77);
        }
    }

    #[test]
    fn test_tree_threshold_base_formula() {
        // speed_tier = 10 - effort
        // threshold = 75 + 14 * speed_tier
        let p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 3.0); // speed_tier=3
        let p = EffortProfile::lossy(9, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 1.0); // speed_tier=1
        let p = EffortProfile::lossy(5, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 5.0); // speed_tier=5
    }
}
