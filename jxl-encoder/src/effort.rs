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
    /// Phase 3 of issue #41 — switch gather-time dedup to the
    /// inline-fingerprint cuckoo table
    /// (`crate::modular::inline_dedup_table::InlineDedupTable`) instead of
    /// Phase 2's [`Self::gather_dedup`] SoA-indexed table.
    ///
    /// Only meaningful when [`Self::gather_dedup`] is also `true`: the
    /// switch happens inside `gather_samples_strided_with_dedup`, where
    /// Phase 2 builds a `GatherDedupTable` and Phase 3 builds an
    /// `InlineDedupTable` instead. Both produce the same unique-set
    /// semantics (strict superset of the post-sort merge); the
    /// post-`pre_quantize` sort pass collapses the difference downstream.
    /// Hash-locks therefore stay the same as Phase 2's locked variant —
    /// the post-sort arbiter remains the final byte-determinant.
    ///
    /// Default `false`. Callers opt in via the `__expert` lossless
    /// override (`LosslessInternalParams::gather_dedup_phase3`).
    ///
    /// The microbench (`benches/dedup_samples_strategies.rs`,
    /// `benchmarks/inline_dedup_microbench_2026-05-17.txt`) shows
    /// +36 %-53 % gather-throughput vs Phase 1 on high-duplication
    /// streams; real-photo gather payoff depends on spatial duplication
    /// ratios and is decided by Chunk 2's end-to-end A/B at e7 / 1.05 MP.
    pub gather_dedup_phase3: bool,

    // ─── Parallel tree-learning tuning ────────────────────────────────────
    // Read by `modular/tree_learn.rs` (gated on
    // `feature = "parallel-tree-learning"`). These control the rayon
    // fan-out shape in the divide-and-conquer subtree builder. The
    // original constants (depth=4, floor=16384, root_threshold=8192) were
    // tuned on e7 trees (~2,425 nodes on a 1024² photo). At e8/e9 the
    // tree is +30%/+118% larger and the per-fork work is heavier, so
    // deeper fanout + lower floor saturate more cores. See chunk-2 of
    // `lossless_e8_e9_cliff_2026-05-16.md`.
    /// Maximum depth of parallel recursion in the borrowed-view subtree
    /// builder. `2^depth` is the upper bound on parallel leaf tasks.
    /// Effort interaction: 4 at effort ≤ 7 (16 leaf tasks), 5 at effort ≥ 8
    /// (32 leaf tasks). Picker may override; raising costs nothing at
    /// small inputs because the floor terminates fanout early.
    pub tree_parallel_max_depth: u32,
    /// Minimum subtree size below which further parallel fork is skipped
    /// and the iterative sequential builder runs instead. Below this
    /// rayon::join + workspace setup exceeds the parallel savings.
    /// Effort interaction: 16384 at effort ≤ 7, 8192 at effort ≥ 8.
    pub tree_parallel_floor: usize,
    /// Minimum total sample count required before attempting the parallel
    /// root split. Below this the sequential loop is faster.
    /// Effort interaction: 8192 at effort ≤ 7, 4096 at effort ≥ 8.
    pub tree_parallel_root_threshold: usize,
    /// Small-image fallback for the parallel-tree-learning path.
    ///
    /// When `true`, the tree-learner bypasses the thread-local
    /// [`SplitWorkspace`] cache (allocating a fresh workspace per
    /// `find_best_split` call instead of routing through the
    /// `RefCell::borrow_mut` indirection). On small inputs the cache
    /// pays its own per-call cost without meaningful amortisation,
    /// matching the +0.85% small-mean regression documented in the
    /// `cb5e202` commit body.
    ///
    /// The parallel root split + recursive borrowed-view fan-out
    /// remain ENABLED in this fallback regime — they are still the
    /// largest single wall-clock win at 8 threads and stripping them
    /// out costs more (~50-80% slowdown on small images observed
    /// during dev of this fix) than the thread-local cache they sit
    /// inside.
    ///
    /// The companion small-image regression from the borrowed-view
    /// zero-clone fork (`fe2d3a27`, +6.2% small-mean) is NOT addressed
    /// by this flag — fixing it requires resurrecting the deleted
    /// owned-clone path (`split_tree_samples_owned` / `split_pq_owned`
    /// / `build_subtree_recursive_parallel`), which is tracked as a
    /// separate follow-up. See
    /// `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/
    ///  rejected_optimizations_conditional_value_2026-05-17.md` #9.
    ///
    /// Bitstream-equivalent: tree topology depends on the samples,
    /// not the workspace identity, so hash-locks stay byte-identical.
    ///
    /// Default `false`. Set automatically to `true` by
    /// [`Self::adapt_small_image_fallback`] when
    /// `pixels < SMALL_IMAGE_PIXEL_THRESHOLD` (1 MP).
    pub tree_parallel_small_image_fallback: bool,

    /// Number of random-seeded tree-learning runs to perform per encode,
    /// keeping the tree whose tokens have the lowest entropy cost.
    ///
    /// libjxl's `FindBestSplit` is greedy ID3 — locally optimal at each
    /// split but sensitive to which pixels were sampled. Running gather→
    /// tree with multiple stride offsets (or RNG seeds in the future)
    /// and picking the cheapest-encoding tree closes part of that gap.
    ///
    /// Effort interaction (set by `Self::tree_learn_seeds_for`):
    /// - effort ≤ 9: `1` (single run — libjxl-equivalent, byte-identical
    ///   to the pre-RFC#45-chunk-2 baseline)
    /// - effort = 10: `2`
    /// - effort = 11: `4`
    ///
    /// Bitstream-valid: each seed produces a normal, spec-valid tree;
    /// the picker just chooses among them. Bytes change only when
    /// `seeds > 1` (e10/e11) — no hash-lock churn at e ≤ 9.
    ///
    /// `0` is treated as `1` (defensive). Read by
    /// `modular/tree_learn.rs::select_best_tree_multi_seed`.
    pub tree_learn_seeds: u8,

    /// Number of butteraugli quantization-loop seeds to run in parallel,
    /// then pick the smallest-bytes result among those that meet the
    /// target butteraugli (RFC#45 pick #1 chunk 3 — lossy analog of
    /// [`Self::tree_learn_seeds`]).
    ///
    /// libjxl's `FindBestQuantization` uses a single hard-coded
    /// `kInitMul = 0.6` (`enc_adaptive_quantization.cc:1042`) which
    /// pulls the post-iteration-1 quant field back toward the initial
    /// AC heuristic field. That single starting point is locally
    /// optimal but the optimization surface has multiple basins —
    /// different `kInitMul` values converge to different (qf, scale)
    /// pairs with measurably different output bytes at the same
    /// butteraugli target.
    ///
    /// At `seeds > 1` we run the loop N times with the seed values
    /// from [`crate::vardct::butteraugli_loop::init_mul_seeds`] (the
    /// libjxl default `0.6` is always included as the first seed so
    /// the worst case is no-regression). The picker keeps the seed
    /// with the largest mean(quant_field_float) (proxy for smallest
    /// encoded bytes — coarser quant → fewer non-zero coefficients)
    /// whose final butteraugli score does not exceed
    /// `1.05 * target_distance`. If no seed meets that bound, the
    /// seed with the smallest final butteraugli score wins.
    ///
    /// Effort interaction (set by `Self::lossy_search_seeds_for`):
    /// - effort ≤ 9: `1` (libjxl-equivalent, bit-identical to pre-RFC#45-chunk-3)
    /// - effort = 10: `2`
    /// - effort = 11: `4`
    ///
    /// Bitstream-valid: each seed produces a normal, spec-valid encode;
    /// the picker just chooses among them. Bytes change only when
    /// `seeds > 1` (e10/e11) — no hash-lock churn at e ≤ 9.
    ///
    /// `0` is treated as `1` (defensive). Read by
    /// `vardct/butteraugli_loop.rs::butteraugli_refine_quant_field`.
    pub lossy_search_seeds: u8,
}

impl EffortProfile {
    /// Create an effort profile for lossy (VarDCT) encoding.
    ///
    /// Accepts effort in `1..=11`. e10/e11 are our extensions beyond libjxl's
    /// kTortoise=9 ceiling: longer search budgets (more butteraugli iters at
    /// e10/e11, multi-seed tree learning at e10+ in a follow-on chunk). The
    /// bitstream remains 100% spec-valid — only encoder search effort changes.
    pub fn lossy(effort: u8, mode: EncoderMode) -> Self {
        let effort = effort.clamp(1, 11);
        match mode {
            EncoderMode::Reference => Self::lossy_reference(effort),
            EncoderMode::Experimental => Self::lossy_experimental(effort),
        }
    }

    /// Create an effort profile for lossless (modular) encoding.
    ///
    /// Accepts effort in `1..=11`. e10/e11 reserve future multi-seed tree
    /// learning (chunk 2 of RFC#45 pick #1). Today they fall through to the
    /// e9 (kTortoise) lossless code paths.
    pub fn lossless(effort: u8, mode: EncoderMode) -> Self {
        let effort = effort.clamp(1, 11);
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
                //
                // RFC#45 chunk 1: e10/e11 extend the budget past libjxl's cap.
                // The loop already structurally bounds itself at
                // `MAX_QUANT_LOOP_ITERS=16` (see butteraugli_loop.rs:151), so
                // `_ => 16` is the natural saturation point — no infinite-loop
                // risk even if a future effort level requests more.
                0..=7 => 0,
                8 => 2,
                9 => 4,
                10 => 8,
                _ => 16,
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
            // Default OFF: Phase 3 inline-fingerprint dedup is opt-in
            // (the post-sort arbiter keeps hash-locks stable, but the
            // gather-time table layout switch is still a perf-only
            // override decided by Chunk 2's end-to-end A/B).
            gather_dedup_phase3: false,

            // Parallel-tree-learning fanout (only used on the lossless
            // path, but set on the lossy profile too for shape parity).
            tree_parallel_max_depth: Self::tree_parallel_max_depth_for(effort),
            tree_parallel_floor: Self::tree_parallel_floor_for(effort),
            tree_parallel_root_threshold: Self::tree_parallel_root_threshold_for(effort),
            // Default false; adapt_to_image() flips this on for <1 MP inputs.
            tree_parallel_small_image_fallback: false,

            // RFC#45 chunk 2: 1 at e ≤ 9 (libjxl-equivalent, byte-identical),
            // 2 at e10, 4 at e11.
            tree_learn_seeds: Self::tree_learn_seeds_for(effort),

            // RFC#45 chunk 3 (lossy multi-seed butteraugli sweep): 1 at e ≤ 9
            // (libjxl-equivalent, bit-identical), 2 at e10, 4 at e11. The
            // butteraugli loop is no-op below e8 (butteraugli_iters = 0) so
            // this field only takes effect at e10/e11.
            lossy_search_seeds: Self::lossy_search_seeds_for(effort),
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
            // Default OFF: Phase 3 inline-fingerprint dedup is opt-in
            // (the post-sort arbiter keeps hash-locks stable, but the
            // gather-time table layout switch is still a perf-only
            // override decided by Chunk 2's end-to-end A/B).
            gather_dedup_phase3: false,

            // Parallel-tree-learning fanout. e8/e9 trees are larger and
            // the per-leaf work is heavier — deeper fanout + lower floor
            // saturate more cores. See chunk-2 of
            // `lossless_e8_e9_cliff_2026-05-16.md`.
            tree_parallel_max_depth: Self::tree_parallel_max_depth_for(effort),
            tree_parallel_floor: Self::tree_parallel_floor_for(effort),
            tree_parallel_root_threshold: Self::tree_parallel_root_threshold_for(effort),
            // Default false; adapt_to_image() flips this on for <1 MP inputs.
            tree_parallel_small_image_fallback: false,

            // RFC#45 chunk 2: 1 at e ≤ 9 (libjxl-equivalent, byte-identical),
            // 2 at e10, 4 at e11.
            tree_learn_seeds: Self::tree_learn_seeds_for(effort),

            // Lossless never runs the butteraugli loop — keep at 1 so the
            // shared `EffortProfile` struct stays well-formed without
            // implying a phantom lossy sweep on lossless encodes.
            lossy_search_seeds: 1,
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

    /// Parallel-tree-learning fanout depth by effort.
    ///
    /// e8/e9 trees are 30-118% larger than e7 (`lossless_e8_e9_cliff_2026-05-16.md`),
    /// and each per-leaf subtree-build call is heavier. Doubling the leaf-task
    /// budget at high effort lets rayon saturate idle workers; at low effort
    /// the floor terminates fanout early so the extra budget is harmless.
    fn tree_parallel_max_depth_for(effort: u8) -> u32 {
        if effort >= 8 { 5 } else { 4 }
    }

    /// Subtree-size floor below which parallel fork is skipped.
    fn tree_parallel_floor_for(effort: u8) -> usize {
        if effort >= 8 { 8_192 } else { 16_384 }
    }

    /// Total-sample threshold to attempt the parallel root split.
    fn tree_parallel_root_threshold_for(effort: u8) -> usize {
        if effort >= 8 { 4_096 } else { 8_192 }
    }

    /// Number of multi-seed tree-learning runs by effort (RFC#45 pick #1
    /// chunk 2). e ≤ 9 keeps the single-pass libjxl behaviour
    /// (byte-identical hash-locks); e10/e11 fan out 2 / 4 seeded runs and
    /// pick the cheapest-encoding tree.
    fn tree_learn_seeds_for(effort: u8) -> u8 {
        match effort {
            0..=9 => 1,
            10 => 2,
            _ => 4,
        }
    }

    /// Number of butteraugli-loop seeds to run by effort (RFC#45 pick #1
    /// chunk 3). e ≤ 9 keeps the single-seed libjxl behaviour
    /// (bit-identical hash-locks); e10 fans out 2 seeds, e11 fans out 4,
    /// and the picker keeps the smallest-bytes seed that meets target
    /// butteraugli. See [`EffortProfile::lossy_search_seeds`] for the
    /// selection rule and seed values.
    fn lossy_search_seeds_for(effort: u8) -> u8 {
        match effort {
            0..=9 => 1,
            10 => 2,
            _ => 4,
        }
    }

    /// Smart per-image fanout adapter (opt-in via
    /// [`crate::api::LosslessConfig::with_smart_fanout`]).
    ///
    /// Re-tunes the three `tree_parallel_*` fields based on the input
    /// image's pixel count, not just effort. Per the
    /// `smart_fanout_sweep_2026-05-17` (8-image × 3-effort × 6-cell)
    /// investigation, depth=6 + floor=4096 wins or ties the
    /// effort-only defaults on every (image, effort) cell measured,
    /// EXCEPT large + e9 where the per-leaf subtree-build is enormous
    /// (~21 s sequential) and the current depth=5 is already optimal.
    ///
    /// Rule (post-sweep):
    /// - `pixels >= 4_000_000` and `effort >= 9`: keep effort default
    ///   (depth=5, floor=8192) — large e9 ceiling is the per-leaf
    ///   subtree, not parallel granularity.
    /// - otherwise: bump to depth=6, floor=4096, root_threshold=4096.
    ///
    /// Parallelism does not change the bitstream — the tree topology
    /// is determined by the samples, not the build order — so
    /// hash_lock sidecars stay byte-identical. This is purely a
    /// wall-clock knob.
    ///
    /// Investigation memory file:
    /// `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/
    ///  zenanalyze_tree_size_correlation_2026-05-17.md`.
    pub fn adapt_to_image(&mut self, pixels: u64) {
        let effort = self.effort;
        let large = pixels >= 4_000_000;
        if large && effort >= 9 {
            // Keep effort-only default (already tuned for the huge-tree case).
            return;
        }
        self.tree_parallel_max_depth = 6;
        self.tree_parallel_floor = 4_096;
        self.tree_parallel_root_threshold = 4_096;
    }

    /// Pixel-count + effort gate for the small-image parallel-tree-
    /// learning fallback. Always-on (NOT opt-in) — addresses the
    /// +0.85% small-image mean wall-clock regression documented in
    /// commit `cb5e202` (thread-local [`SplitWorkspace`] cache).
    ///
    /// When `pixels < SMALL_IMAGE_PIXEL_THRESHOLD` (1 MP) AND
    /// `effort <= 7`, flips `tree_parallel_small_image_fallback` to
    /// `true`. That causes
    /// [`crate::modular::tree_learn::compute_best_tree`] to allocate a
    /// fresh [`SplitWorkspace`] per `find_best_split` call instead of
    /// routing through the thread-local cache. The cache pays its own
    /// `RefCell::borrow_mut` indirection cost without amortising on
    /// small inputs at low effort (the workspace allocates once per
    /// encode anyway, and the tree is small enough that the cache hit
    /// rate doesn't matter).
    ///
    /// At effort >= 8 the tree grows enough that the per-call
    /// `SplitWorkspace::new` cost dominates the cache's `borrow_mut`
    /// indirection (paired bench at 0.26 MP × e9 measured the no-cache
    /// variant 7.45% SLOWER than the cached variant — exceeds the
    /// audit's small-image regression by an order of magnitude). The
    /// gate excludes e8+ to avoid that regression.
    ///
    /// The parallel root split + recursive borrowed-view fan-out
    /// remain ENABLED in this fallback regime — they are still the
    /// largest single wall-clock win at 8 threads, even on 0.26 MP.
    ///
    /// Bitstream-equivalent: tree topology depends only on the samples,
    /// not the workspace identity. Hash-locks stay byte-identical.
    ///
    /// Threshold rationale: per the
    /// `rejected_optimizations_conditional_value_2026-05-17.md` audit
    /// (item #10), the cache regression pivot is between 0.26 MP
    /// (small, +0.85% slower with cache) and 1.05 MP (medium, -2.6%
    /// faster with cache), measured at e7. The size gate is 1 MP and
    /// the effort gate is e7 (the audit's measurement effort).
    pub fn adapt_small_image_fallback(&mut self, pixels: u64) {
        if pixels < SMALL_IMAGE_PIXEL_THRESHOLD && self.effort <= 7 {
            self.tree_parallel_small_image_fallback = true;
        }
    }

    /// Pixel-count + effort gate for the `tree_max_buckets` dispatch
    /// (audit item #3, conditional-value catalog
    /// `rejected_optimizations_conditional_value_2026-05-17.md`).
    /// Always-on (NOT opt-in) — bytes change at large+e9 only, where
    /// the dispatch saves wall-clock at near-zero byte cost.
    ///
    /// When `pixels >= LARGE_IMAGE_PIXEL_THRESHOLD` (4 MP) AND
    /// `effort >= 9`, drops `tree_max_buckets` from the effort default
    /// (256 at e9) to [`LARGE_E9_TREE_MAX_BUCKETS`] (192).
    ///
    /// **Pareto evidence**
    /// (commit `4572790` Pareto sweep, `benchmarks/tree_max_buckets_pareto_2026-05-17.tsv`,
    /// 5 samples × 3 profile images × 6 bucket values @ effort 9,
    /// `RAYON_NUM_THREADS=8`, release build with `parallel-tree-learning`):
    ///
    /// | buckets | small_0.26MP   | medium_1.05MP  | large_4.19MP        |
    /// |---------|----------------|----------------|---------------------|
    /// | **192** | +1.79% / -4.9% | +0.10% / +1.9% | **+0.09% / -12.1%** |
    /// | 256     | baseline       | baseline       | baseline            |
    ///
    /// 192 is the only candidate where bytes stay in the noise floor on
    /// large (+0.09%) AND wall-clock wins are real (-12.1%). The
    /// per-property bucket-sweep in `find_best_split` scales roughly
    /// linearly with the bucket cap; the savings amortise only when
    /// the tree is deep enough — i.e. at the largest tier (≥4 MP at
    /// the most-expensive effort).
    ///
    /// Hash-locks: change at large+e9 cells (+0.09% byte cost is
    /// intentional). All other (size, effort) cells stay byte-identical
    /// because the dispatch does not fire.
    ///
    /// Threshold rationale: the same `pixels >= 4_000_000` boundary as
    /// [`Self::adapt_to_image`] (smart-fanout's large carve-out) and
    /// the audit's "≥3 MP" guidance. Effort gate is `>= 9` because the
    /// Pareto sweep was run only at e9 — at e7/e8 the win was not
    /// measured (`tree_max_buckets_for` returns 96/128 at e7/e8, so
    /// 192 would be an INCREASE, never measured).
    pub fn adapt_tree_max_buckets_for_image(&mut self, pixels: u64) {
        if pixels >= LARGE_IMAGE_PIXEL_THRESHOLD && self.effort >= 9 {
            self.tree_max_buckets = LARGE_E9_TREE_MAX_BUCKETS;
        }
    }

    /// Pixel-count + distance gate for the lossy VarDCT
    /// `try_dct64` evaluation. Always-on (NOT opt-in) — purely a
    /// wall-clock win on the small + low-distance cell.
    ///
    /// When `pixels < LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD` (500_000)
    /// AND `distance < LOSSY_LOW_DISTANCE_THRESHOLD` (2.0), drops
    /// `try_dct64` from the effort default (`true` at effort ≥ 7) to
    /// `false`. Skips the entire
    /// [`crate::vardct::ac_strategy_search::find_best_64x64_transform`]
    /// pipeline (DCT64x64, 2×DCT64x32, 2×DCT32x64 candidates plus their
    /// 4×`find_best_32x32_transform` reuse path).
    ///
    /// **Rationale**: DCT64-class transforms cover 64×64 pixels. On a
    /// small image at low distance the cost-model entropy_mul
    /// (`2.25` for DCT64x64/DCT64x32 in pixel-domain mode) heavily
    /// penalises the 4096-coefficient block. On 512×512 (8×8 of 64×64
    /// tiles) at `d ≤ 1.0` they are essentially never picked —
    /// the per-tile cost gate in `find_best_64x64_transform` falls
    /// through to four recursive `find_best_32x32_transform` calls.
    /// The wasted work is the upfront DCT64x64 + 4 DCT64x32 +
    /// 4 DCT32x64 entropy estimates per 64×64 tile.
    ///
    /// **Hash-locks**: byte-identical at the gated cells (the skipped
    /// strategies were not winning at those sizes anyway — verified by
    /// the per-effort hash_lock sidecars at 13×17 / 32×32 / 48×48,
    /// none of which can evaluate a 64×64 block to begin with).
    ///
    /// **Threshold rationale**:
    /// - `pixels < 500_000`: covers the bench harness's `small_0.26MP`
    ///   cell (512×512 = 262_144 px). At ≥ 1 MP the corpus_regression
    ///   bench shows DCT64 starts winning on smooth regions, so the
    ///   gate stops short of medium.
    /// - `distance < 2.0`: matches the conservative gate documented in
    ///   `dropped_optimizations_for_parity_2026-05-15.md` (item #1
    ///   neighbourhood — DCT64 is "gated to d≥3.0" in the cost model
    ///   notes, and at d ∈ [2.0, 3.0] some images do pick DCT64).
    ///
    /// **Effort gate**: only applies when `try_dct64` is already on
    /// (`effort ≥ 7`). At effort < 7 this is a no-op.
    ///
    /// Bench provenance: paired A/B in
    /// `jxl-encoder/examples/vardct_ac_dispatch_paired_ab.rs`, results
    /// in `benchmarks/vardct_ac_dispatch_paired_2026-05-17.tsv`.
    pub fn adapt_to_image_lossy(&mut self, pixels: u64, distance: f32) {
        if pixels < LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD
            && distance < LOSSY_LOW_DISTANCE_THRESHOLD
            && self.try_dct64
        {
            self.try_dct64 = false;
        }
    }

    /// Content-class-aware per-image adapter (RFC #45 pick #4 chunk 1).
    ///
    /// Specializes encoder defaults based on a coarse content class
    /// (`Photo` vs `Screenshot` vs `Other` vs `Unknown`). Designed to be
    /// called *after* [`Self::adapt_to_image_lossy`] /
    /// [`Self::adapt_to_image`] so the size-dependent gates run first.
    ///
    /// **Current dispatch rule (chunk 1)** — `Screenshot`-class content
    /// at lossy effort 5 / 6 with `pixels >= CONTENT_CLASS_MIN_PIXELS`
    /// (256 × 256 = 65 536) and `distance > 0.0` flips
    /// `self.patches = true`. The libjxl default keeps patches off until
    /// effort 7 for VarDCT; on screenshots the per-corpus measured
    /// savings (≈ 37 % on GB82-SC at e7) justify enabling them one or two
    /// effort levels earlier. Photos and unknown-class inputs are
    /// untouched, so hash-locks on the standard fixtures stay byte-
    /// identical (those fixtures are all sub-256² synthetic test images,
    /// well below the `CONTENT_CLASS_MIN_PIXELS` gate).
    ///
    /// All other content classes / effort levels are no-ops; the dispatch
    /// surface is extensible and future chunks can add more rules
    /// (per-class `tree_max_buckets`, per-class `try_dct4x8_afv`, etc.)
    /// without breaking callers.
    ///
    /// **Spec-compliance**: every dispatched change leaves the bitstream
    /// 100 % spec-valid (patches is a normal encoder feature, libjxl
    /// decoder reads it natively).
    ///
    /// **Effort gate rationale**: the dispatch fires at e ∈ {5, 6} because
    /// (a) e7 already has patches on by default and (b) e ≤ 4 disables
    /// most VarDCT machinery that patches piggybacks on (AC strategy
    /// search). The pixel gate excludes synthetic fixtures (the largest
    /// hash-lock fixture is 48 × 48 = 2 304 px, three orders of magnitude
    /// below 65 536).
    pub fn adapt_to_image_content(
        &mut self,
        pixels: u64,
        distance: f32,
        content_class: ImageContentClass,
    ) {
        if pixels < CONTENT_CLASS_MIN_PIXELS {
            return;
        }
        if content_class == ImageContentClass::Screenshot
            && distance > 0.0
            && (self.effort == 5 || self.effort == 6)
            && !self.patches
        {
            self.patches = true;
        }
    }

    /// Bias the profile toward bitstreams that decode faster, at the cost
    /// of compression. Mirrors libjxl `cparams.decoding_speed_tier` /
    /// `cjxl --faster_decoding 0..4`. Applied on top of the effort-derived
    /// profile and any `__expert` overrides — call last.
    ///
    /// Per-tier effects (additive — tier N applies the changes for tiers
    /// 1..=N):
    ///
    /// - `0`: no-op (default).
    /// - `1`: disable LZ77 backward references.
    ///   - VarDCT: AC stream tokens no longer rate-search LZ77 (libjxl
    ///     `enc_ans.cc:1372` flips `lz77_method = kNone` for VarDCT at
    ///     `decoding_speed_tier >= 1`).
    ///   - Modular: residual streams skip LZ77 (libjxl `enc_modular.cc`
    ///     `cparams_.decoding_speed_tier >= 1` clamps the histogram-pass
    ///     LZ77 method).
    ///   - Modular DC stream switches to the fixed `kGradientFixedDC` tree
    ///     (libjxl `enc_modular.cc:1600`) — handled by [`Self::tree_learning`]
    ///     being false on the DC sub-stream below.
    /// - `2`: tier 1 plus drop enhanced (pair-merge) histogram clustering
    ///   for VarDCT. libjxl caps modular `max_histograms = 12` and forces
    ///   `modular_group_size_shift = 0` at this tier; the group-size
    ///   override is applied by the per-config getter
    ///   ([`crate::api::LosslessConfig::effective_modular_group_size_shift`]),
    ///   not on this profile.
    /// - `3`: tier 2 plus drop custom coefficient orders. Decoders skip the
    ///   per-block permutation lookup and use the fixed natural order
    ///   (libjxl `enc_modular.cc:533` raises the tree-split threshold by
    ///   `+10 * decoding_speed_tier` — captured here by lowering tree
    ///   shape parameters).
    /// - `4`: tier 3 plus simpler context tree + no patches/tree-learning
    ///   pass on the modular path. libjxl also disables gaborish
    ///   (`enc_frame.cc:280`), DCT32X32 (`enc_ac_strategy.cc:936`), and
    ///   the `decoding_speed_tier_max_limit < 4` AC merges; mirrored here
    ///   by flipping `gaborish` / `try_dct32` / `try_dct64`.
    ///
    /// Bitstream remains 100 % spec-valid at every tier — these are encoder
    /// choices the libjxl decoder reads natively.
    pub fn apply_faster_decoding(&mut self, tier: u8) {
        if tier == 0 {
            return;
        }
        // Tier 1: disable LZ77.
        if tier >= 1 {
            self.lz77 = false;
        }
        // Tier 2: + disable enhanced (pair-merge) clustering for VarDCT.
        if tier >= 2 {
            self.enhanced_clustering_vardct = false;
        }
        // Tier 3: + drop custom coefficient orders, raise tree-split
        // threshold (libjxl enc_modular.cc:533 `+10 * speed_tier`).
        if tier >= 3 {
            self.custom_orders = false;
            // Mirror libjxl `splitting_heuristics_node_threshold +=
            // 10 * decoding_speed_tier` — at tier 3 that's +30 over the
            // effort-derived base, biasing the tree shallower.
            self.tree_threshold_base += 10.0 * tier as f32;
        }
        // Tier 4: + no MA tree learning, no patches; force-disable the
        // libjxl-gated VarDCT features (gaborish, DCT32X32, DCT64).
        if tier >= 4 {
            self.tree_learning = false;
            self.patches = false;
            self.gaborish = false;
            self.try_dct32 = false;
            self.try_dct64 = false;
            // Tighter MA-tree shape on the modular side (libjxl
            // enc_modular.cc:506-513 `nb_repeats = 0` is the strongest
            // signal — captured by zeroing tree_sample_fraction so the
            // sampler returns the 65k floor and the tree learner sees
            // minimal data).
            self.tree_sample_fraction = 0.0;
        }
    }
}

/// Coarse content class used by [`EffortProfile::adapt_to_image_content`].
///
/// Computed externally (typically via the optional `zenanalyze` integration
/// in [`crate::api`]); the [`EffortProfile`] surface intentionally
/// does not depend on the feature-extraction crate. Callers that don't have
/// classification available should pass [`Self::Unknown`] — every dispatch
/// rule treats it as "no change".
///
/// **Stability**: the variant set is `#[non_exhaustive]`; future chunks may
/// add classes (e.g., `Illustration`, `Document`, `LineArt`) without a
/// breaking change. Match arms must use `_` for the catch-all.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageContentClass {
    /// No classification available (default). Every dispatch rule treats
    /// this as "leave profile alone".
    #[default]
    Unknown,
    /// Natural photograph — high `edge_density`, low
    /// `flat_color_block_ratio`, non-zero `skin_tone_fraction` on portraits.
    Photo,
    /// Screen content — UI / document / terminal capture. High
    /// `flat_color_block_ratio` and `uniformity`, low `chroma_complexity`.
    /// Drives `patches` enablement at lower effort levels.
    Screenshot,
    /// Other / mixed content that fits none of the above buckets cleanly.
    /// No dispatch rules fire on this class today.
    Other,
}

/// Pixel-count threshold below which the parallel-tree-learning path
/// bypasses the thread-local [`SplitWorkspace`] cache (per-call
/// `SplitWorkspace::new` instead). The parallel root split + recursive
/// fan-out remain enabled — only the workspace allocation strategy
/// changes. See [`EffortProfile::adapt_small_image_fallback`].
pub const SMALL_IMAGE_PIXEL_THRESHOLD: u64 = 1_000_000;

/// Pixel-count threshold at or above which the `tree_max_buckets`
/// dispatch fires (at effort >= 9). See
/// [`EffortProfile::adapt_tree_max_buckets_for_image`].
pub const LARGE_IMAGE_PIXEL_THRESHOLD: u64 = 4_000_000;

/// `tree_max_buckets` value at large+e9 cells. Replaces the e9 default
/// of 256. See [`EffortProfile::adapt_tree_max_buckets_for_image`].
pub const LARGE_E9_TREE_MAX_BUCKETS: u16 = 192;

/// Pixel-count threshold below which the lossy VarDCT
/// `adapt_to_image_lossy` adapter disables the DCT64 strategy class
/// at low distance. See [`EffortProfile::adapt_to_image_lossy`].
pub const LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD: u64 = 500_000;

/// Distance below which the lossy VarDCT `adapt_to_image_lossy`
/// adapter disables the DCT64 strategy class on small images.
/// See [`EffortProfile::adapt_to_image_lossy`].
pub const LOSSY_LOW_DISTANCE_THRESHOLD: f32 = 2.0;

/// Minimum pixel count for content-class dispatch to consider firing.
/// Below this the classifier is unreliable (synthetic / thumbnail content)
/// and the per-fixture hash-locks are well below the threshold.
/// See [`EffortProfile::adapt_to_image_content`].
pub const CONTENT_CLASS_MIN_PIXELS: u64 = 65_536;

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

    /// Override the number of butteraugli-loop seeds (RFC#45 pick #1
    /// chunk 3). See [`EffortProfile::lossy_search_seeds`] for the
    /// per-effort defaults and the seed-selection rule. Setting this to
    /// `Some(1)` reverts to libjxl's single-seed loop even at e10/e11.
    pub lossy_search_seeds: Option<u8>,
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

    /// Phase 3 of issue #41 — when [`Self::gather_dedup`] is `Some(true)`,
    /// route the gather-time dedup through
    /// `crate::modular::inline_dedup_table::InlineDedupTable` instead of
    /// Phase 2's [`crate::modular::tree_learn::GatherDedupTable`].
    ///
    /// `Some(true)` enables the inline-fingerprint cuckoo table; `Some(false)`
    /// stays on the Phase 2 (SoA-indexed) table; `None` leaves the
    /// effort-profile default (always `false` today; see
    /// [`EffortProfile::gather_dedup_phase3`]).
    ///
    /// Has no effect unless [`Self::gather_dedup`] also routes traffic into
    /// the gather-time dedup path; gather-time dedup is a prerequisite.
    ///
    /// Hash-locks behave identically to Phase 2 (the post-`pre_quantize`
    /// sort path remains the byte-determining arbiter), so flipping this
    /// switch on top of an already-enabled `gather_dedup` does NOT require
    /// re-baking hash_lock sidecars — but it DOES change end-to-end
    /// wall-clock, which is the only reason to use it.
    pub gather_dedup_phase3: Option<bool>,

    /// Maximum depth of parallel recursion in the tree learner
    /// (`tree_learn.rs` `build_subtree_recursive_parallel_borrowed`).
    /// `2^depth` is the upper bound on parallel leaf tasks.
    /// Default schedule: 4 at effort ≤ 7 (16 leaf tasks), 5 at effort ≥ 8
    /// (32 leaf tasks — deeper e8/e9 trees benefit from finer-grained fanout).
    pub tree_parallel_max_depth: Option<u32>,

    /// Minimum subtree size below which recursive parallel fork is skipped
    /// (`tree_learn.rs` `PARALLEL_RECURSION_FLOOR`). Below this sample
    /// count rayon task overhead exceeds the parallel savings.
    /// Default schedule: 16384 at effort ≤ 7, 8192 at effort ≥ 8.
    pub tree_parallel_floor: Option<usize>,

    /// Minimum total sample count to even attempt the parallel root split
    /// (`tree_learn.rs` `PARALLEL_THRESHOLD`). Below this the sequential
    /// loop is faster overall.
    /// Default schedule: 8192 at effort ≤ 7, 4096 at effort ≥ 8.
    pub tree_parallel_root_threshold: Option<usize>,

    /// Override the small-image parallel-tree-learning fallback
    /// (see [`EffortProfile::tree_parallel_small_image_fallback`]).
    ///
    /// `Some(true)`: force the sequential fallback regardless of image
    /// size. `Some(false)`: force the parallel + thread-local-cache path
    /// regardless of image size (the pre-audit default behaviour).
    /// `None`: keep the always-on auto-gate that flips this to `true`
    /// for inputs smaller than [`SMALL_IMAGE_PIXEL_THRESHOLD`] (1 MP).
    ///
    /// Intended for sweep harnesses A/B-ing the gate; production
    /// callers should leave this `None`.
    pub tree_parallel_small_image_fallback: Option<bool>,

    /// Override the number of multi-seed tree-learning runs
    /// (see [`EffortProfile::tree_learn_seeds`]).
    ///
    /// `Some(1)` forces single-pass tree learning (libjxl-equivalent,
    /// byte-identical to the pre-RFC#45-chunk-2 default at any effort).
    /// `Some(N)` with `N >= 2` runs gather→tree `N` times with different
    /// stride offsets and keeps the tree whose tokens have the lowest
    /// entropy cost. `None` keeps the effort-derived default (1 at
    /// e ≤ 9, 2 at e10, 4 at e11).
    ///
    /// Output is bitstream-valid for any `N`. Sweep harnesses re-baking
    /// hash_lock sidecars should be aware that `N >= 2` *can* change the
    /// chosen tree per (image, distance) cell.
    pub tree_learn_seeds: Option<u8>,
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
            lossy_search_seeds,
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
        if let Some(v) = lossy_search_seeds {
            profile.lossy_search_seeds = v;
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
            gather_dedup_phase3,
            tree_parallel_max_depth,
            tree_parallel_floor,
            tree_parallel_root_threshold,
            tree_parallel_small_image_fallback,
            tree_learn_seeds,
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
        if let Some(v) = gather_dedup_phase3 {
            profile.gather_dedup_phase3 = v;
        }
        if let Some(v) = tree_parallel_max_depth {
            profile.tree_parallel_max_depth = v;
        }
        if let Some(v) = tree_parallel_floor {
            profile.tree_parallel_floor = v;
        }
        if let Some(v) = tree_parallel_root_threshold {
            profile.tree_parallel_root_threshold = v;
        }
        if let Some(v) = tree_parallel_small_image_fallback {
            profile.tree_parallel_small_image_fallback = v;
        }
        if let Some(v) = tree_learn_seeds {
            profile.tree_learn_seeds = v;
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
        // RFC#45 chunk 1: clamp bumped 10 → 11 to admit e10/e11.
        let p = EffortProfile::lossy(99, EncoderMode::Reference);
        assert_eq!(p.effort, 11);
    }

    #[test]
    fn test_lossy_search_seeds_e10_e11_extended() {
        // RFC#45 chunk 3: multi-seed butteraugli sweep at e10/e11.
        // e ≤ 9 keeps the libjxl single-seed behaviour (bit-identical
        // hash-locks); e10/e11 fan out 2/4 seeds and pick smallest bytes.
        for effort in 1..=9 {
            let p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert_eq!(
                p.lossy_search_seeds, 1,
                "e{effort}: single seed (libjxl-equivalent)"
            );
        }
        let p10 = EffortProfile::lossy(10, EncoderMode::Reference);
        let p11 = EffortProfile::lossy(11, EncoderMode::Reference);
        assert_eq!(p10.lossy_search_seeds, 2, "e10 = 2× seeds");
        assert_eq!(p11.lossy_search_seeds, 4, "e11 = 4× seeds");

        // Lossless never runs the buttloop; field must stay at 1 so a
        // future lossless caller that accidentally checks it doesn't
        // launch a phantom sweep.
        for effort in 1..=11 {
            let p = EffortProfile::lossless(effort, EncoderMode::Reference);
            assert_eq!(
                p.lossy_search_seeds, 1,
                "lossless e{effort}: never fans out"
            );
        }

        // Experimental inherits the reference value.
        let pe11 = EffortProfile::lossy(11, EncoderMode::Experimental);
        assert_eq!(pe11.lossy_search_seeds, 4);
    }

    #[test]
    #[cfg(feature = "butteraugli-loop")]
    fn test_init_mul_seeds_invariants() {
        use crate::vardct::butteraugli_loop::{LIBJXL_INIT_MUL, init_mul_seeds};
        // Index 0 must ALWAYS be the libjxl default so multi-seed can
        // never regress below single-seed worst-case.
        for seeds in [1, 2, 3, 4, 5, 10, 99, 255_u8] {
            let table = init_mul_seeds(seeds);
            assert!(!table.is_empty(), "seeds={seeds}: table empty");
            assert!(
                (table[0] - LIBJXL_INIT_MUL).abs() < f64::EPSILON,
                "seeds={seeds}: index 0 ({}) must equal LIBJXL_INIT_MUL ({LIBJXL_INIT_MUL})",
                table[0]
            );
            // Saturation cap: each seed is unique, no NaN/inf, bounded.
            for (i, &v) in table.iter().enumerate() {
                assert!(v.is_finite(), "seeds={seeds}[{i}]: non-finite {v}");
                assert!(
                    (0.0..=1.0).contains(&v),
                    "seeds={seeds}[{i}]: {v} outside [0, 1]"
                );
            }
        }
        // `0` defensively bumps to `1` (same single-seed behaviour).
        assert_eq!(init_mul_seeds(0).len(), 1);
        assert_eq!(init_mul_seeds(1).len(), 1);
        assert_eq!(init_mul_seeds(2).len(), 2);
        assert_eq!(init_mul_seeds(3).len(), 3);
        assert_eq!(init_mul_seeds(4).len(), 4);
        // Saturate at table length so requesting more is safe.
        assert_eq!(init_mul_seeds(255).len(), 4);
    }

    #[test]
    fn test_butteraugli_iters_e10_e11_extended() {
        // RFC#45 chunk 1: longer butteraugli search budgets at e10/e11.
        // e9 = libjxl kTortoise max (4 iters), e10 = 8, e11 = 16
        // (saturated at MAX_QUANT_LOOP_ITERS = ITER_MAX = 16).
        let p9 = EffortProfile::lossy(9, EncoderMode::Reference);
        let p10 = EffortProfile::lossy(10, EncoderMode::Reference);
        let p11 = EffortProfile::lossy(11, EncoderMode::Reference);
        assert_eq!(p9.butteraugli_iters, 4, "e9 = libjxl kTortoise default");
        assert_eq!(p10.butteraugli_iters, 8, "e10 = 2× e9 budget");
        assert_eq!(
            p11.butteraugli_iters, 16,
            "e11 = 4× e9, saturated at MAX_QUANT_LOOP_ITERS"
        );
        // Sanity: stays at saturation cap even if effort overshoots.
        // (The lossy() clamp pins at 11; verify the table never returns
        // anything above the loop's structural cap.)
        assert!(
            p11.butteraugli_iters <= crate::api::MAX_QUANT_LOOP_ITERS,
            "butteraugli_iters must not exceed MAX_QUANT_LOOP_ITERS"
        );
    }

    #[test]
    fn test_experimental_diverges_from_reference() {
        // Experimental should share effort/feature-flag structure with reference
        for effort in 1..=11 {
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

    /// chunk-2 (`lossless_e8_e9_cliff_2026-05-16.md`): effort-tune the rayon
    /// fanout shape for the parallel tree learner. At e ≤ 7 the schedule
    /// matches the pre-chunk-2 hardcoded constants exactly so the e7 hash
    /// lock and bytes are byte-identical. At e ≥ 8 the deeper trees +
    /// heavier per-leaf work benefit from a deeper fanout + lower floor.
    #[test]
    fn test_tree_parallel_schedule_lossless() {
        for effort in [1u8, 5, 7] {
            let p = EffortProfile::lossless(effort, EncoderMode::Reference);
            assert_eq!(p.tree_parallel_max_depth, 4, "e{}", effort);
            assert_eq!(p.tree_parallel_floor, 16_384, "e{}", effort);
            assert_eq!(p.tree_parallel_root_threshold, 8_192, "e{}", effort);
        }
        for effort in [8u8, 9, 10] {
            let p = EffortProfile::lossless(effort, EncoderMode::Reference);
            assert_eq!(p.tree_parallel_max_depth, 5, "e{}", effort);
            assert_eq!(p.tree_parallel_floor, 8_192, "e{}", effort);
            assert_eq!(p.tree_parallel_root_threshold, 4_096, "e{}", effort);
        }
    }

    #[test]
    fn test_tree_parallel_schedule_lossy_matches_lossless() {
        // Lossy and lossless both surface the parallel-tree-learning knobs
        // (lossy uses tree learning for patch reference frames). The defaults
        // must match so a picker sees one canonical schedule per effort.
        for effort in 1u8..=10 {
            let l = EffortProfile::lossless(effort, EncoderMode::Reference);
            let v = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert_eq!(l.tree_parallel_max_depth, v.tree_parallel_max_depth);
            assert_eq!(l.tree_parallel_floor, v.tree_parallel_floor);
            assert_eq!(
                l.tree_parallel_root_threshold,
                v.tree_parallel_root_threshold
            );
        }
    }

    /// `adapt_to_image` is the per-image smart-fanout rule shipped with the
    /// `smart_fanout_sweep_2026-05-17` chunk-1 investigation. For all
    /// `(effort, pixels)` combos EXCEPT large+e9 (where the per-leaf
    /// subtree-build ceiling dominates) it bumps the schedule to
    /// depth=6 / floor=4096 / root_threshold=4096. Large+e9 keeps the
    /// effort-only schedule.
    #[test]
    fn test_adapt_to_image_smart_fanout_rule() {
        // Small / medium / large @ e7: rule should kick in for all.
        for &pixels in &[262_144u64, 1_048_576, 4_194_304] {
            let mut p = EffortProfile::lossless(7, EncoderMode::Reference);
            p.adapt_to_image(pixels);
            assert_eq!(p.tree_parallel_max_depth, 6, "e7 pixels={pixels}");
            assert_eq!(p.tree_parallel_floor, 4_096, "e7 pixels={pixels}");
            assert_eq!(p.tree_parallel_root_threshold, 4_096, "e7 pixels={pixels}");
        }
        // e8: same as e7 (rule applies to all sizes).
        for &pixels in &[262_144u64, 1_048_576, 4_194_304] {
            let mut p = EffortProfile::lossless(8, EncoderMode::Reference);
            p.adapt_to_image(pixels);
            assert_eq!(p.tree_parallel_max_depth, 6, "e8 pixels={pixels}");
            assert_eq!(p.tree_parallel_floor, 4_096, "e8 pixels={pixels}");
        }
        // e9 large: keep effort-only (depth=5 / floor=8192).
        let mut p = EffortProfile::lossless(9, EncoderMode::Reference);
        p.adapt_to_image(8_000_000);
        assert_eq!(p.tree_parallel_max_depth, 5, "e9 large");
        assert_eq!(p.tree_parallel_floor, 8_192, "e9 large");
        // e9 medium: rule still kicks in.
        let mut p = EffortProfile::lossless(9, EncoderMode::Reference);
        p.adapt_to_image(1_048_576);
        assert_eq!(p.tree_parallel_max_depth, 6, "e9 medium");
        assert_eq!(p.tree_parallel_floor, 4_096, "e9 medium");
    }

    /// `adapt_small_image_fallback` is the always-on per-image gate (NOT
    /// opt-in) that flips `tree_parallel_small_image_fallback` to `true`
    /// for inputs below 1 MP AT EFFORT <= 7. Fixes the cache regression
    /// from `cb5e202` (+0.85% mean) at e7 small without triggering the
    /// inverse regression at e8/e9 where the tree is large enough that
    /// per-call `SplitWorkspace::new` dominates the cache's `borrow_mut`
    /// indirection.
    #[test]
    fn test_adapt_small_image_fallback_threshold() {
        // Default profile starts with fallback OFF.
        for effort in 1u8..=10 {
            let p = EffortProfile::lossless(effort, EncoderMode::Reference);
            assert!(
                !p.tree_parallel_small_image_fallback,
                "default profile must not pre-set fallback (effort={effort})"
            );
            let pl = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(
                !pl.tree_parallel_small_image_fallback,
                "lossy default profile must not pre-set fallback (effort={effort})"
            );
        }

        // Below size threshold AND effort <= 7: gate flips ON.
        for &pixels in &[1u64, 1_024, 262_144, 524_288, 999_999] {
            let mut p = EffortProfile::lossless(7, EncoderMode::Reference);
            p.adapt_small_image_fallback(pixels);
            assert!(
                p.tree_parallel_small_image_fallback,
                "fallback must be ON for pixels={pixels} (< {SMALL_IMAGE_PIXEL_THRESHOLD}) at e7"
            );
        }

        // At/above size threshold: gate stays OFF (regardless of effort).
        for &pixels in &[SMALL_IMAGE_PIXEL_THRESHOLD, 1_048_576, 4_194_304, 8_000_000] {
            for effort in 1u8..=10 {
                let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
                p.adapt_small_image_fallback(pixels);
                assert!(
                    !p.tree_parallel_small_image_fallback,
                    "fallback must be OFF for pixels={pixels} \
                     (>= {SMALL_IMAGE_PIXEL_THRESHOLD}) at effort={effort}"
                );
            }
        }

        // At small size: gate applies ONLY at effort <= 7. At e8+ the cache
        // dominates per-call alloc and disabling it regresses by 7%+ (audit
        // bench evidence — see effort.rs:adapt_small_image_fallback docs).
        for effort in 1u8..=7 {
            let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
            p.adapt_small_image_fallback(262_144);
            assert!(
                p.tree_parallel_small_image_fallback,
                "fallback must be ON at effort={effort} for 0.26 MP"
            );
        }
        for effort in 8u8..=10 {
            let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
            p.adapt_small_image_fallback(262_144);
            assert!(
                !p.tree_parallel_small_image_fallback,
                "fallback must be OFF at effort={effort} for 0.26 MP \
                 (cache helps at high effort — per-call alloc dominates)"
            );
        }
    }

    /// `adapt_tree_max_buckets_for_image` is the always-on per-image
    /// dispatch (audit item #3) that drops `tree_max_buckets` from 256
    /// to [`LARGE_E9_TREE_MAX_BUCKETS`] (192) on large+e9 cells only.
    /// Verifies the gate boundaries on both sides (pixels, effort) and
    /// confirms the rule never fires at e7/e8 or below 4 MP.
    #[test]
    fn test_adapt_tree_max_buckets_for_image_threshold() {
        // Pre-dispatch baseline values (matches tree_max_buckets_for).
        let baseline = |effort: u8| -> u16 {
            match effort {
                0..=4 => 32,
                5 => 48,
                6 => 64,
                7 => 96,
                8 => 128,
                _ => 256,
            }
        };

        // 1. e9 large (>= 4 MP): rule fires, buckets drop to 192.
        for &pixels in &[
            LARGE_IMAGE_PIXEL_THRESHOLD,
            4_194_304,
            8_000_000,
            16_777_216,
        ] {
            let mut p = EffortProfile::lossless(9, EncoderMode::Reference);
            assert_eq!(p.tree_max_buckets, 256, "e9 baseline buckets=256");
            p.adapt_tree_max_buckets_for_image(pixels);
            assert_eq!(
                p.tree_max_buckets, LARGE_E9_TREE_MAX_BUCKETS,
                "e9 pixels={pixels}: must drop to 192"
            );
        }
        // e10 large: same dispatch fires.
        let mut p = EffortProfile::lossless(10, EncoderMode::Reference);
        p.adapt_tree_max_buckets_for_image(8_000_000);
        assert_eq!(p.tree_max_buckets, LARGE_E9_TREE_MAX_BUCKETS, "e10 large");

        // 2. e9 below threshold (< 4 MP): rule does NOT fire, buckets stay 256.
        for &pixels in &[1u64, 1_024, 262_144, 1_048_576, 3_999_999] {
            let mut p = EffortProfile::lossless(9, EncoderMode::Reference);
            p.adapt_tree_max_buckets_for_image(pixels);
            assert_eq!(
                p.tree_max_buckets, 256,
                "e9 pixels={pixels} (< {LARGE_IMAGE_PIXEL_THRESHOLD}): must stay 256"
            );
        }

        // 3. e7/e8 large: rule does NOT fire (effort gate), keep effort default.
        for effort in 1u8..=8 {
            for &pixels in &[
                LARGE_IMAGE_PIXEL_THRESHOLD,
                4_194_304,
                8_000_000,
                16_777_216,
            ] {
                let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
                let want = baseline(effort);
                p.adapt_tree_max_buckets_for_image(pixels);
                assert_eq!(
                    p.tree_max_buckets, want,
                    "effort={effort} pixels={pixels}: \
                     must stay at baseline {want} (effort < 9)"
                );
            }
        }

        // 4. Cross-product spot check: all (effort, pixels) cells outside
        //    the (effort>=9 AND pixels>=4MP) box leave the profile unchanged.
        for effort in 1u8..=10 {
            for &pixels in &[262_144u64, 1_048_576, 3_999_999] {
                let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
                let want = baseline(effort);
                p.adapt_tree_max_buckets_for_image(pixels);
                assert_eq!(
                    p.tree_max_buckets, want,
                    "effort={effort} pixels={pixels}: no dispatch fire"
                );
            }
        }
    }

    /// Lossy profile must also honour the dispatch (lossy patches /
    /// reference frames go through tree learning too — the constants
    /// must stay consistent so a single canonical schedule applies).
    #[test]
    fn test_adapt_tree_max_buckets_lossy_profile_parity() {
        for effort in 9u8..=10 {
            let mut pl = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert_eq!(pl.tree_max_buckets, 256);
            pl.adapt_tree_max_buckets_for_image(8_000_000);
            assert_eq!(
                pl.tree_max_buckets, LARGE_E9_TREE_MAX_BUCKETS,
                "lossy e{effort} large: dispatch must apply"
            );
        }
    }

    /// Chunk 1 VarDCT AC strategy dispatch: `adapt_to_image_lossy`
    /// must flip `try_dct64` to `false` only on the
    /// (`pixels < LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD`,
    ///  `distance < LOSSY_LOW_DISTANCE_THRESHOLD`) cell, and only when
    /// effort already had `try_dct64 = true` (effort >= 7).
    #[test]
    fn test_adapt_to_image_lossy_dct64_gate() {
        // 1. Small + low-d at e7+: dispatch fires.
        for effort in 7u8..=10 {
            for &pixels in &[1u64, 1024, 262_144, 499_999] {
                for &distance in &[0.1_f32, 0.5, 1.0, 1.5, 1.999] {
                    let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
                    assert!(p.try_dct64, "baseline e{effort}: try_dct64 must be true");
                    p.adapt_to_image_lossy(pixels, distance);
                    assert!(
                        !p.try_dct64,
                        "e{effort} pixels={pixels} d={distance}: \
                         try_dct64 must drop to false"
                    );
                }
            }
        }

        // 2. Above pixel threshold: no fire.
        for &pixels in &[
            LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD,
            1_048_576,
            4_194_304,
            16_777_216,
        ] {
            for &distance in &[0.5_f32, 1.0, 1.5] {
                let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
                p.adapt_to_image_lossy(pixels, distance);
                assert!(
                    p.try_dct64,
                    "pixels={pixels} d={distance}: must stay true (pixel gate)"
                );
            }
        }

        // 3. At or above distance threshold: no fire.
        for &distance in &[LOSSY_LOW_DISTANCE_THRESHOLD, 2.5_f32, 3.0, 5.0, 10.0] {
            for &pixels in &[1u64, 262_144, 499_999] {
                let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
                p.adapt_to_image_lossy(pixels, distance);
                assert!(
                    p.try_dct64,
                    "pixels={pixels} d={distance}: must stay true (distance gate)"
                );
            }
        }

        // 4. Effort < 7: baseline try_dct64 already false — adapter
        //    must not flip it to true and must not panic.
        for effort in 1u8..=6 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(
                !p.try_dct64,
                "baseline e{effort}: try_dct64 should be false"
            );
            p.adapt_to_image_lossy(262_144, 1.0);
            assert!(
                !p.try_dct64,
                "e{effort}: adapter must not flip false → true"
            );
        }

        // 5. Lossy "experimental" mode also covered (try_dct64
        //    follows the same effort schedule).
        let mut p = EffortProfile::lossy(7, EncoderMode::Experimental);
        p.adapt_to_image_lossy(262_144, 1.0);
        assert!(
            !p.try_dct64,
            "lossy experimental e7 small + low-d: adapter still fires"
        );
    }

    /// RFC #45 pick #4 chunk 1 — `adapt_to_image_content` must flip
    /// `patches = true` on Screenshot-class content at e ∈ {5, 6} with
    /// `pixels >= CONTENT_CLASS_MIN_PIXELS`. All other (class, effort,
    /// pixels, distance) tuples must be no-ops.
    #[test]
    fn test_adapt_to_image_content_screenshot_enables_patches_at_e5_e6() {
        // 1. Screenshot at e5/e6, above pixel + distance threshold: fires.
        for effort in [5u8, 6] {
            for &pixels in &[CONTENT_CLASS_MIN_PIXELS, 262_144, 1_048_576, 4_194_304] {
                for &distance in &[0.5_f32, 1.0, 2.0, 5.0] {
                    let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
                    assert!(
                        !p.patches,
                        "baseline e{effort}: patches must be false (gate is e>=7)"
                    );
                    p.adapt_to_image_content(pixels, distance, ImageContentClass::Screenshot);
                    assert!(
                        p.patches,
                        "e{effort} pixels={pixels} d={distance} Screenshot: \
                         patches must flip to true"
                    );
                }
            }
        }

        // 2. Other content classes: no fire at e5/e6.
        for class in [
            ImageContentClass::Unknown,
            ImageContentClass::Photo,
            ImageContentClass::Other,
        ] {
            for effort in [5u8, 6] {
                let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
                p.adapt_to_image_content(262_144, 1.0, class);
                assert!(
                    !p.patches,
                    "e{effort} class={class:?}: patches must stay false"
                );
            }
        }

        // 3. Below pixel threshold: no fire even on Screenshot.
        for &pixels in &[0u64, 1, 1024, 65_535] {
            let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
            p.adapt_to_image_content(pixels, 1.0, ImageContentClass::Screenshot);
            assert!(
                !p.patches,
                "pixels={pixels} Screenshot: pixel gate must hold"
            );
        }

        // 4. distance == 0.0: no fire (lossless-equivalent reserved path).
        let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
        p.adapt_to_image_content(262_144, 0.0, ImageContentClass::Screenshot);
        assert!(!p.patches, "distance=0.0 Screenshot: must stay false");

        // 5. Effort 7+ (patches already on) — adapter is a no-op flag-wise.
        for effort in 7u8..=10 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(p.patches, "baseline e{effort}: patches must be true");
            p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
            assert!(p.patches, "e{effort} Screenshot: patches must remain true");
        }

        // 6. Effort < 5: adapter must NOT enable patches (libjxl path
        //    needs AC strategy search which is off at e<5).
        for effort in 1u8..=4 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(!p.patches, "baseline e{effort}: patches must be false");
            p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
            assert!(
                !p.patches,
                "e{effort} Screenshot: must respect effort floor"
            );
        }

        // 7. Default ImageContentClass is Unknown.
        let default_class: ImageContentClass = Default::default();
        assert_eq!(default_class, ImageContentClass::Unknown);
    }
}
