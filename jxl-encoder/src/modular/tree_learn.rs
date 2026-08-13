// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Content-adaptive MA tree learning for modular encoding.
//!
//! Replaces the fixed single-leaf gradient tree with a learned multi-leaf tree
//! that assigns optimal predictors and entropy contexts per image region.
//! Port of libjxl's `FindBestSplit` algorithm from `enc_ma.cc`.

use super::channel::{Channel, ModularImage};
use super::predictor::{
    Neighbors, Predictor, WeightedPredictorParams, WeightedPredictorState, pack_signed,
};
use super::predictor_prune::{
    PredictorDecision, decide_predictor, predictor_extra_bits_lower_bound,
};
use super::tree::{PropertyDecisionNode, Tree, assign_sequential_contexts, validate_tree_djxl};
use super::tree_learn_split;
use crate::entropy_coding::hybrid_uint::HybridUintConfig;

/// HybridUint config used during sample gathering: {4, 1, 2}.
/// Matches libjxl's gathering phase config.
const GATHER_HYBRID_UINT: HybridUintConfig = HybridUintConfig {
    split_exponent: 4,
    split: 16, // 1 << 4
    msb_in_token: 1,
    lsb_in_token: 2,
};

/// Number of properties used in tree learning (spec indices 0..16).
const NUM_PROPERTIES: usize = 16;

/// Candidate predictors for tree learning.
/// All 14 predictors (0-13). Weighted (6) uses WP state which is bit-exact with jxl-rs.
/// Property 15 (wp_max_error) is included in PROP_ORDER_NO_SQUEEZE and used at effort >= 7.
const CANDIDATE_PREDICTORS: &[Predictor] = &[
    Predictor::Zero,
    Predictor::Left,
    Predictor::Top,
    Predictor::Average0,
    Predictor::Select,
    Predictor::Gradient,
    Predictor::Weighted,
    Predictor::TopRight,
    Predictor::TopLeft,
    Predictor::LeftLeft,
    Predictor::Average1,
    Predictor::Average2,
    Predictor::Average3,
    Predictor::Average4,
];

/// Per-seed predictor permutations for the multi-seed tree-learning loop
/// (RFC#45 chunk 4 — predictor-evaluation-order variance).
///
/// `find_best_predictor` iterates `samples.candidate_predictors` in array
/// order and applies a strict-`<` tie-break (lowest index wins on equal
/// cost). Re-ordering the array therefore changes which predictor wins
/// every tied comparison — typically the cheap `Zero` / `Left` / `Top`
/// predictors that dominate flat regions, and the strong `Gradient` /
/// `Weighted` predictors that dominate textured regions.
///
/// Index 0 = canonical libjxl order (seed 0 / e ≤ 9 path).
/// Index 1 = strong-first (Gradient, Weighted promoted before Zero).
/// Index 2 = directional-first (TopLeft, TopRight, Average1..4 promoted).
/// Index 3 = full reverse.
///
/// All four arrays contain the same 14 predictors as a set — only the
/// order varies — so every per-seed tree is spec-valid and the chunk-2
/// `estimate_token_cost` picker still chooses among them on equal terms.
const CANDIDATE_PREDICTORS_PERMS: [&[Predictor]; 4] = [
    // 0: canonical (matches CANDIDATE_PREDICTORS exactly).
    &[
        Predictor::Zero,
        Predictor::Left,
        Predictor::Top,
        Predictor::Average0,
        Predictor::Select,
        Predictor::Gradient,
        Predictor::Weighted,
        Predictor::TopRight,
        Predictor::TopLeft,
        Predictor::LeftLeft,
        Predictor::Average1,
        Predictor::Average2,
        Predictor::Average3,
        Predictor::Average4,
    ],
    // 1: strong-first — Gradient + Weighted lead, so on ties they win.
    &[
        Predictor::Gradient,
        Predictor::Weighted,
        Predictor::Zero,
        Predictor::Left,
        Predictor::Top,
        Predictor::Average0,
        Predictor::Select,
        Predictor::TopRight,
        Predictor::TopLeft,
        Predictor::LeftLeft,
        Predictor::Average1,
        Predictor::Average2,
        Predictor::Average3,
        Predictor::Average4,
    ],
    // 2: directional-first — TopRight/TopLeft + the Average1..4 family lead.
    &[
        Predictor::TopRight,
        Predictor::TopLeft,
        Predictor::Average1,
        Predictor::Average2,
        Predictor::Average3,
        Predictor::Average4,
        Predictor::LeftLeft,
        Predictor::Zero,
        Predictor::Left,
        Predictor::Top,
        Predictor::Average0,
        Predictor::Select,
        Predictor::Gradient,
        Predictor::Weighted,
    ],
    // 3: full reverse of canonical.
    &[
        Predictor::Average4,
        Predictor::Average3,
        Predictor::Average2,
        Predictor::Average1,
        Predictor::LeftLeft,
        Predictor::TopLeft,
        Predictor::TopRight,
        Predictor::Weighted,
        Predictor::Gradient,
        Predictor::Select,
        Predictor::Average0,
        Predictor::Top,
        Predictor::Left,
        Predictor::Zero,
    ],
];

/// Full non-squeeze property order, matching libjxl's enc_modular.cc:544.
/// 16 elements with group_id at index 1. Used at effort 9+ (speed_tier <= kTortoise)
/// where libjxl does NOT erase group_id.
const PROP_ORDER_NO_SQUEEZE: &[usize] = &[
    0,  // Channel
    1,  // GroupId
    15, // WpMaxError
    9,  // W + N - NW (gradient)
    10, // W - NW
    11, // NW - N
    12, // N - NE
    13, // N - NN
    14, // W - WW
    2,  // Y
    3,  // X
    4,  // |N|
    5,  // |W|
    6,  // N
    7,  // W
    8,  // W - prev_gradient
];

/// Non-squeeze property order after group_id erasure, matching libjxl's
/// enc_modular.cc:546-549. 15 elements. Used at effort < 9 for lossless
/// modular with fewer than 30 streams (our typical single-group case).
const PROP_ORDER_NO_SQUEEZE_NO_GID: &[usize] = &[
    0,  // Channel
    15, // WpMaxError
    9,  // W + N - NW (gradient)
    10, // W - NW
    11, // NW - N
    12, // N - NE
    13, // N - NN
    14, // W - WW
    2,  // Y
    3,  // X
    4,  // |N|
    5,  // |W|
    6,  // N
    7,  // W
    8,  // W - prev_gradient
];

/// Squeeze-specific property order, matching libjxl's enc_modular.cc:538-541.
/// Squeeze residuals (Haar wavelet coefficients) benefit from spatial correlation
/// properties (|N|, |W|, N, W) earlier than gradient-difference properties.
///
/// 16 elements. Property 1 (group_id) is always included for squeeze mode in
/// libjxl — the group_id erasure only applies to non-squeeze lossless paths.
/// At effort 7 (kSquirrel), first 7 properties = {0, 1, 4, 5, 6, 7, 8}.
const PROP_ORDER_SQUEEZE: &[usize] = &[
    0,  // Channel
    1,  // GroupId
    4,  // |N|
    5,  // |W|
    6,  // N
    7,  // W
    8,  // W - prev_gradient
    15, // WpMaxError
    9,  // W + N - NW (gradient)
    10, // W - NW
    11, // NW - N
    12, // N - NE
    13, // N - NN
    14, // W - WW
    2,  // Y
    3,  // X
];

/// Squeeze candidate predictors: just Zero.
/// libjxl forces Predictor::Zero for squeeze residuals (enc_modular.cc:629-633):
/// "zero predictor for Squeeze residues and lossy palette indices"
/// Squeeze already decorrelates via Haar wavelet; adding prediction doesn't help.
const CANDIDATE_PREDICTORS_SQUEEZE: &[Predictor] = &[Predictor::Zero];

/// Parameters for tree learning, effort-dependent.
///
/// Matches libjxl's enc_modular.cc speed tier configuration:
/// - Squirrel (e7): first 7 properties, max 48 property values, threshold 131
/// - Kitten (e8): first 10 properties, max 96 property values, threshold 89
/// - Tortoise (e9/e10): all properties, max 256 property values, threshold 75
pub struct TreeLearningParams {
    /// Properties to consider for splits, in priority order.
    /// Includes base properties (0..16) and optionally reference channel
    /// properties (16+). Changed from `&'static [usize]` to `Vec<usize>` to
    /// support dynamic ref channel property indices.
    pub properties: Vec<usize>,
    /// Maximum number of quantized threshold buckets per property.
    pub max_property_values: usize,
    /// Base split threshold: scaled by `pixel_fraction * 0.9 + 0.1` to get effective threshold.
    /// A split must save at least `effective_threshold` bits to be accepted.
    pub split_threshold: f64,
    /// Maximum tree nodes. Absolute cap is `kMaxTreeSize = 1<<22` (ma_common.h).
    /// Per-frame decoder limit is `min(1<<20, 1024 + total_channel_pixels)`
    /// (encoding.cc:606-616). Encoder must not exceed these or output is un-decodable.
    pub max_nodes: usize,
    /// Fraction of pixels actually sampled (num_samples / total_pixels).
    /// Used to scale the split threshold: effective = threshold * (fraction * 0.9 + 0.1).
    /// Matches libjxl's `required_cost = pixel_fraction * 0.9 + 0.1` in LearnTree().
    /// Set to 1.0 if all pixels are sampled (no subsampling).
    pub pixel_fraction: f64,
    /// Use the streaming hash-table dedup (libjxl `AddSample` parity) instead
    /// of the default packed-key sort. Default `false` (sort path).
    ///
    /// The streaming path ports libjxl `AddToTableAndMerge` / `AddSample`
    /// (`enc_ma.cc:602-655`, `enc_ma.cc:711`) and avoids the O(n log n) sort
    /// over `n × 64 B` packed keys. In libjxl's source-tree layout the win is
    /// substantial because keys are built once during the gather pass.
    ///
    /// In our post-gather pipeline the streaming path **regresses** wall-clock
    /// by +3 % to +8 % on real CLIC photos at e7 (issue #41) — `pack_sample_key`
    /// random-accesses parallel SoA arrays per sample, defeating cache
    /// locality, and the sort path benefits from packed-key spatial coherence
    /// that the hash path cannot exploit. The streaming knob is retained for
    /// experimentation toward issue #41 Phase 2 (integrate dedup into the
    /// gather pass itself, eliminating the random-access pattern).
    pub use_streaming_dedup: bool,
    /// Integrate the two-hash cuckoo dedup *into* the gather loop itself
    /// (libjxl `AddSample` parity, `enc_ma.cc:711`).
    ///
    /// This is the true Phase 2 of issue #41: each pushed sample is
    /// immediately probed against a per-thread [`GatherDedupTable`] and
    /// either merged (pop_back + count++) or retained. No post-pass over
    /// `pack_sample_key`, so the cold-cache SoA reads that doomed Phase 1
    /// disappear. Output is **not** byte-identical to the sort-dedup
    /// default because gather-time dedup hashes on raw i32 property values
    /// (pre-quantization is run later, on the already-deduplicated set).
    ///
    /// Default `false`. Callers opt in via
    /// [`crate::api::LosslessConfig`] `__expert` overrides and re-bake
    /// the hash-lock sidecars; the sort path remains the byte-identical
    /// default.
    ///
    /// **MEASURED: ALWAYS WORSE OR TIE** at every size (W44-137 at
    /// 1024²; `perf_dedup_8mp_rebench_2026-06-10.meta` at 4-12 MP) —
    /// A/B infrastructure only.
    pub gather_dedup: bool,
    /// Phase 3 of issue #41 — when [`Self::gather_dedup`] is also `true`,
    /// route the gather-time dedup table through
    /// [`crate::modular::inline_dedup_table::InlineDedupTable`] instead of
    /// [`GatherDedupTable`]. The post-sort arbiter (`dedup_samples`) still
    /// runs, so bitstream hash-locks stay byte-identical to Phase 2's
    /// gather-dedup baseline.
    ///
    /// Default `false`. Has no effect when `gather_dedup` is `false`.
    ///
    /// **MEASURED: P2↔P3 is noise (±1.3 %) and the inline-dedup family
    /// loses end-to-end at every size** — A/B infrastructure only
    /// (`perf_dedup_8mp_rebench_2026-06-10.meta`).
    pub gather_dedup_phase3: bool,
    /// Maximum depth of parallel recursion in the borrowed-view subtree
    /// builder (`build_subtree_recursive_parallel_borrowed`).
    /// `2^depth` is the upper bound on parallel leaf tasks.
    /// Read by the parallel-tree-learning gated path; ignored when the
    /// feature is disabled.
    pub parallel_max_depth: u32,
    /// Minimum subtree size below which further parallel fork is skipped
    /// and the iterative sequential builder runs instead.
    /// Read by the parallel-tree-learning gated path; ignored otherwise.
    pub parallel_recursion_floor: usize,
    /// Minimum total sample count required before attempting the parallel
    /// root split. Below this the sequential loop is faster overall.
    /// Read by the parallel-tree-learning gated path; ignored otherwise.
    pub parallel_root_threshold: usize,
    /// Small-image fallback: when `true`, [`compute_best_tree`]
    /// bypasses the thread-local [`SplitWorkspace`] cache (per-call
    /// `SplitWorkspace::new` instead of `RefCell::borrow_mut` +
    /// reset_for). The parallel root split + borrowed-view fan-out
    /// REMAIN ENABLED — only the cache layer changes. Addresses the
    /// +0.85% small-image regression from commit `cb5e202`.
    ///
    /// Set automatically by
    /// [`crate::effort::EffortProfile::adapt_small_image_fallback`]
    /// when the input image is below
    /// [`crate::effort::SMALL_IMAGE_PIXEL_THRESHOLD`].
    /// Bitstream-equivalent.
    pub parallel_small_image_fallback: bool,
    /// Use **Lloyd-Max iterative clustering** to choose bucket boundaries
    /// for the energy-correlated tree-learning properties (4 = `|N|`,
    /// 5 = `|W|`, 15 = `wp_max_error`) inside [`pre_quantize`].
    ///
    /// The default sort-quantile path picks bucket edges by taking every
    /// `N/k`-th element of the sorted unique-values list — uniform in
    /// rank, which over-quantises the dense low-energy regime that most
    /// tree leaves end up routing through (residual prediction is good
    /// → energy values cluster near zero). Lloyd-Max iterates centroid
    /// vs cell-boundary updates to minimise total quantisation error
    /// weighted by sample frequency, so split candidates concentrate
    /// where the samples actually live.
    ///
    /// This is a spec-legal reinterpretation of EX-J5 (CALIC energy-
    /// quantized context, Golchin & Paliwal 1998). The original proposal
    /// adds a 17th MA-tree property index for an energy bin — JXL hard-
    /// codes `kNumNonrefProperties = 16` (`context_predict.h:378-379`,
    /// jxl-rs `tree.rs:197`), so any `property_idx >= 16` is interpreted
    /// as a (nonexistent) reference-channel property by decoders.
    /// Refining the candidate **bucket boundaries** of the existing
    /// energy proxies preserves the spec, changes only encoder-side
    /// candidate splitvals, and captures the same "give the tree learner
    /// better energy-aware thresholds" intent.
    ///
    /// Only properties 4, 5, and 15 — the documented residual-energy
    /// proxies in the JXL property set — are refined. The other 13
    /// properties keep the cheap sort-quantile path because their
    /// distributions are not energy-shaped (channel id, group id,
    /// signed gradient differences whose distributions are roughly
    /// symmetric around zero).
    ///
    /// Bitstream-affecting (different candidate thresholds change the
    /// tree learner's chosen splitvals), but spec-legal. Hash-lock
    /// fixtures must be re-baked when this flag is flipped on by
    /// default.
    ///
    /// Default `false` (sort-quantile path). Callers opt in via
    /// [`crate::api::LosslessConfig`] `__expert` overrides
    /// ([`crate::effort::LosslessInternalParams::lloyd_max_buckets`]).
    pub lloyd_max_buckets: bool,
}

impl TreeLearningParams {
    /// Create tree learning parameters from an [`EffortProfile`].
    ///
    /// Reads `tree_num_properties`, `tree_max_buckets`, and `tree_threshold_base`
    /// from the profile instead of computing them from effort inline.
    pub fn from_profile(profile: &crate::effort::EffortProfile) -> Self {
        Self::from_profile_impl(profile, false)
    }

    /// Create tree learning parameters for squeeze mode.
    ///
    /// Uses squeeze-specific property order (matching libjxl enc_modular.cc:538-541)
    /// which prioritizes spatial correlation properties over gradient-difference ones.
    pub fn from_profile_squeeze(profile: &crate::effort::EffortProfile) -> Self {
        Self::from_profile_impl(profile, true)
    }

    fn from_profile_impl(profile: &crate::effort::EffortProfile, is_squeeze: bool) -> Self {
        let order = if is_squeeze {
            // Squeeze always includes group_id (libjxl enc_modular.cc:538-541).
            PROP_ORDER_SQUEEZE
        } else if profile.effort >= 9 {
            // At effort 9+ (speed_tier <= kTortoise), libjxl keeps group_id.
            PROP_ORDER_NO_SQUEEZE
        } else {
            // At effort < 9 for lossless with <30 streams, libjxl erases group_id
            // (enc_modular.cc:546-549). This is our typical single-group case.
            PROP_ORDER_NO_SQUEEZE_NO_GID
        };
        // Surface caller misconfiguration loudly. The `__expert` setter on
        // LosslessConfig accepts any u8; previously the over-bound case
        // silently clamped here. Runtime validation lives in
        // `validate()` (opt-in); this debug_assert catches misconfigured
        // sweep harnesses early during testing without panicking release
        // builds (the `.min(order.len())` clamp below remains as a safety
        // net so we never panic on out-of-bounds slice access).
        debug_assert!(
            (profile.tree_num_properties as usize) <= order.len(),
            "tree_num_properties = {} exceeds property-order length {} \
             for {}; clamp here is hiding a misconfigured \
             LosslessInternalParams. Validate via LosslessConfig::validate().",
            profile.tree_num_properties,
            order.len(),
            if is_squeeze { "squeeze" } else { "no-squeeze" },
        );
        let num_props = (profile.tree_num_properties as usize).min(order.len());

        Self {
            properties: order[..num_props].to_vec(),
            max_property_values: profile.tree_max_buckets as usize,
            split_threshold: profile.tree_threshold_base as f64,
            // kMaxTreeSize from libjxl ma_common.h:24 — absolute decoder cap.
            // with_total_pixels() further tightens this to the per-frame limit.
            max_nodes: 1 << 22,
            pixel_fraction: 1.0,
            use_streaming_dedup: profile.use_streaming_dedup,
            gather_dedup: profile.gather_dedup,
            gather_dedup_phase3: profile.gather_dedup_phase3,
            parallel_max_depth: profile.tree_parallel_max_depth,
            parallel_recursion_floor: profile.tree_parallel_floor,
            parallel_root_threshold: profile.tree_parallel_root_threshold,
            parallel_small_image_fallback: profile.tree_parallel_small_image_fallback,
            lloyd_max_buckets: profile.lloyd_max_buckets,
        }
    }

    /// Create tree learning parameters for the given effort level (test use only).
    ///
    /// Production code should use [`from_profile`](Self::from_profile) instead.
    #[cfg(test)]
    pub fn for_effort(effort: u8) -> Self {
        // Match libjxl: e9+ keeps group_id, e<9 erases it for lossless with <30 streams.
        let order = if effort >= 9 {
            PROP_ORDER_NO_SQUEEZE
        } else {
            PROP_ORDER_NO_SQUEEZE_NO_GID
        };
        let speed_tier = 10u8.saturating_sub(effort);
        let (num_props, max_property_values) = match effort {
            0..=4 => (3, 32),
            5 => (4, 48),
            6 => (5, 64),
            7 => (7, 96),
            8 => (10, 128),
            _ => (order.len(), 256),
        };
        let threshold_base = 75.0 + 14.0 * speed_tier as f64;
        let num_props = num_props.min(order.len());

        // Match `lossless_reference` schedule: e>=8 takes the deeper /
        // lower-floor parallel knobs; e<=7 keeps the e7-tuned values.
        let (parallel_max_depth, parallel_recursion_floor, parallel_root_threshold) = if effort >= 8
        {
            (5u32, 8_192usize, 4_096usize)
        } else {
            (4u32, 16_384usize, 8_192usize)
        };

        Self {
            properties: order[..num_props].to_vec(),
            max_property_values,
            split_threshold: threshold_base,
            max_nodes: 1 << 22,
            pixel_fraction: 1.0,
            use_streaming_dedup: false,
            gather_dedup: false,
            gather_dedup_phase3: false,
            parallel_max_depth,
            parallel_recursion_floor,
            parallel_root_threshold,
            parallel_small_image_fallback: false,
            lloyd_max_buckets: false,
        }
    }

    /// Set the pixel fraction (num_samples / total_pixels) for threshold scaling.
    /// This matches libjxl's `required_cost = pixel_fraction * 0.9 + 0.1`.
    #[must_use]
    pub fn with_pixel_fraction(mut self, fraction: f64) -> Self {
        self.pixel_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Cap max_nodes to the decoder's per-frame tree size limit.
    /// Formula from libjxl encoding.cc:606-616 (decoder side):
    ///   `min(1<<20, 1024 + sum_of_channel_pixels)`
    /// Then capped at `kMaxTreeSize = 1<<22` in dec_ma.cc:141.
    /// `total_pixels` should be `sum(channel.w * channel.h)` for all encoded channels.
    #[must_use]
    pub fn with_total_pixels(mut self, total_pixels: usize) -> Self {
        let decoder_limit = (1024 + total_pixels).min(1 << 20);
        self.max_nodes = self.max_nodes.min(decoder_limit);
        self
    }

    /// Append reference channel property indices to the property list.
    ///
    /// Matches libjxl enc_modular.cc:593-605:
    /// - At effort < 9 (speed > Tortoise): only the gradient residual property
    ///   per ref channel (`kNumNonrefProperties + i*4 + 3`)
    /// - At effort 9+ (Tortoise): all 4 properties per ref channel
    ///
    /// `num_ref_channels` is the maximum number of reference channels across
    /// all channels in the image (typically `num_color_channels - 1` for RGB).
    #[must_use]
    pub fn with_ref_properties(mut self, num_ref_channels: usize, effort: u8) -> Self {
        if num_ref_channels == 0 {
            return self;
        }
        if effort >= 9 {
            // Tortoise: all 4 properties per ref channel
            for i in 0..num_ref_channels * 4 {
                self.properties.push(NUM_PROPERTIES + i);
            }
        } else {
            // Non-Tortoise: only gradient residual (property offset 3) per ref channel
            for i in 0..num_ref_channels {
                self.properties.push(NUM_PROPERTIES + i * 4 + 3);
            }
        }
        self
    }
}

/// Per-column-family heap accounting for a [`TreeSamples`], in bytes.
///
/// Diagnostic support for the encode peak-memory work: `TreeSamples` is the
/// dominant lossless allocation at effort >= 7, and its cost splits across
/// three families with very different reduction options, so a single total
/// would not say which one to attack.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TreeSamplesHeapBytes {
    pub num_samples: usize,
    pub residual_tokens: usize,
    pub extra_bits: usize,
    pub props: usize,
    pub sample_counts: usize,
    pub num_prop_columns: usize,
    pub num_pred_columns: usize,
}

impl TreeSamplesHeapBytes {
    pub fn total(&self) -> usize {
        self.residual_tokens + self.extra_bits + self.props + self.sample_counts
    }
}

/// Collected samples for tree learning.
pub struct TreeSamples {
    /// Number of samples collected.
    pub num_samples: usize,
    /// Candidate predictor list. Full 14 predictors for normal mode,
    /// just `[Zero]` for squeeze mode (matching libjxl enc_modular.cc:629-633).
    candidate_predictors: &'static [Predictor],
    /// Residual token per predictor: residual_tokens[predictor_idx][sample_idx].
    /// Tokens fit in u8 (max ~55 for HybridUint {4,2,0} on 8-bit data).
    residual_tokens: Vec<Vec<u8>>,
    /// Extra bits per predictor: extra_bits[predictor_idx][sample_idx].
    /// These are the HybridUint extra bits (non-token part), matching libjxl's ResidualToken.nbits.
    /// Fits in u8 (max ~14 bits for 8-bit image residuals).
    extra_bits: Vec<Vec<u8>>,
    /// Spec-matching property values: props[property_idx][sample_idx].
    /// These are the actual (unquantized) property values.
    /// Length is `NUM_PROPERTIES + 4 * num_ref_channels` (base 16 + 4 per ref channel).
    props: Vec<Vec<i32>>,
    /// Sample counts after deduplication: sample_counts[sample_idx].
    /// Before dedup, all 1s. After dedup, each unique sample's count of merged originals.
    sample_counts: Vec<u32>,
    /// Maximum number of reference channels across all channels in the image.
    /// 0 for squeeze mode or single-channel images.
    num_ref_channels: usize,
    /// When true, [`gather_channel_samples`] draws randomized (de-aliased)
    /// sample gaps for THIS gather regardless of the `JXL_TREE_SAMPLE_RANDOM`
    /// env default. Set by the cost-based tree self-repair (task #14, the
    /// `JXL_TREE_SELF_REPAIR` path in `section.rs`) for its second, de-aliased
    /// re-gather. Only read during gather — its value in split/merge results is
    /// irrelevant. Default `false` ⇒ env-driven.
    pub(crate) randomize_gather: bool,
}

impl Default for TreeSamples {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeSamples {
    /// Creates an empty TreeSamples structure with full 14-predictor candidate list
    /// and no reference channel properties.
    pub fn new() -> Self {
        Self::with_predictors_and_refs(CANDIDATE_PREDICTORS, 0)
    }

    /// Heap bytes currently *reserved* by the SoA columns, broken out per
    /// column family. Uses `capacity()` (not `len()`) because the reserve
    /// above deliberately over-allocates — the reserved bytes are what the
    /// process actually pays for in RSS.
    ///
    /// Diagnostic only; nothing in the encode path branches on it. Printed by
    /// `JXL_TREE_SAMPLES_STATS=1` at the end of each gather so the peak-memory
    /// work has a measured breakdown instead of an estimate.
    pub(crate) fn heap_bytes(&self) -> TreeSamplesHeapBytes {
        let sum = |vs: &[Vec<u8>]| -> usize { vs.iter().map(Vec::capacity).sum() };
        TreeSamplesHeapBytes {
            num_samples: self.num_samples,
            residual_tokens: sum(&self.residual_tokens),
            extra_bits: sum(&self.extra_bits),
            props: self.props.iter().map(|v| v.capacity() * 4).sum(),
            sample_counts: self.sample_counts.capacity() * 4,
            num_prop_columns: self.props.len(),
            num_pred_columns: self.residual_tokens.len(),
        }
    }

    /// Reserve capacity for `additional` more samples in every SoA
    /// column. Perf (/goal hunt): without this, the per-row bulk extends
    /// grow each of the ~60 columns through doubling reallocs — libc
    /// `__memmove` was 10.9 % of CPU on terminal — re-copying ~2× the
    /// final bytes per column. Capacity is observationally invisible:
    /// byte-identical.
    pub(crate) fn reserve_additional(&mut self, additional: usize) {
        for v in &mut self.residual_tokens {
            v.reserve(additional);
        }
        for v in &mut self.extra_bits {
            v.reserve(additional);
        }
        for v in &mut self.props {
            v.reserve(additional);
        }
    }

    /// Creates an empty TreeSamples with reference channel properties.
    ///
    /// `num_ref_channels` is the maximum number of reference channels across all
    /// channels in the image. For RGB with no extra channels, this is 2
    /// (channel 1 can reference channel 0, channel 2 can reference 0 and 1).
    pub fn new_with_ref_channels(num_ref_channels: usize) -> Self {
        Self::with_predictors_and_refs(CANDIDATE_PREDICTORS, num_ref_channels)
    }

    /// Creates an empty TreeSamples for squeeze mode (Zero predictor only).
    /// Matches libjxl enc_modular.cc:629-633.
    /// No reference channels: squeeze creates channels with different dimensions.
    pub fn new_for_squeeze() -> Self {
        Self::with_predictors_and_refs(CANDIDATE_PREDICTORS_SQUEEZE, 0)
    }

    /// Creates an empty TreeSamples whose candidate-predictor list is the
    /// chunk-4 permutation for `seed` (RFC#45). Seed 0 is identical to
    /// [`Self::new_with_ref_channels`] and therefore preserves the e ≤ 9
    /// byte-identical hash-locks (e ≤ 9 has `tree_learn_seeds = 1` so the
    /// canonical seed-0 path is the only path taken).
    ///
    /// Different permutations produce different trees only when
    /// [`find_best_predictor`]'s strict-`<` tie-break flips at equal
    /// histogram entropies — typically on flat / synthetic regions where
    /// several cheap predictors share the same residual distribution. On
    /// photographic content the bytes-saving signal is small per cell
    /// but cumulative across 1 to 8 K-node trees.
    pub fn new_with_predictor_order_for_seed(num_ref_channels: usize, seed: u64) -> Self {
        let order = derive_seeded_predictor_order(seed);
        Self::with_predictors_and_refs(order, num_ref_channels)
    }

    /// Creates an empty TreeSamples with a custom predictor list and ref channel count.
    fn with_predictors_and_refs(predictors: &'static [Predictor], num_ref_channels: usize) -> Self {
        let num_predictors = predictors.len();
        let total_props = NUM_PROPERTIES + 4 * num_ref_channels;
        Self {
            num_samples: 0,
            candidate_predictors: predictors,
            residual_tokens: vec![Vec::new(); num_predictors],
            extra_bits: vec![Vec::new(); num_predictors],
            props: vec![Vec::new(); total_props],
            sample_counts: Vec::new(),
            num_ref_channels,
            randomize_gather: false,
        }
    }

    /// Returns the total number of properties (base 16 + 4 per ref channel).
    pub fn total_num_properties(&self) -> usize {
        NUM_PROPERTIES + 4 * self.num_ref_channels
    }

    /// Returns the number of candidate predictors.
    pub fn num_predictors(&self) -> usize {
        self.candidate_predictors.len()
    }

    /// Total gathered weight across all samples. Equals `num_samples`
    /// when no dedup has run yet (sample_counts empty), otherwise the
    /// sum of per-row counts. Used by `pixel_fraction` callers so the
    /// threshold scaling stays correct when gather-time dedup
    /// (Phase 2 of issue #41) is enabled.
    pub(crate) fn total_gathered_weight(&self) -> usize {
        if self.sample_counts.is_empty() {
            self.num_samples
        } else {
            self.sample_counts.iter().map(|&c| c as usize).sum()
        }
    }

    /// Reserve capacity in all parallel SoA arrays for `additional` more samples.
    ///
    /// Optional micro-optimization for callers that know the total sample count
    /// up-front — avoids `Vec` reallocations during the gather hot loop.
    pub(crate) fn reserve(&mut self, additional: usize) {
        for v in &mut self.residual_tokens {
            v.reserve(additional);
        }
        for v in &mut self.extra_bits {
            v.reserve(additional);
        }
        for v in &mut self.props {
            v.reserve(additional);
        }
    }

    /// Release the raw `props` columns, which are the largest thing this
    /// struct owns (24 i32 columns = 96 of the ~124 B/sample; 1152 MiB of a
    /// 4K/e9 encode).
    ///
    /// Only legal once `props` is provably dead. On the lossless main path
    /// (`compute_best_tree_with_budget`) it is: that path consumes `props`
    /// exactly twice — in `pre_quantize`, which projects it into
    /// `PreQuantizedProps::bucket_indices`, and in `dedup_samples`, which
    /// gathers compact — and thereafter partitions with
    /// `PartitionKey::Bucket` only, reading `bucket_indices` rather than
    /// `props`. That is the same precondition
    /// [`tree_learn_split::SplittableSamples::skip_props_swap`] documents, and
    /// the same call sites already pass `skip_props_swap = true`. `swap_rows`
    /// explicitly tolerates empty parallel arrays, so the emptied columns stay
    /// well-formed for the rest of the tree build.
    ///
    /// The multipliers path (`compute_best_tree_with_multipliers`) must NOT
    /// call this — its static-prop axes use `PartitionKey::Property`, which
    /// reads `props` directly.
    pub(crate) fn free_props(&mut self) {
        for v in &mut self.props {
            // Both are needed: `clear` drops the length, `shrink_to_fit`
            // returns the capacity. Only the second gives the memory back.
            v.clear();
            v.shrink_to_fit();
        }
    }

    /// Size every SoA column ONCE to an exact upper bound on the final sample
    /// count, so no column ever reallocates during the gather/merge.
    ///
    /// This is the allocator-agnostic half of the peak-memory work. `reserve`
    /// (amortized) grows a column by reallocating: the allocator must hold the
    /// old and new buffers simultaneously, so a 48 MiB props column costs a
    /// ~96 MiB transient and leaves a 48 MiB hole behind. Across 52 columns and
    /// one growth per merge step that transient overshoot — and the resulting
    /// churn of large freed blocks — is what dominates the peak, and it does so
    /// on EVERY allocator (glibc, macOS libmalloc, jemalloc, mimalloc all pay
    /// the copy; they differ only in how much of the hole they hand back).
    /// `reserve_exact` against a known upper bound removes the growth entirely:
    /// peak becomes exactly the data size.
    ///
    /// `upper_bound` must be >= the total samples ultimately appended. The
    /// gather's per-channel `ceil(w*h / stride)` is such a bound (dedup and
    /// skipped pixels only ever reduce the count), so over-estimating is safe
    /// and merely leaves unused tail capacity.
    pub(crate) fn reserve_exact_total(&mut self, upper_bound: usize) {
        for v in &mut self.residual_tokens {
            v.reserve_exact(upper_bound.saturating_sub(v.len()));
        }
        for v in &mut self.extra_bits {
            v.reserve_exact(upper_bound.saturating_sub(v.len()));
        }
        for v in &mut self.props {
            v.reserve_exact(upper_bound.saturating_sub(v.len()));
        }
    }

    /// Append all samples from `other` into `self`. Both must have the same
    /// predictor list and reference-channel count (the gather call site
    /// guarantees this; we debug-assert it).
    ///
    /// Used by parallel gather: each task builds an isolated `TreeSamples`,
    /// then the main thread merges them in deterministic index order. Concat
    /// is the right merge because gather happens BEFORE dedup, so
    /// `sample_counts` is still empty and the parallel SoA arrays are simply
    /// extended.
    pub(crate) fn append_from(&mut self, mut other: TreeSamples) {
        debug_assert_eq!(self.num_ref_channels, other.num_ref_channels);
        debug_assert_eq!(
            self.candidate_predictors.len(),
            other.candidate_predictors.len()
        );
        // sample_counts may be populated by gather-time dedup (Phase 2 of
        // issue #41). Both sides must agree: either both empty (no
        // gather dedup) or both lengths equal to their respective
        // num_samples. Concatenating mixed regimes would desync the
        // weight array vs the SoA columns.
        debug_assert!(
            (self.sample_counts.is_empty() && other.sample_counts.is_empty())
                || (self.sample_counts.len() == self.num_samples
                    && other.sample_counts.len() == other.num_samples),
            "TreeSamples::append_from: sample_counts mismatch — left has {}/{}, right has {}/{}",
            self.sample_counts.len(),
            self.num_samples,
            other.sample_counts.len(),
            other.num_samples,
        );
        for (dst, src) in self
            .residual_tokens
            .iter_mut()
            .zip(other.residual_tokens.iter_mut())
        {
            dst.append(src);
        }
        for (dst, src) in self.extra_bits.iter_mut().zip(other.extra_bits.iter_mut()) {
            dst.append(src);
        }
        for (dst, src) in self.props.iter_mut().zip(other.props.iter_mut()) {
            dst.append(src);
        }
        self.sample_counts.append(&mut other.sample_counts);
        self.num_samples += other.num_samples;
    }

    /// Pre-quantize all property values into bucket indices.
    /// This is done once before tree building, replacing per-node binary_search
    /// and threshold_set allocation with a single upfront pass.
    ///
    /// When [`TreeLearningParams::lloyd_max_buckets`] is set, properties
    /// 4 (`|N|`), 5 (`|W|`), and 15 (`wp_max_error`) — the residual-
    /// energy proxies — use Lloyd-Max iterative clustering for bucket
    /// boundaries instead of sort-quantile picks. See
    /// [`lloyd_max_thresholds`].
    fn pre_quantize(&self, params: &TreeLearningParams) -> PreQuantizedProps {
        let max_buckets = params.max_property_values;
        let n = self.num_samples;
        let total_props = self.total_num_properties();
        let mut threshold_sets = vec![Vec::new(); total_props];
        let mut bucket_indices = vec![Vec::new(); total_props];

        // Per-property pre-quantization is independent: each prop_idx reads
        // its own `self.props[prop_idx]` slice and writes its own slot in
        // `threshold_sets[prop_idx]` + `bucket_indices[prop_idx]`. Properties
        // not in `params.properties` get an empty slot (already initialized
        // to `Vec::new()` above), so we only fan out across the requested
        // property list and stitch results back into the right slots.
        //
        // At effort 7 this fans out over 7 properties (one per tree-learning
        // candidate), with per-prop work O(n) for n up to ~1.5M samples on
        // 4.19 MP — each task is large enough to amortize rayon spawn cost.
        let per_prop: Vec<(Vec<i32>, Vec<u8>)> =
            crate::parallel::parallel_map(params.properties.len(), |i| {
                let prop_idx = params.properties[i];
                let props = &self.props[prop_idx];

                // Find min/max across ALL samples
                let mut min_val = i32::MAX;
                let mut max_val = i32::MIN;
                for &v in &props[..n] {
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }
                if min_val == max_val {
                    // Constant property — empty threshold set, all bucket 0
                    return (Vec::new(), vec![0u8; n]);
                }

                // EX-J5 reinterpretation: Lloyd-Max bucket boundaries for the
                // three residual-energy proxy properties (4 = |N|, 5 = |W|,
                // 15 = wp_max_error). Same property indices, same bitstream
                // format — just better candidate splitvals derived from the
                // empirical sample distribution rather than uniform rank.
                //
                // Other 13 properties keep the cheap sort-quantile path: their
                // distributions are not energy-shaped (channel id, group id,
                // signed gradient differences ~symmetric around zero), so
                // Lloyd-Max would only add cost without compression payoff.
                if params.lloyd_max_buckets && (prop_idx == 4 || prop_idx == 5 || prop_idx == 15) {
                    let ts = lloyd_max_thresholds(&props[..n], min_val, max_val, max_buckets);
                    if ts.is_empty() {
                        return (Vec::new(), vec![0u8; n]);
                    }
                    let num_thresholds = ts.len();
                    let mut bi = vec![0u8; n];
                    for (bi_val, &v) in bi.iter_mut().zip(props[..n].iter()) {
                        let bucket = match ts.binary_search(&v) {
                            Ok(pos) => pos,
                            Err(pos) => pos,
                        };
                        *bi_val = bucket.min(num_thresholds) as u8;
                    }
                    return (ts, bi);
                }

                // Build threshold set from unique values
                let range = max_val as i64 - min_val as i64 + 1;
                let ts: Vec<i32>;

                if range <= (max_buckets * 4) as i64 {
                    let range_usize = range as usize;
                    let mut present = vec![false; range_usize];
                    for i in 0..n {
                        present[(props[i] - min_val) as usize] = true;
                    }
                    let mut unique_vals: Vec<i32> = present
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| **p)
                        .map(|(i, _)| min_val + i as i32)
                        .collect();
                    if unique_vals.len() <= 1 {
                        return (Vec::new(), vec![0u8; n]);
                    }
                    unique_vals.pop();
                    ts = if unique_vals.len() <= max_buckets {
                        unique_vals
                    } else {
                        let step = unique_vals.len().div_ceil(max_buckets);
                        unique_vals
                            .iter()
                            .step_by(step.max(1))
                            .take(max_buckets)
                            .copied()
                            .collect()
                    };
                } else {
                    let mut sample_vals: Vec<i32> = props[..n].to_vec();
                    sample_vals.sort_unstable();
                    sample_vals.dedup();
                    if sample_vals.len() <= 1 {
                        return (Vec::new(), vec![0u8; n]);
                    }
                    sample_vals.pop();
                    ts = if sample_vals.len() <= max_buckets {
                        sample_vals
                    } else {
                        let step = sample_vals.len() / max_buckets;
                        sample_vals
                            .iter()
                            .step_by(step.max(1))
                            .take(max_buckets)
                            .copied()
                            .collect()
                    };
                }

                // Assign each sample to a bucket using binary search
                let num_thresholds = ts.len();
                let mut bi = vec![0u8; n];
                for (bi_val, &v) in bi.iter_mut().zip(props[..n].iter()) {
                    let bucket = match ts.binary_search(&v) {
                        Ok(pos) => pos,
                        Err(pos) => {
                            if pos == 0 {
                                0
                            } else {
                                pos
                            }
                        }
                    };
                    *bi_val = bucket.min(num_thresholds) as u8;
                }

                (ts, bi)
            });

        // Stitch per-property results back into the global slots. Properties
        // not in `params.properties` remain empty `Vec::new()`, matching the
        // pre-parallel behavior.
        for (i, (ts, bi)) in per_prop.into_iter().enumerate() {
            let prop_idx = params.properties[i];
            threshold_sets[prop_idx] = ts;
            bucket_indices[prop_idx] = bi;
        }

        PreQuantizedProps {
            threshold_sets,
            bucket_indices,
        }
    }
}

/// Choose `<= max_buckets` threshold values for `samples` using **Lloyd-Max
/// iterative clustering** instead of sort-quantile picks.
///
/// Lloyd-Max minimises the total mean-square quantisation error of the
/// samples by alternating:
///
/// 1. **Cell assignment** — each sample is assigned to the nearest centroid.
///    The threshold between centroid `i` and `i+1` is the midpoint
///    `(c_i + c_{i+1}) / 2`.
/// 2. **Centroid update** — each centroid is replaced by the (weighted) mean
///    of the samples in its cell.
///
/// For the JXL MA-tree pre-quantisation use case the algorithm operates on
/// the empirical histogram of `samples` rather than the raw sample list, so
/// runtime is O(num_unique × iters × k) instead of O(n × iters × k). At our
/// 50 %-pixel sampling rate (≈65 k–1.5 M samples) and the per-property
/// dynamic range (~10²–10⁴ unique values), this is well under 1 ms per
/// property on a single core.
///
/// **Initialisation** — `k`-quantile picks over the sorted unique values.
/// This gives Lloyd-Max a balanced starting partition that converges in
/// 3–5 iterations on every property distribution observed in CID22 / CLIC
/// corpora.
///
/// **Returns** — strictly-increasing list of bucket edges (one fewer than
/// the resulting bucket count), ready to be used with `binary_search`. The
/// list is empty when there are no actionable buckets (constant property,
/// single unique value).
///
/// **Spec note** — these thresholds drive only encoder-side candidate split
/// selection inside the MA-tree learner; the decoder reads whatever
/// `splitval` the tree node ends up encoding, regardless of how it was
/// chosen. Lloyd-Max-derived thresholds are 100 % spec-legal.
fn lloyd_max_thresholds(
    samples: &[i32],
    min_val: i32,
    max_val: i32,
    max_buckets: usize,
) -> Vec<i32> {
    // Build empirical histogram. Range fits in (max_val - min_val + 1)
    // buckets; for the energy properties this is typically <= 4096 entries
    // (8-bit |N| / |W|) or <= 2*wp_max_error range (~512 entries).
    let range = (max_val as i64 - min_val as i64 + 1) as usize;
    let mut hist = vec![0u32; range];
    for &v in samples {
        hist[(v - min_val) as usize] += 1;
    }

    // Compact histogram → unique values + counts.
    let mut unique_vals: Vec<i32> = Vec::with_capacity(range);
    let mut counts: Vec<u32> = Vec::with_capacity(range);
    for (i, &c) in hist.iter().enumerate() {
        if c != 0 {
            unique_vals.push(min_val + i as i32);
            counts.push(c);
        }
    }

    let num_unique = unique_vals.len();
    if num_unique <= 1 {
        return Vec::new();
    }

    // libjxl's threshold set is `max_buckets` entries → at most
    // `max_buckets + 1` bucket cells. Cap `k` at `min(max_buckets + 1, num_unique)`
    // so we don't create more clusters than there are distinct values.
    let k = (max_buckets + 1).min(num_unique);
    if k <= 1 {
        return Vec::new();
    }

    // Initialise centroids by k-quantile picks over the **count-weighted**
    // cumulative distribution. This ensures each starting centroid covers
    // roughly equal sample mass, giving Lloyd-Max a near-converged start
    // on energy distributions that are heavily concentrated near zero.
    let total_count: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut centroids: Vec<f64> = Vec::with_capacity(k);
    let mut next_target = total_count / (k as u64 * 2).max(1);
    let step = (total_count / k as u64).max(1);
    let mut cum: u64 = 0;
    let mut picked = 0usize;
    for j in 0..num_unique {
        cum += counts[j] as u64;
        while picked < k && cum >= next_target {
            centroids.push(unique_vals[j] as f64);
            picked += 1;
            next_target = next_target.saturating_add(step);
        }
        if picked == k {
            break;
        }
    }
    // Fill any shortfall (rare; happens only when cumulative-mass picks
    // fall short of k due to integer rounding in the step / next_target).
    while centroids.len() < k {
        centroids.push(unique_vals[num_unique - 1] as f64);
    }
    // Strictly-increasing centroids are required for the midpoint
    // partitioning to be monotone. Deduplicate by nudging successors up by
    // 1 ULP-equivalent (one input unit) when initial picks collide on the
    // same unique value.
    for i in 1..k {
        if centroids[i] <= centroids[i - 1] {
            centroids[i] = centroids[i - 1] + 1.0;
        }
    }

    // Lloyd-Max iterations. Convergence criterion: centroid movement below
    // 0.5 input units (sub-quantisation-step) OR max 8 iterations. Empirical
    // convergence on CID22 photos at e7 is 3–5 iters.
    const MAX_ITERS: usize = 8;
    const CONVERGENCE_EPS: f64 = 0.5;
    let mut new_centroids = vec![0.0f64; k];
    let mut sums = vec![0.0f64; k];
    let mut weights = vec![0.0f64; k];

    for _iter in 0..MAX_ITERS {
        // Build cell boundaries (midpoints between consecutive centroids).
        // boundaries[i] = midpoint between centroid i and centroid i+1.
        // A value v belongs to cell i iff v >= boundaries[i-1] (for i>0)
        // and v < boundaries[i] (for i<k-1).
        let mut boundaries = Vec::with_capacity(k - 1);
        for i in 0..k - 1 {
            boundaries.push((centroids[i] + centroids[i + 1]) * 0.5);
        }

        // Reset accumulators.
        weights.fill(0.0);
        sums.fill(0.0);

        // Assign each unique value to its nearest centroid and accumulate
        // (count, count*value) sums. The boundaries are sorted, so we walk
        // unique values in order and bump the cell index when we cross a
        // boundary. This is O(num_unique + k) per iteration.
        let mut cell = 0usize;
        for j in 0..num_unique {
            let v = unique_vals[j] as f64;
            while cell + 1 < k && v >= boundaries[cell] {
                cell += 1;
            }
            let w = counts[j] as f64;
            weights[cell] += w;
            sums[cell] += w * v;
        }

        // Update centroids: weighted mean of each cell. Empty cells keep
        // their previous centroid (rare but possible at extreme distributions).
        for i in 0..k {
            new_centroids[i] = if weights[i] > 0.0 {
                sums[i] / weights[i]
            } else {
                centroids[i]
            };
        }

        // Enforce strictly-increasing centroids after the mean update.
        // Empty cells can produce duplicates; bump duplicates by 1 unit.
        for i in 1..k {
            if new_centroids[i] <= new_centroids[i - 1] {
                new_centroids[i] = new_centroids[i - 1] + 1.0;
            }
        }

        // Convergence check.
        let mut max_move = 0.0f64;
        for i in 0..k {
            let d = (new_centroids[i] - centroids[i]).abs();
            if d > max_move {
                max_move = d;
            }
        }
        core::mem::swap(&mut centroids, &mut new_centroids);
        if max_move < CONVERGENCE_EPS {
            break;
        }
    }

    // Emit final thresholds as integer midpoints between consecutive
    // centroids. Decoder splitvals are i32, and the MA-tree binary_search
    // requires strictly increasing entries.
    //
    // Like the sort-quantile path, the final threshold set has at most
    // `max_buckets` entries (k-1 midpoints when k=max_buckets+1).
    let mut thresholds: Vec<i32> = Vec::with_capacity(k - 1);
    for i in 0..k - 1 {
        // Midpoint rounded to nearest integer; ties round half-to-even via
        // f64::round_ties_even (no Rust 1.85 stability concern: round on f64
        // rounds half-away-from-zero which is also acceptable here, but
        // ties-to-even matches our other tree-learning rounding choices).
        let mid = (centroids[i] + centroids[i + 1]) * 0.5;
        let edge = mid.round_ties_even() as i32;
        // Maintain strict monotonicity. If rounding collapsed two midpoints
        // to the same i32, bump up by 1 (the bucket would have been empty
        // anyway, but binary_search requires sorted-unique input).
        if let Some(&prev) = thresholds.last()
            && edge <= prev
        {
            thresholds.push(prev + 1);
            continue;
        }
        thresholds.push(edge);
    }

    // Clamp thresholds to the actual data range. Edges outside [min_val,
    // max_val) are degenerate (empty buckets); trimming them avoids
    // wasting splitval-encoding bits.
    thresholds.retain(|&t| t > min_val && t <= max_val);
    thresholds
}

/// Find reference channels for a given channel in a modular image.
///
/// A reference channel is any preceding channel (j < i) with matching
/// `(width, height, hshift, vshift)`. Matches libjxl's `PrecomputeReferences`
/// in `context_predict.h:411-443`.
///
/// Returns indices of matching channels in the image's channel list.
fn find_ref_channels(image: &ModularImage, channel_idx: usize) -> Vec<usize> {
    if channel_idx == 0 {
        return Vec::new();
    }
    let ch = &image.channels[channel_idx];
    let w = ch.width();
    let h = ch.height();
    let hs = ch.hshift;
    let vs = ch.vshift;

    let mut refs = Vec::new();
    for j in (0..channel_idx).rev() {
        let ref_ch = &image.channels[j];
        if ref_ch.width() == w && ref_ch.height() == h && ref_ch.hshift == hs && ref_ch.vshift == vs
        {
            refs.push(j);
        }
    }
    // refs[0] = closest preceding channel (j = channel_idx-1), matching decoder's
    // PrecomputeReferences which iterates backward from channel_idx-1 to 0.
    refs
}

/// Compute the maximum number of reference channels across all channels.
///
/// This determines how many extra property slots (4 per ref channel) are needed
/// in the TreeSamples structure.
pub fn max_ref_channels(image: &ModularImage) -> usize {
    let mut max_refs = 0;
    for i in 0..image.channels.len() {
        let refs = find_ref_channels(image, i);
        max_refs = max_refs.max(refs.len());
    }
    max_refs
}

/// Compute the 16 spec-matching properties for a pixel.
///
/// These match jxl-rs decoder's `compute_properties()` exactly:
///   [0] = channel, [1] = group_id, [2] = y, [3] = x,
///   [4] = |N|, [5] = |W|, [6] = N, [7] = W,
///   [8] = W - prev_gradient, [9] = W + N - NW,
///   [10] = W - NW, [11] = NW - N, [12] = N - NE,
///   [13] = N - NN, [14] = W - WW, [15] = wp_max_error
#[inline]
fn compute_spec_properties(
    channel_idx: u32,
    group_id: u32,
    x: usize,
    y: usize,
    n: &Neighbors,
    prev_gradient: i32,
    wp_max_error: i32,
) -> [i32; NUM_PROPERTIES] {
    let mut props = [0i32; NUM_PROPERTIES];
    compute_spec_properties_into(
        &mut props,
        channel_idx,
        group_id,
        x,
        y,
        n,
        prev_gradient,
        wp_max_error,
    );
    props
}

/// [`compute_spec_properties`] writing into a caller-provided buffer —
/// lets `collect_residuals_with_tree*` fill the prefix of its extended
/// property buffer directly instead of copying a returned array per
/// pixel (the copy was a measured hot spot,
/// `benchmarks/perf_gather_profile_2026-06-10.meta` addendum).
#[allow(clippy::too_many_arguments)]
#[inline]
fn compute_spec_properties_into(
    props: &mut [i32; NUM_PROPERTIES],
    channel_idx: u32,
    group_id: u32,
    x: usize,
    y: usize,
    n: &Neighbors,
    prev_gradient: i32,
    wp_max_error: i32,
) {
    props[0] = channel_idx as i32;
    props[1] = group_id as i32;
    props[2] = y as i32;
    props[3] = x as i32;
    props[4] = n.n.wrapping_abs();
    props[5] = n.w.wrapping_abs();
    props[6] = n.n;
    props[7] = n.w;
    // Property 8 is the delta from the previous gradient value (stored in property 9)
    let gradient = n.w.wrapping_add(n.n).wrapping_sub(n.nw);
    props[8] = n.w.wrapping_sub(prev_gradient);
    props[9] = gradient;
    props[10] = n.w.wrapping_sub(n.nw);
    props[11] = n.nw.wrapping_sub(n.n);
    props[12] = n.n.wrapping_sub(n.ne);
    props[13] = n.n.wrapping_sub(n.nn);
    props[14] = n.w.wrapping_sub(n.ww);
    props[15] = wp_max_error;
}

/// Gather samples from all channels in an image for tree learning (no subsampling).
///
/// For production use on large images, prefer `gather_samples_strided` with a stride
/// computed by `compute_gather_stride_from_profile` to avoid O(n^2) tree learning time.
#[cfg(test)]
pub fn gather_samples(samples: &mut TreeSamples, image: &ModularImage, group_id: u32) {
    gather_samples_strided(
        samples,
        image,
        group_id,
        0,
        1,
        &WeightedPredictorParams::default(),
    );
}

/// Gather samples with stride-based subsampling.
///
/// When `stride > 1`, only every `stride`-th pixel in scan order is sampled.
/// Use `compute_gather_stride_from_profile` to determine the appropriate stride.
pub fn gather_samples_strided(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
    wp_params: &WeightedPredictorParams,
) {
    // Backwards-compatible wrapper: budget-less. Allocations fall back to
    // panicking on OOM, same as before. New callers should prefer the
    // `_with_budget` variant.
    gather_samples_strided_with_budget(
        samples,
        image,
        group_id,
        channel_offset,
        stride,
        wp_params,
        None,
    )
    .expect("budget-less gather_samples_strided must not return AllocationLimit")
}

/// Gather samples with stride-based subsampling plus a `start_offset` that
/// shifts which pixels are sampled.
///
/// `start_offset = 0` is byte-identical to [`gather_samples_strided`].
/// Non-zero offsets (0..stride) draw a different pixel subset — used by
/// RFC#45 chunk 2's multi-seed tree learning (e10/e11). WP error state
/// updates per-pixel regardless, so prediction quality is unaffected.
///
/// Has no effect when `stride <= 1` (every pixel is sampled).
pub fn gather_samples_strided_with_offset(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
    start_offset: usize,
    wp_params: &WeightedPredictorParams,
) {
    gather_samples_strided_with_budget_inner_backend(
        samples,
        image,
        group_id,
        channel_offset,
        stride,
        start_offset,
        wp_params,
        None,
        None,
    )
    .expect("budget-less gather_samples_strided_with_offset must not return AllocationLimit")
}

/// `gather_samples_strided` with explicit allocation budget.
///
/// Per-channel `WeightedPredictorState` scratch (`(width + 2) * 2` errors
/// plus same length × 4 sub-predictor errors) is reserved against the
/// cap. `budget = None` is zero-overhead.
pub(crate) fn gather_samples_strided_with_budget(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
    wp_params: &WeightedPredictorParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<()> {
    gather_samples_strided_with_budget_inner(
        samples,
        image,
        group_id,
        channel_offset,
        stride,
        wp_params,
        budget,
        None,
    )
}

/// `gather_samples_strided_with_budget` plus a flag that controls
/// whether gather-time dedup runs (Phase 2 of issue #41, libjxl
/// `AddSample` parity).
///
/// When `enable_gather_dedup` is `true`, a per-call [`GatherDedupTable`]
/// is constructed (sized from the channel pixel-count / stride estimate)
/// and threaded through every channel of `image`. Cross-channel
/// duplicates merge into the same unique row. When `false`, the call is
/// identical to [`gather_samples_strided_with_budget`].
///
/// `dedup_properties` constrains which property slots feed the hash
/// (production callers thread `params.properties` here so the
/// gather-time merge stays at-or-below the post-sort merge in
/// aggressiveness). Pass an empty slice for the legacy "hash all
/// non-y/x properties" mode.
///
/// Defers to [`gather_samples_strided_with_dedup_backend`] with
/// `enable_phase3 = false` (Phase 2 backend selected unconditionally).
///
/// Phase 2 entry wrapper retained for tests / fallback wiring; production
/// callers go straight to the `_backend` variant with an explicit
/// `enable_phase3` flag. Flagged dead-code in default-features clippy.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn gather_samples_strided_with_dedup(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
    wp_params: &WeightedPredictorParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    enable_gather_dedup: bool,
    dedup_properties: &[usize],
) -> crate::error::Result<()> {
    gather_samples_strided_with_dedup_backend(
        samples,
        image,
        group_id,
        channel_offset,
        stride,
        wp_params,
        budget,
        enable_gather_dedup,
        false,
        dedup_properties,
    )
}

/// Backend-selectable variant of [`gather_samples_strided_with_dedup`].
///
/// `enable_phase3` chooses [`InlineDedupTable`] (Phase 3 of issue #41)
/// over [`GatherDedupTable`] (Phase 2). Has no effect when
/// `enable_gather_dedup` is `false` (no gather-time dedup runs at all).
///
/// The Phase 3 table only activates when the local-key packing fits the
/// [`crate::modular::inline_dedup_table::KEY_BYTES`] budget
/// (`2 * num_pred + 4 * num_properties_hashed`). At e7 RGB this is
/// `28 + 4 * 9 = 64` bytes — exact fit. At e9 RGB this would be
/// `28 + 4 * 24 = 124` bytes (overflow), so the dispatcher falls back to
/// Phase 2 at construction time to avoid silently over-merging
/// bit-different samples. The runtime probe also re-checks the budget
/// per-sample for the squeeze / ref-channel-heavy paths where the
/// upper-bound estimate may not be tight.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_samples_strided_with_dedup_backend(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
    wp_params: &WeightedPredictorParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    enable_gather_dedup: bool,
    enable_phase3: bool,
    dedup_properties: &[usize],
) -> crate::error::Result<()> {
    if !enable_gather_dedup {
        return gather_samples_strided_with_budget_inner(
            samples,
            image,
            group_id,
            channel_offset,
            stride,
            wp_params,
            budget,
            None,
        );
    }
    // Upper-bound estimate of gathered samples for this image: sum of
    // channel pixel counts divided by stride (rounded up). The cuckoo
    // table is sized to keep load < 2/3 throughout.
    let total_pixels: usize = image
        .channels
        .iter()
        .map(|c| c.width().saturating_mul(c.height()))
        .sum();
    let est_samples = total_pixels.div_ceil(stride.max(1)).max(1);

    // Phase 3 fits the inline-key budget only when
    //   2 * max_num_predictors + 4 * num_properties_hashed <= KEY_BYTES.
    // num_predictors is the candidate-list length on `samples`; the
    // property list passed in is what each channel's gather will hash.
    // When the upper bound overflows we silently fall back to Phase 2 —
    // the post-sort arbiter still produces correct bitstream output, so
    // the only consequence is "no Phase 3 win on this image".
    let num_pred_max = samples.num_predictors();
    // Phase 3 hashes the same y/x-skipped property subset Phase 2 uses;
    // the construction filters at `new_with_properties` (Phase 2) and at
    // `properties_kept_for_phase3` (Phase 3) so both backends see the
    // same hashed-property count.
    let num_props_hashed = properties_kept_for_phase3(dedup_properties).len();
    let phase3_fits = phase3_packing_fits(num_pred_max, num_props_hashed);
    let use_phase3 = enable_phase3 && phase3_fits;

    if use_phase3 {
        let mut table = crate::modular::inline_dedup_table::InlineDedupTable::new(est_samples);
        let hashed_props: Vec<u8> = properties_kept_for_phase3(dedup_properties);
        gather_samples_strided_with_budget_inner_backend(
            samples,
            image,
            group_id,
            channel_offset,
            stride,
            0,
            wp_params,
            budget,
            Some(GatherDedupBackend::Phase3 {
                table: &mut table,
                properties: &hashed_props,
            }),
        )
    } else {
        let mut table = if dedup_properties.is_empty() {
            GatherDedupTable::new(est_samples)
        } else {
            GatherDedupTable::new_with_properties(est_samples, dedup_properties)
        };
        gather_samples_strided_with_budget_inner_backend(
            samples,
            image,
            group_id,
            channel_offset,
            stride,
            0,
            wp_params,
            budget,
            Some(GatherDedupBackend::Phase2(&mut table)),
        )
    }
}

/// Filter `dedup_properties` the same way [`GatherDedupTable::new_with_properties`]
/// does — drop slots that `skip_prop_for_gather_dedup` rejects (the static
/// y/x coordinates) and downcast to `u8`. Built once per gather call so
/// the hot loop's match arm reads from this slice rather than re-running
/// the filter per sample.
#[inline]
fn properties_kept_for_phase3(properties: &[usize]) -> Vec<u8> {
    let mut kept = Vec::with_capacity(properties.len());
    for &p in properties {
        if !skip_prop_for_gather_dedup(p) {
            debug_assert!(p < 256, "property index {p} exceeds u8 range");
            kept.push(p as u8);
        }
    }
    kept
}

/// Pure-function precondition test for Phase 3 inline-key packing
/// (issue #41). Returns `true` when the per-sample key
/// `2 * num_pred + 4 * num_props_hashed` bytes fits the
/// [`crate::modular::inline_dedup_table::KEY_BYTES`] budget. Used at
/// `gather_samples_strided_with_dedup_backend` construction time to pick
/// the backend; mirrored at runtime per-sample by `pack_local_key_phase3`'s
/// `LocalKeyPackResult::Overflow`.
#[inline]
fn phase3_packing_fits(num_pred: usize, num_props_hashed: usize) -> bool {
    let bytes_needed = num_pred
        .saturating_mul(2)
        .saturating_add(num_props_hashed.saturating_mul(4));
    bytes_needed <= crate::modular::inline_dedup_table::KEY_BYTES
}

/// Backend dispatch enum for [`gather_channel_samples`]. Holds a mutable
/// reference to either Phase 2's [`GatherDedupTable`] or Phase 3's
/// [`crate::modular::inline_dedup_table::InlineDedupTable`] along with
/// the property list Phase 3 needs to pack its inline key.
///
/// Lifetime `'a` is tied to the per-call table allocated in
/// [`gather_samples_strided_with_dedup_backend`]; both arms are short-lived
/// and exist only for the duration of the gather pass.
///
/// Hot-loop dispatch cost: one byte-tag match per sample (well-predicted
/// branch, single cmov on x86-64). Effectively free compared to the cuckoo
/// probe inside each arm.
pub(crate) enum GatherDedupBackend<'a> {
    /// Phase 2 backend (commit 63e5ea2): SoA-indexed two-hash cuckoo table.
    Phase2(&'a mut GatherDedupTable),
    /// Phase 3 backend (commit 36a7a73): fingerprint-cached inline-key
    /// cuckoo table. `properties` is the y/x-skipped property list the
    /// inline key packs (matches Phase 2's `GatherDedupTable.properties`).
    Phase3 {
        table: &'a mut crate::modular::inline_dedup_table::InlineDedupTable,
        properties: &'a [u8],
    },
}

/// `gather_samples_strided_with_budget` plus an optional gather-time
/// dedup table (Phase 2 of issue #41, libjxl `AddSample` parity).
///
/// The table is threaded through every channel so cross-channel
/// duplicates merge into the same unique row. Sized by the caller
/// (`GatherDedupTable::new`) from an upper-bound sample estimate; the
/// caller is responsible for picking a power-of-two cap large enough
/// to keep load < 2/3 throughout the gather.
///
/// Thin wrapper over [`gather_samples_strided_with_budget_inner_backend`]
/// that wraps the Phase 2 table in [`GatherDedupBackend::Phase2`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_samples_strided_with_budget_inner(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
    wp_params: &WeightedPredictorParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    dedup_table: Option<&mut GatherDedupTable>,
) -> crate::error::Result<()> {
    gather_samples_strided_with_budget_inner_backend(
        samples,
        image,
        group_id,
        channel_offset,
        stride,
        0,
        wp_params,
        budget,
        dedup_table.map(GatherDedupBackend::Phase2),
    )
}

/// Backend-aware variant of [`gather_samples_strided_with_budget_inner`]
/// that dispatches into either Phase 2 or Phase 3 of issue #41 based on
/// the [`GatherDedupBackend`] variant supplied.
///
/// `start_offset` (0..stride) shifts which pixels in scan order are
/// gathered — `0` matches the legacy behaviour, non-zero is used by
/// RFC#45 chunk 2's multi-seed tree learning to draw a different
/// pixel subset per seed. WP error state is unaffected (always updates
/// per pixel).
#[allow(clippy::too_many_arguments)]
fn gather_samples_strided_with_budget_inner_backend(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
    start_offset: usize,
    wp_params: &WeightedPredictorParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    mut dedup_backend: Option<GatherDedupBackend<'_>>,
) -> crate::error::Result<()> {
    // Upper-bound gathered-sample count: ceil(w*h / stride) per channel
    // (dedup backends merge some pushes away — over-reserve is fine).
    // Capacity only — byte-identical; kills the doubling-realloc
    // memmoves in the per-row bulk extends.
    let est: usize = image
        .channels
        .iter()
        .map(|c| (c.width() * c.height()).div_ceil(stride.max(1)))
        .sum();
    samples.reserve_additional(est);

    for (ch_idx, channel) in image.channels.iter().enumerate() {
        // Find reference channels for this channel (preceding channels with matching dims)
        let ref_channel_indices = if samples.num_ref_channels > 0 {
            find_ref_channels(image, ch_idx)
        } else {
            Vec::new()
        };

        gather_channel_samples(
            samples,
            channel,
            ch_idx as u32 + channel_offset,
            group_id,
            stride,
            start_offset,
            wp_params,
            image,
            &ref_channel_indices,
            budget,
            // Re-borrow the backend per-channel so it threads through
            // every channel in the image. The Phase 2 and Phase 3 tables
            // both accumulate state across channels so cross-channel
            // duplicates merge into the same unique row.
            dedup_backend.as_mut().map(|b| match b {
                GatherDedupBackend::Phase2(t) => GatherDedupBackend::Phase2(t),
                GatherDedupBackend::Phase3 { table, properties } => {
                    GatherDedupBackend::Phase3 { table, properties }
                }
            }),
        )?;
    }
    report_tree_sample_stats(samples, stride, est);
    Ok(())
}

/// `JXL_TREE_SAMPLES_STATS=1` — print the gathered `TreeSamples` heap
/// breakdown to stderr. Diagnostic for the encode peak-memory work; costs one
/// `env::var_os` per gather (cached) and nothing else when unset.
fn report_tree_sample_stats(samples: &TreeSamples, stride: usize, reserved_samples: usize) {
    use core::sync::atomic::{AtomicU8, Ordering};
    static ENABLED: AtomicU8 = AtomicU8::new(u8::MAX);
    let on = match ENABLED.load(Ordering::Relaxed) {
        u8::MAX => {
            let v = u8::from(std::env::var_os("JXL_TREE_SAMPLES_STATS").is_some());
            ENABLED.store(v, Ordering::Relaxed);
            v
        }
        v => v,
    };
    if on == 0 {
        return;
    }
    let b = samples.heap_bytes();
    let mib = |n: usize| n as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[tree-samples] stride={stride} reserved={reserved_samples} samples={} \
         cols(pred={}, prop={}) | residual_tokens={:.1} MiB extra_bits={:.1} MiB \
         props={:.1} MiB counts={:.1} MiB TOTAL={:.1} MiB ({:.1} B/sample)",
        b.num_samples,
        b.num_pred_columns,
        b.num_prop_columns,
        mib(b.residual_tokens),
        mib(b.extra_bits),
        mib(b.props),
        mib(b.sample_counts),
        mib(b.total()),
        if b.num_samples > 0 {
            b.total() as f64 / b.num_samples as f64
        } else {
            0.0
        },
    );
}

/// Compute maximum tree samples from an [`EffortProfile`].
///
/// Uses `tree_max_samples_fixed` (when > 0) or `tree_sample_fraction` (when > 0).
pub fn max_tree_samples_from_profile(
    profile: &crate::effort::EffortProfile,
    total_pixels: usize,
) -> usize {
    let base = if profile.tree_sample_fraction > 0.0 {
        // Fraction-based: e.g. 50% of pixels, min 65K
        ((total_pixels as f32 * profile.tree_sample_fraction) as usize).max(65_536)
    } else if profile.tree_max_samples_fixed > 0 {
        profile.tree_max_samples_fixed as usize
    } else {
        32_768
    };
    // Absolute ceiling (0 = uncapped). Without it the sample count — and so the
    // merged TreeSamples accumulator, the encoder's largest live allocation —
    // scales linearly with resolution. Capping here makes the gather stride
    // grow with the image instead, bounding tree-learning memory.
    match tree_sample_ceiling_override().unwrap_or(profile.tree_max_samples_ceiling as usize) {
        0 => base,
        ceiling => base.min(ceiling.max(65_536)),
    }
}

/// `JXL_TREE_MAX_SAMPLES=<n>` overrides the profile's tree-sample ceiling
/// (`0` = uncapped). Sweep knob for the bytes-vs-peak calibration that sets
/// [`crate::effort::EffortProfile::tree_max_samples_ceiling`]; unset in
/// production, where the profile value governs.
fn tree_sample_ceiling_override() -> Option<usize> {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("JXL_TREE_MAX_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// Compute the stride for subsampling from an [`EffortProfile`].
pub fn compute_gather_stride_from_profile(
    total_pixels: usize,
    profile: &crate::effort::EffortProfile,
) -> usize {
    let max_samples = max_tree_samples_from_profile(profile, total_pixels);
    if total_pixels > max_samples {
        total_pixels.div_ceil(max_samples)
    } else {
        1
    }
}

/// Issue #41 chunk B1: flat column-major row staging for the no-dedup
/// gather path.
///
/// The per-sample SoA push block costs `2*num_pred + total_props`
/// (~45-60) individual `Vec::push` calls per gathered sample — each with
/// its own capacity check and length update; the step-0 annotate showed
/// 22% of `gather_samples`' cycles on the scratch->SoA store path
/// (`benchmarks/perf_gather_profile_2026-06-10.meta`). Staging writes
/// each sample into a flat per-column scratch (plain indexed stores, no
/// branches) and flushes once per row as per-column
/// `extend_from_slice` memcpys. Byte-identical: the same values are
/// appended to the same columns in the same order.
///
/// Dedup-backend paths keep the per-sample pushes — the inline probe
/// needs per-sample interaction with the last unique row.
struct GatherRowStaging {
    cap: usize,
    n: usize,
    num_pred: usize,
    total_props: usize,
    tokens: Vec<u8>,
    ebits: Vec<u8>,
    props: Vec<i32>,
}

impl GatherRowStaging {
    fn new(cap: usize, num_pred: usize, total_props: usize) -> Self {
        Self {
            cap,
            n: 0,
            num_pred,
            total_props,
            tokens: vec![0; num_pred * cap],
            ebits: vec![0; num_pred * cap],
            props: vec![0; total_props * cap],
        }
    }

    #[inline]
    fn stage(
        &mut self,
        local_tokens: &[u8],
        local_ebits: &[u8],
        base_props: &[i32; NUM_PROPERTIES],
        local_ref_props: &[i32],
        max_refs: usize,
    ) {
        let i = self.n;
        let cap = self.cap;
        debug_assert!(i < cap);
        for (p, (&t, &e)) in local_tokens.iter().zip(local_ebits.iter()).enumerate() {
            self.tokens[p * cap + i] = t;
            self.ebits[p * cap + i] = e;
        }
        for (prop, &v) in base_props.iter().enumerate() {
            self.props[prop * cap + i] = v;
        }
        for r in 0..max_refs {
            let col = NUM_PROPERTIES + r * 4;
            let off = r * 4;
            self.props[col * cap + i] = local_ref_props[off];
            self.props[(col + 1) * cap + i] = local_ref_props[off + 1];
            self.props[(col + 2) * cap + i] = local_ref_props[off + 2];
            self.props[(col + 3) * cap + i] = local_ref_props[off + 3];
        }
        self.n = i + 1;
    }

    fn flush(&mut self, samples: &mut TreeSamples) {
        let n = self.n;
        if n == 0 {
            return;
        }
        let cap = self.cap;
        debug_assert_eq!(samples.props.len(), self.total_props);
        for p in 0..self.num_pred {
            samples.residual_tokens[p].extend_from_slice(&self.tokens[p * cap..p * cap + n]);
            samples.extra_bits[p].extend_from_slice(&self.ebits[p * cap..p * cap + n]);
        }
        for c in 0..self.total_props {
            samples.props[c].extend_from_slice(&self.props[c * cap..c * cap + n]);
        }
        samples.num_samples += n;
        self.n = 0;
    }
}

/// Gather samples from a single channel with stride-based subsampling.
///
/// When `stride > 1`, only every `stride`-th pixel in scan order is sampled.
/// WP state is still updated for every pixel to maintain correct error tracking.
///
/// `ref_channel_indices` contains indices into `image.channels` of preceding channels
/// with matching dimensions. For each ref channel, 4 properties are computed per pixel.
///
/// `dedup_backend`: when `Some(_)`, each pushed sample is immediately probed
/// against the table; duplicates are popped back from the SoA columns and
/// the existing unique sample's `sample_counts` entry is bumped. The table
/// is passed by `&mut` because it accumulates state across all channels
/// gathered into the same `TreeSamples` (libjxl `AddSample` parity,
/// `enc_ma.cc:711`).
///
/// The backend enum chooses between Phase 2 ([`GatherDedupTable`] —
/// SoA-indexed cuckoo, default when [`TreeLearningParams::gather_dedup`]
/// is on) and Phase 3 ([`InlineDedupTable`] — fingerprint-cached inline-key
/// cuckoo, opt-in via [`TreeLearningParams::gather_dedup_phase3`]). See
/// [`GatherDedupBackend`] for the full contract — both variants produce a
/// strict superset of the post-sort bucket-equivalence set, so the
/// `dedup_samples` arbiter that runs after gather still owns the final
/// byte-determining unique set and hash-locks stay byte-identical when
/// the knob defaults are unchanged.
/// Runtime opt-in (behaviour override — per CLAUDE.md "BEHAVIOUR-override env
/// hooks stay runtime") for libjxl-parity RANDOMIZED tree-learning sample
/// gather. The default FIXED-stride gather aliases against periodic image
/// structure (e.g. document text-line spacing) and can pick a catastrophically
/// non-representative sample set: measured on noaa-leslie 5336 (8.4 MP scan),
/// e7 fixed-stride is +29.6 % vs cjxl, yet the same content compresses to
/// -30 % under a non-aliasing sample. libjxl's `CollectPixelSamples`
/// (lib/jxl/modular/encoding/enc_ma.cc) samples at geometric-random gaps for
/// exactly this reason. When enabled, the per-sample gap is randomized
/// (mean = `stride`) via a deterministic per-(group, channel) xorshift so
/// encoding stays reproducible. Env unset ⇒ byte-identical to before.
/// Provenance: benchmarks/lossless_stride_alias_2026-07-15.* + jxl-encoder#24
/// task #14.
#[cfg(feature = "std")]
fn tree_sample_random_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JXL_TREE_SAMPLE_RANDOM").is_some())
}
#[cfg(not(feature = "std"))]
fn tree_sample_random_enabled() -> bool {
    false
}

/// Next subsample gap for the tree-learning gather. Returns the fixed `stride`
/// (byte-identical to the historical behaviour) unless `randomize`, in which
/// case a xorshift64 draw yields a gap in `[1, 2*stride-1]` (mean = `stride`)
/// that de-aliases the sample positions. `stride <= 1` always gathers every
/// pixel. See [`tree_sample_random_enabled`].
#[inline]
fn next_subsample_gap(rng_state: &mut u64, stride: usize, randomize: bool) -> usize {
    if !randomize || stride <= 1 {
        return stride;
    }
    // xorshift64 — cheap, deterministic, good enough for de-aliasing.
    let mut x = *rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng_state = x;
    1 + (x as usize) % (2 * stride - 1)
}

#[allow(clippy::too_many_arguments)]
fn gather_channel_samples(
    samples: &mut TreeSamples,
    channel: &Channel,
    channel_idx: u32,
    group_id: u32,
    stride: usize,
    start_offset: usize,
    wp_params: &WeightedPredictorParams,
    image: &ModularImage,
    ref_channel_indices: &[usize],
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    dedup_backend: Option<GatherDedupBackend<'_>>,
) -> crate::error::Result<()> {
    let width = channel.width();
    let height = channel.height();
    if width == 0 || height == 0 {
        return Ok(());
    }

    // WP state for computing weighted predictions and property 15
    let mut wp_state = WeightedPredictorState::new_with_budget(wp_params, width, budget)?;

    // prev_gradient tracks the gradient from the previous pixel in scan order.
    // Property 8 = W - prev_gradient. At the start of each row, prev_gradient = 0.
    let mut prev_gradient: i32;

    // Counter for subsampling: only gather when counter == 0.
    //
    // `start_offset` (0..stride) skips the first `start_offset` candidate
    // samples in scan order — used by RFC#45 chunk 2 (multi-seed tree
    // learning) to draw a different pixel subset per seed without
    // touching WP state continuity. WP error tracking still updates on
    // every pixel; only the sample-push gate shifts.
    let mut subsample_counter: usize = if stride > 0 { start_offset % stride } else { 0 };

    // Randomized (libjxl-parity) vs fixed-stride sample gaps. The seed is
    // deterministic per-(group, channel) so encoding stays reproducible;
    // libjxl seeds its `Rng` from group_id. Env unset ⇒ `randomize_sampling`
    // is false ⇒ `next_subsample_gap` returns `stride` ⇒ byte-identical.
    let randomize_sampling = samples.randomize_gather || tree_sample_random_enabled();
    let mut sample_rng: u64 = {
        let s = (group_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (channel_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ 0x2545_F491_4F6C_DD1D;
        if s == 0 { 0xDEAD_BEEF_CAFE_F00D } else { s }
    };

    let max_refs = samples.num_ref_channels;

    // Cache field-counts referenced from the inner loop's dedup probe.
    let num_pred = samples.num_predictors();
    let total_props = samples.total_num_properties();
    // Detach the optional borrow so each `add` step can decide whether to
    // call `try_merge_last` without re-asking for the option.
    let mut dedup_backend = dedup_backend;

    // Issue #41 queue item 1: when the candidate list is the canonical
    // 14-predictor set (content equality — covers the default and any
    // identical-order alias), the per-pixel predictor loop uses the
    // straight-line [`Predictor::predict_all_canonical`] instead of 14
    // match dispatches. Checked once per channel, not per pixel.
    let canonical_preds = samples.candidate_predictors == CANDIDATE_PREDICTORS;

    // Issue #41 chunk B1: row staging on the default (no-dedup) path —
    // see [`GatherRowStaging`]. ~0.3 MB scratch at width 4096.
    let mut staging = if dedup_backend.is_none() {
        Some(GatherRowStaging::new(width, num_pred, total_props))
    } else {
        None
    };

    // Stack scratch for accumulating per-sample fields before the SoA
    // push. Lets the dedup hash read from registers / L1 instead of
    // chasing the `samples.residual_tokens[pred][last_idx]` pointer
    // chain just to re-read what we computed one cycle earlier.
    //
    // 14 candidate predictors (max), 16 base properties + 4 * max_refs
    // for ref-channel properties. We pad the ref-prop array to 32
    // (max_refs <= 8 in practice; we debug_assert below).
    const MAX_CAND_PRED: usize = 16; // > 14 candidates, leaves headroom
    const MAX_REF_PROPS: usize = 32; // 4 props * 8 ref channels
    debug_assert!(num_pred <= MAX_CAND_PRED);
    debug_assert!(4 * max_refs <= MAX_REF_PROPS);

    for y in 0..height {
        prev_gradient = 0;
        for x in 0..width {
            let pixel = channel.get(x, y);

            let n = Neighbors::gather(channel, x, y);

            // Fused WP predict + error update (issue #41 item 2): same
            // values/state sequence as the separate calls — the update
            // always ran immediately after predict here.
            let (wp_pred, wp_max_error) = wp_state.predict_property_update(pixel, x, y, width, &n);

            // Subsample: only gather every stride-th pixel
            if subsample_counter == 0 {
                let props = compute_spec_properties(
                    channel_idx,
                    group_id,
                    x,
                    y,
                    &n,
                    prev_gradient,
                    wp_max_error,
                );

                // Update prev_gradient for next pixel
                prev_gradient = props[9]; // gradient = W + N - NW

                // Stack scratch for predictor outputs. Filled below.
                let mut local_tokens = [0u8; MAX_CAND_PRED];
                let mut local_ebits = [0u8; MAX_CAND_PRED];
                if canonical_preds {
                    // Straight-line all-14 predictions (issue #41 item 1):
                    // identical formulas, no per-predictor match dispatch;
                    // the residual+tokenize loop runs over a fixed array.
                    let mut preds = [0i32; MAX_CAND_PRED];
                    Predictor::predict_all_canonical(&n, wp_pred as i32, &mut preds);
                    for (pred_idx, &prediction) in preds[..num_pred].iter().enumerate() {
                        let residual = pixel - prediction;
                        let packed = pack_signed(residual);
                        let (token, _extra_bits, num_extra) = GATHER_HYBRID_UINT.encode(packed);
                        local_tokens[pred_idx] = token as u8;
                        local_ebits[pred_idx] = num_extra as u8;
                    }
                } else {
                    // Compute residual for each candidate predictor
                    for (pred_idx, &predictor) in samples.candidate_predictors.iter().enumerate() {
                        let prediction = if predictor == Predictor::Weighted {
                            wp_pred as i32
                        } else {
                            predictor.predict_from_neighbors(&n)
                        };
                        let residual = pixel - prediction;
                        let packed = pack_signed(residual);
                        let (token, _extra_bits, num_extra) = GATHER_HYBRID_UINT.encode(packed);
                        local_tokens[pred_idx] = token as u8;
                        local_ebits[pred_idx] = num_extra as u8;
                    }
                }

                // Compute reference channel properties into a local
                // buffer. Same layout as the SoA push below:
                // [|ref0|, ref0, |ref0-gradient0|, ref0-gradient0,
                //  |ref1|, ref1, ..., 0, 0, 0, 0 (zero-pad)]
                let mut local_ref_props = [0i32; MAX_REF_PROPS];
                if max_refs > 0 {
                    for (r, &ref_ch_idx) in ref_channel_indices.iter().enumerate() {
                        let ref_ch = &image.channels[ref_ch_idx];
                        let v = ref_ch.get(x, y);
                        let ref_left = if x > 0 { ref_ch.get(x - 1, y) } else { 0 };
                        let ref_top = if y > 0 {
                            ref_ch.get(x, y - 1)
                        } else {
                            ref_left
                        };
                        let ref_topleft = if x > 0 && y > 0 {
                            ref_ch.get(x - 1, y - 1)
                        } else {
                            ref_left
                        };
                        let ref_predicted = crate::vardct::dc_coding::clamped_gradient(
                            ref_top,
                            ref_left,
                            ref_topleft,
                        );
                        let off = r * 4;
                        local_ref_props[off] = v.wrapping_abs();
                        local_ref_props[off + 1] = v;
                        local_ref_props[off + 2] = v.wrapping_sub(ref_predicted).wrapping_abs();
                        local_ref_props[off + 3] = v.wrapping_sub(ref_predicted);
                    }
                    // Slots beyond ref_channel_indices.len() stay 0 by
                    // local_ref_props initialisation.
                }

                // Chunk B1 fast path: stage into flat row scratch; the
                // per-row flush does the heap appends as per-column
                // memcpys. Same values, same append order — byte-
                // identical to the per-sample push block below.
                if let Some(st) = staging.as_mut() {
                    st.stage(
                        &local_tokens[..num_pred],
                        &local_ebits[..num_pred],
                        &props,
                        &local_ref_props,
                        max_refs,
                    );
                    subsample_counter =
                        next_subsample_gap(&mut sample_rng, stride, randomize_sampling) - 1;
                    continue;
                }

                // Probe the dedup backend BEFORE pushing to SoA columns.
                // The hash reads from registers (the just-computed local
                // arrays), not from the heap — that's the cache-cost
                // difference Phase 1 missed. On a hit we still need to
                // compare against the existing unique row (cold reads of
                // one historical row, which amortises across many merges
                // as the cuckoo table points repeat-pattern samples to
                // the same slot).
                //
                // Backend dispatch (issue #41, Phase 2 vs Phase 3):
                //   * [`GatherDedupBackend::Phase2`]: Phase 2's
                //     [`GatherDedupTable`] — SoA-indexed cuckoo; the
                //     verify on cuckoo-slot collision chases SoA columns.
                //   * [`GatherDedupBackend::Phase3`]: Phase 3's
                //     [`InlineDedupTable`] — fingerprint-cached cuckoo
                //     with canonical key stored inline; the verify reads
                //     only `unique_keys[i]` (single cacheline) instead of
                //     chasing the parallel SoA arrays.
                //
                // Both produce a strict superset of the post-sort
                // bucket-equivalence set (the post-`pre_quantize` arbiter
                // collapses any extra rows); hash-locks therefore stay
                // stable when either knob flips.
                let merge_hit = match dedup_backend.as_mut() {
                    Some(GatherDedupBackend::Phase2(tbl)) => tbl.try_merge_local(
                        samples,
                        &local_tokens[..num_pred],
                        &local_ebits[..num_pred],
                        &props,
                        &local_ref_props[..4 * max_refs],
                    ),
                    Some(GatherDedupBackend::Phase3 { table, properties }) => {
                        match super::inline_dedup_table::pack_local_key_phase3(
                            &local_tokens[..num_pred],
                            &local_ebits[..num_pred],
                            &props,
                            &local_ref_props[..4 * max_refs],
                            properties,
                            NUM_PROPERTIES,
                        ) {
                            super::inline_dedup_table::LocalKeyPackResult::Packed(key) => {
                                // `next_index` here is purely a sentinel
                                // check — the table assigns its own
                                // canonical-key indices. We pass
                                // `samples.num_samples` so debug builds
                                // catch the SLOT_EMPTY-as-fresh-id edge.
                                let probe_idx = (samples.num_samples as u32).min(u32::MAX - 1);
                                table.lookup_or_insert(&key, probe_idx)
                            }
                            super::inline_dedup_table::LocalKeyPackResult::Overflow => {
                                // Phase 3 packing budget exceeded; treat
                                // as miss (no merge). The post-sort
                                // dedup still collapses bucket-equivalent
                                // rows downstream, so output stays
                                // correct — just no gather-time merge
                                // for this row. The `gather_samples_strided_with_dedup`
                                // dispatcher prevents this from being a
                                // hot path by falling back to Phase 2 at
                                // construction time when the worst-case
                                // packing wouldn't fit.
                                None
                            }
                        }
                    }
                    None => None,
                };
                if let Some(existing) = merge_hit {
                    // Hit: bump the existing unique row's count and
                    // skip the SoA push entirely.
                    //
                    // Phase 2 indexes `sample_counts` by the SoA row
                    // index it stored in the cuckoo slot. Phase 3 returns
                    // an index into its own canonical-key array, which is
                    // identical to the SoA row index because both grow
                    // in lockstep with `num_samples` (and we never
                    // pop_back in the local-probe path).
                    samples.sample_counts[existing as usize] += 1;
                    subsample_counter =
                        next_subsample_gap(&mut sample_rng, stride, randomize_sampling) - 1;
                    continue;
                }

                // No dedup or miss: push everything to SoA columns.
                for pred_idx in 0..num_pred {
                    samples.residual_tokens[pred_idx].push(local_tokens[pred_idx]);
                    samples.extra_bits[pred_idx].push(local_ebits[pred_idx]);
                }
                for (prop_list, &val) in samples
                    .props
                    .iter_mut()
                    .zip(props.iter())
                    .take(NUM_PROPERTIES)
                {
                    prop_list.push(val);
                }
                if max_refs > 0 {
                    for r in 0..max_refs {
                        let base = NUM_PROPERTIES + r * 4;
                        let off = r * 4;
                        samples.props[base].push(local_ref_props[off]);
                        samples.props[base + 1].push(local_ref_props[off + 1]);
                        samples.props[base + 2].push(local_ref_props[off + 2]);
                        samples.props[base + 3].push(local_ref_props[off + 3]);
                    }
                }
                samples.num_samples += 1;
                // Seed the new unique row's count when dedup is active.
                // For Phase 3 the table already pushed the canonical key
                // on the inner `lookup_or_insert` miss (no extra work);
                // for Phase 2 we explicitly call `insert_last` to wire
                // the cuckoo slot to the SoA row index we just pushed.
                if dedup_backend.is_some() {
                    samples.sample_counts.push(1);
                    if let Some(GatherDedupBackend::Phase2(ref mut tbl)) = dedup_backend {
                        // Reads are cache-hot (we just pushed). Phase 3
                        // does NOT need this — the canonical key was
                        // already stored on the miss path inside
                        // `lookup_or_insert`.
                        tbl.insert_last(samples, num_pred, total_props);
                    }
                }
                // Sanity (paranoia, debug only): when dedup is OFF, the
                // hash table never inserts and sample_counts stays
                // empty; when dedup is ON, both stay in lockstep.
                debug_assert!(
                    dedup_backend.is_none() || samples.sample_counts.len() == samples.num_samples,
                );

                subsample_counter =
                    next_subsample_gap(&mut sample_rng, stride, randomize_sampling) - 1;
            } else {
                // Still need to track gradient for subsequent pixels
                let grad = n.w.wrapping_add(n.n).wrapping_sub(n.nw);
                prev_gradient = grad;

                subsample_counter -= 1;
            }
        }

        // Row boundary: bulk-flush this row's staged samples (no-op when
        // a dedup backend is active or the row staged nothing).
        if let Some(st) = staging.as_mut() {
            st.flush(samples);
        }
    }
    Ok(())
}

/// libjxl-style EstimateBits cost over a histogram (probability-floor formula).
///
/// Uses log2 with a probability floor of 1/4096, matching libjxl's EstimateBits
/// (enc_ma.cc:54-71). Used for BOTH parent node and sweep child cost estimation,
/// ensuring the split criterion compares costs from the same formula.
///
/// Production callers (find_best_split / compute_predictor_entropy) call
/// [`jxl_simd::estimate_bits_u32`] directly for the SIMD path. This scalar
/// wrapper is retained as the unit-test reference and as a documentation
/// anchor for the cost formula.
#[inline]
#[allow(dead_code)]
pub fn estimate_bits(counts: &[u32], total: u32) -> f64 {
    jxl_simd::estimate_bits_scalar_f64(counts, total)
}

/// Pre-quantized property data for all properties across all samples.
/// Computed once before tree building, eliminating per-node binary_search
/// and threshold_set allocation.
struct PreQuantizedProps {
    /// threshold_sets[prop_idx] = sorted unique thresholds for this property.
    threshold_sets: Vec<Vec<i32>>,
    /// bucket_indices[prop_idx][sample_idx] = bucket index (0..num_thresholds).
    /// Bucket k means: threshold_set[k-1] < value <= threshold_set[k].
    bucket_indices: Vec<Vec<u8>>,
}

impl PreQuantizedProps {
    /// Returns the number of thresholds for a property.
    fn num_thresholds(&self, prop_idx: usize) -> usize {
        self.threshold_sets[prop_idx].len()
    }
}

/// Maximum bytes per packed composite key for [`dedup_samples`]. Covers the
/// worst case of 16 base properties + 16 ref-channel properties = 32 props
/// (1 byte each, pre-quantized bucket index) plus 14 candidate predictors ×
/// 2 bytes (token + extra-bits) = 28 bytes. Total worst-case = 60 bytes;
/// rounded up to a 64-byte cacheline for alignment. Production e7 uses
/// 9 props + 28 = 37 bytes, leaving the tail zero-padded — the trailing
/// zeros are identical across all samples so they don't affect cmp result.
const DEDUP_KEY_BYTES: usize = 64;

/// Empty-slot sentinel for [`StreamingDedupTable`]. Matches libjxl
/// `kDedupEntryUnused` (`lib/jxl/modular/encoding/enc_ma.h:153`).
const DEDUP_EMPTY: u32 = u32::MAX;

/// Multiplicative-hash constants from libjxl `enc_ma.cc:658,673`.
/// Two distinct constants give two independent hash positions per key
/// (cuckoo-style open addressing).
const HASH1_CONST: u64 = 0x1e35a7bd;
const HASH2_CONST: u64 = 0x1e35a7bd1e35a7bd;

/// Property slots that gather-time dedup deliberately skips, even
/// when they appear in `params.properties`. Property 2 = y, 3 = x
/// have raw values unique per pixel — hashing on them blocks every
/// merge. The post-gather sort dedup quantizes these into a small
/// number of coordinate buckets, but gather-time dedup runs before
/// thresholds are known so we can't replicate that here. libjxl
/// applies `QuantizeStaticProperty` to (y, x) before hashing in
/// `AddSample`, achieving the same goal differently.
///
/// Channel (prop 0) and group_id (prop 1) are categorical — they
/// take only a handful of distinct values across the whole image,
/// so hashing on raw values still allows merges within each
/// channel/group cohort.
#[inline]
fn skip_prop_for_gather_dedup(prop_idx: usize) -> bool {
    prop_idx == 2 || prop_idx == 3
}

/// Open-addressing dedup table with two-hash cuckoo placement, ported from
/// libjxl `TreeSamples::AddToTableAndMerge` (`enc_ma.cc:602-655`).
///
/// Each composite key (`[u8; DEDUP_KEY_BYTES]`) is hashed into two slots
/// (`Hash1`, `Hash2`); a sample is considered a duplicate if either slot
/// already contains a unique sample whose key bytes equal the candidate.
///
/// Capacity is sized to `next_pow2(n * 3 / 2)` so the table stays at most
/// 2/3 full at the end of the dedup pass, keeping the expected probe count
/// at 1-2 hits per insert (load factor matching libjxl
/// `PrepareForSamples` at `enc_ma.cc:653`).
struct StreamingDedupTable {
    /// Slot → unique-sample index, or `DEDUP_EMPTY`.
    slots: Box<[u32]>,
    /// `slots.len() - 1`; `&` mask for pow-2 indexing.
    mask: u32,
}

impl StreamingDedupTable {
    fn new(expected_samples: usize) -> Self {
        // Size ≈ 1.5 × n, rounded up to the next power of two so probing
        // can use & (mask) instead of %. Floor at 16 to avoid pathological
        // microscopic tables for tiny tile/group sample counts.
        let target = expected_samples.saturating_mul(3).div_ceil(2).max(16);
        let cap = target.next_power_of_two();
        // 4 GB ceiling — well above any sane modular-tree sample count.
        // u32 indices into unique_keys means `cap` must fit; the floor of
        // expected_samples <= u32::MAX is enforced upstream in dedup_samples.
        let slots = vec![DEDUP_EMPTY; cap].into_boxed_slice();
        Self {
            slots,
            mask: (cap - 1) as u32,
        }
    }

    /// libjxl `Hash1` (`enc_ma.cc:657-671`): multiply-add fold over key
    /// bytes with `0x1e35a7bd`, then `>> 16 & mask`. We hash bytes pairwise
    /// (token,ebits) and bucket-indices in order, matching the libjxl
    /// per-array iteration. The key layout in this Rust port already packs
    /// all those bytes contiguously, so iterating the bytes in slot order
    /// is equivalent.
    #[inline]
    fn hash1(&self, key: &[u8; DEDUP_KEY_BYTES]) -> u32 {
        let mut h: u64 = HASH1_CONST;
        for &b in key.iter() {
            h = h.wrapping_mul(HASH1_CONST).wrapping_add(b as u64);
        }
        ((h >> 16) as u32) & self.mask
    }

    /// libjxl `Hash2` (`enc_ma.cc:672-686`): same fold but with the
    /// `0x1e35a7bd1e35a7bd` 64-bit constant and XOR instead of ADD so the
    /// two functions decorrelate.
    #[inline]
    fn hash2(&self, key: &[u8; DEDUP_KEY_BYTES]) -> u32 {
        let mut h: u64 = HASH2_CONST;
        for &b in key.iter() {
            h = h.wrapping_mul(HASH2_CONST) ^ (b as u64);
        }
        ((h >> 16) as u32) & self.mask
    }

    /// libjxl `AddToTableAndMerge` (`enc_ma.cc:603-630`): probe both hash
    /// slots; on match return `Some(unique_idx)` (caller bumps count); on
    /// miss insert into the first empty slot and return `None`.
    ///
    /// `unique_keys[idx]` gives the canonical key for an already-deduped
    /// sample at unique-sample index `idx`. We compare full keys because
    /// hash collisions are real (cuckoo-style with only two positions can
    /// have false-positive hash matches; eq on the packed key bytes is the
    /// final arbiter).
    #[inline]
    fn lookup_or_insert(
        &mut self,
        key: &[u8; DEDUP_KEY_BYTES],
        unique_keys: &[[u8; DEDUP_KEY_BYTES]],
        next_unique_idx: u32,
    ) -> Option<u32> {
        let h1 = self.hash1(key) as usize;
        let s1 = self.slots[h1];
        if s1 != DEDUP_EMPTY && &unique_keys[s1 as usize] == key {
            return Some(s1);
        }
        let h2 = self.hash2(key) as usize;
        let s2 = self.slots[h2];
        if s2 != DEDUP_EMPTY && &unique_keys[s2 as usize] == key {
            return Some(s2);
        }
        // Miss: insert into the first empty slot. If both are occupied
        // (rare at load factor ≤ 2/3), the new sample is unreachable from
        // future lookups — this is the libjxl behavior (`AddToTable` at
        // `enc_ma.cc:632`). The penalty is one extra unique entry that
        // could have been merged; output bytes are unaffected because
        // identical samples both produce the same residual tokens.
        if s1 == DEDUP_EMPTY {
            self.slots[h1] = next_unique_idx;
        } else if s2 == DEDUP_EMPTY {
            self.slots[h2] = next_unique_idx;
        }
        None
    }
}

/// Build a packed composite key for `sample_idx` from the parallel SoA
/// arrays.
///
/// Layout matches the previous sort-based path so the same `DEDUP_KEY_BYTES`
/// budget applies: `properties.len()` bucket-index bytes followed by
/// `[tok_pred0, eb_pred0, tok_pred1, eb_pred1, ...]` for each candidate
/// predictor. Trailing bytes stay zero-padded so two samples with identical
/// field values produce byte-identical `[u8; 64]` keys.
#[inline]
fn pack_sample_key(
    sample_idx: usize,
    properties: &[usize],
    pq: &PreQuantizedProps,
    samples: &TreeSamples,
    num_pred: usize,
) -> [u8; DEDUP_KEY_BYTES] {
    let mut key = [0u8; DEDUP_KEY_BYTES];
    let mut off = 0;
    for &prop_idx in properties {
        let bi = &pq.bucket_indices[prop_idx];
        if !bi.is_empty() {
            key[off] = bi[sample_idx];
        }
        off += 1;
    }
    for pred in 0..num_pred {
        key[off] = samples.residual_tokens[pred][sample_idx];
        off += 1;
        key[off] = samples.extra_bits[pred][sample_idx];
        off += 1;
    }
    key
}

/// Inline gather-time dedup table — direct translation of libjxl's
/// `dedup_table_` (`enc_ma.h:151-153`) + `AddSample` /
/// `AddToTableAndMerge` (`enc_ma.cc:602-655`, `enc_ma.cc:711`).
///
/// Unlike [`StreamingDedupTable`], this table is consumed *during* the
/// gather pass: each new sample is pushed to every SoA column, then this
/// table is queried with the just-written index. On hit, the SoA columns
/// are popped back (counterpart of libjxl's `pop_back` cascade) and the
/// existing unique sample's count is bumped. On miss, the new sample is
/// retained and inserted into the table for future merges.
///
/// Reading from `samples.residual_tokens[*].last()` /
/// `samples.props[*].last()` is cache-hot because those positions were
/// just written. That is the structural fix Phase 1 missed: the
/// post-pass [`dedup_samples_streaming`] reads SoA columns at arbitrary
/// indices after gather has moved on, taking a full miss-stride per
/// `pack_sample_key`; the gather-time variant pays for the writes only.
///
/// Layout/size matches [`StreamingDedupTable`] — pow-2 cap with mask,
/// `DEDUP_EMPTY` sentinel, two hash positions per entry. Sized once at
/// construction from an upper-bound sample estimate (`expected_samples`)
/// so probes stay branch-stable; the cap is never grown mid-gather.
///
/// Hash inputs: per-predictor `(tok, nbits)` byte pairs followed by raw
/// (non-bucket) i32 property values for the **content-dependent**
/// properties only (`PROPS_START_FOR_GATHER_DEDUP..total_num_properties`).
/// The four static properties — channel, group_id, y, x — are
/// deliberately skipped because their raw values are unique per pixel
/// and would prevent any merges. libjxl's `AddSample` hashes on
/// `QuantizeStaticProperty(...)` outputs that collapse adjacent
/// pixels' (y, x) into the same bucket; we approximate by ignoring
/// those slots entirely. The neighbour-derived properties (|N|, |W|,
/// N, W, gradient differences, wp_max_error) are what actually drive
/// merges on smooth regions / screenshots where adjacent pixels
/// produce identical neighbour patterns.
///
/// Two samples whose neighbour rows differ will not merge here; the
/// post-gather sort dedup may still collapse them once thresholds are
/// known (it hashes on bucket indices and tolerates equivalent values
/// within a bucket). The end result is a strict superset of the
/// post-gather unique set, so wiring is gated behind
/// [`TreeLearningParams::gather_dedup`] and hash-locks regenerated only
/// when callers opt in.
pub(crate) struct GatherDedupTable {
    /// Slot → unique-sample index, or `DEDUP_EMPTY`.
    slots: Box<[u32]>,
    /// `slots.len() - 1`; `&` mask for pow-2 indexing.
    mask: u32,
    /// Pre-computed property indices the hash + IsSameSample check
    /// walks. Built once at construction so the hot loop avoids
    /// re-deriving the (post-y/x-skip) sequence per sample. Production
    /// callers populate from `params.properties`; legacy callers get
    /// the all-but-(y,x) default. Stored as u8 because every spec
    /// property index fits (NUM_PROPERTIES + 4 * max_refs < 256).
    properties: Vec<u8>,
}

impl GatherDedupTable {
    /// Create a table sized for `expected_samples` and configured to
    /// hash on the given property list (post-`skip_prop_for_gather_dedup`
    /// filter). Production callers pass `params.properties`; the table
    /// drops y/x to keep the merge non-trivial.
    pub(crate) fn new_with_properties(expected_samples: usize, properties: &[usize]) -> Self {
        let target = expected_samples.saturating_mul(3).div_ceil(2).max(16);
        let cap = target.next_power_of_two();
        let slots = vec![DEDUP_EMPTY; cap].into_boxed_slice();
        // Filter and downcast to u8: every spec property index fits in
        // a byte (NUM_PROPERTIES + 4 * max_refs is < 256 for any sane
        // ref-channel count).
        let mut props_kept: Vec<u8> = Vec::with_capacity(properties.len());
        for &p in properties {
            if !skip_prop_for_gather_dedup(p) {
                debug_assert!(p < 256, "property index {p} exceeds u8 range");
                props_kept.push(p as u8);
            }
        }
        Self {
            slots,
            mask: (cap - 1) as u32,
            properties: props_kept,
        }
    }

    /// Backwards-compatible constructor that hashes on the legacy
    /// all-but-(y,x) default property set up to a generous upper bound
    /// (`MAX_LEGACY_PROPS`). Retained for callers that haven't yet
    /// threaded `params.properties` through. Production sites use
    /// `new_with_properties`.
    pub(crate) fn new(expected_samples: usize) -> Self {
        // Match the historical NUM_PROPERTIES + 4 * max_refs layout
        // with a generous max_refs bound (8 ⇒ 16 + 32 = 48 slots).
        // Tests / fallback callers rarely have many ref channels; this
        // upper bound just ensures we don't truncate.
        const MAX_LEGACY_PROPS: usize = 48;
        let mut properties: Vec<u8> = Vec::with_capacity(MAX_LEGACY_PROPS);
        for p in 0..MAX_LEGACY_PROPS {
            if !skip_prop_for_gather_dedup(p) {
                properties.push(p as u8);
            }
        }
        let cap = expected_samples
            .saturating_mul(3)
            .div_ceil(2)
            .max(16)
            .next_power_of_two();
        Self {
            slots: vec![DEDUP_EMPTY; cap].into_boxed_slice(),
            mask: (cap - 1) as u32,
            properties,
        }
    }

    /// Hash residual-token bytes then raw property i32 bytes for the
    /// pre-computed property slots. Mirrors libjxl `Hash1` —
    /// multiply-add fold with `0x1e35a7bd`.
    #[inline]
    fn hash1(&self, samples: &TreeSamples, idx: usize, num_pred: usize, total_props: usize) -> u32 {
        let mut h: u64 = HASH1_CONST;
        for pred in 0..num_pred {
            h = h
                .wrapping_mul(HASH1_CONST)
                .wrapping_add(samples.residual_tokens[pred][idx] as u64);
            h = h
                .wrapping_mul(HASH1_CONST)
                .wrapping_add(samples.extra_bits[pred][idx] as u64);
        }
        for &prop in &self.properties {
            let prop = prop as usize;
            if prop >= total_props {
                break;
            }
            let v = samples.props[prop].get(idx).copied().unwrap_or(0) as i64 as u64;
            h = h.wrapping_mul(HASH1_CONST).wrapping_add(v);
        }
        ((h >> 16) as u32) & self.mask
    }

    /// libjxl `Hash2`: same fold with the 64-bit constant + XOR.
    #[inline]
    fn hash2(&self, samples: &TreeSamples, idx: usize, num_pred: usize, total_props: usize) -> u32 {
        let mut h: u64 = HASH2_CONST;
        for &prop in &self.properties {
            let prop = prop as usize;
            if prop >= total_props {
                break;
            }
            let v = samples.props[prop].get(idx).copied().unwrap_or(0) as i64 as u64;
            h = h.wrapping_mul(HASH2_CONST) ^ v;
        }
        for pred in 0..num_pred {
            h = h.wrapping_mul(HASH2_CONST) ^ (samples.residual_tokens[pred][idx] as u64);
            h = h.wrapping_mul(HASH2_CONST) ^ (samples.extra_bits[pred][idx] as u64);
        }
        ((h >> 16) as u32) & self.mask
    }

    /// libjxl `IsSameSample` (`enc_ma.cc:688-708`): branch-free compare
    /// of two samples across the SoA columns considered by the hash —
    /// residual tokens, extra-bit counts, and the configured property
    /// slots.
    #[inline]
    fn is_same_sample(
        &self,
        samples: &TreeSamples,
        a: usize,
        b: usize,
        num_pred: usize,
        total_props: usize,
    ) -> bool {
        for pred in 0..num_pred {
            if samples.residual_tokens[pred][a] != samples.residual_tokens[pred][b] {
                return false;
            }
            if samples.extra_bits[pred][a] != samples.extra_bits[pred][b] {
                return false;
            }
        }
        for &prop in &self.properties {
            let prop = prop as usize;
            if prop >= total_props {
                break;
            }
            let pa = &samples.props[prop];
            // Defensive: skip prop slots that were never populated.
            // gather_channel_samples touches every slot up to
            // total_num_properties(), so this branch is dead code on
            // production paths but keeps tests / future call sites safe.
            if pa.is_empty() {
                continue;
            }
            if pa[a] != pa[b] {
                return false;
            }
        }
        true
    }

    /// Hash from local stack arrays (computed by the gather loop just
    /// before the SoA push). Eliminates the pre-push-then-read SoA
    /// chase that costs cache misses on every probe — the input is
    /// already in registers / L1.
    ///
    /// `local_props` is the 16-element base property array;
    /// `local_ref_props` is the 4 * num_refs ref-property buffer in
    /// the same layout as `samples.props[NUM_PROPERTIES..]`.
    #[inline]
    fn hash1_local(
        &self,
        local_tokens: &[u8],
        local_ebits: &[u8],
        local_props: &[i32; NUM_PROPERTIES],
        local_ref_props: &[i32],
    ) -> u32 {
        let mut h: u64 = HASH1_CONST;
        for (&t, &e) in local_tokens.iter().zip(local_ebits.iter()) {
            h = h.wrapping_mul(HASH1_CONST).wrapping_add(t as u64);
            h = h.wrapping_mul(HASH1_CONST).wrapping_add(e as u64);
        }
        for &prop in &self.properties {
            let prop = prop as usize;
            let v = if prop < NUM_PROPERTIES {
                local_props[prop] as i64 as u64
            } else {
                let off = prop - NUM_PROPERTIES;
                if off < local_ref_props.len() {
                    local_ref_props[off] as i64 as u64
                } else {
                    0
                }
            };
            h = h.wrapping_mul(HASH1_CONST).wrapping_add(v);
        }
        ((h >> 16) as u32) & self.mask
    }

    #[inline]
    fn hash2_local(
        &self,
        local_tokens: &[u8],
        local_ebits: &[u8],
        local_props: &[i32; NUM_PROPERTIES],
        local_ref_props: &[i32],
    ) -> u32 {
        let mut h: u64 = HASH2_CONST;
        for &prop in &self.properties {
            let prop = prop as usize;
            let v = if prop < NUM_PROPERTIES {
                local_props[prop] as i64 as u64
            } else {
                let off = prop - NUM_PROPERTIES;
                if off < local_ref_props.len() {
                    local_ref_props[off] as i64 as u64
                } else {
                    0
                }
            };
            h = h.wrapping_mul(HASH2_CONST) ^ v;
        }
        for (&t, &e) in local_tokens.iter().zip(local_ebits.iter()) {
            h = h.wrapping_mul(HASH2_CONST) ^ (t as u64);
            h = h.wrapping_mul(HASH2_CONST) ^ (e as u64);
        }
        ((h >> 16) as u32) & self.mask
    }

    /// Branch-free compare between the *local* (just-computed) row and
    /// an *existing* (cold) row at `samples.*[b]`. Counterpart of
    /// `is_same_sample` that lets the caller skip writing to SoA on a
    /// hit.
    #[inline]
    fn is_same_local(
        &self,
        local_tokens: &[u8],
        local_ebits: &[u8],
        local_props: &[i32; NUM_PROPERTIES],
        local_ref_props: &[i32],
        samples: &TreeSamples,
        b: usize,
    ) -> bool {
        let num_pred = local_tokens.len();
        for pred in 0..num_pred {
            if local_tokens[pred] != samples.residual_tokens[pred][b] {
                return false;
            }
            if local_ebits[pred] != samples.extra_bits[pred][b] {
                return false;
            }
        }
        for &prop in &self.properties {
            let prop = prop as usize;
            let pa = &samples.props[prop];
            if pa.is_empty() {
                continue;
            }
            let vb = pa[b];
            let va = if prop < NUM_PROPERTIES {
                local_props[prop]
            } else {
                let off = prop - NUM_PROPERTIES;
                if off < local_ref_props.len() {
                    local_ref_props[off]
                } else {
                    0
                }
            };
            if va != vb {
                return false;
            }
        }
        true
    }

    /// Probe the table with the local (about-to-be-pushed) values.
    /// Returns `Some(existing_idx)` when the caller should skip the SoA
    /// push and bump `sample_counts[existing_idx]`. Returns `None` when
    /// the row is new — the caller pushes it to SoA, then calls
    /// `insert_last` to seed the table for future probes.
    ///
    /// Note: when this returns `None` we deliberately do NOT update the
    /// slot table here — the caller hasn't pushed yet, so the index
    /// the table would store is not yet `samples.num_samples - 1`.
    /// Doing the insert post-push lets the caller short-circuit on
    /// hits without touching the SoA columns at all.
    #[inline]
    fn try_merge_local(
        &mut self,
        samples: &TreeSamples,
        local_tokens: &[u8],
        local_ebits: &[u8],
        local_props: &[i32; NUM_PROPERTIES],
        local_ref_props: &[i32],
    ) -> Option<u32> {
        let pos1 =
            self.hash1_local(local_tokens, local_ebits, local_props, local_ref_props) as usize;
        let s1 = self.slots[pos1];
        if s1 != DEDUP_EMPTY
            && self.is_same_local(
                local_tokens,
                local_ebits,
                local_props,
                local_ref_props,
                samples,
                s1 as usize,
            )
        {
            return Some(s1);
        }
        let pos2 =
            self.hash2_local(local_tokens, local_ebits, local_props, local_ref_props) as usize;
        let s2 = self.slots[pos2];
        if s2 != DEDUP_EMPTY
            && self.is_same_local(
                local_tokens,
                local_ebits,
                local_props,
                local_ref_props,
                samples,
                s2 as usize,
            )
        {
            return Some(s2);
        }
        None
    }

    /// Mirror of libjxl `AddToTable` — insert the just-pushed row's
    /// index into the first empty cuckoo slot. Both probe slots
    /// occupied = silent drop (libjxl behaviour at `enc_ma.cc:632`);
    /// future identical rows simply won't merge, costing at worst one
    /// extra unique row.
    #[inline]
    fn insert_last(&mut self, samples: &TreeSamples, num_pred: usize, total_props: usize) {
        debug_assert!(samples.num_samples >= 1);
        let a = samples.num_samples - 1;
        let pos1 = self.hash1(samples, a, num_pred, total_props) as usize;
        if self.slots[pos1] == DEDUP_EMPTY {
            self.slots[pos1] = a as u32;
            return;
        }
        let pos2 = self.hash2(samples, a, num_pred, total_props) as usize;
        if self.slots[pos2] == DEDUP_EMPTY {
            self.slots[pos2] = a as u32;
        }
    }

    /// libjxl `AddToTableAndMerge` (`enc_ma.cc:603-630`) called with the
    /// just-pushed sample index. Returns `Some(existing_idx)` when the
    /// caller should pop_back and bump `sample_counts[existing_idx]`, or
    /// `None` when the sample stays and the table is updated with its
    /// index (after the caller pushes 1 to `sample_counts`).
    #[inline]
    #[allow(dead_code)] // kept for the test that exercises the post-push variant
    fn try_merge_last(
        &mut self,
        samples: &TreeSamples,
        num_pred: usize,
        total_props: usize,
    ) -> Option<u32> {
        debug_assert!(samples.num_samples >= 1);
        let a = samples.num_samples - 1;
        let pos1 = self.hash1(samples, a, num_pred, total_props) as usize;
        let s1 = self.slots[pos1];
        if s1 != DEDUP_EMPTY && self.is_same_sample(samples, a, s1 as usize, num_pred, total_props)
        {
            return Some(s1);
        }
        let pos2 = self.hash2(samples, a, num_pred, total_props) as usize;
        let s2 = self.slots[pos2];
        if s2 != DEDUP_EMPTY && self.is_same_sample(samples, a, s2 as usize, num_pred, total_props)
        {
            return Some(s2);
        }
        // Miss: insert `a` into the first empty slot (libjxl `AddToTable`,
        // `enc_ma.cc:632-640`). If both are occupied, the new sample is
        // unreachable from future probes — that costs one missed merge
        // (extra row in the unique set, still byte-correct downstream).
        let a32 = a as u32;
        if s1 == DEDUP_EMPTY {
            self.slots[pos1] = a32;
        } else if s2 == DEDUP_EMPTY {
            self.slots[pos2] = a32;
        }
        None
    }

    /// Pop the just-pushed sample from every SoA column. Mirror of
    /// libjxl's pop_back cascade in `AddSample` (`enc_ma.cc:731-736`).
    /// Caller is responsible for bumping the existing unique sample's
    /// count and **not** pushing to `sample_counts` for the popped row.
    ///
    /// Used by the test-only push-then-merge path; production gather
    /// short-circuits before the SoA push via `try_merge_local` so
    /// nothing needs popping.
    #[inline]
    #[allow(dead_code)]
    fn pop_last_sample(samples: &mut TreeSamples) {
        for v in &mut samples.residual_tokens {
            v.pop();
        }
        for v in &mut samples.extra_bits {
            v.pop();
        }
        for v in &mut samples.props {
            if !v.is_empty() {
                v.pop();
            }
        }
        samples.num_samples -= 1;
    }
}

/// Deduplicate samples with identical quantized properties and residuals.
///
/// Matching libjxl's approach: after pre-quantization, many pixels in smooth
/// regions have identical (bucket indices, tokens, extra bits) tuples. Merging
/// these with counts reduces the inner loop iterations in FindBestSplit by
/// 1.4-10x on typical photos.
///
/// Dispatches between two backends based on
/// [`TreeLearningParams::use_streaming_dedup`]:
///
/// - [`dedup_samples_packed_sort`] (default, `use_streaming_dedup = false`):
///   materialize all packed composite keys, sort indices by key, walk + merge
///   consecutive runs, then compact SoA columns by gather. O(n log n).
///
/// - [`dedup_samples_streaming`] (`use_streaming_dedup = true`): port of
///   libjxl `AddSample` (`enc_ma.cc:711`) — pack each sample's key inline and
///   look it up in a two-hash cuckoo open-addressing table; either bump the
///   existing unique count or push a new unique slot. O(n) expected.
///
/// **Both paths produce byte-identical bitstreams** (`hash_lock_features`
/// verifies). The streaming path retains *first-seen* order; the sort path
/// retains composite-key-sorted order. `find_best_split` only sees sample
/// values + bucket indices, not row order, so the tree-learning result is
/// invariant to the ordering choice.
///
/// **Default rationale (issue #41):** On our post-gather SoA pipeline the
/// streaming path actually **regresses** end-to-end wall-clock by +3 % to
/// +8 % at e7 on real CLIC photos (0.26 / 1.05 / 4.19 MP). The microbench
/// (`dedup_samples_strategies`) suggests parity, but
/// `dedup_samples_streaming` random-accesses the parallel SoA arrays in
/// `pack_sample_key` (one sample at a time), defeating cache locality. The
/// sort path benefits from spatial coherence — adjacent samples on a photo
/// often share quantized buckets, so packed-key comparisons short-circuit
/// fast. The streaming path cannot exploit that. The opt-in stays in place
/// for issue #41 Phase 2 (integrate dedup into the gather pass itself,
/// where libjxl gets the actual win).
fn dedup_samples(
    samples: &mut TreeSamples,
    pq: &mut PreQuantizedProps,
    params: &TreeLearningParams,
) {
    if params.use_streaming_dedup {
        dedup_samples_streaming(samples, pq, params);
    } else {
        dedup_samples_packed_sort(samples, pq, params);
    }
}

/// Default dedup backend: packed-key sort + walk-and-merge + gather-compact.
///
/// Materializes a packed composite key per sample (`properties.len()`
/// bucket-index bytes followed by `[tok_pred0, eb_pred0, tok_pred1, eb_pred1,
/// ...]` for each candidate predictor), sorts indices by packed key with a
/// fixed-size `[u8; DEDUP_KEY_BYTES]` cmp, walks the sorted run to merge
/// consecutive identical entries, then gathers the unique rows into compact
/// SoA arrays. The cmp reads two adjacent cachelines instead of ~42 scattered
/// `Vec<Vec<u8>>` bytes per side, dropping dedup wall-clock on a 4 MP photo
/// from 8.4 s to 2.4 s (-72 %) vs the pre-packed-key closure path — issue #40
/// follow-on (commit 61129874).
/// Lexicographic compare of two packed dedup keys as big-endian u64
/// words — the identical ordering function to `<[u8; 64]>::cmp`
/// (byte-wise memcmp) but inline with early exit on the first differing
/// word, no libc call. Issue #41 "radix" chunk 1: `__memcmp` from this
/// sort's comparator was 17.7 % / 10.0 % of CPU on the 12 MP / 1 MP
/// photo cells (`perf_gather_profile_2026-06-10.meta` addendum 2).
/// Byte-identity is structural: same ordering function => pdqsort
/// produces the same permutation => identical downstream bytes. (A true
/// radix sort is NOT order-safe here: the sort is unstable and the
/// equal-key representative choice would change.)
#[inline(always)]
fn cmp_packed_key(a: &[u8; DEDUP_KEY_BYTES], b: &[u8; DEDUP_KEY_BYTES]) -> core::cmp::Ordering {
    let mut i = 0;
    while i < DEDUP_KEY_BYTES {
        let wa = u64::from_be_bytes(a[i..i + 8].try_into().unwrap());
        let wb = u64::from_be_bytes(b[i..i + 8].try_into().unwrap());
        if wa != wb {
            return wa.cmp(&wb);
        }
        i += 8;
    }
    core::cmp::Ordering::Equal
}

fn dedup_samples_packed_sort(
    samples: &mut TreeSamples,
    pq: &mut PreQuantizedProps,
    params: &TreeLearningParams,
) {
    let n = samples.num_samples;
    if n <= 1 {
        // Preserve a pre-populated `sample_counts` (Phase 2 gather-time
        // dedup writes it during gather); otherwise seed with 1s.
        if samples.sample_counts.len() != n {
            samples.sample_counts = vec![1; n];
        }
        return;
    }
    // If the upstream gather already produced sample_counts (Phase 2 of
    // issue #41), use those as the initial multiplicity instead of the
    // unconditional `+= 1` per sorted run.
    let preexisting_counts: Option<Vec<u32>> = if samples.sample_counts.len() == n {
        Some(core::mem::take(&mut samples.sample_counts))
    } else {
        None
    };

    let num_pred = samples.num_predictors();
    let properties = &params.properties;

    let key_len = properties.len() + 2 * num_pred;
    debug_assert!(
        key_len <= DEDUP_KEY_BYTES,
        "dedup composite key needs {} bytes, DEDUP_KEY_BYTES = {}",
        key_len,
        DEDUP_KEY_BYTES,
    );

    // Per-sample key build is embarrassingly parallel — each task reads
    // a fixed offset from the parallel SoA arrays and writes a single
    // 64-byte key. Use parallel_map to fan out over the n samples.
    let keys: Vec<[u8; DEDUP_KEY_BYTES]> = crate::parallel::parallel_map(n, |sample_idx| {
        let mut key = [0u8; DEDUP_KEY_BYTES];
        let mut off = 0;
        for &prop_idx in properties {
            let bi = &pq.bucket_indices[prop_idx];
            if !bi.is_empty() {
                key[off] = bi[sample_idx];
            }
            off += 1;
        }
        for pred in 0..num_pred {
            key[off] = samples.residual_tokens[pred][sample_idx];
            off += 1;
            key[off] = samples.extra_bits[pred][sample_idx];
            off += 1;
        }
        key
    });

    // Using u32 indices halves the memory footprint vs Vec<usize>; the
    // tree-learn sample cap (max_tree_samples_from_profile) tops out
    // around 4 M entries, well within u32 range.
    assert!(
        n <= u32::MAX as usize,
        "dedup_samples_packed_sort: n = {n} exceeds u32::MAX; widen key index type"
    );
    // Using u32 indices halves the memory footprint vs Vec<usize>; the
    // tree-learn sample cap (max_tree_samples_from_profile) tops out
    // around 4 M entries, well within u32 range.
    //
    // Sort-shape experiments (perf_sortloc_2026-06-10.meta): sorting
    // (first-word, index) PAIRS — identical ordering function, most
    // compares resolving in-element — measured NEUTRAL-TO-WORSE
    // (clic ~0 %, city12mp +1.8 %, terminal +2.4 %): the 4x element
    // movement inside pdqsort offsets the avoided random key loads.
    // Bare-index sort with the inline word comparator stands.
    let mut order: Vec<u32> = (0..n as u32).collect();
    #[cfg(feature = "parallel")]
    {
        use rayon::slice::ParallelSliceMut;
        order.par_sort_unstable_by(|&a, &b| cmp_packed_key(&keys[a as usize], &keys[b as usize]));
    }
    #[cfg(not(feature = "parallel"))]
    {
        order.sort_unstable_by(|&a, &b| cmp_packed_key(&keys[a as usize], &keys[b as usize]));
    }

    // Walk sorted order, merge consecutive identical samples.
    let mut unique_indices: Vec<usize> = Vec::with_capacity(n / 2);
    let mut counts: Vec<u32> = Vec::with_capacity(n / 2);

    let first = order[0] as usize;
    unique_indices.push(first);
    counts.push(preexisting_counts.as_ref().map(|c| c[first]).unwrap_or(1));
    let mut prev_key_idx = first;
    for &curr_idx in &order[1..] {
        let curr = curr_idx as usize;
        let weight = preexisting_counts.as_ref().map(|c| c[curr]).unwrap_or(1);
        if cmp_packed_key(&keys[curr], &keys[prev_key_idx]) == core::cmp::Ordering::Equal {
            *counts.last_mut().unwrap() += weight;
        } else {
            unique_indices.push(curr);
            counts.push(weight);
            prev_key_idx = curr;
        }
    }

    // Free the packed-key buffer before compaction allocates new SoA
    // columns — peak working set: keys (n × 64 B) + new SoA columns
    // (n × ~70 B) = ~400 MB at 3 M samples; dropping `keys` cuts to ~200 MB.
    drop(keys);
    drop(order);

    let num_unique = unique_indices.len();

    // Compact all parallel arrays to contain only unique samples.
    // Packed-key sort order is preserved, giving spatial locality when the
    // tree builder groups samples by property bucket.
    //
    // Each predictor's (tokens, ebits) compaction and each property's
    // compaction are independent O(num_unique) gathers. Fan out across
    // predictors/properties, then assign the new Vecs back sequentially
    // (sequential because samples / pq are `&mut`).
    let new_per_pred: Vec<(Vec<u8>, Vec<u8>)> = crate::parallel::parallel_map(num_pred, |pred| {
        let old_tokens = &samples.residual_tokens[pred];
        let old_ebits = &samples.extra_bits[pred];
        let new_tokens: Vec<u8> = unique_indices.iter().map(|&i| old_tokens[i]).collect();
        let new_ebits: Vec<u8> = unique_indices.iter().map(|&i| old_ebits[i]).collect();
        (new_tokens, new_ebits)
    });
    for (pred, (new_tokens, new_ebits)) in new_per_pred.into_iter().enumerate() {
        samples.residual_tokens[pred] = new_tokens;
        samples.extra_bits[pred] = new_ebits;
    }

    let total_props = samples.total_num_properties();
    // Compact the property columns in WAVES rather than building every new
    // column before assigning any of them.
    //
    // The all-at-once form held the complete old set AND the complete new set
    // simultaneously — a transient 2x of the single largest allocation in the
    // encoder. At 3840x2160 lossless e9 the props columns are ~1.2 GB, so that
    // rebuild alone put ~1.2 GB of duplicate on top of the peak, and dedup runs
    // exactly where the peak is (the RSS timeline spikes during gather/merge/
    // dedup, then falls for the 70 s tree build).
    //
    // Waves assign and drop each batch before building the next, bounding the
    // duplicate to `DEDUP_COMPACT_WAVE` columns. Byte-identical: each column is
    // gathered by the same `unique_indices` in the same order, and columns are
    // independent of one another — only the interleaving of allocation and
    // release changes.
    //
    // Not done in place (which would need no duplicate at all): the
    // representatives in `unique_indices` are in sorted-KEY order, not
    // ascending sample order, so a forward in-place compaction would read
    // already-overwritten slots. Sorting them first would reorder the samples
    // and change tie-breaks in the tree, so it is not a byte-identical option.
    const DEDUP_COMPACT_WAVE: usize = 4;

    let mut start = 0usize;
    while start < total_props {
        let end = (start + DEDUP_COMPACT_WAVE).min(total_props);
        let new_props: Vec<Vec<i32>> = crate::parallel::parallel_map(end - start, |k| {
            let old_props = &samples.props[start + k];
            if old_props.is_empty() {
                Vec::new()
            } else {
                unique_indices.iter().map(|&i| old_props[i]).collect()
            }
        });
        for (k, np) in new_props.into_iter().enumerate() {
            if !samples.props[start + k].is_empty() {
                samples.props[start + k] = np; // old column dropped here
            }
        }
        start = end;
    }

    let bi_total = pq.bucket_indices.len().min(total_props);
    let mut start = 0usize;
    while start < bi_total {
        let end = (start + DEDUP_COMPACT_WAVE).min(bi_total);
        let new_bi: Vec<Vec<u8>> = crate::parallel::parallel_map(end - start, |k| {
            let old_bi = &pq.bucket_indices[start + k];
            if old_bi.is_empty() {
                Vec::new()
            } else {
                unique_indices.iter().map(|&i| old_bi[i]).collect()
            }
        });
        for (k, nb) in new_bi.into_iter().enumerate() {
            if !pq.bucket_indices[start + k].is_empty() {
                pq.bucket_indices[start + k] = nb;
            }
        }
        start = end;
    }

    samples.num_samples = num_unique;
    samples.sample_counts = counts;
}

/// Opt-in dedup backend: streaming two-hash cuckoo open addressing.
///
/// Ports libjxl's `AddSample` / `AddToTableAndMerge` (`enc_ma.cc:602-655`,
/// `enc_ma.cc:711`). Each sample's packed composite key is looked up in a
/// pow-2-sized hash table with two hash positions per key; on hit, bump the
/// unique sample's count; on miss, allocate a new unique-sample slot. No
/// post-pass sort, no walk-and-merge — compaction runs once over the
/// unique-row representatives.
///
/// Memory: peak working set is `n × 64 B` for `unique_keys` (canonical
/// per-unique-sample packed keys, retained for collision verification) plus
/// the slot table (`next_pow2(n * 3 / 2) × 4 B`). At 3 M samples both fit
/// in ~200 MB, equal to the sort path's peak.
///
/// **Wall-clock regression vs `dedup_samples_packed_sort`**: +3 % to +8 %
/// end-to-end at e7 on CLIC photos (issue #41 measurement, 2026-05-16).
/// Cause: `pack_sample_key` random-accesses the parallel SoA arrays per
/// sample (column-strided reads with no locality), and the sort path
/// benefits from spatial coherence of adjacent photo pixels that the hash
/// path cannot exploit. Retained for experimentation toward issue #41
/// Phase 2 (gather-integrated dedup), where libjxl gets its actual win
/// because keys are built once during sample ingestion.
fn dedup_samples_streaming(
    samples: &mut TreeSamples,
    pq: &mut PreQuantizedProps,
    params: &TreeLearningParams,
) {
    let n = samples.num_samples;
    if n <= 1 {
        if samples.sample_counts.len() != n {
            samples.sample_counts = vec![1; n];
        }
        return;
    }
    // Mirror the sort path: respect any pre-existing sample_counts so
    // gather-time dedup composes cleanly with the streaming backend too.
    let preexisting_counts: Option<Vec<u32>> = if samples.sample_counts.len() == n {
        Some(core::mem::take(&mut samples.sample_counts))
    } else {
        None
    };

    let num_pred = samples.num_predictors();
    let properties = &params.properties;

    let key_len = properties.len() + 2 * num_pred;
    debug_assert!(
        key_len <= DEDUP_KEY_BYTES,
        "dedup composite key needs {} bytes, DEDUP_KEY_BYTES = {}",
        key_len,
        DEDUP_KEY_BYTES,
    );

    assert!(
        n <= u32::MAX as usize,
        "dedup_samples_streaming: n = {n} exceeds u32::MAX; widen key index type"
    );

    // Hash table sized for the worst case (all n unique). Real photos
    // dedup to roughly 60-90 % unique, so the table stays well-loaded but
    // not pathologically full.
    let mut table = StreamingDedupTable::new(n);

    // unique_keys[u] = canonical packed key for unique-sample index u.
    // Reserve worst-case capacity to avoid reallocation during the streaming
    // pass — peak memory matches the sort path's `keys` allocation.
    let mut unique_keys: Vec<[u8; DEDUP_KEY_BYTES]> = Vec::with_capacity(n);

    // unique_indices[u] = first sample index that mapped to unique u
    // (a representative, for compacting SoA arrays).
    let mut unique_indices: Vec<u32> = Vec::with_capacity(n / 2 + 1);
    let mut counts: Vec<u32> = Vec::with_capacity(n / 2 + 1);

    // Streaming dedup: walk samples in scan order, hash composite key,
    // either bump count or push a new unique entry.
    for sample_idx in 0..n {
        let key = pack_sample_key(sample_idx, properties, pq, samples, num_pred);
        let next_idx = unique_indices.len() as u32;
        let weight = preexisting_counts
            .as_ref()
            .map(|c| c[sample_idx])
            .unwrap_or(1);
        if let Some(existing) = table.lookup_or_insert(&key, &unique_keys, next_idx) {
            counts[existing as usize] += weight;
        } else {
            unique_indices.push(sample_idx as u32);
            counts.push(weight);
            unique_keys.push(key);
        }
    }

    // Free the hash table + unique_keys before compaction allocates the
    // new SoA columns — these are the largest working buffers.
    drop(table);
    drop(unique_keys);

    let num_unique = unique_indices.len();

    // Compact all parallel arrays to contain only unique samples, in
    // first-seen order.
    for pred in 0..num_pred {
        let old_tokens = &samples.residual_tokens[pred];
        let old_ebits = &samples.extra_bits[pred];
        let new_tokens: Vec<u8> = unique_indices
            .iter()
            .map(|&i| old_tokens[i as usize])
            .collect();
        let new_ebits: Vec<u8> = unique_indices
            .iter()
            .map(|&i| old_ebits[i as usize])
            .collect();
        samples.residual_tokens[pred] = new_tokens;
        samples.extra_bits[pred] = new_ebits;
    }
    let total_props = samples.total_num_properties();
    for prop_idx in 0..total_props {
        let old_props = &samples.props[prop_idx];
        if old_props.is_empty() {
            continue;
        }
        let new_props: Vec<i32> = unique_indices
            .iter()
            .map(|&i| old_props[i as usize])
            .collect();
        samples.props[prop_idx] = new_props;
    }
    for prop_idx in 0..total_props {
        if prop_idx >= pq.bucket_indices.len() {
            break;
        }
        let old_bi = &pq.bucket_indices[prop_idx];
        if old_bi.is_empty() {
            continue;
        }
        let new_bi: Vec<u8> = unique_indices.iter().map(|&i| old_bi[i as usize]).collect();
        pq.bucket_indices[prop_idx] = new_bi;
    }

    samples.num_samples = num_unique;
    samples.sample_counts = counts;
}

/// Context for a node being considered for splitting.
struct SplitCandidate {
    /// Index into the tree's node vector.
    node_idx: usize,
    /// Range of samples belonging to this node: [start, end).
    start: usize,
    end: usize,
    /// Best predictor index for this node (if kept as leaf).
    best_predictor: usize,
    /// Entropy in bits if kept as leaf with best predictor.
    base_bits: f64,
    /// Multiplier for this leaf (set by lossy modular quantization).
    multiplier: Option<u32>,
    /// PERF-HIST-SUB-LOSSLESS: this node's pre-computed aggregate tensor
    /// (built for the smaller sibling, derived by subtraction for the
    /// larger). `None` on engines/paths without tensor support and below
    /// the profitability gates. Dropped (freeing the buffers) whenever the
    /// node leaf-finalizes without splitting.
    tensor: Option<NodeTensor>,
}

/// Learn an optimal MA tree from gathered samples.
///
/// Uses a greedy top-down splitting approach:
/// 1. Start with all samples in one leaf, pick the best predictor.
/// 2. For each property and threshold, compute entropy of left/right partitions.
/// 3. Split on the (property, threshold) that reduces entropy most.
/// 4. Repeat until no beneficial split or max_nodes reached.
///
/// Parameters are effort-dependent via `TreeLearningParams`:
/// - `params.properties`: which properties to consider for splits
/// - `params.max_property_values`: max quantization buckets per property
/// - `params.split_threshold`: minimum bits saved for a split to be accepted
/// - `params.max_nodes`: maximum tree nodes
pub fn compute_best_tree(samples: &mut TreeSamples, params: &TreeLearningParams) -> Tree {
    compute_best_tree_with_budget(samples, params, None)
        .expect("budget-less compute_best_tree must not return AllocationLimit")
}

/// `compute_best_tree` with explicit allocation budget.
///
/// Charges the dimension-driven allocations against the cap:
///
/// - `indices: Vec<usize>` of `num_samples` entries (8 B each)
/// - `bucket_indices` from [`TreeSamples::pre_quantize`]: up to
///   `total_num_properties × num_samples` u8 entries (1 B each)
/// - `entropy_counts: Vec<u32>` of `histogram_size` (small, but charged
///   for completeness)
///
/// `num_samples` itself is dim-driven: see
/// [`max_tree_samples_from_profile`], which scales with image area
/// (`tree_sample_fraction × total_pixels`, capped at `tree_max_samples_fixed`).
///
/// `budget = None` is zero-overhead.
pub(crate) fn compute_best_tree_with_budget(
    samples: &mut TreeSamples,
    params: &TreeLearningParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Tree> {
    // Scale threshold by pixel_fraction, matching libjxl's required_cost formula.
    let required_cost = params.pixel_fraction * 0.9 + 0.1;
    let threshold = params.split_threshold * required_cost;
    let n = samples.num_samples;
    if n == 0 {
        return Ok(vec![PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Gradient,
            context_id: 0,
            multiplier: 1,
            ..Default::default()
        }]);
    }

    // Reserve the dim-driven scratch up front.
    //  - `bucket_indices`: up to (total_num_properties × n) bytes
    //
    // Issue #40 chunk 2: dropped the previous `indices: Vec<usize>` (n × usize)
    // — the tree builder now partitions the SoA arrays in-place via
    // `split_tree_samples_in_place`, so no auxiliary index array is allocated.
    let bi_bytes = (samples.total_num_properties() as u64).saturating_mul(n as u64);
    let total_bytes = bi_bytes;
    crate::budget::MemoryBudget::reserve_permanent_opt(budget, total_bytes)?;

    // Pre-quantize all properties globally (replaces per-node binary_search)
    let mut pq = crate::profile_time!("tree/pre_quantize", { samples.pre_quantize(params) });

    // When `params.gather_dedup` was set, the gather loop already
    // populated `sample_counts`. Either: counts.len() == num_samples
    // (gather-time dedup ran) OR counts is empty (call site did not
    // honor the flag — fall back to the post-pass dedup default).
    debug_assert!(
        !params.gather_dedup
            || samples.sample_counts.is_empty()
            || samples.sample_counts.len() == samples.num_samples,
        "gather_dedup=true but sample_counts.len()={} != num_samples={}",
        samples.sample_counts.len(),
        samples.num_samples,
    );

    // Sample deduplication: group samples with identical (quantized props, tokens, ebits).
    // Matching libjxl's approach, this reduces inner loop iterations in
    // FindBestSplit by 1.4-10x on typical photos.
    //
    // When gather-time dedup populated `sample_counts` already, the
    // post-pass sort still has a residual role: it merges
    // bucket-equivalent rows whose raw values differed (so the gather
    // hash kept them apart). Skipping it would leave those collisions
    // unmerged — find_best_split would then evaluate more unique rows
    // than necessary. We keep it but pass the pre-computed counts.
    // `props` is dead the moment `pre_quantize` has projected it into
    // `pq.bucket_indices`, which is BEFORE dedup — `dedup_samples` builds its
    // packed composite keys from `bucket_indices`, never from `props`, and
    // every partition afterwards uses `PartitionKey::Bucket` (both call sites
    // below pass `skip_props_swap = true`).
    //
    // Releasing it here rather than after dedup matters because dedup IS the
    // peak phase: it allocates an n x 64 B packed-key buffer plus a fresh set
    // of SoA columns while the old ones are alive. Holding ~1.2 GB of
    // already-dead property columns across that is the single largest avoidable
    // overlap in the encoder. Freeing first also makes dedup's own props
    // rebuild a no-op — every column is empty, so its `is_empty()` guards skip
    // it — which removes that work and its transient entirely.
    //
    // Gated on the chunk-3c escape hatch: with `JXL_DISABLE_CHUNK3C` set,
    // `swap_rows` DOES swap props, so they must stay alive AND stay aligned,
    // which means dedup must keep compacting them.
    if !chunk3c_skip_is_disabled() {
        samples.free_props();
    }

    crate::profile_time!("tree/dedup_samples", {
        dedup_samples(samples, &mut pq, params);
    });

    let n = samples.num_samples; // Update n to unique count

    let max_nodes = params.max_nodes;

    // Max token value across all predictors (for histogram sizing)
    let max_token = samples
        .residual_tokens
        .iter()
        .flat_map(|v| v.iter())
        .copied()
        .max()
        .unwrap_or(0) as usize;
    let histogram_size = max_token + 1;

    // Build the tree
    let mut tree: Tree = Vec::new();

    // Reusable buffer for entropy computation (avoids per-call Vec allocation).
    let mut entropy_counts = vec![0u32; histogram_size];

    // Start with root node
    let root_predictor = crate::profile_time!("tree/find_best_predictor", {
        find_best_predictor(samples, 0, n, histogram_size, &mut entropy_counts)
    });
    let root_bits = crate::profile_time!("tree/compute_predictor_entropy", {
        compute_predictor_entropy(
            samples,
            0,
            n,
            root_predictor,
            histogram_size,
            &mut entropy_counts,
        )
    });

    // LIFO stack for greedy splitting
    let mut stack: Vec<SplitCandidate> = Vec::new();

    // Reserve slot 0 for root
    tree.push(PropertyDecisionNode::default());
    stack.push(SplitCandidate {
        node_idx: 0,
        start: 0,
        end: n,
        best_predictor: root_predictor,
        base_bits: root_bits,
        multiplier: None,
        tensor: None,
    });

    // The workspace lives in the thread-local cache (see
    // `with_thread_local_workspace`) so we don't allocate ~12 MB per fork on
    // the parallel path. The cache grows in place; subsequent calls on the
    // same worker thread are allocation-free.
    let max_buckets = params.max_property_values + 1;

    // PERF-HIST-SUB-LOSSLESS: shared tensor layout for this tree build.
    // Cheap to construct (one entry per property); the capture/derive gates
    // decide per-node whether tensors actually materialise.
    let tensor_layout = TensorLayout::new(params, samples.num_predictors(), histogram_size, |p| {
        pq.num_thresholds(p)
    });

    // ── Parallel tree learning (chunk-1 POC, issue #41 follow-on) ────────────
    //
    // When the `parallel-tree-learning` feature is on AND we have enough samples
    // to benefit, do the root split sequentially, then build the two sibling
    // subtrees in parallel via owned per-side clones of (samples, pq).
    //
    // Theoretical speedup capped at ~2× from the root split alone; deeper levels
    // remain sequential in this chunk. Bitstream-equivalent: topology is data-
    // determined (same samples → same splits) and serialization is BFS-from-root
    // (`collect_tree_tokens`) so internal node-vec indexing is invisible.
    //
    // The clone overhead is O(N) per side (split_off is amortized linear in the
    // detached tail length); on a 4.19 MP image with ~1.3M post-dedup samples,
    // the two clones cost ~50-100 ms total — negligible vs the multi-second
    // tree-build that follows.
    #[cfg(feature = "parallel-tree-learning")]
    {
        // Effort-tuned via `params.parallel_root_threshold` (see
        // `EffortProfile::tree_parallel_root_threshold_for`). Default
        // schedule: 8192 at effort ≤ 7, 4096 at effort ≥ 8 — the larger
        // e8/e9 trees benefit from a lower gate so the root-split
        // amortises across more downstream work.
        let parallel_root_threshold = params.parallel_root_threshold;
        if std::env::var("JXL_DBG_PARALLEL_TREE").is_ok() {
            eprintln!(
                "PARALLEL_TREE: n={}, max_nodes={}, root_bits={:.1}, threshold={:.1}, \
                 root_thresh={}, max_depth={}, floor={}, gate={}",
                n,
                max_nodes,
                root_bits,
                threshold,
                parallel_root_threshold,
                params.parallel_max_depth,
                params.parallel_recursion_floor,
                n >= parallel_root_threshold && max_nodes >= 4 && root_bits > threshold
            );
        }
        // Only attempt parallel root split when there's enough work AND we
        // haven't been told to stop early (max_nodes <= 3 means root + 2
        // children is already the budget; sequential path is fine).
        //
        // The small-image fallback (audit items #9 + #10) does NOT skip
        // the parallel path — the 8-thread fan-out is the largest single
        // win on the lossless e7 pipeline regardless of image size. The
        // fallback only bypasses the thread-local SplitWorkspace cache
        // (see `with_workspace_dispatched`), which is the cheaper
        // intervention and addresses the +0.85% small-image regression
        // documented in commit `cb5e202`. Resurrecting the owned-clone
        // path to fix the +6.2% borrowed-view regression on top of
        // that is tracked separately (see CLAUDE.md / the audit memory).
        if n >= parallel_root_threshold && max_nodes >= 4 && root_bits > threshold {
            // Pop the root candidate and try its split.
            let root_candidate = stack.pop().expect("root candidate just pushed");
            // PERF-HIST-SUB-LOSSLESS: capture the root's tensor while its
            // per-sample loops run anyway, so the two subtree roots can be
            // derived (smaller built + larger subtracted) without a second
            // full pass. Skipped on the owned small-image fallback — that
            // path keeps the documented full-rebuild.
            let mut root_capture: Option<NodeTensor> = if !params.parallel_small_image_fallback
                && tensor_capture_pays(&tensor_layout, n)
            {
                Some(NodeTensor::zeroed(&tensor_layout))
            } else {
                None
            };
            let best_split = with_workspace_dispatched(
                params.parallel_small_image_fallback,
                n,
                histogram_size,
                max_buckets,
                |workspace| {
                    find_best_split(
                        samples,
                        root_candidate.start,
                        root_candidate.end,
                        histogram_size,
                        root_candidate.base_bits,
                        params,
                        root_candidate.best_predictor,
                        threshold,
                        &pq,
                        workspace,
                        match root_capture.as_mut() {
                            Some(t) => TensorMode::Capture(&tensor_layout, t),
                            None => TensorMode::Off,
                        },
                    )
                },
            );

            match best_split {
                Some(split) if root_candidate.base_bits - split.total_bits > threshold => {
                    let bucket_split =
                        bucket_for_splitval(&pq.threshold_sets[split.property], split.splitval);
                    // Issue #40 chunk-3c: lossless path uses Bucket partition
                    // exclusively and never reads `samples.props` after this
                    // point — skip the per-property Vec<i32> swaps.
                    let abs_mid = partition_node_in_place_with(
                        samples,
                        &mut pq,
                        root_candidate.start,
                        root_candidate.end,
                        split.left_count,
                        tree_learn_split::PartitionKey::Bucket {
                            prop_idx: split.property,
                            val: bucket_split as u8,
                        },
                        true,
                    );

                    // Per-side base bits at the winning threshold, carried
                    // out of the split sweep (issue #64 side-costs rider) —
                    // bitwise-identical to the compute_predictor_entropy
                    // recompute this replaces; debug builds re-derive + assert.
                    let lb = split.left_bits;
                    let rb = split.right_bits;
                    debug_verify_carried_side_bits(
                        samples,
                        &split,
                        root_candidate.start,
                        abs_mid,
                        root_candidate.end,
                        histogram_size,
                        &mut entropy_counts,
                    );

                    // PERF-HIST-SUB-LOSSLESS: derive the two subtree-root
                    // tensors from the captured root tensor. Must happen
                    // before the borrowed views are formed (the build pass
                    // reads `samples` + `pq` directly).
                    let (left_tensor, right_tensor) = match root_capture.take() {
                        Some(parent_t) => derive_child_tensors(
                            samples,
                            &pq,
                            params,
                            &tensor_layout,
                            histogram_size,
                            root_candidate.start,
                            abs_mid,
                            root_candidate.end,
                            parent_t,
                            lb,
                            rb,
                            threshold,
                        ),
                        None => (None, None),
                    };

                    // Set the root split node in the parent tree.
                    // Children indices are filled in after stitching.
                    let left_predictor = split.left_predictor;
                    let right_predictor = split.right_predictor;
                    let split_property = split.property as i32;
                    let split_splitval = split.splitval;

                    // Halve the node budget for each side, leaving the root
                    // node itself accounted for in the parent.
                    let per_side_budget = (max_nodes - 1) / 2;

                    // Recursive parallel decomposition. Effort-tuned via
                    // `params.parallel_max_depth` (see
                    // `EffortProfile::tree_parallel_max_depth_for`). Default
                    // schedule: 4 at effort ≤ 7 (16 leaf tasks), 5 at effort
                    // ≥ 8 (32 leaf tasks — deeper e8/e9 trees have enough
                    // per-leaf work to amortise the extra spawns). Deeper
                    // recursion gives diminishing returns once subtrees shrink
                    // below `params.parallel_recursion_floor`.
                    let max_parallel_depth: u32 = params.parallel_max_depth;

                    // Issue #42 (2026-05-25): on small inputs (< 1 MP, e ≤ 7,
                    // gated by `params.parallel_small_image_fallback`), dispatch
                    // to the owned-clone path. The borrowed-view path's per-fork
                    // slice-tracking containers + indirection cost outpace the
                    // saved `split_off` memcpy on small inputs. The owned-clone
                    // path is bitstream-equivalent — same partition semantics,
                    // same find_best_split inputs, same tree topology.
                    let (left_tree, right_tree) = if params.parallel_small_image_fallback {
                        let ((left_samples, left_pq), (right_samples, right_pq)) =
                            split_owned_from_borrowed(samples, &mut pq, abs_mid);

                        if std::env::var("JXL_DBG_PARALLEL_TREE").is_ok() {
                            let l = left_samples.num_samples;
                            let r = right_samples.num_samples;
                            eprintln!(
                                "PARALLEL_TREE[owned]: root split → left={} right={} (imbalance={:.2}x)",
                                l,
                                r,
                                if l > r {
                                    l as f64 / r.max(1) as f64
                                } else {
                                    r as f64 / l.max(1) as f64
                                },
                            );
                        }

                        crate::parallel::parallel_join(
                            || {
                                build_subtree_recursive_parallel(
                                    left_samples,
                                    left_pq,
                                    params,
                                    threshold,
                                    per_side_budget,
                                    histogram_size,
                                    left_predictor,
                                    lb,
                                    max_parallel_depth,
                                )
                            },
                            || {
                                build_subtree_recursive_parallel(
                                    right_samples,
                                    right_pq,
                                    params,
                                    threshold,
                                    per_side_budget,
                                    histogram_size,
                                    right_predictor,
                                    rb,
                                    max_parallel_depth,
                                )
                            },
                        )
                    } else {
                        // Borrow samples + pq as a single view with mutable slice
                        // refs (issue #41 follow-on, 2026-05-16). The original
                        // `split_tree_samples_owned` + `split_pq_owned` cloned
                        // ~13 MB at the top fork via 52 Vec::split_off calls.
                        // The borrowed view splits each underlying Vec in half
                        // via `split_at_mut` for zero memcpy and zero allocator
                        // pressure.
                        let view = BorrowedSamples::from_owned(samples, &mut pq);
                        let (left_view, right_view) = view.split_at_mut(abs_mid);

                        if std::env::var("JXL_DBG_PARALLEL_TREE").is_ok() {
                            eprintln!(
                                "PARALLEL_TREE: root split → left={} right={} (imbalance={:.2}x)",
                                left_view.len,
                                right_view.len,
                                if left_view.len > right_view.len {
                                    left_view.len as f64 / right_view.len.max(1) as f64
                                } else {
                                    right_view.len as f64 / left_view.len.max(1) as f64
                                },
                            );
                        }

                        crate::parallel::parallel_join(
                            || {
                                build_subtree_recursive_parallel_borrowed(
                                    left_view,
                                    params,
                                    threshold,
                                    per_side_budget,
                                    histogram_size,
                                    left_predictor,
                                    lb,
                                    max_parallel_depth,
                                    &tensor_layout,
                                    left_tensor,
                                )
                            },
                            || {
                                build_subtree_recursive_parallel_borrowed(
                                    right_view,
                                    params,
                                    threshold,
                                    per_side_budget,
                                    histogram_size,
                                    right_predictor,
                                    rb,
                                    max_parallel_depth,
                                    &tensor_layout,
                                    right_tensor,
                                )
                            },
                        )
                    };

                    // Splice subtrees into the parent tree, capturing the
                    // index of each subtree's root in the parent's storage.
                    let lchild_idx = splice_subtree(&mut tree, left_tree);
                    let rchild_idx = splice_subtree(&mut tree, right_tree);

                    // Now we can fill in the root split node's child pointers.
                    tree[0] = PropertyDecisionNode {
                        property: split_property,
                        splitval: split_splitval,
                        lchild: lchild_idx,
                        rchild: rchild_idx,
                        ..Default::default()
                    };

                    // Restore samples for downstream code (validation etc.
                    // do not read `samples` after this point — the build
                    // sequence is finished; only `assign_sequential_contexts`
                    // and `validate_tree_djxl` follow). Clear the stack so
                    // the fallthrough loop below sees nothing.
                    stack.clear();
                }
                _ => {
                    // No beneficial root split — push the root candidate back
                    // and fall through to the sequential loop, which will
                    // leaf-finalize on the first iteration.
                    stack.push(root_candidate);
                }
            }
        }
    }

    while let Some(mut candidate) = stack.pop() {
        if tree.len() + 2 > max_nodes {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }

        let count = candidate.end - candidate.start;
        if count < 2 {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }

        // Early termination gate: if base_bits is already below threshold,
        // no split can save enough bits. Matches libjxl enc_ma.cc:304.
        if candidate.base_bits <= threshold {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }

        // Find best split across all properties and thresholds.
        // Workspace lives in a per-thread cache (12 MB at large n) — the
        // outer loop runs on the main thread so this is one calloc per
        // encode (first iter) and zero on subsequent iters.
        //
        // PERF-HIST-SUB-LOSSLESS: a node arriving with a tensor reads its
        // bucket stats + (prop, pred) rows from it (per-sample loops
        // skipped); a big-enough node without one captures its tensor while
        // the loops run, re-seeding derivation for its children.
        let n_node = candidate.end - candidate.start;
        let node_tensor_in = candidate.tensor.take();
        let mut capture_tensor: Option<NodeTensor> =
            if node_tensor_in.is_none() && tensor_capture_pays(&tensor_layout, n_node) {
                Some(NodeTensor::zeroed(&tensor_layout))
            } else {
                None
            };
        let best_split = crate::profile_time!("tree/find_best_split", {
            with_workspace_dispatched(
                params.parallel_small_image_fallback,
                n_node,
                histogram_size,
                max_buckets,
                |workspace| {
                    find_best_split(
                        samples,
                        candidate.start,
                        candidate.end,
                        histogram_size,
                        candidate.base_bits,
                        params,
                        candidate.best_predictor,
                        threshold,
                        &pq,
                        workspace,
                        match (&node_tensor_in, capture_tensor.as_mut()) {
                            (Some(t), _) => TensorMode::Use(&tensor_layout, t),
                            (None, Some(t)) => TensorMode::Capture(&tensor_layout, t),
                            (None, None) => TensorMode::Off,
                        },
                    )
                },
            )
        });

        match best_split {
            Some(split) if candidate.base_bits - split.total_bits > threshold => {
                // Perform the split: permute SoA rows in-place so that rows with
                // bucket_indices[prop][i] <= bucket_split end up in [start..mid).
                let bucket_split =
                    bucket_for_splitval(&pq.threshold_sets[split.property], split.splitval);
                // Issue #40 chunk-3c: lossless path — see `partition_node_in_place_with`.
                let abs_mid = crate::profile_time!("tree/partition", {
                    partition_node_in_place_with(
                        samples,
                        &mut pq,
                        candidate.start,
                        candidate.end,
                        split.left_count,
                        tree_learn_split::PartitionKey::Bucket {
                            prop_idx: split.property,
                            val: bucket_split as u8,
                        },
                        true,
                    )
                });

                // Create child nodes
                let lchild_idx = tree.len();
                let rchild_idx = tree.len() + 1;
                tree.push(PropertyDecisionNode::default());
                tree.push(PropertyDecisionNode::default());

                // Set split node
                tree[candidate.node_idx] = PropertyDecisionNode {
                    property: split.property as i32,
                    splitval: split.splitval,
                    lchild: lchild_idx,
                    rchild: rchild_idx,
                    ..Default::default()
                };

                // Each child's leaf cost (base_bits) for its stack entry,
                // carried out of the split sweep at the winning threshold
                // (issue #64 side-costs rider — the sweep scores every
                // sample in range, there is no sampled "eval subset", so the
                // carried costs are bitwise-identical to the 2×O(N)
                // recompute this replaces). Debug builds re-derive + assert.
                let (left_bits, right_bits) = (split.left_bits, split.right_bits);
                debug_verify_carried_side_bits(
                    samples,
                    &split,
                    candidate.start,
                    abs_mid,
                    candidate.end,
                    histogram_size,
                    &mut entropy_counts,
                );

                // PERF-HIST-SUB-LOSSLESS: with this node's tensor in hand
                // (arrived via derivation, or captured above), build the
                // smaller child's tensor and derive the larger child's by
                // subtraction. The parent tensor is consumed here either
                // way — at most two child tensors stay live per stack level.
                let node_tensor = node_tensor_in.or(capture_tensor.take());
                let (left_tensor, right_tensor) = match node_tensor {
                    Some(parent_t) => derive_child_tensors(
                        samples,
                        &pq,
                        params,
                        &tensor_layout,
                        histogram_size,
                        candidate.start,
                        abs_mid,
                        candidate.end,
                        parent_t,
                        left_bits,
                        right_bits,
                        threshold,
                    ),
                    None => (None, None),
                };

                stack.push(SplitCandidate {
                    node_idx: rchild_idx,
                    start: abs_mid,
                    end: candidate.end,
                    best_predictor: split.right_predictor,
                    base_bits: right_bits,
                    multiplier: None,
                    tensor: right_tensor,
                });

                stack.push(SplitCandidate {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: split.left_predictor,
                    base_bits: left_bits,
                    multiplier: None,
                    tensor: left_tensor,
                });
            }
            _ => {
                finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            }
        }
    }

    // Assign sequential context IDs to leaves
    assign_sequential_contexts(&mut tree);

    // Validate tree structure (matching libjxl's ValidateTree in dec_ma.cc).
    loop {
        match validate_tree_djxl(&tree) {
            Ok(()) => break,
            Err(msg) => {
                #[cfg(feature = "debug-rect")]
                eprintln!("tree/validate: fixing invalid node: {}", msg);
                let node_idx = msg
                    .strip_prefix("Node ")
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<usize>().ok())
                    .expect("validate_tree_djxl error format changed");
                tree[node_idx] = PropertyDecisionNode {
                    property: -1,
                    splitval: 0,
                    predictor: super::predictor::Predictor::Gradient,
                    predictor_offset: 0,
                    multiplier: 1,
                    lchild: 0,
                    rchild: 0,
                    context_id: 0,
                };
                assign_sequential_contexts(&mut tree);
            }
        }
    }

    let _num_leaves = tree.iter().filter(|n| n.property == -1).count();
    crate::trace::debug_eprintln!(
        "compute_best_tree: {} samples, pf={:.3}, threshold={:.1} (base={:.0}*rc={:.3}), \
         {} nodes, {} leaves, max_nodes={}",
        n,
        params.pixel_fraction,
        threshold,
        params.split_threshold,
        required_cost,
        tree.len(),
        _num_leaves,
        max_nodes,
    );

    Ok(tree)
}

/// Make a tree node into a leaf with the given predictor.
fn finalize_leaf(tree: &mut Tree, candidate: &SplitCandidate, predictors: &[Predictor]) {
    tree[candidate.node_idx] = PropertyDecisionNode {
        property: -1,
        predictor: predictors[candidate.best_predictor],
        predictor_offset: 0,
        multiplier: candidate.multiplier.unwrap_or(1) as i32,
        context_id: 0, // Will be reassigned by assign_sequential_contexts
        ..Default::default()
    };
}

// The pre-chunk-2 hardcoded constants
// (`PARALLEL_THRESHOLD=8192`, `max_parallel_depth=4`,
// `PARALLEL_RECURSION_FLOOR=16384`) now live on
// [`TreeLearningParams::parallel_root_threshold`],
// [`TreeLearningParams::parallel_max_depth`], and
// [`TreeLearningParams::parallel_recursion_floor`], wired from
// [`crate::effort::EffortProfile`] so the picker / sweep harness can
// retune them per effort without touching tree_learn.rs.
//
// At effort ≤ 7 the defaults match the original constants exactly
// (byte-identical output gated by the parallelism path). At effort ≥ 8
// the defaults are halved (depth = 5, floor = 8192, root_threshold = 4096)
// to expose finer-grained fanout for the deeper trees produced at
// Kitten/Tortoise speed tiers.

/// Greedy DFS subtree builder. Runs the same logic as the main loop in
/// [`compute_best_tree_with_budget`], but on an isolated `(samples, pq)` pair
/// representing a contiguous sample range. Returns a `Tree` rooted at a single
/// pre-allocated root node (index 0 in the returned vec).
///
/// Used by the parallel-tree-learning path: the parent does the root split,
/// clones samples + pq into two halves, then calls this twice in parallel.
/// Each call's returned `Tree` is then stitched into the parent's main tree
/// with index remapping.
///
/// `seed_predictor` / `seed_base_bits` initialise the root SplitCandidate so
/// the caller doesn't have to recompute them (the parent already had them
/// from its `find_best_split` + `compute_predictor_entropy` work).
///
/// `max_nodes_budget` caps the number of nodes this subtree may add. The
/// caller must compute it as `parent.max_nodes - parent.tree.len()` minus a
/// safety margin (e.g. divide by 2 to leave room for the sibling subtree).
///
/// Issue #42 (2026-05-25): un-`cfg(test)`'d for the small-image owned-clone
/// fallback path (`build_subtree_recursive_parallel`). The owned-clone path
/// uses this as its recursion-floor sequential leaf builder.
#[cfg(feature = "parallel-tree-learning")]
#[allow(clippy::too_many_arguments)]
fn build_subtree_sequential(
    samples: &mut TreeSamples,
    pq: &mut PreQuantizedProps,
    params: &TreeLearningParams,
    threshold: f64,
    max_nodes_budget: usize,
    histogram_size: usize,
    seed_predictor: usize,
    seed_base_bits: f64,
) -> Tree {
    let n = samples.num_samples;
    let max_buckets = params.max_property_values + 1;
    // Workspace lives in this thread's cache (see `with_thread_local_workspace`)
    // — no per-call ~12 MB allocation. Used by both the layer-2 invariant test
    // `test_parallel_tree_matches_sequential` AS WELL AS the small-image
    // owned-clone fallback path (issue #42). The borrowed-view production
    // path uses `build_subtree_sequential_borrowed`.
    let mut entropy_counts = vec![0u32; histogram_size];

    let mut tree: Tree = Vec::new();
    tree.push(PropertyDecisionNode::default()); // root, index 0

    let mut stack: Vec<SplitCandidate> = Vec::new();
    stack.push(SplitCandidate {
        node_idx: 0,
        start: 0,
        end: n,
        best_predictor: seed_predictor,
        base_bits: seed_base_bits,
        multiplier: None,
        tensor: None,
    });

    while let Some(candidate) = stack.pop() {
        if tree.len() + 2 > max_nodes_budget {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }
        let count = candidate.end - candidate.start;
        if count < 2 {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }
        if candidate.base_bits <= threshold {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }

        // PERF-HIST-SUB-LOSSLESS: TensorMode::Off — the owned small-image
        // fallback keeps the documented full-rebuild (issue #42 / the
        // `.meta` plan point 3): it only runs on inputs small enough that
        // the tensor profitability gates would not fire.
        let best_split = with_workspace_dispatched(
            params.parallel_small_image_fallback,
            count,
            histogram_size,
            max_buckets,
            |workspace| {
                find_best_split(
                    samples,
                    candidate.start,
                    candidate.end,
                    histogram_size,
                    candidate.base_bits,
                    params,
                    candidate.best_predictor,
                    threshold,
                    pq,
                    workspace,
                    TensorMode::Off,
                )
            },
        );

        match best_split {
            Some(split) if candidate.base_bits - split.total_bits > threshold => {
                let bucket_split =
                    bucket_for_splitval(&pq.threshold_sets[split.property], split.splitval);
                let abs_mid = partition_node_in_place(
                    samples,
                    pq,
                    candidate.start,
                    candidate.end,
                    split.left_count,
                    tree_learn_split::PartitionKey::Bucket {
                        prop_idx: split.property,
                        val: bucket_split as u8,
                    },
                );

                let lchild_idx = tree.len();
                let rchild_idx = tree.len() + 1;
                tree.push(PropertyDecisionNode::default());
                tree.push(PropertyDecisionNode::default());

                tree[candidate.node_idx] = PropertyDecisionNode {
                    property: split.property as i32,
                    splitval: split.splitval,
                    lchild: lchild_idx,
                    rchild: rchild_idx,
                    ..Default::default()
                };

                // Carried from the split sweep (issue #64 side-costs rider);
                // debug builds re-derive + assert bitwise identity.
                let lb = split.left_bits;
                let rb = split.right_bits;
                debug_verify_carried_side_bits(
                    samples,
                    &split,
                    candidate.start,
                    abs_mid,
                    candidate.end,
                    histogram_size,
                    &mut entropy_counts,
                );

                stack.push(SplitCandidate {
                    node_idx: rchild_idx,
                    start: abs_mid,
                    end: candidate.end,
                    best_predictor: split.right_predictor,
                    base_bits: rb,
                    multiplier: None,
                    tensor: None,
                });
                stack.push(SplitCandidate {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: split.left_predictor,
                    base_bits: lb,
                    multiplier: None,
                    tensor: None,
                });
            }
            _ => {
                finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            }
        }
    }

    tree
}

/// Splice a subtree's nodes into the parent tree, remapping child indices to
/// the parent's allocation offset. Returns the parent-tree index of the
/// subtree's root.
#[cfg(feature = "parallel-tree-learning")]
fn splice_subtree(parent: &mut Tree, subtree: Tree) -> usize {
    let offset = parent.len();
    for mut node in subtree {
        if node.property >= 0 {
            // Internal node — remap child indices to parent's allocation space.
            node.lchild += offset;
            node.rchild += offset;
        }
        parent.push(node);
    }
    offset
}

// ─── Owned-clone parallel path (issue #42, 2026-05-25 — resurrected from
//     pre-`fe2d3a27` after the borrowed-view path regressed +6.2% wall on
//     0.26 MP inputs) ───────────────────────────────────────────────────────────
//
// The borrowed-view path (introduced in `fe2d3a27`, see the section below) wins
// -4.5% to -8.1% wall on medium/large images but regresses +6.2% mean wall on
// 0.26 MP. The owned-clone path inverts that trade-off — it pays the
// `Vec::split_off` memcpy at every fork, which dominates on large inputs but
// is invisible on small ones where the borrowed-view's slice-tracking
// allocations (per-fork `Vec<&mut [u8]>` containers) and indirection cost
// outpace the saved memcpy. Dispatched at the parallel-root-fan-out site by
// [`TreeLearningParams::parallel_small_image_fallback`], which fires when
// `pixels < SMALL_IMAGE_PIXEL_THRESHOLD` (1 MP) AND `effort <= 7`.
//
// Bitstream equivalence with the borrowed-view path: the SoA permutation is
// done by the SAME `partition_node_in_place_with(..., skip_props_swap=true)`
// primitive at the root, then by `partition_node_in_place` (no skip) inside
// each owned-clone fork. `find_best_split` reads only
// residual_tokens/extra_bits/bucket_indices/sample_counts, never `props` —
// so the props-swap omission at the root is invisible regardless of which
// recursive path runs. Tree topology is data-determined; serialization is
// BFS-from-root via `collect_tree_tokens`.

/// Split a [`TreeSamples`] taken by-value into two owned halves at `mid`.
/// Resurrected from pre-`fe2d3a27` for the small-image owned-clone fallback
/// (issue #42). Used by the recursive parallel fan-out path inside
/// [`build_subtree_recursive_parallel`].
#[cfg(feature = "parallel-tree-learning")]
fn split_tree_samples_owned(mut samples: TreeSamples, mid: usize) -> (TreeSamples, TreeSamples) {
    let n = samples.num_samples;
    debug_assert!(mid <= n);

    let num_pred = samples.residual_tokens.len();
    let num_props = samples.props.len();

    let mut right_residual_tokens: Vec<Vec<u8>> = Vec::with_capacity(num_pred);
    let mut right_extra_bits: Vec<Vec<u8>> = Vec::with_capacity(num_pred);
    let mut right_props: Vec<Vec<i32>> = Vec::with_capacity(num_props);

    for v in &mut samples.residual_tokens {
        if v.is_empty() {
            right_residual_tokens.push(Vec::new());
        } else {
            right_residual_tokens.push(v.split_off(mid));
        }
    }
    for v in &mut samples.extra_bits {
        if v.is_empty() {
            right_extra_bits.push(Vec::new());
        } else {
            right_extra_bits.push(v.split_off(mid));
        }
    }
    for v in &mut samples.props {
        if v.is_empty() {
            right_props.push(Vec::new());
        } else {
            right_props.push(v.split_off(mid));
        }
    }
    let right_sample_counts = samples.sample_counts.split_off(mid);

    let right_n = n - mid;
    samples.num_samples = mid;

    let right = TreeSamples {
        num_samples: right_n,
        candidate_predictors: samples.candidate_predictors,
        residual_tokens: right_residual_tokens,
        extra_bits: right_extra_bits,
        props: right_props,
        sample_counts: right_sample_counts,
        num_ref_channels: samples.num_ref_channels,
        randomize_gather: samples.randomize_gather,
    };

    (samples, right)
}

/// Split a [`PreQuantizedProps`] taken by-value into two owned halves at `mid`.
/// `threshold_sets` is shared (cloned) — it's read-only during tree building
/// and small (≤ 16 props × ≤ 256 i32 = ~16 KB). Resurrected from pre-`fe2d3a27`
/// for the small-image owned-clone fallback (issue #42).
#[cfg(feature = "parallel-tree-learning")]
fn split_pq_owned(mut pq: PreQuantizedProps, mid: usize) -> (PreQuantizedProps, PreQuantizedProps) {
    let num_props = pq.bucket_indices.len();
    let mut right_bi: Vec<Vec<u8>> = Vec::with_capacity(num_props);
    for v in &mut pq.bucket_indices {
        if v.is_empty() {
            right_bi.push(Vec::new());
        } else {
            right_bi.push(v.split_off(mid));
        }
    }
    let right = PreQuantizedProps {
        threshold_sets: pq.threshold_sets.clone(),
        bucket_indices: right_bi,
    };
    (pq, right)
}

/// Take the SoA data out of `&mut TreeSamples` + `&mut PreQuantizedProps` via
/// [`core::mem::take`] and partition into two owned halves at `mid`. After the
/// call, the parent's parallel-array Vecs are EMPTY (their data has been moved
/// into the returned halves); other fields are intact. Callers must NOT read
/// `samples.{residual_tokens, extra_bits, props, sample_counts}` or
/// `pq.bucket_indices` after this point.
///
/// Used at the parallel-root-fan-out site to bridge the borrowed `&mut`
/// upstream context with the owned-clone recursive fork
/// ([`build_subtree_recursive_parallel`]).
#[cfg(feature = "parallel-tree-learning")]
fn split_owned_from_borrowed(
    samples: &mut TreeSamples,
    pq: &mut PreQuantizedProps,
    mid: usize,
) -> (
    (TreeSamples, PreQuantizedProps),
    (TreeSamples, PreQuantizedProps),
) {
    let n = samples.num_samples;
    debug_assert!(mid <= n);
    // Move the parallel-array data out of the parent; non-array fields stay.
    let taken_samples = TreeSamples {
        num_samples: n,
        candidate_predictors: samples.candidate_predictors,
        residual_tokens: core::mem::take(&mut samples.residual_tokens),
        extra_bits: core::mem::take(&mut samples.extra_bits),
        props: core::mem::take(&mut samples.props),
        sample_counts: core::mem::take(&mut samples.sample_counts),
        num_ref_channels: samples.num_ref_channels,
        randomize_gather: samples.randomize_gather,
    };
    let taken_pq = PreQuantizedProps {
        threshold_sets: core::mem::take(&mut pq.threshold_sets),
        bucket_indices: core::mem::take(&mut pq.bucket_indices),
    };
    let (left_samples, right_samples) = split_tree_samples_owned(taken_samples, mid);
    let (left_pq, right_pq) = split_pq_owned(taken_pq, mid);
    ((left_samples, left_pq), (right_samples, right_pq))
}

/// Owned-clone recursive divide-and-conquer subtree builder (issue #42,
/// resurrected from pre-`fe2d3a27`). At each split, optionally forks both
/// child subtree builds via `parallel_join` when the range is large enough to
/// amortise rayon task overhead AND parallel budget remains.
///
/// `parallel_budget` starts at `max_parallel_depth` and decrements per fork.
/// Bounds total rayon tasks to `2^max_parallel_depth` regardless of tree
/// shape. `max_nodes_budget` is the same hard cap as
/// [`build_subtree_sequential`].
///
/// Owned-clone strategy: at each fork, `split_off`s detach per-side data
/// into fresh allocations. Costs O(N) memcpy per level; total split cost is
/// O(N log N), below the O(N log² N) tree-search cost. On small inputs
/// (< 1 MP) this wins over the borrowed-view path because the per-fork
/// slice-tracking containers and indirection in the borrowed path outpace
/// the saved memcpy.
#[cfg(feature = "parallel-tree-learning")]
#[allow(clippy::too_many_arguments)]
fn build_subtree_recursive_parallel(
    mut samples: TreeSamples,
    mut pq: PreQuantizedProps,
    params: &TreeLearningParams,
    threshold: f64,
    max_nodes_budget: usize,
    histogram_size: usize,
    seed_predictor: usize,
    seed_base_bits: f64,
    parallel_budget: u32,
) -> Tree {
    let n = samples.num_samples;

    // Recursion floor: small subtrees go through the simpler iterative
    // sequential path with no further parallel forks.
    if parallel_budget == 0 || n < params.parallel_recursion_floor {
        return build_subtree_sequential(
            &mut samples,
            &mut pq,
            params,
            threshold,
            max_nodes_budget,
            histogram_size,
            seed_predictor,
            seed_base_bits,
        );
    }

    // Leaf-now gates.
    if n < 2 || seed_base_bits <= threshold || max_nodes_budget < 4 {
        let mut tree: Tree = alloc::vec::Vec::new();
        let leaf_candidate = SplitCandidate {
            node_idx: 0,
            start: 0,
            end: n,
            best_predictor: seed_predictor,
            base_bits: seed_base_bits,
            multiplier: None,
            tensor: None,
        };
        tree.push(PropertyDecisionNode::default());
        finalize_leaf(&mut tree, &leaf_candidate, samples.candidate_predictors);
        return tree;
    }

    // Find best split for the root of this subtree.
    // Workspace allocation strategy matches the calling context: when the
    // small-image fallback flag is set, bypass the thread-local cache (per the
    // `with_workspace_dispatched` doc — `RefCell::borrow_mut` indirection
    // outpaces the calloc savings on small inputs).
    // PERF-HIST-SUB-LOSSLESS: TensorMode::Off — owned small-image fallback
    // keeps the documented full-rebuild (`.meta` plan point 3).
    let max_buckets = params.max_property_values + 1;
    let mut entropy_counts = alloc::vec![0u32; histogram_size];

    let split = match with_workspace_dispatched(
        params.parallel_small_image_fallback,
        n,
        histogram_size,
        max_buckets,
        |workspace| {
            find_best_split(
                &samples,
                0,
                n,
                histogram_size,
                seed_base_bits,
                params,
                seed_predictor,
                threshold,
                &pq,
                workspace,
                TensorMode::Off,
            )
        },
    ) {
        Some(s) if seed_base_bits - s.total_bits > threshold => s,
        _ => {
            // No beneficial split — single-leaf subtree.
            let mut tree: Tree = alloc::vec::Vec::new();
            let leaf_candidate = SplitCandidate {
                node_idx: 0,
                start: 0,
                end: n,
                best_predictor: seed_predictor,
                base_bits: seed_base_bits,
                multiplier: None,
                tensor: None,
            };
            tree.push(PropertyDecisionNode::default());
            finalize_leaf(&mut tree, &leaf_candidate, samples.candidate_predictors);
            return tree;
        }
    };

    // Partition in-place to separate left/right. Uses the
    // `skip_props_swap=false` variant (original behaviour) for safety — the
    // owned-clone fork's recursive descendants don't read `samples.props`
    // either, but keeping the per-property swap matches the pre-`fe2d3a27`
    // baseline exactly so bitstream identity is byte-trivially provable.
    let bucket_split = bucket_for_splitval(&pq.threshold_sets[split.property], split.splitval);
    let abs_mid = partition_node_in_place(
        &mut samples,
        &mut pq,
        0,
        n,
        split.left_count,
        tree_learn_split::PartitionKey::Bucket {
            prop_idx: split.property,
            val: bucket_split as u8,
        },
    );

    // Child base bits carried from the split sweep (issue #64 side-costs
    // rider); debug builds re-derive + assert bitwise identity.
    let left_bits = split.left_bits;
    let right_bits = split.right_bits;
    debug_verify_carried_side_bits(
        &samples,
        &split,
        0,
        abs_mid,
        n,
        histogram_size,
        &mut entropy_counts,
    );

    // Free the entropy buffer before the split_off allocations. The workspace
    // is held in the per-thread cache and intentionally outlives this call —
    // sibling forks scheduled on the same worker will reuse it.
    drop(entropy_counts);

    // Split data into per-side owned halves via `Vec::split_off`.
    let (left_samples, right_samples) = split_tree_samples_owned(samples, abs_mid);
    let (left_pq, right_pq) = split_pq_owned(pq, abs_mid);

    let left_predictor = split.left_predictor;
    let right_predictor = split.right_predictor;
    let split_property = split.property as i32;
    let split_splitval = split.splitval;

    let per_side_budget = (max_nodes_budget - 1) / 2;
    let next_parallel_budget = parallel_budget - 1;

    // Decide whether to actually fork. If one side is tiny, don't bother
    // paying rayon task overhead — let it run on this thread sequentially
    // before recursing into the larger side.
    let left_size = left_samples.num_samples;
    let right_size = right_samples.num_samples;
    let parallel_floor = params.parallel_recursion_floor;
    let both_big_enough = left_size >= parallel_floor && right_size >= parallel_floor;

    let (left_tree, right_tree) = if both_big_enough {
        crate::parallel::parallel_join(
            || {
                build_subtree_recursive_parallel(
                    left_samples,
                    left_pq,
                    params,
                    threshold,
                    per_side_budget,
                    histogram_size,
                    left_predictor,
                    left_bits,
                    next_parallel_budget,
                )
            },
            || {
                build_subtree_recursive_parallel(
                    right_samples,
                    right_pq,
                    params,
                    threshold,
                    per_side_budget,
                    histogram_size,
                    right_predictor,
                    right_bits,
                    next_parallel_budget,
                )
            },
        )
    } else {
        // At least one side is small — do them sequentially (no rayon spawn).
        let l = build_subtree_recursive_parallel(
            left_samples,
            left_pq,
            params,
            threshold,
            per_side_budget,
            histogram_size,
            left_predictor,
            left_bits,
            next_parallel_budget,
        );
        let r = build_subtree_recursive_parallel(
            right_samples,
            right_pq,
            params,
            threshold,
            per_side_budget,
            histogram_size,
            right_predictor,
            right_bits,
            next_parallel_budget,
        );
        (l, r)
    };

    // Assemble the result tree: root split node + spliced subtrees.
    let mut tree: Tree = alloc::vec::Vec::new();
    tree.push(PropertyDecisionNode::default()); // root, index 0
    let lchild_idx = splice_subtree(&mut tree, left_tree);
    let rchild_idx = splice_subtree(&mut tree, right_tree);
    tree[0] = PropertyDecisionNode {
        property: split_property,
        splitval: split_splitval,
        lchild: lchild_idx,
        rchild: rchild_idx,
        ..Default::default()
    };

    tree
}

// ─── Borrowed-view parallel path (issue #41 follow-on, 2026-05-16) ─────────────
//
// `split_tree_samples_owned` / `split_pq_owned` clone the SoA arrays via
// `Vec::split_off` at every fork. On 1.05 MP at 8 threads that's ~13 MB of
// memcpy at the top fork and ~52 allocator calls (one per parallel array
// per fork). Cumulative across the recursion: ~25 MB copied and 30+ forks
// per encode.
//
// This code path replaces those clones with slice borrows. The parent fork
// calls `partition_node_in_place` (which already permutes the SoA arrays so
// the left subtree's data is contiguous in `[0..mid)` and the right's in
// `[mid..len)`), then `split_at_mut`s every parallel array at `mid` and
// constructs two `BorrowedSamples<'a>` views. Each view holds non-overlapping
// `&'a mut [u8]` slices. Both child views are handed to `rayon::join`'s
// closures, which can mutate their own halves independently. Total allocation
// per fork: zero (the small per-view `Vec<&mut [u8]>` containers reuse the
// same predictor/property count and live on the stack-allocated frame's
// thread-local Vec backing).
//
// Bitstream equivalence: the SoA permutation done inside the borrowed path
// is identical to the owned-clone path (same `partition_node_in_place`
// primitive operates on the same rows). The split-point math, find-best-split
// inner loop, and tree topology are all data-determined.

/// A borrowed view into the SoA arrays of `TreeSamples + PreQuantizedProps`,
/// holding mutable slice references rather than owning the data.
///
/// Used by the parallel-tree-learning path so each fork operates on its own
/// disjoint sub-range without cloning. The parent constructs this view from
/// the top-level `TreeSamples + PreQuantizedProps` (after the root partition),
/// then splits it in half at each fork via [`Self::split_at_mut`].
///
/// Field invariants:
/// - Every non-empty inner slice has length == `len`.
/// - Empty inner slices represent properties/predictors not gathered
///   (production [`TreeSamples`] carries empty `props[i]` for properties
///   outside `params.properties`, and [`PreQuantizedProps::bucket_indices`]
///   holds empty rows in the same slots).
/// - All non-empty arrays use the same row indexing: row `i` is sample `i`.
#[cfg(feature = "parallel-tree-learning")]
struct BorrowedSamples<'a> {
    /// Per-predictor residual tokens: `residual_tokens[pred][sample]`.
    residual_tokens: alloc::vec::Vec<&'a mut [u8]>,
    /// Per-predictor extra bits: `extra_bits[pred][sample]`.
    extra_bits: alloc::vec::Vec<&'a mut [u8]>,
    /// Per-property quantized values: `props[prop][sample]`. May be empty
    /// for properties outside `params.properties`.
    props: alloc::vec::Vec<&'a mut [i32]>,
    /// Per-property bucket indices: `bucket_indices[prop][sample]`. May be
    /// empty for properties outside `params.properties`.
    bucket_indices: alloc::vec::Vec<&'a mut [u8]>,
    /// Dedup weights: `sample_counts[sample]`.
    sample_counts: &'a mut [u32],
    /// Read-only threshold sets, shared across all forks (immutable for the
    /// duration of tree building).
    threshold_sets: &'a [alloc::vec::Vec<i32>],
    /// Candidate predictor list; mirrors `TreeSamples::candidate_predictors`.
    candidate_predictors: &'static [Predictor],
    /// Logical sample count (== slice length for non-empty parallel arrays).
    len: usize,
}

#[cfg(feature = "parallel-tree-learning")]
impl<'a> BorrowedSamples<'a> {
    /// Build a borrowed view over the entire live range of a
    /// `TreeSamples + PreQuantizedProps` pair.
    fn from_owned(samples: &'a mut TreeSamples, pq: &'a mut PreQuantizedProps) -> Self {
        let len = samples.num_samples;
        let candidate_predictors = samples.candidate_predictors;

        // Slice each parallel array to `[..len]`. Empty arrays stay empty.
        let residual_tokens: alloc::vec::Vec<&'a mut [u8]> = samples
            .residual_tokens
            .iter_mut()
            .map(|v| {
                if v.is_empty() {
                    &mut v[..]
                } else {
                    &mut v[..len]
                }
            })
            .collect();
        let extra_bits: alloc::vec::Vec<&'a mut [u8]> = samples
            .extra_bits
            .iter_mut()
            .map(|v| {
                if v.is_empty() {
                    &mut v[..]
                } else {
                    &mut v[..len]
                }
            })
            .collect();
        let props: alloc::vec::Vec<&'a mut [i32]> = samples
            .props
            .iter_mut()
            .map(|v| {
                if v.is_empty() {
                    &mut v[..]
                } else {
                    &mut v[..len]
                }
            })
            .collect();
        let bucket_indices: alloc::vec::Vec<&'a mut [u8]> = pq
            .bucket_indices
            .iter_mut()
            .map(|v| {
                if v.is_empty() {
                    &mut v[..]
                } else {
                    &mut v[..len]
                }
            })
            .collect();
        let sample_counts = &mut samples.sample_counts[..len];

        Self {
            residual_tokens,
            extra_bits,
            props,
            bucket_indices,
            sample_counts,
            threshold_sets: &pq.threshold_sets,
            candidate_predictors,
            len,
        }
    }

    /// Consume this view and produce two non-overlapping child views split
    /// at `mid`. The left child covers rows `[0..mid)`, the right `[mid..len)`.
    ///
    /// All parallel array slices are split via `split_at_mut`. The disjoint
    /// borrows can be sent to separate threads.
    fn split_at_mut(self, mid: usize) -> (BorrowedSamples<'a>, BorrowedSamples<'a>) {
        debug_assert!(mid <= self.len);
        let right_len = self.len - mid;

        // Unzip per-array splits into left/right halves.
        let mut left_residual_tokens = alloc::vec::Vec::with_capacity(self.residual_tokens.len());
        let mut right_residual_tokens = alloc::vec::Vec::with_capacity(self.residual_tokens.len());
        for slice in self.residual_tokens {
            if slice.is_empty() {
                left_residual_tokens.push(&mut [][..]);
                right_residual_tokens.push(&mut [][..]);
            } else {
                let (l, r) = slice.split_at_mut(mid);
                left_residual_tokens.push(l);
                right_residual_tokens.push(r);
            }
        }

        let mut left_extra_bits = alloc::vec::Vec::with_capacity(self.extra_bits.len());
        let mut right_extra_bits = alloc::vec::Vec::with_capacity(self.extra_bits.len());
        for slice in self.extra_bits {
            if slice.is_empty() {
                left_extra_bits.push(&mut [][..]);
                right_extra_bits.push(&mut [][..]);
            } else {
                let (l, r) = slice.split_at_mut(mid);
                left_extra_bits.push(l);
                right_extra_bits.push(r);
            }
        }

        let mut left_props = alloc::vec::Vec::with_capacity(self.props.len());
        let mut right_props = alloc::vec::Vec::with_capacity(self.props.len());
        for slice in self.props {
            if slice.is_empty() {
                left_props.push(&mut [][..]);
                right_props.push(&mut [][..]);
            } else {
                let (l, r) = slice.split_at_mut(mid);
                left_props.push(l);
                right_props.push(r);
            }
        }

        let mut left_bucket_indices = alloc::vec::Vec::with_capacity(self.bucket_indices.len());
        let mut right_bucket_indices = alloc::vec::Vec::with_capacity(self.bucket_indices.len());
        for slice in self.bucket_indices {
            if slice.is_empty() {
                left_bucket_indices.push(&mut [][..]);
                right_bucket_indices.push(&mut [][..]);
            } else {
                let (l, r) = slice.split_at_mut(mid);
                left_bucket_indices.push(l);
                right_bucket_indices.push(r);
            }
        }

        let (left_sample_counts, right_sample_counts) = self.sample_counts.split_at_mut(mid);

        let left = BorrowedSamples {
            residual_tokens: left_residual_tokens,
            extra_bits: left_extra_bits,
            props: left_props,
            bucket_indices: left_bucket_indices,
            sample_counts: left_sample_counts,
            threshold_sets: self.threshold_sets,
            candidate_predictors: self.candidate_predictors,
            len: mid,
        };
        let right = BorrowedSamples {
            residual_tokens: right_residual_tokens,
            extra_bits: right_extra_bits,
            props: right_props,
            bucket_indices: right_bucket_indices,
            sample_counts: right_sample_counts,
            threshold_sets: self.threshold_sets,
            candidate_predictors: self.candidate_predictors,
            len: right_len,
        };
        (left, right)
    }

    fn num_predictors(&self) -> usize {
        self.candidate_predictors.len()
    }

    fn num_thresholds(&self, prop_idx: usize) -> usize {
        self.threshold_sets[prop_idx].len()
    }
}

/// MABSplit Phase-0 instrumentation (issue #64 CHUNK 2 Phase 0): when the
/// `__env_var_diagnostics` build sets `JXL_MABSPLIT_DUMP=<path>`, every
/// find_best_split call (borrowed AND owned variants) appends one TSV line:
/// node weighted_total, base_bits, chosen property (-1 = no split beat
/// base), best_bits, then `prop:best_total` for every evaluated property —
/// the raw material for the Hoeffding early-stop variance analysis
/// (docs/MABSPLIT_VARIANCE_REPORT.md). Compiled out of default builds
/// entirely per the lossy-low hygiene rule.
#[cfg(feature = "__env_var_diagnostics")]
mod mabsplit_dump {
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    static SINK: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

    pub(super) fn sink() -> Option<&'static Mutex<std::fs::File>> {
        SINK.get_or_init(|| {
            let path = std::env::var("JXL_MABSPLIT_DUMP").ok()?;
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()?;
            Some(Mutex::new(f))
        })
        .as_ref()
    }

    pub(super) fn record(
        weighted_total: u32,
        base_bits: f64,
        chosen: i32,
        best_bits: f64,
        per_prop: &[(u8, f64)],
    ) {
        if let Some(m) = sink()
            && let Ok(mut f) = m.lock()
        {
            let mut line = format!("{weighted_total}\t{base_bits:.2}\t{chosen}\t{best_bits:.2}\t");
            for (i, (p, v)) in per_prop.iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                line.push_str(&format!("{p}:{v:.2}"));
            }
            line.push('\n');
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// Borrowed-view counterpart to [`find_best_split`]. Operates on the live
/// range `[start..end)` of a [`BorrowedSamples`].
///
/// Algorithm is byte-identical to [`find_best_split`]: same predictor pruning,
/// same property pruning, same bucket sweep, same predictor-change penalty,
/// same split decision. Only the input access path differs.
#[cfg(feature = "parallel-tree-learning")]
#[allow(clippy::too_many_arguments)]
fn find_best_split_borrowed(
    samples: &BorrowedSamples<'_>,
    start: usize,
    end: usize,
    histogram_size: usize,
    base_bits: f64,
    params: &TreeLearningParams,
    parent_predictor: usize,
    threshold: f64,
    ws: &mut SplitWorkspace,
    tensor_mode: TensorMode<'_>,
) -> Option<BestSplit> {
    let count = end - start;
    if count < 2 {
        return None;
    }

    let total_num_pred = samples.num_predictors();
    let mut best: Option<BestSplit> = None;
    let mut best_bits = base_bits;
    #[cfg(feature = "__env_var_diagnostics")]
    let mut mab_per_prop: Vec<(u8, f64)> = Vec::new();

    let sample_counts = &samples.sample_counts[start..end];

    let weighted_total: u32 = sample_counts.iter().sum();

    // See `find_best_split` for the tensor-mode mechanics; this is the same
    // decomposition with the borrowed access path.
    let (tensor_in, mut capture) = match tensor_mode {
        TensorMode::Off => (None, None),
        TensorMode::Use(l, t) => (Some((l, t)), None),
        TensorMode::Capture(l, t) => (None, Some((l, t))),
    };
    debug_assert!(
        (tensor_in.is_none() && capture.is_none()) || weighted_total >= TENSOR_MIN_CHILD_WEIGHT,
        "tensor modes require full predictor/property coverage (weighted_total >= 2048)"
    );
    let mut cap_totals: Vec<u32> = Vec::new();
    let mut cap_total_ebits: Vec<u64> = Vec::new();
    let mut cap_totals_done = false;
    let mut cap_single_bucket: Vec<(usize, usize)> = Vec::new();

    let change_pred_penalty = 800.0 / (100.0 + threshold);

    let weighted_idx = samples
        .candidate_predictors
        .iter()
        .position(|&p| p == Predictor::Weighted)
        .unwrap_or(usize::MAX);
    let zero_idx = CANDIDATE_PREDICTORS
        .iter()
        .position(|&p| p == Predictor::Zero)
        .unwrap_or(usize::MAX);

    let num_pred = (if weighted_total >= 2048 {
        total_num_pred
    } else if weighted_total >= 512 {
        10
    } else if weighted_total >= 64 {
        7
    } else {
        4
    })
    .min(total_num_pred);

    let effective_histo = histogram_size;
    if effective_histo == 0 {
        return None;
    }

    if capture.is_some() {
        cap_totals = vec![0u32; num_pred * effective_histo];
        cap_total_ebits = vec![0u64; num_pred];
    }

    let count_increase = ws.count_increase.as_mut_slice();
    let extra_bits_increase = ws.extra_bits_increase.as_mut_slice();
    let bucket_counts = ws.bucket_counts.as_mut_slice();
    let right_counts = ws.right_counts.as_mut_slice();
    let left_counts = ws.left_counts.as_mut_slice();
    let best_l_cost = ws.best_l_cost.as_mut_slice();
    let best_r_cost = ws.best_r_cost.as_mut_slice();
    let best_l_penalized = ws.best_l_penalized.as_mut_slice();
    let best_r_penalized = ws.best_r_penalized.as_mut_slice();
    let best_l_pred = ws.best_l_pred.as_mut_slice();
    let best_r_pred = ws.best_r_pred.as_mut_slice();
    let sorted_by_bucket = ws.sorted_by_bucket.as_mut_slice();
    let bucket_starts = ws.bucket_starts.as_mut_slice();
    let bucket_write_pos = ws.bucket_write_pos.as_mut_slice();

    let num_props = if weighted_total >= 256 {
        params.properties.len()
    } else if weighted_total >= 32 {
        params.properties.len().min(4)
    } else {
        params.properties.len().min(2)
    };

    for (prop_pos, &prop_idx) in params.properties[..num_props].iter().enumerate() {
        let num_thresholds = samples.num_thresholds(prop_idx);
        if num_thresholds == 0 {
            continue;
        }

        let pq_buckets = &samples.bucket_indices[prop_idx][start..end];
        let threshold_set = &samples.threshold_sets[prop_idx];

        let (bmin, bmax) = if let Some((layout, tensor)) = tensor_in {
            let entry = &layout.prop_entries[prop_pos];
            debug_assert_eq!(entry.prop_idx, prop_idx);
            let uni = &tensor.unique[entry.bucket_base..entry.bucket_base + entry.num_buckets];
            let Some(first) = uni.iter().position(|&c| c != 0) else {
                continue;
            };
            let last = uni
                .iter()
                .rposition(|&c| c != 0)
                .expect("nonzero unique count exists");
            (first, last)
        } else {
            let mut bmin: u8 = u8::MAX;
            let mut bmax: u8 = 0;
            for &b in pq_buckets {
                if b < bmin {
                    bmin = b;
                }
                if b > bmax {
                    bmax = b;
                }
            }
            (bmin as usize, bmax as usize)
        };
        if bmin == bmax {
            if let Some((layout, tensor)) = capture.as_mut() {
                let entry = &layout.prop_entries[prop_pos];
                debug_assert_eq!(entry.prop_idx, prop_idx);
                tensor.unique[entry.bucket_base + bmin] = count as u32;
                tensor.weighted[entry.bucket_base + bmin] = weighted_total;
                cap_single_bucket.push((prop_pos, bmin));
            }
            continue;
        }

        let local_num_buckets = bmax - bmin + 1;
        let local_num_thresholds = bmax - bmin;

        if let Some((layout, tensor)) = tensor_in {
            let entry = &layout.prop_entries[prop_pos];
            bucket_counts[..local_num_buckets].copy_from_slice(
                &tensor.weighted
                    [entry.bucket_base + bmin..entry.bucket_base + bmin + local_num_buckets],
            );
            let uni = &tensor.unique
                [entry.bucket_base + bmin..entry.bucket_base + bmin + local_num_buckets];
            bucket_starts[0] = 0;
            for b in 0..local_num_buckets {
                bucket_starts[b + 1] = bucket_starts[b] + uni[b] as usize;
            }
        } else {
            let mut unique_per_bucket = [0u32; 256];
            bucket_counts[..local_num_buckets].fill(0);
            for (offset, &b) in pq_buckets.iter().enumerate() {
                let local_b = (b as usize) - bmin;
                unique_per_bucket[local_b] += 1;
                bucket_counts[local_b] += sample_counts[offset];
            }

            bucket_starts[0] = 0;
            for b in 0..local_num_buckets {
                bucket_starts[b + 1] = bucket_starts[b] + unique_per_bucket[b] as usize;
            }

            bucket_write_pos[..local_num_buckets]
                .copy_from_slice(&bucket_starts[..local_num_buckets]);
            for (offset, &b) in pq_buckets.iter().enumerate() {
                let local_b = (b as usize) - bmin;
                sorted_by_bucket[bucket_write_pos[local_b]] = offset;
                bucket_write_pos[local_b] += 1;
            }

            if let Some((layout, tensor)) = capture.as_mut() {
                let entry = &layout.prop_entries[prop_pos];
                debug_assert_eq!(entry.prop_idx, prop_idx);
                tensor.unique
                    [entry.bucket_base + bmin..entry.bucket_base + bmin + local_num_buckets]
                    .copy_from_slice(&unique_per_bucket[..local_num_buckets]);
                tensor.weighted
                    [entry.bucket_base + bmin..entry.bucket_base + bmin + local_num_buckets]
                    .copy_from_slice(&bucket_counts[..local_num_buckets]);
            }
        }

        best_l_cost[..local_num_thresholds].fill(f64::MAX);
        best_r_cost[..local_num_thresholds].fill(f64::MAX);
        best_l_penalized[..local_num_thresholds].fill(f64::MAX);
        best_r_penalized[..local_num_thresholds].fill(f64::MAX);
        best_l_pred[..local_num_thresholds].fill(0);
        best_r_pred[..local_num_thresholds].fill(0);

        for pred in 0..num_pred {
            let mut penalty: f64 = 0.0;
            if pred != parent_predictor && parent_predictor != weighted_idx {
                penalty = change_pred_penalty;
            }
            if pred == weighted_idx {
                penalty += 1e-8;
            } else if pred == zero_idx {
                penalty -= 1e-8;
            }

            if let Some((layout, tensor)) = tensor_in {
                let entry = &layout.prop_entries[prop_pos];
                let nb = entry.num_buckets;
                for b in 0..local_num_buckets {
                    let src = entry.token_base + (pred * nb + bmin + b) * effective_histo;
                    count_increase[b * HISTO_PADDED..b * HISTO_PADDED + effective_histo]
                        .copy_from_slice(&tensor.token_counts[src..src + effective_histo]);
                    extra_bits_increase[b] =
                        tensor.ebit_sums[entry.ebit_base + pred * nb + bmin + b];
                }
            } else {
                let tokens = &samples.residual_tokens[pred][start..end];
                let ebits = &samples.extra_bits[pred][start..end];

                for b in 0..local_num_buckets {
                    count_increase[b * HISTO_PADDED..b * HISTO_PADDED + effective_histo].fill(0);
                }
                extra_bits_increase[..local_num_buckets].fill(0);

                for local_bucket in 0..local_num_buckets {
                    let bs = bucket_starts[local_bucket];
                    let be = bucket_starts[local_bucket + 1];
                    let ci_base = local_bucket * HISTO_PADDED;
                    let ci_slice = &mut count_increase[ci_base..ci_base + HISTO_PADDED];
                    let mut eb_sum: u64 = 0;
                    for &rel_off in &sorted_by_bucket[bs..be] {
                        let tok = tokens[rel_off];
                        let sc = sample_counts[rel_off];
                        ci_slice[tok as usize & HISTO_MASK] += sc;
                        eb_sum += ebits[rel_off] as u64 * sc as u64;
                    }
                    extra_bits_increase[local_bucket] = eb_sum;
                }

                if let Some((layout, tensor)) = capture.as_mut() {
                    let entry = &layout.prop_entries[prop_pos];
                    let nb = entry.num_buckets;
                    for b in 0..local_num_buckets {
                        let dst = entry.token_base + (pred * nb + bmin + b) * effective_histo;
                        tensor.token_counts[dst..dst + effective_histo].copy_from_slice(
                            &count_increase[b * HISTO_PADDED..b * HISTO_PADDED + effective_histo],
                        );
                        tensor.ebit_sums[entry.ebit_base + pred * nb + bmin + b] =
                            extra_bits_increase[b];
                    }
                }
            }

            right_counts[..effective_histo].fill(0);
            let mut right_extra: u64 = 0;
            let mut right_total: u32 = weighted_total;
            for (local_bucket, &eb) in extra_bits_increase[..local_num_buckets].iter().enumerate() {
                let ci_base = local_bucket * HISTO_PADDED;
                let ci_row = &count_increase[ci_base..ci_base + effective_histo];
                for (rc, &ci) in right_counts[..effective_histo]
                    .iter_mut()
                    .zip(ci_row.iter())
                {
                    *rc += ci;
                }
                right_extra += eb;
            }

            if capture.is_some() && !cap_totals_done {
                cap_totals[pred * effective_histo..(pred + 1) * effective_histo]
                    .copy_from_slice(&right_counts[..effective_histo]);
                cap_total_ebits[pred] = right_extra;
            }

            left_counts[..effective_histo].fill(0);
            let mut left_extra: u64 = 0;
            let mut left_total: u32 = 0;

            for local_k in 0..local_num_thresholds {
                let bc = bucket_counts[local_k];
                if bc == 0 {
                    continue;
                }

                let ci_base = local_k * HISTO_PADDED;
                let ci_row = &count_increase[ci_base..ci_base + effective_histo];
                for (i, &ci) in ci_row.iter().enumerate() {
                    if ci > 0 {
                        left_counts[i] += ci;
                        right_counts[i] -= ci;
                    }
                }
                left_extra += extra_bits_increase[local_k];
                right_extra -= extra_bits_increase[local_k];
                left_total += bc;
                right_total -= bc;

                if left_total == 0 || right_total == 0 {
                    continue;
                }

                let l_bits =
                    jxl_simd::estimate_bits_u32(&left_counts[..effective_histo], left_total)
                        + left_extra as f64;
                let r_bits =
                    jxl_simd::estimate_bits_u32(&right_counts[..effective_histo], right_total)
                        + right_extra as f64;

                if l_bits + penalty < best_l_penalized[local_k] {
                    best_l_penalized[local_k] = l_bits + penalty;
                    best_l_cost[local_k] = l_bits;
                    best_l_pred[local_k] = pred;
                }
                if r_bits + penalty < best_r_penalized[local_k] {
                    best_r_penalized[local_k] = r_bits + penalty;
                    best_r_cost[local_k] = r_bits;
                    best_r_pred[local_k] = pred;
                }
            }
        }

        if capture.is_some() {
            cap_totals_done = true;
        }

        #[cfg(feature = "__env_var_diagnostics")]
        let mut mab_prop_best = f64::MAX;
        for local_k in 0..local_num_thresholds {
            if best_l_cost[local_k] == f64::MAX || best_r_cost[local_k] == f64::MAX {
                continue;
            }

            let total = best_l_cost[local_k] + best_r_cost[local_k];
            #[cfg(feature = "__env_var_diagnostics")]
            if total < mab_prop_best {
                mab_prop_best = total;
            }

            if total < best_bits {
                best_bits = total;
                let global_k = bmin + local_k;
                let left_count = bucket_starts[local_k + 1];
                best = Some(BestSplit {
                    property: prop_idx,
                    splitval: threshold_set[global_k],
                    left_predictor: best_l_pred[local_k],
                    right_predictor: best_r_pred[local_k],
                    total_bits: total,
                    left_count,
                    left_bits: best_l_cost[local_k],
                    right_bits: best_r_cost[local_k],
                });
            }
        }
        #[cfg(feature = "__env_var_diagnostics")]
        if mabsplit_dump::sink().is_some() && mab_prop_best < f64::MAX {
            mab_per_prop.push((prop_idx as u8, mab_prop_best));
        }
    }

    if let Some((layout, tensor)) = capture.as_mut() {
        if cap_totals_done {
            for &(prop_pos, bucket) in &cap_single_bucket {
                let entry = &layout.prop_entries[prop_pos];
                let nb = entry.num_buckets;
                for pred in 0..num_pred {
                    let dst = entry.token_base + (pred * nb + bucket) * effective_histo;
                    tensor.token_counts[dst..dst + effective_histo].copy_from_slice(
                        &cap_totals[pred * effective_histo..(pred + 1) * effective_histo],
                    );
                    tensor.ebit_sums[entry.ebit_base + pred * nb + bucket] = cap_total_ebits[pred];
                }
            }
        } else {
            debug_assert!(best.is_none());
        }
    }

    #[cfg(feature = "__env_var_diagnostics")]
    mabsplit_dump::record(
        weighted_total,
        base_bits,
        best.as_ref().map_or(-1, |b| b.property as i32),
        best_bits,
        &mab_per_prop,
    );
    best
}

/// Borrowed-view counterpart to [`debug_verify_carried_side_bits`].
#[cfg(feature = "parallel-tree-learning")]
#[inline]
fn debug_verify_carried_side_bits_borrowed(
    samples: &BorrowedSamples<'_>,
    split: &BestSplit,
    start: usize,
    mid: usize,
    end: usize,
    histogram_size: usize,
    counts_buf: &mut [u32],
) {
    if cfg!(debug_assertions) {
        let lb = compute_predictor_entropy_borrowed(
            samples,
            start,
            mid,
            split.left_predictor,
            histogram_size,
            counts_buf,
        );
        let rb = compute_predictor_entropy_borrowed(
            samples,
            mid,
            end,
            split.right_predictor,
            histogram_size,
            counts_buf,
        );
        assert_eq!(
            split.left_bits.to_bits(),
            lb.to_bits(),
            "BestSplit.left_bits diverged from post-split recompute (borrowed)"
        );
        assert_eq!(
            split.right_bits.to_bits(),
            rb.to_bits(),
            "BestSplit.right_bits diverged from post-split recompute (borrowed)"
        );
    }
}

/// Borrowed-view counterpart to [`compute_predictor_entropy`].
#[cfg(feature = "parallel-tree-learning")]
fn compute_predictor_entropy_borrowed(
    samples: &BorrowedSamples<'_>,
    start: usize,
    end: usize,
    predictor_idx: usize,
    histogram_size: usize,
    counts_buf: &mut [u32],
) -> f64 {
    let tokens = &samples.residual_tokens[predictor_idx][start..end];
    let ebits = &samples.extra_bits[predictor_idx][start..end];
    let sample_counts = &samples.sample_counts[start..end];
    counts_buf[..histogram_size].fill(0);
    let mut total = 0u32;
    let mut tot_extra: u64 = 0;

    for ((&tok, &eb), &count) in tokens.iter().zip(ebits.iter()).zip(sample_counts.iter()) {
        let tok = tok as usize;
        if tok < histogram_size {
            counts_buf[tok] += count;
            total += count;
        }
        tot_extra += eb as u64 * count as u64;
    }

    jxl_simd::estimate_bits_u32(&counts_buf[..histogram_size], total) + tot_extra as f64
}

/// Borrowed-view counterpart to [`find_best_predictor`]. Currently unused
/// in the production path — the root predictor is computed once before the
/// parallel fork and runs against the Vec-based [`TreeSamples`]. Kept for
/// future use (e.g. if the root predictor selection moves into the parallel
/// path or for tests).
#[cfg(feature = "parallel-tree-learning")]
#[allow(dead_code)]
fn find_best_predictor_borrowed(
    samples: &BorrowedSamples<'_>,
    start: usize,
    end: usize,
    histogram_size: usize,
    counts_buf: &mut [u32],
) -> usize {
    let num_pred = samples.num_predictors();
    let mut best_pred = 0;
    let mut best_bits = f64::MAX;

    for pred_idx in 0..num_pred {
        let bits = compute_predictor_entropy_borrowed(
            samples,
            start,
            end,
            pred_idx,
            histogram_size,
            counts_buf,
        );
        if bits < best_bits {
            best_bits = bits;
            best_pred = pred_idx;
        }
    }

    best_pred
}

/// Borrowed-view counterpart to [`partition_node_in_place`]. Permutes rows
/// in-place across all parallel array slices held by `samples`.
///
/// Issue #40 chunk-3c: the lossless parallel-tree-learning path is also
/// Bucket-only and never reads `samples.props` after pre-quantize — pass
/// `skip_props_swap=true` to elide the per-property `Vec<i32>` swaps.
#[cfg(feature = "parallel-tree-learning")]
fn partition_node_in_place_borrowed(
    samples: &mut BorrowedSamples<'_>,
    start: usize,
    end: usize,
    left_count: usize,
    prop_idx: usize,
    bucket_split: u8,
    skip_props_swap: bool,
) -> usize {
    debug_assert!(left_count <= end - start);
    let pos = start + left_count;
    // Process-cached env-var override (`JXL_DISABLE_CHUNK3C=1`) — see
    // `chunk3c_skip_is_disabled` doc for rationale.
    let skip_props_swap = skip_props_swap && !chunk3c_skip_is_disabled();
    swap_partition_borrowed(
        samples,
        start,
        pos,
        end,
        prop_idx,
        bucket_split,
        skip_props_swap,
    );
    pos
}

/// Hoare-style in-place partition over a [`BorrowedSamples`]. Mirrors
/// [`tree_learn_split::split_tree_samples_in_place`] but operates on borrowed
/// slices instead of a `SplittableSamples` view bundling `&mut Vec<...>`.
///
/// All parallel array slices are permuted as atomic rows; row alignment
/// across the partition boundary is preserved.
#[cfg(feature = "parallel-tree-learning")]
fn swap_partition_borrowed(
    samples: &mut BorrowedSamples<'_>,
    begin: usize,
    pos: usize,
    end: usize,
    prop_idx: usize,
    bucket_split: u8,
    skip_props_swap: bool,
) {
    debug_assert!(begin <= pos);
    debug_assert!(pos <= end);
    debug_assert!(end <= samples.len);

    let mut begin_pos = begin;
    let mut end_pos = pos;

    loop {
        // Skip rows already on the correct left side.
        while begin_pos < pos && samples.bucket_indices[prop_idx][begin_pos] <= bucket_split {
            begin_pos += 1;
        }
        // Skip rows already on the correct right side.
        while end_pos < end && samples.bucket_indices[prop_idx][end_pos] > bucket_split {
            end_pos += 1;
        }
        if begin_pos < pos && end_pos < end {
            swap_rows_borrowed(samples, begin_pos, end_pos, skip_props_swap);
        }
        begin_pos += 1;
        end_pos += 1;
        if begin_pos >= pos || end_pos >= end {
            break;
        }
    }
}

/// Swap row `a` with row `b` across every non-empty parallel array slice.
///
/// Issue #40 chunk-3c: when `skip_props_swap` is `true`, the per-property
/// `&mut [i32]` swaps are elided. Same safety conditions as
/// [`tree_learn_split::SplittableSamples::skip_props_swap`] — the caller
/// must guarantee no downstream consumer reads `samples.props`.
#[cfg(feature = "parallel-tree-learning")]
fn swap_rows_borrowed(
    samples: &mut BorrowedSamples<'_>,
    a: usize,
    b: usize,
    skip_props_swap: bool,
) {
    if a == b {
        return;
    }
    for row in samples.residual_tokens.iter_mut() {
        if !row.is_empty() {
            row.swap(a, b);
        }
    }
    for row in samples.extra_bits.iter_mut() {
        if !row.is_empty() {
            row.swap(a, b);
        }
    }
    if !skip_props_swap {
        for row in samples.props.iter_mut() {
            if !row.is_empty() {
                row.swap(a, b);
            }
        }
    }
    for row in samples.bucket_indices.iter_mut() {
        if !row.is_empty() {
            row.swap(a, b);
        }
    }
    samples.sample_counts.swap(a, b);
}

/// Borrowed-view counterpart to [`build_subtree_sequential`]. Identical
/// algorithm; only the data access path differs.
#[cfg(feature = "parallel-tree-learning")]
#[allow(clippy::too_many_arguments)]
fn build_subtree_sequential_borrowed(
    samples: &mut BorrowedSamples<'_>,
    params: &TreeLearningParams,
    threshold: f64,
    max_nodes_budget: usize,
    histogram_size: usize,
    seed_predictor: usize,
    seed_base_bits: f64,
    tensor_layout: &TensorLayout,
    root_tensor: Option<NodeTensor>,
) -> Tree {
    let n = samples.len;
    let max_buckets = params.max_property_values + 1;
    let mut entropy_counts = alloc::vec![0u32; histogram_size];

    let mut tree: Tree = alloc::vec::Vec::new();
    tree.push(PropertyDecisionNode::default());

    let mut stack: alloc::vec::Vec<SplitCandidate> = alloc::vec::Vec::new();
    stack.push(SplitCandidate {
        node_idx: 0,
        start: 0,
        end: n,
        best_predictor: seed_predictor,
        base_bits: seed_base_bits,
        multiplier: None,
        tensor: root_tensor,
    });

    while let Some(mut candidate) = stack.pop() {
        if tree.len() + 2 > max_nodes_budget {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }
        let count = candidate.end - candidate.start;
        if count < 2 {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }
        if candidate.base_bits <= threshold {
            finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            continue;
        }

        // PERF-HIST-SUB-LOSSLESS: same Use/Capture/Off dispatch as the main
        // sequential loop in `compute_best_tree_with_budget`.
        let node_tensor_in = candidate.tensor.take();
        let mut capture_tensor: Option<NodeTensor> =
            if node_tensor_in.is_none() && tensor_capture_pays(tensor_layout, count) {
                Some(NodeTensor::zeroed(tensor_layout))
            } else {
                None
            };
        let best_split = with_workspace_dispatched(
            params.parallel_small_image_fallback,
            count,
            histogram_size,
            max_buckets,
            |workspace| {
                find_best_split_borrowed(
                    samples,
                    candidate.start,
                    candidate.end,
                    histogram_size,
                    candidate.base_bits,
                    params,
                    candidate.best_predictor,
                    threshold,
                    workspace,
                    match (&node_tensor_in, capture_tensor.as_mut()) {
                        (Some(t), _) => TensorMode::Use(tensor_layout, t),
                        (None, Some(t)) => TensorMode::Capture(tensor_layout, t),
                        (None, None) => TensorMode::Off,
                    },
                )
            },
        );

        match best_split {
            Some(split) if candidate.base_bits - split.total_bits > threshold => {
                let bucket_split =
                    bucket_for_splitval(&samples.threshold_sets[split.property], split.splitval);
                // Issue #40 chunk-3c: borrowed lossless path — see
                // `partition_node_in_place_borrowed` doc.
                let abs_mid = partition_node_in_place_borrowed(
                    samples,
                    candidate.start,
                    candidate.end,
                    split.left_count,
                    split.property,
                    bucket_split as u8,
                    true,
                );

                let lchild_idx = tree.len();
                let rchild_idx = tree.len() + 1;
                tree.push(PropertyDecisionNode::default());
                tree.push(PropertyDecisionNode::default());

                tree[candidate.node_idx] = PropertyDecisionNode {
                    property: split.property as i32,
                    splitval: split.splitval,
                    lchild: lchild_idx,
                    rchild: rchild_idx,
                    ..Default::default()
                };

                // Carried from the split sweep (issue #64 side-costs rider);
                // debug builds re-derive + assert bitwise identity.
                let lb = split.left_bits;
                let rb = split.right_bits;
                debug_verify_carried_side_bits_borrowed(
                    samples,
                    &split,
                    candidate.start,
                    abs_mid,
                    candidate.end,
                    histogram_size,
                    &mut entropy_counts,
                );

                // PERF-HIST-SUB-LOSSLESS: build smaller child / derive
                // larger child from this node's tensor (see the main loop).
                let node_tensor = node_tensor_in.or(capture_tensor.take());
                let (left_tensor, right_tensor) = match node_tensor {
                    Some(parent_t) => derive_child_tensors_borrowed(
                        samples,
                        params,
                        tensor_layout,
                        histogram_size,
                        candidate.start,
                        abs_mid,
                        candidate.end,
                        parent_t,
                        lb,
                        rb,
                        threshold,
                    ),
                    None => (None, None),
                };

                stack.push(SplitCandidate {
                    node_idx: rchild_idx,
                    start: abs_mid,
                    end: candidate.end,
                    best_predictor: split.right_predictor,
                    base_bits: rb,
                    multiplier: None,
                    tensor: right_tensor,
                });
                stack.push(SplitCandidate {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: split.left_predictor,
                    base_bits: lb,
                    multiplier: None,
                    tensor: left_tensor,
                });
            }
            _ => {
                finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
            }
        }
    }

    tree
}

/// Borrowed-view counterpart to [`build_subtree_recursive_parallel`].
///
/// Owned-clone path called `split_tree_samples_owned` + `split_pq_owned` at
/// every fork — ~52 `Vec::split_off`s totalling 10s of MB of memcpy. The
/// borrowed-view path consumes the parent `BorrowedSamples` and splits it
/// via [`BorrowedSamples::split_at_mut`], which costs only N slice splits
/// (one per parallel array) — no memcpy, no allocator pressure.
#[cfg(feature = "parallel-tree-learning")]
#[allow(clippy::too_many_arguments)]
fn build_subtree_recursive_parallel_borrowed(
    mut samples: BorrowedSamples<'_>,
    params: &TreeLearningParams,
    threshold: f64,
    max_nodes_budget: usize,
    histogram_size: usize,
    seed_predictor: usize,
    seed_base_bits: f64,
    parallel_budget: u32,
    tensor_layout: &TensorLayout,
    tensor: Option<NodeTensor>,
) -> Tree {
    let n = samples.len;

    if parallel_budget == 0 || n < params.parallel_recursion_floor {
        return build_subtree_sequential_borrowed(
            &mut samples,
            params,
            threshold,
            max_nodes_budget,
            histogram_size,
            seed_predictor,
            seed_base_bits,
            tensor_layout,
            tensor,
        );
    }

    if n < 2 || seed_base_bits <= threshold || max_nodes_budget < 4 {
        let mut tree: Tree = alloc::vec::Vec::new();
        let leaf_candidate = SplitCandidate {
            node_idx: 0,
            start: 0,
            end: n,
            best_predictor: seed_predictor,
            base_bits: seed_base_bits,
            multiplier: None,
            tensor: None,
        };
        tree.push(PropertyDecisionNode::default());
        finalize_leaf(&mut tree, &leaf_candidate, samples.candidate_predictors);
        return tree;
    }

    let max_buckets = params.max_property_values + 1;
    let mut entropy_counts = alloc::vec![0u32; histogram_size];

    // PERF-HIST-SUB-LOSSLESS: same Use/Capture/Off dispatch as the
    // sequential engines; this node's tensor (input or captured) seeds the
    // per-fork child tensors below.
    let node_tensor_in = tensor;
    let mut capture_tensor: Option<NodeTensor> =
        if node_tensor_in.is_none() && tensor_capture_pays(tensor_layout, n) {
            Some(NodeTensor::zeroed(tensor_layout))
        } else {
            None
        };
    let split = match with_workspace_dispatched(
        params.parallel_small_image_fallback,
        n,
        histogram_size,
        max_buckets,
        |workspace| {
            find_best_split_borrowed(
                &samples,
                0,
                n,
                histogram_size,
                seed_base_bits,
                params,
                seed_predictor,
                threshold,
                workspace,
                match (&node_tensor_in, capture_tensor.as_mut()) {
                    (Some(t), _) => TensorMode::Use(tensor_layout, t),
                    (None, Some(t)) => TensorMode::Capture(tensor_layout, t),
                    (None, None) => TensorMode::Off,
                },
            )
        },
    ) {
        Some(s) if seed_base_bits - s.total_bits > threshold => s,
        _ => {
            let mut tree: Tree = alloc::vec::Vec::new();
            let leaf_candidate = SplitCandidate {
                node_idx: 0,
                start: 0,
                end: n,
                best_predictor: seed_predictor,
                base_bits: seed_base_bits,
                multiplier: None,
                tensor: None,
            };
            tree.push(PropertyDecisionNode::default());
            finalize_leaf(&mut tree, &leaf_candidate, samples.candidate_predictors);
            return tree;
        }
    };

    let bucket_split = bucket_for_splitval(&samples.threshold_sets[split.property], split.splitval);
    // Issue #40 chunk-3c: borrowed lossless root-split — see
    // `partition_node_in_place_borrowed` doc.
    let abs_mid = partition_node_in_place_borrowed(
        &mut samples,
        0,
        n,
        split.left_count,
        split.property,
        bucket_split as u8,
        true,
    );

    // Carried from the split sweep (issue #64 side-costs rider); debug
    // builds re-derive + assert bitwise identity.
    let left_bits = split.left_bits;
    let right_bits = split.right_bits;
    debug_verify_carried_side_bits_borrowed(
        &samples,
        &split,
        0,
        abs_mid,
        n,
        histogram_size,
        &mut entropy_counts,
    );

    drop(entropy_counts);

    // PERF-HIST-SUB-LOSSLESS: derive the child tensors BEFORE the view is
    // consumed by `split_at_mut` (the smaller child's build pass reads the
    // whole view); each fork then owns exactly one child tensor — clean
    // per-branch ownership across `parallel_join`.
    let node_tensor = node_tensor_in.or(capture_tensor.take());
    let (left_tensor, right_tensor) = match node_tensor {
        Some(parent_t) => derive_child_tensors_borrowed(
            &samples,
            params,
            tensor_layout,
            histogram_size,
            0,
            abs_mid,
            n,
            parent_t,
            left_bits,
            right_bits,
            threshold,
        ),
        None => (None, None),
    };

    // Split the borrowed view into two non-overlapping child views at
    // abs_mid. Zero allocations beyond the per-side Vec<&mut [_]> containers
    // (small: one entry per predictor/property, ~30 total).
    let (left_samples, right_samples) = samples.split_at_mut(abs_mid);

    let left_predictor = split.left_predictor;
    let right_predictor = split.right_predictor;
    let split_property = split.property as i32;
    let split_splitval = split.splitval;

    let per_side_budget = (max_nodes_budget - 1) / 2;
    let next_parallel_budget = parallel_budget - 1;

    let left_size = left_samples.len;
    let right_size = right_samples.len;
    let parallel_floor = params.parallel_recursion_floor;
    let both_big_enough = left_size >= parallel_floor && right_size >= parallel_floor;

    let (left_tree, right_tree) = if both_big_enough {
        crate::parallel::parallel_join(
            || {
                build_subtree_recursive_parallel_borrowed(
                    left_samples,
                    params,
                    threshold,
                    per_side_budget,
                    histogram_size,
                    left_predictor,
                    left_bits,
                    next_parallel_budget,
                    tensor_layout,
                    left_tensor,
                )
            },
            || {
                build_subtree_recursive_parallel_borrowed(
                    right_samples,
                    params,
                    threshold,
                    per_side_budget,
                    histogram_size,
                    right_predictor,
                    right_bits,
                    next_parallel_budget,
                    tensor_layout,
                    right_tensor,
                )
            },
        )
    } else {
        let l = build_subtree_recursive_parallel_borrowed(
            left_samples,
            params,
            threshold,
            per_side_budget,
            histogram_size,
            left_predictor,
            left_bits,
            next_parallel_budget,
            tensor_layout,
            left_tensor,
        );
        let r = build_subtree_recursive_parallel_borrowed(
            right_samples,
            params,
            threshold,
            per_side_budget,
            histogram_size,
            right_predictor,
            right_bits,
            next_parallel_budget,
            tensor_layout,
            right_tensor,
        );
        (l, r)
    };

    let mut tree: Tree = alloc::vec::Vec::new();
    tree.push(PropertyDecisionNode::default());
    let lchild_idx = splice_subtree(&mut tree, left_tree);
    let rchild_idx = splice_subtree(&mut tree, right_tree);
    tree[0] = PropertyDecisionNode {
        property: split_property,
        splitval: split_splitval,
        lchild: lchild_idx,
        rchild: rchild_idx,
        ..Default::default()
    };

    tree
}

/// Learn an optimal MA tree with forced splits for lossy modular quantization.
///
/// Like [`compute_best_tree`] but additionally:
/// 1. Tracks `static_prop_range` (channel, group_id ranges) per node
/// 2. Before normal split evaluation, checks each `multiplier_info` entry:
///    - `Inside` → set the leaf's multiplier and finalize immediately
///    - `Partial` → force a split on the boundary axis/value
///    - `None` → skip this entry
/// 3. Only falls back to normal entropy-based splitting if no forced split applies
///
/// This produces a tree where each leaf's multiplier matches the channel's quantizer,
/// which is required for the `residual / multiplier` division to be exact.
pub fn compute_best_tree_with_multipliers(
    samples: &mut TreeSamples,
    params: &TreeLearningParams,
    multiplier_info: &[super::quantize::ModularMultiplierInfo],
    initial_range: [[u32; 2]; 2],
) -> Tree {
    use super::quantize::{IntersectionType, box_intersects};

    let required_cost = params.pixel_fraction * 0.9 + 0.1;
    let threshold = params.split_threshold * required_cost;
    let n = samples.num_samples;
    if n == 0 {
        return vec![PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Zero,
            context_id: 0,
            multiplier: 1,
            ..Default::default()
        }];
    }

    let mut pq = samples.pre_quantize(params);
    dedup_samples(samples, &mut pq, params);
    let n = samples.num_samples;

    let max_nodes = params.max_nodes;

    let max_token = samples
        .residual_tokens
        .iter()
        .flat_map(|v| v.iter())
        .copied()
        .max()
        .unwrap_or(0) as usize;
    let histogram_size = max_token + 1;

    let mut tree: Tree = Vec::new();
    let mut entropy_counts = vec![0u32; histogram_size];

    let root_predictor = find_best_predictor(samples, 0, n, histogram_size, &mut entropy_counts);
    let root_bits = compute_predictor_entropy(
        samples,
        0,
        n,
        root_predictor,
        histogram_size,
        &mut entropy_counts,
    );

    struct SplitCandidateWithRange {
        node_idx: usize,
        start: usize,
        end: usize,
        best_predictor: usize,
        base_bits: f64,
        static_prop_range: [[u32; 2]; 2],
    }

    let mut stack: Vec<SplitCandidateWithRange> = Vec::new();

    tree.push(PropertyDecisionNode::default());
    stack.push(SplitCandidateWithRange {
        node_idx: 0,
        start: 0,
        end: n,
        best_predictor: root_predictor,
        base_bits: root_bits,
        static_prop_range: initial_range,
    });

    let max_buckets = params.max_property_values + 1;
    // Workspace lives in the thread-local cache (see `with_thread_local_workspace`).

    while let Some(candidate) = stack.pop() {
        if candidate.end <= candidate.start {
            continue;
        }

        // Check multiplier_info for forced splits or direct multiplier assignment
        let mut forced_split: Option<(usize, u32)> = None; // (axis, val)
        let mut assigned_multiplier: Option<u32> = None;

        for mmi in multiplier_info {
            let (t, axis, val) = box_intersects(&candidate.static_prop_range, &mmi.range);
            match t {
                IntersectionType::None => continue,
                IntersectionType::Inside => {
                    assigned_multiplier = Some(mmi.multiplier);
                    break;
                }
                IntersectionType::Partial => {
                    forced_split = Some((axis, val));
                    break;
                }
            }
        }

        // If multiplier fully determined, finalize as leaf.
        // Force Zero predictor when multiplier > 1 to guarantee the
        // divisibility invariant: prediction=0 means residual=pixel,
        // and pixels are pre-quantized to multiples of q.
        if let Some(mult) = assigned_multiplier {
            let predictor = if mult > 1 {
                Predictor::Zero
            } else {
                CANDIDATE_PREDICTORS[candidate.best_predictor]
            };
            tree[candidate.node_idx] = PropertyDecisionNode {
                property: -1,
                predictor,
                predictor_offset: 0,
                multiplier: mult as i32,
                context_id: 0,
                ..Default::default()
            };
            continue;
        }

        // If forced split needed, do it without entropy evaluation
        if let Some((axis, splitval)) = forced_split {
            if tree.len() + 2 > max_nodes {
                // Can't split further, finalize
                tree[candidate.node_idx] = PropertyDecisionNode {
                    property: -1,
                    predictor: CANDIDATE_PREDICTORS[candidate.best_predictor],
                    predictor_offset: 0,
                    multiplier: 1,
                    context_id: 0,
                    ..Default::default()
                };
                continue;
            }

            // Partition samples on the static property (0=channel, 1=group_id).
            // Static props are matched by raw value (not bucket index), so we
            // count matching samples on the fly — no sweep produced left_count.
            let splitval_i32 = splitval as i32;
            let left_count = samples.props[axis][candidate.start..candidate.end]
                .iter()
                .filter(|&&v| v <= splitval_i32)
                .count();
            let abs_mid = partition_node_in_place(
                samples,
                &mut pq,
                candidate.start,
                candidate.end,
                left_count,
                tree_learn_split::PartitionKey::Property {
                    prop_idx: axis,
                    val: splitval_i32,
                },
            );

            let lchild_idx = tree.len();
            let rchild_idx = tree.len() + 1;
            tree.push(PropertyDecisionNode::default());
            tree.push(PropertyDecisionNode::default());

            tree[candidate.node_idx] = PropertyDecisionNode {
                property: axis as i32,
                splitval: splitval as i32,
                lchild: lchild_idx,
                rchild: rchild_idx,
                ..Default::default()
            };

            // Narrow ranges for children
            // lchild = property <= splitval: range[axis][1] = splitval + 1
            let mut lchild_range = candidate.static_prop_range;
            lchild_range[axis][1] = splitval + 1;

            // rchild = property > splitval: range[axis][0] = splitval + 1
            let mut rchild_range = candidate.static_prop_range;
            rchild_range[axis][0] = splitval + 1;

            // Compute predictors for children
            let left_predictor = if abs_mid > candidate.start {
                find_best_predictor(
                    samples,
                    candidate.start,
                    abs_mid,
                    histogram_size,
                    &mut entropy_counts,
                )
            } else {
                candidate.best_predictor
            };
            let right_predictor = if abs_mid < candidate.end {
                find_best_predictor(
                    samples,
                    abs_mid,
                    candidate.end,
                    histogram_size,
                    &mut entropy_counts,
                )
            } else {
                candidate.best_predictor
            };

            let left_bits = if abs_mid > candidate.start {
                compute_predictor_entropy(
                    samples,
                    candidate.start,
                    abs_mid,
                    left_predictor,
                    histogram_size,
                    &mut entropy_counts,
                )
            } else {
                0.0
            };
            let right_bits = if abs_mid < candidate.end {
                compute_predictor_entropy(
                    samples,
                    abs_mid,
                    candidate.end,
                    right_predictor,
                    histogram_size,
                    &mut entropy_counts,
                )
            } else {
                0.0
            };

            // Push right first (LIFO), so left is processed first
            stack.push(SplitCandidateWithRange {
                node_idx: rchild_idx,
                start: abs_mid,
                end: candidate.end,
                best_predictor: right_predictor,
                base_bits: right_bits,
                static_prop_range: rchild_range,
            });
            stack.push(SplitCandidateWithRange {
                node_idx: lchild_idx,
                start: candidate.start,
                end: abs_mid,
                best_predictor: left_predictor,
                base_bits: left_bits,
                static_prop_range: lchild_range,
            });
            continue;
        }

        // No forced split — proceed with normal entropy-based splitting
        if tree.len() + 2 > max_nodes {
            tree[candidate.node_idx] = PropertyDecisionNode {
                property: -1,
                predictor: CANDIDATE_PREDICTORS[candidate.best_predictor],
                predictor_offset: 0,
                multiplier: 1,
                context_id: 0,
                ..Default::default()
            };
            continue;
        }

        let count = candidate.end - candidate.start;
        if count < 2 || candidate.base_bits <= threshold {
            tree[candidate.node_idx] = PropertyDecisionNode {
                property: -1,
                predictor: CANDIDATE_PREDICTORS[candidate.best_predictor],
                predictor_offset: 0,
                multiplier: 1,
                context_id: 0,
                ..Default::default()
            };
            continue;
        }

        let best_split = with_workspace_dispatched(
            params.parallel_small_image_fallback,
            count,
            histogram_size,
            max_buckets,
            |workspace| {
                find_best_split(
                    samples,
                    candidate.start,
                    candidate.end,
                    histogram_size,
                    candidate.base_bits,
                    params,
                    candidate.best_predictor,
                    threshold,
                    &pq,
                    workspace,
                    // Lossy modular quantization path — full-rebuild (the
                    // forced-split structure rarely produces tensor-sized
                    // nodes; out of PERF-HIST-SUB-LOSSLESS scope).
                    TensorMode::Off,
                )
            },
        );

        match best_split {
            Some(split) if candidate.base_bits - split.total_bits > threshold => {
                let bucket_split =
                    bucket_for_splitval(&pq.threshold_sets[split.property], split.splitval);
                let abs_mid = partition_node_in_place(
                    samples,
                    &mut pq,
                    candidate.start,
                    candidate.end,
                    split.left_count,
                    tree_learn_split::PartitionKey::Bucket {
                        prop_idx: split.property,
                        val: bucket_split as u8,
                    },
                );

                let lchild_idx = tree.len();
                let rchild_idx = tree.len() + 1;
                tree.push(PropertyDecisionNode::default());
                tree.push(PropertyDecisionNode::default());

                tree[candidate.node_idx] = PropertyDecisionNode {
                    property: split.property as i32,
                    splitval: split.splitval,
                    lchild: lchild_idx,
                    rchild: rchild_idx,
                    ..Default::default()
                };

                // Narrow static_prop_range if split is on a static property
                let mut lchild_range = candidate.static_prop_range;
                let mut rchild_range = candidate.static_prop_range;
                if split.property < 2 {
                    // Static property (channel or group_id)
                    lchild_range[split.property][1] =
                        (split.splitval + 1).min(lchild_range[split.property][1] as i32) as u32;
                    rchild_range[split.property][0] =
                        (split.splitval + 1).max(rchild_range[split.property][0] as i32) as u32;
                }

                // Carried from the split sweep (issue #64 side-costs rider);
                // debug builds re-derive + assert bitwise identity.
                let left_bits = split.left_bits;
                let right_bits = split.right_bits;
                debug_verify_carried_side_bits(
                    samples,
                    &split,
                    candidate.start,
                    abs_mid,
                    candidate.end,
                    histogram_size,
                    &mut entropy_counts,
                );

                stack.push(SplitCandidateWithRange {
                    node_idx: rchild_idx,
                    start: abs_mid,
                    end: candidate.end,
                    best_predictor: split.right_predictor,
                    base_bits: right_bits,
                    static_prop_range: rchild_range,
                });
                stack.push(SplitCandidateWithRange {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: split.left_predictor,
                    base_bits: left_bits,
                    static_prop_range: lchild_range,
                });
            }
            _ => {
                tree[candidate.node_idx] = PropertyDecisionNode {
                    property: -1,
                    predictor: CANDIDATE_PREDICTORS[candidate.best_predictor],
                    predictor_offset: 0,
                    multiplier: 1,
                    context_id: 0,
                    ..Default::default()
                };
            }
        }
    }

    // Assign sequential context IDs to leaves
    assign_sequential_contexts(&mut tree);

    // Validate tree structure
    loop {
        match validate_tree_djxl(&tree) {
            Ok(()) => break,
            Err(msg) => {
                #[cfg(feature = "debug-rect")]
                eprintln!("tree/validate: fixing invalid node: {}", msg);
                let node_idx = msg
                    .strip_prefix("Node ")
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<usize>().ok())
                    .expect("validate_tree_djxl error format changed");
                tree[node_idx] = PropertyDecisionNode {
                    property: -1,
                    splitval: 0,
                    predictor: Predictor::Gradient,
                    predictor_offset: 0,
                    multiplier: 1,
                    lchild: 0,
                    rchild: 0,
                    context_id: 0,
                };
                assign_sequential_contexts(&mut tree);
            }
        }
    }

    let _num_leaves = tree.iter().filter(|n| n.property == -1).count();
    crate::trace::debug_eprintln!(
        "compute_best_tree_with_multipliers: {} samples, {} nodes, {} leaves, {} mul_info entries",
        n,
        tree.len(),
        _num_leaves,
        multiplier_info.len(),
    );

    tree
}

/// Padded histogram size for `count_increase`: power-of-2 stride above the
/// maximum token `GATHER_HYBRID_UINT.encode` can produce. Bitmask indexing
/// (`tok & HISTO_MASK`) is bounds-check-free given `tok < HISTO_PADDED`.
///
/// Bound derivation for `GATHER_HYBRID_UINT { split=16, m=1, l=2 }`:
/// `token = 16 + 8·n_extra + 4·token_shift + low_bits`. With u32 input,
/// `value_shifted ≤ 2^30 − 1`, `value_bits ≤ 30`, `n ≤ 28`, `n_extra ≤ 27`,
/// so `token ≤ 16 + 216 + 4 + 3 = 239`. The previous size (128) was
/// reachable past via RCT/Squeeze-amplified 16-bit residuals — closed
/// by the bump to 256 (security audit H3).
const HISTO_PADDED: usize = 256;
const HISTO_MASK: usize = HISTO_PADDED - 1;

// ───────────────────────────────────────────────────────────────────────────
// PERF-HIST-SUB-LOSSLESS (issue #64 chunk 1, plan in
// `benchmarks/perf_hist_sub_2026-06-10.meta`): parent-histogram subtraction.
//
// `find_best_split` spends ~87% of its time in per-sample loops (the
// counting sort + the per-(prop,pred) token/extra-bits accumulate). All of
// those loops compute *additive integer aggregates* over the node's sample
// rows: per-(prop, pred, bucket, token) u32 counts, per-(prop, pred, bucket)
// u64 extra-bit sums, and per-(prop, bucket) u32 weighted / unique counts.
// Additivity over disjoint row sets means a child node's aggregates can be
// derived exactly: `larger_child = parent − smaller_child` (u32/u64 integer
// subtraction — no rounding, no float). The engines build the smaller
// child's tensor from its samples and derive the larger child's by
// subtraction, so the larger child's `find_best_split` skips its per-sample
// loops entirely.
//
// Byte-identity argument: `estimate_bits_u32` consumes the per-bucket u32
// histograms + u32 totals + u64 extra-bit sums. Derived tensors contain
// bit-identical integers to from-scratch tensors (proven by
// `test_node_tensor_*` below + hash-locks), so every f64 cost — and hence
// every split decision — is identical. Speed-only change.
//
// Scope: the Vec-based sequential engine (`compute_best_tree_with_budget`),
// the borrowed sequential engine (`build_subtree_sequential_borrowed`) and
// the borrowed recursive-parallel engine
// (`build_subtree_recursive_parallel_borrowed`). The owned small-image
// fallback path (issue #42, `build_subtree_{sequential,recursive_parallel}`)
// keeps the documented full-rebuild: it only runs on small inputs where the
// tensors never pass the profitability gate. `compute_best_tree_with_
// multipliers` (lossy modular quantization) is also full-rebuild.
// ───────────────────────────────────────────────────────────────────────────

/// Minimum *weighted* sample count for a child to receive a tensor.
///
/// At `weighted_total >= 2048`, `find_best_split` uses the FULL predictor
/// set and (since 2048 ≥ 256) the FULL property set, so a parent at
/// ≥ 2·2048 and both children at ≥ 2048 evaluate identical (prop, pred)
/// coverage — the precondition for tensor capture/use to be exhaustive.
const TENSOR_MIN_CHILD_WEIGHT: u32 = 2048;

/// Per-property addressing for [`NodeTensor`] storage. Entries are indexed
/// by *position* in `params.properties` (the same order `find_best_split`
/// iterates), not by raw property index.
struct TensorPropEntry {
    /// Raw property index (== `params.properties[pos]`); for debug asserts.
    prop_idx: usize,
    /// `num_thresholds + 1` absolute buckets, or 0 when the property has no
    /// thresholds at all (globally degenerate — skipped by every engine
    /// before any tensor access, so zero cells are reserved).
    num_buckets: usize,
    /// Base offset into [`NodeTensor::token_counts`]; the row for
    /// `(pred, bucket)` starts at `token_base + (pred*num_buckets + bucket)*histo`.
    token_base: usize,
    /// Base offset into [`NodeTensor::ebit_sums`]: `+ pred*num_buckets + bucket`.
    ebit_base: usize,
    /// Base offset into [`NodeTensor::weighted`] / [`NodeTensor::unique`]: `+ bucket`.
    bucket_base: usize,
}

/// Storage layout for [`NodeTensor`]s — computed once per tree build and
/// shared (by reference) by every tensor of that build. Token rows are
/// `histogram_size`-strided (tight), NOT `HISTO_PADDED`-strided.
struct TensorLayout {
    /// One entry per position in `params.properties`, same order.
    prop_entries: Vec<TensorPropEntry>,
    num_preds: usize,
    histo: usize,
    /// Total u32 cells in `token_counts` = Σ_p num_preds × nb_p × histo.
    /// This is also the cost model for one tensor build/subtract pass.
    token_cells: usize,
    /// Total u64 cells in `ebit_sums` = Σ_p num_preds × nb_p.
    ebit_cells: usize,
    /// Total u32 cells in `weighted` / `unique` = Σ_p nb_p.
    bucket_cells: usize,
}

impl TensorLayout {
    fn new(
        params: &TreeLearningParams,
        num_preds: usize,
        histo: usize,
        num_thresholds_of: impl Fn(usize) -> usize,
    ) -> Self {
        let mut prop_entries = Vec::with_capacity(params.properties.len());
        let mut token_base = 0usize;
        let mut ebit_base = 0usize;
        let mut bucket_base = 0usize;
        for &prop_idx in &params.properties {
            let nt = num_thresholds_of(prop_idx);
            let num_buckets = if nt == 0 { 0 } else { nt + 1 };
            prop_entries.push(TensorPropEntry {
                prop_idx,
                num_buckets,
                token_base,
                ebit_base,
                bucket_base,
            });
            token_base += num_preds * num_buckets * histo;
            ebit_base += num_preds * num_buckets;
            bucket_base += num_buckets;
        }
        Self {
            prop_entries,
            num_preds,
            histo,
            token_cells: token_base,
            ebit_cells: ebit_base,
            bucket_cells: bucket_base,
        }
    }
}

/// Additive per-node aggregates of everything `find_best_split`'s per-sample
/// loops compute. All four arrays are sums over the node's rows, so tensors
/// of disjoint row sets add — and a child can be derived from its parent by
/// exact integer subtraction ([`NodeTensor::subtract_in_place`]).
struct NodeTensor {
    /// Per-(prop, pred, absolute bucket, token) dedup-weighted counts.
    token_counts: Vec<u32>,
    /// Per-(prop, pred, absolute bucket) `Σ ebits·count` sums.
    ebit_sums: Vec<u64>,
    /// Per-(prop, absolute bucket) weighted row counts (`Σ sample_counts`).
    weighted: Vec<u32>,
    /// Per-(prop, absolute bucket) unique row counts.
    unique: Vec<u32>,
}

impl NodeTensor {
    fn zeroed(layout: &TensorLayout) -> Self {
        Self {
            token_counts: vec![0u32; layout.token_cells],
            ebit_sums: vec![0u64; layout.ebit_cells],
            weighted: vec![0u32; layout.bucket_cells],
            unique: vec![0u32; layout.bucket_cells],
        }
    }

    /// `self = self − smaller`, elementwise. Exact for tensors over nested
    /// row sets (every count in `smaller` ≤ the matching count in `self`);
    /// debug builds panic on underflow (which would mean the two tensors
    /// were not built over parent/child row sets of the same layout).
    fn subtract_in_place(&mut self, smaller: &NodeTensor) {
        debug_assert_eq!(self.token_counts.len(), smaller.token_counts.len());
        debug_assert_eq!(self.ebit_sums.len(), smaller.ebit_sums.len());
        debug_assert_eq!(self.weighted.len(), smaller.weighted.len());
        debug_assert_eq!(self.unique.len(), smaller.unique.len());
        for (a, b) in self
            .token_counts
            .iter_mut()
            .zip(smaller.token_counts.iter())
        {
            *a -= b;
        }
        for (a, b) in self.ebit_sums.iter_mut().zip(smaller.ebit_sums.iter()) {
            *a -= b;
        }
        for (a, b) in self.weighted.iter_mut().zip(smaller.weighted.iter()) {
            *a -= b;
        }
        for (a, b) in self.unique.iter_mut().zip(smaller.unique.iter()) {
            *a -= b;
        }
    }
}

/// Tensor participation of one `find_best_split` call.
enum TensorMode<'a> {
    /// No tensor involvement — byte-for-byte the pre-chunk behaviour.
    Off,
    /// The node's tensor already exists (built or derived): bucket stats and
    /// per-(prop,pred) rows are read from it and the per-sample loops are
    /// skipped. Requires `weighted_total >= TENSOR_MIN_CHILD_WEIGHT` (full
    /// predictor + property coverage), which the engines guarantee.
    Use(&'a TensorLayout, &'a NodeTensor),
    /// The per-sample loops run as normal AND their aggregates are copied
    /// into the (pre-zeroed) tensor, making this node a future subtraction
    /// parent without a second pass over its rows. Same coverage
    /// requirement as `Use`. The captured tensor is only complete when at
    /// least one property produced a populated bucket range — which is
    /// implied whenever a split is found, the only case the engines consume
    /// the capture.
    Capture(&'a TensorLayout, &'a mut NodeTensor),
}

/// Profitability gate for deriving children at one split: the larger
/// child's skipped per-sample work (≈ `larger_unique × num_preds` row
/// visits across the property loop) must exceed one tensor-sized pass
/// (`token_cells` ops ≈ the subtraction + the row copies). Mirrors the
/// `.meta` plan's "n_larger × num_pred exceeds tensor_subtract cost".
fn tensor_derive_pays(layout: &TensorLayout, larger_unique: usize) -> bool {
    (larger_unique as u64).saturating_mul(layout.num_preds as u64) > layout.token_cells as u64
}

/// Capture gate for a node without a tensor: both children can only reach
/// [`TENSOR_MIN_CHILD_WEIGHT`] if the node has ≥ 2× that weight
/// (`unique_count` lower-bounds the weighted total), and capturing is
/// pointless unless a child split could pass [`tensor_derive_pays`].
fn tensor_capture_pays(layout: &TensorLayout, unique_count: usize) -> bool {
    layout.token_cells > 0
        && unique_count >= (2 * TENSOR_MIN_CHILD_WEIGHT) as usize
        && tensor_derive_pays(layout, unique_count)
}

/// Builds a node's [`NodeTensor`] from its sample rows `[start..end)`.
/// `out` MUST be zeroed (fresh from [`NodeTensor::zeroed`]).
///
/// Populates EVERY property in `params.properties` with `num_buckets > 0`
/// and EVERY predictor, regardless of per-node pruning or single-bucket
/// collapse — full population is what makes parent − smaller-child
/// subtraction exact for every field a descendant may read.
///
/// Loop structure mirrors `find_best_split`'s counting-sort + per-(pred,
/// bucket) accumulate so the memory-access pattern (and the produced
/// integers) match the capture path exactly.
#[allow(clippy::too_many_arguments)]
fn build_node_tensor(
    samples: &TreeSamples,
    pq: &PreQuantizedProps,
    params: &TreeLearningParams,
    layout: &TensorLayout,
    histogram_size: usize,
    start: usize,
    end: usize,
    out: &mut NodeTensor,
) {
    let count = end - start;
    let num_pred = samples.num_predictors();
    debug_assert_eq!(num_pred, layout.num_preds);
    debug_assert_eq!(histogram_size, layout.histo);
    let sample_counts = &samples.sample_counts[start..end];
    let max_buckets = params.max_property_values + 1;
    with_workspace_dispatched(
        params.parallel_small_image_fallback,
        count,
        histogram_size,
        max_buckets,
        |ws| {
            let sorted_by_bucket = ws.sorted_by_bucket.as_mut_slice();
            let bucket_starts = ws.bucket_starts.as_mut_slice();
            let bucket_write_pos = ws.bucket_write_pos.as_mut_slice();
            for (prop_pos, &prop_idx) in params.properties.iter().enumerate() {
                let entry = &layout.prop_entries[prop_pos];
                debug_assert_eq!(entry.prop_idx, prop_idx);
                let nb = entry.num_buckets;
                if nb == 0 {
                    continue;
                }
                let pq_buckets = &pq.bucket_indices[prop_idx][start..end];

                // Bucket stats accumulate straight into the zeroed tensor,
                // in ABSOLUTE bucket space.
                {
                    let uni = &mut out.unique[entry.bucket_base..entry.bucket_base + nb];
                    let wei = &mut out.weighted[entry.bucket_base..entry.bucket_base + nb];
                    for (offset, &b) in pq_buckets.iter().enumerate() {
                        uni[b as usize] += 1;
                        wei[b as usize] += sample_counts[offset];
                    }
                    bucket_starts[0] = 0;
                    for b in 0..nb {
                        bucket_starts[b + 1] = bucket_starts[b] + uni[b] as usize;
                    }
                }
                bucket_write_pos[..nb].copy_from_slice(&bucket_starts[..nb]);
                for (offset, &b) in pq_buckets.iter().enumerate() {
                    sorted_by_bucket[bucket_write_pos[b as usize]] = offset;
                    bucket_write_pos[b as usize] += 1;
                }

                for pred in 0..num_pred {
                    let tokens = &samples.residual_tokens[pred][start..end];
                    let ebits = &samples.extra_bits[pred][start..end];
                    for b in 0..nb {
                        let bs = bucket_starts[b];
                        let be = bucket_starts[b + 1];
                        if bs == be {
                            continue;
                        }
                        let row_base = entry.token_base + (pred * nb + b) * histogram_size;
                        let row = &mut out.token_counts[row_base..row_base + histogram_size];
                        let mut eb_sum: u64 = 0;
                        for &rel_off in &sorted_by_bucket[bs..be] {
                            let tok = tokens[rel_off] as usize;
                            let sc = sample_counts[rel_off];
                            debug_assert!(tok < histogram_size);
                            row[tok] += sc;
                            eb_sum += ebits[rel_off] as u64 * sc as u64;
                        }
                        out.ebit_sums[entry.ebit_base + pred * nb + b] = eb_sum;
                    }
                }
            }
        },
    );
}

/// Borrowed-view counterpart to [`build_node_tensor`]. Identical algorithm;
/// only the data access path differs.
#[cfg(feature = "parallel-tree-learning")]
#[allow(clippy::too_many_arguments)]
fn build_node_tensor_borrowed(
    samples: &BorrowedSamples<'_>,
    params: &TreeLearningParams,
    layout: &TensorLayout,
    histogram_size: usize,
    start: usize,
    end: usize,
    out: &mut NodeTensor,
) {
    let count = end - start;
    let num_pred = samples.num_predictors();
    debug_assert_eq!(num_pred, layout.num_preds);
    debug_assert_eq!(histogram_size, layout.histo);
    let sample_counts = &samples.sample_counts[start..end];
    let max_buckets = params.max_property_values + 1;
    with_workspace_dispatched(
        params.parallel_small_image_fallback,
        count,
        histogram_size,
        max_buckets,
        |ws| {
            let sorted_by_bucket = ws.sorted_by_bucket.as_mut_slice();
            let bucket_starts = ws.bucket_starts.as_mut_slice();
            let bucket_write_pos = ws.bucket_write_pos.as_mut_slice();
            for (prop_pos, &prop_idx) in params.properties.iter().enumerate() {
                let entry = &layout.prop_entries[prop_pos];
                debug_assert_eq!(entry.prop_idx, prop_idx);
                let nb = entry.num_buckets;
                if nb == 0 {
                    continue;
                }
                let pq_buckets = &samples.bucket_indices[prop_idx][start..end];

                {
                    let uni = &mut out.unique[entry.bucket_base..entry.bucket_base + nb];
                    let wei = &mut out.weighted[entry.bucket_base..entry.bucket_base + nb];
                    for (offset, &b) in pq_buckets.iter().enumerate() {
                        uni[b as usize] += 1;
                        wei[b as usize] += sample_counts[offset];
                    }
                    bucket_starts[0] = 0;
                    for b in 0..nb {
                        bucket_starts[b + 1] = bucket_starts[b] + uni[b] as usize;
                    }
                }
                bucket_write_pos[..nb].copy_from_slice(&bucket_starts[..nb]);
                for (offset, &b) in pq_buckets.iter().enumerate() {
                    sorted_by_bucket[bucket_write_pos[b as usize]] = offset;
                    bucket_write_pos[b as usize] += 1;
                }

                for pred in 0..num_pred {
                    let tokens = &samples.residual_tokens[pred][start..end];
                    let ebits = &samples.extra_bits[pred][start..end];
                    for b in 0..nb {
                        let bs = bucket_starts[b];
                        let be = bucket_starts[b + 1];
                        if bs == be {
                            continue;
                        }
                        let row_base = entry.token_base + (pred * nb + b) * histogram_size;
                        let row = &mut out.token_counts[row_base..row_base + histogram_size];
                        let mut eb_sum: u64 = 0;
                        for &rel_off in &sorted_by_bucket[bs..be] {
                            let tok = tokens[rel_off] as usize;
                            let sc = sample_counts[rel_off];
                            debug_assert!(tok < histogram_size);
                            row[tok] += sc;
                            eb_sum += ebits[rel_off] as u64 * sc as u64;
                        }
                        out.ebit_sums[entry.ebit_base + pred * nb + b] = eb_sum;
                    }
                }
            }
        },
    );
}

/// Shared per-split derivation gate + smaller/larger bookkeeping. Returns
/// `Some((smaller_is_left, smaller_range))` when deriving pays, else `None`.
///
/// Gates (all speed-only; never affect tree topology or bytes):
/// - either child immediately leafs (`bits <= threshold`) → its
///   `find_best_split` never runs, so skip the whole derivation;
/// - the larger child's skipped loops must outweigh a tensor pass
///   ([`tensor_derive_pays`]);
/// - both children need `weighted >= TENSOR_MIN_CHILD_WEIGHT` so their
///   `find_best_split` runs with full (prop, pred) coverage, keeping
///   tensor contents exhaustive for *their* descendants.
#[allow(clippy::too_many_arguments)]
fn tensor_split_plan(
    layout: &TensorLayout,
    sample_counts: &[u32],
    start: usize,
    mid: usize,
    end: usize,
    left_bits: f64,
    right_bits: f64,
    threshold: f64,
) -> Option<(bool, usize, usize)> {
    if left_bits <= threshold || right_bits <= threshold {
        return None;
    }
    let left_unique = mid - start;
    let right_unique = end - mid;
    let larger_unique = left_unique.max(right_unique);
    if !tensor_derive_pays(layout, larger_unique) {
        return None;
    }
    let left_w: u32 = sample_counts[start..mid].iter().sum();
    let right_w: u32 = sample_counts[mid..end].iter().sum();
    if left_w < TENSOR_MIN_CHILD_WEIGHT || right_w < TENSOR_MIN_CHILD_WEIGHT {
        return None;
    }
    if left_unique <= right_unique {
        Some((true, start, mid))
    } else {
        Some((false, mid, end))
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only: counts successful child-tensor derivations on THIS thread.
    /// Thread-local (not atomic) so concurrent tests in the same binary
    /// can't bump a measurement that belongs to another test — only the
    /// sequential engine (which derives on the calling thread) is observed.
    pub(crate) static TENSOR_DERIVE_COUNT: core::cell::Cell<usize> =
        const { core::cell::Cell::new(0) };
}

/// Derive child tensors for an accepted split of a node owning `parent`:
/// build the smaller child's tensor from its rows, derive the larger child
/// in place (`parent −= smaller`), return `(left, right)` tensors. The
/// parent tensor is consumed either way (freed here when gates fail —
/// the meta's "parent freed after deriving" memory bound).
#[allow(clippy::too_many_arguments)]
fn derive_child_tensors(
    samples: &TreeSamples,
    pq: &PreQuantizedProps,
    params: &TreeLearningParams,
    layout: &TensorLayout,
    histogram_size: usize,
    start: usize,
    mid: usize,
    end: usize,
    parent: NodeTensor,
    left_bits: f64,
    right_bits: f64,
    threshold: f64,
) -> (Option<NodeTensor>, Option<NodeTensor>) {
    let Some((smaller_is_left, s_start, s_end)) = tensor_split_plan(
        layout,
        &samples.sample_counts,
        start,
        mid,
        end,
        left_bits,
        right_bits,
        threshold,
    ) else {
        return (None, None);
    };
    let mut small = NodeTensor::zeroed(layout);
    build_node_tensor(
        samples,
        pq,
        params,
        layout,
        histogram_size,
        s_start,
        s_end,
        &mut small,
    );
    let mut large = parent;
    large.subtract_in_place(&small);
    #[cfg(test)]
    TENSOR_DERIVE_COUNT.with(|c| c.set(c.get() + 1));
    if smaller_is_left {
        (Some(small), Some(large))
    } else {
        (Some(large), Some(small))
    }
}

/// Borrowed-view counterpart to [`derive_child_tensors`].
#[cfg(feature = "parallel-tree-learning")]
#[allow(clippy::too_many_arguments)]
fn derive_child_tensors_borrowed(
    samples: &BorrowedSamples<'_>,
    params: &TreeLearningParams,
    layout: &TensorLayout,
    histogram_size: usize,
    start: usize,
    mid: usize,
    end: usize,
    parent: NodeTensor,
    left_bits: f64,
    right_bits: f64,
    threshold: f64,
) -> (Option<NodeTensor>, Option<NodeTensor>) {
    let Some((smaller_is_left, s_start, s_end)) = tensor_split_plan(
        layout,
        samples.sample_counts,
        start,
        mid,
        end,
        left_bits,
        right_bits,
        threshold,
    ) else {
        return (None, None);
    };
    let mut small = NodeTensor::zeroed(layout);
    build_node_tensor_borrowed(
        samples,
        params,
        layout,
        histogram_size,
        s_start,
        s_end,
        &mut small,
    );
    let mut large = parent;
    large.subtract_in_place(&small);
    if smaller_is_left {
        (Some(small), Some(large))
    } else {
        (Some(large), Some(small))
    }
}

/// Pre-allocated workspace for find_best_split, reused across calls.
/// Avoids per-call Vec allocation and resize overhead.
struct SplitWorkspace {
    count_increase: Vec<u32>,
    extra_bits_increase: Vec<u64>,
    bucket_counts: Vec<u32>,
    right_counts: Vec<u32>,
    left_counts: Vec<u32>,
    best_l_cost: Vec<f64>,
    best_r_cost: Vec<f64>,
    /// Per-side penalized cost (raw cost + predictor change penalty).
    /// Used for predictor selection; the final split decision uses raw costs only.
    best_l_penalized: Vec<f64>,
    best_r_penalized: Vec<f64>,
    best_l_pred: Vec<usize>,
    best_r_pred: Vec<usize>,
    sorted_by_bucket: Vec<usize>,
    bucket_starts: Vec<usize>,
    bucket_write_pos: Vec<usize>,
}

/// Test-only allocation counter for [`SplitWorkspace::new`]. Used by the
/// thread-local cache invariant test to prove that a full encode triggers
/// only `O(num_threads)` workspace allocations, not `O(forks)`.
#[cfg(test)]
pub(crate) static SPLIT_WS_ALLOC_COUNT: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// Test-only allocation counter that increments **only** on threads marked as
/// belonging to the cap-allocations test's private rayon pool (via
/// [`IS_TEST_POOL_THREAD`]). The cap test reads this counter so it sees ONLY
/// allocations made by its own controlled thread set, immune to allocations
/// from other concurrent tests in the same test binary. Tracks issue #51.
#[cfg(test)]
pub(crate) static SPLIT_WS_ALLOC_COUNT_TEST_POOL: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
thread_local! {
    /// Set to `true` on threads owned by the cap-allocations test's private
    /// rayon pool (and on the test's calling thread for the duration of the
    /// measurement). When set, [`SplitWorkspace::new`] also increments
    /// [`SPLIT_WS_ALLOC_COUNT_TEST_POOL`]. Issue #51.
    pub(crate) static IS_TEST_POOL_THREAD: core::cell::Cell<bool> =
        const { core::cell::Cell::new(false) };
}

impl SplitWorkspace {
    fn new(max_count: usize, histogram_size: usize, max_buckets: usize) -> Self {
        // Provable: `histogram_size` derives from `GATHER_HYBRID_UINT.encode`
        // tokens, max 239 for any u32 input (see HISTO_PADDED comment).
        debug_assert!(histogram_size <= HISTO_PADDED);
        #[cfg(test)]
        {
            SPLIT_WS_ALLOC_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            if IS_TEST_POOL_THREAD.with(|f| f.get()) {
                SPLIT_WS_ALLOC_COUNT_TEST_POOL.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
        }
        Self {
            count_increase: vec![0u32; max_buckets * HISTO_PADDED],
            extra_bits_increase: vec![0u64; max_buckets],
            bucket_counts: vec![0u32; max_buckets],
            right_counts: vec![0u32; histogram_size],
            left_counts: vec![0u32; histogram_size],
            best_l_cost: vec![f64::MAX; max_buckets],
            best_r_cost: vec![f64::MAX; max_buckets],
            best_l_penalized: vec![f64::MAX; max_buckets],
            best_r_penalized: vec![f64::MAX; max_buckets],
            best_l_pred: vec![0usize; max_buckets],
            best_r_pred: vec![0usize; max_buckets],
            sorted_by_bucket: vec![0usize; max_count],
            bucket_starts: vec![0usize; max_buckets + 2],
            bucket_write_pos: vec![0usize; max_buckets],
        }
    }

    /// Grow the cached buffers in-place to fit `(max_count, histogram_size,
    /// max_buckets)`. Used by [`with_thread_local_workspace`] to reuse a
    /// per-thread `SplitWorkspace` across many `find_best_split` calls.
    ///
    /// All buffers are overwritten before read inside `find_best_split` (the
    /// `.fill(...)` and bucket-counting passes), so resize-with-zero (or with
    /// `f64::MAX` / `0usize`) is sufficient — there is no live data carrying
    /// between calls. `Vec::resize` is a no-op (and zero realloc) when the
    /// existing `len` already covers the request, which is the steady-state
    /// case after the first fork on a worker thread.
    ///
    /// Why this matters: at n=1.5M samples the `sorted_by_bucket` Vec alone
    /// is ~12 MB. Allocating one per `build_subtree_recursive_parallel` fork
    /// (up to 16 leaf tasks) creates ~200 MB of allocator + zero-fill churn
    /// per encode — the measured Amdahl ceiling on the
    /// `parallel-tree-learning` path (memory file:
    /// `rayon_modular_groups_2026-05-16.md`, OUTCOME section).
    ///
    /// Caching one workspace per rayon worker thread caps the live allocation
    /// at `num_threads × 12 MB` regardless of fork count, and subsequent
    /// reuses skip the `vec![0; 12M]` calloc entirely (resize is a no-op).
    fn reset_for(&mut self, max_count: usize, histogram_size: usize, max_buckets: usize) {
        debug_assert!(histogram_size <= HISTO_PADDED);
        // Per-buckets buffers
        self.count_increase.resize(max_buckets * HISTO_PADDED, 0u32);
        self.extra_bits_increase.resize(max_buckets, 0u64);
        self.bucket_counts.resize(max_buckets, 0u32);
        self.best_l_cost.resize(max_buckets, f64::MAX);
        self.best_r_cost.resize(max_buckets, f64::MAX);
        self.best_l_penalized.resize(max_buckets, f64::MAX);
        self.best_r_penalized.resize(max_buckets, f64::MAX);
        self.best_l_pred.resize(max_buckets, 0usize);
        self.best_r_pred.resize(max_buckets, 0usize);
        self.bucket_starts.resize(max_buckets + 2, 0usize);
        self.bucket_write_pos.resize(max_buckets, 0usize);
        // Per-histogram buffers
        self.right_counts.resize(histogram_size, 0u32);
        self.left_counts.resize(histogram_size, 0u32);
        // Per-sample buffer — the dominant ~12 MB allocation at large n.
        self.sorted_by_bucket.resize(max_count, 0usize);
    }
}

// Thread-local cache of one `SplitWorkspace` per worker thread.
//
// Each rayon worker (or the main thread, when running serially) keeps a
// single `SplitWorkspace` alive across `find_best_split` calls, eliminating
// the per-call `Vec::with_capacity(...) + zero-fill` cost. The workspace
// grows in-place via `SplitWorkspace::reset_for` when a larger node
// arrives; steady-state reuse is allocation-free.
//
// Workspaces are NEVER shared across threads — each thread owns its slot,
// so there is no contention beyond the `RefCell::borrow_mut` (which sees
// no concurrent access because the cache is `thread_local!`).
//
// The first-fork cost on a new worker is the same as a one-shot
// `SplitWorkspace::new` (≈12 MB calloc at n=1.5M). After that, additional
// forks on the same worker pay zero allocation. With 8 rayon workers,
// total live allocation caps at `~8 × 12 MB = ~96 MB` regardless of how
// many subtree forks a single encode produces — versus the previous
// `forks × 12 MB ≈ 200 MB` allocator churn that was the measured Amdahl
// ceiling on the parallel path (see
// `memory/rayon_modular_groups_2026-05-16.md`).
thread_local! {
    static SPLIT_WORKSPACE_CACHE: core::cell::RefCell<Option<SplitWorkspace>> =
        const { core::cell::RefCell::new(None) };
}

/// Borrow this thread's cached [`SplitWorkspace`], grow it to fit
/// `(max_count, histogram_size, max_buckets)`, and pass `&mut SplitWorkspace`
/// to `f`. The workspace stays in the cache after `f` returns, ready for
/// the next `find_best_split` on the same thread.
///
/// The closure runs while the `RefCell` is mutably borrowed — calling this
/// function reentrantly from the same thread (e.g. recursing into another
/// helper that also wants the workspace) will panic. The current call
/// sites are leaf-level (`find_best_split` only) so reentrancy is not
/// possible.
fn with_thread_local_workspace<R>(
    max_count: usize,
    histogram_size: usize,
    max_buckets: usize,
    f: impl FnOnce(&mut SplitWorkspace) -> R,
) -> R {
    SPLIT_WORKSPACE_CACHE.with(|cell| {
        let mut borrowed = cell.borrow_mut();
        let ws = borrowed
            .get_or_insert_with(|| SplitWorkspace::new(max_count, histogram_size, max_buckets));
        ws.reset_for(max_count, histogram_size, max_buckets);
        f(ws)
    })
}

/// Dispatch between the thread-local cache and a per-call
/// [`SplitWorkspace::new`] allocation. On small images the
/// `RefCell::borrow_mut` indirection costs more than it saves (the
/// audit-documented +0.85% small-image regression from commit `cb5e202`).
/// When `bypass_cache` is true we go straight to `SplitWorkspace::new`
/// (the pre-`cb5e202` behaviour); otherwise we route through the cache.
#[inline]
fn with_workspace_dispatched<R>(
    bypass_cache: bool,
    max_count: usize,
    histogram_size: usize,
    max_buckets: usize,
    f: impl FnOnce(&mut SplitWorkspace) -> R,
) -> R {
    if bypass_cache {
        let mut ws = SplitWorkspace::new(max_count, histogram_size, max_buckets);
        f(&mut ws)
    } else {
        with_thread_local_workspace(max_count, histogram_size, max_buckets, f)
    }
}

/// Result of finding the best split for a node.
struct BestSplit {
    property: usize,
    splitval: i32,
    left_predictor: usize,
    right_predictor: usize,
    total_bits: f64,
    /// Number of unique samples that belong on the LEFT side of the split
    /// (i.e., rows with `bucket_index <= local_k`). Captured during the sweep
    /// (= `bucket_starts[local_k + 1]`) so the caller can pass it directly as
    /// the `pos` argument to `split_tree_samples_in_place` without rescanning.
    left_count: usize,
    /// Entropy cost (`estimate_bits_u32` + weighted extra bits) of the LEFT
    /// side at the winning threshold under `left_predictor`, captured from
    /// the sweep (issue #64 side-costs rider). Bitwise-identical to the
    /// `compute_predictor_entropy` recompute over the partitioned left range:
    /// same u32 histogram contents (integer accumulation, order-free), same
    /// u64 `eb * count` sum, and the same `estimate_bits_u32` call shape —
    /// both sweep and recompute pass `..histogram_size` slices
    /// (`effective_histo == histogram_size` in both split fns). Engines
    /// consume this directly and skip the 2×O(n_side) post-split recompute;
    /// debug builds re-derive and assert via
    /// `debug_verify_carried_side_bits`.
    left_bits: f64,
    /// Right-side counterpart of `left_bits` (under `right_predictor`).
    right_bits: f64,
}

/// Find the best (property, threshold) split for the contiguous sample range
/// `[start..end)`.
///
/// Uses pre-quantized property buckets and a count_increase table approach
/// matching libjxl's enc_ma.cc:FindBestSplit.
///
/// Key optimizations over baseline:
/// - Pre-quantized bucket indices (no per-node binary_search or threshold allocation)
/// - Bucket range narrowing: only iterate bmin..bmax for this node's samples
/// - Effective histogram size: track max token across all predictors per node
/// - Zip iterators in sweep loop for bounds check elimination
/// - Cached left_bits/right_bits in BestSplit to avoid redundant entropy computation
/// - Pre-allocated workspace buffers (eliminates per-call Vec allocation)
///
/// Post-issue-#40-chunk-2: replaced `indices: &[usize]` with `[start..end)`.
/// The SoA arrays are kept contiguous in this range by `split_tree_samples_in_place`
/// at partition time, so the bmin/bmax scan, the bucket-count phase, and the
/// counting-sort population now read `pq_buckets[start..end]`, `sample_counts[start..end]`
/// sequentially instead of chasing scattered indices.
#[allow(clippy::too_many_arguments)]
fn find_best_split(
    samples: &TreeSamples,
    start: usize,
    end: usize,
    histogram_size: usize,
    base_bits: f64,
    params: &TreeLearningParams,
    parent_predictor: usize,
    threshold: f64,
    pq: &PreQuantizedProps,
    ws: &mut SplitWorkspace,
    tensor_mode: TensorMode<'_>,
) -> Option<BestSplit> {
    let count = end - start;
    if count < 2 {
        return None;
    }

    let total_num_pred = samples.num_predictors();
    let mut best: Option<BestSplit> = None;
    let mut best_bits = base_bits;
    #[cfg(feature = "__env_var_diagnostics")]
    let mut mab_per_prop: Vec<(u8, f64)> = Vec::new();

    let sample_counts_full = &samples.sample_counts;
    let sample_counts = &sample_counts_full[start..end];

    // Compute weighted total: sum of sample_counts for this node's samples.
    // After dedup, each unique sample represents `count` original samples.
    let weighted_total: u32 = sample_counts.iter().sum();

    // Decompose the tensor mode into its two orthogonal capabilities so the
    // borrows stay disjoint inside the loops below.
    let (tensor_in, mut capture) = match tensor_mode {
        TensorMode::Off => (None, None),
        TensorMode::Use(l, t) => (Some((l, t)), None),
        TensorMode::Capture(l, t) => (None, Some((l, t))),
    };
    debug_assert!(
        (tensor_in.is_none() && capture.is_none()) || weighted_total >= TENSOR_MIN_CHILD_WEIGHT,
        "tensor modes require full predictor/property coverage (weighted_total >= 2048)"
    );

    // Capture scratch: per-pred whole-node histograms + extra-bit totals,
    // snapshotted from the FIRST populated property's right-init (they are
    // property-independent — every row lands in exactly one bucket of any
    // property). Used to fill single-bucket properties' tensor rows at the
    // end, keeping captured tensors fully populated (subtraction-exact)
    // without an extra per-sample pass.
    let mut cap_totals: Vec<u32> = Vec::new();
    let mut cap_total_ebits: Vec<u64> = Vec::new();
    let mut cap_totals_done = false;
    // (prop position, absolute bucket) of properties whose rows collapsed to
    // a single bucket in this node — deferred until totals are known.
    let mut cap_single_bucket: Vec<(usize, usize)> = Vec::new();

    // Predictor change penalty matching libjxl's enc_ma.cc:303
    let change_pred_penalty = 800.0 / (100.0 + threshold);

    let weighted_idx = samples
        .candidate_predictors
        .iter()
        .position(|&p| p == Predictor::Weighted)
        .unwrap_or(usize::MAX);
    let zero_idx = CANDIDATE_PREDICTORS
        .iter()
        .position(|&p| p == Predictor::Zero)
        .unwrap_or(usize::MAX);

    // Count-based predictor pruning: for small nodes, only evaluate a subset
    // of predictors. The most important are Gradient(5), Weighted(6), and the
    // parent's predictor. This reduces inner loop iterations for deep nodes.
    // Use weighted_total (original sample count) for thresholds.
    // Cap at total_num_pred (may be 1 in squeeze mode with Zero-only predictor).
    let num_pred = (if weighted_total >= 2048 {
        total_num_pred // All predictors
    } else if weighted_total >= 512 {
        10
    } else if weighted_total >= 64 {
        7
    } else {
        4
    })
    .min(total_num_pred);

    // Use global histogram_size instead of per-node effective_histo scan.
    // The scan was O(N * num_pred) per node — costly at the root with 131K samples.
    // The sweep loop iterates histogram_size entries per bucket regardless, so the
    // extra work from slightly overestimating histogram_size is minimal (sweep is
    // O(B * H) which is tiny compared to the O(N) count_increase building).
    let effective_histo = histogram_size;
    if effective_histo == 0 {
        return None;
    }

    if capture.is_some() {
        cap_totals = vec![0u32; num_pred * effective_histo];
        cap_total_ebits = vec![0u64; num_pred];
    }

    // Pre-slice workspace buffers to avoid repeated Vec deref overhead.
    // Each Vec deref goes through raw_vec.ptr() + from_raw_parts() (~434M overhead
    // in profile). Slicing once here gives &mut [T] for all subsequent access.
    let count_increase = ws.count_increase.as_mut_slice();
    let extra_bits_increase = ws.extra_bits_increase.as_mut_slice();
    let bucket_counts = ws.bucket_counts.as_mut_slice();
    let right_counts = ws.right_counts.as_mut_slice();
    let left_counts = ws.left_counts.as_mut_slice();
    let best_l_cost = ws.best_l_cost.as_mut_slice();
    let best_r_cost = ws.best_r_cost.as_mut_slice();
    let best_l_penalized = ws.best_l_penalized.as_mut_slice();
    let best_r_penalized = ws.best_r_penalized.as_mut_slice();
    let best_l_pred = ws.best_l_pred.as_mut_slice();
    let best_r_pred = ws.best_r_pred.as_mut_slice();
    let sorted_by_bucket = ws.sorted_by_bucket.as_mut_slice();
    let bucket_starts = ws.bucket_starts.as_mut_slice();
    let bucket_write_pos = ws.bucket_write_pos.as_mut_slice();

    // Count-based property pruning: for very small nodes, only try the first few properties.
    // Use weighted_total (original sample count) for thresholds since count is now unique samples.
    let num_props = if weighted_total >= 256 {
        params.properties.len()
    } else if weighted_total >= 32 {
        params.properties.len().min(4)
    } else {
        params.properties.len().min(2)
    };

    for (prop_pos, &prop_idx) in params.properties[..num_props].iter().enumerate() {
        let num_thresholds = pq.num_thresholds(prop_idx);
        if num_thresholds == 0 {
            continue;
        }

        let pq_buckets = &pq.bucket_indices[prop_idx][start..end];
        let threshold_set = &pq.threshold_sets[prop_idx];

        // Bucket range narrowing: find min/max bucket for this node's
        // samples. With a tensor, the per-bucket unique counts already
        // carry the range (nonzero exactly where this node has rows) —
        // the O(n) scan is skipped.
        let (bmin, bmax) = if let Some((layout, tensor)) = tensor_in {
            let entry = &layout.prop_entries[prop_pos];
            debug_assert_eq!(entry.prop_idx, prop_idx);
            let uni = &tensor.unique[entry.bucket_base..entry.bucket_base + entry.num_buckets];
            let Some(first) = uni.iter().position(|&c| c != 0) else {
                continue;
            };
            let last = uni
                .iter()
                .rposition(|&c| c != 0)
                .expect("nonzero unique count exists");
            (first, last)
        } else {
            // Contiguous scan now that the SoA is kept aligned by
            // `split_tree_samples_in_place` (issue #40 chunk 2).
            let mut bmin: u8 = u8::MAX;
            let mut bmax: u8 = 0;
            for &b in pq_buckets {
                if b < bmin {
                    bmin = b;
                }
                if b > bmax {
                    bmax = b;
                }
            }
            (bmin as usize, bmax as usize)
        };
        if bmin == bmax {
            // All samples in same bucket — no useful split. When capturing,
            // record the bucket stats now (trivial: everything in one
            // bucket) and defer the per-(pred, token) rows until the
            // whole-node totals are known (end of the property loop).
            if let Some((layout, tensor)) = capture.as_mut() {
                let entry = &layout.prop_entries[prop_pos];
                debug_assert_eq!(entry.prop_idx, prop_idx);
                tensor.unique[entry.bucket_base + bmin] = count as u32;
                tensor.weighted[entry.bucket_base + bmin] = weighted_total;
                cap_single_bucket.push((prop_pos, bmin));
            }
            continue;
        }

        // Effective number of buckets for this node
        let local_num_buckets = bmax - bmin + 1;

        let local_num_thresholds = bmax - bmin;

        if let Some((layout, tensor)) = tensor_in {
            // Tensor path: bucket stats are read straight from the node's
            // tensor — the O(n) counting sort and the `sorted_by_bucket`
            // population are skipped (the per-(pred, bucket) rows come from
            // the tensor too, below).
            let entry = &layout.prop_entries[prop_pos];
            bucket_counts[..local_num_buckets].copy_from_slice(
                &tensor.weighted
                    [entry.bucket_base + bmin..entry.bucket_base + bmin + local_num_buckets],
            );
            let uni = &tensor.unique
                [entry.bucket_base + bmin..entry.bucket_base + bmin + local_num_buckets];
            bucket_starts[0] = 0;
            for b in 0..local_num_buckets {
                bucket_starts[b + 1] = bucket_starts[b] + uni[b] as usize;
            }
        } else {
            // Counting sort: group unique samples by bucket. Stored as RELATIVE
            // offsets into `[start..end)` so the per-bucket access pattern in the
            // pred loop stays inside the contiguous SoA slice (good cache locality
            // vs the old absolute-index path).
            // bucket_counts tracks the NUMBER OF UNIQUE SAMPLES per bucket (for sorted_by_bucket sizing).
            // We compute weighted counts separately for the sweep.
            let mut unique_per_bucket = [0u32; 256];
            bucket_counts[..local_num_buckets].fill(0); // weighted counts for sweep
            for (offset, &b) in pq_buckets.iter().enumerate() {
                let local_b = (b as usize) - bmin;
                unique_per_bucket[local_b] += 1;
                bucket_counts[local_b] += sample_counts[offset];
            }

            bucket_starts[0] = 0;
            for b in 0..local_num_buckets {
                bucket_starts[b + 1] = bucket_starts[b] + unique_per_bucket[b] as usize;
            }

            bucket_write_pos[..local_num_buckets]
                .copy_from_slice(&bucket_starts[..local_num_buckets]);
            for (offset, &b) in pq_buckets.iter().enumerate() {
                let local_b = (b as usize) - bmin;
                // Store RELATIVE offset; downstream loops add `start` when
                // indexing the parent SoA arrays.
                sorted_by_bucket[bucket_write_pos[local_b]] = offset;
                bucket_write_pos[local_b] += 1;
            }

            if let Some((layout, tensor)) = capture.as_mut() {
                let entry = &layout.prop_entries[prop_pos];
                debug_assert_eq!(entry.prop_idx, prop_idx);
                tensor.unique
                    [entry.bucket_base + bmin..entry.bucket_base + bmin + local_num_buckets]
                    .copy_from_slice(&unique_per_bucket[..local_num_buckets]);
                tensor.weighted
                    [entry.bucket_base + bmin..entry.bucket_base + bmin + local_num_buckets]
                    .copy_from_slice(&bucket_counts[..local_num_buckets]);
            }
        }

        // Initialize per-threshold best costs
        best_l_cost[..local_num_thresholds].fill(f64::MAX);
        best_r_cost[..local_num_thresholds].fill(f64::MAX);
        best_l_penalized[..local_num_thresholds].fill(f64::MAX);
        best_r_penalized[..local_num_thresholds].fill(f64::MAX);
        best_l_pred[..local_num_thresholds].fill(0);
        best_r_pred[..local_num_thresholds].fill(0);

        for pred in 0..num_pred {
            // Predictor change penalty: applied when choosing best predictor per side,
            // but NOT included in the final split decision (matching libjxl enc_ma.cc:375-390).
            // This biases predictor selection toward keeping the parent's predictor
            // while allowing the split itself to be judged on pure entropy cost.
            let mut penalty: f64 = 0.0;
            if pred != parent_predictor && parent_predictor != weighted_idx {
                penalty = change_pred_penalty;
            }
            // Tiebreakers matching libjxl: disfavor Weighted (slower decode),
            // favor Zero (faster if only predictor in group+channel combination).
            if pred == weighted_idx {
                penalty += 1e-8;
            } else if pred == zero_idx {
                penalty -= 1e-8;
            }

            if let Some((layout, tensor)) = tensor_in {
                // Tensor path: this (prop, pred)'s per-bucket rows + extra-bit
                // sums are copied out of the node's tensor — replacing BOTH
                // the fill(0) below (same bytes written) and the O(n)
                // per-sample accumulate (skipped entirely).
                let entry = &layout.prop_entries[prop_pos];
                let nb = entry.num_buckets;
                for b in 0..local_num_buckets {
                    let src = entry.token_base + (pred * nb + bmin + b) * effective_histo;
                    count_increase[b * HISTO_PADDED..b * HISTO_PADDED + effective_histo]
                        .copy_from_slice(&tensor.token_counts[src..src + effective_histo]);
                    extra_bits_increase[b] =
                        tensor.ebit_sums[entry.ebit_base + pred * nb + bmin + b];
                }
            } else {
                // Slice into the contiguous range [start..end) — sequential token
                // and extra-bits reads, no per-index pointer chase across the
                // whole SoA.
                let tokens = &samples.residual_tokens[pred][start..end];
                let ebits = &samples.extra_bits[pred][start..end];

                // Clear only effective_histo entries per bucket (HISTO_PADDED stride
                // leaves gaps that are never read). Same total bytes as original code.
                for b in 0..local_num_buckets {
                    count_increase[b * HISTO_PADDED..b * HISTO_PADDED + effective_histo].fill(0);
                }
                extra_bits_increase[..local_num_buckets].fill(0);

                for local_bucket in 0..local_num_buckets {
                    let bs = bucket_starts[local_bucket];
                    let be = bucket_starts[local_bucket + 1];
                    let ci_base = local_bucket * HISTO_PADDED;
                    let ci_slice = &mut count_increase[ci_base..ci_base + HISTO_PADDED];
                    let mut eb_sum: u64 = 0;
                    // Inner loop: uses sorted_by_bucket RELATIVE offsets directly into
                    // the contiguous token/ebit/sample_counts slices. Reads stay inside
                    // the small `[start..end)` window — even when scattered by bucket
                    // sort, each cache line covers ~64 contiguous samples.
                    // ci_slice[tok & HISTO_MASK]: bitmask guarantees < HISTO_PADDED = ci_slice.len()
                    // Each unique sample contributes its count (dedup weight).
                    for &rel_off in &sorted_by_bucket[bs..be] {
                        let tok = tokens[rel_off];
                        let sc = sample_counts[rel_off];
                        ci_slice[tok as usize & HISTO_MASK] += sc;
                        eb_sum += ebits[rel_off] as u64 * sc as u64;
                    }
                    extra_bits_increase[local_bucket] = eb_sum;
                }

                if let Some((layout, tensor)) = capture.as_mut() {
                    // Copy this (prop, pred)'s freshly-accumulated rows into
                    // the node tensor (absolute bucket space). Pure extra
                    // writes — the workspace contents and control flow are
                    // untouched, so the split search below is unaffected.
                    let entry = &layout.prop_entries[prop_pos];
                    let nb = entry.num_buckets;
                    for b in 0..local_num_buckets {
                        let dst = entry.token_base + (pred * nb + bmin + b) * effective_histo;
                        tensor.token_counts[dst..dst + effective_histo].copy_from_slice(
                            &count_increase[b * HISTO_PADDED..b * HISTO_PADDED + effective_histo],
                        );
                        tensor.ebit_sums[entry.ebit_base + pred * nb + bmin + b] =
                            extra_bits_increase[b];
                    }
                }
            }

            // Build initial right histogram (all local buckets on the right
            // side). LLVM auto-vectorizes this loop to SSE2 movdqu/paddd
            // (4-wide u32 with 2× unroll) — see
            // benchmarks/find_best_split_asm_post_6011f10_2026-05-17.txt
            // lines 1320-1339 for the cargo-asm dump of the SSE2 codegen.
            //
            // Forcing AVX2 8-wide via a #[archmage::arcane] entry point
            // (column-major iteration, dst held in ymm across all rows)
            // was tried 2026-05-17 and asm-verified to use vpaddd ymm.
            // Wall-clock impact at the gate cell (1.05 MP @ e9) was
            // **zero**: median delta -0.2%, min delta 0.0% across 7
            // paired samples (benchmarks/fbs_simd_ab_2026-05-17.{tsv,meta}).
            //
            // Root cause: this fold runs num_pred × num_props ≈ 176 times
            // per node-split processing ~768 u32-adds each ≈ 5,280 cycles
            // total — vs estimate_bits at ~739,200 cycles per split. The
            // right-init fold is <1% of find_best_split's CPU; even
            // infinite speedup is invisible at wall-clock scope. The next
            // actionable gap lives in OTHER functions (find_best_predictor,
            // compute_best_tree fan-out depth, pre_quantize, gather_samples,
            // dedup_samples). See benchmarks/fbs_simd_ab_2026-05-17.meta
            // for the full asm-evidenced post-mortem.
            right_counts[..effective_histo].fill(0);
            let mut right_extra: u64 = 0;
            let mut right_total: u32 = weighted_total;
            for (local_bucket, &eb) in extra_bits_increase[..local_num_buckets].iter().enumerate() {
                let ci_base = local_bucket * HISTO_PADDED;
                let ci_row = &count_increase[ci_base..ci_base + effective_histo];
                for (rc, &ci) in right_counts[..effective_histo]
                    .iter_mut()
                    .zip(ci_row.iter())
                {
                    *rc += ci;
                }
                right_extra += eb;
            }

            if capture.is_some() && !cap_totals_done {
                // The fully-initialized right histogram == the whole-node
                // histogram for this predictor (property-independent).
                // Snapshot it before the sweep mutates it; consumed by the
                // single-bucket-property fill after the property loop.
                cap_totals[pred * effective_histo..(pred + 1) * effective_histo]
                    .copy_from_slice(&right_counts[..effective_histo]);
                cap_total_ebits[pred] = right_extra;
            }

            left_counts[..effective_histo].fill(0);
            let mut left_extra: u64 = 0;
            let mut left_total: u32 = 0;

            // Sweep through local buckets, moving each from right to left.
            // Cost computed via estimate_bits (with 1/4096 probability floor),
            // matching libjxl's EstimateBits used for both parent and child costs.
            for local_k in 0..local_num_thresholds {
                let bc = bucket_counts[local_k];
                if bc == 0 {
                    continue;
                }

                // Move bucket from right to left
                let ci_base = local_k * HISTO_PADDED;
                let ci_row = &count_increase[ci_base..ci_base + effective_histo];
                for (i, &ci) in ci_row.iter().enumerate() {
                    if ci > 0 {
                        left_counts[i] += ci;
                        right_counts[i] -= ci;
                    }
                }
                left_extra += extra_bits_increase[local_k];
                right_extra -= extra_bits_increase[local_k];
                left_total += bc;
                right_total -= bc;

                if left_total == 0 || right_total == 0 {
                    continue;
                }

                // Recompute costs using estimate_bits with probability floor,
                // matching libjxl's EstimateBits at each threshold position.
                // SIMD path: see jxl-encoder-simd/src/entropy.rs (≥4× win over
                // scalar in the find_best_split sweep — Phase-A asm showed the
                // scalar `subsd` dep chain serialized the inner loop at ~25
                // cycles/iter; SIMD breaks it into 2 independent f32 lanes
                // and hides the fast_log2f latency).
                let l_bits =
                    jxl_simd::estimate_bits_u32(&left_counts[..effective_histo], left_total)
                        + left_extra as f64;
                let r_bits =
                    jxl_simd::estimate_bits_u32(&right_counts[..effective_histo], right_total)
                        + right_extra as f64;

                // Predictor selection uses penalized cost (matching libjxl).
                // Raw cost stored separately for the final split decision.
                if l_bits + penalty < best_l_penalized[local_k] {
                    best_l_penalized[local_k] = l_bits + penalty;
                    best_l_cost[local_k] = l_bits;
                    best_l_pred[local_k] = pred;
                }
                if r_bits + penalty < best_r_penalized[local_k] {
                    best_r_penalized[local_k] = r_bits + penalty;
                    best_r_cost[local_k] = r_bits;
                    best_r_pred[local_k] = pred;
                }
            }
        }

        if capture.is_some() {
            // All predictors of the first populated property have snapshotted
            // their whole-node histograms.
            cap_totals_done = true;
        }

        // Find best threshold across all predictors for this property.
        // Split decision uses RAW costs (no penalty), matching libjxl enc_ma.cc:424.
        // The penalty only influenced which predictor was chosen for each side above.
        #[cfg(feature = "__env_var_diagnostics")]
        let mut mab_prop_best = f64::MAX;
        for local_k in 0..local_num_thresholds {
            if best_l_cost[local_k] == f64::MAX || best_r_cost[local_k] == f64::MAX {
                continue;
            }

            let total = best_l_cost[local_k] + best_r_cost[local_k];
            #[cfg(feature = "__env_var_diagnostics")]
            if total < mab_prop_best {
                mab_prop_best = total;
            }

            if total < best_bits {
                best_bits = total;
                // Map local_k back to global threshold index: bmin + local_k
                let global_k = bmin + local_k;
                // left_count is the count of unique samples in buckets [0..=local_k]
                // (== bucket_starts[local_k + 1]). This becomes the `pos` argument
                // for split_tree_samples_in_place — the caller doesn't have to
                // rescan to determine the partition split point.
                let left_count = bucket_starts[local_k + 1];
                best = Some(BestSplit {
                    property: prop_idx,
                    splitval: threshold_set[global_k],
                    left_predictor: best_l_pred[local_k],
                    right_predictor: best_r_pred[local_k],
                    total_bits: total,
                    left_count,
                    left_bits: best_l_cost[local_k],
                    right_bits: best_r_cost[local_k],
                });
            }
        }
        #[cfg(feature = "__env_var_diagnostics")]
        if mabsplit_dump::sink().is_some() && mab_prop_best < f64::MAX {
            mab_per_prop.push((prop_idx as u8, mab_prop_best));
        }
    }

    if let Some((layout, tensor)) = capture.as_mut() {
        if cap_totals_done {
            // Fill the rows of single-bucket properties from the whole-node
            // per-predictor totals: with every row in one bucket, that
            // bucket's (pred, token) histogram IS the node histogram.
            for &(prop_pos, bucket) in &cap_single_bucket {
                let entry = &layout.prop_entries[prop_pos];
                let nb = entry.num_buckets;
                for pred in 0..num_pred {
                    let dst = entry.token_base + (pred * nb + bucket) * effective_histo;
                    tensor.token_counts[dst..dst + effective_histo].copy_from_slice(
                        &cap_totals[pred * effective_histo..(pred + 1) * effective_histo],
                    );
                    tensor.ebit_sums[entry.ebit_base + pred * nb + bucket] = cap_total_ebits[pred];
                }
            }
        } else {
            // No property produced a populated bucket range, so the capture
            // is incomplete — but then no split exists either, the engines
            // leaf-finalize, and the captured tensor is dropped unused.
            debug_assert!(best.is_none());
        }
    }

    #[cfg(feature = "__env_var_diagnostics")]
    mabsplit_dump::record(
        weighted_total,
        base_bits,
        best.as_ref().map_or(-1, |b| b.property as i32),
        best_bits,
        &mab_per_prop,
    );
    best
}

/// Find the best predictor for the given contiguous sample range `[start..end)`.
///
/// The 14 candidate predictors are evaluated independently — each builds a
/// fresh residual histogram from its own per-predictor SoA columns
/// (`samples.residual_tokens[pred_idx]`, `samples.extra_bits[pred_idx]`), so
/// the loop is embarrassingly parallel.
///
/// With the `parallel-tree-learning` feature on, evaluations fan out across
/// rayon worker threads via [`crate::parallel::parallel_map`]. Each task
/// allocates its own histogram buffer (cheap relative to the O(n) entropy
/// scan). Reduction preserves the sequential tie-break: on equal cost we
/// keep the lowest predictor index, matching the original `<` (strict) loop
/// — required for byte-identical bitstream output.
///
/// A `range_size >= PARALLEL_PRED_THRESHOLD` gate keeps deep-recursion
/// per-node calls on the sequential path, where the histogram-buf alloc
/// would dominate the per-task work. The root call (full sample range,
/// typically ~10⁵–10⁶ samples) always exceeds the gate; the optional
/// `compute_best_tree_with_multipliers` per-child calls may not.
#[cfg(feature = "parallel-tree-learning")]
fn find_best_predictor(
    samples: &TreeSamples,
    start: usize,
    end: usize,
    histogram_size: usize,
    counts_buf: &mut [u32],
) -> usize {
    let num_pred = samples.num_predictors();
    let range = end - start;

    /// Below this range size, parallel fan-out costs more than it saves
    /// (per-task `Vec<u32>` histogram alloc dominates the entropy scan).
    /// Root calls are millions of samples — well above the gate. Per-node
    /// calls in `compute_best_tree_with_multipliers` may fall below.
    const PARALLEL_PRED_THRESHOLD: usize = 1024;

    if num_pred <= 1 || range < PARALLEL_PRED_THRESHOLD {
        // Sequential fallback — also covers `cfg(not(parallel))` callers
        // when this feature isn't even built.
        //
        // Issue #23 chunk 2: skip predictors whose extra-bits lower bound
        // already meets-or-exceeds the best total cost seen so far. The
        // bound is provably sound (`compute_predictor_entropy = entropy +
        // extra_bits`, both non-negative), and `decide_predictor`'s strict
        // `<` matches this loop's tie-break so the lowest-index winner on
        // equal cost is preserved — byte-identical bitstream output.
        let mut best_pred = 0;
        let mut best_bits = f64::MAX;
        for pred_idx in 0..num_pred {
            let lb = predictor_extra_bits_lower_bound(
                &samples.extra_bits[pred_idx],
                &samples.sample_counts,
                start,
                end,
            );
            if decide_predictor(lb, best_bits) == PredictorDecision::Skip {
                continue;
            }
            let bits = compute_predictor_entropy(
                samples,
                start,
                end,
                pred_idx,
                histogram_size,
                counts_buf,
            );
            if bits < best_bits {
                best_bits = bits;
                best_pred = pred_idx;
            }
        }
        return best_pred;
    }

    // Fan out across the `num_pred` candidates. Each task allocates its own
    // histogram buffer; reductions stay associative because we materialise
    // all costs and then pick the lowest-index minimum scalarly.
    //
    // Issue #23 chunk 4: seed-first hybrid. Chunk 3 (52f8e816 / 685244b)
    // shipped a concurrent atomic-best `lb >= best` prune that capped at
    // ~40 % effective prune (instead of the microbench-observed 80 %)
    // because the atomic starts at `f64::MAX` and ~half of the 14 workers
    // dispatch concurrently into that empty seed — early-wave workers
    // observe `MAX`, never skip, and pay the LB overhead.
    //
    // The hybrid first computes ALL LBs in parallel (cheap; ~half the bytes
    // of a full eval each), picks the predictor with the lowest LB
    // (lowest-index tie-break), evaluates its full cost SEQUENTIALLY before
    // the fan-out, and seeds the atomic with that real cost. The remaining
    // `num_pred - 1` workers then race against a tight seed from the very
    // first read, so the early wave benefits from the prune as well.
    //
    // ## Why the seed predictor is "lowest LB"
    //
    // The LB equals the worker's full cost if its histogram cost is zero
    // (perfectly skewed residuals). For real photos the best full-cost
    // predictor (typically Gradient/Weighted) tends to also have one of
    // the lowest LBs because its residuals concentrate in low-magnitude
    // tokens with small `extra_bits` HybridUint nbits. Picking the
    // lowest-LB predictor as the seed therefore yields a near-optimal
    // seed value on average without prejudging the answer.
    //
    // ## Byte-identity argument
    //
    // The sequential reduction is `winner = argmin_lowest_i full[i]` under
    // strict-`<` tie-break. To preserve this from the parallel fan-out:
    //
    // 1. The seed worker `s` always evaluates and contributes
    //    `costs[s] = full[s]` — it never participates in the prune.
    // 2. Every other worker `i` either evaluates (contributing `full[i]`)
    //    or skips. When skipping, we store `costs[i] = seed_at_skip_read`
    //    — the atomic value the worker observed at the skip decision.
    //    `seed_at_skip_read = full[k]` for some worker `k` that already
    //    completed a full eval (initially `k = s`, possibly updated by a
    //    later CAS to some other `k`). The skip condition is
    //    `lb[i] >= seed_at_skip_read`, which implies
    //    `full[i] >= seed_at_skip_read = full[k]`.
    //
    // The reduction picks lowest-index `i` with the smallest `costs[i]`
    // under strict-`<`. The sequential argmin is `winner_seq`. Two cases:
    //
    //   * `costs[winner_seq] == full[winner_seq]` (winner_seq evaluated):
    //     reduction sees the true cost. No skipped slot beats it because
    //     skipped `costs[i] = full[k] >= full[winner_seq]` (the global
    //     min). Tie-break among slots equal to the min uses lowest index,
    //     matching sequential.
    //   * `costs[winner_seq] = full[k]` (winner_seq skipped):
    //     `full[winner_seq] >= full[k]` and `winner_seq` is sequential
    //     argmin ⇒ `full[winner_seq] == full[k]` (else `k` would beat
    //     it sequentially when `k <= winner_seq`, or `winner_seq` wouldn't
    //     be the argmin when `k > winner_seq`). Then `costs[winner_seq] =
    //     full[k] = full[winner_seq]` — equal to the global min. The
    //     other slots are either `>= global_min` (skipped, equality
    //     possible) or `>= global_min` (evaluated, equality possible
    //     only at indices with `full == global_min`). Lowest-index
    //     reduction picks the same `winner_seq` because either:
    //       - `k < winner_seq`: sequential would have `k` win (lower
    //         index with full[k] == global_min visited first under
    //         strict-<). Contradiction with our premise that `winner_seq`
    //         is the sequential argmin. So this case is impossible.
    //       - `k > winner_seq`: sequential picks `winner_seq` (visited
    //         first under strict-<). Parallel picks lowest-index slot
    //         with `costs == global_min`. Both `winner_seq` and `k` are
    //         candidates; lowest-index wins → `winner_seq`. Match.
    //       - `k == winner_seq`: trivially matches (the seed itself
    //         doesn't skip).
    //
    // Therefore the parallel argmin always equals the sequential argmin.
    //
    // ## Cost
    //
    // - LB phase: `num_pred` cheap LB computes, embarrassingly parallel.
    //   Each LB is ~half the bytes of a full eval; total work ≈ `num_pred * 0.5`
    //   full-eval-equivalents, spread across worker threads.
    // - Seed phase: 1 sequential full eval. Adds one serial step to the
    //   critical path but eliminates the early-wave waste.
    // - Parallel phase: `num_pred - 1` workers; on real photos the prune
    //   typically fires for ~10 of them (extra_bits dominates entropy for
    //   weak predictors with high-magnitude residuals).
    use core::sync::atomic::{AtomicU64, Ordering};

    // Phase 1: compute all LBs in parallel. The LB compute is read-only on
    // the per-predictor SoA slices and uses no shared state.
    let lbs: Vec<f64> = crate::parallel::parallel_map(num_pred, |pred_idx| {
        predictor_extra_bits_lower_bound(
            &samples.extra_bits[pred_idx],
            &samples.sample_counts,
            start,
            end,
        )
    });

    // Phase 2: pick the seed predictor (lowest LB, lowest-index tie-break).
    // `num_pred >= 1` is guaranteed by the early-return branch above
    // (`num_pred <= 1` falls through to the sequential path).
    let mut seed_idx = 0usize;
    let mut seed_lb = lbs[0];
    for (i, &lb) in lbs.iter().enumerate().skip(1) {
        if lb < seed_lb {
            seed_lb = lb;
            seed_idx = i;
        }
    }

    // Phase 3: evaluate the seed predictor's full cost sequentially.
    let mut seed_counts = vec![0u32; histogram_size];
    let seed_cost = compute_predictor_entropy(
        samples,
        start,
        end,
        seed_idx,
        histogram_size,
        &mut seed_counts,
    );

    // Phase 4: dispatch the remaining `num_pred - 1` workers in parallel
    // with the atomic seeded by the real `seed_cost`. The fan-out covers
    // all indices `0..num_pred`; the seed index short-circuits to the
    // pre-computed cost without re-evaluating.
    let best_atomic = AtomicU64::new(seed_cost.to_bits());
    let costs: Vec<f64> = crate::parallel::parallel_map(num_pred, |pred_idx| {
        if pred_idx == seed_idx {
            return seed_cost;
        }
        let lb = lbs[pred_idx];
        // Read the atomic ONCE; reuse the same `current_best` for both the
        // skip decision and the skipped-slot cost record. This guarantees
        // `costs[i]` on skip equals the seed value that triggered the
        // skip, which is required for the byte-identity argument above.
        let current_best_bits = best_atomic.load(Ordering::Relaxed);
        let current_best = f64::from_bits(current_best_bits);
        if decide_predictor(lb, current_best) == PredictorDecision::Skip {
            return current_best;
        }
        let mut local_counts = vec![0u32; histogram_size];
        let bits = compute_predictor_entropy(
            samples,
            start,
            end,
            pred_idx,
            histogram_size,
            &mut local_counts,
        );
        // Strict-`<` CAS update so future workers can prune. The retry
        // loop terminates when either (a) we successfully install our
        // value, or (b) we observe a value `<= bits` and step aside.
        let mut current_bits = best_atomic.load(Ordering::Relaxed);
        loop {
            let current = f64::from_bits(current_bits);
            if bits >= current {
                break;
            }
            match best_atomic.compare_exchange_weak(
                current_bits,
                bits.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_bits = actual,
            }
        }
        bits
    });

    // Tie-break: lowest index wins (strict `<`, same as the original loop).
    // Skipped slots carry `seed_at_skip_read` (the atomic value observed at
    // skip time), not `f64::INFINITY`. This keeps the lowest-index tie-break
    // sound when the seed predictor's full cost ties with the global min
    // computed by a later worker — see the byte-identity argument above.
    let mut best_pred = 0;
    let mut best_bits = f64::MAX;
    for (i, &c) in costs.iter().enumerate() {
        if c < best_bits {
            best_bits = c;
            best_pred = i;
        }
    }
    best_pred
}

/// Sequential fallback for builds without the `parallel-tree-learning`
/// feature. Identical to the in-feature sequential branch.
#[cfg(not(feature = "parallel-tree-learning"))]
fn find_best_predictor(
    samples: &TreeSamples,
    start: usize,
    end: usize,
    histogram_size: usize,
    counts_buf: &mut [u32],
) -> usize {
    let num_pred = samples.num_predictors();
    let mut best_pred = 0;
    let mut best_bits = f64::MAX;

    // Issue #23 chunk 2: extra-bits lower-bound early-skip. See the
    // sequential branch above for the soundness proof + tie-break rationale.
    // Byte-identical to the unconditional loop.
    for pred_idx in 0..num_pred {
        let lb = predictor_extra_bits_lower_bound(
            &samples.extra_bits[pred_idx],
            &samples.sample_counts,
            start,
            end,
        );
        if decide_predictor(lb, best_bits) == PredictorDecision::Skip {
            continue;
        }
        let bits =
            compute_predictor_entropy(samples, start, end, pred_idx, histogram_size, counts_buf);
        if bits < best_bits {
            best_bits = bits;
            best_pred = pred_idx;
        }
    }

    best_pred
}

/// Compute total cost for a given predictor's residuals over the indexed samples.
/// Returns estimated bits (probability-floor formula) + total extra bits, weighted
/// by sample counts. Uses the same estimate_bits formula as the sweep child costs,
/// ensuring consistent cost comparison for split decisions.
///
/// `counts_buf` is a reusable histogram buffer (len >= histogram_size), cleared on entry.
///
/// Post-issue-#40-chunk-2: the sample set is identified by a contiguous range
/// `[start..end)` into the underlying SoA arrays (residual_tokens, extra_bits,
/// sample_counts). Callers maintain this contiguity via
/// `split_tree_samples_in_place` at partition time, so this loop is now a
/// pure linear scan over sequential memory instead of indexed random reads.
/// Debug-build verifier for the issue-#64 side-costs rider: asserts the
/// sweep-carried [`BestSplit::left_bits`] / [`BestSplit::right_bits`] are
/// bitwise-identical to the post-partition [`compute_predictor_entropy`]
/// recompute they replaced at the engine call sites. Uses `cfg!` (not
/// `#[cfg]`/`debug_assert!`) so the arguments stay borrowed on every build
/// profile (no unused-variable churn in release); the `if false` body is
/// dead-code-eliminated in release builds.
#[inline]
fn debug_verify_carried_side_bits(
    samples: &TreeSamples,
    split: &BestSplit,
    start: usize,
    mid: usize,
    end: usize,
    histogram_size: usize,
    counts_buf: &mut [u32],
) {
    if cfg!(debug_assertions) {
        let lb = compute_predictor_entropy(
            samples,
            start,
            mid,
            split.left_predictor,
            histogram_size,
            counts_buf,
        );
        let rb = compute_predictor_entropy(
            samples,
            mid,
            end,
            split.right_predictor,
            histogram_size,
            counts_buf,
        );
        assert_eq!(
            split.left_bits.to_bits(),
            lb.to_bits(),
            "BestSplit.left_bits diverged from post-split recompute"
        );
        assert_eq!(
            split.right_bits.to_bits(),
            rb.to_bits(),
            "BestSplit.right_bits diverged from post-split recompute"
        );
    }
}

fn compute_predictor_entropy(
    samples: &TreeSamples,
    start: usize,
    end: usize,
    predictor_idx: usize,
    histogram_size: usize,
    counts_buf: &mut [u32],
) -> f64 {
    let tokens = &samples.residual_tokens[predictor_idx][start..end];
    let ebits = &samples.extra_bits[predictor_idx][start..end];
    let sample_counts = &samples.sample_counts[start..end];
    counts_buf[..histogram_size].fill(0);
    let mut total = 0u32;
    let mut tot_extra: u64 = 0;

    // Zip-iterate: contiguous reads over three parallel slices. Bounds-check
    // elimination via the matched zip; no `[idx]` indexing into the parent
    // arrays.
    for ((&tok, &eb), &count) in tokens.iter().zip(ebits.iter()).zip(sample_counts.iter()) {
        let tok = tok as usize;
        if tok < histogram_size {
            counts_buf[tok] += count;
            total += count;
        }
        tot_extra += eb as u64 * count as u64;
    }

    jxl_simd::estimate_bits_u32(&counts_buf[..histogram_size], total) + tot_extra as f64
}

/// Partition the contiguous sample range `[start..end)` in-place so that rows
/// with `bucket_indices[prop_idx][i] <= bucket_split` occupy the left half
/// `[start..mid)` and rows with `> bucket_split` occupy `[mid..end)`.
///
/// `mid = start + left_count`, where `left_count` is the caller-supplied
/// number of unique samples on the left side (computed by `find_best_split`
/// from `bucket_starts[local_k + 1]`).
///
/// All parallel SoA arrays (per-predictor `residual_tokens` and `extra_bits`,
/// per-property `props` and `bucket_indices`, `sample_counts`) are permuted
/// as atomic rows by `split_tree_samples_in_place`, preserving row alignment
/// across the partition boundary. This is the chunk-2 wiring of the chunk-1
/// `tree_learn_split` primitive (issue #40).
///
/// `bucket_split` is the **bucket index** on which to partition (matches the
/// pre-quantized space `find_best_split` operates in). The bucket-equivalent
/// of the cost-model split threshold is `local_k` (the sweep step at which
/// the split was chosen); the bucket value `bucket_split = bmin + local_k`
/// is recovered from the split's `splitval` via `threshold_set`.
///
/// # Returns
/// Absolute mid index = `start + left_count`.
fn partition_node_in_place(
    samples: &mut TreeSamples,
    pq: &mut PreQuantizedProps,
    start: usize,
    end: usize,
    left_count: usize,
    key: tree_learn_split::PartitionKey,
) -> usize {
    partition_node_in_place_with(samples, pq, start, end, left_count, key, false)
}

/// Issue #40 chunk-3c: env-var override `JXL_DISABLE_CHUNK3C=1` forces the
/// props-swapping path even when the caller asked to skip it. Used by the
/// paired A/B bench harness to compare BASELINE (props swap) vs NEW (skip)
/// using the same binary. Cached after first lookup so production pays one
/// `std::env::var` call per process, not per partition.
#[cfg(feature = "std")]
#[inline]
fn chunk3c_skip_is_disabled() -> bool {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("JXL_DISABLE_CHUNK3C")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

#[cfg(not(feature = "std"))]
#[inline]
fn chunk3c_skip_is_disabled() -> bool {
    false
}

/// Issue #40 chunk-3c: variant of [`partition_node_in_place`] that takes the
/// `skip_props_swap` flag explicitly. Callers on the lossless main path pass
/// `true` to elide the per-property `Vec<i32>` swaps. See
/// [`tree_learn_split::SplittableSamples::skip_props_swap`] for safety
/// conditions.
fn partition_node_in_place_with(
    samples: &mut TreeSamples,
    pq: &mut PreQuantizedProps,
    start: usize,
    end: usize,
    left_count: usize,
    key: tree_learn_split::PartitionKey,
    skip_props_swap: bool,
) -> usize {
    debug_assert!(left_count <= end - start);
    let num_samples = samples.num_samples;
    let skip_props_swap = skip_props_swap && !chunk3c_skip_is_disabled();
    let mut view = tree_learn_split::SplittableSamples {
        residual_tokens: &mut samples.residual_tokens,
        extra_bits: &mut samples.extra_bits,
        props: &mut samples.props,
        bucket_indices: &mut pq.bucket_indices,
        sample_counts: &mut samples.sample_counts,
        len: num_samples,
        skip_props_swap,
    };
    let pos = start + left_count;
    tree_learn_split::split_tree_samples_in_place(&mut view, start, pos, end, key);
    pos
}

/// Look up the bucket index for a given threshold value in a pre-quantized
/// property's threshold set. Returns the index `k` such that
/// `threshold_set[k] == splitval`, or panics if not found.
///
/// `find_best_split` always emits `splitval = threshold_set[global_k]`, so the
/// reverse lookup is exact (no rounding, no off-by-one).
fn bucket_for_splitval(threshold_set: &[i32], splitval: i32) -> usize {
    threshold_set
        .iter()
        .position(|&t| t == splitval)
        .expect("splitval came from threshold_set; reverse lookup must succeed")
}

/// Collect residuals using a learned tree for encoding.
///
/// For each pixel: gather neighbors → compute spec properties → traverse tree →
/// predict using leaf's predictor → pack_signed → produce AnsToken with
/// context = leaf.context_id and value = raw packed residual.
///
/// The raw packed residual is stored as the token value. The HybridUint encoding
/// is applied later by `build_entropy_code_ans` (for histogram building) and
/// `write_tokens_ans` (for bitstream writing) — both use UintCoder which implements
/// HybridUint {4,2,0}.
pub fn collect_residuals_with_tree(
    image: &ModularImage,
    tree: &Tree,
    group_id: u32,
    wp_params: &WeightedPredictorParams,
) -> Vec<crate::entropy_coding::token::Token> {
    collect_residuals_with_tree_offset(image, tree, group_id, 0, wp_params)
}

/// `collect_residuals_with_tree` with explicit allocation budget.
///
/// Per-channel `WeightedPredictorState` scratch is reserved against the
/// cap. Returns [`crate::error::Error::AllocationLimit`] when the cap is
/// exceeded. `budget = None` is zero-overhead.
pub(crate) fn collect_residuals_with_tree_with_budget(
    image: &ModularImage,
    tree: &Tree,
    group_id: u32,
    wp_params: &WeightedPredictorParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Vec<crate::entropy_coding::token::Token>> {
    collect_residuals_with_tree_offset_with_budget(image, tree, group_id, 0, wp_params, budget)
}

/// Collect residuals using a learned tree, with a channel index offset.
///
/// When collecting from a sub-image that represents channels [offset..offset+N] of a larger
/// image, pass `channel_offset = offset` so property[0] (channel index) matches the tree
/// that was trained on the full image.
pub fn collect_residuals_with_tree_offset(
    image: &ModularImage,
    tree: &Tree,
    group_id: u32,
    channel_offset: u32,
    wp_params: &WeightedPredictorParams,
) -> Vec<crate::entropy_coding::token::Token> {
    // Budget-less call cannot trip the AllocationLimit branch. The only
    // other error path is `Error::InvalidInput("Residual overflow ...")`
    // from the fuzz-hardening guard (mirrors libjxl `SubOverflow` in
    // `EncodeModularChannelMAANS`, commit `87bee19`); valid-channel
    // input never reaches it. The .expect propagates a descriptive
    // panic message if either invariant is ever violated.
    collect_residuals_with_tree_offset_with_budget(
        image,
        tree,
        group_id,
        channel_offset,
        wp_params,
        None,
    )
    .expect("budget-less collect_residuals_with_tree_offset: no AllocationLimit, no residual overflow on valid input")
}

/// `collect_residuals_with_tree_offset` with explicit allocation budget.
///
/// Per-channel `WeightedPredictorState` scratch is reserved against the
/// cap. `budget = None` is zero-overhead.
pub(crate) fn collect_residuals_with_tree_offset_with_budget(
    image: &ModularImage,
    tree: &Tree,
    group_id: u32,
    channel_offset: u32,
    wp_params: &WeightedPredictorParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Vec<crate::entropy_coding::token::Token>> {
    use crate::entropy_coding::token::Token as AnsToken;

    // Check if the tree uses any reference channel properties (indices >= 16).
    // If so, we need to compute extended properties per pixel.
    let max_tree_prop = tree
        .iter()
        .filter(|n| n.property >= 0)
        .map(|n| n.property as usize)
        .max()
        .unwrap_or(0);
    let needs_ref_props = max_tree_prop >= NUM_PROPERTIES;

    // Pre-size to the exact token count (one per pixel across channels):
    // capacity only — byte-identical; the per-pixel pushes otherwise grow
    // through doubling reallocs (part of the 10.9 % __memmove share).
    let total_px: usize = image.channels.iter().map(|c| c.width() * c.height()).sum();
    let mut tokens = Vec::with_capacity(total_px);

    // Width of one pixel's property record (row-major in `props_row`).
    //
    // #68: round the ref-property tail up to a WHOLE 4-slot ref-group.
    // The per-pixel writer fills ref groups four-at-a-time behind a
    // `base + 3 < prop_stride` guard; with the stride cut at
    // `max_tree_prop + 1`, a tree whose maximum property id lands
    // mid-group (id % 4 != 3 — only reachable at e9+, where the learner
    // may split on slots 0..2) made that guard skip the ENTIRE top
    // group, so the walk read zeros where every decoder computes real
    // values: context desync → truncated-section EOF in zenjxl-decoder
    // / jxl-oxide / djxl. e≤8 trees only ever split on slot 3, whose
    // group is always complete — which is why this was e9+-only.
    let num_extended_props = if needs_ref_props {
        let ref_tail = max_tree_prop + 1 - NUM_PROPERTIES;
        NUM_PROPERTIES + ref_tail.div_ceil(4) * 4
    } else {
        NUM_PROPERTIES
    };

    for (ch_idx, channel) in image.channels.iter().enumerate() {
        let width = channel.width();
        let height = channel.height();
        if width == 0 || height == 0 {
            continue;
        }

        // Find reference channels for this channel
        let ref_channel_indices = if needs_ref_props {
            find_ref_channels(image, ch_idx)
        } else {
            Vec::new()
        };

        let mut wp_state = WeightedPredictorState::new_with_budget(wp_params, width, budget)?;
        let mut prev_gradient: i32;

        // ── Issue #41 chunk B2: 3-pass row restructure ──────────────────
        // Pass 1 walks the row in the exact legacy per-pixel order
        // (neighbors → WP predict/property → spec(+ref) properties → WP
        // error update) storing per-pixel state into row buffers; pass 2
        // resolves all leaf indices with an ILP-friendly interleaved tree
        // walk ([`batch_traverse_row`] — the serial dependent-load walk
        // was 8.7 % of CPU per the step-0 annotate); pass 3 emits tokens
        // in the legacy order. Leaf choice is a pure function of (tree,
        // props), and props/WP state are produced in the identical
        // per-pixel order, so the output is byte-identical.
        //
        // Buffers are fresh per channel: width varies, and a zeroed
        // buffer gives the ref-prop tail its invariant zeros (per-pixel
        // writes only touch [..16 + 4*num_refs)).
        let prop_stride = num_extended_props;
        let mut props_row: Vec<i32> = vec![0; prop_stride * width];
        let mut wp_pred_row: Vec<i64> = vec![0; width];
        let mut neigh_row: Vec<Neighbors> = vec![Neighbors::default(); width];
        let mut pixel_row: Vec<i32> = vec![0; width];
        let mut leaf_row: Vec<u32> = vec![0; width];

        for y in 0..height {
            prev_gradient = 0;

            // Pass 1: neighbors + WP + properties, in the legacy
            // per-pixel order, stored into the row buffers.
            for x in 0..width {
                let pixel = channel.get(x, y);
                let n = Neighbors::gather(channel, x, y);

                // Fused WP predict + error update (issue #41 item 2):
                // nothing between the legacy predict(x) and update(x)
                // read WP state, so fusing preserves the sequence.
                let (wp_pred, wp_max_error) =
                    wp_state.predict_property_update(pixel, x, y, width, &n);

                let row_props = &mut props_row[x * prop_stride..x * prop_stride + prop_stride];
                compute_spec_properties_into(
                    (&mut row_props[..NUM_PROPERTIES])
                        .try_into()
                        .expect("prefix is exactly NUM_PROPERTIES wide"),
                    ch_idx as u32 + channel_offset,
                    group_id,
                    x,
                    y,
                    &n,
                    prev_gradient,
                    wp_max_error,
                );
                prev_gradient = row_props[9];

                if needs_ref_props {
                    // Compute reference channel properties
                    for (r, &ref_ch_idx) in ref_channel_indices.iter().enumerate() {
                        let ref_ch = &image.channels[ref_ch_idx];
                        let v = ref_ch.get(x, y);
                        let ref_left = if x > 0 { ref_ch.get(x - 1, y) } else { 0 };
                        let ref_top = if y > 0 {
                            ref_ch.get(x, y - 1)
                        } else {
                            ref_left
                        };
                        let ref_topleft = if x > 0 && y > 0 {
                            ref_ch.get(x - 1, y - 1)
                        } else {
                            ref_left
                        };
                        let ref_predicted = crate::vardct::dc_coding::clamped_gradient(
                            ref_top,
                            ref_left,
                            ref_topleft,
                        );

                        let base = NUM_PROPERTIES + r * 4;
                        if base + 3 < prop_stride {
                            row_props[base] = v.wrapping_abs();
                            row_props[base + 1] = v;
                            row_props[base + 2] = v.wrapping_sub(ref_predicted).wrapping_abs();
                            row_props[base + 3] = v.wrapping_sub(ref_predicted);
                        }
                    }
                }

                wp_pred_row[x] = wp_pred;
                neigh_row[x] = n;
                pixel_row[x] = pixel;
            }

            // Pass 2: resolve every pixel's leaf with the interleaved walk.
            batch_traverse_row(tree, &props_row, prop_stride, &mut leaf_row);

            // Pass 3: emit tokens in the legacy order.
            for x in 0..width {
                let leaf = &tree[leaf_row[x] as usize];

                // Predict using leaf's predictor
                let prediction = if leaf.predictor == Predictor::Weighted {
                    wp_pred_row[x] as i32
                } else {
                    leaf.predictor.predict_from_neighbors(&neigh_row[x])
                };
                // Fuzz-hardening: reject i32-overflowing residuals before
                // they corrupt token stream. Mirrors libjxl `SubOverflow`
                // guard in `EncodeModularChannelMAANS`
                // (`modular/encoding/enc_encoding.cc:307`, commit `87bee19`).
                // Valid input never trips this (predictor range is
                // bounded by channel range), so the fast path is one
                // branch on success.
                let residual = super::fuzz_safety::checked_residual(pixel_row[x], prediction)?;

                // Divide by multiplier for lossy modular quantization.
                // When multiplier > 1, pixels have been pre-quantized to multiples of q
                // and the tree forces splits so each leaf's multiplier matches the
                // channel's quantizer. The decoder reconstructs:
                //   pixel = unpack_signed(token) * multiplier + prediction
                let multiplier = leaf.multiplier;
                let divided = if multiplier == 1 {
                    residual
                } else {
                    debug_assert!(
                        residual % multiplier == 0,
                        "residual {} not divisible by multiplier {} at ({},{}) ch={}",
                        residual,
                        multiplier,
                        x,
                        y,
                        ch_idx,
                    );
                    residual / multiplier
                };
                let packed = pack_signed(divided);

                // Store raw packed residual — UintCoder (HybridUint {4,2,0}) encoding
                // is applied by build_entropy_code_ans and write_tokens_ans
                tokens.push(AnsToken::new(leaf.context_id, packed));
            }
        }
    }

    Ok(tokens)
}

/// Issue #41 chunk B2: resolve a whole row's MA-tree leaves with an
/// interleaved K-lane walk.
///
/// The scalar traversal is a serial dependent-load chain per pixel
/// (node → property value → compare → child index → next node), measured
/// at 8.7 % of CPU inlined into `collect_residuals_with_tree*`
/// (`benchmarks/perf_gather_profile_2026-06-10.meta` addendum). Walking
/// K pixels at once overlaps K independent chains: each loop iteration
/// advances every still-active lane one level; lanes that reached a leaf
/// idle on the `property < 0` check until the slowest lane finishes.
/// Identical leaf selection to the scalar walk — same compares, same
/// child links — just a different evaluation order across pixels.
///
/// `props_row` is row-major: pixel `x`'s properties live at
/// `[x * stride .. (x + 1) * stride)`. Leaf node INDICES are written to
/// `out` (callers re-borrow the node — indices avoid aliasing the tree
/// borrow across the pass boundary).
fn batch_traverse_row(tree: &Tree, props_row: &[i32], stride: usize, out: &mut [u32]) {
    const K: usize = 8;
    let w = out.len();
    debug_assert!(props_row.len() >= w * stride);

    let mut x = 0;
    while x + K <= w {
        let mut idxs = [0usize; K];
        loop {
            let mut active = false;
            for (k, idx) in idxs.iter_mut().enumerate() {
                let node = &tree[*idx];
                if node.property >= 0 {
                    active = true;
                    let pval = props_row[(x + k) * stride + node.property as usize];
                    *idx = if pval <= node.splitval {
                        node.lchild
                    } else {
                        node.rchild
                    };
                }
            }
            if !active {
                break;
            }
        }
        for (k, &idx) in idxs.iter().enumerate() {
            out[x + k] = idx as u32;
        }
        x += K;
    }

    // Scalar tail (row % K pixels) — same walk, one lane.
    while x < w {
        let mut idx = 0usize;
        loop {
            let node = &tree[idx];
            if node.property < 0 {
                break;
            }
            let pval = props_row[x * stride + node.property as usize];
            idx = if pval <= node.splitval {
                node.lchild
            } else {
                node.rchild
            };
        }
        out[x] = idx as u32;
        x += 1;
    }
}

/// Traverse a tree using spec-matching property values (base 16 properties only).
///
/// Our tree convention: lchild = property <= splitval, rchild = property > splitval.
///
/// Production traversal moved to [`batch_traverse_row`] (issue #41 chunk
/// B2); this scalar walk stays as the test-side reference verifier.
#[cfg(test)]
fn traverse_with_spec_props<'a>(
    tree: &'a Tree,
    props: &[i32; NUM_PROPERTIES],
) -> &'a PropertyDecisionNode {
    let mut idx = 0;
    loop {
        let node = &tree[idx];
        if node.property < 0 {
            return node;
        }
        let pval = props[node.property as usize];
        if pval <= node.splitval {
            idx = node.lchild;
        } else {
            idx = node.rchild;
        }
    }
}

/// Traverse a tree using a dynamic-length property slice.
///
/// Used when reference channel properties (indices >= 16) are present in the tree.
/// Falls back to the same traversal logic but with a slice instead of a fixed array.
///
/// Production traversal moved to [`batch_traverse_row`] (issue #41 chunk
/// B2); kept as a test-side reference verifier.
#[cfg(test)]
#[allow(dead_code)]
fn traverse_with_props<'a>(tree: &'a Tree, props: &[i32]) -> &'a PropertyDecisionNode {
    let mut idx = 0;
    loop {
        let node = &tree[idx];
        if node.property < 0 {
            return node;
        }
        let pval = props[node.property as usize];
        if pval <= node.splitval {
            idx = node.lchild;
        } else {
            idx = node.rchild;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RFC#45 chunk 2 — multi-seed tree learning (e10/e11)
// ────────────────────────────────────────────────────────────────────────────

/// Estimate the encoded bit cost of a token stream produced by
/// [`collect_residuals_with_tree`], for the purposes of comparing trees.
///
/// Per-context histograms are built over the HybridUint{4,2,0}-encoded
/// `token` of each [`crate::entropy_coding::token::Token`]. Cost is the
/// libjxl-parity `estimate_bits` (probability floor 1/4096) plus the
/// extra bits encoded outside the histogram.
///
/// Includes a coarse per-context header term (`~50 bits` per non-empty
/// context — the rough average ANS-histogram header size in our
/// production builds) to discourage trees that pile up many small
/// contexts. Without this the picker would happily choose deeper trees
/// that lower per-token entropy but bloat the ANS header.
///
/// This is the same cost model `compute_best_tree` uses internally for
/// its split decisions, with the header term added. It is a proxy for
/// the eventual ANS bitstream size — accurate to within a few percent
/// in our measurements on CLIC photos.
pub fn estimate_token_cost(tokens: &[crate::entropy_coding::token::Token]) -> f64 {
    use crate::entropy_coding::hybrid_uint::HybridUintConfig;
    const MODULAR_HYBRID_UINT: HybridUintConfig = HybridUintConfig {
        split_exponent: 4,
        split: 16,
        msb_in_token: 2,
        lsb_in_token: 0,
    };
    /// Rough per-context ANS-histogram header cost (libjxl encodes flat
    /// histograms in ~8 bits, but typical post-clustering histograms run
    /// 30–80 bits + ~8 bits context-map entry). 50 is a compromise that
    /// kills false-positive picks where two trees differ only in
    /// trivial leaf splits.
    const HEADER_BITS_PER_CONTEXT: f64 = 50.0;

    if tokens.is_empty() {
        return 0.0;
    }

    // Per-context histograms keyed by context index.
    // Token range for HybridUint{4,2,0} is small (<= ~55 for typical
    // 8-bit residuals); we let Vec grow on demand rather than fixing
    // a max.
    let mut per_context: Vec<Vec<u32>> = Vec::new();
    let mut per_context_total: Vec<u32> = Vec::new();
    let mut extra_bits_total: u64 = 0;

    for tok in tokens {
        let (sym, _bits, nbits) = MODULAR_HYBRID_UINT.encode(tok.value);
        let ctx = tok.context() as usize;
        if ctx >= per_context.len() {
            per_context.resize(ctx + 1, Vec::new());
            per_context_total.resize(ctx + 1, 0);
        }
        let sym_u = sym as usize;
        if sym_u >= per_context[ctx].len() {
            per_context[ctx].resize(sym_u + 1, 0);
        }
        per_context[ctx][sym_u] += 1;
        per_context_total[ctx] += 1;
        extra_bits_total += nbits as u64;
    }

    let mut bits = extra_bits_total as f64;
    let mut nonempty_contexts: u64 = 0;
    for (counts, &total) in per_context.iter().zip(per_context_total.iter()) {
        if total > 0 {
            bits += jxl_simd::estimate_bits_scalar_f64(counts, total);
            nonempty_contexts += 1;
        }
    }
    bits += nonempty_contexts as f64 * HEADER_BITS_PER_CONTEXT;
    bits
}

/// Per-seed parameter variance for the multi-seed tree-learning loop
/// (RFC#45 chunk 3 — broader seed variance follow-on to chunk 2).
///
/// Chunk 2 only varied `start_offset` (which pixels feed
/// [`gather_samples_strided_with_offset`]). On 3 CID22 photos that turned
/// out to be too narrow — seed 0 always won because the sample subsets
/// were highly correlated. This helper widens the candidate space by
/// jittering two greedy-ID3 knobs that materially change which splits
/// survive the threshold gate:
///
/// 1. **`split_threshold` jitter**: per-seed multiplier in
///    `{1.0, 0.7, 1.3, 0.85}` (cycled by `seed % 4`). Lower thresholds
///    accept marginal splits the canonical run would reject; higher
///    thresholds force a smaller / shallower tree that may compress
///    better when the corpus is noisy.
/// 2. **Property-order shuffle**: cycles a small deterministic
///    permutation past the structural prefix (`Channel`, optionally
///    `GroupId`). Greedy ID3 evaluates properties in `params.properties`
///    order — re-ordering changes tie-breaks at split selection and
///    surfaces trees built around different "first cut" properties.
///
/// `seed == 0` is **always** a no-op: returns a clone of `base`. This
/// preserves the chunk-2 invariant that seed 0 equals the legacy
/// single-pass pipeline, which keeps e ≤ 9 hash-locks byte-identical
/// (e ≤ 9 has `tree_learn_seeds = 1` and therefore never enters this
/// helper).
///
/// Greedy-ID3 still picks the lowest-cost split at every node, so each
/// seed produces a *spec-valid* tree. The chunk-2 [`estimate_token_cost`]
/// picker then chooses the cheapest among all candidates.
#[must_use]
pub fn derive_seeded_params(base: &TreeLearningParams, seed: u64) -> TreeLearningParams {
    let clone_params = |b: &TreeLearningParams| -> TreeLearningParams {
        TreeLearningParams {
            properties: b.properties.clone(),
            max_property_values: b.max_property_values,
            split_threshold: b.split_threshold,
            max_nodes: b.max_nodes,
            pixel_fraction: b.pixel_fraction,
            use_streaming_dedup: b.use_streaming_dedup,
            gather_dedup: b.gather_dedup,
            gather_dedup_phase3: b.gather_dedup_phase3,
            parallel_max_depth: b.parallel_max_depth,
            parallel_recursion_floor: b.parallel_recursion_floor,
            parallel_root_threshold: b.parallel_root_threshold,
            parallel_small_image_fallback: b.parallel_small_image_fallback,
            lloyd_max_buckets: b.lloyd_max_buckets,
        }
    };
    if seed == 0 {
        return clone_params(base);
    }
    let mut out = clone_params(base);

    // 1. split_threshold jitter (seed % 4).
    //    [1.0, 0.7, 1.3, 0.85] — multipliers chosen to span ±30% around
    //    baseline while keeping seed 0 unchanged. Lower → accept more
    //    splits → deeper tree; higher → reject more → shallower tree.
    const THRESHOLD_MUL: [f64; 4] = [1.0, 0.7, 1.3, 0.85];
    let mul = THRESHOLD_MUL[(seed as usize) % THRESHOLD_MUL.len()];
    out.split_threshold = base.split_threshold * mul;

    // 2. Property-order shuffle.
    //    Structural prefix stays put (`Channel` at index 0, optionally
    //    `GroupId` at index 1) — those carry the stream-multiplexing
    //    semantics every well-formed tree needs early. Past that, we
    //    apply a deterministic per-seed rotation: rotate the tail left
    //    by `(seed * 3) % tail.len()`. Rotation (rather than full
    //    shuffle) preserves locality between related gradient-difference
    //    properties (9..14) while still changing the first non-structural
    //    property the greedy split picks.
    let structural_prefix = if !out.properties.is_empty() && out.properties[0] == 0 {
        if out.properties.len() >= 2 && out.properties[1] == 1 {
            2 // Channel + GroupId
        } else {
            1 // Channel only
        }
    } else {
        0
    };
    if out.properties.len() > structural_prefix + 1 {
        let tail_len = out.properties.len() - structural_prefix;
        let rot = ((seed as usize).wrapping_mul(3)) % tail_len;
        if rot > 0 {
            let tail = &mut out.properties[structural_prefix..];
            tail.rotate_left(rot);
        }
    }

    out
}

/// Per-seed stride perturbation for the multi-seed tree-learning loop
/// (RFC#45 chunk 3).
///
/// Returns a stride for the given seed derived from the canonical
/// `base_stride`. Different strides change the *density* of the sample
/// pool (not just the offset within it), exposing the greedy ID3 builder
/// to a different ratio of unique-vs-duplicate samples — which in turn
/// changes which splits clear the `pixel_fraction`-scaled threshold gate.
///
/// `seed == 0` returns `base_stride` unchanged (preserves seed-0
/// bit-identicality with chunk 2). Higher seeds cycle through
/// `{base, base+1, base-1 (>=1), base*2}` while clamping to `>= 1`. The
/// `+1`/`-1` neighbors capture small density perturbations cheaply; the
/// `*2` variant doubles stride for a much sparser sample subset (faster
/// gather, more "skipped" pixels — surfaces a different split set on
/// highly-textured images).
#[must_use]
pub fn derive_seeded_stride(base_stride: usize, seed: u64) -> usize {
    if seed == 0 || base_stride == 0 {
        return base_stride.max(1);
    }
    let candidates: [usize; 4] = [
        base_stride,
        base_stride.saturating_add(1),
        base_stride.saturating_sub(1).max(1),
        base_stride.saturating_mul(2),
    ];
    candidates[(seed as usize) % candidates.len()].max(1)
}

/// Per-seed `tree_sample_fraction` override for the multi-seed tree-learning
/// loop (RFC#45 chunk 4 — sample-fraction variance follow-on to chunk 3,
/// gated to seeds 4..7 by chunk 5).
///
/// Returns the absolute sample fraction the gather should target for this
/// seed, or `None` to leave the canonical profile fraction untouched.
///
/// Layout (chunk 5):
/// - seeds 0..=3 → `None` (chunk-3-only perturbations apply; chunk-4
///   dimensions held to canonical)
/// - seed 4 → `Some(0.40)` (sparser sample set, faster gather)
/// - seed 5 → `Some(0.60)` (denser than canonical 0.50)
/// - seed 6 → `Some(0.70)` (densest, most expensive)
/// - seed 7 → `None` (canonical fraction; pairs with chunk-4's predictor
///   permutation #3)
/// - seed ≥ 8 → cycled by `(seed - 4) % 4`
///
/// The 0.40 / 0.60 / 0.70 triplet straddles the canonical 0.50 default
/// (set by [`EffortProfile::tree_sample_fraction_for`] at effort ≥ 7) and
/// adds one substantially denser sample (0.70) that captures rare-bucket
/// splits the canonical run misses. Density only matters when the gather
/// stride changes — at small images the stride is already 1 and the
/// override is effectively a no-op.
///
/// Seeds 0..=3 stay `None` so chunk 3's split_threshold-jitter and
/// property-order-rotation perturbations can hit their best minima
/// without being recombined with sample-fraction overrides — honest
/// W8-3-r2 benching showed combining all variance dimensions inside a
/// 4-seed budget regressed vs chunk 3 on 3 of 5 photos at e11.
/// Chunk 5 raised `tree_learn_seeds_for(11)` from 4 → 8 so chunk-4
/// dimensions get their own seed slots (4..7) on top of the chunk-3
/// candidates rather than replacing them.
///
/// e ≤ 9 has `tree_learn_seeds = 1` so this helper is never called
/// outside e10/e11.
#[must_use]
pub fn derive_seeded_sample_fraction(seed: u64) -> Option<f32> {
    if seed < 4 {
        return None;
    }
    const FRACTIONS: [Option<f32>; 4] = [Some(0.40), Some(0.60), Some(0.70), None];
    FRACTIONS[((seed - 4) as usize) % FRACTIONS.len()]
}

/// Per-seed predictor evaluation order for the multi-seed tree-learning
/// loop (RFC#45 chunk 4 — predictor-order variance follow-on to chunk 3,
/// gated to seeds 4..7 by chunk 5).
///
/// Layout (chunk 5):
/// - seeds 0..=3 → canonical [`CANDIDATE_PREDICTORS`] order (preserves
///   chunk-3-only seed slots — chunk-4 dimensions held to canonical so
///   chunk 3's threshold/property-rotation/stride perturbations can hit
///   their best minima cleanly)
/// - seeds 4..=7 → cycled through the four [`CANDIDATE_PREDICTORS_PERMS`]
///   permutations (canonical / strong-first / directional-first /
///   full-reverse)
/// - seed ≥ 8 → cycled by `(seed - 4) % 4`
///
/// Reason: [`find_best_predictor`] (and the parallel hybrid variant)
/// iterates the candidate list in array order and applies a strict-`<`
/// tie-break. Different orders therefore resolve ties differently and
/// can promote `Gradient` / `Weighted` (strong-first), the directional
/// `Average1..4` family (directional-first), or the cheap-residual
/// predictors (reverse) ahead of the canonical lowest-index winner.
///
/// All permutations contain the same 14 predictors as a set, so every
/// per-seed tree remains spec-valid and the chunk-2 picker still chooses
/// among them on equal terms. The chunk-5 gating preserves seed 0's
/// byte-identical hash-lock invariant (e ≤ 9 has `tree_learn_seeds = 1`
/// → never enters this helper anyway) and additionally fixes seeds 1..=3
/// to the canonical predictor order so chunk-3's perturbation set is
/// applied without chunk-4 interference.
#[must_use]
pub fn derive_seeded_predictor_order(seed: u64) -> &'static [Predictor] {
    if seed < 4 {
        return CANDIDATE_PREDICTORS_PERMS[0];
    }
    CANDIDATE_PREDICTORS_PERMS[((seed - 4) as usize) % CANDIDATE_PREDICTORS_PERMS.len()]
}

/// Convert an absolute target sample fraction into a gather stride for a
/// channel pool of `total_pixels` (RFC#45 chunk 4 helper).
///
/// Returns the stride that subsamples roughly `(total_pixels * fraction)`
/// pixels, clamped to the same 65 K-sample floor enforced by
/// [`max_tree_samples_from_profile`]. Returns `1` when the floor would
/// already cover the whole pool (no subsampling needed).
///
/// Used by the multi-seed section.rs loop to apply
/// [`derive_seeded_sample_fraction`] without mutating the shared
/// [`EffortProfile`]. Seed 0 still uses the canonical
/// [`compute_gather_stride_from_profile`] path.
#[must_use]
pub fn stride_for_seeded_sample_fraction(total_pixels: usize, fraction: f32) -> usize {
    let target = ((total_pixels as f32 * fraction) as usize).max(65_536);
    if total_pixels > target {
        total_pixels.div_ceil(target)
    } else {
        1
    }
}

/// Per-seed `max_property_values` override for the multi-seed tree-learning
/// loop (RFC#45 chunk 6 — split-bucket-count variance follow-on to chunk 5).
///
/// Returns an override bucket count for the given seed, or `None` to leave
/// the canonical `TreeLearningParams::max_property_values` (set by
/// [`EffortProfile::tree_max_buckets`]) untouched.
///
/// Layout (chunk 6):
/// - seeds 0..=7  → `None` (chunk-3 / chunk-4 / chunk-5 slots)
/// - seed 8       → `Some(64)`  (coarsest grid; 4× fewer split candidates)
/// - seed 9       → `Some(128)` (mid-coarse)
/// - seed 10      → `Some(192)` (just under canonical)
/// - seed 11      → `None`      (canonical; pairs with chunk-3 perm[3])
/// - seed ≥ 12    → `None` (chunk-6 truncation slot; bucket helper holds
///   canonical so the two chunk-6 dimensions never stack on a single
///   seed — preserves the chunk-3..chunk-6 seed-slot doctrine that each
///   chunk's dimension owns its own 4-seed block)
///
/// Why this dimension is orthogonal to chunks 3-5:
/// - Chunk 3 perturbs the *acceptance* threshold (which splits clear the
///   gate).
/// - Chunk 4 perturbs the *sample density* and *predictor evaluation
///   order* (which residuals the gate is computed over, and tie-break
///   order at split selection).
/// - Chunk 6 perturbs the *granularity of the split-value search* itself.
///   For each property, `find_best_split` quantizes the value range into
///   `max_property_values` buckets and picks the best boundary. With 256
///   buckets (canonical at e9+), the boundary that minimizes residual
///   entropy may sit between two coarse bins that a smaller grid would
///   have collapsed onto a different (and sometimes better) discrete
///   threshold. The token-cost picker keeps the cheapest of the 4
///   chunk-6 candidates, so this is purely a strict-≥ improvement.
///
/// The 64 / 128 / 192 / 256 sweep straddles the canonical 256 and adds
/// three substantially coarser variants. 256 stays as a "null" slot
/// (seed 11) so chunk-3 perm[3] gets a clean canonical-bucket run.
///
/// e ≤ 9 has `tree_learn_seeds = 1` so this helper is never called
/// outside e10/e11.
#[must_use]
pub fn derive_seeded_max_property_values(seed: u64) -> Option<usize> {
    if !(8..12).contains(&seed) {
        // Seeds < 8 → chunks 3-5 slots (canonical preserved).
        // Seeds >= 12 → chunk-6 truncation slot (canonical preserved
        // so the two chunk-6 dimensions never stack on a single seed).
        return None;
    }
    const BUCKETS: [Option<usize>; 4] = [Some(64), Some(128), Some(192), None];
    BUCKETS[(seed - 8) as usize]
}

/// Per-seed `properties`-slice truncation for the multi-seed tree-learning
/// loop (RFC#45 chunk 6 — property-set-size variance follow-on to chunk 5).
///
/// Returns the maximum number of leading properties from the canonical
/// `TreeLearningParams::properties` Vec the seeded run should consider,
/// or `None` to leave the canonical slice length untouched.
///
/// Layout (chunk 6):
/// - seeds 0..=11 → `None` (chunk-3 / chunk-4 / chunk-5 / chunk-6 bucket
///   slots)
/// - seed 12      → `Some(8)`  (smallest set; forces aggressive
///   regularization)
/// - seed 13      → `Some(10)` (Kitten-equivalent at e8)
/// - seed 14      → `Some(12)` (just below canonical 14 reference props)
/// - seed 15      → `None`     (canonical; pairs with chunk-3 perm[3])
/// - seed ≥ 16    → `None` (out of chunk-6 truncation budget; any future
///   chunk-7 dimension owns its own seed slots above 15 by symmetry
///   with chunks 3-6)
///
/// Why this dimension is orthogonal to chunks 3-5 and chunk-6 bucket:
/// - Chunks 3-5 vary *which* splits are tried among a fixed property
///   list and *how* their costs are computed.
/// - Chunk-6 buckets vary the *split-value granularity within* each
///   property.
/// - Chunk-6 truncation varies the *property-set size itself*. Reducing
///   the candidate property list forces the greedy ID3 builder to choose
///   from fewer high-information properties first — a form of structural
///   regularization that can outperform the full-property tree when the
///   canonical run over-fits late-tier properties (e.g., the
///   `WPMaxError`-derived properties at indices 10-15, which can chase
///   bucket noise on smooth content).
///
/// The 8 / 10 / 12 / 14 sweep covers Kitten / Wombat / pre-Tortoise /
/// near-Tortoise property-set sizes — three steps below canonical that
/// the multi-seed picker can fall back on when smaller is cheaper.
///
/// `Some(n)` is clamped to `properties.len()` at the consumer (see
/// section.rs) so an aggressive cap on a short property Vec is a no-op
/// rather than an out-of-range error. Truncation **preserves** the
/// structural prefix (Channel + GroupId) the rotation in
/// [`derive_seeded_params`] always keeps at the front, because the
/// canonical [`PROP_ORDER_NO_SQUEEZE`] places structural props at
/// indices 0-1 and truncation only drops from the tail.
///
/// e ≤ 9 has `tree_learn_seeds = 1` so this helper is never called
/// outside e10/e11.
#[must_use]
pub fn derive_seeded_properties_truncation(seed: u64) -> Option<usize> {
    if !(12..16).contains(&seed) {
        // Seeds < 12 → chunks 3-5 + chunk-6 bucket slots (canonical
        // preserved). Seeds >= 16 → out of chunk-6 budget.
        return None;
    }
    const PROP_CAPS: [Option<usize>; 4] = [Some(8), Some(10), Some(12), None];
    PROP_CAPS[(seed - 12) as usize]
}

/// Multi-seed early-out decision (RFC#45 chunk 7 — Pareto-aware wall-clock
/// short-circuit for the e10/e11 multi-seed tree-learning fan-out).
///
/// Examines the relative spread of the first `probe_seeds` token costs and
/// returns `true` when the spread is below
/// [`MULTI_SEED_EARLY_OUT_SPREAD_THRESHOLD`]. A tight chunk-3 cost cluster
/// indicates the per-image entropy structure is already well-pinned by the
/// chunk-3 perturbation slot; the remaining 12 seeds (chunks-4/5/6
/// dimensions) can still find marginal improvements, but the wall-clock
/// cost of running them outweighs the small expected byte savings.
///
/// Spread metric: `(max_cost - min_cost) / min_cost` over `costs[0..probe_seeds]`.
/// This is a relative-range measure (not std-dev) — small images where every
/// seed picks essentially the same tree show spread ≈ 0; images with high
/// chunk-3 variance show spread several percent.
///
/// `probe_seeds` is the number of completed seeds to examine; the caller is
/// responsible for calling this only after running at least that many seeds.
/// `total_seeds` is the budget; early-out only fires when
/// `probe_seeds < total_seeds` (i.e., there's still work to skip).
///
/// Returns `false` when fewer than 2 costs are available (no spread can be
/// computed) or when `probe_seeds >= total_seeds` (nothing to skip).
///
/// Trade-off (calibrated on chunk-6 paired bench, CID22-512 e11):
///
/// | image     | chunk-3 spread | full-16 finds better? | fires at 5% | bytes Δ |
/// |-----------|---------------|----------------------|------------|---------|
/// | 1025469   | 1.71%         | yes (cost -0.034%)   | yes        | +334 B  |
/// | 1044329   | 4.14%         | no                   | yes        |   0 B   |
/// | 1189261   | 2.59%         | no                   | yes        |   0 B   |
/// | 1279330   | 0.31%         | yes (cost -0.69%)    | yes        | +813 B  |
/// | 1418519   | 1.68%         | no                   | yes        |   0 B   |
///
/// Net: **+0.09% bytes regression, 3.36× wall-clock speedup at e11**. The
/// trade favours wall-clock heavily — equivalent to "e11 is now a fast e10+
/// instead of a slow exhaustive search" on images where the chunk-3 slot
/// has already pinned the solution.
///
/// **Important caveat**: low chunk-3 spread does NOT guarantee seeds 4..15
/// would not improve the picked minimum. Chunks-4/5/6 explore DIFFERENT
/// variance dimensions (sample fraction, predictor order, bucket count,
/// property truncation) than chunk-3 (threshold jitter, property rotation,
/// stride). The early-out is a heuristic that accepts a small bytes
/// regression on the rare cell where a later-slot seed would have won, in
/// exchange for large wall-clock savings on the majority of cells.
///
/// Bitstream invariant: byte counts can never IMPROVE vs no-early-out;
/// they match on cells where seeds 0..3 already contained the picker's
/// minimum and regress slightly on cells where a later seed would have won.
#[must_use]
pub fn multi_seed_early_out_after_probe(
    costs: &[f64],
    probe_seeds: usize,
    total_seeds: usize,
) -> bool {
    if probe_seeds >= total_seeds {
        return false;
    }
    let slice = match costs.get(..probe_seeds) {
        Some(s) if s.len() >= 2 => s,
        _ => return false,
    };
    let (mut min_cost, mut max_cost) = (slice[0], slice[0]);
    for &c in &slice[1..] {
        if c < min_cost {
            min_cost = c;
        }
        if c > max_cost {
            max_cost = c;
        }
    }
    // Guard against zero / negative costs (shouldn't happen for entropy
    // estimates of non-empty token streams, but be defensive).
    if min_cost <= 0.0 {
        return false;
    }
    let spread = (max_cost - min_cost) / min_cost;
    spread < MULTI_SEED_EARLY_OUT_SPREAD_THRESHOLD
}

/// Relative cost-spread threshold for [`multi_seed_early_out_after_probe`]
/// (RFC#45 chunk 7).
///
/// `0.05` = 5% spread. Calibrated against per-seed cost dumps from the
/// chunk-6 paired bench (CID22-512 photos, e11, 16 seeds):
///
/// | image    | chunk-3 spread | improvement from seeds 4..15 |
/// |----------|---------------|----------------------------|
/// | 1025469  | 1.71%         | 0.034% (cost), 0 B (bytes)  |
/// | 1189261  | 2.59%         | 0.000%                      |
/// | 1044329  | 4.14%         | 0.000%                      |
/// | 1279330  | 25.93%        | 0.781% (the one improving cell) |
///
/// The 5% threshold cleanly separates the 3 converged images (skip seeds
/// 4..15 → 4-image search) from the 1 high-variance image (keep all 16).
/// On converged images, the picker's best-of-4 tree (the same tree the
/// e10 2-seed path picks for that image, since e10 uses seeds 0..=1) is
/// expected to produce bytes within ≤ 0.05% of the best-of-16 tree.
///
/// The threshold is a relative-range measure (max-min)/min over the first
/// `MULTI_SEED_EARLY_OUT_PROBE_SEEDS` token costs. Strict-`<` comparison
/// means a spread of exactly 5% does not fire (preserves chunk-6 behaviour
/// on borderline cells).
///
/// A tighter threshold (e.g., 1%) never fires on this corpus → no
/// wall-clock win. A looser threshold (e.g., 30%) would fire on every
/// image including 1279330, losing the 0.78% cost improvement that
/// chunk-6 captured. 5% is calibrated to the empirical bimodal split.
pub const MULTI_SEED_EARLY_OUT_SPREAD_THRESHOLD: f64 = 0.05;

/// Number of "probe" seeds the multi-seed loop runs before consulting
/// [`multi_seed_early_out_after_probe`] (RFC#45 chunk 7).
///
/// `4` covers the chunk-3 seed slot (chunk-3 perturbations: split_threshold
/// jitter, property-order rotation, per-seed stride) which is the variance
/// dimension most coupled to per-image entropy structure. Spread over these
/// 4 candidates is the strongest single-source signal for whether the
/// remaining 12 seeds (chunks-4/5/6 dimensions) can find further improvements.
pub const MULTI_SEED_EARLY_OUT_PROBE_SEEDS: usize = 4;

/// Multi-seed tree-learning fan-out (RFC#45 chunk 2 — pick #1).
///
/// Runs `gather_fn` once per seed in `0..seeds`, hands the resulting
/// [`TreeSamples`] to [`compute_best_tree`], then encodes residuals
/// with [`collect_residuals_with_tree`] on `image` to score the
/// candidate tree by [`estimate_token_cost`]. Returns the
/// `(tree, tokens, cost)` tuple of the cheapest candidate.
///
/// `seeds <= 1` short-circuits to a single run — byte-equivalent to
/// calling `gather_fn(0)` + `compute_best_tree` + `collect_residuals_with_tree`
/// directly. e ≤ 9 keeps `tree_learn_seeds = 1` (see
/// [`crate::effort::EffortProfile::tree_learn_seeds`]), so the default
/// path stays bit-identical to the pre-chunk-2 hash-locks.
///
/// The picker is greedy: each seed's full pipeline runs sequentially.
/// At `seeds = 4` (e11) wall-clock is roughly 4× the e9 tree-learning
/// budget — acceptable for the "longest-search" effort levels.
///
/// `gather_fn` signature: `Fn(seed: u64) -> TreeSamples`. It is
/// responsible for gathering with the appropriate per-seed stride
/// offset (or other seed-derived variation). The default
/// section.rs / encode.rs callers wire it to
/// [`gather_samples_strided_with_offset`] with
/// `start_offset = (seed as usize) % stride.max(1)`.
///
/// Bitstream-validity: every candidate tree is a normal, spec-valid
/// JXL tree; the picker just chooses among them. djxl / jxl-rs / jxl-oxide
/// decode every candidate identically.
///
/// Public API helper exposed for downstream multi-seed wiring (RFC#45
/// chunk 2). The current `section.rs` / `encode.rs` callers use an
/// inline multi-seed loop with cost-tracked entropy estimation; this
/// helper offers the simpler "pick best by token cost" path for
/// experimental harnesses and e10/e11 multi-seed experiments. Default
/// clippy reports it as unused because production routes through the
/// inline loop.
#[allow(dead_code)]
pub fn select_best_tree_multi_seed<F>(
    seeds: u8,
    image: &ModularImage,
    group_id: u32,
    wp_params: &WeightedPredictorParams,
    params: &TreeLearningParams,
    gather_fn: F,
) -> (Tree, Vec<crate::entropy_coding::token::Token>, f64)
where
    F: Fn(u64) -> TreeSamples,
{
    let n = seeds.max(1) as u64;
    let mut best: Option<(Tree, Vec<crate::entropy_coding::token::Token>, f64)> = None;

    for seed in 0..n {
        let mut samples = gather_fn(seed);
        let tree = compute_best_tree(&mut samples, params);
        let tokens = collect_residuals_with_tree(image, &tree, group_id, wp_params);
        let cost = estimate_token_cost(&tokens);

        crate::trace::debug_eprintln!(
            "MULTI_SEED_TREE seed={}/{} cost={:.0} bits ({} tokens, {} nodes)",
            seed,
            n,
            cost,
            tokens.len(),
            tree.len(),
        );

        match best {
            None => best = Some((tree, tokens, cost)),
            Some((_, _, prev_cost)) if cost < prev_cost => {
                best = Some((tree, tokens, cost));
            }
            _ => {}
        }
    }

    best.expect("seeds >= 1 guarantees at least one candidate")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::channel::ModularImage;

    #[test]
    fn test_lloyd_max_thresholds_monotone() {
        // Skewed energy-like distribution (lots of low values, few high).
        // Verify the returned thresholds are strictly increasing and stay
        // inside the (min, max] half-open range so bucket assignment is
        // well-defined for downstream binary_search.
        let mut samples: Vec<i32> = Vec::new();
        samples.extend(core::iter::repeat_n(0i32, 1000));
        samples.extend(core::iter::repeat_n(1i32, 500));
        samples.extend(core::iter::repeat_n(10i32, 100));
        samples.extend(core::iter::repeat_n(50i32, 10));
        samples.extend(core::iter::repeat_n(200i32, 2));
        let ts = lloyd_max_thresholds(&samples, 0, 200, 6);
        assert!(!ts.is_empty(), "expected non-empty thresholds");
        for w in ts.windows(2) {
            assert!(
                w[0] < w[1],
                "thresholds must be strictly increasing: {ts:?}"
            );
        }
        for &t in &ts {
            assert!(t > 0, "threshold must be > min_val (0): {t}");
            assert!(t <= 200, "threshold must be <= max_val (200): {t}");
        }
        assert!(
            ts.len() <= 6,
            "threshold count must be <= max_buckets (6), got {}",
            ts.len()
        );
    }

    #[test]
    fn test_lloyd_max_thresholds_constant_property() {
        // All samples equal — no actionable buckets.
        let samples = vec![42i32; 1000];
        let ts = lloyd_max_thresholds(&samples, 42, 42, 8);
        assert!(
            ts.is_empty(),
            "constant property must produce no thresholds"
        );
    }

    #[test]
    fn test_lloyd_max_thresholds_two_clusters() {
        // Bimodal distribution: 500 samples at value=10, 500 at value=100.
        // Lloyd-Max with k=2 cells should land the single threshold near 55.
        let mut samples: Vec<i32> = Vec::with_capacity(1000);
        samples.extend(core::iter::repeat_n(10i32, 500));
        samples.extend(core::iter::repeat_n(100i32, 500));
        let ts = lloyd_max_thresholds(&samples, 10, 100, 1);
        assert_eq!(ts.len(), 1, "k=2 cells → 1 threshold");
        // The optimal Lloyd-Max midpoint is (10 + 100) / 2 = 55; tolerate
        // ±2 input units of clustering noise from the count-weighted init.
        assert!(
            (ts[0] - 55).abs() <= 2,
            "expected threshold near 55, got {}",
            ts[0]
        );
    }

    #[test]
    fn test_lloyd_max_thresholds_clamps_to_max_buckets() {
        // 100 distinct values, max_buckets=4 → at most 4 thresholds.
        let samples: Vec<i32> = (0..100).collect();
        let ts = lloyd_max_thresholds(&samples, 0, 99, 4);
        assert!(
            ts.len() <= 4,
            "threshold count must be <= max_buckets, got {}",
            ts.len()
        );
        assert!(!ts.is_empty());
        for w in ts.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_lloyd_max_thresholds_partition_samples() {
        // Verify the resulting thresholds + binary_search produce a
        // valid bucket index for every sample (the post-pre_quantize
        // contract). Uses an energy-shaped distribution from
        // [0, 255].
        let mut samples: Vec<i32> = Vec::new();
        for i in 0..256i32 {
            // Triangular falloff: lots of small values, few large.
            let count = (256 - i).max(1) as usize;
            samples.extend(core::iter::repeat_n(i, count));
        }
        let min = 0;
        let max = 255;
        let ts = lloyd_max_thresholds(&samples, min, max, 8);
        assert!(!ts.is_empty());
        let num_buckets = ts.len() + 1;
        let mut bucket_counts = vec![0u64; num_buckets];
        for &v in &samples {
            let bucket = match ts.binary_search(&v) {
                Ok(pos) => pos,
                Err(pos) => pos,
            };
            let bucket = bucket.min(ts.len());
            bucket_counts[bucket] += 1;
        }
        // No bucket should be empty in an energy-shaped distribution
        // with k <= num_unique — Lloyd-Max would have re-centered any
        // empty cell during the iteration.
        for (i, &c) in bucket_counts.iter().enumerate() {
            assert!(c > 0, "bucket {i} unexpectedly empty (thresholds {ts:?})");
        }
    }

    #[test]
    fn test_estimate_bits_uniform() {
        // 4 symbols each appearing 100 times: entropy = 4 * 100 * log2(4) = 800
        let counts = [100u32, 100, 100, 100];
        let total = 400;
        let bits = estimate_bits(&counts, total);
        assert!(
            (bits - 800.0).abs() < 0.01,
            "expected 800 bits, got {}",
            bits
        );
    }

    #[test]
    fn test_estimate_bits_single_symbol() {
        // 1 symbol appearing 100 times: entropy ≈ 0 (or very small due to floor)
        let counts = [100u32];
        let total = 100;
        let bits = estimate_bits(&counts, total);
        // With prob floor, -100 * log2(1.0) = 0
        assert!(
            bits < 1.0,
            "single symbol should have near-zero entropy, got {}",
            bits
        );
    }

    #[test]
    fn test_gather_samples_simple() {
        // 4x4 constant image: all residuals should be 0 (predictor matches)
        let image = ModularImage::from_gray8(&[128u8; 16], 4, 4).unwrap();
        let mut samples = TreeSamples::new();
        gather_samples(&mut samples, &image, 0);

        assert_eq!(samples.num_samples, 16);
        // All predictors should produce token 0 for constant image (residual=0 except first pixel)
        // First pixel (0,0) has pred=0 for most predictors, pixel=128, residual=128
        // But Gradient: left=0, top=0, tl=0 → pred=0, residual=128
        // So not all tokens are 0
    }

    #[test]
    fn test_compute_best_tree_constant() {
        // Constant image: tree should be a single leaf
        let image = ModularImage::from_gray8(&[100u8; 64], 8, 8).unwrap();
        let mut samples = TreeSamples::new();
        gather_samples(&mut samples, &image, 0);

        let params = TreeLearningParams::for_effort(9);
        let tree = compute_best_tree(&mut samples, &params);
        // Should have at least 1 node (the root leaf)
        assert!(!tree.is_empty());
        // Root should be a leaf
        assert_eq!(tree[0].property, -1);
    }

    /// Layer-2 invariant for the parallel-tree-learning feature (issue #41 follow-on).
    ///
    /// Proof: the parallel path produces a tree that serializes to byte-identical
    /// tokens as the sequential path. Topology is data-determined (same samples →
    /// same splits, same predictors, same splitvals). Internal node-vec indexing
    /// is invisible to the bitstream because `collect_tree_tokens` traverses BFS
    /// from root via child pointers (not by visiting `tree[i]` in index order).
    ///
    /// Tests with a 2-channel image large enough to trigger the parallel threshold
    /// (n >= 8192 unique samples after dedup).
    #[cfg(feature = "parallel-tree-learning")]
    #[test]
    fn test_parallel_tree_matches_sequential() {
        use crate::modular::tree::collect_tree_tokens;

        // Build a non-trivial 2-channel image: 128x128, ch0=gradient, ch1=noise.
        // 128x128 = 16,384 pixels per channel × 2 = 32,768 samples — above the
        // parallel threshold of 8192.
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut ch0 = Channel::new(128, 128).unwrap();
        for y in 0..128 {
            for x in 0..128 {
                ch0.set(x, y, ((x * 3 + y * 7) & 0xFF) as i32);
            }
        }
        image.channels.push(ch0);
        let mut ch1 = Channel::new(128, 128).unwrap();
        for y in 0u32..128 {
            for x in 0u32..128 {
                // Pseudo-random pattern (deterministic).
                let v = (x.wrapping_mul(0x9e37) ^ y.wrapping_mul(0x7f4a)) & 0xFF;
                ch1.set(x as usize, y as usize, v as i32);
            }
        }
        image.channels.push(ch1);

        let params = TreeLearningParams::for_effort(7);

        // Build sequential tree by disabling the parallel path. We do this by
        // gathering twice (gather is deterministic) and calling
        // compute_best_tree — the parallel path activates only when the feature
        // is on AND n >= threshold. To get a true sequential baseline with the
        // feature on, we run with n < threshold by gathering a tiny image first
        // — but that wouldn't exercise the parallel path. Instead, we build
        // BOTH trees with the same samples but use the build_subtree_sequential
        // helper directly for the "sequential reference".
        //
        // The simpler proof: build via the public compute_best_tree with the
        // feature enabled (which uses the parallel path for n >= threshold),
        // AND build via build_subtree_sequential directly on the SAME pre-dedup
        // samples + pq. The trees should serialize to identical tokens.
        let mut samples_par = TreeSamples::new();
        gather_samples(&mut samples_par, &image, 0);
        let par_tree = compute_best_tree(&mut samples_par, &params);

        // Build the reference tree via the sequential path. We need to flip the
        // parallel feature off at runtime, which we can't do via cfg. So we
        // emulate the sequential path by running the same logic with a
        // sub-threshold sample count check disabled — call build_subtree_sequential
        // directly after pre-quantize + dedup + root-predictor computation.
        let mut samples_seq = TreeSamples::new();
        gather_samples(&mut samples_seq, &image, 0);
        let mut pq_seq = samples_seq.pre_quantize(&params);
        dedup_samples(&mut samples_seq, &mut pq_seq, &params);
        // Match compute_best_tree_with_budget's threshold computation: it uses
        // params.pixel_fraction (not derived from sample counts). The default
        // for TreeLearningParams::for_effort is 1.0.
        let required_cost = params.pixel_fraction * 0.9 + 0.1;
        let threshold = params.split_threshold * required_cost;
        let n = samples_seq.num_samples;
        let max_token = samples_seq
            .residual_tokens
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let histogram_size = max_token + 1;
        let mut entropy_counts = vec![0u32; histogram_size];
        let root_pred =
            find_best_predictor(&samples_seq, 0, n, histogram_size, &mut entropy_counts);
        let root_bits = compute_predictor_entropy(
            &samples_seq,
            0,
            n,
            root_pred,
            histogram_size,
            &mut entropy_counts,
        );
        let seq_tree = build_subtree_sequential(
            &mut samples_seq,
            &mut pq_seq,
            &params,
            threshold,
            params.max_nodes,
            histogram_size,
            root_pred,
            root_bits,
        );

        // The trees must serialize to identical token streams. Compare token
        // values (context, value, is_signed). Topology + node contents are
        // proven equal if every emitted token matches.
        let par_tokens = collect_tree_tokens(&par_tree);
        let seq_tokens = collect_tree_tokens(&seq_tree);

        // Sanity: both paths produced a non-trivial tree (more than just a root leaf).
        assert!(
            par_tree.len() >= 3,
            "parallel tree must split at least once"
        );
        assert!(
            seq_tree.len() >= 3,
            "sequential tree must split at least once"
        );

        assert_eq!(
            par_tokens.len(),
            seq_tokens.len(),
            "tree token count differs: parallel={} sequential={}",
            par_tokens.len(),
            seq_tokens.len(),
        );
        for (i, (p, s)) in par_tokens.iter().zip(seq_tokens.iter()).enumerate() {
            assert_eq!(
                (p.context, p.value, p.is_signed),
                (s.context, s.value, s.is_signed),
                "token #{i} differs: parallel=({},{},{}) sequential=({},{},{})",
                p.context,
                p.value,
                p.is_signed,
                s.context,
                s.value,
                s.is_signed,
            );
        }
    }

    /// Layer-2 invariant for the thread-local SplitWorkspace cache.
    ///
    /// Proves that a full `compute_best_tree` call on a large input
    /// (≥ parallel-tree-learning threshold) triggers at most ONE
    /// `SplitWorkspace::new` allocation on the main thread (regardless of
    /// how many `find_best_split` calls happen during the tree build).
    /// Before this fix, each `find_best_split` call allocated a fresh
    /// ~12 MB workspace; with the thread-local cache, only the first
    /// call on each thread allocates.
    ///
    /// The parallel path may add up to `num_rayon_workers` additional
    /// allocations (one per fresh worker thread that participates), so
    /// we assert an upper bound rather than equality.
    ///
    /// **Race-free measurement (issue #51)**: this test runs inside a
    /// private `rayon::ThreadPool` whose workers are marked with the
    /// thread-local [`IS_TEST_POOL_THREAD`] flag (via `start_handler`).
    /// `SplitWorkspace::new` bumps the dedicated
    /// [`SPLIT_WS_ALLOC_COUNT_TEST_POOL`] counter ONLY on threads where
    /// that flag is set. This guarantees the measurement is immune to
    /// allocations from any other test in the same test binary that
    /// happens to call `compute_best_tree` concurrently — they run on
    /// the global rayon pool or unmarked threads, neither of which bump
    /// the test-pool counter.
    #[test]
    fn test_thread_local_workspace_caps_allocations() {
        use core::sync::atomic::Ordering;

        // Build a 128×128 2-channel image — yields > 8192 unique samples after
        // dedup, large enough to trigger many `find_best_split` calls.
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut ch0 = Channel::new(128, 128).unwrap();
        for y in 0..128 {
            for x in 0..128 {
                ch0.set(x, y, ((x * 3 + y * 7) & 0xFF) as i32);
            }
        }
        image.channels.push(ch0);
        let mut ch1 = Channel::new(128, 128).unwrap();
        for y in 0u32..128 {
            for x in 0u32..128 {
                let v = (x.wrapping_mul(0x9e37) ^ y.wrapping_mul(0x7f4a)) & 0xFF;
                ch1.set(x as usize, y as usize, v as i32);
            }
        }
        image.channels.push(ch1);

        let params = TreeLearningParams::for_effort(7);

        // Use a private rayon pool with a fixed worker count we control. Its
        // workers (and only its workers) set IS_TEST_POOL_THREAD = true so
        // SplitWorkspace::new can attribute allocations to this test alone.
        // Without a private pool we'd inherit the global rayon pool, whose
        // workers may be shared with concurrently-running tests in the same
        // test binary — that race is exactly what made this test flaky
        // (issue #51).
        #[cfg(feature = "parallel-tree-learning")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("flaky-test-51-worker-{i}"))
            .start_handler(|_idx| {
                IS_TEST_POOL_THREAD.with(|f| f.set(true));
            })
            .build()
            .expect("build private rayon pool");

        // Run everything inside `pool.install` so any parallel fan-out from
        // compute_best_tree uses our marked workers. The calling thread itself
        // also runs portions of the work; flag it for the measurement window.
        IS_TEST_POOL_THREAD.with(|f| f.set(true));

        // Issue #42 (2026-05-25): inlined the previous `run(&mut dyn FnMut())`
        // helper because `pool.install` requires `FnOnce + Send`, which a
        // `&mut dyn FnMut()` trait object cannot satisfy. The bodies below are
        // equivalent to the previous calls — they run inside `pool.install` so
        // the rayon fan-out uses our private pool's workers.

        // First call: warm any state the test runtime might have lazily
        // initialised, AND warm the calling thread + private-pool workers'
        // caches so the second measurement is stable.
        #[cfg(feature = "parallel-tree-learning")]
        pool.install(|| {
            let mut samples_warm = TreeSamples::new();
            gather_samples(&mut samples_warm, &image, 0);
            let _ = compute_best_tree(&mut samples_warm, &params);
        });
        #[cfg(not(feature = "parallel-tree-learning"))]
        {
            let mut samples_warm = TreeSamples::new();
            gather_samples(&mut samples_warm, &image, 0);
            let _ = compute_best_tree(&mut samples_warm, &params);
        }

        // Snapshot the test-pool-only counter, then run a real encode and
        // measure how many NEW workspace allocations happened on threads we
        // own. Allocations from any other concurrent test go to the global
        // counter only, not this one.
        let before = SPLIT_WS_ALLOC_COUNT_TEST_POOL.load(Ordering::Relaxed);
        let tree_len: usize;
        #[cfg(feature = "parallel-tree-learning")]
        {
            tree_len = pool.install(|| {
                let mut samples = TreeSamples::new();
                gather_samples(&mut samples, &image, 0);
                let tree = compute_best_tree(&mut samples, &params);
                tree.len()
            });
        }
        #[cfg(not(feature = "parallel-tree-learning"))]
        {
            let mut samples = TreeSamples::new();
            gather_samples(&mut samples, &image, 0);
            let tree = compute_best_tree(&mut samples, &params);
            tree_len = tree.len();
        }
        let after = SPLIT_WS_ALLOC_COUNT_TEST_POOL.load(Ordering::Relaxed);
        let added = after - before;

        // Clear the flag on the calling thread — important if cargo runs more
        // tests on this same OS thread later.
        IS_TEST_POOL_THREAD.with(|f| f.set(false));

        // Sanity: this WAS a real tree-build (multiple splits).
        assert!(
            tree_len >= 3,
            "expected non-trivial tree, got {tree_len} nodes",
        );

        // With the thread-local cache, every thread in our private pool is
        // already warm from the first `run(...)` above, so the second call
        // should allocate 0 workspaces. We still allow `num_threads + 1` to
        // tolerate (a) a worker that didn't participate in the warm-up but
        // participated in the measurement, and (b) the calling thread.
        //
        // The old code (per-fork `SplitWorkspace::new`) would have allocated
        // up to `2^max_parallel_depth = 16` workspaces in the recursive path
        // PLUS one for the outer loop PLUS one for the seed find_best_split
        // — so the test would have caught a regression with > 16 every time.
        let cap = {
            #[cfg(feature = "parallel-tree-learning")]
            {
                // Private pool worker count + the calling thread.
                pool.current_num_threads() + 1
            }
            #[cfg(not(feature = "parallel-tree-learning"))]
            {
                1
            }
        };
        assert!(
            added <= cap,
            "thread-local workspace cache leaked: {added} new SplitWorkspace::new \
             calls on test-pool threads (cap = {cap}). With the cache, only the \
             first call on each worker thread should allocate.",
        );
    }

    /// Layer-1 invariant for the small-image parallel-tree-learning
    /// fallback (audit conditional-value catalog item #10).
    ///
    /// The fallback bypasses the thread-local SplitWorkspace cache on
    /// small images. This MUST be bitstream-equivalent: tree topology
    /// depends only on the samples, not on the workspace identity or
    /// per-call vs cached allocation. This test builds a tree twice
    /// for the same input — once with the fallback ON, once OFF — and
    /// asserts every emitted tree token is identical.
    ///
    /// Uses 32x32 to stay below the parallel-tree root threshold so
    /// the test does not race with `test_thread_local_workspace_caps_allocations`'s
    /// global allocation counter (the fallback path's per-call
    /// `SplitWorkspace::new` would inflate the cap test's `after -
    /// before` snapshot if both tests ran on overlapping rayon workers).
    /// 32×32 = 1024 samples per channel × 2 channels = 2048, below
    /// the e7 `parallel_root_threshold = 8192`, so the sequential
    /// loop runs and the workspace count stays bounded.
    #[test]
    fn test_small_image_fallback_byte_equivalent() {
        use crate::modular::tree::collect_tree_tokens;
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut ch0 = Channel::new(32, 32).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                ch0.set(x, y, ((x * 3 + y * 7) & 0xFF) as i32);
            }
        }
        image.channels.push(ch0);
        let mut ch1 = Channel::new(32, 32).unwrap();
        for y in 0u32..32 {
            for x in 0u32..32 {
                let v = (x.wrapping_mul(0x9e37) ^ y.wrapping_mul(0x7f4a)) & 0xFF;
                ch1.set(x as usize, y as usize, v as i32);
            }
        }
        image.channels.push(ch1);

        let mut params_off = TreeLearningParams::for_effort(7);
        params_off.parallel_small_image_fallback = false;
        let mut samples_off = TreeSamples::new();
        gather_samples(&mut samples_off, &image, 0);
        let tree_off = compute_best_tree(&mut samples_off, &params_off);

        let mut params_on = TreeLearningParams::for_effort(7);
        params_on.parallel_small_image_fallback = true;
        let mut samples_on = TreeSamples::new();
        gather_samples(&mut samples_on, &image, 0);
        let tree_on = compute_best_tree(&mut samples_on, &params_on);

        let tokens_off = collect_tree_tokens(&tree_off);
        let tokens_on = collect_tree_tokens(&tree_on);

        assert_eq!(
            tokens_off.len(),
            tokens_on.len(),
            "tree token count differs between fallback OFF ({}) and ON ({})",
            tokens_off.len(),
            tokens_on.len(),
        );
        for (i, (off, on)) in tokens_off.iter().zip(tokens_on.iter()).enumerate() {
            assert_eq!(
                (off.context, off.value, off.is_signed),
                (on.context, on.value, on.is_signed),
                "token #{i} differs: fallback OFF=({},{},{}) vs ON=({},{},{})",
                off.context,
                off.value,
                off.is_signed,
                on.context,
                on.value,
                on.is_signed,
            );
        }
    }

    #[test]
    fn test_compute_best_tree_two_channels() {
        // 2-channel image: ch0=constant 100, ch1=gradient ramp
        // Tree should split on channel property
        // Use 32x32 to ensure enough samples for split evaluation
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };

        // Channel 0: constant
        let mut ch0 = Channel::new(32, 32).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                ch0.set(x, y, 100);
            }
        }
        image.channels.push(ch0);

        // Channel 1: ramp
        let mut ch1 = Channel::new(32, 32).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                ch1.set(x, y, (x * 7 + y * 5) as i32);
            }
        }
        image.channels.push(ch1);

        let mut samples = TreeSamples::new();
        gather_samples(&mut samples, &image, 0);

        let params = TreeLearningParams::for_effort(9);
        let tree = compute_best_tree(&mut samples, &params);

        // Count leaves
        let num_leaves = tree.iter().filter(|n| n.property < 0).count();
        // Should have multiple leaves (split on channel or spatial properties)
        assert!(num_leaves >= 2, "expected >= 2 leaves, got {}", num_leaves);
    }

    #[test]
    fn test_collect_residuals_with_tree() {
        // Simple single-leaf tree with gradient predictor
        let tree = vec![PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Gradient,
            context_id: 0,
            multiplier: 1,
            ..Default::default()
        }];

        let image = ModularImage::from_gray8(&[100u8; 16], 4, 4).unwrap();
        let tokens =
            collect_residuals_with_tree(&image, &tree, 0, &WeightedPredictorParams::default());

        assert_eq!(tokens.len(), 16);
        // All tokens should have context 0
        for t in &tokens {
            assert_eq!(t.context(), 0);
        }
    }

    #[test]
    fn test_traverse_with_spec_props() {
        // 3-node tree: split on channel (property 0) at splitval=0
        // lchild (channel <= 0) = Zero predictor
        // rchild (channel > 0) = Gradient predictor
        let tree = vec![
            PropertyDecisionNode {
                property: 0, // Channel
                splitval: 0,
                lchild: 1,
                rchild: 2,
                ..Default::default()
            },
            PropertyDecisionNode {
                property: -1,
                predictor: Predictor::Zero,
                context_id: 0,
                multiplier: 1,
                ..Default::default()
            },
            PropertyDecisionNode {
                property: -1,
                predictor: Predictor::Gradient,
                context_id: 1,
                multiplier: 1,
                ..Default::default()
            },
        ];

        // Channel 0 should hit lchild (Zero)
        let mut props = [0i32; NUM_PROPERTIES];
        props[0] = 0;
        let leaf = traverse_with_spec_props(&tree, &props);
        assert_eq!(leaf.predictor, Predictor::Zero);

        // Channel 1 should hit rchild (Gradient)
        props[0] = 1;
        let leaf = traverse_with_spec_props(&tree, &props);
        assert_eq!(leaf.predictor, Predictor::Gradient);
    }

    #[test]
    fn split_workspace_handles_boundary_histogram_size() {
        // HISTO_PADDED = 256 covers the max token (239) the GATHER_HYBRID_UINT
        // config can produce; this confirms the workspace constructs at the cap.
        let _ws = SplitWorkspace::new(8, HISTO_PADDED, 2);
        let _ws = SplitWorkspace::new(8, HISTO_PADDED - 1, 2);
        let _ws = SplitWorkspace::new(8, 1, 2);
    }

    #[test]
    fn gather_hybrid_uint_token_bound() {
        // Proof companion to HISTO_PADDED's bound derivation: any u32 input
        // produces a token <= 239, which fits in a u8 without saturation.
        let probes: [u32; 8] = [
            0,
            15,
            16,
            (1u32 << 17) - 2, // max packed for ±65535 (16-bit pixel residual)
            (1u32 << 20),
            (1u32 << 25),
            (1u32 << 30),
            u32::MAX,
        ];
        for v in probes {
            let (token, _, _) = GATHER_HYBRID_UINT.encode(v);
            assert!(token <= 239, "token {token} exceeded 239 for input {v}");
        }
    }

    #[test]
    fn test_partition_node_in_place() {
        // 4x4 image: gather all 16 pixels, partition on X (property 3) at
        // splitval=1. Pixels with x<=1 should land in [0..8), x>1 in [8..16).
        // Verifies the chunk-2 in-place SoA permutation (issue #40).
        let image = ModularImage::from_gray8(&[0u8; 16], 4, 4).unwrap();
        let mut samples = TreeSamples::new();
        gather_samples(&mut samples, &image, 0);

        // Build a params struct that includes property 3 (X coord) so
        // pre_quantize populates its bucket_indices. Then dedup, which
        // populates `sample_counts` (required by `swap_rows`).
        let params = TreeLearningParams::for_effort(7);
        let mut pq = samples.pre_quantize(&params);
        dedup_samples(&mut samples, &mut pq, &params);

        let n = samples.num_samples;
        // After dedup on a constant-zero image, all 16 pixels merge to a few
        // unique sample groups — count what we actually have.
        assert!(n > 0, "expected at least one unique sample");

        // Count left side using PartitionKey::Property semantics
        let left_count = samples.props[3][..n].iter().filter(|&&v| v <= 1).count();

        let mid = partition_node_in_place(
            &mut samples,
            &mut pq,
            0,
            n,
            left_count,
            tree_learn_split::PartitionKey::Property {
                prop_idx: 3,
                val: 1,
            },
        );
        assert_eq!(mid, left_count);

        // Left side: x <= 1
        for v in &samples.props[3][..mid] {
            assert!(*v <= 1, "left-side row should have x<=1 but got {v}");
        }
        // Right side: x > 1
        for v in &samples.props[3][mid..n] {
            assert!(*v > 1, "right-side row should have x>1 but got {v}");
        }

        // Re-verify SoA row alignment: every parallel array must hold values
        // consistent with the post-partition row layout. The strictest check
        // is that the sum of sample_counts equals the original sample count
        // (16 pixels in this 4x4 image) and that each predictor's
        // residual_tokens has length matching num_samples.
        let total_count: u32 = samples.sample_counts[..n].iter().sum();
        assert_eq!(total_count, 16, "permutation must preserve total weight");
        for pred in 0..samples.num_predictors() {
            assert_eq!(samples.residual_tokens[pred].len(), n);
            assert_eq!(samples.extra_bits[pred].len(), n);
        }
    }

    /// Invariant test (issue #41): both dedup backends must produce the same
    /// unique-sample set with the same multiplicities.
    ///
    /// Row order differs (sort path = composite-key-sorted, streaming path =
    /// first-seen), so we compare *multisets* of canonical packed keys
    /// weighted by `sample_counts`. The tree learner is order-invariant in
    /// the sample axis (it groups by property bucket, not row position) — so
    /// this multiset equality is necessary and sufficient for bitstream
    /// identity, which `hash_lock_features` separately confirms for the
    /// default (sort) backend.
    #[test]
    fn test_dedup_backends_agree_on_unique_set() {
        let mut pixels = [0u8; 16 * 16 * 3];
        // Non-trivial pattern with real duplicates so both paths exercise
        // their dedup logic. ~30-40 % unique colors on this 16×16.
        for y in 0..16u8 {
            for x in 0..16u8 {
                let base = (y * 16 + x) as usize * 3;
                pixels[base] = (x & 0b1100) << 2;
                pixels[base + 1] = (y & 0b1100) << 2;
                pixels[base + 2] = ((x ^ y) & 0b0111) << 4;
            }
        }
        let image = ModularImage::from_rgb8(&pixels, 16, 16).unwrap();

        let collect = |params: &TreeLearningParams| -> std::collections::BTreeMap<Vec<u8>, u32> {
            let mut samples = TreeSamples::new();
            gather_samples(&mut samples, &image, 0);
            let mut pq = samples.pre_quantize(params);
            dedup_samples(&mut samples, &mut pq, params);

            let n = samples.num_samples;
            let num_pred = samples.num_predictors();
            let mut multiset: std::collections::BTreeMap<Vec<u8>, u32> =
                std::collections::BTreeMap::new();
            for i in 0..n {
                let mut key = Vec::with_capacity(params.properties.len() + 2 * num_pred);
                for &prop_idx in &params.properties {
                    let bi = &pq.bucket_indices[prop_idx];
                    key.push(if bi.is_empty() { 0 } else { bi[i] });
                }
                for pred in 0..num_pred {
                    key.push(samples.residual_tokens[pred][i]);
                    key.push(samples.extra_bits[pred][i]);
                }
                *multiset.entry(key).or_insert(0) += samples.sample_counts[i];
            }
            multiset
        };

        let mut params_sort = TreeLearningParams::for_effort(7);
        params_sort.use_streaming_dedup = false;
        let multiset_sort = collect(&params_sort);

        let mut params_stream = TreeLearningParams::for_effort(7);
        params_stream.use_streaming_dedup = true;
        let multiset_stream = collect(&params_stream);

        assert_eq!(
            multiset_sort, multiset_stream,
            "packed-sort and streaming dedup must agree on the unique-sample multiset",
        );
        // 16×16 RGB = 256 pixels × 3 channels = 768 gathered samples,
        // regardless of dedup unique count.
        let total_sort: u32 = multiset_sort.values().sum();
        let total_stream: u32 = multiset_stream.values().sum();
        assert_eq!(total_sort, 768, "sort dedup must conserve total weight");
        assert_eq!(total_stream, 768, "stream dedup must conserve total weight");
    }

    /// Phase 2 of issue #41: gather-time dedup conserves the total
    /// gathered-weight invariant. Sum of `sample_counts` after a
    /// gather-with-dedup pass must equal the count of pixels actually
    /// gathered (channels × width × height when stride = 1), regardless
    /// of how many merges happen inside the cuckoo table.
    #[test]
    fn test_gather_dedup_conserves_total_weight() {
        // Use a constant 32x32 RGB image — every pixel has identical
        // (token, ebits, props), so the gather-time merge collapses
        // them all into 3 unique rows (one per channel). This is the
        // tightest possible invariant exercise.
        let pixels = vec![128u8; 32 * 32 * 3];
        let image = ModularImage::from_rgb8(&pixels, 32, 32).unwrap();

        // Baseline: no gather-time dedup.
        let mut baseline = TreeSamples::new();
        gather_samples(&mut baseline, &image, 0);
        let baseline_total = baseline.num_samples as u32;
        assert_eq!(
            baseline_total,
            32 * 32 * 3,
            "constant 32x32 RGB sequential gather expects 3072"
        );

        // With gather-time dedup using the e7 property set
        // (production callers thread `params.properties` here).
        let params = TreeLearningParams::for_effort(7);
        let mut deduped = TreeSamples::new();
        gather_samples_strided_with_dedup(
            &mut deduped,
            &image,
            0,
            0,
            1,
            &WeightedPredictorParams::default(),
            None,
            true,
            &params.properties,
        )
        .unwrap();

        assert_eq!(
            deduped.sample_counts.len(),
            deduped.num_samples,
            "gather-time dedup must keep sample_counts in lockstep with num_samples",
        );
        let dedup_total: u32 = deduped.sample_counts.iter().sum();
        assert_eq!(
            dedup_total, baseline_total,
            "gather-time dedup must conserve gathered-weight total",
        );
        // A constant image collapses to a tiny number of unique rows
        // (one per channel × neighbour-class) — definitely many merges.
        assert!(
            deduped.num_samples < baseline_total as usize / 4,
            "expected aggressive merging on a constant image; got num_samples={} vs total {}",
            deduped.num_samples,
            baseline_total,
        );
    }

    /// End-to-end: with `gather_dedup` on, the unique-set multiset still
    /// agrees with the sort-only path AFTER the post-gather sort pass.
    /// This is the byte-equivalent invariant: the bucket-equivalence
    /// dedup is the final arbiter, and gather-time dedup is a (lossless)
    /// strict subset that the final sort pass collapses correctly.
    #[test]
    fn test_gather_dedup_then_sort_matches_sort_only() {
        let mut pixels = [0u8; 16 * 16 * 3];
        for y in 0..16u8 {
            for x in 0..16u8 {
                let base = (y * 16 + x) as usize * 3;
                pixels[base] = (x & 0b1100) << 2;
                pixels[base + 1] = (y & 0b1100) << 2;
                pixels[base + 2] = ((x ^ y) & 0b0111) << 4;
            }
        }
        let image = ModularImage::from_rgb8(&pixels, 16, 16).unwrap();

        let collect_multiset = |samples: &TreeSamples,
                                pq: &PreQuantizedProps,
                                params: &TreeLearningParams|
         -> std::collections::BTreeMap<Vec<u8>, u32> {
            let n = samples.num_samples;
            let num_pred = samples.num_predictors();
            let mut multiset: std::collections::BTreeMap<Vec<u8>, u32> =
                std::collections::BTreeMap::new();
            for i in 0..n {
                let mut key = Vec::with_capacity(params.properties.len() + 2 * num_pred);
                for &prop_idx in &params.properties {
                    let bi = &pq.bucket_indices[prop_idx];
                    key.push(if bi.is_empty() { 0 } else { bi[i] });
                }
                for pred in 0..num_pred {
                    key.push(samples.residual_tokens[pred][i]);
                    key.push(samples.extra_bits[pred][i]);
                }
                *multiset.entry(key).or_insert(0) += samples.sample_counts[i];
            }
            multiset
        };

        // Path A: sort-only (existing default).
        let params_sort = TreeLearningParams::for_effort(7);
        let mut samples_a = TreeSamples::new();
        gather_samples(&mut samples_a, &image, 0);
        let mut pq_a = samples_a.pre_quantize(&params_sort);
        dedup_samples(&mut samples_a, &mut pq_a, &params_sort);
        let multiset_a = collect_multiset(&samples_a, &pq_a, &params_sort);

        // Path B: gather-time dedup + post-gather sort.
        let mut params_b = TreeLearningParams::for_effort(7);
        params_b.gather_dedup = true;
        let mut samples_b = TreeSamples::new();
        gather_samples_strided_with_dedup(
            &mut samples_b,
            &image,
            0,
            0,
            1,
            &WeightedPredictorParams::default(),
            None,
            true,
            &params_b.properties,
        )
        .unwrap();
        let mut pq_b = samples_b.pre_quantize(&params_b);
        dedup_samples(&mut samples_b, &mut pq_b, &params_b);
        let multiset_b = collect_multiset(&samples_b, &pq_b, &params_b);

        assert_eq!(
            multiset_a, multiset_b,
            "gather-time dedup followed by sort dedup must reproduce the sort-only unique multiset",
        );
        let total_a: u32 = multiset_a.values().sum();
        let total_b: u32 = multiset_b.values().sum();
        assert_eq!(total_a, total_b, "total weight must match between paths");
        assert_eq!(total_a, 768, "16x16 RGB single-stride gather expects 768");
    }

    /// Phase 3 of issue #41: dispatching through
    /// [`gather_samples_strided_with_dedup_backend`] with
    /// `enable_phase3 = true` must conserve the total gathered weight
    /// just like Phase 2 does, and produce a unique-sample multiset that
    /// — after the post-`pre_quantize` sort dedup — agrees with the
    /// Phase 2 backend on the bucket-equivalence partition.
    ///
    /// This is the Layer 1 invariant test for Chunk 2: it doesn't measure
    /// performance, just bitstream-determining correctness of the new
    /// dispatch path. The end-to-end real-photo bench owns the perf side.
    #[test]
    fn test_gather_dedup_phase3_dispatch_conserves_weight() {
        let mut pixels = [0u8; 16 * 16 * 3];
        for y in 0..16u8 {
            for x in 0..16u8 {
                let base = (y * 16 + x) as usize * 3;
                // Same low-entropy pattern used by
                // test_gather_dedup_then_sort_matches_sort_only — produces
                // a healthy mix of duplicate and unique samples that
                // exercises both the Phase 3 fingerprint hit and miss
                // paths.
                pixels[base] = (x & 0b1100) << 2;
                pixels[base + 1] = (y & 0b1100) << 2;
                pixels[base + 2] = ((x ^ y) & 0b0111) << 4;
            }
        }
        let image = ModularImage::from_rgb8(&pixels, 16, 16).unwrap();
        let params = TreeLearningParams::for_effort(7);

        // Path A: Phase 2 backend (existing gather-time dedup).
        let mut samples_p2 = TreeSamples::new();
        gather_samples_strided_with_dedup_backend(
            &mut samples_p2,
            &image,
            0,
            0,
            1,
            &WeightedPredictorParams::default(),
            None,
            true,
            false, // enable_phase3
            &params.properties,
        )
        .unwrap();
        let p2_total: u32 = samples_p2.sample_counts.iter().sum();
        assert_eq!(
            p2_total, 768,
            "Phase 2 dispatch must conserve total weight (16x16 RGB stride=1)"
        );
        assert_eq!(
            samples_p2.sample_counts.len(),
            samples_p2.num_samples,
            "Phase 2 sample_counts must stay in lockstep with num_samples",
        );

        // Path B: Phase 3 backend (new InlineDedupTable dispatch).
        let mut samples_p3 = TreeSamples::new();
        gather_samples_strided_with_dedup_backend(
            &mut samples_p3,
            &image,
            0,
            0,
            1,
            &WeightedPredictorParams::default(),
            None,
            true,
            true, // enable_phase3
            &params.properties,
        )
        .unwrap();
        let p3_total: u32 = samples_p3.sample_counts.iter().sum();
        assert_eq!(
            p3_total, 768,
            "Phase 3 dispatch must conserve total weight (16x16 RGB stride=1)"
        );
        assert_eq!(
            samples_p3.sample_counts.len(),
            samples_p3.num_samples,
            "Phase 3 sample_counts must stay in lockstep with num_samples",
        );

        // Phase 3 should also collapse some duplicate rows — same
        // smoke-test as the conservation test for Phase 2.
        assert!(
            samples_p3.num_samples < 768,
            "Phase 3 expected to collapse at least some duplicates on a low-entropy pattern (got num_samples={})",
            samples_p3.num_samples,
        );

        // Post-`pre_quantize` sort dedup arbitration: both backends must
        // produce the same bucket-equivalence multiset. This is the
        // invariant that lets hash-locks stay byte-identical.
        let mut params_b2 = TreeLearningParams::for_effort(7);
        params_b2.gather_dedup = true;
        let mut pq_p2 = samples_p2.pre_quantize(&params_b2);
        dedup_samples(&mut samples_p2, &mut pq_p2, &params_b2);
        let mut pq_p3 = samples_p3.pre_quantize(&params_b2);
        dedup_samples(&mut samples_p3, &mut pq_p3, &params_b2);

        let collect_multiset = |samples: &TreeSamples,
                                pq: &PreQuantizedProps,
                                params: &TreeLearningParams|
         -> std::collections::BTreeMap<Vec<u8>, u32> {
            let n = samples.num_samples;
            let num_pred = samples.num_predictors();
            let mut multiset: std::collections::BTreeMap<Vec<u8>, u32> =
                std::collections::BTreeMap::new();
            for i in 0..n {
                let mut key = Vec::with_capacity(params.properties.len() + 2 * num_pred);
                for &prop_idx in &params.properties {
                    let bi = &pq.bucket_indices[prop_idx];
                    key.push(if bi.is_empty() { 0 } else { bi[i] });
                }
                for pred in 0..num_pred {
                    key.push(samples.residual_tokens[pred][i]);
                    key.push(samples.extra_bits[pred][i]);
                }
                *multiset.entry(key).or_insert(0) += samples.sample_counts[i];
            }
            multiset
        };

        let multiset_p2 = collect_multiset(&samples_p2, &pq_p2, &params_b2);
        let multiset_p3 = collect_multiset(&samples_p3, &pq_p3, &params_b2);
        assert_eq!(
            multiset_p2, multiset_p3,
            "Phase 2 and Phase 3 backends must produce the same post-sort bucket-equivalence multiset",
        );
    }

    /// `phase3_packing_fits` is the construction-time precondition test
    /// the dispatcher uses to decide whether [`crate::modular::inline_dedup_table::InlineDedupTable`]
    /// can be activated for the configured (num_pred, num_props) combination.
    /// Pinned cases to keep the boundary explicit.
    #[test]
    fn test_phase3_packing_fits_pinned_cases() {
        use crate::modular::inline_dedup_table::KEY_BYTES;
        assert_eq!(KEY_BYTES, 64, "test pinned to KEY_BYTES = 64");

        // e7 RGB: 14 candidate predictors × 2 = 28 bytes, 9 props × 4 = 36 bytes.
        // Total 64 bytes — exact fit.
        assert!(
            phase3_packing_fits(14, 9),
            "e7 RGB (14 pred, 9 props) must fit"
        );

        // e9 RGB: 14 × 2 + 24 × 4 = 28 + 96 = 124. Overflow.
        assert!(
            !phase3_packing_fits(14, 24),
            "e9 RGB (14 pred, 24 props) must overflow → Phase 2 fallback"
        );

        // Edge cases.
        assert!(
            phase3_packing_fits(0, 16),
            "0 pred + 16 props × 4 = 64 bytes exactly"
        );
        assert!(
            !phase3_packing_fits(0, 17),
            "0 pred + 17 props × 4 = 68 bytes overflow"
        );
        assert!(phase3_packing_fits(32, 0), "32 pred × 2 = 64 bytes exactly");
        assert!(
            !phase3_packing_fits(33, 0),
            "33 pred × 2 = 66 bytes overflow"
        );
        assert!(
            phase3_packing_fits(0, 0),
            "empty configuration trivially fits"
        );
    }

    // ── RFC#45 chunk 2 — multi-seed scaffolding tests ──

    #[test]
    fn test_estimate_token_cost_empty_is_zero() {
        assert_eq!(estimate_token_cost(&[]), 0.0);
    }

    #[test]
    fn test_estimate_token_cost_grows_with_context_count() {
        use crate::entropy_coding::token::Token;
        // Two streams: same N tokens, same single-symbol value, but one uses
        // a single context, the other uses 4 different contexts. The
        // per-context header term must make the multi-context stream cost
        // strictly more.
        let single_ctx: Vec<Token> = (0..40).map(|_| Token::new(0, 5)).collect();
        let multi_ctx: Vec<Token> = (0..40).map(|i| Token::new(i as u32 / 10, 5)).collect();
        let c_single = estimate_token_cost(&single_ctx);
        let c_multi = estimate_token_cost(&multi_ctx);
        assert!(
            c_multi > c_single,
            "multi-context cost {c_multi} must exceed single-context cost {c_single} \
             via the per-context header term",
        );
    }

    #[test]
    fn test_gather_with_offset_zero_matches_legacy() {
        // start_offset = 0 must produce byte-identical TreeSamples to the
        // legacy `gather_samples_strided`. Regression guard for the seeds=1
        // path that all e ≤ 9 callers hit.
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut ch = Channel::new(32, 32).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                ch.set(x, y, ((x * 7 + y * 11) & 0xFF) as i32);
            }
        }
        image.channels.push(ch);

        let wp = WeightedPredictorParams::default();
        let mut legacy = TreeSamples::new();
        gather_samples_strided(&mut legacy, &image, 0, 0, 3, &wp);

        let mut offset0 = TreeSamples::new();
        gather_samples_strided_with_offset(&mut offset0, &image, 0, 0, 3, 0, &wp);

        assert_eq!(legacy.num_samples, offset0.num_samples);
        assert_eq!(legacy.residual_tokens, offset0.residual_tokens);
        assert_eq!(legacy.extra_bits, offset0.extra_bits);
        assert_eq!(legacy.props, offset0.props);
    }

    #[test]
    fn test_gather_with_offset_nonzero_differs() {
        // start_offset > 0 must yield a different sample set than offset 0
        // for a stride > 1. This is the seed-variance guarantee.
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut ch = Channel::new(64, 64).unwrap();
        for y in 0..64 {
            for x in 0..64 {
                ch.set(
                    x,
                    y,
                    ((x.wrapping_mul(13) ^ y.wrapping_mul(17)) & 0xFF) as i32,
                );
            }
        }
        image.channels.push(ch);

        let wp = WeightedPredictorParams::default();
        let mut s0 = TreeSamples::new();
        gather_samples_strided_with_offset(&mut s0, &image, 0, 0, 4, 0, &wp);
        let mut s1 = TreeSamples::new();
        gather_samples_strided_with_offset(&mut s1, &image, 0, 0, 4, 1, &wp);

        // Same count (both gather every 4th candidate), but different rows.
        assert_eq!(s0.num_samples, s1.num_samples);
        assert!(
            s0.residual_tokens != s1.residual_tokens || s0.props != s1.props,
            "offset 1 must select a different sample subset than offset 0",
        );
    }

    // ── RFC#45 chunk 3 — broader seed variance tests ──

    #[test]
    fn test_derive_seeded_params_seed_zero_is_clone() {
        // Seed 0 must be a faithful no-op clone (preserves the chunk-2
        // invariant that hash-locks at e ≤ 9 stay byte-identical).
        let base = TreeLearningParams::for_effort(9);
        let s0 = derive_seeded_params(&base, 0);
        assert_eq!(s0.properties, base.properties);
        assert_eq!(s0.split_threshold, base.split_threshold);
        assert_eq!(s0.max_property_values, base.max_property_values);
        assert_eq!(s0.max_nodes, base.max_nodes);
    }

    #[test]
    fn test_derive_seeded_params_nonzero_perturbs_threshold() {
        // Seeds 1, 2, 3 must each yield a split_threshold different from
        // base. The four multipliers [1.0, 0.7, 1.3, 0.85] guarantee
        // distinct values for seeds 1..=3.
        let base = TreeLearningParams::for_effort(9);
        let s1 = derive_seeded_params(&base, 1);
        let s2 = derive_seeded_params(&base, 2);
        let s3 = derive_seeded_params(&base, 3);
        assert!(
            (s1.split_threshold - base.split_threshold).abs() > 1e-6,
            "seed 1 split_threshold must differ from base",
        );
        assert!(
            (s2.split_threshold - base.split_threshold).abs() > 1e-6,
            "seed 2 split_threshold must differ from base",
        );
        assert!(
            (s3.split_threshold - base.split_threshold).abs() > 1e-6,
            "seed 3 split_threshold must differ from base",
        );
        // All three must be distinct from each other.
        assert!((s1.split_threshold - s2.split_threshold).abs() > 1e-6);
        assert!((s2.split_threshold - s3.split_threshold).abs() > 1e-6);
        assert!((s1.split_threshold - s3.split_threshold).abs() > 1e-6);
    }

    #[test]
    fn test_derive_seeded_params_preserves_structural_prefix() {
        // The property-order rotation must NOT disturb the structural
        // prefix (Channel at index 0, GroupId at index 1 when present).
        let base = TreeLearningParams::for_effort(9);
        // for_effort(9) uses PROP_ORDER_NO_SQUEEZE which has Channel + GroupId
        // as the first two entries.
        assert_eq!(base.properties[0], 0, "base must start with Channel");
        assert_eq!(base.properties[1], 1, "base must have GroupId at index 1");

        for seed in 1..=8 {
            let p = derive_seeded_params(&base, seed);
            assert_eq!(
                p.properties[0], 0,
                "seed {seed}: Channel must remain at index 0",
            );
            assert_eq!(
                p.properties[1], 1,
                "seed {seed}: GroupId must remain at index 1",
            );
            assert_eq!(
                p.properties.len(),
                base.properties.len(),
                "seed {seed}: rotation must preserve length",
            );
        }
    }

    #[test]
    fn test_derive_seeded_params_property_order_varies_across_seeds() {
        // At least one of seeds 1..=4 must produce a property order
        // different from base (since `rot = (seed * 3) % tail_len` is
        // non-zero for some seed in 1..=4).
        let base = TreeLearningParams::for_effort(9);
        let any_changed =
            (1u64..=4).any(|seed| derive_seeded_params(&base, seed).properties != base.properties);
        assert!(
            any_changed,
            "at least one seed in 1..=4 must perturb property order",
        );
    }

    #[test]
    fn test_derive_seeded_stride_seed_zero_returns_base() {
        // Seed 0 must always return base_stride (preserves chunk-2
        // byte-identicality on the canonical first seed).
        assert_eq!(derive_seeded_stride(1, 0), 1);
        assert_eq!(derive_seeded_stride(3, 0), 3);
        assert_eq!(derive_seeded_stride(7, 0), 7);
        // base_stride == 0 is clamped to >= 1.
        assert_eq!(derive_seeded_stride(0, 0), 1);
    }

    #[test]
    fn test_derive_seeded_stride_nonzero_perturbs_density() {
        // Higher seeds must produce a different (or at least not
        // monotonically equal) stride than base when base > 1.
        let base = 5;
        let s0 = derive_seeded_stride(base, 0);
        let strides: Vec<usize> = (1u64..=4)
            .map(|seed| derive_seeded_stride(base, seed))
            .collect();
        assert_eq!(s0, base, "seed 0 must equal base stride");
        // At least two distinct values across seeds 1..=4.
        let distinct: std::collections::BTreeSet<usize> = strides.iter().copied().collect();
        assert!(
            distinct.len() >= 2,
            "seeds 1..=4 must cover at least 2 distinct strides (got {strides:?})",
        );
        // All strides must be >= 1 (clamp invariant).
        for s in &strides {
            assert!(*s >= 1, "stride must be >= 1");
        }
    }

    // ---- RFC#45 chunk 4 helpers (sample-fraction jitter + predictor order) ----
    //      Chunk 5 gates these to seeds 4..7 so chunk-3 perturbations
    //      (seeds 0..3) are not recombined inside a fixed budget.

    #[test]
    fn test_derive_seeded_sample_fraction_low_seeds_are_none() {
        // Chunk 5: seeds 0..=3 are reserved for chunk-3-only perturbations.
        // The sample-fraction helper MUST return None for all of them so
        // the canonical profile fraction (set by
        // EffortProfile::tree_sample_fraction_for) is used verbatim.
        for seed in 0u64..=3 {
            assert_eq!(
                derive_seeded_sample_fraction(seed),
                None,
                "chunk 5: seed {seed} must return None (chunk-3-only slot)",
            );
        }
    }

    #[test]
    fn test_derive_seeded_sample_fraction_high_seeds_active() {
        // Chunk 5: seeds 4..=7 carry the three sample-fraction overrides
        // plus one canonical-fraction slot (seed 7).
        assert_eq!(derive_seeded_sample_fraction(4), Some(0.40));
        assert_eq!(derive_seeded_sample_fraction(5), Some(0.60));
        assert_eq!(derive_seeded_sample_fraction(6), Some(0.70));
        assert_eq!(derive_seeded_sample_fraction(7), None);
        // Wrap-around: seed 8 cycles back to the seed-4 entry.
        assert_eq!(derive_seeded_sample_fraction(8), Some(0.40));
        // Distinct: 3 non-None values across seeds 4..=6.
        let vals: std::collections::BTreeSet<u32> = (4u64..=6)
            .map(|s| (derive_seeded_sample_fraction(s).unwrap() * 100.0) as u32)
            .collect();
        assert_eq!(
            vals.len(),
            3,
            "seeds 4..=6 must produce 3 distinct fractions"
        );
    }

    #[test]
    fn test_stride_for_seeded_sample_fraction_under_floor() {
        // total_pixels well under the 65_536 floor → stride must be 1.
        // (max(65_536, 0) = 65_536 ≥ 1_000 → stride = 1)
        assert_eq!(stride_for_seeded_sample_fraction(1_000, 0.40), 1);
        assert_eq!(stride_for_seeded_sample_fraction(1_000, 0.70), 1);
    }

    #[test]
    fn test_stride_for_seeded_sample_fraction_above_floor() {
        // total_pixels = 1_000_000, fraction = 0.40 → target = 400_000,
        // stride = ceil(1_000_000 / 400_000) = 3.
        assert_eq!(stride_for_seeded_sample_fraction(1_000_000, 0.40), 3);
        // fraction = 0.70 → target = 700_000, stride = 2.
        assert_eq!(stride_for_seeded_sample_fraction(1_000_000, 0.70), 2);
        // fraction = 0.50 → target = 500_000, stride = 2.
        assert_eq!(stride_for_seeded_sample_fraction(1_000_000, 0.50), 2);
    }

    #[test]
    fn test_derive_seeded_predictor_order_low_seeds_canonical() {
        // Chunk 5: seeds 0..=3 are reserved for chunk-3-only perturbations
        // and MUST receive the canonical predictor order (no chunk-4
        // interference). This preserves chunk-2/chunk-3 seed-0
        // byte-identicality AND keeps chunk-3's threshold/property/stride
        // perturbations from being recombined with predictor permutations.
        for seed in 0u64..=3 {
            let order = derive_seeded_predictor_order(seed);
            assert_eq!(order.len(), CANDIDATE_PREDICTORS.len());
            for (a, b) in order.iter().zip(CANDIDATE_PREDICTORS.iter()) {
                assert_eq!(
                    a, b,
                    "chunk 5: seed {seed} predictor order must equal CANDIDATE_PREDICTORS",
                );
            }
        }
    }

    #[test]
    fn test_derive_seeded_predictor_order_high_seeds_perturb() {
        // Chunk 5: seeds 5..=7 must each produce a permutation that
        // differs from the canonical order (seed 4 maps to perm[0], the
        // canonical, so we start the differs-check at seed 5).
        let canonical = derive_seeded_predictor_order(0);
        for seed in 5u64..=7 {
            let perm = derive_seeded_predictor_order(seed);
            assert_eq!(perm.len(), canonical.len());
            let differs = perm.iter().zip(canonical.iter()).any(|(a, b)| a != b);
            assert!(
                differs,
                "chunk 5: seed {seed} predictor order must differ from canonical",
            );
        }
        // Seed 4 IS the canonical (chunk-4 perm index 0).
        let perm4 = derive_seeded_predictor_order(4);
        for (a, b) in perm4.iter().zip(canonical.iter()) {
            assert_eq!(a, b, "chunk 5: seed 4 maps to perm[0] (canonical)");
        }
    }

    #[test]
    fn test_derive_seeded_predictor_order_preserves_predictor_set() {
        // Each permutation must contain exactly the same 14 predictors
        // (set equality) as CANDIDATE_PREDICTORS — only the order varies.
        let canonical_set: std::collections::BTreeSet<u32> =
            CANDIDATE_PREDICTORS.iter().map(|p| *p as u32).collect();
        for seed in 0u64..=7 {
            let perm = derive_seeded_predictor_order(seed);
            let perm_set: std::collections::BTreeSet<u32> =
                perm.iter().map(|p| *p as u32).collect();
            assert_eq!(
                perm_set, canonical_set,
                "seed {seed} permutation must contain the same predictor set",
            );
            assert_eq!(perm.len(), CANDIDATE_PREDICTORS.len());
        }
    }

    #[test]
    fn test_new_with_predictor_order_for_seed_low_seeds_match_default() {
        // Chunk 5: the constructor with any seed in 0..=3 MUST produce
        // a TreeSamples whose candidate_predictors equal the canonical
        // CANDIDATE_PREDICTORS list element-for-element. This preserves
        // the chunk-3-only invariant for low seeds.
        let base = TreeSamples::new_with_ref_channels(0);
        for seed in 0u64..=3 {
            let s = TreeSamples::new_with_predictor_order_for_seed(0, seed);
            assert_eq!(s.num_predictors(), CANDIDATE_PREDICTORS.len());
            assert_eq!(s.num_predictors(), base.num_predictors());
            for (i, c) in CANDIDATE_PREDICTORS.iter().enumerate() {
                assert_eq!(
                    s.candidate_predictors[i] as u32, *c as u32,
                    "chunk 5: seed {seed} constructor must preserve canonical predictor order at idx {i}"
                );
            }
        }
    }

    // ---- RFC#45 chunk 6 helpers (split-bucket-count + properties truncation) ----
    //      Chunk 6 gates both helpers to seeds 8..=15 so chunk-3, chunk-4,
    //      and chunk-5 perturbations on seeds 0..=7 are not recombined
    //      inside the doubled budget.

    #[test]
    fn test_derive_seeded_max_property_values_low_seeds_are_none() {
        // Chunk 6: seeds 0..=7 are reserved for chunk-3/chunk-4 slots and
        // MUST return None (canonical bucket count preserved).
        for seed in 0u64..=7 {
            assert_eq!(
                derive_seeded_max_property_values(seed),
                None,
                "chunk 6: seed {seed} must return None (chunks 3-5 slot)",
            );
        }
    }

    #[test]
    fn test_derive_seeded_max_property_values_high_seeds_active() {
        // Chunk 6: seeds 8..=11 carry three coarser bucket counts plus
        // one canonical-bucket slot (seed 11).
        assert_eq!(derive_seeded_max_property_values(8), Some(64));
        assert_eq!(derive_seeded_max_property_values(9), Some(128));
        assert_eq!(derive_seeded_max_property_values(10), Some(192));
        assert_eq!(derive_seeded_max_property_values(11), None);
        // Seeds 12..=15 are the chunk-6 truncation slot; bucket helper
        // returns None there so the two chunk-6 dimensions never stack.
        for seed in 12u64..=15 {
            assert_eq!(
                derive_seeded_max_property_values(seed),
                None,
                "seed {seed} is a truncation slot, bucket helper must hold canonical",
            );
        }
        // Distinct: 3 non-None values across seeds 8..=10.
        let vals: std::collections::BTreeSet<usize> = (8u64..=10)
            .map(|s| derive_seeded_max_property_values(s).unwrap())
            .collect();
        assert_eq!(
            vals.len(),
            3,
            "seeds 8..=10 must produce 3 distinct bucket counts"
        );
    }

    #[test]
    fn test_derive_seeded_properties_truncation_low_seeds_are_none() {
        // Chunk 6: seeds 0..=11 are reserved for chunks 3-5 + chunk-6
        // bucket slots and MUST return None (canonical slice length
        // preserved).
        for seed in 0u64..=11 {
            assert_eq!(
                derive_seeded_properties_truncation(seed),
                None,
                "chunk 6: seed {seed} must return None (chunks 3-5/bucket slot)",
            );
        }
    }

    #[test]
    fn test_derive_seeded_properties_truncation_high_seeds_active() {
        // Chunk 6: seeds 12..=15 carry three smaller property-set sizes
        // plus one canonical-size slot (seed 15).
        assert_eq!(derive_seeded_properties_truncation(12), Some(8));
        assert_eq!(derive_seeded_properties_truncation(13), Some(10));
        assert_eq!(derive_seeded_properties_truncation(14), Some(12));
        assert_eq!(derive_seeded_properties_truncation(15), None);
        // Seeds >= 16 fall outside the chunk-6 budget; helper returns
        // None (any future chunk-7 dimension owns its own slot range).
        assert_eq!(derive_seeded_properties_truncation(16), None);
        assert_eq!(derive_seeded_properties_truncation(17), None);
        // Distinct: 3 non-None values across seeds 12..=14.
        let vals: std::collections::BTreeSet<usize> = (12u64..=14)
            .map(|s| derive_seeded_properties_truncation(s).unwrap())
            .collect();
        assert_eq!(
            vals.len(),
            3,
            "seeds 12..=14 must produce 3 distinct truncation caps"
        );
    }

    #[test]
    fn test_chunk6_dimensions_are_orthogonal() {
        // Sanity: bucket-count slot (8..=11) MUST return None from the
        // truncation helper, and truncation slot (12..=15) MUST return
        // None from the bucket-count helper. Otherwise a single seed
        // would activate two chunk-6 dimensions at once, defeating the
        // seed-slot split discipline.
        for seed in 8u64..=11 {
            assert_eq!(
                derive_seeded_properties_truncation(seed),
                None,
                "chunk 6: bucket-count slot seed {seed} must NOT trigger truncation",
            );
        }
        for seed in 12u64..=15 {
            assert_eq!(
                derive_seeded_max_property_values(seed),
                None,
                "chunk 6: truncation slot seed {seed} must NOT trigger bucket override",
            );
        }
    }

    // ---------------- RFC#45 chunk 7 early-out helper tests ----------------

    #[test]
    fn test_early_out_below_threshold_fires() {
        // Tight cluster of costs (spread ~0.1%) below the 5% threshold
        // → early-out fires.
        let costs = [100_000.0, 100_050.0, 100_010.0, 100_080.0];
        assert!(multi_seed_early_out_after_probe(&costs, 4, 16));
    }

    #[test]
    fn test_early_out_above_threshold_does_not_fire() {
        // Spread ~26% (well above 5%) → keep running. Mirrors the
        // 1279330 cell where chunk 6's seeds 4..15 add 0.78% improvement.
        let costs = [100_000.0, 126_000.0, 102_000.0, 110_000.0];
        assert!(!multi_seed_early_out_after_probe(&costs, 4, 16));
    }

    #[test]
    fn test_early_out_at_4pct_fires() {
        // Spread ~4% (under 5% threshold) → fires. Mirrors the 1044329
        // cell where chunk-3 spread is 4.14% and seeds 4..15 add 0%
        // improvement.
        let costs = [100_000.0, 104_000.0, 102_000.0, 103_000.0];
        assert!(multi_seed_early_out_after_probe(&costs, 4, 16));
    }

    #[test]
    fn test_early_out_no_skip_when_probe_eq_total() {
        // No seeds to skip → must return false even when spread is tiny.
        let costs = [100_000.0, 100_001.0, 100_002.0, 100_003.0];
        assert!(!multi_seed_early_out_after_probe(&costs, 4, 4));
    }

    #[test]
    fn test_early_out_no_skip_when_probe_gt_total() {
        // Defensive: probe > total → return false (no skip possible).
        let costs = [100_000.0, 100_001.0];
        assert!(!multi_seed_early_out_after_probe(&costs, 4, 2));
    }

    #[test]
    fn test_early_out_handles_single_cost() {
        // With < 2 probe samples, spread is undefined → don't fire.
        let costs = [100_000.0];
        assert!(!multi_seed_early_out_after_probe(&costs, 1, 16));
    }

    #[test]
    fn test_early_out_handles_zero_or_negative_cost() {
        // Defensive guard against pathological inputs — should not panic
        // and should not fire (we have no way to compute relative spread).
        let costs = [0.0, 100.0, 200.0, 300.0];
        assert!(!multi_seed_early_out_after_probe(&costs, 4, 16));
    }

    #[test]
    fn test_early_out_identical_costs_fires() {
        // Perfectly identical costs → spread = 0 < 0.5% threshold → fire.
        let costs = [100_000.0; 4];
        assert!(multi_seed_early_out_after_probe(&costs, 4, 16));
    }

    #[test]
    fn test_early_out_just_above_threshold_does_not_fire() {
        // Spread slightly above 0.5% → must NOT fire (preserves chunk-6
        // behaviour on borderline cells).
        let lo = 100_000.0_f64;
        let hi = lo * (1.0 + MULTI_SEED_EARLY_OUT_SPREAD_THRESHOLD * 2.0);
        let costs = [lo, hi, lo, hi];
        assert!(!multi_seed_early_out_after_probe(&costs, 4, 16));
    }

    #[test]
    fn test_early_out_just_under_threshold_fires() {
        // Spread just under 0.5% → fires.
        let lo = 100_000.0_f64;
        let hi = lo * (1.0 + MULTI_SEED_EARLY_OUT_SPREAD_THRESHOLD * 0.5);
        let costs = [lo, hi, lo, hi];
        assert!(multi_seed_early_out_after_probe(&costs, 4, 16));
    }

    #[test]
    fn test_early_out_probe_seeds_constant_is_4() {
        // Hard-locked to chunk-3 seed slot. Changing this is a knob
        // change that needs paired bench evidence.
        assert_eq!(MULTI_SEED_EARLY_OUT_PROBE_SEEDS, 4);
    }

    #[test]
    fn test_early_out_threshold_constant_is_5pct() {
        // Hard-locked to 5%. Same caveat as above — calibrated against
        // chunk-6 paired bench's bimodal split (1.7-4.1% on converged
        // cells vs 25.9% on the one improving cell).
        assert!((MULTI_SEED_EARLY_OUT_SPREAD_THRESHOLD - 0.05).abs() < 1e-12);
    }

    /// Integration test for the SubOverflow fuzz-hardening guard ported
    /// from libjxl commit `87bee19` (PR #4759).
    ///
    /// Constructs a 1×2 single-channel image whose second row triggers
    /// `pixel - prediction` integer overflow under the `Top` predictor:
    ///
    /// - row 0: a moderate negative value `-1_000_000` (avoids the
    ///   `pack_signed(i32::MIN)` separate-but-related overflow at the
    ///   pack step, which would mask the residual-overflow guard).
    ///   This serves as the `Top` neighbor for row 1.
    /// - row 1: `i32::MAX`. residual = `i32::MAX - (-1_000_000)` =
    ///   `i32::MAX + 1_000_000` → overflows i32.
    ///
    /// The budget-aware entry returns `Err(InvalidInput("Residual
    /// overflow ..."))` instead of corrupting the token stream. Without
    /// the guard the `pixel - prediction` subtraction panics in debug
    /// (`attempt to subtract with overflow`) and silently wraps in
    /// release, both of which are wrong for adversarial fuzz input.
    #[test]
    fn test_residual_overflow_rejected_with_top_predictor() {
        use crate::modular::channel::Channel;
        use crate::modular::predictor::{Predictor, WeightedPredictorParams};
        use crate::modular::tree::{PropertyDecisionNode, Tree};

        let channel = Channel::from_vec(vec![-1_000_000, i32::MAX], 1, 2)
            .expect("Channel::from_vec accepts adversarial i32 values");
        let image = ModularImage {
            channels: vec![channel],
            bit_depth: 32,
            is_grayscale: true,
            has_alpha: false,
        };

        // Single-leaf tree using the Top predictor.
        let tree: Tree = vec![PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Top,
            context_id: 0,
            ..Default::default()
        }];
        let wp_params = WeightedPredictorParams::default();

        let result =
            collect_residuals_with_tree_offset_with_budget(&image, &tree, 0, 0, &wp_params, None);
        match result {
            Err(crate::error::Error::InvalidInput(msg)) => {
                assert!(
                    msg.contains("Residual overflow") || msg.contains("overflow"),
                    "expected residual-overflow error, got: {msg}"
                );
            }
            Err(other) => panic!("expected Error::InvalidInput, got: {other:?}"),
            Ok(_) => panic!("adversarial i32::MAX - (-1_000_000) should have overflowed"),
        }
    }

    /// Companion to [`test_residual_overflow_rejected_with_top_predictor`]:
    /// valid 8-bit input through the same path must NOT trip the guard.
    /// This pins down the "valid input never reaches it" invariant the
    /// `.expect` on the budget-less wrapper relies on.
    #[test]
    fn test_residual_overflow_guard_zero_overhead_on_valid_input() {
        use crate::modular::channel::Channel;
        use crate::modular::predictor::{Predictor, WeightedPredictorParams};
        use crate::modular::tree::{PropertyDecisionNode, Tree};

        let channel = Channel::from_vec(vec![100, 110, 120, 130], 2, 2).unwrap();
        let image = ModularImage {
            channels: vec![channel],
            bit_depth: 8,
            is_grayscale: true,
            has_alpha: false,
        };
        let tree: Tree = vec![PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Top,
            context_id: 0,
            ..Default::default()
        }];
        let wp_params = WeightedPredictorParams::default();

        let tokens =
            collect_residuals_with_tree_offset_with_budget(&image, &tree, 0, 0, &wp_params, None)
                .expect("valid 8-bit input must not trip the overflow guard");
        assert_eq!(tokens.len(), 4);
    }

    // ── PERF-HIST-SUB-LOSSLESS tests (issue #64 chunk 1) ─────────────────
    //
    // Byte-identity proof obligations from
    // `benchmarks/perf_hist_sub_2026-06-10.meta` point 6: derived child
    // tensors equal from-scratch child tensors elementwise; captured
    // tensors equal built tensors; `find_best_split` with a tensor input
    // returns a bit-identical split to the per-sample path; the engine's
    // tensor path produces an identical tree to the full-rebuild reference.

    fn hist_sub_xorshift(state: &mut u32) -> u32 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *state = s;
        s
    }

    /// Deterministic noise+gradient image → gathered, pre-quantized,
    /// deduped samples. Noise dedups poorly (photo-like — the chunk's
    /// lossless-photo target), so `n` stays close to the pixel count.
    fn hist_sub_test_samples(
        seed: u32,
        w: usize,
        h: usize,
        params: &TreeLearningParams,
    ) -> (TreeSamples, PreQuantizedProps, usize) {
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
        let mut ch0 = Channel::new(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                // Structured gradient with seeded noise — non-degenerate
                // residuals for every predictor.
                let noise = (hist_sub_xorshift(&mut state) >> 24) & 0x3F;
                ch0.set(x, y, (((x * 3 + y * 7) as u32 + noise) & 0xFF) as i32);
            }
        }
        image.channels.push(ch0);
        let mut ch1 = Channel::new(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                let v = hist_sub_xorshift(&mut state) & 0xFF;
                ch1.set(x, y, v as i32);
            }
        }
        image.channels.push(ch1);

        let mut samples = TreeSamples::new();
        gather_samples(&mut samples, &image, 0);
        let mut pq = samples.pre_quantize(params);
        dedup_samples(&mut samples, &mut pq, params);
        let max_token = samples
            .residual_tokens
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        (samples, pq, max_token + 1)
    }

    fn assert_tensors_identical(a: &NodeTensor, b: &NodeTensor, ctx: &str) {
        assert_eq!(a.token_counts, b.token_counts, "{ctx}: token_counts");
        assert_eq!(a.ebit_sums, b.ebit_sums, "{ctx}: ebit_sums");
        assert_eq!(a.weighted, b.weighted, "{ctx}: weighted");
        assert_eq!(a.unique, b.unique, "{ctx}: unique");
    }

    /// Meta point 6 core: random samples → parent tensor + split → derived
    /// child tensor == from-scratch child tensor, EXACT equality, several
    /// seeds + sizes including weighted totals above/below the 2048 gate
    /// (the math is gate-independent; the gate itself is tested separately).
    #[test]
    fn test_node_tensor_derived_equals_built() {
        let params = TreeLearningParams::for_effort(7);
        // (seed, w, h): 96×96×2ch ≈ 18K weighted (far above 2·2048);
        // 48×40×2ch = 3840 (between 2048 and 4096); 24×20×2ch = 960
        // (below the 2048 child gate).
        for &(seed, w, h) in &[(1u32, 96usize, 96usize), (2, 48, 40), (3, 24, 20)] {
            let (samples, pq, histogram_size) = hist_sub_test_samples(seed, w, h, &params);
            let n = samples.num_samples;
            let layout =
                TensorLayout::new(&params, samples.num_predictors(), histogram_size, |p| {
                    pq.num_thresholds(p)
                });
            let mut parent = NodeTensor::zeroed(&layout);
            build_node_tensor(
                &samples,
                &pq,
                &params,
                &layout,
                histogram_size,
                0,
                n,
                &mut parent,
            );

            for &(num, den) in &[(1usize, 3usize), (1, 2), (7, 8)] {
                let mid = (n * num / den).max(1).min(n - 1);
                let mut left = NodeTensor::zeroed(&layout);
                build_node_tensor(
                    &samples,
                    &pq,
                    &params,
                    &layout,
                    histogram_size,
                    0,
                    mid,
                    &mut left,
                );
                let mut right = NodeTensor::zeroed(&layout);
                build_node_tensor(
                    &samples,
                    &pq,
                    &params,
                    &layout,
                    histogram_size,
                    mid,
                    n,
                    &mut right,
                );

                // Derived larger (right) = parent − built smaller (left).
                let mut derived_right = NodeTensor::zeroed(&layout);
                build_node_tensor(
                    &samples,
                    &pq,
                    &params,
                    &layout,
                    histogram_size,
                    0,
                    n,
                    &mut derived_right,
                );
                derived_right.subtract_in_place(&left);
                assert_tensors_identical(
                    &derived_right,
                    &right,
                    &format!("seed={seed} {w}x{h} mid={mid} parent-left"),
                );

                // And the mirror: parent − right == left.
                let mut derived_left = NodeTensor::zeroed(&layout);
                build_node_tensor(
                    &samples,
                    &pq,
                    &params,
                    &layout,
                    histogram_size,
                    0,
                    n,
                    &mut derived_left,
                );
                derived_left.subtract_in_place(&right);
                assert_tensors_identical(
                    &derived_left,
                    &left,
                    &format!("seed={seed} {w}x{h} mid={mid} parent-right"),
                );
            }
            // Sanity: the parent tensor is non-trivial.
            assert!(parent.token_counts.iter().any(|&c| c != 0));
        }
    }

    /// The capture path inside `find_best_split` must produce the exact
    /// tensor `build_node_tensor` produces (single producer-pair
    /// consistency — what makes engine-level subtraction exact).
    #[test]
    fn test_node_tensor_capture_equals_built() {
        let params = TreeLearningParams::for_effort(7);
        for seed in [11u32, 12, 13] {
            let (samples, pq, histogram_size) = hist_sub_test_samples(seed, 96, 96, &params);
            let n = samples.num_samples;
            let layout =
                TensorLayout::new(&params, samples.num_predictors(), histogram_size, |p| {
                    pq.num_thresholds(p)
                });
            let max_buckets = params.max_property_values + 1;
            let mut entropy_counts = vec![0u32; histogram_size];
            let root_pred =
                find_best_predictor(&samples, 0, n, histogram_size, &mut entropy_counts);
            let root_bits = compute_predictor_entropy(
                &samples,
                0,
                n,
                root_pred,
                histogram_size,
                &mut entropy_counts,
            );
            let required_cost = params.pixel_fraction * 0.9 + 0.1;
            let threshold = params.split_threshold * required_cost;

            let mut captured = NodeTensor::zeroed(&layout);
            let mut ws = SplitWorkspace::new(n, histogram_size, max_buckets);
            let split = find_best_split(
                &samples,
                0,
                n,
                histogram_size,
                root_bits,
                &params,
                root_pred,
                threshold,
                &pq,
                &mut ws,
                TensorMode::Capture(&layout, &mut captured),
            );
            // Capture completeness is only guaranteed when a split exists —
            // which it must on this noise+gradient input.
            assert!(split.is_some(), "seed={seed}: expected a split");

            let mut built = NodeTensor::zeroed(&layout);
            build_node_tensor(
                &samples,
                &pq,
                &params,
                &layout,
                histogram_size,
                0,
                n,
                &mut built,
            );
            assert_tensors_identical(&captured, &built, &format!("seed={seed} capture-vs-build"));
        }
    }

    /// `find_best_split` with `TensorMode::Use` must return a bit-identical
    /// split to `TensorMode::Off` — same property, splitval, predictors,
    /// left_count, and bit-equal f64 cost.
    #[test]
    fn test_find_best_split_tensor_use_matches_off() {
        let params = TreeLearningParams::for_effort(7);
        for seed in [21u32, 22] {
            let (samples, pq, histogram_size) = hist_sub_test_samples(seed, 96, 96, &params);
            let n = samples.num_samples;
            let layout =
                TensorLayout::new(&params, samples.num_predictors(), histogram_size, |p| {
                    pq.num_thresholds(p)
                });
            let max_buckets = params.max_property_values + 1;
            let mut entropy_counts = vec![0u32; histogram_size];
            let required_cost = params.pixel_fraction * 0.9 + 0.1;
            let threshold = params.split_threshold * required_cost;

            // Full range + an interior subrange (both ≥ 2048 weighted).
            let ranges = [
                (0usize, n),
                (n / 4, n / 4 + (n / 2).max(4096).min(n - n / 4)),
            ];
            for &(start, end) in &ranges {
                let count = end - start;
                let weighted: u32 = samples.sample_counts[start..end].iter().sum();
                assert!(weighted >= TENSOR_MIN_CHILD_WEIGHT, "test range too small");
                let root_pred =
                    find_best_predictor(&samples, start, end, histogram_size, &mut entropy_counts);
                let base_bits = compute_predictor_entropy(
                    &samples,
                    start,
                    end,
                    root_pred,
                    histogram_size,
                    &mut entropy_counts,
                );

                let mut ws = SplitWorkspace::new(count, histogram_size, max_buckets);
                let split_off = find_best_split(
                    &samples,
                    start,
                    end,
                    histogram_size,
                    base_bits,
                    &params,
                    root_pred,
                    threshold,
                    &pq,
                    &mut ws,
                    TensorMode::Off,
                );

                let mut tensor = NodeTensor::zeroed(&layout);
                build_node_tensor(
                    &samples,
                    &pq,
                    &params,
                    &layout,
                    histogram_size,
                    start,
                    end,
                    &mut tensor,
                );
                let mut ws2 = SplitWorkspace::new(count, histogram_size, max_buckets);
                let split_use = find_best_split(
                    &samples,
                    start,
                    end,
                    histogram_size,
                    base_bits,
                    &params,
                    root_pred,
                    threshold,
                    &pq,
                    &mut ws2,
                    TensorMode::Use(&layout, &tensor),
                );

                match (split_off, split_use) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert_eq!(a.property, b.property, "seed={seed} [{start},{end})");
                        assert_eq!(a.splitval, b.splitval, "seed={seed} [{start},{end})");
                        assert_eq!(a.left_predictor, b.left_predictor, "seed={seed}");
                        assert_eq!(a.right_predictor, b.right_predictor, "seed={seed}");
                        assert_eq!(a.left_count, b.left_count, "seed={seed}");
                        assert_eq!(
                            a.total_bits.to_bits(),
                            b.total_bits.to_bits(),
                            "seed={seed} [{start},{end}): total_bits must be BIT-identical"
                        );
                    }
                    (a, b) => panic!(
                        "seed={seed} [{start},{end}): Off={:?} Use={:?} disagree on Some/None",
                        a.map(|s| s.property),
                        b.map(|s| s.property)
                    ),
                }
            }
        }
    }

    /// The 2048-weighted child gate of `tensor_split_plan`, just above and
    /// just below (meta point 6's gate-boundary requirement).
    #[test]
    fn test_tensor_split_plan_2048_gate() {
        // Tiny synthetic layout where derive always pays: 1 property,
        // 1 predictor, 2 buckets, histo 1 → token_cells = 2.
        let params = TreeLearningParams {
            properties: vec![0],
            ..TreeLearningParams::for_effort(7)
        };
        let layout = TensorLayout::new(&params, 1, 1, |_| 1);
        assert_eq!(layout.token_cells, 2);

        let n = 8192usize;
        let counts = vec![1u32; n];
        let above = threshold_gate_case(&layout, &counts, 2048);
        assert!(above.is_some(), "left_w=2048 must pass the >= 2048 gate");
        let below = threshold_gate_case(&layout, &counts, 2047);
        assert!(below.is_none(), "left_w=2047 must fail the >= 2048 gate");
        // Right side just below: mid = n-2047 → right_w = 2047.
        let right_below = threshold_gate_case(&layout, &counts, n - 2047);
        assert!(right_below.is_none(), "right_w=2047 must fail the gate");
        let right_above = threshold_gate_case(&layout, &counts, n - 2048);
        assert!(right_above.is_some(), "right_w=2048 must pass the gate");

        // Leaf-bound children skip derivation outright.
        assert!(
            tensor_split_plan(&layout, &counts, 0, 4096, n, 0.5, 1e9, 1.0).is_none(),
            "left child below threshold must skip derivation"
        );
    }

    fn threshold_gate_case(
        layout: &TensorLayout,
        counts: &[u32],
        mid: usize,
    ) -> Option<(bool, usize, usize)> {
        tensor_split_plan(layout, counts, 0, mid, counts.len(), 1e9, 1e9, 1.0)
    }

    /// Reference implementation of the greedy stack engine with
    /// `TensorMode::Off` everywhere — byte-for-byte the pre-PERF-HIST-SUB
    /// algorithm (find_best_split → partition → recompute child bits →
    /// push), independent of the parallel-tree-learning feature.
    fn reference_greedy_tree_no_tensors(
        samples: &mut TreeSamples,
        params: &TreeLearningParams,
    ) -> Tree {
        let mut pq = samples.pre_quantize(params);
        dedup_samples(samples, &mut pq, params);
        let required_cost = params.pixel_fraction * 0.9 + 0.1;
        let threshold = params.split_threshold * required_cost;
        let n = samples.num_samples;
        let max_token = samples
            .residual_tokens
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let histogram_size = max_token + 1;
        let max_buckets = params.max_property_values + 1;
        let mut entropy_counts = vec![0u32; histogram_size];
        let root_pred = find_best_predictor(samples, 0, n, histogram_size, &mut entropy_counts);
        let root_bits = compute_predictor_entropy(
            samples,
            0,
            n,
            root_pred,
            histogram_size,
            &mut entropy_counts,
        );

        let mut tree: Tree = Vec::new();
        tree.push(PropertyDecisionNode::default());
        let mut stack: Vec<SplitCandidate> = Vec::new();
        stack.push(SplitCandidate {
            node_idx: 0,
            start: 0,
            end: n,
            best_predictor: root_pred,
            base_bits: root_bits,
            multiplier: None,
            tensor: None,
        });
        while let Some(candidate) = stack.pop() {
            let count = candidate.end - candidate.start;
            if tree.len() + 2 > params.max_nodes || count < 2 || candidate.base_bits <= threshold {
                finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
                continue;
            }
            let mut ws = SplitWorkspace::new(count, histogram_size, max_buckets);
            let best_split = find_best_split(
                samples,
                candidate.start,
                candidate.end,
                histogram_size,
                candidate.base_bits,
                params,
                candidate.best_predictor,
                threshold,
                &pq,
                &mut ws,
                TensorMode::Off,
            );
            match best_split {
                Some(split) if candidate.base_bits - split.total_bits > threshold => {
                    let bucket_split =
                        bucket_for_splitval(&pq.threshold_sets[split.property], split.splitval);
                    let abs_mid = partition_node_in_place_with(
                        samples,
                        &mut pq,
                        candidate.start,
                        candidate.end,
                        split.left_count,
                        tree_learn_split::PartitionKey::Bucket {
                            prop_idx: split.property,
                            val: bucket_split as u8,
                        },
                        true,
                    );
                    let lchild_idx = tree.len();
                    let rchild_idx = tree.len() + 1;
                    tree.push(PropertyDecisionNode::default());
                    tree.push(PropertyDecisionNode::default());
                    tree[candidate.node_idx] = PropertyDecisionNode {
                        property: split.property as i32,
                        splitval: split.splitval,
                        lchild: lchild_idx,
                        rchild: rchild_idx,
                        ..Default::default()
                    };
                    let lb = compute_predictor_entropy(
                        samples,
                        candidate.start,
                        abs_mid,
                        split.left_predictor,
                        histogram_size,
                        &mut entropy_counts,
                    );
                    let rb = compute_predictor_entropy(
                        samples,
                        abs_mid,
                        candidate.end,
                        split.right_predictor,
                        histogram_size,
                        &mut entropy_counts,
                    );
                    stack.push(SplitCandidate {
                        node_idx: rchild_idx,
                        start: abs_mid,
                        end: candidate.end,
                        best_predictor: split.right_predictor,
                        base_bits: rb,
                        multiplier: None,
                        tensor: None,
                    });
                    stack.push(SplitCandidate {
                        node_idx: lchild_idx,
                        start: candidate.start,
                        end: abs_mid,
                        best_predictor: split.left_predictor,
                        base_bits: lb,
                        multiplier: None,
                        tensor: None,
                    });
                }
                _ => {
                    finalize_leaf(&mut tree, &candidate, samples.candidate_predictors);
                }
            }
        }
        assign_sequential_contexts(&mut tree);
        tree
    }

    /// Engine-level proof: the sequential stack engine WITH the tensor path
    /// active (capture + derive fire — asserted via the thread-local derive
    /// counter) produces a token-identical tree to the owned full-rebuild
    /// engine (`build_subtree_sequential`, TensorMode::Off).
    #[test]
    fn test_engine_tensor_path_tree_identical_to_full_rebuild() {
        use crate::modular::tree::collect_tree_tokens;

        // Shrink the per-tensor cost so capture/derive gates fire at test
        // sizes: 4 properties × ≤9 buckets → token_cells ≈ 25K, so
        // derive_pays needs larger_unique > ~1.8K (96×96×2ch ≈ 18K rows).
        let mut params = TreeLearningParams::for_effort(7);
        params.properties.truncate(4);
        params.max_property_values = 8;
        // Force the sequential stack loop even with parallel-tree-learning
        // compiled in (the parallel root path is exercised by
        // `test_parallel_tree_matches_sequential`).
        params.parallel_root_threshold = usize::MAX;

        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut state = 0xC0FF_EE01u32;
        let mut ch0 = Channel::new(96, 96).unwrap();
        for y in 0..96 {
            for x in 0..96 {
                let noise = (hist_sub_xorshift(&mut state) >> 26) & 0x1F;
                ch0.set(x, y, (((x * 5 + y * 3) as u32 + noise) & 0xFF) as i32);
            }
        }
        image.channels.push(ch0);
        let mut ch1 = Channel::new(96, 96).unwrap();
        for y in 0..96 {
            for x in 0..96 {
                ch1.set(x, y, (hist_sub_xorshift(&mut state) & 0xFF) as i32);
            }
        }
        image.channels.push(ch1);

        // Tensor-path tree via the production engine.
        TENSOR_DERIVE_COUNT.with(|c| c.set(0));
        let mut samples_tensor = TreeSamples::new();
        gather_samples(&mut samples_tensor, &image, 0);
        let tensor_tree = compute_best_tree(&mut samples_tensor, &params);
        let derives = TENSOR_DERIVE_COUNT.with(|c| c.get());
        assert!(
            derives > 0,
            "tensor derivation must actually fire in this test (gates mis-tuned?)"
        );

        // Full-rebuild reference: the pre-chunk greedy loop, replicated
        // with TensorMode::Off (cfg-independent — `build_subtree_sequential`
        // only exists under parallel-tree-learning).
        let mut samples_ref = TreeSamples::new();
        gather_samples(&mut samples_ref, &image, 0);
        let ref_tree = reference_greedy_tree_no_tensors(&mut samples_ref, &params);

        assert!(
            tensor_tree.len() >= 3,
            "tensor tree must split at least once"
        );
        let a = collect_tree_tokens(&tensor_tree);
        let b = collect_tree_tokens(&ref_tree);
        assert_eq!(a.len(), b.len(), "tree token count differs");
        for (i, (p, s)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                (p.context, p.value, p.is_signed),
                (s.context, s.value, s.is_signed),
                "token #{i} differs between tensor path and full rebuild"
            );
        }
    }

    /// Borrowed-view tensor build must equal the owned build (same rows).
    #[cfg(feature = "parallel-tree-learning")]
    #[test]
    fn test_node_tensor_borrowed_build_matches_owned() {
        let params = TreeLearningParams::for_effort(7);
        let (mut samples, mut pq, histogram_size) = hist_sub_test_samples(31, 64, 64, &params);
        let n = samples.num_samples;
        let layout = TensorLayout::new(&params, samples.num_predictors(), histogram_size, |p| {
            pq.num_thresholds(p)
        });

        let mut owned = NodeTensor::zeroed(&layout);
        build_node_tensor(
            &samples,
            &pq,
            &params,
            &layout,
            histogram_size,
            0,
            n,
            &mut owned,
        );

        let view = BorrowedSamples::from_owned(&mut samples, &mut pq);
        let mut borrowed = NodeTensor::zeroed(&layout);
        build_node_tensor_borrowed(&view, &params, &layout, histogram_size, 0, n, &mut borrowed);
        assert_tensors_identical(&owned, &borrowed, "borrowed-vs-owned build");
    }
}
