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
    pub gather_dedup: bool,
    /// Phase 3 of issue #41 — when [`Self::gather_dedup`] is also `true`,
    /// route the gather-time dedup table through
    /// [`crate::modular::inline_dedup_table::InlineDedupTable`] instead of
    /// [`GatherDedupTable`]. The post-sort arbiter (`dedup_samples`) still
    /// runs, so bitstream hash-locks stay byte-identical to Phase 2's
    /// gather-dedup baseline.
    ///
    /// Default `false`. Has no effect when `gather_dedup` is `false`.
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
    props
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
#[allow(clippy::too_many_arguments)]
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
        wp_params,
        budget,
        dedup_table.map(GatherDedupBackend::Phase2),
    )
}

/// Backend-aware variant of [`gather_samples_strided_with_budget_inner`]
/// that dispatches into either Phase 2 or Phase 3 of issue #41 based on
/// the [`GatherDedupBackend`] variant supplied.
#[allow(clippy::too_many_arguments)]
fn gather_samples_strided_with_budget_inner_backend(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
    wp_params: &WeightedPredictorParams,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    mut dedup_backend: Option<GatherDedupBackend<'_>>,
) -> crate::error::Result<()> {
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
    Ok(())
}

/// Compute maximum tree samples from an [`EffortProfile`].
///
/// Uses `tree_max_samples_fixed` (when > 0) or `tree_sample_fraction` (when > 0).
pub fn max_tree_samples_from_profile(
    profile: &crate::effort::EffortProfile,
    total_pixels: usize,
) -> usize {
    if profile.tree_sample_fraction > 0.0 {
        // Fraction-based: e.g. 50% of pixels, min 65K
        ((total_pixels as f32 * profile.tree_sample_fraction) as usize).max(65_536)
    } else if profile.tree_max_samples_fixed > 0 {
        profile.tree_max_samples_fixed as usize
    } else {
        32_768
    }
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
#[allow(clippy::too_many_arguments)]
fn gather_channel_samples(
    samples: &mut TreeSamples,
    channel: &Channel,
    channel_idx: u32,
    group_id: u32,
    stride: usize,
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

    // Counter for subsampling: only gather when counter == 0
    let mut subsample_counter: usize = 0;

    let max_refs = samples.num_ref_channels;

    // Cache field-counts referenced from the inner loop's dedup probe.
    let num_pred = samples.num_predictors();
    let total_props = samples.total_num_properties();
    // Detach the optional borrow so each `add` step can decide whether to
    // call `try_merge_last` without re-asking for the option.
    let mut dedup_backend = dedup_backend;

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

            // Compute WP prediction and max_error property
            let (wp_pred, wp_max_error) = wp_state.predict_and_property(x, y, width, &n);

            // Always update WP error tracking to maintain state continuity
            wp_state.update_errors(pixel, x, y, width);

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
                    subsample_counter = stride - 1;
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

                subsample_counter = stride - 1;
            } else {
                // Still need to track gradient for subsequent pixels
                let grad = n.w.wrapping_add(n.n).wrapping_sub(n.nw);
                prev_gradient = grad;

                subsample_counter -= 1;
            }
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
    let mut order: Vec<u32> = (0..n as u32).collect();
    // Use rayon's par_sort_unstable_by when the parallel feature is on —
    // dropping into the standard sort path otherwise. The cmp reads two
    // adjacent 64-byte keys (sequential memory accesses, no shared
    // mutable state) so rayon's pdqsort backend parallelizes cleanly.
    #[cfg(feature = "parallel")]
    {
        use rayon::slice::ParallelSliceMut;
        order.par_sort_unstable_by(|&a, &b| {
            let ka = &keys[a as usize];
            let kb = &keys[b as usize];
            ka.cmp(kb)
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        order.sort_unstable_by(|&a, &b| {
            let ka = &keys[a as usize];
            let kb = &keys[b as usize];
            ka.cmp(kb)
        });
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
        if keys[curr] == keys[prev_key_idx] {
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
    let new_props_per_idx: Vec<Vec<i32>> = crate::parallel::parallel_map(total_props, |prop_idx| {
        let old_props = &samples.props[prop_idx];
        if old_props.is_empty() {
            Vec::new()
        } else {
            unique_indices.iter().map(|&i| old_props[i]).collect()
        }
    });
    for (prop_idx, new_props) in new_props_per_idx.into_iter().enumerate() {
        if !samples.props[prop_idx].is_empty() {
            samples.props[prop_idx] = new_props;
        }
    }

    let bi_total = pq.bucket_indices.len().min(total_props);
    let new_bi_per_idx: Vec<Vec<u8>> = crate::parallel::parallel_map(bi_total, |prop_idx| {
        let old_bi = &pq.bucket_indices[prop_idx];
        if old_bi.is_empty() {
            Vec::new()
        } else {
            unique_indices.iter().map(|&i| old_bi[i]).collect()
        }
    });
    for (prop_idx, new_bi) in new_bi_per_idx.into_iter().enumerate() {
        if !pq.bucket_indices[prop_idx].is_empty() {
            pq.bucket_indices[prop_idx] = new_bi;
        }
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
    });

    // The workspace lives in the thread-local cache (see
    // `with_thread_local_workspace`) so we don't allocate ~12 MB per fork on
    // the parallel path. The cache grows in place; subsequent calls on the
    // same worker thread are allocation-free.
    let max_buckets = params.max_property_values + 1;

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

                    // Compute per-side base bits before splitting (uses the
                    // already-allocated entropy_counts and the immutable
                    // samples view; cheap relative to subtree builds).
                    let lb = compute_predictor_entropy(
                        samples,
                        root_candidate.start,
                        abs_mid,
                        split.left_predictor,
                        histogram_size,
                        &mut entropy_counts,
                    );
                    let rb = compute_predictor_entropy(
                        samples,
                        abs_mid,
                        root_candidate.end,
                        split.right_predictor,
                        histogram_size,
                        &mut entropy_counts,
                    );

                    // Set the root split node in the parent tree.
                    // Children indices are filled in after stitching.
                    let left_predictor = split.left_predictor;
                    let right_predictor = split.right_predictor;
                    let split_property = split.property as i32;
                    let split_splitval = split.splitval;

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

                    let (left_tree, right_tree) = crate::parallel::parallel_join(
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
                            )
                        },
                    );

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

    while let Some(candidate) = stack.pop() {
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
        let n_node = candidate.end - candidate.start;
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

                // Recompute child costs from ALL samples (not the eval subset).
                // The eval subset's costs are scaled by cost_scale which introduces
                // error at high strides. Re-scoring with full samples prevents error
                // accumulation down the tree. This is O(N) per split — negligible
                // compared to the O(N*P*K) search.
                let (left_bits, right_bits) = crate::profile_time!("tree/recompute_child_bits", {
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
                    (lb, rb)
                });

                stack.push(SplitCandidate {
                    node_idx: rchild_idx,
                    start: abs_mid,
                    end: candidate.end,
                    best_predictor: split.right_predictor,
                    base_bits: right_bits,
                    multiplier: None,
                });

                stack.push(SplitCandidate {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: split.left_predictor,
                    base_bits: left_bits,
                    multiplier: None,
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
#[cfg(feature = "parallel-tree-learning")]
#[cfg(test)]
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
    // — no per-call ~12 MB allocation. Only used by the layer-2 invariant test
    // `test_parallel_tree_matches_sequential` as a Vec-based reference; the
    // production parallel path uses `build_subtree_sequential_borrowed`.
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
                });
                stack.push(SplitCandidate {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: split.left_predictor,
                    base_bits: lb,
                    multiplier: None,
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
) -> Option<BestSplit> {
    let count = end - start;
    if count < 2 {
        return None;
    }

    let total_num_pred = samples.num_predictors();
    let mut best: Option<BestSplit> = None;
    let mut best_bits = base_bits;

    let sample_counts = &samples.sample_counts[start..end];

    let weighted_total: u32 = sample_counts.iter().sum();

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

    for &prop_idx in &params.properties[..num_props] {
        let num_thresholds = samples.num_thresholds(prop_idx);
        if num_thresholds == 0 {
            continue;
        }

        let pq_buckets = &samples.bucket_indices[prop_idx][start..end];
        let threshold_set = &samples.threshold_sets[prop_idx];

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
        if bmin == bmax {
            continue;
        }
        let bmin = bmin as usize;
        let bmax = bmax as usize;

        let local_num_buckets = bmax - bmin + 1;
        let local_num_thresholds = bmax - bmin;

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

        bucket_write_pos[..local_num_buckets].copy_from_slice(&bucket_starts[..local_num_buckets]);
        for (offset, &b) in pq_buckets.iter().enumerate() {
            let local_b = (b as usize) - bmin;
            sorted_by_bucket[bucket_write_pos[local_b]] = offset;
            bucket_write_pos[local_b] += 1;
        }

        best_l_cost[..local_num_thresholds].fill(f64::MAX);
        best_r_cost[..local_num_thresholds].fill(f64::MAX);
        best_l_penalized[..local_num_thresholds].fill(f64::MAX);
        best_r_penalized[..local_num_thresholds].fill(f64::MAX);
        best_l_pred[..local_num_thresholds].fill(0);
        best_r_pred[..local_num_thresholds].fill(0);

        for pred in 0..num_pred {
            let tokens = &samples.residual_tokens[pred][start..end];
            let ebits = &samples.extra_bits[pred][start..end];

            let mut penalty: f64 = 0.0;
            if pred != parent_predictor && parent_predictor != weighted_idx {
                penalty = change_pred_penalty;
            }
            if pred == weighted_idx {
                penalty += 1e-8;
            } else if pred == zero_idx {
                penalty -= 1e-8;
            }

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

        for local_k in 0..local_num_thresholds {
            if best_l_cost[local_k] == f64::MAX || best_r_cost[local_k] == f64::MAX {
                continue;
            }

            let total = best_l_cost[local_k] + best_r_cost[local_k];

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
                });
            }
        }
    }

    best
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

                let lb = compute_predictor_entropy_borrowed(
                    samples,
                    candidate.start,
                    abs_mid,
                    split.left_predictor,
                    histogram_size,
                    &mut entropy_counts,
                );
                let rb = compute_predictor_entropy_borrowed(
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
                });
                stack.push(SplitCandidate {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: split.left_predictor,
                    base_bits: lb,
                    multiplier: None,
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
        };
        tree.push(PropertyDecisionNode::default());
        finalize_leaf(&mut tree, &leaf_candidate, samples.candidate_predictors);
        return tree;
    }

    let max_buckets = params.max_property_values + 1;
    let mut entropy_counts = alloc::vec![0u32; histogram_size];

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

    let left_bits = compute_predictor_entropy_borrowed(
        &samples,
        0,
        abs_mid,
        split.left_predictor,
        histogram_size,
        &mut entropy_counts,
    );
    let right_bits = compute_predictor_entropy_borrowed(
        &samples,
        abs_mid,
        n,
        split.right_predictor,
        histogram_size,
        &mut entropy_counts,
    );

    drop(entropy_counts);

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

                let left_bits = compute_predictor_entropy(
                    samples,
                    candidate.start,
                    abs_mid,
                    split.left_predictor,
                    histogram_size,
                    &mut entropy_counts,
                );
                let right_bits = compute_predictor_entropy(
                    samples,
                    abs_mid,
                    candidate.end,
                    split.right_predictor,
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

impl SplitWorkspace {
    fn new(max_count: usize, histogram_size: usize, max_buckets: usize) -> Self {
        // Provable: `histogram_size` derives from `GATHER_HYBRID_UINT.encode`
        // tokens, max 239 for any u32 input (see HISTO_PADDED comment).
        debug_assert!(histogram_size <= HISTO_PADDED);
        #[cfg(test)]
        SPLIT_WS_ALLOC_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
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
) -> Option<BestSplit> {
    let count = end - start;
    if count < 2 {
        return None;
    }

    let total_num_pred = samples.num_predictors();
    let mut best: Option<BestSplit> = None;
    let mut best_bits = base_bits;

    let sample_counts_full = &samples.sample_counts;
    let sample_counts = &sample_counts_full[start..end];

    // Compute weighted total: sum of sample_counts for this node's samples.
    // After dedup, each unique sample represents `count` original samples.
    let weighted_total: u32 = sample_counts.iter().sum();

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

    for &prop_idx in &params.properties[..num_props] {
        let num_thresholds = pq.num_thresholds(prop_idx);
        if num_thresholds == 0 {
            continue;
        }

        let pq_buckets = &pq.bucket_indices[prop_idx][start..end];
        let threshold_set = &pq.threshold_sets[prop_idx];

        // Bucket range narrowing: find min/max bucket for this node's samples.
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
        if bmin == bmax {
            continue; // All samples in same bucket — no useful split
        }
        let bmin = bmin as usize;
        let bmax = bmax as usize;

        // Effective number of buckets for this node
        let local_num_buckets = bmax - bmin + 1;

        let local_num_thresholds = bmax - bmin;

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

        bucket_write_pos[..local_num_buckets].copy_from_slice(&bucket_starts[..local_num_buckets]);
        for (offset, &b) in pq_buckets.iter().enumerate() {
            let local_b = (b as usize) - bmin;
            // Store RELATIVE offset; downstream loops add `start` when
            // indexing the parent SoA arrays.
            sorted_by_bucket[bucket_write_pos[local_b]] = offset;
            bucket_write_pos[local_b] += 1;
        }

        // Initialize per-threshold best costs
        best_l_cost[..local_num_thresholds].fill(f64::MAX);
        best_r_cost[..local_num_thresholds].fill(f64::MAX);
        best_l_penalized[..local_num_thresholds].fill(f64::MAX);
        best_r_penalized[..local_num_thresholds].fill(f64::MAX);
        best_l_pred[..local_num_thresholds].fill(0);
        best_r_pred[..local_num_thresholds].fill(0);

        for pred in 0..num_pred {
            // Slice into the contiguous range [start..end) — sequential token
            // and extra-bits reads, no per-index pointer chase across the
            // whole SoA.
            let tokens = &samples.residual_tokens[pred][start..end];
            let ebits = &samples.extra_bits[pred][start..end];

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

        // Find best threshold across all predictors for this property.
        // Split decision uses RAW costs (no penalty), matching libjxl enc_ma.cc:424.
        // The penalty only influenced which predictor was chosen for each side above.
        for local_k in 0..local_num_thresholds {
            if best_l_cost[local_k] == f64::MAX || best_r_cost[local_k] == f64::MAX {
                continue;
            }

            let total = best_l_cost[local_k] + best_r_cost[local_k];

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
                });
            }
        }
    }

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
    // Issue #23 chunk 3: extend predictor-pruning lb-skip into the parallel
    // branch. A shared `AtomicU64` carries the best full cost seen by any
    // worker so far (encoded as `f64::to_bits()`). Each worker:
    //   (1) computes its extra-bits lower bound (cheap linear scan);
    //   (2) reads the shared best (relaxed); skips the full eval if
    //       `lb >= best`;
    //   (3) otherwise runs `compute_predictor_entropy` and CAS-updates the
    //       shared best with a strict-`<` discipline.
    // Skipped slots emit `f64::INFINITY`, which loses every strict-`<`
    // comparison in the post-fanout reduction below — so the lowest-index
    // tie-break behavior of the sequential path is preserved exactly.
    //
    // ## Byte-identity argument
    //
    // The sequential reduction iterates `i in 0..num_pred` with strict `<`,
    // i.e. winner = lowest index achieving the global min cost. Suppose a
    // worker `i` is skipped here. The skip implies `lb[i] >= best_seen`,
    // where `best_seen` is some full cost computed by a previously-completed
    // worker `j` (possibly `j > i`). Since `lb[i] <= full[i]`, we have
    // `full[i] >= best_seen`. Two sub-cases for the global min `m`:
    //   * `full[i] > m`: `i` was never the winner anyway, so omission is safe.
    //   * `full[i] == m`: then `best_seen <= m` AND `best_seen >= full[i] == m`,
    //     so `best_seen == m`. Worker `j` therefore evaluated and recorded
    //     cost `m`. If `j < i`, then `j` strictly beats `i` in the sequential
    //     tie-break — winner unchanged. If `j > i`, then sequentially `i`
    //     would have been visited first and won the tie-break. But the
    //     atomic only carries `m` if some worker actually computed `m`
    //     before `i` started its skip check; that worker has `full == m`
    //     and `lb <= m`. Sequentially, if any index `k < i` also has
    //     `full[k] == m`, then `pred = k` after step `k`, and `i`'s loop
    //     pass with strict-`<` would not flip it — so the winner is `k`,
    //     also `< i`, also in results (`k` is never skipped because the
    //     atomic was MAX when `k` started). If no such `k` exists, then
    //     sequentially `pred` first becomes `i`; for parallel to disagree
    //     would require some `j > i` with `full[j] == m` to have stored
    //     before `i` ran. But sequentially `j > i` does not update `pred`
    //     under strict-`<`, so the answer is still `i`. The race that
    //     skips `i` requires `full[j] < m` (impossible: `m` is the min) or
    //     a `k < i` with `full[k] == m` (handled above) — so the only way
    //     `i` is skipped is when a `k < i` already populated the atomic
    //     with `m`, and `k < i` wins the tie-break in both orderings.
    //
    // ## Cost
    //
    // Atomic ops are relaxed loads + CAS; no fences. Each worker pays at
    // most one CAS retry loop, bounded by the number of concurrent winners
    // (typically 1-2). The LB compute is ~half the bytes of the full
    // entropy compute — a worthwhile early-exit when the prune fires.
    use core::sync::atomic::{AtomicU64, Ordering};
    let best_atomic = AtomicU64::new(f64::MAX.to_bits());
    let costs: Vec<f64> = crate::parallel::parallel_map(num_pred, |pred_idx| {
        let lb = predictor_extra_bits_lower_bound(
            &samples.extra_bits[pred_idx],
            &samples.sample_counts,
            start,
            end,
        );
        let current_best = f64::from_bits(best_atomic.load(Ordering::Relaxed));
        if decide_predictor(lb, current_best) == PredictorDecision::Skip {
            return f64::INFINITY;
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
    // `f64::INFINITY` for skipped slots is `>= any finite cost`, so strict-`<`
    // ensures it never wins. At least one worker always evaluates: the very
    // first worker sees the sentinel `f64::MAX`, and any finite `lb < MAX`
    // — including `lb == 0` — passes the strict-`<` check inside
    // `decide_predictor`.
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
    collect_residuals_with_tree_offset_with_budget(
        image,
        tree,
        group_id,
        channel_offset,
        wp_params,
        None,
    )
    .expect("budget-less collect_residuals_with_tree_offset must not return AllocationLimit")
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

    let mut tokens = Vec::new();

    // Pre-allocated extended property buffer (reused per pixel)
    let num_extended_props = if needs_ref_props {
        max_tree_prop + 1
    } else {
        NUM_PROPERTIES
    };
    let mut extended_props = vec![0i32; num_extended_props];

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

        for y in 0..height {
            prev_gradient = 0;
            for x in 0..width {
                let pixel = channel.get(x, y);
                let n = Neighbors::gather(channel, x, y);

                // Compute WP prediction and property
                let (wp_pred, wp_max_error) = wp_state.predict_and_property(x, y, width, &n);

                let base_props = compute_spec_properties(
                    ch_idx as u32 + channel_offset,
                    group_id,
                    x,
                    y,
                    &n,
                    prev_gradient,
                    wp_max_error,
                );
                prev_gradient = base_props[9];

                let leaf = if needs_ref_props {
                    // Copy base properties into extended buffer
                    extended_props[..NUM_PROPERTIES].copy_from_slice(&base_props);

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
                        if base + 3 < num_extended_props {
                            extended_props[base] = v.wrapping_abs();
                            extended_props[base + 1] = v;
                            extended_props[base + 2] = v.wrapping_sub(ref_predicted).wrapping_abs();
                            extended_props[base + 3] = v.wrapping_sub(ref_predicted);
                        }
                    }
                    // Zero-fill for channels with fewer ref channels
                    let num_ref_slots = (num_extended_props - NUM_PROPERTIES) / 4;
                    for r in ref_channel_indices.len()..num_ref_slots {
                        let base = NUM_PROPERTIES + r * 4;
                        if base + 3 < num_extended_props {
                            extended_props[base] = 0;
                            extended_props[base + 1] = 0;
                            extended_props[base + 2] = 0;
                            extended_props[base + 3] = 0;
                        }
                    }

                    traverse_with_props(tree, &extended_props)
                } else {
                    // Fast path: no ref properties needed
                    traverse_with_spec_props(tree, &base_props)
                };

                // Predict using leaf's predictor
                let prediction = if leaf.predictor == Predictor::Weighted {
                    wp_pred as i32
                } else {
                    leaf.predictor.predict_from_neighbors(&n)
                };
                let residual = pixel - prediction;

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

                // Update WP error tracking
                wp_state.update_errors(pixel, x, y, width);

                // Store raw packed residual — UintCoder (HybridUint {4,2,0}) encoding
                // is applied by build_entropy_code_ans and write_tokens_ans
                tokens.push(AnsToken::new(leaf.context_id, packed));
            }
        }
    }

    Ok(tokens)
}

/// Traverse a tree using spec-matching property values (base 16 properties only).
///
/// Our tree convention: lchild = property <= splitval, rchild = property > splitval.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::channel::ModularImage;

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

        // First call: warm any state the test runtime might have lazily
        // initialised, AND warm this thread's cache so the count is stable.
        let mut samples_warm = TreeSamples::new();
        gather_samples(&mut samples_warm, &image, 0);
        let _ = compute_best_tree(&mut samples_warm, &params);

        // Snapshot, then run a real encode and measure how many NEW workspace
        // allocations happened.
        let before = SPLIT_WS_ALLOC_COUNT.load(Ordering::Relaxed);
        let mut samples = TreeSamples::new();
        gather_samples(&mut samples, &image, 0);
        let tree = compute_best_tree(&mut samples, &params);
        let after = SPLIT_WS_ALLOC_COUNT.load(Ordering::Relaxed);
        let added = after - before;

        // Sanity: this WAS a real tree-build (multiple splits).
        assert!(
            tree.len() >= 3,
            "expected non-trivial tree, got {} nodes",
            tree.len()
        );

        // With the thread-local cache, the main thread's workspace is already
        // alive from the warmup call, so the second call should allocate 0
        // workspaces on it. Parallel forks may schedule onto rayon workers
        // that haven't run a tree-learn before in this test process, so we
        // allow up to `num_threads + 1` additional allocations.
        //
        // The old code (per-fork `SplitWorkspace::new`) would have allocated
        // 16 (up to `2^max_parallel_depth`) workspaces in the recursive path
        // PLUS one for the outer loop PLUS one for the seed find_best_split
        // — so > 16 every time on the same machine.
        let cap = {
            #[cfg(feature = "parallel-tree-learning")]
            {
                // Allow num_rayon_workers (workers may not all be warm).
                rayon::current_num_threads() + 1
            }
            #[cfg(not(feature = "parallel-tree-learning"))]
            {
                1
            }
        };
        assert!(
            added <= cap,
            "thread-local workspace cache leaked: {} new SplitWorkspace::new \
             calls (cap = {}). With the cache, only the first call on each \
             worker thread should allocate.",
            added,
            cap,
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
}
