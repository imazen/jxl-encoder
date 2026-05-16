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

        Self {
            properties: order[..num_props].to_vec(),
            max_property_values,
            split_threshold: threshold_base,
            max_nodes: 1 << 22,
            pixel_fraction: 1.0,
            use_streaming_dedup: false,
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

    /// Pre-quantize all property values into bucket indices.
    /// This is done once before tree building, replacing per-node binary_search
    /// and threshold_set allocation with a single upfront pass.
    fn pre_quantize(&self, params: &TreeLearningParams) -> PreQuantizedProps {
        let max_buckets = params.max_property_values;
        let n = self.num_samples;
        let total_props = self.total_num_properties();
        let mut threshold_sets = vec![Vec::new(); total_props];
        let mut bucket_indices = vec![Vec::new(); total_props];

        for &prop_idx in &params.properties {
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
                bucket_indices[prop_idx] = vec![0u8; n];
                continue;
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
                    bucket_indices[prop_idx] = vec![0u8; n];
                    continue;
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
                    bucket_indices[prop_idx] = vec![0u8; n];
                    continue;
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
                    samples.residual_tokens[pred_idx].push(token as u8);
                    samples.extra_bits[pred_idx].push(num_extra as u8);
                }

                // Store base property values (0..16)
                for (prop_list, &val) in samples
                    .props
                    .iter_mut()
                    .zip(props.iter())
                    .take(NUM_PROPERTIES)
                {
                    prop_list.push(val);
                }

                // Store reference channel properties (16+)
                // For each ref channel: |ref|, ref, |ref - gradient(ref)|, ref - gradient(ref)
                // Matches libjxl context_predict.h:411-443 PrecomputeReferences
                if max_refs > 0 {
                    for (r, &ref_ch_idx) in ref_channel_indices.iter().enumerate() {
                        let ref_ch = &image.channels[ref_ch_idx];
                        let v = ref_ch.get(x, y);

                        // Compute clamped gradient prediction for reference channel
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
                        samples.props[base].push(v.wrapping_abs()); // |ref|
                        samples.props[base + 1].push(v); // ref
                        samples.props[base + 2].push(v.wrapping_sub(ref_predicted).wrapping_abs()); // |ref - gradient|
                        samples.props[base + 3].push(v.wrapping_sub(ref_predicted)); // ref - gradient
                    }
                    // Zero-pad for channels with fewer ref channels than the max
                    for r in ref_channel_indices.len()..max_refs {
                        let base = NUM_PROPERTIES + r * 4;
                        samples.props[base].push(0);
                        samples.props[base + 1].push(0);
                        samples.props[base + 2].push(0);
                        samples.props[base + 3].push(0);
                    }
                }

                samples.num_samples += 1;

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
        samples.sample_counts = vec![1; n];
        return;
    }

    let num_pred = samples.num_predictors();
    let properties = &params.properties;

    let key_len = properties.len() + 2 * num_pred;
    debug_assert!(
        key_len <= DEDUP_KEY_BYTES,
        "dedup composite key needs {} bytes, DEDUP_KEY_BYTES = {}",
        key_len,
        DEDUP_KEY_BYTES,
    );

    let mut keys: Vec<[u8; DEDUP_KEY_BYTES]> = vec![[0u8; DEDUP_KEY_BYTES]; n];
    for (sample_idx, key) in keys.iter_mut().enumerate() {
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
    }

    // Using u32 indices halves the memory footprint vs Vec<usize>; the
    // tree-learn sample cap (max_tree_samples_from_profile) tops out
    // around 4 M entries, well within u32 range.
    assert!(
        n <= u32::MAX as usize,
        "dedup_samples_packed_sort: n = {n} exceeds u32::MAX; widen key index type"
    );
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let ka = &keys[a as usize];
        let kb = &keys[b as usize];
        ka.cmp(kb)
    });

    // Walk sorted order, merge consecutive identical samples.
    let mut unique_indices: Vec<usize> = Vec::with_capacity(n / 2);
    let mut counts: Vec<u32> = Vec::with_capacity(n / 2);

    let first = order[0] as usize;
    unique_indices.push(first);
    counts.push(1);
    let mut prev_key_idx = first;
    for &curr_idx in &order[1..] {
        let curr = curr_idx as usize;
        if keys[curr] == keys[prev_key_idx] {
            *counts.last_mut().unwrap() += 1;
        } else {
            unique_indices.push(curr);
            counts.push(1);
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
    for pred in 0..num_pred {
        let old_tokens = &samples.residual_tokens[pred];
        let old_ebits = &samples.extra_bits[pred];
        let new_tokens: Vec<u8> = unique_indices.iter().map(|&i| old_tokens[i]).collect();
        let new_ebits: Vec<u8> = unique_indices.iter().map(|&i| old_ebits[i]).collect();
        samples.residual_tokens[pred] = new_tokens;
        samples.extra_bits[pred] = new_ebits;
    }
    let total_props = samples.total_num_properties();
    for prop_idx in 0..total_props {
        let old_props = &samples.props[prop_idx];
        if old_props.is_empty() {
            continue;
        }
        let new_props: Vec<i32> = unique_indices.iter().map(|&i| old_props[i]).collect();
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
        let new_bi: Vec<u8> = unique_indices.iter().map(|&i| old_bi[i]).collect();
        pq.bucket_indices[prop_idx] = new_bi;
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
        samples.sample_counts = vec![1; n];
        return;
    }

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
        if let Some(existing) = table.lookup_or_insert(&key, &unique_keys, next_idx) {
            counts[existing as usize] += 1;
        } else {
            unique_indices.push(sample_idx as u32);
            counts.push(1);
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
        let new_tokens: Vec<u8> = unique_indices.iter().map(|&i| old_tokens[i as usize]).collect();
        let new_ebits: Vec<u8> = unique_indices.iter().map(|&i| old_ebits[i as usize]).collect();
        samples.residual_tokens[pred] = new_tokens;
        samples.extra_bits[pred] = new_ebits;
    }
    let total_props = samples.total_num_properties();
    for prop_idx in 0..total_props {
        let old_props = &samples.props[prop_idx];
        if old_props.is_empty() {
            continue;
        }
        let new_props: Vec<i32> = unique_indices.iter().map(|&i| old_props[i as usize]).collect();
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

    // Sample deduplication: group samples with identical (quantized props, tokens, ebits).
    // Matching libjxl's approach, this reduces inner loop iterations on typical photos,
    // eliminating the need for the per-node eval sample cap.
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

    // Pre-allocate workspace with maximum possible sizes
    let max_buckets = params.max_property_values + 1;
    let mut workspace = SplitWorkspace::new(n, histogram_size, max_buckets);

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
        const PARALLEL_THRESHOLD: usize = 8192;
        if std::env::var("JXL_DBG_PARALLEL_TREE").is_ok() {
            eprintln!(
                "PARALLEL_TREE: n={}, max_nodes={}, root_bits={:.1}, threshold={:.1}, gate={}",
                n,
                max_nodes,
                root_bits,
                threshold,
                n >= PARALLEL_THRESHOLD && max_nodes >= 4 && root_bits > threshold
            );
        }
        // Only attempt parallel root split when there's enough work AND we
        // haven't been told to stop early (max_nodes <= 3 means root + 2
        // children is already the budget; sequential path is fine).
        if n >= PARALLEL_THRESHOLD && max_nodes >= 4 && root_bits > threshold {
            // Pop the root candidate and try its split.
            let root_candidate = stack.pop().expect("root candidate just pushed");
            let best_split = find_best_split(
                samples,
                root_candidate.start,
                root_candidate.end,
                histogram_size,
                root_candidate.base_bits,
                params,
                root_candidate.best_predictor,
                threshold,
                &pq,
                &mut workspace,
            );

            match best_split {
                Some(split) if root_candidate.base_bits - split.total_bits > threshold => {
                    let bucket_split =
                        bucket_for_splitval(&pq.threshold_sets[split.property], split.splitval);
                    let abs_mid = partition_node_in_place(
                        samples,
                        &mut pq,
                        root_candidate.start,
                        root_candidate.end,
                        split.left_count,
                        tree_learn_split::PartitionKey::Bucket {
                            prop_idx: split.property,
                            val: bucket_split as u8,
                        },
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

                    // Clone samples + pq into two owned halves at abs_mid.
                    // `samples_owned` consumes the per-side data via split_off.
                    let samples_taken = core::mem::replace(
                        samples,
                        TreeSamples::with_predictors_and_refs(
                            samples.candidate_predictors,
                            samples.num_ref_channels,
                        ),
                    );
                    let pq_taken = core::mem::replace(
                        &mut pq,
                        PreQuantizedProps {
                            threshold_sets: Vec::new(),
                            bucket_indices: Vec::new(),
                        },
                    );

                    let (left_samples, right_samples) =
                        split_tree_samples_owned(samples_taken, abs_mid);
                    let (left_pq, right_pq) = split_pq_owned(pq_taken, abs_mid);

                    if std::env::var("JXL_DBG_PARALLEL_TREE").is_ok() {
                        eprintln!(
                            "PARALLEL_TREE: root split → left={} right={} (imbalance={:.2}x)",
                            left_samples.num_samples,
                            right_samples.num_samples,
                            if left_samples.num_samples > right_samples.num_samples {
                                left_samples.num_samples as f64
                                    / right_samples.num_samples.max(1) as f64
                            } else {
                                right_samples.num_samples as f64
                                    / left_samples.num_samples.max(1) as f64
                            },
                        );
                    }

                    // Halve the node budget for each side, leaving the root
                    // node itself accounted for in the parent.
                    let per_side_budget = (max_nodes - 1) / 2;

                    // Recursive parallel decomposition. Budget = 4 means up to
                    // 2^4 = 16 leaf tasks, sufficient to saturate an 8-16 core
                    // CPU and amortize the rayon spawn cost. Deeper recursion
                    // gives diminishing returns as subtrees shrink below
                    // PARALLEL_RECURSION_FLOOR.
                    let max_parallel_depth: u32 = 4;

                    let (left_tree, right_tree) = crate::parallel::parallel_join(
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

        // Find best split across all properties and thresholds
        let best_split = crate::profile_time!("tree/find_best_split", {
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
                &mut workspace,
            )
        });

        match best_split {
            Some(split) if candidate.base_bits - split.total_bits > threshold => {
                // Perform the split: permute SoA rows in-place so that rows with
                // bucket_indices[prop][i] <= bucket_split end up in [start..mid).
                let bucket_split =
                    bucket_for_splitval(&pq.threshold_sets[split.property], split.splitval);
                let abs_mid = crate::profile_time!("tree/partition", {
                    partition_node_in_place(
                        samples,
                        &mut pq,
                        candidate.start,
                        candidate.end,
                        split.left_count,
                        tree_learn_split::PartitionKey::Bucket {
                            prop_idx: split.property,
                            val: bucket_split as u8,
                        },
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

/// Below this sample count, build the subtree sequentially (no further
/// parallel forks). Rayon task overhead (~10-50 µs per spawn + workspace
/// allocation) exceeds the savings for small subtrees.
#[cfg(feature = "parallel-tree-learning")]
const PARALLEL_RECURSION_FLOOR: usize = 16384;

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
    let mut workspace = SplitWorkspace::new(n, histogram_size, max_buckets);
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

        let best_split = find_best_split(
            samples,
            candidate.start,
            candidate.end,
            histogram_size,
            candidate.base_bits,
            params,
            candidate.best_predictor,
            threshold,
            pq,
            &mut workspace,
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

/// Split a [`TreeSamples`] into two owned halves at `mid`.
/// The original is consumed; left half holds rows `[0..mid)`, right holds
/// rows `[mid..n)`. Each side keeps the same parallel-array layout (same
/// number of predictors, properties, num_ref_channels).
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
    };

    (samples, right)
}

/// Split a [`PreQuantizedProps`] into two owned halves at `mid`.
/// `threshold_sets` is shared (cloned) — it's read-only during tree building
/// and small (16 props × ≤256 i32 = ~16 KB).
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

/// Recursive divide-and-conquer subtree builder. At each split, optionally
/// forks both child subtree builds via `parallel_join` (when the range is
/// large enough to amortize rayon task overhead AND there's parallel budget
/// left to spend).
///
/// `parallel_budget` is the number of recursion levels still permitted to
/// fork. Starts at `max_parallel_depth` and decrements at each fork. Once
/// it hits zero, descend sequentially. This bounds the total number of
/// rayon tasks to `2^max_parallel_depth` regardless of tree shape, keeping
/// the task fanout under control.
///
/// `max_nodes_budget` is the same hard cap as in [`build_subtree_sequential`].
///
/// Owned-clone strategy: at each fork, `split_off`s detach the per-side data
/// into fresh allocations. This costs O(N) memcpy per level but each level
/// halves N, so total split cost is O(N log N) (well below the O(N log² N)
/// tree-search cost).
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
    if parallel_budget == 0 || n < PARALLEL_RECURSION_FLOOR {
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
        let mut tree: Tree = Vec::new();
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

    // Find best split for the root of this subtree.
    let max_buckets = params.max_property_values + 1;
    let mut workspace = SplitWorkspace::new(n, histogram_size, max_buckets);
    let mut entropy_counts = vec![0u32; histogram_size];

    let split = match find_best_split(
        &samples,
        0,
        n,
        histogram_size,
        seed_base_bits,
        params,
        seed_predictor,
        threshold,
        &pq,
        &mut workspace,
    ) {
        Some(s) if seed_base_bits - s.total_bits > threshold => s,
        _ => {
            // No beneficial split — single-leaf subtree.
            let mut tree: Tree = Vec::new();
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

    // Partition in-place to separate left/right.
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

    // Recompute child base bits using the split's predictors.
    let left_bits = compute_predictor_entropy(
        &samples,
        0,
        abs_mid,
        split.left_predictor,
        histogram_size,
        &mut entropy_counts,
    );
    let right_bits = compute_predictor_entropy(
        &samples,
        abs_mid,
        n,
        split.right_predictor,
        histogram_size,
        &mut entropy_counts,
    );

    // Free the workspace + entropy buffer before the split_off allocations.
    drop(workspace);
    drop(entropy_counts);

    // Split data into per-side owned halves.
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
    // before recursing into the larger side. We still hand both sides to
    // parallel_join when both are big enough to amortize the spawn.
    let left_size = left_samples.num_samples;
    let right_size = right_samples.num_samples;
    let both_big_enough =
        left_size >= PARALLEL_RECURSION_FLOOR && right_size >= PARALLEL_RECURSION_FLOOR;

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
        // The larger side may still benefit from internal parallel recursion;
        // pass `next_parallel_budget` through.
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
    let mut tree: Tree = Vec::new();
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
    let mut workspace = SplitWorkspace::new(n, histogram_size, max_buckets);

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
            &mut workspace,
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

impl SplitWorkspace {
    fn new(max_count: usize, histogram_size: usize, max_buckets: usize) -> Self {
        // Provable: `histogram_size` derives from `GATHER_HYBRID_UINT.encode`
        // tokens, max 239 for any u32 input (see HISTO_PADDED comment).
        debug_assert!(histogram_size <= HISTO_PADDED);
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

            // Build initial right histogram (all local buckets on the right side)
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

    for pred_idx in 0..num_pred {
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
    debug_assert!(left_count <= end - start);
    let num_samples = samples.num_samples;
    let mut view = tree_learn_split::SplittableSamples {
        residual_tokens: &mut samples.residual_tokens,
        extra_bits: &mut samples.extra_bits,
        props: &mut samples.props,
        bucket_indices: &mut pq.bucket_indices,
        sample_counts: &mut samples.sample_counts,
        len: num_samples,
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
}
