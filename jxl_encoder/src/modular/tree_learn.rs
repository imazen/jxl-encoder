// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Content-adaptive MA tree learning for modular encoding.
//!
//! Replaces the fixed single-leaf gradient tree with a learned multi-leaf tree
//! that assigns optimal predictors and entropy contexts per image region.
//! Port of libjxl's `FindBestSplit` algorithm from `enc_ma.cc`.

use super::channel::{Channel, ModularImage};
use super::predictor::{Neighbors, Predictor, WeightedPredictorState, pack_signed};
use super::tree::{PropertyDecisionNode, Tree, assign_sequential_contexts, validate_tree_djxl};
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

/// Non-squeeze lossless property order, matching libjxl's enc_modular.cc.
/// Properties are ordered by likelihood of being useful for non-squeeze residuals.
/// GroupId (1) is removed when num_groups < 30 (which is always true for us currently).
const PROP_ORDER_NO_SQUEEZE: &[usize] = &[
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

/// Parameters for tree learning, effort-dependent.
///
/// Matches libjxl's enc_modular.cc speed tier configuration:
/// - Squirrel (e7): first 7 properties, max 48 property values, threshold 131
/// - Kitten (e8): first 10 properties, max 96 property values, threshold 89
/// - Tortoise (e9/e10): all properties, max 256 property values, threshold 75
pub struct TreeLearningParams {
    /// Properties to consider for splits, in priority order.
    pub properties: &'static [usize],
    /// Maximum number of quantized threshold buckets per property.
    pub max_property_values: usize,
    /// Base split threshold: scaled by `pixel_fraction * 0.9 + 0.1` to get effective threshold.
    /// A split must save at least `effective_threshold` bits to be accepted.
    pub split_threshold: f64,
    /// Maximum tree nodes (libjxl uses 1<<22, effectively unlimited).
    pub max_nodes: usize,
    /// Fraction of pixels actually sampled (num_samples / total_pixels).
    /// Used to scale the split threshold: effective = threshold * (fraction * 0.9 + 0.1).
    /// Matches libjxl's `required_cost = pixel_fraction * 0.9 + 0.1` in LearnTree().
    /// Set to 1.0 if all pixels are sampled (no subsampling).
    pub pixel_fraction: f64,
}

impl TreeLearningParams {
    /// Create tree learning parameters for the given effort level.
    ///
    /// Matches libjxl's speed tier mapping (effort = 10 - speed_tier):
    /// - effort 5 (Hare, speed_tier=5): 4 properties, 24 buckets, threshold 145
    /// - effort 6 (Wombat, speed_tier=4): 5 properties, 32 buckets, threshold 131
    /// - effort 7 (Squirrel, speed_tier=3): 7 properties, 48 buckets, threshold 117
    /// - effort 8 (Kitten, speed_tier=2): 10 properties, 96 buckets, threshold 103
    /// - effort 9-10 (Tortoise/Glacier): all properties, 256 buckets, threshold 89/75
    pub fn for_effort(effort: u8) -> Self {
        let order = PROP_ORDER_NO_SQUEEZE;
        // speed_tier = 10 - effort (libjxl convention)
        let speed_tier = 10u8.saturating_sub(effort);
        let (num_props, max_property_values) = match effort {
            0..=4 => (3, 16),        // Below Hare
            5 => (4, 24),            // Hare
            6 => (5, 32),            // Wombat
            7 => (7, 48),            // Squirrel
            8 => (10, 96),           // Kitten
            _ => (order.len(), 256), // Tortoise/Glacier
        };

        // libjxl formula: threshold = 75 + 14 * speed_tier
        let threshold_base = 75.0 + 14.0 * speed_tier as f64;

        let num_props = num_props.min(order.len());

        Self {
            properties: &order[..num_props],
            max_property_values,
            split_threshold: threshold_base,
            max_nodes: 8192,
            pixel_fraction: 1.0,
        }
    }

    /// Set the pixel fraction (num_samples / total_pixels) for threshold scaling.
    /// This matches libjxl's `required_cost = pixel_fraction * 0.9 + 0.1`.
    #[must_use]
    pub fn with_pixel_fraction(mut self, fraction: f64) -> Self {
        self.pixel_fraction = fraction.clamp(0.0, 1.0);
        self
    }
}

/// Collected samples for tree learning.
pub struct TreeSamples {
    /// Number of samples collected.
    pub num_samples: usize,
    /// Residual token per predictor: residual_tokens[predictor_idx][sample_idx].
    /// Tokens fit in u8 (max ~55 for HybridUint {4,2,0} on 8-bit data).
    residual_tokens: Vec<Vec<u8>>,
    /// Extra bits per predictor: extra_bits[predictor_idx][sample_idx].
    /// These are the HybridUint extra bits (non-token part), matching libjxl's ResidualToken.nbits.
    /// Fits in u8 (max ~14 bits for 8-bit image residuals).
    extra_bits: Vec<Vec<u8>>,
    /// Spec-matching property values: props[property_idx][sample_idx].
    /// These are the actual (unquantized) property values.
    props: Vec<Vec<i32>>,
}

impl Default for TreeSamples {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeSamples {
    /// Creates an empty TreeSamples structure.
    pub fn new() -> Self {
        let num_predictors = CANDIDATE_PREDICTORS.len();
        Self {
            num_samples: 0,
            residual_tokens: vec![Vec::new(); num_predictors],
            extra_bits: vec![Vec::new(); num_predictors],
            props: vec![Vec::new(); NUM_PROPERTIES],
        }
    }

    /// Returns the number of candidate predictors.
    pub fn num_predictors(&self) -> usize {
        CANDIDATE_PREDICTORS.len()
    }

    /// Pre-quantize all property values into bucket indices.
    /// This is done once before tree building, replacing per-node binary_search
    /// and threshold_set allocation with a single upfront pass.
    fn pre_quantize(&self, params: &TreeLearningParams) -> PreQuantizedProps {
        let max_buckets = params.max_property_values;
        let n = self.num_samples;
        let mut threshold_sets = vec![Vec::new(); NUM_PROPERTIES];
        let mut bucket_indices = vec![Vec::new(); NUM_PROPERTIES];

        for &prop_idx in params.properties {
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
                    sample_vals.iter().step_by(step.max(1)).copied().collect()
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
/// computed by `compute_gather_stride` to avoid O(n^2) tree learning time.
#[cfg(test)]
pub fn gather_samples(samples: &mut TreeSamples, image: &ModularImage, group_id: u32) {
    gather_samples_strided(samples, image, group_id, 0, 1);
}

/// Gather samples with stride-based subsampling.
///
/// When `stride > 1`, only every `stride`-th pixel in scan order is sampled.
/// Use `compute_gather_stride` to determine the appropriate stride.
pub fn gather_samples_strided(
    samples: &mut TreeSamples,
    image: &ModularImage,
    group_id: u32,
    channel_offset: u32,
    stride: usize,
) {
    for (ch_idx, channel) in image.channels.iter().enumerate() {
        gather_channel_samples(
            samples,
            channel,
            ch_idx as u32 + channel_offset,
            group_id,
            stride,
        );
    }
}

/// Maximum number of samples to gather for tree learning, per effort level.
///
/// libjxl uses `nb_repeats = 0.5` at all efforts >= 5, meaning 50% of pixels per channel
/// are sampled. For a 1024x1024 RGB image, that's ~1.57M samples from 3.1M total pixels.
///
/// We use a fraction-based approach: at effort 7+, sample 50% of total pixels (matching
/// libjxl's nb_repeats=0.5). At lower efforts, use smaller fixed caps for speed.
pub fn max_tree_samples(effort: u8, total_pixels: usize) -> usize {
    match effort {
        0..=4 => 32_768,
        5..=6 => 65_536,
        // effort 7+: match libjxl's nb_repeats=0.5 (50% of pixels)
        _ => (total_pixels / 2).max(65_536),
    }
}

/// Compute the stride for subsampling based on total pixel count and effort level.
///
/// Call this with the total pixel count across ALL images/groups that will
/// be gathered, then pass the stride to `gather_samples_strided`.
pub fn compute_gather_stride(total_pixels: usize, effort: u8) -> usize {
    let max_samples = max_tree_samples(effort, total_pixels);
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
fn gather_channel_samples(
    samples: &mut TreeSamples,
    channel: &Channel,
    channel_idx: u32,
    group_id: u32,
    stride: usize,
) {
    let width = channel.width();
    let height = channel.height();
    if width == 0 || height == 0 {
        return;
    }

    // WP state for computing weighted predictions and property 15
    let mut wp_state = WeightedPredictorState::with_defaults(width);

    // prev_gradient tracks the gradient from the previous pixel in scan order.
    // Property 8 = W - prev_gradient. At the start of each row, prev_gradient = 0.
    let mut prev_gradient: i32;

    // Counter for subsampling: only gather when counter == 0
    let mut subsample_counter: usize = 0;

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
                for (pred_idx, &predictor) in CANDIDATE_PREDICTORS.iter().enumerate() {
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

                // Store property values
                for (prop_list, &val) in samples
                    .props
                    .iter_mut()
                    .zip(props.iter())
                    .take(NUM_PROPERTIES)
                {
                    prop_list.push(val);
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
}

/// Size of the precomputed n*log2(n) lookup table.
/// 65536 entries × 8 bytes = 512KB, fits comfortably in L2 cache.
/// Covers all per-symbol counts in tree learning for typical images.
const NLOG2N_TABLE_SIZE: usize = 65536;

/// Build the nlog2n lookup table. Called once at the start of tree learning.
fn build_nlog2n_table() -> Vec<f64> {
    let mut table = vec![0.0f64; NLOG2N_TABLE_SIZE];
    for (n, entry) in table.iter_mut().enumerate().skip(1) {
        let nf = n as f64;
        *entry = nf * nf.log2();
    }
    table
}

/// Compute n * log2(n), using a lookup table for small values.
#[inline(always)]
fn nlog2n(table: &[f64], n: u32) -> f64 {
    let idx = n as usize;
    if idx < table.len() {
        // SAFETY: bounds checked by the if condition above
        table[idx]
    } else {
        let nf = n as f64;
        nf * nf.log2()
    }
}

/// Uses log2 with a probability floor of 1/4096, matching libjxl's ANS coding.
/// Used for parent node cost estimation (consistent with old code's cost model).
#[inline]
pub fn estimate_bits(counts: &[u32], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    // Floor probability at 1/4096 (ANS precision is 12 bits)
    let min_prob = 1.0 / 4096.0;
    let mut bits = 0.0;
    for &c in counts {
        if c > 0 {
            let p = (c as f64 / total_f).max(min_prob);
            bits -= c as f64 * p.log2();
        }
    }
    bits
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
    let max_nodes = params.max_nodes;
    // Scale threshold by pixel_fraction, matching libjxl's required_cost formula.
    let required_cost = params.pixel_fraction * 0.9 + 0.1;
    let threshold = params.split_threshold * required_cost;
    let n = samples.num_samples;
    if n == 0 {
        return vec![PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Gradient,
            context_id: 0,
            multiplier: 1,
            ..Default::default()
        }];
    }

    // Build nlog2n lookup table once (65536 entries = 512KB, fits L2)
    let nlog2n_table = build_nlog2n_table();

    // Pre-quantize all properties globally (replaces per-node binary_search)
    let pq = samples.pre_quantize(params);

    // Working index array: we partition this instead of moving actual data.
    let mut indices: Vec<usize> = (0..n).collect();

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

    // Start with root node
    let root_predictor = find_best_predictor(samples, &indices[..n], histogram_size);
    let root_bits =
        compute_predictor_entropy(samples, &indices[..n], root_predictor, histogram_size);

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
    });

    while let Some(candidate) = stack.pop() {
        if tree.len() + 2 > max_nodes {
            finalize_leaf(&mut tree, &candidate);
            continue;
        }

        let count = candidate.end - candidate.start;
        if count < 2 {
            finalize_leaf(&mut tree, &candidate);
            continue;
        }

        // Early termination gate: if base_bits is already below threshold,
        // no split can save enough bits. Matches libjxl enc_ma.cc:304.
        if candidate.base_bits <= threshold {
            finalize_leaf(&mut tree, &candidate);
            continue;
        }

        // Find best split across all properties and thresholds
        let best_split = find_best_split(
            samples,
            &indices[candidate.start..candidate.end],
            histogram_size,
            candidate.base_bits,
            params,
            candidate.best_predictor,
            threshold,
            &nlog2n_table,
            &pq,
        );

        match best_split {
            Some(split) if candidate.base_bits - split.total_bits > threshold => {
                // Perform the split: partition indices
                let mid = partition_indices(
                    &mut indices[candidate.start..candidate.end],
                    samples,
                    split.property,
                    split.splitval,
                );
                let abs_mid = candidate.start + mid;

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

                // Push children with cached costs from find_best_split
                // (avoids redundant compute_predictor_entropy calls)
                stack.push(SplitCandidate {
                    node_idx: rchild_idx,
                    start: abs_mid,
                    end: candidate.end,
                    best_predictor: split.right_predictor,
                    base_bits: split.right_bits,
                });

                stack.push(SplitCandidate {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: split.left_predictor,
                    base_bits: split.left_bits,
                });
            }
            _ => {
                finalize_leaf(&mut tree, &candidate);
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

    tree
}

/// Make a tree node into a leaf with the given predictor.
fn finalize_leaf(tree: &mut Tree, candidate: &SplitCandidate) {
    tree[candidate.node_idx] = PropertyDecisionNode {
        property: -1,
        predictor: CANDIDATE_PREDICTORS[candidate.best_predictor],
        predictor_offset: 0,
        multiplier: 1,
        context_id: 0, // Will be reassigned by assign_sequential_contexts
        ..Default::default()
    };
}

/// Result of finding the best split for a node.
struct BestSplit {
    property: usize,
    splitval: i32,
    left_predictor: usize,
    right_predictor: usize,
    total_bits: f64,
    /// Cached cost for left child (avoids redundant compute_predictor_entropy).
    left_bits: f64,
    /// Cached cost for right child (avoids redundant compute_predictor_entropy).
    right_bits: f64,
}

/// Find the best (property, threshold) split for the given samples.
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
#[allow(clippy::too_many_arguments)]
fn find_best_split(
    samples: &TreeSamples,
    indices: &[usize],
    histogram_size: usize,
    base_bits: f64,
    params: &TreeLearningParams,
    parent_predictor: usize,
    threshold: f64,
    nlog2n_table: &[f64],
    pq: &PreQuantizedProps,
) -> Option<BestSplit> {
    let count = indices.len();
    if count < 2 {
        return None;
    }

    let num_pred = samples.num_predictors();
    let mut best: Option<BestSplit> = None;
    let mut best_bits = base_bits;

    // Predictor change penalty matching libjxl's enc_ma.cc:303
    let change_pred_penalty = 800.0 / (100.0 + threshold);

    let weighted_idx = CANDIDATE_PREDICTORS
        .iter()
        .position(|&p| p == Predictor::Weighted)
        .unwrap_or(usize::MAX);

    // Compute effective histogram size for this node: max token + 1 across all predictors.
    // Deep tree nodes often have much smaller max tokens than the root.
    let mut effective_histo = 0usize;
    for pred in 0..num_pred {
        let tokens = &samples.residual_tokens[pred];
        for &idx in indices {
            let tok = tokens[idx] as usize;
            if tok >= effective_histo {
                effective_histo = tok + 1;
            }
        }
    }
    let effective_histo = effective_histo.min(histogram_size);
    if effective_histo == 0 {
        return None;
    }

    // Reusable buffers for the bucket sweep (allocated once, reused per property).
    let mut count_increase: Vec<u32> = Vec::new();
    let mut extra_bits_increase: Vec<u64> = Vec::new();
    let mut bucket_counts: Vec<u32> = Vec::new();

    let mut right_counts = vec![0u32; effective_histo];
    let mut left_counts = vec![0u32; effective_histo];

    let mut best_l_cost: Vec<f64> = Vec::new();
    let mut best_r_cost: Vec<f64> = Vec::new();
    let mut best_l_pred: Vec<usize> = Vec::new();
    let mut best_r_pred: Vec<usize> = Vec::new();

    // Counting sort buffers
    let mut sorted_by_bucket: Vec<usize> = vec![0; count];
    let mut bucket_starts: Vec<usize> = Vec::new();
    let mut bucket_write_pos: Vec<usize> = Vec::new();

    for &prop_idx in params.properties {
        let num_thresholds = pq.num_thresholds(prop_idx);
        if num_thresholds == 0 {
            continue;
        }

        let pq_buckets = &pq.bucket_indices[prop_idx];
        let threshold_set = &pq.threshold_sets[prop_idx];

        // Bucket range narrowing: find min/max bucket for this node's samples
        let mut bmin: u8 = u8::MAX;
        let mut bmax: u8 = 0;
        for &idx in indices {
            let b = pq_buckets[idx];
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

        // Resize buffers
        let ci_size = local_num_buckets * effective_histo;
        if count_increase.len() < ci_size {
            count_increase.resize(ci_size, 0);
        }
        if extra_bits_increase.len() < local_num_buckets {
            extra_bits_increase.resize(local_num_buckets, 0);
        }
        if bucket_counts.len() < local_num_buckets {
            bucket_counts.resize(local_num_buckets, 0);
        }

        // Number of thresholds within our bucket range
        // Bucket bmin..bmax spans thresholds bmin..bmax-1 (threshold k separates bucket k from k+1)
        // But we need thresholds that are BETWEEN bmin and bmax
        let local_num_thresholds = bmax - bmin; // thresholds at positions bmin..bmax-1

        if best_l_cost.len() < local_num_thresholds {
            best_l_cost.resize(local_num_thresholds, f64::MAX);
            best_r_cost.resize(local_num_thresholds, f64::MAX);
            best_l_pred.resize(local_num_thresholds, 0);
            best_r_pred.resize(local_num_thresholds, 0);
        }

        // Counting sort: group indices by bucket
        bucket_counts[..local_num_buckets].fill(0);
        for &idx in indices {
            let b = (pq_buckets[idx] as usize) - bmin;
            bucket_counts[b] += 1;
        }

        if bucket_starts.len() < local_num_buckets + 1 {
            bucket_starts.resize(local_num_buckets + 1, 0);
        }
        bucket_starts[0] = 0;
        for b in 0..local_num_buckets {
            bucket_starts[b + 1] = bucket_starts[b] + bucket_counts[b] as usize;
        }

        if bucket_write_pos.len() < local_num_buckets {
            bucket_write_pos.resize(local_num_buckets, 0);
        }
        bucket_write_pos[..local_num_buckets].copy_from_slice(&bucket_starts[..local_num_buckets]);
        for &idx in indices {
            let b = (pq_buckets[idx] as usize) - bmin;
            sorted_by_bucket[bucket_write_pos[b]] = idx;
            bucket_write_pos[b] += 1;
        }

        // Initialize per-threshold best costs
        best_l_cost[..local_num_thresholds].fill(f64::MAX);
        best_r_cost[..local_num_thresholds].fill(f64::MAX);
        best_l_pred[..local_num_thresholds].fill(0);
        best_r_pred[..local_num_thresholds].fill(0);

        for pred in 0..num_pred {
            let tokens = &samples.residual_tokens[pred];
            let ebits = &samples.extra_bits[pred];

            // Collect count_increase by iterating bucket groups
            count_increase[..ci_size].fill(0);
            extra_bits_increase[..local_num_buckets].fill(0);

            for local_bucket in 0..local_num_buckets {
                let start = bucket_starts[local_bucket];
                let end = bucket_starts[local_bucket + 1];
                let ci_base = local_bucket * effective_histo;
                let ci_slice = &mut count_increase[ci_base..ci_base + effective_histo];
                let mut eb_sum: u64 = 0;
                for &idx in &sorted_by_bucket[start..end] {
                    let tok = tokens[idx] as usize;
                    ci_slice[tok] += 1;
                    eb_sum += ebits[idx] as u64;
                }
                extra_bits_increase[local_bucket] = eb_sum;
            }

            // Build initial right histogram (all local buckets on the right side)
            right_counts[..effective_histo].fill(0);
            let mut right_extra: u64 = 0;
            let mut right_total: u32 = count as u32;
            for (local_bucket, &eb) in extra_bits_increase[..local_num_buckets]
                .iter()
                .enumerate()
            {
                let ci_base = local_bucket * effective_histo;
                let ci_row = &count_increase[ci_base..ci_base + effective_histo];
                for (rc, &ci) in right_counts[..effective_histo]
                    .iter_mut()
                    .zip(ci_row.iter())
                {
                    *rc += ci;
                }
                right_extra += eb;
            }

            // Compute initial nlog2n sum for right side
            let mut right_nlogn_sum: f64 = 0.0;
            for &c in &right_counts[..effective_histo] {
                right_nlogn_sum += nlog2n(nlog2n_table, c);
            }

            left_counts[..effective_histo].fill(0);
            let mut left_extra: u64 = 0;
            let mut left_total: u32 = 0;
            let mut left_nlogn_sum: f64 = 0.0;

            // Sweep through local buckets, moving each from right to left
            for local_k in 0..local_num_thresholds {
                let bc = bucket_counts[local_k];
                if bc == 0 {
                    continue;
                }

                // Move bucket from right to left using zip iterators
                let ci_base = local_k * effective_histo;
                let ci_row = &count_increase[ci_base..ci_base + effective_histo];
                for ((ci, left), right) in ci_row
                    .iter()
                    .zip(left_counts[..effective_histo].iter_mut())
                    .zip(right_counts[..effective_histo].iter_mut())
                {
                    let delta = *ci;
                    if delta > 0 {
                        let old_l = *left;
                        let new_l = old_l + delta;
                        left_nlogn_sum += nlog2n(nlog2n_table, new_l) - nlog2n(nlog2n_table, old_l);
                        *left = new_l;

                        let old_r = *right;
                        let new_r = old_r - delta;
                        right_nlogn_sum +=
                            nlog2n(nlog2n_table, new_r) - nlog2n(nlog2n_table, old_r);
                        *right = new_r;
                    }
                }
                left_extra += extra_bits_increase[local_k];
                right_extra -= extra_bits_increase[local_k];
                left_total += bc;
                right_total -= bc;

                if left_total == 0 || right_total == 0 {
                    continue;
                }

                let l_bits = nlog2n(nlog2n_table, left_total) - left_nlogn_sum + left_extra as f64;
                let r_bits =
                    nlog2n(nlog2n_table, right_total) - right_nlogn_sum + right_extra as f64;

                if l_bits < best_l_cost[local_k] {
                    best_l_cost[local_k] = l_bits;
                    best_l_pred[local_k] = pred;
                }
                if r_bits < best_r_cost[local_k] {
                    best_r_cost[local_k] = r_bits;
                    best_r_pred[local_k] = pred;
                }
            }
        }

        // Find best threshold across all predictors for this property
        for local_k in 0..local_num_thresholds {
            if best_l_cost[local_k] == f64::MAX || best_r_cost[local_k] == f64::MAX {
                continue;
            }

            let mut total = best_l_cost[local_k] + best_r_cost[local_k];

            if best_l_pred[local_k] != parent_predictor && parent_predictor != weighted_idx {
                total += change_pred_penalty;
            }
            if best_r_pred[local_k] != parent_predictor && parent_predictor != weighted_idx {
                total += change_pred_penalty;
            }

            if total < best_bits {
                best_bits = total;
                // Map local_k back to global threshold index: bmin + local_k
                let global_k = bmin + local_k;
                best = Some(BestSplit {
                    property: prop_idx,
                    splitval: threshold_set[global_k],
                    left_predictor: best_l_pred[local_k],
                    right_predictor: best_r_pred[local_k],
                    total_bits: total,
                    left_bits: best_l_cost[local_k],
                    right_bits: best_r_cost[local_k],
                });
            }
        }
    }

    best
}

/// Find the best predictor for the given sample indices.
fn find_best_predictor(samples: &TreeSamples, indices: &[usize], histogram_size: usize) -> usize {
    let num_pred = samples.num_predictors();
    let mut best_pred = 0;
    let mut best_bits = f64::MAX;

    for pred_idx in 0..num_pred {
        let bits = compute_predictor_entropy(samples, indices, pred_idx, histogram_size);
        if bits < best_bits {
            best_bits = bits;
            best_pred = pred_idx;
        }
    }

    best_pred
}

/// Compute total cost for a given predictor's residuals over the indexed samples.
/// Returns Shannon entropy of tokens + total extra bits.
/// Uses estimate_bits (probability-floor formula) for consistency with the split
/// threshold comparison — the sweep in find_best_split uses nlog2n, but the
/// parent's base_bits must use the same formula as the old code to avoid
/// inflated base_bits that accept too many splits (10x tree size regression).
fn compute_predictor_entropy(
    samples: &TreeSamples,
    indices: &[usize],
    predictor_idx: usize,
    histogram_size: usize,
) -> f64 {
    let tokens = &samples.residual_tokens[predictor_idx];
    let ebits = &samples.extra_bits[predictor_idx];
    let mut counts = vec![0u32; histogram_size];
    let mut total = 0u32;
    let mut tot_extra: u64 = 0;

    for &idx in indices {
        let tok = tokens[idx] as usize;
        if tok < histogram_size {
            counts[tok] += 1;
            total += 1;
        }
        tot_extra += ebits[idx] as u64;
    }

    estimate_bits(&counts, total) + tot_extra as f64
}

/// Partition indices in-place so that indices with property <= splitval come first.
/// Returns the number of indices on the left (property <= splitval) side.
fn partition_indices(
    indices: &mut [usize],
    samples: &TreeSamples,
    prop_idx: usize,
    splitval: i32,
) -> usize {
    let props = &samples.props[prop_idx];
    let mut left = 0;
    let mut right = indices.len();

    while left < right {
        if props[indices[left]] <= splitval {
            left += 1;
        } else {
            right -= 1;
            indices.swap(left, right);
        }
    }

    left
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
) -> Vec<crate::entropy_coding::token::Token> {
    collect_residuals_with_tree_offset(image, tree, group_id, 0)
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
) -> Vec<crate::entropy_coding::token::Token> {
    use crate::entropy_coding::token::Token as AnsToken;

    let mut tokens = Vec::new();

    for (ch_idx, channel) in image.channels.iter().enumerate() {
        let width = channel.width();
        let height = channel.height();
        if width == 0 || height == 0 {
            continue;
        }

        let mut wp_state = WeightedPredictorState::with_defaults(width);
        let mut prev_gradient: i32;

        for y in 0..height {
            prev_gradient = 0;
            for x in 0..width {
                let pixel = channel.get(x, y);
                let n = Neighbors::gather(channel, x, y);

                // Compute WP prediction and property
                let (wp_pred, wp_max_error) = wp_state.predict_and_property(x, y, width, &n);

                let props = compute_spec_properties(
                    ch_idx as u32 + channel_offset,
                    group_id,
                    x,
                    y,
                    &n,
                    prev_gradient,
                    wp_max_error,
                );
                prev_gradient = props[9];

                // Traverse tree to find leaf
                let leaf = traverse_with_spec_props(tree, &props);

                // Predict using leaf's predictor
                let prediction = if leaf.predictor == Predictor::Weighted {
                    wp_pred as i32
                } else {
                    leaf.predictor.predict_from_neighbors(&n)
                };
                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                // Update WP error tracking
                wp_state.update_errors(pixel, x, y, width);

                // Store raw packed residual — UintCoder (HybridUint {4,2,0}) encoding
                // is applied by build_entropy_code_ans and write_tokens_ans
                tokens.push(AnsToken::new(leaf.context_id, packed));
            }
        }
    }

    tokens
}

/// Traverse a tree using spec-matching property values.
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

    #[test]
    fn test_compute_best_tree_two_channels() {
        // 2-channel image: ch0=constant 100, ch1=gradient ramp
        // Tree should split on channel property
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        };

        // Channel 0: constant
        let mut ch0 = Channel::new(8, 8).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                ch0.set(x, y, 100);
            }
        }
        image.channels.push(ch0);

        // Channel 1: ramp
        let mut ch1 = Channel::new(8, 8).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                ch1.set(x, y, (x * 30 + y * 20) as i32);
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
        let tokens = collect_residuals_with_tree(&image, &tree, 0);

        assert_eq!(tokens.len(), 16);
        // All tokens should have context 0
        for t in &tokens {
            assert_eq!(t.context, 0);
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
    fn test_partition_indices() {
        let image = ModularImage::from_gray8(&[0u8; 16], 4, 4).unwrap();
        let mut samples = TreeSamples::new();
        gather_samples(&mut samples, &image, 0);

        // Partition on X (property 3) at splitval=1
        // Pixels with x<=1 should be on left, x>1 on right
        let mut indices: Vec<usize> = (0..samples.num_samples).collect();
        let mid = partition_indices(&mut indices, &samples, 3, 1);

        // 4x4 image: x=0,1 → 8 pixels left, x=2,3 → 8 pixels right
        assert_eq!(mid, 8);
        for &i in &indices[..mid] {
            assert!(samples.props[3][i] <= 1);
        }
        for &i in &indices[mid..] {
            assert!(samples.props[3][i] > 1);
        }
    }
}
