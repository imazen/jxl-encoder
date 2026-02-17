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
/// Property 15 (wp_max_error) is disabled in SPLIT_PROPERTIES pending debugging.
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
            max_nodes: 4096,
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
    residual_tokens: Vec<Vec<u32>>,
    /// Extra bits per predictor: extra_bits[predictor_idx][sample_idx].
    /// These are the HybridUint extra bits (non-token part), matching libjxl's ResidualToken.nbits.
    extra_bits: Vec<Vec<u32>>,
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
/// libjxl uses 16K/group at e7, 65K at e8, 262K at e9. Our single-group approach
/// means we use a global cap. Lower = faster tree learning with slightly less optimal trees.
pub fn max_tree_samples(effort: u8) -> usize {
    match effort {
        0..=6 => 32_768,
        7 => 65_536,
        8 => 131_072,
        _ => 524_288,
    }
}

/// Compute the stride for subsampling based on total pixel count and effort level.
///
/// Call this with the total pixel count across ALL images/groups that will
/// be gathered, then pass the stride to `gather_samples_strided`.
pub fn compute_gather_stride(total_pixels: usize, effort: u8) -> usize {
    let max_samples = max_tree_samples(effort);
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
                    samples.residual_tokens[pred_idx].push(token);
                    samples.extra_bits[pred_idx].push(num_extra);
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

/// Estimate the Shannon entropy (in bits) for a histogram of token counts.
///
/// Uses log2 with a probability floor of 1/4096, matching libjxl's ANS coding.
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
    // When only a fraction of pixels are sampled, the entropy estimates are noisier
    // but the threshold needs to be lower to allow the tree to grow appropriately.
    let required_cost = params.pixel_fraction * 0.9 + 0.1;
    let threshold = params.split_threshold * required_cost;
    let n = samples.num_samples;
    if n == 0 {
        // Empty samples: return single gradient leaf
        return vec![PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Gradient,
            context_id: 0,
            multiplier: 1,
            ..Default::default()
        }];
    }

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
            // No room for two more children, keep as leaf
            finalize_leaf(&mut tree, &candidate);
            continue;
        }

        let range = &indices[candidate.start..candidate.end];
        let count = range.len();
        if count < 2 {
            finalize_leaf(&mut tree, &candidate);
            continue;
        }

        // Find best split across all properties and thresholds
        let best_split =
            find_best_split(samples, range, histogram_size, candidate.base_bits, params);

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
                tree.push(PropertyDecisionNode::default()); // lchild placeholder
                tree.push(PropertyDecisionNode::default()); // rchild placeholder

                // Set split node
                tree[candidate.node_idx] = PropertyDecisionNode {
                    property: split.property as i32,
                    splitval: split.splitval,
                    lchild: lchild_idx,
                    rchild: rchild_idx,
                    ..Default::default()
                };

                // Push children onto stack (right first for depth-first-like behavior)
                // rchild = samples with property > splitval
                let rchild_range = &indices[abs_mid..candidate.end];
                let rchild_pred = split.right_predictor;
                let rchild_bits =
                    compute_predictor_entropy(samples, rchild_range, rchild_pred, histogram_size);
                stack.push(SplitCandidate {
                    node_idx: rchild_idx,
                    start: abs_mid,
                    end: candidate.end,
                    best_predictor: rchild_pred,
                    base_bits: rchild_bits,
                });

                // lchild = samples with property <= splitval
                let lchild_range = &indices[candidate.start..abs_mid];
                let lchild_pred = split.left_predictor;
                let lchild_bits =
                    compute_predictor_entropy(samples, lchild_range, lchild_pred, histogram_size);
                stack.push(SplitCandidate {
                    node_idx: lchild_idx,
                    start: candidate.start,
                    end: abs_mid,
                    best_predictor: lchild_pred,
                    base_bits: lchild_bits,
                });
            }
            _ => {
                // No beneficial split found
                finalize_leaf(&mut tree, &candidate);
            }
        }
    }

    // Assign sequential context IDs to leaves
    assign_sequential_contexts(&mut tree);

    // Validate tree structure (matching libjxl's ValidateTree in dec_ma.cc).
    // If any node violates property range constraints, collapse it to a leaf.
    // This prevents djxl decode failures from adversarial or edge-case inputs.
    loop {
        match validate_tree_djxl(&tree) {
            Ok(()) => break,
            Err(msg) => {
                eprintln!("Tree validation: fixing invalid node: {}", msg);
                // Extract the node index from the error message
                let node_idx = msg
                    .strip_prefix("Node ")
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<usize>().ok())
                    .expect("validate_tree_djxl error format changed");
                // Collapse this split node to a leaf with gradient predictor
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
                // Re-assign contexts and re-validate
                assign_sequential_contexts(&mut tree);
            }
        }
    }

    let num_leaves = tree.iter().filter(|n| n.property == -1).count();
    eprintln!(
        "compute_best_tree: {} samples, pf={:.3}, threshold={:.1} (base={:.0}*rc={:.3}), \
         {} nodes, {} leaves, max_nodes={}",
        n,
        params.pixel_fraction,
        threshold,
        params.split_threshold,
        required_cost,
        tree.len(),
        num_leaves,
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
}

/// Find the best (property, threshold) split for the given samples.
///
/// For each property in params.properties, try a set of threshold values
/// and pick the one that minimizes total entropy of both sides.
fn find_best_split(
    samples: &TreeSamples,
    indices: &[usize],
    histogram_size: usize,
    base_bits: f64,
    params: &TreeLearningParams,
) -> Option<BestSplit> {
    let count = indices.len();
    if count < 2 {
        return None;
    }

    let mut best: Option<BestSplit> = None;
    let mut best_bits = base_bits;

    for &prop_idx in params.properties {
        // Collect unique sorted property values for threshold candidates
        let thresholds = compute_thresholds(samples, indices, prop_idx, params.max_property_values);
        if thresholds.is_empty() {
            continue;
        }

        for &splitval in &thresholds {
            // Count tokens on each side for each predictor
            let (left_side, right_side) =
                count_split(samples, indices, prop_idx, splitval, histogram_size);

            if left_side.total == 0 || right_side.total == 0 {
                continue; // Degenerate split
            }

            // Find best predictor for each side
            let (left_pred, left_bits) = best_predictor_from_counts(&left_side);
            let (right_pred, right_bits) = best_predictor_from_counts(&right_side);

            let total = left_bits + right_bits;
            if total < best_bits {
                best_bits = total;
                best = Some(BestSplit {
                    property: prop_idx,
                    splitval,
                    left_predictor: left_pred,
                    right_predictor: right_pred,
                    total_bits: total,
                });
            }
        }
    }

    best
}

/// Compute threshold candidates for a property. Returns sorted unique values,
/// subsampled to at most `max_buckets` if there are too many.
fn compute_thresholds(
    samples: &TreeSamples,
    indices: &[usize],
    prop_idx: usize,
    max_buckets: usize,
) -> Vec<i32> {
    let props = &samples.props[prop_idx];
    let mut values: Vec<i32> = indices.iter().map(|&i| props[i]).collect();
    values.sort_unstable();
    values.dedup();

    if values.len() <= 1 {
        return Vec::new(); // Can't split on a constant property
    }

    // Use midpoints between consecutive unique values as thresholds.
    // This ensures both sides are non-empty for each threshold.
    if values.len() <= max_buckets + 1 {
        // Use all midpoints
        values
            .windows(2)
            .map(|w| {
                // Use the lower value as the splitval (property <= splitval goes left)
                w[0]
            })
            .collect()
    } else {
        // Subsample: pick evenly spaced thresholds
        let step = values.len() / max_buckets;
        values.iter().step_by(step.max(1)).copied().collect()
    }
}

/// Per-side split statistics: token histogram counts and total extra bits per predictor.
struct SplitSideCounts {
    /// Token histogram: counts[pred][token].
    counts: Vec<Vec<u32>>,
    /// Total extra bits per predictor.
    extra_bits: Vec<u64>,
    /// Number of samples on this side.
    total: u32,
}

/// Count tokens on left (prop <= splitval) and right (prop > splitval) sides
/// for each predictor.
fn count_split(
    samples: &TreeSamples,
    indices: &[usize],
    prop_idx: usize,
    splitval: i32,
    histogram_size: usize,
) -> (SplitSideCounts, SplitSideCounts) {
    let num_pred = samples.num_predictors();
    let mut left = SplitSideCounts {
        counts: vec![vec![0u32; histogram_size]; num_pred],
        extra_bits: vec![0u64; num_pred],
        total: 0,
    };
    let mut right = SplitSideCounts {
        counts: vec![vec![0u32; histogram_size]; num_pred],
        extra_bits: vec![0u64; num_pred],
        total: 0,
    };

    let props = &samples.props[prop_idx];

    for &idx in indices {
        let pval = props[idx];
        let side = if pval <= splitval {
            left.total += 1;
            &mut left
        } else {
            right.total += 1;
            &mut right
        };
        for pred in 0..num_pred {
            let tok = samples.residual_tokens[pred][idx] as usize;
            if tok < histogram_size {
                side.counts[pred][tok] += 1;
            }
            side.extra_bits[pred] += samples.extra_bits[pred][idx] as u64;
        }
    }

    (left, right)
}

/// Find the predictor with lowest total cost from pre-counted histograms.
/// Total cost = Shannon entropy of tokens + extra bits (matching libjxl).
/// Returns (best_predictor_idx, best_total_bits).
fn best_predictor_from_counts(side: &SplitSideCounts) -> (usize, f64) {
    let mut best_pred = 0;
    let mut best_bits = f64::MAX;

    for (pred_idx, hist) in side.counts.iter().enumerate() {
        let bits = estimate_bits(hist, side.total) + side.extra_bits[pred_idx] as f64;
        if bits < best_bits {
            best_bits = bits;
            best_pred = pred_idx;
        }
    }

    (best_pred, best_bits)
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
/// Returns Shannon entropy of tokens + total extra bits (matching libjxl's cost model).
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
