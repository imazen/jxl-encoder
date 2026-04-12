// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

// Module contains experimental/WIP code with some unused items and complex types.
// Allow various clippy warnings that don't affect correctness.
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::let_and_return)]

//! DC coefficient tree learning for VarDCT encoding.
//!
//! Learns an optimal context tree for DC coding based on image content,
//! replacing the fixed GRADIENT_CONTEXT_LUT with a data-driven tree.
//! This can provide 0.3-1.0% compression improvement on DC stream.
//!
//! Port of libjxl's DC tree learning from `enc_modular.cc`.

use super::common::pack_signed;
use super::dc_coding::clamped_gradient;

/// Number of properties used in DC tree learning.
/// Must match jxl-rs decoder's property buffer layout:
/// - 0: channel (static, set by caller)
/// - 1: group_id/stream (static, typically 0 for DC)
/// - 2: y position
/// - 3: x position
/// - 4: |top|
/// - 5: |left|
/// - 6: top
/// - 7: left
/// - 8: local gradient (left - prev_left, maintained across row)
/// - 9: gradient (left + top - topleft) ← PRIMARY SPLIT PROPERTY
/// - 10: left - topleft (FFV1)
/// - 11: topleft - top (FFV1)
/// - 12: top - topright (FFV1)
/// - 13: top - toptop (FFV1)
/// - 14: left - leftleft (FFV1)
const NUM_DC_PROPERTIES: usize = 15;

/// Properties to consider for splits.
/// Property 9 (gradient) is the most effective for DC coding.
const SPLIT_PROPERTIES: &[usize] = &[
    9,  // gradient (left + top - topleft) - most important
    4,  // |top|
    5,  // |left|
    6,  // top
    7,  // left
    10, // left - topleft (FFV1)
];

/// Maximum tree depth to prevent overfitting.
const MAX_TREE_DEPTH: usize = 8;

/// Minimum samples per leaf to prevent overfitting.
const MIN_SAMPLES_PER_LEAF: usize = 64;

/// HybridUint config for sample gathering: {4, 1, 2}.
const GATHER_SPLIT: u32 = 16; // 1 << 4
const GATHER_MSB_IN_TOKEN: u32 = 1;
const GATHER_LSB_IN_TOKEN: u32 = 2;

/// Encode a value using HybridUint config for gathering.
#[inline]
fn encode_hybrid_uint(value: u32) -> u32 {
    if value < GATHER_SPLIT {
        value
    } else {
        let n = 32 - value.leading_zeros(); // floor_log2(value) + 1
        let n_minus_split_exp = n - 4 - 1; // n - split_exponent - 1
        let token = GATHER_SPLIT + n_minus_split_exp * (GATHER_MSB_IN_TOKEN + GATHER_LSB_IN_TOKEN);
        token
    }
}

/// Collected samples for DC tree learning.
pub struct DcTreeSamples {
    /// Number of samples collected.
    pub num_samples: usize,
    /// Residual tokens (packed residuals converted to HybridUint tokens).
    residual_tokens: Vec<u32>,
    /// Property values: props[property_idx][sample_idx].
    props: Vec<Vec<i32>>,
}

impl Default for DcTreeSamples {
    fn default() -> Self {
        Self::new()
    }
}

impl DcTreeSamples {
    /// Creates an empty DcTreeSamples structure.
    pub fn new() -> Self {
        Self {
            num_samples: 0,
            residual_tokens: Vec::new(),
            props: vec![Vec::new(); NUM_DC_PROPERTIES],
        }
    }

    /// Add a sample with its properties and residual.
    #[inline]
    pub fn add_sample(&mut self, residual: i32, props: [i32; NUM_DC_PROPERTIES]) {
        let packed = pack_signed(residual);
        let token = encode_hybrid_uint(packed);
        self.residual_tokens.push(token);

        for (i, &p) in props.iter().enumerate() {
            self.props[i].push(p);
        }
        self.num_samples += 1;
    }
}

/// Compute properties for a DC value given its neighbors.
#[inline]
/// Compute DC properties matching jxl-rs decoder's property buffer layout.
///
/// # Arguments
/// * `channel_idx` - Channel index in encoding order (0=Y, 1=X, 2=B after reorder)
/// * `x` - X position in block coordinates
/// * `y` - Y position in block coordinates
/// * `top` - DC value of block above
/// * `left` - DC value of block to the left
/// * `topleft` - DC value of block diagonally above-left
/// * `topright` - DC value of block diagonally above-right
/// * `toptop` - DC value of block two rows above
/// * `leftleft` - DC value of block two columns left
/// * `prev_local_grad` - Previous local gradient (for property 8)
///
/// Returns (properties, new_local_grad) where new_local_grad should be passed
/// as prev_local_grad for the next pixel in the row.
pub fn compute_dc_properties(
    channel_idx: u32,
    x: usize,
    y: usize,
    top: i32,
    left: i32,
    topleft: i32,
    topright: i32,
    toptop: i32,
    leftleft: i32,
    prev_local_grad: i32,
) -> ([i32; NUM_DC_PROPERTIES], i32) {
    let mut props = [0i32; NUM_DC_PROPERTIES];

    // Static properties
    props[0] = channel_idx as i32;
    props[1] = 0; // group_id/stream, typically 0 for DC

    // Position
    props[2] = y as i32;
    props[3] = x as i32;

    // Absolute neighbors
    props[4] = top.wrapping_abs();
    props[5] = left.wrapping_abs();

    // Raw neighbors
    props[6] = top;
    props[7] = left;

    // Local gradient (left - prev_local_grad) - maintained across row
    let local_grad = left.wrapping_add(top).wrapping_sub(topleft);
    props[8] = left.wrapping_sub(prev_local_grad);

    // Gradient (left + top - topleft) - PRIMARY SPLIT PROPERTY
    props[9] = local_grad;

    // FFV1 context properties
    props[10] = left.wrapping_sub(topleft);
    props[11] = topleft.wrapping_sub(top);
    props[12] = top.wrapping_sub(topright);
    props[13] = top.wrapping_sub(toptop);
    props[14] = left.wrapping_sub(leftleft);

    (props, local_grad)
}

/// Gather DC samples from quantized DC values.
///
/// # Arguments
/// * `samples` - Sample collection to add to
/// * `quant_dc` - Quantized DC values [channel][y][x]
pub fn gather_dc_samples(samples: &mut DcTreeSamples, quant_dc: &[Vec<Vec<i16>>; 3]) {
    if quant_dc[0].is_empty() || quant_dc[0][0].is_empty() {
        return;
    }

    let height = quant_dc[0].len();
    let width = quant_dc[0][0].len();

    // Gather in encoding channel order: Y (1), X (0), B (2)
    for (enc_idx, &c) in [1usize, 0, 2].iter().enumerate() {
        let channel = &quant_dc[c];

        for y in 0..height {
            let mut prev_local_grad = 0i32;

            for x in 0..width {
                let dc_val = channel[y][x] as i32;

                // Get neighbors with edge handling matching jxl-rs decoder
                let left = if x > 0 {
                    channel[y][x - 1] as i32
                } else if y > 0 {
                    channel[y - 1][x] as i32
                } else {
                    0
                };

                let top = if y > 0 {
                    channel[y - 1][x] as i32
                } else {
                    left
                };

                let topleft = if x > 0 && y > 0 {
                    channel[y - 1][x - 1] as i32
                } else {
                    left
                };

                let topright = if y > 0 && x + 1 < width {
                    channel[y - 1][x + 1] as i32
                } else {
                    top
                };

                let toptop = if y > 1 { channel[y - 2][x] as i32 } else { top };

                let leftleft = if x > 1 {
                    channel[y][x - 2] as i32
                } else {
                    left
                };

                // Compute prediction and residual
                let prediction = clamped_gradient(top, left, topleft);
                let residual = dc_val - prediction;

                // Compute properties and add sample
                let (props, new_local_grad) = compute_dc_properties(
                    enc_idx as u32,
                    x,
                    y,
                    top,
                    left,
                    topleft,
                    topright,
                    toptop,
                    leftleft,
                    prev_local_grad,
                );
                samples.add_sample(residual, props);

                prev_local_grad = new_local_grad;
            }
        }
    }
}

/// A decision tree node for DC context assignment.
#[derive(Clone, Debug)]
pub struct DcTreeNode {
    /// Property to split on (-1 for leaf).
    pub property: i32,
    /// Split value (samples with property <= splitval go left).
    pub splitval: i32,
    /// Left child index (for internal nodes).
    pub lchild: usize,
    /// Right child index (for internal nodes).
    pub rchild: usize,
    /// Context ID (for leaf nodes).
    pub context_id: u32,
    /// Predictor for leaf nodes (0=Zero, 5=Gradient, etc.)
    pub predictor: u32,
}

impl Default for DcTreeNode {
    fn default() -> Self {
        Self {
            property: -1,
            splitval: 0,
            lchild: 0,
            rchild: 0,
            context_id: 0,
            predictor: 5, // Default: Gradient (matches DC prediction)
        }
    }
}

/// A learned DC context tree.
pub type DcTree = Vec<DcTreeNode>;

/// Estimate bits needed to encode tokens with a given distribution.
fn estimate_bits(counts: &[u32], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut bits = 0.0;

    for &count in counts {
        if count > 0 {
            let p = count as f64 / total_f;
            bits -= (count as f64) * jxl_simd::fast_log2f(p as f32) as f64;
        }
    }
    bits
}

/// Estimate entropy cost for a subset of samples.
fn estimate_subset_cost(samples: &DcTreeSamples, indices: &[usize], max_token: u32) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }

    let histogram_size = (max_token + 1) as usize;
    let mut counts = vec![0u32; histogram_size];
    let mut total = 0u32;

    for &idx in indices {
        let tok = samples.residual_tokens[idx];
        if (tok as usize) < histogram_size {
            counts[tok as usize] += 1;
            total += 1;
        }
    }

    estimate_bits(&counts, total)
}

/// Find the best split for a set of samples.
///
/// Returns (property_idx, splitval, left_indices, right_indices, gain)
/// where gain is the entropy reduction from the split.
fn find_best_split(
    samples: &DcTreeSamples,
    indices: &[usize],
    max_token: u32,
) -> Option<(usize, i32, Vec<usize>, Vec<usize>, f64)> {
    if indices.len() < MIN_SAMPLES_PER_LEAF * 2 {
        return None;
    }

    let current_cost = estimate_subset_cost(samples, indices, max_token);
    let mut best_gain = 0.0f64;
    let mut best_split: Option<(usize, i32, Vec<usize>, Vec<usize>)> = None;

    for &prop_idx in SPLIT_PROPERTIES {
        // Collect unique split values for this property
        let props = &samples.props[prop_idx];
        let mut values: Vec<i32> = indices.iter().map(|&i| props[i]).collect();
        values.sort_unstable();
        values.dedup();

        // Try splits at quantile boundaries (for efficiency)
        let num_quantiles = 32.min(values.len() - 1);
        if num_quantiles == 0 {
            continue;
        }

        for q in 0..num_quantiles {
            let split_idx = (values.len() * (q + 1)) / (num_quantiles + 1);
            if split_idx == 0 || split_idx >= values.len() {
                continue;
            }
            let splitval = values[split_idx - 1];

            // Partition samples
            let (left, right): (Vec<usize>, Vec<usize>) =
                indices.iter().copied().partition(|&i| props[i] <= splitval);

            if left.len() < MIN_SAMPLES_PER_LEAF || right.len() < MIN_SAMPLES_PER_LEAF {
                continue;
            }

            // Compute cost reduction
            let left_cost = estimate_subset_cost(samples, &left, max_token);
            let right_cost = estimate_subset_cost(samples, &right, max_token);
            let new_cost = left_cost + right_cost;
            let gain = current_cost - new_cost;

            // Add overhead for the split itself (approximate)
            let overhead = 10.0; // bits for property + splitval encoding
            let net_gain = gain - overhead;

            if net_gain > best_gain {
                best_gain = net_gain;
                best_split = Some((prop_idx, splitval, left, right));
            }
        }
    }

    best_split.map(|(prop, sv, l, r)| (prop, sv, l, r, best_gain))
}

/// Recursively build the DC tree.
fn build_tree_recursive(
    samples: &DcTreeSamples,
    indices: &[usize],
    depth: usize,
    tree: &mut DcTree,
    next_context: &mut u32,
    max_token: u32,
) -> usize {
    let node_idx = tree.len();
    tree.push(DcTreeNode::default());

    // Check if we should make this a leaf
    if depth >= MAX_TREE_DEPTH || indices.len() < MIN_SAMPLES_PER_LEAF * 2 {
        tree[node_idx].property = -1;
        tree[node_idx].context_id = *next_context;
        *next_context += 1;
        return node_idx;
    }

    // Try to find a beneficial split
    if let Some((prop_idx, splitval, left_indices, right_indices, _gain)) =
        find_best_split(samples, indices, max_token)
    {
        // Build children first
        let lchild = build_tree_recursive(
            samples,
            &left_indices,
            depth + 1,
            tree,
            next_context,
            max_token,
        );
        let rchild = build_tree_recursive(
            samples,
            &right_indices,
            depth + 1,
            tree,
            next_context,
            max_token,
        );

        tree[node_idx].property = prop_idx as i32;
        tree[node_idx].splitval = splitval;
        tree[node_idx].lchild = lchild;
        tree[node_idx].rchild = rchild;
    } else {
        // No beneficial split found, make this a leaf
        tree[node_idx].property = -1;
        tree[node_idx].context_id = *next_context;
        *next_context += 1;
    }

    node_idx
}

/// Learn an optimal DC context tree from samples.
///
/// # Arguments
/// * `samples` - Collected DC samples
/// * `max_token` - Maximum token value (for histogram sizing)
///
/// # Returns
/// A learned tree and the number of contexts it uses.
pub fn learn_dc_tree(samples: &DcTreeSamples, max_token: u32) -> (DcTree, u32) {
    if samples.num_samples == 0 {
        // Empty samples: return single-leaf tree
        let tree = vec![DcTreeNode {
            property: -1,
            context_id: 0,
            ..Default::default()
        }];
        return (tree, 1);
    }

    let mut tree = DcTree::new();
    let mut next_context = 0u32;
    let indices: Vec<usize> = (0..samples.num_samples).collect();

    build_tree_recursive(
        samples,
        &indices,
        0,
        &mut tree,
        &mut next_context,
        max_token,
    );

    (tree, next_context)
}

/// Traverse the learned tree to get a context for a DC value.
#[inline]
pub fn get_dc_context(tree: &DcTree, props: &[i32; NUM_DC_PROPERTIES]) -> u32 {
    let mut idx = 0;
    loop {
        let node = &tree[idx];
        if node.property < 0 {
            return node.context_id;
        }
        let pval = props[node.property as usize];
        if pval <= node.splitval {
            idx = node.lchild;
        } else {
            idx = node.rchild;
        }
    }
}

/// Convert a learned DC tree to context tree tokens for bitstream encoding.
///
/// The token format matches the modular tree format:
/// - Internal node: (property, splitval) pairs
/// - Leaf node: (predictor, multiplier, offset) but for DC we just use context
///
/// Format: sequence of (context, value) tokens that describe the tree structure.
///
/// IMPORTANT: Tokens must be in BFS (breadth-first/level-order) order, NOT DFS.
/// The decoder computes child indices assuming BFS order.
pub fn tree_to_tokens(tree: &DcTree) -> Vec<(u32, u32)> {
    use super::common::pack_signed;
    use alloc::collections::VecDeque;

    let mut tokens = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(0usize);

    #[cfg(feature = "debug-tokens")]
    eprintln!("tree_to_tokens: tree has {} nodes", tree.len());
    #[cfg(feature = "debug-tokens")]
    let mut leaf_count = 0;

    while let Some(idx) = queue.pop_front() {
        let node = &tree[idx];

        if node.property < 0 {
            // Leaf node: emit predictor, multiplier, offset
            #[cfg(feature = "debug-tokens")]
            {
                eprintln!(
                    "  BFS node {}: LEAF (context_id={}, predictor={}, leaf_order={})",
                    idx, node.context_id, node.predictor, leaf_count
                );
                leaf_count += 1;
            }
            // Context 1: property = 0 signals leaf node (decoder subtracts 1, gets -1)
            tokens.push((1, 0));
            // Context 2: predictor (use node's predictor field)
            tokens.push((2, node.predictor));
            // Context 3: offset (0)
            tokens.push((3, 0));
            // Context 4: multiplier log (0 for multiplier=1 since (0+1)<<0 = 1)
            tokens.push((4, 0));
            // Context 5: multiplier bits (0)
            tokens.push((5, 0));
        } else {
            // Internal node: emit property and splitval
            #[cfg(feature = "debug-tokens")]
            eprintln!(
                "  BFS node {}: INTERNAL (prop={}, split={}, left={}, right={})",
                idx, node.property, node.splitval, node.lchild, node.rchild
            );
            // Context 1: property+1 (decoder subtracts 1 to get actual property index)
            let prop_token = (node.property + 1) as u32;
            tokens.push((1, prop_token));
            // Context 0: splitval (packed signed)
            tokens.push((0, pack_signed(node.splitval)));

            // Queue children for BFS traversal (left first, then right)
            queue.push_back(node.lchild);
            queue.push_back(node.rchild);
        }
    }

    #[cfg(feature = "debug-tokens")]
    eprintln!("  Total: {} tokens, {} leaves", tokens.len(), leaf_count);
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_dc_properties() {
        // Test gradient property (property 9 = left + top - topleft)
        let (props, _) = compute_dc_properties(
            0,   // channel
            5,   // x
            3,   // y
            100, // top
            100, // left
            100, // topleft
            100, // topright
            100, // toptop
            100, // leftleft
            0,   // prev_local_grad
        );
        // Gradient: 100 + 100 - 100 = 100
        assert_eq!(props[9], 100);

        // Test position properties
        assert_eq!(props[2], 3); // y
        assert_eq!(props[3], 5); // x

        // Test absolute values
        assert_eq!(props[4], 100); // |top|
        assert_eq!(props[5], 100); // |left|

        // Test raw values
        assert_eq!(props[6], 100); // top
        assert_eq!(props[7], 100); // left

        // Test FFV1 properties
        let (props2, _) = compute_dc_properties(0, 0, 0, 200, 150, 100, 180, 200, 120, 0);
        // Gradient: 150 + 200 - 100 = 250
        assert_eq!(props2[9], 250);
        // FFV1 left - topleft: 150 - 100 = 50
        assert_eq!(props2[10], 50);
        // FFV1 topleft - top: 100 - 200 = -100
        assert_eq!(props2[11], -100);
    }

    #[test]
    fn test_gather_dc_samples_empty() {
        let quant_dc: [Vec<Vec<i16>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut samples = DcTreeSamples::new();
        gather_dc_samples(&mut samples, &quant_dc);
        assert_eq!(samples.num_samples, 0);
    }

    #[test]
    fn test_gather_dc_samples_simple() {
        // 4x4 constant DC values
        let channel = vec![vec![100i16; 4]; 4];
        let quant_dc: [Vec<Vec<i16>>; 3] = [channel.clone(), channel.clone(), channel];

        let mut samples = DcTreeSamples::new();
        gather_dc_samples(&mut samples, &quant_dc);

        // 4x4 x 3 channels = 48 samples
        assert_eq!(samples.num_samples, 48);
    }

    #[test]
    fn test_learn_dc_tree_empty() {
        let samples = DcTreeSamples::new();
        let (tree, num_contexts) = learn_dc_tree(&samples, 64);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].property, -1);
        assert_eq!(num_contexts, 1);
    }

    #[test]
    fn test_learn_dc_tree_constant() {
        // Constant DC values should produce single-leaf tree
        let channel = vec![vec![50i16; 8]; 8];
        let quant_dc: [Vec<Vec<i16>>; 3] = [channel.clone(), channel.clone(), channel];

        let mut samples = DcTreeSamples::new();
        gather_dc_samples(&mut samples, &quant_dc);

        let (tree, num_contexts) = learn_dc_tree(&samples, 64);

        // Should have at least 1 context
        assert!(num_contexts >= 1);
        // Root should exist
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_get_dc_context() {
        // Create a simple 2-leaf tree that splits on gradient property (property 9)
        let tree = vec![
            DcTreeNode {
                property: 9, // Gradient (left + top - topleft)
                splitval: 150,
                lchild: 1,
                rchild: 2,
                context_id: 0,
                predictor: 0, // Not used for internal nodes
            },
            DcTreeNode {
                property: -1,
                context_id: 0,
                ..Default::default()
            },
            DcTreeNode {
                property: -1,
                context_id: 1,
                ..Default::default()
            },
        ];

        // Gradient <= 150 should go to context 0
        // top=100, left=100, topleft=100 => gradient = 100 + 100 - 100 = 100
        let (props_low, _) = compute_dc_properties(0, 0, 0, 100, 100, 100, 100, 100, 100, 0);
        assert_eq!(props_low[9], 100);
        assert_eq!(get_dc_context(&tree, &props_low), 0);

        // Gradient > 150 should go to context 1
        // top=200, left=100, topleft=50 => gradient = 100 + 200 - 50 = 250
        let (props_high, _) = compute_dc_properties(0, 0, 0, 200, 100, 50, 200, 200, 100, 0);
        assert_eq!(props_high[9], 250);
        assert_eq!(get_dc_context(&tree, &props_high), 1);
    }

    #[test]
    fn test_tree_to_tokens() {
        // Single leaf tree
        let tree = vec![DcTreeNode {
            property: -1,
            context_id: 0,
            ..Default::default()
        }];

        let tokens = tree_to_tokens(&tree);
        // Leaf emits 5 tokens: property marker, predictor, offset, multiplier, unused
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], (1, 0)); // property = -1 (leaf marker)
    }
}

/// Number of AC metadata contexts (EPF=1, CfL=2, QF=4, ACS=4).
const NUM_AC_META_CONTEXTS: u32 = 11;

/// Create tree tokens for a merged MA tree with AC metadata routing and learned DC subtree.
///
/// Builds a tree where:
/// - Root splits on stream_id (property 1, splitval=2): LEFT → AC metadata, RIGHT → DC
/// - AC metadata subtree routes based on channel/y/left properties to 11 contexts
/// - DC subtree uses the learned tree for context assignment
/// - A padding chain pushes DC leaves deep enough in BFS that they appear after
///   all AC metadata leaves (dummy chain leaves get "wasted" context IDs)
///
/// Returns `(tokens, total_contexts, dc_ctx_remap, ac_meta_ctx_map)` where:
/// - `tokens`: BFS-ordered tree token stream for bitstream encoding
/// - `total_contexts`: total number of contexts (AC meta + dummy + DC)
/// - `dc_ctx_remap`: maps original DC context ID → BFS context ID
///   (needed because BFS leaf order may differ from DFS context assignment)
/// - `ac_meta_ctx_map`: maps original AC metadata context [0-10] → BFS context ID
pub fn tree_tokens_with_ac_metadata_prefix(
    dc_tree: &DcTree,
    learned_num_contexts: u32,
    num_dc_groups: usize,
) -> (
    Vec<(u32, u32)>,
    u32,
    Vec<u32>,
    [u32; NUM_AC_META_CONTEXTS as usize],
) {
    use super::common::pack_signed;
    use alloc::collections::VecDeque;

    // ─── Node types for building the merged tree ───

    enum LeafType {
        AcMeta(u32), // original AC metadata context 0-10
        Dummy,       // padding chain leaf (no tokens, wasted context)
        Dc(u32),     // original DC context from learned tree
    }

    struct FlatNode {
        property: i32,
        splitval: i32,
        predictor: u32,
        left: usize,
        right: usize,
        leaf_type: LeafType,
    }

    let mut flat: Vec<FlatNode> = Vec::new();

    let mk_internal =
        |flat: &mut Vec<FlatNode>, prop: i32, split: i32, l: usize, r: usize| -> usize {
            let idx = flat.len();
            flat.push(FlatNode {
                property: prop,
                splitval: split,
                predictor: 0,
                left: l,
                right: r,
                leaf_type: LeafType::Dummy,
            });
            idx
        };

    let mk_leaf = |flat: &mut Vec<FlatNode>, pred: u32, lt: LeafType| -> usize {
        let idx = flat.len();
        flat.push(FlatNode {
            property: -1,
            splitval: 0,
            predictor: pred,
            left: 0,
            right: 0,
            leaf_type: lt,
        });
        idx
    };

    // ─── Build AC metadata subtree (bottom-up for correct index references) ───
    //
    // Channel ordering (from jxl-oxide hf_metadata.rs):
    //   ch0 = x_from_y (YtoX CfL), ch1 = b_from_y (YtoB CfL),
    //   ch2 = block_info (ACS at y=0, QF at y=1), ch3 = sharpness (EPF)
    //
    // Context assignment (from dc_coding.rs):
    //   EPF=0(Zero), YtoB=1(Gradient), YtoX=2(Gradient),
    //   QF=3-6(Left), ACS=7-10(Zero)

    // QF leaves: predictor=1 (Left), contexts 3-6
    let qf3 = mk_leaf(&mut flat, 1, LeafType::AcMeta(3));
    let qf4 = mk_leaf(&mut flat, 1, LeafType::AcMeta(4));
    let qf5 = mk_leaf(&mut flat, 1, LeafType::AcMeta(5));
    let qf6 = mk_leaf(&mut flat, 1, LeafType::AcMeta(6));
    // ACS leaves: predictor=0 (Zero), contexts 7-10
    let acs7 = mk_leaf(&mut flat, 0, LeafType::AcMeta(7));
    let acs8 = mk_leaf(&mut flat, 0, LeafType::AcMeta(8));
    let acs9 = mk_leaf(&mut flat, 0, LeafType::AcMeta(9));
    let acs10 = mk_leaf(&mut flat, 0, LeafType::AcMeta(10));
    // QF splits on property 7 (left neighbor): >11, >5, >3, <=3
    let qf_l = mk_internal(&mut flat, 7, 11, qf3, qf4);
    let qf_r = mk_internal(&mut flat, 7, 3, qf5, qf6);
    let qf_root = mk_internal(&mut flat, 7, 5, qf_l, qf_r);
    // ACS splits on property 7 (left neighbor): same thresholds
    let acs_l = mk_internal(&mut flat, 7, 11, acs7, acs8);
    let acs_r = mk_internal(&mut flat, 7, 3, acs9, acs10);
    let acs_root = mk_internal(&mut flat, 7, 5, acs_l, acs_r);
    // Block info: property 2 (y), splitval=0 → LEFT=QF(y>0), RIGHT=ACS(y=0)
    let blockinfo = mk_internal(&mut flat, 2, 0, qf_root, acs_root);
    // Channel leaves
    let epf = mk_leaf(&mut flat, 0, LeafType::AcMeta(0)); // ch3, Zero pred
    let ytob = mk_leaf(&mut flat, 5, LeafType::AcMeta(1)); // ch1, Gradient pred
    let ytox = mk_leaf(&mut flat, 5, LeafType::AcMeta(2)); // ch0, Gradient pred
    // Channel routing: prop 0 (channel)
    let ch2 = mk_internal(&mut flat, 0, 2, epf, blockinfo); // ch>2→EPF, ch<=2→blockinfo
    let ch0 = mk_internal(&mut flat, 0, 0, ytob, ytox); // ch>0→YtoB, ch<=0→YtoX
    let ac_root = mk_internal(&mut flat, 0, 1, ch2, ch0); // ch>1→ch2, ch<=1→ch0

    // ─── Build DC subtree ───
    //
    // IMPORTANT: The JXL spec convention is LEFT = property > splitval,
    // RIGHT = property <= splitval. But our DC tree builder uses the opposite:
    // lchild = property <= splitval, rchild = property > splitval.
    // We SWAP the children here so the decoder evaluates correctly.

    let dc_start = flat.len();
    for node in dc_tree {
        if node.property < 0 {
            mk_leaf(&mut flat, node.predictor, LeafType::Dc(node.context_id));
        } else {
            mk_internal(
                &mut flat,
                node.property,
                node.splitval,
                dc_start + node.rchild, // JXL LEFT = property > splitval = our rchild
                dc_start + node.lchild, // JXL RIGHT = property <= splitval = our lchild
            );
        }
    }
    let dc_root_idx = dc_start;

    // ─── Build merged root ───
    //
    // No padding chain needed: we use a full context remap (dc_ctx_remap) that
    // correctly maps each DC tree context to its BFS position, regardless of
    // where DC leaves appear relative to AC metadata leaves in BFS order.
    //
    // Previous versions used a padding chain (property 1 splits) to push DC
    // leaves deeper in BFS, but decoders validate that splitval is within the
    // property's narrowing range, making repeated same-property splits fail.
    //
    // Property 1 (stream_id), splitval=num_dc_groups:
    //   LEFT (stream_id > num_dc_groups): AC metadata
    //   RIGHT (stream_id <= num_dc_groups): DC subtree
    //
    // DC groups have stream_ids 1..num_dc_groups (from ModularStreamId::VarDCTDC).
    // AC metadata groups have stream_ids 1+2*num_dc_groups.. (from ModularStreamId::ACMetadata).
    // So splitval=num_dc_groups correctly routes all DC groups to the DC subtree
    // and all AC metadata groups to the AC metadata subtree.
    let root = mk_internal(&mut flat, 1, num_dc_groups as i32, ac_root, dc_root_idx);

    // ─── BFS to generate token stream and track context ID mapping ───
    //
    // The decoder reads tokens in BFS order, assigning sequential context IDs
    // to leaves. Dummy leaves from the padding chain get context IDs between
    // AC metadata groups (they interleave at each BFS depth level).
    // We track the actual BFS context for each AC metadata and DC leaf.

    let mut tokens = Vec::new();
    let mut queue = VecDeque::new();
    let mut leaf_ctx = 0u32;
    let mut ac_meta_ctx_map = [0u32; NUM_AC_META_CONTEXTS as usize];
    let mut dc_ctx_map = Vec::new();

    // Emit root token
    let rn = &flat[root];
    tokens.push((1, (rn.property + 1) as u32));
    tokens.push((0, pack_signed(rn.splitval)));
    queue.push_back(root);

    while let Some(idx) = queue.pop_front() {
        for child_idx in [flat[idx].left, flat[idx].right] {
            let cn = &flat[child_idx];
            if cn.property < 0 {
                // Leaf: emit 5 tokens (property marker, predictor, offset, multiplier, unused)
                tokens.push((1, 0)); // property = -1 → encoded as 0
                tokens.push((2, cn.predictor));
                tokens.push((3, 0)); // offset
                tokens.push((4, 0)); // multiplier
                tokens.push((5, 0)); // unused
                match cn.leaf_type {
                    LeafType::AcMeta(orig) => {
                        ac_meta_ctx_map[orig as usize] = leaf_ctx;
                    }
                    LeafType::Dc(orig) => {
                        dc_ctx_map.push((orig, leaf_ctx));
                    }
                    LeafType::Dummy => {}
                }
                leaf_ctx += 1;
            } else {
                // Internal: emit 2 tokens (property, splitval)
                tokens.push((1, (cn.property + 1) as u32));
                tokens.push((0, pack_signed(cn.splitval)));
                queue.push_back(child_idx);
            }
        }
    }

    // Build DC context remap: dc_ctx_remap[orig_dc_ctx] = BFS context ID.
    // BFS and DFS can produce different leaf orderings for unbalanced trees,
    // plus the child swap changes BFS order, so we need a full remap.
    let mut dc_ctx_remap = vec![0u32; learned_num_contexts as usize];
    for &(orig, bfs) in &dc_ctx_map {
        dc_ctx_remap[orig as usize] = bfs;
    }
    let total_contexts = leaf_ctx;

    (tokens, total_contexts, dc_ctx_remap, ac_meta_ctx_map)
}

/// Build a context tree with AC metadata contexts only (no DC).
///
/// Used when `use_lf_frame` is true: DC is encoded in a separate frame,
/// so the main VarDCT frame's LfGlobal tree only needs AC metadata contexts.
///
/// Returns (tree_tokens, total_contexts, ac_meta_ctx_map).
pub fn ac_metadata_only_tree() -> (Vec<(u32, u32)>, u32, [u32; NUM_AC_META_CONTEXTS as usize]) {
    use super::common::pack_signed;
    use alloc::collections::VecDeque;

    enum LeafType {
        AcMeta(u32),
    }

    struct FlatNode {
        property: i32,
        splitval: i32,
        predictor: u32,
        left: usize,
        right: usize,
        leaf_type: Option<LeafType>,
    }

    let mut flat: Vec<FlatNode> = Vec::new();

    let mk_internal =
        |flat: &mut Vec<FlatNode>, prop: i32, split: i32, l: usize, r: usize| -> usize {
            let idx = flat.len();
            flat.push(FlatNode {
                property: prop,
                splitval: split,
                predictor: 0,
                left: l,
                right: r,
                leaf_type: None,
            });
            idx
        };

    let mk_leaf = |flat: &mut Vec<FlatNode>, pred: u32, lt: LeafType| -> usize {
        let idx = flat.len();
        flat.push(FlatNode {
            property: -1,
            splitval: 0,
            predictor: pred,
            left: 0,
            right: 0,
            leaf_type: Some(lt),
        });
        idx
    };

    // Build AC metadata subtree (same structure as in tree_tokens_with_ac_metadata_prefix)
    let qf3 = mk_leaf(&mut flat, 1, LeafType::AcMeta(3));
    let qf4 = mk_leaf(&mut flat, 1, LeafType::AcMeta(4));
    let qf5 = mk_leaf(&mut flat, 1, LeafType::AcMeta(5));
    let qf6 = mk_leaf(&mut flat, 1, LeafType::AcMeta(6));
    let acs7 = mk_leaf(&mut flat, 0, LeafType::AcMeta(7));
    let acs8 = mk_leaf(&mut flat, 0, LeafType::AcMeta(8));
    let acs9 = mk_leaf(&mut flat, 0, LeafType::AcMeta(9));
    let acs10 = mk_leaf(&mut flat, 0, LeafType::AcMeta(10));
    let qf_l = mk_internal(&mut flat, 7, 11, qf3, qf4);
    let qf_r = mk_internal(&mut flat, 7, 3, qf5, qf6);
    let qf_root = mk_internal(&mut flat, 7, 5, qf_l, qf_r);
    let acs_l = mk_internal(&mut flat, 7, 11, acs7, acs8);
    let acs_r = mk_internal(&mut flat, 7, 3, acs9, acs10);
    let acs_root = mk_internal(&mut flat, 7, 5, acs_l, acs_r);
    let blockinfo = mk_internal(&mut flat, 2, 0, qf_root, acs_root);
    let epf = mk_leaf(&mut flat, 0, LeafType::AcMeta(0));
    let ytob = mk_leaf(&mut flat, 5, LeafType::AcMeta(1));
    let ytox = mk_leaf(&mut flat, 5, LeafType::AcMeta(2));
    let ch2 = mk_internal(&mut flat, 0, 2, epf, blockinfo);
    let ch0 = mk_internal(&mut flat, 0, 0, ytob, ytox);
    let root = mk_internal(&mut flat, 0, 1, ch2, ch0);

    // BFS to generate token stream
    let mut tokens = Vec::new();
    let mut queue = VecDeque::new();
    let mut leaf_ctx = 0u32;
    let mut ac_meta_ctx_map = [0u32; NUM_AC_META_CONTEXTS as usize];

    let rn = &flat[root];
    tokens.push((1, (rn.property + 1) as u32));
    tokens.push((0, pack_signed(rn.splitval)));
    queue.push_back(root);

    while let Some(idx) = queue.pop_front() {
        for child_idx in [flat[idx].left, flat[idx].right] {
            let cn = &flat[child_idx];
            if cn.property < 0 {
                tokens.push((1, 0));
                tokens.push((2, cn.predictor));
                tokens.push((3, 0));
                tokens.push((4, 0));
                tokens.push((5, 0));
                if let Some(LeafType::AcMeta(orig)) = &cn.leaf_type {
                    ac_meta_ctx_map[*orig as usize] = leaf_ctx;
                }
                leaf_ctx += 1;
            } else {
                tokens.push((1, (cn.property + 1) as u32));
                tokens.push((0, pack_signed(cn.splitval)));
                queue.push_back(child_idx);
            }
        }
    }

    let total_contexts = leaf_ctx;
    (tokens, total_contexts, ac_meta_ctx_map)
}

/// Collect DC tokens using a learned tree for context assignment.
///
/// This is the learned-tree version of `collect_dc_tokens_region()` from dc_coding.rs.
/// Instead of using GRADIENT_CONTEXT_LUT, it traverses the learned tree to get contexts.
pub fn collect_dc_tokens_with_tree(
    quant_dc: &[Vec<Vec<i16>>; 3],
    tree: &DcTree,
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> Vec<crate::entropy_coding::token::Token> {
    use crate::entropy_coding::token::Token;

    let region_width = end_bx - start_bx;
    let region_height = end_by - start_by;

    if region_width == 0 || region_height == 0 {
        return Vec::new();
    }

    let mut tokens = Vec::with_capacity(region_width * region_height * 3);

    // Encode in channel order: Y (1), X (0), B (2)
    for (enc_idx, &c) in [1usize, 0, 2].iter().enumerate() {
        let channel = &quant_dc[c];

        for y in start_by..end_by {
            let mut prev_local_grad = 0i32;

            for x in start_bx..end_bx {
                let dc_val = channel[y][x] as i32;

                // Get neighbors with proper edge handling
                let left = if x > start_bx {
                    channel[y][x - 1] as i32
                } else if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    0
                };

                let top = if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    left
                };

                let topleft = if x > start_bx && y > start_by {
                    channel[y - 1][x - 1] as i32
                } else {
                    left
                };

                let topright = if y > start_by && x + 1 < end_bx {
                    channel[y - 1][x + 1] as i32
                } else {
                    top
                };

                let toptop = if y > start_by + 1 {
                    channel[y - 2][x] as i32
                } else {
                    top
                };

                let leftleft = if x > start_bx + 1 {
                    channel[y][x - 2] as i32
                } else {
                    left
                };

                // Compute prediction and residual
                let prediction = clamped_gradient(top, left, topleft);
                let residual = dc_val - prediction;

                // Compute properties and get context from tree
                let (props, new_local_grad) = compute_dc_properties(
                    enc_idx as u32,
                    x - start_bx,
                    y - start_by,
                    top,
                    left,
                    topleft,
                    topright,
                    toptop,
                    leftleft,
                    prev_local_grad,
                );
                let tree_ctx = get_dc_context(tree, &props);
                // DC tree assigns contexts starting from 0; the encoder adds
                // NUM_AC_METADATA_CONTEXTS (11) offset when building final tokens.
                let ctx_id = tree_ctx;

                tokens.push(Token::new(ctx_id, pack_signed(residual)));

                prev_local_grad = new_local_grad;
            }
        }
    }

    tokens
}

// ──────────────────────────────────────────────────────────────────
// kWPFixedDC tree — fixed balanced BSP on property 15 (wp_max_error)
// Matches libjxl's PredefinedTree(kWPFixedDC, ...) exactly.
// ──────────────────────────────────────────────────────────────────

/// kWPFixedDC cutoff values (from libjxl enc_encoding.cc).
/// These are the split thresholds for the wp_max_error property.
const WP_FIXED_DC_CUTOFFS: &[i32] = &[
    -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -4, -3, -1, 0, 1, 3, 5, 7,
    11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
];

/// Property index for wp_max_error in the JXL modular property list.
/// kNumStaticProperties(2) + 13 = 15. Used for tree serialization.
pub const WP_PROP_INDEX: i32 = 15;

/// Build the kWPFixedDC tree: a balanced BSP tree on wp_max_error (property 15)
/// with all leaves using Predictor::Weighted.
///
/// Matches libjxl's `MakeFixedTree(kWPProp, cutoffs, Predictor::Weighted, total_pixels, bitdepth)`.
///
/// # Arguments
/// * `total_pixels` - total DC pixels (width_blocks * height_blocks * 3 channels)
/// * `bitdepth` - bit depth of the DC values (typically 8)
pub fn build_wp_fixed_dc_tree(total_pixels: usize, bitdepth: u32) -> (DcTree, u32) {
    let log_px = if total_pixels > 0 {
        (usize::BITS - total_pixels.leading_zeros()) as usize // ceil_log2
    } else {
        0
    };
    let min_gap = if log_px < 14 { 8 * (14 - log_px) } else { 0 };
    let shift = if bitdepth > 11 {
        (bitdepth - 11).min(4)
    } else {
        0
    };
    let mul = 1i32 << shift;

    let cutoffs = WP_FIXED_DC_CUTOFFS;
    let mut tree = DcTree::new();
    let mut next_context = 0u32;

    build_wp_bsp_recursive(
        cutoffs,
        0,
        cutoffs.len(),
        min_gap,
        mul,
        &mut tree,
        &mut next_context,
    );

    (tree, next_context)
}

/// Recursively build a balanced BSP tree from sorted cutoffs.
///
/// Mirrors libjxl's MakeFixedTree BFS queue, but builds in DFS order
/// (our tree_tokens_with_ac_metadata_prefix handles the BFS conversion).
fn build_wp_bsp_recursive(
    cutoffs: &[i32],
    begin: usize,
    end: usize,
    min_gap: usize,
    mul: i32,
    tree: &mut DcTree,
    next_context: &mut u32,
) -> usize {
    let node_idx = tree.len();

    if begin + min_gap >= end {
        // Leaf node
        tree.push(DcTreeNode {
            property: -1,
            context_id: *next_context,
            predictor: 6, // Predictor::Weighted
            ..Default::default()
        });
        *next_context += 1;
        return node_idx;
    }

    let split = (begin + end) / 2;
    let cutoff = cutoffs[split] * mul;

    // Placeholder — filled after children are built
    tree.push(DcTreeNode::default());

    // rchild = values > cutoff → covers [split+1, end)
    let rchild = build_wp_bsp_recursive(cutoffs, split + 1, end, min_gap, mul, tree, next_context);
    // lchild = values <= cutoff → covers [begin, split)
    let lchild = build_wp_bsp_recursive(cutoffs, begin, split, min_gap, mul, tree, next_context);

    tree[node_idx] = DcTreeNode {
        property: WP_PROP_INDEX,
        splitval: cutoff,
        lchild,
        rchild,
        context_id: 0,
        predictor: 0,
    };

    node_idx
}

/// Traverse the kWPFixedDC tree using wp_max_error value.
///
/// Specialized traversal for the WP fixed tree — only uses the wp_max_error
/// property (property 15), which is the only property this tree splits on.
#[inline]
pub fn get_wp_dc_context(tree: &DcTree, wp_max_error: i32) -> u32 {
    let mut idx = 0;
    loop {
        let node = &tree[idx];
        if node.property < 0 {
            return node.context_id;
        }
        // All splits are on wp_max_error (property 15)
        if wp_max_error <= node.splitval {
            idx = node.lchild;
        } else {
            idx = node.rchild;
        }
    }
}

/// Compress statistics for learned DC tree.
pub struct DcTreeStats {
    /// Number of contexts used by the tree.
    pub num_contexts: u32,
    /// Number of samples collected.
    pub num_samples: usize,
    /// Estimated bits saved compared to fixed LUT (positive = better).
    pub bits_saved: f64,
}

/// Learn DC tree and collect tokens in one pass.
///
/// Returns (tree, tokens, stats) where:
/// - tree is the learned context tree
/// - tokens are DC tokens using the learned contexts
/// - stats contains compression statistics
pub fn learn_and_collect_dc_tokens(
    quant_dc: &[Vec<Vec<i16>>; 3],
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> (
    DcTree,
    Vec<crate::entropy_coding::token::Token>,
    DcTreeStats,
) {
    // First pass: gather samples
    let mut samples = DcTreeSamples::new();

    if !quant_dc[0].is_empty() && !quant_dc[0][0].is_empty() {
        // Create a view of just this region for sample gathering
        let region_dc = extract_dc_region(quant_dc, start_bx, start_by, end_bx, end_by);
        gather_dc_samples(&mut samples, &region_dc);
    }

    // Learn tree
    let max_token = 64; // Reasonable max for DC residual tokens
    let (tree, num_contexts) = learn_dc_tree(&samples, max_token);

    // Collect tokens using learned tree
    let tokens = collect_dc_tokens_with_tree(quant_dc, &tree, start_bx, start_by, end_bx, end_by);

    let stats = DcTreeStats {
        num_contexts,
        num_samples: samples.num_samples,
        bits_saved: 0.0, // TODO: estimate actual savings
    };

    (tree, tokens, stats)
}

/// Extract a region of DC values for sample gathering.
#[allow(clippy::needless_range_loop)]
fn extract_dc_region(
    quant_dc: &[Vec<Vec<i16>>; 3],
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> [Vec<Vec<i16>>; 3] {
    let width = end_bx - start_bx;
    let height = end_by - start_by;

    let mut result: [Vec<Vec<i16>>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for c in 0..3 {
        let mut channel = Vec::with_capacity(height);
        for y in start_by..end_by {
            let mut row = Vec::with_capacity(width);
            for x in start_bx..end_bx {
                row.push(quant_dc[c][y][x]);
            }
            channel.push(row);
        }
        result[c] = channel;
    }

    result
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::bit_writer::BitWriter;
    use crate::vardct::context_tree::{write_context_tree, write_learned_context_tree};

    #[test]
    fn test_static_tokens_through_learned_path() {
        use crate::vardct::common::pack_signed;
        use crate::vardct::context_tree::CONTEXT_TREE_TOKENS;
        let num_dc_groups = 1;

        // Get the static tokens with num_dc_groups adjustment
        let mut static_token_pairs: Vec<(u32, u32)> = CONTEXT_TREE_TOKENS.to_vec();
        static_token_pairs[1].1 = pack_signed(1 + num_dc_groups as i32);

        // Write static tree via static path
        let mut static_writer = BitWriter::new();
        write_context_tree(num_dc_groups, &mut static_writer).unwrap();
        static_writer.zero_pad_to_byte();
        let static_bytes = static_writer.finish();

        // Write same tokens via learned path
        let mut learned_writer = BitWriter::new();
        write_learned_context_tree(&static_token_pairs, num_dc_groups, &mut learned_writer)
            .unwrap();
        learned_writer.zero_pad_to_byte();
        let learned_bytes = learned_writer.finish();

        eprintln!(
            "Static: {} bytes, Learned: {} bytes",
            static_bytes.len(),
            learned_bytes.len()
        );

        // They should be bit-identical since they use the same tokens
        assert_eq!(
            static_bytes, learned_bytes,
            "Static and learned paths produce different output for same tokens"
        );
    }
}

#[test]
fn test_wrapped_tree_tokens() {
    use super::*;

    // Single-leaf learned tree (1 DC context, depth 0)
    // Single-leaf DC tree: total = 11 AC meta + 1 DC = 12
    let tree = vec![DcTreeNode {
        property: -1,
        context_id: 0,
        ..Default::default()
    }];

    let (wrapped_tokens, total_contexts, dc_remap, ac_map) =
        tree_tokens_with_ac_metadata_prefix(&tree, 1, 1);
    eprintln!(
        "Merged tree: {} tokens, {} contexts, dc_remap={:?}, ac_map={:?}",
        wrapped_tokens.len(),
        total_contexts,
        dc_remap,
        ac_map,
    );

    assert_eq!(dc_remap.len(), 1);
    assert_eq!(total_contexts, 12); // 11 AC meta + 1 DC
    // All contexts (DC and AC meta) should be unique and within [0, total)
    let mut all_ctxs = std::collections::HashSet::new();
    for &bfs in &dc_remap {
        assert!(
            bfs < total_contexts,
            "DC ctx {} >= total {}",
            bfs,
            total_contexts
        );
        assert!(all_ctxs.insert(bfs), "Duplicate DC BFS context {}", bfs);
    }
    for &bfs in &ac_map {
        assert!(
            bfs < total_contexts,
            "AC meta ctx {} >= total {}",
            bfs,
            total_contexts
        );
        assert!(
            all_ctxs.insert(bfs),
            "Duplicate AC meta BFS context {}",
            bfs
        );
    }
}

#[test]
fn test_wrapped_tree_tokens_depth1_dc() {
    use super::*;

    // Depth-1 DC tree (2 leaves): total = 11 AC meta + 2 DC = 13
    let tree = vec![
        DcTreeNode {
            property: 9,
            splitval: 0,
            lchild: 1,
            rchild: 2,
            ..Default::default()
        },
        DcTreeNode {
            property: -1,
            context_id: 0,
            predictor: 5,
            ..Default::default()
        },
        DcTreeNode {
            property: -1,
            context_id: 1,
            predictor: 5,
            ..Default::default()
        },
    ];

    let (_, total_contexts, dc_remap, ac_map) = tree_tokens_with_ac_metadata_prefix(&tree, 2, 1);
    eprintln!(
        "Depth-1 DC: total={}, dc_remap={:?}, ac_map={:?}",
        total_contexts, dc_remap, ac_map
    );

    // 11 AC meta + 2 DC = 13 (no padding dummies)
    assert_eq!(total_contexts, 13);
    assert_eq!(dc_remap.len(), 2);
    // All contexts should be unique and within [0, total)
    let mut all_ctxs = std::collections::HashSet::new();
    for (i, &bfs) in dc_remap.iter().enumerate() {
        assert!(
            bfs < total_contexts,
            "DC remap[{}]={} >= total {}",
            i,
            bfs,
            total_contexts
        );
        assert!(
            all_ctxs.insert(bfs),
            "Duplicate DC ctx {} at remap[{}]",
            bfs,
            i
        );
    }
    for (i, &bfs) in ac_map.iter().enumerate() {
        assert!(
            bfs < total_contexts,
            "AC meta ctx {} >= total {} at map[{}]",
            bfs,
            total_contexts,
            i
        );
        assert!(
            all_ctxs.insert(bfs),
            "Duplicate AC meta ctx {} at map[{}]",
            bfs,
            i
        );
    }
}

#[test]
fn test_wrapped_tree_tokens_deep_dc() {
    use super::*;

    // DC tree with depth 5 (no padding needed):
    // Build a balanced binary tree with 32 leaves
    let mut tree = Vec::new();
    for i in 0..31 {
        tree.push(DcTreeNode {
            property: 9,
            splitval: (i as i32) * 10,
            lchild: i * 2 + 1,
            rchild: i * 2 + 2,
            ..Default::default()
        });
    }
    for i in 0..32 {
        tree.push(DcTreeNode {
            property: -1,
            context_id: i,
            predictor: 5,
            ..Default::default()
        });
    }

    let (_, total_contexts, dc_remap, ac_map) = tree_tokens_with_ac_metadata_prefix(&tree, 32, 1);
    eprintln!(
        "Deep DC: total={}, dc_remap={:?}, ac_map={:?}",
        total_contexts, dc_remap, ac_map
    );

    // No padding needed → no dummies → AC metadata contexts are exactly 0-10
    assert_eq!(total_contexts, 43); // 11 AC meta + 32 DC
    assert_eq!(dc_remap.len(), 32);
    // All DC contexts should be >= 11 and unique
    let mut dc_set = std::collections::HashSet::new();
    for (i, &bfs) in dc_remap.iter().enumerate() {
        assert!(bfs >= 11, "DC remap[{}]={} < 11", i, bfs);
        assert!(
            bfs < total_contexts,
            "DC remap[{}]={} >= total {}",
            i,
            bfs,
            total_contexts
        );
        assert!(
            dc_set.insert(bfs),
            "Duplicate DC BFS context {} at remap[{}]",
            bfs,
            i
        );
    }
    for i in 0..11u32 {
        assert_eq!(
            ac_map[i as usize], i,
            "AC meta {} not at expected BFS position",
            i
        );
    }
}
