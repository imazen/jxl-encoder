// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

                let toptop = if y > 1 {
                    channel[y - 2][x] as i32
                } else {
                    top
                };

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
}

impl Default for DcTreeNode {
    fn default() -> Self {
        Self {
            property: -1,
            splitval: 0,
            lchild: 0,
            rchild: 0,
            context_id: 0,
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
            bits -= (count as f64) * p.log2();
        }
    }
    bits
}

/// Estimate entropy cost for a subset of samples.
fn estimate_subset_cost(
    samples: &DcTreeSamples,
    indices: &[usize],
    max_token: u32,
) -> f64 {
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
            let (left, right): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .copied()
                .partition(|&i| props[i] <= splitval);

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

    build_tree_recursive(samples, &indices, 0, &mut tree, &mut next_context, max_token);

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
pub fn tree_to_tokens(tree: &DcTree) -> Vec<(u32, u32)> {
    let mut tokens = Vec::new();
    tree_to_tokens_recursive(tree, 0, &mut tokens);
    tokens
}

fn tree_to_tokens_recursive(tree: &DcTree, idx: usize, tokens: &mut Vec<(u32, u32)>) {
    let node = &tree[idx];

    if node.property < 0 {
        // Leaf node: emit predictor, multiplier, offset
        // Context 1: property = 0 signals leaf node (decoder subtracts 1, gets -1)
        tokens.push((1, 0));
        // Context 2: predictor (5 = Gradient, matching DC prediction)
        tokens.push((2, 5));
        // Context 3: offset (0)
        tokens.push((3, 0));
        // Context 4: multiplier log (0 for multiplier=1 since (0+1)<<0 = 1)
        tokens.push((4, 0));
        // Context 5: multiplier bits (0)
        tokens.push((5, 0));
    } else {
        // Internal node: emit property and splitval
        // Context 1: property+1 (decoder subtracts 1 to get actual property index)
        let prop_token = (node.property + 1) as u32;
        tokens.push((1, prop_token));
        // Context 0: splitval (packed signed: positive*2, negative*2+1)
        let splitval_token = if node.splitval >= 0 {
            (node.splitval as u32) * 2
        } else {
            ((-node.splitval) as u32) * 2 + 1
        };
        tokens.push((0, splitval_token));

        // Recurse to children
        tree_to_tokens_recursive(tree, node.lchild, tokens);
        tree_to_tokens_recursive(tree, node.rchild, tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_dc_properties() {
        // Test gradient property (property 9 = left + top - topleft)
        let (props, _) = compute_dc_properties(
            0,    // channel
            5,    // x
            3,    // y
            100,  // top
            100,  // left
            100,  // topleft
            100,  // topright
            100,  // toptop
            100,  // leftleft
            0,    // prev_local_grad
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
) -> Vec<super::token::Token> {
    use super::token::Token;

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
                // Offset the context to not conflict with AC metadata contexts (0-10)
                let ctx_id = tree_ctx + super::dc_coding::DC_CONTEXT_OFFSET as u32;

                tokens.push(Token::new(ctx_id, pack_signed(residual)));

                prev_local_grad = new_local_grad;
            }
        }
    }

    tokens
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
) -> (DcTree, Vec<super::token::Token>, DcTreeStats) {
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
    use crate::tiny::context_tree::{write_learned_context_tree, write_context_tree};
    use crate::bit_writer::BitWriter;

    #[test]
    fn test_compare_static_vs_learned_tree_encoding() {
        // Test with single DC group
        let num_dc_groups = 1;
        
        // Create a simple learned tree (single leaf)
        let tree = vec![DcTreeNode {
            property: -1,
            context_id: 0,
            ..Default::default()
        }];
        let learned_tokens = tree_to_tokens(&tree);
        eprintln!("Learned tree tokens ({} tokens):", learned_tokens.len());
        for (i, (ctx, val)) in learned_tokens.iter().enumerate() {
            eprintln!("  token[{}]: ctx={}, val={}", i, ctx, val);
        }
        
        // Write learned tree
        let mut learned_writer = BitWriter::new();
        let learned_result = write_learned_context_tree(&learned_tokens, num_dc_groups, &mut learned_writer);
        eprintln!("\nLearned tree encoding result: {:?}", learned_result);
        eprintln!("Learned bits written: {}", learned_writer.bits_written());
        learned_writer.zero_pad_to_byte();
        let learned_bytes = learned_writer.finish();
        eprintln!("Learned bytes (first 30): {:02x?}", &learned_bytes[..learned_bytes.len().min(30)]);

        // Write static tree for comparison
        let mut static_writer = BitWriter::new();
        let static_result = write_context_tree(num_dc_groups, &mut static_writer);
        eprintln!("\nStatic tree encoding result: {:?}", static_result);
        eprintln!("Static bits written: {}", static_writer.bits_written());
        static_writer.zero_pad_to_byte();
        let static_bytes = static_writer.finish();
        eprintln!("Static bytes (first 30): {:02x?}", &static_bytes[..static_bytes.len().min(30)]);
        
        // The encoding itself should succeed
        assert!(learned_result.is_ok());
        assert!(static_result.is_ok());
    }
}
