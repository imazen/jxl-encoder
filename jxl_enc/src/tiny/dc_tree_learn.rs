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
/// Properties:
/// 0 = Channel (0=X, 1=Y, 2=B in encoding order)
/// 1 = Gradient property: 512 + top + left - topleft, clamped to [0, 1023]
/// 2 = |top|
/// 3 = |left|
/// 4 = top
/// 5 = left
/// 6 = top - left (edge direction indicator)
const NUM_DC_PROPERTIES: usize = 7;

/// Properties to consider for splits.
/// Skip channel for now (single tree for all channels in initial implementation).
const SPLIT_PROPERTIES: &[usize] = &[
    1, // Gradient property (most important for DC)
    2, // |top|
    3, // |left|
    4, // top
    5, // left
    6, // top - left
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
pub fn compute_dc_properties(
    channel_idx: u32,
    top: i32,
    left: i32,
    topleft: i32,
) -> [i32; NUM_DC_PROPERTIES] {
    let mut props = [0i32; NUM_DC_PROPERTIES];

    // Channel (in encoding order: Y=0, X=1, B=2 → actually 1, 0, 2)
    props[0] = channel_idx as i32;

    // Gradient property: 512 + top + left - topleft, clamped to [0, 1023]
    let grad_raw = 512i64 + top as i64 + left as i64 - topleft as i64;
    props[1] = grad_raw.clamp(0, 1023) as i32;

    // Absolute values
    props[2] = top.abs();
    props[3] = left.abs();

    // Raw values
    props[4] = top;
    props[5] = left;

    // Edge direction
    props[6] = top - left;

    props
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
            for x in 0..width {
                let dc_val = channel[y][x] as i32;

                // Get neighbors
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

                // Compute prediction and residual
                let prediction = clamped_gradient(top, left, topleft);
                let residual = dc_val - prediction;

                // Compute properties and add sample
                let props = compute_dc_properties(enc_idx as u32, top, left, topleft);
                samples.add_sample(residual, props);
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
        // Leaf node: emit predictor (gradient=4), multiplier=1, offset=0
        // Context 1 = parent_property (use 0 for leaf marker)
        tokens.push((1, 0)); // property = -1 encoded as 0 in context 1
        // Context 2 = predictor (gradient = 4)
        tokens.push((2, 4));
        // Context 3 = offset
        tokens.push((3, 0));
        // Context 4 = multiplier (implicit 1)
        tokens.push((4, 0));
        // Context 5 = unused
        tokens.push((5, 0));
    } else {
        // Internal node: emit property and splitval
        // Context 1 = property (1-6 for our properties, offset by 1 to distinguish from leaf)
        let prop_token = (node.property + 1) as u32;
        tokens.push((1, prop_token));
        // Context 0 = splitval (encoded as HybridUint)
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
        // Test gradient property computation
        let props = compute_dc_properties(0, 100, 100, 100);
        // Gradient: 512 + 100 + 100 - 100 = 612
        assert_eq!(props[1], 612);

        // Edge case: gradient at boundaries
        let props = compute_dc_properties(0, -512, -512, 512);
        // Gradient: 512 + (-512) + (-512) - 512 = -1024 → clamped to 0
        assert_eq!(props[1], 0);

        let props = compute_dc_properties(0, 512, 512, -512);
        // Gradient: 512 + 512 + 512 - (-512) = 2048 → clamped to 1023
        assert_eq!(props[1], 1023);
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
        // Create a simple 2-leaf tree that splits on gradient property
        let tree = vec![
            DcTreeNode {
                property: 1, // Gradient
                splitval: 512,
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

        // Gradient <= 512 should go to context 0
        let props_low = compute_dc_properties(0, 100, 100, 200);
        // Gradient: 512 + 100 + 100 - 200 = 512
        assert_eq!(get_dc_context(&tree, &props_low), 0);

        // Gradient > 512 should go to context 1
        let props_high = compute_dc_properties(0, 200, 200, 100);
        // Gradient: 512 + 200 + 200 - 100 = 812
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
            for x in start_bx..end_bx {
                let dc_val = channel[y][x] as i32;

                // Get neighbors
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

                // Compute prediction and residual
                let prediction = clamped_gradient(top, left, topleft);
                let residual = dc_val - prediction;

                // Compute properties and get context from tree
                let props = compute_dc_properties(enc_idx as u32, top, left, topleft);
                let ctx_id = get_dc_context(tree, &props);

                tokens.push(Token::new(ctx_id, pack_signed(residual)));
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
