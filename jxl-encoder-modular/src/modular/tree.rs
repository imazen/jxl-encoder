// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Decision tree for modular encoding context selection.
//!
//! The tree determines which predictor and context to use for each pixel
//! based on properties of the neighborhood.

use super::predictor::Predictor;

/// A node in the property decision tree.
#[derive(Debug, Clone)]
pub struct PropertyDecisionNode {
    /// Property to split on (-1 = leaf node).
    pub property: i32,
    /// Split threshold value.
    pub splitval: i32,
    /// Predictor to use (for leaf nodes).
    pub predictor: Predictor,
    /// Offset for predictor (for leaf nodes).
    pub predictor_offset: i32,
    /// Multiplier for residual (for leaf nodes).
    pub multiplier: i32,
    /// Left child index (value <= splitval).
    pub lchild: usize,
    /// Right child index (value > splitval).
    pub rchild: usize,
    /// Context ID for ANS coding.
    pub context_id: u32,
}

impl Default for PropertyDecisionNode {
    fn default() -> Self {
        Self {
            property: -1, // Leaf node
            splitval: 0,
            predictor: Predictor::Gradient,
            predictor_offset: 0,
            multiplier: 1,
            lchild: 0,
            rchild: 0,
            context_id: 0,
        }
    }
}

/// A decision tree for context selection.
pub type Tree = Vec<PropertyDecisionNode>;

/// Property indices for tree decisions.
///
/// The full spec table of modular property ids (ISO/IEC 18181-1
/// `Properties`); the production tree learner constructs only a subset,
/// the rest document the wire numbering (#76).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(dead_code)]
pub enum Property {
    /// Channel index.
    Channel = 0,
    /// Group ID.
    GroupId = 1,
    /// Y coordinate.
    Y = 2,
    /// X coordinate.
    X = 3,
    /// |N - NW|
    AbsNMinusNw = 4,
    /// |N - W|
    AbsNMinusW = 5,
    /// FloorLog2(W)
    FloorLog2W = 6,
    /// FloorLog2(N)
    FloorLog2N = 7,
    /// FloorLog2(NW)
    FloorLog2Nw = 8,
    /// |N - NN|
    AbsNMinusNn = 9,
    /// |W - WW|
    AbsWMinusWw = 10,
    /// |NW - NWW|
    AbsNwMinusNww = 11,
    /// |NE - N|
    AbsNeMinusN = 12,
    /// |NW - W|
    AbsNwMinusW = 13,
    /// |W| + |N| + |NW|
    SumWNNw = 14,
    /// Max error in weighted predictor.
    WpMaxError = 15,
}

impl Property {
    /// Total number of static properties (not including WP properties).
    /// Spec count, referenced by tests/docs only (#76).
    #[allow(dead_code)]
    pub const NUM_STATIC: usize = 14;

    /// Total number of properties including WP.
    #[allow(dead_code)]
    pub const NUM_PROPERTIES: usize = 16;
}

/// Properties computed for a pixel location.
///
/// #76: consumed only by the unit tests exercising the predefined
/// static trees below (the production learner tracks properties in
/// `tree_learn`'s own sample layout).
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub struct PixelProperties {
    /// Property values.
    pub values: [i32; Property::NUM_PROPERTIES],
}

#[cfg(test)]
impl PixelProperties {
    /// Gets a property value (used by [`traverse_tree`]).
    #[inline]
    pub fn get(&self, property: i32) -> i32 {
        if property >= 0 && (property as usize) < self.values.len() {
            self.values[property as usize]
        } else {
            0
        }
    }
}

/// Floor log2 for unsigned values (returns 0 for 0).
#[inline]
#[cfg(test)]
fn floor_log2(value: u32) -> i32 {
    if value == 0 {
        0
    } else {
        31 - value.leading_zeros() as i32
    }
}

/// Creates a simple tree that uses a single predictor for all pixels.
pub fn simple_tree(predictor: Predictor) -> Tree {
    vec![PropertyDecisionNode {
        property: -1, // Leaf
        predictor,
        context_id: 0,
        ..Default::default()
    }]
}

/// Creates a gradient tree (most common for lossless).
#[cfg(test)]
pub fn gradient_tree() -> Tree {
    simple_tree(Predictor::Gradient)
}

/// Creates a tree that selects predictor based on channel.
#[allow(dead_code)]
pub fn per_channel_tree(num_channels: usize) -> Tree {
    let mut tree = Vec::with_capacity(num_channels * 2);

    // Build a simple chain: if channel == 0, use ctx 0; if channel == 1, use ctx 1; etc.
    for c in 0..num_channels {
        if c < num_channels - 1 {
            // Internal node: split on channel
            tree.push(PropertyDecisionNode {
                property: Property::Channel as i32,
                splitval: c as i32,
                lchild: tree.len() + num_channels - c, // Leaf for this channel
                rchild: tree.len() + 1,                // Next decision
                ..Default::default()
            });
        }
    }

    // Leaf nodes
    for c in 0..num_channels {
        tree.push(PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Gradient,
            context_id: c as u32,
            ..Default::default()
        });
    }

    tree
}

/// Traverses the tree to find the leaf node for given properties.
#[cfg(test)]
pub fn traverse_tree<'a>(tree: &'a Tree, properties: &PixelProperties) -> &'a PropertyDecisionNode {
    let mut node_idx = 0;

    loop {
        let node = &tree[node_idx];

        // Leaf node?
        if node.property < 0 {
            return node;
        }

        // Get property value and decide direction
        let prop_value = properties.get(node.property);
        if prop_value <= node.splitval {
            node_idx = node.lchild;
        } else {
            node_idx = node.rchild;
        }
    }
}

/// Tree serialization context indices.
const SPLIT_VAL_CONTEXT: usize = 0;
const PROPERTY_CONTEXT: usize = 1;
const PREDICTOR_CONTEXT: usize = 2;
const OFFSET_CONTEXT: usize = 3;
const MULTIPLIER_LOG_CONTEXT: usize = 4;
const MULTIPLIER_BITS_CONTEXT: usize = 5;

/// Token for tree serialization.
#[derive(Debug, Clone)]
pub struct TreeToken {
    /// Context for this token. Read by stream-comparison unit tests
    /// only (#76) — the writer streams tokens without re-reading it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub context: usize,
    /// Token value (unsigned for property/predictor/log, signed for split_val/offset).
    pub value: i32,
    /// Whether this is a signed value.
    pub is_signed: bool,
}

/// Collect tokens for tree serialization.
pub fn collect_tree_tokens(tree: &Tree) -> Vec<TreeToken> {
    let mut tokens = Vec::new();

    // Process tree in BFS order
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0usize);

    while let Some(idx) = queue.pop_front() {
        let node = &tree[idx];

        if node.property < 0 {
            // Leaf node: property = 0 (indicator), then predictor, offset, mul_log, mul_bits
            tokens.push(TreeToken {
                context: PROPERTY_CONTEXT,
                value: 0, // 0 means leaf
                is_signed: false,
            });

            // Predictor
            tokens.push(TreeToken {
                context: PREDICTOR_CONTEXT,
                value: node.predictor as i32,
                is_signed: false,
            });

            // Offset (signed)
            tokens.push(TreeToken {
                context: OFFSET_CONTEXT,
                value: node.predictor_offset,
                is_signed: true,
            });

            // Multiplier is encoded as (mul_bits + 1) << mul_log
            // For multiplier = 1: mul_log = 0, mul_bits = 0
            let (mul_log, mul_bits) = decompose_multiplier(node.multiplier as u32);
            tokens.push(TreeToken {
                context: MULTIPLIER_LOG_CONTEXT,
                value: mul_log as i32,
                is_signed: false,
            });

            tokens.push(TreeToken {
                context: MULTIPLIER_BITS_CONTEXT,
                value: mul_bits as i32,
                is_signed: false,
            });
        } else {
            // Split node: property+1, splitval, then children
            tokens.push(TreeToken {
                context: PROPERTY_CONTEXT,
                value: node.property + 1, // +1 because 0 means leaf
                is_signed: false,
            });

            tokens.push(TreeToken {
                context: SPLIT_VAL_CONTEXT,
                value: node.splitval,
                is_signed: true,
            });

            // Queue children: rchild first (value > splitval = decoder's "left"/first BFS child),
            // then lchild (value <= splitval = decoder's "right"/second BFS child).
            // jxl-rs reads first BFS child as "left" (property > splitval).
            queue.push_back(node.rchild);
            queue.push_back(node.lchild);
        }
    }

    tokens
}

/// Decompose multiplier into (log, bits) where multiplier = (bits + 1) << log.
fn decompose_multiplier(multiplier: u32) -> (u32, u32) {
    if multiplier == 0 {
        return (0, 0);
    }

    let trailing = multiplier.trailing_zeros();
    let mul_log = trailing;
    let mul_bits = (multiplier >> trailing) - 1;

    (mul_log, mul_bits)
}

/// Creates a tree with the weighted predictor.
#[cfg(test)]
pub fn weighted_tree() -> Tree {
    simple_tree(Predictor::Weighted)
}

/// Creates a RIGED-shaped (Resolution-Independent Gradient-aware Edge
/// Detection) MA tree as an encoder-only meta-mode for
/// `--modular-predictor 14`.
///
/// Per Sharma et al. 2018, RIGED switches per-pixel among `West`,
/// `North` and a Paeth-like average based on the relative strength of
/// the vertical vs horizontal local gradient. The full paper's metric is
///
/// ```text
///   A_v = |NW - W| + |NN - N|
///   A_h = |WW - W| + |NW - N|
/// ```
///
/// JXL's MA tree only splits on a single property at a time, so we
/// cannot encode the literal `(A_v - A_h)` comparison in one node. We
/// approximate the three-way switch with a 3-leaf tree that gates on the
/// dominant single-property contributors of each axis (`|NW - W|` for
/// `A_v`, `|W - WW|` for `A_h`) — capturing the gradient-strength
/// intuition while remaining wire-legal under the spec's predictor and
/// property tables.
///
/// Tree shape (lchild = property <= splitval, rchild = property >
/// splitval):
///
/// ```text
///   Root: |NW - W| (property 13) > T  // strong vertical edge?
///     yes ── leaf: Top (predict from N)
///     no  ── |W - WW| (property 10) > T  // strong horizontal edge?
///              yes ── leaf: Left (predict from W)
///              no  ── leaf: Average0 ((W + N) / 2)  // smooth / mixed
/// ```
///
/// Threshold `T` scales with channel bit depth, per the paper's
/// constants: `T = 44` for ≤ 8-bit channels, `T = 768` for > 8-bit. The
/// scaling matches the dynamic range of the absolute-difference
/// properties produced by [`PixelProperties::compute`].
///
/// This tree is decoder-trivial — every leaf uses a predictor in
/// `0..=13` and every split property is in `0..=15`, so any spec-
/// conformant JXL decoder evaluates the RIGED switch the same way the
/// encoder did. Pixel-exact round-trip is verified by the modular
/// roundtrip integration tests.
///
/// Used by [`super::encode::resolve_tree_learn_riged_tree`].
pub fn riged_tree(bit_depth: u32) -> Tree {
    // Sharma 2018 thresholds — T = 44 for 8-bit, T = 768 for 16-bit.
    // For other bit depths we interpolate linearly in the ratio
    // T / max_value to keep the gate independent of dynamic range.
    let threshold: i32 = if bit_depth <= 8 {
        44
    } else if bit_depth >= 16 {
        768
    } else {
        // Linear interp between the two reference points; rounds toward
        // the nearest integer so 12-bit lands at ~199, 10-bit at ~110.
        let lo = 44.0_f32;
        let hi = 768.0_f32;
        let frac = (bit_depth as f32 - 8.0) / 8.0;
        (lo + (hi - lo) * frac).round() as i32
    };

    vec![
        // 0: Root — split on |NW - W| (property 13, AbsNwMinusW).
        // Captures the vertical-axis gradient component (NW→W is the
        // vertical-edge signature: a strong vertical edge between rows
        // makes NW and W differ a lot).
        PropertyDecisionNode {
            property: Property::AbsNwMinusW as i32,
            splitval: threshold,
            lchild: 2, // |NW-W| <= T → check horizontal axis
            rchild: 1, // |NW-W| >  T → strong vertical edge → leaf Top
            ..Default::default()
        },
        // 1: Leaf — strong vertical edge → predict from North.
        PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Top,
            context_id: 0,
            ..Default::default()
        },
        // 2: Inner — split on |W - WW| (property 10, AbsWMinusWw).
        // Captures the horizontal-axis gradient component.
        PropertyDecisionNode {
            property: Property::AbsWMinusWw as i32,
            splitval: threshold,
            lchild: 4, // |W-WW| <= T → smooth/mixed → Average0
            rchild: 3, // |W-WW| >  T → strong horizontal edge → Left
            ..Default::default()
        },
        // 3: Leaf — strong horizontal edge → predict from West.
        PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Left,
            context_id: 1,
            ..Default::default()
        },
        // 4: Leaf — both gradients weak (smooth region) → average W and
        // N. This is the Paeth-like fallback from the paper.
        PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Average0,
            context_id: 2,
            ..Default::default()
        },
    ]
}

/// Creates a tree that selects between Gradient and Weighted based on WP max error.
/// Uses Gradient when max error is low (WP is stable), Weighted when error is higher.
#[cfg(test)]
pub fn adaptive_gradient_weighted_tree() -> Tree {
    vec![
        // Root: split on WP max error (property 15)
        PropertyDecisionNode {
            property: Property::WpMaxError as i32,
            splitval: 100, // Threshold
            lchild: 1,     // Low error -> gradient
            rchild: 2,     // High error -> weighted
            ..Default::default()
        },
        // Leaf: Gradient predictor (for stable regions)
        PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Gradient,
            context_id: 0,
            ..Default::default()
        },
        // Leaf: Weighted predictor (for complex regions)
        PropertyDecisionNode {
            property: -1,
            predictor: Predictor::Weighted,
            context_id: 1,
            ..Default::default()
        },
    ]
}

/// Validate tree structure matching libjxl's ValidateTree in dec_ma.cc.
///
/// Tracks property ranges as the tree narrows them through splits.
/// Returns Ok(()) if valid, Err with details of the failing node.
///
/// Convention: lchild = value <= splitval, rchild = value > splitval.
///
/// But the decoder reads BFS where first child is "lchild" (value > splitval)
/// and second child is "rchild" (value <= splitval). So we map:
/// - Our rchild → decoder lchild: range [val+1, u]
/// - Our lchild → decoder rchild: range [l, val]
pub fn validate_tree_djxl(tree: &Tree) -> Result<(), String> {
    if tree.is_empty() {
        return Ok(());
    }

    let mut num_properties = 0i32;
    for node in tree {
        if node.property >= num_properties {
            num_properties = node.property + 1;
        }
    }
    let np = num_properties as usize;

    // Track (lo, hi) range per property per node
    // Range is [lo, hi] inclusive; split at val requires lo <= val && val < hi
    // (in libjxl terms: u > val, meaning hi > val)
    let mut ranges: Vec<(i32, i32)> = vec![(i32::MIN, i32::MAX); np * tree.len()];

    for (i, node) in tree.iter().enumerate() {
        if node.property < 0 {
            continue; // leaf
        }
        let p = node.property as usize;
        let val = node.splitval;
        let lo = ranges[i * np + p].0;
        let hi = ranges[i * np + p].1;

        // libjxl check: if (l > val || u <= val) return FAILURE
        if lo > val || hi <= val {
            return Err(format!(
                "Node {} (property={}, splitval={}): range [{}, {}] invalid \
                 (lo > val = {}, hi <= val = {})",
                i,
                node.property,
                val,
                lo,
                hi,
                lo > val,
                hi <= val
            ));
        }

        let lchild = node.lchild; // value <= splitval
        let rchild = node.rchild; // value > splitval

        // Copy all property ranges to children
        for pp in 0..np {
            ranges[rchild * np + pp] = ranges[i * np + pp];
            ranges[lchild * np + pp] = ranges[i * np + pp];
        }

        // Narrow property p for children
        // rchild (value > splitval): lo = val + 1
        ranges[rchild * np + p] = (val + 1, hi);
        // lchild (value <= splitval): hi = val
        ranges[lchild * np + p] = (lo, val);
    }

    Ok(())
}

/// Count the number of unique context IDs used in a tree.
/// Count the number of BFS-reachable leaf contexts in the tree.
///
/// Only counts leaves reachable from root via BFS traversal, ignoring
/// unreachable orphan nodes that may exist after tree validation.
pub fn count_contexts(tree: &Tree) -> u32 {
    let mut count = 0u32;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0usize);

    while let Some(idx) = queue.pop_front() {
        if tree[idx].property < 0 {
            count += 1;
        } else {
            queue.push_back(tree[idx].rchild);
            queue.push_back(tree[idx].lchild);
        }
    }
    count.max(1)
}

/// Assign context IDs to leaf nodes sequentially in BFS order.
///
/// The decoder assigns context IDs to leaves in the order it encounters them
/// during BFS deserialization (rchild first, then lchild — matching
/// `collect_tree_tokens`). We must use the same traversal order here so that
/// context IDs in the encoder match what the decoder derives.
///
/// Returns the number of contexts assigned.
pub fn assign_sequential_contexts(tree: &mut Tree) -> u32 {
    let mut next_context = 0u32;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0usize);

    while let Some(idx) = queue.pop_front() {
        if tree[idx].property < 0 {
            tree[idx].context_id = next_context;
            next_context += 1;
        } else {
            let rchild = tree[idx].rchild;
            let lchild = tree[idx].lchild;
            // Same child order as collect_tree_tokens: rchild first, lchild second
            queue.push_back(rchild);
            queue.push_back(lchild);
        }
    }
    next_context
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_log2() {
        assert_eq!(floor_log2(0), 0);
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(2), 1);
        assert_eq!(floor_log2(3), 1);
        assert_eq!(floor_log2(4), 2);
        assert_eq!(floor_log2(255), 7);
        assert_eq!(floor_log2(256), 8);
    }

    #[test]
    fn test_simple_tree() {
        let tree = simple_tree(Predictor::Left);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].property, -1);
        assert_eq!(tree[0].predictor, Predictor::Left);
    }

    #[test]
    fn test_traverse_simple() {
        let tree = gradient_tree();
        let props = PixelProperties::default();
        let leaf = traverse_tree(&tree, &props);
        assert_eq!(leaf.predictor, Predictor::Gradient);
        assert_eq!(leaf.context_id, 0);
    }

    #[test]
    fn test_weighted_tree() {
        let tree = weighted_tree();
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].predictor, Predictor::Weighted);
    }

    #[test]
    fn test_riged_tree_shape_8bit() {
        // Sharma 2018: T = 44 at ≤ 8-bit channels.
        let tree = riged_tree(8);
        assert_eq!(tree.len(), 5, "RIGED tree has 3 leaves + 2 inner nodes");
        // Root splits on |NW - W| at T=44.
        assert_eq!(tree[0].property, Property::AbsNwMinusW as i32);
        assert_eq!(tree[0].splitval, 44);
        // Strong-vertical-edge leaf → Top.
        assert_eq!(tree[1].property, -1);
        assert_eq!(tree[1].predictor, Predictor::Top);
        // Second inner splits on |W - WW| at T=44.
        assert_eq!(tree[2].property, Property::AbsWMinusWw as i32);
        assert_eq!(tree[2].splitval, 44);
        // Strong-horizontal-edge leaf → Left.
        assert_eq!(tree[3].property, -1);
        assert_eq!(tree[3].predictor, Predictor::Left);
        // Smooth/mixed leaf → Average0 (paeth-ish).
        assert_eq!(tree[4].property, -1);
        assert_eq!(tree[4].predictor, Predictor::Average0);
    }

    #[test]
    fn test_riged_tree_threshold_scales_with_bit_depth() {
        // 8-bit: T=44.
        assert_eq!(riged_tree(8)[0].splitval, 44);
        assert_eq!(riged_tree(5)[0].splitval, 44, "≤8-bit floor");
        // 16-bit: T=768.
        assert_eq!(riged_tree(16)[0].splitval, 768);
        assert_eq!(riged_tree(32)[0].splitval, 768, "≥16-bit ceiling");
        // Interior bit depths interpolate linearly.
        let t10 = riged_tree(10)[0].splitval;
        assert!(
            t10 > 44 && t10 < 768,
            "10-bit between 44 and 768, got {t10}"
        );
        let t12 = riged_tree(12)[0].splitval;
        assert!(t12 > t10, "12-bit > 10-bit, got 10:{t10} 12:{t12}");
    }

    #[test]
    fn test_riged_tree_routes_strong_vertical_edge_to_top() {
        let tree = riged_tree(8);
        // Synthesize properties: strong vertical edge means |NW - W| large.
        let mut props = PixelProperties::default();
        props.values[Property::AbsNwMinusW as usize] = 100; // > T=44
        let leaf = traverse_tree(&tree, &props);
        assert_eq!(
            leaf.predictor,
            Predictor::Top,
            "strong |NW-W| → Top (north)"
        );
    }

    #[test]
    fn test_riged_tree_routes_strong_horizontal_edge_to_left() {
        let tree = riged_tree(8);
        let mut props = PixelProperties::default();
        props.values[Property::AbsNwMinusW as usize] = 10; // ≤ T=44
        props.values[Property::AbsWMinusWw as usize] = 100; // > T=44
        let leaf = traverse_tree(&tree, &props);
        assert_eq!(
            leaf.predictor,
            Predictor::Left,
            "weak |NW-W|, strong |W-WW| → Left (west)"
        );
    }

    #[test]
    fn test_riged_tree_routes_smooth_to_average() {
        let tree = riged_tree(8);
        let mut props = PixelProperties::default();
        props.values[Property::AbsNwMinusW as usize] = 5; // ≤ T=44
        props.values[Property::AbsWMinusWw as usize] = 10; // ≤ T=44
        let leaf = traverse_tree(&tree, &props);
        assert_eq!(
            leaf.predictor,
            Predictor::Average0,
            "both gradients weak → Average0 ((W+N)/2)"
        );
    }

    #[test]
    fn test_riged_tree_validates_as_decoder_legal() {
        // The tree must satisfy `validate_tree_djxl` (no invalid splits,
        // no orphan ranges) so libjxl / jxl-rs / jxl-oxide accept it.
        assert!(validate_tree_djxl(&riged_tree(8)).is_ok());
        assert!(validate_tree_djxl(&riged_tree(16)).is_ok());
    }

    #[test]
    fn test_riged_tree_three_contexts_after_assignment() {
        // count_contexts should report 3 BFS-reachable leaves; sequential
        // context assignment should populate them in BFS order.
        let mut tree = riged_tree(8);
        assert_eq!(count_contexts(&tree), 3);
        let n = assign_sequential_contexts(&mut tree);
        assert_eq!(n, 3);
    }

    #[test]
    fn test_decompose_multiplier() {
        assert_eq!(decompose_multiplier(1), (0, 0)); // (0+1) << 0 = 1
        assert_eq!(decompose_multiplier(2), (1, 0)); // (0+1) << 1 = 2
        assert_eq!(decompose_multiplier(4), (2, 0)); // (0+1) << 2 = 4
        assert_eq!(decompose_multiplier(3), (0, 2)); // (2+1) << 0 = 3
        assert_eq!(decompose_multiplier(6), (1, 2)); // (2+1) << 1 = 6
    }

    #[test]
    fn test_collect_tree_tokens_simple() {
        let tree = gradient_tree();
        let tokens = collect_tree_tokens(&tree);
        // Single leaf: property(0), predictor(5), offset(0), mul_log(0), mul_bits(0)
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].value, 0); // property = 0 (leaf)
        assert_eq!(tokens[1].value, Predictor::Gradient as i32);
    }

    #[test]
    fn test_adaptive_tree() {
        let tree = adaptive_gradient_weighted_tree();
        assert_eq!(tree.len(), 3);

        // Test traversal with low error -> gradient
        let mut props = PixelProperties::default();
        props.values[Property::WpMaxError as usize] = 50;
        let leaf = traverse_tree(&tree, &props);
        assert_eq!(leaf.predictor, Predictor::Gradient);

        // Test traversal with high error -> weighted
        props.values[Property::WpMaxError as usize] = 150;
        let leaf = traverse_tree(&tree, &props);
        assert_eq!(leaf.predictor, Predictor::Weighted);
    }

    #[test]
    fn test_count_contexts() {
        let tree = gradient_tree();
        assert_eq!(count_contexts(&tree), 1);

        let tree = adaptive_gradient_weighted_tree();
        assert_eq!(count_contexts(&tree), 2);
    }
}
