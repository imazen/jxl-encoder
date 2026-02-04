// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
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
    pub const NUM_STATIC: usize = 14;

    /// Total number of properties including WP.
    pub const NUM_PROPERTIES: usize = 16;
}

/// Properties computed for a pixel location.
#[derive(Debug, Clone, Default)]
pub struct PixelProperties {
    /// Property values.
    pub values: [i32; Property::NUM_PROPERTIES],
}

impl PixelProperties {
    /// Computes properties for a pixel.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        channel_idx: u32,
        group_id: u32,
        x: usize,
        y: usize,
        n: i32,
        w: i32,
        nw: i32,
        ne: i32,
        nn: i32,
        ww: i32,
        nww: i32,
    ) -> Self {
        let mut values = [0i32; Property::NUM_PROPERTIES];

        values[Property::Channel as usize] = channel_idx as i32;
        values[Property::GroupId as usize] = group_id as i32;
        values[Property::Y as usize] = y as i32;
        values[Property::X as usize] = x as i32;
        values[Property::AbsNMinusNw as usize] = (n - nw).abs();
        values[Property::AbsNMinusW as usize] = (n - w).abs();
        values[Property::FloorLog2W as usize] = floor_log2(w.unsigned_abs());
        values[Property::FloorLog2N as usize] = floor_log2(n.unsigned_abs());
        values[Property::FloorLog2Nw as usize] = floor_log2(nw.unsigned_abs());
        values[Property::AbsNMinusNn as usize] = (n - nn).abs();
        values[Property::AbsWMinusWw as usize] = (w - ww).abs();
        values[Property::AbsNwMinusNww as usize] = (nw - nww).abs();
        values[Property::AbsNeMinusN as usize] = (ne - n).abs();
        values[Property::AbsNwMinusW as usize] = (nw - w).abs();
        values[Property::SumWNNw as usize] = w.abs() + n.abs() + nw.abs();
        values[Property::WpMaxError as usize] = 0; // Filled in by WP state

        Self { values }
    }

    /// Gets a property value.
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
pub fn gradient_tree() -> Tree {
    simple_tree(Predictor::Gradient)
}

/// Creates a tree that selects predictor based on channel.
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
    /// Context for this token.
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
pub fn weighted_tree() -> Tree {
    simple_tree(Predictor::Weighted)
}

/// Creates a tree that selects between Gradient and Weighted based on WP max error.
/// Uses Gradient when max error is low (WP is stable), Weighted when error is higher.
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

/// Count the number of unique context IDs used in a tree.
pub fn count_contexts(tree: &Tree) -> u32 {
    tree.iter()
        .filter(|n| n.property < 0)
        .map(|n| n.context_id)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1)
}

/// Assign context IDs to leaf nodes sequentially.
pub fn assign_sequential_contexts(tree: &mut Tree) {
    let mut next_context = 0u32;
    for node in tree.iter_mut() {
        if node.property < 0 {
            node.context_id = next_context;
            next_context += 1;
        }
    }
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
