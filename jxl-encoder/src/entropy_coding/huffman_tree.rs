// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Full Huffman tree encoder supporting arbitrary alphabet sizes.
//!
//! This module ports libjxl's `enc_huffman_tree.cc` and parts of `enc_huffman.cc`.
//! It supports encoding Huffman tables with more than 4 symbols using the
//! code length table format with RLE compression.
//!
//! # Architecture
//! 1. `create_huffman_tree` - builds optimal tree, outputs depth array
//! 2. `convert_bit_depths_to_symbols` - converts depths to canonical codes
//! 3. `write_huffman_tree` - RLE-compresses depth array for storage
//! 4. `store_huffman_tree` - writes compressed tree to bitstream

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Number of code length codes (0-15 for depths, 16-17 for RLE).
const CODE_LENGTH_CODES: usize = 18;

/// Maximum code depth for regular Huffman codes.
const MAX_CODE_DEPTH: u8 = 15;

/// Maximum code depth for code length codes (meta-Huffman).
const MAX_CODE_LENGTH_DEPTH: u8 = 5;

/// Order in which code length codes are stored.
/// This ordering groups commonly-used codes together to allow truncation.
///
/// ```cpp
/// // libjxl: enc_huffman.cc
/// static const uint8_t kStorageOrder[kCodeLengthCodes] = {
///     1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15};
/// ```
const STORAGE_ORDER: [usize; CODE_LENGTH_CODES] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Static Huffman code for encoding code length code depths.
/// Maps depth values 0-5 to their codes.
///
/// ```cpp
/// // libjxl: enc_huffman.cc
/// // Symbol   Code
/// // ------   ----
/// // 0          00
/// // 1        1110
/// // 2         110
/// // 3          01
/// // 4          10
/// // 5        1111
/// static const uint8_t kHuffmanBitLengthHuffmanCodeSymbols[6] = {0, 7, 3, 2, 1, 15};
/// static const uint8_t kHuffmanBitLengthHuffmanCodeBitLengths[6] = {2, 4, 3, 2, 2, 4};
/// ```
const DEPTH_CODE_SYMBOLS: [u8; 6] = [0, 7, 3, 2, 1, 15];
const DEPTH_CODE_BIT_LENGTHS: [u8; 6] = [2, 4, 3, 2, 2, 4];

// ============================================================================
// Huffman Tree Building
// ============================================================================

/// Node in the Huffman tree during construction.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.h
/// struct HuffmanTree {
///   uint32_t total_count;
///   int16_t index_left;
///   int16_t index_right_or_value;
/// };
/// ```
#[derive(Clone, Copy, Debug)]
struct HuffmanNode {
    /// Total count (frequency) of this subtree.
    total_count: u32,
    /// Index of left child, or -1 if leaf.
    index_left: i32,
    /// Index of right child, or symbol value if leaf.
    /// Note: libjxl uses i16, but we need i32 for alphabets > 32K symbols.
    index_right_or_value: i32,
}

impl HuffmanNode {
    fn leaf(count: u32, symbol: usize) -> Self {
        Self {
            total_count: count,
            index_left: -1,
            index_right_or_value: symbol as i32,
        }
    }

    fn sentinel() -> Self {
        Self {
            total_count: u32::MAX,
            index_left: -1,
            index_right_or_value: -1,
        }
    }

    #[allow(dead_code)]
    fn is_leaf(&self) -> bool {
        self.index_left < 0
    }
}

/// Recursively set depth for all symbols in a subtree.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.cc
/// void SetDepth(const HuffmanTree& p, HuffmanTree* pool, uint8_t* depth, uint8_t level) {
///   if (p.index_left >= 0) {
///     ++level;
///     SetDepth(pool[p.index_left], pool, depth, level);
///     SetDepth(pool[p.index_right_or_value], pool, depth, level);
///   } else {
///     depth[p.index_right_or_value] = level;
///   }
/// }
/// ```
fn set_depth(pool: &[HuffmanNode], root_idx: usize, depth: &mut [u8]) {
    // Iterative DFS to avoid stack overflow on deep/degenerate Huffman trees.
    // A degenerate tree with N leaves can have depth N-1, which overflows the
    // call stack for N > ~10K symbols (common with patch position histograms).
    let mut stack: Vec<(usize, u8)> = Vec::with_capacity(64);
    stack.push((root_idx, 0));

    while let Some((idx, level)) = stack.pop() {
        let node = &pool[idx];
        if node.index_left >= 0 {
            let next_level = level + 1;
            stack.push((node.index_left as usize, next_level));
            stack.push((node.index_right_or_value as usize, next_level));
        } else {
            depth[node.index_right_or_value as usize] = level;
        }
    }
}

/// Creates an optimal Huffman tree from symbol frequencies.
///
/// Returns the depth (code length) for each symbol. Symbols with zero
/// frequency get depth 0 (meaning they don't appear in the tree).
///
/// The tree depth is limited to `tree_limit` bits. If the natural tree
/// would exceed this, the algorithm retries with inflated minimum counts.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.cc
/// void CreateHuffmanTree(const uint32_t* data, const size_t length,
///                        const int tree_limit, uint8_t* depth) {
///   for (uint32_t count_limit = 1;; count_limit *= 2) {
///     std::vector<HuffmanTree> tree;
///     tree.reserve(2 * length + 1);
///     // ... build tree ...
///     if (*std::max_element(&depth[0], &depth[length]) <= tree_limit) {
///       break;
///     }
///   }
/// }
/// ```
pub fn create_huffman_tree(histogram: &[u32], tree_limit: u8) -> Vec<u8> {
    let length = histogram.len();
    let mut depth = vec![0u8; length];

    // Retry with increasing count_limit until tree fits in tree_limit bits
    let mut count_limit: u32 = 1;
    loop {
        // Build list of leaf nodes for non-zero symbols
        // Process in reverse order so that ties favor lower symbol indices
        let mut tree: Vec<HuffmanNode> = Vec::with_capacity(2 * length + 1);
        for i in (0..length).rev() {
            if histogram[i] > 0 {
                // Clamp count to at least count_limit - 1 to limit tree depth
                let count = histogram[i].max(count_limit.saturating_sub(1));
                tree.push(HuffmanNode::leaf(count, i));
            }
        }

        let n = tree.len();
        if n == 0 {
            // No symbols - return all zeros
            return depth;
        }
        if n == 1 {
            // Single symbol - give it depth 1 (will be corrected by caller)
            depth[tree[0].index_right_or_value as usize] = 1;
            return depth;
        }

        // Sort by count (ascending), then by symbol index (descending for ties)
        tree.sort_by(|a, b| {
            if a.total_count != b.total_count {
                a.total_count.cmp(&b.total_count)
            } else {
                // Higher index first for ties (reverse order)
                b.index_right_or_value.cmp(&a.index_right_or_value)
            }
        });

        // Add sentinel nodes
        tree.push(HuffmanNode::sentinel());
        tree.push(HuffmanNode::sentinel());

        // Build tree using two-queue algorithm
        // i: next leaf node, j: next internal node
        let mut i = 0;
        let mut j = n + 1;

        for _ in 0..n - 1 {
            // Pick two smallest nodes
            let left = if tree[i].total_count <= tree[j].total_count {
                let idx = i;
                i += 1;
                idx
            } else {
                let idx = j;
                j += 1;
                idx
            };

            let right = if tree[i].total_count <= tree[j].total_count {
                let idx = i;
                i += 1;
                idx
            } else {
                let idx = j;
                j += 1;
                idx
            };

            // Create parent node at the sentinel position
            let j_end = tree.len() - 1;
            tree[j_end].total_count = tree[left].total_count + tree[right].total_count;
            tree[j_end].index_left = left as i32;
            tree[j_end].index_right_or_value = right as i32;

            // Add new sentinel
            tree.push(HuffmanNode::sentinel());
        }

        // Extract depths from tree
        depth.fill(0);
        set_depth(&tree, 2 * n - 1, &mut depth);

        // Check if tree fits in limit
        if depth.iter().copied().max().unwrap_or(0) <= tree_limit {
            break;
        }

        // Tree too deep - retry with higher count_limit
        count_limit = match count_limit.checked_mul(2) {
            Some(v) => v,
            None => break, // u32 overflow — stop retrying, use best depth so far
        };
    }

    depth
}

// ============================================================================
// Bit Code Generation
// ============================================================================

/// Reverses bits in a value.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.cc
/// uint16_t ReverseBits(int num_bits, uint16_t bits) {
///   static const size_t kLut[16] = {
///       0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
///       0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};
///   size_t retval = kLut[bits & 0xf];
///   for (int i = 4; i < num_bits; i += 4) {
///     retval <<= 4;
///     bits = static_cast<uint16_t>(bits >> 4);
///     retval |= kLut[bits & 0xf];
///   }
///   retval >>= (-num_bits & 0x3);
///   return static_cast<uint16_t>(retval);
/// }
/// ```
fn reverse_bits(num_bits: u8, bits: u16) -> u16 {
    const LUT: [u16; 16] = [
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
    ];

    let mut retval = LUT[(bits & 0xf) as usize];
    let mut bits = bits;
    let mut i = 4i32;
    while i < num_bits as i32 {
        retval <<= 4;
        bits >>= 4;
        retval |= LUT[(bits & 0xf) as usize];
        i += 4;
    }
    retval >>= (-(num_bits as i32) & 0x3) as u32;
    retval
}

/// Converts code depths to canonical Huffman codes.
///
/// Returns (codes, depths) where `codes[i]` is the bit pattern for symbol `i`.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.cc
/// void ConvertBitDepthsToSymbols(const uint8_t* depth, size_t len, uint16_t* bits) {
///   const int kMaxBits = 16;
///   uint16_t bl_count[kMaxBits] = {0};
///   for (size_t i = 0; i < len; ++i) {
///     ++bl_count[depth[i]];
///   }
///   bl_count[0] = 0;
///   uint16_t next_code[kMaxBits];
///   next_code[0] = 0;
///   int code = 0;
///   for (size_t i = 1; i < kMaxBits; ++i) {
///     code = (code + bl_count[i - 1]) << 1;
///     next_code[i] = static_cast<uint16_t>(code);
///   }
///   for (size_t i = 0; i < len; ++i) {
///     if (depth[i]) {
///       bits[i] = ReverseBits(depth[i], next_code[depth[i]]++);
///     }
///   }
/// }
/// ```
pub fn convert_bit_depths_to_symbols(depth: &[u8]) -> Vec<u16> {
    const MAX_BITS: usize = 16;
    let len = depth.len();
    let mut bits = vec![0u16; len];

    // Count symbols at each depth
    let mut bl_count = [0u16; MAX_BITS];
    for &d in depth {
        bl_count[d as usize] += 1;
    }
    bl_count[0] = 0; // Depth 0 means symbol doesn't exist

    // Compute first code at each depth
    let mut next_code = [0u16; MAX_BITS];
    let mut code = 0i32;
    for i in 1..MAX_BITS {
        code = (code + bl_count[i - 1] as i32) << 1;
        next_code[i] = code as u16;
    }

    // Assign codes to symbols
    for i in 0..len {
        let d = depth[i];
        if d > 0 {
            bits[i] = reverse_bits(d, next_code[d as usize]);
            next_code[d as usize] += 1;
        }
    }

    bits
}

// ============================================================================
// RLE Compression of Code Lengths
// ============================================================================

/// RLE-compressed representation of a Huffman tree.
#[derive(Debug, Clone)]
pub struct CompressedTree {
    /// Code length codes (0-17)
    pub codes: Vec<u8>,
    /// Extra bits for each code (used by codes 16 and 17)
    pub extra_bits: Vec<u8>,
}

/// Reverse a slice in place.
fn reverse_slice(v: &mut [u8], start: usize, end: usize) {
    let mut s = start;
    let mut e = end - 1;
    while s < e {
        v.swap(s, e);
        s += 1;
        e -= 1;
    }
}

/// Write RLE-encoded repetitions of zeros.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.cc
/// void WriteHuffmanTreeRepetitionsZeros(size_t repetitions, size_t* tree_size,
///                                       uint8_t* tree, uint8_t* extra_bits_data) {
///   if (repetitions == 11) {
///     tree[*tree_size] = 0;
///     extra_bits_data[*tree_size] = 0;
///     ++(*tree_size);
///     --repetitions;
///   }
///   if (repetitions < 3) {
///     for (size_t i = 0; i < repetitions; ++i) {
///       tree[*tree_size] = 0;
///       extra_bits_data[*tree_size] = 0;
///       ++(*tree_size);
///     }
///   } else {
///     repetitions -= 3;
///     size_t start = *tree_size;
///     while (true) {
///       tree[*tree_size] = 17;
///       extra_bits_data[*tree_size] = repetitions & 0x7;
///       ++(*tree_size);
///       repetitions >>= 3;
///       if (repetitions == 0) {
///         break;
///       }
///       --repetitions;
///     }
///     Reverse(tree, start, *tree_size);
///     Reverse(extra_bits_data, start, *tree_size);
///   }
/// }
/// ```
fn write_repetitions_zeros(mut repetitions: usize, tree: &mut CompressedTree) {
    // Special case: 11 reps would need code 17 with extra=8, but max extra is 7
    if repetitions == 11 {
        tree.codes.push(0);
        tree.extra_bits.push(0);
        repetitions -= 1;
    }

    if repetitions < 3 {
        // Write individual zeros
        for _ in 0..repetitions {
            tree.codes.push(0);
            tree.extra_bits.push(0);
        }
    } else {
        // Use code 17: repeat zeros with 3 extra bits (3-10 reps)
        repetitions -= 3;
        let start = tree.codes.len();
        loop {
            tree.codes.push(17);
            tree.extra_bits.push((repetitions & 0x7) as u8);
            repetitions >>= 3;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        // Reverse the codes we just wrote
        let end = tree.codes.len();
        reverse_slice(&mut tree.codes, start, end);
        reverse_slice(&mut tree.extra_bits, start, end);
    }
}

/// Write RLE-encoded repetitions of non-zero values.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.cc
/// void WriteHuffmanTreeRepetitions(const uint8_t previous_value,
///                                  const uint8_t value, size_t repetitions,
///                                  size_t* tree_size, uint8_t* tree,
///                                  uint8_t* extra_bits_data) {
///   JXL_DASSERT(repetitions > 0);
///   if (previous_value != value) {
///     tree[*tree_size] = value;
///     extra_bits_data[*tree_size] = 0;
///     ++(*tree_size);
///     --repetitions;
///   }
///   if (repetitions == 7) {
///     tree[*tree_size] = value;
///     extra_bits_data[*tree_size] = 0;
///     ++(*tree_size);
///     --repetitions;
///   }
///   if (repetitions < 3) {
///     for (size_t i = 0; i < repetitions; ++i) {
///       tree[*tree_size] = value;
///       extra_bits_data[*tree_size] = 0;
///       ++(*tree_size);
///     }
///   } else {
///     repetitions -= 3;
///     size_t start = *tree_size;
///     while (true) {
///       tree[*tree_size] = 16;
///       extra_bits_data[*tree_size] = repetitions & 0x3;
///       ++(*tree_size);
///       repetitions >>= 2;
///       if (repetitions == 0) {
///         break;
///       }
///       --repetitions;
///     }
///     Reverse(tree, start, *tree_size);
///     Reverse(extra_bits_data, start, *tree_size);
///   }
/// }
/// ```
fn write_repetitions_nonzero(
    previous_value: u8,
    value: u8,
    mut repetitions: usize,
    tree: &mut CompressedTree,
) {
    debug_assert!(repetitions > 0);

    // If value changed, write it literally first
    if previous_value != value {
        tree.codes.push(value);
        tree.extra_bits.push(0);
        repetitions -= 1;
    }

    // Special case: 7 reps would need code 16 with extra=4, but max extra is 3
    if repetitions == 7 {
        tree.codes.push(value);
        tree.extra_bits.push(0);
        repetitions -= 1;
    }

    if repetitions < 3 {
        // Write individual values
        for _ in 0..repetitions {
            tree.codes.push(value);
            tree.extra_bits.push(0);
        }
    } else {
        // Use code 16: repeat previous with 2 extra bits (3-6 reps)
        repetitions -= 3;
        let start = tree.codes.len();
        loop {
            tree.codes.push(16);
            tree.extra_bits.push((repetitions & 0x3) as u8);
            repetitions >>= 2;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        // Reverse the codes we just wrote
        let end = tree.codes.len();
        reverse_slice(&mut tree.codes, start, end);
        reverse_slice(&mut tree.extra_bits, start, end);
    }
}

/// Decide whether to use RLE for zeros and non-zeros.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.cc
/// static void DecideOverRleUse(const uint8_t* depth, const size_t length,
///                              bool* use_rle_for_non_zero, bool* use_rle_for_zero) {
///   size_t total_reps_zero = 0;
///   size_t total_reps_non_zero = 0;
///   size_t count_reps_zero = 1;
///   size_t count_reps_non_zero = 1;
///   for (size_t i = 0; i < length;) {
///     const uint8_t value = depth[i];
///     size_t reps = 1;
///     for (size_t k = i + 1; k < length && depth[k] == value; ++k) {
///       ++reps;
///     }
///     if (reps >= 3 && value == 0) {
///       total_reps_zero += reps;
///       ++count_reps_zero;
///     }
///     if (reps >= 4 && value != 0) {
///       total_reps_non_zero += reps;
///       ++count_reps_non_zero;
///     }
///     i += reps;
///   }
///   *use_rle_for_non_zero = total_reps_non_zero > count_reps_non_zero * 2;
///   *use_rle_for_zero = total_reps_zero > count_reps_zero * 2;
/// }
/// ```
fn decide_rle_use(depth: &[u8]) -> (bool, bool) {
    let mut total_reps_zero = 0usize;
    let mut total_reps_non_zero = 0usize;
    let mut count_reps_zero = 1usize;
    let mut count_reps_non_zero = 1usize;

    let mut i = 0;
    while i < depth.len() {
        let value = depth[i];
        let mut reps = 1usize;
        while i + reps < depth.len() && depth[i + reps] == value {
            reps += 1;
        }
        if reps >= 3 && value == 0 {
            total_reps_zero += reps;
            count_reps_zero += 1;
        }
        if reps >= 4 && value != 0 {
            total_reps_non_zero += reps;
            count_reps_non_zero += 1;
        }
        i += reps;
    }

    let use_rle_for_non_zero = total_reps_non_zero > count_reps_non_zero * 2;
    let use_rle_for_zero = total_reps_zero > count_reps_zero * 2;
    (use_rle_for_non_zero, use_rle_for_zero)
}

/// Compresses a depth array into RLE-encoded code length codes.
///
/// ```cpp
/// // libjxl: enc_huffman_tree.cc
/// void WriteHuffmanTree(const uint8_t* depth, size_t length, size_t* tree_size,
///                       uint8_t* tree, uint8_t* extra_bits_data) {
///   uint8_t previous_value = 8;
///   // Throw away trailing zeros.
///   size_t new_length = length;
///   for (size_t i = 0; i < length; ++i) {
///     if (depth[length - i - 1] == 0) {
///       --new_length;
///     } else {
///       break;
///     }
///   }
///   // First gather statistics on if it is a good idea to do rle.
///   bool use_rle_for_non_zero = false;
///   bool use_rle_for_zero = false;
///   if (length > 50) {
///     DecideOverRleUse(depth, new_length, &use_rle_for_non_zero, &use_rle_for_zero);
///   }
///   // Actual rle coding.
///   for (size_t i = 0; i < new_length;) {
///     const uint8_t value = depth[i];
///     size_t reps = 1;
///     if ((value != 0 && use_rle_for_non_zero) || (value == 0 && use_rle_for_zero)) {
///       for (size_t k = i + 1; k < new_length && depth[k] == value; ++k) {
///         ++reps;
///       }
///     }
///     if (value == 0) {
///       WriteHuffmanTreeRepetitionsZeros(reps, tree_size, tree, extra_bits_data);
///     } else {
///       WriteHuffmanTreeRepetitions(previous_value, value, reps, tree_size, tree, extra_bits_data);
///       previous_value = value;
///     }
///     i += reps;
///   }
/// }
/// ```
pub fn write_huffman_tree(depth: &[u8]) -> CompressedTree {
    let mut tree = CompressedTree {
        codes: Vec::new(),
        extra_bits: Vec::new(),
    };

    // Throw away trailing zeros
    let mut new_length = depth.len();
    while new_length > 0 && depth[new_length - 1] == 0 {
        new_length -= 1;
    }

    if new_length == 0 {
        return tree;
    }

    // Decide whether to use RLE (only for longer sequences)
    let (use_rle_for_non_zero, use_rle_for_zero) = if depth.len() > 50 {
        decide_rle_use(&depth[..new_length])
    } else {
        (false, false)
    };

    // Encode with RLE
    let mut previous_value = 8u8; // Initial "previous" value (won't match any real depth)
    let mut i = 0;
    while i < new_length {
        let value = depth[i];
        let mut reps = 1usize;

        // Count repetitions if RLE is enabled for this value type
        if (value != 0 && use_rle_for_non_zero) || (value == 0 && use_rle_for_zero) {
            while i + reps < new_length && depth[i + reps] == value {
                reps += 1;
            }
        }

        if value == 0 {
            write_repetitions_zeros(reps, &mut tree);
        } else {
            write_repetitions_nonzero(previous_value, value, reps, &mut tree);
            previous_value = value;
        }
        i += reps;
    }

    tree
}

// ============================================================================
// Bitstream Encoding
// ============================================================================

/// Writes the meta-Huffman tree (tree of tree) to the bitstream.
///
/// This encodes which code length codes (0-17) are used and their depths.
///
/// ```cpp
/// // libjxl: enc_huffman.cc
/// void StoreHuffmanTreeOfHuffmanTreeToBitMask(const int num_codes,
///                                             const uint8_t* code_length_bitdepth,
///                                             BitWriter* writer) {
///   // Throw away trailing zeros:
///   size_t codes_to_store = kCodeLengthCodes;
///   if (num_codes > 1) {
///     for (; codes_to_store > 0; --codes_to_store) {
///       if (code_length_bitdepth[kStorageOrder[codes_to_store - 1]] != 0) {
///         break;
///       }
///     }
///   }
///   size_t skip_some = 0;
///   if (code_length_bitdepth[kStorageOrder[0]] == 0 &&
///       code_length_bitdepth[kStorageOrder[1]] == 0) {
///     skip_some = 2;
///     if (code_length_bitdepth[kStorageOrder[2]] == 0) {
///       skip_some = 3;
///     }
///   }
///   writer->Write(2, skip_some);
///   for (size_t i = skip_some; i < codes_to_store; ++i) {
///     size_t l = code_length_bitdepth[kStorageOrder[i]];
///     writer->Write(kHuffmanBitLengthHuffmanCodeBitLengths[l],
///                   kHuffmanBitLengthHuffmanCodeSymbols[l]);
///   }
/// }
/// ```
pub fn store_meta_huffman_tree(
    num_codes: usize,
    code_length_depths: &[u8; CODE_LENGTH_CODES],
    writer: &mut BitWriter,
) -> Result<()> {
    // Find how many codes to store (trim trailing zeros in storage order)
    let mut codes_to_store = CODE_LENGTH_CODES;
    if num_codes > 1 {
        while codes_to_store > 0 && code_length_depths[STORAGE_ORDER[codes_to_store - 1]] == 0 {
            codes_to_store -= 1;
        }
    }

    // Determine skip count (how many leading zeros in storage order)
    let mut skip_some = 0usize;
    if code_length_depths[STORAGE_ORDER[0]] == 0 && code_length_depths[STORAGE_ORDER[1]] == 0 {
        skip_some = 2;
        if code_length_depths[STORAGE_ORDER[2]] == 0 {
            skip_some = 3;
        }
    }

    crate::trace::debug_eprintln!(
        "STORE_META: codes_to_store={}, skip_some={}",
        codes_to_store,
        skip_some
    );

    // Write skip count (2 bits) - this is the simple_code_or_skip field
    // Values 0, 2, 3 indicate full tree with skip; value 1 would be simple code
    let _bit_pos_start = writer.bits_written();
    writer.write(2, skip_some as u64)?;
    crate::trace::debug_eprintln!(
        "  META: wrote hskip={} at bit {}",
        skip_some,
        _bit_pos_start
    );

    // Write code length depths using static Huffman code
    // The decoder reads code lengths for symbols in storage order
    let mut _bitacc = 0usize;
    #[allow(clippy::unused_enumerate_index)]
    for (_idx, &symbol) in STORAGE_ORDER[skip_some..codes_to_store].iter().enumerate() {
        let depth_value = code_length_depths[symbol] as usize;
        debug_assert!(depth_value <= 5, "Code length depth must be 0-5");
        let bits = DEPTH_CODE_BIT_LENGTHS[depth_value] as usize;
        let code = DEPTH_CODE_SYMBOLS[depth_value] as u64;
        writer.write(bits, code)?;
        if depth_value > 0 {
            _bitacc += 32 >> depth_value;
        }
        crate::trace::debug_eprintln!(
            "  META[{}]: symbol={}, depth={}, code=0b{:0width$b} ({} bits), bitacc={}",
            skip_some + _idx,
            symbol,
            depth_value,
            code,
            bits,
            _bitacc,
            width = bits
        );
    }
    crate::trace::debug_eprintln!("  META: final bitacc={} (should be 32)", _bitacc);

    Ok(())
}

/// Writes the RLE-compressed Huffman tree to the bitstream.
///
/// ```cpp
/// // libjxl: enc_huffman.cc
/// Status StoreHuffmanTreeToBitMask(const size_t huffman_tree_size,
///                                  const uint8_t* huffman_tree,
///                                  const uint8_t* huffman_tree_extra_bits,
///                                  const uint8_t* code_length_bitdepth,
///                                  const uint16_t* code_length_bitdepth_symbols,
///                                  BitWriter* writer) {
///   for (size_t i = 0; i < huffman_tree_size; ++i) {
///     size_t ix = huffman_tree[i];
///     writer->Write(code_length_bitdepth[ix], code_length_bitdepth_symbols[ix]);
///     switch (ix) {
///       case 16: writer->Write(2, huffman_tree_extra_bits[i]); break;
///       case 17: writer->Write(3, huffman_tree_extra_bits[i]); break;
///     }
///   }
///   return true;
/// }
/// ```
pub fn store_compressed_tree(
    tree: &CompressedTree,
    code_length_depths: &[u8; CODE_LENGTH_CODES],
    code_length_codes: &[u16; CODE_LENGTH_CODES],
    writer: &mut BitWriter,
) -> Result<()> {
    crate::trace::debug_eprintln!("  TREE: meta_codes = {:?}", &code_length_codes[..5]);
    crate::trace::debug_eprintln!("  TREE: meta_depths = {:?}", &code_length_depths[..5]);

    for i in 0..tree.codes.len().min(10) {
        let ix = tree.codes[i] as usize;
        debug_assert!(ix < CODE_LENGTH_CODES);

        // Write the code for this code length
        let depth = code_length_depths[ix] as usize;
        let code = code_length_codes[ix] as u64;
        let _bit_pos = writer.bits_written();
        if depth > 0 {
            writer.write(depth, code)?;
        }
        crate::trace::debug_eprintln!(
            "  TREE[{}]: code_len_code={}, meta_depth={}, meta_code=0b{:b}, extra_bits={}",
            i,
            ix,
            depth,
            code,
            tree.extra_bits[i]
        );

        // Write extra bits for RLE codes
        match ix {
            16 => writer.write(2, tree.extra_bits[i] as u64)?,
            17 => writer.write(3, tree.extra_bits[i] as u64)?,
            _ => {}
        }
    }

    // Write remaining codes silently
    for i in 10..tree.codes.len() {
        let ix = tree.codes[i] as usize;
        let depth = code_length_depths[ix] as usize;
        let code = code_length_codes[ix] as u64;
        if depth > 0 {
            writer.write(depth, code)?;
        }
        match ix {
            16 => writer.write(2, tree.extra_bits[i] as u64)?,
            17 => writer.write(3, tree.extra_bits[i] as u64)?,
            _ => {}
        }
    }

    Ok(())
}

/// Stores a full Huffman tree to the bitstream.
///
/// This is used when there are more than 4 unique symbols.
///
/// ```cpp
/// // libjxl: enc_huffman.cc
/// Status StoreHuffmanTree(const uint8_t* depths, size_t num, BitWriter* writer) {
///   // Write the Huffman tree into the compact representation.
///   auto arena = ...;
///   uint8_t* huffman_tree = arena.data();
///   uint8_t* huffman_tree_extra_bits = arena.data() + num;
///   size_t huffman_tree_size = 0;
///   WriteHuffmanTree(depths, num, &huffman_tree_size, huffman_tree, huffman_tree_extra_bits);
///
///   // Calculate the statistics of the Huffman tree.
///   uint32_t huffman_tree_histogram[kCodeLengthCodes] = {0};
///   for (size_t i = 0; i < huffman_tree_size; ++i) {
///     ++huffman_tree_histogram[huffman_tree[i]];
///   }
///
///   int num_codes = 0;
///   int code = 0;
///   for (int i = 0; i < kCodeLengthCodes; ++i) {
///     if (huffman_tree_histogram[i]) {
///       if (num_codes == 0) { code = i; num_codes = 1; }
///       else if (num_codes == 1) { num_codes = 2; break; }
///     }
///   }
///
///   // Calculate another Huffman tree for compressing the first.
///   uint8_t code_length_bitdepth[kCodeLengthCodes] = {0};
///   uint16_t code_length_bitdepth_symbols[kCodeLengthCodes] = {0};
///   CreateHuffmanTree(&huffman_tree_histogram[0], kCodeLengthCodes, 5, &code_length_bitdepth[0]);
///   ConvertBitDepthsToSymbols(code_length_bitdepth, kCodeLengthCodes, &code_length_bitdepth_symbols[0]);
///
///   StoreHuffmanTreeOfHuffmanTreeToBitMask(num_codes, code_length_bitdepth, writer);
///
///   if (num_codes == 1) {
///     code_length_bitdepth[code] = 0;
///   }
///
///   StoreHuffmanTreeToBitMask(...);
///   return true;
/// }
/// ```
pub fn store_huffman_tree(depths: &[u8], writer: &mut BitWriter) -> Result<()> {
    // Debug: show raw depths for first and last few elements
    let _first_10: Vec<u8> = depths.iter().take(10).copied().collect();
    let _last_10: Vec<u8> = depths.iter().rev().take(10).rev().copied().collect();
    crate::trace::debug_eprintln!(
        "STORE_HUFF: depths len={}, first_10={:?}, last_10={:?}",
        depths.len(),
        _first_10,
        _last_10
    );

    // RLE-compress the depth array
    let compressed = write_huffman_tree(depths);
    crate::trace::debug_eprintln!(
        "STORE_HUFF: compressed codes={:?}, extra_bits={:?}",
        compressed.codes,
        compressed.extra_bits
    );

    // Build histogram of code length codes
    let mut histogram = [0u32; CODE_LENGTH_CODES];
    for &code in &compressed.codes {
        histogram[code as usize] += 1;
    }
    crate::trace::debug_eprintln!("STORE_HUFF: code_length_histogram={:?}", histogram);

    // Count how many distinct code length codes are used
    let mut num_codes = 0;
    let mut single_code = 0;
    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            if num_codes == 0 {
                single_code = i;
                num_codes = 1;
            } else if num_codes == 1 {
                num_codes = 2;
                break;
            }
        }
    }

    // Build Huffman tree for the code length codes (meta-Huffman)
    let code_length_depths = create_huffman_tree(&histogram, MAX_CODE_LENGTH_DEPTH);
    let mut code_length_depths_arr = [0u8; CODE_LENGTH_CODES];
    code_length_depths_arr.copy_from_slice(&code_length_depths);

    let code_length_codes_vec = convert_bit_depths_to_symbols(&code_length_depths);
    let mut code_length_codes_arr = [0u16; CODE_LENGTH_CODES];
    code_length_codes_arr.copy_from_slice(&code_length_codes_vec);

    crate::trace::debug_eprintln!(
        "STORE_HUFF: num_codes={}, meta_depths={:?}",
        num_codes,
        code_length_depths_arr
    );

    // Write meta-Huffman tree
    store_meta_huffman_tree(num_codes, &code_length_depths_arr, writer)?;

    // Special case: single code length code - set its depth to 0 for implicit encoding
    if num_codes == 1 {
        code_length_depths_arr[single_code] = 0;
    }

    // Write the compressed tree
    store_compressed_tree(
        &compressed,
        &code_length_depths_arr,
        &code_length_codes_arr,
        writer,
    )?;

    Ok(())
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Result of building a Huffman table.
pub struct HuffmanTable {
    /// Code depth (bit length) for each symbol.
    pub depths: Vec<u8>,
    /// Bit code for each symbol.
    pub codes: Vec<u16>,
}

/// Builds and stores a Huffman table from a histogram.
///
/// This is the main entry point. It handles:
/// - Single symbol (special 4-bit encoding)
/// - 2-4 symbols (simple Huffman code)
/// - 5+ symbols (full code length table)
///
/// ```cpp
/// // libjxl: enc_huffman.cc
/// Status BuildAndStoreHuffmanTree(const uint32_t* histogram, const size_t length,
///                                 uint8_t* depth, uint16_t* bits, BitWriter* writer) {
///   size_t count = 0;
///   size_t s4[4] = {0};
///   for (size_t i = 0; i < length; i++) {
///     if (histogram[i]) {
///       if (count < 4) { s4[count] = i; }
///       else if (count > 4) { break; }
///       count++;
///     }
///   }
///
///   size_t max_bits = 0;
///   size_t max_bits_counter = length - 1;
///   while (max_bits_counter) { max_bits_counter >>= 1; ++max_bits; }
///
///   if (count <= 1) {
///     writer->Write(4, 1);
///     writer->Write(max_bits, s4[0]);
///     return true;
///   }
///
///   CreateHuffmanTree(histogram, length, 15, depth);
///   ConvertBitDepthsToSymbols(depth, length, bits);
///
///   if (count <= 4) {
///     StoreSimpleHuffmanTree(depth, s4, count, max_bits, writer);
///   } else {
///     JXL_RETURN_IF_ERROR(StoreHuffmanTree(depth, length, writer));
///   }
///   return true;
/// }
/// ```
pub fn build_and_store_huffman_tree(
    histogram: &[u32],
    writer: &mut BitWriter,
) -> Result<HuffmanTable> {
    let length = histogram.len();

    // Debug: print non-zero symbols
    let nonzero: Vec<(usize, u32)> = histogram
        .iter()
        .enumerate()
        .filter(|&(_, h)| *h > 0)
        .map(|(i, h)| (i, *h))
        .collect();
    crate::trace::debug_eprintln!(
        "HUFFMAN_BUILD: alphabet_size={}, nonzero_symbols={:?}",
        length,
        nonzero
    );

    // Count non-zero symbols and track first 4
    let mut count = 0usize;
    let mut s4 = [0usize; 4];
    for (i, &h) in histogram.iter().enumerate() {
        if h > 0 {
            if count < 4 {
                s4[count] = i;
            }
            count += 1;
            if count > 4 {
                break; // We know it's >4, that's all we need
            }
        }
    }

    // Calculate max_bits = ceil(log2(length))
    let max_bits = if length <= 1 {
        0
    } else {
        (usize::BITS - (length - 1).leading_zeros()) as usize
    };

    // Single symbol case
    if count <= 1 {
        let mut depths = vec![0u8; length];
        let codes = vec![0u16; length];
        // Write: 4 bits = 0001, then max_bits for symbol
        writer.write(4, 1)?;
        writer.write(max_bits, s4[0] as u64)?;
        // Single symbol has depth 0 (implicit, no bits needed to encode)
        if count == 1 {
            depths[s4[0]] = 0;
        }
        return Ok(HuffmanTable { depths, codes });
    }

    // Build optimal Huffman tree
    let depths = create_huffman_tree(histogram, MAX_CODE_DEPTH);
    let codes = convert_bit_depths_to_symbols(&depths);

    // Debug: print depths for non-zero symbols
    let _depths_info: Vec<(usize, u8, u16)> = nonzero
        .iter()
        .map(|(i, _)| (*i, depths[*i], codes[*i]))
        .collect();
    crate::trace::debug_eprintln!(
        "HUFFMAN_BUILD: depths/codes for used symbols: {:?}",
        _depths_info
    );

    if count <= 4 {
        // Simple Huffman code
        crate::trace::debug_eprintln!("HUFFMAN_BUILD: using simple code for {} symbols", count);
        store_simple_huffman_tree(&depths, &mut s4, count, max_bits, writer)?;
    } else {
        // Full code length table
        crate::trace::debug_eprintln!("HUFFMAN_BUILD: using full table for {} symbols", count);
        store_huffman_tree(&depths, writer)?;
    }

    Ok(HuffmanTable { depths, codes })
}

/// Stores a simple Huffman tree (1-4 symbols).
///
/// ```cpp
/// // libjxl: enc_huffman.cc
/// void StoreSimpleHuffmanTree(const uint8_t* depths, size_t symbols[4],
///                             size_t num_symbols, size_t max_bits, BitWriter* writer) {
///   writer->Write(2, 1);  // simple code marker
///   writer->Write(2, num_symbols - 1);  // NSYM - 1
///   // Sort by depth
///   for (size_t i = 0; i < num_symbols; i++) {
///     for (size_t j = i + 1; j < num_symbols; j++) {
///       if (depths[symbols[j]] < depths[symbols[i]]) {
///         std::swap(symbols[j], symbols[i]);
///       }
///     }
///   }
///   // Write symbols
///   for (size_t i = 0; i < num_symbols; i++) {
///     writer->Write(max_bits, symbols[i]);
///   }
///   // tree_select for 4 symbols
///   if (num_symbols == 4) {
///     writer->Write(1, depths[symbols[0]] == 1 ? 1 : 0);
///   }
/// }
/// ```
fn store_simple_huffman_tree(
    depths: &[u8],
    symbols: &mut [usize; 4],
    num_symbols: usize,
    max_bits: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    // Write simple code marker
    writer.write(2, 1)?;
    writer.write(2, (num_symbols - 1) as u64)?;

    // Sort symbols by depth (ascending)
    for i in 0..num_symbols {
        for j in i + 1..num_symbols {
            if depths[symbols[j]] < depths[symbols[i]] {
                symbols.swap(i, j);
            }
        }
    }

    // Write symbol indices
    for &sym in symbols.iter().take(num_symbols) {
        writer.write(max_bits, sym as u64)?;
    }

    // tree_select for 4 symbols
    if num_symbols == 4 {
        let tree_select = if depths[symbols[0]] == 1 { 1 } else { 0 };
        writer.write(1, tree_select)?;
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_huffman_tree_single_symbol() {
        let histogram = [0, 0, 100, 0, 0];
        let depths = create_huffman_tree(&histogram, 15);
        assert_eq!(depths, vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn test_create_huffman_tree_two_symbols() {
        let histogram = [50, 0, 50, 0, 0];
        let depths = create_huffman_tree(&histogram, 15);
        // Both symbols should have depth 1
        assert_eq!(depths[0], 1);
        assert_eq!(depths[2], 1);
    }

    #[test]
    fn test_create_huffman_tree_four_symbols() {
        let histogram = [10, 20, 30, 40, 0];
        let depths = create_huffman_tree(&histogram, 15);
        // Most frequent (40) should have shortest depth
        assert!(depths[3] <= depths[2]);
        assert!(depths[2] <= depths[1]);
        assert!(depths[1] <= depths[0]);
    }

    #[test]
    fn test_create_huffman_tree_many_symbols() {
        // 10 symbols with varying frequencies
        let histogram = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let depths = create_huffman_tree(&histogram, 15);

        // All non-zero symbols should have non-zero depth
        for (i, &d) in depths.iter().enumerate() {
            assert!(d > 0, "Symbol {} should have non-zero depth", i);
        }

        // More frequent symbols should have shorter or equal depth
        for i in 1..10 {
            assert!(
                depths[i] <= depths[i - 1],
                "Symbol {} (freq {}) should have depth <= symbol {} (freq {})",
                i,
                histogram[i],
                i - 1,
                histogram[i - 1]
            );
        }
    }

    #[test]
    fn test_create_huffman_tree_depth_limit() {
        // Many symbols with same frequency - could create deep tree
        let histogram = vec![1u32; 100];
        let depths = create_huffman_tree(&histogram, 15);

        // All depths should be <= 15
        for &d in &depths {
            assert!(d <= 15, "Depth {} exceeds limit of 15", d);
        }
    }

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(1, 0), 0);
        assert_eq!(reverse_bits(1, 1), 1);
        assert_eq!(reverse_bits(2, 0b01), 0b10);
        assert_eq!(reverse_bits(2, 0b10), 0b01);
        assert_eq!(reverse_bits(3, 0b001), 0b100);
        assert_eq!(reverse_bits(4, 0b0001), 0b1000);
        assert_eq!(reverse_bits(8, 0b10101010), 0b01010101);
    }

    #[test]
    fn test_convert_bit_depths_to_symbols() {
        // Two symbols at depth 1
        let depths = [1, 1, 0];
        let codes = convert_bit_depths_to_symbols(&depths);
        // Should get codes 0 and 1 (reversed)
        assert_eq!(codes[0], 0);
        assert_eq!(codes[1], 1);
    }

    #[test]
    fn test_convert_bit_depths_three_symbols() {
        // Three symbols: depths 1, 2, 2
        let depths = [1, 2, 2, 0];
        let codes = convert_bit_depths_to_symbols(&depths);
        // Symbol 0: depth 1, code 0
        // Symbol 1: depth 2, code 10 -> reversed = 01
        // Symbol 2: depth 2, code 11 -> reversed = 11
        assert_eq!(codes[0], 0b0);
        assert_eq!(codes[1], 0b01);
        assert_eq!(codes[2], 0b11);
    }

    #[test]
    fn test_write_huffman_tree_simple() {
        // Simple depth array
        let depths = [2, 2, 0, 0];
        let tree = write_huffman_tree(&depths);
        // Should just have two 2s
        assert_eq!(tree.codes, vec![2, 2]);
        assert!(tree.extra_bits.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_write_huffman_tree_with_zeros() {
        let depths = [3, 0, 0, 0, 3];
        let tree = write_huffman_tree(&depths);
        // Should encode 3, zeros (maybe RLE), 3
        // Note: trailing zeros are stripped, so we only go to index 4
        assert!(!tree.codes.is_empty());
    }

    #[test]
    fn test_write_huffman_tree_long_zeros() {
        // Many zeros in the middle - should trigger RLE
        let mut depths = vec![0u8; 100];
        depths[0] = 3;
        depths[99] = 3;
        let tree = write_huffman_tree(&depths);
        // Should use code 17 for zero runs
        assert!(tree.codes.contains(&17));
    }

    #[test]
    fn test_build_and_store_single_symbol() {
        let histogram = [0, 0, 100, 0, 0];
        let mut writer = BitWriter::new();
        let table = build_and_store_huffman_tree(&histogram, &mut writer).unwrap();

        // Single symbol should have depth 0
        assert_eq!(table.depths[2], 0);

        // Check output format: 4 bits = 0001, then max_bits for symbol
        let bytes = writer.finish_with_padding();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_build_and_store_two_symbols() {
        let histogram = [10, 0, 20, 0, 0];
        let mut writer = BitWriter::new();
        let table = build_and_store_huffman_tree(&histogram, &mut writer).unwrap();

        // Both should have depth 1
        assert_eq!(table.depths[0], 1);
        assert_eq!(table.depths[2], 1);

        let bytes = writer.finish_with_padding();
        // Should start with simple code marker (bits: 01)
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_build_and_store_many_symbols() {
        // 10 symbols - requires full code length table
        let histogram: Vec<u32> = (1..=10).map(|x| x * 10).collect();
        let mut writer = BitWriter::new();
        let table = build_and_store_huffman_tree(&histogram, &mut writer).unwrap();

        // All symbols should have non-zero depth
        for i in 0..10 {
            assert!(table.depths[i] > 0);
        }

        let bytes = writer.finish_with_padding();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_build_and_store_256_symbols() {
        // Full byte alphabet with varying frequencies
        let mut histogram = vec![0u32; 256];
        for (i, h) in histogram.iter_mut().enumerate() {
            *h = (i as u32 + 1) * 2;
        }

        let mut writer = BitWriter::new();
        let table = build_and_store_huffman_tree(&histogram, &mut writer).unwrap();

        // All symbols should have valid depth
        for i in 0..256 {
            assert!(table.depths[i] > 0 && table.depths[i] <= 15);
        }

        let bytes = writer.finish_with_padding();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_canonical_codes_are_prefix_free() {
        let histogram: Vec<u32> = (1..=20).map(|x| x * 5).collect();
        let depths = create_huffman_tree(&histogram, 15);
        let codes = convert_bit_depths_to_symbols(&depths);

        // Verify no code is a prefix of another
        for i in 0..20 {
            if depths[i] == 0 {
                continue;
            }
            for j in i + 1..20 {
                if depths[j] == 0 {
                    continue;
                }
                let (shorter, longer) = if depths[i] <= depths[j] {
                    ((codes[i], depths[i]), (codes[j], depths[j]))
                } else {
                    ((codes[j], depths[j]), (codes[i], depths[i]))
                };

                // Extract prefix of longer code
                let prefix_mask = (1u16 << shorter.1) - 1;
                let longer_prefix = longer.0 & prefix_mask;

                assert_ne!(
                    shorter.0, longer_prefix,
                    "Code {} (depth {}) is prefix of code {} (depth {})",
                    shorter.0, shorter.1, longer.0, longer.1
                );
            }
        }
    }
}
