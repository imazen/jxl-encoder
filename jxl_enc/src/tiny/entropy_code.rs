// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Entropy code types and token writing.
//!
//! These are ported from libjxl-tiny and will be used for Huffman coding.

#![allow(dead_code)]

use super::token::{Token, UintCoder};
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::error::Result;

/// Number of code length codes used in Huffman tree serialization.
const CODE_LENGTH_CODES: usize = 18;

/// Maximum number of symbols in the Huffman alphabet.
pub const ALPHABET_SIZE: usize = 64;

/// Maximum number of contexts.
pub const MAX_CONTEXTS: usize = 128;

/// Maximum bits per token (upper bound for allotment).
pub const MAX_BITS_PER_TOKEN: usize = 24;

/// A Huffman prefix code.
///
/// Contains the bit depths (lengths) and bit patterns for each symbol.
#[derive(Clone, Copy)]
pub struct PrefixCode {
    /// Bit depth (length) for each symbol in the alphabet.
    pub depths: [u8; ALPHABET_SIZE],
    /// Bit pattern for each symbol in the alphabet.
    pub bits: [u16; ALPHABET_SIZE],
}

impl Default for PrefixCode {
    fn default() -> Self {
        Self {
            depths: [0; ALPHABET_SIZE],
            bits: [0; ALPHABET_SIZE],
        }
    }
}

impl std::fmt::Debug for PrefixCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixCode")
            .field("depths", &&self.depths[..])
            .field("bits", &&self.bits[..])
            .finish()
    }
}

/// An entropy code consisting of context map and prefix codes.
#[derive(Debug)]
pub struct EntropyCode<'a> {
    /// Context map: maps context ID -> prefix code index.
    pub context_map: &'a [u8],
    /// Number of contexts.
    pub num_contexts: usize,
    /// Prefix codes (Huffman codes).
    pub prefix_codes: &'a [PrefixCode],
    /// Number of prefix codes.
    pub num_prefix_codes: usize,
}

impl<'a> EntropyCode<'a> {
    /// Create a new entropy code from static tables.
    pub const fn new(context_map: &'a [u8], prefix_codes: &'a [PrefixCode]) -> Self {
        Self {
            context_map,
            num_contexts: context_map.len(),
            prefix_codes,
            num_prefix_codes: prefix_codes.len(),
        }
    }
}

/// Write a token using the given entropy code.
///
/// This encodes the value using the UintCoder, looks up the prefix code
/// via the context map, and writes the Huffman code followed by extra bits.
#[inline]
pub fn write_token(token: &Token, code: &EntropyCode, writer: &mut BitWriter) -> Result<()> {
    let encoded = UintCoder::encode(token.value);

    // Look up the prefix code index from the context map
    let prefix_idx = code.context_map[token.context as usize] as usize;
    let pc = &code.prefix_codes[prefix_idx];

    // Get the Huffman code for this token
    let tok = encoded.token as usize;
    let depth = pc.depths[tok] as usize;
    let bits = pc.bits[tok] as u64;

    // Combine Huffman bits and extra bits
    let data = bits | ((encoded.bits as u64) << depth);
    let total_bits = depth + encoded.nbits as usize;

    writer.write(total_bits, data)
}

/// Write VarLenUint16 encoding (0-65535).
fn write_var_len_uint16(n: usize, writer: &mut BitWriter) -> Result<()> {
    debug_assert!(n <= 65535);
    if n == 0 {
        writer.write(1, 0)?;
    } else {
        writer.write(1, 1)?;
        let nbits = floor_log2_nonzero(n as u32);
        writer.write(4, nbits as u64)?;
        writer.write(nbits as usize, (n - (1 << nbits)) as u64)?;
    }
    Ok(())
}

/// Floor of log2 for non-zero values.
fn floor_log2_nonzero(n: u32) -> u32 {
    debug_assert!(n > 0);
    31 - n.leading_zeros()
}

/// Reverse bits in a value.
fn reverse_bits(num_bits: u32, bits: u16) -> u16 {
    static LUT: [u16; 16] = [
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
    ];
    let mut retval: usize = LUT[(bits & 0xf) as usize] as usize;
    let mut bits = bits;
    let mut i = 4;
    while i < num_bits {
        retval <<= 4;
        bits >>= 4;
        retval |= LUT[(bits & 0xf) as usize] as usize;
        i += 4;
    }
    retval >>= (4u32.wrapping_sub(num_bits) & 0x3) as usize;
    retval as u16
}

/// Convert bit depths to Huffman symbols.
pub fn convert_bit_depths_to_symbols(depth: &[u8], bits: &mut [u16]) {
    const MAX_BITS: usize = 16;
    let mut bl_count = [0u16; MAX_BITS];

    for &d in depth.iter() {
        bl_count[d as usize] += 1;
    }
    bl_count[0] = 0;

    let mut next_code = [0u16; MAX_BITS];
    let mut code = 0i32;
    for i in 1..MAX_BITS {
        code = (code + bl_count[i - 1] as i32) << 1;
        next_code[i] = code as u16;
    }

    for (i, &d) in depth.iter().enumerate() {
        if d > 0 {
            bits[i] = reverse_bits(d as u32, next_code[d as usize]);
            next_code[d as usize] += 1;
        }
    }
}

/// Storage order for code length codes.
static STORAGE_ORDER: [u8; CODE_LENGTH_CODES] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Huffman codes for encoding code lengths.
static HUFFMAN_BIT_LENGTH_SYMBOLS: [u8; 6] = [0, 7, 3, 2, 1, 15];
static HUFFMAN_BIT_LENGTH_BIT_LENGTHS: [u8; 6] = [2, 4, 3, 2, 2, 4];

/// Store the Huffman tree of Huffman tree to bit mask.
fn store_huffman_tree_of_huffman_tree_to_bit_mask(
    num_codes: usize,
    code_length_bitdepth: &[u8; CODE_LENGTH_CODES],
    writer: &mut BitWriter,
) -> Result<()> {
    // Throw away trailing zeros
    let mut codes_to_store = CODE_LENGTH_CODES;
    if num_codes > 1 {
        while codes_to_store > 0 {
            if code_length_bitdepth[STORAGE_ORDER[codes_to_store - 1] as usize] != 0 {
                break;
            }
            codes_to_store -= 1;
        }
    }

    let mut skip_some = 0usize;
    if code_length_bitdepth[STORAGE_ORDER[0] as usize] == 0
        && code_length_bitdepth[STORAGE_ORDER[1] as usize] == 0
    {
        skip_some = 2;
        if code_length_bitdepth[STORAGE_ORDER[2] as usize] == 0 {
            skip_some = 3;
        }
    }

    writer.write(2, skip_some as u64)?;
    for i in skip_some..codes_to_store {
        let l = code_length_bitdepth[STORAGE_ORDER[i] as usize] as usize;
        writer.write(
            HUFFMAN_BIT_LENGTH_BIT_LENGTHS[l] as usize,
            HUFFMAN_BIT_LENGTH_SYMBOLS[l] as u64,
        )?;
    }
    Ok(())
}

/// Store Huffman tree to bit mask.
fn store_huffman_tree_to_bit_mask(
    huffman_tree_size: usize,
    huffman_tree: &[u8],
    huffman_tree_extra_bits: &[u8],
    code_length_bitdepth: &[u8; CODE_LENGTH_CODES],
    code_length_bitdepth_symbols: &[u16; CODE_LENGTH_CODES],
    writer: &mut BitWriter,
) -> Result<()> {
    for i in 0..huffman_tree_size {
        let ix = huffman_tree[i] as usize;
        writer.write(
            code_length_bitdepth[ix] as usize,
            code_length_bitdepth_symbols[ix] as u64,
        )?;
        // Extra bits
        match ix {
            16 => writer.write(2, huffman_tree_extra_bits[i] as u64)?,
            17 => writer.write(3, huffman_tree_extra_bits[i] as u64)?,
            _ => {}
        }
    }
    Ok(())
}

/// Store a simple Huffman tree (2-4 symbols).
fn store_simple_huffman_tree(
    depths: &[u8],
    symbols: &mut [usize; 4],
    num_symbols: usize,
    max_bits: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    // value of 1 indicates a simple Huffman code
    writer.write(2, 1)?;
    writer.write(2, (num_symbols - 1) as u64)?; // NSYM - 1

    // Sort by depth
    for i in 0..num_symbols {
        for j in (i + 1)..num_symbols {
            if depths[symbols[j]] < depths[symbols[i]] {
                symbols.swap(i, j);
            }
        }
    }

    match num_symbols {
        2 => {
            writer.write(max_bits, symbols[0] as u64)?;
            writer.write(max_bits, symbols[1] as u64)?;
        }
        3 => {
            writer.write(max_bits, symbols[0] as u64)?;
            writer.write(max_bits, symbols[1] as u64)?;
            writer.write(max_bits, symbols[2] as u64)?;
        }
        4 => {
            writer.write(max_bits, symbols[0] as u64)?;
            writer.write(max_bits, symbols[1] as u64)?;
            writer.write(max_bits, symbols[2] as u64)?;
            writer.write(max_bits, symbols[3] as u64)?;
            // tree-select
            writer.write(1, if depths[symbols[0]] == 1 { 1 } else { 0 })?;
        }
        _ => {}
    }
    Ok(())
}

/// Reverse a slice in place.
fn reverse_slice(v: &mut [u8], start: usize, end: usize) {
    let mut start = start;
    let mut end = end - 1;
    while start < end {
        v.swap(start, end);
        start += 1;
        end -= 1;
    }
}

/// Write Huffman tree repetitions for non-zero values.
fn write_huffman_tree_repetitions(
    previous_value: u8,
    value: u8,
    mut repetitions: usize,
    tree_size: &mut usize,
    tree: &mut [u8],
    extra_bits_data: &mut [u8],
) {
    debug_assert!(repetitions > 0);
    if previous_value != value {
        tree[*tree_size] = value;
        extra_bits_data[*tree_size] = 0;
        *tree_size += 1;
        repetitions -= 1;
    }
    if repetitions == 7 {
        tree[*tree_size] = value;
        extra_bits_data[*tree_size] = 0;
        *tree_size += 1;
        repetitions -= 1;
    }
    if repetitions < 3 {
        for _ in 0..repetitions {
            tree[*tree_size] = value;
            extra_bits_data[*tree_size] = 0;
            *tree_size += 1;
        }
    } else {
        repetitions -= 3;
        let start = *tree_size;
        loop {
            tree[*tree_size] = 16;
            extra_bits_data[*tree_size] = (repetitions & 0x3) as u8;
            *tree_size += 1;
            repetitions >>= 2;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        reverse_slice(tree, start, *tree_size);
        reverse_slice(extra_bits_data, start, *tree_size);
    }
}

/// Write Huffman tree repetitions for zero values.
fn write_huffman_tree_repetitions_zeros(
    mut repetitions: usize,
    tree_size: &mut usize,
    tree: &mut [u8],
    extra_bits_data: &mut [u8],
) {
    if repetitions == 11 {
        tree[*tree_size] = 0;
        extra_bits_data[*tree_size] = 0;
        *tree_size += 1;
        repetitions -= 1;
    }
    if repetitions < 3 {
        for _ in 0..repetitions {
            tree[*tree_size] = 0;
            extra_bits_data[*tree_size] = 0;
            *tree_size += 1;
        }
    } else {
        repetitions -= 3;
        let start = *tree_size;
        loop {
            tree[*tree_size] = 17;
            extra_bits_data[*tree_size] = (repetitions & 0x7) as u8;
            *tree_size += 1;
            repetitions >>= 3;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        reverse_slice(tree, start, *tree_size);
        reverse_slice(extra_bits_data, start, *tree_size);
    }
}

/// Decide whether to use RLE encoding for zeros and non-zeros.
fn decide_over_rle_use(depth: &[u8], length: usize) -> (bool, bool) {
    let mut total_reps_zero = 0usize;
    let mut total_reps_non_zero = 0usize;
    let mut count_reps_zero = 1usize;
    let mut count_reps_non_zero = 1usize;

    let mut i = 0;
    while i < length {
        let value = depth[i];
        let mut reps = 1usize;
        let mut k = i + 1;
        while k < length && depth[k] == value {
            reps += 1;
            k += 1;
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

/// Write a Huffman tree from bit depths into the compact representation.
fn write_huffman_tree(
    depth: &[u8],
    length: usize,
    tree_size: &mut usize,
    tree: &mut [u8],
    extra_bits_data: &mut [u8],
) {
    let mut previous_value = 8u8;

    // Throw away trailing zeros
    let mut new_length = length;
    for i in 0..length {
        if depth[length - i - 1] == 0 {
            new_length -= 1;
        } else {
            break;
        }
    }

    // First gather statistics on if it is a good idea to do RLE
    let (use_rle_for_non_zero, use_rle_for_zero) = if length > 50 {
        decide_over_rle_use(depth, new_length)
    } else {
        (false, false)
    };

    // Actual RLE coding
    let mut i = 0;
    while i < new_length {
        let value = depth[i];
        let mut reps = 1usize;
        if (value != 0 && use_rle_for_non_zero) || (value == 0 && use_rle_for_zero) {
            let mut k = i + 1;
            while k < new_length && depth[k] == value {
                reps += 1;
                k += 1;
            }
        }
        if value == 0 {
            write_huffman_tree_repetitions_zeros(reps, tree_size, tree, extra_bits_data);
        } else {
            write_huffman_tree_repetitions(
                previous_value,
                value,
                reps,
                tree_size,
                tree,
                extra_bits_data,
            );
            previous_value = value;
        }
        i += reps;
    }
}

/// Create a Huffman tree from histogram counts.
/// Returns the bit depths for each symbol.
pub fn create_huffman_tree(data: &[u32], length: usize, tree_limit: u8, depth: &mut [u8]) {
    #[derive(Clone, Copy)]
    struct HuffmanNode {
        total_count: u32,
        index_left: i16,
        index_right_or_value: i16,
    }

    fn set_depth(p: &HuffmanNode, pool: &[HuffmanNode], depth: &mut [u8], level: u8) {
        if p.index_left >= 0 {
            let level = level + 1;
            set_depth(&pool[p.index_left as usize], pool, depth, level);
            set_depth(&pool[p.index_right_or_value as usize], pool, depth, level);
        } else {
            depth[p.index_right_or_value as usize] = level;
        }
    }

    for count_limit in (0..).map(|i| 1u32 << i) {
        let mut tree = Vec::new();
        tree.reserve(2 * length + 1);

        for i in (0..length).rev() {
            if data[i] > 0 {
                let count = data[i].max(count_limit.saturating_sub(1));
                tree.push(HuffmanNode {
                    total_count: count,
                    index_left: -1,
                    index_right_or_value: i as i16,
                });
            }
        }

        let n = tree.len();
        if n == 1 {
            // Fake value; will be fixed on upper level
            depth[tree[0].index_right_or_value as usize] = 1;
            break;
        }

        if n == 0 {
            // No symbols at all
            break;
        }

        tree.sort_by(|a, b| a.total_count.cmp(&b.total_count));

        // Add sentinels
        let sentinel = HuffmanNode {
            total_count: u32::MAX,
            index_left: -1,
            index_right_or_value: -1,
        };
        tree.push(sentinel.clone());
        tree.push(sentinel);

        let mut i = 0usize;
        let mut j = n + 1;
        for _ in 0..(n - 1) {
            let left = if tree[i].total_count <= tree[j].total_count {
                let l = i;
                i += 1;
                l
            } else {
                let l = j;
                j += 1;
                l
            };
            let right = if tree[i].total_count <= tree[j].total_count {
                let r = i;
                i += 1;
                r
            } else {
                let r = j;
                j += 1;
                r
            };

            // The sentinel node becomes the parent node
            let j_end = tree.len() - 1;
            tree[j_end].total_count = tree[left].total_count + tree[right].total_count;
            tree[j_end].index_left = left as i16;
            tree[j_end].index_right_or_value = right as i16;

            // Add back the last sentinel node
            tree.push(sentinel.clone());
        }

        debug_assert_eq!(tree.len(), 2 * n + 1);

        // Clear depths
        for d in depth.iter_mut().take(length) {
            *d = 0;
        }

        set_depth(&tree[2 * n - 1], &tree, depth, 0);

        // Check if depths are within limit
        let max_depth = depth.iter().take(length).copied().max().unwrap_or(0);
        if max_depth <= tree_limit {
            break;
        }
    }
}

/// Store a full Huffman tree (for >4 symbols).
fn store_huffman_tree(depths: &[u8], num: usize, writer: &mut BitWriter) -> Result<()> {
    // Write the Huffman tree into the compact representation
    let mut huffman_tree = vec![0u8; num];
    let mut huffman_tree_extra_bits = vec![0u8; num];
    let mut huffman_tree_size = 0usize;

    write_huffman_tree(
        depths,
        num,
        &mut huffman_tree_size,
        &mut huffman_tree,
        &mut huffman_tree_extra_bits,
    );

    // Calculate the statistics of the Huffman tree in the compact representation
    let mut huffman_tree_histogram = [0u32; CODE_LENGTH_CODES];
    for i in 0..huffman_tree_size {
        huffman_tree_histogram[huffman_tree[i] as usize] += 1;
    }

    let mut num_codes = 0;
    let mut code = 0usize;
    for i in 0..CODE_LENGTH_CODES {
        if huffman_tree_histogram[i] > 0 {
            if num_codes == 0 {
                code = i;
                num_codes = 1;
            } else if num_codes == 1 {
                num_codes = 2;
                break;
            }
        }
    }

    // Calculate another Huffman tree to use for compressing the earlier Huffman tree
    let mut code_length_bitdepth = [0u8; CODE_LENGTH_CODES];
    let mut code_length_bitdepth_symbols = [0u16; CODE_LENGTH_CODES];
    create_huffman_tree(
        &huffman_tree_histogram,
        CODE_LENGTH_CODES,
        5,
        &mut code_length_bitdepth,
    );
    convert_bit_depths_to_symbols(&code_length_bitdepth, &mut code_length_bitdepth_symbols);

    // Now, we have all the data, let's start storing it
    store_huffman_tree_of_huffman_tree_to_bit_mask(num_codes, &code_length_bitdepth, writer)?;

    if num_codes == 1 {
        code_length_bitdepth[code] = 0;
    }

    // Store the real Huffman tree now
    store_huffman_tree_to_bit_mask(
        huffman_tree_size,
        &huffman_tree,
        &huffman_tree_extra_bits,
        &code_length_bitdepth,
        &code_length_bitdepth_symbols,
        writer,
    )?;

    Ok(())
}

/// Write a single prefix code.
fn write_prefix_code(code: &PrefixCode, writer: &mut BitWriter) -> Result<()> {
    let mut count = 0usize;
    let mut s4 = [0usize; 4];
    let mut length = 0usize;

    for i in 0..ALPHABET_SIZE {
        if code.depths[i] > 0 {
            if count < 4 {
                s4[count] = i;
            }
            count += 1;
            length = i + 1;
        }
    }

    let mut max_bits_counter = length.saturating_sub(1);
    let mut max_bits = 0usize;
    while max_bits_counter > 0 {
        max_bits_counter >>= 1;
        max_bits += 1;
    }

    if count <= 1 {
        // Single symbol or empty code
        writer.write(4, 1)?; // Simple code marker
        writer.write(max_bits, s4[0] as u64)?;
        return Ok(());
    }

    if count <= 4 {
        store_simple_huffman_tree(&code.depths, &mut s4, count, max_bits, writer)?;
    } else {
        store_huffman_tree(&code.depths, length, writer)?;
    }

    Ok(())
}

/// Write all prefix codes.
pub fn write_prefix_codes(prefix_codes: &[PrefixCode], writer: &mut BitWriter) -> Result<()> {
    #[cfg(feature = "debug-tokens")]
    let start_bits = writer.bits_written();

    writer.write(1, 1)?; // use_prefix_code = true (Huffman, not ANS)

    // Write HybridUint config for each code
    for _ in prefix_codes {
        writer.write(4, 4)?; // split_exponent = 4
        writer.write(3, 2)?; // msb_in_token = 2
        writer.write(2, 0)?; // lsb_in_token = 0
    }

    #[cfg(feature = "debug-tokens")]
    let after_config = writer.bits_written();

    // Write alphabet sizes
    for pc in prefix_codes {
        let mut num_symbol = 1usize;
        for i in 0..ALPHABET_SIZE {
            if pc.depths[i] > 0 {
                num_symbol = i + 1;
            }
        }
        write_var_len_uint16(num_symbol - 1, writer)?;
    }

    #[cfg(feature = "debug-tokens")]
    let after_sizes = writer.bits_written();

    // Write each prefix code
    for (idx, pc) in prefix_codes.iter().enumerate() {
        let mut num_symbol = 1usize;
        for i in 0..ALPHABET_SIZE {
            if pc.depths[i] > 0 {
                num_symbol = i + 1;
            }
        }
        #[cfg(feature = "debug-tokens")]
        let before_code = writer.bits_written();

        if num_symbol > 1 {
            write_prefix_code(pc, writer)?;
        }

        #[cfg(feature = "debug-tokens")]
        {
            let code_bits = writer.bits_written() - before_code;
            if prefix_codes.len() <= 8 && code_bits > 0 {
                let depth_slice: Vec<u8> = pc.depths.iter().take(num_symbol.min(16)).copied().collect();
                debug_log!(
                    "    prefix_code[{}]: num_symbol={}, {} bits, depths={:?}{}",
                    idx, num_symbol, code_bits, depth_slice,
                    if num_symbol > 16 { ", ..." } else { "" }
                );
            }
        }
    }

    #[cfg(feature = "debug-tokens")]
    {
        let total = writer.bits_written() - start_bits;
        debug_log!(
            "  write_prefix_codes: {} codes, config={} bits, sizes={} bits, codes={} bits, total={} bits",
            prefix_codes.len(),
            after_config - start_bits - 1,  // -1 for use_prefix_code bit
            after_sizes - after_config,
            writer.bits_written() - after_sizes,
            total
        );
    }

    Ok(())
}

/// Write the context map.
pub fn write_context_map(code: &EntropyCode, writer: &mut BitWriter) -> Result<()> {
    #[cfg(feature = "debug-tokens")]
    let start_bits = writer.bits_written();

    if code.num_contexts == 0 {
        return Ok(());
    }

    // Check if all context map values are 0
    let max_val = *code.context_map.iter().max().unwrap_or(&0);
    if max_val == 0 {
        writer.write(3, 1)?; // simple code, 0 bits per entry
        return Ok(());
    }

    // Not simple: write 0, no MTF, no LZ77
    writer.write(3, 0)?;

    // Build tokens from context map
    let mut tokens: Vec<Token> = Vec::with_capacity(code.num_contexts);
    for i in 0..code.num_contexts {
        tokens.push(Token::new(0, code.context_map[i] as u32));
    }

    // Build histogram for context map values
    let mut histogram = [0u32; ALPHABET_SIZE];
    for t in &tokens {
        let encoded = UintCoder::encode(t.value);
        histogram[encoded.token as usize] += 1;
    }

    // Create a single prefix code for the context map
    let mut ctxmap_depths = [0u8; ALPHABET_SIZE];
    let mut length = ALPHABET_SIZE;
    while length > 0 && histogram[length - 1] == 0 {
        length -= 1;
    }
    create_huffman_tree(&histogram, length.max(1), 15, &mut ctxmap_depths);

    #[cfg(feature = "debug-tokens")]
    {
        let depth_slice: Vec<u8> = ctxmap_depths.iter().take(length).copied().collect();
        debug_log!(
            "  write_context_map: {} contexts, length={}, depths={:?}",
            code.num_contexts, length, depth_slice
        );
    }

    let mut ctxmap_bits = [0u16; ALPHABET_SIZE];
    convert_bit_depths_to_symbols(&ctxmap_depths, &mut ctxmap_bits);

    let ctxmap_code = PrefixCode {
        depths: ctxmap_depths,
        bits: ctxmap_bits,
    };

    #[cfg(feature = "debug-tokens")]
    let before_prefix = writer.bits_written();

    // Write the prefix code for the context map
    write_prefix_codes(&[ctxmap_code], writer)?;

    #[cfg(feature = "debug-tokens")]
    let after_prefix = writer.bits_written();

    // Write the context map tokens
    for t in &tokens {
        let encoded = UintCoder::encode(t.value);
        let tok = encoded.token as usize;
        let depth = ctxmap_code.depths[tok] as usize;
        let bits = ctxmap_code.bits[tok] as u64;

        // Combine Huffman bits and extra bits
        let data = bits | ((encoded.bits as u64) << depth);
        let total_bits = depth + encoded.nbits as usize;

        writer.write(total_bits, data)?;
    }

    #[cfg(feature = "debug-tokens")]
    {
        let total = writer.bits_written() - start_bits;
        let prefix_bits = after_prefix - before_prefix;
        let token_bits = writer.bits_written() - after_prefix;
        debug_log!(
            "  write_context_map bits: header=3, prefix_code={}, tokens={}, total={}",
            prefix_bits, token_bits, total
        );
    }

    Ok(())
}

/// Write a complete entropy code (context map + prefix codes).
pub fn write_entropy_code(code: &EntropyCode, writer: &mut BitWriter) -> Result<()> {
    write_context_map(code, writer)?;
    write_prefix_codes(code.prefix_codes, writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_code_default() {
        let pc = PrefixCode::default();
        for i in 0..ALPHABET_SIZE {
            assert_eq!(pc.depths[i], 0);
            assert_eq!(pc.bits[i], 0);
        }
    }
}
