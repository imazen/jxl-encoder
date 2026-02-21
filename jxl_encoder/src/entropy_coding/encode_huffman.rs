// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Huffman entropy code building and serialization.
//!
//! Contains all Huffman-specific code: tree building, prefix code writing,
//! context map writing, and token writing for Huffman-coded bitstreams.

#![allow(dead_code)]

use super::encode::{
    ALPHABET_SIZE, CODE_LENGTH_CODES, EntropyCode, PrefixCode, encode_token_value,
    write_var_len_uint16,
};
use super::lz77::Lz77Params;
use super::token::{Token, UintCoder};
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::error::Result;

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

    fn set_depth(pool: &[HuffmanNode], root_idx: usize, depth: &mut [u8]) {
        // Iterative DFS to avoid stack overflow on deep/degenerate Huffman trees.
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

    for count_limit in (0..).map(|i| 1u32 << i) {
        let mut tree = Vec::with_capacity(2 * length + 1);

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

        tree.sort_by_key(|a| a.total_count);

        // Add sentinels
        let sentinel = HuffmanNode {
            total_count: u32::MAX,
            index_left: -1,
            index_right_or_value: -1,
        };
        tree.push(sentinel);
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
            tree.push(sentinel);
        }

        debug_assert_eq!(tree.len(), 2 * n + 1);

        // Clear depths
        for d in depth.iter_mut().take(length) {
            *d = 0;
        }

        set_depth(&tree, 2 * n - 1, depth);

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
    for (i, &hist_val) in huffman_tree_histogram
        .iter()
        .enumerate()
        .take(CODE_LENGTH_CODES)
    {
        if hist_val > 0 {
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
pub(super) fn write_prefix_code(code: &PrefixCode, writer: &mut BitWriter) -> Result<()> {
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
    #[allow(clippy::unused_enumerate_index)]
    for (_idx, pc) in prefix_codes.iter().enumerate() {
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
                let depth_slice: Vec<u8> =
                    pc.depths.iter().take(num_symbol.min(16)).copied().collect();
                debug_log!(
                    "    prefix_code[{}]: num_symbol={}, {} bits, depths={:?}{}",
                    _idx,
                    num_symbol,
                    code_bits,
                    depth_slice,
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
            after_config - start_bits - 1, // -1 for use_prefix_code bit
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
            code.num_contexts,
            length,
            depth_slice
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
            prefix_bits,
            token_bits,
            total
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

/// An owned entropy code (context map + prefix codes on the heap).
///
/// Unlike `EntropyCode` which borrows from static data, this holds owned
/// data built dynamically from actual token frequencies.
pub struct OwnedEntropyCode {
    /// Context map: maps context ID -> prefix code index.
    pub context_map: Vec<u8>,
    /// Prefix codes (Huffman codes).
    pub prefix_codes: Vec<PrefixCode>,
}

impl OwnedEntropyCode {
    /// Borrow as an `EntropyCode` for use with `write_token` etc.
    pub fn as_entropy_code(&self) -> EntropyCode<'_> {
        EntropyCode::new(&self.context_map, &self.prefix_codes)
    }
}

/// Build an optimal entropy code from collected tokens.
///
/// 1. Creates per-context histograms from all tokens.
/// 2. Clusters histograms (max 8 clusters) to produce a context map.
/// 3. Builds a Huffman tree for each cluster.
///
/// Returns an `OwnedEntropyCode` ready for writing.
pub fn build_entropy_code(tokens: &[Token], num_contexts: usize) -> OwnedEntropyCode {
    build_entropy_code_with_options(tokens, num_contexts, false, None)
}

/// Build an optimal entropy code from collected tokens with optional enhanced clustering.
///
/// When `enhanced_clustering` is true, uses pair merge refinement. Note that the enhanced
/// clustering algorithm was designed for ANS entropy coding and the cost model may not
/// accurately predict Huffman code sizes. This is experimental.
///
/// When `lz77` is Some, LZ77 length tokens use Lz77UintCoder and are offset by min_symbol.
pub fn build_entropy_code_with_options(
    tokens: &[Token],
    num_contexts: usize,
    enhanced_clustering: bool,
    lz77: Option<&Lz77Params>,
) -> OwnedEntropyCode {
    use crate::vardct::cluster::{Histogram as TinyHistogram, cluster_histograms};

    // Build per-context histograms using VarDCT's histogram type
    let mut histograms: Vec<TinyHistogram> =
        (0..num_contexts).map(|_| TinyHistogram::new()).collect();
    for token in tokens {
        let ctx = token.context as usize;
        let (_encoded, sym) = encode_token_value(token, lz77);
        histograms[ctx].add(sym as usize);
    }

    let (context_map, clustered_histograms) = if enhanced_clustering {
        // Use the enhanced clustering from entropy_coding::cluster
        use crate::entropy_coding::cluster::{
            ClusteringType, EntropyType, cluster_histograms as enhanced_cluster,
        };
        use crate::entropy_coding::histogram::Histogram as EnhancedHistogram;

        // Convert tiny histograms to enhanced histogram type
        let enhanced_histos: Vec<EnhancedHistogram> = histograms
            .iter()
            .map(|h| {
                let counts: Vec<i32> = h.counts.iter().map(|&c| c as i32).collect();
                EnhancedHistogram::from_counts(&counts)
            })
            .collect();

        // Run enhanced clustering with pair merge refinement
        // Use Huffman cost model since tiny encoder uses Huffman codes
        let result = enhanced_cluster(
            ClusteringType::Best,
            EntropyType::Huffman,
            &enhanced_histos,
            8,
        )
        .expect("Enhanced clustering failed");

        // Convert back to tiny histogram type for Huffman tree building
        let out_histos: Vec<TinyHistogram> = result
            .histograms
            .iter()
            .map(|h| {
                let mut th = TinyHistogram::new();
                for (i, &count) in h.counts.iter().enumerate() {
                    if i < ALPHABET_SIZE {
                        th.counts[i] = count as u32;
                        th.total_count += count as u32;
                    }
                }
                th
            })
            .collect();

        // Convert symbols to context map
        let ctx_map: Vec<u8> = result.symbols.iter().map(|&s| s as u8).collect();
        (ctx_map, out_histos)
    } else {
        // Use the simple fast clustering
        let context_map = cluster_histograms(&mut histograms);
        (context_map, histograms)
    };

    // Build a PrefixCode from each clustered histogram
    let prefix_codes: Vec<PrefixCode> = clustered_histograms
        .iter()
        .map(|h| {
            let mut depths = [0u8; ALPHABET_SIZE];
            let mut bits = [0u16; ALPHABET_SIZE];
            if h.total_count > 0 {
                create_huffman_tree(&h.counts, ALPHABET_SIZE, 15, &mut depths);
            } else {
                // Empty histogram: single-symbol code for symbol 0
                depths[0] = 1;
            }
            convert_bit_depths_to_symbols(&depths, &mut bits);
            PrefixCode { depths, bits }
        })
        .collect();

    OwnedEntropyCode {
        context_map,
        prefix_codes,
    }
}

/// Write pre-collected tokens using the given entropy code.
pub fn write_tokens(
    tokens: &[Token],
    code: &EntropyCode,
    lz77: Option<&Lz77Params>,
    writer: &mut BitWriter,
) -> Result<()> {
    for token in tokens {
        super::encode::write_token(token, code, lz77, writer)?;
    }
    Ok(())
}
