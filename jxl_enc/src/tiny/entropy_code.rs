// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Entropy code types and token writing.
//!
//! These are ported from libjxl-tiny and will be used for Huffman coding.

#![allow(dead_code)]

use super::lz77::Lz77Params;
use super::token::{Lz77UintCoder, Token, UintCoder};
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::error::{Error, Result};

use super::token::EncodedUint;

/// Encode a token's value, using the LZ77 uint config when `is_lz77_length` is set.
/// Returns (encoded_uint, symbol_for_histogram) where symbol_for_histogram includes
/// the min_symbol offset for LZ77 length tokens.
#[inline]
fn encode_token_value(token: &Token, lz77: Option<&Lz77Params>) -> (EncodedUint, u32) {
    if token.is_lz77_length {
        let lz77 = lz77.expect("LZ77 length token without LZ77 params");
        let encoded = Lz77UintCoder::encode(token.value);
        let sym = encoded.token + lz77.min_symbol;
        (encoded, sym)
    } else {
        let encoded = UintCoder::encode(token.value);
        (encoded, encoded.token)
    }
}

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
#[derive(Debug, Clone, Copy)]
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
/// This encodes the value using the UintCoder (or Lz77UintCoder for LZ77 length tokens),
/// looks up the prefix code via the context map, and writes the Huffman code followed
/// by extra bits.
#[inline]
pub fn write_token(
    token: &Token,
    code: &EntropyCode,
    lz77: Option<&Lz77Params>,
    writer: &mut BitWriter,
) -> Result<()> {
    let (encoded, sym) = encode_token_value(token, lz77);

    // Look up the prefix code index from the context map
    let prefix_idx = code.context_map[token.context as usize] as usize;
    let pc = &code.prefix_codes[prefix_idx];

    // Get the Huffman code for this token
    let tok = sym as usize;
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
    use super::cluster::{Histogram as TinyHistogram, cluster_histograms};

    // Build per-context histograms using tiny's histogram type
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
        write_token(token, code, lz77, writer)?;
    }
    Ok(())
}

// ============================================================================
// ANS Entropy Coding
// ============================================================================

use crate::entropy_coding::ans::{
    ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution, AnsEncoder,
};

/// Log2 of alphabet size for ANS. With split=4, max token is 4+31=35, so we need 6 bits.
pub const ANS_LOG_ALPHA_SIZE: usize = 6;
/// Alphabet size for ANS (2^6 = 64, matches ALPHABET_SIZE).
pub const ANS_ALPHA_SIZE: usize = 1 << ANS_LOG_ALPHA_SIZE;

/// An owned ANS entropy code (context map + ANS distributions on the heap).
#[derive(Debug)]
pub struct OwnedAnsEntropyCode {
    /// Context map: maps context ID -> distribution index.
    pub context_map: Vec<u8>,
    /// ANS encoding histograms for header serialization.
    pub histograms: Vec<ANSEncodingHistogram>,
    /// ANS distributions for runtime encoding.
    pub distributions: Vec<AnsDistribution>,
    /// Log2 of alphabet size. 6 for normal, 8 when LZ77 is enabled.
    pub log_alpha_size: usize,
}

impl OwnedAnsEntropyCode {
    /// Get the number of distributions.
    pub fn num_distributions(&self) -> usize {
        self.distributions.len()
    }
}

/// Build an ANS entropy code from collected tokens.
///
/// 1. Creates per-context histograms from all tokens.
/// 2. Clusters histograms (max 8 clusters) to produce a context map.
/// 3. Normalizes each cluster histogram to sum to 4096.
/// 4. Builds ANS distributions for encoding.
pub fn build_entropy_code_ans(tokens: &[Token], num_contexts: usize) -> OwnedAnsEntropyCode {
    build_entropy_code_ans_with_options(tokens, num_contexts, false, None)
}

/// Build an ANS entropy code with optional enhanced clustering.
///
/// When `lz77` is Some, LZ77 length tokens use Lz77UintCoder and are offset by min_symbol.
pub fn build_entropy_code_ans_with_options(
    tokens: &[Token],
    num_contexts: usize,
    enhanced_clustering: bool,
    lz77: Option<&Lz77Params>,
) -> OwnedAnsEntropyCode {
    use crate::entropy_coding::cluster::{
        ClusteringType, EntropyType, cluster_histograms as enhanced_cluster,
    };
    use crate::entropy_coding::histogram::Histogram as EnhancedHistogram;

    // Build per-context histograms
    let mut histograms: Vec<EnhancedHistogram> = (0..num_contexts)
        .map(|_| EnhancedHistogram::new())
        .collect();

    for token in tokens {
        let ctx = token.context as usize;
        let (_encoded, sym) = encode_token_value(token, lz77);
        if ctx < histograms.len() {
            histograms[ctx].add(sym as usize);
        }
    }

    // Cluster histograms
    let cluster_type = if enhanced_clustering {
        ClusteringType::Best
    } else {
        ClusteringType::Fast
    };

    // Limit to 8 clusters to use simple context map format
    // (simple format supports max 3 bits per entry = 8 histograms)
    let result = enhanced_cluster(cluster_type, EntropyType::Ans, &histograms, 8)
        .expect("ANS clustering failed");

    let context_map: Vec<u8> = result.symbols.iter().map(|&s| s as u8).collect();

    // Verify context map size matches input
    debug_assert_eq!(
        context_map.len(),
        num_contexts,
        "ANS context map size {} doesn't match num_contexts {}",
        context_map.len(),
        num_contexts
    );

    // Build ANS histograms and distributions from clustered histograms
    let mut ans_histograms = Vec::with_capacity(result.histograms.len());
    let mut ans_distributions = Vec::with_capacity(result.histograms.len());

    for histo in &result.histograms {
        // Create ANS encoding histogram (normalized)
        let ans_histo = ANSEncodingHistogram::from_histogram(histo, ANSHistogramStrategy::Precise)
            .expect("ANS histogram normalization failed");

        // Build ANS distribution for encoding
        let ans_dist = AnsDistribution::from_normalized_counts(&ans_histo.counts)
            .expect("ANS distribution building failed");

        ans_histograms.push(ans_histo);
        ans_distributions.push(ans_dist);
    }

    // Use log_alpha_size=8 when LZ77 is enabled (symbols up to 255),
    // otherwise log_alpha_size=6 (symbols up to 63).
    let log_alpha_size = if lz77.is_some_and(|p| p.enabled) {
        8
    } else {
        ANS_LOG_ALPHA_SIZE
    };

    let code = OwnedAnsEntropyCode {
        context_map,
        histograms: ans_histograms,
        distributions: ans_distributions,
        log_alpha_size,
    };

    // Validate: every token in the stream must have a valid, non-zero frequency
    // in the distribution it maps to.
    for (i, token) in tokens.iter().enumerate() {
        let ctx = token.context as usize;
        let (_encoded, sym) = encode_token_value(token, lz77);
        let dist_idx = code.context_map.get(ctx).copied().unwrap_or(0) as usize;
        let dist = &code.distributions[dist_idx];
        let tok = sym as usize;
        if tok >= dist.symbols.len() {
            panic!(
                "ANS validation: token[{}] ctx={} val={} tok={} exceeds distribution alphabet_size={} (dist_idx={})",
                i,
                ctx,
                token.value,
                tok,
                dist.symbols.len(),
                dist_idx
            );
        }
        if dist.symbols[tok].freq == 0 {
            panic!(
                "ANS validation: token[{}] ctx={} val={} tok={} has zero frequency in distribution (dist_idx={})",
                i, ctx, token.value, tok, dist_idx
            );
        }
    }

    code
}

/// Write ANS entropy code header (context map + distributions).
pub fn write_entropy_code_ans(code: &OwnedAnsEntropyCode, writer: &mut BitWriter) -> Result<()> {
    #[cfg(feature = "debug-tokens")]
    {
        eprintln!("write_entropy_code_ans:");
        eprintln!("  num_contexts: {}", code.context_map.len());
        eprintln!("  num_histograms: {}", code.histograms.len());
        eprintln!(
            "  context_map: {:?}",
            &code.context_map[..code.context_map.len().min(20)]
        );
        for (i, h) in code.histograms.iter().enumerate() {
            eprintln!(
                "  histogram[{}]: alphabet_size={}, method={}, counts[..8]={:?}",
                i,
                h.alphabet_size,
                h.method,
                &h.counts[..h.counts.len().min(8)]
            );
        }
    }

    // Write context map (same format as Huffman)
    // Note: LZ77 is already written by the caller (write_dc_global or write_ac_global)
    let _cm_start = writer.bits_written();
    write_context_map_for_ans(code, writer)?;

    #[cfg(feature = "debug-tokens")]
    eprintln!("  context_map: {} bits", writer.bits_written() - _cm_start);

    // Write use_prefix_code = 0 (use ANS, not Huffman)
    writer.write(1, 0)?;

    // Write log_alpha_size - 5
    let las = code.log_alpha_size;
    writer.write(2, (las - 5) as u64)?;

    #[cfg(feature = "debug-tokens")]
    eprintln!("  use_prefix_code=0, log_alpha_size={}", las);

    // Write HybridUint configs for each histogram
    let _cfg_start = writer.bits_written();
    for _ in &code.histograms {
        write_hybrid_uint_config(las, writer)?;
    }

    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "  HybridUint configs: {} bits ({} histograms)",
        writer.bits_written() - _cfg_start,
        code.histograms.len()
    );

    // Write ANS distributions
    let _hist_start = writer.bits_written();
    #[allow(clippy::unused_enumerate_index)]
    for (_i, histo) in code.histograms.iter().enumerate() {
        let _h_start = writer.bits_written();
        histo.write(writer)?;
        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "  histogram[{}]: {} bits",
            _i,
            writer.bits_written() - _h_start
        );
    }

    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "  All histograms: {} bits",
        writer.bits_written() - _hist_start
    );

    Ok(())
}

/// Write context map for ANS entropy code.
fn write_context_map_for_ans(code: &OwnedAnsEntropyCode, writer: &mut BitWriter) -> Result<()> {
    let num_histograms = code.histograms.len();

    if num_histograms == 1 {
        // Simple context map: all contexts map to histogram 0
        writer.write(1, 1)?; // simple_context_map = true
        writer.write(2, 0)?; // nbits = 0
    } else if num_histograms <= 8
        && code
            .context_map
            .iter()
            .all(|&c| (c as usize) < num_histograms)
    {
        // Simple context map with multiple histograms
        // bits_per_entry: 0 = all zeros (handled above), 1 = 2 histos, 2 = 4 histos, 3 = 8 histos
        let nbits = if num_histograms <= 2 {
            1
        } else if num_histograms <= 4 {
            2
        } else {
            3
        };
        writer.write(1, 1)?; // simple_context_map = true
        writer.write(2, nbits as u64)?;

        for &ctx in &code.context_map {
            writer.write(nbits, ctx as u64)?;
        }
    } else {
        // Complex context map is not yet implemented for ANS
        return Err(Error::NotImplemented(format!(
            "ANS context map with {} histograms (max 8)",
            num_histograms
        )));
    }

    Ok(())
}

/// Write HybridUint config (split_exponent, msb_in_token, lsb_in_token).
///
/// Uses the same config as our UintCoder: split=4, msb=2, lsb=0.
fn write_hybrid_uint_config(log_alpha_size: usize, writer: &mut BitWriter) -> Result<()> {
    let split_exponent = 4u32;
    let msb_in_token = 2u32;
    let lsb_in_token = 0u32;

    // CeilLog2Nonzero(log_alpha_size + 1) bits for split_exponent
    let se_bits = ceil_log2_nonzero_usize(log_alpha_size + 1);
    writer.write(se_bits, split_exponent as u64)?;

    if split_exponent as usize == log_alpha_size {
        // msb/lsb don't matter when split_exponent == log_alpha_size
        return Ok(());
    }

    // CeilLog2Nonzero(split_exponent + 1) bits for msb_in_token
    let msb_bits = ceil_log2_nonzero_usize(split_exponent as usize + 1);
    writer.write(msb_bits, msb_in_token as u64)?;

    // CeilLog2Nonzero(split_exponent - msb_in_token + 1) bits for lsb_in_token
    let lsb_bits = ceil_log2_nonzero_usize((split_exponent - msb_in_token) as usize + 1);
    writer.write(lsb_bits, lsb_in_token as u64)?;

    Ok(())
}

/// CeilLog2Nonzero for usize, matching libjxl.
fn ceil_log2_nonzero_usize(x: usize) -> usize {
    debug_assert!(x > 0);
    let x = x as u32;
    let floor = 31 - x.leading_zeros();
    if x.is_power_of_two() {
        floor as usize
    } else {
        (floor + 1) as usize
    }
}

/// Write tokens using ANS entropy coding.
///
/// Tokens are processed in reverse order (ANS requirement), and the output
/// is written in the correct forward order for the decoder.
pub fn write_tokens_ans(
    tokens: &[Token],
    code: &OwnedAnsEntropyCode,
    lz77: Option<&Lz77Params>,
    writer: &mut BitWriter,
) -> Result<()> {
    let mut encoder = AnsEncoder::new();

    #[cfg(feature = "debug-tokens")]
    {
        eprintln!(
            "write_tokens_ans: {} tokens, {} distributions, context_map len={}",
            tokens.len(),
            code.distributions.len(),
            code.context_map.len()
        );
        eprintln!("  initial state: 0x{:08x}", encoder.state());
    }

    // Process tokens in reverse order
    #[allow(clippy::unused_enumerate_index)]
    for (_i, token) in tokens.iter().rev().enumerate() {
        let ctx = token.context as usize;
        let (encoded, sym) = encode_token_value(token, lz77);

        // Get the distribution for this context
        let dist_idx = code.context_map.get(ctx).copied().unwrap_or(0) as usize;
        let dist = code.distributions.get(dist_idx).unwrap_or_else(|| {
            panic!(
                "ANS: missing distribution at index {} for context {}",
                dist_idx, ctx
            )
        });

        // Push extra bits first (they come after the symbol in forward order)
        encoder.push_bits(encoded.bits, encoded.nbits as u8);

        // Push the ANS symbol
        let info = dist.get(sym as usize).unwrap_or_else(|| {
            panic!(
                "ANS: symbol {} not in distribution (ctx={}, dist_idx={})",
                sym, ctx, dist_idx
            )
        });

        #[cfg(feature = "debug-tokens")]
        if _i < 5 || _i >= tokens.len() - 3 {
            eprintln!(
                "  token[{}]: ctx={}, val={}, tok={}, freq={}, state before=0x{:08x}",
                tokens.len() - 1 - _i,
                ctx,
                token.value,
                sym,
                info.freq,
                encoder.state()
            );
        }

        encoder.put_symbol(info);
    }

    #[cfg(feature = "debug-tokens")]
    eprintln!("  final state: 0x{:08x}", encoder.state());

    // Finalize: writes state + reversed bits
    encoder.finalize(writer)?;

    Ok(())
}

/// Verify that each ANS histogram serializes and deserializes correctly.
///
/// Writes each histogram to bits, decodes it back with our decoder, and compares frequencies.
pub fn verify_histogram_serialization(code: &OwnedAnsEntropyCode, label: &str) -> Result<()> {
    use crate::entropy_coding::ans_decode::{AnsHistogram, BitReader};

    for (i, histo) in code.histograms.iter().enumerate() {
        // Write histogram to bits
        let mut writer = BitWriter::new();
        histo.write(&mut writer)?;
        writer.zero_pad_to_byte();
        let bytes = writer.finish();

        // Decode it back
        let mut br = BitReader::new(&bytes);
        let decoded = match AnsHistogram::decode(&mut br, code.log_alpha_size) {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "  {} histogram[{}]: DECODE FAILED - {} (method={}, alphabet_size={}, omit_pos={})",
                    label, i, e, histo.method, histo.alphabet_size, histo.omit_pos
                );
                eprintln!(
                    "    counts[..16]: {:?}",
                    &histo.counts[..histo.counts.len().min(16)]
                );
                eprintln!("    bytes: {:02x?}", &bytes[..bytes.len().min(32)]);
                return Err(e);
            }
        };

        // Compare frequencies
        let mut mismatch = false;
        for j in 0..histo.alphabet_size {
            let expected = histo.counts[j] as u16;
            let got = decoded.frequencies[j];
            if expected != got {
                if !mismatch {
                    eprintln!(
                        "  {} histogram[{}]: FREQUENCY MISMATCH (method={}, alphabet_size={})",
                        label, i, histo.method, histo.alphabet_size
                    );
                }
                eprintln!("    symbol[{}]: expected={}, got={}", j, expected, got);
                mismatch = true;
            }
        }

        if mismatch {
            eprintln!("    counts: {:?}", &histo.counts[..histo.alphabet_size]);
            // Check omit_pos: what would the decoder pick vs what encoder used?
            let mut encoder_omit_logcount = 0u32;
            let mut encoder_omit = 0;
            for (k, &c) in histo.counts.iter().enumerate().take(histo.alphabet_size) {
                if c > 0 {
                    let lc = crate::entropy_coding::ans::floor_log2_ans(c as u32) + 1;
                    if lc > encoder_omit_logcount {
                        encoder_omit_logcount = lc;
                        encoder_omit = k;
                    }
                }
            }
            eprintln!(
                "    encoder omit_pos={} (from stored histo: method={}, omit_pos={})",
                encoder_omit, histo.method, histo.omit_pos
            );
            eprintln!(
                "    encoder omit logcount={}, count at omit={}",
                encoder_omit_logcount, histo.counts[histo.omit_pos]
            );
            // Check what decoder would see
            for k in 0..histo.alphabet_size.min(40) {
                let c = histo.counts[k];
                if c > 0 {
                    let lc = crate::entropy_coding::ans::floor_log2_ans(c as u32) + 1;
                    if lc == encoder_omit_logcount {
                        eprintln!(
                            "    symbol[{}]: count={}, logcount={} (same as max)",
                            k, c, lc
                        );
                    }
                }
            }
            return Err(Error::InvalidHistogram(format!(
                "{} histogram[{}] serialization roundtrip failed",
                label, i
            )));
        }

        // Histogram OK - only log when debug-tokens feature is enabled
        #[cfg(feature = "debug-tokens")]
        {
            let method_desc = match histo.method {
                0 => "flat",
                1 => "small",
                _ => "general",
            };
            eprintln!(
                "  {} histogram[{}]: OK ({}, {} symbols, {} bytes)",
                label,
                i,
                method_desc,
                histo.alphabet_size,
                bytes.len()
            );
        }
    }

    Ok(())
}

/// Verify ANS roundtrip: encode tokens, then decode with our local decoder.
///
/// Returns Ok(()) if all decoded symbols match, or Err with details of first mismatch.
/// This is the critical invariant test for ANS encoding correctness.
pub fn verify_ans_roundtrip(tokens: &[Token], code: &OwnedAnsEntropyCode) -> Result<()> {
    use crate::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};

    if tokens.is_empty() {
        return Ok(());
    }

    // Step 1: Write the ANS-encoded histogram header + tokens to a buffer
    let mut header_writer = BitWriter::new();
    write_entropy_code_ans(code, &mut header_writer)?;
    let header_bits = header_writer.bits_written();

    let mut token_writer = BitWriter::new();
    write_tokens_ans(tokens, code, None, &mut token_writer)?;
    let _token_bits = token_writer.bits_written();

    // Combine header + tokens into one buffer for decoding
    let mut combined_writer = BitWriter::new();
    write_entropy_code_ans(code, &mut combined_writer)?;
    write_tokens_ans(tokens, code, None, &mut combined_writer)?;
    combined_writer.zero_pad_to_byte();
    let encoded_bytes = combined_writer.finish();

    // Step 2: Decode the histogram header
    let mut br = BitReader::new(&encoded_bytes);

    // Read context map
    let _num_histograms = code.histograms.len();
    let _simple = br.read(1)?; // simple_context_map flag
    // Skip full context map decoding — we'll decode each histogram directly.
    // Instead, just skip to where the histograms start by re-reading the full header.
    let mut br2 = BitReader::new(&encoded_bytes);

    // We need to skip the header and go straight to the token data.
    // The easiest way: just read past header_bits.
    for _ in 0..header_bits {
        br2.read(1)?;
    }

    // Step 3: Decode ANS tokens
    let mut ans_reader = AnsReader::init(&mut br2)?;

    // Decode each histogram from the full header for verification
    let _br_hist = BitReader::new(&encoded_bytes);
    // Skip context map to get to histograms...
    // Actually, let's take a simpler approach: decode histograms independently
    // and build decoder tables from the encoder's known frequencies.

    // Build decoder histograms directly from the encoder's known distributions
    let decoder_histograms: Vec<AnsHistogram> = code
        .distributions
        .iter()
        .map(|dist| {
            // Build frequency array for the decoder
            let mut freqs = vec![0u16; dist.symbols.len().max(64)];
            for (i, sym) in dist.symbols.iter().enumerate() {
                freqs[i] = sym.freq;
            }

            // Build alias map using the decoder's method
            let log_bucket_size = 12 - 6; // LOG_SUM_PROBS - log_alpha_size
            let bucket_size = 1u16 << log_bucket_size;
            let bucket_mask = bucket_size as u32 - 1;

            // Check for single-symbol case
            if let Some(single_idx) = freqs.iter().position(|&f| f == 4096) {
                let buckets = freqs
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| crate::entropy_coding::ans_decode::Bucket {
                        dist: f,
                        alias_symbol: single_idx as u8,
                        alias_offset: bucket_size * i as u16,
                        alias_cutoff: 0,
                        alias_dist_xor: f ^ 4096,
                    })
                    .collect();
                return AnsHistogram {
                    buckets,
                    log_bucket_size,
                    bucket_mask,
                    single_symbol: Some(single_idx as u32),
                    frequencies: freqs,
                };
            }

            let buckets =
                AnsHistogram::build_alias_map_from_freqs(freqs.len(), log_bucket_size, &freqs);
            AnsHistogram {
                buckets,
                log_bucket_size,
                bucket_mask,
                single_symbol: None,
                frequencies: freqs,
            }
        })
        .collect();

    // Step 4: Decode tokens and compare
    let mut mismatches = 0;
    for (i, token) in tokens.iter().enumerate() {
        let ctx = token.context as usize;
        let dist_idx = code.context_map.get(ctx).copied().unwrap_or(0) as usize;
        let decoder_hist = &decoder_histograms[dist_idx];

        // Decode one ANS symbol
        let decoded_symbol = decoder_hist.read(&mut br2, &mut ans_reader.0);

        // Read extra bits (HybridUint)
        let expected = UintCoder::encode(token.value);
        let decoded_extra = if expected.nbits > 0 {
            br2.read(expected.nbits as usize).unwrap_or(0) as u32
        } else {
            0
        };

        // Compare token (ANS symbol)
        if decoded_symbol != expected.token {
            if mismatches < 5 {
                eprintln!(
                    "ANS roundtrip MISMATCH at token[{}]: ctx={}, val={}, expected_tok={}, decoded_tok={}, state=0x{:08x}",
                    i, ctx, token.value, expected.token, decoded_symbol, ans_reader.0
                );
            }
            mismatches += 1;
        }

        // Compare extra bits
        if decoded_extra != expected.bits {
            if mismatches < 5 {
                eprintln!(
                    "ANS roundtrip extra bits MISMATCH at token[{}]: expected_bits=0x{:x}, decoded_bits=0x{:x}",
                    i, expected.bits, decoded_extra
                );
            }
            mismatches += 1;
        }
    }

    // Step 5: Verify final state
    if let Err(e) = ans_reader.check_final_state() {
        return Err(Error::Bitstream(format!(
            "ANS roundtrip final state check failed ({} token mismatches): {}",
            mismatches, e
        )));
    }

    if mismatches > 0 {
        return Err(Error::Bitstream(format!(
            "ANS roundtrip had {} mismatches out of {} tokens",
            mismatches,
            tokens.len()
        )));
    }

    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "ANS roundtrip OK: {} tokens, header={} bits, data={} bits",
        tokens.len(),
        header_bits,
        _token_bits
    );

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
