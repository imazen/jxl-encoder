// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Context tree tokens for modular DC header.
//!
//! Ported from libjxl-tiny enc_frame.cc

use super::cluster::{cluster_histograms, Histogram};
use super::common::pack_signed;
use super::entropy_code::{
    convert_bit_depths_to_symbols, create_huffman_tree, write_entropy_code, write_prefix_codes,
    write_token, EntropyCode, PrefixCode, ALPHABET_SIZE,
};
use super::token::{Token, UintCoder};
use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Number of contexts for the context tree.
pub const NUM_TREE_CONTEXTS: usize = 6;

/// Number of tokens in the context tree.
pub const NUM_CONTEXT_TREE_TOKENS: usize = 313;

/// Context tree tokens for modular stream DC coding.
/// From libjxl-tiny enc_frame.cc:181-222.
///
/// Format: (context, value) pairs
pub static CONTEXT_TREE_TOKENS: [(u32, u32); NUM_CONTEXT_TREE_TOKENS] = [
    (1, 2),   (0, 4),  (1, 1),   (0, 2),  (1, 10),   (0, 0),  (1, 1),   (0, 4),
    (1, 1),   (0, 0),  (1, 10),  (0, 94), (1, 10),   (0, 61), (1, 0),   (2, 0),
    (3, 0),   (4, 0),  (5, 0),   (1, 3),  (0, 0),    (1, 0),  (2, 5),   (3, 0),
    (4, 0),   (5, 0),  (1, 0),   (2, 5),  (3, 0),    (4, 0),  (5, 0),   (1, 10),
    (0, 382), (1, 10), (0, 22),  (1, 10), (0, 13),   (1, 10), (0, 253), (1, 8),
    (0, 10),  (1, 8),  (0, 10),  (1, 10), (0, 784),  (1, 10), (0, 190), (1, 10),
    (0, 46),  (1, 10), (0, 10),  (1, 10), (0, 5),    (1, 10), (0, 29),  (1, 10),
    (0, 125), (1, 10), (0, 509), (1, 8),  (0, 22),   (1, 8),  (0, 6),   (1, 8),
    (0, 22),  (1, 8),  (0, 6),   (1, 10), (0, 1000), (1, 10), (0, 510), (1, 10),
    (0, 254), (1, 10), (0, 126), (1, 10), (0, 62),   (1, 10), (0, 30),  (1, 10),
    (0, 14),  (1, 10), (0, 6),   (1, 10), (0, 1),    (1, 10), (0, 7),   (1, 10),
    (0, 21),  (1, 10), (0, 45),  (1, 10), (0, 93),   (1, 10), (0, 189), (1, 10),
    (0, 381), (1, 10), (0, 783), (1, 0),  (2, 1),    (3, 0),  (4, 0),   (5, 0),
    (1, 0),   (2, 1),  (3, 0),   (4, 0),  (5, 0),    (1, 0),  (2, 1),   (3, 0),
    (4, 0),   (5, 0),  (1, 0),   (2, 1),  (3, 0),    (4, 0),  (5, 0),   (1, 0),
    (2, 0),   (3, 0),  (4, 0),   (5, 0),  (1, 0),    (2, 0),  (3, 0),   (4, 0),
    (5, 0),   (1, 0),  (2, 0),   (3, 0),  (4, 0),    (5, 0),  (1, 0),   (2, 0),
    (3, 0),   (4, 0),  (5, 0),   (1, 0),  (2, 5),    (3, 0),  (4, 0),   (5, 0),
    (1, 0),   (2, 5),  (3, 0),   (4, 0),  (5, 0),    (1, 0),  (2, 5),   (3, 0),
    (4, 0),   (5, 0),  (1, 0),   (2, 5),  (3, 0),    (4, 0),  (5, 0),   (1, 0),
    (2, 5),   (3, 0),  (4, 0),   (5, 0),  (1, 0),    (2, 5),  (3, 0),   (4, 0),
    (5, 0),   (1, 0),  (2, 5),   (3, 0),  (4, 0),    (5, 0),  (1, 0),   (2, 5),
    (3, 0),   (4, 0),  (5, 0),   (1, 0),  (2, 5),    (3, 0),  (4, 0),   (5, 0),
    (1, 0),   (2, 5),  (3, 0),   (4, 0),  (5, 0),    (1, 0),  (2, 5),   (3, 0),
    (4, 0),   (5, 0),  (1, 0),   (2, 5),  (3, 0),    (4, 0),  (5, 0),   (1, 0),
    (2, 5),   (3, 0),  (4, 0),   (5, 0),  (1, 0),    (2, 5),  (3, 0),   (4, 0),
    (5, 0),   (1, 0),  (2, 5),   (3, 0),  (4, 0),    (5, 0),  (1, 10),  (0, 2),
    (1, 0),   (2, 5),  (3, 0),   (4, 0),  (5, 0),    (1, 0),  (2, 5),   (3, 0),
    (4, 0),   (5, 0),  (1, 0),   (2, 5),  (3, 0),    (4, 0),  (5, 0),   (1, 0),
    (2, 5),   (3, 0),  (4, 0),   (5, 0),  (1, 0),    (2, 5),  (3, 0),   (4, 0),
    (5, 0),   (1, 0),  (2, 5),   (3, 0),  (4, 0),    (5, 0),  (1, 0),   (2, 5),
    (3, 0),   (4, 0),  (5, 0),   (1, 0),  (2, 5),    (3, 0),  (4, 0),   (5, 0),
    (1, 0),   (2, 5),  (3, 0),   (4, 0),  (5, 0),    (1, 0),  (2, 5),   (3, 0),
    (4, 0),   (5, 0),  (1, 0),   (2, 5),  (3, 0),    (4, 0),  (5, 0),   (1, 0),
    (2, 5),   (3, 0),  (4, 0),   (5, 0),  (1, 0),    (2, 5),  (3, 0),   (4, 0),
    (5, 0),   (1, 0),  (2, 5),   (3, 0),  (4, 0),    (5, 0),  (1, 0),   (2, 5),
    (3, 0),   (4, 0),  (5, 0),   (1, 10), (0, 999),  (1, 0),  (2, 5),   (3, 0),
    (4, 0),   (5, 0),  (1, 0),   (2, 5),  (3, 0),    (4, 0),  (5, 0),   (1, 0),
    (2, 5),   (3, 0),  (4, 0),   (5, 0),  (1, 0),    (2, 5),  (3, 0),   (4, 0),
    (5, 0),
];

/// Compact block context map for AC coefficient coding.
/// From libjxl-tiny ac_context.h:45-48.
pub static COMPACT_BLOCK_CONTEXT_MAP: [u8; 39] = [
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // Y
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,  // X
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,  // B
];

/// Build an optimized entropy code for the context tree tokens.
///
/// This builds histograms from the tokens, clusters them, then creates Huffman codes.
fn build_context_tree_entropy_code(tokens: &[Token]) -> (Vec<u8>, Vec<PrefixCode>) {
    // Build histograms for each context using the Histogram struct
    let mut histograms: Vec<Histogram> = (0..NUM_TREE_CONTEXTS).map(|_| Histogram::new()).collect();

    for token in tokens {
        let encoded = UintCoder::encode(token.value);
        let ctx = token.context as usize;
        histograms[ctx].add(encoded.token as usize);
    }

    // Cluster similar histograms together
    // Note: cluster_histograms modifies histograms in place and returns context_map
    let context_map = cluster_histograms(&mut histograms);

    // Build Huffman codes for each clustered histogram
    let mut prefix_codes = Vec::with_capacity(histograms.len());
    for (i, hist) in histograms.iter().enumerate() {
        let mut depths = [0u8; ALPHABET_SIZE];
        let mut length = ALPHABET_SIZE;
        while length > 0 && hist.counts[length - 1] == 0 {
            length -= 1;
        }
        if length == 0 {
            length = 1;
        }
        create_huffman_tree(&hist.counts, length, 15, &mut depths);

        let mut bits = [0u16; ALPHABET_SIZE];
        convert_bit_depths_to_symbols(&depths, &mut bits);

        #[cfg(feature = "debug-tokens")]
        {
            let depth_slice: Vec<u8> = depths.iter().take(length.min(20)).copied().collect();
            eprintln!(
                "  context_tree BuildHuffmanCodes[{}]: length={}, depths={:?}{}",
                i,
                length,
                depth_slice,
                if length > 20 { ", ..." } else { "" }
            );
        }

        prefix_codes.push(PrefixCode { depths, bits });
    }

    #[cfg(feature = "debug-tokens")]
    {
        eprintln!(
            "  context_tree_entropy: {} histograms -> {} prefix codes, context_map len={}",
            NUM_TREE_CONTEXTS,
            prefix_codes.len(),
            context_map.len()
        );
    }

    (context_map, prefix_codes)
}

/// Write the context tree for modular stream DC coding.
///
/// This writes the context tree tokens that tell the decoder how to
/// interpret DC coefficients in the modular stream.
pub fn write_context_tree(num_dc_groups: usize, writer: &mut BitWriter) -> Result<()> {
    // Copy tokens and modify token[1].value for num_dc_groups
    let mut tokens: Vec<Token> = CONTEXT_TREE_TOKENS
        .iter()
        .map(|&(ctx, val)| Token::new(ctx, val))
        .collect();

    // Token[1] value encodes the number of streams (1 + num_dc_groups)
    tokens[1].value = pack_signed(1 + num_dc_groups as i32) as u32;

    // Build entropy code for the tokens
    let (context_map, prefix_codes) = build_context_tree_entropy_code(&tokens);

    #[cfg(feature = "debug-tokens")]
    {
        eprintln!(
            "context_tree: {} contexts, {} prefix codes, context_map={:?}",
            context_map.len(),
            prefix_codes.len(),
            context_map
        );
    }

    // Write tree header
    writer.write(1, 1)?; // not an empty tree
    writer.write(1, 0)?; // no lz77

    // Write the entropy code (context map + prefix codes)
    let code = EntropyCode::new(&context_map, &prefix_codes);
    write_entropy_code(&code, writer)?;

    // Write all the tokens
    for token in &tokens {
        write_token(token, &code, writer)?;
    }

    Ok(())
}

/// Write the compact block context map.
///
/// This is written as a context map in the DC global section.
pub fn write_block_context_map(writer: &mut BitWriter) -> Result<()> {
    #[cfg(feature = "debug-tokens")]
    let start_bits = writer.bits_written();

    // Check if all values are 0 (simple case)
    let max_val = *COMPACT_BLOCK_CONTEXT_MAP.iter().max().unwrap_or(&0);
    if max_val == 0 {
        writer.write(3, 1)?; // simple code, 0 bits per entry
        return Ok(());
    }

    // Not simple: write 0, no MTF, no LZ77
    writer.write(3, 0)?;

    // Build tokens from context map
    let tokens: Vec<Token> = COMPACT_BLOCK_CONTEXT_MAP
        .iter()
        .map(|&v| Token::new(0, v as u32))
        .collect();

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
        eprintln!(
            "  write_block_context_map: {} entries, length={}, depths={:?}",
            COMPACT_BLOCK_CONTEXT_MAP.len(),
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
        eprintln!(
            "  write_block_context_map bits: header=3, prefix_code={}, tokens={}, total={}",
            prefix_bits, token_bits, total
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_tree_tokens_count() {
        assert_eq!(CONTEXT_TREE_TOKENS.len(), NUM_CONTEXT_TREE_TOKENS);
    }

    #[test]
    fn test_context_tree_tokens_contexts_in_range() {
        for (i, &(ctx, _)) in CONTEXT_TREE_TOKENS.iter().enumerate() {
            assert!(
                (ctx as usize) < NUM_TREE_CONTEXTS,
                "Token {} has context {} >= {}",
                i,
                ctx,
                NUM_TREE_CONTEXTS
            );
        }
    }

    #[test]
    fn test_compact_block_context_map_size() {
        assert_eq!(COMPACT_BLOCK_CONTEXT_MAP.len(), 39);
    }

    #[test]
    fn test_write_context_tree() {
        let mut writer = BitWriter::new();
        let result = write_context_tree(1, &mut writer);
        assert!(result.is_ok());
        // Should have written something
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_block_context_map() {
        let mut writer = BitWriter::new();
        let result = write_block_context_map(&mut writer);
        assert!(result.is_ok());
        // Should have written something
        assert!(writer.bits_written() > 0);
    }
}
