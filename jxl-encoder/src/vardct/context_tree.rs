// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Context tree tokens for modular DC header.
//!
//! Ported from libjxl-tiny enc_frame.cc

use super::ac_context::BlockCtxMap;
use super::cluster::{Histogram, cluster_histograms};
use super::common::{ceil_log2_nonzero, pack_signed};
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::entropy_coding::encode::{
    ALPHABET_SIZE, EntropyCode, PrefixCode, convert_bit_depths_to_symbols, create_huffman_tree,
    write_entropy_code, write_prefix_codes, write_token,
};
use crate::entropy_coding::move_to_front_transform;
use crate::entropy_coding::token::{Token, UintCoder};
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
    (1, 2),
    (0, 4),
    (1, 1),
    (0, 2),
    (1, 10),
    (0, 0),
    (1, 1),
    (0, 4),
    (1, 1),
    (0, 0),
    (1, 10),
    (0, 94),
    (1, 10),
    (0, 61),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 3),
    (0, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 10),
    (0, 382),
    (1, 10),
    (0, 22),
    (1, 10),
    (0, 13),
    (1, 10),
    (0, 253),
    (1, 8),
    (0, 10),
    (1, 8),
    (0, 10),
    (1, 10),
    (0, 784),
    (1, 10),
    (0, 190),
    (1, 10),
    (0, 46),
    (1, 10),
    (0, 10),
    (1, 10),
    (0, 5),
    (1, 10),
    (0, 29),
    (1, 10),
    (0, 125),
    (1, 10),
    (0, 509),
    (1, 8),
    (0, 22),
    (1, 8),
    (0, 6),
    (1, 8),
    (0, 22),
    (1, 8),
    (0, 6),
    (1, 10),
    (0, 1000),
    (1, 10),
    (0, 510),
    (1, 10),
    (0, 254),
    (1, 10),
    (0, 126),
    (1, 10),
    (0, 62),
    (1, 10),
    (0, 30),
    (1, 10),
    (0, 14),
    (1, 10),
    (0, 6),
    (1, 10),
    (0, 1),
    (1, 10),
    (0, 7),
    (1, 10),
    (0, 21),
    (1, 10),
    (0, 45),
    (1, 10),
    (0, 93),
    (1, 10),
    (0, 189),
    (1, 10),
    (0, 381),
    (1, 10),
    (0, 783),
    (1, 0),
    (2, 1),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 1),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 1),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 1),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 10),
    (0, 2),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 10),
    (0, 999),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
];

/// Compact block context map for AC coefficient coding.
/// From libjxl-tiny ac_context.h:45-48.
pub static COMPACT_BLOCK_CONTEXT_MAP: [u8; 39] = [
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Y
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // X
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // B
];

/// Number of tokens in the JPEG-transcode context tree.
///
/// Lever A (2026-05-28): mirror libjxl's JPEG-mode tree shape
/// (`kJpegTranscodeACMeta` + WP/gradient-fixed DC). The AC metadata subtree
/// shrinks from 11 leaves (4 Left + 5 Zero + 2 Gradient) to a single
/// `Leaf(Zero)` (libjxl `enc_encoding.cc:475-477`); the DC subtree is the
/// same 34-leaf gradient-fixed BSP we already use. Tokens drop 313 → 243
/// (-70) and the data-side context count drops 45 → 35.
#[cfg(feature = "jpeg-reencoding")]
pub const NUM_JPEG_TRANSCODE_CONTEXT_TREE_TOKENS: usize = 243;

/// Context tree tokens for JPEG-mode lossless transcoding.
///
/// BFS-ordered. Same structure as [`CONTEXT_TREE_TOKENS`] except the
/// AC-metadata subtree (11 leaves, 10 internal nodes) is replaced by a
/// single `Leaf(Predictor::Zero)`. Token[1] is the root splitval —
/// rewritten to `pack_signed(1 + num_dc_groups)` at write time so the
/// decoder routes `stream_id > 1+num_dc_groups` (AC metadata streams) to
/// the LEFT (single zero-context) leaf and `stream_id <= 1+num_dc_groups`
/// (DC streams) to the RIGHT (gradient-DC) subtree, matching
/// [`write_context_tree`].
#[cfg(feature = "jpeg-reencoding")]
pub static JPEG_TRANSCODE_CONTEXT_TREE_TOKENS: [(u32, u32);
    NUM_JPEG_TRANSCODE_CONTEXT_TREE_TOKENS] = [
    (1, 2),
    (0, 2),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 10),
    (0, 0),
    (1, 10),
    (0, 94),
    (1, 10),
    (0, 61),
    (1, 10),
    (0, 382),
    (1, 10),
    (0, 22),
    (1, 10),
    (0, 13),
    (1, 10),
    (0, 253),
    (1, 10),
    (0, 784),
    (1, 10),
    (0, 190),
    (1, 10),
    (0, 46),
    (1, 10),
    (0, 10),
    (1, 10),
    (0, 5),
    (1, 10),
    (0, 29),
    (1, 10),
    (0, 125),
    (1, 10),
    (0, 509),
    (1, 10),
    (0, 1000),
    (1, 10),
    (0, 510),
    (1, 10),
    (0, 254),
    (1, 10),
    (0, 126),
    (1, 10),
    (0, 62),
    (1, 10),
    (0, 30),
    (1, 10),
    (0, 14),
    (1, 10),
    (0, 6),
    (1, 10),
    (0, 1),
    (1, 10),
    (0, 7),
    (1, 10),
    (0, 21),
    (1, 10),
    (0, 45),
    (1, 10),
    (0, 93),
    (1, 10),
    (0, 189),
    (1, 10),
    (0, 381),
    (1, 10),
    (0, 783),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 10),
    (0, 2),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 10),
    (0, 999),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (1, 0),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
];

/// Build an optimized entropy code for the context tree tokens.
///
/// This builds histograms from the tokens, clusters them, then creates Huffman codes.
fn build_context_tree_entropy_code(tokens: &[Token]) -> (Vec<u8>, Vec<PrefixCode>) {
    // Build histograms for each context using the Histogram struct
    let mut histograms: Vec<Histogram> = (0..NUM_TREE_CONTEXTS).map(|_| Histogram::new()).collect();

    for token in tokens {
        let encoded = UintCoder::encode(token.value);
        let ctx = token.context() as usize;
        histograms[ctx].add(encoded.token as usize);
    }

    // Cluster similar histograms together
    // Note: cluster_histograms modifies histograms in place and returns context_map
    let context_map = cluster_histograms(&mut histograms);

    // Build Huffman codes for each clustered histogram
    let mut prefix_codes = Vec::with_capacity(histograms.len());
    #[allow(clippy::unused_enumerate_index)]
    for (_i, hist) in histograms.iter().enumerate() {
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
            debug_log!(
                "  context_tree BuildHuffmanCodes[{}]: length={}, depths={:?}{}",
                _i,
                length,
                depth_slice,
                if length > 20 { ", ..." } else { "" }
            );
        }

        prefix_codes.push(PrefixCode { depths, bits });
    }

    #[cfg(feature = "debug-tokens")]
    {
        debug_log!(
            "  context_tree_entropy: {} histograms -> {} prefix codes, context_map len={}",
            NUM_TREE_CONTEXTS,
            prefix_codes.len(),
            context_map.len()
        );
    }

    (context_map, prefix_codes)
}

/// Write the JPEG-transcode context tree for modular DC + AC-metadata streams.
///
/// Lever A (2026-05-28): same as [`write_context_tree`] but emits the
/// shrunk-AC-metadata tree from [`JPEG_TRANSCODE_CONTEXT_TREE_TOKENS`].
/// Mirrors libjxl `enc_modular.cc:1742` setting `tree_kind =
/// kJpegTranscodeACMeta` for `ACMetadata` streams when `jpeg_transcode=true`,
/// while keeping the DC streams' default tree shape.
///
/// The caller MUST emit AC-metadata data tokens with `context = 0` and
/// `predictor = Zero` (raw values, no gradient/left/etc residuals); see
/// `dc_coding::collect_ac_metadata_tokens_region_jpeg_transcode` for the
/// paired collector. DC data tokens use
/// `context = GRADIENT_CONTEXT_LUT[grad] - DC_CONTEXT_OFFSET_JPEG_TRANSCODE`
/// (= LUT value − 10), matching the BFS context IDs the new tree produces.
#[cfg(feature = "jpeg-reencoding")]
pub fn write_jpeg_transcode_context_tree(
    num_dc_groups: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    let mut tokens: Vec<Token> = JPEG_TRANSCODE_CONTEXT_TREE_TOKENS
        .iter()
        .map(|&(ctx, val)| Token::new(ctx, val))
        .collect();

    // Root splitval encodes (1 + num_dc_groups). Decoder routes:
    //   stream_id >  (1 + num_dc_groups) → LEFT subtree (AC-metadata, single Zero leaf)
    //   stream_id <= (1 + num_dc_groups) → RIGHT subtree (gradient-DC)
    // Matches the [`write_context_tree`] root layout.
    tokens[1].value = pack_signed(1 + num_dc_groups as i32);

    let (context_map, prefix_codes) = build_context_tree_entropy_code(&tokens);

    writer.write(1, 1)?; // not an empty tree
    writer.write(1, 0)?; // no lz77

    let code = EntropyCode::new(&context_map, &prefix_codes);
    write_entropy_code(&code, writer)?;

    for token in &tokens {
        write_token(token, &code, None, writer)?;
    }

    Ok(())
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
    tokens[1].value = pack_signed(1 + num_dc_groups as i32);

    // Build entropy code for the tokens
    let (context_map, prefix_codes) = build_context_tree_entropy_code(&tokens);

    #[cfg(feature = "debug-tokens")]
    {
        debug_log!(
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
        write_token(token, &code, None, writer)?;
    }

    Ok(())
}

/// Write a learned context tree for modular stream DC coding.
///
/// This is the learned-tree version of `write_context_tree`. Instead of using
/// the static `CONTEXT_TREE_TOKENS`, it uses tokens generated from a learned
/// DC tree via `tree_to_tokens`.
///
/// # Arguments
/// * `tree_tokens` - Tokens from `dc_tree_learn::tree_to_tokens()`
/// * `num_dc_groups` - Number of DC groups (for multi-group images)
/// * `writer` - Bitstream writer
pub fn write_learned_context_tree(
    tree_tokens: &[(u32, u32)],
    _num_dc_groups: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    // The learned tree already has the correct root split on property 1
    // (stream_id) with splitval=num_dc_groups, set by
    // tree_tokens_with_ac_metadata_prefix. This routes DC groups
    // (stream_ids 1..num_dc_groups) to the DC subtree and AC metadata
    // (stream_ids 1+2*num_dc_groups..) to the AC metadata subtree.
    // Works for any number of DC groups.

    // Convert tree tokens to Token objects
    let tokens: Vec<Token> = tree_tokens
        .iter()
        .map(|&(ctx, val)| Token::new(ctx, val))
        .collect();

    if tokens.is_empty() {
        // Empty tree - write a simple single-context tree
        let simple_tree = vec![
            Token::new(1, 0), // leaf marker
            Token::new(2, 5), // predictor = Gradient
            Token::new(3, 0), // offset = 0
            Token::new(4, 0), // mul_log = 0
            Token::new(5, 0), // mul_bits = 0
        ];
        return write_learned_context_tree_inner(&simple_tree, writer);
    }

    write_learned_context_tree_inner(&tokens, writer)
}

/// Inner function to write context tree tokens to bitstream.
fn write_learned_context_tree_inner(tokens: &[Token], writer: &mut BitWriter) -> Result<()> {
    // Build entropy code for the tokens
    let (context_map, prefix_codes) = build_context_tree_entropy_code(tokens);

    #[cfg(feature = "debug-tokens")]
    {
        debug_log!(
            "learned_context_tree: {} tokens, {} contexts, {} prefix codes",
            tokens.len(),
            context_map.len(),
            prefix_codes.len()
        );
    }

    // Write tree header
    writer.write(1, 1)?; // not an empty tree
    writer.write(1, 0)?; // no lz77

    // Write the entropy code (context map + prefix codes)
    let code = EntropyCode::new(&context_map, &prefix_codes);
    write_entropy_code(&code, writer)?;

    // Write all the tokens
    for token in tokens {
        write_token(token, &code, None, writer)?;
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
        debug_log!(
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
        debug_log!(
            "  write_block_context_map bits: header=3, prefix_code={}, tokens={}, total={}",
            prefix_bits,
            token_bits,
            total
        );
    }

    Ok(())
}

/// Write a U32-coded QF threshold value.
///
/// Matches libjxl's kQFThresholdDist: Bits(2), BitsOffset(3,4), BitsOffset(5,12), BitsOffset(8,44).
/// The decoder reads the value and adds 1, so we write `qf_threshold - 1`.
fn write_qf_threshold(value: u32, writer: &mut BitWriter) -> Result<()> {
    // Decoder does: read + 1, so encode value - 1
    let v = value - 1;
    if v < 4 {
        // Selector 0: Bits(2), value 0-3
        writer.write(2, 0)?; // selector
        writer.write(2, v as u64)?;
    } else if v < 12 {
        // Selector 1: BitsOffset(3, 4), value 4-11
        writer.write(2, 1)?;
        writer.write(3, (v - 4) as u64)?;
    } else if v < 44 {
        // Selector 2: BitsOffset(5, 12), value 12-43
        writer.write(2, 2)?;
        writer.write(5, (v - 12) as u64)?;
    } else {
        // Selector 3: BitsOffset(8, 44), value 44-299
        writer.write(2, 3)?;
        writer.write(8, (v - 44) as u64)?;
    }
    Ok(())
}

/// Write a U32-coded signed DC threshold value.
///
/// Matches libjxl's `kDCThresholdDist` (entropy_coder.h:37-39):
/// `U32Enc(Bits(4), BitsOffset(8, 16), BitsOffset(16, 272),
///         BitsOffset(32, 65808))`.
///
/// The decoder applies `UnpackSigned` to recover the signed value, so the
/// encoder PackSigned-encodes first, then U32-selects. Mirrors libjxl
/// `enc_context_map.cc:160-163` `U32Coder::Write(kDCThresholdDist,
/// PackSigned(i), writer)`.
fn write_dc_threshold(value: i32, writer: &mut BitWriter) -> Result<()> {
    let packed = pack_signed(value);
    if packed < 16 {
        // Selector 0: Bits(4)
        writer.write(2, 0)?;
        writer.write(4, packed as u64)?;
    } else if packed < 272 {
        // Selector 1: BitsOffset(8, 16) — value-16 in 8 bits
        writer.write(2, 1)?;
        writer.write(8, (packed - 16) as u64)?;
    } else if packed < 65808 {
        // Selector 2: BitsOffset(16, 272)
        writer.write(2, 2)?;
        writer.write(16, (packed - 272) as u64)?;
    } else {
        // Selector 3: BitsOffset(32, 65808). DC thresholds in the JPEG
        // path are in [-1025, 1022], so packed in [0, 2049], well within
        // selector 1. This branch is only reachable on user-constructed
        // BlockCtxMap with extreme values; gracefully emit 32 raw bits.
        writer.write(2, 3)?;
        writer.write(32, (packed - 65808) as u64)?;
    }
    Ok(())
}

/// Write an adaptive block context map (non-default).
///
/// This writes the full BlockCtxMap header (libjxl `enc_context_map.cc:141-172`):
/// 1. Non-default flag (0)
/// 2. Per-channel DC threshold count (4 bits) + threshold values
///    (kDCThresholdDist U32-coded, signed via PackSigned)
/// 3. QF threshold count (4 bits) + values (kQFThresholdDist U32-coded)
/// 4. Entropy-coded context map
///
/// Issue #65: previously hardcoded zero DC threshold counts, blocking the
/// JPEG DC-quantile context model. Now serialises whatever counts the
/// caller's `BlockCtxMap` carries.
pub fn write_block_ctx_map_adaptive(ctx_map: &BlockCtxMap, writer: &mut BitWriter) -> Result<()> {
    write_block_ctx_map_adaptive_with_mode(ctx_map, false, writer)
}

/// Like [`write_block_ctx_map_adaptive`] but `jpeg_mode` enables the EX-J28
/// ANS+LZ77 candidate in the context-map encoding selector. The VarDCT path
/// passes `false` to remain byte-identical; the JPEG transcode path passes
/// `true` (its block_ctx_map is large/repetitive enough that ANS+LZ77 can
/// beat Huffman, and the selector only picks it when strictly smaller).
pub fn write_block_ctx_map_adaptive_with_mode(
    ctx_map: &BlockCtxMap,
    jpeg_mode: bool,
    writer: &mut BitWriter,
) -> Result<()> {
    #[cfg(feature = "debug-tokens")]
    let start_bits = writer.bits_written();

    // Non-default BlockCtxMap
    writer.write(1, 0)?;

    // DC thresholds: 3 channels, each with count + values
    for j in 0..3 {
        let dct = &ctx_map.dc_thresholds[j];
        debug_assert!(
            dct.len() < 16,
            "dc_thresholds[{}].len()={} >= 16",
            j,
            dct.len()
        );
        writer.write(4, dct.len() as u64)?;
        for &v in dct {
            write_dc_threshold(v, writer)?;
        }
    }

    // QF thresholds
    writer.write(4, ctx_map.qf_thresholds.len() as u64)?;
    for &t in &ctx_map.qf_thresholds {
        write_qf_threshold(t, writer)?;
    }

    #[cfg(feature = "debug-tokens")]
    {
        debug_log!(
            "  write_block_ctx_map_adaptive: {} qf_thresholds={:?}, {} ctxs, map_len={}",
            ctx_map.qf_thresholds.len(),
            ctx_map.qf_thresholds,
            ctx_map.num_ctxs,
            ctx_map.ctx_map.len()
        );
    }

    // Write context map using existing entropy-coded format
    write_context_map_from_slice(&ctx_map.ctx_map, jpeg_mode, writer)?;

    #[cfg(feature = "debug-tokens")]
    {
        let total = writer.bits_written() - start_bits;
        debug_log!("  write_block_ctx_map_adaptive total: {} bits", total);
    }

    Ok(())
}

/// Build a Huffman prefix code from a slice of context-map bytes.
///
/// Returns `(prefix_code, total_data_bits)` where `total_data_bits` is the
/// exact number of bits the token stream will consume given this code
/// (sum of `count[i] * depth[i]` plus extra UintCoder bits per token).
fn build_ctxmap_prefix_code(bytes: &[u8]) -> (PrefixCode, usize) {
    let mut histogram = [0u32; ALPHABET_SIZE];
    let mut extra_bits_total: usize = 0;
    for &v in bytes {
        let encoded = UintCoder::encode(v as u32);
        histogram[encoded.token as usize] += 1;
        extra_bits_total += encoded.nbits as usize;
    }

    let mut depths = [0u8; ALPHABET_SIZE];
    let mut length = ALPHABET_SIZE;
    while length > 0 && histogram[length - 1] == 0 {
        length -= 1;
    }
    create_huffman_tree(&histogram, length.max(1), 15, &mut depths);

    let mut bits = [0u16; ALPHABET_SIZE];
    convert_bit_depths_to_symbols(&depths, &mut bits);

    let code = PrefixCode { depths, bits };

    // Sum count[i] * depth[i] using histogram + depth.
    let mut data_bits: usize = extra_bits_total;
    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            data_bits += (count as usize) * (depths[i] as usize);
        }
    }

    (code, data_bits)
}

/// Trial-encode a Huffman-coded context-map candidate and return its cost
/// in bits (header + data), exclusive of the leading 3-bit selector.
///
/// This matches libjxl `enc_context_map.cc::EncodeContextMap`'s
/// `BuildAndEncodeHistograms` cost measurement pass: we actually write the
/// prefix-code header to a scratch BitWriter and count the bits, then add
/// the data-bit total. Using the real writer (not a Shannon proxy) is
/// necessary because the simple/no-MTF/MTF decisions can hinge on small
/// differences in the Huffman tree serialisation overhead.
fn trial_ctxmap_huffman_cost(bytes: &[u8]) -> Result<(PrefixCode, usize)> {
    let (code, data_bits) = build_ctxmap_prefix_code(bytes);
    let mut scratch = BitWriter::with_capacity(64);
    write_prefix_codes(&[code], &mut scratch)?;
    let header_bits = scratch.bits_written();
    Ok((code, header_bits + data_bits))
}

/// Emit Huffman-coded token stream for a context-map candidate.
///
/// Writes the prefix-code header followed by one token per byte, using the
/// `(symbol_bits | extra_bits)` packing the decoder reads back via the
/// inverse `UintCoder`.
fn write_ctxmap_huffman_payload(
    bytes: &[u8],
    code: &PrefixCode,
    writer: &mut BitWriter,
) -> Result<()> {
    write_prefix_codes(&[*code], writer)?;
    for &v in bytes {
        let encoded = UintCoder::encode(v as u32);
        let tok = encoded.token as usize;
        let depth = code.depths[tok] as usize;
        let huff_bits = code.bits[tok] as u64;

        let data = huff_bits | ((encoded.bits as u64) << depth);
        let total_bits = depth + encoded.nbits as usize;

        writer.write(total_bits, data)?;
    }
    Ok(())
}

/// Write an entropy-coded context map from a byte slice.
///
/// Ports the 3-way cost comparison from libjxl
/// `enc_context_map.cc::EncodeContextMap` (commit-hashed reference in
/// `~/work/jxl-efforts/libjxl/lib/jxl/enc_context_map.cc`):
///
/// 1. **Simple mode** — only legal when `entry_bits < 4` (≤ 8 histograms).
///    Cost = `entry_bits * map.len()` bits.
/// 2. **Huffman no-MTF** — raw tokens, single Huffman tree.
///    Cost = `header_bits + sum(count * depth)`.
/// 3. **Huffman with MTF** — same shape, but tokens are first
///    [`move_to_front_transform`]ed. MTF clusters repeated symbols at low
///    indices, which usually shrinks the Huffman tree AND the data stream
///    when the input has runs of identical values.
///
/// The 3-bit selector (`simple_flag : use_mtf : lz77`) is the same length
/// for every candidate, so we compare post-selector costs and pick the
/// minimum. Picking MTF when its data savings cover the extra `use_mtf`
/// bit is automatic (the comparison is on total bits, not data alone).
///
/// The all-zeros fast path is preserved bit-exact for callers (legacy
/// behaviour: `writer.write(3, 1)`).
///
/// **Wire-format note**: only the Huffman path writes a `use_mtf` bit;
/// simple mode encodes the entry width directly. The 3-bit selector for
/// simple is `(1, entry_bits[2])`, for Huffman it's `(0, use_mtf,
/// lz77=0)`. Both are 3 bits, so the leading writer offset is the same.
///
/// **lossless / lossy hash-lock invariance**: this writer is only called
/// via `write_block_ctx_map_adaptive`. The compact default lossy path
/// uses [`write_block_context_map`] directly with a hardcoded Huffman
/// no-MTF emission. As a result the 3-way comparison does NOT affect the
/// 40-cell hash-lock corpus (synthetic 32×32/48×48 fixtures, default
/// 4-context map).
fn write_context_map_from_slice(map: &[u8], jpeg_mode: bool, writer: &mut BitWriter) -> Result<()> {
    // Check if all values are 0 (simple case, 0 bits per entry).
    let max_val = *map.iter().max().unwrap_or(&0);
    if max_val == 0 {
        writer.write(3, 1)?; // simple_flag=1, entry_bits=0
        return Ok(());
    }

    let n = map.len();

    // Candidate A: simple mode (libjxl gate: entry_bits < 4).
    // libjxl uses `CeilLog2Nonzero(num_histograms)` — `num_histograms` is
    // typically `max_val + 1` since contexts are dense from 0..num_histograms-1.
    let entry_bits = ceil_log2_nonzero(max_val as usize + 1) as usize;
    let simple_bits: Option<usize> = if entry_bits < 4 {
        Some(entry_bits * n)
    } else {
        None
    };

    // Candidate B: Huffman without MTF.
    let (code_raw, cost_raw) = trial_ctxmap_huffman_cost(map)?;

    // Candidate C: Huffman with MTF.
    let mtf_bytes = move_to_front_transform(map);
    let (code_mtf, cost_mtf) = trial_ctxmap_huffman_cost(&mtf_bytes)?;

    // Pick the cheapest candidate. Ties prefer simple > no-MTF > MTF (lowest
    // implementation/decode complexity first); the lever cares about strict
    // less-than wins anyway.
    enum Pick {
        Simple,
        Huffman,
        HuffmanMtf,
        AnsLz77,
    }

    // EX-J28 (2026-05-28): add ANS+LZ77 candidate to the block_ctx_map
    // encoding selector. The existing `write_context_map_from_slice`
    // only compared Simple / Huffman / Huffman+MTF, while libjxl's
    // `EncodeContextMap` (enc_context_map.cc:65-139) uses
    // `BuildAndEncodeHistograms` which includes the ANS+LZ77 path with
    // `HybridUintMethod::kContextMap`. On repetitive maps (especially
    // the JPEG block_ctx_map with ~312 entries × small alphabet)
    // ANS+LZ77 can beat Huffman. Env hook
    // `JXL_NO_ANS_LZ77_CTXMAP=1` disables the candidate.
    let ans_lz77_scratch = if !jpeg_mode || std::env::var_os("JXL_NO_ANS_LZ77_CTXMAP").is_some() {
        // EX-J28 scoped to JPEG transcode only — the VarDCT path
        // (jpeg_mode=false) keeps the Simple/Huffman/MTF selector to stay
        // byte-identical (preserves hash-locks + strategy byte-locks).
        None
    } else {
        crate::entropy_coding::encode_ans::build_context_map_nonsimple_ans_lz77(map).ok()
    };
    let cost_ans_lz77 = ans_lz77_scratch.as_ref().map(|b| b.bits_written());

    let mut pick = Pick::Huffman;
    let mut best_cost = cost_raw;

    if cost_mtf < best_cost {
        pick = Pick::HuffmanMtf;
        best_cost = cost_mtf;
    }
    if let Some(cost) = cost_ans_lz77
        && cost < best_cost
    {
        pick = Pick::AnsLz77;
        best_cost = cost;
    }
    if let Some(sb) = simple_bits
        && sb < best_cost
    {
        pick = Pick::Simple;
        // best_cost unused after this branch but kept for clarity.
        let _ = best_cost;
    }

    #[cfg(feature = "debug-tokens")]
    {
        debug_log!(
            "  write_context_map_from_slice: n={} max_val={} simple_bits={:?} cost_raw={} cost_mtf={} picked={}",
            n,
            max_val,
            simple_bits,
            cost_raw,
            cost_mtf,
            match pick {
                Pick::Simple => "simple",
                Pick::Huffman => "huffman_no_mtf",
                Pick::HuffmanMtf => "huffman_mtf",
                Pick::AnsLz77 => "ans_lz77",
            }
        );
    }

    match pick {
        Pick::Simple => {
            // Selector: simple_flag=1, entry_bits in 2 bits.
            writer.write(1, 1)?;
            writer.write(2, entry_bits as u64)?;
            // Entry bits assumed to fit because entry_bits comes from
            // ceil_log2(max+1); every entry is in [0, max].
            for &v in map {
                writer.write(entry_bits, v as u64)?;
            }
        }
        Pick::Huffman => {
            // Selector: simple_flag=0, use_mtf=0, lz77=0 -> 3-bit value 0.
            writer.write(3, 0)?;
            write_ctxmap_huffman_payload(map, &code_raw, writer)?;
        }
        Pick::HuffmanMtf => {
            // Selector: simple_flag=0, use_mtf=1, lz77=0 -> 3-bit value 2
            // (LSB-first: bit 0 = simple_flag = 0, bit 1 = use_mtf = 1,
            // bit 2 = lz77 = 0 → integer 0b010 = 2).
            writer.write(3, 0b010)?;
            write_ctxmap_huffman_payload(&mtf_bytes, &code_mtf, writer)?;
        }
        Pick::AnsLz77 => {
            // ANS+LZ77 path emits its own selector + headers inside
            // build_context_map_nonsimple_ans_lz77. Copy bit-exact.
            let buf = ans_lz77_scratch.expect("Pick::AnsLz77 requires ans_lz77_scratch to be Some");
            let bits_to_copy = buf.bits_written();
            let bytes = buf.finish_with_padding();
            crate::entropy_coding::encode_ans::copy_bits(&bytes, bits_to_copy, writer)?;
        }
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

    #[test]
    fn test_write_block_ctx_map_adaptive_default() {
        // Writing the default map as adaptive should succeed
        let map = BlockCtxMap::default();
        let mut writer = BitWriter::new();
        let result = write_block_ctx_map_adaptive(&map, &mut writer);
        assert!(result.is_ok());
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_block_ctx_map_adaptive_with_qf() {
        // Write a map with 1 QF threshold
        use super::super::coeff_order::NUM_ORDER_BUCKETS;
        let num_qf_segs = 2;
        let section_size = NUM_ORDER_BUCKETS * num_qf_segs;
        let mut ctx_map = vec![0u8; section_size * 3];
        // Set some non-zero contexts
        for (i, val) in ctx_map[..section_size].iter_mut().enumerate() {
            *val = (i % 3) as u8;
        }
        for (i, val) in ctx_map[section_size..].iter_mut().enumerate() {
            *val = 3 + ((section_size + i) % 2) as u8;
        }
        let map = BlockCtxMap {
            dc_thresholds: [vec![], vec![], vec![]],
            qf_thresholds: vec![10],
            ctx_map,
            num_dc_ctxs: 1,
            num_ctxs: 5,
        };
        let mut writer = BitWriter::new();
        let result = write_block_ctx_map_adaptive(&map, &mut writer);
        assert!(result.is_ok());
        assert!(writer.bits_written() > 0);
    }

    /// Issue #65: a JPEG-shaped BlockCtxMap with luma DC thresholds
    /// serialises cleanly via `write_block_ctx_map_adaptive`. The
    /// resulting bitstream is round-tripped by the decoder's
    /// `DecodeBlockCtxMap` (verified separately via roundtrip tests on
    /// real JPEGs).
    #[test]
    fn test_write_block_ctx_map_adaptive_with_dc_thresholds() {
        let map = BlockCtxMap::jpeg_dc_quantile(
            &{
                let mut h = [0usize; 2048];
                for &(i, n) in &[(1020usize, 50usize), (1024, 600), (1028, 300), (1032, 50)] {
                    h[i] = n;
                }
                h
            },
            1000,
            4,
            false,
        );
        assert!(!map.dc_thresholds[1].is_empty());
        let mut writer = BitWriter::new();
        let result = write_block_ctx_map_adaptive(&map, &mut writer);
        assert!(result.is_ok());
        assert!(writer.bits_written() > 0);
    }

    /// Issue #65: write_dc_threshold round-trip — every selector branch
    /// produces a valid bit pattern (selector + payload widths match the
    /// libjxl decoder's `U32Coder::Read(kDCThresholdDist, br)` —
    /// `Bits(4), BitsOffset(8, 16), BitsOffset(16, 272),
    /// BitsOffset(32, 65808)`).
    #[test]
    fn test_write_dc_threshold_selector_widths() {
        for v in [-1025i32, -1024, -1, 0, 1, 7, -7, 1022, 200, -200] {
            let mut writer = BitWriter::new();
            // Should not panic / error
            let r = write_dc_threshold(v, &mut writer);
            assert!(r.is_ok(), "v={} failed", v);
            // Each value: 2 bits selector + payload bits.
            // PackSigned(v) is in [0, 2049] for v in [-1025, 1022].
            // Selector 0 (Bits(4)): packed < 16 → 6 bits total
            // Selector 1 (BitsOffset(8,16)): 16..272 → 10 bits total
            // Selector 2 (BitsOffset(16,272)): 272..65808 → 18 bits total
            let bits = writer.bits_written();
            assert!((6..=18).contains(&bits), "v={} bits={}", v, bits);
        }
    }

    /// Lever #1: the all-zeros input must still write the legacy 3-bit
    /// header `(3, 1)` byte-identical (simple_flag=1, entry_bits=0). This
    /// is the fast path that lets callers with a literal-zero block
    /// context map skip MTF entirely.
    #[test]
    fn test_write_context_map_from_slice_all_zeros_byte_identical() {
        let map = vec![0u8; 39];
        let mut writer = BitWriter::new();
        write_context_map_from_slice(&map, false, &mut writer).unwrap();
        let bytes = writer.finish_with_padding();
        // 3 bits written + zero padding to byte = 1 byte. Value:
        // LSB-first bits 1, 0, 0 → 0b00000001 = 0x01.
        assert_eq!(
            bytes,
            vec![0x01],
            "all-zeros must produce 1-byte simple header"
        );
    }

    /// Lever #1 regression gate: the 3-way comparison must pick MTF on a
    /// classic MTF-favourable input — alternating long runs of multiple
    /// distinct values (the canonical MTF win pattern, mirrors the JPEG
    /// `BlockCtxMap::libjxl_default` 15-cluster map structure where each
    /// channel/order region carries one cluster id at a time).
    #[test]
    fn test_write_context_map_from_slice_mtf_wins_on_run_length_input() {
        // Construct an input with multiple distinct runs:
        //   AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC DDDDDDDDDD AAAAAAAAAA ...
        // Raw Huffman: 4 equally-likely tokens, ~2 bits/token.
        // MTF: each run starts with a small index (0 after the first
        // occurrence) then immediately becomes 0 — heavily skewed to 0,
        // ~0.5 bits/token after MTF + small alphabet → smaller tree.
        let mut map = Vec::new();
        for cycle in 0..8 {
            for v in 0u8..=4 {
                let run_len = if cycle == 0 && v <= 2 { 1 } else { 15 };
                map.extend(std::iter::repeat_n(v, run_len));
            }
        }

        let mut writer_a = BitWriter::new();
        let mut writer_b = BitWriter::new();
        write_context_map_from_slice(&map, false, &mut writer_a).unwrap();
        let bytes = writer_a.finish_with_padding();
        assert!(!bytes.is_empty());

        // For a sanity check, we also confirm the 3-way comparison does
        // strictly prefer MTF over no-MTF for this input by trial-encoding
        // both and comparing costs.
        let (_, cost_raw) = trial_ctxmap_huffman_cost(&map).unwrap();
        let mtf_bytes = move_to_front_transform(&map);
        let (_, cost_mtf) = trial_ctxmap_huffman_cost(&mtf_bytes).unwrap();
        assert!(
            cost_mtf < cost_raw,
            "MTF should be cheaper on run-length input: raw={} mtf={}",
            cost_raw,
            cost_mtf
        );

        // Encode again to verify writer is deterministic.
        write_context_map_from_slice(&map, false, &mut writer_b).unwrap();
        let bytes_b = writer_b.finish_with_padding();
        assert_eq!(bytes, bytes_b, "writer must be deterministic");
    }

    /// Lever #1 regression gate: a SHORT input with HIGH symbol entropy
    /// makes simple mode dominate (Huffman tree overhead > data savings).
    /// Input: 4 random-ish bytes ≤ entry_bits=2 → simple=8 bits, Huffman
    /// header alone is >30 bits.
    #[test]
    fn test_write_context_map_from_slice_picks_simple_when_short() {
        let map: Vec<u8> = vec![0, 1, 2, 3];
        let mut writer = BitWriter::new();
        write_context_map_from_slice(&map, false, &mut writer).unwrap();
        let bytes = writer.finish_with_padding();
        assert!(!bytes.is_empty());
        // Simple-flag bit must be 1 (simple mode picked, since Huffman
        // tree header for 4 symbols dwarfs the 8-bit simple payload).
        assert_eq!(
            bytes[0] & 1,
            1,
            "simple_flag bit should be 1 for short fixture (Huffman header too costly)"
        );
    }

    /// Lever #1 regression gate: confirm cost comparison correctness on the
    /// classic libjxl JPEG-class input — 7425-entry 15-cluster ctx_map with
    /// runs of same cluster id. Validates the lever's core claim: MTF wins
    /// on the actual production input shape.
    #[test]
    fn test_write_context_map_from_slice_mtf_wins_on_libjxl_15cluster_shape() {
        // Approximate the shape of libjxl's 15-cluster default ctx_map:
        // ~7425 entries with strong per-segment clustering (each 39-byte
        // segment carries 4-6 distinct cluster ids in runs).
        let mut map = Vec::with_capacity(7425);
        let palette = [0u8, 1, 5, 7, 11, 14, 3, 9, 2, 6, 8, 13, 4, 10, 12];
        for seg in 0..(7425 / 39 + 1) {
            for k in 0..39 {
                if map.len() >= 7425 {
                    break;
                }
                // Each row of 13 entries within a segment uses ~3 distinct
                // values in runs of 3-4. The seed per row cycles slowly.
                let row = k / 13;
                let col = k % 13;
                let pick = (seg + row * 5 + col / 4) % palette.len();
                map.push(palette[pick]);
            }
        }
        assert_eq!(map.len(), 7425);

        let (_, cost_raw) = trial_ctxmap_huffman_cost(&map).unwrap();
        let mtf_bytes = move_to_front_transform(&map);
        let (_, cost_mtf) = trial_ctxmap_huffman_cost(&mtf_bytes).unwrap();
        assert!(
            cost_mtf < cost_raw,
            "MTF must beat raw Huffman on 15-cluster shape: raw={} mtf={}",
            cost_raw,
            cost_mtf
        );

        let mut writer = BitWriter::new();
        write_context_map_from_slice(&map, false, &mut writer).unwrap();
        let bytes = writer.finish_with_padding();
        // First 3 bits must encode (simple=0, use_mtf=1, lz77=0) = 0b010 = 2.
        // LSB-first byte 0 has bits 0,1,2 = 0,1,0 → low nibble 0b???0010.
        assert_eq!(
            bytes[0] & 0b111,
            0b010,
            "selector should be (simple=0, use_mtf=1, lz77=0)"
        );
    }
}
