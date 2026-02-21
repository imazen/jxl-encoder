// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Tree histogram writing and tree serialization for modular encoding.
//!
//! Contains tree histogram construction (for zero, gradient, and weighted predictors),
//! tree token writing, weighted predictor header, and general MA tree serialization.

#![allow(dead_code)]

use super::encode_primitives::{pack_signed, write_integer_config, write_varlen_u16};
use crate::bit_writer::BitWriter;
use crate::entropy_coding::huffman_tree::{
    build_and_store_huffman_tree, convert_bit_depths_to_symbols, create_huffman_tree,
};
use crate::error::Result;

/// Write a complete single-leaf Zero predictor tree (histogram + tokens).
/// All tokens are 0, so alphabet size = 1 and no Huffman coding is needed.
pub(super) fn write_zero_tree_complete(writer: &mut BitWriter) -> Result<()> {
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: Starting tree histogram (Zero)",
        writer.bits_written()
    );

    // lz77.enabled = 0
    writer.write(1, 0)?;

    // Context map for 6 contexts: is_simple=1, bits_per_entry=0
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0

    // use_prefix_code = 1
    writer.write(1, 1)?;

    // IntegerConfig: When use_prefix_code=1, the decoder uses log_alphabet_size=15
    // for parsing IntegerConfig, regardless of actual alphabet size.
    // For raw symbol encoding with only symbol 0, use split_exponent = 15.
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;

    // alphabet_size - 1 = 0 using VarLenUint16 encoding
    write_varlen_u16(writer, 0)?;

    // No Huffman table needed for alphabet_size = 1
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: After tree histogram (Zero)",
        writer.bits_written()
    );

    Ok(())
}

/// Write a tree histogram that can encode tokens 0-5 (for Gradient predictor).
/// Returns (depths, codes) for use in encoding tree tokens.
pub(crate) fn write_tree_histogram_for_gradient(
    writer: &mut BitWriter,
) -> Result<(Vec<u8>, Vec<u16>)> {
    // NOTE: This is for LfGlobal trees which use allow_lz77=true in the decoder
    // Tree tokens are raw symbols (0-5), not hybrid uints.
    // Use split_exponent = log_alphabet_size for raw symbol encoding.
    write_tree_histogram_for_predictor_impl(writer, true, 5)
}

/// Write a tree histogram for a single-leaf tree with Zero predictor.
/// Returns (depths, codes) for use in encoding tree tokens.
pub(crate) fn write_tree_histogram_for_zero(writer: &mut BitWriter) -> Result<(Vec<u8>, Vec<u16>)> {
    write_tree_histogram_for_predictor_impl(writer, true, 0)
}

/// Write tree histogram and return (depths, codes) for encoding tree tokens.
fn write_tree_histogram_for_predictor_impl(
    writer: &mut BitWriter,
    write_lz77: bool,
    predictor_id: u32,
) -> Result<(Vec<u8>, Vec<u16>)> {
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: Starting tree histogram (lz77={})",
        writer.bits_written(),
        write_lz77
    );

    // For LfGlobal trees, allow_lz77=true so we write lz77.enabled
    // For modular substream trees (VarDCT), allow_lz77=false so we skip lz77.enabled
    if write_lz77 {
        writer.write(1, 0)?; // lz77.enabled = 0
        crate::trace::debug_eprintln!(
            "  TREE_HIST [bit {}]: lz77.enabled = 0",
            writer.bits_written()
        );
    }

    // Context map for 6 contexts: is_simple=1, bits_per_entry=0
    // (all contexts share same histogram)
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: context_map (is_simple=1, bits=0)",
        writer.bits_written()
    );

    // use_prefix_code = 1
    writer.write(1, 1)?;
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );

    // Build full Huffman table
    // Tree tokens for a LEAF node: property(ctx0), predictor(ctx1), offset(ctx2), mul_log(ctx3), mul_bits(ctx4)
    //
    // NOTE: For a leaf node, property should be < 0 after unpack_signed().
    // pack_signed(-1) = 1, so we should encode property=1.
    // HOWEVER, jxl-oxide seems to interpret property=0 specially, treating it as a leaf.
    // Using property=0 works with current decoders; property=1 causes decode failures.
    //
    // For TREE_PREDICTOR=5 (Gradient): tokens [0, 5, 0, 0, 0]
    //   - property = 0 (works with jxl-oxide as leaf marker)
    //   - predictor = 5 (Gradient)
    //   - offset = 0
    //   - mul_log = 0
    //   - mul_bits = 0
    // Histogram: symbol 0 appears 4 times, symbol 5 appears 1 time
    let max_symbol = if predictor_id == 0 {
        0u16
    } else {
        predictor_id as u16
    };
    let tree_histogram: &[u32] = if predictor_id == 0 {
        // For Zero predictor: tokens [0, 0, 0, 0, 0]
        // symbol 0 appears 5 times
        &[5u32]
    } else {
        // For Gradient predictor: tokens [0, 5, 0, 0, 0]
        // symbol 0 appears 4 times, symbol 5 appears 1 time
        &[4u32, 0, 0, 0, 0, 1]
    };

    // IntegerConfig: When use_prefix_code=1, the decoder uses log_alphabet_size=15
    // for parsing IntegerConfig, regardless of actual alphabet size.
    // For raw symbols (no hybrid uint), use split_exponent = 15 (max for prefix codes).
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: IntegerConfig (log_alpha={}, split_exp={}, raw symbols)",
        writer.bits_written(),
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX
    );

    // alphabet_size - 1 using VarLenUint16 encoding (matches libjxl)
    // For prefix codes, this is written AFTER IntegerConfigs, BEFORE Huffman tables
    let _alphabet_size = (max_symbol + 1) as u32;
    write_varlen_u16(writer, max_symbol)?;
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: alphabet_size-1 = {} (alphabet_size={})",
        writer.bits_written(),
        max_symbol,
        _alphabet_size
    );

    // Huffman table: skip if alphabet_size == 1 (only one possible symbol)
    // IMPORTANT: Return the codes from build_and_store_huffman_tree to ensure
    // tree tokens are encoded with the same codes that were written to the bitstream.
    let al_size = (max_symbol + 1) as usize;
    let (depths, codes) = if al_size > 1 {
        let table = build_and_store_huffman_tree(tree_histogram, writer)?;
        crate::trace::debug_eprintln!(
            "  TREE_HIST [bit {}]: After Huffman table",
            writer.bits_written()
        );
        (table.depths, table.codes)
    } else {
        crate::trace::debug_eprintln!(
            "  TREE_HIST [bit {}]: No Huffman table (al_size=1)",
            writer.bits_written()
        );
        (vec![0u8; al_size], vec![0u16; al_size])
    };

    Ok((depths, codes))
}

/// Write tree tokens for a single leaf with Zero predictor.
/// Uses the provided (depths, codes) from write_tree_histogram_for_zero.
pub(crate) fn write_zero_tree_tokens(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
) -> Result<()> {
    write_single_leaf_tree_tokens(writer, depths, codes, 0)
}

/// Write tree tokens for a single leaf with Gradient predictor.
/// Uses the provided (depths, codes) from write_tree_histogram_for_gradient.
pub(crate) fn write_gradient_tree_tokens(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
) -> Result<()> {
    write_single_leaf_tree_tokens(writer, depths, codes, 5)
}

/// Write tree tokens for a single leaf with the given predictor ID.
fn write_single_leaf_tree_tokens(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
    predictor_id: u32,
) -> Result<()> {
    crate::trace::debug_eprintln!(
        "  TREE_TOKENS [bit {}]: Starting tree tokens",
        writer.bits_written()
    );

    // Tree tokens for a single LEAF node:
    // - property = 0 (works with jxl-oxide as leaf marker, though spec says should be pack_signed(-1)=1)
    // - predictor (context 1)
    // - offset = 0 (context 2)
    // - mul_log = 0 (context 3)
    // - mul_bits = 0 (context 4) → multiplier = (0+1) << 0 = 1

    let tree_predictor = predictor_id;

    crate::trace::debug_eprintln!("  TREE_TOKENS: depths = {:?}", depths);
    crate::trace::debug_eprintln!("  TREE_TOKENS: codes = {:?}", codes);

    // Encode: property=0, predictor, offset=0, mul_log=0, mul_bits=0
    let tokens = [0u32, tree_predictor, 0, 0, 0];
    let _token_names = ["property", "predictor", "offset", "mul_log", "mul_bits"];

    #[allow(clippy::unused_enumerate_index)]
    for (_i, &token) in tokens.iter().enumerate() {
        let depth = depths.get(token as usize).copied().unwrap_or(0);
        let code = codes.get(token as usize).copied().unwrap_or(0);
        crate::trace::debug_eprintln!(
            "  TREE_TOKENS [bit {}]: {} = {} (depth={}, code={:0width$b})",
            writer.bits_written(),
            _token_names[_i],
            token,
            depth,
            code,
            width = depth.max(1) as usize
        );
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
    }

    crate::trace::debug_eprintln!("  TREE_TOKENS [bit {}]: Done", writer.bits_written());
    Ok(())
}

// Set to true to use Zero predictor for debugging
// Try: false = gradient tree path, true = zero tree path (works)
const USE_ZERO_PREDICTOR: bool = false;

/// Write a tree histogram that can encode tokens 0-6 (for Weighted predictor).
pub(super) fn write_tree_histogram_for_weighted(writer: &mut BitWriter) -> Result<()> {
    // lz77.enabled = 0
    writer.write(1, 0)?;

    // Context map for 6 contexts: is_simple=1, bits_per_entry=0
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0

    // use_prefix_code = 1
    writer.write(1, 1)?;

    // Tree tokens for predictor=6: [0,6,0,0,0] → histogram [4,0,0,0,0,0,1]
    const TREE_PREDICTOR: u32 = 6;
    let max_symbol = TREE_PREDICTOR as u16;
    let tree_histogram: &[u32] = &[4u32, 0, 0, 0, 0, 0, 1]; // 5 tokens: 4x symbol 0, 1x symbol 6

    // IntegerConfig: When use_prefix_code=1, the decoder uses log_alphabet_size=15
    // for parsing IntegerConfig, regardless of actual alphabet size.
    // Tree tokens are raw symbols - set split_exponent = 15.
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;

    // alphabet_size-1 using VarLenUint16 encoding
    write_varlen_u16(writer, max_symbol)?;

    // Huffman table
    let al_size = (max_symbol + 1) as usize;
    if al_size > 1 {
        build_and_store_huffman_tree(tree_histogram, writer)?;
    }

    Ok(())
}

/// Write tree tokens for a single leaf with Weighted predictor.
pub(super) fn write_weighted_tree_tokens(writer: &mut BitWriter) -> Result<()> {
    const TREE_PREDICTOR: u32 = 6;

    // Compute Huffman codes for the tree histogram
    let tree_histogram = &[4u32, 0, 0, 0, 0, 0, 1]; // 5 tokens: 4x symbol 0, 1x symbol 6
    let depths = create_huffman_tree(tree_histogram, 15);
    let codes = convert_bit_depths_to_symbols(&depths);

    // Encode: property=0 (leaf), predictor=6, offset=0, mul_log=0, mul_bits=0
    let tokens = [0u32, TREE_PREDICTOR, 0, 0, 0];

    for &token in &tokens {
        let depth = depths.get(token as usize).copied().unwrap_or(0);
        let code = codes.get(token as usize).copied().unwrap_or(0);
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
    }

    Ok(())
}

/// Write weighted predictor header parameters.
pub(super) fn write_wp_header(
    writer: &mut BitWriter,
    params: &super::predictor::WeightedPredictorParams,
) -> Result<()> {
    if params.is_default() {
        // all_default = 1 (no additional fields)
        writer.write(1, 1)?;
    } else {
        // all_default = 0, write all parameters
        writer.write(1, 0)?;
        writer.write(5, params.p1c as u64)?;
        writer.write(5, params.p2c as u64)?;
        writer.write(5, params.p3ca as u64)?;
        writer.write(5, params.p3cb as u64)?;
        writer.write(5, params.p3cc as u64)?;
        writer.write(5, params.p3cd as u64)?;
        writer.write(5, params.p3ce as u64)?;
        writer.write(4, params.w0 as u64)?;
        writer.write(4, params.w1 as u64)?;
        writer.write(4, params.w2 as u64)?;
        writer.write(4, params.w3 as u64)?;
    }
    Ok(())
}

/// Write an arbitrary MA tree to the bitstream.
///
/// The tree is serialized in BFS order using 6 token contexts:
/// - SPLIT_VAL_CONTEXT (0): splitval (signed via pack_signed)
/// - PROPERTY_CONTEXT (1): property+1 for split nodes, 0 for leaf nodes
/// - PREDICTOR_CONTEXT (2): predictor index
/// - OFFSET_CONTEXT (3): predictor offset (signed via pack_signed)
/// - MULTIPLIER_LOG_CONTEXT (4): multiplier log component
/// - MULTIPLIER_BITS_CONTEXT (5): multiplier bits component
///
/// Layout: lz77.enabled=0 + context_map (6 contexts → 1 histogram) +
///         use_prefix_code=1 + IntegerConfig + alphabet_size + Huffman table + tokens
pub fn write_tree(writer: &mut BitWriter, tree: &super::tree::Tree) -> Result<()> {
    use super::tree::collect_tree_tokens;
    use crate::entropy_coding::hybrid_uint::HybridUintConfig;

    let tokens = collect_tree_tokens(tree);

    // Encode tree token values through HybridUint to keep the Huffman alphabet
    // within the 32768 symbol limit. Config {4,2,0} maps values up to ~40000
    // into tokens 0..63, with extra bits for the remaining value.
    //
    // Previously used raw symbol encoding (split_exponent=15, msb=15), which
    // worked for small splitvals but exceeded the Huffman alphabet limit when
    // tree learning produced large splitval thresholds (e.g., LfFrame DC integers
    // scaled by inv_dc_quant can reach ~40000 for the X channel).
    let hybrid_config = HybridUintConfig::new(4, 2, 0);

    // Encode all values through HybridUint and collect (token, extra_bits, num_extra)
    struct EncodedTreeToken {
        token: u32,
        extra_bits: u32,
        num_extra: u32,
    }

    let mut encoded: Vec<EncodedTreeToken> = Vec::with_capacity(tokens.len());
    let mut max_token: u32 = 0;
    for t in &tokens {
        let val = if t.is_signed {
            pack_signed(t.value)
        } else {
            t.value as u32
        };
        let (token, extra_bits, num_extra) = hybrid_config.encode(val);
        max_token = max_token.max(token);
        encoded.push(EncodedTreeToken {
            token,
            extra_bits,
            num_extra,
        });
    }

    let histogram_size = (max_token + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for e in &encoded {
        histogram[e.token as usize] += 1;
    }

    // lz77.enabled = 0
    writer.write(1, 0)?;

    // Context map: 6 contexts all map to histogram 0
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0

    // use_prefix_code = 1
    writer.write(1, 1)?;

    // IntegerConfig with HybridUint {4,2,0}
    // For Huffman (use_prefix_code=1), the decoder hardcodes log_alpha_size=15
    // (HUFFMAN_MAX_BITS). We must match this when writing the config header,
    // since the number of bits for split_exponent = ceil_log2(log_alpha_size+1).
    const LOG_ALPHABET_SIZE: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE,
        hybrid_config.split_exponent,
        hybrid_config.msb_in_token,
        hybrid_config.lsb_in_token,
    )?;

    // alphabet_size - 1
    write_varlen_u16(writer, max_token as u16)?;

    // Huffman table
    let (depths, codes) = if histogram_size > 1 {
        let table = build_and_store_huffman_tree(&histogram[..histogram_size], writer)?;
        (table.depths, table.codes)
    } else {
        (vec![0u8; histogram_size], vec![0u16; histogram_size])
    };

    // Write tokens + extra bits
    for e in &encoded {
        let depth = depths.get(e.token as usize).copied().unwrap_or(0);
        let code = codes.get(e.token as usize).copied().unwrap_or(0);
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
        if e.num_extra > 0 {
            writer.write(e.num_extra as usize, e.extra_bits as u64)?;
        }
    }

    Ok(())
}
