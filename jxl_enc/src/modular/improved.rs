// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Improved modular encoder with gradient prediction and LZ77.
//!
//! This encoder achieves much better compression than the minimal encoder by:
//! - Using gradient prediction (ClampedGradient) instead of zero prediction
//! - Using LZ77 run-length encoding for consecutive zero residuals
//!
//! Based on techniques from zune-jpegxl.

use crate::bit_writer::BitWriter;
use crate::entropy_coding::huffman_tree::{
    build_and_store_huffman_tree, convert_bit_depths_to_symbols, create_huffman_tree,
};
use crate::error::Result;
use crate::modular::channel::ModularImage;
use crate::modular::rct::{RctType, forward_rct};
use std::collections::HashMap;

// LZ77 constants (from zune-jpegxl)
const K_NUM_RAW_SYMBOLS: usize = 19;
const K_NUM_LZ77: usize = 33;
const K_LZ77_MIN_LENGTH: usize = 7;

/// Pack a signed integer into an unsigned one (zigzag encoding).
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, etc.
#[inline]
pub fn pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

/// Clamped gradient prediction.
/// This is the "Select" predictor used by zune-jpegxl:
/// - Computes gradient = left + top - topleft
/// - Selects between gradient, left, or top based on edge direction
#[inline]
fn predict_clamped_gradient(left: i32, top: i32, topleft: i32) -> i32 {
    let ac = left - topleft;
    let bc = top - topleft;
    let grad = ac + top; // = left + top - topleft

    // XOR trick to detect edge direction
    let d = (left - top) ^ bc;
    let s = ac ^ bc;

    let clamp = if d < 0 { top } else { left };
    if s < 0 { grad } else { clamp }
}

/// Encode a hybrid uint for LZ77 run length.
/// Returns (token, nbits, bits)
#[inline]
fn encode_hybrid_uint_lz77(value: u32) -> (u32, u32, u32) {
    if value < 16 {
        (value, 0, 0)
    } else {
        let n = 31 - (value | 1).leading_zeros();
        let token = 16 + n - 4;
        let bits = value - (1 << n);
        (token, n, bits)
    }
}

/// Encode a hybrid uint for raw symbols (split_exponent=0, msb_in_token=0, lsb_in_token=0).
/// Returns (token, nbits, bits)
#[inline]
fn encode_hybrid_uint_000(value: u32) -> (u32, u32, u32) {
    if value == 0 {
        (0, 0, 0)
    } else {
        let n = 31 - (value | 1).leading_zeros();
        let token = n + 1;
        let bits = value - (1 << n);
        (token, n, bits)
    }
}

/// Residual with run-length information.
enum Token {
    /// A raw residual value
    Raw(u32),
    /// An LZ77 run of zeros (count includes the K_LZ77_MIN_LENGTH offset)
    Lz77Run(usize),
}

/// Collect residuals using gradient prediction and identify LZ77 runs.
fn collect_residuals_with_prediction(image: &ModularImage) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current_run = 0usize;

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                // Get neighbors
                let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                let top = if y > 0 { channel.get(x, y - 1) } else { 0 };
                let topleft = if x > 0 && y > 0 {
                    channel.get(x - 1, y - 1)
                } else if y > 0 {
                    channel.get(0, y - 1)
                } else {
                    0
                };

                // Predict
                let prediction = predict_clamped_gradient(left, top, topleft);
                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                if packed == 0 {
                    current_run += 1;
                } else {
                    // Flush any accumulated run
                    if current_run > K_LZ77_MIN_LENGTH {
                        tokens.push(Token::Lz77Run(current_run));
                    } else {
                        // Output individual zeros
                        for _ in 0..current_run {
                            tokens.push(Token::Raw(0));
                        }
                    }
                    current_run = 0;
                    tokens.push(Token::Raw(packed));
                }
            }
        }
    }

    // Flush final run
    if current_run > K_LZ77_MIN_LENGTH {
        tokens.push(Token::Lz77Run(current_run));
    } else {
        for _ in 0..current_run {
            tokens.push(Token::Raw(0));
        }
    }

    tokens
}

/// LZ77 min_symbol value - symbols >= this are LZ77 length tokens
const K_LZ77_MIN_SYMBOL: usize = 224;

/// Build a single sparse histogram for symbols [0..K_NUM_RAW_SYMBOLS) and [K_LZ77_MIN_SYMBOL..K_LZ77_MIN_SYMBOL+K_NUM_LZ77)
fn build_sparse_histogram(tokens: &[Token]) -> Vec<u64> {
    // Sparse alphabet: 19 raw symbols + 33 LZ77 symbols = 52 symbols
    // We'll encode raw [0..18] directly, LZ77 as [224..256]
    let total_symbols = K_LZ77_MIN_SYMBOL + K_NUM_LZ77;
    let mut counts = vec![0u64; total_symbols];

    for token in tokens {
        match token {
            Token::Raw(value) => {
                let (tok, _, _) = encode_hybrid_uint_000(*value);
                if (tok as usize) < K_NUM_RAW_SYMBOLS {
                    counts[tok as usize] += 1;
                }
            }
            Token::Lz77Run(count) => {
                let adjusted = count - K_LZ77_MIN_LENGTH - 1;
                let (tok, _, _) = encode_hybrid_uint_lz77(adjusted as u32);
                let symbol = K_LZ77_MIN_SYMBOL + tok as usize;
                if symbol < total_symbols {
                    counts[symbol] += 1;
                }
            }
        }
    }

    // Also count distance symbols (all 1 for RLE)
    // Distance context is separate, we'll handle it differently

    counts
}

/// Compute Huffman code lengths using a simple algorithm.
#[allow(dead_code)]
fn compute_code_lengths(counts: &[u64], max_len: u8) -> Vec<u8> {
    let n = counts.len();
    if n == 0 {
        return vec![];
    }

    // Find number of non-zero counts
    let num_used: usize = counts.iter().filter(|&&c| c > 0).count();
    if num_used == 0 {
        return vec![0; n];
    }
    if num_used == 1 {
        // Single symbol - use depth 1
        let mut depths = vec![0u8; n];
        for (i, &c) in counts.iter().enumerate() {
            if c > 0 {
                depths[i] = 1;
                break;
            }
        }
        return depths;
    }

    // Use our existing Huffman tree builder
    let histogram: Vec<u32> = counts.iter().map(|&c| c as u32).collect();
    create_huffman_tree(&histogram, max_len)
}

/// Compute canonical Huffman codes from bit depths.
fn compute_codes_from_depths(depths: &[u8]) -> Vec<u16> {
    let max_depth = *depths.iter().max().unwrap_or(&0) as usize;
    if max_depth == 0 {
        return vec![0; depths.len()];
    }

    // Count symbols at each depth
    let mut bl_count = vec![0u32; max_depth + 1];
    for &d in depths {
        if d > 0 {
            bl_count[d as usize] += 1;
        }
    }

    // Compute first code for each depth
    let mut next_code = vec![0u16; max_depth + 1];
    let mut code = 0u16;
    for bits in 1..=max_depth {
        code = (code + bl_count[bits - 1] as u16) << 1;
        next_code[bits] = code;
    }

    // Assign codes
    let mut codes = vec![0u16; depths.len()];
    for (i, &d) in depths.iter().enumerate() {
        if d > 0 {
            codes[i] = next_code[d as usize];
            next_code[d as usize] += 1;
        }
    }

    codes
}

/// Writes a varint16 value to the bitstream.
fn write_varint16(writer: &mut BitWriter, value: u16) -> Result<()> {
    if value == 0 {
        writer.write(1, 0)?;
    } else if value == 1 {
        writer.write(1, 1)?;
        writer.write(4, 0)?;
    } else {
        writer.write(1, 1)?;
        let nbits = 15 - value.leading_zeros() as usize;
        let mantissa = value - (1 << nbits);
        writer.write(4, nbits as u64)?;
        writer.write(nbits, mantissa as u64)?;
    }
    Ok(())
}

/// Write the LZ77-enabled histogram using sparse alphabet.
/// Returns (depths, codes) for the full sparse alphabet [0..257]
fn write_sparse_lz77_histogram(
    writer: &mut BitWriter,
    sparse_counts: &[u64],
) -> Result<(Vec<u8>, Vec<u16>)> {
    eprintln!("SPARSE_HIST: Writing LZ77-enabled histogram");

    // lz77.enabled = 1
    writer.write(1, 1)?;
    eprintln!(
        "SPARSE_HIST [bit {}]: lz77.enabled = 1",
        writer.bits_written()
    );

    // lz77.min_symbol = 224 (u2S encoding)
    // u2S(224, Bits(8)+225, Bits(16)+481, Bits(32)+65537)
    // 224 = selector 0 means value IS 224
    writer.write(2, 0)?; // selector 0: value = 224
    eprintln!(
        "SPARSE_HIST [bit {}]: min_symbol = 224",
        writer.bits_written()
    );

    // lz77.min_length = K_LZ77_MIN_LENGTH = 7
    // u2S(3, 4, Bits(2)+5, Bits(8)+9)
    // 7 = Bits(2)+5 with bits=2, so selector 2
    writer.write(2, 2)?; // selector 2
    writer.write(2, 2)?; // 7 - 5 = 2
    eprintln!(
        "SPARSE_HIST [bit {}]: min_length = 7",
        writer.bits_written()
    );

    // Find the actual used symbols
    let max_raw_symbol = sparse_counts[..K_NUM_RAW_SYMBOLS]
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, _)| i)
        .max()
        .unwrap_or(0);

    let max_lz77_symbol = sparse_counts[K_LZ77_MIN_SYMBOL..]
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, _)| K_LZ77_MIN_SYMBOL + i)
        .max()
        .unwrap_or(K_LZ77_MIN_SYMBOL);

    eprintln!(
        "SPARSE_HIST: max_raw={}, max_lz77={} (count at lz77={})",
        max_raw_symbol,
        max_lz77_symbol,
        sparse_counts.get(K_LZ77_MIN_SYMBOL).unwrap_or(&0)
    );

    // Build histogram for Huffman tree - only non-zero symbols
    // For sparse alphabets, we use the complex prefix code path
    let histogram: Vec<u32> = sparse_counts.iter().map(|&c| c as u32).collect();

    // Count actual used symbols
    let num_used: usize = histogram.iter().filter(|&&c| c > 0).count();
    eprintln!("SPARSE_HIST: {} used symbols", num_used);

    // Compute depths for all symbols (including zeros in gaps)
    let depths = create_huffman_tree(&histogram, 15);
    let codes = compute_codes_from_depths(&depths);

    // Use the Huffman tree builder to store the prefix code
    // First write use_prefix_code = 1
    writer.write(1, 1)?;
    eprintln!(
        "SPARSE_HIST [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );

    // split_exponent = 15
    writer.write(4, 15)?;
    eprintln!(
        "SPARSE_HIST [bit {}]: split_exponent = 15",
        writer.bits_written()
    );

    // Alphabet size - for LZ77 histograms, this is max symbol + 1
    // But with sparse, we need to account for the full range up to max_lz77_symbol
    let alphabet_size = max_lz77_symbol + 1;
    write_varint16(writer, (alphabet_size - 1) as u16)?;
    eprintln!(
        "SPARSE_HIST [bit {}]: alphabet_size = {} (max_symbol={})",
        writer.bits_written(),
        alphabet_size,
        alphabet_size - 1
    );

    // Write Huffman table if alphabet size > 1
    if alphabet_size > 1 {
        // For sparse alphabets, we use build_and_store_huffman_tree
        // which handles the complex prefix code encoding
        build_and_store_huffman_tree(&histogram[..alphabet_size], writer)?;
        eprintln!(
            "SPARSE_HIST [bit {}]: After Huffman table",
            writer.bits_written()
        );
    }

    // Distance context histogram - for RLE we always use distance=1
    // This is a single-symbol histogram (only symbol for distance=1)
    eprintln!(
        "SPARSE_HIST [bit {}]: Writing distance histogram",
        writer.bits_written()
    );

    // use_prefix_code = 1
    writer.write(1, 1)?;
    // split_exponent = 15
    writer.write(4, 15)?;
    // alphabet_size = 1 (only distance symbol 0 which encodes distance=1)
    write_varint16(writer, 0)?;
    eprintln!(
        "SPARSE_HIST [bit {}]: Distance histogram done (single symbol)",
        writer.bits_written()
    );

    Ok((depths, codes))
}

/// Writes an improved modular stream with gradient prediction and LZ77.
pub fn write_improved_modular_stream(image: &ModularImage, writer: &mut BitWriter) -> Result<()> {
    // Collect residuals with gradient prediction
    let tokens = collect_residuals_with_prediction(image);

    // Build sparse histogram [0..18] + [224..256]
    let sparse_counts = build_sparse_histogram(&tokens);

    let num_raw_used = sparse_counts[..K_NUM_RAW_SYMBOLS]
        .iter()
        .filter(|&&c| c > 0)
        .count();
    let num_lz77_used = sparse_counts[K_LZ77_MIN_SYMBOL..]
        .iter()
        .filter(|&&c| c > 0)
        .count();
    let num_lz77_runs = tokens
        .iter()
        .filter(|t| matches!(t, Token::Lz77Run(_)))
        .count();

    eprintln!(
        "IMPROVED: {} tokens, {} raw symbols used, {} lz77 tokens used, {} lz77 runs",
        tokens.len(),
        num_raw_used,
        num_lz77_used,
        num_lz77_runs
    );

    // If no LZ77 runs, fall back to simple encoding
    if num_lz77_runs == 0 {
        return write_simple_modular_stream(image, writer);
    }

    // === Global section (LfGlobal) ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    // Tree histogram for single-leaf tree with Gradient predictor
    write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer)?;

    // Data histogram with LZ77 (sparse alphabet)
    let (depths, codes) = write_sparse_lz77_histogram(writer, &sparse_counts)?;

    // GroupHeader
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true
    writer.write(2, 0)?; // num_transforms = 0

    // Encode tokens using sparse alphabet
    for token in &tokens {
        match token {
            Token::Raw(value) => {
                // Encode value using hybrid uint (split_exponent=0, msb_in_token=0, lsb_in_token=0)
                let (tok, nbits, extra) = encode_hybrid_uint_000(*value);
                let symbol = tok as usize;
                let depth = depths[symbol];
                let code = codes[symbol];
                if depth > 0 {
                    writer.write(depth as usize, code as u64)?;
                }
                // Write extra bits
                if nbits > 0 {
                    writer.write(nbits as usize, extra as u64)?;
                }
            }
            Token::Lz77Run(count) => {
                // LZ77: encode as single symbol >= 224
                let adjusted = count - K_LZ77_MIN_LENGTH - 1;
                let (tok, nbits, extra) = encode_hybrid_uint_lz77(adjusted as u32);

                // Symbol in sparse alphabet
                let symbol = K_LZ77_MIN_SYMBOL + tok as usize;
                let depth = depths[symbol];
                let code = codes[symbol];
                if depth > 0 {
                    writer.write(depth as usize, code as u64)?;
                }

                // Write extra bits for length
                if nbits > 0 {
                    writer.write(nbits as usize, extra as u64)?;
                }

                // Write distance symbol (always 0 for RLE, distance=1)
                // Distance histogram has single symbol, so no bits needed
            }
        }
    }

    eprintln!(
        "LZ77 [bit {}]: Encoded {} tokens",
        writer.bits_written(),
        tokens.len()
    );

    writer.zero_pad_to_byte();
    Ok(())
}

/// Clamped gradient predictor (predictor 5 in JXL spec).
/// Returns `left + top - topleft` clamped to [min(left,top), max(left,top)]
/// when topleft is outside that range.
#[inline]
fn predict_gradient(left: i32, top: i32, topleft: i32) -> i32 {
    let min = left.min(top);
    let max = left.max(top);
    let grad = left + top - topleft;
    let grad_clamp_max = if topleft < min { max } else { grad };
    if topleft > max { min } else { grad_clamp_max }
}

/// Write a tree histogram for a single-leaf tree with Zero predictor.
/// This should produce the same output as minimal.rs for tree histogram.
fn write_tree_histogram_for_zero(writer: &mut BitWriter) -> Result<()> {
    eprintln!(
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

    // split_exponent = 15
    writer.write(4, 15)?;

    // alphabet_size = 1 (max_symbol = 0)
    write_varint16(writer, 0)?;

    // No Huffman table needed for alphabet_size = 1
    eprintln!(
        "  TREE_HIST [bit {}]: After tree histogram (Zero)",
        writer.bits_written()
    );

    Ok(())
}

/// Write a tree histogram that can encode tokens 0-5 (for Gradient predictor).
/// Uses full Huffman encoding with histogram [3,0,0,0,0,1] for tokens [0,5,0,0].
fn write_tree_histogram_for_gradient(writer: &mut BitWriter) -> Result<()> {
    eprintln!(
        "  TREE_HIST [bit {}]: Starting tree histogram",
        writer.bits_written()
    );

    // lz77.enabled = 0
    writer.write(1, 0)?;
    eprintln!(
        "  TREE_HIST [bit {}]: lz77.enabled = 0",
        writer.bits_written()
    );

    // Context map for 6 contexts: is_simple=1, bits_per_entry=0
    // (all contexts share same histogram)
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0
    eprintln!(
        "  TREE_HIST [bit {}]: context_map (is_simple=1, bits=0)",
        writer.bits_written()
    );

    // use_prefix_code = 1
    writer.write(1, 1)?;
    eprintln!(
        "  TREE_HIST [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );

    // split_exponent = 15
    writer.write(4, 15)?;
    eprintln!(
        "  TREE_HIST [bit {}]: split_exponent = 15",
        writer.bits_written()
    );

    // Build full Huffman table
    // Tree tokens for leaf: property(ctx1), predictor(ctx2), offset(ctx3), mul_log(ctx4), mul_bits(ctx5)
    // For TREE_PREDICTOR=0: tokens [0,0,0,0,0] → histogram [5] (single symbol 0)
    // For TREE_PREDICTOR=5: tokens [0,5,0,0,0] → histogram [4,0,0,0,0,1] (symbols 0 and 5)
    const TREE_PREDICTOR: u32 = 5; // Must match write_gradient_tree_tokens (0=Zero, 5=Gradient)
    let max_symbol = if TREE_PREDICTOR == 0 { 0u16 } else { 5u16 };
    let tree_histogram: &[u32] = if TREE_PREDICTOR == 0 {
        &[5u32] // Single symbol 0 (5 tokens)
    } else {
        &[4u32, 0, 0, 0, 0, 1] // Symbols 0 (4x) and 5 (1x)
    };

    // alphabet_size via varint16
    write_varint16(writer, max_symbol)?;
    eprintln!(
        "  TREE_HIST [bit {}]: max_symbol = {}",
        writer.bits_written(),
        max_symbol
    );

    // Huffman table: skip if alphabet_size == 1 (only one possible symbol)
    let al_size = (max_symbol + 1) as usize;
    if al_size > 1 {
        build_and_store_huffman_tree(tree_histogram, writer)?;
        eprintln!(
            "  TREE_HIST [bit {}]: After Huffman table",
            writer.bits_written()
        );
    } else {
        eprintln!(
            "  TREE_HIST [bit {}]: No Huffman table (al_size=1)",
            writer.bits_written()
        );
    }

    Ok(())
}

/// Write tree tokens for a single leaf with Gradient predictor.
fn write_gradient_tree_tokens(writer: &mut BitWriter) -> Result<()> {
    eprintln!(
        "  TREE_TOKENS [bit {}]: Starting tree tokens",
        writer.bits_written()
    );

    // Tree tokens for single leaf:
    // - property = 0 (leaf indicator, context 1)
    // - predictor (context 2)
    // - offset = 0 (context 3)
    // - mul_log = 0 (context 4)
    // - mul_bits = 0 (context 5) → multiplier = (0+1) << 0 = 1

    // The predictor to use (0=Zero, 5=Gradient) - must match write_tree_histogram_for_gradient
    const TREE_PREDICTOR: u32 = 5;

    // For single-symbol histogram, depth is 0 (no bits needed to encode)
    // For 2+ symbols, we compute the Huffman codes
    let (depths, codes): (Vec<u8>, Vec<u16>) = if TREE_PREDICTOR == 0 {
        // Single symbol 0: depth 0, no bits needed
        (vec![0u8], vec![0u16])
    } else {
        let tree_histogram = &[4u32, 0, 0, 0, 0, 1]; // 5 tokens: 4x symbol 0, 1x symbol 5
        let d = create_huffman_tree(tree_histogram, 15);
        let c = convert_bit_depths_to_symbols(&d);
        (d, c)
    };

    eprintln!("  TREE_TOKENS: depths = {:?}", depths);
    eprintln!("  TREE_TOKENS: codes = {:?}", codes);

    // Encode: property=0 (leaf), predictor, offset=0, mul_log=0, mul_bits=0
    let tokens = [0u32, TREE_PREDICTOR, 0, 0, 0];
    let token_names = ["property", "predictor", "offset", "mul_log", "mul_bits"];

    for (i, &token) in tokens.iter().enumerate() {
        let depth = depths.get(token as usize).copied().unwrap_or(0);
        let code = codes.get(token as usize).copied().unwrap_or(0);
        eprintln!(
            "  TREE_TOKENS [bit {}]: {} = {} (depth={}, code={:0width$b})",
            writer.bits_written(),
            token_names[i],
            token,
            depth,
            code,
            width = depth.max(1) as usize
        );
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
    }

    eprintln!("  TREE_TOKENS [bit {}]: Done", writer.bits_written());
    Ok(())
}

// Set to true to use Zero predictor for debugging
// Try: false = gradient tree path, true = zero tree path (works)
const USE_ZERO_PREDICTOR: bool = false;

/// Simpler stream without LZ77 but with gradient prediction.
pub fn write_simple_modular_stream(image: &ModularImage, writer: &mut BitWriter) -> Result<()> {
    // Collect residuals with gradient prediction
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                let prediction = if USE_ZERO_PREDICTOR {
                    // Zero predictor: predict 0 always
                    0
                } else {
                    // Get neighbors (matching jxl-rs decoder)
                    let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                    let top = if y > 0 { channel.get(x, y - 1) } else { left };
                    let topleft = if x > 0 && y > 0 {
                        channel.get(x - 1, y - 1)
                    } else {
                        left
                    };
                    // Predict using clamped gradient (predictor 5)
                    predict_gradient(left, top, topleft)
                };

                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                residuals.push(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    // Build histogram
    let histogram_size = (max_residual + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for &r in &residuals {
        histogram[r as usize] += 1;
    }

    let num_symbols = histogram.iter().filter(|&&c| c > 0).count();
    let num_zeros = histogram.first().copied().unwrap_or(0);
    eprintln!(
        "GRADIENT: {} residuals, {} unique symbols, max={}, zeros={}",
        residuals.len(),
        num_symbols,
        max_residual,
        num_zeros
    );

    // === Global section ===
    eprintln!(
        "IMPROVED [bit {}]: Starting modular stream",
        writer.bits_written()
    );
    writer.write(1, 1)?; // dc_quant.all_default = true
    eprintln!(
        "IMPROVED [bit {}]: dc_quant.all_default = 1",
        writer.bits_written()
    );
    writer.write(1, 1)?; // has_tree = true
    eprintln!("IMPROVED [bit {}]: has_tree = 1", writer.bits_written());

    if USE_ZERO_PREDICTOR {
        // Tree histogram for Zero predictor (single symbol, no explicit tree tokens)
        write_tree_histogram_for_zero(writer)?;
        // No tree tokens needed - all implicit zeros
    } else {
        // Tree histogram (supports symbols 0-5 for Gradient predictor)
        write_tree_histogram_for_gradient(writer)?;
        // Tree tokens for single leaf with Gradient predictor
        write_gradient_tree_tokens(writer)?;
    }

    eprintln!(
        "IMPROVED [bit {}]: Starting data histogram",
        writer.bits_written()
    );
    // Data histogram
    writer.write(1, 0)?; // lz77.enabled = 0
    eprintln!("IMPROVED [bit {}]: lz77.enabled = 0", writer.bits_written());
    writer.write(1, 1)?; // use_prefix_code = 1
    eprintln!(
        "IMPROVED [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );
    writer.write(4, 15)?; // split_exponent = 15
    eprintln!(
        "IMPROVED [bit {}]: split_exponent = 15",
        writer.bits_written()
    );
    write_varint16(writer, max_residual as u16)?;
    eprintln!(
        "IMPROVED [bit {}]: max_symbol = {}",
        writer.bits_written(),
        max_residual
    );

    if histogram_size > 1 {
        eprintln!(
            "IMPROVED [bit {}]: Starting Huffman table (histogram_size={})",
            writer.bits_written(),
            histogram_size
        );
        build_and_store_huffman_tree(&histogram, writer)?;
        eprintln!(
            "IMPROVED [bit {}]: After Huffman table",
            writer.bits_written()
        );
    }

    // GroupHeader
    eprintln!(
        "IMPROVED [bit {}]: Writing GroupHeader",
        writer.bits_written()
    );
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true
    writer.write(2, 0)?; // num_transforms = 0
    eprintln!(
        "IMPROVED [bit {}]: After GroupHeader",
        writer.bits_written()
    );

    // Encode residuals
    let depths = create_huffman_tree(&histogram, 15);
    let codes = convert_bit_depths_to_symbols(&depths);
    let code_map: HashMap<u32, (u16, u8)> = depths
        .iter()
        .zip(codes.iter())
        .enumerate()
        .filter(|(_, (d, _))| **d > 0)
        .map(|(i, (d, c))| (i as u32, (*c, *d)))
        .collect();

    eprintln!(
        "IMPROVED [bit {}]: Starting residual encoding",
        writer.bits_written()
    );
    eprintln!(
        "IMPROVED: First 20 residuals: {:?}",
        &residuals[..residuals.len().min(20)]
    );
    eprintln!("IMPROVED: code_map has {} entries", code_map.len());
    for (&sym, &(code, depth)) in &code_map {
        eprintln!(
            "  sym {} -> code {:0width$b} (depth {})",
            sym,
            code,
            depth,
            width = depth as usize
        );
    }

    for (i, &r) in residuals.iter().enumerate() {
        if let Some(&(code, depth)) = code_map.get(&r) {
            if depth > 0 {
                if i < 10 {
                    eprintln!(
                        "  residual[{}] = {} -> code {:0width$b} (depth {})",
                        i,
                        r,
                        code,
                        depth,
                        width = depth as usize
                    );
                }
                writer.write(depth as usize, code as u64)?;
            }
        } else {
            eprintln!("  WARNING: residual[{}] = {} has no code!", i, r);
        }
    }

    eprintln!("IMPROVED [bit {}]: Before padding", writer.bits_written());
    writer.zero_pad_to_byte();
    eprintln!(
        "IMPROVED [bit {}]: After padding (done)",
        writer.bits_written()
    );
    Ok(())
}

/// Write the RCT transform descriptor to the bitstream.
///
/// Format (for YCoCg with begin_c=0):
/// - TransformId: 2 bits (selector 0 = RCT)
/// - begin_c: 2 bits selector + 3 bits value = 5 bits for value 0
/// - rct_type: 2 bits (selector 0 = 6 = YCoCg)
fn write_rct_transform(writer: &mut BitWriter, begin_c: usize, rct_type: RctType) -> Result<()> {
    // TransformId: U32(Val(0)=RCT, Val(1)=Palette, Val(2)=Squeeze, Val(3)=Invalid)
    // RCT = selector 0 = 2 bits "00"
    writer.write(2, 0)?;

    // begin_c: U32(Bits(3), BitsOffset(6, 8), BitsOffset(10, 72), BitsOffset(13, 1096), 0)
    // For begin_c 0-7: selector 0 = 2 bits + 3 bits value
    if begin_c < 8 {
        writer.write(2, 0)?; // selector 0
        writer.write(3, begin_c as u64)?;
    } else if begin_c < 72 {
        writer.write(2, 1)?; // selector 1 = BitsOffset(6, 8)
        writer.write(6, (begin_c - 8) as u64)?;
    } else if begin_c < 1096 {
        writer.write(2, 2)?; // selector 2 = BitsOffset(10, 72)
        writer.write(10, (begin_c - 72) as u64)?;
    } else {
        writer.write(2, 3)?; // selector 3 = BitsOffset(13, 1096)
        writer.write(13, (begin_c - 1096) as u64)?;
    }

    // rct_type: U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 18), 6)
    // Val(6) = YCoCg at selector 0
    // Bits(2) = 0-3 at selector 1
    // BitsOffset(4, 2) = 2-17 at selector 2
    // BitsOffset(6, 18) = 18-81 at selector 3
    let rct_val = rct_type.0 as u64;
    if rct_val == 6 {
        writer.write(2, 0)?; // selector 0 = Val(6)
    } else if rct_val < 2 {
        // 0-1 encoded as Bits(2) at selector 1, but 0 and 1 need 2 bits
        writer.write(2, 1)?;
        writer.write(2, rct_val)?;
    } else if rct_val < 18 {
        // 2-17 encoded as BitsOffset(4, 2) at selector 2
        writer.write(2, 2)?;
        writer.write(4, rct_val - 2)?;
    } else {
        // 18-81 encoded as BitsOffset(6, 18) at selector 3
        writer.write(2, 3)?;
        writer.write(6, rct_val - 18)?;
    }

    Ok(())
}

/// Write modular stream with RCT (YCoCg) transform for RGB images.
///
/// This function:
/// 1. Applies YCoCg RCT to decorrelate RGB channels
/// 2. Signals the transform in the bitstream
/// 3. Encodes the transformed data
///
/// YCoCg improves compression by 15-20% for typical RGB images.
pub fn write_modular_stream_with_rct(image: &ModularImage, writer: &mut BitWriter) -> Result<()> {
    // Only apply RCT to RGB images (3+ channels)
    if image.channels.len() < 3 {
        // Fall back to simple encoding for grayscale
        return write_simple_modular_stream(image, writer);
    }

    // Clone the image and apply forward RCT
    let mut transformed = image.clone();
    let rct_type = RctType::YCOCG; // Best for typical images
    forward_rct(&mut transformed.channels, 0, rct_type)?;

    eprintln!(
        "RCT: Applied YCoCg transform to {} channels",
        transformed.channels.len()
    );

    // Collect residuals with gradient prediction on transformed channels
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &transformed.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                let top = if y > 0 { channel.get(x, y - 1) } else { left };
                let topleft = if x > 0 && y > 0 {
                    channel.get(x - 1, y - 1)
                } else {
                    left
                };
                let prediction = predict_gradient(left, top, topleft);

                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                residuals.push(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    // Build histogram
    let histogram_size = (max_residual + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for &r in &residuals {
        histogram[r as usize] += 1;
    }

    let num_symbols = histogram.iter().filter(|&&c| c > 0).count();
    eprintln!(
        "RCT: {} residuals, {} unique symbols, max={}",
        residuals.len(),
        num_symbols,
        max_residual
    );

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    // Tree histogram (Gradient predictor)
    write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer)?;

    // Data histogram
    writer.write(1, 0)?; // lz77.enabled = 0
    writer.write(1, 1)?; // use_prefix_code = 1
    writer.write(4, 15)?; // split_exponent = 15
    write_varint16(writer, max_residual as u16)?;

    if histogram_size > 1 {
        build_and_store_huffman_tree(&histogram, writer)?;
    }

    // GroupHeader with 1 transform
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true

    // num_transforms = 1: U32(Val(0), Val(1), BitsOffset(4, 2), BitsOffset(8, 18), 0)
    // Val(1) at selector 1 = 2 bits "01"
    writer.write(2, 1)?;

    // Write the RCT transform descriptor
    write_rct_transform(writer, 0, rct_type)?;

    eprintln!(
        "RCT [bit {}]: After GroupHeader with transform",
        writer.bits_written()
    );

    // Encode residuals
    let depths = create_huffman_tree(&histogram, 15);
    let codes = convert_bit_depths_to_symbols(&depths);
    let code_map: HashMap<u32, (u16, u8)> = depths
        .iter()
        .zip(codes.iter())
        .enumerate()
        .filter(|(_, (d, _))| **d > 0)
        .map(|(i, (d, c))| (i as u32, (*c, *d)))
        .collect();

    for &r in &residuals {
        if let Some(&(code, depth)) = code_map.get(&r)
            && depth > 0
        {
            writer.write(depth as usize, code as u64)?;
        }
    }

    writer.zero_pad_to_byte();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_signed() {
        assert_eq!(pack_signed(0), 0);
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(1), 2);
        assert_eq!(pack_signed(-2), 3);
        assert_eq!(pack_signed(2), 4);
    }

    #[test]
    fn test_predict_clamped_gradient() {
        // Smooth gradient: should predict correctly
        assert_eq!(predict_clamped_gradient(10, 10, 10), 10);

        // Diagonal gradient
        assert_eq!(predict_clamped_gradient(20, 10, 10), 20);
    }

    #[test]
    fn test_encode_hybrid_uint() {
        assert_eq!(encode_hybrid_uint_000(0), (0, 0, 0));
        assert_eq!(encode_hybrid_uint_000(1), (1, 0, 0));
        assert_eq!(encode_hybrid_uint_000(2), (2, 1, 0));
        assert_eq!(encode_hybrid_uint_000(3), (2, 1, 1));
    }

    #[test]
    fn test_gradient_stream() {
        // 4x4 image
        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_simple_modular_stream(&image, &mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("Gradient stream: {} bytes", bytes.len());
    }

    #[test]
    fn test_rct_stream() {
        // 4x4 RGB image with smooth gradients (good for RCT)
        let mut data = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let base = (y * 4 + x) * 10;
                data.push(base as u8); // R
                data.push((base + 5) as u8); // G
                data.push((base + 10) as u8); // B
            }
        }
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_rct(&image, &mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("RCT stream: {} bytes", bytes.len());
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_write_rct_transform() {
        use crate::bit_writer::BitWriter;

        let mut writer = BitWriter::new();
        write_rct_transform(&mut writer, 0, RctType::YCOCG).unwrap();

        // YCoCg with begin_c=0 should be:
        // - TransformId=RCT: 2 bits (00)
        // - begin_c=0: 5 bits (00 + 000)
        // - rct_type=6: 2 bits (00)
        // Total: 9 bits
        assert_eq!(writer.bits_written(), 9);
    }
}
