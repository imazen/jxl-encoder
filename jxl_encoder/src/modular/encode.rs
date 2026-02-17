// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Improved modular encoder with gradient prediction and LZ77.
//!
//! This encoder achieves much better compression than the minimal encoder by:
//! - Using gradient prediction (ClampedGradient) instead of zero prediction
//! - Using LZ77 run-length encoding for consecutive zero residuals
//!
//! Based on techniques from zune-jpegxl.

use crate::bit_writer::BitWriter;
use crate::entropy_coding::encode::{
    OwnedAnsEntropyCode, build_entropy_code_ans, write_tokens_ans,
};
use crate::entropy_coding::huffman_tree::{
    build_and_store_huffman_tree, convert_bit_depths_to_symbols, create_huffman_tree,
};
use crate::entropy_coding::hybrid_uint::HybridUintConfig;
use crate::entropy_coding::token::Token as AnsToken;
use crate::error::Result;
use crate::modular::channel::ModularImage;
use crate::modular::rct::{RctType, forward_rct};

// LZ77 constants (from zune-jpegxl)
pub(crate) const K_NUM_RAW_SYMBOLS: usize = 19;
pub(crate) const K_NUM_LZ77: usize = 33;
pub(crate) const K_LZ77_MIN_LENGTH: usize = 7;

/// Pack a signed integer into an unsigned one (zigzag encoding).
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, etc.
#[inline]
pub fn pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

/// Encode a hybrid uint for LZ77 run length using config {0, 0, 0}.
/// This matches libjxl's default length_uint_config.
/// Returns (token, nbits, bits)
#[inline]
pub(crate) fn encode_hybrid_uint_lz77_length(value: u32) -> (u32, u32, u32) {
    // LZ77 length uses HybridUintConfig{0, 0, 0} (same as raw symbols)
    encode_hybrid_uint_000(value)
}

/// Encode a hybrid uint for raw symbols (split_exponent=0, msb_in_token=0, lsb_in_token=0).
/// Returns (token, nbits, bits)
#[inline]
pub(crate) fn encode_hybrid_uint_000(value: u32) -> (u32, u32, u32) {
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
pub(crate) enum Token {
    /// A raw residual value
    Raw(u32),
    /// An LZ77 run of zeros (count includes the K_LZ77_MIN_LENGTH offset)
    Lz77Run(usize),
}

/// Default HybridUint config for modular data: split_exponent=4, msb_in_token=2, lsb_in_token=0.
/// This reduces the token alphabet from hundreds of symbols (raw) to ~36 tokens + extra bits.
const MODULAR_HYBRID_UINT: HybridUintConfig = HybridUintConfig {
    split_exponent: 4,
    split: 16, // 1 << 4
    msb_in_token: 2,
    lsb_in_token: 0,
};

/// Pre-encoded residual: the HybridUint token and its extra bits.
struct EncodedResidual {
    token: u32,
    extra_bits: u32,
    num_extra: u32,
}

/// Encode a list of packed residuals through HybridUint, returning encoded tokens
/// and the maximum token value (for histogram sizing).
fn encode_residuals_hybrid(residuals: &[u32]) -> (Vec<EncodedResidual>, u32) {
    let mut encoded = Vec::with_capacity(residuals.len());
    let mut max_token: u32 = 0;
    for &r in residuals {
        let (token, extra_bits, num_extra) = MODULAR_HYBRID_UINT.encode(r);
        max_token = max_token.max(token);
        encoded.push(EncodedResidual {
            token,
            extra_bits,
            num_extra,
        });
    }
    (encoded, max_token)
}

/// Build a histogram from HybridUint-encoded tokens.
fn build_token_histogram(encoded: &[EncodedResidual], max_token: u32) -> Vec<u32> {
    let size = (max_token + 1) as usize;
    let mut histogram = vec![0u32; size];
    for e in encoded {
        histogram[e.token as usize] += 1;
    }
    histogram
}

/// Write the data histogram header using HybridUint config {4,2,0} and Huffman prefix codes.
/// Returns (depths, codes) for encoding tokens.
pub(crate) fn write_hybrid_data_histogram(
    writer: &mut BitWriter,
    histogram: &[u32],
    max_token: u32,
) -> Result<(Vec<u8>, Vec<u16>)> {
    // lz77.enabled = 0
    writer.write(1, 0)?;
    // use_prefix_code = 1
    writer.write(1, 1)?;

    // IntegerConfig with HybridUint {4, 2, 0}
    // When use_prefix_code=1, decoder uses log_alphabet_size=15 for parsing IntegerConfig.
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(writer, LOG_ALPHABET_SIZE_PREFIX, 4, 2, 0)?;

    // alphabet_size - 1
    write_varlen_u16(writer, max_token as u16)?;

    // Huffman table
    let histogram_size = (max_token + 1) as usize;
    let (depths, codes) = if histogram_size > 1 {
        let table = build_and_store_huffman_tree(&histogram[..histogram_size], writer)?;
        (table.depths, table.codes)
    } else {
        (vec![0u8; histogram_size], vec![0u16; histogram_size])
    };

    Ok((depths, codes))
}

/// Encode HybridUint residuals using Huffman codes + extra bits.
fn write_hybrid_residuals(
    writer: &mut BitWriter,
    encoded: &[EncodedResidual],
    depths: &[u8],
    codes: &[u16],
) -> Result<()> {
    for e in encoded {
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

/// Build ANS tokens from packed residuals (all context 0, single-context modular stream).
fn build_ans_tokens(residuals: &[u32]) -> Vec<AnsToken> {
    residuals.iter().map(|&r| AnsToken::new(0, r)).collect()
}

/// Build the ANS entropy code for modular residuals.
/// Returns (tokens, code) for separate header/token writing.
pub(crate) fn build_ans_modular_code(residuals: &[u32]) -> (Vec<AnsToken>, OwnedAnsEntropyCode) {
    let tokens = build_ans_tokens(residuals);
    let code = build_entropy_code_ans(&tokens, 1); // 1 context for single-leaf tree
    (tokens, code)
}

/// Write ANS data histogram header for a single-context modular stream.
///
/// For modular with a single-leaf MA tree (num_dist=1), the context map is NOT written
/// (the spec skips it when num_dist=1). This differs from VarDCT which has multiple contexts
/// and always writes a context map via write_entropy_code_ans.
///
/// Layout: lz77.enabled=0 + use_prefix_code=0 + log_alpha_size + HybridUint config + ANS distribution
pub(crate) fn write_ans_modular_header(
    writer: &mut BitWriter,
    code: &OwnedAnsEntropyCode,
) -> Result<()> {
    assert_eq!(
        code.histograms.len(),
        1,
        "modular ANS header only supports single-distribution (single-leaf tree)"
    );

    // lz77.enabled = 0
    writer.write(1, 0)?;

    // NO context map for num_dist=1 (spec: context map is only written when num_dist > 1)

    // use_prefix_code = 0 (ANS, not Huffman)
    writer.write(1, 0)?;

    // log_alpha_size - 5 (2 bits)
    let las = code.log_alpha_size;
    writer.write(2, (las - 5) as u64)?;

    // HybridUint config (per-histogram optimized, or default {4,2,0})
    let config = code
        .uint_configs
        .first()
        .copied()
        .unwrap_or(crate::entropy_coding::hybrid_uint::HybridUintConfig::default_config());
    let se_bits = ceil_log2_nonzero(las as u32 + 1);
    writer.write(se_bits as usize, config.split_exponent as u64)?;
    if (config.split_exponent as usize) != las {
        let msb_bits = ceil_log2_nonzero(config.split_exponent + 1);
        writer.write(msb_bits as usize, config.msb_in_token as u64)?;
        let lsb_bits = ceil_log2_nonzero(config.split_exponent - config.msb_in_token + 1);
        writer.write(lsb_bits as usize, config.lsb_in_token as u64)?;
    }

    // Write the single ANS distribution
    code.histograms[0].write(writer)?;

    Ok(())
}

/// CeilLog2Nonzero matching the JXL spec. Returns number of bits needed to represent values 0..x.
fn ceil_log2_nonzero(x: u32) -> u32 {
    debug_assert!(x > 0);
    let floor = 31 - x.leading_zeros();
    if x.is_power_of_two() {
        floor
    } else {
        floor + 1
    }
}

/// Write ANS-encoded tokens.
pub(crate) fn write_ans_modular_tokens(
    writer: &mut BitWriter,
    tokens: &[AnsToken],
    code: &OwnedAnsEntropyCode,
) -> Result<()> {
    write_tokens_ans(tokens, code, None, writer)?;
    Ok(())
}

/// Collect residuals using gradient prediction and identify LZ77 runs.
fn collect_residuals_with_prediction(image: &ModularImage) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current_run = 0usize;
    let mut num_decoded = 0usize; // Track how many values we've output (for LZ77 validity)
    let mut last_value = 0u32; // Track last value output (for LZ77 copy)
    let mut debug_count = 0;

    for channel in &image.channels {
        // Flush any accumulated run at channel boundary
        // LZ77 should not span channel boundaries because each channel is decoded separately
        if current_run > K_LZ77_MIN_LENGTH {
            tokens.push(Token::Lz77Run(current_run));
            num_decoded += current_run;
        } else {
            for _ in 0..current_run {
                tokens.push(Token::Raw(last_value));
                num_decoded += 1;
            }
        }
        current_run = 0;
        // Reset last_value to an impossible value to prevent LZ77 from first pixel of new channel
        last_value = u32::MAX;

        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                // Get neighbors (matching JXL decoder and write_simple_modular_stream)
                let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                let top = if y > 0 { channel.get(x, y - 1) } else { left };
                let topleft = if x > 0 && y > 0 {
                    channel.get(x - 1, y - 1)
                } else {
                    left
                };

                // Predict using ClampedGradient (predictor 5)
                let prediction = predict_gradient(left, top, topleft);
                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                if debug_count < 20 {
                    let _channel_idx = image
                        .channels
                        .iter()
                        .position(|c| std::ptr::eq(c, channel))
                        .unwrap();
                    crate::trace::debug_eprintln!(
                        "RESIDUAL[{}]: ch={} y={} x={} pixel={}, pred={}, residual={}, packed={}",
                        debug_count,
                        _channel_idx,
                        y,
                        x,
                        pixel,
                        prediction,
                        residual,
                        packed
                    );
                    debug_count += 1;
                }

                // LZ77 with distance=1 copies the last value.
                // We can only use LZ77 if:
                // 1. We have at least one value in the window (num_decoded > 0)
                // 2. The value to repeat (packed) matches the last value
                let can_use_lz77 = num_decoded > 0 && packed == last_value;

                if can_use_lz77 {
                    current_run += 1;
                } else {
                    // Flush any accumulated run
                    if current_run > K_LZ77_MIN_LENGTH {
                        tokens.push(Token::Lz77Run(current_run));
                        num_decoded += current_run;
                        // Note: after LZ77 copy, last_value stays the same
                    } else {
                        // Output individual copies of last_value
                        for _ in 0..current_run {
                            tokens.push(Token::Raw(last_value));
                            num_decoded += 1;
                        }
                    }
                    current_run = 0;
                    tokens.push(Token::Raw(packed));
                    num_decoded += 1;
                    last_value = packed;
                }
            }
        }
    }

    // Flush final run
    if current_run > K_LZ77_MIN_LENGTH {
        tokens.push(Token::Lz77Run(current_run));
    } else {
        for _ in 0..current_run {
            tokens.push(Token::Raw(last_value));
        }
    }

    tokens
}

/// LZ77 min_symbol value - symbols >= this are LZ77 length tokens
pub(crate) const K_LZ77_MIN_SYMBOL: usize = 224;

/// Build a single sparse histogram for symbols [0..K_NUM_RAW_SYMBOLS) and [K_LZ77_MIN_SYMBOL..K_LZ77_MIN_SYMBOL+K_NUM_LZ77)
pub(crate) fn build_sparse_histogram(tokens: &[Token]) -> Vec<u64> {
    // Sparse alphabet: 19 raw symbols + 33 LZ77 symbols = 52 symbols
    // We'll encode raw [0..18] directly, LZ77 as [224..256]
    let total_symbols = K_LZ77_MIN_SYMBOL + K_NUM_LZ77;
    let mut counts = vec![0u64; total_symbols];

    for token in tokens {
        match token {
            Token::Raw(value) => {
                let (tok, _, _) = encode_hybrid_uint_000(*value);
                if (tok as usize) < total_symbols {
                    counts[tok as usize] += 1;
                }
            }
            Token::Lz77Run(count) => {
                // LZ77 encodes: length - min_length (not -1)
                let adjusted = count - K_LZ77_MIN_LENGTH;
                let (tok, _, _) = encode_hybrid_uint_lz77_length(adjusted as u32);
                let symbol = K_LZ77_MIN_SYMBOL + tok as usize;
                if symbol < total_symbols {
                    counts[symbol] += 1;
                }
                // Count distance symbol for distance=1
                // With dist_multiplier = image_width, SPECIAL_DISTANCES[1] = (1, 0) gives distance=1
                // Distance symbol 1 is encoded as HybridUint token 1 (no extra bits)
                let (dist_tok, _, _) = encode_hybrid_uint_000(1);
                counts[dist_tok as usize] += 1;
            }
        }
    }

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

/// Writes a U8 value to the bitstream (JXL U8 encoding).
///
/// U8 encoding:
/// - value=0: single bit 0
/// - value>=2: bit 1, then 3 bits for n, then (n+1) bits for val
///   where value = (1 << (n+1)) + val
///
/// NOTE: U8 encoding CANNOT represent value 1! For alphabet_size=2,
/// we must encode max_symbol=2 instead (giving alphabet_size=3).
/// Write VarLenUint16 encoding - matches libjxl's StoreVarLenUint16.
/// Used for alphabet_size-1 in prefix code histograms.
pub(crate) fn write_varlen_u16(writer: &mut BitWriter, value: u16) -> Result<()> {
    if value == 0 {
        writer.write(1, 0)?;
    } else {
        writer.write(1, 1)?;
        // nbits = floor(log2(value))
        let nbits = (16 - value.leading_zeros()) as usize - 1;
        writer.write(4, nbits as u64)?;
        writer.write(nbits, (value - (1u16 << nbits)) as u64)?;
    }
    Ok(())
}

/// Compute ceil(log2(n+1)) - number of bits needed to encode values 0..n
/// This matches jxl-oxide's add_log2_ceil function exactly.
#[inline]
fn add_log2_ceil(x: u32) -> u32 {
    if x >= 0x80000000 {
        32
    } else {
        (x + 1).next_power_of_two().trailing_zeros()
    }
}

/// Compute ceil(log2(n)) for alphabet size.
/// Returns 0 for n <= 1.
#[inline]
#[allow(dead_code)] // May be used in future for ANS encoding
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Write IntegerConfig to the bitstream.
///
/// The IntegerConfig encodes how hybrid uint values are encoded:
/// - split_exponent: values < 2^split_exponent are raw symbols
/// - msb_in_token/lsb_in_token: bits embedded in the token for values >= split
///
/// For raw symbols (no hybrid uint), use split_exponent = log_alphabet_size.
/// This makes split >= alphabet_size, so all symbols are raw.
///
/// For hybrid uint with config {0, 0, 0}, use split_exponent = 0.
pub(crate) fn write_integer_config(
    writer: &mut BitWriter,
    log_alphabet_size: u32,
    split_exponent: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
) -> Result<()> {
    // Number of bits to encode split_exponent
    let split_exponent_bits = add_log2_ceil(log_alphabet_size) as usize;
    writer.write(split_exponent_bits, split_exponent as u64)?;

    if split_exponent != log_alphabet_size {
        // Must write msb_in_token and lsb_in_token
        let msb_bits = add_log2_ceil(split_exponent) as usize;
        writer.write(msb_bits, msb_in_token as u64)?;
        let lsb_bits = add_log2_ceil(split_exponent.saturating_sub(msb_in_token)) as usize;
        writer.write(lsb_bits, lsb_in_token as u64)?;
    }
    // When split_exponent == log_alphabet_size, msb/lsb are implicitly 0

    Ok(())
}

/// Writes a varint16 value to the bitstream.
/// Note: For prefix code histograms, use write_varlen_u16 for alphabet_size-1.
#[allow(dead_code)] // May be used in future
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
pub(crate) fn write_sparse_lz77_histogram(
    writer: &mut BitWriter,
    sparse_counts: &[u64],
) -> Result<(Vec<u8>, Vec<u16>)> {
    crate::trace::debug_eprintln!("SPARSE_HIST: Writing LZ77-enabled histogram");

    // lz77.enabled = 1
    writer.write(1, 1)?;
    crate::trace::debug_eprintln!(
        "SPARSE_HIST [bit {}]: lz77.enabled = 1",
        writer.bits_written()
    );

    // lz77.min_symbol = 224 (u2S encoding)
    // u2S(224, Bits(8)+225, Bits(16)+481, Bits(32)+65537)
    // 224 = selector 0 means value IS 224
    writer.write(2, 0)?; // selector 0: value = 224
    crate::trace::debug_eprintln!(
        "SPARSE_HIST [bit {}]: min_symbol = 224",
        writer.bits_written()
    );

    // lz77.min_length = K_LZ77_MIN_LENGTH = 7
    // u2S(3, 4, Bits(2)+5, Bits(8)+9)
    // 7 = Bits(2)+5 with bits=2, so selector 2
    writer.write(2, 2)?; // selector 2
    writer.write(2, 2)?; // 7 - 5 = 2
    crate::trace::debug_eprintln!(
        "SPARSE_HIST [bit {}]: min_length = 7",
        writer.bits_written()
    );

    // length_uint_config: HybridUintConfig for LZ77 run lengths
    // We use {0, 0, 0} which matches libjxl's default.
    // log_alphabet_size for LZ77 length is 8 (per spec).
    // When split_exponent=0, msb_bits = ceil_log2(1) = 0 and lsb_bits = 0, so they're implicit.
    const LZ77_LENGTH_LOG_ALPHA: u32 = 8;
    write_integer_config(writer, LZ77_LENGTH_LOG_ALPHA, 0, 0, 0)?;
    crate::trace::debug_eprintln!(
        "SPARSE_HIST [bit {}]: length_uint_config = {{0, 0, 0}}",
        writer.bits_written()
    );

    // Context map: With LZ77 enabled, we have num_contexts = 2:
    //   - context 0: original token context
    //   - context 1: LZ77 distance context
    // Both map to histogram 0.
    // Format: is_simple=1, bits_per_entry=0 (all zeros)
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0 (all contexts map to 0)
    crate::trace::debug_eprintln!(
        "SPARSE_HIST [bit {}]: context_map (is_simple=1, bits=0)",
        writer.bits_written()
    );
    // distance_context = context_map[1] = 0

    // Find the actual used symbols
    let _max_raw_symbol = sparse_counts[..K_NUM_RAW_SYMBOLS]
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

    crate::trace::debug_eprintln!(
        "SPARSE_HIST: max_raw={}, max_lz77={} (count at lz77={})",
        _max_raw_symbol,
        max_lz77_symbol,
        sparse_counts.get(K_LZ77_MIN_SYMBOL).unwrap_or(&0)
    );

    // Build histogram for Huffman tree - only non-zero symbols
    // For sparse alphabets, we use the complex prefix code path
    let histogram: Vec<u32> = sparse_counts.iter().map(|&c| c as u32).collect();

    // Count actual used symbols
    let _num_used: usize = histogram.iter().filter(|&&c| c > 0).count();
    crate::trace::debug_eprintln!("SPARSE_HIST: {} used symbols", _num_used);

    // Use the Huffman tree builder to store the prefix code
    // First write use_prefix_code = 1
    writer.write(1, 1)?;
    crate::trace::debug_eprintln!(
        "SPARSE_HIST [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );

    // Alphabet size - for LZ77 histograms, this is max symbol + 1
    // But with sparse, we need to account for the full range up to max_lz77_symbol
    let alphabet_size = max_lz77_symbol + 1;

    // IntegerConfig: When use_prefix_code=1, the decoder uses log_alphabet_size=15
    // for parsing IntegerConfig, regardless of actual alphabet size.
    // uint_config for data tokens: {split_exponent=0, msb=0, lsb=0}
    // This MUST match encode_hybrid_uint_000 which uses config {0,0,0}
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(writer, LOG_ALPHABET_SIZE_PREFIX, 0, 0, 0)?;
    crate::trace::debug_eprintln!(
        "SPARSE_HIST [bit {}]: uint_config (log_alpha={}, split_exp=0)",
        writer.bits_written(),
        LOG_ALPHABET_SIZE_PREFIX
    );

    // alphabet_size - 1 using VarLenUint16 encoding (matches libjxl)
    write_varlen_u16(writer, (alphabet_size - 1) as u16)?;
    crate::trace::debug_eprintln!(
        "SPARSE_HIST [bit {}]: alphabet_size = {} (max_symbol={})",
        writer.bits_written(),
        alphabet_size,
        alphabet_size - 1
    );

    // Write Huffman table and get the depths/codes that were actually stored
    // IMPORTANT: We must use the codes returned by build_and_store_huffman_tree,
    // not compute them ourselves, because the Huffman encoder uses bit-reversed
    // canonical codes.
    let (depths, codes) = if alphabet_size > 1 {
        let table = build_and_store_huffman_tree(&histogram[..alphabet_size], writer)?;
        crate::trace::debug_eprintln!(
            "SPARSE_HIST [bit {}]: After Huffman table",
            writer.bits_written()
        );
        (table.depths, table.codes)
    } else {
        (vec![0u8; alphabet_size], vec![0u16; alphabet_size])
    };

    // Note: With context_map [0, 0], both token context and distance context
    // use histogram 0. We don't need a separate distance histogram - the
    // distance symbols (always 0 for our RLE with distance=1) are encoded
    // using the same histogram as regular tokens.

    Ok((depths, codes))
}

/// Writes an improved modular stream with gradient prediction and LZ77.
///
/// For VarDCT subbitstreams, set `skip_group_header = true` since the GroupHeader
/// is written separately before calling this function.
pub fn write_improved_modular_stream(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    write_improved_modular_stream_inner(image, writer, false, use_ans)
}

fn write_improved_modular_stream_inner(
    image: &ModularImage,
    writer: &mut BitWriter,
    _skip_group_header: bool,
    use_ans: bool,
) -> Result<()> {
    // Collect residuals with gradient prediction
    let tokens = collect_residuals_with_prediction(image);

    // Build sparse histogram [0..18] + [224..256]
    let sparse_counts = build_sparse_histogram(&tokens);

    let _num_raw_used = sparse_counts[..K_NUM_RAW_SYMBOLS]
        .iter()
        .filter(|&&c| c > 0)
        .count();
    let _num_lz77_used = sparse_counts[K_LZ77_MIN_SYMBOL..]
        .iter()
        .filter(|&&c| c > 0)
        .count();
    let num_lz77_runs = tokens
        .iter()
        .filter(|t| matches!(t, Token::Lz77Run(_)))
        .count();

    crate::trace::debug_eprintln!(
        "IMPROVED: {} tokens, {} raw symbols used, {} lz77 tokens used, {} lz77 runs",
        tokens.len(),
        _num_raw_used,
        _num_lz77_used,
        num_lz77_runs
    );

    // If no LZ77 runs, fall back to simple encoding
    if num_lz77_runs == 0 {
        return write_simple_modular_stream(image, writer, use_ans);
    }

    // === Global section (LfGlobal) ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    // Tree histogram for single-leaf tree with Gradient predictor
    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

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
                // Length encoding: symbol = min_symbol + HybridUint(length - min_length)
                let adjusted = count - K_LZ77_MIN_LENGTH;
                let (tok, nbits, extra) = encode_hybrid_uint_lz77_length(adjusted as u32);

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

                // Write distance symbol for distance=1 (RLE)
                // With dist_multiplier = image_width, distance formula is:
                //   distance = dist_multiplier * dist + offset
                // SPECIAL_DISTANCES[0] = (0, 1): distance = width * 1 + 0 = width
                // SPECIAL_DISTANCES[1] = (1, 0): distance = width * 0 + 1 = 1 ✓
                // So for distance=1, we need distance symbol 1, not 0!
                let dist_symbol = 1u32;
                let (dist_tok, dist_nbits, dist_extra) = encode_hybrid_uint_000(dist_symbol);
                let dist_depth = depths[dist_tok as usize];
                let dist_code = codes[dist_tok as usize];
                if dist_depth > 0 {
                    writer.write(dist_depth as usize, dist_code as u64)?;
                }
                if dist_nbits > 0 {
                    writer.write(dist_nbits as usize, dist_extra as u64)?;
                }
            }
        }
    }

    crate::trace::debug_eprintln!(
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
    write_tree_histogram_for_gradient_impl(writer, true)
}

/// Write tree histogram and return (depths, codes) for encoding tree tokens.
fn write_tree_histogram_for_gradient_impl(
    writer: &mut BitWriter,
    write_lz77: bool,
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
    const TREE_PREDICTOR: u32 = 5; // Must match write_gradient_tree_tokens (0=Zero, 5=Gradient)
    let max_symbol = if TREE_PREDICTOR == 0 { 0u16 } else { 5u16 };
    let tree_histogram: &[u32] = if TREE_PREDICTOR == 0 {
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

/// Write tree tokens for a single leaf with Gradient predictor.
/// Uses the provided (depths, codes) from write_tree_histogram_for_gradient.
pub(crate) fn write_gradient_tree_tokens(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
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

    // The predictor to use (0=Zero, 5=Gradient) - must match write_tree_histogram_for_gradient
    const TREE_PREDICTOR: u32 = 5;

    crate::trace::debug_eprintln!("  TREE_TOKENS: depths = {:?}", depths);
    crate::trace::debug_eprintln!("  TREE_TOKENS: codes = {:?}", codes);

    // Encode: property=0, predictor, offset=0, mul_log=0, mul_bits=0
    let tokens = [0u32, TREE_PREDICTOR, 0, 0, 0];
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

/// Simpler stream without LZ77 but with gradient prediction.
pub fn write_simple_modular_stream(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    // Collect residuals with gradient prediction
    let mut residuals = Vec::new();

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                let prediction = if USE_ZERO_PREDICTOR {
                    0
                } else {
                    let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                    let top = if y > 0 { channel.get(x, y - 1) } else { left };
                    let topleft = if x > 0 && y > 0 {
                        channel.get(x - 1, y - 1)
                    } else {
                        left
                    };
                    predict_gradient(left, top, topleft)
                };

                let residual = pixel - prediction;
                let packed = pack_signed(residual);
                residuals.push(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    if USE_ZERO_PREDICTOR {
        write_tree_histogram_for_zero(writer)?;
    } else {
        let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
        write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;
    }

    if use_ans {
        // ANS entropy coding path
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        // GroupHeader
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 0)?; // num_transforms = 0

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        // Huffman path with HybridUint {4,2,0}
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 0)?; // num_transforms = 0

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write the RCT transform descriptor to the bitstream.
///
/// Format (for YCoCg with begin_c=0):
/// - TransformId: 2 bits (selector 0 = RCT)
/// - begin_c: 2 bits selector + 3 bits value = 5 bits for value 0
/// - rct_type: 2 bits (selector 0 = 6 = YCoCg)
pub(crate) fn write_rct_transform(
    writer: &mut BitWriter,
    begin_c: usize,
    rct_type: RctType,
) -> Result<()> {
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

    // rct_type: U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10), 6)
    // Val(6) = YCoCg at selector 0
    // Bits(2) = 0-3 at selector 1
    // BitsOffset(4, 2) = 2-17 at selector 2
    // BitsOffset(6, 10) = 10-73 at selector 3
    let rct_val = rct_type.0 as u64;
    if rct_val == 6 {
        writer.write(2, 0)?; // selector 0 = Val(6)
    } else if rct_val < 2 {
        // 0-1 encoded as Bits(2) at selector 1, but 0 and 1 need 2 bits
        writer.write(2, 1)?;
        writer.write(2, rct_val)?;
    } else if rct_val < 10 {
        // 2-9 encoded as BitsOffset(4, 2) at selector 2
        writer.write(2, 2)?;
        writer.write(4, rct_val - 2)?;
    } else {
        // 10-73 encoded as BitsOffset(6, 10) at selector 3
        writer.write(2, 3)?;
        writer.write(6, rct_val - 10)?;
    }

    Ok(())
}

/// Write the Palette transform descriptor to the bitstream.
///
/// Format:
/// - TransformId: 2 bits (selector 1 = Palette)
/// - begin_c: U32(Bits(3), BitsOffset(6,8), BitsOffset(10,72), BitsOffset(13,1096))
/// - num_c: U32(Val(1), Val(3), Val(4), BitsOffset(13,1))
/// - nb_colors: U32(BitsOffset(8,0), BitsOffset(11,256), BitsOffset(14,1280), BitsOffset(16,5376))
/// - nb_deltas: U32(Val(0), BitsOffset(8,1), BitsOffset(10,257), BitsOffset(16,1281))
/// - predictor: 4 bits (0=Zero for lossless)
fn write_palette_transform(
    writer: &mut BitWriter,
    begin_c: usize,
    num_c: usize,
    nb_colors: usize,
) -> Result<()> {
    // TransformId: U32(Val(0)=RCT, Val(1)=Palette, Val(2)=Squeeze, Val(3)=Invalid)
    // Palette = selector 1 = 2 bits "01"
    writer.write(2, 1)?;

    // begin_c: U32(Bits(3), BitsOffset(6, 8), BitsOffset(10, 72), BitsOffset(13, 1096))
    if begin_c < 8 {
        writer.write(2, 0)?;
        writer.write(3, begin_c as u64)?;
    } else if begin_c < 72 {
        writer.write(2, 1)?;
        writer.write(6, (begin_c - 8) as u64)?;
    } else if begin_c < 1096 {
        writer.write(2, 2)?;
        writer.write(10, (begin_c - 72) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(13, (begin_c - 1096) as u64)?;
    }

    // num_c: U32(Val(1), Val(3), Val(4), BitsOffset(13, 1))
    match num_c {
        1 => writer.write(2, 0)?, // selector 0 = Val(1)
        3 => writer.write(2, 1)?, // selector 1 = Val(3)
        4 => writer.write(2, 2)?, // selector 2 = Val(4)
        _ => {
            writer.write(2, 3)?; // selector 3 = BitsOffset(13, 1)
            writer.write(13, (num_c - 1) as u64)?;
        }
    }

    // nb_colors: U32(BitsOffset(8,0), BitsOffset(11,256), BitsOffset(14,1280), BitsOffset(16,5376))
    if nb_colors < 256 {
        writer.write(2, 0)?;
        writer.write(8, nb_colors as u64)?;
    } else if nb_colors < 256 + 2048 {
        writer.write(2, 1)?;
        writer.write(11, (nb_colors - 256) as u64)?;
    } else if nb_colors < 1280 + 16384 {
        writer.write(2, 2)?;
        writer.write(14, (nb_colors - 1280) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(16, (nb_colors - 5376) as u64)?;
    }

    // nb_deltas: U32(Val(0), BitsOffset(8,1), BitsOffset(10,257), BitsOffset(16,1281))
    // For lossless: nb_deltas = 0 → selector 0
    writer.write(2, 0)?;

    // predictor: 4 bits (0 = Zero predictor for lossless with nb_deltas=0)
    writer.write(4, 0)?;

    Ok(())
}

/// Write the Squeeze transform descriptor to the bitstream.
///
/// Format:
/// - TransformId: 2 bits (selector 2 = Squeeze)
/// - num_squeezes: U32(Val(0), BitsOffset(4,1), BitsOffset(6,9), BitsOffset(8,41))
///   Val(0) = default squeeze (decoder computes parameters)
/// - For each squeeze: horizontal(1 bit), in_place(1 bit),
///   begin_c(U32), num_c(U32(Val(1),Val(2),Val(3),BitsOffset(4,4)))
pub(crate) fn write_squeeze_transform(
    writer: &mut BitWriter,
    params: &[super::squeeze::SqueezeParams],
) -> Result<()> {
    // TransformId: Val(2) = Squeeze = selector 2 = "10"
    writer.write(2, 2)?;

    if params.is_empty() {
        // num_squeezes = 0: use default squeeze
        writer.write(2, 0)?; // selector 0 = Val(0)
    } else {
        // Encode num_squeezes
        let n = params.len();
        if (1..=16).contains(&n) {
            writer.write(2, 1)?; // selector 1 = BitsOffset(4, 1)
            writer.write(4, (n - 1) as u64)?;
        } else if (9..=72).contains(&n) {
            writer.write(2, 2)?; // selector 2 = BitsOffset(6, 9)
            writer.write(6, (n - 9) as u64)?;
        } else {
            writer.write(2, 3)?; // selector 3 = BitsOffset(8, 41)
            writer.write(8, (n - 41) as u64)?;
        }

        // Write each squeeze parameter
        for sp in params {
            writer.write(1, sp.horizontal as u64)?;
            writer.write(1, sp.in_place as u64)?;

            // begin_c: U32(Bits(3), BitsOffset(6,8), BitsOffset(10,72), BitsOffset(13,1096))
            let bc = sp.begin_c as usize;
            if bc < 8 {
                writer.write(2, 0)?;
                writer.write(3, bc as u64)?;
            } else if bc < 72 {
                writer.write(2, 1)?;
                writer.write(6, (bc - 8) as u64)?;
            } else if bc < 1096 {
                writer.write(2, 2)?;
                writer.write(10, (bc - 72) as u64)?;
            } else {
                writer.write(2, 3)?;
                writer.write(13, (bc - 1096) as u64)?;
            }

            // num_c: U32(Val(1), Val(2), Val(3), BitsOffset(4, 4))
            match sp.num_c {
                1 => writer.write(2, 0)?,
                2 => writer.write(2, 1)?,
                3 => writer.write(2, 2)?,
                _ => {
                    writer.write(2, 3)?;
                    writer.write(4, (sp.num_c - 4) as u64)?;
                }
            }
        }
    }

    Ok(())
}

/// Write modular stream with palette transform for few-color images.
///
/// When an image has few unique colors (≤256 for 8-bit), palette encoding
/// replaces multi-channel data with a palette meta-channel + index channel.
/// This provides 19-57% compression improvement on graphics/screenshots.
pub fn write_modular_stream_with_palette(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    begin_c: usize,
    num_c: usize,
) -> Result<()> {
    use super::palette::{analyze_palette, apply_palette};

    let max_colors = 1usize << image.bit_depth.min(12);
    let analysis = analyze_palette(image, begin_c, num_c, max_colors);

    if !analysis.use_palette {
        // Fallback: use RCT if RGB, otherwise simple
        if image.channels.len() >= 3 {
            return write_modular_stream_with_rct(image, writer, use_ans);
        } else {
            return write_simple_modular_stream(image, writer, use_ans);
        }
    }

    // Apply palette transform
    let mut transformed = image.clone();
    let nb_colors = apply_palette(&mut transformed, begin_c, num_c, &analysis)?;

    crate::trace::debug_eprintln!(
        "PALETTE: {} unique colors, {} channels → palette({}) + index",
        analysis.num_colors,
        num_c,
        nb_colors
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

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        // GroupHeader with 1 transform (Palette)
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_palette_transform(writer, begin_c, num_c, nb_colors)?;

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader with 1 transform (Palette)
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_palette_transform(writer, begin_c, num_c, nb_colors)?;

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write modular stream with Squeeze (Haar wavelet) transform.
///
/// Decomposes channels into low-frequency (average) + high-frequency (residual)
/// pairs by halving resolution. Enables progressive decoding and improves
/// compression on smooth content.
pub fn write_modular_stream_with_squeeze(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    use super::squeeze::{apply_squeeze, default_squeeze_params};

    let params = default_squeeze_params(image);
    if params.is_empty() {
        // Image too small for squeeze, fall back
        if image.channels.len() >= 3 {
            return write_modular_stream_with_rct(image, writer, use_ans);
        } else {
            return write_simple_modular_stream(image, writer, use_ans);
        }
    }

    // Apply RCT (YCoCg) before squeeze for RGB images to decorrelate channels
    let mut transformed = image.clone();
    let has_rct = transformed.channels.len() >= 3;
    if has_rct {
        let rct_type = RctType::YCOCG;
        forward_rct(&mut transformed.channels, 0, rct_type)?;
    }

    // Apply forward squeeze
    apply_squeeze(&mut transformed, &params)?;

    crate::trace::debug_eprintln!(
        "SQUEEZE: {} steps, {} → {} channels, rct={}",
        params.len(),
        image.channels.len(),
        transformed.channels.len(),
        has_rct,
    );

    // Collect residuals with gradient prediction on all transformed channels
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

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        // GroupHeader with transforms: RCT (if RGB) + Squeeze
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        if has_rct {
            // num_transforms = 2: U32 BitsOffset(4,2), offset=0
            writer.write(2, 2)?;
            writer.write(4, 0)?;
            write_rct_transform(writer, 0, RctType::YCOCG)?;
            write_squeeze_transform(writer, &params)?;
        } else {
            writer.write(2, 1)?; // num_transforms = 1
            write_squeeze_transform(writer, &params)?;
        }

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader with transforms: RCT (if RGB) + Squeeze
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        if has_rct {
            // num_transforms = 2: U32 BitsOffset(4,2), offset=0
            writer.write(2, 2)?;
            writer.write(4, 0)?;
            write_rct_transform(writer, 0, RctType::YCOCG)?;
            write_squeeze_transform(writer, &params)?;
        } else {
            writer.write(2, 1)?; // num_transforms = 1
            write_squeeze_transform(writer, &params)?;
        }

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
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
pub fn write_modular_stream_with_rct(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    // Check if palette is more beneficial than RCT
    if let Some((begin_c, num_c)) = super::palette::should_use_palette(image) {
        return write_modular_stream_with_palette(image, writer, use_ans, begin_c, num_c);
    }

    // Only apply RCT to RGB images (3+ channels)
    if image.channels.len() < 3 {
        return write_simple_modular_stream(image, writer, use_ans);
    }

    // Clone the image and apply forward RCT (YCoCg)
    let mut transformed = image.clone();
    let rct_type = RctType::YCOCG;
    forward_rct(&mut transformed.channels, 0, rct_type)?;

    crate::trace::debug_eprintln!(
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

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        // GroupHeader with 1 transform
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader with 1 transform
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write a tree histogram that can encode tokens 0-6 (for Weighted predictor).
fn write_tree_histogram_for_weighted(writer: &mut BitWriter) -> Result<()> {
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
fn write_weighted_tree_tokens(writer: &mut BitWriter) -> Result<()> {
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
fn write_wp_header(
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

/// Write modular stream using the Weighted predictor for better compression.
///
/// The Weighted predictor adapts to local image statistics and typically
/// achieves better compression than the Gradient predictor for natural images.
pub fn write_modular_stream_with_weighted(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    use super::predictor::{Neighbors, WeightedPredictorParams, WeightedPredictorState};

    let params = WeightedPredictorParams::default();

    // Collect residuals with weighted prediction
    let mut residuals = Vec::new();

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();
        let mut wp_state = WeightedPredictorState::new(&params, width);

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let neighbors = Neighbors::gather(channel, x, y);
                let prediction = wp_state.predict(x, y, width, &neighbors);

                let residual = pixel - prediction;
                let packed = pack_signed(residual);
                residuals.push(packed);

                wp_state.update_errors(pixel, x, y, width);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    write_tree_histogram_for_weighted(writer)?;
    write_weighted_tree_tokens(writer)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        writer.write(1, 1)?; // use_global_tree = true
        write_wp_header(writer, &params)?;
        writer.write(2, 0)?; // num_transforms = 0

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        writer.write(1, 1)?; // use_global_tree = true
        write_wp_header(writer, &params)?;
        writer.write(2, 0)?; // num_transforms = 0

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write modular stream with RCT and Weighted predictor for best compression.
///
/// Combines YCoCg color transform with adaptive weighted prediction.
pub fn write_modular_stream_with_rct_weighted(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    use super::predictor::{Neighbors, WeightedPredictorParams, WeightedPredictorState};

    // Check if palette is more beneficial
    if let Some((begin_c, num_c)) = super::palette::should_use_palette(image) {
        return write_modular_stream_with_palette(image, writer, use_ans, begin_c, num_c);
    }

    if image.channels.len() < 3 {
        return write_modular_stream_with_weighted(image, writer, use_ans);
    }

    let mut transformed = image.clone();
    let rct_type = RctType::YCOCG;
    forward_rct(&mut transformed.channels, 0, rct_type)?;

    let params = WeightedPredictorParams::default();

    // Collect residuals with weighted prediction on transformed channels
    let mut residuals = Vec::new();

    for channel in &transformed.channels {
        let width = channel.width();
        let height = channel.height();
        let mut wp_state = WeightedPredictorState::new(&params, width);

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let neighbors = Neighbors::gather(channel, x, y);
                let prediction = wp_state.predict(x, y, width, &neighbors);

                let residual = pixel - prediction;
                residuals.push(pack_signed(residual));

                wp_state.update_errors(pixel, x, y, width);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    write_tree_histogram_for_weighted(writer)?;
    write_weighted_tree_tokens(writer)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        writer.write(1, 1)?; // use_global_tree = true
        write_wp_header(writer, &params)?;
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        writer.write(1, 1)?; // use_global_tree = true
        write_wp_header(writer, &params)?;
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

// ===== Tree-learned modular encoding =====

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

    let tokens = collect_tree_tokens(tree);

    // Build histogram of all token values across all contexts.
    // Tree tokens use raw symbol encoding (no HybridUint).
    let mut max_symbol: u32 = 0;
    for t in &tokens {
        let val = if t.is_signed {
            pack_signed(t.value)
        } else {
            t.value as u32
        };
        max_symbol = max_symbol.max(val);
    }

    let histogram_size = (max_symbol + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for t in &tokens {
        let val = if t.is_signed {
            pack_signed(t.value)
        } else {
            t.value as u32
        };
        histogram[val as usize] += 1;
    }

    // lz77.enabled = 0
    writer.write(1, 0)?;

    // Context map: 6 contexts all map to histogram 0
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0

    // use_prefix_code = 1
    writer.write(1, 1)?;

    // IntegerConfig: raw symbols (split_exponent = log_alphabet_size = 15)
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;

    // alphabet_size - 1
    write_varlen_u16(writer, max_symbol as u16)?;

    // Huffman table
    let (depths, codes) = if histogram_size > 1 {
        let table = build_and_store_huffman_tree(&histogram[..histogram_size], writer)?;
        (table.depths, table.codes)
    } else {
        (vec![0u8; histogram_size], vec![0u16; histogram_size])
    };

    // Write tokens
    for t in &tokens {
        let val = if t.is_signed {
            pack_signed(t.value)
        } else {
            t.value as u32
        };
        let depth = depths.get(val as usize).copied().unwrap_or(0);
        let code = codes.get(val as usize).copied().unwrap_or(0);
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
    }

    Ok(())
}

/// Fast cost estimate for an image using gradient prediction.
///
/// Matches libjxl's `EstimateCost()` in enc_modular_simd.cc.
/// Uses clamped gradient prediction with residuals bucketed by local complexity
/// (max neighbor difference). Returns total estimated bits.
fn estimate_cost(image: &ModularImage) -> f64 {
    use super::predictor::pack_signed;
    use crate::entropy_coding::hybrid_uint::HybridUintConfig;

    let config = HybridUintConfig::new(4, 2, 0);
    let cutoffs: &[u32] = &[
        0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
    ];
    let nc = cutoffs.len() + 1; // 18 context buckets

    let mut total_bits: f64 = 0.0;
    let mut extra_bits: u64 = 0;
    let mut histograms: Vec<Vec<u32>> = vec![vec![]; nc];

    for ch in &image.channels {
        let w = ch.width();
        let h = ch.height();
        if w == 0 || h == 0 {
            continue;
        }

        for y in 0..h {
            for x in 0..w {
                let val = ch.data()[y * w + x];
                let left = if x > 0 {
                    ch.data()[y * w + x - 1]
                } else if y > 0 {
                    ch.data()[(y - 1) * w + x]
                } else {
                    0
                };
                let top = if y > 0 {
                    ch.data()[(y - 1) * w + x]
                } else {
                    left
                };
                let topleft = if x > 0 && y > 0 {
                    ch.data()[(y - 1) * w + x - 1]
                } else {
                    left
                };

                let max_diff = left.max(top).max(topleft) - left.min(top).min(topleft);
                let max_diff = max_diff as u32;

                // Find context bucket (count how many cutoffs are > max_diff)
                let mut ctx = 0usize;
                for &c in cutoffs {
                    if max_diff < c {
                        ctx += 1;
                    }
                }

                // Gradient prediction residual
                let grad = left + top - topleft;
                let pred = grad.max(left.min(top)).min(left.max(top)); // clamped gradient
                let res = val - pred;
                let packed = pack_signed(res);

                let (token, _bits, nbits) = config.encode(packed);
                if histograms[ctx].len() <= token as usize {
                    histograms[ctx].resize(token as usize + 1, 0);
                }
                histograms[ctx][token as usize] += 1;
                extra_bits += nbits as u64;
            }
        }

        // Sum Shannon entropy per context bucket, then reset
        for hist in &mut histograms {
            let total: u32 = hist.iter().sum();
            if total > 0 {
                let total_f = total as f64;
                for &count in hist.iter() {
                    if count > 0 {
                        let p = count as f64 / total_f;
                        total_bits -= count as f64 * p.log2();
                    }
                }
            }
            hist.clear();
        }
    }

    total_bits + extra_bits as f64
}

/// RCT variants to try at each effort level, matching libjxl's enc_modular.cc.
/// The order is: identity, YCoCg, YCbCr-like, GBR+SubGR, RBG+YCoCg, BGR+YCoCg, GBR+YCoCg.
#[allow(clippy::identity_op, clippy::erasing_op)]
const RCT_CANDIDATES: &[u8] = &[
    0 * 7 + 0, // identity (no transform)
    0 * 7 + 6, // YCoCg
    0 * 7 + 5, // type 5
    1 * 7 + 3, // GBR + SubGR
    3 * 7 + 5, // RBG + type 5
    5 * 7 + 5, // BGR + type 5
    1 * 7 + 5, // GBR + type 5
];

/// Select the best RCT variant by trying candidates and picking the lowest cost.
///
/// Returns the RctType and the transformed image. At effort 7, tries 7 RCT variants
/// matching libjxl's kSquirrel behavior.
pub(crate) fn select_best_rct(image: &ModularImage, effort: u8) -> (RctType, ModularImage) {
    use super::rct::{RctType, forward_rct};

    let nb_rcts_to_try = match effort {
        0..=4 => 0, // No RCT selection below Hare
        5 => 4,     // Hare
        6 => 5,     // Wombat
        7 => 7,     // Squirrel
        8 => 9,     // Kitten
        _ => 19,    // Tortoise/Glacier
    };

    if nb_rcts_to_try == 0 || image.channels.len() < 3 {
        // Default to YCoCg
        let mut transformed = image.clone();
        forward_rct(&mut transformed.channels, 0, RctType::YCOCG).ok();
        return (RctType::YCOCG, transformed);
    }

    let mut best_cost = f64::MAX;
    let mut best_rct = RctType::YCOCG;
    let mut best_image = None;

    for (i, &rct_val) in RCT_CANDIDATES.iter().enumerate() {
        if i >= nb_rcts_to_try {
            break;
        }
        let rct_type = RctType(rct_val);

        if rct_type.is_noop() {
            // Identity: estimate cost of the original image
            let cost = estimate_cost(image);
            crate::trace::debug_eprintln!("  RCT {:2}: cost={:.0}", rct_val, cost);
            if cost < best_cost {
                best_cost = cost;
                best_rct = rct_type;
                best_image = Some(image.clone());
            }
        } else {
            let mut transformed = image.clone();
            if forward_rct(&mut transformed.channels, 0, rct_type).is_ok() {
                let cost = estimate_cost(&transformed);
                crate::trace::debug_eprintln!("  RCT {:2}: cost={:.0}", rct_val, cost);
                if cost < best_cost {
                    best_cost = cost;
                    best_rct = rct_type;
                    best_image = Some(transformed);
                }
            }
        }
    }

    let work_image = best_image.unwrap_or_else(|| {
        let mut t = image.clone();
        forward_rct(&mut t.channels, 0, RctType::YCOCG).ok();
        t
    });

    crate::trace::debug_eprintln!(
        "RCT_SELECT: best={} (cost={:.0}), tried {} variants",
        best_rct.0,
        best_cost,
        nb_rcts_to_try.min(RCT_CANDIDATES.len()),
    );

    (best_rct, work_image)
}

/// Write a modular stream using a learned MA tree with multi-context ANS.
///
/// This is the single-group version. For multi-group, see section.rs.
///
/// Layout:
/// - dc_quant.all_default = 1
/// - has_tree = 1
/// - Tree (write_tree)
/// - lz77.enabled = 0 for data
/// - Multi-context ANS histogram (write_entropy_code_ans)
/// - GroupHeader (use_global_tree=1, wp_header.all_default=1, num_transforms=0 or 1)
/// - ANS-encoded residuals (write_tokens_ans)
/// - byte padding
pub fn write_modular_stream_with_tree(
    image: &ModularImage,
    writer: &mut BitWriter,
    effort: u8,
    rct: bool,
) -> Result<()> {
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_gather_stride, gather_samples_strided,
    };
    use crate::entropy_coding::encode::{build_entropy_code_ans, write_entropy_code_ans};

    // Select best RCT variant by trying multiple candidates (effort-dependent).
    // Tree learning uses RCT only (not palette) because palette+tree has a
    // meta-channel encoding mismatch that causes decoder failures.
    let (work_image, rct_type) = if rct && image.channels.len() >= 3 {
        let (selected_rct, transformed) = select_best_rct(image, effort);
        (transformed, Some(selected_rct))
    } else {
        (image.clone(), None)
    };

    // Step 1: Gather samples (with subsampling for large images)
    let total_pixels: usize = work_image
        .channels
        .iter()
        .map(|ch| ch.width() * ch.height())
        .sum();
    let stride = compute_gather_stride(total_pixels, effort);
    let mut samples = TreeSamples::new();
    gather_samples_strided(&mut samples, &work_image, 0, 0, stride);

    // Step 2: Learn tree with effort-dependent parameters
    let pixel_fraction = if total_pixels > 0 {
        samples.num_samples as f64 / total_pixels as f64
    } else {
        1.0
    };
    let params = TreeLearningParams::for_effort(effort).with_pixel_fraction(pixel_fraction);
    let tree = compute_best_tree(&mut samples, &params);
    let num_contexts = count_contexts(&tree) as usize;

    crate::trace::debug_eprintln!(
        "TREE_LEARN: effort={}, {} props, {} max_buckets, threshold={:.0}*{:.3}={:.1}, \
         {} nodes, {} leaves/contexts, {} samples",
        effort,
        params.properties.len(),
        params.max_property_values,
        params.split_threshold,
        params.pixel_fraction * 0.9 + 0.1,
        params.split_threshold * (params.pixel_fraction * 0.9 + 0.1),
        tree.len(),
        num_contexts,
        samples.num_samples
    );

    // Step 3: Collect residuals with learned tree
    let tokens = collect_residuals_with_tree(&work_image, &tree, 0);

    // Step 4: Build multi-context ANS code
    let code = build_entropy_code_ans(&tokens, num_contexts);

    // Step 5: Write bitstream
    // dc_quant.all_default = true
    writer.write(1, 1)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Write the learned tree
    write_tree(writer, &tree)?;

    // Write ANS data histogram.
    // JXL spec: context map is only written when num_contexts > 1.
    // write_entropy_code_ans doesn't write lz77 (caller handles it).
    // write_ans_modular_header includes lz77.enabled=0 itself.
    if num_contexts > 1 {
        writer.write(1, 0)?; // lz77.enabled = 0
        write_entropy_code_ans(&code, writer)?;
    } else {
        // write_ans_modular_header writes lz77.enabled=0 + omits context map
        use super::section::write_ans_modular_header;
        write_ans_modular_header(writer, &code)?;
    }

    // GroupHeader
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true

    if let Some(rct_type) = rct_type {
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;
    } else {
        writer.write(2, 0)?; // num_transforms = 0
    }

    // Debug: verify ANS encoding correctness
    #[cfg(debug_assertions)]
    {
        let roundtrip_result = crate::entropy_coding::encode::verify_ans_roundtrip(&tokens, &code);
        if roundtrip_result.is_err() {
            eprintln!(
                "ANS ROUNDTRIP VERIFICATION FAILED for tree learning data (num_contexts={}, num_histograms={}, tokens={}): {:?}",
                num_contexts,
                code.histograms.len(),
                tokens.len(),
                roundtrip_result
            );
        }
    }

    // Write ANS tokens
    write_tokens_ans(&tokens, &code, None, writer)?;

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write a single-group modular stream using squeeze + tree learning.
///
/// Combines the Haar wavelet (squeeze) transform with learned MA tree
/// for multi-context ANS encoding. This gives the benefits of both:
/// - Squeeze decorrelates spatial frequencies (better for smooth gradients)
/// - Tree learning adapts prediction and contexts per-channel/per-region
///
/// Pipeline: RCT → squeeze → gather samples → learn tree → collect residuals → ANS
pub fn write_modular_stream_with_squeeze_and_tree(
    image: &ModularImage,
    writer: &mut BitWriter,
    effort: u8,
) -> Result<()> {
    use super::rct::{RctType, forward_rct};
    use super::squeeze::{apply_squeeze, default_squeeze_params};
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_gather_stride, gather_samples_strided,
    };
    use crate::entropy_coding::encode::{build_entropy_code_ans, write_entropy_code_ans};

    let params = default_squeeze_params(image);
    if params.is_empty() {
        // Image too small for squeeze, fall back to tree learning without squeeze
        return write_modular_stream_with_tree(image, writer, effort, image.channels.len() >= 3);
    }

    // Step 1: Apply RCT (YCoCg) before squeeze for RGB images
    let mut transformed = image.clone();
    let has_rct = transformed.channels.len() >= 3;
    if has_rct {
        let rct_type = RctType::YCOCG;
        forward_rct(&mut transformed.channels, 0, rct_type)?;
    }

    // Step 2: Apply forward squeeze
    apply_squeeze(&mut transformed, &params)?;

    crate::trace::debug_eprintln!(
        "SQUEEZE+TREE: {} squeeze steps, {} → {} channels, rct={}",
        params.len(),
        image.channels.len(),
        transformed.channels.len(),
        has_rct,
    );

    // Step 3: Gather samples from squeezed image (with subsampling for large images)
    let total_pixels: usize = transformed
        .channels
        .iter()
        .map(|ch| ch.width() * ch.height())
        .sum();
    let stride = compute_gather_stride(total_pixels, effort);
    let mut samples = TreeSamples::new();
    gather_samples_strided(&mut samples, &transformed, 0, 0, stride);

    // Step 4: Learn tree with effort-dependent parameters
    let pixel_fraction = if total_pixels > 0 {
        samples.num_samples as f64 / total_pixels as f64
    } else {
        1.0
    };
    let tree_params = TreeLearningParams::for_effort(effort).with_pixel_fraction(pixel_fraction);
    let tree = compute_best_tree(&mut samples, &tree_params);
    let num_contexts = count_contexts(&tree) as usize;

    crate::trace::debug_eprintln!(
        "SQUEEZE+TREE: effort={}, {} nodes, {} contexts, {} samples (pf={:.3})",
        effort,
        tree.len(),
        num_contexts,
        samples.num_samples,
        pixel_fraction,
    );

    // Step 5: Collect residuals with learned tree
    let tokens = collect_residuals_with_tree(&transformed, &tree, 0);

    // Step 6: Build multi-context ANS code
    let code = build_entropy_code_ans(&tokens, num_contexts);

    // Step 7: Write bitstream
    // dc_quant.all_default = true
    writer.write(1, 1)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Write the learned tree
    write_tree(writer, &tree)?;

    // Write ANS data histogram
    if num_contexts > 1 {
        writer.write(1, 0)?; // lz77.enabled = 0
        write_entropy_code_ans(&code, writer)?;
    } else {
        use super::section::write_ans_modular_header;
        write_ans_modular_header(writer, &code)?;
    }

    // GroupHeader with transforms: RCT (if RGB) + Squeeze
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true

    if has_rct {
        // num_transforms = 2: U32 BitsOffset(4,2), offset=0
        writer.write(2, 2)?;
        writer.write(4, 0)?;
        write_rct_transform(writer, 0, RctType::YCOCG)?;
        write_squeeze_transform(writer, &params)?;
    } else {
        writer.write(2, 1)?; // num_transforms = 1
        write_squeeze_transform(writer, &params)?;
    }

    // Debug: verify ANS encoding correctness
    #[cfg(debug_assertions)]
    {
        let roundtrip_result = crate::entropy_coding::encode::verify_ans_roundtrip(&tokens, &code);
        if roundtrip_result.is_err() {
            eprintln!(
                "ANS ROUNDTRIP VERIFICATION FAILED for squeeze+tree data (num_contexts={}, num_histograms={}, tokens={}): {:?}",
                num_contexts,
                code.histograms.len(),
                tokens.len(),
                roundtrip_result
            );
        }
    }

    // Write ANS tokens
    write_tokens_ans(&tokens, &code, None, writer)?;

    writer.zero_pad_to_byte();
    Ok(())
}

// ===== Multi-group support =====
// These functions are now in the section module for better organization

pub use super::section::{
    build_histogram_from_residuals, collect_all_residuals, write_global_modular_section,
    write_group_modular_section,
};

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
    fn test_predict_gradient() {
        // Smooth gradient: should predict correctly
        assert_eq!(predict_gradient(10, 10, 10), 10);

        // Gradient clamping
        // predict_gradient(left, top, topleft) = clamp(left + top - topleft, min, max)
        // where min = min(left, top), max = max(left, top)
        assert_eq!(predict_gradient(20, 10, 10), 20); // grad = 20, clamped to [10,20]
        assert_eq!(predict_gradient(10, 20, 30), 10); // grad = 0, topleft > max, return min
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
        write_simple_modular_stream(&image, &mut writer, false).unwrap();

        let _bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("Gradient stream: {} bytes", _bytes.len());
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
        write_modular_stream_with_rct(&image, &mut writer, false).unwrap();

        let bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("RCT stream: {} bytes", bytes.len());
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

    #[test]
    fn test_rct_type_u32_encoding() {
        use crate::modular::rct::RctType;

        // Verify that write_rct_transform produces correct bit lengths
        // for all rct_type values. The U32 distribution is:
        // U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10))
        // TransformId (2 bits) + begin_c (5 bits for 0) + rct_type
        let base_bits = 2 + 5; // TransformId + begin_c=0

        for rct_val in 0..42u8 {
            let rct_type = RctType(rct_val);
            let mut writer = BitWriter::new();
            write_rct_transform(&mut writer, 0, rct_type).unwrap();

            let expected_rct_bits = if rct_val == 6 {
                2 // selector only (Val(6))
            } else if rct_val < 2 {
                2 + 2 // selector + Bits(2)
            } else if rct_val < 10 {
                2 + 4 // selector + BitsOffset(4, 2)
            } else {
                2 + 6 // selector + BitsOffset(6, 10)
            };
            let total_expected = base_bits + expected_rct_bits;

            assert_eq!(
                writer.bits_written(),
                total_expected,
                "Wrong bit count for rct_type={}: expected {} bits, got {}",
                rct_val,
                total_expected,
                writer.bits_written()
            );
        }
    }

    #[test]
    fn test_weighted_stream() {
        // 4x4 grayscale image with gradient
        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_weighted(&image, &mut writer, false).unwrap();

        let bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("Weighted stream: {} bytes", bytes.len());
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_rct_weighted_stream() {
        // 4x4 RGB image with smooth gradients
        let mut data = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let base = (y * 4 + x) * 10;
                data.push(base as u8);
                data.push((base + 5) as u8);
                data.push((base + 10) as u8);
            }
        }
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_rct_weighted(&image, &mut writer, false).unwrap();

        let bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("RCT+Weighted stream: {} bytes", bytes.len());
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_lz77_bit_trace() {
        // Create a 16x16 image that will trigger LZ77
        // Wider rows = runs of 15 zeros per row (>7, so LZ77 triggers)
        let mut data = Vec::new();
        for _ in 0..256 {
            // 16x16 image
            data.push(100u8);
            data.push(100u8);
            data.push(100u8);
        }
        let image = ModularImage::from_rgb8(&data, 16, 16).unwrap();

        crate::trace::debug_eprintln!("\n=== LZ77 BIT TRACE TEST ===");

        let mut writer = BitWriter::new();
        write_improved_modular_stream(&image, &mut writer, false).unwrap();

        let _bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("LZ77 stream: {} bytes", _bytes.len());
        crate::trace::debug_eprintln!("Raw bytes: {:02x?}", &_bytes[.._bytes.len().min(50)]);

        // Now let's trace through what the decoder expects:
        crate::trace::debug_eprintln!("\n=== EXPECTED DECODER INTERPRETATION ===");
        crate::trace::debug_eprintln!("Bit 0: dc_quant.all_default = 1");
        crate::trace::debug_eprintln!("Bit 1: has_tree = 1");
        crate::trace::debug_eprintln!("--- TREE HISTOGRAM (6 contexts) ---");
        crate::trace::debug_eprintln!("Bit 2: lz77.enabled = 0");
        crate::trace::debug_eprintln!("Bits 3-5: context_map (is_simple=1, bits_per_entry=0)");
        // ... etc
    }

    #[test]
    fn test_ans_roundtrip_gray() {
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        // Build full JXL bitstream with ANS modular
        let mut writer = BitWriter::new();
        let file_header = FileHeader::new_gray(4, 4);
        file_header.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        let frame_options = FrameEncoderOptions {
            use_modular: true,
            effort: 7,
            use_ans: true,
            use_tree_learning: false,
            use_squeeze: false,
            ..Default::default()
        };
        let frame_encoder = FrameEncoder::new(4, 4, frame_options);
        let color_encoding = ColorEncoding::srgb();
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("ANS modular gray 4x4: {} bytes", bytes.len());

        // Decode with jxl-oxide
        let jxl_image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

        assert_eq!(jxl_image.width(), 4);
        assert_eq!(jxl_image.height(), 4);

        let render = jxl_image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        assert_eq!(
            decoded.len(),
            data.len(),
            "decoded size mismatch: {} vs {}",
            decoded.len(),
            data.len()
        );

        for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig, dec,
                "pixel {} differs: orig={} decoded={}",
                i, orig, dec
            );
        }
    }

    #[test]
    fn test_ans_roundtrip_gray_varied() {
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        let data = vec![0u8, 64, 128, 192, 255, 100, 50, 200];
        let image = ModularImage::from_gray8(&data, 4, 2).unwrap();

        // First write with Huffman to get reference bytes
        {
            let mut writer = BitWriter::new();
            let file_header = FileHeader::new_gray(4, 2);
            file_header.write(&mut writer).unwrap();
            writer.zero_pad_to_byte();

            let frame_options = FrameEncoderOptions {
                use_modular: true,
                effort: 7,
                use_ans: false,
                use_tree_learning: false,
                use_squeeze: false,
                ..Default::default()
            };
            let frame_encoder = FrameEncoder::new(4, 2, frame_options);
            let color_encoding = ColorEncoding::srgb();
            frame_encoder
                .encode_modular(&image, &color_encoding, &mut writer)
                .unwrap();
            let huf_bytes = writer.finish_with_padding();
            eprintln!("Huffman modular gray varied 4x2: {} bytes", huf_bytes.len());
            eprintln!("Huffman bytes: {:02x?}", &huf_bytes);
        }

        // Now write with ANS
        let mut writer = BitWriter::new();
        let file_header = FileHeader::new_gray(4, 2);
        file_header.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        let frame_options = FrameEncoderOptions {
            use_modular: true,
            effort: 7,
            use_ans: true,
            use_tree_learning: false,
            use_squeeze: false,
            ..Default::default()
        };
        let frame_encoder = FrameEncoder::new(4, 2, frame_options);
        let color_encoding = ColorEncoding::srgb();
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("ANS modular gray varied 4x2: {} bytes", bytes.len());
        eprintln!("ANS bytes: {:02x?}", &bytes);

        // Save for external debugging
        std::fs::write(std::env::temp_dir().join("ans_modular_varied.jxl"), &bytes).ok();

        let jxl_image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

        let render = jxl_image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig, dec,
                "pixel {} differs: orig={} decoded={}",
                i, orig, dec
            );
        }
    }

    #[test]
    fn test_ans_roundtrip_rgb_gradient() {
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                data[idx] = (x * 32) as u8;
                data[idx + 1] = (y * 32) as u8;
                data[idx + 2] = ((x + y) * 16) as u8;
            }
        }
        let image = ModularImage::from_rgb8(&data, 8, 8).unwrap();

        let mut writer = BitWriter::new();
        let file_header = FileHeader::new_rgb(8, 8);
        file_header.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        let frame_options = FrameEncoderOptions {
            use_modular: true,
            effort: 7,
            use_ans: true,
            use_tree_learning: false,
            use_squeeze: false,
            ..Default::default()
        };
        let frame_encoder = FrameEncoder::new(8, 8, frame_options);
        let color_encoding = ColorEncoding::srgb();
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("ANS modular RGB gradient 8x8: {} bytes", bytes.len());

        let jxl_image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

        let render = jxl_image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        assert_eq!(decoded.len(), data.len());
        let mut max_diff = 0i32;
        for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
            let diff = (orig as i32 - dec as i32).abs();
            if diff > max_diff {
                max_diff = diff;
                eprintln!(
                    "pixel {} ch {}: orig={} decoded={} diff={}",
                    i / 3,
                    i % 3,
                    orig,
                    dec,
                    diff
                );
            }
        }
        assert_eq!(max_diff, 0, "lossless roundtrip should have zero diff");
    }

    #[test]
    fn test_ans_simple_stream() {
        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_simple_modular_stream(&image, &mut writer, true).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(
            !bytes.is_empty(),
            "ANS stream should produce non-empty output"
        );
    }

    #[test]
    fn test_ans_rct_stream() {
        let mut data = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let base = (y * 4 + x) * 10;
                data.push(base as u8);
                data.push((base + 5) as u8);
                data.push((base + 10) as u8);
            }
        }
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_rct(&image, &mut writer, true).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(
            !bytes.is_empty(),
            "ANS RCT stream should produce non-empty output"
        );
    }

    #[test]
    fn test_ans_weighted_stream() {
        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_weighted(&image, &mut writer, true).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(
            !bytes.is_empty(),
            "ANS weighted stream should produce non-empty output"
        );
    }

    #[test]
    fn test_ans_rct_weighted_stream() {
        let mut data = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let base = (y * 4 + x) * 10;
                data.push(base as u8);
                data.push((base + 5) as u8);
                data.push((base + 10) as u8);
            }
        }
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_rct_weighted(&image, &mut writer, true).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(
            !bytes.is_empty(),
            "ANS RCT+weighted stream should produce non-empty output"
        );
    }

    /// Compare ANS vs Huffman file sizes for a lossless encode.
    #[test]
    fn test_ans_vs_huffman_size() {
        use crate::{LosslessConfig, PixelLayout};

        // Create a non-trivial 32x32 RGB image
        let mut data = vec![0u8; 32 * 32 * 3];
        for y in 0..32 {
            for x in 0..32 {
                let idx = (y * 32 + x) * 3;
                data[idx] = ((x * 8 + y * 2) % 256) as u8;
                data[idx + 1] = ((y * 8 + x * 3) % 256) as u8;
                data[idx + 2] = (((x + y) * 5) % 256) as u8;
            }
        }

        // Encode with Huffman
        let huf_encoded = LosslessConfig::new()
            .with_ans(false)
            .encode(&data, 32, 32, PixelLayout::Rgb8)
            .unwrap();

        // Encode with ANS
        let ans_encoded = LosslessConfig::new()
            .with_ans(true)
            .encode(&data, 32, 32, PixelLayout::Rgb8)
            .unwrap();

        eprintln!(
            "32x32 RGB: Huffman={} bytes, ANS={} bytes, savings={:.1}%",
            huf_encoded.len(),
            ans_encoded.len(),
            (1.0 - ans_encoded.len() as f64 / huf_encoded.len() as f64) * 100.0
        );

        // ANS should not be significantly larger than Huffman
        // (for small images the overhead can make ANS larger, but it should be close)
        assert!(
            ans_encoded.len() <= huf_encoded.len() + huf_encoded.len() / 5,
            "ANS should not be >20% larger than Huffman"
        );
    }
}
