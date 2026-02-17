// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Low-level modular encoding utilities.
//!
//! Contains HybridUint encoding, histogram building, LZ77 histogram writing,
//! bitstream config writing, and ANS modular helpers.

#![allow(dead_code)]

use crate::bit_writer::BitWriter;
use crate::entropy_coding::encode::{
    OwnedAnsEntropyCode, build_entropy_code_ans, write_tokens_ans,
};
use crate::entropy_coding::huffman_tree::{build_and_store_huffman_tree, create_huffman_tree};
use crate::entropy_coding::hybrid_uint::HybridUintConfig;
use crate::entropy_coding::token::Token as AnsToken;
use crate::error::Result;

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
pub(super) struct EncodedResidual {
    token: u32,
    extra_bits: u32,
    num_extra: u32,
}

/// Encode a list of packed residuals through HybridUint, returning encoded tokens
/// and the maximum token value (for histogram sizing).
pub(super) fn encode_residuals_hybrid(residuals: &[u32]) -> (Vec<EncodedResidual>, u32) {
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
pub(super) fn build_token_histogram(encoded: &[EncodedResidual], max_token: u32) -> Vec<u32> {
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
pub(super) fn write_hybrid_residuals(
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
