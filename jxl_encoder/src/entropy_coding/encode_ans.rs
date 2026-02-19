// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! ANS (Asymmetric Numeral Systems) entropy code building and serialization.
//!
//! Contains all ANS-specific code: distribution building, header writing,
//! token writing, and verification/roundtrip utilities.

#![allow(dead_code)]

use super::ans::{ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution, AnsEncoder};
use super::context_map::move_to_front_transform;
use super::encode::{
    ALPHABET_SIZE, PrefixCode, encode_token_value, encode_token_value_with_config,
    write_var_len_uint16,
};
use super::encode_huffman::{
    convert_bit_depths_to_symbols, create_huffman_tree, write_prefix_code,
};
use super::hybrid_uint::HybridUintConfig;
use super::lz77::Lz77Params;
use super::token::{Lz77UintCoder, Token, UintCoder};
use crate::bit_writer::BitWriter;
use crate::error::{Error, Result};

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
    /// Per-histogram HybridUint configs (one per histogram).
    /// When empty, all histograms use the default {4, 2, 0} config.
    pub uint_configs: Vec<HybridUintConfig>,
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
    build_entropy_code_ans_with_options(tokens, num_contexts, false, None, None)
}

/// Build an ANS entropy code with optional enhanced clustering.
///
/// When `lz77` is Some, LZ77 length tokens use Lz77UintCoder and are offset by min_symbol.
/// When `total_pixel_hint` is Some, max_histograms is capped to `total_pixels / 2048` (min 1)
/// to prevent header overhead from dominating on small images.
pub fn build_entropy_code_ans_with_options(
    tokens: &[Token],
    num_contexts: usize,
    enhanced_clustering: bool,
    lz77: Option<&Lz77Params>,
    total_pixel_hint: Option<usize>,
) -> OwnedAnsEntropyCode {
    use crate::entropy_coding::cluster::{
        ClusteringType, EntropyType, cluster_histograms as enhanced_cluster,
    };
    use crate::entropy_coding::histogram::Histogram as EnhancedHistogram;

    // Build per-context histograms.
    // First pass: find max symbol per context to pre-allocate (avoids Vec growth).
    let mut max_sym_per_ctx = vec![0usize; num_contexts];
    for token in tokens {
        let ctx = token.context as usize;
        if ctx < num_contexts {
            let (_encoded, sym) = encode_token_value(token, lz77);
            let sym = sym as usize;
            if sym >= max_sym_per_ctx[ctx] {
                max_sym_per_ctx[ctx] = sym + 1;
            }
        }
    }

    let mut histograms: Vec<EnhancedHistogram> = max_sym_per_ctx
        .iter()
        .map(|&max_sym| EnhancedHistogram::with_capacity(max_sym))
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

    // Allow up to 96 clusters — non-simple context map format supports arbitrary counts.
    // Tested: 64→96 helps marginally, >96 has diminishing returns due to context map overhead.
    let mut max_histograms = num_contexts.min(96);
    // Scale down for small images: each histogram needs ~36 bytes ANS header overhead,
    // so for a 128x128 RGB image (~48K values), 96 histograms = ~3.4KB overhead (21% of file).
    // Cap to total_pixels/2048 so overhead stays proportional to content.
    if let Some(tp) = total_pixel_hint {
        max_histograms = max_histograms.min((tp / 2048).max(1));
    }
    let result = enhanced_cluster(cluster_type, EntropyType::Ans, &histograms, max_histograms)
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

    // Optimize per-histogram HybridUint configs (matches libjxl kFast method).
    // Try 4 configs per histogram and pick the cheapest.
    let num_histograms = result.histograms.len();
    let uint_configs = optimize_uint_configs_fast(tokens, &context_map, num_histograms, lz77);

    // Build ANS histograms and distributions with the optimized configs.
    // We need to rebuild histograms because different configs produce different token distributions.
    let mut ans_histograms = Vec::with_capacity(num_histograms);
    let mut ans_distributions = Vec::with_capacity(num_histograms);

    // Collect raw values per histogram to rebuild with optimal configs
    let mut values_per_histo: Vec<Vec<u32>> = vec![Vec::new(); num_histograms];
    for token in tokens {
        if token.is_lz77_length {
            continue; // LZ77 length tokens always use Lz77UintCoder, not affected by config
        }
        let ctx = token.context as usize;
        if ctx < context_map.len() {
            let histo_idx = context_map[ctx] as usize;
            if histo_idx < num_histograms {
                values_per_histo[histo_idx].push(token.value);
            }
        }
    }

    // Also collect LZ77 length tokens per histogram (they use Lz77UintCoder, config doesn't change)
    let mut lz77_tokens_per_histo: Vec<Vec<u32>> = vec![Vec::new(); num_histograms];
    if let Some(lz77_params) = lz77 {
        for token in tokens {
            if token.is_lz77_length {
                let ctx = token.context as usize;
                if ctx < context_map.len() {
                    let histo_idx = context_map[ctx] as usize;
                    if histo_idx < num_histograms {
                        let encoded = Lz77UintCoder::encode(token.value);
                        let sym = encoded.token + lz77_params.min_symbol;
                        lz77_tokens_per_histo[histo_idx].push(sym);
                    }
                }
            }
        }
    }

    // Phase 1: Build normalized histograms to determine alphabet sizes.
    for h in 0..num_histograms {
        let config = &uint_configs[h];
        // Build histogram from values re-encoded with the optimal config
        let mut max_sym = 0usize;
        for &val in &values_per_histo[h] {
            let (tok, _, _) = config.encode(val);
            let sym = tok as usize + 1;
            if sym > max_sym {
                max_sym = sym;
            }
        }
        for &sym in &lz77_tokens_per_histo[h] {
            let s = sym as usize + 1;
            if s > max_sym {
                max_sym = s;
            }
        }
        max_sym = max_sym.max(1);

        let mut counts = vec![0u32; max_sym];
        for &val in &values_per_histo[h] {
            let (tok, _, _) = config.encode(val);
            counts[tok as usize] += 1;
        }
        for &sym in &lz77_tokens_per_histo[h] {
            counts[sym as usize] += 1;
        }

        // Build EnhancedHistogram from counts for ANS normalization
        let mut histo = EnhancedHistogram::with_capacity(max_sym);
        for (sym, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                histo.add(sym);
            }
        }

        let ans_histo = ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise)
            .expect("ANS histogram normalization failed");
        ans_histograms.push(ans_histo);
    }

    // Phase 2: Compute global log_alpha_size from the max alphabet across all histograms.
    // This value is written to the bitstream header and used by the decoder for ALL
    // distributions. Every distribution's alias table must be built with this same value.
    // Must be at least 5, at most 8 (JXL spec: stored as 2-bit value + 5).
    let max_alphabet_size = ans_histograms
        .iter()
        .map(|h| h.counts.len())
        .max()
        .unwrap_or(1);
    let log_alpha_size = if lz77.is_some_and(|p| p.enabled) {
        // LZ77 needs at least 8 to accommodate length symbols
        8
    } else if max_alphabet_size <= (1 << ANS_LOG_ALPHA_SIZE) {
        ANS_LOG_ALPHA_SIZE
    } else {
        // Need larger alphabet: compute minimum log_alpha_size
        let min_bits = if max_alphabet_size <= 1 {
            5
        } else {
            (max_alphabet_size - 1).ilog2() as usize + 1
        };
        min_bits.clamp(5, 8)
    };

    // Phase 3: Build distributions with the global log_alpha_size.
    for ans_histo in &ans_histograms {
        let ans_dist = AnsDistribution::from_normalized_counts_with_log_alpha(
            &ans_histo.counts,
            log_alpha_size,
        )
        .expect("ANS distribution building failed");
        ans_distributions.push(ans_dist);
    }

    let code = OwnedAnsEntropyCode {
        context_map,
        histograms: ans_histograms,
        distributions: ans_distributions,
        log_alpha_size,
        uint_configs,
    };

    // Validate: every token in the stream must have a valid, non-zero frequency
    // in the distribution it maps to.
    for (i, token) in tokens.iter().enumerate() {
        let ctx = token.context as usize;
        let dist_idx = code.context_map.get(ctx).copied().unwrap_or(0) as usize;
        let config = &code.uint_configs[dist_idx];
        let (_encoded, sym) = encode_token_value_with_config(token, lz77, config);
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

/// Optimize HybridUint config per histogram cluster (matches libjxl kFast method).
///
/// For each histogram, tries 4 configs and picks the one with lowest estimated cost
/// (Shannon entropy of re-encoded tokens + extra bits + signaling cost).
fn optimize_uint_configs_fast(
    tokens: &[Token],
    context_map: &[u8],
    num_histograms: usize,
    lz77: Option<&Lz77Params>,
) -> Vec<HybridUintConfig> {
    use crate::entropy_coding::ans::ANS_MAX_ALPHABET_SIZE;

    // The 4 configs libjxl tries at effort 7 (kFast)
    let candidates = [
        HybridUintConfig::new(4, 2, 0), // default
        HybridUintConfig::new(4, 1, 2), // parity
        HybridUintConfig::new(0, 0, 0), // varlenuint (smallest alphabet)
        HybridUintConfig::new(2, 0, 1), // ctxmap-style
    ];

    // Collect raw values per histogram
    let mut values_per_histo: Vec<Vec<u32>> = vec![Vec::new(); num_histograms];
    for token in tokens {
        if token.is_lz77_length {
            continue; // LZ77 tokens always use Lz77UintCoder
        }
        let ctx = token.context as usize;
        if ctx < context_map.len() {
            let histo_idx = context_map[ctx] as usize;
            if histo_idx < num_histograms {
                values_per_histo[histo_idx].push(token.value);
            }
        }
    }

    let max_alpha = ANS_MAX_ALPHABET_SIZE;

    let mut best_configs = vec![HybridUintConfig::new(4, 2, 0); num_histograms];

    for h in 0..num_histograms {
        let values = &values_per_histo[h];
        if values.is_empty() {
            continue;
        }

        let max_value = values.iter().copied().max().unwrap_or(0);
        let mut best_cost = f64::MAX;

        for &cfg in &candidates {
            // Check that the max token fits in the alphabet
            let (max_tok, _, _) = cfg.encode(max_value);
            let max_tok_with_lsb = max_tok | ((1u32 << cfg.lsb_in_token) - 1);
            if max_tok_with_lsb as usize >= max_alpha {
                continue;
            }
            if let Some(lz77_params) = lz77
                && max_tok_with_lsb >= lz77_params.min_symbol
            {
                continue;
            }

            // Re-encode all values with this config
            let capacity = max_tok_with_lsb as usize + 1;
            let mut counts = vec![0u32; capacity];
            let mut extra_bits_total: u64 = 0;
            for &val in values {
                let (tok, _, nbits) = cfg.encode(val);
                counts[tok as usize] += 1;
                extra_bits_total += nbits as u64;
            }

            // Compute Shannon entropy cost of the token histogram
            let total = values.len() as f64;
            let mut entropy_cost = 0.0f64;
            for &count in &counts {
                if count > 0 {
                    let p = count as f64 / total;
                    entropy_cost -= count as f64 * p.log2();
                }
            }

            // Total cost = entropy (in bits) + extra bits + signaling cost
            let signaling_cost = if cfg.split_exponent == 0 {
                0.0
            } else {
                ceil_log2_nonzero_usize(cfg.split_exponent as usize + 1) as f64
                    + ceil_log2_nonzero_usize((cfg.split_exponent - cfg.msb_in_token) as usize + 1)
                        as f64
            };
            let cost = entropy_cost + extra_bits_total as f64 + signaling_cost;

            if cost < best_cost {
                best_cost = cost;
                best_configs[h] = cfg;
            }
        }
    }

    best_configs
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
    for (i, _) in code.histograms.iter().enumerate() {
        let config = code.uint_configs.get(i).copied().unwrap_or_default();
        write_hybrid_uint_config_value(las, &config, writer)?;
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
///
/// Supports both simple format (≤8 histograms, raw bits) and non-simple format
/// (>8 histograms, Huffman-encoded with optional MTF transform).
fn write_context_map_for_ans(code: &OwnedAnsEntropyCode, writer: &mut BitWriter) -> Result<()> {
    let num_histograms = code.histograms.len();

    if num_histograms == 1 {
        // Simple context map: all contexts map to histogram 0
        writer.write(1, 1)?; // simple_context_map = true
        writer.write(2, 0)?; // nbits = 0
        return Ok(());
    }

    if num_histograms <= 8 {
        // Simple context map with multiple histograms
        // bits_per_entry: 1 = 2 histos, 2 = 4 histos, 3 = 8 histos
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
        return Ok(());
    }

    // Non-simple context map for > 8 histograms.
    // Format: is_simple=0, use_mtf, then Huffman-encoded context map entries.
    //
    // Try both direct and MTF, pick whichever has lower estimated cost.
    let direct_tokens = &code.context_map;
    let mtf_tokens = move_to_front_transform(&code.context_map);

    let direct_cost = estimate_context_map_cost(direct_tokens);
    let mtf_cost = estimate_context_map_cost(&mtf_tokens);
    let use_mtf = mtf_cost < direct_cost;
    let tokens: &[u8] = if use_mtf { &mtf_tokens } else { direct_tokens };

    // is_simple=0, use_mtf, lz77_enabled=0 (3 bits packed)
    let header_bits = if use_mtf { 0b010u64 } else { 0b000u64 };
    writer.write(3, header_bits)?;

    // Now write a Huffman-encoded entropy code for the context map values.
    // Since num_contexts=1 for the context map's own entropy code, no inner context map.

    // use_prefix_code = 1 (Huffman)
    writer.write(1, 1)?;

    // HybridUint config: split=4, msb=2, lsb=0 (same as our UintCoder)
    writer.write(4, 4)?; // split_exponent = 4
    writer.write(3, 2)?; // msb_in_token = 2
    writer.write(2, 0)?; // lsb_in_token = 0

    // Build histogram of encoded token symbols
    let mut histogram = [0u32; ALPHABET_SIZE];
    for &t in tokens {
        let encoded = UintCoder::encode(t as u32);
        histogram[encoded.token as usize] += 1;
    }

    // Find alphabet length (trim trailing zeros)
    let mut length = ALPHABET_SIZE;
    while length > 0 && histogram[length - 1] == 0 {
        length -= 1;
    }
    length = length.max(1);

    // Create Huffman tree
    let mut depths = [0u8; ALPHABET_SIZE];
    create_huffman_tree(&histogram, length, 15, &mut depths);

    let mut bits = [0u16; ALPHABET_SIZE];
    convert_bit_depths_to_symbols(&depths, &mut bits);

    // Write alphabet size
    write_var_len_uint16(length - 1, writer)?;

    // Write prefix code tree
    if length > 1 {
        let pc = PrefixCode { depths, bits };
        // Write the prefix code directly using the internal function
        write_prefix_code(&pc, writer)?;
    }

    // Write encoded context map entries
    for &t in tokens {
        let encoded = UintCoder::encode(t as u32);
        let tok = encoded.token as usize;
        let depth = depths[tok] as usize;
        let b = bits[tok] as u64;
        let data = b | ((encoded.bits as u64) << depth);
        let total_bits = depth + encoded.nbits as usize;
        writer.write(total_bits, data)?;
    }

    Ok(())
}

/// Estimate the Shannon entropy cost of a byte sequence (for context map cost comparison).
fn estimate_context_map_cost(tokens: &[u8]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &t in tokens {
        counts[t as usize] += 1;
    }
    let total = tokens.len() as f64;
    let mut cost = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            cost -= p * p.log2();
        }
    }
    cost * total
}

/// Write HybridUint config with specific split/msb/lsb values.
fn write_hybrid_uint_config_value(
    log_alpha_size: usize,
    config: &HybridUintConfig,
    writer: &mut BitWriter,
) -> Result<()> {
    let split_exponent = config.split_exponent;
    let msb_in_token = config.msb_in_token;
    let lsb_in_token = config.lsb_in_token;

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
    let mut encoder = AnsEncoder::with_capacity(tokens.len());

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
        let dist_idx_for_cfg = code.context_map.get(ctx).copied().unwrap_or(0) as usize;
        let config = code
            .uint_configs
            .get(dist_idx_for_cfg)
            .copied()
            .unwrap_or_default();
        let (encoded, sym) = encode_token_value_with_config(token, lz77, &config);

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
        // Add padding bytes so the decoder's peek(7) doesn't read past the end.
        // In a real bitstream, more data follows the histogram. In this isolated
        // test, we add explicit zero padding.
        writer.write(8, 0)?;
        writer.zero_pad_to_byte();
        let bytes = writer.finish();

        // Decode it back
        let mut br = BitReader::new(&bytes);
        let decoded = match AnsHistogram::decode(&mut br, code.log_alpha_size) {
            Ok(d) => d,
            Err(e) => {
                #[cfg(feature = "debug-rect")]
                eprintln!(
                    "{} histo[{}]: DECODE FAILED - {} (method={} alpha={} omit={})",
                    label, i, e, histo.method, histo.alphabet_size, histo.omit_pos
                );
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
                    #[cfg(feature = "debug-rect")]
                    eprintln!(
                        "{} histo[{}]: FREQ MISMATCH (method={} alpha={})",
                        label, i, histo.method, histo.alphabet_size
                    );
                }
                #[cfg(feature = "debug-rect")]
                eprintln!("sym[{}]: expected={} got={}", j, expected, got);
                mismatch = true;
            }
        }

        if mismatch {
            #[cfg(feature = "debug-rect")]
            {
                eprintln!("counts: {:?}", &histo.counts[..histo.alphabet_size]);
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
                    "omit_pos={} (stored: method={} omit={})",
                    encoder_omit, histo.method, histo.omit_pos
                );
                eprintln!(
                    "omit logcount={} count_at_omit={}",
                    encoder_omit_logcount, histo.counts[histo.omit_pos]
                );
                // Check what decoder would see
                for k in 0..histo.alphabet_size.min(40) {
                    let c = histo.counts[k];
                    if c > 0 {
                        let lc = crate::entropy_coding::ans::floor_log2_ans(c as u32) + 1;
                        if lc == encoder_omit_logcount {
                            eprintln!("sym[{}]: count={} logcount={} (same as max)", k, c, lc);
                        }
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
    let log_alpha_size = code.log_alpha_size;
    let table_size = 1usize << log_alpha_size;
    let decoder_histograms: Vec<AnsHistogram> = code
        .distributions
        .iter()
        .map(|dist| {
            // Build frequency array padded to alias table size
            let mut freqs = vec![0u16; dist.symbols.len().max(table_size)];
            for (i, sym) in dist.symbols.iter().enumerate() {
                freqs[i] = sym.freq;
            }

            // Build alias map using the decoder's method
            let log_bucket_size = 12 - log_alpha_size; // LOG_SUM_PROBS - log_alpha_size
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
    #[allow(clippy::unused_enumerate_index)] // _i used in #[cfg(feature = "debug-rect")] output
    for (_i, token) in tokens.iter().enumerate() {
        let ctx = token.context as usize;
        let dist_idx = code.context_map.get(ctx).copied().unwrap_or(0) as usize;
        let decoder_hist = &decoder_histograms[dist_idx];

        // Decode one ANS symbol
        let decoded_symbol = decoder_hist.read(&mut br2, &mut ans_reader.0);

        // Read extra bits (HybridUint) — use per-histogram config
        let config = code.uint_configs.get(dist_idx).copied().unwrap_or_default();
        let (expected_encoded, _expected_sym) =
            encode_token_value_with_config(token, None, &config);
        let decoded_extra = if expected_encoded.nbits > 0 {
            br2.read(expected_encoded.nbits as usize).unwrap_or(0) as u32
        } else {
            0
        };

        // Compare token (ANS symbol)
        if decoded_symbol != expected_encoded.token {
            if mismatches < 5 {
                #[cfg(feature = "debug-rect")]
                eprintln!(
                    "MISMATCH token[{}]: ctx={} val={} exp={} got={} state=0x{:08x}",
                    _i, ctx, token.value, expected_encoded.token, decoded_symbol, ans_reader.0
                );
            }
            mismatches += 1;
        }

        // Compare extra bits
        if decoded_extra != expected_encoded.bits {
            if mismatches < 5 {
                #[cfg(feature = "debug-rect")]
                eprintln!(
                    "BITS MISMATCH token[{}]: exp=0x{:x} got=0x{:x}",
                    _i, expected_encoded.bits, decoded_extra
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

/// Test ANS roundtrip using PARSED histogram (not known distributions).
///
/// This exercises the exact format that a real decoder uses:
/// write_ans_modular_header → parse histogram from bitstream → decode tokens.
/// Unlike verify_ans_roundtrip which builds decoder histograms from encoder's
/// known distributions, this test catches format mismatches where our encoder
/// and our internal decoder agree but external decoders (jxl-rs, djxl) disagree.
#[cfg(debug_assertions)]
pub fn verify_ans_roundtrip_parsed(tokens: &[Token], code: &OwnedAnsEntropyCode) -> Result<()> {
    use crate::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};

    #[inline]
    fn ceil_log2_nonzero(x: u32) -> u32 {
        if x <= 1 {
            0
        } else {
            u32::BITS - (x - 1).leading_zeros()
        }
    }

    if tokens.is_empty() {
        return Ok(());
    }

    assert_eq!(
        code.histograms.len(),
        1,
        "verify_ans_roundtrip_parsed only supports single-distribution"
    );

    // Write exactly what the modular encoder writes: header + tokens
    let mut writer = BitWriter::new();
    // Inline the modular ANS header format:
    // lz77.enabled = 0
    writer.write(1, 0)?;
    // No context map for num_dist=1
    // use_prefix_code = 0 (ANS)
    writer.write(1, 0)?;
    // log_alpha_size - 5 (2 bits)
    let las = code.log_alpha_size;
    writer.write(2, (las - 5) as u64)?;
    // HybridUint config
    let config = code
        .uint_configs
        .first()
        .copied()
        .unwrap_or(crate::entropy_coding::hybrid_uint::HybridUintConfig::default_config());
    let se_bits = ceil_log2_nonzero(las as u32 + 1) as usize;
    writer.write(se_bits, config.split_exponent as u64)?;
    if (config.split_exponent as usize) != las {
        let msb_bits = ceil_log2_nonzero(config.split_exponent + 1) as usize;
        writer.write(msb_bits, config.msb_in_token as u64)?;
        let lsb_bits = ceil_log2_nonzero(config.split_exponent - config.msb_in_token + 1) as usize;
        writer.write(lsb_bits, config.lsb_in_token as u64)?;
    }
    // Write the single ANS distribution
    code.histograms[0].write(&mut writer)?;
    let header_bits = writer.bits_written();
    write_tokens_ans(tokens, code, None, &mut writer)?;
    writer.zero_pad_to_byte();
    let encoded_bytes = writer.finish();

    eprintln!(
        "verify_ans_roundtrip_parsed: header={} bits, total={} bytes",
        header_bits,
        encoded_bytes.len()
    );

    // Now parse it back exactly as a decoder would
    let mut br = BitReader::new(&encoded_bytes);

    // 1. lz77.enabled
    let lz77_enabled = br.read(1)?;
    assert_eq!(lz77_enabled, 0, "expected lz77.enabled=0");

    // 2. context_map skipped for num_dist=1

    // 3. use_prefix_code
    let use_prefix_code = br.read(1)?;
    assert_eq!(use_prefix_code, 0, "expected use_prefix_code=0 (ANS)");

    // 4. log_alpha_size
    let las = br.read(2)? as usize + 5;
    eprintln!("  parsed log_alpha_size={}", las);
    assert_eq!(las, code.log_alpha_size, "log_alpha_size mismatch");

    // 5. HybridUint config
    let se_bits = ceil_log2_nonzero(las as u32 + 1) as usize;
    let split_exponent = br.read(se_bits)? as u32;
    let (msb_in_token, lsb_in_token) = if split_exponent != las as u32 {
        let msb_bits = ceil_log2_nonzero(split_exponent + 1) as usize;
        let msb = br.read(msb_bits)? as u32;
        let lsb_bits = ceil_log2_nonzero(split_exponent - msb + 1) as usize;
        let lsb = br.read(lsb_bits)? as u32;
        (msb, lsb)
    } else {
        (0, 0)
    };
    let expected_config = code
        .uint_configs
        .first()
        .copied()
        .unwrap_or(crate::entropy_coding::hybrid_uint::HybridUintConfig::default_config());
    eprintln!(
        "  parsed uint_config: se={} msb={} lsb={} (expected se={} msb={} lsb={})",
        split_exponent,
        msb_in_token,
        lsb_in_token,
        expected_config.split_exponent,
        expected_config.msb_in_token,
        expected_config.lsb_in_token
    );
    assert_eq!(
        split_exponent, expected_config.split_exponent,
        "split_exponent mismatch"
    );
    assert_eq!(
        msb_in_token, expected_config.msb_in_token,
        "msb_in_token mismatch"
    );
    assert_eq!(
        lsb_in_token, expected_config.lsb_in_token,
        "lsb_in_token mismatch"
    );

    // 6. Parse the ANS histogram (this is what jxl-rs does)
    let parsed_histo = AnsHistogram::decode(&mut br, las)?;
    let bits_after_histo = br.bits_read();
    eprintln!(
        "  histogram parsed OK at bit {}, freqs: {:?}",
        bits_after_histo,
        &parsed_histo.frequencies[..parsed_histo.frequencies.len().min(10)]
    );

    // Compare parsed frequencies with encoder's distribution
    for (i, sym) in code.distributions[0].symbols.iter().enumerate() {
        let parsed_freq = parsed_histo.frequencies.get(i).copied().unwrap_or(0);
        if parsed_freq != sym.freq {
            eprintln!(
                "  FREQ MISMATCH at symbol {}: encoder={} parsed={}",
                i, sym.freq, parsed_freq
            );
            return Err(Error::Bitstream(format!(
                "Parsed histogram frequency mismatch at symbol {}: encoder={} parsed={}",
                i, sym.freq, parsed_freq
            )));
        }
    }
    eprintln!("  frequencies match encoder's distribution");

    // 7. Read 32-bit ANS state
    let mut ans_reader = AnsReader::init(&mut br)?;
    eprintln!("  ANS initial state: 0x{:08x}", ans_reader.state());

    // 8. Decode tokens
    let config = crate::entropy_coding::hybrid_uint::HybridUintConfig {
        split_exponent,
        split: 1 << split_exponent,
        msb_in_token,
        lsb_in_token,
    };
    let mut mismatches = 0;
    for (i, token) in tokens.iter().enumerate() {
        let (expected_encoded, _) =
            crate::entropy_coding::encode::encode_token_value_with_config(token, None, &config);

        // Decode ANS symbol
        let decoded_symbol = parsed_histo.read(&mut br, &mut ans_reader.0);

        // Read extra bits
        let decoded_extra = if expected_encoded.nbits > 0 {
            br.read(expected_encoded.nbits as usize).unwrap_or(0) as u32
        } else {
            0
        };

        if decoded_symbol != expected_encoded.token || decoded_extra != expected_encoded.bits {
            if mismatches < 5 {
                eprintln!(
                    "  MISMATCH token[{}]: val={} exp_tok={} got_tok={} exp_bits=0x{:x} got_bits=0x{:x}",
                    i,
                    token.value,
                    expected_encoded.token,
                    decoded_symbol,
                    expected_encoded.bits,
                    decoded_extra
                );
            }
            mismatches += 1;
        }
    }

    // 9. Check final state
    if let Err(e) = ans_reader.check_final_state() {
        eprintln!(
            "  FINAL STATE FAILED: {} mismatches, state=0x{:08x}",
            mismatches,
            ans_reader.state()
        );
        return Err(Error::Bitstream(format!(
            "ANS parsed roundtrip final state check failed ({} mismatches): {}",
            mismatches, e
        )));
    }

    if mismatches > 0 {
        return Err(Error::Bitstream(format!(
            "ANS parsed roundtrip had {} mismatches",
            mismatches
        )));
    }

    eprintln!("  ANS parsed roundtrip OK: {} tokens", tokens.len());
    Ok(())
}
