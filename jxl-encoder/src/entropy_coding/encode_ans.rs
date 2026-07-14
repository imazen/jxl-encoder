// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! ANS (Asymmetric Numeral Systems) entropy code building and serialization.
//!
//! Contains all ANS-specific code: distribution building, header writing,
//! token writing, and verification/roundtrip utilities.

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

/// Accumulated histogram data from a token stream (or multiple token streams).
///
/// This struct captures all the statistical information needed to build an ANS
/// entropy code, without retaining the tokens themselves. This enables a two-phase
/// encode approach: accumulate statistics (Phase 1) → build codes → re-tokenize
/// and write (Phase 2), avoiding O(total_tokens) memory for large images.
pub struct AccumulatedAnsData {
    /// Per-context symbol frequency counts.
    pub histograms: Vec<super::histogram::Histogram>,
    /// Per-context raw value frequencies (value → count).
    /// Used for HybridUint config optimization after histogram clustering.
    pub value_freqs: Vec<alloc::collections::BTreeMap<u32, u32>>,
    /// Per-context LZ77 symbol frequencies (symbol → count).
    pub lz77_freqs: Vec<alloc::collections::BTreeMap<u32, u32>>,
    /// Number of contexts.
    pub num_contexts: usize,
}

impl AccumulatedAnsData {
    /// Create a new empty accumulator for the given number of contexts.
    pub fn new(num_contexts: usize) -> Self {
        Self {
            histograms: (0..num_contexts)
                .map(|_| super::histogram::Histogram::new())
                .collect(),
            value_freqs: (0..num_contexts)
                .map(|_| alloc::collections::BTreeMap::new())
                .collect(),
            lz77_freqs: (0..num_contexts)
                .map(|_| alloc::collections::BTreeMap::new())
                .collect(),
            num_contexts,
        }
    }

    /// Accumulate a token into the histograms and value frequency maps.
    #[inline]
    pub fn add_token(&mut self, token: &Token, lz77: Option<&Lz77Params>) {
        let ctx = token.context() as usize;
        if ctx < self.num_contexts {
            let (_encoded, sym) = encode_token_value(token, lz77);
            self.histograms[ctx].add(sym as usize);
            if token.is_lz77_length() {
                if let Some(lz77_params) = lz77 {
                    let encoded = Lz77UintCoder::encode(token.value);
                    let lz77_sym = encoded.token + lz77_params.min_symbol;
                    *self.lz77_freqs[ctx].entry(lz77_sym).or_insert(0) += 1;
                }
            } else {
                *self.value_freqs[ctx].entry(token.value).or_insert(0) += 1;
            }
        }
    }

    /// Accumulate all tokens from a slice.
    pub fn add_tokens(&mut self, tokens: &[Token], lz77: Option<&Lz77Params>) {
        for token in tokens {
            self.add_token(token, lz77);
        }
    }

    /// Merge another accumulator into this one (for combining per-thread results).
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))] // used by the parallel accumulate map-reduce
    pub fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.num_contexts, other.num_contexts);
        for ctx in 0..self.num_contexts {
            self.histograms[ctx].add_histogram(&other.histograms[ctx]);
            for (&val, &count) in &other.value_freqs[ctx] {
                *self.value_freqs[ctx].entry(val).or_insert(0) += count;
            }
            for (&sym, &count) in &other.lz77_freqs[ctx] {
                *self.lz77_freqs[ctx].entry(sym).or_insert(0) += count;
            }
        }
    }
}

/// Parallel map-reduce: build an accumulator per group on a worker
/// thread, then merge into one. Falls through to a sequential loop
/// when the `parallel` feature is disabled or there is only one group.
#[cfg(feature = "parallel")]
fn accumulate_groups_parallel(
    groups: &[&[Token]],
    num_contexts: usize,
    lz77: Option<&Lz77Params>,
) -> AccumulatedAnsData {
    use rayon::prelude::*;
    if groups.len() <= 1 {
        let mut acc = AccumulatedAnsData::new(num_contexts);
        for group in groups {
            acc.add_tokens(group, lz77);
        }
        return acc;
    }
    groups
        .par_iter()
        .map(|group| {
            let mut acc = AccumulatedAnsData::new(num_contexts);
            acc.add_tokens(group, lz77);
            acc
        })
        .reduce(
            || AccumulatedAnsData::new(num_contexts),
            |mut a, b| {
                a.merge(&b);
                a
            },
        )
}

/// Sequential fallback for [`accumulate_groups_parallel`].
#[cfg(not(feature = "parallel"))]
fn accumulate_groups_parallel(
    groups: &[&[Token]],
    num_contexts: usize,
    lz77: Option<&Lz77Params>,
) -> AccumulatedAnsData {
    let mut acc = AccumulatedAnsData::new(num_contexts);
    for group in groups {
        acc.add_tokens(group, lz77);
    }
    acc
}

/// Build an ANS entropy code from accumulated histogram data.
///
/// This is Phase B of the two-phase approach: takes pre-accumulated statistics
/// and builds the entropy code (clustering, HybridUint optimization, ANS distributions).
#[allow(dead_code)] // simple-form entry point; live callers thread strategy via _with_strategy
pub fn build_entropy_code_from_accumulated_ans(
    data: AccumulatedAnsData,
    enhanced_clustering: bool,
    optimize_uint_configs: bool,
    lz77: Option<&Lz77Params>,
    total_pixel_hint: Option<usize>,
) -> OwnedAnsEntropyCode {
    build_entropy_code_from_accumulated_ans_with_strategy(
        data,
        enhanced_clustering,
        optimize_uint_configs,
        lz77,
        total_pixel_hint,
        ANSHistogramStrategy::Precise,
    )
}

/// Build an ANS entropy code from accumulated histogram data, with explicit
/// ANS histogram normalization strategy.
///
/// `ans_strategy` controls the shift-grid density used by
/// [`ANSEncodingHistogram::from_histogram_cached`]:
/// - `Precise` (libjxl default for `tier < kSquirrel`, our effort >= 8): all
///   12 shifts tried, picks the one with lowest total cost.
/// - `Approximate` (libjxl `tier >= kSquirrel`, our effort <= 7): every other
///   shift (7 values). Headers are typically smaller because the chosen
///   shift sits on a coarser grid; the very small data-fit penalty is more
///   than recovered by lower header overhead on streams with many histograms.
/// - `Fast`: only 3 shifts (0, mid, max).
pub fn build_entropy_code_from_accumulated_ans_with_strategy(
    data: AccumulatedAnsData,
    enhanced_clustering: bool,
    optimize_uint_configs: bool,
    lz77: Option<&Lz77Params>,
    total_pixel_hint: Option<usize>,
    ans_strategy: ANSHistogramStrategy,
) -> OwnedAnsEntropyCode {
    use crate::entropy_coding::cluster::{
        ClusteringType, EntropyType, cluster_histograms as enhanced_cluster,
    };
    use crate::entropy_coding::histogram::Histogram as EnhancedHistogram;

    let num_contexts = data.num_contexts;

    // Cluster histograms
    let cluster_type = if enhanced_clustering {
        ClusteringType::Best
    } else {
        ClusteringType::Fast
    };

    let mut max_histograms = num_contexts.min(128);
    if let Some(tp) = total_pixel_hint {
        max_histograms = max_histograms.min((tp / 2048).max(1));
    }
    let result = enhanced_cluster(
        cluster_type,
        EntropyType::Ans,
        &data.histograms,
        max_histograms,
    )
    .expect("ANS clustering failed");

    let context_map: Vec<u8> = result.symbols.iter().map(|&s| s as u8).collect();
    debug_assert_eq!(context_map.len(), num_contexts);

    // Merge per-context value frequencies into per-merged-histogram frequencies
    // using the context map from clustering.
    let num_histograms = result.histograms.len();
    let mut merged_value_freqs: Vec<alloc::collections::BTreeMap<u32, u32>> = (0..num_histograms)
        .map(|_| alloc::collections::BTreeMap::new())
        .collect();
    let mut merged_lz77_freqs: Vec<alloc::collections::BTreeMap<u32, u32>> = (0..num_histograms)
        .map(|_| alloc::collections::BTreeMap::new())
        .collect();
    for (ctx, &cm) in context_map.iter().enumerate() {
        let histo_idx = cm as usize;
        if histo_idx < num_histograms {
            for (&val, &count) in &data.value_freqs[ctx] {
                *merged_value_freqs[histo_idx].entry(val).or_insert(0) += count;
            }
            for (&sym, &count) in &data.lz77_freqs[ctx] {
                *merged_lz77_freqs[histo_idx].entry(sym).or_insert(0) += count;
            }
        }
    }

    // Optimize per-histogram HybridUint configs from merged value frequencies.
    // libjxl uses uint_method=kNone (no optimization) for VarDCT AC/DC at effort < 9.
    // The fast optimization can pick non-default configs whose signaling overhead
    // exceeds their coding benefit on VarDCT streams.
    let uint_configs = if !optimize_uint_configs {
        vec![HybridUintConfig::new(4, 2, 0); num_histograms]
    } else if enhanced_clustering {
        optimize_uint_configs_best_from_freqs(&merged_value_freqs, lz77)
    } else {
        optimize_uint_configs_fast_from_freqs(&merged_value_freqs, lz77)
    };

    // Build ANS histograms with the optimized configs. Per-histogram
    // independent: each iteration reads merged_value_freqs[h],
    // merged_lz77_freqs[h], uint_configs[h] and the read-only
    // allowed_cache. Parallelizes cleanly via parallel_map.
    let allowed_cache = super::ans::AllowedCountsCache::new();
    let ans_histograms: Vec<ANSEncodingHistogram> =
        crate::parallel::parallel_map(num_histograms, |h| {
            let config = &uint_configs[h];
            let mut counts: Vec<u32> = Vec::new();
            for (&val, &freq) in &merged_value_freqs[h] {
                let (tok, _, _) = config.encode(val);
                let sym = tok as usize;
                if sym >= counts.len() {
                    counts.resize(sym + 1, 0);
                }
                counts[sym] += freq;
            }
            for (&sym, &freq) in &merged_lz77_freqs[h] {
                let s = sym as usize;
                if s >= counts.len() {
                    counts.resize(s + 1, 0);
                }
                counts[s] += freq;
            }
            if counts.is_empty() {
                counts.push(0);
            }
            let i32_counts: Vec<i32> = counts.iter().map(|&c| c as i32).collect();
            let histo = EnhancedHistogram::from_counts(&i32_counts);
            ANSEncodingHistogram::from_histogram_cached(&histo, ans_strategy, &allowed_cache)
                .expect("ANS histogram normalization failed")
        });

    // Compute global log_alpha_size
    let max_alphabet_size = ans_histograms
        .iter()
        .map(|h| h.counts.len())
        .max()
        .unwrap_or(1);
    let log_alpha_size = if lz77.is_some_and(|p| p.enabled) {
        8
    } else if max_alphabet_size <= (1 << ANS_LOG_ALPHA_SIZE) {
        ANS_LOG_ALPHA_SIZE
    } else {
        let min_bits = if max_alphabet_size <= 1 {
            5
        } else {
            (max_alphabet_size - 1).ilog2() as usize + 1
        };
        min_bits.clamp(5, 8)
    };

    // Build ANS distributions per histogram (also independent — each
    // pass-and-build uses one ans_histograms[h] entry).
    let ans_distributions: Vec<AnsDistribution> =
        crate::parallel::parallel_map(ans_histograms.len(), |h| {
            AnsDistribution::from_normalized_counts_with_log_alpha(
                &ans_histograms[h].counts,
                log_alpha_size,
            )
            .expect("ANS distribution building failed")
        });

    OwnedAnsEntropyCode {
        context_map,
        histograms: ans_histograms,
        distributions: ans_distributions,
        log_alpha_size,
        uint_configs,
    }
}

/// Build an ANS entropy code from collected tokens.
///
/// 1. Creates per-context histograms from all tokens.
/// 2. Clusters histograms (max 8 clusters) to produce a context map.
/// 3. Normalizes each cluster histogram to sum to 4096.
/// 4. Builds ANS distributions for encoding.
pub fn build_entropy_code_ans(tokens: &[Token], num_contexts: usize) -> OwnedAnsEntropyCode {
    build_entropy_code_ans_with_options(tokens, num_contexts, false, true, None, None)
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
    optimize_uint_configs: bool,
    lz77: Option<&Lz77Params>,
    total_pixel_hint: Option<usize>,
) -> OwnedAnsEntropyCode {
    build_entropy_code_ans_from_token_groups(
        &[tokens],
        num_contexts,
        enhanced_clustering,
        optimize_uint_configs,
        lz77,
        total_pixel_hint,
    )
}

/// Build an ANS entropy code from multiple token groups without merging.
///
/// Like `build_entropy_code_ans_with_options`, but accepts separate token slices
/// (e.g., per-group tokens) and iterates them without creating a merged copy.
/// This avoids allocating a merged Vec that can be hundreds of MB for large images.
///
/// Internally uses the two-phase accumulate + build approach: collects per-context
/// histograms and value frequencies in a single pass, then builds codes from the
/// accumulated data.
pub fn build_entropy_code_ans_from_token_groups(
    groups: &[&[Token]],
    num_contexts: usize,
    enhanced_clustering: bool,
    optimize_uint_configs: bool,
    lz77: Option<&Lz77Params>,
    total_pixel_hint: Option<usize>,
) -> OwnedAnsEntropyCode {
    build_entropy_code_ans_from_token_groups_with_strategy(
        groups,
        num_contexts,
        enhanced_clustering,
        optimize_uint_configs,
        lz77,
        total_pixel_hint,
        ANSHistogramStrategy::Precise,
    )
}

/// Like [`build_entropy_code_ans_from_token_groups`] but lets the caller pick
/// the ANS histogram normalization strategy. See
/// [`build_entropy_code_from_accumulated_ans_with_strategy`] for the meaning of
/// the strategy values; in normal usage the caller should pass the value from
/// the active [`crate::effort::EffortProfile::ans_histogram_strategy_vardct`]
/// (VarDCT path) or leave the default `Precise` (lossless / one-off paths).
pub fn build_entropy_code_ans_from_token_groups_with_strategy(
    groups: &[&[Token]],
    num_contexts: usize,
    enhanced_clustering: bool,
    optimize_uint_configs: bool,
    lz77: Option<&Lz77Params>,
    total_pixel_hint: Option<usize>,
    ans_strategy: ANSHistogramStrategy,
) -> OwnedAnsEntropyCode {
    // Phase A: Accumulate per-context histograms and value frequencies.
    // Per-group accumulators are independent and merge associatively;
    // run a parallel map-reduce over the groups.
    let accumulated = accumulate_groups_parallel(groups, num_contexts, lz77);

    // Phase B: Build entropy code from accumulated data.
    let code = build_entropy_code_from_accumulated_ans_with_strategy(
        accumulated,
        enhanced_clustering,
        optimize_uint_configs,
        lz77,
        total_pixel_hint,
        ans_strategy,
    );

    // Validate: every token in the stream must have a valid, non-zero frequency
    // in the distribution it maps to. Only in debug builds — this is O(n) over all tokens.
    #[cfg(debug_assertions)]
    for group in groups {
        for (i, token) in group.iter().enumerate() {
            let ctx = token.context() as usize;
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
    }

    code
}

/// Optimize HybridUint config per histogram cluster (matches libjxl kFast method).
///
/// For each histogram, tries 4 configs and picks the one with lowest estimated cost
/// (Shannon entropy of re-encoded tokens + extra bits + signaling cost).
/// Optimize HybridUint config per histogram from value frequency maps (kFast method).
///
/// Tries 4 configs per histogram (libjxl effort 7). Iterates (value, count) pairs
/// from frequency maps instead of individual values, avoiding O(tokens) storage.
fn optimize_uint_configs_fast_from_freqs(
    freqs_per_histo: &[alloc::collections::BTreeMap<u32, u32>],
    lz77: Option<&Lz77Params>,
) -> Vec<HybridUintConfig> {
    use crate::entropy_coding::ans::ANS_MAX_ALPHABET_SIZE;

    let candidates = [
        HybridUintConfig::new(4, 2, 0),
        HybridUintConfig::new(4, 1, 2),
        HybridUintConfig::new(0, 0, 0),
        HybridUintConfig::new(2, 0, 1),
    ];

    let num_histograms = freqs_per_histo.len();
    let max_alpha = ANS_MAX_ALPHABET_SIZE;

    let mut best_configs = vec![HybridUintConfig::new(4, 2, 0); num_histograms];
    let mut counts_buf: Vec<u32> = Vec::new();

    for h in 0..num_histograms {
        let freqs = &freqs_per_histo[h];
        if freqs.is_empty() {
            continue;
        }

        let max_value = freqs.keys().copied().max().unwrap_or(0);
        let total: u32 = freqs.values().sum();
        let mut best_cost = f64::MAX;

        for &cfg in &candidates {
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

            let capacity = max_tok_with_lsb as usize + 1;
            counts_buf.clear();
            counts_buf.resize(capacity, 0);
            let mut extra_bits_total: u64 = 0;
            for (&val, &freq) in freqs {
                let (tok, _, nbits) = cfg.encode(val);
                counts_buf[tok as usize] += freq;
                extra_bits_total += nbits as u64 * freq as u64;
            }

            let inv_total = 1.0f32 / total as f32;
            let mut entropy_cost = 0.0f64;
            let mut nonzero_count: usize = 0;
            for &count in &counts_buf[..capacity] {
                if count > 0 {
                    nonzero_count += 1;
                    let c = count as f32;
                    entropy_cost -= c as f64 * jxl_simd::fast_log2f(c * inv_total) as f64;
                }
            }

            let signaling_cost = if cfg.split_exponent == 0 {
                0.0
            } else {
                ceil_log2_nonzero_usize(cfg.split_exponent as usize + 1) as f64
                    + ceil_log2_nonzero_usize((cfg.split_exponent - cfg.msb_in_token) as usize + 1)
                        as f64
            };
            // ANS distribution header cost: scales with the number of non-zero
            // symbols in the per-config histogram. libjxl's `Histogram::ANS-
            // PopulationCost()` (`enc_ans.cc:823-863`) ADDS this term; we were
            // missing it, which caused the optimizer to UNDER-cost wide-alphabet
            // configs like (0,0,0) (which can span up to 256 tokens) and pick
            // them when (4,2,0)-style narrow-alphabet configs were actually
            // smaller end-to-end. See `cluster.rs::ans_population_cost` for the
            // companion estimate used in pair-merge cluster cost: `alphabet_size
            // * 5.0`. Here we use `nonzero_count * 8.0` because the writer's
            // RLE on zero runs (`ans.rs:1278-1313`) makes the per-zero-symbol
            // cost negligible, so cost scales with non-zero entries (each
            // costing ~3-6 bits logcount prefix + ~4-12 bits precision).
            //
            // Approximation: 8 bits per non-zero symbol. Calibrated against
            // libjxl's actual `ANSPopulationCost()` differential between
            // (0,0,0) and (4,2,0) on JPEG AC histograms.
            let header_cost = nonzero_count as f64 * 8.0;
            let cost = entropy_cost + extra_bits_total as f64 + signaling_cost + header_cost;

            if cost < best_cost {
                best_cost = cost;
                best_configs[h] = cfg;
            }
        }
    }

    best_configs
}

/// Optimize HybridUint config per histogram from value frequency maps (kBest method).
///
/// Tries 28 curated configs per histogram (from libjxl enc_ans.cc:747-783).
/// More thorough than kFast (4 configs) but 7x more work. Iterates (value, count)
/// pairs from frequency maps instead of individual values, avoiding O(tokens) storage.
fn optimize_uint_configs_best_from_freqs(
    freqs_per_histo: &[alloc::collections::BTreeMap<u32, u32>],
    lz77: Option<&Lz77Params>,
) -> Vec<HybridUintConfig> {
    use crate::entropy_coding::ans::ANS_MAX_ALPHABET_SIZE;

    // EX-J26 (2026-05-28): tested porting libjxl's exact 28-candidate
    // set from `enc_ans.cc:748-774` for parity. Measurement on 50-file
    // paired A/B: +133 bytes (noise-level regression). Our existing
    // 28-candidate set is similarly effective — both converge to similar
    // optima per histogram via brute-force. Reverted, kept our set
    // (better-by-noise margin).
    #[rustfmt::skip]
    let candidates = [
        HybridUintConfig::new(0,0,0),  HybridUintConfig::new(1,0,0),
        HybridUintConfig::new(2,0,0),  HybridUintConfig::new(2,0,1),
        HybridUintConfig::new(3,0,0),  HybridUintConfig::new(3,1,0),
        HybridUintConfig::new(3,0,1),  HybridUintConfig::new(3,1,1),
        HybridUintConfig::new(4,0,0),  HybridUintConfig::new(4,2,0),
        HybridUintConfig::new(4,1,0),  HybridUintConfig::new(4,0,1),
        HybridUintConfig::new(4,2,1),  HybridUintConfig::new(4,1,1),
        HybridUintConfig::new(5,0,0),  HybridUintConfig::new(5,2,0),
        HybridUintConfig::new(5,1,0),  HybridUintConfig::new(5,0,1),
        HybridUintConfig::new(5,2,1),  HybridUintConfig::new(6,0,0),
        HybridUintConfig::new(6,2,0),  HybridUintConfig::new(6,1,0),
        HybridUintConfig::new(7,0,0),  HybridUintConfig::new(7,2,0),
        HybridUintConfig::new(8,0,0),  HybridUintConfig::new(8,2,0),
        HybridUintConfig::new(10,0,0), HybridUintConfig::new(12,0,0),
    ];

    let num_histograms = freqs_per_histo.len();
    let max_alpha = ANS_MAX_ALPHABET_SIZE;

    let mut best_configs = vec![HybridUintConfig::new(4, 2, 0); num_histograms];
    let mut counts_buf: Vec<u32> = Vec::new();

    for h in 0..num_histograms {
        let freqs = &freqs_per_histo[h];
        if freqs.is_empty() {
            continue;
        }

        let max_value = freqs.keys().copied().max().unwrap_or(0);
        let total: u32 = freqs.values().sum();
        let mut best_cost = f64::MAX;

        for &cfg in &candidates {
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

            let capacity = max_tok_with_lsb as usize + 1;
            counts_buf.clear();
            counts_buf.resize(capacity, 0);
            let mut extra_bits_total: u64 = 0;
            for (&val, &freq) in freqs {
                let (tok, _, nbits) = cfg.encode(val);
                counts_buf[tok as usize] += freq;
                extra_bits_total += nbits as u64 * freq as u64;
            }

            let inv_total = 1.0f32 / total as f32;
            let mut entropy_cost = 0.0f64;
            let mut nonzero_count: usize = 0;
            for &count in &counts_buf[..capacity] {
                if count > 0 {
                    nonzero_count += 1;
                    let c = count as f32;
                    entropy_cost -= c as f64 * jxl_simd::fast_log2f(c * inv_total) as f64;
                }
            }

            let signaling_cost = if cfg.split_exponent == 0 {
                0.0
            } else {
                ceil_log2_nonzero_usize(cfg.split_exponent as usize + 1) as f64
                    + ceil_log2_nonzero_usize((cfg.split_exponent - cfg.msb_in_token) as usize + 1)
                        as f64
            };
            // ANS distribution header cost — same fix as in
            // `optimize_uint_configs_fast_from_freqs`. See that function's
            // comment block for the libjxl-parity rationale. 28 candidates
            // in the kBest set include wide-alphabet configs like (12,0,0)
            // which are even more bloat-prone than the kFast (0,0,0).
            let header_cost = nonzero_count as f64 * 8.0;
            let cost = entropy_cost + extra_bits_total as f64 + signaling_cost + header_cost;

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
/// Matches libjxl's EncodeContextMap: always compares simple (raw bits) vs
/// non-simple (Huffman+MTF) and picks whichever is smaller. Previous code
/// unconditionally used simple for ≤8 histograms, which wastes bits when the
/// context map is large and repetitive (e.g. 1485 AC contexts with 8 histograms:
/// simple = 4455 bits, Huffman+MTF ≈ 800 bits).
fn write_context_map_for_ans(code: &OwnedAnsEntropyCode, writer: &mut BitWriter) -> Result<()> {
    let num_histograms = code.histograms.len();

    if num_histograms == 1 {
        // Simple context map: all contexts map to histogram 0
        writer.write(1, 1)?; // simple_context_map = true
        writer.write(2, 0)?; // nbits = 0
        return Ok(());
    }

    // Compute entry_bits for simple encoding: CeilLog2Nonzero(num_histograms)
    let entry_bits = ceil_log2_nonzero_usize(num_histograms);

    // Simple encoding is only possible when entry_bits < 4 (≤8 histograms).
    // When possible, compare simple vs non-simple and pick the cheaper one.
    // This matches libjxl enc_context_map.cc:113.
    if entry_bits < 4 {
        let simple_cost = 3 + entry_bits * code.context_map.len(); // 1 (is_simple) + 2 (nbits) + data

        // Write non-simple to a scratch writer to measure actual cost
        let mut scratch = BitWriter::with_capacity(code.context_map.len());
        write_context_map_nonsimple(&code.context_map, &mut scratch)?;
        let nonsimple_cost = scratch.bits_written();

        if simple_cost <= nonsimple_cost {
            // Simple is cheaper (or equal), use it
            writer.write(1, 1)?; // simple_context_map = true
            writer.write(2, entry_bits as u64)?;
            for &ctx in &code.context_map {
                writer.write(entry_bits, ctx as u64)?;
            }
            return Ok(());
        }
        // Non-simple is cheaper — copy the scratch bits
        let scratch_bytes = scratch.finish_with_padding();
        let bits_to_copy = nonsimple_cost;
        // Copy bit-by-bit from scratch to writer (scratch is byte-aligned but
        // writer may not be). Use the raw bytes and copy the exact bit count.
        copy_bits(&scratch_bytes, bits_to_copy, writer)?;
        return Ok(());
    }

    // > 8 histograms: always use non-simple
    write_context_map_nonsimple(&code.context_map, writer)
}

/// Write a non-simple context map. Picks between Huffman+MTF (legacy) and
/// ANS+LZ77 (libjxl-parity) based on actual byte cost.
///
/// libjxl's [`EncodeContextMap`](https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_context_map.cc)
/// (`enc_context_map.cc:65-139`) uses `BuildAndEncodeHistograms` with default
/// `HistogramParams` (`LZ77Method::kRLE` + `HybridUintMethod::kContextMap` →
/// `HybridUintConfig(2, 0, 1)`). On large repetitive maps (e.g. the 7425-entry
/// 15-cluster block context map default), the ANS+LZ77 path saves hundreds of
/// bytes per HfGlobal section vs Huffman+MTF.
///
/// Format (non-simple branch):
/// `is_simple=0` | `use_mtf` | `LZ77 header` | inner `entropy_code` | tokens.
///
/// The inner entropy code has 1 context (the context map itself). When LZ77 is
/// enabled, the inner Histograms decoder bumps that to 2 (LZ77 distance
/// context) and reads a 2-entry inner-inner context map.
fn write_context_map_nonsimple(context_map: &[u8], writer: &mut BitWriter) -> Result<()> {
    // Strategy 1: legacy Huffman+MTF, write to scratch and measure cost.
    let mut huffman_scratch = BitWriter::with_capacity(context_map.len());
    write_context_map_nonsimple_huffman(context_map, &mut huffman_scratch)?;
    let huffman_cost = huffman_scratch.bits_written();

    // Strategy 2: libjxl-parity ANS+LZ77, write to scratch and measure cost.
    // Wrap in Result so we can fall back to Huffman if ANS path errors
    // (e.g. degenerate input that exposes a histogram-builder edge case).
    let ans_lz77_scratch = build_context_map_nonsimple_ans_lz77(context_map).ok();

    let pick_ans = match &ans_lz77_scratch {
        Some(buf) => buf.bits_written() < huffman_cost,
        None => false,
    };

    if pick_ans {
        let buf = ans_lz77_scratch.expect("pick_ans true requires Some buffer");
        let bits_to_copy = buf.bits_written();
        let bytes = buf.finish_with_padding();
        copy_bits(&bytes, bits_to_copy, writer)?;
    } else {
        let bits_to_copy = huffman_cost;
        let bytes = huffman_scratch.finish_with_padding();
        copy_bits(&bytes, bits_to_copy, writer)?;
    }
    Ok(())
}

/// Legacy Huffman+MTF non-simple context map writer.
///
/// Format: `is_simple=0` | `use_mtf` | `lz77_enabled=0` | Huffman prefix code + data.
/// Retained as a fast fallback (and as one of the two candidates compared in
/// [`write_context_map_nonsimple`]).
fn write_context_map_nonsimple_huffman(context_map: &[u8], writer: &mut BitWriter) -> Result<()> {
    // Lever #1: trial-encode both direct and MTF candidates to a scratch
    // BitWriter and pick the strictly shorter one — matches libjxl
    // `enc_context_map.cc::EncodeContextMap`'s `BuildAndEncodeHistograms`
    // cost-measurement pass exactly.
    //
    // Previously this site used a Shannon entropy proxy
    // (`estimate_context_map_cost`) which ignores Huffman tree
    // serialisation overhead. On the 7425-entry libjxl-default 15-cluster
    // map the proxy is close enough that picks line up with reality, but
    // on smaller ctx_maps (typically 32-128 entries from cluster pair-merge
    // on a JPEG AC stream) the tree overhead is comparable to the data
    // bits, so the proxy can mispick. Direct trial-encoding closes the gap.
    let mtf_tokens = move_to_front_transform(context_map);

    let mut direct_scratch = BitWriter::with_capacity(context_map.len() + 16);
    let mut mtf_scratch = BitWriter::with_capacity(context_map.len() + 16);
    write_huffman_payload_no_selector(context_map, &mut direct_scratch)?;
    write_huffman_payload_no_selector(&mtf_tokens, &mut mtf_scratch)?;
    let direct_cost = direct_scratch.bits_written();
    let mtf_cost = mtf_scratch.bits_written();

    let use_mtf = mtf_cost < direct_cost;
    let tokens: &[u8] = if use_mtf { &mtf_tokens } else { context_map };

    // is_simple=0, use_mtf, lz77_enabled=0 (3 bits packed)
    let header_bits = if use_mtf { 0b010u64 } else { 0b000u64 };
    writer.write(3, header_bits)?;

    write_huffman_payload_no_selector(tokens, writer)
}

/// Write the post-selector Huffman payload for a non-simple context map:
/// `use_prefix_code=1` | `HybridUint config` | `alphabet size` | `prefix code tree` | tokens.
///
/// Both the trial-encoding cost measurement and the real emission share this
/// path so the cost comparison and the final bytes use the SAME bit-count
/// (no Shannon-proxy mismatch).
fn write_huffman_payload_no_selector(tokens: &[u8], writer: &mut BitWriter) -> Result<()> {
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

/// Build the ANS+LZ77 encoding of a non-simple context map into a scratch
/// [`BitWriter`]. Returns the buffer so the caller can compare its bit length
/// against the Huffman+MTF candidate.
///
/// Mirrors libjxl's `EncodeContextMap` (`enc_context_map.cc:77-137`):
///   - Build BOTH a raw-tokens stream and an MTF-tokens stream.
///   - Run LZ77-RLE on each, picking the cheaper encoded form.
///   - Force `HybridUintConfig(2, 0, 1)` per `HistogramParams::UintConfig()`
///     when `uint_method == kContextMap` (`enc_ans_params.h:81-90`).
///   - Write `is_simple=0`, `use_mtf`, then a full inner entropy code
///     (LZ77 header + 1-context map + ANS distributions) + tokens.
///
/// The decoder side calls [`Histograms::decode(1, br, allow_lz77 = num_contexts > 2)`](https://github.com/libjxl/libjxl/blob/main/lib/jxl/dec_ans.cc)
/// so we only emit the LZ77 header — never `lz77.enabled = true` when the
/// outer `num_contexts <= 2` (in which case `decode_context_map` would reject
/// it). That gate matches libjxl: `ApplyLZ77` skips LZ77 for streams whose
/// post-RLE token count is too short to make headers pay off, which is
/// effectively the same constraint for the tiny contexts.
pub(crate) fn build_context_map_nonsimple_ans_lz77(context_map: &[u8]) -> Result<BitWriter> {
    use super::ans::ANSHistogramStrategy;
    use super::lz77::apply_lz77_rle;

    // Outer allow_lz77 gate: jxl-rs `decode_context_map` calls
    // `Histograms::decode(1, br, allow_lz77 = num_contexts > 2)`. When the
    // outer num_contexts (= context_map.len()) is <= 2, we cannot signal
    // LZ77-enabled — the decoder would error with `Lz77Disallowed`.
    let lz77_allowed_outer = context_map.len() > 2;

    // Build both candidate token streams.
    let raw_tokens: Vec<Token> = context_map
        .iter()
        .map(|&v| Token::new(0, v as u32))
        .collect();
    let mtf_bytes = move_to_front_transform(context_map);
    let mtf_tokens: Vec<Token> = mtf_bytes.iter().map(|&v| Token::new(0, v as u32)).collect();

    let raw_cost = estimate_context_map_cost(context_map);
    let mtf_cost_est = estimate_context_map_cost(&mtf_bytes);
    let use_mtf = mtf_cost_est < raw_cost;
    let tokens: &[Token] = if use_mtf { &mtf_tokens } else { &raw_tokens };

    // Optionally apply LZ77-RLE. distance_multiplier=0 (no special distances
    // for a 1-D context map). num_contexts=1 (single context).
    let (final_tokens_owned, lz77_params) = if lz77_allowed_outer
        && let Some((lz_tokens, lz_params)) = apply_lz77_rle(
            tokens, /*num_contexts=*/ 1, /*force_huffman=*/ false, 0,
        ) {
        (Some(lz_tokens), Some(lz_params))
    } else {
        (None, None)
    };
    let final_tokens: &[Token] = final_tokens_owned.as_deref().unwrap_or(tokens);

    // Build a 1-context (literals only) or 2-context (literals + LZ77 distance)
    // ANS code over the (possibly LZ77-transformed) tokens. libjxl post-LZ77 has
    // num_contexts incremented by 1 for the distance context (`enc_ans.cc:1121`);
    // `apply_lz77_rle` uses `distance_context = num_contexts_pre_lz77 = 1`.
    let post_lz77_num_contexts = if lz77_params.is_some() { 2 } else { 1 };
    let mut code = build_entropy_code_ans_from_token_groups_with_strategy(
        &[final_tokens],
        post_lz77_num_contexts,
        /*enhanced_clustering=*/ false,
        /*optimize_uint_configs=*/ false,
        lz77_params.as_ref(),
        /*total_pixel_hint=*/ None,
        ANSHistogramStrategy::Precise,
    );

    // Override the HybridUint config with libjxl's kContextMap = (2, 0, 1).
    // The build path above set it to the default (4, 2, 0); re-normalize
    // each histogram against the kContextMap config so the bitstream encoding
    // matches.
    use super::ans::{ANSEncodingHistogram, AllowedCountsCache, AnsDistribution};
    use super::histogram::Histogram as EnhancedHistogram;
    let new_config = HybridUintConfig::new(2, 0, 1);
    let allowed_cache = AllowedCountsCache::new();
    let mut new_histograms = Vec::with_capacity(code.histograms.len());
    let mut new_distributions = Vec::with_capacity(code.distributions.len());

    // Re-accumulate per-histogram counts under the new uint config.
    // Since num_contexts == 1, all tokens map to histogram 0.
    let num_hist = code.histograms.len().max(1);
    let mut counts_per_hist: Vec<Vec<u32>> = vec![Vec::new(); num_hist];
    for tok in final_tokens {
        let cm_idx = code
            .context_map
            .get(tok.context() as usize)
            .copied()
            .unwrap_or(0) as usize;
        let sym = if tok.is_lz77_length() {
            let lz_params = lz77_params
                .as_ref()
                .expect("LZ77 length token requires lz77_params");
            let e = super::token::Lz77UintCoder::encode(tok.value);
            e.token + lz_params.min_symbol
        } else {
            let (t, _, _) = new_config.encode(tok.value);
            t
        };
        let counts = &mut counts_per_hist[cm_idx];
        if sym as usize >= counts.len() {
            counts.resize(sym as usize + 1, 0);
        }
        counts[sym as usize] += 1;
    }

    for counts in &counts_per_hist {
        let i32_counts: Vec<i32> = counts.iter().map(|&c| c as i32).collect();
        let histo = EnhancedHistogram::from_counts(&i32_counts);
        let ans_hist = ANSEncodingHistogram::from_histogram_cached(
            &histo,
            ANSHistogramStrategy::Precise,
            &allowed_cache,
        )?;
        new_histograms.push(ans_hist);
    }

    // Recompute global log_alpha_size after re-normalization.
    let max_alpha = new_histograms
        .iter()
        .map(|h| h.counts.len())
        .max()
        .unwrap_or(1);
    let log_alpha_size = if lz77_params.as_ref().is_some_and(|p| p.enabled) {
        8
    } else if max_alpha <= (1 << ANS_LOG_ALPHA_SIZE) {
        ANS_LOG_ALPHA_SIZE
    } else {
        let min_bits = if max_alpha <= 1 {
            5
        } else {
            (max_alpha - 1).ilog2() as usize + 1
        };
        min_bits.clamp(5, 8)
    };

    for h in &new_histograms {
        let dist =
            AnsDistribution::from_normalized_counts_with_log_alpha(&h.counts, log_alpha_size)?;
        new_distributions.push(dist);
    }
    code.histograms = new_histograms;
    code.distributions = new_distributions;
    code.log_alpha_size = log_alpha_size;
    code.uint_configs = vec![new_config; code.histograms.len()];

    // Now write to the scratch BitWriter:
    //
    //   is_simple=0 | use_mtf | LZ77 header
    //   [if outer-num_contexts (post-LZ77) > 1: inner context map]
    //   use_prefix_code=0 | log_alpha_size-5 | HybridUint config | histograms
    //   tokens
    //
    // Matches the decoder side `Histograms::decode(num_contexts=1, ..., allow_lz77)`
    // (`jxl-rs entropy_coding/decode.rs:576-634` + `libjxl dec_ans.cc:341-358`):
    // when post-LZ77 num_contexts == 1, the inner `decode_context_map` is
    // SKIPPED entirely (the decoder falls back to `vec![0]`). Writing the
    // 3-bit "simple, nbits=0" shortcut in that case would misalign every
    // subsequent bit.
    let mut scratch = BitWriter::with_capacity(context_map.len() * 2);
    scratch.write(1, 0)?; // is_simple = 0
    scratch.write(1, if use_mtf { 1 } else { 0 })?; // use_mtf

    super::lz77::write_lz77_header(lz77_params.as_ref(), &mut scratch)?;

    // Inner context map: ONLY when decoder will actually read it.
    let inner_num_contexts = post_lz77_num_contexts;
    if inner_num_contexts > 1 {
        write_context_map_for_ans(&code, &mut scratch)?;
    }

    // use_prefix_code = 0 (ANS)
    scratch.write(1, 0)?;
    let las = code.log_alpha_size;
    scratch.write(2, (las - 5) as u64)?;

    // HybridUint configs (one per histogram).
    for (i, _) in code.histograms.iter().enumerate() {
        let cfg = code.uint_configs.get(i).copied().unwrap_or_default();
        write_hybrid_uint_config_value(las, &cfg, &mut scratch)?;
    }

    // ANS distributions.
    for h in &code.histograms {
        h.write(&mut scratch)?;
    }

    // Tokens.
    write_tokens_ans(final_tokens, &code, lz77_params.as_ref(), &mut scratch)?;

    Ok(scratch)
}

/// Copy `num_bits` from a byte slice into a BitWriter.
pub(crate) fn copy_bits(src: &[u8], num_bits: usize, writer: &mut BitWriter) -> Result<()> {
    let full_bytes = num_bits / 8;
    let remaining_bits = num_bits % 8;

    for &byte in &src[..full_bytes] {
        writer.write(8, byte as u64)?;
    }
    if remaining_bits > 0 {
        let last_byte = src[full_bytes];
        let mask = (1u64 << remaining_bits) - 1;
        writer.write(remaining_bits, (last_byte as u64) & mask)?;
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
    let inv_total = 1.0f32 / tokens.len() as f32;
    let mut cost = 0.0f32;
    for &c in &counts {
        if c > 0 {
            let cf = c as f32;
            let p = cf * inv_total;
            cost -= p * jxl_simd::fast_log2f(p);
        }
    }
    (cost * tokens.len() as f32) as f64
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
        let ctx = token.context() as usize;
        let dist_idx = code.context_map.get(ctx).copied().unwrap_or(0) as usize;
        let config = code.uint_configs.get(dist_idx).copied().unwrap_or_default();
        let (encoded, sym) = encode_token_value_with_config(token, lz77, &config);

        // Get the distribution for this context. Out-of-range here is an
        // internal-consistency bug, not user-recoverable, but we still
        // surface it as a Result rather than panicking — a panic in this
        // code path would be a DoS vector if a malformed Lz77Params or
        // tokenization-bug ever produced a sym/ctx outside the distribution.
        let dist = code.distributions.get(dist_idx).ok_or_else(|| {
            crate::error::Error::InvalidInput(format!(
                "ANS internal: missing distribution at index {dist_idx} for context {ctx}"
            ))
        })?;

        // Push extra bits first (they come after the symbol in forward order)
        encoder.push_bits(encoded.bits, encoded.nbits as u8);

        // Push the ANS symbol
        let info = dist.get(sym as usize).ok_or_else(|| {
            crate::error::Error::InvalidInput(format!(
                "ANS internal: symbol {sym} not in distribution (ctx={ctx}, dist_idx={dist_idx})"
            ))
        })?;

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
///
/// Test-only invariant helper (exercised by the `verify_tests` module): re-arms
/// the Layer-2 histogram-serialization check from the omit_pos investigation.
/// Nothing in production calls it, so it is scoped to test builds.
#[cfg(test)]
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
        let ctx = token.context() as usize;
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
// Test-only invariant helper (exercised by the `verify_tests` module). Was
// `#[cfg(debug_assertions)]` during the omit_pos investigation; nothing in
// production calls it, so it is scoped to test builds.
#[cfg(test)]
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

#[cfg(test)]
mod verify_tests {
    use super::*;

    /// Re-arms the two ANS conformance invariants that lost their callers when
    /// the ad-hoc omit_pos investigation ended. Per CLAUDE.md "Proof-by-Tests
    /// Methodology" these are permanent invariant checks, not throwaway probes:
    ///
    ///  * [`verify_histogram_serialization`] — write each ANS histogram, decode
    ///    it back with our own decoder, and compare frequencies. This is the
    ///    exact Layer-2 check that pinpointed the omit_pos histogram-serialization
    ///    bug (see CLAUDE.md Investigation Notes).
    ///  * [`verify_ans_roundtrip_parsed`] — encode the tokens, then re-parse the
    ///    histogram out of the bitstream (the path a real decoder walks) and
    ///    decode, catching format mismatches our own encoder/decoder would
    ///    otherwise agree on.
    ///
    /// The distribution is deliberately non-degenerate: ~120 distinct values on
    /// a power-law-ish curve so many symbols share a `log2(count)` bucket (the
    /// condition under which omit_pos misbehaved — a flat gradient never
    /// triggers it), plus a few sparse high-value outliers so the token
    /// alphabet is not contiguous-dense. Full-photo ANS *quality* is covered by
    /// the decoder roundtrip integration tests; this test covers the
    /// serialization/roundtrip *machinery* itself.
    #[test]
    fn ans_verify_helpers_rearm() {
        let mut tokens = Vec::new();
        for v in 0u32..120 {
            // Varied counts; the +2 floor keeps low-frequency symbols present,
            // and integer division makes several adjacent values share a count.
            let count = (240 / (v + 3)) as usize + 2;
            for _ in 0..count {
                tokens.push(Token::new(0, v));
            }
        }
        for &v in &[200u32, 511, 1000, 4095] {
            tokens.push(Token::new(0, v));
        }

        // Single context => exactly one histogram, which
        // verify_ans_roundtrip_parsed requires.
        let code = build_entropy_code_ans(&tokens, 1);
        assert_eq!(
            code.histograms.len(),
            1,
            "expected a single ANS distribution"
        );

        verify_histogram_serialization(&code, "ans_verify_helpers_rearm")
            .expect("every ANS histogram must serialize and decode back bit-exact");

        verify_ans_roundtrip_parsed(&tokens, &code)
            .expect("parsed-header ANS roundtrip must reproduce every token");
    }
}
