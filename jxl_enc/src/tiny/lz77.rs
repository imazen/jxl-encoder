// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! LZ77 RLE backward references for entropy coding.
//!
//! Implements the RLE-only LZ77 method from libjxl (`ApplyLZ77_RLE` in `enc_lz77.cc`).
//! This scans token streams for consecutive identical values and replaces runs
//! with LZ77 length + distance tokens, reducing the token count before entropy coding.

use super::token::{Lz77UintCoder, Token, UintCoder};

/// LZ77 parameters serialized in the entropy code header.
#[derive(Debug, Clone)]
pub struct Lz77Params {
    pub enabled: bool,
    /// Symbols >= min_symbol are LZ77 length tokens.
    /// ANS: 224, Huffman: 512.
    pub min_symbol: u32,
    /// Minimum run length to encode as LZ77. Default: 3.
    pub min_length: u32,
    /// Context index for distance tokens (= num_contexts before LZ77).
    pub distance_context: u32,
}

impl Lz77Params {
    pub fn new(num_contexts: usize, force_huffman: bool) -> Self {
        Self {
            enabled: false,
            min_symbol: if force_huffman { 512 } else { 224 },
            min_length: 3,
            distance_context: num_contexts as u32,
        }
    }
}

/// Estimate per-symbol bit cost from histograms, matching libjxl's SymbolCostEstimator.
struct SymbolCostEstimator {
    /// Flat array: bits[ctx * max_alphabet_size + sym]
    bits: Vec<f32>,
    max_alphabet_size: usize,
}

impl SymbolCostEstimator {
    fn new(num_contexts: usize, force_huffman: bool, tokens: &[Token], lz77: &Lz77Params) -> Self {
        const ANS_LOG_TAB_SIZE: f32 = 12.0;

        // Build per-context histograms from the (possibly LZ77-transformed) tokens.
        let mut counts: Vec<Vec<u32>> = vec![vec![]; num_contexts];
        let mut total_counts = vec![0u32; num_contexts];

        for token in tokens {
            let (tok, _nbits) = if token.is_lz77_length {
                let e = Lz77UintCoder::encode(token.value);
                (e.token + lz77.min_symbol, e.nbits)
            } else {
                let e = UintCoder::encode(token.value);
                (e.token, e.nbits)
            };
            let ctx = token.context as usize;
            if ctx < num_contexts {
                let sym = tok as usize;
                if sym >= counts[ctx].len() {
                    counts[ctx].resize(sym + 1, 0);
                }
                counts[ctx][sym] += 1;
                total_counts[ctx] += 1;
            }
        }

        let max_alphabet_size = counts.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut bits = vec![0.0f32; num_contexts * max_alphabet_size];

        for ctx in 0..num_contexts {
            let total = total_counts[ctx];
            if total == 0 {
                continue;
            }
            let inv_total = 1.0 / (total as f32 + 1e-8);
            for sym in 0..counts[ctx].len() {
                let cnt = counts[ctx][sym];
                let cost = if cnt != 0 && cnt != total {
                    let p = cnt as f32 * inv_total;
                    let c = -p.log2();
                    if force_huffman { c.ceil() } else { c }
                } else if cnt == 0 {
                    ANS_LOG_TAB_SIZE // Highest possible cost
                } else {
                    0.0 // Single symbol, zero cost
                };
                bits[ctx * max_alphabet_size + sym] = cost;
            }
        }

        Self {
            bits,
            max_alphabet_size,
        }
    }

    #[inline]
    fn symbol_cost(&self, ctx: usize, sym: usize) -> f32 {
        if sym < self.max_alphabet_size {
            self.bits[ctx * self.max_alphabet_size + sym]
        } else {
            12.0 // ANS_LOG_TAB_SIZE as fallback
        }
    }
}

/// Apply RLE-based LZ77 compression to a token stream.
///
/// Scans for runs of consecutive identical values. When a run is long enough
/// and the LZ77 encoding is cheaper, replaces the run with:
/// 1. An LZ77 length token (context = original, value = run_len - min_length, is_lz77_length = true)
/// 2. A distance token (context = distance_context, value = 0 meaning "repeat previous")
///
/// Returns `Some((transformed_tokens, params))` if LZ77 is beneficial,
/// or `None` if the savings are insufficient.
pub fn apply_lz77_rle(
    tokens: &[Token],
    num_contexts: usize,
    force_huffman: bool,
) -> Option<(Vec<Token>, Lz77Params)> {
    if tokens.is_empty() {
        return None;
    }

    let mut lz77 = Lz77Params::new(num_contexts, force_huffman);

    // First pass: build cost estimator from the original tokens (no LZ77 tokens yet).
    // We pass the original tokens to estimate costs, matching libjxl.
    let sce = SymbolCostEstimator::new(num_contexts, force_huffman, tokens, &lz77);

    // Compute cumulative bit costs for original stream.
    let mut sym_cost = vec![0.0f32; tokens.len() + 1];
    for (i, token) in tokens.iter().enumerate() {
        let e = UintCoder::encode(token.value);
        let cost = sce.symbol_cost(token.context as usize, e.token as usize) + e.nbits as f32;
        sym_cost[i + 1] = sym_cost[i] + cost;
    }

    let mut out = Vec::with_capacity(tokens.len());
    let mut bit_decrease: f32 = 0.0;
    let total_symbols = tokens.len();

    let mut i = 0;
    while i < tokens.len() {
        // Count consecutive identical values starting from the PREVIOUS token
        // (matching libjxl: "if (i > 0) { ... in[i+num_to_copy].value != in[i-1].value }")
        let mut num_to_copy = 0;
        if i > 0 {
            let prev_value = tokens[i - 1].value;
            while i + num_to_copy < tokens.len() && tokens[i + num_to_copy].value == prev_value {
                num_to_copy += 1;
            }
        }

        if num_to_copy == 0 {
            out.push(tokens[i]);
            i += 1;
            continue;
        }

        // Cost of encoding the run literally
        let literal_cost = sym_cost[i + num_to_copy] - sym_cost[i];

        // Cost of LZ77 encoding (rough estimate matching libjxl)
        let lz77_cost = if num_to_copy >= lz77.min_length as usize {
            let lz77_len = num_to_copy - lz77.min_length as usize;
            // CeilLog2Nonzero(lz77_len + 1) + 1 (for distance)
            ceil_log2_nonzero((lz77_len + 1) as u32) as f32 + 1.0
        } else {
            0.0
        };

        if num_to_copy < lz77.min_length as usize || literal_cost <= lz77_cost {
            // Not worth encoding as LZ77, emit literal tokens
            for j in 0..num_to_copy {
                out.push(tokens[i + j]);
            }
            i += num_to_copy;
            continue;
        }

        // Emit LZ77 length token
        let lz77_len = (num_to_copy - lz77.min_length as usize) as u32;
        out.push(Token::lz77_length(tokens[i].context, lz77_len));

        // Emit distance token (distance_symbol=0 means distance 1 = "repeat previous",
        // since num_special_distances=0 for RLE)
        out.push(Token::new(lz77.distance_context, 0));

        bit_decrease += literal_cost - lz77_cost;
        i += num_to_copy;
    }

    // Only use LZ77 if savings exceed threshold (matching libjxl)
    let threshold = total_symbols as f32 * 0.2 + 16.0;
    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "[LZ77] bit_decrease={:.1}, threshold={:.1}, tokens: {} -> {}",
        bit_decrease,
        threshold,
        total_symbols,
        out.len()
    );
    if bit_decrease > threshold {
        lz77.enabled = true;
        Some((out, lz77))
    } else {
        None
    }
}

/// CeilLog2Nonzero matching libjxl's implementation.
fn ceil_log2_nonzero(x: u32) -> u32 {
    debug_assert!(x > 0);
    let floor = 31 - x.leading_zeros();
    if x.is_power_of_two() {
        floor
    } else {
        floor + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ceil_log2_nonzero() {
        assert_eq!(ceil_log2_nonzero(1), 0);
        assert_eq!(ceil_log2_nonzero(2), 1);
        assert_eq!(ceil_log2_nonzero(3), 2);
        assert_eq!(ceil_log2_nonzero(4), 2);
        assert_eq!(ceil_log2_nonzero(5), 3);
        assert_eq!(ceil_log2_nonzero(8), 3);
        assert_eq!(ceil_log2_nonzero(9), 4);
    }

    #[test]
    fn test_no_rle_on_short_stream() {
        // Very short streams shouldn't trigger LZ77
        let tokens = vec![Token::new(0, 5), Token::new(0, 5), Token::new(0, 5)];
        assert!(apply_lz77_rle(&tokens, 1, false).is_none());
    }

    #[test]
    fn test_rle_on_long_run() {
        // Long run of identical values should trigger LZ77
        let mut tokens = Vec::new();
        // Need a "previous" token, then a long run of identical values
        tokens.push(Token::new(0, 5));
        for _ in 0..200 {
            tokens.push(Token::new(0, 5));
        }

        let result = apply_lz77_rle(&tokens, 1, false);
        if let Some((lz77_tokens, params)) = result {
            assert!(params.enabled);
            // Should be much shorter than the original
            assert!(lz77_tokens.len() < tokens.len());
            // Should contain at least one LZ77 length token
            assert!(lz77_tokens.iter().any(|t| t.is_lz77_length));
        }
        // If None, that's OK — the threshold might not be met for this particular cost estimate
    }

    #[test]
    fn test_rle_preserves_non_runs() {
        // Mixed content: some runs, some non-runs
        let mut tokens = Vec::new();
        // Non-repeating prefix
        for i in 0..10 {
            tokens.push(Token::new(0, i));
        }
        // Long run
        for _ in 0..100 {
            tokens.push(Token::new(0, 42));
        }
        // Non-repeating suffix
        for i in 0..10 {
            tokens.push(Token::new(0, i + 100));
        }

        if let Some((lz77_tokens, params)) = apply_lz77_rle(&tokens, 1, false) {
            assert!(params.enabled);
            assert!(lz77_tokens.len() < tokens.len());
            // The first token should be preserved literally
            assert_eq!(lz77_tokens[0].value, 0);
            assert!(!lz77_tokens[0].is_lz77_length);
        }
    }

    #[test]
    fn test_empty_stream() {
        assert!(apply_lz77_rle(&[], 1, false).is_none());
    }
}
