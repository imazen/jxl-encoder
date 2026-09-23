// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Entropy code types, shared helpers, and re-exports.
//!
//! Huffman-specific code lives in `encode_huffman`, ANS-specific code in `encode_ans`.
//! All public items are re-exported here so consumers can continue using
//! `crate::entropy_coding::encode::*`.

use super::hybrid_uint::HybridUintConfig;
use super::lz77::Lz77Params;
use super::token::{Lz77UintCoder, Token, UintCoder};
use crate::bit_writer::BitWriter;
use crate::error::Result;

use super::token::EncodedUint;

// Re-export everything from the Huffman and ANS sub-modules so existing
// `use crate::entropy_coding::encode::*` paths continue to work.
pub use super::encode_ans::*;
pub use super::encode_huffman::*;

/// Encode a token's value, using the LZ77 uint config when `is_lz77_length` is set.
/// Returns (encoded_uint, symbol_for_histogram) where symbol_for_histogram includes
/// the min_symbol offset for LZ77 length tokens.
#[inline]
pub(super) fn encode_token_value(token: &Token, lz77: Option<&Lz77Params>) -> (EncodedUint, u32) {
    if token.is_lz77_length() {
        let lz77 = lz77.expect("LZ77 length token without LZ77 params");
        let encoded = Lz77UintCoder::encode(token.value);
        let sym = encoded.token + lz77.min_symbol;
        (encoded, sym)
    } else {
        let encoded = UintCoder::encode(token.value);
        (encoded, encoded.token)
    }
}

/// Encode a token's value using a specific HybridUint config (for per-histogram configs).
#[inline]
/// libjxl `enc_ans.cc` prefix-vs-ANS auto choice (the
/// `initialize_global_state` heuristic): prefix codes win when the
/// stream is tiny (`total_tokens < 100`) or when every context is
/// deterministic ("all_singleton" — per-context Shannon entropy below
/// 1e-5, which for real token streams means one distinct hybrid-uint
/// symbol per context). A singleton prefix symbol costs 0 bits and a
/// prefix stream carries no 32-bit ANS final state, so a
/// fully-deterministic section costs 0 bytes — cjxl emits literal
/// 0-byte GroupPass sections on smooth content this way, while an ANS
/// stream pays the 4-byte state flush per section regardless.
///
/// Symbols are mapped with the default `HybridUintConfig` and no LZ77,
/// mirroring libjxl which runs the heuristic before per-context config
/// optimization. Callers must keep ANS for streams that carry LZ77
/// params (our LZ77 writer is ANS-only).
pub fn prefix_beats_ans_for_token_groups(groups: &[&[Token]], num_contexts: usize) -> bool {
    let total: usize = groups.iter().map(|g| g.len()).sum();
    if total < 100 {
        return true;
    }
    let mut seen: Vec<Option<u32>> = alloc::vec![None; num_contexts];
    for g in groups {
        for t in g.iter() {
            let (_, sym) = encode_token_value(t, None);
            let Some(slot) = seen.get_mut(t.context() as usize) else {
                // Out-of-range context — internal inconsistency; keep ANS.
                return false;
            };
            match *slot {
                None => *slot = Some(sym),
                Some(s) if s == sym => {}
                Some(_) => return false,
            }
        }
    }
    true
}

pub(super) fn encode_token_value_with_config(
    token: &Token,
    lz77: Option<&Lz77Params>,
    config: &HybridUintConfig,
) -> (EncodedUint, u32) {
    if token.is_lz77_length() {
        let lz77 = lz77.expect("LZ77 length token without LZ77 params");
        let encoded = Lz77UintCoder::encode(token.value);
        let sym = encoded.token + lz77.min_symbol;
        (encoded, sym)
    } else {
        let (tok, bits, nbits) = config.encode(token.value);
        let encoded = EncodedUint {
            token: tok,
            nbits,
            bits,
        };
        (encoded, tok)
    }
}

/// Number of code length codes used in Huffman tree serialization.
pub(super) const CODE_LENGTH_CODES: usize = 18;

/// Maximum number of symbols in the Huffman alphabet.
pub const ALPHABET_SIZE: usize = 64;

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

/// HybridUint config selection for entropy-code building.
///
/// Mirrors libjxl `HistogramParams::HybridUintMethod` (`enc_ans_params.h`):
/// streams built with default params (permutation codes, MA-tree codes)
/// get `Best`; the VarDCT modular DC code gets `Fast` when
/// `extra_dc_precision != 0` (`ForModular`); AC streams get `None` below
/// kTortoise and `Best` there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UintConfigMethod {
    /// Fixed {4,2,0} — libjxl `kNone`.
    #[default]
    None,
    /// 4-candidate optimization — libjxl `kFast`.
    Fast,
    /// Full candidate sweep — libjxl `kBest`.
    Best,
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
    /// Per-prefix-code HybridUint configs (empty = default {4,2,0} for all).
    ///
    /// libjxl runs `ChooseUintConfigs` for prefix streams exactly like ANS
    /// (`enc_ans.cc` — the uint config is serialized regardless of coder
    /// type). Streams built under libjxl-parity populate this with the
    /// optimized per-histogram configs.
    pub uint_configs: &'a [HybridUintConfig],
    /// When true, a nested non-simple context map is coded with libjxl's
    /// `log_alpha_size` convention (default 7, refined only by adaptive
    /// uint methods) instead of the historical fixed 6.
    pub libjxl_log_alpha: bool,
}

impl<'a> EntropyCode<'a> {
    /// Create a new entropy code from static tables.
    pub const fn new(context_map: &'a [u8], prefix_codes: &'a [PrefixCode]) -> Self {
        Self {
            context_map,
            num_contexts: context_map.len(),
            prefix_codes,
            num_prefix_codes: prefix_codes.len(),
            uint_configs: &[],
            libjxl_log_alpha: false,
        }
    }

    /// Attach per-prefix-code HybridUint configs (one per entry in
    /// `prefix_codes`).
    pub fn with_uint_configs(mut self, uint_configs: &'a [HybridUintConfig]) -> Self {
        self.uint_configs = uint_configs;
        self
    }

    /// Mark the nested context-map stream for libjxl `log_alpha_size`
    /// handling (strict-parity callers only).
    pub fn with_libjxl_log_alpha(mut self, libjxl_log_alpha: bool) -> Self {
        self.libjxl_log_alpha = libjxl_log_alpha;
        self
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
    let prefix_idx = code.context_map[token.context() as usize] as usize;
    let (encoded, sym) = match code.uint_configs.get(prefix_idx) {
        Some(cfg) => encode_token_value_with_config(token, lz77, cfg),
        None => encode_token_value(token, lz77),
    };

    let pc = &code.prefix_codes[prefix_idx];

    // Get the Huffman code for this token. A token beyond the table's
    // alphabet used to index-panic here; a codeless symbol (depth 0 in a
    // multi-symbol code) silently wrote only the extra bits — the zenjpeg
    // #194 zero-bit mechanism (sweep issue #97). Both are loud errors now.
    let tok = sym as usize;
    let Some(&depth_u8) = pc.depths.get(tok) else {
        return Err(crate::error::Error::InvalidInput(alloc::format!(
            "token symbol {tok} exceeds the prefix-code alphabet"
        )));
    };
    let depth = depth_u8 as usize;
    if depth == 0 && !crate::entropy_coding::encode_huffman::has_single_used_symbol(&pc.depths) {
        return Err(crate::error::Error::InvalidInput(alloc::format!(
            "prefix code has no bits for symbol {tok} — emitting it would \
             silently corrupt the stream"
        )));
    }
    let bits = pc.bits[tok] as u64;

    // Combine Huffman bits and extra bits
    let data = bits | ((encoded.bits as u64) << depth);
    let total_bits = depth + encoded.nbits as usize;

    writer.write(total_bits, data)
}

/// Write VarLenUint16 encoding (0-65535).
pub(super) fn write_var_len_uint16(n: usize, writer: &mut BitWriter) -> Result<()> {
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
pub(super) fn floor_log2_nonzero(n: u32) -> u32 {
    debug_assert!(n > 0);
    31 - n.leading_zeros()
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
