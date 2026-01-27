// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Entropy code types and token writing.
//!
//! These are ported from libjxl-tiny and will be used for Huffman coding.

#![allow(dead_code)]

use super::token::{Token, UintCoder};
use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Maximum number of symbols in the Huffman alphabet.
pub const ALPHABET_SIZE: usize = 64;

/// Maximum number of contexts.
pub const MAX_CONTEXTS: usize = 128;

/// Maximum bits per token (upper bound for allotment).
pub const MAX_BITS_PER_TOKEN: usize = 24;

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

/// An entropy code consisting of context map and prefix codes.
#[derive(Debug)]
pub struct EntropyCode<'a> {
    /// Context map: maps context ID -> prefix code index.
    pub context_map: &'a [u8],
    /// Number of contexts.
    pub num_contexts: usize,
    /// Prefix codes (Huffman codes).
    pub prefix_codes: &'a [PrefixCode],
    /// Number of prefix codes.
    pub num_prefix_codes: usize,
}

impl<'a> EntropyCode<'a> {
    /// Create a new entropy code from static tables.
    pub const fn new(context_map: &'a [u8], prefix_codes: &'a [PrefixCode]) -> Self {
        Self {
            context_map,
            num_contexts: context_map.len(),
            prefix_codes,
            num_prefix_codes: prefix_codes.len(),
        }
    }
}

/// Write a token using the given entropy code.
///
/// This encodes the value using the UintCoder, looks up the prefix code
/// via the context map, and writes the Huffman code followed by extra bits.
#[inline]
pub fn write_token(token: &Token, code: &EntropyCode, writer: &mut BitWriter) -> Result<()> {
    let encoded = UintCoder::encode(token.value);

    // Look up the prefix code index from the context map
    let prefix_idx = code.context_map[token.context as usize] as usize;
    let pc = &code.prefix_codes[prefix_idx];

    // Get the Huffman code for this token
    let tok = encoded.token as usize;
    let depth = pc.depths[tok] as usize;
    let bits = pc.bits[tok] as u64;

    // Combine Huffman bits and extra bits
    let data = bits | ((encoded.bits as u64) << depth);
    let total_bits = depth + encoded.nbits as usize;

    writer.write(total_bits, data)
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
