// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Huffman encoder for JPEG XL.
//!
//! This module provides Huffman coding for small alphabets (prefix codes)
//! as used in various parts of the JXL bitstream.

use crate::bit_writer::BitWriter;
use crate::error::{Error, Result};

/// Maximum code length for Huffman codes in JXL.
pub const HUFFMAN_MAX_BITS: usize = 15;

/// Maximum alphabet size for Huffman coding.
pub const HUFFMAN_MAX_ALPHABET: usize = 1 << HUFFMAN_MAX_BITS;

/// A Huffman encoder for a specific alphabet.
pub struct HuffmanEncoder {
    /// Code lengths for each symbol.
    lengths: Vec<u8>,
    /// Actual bit codes for each symbol.
    codes: Vec<u16>,
}

impl HuffmanEncoder {
    /// Builds a Huffman encoder from symbol frequencies.
    ///
    /// # Arguments
    ///
    /// * `freqs` - Frequency of each symbol.
    /// * `max_bits` - Maximum allowed code length.
    pub fn from_frequencies(freqs: &[u32], max_bits: usize) -> Result<Self> {
        if freqs.len() > HUFFMAN_MAX_ALPHABET {
            return Err(Error::HuffmanTableTooLarge(freqs.len()));
        }

        let lengths = compute_code_lengths(freqs, max_bits)?;
        let codes = compute_codes_from_lengths(&lengths)?;

        Ok(Self { lengths, codes })
    }

    /// Builds a Huffman encoder from pre-computed code lengths.
    pub fn from_lengths(lengths: Vec<u8>) -> Result<Self> {
        let codes = compute_codes_from_lengths(&lengths)?;
        Ok(Self { lengths, codes })
    }

    /// Encodes a single symbol.
    #[inline]
    pub fn encode(&self, writer: &mut BitWriter, symbol: usize) -> Result<()> {
        let len = self.lengths[symbol] as usize;
        let code = self.codes[symbol] as u64;
        writer.write(len, code)
    }

    /// Returns the code length for a symbol.
    #[inline]
    pub fn code_length(&self, symbol: usize) -> u8 {
        self.lengths[symbol]
    }

    /// Returns the number of symbols in the alphabet.
    pub fn alphabet_size(&self) -> usize {
        self.lengths.len()
    }

    /// Writes the Huffman table to the bitstream.
    pub fn write_table(&self, _writer: &mut BitWriter) -> Result<()> {
        // TODO: Implement Huffman table serialization
        // JXL uses a specific format for encoding Huffman tables
        todo!("Huffman table serialization not yet implemented")
    }
}

/// Computes optimal Huffman code lengths using package-merge algorithm.
fn compute_code_lengths(freqs: &[u32], max_bits: usize) -> Result<Vec<u8>> {
    let n = freqs.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![1]); // Single symbol needs 1 bit
    }

    // Count non-zero frequencies
    let non_zero: Vec<(u32, usize)> = freqs
        .iter()
        .enumerate()
        .filter(|&(_, f)| *f > 0)
        .map(|(i, f)| (*f, i))
        .collect();

    if non_zero.is_empty() {
        return Ok(vec![0; n]);
    }

    if non_zero.len() == 1 {
        let mut lengths = vec![0; n];
        lengths[non_zero[0].1] = 1;
        return Ok(lengths);
    }

    // For now, use a simple length-limited Huffman algorithm
    // TODO: Implement full package-merge for optimal codes
    let mut lengths = vec![0u8; n];

    // Simple heuristic: assign lengths based on frequency ranking
    let mut sorted = non_zero.clone();
    sorted.sort_by(|a, b| b.0.cmp(&a.0)); // Sort by frequency descending

    let num_symbols = sorted.len();
    for (rank, (_, symbol)) in sorted.into_iter().enumerate() {
        // Assign length based on rank (this is a placeholder)
        let length = ((rank * max_bits) / num_symbols + 1).min(max_bits);
        lengths[symbol] = length as u8;
    }

    // Verify and adjust lengths to satisfy Kraft inequality
    // sum(2^-length) <= 1
    adjust_for_kraft(&mut lengths, max_bits)?;

    Ok(lengths)
}

/// Adjusts code lengths to satisfy the Kraft inequality.
fn adjust_for_kraft(lengths: &mut [u8], max_bits: usize) -> Result<()> {
    // Kraft sum: sum(2^(max_bits - length)) should equal 2^max_bits
    let target = 1u64 << max_bits;

    loop {
        let kraft_sum: u64 = lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (max_bits - l as usize))
            .sum();

        if kraft_sum == target {
            return Ok(());
        } else if kraft_sum < target {
            // Under-full: decrease some lengths (make codes shorter)
            // Find the longest code and shorten it
            if let Some(idx) = lengths
                .iter()
                .enumerate()
                .filter(|&(_, &l)| l > 1)
                .max_by_key(|&(_, &l)| l)
                .map(|(i, _)| i)
            {
                lengths[idx] -= 1;
            } else {
                break;
            }
        } else {
            // Over-full: increase some lengths (make codes longer)
            if let Some(idx) = lengths
                .iter()
                .enumerate()
                .filter(|&(_, &l)| l > 0 && (l as usize) < max_bits)
                .min_by_key(|&(_, &l)| l)
                .map(|(i, _)| i)
            {
                lengths[idx] += 1;
            } else {
                return Err(Error::InvalidHuffmanCodeLengths);
            }
        }
    }

    Ok(())
}

/// Computes canonical Huffman codes from code lengths.
fn compute_codes_from_lengths(lengths: &[u8]) -> Result<Vec<u16>> {
    let n = lengths.len();
    let mut codes = vec![0u16; n];

    if n == 0 {
        return Ok(codes);
    }

    // Count codes of each length
    let max_len = *lengths.iter().max().unwrap_or(&0) as usize;
    if max_len > 15 {
        return Err(Error::InvalidHuffmanCodeLengths);
    }

    let mut bl_count = vec![0u32; max_len + 1];
    for &len in lengths {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    // Compute starting code for each length
    let mut next_code = vec![0u16; max_len + 1];
    let mut code = 0u16;
    for bits in 1..=max_len {
        code = (code + bl_count[bits - 1] as u16) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols
    for (symbol, &len) in lengths.iter().enumerate() {
        if len > 0 {
            codes[symbol] = next_code[len as usize];
            next_code[len as usize] += 1;
        }
    }

    Ok(codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_codes() {
        // Example: lengths [2, 1, 3, 3]
        // Expected codes: A=10, B=0, C=110, D=111
        let lengths = vec![2, 1, 3, 3];
        let codes = compute_codes_from_lengths(&lengths).unwrap();

        assert_eq!(codes[0], 0b10); // A: 2 bits
        assert_eq!(codes[1], 0b0); // B: 1 bit
        assert_eq!(codes[2], 0b110); // C: 3 bits
        assert_eq!(codes[3], 0b111); // D: 3 bits
    }

    #[test]
    fn test_encoder_from_frequencies() {
        let freqs = vec![10, 5, 1, 1];
        let encoder = HuffmanEncoder::from_frequencies(&freqs, 15).unwrap();

        // All symbols should have valid (non-zero) code lengths
        assert!(encoder.code_length(0) > 0);
        assert!(encoder.code_length(1) > 0);
        assert!(encoder.code_length(2) > 0);
        assert!(encoder.code_length(3) > 0);

        // Code lengths should be <= max_bits
        assert!(encoder.code_length(0) as usize <= 15);
    }

    #[test]
    fn test_single_symbol() {
        let freqs = vec![0, 10, 0, 0];
        let encoder = HuffmanEncoder::from_frequencies(&freqs, 15).unwrap();

        // Only symbol 1 has non-zero frequency
        assert_eq!(encoder.code_length(1), 1);
    }
}
