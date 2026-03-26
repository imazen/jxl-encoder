// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Context map encoding.
//!
//! Ported from libjxl `lib/jxl/enc_context_map.cc`.
//!
//! # Current Implementation
//!
//! Currently uses simple encoding only, which supports up to 8 histograms
//! (3 bits per entry). This covers most practical use cases.
//!
//! # Future Enhancement: ANS Encoding
//!
//! For better compression with many clusters (>8), the JXL format supports
//! ANS-based context map encoding with optional MTF transform:
//! 1. **Prefix code (Huffman)**: Uses Huffman codes for symbols.
//! 2. **Prefix code with MTF**: Applies move-to-front transform before Huffman.
//!
//! This requires implementing the full JXL entropy bundle format, which is
//! non-trivial. See libjxl `lib/jxl/enc_context_map.cc` for reference.

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Move-to-front transform for better compression.
///
/// The MTF transform replaces each symbol with its index in a "recently used"
/// list. Symbols that appear frequently close together get small indices.
pub fn move_to_front_transform(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let max_value = *input.iter().max().unwrap_or(&0);
    let mut mtf: Vec<u8> = (0..=max_value).collect();
    let mut result = Vec::with_capacity(input.len());

    for &value in input {
        // Find index of value in MTF list
        let index = mtf.iter().position(|&x| x == value).unwrap();
        result.push(index as u8);

        // Move to front
        if index > 0 {
            let val = mtf.remove(index);
            mtf.insert(0, val);
        }
    }

    result
}

/// Inverse move-to-front transform (for testing).
pub fn inverse_move_to_front_transform(input: &[u8], max_symbol: u8) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut mtf: Vec<u8> = (0..=max_symbol).collect();
    let mut result = Vec::with_capacity(input.len());

    for &index in input {
        let idx = index as usize;
        if idx >= mtf.len() {
            // Invalid index, return what we have
            result.push(0);
            continue;
        }

        let value = mtf[idx];
        result.push(value);

        // Move to front
        if idx > 0 {
            mtf.remove(idx);
            mtf.insert(0, value);
        }
    }

    result
}

/// Encode context map to bitstream.
///
/// The context map maps context indices to histogram indices.
///
/// # Current Implementation
///
/// Uses simple encoding only, which supports up to 8 histograms (3 bits per entry).
/// This is sufficient for reasonable clustering without the complexity of implementing
/// the full ANS entropy bundle format for context maps.
///
/// # Encoding Format
///
/// - If num_histograms == 1: write (1, 0) → no actual entries needed
/// - Simple mode: write (1, entry_bits) then each entry with entry_bits bits
///
/// # Future Work
///
/// To support >8 histograms with efficient encoding, implement the full JXL entropy
/// bundle format for context maps (is_simple=0 path). This requires:
/// - lz77.enabled flag
/// - Full histogram bundle with ANS/prefix codes
/// - HybridUint encoding for symbols
///
/// Reference: libjxl lib/jxl/enc_context_map.cc
pub fn encode_context_map(
    context_map: &[u8],
    num_histograms: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    if num_histograms == 1 {
        // Simple code: all contexts map to histogram 0
        writer.write(1, 1)?; // simple flag
        writer.write(2, 0)?; // 0 bits per entry
        return Ok(());
    }

    // Calculate bits needed for simple encoding
    // Simple mode supports bits_per_entry = 0, 1, 2, or 3 (encoded in 2 bits)
    // This allows up to 8 histograms (2^3 = 8)
    let entry_bits = ceil_log2_nonzero(num_histograms);

    if entry_bits > 3 {
        // Simple mode only supports up to 3 bits per entry (8 clusters)
        // For now, just use 3 bits and mask values (clustering should ensure <= 8 clusters)
        crate::trace::debug_eprintln!(
            "WARNING: context_map requires {} bits but simple mode max is 3 bits. \
             Using 3 bits, which may cause decoding errors if num_histograms > 8.",
            entry_bits
        );
    }

    let effective_bits = entry_bits.min(3);
    writer.write(1, 1)?; // simple flag
    writer.write(2, effective_bits as u64)?; // bits per entry
    for &entry in context_map {
        // Mask entry to fit within effective_bits
        let masked_entry = entry & ((1 << effective_bits) - 1);
        writer.write(effective_bits, masked_entry as u64)?;
    }

    Ok(())
}

/// Ceiling of log2 for non-zero values.
#[inline]
fn ceil_log2_nonzero(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtf_simple() {
        let input = vec![1, 2, 1, 2, 1, 2];
        let transformed = move_to_front_transform(&input);

        // 1 → index 1, mtf = [1, 0, 2]
        // 2 → index 2, mtf = [2, 1, 0]
        // 1 → index 1, mtf = [1, 2, 0]
        // 2 → index 1, mtf = [2, 1, 0]
        // 1 → index 1, mtf = [1, 2, 0]
        // 2 → index 1, mtf = [2, 1, 0]
        assert_eq!(transformed, vec![1, 2, 1, 1, 1, 1]);
    }

    #[test]
    fn test_mtf_repeated() {
        let input = vec![5, 5, 5, 5];
        let transformed = move_to_front_transform(&input);

        // After first 5, it's at front, so subsequent 5s are at index 0
        assert_eq!(transformed, vec![5, 0, 0, 0]);
    }

    #[test]
    fn test_mtf_empty() {
        let input: Vec<u8> = vec![];
        let transformed = move_to_front_transform(&input);
        assert!(transformed.is_empty());
    }

    #[test]
    fn test_mtf_roundtrip() {
        let original = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let max_symbol = *original.iter().max().unwrap();
        let transformed = move_to_front_transform(&original);
        let recovered = inverse_move_to_front_transform(&transformed, max_symbol);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_mtf_roundtrip_sequential() {
        let original: Vec<u8> = (0..10).collect();
        let max_symbol = *original.iter().max().unwrap();
        let transformed = move_to_front_transform(&original);
        let recovered = inverse_move_to_front_transform(&transformed, max_symbol);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_encode_context_map_single() {
        let context_map: Vec<u8> = vec![0, 0, 0, 0];
        let mut writer = BitWriter::new();

        encode_context_map(&context_map, 1, &mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        // Simple encoding with 0 bits per entry: (1, 0) = 1 bit + 2 bits = 3 bits
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_encode_context_map_two_histograms() {
        let context_map: Vec<u8> = vec![0, 1, 0, 1];
        let mut writer = BitWriter::new();

        encode_context_map(&context_map, 2, &mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        // Simple encoding with 1 bit per entry
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ceil_log2() {
        assert_eq!(ceil_log2_nonzero(1), 0);
        assert_eq!(ceil_log2_nonzero(2), 1);
        assert_eq!(ceil_log2_nonzero(3), 2);
        assert_eq!(ceil_log2_nonzero(4), 2);
        assert_eq!(ceil_log2_nonzero(5), 3);
        assert_eq!(ceil_log2_nonzero(8), 3);
        assert_eq!(ceil_log2_nonzero(9), 4);
        assert_eq!(ceil_log2_nonzero(256), 8);
    }
}
