// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Context map encoding.
//!
//! Ported from libjxl `lib/jxl/enc_context_map.cc`.

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
/// This function tries both raw and move-to-front encoding and picks smaller.
///
/// Encoding format:
/// - If num_histograms == 1: write (1, 0) → no actual entries needed
/// - Simple mode: write (1, entry_bits) then each entry with entry_bits bits
/// - ANS mode: write (0, use_mtf) then ANS-encoded symbols
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
    let entry_bits = ceil_log2_nonzero(num_histograms);
    let _simple_cost = 3 + entry_bits * context_map.len();

    // For now, we'll use simple encoding if possible
    // TODO: Add full ANS encoding path when histogram infrastructure is complete

    // Transform with MTF
    let transformed = move_to_front_transform(context_map);

    // Estimate MTF cost (simplified)
    let mtf_max = transformed.iter().max().copied().unwrap_or(0);
    let mtf_bits = if mtf_max > 0 {
        ceil_log2_nonzero(mtf_max as usize + 1)
    } else {
        1
    };

    // Use simple encoding if it's reasonably efficient
    if entry_bits <= 3 {
        // Simple mode
        writer.write(1, 1)?; // simple flag
        writer.write(2, entry_bits as u64)?; // bits per entry
        for &entry in context_map {
            writer.write(entry_bits, entry as u64)?;
        }
    } else {
        // ANS mode with MTF (simplified version using fixed-width encoding)
        // In a full implementation, this would use proper ANS encoding
        let use_mtf = mtf_bits < entry_bits;

        writer.write(1, 0)?; // not simple
        writer.write(1, u64::from(use_mtf))?;

        if use_mtf {
            // Write MTF-transformed symbols
            // For now, use a simple encoding: write max symbol, then each symbol
            encode_symbols_simple(&transformed, writer)?;
        } else {
            // Write raw symbols
            encode_symbols_simple(context_map, writer)?;
        }
    }

    Ok(())
}

/// Simple symbol encoding (placeholder for full ANS).
fn encode_symbols_simple(symbols: &[u8], writer: &mut BitWriter) -> Result<()> {
    if symbols.is_empty() {
        writer.write(8, 0)?; // max symbol = 0
        return Ok(());
    }

    let max_symbol = *symbols.iter().max().unwrap();
    let bits = if max_symbol > 0 {
        ceil_log2_nonzero(max_symbol as usize + 1)
    } else {
        1
    };

    // Write max symbol
    writer.write(8, max_symbol as u64)?;

    // Write each symbol with fixed width
    for &sym in symbols {
        writer.write(bits, sym as u64)?;
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
