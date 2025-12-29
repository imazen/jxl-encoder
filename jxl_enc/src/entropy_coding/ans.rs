// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! ANS (Asymmetric Numeral Systems) encoder.
//!
//! ANS is an entropy coding method used in JPEG XL for efficient symbol
//! compression. This module implements the rANS (range ANS) variant.

use crate::bit_writer::BitWriter;
use crate::error::{Error, Result};

/// ANS table size (2^12 = 4096).
pub const ANS_LOG_TAB_SIZE: u32 = 12;
pub const ANS_TAB_SIZE: u32 = 1 << ANS_LOG_TAB_SIZE;
pub const ANS_TAB_MASK: u32 = ANS_TAB_SIZE - 1;

/// Maximum alphabet size for ANS.
pub const ANS_MAX_ALPHABET_SIZE: usize = 256;

/// Initial state marker.
pub const ANS_SIGNATURE: u32 = 0x13;

/// Precision for reciprocal multiplication (avoids division).
const RECIPROCAL_PRECISION: u32 = 44;

/// Symbol information for ANS encoding.
#[derive(Debug, Clone)]
pub struct AnsEncSymbolInfo {
    /// Normalized frequency (1 to ANS_TAB_SIZE).
    pub freq: u16,
    /// Reciprocal of frequency for fast division: ceil((1 << 44) / freq).
    pub ifreq: u64,
    /// Maps remainder values to table offsets.
    pub reverse_map: Vec<u16>,
}

impl AnsEncSymbolInfo {
    /// Creates symbol info for a given frequency.
    pub fn new(freq: u16) -> Self {
        let ifreq = if freq > 0 {
            (1u64 << RECIPROCAL_PRECISION).div_ceil(freq as u64)
        } else {
            0
        };

        Self {
            freq,
            ifreq,
            reverse_map: vec![0; freq as usize],
        }
    }
}

/// ANS encoder state.
pub struct AnsEncoder {
    /// ANS state (normalized to range).
    state: u32,
    /// Accumulated output bits (stored reversed).
    bits: Vec<(u32, u8)>, // (bits, nbits)
}

impl AnsEncoder {
    /// Creates a new ANS encoder.
    pub fn new() -> Self {
        Self {
            state: ANS_SIGNATURE << 16,
            bits: Vec::new(),
        }
    }

    /// Encodes a single symbol using precomputed symbol info.
    ///
    /// Returns the bits that should be output (if any).
    #[inline]
    pub fn put_symbol(&mut self, info: &AnsEncSymbolInfo) {
        let freq = info.freq as u32;

        // Renormalization: if state is too large, emit 16 bits
        if (self.state >> (32 - ANS_LOG_TAB_SIZE)) >= freq {
            self.bits.push((self.state & 0xFFFF, 16));
            self.state >>= 16;
        }

        // State update using multiplication-by-reciprocal
        // v = state / freq (approximately)
        let v = ((self.state as u64 * info.ifreq) >> RECIPROCAL_PRECISION) as u32;
        let remainder = self.state - v * freq;

        // Look up offset in reverse map
        let offset = info.reverse_map[remainder as usize] as u32;

        // Update state
        self.state = (v << ANS_LOG_TAB_SIZE) + offset;
    }

    /// Finalizes encoding and writes to a BitWriter.
    ///
    /// Writes the final state followed by all accumulated bits in reverse order.
    pub fn finalize(self, writer: &mut BitWriter) -> Result<()> {
        // Write final state (32 bits)
        writer.write(32, self.state as u64)?;

        // Write accumulated bits in reverse order
        for &(bits, nbits) in self.bits.iter().rev() {
            writer.write(nbits as usize, bits as u64)?;
        }

        Ok(())
    }

    /// Returns the current state.
    pub fn state(&self) -> u32 {
        self.state
    }
}

impl Default for AnsEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A complete ANS distribution with encoding info for all symbols.
#[derive(Debug, Clone)]
pub struct AnsDistribution {
    /// Symbol encoding information.
    pub symbols: Vec<AnsEncSymbolInfo>,
    /// Log2 of distribution size (typically 12).
    pub log_alpha_size: u32,
    /// Total of normalized frequencies (should be ANS_TAB_SIZE).
    pub total: u32,
}

impl AnsDistribution {
    /// Creates a distribution from raw frequencies.
    ///
    /// Normalizes frequencies to sum to ANS_TAB_SIZE.
    pub fn from_frequencies(freqs: &[u32]) -> Result<Self> {
        if freqs.is_empty() {
            return Err(Error::InvalidHistogram("empty distribution".to_string()));
        }

        let total_count: u64 = freqs.iter().map(|&f| f as u64).sum();
        if total_count == 0 {
            return Err(Error::InvalidHistogram("all zero frequencies".to_string()));
        }

        // Normalize frequencies to sum to ANS_TAB_SIZE
        let mut normalized: Vec<u16> = Vec::with_capacity(freqs.len());
        let mut running_total: u32 = 0;

        for &freq in freqs.iter() {
            let normalized_freq = if freq == 0 {
                0
            } else {
                // Scale to ANS_TAB_SIZE, ensuring at least 1 for non-zero
                ((freq as u64 * ANS_TAB_SIZE as u64) / total_count).max(1) as u16
            };
            normalized.push(normalized_freq);
            running_total += normalized_freq as u32;
        }

        // Adjust to exactly sum to ANS_TAB_SIZE
        let diff = running_total as i32 - ANS_TAB_SIZE as i32;
        if diff != 0 {
            // Find largest frequency and adjust it
            if let Some((max_idx, _)) = normalized
                .iter()
                .enumerate()
                .filter(|&(_, &f)| f > 0)
                .max_by_key(|&(_, &f)| f)
            {
                let new_val = (normalized[max_idx] as i32 - diff).max(1) as u16;
                normalized[max_idx] = new_val;
            }
        }

        // Build symbol info with reverse maps
        let mut symbols: Vec<AnsEncSymbolInfo> = normalized
            .iter()
            .map(|&f| AnsEncSymbolInfo::new(f))
            .collect();

        // Build reverse map (alias table)
        Self::build_reverse_maps(&mut symbols)?;

        Ok(Self {
            symbols,
            log_alpha_size: ANS_LOG_TAB_SIZE,
            total: ANS_TAB_SIZE,
        })
    }

    /// Creates a flat (uniform) distribution.
    pub fn flat(alphabet_size: usize) -> Result<Self> {
        if alphabet_size == 0 || alphabet_size > ANS_TAB_SIZE as usize {
            return Err(Error::InvalidHistogram(format!(
                "invalid alphabet size: {}",
                alphabet_size
            )));
        }

        let base_freq = ANS_TAB_SIZE as usize / alphabet_size;
        let remainder = ANS_TAB_SIZE as usize % alphabet_size;

        let mut freqs = vec![base_freq as u32; alphabet_size];
        for freq in freqs.iter_mut().take(remainder) {
            *freq += 1;
        }

        Self::from_frequencies(&freqs)
    }

    /// Builds reverse maps for all symbols.
    fn build_reverse_maps(symbols: &mut [AnsEncSymbolInfo]) -> Result<()> {
        // First pass: compute cumulative positions
        let mut cumul = vec![0u32; symbols.len() + 1];
        for (i, sym) in symbols.iter().enumerate() {
            cumul[i + 1] = cumul[i] + sym.freq as u32;
        }

        // Verify total
        let total = *cumul.last().unwrap_or(&0);
        if total != ANS_TAB_SIZE {
            // This shouldn't happen if normalization is correct
            return Err(Error::InvalidHistogram(format!(
                "frequencies sum to {} instead of {}",
                total, ANS_TAB_SIZE
            )));
        }

        // Build reverse maps: for each position in [0, ANS_TAB_SIZE),
        // find which symbol it belongs to and its offset within that symbol
        for pos in 0..ANS_TAB_SIZE {
            // Binary search for the symbol
            let mut lo = 0;
            let mut hi = symbols.len();
            while lo < hi {
                let mid = (lo + hi) / 2;
                if cumul[mid + 1] <= pos {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            if lo < symbols.len() && symbols[lo].freq > 0 {
                let offset = pos - cumul[lo];
                symbols[lo].reverse_map[offset as usize] = pos as u16;
            }
        }

        Ok(())
    }

    /// Returns the number of symbols in this distribution.
    pub fn alphabet_size(&self) -> usize {
        self.symbols.len()
    }

    /// Gets the encoding info for a symbol.
    pub fn get(&self, symbol: usize) -> Option<&AnsEncSymbolInfo> {
        self.symbols.get(symbol)
    }

    /// Writes this distribution to a BitWriter.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // Check if this is a flat distribution
        let is_flat = self.is_flat();

        writer.write(1, 0)?; // Non-small tree marker
        writer.write(1, u64::from(is_flat))?;

        if is_flat {
            // Flat distribution: just encode alphabet size
            write_var_len_uint8(writer, (self.alphabet_size() - 1) as u8)?;
        } else {
            // General distribution encoding
            // For now, use the simplest encoding (shift = 0, meaning power-of-2 counts)
            self.write_general(writer)?;
        }

        Ok(())
    }

    /// Checks if this is a flat (uniform) distribution.
    fn is_flat(&self) -> bool {
        let first_freq = self.symbols.first().map(|s| s.freq).unwrap_or(0);
        if first_freq == 0 {
            return false;
        }
        self.symbols
            .iter()
            .all(|s| s.freq == first_freq || s.freq == first_freq - 1)
    }

    /// Writes a general (non-flat) distribution.
    fn write_general(&self, writer: &mut BitWriter) -> Result<()> {
        // Encode shift (we use shift=12 for maximum precision)
        let method: u64 = 13; // shift + 1
        let upper_bound_log = 4; // floor_log2(13)
        let log = floor_log2(method as u32);

        // Write unary prefix
        writer.write(log as usize, (1u64 << log) - 1)?;
        if log != upper_bound_log {
            writer.write(1, 0)?;
        }
        // Write value suffix
        writer.write(log as usize, ((1u64 << log) - 1) & method)?;

        // Encode alphabet size
        write_var_len_uint8(writer, (self.alphabet_size() - 3) as u8)?;

        // For a simple implementation, encode each frequency directly
        // Full implementation would use Huffman for bit-widths + RLE
        for sym in &self.symbols {
            // Encode frequency using a simple variable-length code
            let freq = sym.freq;
            if freq == 0 {
                writer.write(1, 0)?;
            } else {
                writer.write(1, 1)?;
                let bits = 16 - freq.leading_zeros();
                writer.write(4, bits as u64)?;
                if bits > 0 {
                    writer.write(bits as usize, freq as u64)?;
                }
            }
        }

        Ok(())
    }
}

/// Writes a variable-length uint8.
fn write_var_len_uint8(writer: &mut BitWriter, n: u8) -> Result<()> {
    if n == 0 {
        writer.write(1, 0)?;
    } else {
        writer.write(1, 1)?;
        let nbits = 8 - n.leading_zeros();
        writer.write(3, (nbits - 1) as u64)?;
        writer.write((nbits - 1) as usize, (n as u64) - (1u64 << (nbits - 1)))?;
    }
    Ok(())
}

/// Floor log2 of a value.
#[inline]
fn floor_log2(n: u32) -> u32 {
    if n == 0 { 0 } else { 31 - n.leading_zeros() }
}

/// Precision calculation for frequency encoding.
#[allow(dead_code)]
fn get_population_count_precision(logcount: i32, shift: i32) -> i32 {
    let r = logcount.min(shift - ((ANS_LOG_TAB_SIZE as i32 - logcount) >> 1));
    r.max(0)
}

/// Encodes tokens using ANS.
pub fn encode_tokens_ans(
    tokens: &[(u32, u32)], // (context, value) pairs
    distributions: &[AnsDistribution],
    context_map: &[usize],
    writer: &mut BitWriter,
) -> Result<()> {
    let mut encoder = AnsEncoder::new();

    // Process tokens in reverse order (ANS requirement)
    for &(context, value) in tokens.iter().rev() {
        let dist_idx = context_map.get(context as usize).copied().unwrap_or(0);
        if let Some(dist) = distributions.get(dist_idx)
            && let Some(info) = dist.get(value as usize)
        {
            encoder.put_symbol(info);
        }
    }

    encoder.finalize(writer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_distribution() {
        let dist = AnsDistribution::flat(16).unwrap();
        assert_eq!(dist.alphabet_size(), 16);

        // All frequencies should be 256 (4096 / 16)
        for sym in &dist.symbols {
            assert_eq!(sym.freq, 256);
        }
    }

    #[test]
    fn test_from_frequencies() {
        let freqs = vec![100, 200, 300, 400];
        let dist = AnsDistribution::from_frequencies(&freqs).unwrap();
        assert_eq!(dist.alphabet_size(), 4);

        // Total should be ANS_TAB_SIZE
        let total: u32 = dist.symbols.iter().map(|s| s.freq as u32).sum();
        assert_eq!(total, ANS_TAB_SIZE);
    }

    #[test]
    fn test_ans_encoder_basic() {
        let dist = AnsDistribution::flat(4).unwrap();
        let mut encoder = AnsEncoder::new();

        // Encode a few symbols
        encoder.put_symbol(&dist.symbols[0]);
        encoder.put_symbol(&dist.symbols[1]);
        encoder.put_symbol(&dist.symbols[2]);

        // State should have changed from initial
        assert_ne!(encoder.state(), ANS_SIGNATURE << 16);
    }

    #[test]
    fn test_reverse_map() {
        let dist = AnsDistribution::flat(4).unwrap();

        // Each symbol should have freq entries in reverse_map
        for sym in &dist.symbols {
            assert_eq!(sym.reverse_map.len(), sym.freq as usize);
        }

        // All positions 0..4096 should be covered exactly once
        let mut covered = vec![false; ANS_TAB_SIZE as usize];
        for sym in &dist.symbols {
            for &pos in &sym.reverse_map {
                assert!(!covered[pos as usize], "position {} covered twice", pos);
                covered[pos as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "not all positions covered");
    }

    #[test]
    fn test_write_distribution() {
        let dist = AnsDistribution::flat(16).unwrap();
        let mut writer = BitWriter::new();
        dist.write(&mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        // Should produce some output
        assert!(!bytes.is_empty());
    }
}
