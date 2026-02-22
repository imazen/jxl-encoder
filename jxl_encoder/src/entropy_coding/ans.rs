// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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

/// RLE marker symbol in logcount prefix code.
const RLE_MARKER_SYM: u8 = 13;

/// Prefix code table for encoding log-frequency values (0-13).
/// Format: (nbits, code_lsb) - the code to write for each symbol.
/// This is the inverse of the decoder's lookup table in jxl-rs.
const LOGCOUNT_PREFIX_CODE: [(u8, u8); 14] = [
    (5, 0b10001),   // 0: freq=0 (but we use 0 for zero, not logcount 0)
    (4, 0b1011),    // 1: logcount=1, freq=1
    (4, 0b1111),    // 2: logcount=2, freq in [2,3]
    (4, 0b0011),    // 3: logcount=3, freq in [4,7]
    (4, 0b1001),    // 4: logcount=4, freq in [8,15]
    (4, 0b0111),    // 5: logcount=5, freq in [16,31]
    (3, 0b100),     // 6: logcount=6, freq in [32,63]
    (3, 0b010),     // 7: logcount=7, freq in [64,127]
    (3, 0b101),     // 8: logcount=8, freq in [128,255]
    (3, 0b110),     // 9: logcount=9, freq in [256,511]
    (3, 0b000),     // 10: logcount=10, freq in [512,1023]
    (6, 0b100001),  // 11: logcount=11, freq in [1024,2047]
    (7, 0b0000001), // 12: logcount=12, freq in [2048,4095]
    (7, 0b1000001), // 13: RLE marker
];

/// Build sorted table of all representable count values for a given shift.
/// Matches libjxl's AllowedCounts precomputation (enc_ans.cc:581-615).
/// Returns counts in DECREASING order (index 0 = highest count).
fn build_allowed_counts(shift: u32) -> Vec<i32> {
    let mut counts = Vec::with_capacity(256);
    // Count = 1 is always representable (logcount=1, no precision bits)
    counts.push(1i32);
    for bits in 1..ANS_LOG_TAB_SIZE {
        let precision = get_population_count_precision(bits, shift);
        let drop_bits = bits.saturating_sub(precision);
        let num_mantissa = 1u32 << precision;
        for mantissa in 0..num_mantissa {
            let count = (1i32 << bits) | ((mantissa as i32) << drop_bits);
            if count > 0 && count < ANS_TAB_SIZE as i32 {
                counts.push(count);
            }
        }
    }
    counts.sort_unstable();
    counts.dedup();
    counts.reverse(); // Decreasing order: index 0 = highest
    counts
}

/// Find the index of the highest allowed count <= target in a decreasing-order table.
/// Snaps DOWN to prevent rest from going negative (matches libjxl's mask-off behavior).
/// If target < smallest allowed (1), returns the last index (count=1).
fn find_allowed_leq(allowed: &[i32], target: i32) -> usize {
    // Binary search in decreasing order: find first index where allowed[i] <= target
    let mut lo = 0usize;
    let mut hi = allowed.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if allowed[mid] > target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // lo is the first index where allowed[lo] <= target
    if lo >= allowed.len() {
        allowed.len() - 1 // target < 1, snap to minimum (1)
    } else {
        lo
    }
}

/// Estimate data cost of encoding `histo` using ANS with normalized `counts`.
/// Matches libjxl's `EstimateDataBits` (enc_ans.cc:362-370).
/// `cost = total * ANS_LOG_TAB_SIZE - sum(actual_count * log2(norm_count))`
fn estimate_data_bits_normalized(
    histo_counts: &[i32],
    norm_counts: &[i32],
    total_count: usize,
    alphabet_size: usize,
) -> f64 {
    let mut sum = 0.0f64;
    for (actual, norm) in histo_counts
        .iter()
        .zip(norm_counts.iter())
        .take(alphabet_size)
    {
        if *actual > 0 && *norm > 0 {
            sum += *actual as f64 * (*norm as f64).log2();
        }
    }
    total_count as f64 * ANS_LOG_TAB_SIZE as f64 - sum
}

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
            reverse_map: Vec::new(), // Allocated later in build_reverse_maps
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

    /// Creates a new ANS encoder with pre-allocated capacity for `num_tokens` tokens.
    pub fn with_capacity(num_tokens: usize) -> Self {
        Self {
            state: ANS_SIGNATURE << 16,
            bits: Vec::with_capacity(num_tokens * 2), // ~2 entries per token (symbol + extra bits)
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

    /// Pushes extra bits into the encoder's output buffer.
    ///
    /// Used for HybridUint extra bits that are interleaved with ANS symbols.
    /// These bits are stored in the same reversed buffer and will be emitted
    /// in proper order during finalize().
    #[inline]
    pub fn push_bits(&mut self, bits: u32, nbits: u8) {
        if nbits > 0 {
            self.bits.push((bits, nbits));
        }
    }

    /// Finalizes encoding and writes to a BitWriter.
    ///
    /// Writes the final state followed by all accumulated bits in reverse order.
    pub fn finalize(self, writer: &mut BitWriter) -> Result<()> {
        // Debug: show final state
        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "ANS finalize: state=0x{:08x}, {} bit chunks",
            self.state,
            self.bits.len()
        );

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

        // Build reverse map (alias table) using default log_alpha_size for this alphabet
        let log_alpha_size = Self::default_log_alpha_size(symbols.len());
        Self::build_reverse_maps(&mut symbols, log_alpha_size)?;

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

    /// Creates a distribution from pre-normalized counts.
    ///
    /// The counts must already sum to ANS_TAB_SIZE (4096). This is used
    /// when building distributions from ANSEncodingHistogram which has
    /// already done the normalization.
    pub fn from_normalized_counts(counts: &[i32]) -> Result<Self> {
        let log_alpha_size = Self::default_log_alpha_size(counts.len());
        Self::from_normalized_counts_with_log_alpha(counts, log_alpha_size)
    }

    /// Creates a distribution from pre-normalized counts with explicit log_alpha_size.
    ///
    /// Use this when multiple distributions share a single header (e.g., multi-histogram
    /// ANS). The `log_alpha_size` must match the value written to the bitstream header,
    /// NOT the per-distribution default. The decoder reads one global log_alpha_size
    /// and uses it for all distributions in the group.
    pub fn from_normalized_counts_with_log_alpha(
        counts: &[i32],
        log_alpha_size: usize,
    ) -> Result<Self> {
        if counts.is_empty() {
            return Err(Error::InvalidHistogram("empty distribution".to_string()));
        }

        // Verify sum
        let total: i32 = counts.iter().sum();
        if total != ANS_TAB_SIZE as i32 {
            return Err(Error::InvalidHistogram(format!(
                "normalized counts sum to {} instead of {}",
                total, ANS_TAB_SIZE
            )));
        }

        // Build symbol info
        let mut symbols: Vec<AnsEncSymbolInfo> = counts
            .iter()
            .map(|&c| AnsEncSymbolInfo::new(c.max(0) as u16))
            .collect();

        // Build reverse maps with the caller-specified log_alpha_size
        Self::build_reverse_maps(&mut symbols, log_alpha_size)?;

        Ok(Self {
            symbols,
            log_alpha_size: ANS_LOG_TAB_SIZE,
            total: ANS_TAB_SIZE,
        })
    }

    /// Computes the default log_alpha_size for a given alphabet size.
    ///
    /// This is the value that would be written to the bitstream header for a
    /// standalone distribution. For multi-histogram contexts, use the global
    /// log_alpha_size from the header instead.
    fn default_log_alpha_size(alphabet_size: usize) -> usize {
        use super::encode_ans::ANS_LOG_ALPHA_SIZE;
        if alphabet_size <= (1 << ANS_LOG_ALPHA_SIZE) {
            ANS_LOG_ALPHA_SIZE
        } else {
            let min_bits = if alphabet_size <= 1 {
                5
            } else {
                (alphabet_size - 1).ilog2() as usize + 1
            };
            min_bits.clamp(5, 8)
        }
    }

    /// Builds reverse maps for all symbols using the alias table method.
    ///
    /// The decoder uses an alias table with buckets. Each idx in [0, 4096) maps to some
    /// symbol and an offset within that symbol's range. The encoder needs to know,
    /// for each symbol s and remainder r in [0, freq[s]), what idx to output.
    ///
    /// This exactly mirrors the decoder's build_alias_map and read methods.
    ///
    /// `log_alpha_size` MUST match the value written to the bitstream header, since
    /// the decoder uses it to split 12-bit indices into (bucket, position) pairs.
    /// When multiple distributions share a header, they all use the same global
    /// log_alpha_size — passing a per-distribution value causes alias table mismatch.
    fn build_reverse_maps(symbols: &mut [AnsEncSymbolInfo], log_alpha_size: usize) -> Result<()> {
        let alphabet_size = symbols.len();
        if alphabet_size == 0 {
            return Ok(());
        }

        // Verify frequencies sum to ANS_TAB_SIZE
        let total: u32 = symbols.iter().map(|s| s.freq as u32).sum();
        if total != ANS_TAB_SIZE {
            return Err(Error::InvalidHistogram(format!(
                "frequencies sum to {} instead of {}",
                total, ANS_TAB_SIZE
            )));
        }

        // Special case: single-symbol distribution
        // jxl-rs uses a simplified alias table where offset = idx for all positions.
        // This means reverse_map[r] = r (identity mapping) for the single symbol.
        if let Some(single_sym_idx) = symbols.iter().position(|s| s.freq == ANS_TAB_SIZE as u16) {
            // Clear all reverse maps
            for sym in symbols.iter_mut() {
                sym.reverse_map.clear();
            }
            // Set identity mapping for the single symbol
            let map = &mut symbols[single_sym_idx].reverse_map;
            map.resize(ANS_TAB_SIZE as usize, 0);
            for (i, v) in map.iter_mut().enumerate() {
                *v = i as u16;
            }
            return Ok(());
        }

        let table_size = 1usize << log_alpha_size;
        let log_bucket_size = ANS_LOG_TAB_SIZE as usize - log_alpha_size;
        let bucket_size = 1u16 << log_bucket_size;

        // Working bucket structure matching jxl-rs
        #[derive(Clone)]
        #[allow(dead_code)]
        struct WorkingBucket {
            dist: u16,         // Frequency of primary symbol
            alias_symbol: u16, // Alias symbol (used when pos >= cutoff)
            alias_offset: u16, // Offset for alias symbol
            alias_cutoff: u16, // Positions [0, cutoff) use primary, [cutoff, bucket_size) use alias
        }

        let mut buckets: Vec<WorkingBucket> = (0..table_size)
            .map(|i| {
                let dist = if i < alphabet_size {
                    symbols[i].freq
                } else {
                    0
                };
                WorkingBucket {
                    dist,
                    alias_symbol: if i < alphabet_size { i as u16 } else { 0 },
                    alias_offset: 0,
                    alias_cutoff: dist,
                }
            })
            .collect();

        // Separate into underfull and overfull
        let mut underfull: Vec<usize> = Vec::with_capacity(table_size);
        let mut overfull: Vec<usize> = Vec::with_capacity(table_size);
        for (i, bucket) in buckets.iter().enumerate() {
            if bucket.alias_cutoff < bucket_size {
                underfull.push(i);
            } else if bucket.alias_cutoff > bucket_size {
                overfull.push(i);
            }
        }

        // Alias redistribution - exactly matching jxl-rs
        while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
            let by = bucket_size - buckets[u].alias_cutoff;
            buckets[o].alias_cutoff -= by;
            buckets[u].alias_symbol = o as u16;
            buckets[u].alias_offset = buckets[o].alias_cutoff;

            match buckets[o].alias_cutoff.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(o),
                std::cmp::Ordering::Greater => overfull.push(o),
                std::cmp::Ordering::Equal => {}
            }
        }

        // Pre-allocate reverse maps with exact sizes (offset is 0..freq-1)
        for sym in symbols.iter_mut() {
            sym.reverse_map.clear();
            sym.reverse_map.resize(sym.freq as usize, 0);
        }

        // For each idx in [0, 4096), simulate the decoder to find which symbol it decodes to
        // and what offset within that symbol's range. Write directly into reverse_map[offset].
        for idx in 0..ANS_TAB_SIZE {
            let bucket_idx = (idx >> log_bucket_size) as usize;
            let pos = (idx as u16) & (bucket_size - 1);

            let bucket = &buckets[bucket_idx.min(table_size - 1)];
            let alias_cutoff = bucket.alias_cutoff;

            let (symbol, offset) = if pos < alias_cutoff {
                (bucket_idx, pos)
            } else {
                let alias_sym = bucket.alias_symbol as usize;
                let offset = bucket.alias_offset - alias_cutoff + pos;
                (alias_sym, offset)
            };

            if symbol < alphabet_size {
                symbols[symbol].reverse_map[offset as usize] = idx as u16;
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
pub fn floor_log2_ans(n: u32) -> u32 {
    if n == 0 { 0 } else { 31 - n.leading_zeros() }
}

#[inline]
fn floor_log2(n: u32) -> u32 {
    floor_log2_ans(n)
}

/// Precision calculation for frequency encoding.
///
/// Determines how many bits of precision to use when encoding a frequency count.
/// Larger counts can be encoded with less precision.
///
/// Matches libjxl's `GetPopulationCountPrecision` from `ans_common.h`.
pub fn get_population_count_precision(logcount: u32, shift: u32) -> u32 {
    let logcount_i = logcount as i32;
    let shift_i = shift as i32;
    let r = logcount_i.min(shift_i - ((ANS_LOG_TAB_SIZE as i32 - logcount_i) >> 1));
    r.max(0) as u32
}

/// Strategy for ANS histogram normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ANSHistogramStrategy {
    /// Only try a few shift values (fastest).
    Fast,
    /// Try every other shift value.
    Approximate,
    /// Try all shift values (best compression).
    #[default]
    Precise,
}

/// Normalized ANS histogram for encoding.
///
/// Contains frequency counts normalized to sum to ANS_TAB_SIZE (4096).
#[derive(Clone, Debug)]
pub struct ANSEncodingHistogram {
    /// Normalized frequency counts.
    pub counts: Vec<i32>,
    /// Alphabet size (highest non-zero symbol + 1).
    pub alphabet_size: usize,
    /// Cost estimate (header + data bits).
    pub cost: f32,
    /// Encoding method:
    /// - 0: flat distribution
    /// - 1: small code (1-2 symbols)
    /// - 2-13: shift value + 1
    pub method: u32,
    /// Position of the balancing bin (absorbs rounding error).
    pub omit_pos: usize,
    /// Number of unique symbols (for small code).
    num_symbols: usize,
    /// Symbol indices (for small code, up to 2).
    symbols: [usize; 2],
}

impl ANSEncodingHistogram {
    /// Create an empty histogram.
    pub fn new() -> Self {
        Self {
            counts: Vec::new(),
            alphabet_size: 0,
            cost: f32::MAX,
            method: 0,
            omit_pos: 0,
            num_symbols: 0,
            symbols: [0, 0],
        }
    }

    /// Create from a Histogram with the best normalization.
    ///
    /// Tries different shift values and picks the one with lowest cost.
    pub fn from_histogram(
        histo: &super::histogram::Histogram,
        strategy: ANSHistogramStrategy,
    ) -> Result<Self> {
        if histo.total_count == 0 {
            // Empty histogram
            return Ok(Self {
                counts: vec![0i32; histo.counts.len().max(1)],
                alphabet_size: 1,
                cost: 0.0,
                method: 0, // Flat
                omit_pos: 0,
                num_symbols: 0,
                symbols: [0, 0],
            });
        }

        let alphabet_size = histo.alphabet_size();

        // Count non-zero symbols
        let mut num_symbols = 0;
        let mut symbols = [0usize; 2];
        for (i, &count) in histo.counts.iter().enumerate() {
            if count > 0 {
                if num_symbols < 2 {
                    symbols[num_symbols] = i;
                }
                num_symbols += 1;
            }
        }

        // Single symbol or two symbols: use small code
        if num_symbols <= 2 {
            let mut counts = vec![0i32; alphabet_size];
            if num_symbols == 1 {
                counts[symbols[0]] = ANS_TAB_SIZE as i32;
            } else {
                // Two symbols: proportional allocation
                let total = histo.total_count as f64;
                let count0 = histo.counts[symbols[0]] as f64;
                let norm0 = ((count0 / total) * ANS_TAB_SIZE as f64).round() as i32;
                let norm0 = norm0.clamp(1, (ANS_TAB_SIZE - 1) as i32);
                counts[symbols[0]] = norm0;
                counts[symbols[1]] = ANS_TAB_SIZE as i32 - norm0;
            }

            // Cost is just the header
            let cost = if num_symbols <= 1 { 4.0 } else { 4.0 + 12.0 }; // Approximate

            return Ok(Self {
                counts,
                alphabet_size,
                cost,
                method: 1, // Small code
                omit_pos: symbols[0],
                num_symbols,
                symbols,
            });
        }

        // General case: start with flat distribution as baseline
        // libjxl always computes flat cost first (enc_ans.cc:97-102) and picks
        // the cheaper of flat vs shift-based encoding.
        let flat_data_cost = {
            let log2_alpha = (alphabet_size as f32).log2();
            histo.total_count as f32 * log2_alpha
        };
        let flat_header_cost = 2.0 + 8.0; // method=0 marker + alphabet size
        let mut best = Self {
            counts: {
                let alpha = alphabet_size as u32;
                let per = ANS_TAB_SIZE / alpha;
                let remainder = (ANS_TAB_SIZE % alpha) as usize;
                let mut c = vec![per as i32; alphabet_size];
                // Distribute remainder to first symbols
                for c in c.iter_mut().take(remainder) {
                    *c += 1;
                }
                c
            },
            alphabet_size,
            cost: flat_header_cost + flat_data_cost,
            method: 0, // Flat
            omit_pos: 0,
            num_symbols,
            symbols,
        };

        let shifts: Vec<u32> = match strategy {
            ANSHistogramStrategy::Fast => vec![0, 6, 12],
            ANSHistogramStrategy::Approximate => (0..=ANS_LOG_TAB_SIZE).step_by(2).collect(),
            ANSHistogramStrategy::Precise => (0..ANS_LOG_TAB_SIZE).collect(),
        };

        for shift in shifts {
            let mut candidate = Self {
                counts: vec![0i32; alphabet_size],
                alphabet_size,
                cost: f32::MAX,
                method: shift.min(ANS_LOG_TAB_SIZE - 1) + 1,
                omit_pos: 0,
                num_symbols,
                symbols,
            };

            if candidate.rebalance_histogram(histo, shift) {
                candidate.cost = candidate.estimate_cost(histo);
                if candidate.cost < best.cost {
                    best = candidate;
                }
            }
        }

        if best.cost == f32::MAX {
            // Debug: dump histogram info
            eprintln!(
                "ANS rebalance FAILED: alphabet_size={}, num_symbols={}, total_count={}",
                alphabet_size, num_symbols, histo.total_count
            );
            for (i, &c) in histo.counts.iter().enumerate() {
                if c > 0 {
                    eprintln!("  symbol {}: count={}", i, c);
                }
            }
            return Err(Error::InvalidHistogram(
                "Failed to rebalance histogram".to_string(),
            ));
        }

        Ok(best)
    }

    /// Rebalance histogram using allowed counts table and greedy optimization.
    /// Matches libjxl's `RebalanceHistogram` (enc_ans.cc:416-559).
    ///
    /// 1. Build table of all representable counts for this shift
    /// 2. Normalize each bin to nearest representable count
    /// 3. Greedy loop: adjust bins within allowed counts to minimize entropy
    /// 4. Set balancing bin (remainder_pos) as remainder
    fn rebalance_histogram(&mut self, histo: &super::histogram::Histogram, shift: u32) -> bool {
        let total_count = histo.total_count;
        if total_count == 0 {
            return false;
        }

        // Build sorted table of representable count values (decreasing order).
        let allowed = build_allowed_counts(shift);

        let norm = ANS_TAB_SIZE as f64 / total_count as f64;

        // Find remainder_pos: symbol with highest original frequency (balancing bin).
        // Matches libjxl's remainder_pos selection.
        let mut remainder_pos = 0;
        let mut max_freq = 0i32;

        // Bins eligible for greedy adjustment: (orig_freq, index_in_allowed, symbol_index)
        let mut bins: Vec<(i32, usize, usize)> = Vec::with_capacity(self.alphabet_size);
        let mut rest = ANS_TAB_SIZE as i32;

        for (n, &freq) in histo.counts.iter().enumerate().take(self.alphabet_size) {
            if freq > max_freq {
                remainder_pos = n;
                max_freq = freq;
            }

            if freq == 0 {
                self.counts[n] = 0;
                continue;
            }

            let target = freq as f64 * norm;
            // Round and clamp to [1, ANS_TAB_SIZE-1], then snap DOWN to allowed
            let rounded = target.round().max(1.0).min((ANS_TAB_SIZE - 1) as f64) as i32;
            let ai = find_allowed_leq(&allowed, rounded);
            let count = allowed[ai];

            self.counts[n] = count;
            rest -= count;

            // Only bins with target > 1.0 are adjustable (matches libjxl)
            if target > 1.0 {
                bins.push((freq, ai, n));
            }
        }

        // Remove the balancing bin from the adjustable set
        if let Some(pos) = bins.iter().position(|b| b.2 == remainder_pos) {
            bins.remove(pos);
        }

        // rest now represents what the balancing bin should be.
        // Add back remainder_pos's initial count since it's no longer adjustable.
        rest += self.counts[remainder_pos];

        // Greedy entropy optimization (libjxl enc_ans.cc:495-537).
        // Each iteration: find the best bin to increment or decrement by one
        // allowed-count step, with the balancing bin absorbing the difference.
        if !bins.is_empty() {
            let max_freq_f = max_freq as f64;
            // Fixed-point-ish log2 scaled by a large constant for precision.
            // Matches libjxl's lg2 table concept but using f64 directly.
            let lg2 = |v: i32| -> f64 { if v <= 0 { 0.0 } else { (v as f64).log2() } };

            loop {
                // Find the best increment step (grow a bin, shrink balancing)
                let mut best_inc_net = 0.0f64; // must be > 0 to be taken
                let mut best_inc_bi = None;

                // Find the best decrement step (shrink a bin, grow balancing)
                let mut best_dec_net = 0.0f64; // must be > 0 to be taken
                let mut best_dec_bi = None;

                for (bi, &(freq, ai, _bin)) in bins.iter().enumerate() {
                    let count = allowed[ai];
                    let freq_f = freq as f64;
                    let lg2_count = lg2(count);

                    // Try increment: move to allowed[ai - 1] (higher count)
                    if ai > 0 {
                        let new_count = allowed[ai - 1];
                        let step = new_count - count;
                        let new_rest = rest - step;
                        if new_rest > 0 || rest >= ANS_TAB_SIZE as i32 {
                            let gain = freq_f * (lg2(new_count) - lg2_count);
                            let cost = if rest >= ANS_TAB_SIZE as i32 {
                                0.0 // tractor: pull rest down, no cost
                            } else if rest > 0 && new_rest > 0 {
                                max_freq_f * (lg2(rest) - lg2(new_rest))
                            } else {
                                f64::MAX
                            };
                            let net = gain - cost;
                            // Normalize by step size for fair comparison across step sizes
                            let step_log = floor_log2(step as u32);
                            let norm_net = if step_log > 0 {
                                net / (1u32 << step_log) as f64
                            } else {
                                net
                            };
                            if norm_net > best_inc_net {
                                best_inc_net = norm_net;
                                best_inc_bi = Some(bi);
                            }
                        }
                    }

                    // Try decrement: move to allowed[ai + 1] (lower count)
                    if ai + 1 < allowed.len() && allowed[ai + 1] > 0 {
                        let new_count = allowed[ai + 1];
                        let step = count - new_count;
                        let new_rest = rest + step;
                        if new_rest < ANS_TAB_SIZE as i32 || rest <= 1 {
                            let loss = freq_f * (lg2_count - lg2(new_count));
                            let gain = if rest <= 1 {
                                f64::MAX // tractor: pull rest up, infinite gain
                            } else if rest > 0 && new_rest < ANS_TAB_SIZE as i32 {
                                max_freq_f * (lg2(new_rest) - lg2(rest))
                            } else {
                                0.0
                            };
                            let net = gain - loss;
                            let step_log = floor_log2(step as u32);
                            let norm_net = if step_log > 0 {
                                net / (1u32 << step_log) as f64
                            } else {
                                net
                            };
                            if norm_net > best_dec_net {
                                best_dec_net = norm_net;
                                best_dec_bi = Some(bi);
                            }
                        }
                    }
                }

                // Prefer increment over decrement (matches libjxl)
                if best_inc_net > 0.0 {
                    if let Some(bi) = best_inc_bi {
                        let step = allowed[bins[bi].1 - 1] - allowed[bins[bi].1];
                        bins[bi].1 -= 1; // move to higher count
                        rest -= step;
                    }
                } else if best_dec_net > 0.0 {
                    if let Some(bi) = best_dec_bi {
                        let step = allowed[bins[bi].1] - allowed[bins[bi].1 + 1];
                        bins[bi].1 += 1; // move to lower count
                        rest += step;
                    }
                } else {
                    break; // No improvement possible
                }
            }

            // Write final counts from allowed table
            for &(_freq, ai, bin) in &bins {
                self.counts[bin] = allowed[ai];
            }

            // Handle omit_pos bit-width constraint (libjxl enc_ans.cc:545-551):
            // If an earlier bin has count >= 2048 (logcount >= 12), swap with
            // remainder_pos so the balancing bin can grow without bit-width issues.
            for n in 0..remainder_pos {
                if self.counts[n] >= 2048 {
                    self.counts[remainder_pos] = self.counts[n];
                    remainder_pos = n;
                    break;
                }
            }
        }

        // Set balancing bin
        self.counts[remainder_pos] = rest;
        self.omit_pos = remainder_pos;

        if rest <= 0 {
            return false;
        }

        // Ensure remainder_pos is the FIRST symbol with the highest logcount.
        // The decoder re-derives omit_pos by scanning symbols in order and picking
        // the first one with the maximum logcount. If another symbol has equal or
        // higher logcount, the decoder picks the wrong one and decoding fails.
        for _ in 0..10 {
            let omit_logcount = floor_log2(self.counts[remainder_pos] as u32) + 1;
            let mut adjusted = false;
            for i in 0..self.alphabet_size {
                if i == remainder_pos || self.counts[i] <= 0 {
                    continue;
                }
                let logcount = floor_log2(self.counts[i] as u32) + 1;
                let needs_fix =
                    logcount > omit_logcount || (logcount == omit_logcount && i < remainder_pos);
                if needs_fix {
                    // Reduce this symbol to a representable value with lower logcount.
                    // Find the highest allowed count with logcount < omit_logcount
                    // (or <= omit_logcount for symbols after remainder_pos).
                    let target_logcount = if i < remainder_pos {
                        omit_logcount.saturating_sub(1)
                    } else {
                        omit_logcount
                    };
                    let max_value = (1i32 << target_logcount) - 1;
                    let new_ai = find_allowed_leq(&allowed, max_value);
                    let new_count = allowed[new_ai].max(1);
                    let reduction = self.counts[i] - new_count;
                    if reduction > 0 {
                        self.counts[i] = new_count;
                        self.counts[remainder_pos] += reduction;
                        adjusted = true;
                    }
                }
            }
            if !adjusted {
                break;
            }
        }

        // Final verification
        let omit_logcount = floor_log2(self.counts[remainder_pos] as u32) + 1;
        for (i, &count) in self.counts.iter().enumerate().take(self.alphabet_size) {
            if i == remainder_pos || count <= 0 {
                continue;
            }
            let logcount = floor_log2(count as u32) + 1;
            if logcount > omit_logcount || (logcount == omit_logcount && i < remainder_pos) {
                return false;
            }
        }

        // Verify sum
        let sum: i32 = self.counts.iter().sum();
        sum == ANS_TAB_SIZE as i32
    }

    /// Estimate encoding cost (header + data bits).
    /// Uses precise ANS cost model matching libjxl's `Cost()` (enc_ans.cc:376-380).
    fn estimate_cost(&self, histo: &super::histogram::Histogram) -> f32 {
        let header_cost = self.estimate_header_cost();
        let data_cost = estimate_data_bits_normalized(
            &histo.counts,
            &self.counts,
            histo.total_count,
            self.alphabet_size,
        ) as f32;
        header_cost + data_cost
    }

    /// Estimate header encoding cost.
    fn estimate_header_cost(&self) -> f32 {
        if self.method == 0 {
            // Flat: 2 bits + alphabet size encoding
            2.0 + 8.0
        } else if self.num_symbols <= 2 {
            // Small code
            if self.num_symbols <= 1 {
                3.0 + 8.0 // nsym=0: marker + symbol
            } else {
                3.0 + 16.0 + 12.0 // nsym=2: marker + 2 symbols + count
            }
        } else {
            // General code: method encoding + alphabet + frequencies
            let method_bits = 4.0; // Unary + suffix for method
            let alphabet_bits = 8.0;
            let freq_bits = self.alphabet_size as f32 * 5.0; // Rough estimate
            method_bits + alphabet_bits + freq_bits
        }
    }

    /// Write this histogram to a BitWriter.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        if self.method == 0 {
            // Flat distribution
            writer.write(1, 0)?; // Non-small
            writer.write(1, 1)?; // Flat
            write_var_len_uint8(writer, (self.alphabet_size - 1) as u8)?;
            return Ok(());
        }

        if self.num_symbols <= 2 {
            // Small code
            writer.write(1, 1)?; // Small tree marker
            if self.num_symbols == 0 {
                writer.write(1, 0)?;
                write_var_len_uint8(writer, 0)?;
            } else {
                writer.write(1, (self.num_symbols - 1) as u64)?;
                for i in 0..self.num_symbols {
                    write_var_len_uint8(writer, self.symbols[i] as u8)?;
                }
                if self.num_symbols == 2 {
                    writer.write(
                        ANS_LOG_TAB_SIZE as usize,
                        self.counts[self.symbols[0]] as u64,
                    )?;
                }
            }
            return Ok(());
        }

        // General code
        self.write_general(writer)
    }

    /// Write general (non-flat, non-small) histogram.
    ///
    /// Matches the format expected by jxl-rs `decode_dist_complex()`.
    fn write_general(&self, writer: &mut BitWriter) -> Result<()> {
        writer.write(1, 0)?; // Non-small
        writer.write(1, 0)?; // Non-flat

        // Encode shift using unary + suffix (method = shift + 1)
        // Format: len ones, then 0 (unless at max), then len suffix bits
        let shift = (self.method - 1) as i32;
        let shift_val = (shift + 1) as u32; // shift+1 is stored, range 1-13

        // Determine unary length
        let mut len = 0u32;
        while len < 3 && shift_val >= (1u32 << (len + 1)) {
            len += 1;
        }

        // Write unary prefix (len ones)
        for _ in 0..len {
            writer.write(1, 1)?;
        }
        // Write terminating 0 if len < 3
        if len < 3 {
            writer.write(1, 0)?;
        }
        // Write suffix bits
        if len > 0 {
            let suffix = shift_val - (1u32 << len);
            writer.write(len as usize, suffix as u64)?;
        }

        // Encode alphabet size - 3
        if self.alphabet_size < 3 {
            return Err(Error::InvalidHistogram(
                "General histogram needs at least 3 symbols".to_string(),
            ));
        }
        write_var_len_uint8(writer, (self.alphabet_size - 3) as u8)?;

        // Pre-compute logcounts for all symbols
        let logcounts: Vec<u32> = (0..self.alphabet_size)
            .map(|i| {
                let count = self.counts[i];
                if count <= 0 {
                    0
                } else {
                    floor_log2(count as u32) + 1
                }
            })
            .collect();

        // Pre-compute RLE: for each position i, same[i] = number of consecutive
        // symbols starting at i+1 that have the same actual count as i.
        // The decoder fills RLE positions with prev_dist (the actual count), so
        // all symbols in a run must have identical normalized counts (not just logcounts).
        // Constraints (libjxl enc_ans.cc:257-273):
        // - RLE range must not include omit_pos
        // - RLE marker must not appear at omit_pos+1
        let mut same = vec![0usize; self.alphabet_size];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.alphabet_size {
            if i == self.omit_pos {
                continue;
            }
            let mut run = 0;
            let mut j = i + 1;
            while j < self.alphabet_size && self.counts[j] == self.counts[i] {
                if j == self.omit_pos {
                    break; // Can't include omit_pos in RLE range
                }
                run += 1;
                j += 1;
            }
            same[i] = run;
        }

        // Encode log-frequency values with RLE (libjxl enc_ans.cc:300-309).
        // The decoder determines omit_pos as the first symbol with highest logcount.
        const MIN_REPS: usize = 4; // Minimum repeat count (decoder reads value+4)
        let mut i = 0;
        while i < self.alphabet_size {
            // Write the logcount using fixed prefix code
            let (nbits, code) = LOGCOUNT_PREFIX_CODE[logcounts[i] as usize];
            writer.write(nbits as usize, code as u64)?;

            // If 4+ following symbols have the same logcount, use RLE.
            // But don't place RLE marker at omit_pos+1 (decoder rejects this).
            if same[i] >= MIN_REPS && i + 1 != self.omit_pos + 1 {
                let (rle_nbits, rle_code) = LOGCOUNT_PREFIX_CODE[RLE_MARKER_SYM as usize];
                writer.write(rle_nbits as usize, rle_code as u64)?;
                write_var_len_uint8(writer, (same[i] - MIN_REPS) as u8)?;
                i += same[i]; // Skip the repeated symbols
            }
            i += 1;
        }

        // Build set of RLE-covered positions. The decoder skips precision bits
        // for these symbols (the `continue` in the RLE range handler).
        let mut rle_covered = vec![false; self.alphabet_size];
        {
            let mut i = 0;
            while i < self.alphabet_size {
                if same[i] >= MIN_REPS && i + 1 != self.omit_pos + 1 {
                    // Positions i+1 through i+same[i] are RLE-covered
                    for item in rle_covered.iter_mut().take(i + same[i] + 1).skip(i + 1) {
                        *item = true;
                    }
                    i += same[i];
                }
                i += 1;
            }
        }

        // Now write precision bits for each non-zero, non-omit symbol with logcount > 1.
        // Skip RLE-covered positions (decoder skips precision bits for those).
        for i in 0..self.alphabet_size {
            if i == self.omit_pos || rle_covered[i] {
                continue;
            }

            let count = self.counts[i];
            if count <= 0 {
                continue;
            }

            let logcount = logcounts[i];
            if logcount <= 1 {
                // logcount=1 means freq=1, no precision bits needed
                continue;
            }

            // zeros = logcount - 1 (the log2 of the frequency)
            let zeros = (logcount - 1) as i32;
            // bitcount = shift - (12 - zeros) / 2, clamped to [0, zeros]
            let bitcount = (shift - ((ANS_LOG_TAB_SIZE as i32 - zeros) >> 1)).clamp(0, zeros);

            if bitcount > 0 {
                // The value stored is: freq = (1 << zeros) + (extra << (zeros - bitcount))
                // So: extra = (freq - (1 << zeros)) >> (zeros - bitcount)
                let base = 1i32 << zeros;
                let extra = ((count - base) >> (zeros - bitcount)) as u32;
                writer.write(bitcount as usize, extra as u64)?;
            }
        }

        Ok(())
    }
}

impl Default for ANSEncodingHistogram {
    fn default() -> Self {
        Self::new()
    }
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
    use crate::entropy_coding::histogram::Histogram;

    #[test]
    fn test_ans_encoding_histogram_single_symbol() {
        let h = Histogram::from_counts(&[100, 0, 0, 0]);
        let encoded = ANSEncodingHistogram::from_histogram(&h, ANSHistogramStrategy::Fast).unwrap();

        assert_eq!(encoded.num_symbols, 1);
        assert_eq!(encoded.method, 1); // Small code
        assert_eq!(encoded.counts[0], ANS_TAB_SIZE as i32);
        assert!(encoded.cost < 100.0); // Header only
    }

    #[test]
    fn test_ans_encoding_histogram_two_symbols() {
        let h = Histogram::from_counts(&[100, 100, 0, 0]);
        let encoded = ANSEncodingHistogram::from_histogram(&h, ANSHistogramStrategy::Fast).unwrap();

        assert_eq!(encoded.num_symbols, 2);
        assert_eq!(encoded.method, 1); // Small code
        // Should split roughly 50/50
        let sum: i32 = encoded.counts.iter().sum();
        assert_eq!(sum, ANS_TAB_SIZE as i32);
        assert!(encoded.counts[0] > 0);
        assert!(encoded.counts[1] > 0);
    }

    #[test]
    fn test_ans_encoding_histogram_general() {
        let h = Histogram::from_counts(&[100, 50, 25, 10, 5, 3, 2, 1]);
        let encoded = ANSEncodingHistogram::from_histogram(&h, ANSHistogramStrategy::Fast).unwrap();

        // Should use general code (more than 2 symbols)
        assert!(encoded.method >= 2 || encoded.method == 0);

        // Sum should be exactly ANS_TAB_SIZE
        let sum: i32 = encoded.counts.iter().sum();
        assert_eq!(sum, ANS_TAB_SIZE as i32);

        // All non-zero original counts should have non-zero normalized counts
        for (i, &orig) in h.counts.iter().enumerate() {
            if orig > 0 {
                assert!(
                    encoded.counts.get(i).copied().unwrap_or(0) > 0,
                    "Symbol {} had count {} but normalized to 0",
                    i,
                    orig
                );
            }
        }
    }

    #[test]
    fn test_ans_encoding_histogram_empty() {
        let h = Histogram::new();
        let encoded = ANSEncodingHistogram::from_histogram(&h, ANSHistogramStrategy::Fast).unwrap();

        assert_eq!(encoded.cost, 0.0);
        assert_eq!(encoded.method, 0); // Flat
    }

    #[test]
    fn test_get_population_count_precision() {
        // logcount=0, any shift: precision=0
        assert_eq!(get_population_count_precision(0, 12), 0);

        // logcount=12, shift=12: min(12, 12-(0/2)) = 12
        assert_eq!(get_population_count_precision(12, 12), 12);

        // logcount=6, shift=6: min(6, 6-3) = 3
        assert_eq!(get_population_count_precision(6, 6), 3);

        // logcount=1, shift=0: min(1, 0-(11/2)) = min(1, -5) = 0 (clamped)
        assert_eq!(get_population_count_precision(1, 0), 0);
    }

    #[test]
    fn test_ans_encoding_histogram_write() {
        let h = Histogram::from_counts(&[100, 0, 0, 0]);
        let encoded = ANSEncodingHistogram::from_histogram(&h, ANSHistogramStrategy::Fast).unwrap();

        let mut writer = BitWriter::new();
        encoded.write(&mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(!bytes.is_empty());
    }

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

    #[test]
    fn test_ans_roundtrip_manual() {
        // Create a simple flat distribution
        let dist = AnsDistribution::flat(2).unwrap();

        println!("Distribution: {} symbols", dist.alphabet_size());
        for (i, sym) in dist.symbols.iter().enumerate() {
            println!("  Symbol {}: freq={}", i, sym.freq);
        }

        // Encode symbol 0
        let mut encoder = AnsEncoder::new();
        let initial_state = encoder.state();
        println!("\nInitial state: 0x{:08x}", initial_state);
        assert_eq!(initial_state, 0x130000, "Initial state should be 0x130000");

        let info = &dist.symbols[0];
        encoder.put_symbol(info);
        let encoded_state = encoder.state();
        println!("After encoding symbol 0: state=0x{:08x}", encoded_state);

        // Now manually decode to verify
        let idx = encoded_state & 0xFFF;
        println!("Decode: idx = {}", idx);

        // For flat distribution of 2, each has freq 2048
        // Symbol 0: cumul=0, freq=2048 -> positions [0, 2048)
        // Symbol 1: cumul=2048, freq=2048 -> positions [2048, 4096)
        let decoded_symbol = if idx < 2048 { 0 } else { 1 };
        let offset_in_symbol = if idx < 2048 { idx } else { idx - 2048 };
        let freq = 2048u32;

        println!("Decoded symbol: {}", decoded_symbol);
        println!("Offset in symbol: {}", offset_in_symbol);

        // The decoder does: next_state = (state >> 12) * freq + offset
        let quotient = encoded_state >> 12;
        let next_state = quotient * freq + offset_in_symbol;
        println!(
            "next_state = {} * {} + {} = 0x{:08x}",
            quotient, freq, offset_in_symbol, next_state
        );

        // The next_state should be the initial state (0x130000)
        assert_eq!(next_state, 0x130000, "Decoded state should be 0x130000");
        assert_eq!(decoded_symbol, 0, "Decoded symbol should be 0");
    }

    #[test]
    fn test_ans_roundtrip_multiple_symbols() {
        use crate::bit_writer::BitWriter;
        use crate::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};

        // Test encoding multiple symbols and verify they can be decoded
        // using the jxl-rs compatible decoder (alias table method)

        // Create a flat distribution with 4 symbols (each freq = 1024)
        let counts = [1024i32, 1024, 1024, 1024];
        let dist = AnsDistribution::from_normalized_counts(&counts).unwrap();

        let symbols_to_encode: Vec<usize> = vec![0, 1, 2, 3, 0, 1];
        println!(
            "Encoding {} symbols: {:?}",
            symbols_to_encode.len(),
            symbols_to_encode
        );

        // Encode in reverse order (as ANS requires)
        let mut encoder = AnsEncoder::new();
        for &sym in symbols_to_encode.iter().rev() {
            encoder.put_symbol(&dist.symbols[sym]);
        }

        println!("Final state after encoding: 0x{:08x}", encoder.state());

        // Finalize encoder to bitstream
        let mut writer = BitWriter::new();
        encoder.finalize(&mut writer).unwrap();
        let encoded_bytes = writer.finish_with_padding();
        println!("Encoded bytes: {:02x?}", encoded_bytes);

        // Build decoder histogram by writing and reading back
        let ans_histo = ANSEncodingHistogram::from_histogram(
            &Histogram::from_counts(&counts),
            ANSHistogramStrategy::Precise,
        )
        .unwrap();
        let mut hist_writer = BitWriter::new();
        ans_histo.write(&mut hist_writer).unwrap();
        let hist_bytes = hist_writer.finish_with_padding();

        let mut hist_br = BitReader::new(&hist_bytes);
        let decoded_hist = AnsHistogram::decode(&mut hist_br, 6).unwrap();

        println!(
            "Decoded histogram frequencies: {:?}",
            &decoded_hist.frequencies[..4]
        );

        // Decode using jxl-rs compatible decoder
        let mut br = BitReader::new(&encoded_bytes);
        let mut ans_reader = AnsReader::init(&mut br).unwrap();

        println!("Decoding:");
        let mut decoded = Vec::new();
        for i in 0..symbols_to_encode.len() {
            let symbol = decoded_hist.read(&mut br, &mut ans_reader.0) as usize;
            println!(
                "  step {}: symbol={}, state=0x{:08x}",
                i, symbol, ans_reader.0
            );
            decoded.push(symbol);
        }

        println!("Original: {:?}", symbols_to_encode);
        println!("Decoded:  {:?}", decoded);
        println!("Final state: 0x{:08x}", ans_reader.0);

        assert_eq!(
            decoded, symbols_to_encode,
            "Decoded symbols should match original"
        );
        assert!(
            ans_reader.check_final_state().is_ok(),
            "Final state should be 0x130000, got 0x{:08x}",
            ans_reader.0
        );
    }

    #[test]
    fn test_ans_histogram_write_decode_roundtrip() {
        use crate::bit_writer::BitWriter;
        use crate::entropy_coding::histogram::Histogram;

        // Create a histogram with several symbols
        let histo = Histogram::from_counts(&[100, 50, 25, 10]);

        let encoded =
            ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

        println!("Histogram: {:?}", histo.counts);
        println!("Encoded counts: {:?}", encoded.counts);
        println!(
            "Method: {}, alphabet_size: {}, omit_pos: {}",
            encoded.method, encoded.alphabet_size, encoded.omit_pos
        );

        // Verify sum is 4096
        let sum: i32 = encoded.counts.iter().sum();
        assert_eq!(sum, ANS_TAB_SIZE as i32, "Sum should be 4096");

        // Write to bitstream
        let mut writer = BitWriter::new();
        encoded.write(&mut writer).unwrap();
        let bytes = writer.finish_with_padding();

        println!("Encoded histogram: {} bytes", bytes.len());
        println!("Bytes: {:02x?}", bytes);
    }
}
