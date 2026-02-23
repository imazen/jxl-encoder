// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Histogram data structure with entropy calculations.
//!
//! Ported from libjxl `lib/jxl/enc_ans_params.h` and `lib/jxl/enc_cluster.cc`.

use core::cell::Cell;

/// Alignment for SIMD-friendly histogram operations.
/// Matches libjxl's `Histogram::kRounding`.
pub const HISTOGRAM_ROUNDING: usize = 8;

/// Minimum distance threshold for creating distinct clusters.
/// Below this threshold, histograms are considered similar enough to merge.
pub const MIN_DISTANCE_FOR_DISTINCT: f32 = 48.0;

/// A histogram counting symbol occurrences.
///
/// This is the encoder-side histogram structure, corresponding to libjxl's
/// `Histogram` class in `enc_ans_params.h`.
#[derive(Clone, Debug)]
pub struct Histogram {
    /// Symbol counts (aligned to HISTOGRAM_ROUNDING).
    /// Uses i32 to match libjxl's `ANSHistBin` type.
    pub counts: Vec<i32>,
    /// Sum of all counts.
    pub total_count: usize,
    /// Cached entropy value.
    /// WARNING: Not automatically kept up-to-date - call `shannon_entropy()` to refresh.
    entropy: Cell<f32>,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// Creates an empty histogram.
    pub fn new() -> Self {
        Self {
            counts: Vec::new(),
            total_count: 0,
            entropy: Cell::new(0.0),
        }
    }

    /// Creates a histogram with pre-allocated capacity for `length` symbols.
    /// The capacity is rounded up to HISTOGRAM_ROUNDING.
    pub fn with_capacity(length: usize) -> Self {
        let rounded_len = div_ceil(length, HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING;
        Self {
            counts: vec![0; rounded_len],
            total_count: 0,
            entropy: Cell::new(0.0),
        }
    }

    /// Creates a histogram from a slice of counts.
    pub fn from_counts(counts: &[i32]) -> Self {
        let total: i32 = counts.iter().sum();
        let rounded_len = div_ceil(counts.len(), HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING;
        let mut result_counts = vec![0i32; rounded_len];
        result_counts[..counts.len()].copy_from_slice(counts);

        Self {
            counts: result_counts,
            total_count: total as usize,
            entropy: Cell::new(0.0),
        }
    }

    /// Creates a flat (uniform) histogram.
    pub fn flat(length: usize, total_count: usize) -> Self {
        let base = (total_count / length) as i32;
        let remainder = total_count % length;

        let rounded_len = div_ceil(length, HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING;
        let mut counts = vec![0i32; rounded_len];

        for (i, count) in counts.iter_mut().enumerate().take(length) {
            *count = base + if i < remainder { 1 } else { 0 };
        }

        Self {
            counts,
            total_count,
            entropy: Cell::new(0.0),
        }
    }

    /// Clears all counts.
    pub fn clear(&mut self) {
        self.counts.clear();
        self.total_count = 0;
        self.entropy.set(0.0);
    }

    /// Add one occurrence of a symbol.
    pub fn add(&mut self, symbol: usize) {
        self.ensure_capacity(symbol + 1);
        self.counts[symbol] += 1;
        self.total_count += 1;
    }

    /// Ensures the histogram can hold at least `length` symbols.
    pub fn ensure_capacity(&mut self, length: usize) {
        let rounded_len = div_ceil(length, HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING;
        if self.counts.len() < rounded_len {
            self.counts.resize(rounded_len, 0);
        }
    }

    /// Fast add (caller must ensure capacity first).
    #[inline]
    pub fn fast_add(&mut self, symbol: usize) {
        debug_assert!(symbol < self.counts.len());
        self.counts[symbol] += 1;
    }

    /// Add another histogram's counts to this one.
    pub fn add_histogram(&mut self, other: &Histogram) {
        if other.counts.len() > self.counts.len() {
            self.counts.resize(other.counts.len(), 0);
        }
        for (i, &count) in other.counts.iter().enumerate() {
            self.counts[i] += count;
        }
        self.total_count += other.total_count;
    }

    /// Trim trailing zeros and update total_count.
    /// Should be called after a sequence of `fast_add` calls.
    pub fn condition(&mut self) {
        // Find the last non-zero position
        let mut last_nonzero: i32 = -1;
        let mut total: i64 = 0;

        for (i, &count) in self.counts.iter().enumerate() {
            total += count as i64;
            if count != 0 {
                last_nonzero = i as i32;
            }
        }

        // Resize to rounded length past last non-zero
        let new_len = if last_nonzero >= 0 {
            div_ceil((last_nonzero + 1) as usize, HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING
        } else {
            0
        };
        self.counts.resize(new_len, 0);
        self.total_count = total as usize;
    }

    /// Compute Shannon entropy: -sum(count * log2(count / total)).
    /// Result is in bits (not nats).
    ///
    /// Formula: sum of -(count/total) * log2(count/total) * total
    ///        = sum of -count * log2(count/total)
    ///        = sum of -count * (log2(count) - log2(total))
    ///        = sum of -count * log2(count) + count * log2(total)
    ///        = -sum(count * log2(count)) + total * log2(total)
    ///
    /// libjxl uses: -count * log2(count / total), excluding when count == total.
    pub fn shannon_entropy(&self) -> f32 {
        if self.total_count == 0 {
            self.entropy.set(0.0);
            return 0.0;
        }

        let entropy = jxl_simd::shannon_entropy_bits(&self.counts, self.total_count);
        self.entropy.set(entropy);
        entropy
    }

    /// Get the cached entropy value.
    /// Call `shannon_entropy()` first to ensure it's up-to-date.
    pub fn cached_entropy(&self) -> f32 {
        self.entropy.get()
    }

    /// Set the cached entropy value (used when loading from test data).
    pub fn set_cached_entropy(&self, entropy: f32) {
        self.entropy.set(entropy);
    }

    /// Alphabet size (highest non-zero symbol + 1).
    pub fn alphabet_size(&self) -> usize {
        for i in (0..self.counts.len()).rev() {
            if self.counts[i] > 0 {
                return i + 1;
            }
        }
        0
    }

    /// Returns the index of the maximum symbol with non-zero count.
    pub fn max_symbol(&self) -> usize {
        if self.total_count == 0 {
            return 0;
        }
        for i in (1..self.counts.len()).rev() {
            if self.counts[i] > 0 {
                return i;
            }
        }
        0
    }

    /// Check if histogram is empty (all zeros).
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Copy contents from another histogram, reusing this histogram's allocation.
    ///
    /// Unlike `clone()`, this avoids allocating a new `Vec` when `self` already
    /// has sufficient capacity.
    pub fn copy_from(&mut self, source: &Histogram) {
        let src_len = source.counts.len();
        if self.counts.len() < src_len {
            self.counts.resize(src_len, 0);
        }
        self.counts[..src_len].copy_from_slice(&source.counts[..src_len]);
        if self.counts.len() > src_len {
            self.counts[src_len..].fill(0);
        }
        self.total_count = source.total_count;
        self.entropy.set(source.cached_entropy());
    }
}

/// Scratch buffer for `histogram_distance` to avoid per-call heap allocation.
///
/// Reuse across multiple calls in hot clustering loops.
pub struct DistanceScratch {
    combined_counts: Vec<i32>,
}

impl Default for DistanceScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceScratch {
    /// Create a new scratch buffer.
    pub fn new() -> Self {
        Self {
            combined_counts: Vec::new(),
        }
    }

    /// Ensure the scratch buffer has at least `len` elements.
    /// Does NOT zero — caller is responsible for writing all used positions.
    #[inline]
    fn ensure_capacity(&mut self, len: usize) {
        if self.combined_counts.len() < len {
            self.combined_counts.resize(len, 0);
        }
    }
}

/// Distance between two histograms (for clustering).
///
/// This measures how many extra bits are needed to encode the combined
/// distribution vs encoding them separately. Lower = more similar.
///
/// Formula: entropy(combined) - entropy(a) - entropy(b)
///
/// IMPORTANT: Both histograms must have their entropy pre-computed
/// (call `shannon_entropy()` first).
pub fn histogram_distance(a: &Histogram, b: &Histogram) -> f32 {
    let mut scratch = DistanceScratch::new();
    histogram_distance_reuse(a, b, &mut scratch)
}

/// Like [`histogram_distance`] but reuses a scratch buffer to avoid allocation.
pub fn histogram_distance_reuse(
    a: &Histogram,
    b: &Histogram,
    scratch: &mut DistanceScratch,
) -> f32 {
    if a.total_count == 0 || b.total_count == 0 {
        return 0.0;
    }

    let combined_total = a.total_count + b.total_count;
    let a_len = a.counts.len();
    let b_len = b.counts.len();
    let max_len = a_len.max(b_len);

    // Build combined counts (HISTOGRAM_ROUNDING-aligned for SIMD)
    let aligned_len = div_ceil(max_len, HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING;
    scratch.ensure_capacity(aligned_len);
    let combined_counts = &mut scratch.combined_counts[..aligned_len];

    // Add overlapping region using zip (no per-element bounds checks)
    let min_len = a_len.min(b_len);
    for ((slot, &ac), &bc) in combined_counts[..min_len]
        .iter_mut()
        .zip(&a.counts[..min_len])
        .zip(&b.counts[..min_len])
    {
        *slot = ac + bc;
    }
    // Copy non-overlapping tail from whichever histogram is longer
    if a_len > min_len {
        combined_counts[min_len..a_len].copy_from_slice(&a.counts[min_len..a_len]);
    } else if b_len > min_len {
        combined_counts[min_len..b_len].copy_from_slice(&b.counts[min_len..b_len]);
    }
    // Zero only the SIMD padding tail (positions max_len..aligned_len)
    if max_len < aligned_len {
        combined_counts[max_len..aligned_len].fill(0);
    }

    let combined_entropy = jxl_simd::shannon_entropy_bits(combined_counts, combined_total);

    // Distance = combined_entropy - a.entropy - b.entropy
    combined_entropy - a.cached_entropy() - b.cached_entropy()
}

/// KL divergence: cost of encoding `actual` using `coding` histogram.
///
/// Returns the extra bits needed to encode `actual`'s symbols using
/// `coding`'s probability distribution, compared to using `actual`'s
/// own distribution.
///
/// Returns infinity if `actual` has symbols not present in `coding`.
///
/// IMPORTANT: Both histograms must have their entropy pre-computed.
pub fn histogram_kl_divergence(actual: &Histogram, coding: &Histogram) -> f32 {
    if actual.total_count == 0 {
        return 0.0;
    }
    if coding.total_count == 0 {
        return f32::INFINITY;
    }

    let coding_inv = 1.0 / coding.total_count as f32;
    let mut cost = 0.0f32;

    for (i, &count) in actual.counts.iter().enumerate() {
        if count > 0 {
            let coding_count = coding.counts.get(i).copied().unwrap_or(0);
            if coding_count == 0 {
                // Symbol in actual but not in coding -> infinite cost
                return f32::INFINITY;
            }
            let coding_prob = coding_count as f32 * coding_inv;
            // Cost: -count * log2(coding_prob)
            cost -= count as f32 * jxl_simd::fast_log2f(coding_prob);
        }
    }

    // KL divergence = cost - entropy(actual)
    cost - actual.cached_entropy()
}

/// Ceiling division.
#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_new() {
        let h = Histogram::new();
        assert!(h.is_empty());
        assert_eq!(h.total_count, 0);
        assert_eq!(h.alphabet_size(), 0);
    }

    #[test]
    fn test_histogram_from_counts() {
        let h = Histogram::from_counts(&[10, 20, 30]);
        assert_eq!(h.total_count, 60);
        assert_eq!(h.alphabet_size(), 3);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_histogram_add() {
        let mut h = Histogram::new();
        h.add(0);
        h.add(0);
        h.add(5);

        assert_eq!(h.total_count, 3);
        assert_eq!(h.counts[0], 2);
        assert_eq!(h.counts[5], 1);
        assert_eq!(h.alphabet_size(), 6);
    }

    #[test]
    fn test_histogram_flat() {
        let h = Histogram::flat(4, 100);
        assert_eq!(h.total_count, 100);
        // 100 / 4 = 25 each
        assert_eq!(h.counts[0], 25);
        assert_eq!(h.counts[1], 25);
        assert_eq!(h.counts[2], 25);
        assert_eq!(h.counts[3], 25);
    }

    #[test]
    fn test_histogram_flat_remainder() {
        let h = Histogram::flat(4, 10);
        assert_eq!(h.total_count, 10);
        // 10 / 4 = 2 remainder 2, so first two get 3, last two get 2
        assert_eq!(h.counts[0], 3);
        assert_eq!(h.counts[1], 3);
        assert_eq!(h.counts[2], 2);
        assert_eq!(h.counts[3], 2);
    }

    #[test]
    fn test_histogram_condition() {
        let mut h = Histogram::with_capacity(100);
        h.fast_add(0);
        h.fast_add(0);
        h.fast_add(5);
        h.condition();

        assert_eq!(h.total_count, 3);
        assert_eq!(h.counts.len(), HISTOGRAM_ROUNDING); // Rounded up from 6
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution: entropy = log2(n) bits per symbol
        let h = Histogram::from_counts(&[100, 100, 100, 100]);
        let entropy = h.shannon_entropy();
        // Expected: 400 * log2(4) = 400 * 2 = 800 bits total
        // But our formula gives bits for the total, which is:
        // sum of -count * log2(count/total) = 4 * (-100 * log2(0.25)) = 4 * 100 * 2 = 800
        assert!((entropy - 800.0).abs() < 0.01, "entropy = {}", entropy);
    }

    #[test]
    fn test_shannon_entropy_skewed() {
        // Single symbol: entropy = 0 (no uncertainty)
        let h = Histogram::from_counts(&[100, 0, 0, 0]);
        let entropy = h.shannon_entropy();
        assert!((entropy - 0.0).abs() < 0.01, "entropy = {}", entropy);
    }

    #[test]
    fn test_shannon_entropy_binary() {
        // Two equal symbols: entropy = n * 1 bit = n
        let h = Histogram::from_counts(&[50, 50]);
        let entropy = h.shannon_entropy();
        // 2 * (-50 * log2(0.5)) = 2 * 50 * 1 = 100
        assert!((entropy - 100.0).abs() < 0.01, "entropy = {}", entropy);
    }

    #[test]
    fn test_histogram_distance_identical() {
        let a = Histogram::from_counts(&[100, 50, 25]);
        let b = Histogram::from_counts(&[100, 50, 25]);
        a.shannon_entropy();
        b.shannon_entropy();

        let dist = histogram_distance(&a, &b);
        // Identical histograms: combined entropy = 2x each entropy
        // So distance = 2*E - E - E = 0
        assert!(dist.abs() < 0.01, "distance = {}", dist);
    }

    #[test]
    fn test_histogram_distance_different() {
        let a = Histogram::from_counts(&[100, 0, 0]);
        let b = Histogram::from_counts(&[0, 0, 100]);
        a.shannon_entropy();
        b.shannon_entropy();

        let dist = histogram_distance(&a, &b);
        // a has entropy 0, b has entropy 0 (single symbol each)
        // Combined has 100 each in symbols 0 and 2
        // Combined entropy = 2 * (-100 * log2(0.5)) = 200
        // Distance = 200 - 0 - 0 = 200
        assert!((dist - 200.0).abs() < 0.01, "distance = {}", dist);
    }

    #[test]
    fn test_histogram_distance_empty() {
        let a = Histogram::new();
        let b = Histogram::from_counts(&[100]);
        a.shannon_entropy();
        b.shannon_entropy();

        let dist = histogram_distance(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let a = Histogram::from_counts(&[100, 50, 25]);
        a.shannon_entropy();

        let div = histogram_kl_divergence(&a, &a);
        assert!(div.abs() < 0.01, "kl = {}", div);
    }

    #[test]
    fn test_kl_divergence_missing_symbol() {
        let a = Histogram::from_counts(&[100, 50, 25]);
        let b = Histogram::from_counts(&[100, 50, 0]); // Missing symbol 2
        a.shannon_entropy();
        b.shannon_entropy();

        let div = histogram_kl_divergence(&a, &b);
        assert!(div.is_infinite(), "kl = {}", div);
    }

    #[test]
    fn test_add_histogram() {
        let mut a = Histogram::from_counts(&[10, 20]);
        let b = Histogram::from_counts(&[5, 10, 15]);

        a.add_histogram(&b);

        assert_eq!(a.total_count, 60);
        assert_eq!(a.counts[0], 15);
        assert_eq!(a.counts[1], 30);
        assert_eq!(a.counts[2], 15);
    }

    #[test]
    fn test_max_symbol() {
        let h = Histogram::from_counts(&[10, 20, 0, 5, 0, 0]);
        assert_eq!(h.max_symbol(), 3);

        let h2 = Histogram::from_counts(&[10]);
        assert_eq!(h2.max_symbol(), 0);

        let h3 = Histogram::new();
        assert_eq!(h3.max_symbol(), 0);
    }
}
