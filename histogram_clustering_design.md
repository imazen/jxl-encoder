# Rust Design Specification: Entropy Coding Pipeline
## From Tokens to Final Bytes (Matching libjxl Exactly)

---

## Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Constants & Parameters](#2-constants--parameters)
3. [Core Data Structures](#3-core-data-structures)
4. [Module 1: Histogram](#4-module-1-histogram)
5. [Module 2: Histogram Clustering](#5-module-2-histogram-clustering)
6. [Module 3: ANS Encoding](#6-module-3-ans-encoding)
7. [Module 4: Context Map](#7-module-4-context-map)
8. [Module 5: Bit Writing](#8-module-5-bit-writing)
9. [Module 6: Orchestration](#9-module-6-orchestration)
10. [Implementation Order](#10-implementation-order)
11. [Test Strategy](#11-test-strategy)
12. [Appendix: Bitstream Format](#12-appendix-bitstream-format)

---

## 1. Overview & Architecture

### Pipeline Flow
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              RUST MODULE ARCHITECTURE                         │
└──────────────────────────────────────────────────────────────────────────────┘

    Input: Vec<Vec<Token>>                    Output: Vec<u8> (bitstream)
              │                                          ▲
              │                                          │
    ┌─────────▼─────────┐                               │
    │   build_histograms │  tokens → Histogram per ctx  │
    └─────────┬─────────┘                               │
              │                                          │
    ┌─────────▼─────────┐                               │
    │ cluster_histograms │  many → few histograms       │
    └─────────┬─────────┘  + context_map                │
              │                                          │
    ┌─────────▼─────────┐                               │
    │ choose_uint_config │  select best hybrid uint     │
    └─────────┬─────────┘                               │
              │                                          │
    ┌─────────▼─────────┐                               │
    │ normalize_histograms│  scale to ANS_TAB_SIZE      │
    └─────────┬─────────┘                               │
              │                                          │
    ┌─────────▼─────────┐                               │
    │ build_alias_tables │  create encoding tables      │
    └─────────┬─────────┘                               │
              │                                          │
    ┌─────────▼─────────┐                               │
    │  encode_histograms │  write histogram data        │
    └─────────┬─────────┘                               │
              │                                          │
    ┌─────────▼─────────┐                               │
    │ encode_context_map │  write context mapping       │
    └─────────┬─────────┘                               │
              │                                          │
    ┌─────────▼─────────────────────────────────────────┤
    │     write_tokens    │  ANS encode all tokens ─────┘
    └─────────────────────┘
```

### Crate Structure
```
jxl-entropy/
├── src/
│   ├── lib.rs              // Public API
│   ├── constants.rs        // ANS_LOG_TAB_SIZE, etc.
│   ├── token.rs            // Token struct
│   ├── histogram.rs        // Histogram struct + entropy
│   ├── cluster.rs          // FastClusterHistograms + refinement
│   ├── hybrid_uint.rs      // HybridUintConfig
│   ├── ans_normalize.rs    // Histogram normalization/rebalancing
│   ├── ans_encode.rs       // ANSCoder, encoding tables
│   ├── alias_table.rs      // AliasTable for encoding
│   ├── context_map.rs      // Context map encoding
│   ├── bitwriter.rs        // Bit-level writing
│   └── lz77.rs             // Optional LZ77 preprocessing
└── tests/
    ├── parity_histogram.rs
    ├── parity_cluster.rs
    ├── parity_ans.rs
    └── integration.rs
```

---

## 2. Constants & Parameters

```rust
// src/constants.rs

/// ANS table size = 2^12 = 4096
pub const ANS_LOG_TAB_SIZE: u32 = 12;
pub const ANS_TAB_SIZE: u32 = 1 << ANS_LOG_TAB_SIZE;  // 4096
pub const ANS_TAB_MASK: u32 = ANS_TAB_SIZE - 1;       // 4095

/// Maximum alphabet sizes
pub const ANS_MAX_ALPHABET_SIZE: usize = 256;
pub const PREFIX_MAX_ALPHABET_SIZE: usize = 4096;

/// Prefix (Huffman) coding parameters
pub const PREFIX_MAX_BITS: usize = 15;

/// ANS state signature (used for initialization and as CRC)
pub const ANS_SIGNATURE: u32 = 0x13;

/// Reciprocal precision for fast division
pub const RECIPROCAL_PRECISION: u32 = 32 + ANS_LOG_TAB_SIZE;  // 44

/// Maximum number of histogram clusters
pub const CLUSTERS_LIMIT: usize = 256;

/// Minimum distance for creating a distinct cluster
pub const MIN_DISTANCE_FOR_DISTINCT: f32 = 48.0;

/// Histogram count alignment (for SIMD)
pub const HISTOGRAM_ROUNDING: usize = 8;
```

---

## 3. Core Data Structures

### 3.1 Token
```rust
// src/token.rs

/// A symbol to be entropy-coded, with its context.
#[derive(Clone, Copy, Debug, Default)]
pub struct Token {
    /// Context ID (determines which histogram to use)
    /// Uses 31 bits
    pub context: u32,

    /// The value to encode
    pub value: u32,

    /// True if this is an LZ77 length token
    pub is_lz77_length: bool,
}

impl Token {
    #[inline]
    pub fn new(context: u32, value: u32) -> Self {
        Self {
            context,
            value,
            is_lz77_length: false,
        }
    }

    #[inline]
    pub fn lz77_length(context: u32, value: u32) -> Self {
        Self {
            context,
            value,
            is_lz77_length: true,
        }
    }
}
```

### 3.2 HybridUintConfig
```rust
// src/hybrid_uint.rs

/// Configuration for hybrid uint encoding.
///
/// Splits a value into:
/// - Direct token bits (for small values < split_token)
/// - Token + extra bits (for larger values)
///
/// The token encodes: exponent + msb_in_token MSBs + lsb_in_token LSBs
/// Extra bits encode the remaining middle bits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridUintConfig {
    pub split_exponent: u32,
    pub split_token: u32,      // = 1 << split_exponent
    pub msb_in_token: u32,
    pub lsb_in_token: u32,
}

impl Default for HybridUintConfig {
    fn default() -> Self {
        Self::new(4, 2, 0)  // Default: split_exponent=4, msb=2, lsb=0
    }
}

impl HybridUintConfig {
    pub fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        debug_assert!(split_exponent >= msb_in_token + lsb_in_token);
        Self {
            split_exponent,
            split_token: 1 << split_exponent,
            msb_in_token,
            lsb_in_token,
        }
    }

    /// Encode a value into (token, extra_bits_count, extra_bits_value)
    #[inline]
    pub fn encode(&self, value: u32) -> (u32, u32, u32) {
        if value < self.split_token {
            // Small value: encode directly as token
            (value, 0, 0)
        } else {
            // Large value: encode exponent + partial bits in token
            let n = floor_log2(value);
            let m = value - (1 << n);

            let token = self.split_token
                + ((n - self.split_exponent) << (self.msb_in_token + self.lsb_in_token))
                + ((m >> (n - self.msb_in_token)) << self.lsb_in_token)
                + (m & ((1 << self.lsb_in_token) - 1));

            let nbits = n - self.msb_in_token - self.lsb_in_token;
            let bits = (value >> self.lsb_in_token) & ((1u32 << nbits) - 1);

            (token, nbits, bits)
        }
    }

    #[inline]
    pub fn lsb_mask(&self) -> u32 {
        (1 << self.lsb_in_token) - 1
    }
}

#[inline]
fn floor_log2(value: u32) -> u32 {
    debug_assert!(value > 0);
    31 - value.leading_zeros()
}
```

### 3.3 HistogramParams
```rust
// src/histogram.rs

/// Clustering aggressiveness
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClusteringType {
    /// Maximum 4 clusters, fastest encoding
    Fastest,
    /// Balanced speed/compression
    Fast,
    /// Best compression with pair merging refinement
    #[default]
    Best,
}

/// ANS histogram optimization strategy
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ANSHistogramStrategy {
    /// Try only 3 shift values
    Fast,
    /// Try every other shift value
    Approximate,
    /// Try all shift values
    #[default]
    Precise,
}

/// LZ77 preprocessing method
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LZ77Method {
    #[default]
    None,
    RLE,
    LZ77,
    Optimal,
}

/// Hybrid uint configuration selection
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HybridUintMethod {
    None,       // Use default config
    K000,       // Force (0,0,0)
    Fast,       // Try a few options
    ContextMap, // Optimized for context maps
    #[default]
    Best,       // Try many options
}

/// Full configuration for histogram building and encoding
#[derive(Clone, Debug)]
pub struct HistogramParams {
    pub clustering: ClusteringType,
    pub uint_method: HybridUintMethod,
    pub lz77_method: LZ77Method,
    pub ans_histogram_strategy: ANSHistogramStrategy,
    pub max_histograms: usize,
    pub force_huffman: bool,
    pub streaming_mode: bool,
}

impl Default for HistogramParams {
    fn default() -> Self {
        Self {
            clustering: ClusteringType::Best,
            uint_method: HybridUintMethod::Best,
            lz77_method: LZ77Method::RLE,
            ans_histogram_strategy: ANSHistogramStrategy::Precise,
            max_histograms: usize::MAX,
            force_huffman: false,
            streaming_mode: false,
        }
    }
}

impl HistogramParams {
    /// Create params based on speed tier
    pub fn from_speed(tier: SpeedTier, num_contexts: usize) -> Self {
        // Match libjxl's HistogramParams(SpeedTier, size_t) constructor
        let mut params = Self::default();

        if tier > SpeedTier::Falcon {
            params.clustering = ClusteringType::Fastest;
            params.lz77_method = LZ77Method::None;
        } else if tier > SpeedTier::Tortoise {
            params.clustering = ClusteringType::Fast;
        }

        if tier > SpeedTier::Tortoise {
            params.uint_method = HybridUintMethod::None;
        }

        if tier >= SpeedTier::Squirrel {
            params.ans_histogram_strategy = ANSHistogramStrategy::Approximate;
        }

        params
    }

    pub fn uint_config(&self) -> HybridUintConfig {
        match self.uint_method {
            HybridUintMethod::ContextMap => HybridUintConfig::new(2, 0, 1),
            HybridUintMethod::K000 => HybridUintConfig::new(0, 0, 0),
            _ => HybridUintConfig::default(), // (4, 2, 0)
        }
    }
}
```

---

## 4. Module 1: Histogram

```rust
// src/histogram.rs

use std::cell::Cell;

/// A histogram counting symbol occurrences.
///
/// Counts are stored aligned to HISTOGRAM_ROUNDING for SIMD efficiency.
#[derive(Clone, Debug)]
pub struct Histogram {
    /// Symbol counts (aligned to HISTOGRAM_ROUNDING)
    pub counts: Vec<i32>,

    /// Sum of all counts
    pub total_count: usize,

    /// Cached entropy value (not always up-to-date!)
    /// Use `shannon_entropy()` to compute fresh value
    entropy: Cell<f32>,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            counts: Vec::new(),
            total_count: 0,
            entropy: Cell::new(0.0),
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(length: usize) -> Self {
        let aligned_len = div_ceil(length, HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING;
        Self {
            counts: vec![0; aligned_len],
            total_count: 0,
            entropy: Cell::new(0.0),
        }
    }

    /// Create a flat histogram where all symbols have equal probability
    /// Implements libjxl's CreateFlatHistogram
    pub fn flat(length: usize, total_count: i32) -> Self {
        assert!(length > 0);
        assert!(length <= total_count as usize);

        let count = total_count / length as i32;
        let rem_counts = (total_count % length as i32) as usize;

        let aligned_len = div_ceil(length, HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING;
        let mut counts = vec![0i32; aligned_len];

        for i in 0..length {
            counts[i] = count + if i < rem_counts { 1 } else { 0 };
        }

        Self {
            counts,
            total_count: total_count as usize,
            entropy: Cell::new(0.0),
        }
    }

    /// Clear all counts
    pub fn clear(&mut self) {
        self.counts.clear();
        self.total_count = 0;
        self.entropy.set(0.0);
    }

    /// Add one occurrence of a symbol
    pub fn add(&mut self, symbol: usize) {
        self.ensure_capacity(symbol + 1);
        self.counts[symbol] += 1;
        self.total_count += 1;
    }

    /// Fast add (caller must ensure capacity)
    #[inline]
    pub fn fast_add(&mut self, symbol: usize) {
        debug_assert!(symbol < self.counts.len());
        self.counts[symbol] += 1;
    }

    /// Ensure counts vector can hold `length` symbols
    pub fn ensure_capacity(&mut self, length: usize) {
        let aligned_len = div_ceil(length, HISTOGRAM_ROUNDING) * HISTOGRAM_ROUNDING;
        if self.counts.len() < aligned_len {
            self.counts.resize(aligned_len, 0);
        }
    }

    /// Condition the histogram: trim trailing zeros, update total_count
    /// Implements libjxl's HistogramCondition
    pub fn condition(&mut self) {
        // Find last non-zero position (in chunks of HISTOGRAM_ROUNDING)
        let mut nz_pos: isize = -(HISTOGRAM_ROUNDING as isize);
        let mut total = 0i32;

        for (i, chunk) in self.counts.chunks(HISTOGRAM_ROUNDING).enumerate() {
            let chunk_sum: i32 = chunk.iter().sum();
            total += chunk_sum;

            // Check if any element in chunk is non-zero
            if chunk.iter().any(|&c| c != 0) {
                nz_pos = (i * HISTOGRAM_ROUNDING) as isize;
            }
        }

        let new_len = (nz_pos + HISTOGRAM_ROUNDING as isize) as usize;
        self.counts.truncate(new_len.max(0));
        self.total_count = total as usize;
    }

    /// Add another histogram's counts to this one
    pub fn add_histogram(&mut self, other: &Histogram) {
        if other.counts.len() > self.counts.len() {
            self.counts.resize(other.counts.len(), 0);
        }
        for (dst, src) in self.counts.iter_mut().zip(other.counts.iter()) {
            *dst += *src;
        }
        self.total_count += other.total_count;
    }

    /// Size of alphabet (highest non-zero symbol + 1)
    pub fn alphabet_size(&self) -> usize {
        for i in (0..self.counts.len()).rev() {
            if self.counts[i] > 0 {
                return i + 1;
            }
        }
        0
    }

    /// Maximum symbol with non-zero count
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

    /// Compute Shannon entropy: sum of -p*log2(p) for all symbols
    /// Implements libjxl's HistogramEntropy (SIMD version simplified)
    pub fn shannon_entropy(&self) -> f32 {
        if self.total_count == 0 {
            return 0.0;
        }

        let inv_total = 1.0 / self.total_count as f32;
        let total_f = self.total_count as f32;

        let mut entropy = 0.0f32;
        for &count in &self.counts {
            if count > 0 && count as usize != self.total_count {
                let p = count as f32 * inv_total;
                entropy -= count as f32 * p.log2();
            }
        }

        self.entropy.set(entropy);
        entropy
    }

    /// Estimate cost in bits to encode this histogram + data using ANS
    /// Implements libjxl's Histogram::ANSPopulationCost
    pub fn ans_population_cost(&self) -> Result<f32, Error> {
        if self.counts.len() > ANS_MAX_ALPHABET_SIZE {
            return Ok(f32::MAX);
        }

        let normalized = ANSEncodingHistogram::compute_best(
            self,
            ANSHistogramStrategy::Fast,
        )?;

        Ok(normalized.cost())
    }
}

/// Distance between two histograms (for clustering)
/// Implements libjxl's HistogramDistance
pub fn histogram_distance(a: &Histogram, b: &Histogram) -> f32 {
    if a.total_count == 0 || b.total_count == 0 {
        return 0.0;
    }

    let combined_total = (a.total_count + b.total_count) as f32;
    let inv_total = 1.0 / combined_total;

    let max_len = a.counts.len().max(b.counts.len());
    let mut distance = 0.0f32;

    for i in 0..max_len {
        let a_count = a.counts.get(i).copied().unwrap_or(0);
        let b_count = b.counts.get(i).copied().unwrap_or(0);
        let combined = (a_count + b_count) as f32;

        if combined > 0.0 && combined < combined_total {
            distance -= combined * (combined * inv_total).log2();
        }
    }

    distance - a.entropy.get() - b.entropy.get()
}

/// KL divergence: cost of encoding `actual` using `coding` histogram
/// Implements libjxl's HistogramKLDivergence
pub fn histogram_kl_divergence(actual: &Histogram, coding: &Histogram) -> f32 {
    if actual.total_count == 0 {
        return 0.0;
    }
    if coding.total_count == 0 {
        return f32::INFINITY;
    }

    let coding_inv = 1.0 / coding.total_count as f32;
    let mut cost = 0.0f32;

    for i in 0..actual.counts.len() {
        let actual_count = actual.counts[i];
        if actual_count == 0 {
            continue;
        }

        let coding_count = coding.counts.get(i).copied().unwrap_or(0);
        if coding_count == 0 {
            return f32::INFINITY;
        }

        let coding_prob = coding_count as f32 * coding_inv;
        cost += actual_count as f32 * (-coding_prob.log2());
    }

    cost - actual.entropy.get()
}

#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
```

---

## 5. Module 2: Histogram Clustering

```rust
// src/cluster.rs

use crate::{
    Histogram, HistogramParams, ClusteringType,
    histogram_distance, histogram_kl_divergence,
    MIN_DISTANCE_FOR_DISTINCT,
};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Result of clustering histograms
pub struct ClusterResult {
    /// The clustered histograms
    pub histograms: Vec<Histogram>,
    /// Mapping from input histogram index to cluster index
    pub symbols: Vec<u32>,
}

/// Cluster similar histograms together.
///
/// Implements libjxl's ClusterHistograms
pub fn cluster_histograms(
    params: &HistogramParams,
    input: &[Histogram],
    max_histograms: usize,
    existing: &mut Vec<Histogram>,
) -> Result<Vec<u32>, Error> {
    let prev_histograms = existing.len();
    let mut max_histograms = max_histograms
        .min(params.max_histograms)
        .min(input.len());

    if params.clustering == ClusteringType::Fastest {
        max_histograms = max_histograms.min(4);
    }

    // Phase 1: Fast k-means-like clustering
    let mut symbols = fast_cluster_histograms(
        input,
        prev_histograms + max_histograms,
        existing,
    )?;

    // Phase 2: Pair merging refinement (only for kBest and new histograms)
    if prev_histograms == 0 && params.clustering == ClusteringType::Best {
        refine_clusters_by_merging(existing, &mut symbols)?;
    }

    // Phase 3: Reindex to canonical form
    histogram_reindex(existing, prev_histograms, &mut symbols);

    Ok(symbols)
}

/// Fast clustering using a k-means-like approach.
///
/// Algorithm:
/// 1. Start with largest histogram as first cluster
/// 2. Repeatedly add the most distant histogram as a new cluster
/// 3. Stop when max clusters reached or distance < threshold
/// 4. Assign remaining histograms to nearest cluster
///
/// Implements libjxl's FastClusterHistograms
fn fast_cluster_histograms(
    input: &[Histogram],
    max_histograms: usize,
    output: &mut Vec<Histogram>,
) -> Result<Vec<u32>, Error> {
    let prev_histograms = output.len();
    output.reserve(max_histograms);

    let mut symbols = vec![max_histograms as u32; input.len()];  // Unassigned marker
    let mut dists = vec![f32::MAX; input.len()];

    // Find largest (by total_count) non-empty histogram
    let mut largest_idx = 0;
    for (i, h) in input.iter().enumerate() {
        if h.total_count == 0 {
            symbols[i] = 0;  // Empty histograms map to cluster 0
            dists[i] = 0.0;
            continue;
        }

        // Compute entropy (needed for distance calculations)
        h.shannon_entropy();

        if h.total_count > input[largest_idx].total_count {
            largest_idx = i;
        }
    }

    // If we have existing histograms, update distances and find furthest
    if prev_histograms > 0 {
        for h in output.iter() {
            h.shannon_entropy();
        }

        for (i, h) in input.iter().enumerate() {
            if dists[i] == 0.0 {
                continue;
            }
            for (j, cluster) in output.iter().enumerate() {
                let d = histogram_kl_divergence(h, cluster);
                dists[i] = dists[i].min(d);
            }
        }

        // Find furthest from existing clusters
        if let Some((idx, &d)) = dists.iter().enumerate().max_by(|a, b| {
            a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)
        }) {
            if d > 0.0 {
                largest_idx = idx;
            }
        }
    }

    // Add clusters until we have enough or distance is too small
    while output.len() < max_histograms {
        symbols[largest_idx] = output.len() as u32;
        output.push(input[largest_idx].clone());
        dists[largest_idx] = 0.0;

        // Find next furthest histogram
        largest_idx = 0;
        for (i, h) in input.iter().enumerate() {
            if dists[i] == 0.0 {
                continue;
            }

            let d = histogram_distance(h, output.last().unwrap());
            dists[i] = dists[i].min(d);

            if dists[i] > dists[largest_idx] {
                largest_idx = i;
            }
        }

        if dists[largest_idx] < MIN_DISTANCE_FOR_DISTINCT {
            break;
        }
    }

    // Assign remaining histograms to nearest cluster
    for (i, h) in input.iter().enumerate() {
        if symbols[i] != max_histograms as u32 {
            continue;  // Already assigned
        }

        let mut best = 0;
        let mut best_dist = f32::MAX;

        for (j, cluster) in output.iter().enumerate() {
            let dist = if j < prev_histograms {
                histogram_kl_divergence(h, cluster)
            } else {
                histogram_distance(h, cluster)
            };

            if dist < best_dist {
                best = j;
                best_dist = dist;
            }
        }

        assert!(best_dist < f32::MAX);

        // Merge into cluster if it's a new cluster
        if best >= prev_histograms {
            output[best].add_histogram(h);
            output[best].shannon_entropy();
        }

        symbols[i] = best as u32;
    }

    Ok(symbols)
}

/// Refine clusters by trying to merge pairs.
/// Uses a priority queue to always merge the best pair.
///
/// Implements the pair-merging refinement in libjxl's ClusterHistograms
fn refine_clusters_by_merging(
    histograms: &mut Vec<Histogram>,
    symbols: &mut [u32],
) -> Result<(), Error> {
    // Compute ANS population cost for each histogram
    for h in histograms.iter_mut() {
        h.entropy.set(h.ans_population_cost()?);
    }

    let mut next_version = 2u32;
    let mut version = vec![1u32; histograms.len()];
    let mut renumbering: Vec<u32> = (0..histograms.len() as u32).collect();

    // Priority queue of pairs to potentially merge
    // Lower cost = higher priority
    let mut pairs: BinaryHeap<HistogramPair> = BinaryHeap::new();

    // Initialize with all pairs
    for i in 0..histograms.len() {
        for j in (i + 1)..histograms.len() {
            let mut combined = histograms[i].clone();
            combined.add_histogram(&histograms[j]);
            let cost = combined.ans_population_cost()?;
            let merge_cost = cost - histograms[i].entropy.get() - histograms[j].entropy.get();

            // Only enqueue if merging saves space
            if merge_cost < 0.0 {
                pairs.push(HistogramPair {
                    cost: merge_cost,
                    first: i as u32,
                    second: j as u32,
                    version: version[i].max(version[j]),
                });
            }
        }
    }

    // Process pairs
    while let Some(pair) = pairs.pop() {
        let first = pair.first as usize;
        let second = pair.second as usize;

        // Check if pair is still valid
        if pair.version != version[first].max(version[second])
            || version[first] == 0
            || version[second] == 0
        {
            continue;
        }

        // Merge second into first
        histograms[first].add_histogram(&histograms[second]);
        histograms[first].entropy.set(histograms[first].ans_population_cost()?);

        // Update renumbering
        for item in renumbering.iter_mut() {
            if *item == second as u32 {
                *item = first as u32;
            }
        }

        // Mark second as dead
        version[second] = 0;
        version[first] = next_version;
        next_version += 1;

        // Add new pairs involving the merged cluster
        for j in 0..histograms.len() {
            if j == first || version[j] == 0 {
                continue;
            }

            let mut combined = histograms[first].clone();
            combined.add_histogram(&histograms[j]);
            let cost = combined.ans_population_cost()?;
            let merge_cost = cost - histograms[first].entropy.get() - histograms[j].entropy.get();

            if merge_cost < 0.0 {
                pairs.push(HistogramPair {
                    cost: merge_cost,
                    first: (first as u32).min(j as u32),
                    second: (first as u32).max(j as u32),
                    version: version[first].max(version[j]),
                });
            }
        }
    }

    // Compact: remove dead clusters
    let mut reverse_renumbering = vec![u32::MAX; histograms.len()];
    let mut num_alive = 0;

    for i in 0..histograms.len() {
        if version[i] == 0 {
            continue;
        }
        histograms[num_alive] = std::mem::take(&mut histograms[i]);
        reverse_renumbering[i] = num_alive as u32;
        num_alive += 1;
    }
    histograms.truncate(num_alive);

    // Update symbols
    for sym in symbols.iter_mut() {
        *sym = reverse_renumbering[renumbering[*sym as usize] as usize];
    }

    Ok(())
}

/// Pair of histograms for the merge priority queue
#[derive(Clone, Copy)]
struct HistogramPair {
    cost: f32,
    first: u32,
    second: u32,
    version: u32,
}

impl PartialEq for HistogramPair {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
            && self.first == other.first
            && self.second == other.second
    }
}

impl Eq for HistogramPair {}

impl PartialOrd for HistogramPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HistogramPair {
    fn cmp(&self, other: &Self) -> Ordering {
        // Note: Reversed because BinaryHeap is max-heap but we want min-cost
        // Use tuple comparison matching libjxl's operator<
        (other.cost, other.first, other.second, other.version)
            .partial_cmp(&(self.cost, self.first, self.second, self.version))
            .unwrap_or(Ordering::Equal)
    }
}

/// Reindex histograms so symbols appear in increasing order
fn histogram_reindex(
    histograms: &mut Vec<Histogram>,
    prev_histograms: usize,
    symbols: &mut [u32],
) {
    let tmp = histograms.clone();
    let mut new_index = std::collections::HashMap::new();

    for i in 0..prev_histograms {
        new_index.insert(i as u32, i as u32);
    }

    let mut next_index = prev_histograms as u32;
    for &symbol in symbols.iter() {
        if !new_index.contains_key(&symbol) {
            new_index.insert(symbol, next_index);
            histograms[next_index as usize] = tmp[symbol as usize].clone();
            next_index += 1;
        }
    }

    histograms.truncate(next_index as usize);

    for symbol in symbols.iter_mut() {
        *symbol = new_index[symbol];
    }
}
```

---

## 6. Module 3: ANS Encoding

```rust
// src/ans_encode.rs

use crate::{
    Histogram, ANSHistogramStrategy, HybridUintConfig,
    ANS_LOG_TAB_SIZE, ANS_TAB_SIZE, ANS_TAB_MASK, ANS_MAX_ALPHABET_SIZE,
    ANS_SIGNATURE, RECIPROCAL_PRECISION,
};

/// Symbol encoding information for one symbol
#[derive(Clone, Debug, Default)]
pub struct ANSEncSymbolInfo {
    /// ANS frequency
    pub freq: u16,
    /// Reciprocal of frequency for fast division
    pub ifreq: u64,
    /// Reverse mapping from state offset to original position
    pub reverse_map: Vec<u16>,

    /// Prefix coding depth (for Huffman mode)
    pub depth: u8,
    /// Prefix coding bits (for Huffman mode)
    pub bits: u16,
}

/// ANS encoder state machine
pub struct ANSCoder {
    state: u32,
}

impl ANSCoder {
    pub fn new() -> Self {
        Self {
            state: (ANS_SIGNATURE as u32) << 16,
        }
    }

    /// Encode a symbol, returns (bits_to_output, num_bits)
    #[inline]
    pub fn put_symbol(&mut self, info: &ANSEncSymbolInfo) -> (u32, u8) {
        let mut bits = 0u32;
        let mut nbits = 0u8;

        // Renormalize: emit bits if state is too large
        if (self.state >> (32 - ANS_LOG_TAB_SIZE)) >= info.freq as u32 {
            bits = self.state & 0xFFFF;
            self.state >>= 16;
            nbits = 16;
        }

        // Encode symbol
        let v = ((self.state as u64 * info.ifreq) >> RECIPROCAL_PRECISION) as u32;
        let offset = info.reverse_map[(self.state - v * info.freq as u32) as usize];
        self.state = (v << ANS_LOG_TAB_SIZE) + offset as u32;

        (bits, nbits)
    }

    pub fn get_state(&self) -> u32 {
        self.state
    }
}

/// Normalized histogram for ANS encoding
pub struct ANSEncodingHistogram {
    counts: Vec<i32>,
    cost: f32,
    method: u32,       // 0=flat, 1=small, 2-13=shift+1
    omit_pos: usize,   // Position of balancing bin
    alphabet_size: usize,
    num_symbols: usize,
    symbols: [usize; 2],  // For small histograms
}

impl ANSEncodingHistogram {
    /// Compute the best normalized histogram for encoding.
    /// Tries different normalization methods and picks lowest cost.
    ///
    /// Implements libjxl's ANSEncodingHistogram::ComputeBest
    pub fn compute_best(
        histo: &Histogram,
        strategy: ANSHistogramStrategy,
    ) -> Result<Self, Error> {
        let mut result = Self {
            counts: Vec::new(),
            cost: 0.0,
            method: 0,
            omit_pos: 0,
            alphabet_size: histo.alphabet_size(),
            num_symbols: 0,
            symbols: [0; 2],
        };

        if result.alphabet_size > ANS_MAX_ALPHABET_SIZE {
            return Err(Error::TooManySymbols(result.alphabet_size));
        }

        if result.alphabet_size > 0 {
            // Try flat histogram first
            result.method = 0;
            result.num_symbols = result.alphabet_size;
            result.counts = create_flat_histogram(result.alphabet_size, ANS_TAB_SIZE as i32);
            result.counts.resize(histo.counts.len(), 0);
            result.cost = result.encode_size() + estimate_data_bits_flat(histo);
        } else {
            // Empty histogram
            result.method = 1;
            result.num_symbols = 0;
            result.cost = 3.0;
            return Ok(result);
        }

        // Count symbols and record first two
        let mut symbol_count = 0;
        for (n, &count) in histo.counts.iter().enumerate() {
            if count > 0 {
                if symbol_count < 2 {
                    result.symbols[symbol_count] = n;
                }
                symbol_count += 1;
            }
        }
        result.num_symbols = symbol_count;

        // Single-symbol histogram: just store the symbol
        if symbol_count == 1 {
            result.method = 1;
            result.counts = histo.counts.clone();
            result.counts[result.symbols[0]] = ANS_TAB_SIZE as i32;
            result.cost = result.encode_size();
            return Ok(result);
        }

        // Try different shift values
        let try_shift = |result: &mut Self, shift: u32| -> Result<(), Error> {
            let method = shift.min(ANS_LOG_TAB_SIZE - 1) + 1;

            let mut normalized = result.clone();
            normalized.method = method;

            if !normalized.rebalance_histogram(histo) {
                return Err(Error::RebalanceFailed);
            }

            let cost = normalized.encode_size() + normalized.estimate_data_bits(histo);
            if cost < result.cost {
                *result = normalized;
                result.cost = cost;
            }

            Ok(())
        };

        match strategy {
            ANSHistogramStrategy::Precise => {
                for shift in 0..ANS_LOG_TAB_SIZE {
                    try_shift(&mut result, shift)?;
                }
            }
            ANSHistogramStrategy::Approximate => {
                for shift in (0..=ANS_LOG_TAB_SIZE).step_by(2) {
                    try_shift(&mut result, shift)?;
                }
            }
            ANSHistogramStrategy::Fast => {
                try_shift(&mut result, 0)?;
                try_shift(&mut result, ANS_LOG_TAB_SIZE / 2)?;
                try_shift(&mut result, ANS_LOG_TAB_SIZE)?;
            }
        }

        Ok(result)
    }

    pub fn cost(&self) -> f32 {
        self.cost
    }

    pub fn counts(&self) -> &[i32] {
        &self.counts
    }

    /// Encode histogram to bitstream
    /// Implements libjxl's ANSEncodingHistogram::Encode
    pub fn encode<W: BitWrite>(&self, writer: &mut W) -> Result<(), Error> {
        // See bitstream format appendix for details
        // ... (detailed implementation matching libjxl)
        todo!("Implement encode")
    }

    /// Build the alias table and encoding info
    pub fn build_info_table(
        &self,
        table: &[AliasTableEntry],
        log_alpha_size: usize,
        info: &mut [ANSEncSymbolInfo],
    ) {
        let log_entry_size = ANS_LOG_TAB_SIZE as usize - log_alpha_size;
        let entry_size_minus_1 = (1 << log_entry_size) - 1;

        // Initialize symbol info
        for (s, info) in info.iter_mut().enumerate().take(self.alphabet_size.max(1)) {
            let freq = if s == self.alphabet_size {
                ANS_TAB_SIZE as i32
            } else {
                self.counts.get(s).copied().unwrap_or(0)
            };

            info.freq = freq as u16;
            if freq != 0 {
                info.ifreq = ((1u64 << RECIPROCAL_PRECISION) + freq as u64 - 1) / freq as u64;
            } else {
                info.ifreq = 1;
            }
            info.reverse_map.resize(freq as usize, 0);
        }

        // Build reverse map from alias table
        for i in 0..ANS_TAB_SIZE as usize {
            let symbol = alias_table_lookup(table, i, log_entry_size, entry_size_minus_1);
            info[symbol.value].reverse_map[symbol.offset] = i as u16;
        }
    }

    /// Estimate bits needed to encode data with this histogram
    fn estimate_data_bits(&self, histo: &Histogram) -> f32 {
        // Uses fixed-point log2 lookup table
        // ... (implementation matching libjxl)
        todo!("Implement estimate_data_bits")
    }

    fn encode_size(&self) -> f32 {
        // Count bits needed to encode the histogram itself
        // ... (implementation matching libjxl)
        todo!("Implement encode_size")
    }

    /// Rebalance histogram counts to sum to ANS_TAB_SIZE
    /// while respecting precision constraints.
    ///
    /// Implements libjxl's ANSEncodingHistogram::RebalanceHistogram
    fn rebalance_histogram(&mut self, histo: &Histogram) -> bool {
        // Complex algorithm - see libjxl for details
        // Key points:
        // 1. Scale counts proportionally to ANS_TAB_SIZE
        // 2. Round to allowed values based on shift
        // 3. Use greedy entropy optimization
        // 4. Balance using one "remainder" bin
        todo!("Implement rebalance_histogram")
    }
}

/// Create flat histogram where counts differ by at most 1
fn create_flat_histogram(length: usize, total_count: i32) -> Vec<i32> {
    let count = total_count / length as i32;
    let rem = (total_count % length as i32) as usize;

    let mut result = vec![count; length];
    for i in 0..rem {
        result[i] += 1;
    }
    result
}

fn estimate_data_bits_flat(histo: &Histogram) -> f32 {
    let len = histo.alphabet_size();
    if len == 0 {
        return 0.0;
    }
    histo.total_count as f32 * (len as f32).log2()
}

/// Get precision bits for a count with given log2
/// Implements libjxl's GetPopulationCountPrecision
pub fn get_population_count_precision(logcount: u32, shift: u32) -> u32 {
    let r = (logcount as i32).min(
        shift as i32 - ((ANS_LOG_TAB_SIZE - logcount) / 2) as i32
    );
    if r < 0 { 0 } else { r as u32 }
}
```

---

## 7. Module 4: Context Map

```rust
// src/context_map.rs

use crate::{BitWrite, Token, HistogramParams, EntropyEncodingData};

/// Encode context map to bitstream.
///
/// The context map tells the decoder which histogram to use for each context.
/// Uses move-to-front transform for better compression.
///
/// Implements libjxl's EncodeContextMap
pub fn encode_context_map<W: BitWrite>(
    context_map: &[u8],
    num_histograms: usize,
    writer: &mut W,
) -> Result<(), Error> {
    if num_histograms == 1 {
        // Simple code: single histogram
        writer.write(1, 1)?;  // Simple marker
        writer.write(2, 0)?;  // 0 bits per entry
        return Ok(());
    }

    // Try both raw and move-to-front encoding
    let transformed = move_to_front_transform(context_map);

    // Create tokens for both versions
    let raw_tokens: Vec<_> = context_map.iter()
        .map(|&v| Token::new(0, v as u32))
        .collect();
    let mtf_tokens: Vec<_> = transformed.iter()
        .map(|&v| Token::new(0, v as u32))
        .collect();

    // Estimate costs
    let params = HistogramParams {
        uint_method: HybridUintMethod::ContextMap,
        ..Default::default()
    };

    let raw_cost = estimate_encoding_cost(&params, &[raw_tokens.clone()])?;
    let mtf_cost = estimate_encoding_cost(&params, &[mtf_tokens.clone()])?;

    let use_mtf = mtf_cost < raw_cost;
    let tokens = if use_mtf { &mtf_tokens } else { &raw_tokens };

    // Try simple fixed-width encoding
    let entry_bits = ceil_log2(num_histograms);
    let simple_cost = entry_bits * context_map.len();

    if entry_bits < 4 && simple_cost < raw_cost.min(mtf_cost) {
        // Use simple fixed-width code
        writer.write(1, 1)?;  // Simple marker
        writer.write(2, entry_bits as u64)?;
        for &entry in context_map {
            writer.write(entry_bits, entry as u64)?;
        }
    } else {
        // Use entropy-coded context map
        writer.write(1, 0)?;  // Not simple
        writer.write(1, use_mtf as u64)?;

        let mut codes = EntropyEncodingData::new();
        build_and_encode_histograms(&params, 1, &[tokens.clone()], &mut codes, writer)?;
        write_tokens(tokens, &codes, 0, writer)?;
    }

    Ok(())
}

/// Move-to-front transform for context map compression
fn move_to_front_transform(v: &[u8]) -> Vec<u8> {
    if v.is_empty() {
        return Vec::new();
    }

    let max_value = *v.iter().max().unwrap();
    let mut mtf: Vec<u8> = (0..=max_value).collect();

    let mut result = Vec::with_capacity(v.len());
    for &value in v {
        let index = mtf.iter().position(|&x| x == value).unwrap();
        result.push(index as u8);

        // Move to front
        let val = mtf.remove(index);
        mtf.insert(0, val);
    }

    result
}

fn ceil_log2(n: usize) -> usize {
    if n <= 1 { 0 } else { (n - 1).ilog2() as usize + 1 }
}
```

---

## 8. Module 5: Bit Writing

```rust
// src/bitwriter.rs

/// Trait for writing bits to output
pub trait BitWrite {
    /// Write `num_bits` bits from `bits` (least significant bits used)
    fn write(&mut self, num_bits: usize, bits: u64) -> Result<(), Error>;

    /// Number of bits written so far
    fn bits_written(&self) -> usize;
}

/// Bit writer that accumulates bits and outputs bytes
pub struct BitWriter {
    buffer: Vec<u8>,
    bit_buffer: u64,
    bits_in_buffer: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Flush any remaining bits (padding with zeros)
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_buffer > 0 {
            // Pad to byte boundary
            let padding = 8 - (self.bits_in_buffer % 8);
            if padding < 8 {
                self.bits_in_buffer += padding;
            }

            while self.bits_in_buffer >= 8 {
                self.buffer.push(self.bit_buffer as u8);
                self.bit_buffer >>= 8;
                self.bits_in_buffer -= 8;
            }
        }
        self.buffer
    }
}

impl BitWrite for BitWriter {
    fn write(&mut self, num_bits: usize, bits: u64) -> Result<(), Error> {
        debug_assert!(num_bits <= 56);
        debug_assert!(num_bits == 64 || bits < (1u64 << num_bits));

        self.bit_buffer |= bits << self.bits_in_buffer;
        self.bits_in_buffer += num_bits;

        // Emit complete bytes
        while self.bits_in_buffer >= 8 {
            self.buffer.push(self.bit_buffer as u8);
            self.bit_buffer >>= 8;
            self.bits_in_buffer -= 8;
        }

        Ok(())
    }

    fn bits_written(&self) -> usize {
        self.buffer.len() * 8 + self.bits_in_buffer
    }
}

/// Size-counting writer (for cost estimation)
pub struct SizeWriter {
    pub size: usize,
}

impl BitWrite for SizeWriter {
    fn write(&mut self, num_bits: usize, _bits: u64) -> Result<(), Error> {
        self.size += num_bits;
        Ok(())
    }

    fn bits_written(&self) -> usize {
        self.size
    }
}

// Variable-length unsigned integer encoding (matches libjxl)

/// Write VarLenUint8: 0 → [0], n → [1, log2(n), n - 2^log2(n)]
pub fn write_var_len_uint8<W: BitWrite>(n: usize, writer: &mut W) -> Result<(), Error> {
    debug_assert!(n <= 255);
    if n == 0 {
        writer.write(1, 0)?;
    } else {
        writer.write(1, 1)?;
        let nbits = floor_log2(n as u32);
        writer.write(3, nbits as u64)?;
        writer.write(nbits as usize, (n - (1 << nbits)) as u64)?;
    }
    Ok(())
}

/// Write VarLenUint16: similar but 4-bit length field
pub fn write_var_len_uint16<W: BitWrite>(n: usize, writer: &mut W) -> Result<(), Error> {
    debug_assert!(n <= 65535);
    if n == 0 {
        writer.write(1, 0)?;
    } else {
        writer.write(1, 1)?;
        let nbits = floor_log2(n as u32);
        writer.write(4, nbits as u64)?;
        writer.write(nbits as usize, (n - (1 << nbits)) as u64)?;
    }
    Ok(())
}

fn floor_log2(n: u32) -> u32 {
    debug_assert!(n > 0);
    31 - n.leading_zeros()
}
```

---

## 9. Module 6: Orchestration

```rust
// src/lib.rs

use crate::{
    Token, Histogram, HistogramParams, HybridUintConfig,
    ClusterResult, ANSEncSymbolInfo, BitWriter,
};

/// Complete entropy encoding data
pub struct EntropyEncodingData {
    /// Encoding info for each clustered histogram, for each symbol
    pub encoding_info: Vec<Vec<ANSEncSymbolInfo>>,

    /// True if using Huffman prefix codes instead of ANS
    pub use_prefix_code: bool,

    /// Hybrid uint config per histogram
    pub uint_config: Vec<HybridUintConfig>,

    /// Log2 of alphabet size
    pub log_alpha_size: usize,

    /// LZ77 parameters
    pub lz77: LZ77Params,

    /// Context map: context → histogram index
    pub context_map: Vec<u8>,
}

/// Main entry point: build histograms, cluster, encode everything.
///
/// Implements libjxl's BuildAndEncodeHistograms
pub fn build_and_encode_histograms(
    params: &HistogramParams,
    num_contexts: usize,
    tokens: &[Vec<Token>],
    codes: &mut EntropyEncodingData,
    writer: Option<&mut BitWriter>,
) -> Result<usize, Error> {
    // Phase 1: Apply LZ77 if configured
    let (tokens, lz77_enabled) = apply_lz77(params, num_contexts, tokens, &mut codes.lz77)?;
    let num_contexts = if lz77_enabled { num_contexts + 1 } else { num_contexts };

    // Write LZ77 params
    let mut cost = 0;
    if let Some(w) = writer.as_deref_mut() {
        cost += write_lz77_params(&codes.lz77, w)?;
    }

    // Phase 2: Build initial histograms from tokens
    let mut builder = build_histograms_from_tokens(
        &tokens,
        num_contexts,
        &params.uint_config(),
        &codes.lz77,
    );

    // Phase 3: Cluster histograms
    let mut clustered = Vec::new();
    let symbols = cluster_histograms(
        params,
        &builder,
        CLUSTERS_LIMIT,
        &mut clustered,
    )?;

    // Build context map
    codes.context_map.resize(num_contexts, 0);
    for (c, &sym) in symbols.iter().enumerate() {
        codes.context_map[c] = sym as u8;
    }

    // Decide: prefix code or ANS?
    let total_tokens: usize = tokens.iter().map(|t| t.len()).sum();
    codes.use_prefix_code = params.force_huffman
        || total_tokens < 100
        || params.clustering == ClusteringType::Fastest;

    // Phase 4: Choose optimal uint configs
    choose_uint_configs(params, &tokens, &mut clustered, codes)?;

    // Phase 5: Encode context map
    if let Some(w) = writer.as_deref_mut() {
        if clustered.len() > 1 {
            encode_context_map(&codes.context_map, clustered.len(), w)?;
        }
    }

    // Phase 6: Normalize and encode histograms
    cost += encode_histograms(params, &clustered, codes, writer)?;

    Ok(cost)
}

/// Write all tokens to bitstream using the encoding data.
///
/// Implements libjxl's WriteTokens
pub fn write_tokens<W: BitWrite>(
    tokens: &[Token],
    codes: &EntropyEncodingData,
    context_offset: usize,
    writer: &mut W,
) -> Result<usize, Error> {
    let mut num_extra_bits = 0;

    if codes.use_prefix_code {
        // Huffman/prefix coding path
        for token in tokens {
            let histo = codes.context_map[context_offset + token.context as usize] as usize;
            let config = if token.is_lz77_length {
                &codes.lz77.length_uint_config
            } else {
                &codes.uint_config[histo]
            };

            let (tok, nbits, bits) = config.encode(token.value);
            let tok = if token.is_lz77_length {
                tok + codes.lz77.min_symbol
            } else {
                tok
            };

            let info = &codes.encoding_info[histo][tok as usize];

            // Write Huffman code + extra bits
            let combined = info.bits as u64 | ((bits as u64) << info.depth);
            writer.write(info.depth as usize + nbits as usize, combined)?;
            num_extra_bits += nbits as usize;
        }
    } else {
        // ANS coding path (writes in reverse order)
        let mut out: Vec<(u64, u8)> = Vec::with_capacity(tokens.len());
        let mut ans = ANSCoder::new();

        for token in tokens.iter().rev() {
            let histo = codes.context_map[context_offset + token.context as usize] as usize;
            let config = if token.is_lz77_length {
                &codes.lz77.length_uint_config
            } else {
                &codes.uint_config[histo]
            };

            let (tok, nbits, bits) = config.encode(token.value);
            let tok = if token.is_lz77_length {
                tok + codes.lz77.min_symbol
            } else {
                tok
            };

            let info = &codes.encoding_info[histo][tok as usize];

            // Extra bits first (reversed order)
            if nbits > 0 {
                out.push((bits as u64, nbits as u8));
                num_extra_bits += nbits as usize;
            }

            // ANS encode
            let (ans_bits, ans_nbits) = ans.put_symbol(info);
            if ans_nbits > 0 {
                out.push((ans_bits as u64, ans_nbits));
            }
        }

        // Write state, then bits in reverse order
        writer.write(32, ans.get_state() as u64)?;
        for (bits, nbits) in out.into_iter().rev() {
            writer.write(nbits as usize, bits)?;
        }
    }

    Ok(num_extra_bits)
}

// Helper functions

fn build_histograms_from_tokens(
    tokens: &[Vec<Token>],
    num_contexts: usize,
    uint_config: &HybridUintConfig,
    lz77: &LZ77Params,
) -> Vec<Histogram> {
    let mut builder: Vec<Histogram> = (0..num_contexts)
        .map(|_| Histogram::new())
        .collect();

    for stream in tokens {
        for token in stream {
            let config = if token.is_lz77_length {
                &lz77.length_uint_config
            } else {
                uint_config
            };

            let (tok, _, _) = config.encode(token.value);
            let tok = if token.is_lz77_length {
                tok + lz77.min_symbol
            } else {
                tok
            };

            builder[token.context as usize].add(tok as usize);
        }
    }

    builder
}
```

---

## 10. Implementation Order

Recommended order with dependencies:

```
Phase 1: Foundation (no dependencies)
├── constants.rs
├── token.rs
├── hybrid_uint.rs
└── bitwriter.rs

Phase 2: Histogram Core
├── histogram.rs (depends on constants)
│   ├── Histogram struct
│   ├── shannon_entropy()
│   └── histogram_distance()
└── Tests: exact match vs libjxl output

Phase 3: Clustering
├── cluster.rs (depends on histogram)
│   ├── fast_cluster_histograms()
│   ├── refine_clusters_by_merging()
│   └── histogram_reindex()
└── Tests: exact cluster assignments vs libjxl

Phase 4: ANS Normalization
├── ans_normalize.rs (depends on histogram, constants)
│   ├── ANSEncodingHistogram
│   ├── rebalance_histogram()
│   └── get_population_count_precision()
└── Tests: exact normalized counts vs libjxl

Phase 5: ANS Encoding
├── ans_encode.rs (depends on ans_normalize)
│   ├── ANSCoder
│   ├── ANSEncSymbolInfo
│   └── build_info_table()
└── alias_table.rs
└── Tests: exact bitstream vs libjxl

Phase 6: Context Map
├── context_map.rs (depends on ans_encode, bitwriter)
│   ├── encode_context_map()
│   └── move_to_front_transform()
└── Tests: exact context map encoding vs libjxl

Phase 7: Integration
├── lib.rs (orchestration)
│   ├── build_and_encode_histograms()
│   └── write_tokens()
└── Full pipeline tests
```

---

## 11. Test Strategy

### 11.1 Unit Tests (Per Module)

```rust
// tests/parity_histogram.rs

#[test]
fn test_shannon_entropy_matches_libjxl() {
    // Test cases extracted from libjxl
    let cases = vec![
        (vec![100, 50, 25, 10, 5], expected_entropy),
        // ...
    ];

    for (counts, expected) in cases {
        let h = Histogram { counts, ..Default::default() };
        h.condition();
        let entropy = h.shannon_entropy();
        assert!((entropy - expected).abs() < 1e-5,
            "Entropy mismatch: got {}, expected {}", entropy, expected);
    }
}

#[test]
fn test_histogram_distance_matches_libjxl() {
    // Test with known histogram pairs
}
```

### 11.2 Integration Tests

```rust
// tests/integration.rs

/// End-to-end test: given same tokens, produce identical bitstream
#[test]
fn test_full_pipeline_parity() {
    let tokens = load_test_tokens("testdata/tokens.bin");
    let expected_output = load_expected("testdata/expected_output.bin");

    let params = HistogramParams::default();
    let mut codes = EntropyEncodingData::new();
    let mut writer = BitWriter::new();

    build_and_encode_histograms(&params, num_contexts, &tokens, &mut codes, Some(&mut writer)).unwrap();
    write_tokens(&tokens[0], &codes, 0, &mut writer).unwrap();

    let output = writer.finish();
    assert_eq!(output, expected_output, "Bitstream mismatch");
}
```

### 11.3 Parity Test Data Generation

Create a C++ test harness that dumps intermediate values:

```cpp
// Reference output generator
void DumpTestCase(const std::vector<Token>& tokens) {
    // Dump: tokens, histogram counts, cluster assignments,
    //       normalized histograms, context map, final bitstream
}
```

### 11.4 Property-Based Tests

```rust
#[test]
fn test_roundtrip_any_tokens() {
    // Generate random tokens
    // Encode with Rust implementation
    // Decode with libjxl decoder
    // Verify values match
}
```

---

## 12. Appendix: Bitstream Format

### Histogram Encoding Format

```
HISTOGRAM:
├── [1 bit] is_small_tree
│   ├── if 1: Small tree (0-2 symbols)
│   │   ├── [1 bit] num_symbols - 1 (if 0, single symbol)
│   │   ├── [VarLenUint8] symbol_0
│   │   ├── [VarLenUint8] symbol_1 (if 2 symbols)
│   │   └── [12 bits] count_0 (if 2 symbols)
│   │
│   └── if 0: Not small tree
│       ├── [1 bit] is_uniform
│       │   ├── if 1: Flat histogram
│       │   │   └── [VarLenUint8] alphabet_size - 1
│       │   │
│       │   └── if 0: General histogram
│       │       ├── [Elias gamma] shift
│       │       ├── [VarLenUint8] alphabet_size - 3
│       │       ├── [Huffman] bit_widths[] (with RLE)
│       │       └── [variable] extra precision bits
```

### Context Map Encoding Format

```
CONTEXT_MAP:
├── [1 bit] is_simple
│   ├── if 1: Simple encoding
│   │   ├── [2 bits] bits_per_entry (0-3)
│   │   └── [bits_per_entry × num_contexts] entries
│   │
│   └── if 0: Entropy-coded
│       ├── [1 bit] use_mtf
│       ├── [HISTOGRAM] histogram for map entries
│       └── [ANS/Huffman] encoded entries
```

### ANS Stream Format

```
ANS_STREAM (for ANS mode):
├── [32 bits] final_state
└── [variable] interleaved (symbol_codes, extra_bits) in REVERSE order
```

---

## Critical Implementation Notes

1. **Floating-point determinism**: Use identical formulas to libjxl. The entropy calculations must match exactly.

2. **Priority queue ordering**: The `HistogramPair` comparison must match libjxl's tuple comparison exactly.

3. **Histogram rebalancing**: This is the most complex algorithm. Port line-by-line from libjxl.

4. **VarLenUint encoding**: Must match exactly - off-by-one errors will corrupt the stream.

5. **ANS state**: Initialize to `ANS_SIGNATURE << 16`, write final state as first 32 bits.

6. **Reverse order**: ANS encodes in reverse! Build output buffer backwards, then reverse.

7. **Alias table**: Must match libjxl's `InitAliasTable` exactly for decoder compatibility.

---

## Reference: libjxl Source Files

| Component | libjxl File |
|-----------|-------------|
| Histogram | `lib/jxl/enc_ans_params.h` |
| Clustering | `lib/jxl/enc_cluster.cc` |
| ANS Encoding | `lib/jxl/enc_ans.cc` |
| Context Map | `lib/jxl/enc_context_map.cc` |
| Alias Table | `lib/jxl/ans_common.h` |
| Bit Writing | `lib/jxl/enc_bit_writer.h` |
| Constants | `lib/jxl/ans_params.h` |
