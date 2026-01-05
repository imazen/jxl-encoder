# Histogram Clustering Implementation Plan

**Purpose**: Implement ANS entropy coding with histogram clustering as an independently testable module, verified against libjxl reference outputs.

**Status**: Planning
**Created**: 2026-01-04
**Target**: Replace current Huffman-only fallback with full ANS pipeline matching libjxl

---

## Table of Contents

1. [Overview](#1-overview)
2. [Reference Test Harness (C++)](#2-reference-test-harness-c)
3. [Test Data Format](#3-test-data-format)
4. [Rust Module Structure](#4-rust-module-structure)
5. [Implementation Order](#5-implementation-order)
6. [Algorithm Details](#6-algorithm-details)
7. [Parity Testing Strategy](#7-parity-testing-strategy)
8. [File Locations](#8-file-locations)
9. [Constants Reference](#9-constants-reference)
10. [Checklist](#10-checklist)

---

## 1. Overview

### Current State

The encoder has basic ANS infrastructure but lacks:
- Histogram clustering (currently uses 1:1 context-to-histogram mapping)
- Proper ANS histogram rebalancing (uses simple normalization)
- Context map encoding with move-to-front
- HybridUint config optimization

### Goal

Implement the full entropy coding pipeline from libjxl:

```
tokens → build_histograms → cluster_histograms → normalize_ans →
encode_context_map → encode_histograms → write_tokens → bitstream
```

### Key Principle

**Every function must produce byte-identical output to libjxl for the same inputs.**

This is achieved by:
1. Extracting reference outputs from libjxl using a C++ test harness
2. Storing these as versioned test fixtures
3. Running Rust implementation against same inputs, comparing outputs

---

## 2. Reference Test Harness (C++)

### Location

```
~/work/jxl-efforts/libjxl/tools/histogram_test_harness.cc
```

### Purpose

Extract intermediate values from libjxl's entropy coding pipeline for parity testing.

### Build Integration

Add to `~/work/jxl-efforts/libjxl/tools/CMakeLists.txt`:

```cmake
add_executable(histogram_test_harness histogram_test_harness.cc)
target_link_libraries(histogram_test_harness jxl_enc-internal jxl-internal)
```

### Harness Code

```cpp
// histogram_test_harness.cc
// Extracts reference data from libjxl histogram/clustering internals

#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>

#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_cluster.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/enc_ans_params.h"

namespace jxl {

// Writes binary data with simple format:
// [4 bytes: count][data...]
void WriteVec(std::ofstream& f, const std::vector<int32_t>& v) {
    uint32_t count = v.size();
    f.write(reinterpret_cast<const char*>(&count), 4);
    f.write(reinterpret_cast<const char*>(v.data()), count * 4);
}

void WriteVec(std::ofstream& f, const std::vector<uint32_t>& v) {
    uint32_t count = v.size();
    f.write(reinterpret_cast<const char*>(&count), 4);
    f.write(reinterpret_cast<const char*>(v.data()), count * 4);
}

void WriteVecF32(std::ofstream& f, const std::vector<float>& v) {
    uint32_t count = v.size();
    f.write(reinterpret_cast<const char*>(&count), 4);
    f.write(reinterpret_cast<const char*>(v.data()), count * 4);
}

void WriteHistogram(std::ofstream& f, const Histogram& h) {
    // Write counts
    std::vector<int32_t> counts(h.data_.size());
    for (size_t i = 0; i < h.data_.size(); i++) {
        counts[i] = h.data_[i];
    }
    WriteVec(f, counts);

    // Write total_count
    uint64_t total = h.total_count_;
    f.write(reinterpret_cast<const char*>(&total), 8);
}

// Test case 1: Shannon entropy calculation
void DumpEntropyTest(const char* output_dir) {
    std::vector<std::vector<int32_t>> test_cases = {
        {100, 50, 25, 10, 5},           // Skewed
        {100, 100, 100, 100},           // Uniform
        {1000, 1},                       // Highly skewed
        {1, 1, 1, 1, 1, 1, 1, 1},       // Flat
        {255, 128, 64, 32, 16, 8, 4, 2, 1}, // Power law
    };

    std::ofstream f(std::string(output_dir) + "/entropy_test.bin",
                    std::ios::binary);

    uint32_t num_cases = test_cases.size();
    f.write(reinterpret_cast<const char*>(&num_cases), 4);

    for (const auto& counts : test_cases) {
        Histogram h;
        for (size_t i = 0; i < counts.size(); i++) {
            h.data_.resize(std::max(h.data_.size(), i + 1));
            h.data_[i] = counts[i];
            h.total_count_ += counts[i];
        }

        float entropy = h.ShannonEntropy();

        WriteVec(f, counts);
        f.write(reinterpret_cast<const char*>(&entropy), 4);
    }
}

// Test case 2: Histogram distance calculation
void DumpDistanceTest(const char* output_dir) {
    std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>> test_cases = {
        {{100, 50, 25}, {100, 50, 25}},           // Identical
        {{100, 0, 0}, {0, 0, 100}},               // Disjoint
        {{100, 100}, {50, 150}},                  // Similar
        {{1, 1, 1, 1}, {4, 0, 0, 0}},             // Very different
    };

    std::ofstream f(std::string(output_dir) + "/distance_test.bin",
                    std::ios::binary);

    uint32_t num_cases = test_cases.size();
    f.write(reinterpret_cast<const char*>(&num_cases), 4);

    for (const auto& [a_counts, b_counts] : test_cases) {
        Histogram a, b;
        for (size_t i = 0; i < a_counts.size(); i++) {
            a.data_.resize(std::max(a.data_.size(), i + 1));
            a.data_[i] = a_counts[i];
            a.total_count_ += a_counts[i];
        }
        for (size_t i = 0; i < b_counts.size(); i++) {
            b.data_.resize(std::max(b.data_.size(), i + 1));
            b.data_[i] = b_counts[i];
            b.total_count_ += b_counts[i];
        }

        // Compute entropies first (required for distance)
        a.ShannonEntropy();
        b.ShannonEntropy();

        float distance = HistogramDistance(a, b);

        WriteVec(f, a_counts);
        WriteVec(f, b_counts);
        f.write(reinterpret_cast<const char*>(&distance), 4);
    }
}

// Test case 3: Histogram clustering
void DumpClusteringTest(const char* output_dir) {
    // Create test histograms that should cluster in predictable ways
    std::vector<Histogram> histograms;

    // Group 1: Similar distributions (should cluster together)
    for (int i = 0; i < 3; i++) {
        Histogram h;
        h.data_ = {100 + i*5, 50, 25, 10};
        h.total_count_ = 185 + i*5;
        histograms.push_back(h);
    }

    // Group 2: Different distribution
    for (int i = 0; i < 3; i++) {
        Histogram h;
        h.data_ = {10, 25, 50, 100 + i*5};
        h.total_count_ = 185 + i*5;
        histograms.push_back(h);
    }

    // Compute entropies
    for (auto& h : histograms) {
        h.ShannonEntropy();
    }

    std::vector<Histogram> clustered;
    std::vector<uint32_t> symbols = FastClusterHistograms(
        histograms, 4, &clustered);

    std::ofstream f(std::string(output_dir) + "/clustering_test.bin",
                    std::ios::binary);

    // Write input histograms
    uint32_t num_input = histograms.size();
    f.write(reinterpret_cast<const char*>(&num_input), 4);
    for (const auto& h : histograms) {
        WriteHistogram(f, h);
    }

    // Write max_histograms parameter
    uint32_t max_histograms = 4;
    f.write(reinterpret_cast<const char*>(&max_histograms), 4);

    // Write output: symbols (cluster assignments)
    WriteVec(f, symbols);

    // Write output: clustered histograms
    uint32_t num_output = clustered.size();
    f.write(reinterpret_cast<const char*>(&num_output), 4);
    for (const auto& h : clustered) {
        WriteHistogram(f, h);
    }
}

// Test case 4: ANS histogram normalization/rebalancing
void DumpRebalanceTest(const char* output_dir) {
    std::vector<std::vector<int32_t>> test_cases = {
        {100, 50, 25, 10, 5},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // 10 symbols, flat
        {1000, 500, 250, 125, 62, 31, 15, 8, 4, 2, 1, 1, 1},
        {1, 2, 4, 8, 16, 32, 64, 128, 256},
    };

    std::ofstream f(std::string(output_dir) + "/rebalance_test.bin",
                    std::ios::binary);

    uint32_t num_cases = test_cases.size();
    f.write(reinterpret_cast<const char*>(&num_cases), 4);

    for (const auto& counts : test_cases) {
        Histogram h;
        for (size_t i = 0; i < counts.size(); i++) {
            h.data_.resize(std::max(h.data_.size(), i + 1));
            h.data_[i] = counts[i];
            h.total_count_ += counts[i];
        }

        // Try different strategies
        for (int strategy = 0; strategy < 3; strategy++) {
            ANSHistogramStrategy strat = static_cast<ANSHistogramStrategy>(strategy);

            EntropyEncodingData codes;
            // This populates codes with normalized histogram
            // ... (need to check exact API)
        }

        WriteVec(f, counts);
        // Write normalized counts
        // ...
    }
}

// Test case 5: Full encode pipeline (tokens → bitstream)
void DumpFullPipelineTest(const char* output_dir) {
    // Create a set of tokens that exercises the full pipeline
    std::vector<Token> tokens;

    // Generate predictable token sequence
    for (int ctx = 0; ctx < 8; ctx++) {
        for (int i = 0; i < 100; i++) {
            Token t;
            t.context = ctx;
            t.value = (ctx * 17 + i * 7) % 64;  // Deterministic pattern
            tokens.push_back(t);
        }
    }

    std::ofstream f(std::string(output_dir) + "/full_pipeline_test.bin",
                    std::ios::binary);

    // Write tokens
    uint32_t num_tokens = tokens.size();
    f.write(reinterpret_cast<const char*>(&num_tokens), 4);
    for (const auto& t : tokens) {
        f.write(reinterpret_cast<const char*>(&t.context), 4);
        f.write(reinterpret_cast<const char*>(&t.value), 4);
    }

    // Write num_contexts
    uint32_t num_contexts = 8;
    f.write(reinterpret_cast<const char*>(&num_contexts), 4);

    // Encode using libjxl
    HistogramParams params;  // Default params
    EntropyEncodingData codes;
    std::vector<uint8_t> output;

    // ... call BuildAndEncodeHistograms + WriteTokens

    // Write context_map
    // Write final bitstream
}

}  // namespace jxl

int main(int argc, char** argv) {
    const char* output_dir = argc > 1 ? argv[1] : "/tmp/histogram_test_data";

    printf("Generating histogram test data to %s\n", output_dir);

    jxl::DumpEntropyTest(output_dir);
    printf("  - entropy_test.bin\n");

    jxl::DumpDistanceTest(output_dir);
    printf("  - distance_test.bin\n");

    jxl::DumpClusteringTest(output_dir);
    printf("  - clustering_test.bin\n");

    jxl::DumpRebalanceTest(output_dir);
    printf("  - rebalance_test.bin\n");

    jxl::DumpFullPipelineTest(output_dir);
    printf("  - full_pipeline_test.bin\n");

    printf("Done!\n");
    return 0;
}
```

### Notes on C++ Harness

The above is a **template** - the exact libjxl internal APIs need verification:

1. Check `lib/jxl/enc_ans.h` for `Histogram` struct layout
2. Check `lib/jxl/enc_cluster.h` for `FastClusterHistograms` signature
3. Check `lib/jxl/ans_params.h` for constants
4. May need to use `ThreadPool` for some functions

Key files to examine in libjxl:
- `lib/jxl/enc_ans.cc` - `BuildAndEncodeHistograms`, `WriteTokens`
- `lib/jxl/enc_cluster.cc` - `FastClusterHistograms`, cluster refinement
- `lib/jxl/enc_ans_params.h` - `Histogram` class
- `lib/jxl/enc_context_map.cc` - Context map encoding

---

## 3. Test Data Format

### Directory Structure

```
jxl_enc/src/entropy_coding/testdata/
├── v1/
│   ├── entropy_test.bin
│   ├── distance_test.bin
│   ├── clustering_test.bin
│   ├── rebalance_test.bin
│   └── full_pipeline_test.bin
└── README.md
```

### Binary Format (all files)

All multi-byte values are **little-endian**.

#### entropy_test.bin
```
[4 bytes] num_cases: u32
For each case:
    [4 bytes] num_counts: u32
    [num_counts * 4 bytes] counts: [i32]
    [4 bytes] expected_entropy: f32
```

#### distance_test.bin
```
[4 bytes] num_cases: u32
For each case:
    [4 bytes] num_counts_a: u32
    [num_counts_a * 4 bytes] counts_a: [i32]
    [4 bytes] num_counts_b: u32
    [num_counts_b * 4 bytes] counts_b: [i32]
    [4 bytes] expected_distance: f32
```

#### clustering_test.bin
```
[4 bytes] num_input_histograms: u32
For each input histogram:
    [4 bytes] num_counts: u32
    [num_counts * 4 bytes] counts: [i32]
    [8 bytes] total_count: u64
[4 bytes] max_histograms: u32
[4 bytes] num_symbols: u32
[num_symbols * 4 bytes] symbols: [u32]  // cluster assignments
[4 bytes] num_output_histograms: u32
For each output histogram:
    [4 bytes] num_counts: u32
    [num_counts * 4 bytes] counts: [i32]
    [8 bytes] total_count: u64
```

#### rebalance_test.bin
```
[4 bytes] num_cases: u32
For each case:
    [4 bytes] num_counts: u32
    [num_counts * 4 bytes] input_counts: [i32]
    [4 bytes] shift: u32
    [4 bytes] num_normalized: u32
    [num_normalized * 4 bytes] normalized_counts: [i32]
    [4 bytes] omit_pos: u32
    [4 bytes] cost: f32
```

#### full_pipeline_test.bin
```
[4 bytes] num_tokens: u32
For each token:
    [4 bytes] context: u32
    [4 bytes] value: u32
[4 bytes] num_contexts: u32
[4 bytes] context_map_len: u32
[context_map_len bytes] context_map: [u8]
[4 bytes] bitstream_len: u32
[bitstream_len bytes] bitstream: [u8]
```

### Versioning

- Test data is versioned (v1, v2, etc.)
- New versions created if libjxl changes internal algorithms
- Old versions kept for regression testing
- Version tracked in `testdata/VERSION` file

---

## 4. Rust Module Structure

### New Files to Create

```
jxl_enc/src/entropy_coding/
├── mod.rs              (update exports)
├── histogram.rs        (NEW - Histogram struct, entropy, distance)
├── cluster.rs          (NEW - clustering algorithms)
├── context_map.rs      (NEW - context map encoding)
├── ans.rs              (UPDATE - add rebalancing)
└── testdata/           (NEW - reference test data)
```

### Module: histogram.rs

```rust
//! Histogram data structure with entropy calculations.

use std::cell::Cell;

/// Alignment for SIMD-friendly histogram operations.
pub const HISTOGRAM_ROUNDING: usize = 8;

/// A histogram counting symbol occurrences.
#[derive(Clone, Debug)]
pub struct Histogram {
    /// Symbol counts (aligned to HISTOGRAM_ROUNDING).
    pub counts: Vec<i32>,
    /// Sum of all counts.
    pub total_count: usize,
    /// Cached entropy (may be stale).
    entropy_cache: Cell<f32>,
}

impl Histogram {
    pub fn new() -> Self;
    pub fn with_capacity(length: usize) -> Self;
    pub fn from_counts(counts: &[i32]) -> Self;

    /// Add one occurrence of a symbol.
    pub fn add(&mut self, symbol: usize);

    /// Add another histogram's counts.
    pub fn add_histogram(&mut self, other: &Histogram);

    /// Trim trailing zeros and update total_count.
    pub fn condition(&mut self);

    /// Compute Shannon entropy: sum of -p*log2(p).
    /// Result is in bits (not nats).
    pub fn shannon_entropy(&self) -> f32;

    /// Alphabet size (highest non-zero symbol + 1).
    pub fn alphabet_size(&self) -> usize;
}

/// Distance between two histograms (for clustering).
/// Lower = more similar.
pub fn histogram_distance(a: &Histogram, b: &Histogram) -> f32;

/// KL divergence: cost of encoding `actual` using `coding` histogram.
pub fn histogram_kl_divergence(actual: &Histogram, coding: &Histogram) -> f32;
```

### Module: cluster.rs

```rust
//! Histogram clustering for entropy coding.

use super::histogram::{Histogram, histogram_distance, histogram_kl_divergence};

/// Minimum distance threshold for creating distinct clusters.
pub const MIN_DISTANCE_FOR_DISTINCT: f32 = 48.0;

/// Maximum number of histogram clusters.
pub const CLUSTERS_LIMIT: usize = 256;

/// Result of clustering histograms.
pub struct ClusterResult {
    /// The clustered histograms.
    pub histograms: Vec<Histogram>,
    /// Mapping from input index to cluster index.
    pub symbols: Vec<u32>,
}

/// Clustering aggressiveness level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusteringType {
    Fastest,  // Max 4 clusters
    Fast,     // Default clustering
    Best,     // With pair merge refinement
}

/// Fast k-means-like clustering.
///
/// Algorithm:
/// 1. Start with largest histogram as first cluster
/// 2. Repeatedly add most distant histogram as new cluster
/// 3. Stop when max clusters reached or distance < threshold
/// 4. Assign remaining histograms to nearest cluster
pub fn fast_cluster_histograms(
    input: &[Histogram],
    max_histograms: usize,
) -> ClusterResult;

/// Refine clusters by merging pairs that reduce total cost.
pub fn refine_clusters_by_merging(
    histograms: &mut Vec<Histogram>,
    symbols: &mut [u32],
) -> Result<(), Error>;

/// Full clustering pipeline.
pub fn cluster_histograms(
    clustering_type: ClusteringType,
    input: &[Histogram],
    max_histograms: usize,
) -> ClusterResult;
```

### Module: context_map.rs

```rust
//! Context map encoding.

use crate::bit_writer::BitWriter;

/// Encode context map to bitstream.
///
/// Tries both raw and move-to-front encoding, picks smaller.
pub fn encode_context_map(
    context_map: &[u8],
    num_histograms: usize,
    writer: &mut BitWriter,
) -> Result<(), Error>;

/// Move-to-front transform for better compression.
fn move_to_front_transform(v: &[u8]) -> Vec<u8>;

/// Inverse move-to-front transform (for testing).
fn inverse_move_to_front_transform(v: &[u8]) -> Vec<u8>;
```

### Updates to ans.rs

Add the rebalancing algorithm:

```rust
/// ANS histogram normalization parameters.
pub struct ANSEncodingHistogram {
    pub counts: Vec<i32>,
    pub cost: f32,
    pub method: u32,      // 0=flat, 1=small, 2-13=shift+1
    pub omit_pos: usize,  // Position of balancing bin
}

impl ANSEncodingHistogram {
    /// Compute best normalized histogram.
    /// Tries different shift values and picks lowest cost.
    pub fn compute_best(
        histo: &Histogram,
        strategy: ANSHistogramStrategy,
    ) -> Result<Self, Error>;

    /// Rebalance counts to sum to ANS_TAB_SIZE.
    fn rebalance_histogram(&mut self, histo: &Histogram, shift: u32) -> bool;
}

/// Get precision bits for a count.
pub fn get_population_count_precision(logcount: u32, shift: u32) -> u32;
```

---

## 5. Implementation Order

### Phase 1: Foundation (Day 1)

1. **Create `histogram.rs`**
   - `Histogram` struct
   - `shannon_entropy()` - MUST match libjxl exactly
   - `histogram_distance()` - MUST match libjxl exactly
   - `histogram_kl_divergence()`

2. **Write C++ harness** (entropy + distance tests only)
   - Build against libjxl
   - Generate `entropy_test.bin` and `distance_test.bin`

3. **Parity tests**
   - Load test data
   - Compare Rust output to reference
   - Tolerance: entropy within 1e-5, distance within 1e-4

### Phase 2: Clustering (Day 2)

1. **Create `cluster.rs`**
   - `fast_cluster_histograms()` - port from libjxl
   - Priority: get cluster assignments matching exactly

2. **Extend C++ harness** (clustering tests)
   - Generate `clustering_test.bin`

3. **Parity tests**
   - Exact match on cluster assignments (symbols)
   - Exact match on merged histogram counts

### Phase 3: ANS Rebalancing (Day 2-3)

1. **Update `ans.rs`**
   - `ANSEncodingHistogram::rebalance_histogram()` - complex algorithm
   - `get_population_count_precision()`
   - `compute_best()` - try multiple shift values

2. **Extend C++ harness** (rebalance tests)
   - Generate `rebalance_test.bin`

3. **Parity tests**
   - Exact match on normalized counts
   - Exact match on omit_pos
   - Cost within 1e-3

### Phase 4: Context Map (Day 3)

1. **Create `context_map.rs`**
   - `move_to_front_transform()`
   - `encode_context_map()`

2. **Unit tests**
   - MTF round-trip test
   - Encoding matches expected bits

### Phase 5: Pair Merge Refinement (Day 3-4)

1. **Add to `cluster.rs`**
   - `refine_clusters_by_merging()`
   - Priority queue implementation

2. **Parity tests**
   - After refinement, cluster assignments match libjxl

### Phase 6: Full Pipeline Integration (Day 4)

1. **Create orchestration function**
   - `build_and_encode_histograms()`
   - `write_tokens()`

2. **Full pipeline test**
   - Same tokens → same bitstream

3. **Integration with VarDCT**
   - Replace current Huffman-only path
   - Verify all 357 tests still pass

---

## 6. Algorithm Details

### Shannon Entropy (CRITICAL - must match exactly)

```rust
pub fn shannon_entropy(&self) -> f32 {
    if self.total_count == 0 {
        return 0.0;
    }

    let inv_total = 1.0 / self.total_count as f32;
    let mut entropy = 0.0f32;

    for &count in &self.counts {
        if count > 0 && (count as usize) != self.total_count {
            let p = count as f32 * inv_total;
            entropy -= count as f32 * p.log2();
        }
    }

    self.entropy_cache.set(entropy);
    entropy
}
```

**Note**: libjxl may use a lookup table for log2. Check `lib/jxl/fast_math-inl.h`.

### Histogram Distance

```rust
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

    // Subtract individual entropies (must be pre-computed!)
    distance - a.entropy_cache.get() - b.entropy_cache.get()
}
```

### Fast Clustering Algorithm

```
1. Mark all histograms as unassigned
2. Set dists[i] = MAX for all i
3. Find largest (by total_count) non-empty histogram → first cluster
4. Loop:
   a. For unassigned histograms, update dists[i] = min(dists[i], distance to new cluster)
   b. Find histogram with max dists[i]
   c. If dists[max_i] < MIN_DISTANCE_FOR_DISTINCT, break
   d. If num_clusters >= max_histograms, break
   e. Add histogram max_i as new cluster
5. Assign remaining histograms to nearest cluster (merge counts)
6. Return cluster assignments
```

### ANS Rebalancing (Complex)

Key insight: normalized counts must:
- Sum to exactly ANS_TAB_SIZE (4096)
- Non-zero original → non-zero normalized (at least 1)
- Respect precision constraints based on shift parameter

The algorithm uses greedy entropy optimization with a "remainder" bin.

See `lib/jxl/enc_ans.cc` function `RebalanceHistogram`.

---

## 7. Parity Testing Strategy

### Test Levels

1. **Unit tests** - Individual functions with hardcoded expected values
2. **Reference tests** - Compare against binary test data from libjxl
3. **Integration tests** - Full pipeline produces decodable bitstream

### Reference Test Framework

```rust
// jxl_enc/src/entropy_coding/tests/parity.rs

use std::io::{Read, Cursor};
use byteorder::{LittleEndian, ReadBytesExt};

fn load_test_data(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/src/entropy_coding/testdata/v1/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read(&path).expect(&format!("Failed to read {}", path))
}

#[test]
fn test_entropy_parity() {
    let data = load_test_data("entropy_test.bin");
    let mut cursor = Cursor::new(data);

    let num_cases = cursor.read_u32::<LittleEndian>().unwrap();

    for _ in 0..num_cases {
        let num_counts = cursor.read_u32::<LittleEndian>().unwrap();
        let mut counts = vec![0i32; num_counts as usize];
        for c in &mut counts {
            *c = cursor.read_i32::<LittleEndian>().unwrap();
        }
        let expected = cursor.read_f32::<LittleEndian>().unwrap();

        let h = Histogram::from_counts(&counts);
        let actual = h.shannon_entropy();

        assert!(
            (actual - expected).abs() < 1e-5,
            "Entropy mismatch: counts={:?}, expected={}, got={}",
            counts, expected, actual
        );
    }
}
```

### Tolerance Guidelines

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| Entropy | 1e-5 | Floating point precision |
| Distance | 1e-4 | Compound calculation |
| Normalized counts | Exact | Must be identical |
| Cluster assignments | Exact | Must be identical |
| Bitstream | Exact | Must be byte-identical |

---

## 8. File Locations

### libjxl Reference (read-only)

```
~/work/jxl-efforts/libjxl/
├── lib/jxl/
│   ├── enc_ans.cc          # BuildAndEncodeHistograms, WriteTokens
│   ├── enc_ans.h
│   ├── enc_ans_params.h    # Histogram class
│   ├── enc_cluster.cc      # FastClusterHistograms
│   ├── enc_cluster.h
│   ├── enc_context_map.cc  # EncodeContextMap
│   ├── ans_params.h        # ANS_LOG_TAB_SIZE, etc.
│   └── fast_math-inl.h     # FastLog2f (if used)
└── tools/
    └── histogram_test_harness.cc  # (to create)
```

### Our Encoder

```
~/work/jxl-encoder-rs/
├── jxl_enc/src/
│   ├── entropy_coding/
│   │   ├── mod.rs
│   │   ├── ans.rs           # UPDATE
│   │   ├── histogram.rs     # NEW
│   │   ├── cluster.rs       # NEW
│   │   ├── context_map.rs   # NEW
│   │   └── testdata/v1/     # NEW - reference data
│   └── vardct/
│       └── histogram.rs     # May refactor/merge
└── histogram_clustering_design.md  # Design spec (existing)
```

### Other Reference (use only if stuck)

```
~/work/jpegli-rs/          # Has histogram clustering, not fully trusted
```

---

## 9. Constants Reference

```rust
// ANS parameters
pub const ANS_LOG_TAB_SIZE: u32 = 12;
pub const ANS_TAB_SIZE: u32 = 1 << ANS_LOG_TAB_SIZE;  // 4096
pub const ANS_TAB_MASK: u32 = ANS_TAB_SIZE - 1;       // 4095
pub const ANS_MAX_ALPHABET_SIZE: usize = 256;
pub const ANS_SIGNATURE: u32 = 0x13;
pub const RECIPROCAL_PRECISION: u32 = 44;

// Clustering parameters
pub const CLUSTERS_LIMIT: usize = 256;
pub const MIN_DISTANCE_FOR_DISTINCT: f32 = 48.0;
pub const HISTOGRAM_ROUNDING: usize = 8;

// Prefix coding parameters (for Huffman fallback)
pub const PREFIX_MAX_ALPHABET_SIZE: usize = 4096;
pub const PREFIX_MAX_BITS: usize = 15;
```

---

## 10. Checklist

### Setup

- [ ] Create `~/work/jxl-efforts/libjxl/tools/histogram_test_harness.cc`
- [ ] Add to CMakeLists.txt
- [ ] Build libjxl with harness
- [ ] Run harness, verify it produces output files
- [ ] Copy output files to `jxl_enc/src/entropy_coding/testdata/v1/`

### Phase 1: Histogram Core

- [ ] Create `jxl_enc/src/entropy_coding/histogram.rs`
- [ ] Implement `Histogram` struct
- [ ] Implement `shannon_entropy()`
- [ ] Implement `histogram_distance()`
- [ ] Implement `histogram_kl_divergence()`
- [ ] Add parity test for entropy
- [ ] Add parity test for distance
- [ ] All tests pass

### Phase 2: Clustering

- [ ] Create `jxl_enc/src/entropy_coding/cluster.rs`
- [ ] Implement `fast_cluster_histograms()`
- [ ] Add parity test for clustering
- [ ] Verify cluster assignments match exactly

### Phase 3: ANS Rebalancing

- [ ] Add `ANSEncodingHistogram` to `ans.rs`
- [ ] Implement `rebalance_histogram()`
- [ ] Implement `compute_best()`
- [ ] Add parity test for rebalancing
- [ ] Verify normalized counts match exactly

### Phase 4: Context Map

- [ ] Create `jxl_enc/src/entropy_coding/context_map.rs`
- [ ] Implement `move_to_front_transform()`
- [ ] Implement `encode_context_map()`
- [ ] Add round-trip test for MTF

### Phase 5: Pair Merge Refinement

- [ ] Implement `refine_clusters_by_merging()` in `cluster.rs`
- [ ] Add parity test with refinement enabled
- [ ] Verify works with ClusteringType::Best

### Phase 6: Integration

- [ ] Create `build_and_encode_histograms()` orchestration
- [ ] Update `write_tokens()` to use clustered histograms
- [ ] Full pipeline parity test (tokens → bitstream)
- [ ] Integration with VarDCT encoder
- [ ] All 357+ existing tests pass
- [ ] Quality/size comparison with cjxl

### Documentation

- [ ] Update ENCODING_PARITY.md
- [ ] Update module-level docs
- [ ] Document any deviations from libjxl

---

## Notes

### Floating-Point Determinism

libjxl uses specific floating-point formulas. Our implementation must use identical formulas:
- Same order of operations
- Same constant values
- No "optimization" that changes results

If libjxl uses a lookup table for log2 (check `fast_math-inl.h`), we may need to replicate it.

### Priority Queue in Pair Merging

The pair merge refinement uses a priority queue where:
- Lower cost = higher priority
- Tuple comparison: (cost, first, second, version)

Rust's `BinaryHeap` is a max-heap, so we need to reverse the comparison.

### Existing Code

The current `vardct/histogram.rs` has `HistogramBuilder` - this may be kept separate or merged with the new `histogram.rs`. The key difference is that the new code needs the full entropy/distance calculations for clustering.

---

*Document created: 2026-01-04*
*For: jxl-encoder-rs histogram clustering implementation*
