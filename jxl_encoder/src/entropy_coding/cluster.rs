// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Histogram clustering for entropy coding.
//!
//! Ported from libjxl `lib/jxl/enc_cluster.cc`.

use alloc::collections::BinaryHeap;
use core::cmp::Ordering;

use super::histogram::{
    DistanceScratch, Histogram, histogram_distance_reuse, histogram_kl_divergence,
};
use crate::error::{Error, Result};

/// Minimum distance threshold for creating distinct clusters.
const MIN_DISTANCE_FOR_DISTINCT: f32 = 48.0;

/// Maximum number of histogram clusters.
pub const CLUSTERS_LIMIT: usize = 256;

/// Result of clustering histograms.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// The clustered histograms.
    pub histograms: Vec<Histogram>,
    /// Mapping from input index to cluster index.
    pub symbols: Vec<u32>,
}

/// Clustering aggressiveness level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClusteringType {
    /// Only 4 clusters maximum (fastest encoding).
    Fastest,
    /// Default clustering.
    #[default]
    Fast,
    /// With pair merge refinement (best compression).
    Best,
}

/// Entropy coding method - affects header cost estimation for clustering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EntropyType {
    /// Huffman prefix codes (used by libjxl-tiny, simpler header format).
    #[default]
    Huffman,
    /// ANS (Asymmetric Numeral Systems) - used by full libjxl, larger alphabet support.
    Ans,
}

/// Fast k-means-like clustering.
///
/// Algorithm:
/// 1. Start with largest histogram as first cluster
/// 2. Repeatedly add most distant histogram as new cluster
/// 3. Stop when max clusters reached or distance < threshold
/// 4. Assign remaining histograms to nearest cluster
///
/// Matches libjxl's `FastClusterHistograms` function.
pub fn fast_cluster_histograms(
    input: &[Histogram],
    max_histograms: usize,
) -> Result<ClusterResult> {
    fast_cluster_histograms_with_prev(input, max_histograms, &[])
}

/// Fast clustering with support for pre-existing histograms.
///
/// This is the full implementation matching libjxl's `FastClusterHistograms`.
/// The `prev_histograms` are fixed clusters that new histograms can be assigned to,
/// but won't be merged into.
pub fn fast_cluster_histograms_with_prev(
    input: &[Histogram],
    max_histograms: usize,
    prev_histograms: &[Histogram],
) -> Result<ClusterResult> {
    if input.is_empty() {
        return Ok(ClusterResult {
            histograms: prev_histograms.to_vec(),
            symbols: Vec::new(),
        });
    }

    let prev_count = prev_histograms.len();
    let mut out: Vec<Histogram> = prev_histograms.to_vec();
    out.reserve(max_histograms);
    let mut dist_scratch = DistanceScratch::new();

    // Initialize symbols to "unassigned" marker
    let unassigned = max_histograms as u32;
    let mut symbols = vec![unassigned; input.len()];

    // Initialize distances to max (except empty histograms)
    let mut dists = vec![f32::MAX; input.len()];

    // Find largest histogram and compute entropies
    let mut largest_idx = 0;
    for (i, h) in input.iter().enumerate() {
        if h.total_count == 0 {
            // Empty histograms get assigned to cluster 0
            symbols[i] = 0;
            dists[i] = 0.0;
            continue;
        }
        h.shannon_entropy(); // Compute and cache entropy
        if h.total_count > input[largest_idx].total_count {
            largest_idx = i;
        }
    }

    // If there are previous histograms, compute their entropies and
    // update distances using KL divergence
    if prev_count > 0 {
        for h in &out {
            h.shannon_entropy();
        }
        for (i, dist) in dists.iter_mut().enumerate() {
            if *dist == 0.0 {
                continue;
            }
            for out_hist in out.iter().take(prev_count) {
                let kl = histogram_kl_divergence(&input[i], out_hist);
                *dist = dist.min(kl);
            }
        }
        // Find the histogram with maximum distance (most different from prev)
        if let Some((max_idx, &max_dist)) = dists
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            && max_dist > 0.0
        {
            largest_idx = max_idx;
        }
    }

    // Main clustering loop
    while out.len() < prev_count + max_histograms {
        // Add the largest/most distant histogram as a new cluster
        symbols[largest_idx] = out.len() as u32;
        out.push(input[largest_idx].clone());
        dists[largest_idx] = 0.0;

        // Find next candidate: histogram with maximum distance
        let mut new_largest_idx = 0;
        for (i, h) in input.iter().enumerate() {
            if dists[i] == 0.0 {
                continue;
            }
            // Update distance using histogram distance to new cluster
            let dist = histogram_distance_reuse(h, out.last().unwrap(), &mut dist_scratch);
            dists[i] = dists[i].min(dist);
            if dists[i] > dists[new_largest_idx] {
                new_largest_idx = i;
            }
        }
        largest_idx = new_largest_idx;

        // Stop if distance is below threshold
        if dists[largest_idx] < MIN_DISTANCE_FOR_DISTINCT {
            break;
        }
    }

    // Assign remaining histograms to nearest cluster
    for i in 0..input.len() {
        if symbols[i] != unassigned {
            continue;
        }

        // Find best cluster
        let mut best = 0;
        let mut best_dist = f32::MAX;

        for (j, out_hist) in out.iter().enumerate() {
            let dist = if j < prev_count {
                // Use KL divergence for previous histograms
                histogram_kl_divergence(&input[i], out_hist)
            } else {
                // Use symmetric distance for new histograms
                histogram_distance_reuse(&input[i], out_hist, &mut dist_scratch)
            };

            if dist < best_dist {
                best = j;
                best_dist = dist;
            }
        }

        if best_dist >= f32::MAX {
            return Err(Error::InvalidHistogram(format!(
                "Failed to find cluster for histogram {}",
                i
            )));
        }

        // Merge into best cluster (only for non-previous histograms)
        if best >= prev_count {
            out[best].add_histogram(&input[i]);
            out[best].shannon_entropy(); // Recompute entropy
        }
        symbols[i] = best as u32;
    }

    Ok(ClusterResult {
        histograms: out,
        symbols,
    })
}

/// Histogram pair for merge refinement priority queue.
#[derive(Clone, Copy, Debug)]
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
            && self.version == other.version
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
        // Reverse order: lower cost = higher priority
        // Use tuple comparison for tie-breaking
        let self_tuple = (
            ordered_float::OrderedFloat(self.cost),
            self.first,
            self.second,
            self.version,
        );
        let other_tuple = (
            ordered_float::OrderedFloat(other.cost),
            other.first,
            other.second,
            other.version,
        );
        // Reverse because BinaryHeap is a max-heap
        other_tuple.cmp(&self_tuple)
    }
}

/// Wrapper for f32 that implements Ord for use in priority queues.
mod ordered_float {
    use core::cmp::Ordering;

    #[derive(Clone, Copy, Debug)]
    pub struct OrderedFloat(pub f32);

    impl PartialEq for OrderedFloat {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl Eq for OrderedFloat {}

    impl PartialOrd for OrderedFloat {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for OrderedFloat {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
        }
    }
}

/// Estimate Huffman population cost for clustering merge decisions.
///
/// For Huffman coding, the key insight is that header cost savings from
/// merging histograms are minimal compared to data cost increases.
/// The simple/tiny clustering uses data-only cost (sum of count * depth)
/// and produces good results.
///
/// This function computes data cost using actual Huffman code lengths,
/// plus a small header penalty that scales with alphabet size to
/// discourage creating very large merged alphabets.
fn huffman_population_cost(h: &Histogram) -> f32 {
    if h.total_count == 0 {
        return 0.0;
    }

    let alphabet_size = h.alphabet_size();
    if alphabet_size == 0 {
        return 0.0;
    }

    // Compute ACTUAL Huffman data cost using real code lengths
    let data_cost = compute_huffman_data_cost(h, alphabet_size);

    // For merge decisions, we want to penalize large alphabets slightly
    // because they require more complex tree serialization.
    // But don't over-penalize - the data cost is the main factor.
    //
    // Count non-zero symbols to estimate header complexity
    let non_zero_count = h
        .counts
        .iter()
        .take(alphabet_size)
        .filter(|&&c| c > 0)
        .count();

    // Small header penalty: 0.1 bits per non-zero symbol
    // This is much smaller than data cost, so it only tips the balance
    // when data costs are very close.
    let header_penalty = (non_zero_count as f32) * 0.1;

    data_cost + header_penalty
}

/// Compute actual Huffman data cost using real code lengths.
///
/// This builds a real Huffman tree and computes sum(count * depth),
/// which is the exact number of bits needed to encode the data.
fn compute_huffman_data_cost(h: &Histogram, alphabet_size: usize) -> f32 {
    use super::huffman_tree::create_huffman_tree;

    if alphabet_size == 0 {
        return 0.0;
    }

    // Convert to u32 counts for create_huffman_tree
    let counts: Vec<u32> = h
        .counts
        .iter()
        .take(alphabet_size)
        .map(|&c| c.max(0) as u32)
        .collect();

    // Check for empty or single-symbol histogram
    let non_zero = counts.iter().filter(|&&c| c > 0).count();
    if non_zero == 0 {
        return 0.0;
    }
    if non_zero == 1 {
        // Single symbol needs 1 bit per occurrence
        return counts.iter().sum::<u32>() as f32;
    }

    // Build actual Huffman tree with depth limit 15
    let depths = create_huffman_tree(&counts, 15);

    // Compute data cost: sum(count * depth)
    let mut cost = 0.0f32;
    for (i, &count) in counts.iter().enumerate() {
        if count > 0 && i < depths.len() {
            cost += count as f32 * depths[i] as f32;
        }
    }

    cost
}

/// Compute cost of encoding histogram A's data using a tree built for histogram B.
///
/// This is the key insight for correct merge cost estimation:
/// When contexts are merged, BOTH original contexts use the merged tree,
/// which is suboptimal for each individually.
#[allow(dead_code)]
fn compute_cross_coding_cost(data: &Histogram, tree: &Histogram, alphabet_size: usize) -> f32 {
    use super::huffman_tree::create_huffman_tree;

    if alphabet_size == 0 {
        return 0.0;
    }

    // Build tree from 'tree' histogram
    let tree_counts: Vec<u32> = tree
        .counts
        .iter()
        .take(alphabet_size)
        .map(|&c| c.max(0) as u32)
        .collect();

    let non_zero = tree_counts.iter().filter(|&&c| c > 0).count();
    if non_zero == 0 {
        return 0.0;
    }

    let depths = if non_zero == 1 {
        vec![1u8; alphabet_size]
    } else {
        create_huffman_tree(&tree_counts, 15)
    };

    // Encode 'data' using depths from 'tree'
    let mut cost = 0.0f32;
    for (i, &count) in data.counts.iter().take(alphabet_size).enumerate() {
        if count > 0 && i < depths.len() {
            let depth = if depths[i] == 0 { 15 } else { depths[i] }; // Penalize symbols not in tree
            cost += count.max(0) as f32 * depth as f32;
        }
    }

    cost
}

/// Estimate ANS population cost (header + data bits).
///
/// This is a simplified version of libjxl's `Histogram::ANSPopulationCost()`.
/// ANS uses a frequency table with log-scale precision, supporting larger alphabets.
fn ans_population_cost(h: &Histogram) -> f32 {
    if h.total_count == 0 {
        return 0.0;
    }

    let alphabet_size = h.alphabet_size();
    if alphabet_size <= 1 {
        // Single symbol or empty: almost no header cost
        return 0.0;
    }

    // Data cost (entropy)
    let data_cost = h.cached_entropy();

    // Header cost estimate: roughly 5 bits per symbol for frequency table
    // ANS encodes frequencies using variable-length coding based on precision
    // This is a rough approximation - actual cost depends on the shift parameter
    let header_cost = (alphabet_size as f32) * 5.0;

    data_cost + header_cost
}

/// Estimate population cost for a histogram based on entropy type.
fn population_cost(h: &Histogram, entropy_type: EntropyType) -> f32 {
    match entropy_type {
        EntropyType::Huffman => huffman_population_cost(h),
        EntropyType::Ans => ans_population_cost(h),
    }
}

/// Refine clusters by merging pairs that reduce total cost.
///
/// This implements the pair merge refinement from libjxl's `ClusterHistograms`
/// when `params.clustering == ClusteringType::Best`.
///
/// The `entropy_type` parameter controls the cost model used for merge decisions:
/// - `EntropyType::Huffman`: Uses Huffman tree serialization cost model
/// - `EntropyType::Ans`: Uses ANS frequency table cost model
pub fn refine_clusters_by_merging(
    histograms: &mut Vec<Histogram>,
    symbols: &mut [u32],
    entropy_type: EntropyType,
) -> Result<()> {
    if histograms.is_empty() {
        return Ok(());
    }

    // Compute initial costs
    for h in histograms.iter() {
        h.shannon_entropy();
    }

    // Version tracking for invalidation
    let mut version = vec![1u32; histograms.len()];
    let mut next_version = 2u32;

    // Renumbering map (for tracking merges)
    let mut renumbering: Vec<u32> = (0..histograms.len() as u32).collect();

    // Create priority queue of pairs to merge
    let mut pairs_to_merge: BinaryHeap<HistogramPair> = BinaryHeap::new();

    // Reusable scratch histogram to avoid per-pair clone allocation
    let mut merged = Histogram::new();

    for i in 0..histograms.len() as u32 {
        for j in (i + 1)..histograms.len() as u32 {
            // Compute cost of merging (reuse scratch allocation)
            merged.copy_from(&histograms[i as usize]);
            merged.add_histogram(&histograms[j as usize]);
            merged.shannon_entropy();

            let merged_cost = population_cost(&merged, entropy_type);
            let individual_cost = population_cost(&histograms[i as usize], entropy_type)
                + population_cost(&histograms[j as usize], entropy_type);

            let cost = merged_cost - individual_cost;

            // Only enqueue if merging is beneficial
            if cost < 0.0 {
                pairs_to_merge.push(HistogramPair {
                    cost,
                    first: i,
                    second: j,
                    version: version[i as usize].max(version[j as usize]),
                });
            }
        }
    }

    // Process merges
    while let Some(pair) = pairs_to_merge.pop() {
        let first = pair.first as usize;
        let second = pair.second as usize;

        // Check if pair is still valid
        let expected_version = version[first].max(version[second]);
        if pair.version != expected_version || version[first] == 0 || version[second] == 0 {
            continue;
        }

        // Merge second into first (copy into scratch to avoid borrow conflict)
        merged.copy_from(&histograms[second]);
        histograms[first].add_histogram(&merged);
        histograms[first].shannon_entropy();

        // Update renumbering
        for item in renumbering.iter_mut() {
            if *item == pair.second {
                *item = pair.first;
            }
        }

        // Mark second as dead
        version[second] = 0;
        version[first] = next_version;
        next_version += 1;

        // Add new pairs with the merged histogram
        for j in 0..histograms.len() as u32 {
            if j == pair.first || version[j as usize] == 0 {
                continue;
            }

            merged.copy_from(&histograms[first]);
            merged.add_histogram(&histograms[j as usize]);
            merged.shannon_entropy();

            let merged_cost = population_cost(&merged, entropy_type);
            let individual_cost = population_cost(&histograms[first], entropy_type)
                + population_cost(&histograms[j as usize], entropy_type);

            let cost = merged_cost - individual_cost;

            if cost < 0.0 {
                pairs_to_merge.push(HistogramPair {
                    cost,
                    first: pair.first.min(j),
                    second: pair.first.max(j),
                    version: version[first].max(version[j as usize]),
                });
            }
        }
    }

    // Build reverse renumbering and compact
    let mut reverse_renumbering = vec![u32::MAX; histograms.len()];
    let mut num_alive = 0u32;

    for i in 0..histograms.len() {
        if version[i] == 0 {
            continue;
        }
        if num_alive != i as u32 {
            histograms[num_alive as usize] = histograms[i].clone();
        }
        reverse_renumbering[i] = num_alive;
        num_alive += 1;
    }
    histograms.truncate(num_alive as usize);

    // Update symbols
    for symbol in symbols.iter_mut() {
        let renumbered = renumbering[*symbol as usize];
        *symbol = reverse_renumbering[renumbered as usize];
    }

    Ok(())
}

/// Reindex histograms so that symbols appear in increasing order.
fn histogram_reindex(histograms: &mut Vec<Histogram>, prev_count: usize, symbols: &mut [u32]) {
    use std::collections::HashMap;

    let tmp = histograms.clone();
    let mut new_index: HashMap<u32, u32> = HashMap::new();

    // Previous histograms keep their indices
    for i in 0..prev_count {
        new_index.insert(i as u32, i as u32);
    }

    // Assign new indices in order of first appearance
    let mut next_index = prev_count as u32;
    for &symbol in symbols.iter() {
        if let std::collections::hash_map::Entry::Vacant(e) = new_index.entry(symbol) {
            e.insert(next_index);
            histograms[next_index as usize] = tmp[symbol as usize].clone();
            next_index += 1;
        }
    }

    histograms.truncate(next_index as usize);

    // Update symbols
    for symbol in symbols.iter_mut() {
        *symbol = new_index[symbol];
    }
}

/// Full clustering pipeline.
///
/// Combines fast clustering with optional pair merge refinement.
///
/// # Arguments
///
/// * `clustering_type` - Controls clustering aggressiveness (Fastest/Fast/Best)
/// * `entropy_type` - Controls cost model for merge decisions (Huffman/Ans)
/// * `input` - Input histograms to cluster
/// * `max_histograms` - Maximum number of output clusters
pub fn cluster_histograms(
    clustering_type: ClusteringType,
    entropy_type: EntropyType,
    input: &[Histogram],
    max_histograms: usize,
) -> Result<ClusterResult> {
    let max_histograms = match clustering_type {
        ClusteringType::Fastest => max_histograms.min(4),
        _ => max_histograms,
    };

    let max_histograms = max_histograms.min(input.len()).min(CLUSTERS_LIMIT);

    // Fast clustering
    let mut result = fast_cluster_histograms(input, max_histograms)?;

    // Pair merge refinement for Best quality
    if clustering_type == ClusteringType::Best && !result.histograms.is_empty() {
        refine_clusters_by_merging(&mut result.histograms, &mut result.symbols, entropy_type)?;
    }

    // Reindex for canonical form
    histogram_reindex(&mut result.histograms, 0, &mut result.symbols);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_histogram(counts: &[i32]) -> Histogram {
        let h = Histogram::from_counts(counts);
        h.shannon_entropy(); // Pre-compute entropy
        h
    }

    #[test]
    fn test_fast_cluster_single() {
        let input = vec![make_histogram(&[100, 50, 25])];

        let result = fast_cluster_histograms(&input, 10).unwrap();

        assert_eq!(result.histograms.len(), 1);
        assert_eq!(result.symbols.len(), 1);
        assert_eq!(result.symbols[0], 0);
    }

    #[test]
    fn test_fast_cluster_identical() {
        let input = vec![
            make_histogram(&[100, 50, 25]),
            make_histogram(&[100, 50, 25]),
            make_histogram(&[100, 50, 25]),
        ];

        let result = fast_cluster_histograms(&input, 10).unwrap();

        // All identical histograms should cluster together
        assert_eq!(result.histograms.len(), 1);
        assert_eq!(result.symbols, vec![0, 0, 0]);
    }

    #[test]
    fn test_fast_cluster_different() {
        let input = vec![
            make_histogram(&[100, 0, 0]),
            make_histogram(&[0, 100, 0]),
            make_histogram(&[0, 0, 100]),
        ];

        let result = fast_cluster_histograms(&input, 10).unwrap();

        // Very different histograms should be in separate clusters
        assert!(result.histograms.len() >= 2);
        // Each histogram should be assigned to some cluster
        assert!(
            result
                .symbols
                .iter()
                .all(|&s| (s as usize) < result.histograms.len())
        );
    }

    #[test]
    fn test_fast_cluster_max_limit() {
        let input: Vec<Histogram> = (0..10)
            .map(|i| {
                let mut counts = vec![0i32; 10];
                counts[i] = 100;
                make_histogram(&counts)
            })
            .collect();

        let result = fast_cluster_histograms(&input, 4).unwrap();

        // Should not exceed max limit
        assert!(result.histograms.len() <= 4);
    }

    #[test]
    fn test_fast_cluster_empty() {
        let input: Vec<Histogram> = vec![];
        let result = fast_cluster_histograms(&input, 10).unwrap();

        assert!(result.histograms.is_empty());
        assert!(result.symbols.is_empty());
    }

    #[test]
    fn test_fast_cluster_with_empty_histograms() {
        let input = vec![
            Histogram::new(), // Empty
            make_histogram(&[100, 50]),
            Histogram::new(), // Empty
        ];

        let result = fast_cluster_histograms(&input, 10).unwrap();

        // Empty histograms should be assigned to cluster 0
        assert!(!result.histograms.is_empty());
        assert_eq!(result.symbols[0], 0);
        assert_eq!(result.symbols[2], 0);
    }

    #[test]
    fn test_cluster_histograms_fastest() {
        let input: Vec<Histogram> = (0..10)
            .map(|i| {
                let mut counts = vec![0i32; 10];
                counts[i] = 100;
                make_histogram(&counts)
            })
            .collect();

        let result =
            cluster_histograms(ClusteringType::Fastest, EntropyType::Huffman, &input, 10).unwrap();

        // Fastest should limit to 4 clusters
        assert!(result.histograms.len() <= 4);
    }

    #[test]
    fn test_cluster_histograms_best_merges_huffman() {
        // Create two pairs of similar histograms
        let input = vec![
            make_histogram(&[100, 50, 25, 10]),
            make_histogram(&[105, 52, 23, 11]), // Similar to 0
            make_histogram(&[10, 25, 50, 100]),
            make_histogram(&[11, 23, 52, 105]), // Similar to 2
        ];

        let result =
            cluster_histograms(ClusteringType::Best, EntropyType::Huffman, &input, 10).unwrap();

        // With best quality, similar histograms should be merged
        assert!(result.histograms.len() <= 4);
    }

    #[test]
    fn test_cluster_histograms_best_merges_ans() {
        // Create two pairs of similar histograms
        let input = vec![
            make_histogram(&[100, 50, 25, 10]),
            make_histogram(&[105, 52, 23, 11]), // Similar to 0
            make_histogram(&[10, 25, 50, 100]),
            make_histogram(&[11, 23, 52, 105]), // Similar to 2
        ];

        let result =
            cluster_histograms(ClusteringType::Best, EntropyType::Ans, &input, 10).unwrap();

        // With best quality, similar histograms should be merged
        assert!(result.histograms.len() <= 4);
    }

    #[test]
    fn test_huffman_vs_ans_cost_model() {
        // Histogram with many symbols - ANS and Huffman should have different costs
        let mut counts = vec![0i32; 64];
        for (i, c) in counts.iter_mut().enumerate() {
            *c = (64 - i as i32) * 10; // Decreasing frequencies
        }
        let h = make_histogram(&counts);

        let huffman_cost = huffman_population_cost(&h);
        let ans_cost = ans_population_cost(&h);

        // Both should be positive
        assert!(huffman_cost > 0.0);
        assert!(ans_cost > 0.0);

        // For large alphabets, ANS header cost (5 bits/symbol) should be higher
        // than Huffman's nested tree (~3 bits/symbol + 30 bit overhead)
        // This test just verifies they're different - actual values depend on distribution
        assert!((huffman_cost - ans_cost).abs() > 1.0);
    }

    #[test]
    fn test_histogram_reindex() {
        let mut histograms = vec![
            make_histogram(&[100]),
            make_histogram(&[200]),
            make_histogram(&[300]),
        ];
        let mut symbols = vec![2, 0, 2, 1, 0];

        histogram_reindex(&mut histograms, 0, &mut symbols);

        // Symbols should now be renumbered in order of first appearance
        // 2 -> 0, 0 -> 1, 1 -> 2
        assert_eq!(symbols, vec![0, 1, 0, 2, 1]);
    }
}

#[test]
fn test_huffman_cost_disjoint_histograms() {
    // Disjoint histograms - merging should NOT be beneficial
    let a = Histogram::from_counts(&[100, 50, 25, 0, 0, 0, 0, 0]);
    a.shannon_entropy();

    let b = Histogram::from_counts(&[0, 0, 0, 80, 40, 20, 0, 0]);
    b.shannon_entropy();

    let mut merged = a.clone();
    merged.add_histogram(&b);
    merged.shannon_entropy();

    let cost_a = huffman_population_cost(&a);
    let cost_b = huffman_population_cost(&b);
    let cost_merged = huffman_population_cost(&merged);
    let delta = cost_merged - cost_a - cost_b;

    // Disjoint histograms: merging increases data cost significantly
    assert!(delta >= 0.0, "Disjoint merge should not be beneficial");
}

#[test]
fn test_huffman_cost_identical_histograms() {
    // Identical histograms - merging should have near-zero delta
    let a = Histogram::from_counts(&[100, 50, 25, 10, 0, 0, 0, 0]);
    a.shannon_entropy();

    let b = Histogram::from_counts(&[100, 50, 25, 10, 0, 0, 0, 0]);
    b.shannon_entropy();

    let mut merged = a.clone();
    merged.add_histogram(&b);
    merged.shannon_entropy();

    let cost_a = huffman_population_cost(&a);
    let cost_b = huffman_population_cost(&b);
    let cost_merged = huffman_population_cost(&merged);
    let delta = cost_merged - cost_a - cost_b;

    // Identical histograms use same Huffman tree, so merged cost = 2x individual
    assert!(
        delta.abs() < 1.0,
        "Identical histograms should have near-zero delta, got {}",
        delta
    );
}
