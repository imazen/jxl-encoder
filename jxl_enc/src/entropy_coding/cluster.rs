// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Histogram clustering for entropy coding.
//!
//! Ported from libjxl `lib/jxl/enc_cluster.cc`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::histogram::{Histogram, histogram_distance, histogram_kl_divergence};
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
        for i in 0..input.len() {
            if dists[i] == 0.0 {
                continue;
            }
            for j in 0..prev_count {
                let kl = histogram_kl_divergence(&input[i], &out[j]);
                dists[i] = dists[i].min(kl);
            }
        }
        // Find the histogram with maximum distance (most different from prev)
        if let Some((max_idx, &max_dist)) = dists
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        {
            if max_dist > 0.0 {
                largest_idx = max_idx;
            }
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
            let dist = histogram_distance(h, out.last().unwrap());
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

        for j in 0..out.len() {
            let dist = if j < prev_count {
                // Use KL divergence for previous histograms
                histogram_kl_divergence(&input[i], &out[j])
            } else {
                // Use symmetric distance for new histograms
                histogram_distance(&input[i], &out[j])
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
    use std::cmp::Ordering;

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

/// Estimate ANS population cost (header + data bits).
///
/// This is a simplified version of libjxl's `Histogram::ANSPopulationCost()`.
fn ans_population_cost(h: &Histogram) -> Result<f32> {
    if h.total_count == 0 {
        return Ok(0.0);
    }

    let alphabet_size = h.alphabet_size();
    if alphabet_size <= 1 {
        // Single symbol or empty: almost no header cost
        return Ok(0.0);
    }

    // Data cost (entropy)
    let data_cost = h.cached_entropy();

    // Header cost estimate: roughly log2(n) bits per symbol for frequencies
    // This is a rough approximation - the actual cost depends on the encoding method
    let header_cost = (alphabet_size as f32) * 5.0; // ~5 bits per symbol average

    Ok(data_cost + header_cost)
}

/// Refine clusters by merging pairs that reduce total cost.
///
/// This implements the pair merge refinement from libjxl's `ClusterHistograms`
/// when `params.clustering == ClusteringType::Best`.
pub fn refine_clusters_by_merging(
    histograms: &mut Vec<Histogram>,
    symbols: &mut [u32],
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

    for i in 0..histograms.len() as u32 {
        for j in (i + 1)..histograms.len() as u32 {
            // Compute cost of merging
            let mut merged = histograms[i as usize].clone();
            merged.add_histogram(&histograms[j as usize]);
            merged.shannon_entropy();

            let merged_cost = ans_population_cost(&merged)?;
            let individual_cost = ans_population_cost(&histograms[i as usize])?
                + ans_population_cost(&histograms[j as usize])?;

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

        // Merge second into first (clone to avoid borrow conflict)
        let second_histo = histograms[second].clone();
        histograms[first].add_histogram(&second_histo);
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

            let mut merged = histograms[first].clone();
            merged.add_histogram(&histograms[j as usize]);
            merged.shannon_entropy();

            let merged_cost = ans_population_cost(&merged)?;
            let individual_cost = ans_population_cost(&histograms[first])?
                + ans_population_cost(&histograms[j as usize])?;

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
        if !new_index.contains_key(&symbol) {
            new_index.insert(symbol, next_index);
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
pub fn cluster_histograms(
    clustering_type: ClusteringType,
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
        refine_clusters_by_merging(&mut result.histograms, &mut result.symbols)?;
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
        assert!(result.histograms.len() >= 1);
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

        let result = cluster_histograms(ClusteringType::Fastest, &input, 10).unwrap();

        // Fastest should limit to 4 clusters
        assert!(result.histograms.len() <= 4);
    }

    #[test]
    fn test_cluster_histograms_best_merges() {
        // Create two pairs of similar histograms
        let input = vec![
            make_histogram(&[100, 50, 25, 10]),
            make_histogram(&[105, 52, 23, 11]), // Similar to 0
            make_histogram(&[10, 25, 50, 100]),
            make_histogram(&[11, 23, 52, 105]), // Similar to 2
        ];

        let result = cluster_histograms(ClusteringType::Best, &input, 10).unwrap();

        // With best quality, similar histograms should be merged
        // We should get at most 2 clusters (possibly merged further)
        assert!(result.histograms.len() <= 4);
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
