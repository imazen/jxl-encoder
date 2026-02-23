// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Histogram clustering for entropy coding.
//!
//! Ported from libjxl-tiny enc_cluster.cc

use crate::entropy_coding::encode::{ALPHABET_SIZE, create_huffman_tree};

/// A histogram of symbol counts.
#[derive(Clone)]
pub struct Histogram {
    pub counts: [u32; ALPHABET_SIZE],
    pub total_count: u32,
    pub bit_cost: f32,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            counts: [0u32; ALPHABET_SIZE],
            total_count: 0,
            bit_cost: 0.0,
        }
    }
}

impl Histogram {
    /// Create a new empty histogram.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a symbol to the histogram.
    pub fn add(&mut self, symbol: usize) {
        self.counts[symbol] += 1;
        self.total_count += 1;
    }

    /// Merge another histogram into this one.
    pub fn add_histogram(&mut self, other: &Self) {
        for i in 0..ALPHABET_SIZE {
            self.counts[i] += other.counts[i];
        }
        self.total_count += other.total_count;
    }

    /// Calculate bit cost for this histogram.
    pub fn compute_bit_cost(&mut self) {
        self.bit_cost = 0.0;
        if self.total_count == 0 {
            return;
        }
        let mut depths = [0u8; ALPHABET_SIZE];
        create_huffman_tree(&self.counts, ALPHABET_SIZE, 15, &mut depths);
        for (count, depth) in self.counts.iter().zip(depths.iter()) {
            self.bit_cost += *count as f32 * *depth as f32;
        }
    }
}

/// Calculate distance between two histograms using a reusable scratch buffer.
/// Distance is the increase in bit cost when merging them.
fn histogram_distance_reuse(a: &Histogram, b: &Histogram, scratch: &mut Histogram) -> f32 {
    if a.total_count == 0 || b.total_count == 0 {
        return 0.0;
    }
    // Reuse scratch allocation instead of cloning per call
    scratch.counts = a.counts;
    scratch.total_count = a.total_count;
    scratch.add_histogram(b);
    scratch.compute_bit_cost();
    scratch.bit_cost - a.bit_cost - b.bit_cost
}

/// Maximum number of clusters.
const CLUSTERS_LIMIT: usize = 8;

/// Minimum distance to consider histograms distinct.
const MIN_DISTANCE_FOR_DISTINCT: f32 = 64.0;

/// Fast k-means-like clustering of histograms.
fn fast_cluster_histograms(
    input: &[Histogram],
    max_histograms: usize,
) -> (Vec<Histogram>, Vec<u32>) {
    let mut out = Vec::with_capacity(max_histograms);
    let mut symbols = vec![max_histograms as u32; input.len()];
    let mut dists = vec![f32::MAX; input.len()];
    let mut dist_scratch = Histogram::new();

    // Compute bit costs for all histograms
    let mut input_with_costs: Vec<Histogram> = input.to_vec();
    for (i, h) in input_with_costs.iter_mut().enumerate() {
        if h.total_count == 0 {
            symbols[i] = 0;
            dists[i] = 0.0;
        } else {
            h.compute_bit_cost();
        }
    }

    // Find histogram with largest count
    let mut largest_idx = 0;
    for (i, h) in input_with_costs.iter().enumerate() {
        if h.total_count > input_with_costs[largest_idx].total_count {
            largest_idx = i;
        }
    }

    while out.len() < max_histograms {
        symbols[largest_idx] = out.len() as u32;
        out.push(input_with_costs[largest_idx].clone());
        dists[largest_idx] = 0.0;
        largest_idx = 0;

        for i in 0..input.len() {
            if dists[i] == 0.0 {
                continue;
            }
            let dist = histogram_distance_reuse(
                &input_with_costs[i],
                out.last().unwrap(),
                &mut dist_scratch,
            );
            dists[i] = dists[i].min(dist);
            if dists[i] > dists[largest_idx] {
                largest_idx = i;
            }
        }

        if dists[largest_idx] < MIN_DISTANCE_FOR_DISTINCT {
            break;
        }
    }

    // Assign remaining histograms to closest cluster
    for i in 0..input.len() {
        if symbols[i] != max_histograms as u32 {
            continue;
        }
        let mut best = 0;
        let mut best_dist =
            histogram_distance_reuse(&input_with_costs[i], &out[best], &mut dist_scratch);
        for (j, out_hist) in out.iter().enumerate().skip(1) {
            let dist = histogram_distance_reuse(&input_with_costs[i], out_hist, &mut dist_scratch);
            if dist < best_dist {
                best = j;
                best_dist = dist;
            }
        }
        out[best].add_histogram(&input_with_costs[i]);
        out[best].compute_bit_cost();
        symbols[i] = best as u32;
    }

    (out, symbols)
}

/// Reindex histograms so symbols come in increasing order.
fn histogram_reindex(symbols: &[u32], histograms: &mut Vec<Histogram>) -> Vec<u8> {
    use std::collections::HashMap;

    let tmp = histograms.clone();
    let mut new_index: HashMap<u32, usize> = HashMap::new();
    let mut next_index = 0;

    for &symbol in symbols {
        if let std::collections::hash_map::Entry::Vacant(e) = new_index.entry(symbol) {
            e.insert(next_index);
            if next_index < histograms.len() {
                histograms[next_index] = tmp[symbol as usize].clone();
            }
            next_index += 1;
        }
    }

    histograms.truncate(next_index);

    symbols
        .iter()
        .map(|&s| *new_index.get(&s).unwrap() as u8)
        .collect()
}

/// Cluster histograms and return the context map.
pub fn cluster_histograms(histograms: &mut Vec<Histogram>) -> Vec<u8> {
    if histograms.len() <= 1 {
        return vec![0; histograms.len()];
    }

    let max_histograms = CLUSTERS_LIMIT.min(histograms.len());

    let input = histograms.clone();
    let (mut clustered, symbols) = fast_cluster_histograms(&input, max_histograms);

    // Reindex to canonical form
    let context_map = histogram_reindex(&symbols, &mut clustered);
    *histograms = clustered;

    context_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let mut h = Histogram::new();
        h.add(0);
        h.add(0);
        h.add(1);
        assert_eq!(h.counts[0], 2);
        assert_eq!(h.counts[1], 1);
        assert_eq!(h.total_count, 3);
    }

    #[test]
    fn test_cluster_single() {
        let mut histograms = vec![Histogram::new()];
        histograms[0].add(0);
        let ctx_map = cluster_histograms(&mut histograms);
        assert_eq!(ctx_map, vec![0]);
    }

    #[test]
    fn test_cluster_identical() {
        // Two identical histograms should be merged
        let mut histograms = vec![Histogram::new(), Histogram::new()];
        for _ in 0..100 {
            histograms[0].add(0);
            histograms[1].add(0);
        }
        let ctx_map = cluster_histograms(&mut histograms);
        // Both should map to same cluster
        assert_eq!(ctx_map[0], ctx_map[1]);
    }
}
