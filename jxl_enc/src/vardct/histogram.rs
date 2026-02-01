//! Histogram building for VarDCT coefficient encoding.
//!
//! Collects token statistics and builds histograms for entropy coding.

use super::context::BlockContextMap;
use super::tokenize::Token;
use crate::entropy_coding::ans::{ANS_MAX_ALPHABET_SIZE, AnsDistribution};
use crate::entropy_coding::hybrid_uint::HybridUintConfig;
use crate::entropy_coding::{
    ClusteringType, EntropyType, Histogram as EntropyHistogram, cluster_histograms,
};
use crate::error::Result;

/// Compute the global alphabet size from tokens, applying HybridUint encoding.
///
/// This MUST be used consistently when writing histograms and tokens to ensure
/// the decoder sees the same alphabet size in both places.
pub fn compute_alphabet_size_from_tokens(tokens: &[Token]) -> usize {
    if tokens.is_empty() {
        return 1;
    }

    // Use the same HybridUint config as write_histograms_clustered
    let hybrid_config = HybridUintConfig::new(4, 2, 0);

    let max_symbol = tokens.iter().map(|t| t.value as usize).max().unwrap_or(0);
    let max_token = if max_symbol < hybrid_config.split as usize {
        max_symbol
    } else {
        let (token, _, _) = hybrid_config.encode(max_symbol as u32);
        token as usize
    };
    max_token + 1
}

/// Histogram builder for AC coefficient tokens.
pub struct HistogramBuilder {
    /// Counts per context per symbol.
    /// Indexed as [context][symbol].
    counts: Vec<Vec<u32>>,
    /// Number of contexts.
    num_contexts: usize,
}

impl HistogramBuilder {
    /// Create a new histogram builder for the given number of contexts.
    pub fn new(num_contexts: usize) -> Self {
        Self {
            counts: vec![vec![0u32; ANS_MAX_ALPHABET_SIZE]; num_contexts],
            num_contexts,
        }
    }

    /// Add tokens to the histogram.
    pub fn add_tokens(&mut self, tokens: &[Token]) {
        for token in tokens {
            let ctx = token.context as usize;
            let val = (token.value as usize).min(ANS_MAX_ALPHABET_SIZE - 1);
            if ctx < self.num_contexts {
                self.counts[ctx][val] += 1;
            }
        }
    }

    /// Build ANS distributions from accumulated counts.
    pub fn build_distributions(&self) -> Result<Vec<AnsDistribution>> {
        let mut distributions = Vec::with_capacity(self.num_contexts);

        for ctx in 0..self.num_contexts {
            let counts = &self.counts[ctx];

            // Find the maximum symbol used
            let max_symbol = counts
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0)
                .map(|(i, _)| i)
                .max()
                .unwrap_or(0);

            if max_symbol == 0 && counts[0] == 0 {
                // Empty context - use single-symbol distribution
                distributions.push(AnsDistribution::flat(1)?);
            } else {
                // Create distribution from frequencies
                let freqs: Vec<u32> = counts[..=max_symbol].to_vec();
                distributions.push(AnsDistribution::from_frequencies(&freqs)?);
            }
        }

        Ok(distributions)
    }

    /// Get total count for a context.
    pub fn context_count(&self, ctx: usize) -> u32 {
        if ctx < self.num_contexts {
            self.counts[ctx].iter().sum()
        } else {
            0
        }
    }

    /// Get alphabet size (max symbol + 1) for a context.
    pub fn alphabet_size(&self, ctx: usize) -> usize {
        if ctx < self.num_contexts {
            self.counts[ctx]
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0)
                .map(|(i, _)| i + 1)
                .max()
                .unwrap_or(1)
        } else {
            1
        }
    }

    /// Convert per-context counts to entropy_coding Histograms for clustering.
    ///
    /// This creates a vector of `EntropyHistogram` suitable for use with
    /// the histogram clustering infrastructure in `entropy_coding::cluster`.
    pub fn to_entropy_histograms(&self) -> Vec<EntropyHistogram> {
        self.counts
            .iter()
            .map(|ctx_counts| {
                let counts: Vec<i32> = ctx_counts.iter().map(|&c| c as i32).collect();
                EntropyHistogram::from_counts(&counts)
            })
            .collect()
    }
}

/// Simplified histogram for single-group encoding.
/// Uses a flat context map (identity mapping).
pub struct SimpleHistogramSet {
    /// Distributions per context.
    pub distributions: Vec<AnsDistribution>,
    /// Context map (identity for now).
    pub context_map: Vec<usize>,
}

impl SimpleHistogramSet {
    /// Build from collected tokens.
    pub fn from_tokens(tokens: &[Token], bcm: &BlockContextMap) -> Result<Self> {
        let num_contexts = bcm.num_ac_contexts();
        let mut builder = HistogramBuilder::new(num_contexts);
        builder.add_tokens(tokens);

        let distributions = builder.build_distributions()?;
        let context_map = (0..num_contexts).collect();

        Ok(Self {
            distributions,
            context_map,
        })
    }

    /// Get the number of contexts.
    pub fn num_contexts(&self) -> usize {
        self.distributions.len()
    }
}

/// Histogram set with clustering applied.
///
/// Uses the clustering infrastructure to group similar contexts together,
/// reducing the number of histograms that need to be encoded while maintaining
/// good compression efficiency.
pub struct ClusteredHistogramSet {
    /// Clustered histograms (typically 4-256, much fewer than num_contexts).
    pub clustered_histograms: Vec<EntropyHistogram>,
    /// ANS distributions for encoding (one per cluster).
    pub distributions: Vec<AnsDistribution>,
    /// Context map: context_index → cluster_index.
    pub context_map: Vec<u8>,
    /// Number of original contexts.
    pub num_contexts: usize,
    /// Global alphabet size (after HybridUint encoding) for consistent encoding.
    /// This MUST be used when writing tokens, not recomputed from group tokens.
    pub global_alphabet_size: usize,
}

impl ClusteredHistogramSet {
    /// Build clustered histogram set from tokens.
    ///
    /// # Arguments
    /// * `tokens` - The tokens to build histograms from
    /// * `bcm` - Block context map providing context count
    /// * `clustering_type` - Controls compression/speed trade-off
    pub fn from_tokens(
        tokens: &[Token],
        bcm: &BlockContextMap,
        clustering_type: ClusteringType,
    ) -> Result<Self> {
        let num_contexts = bcm.num_ac_contexts();

        // 1. Build per-context counts
        let mut builder = HistogramBuilder::new(num_contexts);
        builder.add_tokens(tokens);

        // 2. Convert to entropy histograms
        let histograms = builder.to_entropy_histograms();

        // 3. Cluster based on clustering type
        // With ANS+MTF context map encoding implemented, we can now use multiple clusters
        // without the 22KB overhead that simple encoding had.
        // The encoder automatically picks the best encoding strategy (simple, Huffman, or MTF+Huffman).
        const MAX_CLUSTERS: usize = 8; // Typical good value balancing quality vs header size
        let max_clusters = match clustering_type {
            ClusteringType::Fastest => 1, // Single cluster for speed
            ClusteringType::Fast => 4,    // Moderate clustering
            ClusteringType::Best => 8,    // Full clustering for quality
        };
        let max_clusters = max_clusters.min(MAX_CLUSTERS);
        // Use ANS cost model since VarDCT uses ANS entropy coding
        let cluster_result =
            cluster_histograms(clustering_type, EntropyType::Ans, &histograms, max_clusters)?;

        // 4. Build distributions from clustered histograms
        let distributions = cluster_result
            .histograms
            .iter()
            .map(|h| {
                let freqs: Vec<u32> = h.counts.iter().map(|&c| c.max(0) as u32).collect();
                if freqs.iter().all(|&f| f == 0) {
                    AnsDistribution::flat(1)
                } else {
                    AnsDistribution::from_frequencies(&freqs)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // 5. Build context map (u8 since we have at most 256 clusters)
        let context_map: Vec<u8> = cluster_result.symbols.iter().map(|&s| s as u8).collect();

        // 6. Compute global alphabet size from tokens
        // This MUST be used consistently when writing histograms and tokens
        let global_alphabet_size = compute_alphabet_size_from_tokens(tokens);

        Ok(Self {
            clustered_histograms: cluster_result.histograms,
            distributions,
            context_map,
            num_contexts,
            global_alphabet_size,
        })
    }

    /// Get the number of clusters.
    pub fn num_clusters(&self) -> usize {
        self.distributions.len()
    }

    /// Get the global maximum alphabet size across all clusters.
    pub fn max_alphabet_size(&self) -> usize {
        self.clustered_histograms
            .iter()
            .map(|h| h.alphabet_size())
            .max()
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_builder_empty() {
        let builder = HistogramBuilder::new(4);
        assert_eq!(builder.context_count(0), 0);
        assert_eq!(builder.alphabet_size(0), 1);
    }

    #[test]
    fn test_histogram_builder_add_tokens() {
        let mut builder = HistogramBuilder::new(4);

        let tokens = vec![
            Token::new(0, 5),
            Token::new(0, 10),
            Token::new(0, 5),
            Token::new(1, 3),
        ];

        builder.add_tokens(&tokens);

        assert_eq!(builder.context_count(0), 3);
        assert_eq!(builder.context_count(1), 1);
        assert_eq!(builder.alphabet_size(0), 11); // max symbol 10 + 1
        assert_eq!(builder.alphabet_size(1), 4); // max symbol 3 + 1
    }

    #[test]
    fn test_build_distributions() {
        let mut builder = HistogramBuilder::new(2);

        let tokens = vec![
            Token::new(0, 0),
            Token::new(0, 0),
            Token::new(0, 1),
            Token::new(1, 5),
        ];

        builder.add_tokens(&tokens);

        let dists = builder.build_distributions().unwrap();
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0].alphabet_size(), 2);
        assert_eq!(dists[1].alphabet_size(), 6);
    }
}
