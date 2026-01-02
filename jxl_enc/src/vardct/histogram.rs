//! Histogram building for VarDCT coefficient encoding.
//!
//! Collects token statistics and builds histograms for entropy coding.

use super::context::BlockContextMap;
use super::tokenize::Token;
use crate::entropy_coding::ans::{ANS_MAX_ALPHABET_SIZE, AnsDistribution};
use crate::error::Result;

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
