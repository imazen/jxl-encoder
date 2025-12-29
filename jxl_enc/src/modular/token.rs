// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Token types for modular encoding.
//!
//! Tokens are the intermediate representation between prediction residuals
//! and entropy-coded bitstream.

/// A token representing an encoded value.
///
/// Each token stores:
/// - A context ID (determines which entropy code to use)
/// - A value (the prediction residual, packed as unsigned)
#[derive(Debug, Clone, Copy)]
pub struct Token {
    /// Context ID for entropy coding.
    pub context: u32,
    /// The packed value (unsigned, zig-zag encoded).
    pub value: u32,
}

impl Token {
    /// Creates a new token.
    #[inline]
    pub fn new(context: u32, value: u32) -> Self {
        Self { context, value }
    }
}

/// A collection of tokens for encoding.
#[derive(Debug, Clone, Default)]
pub struct TokenList {
    /// The tokens.
    tokens: Vec<Token>,
    /// Histogram of values per context (for entropy coder setup).
    histograms: Vec<Histogram>,
}

impl TokenList {
    /// Creates a new empty token list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a token list with a specific number of contexts.
    pub fn with_contexts(num_contexts: usize) -> Self {
        Self {
            tokens: Vec::new(),
            histograms: vec![Histogram::new(); num_contexts],
        }
    }

    /// Adds a token to the list.
    #[inline]
    pub fn push(&mut self, token: Token) {
        // Update histogram
        let ctx = token.context as usize;
        if ctx >= self.histograms.len() {
            self.histograms.resize(ctx + 1, Histogram::new());
        }
        self.histograms[ctx].add(token.value);

        self.tokens.push(token);
    }

    /// Returns the tokens.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Returns the number of tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the histograms.
    pub fn histograms(&self) -> &[Histogram] {
        &self.histograms
    }

    /// Returns the number of contexts used.
    pub fn num_contexts(&self) -> usize {
        self.histograms.len()
    }
}

/// Histogram of values for entropy coder setup.
#[derive(Debug, Clone, Default)]
pub struct Histogram {
    /// Counts per value (sparse representation).
    counts: Vec<(u32, u32)>, // (value, count)
    /// Total count.
    total: u64,
}

impl Histogram {
    /// Creates a new empty histogram.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a value to the histogram.
    #[inline]
    pub fn add(&mut self, value: u32) {
        // Simple linear search for now - optimize later with hash map if needed
        for &mut (v, ref mut c) in &mut self.counts {
            if v == value {
                *c += 1;
                self.total += 1;
                return;
            }
        }
        // Not found, add new entry
        self.counts.push((value, 1));
        self.total += 1;
    }

    /// Returns the total count.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Returns the number of distinct values.
    pub fn num_symbols(&self) -> usize {
        self.counts.len()
    }

    /// Returns the counts as a dense array up to max_value.
    pub fn to_dense(&self, max_value: u32) -> Vec<u32> {
        let mut dense = vec![0u32; max_value as usize + 1];
        for &(value, count) in &self.counts {
            if value <= max_value {
                dense[value as usize] = count;
            }
        }
        dense
    }

    /// Returns an iterator over (value, count) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.counts.iter().copied()
    }

    /// Merges another histogram into this one.
    pub fn merge(&mut self, other: &Histogram) {
        for &(value, count) in &other.counts {
            for &mut (v, ref mut c) in &mut self.counts {
                if v == value {
                    *c += count;
                    self.total += count as u64;
                    continue;
                }
            }
            self.counts.push((value, count));
            self.total += count as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_list() {
        let mut list = TokenList::with_contexts(2);

        list.push(Token::new(0, 5));
        list.push(Token::new(0, 5));
        list.push(Token::new(1, 10));

        assert_eq!(list.len(), 3);
        assert_eq!(list.num_contexts(), 2);

        // Check histograms
        assert_eq!(list.histograms()[0].total(), 2);
        assert_eq!(list.histograms()[1].total(), 1);
    }

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::new();

        hist.add(0);
        hist.add(1);
        hist.add(1);
        hist.add(5);

        assert_eq!(hist.total(), 4);
        assert_eq!(hist.num_symbols(), 3);

        let dense = hist.to_dense(10);
        assert_eq!(dense[0], 1);
        assert_eq!(dense[1], 2);
        assert_eq!(dense[5], 1);
    }
}
