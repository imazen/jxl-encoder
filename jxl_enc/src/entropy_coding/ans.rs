// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! ANS (Asymmetric Numeral Systems) encoder.
//!
//! ANS is an entropy coding method used in JPEG XL for efficient symbol
//! compression. This module implements the rANS (range ANS) variant.

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// ANS encoder state and configuration.
#[allow(dead_code)] // TODO: Implement ANS encoding
pub struct AnsEncoder {
    /// ANS state (normalized to [L, 2L) range).
    state: u32,
    /// Output buffer for encoded bits (reversed).
    output: Vec<u32>,
}

impl AnsEncoder {
    /// Creates a new ANS encoder.
    pub fn new() -> Self {
        Self {
            state: 1 << 16, // Initial state L
            output: Vec::new(),
        }
    }

    /// Encodes a single symbol using the given distribution.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The symbol to encode.
    /// * `freq` - Frequency of the symbol (probability numerator).
    /// * `cumul` - Cumulative frequency of symbols before this one.
    /// * `log_alpha_size` - Log2 of the total frequency sum.
    pub fn encode_symbol(
        &mut self,
        _symbol: u32,
        _freq: u32,
        _cumul: u32,
        _log_alpha_size: u32,
    ) -> Result<()> {
        // TODO: Implement ANS encoding
        // This requires:
        // 1. State renormalization (output bits when state too large)
        // 2. State update: state = (state / freq) * M + cumul + (state % freq)
        todo!("ANS encoding not yet implemented")
    }

    /// Finalizes encoding and writes to a BitWriter.
    pub fn finalize(self, _writer: &mut BitWriter) -> Result<()> {
        // TODO: Write final state and reversed output
        todo!("ANS finalization not yet implemented")
    }
}

impl Default for AnsEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds an ANS distribution from symbol frequencies.
pub struct AnsDistributionBuilder {
    /// Symbol frequencies.
    freqs: Vec<u32>,
    /// Log2 of the distribution size.
    log_alpha_size: u32,
}

impl AnsDistributionBuilder {
    /// Creates a new distribution builder for the given alphabet size.
    pub fn new(alphabet_size: usize) -> Self {
        Self {
            freqs: vec![0; alphabet_size],
            log_alpha_size: 12, // Default JXL distribution size
        }
    }

    /// Adds a symbol occurrence.
    pub fn add_symbol(&mut self, symbol: usize) {
        if symbol < self.freqs.len() {
            self.freqs[symbol] += 1;
        }
    }

    /// Builds the distribution, normalizing frequencies to sum to 2^log_alpha_size.
    pub fn build(self) -> AnsDistribution {
        // TODO: Implement frequency normalization
        AnsDistribution {
            freqs: self.freqs,
            cumul: Vec::new(),
            log_alpha_size: self.log_alpha_size,
        }
    }
}

/// A finalized ANS distribution for encoding.
pub struct AnsDistribution {
    /// Normalized frequencies.
    pub freqs: Vec<u32>,
    /// Cumulative frequencies.
    pub cumul: Vec<u32>,
    /// Log2 of total frequency sum.
    pub log_alpha_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_builder() {
        let mut builder = AnsDistributionBuilder::new(256);
        builder.add_symbol(0);
        builder.add_symbol(0);
        builder.add_symbol(1);

        let dist = builder.build();
        assert_eq!(dist.freqs[0], 2);
        assert_eq!(dist.freqs[1], 1);
    }
}
