// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! HybridUint encoding for JPEG XL.
//!
//! HybridUint is a variable-length integer encoding used throughout JXL.
//! It splits integers into a "token" (entropy-coded) and extra bits (raw).

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Configuration for HybridUint encoding.
#[derive(Debug, Clone, Copy)]
pub struct HybridUintConfig {
    /// Number of direct-coded values (token = value for values < split_exponent).
    pub split_exponent: u32,
    /// Number of extra bits to include in the token for larger values.
    pub split: u32,
    /// MSB (most significant bits) configuration.
    pub msb_in_token: u32,
    /// LSB (least significant bits) configuration.
    pub lsb_in_token: u32,
}

impl HybridUintConfig {
    /// Creates a new HybridUint configuration.
    pub fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        Self {
            split_exponent,
            split: 1 << split_exponent,
            msb_in_token,
            lsb_in_token,
        }
    }

    /// Default configuration used for most JXL contexts.
    pub fn default_config() -> Self {
        Self::new(4, 2, 0)
    }

    /// Encodes a value and returns (token, extra_bits, num_extra_bits).
    ///
    /// The token should be entropy-coded, while extra_bits are written raw.
    pub fn encode(&self, value: u32) -> (u32, u32, u32) {
        if value < self.split {
            // Direct coding: token = value, no extra bits
            (value, 0, 0)
        } else {
            // Compute the number of bits needed for the high part
            let high_bits = value >> self.split_exponent;
            if high_bits == 0 {
                // Edge case: value is exactly split
                return (self.split, 0, 0);
            }

            let n = 32 - high_bits.leading_zeros();
            let token_base = self.split + (n - 1) * (self.msb_in_token + self.lsb_in_token + 1);

            // Extract MSB and LSB portions for token
            let low_bits = value & ((1 << self.split_exponent) - 1);

            // Compute MSB portion (top msb_in_token bits of high_bits)
            let msb = if n > self.msb_in_token {
                (high_bits >> (n - 1 - self.msb_in_token)) & ((1 << self.msb_in_token) - 1)
            } else {
                high_bits & ((1 << self.msb_in_token) - 1)
            };

            // LSB portion from low_bits
            let lsb = low_bits & ((1 << self.lsb_in_token) - 1);

            let token = token_base + msb * (self.lsb_in_token + 1) + lsb;

            // Extra bits: remaining bits not encoded in token
            let extra_bits_count = if n > self.msb_in_token + 1 {
                n - 1 - self.msb_in_token + self.split_exponent - self.lsb_in_token
            } else {
                self.split_exponent.saturating_sub(self.lsb_in_token)
            };

            let extra_bits = if extra_bits_count > 0 {
                (value >> self.lsb_in_token) & ((1 << extra_bits_count) - 1)
            } else {
                0
            };

            (token, extra_bits, extra_bits_count)
        }
    }

    /// Writes a HybridUint value, given an entropy coder for the token.
    pub fn write<F>(&self, value: u32, writer: &mut BitWriter, mut write_token: F) -> Result<()>
    where
        F: FnMut(&mut BitWriter, u32) -> Result<()>,
    {
        let (token, extra_bits, num_extra) = self.encode(value);
        write_token(writer, token)?;
        if num_extra > 0 {
            writer.write(num_extra as usize, extra_bits as u64)?;
        }
        Ok(())
    }
}

impl Default for HybridUintConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_coding() {
        let config = HybridUintConfig::new(4, 2, 0);

        // Values 0-15 should be direct-coded
        for i in 0..16 {
            let (token, extra, num_extra) = config.encode(i);
            assert_eq!(token, i);
            assert_eq!(extra, 0);
            assert_eq!(num_extra, 0);
        }
    }

    #[test]
    fn test_larger_values() {
        let config = HybridUintConfig::new(4, 2, 0);

        // Value >= 16 should use hybrid encoding
        let (token, _extra, _num_extra) = config.encode(16);
        assert!(token >= 16, "token {} should be >= 16", token);

        // Test a larger value
        let (token2, _extra2, _num_extra2) = config.encode(100);
        assert!(token2 > 0, "token should be positive for value 100");
    }
}
