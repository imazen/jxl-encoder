// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

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
    ///
    /// The decoder reconstructs: value = ((token_high << n) | rest_bits) << lsb_in_token | low_bits
    /// where token_high = (token_shift | (1 << msb_in_token))
    pub fn encode(&self, value: u32) -> (u32, u32, u32) {
        if value < self.split {
            // Direct coding: token = value, no extra bits
            return (value, 0, 0);
        }

        // For values >= split, we need to encode using the hybrid format.
        // The decoder does:
        //   n = split_exponent - (msb_in_token + lsb_in_token) + ((token - split) >> (msb_in_token + lsb_in_token))
        //   low_bits = token & ((1 << lsb_in_token) - 1)
        //   token_shift = (token >> lsb_in_token) & ((1 << msb_in_token) - 1)
        //   token_high = token_shift | (1 << msb_in_token)
        //   result = ((token_high << n) | rest_bits) << lsb_in_token | low_bits

        let l = self.lsb_in_token;
        let m = self.msb_in_token;
        let s = self.split_exponent;

        // Extract the low bits (lsb_in_token bits from value)
        let low_bits = value & ((1 << l) - 1);

        // The remaining value after removing low bits
        let value_shifted = value >> l;

        // The decoder reconstructs: value_shifted = (token_high << n) | rest_bits
        // where:
        //   n = base_n + n_extra = (s - m - l) + ((token - split) >> (m + l))
        //   token_high = (token_shift | (1 << m)), which has (m+1) bits
        //   rest_bits has n bits
        //
        // So value_shifted has (m+1+n) bits total.
        // Given value_shifted, we compute n = value_bits - m - 1, clamped to >= base_n.

        // The number of bits in value_shifted (excluding leading zeros)
        let value_bits = if value_shifted == 0 {
            0
        } else {
            32 - value_shifted.leading_zeros()
        };

        // base_n is the minimum n (when n_extra = 0)
        let base_n = s - (m + l);

        // n = value_bits - (m + 1), but at least base_n
        let n = if value_bits > m + 1 {
            (value_bits - m - 1).max(base_n)
        } else {
            base_n
        };

        // n_extra is how many extra bits beyond base_n
        let n_extra = n - base_n;

        // token_high = value_shifted >> n (this should have the implicit leading 1)
        let token_high = if value_shifted > 0 {
            value_shifted >> n
        } else {
            // Edge case: value_shifted = 0, use minimum token_high
            1 << m
        };

        // Ensure token_high has the implicit leading 1 set
        // If value_shifted is very small, token_high might be 0
        let token_high = token_high.max(1 << m);

        // token_shift = token_high with the implicit leading 1 stripped
        let token_shift = token_high & ((1 << m) - 1);

        // rest_bits = the bits of value_shifted not captured in token_high
        let rest_bits = value_shifted & ((1 << n) - 1);

        // Token bucket offset based on n_extra
        // From decoder: n = base_n + ((token - split) >> (m + l))
        // So ((token - split) >> (m + l)) = n_extra
        // token = split + (n_extra << (m + l)) + (token_shift << l) + low_bits
        let bucket_offset = n_extra << (m + l);
        let token = self.split + bucket_offset + (token_shift << l) + low_bits;

        // The extra bits to write are rest_bits with n bits
        (token, rest_bits, n)
    }

    /// Writes a HybridUint value, given an entropy coder for the token.
    #[allow(dead_code)]
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

    #[test]
    fn test_default_config() {
        let config = HybridUintConfig::default();
        assert_eq!(config.split_exponent, 4);
        assert_eq!(config.msb_in_token, 2);
        assert_eq!(config.lsb_in_token, 0);
        assert_eq!(config.split, 16);
    }

    #[test]
    fn test_split_exponent_zero() {
        // Special case: split_exponent = 0 means only value 0 is direct coded
        let config = HybridUintConfig::new(0, 0, 0);

        let (token, extra, num_extra) = config.encode(0);
        assert_eq!(token, 0);
        assert_eq!(extra, 0);
        assert_eq!(num_extra, 0);

        // Value 1 needs hybrid encoding
        let (token, _extra, _num_extra) = config.encode(1);
        assert!(token >= 1);
    }

    #[test]
    fn test_write_method() {
        let config = HybridUintConfig::new(4, 2, 0);
        let mut writer = BitWriter::new();

        // Test writing with a simple token writer
        config
            .write(5, &mut writer, |w, token| {
                w.write(8, token as u64)?;
                Ok(())
            })
            .unwrap();

        // Should have written 8 bits for token (no extra bits for value < 16)
        assert_eq!(writer.bits_written(), 8);
    }

    #[test]
    fn test_write_with_extra_bits() {
        let config = HybridUintConfig::new(4, 2, 0);
        let mut writer = BitWriter::new();

        // Test writing a value that requires extra bits
        config
            .write(100, &mut writer, |w, token| {
                w.write(8, token as u64)?;
                Ok(())
            })
            .unwrap();

        // Should have written 8 bits for token + extra bits
        assert!(writer.bits_written() > 8);
    }

    #[test]
    fn test_various_configs() {
        // Test different configurations
        let configs = [
            HybridUintConfig::new(0, 0, 0),
            HybridUintConfig::new(4, 0, 0),
            HybridUintConfig::new(4, 2, 0),
            HybridUintConfig::new(4, 2, 2),
            HybridUintConfig::new(8, 4, 0),
        ];

        for config in configs {
            // Encoding should work for all values
            for value in [0, 1, 15, 16, 100, 1000, 10000] {
                let (token, extra, num_extra) = config.encode(value);
                // Token should be reasonable
                assert!(
                    token < 10000,
                    "token {} unreasonable for value {}",
                    token,
                    value
                );
                // Extra bits should be bounded
                assert!(num_extra <= 32, "too many extra bits");
                // Extra should fit in num_extra bits
                if num_extra > 0 {
                    assert!(extra < (1 << num_extra));
                }
            }
        }
    }

    #[test]
    fn test_lsb_in_token() {
        // Test with lsb_in_token > 0
        let config = HybridUintConfig::new(4, 2, 2);

        let (token, _extra, _num_extra) = config.encode(32);
        // With LSB encoding, token should be different
        assert!(token > 0);
    }
}
