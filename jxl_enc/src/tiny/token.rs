// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Token types and uint encoding for entropy coding.
//!
//! These are ported from libjxl-tiny and will be used for entropy coding.

#![allow(dead_code)]

use super::common::floor_log2_nonzero;

/// A token to be entropy coded.
/// Consists of a context and a value.
#[derive(Debug, Clone, Copy, Default)]
pub struct Token {
    pub context: u32,
    pub value: u32,
}

impl Token {
    /// Create a new token with the given context and value.
    #[inline]
    pub const fn new(context: u32, value: u32) -> Self {
        Self { context, value }
    }
}

/// Result of encoding a uint value.
#[derive(Debug, Clone, Copy)]
pub struct EncodedUint {
    /// The token (symbol) to encode with Huffman.
    pub token: u32,
    /// Number of extra bits.
    pub nbits: u32,
    /// The extra bits value.
    pub bits: u32,
}

/// Uint coder for entropy coding.
///
/// Encoding scheme:
/// - N = 0-15: token=N, nbits=0
/// - N = 16 (10000): token=16, nbits=2, bits=00
/// - N = 17 (10001): token=16, nbits=2, bits=01
/// - N = 20 (10100): token=17, nbits=2, bits=00
/// - ...up to N=65535: token=63, nbits=13
pub struct UintCoder;

impl UintCoder {
    /// Encode a uint value into a token and extra bits.
    #[inline]
    pub fn encode(value: u32) -> EncodedUint {
        if value < 16 {
            EncodedUint {
                token: value,
                nbits: 0,
                bits: 0,
            }
        } else {
            let n = floor_log2_nonzero(value);
            let m = value - (1 << n);
            let token = (n << 2) + (m >> (n - 2));
            let nbits = n - 2;
            let bits = value & ((1u32 << nbits) - 1);
            EncodedUint { token, nbits, bits }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uint_coder_small_values() {
        for i in 0..16 {
            let encoded = UintCoder::encode(i);
            assert_eq!(encoded.token, i);
            assert_eq!(encoded.nbits, 0);
            assert_eq!(encoded.bits, 0);
        }
    }

    #[test]
    fn test_uint_coder_values() {
        // N = 16 (10000): token=16, nbits=2, bits=00
        let e = UintCoder::encode(16);
        assert_eq!(e.token, 16);
        assert_eq!(e.nbits, 2);
        assert_eq!(e.bits, 0);

        // N = 17 (10001): token=16, nbits=2, bits=01
        let e = UintCoder::encode(17);
        assert_eq!(e.token, 16);
        assert_eq!(e.nbits, 2);
        assert_eq!(e.bits, 1);

        // N = 20 (10100): token=17, nbits=2, bits=00
        let e = UintCoder::encode(20);
        assert_eq!(e.token, 17);
        assert_eq!(e.nbits, 2);
        assert_eq!(e.bits, 0);

        // N = 24 (11000): token=18, nbits=2, bits=00
        let e = UintCoder::encode(24);
        assert_eq!(e.token, 18);
        assert_eq!(e.nbits, 2);
        assert_eq!(e.bits, 0);

        // N = 28 (11100): token=19, nbits=2, bits=00
        let e = UintCoder::encode(28);
        assert_eq!(e.token, 19);
        assert_eq!(e.nbits, 2);
        assert_eq!(e.bits, 0);

        // N = 32 (100000): token=20, nbits=3, bits=000
        let e = UintCoder::encode(32);
        assert_eq!(e.token, 20);
        assert_eq!(e.nbits, 3);
        assert_eq!(e.bits, 0);
    }

    #[test]
    fn test_uint_coder_large_value() {
        // N = 65535: token=63, nbits=13
        let e = UintCoder::encode(65535);
        assert_eq!(e.token, 63);
        assert_eq!(e.nbits, 13);
        // Verify we can reconstruct the value
        let reconstructed = (1u32 << 15) + ((e.token & 3) << 13) + e.bits;
        assert_eq!(reconstructed, 65535);
    }
}
