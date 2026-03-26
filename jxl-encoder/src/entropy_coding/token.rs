// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Token types and uint encoding for entropy coding.
//!
//! These are ported from libjxl-tiny and will be used for entropy coding.

#![allow(dead_code)]

use crate::vardct::common::floor_log2_nonzero;

/// Bit flag for LZ77 length tokens, packed into bit 31 of context_and_flags.
const LZ77_FLAG: u32 = 1 << 31;
/// Mask to extract the context from context_and_flags (all bits except bit 31).
const CONTEXT_MASK: u32 = !LZ77_FLAG;

/// A token to be entropy coded (8 bytes).
///
/// Consists of a context and a value. The LZ77 length flag is packed into bit 31
/// of the context field, keeping the struct at 8 bytes instead of 12.
#[derive(Debug, Clone, Copy, Default)]
pub struct Token {
    /// Bits 0-30: context index. Bit 31: is_lz77_length flag.
    context_and_flags: u32,
    pub value: u32,
}

impl Token {
    /// Create a new token with the given context and value.
    #[inline]
    pub const fn new(context: u32, value: u32) -> Self {
        Self {
            context_and_flags: context,
            value,
        }
    }

    /// Create a new LZ77 length token.
    #[inline]
    pub const fn lz77_length(context: u32, value: u32) -> Self {
        Self {
            context_and_flags: context | LZ77_FLAG,
            value,
        }
    }

    /// Get the context index (bits 0-30).
    #[inline]
    pub const fn context(&self) -> u32 {
        self.context_and_flags & CONTEXT_MASK
    }

    /// Returns true if this token encodes an LZ77 length value.
    #[inline]
    pub const fn is_lz77_length(&self) -> bool {
        (self.context_and_flags & LZ77_FLAG) != 0
    }

    /// Set the context index, preserving the LZ77 flag.
    #[inline]
    pub fn set_context(&mut self, context: u32) {
        self.context_and_flags = (self.context_and_flags & LZ77_FLAG) | (context & CONTEXT_MASK);
    }

    /// Set or clear the LZ77 length flag, preserving the context.
    #[inline]
    pub fn set_lz77_length(&mut self, is_lz77: bool) {
        if is_lz77 {
            self.context_and_flags |= LZ77_FLAG;
        } else {
            self.context_and_flags &= CONTEXT_MASK;
        }
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

impl From<EncodedUint> for (u32, u32) {
    fn from(e: EncodedUint) -> (u32, u32) {
        (e.token, e.nbits)
    }
}

/// Uint coder for entropy coding.
///
/// Encoding scheme (from libjxl-tiny):
/// - N = 0-15: token=N, nbits=0 (direct values)
/// - N >= 16: token = (n << 2) + (m >> (n - 2)), nbits = n - 2
///   where n = floor_log2(N), m = N - (1 << n)
///
/// Examples:
/// - N = 16 (10000):      (token=16, nbits=2, bits='00')
/// - N = 17 (10001):      (token=16, nbits=2, bits='01')
/// - N = 20 (10100):      (token=17, nbits=2, bits='00')
/// - N = 24 (11000):      (token=18, nbits=2, bits='00')
/// - N = 28 (11100):      (token=19, nbits=2, bits='00')
/// - N = 32 (100000):     (token=20, nbits=3, bits='000')
/// - N = 65535:           (token=63, nbits=13, bits='1111111111111')
pub struct UintCoder;

impl UintCoder {
    /// Encode a uint value into a token and extra bits.
    #[inline]
    pub fn encode(value: u32) -> EncodedUint {
        if value < 16 {
            // Direct encoding for small values
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

/// LZ77 uint coder using HybridUintConfig(0, 0, 0).
///
/// With split_exponent=0, split_token=1:
/// - value 0: token=0, nbits=0, bits=0
/// - value >= 1: token = 1 + floor_log2(value), nbits = floor_log2(value),
///   bits = value - (1 << floor_log2(value))
pub struct Lz77UintCoder;

impl Lz77UintCoder {
    /// Encode a value using HybridUintConfig(0, 0, 0).
    #[inline]
    pub fn encode(value: u32) -> EncodedUint {
        if value == 0 {
            EncodedUint {
                token: 0,
                nbits: 0,
                bits: 0,
            }
        } else {
            let n = floor_log2_nonzero(value);
            let m = value - (1 << n);
            // split_token=1, split_exponent=0, msb_in_token=0, lsb_in_token=0
            // token = split_token + ((n - split_exponent) << (msb + lsb)) + ...
            // = 1 + (n - 0) << 0 + (m >> (n - 0)) << 0 + (m & 0)
            // = 1 + n   (since msb=lsb=0, the m terms contribute nothing to token)
            let token = 1 + n;
            // nbits = n - msb - lsb = n
            let nbits = n;
            // bits = (value >> lsb) & ((1 << nbits) - 1) = value & ((1 << n) - 1) = m
            let bits = m;
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
        // libjxl-tiny encoding: token = (n << 2) + (m >> (n - 2))
        // N = 16: n=4, m=0, token=(4<<2)+(0>>2)=16, nbits=2, bits=0
        let e = UintCoder::encode(16);
        assert_eq!(e.token, 16, "token for 16");
        assert_eq!(e.nbits, 2, "nbits for 16");
        assert_eq!(e.bits, 0, "bits for 16");

        // N = 17: n=4, m=1, token=(4<<2)+(1>>2)=16, nbits=2, bits=1
        let e = UintCoder::encode(17);
        assert_eq!(e.token, 16, "token for 17");
        assert_eq!(e.nbits, 2, "nbits for 17");
        assert_eq!(e.bits, 1, "bits for 17");

        // N = 20: n=4, m=4, token=(4<<2)+(4>>2)=17, nbits=2, bits=0
        let e = UintCoder::encode(20);
        assert_eq!(e.token, 17, "token for 20");
        assert_eq!(e.nbits, 2, "nbits for 20");
        assert_eq!(e.bits, 0, "bits for 20");

        // N = 24: n=4, m=8, token=(4<<2)+(8>>2)=18, nbits=2, bits=0
        let e = UintCoder::encode(24);
        assert_eq!(e.token, 18, "token for 24");
        assert_eq!(e.nbits, 2, "nbits for 24");
        assert_eq!(e.bits, 0, "bits for 24");

        // N = 28: n=4, m=12, token=(4<<2)+(12>>2)=19, nbits=2, bits=0
        let e = UintCoder::encode(28);
        assert_eq!(e.token, 19, "token for 28");
        assert_eq!(e.nbits, 2, "nbits for 28");
        assert_eq!(e.bits, 0, "bits for 28");

        // N = 32: n=5, m=0, token=(5<<2)+(0>>3)=20, nbits=3, bits=0
        let e = UintCoder::encode(32);
        assert_eq!(e.token, 20, "token for 32");
        assert_eq!(e.nbits, 3, "nbits for 32");
        assert_eq!(e.bits, 0, "bits for 32");
    }

    #[test]
    fn test_uint_coder_large_value() {
        // N = 65535: n=15, m=32767
        // token = (15<<2) + (32767>>13) = 60 + 3 = 63
        // nbits = 15 - 2 = 13
        // bits = 65535 & 8191 = 8191
        let e = UintCoder::encode(65535);
        assert_eq!(e.token, 63, "token for 65535");
        assert_eq!(e.nbits, 13, "nbits for 65535");
        assert_eq!(e.bits, 8191, "bits for 65535");
        // Verify we can reconstruct the value
        // exponent n = token >> 2 = 15
        // top 2 bits of m = token & 3 = 3
        // m = (3 << 13) + bits = 24576 + 8191 = 32767
        // value = (1 << 15) + 32767 = 65535
        let n = e.token >> 2;
        let top_bits = e.token & 3;
        let m = (top_bits << e.nbits) + e.bits;
        let reconstructed = (1u32 << n) + m;
        assert_eq!(reconstructed, 65535);
    }

    #[test]
    fn test_token_size() {
        assert_eq!(core::mem::size_of::<Token>(), 8);
    }

    #[test]
    fn test_token_packed_fields() {
        let t = Token::new(42, 100);
        assert_eq!(t.context(), 42);
        assert_eq!(t.value, 100);
        assert!(!t.is_lz77_length());

        let t = Token::lz77_length(42, 100);
        assert_eq!(t.context(), 42);
        assert_eq!(t.value, 100);
        assert!(t.is_lz77_length());

        let mut t = Token::new(10, 200);
        t.set_context(99);
        assert_eq!(t.context(), 99);
        assert!(!t.is_lz77_length());

        t.set_lz77_length(true);
        assert!(t.is_lz77_length());
        assert_eq!(t.context(), 99);

        t.set_context(77);
        assert_eq!(t.context(), 77);
        assert!(t.is_lz77_length());

        t.set_lz77_length(false);
        assert!(!t.is_lz77_length());
        assert_eq!(t.context(), 77);
    }

    #[test]
    fn test_lz77_uint_coder() {
        // value 0: token=0, nbits=0
        let e = Lz77UintCoder::encode(0);
        assert_eq!(e.token, 0);
        assert_eq!(e.nbits, 0);
        assert_eq!(e.bits, 0);

        // value 1: n=0, token=1+0=1, nbits=0, bits=0
        let e = Lz77UintCoder::encode(1);
        assert_eq!(e.token, 1);
        assert_eq!(e.nbits, 0);
        assert_eq!(e.bits, 0);

        // value 2: n=1, m=0, token=2, nbits=1, bits=0
        let e = Lz77UintCoder::encode(2);
        assert_eq!(e.token, 2);
        assert_eq!(e.nbits, 1);
        assert_eq!(e.bits, 0);

        // value 3: n=1, m=1, token=2, nbits=1, bits=1
        let e = Lz77UintCoder::encode(3);
        assert_eq!(e.token, 2);
        assert_eq!(e.nbits, 1);
        assert_eq!(e.bits, 1);

        // value 7: n=2, m=3, token=3, nbits=2, bits=3
        let e = Lz77UintCoder::encode(7);
        assert_eq!(e.token, 3);
        assert_eq!(e.nbits, 2);
        assert_eq!(e.bits, 3);

        // Verify roundtrip reconstruction for a range of values
        for v in 0..1000 {
            let e = Lz77UintCoder::encode(v);
            let reconstructed = if e.token == 0 {
                0
            } else {
                let n = e.token - 1;
                (1u32 << n) + e.bits
            };
            assert_eq!(reconstructed, v, "roundtrip failed for value {}", v);
        }
    }
}
