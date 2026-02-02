// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! AC coefficient context computation for entropy coding.
//!
//! These functions and constants are ported from libjxl-tiny and will be used
//! when the AC group encoding is implemented.

#![allow(dead_code)]

/// Number of predicted nonzeros buckets (0 to 36 inclusive = 37 values).
pub const NON_ZERO_BUCKETS: usize = 37;

/// Number of AC strategy codes.
pub const NUM_AC_STRATEGY_CODES: usize = 27;

/// Number of block contexts.
pub const NUM_BLOCK_CTXS: usize = 4;

/// Supremum of ZeroDensityContext + 1 when x + y < 64.
pub const ZERO_DENSITY_CONTEXT_COUNT: usize = 458;

/// Supremum of ZeroDensityContext + 1 (all cases).
#[allow(dead_code)]
pub const ZERO_DENSITY_CONTEXT_LIMIT: usize = 474;

/// Total number of AC contexts.
pub const NUM_AC_CONTEXTS: usize = NUM_BLOCK_CTXS * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT);

/// Context for coefficient frequency.
/// Maps coefficient index k to a context bucket.
static COEFF_FREQ_CONTEXT: [u16; 64] = [
    0xBAD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

/// Context for number of non-zeros.
/// Maps nonzeros_left to a context bucket offset.
static COEFF_NUM_NONZERO_CONTEXT: [u16; 64] = [
    0xBAD, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152,
    152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206,
];

/// Compact block context map for DC coding.
#[allow(dead_code)]
pub static COMPACT_BLOCK_CONTEXT_MAP: [u8; 39] = [
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Y
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // X
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // B
];

/// Full block context map.
///
/// Indexed by `[c * NUM_AC_STRATEGY_CODES + strategy_code]` where c is encoder
/// channel (0=X, 1=Y, 2=B). Values must be consistent with `COMPACT_BLOCK_CONTEXT_MAP`
/// which the decoder reads, indexed by `[ch_idx * 13 + order_id]` where
/// ch_idx swaps X↔Y (0→1, 1→0, 2→2) and order_id maps from strategy codes via
/// a LUT (e.g., code 0→order 0, code 4→order 2, code 5→order 3, code 6,7→order 4).
static BLOCK_CONTEXT_MAP: [u8; 81] = [
    // X (c=0): decoder reads with ch_idx=1 (compact group 1)
    //  code: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 ...
    2, 0, 0, 0, 2, 2, 3, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Y (c=1): decoder reads with ch_idx=0 (compact group 0)
    //  DCT4X8=12 and DCT8X4=13 have order_id=1, so block_ctx=0 (already correct)
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // B (c=2): decoder reads with ch_idx=2 (compact group 2)
    2, 0, 0, 0, 2, 2, 3, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Get block context from channel and AC strategy code.
#[inline]
pub const fn block_context(c: usize, ac_strategy_code: u8) -> usize {
    BLOCK_CONTEXT_MAP[c * NUM_AC_STRATEGY_CODES + ac_strategy_code as usize] as usize
}

/// Compute context for zero density (AC coefficient symbols).
///
/// This computes the context based on:
/// - Number of non-zeros remaining in the block
/// - Coefficient index k in scan order
/// - Number of covered blocks (for multi-block transforms)
/// - Previous coefficient was non-zero (prev)
#[inline]
pub fn zero_density_context(
    nonzeros_left: usize,
    k: usize,
    covered_blocks: usize,
    log2_covered_blocks: usize,
    prev: usize,
) -> usize {
    // Scale by covered blocks for multi-block transforms
    let nonzeros_left = (nonzeros_left + covered_blocks - 1) >> log2_covered_blocks;
    let k = k >> log2_covered_blocks;

    (COEFF_NUM_NONZERO_CONTEXT[nonzeros_left] as usize + COEFF_FREQ_CONTEXT[k] as usize) * 2 + prev
}

/// Get the offset into the context map for zero density contexts.
#[inline]
pub const fn zero_density_contexts_offset(block_ctx: usize) -> usize {
    NUM_BLOCK_CTXS * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT * block_ctx
}

/// Compute context for the number of non-zeros.
///
/// Non-zero context is based on predicted number of non-zeros and block context.
/// For better clustering, contexts with same number of non-zeros are grouped.
#[inline]
pub const fn non_zero_context(non_zeros: usize, block_ctx: usize) -> usize {
    let nz_bucket = if non_zeros < 8 {
        non_zeros
    } else if non_zeros >= 64 {
        36
    } else {
        4 + non_zeros / 2
    };
    nz_bucket * NUM_BLOCK_CTXS + block_ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_zero_context() {
        // Test small values map directly
        for i in 0..8 {
            assert_eq!(non_zero_context(i, 0), i * NUM_BLOCK_CTXS);
        }

        // Test medium values use 4 + n/2
        assert_eq!(non_zero_context(8, 0), (4 + 4) * NUM_BLOCK_CTXS);
        assert_eq!(non_zero_context(10, 0), (4 + 5) * NUM_BLOCK_CTXS);

        // Test large values cap at 36
        assert_eq!(non_zero_context(64, 0), 36 * NUM_BLOCK_CTXS);
        assert_eq!(non_zero_context(100, 0), 36 * NUM_BLOCK_CTXS);
    }

    #[test]
    fn test_zero_density_context_bounds() {
        // Test that zero_density_context stays within bounds
        // ZERO_DENSITY_CONTEXT_COUNT (458) is the supremum when x + y < 64
        // ZERO_DENSITY_CONTEXT_LIMIT (474) is the overall supremum
        for nz in 1..64 {
            for k in 1..64 {
                for prev in 0..2 {
                    let ctx = zero_density_context(nz, k, 1, 0, prev);
                    assert!(
                        ctx < ZERO_DENSITY_CONTEXT_LIMIT,
                        "ctx {} >= {}",
                        ctx,
                        ZERO_DENSITY_CONTEXT_LIMIT
                    );
                }
            }
        }
    }

    #[test]
    fn test_block_context() {
        // DCT8 for Y channel (strategy code 0)
        let ctx_y = block_context(1, 0);
        assert_eq!(ctx_y, 0);

        // DCT8x16 for Y channel (strategy code 6)
        let ctx_y_16 = block_context(1, 6);
        assert_eq!(ctx_y_16, 1);

        // DCT8 for X channel (strategy code 0)
        let ctx_x = block_context(0, 0);
        assert_eq!(ctx_x, 2);
    }
}
