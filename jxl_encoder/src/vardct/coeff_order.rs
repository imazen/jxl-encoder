// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Custom coefficient ordering for AC coefficients.
//!
//! Reorders AC coefficients per-strategy-type so frequently-zero positions
//! appear last. The encoder stops writing at the last non-zero, so pushing
//! zeros to the end saves bits.
//!
//! The permutation is encoded as Lehmer codes in the HF global section,
//! using a separate 8-context entropy code.

use super::ac_group::ac_strategy_info;
use super::ac_strategy::AcStrategyMap;
use super::common::{DCT_BLOCK_SIZE, ceil_log2_nonzero, floor_log2_nonzero};
use crate::bit_writer::BitWriter;
use crate::entropy_coding::encode::{
    build_entropy_code_ans_with_options, build_entropy_code_with_options, write_entropy_code_ans,
    write_tokens, write_tokens_ans,
};
use crate::entropy_coding::token::Token;
use crate::error::Result;

/// Number of order buckets used by our encoder.
/// Bucket 0 = DCT8, Bucket 2 = DCT16x16, Bucket 3 = DCT32x32, Bucket 4 = DCT8x16/DCT16x8.
/// Total 13 buckets exist in the spec.
pub const NUM_ORDER_BUCKETS: usize = 13;

/// Number of contexts for permutation entropy coding.
pub const NUM_PERMUTATION_CONTEXTS: usize = 8;

/// Strategy code to order bucket mapping.
/// Matches libjxl's kStrategyOrder for the strategies we support.
pub const STRATEGY_TO_BUCKET: [u8; 27] = [
    0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
];

/// Get the order bucket for a raw strategy code.
#[inline]
pub fn strategy_bucket(raw_strategy: u8) -> u8 {
    STRATEGY_TO_BUCKET[raw_strategy as usize]
}

/// Generate the natural (zig-zag) coefficient order for a transform with
/// cx x cy covered blocks. Matches jxl-rs `natural_coeff_order`.
///
/// The natural order is a zig-zag scan of the coefficient grid, where
/// CoefficientLayout ensures cx >= cy (transpose if needed).
pub fn natural_coeff_order(cx: usize, cy: usize) -> Vec<u32> {
    // CoefficientLayout: ensure cx >= cy
    let (cx, cy) = if cx >= cy { (cx, cy) } else { (cy, cx) };

    let xsize = cx * 8;
    let total = cx * cy * DCT_BLOCK_SIZE;

    // xs = cx/cy ratio, for skipping rows in rectangular transforms
    let xs = cx / cy;
    let xsm = xs - 1;
    let xss = if xs <= 1 {
        0
    } else {
        ceil_log2_nonzero(xs) as usize
    };

    let mut out = vec![0u32; total];

    // First half of the block (upper-left triangle + main diagonal)
    let mut cur = cx * cy; // skip LLF positions
    for i in 0..xsize {
        for j in 0..=i {
            let (x, mut y) = if i % 2 != 0 { (i - j, j) } else { (j, i - j) };
            if (y & xsm) != 0 {
                continue;
            }
            y >>= xss;

            let val = if x < cx && y < cy {
                y * cx + x
            } else {
                let v = cur;
                cur += 1;
                v
            };

            if val < total {
                out[val] = (y * xsize + x) as u32;
            }
        }
    }

    // Second half (lower-right triangle)
    for ir in 1..xsize {
        let ip = xsize - ir;
        let i = ip - 1;
        for j in 0..=i {
            let (x, mut y) = if i % 2 != 0 {
                (xsize - 1 - j, xsize - 1 - (i - j))
            } else {
                (xsize - 1 - (i - j), xsize - 1 - j)
            };
            if (y & xsm) != 0 {
                continue;
            }
            y >>= xss;

            if cur < total {
                out[cur] = (y * xsize + x) as u32;
            }
            cur += 1;
        }
    }

    out
}

/// Generate the inverse lookup table for a natural coefficient order.
/// `lut[natural_order[i]] = i` for all i.
pub fn natural_coeff_order_lut(cx: usize, cy: usize) -> Vec<u32> {
    let order = natural_coeff_order(cx, cy);
    let mut lut = vec![0u32; order.len()];
    for (i, &pos) in order.iter().enumerate() {
        lut[pos as usize] = i as u32;
    }
    lut
}

/// Compute the context for a permutation value.
/// Uses HybridUint(0,0,0) semantics: token = 1 + floor_log2(val) for val>=1, else 0.
/// Clamped to [0, 7].
#[inline]
pub fn coeff_order_context(val: u32) -> usize {
    if val == 0 {
        0
    } else {
        ((1 + floor_log2_nonzero(val)) as usize).min(NUM_PERMUTATION_CONTEXTS - 1)
    }
}

/// Compute the Lehmer code of a permutation using a Fenwick tree.
/// O(n log n) time.
///
/// The permutation must contain unique indices in [0..n).
/// The output code[i] represents how many elements smaller than
/// permutation[i] appear after position i.
pub fn compute_lehmer_code(permutation: &[u32]) -> Vec<u32> {
    let n = permutation.len();
    let mut code = vec![0u32; n];
    let mut temp = vec![0u32; n + 1];

    for idx in 0..n {
        let s = permutation[idx];

        // Compute prefix sum in Fenwick tree (count of used elements <= s)
        let mut penalty = 0u32;
        let mut i = s as usize + 1;
        while i != 0 {
            penalty += temp[i];
            i &= i - 1; // clear lowest bit
        }

        debug_assert!(s >= penalty, "Lehmer code: s={} < penalty={}", s, penalty);
        code[idx] = s - penalty;

        // Update Fenwick tree: mark s as used
        i = s as usize + 1;
        while i < n + 1 {
            temp[i] += 1;
            i += i & i.wrapping_neg(); // add lowest bit
        }
    }

    code
}

/// Count zero coefficients per position for each (bucket, channel) combination.
///
/// Returns a map: `zero_counts[bucket][channel][position] = count`.
/// LLF positions get count = -1 to ensure they sort first.
pub fn count_zero_coefficients(
    quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<Vec<Vec<i64>>> {
    // Initialize: 13 buckets x 3 channels x max_size
    let mut counts: Vec<Vec<Vec<i64>>> = (0..NUM_ORDER_BUCKETS)
        .map(|_| vec![Vec::new(); 3])
        .collect();

    // Pre-allocate for buckets we use
    // Bucket 0: DCT8, size 64
    for ch in &mut counts[0] {
        *ch = vec![0i64; 64];
    }
    // Bucket 2: DCT16x16, size 256
    for ch in &mut counts[2] {
        *ch = vec![0i64; 256];
    }
    // Bucket 3: DCT32x32, size 1024
    for ch in &mut counts[3] {
        *ch = vec![0i64; 1024];
    }
    // Bucket 4: DCT8x16/DCT16x8, size 128
    for ch in &mut counts[4] {
        *ch = vec![0i64; 128];
    }
    // Bucket 5: DCT32x8/DCT8x32, size 256 — not implemented, but pre-allocate for safety
    for ch in &mut counts[5] {
        *ch = vec![0i64; 256];
    }
    // Bucket 6: DCT32x16/DCT16x32, size 512
    for ch in &mut counts[6] {
        *ch = vec![0i64; 512];
    }
    // Bucket 7: DCT64x64, size 4096
    for ch in &mut counts[7] {
        *ch = vec![0i64; 4096];
    }
    // Bucket 8: DCT64x32/DCT32x64, size 2048
    for ch in &mut counts[8] {
        *ch = vec![0i64; 2048];
    }

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }

            let raw_strategy = ac_strategy.raw_strategy(bx, by);
            let strategy_code = super::ac_strategy::STRATEGY_CODE_LUT[raw_strategy as usize];
            // strategy_bucket expects bitstream strategy code
            let bucket = strategy_bucket(strategy_code) as usize;
            // ac_strategy_info expects raw strategy codes (0-6)
            let (cx, cy, covered_blocks, _, _) = ac_strategy_info(raw_strategy);
            let size = covered_blocks * DCT_BLOCK_SIZE;

            // Ensure count vector is large enough
            for ch in &mut counts[bucket] {
                if ch.len() < size {
                    ch.resize(size, 0);
                }
            }

            for c in 0..3 {
                if covered_blocks == 1 {
                    // DCT8: coefficients are in quant_ac[c][by][bx]
                    let block = &quant_ac[c][by][bx];
                    for k in 0..DCT_BLOCK_SIZE {
                        if block[k] == 0 {
                            counts[bucket][c][k] += 1;
                        }
                    }
                } else {
                    // Multi-block: iterate by sub-block to avoid per-coeff div/mod
                    let covered_x_local = super::ac_strategy::COVERED_X[raw_strategy as usize];
                    for block_slot in 0..covered_blocks {
                        let slot_by = by + block_slot / covered_x_local;
                        let slot_bx = bx + block_slot % covered_x_local;
                        let block = &quant_ac[c][slot_by][slot_bx];
                        let base_idx = block_slot * DCT_BLOCK_SIZE;
                        for k in 0..DCT_BLOCK_SIZE {
                            if block[k] == 0 {
                                counts[bucket][c][base_idx + k] += 1;
                            }
                        }
                    }
                }

                // Mark LLF positions with -1 so they sort first (ascending)
                let (layout_cx, layout_cy) = if cx >= cy { (cx, cy) } else { (cy, cx) };
                let block_dim = 8;
                for iy in 0..layout_cy {
                    for ix in 0..layout_cx {
                        counts[bucket][c][iy * layout_cx * block_dim + ix] = -1;
                    }
                }
            }
        }
    }

    counts
}

/// Compute custom coefficient orders from zero counts.
///
/// Returns `(orders, used_orders)` where:
/// - `orders[bucket][channel]` is the custom order (positions sorted by ascending zero count)
/// - `used_orders` is a bitmask of buckets that have non-default orders
pub fn compute_custom_orders(zero_counts: &[Vec<Vec<i64>>]) -> (Vec<Vec<Vec<u32>>>, u32) {
    let mut orders: Vec<Vec<Vec<u32>>> = (0..NUM_ORDER_BUCKETS)
        .map(|_| vec![Vec::new(); 3])
        .collect();
    let mut used_orders = 0u32;

    for bucket in 0..NUM_ORDER_BUCKETS {
        // libjxl's ComputeUsedOrders (enc_coeff_order.cc:53-58) skips buckets > 6.
        // Buckets 7+ (DCT64x64, DCT64x32/DCT32x64) are never customized.
        if bucket > 6 {
            continue;
        }

        if zero_counts[bucket][0].is_empty() {
            continue;
        }

        let size = zero_counts[bucket][0].len();
        let inv_sqrt_sz = 1.0 / (size as f64).sqrt();

        // Get natural order for comparison
        let (cx, cy) = bucket_to_cx_cy(bucket);
        if cx == 0 {
            continue;
        }
        let natural = natural_coeff_order(cx, cy);

        let mut bucket_has_custom = false;

        for c in 0..3 {
            let counts = &zero_counts[bucket][c];
            if counts.is_empty() {
                continue;
            }

            // Build (position, sort_key) array
            // sort_key = (quantized_count << 16) | natural_index for stable sort
            let mut pos_and_count: Vec<(u32, u64)> = Vec::with_capacity(size);
            for (i, &pos) in natural.iter().enumerate().take(size) {
                let raw_count = counts[pos as usize];
                // Quantize count to reduce distinct values (matching C++ line 210)
                let quantized = if raw_count < 0 {
                    0u64 // LLF: keep at front
                } else {
                    // raw_count * inv_sqrt_sz + 0.1, truncated
                    (raw_count as f64 * inv_sqrt_sz + 0.1) as u64
                };
                pos_and_count.push((pos, (quantized << 16) | (i as u64)));
            }

            // Stable sort by count (ascending), preserving natural order for ties
            pos_and_count.sort_by_key(|&(_, key)| key);

            // Extract the order
            let order: Vec<u32> = pos_and_count.iter().map(|&(pos, _)| pos).collect();

            // Check if different from natural order
            let is_different = order.iter().zip(natural.iter()).any(|(a, b)| a != b);
            if is_different {
                bucket_has_custom = true;
            }

            orders[bucket][c] = order;
        }

        if bucket_has_custom {
            used_orders |= 1 << bucket;
        }
    }

    (orders, used_orders)
}

/// Convert order bucket index to (cx, cy) for natural order generation.
fn bucket_to_cx_cy(bucket: usize) -> (usize, usize) {
    match bucket {
        0 => (1, 1), // DCT8: 1x1 blocks
        2 => (2, 2), // DCT16x16: 2x2 blocks
        3 => (4, 4), // DCT32x32: 4x4 blocks
        4 => (2, 1), // DCT8x16/DCT16x8: 2x1 blocks (after CoefficientLayout)
        5 => (4, 1), // DCT32x8/DCT8x32: 4x1 blocks (after CoefficientLayout) — not implemented
        6 => (4, 2), // DCT32x16/DCT16x32: 4x2 blocks (after CoefficientLayout)
        7 => (8, 8), // DCT64x64: 8x8 blocks
        8 => (8, 4), // DCT64x32/DCT32x64: 8x4 blocks (after CoefficientLayout)
        _ => (0, 0), // Not supported by our encoder
    }
}

/// Tokenize coefficient orders as Lehmer codes.
///
/// For each (bucket, channel) with its bit set in `used_orders`:
/// 1. Convert the custom order to natural-order-index space (order_zigzag)
/// 2. Compute the Lehmer code
/// 3. Trim trailing zeros
/// 4. Emit tokens with coeff_order_context
pub fn tokenize_coeff_orders(orders: &[Vec<Vec<u32>>], used_orders: u32) -> Vec<Token> {
    let mut tokens = Vec::new();

    for (bucket, bucket_orders) in orders.iter().enumerate().take(NUM_ORDER_BUCKETS) {
        if used_orders & (1 << bucket) == 0 {
            continue;
        }

        let (cx, cy) = bucket_to_cx_cy(bucket);
        if cx == 0 {
            continue;
        }

        let natural_lut = natural_coeff_order_lut(cx, cy);
        let llf = cx * cy; // CoefficientLayout normalized
        let (cx_norm, cy_norm) = if cx >= cy { (cx, cy) } else { (cy, cx) };
        let llf_norm = cx_norm * cy_norm;
        let size = llf_norm * DCT_BLOCK_SIZE;

        for order in bucket_orders {
            if order.is_empty() {
                continue;
            }

            // Convert to natural-order-index space: order_zigzag[i] = natural_lut[order[i]]
            let order_zigzag: Vec<u32> =
                order.iter().map(|&pos| natural_lut[pos as usize]).collect();

            // Compute Lehmer code
            let lehmer = compute_lehmer_code(&order_zigzag);

            // Find the last non-zero Lehmer code entry (past the skip region)
            let mut end = size;
            while end > llf && lehmer[end - 1] == 0 {
                end -= 1;
            }

            // First token: end - skip (how many Lehmer codes to read)
            tokens.push(Token::new(
                coeff_order_context(size as u32) as u32,
                (end - llf) as u32,
            ));

            // Remaining tokens: Lehmer codes with context from previous value
            let mut last = 0u32;
            for &val in &lehmer[llf..end] {
                tokens.push(Token::new(coeff_order_context(last) as u32, val));
                last = val;
            }
        }
    }

    tokens
}

/// Build entropy code and write coefficient orders to the bitstream.
///
/// This writes the permutation entropy code header followed by all the
/// permutation tokens, matching the format expected by decoders.
pub fn build_and_write_coeff_orders(
    tokens: &[Token],
    use_ans: bool,
    writer: &mut BitWriter,
) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }

    // LZ77 flag: no LZ77 for permutation data
    writer.write(1, 0)?;

    if use_ans {
        let code = build_entropy_code_ans_with_options(
            tokens,
            NUM_PERMUTATION_CONTEXTS,
            false, // no enhanced clustering for permutation
            None,  // no LZ77 for permutation data
            None,  // no pixel hint for permutation data
        );
        write_entropy_code_ans(&code, writer)?;
        write_tokens_ans(tokens, &code, None, writer)?;
    } else {
        let code = build_entropy_code_with_options(tokens, NUM_PERMUTATION_CONTEXTS, false, None);
        let ec = code.as_entropy_code();
        crate::entropy_coding::encode::write_entropy_code(&ec, writer)?;
        write_tokens(tokens, &ec, None, writer)?;
    }

    Ok(())
}

/// Get the custom order for a specific (bucket, channel) from the orders map.
/// Returns None if no custom order exists for this combination.
pub fn get_custom_order(
    orders: &[Vec<Vec<u32>>],
    used_orders: u32,
    raw_strategy: u8,
    channel: usize,
) -> Option<&[u32]> {
    let bucket = strategy_bucket(raw_strategy) as usize;
    if used_orders & (1 << bucket) == 0 {
        return None;
    }
    let order = &orders[bucket][channel];
    if order.is_empty() {
        return None;
    }
    Some(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_coeff_order_8x8() {
        // Must match jxl-rs golden data and our COEFF_ORDER_8X8
        let order = natural_coeff_order(1, 1);
        assert_eq!(order.len(), 64);

        let expected: [u32; 64] = [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37,
            44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        ];
        assert_eq!(order, expected);

        // Must match our existing COEFF_ORDER_8X8
        let existing = super::super::ac_group::COEFF_ORDER_8X8;
        assert_eq!(order, &existing[..]);
    }

    #[test]
    fn test_natural_coeff_order_8x16() {
        // 2x1 covered blocks (after CoefficientLayout normalization)
        let order = natural_coeff_order(2, 1);
        assert_eq!(order.len(), 128);

        let expected: [u32; 128] = [
            0, 1, 16, 2, 3, 17, 32, 18, 4, 5, 19, 33, 48, 34, 20, 6, 7, 21, 35, 49, 64, 50, 36, 22,
            8, 9, 23, 37, 51, 65, 80, 66, 52, 38, 24, 10, 11, 25, 39, 53, 67, 81, 96, 82, 68, 54,
            40, 26, 12, 13, 27, 41, 55, 69, 83, 97, 112, 98, 84, 70, 56, 42, 28, 14, 15, 29, 43,
            57, 71, 85, 99, 113, 114, 100, 86, 72, 58, 44, 30, 31, 45, 59, 73, 87, 101, 115, 116,
            102, 88, 74, 60, 46, 47, 61, 75, 89, 103, 117, 118, 104, 90, 76, 62, 63, 77, 91, 105,
            119, 120, 106, 92, 78, 79, 93, 107, 121, 122, 108, 94, 95, 109, 123, 124, 110, 111,
            125, 126, 127,
        ];
        assert_eq!(order, expected);

        // Must match our existing COEFF_ORDER_8X16
        let existing = super::super::ac_group::COEFF_ORDER_8X16;
        assert_eq!(order, &existing[..]);
    }

    #[test]
    fn test_natural_coeff_order_16x16() {
        // 2x2 covered blocks
        let order = natural_coeff_order(2, 2);
        assert_eq!(order.len(), 256);

        // Must match our existing COEFF_ORDER_16X16
        let existing = super::super::ac_group::COEFF_ORDER_16X16;
        assert_eq!(order, &existing[..]);
    }

    #[test]
    fn test_natural_coeff_order_lut_inverse() {
        // LUT must be inverse of order
        for &(cx, cy) in &[(1, 1), (2, 1), (2, 2)] {
            let order = natural_coeff_order(cx, cy);
            let lut = natural_coeff_order_lut(cx, cy);
            assert_eq!(order.len(), lut.len());

            for (i, &pos) in order.iter().enumerate() {
                assert_eq!(
                    lut[pos as usize], i as u32,
                    "LUT inverse failed at i={}, pos={} for {}x{}",
                    i, pos, cx, cy
                );
            }
        }
    }

    #[test]
    fn test_coeff_order_context() {
        assert_eq!(coeff_order_context(0), 0);
        assert_eq!(coeff_order_context(1), 1); // 1 + floor_log2(1) = 1 + 0 = 1
        assert_eq!(coeff_order_context(2), 2); // 1 + floor_log2(2) = 1 + 1 = 2
        assert_eq!(coeff_order_context(3), 2); // 1 + floor_log2(3) = 1 + 1 = 2
        assert_eq!(coeff_order_context(4), 3); // 1 + floor_log2(4) = 1 + 2 = 3
        assert_eq!(coeff_order_context(7), 3);
        assert_eq!(coeff_order_context(8), 4);
        assert_eq!(coeff_order_context(15), 4);
        assert_eq!(coeff_order_context(16), 5);
        assert_eq!(coeff_order_context(31), 5);
        assert_eq!(coeff_order_context(32), 6);
        assert_eq!(coeff_order_context(63), 6);
        assert_eq!(coeff_order_context(64), 7);
        assert_eq!(coeff_order_context(128), 7); // clamped to 7
        assert_eq!(coeff_order_context(1000), 7);
    }

    #[test]
    fn test_lehmer_code_identity() {
        // Identity permutation [0,1,2,3,4] -> all-zero Lehmer code
        let perm: Vec<u32> = (0..5).collect();
        let code = compute_lehmer_code(&perm);
        assert_eq!(code, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_lehmer_code_reverse() {
        // Reverse permutation [4,3,2,1,0]
        let perm = vec![4u32, 3, 2, 1, 0];
        let code = compute_lehmer_code(&perm);
        assert_eq!(code, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_lehmer_code_known() {
        // Known example from jxl-rs tests:
        // Permutation [2, 4, 0, 1, 3] has Lehmer code [2, 3, 0, 0, 0]
        let perm = vec![2u32, 4, 0, 1, 3];
        let code = compute_lehmer_code(&perm);
        assert_eq!(code, vec![2, 3, 0, 0, 0]);
    }

    #[test]
    fn test_lehmer_code_jxl_rs_golden() {
        // From jxl-rs test: expected permutation [0,1,2,3,5,6,8,10,11,15,4,9,7,12,13,14]
        // The Lehmer code of the part after skip=4 is:
        // Permuted slice [5,6,8,10,11,15,4,9,7,12,13,14] with Lehmer code [1,1,2,3,3,6,0,1]
        // But compute_lehmer_code operates on the full permutation as indices [0..n)
        // So we need to set up the test correctly.
        //
        // The full permutation is [0,1,2,3,5,6,8,10,11,15,4,9,7,12,13,14]
        let full_perm = vec![0u32, 1, 2, 3, 5, 6, 8, 10, 11, 15, 4, 9, 7, 12, 13, 14];
        let code = compute_lehmer_code(&full_perm);
        // First 4 are identity, so their Lehmer codes are 0
        assert_eq!(code[0], 0);
        assert_eq!(code[1], 0);
        assert_eq!(code[2], 0);
        assert_eq!(code[3], 0);
        // Elements 4-11 should have Lehmer code [1,1,2,3,3,6,0,1]
        assert_eq!(&code[4..12], &[1, 1, 2, 3, 3, 6, 0, 1]);
    }

    #[test]
    fn test_tokenize_natural_order_produces_minimal_tokens() {
        // When order equals natural order, Lehmer code is all-zeros.
        // Tokenization should produce one token (end=0) per channel.
        let natural = natural_coeff_order(1, 1);
        let mut orders = vec![vec![Vec::new(); 3]; NUM_ORDER_BUCKETS];
        for order in &mut orders[0] {
            *order = natural.clone();
        }
        let used_orders = 1u32; // bucket 0

        let tokens = tokenize_coeff_orders(&orders, used_orders);
        // 3 channels, each should have exactly 1 token (end=0)
        assert_eq!(tokens.len(), 3);
        for t in &tokens {
            assert_eq!(t.value, 0, "Natural order should produce end=0");
        }
    }

    #[test]
    fn test_strategy_bucket() {
        assert_eq!(strategy_bucket(0), 0); // DCT8 -> bucket 0
        assert_eq!(strategy_bucket(4), 2); // DCT16X16 -> bucket 2
        assert_eq!(strategy_bucket(6), 4); // DCT8X16 -> bucket 4
        assert_eq!(strategy_bucket(7), 4); // DCT16X8 -> bucket 4
    }

    #[test]
    fn test_compute_custom_orders_all_zero_image() {
        // For an all-zero quantized image, all positions have the same zero count.
        // Stable sort should preserve natural order, so no custom order is needed.
        let xsize_blocks = 8;
        let ysize_blocks = 8;

        // Create all-zero quantized blocks
        let quant_ac: [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3] = [
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
        ];

        // All DCT8 strategy
        let ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

        let counts = count_zero_coefficients(&quant_ac, &ac_strategy, xsize_blocks, ysize_blocks);
        let (_, used_orders) = compute_custom_orders(&counts);

        // All-zero should produce natural order (no custom needed)
        assert_eq!(
            used_orders, 0,
            "All-zero image should not need custom orders"
        );
    }
}
