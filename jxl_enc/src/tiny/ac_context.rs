// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! AC coefficient context computation for entropy coding.
//!
//! These functions and constants are ported from libjxl-tiny and will be used
//! when the AC group encoding is implemented.

#![allow(dead_code)]

use super::ac_strategy::AcStrategyMap;
use super::coeff_order::{NUM_ORDER_BUCKETS, STRATEGY_TO_BUCKET};

/// Number of predicted nonzeros buckets (0 to 36 inclusive = 37 values).
pub const NON_ZERO_BUCKETS: usize = 37;

/// Number of AC strategy codes.
pub const NUM_AC_STRATEGY_CODES: usize = 27;

/// Number of block contexts for the default (hardcoded) context map.
pub const NUM_BLOCK_CTXS: usize = 4;

/// Supremum of ZeroDensityContext + 1 when x + y < 64.
pub const ZERO_DENSITY_CONTEXT_COUNT: usize = 458;

/// Supremum of ZeroDensityContext + 1 (all cases).
#[allow(dead_code)]
pub const ZERO_DENSITY_CONTEXT_LIMIT: usize = 474;

/// Total number of AC contexts for the default context map.
pub const NUM_AC_CONTEXTS: usize = NUM_BLOCK_CTXS * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT);

/// Maximum number of distinct block contexts allowed by the spec.
pub const MAX_BLOCK_CTXS: usize = 16;

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

/// Compact block context map for DC coding (the default map).
/// Indexed by `[ch_idx * 13 + order_id]` where ch_idx swaps X↔Y.
#[allow(dead_code)]
pub static COMPACT_BLOCK_CONTEXT_MAP: [u8; 39] = [
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Y
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // X
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // B
];

/// Content-adaptive block context map.
///
/// Provides a mapping from (channel, strategy, quantization field value) to a
/// block context ID. The context map allows the entropy coder to use different
/// distributions for blocks with different characteristics.
///
/// The context map is indexed as:
/// `ctx_map[(c_swapped * NUM_ORDERS + order_id) * num_qf_segments + qf_idx]`
/// where `c_swapped = if c < 2 { c ^ 1 } else { 2 }` (X↔Y swap for decoder).
#[derive(Debug, Clone)]
pub struct BlockCtxMap {
    /// QF value thresholds (0-2 values). Blocks with qf > threshold[i] are in
    /// segment i+1. No thresholds means 1 segment (all blocks same).
    pub qf_thresholds: Vec<u32>,
    /// Context map: maps (channel, order, qf_segment) to block context ID.
    /// Size = 3 * NUM_ORDER_BUCKETS * (qf_thresholds.len() + 1).
    pub ctx_map: Vec<u8>,
    /// Number of distinct context IDs (max context ID + 1).
    pub num_ctxs: usize,
}

impl Default for BlockCtxMap {
    /// Returns the default 4-context map matching COMPACT_BLOCK_CONTEXT_MAP.
    fn default() -> Self {
        BlockCtxMap {
            qf_thresholds: vec![],
            ctx_map: COMPACT_BLOCK_CONTEXT_MAP.to_vec(),
            num_ctxs: NUM_BLOCK_CTXS,
        }
    }
}

impl BlockCtxMap {
    /// Get block context for a given channel, strategy code, and QF value.
    ///
    /// `c` is encoder channel (0=X, 1=Y, 2=B).
    /// `strategy_code` is the bitstream strategy code (0-26).
    /// `qf` is the raw quant field value for this block.
    #[inline]
    pub fn block_context(&self, c: usize, strategy_code: u8, qf: u32) -> usize {
        let order_id = STRATEGY_TO_BUCKET[strategy_code as usize] as usize;
        let mut qf_idx = 0usize;
        for &t in &self.qf_thresholds {
            if qf > t {
                qf_idx += 1;
            }
        }
        let num_qf_segments = self.qf_thresholds.len() + 1;
        // Channel swap: decoder uses c_swapped = if c < 2 { c ^ 1 } else { 2 }
        let c_swapped = if c < 2 { c ^ 1 } else { 2 };
        let idx = (c_swapped * NUM_ORDER_BUCKETS + order_id) * num_qf_segments + qf_idx;
        self.ctx_map[idx] as usize
    }

    /// Compute the total number of AC contexts for this map.
    #[inline]
    pub fn num_ac_contexts(&self) -> usize {
        self.num_ctxs * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT)
    }

    /// Get the offset into the context map for zero density contexts.
    #[inline]
    pub fn zero_density_contexts_offset(&self, block_ctx: usize) -> usize {
        self.num_ctxs * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT * block_ctx
    }

    /// Compute context for the number of non-zeros.
    #[inline]
    pub fn non_zero_context(&self, non_zeros: usize, block_ctx: usize) -> usize {
        let nz_bucket = if non_zeros < 8 {
            non_zeros
        } else if non_zeros >= 64 {
            36
        } else {
            4 + non_zeros / 2
        };
        nz_bucket * self.num_ctxs + block_ctx
    }
}

/// Compute a content-adaptive block context map from the quant field and AC strategy.
///
/// Port of libjxl's `FindBestBlockEntropyModel` from `enc_heuristics.cc:69-204`.
///
/// For small images, returns the default map. For larger images, computes QF
/// thresholds and clusters (qf_segment, order_id) cells to produce a more
/// efficient context map.
pub fn compute_block_ctx_map(
    quant_field: &[u8],
    ac_strategy: &AcStrategyMap,
    distance: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> BlockCtxMap {
    let tot = xsize_blocks * ysize_blocks;

    // Small images: no benefit from adaptive context modeling
    // Matches libjxl: tot < (1 << 10) * distance
    let size_for_ctx_model = ((1u64 << 10) as f64 * distance as f64) as usize;
    if tot < size_for_ctx_model {
        return BlockCtxMap::default();
    }

    // Count QF occurrences and (order, qf) co-occurrences.
    // qf values are u8 (1-255 after raw_quant), we use 0-255 range.
    let mut qf_counts = [0usize; 256];
    let mut qf_ord_counts = [[0usize; 256]; NUM_ORDER_BUCKETS];
    let mut ord_counts = [0usize; NUM_ORDER_BUCKETS];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let qf = quant_field[by * xsize_blocks + bx] as usize;
            // libjxl uses qf_row[x] - 1 but our quant_field is already 0-based raw_quant
            let strategy_code = ac_strategy.strategy_code(bx, by);
            let ord = STRATEGY_TO_BUCKET[strategy_code as usize] as usize;
            qf_counts[qf] += 1;
            qf_ord_counts[ord][qf] += 1;
            ord_counts[ord] += 1;
        }
    }

    // Determine number of QF segments (1 or 2)
    let size_for_qf_split = ((1u64 << 13) as f64 * distance as f64) as usize;
    let num_qf_segments: usize = if tot < size_for_qf_split { 1 } else { 2 };

    // Find QF thresholds by median-cut of the QF distribution
    let mut qf_thresholds: Vec<u32> = Vec::new();
    if num_qf_segments > 1 {
        let mut cumsum = 0usize;
        let mut next = 1usize;
        let mut last_cut = 256usize;
        let mut cut = tot * next / num_qf_segments;
        for j in 0u32..256 {
            cumsum += qf_counts[j as usize];
            if cumsum > cut {
                if j != 0 {
                    qf_thresholds.push(j);
                }
                last_cut = j as usize;
                while cumsum > cut {
                    next += 1;
                    cut = tot * next / num_qf_segments;
                }
            } else if next > qf_thresholds.len() + 1 && j as usize - 1 == last_cut && j != 0 {
                qf_thresholds.push(j);
            }
        }
    }

    let num_qf_segs = qf_thresholds.len() + 1;
    let num_cells = NUM_ORDER_BUCKETS * num_qf_segs;

    // Count blocks per cell: counts[ord * num_qf_segs + qf_seg]
    let mut counts = vec![0usize; num_cells];
    let mut qft_pos = 0usize;
    for j in 0u32..256 {
        if qft_pos < qf_thresholds.len() && j == qf_thresholds[qft_pos] {
            qft_pos += 1;
        }
        for ord in 0..NUM_ORDER_BUCKETS {
            counts[ord * num_qf_segs + qft_pos] += qf_ord_counts[ord][j as usize];
        }
    }

    // Clustering: repeatedly merge the lowest-count pair.
    // remap[cell] = canonical cell it maps to
    let mut remap: Vec<u8> = (0..num_cells as u8).collect();
    let mut clusters: Vec<u8> = remap.clone();
    let nb_clusters_luma = (tot / size_for_ctx_model / 2).clamp(2, 9);
    let nb_clusters_chroma = (tot / size_for_ctx_model / 3).clamp(1, 5);

    while clusters.len() > nb_clusters_luma {
        // Sort by count descending (most common first)
        clusters.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]));
        let last = clusters.len() - 1;
        let second_last = last - 1;
        // Merge last (smallest) into second-to-last
        counts[clusters[second_last] as usize] += counts[clusters[last] as usize];
        counts[clusters[last] as usize] = 0;
        remap[clusters[last] as usize] = clusters[second_last];
        clusters.pop();
    }

    // Flatten remap chains
    for i in 0..remap.len() {
        while remap[remap[i] as usize] != remap[i] {
            remap[i] = remap[remap[i] as usize];
        }
    }

    // Relabel starting from 0
    let mut remap_remap = vec![u8::MAX; num_cells];
    let mut num_luma: u8 = 0;
    for i in 0..remap.len() {
        if remap_remap[remap[i] as usize] == u8::MAX {
            remap_remap[remap[i] as usize] = num_luma;
            num_luma += 1;
        }
        remap[i] = remap_remap[remap[i] as usize];
    }

    // Build context map: luma uses full clustering, chroma uses clamped clustering
    // Layout: [Y (ch_idx=0)] [X (ch_idx=1)] [B (ch_idx=2)]
    // Each section: NUM_ORDER_BUCKETS * num_qf_segs entries
    let section_size = NUM_ORDER_BUCKETS * num_qf_segs;
    let mut ctx_map = vec![0u8; section_size * 3];

    // Luma (Y, ch_idx=0) gets the full remap
    ctx_map[..section_size].copy_from_slice(&remap[..section_size]);

    // Chroma (X, ch_idx=1 and B, ch_idx=2) gets clamped clustering
    // libjxl: ctx_map[i] = num + clamp(remap[i % section_size], 0, nb_clusters_chroma - 1)
    let chroma_max = nb_clusters_chroma as u8 - 1;
    for i in section_size..section_size * 3 {
        let luma_ctx = remap[i % section_size];
        ctx_map[i] = num_luma + luma_ctx.min(chroma_max);
    }

    let num_ctxs = *ctx_map.iter().max().unwrap_or(&0) as usize + 1;

    BlockCtxMap {
        qf_thresholds,
        ctx_map,
        num_ctxs,
    }
}

/// Full block context map.
///
/// Indexed by `[c * NUM_AC_STRATEGY_CODES + strategy_code]` where c is encoder
/// channel (0=X, 1=Y, 2=B). Values must be consistent with `COMPACT_BLOCK_CONTEXT_MAP`
/// which the decoder reads, indexed by `[ch_idx * 13 + order_id]` where
/// ch_idx swaps X↔Y (0→1, 1→0, 2→2) and order_id maps from strategy codes via
/// a LUT (e.g., code 0→order 0, code 4→order 2, code 5→order 3, code 6,7→order 4).
static BLOCK_CONTEXT_MAP: [u8; 81] = [
    // X (c=0): decoder reads with ch_idx=1 (compact group 1)
    //  code: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 ...
    //  IDENTITY=1, DCT2X2=2, DCT4X4=3 all have order_id=1 → compact[14]=2
    //  DCT32X16=10, DCT16X32=11 have order_id=6 → compact[19]=3
    //  DCT64X64=18 has order_id=7 → compact[20]=3
    //  DCT64X32=19, DCT32X64=20 have order_id=8 → compact[21]=3
    2, 2, 2, 2, 2, 2, 3, 3, 0, 0, 3, 3, 2, 2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0,
    // Y (c=1): decoder reads with ch_idx=0 (compact group 0)
    //  IDENTITY=1, DCT2X2=2, DCT4X8=12, DCT8X4=13, DCT4X4=3 all have order_id=1 → compact[1]=0
    //  DCT32X16=10, DCT16X32=11 have order_id=6 → compact[6]=1
    //  DCT64X64=18 has order_id=7 → compact[7]=1
    //  DCT64X32=19, DCT32X64=20 have order_id=8 → compact[8]=1
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    // B (c=2): decoder reads with ch_idx=2 (compact group 2)
    //  IDENTITY=1, DCT2X2=2, DCT4X4=3 all have order_id=1 → compact[27]=2
    //  DCT32X16=10, DCT16X32=11 have order_id=6 → compact[32]=3
    //  DCT64X64=18 has order_id=7 → compact[33]=3
    //  DCT64X32=19, DCT32X64=20 have order_id=8 → compact[34]=3
    2, 2, 2, 2, 2, 2, 3, 3, 0, 0, 3, 3, 2, 2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0,
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

    #[test]
    fn test_block_ctx_map_default() {
        let map = BlockCtxMap::default();
        assert_eq!(map.num_ctxs, NUM_BLOCK_CTXS);
        assert!(map.qf_thresholds.is_empty());
        assert_eq!(map.ctx_map.len(), 39); // 3 * 13 * 1

        // Default map should give same results as hardcoded block_context()
        // for any QF value (no QF thresholds)
        assert_eq!(map.block_context(1, 0, 5), block_context(1, 0));
        assert_eq!(map.block_context(1, 6, 5), block_context(1, 6));
        assert_eq!(map.block_context(0, 0, 5), block_context(0, 0));
    }

    #[test]
    fn test_block_ctx_map_dynamic_methods() {
        let map = BlockCtxMap::default();
        // non_zero_context should match static version
        assert_eq!(map.non_zero_context(5, 0), non_zero_context(5, 0));
        assert_eq!(map.non_zero_context(8, 2), non_zero_context(8, 2));
        // zero_density_contexts_offset should match static version
        assert_eq!(
            map.zero_density_contexts_offset(0),
            zero_density_contexts_offset(0)
        );
        assert_eq!(
            map.zero_density_contexts_offset(3),
            zero_density_contexts_offset(3)
        );
        // num_ac_contexts should match static constant
        assert_eq!(map.num_ac_contexts(), NUM_AC_CONTEXTS);
    }

    #[test]
    fn test_block_ctx_map_num_ctxs_bounded() {
        // Any computed map must have num_ctxs <= MAX_BLOCK_CTXS
        let map = BlockCtxMap::default();
        assert!(map.num_ctxs <= MAX_BLOCK_CTXS);
    }
}
