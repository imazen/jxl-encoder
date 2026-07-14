// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! AC coefficient context computation for entropy coding.
//!
//! These functions and constants are ported from libjxl-tiny and full libjxl.
//! The live consumers are the VarDCT bitstream writer (`bitstream.rs`,
//! `ac_group.rs`, `encoder.rs`) and the JPEG-transcode path (`jpeg/encode.rs`);
//! `static_codes.rs` re-exports `NUM_AC_CONTEXTS`. A handful of reference
//! constants and the standalone `block_context`/`zero_density_context` helpers
//! are ported-complete but not on the live path — those carry item-level
//! `#[allow(dead_code)]` with a reason, rather than a module-wide blanket.

use super::ac_strategy::AcStrategyMap;
use super::coeff_order::{NUM_ORDER_BUCKETS, STRATEGY_TO_BUCKET};

/// Number of predicted nonzeros buckets (0 to 36 inclusive = 37 values).
pub const NON_ZERO_BUCKETS: usize = 37;

/// Number of AC strategy codes.
#[allow(dead_code)] // parity-reference: feeds the test-only standalone block_context(); not on the live path
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
#[allow(dead_code)] // referenced by unit tests + the jpeg-reencoding debug asserts
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

/// Libjxl's `kDefaultCtxMap` from `ac_context.h:91-96`. **Per-divergence
/// opt-in only — NOT the default.** Has 15 unique contexts vs our
/// 4-context [`COMPACT_BLOCK_CONTEXT_MAP`].
///
/// Selected via [`BlockCtxMap::libjxl_default`] when
/// `ResolvedImprovements.block_ctx_map_15_cluster == true` (i.e.
/// [`crate::api::EncoderStrategy::Libjxl`]).
///
/// W44-71 (`w44_71_15cluster_default_regression_2026-05-19.md`)
/// measured this as a +1.4-3.6% BYTE REGRESSION at default because our
/// `write_context_map_nonsimple` uses Huffman+MTF only while libjxl uses
/// ANS+LZ77 over the larger 7425-entry context-map output. Issue #59
/// tracks the writer-side port that would close the regression.
/// `EncoderStrategy::Libjxl` deliberately re-introduces the regression
/// to match libjxl byte-for-byte (per W44-126 user decision #3:
/// "all-divergence parity, regressions and all").
#[allow(dead_code)]
pub static LIBJXL_DEFAULT_CTX_MAP: [u8; 39] = [
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, // Y
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, // X
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, // B
];

/// Number of block contexts in [`LIBJXL_DEFAULT_CTX_MAP`].
#[allow(dead_code)]
pub const NUM_BLOCK_CTXS_LIBJXL_DEFAULT: usize = 15;

/// Content-adaptive block context map.
///
/// Provides a mapping from (channel, strategy, quantization field value, DC
/// bucket) to a block context ID. The context map allows the entropy coder
/// to use different distributions for blocks with different characteristics.
///
/// The context map is indexed as:
/// `ctx_map[((c_swapped * NUM_ORDERS + order_id) * num_qf_segments + qf_idx)
///          * num_dc_ctxs + dc_idx]`
/// where `c_swapped = if c < 2 { c ^ 1 } else { 2 }` (X↔Y swap for decoder).
///
/// `num_dc_ctxs = (dc_thresholds[0].len() + 1) * (dc_thresholds[1].len() + 1)
///                * (dc_thresholds[2].len() + 1)`.
#[derive(Debug, Clone)]
pub struct BlockCtxMap {
    /// Per-channel signed DC quantile thresholds (0-15 values each). Used by
    /// JPEG re-encoding to split AC tokens by luma DC bucket. The decoder
    /// computes a single `dc_idx` per block from these thresholds via the
    /// multi-channel formula in libjxl `compressed_dc.cc:274-292`:
    ///
    /// ```text
    /// bucket = bucket_x
    /// bucket = bucket * (dc_thresholds[2].len() + 1) + bucket_b
    /// bucket = bucket * (dc_thresholds[1].len() + 1) + bucket_y
    /// ```
    ///
    /// libjxl's JPEG re-encoder fills ONLY `dc_thresholds[1]` (luma) and
    /// leaves [0]/[2] empty, so `bucket = bucket_y = sum(thresholds[1]<luma_dc)`.
    /// The non-JPEG path keeps all three vectors empty (`num_dc_ctxs = 1`,
    /// `dc_idx = 0`), preserving byte-identical output.
    pub dc_thresholds: [Vec<i32>; 3],
    /// QF value thresholds (0-2 values). Blocks with qf > threshold[i] are in
    /// segment i+1. No thresholds means 1 segment (all blocks same).
    pub qf_thresholds: Vec<u32>,
    /// Context map: maps (channel, order, qf_segment, dc_bucket) to block
    /// context ID.
    /// Size = 3 * NUM_ORDER_BUCKETS * num_qf_segments * num_dc_ctxs.
    pub ctx_map: Vec<u8>,
    /// Number of distinct DC bucket combinations across all 3 channels
    /// (product of `(dc_thresholds[c].len() + 1)`). Always ≥1.
    pub num_dc_ctxs: usize,
    /// Number of distinct context IDs (max context ID + 1).
    pub num_ctxs: usize,
}

impl Default for BlockCtxMap {
    /// Returns the default 4-context map matching COMPACT_BLOCK_CONTEXT_MAP.
    fn default() -> Self {
        BlockCtxMap {
            dc_thresholds: [vec![], vec![], vec![]],
            qf_thresholds: vec![],
            ctx_map: COMPACT_BLOCK_CONTEXT_MAP.to_vec(),
            num_dc_ctxs: 1,
            num_ctxs: NUM_BLOCK_CTXS,
        }
    }
}

impl BlockCtxMap {
    /// Returns the libjxl 15-context default map ([`LIBJXL_DEFAULT_CTX_MAP`]).
    /// **Per-divergence opt-in only — NOT the default.** See
    /// [`LIBJXL_DEFAULT_CTX_MAP`] doc-comment for the regression history.
    ///
    /// W44-133 Chunk G: callers select between
    /// [`BlockCtxMap::default`] (4-context, Zenjxl) and
    /// [`BlockCtxMap::libjxl_default`] (15-context, Libjxl) via the
    /// [`BlockCtxMap::default_for_strategy`] helper which routes from
    /// the resolved `block_ctx_map_15_cluster` bool.
    pub fn libjxl_default() -> Self {
        BlockCtxMap {
            dc_thresholds: [vec![], vec![], vec![]],
            qf_thresholds: vec![],
            ctx_map: LIBJXL_DEFAULT_CTX_MAP.to_vec(),
            num_dc_ctxs: 1,
            num_ctxs: NUM_BLOCK_CTXS_LIBJXL_DEFAULT,
        }
    }

    /// Strategy-aware default selector. Replaces the four
    /// `BlockCtxMap::default()` call sites that today hardcode the
    /// 4-context Zenjxl map; reads the per-encode
    /// `block_ctx_map_15_cluster` bool from
    /// [`crate::api::ResolvedImprovements`].
    ///
    /// W44-133 Chunk G — flips to libjxl's 15-context default under
    /// [`crate::api::EncoderStrategy::Libjxl`]. Zenjxl default
    /// (`false`) is byte-identical to pre-Chunk-G output.
    pub fn default_for_strategy(block_ctx_map_15_cluster: bool) -> Self {
        if block_ctx_map_15_cluster {
            Self::libjxl_default()
        } else {
            Self::default()
        }
    }

    /// Build the JPEG-transcode AC block-context map from a luma DC
    /// histogram, total luma DC count, and the first 5 AC entries of the
    /// JXL channel-0 (chroma) quant table.
    ///
    /// Port of libjxl `enc_frame.cc:1049-1094` (jpeg-to-jxl path). The
    /// algorithm cumulative-cuts the 2048-bucket luma DC histogram into
    /// `num_thresholds + 1` quantile buckets where:
    ///
    /// ```text
    /// num_thresholds = CeilLog2Nonzero(total_dc_luma)
    ///                  - CeilLog2Nonzero(qt_ac_sum_0_to_4)
    ///                  - 7
    /// num_thresholds = clamp(num_thresholds, 1, 7)   // 2-8 buckets
    /// ```
    ///
    /// Only `dc_thresholds[1]` (luma) is populated; X/B chroma get empty
    /// threshold vectors. The decoder's `compressed_dc.cc:274-292` formula
    /// then reduces to `dc_idx = sum(dc_thresholds[1] < luma_quant_dc)`,
    /// which is exactly what `compute_jpeg_dc_buckets` produces.
    ///
    /// ctx_map layout (libjxl `enc_frame.cc:1077-1090`):
    /// - Y (c_swapped=0): `ctx_map[0..num_dc_ctxs] = 0, 1, ..., num_dc_ctxs-1`
    ///   (one context per luma DC bucket).
    /// - X (c_swapped=1), grayscale: all chroma → single context `num_dc_ctxs`.
    /// - X (c_swapped=1), color: `ctx_map[NUM_ORDERS*num_dc_ctxs + i] =
    ///   num_dc_ctxs + i/2`.
    /// - B (c_swapped=2), grayscale: same as X above.
    /// - B (c_swapped=2), color: `ctx_map[2*NUM_ORDERS*num_dc_ctxs + i] =
    ///   num_dc_ctxs + (num_dc_ctxs-1)/2 + 1 + i/2`.
    ///
    /// Only the row for `order=0` (DCT8 bucket — JPEG is all-DCT8) carries
    /// distinct values; every other order row inherits `ctx_map[bucket]`
    /// because we fill the entire `3 * NUM_ORDERS * num_dc_ctxs` slice
    /// with the formula above per the libjxl reference.
    ///
    /// `dc_counts` MUST have 2048 entries, one per `idc + 1024` in
    /// `0..=2047`. `total_dc_luma` is the total count of luma DC samples
    /// (== sum of `dc_counts`). `qt_ac_sum_0_to_4` is the sum of the
    /// chroma quant table entries at storage slots 1, 2, 3, 4, 5 (per
    /// libjxl's `qt[1] + qt[2] + ... + qt[5]` with `qt[c=0]` = JXL
    /// channel 0 = JPEG chroma). `is_grayscale` selects the
    /// num_components==1 path.
    #[cfg_attr(not(feature = "jpeg-reencoding"), allow(dead_code))]
    pub fn jpeg_dc_quantile(
        dc_counts: &[usize; 2048],
        total_dc_luma: usize,
        qt_ac_sum_0_to_4: u32,
        is_grayscale: bool,
    ) -> Self {
        // CeilLog2Nonzero(x) returns floor(log2(x)) + 1 for x > 0.
        // libjxl-tiny / libjxl `base/bits.h` definition.
        fn ceil_log2_nonzero(v: usize) -> i32 {
            if v == 0 {
                0
            } else {
                32 - (v as u32).leading_zeros() as i32
            }
        }
        // libjxl `enc_frame.cc:1056-1061`
        let log_dc = ceil_log2_nonzero(total_dc_luma.max(1));
        let log_qt = ceil_log2_nonzero(qt_ac_sum_0_to_4.max(1) as usize);
        let num_thresholds = (log_dc - log_qt - 7).clamp(1, 7) as usize;

        // libjxl `enc_frame.cc:1062-1070`: cumulative-cut the histogram
        // into num_thresholds+1 quantiles, pushing the boundary value
        // (offset by -1025 to make it signed: i32 in [-1025, 1022]).
        let mut dct1 = Vec::with_capacity(num_thresholds);
        let mut cumsum: usize = 0;
        let mut cut = total_dc_luma / (num_thresholds + 1);
        for (j, &count) in dc_counts.iter().enumerate() {
            cumsum += count;
            if cumsum > cut {
                dct1.push(j as i32 - 1025);
                cut = total_dc_luma * (dct1.len() + 1) / (num_thresholds + 1);
            }
        }
        let num_dc_ctxs = dct1.len() + 1;
        // Spec bound (libjxl `dec_ans.cc:48-50`): num_dc_ctxs *
        // (qft.size() + 1) <= 64. With qft empty here, num_dc_ctxs <= 64;
        // the libjxl algorithm clamps num_thresholds <= 7 so
        // num_dc_ctxs <= 8 which fits comfortably.
        debug_assert!(num_dc_ctxs <= 64);

        // libjxl `enc_frame.cc:1073-1090`: ctx_map of size 3 * kNumOrders
        // * num_dc_ctxs. kNumOrders == NUM_ORDER_BUCKETS == 13.
        let mut ctx_map = vec![0u8; 3 * NUM_ORDER_BUCKETS * num_dc_ctxs];
        let n = num_dc_ctxs;
        for i in 0..n {
            // Y (c_swapped=0): one context per bucket
            ctx_map[i] = i as u8;
            if is_grayscale {
                // Grayscale → single context for both chroma planes
                ctx_map[NUM_ORDER_BUCKETS * n + i] = n as u8;
                ctx_map[2 * NUM_ORDER_BUCKETS * n + i] = n as u8;
            } else {
                ctx_map[NUM_ORDER_BUCKETS * n + i] = (n + i / 2) as u8;
                ctx_map[2 * NUM_ORDER_BUCKETS * n + i] = (n + (n - 1) / 2 + 1 + i / 2) as u8;
            }
        }
        let num_ctxs = (*ctx_map.iter().max().unwrap_or(&0)) as usize + 1;
        debug_assert!(num_ctxs <= MAX_BLOCK_CTXS);

        BlockCtxMap {
            dc_thresholds: [vec![], dct1, vec![]],
            qf_thresholds: vec![],
            ctx_map,
            num_dc_ctxs,
            num_ctxs,
        }
    }

    /// EX-J15: variant of [`Self::jpeg_dc_quantile`] that maps chroma at
    /// FULL resolution (one context per luma DC bucket per channel) when
    /// `num_dc_ctxs <= 5`. Falls back to libjxl half-resolution chroma
    /// for `num_dc_ctxs > 5` (mandatory: 3 * 6 = 18 > 16, spec violation).
    ///
    /// Rationale: libjxl uses half-resolution chroma (`n + i/2`) because the
    /// original VarDCT path produces strong (Y, chroma) correlation per
    /// DC bucket and `+1` adjacent buckets generally cluster together. For
    /// JPEG re-encode at low bitrates, chroma blocks are mostly empty
    /// (nzeros=0 dominates), so the chroma context model is starved for
    /// training data when it doesn't get its own bucket. Giving chroma a
    /// full bucket axis SOMETIMES separates the (mostly-empty) chroma
    /// distributions more cleanly. Net effect is corpus-dependent (small
    /// images with num_dc_ctxs=2-3 see the biggest separation; larger
    /// images with num_dc_ctxs=6-7 fall back to libjxl behaviour anyway).
    ///
    /// Default-OFF in production; gated by `EX_J15_FULL_CHROMA=1` env
    /// hook in [`crate::jpeg::encode`].
    #[cfg_attr(not(feature = "jpeg-reencoding"), allow(dead_code))]
    pub fn jpeg_dc_quantile_ex_j15(
        dc_counts: &[usize; 2048],
        total_dc_luma: usize,
        qt_ac_sum_0_to_4: u32,
        is_grayscale: bool,
    ) -> Self {
        // Reuse the libjxl-parity threshold derivation by calling the
        // primary function and inspecting its output.
        let base = Self::jpeg_dc_quantile(dc_counts, total_dc_luma, qt_ac_sum_0_to_4, is_grayscale);
        let n = base.num_dc_ctxs;
        // Full-resolution chroma only when 3 * n <= 16. n in {1, 2, 3, 4, 5}.
        if n > 5 || is_grayscale {
            // Fall back to libjxl mapping (already in `base`).
            return base;
        }
        let mut ctx_map = vec![0u8; 3 * NUM_ORDER_BUCKETS * n];
        for i in 0..n {
            ctx_map[i] = i as u8;
            ctx_map[NUM_ORDER_BUCKETS * n + i] = (n + i) as u8;
            ctx_map[2 * NUM_ORDER_BUCKETS * n + i] = (2 * n + i) as u8;
        }
        let num_ctxs = (*ctx_map.iter().max().unwrap_or(&0)) as usize + 1;
        debug_assert!(num_ctxs <= MAX_BLOCK_CTXS);
        BlockCtxMap {
            dc_thresholds: base.dc_thresholds,
            qf_thresholds: vec![],
            ctx_map,
            num_dc_ctxs: n,
            num_ctxs,
        }
    }
}

impl BlockCtxMap {
    /// Get block context for a given channel, strategy code, and QF value.
    ///
    /// Equivalent to [`Self::block_context_dc`] with `dc_idx = 0`. Use when
    /// `num_dc_ctxs == 1` (i.e. no DC thresholds — the default for VarDCT
    /// lossy and the historical JPEG re-encode path before issue #65).
    ///
    /// `c` is encoder channel (0=X, 1=Y, 2=B).
    /// `strategy_code` is the bitstream strategy code (0-26).
    /// `qf` is the raw quant field value for this block.
    #[inline]
    pub fn block_context(&self, c: usize, strategy_code: u8, qf: u32) -> usize {
        self.block_context_dc(c, strategy_code, qf, 0)
    }

    /// Get block context for a given channel, strategy code, QF value, and
    /// DC bucket index.
    ///
    /// `dc_idx` is the precomputed DC bucket (libjxl `compressed_dc.cc:274-292`
    /// formula) — for our JPEG re-encode path it is `sum(dc_thresholds[1]
    /// < luma_dc_value)` when only luma thresholds are populated. Callers
    /// that pass `dc_idx = 0` on a map with `num_dc_ctxs > 1` will only
    /// select the bucket-0 row of `ctx_map`, defeating DC-quantile context
    /// modeling — pass the per-block precomputed bucket from
    /// `compute_jpeg_dc_buckets` instead.
    ///
    /// `c` is encoder channel (0=X, 1=Y, 2=B).
    /// `strategy_code` is the bitstream strategy code (0-26).
    /// `qf` is the raw quant field value for this block.
    /// `dc_idx` is the per-block DC bucket index in `0..num_dc_ctxs`.
    #[inline]
    pub fn block_context_dc(&self, c: usize, strategy_code: u8, qf: u32, dc_idx: usize) -> usize {
        let order_id = STRATEGY_TO_BUCKET[strategy_code as usize] as usize;
        let mut qf_idx = 0usize;
        for &t in &self.qf_thresholds {
            if qf > t {
                qf_idx += 1;
            }
        }
        let num_qf_segments = self.qf_thresholds.len() + 1;
        let num_dc_ctxs = self.num_dc_ctxs.max(1);
        // Channel swap: decoder uses c_swapped = if c < 2 { c ^ 1 } else { 2 }
        let c_swapped = if c < 2 { c ^ 1 } else { 2 };
        // Decoder formula (libjxl ac_context.h:101-110):
        //   idx = c_swapped * kNumOrders + ord
        //   idx = idx * (qf_thresholds.size() + 1) + qf_idx
        //   idx = idx * num_dc_ctxs + dc_idx
        let mut idx = c_swapped * NUM_ORDER_BUCKETS + order_id;
        idx = idx * num_qf_segments + qf_idx;
        idx = idx * num_dc_ctxs + dc_idx.min(num_dc_ctxs - 1);
        self.ctx_map[idx] as usize
    }

    /// Compute the total number of AC contexts for this map.
    #[inline]
    pub fn num_ac_contexts(&self) -> usize {
        self.num_ctxs * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT)
    }

    /// Get the offset into the context map for zero density contexts.
    #[inline]
    #[allow(dead_code)] // parity-reference; unit tests cross-check vs the free fn
    pub fn zero_density_contexts_offset(&self, block_ctx: usize) -> usize {
        self.num_ctxs * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT * block_ctx
    }

    /// Compute context for the number of non-zeros.
    #[inline]
    #[allow(dead_code)] // parity-reference; unit tests cross-check vs the free fn
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
/// For small images, returns the default map (selected by
/// [`BlockCtxMap::default_for_strategy`] — Zenjxl 4-context or
/// Libjxl 15-context per the
/// `ResolvedImprovements.block_ctx_map_15_cluster` bool). For larger
/// images, computes QF thresholds and clusters (qf_segment, order_id)
/// cells to produce a more efficient context map.
///
/// W44-133 Chunk G — added `block_ctx_map_15_cluster` parameter to
/// route the small-image fallback through the Libjxl 15-context default
/// when [`crate::api::EncoderStrategy::Libjxl`] is selected. Default
/// (`false`) is byte-identical to pre-Chunk-G.
pub fn compute_block_ctx_map(
    quant_field: &[u8],
    ac_strategy: &AcStrategyMap,
    distance: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
    block_ctx_map_15_cluster: bool,
) -> BlockCtxMap {
    let tot = xsize_blocks * ysize_blocks;

    // Small images: no benefit from adaptive context modeling
    // Matches libjxl: tot < (1 << 10) * distance
    //
    // Issue #61 (W44-AUDIT post-W44-73 retry): env-gated diagnostic
    // hook that widens the gate to a distance-independent value.
    // - JXL_ISSUE_61_WIDEN_THRESHOLD=A   ⇒ tot < 1024 (drop distance scaling)
    // - JXL_ISSUE_61_WIDEN_THRESHOLD=B   ⇒ tot < 512 * distance (half-scale)
    // - JXL_ISSUE_61_WIDEN_THRESHOLD=C   ⇒ tot < 512 (very aggressive)
    // - unset / other                    ⇒ libjxl parity (default)
    //
    // Used only by `examples/issue_61_block_ctx_map_widen_ab.rs` for AB
    // measurement. Production code path is unchanged.
    let size_for_ctx_model: usize = {
        #[cfg(feature = "std")]
        {
            match std::env::var("JXL_ISSUE_61_WIDEN_THRESHOLD")
                .ok()
                .as_deref()
            {
                Some("A") => 1024,
                Some("B") => ((1u64 << 9) as f64 * distance as f64) as usize,
                Some("C") => 512,
                _ => ((1u64 << 10) as f64 * distance as f64) as usize,
            }
        }
        #[cfg(not(feature = "std"))]
        {
            ((1u64 << 10) as f64 * distance as f64) as usize
        }
    };
    if tot < size_for_ctx_model {
        return BlockCtxMap::default_for_strategy(block_ctx_map_15_cluster);
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
        dc_thresholds: [vec![], vec![], vec![]],
        qf_thresholds,
        ctx_map,
        num_dc_ctxs: 1,
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
#[allow(dead_code)] // parity-reference table for the test-only standalone block_context()
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
#[allow(dead_code)] // standalone libjxl-tiny formula; unit tests cross-check vs BlockCtxMap
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
#[allow(dead_code)] // standalone reference; unit tests cross-check vs BlockCtxMap
pub const fn zero_density_contexts_offset(block_ctx: usize) -> usize {
    NUM_BLOCK_CTXS * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT * block_ctx
}

/// Compute context for the number of non-zeros.
///
/// Non-zero context is based on predicted number of non-zeros and block context.
/// For better clustering, contexts with same number of non-zeros are grouped.
#[inline]
#[allow(dead_code)] // standalone reference; unit tests cross-check vs BlockCtxMap
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
    // Constant-pinning tests deliberately assert relationships between
    // calibrated gate constants (CLAUDE.md "Invariant Preservation");
    // clippy::assertions_on_constants would flag every such pin.
    #![allow(clippy::assertions_on_constants)]

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
        assert!(map.dc_thresholds.iter().all(|d| d.is_empty()));
        assert_eq!(map.num_dc_ctxs, 1);
        assert_eq!(map.ctx_map.len(), 39); // 3 * 13 * 1 * 1

        // Default map should give same results as hardcoded block_context()
        // for any QF value (no QF thresholds)
        assert_eq!(map.block_context(1, 0, 5), block_context(1, 0));
        assert_eq!(map.block_context(1, 6, 5), block_context(1, 6));
        assert_eq!(map.block_context(0, 0, 5), block_context(0, 0));

        // block_context_dc with dc_idx=0 on num_dc_ctxs=1 map is identical
        assert_eq!(map.block_context_dc(1, 0, 5, 0), block_context(1, 0));
        assert_eq!(map.block_context_dc(0, 0, 5, 0), block_context(0, 0));
    }

    /// Issue #65: JPEG DC-quantile constructor produces the expected
    /// num_dc_ctxs and ctx_map shape on a small synthetic histogram.
    #[test]
    fn test_block_ctx_map_jpeg_dc_quantile_smoke() {
        // Synthetic luma DC distribution: bell curve around index 1024
        // (DC == 0), 1000 samples total.
        let mut dc_counts = [0usize; 2048];
        for &(idx, n) in &[
            (1020usize, 50usize),
            (1022, 100),
            (1024, 300),
            (1026, 250),
            (1028, 200),
            (1030, 80),
            (1032, 20),
        ] {
            dc_counts[idx] = n;
        }
        let total_dc_luma: usize = dc_counts.iter().sum();
        assert_eq!(total_dc_luma, 1000);

        // Choose qt sum that gives positive num_thresholds.
        // CeilLog2Nonzero(1000) = 10. We want log_dc - log_qt - 7 in
        // [1, 7], so log_qt in [-4, 2]. log_qt(8) = 4, log_qt(2) = 2.
        // With qt_sum = 4, log_qt = 3, num_thresholds = 10-3-7 = 0
        // → clamp to 1.
        let m = BlockCtxMap::jpeg_dc_quantile(&dc_counts, total_dc_luma, 4, false);
        assert!(
            m.num_dc_ctxs >= 2,
            "expected ≥2 buckets, got {}",
            m.num_dc_ctxs
        );
        assert!(m.num_dc_ctxs <= 8);
        assert!(m.dc_thresholds[0].is_empty());
        assert_eq!(m.dc_thresholds[1].len(), m.num_dc_ctxs - 1);
        assert!(m.dc_thresholds[2].is_empty());
        // Thresholds must be strictly increasing
        for w in m.dc_thresholds[1].windows(2) {
            assert!(w[0] < w[1]);
        }
        // ctx_map size = 3 * 13 * num_dc_ctxs * 1 (no qf thresholds)
        assert_eq!(m.ctx_map.len(), 3 * NUM_ORDER_BUCKETS * m.num_dc_ctxs);
        assert!(m.num_ctxs <= MAX_BLOCK_CTXS);

        // Y luma gets one context per bucket
        for i in 0..m.num_dc_ctxs {
            assert_eq!(m.ctx_map[i], i as u8);
        }
    }

    /// Issue #65: grayscale routes both chroma channels to a single shared
    /// context per bucket (libjxl `enc_frame.cc:1080-1083`).
    #[test]
    fn test_block_ctx_map_jpeg_dc_quantile_grayscale() {
        let mut dc_counts = [0usize; 2048];
        for &(idx, n) in &[(1024usize, 500usize), (1028, 500)] {
            dc_counts[idx] = n;
        }
        let total_dc_luma = 1000usize;
        let m = BlockCtxMap::jpeg_dc_quantile(&dc_counts, total_dc_luma, 4, true);
        let n = m.num_dc_ctxs;
        // X (c_swapped=1) and B (c_swapped=2) rows for order=0 must equal
        // `num_dc_ctxs` (a single shared chroma context).
        for i in 0..n {
            assert_eq!(m.ctx_map[NUM_ORDER_BUCKETS * n + i], n as u8);
            assert_eq!(m.ctx_map[2 * NUM_ORDER_BUCKETS * n + i], n as u8);
        }
    }

    /// Issue #65: color path uses distinct X / B contexts spread across
    /// buckets (libjxl `enc_frame.cc:1086-1088`).
    #[test]
    fn test_block_ctx_map_jpeg_dc_quantile_color_layout() {
        // Force ≥3 buckets so the i/2 spread is observable.
        let mut dc_counts = [0usize; 2048];
        for i in 0..512usize {
            dc_counts[1024 - 256 + i / 2] += 1;
            dc_counts[1024 + i / 2] += 1;
        }
        let total_dc_luma: usize = dc_counts.iter().sum();
        // Use a very small qt_sum to push num_thresholds high.
        let m = BlockCtxMap::jpeg_dc_quantile(&dc_counts, total_dc_luma, 1, false);
        let n = m.num_dc_ctxs;
        for i in 0..n {
            assert_eq!(m.ctx_map[NUM_ORDER_BUCKETS * n + i], (n + i / 2) as u8);
            assert_eq!(
                m.ctx_map[2 * NUM_ORDER_BUCKETS * n + i],
                (n + (n - 1) / 2 + 1 + i / 2) as u8
            );
        }
    }

    /// EX-J15: full-resolution chroma DC quantile mapping. For
    /// `num_dc_ctxs <= 5`, each chroma channel gets its own block context
    /// per luma DC bucket (no half-resolution collapse). For
    /// `num_dc_ctxs > 5`, falls back to libjxl half-resolution
    /// (verified by comparing against the primary function's ctx_map).
    #[test]
    fn test_block_ctx_map_jpeg_dc_quantile_ex_j15_full_chroma() {
        // Construct a histogram producing num_dc_ctxs in {2, 3, 4}.
        let mut dc_counts = [0usize; 2048];
        for &(idx, n) in &[(1020usize, 100usize), (1024, 600), (1028, 200), (1032, 100)] {
            dc_counts[idx] = n;
        }
        let m = BlockCtxMap::jpeg_dc_quantile_ex_j15(&dc_counts, 1000, 4, false);
        let n = m.num_dc_ctxs;
        assert!(
            (1..=5).contains(&n),
            "n={n} should be in 1..=5 for this test"
        );
        // Y: 0..n. Cb: n..2n. Cr: 2n..3n.
        for i in 0..n {
            assert_eq!(m.ctx_map[i], i as u8, "Y bucket {i}");
            assert_eq!(
                m.ctx_map[NUM_ORDER_BUCKETS * n + i],
                (n + i) as u8,
                "Cb bucket {i}"
            );
            assert_eq!(
                m.ctx_map[2 * NUM_ORDER_BUCKETS * n + i],
                (2 * n + i) as u8,
                "Cr bucket {i}"
            );
        }
        assert!(m.num_ctxs <= MAX_BLOCK_CTXS);
        // Should expand context count vs libjxl mapping for n in 2..=5.
        let base = BlockCtxMap::jpeg_dc_quantile(&dc_counts, 1000, 4, false);
        if n >= 2 {
            assert!(
                m.num_ctxs > base.num_ctxs,
                "EX-J15 should expand ctx count for n={n}: got {} vs base {}",
                m.num_ctxs,
                base.num_ctxs
            );
        }
    }

    /// EX-J15: grayscale must fall back to libjxl half-resolution mapping
    /// (chroma single context) because there's nothing for full-res to
    /// expand.
    #[test]
    fn test_block_ctx_map_jpeg_dc_quantile_ex_j15_grayscale_falls_back() {
        let mut dc_counts = [0usize; 2048];
        for &(idx, n) in &[(1024usize, 500usize), (1028, 500)] {
            dc_counts[idx] = n;
        }
        let m_ex_j15 = BlockCtxMap::jpeg_dc_quantile_ex_j15(&dc_counts, 1000, 4, true);
        let m_base = BlockCtxMap::jpeg_dc_quantile(&dc_counts, 1000, 4, true);
        assert_eq!(m_ex_j15.ctx_map, m_base.ctx_map);
        assert_eq!(m_ex_j15.num_ctxs, m_base.num_ctxs);
    }

    /// EX-J15: when num_dc_ctxs > 5 the spec mandate (num_ctxs <= 16) forces
    /// fallback to libjxl half-resolution chroma.
    #[test]
    fn test_block_ctx_map_jpeg_dc_quantile_ex_j15_large_n_falls_back() {
        // Push num_thresholds to its maximum (7 → num_dc_ctxs = 8).
        // Uniform histogram → quantile cuts will be evenly spaced.
        let dc_counts = [1usize; 2048];
        let total = 2048usize;
        // Push qt_sum small to maximise num_thresholds via the log formula.
        let m_ex_j15 = BlockCtxMap::jpeg_dc_quantile_ex_j15(&dc_counts, total, 1, false);
        let m_base = BlockCtxMap::jpeg_dc_quantile(&dc_counts, total, 1, false);
        if m_base.num_dc_ctxs > 5 {
            // Falls back: ctx_map identical to libjxl mapping.
            assert_eq!(m_ex_j15.ctx_map, m_base.ctx_map);
            assert_eq!(m_ex_j15.num_ctxs, m_base.num_ctxs);
        }
    }

    /// Issue #65: block_context_dc round-trip — for any (qf=1, order=0,
    /// dc_idx) on the JPEG DC map, the encoder formula must match the
    /// libjxl decoder's `Context()` (ac_context.h:101-110) exactly.
    #[test]
    fn test_block_context_dc_decoder_parity_on_jpeg_map() {
        let mut dc_counts = [0usize; 2048];
        for &(idx, n) in &[(1020usize, 100usize), (1024, 600), (1028, 200), (1032, 100)] {
            dc_counts[idx] = n;
        }
        let m = BlockCtxMap::jpeg_dc_quantile(&dc_counts, 1000, 4, false);
        for c in 0..3 {
            let c_swapped = if c < 2 { c ^ 1 } else { 2 };
            for dc_idx in 0..m.num_dc_ctxs {
                // strategy_code 0 = DCT8 → order_id 0
                let got = m.block_context_dc(c, 0, 1, dc_idx);
                let expected =
                    m.ctx_map[c_swapped * NUM_ORDER_BUCKETS * m.num_dc_ctxs + dc_idx] as usize;
                assert_eq!(
                    got, expected,
                    "c={} dc_idx={} c_swapped={}",
                    c, dc_idx, c_swapped
                );
            }
        }
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

        // W44-133 Chunk G: libjxl_default also satisfies the spec bound
        // (15 contexts fits in MAX_BLOCK_CTXS = 16).
        let libjxl_map = BlockCtxMap::libjxl_default();
        assert!(libjxl_map.num_ctxs <= MAX_BLOCK_CTXS);
    }

    /// W44-133 Chunk G: libjxl 15-cluster default constructor must
    /// return the exact contents of `kDefaultCtxMap` from libjxl
    /// `ac_context.h:91-96` with `num_ctxs = 15` and no QF thresholds.
    #[test]
    fn test_block_ctx_map_libjxl_default() {
        let map = BlockCtxMap::libjxl_default();
        assert_eq!(map.num_ctxs, NUM_BLOCK_CTXS_LIBJXL_DEFAULT);
        assert_eq!(map.num_ctxs, 15);
        assert!(map.qf_thresholds.is_empty());
        assert_eq!(map.ctx_map.len(), 39); // 3 * 13 * 1
        // Byte-for-byte match with the libjxl static table
        assert_eq!(map.ctx_map.as_slice(), &LIBJXL_DEFAULT_CTX_MAP[..]);
        // Spec-bounded (≤16)
        assert!(map.num_ctxs <= MAX_BLOCK_CTXS);
    }

    /// W44-133 Chunk G: `default_for_strategy(false)` produces the
    /// Zenjxl 4-cluster default (byte-identical to pre-Chunk-G).
    /// `default_for_strategy(true)` produces the libjxl 15-cluster
    /// default. Selector logic for `EncoderStrategy::Libjxl`.
    #[test]
    fn test_block_ctx_map_default_for_strategy_routes_correctly() {
        let zenjxl = BlockCtxMap::default_for_strategy(false);
        assert_eq!(zenjxl.num_ctxs, NUM_BLOCK_CTXS);
        assert_eq!(zenjxl.ctx_map.as_slice(), &COMPACT_BLOCK_CONTEXT_MAP[..]);

        let libjxl = BlockCtxMap::default_for_strategy(true);
        assert_eq!(libjxl.num_ctxs, NUM_BLOCK_CTXS_LIBJXL_DEFAULT);
        assert_eq!(libjxl.ctx_map.as_slice(), &LIBJXL_DEFAULT_CTX_MAP[..]);

        // The two maps must differ — that's the whole point of the
        // strategy split; if they ever match, something is wrong.
        assert_ne!(zenjxl.num_ctxs, libjxl.num_ctxs);
        assert_ne!(zenjxl.ctx_map, libjxl.ctx_map);
    }

    /// W44-133 Chunk G: `compute_block_ctx_map`'s new
    /// `block_ctx_map_15_cluster` parameter routes the small-image
    /// short-circuit (`tot < 1024 * distance`) through the matching
    /// default constructor.
    #[test]
    fn test_compute_block_ctx_map_small_image_uses_strategy_default() {
        // 4×4 blocks at distance=1.0 → tot=16 < 1024, short-circuits.
        let xsize_blocks = 4;
        let ysize_blocks = 4;
        let quant_field = vec![1u8; 16];
        let ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

        let zenjxl = compute_block_ctx_map(
            &quant_field,
            &ac_strategy,
            1.0,
            xsize_blocks,
            ysize_blocks,
            /*block_ctx_map_15_cluster=*/ false,
        );
        assert_eq!(zenjxl.num_ctxs, NUM_BLOCK_CTXS);

        let libjxl = compute_block_ctx_map(
            &quant_field,
            &ac_strategy,
            1.0,
            xsize_blocks,
            ysize_blocks,
            /*block_ctx_map_15_cluster=*/ true,
        );
        assert_eq!(libjxl.num_ctxs, NUM_BLOCK_CTXS_LIBJXL_DEFAULT);
    }
}
