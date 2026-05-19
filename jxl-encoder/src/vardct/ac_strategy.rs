// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Adaptive AC strategy selection.
//!
//! Ported from libjxl-tiny `enc_ac_strategy.cc`. For each 16x16 block region,
//! selects between:
//! - Four DCT8 (8×8) transforms
//! - Two DCT16x8 (16×8) transforms (left column + right column)
//! - Two DCT8x16 (8×16) transforms (top row + bottom row)
//!
//! Selection is based on entropy estimation: the strategy that yields the
//! lowest estimated coded size (including information-loss penalty) wins.

use super::ac_strategy_search::{
    find_best_16x16_transform, find_best_32x32_transform, find_best_64x64_transform,
    try_merge_16x16, try_merge_32x32,
};
use super::afv::afv_transform_from_pixels;
use super::block_extract::*;
use super::chroma_from_luma::{CflMap, ytob_ratio, ytox_ratio};
use super::common::{
    BLOCK_DIM, DCT_BLOCK_SIZE, GROUP_DIM_IN_BLOCKS, TILE_DIM_IN_BLOCKS, as_array_mut, as_array_ref,
    ceil_log2_nonzero, uninit_buf,
};
use super::dct::{
    dct_4x4_full, dct_4x8_full, dct_8x4_full, dct_8x8, dct_8x16, dct_16x8, dct_16x16, dct_16x32,
    dct_32x16, dct_32x32, dct_32x64, dct_64x32, dct_64x64, dct2x2_transform, idct_4x4_full,
    idct_4x8_full, idct_8x4_full, idct_8x8, idct_8x16, idct_16x8, idct_16x16, idct_16x32,
    idct_32x16, idct_32x32, idct_32x64, idct_64x32, idct_64x64, identity_transform,
    inverse_dct2x2_transform, inverse_identity_transform,
};
use super::quant::{dequant_weights, dequant_weights_full, quant_weights, quant_weights_full};
use crate::effort::EffortProfile;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Process-wide override: when `true`, the fused-DCT8 + entropy path in
/// `estimate_entropy_full_impl` is replaced by the explicit non-fused
/// fallback (separate DCT then separate entropy_estimate_coeffs).
///
/// Used by the W44-9 Sub-chunk B investigation (`f_d_fused_vs_unfused_dct8`
/// example) to A/B whether FMA op-ordering in the fused AVX2 kernel gives
/// DCT8 a borderline cost advantage that explains the +2.92pp DCT8 area
/// over-pick on F-D photo cells vs cjxl.
///
/// Default is `false` (uses fused path) — flipping has zero effect on
/// production callers. Only the example crate's harness flips it.
static FORCE_UNFUSED_DCT8: AtomicBool = AtomicBool::new(false);

/// Counter for the unfused-fallback branch (W44-9 verification).
/// Lets the example harness assert the override actually fired.
static UNFUSED_BRANCH_HITS: AtomicU64 = AtomicU64::new(0);

/// Counter for the fused-path branch (W44-9 verification).
static FUSED_BRANCH_HITS: AtomicU64 = AtomicU64::new(0);

/// Set the process-wide override; see [`FORCE_UNFUSED_DCT8`].
///
/// **Investigation-only.** This is a debug knob for the W44-9 Sub-chunk B
/// A/B harness. Production code paths never call this — the default
/// `false` preserves the fused-DCT8 path, which is byte-identical to
/// the prior behaviour.
pub fn set_force_unfused_dct8_entropy(v: bool) {
    FORCE_UNFUSED_DCT8.store(v, Ordering::Relaxed);
}

/// Reset the per-branch hit counters (W44-9 verification).
pub fn reset_dct8_branch_counters() {
    FUSED_BRANCH_HITS.store(0, Ordering::Relaxed);
    UNFUSED_BRANCH_HITS.store(0, Ordering::Relaxed);
}

/// Returns `(fused_hits, unfused_hits)` since the last reset (W44-9).
pub fn dct8_branch_counters() -> (u64, u64) {
    (
        FUSED_BRANCH_HITS.load(Ordering::Relaxed),
        UNFUSED_BRANCH_HITS.load(Ordering::Relaxed),
    )
}

/// Read the process-wide override; see [`FORCE_UNFUSED_DCT8`].
#[inline]
fn force_unfused_dct8_entropy() -> bool {
    let v = FORCE_UNFUSED_DCT8.load(Ordering::Relaxed);
    if v {
        UNFUSED_BRANCH_HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        FUSED_BRANCH_HITS.fetch_add(1, Ordering::Relaxed);
    }
    v
}

/// Pre-allocated scratch buffers for entropy estimation.
/// Avoids per-call heap allocations in the hot `estimate_entropy_full` loop.
pub(super) struct EntropyEstScratch {
    /// DCT coefficients for 3 channels (max 3 × 4096 for DCT64x64).
    pub block: Vec<f32>,
    /// Error coefficients for pixel-domain IDCT (max 4096).
    pub error_coeffs: Vec<f32>,
    /// Pixel-domain error output from IDCT (max 4096).
    pub pixel_error: Vec<f32>,
    /// Per-block entropy estimates for the current 64×64 region.
    /// Indexed as `[iy * 8 + ix]` where (ix, iy) are block coords within the region.
    /// Populated by `find_best_16x16_transform`, consumed by 32×32/64×64 levels
    /// to avoid redundant re-evaluation of sub-block costs.
    pub entropy_estimate: [f32; 64],
    /// Cached 8×8 pixel data for 3 channels. Avoids redundant extraction when
    /// multiple single-block strategies evaluate the same block position.
    pub pixels_8x8: [[f32; 64]; 3],
    /// Block position (bx, by) for which `pixels_8x8` is valid.
    /// Set to `(usize::MAX, usize::MAX)` to invalidate.
    pub pixels_8x8_pos: (usize, usize),
}

impl EntropyEstScratch {
    pub fn new() -> Self {
        const MAX: usize = 4096; // DCT64x64
        Self {
            block: vec![0.0f32; 3 * MAX],
            error_coeffs: vec![0.0f32; MAX],
            pixel_error: vec![0.0f32; MAX],
            entropy_estimate: [0.0; 64],
            pixels_8x8: [[0.0; 64]; 3],
            pixels_8x8_pos: (usize::MAX, usize::MAX),
        }
    }
}

/// Raw strategy codes matching the C++ `AcStrategy::Type` enum.
/// Note: These are internal codes, not bitstream codes. Use STRATEGY_CODE_LUT
/// to convert to bitstream codes.
pub const RAW_STRATEGY_DCT8: u8 = 0;
pub const RAW_STRATEGY_DCT16X8: u8 = 1;
pub const RAW_STRATEGY_DCT8X16: u8 = 2;
pub const RAW_STRATEGY_DCT16X16: u8 = 3;
pub const RAW_STRATEGY_DCT32X32: u8 = 4;
pub const RAW_STRATEGY_DCT4X8: u8 = 5;
pub const RAW_STRATEGY_DCT8X4: u8 = 6;
pub const RAW_STRATEGY_DCT4X4: u8 = 7;
pub const RAW_STRATEGY_IDENTITY: u8 = 8;
pub const RAW_STRATEGY_DCT2X2: u8 = 9;
pub const RAW_STRATEGY_DCT32X16: u8 = 10;
pub const RAW_STRATEGY_DCT16X32: u8 = 11;
pub const RAW_STRATEGY_AFV0: u8 = 12;
pub const RAW_STRATEGY_AFV1: u8 = 13;
pub const RAW_STRATEGY_AFV2: u8 = 14;
pub const RAW_STRATEGY_AFV3: u8 = 15;
pub const RAW_STRATEGY_DCT64X64: u8 = 16;
pub const RAW_STRATEGY_DCT64X32: u8 = 17;
pub const RAW_STRATEGY_DCT32X64: u8 = 18;

/// Number of supported raw strategies.
pub const NUM_RAW_STRATEGIES: usize = 19;

/// Strategy code as written to the bitstream (via `StrategyCode()`).
/// These differ from raw strategy codes.
/// From libjxl ac_strategy.h: DCT=0, IDENTITY=1, DCT2X2=2, DCT4X4=3, DCT16X16=4,
/// DCT32X32=5, DCT16X8=6, DCT8X16=7, DCT32X16=10, DCT16X32=11, DCT4X8=12, DCT8X4=13,
/// AFV0=14, AFV1=15, AFV2=16, AFV3=17, DCT64X64=18, DCT64X32=19, DCT32X64=20.
pub(crate) const STRATEGY_CODE_LUT: [u8; NUM_RAW_STRATEGIES] = [
    0, 6, 7, 4, 5, 12, 13, 3, 1, 2, 10, 11, 14, 15, 16, 17, 18, 19, 20,
];

/// Covered blocks in X direction for each raw strategy.
/// IDENTITY, DCT2X2, DCT4X8, DCT8X4, DCT4X4, and AFV0-3 cover 1×1 blocks.
/// DCT32X16 (32 rows × 16 cols): 2 cols × 4 rows of 8×8 blocks
/// DCT16X32 (16 rows × 32 cols): 4 cols × 2 rows of 8×8 blocks
/// DCT64X64: 8 cols × 8 rows. DCT64X32 (64r × 32c): 4 cols × 8 rows.
/// DCT32X64 (32r × 64c): 8 cols × 4 rows.
pub(crate) const COVERED_X: [usize; NUM_RAW_STRATEGIES] =
    [1, 1, 2, 2, 4, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 8, 4, 8];

/// Covered blocks in Y direction for each raw strategy.
pub(crate) const COVERED_Y: [usize; NUM_RAW_STRATEGIES] =
    [1, 2, 1, 2, 4, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1, 1, 8, 8, 4];

/// Per-block AC strategy map.
///
/// Each byte stores `(raw_strategy << 1) | is_first` matching the C++
/// `AcStrategyImage` layout.
pub struct AcStrategyMap {
    data: Vec<u8>,
    pub xsize_blocks: usize,
    pub ysize_blocks: usize,
}

impl AcStrategyMap {
    /// Create a new map filled with DCT8 (all blocks are first blocks).
    pub fn new_dct8(xsize_blocks: usize, ysize_blocks: usize) -> Self {
        // DCT8: raw_strategy=0, is_first=true → (0 << 1) | 1 = 1
        let data = vec![1u8; xsize_blocks * ysize_blocks];
        Self {
            data,
            xsize_blocks,
            ysize_blocks,
        }
    }

    /// Create a new map forcing a specific strategy for all blocks that fit.
    /// Blocks that don't fit the strategy (e.g., at image edges) use DCT8.
    pub fn force_strategy(xsize_blocks: usize, ysize_blocks: usize, raw_strategy: u8) -> Self {
        let mut map = Self::new_dct8(xsize_blocks, ysize_blocks);
        let cx = COVERED_X[raw_strategy as usize];
        let cy = COVERED_Y[raw_strategy as usize];

        for by in (0..ysize_blocks).step_by(cy) {
            for bx in (0..xsize_blocks).step_by(cx) {
                // Only set if the full coverage fits
                if bx + cx <= xsize_blocks && by + cy <= ysize_blocks {
                    map.set(bx, by, raw_strategy);
                }
            }
        }
        map
    }

    /// Get the raw strategy at (bx, by).
    #[inline]
    pub fn raw_strategy(&self, bx: usize, by: usize) -> u8 {
        self.data[by * self.xsize_blocks + bx] >> 1
    }

    /// Is this the first (top-left) block of the transform?
    #[inline]
    pub fn is_first(&self, bx: usize, by: usize) -> bool {
        (self.data[by * self.xsize_blocks + bx] & 1) != 0
    }

    /// Get the strategy code for bitstream writing.
    #[inline]
    pub fn strategy_code(&self, bx: usize, by: usize) -> u8 {
        STRATEGY_CODE_LUT[self.raw_strategy(bx, by) as usize]
    }

    /// Covered blocks in X for the strategy at (bx, by).
    #[inline]
    pub fn covered_blocks_x(&self, bx: usize, by: usize) -> usize {
        COVERED_X[self.raw_strategy(bx, by) as usize]
    }

    /// Covered blocks in Y for the strategy at (bx, by).
    #[inline]
    pub fn covered_blocks_y(&self, bx: usize, by: usize) -> usize {
        COVERED_Y[self.raw_strategy(bx, by) as usize]
    }

    /// Set a block and all its covered sub-blocks.
    ///
    /// For DCT8 (raw_strategy=0): sets 1 block.
    /// For DCT16X8 (raw_strategy=1): sets 2 blocks vertically (1×2).
    /// For DCT8X16 (raw_strategy=2): sets 2 blocks horizontally (2×1).
    pub fn set(&mut self, bx: usize, by: usize, raw_strategy: u8) {
        let cx = COVERED_X[raw_strategy as usize];
        let cy = COVERED_Y[raw_strategy as usize];

        // The downstream group-transform pipeline assumes every
        // multi-block AC strategy fits inside a single 32×32-block
        // pass-group AND inside the image grid. The in-tree per-tile
        // strategy search satisfies this naturally (tiles align with
        // groups, search bounds clamp to the image), but downstream
        // callers of `__pre_quantized::EncoderPrecomputed::from_parts`
        // (notably jxl-encoder-gpu's strat-search injector) can supply
        // an `AcStrategyMap` whose multi-block entries cross either
        // boundary. Without an explicit guard this used to OOB silently
        // in release mode at `vardct/transform.rs` — e.g. line 544 for
        // DCT64x64 DC writes — with a confusing
        // `index out of bounds: the len is 1024 but the index is 1048`.
        //
        // In debug mode we keep the assertion so first-party callers
        // surface the bug at the source. In release mode we silently
        // downgrade to "leave the touched blocks alone" — they retain
        // whatever value was there (DCT8 from `new_dct8`) — so
        // untrusted producers can't crash the encoder.
        let crosses_group = {
            let gx = bx % GROUP_DIM_IN_BLOCKS;
            let gy = by % GROUP_DIM_IN_BLOCKS;
            gx + cx > GROUP_DIM_IN_BLOCKS || gy + cy > GROUP_DIM_IN_BLOCKS
        };
        let crosses_image = bx + cx > self.xsize_blocks || by + cy > self.ysize_blocks;
        debug_assert!(
            !crosses_group,
            "varblock crosses pass group border: bx={bx}, by={by}, raw_strategy={raw_strategy}, cx={cx}, cy={cy}, gx={}, gy={}",
            bx % GROUP_DIM_IN_BLOCKS,
            by % GROUP_DIM_IN_BLOCKS
        );
        debug_assert!(
            !crosses_image,
            "varblock extends past image grid: bx={bx}, by={by}, raw_strategy={raw_strategy}, cx={cx}, cy={cy}, xsize_blocks={}, ysize_blocks={}",
            self.xsize_blocks, self.ysize_blocks
        );
        if crosses_group || crosses_image {
            return;
        }
        for iy in 0..cy {
            for ix in 0..cx {
                let is_first = (iy | ix) == 0;
                self.data[(by + iy) * self.xsize_blocks + bx + ix] =
                    (raw_strategy << 1) | (is_first as u8);
            }
        }
    }

    /// Get the raw packed byte at (bx, by).
    /// The byte is `(raw_strategy << 1) | is_first`.
    #[inline]
    fn raw_byte(&self, bx: usize, by: usize) -> u8 {
        self.data[by * self.xsize_blocks + bx]
    }

    /// Set the raw packed byte at (bx, by) directly.
    /// Bypasses multi-block coverage logic — use only for save/restore.
    #[inline]
    fn set_raw_byte(&mut self, bx: usize, by: usize, byte: u8) {
        self.data[by * self.xsize_blocks + bx] = byte;
    }

    /// Find the first block (top-left corner) of the transform that owns (bx, by).
    /// Returns (first_x, first_y, raw_strategy).
    #[allow(dead_code)] // used by `split_one_level` (gated behind `zensim-loop`)
    fn find_first_block(&self, bx: usize, by: usize) -> (usize, usize, u8) {
        if self.is_first(bx, by) {
            return (bx, by, self.raw_strategy(bx, by));
        }
        // The first block is at some position (fx, fy) where fx <= bx and fy <= by.
        // Walk up-left to find it. The strategy at (bx, by) tells us the raw strategy,
        // so we know the coverage. We need to find the top-left corner.
        let raw = self.raw_strategy(bx, by);
        let cx = COVERED_X[raw as usize];
        let cy = COVERED_Y[raw as usize];
        // The first block must be at an aligned position for this strategy.
        // For a transform covering cx×cy blocks, the first block (fx,fy) satisfies:
        //   fx <= bx < fx + cx  →  fx = bx - (bx % cx) if aligned
        //   fy <= by < fy + cy  →  fy = by - (by % cy) if aligned
        // But with non-aligned matching, alignment isn't guaranteed.
        // Instead, search backward.
        let min_fy = by.saturating_sub(cy - 1);
        for fy in (min_fy..=by).rev() {
            let min_fx = bx.saturating_sub(cx - 1);
            for fx in (min_fx..=bx).rev() {
                if self.is_first(fx, fy) && self.raw_strategy(fx, fy) == raw {
                    let fcx = COVERED_X[raw as usize];
                    let fcy = COVERED_Y[raw as usize];
                    if fx + fcx > bx && fy + fcy > by {
                        return (fx, fy, raw);
                    }
                }
            }
        }
        // Fallback: treat as single block
        (bx, by, raw)
    }

    /// Split a multi-block transform one level down at the given block position.
    ///
    /// Finds the owning transform and replaces it with the next smaller size:
    /// - DCT64x64/DCT64x32/DCT32x64 → DCT32x32
    /// - DCT32x32/DCT32x16/DCT16x32 → DCT16x16
    /// - DCT16x16/DCT16x8/DCT8x16 → DCT8
    ///
    /// Returns true if a split occurred (block was a multi-block transform).
    #[cfg(feature = "zensim-loop")]
    #[allow(dead_code)]
    pub fn split_one_level(&mut self, bx: usize, by: usize) -> bool {
        let (fx, fy, raw) = self.find_first_block(bx, by);
        let cx = COVERED_X[raw as usize];
        let cy = COVERED_Y[raw as usize];

        if cx <= 1 && cy <= 1 {
            return false;
        }

        let sub = match raw {
            RAW_STRATEGY_DCT64X64 | RAW_STRATEGY_DCT64X32 | RAW_STRATEGY_DCT32X64 => {
                RAW_STRATEGY_DCT32X32
            }
            RAW_STRATEGY_DCT32X32 | RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => {
                RAW_STRATEGY_DCT16X16
            }
            RAW_STRATEGY_DCT16X16 | RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => {
                RAW_STRATEGY_DCT8
            }
            _ => return false,
        };

        let sub_cx = COVERED_X[sub as usize];
        let sub_cy = COVERED_Y[sub as usize];

        for sy in (0..cy).step_by(sub_cy) {
            for sx in (0..cx).step_by(sub_cx) {
                self.set(fx + sx, fy + sy, sub);
            }
        }
        true
    }

    /// Check if a proposed `blocks × blocks` region at `(bx, by)` can be
    /// re-evaluated without breaking any existing larger transform.
    ///
    /// Returns true if it's safe to call `find_best_16x16_transform` (blocks=2)
    /// or `find_best_32x32_transform` (blocks=4) at this position.
    ///
    /// **W44-22 (F-D Sub-G fix Approach A, 2026-05-18):** Tightened to require
    /// that **every block within the region is a single 8×8 strategy** (one
    /// of: DCT8, DCT4×8, DCT8×4, DCT4×4, IDENTITY, DCT2×2, AFV0-3 — i.e.
    /// `COVERED_X == 1 && COVERED_Y == 1`).
    ///
    /// This is *stricter* than libjxl's `MultiBlockTransformCrossesBoundary`
    /// checks (`enc_ac_strategy.cc:725-741`), which only refuse merges that
    /// would orphan non-first markers, but it has a critical property: the
    /// `try_merge_16x16_impl` / `try_merge_32x32_impl` "reset region to DCT8"
    /// prologue (formerly load-bearing under the prior looser invariant) is
    /// now provably redundant, so non-winning halves preserve their per-block
    /// 8×8-class picks (DCT4×4, AFV0-3, DCT2×2, IDENTITY, DCT4×8, DCT8×4)
    /// instead of being clobbered to plain DCT8.
    ///
    /// Cost: a region containing an existing fully-contained multi-block
    /// transform (e.g. an aligned-pass DCT16X16 sitting at the top-left of a
    /// 32×32 try_merge candidate) is no longer re-evaluated. Those regions
    /// already have a strategy that won an entropy comparison; the lost
    /// re-evaluation is small. See `f_d_sub_chunk_g_try_merge_2026-05-18.md`
    /// for full audit and approach (B) (per-half inner-crossing port).
    pub(super) fn can_evaluate_region(&self, bx: usize, by: usize, blocks: usize) -> bool {
        for dy in 0..blocks {
            for dx in 0..blocks {
                let x = bx + dx;
                let y = by + dy;
                if x >= self.xsize_blocks || y >= self.ysize_blocks {
                    return false;
                }
                // Require single-block strategy: this guarantees no existing
                // multi-block transform straddles or sits within the region,
                // so try_merge can paint over without destroying orphaned
                // non-first markers.
                let raw = self.raw_strategy(x, y);
                if COVERED_X[raw as usize] != 1 || COVERED_Y[raw as usize] != 1 {
                    return false;
                }
            }
        }
        true
    }

    /// Outer-only counterpart of `can_evaluate_region` mirroring libjxl's
    /// `FindBestFirstLevelDivisionForSquare` precondition
    /// (`enc_ac_strategy.cc:725-732`).
    ///
    /// Returns true if no existing multi-block transform crosses any of the
    /// four outer edges of the `blocks × blocks` region at `(bx, by)`. Unlike
    /// `can_evaluate_region`, this DOES permit a multi-block transform to sit
    /// fully inside the region — libjxl handles that case at the per-half
    /// inner-crossing layer instead (see
    /// `multi_block_crosses_vertical_boundary` /
    /// `multi_block_crosses_horizontal_boundary`).
    ///
    /// W44-23 (Sub-G Approach B, 2026-05-18): used by `try_merge_*_impl` to
    /// recover the merge attempts W44-22's tighter invariant refused.
    pub(super) fn can_evaluate_region_outer_only(
        &self,
        bx: usize,
        by: usize,
        blocks: usize,
    ) -> bool {
        if bx + blocks > self.xsize_blocks || by + blocks > self.ysize_blocks {
            return false;
        }
        // Four outer edges: top, bottom, left, right. A transform "crosses"
        // an edge iff a block on the inside is a non-first marker whose
        // owner sits on the outside (or vice versa). Mirrors the four
        // calls at libjxl `enc_ac_strategy.cc:725-732` (two horizontal, two
        // vertical) — note libjxl's helpers short-circuit on `% 8 == 0`
        // (64×64 boundary). We do the same here for parity with the JXL
        // pass-group invariant.
        if self.multi_block_crosses_horizontal_boundary(bx, by, bx + blocks) {
            return false;
        }
        if self.multi_block_crosses_horizontal_boundary(bx, by + blocks, bx + blocks) {
            return false;
        }
        if self.multi_block_crosses_vertical_boundary(bx, by, by + blocks) {
            return false;
        }
        if self.multi_block_crosses_vertical_boundary(bx + blocks, by, by + blocks) {
            return false;
        }
        true
    }

    /// True iff a multi-block transform straddles the horizontal scanline at
    /// `y` between `start_x` and `end_x` (exclusive).
    ///
    /// Port of libjxl `MultiBlockTransformCrossesHorizontalBoundary`
    /// (`enc_ac_strategy.cc:302-330`). Short-circuits on `y % 8 == 0`
    /// because nothing crosses a 64×64 boundary (pass-group invariant).
    /// Walks left from `start_x` to the nearest 8-aligned position to find
    /// the first-block of a transform that might start before `start_x`,
    /// then scans the row from there: any non-first block in `[start_x,
    /// end_x)` means a transform crosses this row.
    pub(super) fn multi_block_crosses_horizontal_boundary(
        &self,
        mut start_x: usize,
        y: usize,
        end_x: usize,
    ) -> bool {
        if start_x >= self.xsize_blocks || y >= self.ysize_blocks {
            return false;
        }
        if y.is_multiple_of(8) {
            // Nothing crosses 64×64 boundaries (pass-group invariant).
            return false;
        }
        let end_x = end_x.min(self.xsize_blocks);
        // Back up to the first IsFirstBlock() == true within the 8-aligned
        // window so we correctly count a transform whose first block is
        // before start_x.
        let start_x_limit = start_x & !7;
        while start_x != start_x_limit && !self.is_first(start_x, y) {
            start_x -= 1;
        }
        let mut x = start_x;
        while x < end_x {
            if self.is_first(x, y) {
                let cx = COVERED_X[self.raw_strategy(x, y) as usize];
                x += cx.max(1);
            } else {
                return true;
            }
        }
        false
    }

    /// True iff a multi-block transform straddles the vertical scanline at
    /// `x` between `start_y` and `end_y` (exclusive).
    ///
    /// Port of libjxl `MultiBlockTransformCrossesVerticalBoundary`
    /// (`enc_ac_strategy.cc:332-362`). Same algorithm as the horizontal
    /// helper with rows and columns swapped.
    pub(super) fn multi_block_crosses_vertical_boundary(
        &self,
        x: usize,
        mut start_y: usize,
        end_y: usize,
    ) -> bool {
        if x >= self.xsize_blocks || start_y >= self.ysize_blocks {
            return false;
        }
        if x.is_multiple_of(8) {
            return false;
        }
        let end_y = end_y.min(self.ysize_blocks);
        let start_y_limit = start_y & !7;
        while start_y != start_y_limit && !self.is_first(x, start_y) {
            start_y -= 1;
        }
        let mut y = start_y;
        while y < end_y {
            if self.is_first(x, y) {
                let cy = COVERED_Y[self.raw_strategy(x, y) as usize];
                y += cy.max(1);
            } else {
                return true;
            }
        }
        false
    }

    /// Count the number of "first blocks" (= number of distinct transforms).
    #[cfg(test)]
    pub fn count_first_blocks(&self) -> usize {
        self.data.iter().filter(|&&v| (v & 1) != 0).count()
    }

    /// Copy a rectangular region from `src` into `self`.
    /// Copies blocks from `(start_bx, start_by)` to `(end_bx, end_by)` exclusive.
    pub(crate) fn copy_region_from(
        &mut self,
        src: &AcStrategyMap,
        start_bx: usize,
        start_by: usize,
        end_bx: usize,
        end_by: usize,
    ) {
        debug_assert_eq!(self.xsize_blocks, src.xsize_blocks);
        for by in start_by..end_by {
            let row_start = by * self.xsize_blocks;
            self.data[row_start + start_bx..row_start + end_bx]
                .copy_from_slice(&src.data[row_start + start_bx..row_start + end_bx]);
        }
    }

    /// Return strategy histogram indexed by raw strategy code (0..19).
    /// Counts first blocks only (number of times each transform was selected).
    pub fn strategy_histogram(&self) -> [u32; 19] {
        let mut counts = [0u32; 19];
        for &v in &self.data {
            if (v & 1) != 0 {
                // is_first block
                let raw = (v >> 1) as usize;
                if raw < 19 {
                    counts[raw] += 1;
                }
            }
        }
        counts
    }

    /// Print strategy histogram with names.
    #[cfg(feature = "debug-ac-strategy")]
    pub fn print_histogram(&self) {
        const NAMES: [&str; 19] = [
            "DCT8", "DCT16x8", "DCT8x16", "DCT16x16", "DCT32x32", "DCT4x8", "DCT8x4", "DCT4x4",
            "IDENTITY", "DCT2X2", "DCT32x16", "DCT16x32", "AFV0", "AFV1", "AFV2", "AFV3",
            "DCT64x64", "DCT64x32", "DCT32x64",
        ];
        let hist = self.strategy_histogram();
        let total: u32 = hist.iter().sum();
        eprintln!("Strategy histogram (total {} transforms):", total);
        for (i, &count) in hist.iter().enumerate() {
            if count > 0 {
                let pct = 100.0 * count as f64 / total as f64;
                eprintln!("  {:10}: {:6} ({:5.1}%)", NAMES[i], count, pct);
            }
        }
    }
}

// ─── Entropy estimation ─────────────────────────────────────────────────────

/// Channel offsets for pixel-domain loss masking.
/// From libjxl enc_ac_strategy.cc:446
const MASK_CHANNEL_OFFSET: [f32; 3] = [12.0, 0.0, 4.0];

/// Channel multipliers for pixel-domain loss (8th power).
/// From libjxl enc_ac_strategy.cc:479
/// Pre-computed: 8.2^8 ≈ 2.088e7, 1.0^8 = 1.0, 1.03^8 ≈ 1.267
const CHANNEL_MUL: [f64; 3] = [
    20882706.4655936, // X channel: 8.2^8
    1.0,              // Y channel: 1.0^8
    1.26677008064,    // B channel: 1.03^8
];

/// Distance scaling exponents from libjxl enc_ac_strategy.cc:1115-1120
const K_BIAS: f32 = 0.137_317_43;
const K_POW_INFO_LOSS: f32 = 0.336_778_07;
const K_POW_ZEROS_MUL: f32 = 0.509_909_3;
const K_POW_COST_DELTA: f32 = 0.367_029_4;

/// Constants for coefficient-domain mode (libjxl-tiny style, not distance-scaled).
/// Order: (info_loss_mul, cost_delta, zeros_mul) — matches compute_scaled_constants output.
pub(super) const COEFF_DOMAIN_CONSTANTS: (f32, f32, f32) = (138.0, 5.335_918_5, 7.565_053_4);

/// Compute distance-scaled constants for full libjxl cost model.
/// At d=1.0, returns the base values. At higher distances, increases all values.
///
/// `bases` is `(info_loss_mul_base, zeros_mul_base, cost_delta_base)` from
/// [`EffortProfile`](crate::effort::EffortProfile).
///
/// Returns `(info_loss_mul, cost_delta, zeros_mul)`.
///
/// Call this ONCE per search function (not per estimate_entropy call) since
/// distance and bases are constant within a search. Pass the result as
/// `scaled_constants` to estimate_entropy_with_mask/estimate_entropy_full.
// Visibility note: bumped from pub(super) to pub(crate) so the
// `__internals` cargo feature's free-function wrapper
// `compute_scaled_constants_free` (below) can call it.
pub(crate) fn compute_scaled_constants(distance: f32, bases: (f32, f32, f32)) -> (f32, f32, f32) {
    let (info_loss_base, zeros_base, cost_delta_base) = bases;
    let ratio = (distance + K_BIAS) / (1.0 + K_BIAS);
    let info_loss_mul = info_loss_base * jxl_simd::fast_powf(ratio, K_POW_INFO_LOSS);
    let zeros_mul = zeros_base * jxl_simd::fast_powf(ratio, K_POW_ZEROS_MUL);
    let cost_delta = cost_delta_base * jxl_simd::fast_powf(ratio, K_POW_COST_DELTA);
    (info_loss_mul, cost_delta, zeros_mul)
}

/// Free-function wrapper around `compute_scaled_constants` for the
/// `__internals` cargo feature's downstream parity testing.
#[cfg(feature = "__internals")]
#[doc(hidden)]
pub fn compute_scaled_constants_free(distance: f32, bases: (f32, f32, f32)) -> (f32, f32, f32) {
    compute_scaled_constants(distance, bases)
}

use crate::effort::EntropyMulTable;

/// Get the entropy multiplier for a raw strategy from the given table.
///
/// CRITICAL: libjxl only normalizes 8x8 transforms in FindBest8x8Transform.
/// Larger transforms use RAW values in TryMergeAcs.
///
/// 8x8 transforms (normalized by table.dct8):
/// - DCT8: always 1.0 (self-normalized)
/// - DCT4X8: table.dct4x8 / table.dct8
/// - DCT4X4: table.dct4x4 / table.dct8
///
/// Larger transforms (RAW values, NOT normalized):
/// - DCT16X8: table.dct16x8
/// - DCT16X16: table.dct16x16
/// - DCT32X32: table.dct32x32
pub(super) fn entropy_mul_for_strategy(raw_strategy: u8, table: &EntropyMulTable) -> f32 {
    match raw_strategy {
        // 8x8 transforms: normalize by table.dct8 (so DCT8 = 1.0)
        RAW_STRATEGY_DCT8 => 1.0,
        RAW_STRATEGY_DCT4X8 | RAW_STRATEGY_DCT8X4 => table.dct4x8 / table.dct8,
        RAW_STRATEGY_DCT4X4 => table.dct4x4 / table.dct8,
        RAW_STRATEGY_IDENTITY => table.identity / table.dct8,
        RAW_STRATEGY_DCT2X2 => table.dct2x2 / table.dct8,
        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
            table.afv / table.dct8
        }
        // Larger transforms: use RAW values (libjxl TryMergeAcs uses raw entropy_mul)
        RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => table.dct16x8,
        RAW_STRATEGY_DCT16X16 => table.dct16x16,
        RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => table.dct16x32,
        RAW_STRATEGY_DCT32X32 => table.dct32x32,
        RAW_STRATEGY_DCT64X32 | RAW_STRATEGY_DCT32X64 => table.dct64x32,
        RAW_STRATEGY_DCT64X64 => table.dct64x64,
        _ => 1.0,
    }
}

/// Estimate entropy using coefficient-domain loss (libjxl-tiny style).
///
/// This is a convenience wrapper that calls `estimate_entropy_with_mask` with
/// `mask1x1 = None`, for backward compatibility with tests and code that
/// doesn't need pixel-domain loss.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) fn estimate_entropy(
    raw_strategy: u8,
    xyb: [&[f32]; 3],
    stride: usize,
    bx: usize,
    by: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
) -> f32 {
    let mut scratch = EntropyEstScratch::new();
    let table = EntropyMulTable::reference();
    estimate_entropy_with_mask(
        raw_strategy,
        xyb,
        stride,
        bx,
        by,
        distance,
        quant_field,
        xsize_blocks,
        masking,
        ytox,
        ytob,
        None,
        0,
        0.0,
        COEFF_DOMAIN_CONSTANTS,
        &table,
        &mut scratch,
    )
}

/// Estimate entropy with optional pixel-domain loss.
///
/// When `mask1x1` is Some, uses full libjxl pixel-domain loss model with:
/// - Pre-computed distance-scaled constants (passed as `scaled_constants`)
/// - Fixed entropy multiplier per transform type
///
/// When `mask1x1` is None, uses coefficient-domain loss (libjxl-tiny style).
///
/// `entropy_mul_adjust`: additive adjustment to entropy_mul. In libjxl,
/// kFavor2X2AtHighQuality and kAvoidEntropyOfTransforms modify entropy_mul
/// before passing to EstimateEntropy. Pass 0.0 for no adjustment.
///
/// `scaled_constants`: pre-computed `(info_loss_mul, cost_delta, zeros_mul)` from
/// `compute_scaled_constants(distance, bases)` for pixel-domain mode, or
/// `COEFF_DOMAIN_CONSTANTS` for coefficient-domain mode.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn estimate_entropy_with_mask(
    raw_strategy: u8,
    xyb: [&[f32]; 3],
    stride: usize,
    bx: usize,
    by: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    entropy_mul_adjust: f32,
    scaled_constants: (f32, f32, f32),
    entropy_mul_table: &EntropyMulTable,
    scratch: &mut EntropyEstScratch,
) -> f32 {
    // In pixel-domain mode, use fixed entropy_mul values per transform
    // In coefficient-domain mode, entropy_mul is applied outside by the caller (mul8x8 etc.)
    let entropy_mul = if mask1x1.is_some() {
        (entropy_mul_for_strategy(raw_strategy, entropy_mul_table) + entropy_mul_adjust).max(0.01)
    } else {
        // Coefficient-domain: entropy_mul is 1.0, caller handles multiplier.
        // Still apply adjustment for kFavor2X2/kAvoidEntropy since the
        // returned estimate gets multiplied by the caller's multiplier.
        (1.0 + entropy_mul_adjust).max(0.01)
    };

    estimate_entropy_full(
        raw_strategy,
        xyb,
        stride,
        bx,
        by,
        distance,
        quant_field,
        xsize_blocks,
        masking,
        ytox,
        ytob,
        mask1x1,
        mask1x1_stride,
        entropy_mul,
        scaled_constants,
        scratch,
    )
}

/// Estimate entropy with optional pixel-domain loss calculation.
///
/// When `mask1x1` is Some, uses full libjxl pixel-domain loss model.
/// When `mask1x1` is None, uses coefficient-domain loss (libjxl-tiny style).
///
/// `entropy_mul` multiplies ONLY the entropy part, not the loss. In full libjxl
/// mode, this is a fixed value per transform type. In libjxl-tiny mode, this
/// is 1.0 and the caller applies multipliers externally.
///
/// `scaled_constants`: pre-computed `(info_loss_mul, cost_delta, zeros_mul)`.
/// Use `compute_scaled_constants()` for pixel-domain, `COEFF_DOMAIN_CONSTANTS`
/// for coefficient-domain.
///
/// This function does NOT dispatch to SIMD — it relies on the CALLER being in
/// a `#[target_feature]` context (e.g., via `#[arcane]` on the search functions).
/// SIMD dispatch is hoisted to the search function level to avoid 50K+ redundant
/// `summon()` calls per image (one per estimate_entropy invocation → ~2K per
/// search function invocation).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn estimate_entropy_full(
    raw_strategy: u8,
    xyb: [&[f32]; 3],
    stride: usize,
    bx: usize,
    by: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    entropy_mul: f32,
    scaled_constants: (f32, f32, f32),
    scratch: &mut EntropyEstScratch,
) -> f32 {
    estimate_entropy_full_impl(
        raw_strategy,
        xyb,
        stride,
        bx,
        by,
        distance,
        quant_field,
        xsize_blocks,
        masking,
        ytox,
        ytob,
        mask1x1,
        mask1x1_stride,
        entropy_mul,
        scaled_constants,
        scratch,
    )
}

// NOTE: estimate_entropy_full_avx2 and estimate_entropy_full_neon were removed.
// SIMD dispatch is now hoisted to the strategy search function level (ac_strategy_search.rs).
// The search functions are #[arcane], and estimate_entropy_with_mask + estimate_entropy_full
// are #[inline(always)] — when called from an arcane context, the entire chain is inlined
// and LLVM can optimize the jxl_simd dispatch + intrinsics under target_feature.

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn estimate_entropy_full_impl(
    raw_strategy: u8,
    xyb: [&[f32]; 3],
    stride: usize,
    bx: usize,
    by: usize,
    distance: f32,
    quant_field: &[f32],
    xsize_blocks: usize,
    masking: &[f32],
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    entropy_mul: f32,
    scaled_constants: (f32, f32, f32),
    scratch: &mut EntropyEstScratch,
) -> f32 {
    let cx = COVERED_X[raw_strategy as usize];
    let cy = COVERED_Y[raw_strategy as usize];
    let num_blocks = cx * cy;
    let size = num_blocks * DCT_BLOCK_SIZE;

    // Use different constants based on whether we're using pixel-domain loss
    let use_pixel_domain = mask1x1.is_some();

    // Pre-computed constants: pixel-domain uses compute_scaled_constants(),
    // coefficient-domain uses COEFF_DOMAIN_CONSTANTS.
    let (k_info_loss_mul, k_cost_delta, k_zeros_mul) = scaled_constants;
    const K_INFO_LOSS_MULTIPLIER2: f32 = 50.468_4;
    const K_COST2: f32 = 4.462_815;

    // Fused DCT8 + entropy path: eliminates intermediate scratch.block writes.
    // DCT output stays in SIMD registers (AVX2) or a local array (scalar), with
    // only error_coeffs written out (needed for IDCT pixel-domain loss).
    if raw_strategy == RAW_STRATEGY_DCT8 && use_pixel_domain {
        let quant_for_coeffs = quant_field[by * xsize_blocks + bx];
        let cmap_factors = [ytox_ratio(ytox), 0.0f32, ytob_ratio(ytob)];

        let mut y_dct = [0.0f32; 64];
        let mut entropy = 0.0f32;
        let mut total_pixel_loss = 0.0f64;

        let error_coeffs: &mut [f32; 64] = as_array_mut(&mut scratch.error_coeffs, 0);
        let pixel_error_buf = &mut scratch.pixel_error[..64];
        let mask = mask1x1.unwrap();
        let pixel_x = bx * BLOCK_DIM;
        let pixel_y = by * BLOCK_DIM;
        let mask_row_base = pixel_y * mask1x1_stride + pixel_x;

        // Process per-channel entropy + pixel-domain loss via inline closure
        let mut process_channel =
            |c: usize, y_dct_ref: &[f32; 64], dct_out: Option<&mut [f32; 64]>| {
                let weights: &[f32; 64] = quant_weights(RAW_STRATEGY_DCT8 as usize, c)
                    .try_into()
                    .unwrap();
                let inv_wts: &[f32; 64] = dequant_weights(RAW_STRATEGY_DCT8 as usize, c)
                    .try_into()
                    .unwrap();

                // W44-9 Sub-chunk B: optionally route through the explicit
                // non-fused fallback (separate DCT + entropy) instead of the
                // fused AVX2 kernel. Default is the fused path.
                let coeff_result = if force_unfused_dct8_entropy() {
                    jxl_simd::fused_dct8_entropy_fallback(
                        xyb[c],
                        stride,
                        bx,
                        by,
                        y_dct_ref,
                        weights,
                        inv_wts,
                        cmap_factors[c],
                        quant_for_coeffs,
                        k_cost_delta,
                        error_coeffs,
                        dct_out,
                    )
                } else {
                    jxl_simd::fused_dct8_entropy(
                        xyb[c],
                        stride,
                        bx,
                        by,
                        y_dct_ref,
                        weights,
                        inv_wts,
                        cmap_factors[c],
                        quant_for_coeffs,
                        k_cost_delta,
                        error_coeffs,
                        dct_out,
                    )
                };

                entropy += coeff_result.entropy_sum;
                let num_nzeros = coeff_result.nzeros_sum as usize;
                let nbits = ceil_log2_nonzero(num_nzeros + 1) as usize + 1;
                entropy += k_zeros_mul * (ceil_log2_nonzero(nbits + 17) + nbits as u32) as f32;

                apply_idct_for_strategy(RAW_STRATEGY_DCT8, error_coeffs, pixel_error_buf);
                let mask_offset = MASK_CHANNEL_OFFSET[c];
                let mut channel_loss = jxl_simd::pixel_domain_loss(
                    pixel_error_buf,
                    mask,
                    mask_row_base,
                    mask1x1_stride,
                    mask_offset,
                    BLOCK_DIM,
                    BLOCK_DIM,
                );
                channel_loss *= CHANNEL_MUL[c];
                total_pixel_loss += channel_loss;
            };

        // Y channel first (stores DCT for CfL reference).
        // y_dct is all zeros here, and cmap_factor=0.0, so the CfL term is a no-op.
        let dummy_y = [0.0f32; 64];
        process_channel(1, &dummy_y, Some(&mut y_dct));

        // X and B channels use the stored Y DCT for CfL decorrelation.
        process_channel(0, &y_dct, None);
        process_channel(2, &y_dct, None);

        // Final cost: entropy * entropy_mul + loss
        let n = DCT_BLOCK_SIZE as f64;
        let loss_scalar = (total_pixel_loss / n).sqrt().sqrt().sqrt() * n / quant_for_coeffs as f64;
        let entropy_pre_loss_dct8 = entropy * entropy_mul;
        entropy *= entropy_mul;
        entropy += k_info_loss_mul * loss_scalar as f32;
        // W44-58 / W44-59 fix: dump using libjxl wire strategy code (DCT8=0).
        dump_cost_inputs(
            STRATEGY_CODE_LUT[raw_strategy as usize], bx, by, entropy_mul, 1, 1, quant_for_coeffs,
            ytox, ytob, mask1x1, mask1x1_stride,
            entropy_pre_loss_dct8, loss_scalar as f32,
            k_info_loss_mul, k_cost_delta, k_zeros_mul, entropy,
        );
        return entropy;
    }

    // Use pre-allocated scratch buffers (no fill needed — transforms overwrite all positions)
    let block = &mut scratch.block[..3 * size];

    // For single-block strategies, cache extracted 8×8 pixels across strategy calls.
    // The same block position is evaluated with 10+ strategies in find_best_16x16_transform;
    // extracting once and reusing saves ~90% of extract_block_8x8 overhead.
    let is_single_block = num_blocks == 1;
    if is_single_block && scratch.pixels_8x8_pos != (bx, by) {
        for (c, xyb_c) in xyb.iter().enumerate() {
            extract_block_8x8(xyb_c, stride, bx, by, &mut scratch.pixels_8x8[c]);
        }
        scratch.pixels_8x8_pos = (bx, by);
    }

    for (c, xyb_c) in xyb.iter().enumerate() {
        let offset = c * size;
        match raw_strategy {
            // Single-block strategies: use cached pixels_8x8
            RAW_STRATEGY_DCT8 => {
                dct_8x8(&scratch.pixels_8x8[c], as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT4X8 => {
                dct_4x8_full(&scratch.pixels_8x8[c], as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT8X4 => {
                dct_8x4_full(&scratch.pixels_8x8[c], as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT4X4 => {
                dct_4x4_full(&scratch.pixels_8x8[c], as_array_mut(block, offset));
            }
            RAW_STRATEGY_IDENTITY => {
                identity_transform(&scratch.pixels_8x8[c], as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT2X2 => {
                dct2x2_transform(&scratch.pixels_8x8[c], as_array_mut(block, offset));
            }
            RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
                let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
                afv_transform_from_pixels(
                    &scratch.pixels_8x8[c],
                    afv_kind,
                    as_array_mut(block, offset),
                );
            }
            // Multi-block strategies: extract fresh pixels (different block sizes)
            RAW_STRATEGY_DCT16X8 => {
                let mut input = uninit_buf::<128>();
                extract_block_8x16(xyb_c, stride, bx, by, &mut input);
                dct_16x8(&input, as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT8X16 => {
                let mut input = uninit_buf::<128>();
                extract_block_16x8(xyb_c, stride, bx, by, &mut input);
                dct_8x16(&input, as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT16X16 => {
                let mut input = uninit_buf::<256>();
                extract_block_16x16(xyb_c, stride, bx, by, &mut input);
                dct_16x16(&input, as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT32X32 => {
                let mut input = uninit_buf::<1024>();
                extract_block_32x32(xyb_c, stride, bx, by, &mut input);
                dct_32x32(&input, as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT32X16 => {
                let mut input = uninit_buf::<512>();
                extract_block_32x16(xyb_c, stride, bx, by, &mut input);
                dct_32x16(&input, as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT16X32 => {
                let mut input = uninit_buf::<512>();
                extract_block_16x32(xyb_c, stride, bx, by, &mut input);
                dct_16x32(&input, as_array_mut(block, offset));
            }
            RAW_STRATEGY_DCT64X64 => {
                let mut input = uninit_buf::<4096>();
                extract_block_64x64(xyb_c, stride, bx, by, &mut input);
                dct_64x64(&input, &mut block[offset..offset + 4096]);
            }
            RAW_STRATEGY_DCT64X32 => {
                let mut input = uninit_buf::<2048>();
                extract_block_64x32(xyb_c, stride, bx, by, &mut input);
                dct_64x32(&input, &mut block[offset..offset + 2048]);
            }
            RAW_STRATEGY_DCT32X64 => {
                let mut input = uninit_buf::<2048>();
                extract_block_32x64(xyb_c, stride, bx, by, &mut input);
                dct_32x64(&input, &mut block[offset..offset + 2048]);
            }
            _ => unreachable!(),
        }
    }

    // Zero the LLF (lowest-frequency) positions in the block data for all channels.
    // libjxl zeros these positions in inv_table (quant_weights.cc:342-355) so they
    // contribute nothing to entropy or loss estimates. The DC/LLF coefficients are
    // overwritten by LowestFrequenciesFromDC during decoding, so their quantization
    // cost is irrelevant for strategy selection. Without this, different strategies
    // have wildly different DC dequant weights (e.g., DCT8 Y=560 vs AFV Y=58),
    // creating phantom entropy differences that bias AFV selection.
    //
    // The LLF region uses the transposed layout (cx_t >= cy_t, stride = cx_t * 8),
    // matching libjxl's swap(cx, cy) convention. Non-square DCTs always output in
    // the layout where the longer dimension is the stride (e.g., both DCT16X8 and
    // DCT8X16 output in 8×16 layout with stride 16). The LLF positions are in the
    // top-left corner of this layout.
    {
        let (cx_t, cy_t) = if cy > cx { (cy, cx) } else { (cx, cy) };
        let llf_stride = cx_t * BLOCK_DIM;
        for c in 0..3 {
            let offset = c * size;
            for iy in 0..cy_t {
                for ix in 0..cx_t {
                    block[offset + iy * llf_stride + ix] = 0.0;
                }
            }
        }
    }

    // Load QF and masking: take max over covered blocks
    let mut quant = 0.0f32;
    let mut mask_val = 0.0f32;
    for iy in 0..cy {
        for ix in 0..cx {
            let idx = (by + iy) * xsize_blocks + bx + ix;
            quant = quant.max(quant_field[idx]);
            mask_val = mask_val.max(masking[idx]);
        }
    }

    // Compute quant_norm16 for pixel-domain loss
    // libjxl uses different computation based on block count:
    // - 1 block (DCT8): single quant value
    // - 2 blocks (DCT16x8, DCT8x16): MAX of the two quant values
    // - 4+ blocks (DCT16x16, DCT32x32): 16th norm
    // Reference: lib/jxl/enc_ac_strategy.cc:383-410
    let quant_norm16 = if use_pixel_domain {
        if num_blocks == 1 {
            // Single block: use the quant value directly
            quant_field[by * xsize_blocks + bx]
        } else if num_blocks == 2 {
            // Two blocks: use MAX of the two quant values (NOT 16th norm!)
            let q1 = quant_field[by * xsize_blocks + bx];
            let q2 = if cy == 2 {
                // DCT8x16: blocks are vertically stacked
                quant_field[(by + 1) * xsize_blocks + bx]
            } else {
                // DCT16x8: blocks are horizontally adjacent
                quant_field[by * xsize_blocks + bx + 1]
            };
            q1.max(q2)
        } else {
            // 4+ blocks: use 16th norm
            let mut norm_sum = 0.0f32;
            for iy in 0..cy {
                for ix in 0..cx {
                    let idx = (by + iy) * xsize_blocks + bx + ix;
                    let qval = quant_field[idx];
                    // qval^16 = (qval^2)^8
                    let q2 = qval * qval;
                    let q4 = q2 * q2;
                    let q8 = q4 * q4;
                    let q16 = q8 * q8;
                    norm_sum += q16;
                }
            }
            norm_sum /= num_blocks as f32;
            // x^(1/16) = sqrt(sqrt(sqrt(sqrt(x))))
            norm_sum.sqrt().sqrt().sqrt().sqrt()
        }
    } else {
        0.0 // Not used in coefficient-domain mode
    };

    let cmap_factors = [ytox_ratio(ytox), 0.0f32, ytob_ratio(ytob)];

    let mut entropy = 0.0f32;
    let mut info_loss_sum = 0.0f32;
    let mut info_loss2_sum = 0.0f32;

    // For pixel-domain loss: accumulate loss across all channels
    let mut total_pixel_loss = 0.0f64;

    // Error coefficient buffer for pixel-domain IDCT (reused per channel,
    // no fill needed — entropy_estimate_coeffs writes all positions)
    let error_coeffs = &mut scratch.error_coeffs[..size];

    let slope = (distance / 3.0).min(1.0);
    let cost_of_1 = 1.0 + slope * 8.870_325;

    // Pixel base coordinates
    let pixel_x = bx * BLOCK_DIM;
    let pixel_y = by * BLOCK_DIM;

    // Pre-compute weight slices for all 3 channels (avoids per-channel match dispatch).
    // quant_weights/dequant_weights go through a 19-arm match that the compiler doesn't
    // inline; hoisting saves 4 match dispatches per call × 248K calls.
    let strat_idx = raw_strategy as usize;
    let full_quant_weights = quant_weights_full(strat_idx);
    let full_dequant_weights = dequant_weights_full(strat_idx);
    let per_ch_size = super::quant::WEIGHT_SIZES[strat_idx];

    // Extract mask once to avoid per-channel Option pattern-match
    let mask_row_base = if use_pixel_domain {
        pixel_y * mask1x1_stride + pixel_x
    } else {
        0
    };

    // Pre-split block into per-channel slices so the compiler can prove all
    // accesses within each slice are in-bounds, eliminating per-channel range checks.
    let (block_ch0, block_rest) = block.split_at(size);
    let (block_ch1, block_ch2) = block_rest.split_at(size);
    let block_channels = [block_ch0, block_ch1, block_ch2];

    // Pre-split weight tables into per-channel slices (3 × per_ch_size).
    let weight_channels = [
        &full_quant_weights[..per_ch_size],
        &full_quant_weights[per_ch_size..2 * per_ch_size],
        &full_quant_weights[2 * per_ch_size..3 * per_ch_size],
    ];
    let inv_weight_channels = [
        &full_dequant_weights[..per_ch_size],
        &full_dequant_weights[per_ch_size..2 * per_ch_size],
        &full_dequant_weights[2 * per_ch_size..3 * per_ch_size],
    ];

    let quant_for_coeffs = if use_pixel_domain {
        quant_norm16
    } else {
        quant
    };

    // W44-59: env-var-driven per-coefficient AFV dump. Activates only when
    // `JXL_DUMP_AFV_PATH` is set AND (raw_strategy, bx*8, by*8) matches
    // `JXL_DUMP_AFV_KIND` (0..3 → AFV0..3), `JXL_DUMP_AFV_X`, `JXL_DUMP_AFV_Y`.
    // Emits long-format rows: stage TAB c TAB i TAB value.
    // Stages: "pixels" (8x8 input per channel), "coeffs" (transform output),
    //         "inv_weights" (dequant matrix), "weights" (quant matrix),
    //         "per_coeff_val/rval/diff/error" (entropy estimator stage).
    // ZERO overhead when env var unset (OnceLock + thread-local matched-block flag).
    let afv_coeff_target =
        afv_coeff_dump_target(raw_strategy, bx * BLOCK_DIM, by * BLOCK_DIM);
    if afv_coeff_target {
        for c in 0..3 {
            dump_afv_coeff_block(
                "pixels",
                raw_strategy,
                bx * BLOCK_DIM,
                by * BLOCK_DIM,
                c,
                &scratch.pixels_8x8[c],
            );
            dump_afv_coeff_block(
                "coeffs",
                raw_strategy,
                bx * BLOCK_DIM,
                by * BLOCK_DIM,
                c,
                block_channels[c],
            );
            dump_afv_coeff_block(
                "inv_weights",
                raw_strategy,
                bx * BLOCK_DIM,
                by * BLOCK_DIM,
                c,
                inv_weight_channels[c],
            );
            dump_afv_coeff_block(
                "weights",
                raw_strategy,
                bx * BLOCK_DIM,
                by * BLOCK_DIM,
                c,
                weight_channels[c],
            );
        }
    }

    for (c, &cmap_factor) in cmap_factors.iter().enumerate() {
        // SIMD-accelerated coefficient processing (biggest encoder hotspot).
        // LLF positions are pre-zeroed above (matching libjxl quant_weights.cc:342-355),
        // so DC/LLF contribute nothing to entropy or loss estimates.
        //
        // In pixel-domain mode: use quant_norm16 (L16 norm for 4+ blocks, max for
        // 1-2 blocks) matching libjxl enc_ac_strategy.cc:415.
        // In coefficient-domain mode: use max(quant_field) (libjxl-tiny style).
        let coeff_result = jxl_simd::entropy_estimate_coeffs(
            block_channels[c],
            block_channels[1], // Y channel always channel 1
            weight_channels[c],
            inv_weight_channels[c],
            size,
            cmap_factor,
            quant_for_coeffs,
            k_cost_delta,
            K_COST2,
            use_pixel_domain,
            error_coeffs,
        );
        let mut entropy_sum = coeff_result.entropy_sum;
        let nzeros_sum = coeff_result.nzeros_sum;
        if !use_pixel_domain {
            info_loss_sum += coeff_result.info_loss_sum;
            info_loss2_sum += coeff_result.info_loss2_sum;
        }
        // cost_of_1 term only in coefficient-domain mode (libjxl-tiny style)
        // Full libjxl pixel-domain mode doesn't have this per-nzero term
        if !use_pixel_domain {
            entropy_sum += nzeros_sum * cost_of_1;
        }
        entropy += entropy_sum;

        // W44-59: per-coefficient (val, rval, diff) dump for the target AFV block.
        // Replays the entropy_estimate_coeffs scalar formula so we can compare
        // bit-exact per-coeff values vs libjxl's `Round/Sub/Mul` SIMD output.
        if afv_coeff_target {
            for i in 0..size {
                let in_val = block_channels[c][i];
                let in_y = block_channels[1][i] * cmap_factor;
                let im = inv_weight_channels[c][i];
                let val = (in_val - in_y) * (im * quant_for_coeffs);
                let rval = val.round_ties_even();
                let diff = val - rval;
                let m = weight_channels[c][i];
                let weighted_diff = m * diff;
                dump_afv_per_coeff(
                    raw_strategy,
                    bx * BLOCK_DIM,
                    by * BLOCK_DIM,
                    c,
                    i,
                    val,
                    rval,
                    diff,
                    weighted_diff,
                );
            }
            dump_afv_aggregate(
                raw_strategy,
                bx * BLOCK_DIM,
                by * BLOCK_DIM,
                c,
                "entropy_sum",
                entropy_sum,
            );
            dump_afv_aggregate(
                raw_strategy,
                bx * BLOCK_DIM,
                by * BLOCK_DIM,
                c,
                "nzeros_sum",
                nzeros_sum,
            );
        }

        let num_nzeros = nzeros_sum as usize;
        let nbits = ceil_log2_nonzero(num_nzeros + 1) as usize + 1;
        entropy += k_zeros_mul * (ceil_log2_nonzero(nbits + 17) + nbits as u32) as f32;

        // X channel penalty for large transforms
        if c == 0 && num_blocks >= 2 && use_pixel_domain {
            let w = 1.0 + (num_blocks as f32 / 8.0).min(3.0);
            entropy *= w;
        }

        // Pixel-domain loss calculation
        if use_pixel_domain {
            // mask1x1 is guaranteed Some when use_pixel_domain is true
            let mask = mask1x1.unwrap();

            // Apply IDCT to error coefficients to get pixel-domain error
            let pixel_error_buf = &mut scratch.pixel_error[..size];
            apply_idct_for_strategy(raw_strategy, error_coeffs, pixel_error_buf);
            let pixel_error = &*pixel_error_buf;

            // Compute 8th power norm with per-pixel masking via SIMD kernel.
            // mask1x1 is padded to block-aligned dimensions (xsize_blocks*8 × ysize_blocks*8),
            // and mask1x1_stride = padded_width, so all block pixel accesses are in-bounds.
            let mask_offset = MASK_CHANNEL_OFFSET[c];
            let block_width_px = cx * BLOCK_DIM;
            let block_height_px = cy * BLOCK_DIM;

            let mut channel_loss = jxl_simd::pixel_domain_loss(
                pixel_error,
                mask,
                mask_row_base,
                mask1x1_stride,
                mask_offset,
                block_width_px,
                block_height_px,
            );

            // Apply channel multiplier
            channel_loss *= CHANNEL_MUL[c];

            // W44-59: per-channel pixel loss
            if afv_coeff_target {
                dump_afv_coeff_block(
                    "pixel_error",
                    raw_strategy,
                    bx * BLOCK_DIM,
                    by * BLOCK_DIM,
                    c,
                    pixel_error,
                );
                dump_afv_aggregate(
                    raw_strategy,
                    bx * BLOCK_DIM,
                    by * BLOCK_DIM,
                    c,
                    "channel_loss",
                    channel_loss as f32,
                );
            }

            total_pixel_loss += channel_loss;

            // X channel penalty for large transforms - applied to TOTAL loss accumulator
            // (not per-channel). This matches libjxl enc_ac_strategy.cc:500-501
            // IMPORTANT: Apply AFTER adding channel_loss, so it multiplies entire accumulator
            if c == 0 && num_blocks >= 2 {
                let w = 1.0 + (num_blocks as f64 / 8.0).min(3.0);
                total_pixel_loss *= w;
            }
        }
    }

    // Compute final cost: entropy * entropy_mul + loss
    // CRITICAL: entropy_mul applies ONLY to entropy, not to loss!
    // This matches libjxl enc_ac_strategy.cc:508-509
    let (entropy_pre_loss, loss_scalar_for_dump) = if use_pixel_domain {
        // Pixel-domain loss: (sum/n)^(1/8) * n / quant_norm16
        let n = (num_blocks * DCT_BLOCK_SIZE) as f64;
        // x^(1/8) = sqrt(sqrt(sqrt(x)))
        let loss_scalar = (total_pixel_loss / n).sqrt().sqrt().sqrt() * n / quant_norm16 as f64;
        // Apply entropy_mul to entropy, then add loss
        let entropy_pre = entropy * entropy_mul;
        entropy *= entropy_mul;
        entropy += k_info_loss_mul * loss_scalar as f32;
        (entropy_pre, loss_scalar as f32)
    } else {
        // Coefficient-domain loss (libjxl-tiny style)
        // In this mode, entropy_mul is 1.0 and caller applies multipliers externally
        let infoloss2 = (num_blocks as f32 * info_loss2_sum).sqrt();
        let info_loss_score = k_info_loss_mul * info_loss_sum + K_INFO_LOSS_MULTIPLIER2 * infoloss2;
        let entropy_pre = entropy;
        entropy += mask_val * info_loss_score;
        (entropy_pre, info_loss_score)
    };

    // W44-58 / W44-59 fix: dump using libjxl wire strategy code so AFV codes
    // align with libjxl's `acs.Strategy()` enum (AFV0=14, AFV1=15, AFV2=16,
    // AFV3=17). The original W44-58 dump passed `raw_strategy` (AFV0=12,
    // AFV1=13, AFV2=14, AFV3=15), causing AFV0/1 vs AFV2/3 to be joined at
    // the same coordinates — explaining the entire "12.4% AFV divergence".
    dump_cost_inputs(
        STRATEGY_CODE_LUT[raw_strategy as usize], bx, by, entropy_mul, cx, cy, quant_norm16,
        ytox, ytob, mask1x1, mask1x1_stride,
        entropy_pre_loss, loss_scalar_for_dump,
        k_info_loss_mul, k_cost_delta, k_zeros_mul, entropy,
    );

    // W44-59: per-block aggregates for the targeted AFV block (after all
    // channels have been processed).
    if afv_coeff_target {
        dump_afv_aggregate(
            raw_strategy,
            bx * BLOCK_DIM,
            by * BLOCK_DIM,
            255, // sentinel c for "all channels"
            "total_pixel_loss",
            total_pixel_loss as f32,
        );
        dump_afv_aggregate(
            raw_strategy,
            bx * BLOCK_DIM,
            by * BLOCK_DIM,
            255,
            "quant_norm16",
            quant_norm16,
        );
        dump_afv_aggregate(
            raw_strategy,
            bx * BLOCK_DIM,
            by * BLOCK_DIM,
            255,
            "entropy_pre_loss",
            entropy_pre_loss,
        );
        dump_afv_aggregate(
            raw_strategy,
            bx * BLOCK_DIM,
            by * BLOCK_DIM,
            255,
            "loss_scalar",
            loss_scalar_for_dump,
        );
        dump_afv_aggregate(
            raw_strategy,
            bx * BLOCK_DIM,
            by * BLOCK_DIM,
            255,
            "final_entropy",
            entropy,
        );
    }

    entropy
}

/// W44-58: env-var-driven per-call cost-input dump (parity diff vs libjxl).
///
/// Set `JXL_DUMP_COST_INPUTS_PATH=/path/to/dump.tsv` to enable. Format:
///   strategy x y entropy_mul cbx cby quant_norm16 cmap_x cmap_b
///   mask1x1_mean_y entropy_pre_loss loss_scalar info_loss_mul cost_delta
///   zeros_mul final_entropy
/// Coords reported in pixel units (x=bx*8, y=by*8). Zero overhead when env
/// var unset (one getenv at first call per thread).
#[allow(clippy::too_many_arguments)]
#[inline]
fn dump_cost_inputs(
    raw_strategy: u8,
    bx: usize,
    by: usize,
    entropy_mul: f32,
    cx: usize,
    cy: usize,
    quant_norm16: f32,
    ytox: i8,
    ytob: i8,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    entropy_pre_loss: f32,
    loss_scalar_for_dump: f32,
    k_info_loss_mul: f32,
    k_cost_delta: f32,
    k_zeros_mul: f32,
    entropy: f32,
) {
    #[cfg(feature = "std")]
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        use std::sync::{Mutex, OnceLock};
        static DUMP_FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
        let slot = DUMP_FILE.get_or_init(|| {
            let path = std::env::var("JXL_DUMP_COST_INPUTS_PATH").ok()?;
            if path.is_empty() { return None; }
            let mut f = OpenOptions::new().create(true).append(true).open(&path).ok()?;
            if f.metadata().map(|m| m.len() == 0).unwrap_or(false) {
                let _ = writeln!(
                    f,
                    "strategy\tx\ty\tentropy_mul\tcovered_blocks_x\tcovered_blocks_y\tquant_norm16\tcmap_x\tcmap_b\tmask1x1_mean_y\tentropy_pre_loss\tloss_scalar\tinfo_loss_mul\tcost_delta\tzeros_mul\tfinal_entropy"
                );
            }
            Some(Mutex::new(f))
        });
        if let Some(file_mutex) = slot.as_ref() {
            if let Ok(mut f) = file_mutex.lock() {
                let (mask_mean, _count) = if let Some(m) = mask1x1 {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for iy in 0..(cy * BLOCK_DIM) {
                        for ix in 0..(cx * BLOCK_DIM) {
                            let py = by * BLOCK_DIM + iy;
                            let px = bx * BLOCK_DIM + ix;
                            let idx = py * mask1x1_stride + px;
                            if idx < m.len() {
                                sum += m[idx];
                                count += 1;
                            }
                        }
                    }
                    if count > 0 { (sum / count as f32, count) } else { (0.0, 0) }
                } else { (0.0, 0) };
                let cmap_x = ytox_ratio(ytox);
                let cmap_b = ytob_ratio(ytob);
                let _ = writeln!(
                    f,
                    "{}\t{}\t{}\t{:.6}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                    raw_strategy,
                    bx * BLOCK_DIM,
                    by * BLOCK_DIM,
                    entropy_mul,
                    cx,
                    cy,
                    quant_norm16,
                    cmap_x,
                    cmap_b,
                    mask_mean,
                    entropy_pre_loss,
                    loss_scalar_for_dump,
                    k_info_loss_mul,
                    k_cost_delta,
                    k_zeros_mul,
                    entropy,
                );
            }
        }
    }
    // Suppress unused-variable warnings when std is disabled.
    #[cfg(not(feature = "std"))]
    {
        let _ = (raw_strategy, bx, by, entropy_mul, cx, cy, quant_norm16,
                 ytox, ytob, mask1x1, mask1x1_stride,
                 entropy_pre_loss, loss_scalar_for_dump,
                 k_info_loss_mul, k_cost_delta, k_zeros_mul, entropy);
    }
}

/// W44-59: per-coefficient AFV dump infrastructure.
///
/// Activates when ALL of the following env vars are set:
///   JXL_DUMP_AFV_PATH=/path/to/dump.tsv
///   JXL_DUMP_AFV_KIND=<0..3>     // 0=AFV0, 1=AFV1, 2=AFV2, 3=AFV3
///   JXL_DUMP_AFV_X=<pixel_x>     // block top-left x in pixels
///   JXL_DUMP_AFV_Y=<pixel_y>     // block top-left y in pixels
///
/// Dump format (TSV, one row per cell):
///   stage  raw_strategy  x  y  c  i  value
/// Stages: "pixels" (8x8 input pixels per channel, i=0..63)
///         "coeffs" (AFV transform output, i=0..63)
///         "inv_weights" (dequant matrix, i=0..63)
///         "weights" (forward quant matrix, i=0..63)
///         "per_coeff_val/rval/diff/weighted_diff" (entropy estimator stage)
///         "pixel_error" (post-IDCT pixel error, i=0..63)
///         "aggregate.entropy_sum/.nzeros_sum/.channel_loss" (per-channel scalars,
///            i=0)
///         "aggregate.total_pixel_loss/.quant_norm16/.entropy_pre_loss/
///         .loss_scalar/.final_entropy" (per-block scalars, c=255 sentinel, i=0)
#[inline(always)]
#[allow(dead_code)]
fn afv_coeff_dump_target(raw_strategy: u8, x: usize, y: usize) -> bool {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static TARGET: OnceLock<Option<(u8, usize, usize)>> = OnceLock::new();
        let cached = TARGET.get_or_init(|| {
            let path = std::env::var("JXL_DUMP_AFV_PATH").ok()?;
            if path.is_empty() { return None; }
            let kind: u8 = std::env::var("JXL_DUMP_AFV_KIND").ok()?.parse().ok()?;
            if kind > 3 { return None; }
            let tx: usize = std::env::var("JXL_DUMP_AFV_X").ok()?.parse().ok()?;
            let ty: usize = std::env::var("JXL_DUMP_AFV_Y").ok()?.parse().ok()?;
            // Map AFV kind 0..3 → raw_strategy RAW_STRATEGY_AFV0..AFV3.
            Some((RAW_STRATEGY_AFV0 + kind, tx, ty))
        });
        if let Some((target_strat, tx, ty)) = *cached {
            return raw_strategy == target_strat && x == tx && y == ty;
        }
    }
    let _ = (raw_strategy, x, y);
    false
}

#[allow(dead_code)]
#[cfg(feature = "std")]
fn afv_dump_file() -> Option<&'static std::sync::Mutex<std::fs::File>> {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};
    static DUMP_FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let slot = DUMP_FILE.get_or_init(|| {
        let path = std::env::var("JXL_DUMP_AFV_PATH").ok()?;
        if path.is_empty() { return None; }
        let mut f = OpenOptions::new().create(true).append(true).open(&path).ok()?;
        if f.metadata().map(|m| m.len() == 0).unwrap_or(false) {
            let _ = writeln!(f, "stage\traw_strategy\tx\ty\tc\ti\tvalue");
        }
        Some(Mutex::new(f))
    });
    slot.as_ref()
}

#[allow(dead_code)]
fn dump_afv_coeff_block(
    stage: &str,
    raw_strategy: u8,
    x: usize,
    y: usize,
    c: usize,
    data: &[f32],
) {
    #[cfg(feature = "std")]
    {
        use std::io::Write;
        if let Some(mu) = afv_dump_file() {
            if let Ok(mut f) = mu.lock() {
                for (i, &v) in data.iter().enumerate() {
                    let _ = writeln!(
                        f,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{:.8e}",
                        stage, raw_strategy, x, y, c, i, v
                    );
                }
            }
        }
    }
    let _ = (stage, raw_strategy, x, y, c, data);
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn dump_afv_per_coeff(
    raw_strategy: u8,
    x: usize,
    y: usize,
    c: usize,
    i: usize,
    val: f32,
    rval: f32,
    diff: f32,
    weighted_diff: f32,
) {
    #[cfg(feature = "std")]
    {
        use std::io::Write;
        if let Some(mu) = afv_dump_file() {
            if let Ok(mut f) = mu.lock() {
                let _ = writeln!(
                    f,
                    "per_coeff_val\t{}\t{}\t{}\t{}\t{}\t{:.8e}",
                    raw_strategy, x, y, c, i, val
                );
                let _ = writeln!(
                    f,
                    "per_coeff_rval\t{}\t{}\t{}\t{}\t{}\t{:.8e}",
                    raw_strategy, x, y, c, i, rval
                );
                let _ = writeln!(
                    f,
                    "per_coeff_diff\t{}\t{}\t{}\t{}\t{}\t{:.8e}",
                    raw_strategy, x, y, c, i, diff
                );
                let _ = writeln!(
                    f,
                    "per_coeff_weighted_diff\t{}\t{}\t{}\t{}\t{}\t{:.8e}",
                    raw_strategy, x, y, c, i, weighted_diff
                );
            }
        }
    }
    let _ = (raw_strategy, x, y, c, i, val, rval, diff, weighted_diff);
}

#[allow(dead_code)]
fn dump_afv_aggregate(
    raw_strategy: u8,
    x: usize,
    y: usize,
    c: usize,
    name: &str,
    value: f32,
) {
    #[cfg(feature = "std")]
    {
        use std::io::Write;
        if let Some(mu) = afv_dump_file() {
            if let Ok(mut f) = mu.lock() {
                let _ = writeln!(
                    f,
                    "aggregate.{}\t{}\t{}\t{}\t{}\t0\t{:.8e}",
                    name, raw_strategy, x, y, c, value
                );
            }
        }
    }
    let _ = (raw_strategy, x, y, c, name, value);
}

/// Apply inverse DCT to error coefficients based on strategy.
/// Writes pixel-domain error in row-major layout into `output`.
#[inline(always)]
pub(super) fn apply_idct_for_strategy(raw_strategy: u8, error_coeffs: &[f32], output: &mut [f32]) {
    match raw_strategy {
        RAW_STRATEGY_DCT8 => {
            idct_8x8(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT4X8 => {
            idct_4x8_full(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT8X4 => {
            idct_8x4_full(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT4X4 => {
            idct_4x4_full(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
            let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
            super::afv::inverse_afv_transform(
                as_array_ref(error_coeffs, 0),
                afv_kind,
                as_array_mut(output, 0),
            );
        }
        RAW_STRATEGY_DCT16X8 => {
            // Coefficients are in post-swap 8×16 layout (stride 16), but idct_16x8
            // expects natural 16×8 layout (stride 8). Transpose first.
            // (idct_32x16/idct_64x32 handle this internally; idct_16x8 does not.)
            let mut transposed = uninit_buf::<128>();
            for y in 0..8 {
                for x in 0..16 {
                    transposed[x * 8 + y] = error_coeffs[y * 16 + x];
                }
            }
            idct_16x8(&transposed, as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT8X16 => {
            idct_8x16(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT16X16 => {
            idct_16x16(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT32X32 => {
            idct_32x32(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT32X16 => {
            idct_32x16(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT16X32 => {
            idct_16x32(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT64X64 => {
            idct_64x64(&error_coeffs[..4096], &mut output[..4096]);
        }
        RAW_STRATEGY_DCT64X32 => {
            idct_64x32(&error_coeffs[..2048], &mut output[..2048]);
        }
        RAW_STRATEGY_DCT32X64 => {
            idct_32x64(&error_coeffs[..2048], &mut output[..2048]);
        }
        RAW_STRATEGY_IDENTITY => {
            inverse_identity_transform(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        RAW_STRATEGY_DCT2X2 => {
            inverse_dct2x2_transform(as_array_ref(error_coeffs, 0), as_array_mut(output, 0));
        }
        _ => unreachable!(
            "unknown strategy {} in apply_idct_for_strategy",
            raw_strategy
        ),
    }
}

/// Adjust the float quant field for multi-block transforms.
///
/// Same algorithm as `adjust_quant_field_with_distance` but operates on the
/// float quant field (values ~0.3-1.5) instead of u8 (1-255). This matches
/// libjxl's `AdjustQuantField` which works on `ImageF` before `SetQuantField`.
pub fn adjust_quant_field_float_with_distance(
    ac_strategy: &AcStrategyMap,
    quant_field: &mut [f32],
    butteraugli_target: f32,
) {
    let xsize_blocks = ac_strategy.xsize_blocks;

    const K_LIMIT: f32 = 1.54138;
    const K_MUL: f32 = 0.56391;
    const K_MIN: f32 = 0.0;

    let mut mean_max_mixer = 1.0_f32;
    if butteraugli_target > K_LIMIT {
        mean_max_mixer -= (butteraugli_target - K_LIMIT) * K_MUL;
        if mean_max_mixer < K_MIN {
            mean_max_mixer = K_MIN;
        }
    }

    for by in 0..ac_strategy.ysize_blocks {
        for bx in 0..ac_strategy.xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }
            let cx = ac_strategy.covered_blocks_x(bx, by);
            let cy = ac_strategy.covered_blocks_y(bx, by);
            if cx == 1 && cy == 1 {
                continue;
            }

            // Compute max and mean of covered region
            let mut max_q = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            for iy in 0..cy {
                for ix in 0..cx {
                    let q = quant_field[(by + iy) * xsize_blocks + bx + ix];
                    max_q = max_q.max(q);
                    sum += q as f64;
                }
            }
            let mean = (sum / (cx * cy) as f64) as f32;

            // Blend max and mean (for 4+ block transforms)
            let blended = if cx * cy >= 4 {
                max_q * mean_max_mixer + mean * (1.0 - mean_max_mixer)
            } else {
                max_q
            };

            // Set all covered blocks to blended value (no integer clamping)
            for iy in 0..cy {
                for ix in 0..cx {
                    quant_field[(by + iy) * xsize_blocks + bx + ix] = blended;
                }
            }
        }
    }
}

pub fn adjust_quant_field_with_distance(
    ac_strategy: &AcStrategyMap,
    quant_field: &mut [u8],
    butteraugli_target: f32,
) {
    let xsize_blocks = ac_strategy.xsize_blocks;

    // At low distances use max, at high distances blend toward mean.
    // libjxl constants from enc_adaptive_quantization.cc:1207-1215
    const K_LIMIT: f32 = 1.54138;
    const K_MUL: f32 = 0.56391;
    const K_MIN: f32 = 0.0;

    let mut mean_max_mixer = 1.0_f32;
    if butteraugli_target > K_LIMIT {
        mean_max_mixer -= (butteraugli_target - K_LIMIT) * K_MUL;
        if mean_max_mixer < K_MIN {
            mean_max_mixer = K_MIN;
        }
    }

    for by in 0..ac_strategy.ysize_blocks {
        for bx in 0..ac_strategy.xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }
            let cx = ac_strategy.covered_blocks_x(bx, by);
            let cy = ac_strategy.covered_blocks_y(bx, by);
            if cx == 1 && cy == 1 {
                continue;
            }

            // Compute max and mean of covered region
            let mut max_q = 0u8;
            let mut sum = 0u32;
            for iy in 0..cy {
                for ix in 0..cx {
                    let q = quant_field[(by + iy) * xsize_blocks + bx + ix];
                    max_q = max_q.max(q);
                    sum += q as u32;
                }
            }
            let mean = sum as f32 / (cx * cy) as f32;

            // Blend max and mean (for 4+ block transforms)
            let blended = if cx * cy >= 4 {
                let max_f = max_q as f32;
                max_f * mean_max_mixer + mean * (1.0 - mean_max_mixer)
            } else {
                max_q as f32
            };
            let blended_q = blended.round().clamp(1.0, 255.0) as u8;

            // Set all covered blocks to blended value
            for iy in 0..cy {
                for ix in 0..cx {
                    quant_field[(by + iy) * xsize_blocks + bx + ix] = blended_q;
                }
            }
        }
    }
}

/// Adjust the quant field for non-8×8 transforms (legacy, max-only version).
/// Use `adjust_quant_field_with_distance` for better quality at high distances.
#[allow(dead_code)]
pub fn adjust_quant_field(ac_strategy: &AcStrategyMap, quant_field: &mut [u8]) {
    // Use max-only behavior (mean_max_mixer = 1.0, equivalent to d < 1.54)
    adjust_quant_field_with_distance(ac_strategy, quant_field, 0.0);
}

// ─── Top-level API ──────────────────────────────────────────────────────────

/// Process a single tile's AC strategy selection.
///
/// This is the per-tile body of `compute_ac_strategy`, extracted to enable parallel
/// execution. Each tile writes to its own `AcStrategyMap`; the caller merges results.
#[allow(clippy::too_many_arguments)]
fn process_tile(
    xyb: &[&[f32]; 3],
    stride: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    tile_bx: usize,
    tile_by: usize,
    tile_w: usize,
    tile_h: usize,
    distance: f32,
    quant_field_float: &[f32],
    masking: &[f32],
    cfl_map: &CflMap,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    profile: &EffortProfile,
    ac_strategy: &mut AcStrategyMap,
    scratch: &mut EntropyEstScratch,
) {
    let _ = (xsize_blocks, ysize_blocks); // available for bounds checks

    // Get CfL params for this tile
    let tx = tile_bx / TILE_DIM_IN_BLOCKS;
    let ty = tile_by / TILE_DIM_IN_BLOCKS;
    let ytox = cfl_map.ytox_at(tx, ty);
    let ytob = cfl_map.ytob_at(tx, ty);

    let try_64 = profile.try_dct64;
    let try_32 = profile.try_dct32;

    let mut cy = 0;
    // Process 8-row bands: try DCT64x64/DCT64x32/DCT32x64 at effort 7+
    while try_64 && cy + 7 < tile_h {
        let mut cx = 0;
        while cx + 7 < tile_w {
            find_best_64x64_transform(
                *xyb,
                stride,
                tile_bx,
                tile_by,
                cx,
                cy,
                distance,
                quant_field_float,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
                scratch,
                profile,
            );
            cx += 8;
        }
        // Remaining cols in this 8-row band: 4-block groups, then 2-block groups
        while try_32 && cx + 3 < tile_w {
            find_best_32x32_transform(
                *xyb,
                stride,
                tile_bx,
                tile_by,
                cx,
                cy,
                distance,
                quant_field_float,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
                scratch,
                None,
                profile,
            );
            cx += 4;
        }
        while cx + 1 < tile_w {
            find_best_16x16_transform(
                *xyb,
                stride,
                tile_bx,
                tile_by,
                cx,
                cy,
                distance,
                quant_field_float,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
                scratch,
                1.0,
                None,
                profile,
            );
            cx += 2;
        }
        cy += 8;
    }
    // Remaining rows: 4-row bands for 32×32 at effort 5+, then 2-row bands for 16×16
    while try_32 && cy + 3 < tile_h {
        let mut cx = 0;
        while cx + 3 < tile_w {
            find_best_32x32_transform(
                *xyb,
                stride,
                tile_bx,
                tile_by,
                cx,
                cy,
                distance,
                quant_field_float,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
                scratch,
                None,
                profile,
            );
            cx += 4;
        }
        while cx + 1 < tile_w {
            find_best_16x16_transform(
                *xyb,
                stride,
                tile_bx,
                tile_by,
                cx,
                cy,
                distance,
                quant_field_float,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
                scratch,
                1.0,
                None,
                profile,
            );
            cx += 2;
        }
        cy += 4;
    }
    // Handle remaining rows that don't fit a 32×32 block
    while cy + 1 < tile_h {
        let mut cx = 0;
        while cx + 1 < tile_w {
            find_best_16x16_transform(
                *xyb,
                stride,
                tile_bx,
                tile_by,
                cx,
                cy,
                distance,
                quant_field_float,
                xsize_blocks,
                masking,
                ytox,
                ytob,
                mask1x1,
                mask1x1_stride,
                ac_strategy,
                scratch,
                1.0,
                None,
                profile,
            );
            cx += 2;
        }
        cy += 2;
    }

    // Non-aligned matching: try 16×16/16×8/8×16 at non-2-aligned positions.
    let is_full_tile = tile_w == TILE_DIM_IN_BLOCKS && tile_h == TILE_DIM_IN_BLOCKS;
    for cy in if profile.non_aligned_eval { 0 } else { tile_h }..tile_h.saturating_sub(1) {
        for cx in 0..tile_w.saturating_sub(1) {
            if (cy | cx) % 2 == 0 {
                continue;
            }
            let abs_bx = tile_bx + cx;
            let abs_by = tile_by + cy;
            // W44-23 (Sub-G Approach B, 2026-05-18): loosen the gate from
            // all-single-block (W44-22) to outer-crossing-only for the
            // is_full_tile path, matching libjxl
            // `enc_ac_strategy.cc:725-741`. Partial paints stay safe
            // because `try_merge_16x16_impl` now computes per-half
            // `allow_jxk` / `allow_kxj` flags from the inner-crossing
            // helpers. The non-`is_full_tile` `find_best_16x16_transform`
            // path keeps using the stricter check because its restore /
            // reset logic relies on the all-single invariant (it resets
            // the region to DCT8 unconditionally before searching).
            if is_full_tile {
                if !ac_strategy.can_evaluate_region_outer_only(abs_bx, abs_by, 2) {
                    continue;
                }
                try_merge_16x16(
                    *xyb,
                    stride,
                    tile_bx,
                    tile_by,
                    cx,
                    cy,
                    distance,
                    quant_field_float,
                    xsize_blocks,
                    masking,
                    ytox,
                    ytob,
                    mask1x1,
                    mask1x1_stride,
                    ac_strategy,
                    scratch,
                    profile,
                );
            } else {
                if !ac_strategy.can_evaluate_region(abs_bx, abs_by, 2) {
                    continue;
                }
                let mut saved = [0u8; 4];
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        saved[dy * 2 + dx] = ac_strategy.raw_byte(abs_bx + dx, abs_by + dy);
                    }
                }
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        ac_strategy.set(abs_bx + dx, abs_by + dy, RAW_STRATEGY_DCT8);
                    }
                }
                find_best_16x16_transform(
                    *xyb,
                    stride,
                    tile_bx,
                    tile_by,
                    cx,
                    cy,
                    distance,
                    quant_field_float,
                    xsize_blocks,
                    masking,
                    ytox,
                    ytob,
                    mask1x1,
                    mask1x1_stride,
                    ac_strategy,
                    scratch,
                    1.0,
                    None,
                    profile,
                );
                let has_multi = (0..2usize).any(|dy| {
                    (0..2usize).any(|dx| {
                        let raw = ac_strategy.raw_strategy(abs_bx + dx, abs_by + dy);
                        COVERED_X[raw as usize] > 1 || COVERED_Y[raw as usize] > 1
                    })
                });
                if !has_multi {
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            ac_strategy.set_raw_byte(abs_bx + dx, abs_by + dy, saved[dy * 2 + dx]);
                        }
                    }
                }
            }
        }
    }

    // Non-aligned matching for 32×32/32×16/16×32 at non-4-aligned positions.
    if profile.non_aligned_eval {
        let step = profile.fine_grained_step as usize;
        for cy in (0..tile_h.saturating_sub(3)).step_by(step) {
            for cx in (0..tile_w.saturating_sub(3)).step_by(step) {
                if (cy | cx) % 4 == 0 {
                    continue;
                }
                let abs_bx = tile_bx + cx;
                let abs_by = tile_by + cy;
                // W44-23 (Sub-G Approach B, 2026-05-18): mirror the 16×16
                // pattern — outer-only gate for try_merge_32x32 (with per-half
                // inner-crossing guards inside the impl), stricter
                // all-single gate for the find_best_32x32_transform path.
                if is_full_tile {
                    if !ac_strategy.can_evaluate_region_outer_only(abs_bx, abs_by, 4) {
                        continue;
                    }
                    try_merge_32x32(
                        *xyb,
                        stride,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        quant_field_float,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        ac_strategy,
                        scratch,
                        profile,
                    );
                } else {
                    if !ac_strategy.can_evaluate_region(abs_bx, abs_by, 4) {
                        continue;
                    }
                    let mut saved = [0u8; 16];
                    for dy in 0..4usize {
                        for dx in 0..4usize {
                            saved[dy * 4 + dx] = ac_strategy.raw_byte(abs_bx + dx, abs_by + dy);
                        }
                    }
                    for dy in 0..4usize {
                        for dx in 0..4usize {
                            ac_strategy.set(abs_bx + dx, abs_by + dy, RAW_STRATEGY_DCT8);
                        }
                    }
                    find_best_32x32_transform(
                        *xyb,
                        stride,
                        tile_bx,
                        tile_by,
                        cx,
                        cy,
                        distance,
                        quant_field_float,
                        xsize_blocks,
                        masking,
                        ytox,
                        ytob,
                        mask1x1,
                        mask1x1_stride,
                        ac_strategy,
                        scratch,
                        None,
                        profile,
                    );
                    let has_multi = (0..4usize).any(|dy| {
                        (0..4usize).any(|dx| {
                            let raw = ac_strategy.raw_strategy(abs_bx + dx, abs_by + dy);
                            COVERED_X[raw as usize] > 1 || COVERED_Y[raw as usize] > 1
                        })
                    });
                    if !has_multi {
                        for dy in 0..4usize {
                            for dx in 0..4usize {
                                ac_strategy.set_raw_byte(
                                    abs_bx + dx,
                                    abs_by + dy,
                                    saved[dy * 4 + dx],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Compute the AC strategy map for the entire image.
///
/// Processes tiles in parallel, each with independent scratch buffers and a
/// thread-local `AcStrategyMap`. Tile results are merged into the final map.
#[allow(clippy::too_many_arguments)]
pub fn compute_ac_strategy(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    buf_height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    quant_field_float: &[f32],
    masking: &[f32],
    cfl_map: &CflMap,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    profile: &EffortProfile,
) -> AcStrategyMap {
    // Collect tile coordinates covering the whole image.
    let mut tiles = Vec::new();
    for tile_by in (0..ysize_blocks).step_by(TILE_DIM_IN_BLOCKS) {
        for tile_bx in (0..xsize_blocks).step_by(TILE_DIM_IN_BLOCKS) {
            tiles.push((tile_bx, tile_by));
        }
    }
    compute_ac_strategy_for_tiles(
        xyb_x,
        xyb_y,
        xyb_b,
        stride,
        buf_height,
        xsize_blocks,
        ysize_blocks,
        distance,
        quant_field_float,
        masking,
        cfl_map,
        mask1x1,
        mask1x1_stride,
        profile,
        &tiles,
    )
}

/// Per-tile-list variant of [`compute_ac_strategy`]. Runs the
/// AC strategy search for an arbitrary list of tile top-left
/// coordinates instead of every tile in the image.
///
/// Used by [`super::precomputed::compute_dc_group`] (#11 chunk 3) to
/// process just the 4×4 tiles inside a single DC group while leaving
/// the rest of the returned `AcStrategyMap` at the default DCT8
/// sentinel. The caller then `copy_region_from`s the DC group's tile
/// rectangle into the aggregated whole-image map.
///
/// Byte-identical to [`compute_ac_strategy`] when `tile_list` contains
/// every tile in row-major order. Per-DC-group calls are byte-identical
/// to the corresponding tile slice of the whole-image call because
/// each per-tile worker reads only from the passed-in whole-image
/// slices (XYB / quant_field / masking / mask1x1 / CfL) — there is no
/// cross-tile state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ac_strategy_for_tiles(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    buf_height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    quant_field_float: &[f32],
    masking: &[f32],
    cfl_map: &CflMap,
    mask1x1: Option<&[f32]>,
    mask1x1_stride: usize,
    profile: &EffortProfile,
    tile_list: &[(usize, usize)],
) -> AcStrategyMap {
    let _ = buf_height; // Used for documentation; buffer is padded to ysize_blocks * 8

    let xyb = [xyb_x, xyb_y, xyb_b];

    // Process tiles in parallel. Each tile gets its own AcStrategyMap and scratch
    // buffer. Tiles are spatially disjoint, so results can be merged by copying
    // each tile's region into the final map.
    let tile_results = crate::parallel::parallel_map(tile_list.len(), |tile_idx| {
        let (tile_bx, tile_by) = tile_list[tile_idx];
        let tile_w = TILE_DIM_IN_BLOCKS.min(xsize_blocks - tile_bx);
        let tile_h = TILE_DIM_IN_BLOCKS.min(ysize_blocks - tile_by);

        let mut local_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
        let mut scratch = EntropyEstScratch::new();

        process_tile(
            &xyb,
            stride,
            xsize_blocks,
            ysize_blocks,
            tile_bx,
            tile_by,
            tile_w,
            tile_h,
            distance,
            quant_field_float,
            masking,
            cfl_map,
            mask1x1,
            mask1x1_stride,
            profile,
            &mut local_strategy,
            &mut scratch,
        );

        local_strategy
    });

    // Merge tile results into a single map
    let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
    for (tile_idx, tile_map) in tile_results.into_iter().enumerate() {
        let (tile_bx, tile_by) = tile_list[tile_idx];
        let tile_w = TILE_DIM_IN_BLOCKS.min(xsize_blocks - tile_bx);
        let tile_h = TILE_DIM_IN_BLOCKS.min(ysize_blocks - tile_by);
        ac_strategy.copy_region_from(
            &tile_map,
            tile_bx,
            tile_by,
            tile_bx + tile_w,
            tile_by + tile_h,
        );
    }

    // Validate strategy map consistency in debug builds
    #[cfg(debug_assertions)]
    {
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let raw = ac_strategy.raw_strategy(bx, by);
                if ac_strategy.is_first(bx, by) {
                    let cx = COVERED_X[raw as usize];
                    let cy = COVERED_Y[raw as usize];
                    // Verify all covered blocks have matching raw_strategy and is_first=false
                    for iy in 0..cy {
                        for ix in 0..cx {
                            assert!(
                                bx + ix < xsize_blocks && by + iy < ysize_blocks,
                                "Transform at ({},{}) raw={} extends out of bounds: ({},{}) vs {}x{}",
                                bx,
                                by,
                                raw,
                                bx + ix,
                                by + iy,
                                xsize_blocks,
                                ysize_blocks
                            );
                            assert_eq!(
                                ac_strategy.raw_strategy(bx + ix, by + iy),
                                raw,
                                "Inconsistent raw_strategy at ({},{}) — expected {} (from first block ({},{})), got {}",
                                bx + ix,
                                by + iy,
                                raw,
                                bx,
                                by,
                                ac_strategy.raw_strategy(bx + ix, by + iy)
                            );
                            if (ix | iy) != 0 {
                                assert!(
                                    !ac_strategy.is_first(bx + ix, by + iy),
                                    "Block ({},{}) should not be first (owned by ({},{}) raw={})",
                                    bx + ix,
                                    by + iy,
                                    bx,
                                    by,
                                    raw
                                );
                            }
                        }
                    }
                    // Verify group boundary
                    let gx = bx % 32;
                    let gy = by % 32;
                    assert!(
                        gx + cx <= 32 && gy + cy <= 32,
                        "Transform at ({bx},{by}) raw={raw} cx={cx} cy={cy} crosses group border: gx={gx} gy={gy}"
                    );
                    // Verify fits in LfGroup
                    assert!(
                        bx + cx <= xsize_blocks && by + cy <= ysize_blocks,
                        "Transform at ({bx},{by}) raw={raw} cx={cx} cy={cy} exceeds image: {}x{} blocks",
                        xsize_blocks,
                        ysize_blocks
                    );
                }
            }
        }
    }

    ac_strategy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac_strategy_map_default() {
        let map = AcStrategyMap::new_dct8(4, 4);
        for by in 0..4 {
            for bx in 0..4 {
                assert_eq!(map.raw_strategy(bx, by), 0);
                assert!(map.is_first(bx, by));
                assert_eq!(map.strategy_code(bx, by), 0);
            }
        }
        assert_eq!(map.count_first_blocks(), 16);
    }

    #[test]
    fn test_ac_strategy_map_set_dct16x8() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        // DCT16X8 at (0,0): covers (0,0) and (0,1)
        map.set(0, 0, RAW_STRATEGY_DCT16X8);
        assert_eq!(map.raw_strategy(0, 0), RAW_STRATEGY_DCT16X8);
        assert!(map.is_first(0, 0));
        assert_eq!(map.raw_strategy(0, 1), RAW_STRATEGY_DCT16X8);
        assert!(!map.is_first(0, 1));
        // Strategy code for DCT16X8 is 6
        assert_eq!(map.strategy_code(0, 0), 6);
        // Rest should still be DCT8
        assert_eq!(map.raw_strategy(1, 0), 0);
        assert!(map.is_first(1, 0));
    }

    #[test]
    fn test_ac_strategy_map_set_dct8x16() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        // DCT8X16 at (2,0): covers (2,0) and (3,0)
        map.set(2, 0, RAW_STRATEGY_DCT8X16);
        assert_eq!(map.raw_strategy(2, 0), RAW_STRATEGY_DCT8X16);
        assert!(map.is_first(2, 0));
        assert_eq!(map.raw_strategy(3, 0), RAW_STRATEGY_DCT8X16);
        assert!(!map.is_first(3, 0));
        assert_eq!(map.strategy_code(2, 0), 7);
    }

    #[test]
    fn test_adjust_quant_field() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        // Set a DCT16X8 at (0,0)
        map.set(0, 0, RAW_STRATEGY_DCT16X8);
        let mut qf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        adjust_quant_field(&map, &mut qf);
        // DCT16X8 covers (0,0) and (0,1): max(1, 5) = 5
        assert_eq!(qf[0], 5);
        assert_eq!(qf[4], 5);
        // Other blocks unchanged
        assert_eq!(qf[1], 2);
    }

    #[test]
    fn test_estimate_entropy_finite() {
        // Test that estimate_entropy produces finite positive values
        let stride = 16;
        let buf_height = 16;
        let n = stride * buf_height;
        let xyb_x = vec![0.1f32; n];
        let xyb_y = vec![0.5f32; n];
        let xyb_b = vec![0.3f32; n];
        let xsize_blocks = 2;
        let quant_field = vec![4.0f32; 4];
        let masking = vec![1.0f32; 4];

        let ent = estimate_entropy(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
        );
        assert!(ent.is_finite(), "entropy should be finite: {}", ent);
        assert!(ent >= 0.0, "entropy should be non-negative: {}", ent);
    }

    #[test]
    fn test_count_first_blocks() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        assert_eq!(map.count_first_blocks(), 16);

        // Set one DCT16X8 (covers 2 blocks, 1 first)
        map.set(0, 0, RAW_STRATEGY_DCT16X8);
        assert_eq!(map.count_first_blocks(), 15); // 16 - 2 + 1

        // Set one DCT8X16 (covers 2 blocks, 1 first)
        map.set(2, 0, RAW_STRATEGY_DCT8X16);
        assert_eq!(map.count_first_blocks(), 14);
    }

    #[test]
    fn test_estimate_entropy_pixel_domain() {
        // Test that pixel-domain loss calculation produces finite positive values
        // and differs from coefficient-domain loss
        let stride = 16;
        let buf_height = 16;
        let n = stride * buf_height;
        let xyb_x = vec![0.1f32; n];
        let xyb_y = vec![0.5f32; n];
        let xyb_b = vec![0.3f32; n];
        let xsize_blocks = 2;
        let quant_field = vec![4.0f32; 4];
        let masking = vec![1.0f32; 4];

        // Create a simple mask1x1 field
        let mask1x1_stride = stride;
        let mask1x1 = vec![0.5f32; n];

        let mut scratch = EntropyEstScratch::new();
        let cost_bases = (1.2_f32, 9.308_906_f32, 10.833_273_f32);
        let pixel_constants = compute_scaled_constants(1.0, cost_bases);

        // Calculate coefficient-domain loss (without mask1x1)
        let ent_coeff = estimate_entropy_full(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            None,
            0,
            1.0, // entropy_mul = 1.0 for coefficient-domain (caller applies mul8x8)
            COEFF_DOMAIN_CONSTANTS,
            &mut scratch,
        );

        // Calculate pixel-domain loss (with mask1x1)
        let ent_pixel = estimate_entropy_full(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT8, &EntropyMulTable::reference()), // Normalized entropy_mul for DCT8 = 1.0
            pixel_constants,
            &mut scratch,
        );

        eprintln!("Coefficient-domain entropy: {}", ent_coeff);
        eprintln!("Pixel-domain entropy: {}", ent_pixel);

        // Both should be finite and non-negative
        assert!(
            ent_coeff.is_finite(),
            "coeff entropy should be finite: {}",
            ent_coeff
        );
        assert!(
            ent_coeff >= 0.0,
            "coeff entropy should be non-negative: {}",
            ent_coeff
        );
        assert!(
            ent_pixel.is_finite(),
            "pixel entropy should be finite: {}",
            ent_pixel
        );
        assert!(
            ent_pixel >= 0.0,
            "pixel entropy should be non-negative: {}",
            ent_pixel
        );

        // They should be different (pixel-domain uses different constants and loss calculation)
        // The difference magnitude depends on the specific test data
        // For uniform inputs, both may be similar, but for real images they differ more
    }

    #[test]
    fn test_estimate_entropy_pixel_domain_strategies() {
        // Test pixel-domain loss for different strategies
        let stride = 32;
        let buf_height = 32;
        let n = stride * buf_height;

        // Non-uniform input to exercise the loss calculation
        let xyb_x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.01).collect();
        let xyb_y: Vec<f32> = (0..n).map(|i| 0.3 + (i % 13) as f32 * 0.02).collect();
        let xyb_b: Vec<f32> = (0..n).map(|i| 0.2 + (i % 11) as f32 * 0.015).collect();
        let xsize_blocks = 4;
        let quant_field = vec![4.0f32; 16];
        let masking = vec![1.0f32; 16];

        let mask1x1_stride = stride;
        let mask1x1: Vec<f32> = (0..n).map(|i| 0.3 + (i % 7) as f32 * 0.1).collect();

        let mut scratch = EntropyEstScratch::new();
        let cost_bases = (1.2_f32, 9.308_906_f32, 10.833_273_f32);
        let pixel_constants = compute_scaled_constants(1.0, cost_bases);

        // Test DCT8
        let ent_dct8 = estimate_entropy_full(
            RAW_STRATEGY_DCT8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT8, &EntropyMulTable::reference()),
            pixel_constants,
            &mut scratch,
        );
        eprintln!("DCT8 pixel-domain entropy: {}", ent_dct8);
        assert!(ent_dct8.is_finite() && ent_dct8 >= 0.0);

        // Test DCT16x8 (requires 2-block tall region)
        let ent_dct16x8 = estimate_entropy_full(
            RAW_STRATEGY_DCT16X8,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT16X8, &EntropyMulTable::reference()),
            pixel_constants,
            &mut scratch,
        );
        eprintln!("DCT16x8 pixel-domain entropy: {}", ent_dct16x8);
        assert!(ent_dct16x8.is_finite() && ent_dct16x8 >= 0.0);

        // Test DCT8x16 (requires 2-block wide region)
        let ent_dct8x16 = estimate_entropy_full(
            RAW_STRATEGY_DCT8X16,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT16X8, &EntropyMulTable::reference()),
            pixel_constants,
            &mut scratch,
        );
        eprintln!("DCT8x16 pixel-domain entropy: {}", ent_dct8x16);
        assert!(ent_dct8x16.is_finite() && ent_dct8x16 >= 0.0);

        // Test DCT16x16 (requires 2x2 block region)
        let ent_dct16x16 = estimate_entropy_full(
            RAW_STRATEGY_DCT16X16,
            [&xyb_x, &xyb_y, &xyb_b],
            stride,
            0,
            0,
            1.0,
            &quant_field,
            xsize_blocks,
            &masking,
            0,
            0,
            Some(&mask1x1),
            mask1x1_stride,
            entropy_mul_for_strategy(RAW_STRATEGY_DCT16X16, &EntropyMulTable::reference()),
            pixel_constants,
            &mut scratch,
        );
        eprintln!("DCT16x16 pixel-domain entropy: {}", ent_dct16x16);
        assert!(ent_dct16x16.is_finite() && ent_dct16x16 >= 0.0);
    }
}
