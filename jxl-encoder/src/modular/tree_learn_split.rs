// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithm derived from libjxl's `SplitTreeSamples` in `enc_ma.cc:119-138`
// (BSD-3-Clause). Licensed under AGPL-3.0-or-later. Commercial licenses at
// https://www.imazen.io/pricing
//
// NOTE: This is the chunk-1 standalone primitive for issue #40
// (https://github.com/imazen/jxl-encoder/issues/40). It is NOT yet wired into
// `find_best_split` — that's chunk 2. This file ships the partition primitive,
// a unit-test layer that proves correct partitioning, and a microbench example
// that quantifies the read-pattern speedup vs the current indexed-access path.

//! In-place SoA permutation primitive for tree-learning split partitioning.
//!
//! libjxl's `SplitTreeSamples` (`enc_ma.cc:119-138`) physically swaps entire
//! sample rows across the partition boundary in every parallel array that
//! makes up `TreeSamples` (residuals, props, sample_counts, static_props).
//! After the call, the `[begin..pos)` range satisfies the partition predicate
//! and `[pos..end)` does not, AND each underlying SoA array is contiguous in
//! that range — so the per-(predictor × property) `collect` inner loop in
//! `find_best_split` reads sequential memory instead of chasing an index
//! permutation through random locations.
//!
//! Our current code partitions a separate `indices: &mut [usize]` and reads
//! `tokens[idx]`, `ebits[idx]`, `sample_counts[idx]` with the resulting
//! non-sequential `idx` values. Profile (issue #40 thread, 2026-05-15) showed
//! this `collect` step is 51.9% of wall-clock at 1024² e7.
//!
//! This primitive is the "SoA permutation" half of that fix. It operates on
//! a `SplittableSamples` view — a lightweight bundle of mutable references to
//! the underlying vectors — so it is decoupled from the production
//! `TreeSamples` struct and can be tested + microbenched in isolation.
//!
//! # Algorithm (mirrors libjxl)
//!
//! Hoare-style two-pointer scan:
//! 1. Walk `begin_pos` forward from `begin` looking for rows where the
//!    partition column's value at the row's CURRENT position is `> val`
//!    (i.e., misplaced on the left).
//! 2. Walk `end_pos` forward from `pos` looking for rows where the column's
//!    value is `<= val` (i.e., misplaced on the right).
//! 3. Swap them, advance both, repeat until either pointer reaches its limit.
//!
//! The partition column lives INSIDE the SoA bundle (see [`PartitionKey`])
//! and is swapped along with everything else, so the predicate evaluated at
//! row `i` after a swap correctly reflects the new occupant.
//!
//! Each swap moves an entire sample row in EVERY parallel array, preserving
//! the SoA invariant. Per-sample cost is `O(num_predictors * 2 + num_props *
//! 2 + 1)` per swap.

// Chunk 2 (commit pending): primitive is wired into find_best_split via
// `compute_best_tree` and `compute_best_tree_with_multipliers`. The `Property`
// variant of `PartitionKey` and the standalone view-style API remain available
// for direct callers and tests; nothing else in production reaches for them
// today, so quiet the dead-code warning on the unused arms.

use alloc::vec::Vec;

/// A bundle of mutable references to the parallel SoA arrays that make up a
/// tree-learning sample set.
///
/// This mirrors the shape of [`super::tree_learn::TreeSamples`] plus the
/// per-property `bucket_indices` from `PreQuantizedProps`, all of which need
/// to stay row-aligned across in-place permutation.
///
/// Field invariants:
/// - Every inner `Vec` has length `>= len`.
/// - `len` is the number of distinct samples (== `TreeSamples::num_samples`
///   after dedup).
/// - All arrays use the same row indexing: row `i` is sample `i`.
///
/// The struct does NOT own the arrays — callers pass references to existing
/// production vectors, so a swap in `SplittableSamples` is observable in the
/// caller's data.
pub struct SplittableSamples<'a> {
    /// Per-predictor residual tokens: `residual_tokens[pred][sample]`.
    pub residual_tokens: &'a mut [Vec<u8>],
    /// Per-predictor extra bits: `extra_bits[pred][sample]`.
    pub extra_bits: &'a mut [Vec<u8>],
    /// Per-property quantized values: `props[prop][sample]`.
    pub props: &'a mut [Vec<i32>],
    /// Per-property bucket indices: `bucket_indices[prop][sample]`.
    /// Mirrors `PreQuantizedProps::bucket_indices`.
    pub bucket_indices: &'a mut [Vec<u8>],
    /// Dedup weights: `sample_counts[sample]`.
    pub sample_counts: &'a mut Vec<u32>,
    /// Logical sample count (rows 0..len are live).
    pub len: usize,
    /// Issue #40 chunk-3c resurrection: when `true`, [`swap_rows`] skips
    /// the per-property `Vec<i32>` swaps in `props`. Safe ONLY on call paths
    /// that:
    /// 1. Use [`PartitionKey::Bucket`] exclusively (so the partition predicate
    ///    reads `bucket_indices`, not `props`), AND
    /// 2. Never read `samples.props` again after entering the tree-build loop
    ///    (since props will fall out of alignment with the other SoA arrays).
    ///
    /// The lossless main path (`compute_best_tree_with_budget`) satisfies
    /// both conditions — it consumes `samples.props` once in `pre_quantize`
    /// (which builds `bucket_indices`) and `dedup_samples` (which gathers
    /// compact), then never touches it again. The multipliers path
    /// (`compute_best_tree_with_multipliers`) does NOT satisfy condition 1:
    /// its static-prop axes use [`PartitionKey::Property`] which reads
    /// `samples.props[i]` to evaluate the predicate. Setting this to `true`
    /// on the multipliers path would silently corrupt static-prop splits.
    ///
    /// Per-row savings: ~16-30 `Vec::swap` calls per row swap (one per
    /// populated property — base properties plus per-reference props). At
    /// 1.05 MP e7 with deep trees this is consistently 1-2% wall-clock.
    pub skip_props_swap: bool,
}

impl<'a> SplittableSamples<'a> {
    /// Swap the contents of row `a` and row `b` across every parallel array.
    ///
    /// `a == b` is a no-op. Both indices must be `< len` (debug-checked).
    ///
    /// Empty parallel arrays are skipped silently. Production [`TreeSamples`]
    /// may carry `props[i] = Vec::new()` for property indices that weren't
    /// gathered, and [`PreQuantizedProps::bucket_indices`] holds empty rows
    /// for property indices not in `params.properties`. Both are common —
    /// skipping them lets the caller pass the raw production vectors without
    /// pre-filtering.
    ///
    /// # Cost
    /// `O(num_predictors * 2 + num_props * 2 + 1)` byte/word swaps. With 14
    /// predictors and 16 base properties (plus per-property bucket indices),
    /// this is ~63 element swaps per row swap. Counted toward the partition
    /// cost in the microbench.
    #[inline]
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        debug_assert!(
            a < self.len,
            "row index {} out of range (len={})",
            a,
            self.len
        );
        debug_assert!(
            b < self.len,
            "row index {} out of range (len={})",
            b,
            self.len
        );
        for row in self.residual_tokens.iter_mut() {
            if !row.is_empty() {
                row.swap(a, b);
            }
        }
        for row in self.extra_bits.iter_mut() {
            if !row.is_empty() {
                row.swap(a, b);
            }
        }
        // Issue #40 chunk-3c: skip per-property props swaps on lossless-only
        // call paths. See `skip_props_swap` doc on `SplittableSamples`.
        if !self.skip_props_swap {
            for row in self.props.iter_mut() {
                if !row.is_empty() {
                    row.swap(a, b);
                }
            }
        }
        for row in self.bucket_indices.iter_mut() {
            if !row.is_empty() {
                row.swap(a, b);
            }
        }
        self.sample_counts.swap(a, b);
    }
}

/// Which parallel column to use as the partition key.
///
/// Mirrors the two property paths in libjxl's `SplitTreeSamples<S>`:
/// - `S=true`: `static_props[prop][row]` (we don't carry static props
///   separately yet — they fold into [`PartitionKey::Property`] for chunk 1).
/// - `S=false`: `props[prop][row]`.
///
/// In production wiring (chunk 2) the cost model picks splits from the
/// per-property bucket sweep, so the natural call shape is
/// [`PartitionKey::Bucket`] with the bucket index in
/// `PreQuantizedProps::bucket_indices`. [`PartitionKey::Property`] is here
/// for parity with libjxl's exact code path (used by `partition_indices`
/// today) and is the simpler invariant for tests.
#[derive(Clone, Copy, Debug)]
pub enum PartitionKey {
    /// Partition by `props[prop_idx][row] <= val_i32`.
    Property { prop_idx: usize, val: i32 },
    /// Partition by `bucket_indices[prop_idx][row] <= val_u8`.
    Bucket { prop_idx: usize, val: u8 },
}

impl PartitionKey {
    /// Evaluate the partition predicate at row `i`. Returns `true` if the
    /// row belongs on the LEFT side of the split (mirrors libjxl's
    /// `Property<S>(prop, row) <= val` test).
    #[inline]
    fn matches(&self, samples: &SplittableSamples<'_>, row: usize) -> bool {
        match *self {
            PartitionKey::Property { prop_idx, val } => {
                // Issue #40 chunk-3c: `Property` partition reads `props`,
                // so it MUST see swapped-in-sync values. Skipping the props
                // swap with a `Property` key would partition stale rows.
                debug_assert!(
                    !samples.skip_props_swap,
                    "PartitionKey::Property requires aligned `samples.props`; \
                     caller set skip_props_swap=true (use PartitionKey::Bucket instead)"
                );
                samples.props[prop_idx][row] <= val
            }
            PartitionKey::Bucket { prop_idx, val } => samples.bucket_indices[prop_idx][row] <= val,
        }
    }
}

/// Hoare-style in-place partition primitive matching libjxl's
/// `SplitTreeSamples<S>` in `enc_ma.cc:119-138`.
///
/// Reorganizes rows in `[begin..end)` so that:
/// - Every row in `[begin..pos)` satisfies the partition predicate (key's
///   column value `<= val`).
/// - Every row in `[pos..end)` does not (key's column value `> val`).
///
/// `pos` is provided by the caller. It must equal the count of rows in
/// `[begin..end)` that satisfy the predicate (in the pre-partition layout) —
/// the libjxl cost model produces this count alongside the `(prop, val)`
/// pair as `prop_value_used_count[]`'s cumulative sum.
///
/// # Why a caller-supplied split point
///
/// libjxl's cost model produces `(prop, val, split_point)` together: `val` is
/// the threshold and `split_point` is the count of samples that land on the
/// left. Calling `partition` with that `split_point` reproduces the libjxl
/// behavior exactly. For our standalone tests/microbench, the caller counts
/// matching rows separately and passes the result; in production wiring
/// (chunk 2) this comes from `costs_l`/`prop_value_used_count` cumulants.
///
/// # Panics
/// Debug-asserts that `begin <= pos <= end <= samples.len`.
///
/// # Returns
/// `pos` (echoed back for ergonomics with libjxl's call sites).
///
/// # Algorithm
/// ```text
/// begin_pos = begin
/// end_pos   = pos
/// loop:
///   while begin_pos < pos  and  key.matches(begin_pos): begin_pos += 1
///   while end_pos   < end  and !key.matches(end_pos):   end_pos   += 1
///   if begin_pos < pos and end_pos < end:
///       swap_rows(begin_pos, end_pos)
///   begin_pos += 1
///   end_pos   += 1
///   exit when begin_pos >= pos or end_pos >= end
/// ```
///
/// The trailing unconditional `+= 1` is intentional and matches libjxl: by
/// the time we get there, EITHER the swap happened (both rows are now in the
/// right place, so we can step past both) OR one of the loops hit its bound
/// (so we'll exit on the next iteration anyway). The post-swap step skips
/// the row we just placed; the boundary-hit step is harmless because the
/// next loop iteration will exit.
pub fn split_tree_samples_in_place(
    samples: &mut SplittableSamples<'_>,
    begin: usize,
    pos: usize,
    end: usize,
    key: PartitionKey,
) -> usize {
    debug_assert!(begin <= pos, "begin {} > pos {}", begin, pos);
    debug_assert!(pos <= end, "pos {} > end {}", pos, end);
    debug_assert!(
        end <= samples.len,
        "end {} > samples.len {}",
        end,
        samples.len
    );

    let mut begin_pos = begin;
    let mut end_pos = pos;

    loop {
        // Walk begin_pos forward past rows already on the correct (left) side.
        while begin_pos < pos && key.matches(samples, begin_pos) {
            begin_pos += 1;
        }
        // Walk end_pos forward past rows already on the correct (right) side.
        while end_pos < end && !key.matches(samples, end_pos) {
            end_pos += 1;
        }
        if begin_pos < pos && end_pos < end {
            // Both pointers found a misplaced row — swap them.
            samples.swap_rows(begin_pos, end_pos);
        }
        // Advance unconditionally: post-swap the two rows are settled; on the
        // boundary-hit branch we'll exit the outer loop on the next check.
        begin_pos += 1;
        end_pos += 1;
        if begin_pos >= pos || end_pos >= end {
            break;
        }
    }

    pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a `CanaryStorage` with a single predictor + single property +
    /// single bucket-index column. Row `i` carries the canary value `i` in
    /// every array (modulo the offset multipliers), so a swap is easy to
    /// verify and the prop column directly equals the row's original index.
    fn make_canary(n: usize) -> CanaryStorage {
        let mut residual_tokens = vec![Vec::with_capacity(n)];
        let mut extra_bits = vec![Vec::with_capacity(n)];
        let mut props = vec![Vec::with_capacity(n)];
        let mut bucket_indices = vec![Vec::with_capacity(n)];
        let mut sample_counts = Vec::with_capacity(n);
        for i in 0..n {
            residual_tokens[0].push(i as u8);
            extra_bits[0].push((i.wrapping_mul(7)) as u8);
            props[0].push(i as i32);
            bucket_indices[0].push((i & 0xff) as u8);
            sample_counts.push(i as u32 + 1);
        }
        CanaryStorage {
            residual_tokens,
            extra_bits,
            props,
            bucket_indices,
            sample_counts,
        }
    }

    struct CanaryStorage {
        residual_tokens: Vec<Vec<u8>>,
        extra_bits: Vec<Vec<u8>>,
        props: Vec<Vec<i32>>,
        bucket_indices: Vec<Vec<u8>>,
        sample_counts: Vec<u32>,
    }

    impl CanaryStorage {
        fn view(&mut self) -> SplittableSamples<'_> {
            let len = self.sample_counts.len();
            SplittableSamples {
                residual_tokens: &mut self.residual_tokens,
                extra_bits: &mut self.extra_bits,
                props: &mut self.props,
                bucket_indices: &mut self.bucket_indices,
                sample_counts: &mut self.sample_counts,
                len,
                skip_props_swap: false,
            }
        }

        /// Sanity check: every parallel array at `row` must hold values
        /// derived from the same canary `i` (= `residual_tokens[0][row]`).
        fn assert_row_consistent(&self, row: usize) {
            let canary = self.residual_tokens[0][row];
            let i = canary as usize;
            assert_eq!(self.extra_bits[0][row], (i.wrapping_mul(7)) as u8);
            assert_eq!(self.props[0][row], i as i32);
            assert_eq!(self.bucket_indices[0][row], (i & 0xff) as u8);
            assert_eq!(self.sample_counts[row], i as u32 + 1);
        }
    }

    /// Helper: drive the partition primitive within a borrow scope so the
    /// view's mutable references are released before the assertions read
    /// from `storage`.
    fn partition_canary(
        storage: &mut CanaryStorage,
        begin: usize,
        pos: usize,
        end: usize,
        key: PartitionKey,
    ) -> usize {
        let mut view = storage.view();
        split_tree_samples_in_place(&mut view, begin, pos, end, key)
    }

    #[test]
    fn swap_rows_noop_when_equal() {
        let mut storage = make_canary(4);
        {
            let mut view = storage.view();
            view.swap_rows(2, 2);
        }
        for i in 0..4 {
            assert_eq!(storage.residual_tokens[0][i], i as u8);
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn swap_rows_keeps_soa_aligned() {
        let mut storage = make_canary(8);
        {
            let mut view = storage.view();
            view.swap_rows(1, 6);
        }
        assert_eq!(storage.residual_tokens[0][1], 6);
        assert_eq!(storage.residual_tokens[0][6], 1);
        storage.assert_row_consistent(1);
        storage.assert_row_consistent(6);
        for i in [0usize, 2, 3, 4, 5, 7].iter().copied() {
            assert_eq!(storage.residual_tokens[0][i], i as u8);
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn partition_already_partitioned_is_noop() {
        // Predicate: props[0][row] <= 3 → true for canary 0..3, false for
        // canary 4..7. Rows already ordered, so the partition should be a
        // no-op semantically.
        let mut storage = make_canary(8);
        let returned = partition_canary(
            &mut storage,
            0,
            4,
            8,
            PartitionKey::Property {
                prop_idx: 0,
                val: 3,
            },
        );
        assert_eq!(returned, 4);
        for i in 0..8 {
            assert_eq!(storage.residual_tokens[0][i], i as u8);
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn partition_reversed_full_swap() {
        // Start with canary rotated so rows 0..4 carry 4..7 and rows 4..8
        // carry 0..3, then partition by "prop <= 3" with pos=4. Result: rows
        // 0..4 hold canaries {0,1,2,3} and rows 4..8 hold {4,5,6,7}.
        let mut storage = make_canary(8);
        {
            let mut view = storage.view();
            view.swap_rows(0, 4);
            view.swap_rows(1, 5);
            view.swap_rows(2, 6);
            view.swap_rows(3, 7);
        }
        let returned = partition_canary(
            &mut storage,
            0,
            4,
            8,
            PartitionKey::Property {
                prop_idx: 0,
                val: 3,
            },
        );
        assert_eq!(returned, 4);
        let mut left: Vec<u8> = (0..4).map(|i| storage.residual_tokens[0][i]).collect();
        let mut right: Vec<u8> = (4..8).map(|i| storage.residual_tokens[0][i]).collect();
        left.sort_unstable();
        right.sort_unstable();
        assert_eq!(left, vec![0u8, 1, 2, 3]);
        assert_eq!(right, vec![4u8, 5, 6, 7]);
        for i in 0..8 {
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn partition_interleaved_predicate() {
        // Build storage where props[1][i] = i % 2 (even → 0, odd → 1).
        // Predicate "prop[1] <= 0" matches even canaries; 8 of them in [0..16).
        let mut storage = make_canary(16);
        storage.props.push((0..16i32).map(|i| i % 2).collect());
        storage
            .bucket_indices
            .push((0..16).map(|i| (i & 1) as u8).collect());
        let returned = partition_canary(
            &mut storage,
            0,
            8,
            16,
            PartitionKey::Property {
                prop_idx: 1,
                val: 0,
            },
        );
        assert_eq!(returned, 8);
        let mut left: Vec<u8> = (0..8).map(|i| storage.residual_tokens[0][i]).collect();
        let mut right: Vec<u8> = (8..16).map(|i| storage.residual_tokens[0][i]).collect();
        left.sort_unstable();
        right.sort_unstable();
        assert_eq!(left, vec![0u8, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(right, vec![1u8, 3, 5, 7, 9, 11, 13, 15]);
        for i in 0..16 {
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn partition_within_a_subrange() {
        // Partition only rows [4..12) of a 16-row sample set, leaving rows
        // [0..4) and [12..16) untouched. Predicate "props[1] <= 0" matches
        // even canaries; 4 in [4..12) (4, 6, 8, 10).
        let mut storage = make_canary(16);
        storage.props.push((0..16i32).map(|i| i % 2).collect());
        storage
            .bucket_indices
            .push((0..16).map(|i| (i & 1) as u8).collect());
        let pre_left: Vec<u8> = (0..4).map(|i| storage.residual_tokens[0][i]).collect();
        let pre_right: Vec<u8> = (12..16).map(|i| storage.residual_tokens[0][i]).collect();
        let returned = partition_canary(
            &mut storage,
            4,
            8,
            12,
            PartitionKey::Property {
                prop_idx: 1,
                val: 0,
            },
        );
        assert_eq!(returned, 8);
        let now_left: Vec<u8> = (0..4).map(|i| storage.residual_tokens[0][i]).collect();
        let now_right: Vec<u8> = (12..16).map(|i| storage.residual_tokens[0][i]).collect();
        assert_eq!(now_left, pre_left);
        assert_eq!(now_right, pre_right);
        let mut left: Vec<u8> = (4..8).map(|i| storage.residual_tokens[0][i]).collect();
        let mut right: Vec<u8> = (8..12).map(|i| storage.residual_tokens[0][i]).collect();
        left.sort_unstable();
        right.sort_unstable();
        assert_eq!(left, vec![4u8, 6, 8, 10]);
        assert_eq!(right, vec![5u8, 7, 9, 11]);
        for i in 0..16 {
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn partition_full_left_no_swaps_needed() {
        // pos == end: nothing on the right of the partition.
        let mut storage = make_canary(8);
        let returned = partition_canary(
            &mut storage,
            0,
            8,
            8,
            PartitionKey::Property {
                prop_idx: 0,
                val: 1000,
            },
        );
        assert_eq!(returned, 8);
        for i in 0..8 {
            assert_eq!(storage.residual_tokens[0][i], i as u8);
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn partition_full_right_no_swaps_needed() {
        // pos == begin: nothing on the left.
        let mut storage = make_canary(8);
        let returned = partition_canary(
            &mut storage,
            0,
            0,
            8,
            PartitionKey::Property {
                prop_idx: 0,
                val: -1,
            },
        );
        assert_eq!(returned, 0);
        for i in 0..8 {
            assert_eq!(storage.residual_tokens[0][i], i as u8);
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn partition_by_bucket_indices() {
        // Same logic as `partition_interleaved_predicate`, but routed
        // through PartitionKey::Bucket instead of Property — proves the
        // primitive supports the chunk-2 production call shape (bucket
        // indices from PreQuantizedProps).
        let mut storage = make_canary(16);
        storage
            .bucket_indices
            .push((0..16).map(|i| (i & 1) as u8).collect());
        let returned = partition_canary(
            &mut storage,
            0,
            8,
            16,
            PartitionKey::Bucket {
                prop_idx: 1,
                val: 0,
            },
        );
        assert_eq!(returned, 8);
        let mut left: Vec<u8> = (0..8).map(|i| storage.residual_tokens[0][i]).collect();
        let mut right: Vec<u8> = (8..16).map(|i| storage.residual_tokens[0][i]).collect();
        left.sort_unstable();
        right.sort_unstable();
        assert_eq!(left, vec![0u8, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(right, vec![1u8, 3, 5, 7, 9, 11, 13, 15]);
        for i in 0..16 {
            storage.assert_row_consistent(i);
        }
    }

    #[test]
    fn partition_multi_predictor_multi_prop_soa_alignment() {
        // Realistic shape: 14 predictors × 16 base properties + bucket
        // indices for each prop, 256 rows, non-trivial predicate. Verifies
        // that all 14+14+16+16 parallel arrays remain row-aligned after the
        // swap-heavy partition.
        let n = 256;
        let num_pred = 14;
        let num_props = 16;
        let mut residual_tokens: Vec<Vec<u8>> = (0..num_pred)
            .map(|p| (0..n).map(|i| ((i + p) & 0xff) as u8).collect())
            .collect();
        let mut extra_bits: Vec<Vec<u8>> = (0..num_pred)
            .map(|p: usize| {
                (0..n)
                    .map(|i: usize| ((i.wrapping_mul(13).wrapping_add(p)) & 0xff) as u8)
                    .collect()
            })
            .collect();
        let mut props: Vec<Vec<i32>> = (0..num_props)
            .map(|p| (0..n).map(|i| (i as i32) + (p as i32) * 1000).collect())
            .collect();
        let mut bucket_indices: Vec<Vec<u8>> = (0..num_props)
            .map(|p| (0..n).map(|i| ((i + p) & 0xff) as u8).collect())
            .collect();
        let mut sample_counts: Vec<u32> = (0..n).map(|i| (i as u32) + 1).collect();

        // Per-row signature: depends only on the row's content, so swaps
        // preserve the multiset of signatures.
        let signature_at = |row: usize,
                            residual_tokens: &[Vec<u8>],
                            extra_bits: &[Vec<u8>],
                            props: &[Vec<i32>],
                            bucket_indices: &[Vec<u8>],
                            sample_counts: &[u32]|
         -> (u64, u64) {
            let mut a: u64 = sample_counts[row] as u64;
            for (p, col) in residual_tokens.iter().enumerate() {
                a = a
                    .wrapping_mul(31)
                    .wrapping_add((col[row] as u64) ^ ((p as u64) << 8));
            }
            for (p, col) in extra_bits.iter().enumerate() {
                a = a
                    .wrapping_mul(31)
                    .wrapping_add((col[row] as u64) ^ ((p as u64) << 16));
            }
            let mut b: u64 = 0;
            for (p, col) in props.iter().enumerate() {
                b = b
                    .wrapping_mul(37)
                    .wrapping_add(col[row].wrapping_abs() as u64 ^ ((p as u64) << 4));
            }
            for (p, col) in bucket_indices.iter().enumerate() {
                b = b
                    .wrapping_mul(37)
                    .wrapping_add((col[row] as u64) ^ ((p as u64) << 12));
            }
            (a, b)
        };

        let pre_signatures: Vec<(u64, u64)> = (0..n)
            .map(|i| {
                signature_at(
                    i,
                    &residual_tokens,
                    &extra_bits,
                    &props,
                    &bucket_indices,
                    &sample_counts,
                )
            })
            .collect();

        // Partition by props[0] (which holds canary values 0..n): predicate
        // "props[0] <= 127" matches exactly the first 128 rows in the
        // PRE-partition layout. With pos=128, the result should land those
        // 128 rows in [0..128) and the rest in [128..256).
        let pos = n / 2;
        let returned = {
            let mut view = SplittableSamples {
                residual_tokens: &mut residual_tokens,
                extra_bits: &mut extra_bits,
                props: &mut props,
                bucket_indices: &mut bucket_indices,
                sample_counts: &mut sample_counts,
                len: n,
                skip_props_swap: false,
            };
            // Make the layout non-trivial: reverse the order so the partition
            // does real work.
            let mut a = 0;
            let mut b = n - 1;
            while a < b {
                view.swap_rows(a, b);
                a += 1;
                b -= 1;
            }
            split_tree_samples_in_place(
                &mut view,
                0,
                pos,
                n,
                PartitionKey::Property {
                    prop_idx: 0,
                    val: 127,
                },
            )
        };
        assert_eq!(returned, pos);

        // Predicate must hold on the left, fail on the right.
        for (i, v) in props[0][..pos].iter().enumerate() {
            assert!(*v <= 127, "row {i} should have props[0] <= 127 but is {v}");
        }
        for (offset, v) in props[0][pos..n].iter().enumerate() {
            let i = pos + offset;
            assert!(*v > 127, "row {i} should have props[0] > 127 but is {v}");
        }

        // SoA alignment check: every post-permutation row signature must
        // equal exactly one pre-permutation row signature.
        let post_signatures: Vec<(u64, u64)> = (0..n)
            .map(|i| {
                signature_at(
                    i,
                    &residual_tokens,
                    &extra_bits,
                    &props,
                    &bucket_indices,
                    &sample_counts,
                )
            })
            .collect();
        let mut pre_sorted = pre_signatures.clone();
        let mut post_sorted = post_signatures.clone();
        pre_sorted.sort_unstable();
        post_sorted.sort_unstable();
        assert_eq!(
            pre_sorted, post_sorted,
            "SoA rows must be preserved as atomic units across the permutation"
        );
    }

    #[test]
    fn partition_pseudorandom_property_values() {
        // Pseudo-random property values + caller-supplied split count.
        // Verifies the primitive partitions correctly even when the values
        // are not sorted and the swap pattern is non-trivial.
        let n = 1024;
        let mut residual_tokens: Vec<Vec<u8>> = vec![(0..n).map(|i| (i & 0xff) as u8).collect()];
        let mut extra_bits: Vec<Vec<u8>> = vec![(0..n).map(|i| ((i * 7) & 0xff) as u8).collect()];
        // Property column: pseudo-random in [0, 255].
        let mut state: u32 = 0x12345678;
        let mut prop_col: Vec<i32> = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            prop_col.push((state >> 24) as i32);
        }
        let val = 127i32;
        let expected_left_count = prop_col.iter().filter(|&&v| v <= val).count();
        let mut props: Vec<Vec<i32>> = vec![prop_col.clone()];
        let mut bucket_indices: Vec<Vec<u8>> = vec![(0..n).map(|i| (i & 0xff) as u8).collect()];
        let mut sample_counts: Vec<u32> = (0..n).map(|i| (i as u32) + 1).collect();

        let returned = {
            let mut view = SplittableSamples {
                residual_tokens: &mut residual_tokens,
                extra_bits: &mut extra_bits,
                props: &mut props,
                bucket_indices: &mut bucket_indices,
                sample_counts: &mut sample_counts,
                len: n,
                skip_props_swap: false,
            };
            split_tree_samples_in_place(
                &mut view,
                0,
                expected_left_count,
                n,
                PartitionKey::Property { prop_idx: 0, val },
            )
        };
        assert_eq!(returned, expected_left_count);

        for (i, v) in props[0][..expected_left_count].iter().enumerate() {
            assert!(
                *v <= val,
                "row {i} should be left of split but props[0]={v}"
            );
        }
        for (offset, v) in props[0][expected_left_count..n].iter().enumerate() {
            let i = expected_left_count + offset;
            assert!(
                *v > val,
                "row {i} should be right of split but props[0]={v}"
            );
        }
    }

    /// Issue #40 chunk-3c: `skip_props_swap=true` must:
    /// 1. Leave `props` in its pre-permutation order (every prop stays at its
    ///    original row index — no swaps happened).
    /// 2. Still permute `bucket_indices`, `residual_tokens`, `extra_bits`, and
    ///    `sample_counts` in lockstep so the bucket partition is correct.
    #[test]
    fn skip_props_swap_partitions_bucket_indices_and_leaves_props_untouched() {
        // 16 rows, props[0][i] = i (canary), bucket_indices[0][i] = i % 2.
        // Partition by bucket value <= 0 (matches even rows). pos = 8.
        // After partition: bucket_indices[0][..8] all 0, [8..] all 1.
        // props[0] MUST still equal 0..16 in order (no swaps).
        let n = 16usize;
        let mut residual_tokens: Vec<Vec<u8>> = vec![(0..n).map(|i| i as u8).collect()];
        let mut extra_bits: Vec<Vec<u8>> =
            vec![(0..n).map(|i| (i.wrapping_mul(7)) as u8).collect()];
        let mut props: Vec<Vec<i32>> = vec![(0..n as i32).collect()];
        let mut bucket_indices: Vec<Vec<u8>> = vec![(0..n).map(|i| (i & 1) as u8).collect()];
        let mut sample_counts: Vec<u32> = (0..n).map(|i| i as u32 + 1).collect();

        let props_snapshot = props[0].clone();

        let returned = {
            let mut view = SplittableSamples {
                residual_tokens: &mut residual_tokens,
                extra_bits: &mut extra_bits,
                props: &mut props,
                bucket_indices: &mut bucket_indices,
                sample_counts: &mut sample_counts,
                len: n,
                skip_props_swap: true,
            };
            split_tree_samples_in_place(
                &mut view,
                0,
                8,
                n,
                PartitionKey::Bucket {
                    prop_idx: 0,
                    val: 0,
                },
            )
        };
        assert_eq!(returned, 8);

        // bucket_indices partitioned correctly.
        for (i, &b) in bucket_indices[0][..8].iter().enumerate() {
            assert_eq!(b, 0, "left row {i} should have bucket=0 but is {b}");
        }
        for (offset, &b) in bucket_indices[0][8..].iter().enumerate() {
            let i = 8 + offset;
            assert_eq!(b, 1, "right row {i} should have bucket=1 but is {b}");
        }

        // props was NOT swapped — still in pre-permutation order.
        assert_eq!(
            props[0], props_snapshot,
            "props must be untouched when skip_props_swap=true"
        );

        // residual_tokens / extra_bits / sample_counts moved in lockstep
        // with bucket_indices (so the row content for each post-partition
        // index matches the row content of the pre-partition source row).
        // For our canary: residual_tokens[0][i] == original-row-index, so
        // even rows (originally 0,2,4,6,8,10,12,14) should now land in [0..8).
        let mut left_origin: Vec<u8> = residual_tokens[0][..8].to_vec();
        let mut right_origin: Vec<u8> = residual_tokens[0][8..].to_vec();
        left_origin.sort_unstable();
        right_origin.sort_unstable();
        assert_eq!(left_origin, vec![0u8, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(right_origin, vec![1u8, 3, 5, 7, 9, 11, 13, 15]);

        // Cross-array consistency check: for every row, sample_counts[i]
        // must match the (original_row_index + 1) where original_row_index
        // is recoverable from residual_tokens[0][i].
        for i in 0..n {
            let orig = residual_tokens[0][i] as usize;
            assert_eq!(
                sample_counts[i],
                orig as u32 + 1,
                "sample_counts row {i} (orig {orig}) misaligned",
            );
            assert_eq!(
                extra_bits[0][i],
                orig.wrapping_mul(7) as u8,
                "extra_bits row {i} (orig {orig}) misaligned",
            );
        }
    }

    /// Issue #40 chunk-3c: `skip_props_swap=true` with [`PartitionKey::Property`]
    /// is an API misuse — props are not swapped, so a Property predicate would
    /// read stale row content. Caught by debug-assert at partition time.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "PartitionKey::Property requires aligned `samples.props`")]
    fn skip_props_swap_with_property_key_panics_in_debug() {
        let n = 8usize;
        let mut residual_tokens: Vec<Vec<u8>> = vec![(0..n).map(|i| i as u8).collect()];
        let mut extra_bits: Vec<Vec<u8>> = vec![(0..n).map(|i| i as u8).collect()];
        let mut props: Vec<Vec<i32>> = vec![(0..n as i32).collect()];
        let mut bucket_indices: Vec<Vec<u8>> = vec![(0..n).map(|i| i as u8).collect()];
        let mut sample_counts: Vec<u32> = (0..n as u32).collect();
        let mut view = SplittableSamples {
            residual_tokens: &mut residual_tokens,
            extra_bits: &mut extra_bits,
            props: &mut props,
            bucket_indices: &mut bucket_indices,
            sample_counts: &mut sample_counts,
            len: n,
            skip_props_swap: true,
        };
        // Use a non-already-partitioned layout so the predicate is actually
        // evaluated (otherwise the Hoare scan would exit before any call to
        // `matches`).
        view.swap_rows(0, 7);
        split_tree_samples_in_place(
            &mut view,
            0,
            4,
            n,
            PartitionKey::Property {
                prop_idx: 0,
                val: 3,
            },
        );
    }
}
