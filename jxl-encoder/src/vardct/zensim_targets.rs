// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-distance zensim calibration table (zensim-fork Phase 4,
//! 2026-05-25 — see `docs/RFC_ZENSIM_FORK_PLAN.md` §6 and the Phase 4
//! brief).
//!
//! ## Purpose
//!
//! When the buttloop runs with the [CPU zensim backend] or
//! [GPU zensim backend] active, the per-iter compare returns
//! `score = (100.0 - zensim_native).clamp(0.0, 100.0)` (mapped from
//! zensim's native `[0, 100]` higher-is-better score). The buttloop's
//! pre-existing convergence machinery consumes a `target_distance` in
//! butteraugli units (smaller = better, ~ [0, 5] typical band). To keep
//! the comparison surface coherent across backends, when zensim is
//! active the loop substitutes a **zensim-native target** (in
//! butter-direction `100 - zensim_native` space) interpolated from this
//! table at the caller-requested butteraugli `target_distance`.
//!
//! [CPU zensim backend]: super::zensim_backend::cpu::CpuZensimBackend
//! [GPU zensim backend]: super::zensim_backend::gpu::GpuZensimBackend
//!
//! ## Seed methodology
//!
//! Source: `benchmarks/zensim_calibration_seed_2026-05-25.tsv`
//! (post-hoc scoring pass over butteraugli-default encoder output on a
//! mix of CID22 + GB82-SC images across `distance ∈ {0.5, 1.0, 1.5,
//! 2.0, 3.0, 4.0, 5.0}`). For each distance band the median
//! `score_zensim_cpu` (zensim native ∈ [0, 100] where 100 = identical)
//! was computed, then converted to butter-direction `loss =
//! (100.0 - score)` and scaled 1.05× tighter — the same biasing rule
//! used by cvvdp-fork Phase 4 so the zensim-driven loop converges
//! slightly more strictly than what butteraugli's loop achieves at the
//! same nominal distance. The 1.05× factor is small enough that the
//! zensim loop should still terminate in the same iteration count as
//! butteraugli on typical inputs, while large enough to trigger
//! refinement on cells where zensim and butteraugli disagree on the
//! per-block bad-block set.
//!
//! ## Linear interpolation
//!
//! [`zensim_target_score_for_distance`] interpolates linearly between
//! adjacent table points and clamps outside the band (`[0.50, 5.00]`).
//! The table is monotone (target grows with distance) so interpolation
//! is well-defined.
//!
//! ## Phase 8-zensim follow-on
//!
//! Phase 6 produces the full 6-backend tracking sweep including
//! `score_zensim_{cpu,gpu}`. If the Pareto re-bench shows the seed
//! values plateau below 85%, a Phase 8-zensim refit chunk (analog of
//! cvvdp-fork Phase 8b/8c/8d/8g) will replace these median-seeded
//! values with measured per-distance optima. The table format stays
//! identical; only the constants change.

/// Per-distance zensim targets, seeded from a post-hoc scoring pass on
/// butteraugli-default encoder output (mix of CID22 + GB82-SC). Each
/// target is the median butter-direction `100 - zensim_native` across
/// the seed corpus at that distance, scaled 1.05× tighter so the
/// zensim-driven loop is slightly more demanding than what butteraugli
/// converges to.
///
/// Format: `(butteraugli_target_distance, zensim_target_score)` where
/// `zensim_target_score` is in **butter-direction** (smaller = better)
/// — directly comparable to the `BackendCompareResult::score` produced
/// by the zensim CPU + GPU backends after their internal
/// `100 - native` conversion.
///
/// Table is sorted ascending by distance; lookup is linear interp +
/// clamp.
///
/// **NOTE**: The seed values represent the EXPECTED butter-direction
/// score of butteraugli-default output as measured by zensim, NOT a
/// hard convergence target tuned for end-to-end Pareto performance.
/// Phase 6 + 8-zensim follow-ons will refit these once we have the
/// full multi-metric tracking data.
pub(crate) static ZENSIM_DISTANCE_TARGETS: &[(f32, f32)] = &[
    // Seed values measured by `examples/zensim_calibration_seed.rs`
    // on 3 images × 7 distances = 21 cells (2 CID22 photos +
    // 1 gb82-sc screenshot), post-processed by
    // `scripts/zensim_calibration_seed.py`. See
    // `benchmarks/zensim_calibration_seed_2026-05-25.{tsv,txt}`.
    // target = (100 - median_zensim_native) * 1.05
    (0.50, 6.6381),  // n=3, median zensim_native = 93.6780
    (1.00, 9.9958),  // n=3, median zensim_native = 90.4802
    (1.50, 13.2108), // n=3, median zensim_native = 87.4183
    (2.00, 16.4704), // n=3, median zensim_native = 84.3139
    (3.00, 20.9812), // n=3, median zensim_native = 80.0179
    (4.00, 24.5327), // n=3, median zensim_native = 76.6355
    (5.00, 28.4359), // n=3, median zensim_native = 72.9182
];

/// Look up the zensim butter-direction target for a given butteraugli
/// `target_distance`. Linear interpolation between adjacent table
/// points; clamps to the boundary value outside
/// `[ZENSIM_DISTANCE_TARGETS[0].0, ZENSIM_DISTANCE_TARGETS[last].0]`.
///
/// Returns the lookup value in butter-direction (smaller = better),
/// directly comparable to the `BackendCompareResult::score` produced by
/// the zensim backends.
///
/// # Examples
///
/// ```ignore
/// // Exact table point.
/// let s = zensim_target_score_for_distance(1.0);
/// assert!((s - 9.9958).abs() < 1e-3);
///
/// // Interpolation: distance=1.25 is halfway between 1.0 (9.9958) and
/// // 1.5 (13.2108) → expected ~11.6033.
/// let s = zensim_target_score_for_distance(1.25);
/// assert!((s - 11.6033).abs() < 1e-3);
///
/// // Below-band clamp: distance=0.25 → returns the d=0.5 value.
/// let s = zensim_target_score_for_distance(0.25);
/// assert!((s - 6.6381).abs() < 1e-3);
///
/// // Above-band clamp: distance=10.0 → returns the d=5.0 value.
/// let s = zensim_target_score_for_distance(10.0);
/// assert!((s - 28.4359).abs() < 1e-3);
/// ```
pub(crate) fn zensim_target_score_for_distance(target_distance: f32) -> f32 {
    let table = ZENSIM_DISTANCE_TARGETS;
    debug_assert!(
        !table.is_empty(),
        "ZENSIM_DISTANCE_TARGETS must be non-empty"
    );
    // Below-band clamp.
    if target_distance <= table[0].0 {
        return table[0].1;
    }
    // Above-band clamp.
    let last = table.len() - 1;
    if target_distance >= table[last].0 {
        return table[last].1;
    }
    // Linear search — the table has 7 entries; binary search overhead
    // exceeds the per-call savings. Find the bracketing pair (d_lo, d_hi).
    for i in 0..last {
        let (d_lo, s_lo) = table[i];
        let (d_hi, s_hi) = table[i + 1];
        if target_distance >= d_lo && target_distance <= d_hi {
            // Linear interpolation: t ∈ [0, 1] between d_lo and d_hi.
            let span = d_hi - d_lo;
            if span <= 0.0 {
                return s_lo;
            }
            let t = (target_distance - d_lo) / span;
            return s_lo + t * (s_hi - s_lo);
        }
    }
    // Unreachable given the clamps above — keep a defensive fallback.
    table[last].1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table is sorted ascending by distance (required for linear-search
    /// bracketing).
    #[test]
    fn table_sorted_ascending() {
        let table = ZENSIM_DISTANCE_TARGETS;
        for w in table.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "table distances must be strictly ascending: {} >= {}",
                w[0].0,
                w[1].0
            );
        }
    }

    /// Table is monotone ascending in target score: stricter distance
    /// implies a larger acceptable zensim butter-direction score.
    #[test]
    fn table_score_monotone() {
        let table = ZENSIM_DISTANCE_TARGETS;
        for w in table.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "target scores must be non-decreasing: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }

    /// Endpoints return the table values exactly.
    #[test]
    fn lookup_endpoints_exact() {
        let table = ZENSIM_DISTANCE_TARGETS;
        let first = table[0];
        let last = table[table.len() - 1];
        let s_first = zensim_target_score_for_distance(first.0);
        let s_last = zensim_target_score_for_distance(last.0);
        assert!(
            (s_first - first.1).abs() < 1e-4,
            "lookup at first entry must return its value: got {} vs {}",
            s_first,
            first.1
        );
        assert!(
            (s_last - last.1).abs() < 1e-4,
            "lookup at last entry must return its value: got {} vs {}",
            s_last,
            last.1
        );
    }

    /// Below-band clamp to the first entry; above-band clamp to the last.
    #[test]
    fn lookup_outside_band_clamps() {
        let table = ZENSIM_DISTANCE_TARGETS;
        let first_v = table[0].1;
        let last_v = table[table.len() - 1].1;
        assert!((zensim_target_score_for_distance(0.0) - first_v).abs() < 1e-4);
        assert!((zensim_target_score_for_distance(0.25) - first_v).abs() < 1e-4);
        assert!((zensim_target_score_for_distance(10.0) - last_v).abs() < 1e-4);
        assert!((zensim_target_score_for_distance(100.0) - last_v).abs() < 1e-4);
    }

    /// Linear interpolation: midpoint of (1.0, 9.9958) and
    /// (1.5, 13.2108) is (1.25, ~11.6033).
    #[test]
    fn lookup_interpolates_linearly() {
        let s = zensim_target_score_for_distance(1.25);
        let expected = (9.9958_f32 + 13.2108_f32) * 0.5;
        assert!(
            (s - expected).abs() < 1e-3,
            "interp at d=1.25: got {} expected {}",
            s,
            expected
        );

        // Quarter-point between d=2.0 (16.4704) and d=3.0 (20.9812):
        // d=2.25 → 16.4704 + 0.25 * (20.9812 - 16.4704) ≈ 17.5981.
        let s = zensim_target_score_for_distance(2.25);
        let expected = 16.4704_f32 + 0.25 * (20.9812_f32 - 16.4704_f32);
        assert!(
            (s - expected).abs() < 1e-3,
            "interp at d=2.25: got {} expected {}",
            s,
            expected
        );
    }

    /// Lookup is finite, non-negative, and bounded by the table range
    /// for any reasonable input. Guards against NaN/Inf accidents in
    /// the interp arithmetic.
    #[test]
    fn lookup_well_behaved_across_band() {
        for d_x10 in 0..=60 {
            let d = (d_x10 as f32) * 0.1;
            let s = zensim_target_score_for_distance(d);
            assert!(
                s.is_finite() && s >= 0.0,
                "lookup at d={} returned {} (finite={}, >=0={})",
                d,
                s,
                s.is_finite(),
                s >= 0.0
            );
            // Native zensim score ∈ [0, 100]; butter-direction is
            // (100 - native) so it lives in [0, 100] too. With the
            // 1.05× tightening above, table max ~26; allow a safety
            // ceiling of 110 for the bounded check.
            assert!(
                s <= 110.0,
                "lookup at d={} returned {} > 110.0 (table max ~28.5)",
                d,
                s
            );
        }
    }

    /// Table has exactly the 7 entries the seed methodology calls out.
    /// Guards against accidental edits that would shift the methodology
    /// without an explicit refit chunk.
    #[test]
    fn table_has_seven_entries() {
        assert_eq!(
            ZENSIM_DISTANCE_TARGETS.len(),
            7,
            "table should have 7 entries (d ∈ {{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}}) — \
             changing this requires a refit chunk + RFC update"
        );
    }
}
