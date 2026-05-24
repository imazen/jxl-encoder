// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-distance CVVDP JOD calibration table (cvvdp-fork Phase 4,
//! 2026-05-24 — see `docs/RFC_CVVDP_FORK.md` §2.1, §4 Phase 4 and
//! `docs/RFC_CVVDP_PHASE4_BRIEF.md` Step 3).
//!
//! ## Purpose
//!
//! When the buttloop runs with the [GPU CVVDP backend] active, the
//! per-iter compare returns `score = (10.0 - JOD).clamp(0.0, 10.0)`
//! (mapped from cvvdp-gpu's native JOD ∈ [0, 10] where 10 = identical).
//! The buttloop's pre-existing convergence machinery consumes a
//! `target_distance` in butteraugli units (smaller = better, ~ [0, 5]
//! typical band). To keep the comparison surface coherent across
//! backends, when cvvdp is active the loop substitutes a
//! **cvvdp-native target** (in `score = 10 - JOD` space) interpolated
//! from this table at the caller-requested butteraugli `target_distance`.
//!
//! [GPU CVVDP backend]: super::cvvdp_backend::gpu::GpuCvvdpBackend
//!
//! ## Seed methodology
//!
//! Source: `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
//! (Agent D's baseline, 1,131 backend=B / butteraugli-default rows
//! covering CID22 + GB82-SC + CLIC corpora across `distance ∈ {0.5,
//! 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}`). The pre-computed seed values are
//! checked-in at `benchmarks/cvvdp_jod_calibration_seed_2026-05-24.txt`;
//! the table below was reviewed and pasted from there.
//!
//! For each distance band, the median `score_cvvdp_gpu` (JOD ∈ [0, 10])
//! was computed across the corpus rows at that distance, then converted
//! to butteraugli-direction `score = (10.0 - jod) * 1.05` — the 1.05×
//! tightening biases the cvvdp-driven loop to converge slightly more
//! strictly than what the butteraugli-driven baseline achieves at the
//! same nominal distance. The 1.05× factor is small enough that the
//! cvvdp loop should still terminate in the same iteration count as
//! butteraugli on typical inputs, while large enough to trigger
//! refinement on cells where cvvdp and butteraugli disagree on the
//! per-block bad-block set.
//!
//! ## Linear interpolation
//!
//! [`cvvdp_target_score_for_distance`] interpolates linearly between
//! adjacent table points and clamps outside the band (`[0.50, 5.00]`).
//! The table is monotone (target grows with distance) so interpolation
//! is well-defined.

/// Per-distance CVVDP JOD targets, seeded from Agent D's tracking
/// baseline (1,131 cells of butteraugli-default encoder output scored
/// with cvvdp-gpu). Each target is the median cvvdp JOD across the
/// corpus at that distance, scaled 1.05× tighter so the cvvdp-driven
/// loop is slightly more demanding than what butteraugli converges to.
/// Converted to butteraugli-direction score via `score = (10.0 - jod).clamp(0.0, 10.0)`
/// then multiplied by 1.05 — see module docs for full methodology.
///
/// Format: `(butteraugli_target_distance, cvvdp_target_score)`.
/// Table is sorted ascending by distance; lookup is linear interp +
/// clamp.
pub(crate) static CVVDP_DISTANCE_TARGETS: &[(f32, f32)] = &[
    (0.50, 0.0029), // n=162, median JOD = 9.9972
    (1.00, 0.0238), // n=162, median JOD = 9.9774
    (1.50, 0.0461), // n=162, median JOD = 9.9561
    (2.00, 0.0724), // n=162, median JOD = 9.9311
    (3.00, 0.1336), // n=162, median JOD = 9.8728
    (4.00, 0.2149), // n=159, median JOD = 9.7953
    (5.00, 0.3005), // n=162, median JOD = 9.7138
];

/// Look up the CVVDP `score = 10 - JOD` target for a given butteraugli
/// `target_distance`. Linear interpolation between adjacent table points;
/// clamps to the boundary value outside `[CVVDP_DISTANCE_TARGETS[0].0,
/// CVVDP_DISTANCE_TARGETS[last].0]`.
///
/// Returns the lookup value in score-direction (smaller = better),
/// directly comparable to the `BackendCompareResult::score` produced by
/// the cvvdp GPU backend.
///
/// # Examples
///
/// ```ignore
/// // Exact table point.
/// let s = cvvdp_target_score_for_distance(1.0);
/// assert!((s - 0.0238).abs() < 1e-6);
///
/// // Interpolation: distance=1.25 is halfway between 1.0 (0.0238) and
/// // 1.5 (0.0461) → expected ~0.03495.
/// let s = cvvdp_target_score_for_distance(1.25);
/// assert!((s - 0.03495).abs() < 1e-4);
///
/// // Below-band clamp: distance=0.25 → returns the d=0.5 value.
/// let s = cvvdp_target_score_for_distance(0.25);
/// assert!((s - 0.0029).abs() < 1e-6);
///
/// // Above-band clamp: distance=10.0 → returns the d=5.0 value.
/// let s = cvvdp_target_score_for_distance(10.0);
/// assert!((s - 0.3005).abs() < 1e-6);
/// ```
pub(crate) fn cvvdp_target_score_for_distance(target_distance: f32) -> f32 {
    let table = CVVDP_DISTANCE_TARGETS;
    debug_assert!(
        !table.is_empty(),
        "CVVDP_DISTANCE_TARGETS must be non-empty"
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
        let table = CVVDP_DISTANCE_TARGETS;
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
    /// implies a larger acceptable cvvdp score.
    #[test]
    fn table_score_monotone() {
        let table = CVVDP_DISTANCE_TARGETS;
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
        let table = CVVDP_DISTANCE_TARGETS;
        let first = table[0];
        let last = table[table.len() - 1];
        let s_first = cvvdp_target_score_for_distance(first.0);
        let s_last = cvvdp_target_score_for_distance(last.0);
        assert!(
            (s_first - first.1).abs() < 1e-6,
            "lookup at first entry must return its value: got {} vs {}",
            s_first,
            first.1
        );
        assert!(
            (s_last - last.1).abs() < 1e-6,
            "lookup at last entry must return its value: got {} vs {}",
            s_last,
            last.1
        );
    }

    /// Below-band clamp to the first entry; above-band clamp to the last.
    #[test]
    fn lookup_outside_band_clamps() {
        let table = CVVDP_DISTANCE_TARGETS;
        let first_v = table[0].1;
        let last_v = table[table.len() - 1].1;
        assert!((cvvdp_target_score_for_distance(0.0) - first_v).abs() < 1e-6);
        assert!((cvvdp_target_score_for_distance(0.25) - first_v).abs() < 1e-6);
        assert!((cvvdp_target_score_for_distance(10.0) - last_v).abs() < 1e-6);
        assert!((cvvdp_target_score_for_distance(100.0) - last_v).abs() < 1e-6);
    }

    /// Linear interpolation: midpoint of (1.0, 0.0238) and (1.5, 0.0461)
    /// is (1.25, 0.03495).
    #[test]
    fn lookup_interpolates_linearly() {
        let s = cvvdp_target_score_for_distance(1.25);
        let expected = (0.0238 + 0.0461) * 0.5;
        assert!(
            (s - expected).abs() < 1e-4,
            "interp at d=1.25: got {} expected {}",
            s,
            expected
        );

        // Quarter-point between d=2.0 (0.0724) and d=3.0 (0.1336):
        // d=2.25 → 0.0724 + 0.25 * (0.1336 - 0.0724) = 0.0877
        let s = cvvdp_target_score_for_distance(2.25);
        let expected = 0.0724 + 0.25 * (0.1336 - 0.0724);
        assert!(
            (s - expected).abs() < 1e-4,
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
            let s = cvvdp_target_score_for_distance(d);
            assert!(
                s.is_finite() && s >= 0.0,
                "lookup at d={} returned {} (finite={}, >=0={})",
                d,
                s,
                s.is_finite(),
                s >= 0.0
            );
            assert!(
                s <= 1.0,
                "lookup at d={} returned {} > 1.0 (table max ~0.3)",
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
            CVVDP_DISTANCE_TARGETS.len(),
            7,
            "table should have 7 entries (d ∈ {{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}}) — \
             changing this requires a refit chunk + RFC update"
        );
    }
}
