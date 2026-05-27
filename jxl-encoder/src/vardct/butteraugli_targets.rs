// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Butteraugli `target_score → effective_distance` calibration table
//! (RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`, Phase 1, 2026-05-26).
//!
//! ## Purpose
//!
//! Closes the implicit-identity gap that the cvvdp arc surfaced in the
//! multi-metric API. Before Phase 1, calling
//! [`crate::api::LossyConfig::with_perceptual_target_score(Some(score))`]
//! was a phantom no-op: the field was stored on the config, threaded
//! into [`crate::vardct::perceptual_backend::MetricSelection`] — and
//! then DISCARDED at
//! [`crate::vardct::perceptual_backend::construct_backend`] (the
//! `let _ = selection.target_score` binding). The buttloop body's
//! `effective_metric_target_distance` dispatch
//! ([`crate::vardct::perceptual_loop`]) hard-coded the butteraugli arm
//! as `target_distance` (the identity), even when the caller passed an
//! explicit per-distance target override.
//!
//! Phase 1 ships this calibration table + dispatch wiring so the
//! butteraugli arm honours the override. Direction is the INVERSE of
//! [`crate::vardct::cvvdp_targets`] (cvvdp's table goes
//! `distance → score`; butteraugli's goes `score → distance` — the
//! caller passes a butter-direction target score and the lookup
//! returns the encoder distance that historically converged at that
//! score median).
//!
//! ## Seed methodology
//!
//! Source: `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
//! (1,134 cells across CID22 + GB82-SC + W44-PHASE4-S1 extras). For
//! each nominal `distance` band the per-image
//! `score_butter_cpu` was aggregated over the 162 `backend=B` rows
//! (effort 5, single butteraugli backend); the median is the table's
//! score column.
//!
//! Per-distance medians (re-verified at commit `a3e937d3`):
//!
//! | nominal distance | n   | median butter | p25    | p75    |
//! |---               |---  |---            |---     |---     |
//! | 0.5              | 162 | 0.7223        | 0.6449 | 0.7878 |
//! | 1.0              | 162 | 1.2245        | 1.1519 | 1.2858 |
//! | 1.5              | 162 | 1.7285        | 1.6135 | 1.8230 |
//! | 2.0              | 162 | 2.1936        | 1.9614 | 2.3471 |
//! | 3.0              | 162 | 2.9616        | 2.6421 | 3.2037 |
//! | 4.0              | 162 | 3.7608        | 3.3719 | 3.9843 |
//! | 5.0              | 162 | 4.4004        | 3.9147 | 4.6956 |
//!
//! Per-image variance is ±30-50% from the median (p25..p75 range at
//! d=2.0 is 1.96..2.35); the table is a corpus-median calibration, not
//! a per-image exact-target oracle. Callers needing exact-score
//! convergence on a specific image should use the (future) Phase 4
//! `with_target_score_iterate(true)` outer binary-search wrapper. See
//! [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](../../docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md)
//! §4.2 for the cost/benefit analysis.
//!
//! ## Monotonicity invariants
//!
//! - Score column is strictly ascending (`0.7223 < 1.2245 < ...`).
//! - Distance column is strictly ascending (`0.5 < 1.0 < ...`).
//! - Linear interpolation is therefore well-defined; the lookup
//!   produces an `effective_distance` that grows monotonically with
//!   the caller's requested `target_score`.
//!
//! Both invariants are pinned by unit tests below.
//!
//! ## Linear interpolation
//!
//! [`butteraugli_effective_distance_for_target_score`] interpolates
//! linearly between adjacent table points and clamps outside the band
//! (`[0.7223, 4.4004]`). The table is monotone (distance grows with
//! score) so interpolation is well-defined.
//!
//! ## Default-path byte-identity
//!
//! Phase 1 default is `LossyConfig::perceptual_target_score = None`,
//! which preserves the pre-Phase-1 identity arm in
//! [`crate::vardct::perceptual_loop::run_buttloop`]
//! (`effective_metric_target_distance = target_distance`). Hash-locks
//! 36/36 stay BYTE-IDENTICAL. The Phase 1 lookup ONLY fires when the
//! caller opts in via
//! [`crate::api::LossyConfig::with_perceptual_target_score(Some(_))`].
//!
//! ## EncoderStrategy::Libjxl strict-parity short-circuit
//!
//! The
//! [`crate::api::LossyConfig::resolve_perceptual_target_score`]
//! resolver forces `None` when the active strategy is
//! [`crate::api::EncoderStrategy::Libjxl`] — preserves the W44-126
//! byte-lock invariant (`with_perceptual_target_score(Some(_))` on a
//! Libjxl-strategy config is silently dropped). The dispatch wiring at
//! [`crate::vardct::perceptual_loop`] consumes the resolved value, so
//! the short-circuit fires before the table is consulted.

/// Number of distance bands in the Phase 1 table. Pinned at 7 to match
/// the seed sweep `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
/// (`d ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}`); changing this requires
/// a re-seed + RFC update.
pub(crate) const BUTTERAUGLI_DISTANCE_BANDS: usize = 7;

/// Phase 1 calibration: `(achieved_butter_median, nominal_distance)`
/// pairs. INVERSE direction to [`crate::vardct::cvvdp_targets`].
///
/// - `achieved_butter_median` is the corpus-median butter-direction
///   score the buttloop converged to at the nominal distance (from the
///   seed sweep, 162 cells per distance band). Caller passes this as
///   the per-distance override.
/// - `nominal_distance` is the encoder-internal distance value the
///   buttloop drives the loop at. Returned by the lookup for the
///   buttloop's `effective_metric_target_distance` dispatch.
///
/// Table is sorted ascending in BOTH columns (monotonic), so
/// `linear interp + clamp` is well-defined.
///
/// Format: `(butter_score_target, effective_distance)`.
pub(crate) static BUTTERAUGLI_SCORE_TARGETS: &[(f32, f32); BUTTERAUGLI_DISTANCE_BANDS] = &[
    // (achieved butter median, nominal distance) — caller passes the
    // achieved target as `with_perceptual_target_score(Some(x))`; the
    // table returns the distance that drives the loop to converge
    // there (in the corpus-median sense).
    //
    // Source: `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
    // `backend=B` rows, per-distance median of `score_butter_cpu`
    // (n=162 per band, re-verified at commit `a3e937d3`). See module
    // doc §"Seed methodology" for the full per-band p25/p75 envelope.
    (0.7223, 0.5),
    (1.2245, 1.0),
    (1.7285, 1.5),
    (2.1936, 2.0),
    (2.9616, 3.0),
    (3.7608, 4.0),
    (4.4004, 5.0),
];

/// Phase 1 inverse-direction lookup: given a caller-supplied
/// butter-direction `target_score`, return the encoder-internal
/// `effective_distance` the buttloop should drive the loop at.
///
/// Linear interpolation between adjacent table points; clamps to the
/// boundary distance outside `[BUTTERAUGLI_SCORE_TARGETS[0].0,
/// BUTTERAUGLI_SCORE_TARGETS[last].0]`.
///
/// Returns the lookup value in `target_distance` (butteraugli native
/// distance, smaller = better) — directly consumable by the buttloop's
/// `effective_metric_target_distance` dispatch in
/// [`crate::vardct::perceptual_loop::run_buttloop`].
///
/// Non-finite / non-positive inputs fall back to the band-low distance
/// to guard the buttloop against caller-side NaN/Inf accidents. See
/// the per-row tests `lookup_handles_nan` + `lookup_handles_negative`
/// below.
///
/// # Examples
///
/// ```ignore
/// // Exact table point.
/// let d = butteraugli_effective_distance_for_target_score(1.2245);
/// assert!((d - 1.0).abs() < 1e-6);
///
/// // Interpolation: midway between (0.7223, 0.5) and (1.2245, 1.0).
/// let d = butteraugli_effective_distance_for_target_score(
///     (0.7223 + 1.2245) * 0.5,
/// );
/// assert!((d - 0.75).abs() < 1e-3);
///
/// // Below-band clamp.
/// let d = butteraugli_effective_distance_for_target_score(0.1);
/// assert!((d - 0.5).abs() < 1e-6);
///
/// // Above-band clamp.
/// let d = butteraugli_effective_distance_for_target_score(10.0);
/// assert!((d - 5.0).abs() < 1e-6);
/// ```
pub(crate) fn butteraugli_effective_distance_for_target_score(target_score: f32) -> f32 {
    let table = BUTTERAUGLI_SCORE_TARGETS;
    // Guard against caller-side NaN/Inf/negative accidents (per RFC
    // `RFC_BUTTERAUGLI_FORK_PLAN.md` §7 risk register row). Falls back
    // to the band-low distance — same shape as the below-band clamp.
    if !target_score.is_finite() || target_score <= 0.0 {
        return table[0].1;
    }
    // Below-band clamp.
    if target_score <= table[0].0 {
        return table[0].1;
    }
    // Above-band clamp.
    let last = table.len() - 1;
    if target_score >= table[last].0 {
        return table[last].1;
    }
    // Linear search — the table has 7 entries; binary search overhead
    // exceeds the per-call savings. Find the bracketing pair
    // (score_lo, score_hi) and interpolate the distance.
    for i in 0..last {
        let (s_lo, d_lo) = table[i];
        let (s_hi, d_hi) = table[i + 1];
        if target_score >= s_lo && target_score <= s_hi {
            let span = s_hi - s_lo;
            if span <= 0.0 {
                return d_lo;
            }
            let t = (target_score - s_lo) / span;
            return d_lo + t * (d_hi - d_lo);
        }
    }
    // Unreachable given the clamps above — defensive fallback.
    table[last].1
}

/// Identity-arm shim preserved for the implicit-identity path. When
/// the caller does NOT set `perceptual_target_score`, the buttloop
/// continues to use `target_distance` directly — this fn captures
/// that identity behaviour as a pure const fn for symmetry with the
/// cvvdp / zensim lookup signatures.
///
/// Returns `d.max(0.0)` to guard against caller-side negatives.
#[allow(dead_code)]
pub(crate) fn butteraugli_target_score_for_distance(d: f32) -> f32 {
    if !d.is_finite() {
        return 0.0;
    }
    d.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Score column is strictly ascending (required for linear-search
    /// bracketing on the score axis).
    #[test]
    fn table_score_column_sorted_ascending() {
        let table = BUTTERAUGLI_SCORE_TARGETS;
        for w in table.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "table score column must be strictly ascending: {} >= {}",
                w[0].0,
                w[1].0,
            );
        }
    }

    /// Distance column is strictly ascending (the buttloop expects
    /// monotone behaviour: smaller target_score → smaller distance →
    /// tighter convergence).
    #[test]
    fn table_distance_column_sorted_ascending() {
        let table = BUTTERAUGLI_SCORE_TARGETS;
        for w in table.windows(2) {
            assert!(
                w[0].1 < w[1].1,
                "table distance column must be strictly ascending: {} >= {}",
                w[0].1,
                w[1].1,
            );
        }
    }

    /// Endpoints return the table's distance values exactly.
    #[test]
    fn lookup_endpoints_exact() {
        let table = BUTTERAUGLI_SCORE_TARGETS;
        let first = table[0];
        let last = table[table.len() - 1];
        let d_first = butteraugli_effective_distance_for_target_score(first.0);
        let d_last = butteraugli_effective_distance_for_target_score(last.0);
        assert!(
            (d_first - first.1).abs() < 1e-6,
            "lookup at first score {} must return distance {}, got {}",
            first.0,
            first.1,
            d_first,
        );
        assert!(
            (d_last - last.1).abs() < 1e-6,
            "lookup at last score {} must return distance {}, got {}",
            last.0,
            last.1,
            d_last,
        );
    }

    /// Below-band clamp returns the band-low distance; above-band clamp
    /// returns the band-high distance.
    #[test]
    fn lookup_outside_band_clamps() {
        let table = BUTTERAUGLI_SCORE_TARGETS;
        let first_d = table[0].1;
        let last_d = table[table.len() - 1].1;
        assert!((butteraugli_effective_distance_for_target_score(0.0) - first_d).abs() < 1e-6,);
        assert!((butteraugli_effective_distance_for_target_score(0.1) - first_d).abs() < 1e-6,);
        assert!((butteraugli_effective_distance_for_target_score(10.0) - last_d).abs() < 1e-6,);
        assert!((butteraugli_effective_distance_for_target_score(100.0) - last_d).abs() < 1e-6,);
    }

    /// Linear interpolation at the midpoint of (0.7223, 0.5) and
    /// (1.2245, 1.0) should be ≈ (0.97340, 0.75).
    #[test]
    fn lookup_interpolates_linearly() {
        let table = BUTTERAUGLI_SCORE_TARGETS;
        let (s_a, d_a) = table[0];
        let (s_b, d_b) = table[1];
        let mid_score = (s_a + s_b) * 0.5;
        let expected_d = (d_a + d_b) * 0.5;
        let got_d = butteraugli_effective_distance_for_target_score(mid_score);
        assert!(
            (got_d - expected_d).abs() < 1e-4,
            "interp at mid_score={} expected d={} got d={}",
            mid_score,
            expected_d,
            got_d,
        );

        // Quarter-point between (2.1936, 2.0) and (2.9616, 3.0).
        let (s_a, d_a) = (2.1936_f32, 2.0_f32);
        let (s_b, d_b) = (2.9616_f32, 3.0_f32);
        let target = s_a + 0.25 * (s_b - s_a);
        let expected = d_a + 0.25 * (d_b - d_a);
        let got = butteraugli_effective_distance_for_target_score(target);
        assert!(
            (got - expected).abs() < 1e-4,
            "quarter-point interp expected d={} got d={}",
            expected,
            got,
        );
    }

    /// Lookup is finite, non-negative, and bounded for any reasonable
    /// input. Guards against NaN/Inf accidents in the interp math.
    #[test]
    fn lookup_well_behaved_across_band() {
        for s_x100 in 0..=600 {
            let s = (s_x100 as f32) * 0.01;
            let d = butteraugli_effective_distance_for_target_score(s);
            assert!(
                d.is_finite() && d >= 0.0,
                "lookup at score={} returned {} (finite={}, >=0={})",
                s,
                d,
                d.is_finite(),
                d >= 0.0,
            );
            // Above-band clamp = 5.0. Allow a defensive ceiling of 10.0.
            assert!(
                d <= 10.0,
                "lookup at score={} returned {} > 10.0 (table max distance 5.0)",
                s,
                d,
            );
        }
    }

    /// NaN inputs must NOT propagate into the buttloop. Falls back to
    /// the band-low distance (same shape as below-band clamp).
    #[test]
    fn lookup_handles_nan() {
        let d = butteraugli_effective_distance_for_target_score(f32::NAN);
        let expected = BUTTERAUGLI_SCORE_TARGETS[0].1;
        assert!(
            (d - expected).abs() < 1e-6,
            "NaN input must fall back to band-low distance {}, got {}",
            expected,
            d,
        );
        let d = butteraugli_effective_distance_for_target_score(f32::INFINITY);
        // +Inf is treated like a very large positive — above-band clamp.
        // The above-band clamp happens before the NaN/finite check?
        // We test the actual behaviour: the fn guards `!is_finite()` first,
        // so +Inf returns band-low. Document the chosen contract.
        assert!(
            (d - expected).abs() < 1e-6,
            "+Inf input must fall back to band-low distance {}, got {}",
            expected,
            d,
        );
    }

    /// Negative inputs must NOT propagate into the buttloop. Falls
    /// back to the band-low distance.
    #[test]
    fn lookup_handles_negative() {
        let d = butteraugli_effective_distance_for_target_score(-1.0);
        let expected = BUTTERAUGLI_SCORE_TARGETS[0].1;
        assert!(
            (d - expected).abs() < 1e-6,
            "negative input must fall back to band-low distance {}, got {}",
            expected,
            d,
        );
        let d = butteraugli_effective_distance_for_target_score(-1.0e9);
        assert!(
            (d - expected).abs() < 1e-6,
            "large-negative input must fall back to band-low distance {}, got {}",
            expected,
            d,
        );
    }

    /// Table has exactly the 7 entries the seed methodology calls out.
    /// Guards against accidental edits that would shift the
    /// methodology without an explicit refit chunk.
    #[test]
    fn table_has_seven_entries() {
        assert_eq!(
            BUTTERAUGLI_SCORE_TARGETS.len(),
            BUTTERAUGLI_DISTANCE_BANDS,
            "table should have {} entries (d ∈ {{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}}) — \
             changing this requires a refit chunk + RFC update",
            BUTTERAUGLI_DISTANCE_BANDS,
        );
    }

    /// Identity shim returns the input unchanged for valid inputs.
    /// Negative/NaN inputs are clamped to 0.
    #[test]
    fn identity_shim_returns_input() {
        for d_x10 in 0..=60 {
            let d = (d_x10 as f32) * 0.1;
            let got = butteraugli_target_score_for_distance(d);
            assert!(
                (got - d).abs() < 1e-6,
                "identity shim at d={}: got {} expected {}",
                d,
                got,
                d,
            );
        }
        // Negatives and NaN clamp to 0.
        assert_eq!(butteraugli_target_score_for_distance(-1.0), 0.0);
        assert_eq!(butteraugli_target_score_for_distance(f32::NAN), 0.0);
    }

    /// Cross-test the inverse property: the table direction is
    /// `score → distance`, so feeding the corpus-median score at each
    /// canonical distance band MUST return the matching distance to
    /// within 1e-6 (the table entries themselves).
    #[test]
    fn inverse_at_canonical_bands() {
        // Pairs from the Phase 1 seed methodology (n=162 per band).
        let canonical: [(f32, f32); 7] = [
            (0.7223, 0.5),
            (1.2245, 1.0),
            (1.7285, 1.5),
            (2.1936, 2.0),
            (2.9616, 3.0),
            (3.7608, 4.0),
            (4.4004, 5.0),
        ];
        for (score, expected_distance) in canonical {
            let got = butteraugli_effective_distance_for_target_score(score);
            assert!(
                (got - expected_distance).abs() < 1e-6,
                "canonical band: score={} expected d={} got {}",
                score,
                expected_distance,
                got,
            );
        }
    }

    /// Phase 1 honeypot: the corpus-median seed values must stay within
    /// ±10% of the documented per-band medians. Catches accidental edits
    /// that would shift the methodology without an explicit re-measure.
    #[test]
    fn phase1_table_in_seed_envelope() {
        let documented_medians: [f32; 7] = [0.7223, 1.2245, 1.7285, 2.1936, 2.9616, 3.7608, 4.4004];
        for (i, (got_pair, base)) in BUTTERAUGLI_SCORE_TARGETS
            .iter()
            .zip(documented_medians.iter())
            .enumerate()
        {
            let got = got_pair.0;
            let rel = (got - base).abs() / base;
            assert!(
                rel < 0.10,
                "BUTTERAUGLI_SCORE_TARGETS[{i}].0 = {got} drifted >10% from documented median {base}; \
                 re-verify against benchmarks/cvvdp_vs_buttloop_tracking_*.tsv before shipping",
            );
        }
    }
}
