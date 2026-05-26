// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-distance × per-display CVVDP JOD calibration table
//! (cvvdp-fork Phase 4 + Phase 1 display-config backfill).
//!
//! Phase 4 (2026-05-24) shipped the single-display SDR-200 table
//! (`vardct/cvvdp_targets.rs` initial commit — see
//! `docs/RFC_CVVDP_FORK.md` §2.1, §4 Phase 4 and
//! `docs/RFC_CVVDP_PHASE4_BRIEF.md` Step 3).
//!
//! Phase 1 display-config backfill (2026-05-25, RFC
//! `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`) extended the lookup to a
//! 3-row × 7-distance table covering [`crate::api::DisplayConfig`]'s
//! three Phase 1 variants:
//!
//! - [`crate::api::DisplayConfig::WebSdr80`] — preserved bit-identical
//!   from Phase 4 (the legacy single-table seed). Maps to
//!   `cvvdp_gpu::params::DisplayModel::STANDARD_4K` (200 cd/m² sRGB
//!   BT.709 at 250 lux ambient).
//! - [`crate::api::DisplayConfig::Phone`] — 1000 cd/m² sustained-EDR
//!   sRGB Display-P3 at 200 lux indoor bright. The custom display model
//!   is built inline by [`crate::api::DisplayConfig::display_model`] —
//!   `IPHONE_14_PRO_HDR` is HLG/BT.2020, wrong shape for SDR-on-HDR.
//! - [`crate::api::DisplayConfig::Tv`] — `LG_OLED_2026_HDR_PQ` (3000
//!   cd/m² PQ BT.2020 at 5 lux dim).
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
//! from this table at the caller-requested butteraugli `target_distance`
//! AND at the caller-requested [`crate::api::DisplayConfig`].
//!
//! [GPU CVVDP backend]: super::cvvdp_backend::gpu::GpuCvvdpBackend
//!
//! ## Seed methodology — WebSdr80 row
//!
//! Source: `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
//! (Agent D's baseline, 1,131 backend=B / butteraugli-default rows
//! covering CID22 + GB82-SC + CLIC corpora across `distance ∈ {0.5,
//! 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}`). The pre-computed seed values are
//! checked-in at `benchmarks/cvvdp_jod_calibration_seed_2026-05-24.txt`;
//! the WebSdr80 row below was reviewed and pasted from there.
//!
//! For each distance band, the median `score_cvvdp_gpu` (JOD ∈ [0, 10])
//! was computed across the corpus rows at that distance, then converted
//! to butteraugli-direction `score = (10.0 - jod) * 1.05` — the 1.05×
//! tightening biases the cvvdp-driven loop to converge slightly more
//! strictly than what the butteraugli-driven baseline achieves at the
//! same nominal distance.
//!
//! ## Seed methodology — Phone + Tv rows (Phase 1, 2026-05-25)
//!
//! Phase 1 ships heuristic-based seeds derived from the cvvdp algorithm
//! sensitivity estimates in `docs/RFC_DISPLAY_CONFIG_BACKFILL.md` §4:
//!
//! - **Phone** (1000 cd/m² sRGB Display-P3 at 200 lux): JOD shift
//!   estimated at -0.04 to -0.08 vs WebSdr80 baseline (between Mantiuk
//!   2024 §4.2's SDR-200→Phone-500 ~0.05-0.12 JOD shift and
//!   SDR-200→HDR-1000 ~0.15-0.40 JOD shift; our Phone uses HDR-peak
//!   luminance but SDR EOTF so the structural perception shift sits in
//!   the middle of those ranges). Translated to the `(10-jod)*1.05`
//!   axis: target is ~4% larger than WebSdr80 (tighter convergence).
//! - **Tv** (3000 cd/m² PQ BT.2020 at 5 lux dim): JOD shift estimated
//!   at -0.12 to -0.25 vs WebSdr80 baseline (interpolated between
//!   the §4 SDR-200→HDR-PQ-1000 ~0.15-0.40 JOD shift and HDR-PQ-4000
//!   ~0.25-0.60 JOD shift, dampened for the dim ambient). Translated
//!   to the `(10-jod)*1.05` axis: target is ~12% larger than WebSdr80.
//!
//! The actual seed values are computed as
//! `phone_target[d] = web_target[d] × 1.04` and
//! `tv_target[d]    = web_target[d] × 1.12`. These are the production
//! constants until a follow-on chunk re-seeds them from local
//! `cvvdp_track_baseline.rs` re-scoring against the WebSdr80
//! `cvvdp_vs_buttloop_tracking_2026-05-24.tsv` corpus under each
//! `DisplayConfig`'s `DisplayModel`.
//!
//! **Why heuristic-seeded for Phase 1**: the upstream `CvvdpOpaque::new`
//! API does NOT expose `DisplayGeometry` — only `DisplayModel` is
//! plumbed through `CvvdpParams.display`. Until a future cvvdp-gpu PR
//! adds `new_with_geometry`, the GPU cvvdp backend's PPD stays at
//! `STANDARD_4K.pixels_per_degree() = 75.4` regardless of which
//! `DisplayConfig` is selected. The DisplayModel axis (peak luminance,
//! EOTF, primaries, ambient) DOES dispatch correctly. A future Phase 2
//! re-seed will measure the actual shifts on the corpus once geometry
//! is also plumbed.
//!
//! ## Monotonicity invariants
//!
//! - Within each row (display): targets grow with distance (stricter
//!   distance ⇒ larger acceptable cvvdp score).
//! - At each distance: `Tv >= Phone >= WebSdr80` (more luminance ⇒
//!   stricter convergence requirement).
//!
//! Both invariants are pinned by unit tests in `tests` module below.
//!
//! ## Linear interpolation
//!
//! [`cvvdp_target_score_for_distance_and_display`] interpolates
//! linearly between adjacent table points (within a row) and clamps
//! outside the band (`[0.50, 5.00]`). The table is monotone (target
//! grows with distance) so interpolation is well-defined per row.

use crate::api::DisplayConfig;

/// Number of distance bands in the per-display table. Pinned at 7 to
/// match the Phase 4 seed (d ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0});
/// changing this requires a re-seed and per-display refit.
pub(crate) const CVVDP_DISTANCE_BANDS: usize = 7;

/// Per-display calibration row: distance / target pairs at the 7
/// canonical bands. Within-row monotone ascending in distance AND in
/// target score.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DisplayCalibration {
    /// Display this row was calibrated for.
    pub(crate) display: DisplayConfig,
    /// 7 `(distance, target_score)` pairs.
    pub(crate) entries: [(f32, f32); CVVDP_DISTANCE_BANDS],
}

/// WebSdr80 row — preserved bit-identical from the Phase 4 seed.
/// Source: `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
/// per-distance median converted via `(10.0 - jod) * 1.05`.
pub(crate) const WEB_SDR_80_CALIBRATION: DisplayCalibration = DisplayCalibration {
    display: DisplayConfig::WebSdr80,
    entries: [
        (0.50, 0.0029), // n=162, median JOD = 9.9972
        (1.00, 0.0238), // n=162, median JOD = 9.9774
        (1.50, 0.0461), // n=162, median JOD = 9.9561
        (2.00, 0.0724), // n=162, median JOD = 9.9311
        (3.00, 0.1336), // n=162, median JOD = 9.8728
        (4.00, 0.2149), // n=159, median JOD = 9.7953
        (5.00, 0.3005), // n=162, median JOD = 9.7138
    ],
};

/// Phone row — 1000 cd/m² sRGB Display-P3 at 200 lux. Heuristic-seeded
/// at `WebSdr80 × PHONE_TARGET_MULTIPLIER` per module doc §"Seed
/// methodology — Phone + Tv rows". Future re-seed will measure the
/// actual per-distance shifts on the corpus.
pub(crate) const PHONE_TARGET_MULTIPLIER: f32 = 1.04;

/// Tv row — 3000 cd/m² PQ BT.2020 at 5 lux dim
/// (`cvvdp_gpu::params::DisplayModel::LG_OLED_2026_HDR_PQ`). Heuristic-
/// seeded at `WebSdr80 × TV_TARGET_MULTIPLIER` per module doc.
pub(crate) const TV_TARGET_MULTIPLIER: f32 = 1.12;

/// Phone calibration row, computed at compile time as
/// `WEB_SDR_80_CALIBRATION × PHONE_TARGET_MULTIPLIER`.
pub(crate) const PHONE_CALIBRATION: DisplayCalibration = DisplayCalibration {
    display: DisplayConfig::Phone,
    entries: scale_entries(WEB_SDR_80_CALIBRATION.entries, PHONE_TARGET_MULTIPLIER),
};

/// Tv calibration row, computed at compile time as
/// `WEB_SDR_80_CALIBRATION × TV_TARGET_MULTIPLIER`.
pub(crate) const TV_CALIBRATION: DisplayCalibration = DisplayCalibration {
    display: DisplayConfig::Tv,
    entries: scale_entries(WEB_SDR_80_CALIBRATION.entries, TV_TARGET_MULTIPLIER),
};

/// Const helper: produce a new entries array by multiplying every
/// target score by `factor`. Distances are preserved exactly so
/// interpolation lookups remain bracket-aligned across rows.
const fn scale_entries(
    src: [(f32, f32); CVVDP_DISTANCE_BANDS],
    factor: f32,
) -> [(f32, f32); CVVDP_DISTANCE_BANDS] {
    let mut out = src;
    let mut i = 0;
    while i < CVVDP_DISTANCE_BANDS {
        out[i].1 = src[i].1 * factor;
        i += 1;
    }
    out
}

/// Lookup table indexed by display variant — same shape as the
/// upstream cvvdp-gpu preset surface. Adding a new
/// [`DisplayConfig`] variant requires adding a calibration row here.
pub(crate) static CVVDP_DISPLAY_CALIBRATIONS: &[DisplayCalibration] =
    &[WEB_SDR_80_CALIBRATION, PHONE_CALIBRATION, TV_CALIBRATION];

/// Legacy single-display per-distance table — same data as
/// [`WEB_SDR_80_CALIBRATION.entries`], kept as a const for use sites
/// that don't yet thread `DisplayConfig`. Phase 1 callers should prefer
/// [`cvvdp_target_score_for_distance_and_display`].
///
/// Format: `(butteraugli_target_distance, cvvdp_target_score)`.
/// Table is sorted ascending by distance; lookup is linear interp +
/// clamp.
#[allow(dead_code)]
pub(crate) static CVVDP_DISTANCE_TARGETS: &[(f32, f32)] = &WEB_SDR_80_CALIBRATION.entries;

/// Look up the CVVDP `score = 10 - JOD` target for a given butteraugli
/// `target_distance` AND target [`DisplayConfig`]. Linear interpolation
/// between adjacent table points within the row; clamps to the boundary
/// value outside `[CVVDP_DISTANCE_TARGETS[0].0,
/// CVVDP_DISTANCE_TARGETS[last].0]`.
///
/// Returns the lookup value in score-direction (smaller = better),
/// directly comparable to the `BackendCompareResult::score` produced by
/// the cvvdp backends.
///
/// # Examples
///
/// ```ignore
/// use jxl_encoder::api::DisplayConfig;
/// // WebSdr80 d=1.0 returns the legacy 0.0238 (byte-identical to the
/// // pre-Phase-1 single-table lookup).
/// let s = cvvdp_target_score_for_distance_and_display(1.0, DisplayConfig::WebSdr80);
/// assert!((s - 0.0238).abs() < 1e-6);
///
/// // Phone target is strictly larger (tighter convergence).
/// let s_phone = cvvdp_target_score_for_distance_and_display(1.0, DisplayConfig::Phone);
/// assert!(s_phone > 0.0238);
/// // Tv target is strictly larger than Phone.
/// let s_tv = cvvdp_target_score_for_distance_and_display(1.0, DisplayConfig::Tv);
/// assert!(s_tv > s_phone);
/// ```
pub(crate) fn cvvdp_target_score_for_distance_and_display(
    target_distance: f32,
    display: DisplayConfig,
) -> f32 {
    let row = display_calibration(display);
    let table = &row.entries;
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

/// Look up the CVVDP `score = 10 - JOD` target for a given butteraugli
/// `target_distance`, assuming [`DisplayConfig::WebSdr80`]. Thin wrapper
/// over [`cvvdp_target_score_for_distance_and_display`] preserved for
/// call sites that don't yet thread display config.
///
/// Byte-identical to the pre-Phase-1 lookup (the `WebSdr80` row IS the
/// Phase 4 seed table). All production callers have migrated to the
/// `_and_display` variant; this wrapper is retained as the API-stable
/// shim + the regression-test target for the `WebSdr80 == legacy`
/// byte-identity invariant.
#[allow(dead_code)]
pub(crate) fn cvvdp_target_score_for_distance(target_distance: f32) -> f32 {
    cvvdp_target_score_for_distance_and_display(target_distance, DisplayConfig::WebSdr80)
}

/// Find the per-display calibration row in
/// [`CVVDP_DISPLAY_CALIBRATIONS`]. Falls back to the WebSdr80 row if
/// the table is missing the variant — defensive: every variant in
/// [`DisplayConfig`] MUST have a calibration row (enforced by
/// `calibration_for_every_display_variant` test).
fn display_calibration(display: DisplayConfig) -> &'static DisplayCalibration {
    for row in CVVDP_DISPLAY_CALIBRATIONS {
        if row.display == display {
            return row;
        }
    }
    &WEB_SDR_80_CALIBRATION
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row's distances are sorted ascending (required for linear-
    /// search bracketing).
    #[test]
    fn rows_sorted_ascending_by_distance() {
        for row in CVVDP_DISPLAY_CALIBRATIONS {
            for w in row.entries.windows(2) {
                assert!(
                    w[0].0 < w[1].0,
                    "{:?} row distances must be strictly ascending: {} >= {}",
                    row.display,
                    w[0].0,
                    w[1].0
                );
            }
        }
    }

    /// Each row is monotone ascending in target score: stricter
    /// distance implies a larger acceptable cvvdp score.
    #[test]
    fn cvvdp_target_score_per_display_is_monotone() {
        for row in CVVDP_DISPLAY_CALIBRATIONS {
            for w in row.entries.windows(2) {
                assert!(
                    w[0].1 <= w[1].1,
                    "{:?} target scores must be non-decreasing: {} > {}",
                    row.display,
                    w[0].1,
                    w[1].1
                );
            }
        }
    }

    /// At every distance, the Phone target is strictly larger than the
    /// WebSdr80 target (tighter convergence — Phone is more luminance-
    /// amplified per RFC §4 sensitivity estimate).
    #[test]
    fn cvvdp_target_score_phone_higher_than_websdr80() {
        for d_x100 in [50, 100, 150, 200, 300, 400, 500] {
            let d = (d_x100 as f32) * 0.01;
            let s_web = cvvdp_target_score_for_distance_and_display(d, DisplayConfig::WebSdr80);
            let s_phone = cvvdp_target_score_for_distance_and_display(d, DisplayConfig::Phone);
            assert!(
                s_phone > s_web,
                "at d={}, Phone target ({}) MUST be > WebSdr80 target ({}) — \
                 RFC §4 sensitivity estimate (Phone is brighter / more luminance-amplified)",
                d,
                s_phone,
                s_web
            );
        }
    }

    /// At every distance, the Tv target is strictly larger than the
    /// Phone target (Tv has higher peak luminance AND PQ EOTF AND lower
    /// ambient, all of which amplify perceptible artifacts).
    #[test]
    fn cvvdp_target_score_tv_higher_than_phone() {
        for d_x100 in [50, 100, 150, 200, 300, 400, 500] {
            let d = (d_x100 as f32) * 0.01;
            let s_phone = cvvdp_target_score_for_distance_and_display(d, DisplayConfig::Phone);
            let s_tv = cvvdp_target_score_for_distance_and_display(d, DisplayConfig::Tv);
            assert!(
                s_tv > s_phone,
                "at d={}, Tv target ({}) MUST be > Phone target ({}) — \
                 Tv has higher peak luminance, PQ EOTF, lower ambient (RFC §4)",
                d,
                s_tv,
                s_phone
            );
        }
    }

    /// WebSdr80 lookup MUST be byte-identical to the legacy
    /// `cvvdp_target_score_for_distance(d)` wrapper (which routes
    /// through `display_calibration(WebSdr80)`) on every sweep point.
    #[test]
    fn websdr80_byte_identical_to_legacy_wrapper() {
        for d_x100 in 0..=600 {
            let d = (d_x100 as f32) * 0.01;
            let s_legacy = cvvdp_target_score_for_distance(d);
            let s_explicit =
                cvvdp_target_score_for_distance_and_display(d, DisplayConfig::WebSdr80);
            // bit-equal — wrapper just passes through.
            assert_eq!(
                s_legacy.to_bits(),
                s_explicit.to_bits(),
                "at d={}: legacy wrapper ({}) MUST equal explicit WebSdr80 ({}) bit-for-bit",
                d,
                s_legacy,
                s_explicit
            );
        }
    }

    /// Every variant of [`DisplayConfig`] MUST have a calibration row.
    /// Guards against future enum extensions silently falling back to
    /// the WebSdr80 row.
    #[test]
    fn calibration_for_every_display_variant() {
        for display in [
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            let found = CVVDP_DISPLAY_CALIBRATIONS
                .iter()
                .any(|r| r.display == display);
            assert!(
                found,
                "DisplayConfig::{:?} has no calibration row in \
                 CVVDP_DISPLAY_CALIBRATIONS — add one before shipping",
                display
            );
        }
    }

    /// Below-band clamp to the first entry of each row; above-band
    /// clamp to the last.
    #[test]
    fn lookup_outside_band_clamps_per_display() {
        for &display in &[
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            let row = display_calibration(display);
            let first_v = row.entries[0].1;
            let last_v = row.entries[row.entries.len() - 1].1;
            assert!(
                (cvvdp_target_score_for_distance_and_display(0.0, display) - first_v).abs() < 1e-6
            );
            assert!(
                (cvvdp_target_score_for_distance_and_display(0.25, display) - first_v).abs() < 1e-6
            );
            assert!(
                (cvvdp_target_score_for_distance_and_display(10.0, display) - last_v).abs() < 1e-6
            );
            assert!(
                (cvvdp_target_score_for_distance_and_display(100.0, display) - last_v).abs() < 1e-6
            );
        }
    }

    /// Linear interpolation within each row.
    #[test]
    fn lookup_interpolates_linearly_per_display() {
        // Midpoint of (1.0, _) and (1.5, _) is at d=1.25.
        for &display in &[
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            let row = display_calibration(display);
            let (s_1_0, s_1_5) = (row.entries[1].1, row.entries[2].1);
            let expected = (s_1_0 + s_1_5) * 0.5;
            let got = cvvdp_target_score_for_distance_and_display(1.25, display);
            assert!(
                (got - expected).abs() < 1e-4,
                "{:?} interp at d=1.25: got {} expected {}",
                display,
                got,
                expected
            );
        }
    }

    /// Endpoints return the table values exactly per row.
    #[test]
    fn lookup_endpoints_exact_per_display() {
        for &display in &[
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            let row = display_calibration(display);
            let first = row.entries[0];
            let last = row.entries[row.entries.len() - 1];
            let s_first = cvvdp_target_score_for_distance_and_display(first.0, display);
            let s_last = cvvdp_target_score_for_distance_and_display(last.0, display);
            assert!(
                (s_first - first.1).abs() < 1e-6,
                "{:?} lookup at first entry must return its value",
                display
            );
            assert!(
                (s_last - last.1).abs() < 1e-6,
                "{:?} lookup at last entry must return its value",
                display
            );
        }
    }

    /// Lookup is finite, non-negative, and bounded across the band for
    /// every display variant.
    #[test]
    fn lookup_well_behaved_across_band_per_display() {
        for &display in &[
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            for d_x10 in 0..=60 {
                let d = (d_x10 as f32) * 0.1;
                let s = cvvdp_target_score_for_distance_and_display(d, display);
                assert!(
                    s.is_finite() && s >= 0.0,
                    "{:?} lookup at d={} returned {} (finite={}, >=0={})",
                    display,
                    d,
                    s,
                    s.is_finite(),
                    s >= 0.0
                );
                // Tv at d=5.0 is the strictest (largest); 1.0 is a safe
                // upper bound (Tv multiplier 1.12 × WebSdr80 max 0.3005 ≈ 0.337).
                assert!(
                    s <= 1.0,
                    "{:?} lookup at d={} returned {} > 1.0",
                    display,
                    d,
                    s
                );
            }
        }
    }

    /// Each calibration row has exactly the 7 canonical bands.
    #[test]
    fn each_row_has_seven_entries() {
        for row in CVVDP_DISPLAY_CALIBRATIONS {
            assert_eq!(
                row.entries.len(),
                7,
                "{:?} row should have 7 entries (d ∈ {{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}})",
                row.display
            );
        }
    }

    /// Phase 1 honeypot: ensure the multiplier constants stay in the
    /// expected band (Phone in [1.02, 1.10], Tv in [1.06, 1.20]). If a
    /// future refit moves them outside these bounds the per-display
    /// monotonicity tests above may still pass but the seed methodology
    /// docstring would be stale — fail fast so the doc gets updated.
    #[test]
    fn multipliers_in_documented_band() {
        assert!(
            (1.02..=1.10).contains(&PHONE_TARGET_MULTIPLIER),
            "PHONE_TARGET_MULTIPLIER ({}) outside documented band [1.02, 1.10]; \
             update module docstring before re-tightening",
            PHONE_TARGET_MULTIPLIER
        );
        assert!(
            (1.06..=1.20).contains(&TV_TARGET_MULTIPLIER),
            "TV_TARGET_MULTIPLIER ({}) outside documented band [1.06, 1.20]; \
             update module docstring before re-tightening",
            TV_TARGET_MULTIPLIER
        );
        // Phone must be strictly less than Tv.
        assert!(
            PHONE_TARGET_MULTIPLIER < TV_TARGET_MULTIPLIER,
            "PHONE multiplier ({}) MUST be < TV multiplier ({}) for the \
             per-distance monotonicity invariant",
            PHONE_TARGET_MULTIPLIER,
            TV_TARGET_MULTIPLIER
        );
    }
}
