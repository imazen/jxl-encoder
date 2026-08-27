//! Zq seed head for the zensim target controller — predicts the
//! iteration-1 ENCODER QUALITY for a zensim-targeted encode from zenanalyze
//! content features; callers bridge to distance with
//! [`crate::api::quality_to_distance`] (the same public mapping the training
//! sweep's zenjxl encodes went through). Replaces the content-blind 3-step
//! staircase (`seed_distance_for_target` in the A/B driver) when it wins the
//! registered census gate.
//!
//! Per-codec copy of zenjpeg's `zq_seed` (loop-ownership directive; zenavif
//! `q0_head` is the family exemplar): a deterministic fitted-constants head —
//! no model file — over a hinge target basis with feature×target
//! interactions.
//!
//! # Fit provenance (2026-08-26)
//!
//! `scripts/zq-seed/fit_zq_seed.py` on the 07-01 canonical picker set
//! (`canonical-picker-2026-07-01-zensimA/zenjxl_lossy`, zensimA-profile
//! `score_zensim`): 80,745 full 9-point per-rendition q→zensim curves,
//! PAVA-isotonized, inverse-labeled at targets {40,45,…,90} (775,875 train
//! labels). Greedy LOO-origin-p90 selection. Diagnostic G-J1 (validate,
//! 466,885 labels): |q0−q*| p50 7.09, p90 21.68.
//!
//! Pre-registered wave: `benchmarks/zq_seed_wave_2026-08-26.md`. **The
//! decision gate (G-J2) is the real 27-cell census A/B** — no offline sim is
//! claimed (the in-loop controller is not faithfully portable to stored
//! curves). Until that gate PASSES, this module is inert library surface;
//! the staircase stays the default seed.
//!
//! # Scope
//!
//! Fitted for `[0,100]`-scaled zensim targets in 40–90 (basis clamps to the
//! band). `None` on any non-finite input — callers keep the staircase, so
//! the head can only re-seed, never break (G-J3).

/// The eight zenanalyze features, in fit order. `distinct_color_bins`
/// (index 2) is `ln_1p`-transformed inside [`predict_q0_from_features`] —
/// pass RAW values. (Names match `zenanalyze::feature::AnalysisFeature`:
/// FlatColorBlockRatio, GradientFraction, DistinctColorBins,
/// HighFreqEnergyRatio, AqMapStd, GrayscaleScore, LumaHistogramEntropy,
/// QuantSurvivalY — kept as documentation rather than a typed const so the
/// core lib does not require the `learned-admission`/zenanalyze feature.)
pub const ZQ_FEATURE_NAMES: [&str; 8] = [
    "flat_color_block_ratio",
    "gradient_fraction",
    "distinct_color_bins",
    "high_freq_energy_ratio",
    "aq_map_std",
    "grayscale_score",
    "luma_histogram_entropy",
    "quant_survival_y",
];

/// Fitted robust-L1 coefficients (fit_zq_seed.py 2026-08-26). Layout:
/// `[const, tn, h50, h60, h70, h80, h85, logpx_n, f_0..f_7, f_0*tn..f_7*tn,
/// f_0*h80..f_7*h80]` with `tn=(t−65)/25`, `h_k=max(t−k,0)/10`,
/// `logpx_n=(ln(px)−13)/3`.
const ZQ_COEFS: [f64; 32] = [
    18.236299968062756,  // const,
    14.225576735397231,  // tn,
    6.851106604548416,   // h50,
    5.964310698099696,   // h60,
    -4.165298017327582,  // h70,
    -1.2199460933094541, // h80,
    -18.607663900312122, // h85,
    -5.701052276921821,  // logpx_n,
    -53.93417957141681,  // flat_color_block_ratio,
    -44.31518225784404,  // gradient_fraction,
    2.8328625158937566,  // ln_1p(distinct_color_bins),
    9.268970257945522,   // high_freq_energy_ratio,
    3.337965136386378,   // aq_map_std,
    -2.6861110238041777, // grayscale_score,
    3.468920603273567,   // luma_histogram_entropy,
    -34.44467613388063,  // quant_survival_y,
    -57.349554683122214, // fcbr*tn,
    -15.942396314039918, // gf*tn,
    0.4188634254074954,  // dcb*tn,
    1.1366157440838158,  // hfer*tn,
    12.60834299763201,   // aqs*tn,
    2.9856010278392433,  // gs*tn,
    2.6055194222082547,  // lhe*tn,
    -59.37628136420917,  // qsy*tn,
    103.47921161919237,  // fcbr*h80,
    36.821402449311186,  // gf*h80,
    -0.9023333281312831, // dcb*h80,
    -14.009579928131277, // hfer*h80,
    -8.837165071050778,  // aqs*h80,
    3.0575694167432457,  // gs*h80,
    -3.6172174158927266, // lhe*h80,
    78.4065070669234,    // qsy*h80
];

const ZQ_T_MIN: f64 = 40.0;
const ZQ_T_MAX: f64 = 90.0;

/// Pure evaluation on already-extracted feature values (in
/// [`ZQ_FEATURE_NAMES`] order, RAW — the transform is applied here).
/// Returns the seed ENCODER QUALITY clamped to `[1, 100]`; bridge to
/// distance with [`crate::api::quality_to_distance`]. `None` if any input
/// is non-finite.
#[must_use]
pub fn predict_q0_from_features(features: &[f32; 8], target: f64, pixels: u64) -> Option<f32> {
    if !target.is_finite() || features.iter().any(|f| !f.is_finite()) {
        return None;
    }
    let t = target.clamp(ZQ_T_MIN, ZQ_T_MAX);
    let tn = (t - 65.0) / 25.0;
    let h = |k: f64| (t - k).max(0.0) / 10.0;
    let logpx_n = (f64::from(u32::try_from(pixels.max(1)).unwrap_or(u32::MAX)).ln() - 13.0) / 3.0;

    let mut fv = [0.0f64; 8];
    for (i, f) in features.iter().enumerate() {
        fv[i] = f64::from(*f);
    }
    fv[2] = fv[2].max(0.0).ln_1p(); // distinct_color_bins

    let mut x = [0.0f64; 32];
    x[0] = 1.0;
    x[1] = tn;
    x[2] = h(50.0);
    x[3] = h(60.0);
    x[4] = h(70.0);
    x[5] = h(80.0);
    x[6] = h(85.0);
    x[7] = logpx_n;
    let h80 = h(80.0);
    for i in 0..8 {
        x[8 + i] = fv[i];
        x[16 + i] = fv[i] * tn;
        x[24 + i] = fv[i] * h80;
    }
    let q0: f64 = ZQ_COEFS.iter().zip(x.iter()).map(|(c, v)| c * v).sum();
    #[allow(clippy::cast_possible_truncation)]
    Some((q0 as f32).clamp(1.0, 100.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vector, computed independently by the fit pipeline's Python:
    /// raw [0.25, 0.4, 40.0, 0.3, 0.15, 0.6, 4.2, 0.5], t=80, px=331776
    /// → 13.634312.
    #[test]
    fn golden_matches_fit_pipeline() {
        let q0 =
            predict_q0_from_features(&[0.25, 0.4, 40.0, 0.3, 0.15, 0.6, 4.2, 0.5], 80.0, 331_776)
                .unwrap();
        assert!((q0 - 13.634_312).abs() < 1e-3, "q0 = {q0}");
    }

    #[test]
    fn non_finite_inputs_return_none() {
        assert!(
            predict_q0_from_features(&[f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 70.0, 1000)
                .is_none()
        );
        assert!(predict_q0_from_features(&[0.0; 8], f64::NAN, 1000).is_none());
    }

    #[test]
    fn always_a_valid_quality() {
        for t in [0.0, 40.0, 65.0, 88.0, 100.0] {
            for f in [
                [0.0f32; 8],
                [1.0; 8],
                [0.9, 1.0, 5000.0, 1.0, 1.0, 1.0, 8.0, 1.0],
            ] {
                let q0 = predict_q0_from_features(&f, t, 250_000).unwrap();
                assert!((1.0..=100.0).contains(&q0), "q0 = {q0} at t={t}");
            }
        }
    }

    #[test]
    fn band_edges_clamp_target_basis() {
        let f = [0.2f32, 0.3, 100.0, 0.4, 0.1, 0.5, 5.0, 0.6];
        let a = predict_q0_from_features(&f, 40.0, 100_000).unwrap();
        let b = predict_q0_from_features(&f, 10.0, 100_000).unwrap();
        assert!(
            (a - b).abs() < 1e-6,
            "below-band targets must pin to the band edge"
        );
    }
}
