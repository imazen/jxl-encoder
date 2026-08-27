//! S4 iteration-1 elasticity prior (arm B3) — the per-image default for the
//! zensim loop's controller exponent (`JXL_ZENSIM_CTRL_EXP` unset).
//!
//! Registered wave + census: `benchmarks/s4_iter1_eps_wave_2026-08-27.md` —
//! B3 FULL PASS on the stricter bars (overall median |err| 0.832 → 0.527 =
//! 36.7%, ±2 hits 21→22, nonphoto 1.836→1.593 now IMPROVES). **User-approved
//! default wiring 2026-08-28** (AskUserQuestion; the standing propose-only
//! rule was lifted for this head by an explicit yes).
//!
//! Mechanism (the wave's frozen unit bridge, ported verbatim):
//!   power step  ln g = exp · (ln L − ln L_t),  L = 100 − score
//!   ε̂_prior(t) = (slope(t) / (100 − target)) / DQ(t)
//!   exp        = clamp(−1/ε̂, 0.25, 2.0)
//! with slope(t) = the 8-feature standardized ridge (`s4_b2_refit.py`,
//! coefficients below from `b2_slope_fit.json`, fit 2026-08-27 on the
//! zq_seed feature basis) and DQ = dlog d / dlog q, a numeric central
//! difference of the public [`crate::api::quality_to_distance`] at the
//! zq_seed q0 — the same no-hand-rolled-mapping rule the zq wave registered.
//!
//! Guards (registered): bridge-invalid cells keep exp = 1.0 — q0 < 40,
//! slope ≤ 0, non-negative ε̂, or a target outside the fitted band. Heads
//! exist at t80 and t88 only; production generalization = nearest head for
//! target ≥ 75 (t < 84 → t80, else t88), targets below 75 keep 1.0 (the
//! registration's t70 rule: the prior has no signal there).

use crate::api::quality_to_distance;

/// Feature order (= `zq_seed::ZQ_FEATURE_NAMES`; `distinct_color_bins` at
/// index 2 enters ln_1p BEFORE standardization, matching the fit).
const LN1P_INDEX: usize = 2;

/// (mu, sd, w[9] with intercept last) per head, from `b2_slope_fit.json`.
const T80: ([f64; 8], [f64; 8], [f64; 9]) = (
    [
        0.204472, 0.196569, 6.532981, 0.305915, 1.336029, 0.402614, 3.283066, 0.120037,
    ],
    [
        0.209853, 0.147014, 1.360047, 0.36495, 0.618007, 0.337086, 0.836921, 0.075154,
    ],
    [
        -6.684126, -6.921484, -2.216326, 5.003041, -2.919902, -1.876923, 3.943367, -3.70396,
        32.29655,
    ],
);
const T88: ([f64; 8], [f64; 8], [f64; 9]) = (
    [
        0.384648, 0.212174, 6.142343, 0.229882, 1.763971, 0.587992, 2.740513, 0.088714,
    ],
    [
        0.213994, 0.156557, 1.313925, 0.233934, 0.455524, 0.299188, 0.839728, 0.061458,
    ],
    [
        -10.430866, -5.418874, 0.798541, 0.599832, 2.108624, -0.272945, 2.687273, -1.730443,
        27.766547,
    ],
);

/// dscore/dlogq slope prediction for the nearest fitted head. `None` when
/// the target is below the fitted band.
fn slope_prior(features_raw: &[f32; 8], target: f64) -> Option<f64> {
    if !(75.0..=99.5).contains(&target) {
        return None;
    }
    let (mu, sd, w) = if target < 84.0 { T80 } else { T88 };
    let mut acc = w[8];
    for i in 0..8 {
        let mut v = f64::from(features_raw[i]);
        if !v.is_finite() {
            return None;
        }
        if i == LN1P_INDEX {
            v = v.max(0.0).ln_1p();
        }
        acc += w[i] * (v - mu[i]) / sd[i];
    }
    Some(acc)
}

/// dlog d / dlog q at `q0` — central difference of the public quality→
/// distance mapping in log-log space (h = 0.05 ln-units).
fn dlogd_dlogq(q0: f32) -> Option<f64> {
    const H: f64 = 0.05;
    let q_hi = (f64::from(q0) * H.exp()).clamp(1.0, 100.0);
    let q_lo = (f64::from(q0) * (-H).exp()).clamp(1.0, 100.0);
    if q_hi <= q_lo {
        return None;
    }
    let d_hi = f64::from(quality_to_distance(q_hi as f32));
    let d_lo = f64::from(quality_to_distance(q_lo as f32));
    if !(d_hi.is_finite() && d_lo.is_finite()) || d_hi <= 0.0 || d_lo <= 0.0 {
        return None;
    }
    Some((d_hi.ln() - d_lo.ln()) / (q_hi.ln() - q_lo.ln()))
}

/// The B3 controller exponent from already-extracted zq features. `None`
/// (⇒ caller keeps 1.0) on any registered guard.
#[must_use]
pub(crate) fn iter1_ctrl_exp_from_features(
    features_raw: &[f32; 8],
    target: f64,
    pixels: u64,
) -> Option<f64> {
    let slope = slope_prior(features_raw, target)?;
    if slope <= 0.0 {
        return None;
    }
    let q0 = crate::zq_seed::predict_q0_from_features(features_raw, target, pixels)?;
    if q0 < 40.0 {
        return None;
    }
    let dq = dlogd_dlogq(q0)?;
    let eps = (slope / (100.0 - target)) / dq;
    if !eps.is_finite() || eps >= 0.0 {
        return None;
    }
    Some((-1.0 / eps).clamp(0.25, 2.0))
}

/// Loop-entry form: linear-RGB f32 planes (interleaved rgb, `w*h*3`) →
/// sRGB8 → zenanalyze features → exponent. Fail-open `None` everywhere
/// (extraction failure can only fall back to the shipped constant, never
/// break an encode). Extraction mirrors `vardct::learned_admission`.
#[cfg(feature = "learned-admission")]
#[must_use]
pub(crate) fn iter1_ctrl_exp_linear_rgb(
    linear_rgb: &[f32],
    width: usize,
    height: usize,
    target: f64,
) -> Option<f64> {
    use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};
    if width == 0 || height == 0 || linear_rgb.len() < width * height * 3 {
        return None;
    }
    #[inline]
    fn srgb8(x: f32) -> u8 {
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        let e = if x <= 0.003_130_8 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (e * 255.0 + 0.5) as u8
    }
    let rgb: Vec<u8> = linear_rgb[..width * height * 3]
        .iter()
        .map(|&v| srgb8(v))
        .collect();
    const FEATS: [AnalysisFeature; 8] = [
        AnalysisFeature::FlatColorBlockRatio,
        AnalysisFeature::GradientFraction,
        AnalysisFeature::DistinctColorBins,
        AnalysisFeature::HighFreqEnergyRatio,
        AnalysisFeature::AqMapStd,
        AnalysisFeature::GrayscaleScore,
        AnalysisFeature::LumaHistogramEntropy,
        AnalysisFeature::QuantSurvivalY,
    ];
    let mut set = FeatureSet::just(FEATS[0]);
    for f in &FEATS[1..] {
        set = set.with(*f);
    }
    let a = zenanalyze::try_analyze_features_rgb8(
        &rgb,
        width as u32,
        height as u32,
        &AnalysisQuery::new(set),
    )
    .ok()?;
    let mut fv = [0.0f32; 8];
    for (i, feat) in FEATS.iter().enumerate() {
        fv[i] = a.get_f32(*feat).or_else(|| match a.get(*feat)? {
            FeatureValue::U32(x) => Some(x as f32),
            FeatureValue::F32(x) => Some(x),
            _ => None,
        })?;
    }
    iter1_ctrl_exp_from_features(&fv, target, (width * height) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fixture vector in-range for both heads (raw values near the fit mu).
    const F: [f32; 8] = [0.2, 0.2, 700.0, 0.3, 1.3, 0.4, 3.3, 0.12];

    #[test]
    fn slope_head_matches_hand_computation_t80() {
        // Hand-fold the standardized ridge for the fixture at t=80.
        let (mu, sd, w) = T80;
        let mut want = w[8];
        for i in 0..8 {
            let mut v = f64::from(F[i]);
            if i == LN1P_INDEX {
                v = v.ln_1p();
            }
            want += w[i] * (v - mu[i]) / sd[i];
        }
        let got = slope_prior(&F, 80.0).unwrap();
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }

    #[test]
    fn registered_guards_hold() {
        // Below the fitted band: no prior.
        assert_eq!(slope_prior(&F, 70.0), None);
        assert_eq!(iter1_ctrl_exp_from_features(&F, 70.0, 1 << 20), None);
        // Non-finite feature: no prior.
        let mut bad = F;
        bad[4] = f32::NAN;
        assert_eq!(iter1_ctrl_exp_from_features(&bad, 80.0, 1 << 20), None);
        // Exponent, when it fires, is inside the registered clamp.
        for t in [78.0, 80.0, 85.0, 88.0, 92.0] {
            if let Some(e) = iter1_ctrl_exp_from_features(&F, t, 1 << 20) {
                assert!((0.25..=2.0).contains(&e), "exp {e} at t {t}");
            }
        }
    }

    #[test]
    fn dq_is_negative_over_the_head_band() {
        // Higher quality => lower distance, so dlog d/dlog q < 0 wherever
        // the bridge evaluates it.
        for q in [40.0f32, 60.0, 80.0, 95.0] {
            let dq = dlogd_dlogq(q).unwrap();
            assert!(dq < 0.0, "dq {dq} at q {q}");
        }
    }
}
