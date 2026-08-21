//! W44-231: learned qf-seed **sub-band** admission (confident-BAD exclude).
//!
//! The W44-105/107/108 screenshot qf-seed lift's content gating grew as
//! per-exemplar threshold patches (task-12 p25, W44-176 terminal-class,
//! W44-AUDIT-6 high-colour, W44-230 textured-low-colour). The 2026-08-21
//! held-out validation showed the misfire family is not separable in the
//! four hand proxies — so this module replaces threshold accretion with a
//! LEARNED confident-BAD exclude, trained on TRAIN-digit imazen-26 images
//! only (ids ending 0/2/4/6/8 + gb82-sc; val/test digits 1/3/5/7/9 were
//! never fit on).
//!
//! ## Model
//!
//! 4-feature logistic regression over zenanalyze features
//! (`DistinctColorBins`, `EdgeSlopeStdev`, `LaplacianVarianceP99`,
//! `LaplacianVariance`), z-scored, `P(bad) = 1 - sigmoid(w.z + b)`.
//! The lift is EXCLUDED when `P(bad) >= 0.90` — a strictly-narrowing
//! decision (it can only turn the lift OFF where the band would fire).
//!
//! Label (STRICT-GOOD, web-aggressive product goal): with the raw-band
//! lift on, the encode must Pareto-dominate-or-tie cjxl 0.12 — at least
//! its ssim2 for at most its bytes — at e7 d2.5 (and e9 d3.0 for the
//! first 322 images). Everything else (premium quality-buys included) is
//! BAD *in the d < 3.5 sub-band this model governs*: measured there, the
//! premium class is RD-indistinguishable from the adjudicated misfires
//! (0.03-0.13 ssim2 per +1 % bytes for both). The d >= 3.5 main band —
//! the W44-174 text-sharpness calibration (windows95 gate sentinel) — is
//! NOT consulted here and keeps its ship behaviour.
//!
//! ## Held-out performance (val digits 1/3/5, 56 fired images)
//!
//! Block threshold 0.90: blocks 12, ALL 12 labelled BAD (precision
//! 1.00, zero strict-good losses); recall 12/37 bads. The hand W44-230
//! box blocked 2/37 on the same set. Training/eval data:
//! `benchmarks/w44_231_lift_admission_model_2026-08-21.json` +
//! `benchmarks/imazen26_hunt_2026-08-20.md` (held-out section) +
//! `scripts/hunt/train_lift_admission.py`.
//!
//! ## Fail-open contract
//!
//! Feature extraction failing, the image being too small, or the
//! `learned-admission` cargo feature being off all yield "no exclude" —
//! byte-identical to the pre-W44-231 stack. The registry gate
//! `learned_subband_exclude` (Zenjxl/Aggressive on, Libjxl/LeanFaster
//! off) and env escape `JXL_W44_231_DISABLE=1` sit on top.

/// Feature order matches the frozen model artifact.
const FEATURE_MEAN: [f64; 4] = [
    1_009.183_486_238_532,
    53.550_807_880_733_94,
    193.981_651_376_146_8,
    7.566_871_137_614_677,
];
const FEATURE_STD: [f64; 4] = [
    1_822.840_223_413_524_7,
    16.111_159_501_516_457,
    76.668_233_395_271_7,
    11.431_857_529_368_601,
];
const COEF: [f64; 4] = [
    -1.566_932_465_062_690_8,
    1.132_008_970_463_635_4,
    1.081_667_824_478_384_6,
    -0.485_036_553_420_064_5,
];
const INTERCEPT: f64 = -1.081_110_722_773_767_3;
/// Block the sub-band lift when `P(bad)` is at least this.
const BLOCK_THRESHOLD_P_BAD: f64 = 0.90;

/// Raw 4-feature vector in model order:
/// `[DistinctColorBins, EdgeSlopeStdev, LaplacianVarianceP99, LaplacianVariance]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LiftAdmissionFeatures(pub(crate) [f64; 4]);

/// `P(bad)` for a feature vector — pure model math, unit-testable.
pub(crate) fn p_bad(f: &LiftAdmissionFeatures) -> f64 {
    let mut z = INTERCEPT;
    for i in 0..4 {
        let x = (f.0[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
        z += COEF[i] * x;
    }
    1.0 - 1.0 / (1.0 + (-z).exp())
}

/// Confident-BAD verdict: exclude the sub-band lift on this image.
pub(crate) fn confident_bad(f: &LiftAdmissionFeatures) -> bool {
    p_bad(f) >= BLOCK_THRESHOLD_P_BAD
}

/// Extract the model's features from tightly-packed RGB8 pixels.
///
/// Returns `None` (fail-open: no exclude) when extraction fails or any
/// requested feature is unavailable. The zenanalyze pass is
/// budget-sampled (~ms scale) and only runs when the encode is already
/// inside the proxies band (`effort >= 5 || distance >= 2`), alongside
/// the existing shared-proxies sweep.
#[cfg(feature = "learned-admission")]
pub(crate) fn extract_features(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Option<LiftAdmissionFeatures> {
    use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
    let set = FeatureSet::just(AnalysisFeature::DistinctColorBins)
        .with(AnalysisFeature::EdgeSlopeStdev)
        .with(AnalysisFeature::LaplacianVarianceP99)
        .with(AnalysisFeature::LaplacianVariance);
    let query = AnalysisQuery::new(set);
    let a = zenanalyze::try_analyze_features_rgb8(rgb, width, height, &query).ok()?;
    let g = |f: AnalysisFeature| -> Option<f64> {
        if let Some(v) = a.get_f32(f) {
            if v.is_nan() {
                return None;
            }
            return Some(v as f64);
        }
        match a.get(f)? {
            zenanalyze::feature::FeatureValue::U32(x) => Some(x as f64),
            zenanalyze::feature::FeatureValue::F32(x) if !x.is_nan() => Some(x as f64),
            _ => None,
        }
    };
    Some(LiftAdmissionFeatures([
        g(AnalysisFeature::DistinctColorBins)?,
        g(AnalysisFeature::EdgeSlopeStdev)?,
        g(AnalysisFeature::LaplacianVarianceP99)?,
        g(AnalysisFeature::LaplacianVariance)?,
    ]))
}

/// Layout-adapting entry: build tightly-packed RGB8 (same channel
/// mapping as [`crate::api::ingest::compute_w44_91_zenanalyze_proxies`])
/// and return the confident-BAD verdict. `None` for non-sRGB-u8 layouts
/// or any extraction failure (fail-open: no exclude). Training data was
/// produced on exactly this normalization (alpha dropped, BGR swapped).
#[cfg(feature = "learned-admission")]
pub(crate) fn extract_rgb8_verdict(
    pixels: &[u8],
    width: usize,
    height: usize,
    layout: crate::api::PixelLayout,
) -> Option<bool> {
    use crate::api::PixelLayout;
    let (r_off, g_off, b_off, bpp) = match layout {
        PixelLayout::Rgb8 => (0usize, 1usize, 2usize, 3usize),
        PixelLayout::Rgba8 => (0, 1, 2, 4),
        PixelLayout::Bgr8 => (2, 1, 0, 3),
        PixelLayout::Bgra8 => (2, 1, 0, 4),
        _ => return None,
    };
    let expected_len = width.checked_mul(height)?.checked_mul(bpp)?;
    if pixels.len() < expected_len || width == 0 || height == 0 {
        return None;
    }
    let rgb: Vec<u8> = if bpp == 3 && r_off == 0 {
        pixels[..expected_len].to_vec()
    } else {
        let mut out = Vec::with_capacity(width * height * 3);
        for px in pixels[..expected_len].chunks_exact(bpp) {
            out.push(px[r_off]);
            out.push(px[g_off]);
            out.push(px[b_off]);
        }
        out
    };
    let f = extract_features(&rgb, width as u32, height as u32)?;
    Some(confident_bad(&f))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feature vectors measured by the training extractor on HELD-OUT
    /// (val-digit) images — the embedded model must reproduce the
    /// validated decisions (benchmarks/w44_231_lift_admission_model_
    /// 2026-08-21.json; vectors from the 2026-08-21 val label run).
    #[test]
    fn w44_231_heldout_decisions() {
        // (name, [bins, edge_slope_stdev, lap_p99, lap_var], expect_blocked)
        let cells: &[(&str, [f64; 4], bool)] = &[
            // Smooth white-bg product shot (the 9933 misfire): block.
            ("9933", [501.0, 12.351, 4.0, 0.013], true),
            // Mobile screenshot with photo content: block.
            ("8025", [3739.0, 48.093, 163.0, 1.512], true),
            // NOAA doc page misfire: block.
            ("5343", [654.0, 58.145, 0.0, 0.446], true),
            // Chart (7125 misfire family): block.
            ("7125", [180.0, 42.253, 76.0, 0.285], true),
            // Substack text screenshot — strict-good DOM win: admit.
            ("8023", [2202.0, 65.474_976, 220.0, 6.254_371], false),
            // Margin case: strict-good at p_bad ~0.881, must stay admitted.
            ("8013", [2719.0, 55.881, 207.0, 2.734], false),
        ];
        for (name, f, expect) in cells {
            let feats = LiftAdmissionFeatures(*f);
            assert_eq!(
                confident_bad(&feats),
                *expect,
                "{name}: p_bad={:.3}",
                p_bad(&feats)
            );
        }
    }

    /// Direction sanity at the operating region: DENSER edge structure
    /// (UI text) lowers P(bad); MORE distinct colours (photographic
    /// content in screenshot clothing) raises it.
    #[test]
    fn w44_231_direction_sanity() {
        let base = LiftAdmissionFeatures([2000.0, 55.0, 200.0, 4.0]);
        let denser_edges = LiftAdmissionFeatures([2000.0, 70.0, 200.0, 4.0]);
        assert!(p_bad(&denser_edges) < p_bad(&base));
        let more_colours = LiftAdmissionFeatures([6000.0, 55.0, 200.0, 4.0]);
        assert!(p_bad(&more_colours) > p_bad(&base));
    }
}
