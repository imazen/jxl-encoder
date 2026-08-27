// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Print the zq_seed 8-feature vector + the shipped head's q0 at t∈{80,88}
//! for each PNG path argument (`name\tf0..f7\tq0_t80\tq0_t88`), decoding
//! with the SAME `image::open().to_rgb8()` path the census instrument uses.
//! Offline table builds for the S4 B2 arm
//! (benchmarks/s4_iter1_eps_wave_2026-08-27.md) consume this so the census
//! features/q0 are exactly the in-binary ones.

fn main() {
    use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
    for path in std::env::args().skip(1) {
        let img = image::open(&path).expect("decode").to_rgb8();
        let (w, h) = (img.width(), img.height());
        let set = FeatureSet::just(AnalysisFeature::FlatColorBlockRatio)
            .with(AnalysisFeature::GradientFraction)
            .with(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::HighFreqEnergyRatio)
            .with(AnalysisFeature::AqMapStd)
            .with(AnalysisFeature::GrayscaleScore)
            .with(AnalysisFeature::LumaHistogramEntropy)
            .with(AnalysisFeature::QuantSurvivalY);
        let a = zenanalyze::try_analyze_features_rgb8(img.as_raw(), w, h, &AnalysisQuery::new(set))
            .expect("analyze");
        let g = |f: AnalysisFeature| -> f32 {
            if let Some(v) = a.get_f32(f) {
                return v;
            }
            match a.get(f).expect("feature") {
                zenanalyze::feature::FeatureValue::U32(x) => x as f32,
                zenanalyze::feature::FeatureValue::F32(x) => x,
                _ => f32::NAN,
            }
        };
        let feats = [
            g(AnalysisFeature::FlatColorBlockRatio),
            g(AnalysisFeature::GradientFraction),
            g(AnalysisFeature::DistinctColorBins),
            g(AnalysisFeature::HighFreqEnergyRatio),
            g(AnalysisFeature::AqMapStd),
            g(AnalysisFeature::GrayscaleScore),
            g(AnalysisFeature::LumaHistogramEntropy),
            g(AnalysisFeature::QuantSurvivalY),
        ];
        let px = u64::from(w) * u64::from(h);
        let q80 = jxl_encoder::zq_seed::predict_q0_from_features(&feats, 80.0, px);
        let q88 = jxl_encoder::zq_seed::predict_q0_from_features(&feats, 88.0, px);
        let fs = feats.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join("\t");
        println!(
            "{path}\t{fs}\t{}\t{}",
            q80.map_or("nan".into(), |v| format!("{v}")),
            q88.map_or("nan".into(), |v| format!("{v}"))
        );
    }
}
