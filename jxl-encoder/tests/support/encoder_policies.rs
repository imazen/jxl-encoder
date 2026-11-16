//! Offered policies for issue-144 production-path checks.
use jxl_encoder::api::{EncoderImprovementsCustom, EncoderMode, EncoderStrategy};

pub(crate) fn policies() -> Vec<(String, EncoderStrategy, EncoderMode)> {
    let mut policies = vec![
        (
            "libjxl".into(),
            EncoderStrategy::Libjxl,
            EncoderMode::Reference,
        ),
        (
            "lean".into(),
            EncoderStrategy::LeanFaster,
            EncoderMode::Reference,
        ),
        (
            "zen".into(),
            EncoderStrategy::Zenjxl,
            EncoderMode::Reference,
        ),
        (
            "aggressive".into(),
            EncoderStrategy::Aggressive,
            EncoderMode::Reference,
        ),
        (
            "experimental".into(),
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ];
    for compact in [false, true] {
        for smoothing in [false, true] {
            policies.push((
                format!("custom-{compact}-{smoothing}"),
                EncoderStrategy::Custom(Box::new(EncoderImprovementsCustom {
                    header_all_default_fast_paths: compact,
                    dc_adaptive_smoothing: smoothing,
                    ..Default::default()
                })),
                EncoderMode::Reference,
            ));
        }
    }
    let experimental: Vec<_> = policies
        .iter()
        .filter(|(_, _, mode)| *mode == EncoderMode::Reference)
        .map(|(name, strategy, _)| {
            (
                format!("{name}-experimental"),
                strategy.clone(),
                EncoderMode::Experimental,
            )
        })
        .collect();
    policies.extend(experimental);
    policies
}
