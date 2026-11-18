//! Native-pixel evidence for the direction of iterative quantizer updates.
use super::*;
#[path = "../../tests/support/decoded_frames.rs"]
mod decoded_frames;
#[path = "../../tests/support/pfm.rs"]
mod pfm;

mod policies {
    //! Offered policies for issue-144 production-path checks.
    use crate::api::{EncoderImprovementsCustom, EncoderMode, EncoderStrategy};

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
}

fn linear(value: f32) -> f32 {
    if value.abs() <= 0.04045 {
        value / 12.92
    } else {
        ((value.abs() + 0.055) / 1.055).powf(2.4).copysign(value)
    }
}

#[test]
fn rate_control_adjustment_improves_the_observed_native_tile_metric() {
    use std::io::Write;
    let base = crate::test_helpers::output_dir("rate-control-direction");
    std::fs::create_dir_all(&base).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = base.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let (width, height) = (32_usize, 17_usize);
    let source: Vec<f32> = (0..width * height * 3)
        .map(|i| linear(((i * 73) % 251) as f32 / 255.0))
        .collect();
    let strategy =
        super::super::ac_strategy::AcStrategyMap::new_dct8(width.div_ceil(8), height.div_ceil(8));
    let mut journal = std::fs::File::create(root.join("metrics.tsv")).unwrap();
    writeln!(
        journal,
        "policy\tdistance\titerations\tactual_iterations\tbytes\tnative_p95\tprimary_p95"
    )
    .unwrap();
    let mut failures = Vec::new();
    for (policy, strategy_binding, mode) in policies::policies() {
        for distance in [0.1_f32, 1.0] {
            let mut baseline: Option<[f32; 2]> = None;
            for iterations in [0, 2] {
                let out = root.join(format!("{policy}-d{distance}-i{iterations}"));
                std::fs::create_dir(&out).unwrap();
                let mut encoder = VarDctEncoder::new(distance);
                encoder.profile = crate::LossyConfig::new(distance)
                    .with_effort(1)
                    .with_strategy(strategy_binding.clone())
                    .with_mode(mode)
                    .resolved_profile();
                encoder.effort = 1;
                encoder.encoder_mode = mode;
                encoder.enable_gaborish = false;
                encoder.epf_level_override = Some(0);
                encoder.enable_patches = false;
                encoder.force_strategy = Some(0);
                let config = RateControlConfig {
                    max_iterations: iterations,
                    tolerance: 0.0,
                    ..Default::default()
                };
                let (raw, actual_iterations) = encoder
                    .encode_with_rate_control_config(width, height, &source, &config)
                    .unwrap();
                let prepared = EncoderPrecomputed::compute_with_budget(
                    width,
                    height,
                    &source,
                    distance,
                    encoder.cfl_enabled,
                    encoder.ac_strategy_enabled,
                    encoder.pixel_domain_loss,
                    encoder.enable_noise,
                    encoder.enable_denoise,
                    false,
                    false,
                    false,
                    encoder.use_ans,
                    mode,
                    Some(0),
                    &encoder.profile,
                    None,
                    None,
                )
                .unwrap();
                let (prepared_bytes, prepared_iterations) =
                    encode_with_rate_control(&encoder, &prepared, &config).unwrap();
                assert_eq!(
                    prepared_bytes, raw,
                    "ordinary/prepared bytes: {policy} d{distance} i{iterations}"
                );
                assert_eq!(prepared_iterations, actual_iterations);
                assert_eq!(&raw[..2], &[255, 10]);
                let mut bytes = crate::wrap_in_container(&raw, None, None);
                bytes.splice(32..32, b"\0\0\0\x09jxll\x0a".iter().copied());
                std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                let native_result = std::process::Command::new(crate::test_helpers::djxl_path())
                    .arg(out.join("input.jxl"))
                    .arg(out.join("native.pfm"))
                    .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
                    .output()
                    .unwrap();
                std::fs::write(out.join("native.log"), &native_result.stderr).unwrap();
                let primary_result = std::panic::catch_unwind(|| {
                    decoded_frames::decode_current_frames(&bytes, "rate-control update direction")
                });
                std::fs::write(out.join("decoder-outcomes.json"),serde_json::to_vec_pretty(
                &serde_json::json!({"native_complete":native_result.status.success(),"primary_complete":primary_result.is_ok()})
            ).unwrap()).unwrap();
                assert!(native_result.status.success() && primary_result.is_ok());
                let primary = primary_result.unwrap();
                assert_eq!(primary.len(), 1);
                assert!(primary[0].1.is_empty());
                let primary_linear: Vec<f32> = primary[0].0.iter().copied().map(linear).collect();
                let native = pfm::read_pfm_samples(&out.join("native.pfm"), (width, height));
                assert_eq!(native.len(), source.len());
                assert_eq!(primary_linear.len(), source.len());
                assert!(
                    native
                        .iter()
                        .zip(&primary_linear)
                        .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 0.001)
                );
                let mut observed = [0.0_f32; 2];
                for (result, pixels) in observed.iter_mut().zip([&native, &primary_linear]) {
                    let map =
                        compute_butteraugli_diffmap(&source, pixels, width, height, 80.0, None)
                            .unwrap();
                    *result = TileDistMap::from_diffmap(&map, width, height, &strategy, None)
                        .unwrap()
                        .percentile_95();
                    assert!(result.is_finite());
                }
                writeln!(
                    journal,
                    "{policy}\t{distance}\t{iterations}\t{actual_iterations}\t{}\t{}\t{}",
                    bytes.len(),
                    observed[0],
                    observed[1]
                )
                .unwrap();
                journal.flush().unwrap();
                if let Some(initial) = baseline {
                    assert!(
                        actual_iterations > 0,
                        "selected fixture must run an adjustment"
                    );
                    assert!(
                        initial.iter().all(|v| *v > distance),
                        "selected initial errors must exceed the target"
                    );
                    for (label, after, before) in [
                        ("native", observed[0], initial[0]),
                        ("primary", observed[1], initial[1]),
                    ] {
                        if after >= before {
                            failures
                                .push(format!("{policy} {label} d{distance}: {before} -> {after}"));
                        }
                    }
                } else {
                    baseline = Some(observed);
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "rate-control tile metric worsened: {failures:?}; {root:?}"
    );
}

#[test]
fn above_target_quantizer_updates_never_reverse_at_integer_rounding() {
    let strategy = super::super::ac_strategy::AcStrategyMap::new_dct8(1, 1);
    for old in 1_u8..=255 {
        for ratio in [1.00001_f32, 1.01, 1.1, 2.0, 100.0] {
            let tile = TileDistMap::from_diffmap(&[ratio], 1, 1, &strategy, None).unwrap();
            for (iteration, initial) in [(0, old), (1, old), (1, 1), (2, old)] {
                let mut field = [old];
                adjust_quant_field(&mut field, &tile, &[initial], 1.0, iteration, 1.0, 255.0, 1);
                assert!(
                    field[0] > old || old == 255,
                    "above-target quantizer reversed or stalled: {old} -> {} at ratio {ratio}, iteration {iteration}",
                    field[0]
                );
            }
        }
    }
}

#[test]
fn above_target_quantizer_scales_finer_by_the_error_ratio() {
    let strategy = super::super::ac_strategy::AcStrategyMap::new_dct8(1, 1);
    let tile = TileDistMap::from_diffmap(&[2.0], 1, 1, &strategy, None).unwrap();
    let mut field = [10];
    adjust_quant_field(&mut field, &tile, &[10], 1.0, 0, 1.0, 255.0, 1);
    // At twice the requested error, double the inverse AC step. A single
    // minimum-progress increment does not implement proportional feedback.
    assert_eq!(field, [20]);
}
