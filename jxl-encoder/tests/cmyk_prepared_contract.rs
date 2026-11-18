//! Already-linear engines retain Black and validate the final ICC/channel pair.
#![cfg(all(
    feature = "__expert",
    feature = "__internals",
    feature = "__pre_quantized"
))]
use jxl_encoder::{LossyConfig, api::ExtraChannel};
#[path = "support/jxl_signature.rs"]
mod jxl_signature;
use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;
fn profile() -> Vec<u8> {
    include_str!("fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect()
}
#[test]
fn prepared_linear_engines_keep_black_and_require_the_matching_profile() {
    let root = jxl_encoder::test_helpers::output_dir("cmyk-prepared-contract");
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let profile = profile();
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native decoder observer");
    let source: Vec<f32> = include_str!("fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
        .collect();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            for wide in [false, true] {
                let linear = source.repeat(width * height);
                let k8 = vec![224_u8; width * height];
                let k16 = vec![0xe0e0_u16; width * height];
                let extras = [if wide {
                    ExtraChannel::black_u16(&k16)
                } else {
                    ExtraChannel::black(&k8)
                }];
                let mut encoder = VarDctEncoder::new(0.01);
                encoder.profile = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .resolved_profile();
                encoder.effort = 3;
                encoder.encoder_mode = mode;
                encoder.force_strategy = Some(0);
                encoder.enable_patches = false;
                encoder.enable_gaborish = false;
                encoder.epf_level_override = Some(0);
                let pre = EncoderPrecomputed::compute(
                    width,
                    height,
                    &linear,
                    encoder.distance,
                    encoder.cfl_enabled,
                    encoder.ac_strategy_enabled,
                    encoder.pixel_domain_loss,
                    encoder.enable_noise,
                    encoder.enable_denoise,
                    encoder.enable_gaborish,
                    encoder.force_strategy,
                    &encoder.profile,
                    None,
                )
                .unwrap();
                let mut qf = vec![1; pre.xsize_blocks * pre.ysize_blocks];
                let params =
                    DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
                let transform = encoder
                    .transform_and_quantize_for_test(&pre, &mut qf, &params)
                    .unwrap();
                for (case, icc, black) in [
                    ("valid", Some(profile.as_slice()), true),
                    ("missing", None, true),
                    (
                        "rgb",
                        Some(zenpixels_convert::icc_profiles::DISPLAY_P3_V4),
                        true,
                    ),
                    ("cmyk-without-black", Some(profile.as_slice()), false),
                ] {
                    encoder.icc_profile = icc.map(<[u8]>::to_vec);
                    let extras = if black { &extras[..] } else { &[][..] };
                    for entry in ["linear", "linear-stop", "precomputed", "prequantized"] {
                        let result = match entry {
                            "linear" => encoder
                                .encode_with_extras(width, height, &linear, extras)
                                .map(|out| out.data),
                            "linear-stop" => encoder
                                .encode_with_extras_stop(width, height, &linear, extras, None)
                                .map(|out| out.data),
                            "precomputed" => {
                                encoder.encode_from_precomputed_with_extras(&pre, &qf, extras)
                            }
                            _ => encoder.encode_from_pre_quantized_ac_with_extras(
                                &pre,
                                &qf,
                                &transform.quant_dc,
                                &transform.quant_ac,
                                &transform.nzeros,
                                &transform.raw_nzeros,
                                extras,
                            ),
                        };
                        let label = format!("{policy}-{width}-{wide}-{case}-{entry}");
                        if case != "valid" {
                            let error = result.unwrap_err();
                            assert!(error.to_string().contains("CMYK ICC"), "{label}: {error}");
                            continue;
                        }
                        let mut bytes =
                            jxl_encoder::wrap_in_container(&result.unwrap(), None, None);
                        assert_eq!(&bytes[..32], &jxl_signature::LEVEL_TEN_PREFIX[..32]);
                        bytes.splice(
                            32..32,
                            jxl_signature::LEVEL_TEN_PREFIX[32..].iter().copied(),
                        );
                        jxl_signature::assert_signature(&bytes);
                        let dir = out.join(&label);
                        std::fs::create_dir(&dir).unwrap();
                        let input = dir.join("source.jxl");
                        std::fs::write(&input, &bytes).unwrap();
                        let (colors, extra) = decoded_extras::decode_current_planes(&bytes, &label);
                        assert_eq!(extra.len(), 1);
                        let native = std::process::Command::new(&oracle)
                            .arg(&input)
                            .arg(dir.join("native"))
                            .arg("linear")
                            .output()
                            .unwrap();
                        std::fs::write(dir.join("native.log"), &native.stderr).unwrap();
                        assert!(native.status.success(), "{label}");
                        let native_colors =
                            pfm::read_pfm_samples(&dir.join("native-0.pfm"), (width, height));
                        let native_black =
                            pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (width, height));
                        let maximum = if wide { 65535.0 } else { 255.0 };
                        let code = if wide { 0xe0e0 } else { 224 };
                        for values in [&extra[0], &native_black] {
                            assert!(values.iter().all(|v|v.is_finite()&&(*v*maximum).round()as u16==code),"{label} exact Black");
                        }
                        for values in [&colors, &native_colors] {
                            assert_eq!(values.len(), width * height * 3);
                            assert!(values.iter().all(|v| v.is_finite()));
                            let error = values
                                .iter()
                                .zip(source.iter().cycle())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            assert!(error <= 0.005, "{label} physical {error}");
                        }
                    }
                }
            }
        }
    }
}

#[cfg(feature = "rate-control")]
#[test]
fn rate_control_without_a_black_input_refuses_a_cmyk_profile() {
    use jxl_encoder::{Buffering, vardct::RateControlConfig};
    let profile = profile();
    let linear = [0.25_f32, 0.125, 0.0625].repeat(32 * 17);
    for (_, strategy, mode) in encoder_policies::policies() {
        for buffering in [Buffering::FullBuffered, Buffering::FullStreaming] {
            for iterations in [0, 1] {
                let mut encoder = VarDctEncoder::new(0.01);
                encoder.profile = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .resolved_profile();
                encoder.effort = 3;
                encoder.encoder_mode = mode;
                encoder.enable_patches = false;
                encoder.force_strategy = Some(0);
                encoder.buffering = buffering;
                encoder.icc_profile = Some(profile.clone());
                let error = encoder
                    .encode_with_rate_control_config(
                        32,
                        17,
                        &linear,
                        &RateControlConfig {
                            max_iterations: iterations,
                            ..Default::default()
                        },
                    )
                    .unwrap_err();
                assert!(error.to_string().contains("CMYK ICC"), "{error}");
            }
        }
    }
}
