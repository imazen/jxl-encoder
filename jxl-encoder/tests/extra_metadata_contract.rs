//! Extra-channel syntax limits and their public request admission.
#![cfg(feature = "__expert")]
use jxl_encoder::api::ExtraChannel;
use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};
use std::io::Write;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("extra-metadata-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|index| {
            let path = root.join(format!("run-{index}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", path.display()),
            }
        })
        .unwrap()
}

#[test]
fn extra_metadata_invalid_public_requests_refuse_before_output() {
    let out = directory();
    let pixels = vec![64_u8; 32 * 17 * 3];
    let full = vec![0_u8; 32 * 17];
    let shifted = vec![0_u8; 2 * 2];
    let cases = [
        ("shift4", ExtraChannel::depth(&shifted).with_dim_shift(4)),
        ("cfa275", ExtraChannel::cfa(&full, 275)),
        ("cfa-max", ExtraChannel::cfa(&full, u32::MAX)),
        (
            "solidity2",
            ExtraChannel::spot_color(&full, [0.0, 0.0, 0.0, 2.0]),
        ),
        (
            "solidity-negative",
            ExtraChannel::spot_color(&full, [0.0, 0.0, 0.0, -0.001]),
        ),
        (
            "solidity-nan",
            ExtraChannel::spot_color(&full, [0.0, 0.0, 0.0, f32::NAN]),
        ),
        (
            "spot-infinite",
            ExtraChannel::spot_color(&full, [f32::INFINITY, 0.0, 0.0, 1.0]),
        ),
        (
            "spot-f16-overflow",
            ExtraChannel::spot_color(&full, [65536.0, 0.0, 0.0, 1.0]),
        ),
    ];
    let mut journal = std::fs::File::create(out.join("invalid.tsv")).unwrap();
    writeln!(journal, "policy\tpath\tcase\tresult").unwrap();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        let lossless = LosslessConfig::new()
            .with_effort(1)
            .with_threads(1)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        let lossy = LossyConfig::new(1.0)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode);
        for is_lossless in [true, false] {
            for (name, extra) in &cases {
                for writer in [false, true] {
                    let extras = [extra.clone()];
                    let request = if is_lossless {
                        lossless.encode_request(32, 17, PixelLayout::Rgb8)
                    } else {
                        lossy.encode_request(32, 17, PixelLayout::Rgb8)
                    }
                    .with_extra_channels(&extras);
                    let label = format!("{policy}-{is_lossless}-{name}-{writer}");
                    let mut destination = b"existing destination".to_vec();
                    let before = destination.clone();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if writer {
                            request
                                .encode_to(&pixels, &mut destination)
                                .map(|_| Vec::new())
                        } else {
                            request.encode(&pixels)
                        }
                    }));
                    let status = match &result {
                        Ok(Err(error)) => format!("error {error}"),
                        Ok(Ok(bytes)) => {
                            if !writer {
                                std::fs::write(out.join(format!("{label}.jxl")), bytes).unwrap();
                            }
                            failures.push(format!("{label}: accepted invalid metadata"));
                            "accepted".to_owned()
                        }
                        Err(_) => {
                            failures.push(format!("{label}: panicked"));
                            "panic".to_owned()
                        }
                    };
                    if destination != before {
                        failures.push(format!("{label}: changed destination"));
                    }
                    writeln!(
                        journal,
                        "{policy}\t{is_lossless}/{writer}\t{name}\t{status}"
                    )
                    .unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
    std::fs::write(out.join("failures.txt"), failures.join("\n")).unwrap();
    assert!(
        failures.is_empty(),
        "{}; evidence {}",
        failures.join("\n"),
        out.display()
    );
}

#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/pfm.rs"]
mod pfm;

#[test]
fn extra_metadata_legal_boundaries_render_in_both_decoders() {
    let out = directory();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            let pixels = vec![0_u8; width * height * 3];
            let full = vec![0_u8; width * height];
            let shift3 = vec![0_u8; width.div_ceil(8) * height.div_ceil(8)];
            let cases = [
                ("shift0", ExtraChannel::depth(&full)),
                ("shift3", ExtraChannel::depth(&shift3).with_dim_shift(3)),
                ("cfa0", ExtraChannel::cfa(&full, 0)),
                ("cfa274", ExtraChannel::cfa(&full, 274)),
                (
                    "spot0",
                    ExtraChannel::spot_color(&full, [-2.0, 0.5, 2.0, 0.0]),
                ),
                (
                    "spot1",
                    ExtraChannel::spot_color(&full, [-2.0, 0.5, 2.0, 1.0]),
                ),
            ];
            let lossless = LosslessConfig::new()
                .with_effort(1)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let lossy = LossyConfig::new(1.0)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for is_lossless in [true, false] {
                for (name, extra) in &cases {
                    let extras = [extra.clone()];
                    let request = if is_lossless {
                        lossless.encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                    } else {
                        lossy.encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                    }
                    .with_extra_channels(&extras);
                    let label = format!("{policy}-{width}x{height}-{is_lossless}-{name}");
                    if !is_lossless && *name == "shift3" {
                        assert!(
                            request.encode(&pixels).is_err(),
                            "existing shifted VarDCT limitation"
                        );
                        continue;
                    }
                    let bytes = request.encode(&pixels).unwrap();
                    let stride = width * 3 + 5;
                    let mut padded = vec![0xA5_u8; stride * height];
                    for row in padded.chunks_exact_mut(stride) {
                        row[..width * 3].fill(0);
                    }
                    let strided = if is_lossless {
                        lossless.encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                    } else {
                        lossy.encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                    }
                    .with_extra_channels(&extras)
                    .with_row_stride(stride)
                    .encode(&padded)
                    .unwrap();
                    assert_eq!(strided, bytes, "{label}: packed/strided byte identity");
                    let path = out.join(format!("{label}.jxl"));
                    std::fs::write(&path, &bytes).unwrap();
                    let (color, extras) = decoded_extras::decode_current_planes(&bytes, &label);
                    assert_eq!(color.len(), width * height * 3);
                    assert_eq!(extras.len(), 1);
                    assert_eq!(
                        extras[0],
                        vec![0.0; width * height],
                        "{label}: exact auxiliary samples"
                    );
                    let native = out.join(format!("{label}.pfm"));
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&path)
                        .arg(&native)
                        .args(["--output_extra_channels", "--num_threads=1"])
                        .output()
                        .unwrap();
                    std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr)
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{label}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    let rgb = pfm::read_pfm_samples(&native, (width, height));
                    assert_eq!(rgb.len(), width * height * 3);
                    assert!(rgb.iter().chain(&color).all(|sample| sample.is_finite()));
                    let extra = pfm::read_pfm_samples(
                        &out.join(format!("{label}.pfm-ec1.pfm")),
                        (width, height),
                    );
                    assert_eq!(
                        extra,
                        vec![0.0; width * height],
                        "{label}: exact native auxiliary samples"
                    );
                }
            }
        }
    }
}

#[cfg(all(feature = "__internals", feature = "__pre_quantized"))]
#[test]
fn extra_metadata_prepared_entries_validate_before_working_pixels() {
    use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
    let linear = vec![0.25; 16 * 17 * 3];
    let extra = vec![0_u8; 16 * 17];
    let invalid = [ExtraChannel::spot_color(&[], [0.0, 0.0, 0.0, 2.0])];
    let out = directory();
    for (policy, strategy, mode) in encoder_policies::policies() {
        let mut encoder = VarDctEncoder::new(1.0);
        encoder.profile = LossyConfig::new(1.0)
            .with_effort(3)
            .with_strategy(strategy)
            .with_mode(mode)
            .resolved_profile();
        encoder.force_strategy = Some(0);
        encoder.enable_patches = false;
        encoder.enable_gaborish = false;
        encoder.epf_level_override = Some(0);
        encoder.effort = 3;
        encoder.encoder_mode = mode;
        let pre = EncoderPrecomputed::compute(
            16,
            17,
            &linear,
            encoder.distance,
            encoder.cfl_enabled,
            encoder.ac_strategy_enabled,
            encoder.pixel_domain_loss,
            false,
            false,
            false,
            Some(0),
            &encoder.profile,
            None,
        )
        .unwrap();
        let mut qf = vec![1; pre.xsize_blocks * pre.ysize_blocks];
        let params = DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
        let transform = encoder
            .transform_and_quantize_for_test(&pre, &mut qf, &params)
            .unwrap();
        let valid = [ExtraChannel::cfa(&extra, 274)];
        for (entry, bytes) in [
            (
                "direct",
                encoder
                    .encode_with_extras(16, 17, &linear, &valid)
                    .map(|v| v.data),
            ),
            (
                "prepared",
                encoder.encode_from_precomputed_with_extras(&pre, &qf, &valid),
            ),
            (
                "prequantized",
                encoder.encode_from_pre_quantized_ac_with_extras(
                    &pre,
                    &qf,
                    &transform.quant_dc,
                    &transform.quant_ac,
                    &transform.nzeros,
                    &transform.raw_nzeros,
                    &valid,
                ),
            ),
        ] {
            let bytes = bytes.unwrap();
            let label = format!("{policy}-{entry}");
            let (color, planes) = decoded_extras::decode_current_planes(&bytes, &label);
            assert_eq!(color.len(), 16 * 17 * 3);
            assert_eq!(planes, vec![vec![0.0; 16 * 17]]);
            let path = out.join(format!("{label}.jxl"));
            std::fs::write(&path, &bytes).unwrap();
            let output = out.join(format!("{label}.pfm"));
            let decoded = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(path)
                .arg(&output)
                .args(["--output_extra_channels", "--num_threads=1"])
                .output()
                .unwrap();
            std::fs::write(out.join(format!("{label}.decode.log")), &decoded.stderr).unwrap();
            assert!(
                decoded.status.success(),
                "{label}: {}",
                String::from_utf8_lossy(&decoded.stderr)
            );
            let rgb = pfm::read_pfm_samples(&output, (16, 17));
            assert_eq!(rgb.len(), 16 * 17 * 3);
            assert!(rgb.iter().chain(&color).all(|v| v.is_finite()));
            assert_eq!(
                pfm::read_pfm_samples(&out.join(format!("{label}.pfm-ec1.pfm")), (16, 17)),
                vec![0.0; 16 * 17]
            );
        }
        // Invalid metadata takes precedence over the wrong extra sample count
        // and missing working pixel/coefficient inputs. A downstream header
        // check alone would not establish this entry-boundary order.
        for entry in 0..3 {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match entry {
                0 => encoder
                    .encode_with_extras(16, 17, &[], &invalid)
                    .map(|v| v.data),
                1 => encoder.encode_from_precomputed_with_extras(&pre, &[], &invalid),
                _ => encoder.encode_from_pre_quantized_ac_with_extras(
                    &pre,
                    &[],
                    &Default::default(),
                    &Default::default(),
                    &Default::default(),
                    &Default::default(),
                    &invalid,
                ),
            }));
            let error = result
                .unwrap_or_else(|_| {
                    panic!("{mode:?} entry {entry}: accessed working inputs before metadata")
                })
                .expect_err("invalid solidity");
            assert!(
                error.to_string().contains("solidity"),
                "{mode:?} entry {entry}: {error}"
            );
        }
    }
}

#[test]
fn extra_metadata_public_admission_precedes_memory_preflight() {
    let pixels = vec![0_u8; 32 * 17 * 3];
    let full = vec![0_u8; 32 * 17];
    let extras = [ExtraChannel::cfa(&full, 275)];
    let limits = jxl_encoder::Limits::default().with_max_memory_bytes(1);
    for (policy, strategy, mode) in encoder_policies::policies() {
        let lossless = LosslessConfig::new()
            .with_effort(1)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        let lossy = LossyConfig::new(1.0)
            .with_effort(3)
            .with_strategy(strategy)
            .with_mode(mode);
        for lossless_path in [true, false] {
            let request = if lossless_path {
                lossless.encode_request(32, 17, PixelLayout::Rgb8)
            } else {
                lossy.encode_request(32, 17, PixelLayout::Rgb8)
            };
            let error = request
                .with_extra_channels(&extras)
                .with_limits(&limits)
                .encode(&pixels)
                .unwrap_err();
            assert!(
                error.to_string().contains("CFA"),
                "{policy} {lossless_path}: {error}"
            );
        }
    }
}
