//! Selected public and expert callers for numeric source descriptions.
use super::numeric::cases;
use super::*;
use jxl_encoder::{CIExy, CustomPrimaries, EncodeError, LosslessConfig};

fn is_color_admission(error: &jxl_encoder::At<EncodeError>, selector: bool) -> bool {
    match error.error() {
        EncodeError::InvalidInput { message } | EncodeError::InvalidConfig { message } => {
            if selector {
                message.to_ascii_lowercase().contains("icc")
            } else {
                message.contains("color encoding")
            }
        }
        _ => false,
    }
}

#[test]
fn streaming_color_admission_checks_first_rows_and_changed_encoding() {
    let root = directory();
    let mut failures = Vec::new();
    let mut journal = std::fs::File::create(root.join("streaming.tsv")).unwrap();
    let mut inputs = cases();
    let mut selector = ColorEncoding::srgb();
    selector.want_icc = true;
    inputs.push(("icc-selector-no-payload", selector, PixelLayout::Rgb8));
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (label, encoding, layout) in &inputs {
            let channels = if *layout == PixelLayout::Gray8 { 1 } else { 3 };
            let row = vec![64; 16 * channels];
            let all = row.repeat(17);
            for state in 0..3 {
                for lossy in [false, true] {
                    let mut dest = b"existing-output".to_vec();
                    let limits = jxl_encoder::Limits::default().with_max_memory_bytes(1);
                    macro_rules! stream {
                        ($cfg:expr) => {{
                            let mut encoder = $cfg.encoder(16, 17, *layout).unwrap();
                            if state == 0 {
                                encoder = encoder.with_color_encoding(encoding.clone());
                                let result = encoder.push_rows(&row, 1);
                                if result.is_err() {
                                    assert_eq!(encoder.rows_pushed(), 0);
                                }
                                result
                            } else if state == 1 {
                                encoder.push_rows(&row, 1).unwrap();
                                encoder = encoder.with_color_encoding(encoding.clone());
                                let result = encoder.push_rows(&row, 1);
                                if result.is_err() {
                                    assert_eq!(encoder.rows_pushed(), 1);
                                }
                                result
                            } else {
                                encoder.push_rows(&all, 17).unwrap();
                                encoder = encoder
                                    .with_color_encoding(encoding.clone())
                                    .with_limits(&limits);
                                encoder.finish_into(&mut dest).map(|_| ())
                            }
                        }};
                    }
                    let result = if lossy {
                        stream!(
                            LossyConfig::new(1.0)
                                .with_effort(1)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                        )
                    } else {
                        stream!(
                            LosslessConfig::new()
                                .with_effort(1)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                        )
                    };
                    let label = format!("{policy}-{label}-state{state}-lossy{lossy}");
                    writeln!(journal, "{label}\t{result:?}").unwrap();
                    journal.flush().unwrap();
                    match result {
                        Err(error) if is_color_admission(&error, encoding.want_icc) => {}
                        other => failures.push(format!("{label}: {other:?}")),
                    }
                    assert_eq!(dest, b"existing-output");
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}

#[test]
fn color_admission_request_entries_and_policies() {
    let root = directory();
    let mut failures = Vec::new();
    let mut journal = std::fs::File::create(root.join("requests.tsv")).unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (label, encoding, layout) in cases() {
            let channels = if layout == PixelLayout::Gray8 { 1 } else { 3 };
            let row = vec![64; 16 * channels];
            let pixels = row.repeat(17);
            let stride = row.len() + 7;
            let mut padded = vec![0xCD; stride * 17];
            for target in padded.chunks_exact_mut(stride) {
                target[..row.len()].copy_from_slice(&row);
            }
            for entry in 0..6 {
                for lossy in [false, true] {
                    let mut dest = b"existing-output".to_vec();
                    let limits = jxl_encoder::Limits::default().with_max_memory_bytes(1);
                    let frames = [jxl_encoder::AnimationFrame::new(&pixels, 10)];
                    macro_rules! request {
                        ($cfg:expr) => {{
                            let cfg = $cfg;
                            let request = || {
                                cfg.encode_request(16, 17, layout)
                                    .with_color_encoding(encoding.clone())
                            };
                            match entry {
                                0 => request().encode(&pixels).map(|_| ()),
                                1 => request()
                                    .with_row_stride(stride)
                                    .encode(&padded)
                                    .map(|_| ()),
                                2 => request().encode_into(&pixels, &mut dest).map(|_| ()),
                                3 => request().encode_to(&pixels, &mut dest).map(|_| ()),
                                4 => request()
                                    .encode_animation(
                                        &jxl_encoder::AnimationParams::default(),
                                        &frames,
                                    )
                                    .map(|_| ()),
                                _ => request().with_limits(&limits).encode(&pixels).map(|_| ()),
                            }
                        }};
                    }
                    let result = if lossy {
                        request!(
                            LossyConfig::new(1.0)
                                .with_effort(1)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                        )
                    } else {
                        request!(
                            LosslessConfig::new()
                                .with_effort(1)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                        )
                    };
                    let name = format!("{policy}-{label}-entry{entry}-lossy{lossy}");
                    writeln!(journal, "{name}\t{result:?}").unwrap();
                    journal.flush().unwrap();
                    match result {
                        Err(error) if is_color_admission(&error, false) => {}
                        other => failures.push(format!("{name}: {other:?}")),
                    }
                    assert_eq!(dest, b"existing-output");
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}

#[test]
fn color_admission_icc_flag_keeps_source_conversion() {
    use jxl_encoder::ImageMetadata;
    let root = directory();
    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let metadata = ImageMetadata::new().with_icc_profile(profile);
    let pixels: Vec<u8> = [0.5_f32, 0.2, 0.1]
        .repeat(32 * 17)
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let config = LossyConfig::new(0.01)
        .with_effort(3)
        .with_threads(1)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_force_strategy(Some(0))
        .with_gaborish(false)
        .with_epf_level(0)
        .with_patches(false);
    for label in ["p3-d65", "p3-dci"] {
        let mut outputs = Vec::new();
        for want_icc in [false, true] {
            let mut ce = source_colors::encoding(label);
            ce.want_icc = want_icc;
            let bytes = config
                .encode_request(32, 17, PixelLayout::RgbLinearF32)
                .with_color_encoding(ce)
                .with_metadata(&metadata)
                .encode(&pixels)
                .unwrap();
            let out = root.join(format!("{label}-{want_icc}"));
            std::fs::create_dir(&out).unwrap();
            render_case(&out, &bytes, 32, 17, false);
            let decoded = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(out.join("output.jxl"))
                .arg(out.join("profile.pfm"))
                .arg(format!(
                    "--orig_icc_out={}",
                    out.join("original.icc").display()
                ))
                .arg("--num_threads=1")
                .output()
                .unwrap();
            std::fs::write(out.join("profile.log"), &decoded.stderr).unwrap();
            assert!(decoded.status.success());
            assert_eq!(std::fs::read(out.join("original.icc")).unwrap(), profile);
            outputs.push(bytes);
        }
        assert_eq!(
            outputs[0], outputs[1],
            "{label}: ICC wire selector changed source conversion"
        );
    }
}

#[test]
fn color_admission_xyz_public_paths_and_policies() {
    check_xyz_public_paths(false);
}

#[cfg(feature = "corpus-tests")]
#[test]
fn color_admission_xyz_animation_paths_and_policies() {
    check_xyz_public_paths(true);
}

fn check_xyz_public_paths(animation: bool) {
    use jxl_encoder::{AnimationFrame, AnimationParams, WhitePoint};
    let root = directory();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for d50 in [false, true] {
            let mut ce = ColorEncoding::with_custom_primaries(CustomPrimaries {
                red: CIExy::new(1., 0.),
                green: CIExy::new(0., 1.),
                blue: CIExy::new(0., 0.),
            });
            ce.transfer_function = jxl_encoder::TransferFunction::Linear;
            if d50 {
                ce.white_point = WhitePoint::Custom;
                ce.custom_white_point = Some(CIExy::new(0.3457, 0.3585));
            }
            let key = if d50 {
                "xyz-basis-d50"
            } else {
                "xyz-basis-d65"
            };
            let native: Vec<f32> = include_str!("../../src/vardct/color_admission_v012.tsv")
                .lines()
                .find(|line| line.starts_with(&format!("pixel\t{key}\t")))
                .unwrap()
                .split('\t')
                .skip(2)
                .map(|v| v.parse().unwrap())
                .collect();
            for (width, height) in [(32_usize, 17_usize), (259, 17)] {
                let pixels: Vec<u8> = [0.5_f32, 0.2, 0.1]
                    .repeat(width * height)
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect();
                let config = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_force_strategy(Some(0))
                    .with_gaborish(false)
                    .with_epf_level(0)
                    .with_patches(false);
                let packed = config
                    .encode_request(width as u32, height as u32, PixelLayout::RgbLinearF32)
                    .with_color_encoding(ce.clone())
                    .encode(&pixels)
                    .unwrap();
                let entries: &[&str] = if animation {
                    &["animation"]
                } else {
                    &["packed", "strided", "streaming"]
                };
                for &entry in entries {
                    let bytes = match entry {
                        "packed" => packed.clone(),
                        "strided" => {
                            let row = width * 12;
                            let mut padded = vec![0xFF; (row + 7) * height];
                            for (src, dst) in pixels
                                .chunks_exact(row)
                                .zip(padded.chunks_exact_mut(row + 7))
                            {
                                dst[..row].copy_from_slice(src);
                            }
                            config
                                .encode_request(
                                    width as u32,
                                    height as u32,
                                    PixelLayout::RgbLinearF32,
                                )
                                .with_color_encoding(ce.clone())
                                .with_row_stride(row + 7)
                                .encode(&padded)
                                .unwrap()
                        }
                        "streaming" => {
                            let mut stream = config
                                .encoder(width as u32, height as u32, PixelLayout::RgbLinearF32)
                                .unwrap()
                                .with_color_encoding(ce.clone());
                            for rows in pixels.chunks(width * 12 * 7) {
                                stream
                                    .push_rows(rows, (rows.len() / (width * 12)) as u32)
                                    .unwrap();
                            }
                            stream.finish().unwrap()
                        }
                        _ => config
                            .encode_request(width as u32, height as u32, PixelLayout::RgbLinearF32)
                            .with_color_encoding(ce.clone())
                            .encode_animation(
                                &AnimationParams::default(),
                                &[
                                    AnimationFrame::new(&pixels, 10),
                                    AnimationFrame::new(&pixels, 20),
                                ],
                            )
                            .unwrap(),
                    };
                    if matches!(entry, "strided" | "streaming") {
                        assert_eq!(bytes, packed, "{policy} {key} {entry}");
                    }
                    let out = root.join(format!("{policy}-{key}-{width}-{entry}"));
                    std::fs::create_dir(&out).unwrap();
                    for (decoder, (frames, expected)) in ["primary", "native"].into_iter().zip(
                        render_case(&out, &bytes, width, height, entry == "animation")
                            .iter()
                            .zip([
                                &[0.5_f32, 0.2, 0.1][..],
                                if entry == "animation" {
                                    &[0.5_f32, 0.2, 0.1][..]
                                } else {
                                    native.as_slice()
                                },
                            ]),
                    ) {
                        for frame in frames {
                            assert_eq!(frame.len(), width * height * 3);
                            assert!(frame.iter().all(|v| v.is_finite()));
                            let error = frame
                                .iter()
                                .zip(expected.iter().cycle())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            assert!(
                                error < 0.002,
                                "{} {decoder} max_error={error}",
                                out.display()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(
    feature = "__internals",
    feature = "__pre_quantized",
    feature = "rate-control"
))]
#[test]
fn color_admission_prepared_engine_entries() {
    use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
    use jxl_encoder::EncoderMode;
    use jxl_encoder::vardct::RateControlConfig;
    let linear = [0.25_f32, 0.125, 0.0625].repeat(16 * 17);
    for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
        let mut encoder = VarDctEncoder::new(0.01);
        encoder.profile = LossyConfig::new(0.01).with_effort(3).resolved_profile();
        encoder.effort = 3;
        encoder.encoder_mode = mode;
        encoder.force_strategy = Some(0);
        encoder.enable_patches = false;
        encoder.enable_gaborish = false;
        encoder.epf_level_override = Some(0);
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
        // Absence is a supported input policy too: exercise each direct
        // guard's None arm with actual prepared and rate-controlled output.
        let root = directory();
        for (entry, result) in [
            (
                "direct-none",
                encoder
                    .encode(16, 17, &linear, None)
                    .map(|output| output.data),
            ),
            ("prepared-none", encoder.encode_from_precomputed(&pre, &qf)),
            (
                "prequantized-none",
                encoder.encode_from_pre_quantized_ac(
                    &pre,
                    &qf,
                    &transform.quant_dc,
                    &transform.quant_ac,
                    &transform.nzeros,
                    &transform.raw_nzeros,
                ),
            ),
            (
                "rate-none",
                encoder
                    .encode_with_rate_control_config(
                        16,
                        17,
                        &linear,
                        &RateControlConfig {
                            max_iterations: 1,
                            ..Default::default()
                        },
                    )
                    .map(|(bytes, _)| bytes),
            ),
        ] {
            let bytes = result.unwrap();
            let out = root.join(entry);
            std::fs::create_dir(&out).unwrap();
            for (decoder, frames) in render_case(&out, &bytes, 16, 17, false)
                .into_iter()
                .enumerate()
            {
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].len(), linear.len());
                assert!(frames[0].iter().all(|value| value.is_finite()));
                let error = frames[0]
                    .iter()
                    .zip(&linear)
                    .map(|(&sample, expected)| {
                        // The primary returns the declared default sRGB signal;
                        // the native leg explicitly requests linear sRGB.
                        let value = if decoder == 0 {
                            if sample <= 0.04045 {
                                sample / 12.92
                            } else {
                                ((sample + 0.055) / 1.055).powf(2.4)
                            }
                        } else {
                            sample
                        };
                        (value - expected).abs()
                    })
                    .fold(0_f32, f32::max);
                assert!(error < 0.002, "{entry} decoder{decoder}: {error}");
            }
        }
        for (label, ce, _) in cases() {
            encoder.color_encoding = Some(ce.clone());
            let prepared = EncoderPrecomputed::compute(
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
                Some(&ce),
            );
            assert!(
                matches!(EncodeError::from(prepared.err().expect(label)), EncodeError::InvalidInput { message } if message.contains("color encoding")),
                "{label} prepare"
            );
            for (entry, result) in [
                ("direct", encoder.encode(16, 17, &linear, None).map(|_| ())),
                (
                    "alpha",
                    encoder
                        .encode(16, 17, &linear, Some(&vec![128; 16 * 17]))
                        .map(|_| ()),
                ),
                (
                    "extras",
                    encoder.encode_with_extras(16, 17, &linear, &[]).map(|_| ()),
                ),
                (
                    "prepared",
                    encoder.encode_from_precomputed(&pre, &qf).map(|_| ()),
                ),
                (
                    "prequantized",
                    encoder
                        .encode_from_pre_quantized_ac(
                            &pre,
                            &qf,
                            &transform.quant_dc,
                            &transform.quant_ac,
                            &transform.nzeros,
                            &transform.raw_nzeros,
                        )
                        .map(|_| ()),
                ),
                (
                    "rate-zero",
                    encoder
                        .encode_with_rate_control_config(
                            16,
                            17,
                            &linear,
                            &RateControlConfig {
                                max_iterations: 0,
                                ..Default::default()
                            },
                        )
                        .map(|_| ()),
                ),
                (
                    "rate-positive",
                    encoder
                        .encode_with_rate_control_config(
                            16,
                            17,
                            &linear,
                            &RateControlConfig {
                                max_iterations: 1,
                                ..Default::default()
                            },
                        )
                        .map(|_| ()),
                ),
            ] {
                assert!(
                    matches!(EncodeError::from(result.expect_err(label)), EncodeError::InvalidInput { message } if message.contains("color encoding")),
                    "{mode:?} {label} {entry}"
                );
            }
        }
    }
}

#[test]
fn color_admission_gamma_endpoints_render() {
    let root = directory();
    let pixels = vec![255_u8; 32 * 17 * 3];
    for (policy, strategy, mode) in encoder_policies::policies() {
        for gamma in [f32::from_bits(0x38fff518), 1.0] {
            for lossy in [false, true] {
                let bytes = if lossy {
                    LossyConfig::new(0.01)
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .encode_request(32, 17, PixelLayout::Rgb8)
                        .with_color_encoding(ColorEncoding::with_gamma(gamma))
                        .encode(&pixels)
                } else {
                    LosslessConfig::new()
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .encode_request(32, 17, PixelLayout::Rgb8)
                        .with_color_encoding(ColorEncoding::with_gamma(gamma))
                        .encode(&pixels)
                }
                .unwrap();
                let out = root.join(format!("{policy}-{:x}-{lossy}", gamma.to_bits()));
                std::fs::create_dir(&out).unwrap();
                for frames in render_case(&out, &bytes, 32, 17, false) {
                    for frame in frames {
                        assert_eq!(frame.len(), 32 * 17 * 3);
                        assert!(
                            frame
                                .iter()
                                .all(|v| v.is_finite() && (v - 1.0).abs() < 0.002),
                            "{}",
                            out.display()
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(feature = "jpeg-reencoding", feature = "chroma-subsampling"))]
#[test]
fn color_admission_optional_subsampling_entries() {
    use jxl_encoder::ChromaSubsampling;
    let pixels = vec![64; 16 * 17 * 3];
    for mode in [
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub440,
    ] {
        for (policy, strategy, encoder_mode) in encoder_policies::policies() {
            for (label, ce, layout) in cases()
                .into_iter()
                .filter(|(_, _, layout)| *layout == PixelLayout::Rgb8)
            {
                let config = LossyConfig::new(1.0)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(encoder_mode)
                    .with_chroma_subsampling(mode);
                let mut dest = b"existing-output".to_vec();
                let result = config
                    .encode_request(16, 17, layout)
                    .with_color_encoding(ce)
                    .encode_into(&pixels, &mut dest);
                assert!(
                    is_color_admission(&result.expect_err(label), false),
                    "{policy} {mode:?} {label}"
                );
                assert_eq!(dest, b"existing-output");
            }
        }
    }
}

#[test]
fn color_admission_gamma_white_precision_and_streaming() {
    check_gamma_white_paths(false);
}

#[cfg(feature = "corpus-tests")]
#[test]
fn color_admission_gamma_white_animation() {
    check_gamma_white_paths(true);
}

fn check_gamma_white_paths(animation: bool) {
    let root = directory();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for layout in [
            PixelLayout::Rgb8,
            PixelLayout::Rgba8,
            PixelLayout::Bgr8,
            PixelLayout::Bgra8,
            PixelLayout::Rgb16,
            PixelLayout::Rgba16,
            PixelLayout::Gray8,
            PixelLayout::GrayAlpha8,
            PixelLayout::Gray16,
            PixelLayout::GrayAlpha16,
        ] {
            let (width, height) = (259_u32, 17_u32);
            let pixels = vec![255_u8; width as usize * height as usize * layout.bytes_per_pixel()];
            for gamma in [f32::from_bits(0x38fff518), 1.0] {
                let ce = if layout.is_grayscale() {
                    ColorEncoding::gray_with_gamma(gamma)
                } else {
                    ColorEncoding::with_gamma(gamma)
                };
                let config = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let request = config
                    .encode_request(width, height, layout)
                    .with_color_encoding(ce.clone());
                let bytes = if animation {
                    request.encode_animation(
                        &jxl_encoder::AnimationParams::default(),
                        &[
                            jxl_encoder::AnimationFrame::new(&pixels, 10),
                            jxl_encoder::AnimationFrame::new(&pixels, 20),
                        ],
                    )
                } else {
                    request.encode(&pixels)
                }
                .unwrap();
                if !animation {
                    let stride = width as usize * layout.bytes_per_pixel() + 3;
                    let mut padded = vec![0xCD; stride * height as usize];
                    for (src, dst) in pixels
                        .chunks_exact(stride - 3)
                        .zip(padded.chunks_exact_mut(stride))
                    {
                        dst[..stride - 3].copy_from_slice(src);
                    }
                    assert_eq!(
                        config
                            .encode_request(width, height, layout)
                            .with_color_encoding(ce.clone())
                            .with_row_stride(stride)
                            .encode(&padded)
                            .unwrap(),
                        bytes
                    );
                    let mut stream = config
                        .encoder(width, height, layout)
                        .unwrap()
                        .with_color_encoding(ce);
                    let row = width as usize * layout.bytes_per_pixel();
                    for rows in pixels.chunks(row * 7) {
                        stream.push_rows(rows, (rows.len() / row) as u32).unwrap();
                    }
                    assert_eq!(stream.finish().unwrap(), bytes);
                }
                let out = root.join(format!("{policy}-{layout:?}-{:x}", gamma.to_bits()));
                std::fs::create_dir(&out).unwrap();
                let decoded = render_case(&out, &bytes, width as usize, height as usize, animation);
                for (i, frames) in decoded.into_iter().enumerate() {
                    for frame in frames {
                        let expected_channels = if (i == 0 || animation) && layout.is_grayscale() {
                            1
                        } else {
                            3
                        };
                        assert_eq!(
                            frame.len(),
                            width as usize * height as usize * expected_channels
                        );
                        assert!(
                            frame
                                .iter()
                                .all(|v| v.is_finite() && (v - 1.0).abs() < 0.002),
                            "{} decoder{i} first={:?} min={} max={}",
                            out.display(),
                            &frame[..frame.len().min(3)],
                            frame.iter().copied().fold(f32::INFINITY, f32::min),
                            frame.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn source_gamma_streaming_admission_precedes_rows_and_budget() {
    let row = vec![255_u8; 16 * 3];
    let all = row.repeat(17);
    for (policy, strategy, mode) in encoder_policies::policies() {
        for state in 0..3 {
            for lossy in [false, true] {
                macro_rules! check {
                    ($cfg:expr)=>{{
                        let mut stream=$cfg.encoder(16,17,PixelLayout::Rgb8).unwrap();
                        let result=if state==0 {
                            stream=stream.with_source_gamma(0.001);
                            let result=stream.push_rows(&row,1);
                            assert_eq!(stream.rows_pushed(),0,"{policy} first rows lossy={lossy}");
                            result
                        } else if state==1 {
                            stream.push_rows(&row,1).unwrap();
                            stream=stream.with_source_gamma(0.001);
                            let result=stream.push_rows(&row,1);
                            assert_eq!(stream.rows_pushed(),1,"{policy} changed gamma lossy={lossy}");
                            result
                        } else {
                            stream.push_rows(&all,17).unwrap();
                            stream=stream.with_source_gamma(0.001).with_limits(&jxl_encoder::Limits::default().with_max_memory_bytes(1));
                            let mut output=b"existing-output".to_vec();
                            let result=stream.finish_into(&mut output).map(|_|());
                            assert_eq!(output,b"existing-output");
                            result
                        };
                        assert!(matches!(result,Err(ref e) if matches!(e.error(),EncodeError::InvalidInput{message} if message.contains("source_gamma"))),
                            "{policy} state{state} lossy={lossy}: {result:?}");
                    }};
                }
                if lossy {
                    check!(
                        LossyConfig::new(1.0)
                            .with_effort(1)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                    );
                } else {
                    check!(
                        LosslessConfig::new()
                            .with_effort(1)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                    );
                }
            }
        }
    }
}

#[cfg(feature = "corpus-tests")]
#[test]
fn color_admission_animation_width_includes_later_and_hidden_frames() {
    use jxl_encoder::{AnimationFrame, AnimationParams, BlendMode};
    let root = directory();
    let mut headers = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        let black = vec![0_u8; 32 * 17 * 3];
        let white = vec![255_u8; 32 * 17 * 3];
        let config = LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        for pattern in ["safe", "late", "early", "hidden"] {
            let frames = match pattern {
                "safe" => vec![
                    AnimationFrame::new(&black, 10),
                    AnimationFrame::new(&black, 20),
                ],
                "late" => vec![
                    AnimationFrame::new(&black, 10),
                    AnimationFrame::new(&white, 20),
                ],
                "early" => vec![
                    AnimationFrame::new(&white, 10),
                    AnimationFrame::new(&black, 20),
                ],
                _ => vec![
                    AnimationFrame::new(&black, 10),
                    AnimationFrame::new(&white, 0)
                        .with_reference_only(true)
                        .with_save_as_reference(2),
                    AnimationFrame::new(&black, 20)
                        .with_blend_mode(BlendMode::Add)
                        .with_blend_source(2),
                ],
            };
            let bytes = config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .with_color_encoding(ColorEncoding::with_gamma(1.0))
                .encode_animation(&AnimationParams::default(), &frames)
                .unwrap();
            let out = root.join(format!("{policy}-{pattern}"));
            std::fs::create_dir(&out).unwrap();
            for (decoder, frames) in render_case(&out, &bytes, 32, 17, true)
                .into_iter()
                .enumerate()
            {
                for (index, frame) in frames.into_iter().enumerate() {
                    let expected = if pattern == "early" {
                        f32::from(index == 0)
                    } else {
                        f32::from(index == 1 && pattern != "safe")
                    };
                    assert_eq!(frame.len(), 32 * 17 * 3);
                    let error = frame
                        .iter()
                        .map(|v| (v - expected).abs())
                        .fold(0_f32, f32::max);
                    assert!(
                        error < 0.002,
                        "{} decoder{decoder} frame{index} error{error}",
                        out.display()
                    );
                }
            }
            headers.push(serde_json::json!({"path":out.join("output.jxl"),"case":pattern,"policy":policy,"narrow":if pattern=="safe"{Some(true)}else{None}}));
        }
    }
    // Level selection consumes actual header precision, including reduced
    // integer precision and floating source/alpha descriptors.
    for (layout, bits, lossless, narrow) in [
        (PixelLayout::Rgb16, Some(12), true, true),
        (PixelLayout::Rgb16, Some(13), true, false),
        (PixelLayout::Rgb16, Some(13), false, true),
        (PixelLayout::Rgba16, Some(13), false, false),
        (PixelLayout::RgbLinearF16, None, false, false),
        (PixelLayout::RgbaLinearF32, None, false, false),
    ] {
        let pixels = vec![0_u8; 32 * 17 * layout.bytes_per_pixel()];
        let frames = [
            AnimationFrame::new(&pixels, 10),
            AnimationFrame::new(&pixels, 20),
        ];
        let bytes = if lossless {
            let config = LosslessConfig::new().with_effort(1).with_threads(1);
            let request = config.encode_request(32, 17, layout);
            let request = if let Some(bits) = bits {
                request.with_bits_per_sample(bits)
            } else {
                request
            };
            request.encode_animation(&AnimationParams::default(), &frames)
        } else {
            let config = LossyConfig::new(1.0).with_effort(3).with_threads(1);
            let request = config.encode_request(32, 17, layout);
            let request = if let Some(bits) = bits {
                request.with_bits_per_sample(bits)
            } else {
                request
            };
            request.encode_animation(&AnimationParams::default(), &frames)
        }
        .unwrap();
        let out = root.join(format!("precision-{layout:?}-{bits:?}-{lossless}"));
        std::fs::create_dir(&out).unwrap();
        let _ = render_case(&out, &bytes, 32, 17, true);
        headers.push(
            serde_json::json!({"path":out.join("output.jxl"),"case":"precision","narrow":narrow}),
        );
    }
    let manifest = root.join("headers.json");
    std::fs::write(&manifest, serde_json::to_vec_pretty(&headers).unwrap()).unwrap();
    let result = std::process::Command::new("python3")
        .arg(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../scripts/check_animation_width_headers.py"),
        )
        .arg(&manifest)
        .arg(root.join("observed-headers.json"))
        .output()
        .unwrap();
    std::fs::write(
        root.join("header-check.log"),
        [result.stdout, result.stderr].concat(),
    )
    .unwrap();
    assert!(
        result.status.success(),
        "{} header controls",
        root.display()
    );
}
