//! Reject unrepresentable manual geometry while preserving legal rounded points.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, LossyConfig, PixelLayout, Spline, SplinePoint};
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
fn spline(points: Vec<SplinePoint>) -> Spline {
    let mut color_dct = [[0.0; 32]; 3];
    color_dct[1][0] = 0.03;
    let mut sigma_dct = [0.0; 32];
    sigma_dct[0] = 1.5;
    Spline {
        control_points: points,
        color_dct,
        sigma_dct,
    }
}
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("spline-admission-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn invalid(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let pixels = [96_u8, 128, 160].repeat(32 * 17);
    let mut padded = vec![0xd1_u8; (32 * 3 + 7) * 17];
    for (row, source) in padded
        .as_chunks_mut::<{ 32 * 3 + 7 }>()
        .0
        .iter_mut()
        .zip(pixels.as_chunks::<{ 32 * 3 }>().0)
    {
        row[..32 * 3].copy_from_slice(source);
    }
    let mut bad_color = spline(vec![SplinePoint::new(3.0, 7.0)]);
    bad_color.color_dct[0][2] = f32::NAN;
    let mut bad_sigma = spline(vec![SplinePoint::new(3.0, 7.0)]);
    bad_sigma.sigma_dct[1] = f32::INFINITY;
    let mut wide_color = spline(vec![SplinePoint::new(3.0, 7.0)]);
    wide_color.color_dct[0][0] = f32::MAX;
    let mut wide_sigma = spline(vec![SplinePoint::new(3.0, 7.0)]);
    wide_sigma.sigma_dct[0] = f32::MAX;
    let bad = [
        ("quantized-color-overflow", wide_color),
        ("quantized-sigma-overflow", wide_sigma),
        ("negative-first", spline(vec![SplinePoint::new(-1.0, 7.0)])),
        (
            "unsigned-first-overflow",
            spline(vec![SplinePoint::new(4_294_967_296.0, 7.0)]),
        ),
        ("empty-points", spline(vec![])),
        (
            "same-rounded",
            spline(vec![SplinePoint::new(3.1, 7.1), SplinePoint::new(3.4, 7.4)]),
        ),
        (
            "same-exact",
            spline(vec![SplinePoint::new(3.0, 7.0), SplinePoint::new(3.0, 7.0)]),
        ),
        (
            "nonfinite-point",
            spline(vec![SplinePoint::new(f32::NAN, 7.0)]),
        ),
        ("nonfinite-color", bad_color),
        ("nonfinite-sigma", bad_sigma),
    ];
    let mut journal = std::fs::File::create(root.join("admission.tsv")).unwrap();
    let mut failures = Vec::new();
    for (name, spline) in bad {
        for (policy, strategy, mode) in &policies {
            let config = LossyConfig::new(1.0)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(*mode)
                .with_splines(vec![spline.clone()]);
            for route in [
                "legacy",
                "request",
                "strided",
                "writer",
                "streaming",
                "animation",
            ] {
                let out = root.join(format!("{name}-{policy}-{route}"));
                std::fs::create_dir(&out).unwrap();
                let saved = b"existing destination".to_vec();
                let mut sink = saved.clone();
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<Vec<u8>, String> {
                        match route {
                            "legacy" => config
                                .encode(&pixels, 32, 17, PixelLayout::Rgb8)
                                .map_err(|e| e.to_string()),
                            "request" => config
                                .encode_request(32, 17, PixelLayout::Rgb8)
                                .encode(&pixels)
                                .map_err(|e| e.to_string()),
                            "strided" => config
                                .encode_request(32, 17, PixelLayout::Rgb8)
                                .with_row_stride(32 * 3 + 7)
                                .encode(&padded)
                                .map_err(|e| e.to_string()),
                            "writer" => config
                                .encode_request(32, 17, PixelLayout::Rgb8)
                                .encode_to(&pixels, &mut sink)
                                .map(|_| sink[saved.len()..].to_vec())
                                .map_err(|e| e.to_string()),
                            "streaming" => {
                                let mut encoder = config
                                    .encoder(32, 17, PixelLayout::Rgb8)
                                    .map_err(|e| e.to_string())?;
                                encoder.push_rows(&pixels, 17).map_err(|e| e.to_string())?;
                                encoder.finish().map_err(|e| e.to_string())
                            }
                            "animation" => config
                                .encode_animation(
                                    32,
                                    17,
                                    PixelLayout::Rgb8,
                                    &AnimationParams::default(),
                                    &[AnimationFrame::new(&pixels, 10)],
                                )
                                .map_err(|e| e.to_string()),
                            _ => unreachable!(),
                        }
                    },
                ));
                let verdict = match outcome {
                    Ok(Err(error)) => {
                        std::fs::write(out.join("error.txt"), error).unwrap();
                        "typed-error"
                    }
                    Ok(Ok(bytes)) => {
                        std::fs::write(out.join("accepted.jxl"), bytes).unwrap();
                        failures.push(format!("{name} {policy} {route}: accepted"));
                        "accepted"
                    }
                    Err(_) => {
                        failures.push(format!("{name} {policy} {route}: panic"));
                        "panic"
                    }
                };
                if sink != saved {
                    failures.push(format!("{name} {policy} {route}: destination changed"));
                }
                writeln!(journal, "{name}\t{policy}\t{route}\t{verdict}").unwrap();
                journal.flush().unwrap();
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        root.display()
    );
}
fn valid(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let source = image::load_from_memory(include_bytes!("images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let pixels = image::imageops::crop_imm(&source, 0, 0, 32, 17).to_image();
    for (policy, strategy, mode) in policies {
        let config = LossyConfig::new(1.0)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode);
        assert_eq!(
            config
                .clone()
                .with_splines(vec![])
                .encode(pixels.as_raw(), 32, 17, PixelLayout::Rgb8)
                .unwrap(),
            config
                .clone()
                .with_auto_splines(false)
                .encode(pixels.as_raw(), 32, 17, PixelLayout::Rgb8)
                .unwrap(),
            "outer empty list remains a legal disable"
        );
        for (name, points) in [
            ("one-point", vec![SplinePoint::new(3.0, 7.0)]),
            (
                "signed-start-delta",
                vec![SplinePoint::new(20.0, 7.0), SplinePoint::new(23.0, 7.0)],
            ),
            (
                "round-boundary",
                vec![SplinePoint::new(3.49, 7.49), SplinePoint::new(3.51, 7.49)],
            ),
        ] {
            let out = root.join(format!("{name}-{policy}"));
            std::fs::create_dir(&out).unwrap();
            let mut splines = vec![spline(points)];
            if name == "signed-start-delta" {
                splines.push(spline(vec![
                    SplinePoint::new(3.0, 7.0),
                    SplinePoint::new(6.0, 7.0),
                ]));
            }
            let config = config.clone().with_splines(splines);
            let bytes = config
                .encode(pixels.as_raw(), 32, 17, PixelLayout::Rgb8)
                .unwrap();
            let path = out.join("input.jxl");
            std::fs::write(&path, &bytes).unwrap();
            if policy == "libjxl" && name != "signed-start-delta" {
                use sha2::{Digest, Sha256};
                let expected = if name == "one-point" {
                    "b6b1edadeac8d8af51706c0642a698c6dc72fbcee4e02a95cc8493b0078739d1"
                } else {
                    "e3b9e79748abdb0ea040058cd749f6e4b50a8236ab5eb3d9463ae73f1b293710"
                };
                assert_eq!(
                    Sha256::digest(&bytes)
                        .iter()
                        .map(|v| format!("{v:02x}"))
                        .collect::<String>(),
                    expected,
                    "unchanged legal strict stream retained before the admission fix"
                );
            }
            assert_eq!(
                bytes,
                config
                    .encode_request(32, 17, PixelLayout::Rgb8)
                    .encode(pixels.as_raw())
                    .unwrap()
            );
            let parsed = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap();
            assert_eq!(parsed.num_loaded_frames(), 1);
            assert!(
                parsed.frame(0).unwrap().header().flags.splines(),
                "an actual emitted spline section is required"
            );
            let primary = decoded_frames::decode_current_frames(&bytes, name);
            assert_eq!(primary.len(), 1);
            let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(&path)
                .arg(out.join("native.png"))
                .args(["--num_threads=1", "--bits_per_sample=16"])
                .output()
                .unwrap();
            std::fs::write(
                out.join("native.log"),
                [&native.stdout[..], &native.stderr[..]].concat(),
            )
            .unwrap();
            assert!(
                native.status.success(),
                "{}",
                String::from_utf8_lossy(&native.stderr)
            );
            let decoded = image::open(out.join("native.png")).unwrap().to_rgb16();
            assert_eq!(decoded.dimensions(), (32, 17));
            assert_eq!(primary[0].0.len(), decoded.as_raw().len());
            assert!(
                primary[0]
                    .0
                    .iter()
                    .zip(decoded.as_raw())
                    .all(|(a, b)| (a.clamp(0.0, 1.0) - f32::from(*b) / 65535.0).abs() < 1e-3)
            );
        }
    }
}
#[test]
fn strict_manual_spline_admission() {
    invalid(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_manual_spline_admission() {
    invalid(encoder_policies::policies());
}
#[test]
fn strict_valid_manual_splines_render() {
    valid(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_valid_manual_splines_render() {
    valid(encoder_policies::policies());
}

#[test]
fn strict_animation_manual_splines_refuse_unsupported_payload() {
    let root = directory();
    let pixels = [96_u8, 128, 160].repeat(32 * 17);
    let config = LossyConfig::new(1.0)
        .with_effort(3)
        .with_threads(1)
        .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
        .with_splines(vec![spline(vec![
            SplinePoint::new(3.0, 7.0),
            SplinePoint::new(20.0, 7.0),
        ])]);
    let frames = [AnimationFrame::new(&pixels, 10)];
    let empty = config
        .clone()
        .with_splines(vec![])
        .encode_request(32, 17, PixelLayout::Rgb8)
        .encode_animation(&AnimationParams::default(), &frames)
        .unwrap();
    let plain = LossyConfig::new(1.0)
        .with_effort(3)
        .with_threads(1)
        .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
        .encode_animation(
            32,
            17,
            PixelLayout::Rgb8,
            &AnimationParams::default(),
            &frames,
        )
        .unwrap();
    assert_eq!(empty, plain, "empty outer spline list preserves animation");
    std::fs::write(root.join("spline-free.jxl"), &empty).unwrap();
    let parsed = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&empty))
        .unwrap();
    assert!(!parsed.frame(0).unwrap().header().flags.splines());
    decoded_frames::decode_current_frames(&empty, "empty-list animation positive");
    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(root.join("spline-free.jxl"))
        .arg(root.join("spline-free.png"))
        .args(["--num_threads=1", "--bits_per_sample=16"])
        .output()
        .unwrap();
    std::fs::write(root.join("spline-free.log"), native.stderr).unwrap();
    assert!(native.status.success());
    let result = config
        .encode_request(32, 17, PixelLayout::Rgb8)
        .encode_animation(&AnimationParams::default(), &frames);
    if let Ok(bytes) = &result {
        std::fs::write(root.join("ignored-animation.jxl"), bytes).unwrap();
        let parsed = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(bytes))
            .unwrap();
        std::fs::write(
            root.join("flags.txt"),
            format!(
                "splines={}\n",
                parsed.frame(0).unwrap().header().flags.splines()
            ),
        )
        .unwrap();
        decoded_frames::decode_current_frames(bytes, "manual animation spline before");
    }
    let error = result.expect_err(
        "manual splines must be explicitly refused by the currently unsupported animation writer",
    );
    assert!(
        matches!(
            error.error(),
            jxl_encoder::EncodeError::InvalidConfig { .. }
        ),
        "{error}"
    );
    assert!(error.to_string().contains("splines"), "{error}");
}

#[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
#[test]
fn strict_prepared_manual_splines_refuse_unsupported_payload() {
    use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
    use jxl_encoder::api::ExtraChannel;
    let root = directory();
    let (width, height) = (32_usize, 17_usize);
    let linear = [0.2_f32, 0.4, 0.6].repeat(width * height);
    let alpha = vec![127_u8; width * height];
    let extras = [ExtraChannel::from_alpha_buf(&alpha, false)];
    let mut failures = Vec::new();
    for mode in [
        jxl_encoder::EncoderMode::Reference,
        jxl_encoder::EncoderMode::Experimental,
    ] {
        let mut encoder = VarDctEncoder::new(1.0);
        encoder.profile = LossyConfig::new(1.0)
            .with_effort(4)
            .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
            .resolved_profile();
        encoder.encoder_mode = mode;
        encoder.effort = 4;
        encoder.enable_patches = false;
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
            encoder.color_encoding.as_ref(),
        )
        .unwrap();
        let mut qf = vec![1; pre.xsize_blocks * pre.ysize_blocks];
        let params = DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
        let transformed = encoder
            .transform_and_quantize_for_test(&pre, &mut qf, &params)
            .unwrap();
        for shape in ["none", "empty-list", "manual", "empty-points"] {
            encoder.splines = match shape {
                "none" => None,
                "empty-list" => Some(vec![]),
                "manual" => Some(vec![spline(vec![
                    SplinePoint::new(3.0, 7.0),
                    SplinePoint::new(20.0, 7.0),
                ])]),
                "empty-points" => Some(vec![spline(vec![])]),
                _ => unreachable!(),
            };
            for entry in ["precomputed", "prequantized"] {
                for alpha in [false, true] {
                    let selected = if alpha { &extras[..] } else { &[] };
                    let result = if entry == "precomputed" {
                        encoder.encode_from_precomputed_with_extras(&pre, &qf, selected)
                    } else {
                        encoder.encode_from_pre_quantized_ac_with_extras(
                            &pre,
                            &qf,
                            &transformed.quant_dc,
                            &transformed.quant_ac,
                            &transformed.nzeros,
                            &transformed.raw_nzeros,
                            selected,
                        )
                    };
                    let out = root.join(format!("{mode:?}-{shape}-{entry}-alpha-{alpha}"));
                    std::fs::create_dir(&out).unwrap();
                    if matches!(shape, "none" | "empty-list") {
                        let bytes = result.unwrap();
                        std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                        decoded_frames::decode_current_frames(
                            &bytes,
                            "prepared spline-free positive",
                        );
                        let native =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(out.join("input.jxl"))
                                .arg(out.join("native.png"))
                                .args(["--num_threads=1", "--bits_per_sample=16"])
                                .output()
                                .unwrap();
                        std::fs::write(out.join("native.log"), native.stderr).unwrap();
                        assert!(native.status.success());
                    } else {
                        match result {
                            Err(error) => assert!(error.to_string().contains("spline"), "{error}"),
                            Ok(bytes) => {
                                std::fs::write(out.join("ignored.jxl"), &bytes).unwrap();
                                let parsed = jxl_oxide::JxlImage::builder()
                                    .read(std::io::Cursor::new(&bytes))
                                    .unwrap();
                                std::fs::write(
                                    out.join("flags.txt"),
                                    format!(
                                        "splines={}\n",
                                        parsed.frame(0).unwrap().header().flags.splines()
                                    ),
                                )
                                .unwrap();
                                failures.push(format!(
                                    "{mode:?} {shape} {entry} alpha={alpha}: ignored"
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        root.display()
    );
}

#[path = "support/pfm.rs"]
mod pfm;

#[test]
fn all_offered_spline_precision_and_group_edges() {
    use jxl_encoder::ProgressiveMode;
    let oracle = std::path::PathBuf::from(
        std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
            .expect("explicit pinned v0.12 original-profile float oracle"),
    );
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout).contains("12000"));
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let mut journal = std::fs::File::create(root.join("precision.tsv")).unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17), (32, 259)] {
            let source =
                image::imageops::crop_imm(&fixture, 0, 0, width as u32, height as u32).to_image();
            for layout in [
                PixelLayout::Rgb8,
                PixelLayout::Rgba16,
                PixelLayout::RgbLinearF32,
                PixelLayout::GrayAlphaLinearF16,
            ] {
                let mut pixels = Vec::new();
                let mut alpha_codes = Vec::new();
                let mut alpha_words = Vec::new();
                for (i, pixel) in source.pixels().enumerate() {
                    let rgb = pixel.0;
                    match layout {
                        PixelLayout::Rgb8 => pixels.extend(rgb),
                        PixelLayout::Rgba16 => {
                            for v in rgb {
                                pixels.extend((u16::from(v) * 257).to_ne_bytes());
                            }
                            let alpha = ((i * 197 + 101) % 65536) as u16;
                            pixels.extend(alpha.to_ne_bytes());
                            alpha_codes.push(alpha);
                        }
                        PixelLayout::RgbLinearF32 => {
                            for v in rgb {
                                pixels.extend((f32::from(v) / 255.0).to_ne_bytes());
                            }
                        }
                        PixelLayout::GrayAlphaLinearF16 => {
                            // Explicit normal binary16 grey and alpha codes;
                            // the decoder outputs their exact binary32 values.
                            let gray = 0x3000_u16 + u16::from(rgb[1]) * 4;
                            let (alpha, value) = match i % 3 {
                                0 => (0x3400_u16, 0.25_f32),
                                1 => (0x3800, 0.5),
                                _ => (0x3a00, 0.75),
                            };
                            pixels.extend(gray.to_ne_bytes());
                            pixels.extend(alpha.to_ne_bytes());
                            alpha_words.push(value.to_bits());
                        }
                        _ => unreachable!(),
                    }
                }
                let scenarios = [
                    (1, ProgressiveMode::Single),
                    (3, ProgressiveMode::Single),
                    (4, ProgressiveMode::QuantizedAcFullAc),
                    (7, ProgressiveMode::DcVlfLfAc),
                ];
                #[cfg(feature = "butteraugli-loop")]
                let scenarios = scenarios
                    .into_iter()
                    .chain([(8, ProgressiveMode::QuantizedAcFullAc)]);
                for (effort, progressive) in scenarios {
                    let points = if width > 256 {
                        vec![SplinePoint::new(252.0, 7.0), SplinePoint::new(260.0, 7.0)]
                    } else if height > 256 {
                        vec![SplinePoint::new(7.0, 252.0), SplinePoint::new(7.0, 260.0)]
                    } else {
                        vec![SplinePoint::new(3.49, 7.49), SplinePoint::new(3.51, 7.49)]
                    };
                    let config = LossyConfig::new(1.0)
                        .with_threads(1)
                        .with_effort(effort)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_patches(false)
                        .with_progressive(progressive)
                        .with_splines(vec![spline(points)]);
                    #[cfg(feature = "butteraugli-loop")]
                    let config = config.with_butteraugli_iters(u32::from(effort >= 8));
                    let row = width * layout.bytes_per_pixel();
                    let stride = row + 7;
                    let mut padded = vec![0xd1; stride * height];
                    for (a, b) in padded
                        .chunks_exact_mut(stride)
                        .zip(pixels.chunks_exact(row))
                    {
                        a[..row].copy_from_slice(b);
                    }
                    let request = || config.encode_request(width as u32, height as u32, layout);
                    #[cfg(feature = "butteraugli-loop")]
                    let trace = jxl_encoder::LoopTrace::new();
                    let first = request();
                    #[cfg(feature = "butteraugli-loop")]
                    let first = first.with_loop_trace(&trace);
                    let bytes = first.encode(&pixels).unwrap();
                    #[cfg(feature = "butteraugli-loop")]
                    if effort >= 8 {
                        let frames = trace.take();
                        assert!(
                            !frames.is_empty(),
                            "manual spline encode must actually run the requested loop"
                        );
                        assert!(frames.iter().all(|frame| frame.score.is_finite()));
                    }
                    assert_eq!(
                        config
                            .encode(&pixels, width as u32, height as u32, layout)
                            .unwrap(),
                        bytes
                    );
                    assert_eq!(
                        request().with_row_stride(stride).encode(&padded).unwrap(),
                        bytes
                    );
                    let mut destination = Vec::new();
                    request().encode_to(&pixels, &mut destination).unwrap();
                    assert_eq!(destination, bytes);
                    let mut stream = config.encoder(width as u32, height as u32, layout).unwrap();
                    for rows in pixels.chunks(row * 7) {
                        stream.push_rows(rows, (rows.len() / row) as u32).unwrap();
                    }
                    assert_eq!(stream.finish().unwrap(), bytes);
                    let out = root.join(format!(
                        "{policy}-{width}x{height}-{layout:?}-e{effort}-{progressive:?}"
                    ));
                    std::fs::create_dir(&out).unwrap();
                    let path = out.join("input.jxl");
                    std::fs::write(&path, &bytes).unwrap();
                    let parsed = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    assert!(
                        parsed.frame(0).unwrap().header().flags.splines(),
                        "actual emitted spline section"
                    );
                    let primary = decoded_frames::decode_current_frames(&bytes, &policy);
                    assert_eq!(primary.len(), 1);
                    let native = std::process::Command::new(&oracle)
                        .arg(&path)
                        .arg(out.join("native"))
                        .arg("--source-fields-float")
                        .output()
                        .unwrap();
                    std::fs::write(
                        out.join("native.log"),
                        [native.stdout, native.stderr].concat(),
                    )
                    .unwrap();
                    assert!(native.status.success());
                    let native_color =
                        pfm::read_pfm_samples(&out.join("native-0.pfm"), (width, height));
                    let colors = if layout.is_grayscale() { 1 } else { 3 };
                    let channels = primary[0].0.len() / (width * height);
                    assert!(
                        channels == colors || channels == colors + usize::from(layout.has_alpha())
                    );
                    let actual_color: Vec<f32> = primary[0]
                        .0
                        .chunks_exact(channels)
                        .flat_map(|p| p[..colors].to_vec())
                        .collect();
                    assert_eq!(actual_color.len(), native_color.len());
                    assert!(native_color.iter().all(|v| v.is_finite()));
                    let error = actual_color
                        .iter()
                        .zip(native_color)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0_f32, f32::max);
                    assert!(
                        error < 1e-3,
                        "{policy} {layout:?} primary/native color {error}"
                    );
                    if layout.has_alpha() {
                        let native_alpha =
                            pfm::read_pfm_samples(&out.join("native-0-ec0.pfm"), (width, height));
                        let mut planes = Vec::new();
                        if channels > colors {
                            planes.push(
                                primary[0]
                                    .0
                                    .chunks_exact(channels)
                                    .map(|p| p[colors])
                                    .collect::<Vec<_>>(),
                            );
                        }
                        planes.extend(primary[0].1.clone());
                        assert!(!planes.is_empty());
                        planes.push(native_alpha);
                        for alpha in planes {
                            assert_eq!(alpha.len(), width * height);
                            assert!(alpha.iter().all(|v| v.is_finite()));
                            if !alpha_codes.is_empty() {
                                assert_eq!(
                                    alpha
                                        .iter()
                                        .map(|v| (*v * 65535.0).round() as u16)
                                        .collect::<Vec<_>>(),
                                    alpha_codes
                                );
                            } else {
                                assert_eq!(
                                    alpha.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    alpha_words
                                );
                            }
                        }
                    }
                    writeln!(journal,"{policy}\t{width}x{height}\t{layout:?}\te{effort}\t{progressive:?}\tcolor_max={error}\talpha_exact=true\tentry_bytes_equal=true").unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
}

#[test]
fn strict_spline_geometry_refused_on_first_push() {
    let cases = [
        (
            "negative-first",
            vec![spline(vec![SplinePoint::new(-1.0, 7.0)])],
        ),
        (
            "unsigned-first-overflow",
            vec![spline(vec![SplinePoint::new(4_294_967_296.0, 7.0)])],
        ),
        (
            "coordinate-range",
            vec![spline(vec![SplinePoint::new(f32::MAX, 7.0)])],
        ),
        (
            "double-delta-word",
            vec![spline(vec![
                SplinePoint::new(0.0, 0.0),
                SplinePoint::new(4_294_967_296.0, 0.0),
            ])],
        ),
        (
            "coordinate-subtraction",
            vec![spline(vec![
                SplinePoint::new(4_294_967_040.0, 0.0),
                SplinePoint::new(i64::MIN as f32, 0.0),
            ])],
        ),
        (
            "starting-delta-word",
            vec![
                spline(vec![SplinePoint::new(0.0, 0.0)]),
                spline(vec![SplinePoint::new(4_294_967_040.0, 0.0)]),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for (label, splines) in cases {
        let config = LossyConfig::new(1.0)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
            .with_splines(splines);
        let mut encoder = config.encoder(32, 17, PixelLayout::Rgb8).unwrap();
        match encoder.push_rows(&[96_u8, 128, 160].repeat(32), 1) {
            Err(error) => assert!(error.to_string().contains("spline"), "{label}: {error}"),
            Ok(()) => failures.push(label),
        }
        // Do not render the intentionally enormous invalid geometry. This
        // production admission check must fail on the first input row.
    }
    assert!(
        failures.is_empty(),
        "accepted first rows for invalid geometry: {failures:?}"
    );
}

#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
#[test]
fn subsampled_manual_spline_requests_refuse_unsupported_payload() {
    use jxl_encoder::ChromaSubsampling;
    let root = directory();
    let pixels = [96_u8, 128, 160].repeat(32 * 17);
    for (policy, strategy, mode) in encoder_policies::policies() {
        for subsampling in [
            ChromaSubsampling::Sub440,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub420,
        ] {
            let config = LossyConfig::new(1.0)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_chroma_subsampling(subsampling);
            let absent = config.encode(&pixels, 32, 17, PixelLayout::Rgb8).unwrap();
            let empty = config
                .clone()
                .with_splines(vec![])
                .encode(&pixels, 32, 17, PixelLayout::Rgb8)
                .unwrap();
            assert_eq!(empty, absent);
            let parsed = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&empty))
                .unwrap();
            assert!(!parsed.frame(0).unwrap().header().flags.splines());
            decoded_frames::decode_current_frames(&empty, "subsampled spline-free positive");
            let out = root.join(format!("{policy}-{subsampling:?}"));
            std::fs::create_dir(&out).unwrap();
            std::fs::write(out.join("input.jxl"), &empty).unwrap();
            let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(out.join("input.jxl"))
                .arg(out.join("native.png"))
                .args(["--num_threads=1", "--bits_per_sample=16"])
                .output()
                .unwrap();
            std::fs::write(out.join("native.log"), native.stderr).unwrap();
            assert!(native.status.success());
            let config = config.with_splines(vec![spline(vec![
                SplinePoint::new(3.0, 7.0),
                SplinePoint::new(20.0, 7.0),
            ])]);
            let mut destination = b"existing destination".to_vec();
            let saved = destination.clone();
            let result = config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .encode_to(&pixels, &mut destination);
            match result {
                Err(error) => assert!(error.to_string().contains("splines"), "{error}"),
                Ok(_) => {
                    std::fs::write(out.join("ignored.jxl"), &destination[saved.len()..]).unwrap();
                    panic!("{policy} {subsampling:?}: manual spline silently omitted");
                }
            }
            assert_eq!(destination, saved);
        }
    }
}
