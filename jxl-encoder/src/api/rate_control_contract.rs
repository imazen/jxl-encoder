//! Approved request/controller integration and source-bound refusal contracts.
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
fn directory() -> std::path::PathBuf {
    let base = crate::test_helpers::output_dir("request-rate-control-contract");
    std::fs::create_dir_all(&base).unwrap();
    (0..)
        .find_map(|i| {
            let p = base.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn control(iterations: usize) -> crate::vardct::RateControlConfig {
    crate::vardct::RateControlConfig {
        max_iterations: iterations,
        ..Default::default()
    }
}
#[test]
fn request_rate_control_preserves_layout_alpha_policy_and_rows() {
    let root = directory();
    let mut journal = std::fs::File::create(root.join("cases.tsv")).unwrap();
    use std::io::Write;
    for (name, strategy, mode) in policies::policies() {
        for (layout, samples, alpha) in [
            (PixelLayout::Rgb8, vec![128_u16, 64, 32], None),
            (
                PixelLayout::Rgba8,
                vec![128, 64, 32, 137],
                Some(137_f32 / 255.0),
            ),
            (PixelLayout::Gray8, vec![128], None),
            (
                PixelLayout::GrayAlpha8,
                vec![128, 137],
                Some(137_f32 / 255.0),
            ),
            (PixelLayout::Rgb16, vec![32768, 16384, 8192], None),
            (
                PixelLayout::Rgba16,
                vec![32768, 16384, 8192, 12345],
                Some(12345_f32 / 65535.0),
            ),
            (PixelLayout::Gray16, vec![32768], None),
            (
                PixelLayout::GrayAlpha16,
                vec![32768, 12345],
                Some(12345_f32 / 65535.0),
            ),
        ] {
            for iterations in [0, 2] {
                let out = root.join(format!("{name}-{layout:?}-i{iterations}"));
                std::fs::create_dir(&out).unwrap();
                let pixel: Vec<u8> = samples
                    .iter()
                    .flat_map(|v| {
                        if layout.is_16bit() {
                            v.to_ne_bytes().to_vec()
                        } else {
                            vec![*v as u8]
                        }
                    })
                    .collect();
                let (width, height) = (32_usize, 17_usize);
                let data = pixel.repeat(width * height);
                let cfg = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_gaborish(false)
                    .with_epf_level(0)
                    .with_patches(false)
                    .with_force_strategy(Some(0));
                let bytes = cfg
                    .encode_request(width as u32, height as u32, layout)
                    .with_rate_control(control(iterations))
                    .encode(&data)
                    .unwrap();
                let stride = width * pixel.len() + 11;
                let mut padded = vec![0xab; stride * height];
                for (row, source) in padded
                    .chunks_mut(stride)
                    .zip(data.chunks(width * pixel.len()))
                {
                    row[..source.len()].copy_from_slice(source);
                }
                let packed = cfg
                    .encode_request(width as u32, height as u32, layout)
                    .with_rate_control(control(iterations))
                    .with_row_stride(stride)
                    .encode(&padded)
                    .unwrap();
                assert_eq!(bytes, packed, "{out:?} strided");
                let mut writer = Vec::new();
                cfg.encode_request(width as u32, height as u32, layout)
                    .with_rate_control(control(iterations))
                    .encode_to(&data, &mut writer)
                    .unwrap();
                assert_eq!(bytes, writer, "{out:?} writer");
                let input = out.join("input.jxl");
                std::fs::write(&input, &bytes).unwrap();
                let primary = decoded_frames::decode_current_frames(&bytes, &name);
                assert_eq!(primary.len(), 1);
                let channels = if layout.is_grayscale() { 1 } else { 3 };
                assert_eq!(primary[0].0.len(), width * height * channels);
                let expected: Vec<f32> = samples[..channels]
                    .iter()
                    .map(|v| *v as f32 / if layout.is_16bit() { 65535.0 } else { 255.0 })
                    .collect();
                let native = std::process::Command::new(crate::test_helpers::djxl_path())
                    .arg(&input)
                    .arg(out.join("native.pfm"))
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(out.join("native.log"), &native.stderr).unwrap();
                assert!(native.status.success(), "{out:?}");
                let native = pfm::read_pfm_samples(&out.join("native.pfm"), (width, height));
                assert_eq!(native.len(), primary[0].0.len());
                for actual in [&primary[0].0, &native] {
                    assert!(actual.iter().all(|v| v.is_finite()));
                    let maximum = actual
                        .iter()
                        .zip(expected.iter().cycle())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0_f32, f32::max);
                    assert!(maximum <= 0.005, "{out:?} source error {maximum}");
                }
                match alpha {
                    Some(expected) => {
                        assert_eq!(primary[0].1.len(), 1);
                        assert_eq!(primary[0].1[0].len(), width * height);
                        let maximum = if layout.is_16bit() { 65535.0 } else { 255.0 };
                        let code = (expected * maximum).round() as u16;
                        assert!(
                            primary[0].1[0]
                                .iter()
                                .all(|v| (*v * maximum).round() as u16 == code),
                            "{out:?} primary alpha"
                        );
                        let native_alpha =
                            std::process::Command::new(crate::test_helpers::djxl_path())
                                .arg(&input)
                                .arg(out.join("native-alpha.png"))
                                .args(["--bits_per_sample=16", "--num_threads=1"])
                                .output()
                                .unwrap();
                        std::fs::write(out.join("native-alpha.log"), &native_alpha.stderr).unwrap();
                        assert!(native_alpha.status.success(), "{out:?}");
                        let image = image::open(out.join("native-alpha.png"))
                            .unwrap()
                            .to_rgba16();
                        assert_eq!(image.dimensions(), (width as u32, height as u32));
                        let expected_code = if layout.is_16bit() { code } else { code * 257 };
                        assert!(
                            image.pixels().all(|p| p.0[3] == expected_code),
                            "{out:?} native alpha"
                        );
                    }
                    None => assert!(primary[0].1.is_empty()),
                }
                writeln!(journal, "{name}\t{layout:?}\ti{iterations}\tcomplete").unwrap();
                journal.flush().unwrap();
            }
        }
    }
}
#[test]
fn request_rate_control_retains_gamma_and_refuses_profile_mismatch() {
    let root = directory();
    let cfg = LossyConfig::new(0.01)
        .with_effort(3)
        .with_gaborish(false)
        .with_epf_level(0)
        .with_patches(false);
    let data = vec![128_u8; 32 * 17 * 3];
    for iterations in [0, 2] {
        let bytes = cfg
            .encode_request(32, 17, PixelLayout::Rgb8)
            .with_source_gamma(0.5)
            .with_rate_control(control(iterations))
            .encode(&data)
            .unwrap();
        std::fs::write(root.join(format!("gamma-i{iterations}.jxl")), &bytes).unwrap();
        let parsed = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap();
        assert!(format!("{:?}", parsed.image_header().metadata.colour_encoding).contains("Gamma"));
        let frames = decoded_frames::decode_current_frames(&bytes, "gamma request RC");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0.len(), 32 * 17 * 3);
        let native = std::process::Command::new(crate::test_helpers::djxl_path())
            .arg(root.join(format!("gamma-i{iterations}.jxl")))
            .arg(root.join(format!("gamma-i{iterations}.pfm")))
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(
            root.join(format!("gamma-i{iterations}.log")),
            &native.stderr,
        )
        .unwrap();
        assert!(native.status.success());
        let native =
            pfm::read_pfm_samples(&root.join(format!("gamma-i{iterations}.pfm")), (32, 17));
        assert_eq!(native.len(), frames[0].0.len());
        let difference = native
            .iter()
            .zip(&frames[0].0)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        assert!(difference <= 0.001, "gamma i{iterations}: {difference}");
    }
    let profile: Vec<u8> = include_str!("../../tests/fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|s| {
            (0..s.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        })
        .collect();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let mut writer = b"existing destination".to_vec();
    assert!(
        cfg.encode_request(32, 17, PixelLayout::Rgb8)
            .with_metadata(&metadata)
            .with_rate_control(control(0))
            .encode_to(&data, &mut writer)
            .is_err()
    );
    assert_eq!(writer, b"existing destination");
}
#[test]
fn request_rate_control_validates_limits_and_unsupported_units() {
    let cfg = LossyConfig::new(1.0).with_effort(1);
    let data = vec![128_u8; 32 * 17 * 3];
    let limits = Limits::new().with_max_quant_loop_iters(1);
    assert!(
        cfg.encode_request(32, 17, PixelLayout::Rgb8)
            .with_limits(&limits)
            .with_rate_control(control(2))
            .encode(&data)
            .is_err()
    );
    assert!(
        cfg.encode_request(32, 17, PixelLayout::Rgb8)
            .with_intensity_target(1000.0)
            .with_rate_control(control(0))
            .encode(&data)
            .is_err()
    );
    let cfg = cfg.with_resampling(2);
    assert!(
        cfg.encode_request(32, 17, PixelLayout::Rgb8)
            .with_rate_control(control(0))
            .encode(&data)
            .is_err()
    );
}

#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
#[test]
fn request_rate_control_refuses_subsampling_that_bypasses_its_controller() {
    let pixels = [64_u8, 96, 128].repeat(32 * 17);
    for subsampling in [
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub440,
    ] {
        for iterations in [0, 2] {
            let cfg = LossyConfig::new(1.0)
                .with_effort(3)
                .with_chroma_subsampling(subsampling);
            let mut writer = b"prior destination".to_vec();
            let result = cfg
                .encode_request(32, 17, PixelLayout::Rgb8)
                .with_rate_control(control(iterations))
                .encode_to(&pixels, &mut writer);
            assert!(
                matches!(result, Err(ref e) if matches!(e.error(), EncodeError::InvalidConfig { message } if message.contains("subsampling"))),
                "{subsampling:?} iterations{iterations}: {result:?}"
            );
            assert_eq!(writer, b"prior destination");
        }
    }
}

#[test]
fn request_rate_control_keeps_icc_boxes_orientation_and_requested_depth() {
    use std::io::Write;
    let root = directory();
    let (oracle, _) = crate::test_helpers::native_libjxl_oracle(
        "LIBJXL_ORIGINAL_PROFILE_ORACLE",
        "original-profile",
    );
    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let exif = b"II\x2a\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let xmp = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'/>";
    let metadata = ImageMetadata::new()
        .with_icc_profile(profile)
        .with_exif(exif)
        .with_xmp(xmp)
        .with_orientation(crate::Orientation::FlipH);
    let (width, height) = (32_usize, 17_usize);
    let pixels = [64_u8, 96, 128].repeat(width * height);
    let depth: Vec<u16> = (0..width * height)
        .map(|i| (i * 193 % 65536) as u16)
        .collect();
    let extras = [ExtraChannel::depth_u16(&depth)];
    let mut journal = std::fs::File::create(root.join("metadata.tsv")).unwrap();
    for (policy, strategy, mode) in policies::policies() {
        for iterations in [0, 2] {
            let out = root.join(format!("{policy}-i{iterations}"));
            std::fs::create_dir(&out).unwrap();
            let cfg = LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_patches(false);
            let bytes = cfg
                .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                .with_metadata(&metadata)
                .with_extra_channels(&extras)
                .with_rate_control(control(iterations))
                .encode(&pixels)
                .unwrap();
            std::fs::write(out.join("input.jxl"), &bytes).unwrap();
            let parsed = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap();
            assert_eq!(parsed.original_icc().unwrap(), profile);
            assert_eq!(parsed.image_header().metadata.orientation, 2);
            let mut offset = 0;
            let mut found_exif = false;
            let mut found_xmp = false;
            while offset < bytes.len() {
                let length =
                    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                assert!(length >= 8 && offset + length <= bytes.len());
                let payload = &bytes[offset + 8..offset + length];
                match &bytes[offset + 4..offset + 8] {
                    b"Exif" => {
                        assert_eq!(&payload[..4], &[0; 4]);
                        assert_eq!(&payload[4..], exif);
                        found_exif = true;
                    }
                    b"xml " => {
                        assert_eq!(payload, xmp);
                        found_xmp = true;
                    }
                    _ => {}
                }
                offset += length;
            }
            assert!(found_exif && found_xmp);
            let primary = decoded_frames::decode_current_frames(&bytes, "request RC profile/depth");
            assert_eq!(primary.len(), 1);
            assert_eq!(primary[0].0.len(), width * height * 3);
            assert_eq!(primary[0].1.len(), 1);
            // Native orientation reverses the sample rows in both observers.
            let expected: Vec<u16> = depth
                .chunks(width)
                .flat_map(|row| row.iter().rev().copied())
                .collect();
            assert_eq!(primary[0].1[0].len(), expected.len());
            assert!(
                primary[0].1[0]
                    .iter()
                    .zip(&expected)
                    .all(|(value, code)| (*value * 65535.0).round() as u16 == *code)
            );
            let native = std::process::Command::new(&oracle)
                .arg(out.join("input.jxl"))
                .arg(out.join("native"))
                .arg("--original-profile-float")
                .output()
                .unwrap();
            std::fs::write(
                out.join("native.log"),
                [native.stdout, native.stderr].concat(),
            )
            .unwrap();
            assert!(native.status.success());
            let native_color = pfm::read_pfm_samples(&out.join("native-0.pfm"), (width, height));
            let native_depth =
                pfm::read_pfm_samples(&out.join("native-0-ec0.pfm"), (width, height));
            assert_eq!(native_color.len(), primary[0].0.len());
            assert_eq!(native_depth.len(), expected.len());
            assert!(
                native_depth
                    .iter()
                    .zip(&expected)
                    .all(|(value, code)| (*value * 65535.0).round() as u16 == *code)
            );
            assert!(native_color.iter().all(|v| v.is_finite()));
            // jxl 0.8 cannot output arbitrary ICC: its float default is linear
            // sRGB. Observe native output in that same domain, while retaining
            // the original-profile float/depth decode independently above.
            for (name, color) in [
                ("linear", "RGB_D65_SRG_Rel_Lin"),
                ("signal", "RGB_D65_SRG_Rel_SRG"),
            ] {
                let result = std::process::Command::new(crate::test_helpers::djxl_path())
                    .arg(out.join("input.jxl"))
                    .arg(out.join(format!("{name}.pfm")))
                    .args(["--num_threads=1", "--color_space", color])
                    .output()
                    .unwrap();
                std::fs::write(out.join(format!("{name}.log")), &result.stderr).unwrap();
                assert!(result.status.success());
            }
            let native_linear = pfm::read_pfm_samples(&out.join("linear.pfm"), (width, height));
            let native_signal = pfm::read_pfm_samples(&out.join("signal.pfm"), (width, height));
            assert_eq!(native_linear.len(), primary[0].0.len());
            assert_eq!(native_signal.len(), pixels.len());
            assert!(
                primary[0]
                    .0
                    .iter()
                    .zip(&native_linear)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 0.005)
            );
            assert!(
                native_signal
                    .iter()
                    .zip(&pixels)
                    .all(|(a, b)| a.is_finite() && (*a - f32::from(*b) / 255.0).abs() <= 0.005)
            );
            let source_linear: Vec<f32> = [64.0_f32 / 255.0, 96.0 / 255.0, 128.0 / 255.0]
                .into_iter()
                .map(|v| {
                    if v <= 0.04045 {
                        v / 12.92
                    } else {
                        ((v + 0.055) / 1.055).powf(2.4)
                    }
                })
                .collect();
            for output in [&primary[0].0, &native_linear] {
                assert!(
                    output
                        .iter()
                        .zip(source_linear.iter().cycle())
                        .all(|(a, b)| a.is_finite() && (*a - *b).abs() <= 0.005),
                    "{policy} i{iterations}: physical source color"
                );
            }
            std::fs::write(out.join("color-domains.json"), serde_json::to_vec_pretty(
                &serde_json::json!({"original_icc_first":&native_color[..3],"primary_linear_srgb_first":&primary[0].0[..3],"native_linear_srgb_first":&native_linear[..3],"native_srgb_first":&native_signal[..3]})
            ).unwrap()).unwrap();

            writeln!(journal, "{policy}\ti{iterations}\tcomplete").unwrap();
            journal.flush().unwrap();
        }
    }
}

#[test]
fn request_rate_control_rejects_invalid_bounds_and_incompatible_requests() {
    let cfg = LossyConfig::new(1.0).with_effort(1);
    let pixels = vec![128_u8; 32 * 17 * 3];
    let mut invalid = Vec::new();
    for tolerance in [f32::NAN, f32::INFINITY, -0.001] {
        invalid.push(crate::vardct::RateControlConfig {
            tolerance,
            ..control(0)
        });
    }
    invalid.push(crate::vardct::RateControlConfig {
        qf_min: 0,
        ..control(0)
    });
    invalid.push(crate::vardct::RateControlConfig {
        qf_min: 200,
        qf_max: 199,
        ..control(0)
    });
    for config in invalid {
        let mut destination = b"previous bytes".to_vec();
        let result = cfg
            .encode_request(32, 17, PixelLayout::Rgb8)
            .with_rate_control(config)
            .encode_to(&pixels, &mut destination);
        assert!(
            matches!(result, Err(ref e) if matches!(e.error(), EncodeError::InvalidConfig { .. })),
            "{result:?}"
        );
        assert_eq!(destination, b"previous bytes");
    }
    let lossless = LosslessConfig::new();
    let mut destination = b"previous bytes".to_vec();
    let result = lossless
        .encode_request(32, 17, PixelLayout::Rgb8)
        .with_rate_control(control(0))
        .encode_to(&pixels, &mut destination);
    assert!(
        matches!(result, Err(ref e) if matches!(e.error(), EncodeError::InvalidConfig { message } if message.contains("lossy")))
    );
    assert_eq!(destination, b"previous bytes");
    let trace = crate::LoopTrace::new();
    let result = cfg
        .encode_request(32, 17, PixelLayout::Rgb8)
        .with_loop_trace(&trace)
        .with_rate_control(control(0))
        .encode_to(&pixels, &mut destination);
    assert!(
        matches!(result, Err(ref e) if matches!(e.error(), EncodeError::InvalidConfig { message } if message.contains("traces")))
    );
    assert!(trace.frames().is_empty());
    assert_eq!(destination, b"previous bytes");
    let frames = [AnimationFrame::new(&pixels, 1)];
    let result = cfg
        .encode_request(32, 17, PixelLayout::Rgb8)
        .with_rate_control(control(0))
        .encode_animation(&AnimationParams::default(), &frames);
    assert!(
        matches!(result, Err(ref e) if matches!(e.error(), EncodeError::InvalidConfig { message } if message.contains("animation")))
    );
}

#[test]
fn request_rate_control_observes_late_stop_and_preserves_the_destination() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct CountStop {
        allowed: usize,
        calls: AtomicUsize,
    }
    impl enough::Stop for CountStop {
        fn check(&self) -> core::result::Result<(), enough::StopReason> {
            if self.calls.fetch_add(1, Ordering::Relaxed) >= self.allowed {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }
    let root = directory();
    let image = image::load_from_memory(include_bytes!("../../tests/images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let pixels = image::imageops::crop_imm(&image, 333, 135, 259, 17)
        .to_image()
        .into_raw();
    let cfg = LossyConfig::new(0.5)
        .with_effort(3)
        .with_threads(1)
        .with_patches(false);
    for iterations in [0, 2] {
        let stop = CountStop {
            allowed: usize::MAX,
            calls: AtomicUsize::new(0),
        };
        let baseline = cfg
            .encode_request(259, 17, PixelLayout::Rgb8)
            .with_stop(&stop)
            .with_rate_control(control(iterations))
            .encode(&pixels)
            .unwrap();
        let calls = stop.calls.load(Ordering::Relaxed);
        assert!(calls > 4, "observe actual inner and final checkpoints");
        let out = root.join(format!("i{iterations}"));
        std::fs::create_dir(&out).unwrap();
        std::fs::write(out.join("input.jxl"), &baseline).unwrap();
        let primary = decoded_frames::decode_current_frames(&baseline, "RC cancellation positive");
        assert_eq!(primary.len(), 1);
        assert_eq!(primary[0].0.len(), 259 * 17 * 3);
        let native = std::process::Command::new(crate::test_helpers::djxl_path())
            .arg(out.join("input.jxl"))
            .arg(out.join("native.pfm"))
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(out.join("native.log"), &native.stderr).unwrap();
        assert!(native.status.success());
        let native = pfm::read_pfm_samples(&out.join("native.pfm"), (259, 17));
        assert_eq!(native.len(), primary[0].0.len());
        assert!(
            native
                .iter()
                .zip(&primary[0].0)
                .all(|(a, b)| (a - b).abs() <= 0.001)
        );
        for allowed in 0..calls {
            let stop = CountStop {
                allowed,
                calls: AtomicUsize::new(0),
            };
            let mut destination = b"prior result".to_vec();
            let result = cfg
                .encode_request(259, 17, PixelLayout::Rgb8)
                .with_stop(&stop)
                .with_rate_control(control(iterations))
                .encode_into(&pixels, &mut destination);
            assert!(
                matches!(result,Err(ref e)if matches!(e.error(),EncodeError::Cancelled)),
                "i{iterations} allowed{allowed}: {result:?}"
            );
            assert!(stop.calls.load(Ordering::Relaxed) > allowed);
            assert_eq!(destination, b"prior result");
            let next = cfg
                .encode_request(259, 17, PixelLayout::Rgb8)
                .with_rate_control(control(iterations))
                .encode(&pixels)
                .unwrap();
            assert_eq!(next, baseline, "post-cancellation bytes i{iterations}");
        }
        std::fs::write(
            out.join("checkpoints.txt"),
            format!("successful_checks={calls}\n"),
        )
        .unwrap();
    }
}

#[test]
fn request_rate_control_keeps_swizzled_float_and_explicit_hdr_units() {
    use std::io::Write;
    let root = directory();
    let (oracle, _) = crate::test_helpers::native_libjxl_oracle(
        "LIBJXL_ORIGINAL_PROFILE_ORACLE",
        "original-profile",
    );
    let mut journal = std::fs::File::create(root.join("typed-layouts.tsv")).unwrap();
    for (policy, strategy, mode) in policies::policies() {
        for layout in [
            PixelLayout::Bgr8,
            PixelLayout::Bgra8,
            PixelLayout::RgbLinearF32,
            PixelLayout::RgbaLinearF32,
            PixelLayout::GrayLinearF32,
            PixelLayout::GrayAlphaLinearF32,
            PixelLayout::RgbLinearF16,
            PixelLayout::RgbaLinearF16,
            PixelLayout::GrayLinearF16,
            PixelLayout::GrayAlphaLinearF16,
            PixelLayout::RgbPqF32,
            PixelLayout::RgbaPqF32,
            PixelLayout::RgbHlgF32,
            PixelLayout::RgbaHlgF32,
            PixelLayout::RgbBt709F32,
            PixelLayout::RgbaBt709F32,
        ] {
            let channels = if layout.is_grayscale() { 1 } else { 3 };
            let alpha = layout.has_alpha();
            let swizzled = matches!(layout, PixelLayout::Bgr8 | PixelLayout::Bgra8);
            let expected: Vec<f32> = if swizzled {
                vec![128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0]
            } else {
                vec![0.125, 0.25, 0.5][..channels].to_vec()
            };
            let pixel = if swizzled {
                if alpha {
                    vec![32, 64, 128, 137]
                } else {
                    vec![32, 64, 128]
                }
            } else if layout.is_f16() {
                let mut bits = vec![0x3000_u16, 0x3400, 0x3800][..channels].to_vec();
                if alpha {
                    bits.push(0x3800);
                }
                bits.into_iter()
                    .flat_map(u16::to_ne_bytes)
                    .collect::<Vec<_>>()
            } else {
                let mut values = expected.clone();
                if alpha {
                    values.push(0.5);
                }
                values
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect::<Vec<_>>()
            };
            assert_eq!(pixel.len(), layout.bytes_per_pixel());
            let (width, height) = (9_usize, 3_usize);
            let pixels = pixel.repeat(width * height);
            let stride = width * pixel.len() + 7;
            let mut padded = vec![0xcd_u8; stride * height];
            for (dest, source) in padded
                .chunks_mut(stride)
                .zip(pixels.chunks(width * pixel.len()))
            {
                dest[..source.len()].copy_from_slice(source);
            }
            let cfg = LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_patches(false)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for iterations in [0, 2] {
                let out = root.join(format!("{policy}-{layout:?}-i{iterations}"));
                std::fs::create_dir(&out).unwrap();
                // Prepared XYB is currently defined at 255 nits. HDR default
                // peaks have separate refusal controls; this explicit peak is
                // also the source conversion's requested light basis.
                let request = || {
                    cfg.encode_request(width as u32, height as u32, layout)
                        .with_intensity_target(255.0)
                        .with_rate_control(control(iterations))
                };
                let bytes = request().encode(&pixels).unwrap();
                assert_eq!(
                    request().with_row_stride(stride).encode(&padded).unwrap(),
                    bytes
                );
                let mut writer = Vec::new();
                request().encode_to(&pixels, &mut writer).unwrap();
                assert_eq!(writer, bytes);
                std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                let primary = decoded_frames::decode_current_frames(&bytes, "RC typed layout");
                assert_eq!(primary.len(), 1);
                assert_eq!(primary[0].0.len(), width * height * channels);
                let native = std::process::Command::new(&oracle)
                    .arg(out.join("input.jxl"))
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
                assert_eq!(native_color.len(), primary[0].0.len());
                std::fs::write(out.join("source-observations.json"), serde_json::to_vec_pretty(
                    &serde_json::json!({"primary_first":&primary[0].0[..channels],"native_first":&native_color[..channels],"source":expected})
                ).unwrap()).unwrap();
                for actual in [&primary[0].0, &native_color] {
                    assert!(
                        actual
                            .iter()
                            .zip(expected.iter().cycle())
                            .all(|(a, b)| (a - b).abs() <= 0.005),
                        "{policy}-{layout:?} i{iterations}: original source fields, first {:?}",
                        &actual[..channels]
                    );
                }
                if alpha {
                    let expected_alpha = if swizzled { 137.0_f32 / 255.0 } else { 0.5 };
                    let native_alpha =
                        pfm::read_pfm_samples(&out.join("native-0-ec0.pfm"), (width, height));
                    assert_eq!(primary[0].1.len(), 1);
                    for actual in [&primary[0].1[0], &native_alpha] {
                        assert_eq!(actual.len(), width * height);
                        if swizzled {
                            assert!(
                                actual.iter().all(|v| (*v * 255.0).round() == 137.0),
                                "{policy}-{layout:?}: exact integer alpha codes"
                            );
                        } else {
                            assert!(
                                actual
                                    .iter()
                                    .all(|v| v.to_bits() == expected_alpha.to_bits()),
                                "{policy}-{layout:?}: exact float alpha bits"
                            );
                        }
                    }
                } else {
                    assert!(primary[0].1.is_empty());
                }
                writeln!(journal, "{policy}\t{layout:?}\ti{iterations}\tcomplete").unwrap();
                journal.flush().unwrap();
            }
        }
    }
}

#[test]
fn request_rate_control_budget_changes_delivered_pixels() {
    let root = directory();
    let pixels: Vec<u8> = (0..32 * 17 * 3).map(|i| ((i * 73) % 251) as u8).collect();
    let mut witnesses = 0;
    for distance in [0.1, 1.0, 4.0] {
        let config = LossyConfig::new(distance)
            .with_effort(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_threads(1);
        let mut baseline = None;
        for iterations in [0, 2] {
            let out = root.join(format!("d{distance}-i{iterations}"));
            std::fs::create_dir(&out).unwrap();
            let mut options = control(iterations);
            options.tolerance = 0.0;
            let bytes = config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .with_rate_control(options)
                .encode(&pixels)
                .unwrap();
            std::fs::write(out.join("input.jxl"), &bytes).unwrap();
            let primary =
                decoded_frames::decode_current_frames(&bytes, "request controller witness");
            assert_eq!(primary.len(), 1);
            assert_eq!(primary[0].0.len(), pixels.len());
            let result = std::process::Command::new(crate::test_helpers::djxl_path())
                .arg(out.join("input.jxl"))
                .arg(out.join("native.pfm"))
                .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_SRG"])
                .output()
                .unwrap();
            std::fs::write(out.join("native.log"), &result.stderr).unwrap();
            assert!(result.status.success());
            let native = pfm::read_pfm_samples(&out.join("native.pfm"), (32, 17));
            assert!(
                primary[0]
                    .0
                    .iter()
                    .zip(&native)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 0.001)
            );
            if let Some((original, original_pixels)) = &baseline {
                let changed = bytes != *original
                    && primary[0]
                        .0
                        .iter()
                        .zip(original_pixels)
                        .any(|(a, b): (&f32, &f32)| a.to_bits() != b.to_bits());
                witnesses += usize::from(changed);
                std::fs::write(
                    out.join("changed.txt"),
                    format!("delivered_pixels_changed={changed}\n"),
                )
                .unwrap();
            } else {
                baseline = Some((bytes, primary[0].0.clone()));
            }
        }
    }
    assert!(
        witnesses > 0,
        "iteration budget must reach delivered pixels on the declared fixture grid"
    );
}

#[cfg(feature = "__external_metric")]
#[test]
fn request_rate_control_refuses_an_external_scorer_before_callback_work() {
    struct NeverCalled;
    impl crate::ExternalLoopMetric for NeverCalled {
        fn set_reference(
            &mut self,
            _: &[f32],
            _: &[f32],
            _: &[f32],
            _: usize,
            _: usize,
        ) -> core::result::Result<(), String> {
            panic!("incompatible scorer must be refused before reference work")
        }
        fn compare(
            &mut self,
            _: &[f32],
            _: &[f32],
            _: &[f32],
            _: usize,
            _: usize,
            _: usize,
            _: &mut Vec<f32>,
        ) -> core::result::Result<f64, String> {
            panic!("incompatible scorer must never be invoked")
        }
    }
    let metric = crate::ExternalMetric::new(NeverCalled);
    let config = LossyConfig::new(1.0).with_effort(1);
    let pixels = vec![128_u8; 32 * 17 * 3];
    let mut destination = b"previous bytes".to_vec();
    let result = config
        .encode_request(32, 17, PixelLayout::Rgb8)
        .with_external_metric(&metric)
        .with_rate_control(control(0))
        .encode_to(&pixels, &mut destination);
    assert!(
        matches!(result, Err(ref e) if matches!(e.error(), EncodeError::InvalidConfig {message} if message.contains("own Butteraugli")))
    );
    assert_eq!(destination, b"previous bytes");
}

#[test]
fn request_rate_control_keeps_cmyk_profile_physical_color_and_exact_extras() {
    let root = directory();
    let (oracle, _) =
        crate::test_helpers::native_libjxl_oracle("CMYK_RAW_FRAME_ORACLE", "CMYK raw-frame");
    let profile: Vec<u8> = include_str!("../../tests/fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect();
    let expected: Vec<f32> = include_str!("../../tests/fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
        .collect();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    for (policy, strategy, mode) in policies::policies() {
        let config = LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode)
            .with_patches(false);
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                let wide = layout.is_16bit();
                let maximum = if wide { 65535.0 } else { 255.0 };
                let scale = if wide { 257_u16 } else { 1 };
                let black_code = 224 * scale + u16::from(wide);
                let codes =
                    [64 * scale, 128 * scale, 192 * scale, black_code].repeat(width * height);
                let pixels: Vec<u8> = codes
                    .iter()
                    .flat_map(|v| {
                        if wide {
                            v.to_ne_bytes().to_vec()
                        } else {
                            vec![*v as u8]
                        }
                    })
                    .collect();
                let alpha = vec![137_u8; width * height];
                let extras = [ExtraChannel::from_alpha_buf(&alpha, false)];
                for with_alpha in [false, true] {
                    for iterations in [0, 2] {
                        let out = root.join(format!(
                            "{policy}-{width}x{height}-{layout:?}-alpha{with_alpha}-i{iterations}"
                        ));
                        std::fs::create_dir(&out).unwrap();
                        let bytes = config
                            .encode_request(width as u32, height as u32, layout)
                            .with_metadata(&metadata)
                            .with_extra_channels(if with_alpha { &extras } else { &[] })
                            .with_rate_control(control(iterations))
                            .encode(&pixels)
                            .unwrap();
                        let input = out.join("input.jxl");
                        std::fs::write(&input, &bytes).unwrap();
                        let parsed = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&bytes))
                            .unwrap();
                        assert_eq!(parsed.original_icc(), Some(profile.as_slice()));
                        let primary = decoded_frames::decode_current_frames(
                            &bytes,
                            "request CMYK controller",
                        );
                        let result = std::process::Command::new(&oracle)
                            .arg(&input)
                            .arg(out.join("native"))
                            .arg("linear")
                            .output()
                            .unwrap();
                        std::fs::write(out.join("native.log"), &result.stderr).unwrap();
                        std::fs::write(out.join("headers.tsv"), &result.stdout).unwrap();
                        assert!(result.status.success(), "{out:?}");
                        let native =
                            pfm::read_pfm_samples(&out.join("native-0.pfm"), (width, height));
                        assert_eq!(primary.len(), 1);
                        assert_eq!(primary[0].0.len(), width * height * 3);
                        assert_eq!(primary[0].1.len(), 1 + usize::from(with_alpha));
                        let black =
                            pfm::read_pfm_samples(&out.join("native-0-black.pfm"), (width, height));
                        for plane in [&primary[0].1[0], &black] {
                            assert_eq!(plane.len(), width * height);
                            assert!(
                                plane.iter().all(|v| v.is_finite()
                                    && (*v * maximum).round() as u16 == black_code)
                            );
                        }
                        for plane in [&primary[0].0, &native] {
                            assert_eq!(plane.len(), width * height * 3);
                            assert!(plane.iter().all(|v| v.is_finite()));
                            let error = plane
                                .iter()
                                .zip(expected.iter().cycle())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            assert!(
                                error <= 0.005,
                                "{out:?}: independent physical color error {error}"
                            );
                        }
                        if with_alpha {
                            let native_alpha = pfm::read_pfm_samples(
                                &out.join("native-0-alpha.pfm"),
                                (width, height),
                            );
                            for plane in [&primary[0].1[1], &native_alpha] {
                                assert_eq!(plane.len(), width * height);
                                assert!(plane.iter().all(|v|v.is_finite()&&(*v*255.0).round()as u8==137));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn request_rate_control_survives_canonicalized_strided_input() {
    let root = directory();
    let (width, height) = (32_usize, 17_usize);
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 73) % 251) as u8)
        .collect();
    let stride = width * 4 + 7;
    let mut rgba = vec![0xa5; stride * height];
    for y in 0..height {
        for x in 0..width {
            let input = (y * width + x) * 3;
            let output = y * stride + x * 4;
            rgba[output..output + 3].copy_from_slice(&rgb[input..input + 3]);
            rgba[output + 3] = 255;
        }
    }
    for distance in [0.1, 1.0, 4.0] {
        for iterations in [0, 2] {
            let config = LossyConfig::new(distance)
                .with_effort(1)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_threads(1);
            let mut options = control(iterations);
            options.tolerance = 0.0;
            let expected = config
                .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                .with_rate_control(options.clone())
                .encode(&rgb)
                .unwrap();
            let canonical = config.clone().with_canonicalize_input(true);
            let actual = canonical
                .encode_request(width as u32, height as u32, PixelLayout::Rgba8)
                .with_row_stride(stride)
                .with_rate_control(options)
                .encode(&rgba)
                .unwrap();
            assert_eq!(actual, expected, "d{distance} i{iterations}");
            let out = root.join(format!("d{distance}-i{iterations}"));
            std::fs::create_dir(&out).unwrap();
            std::fs::write(out.join("input.jxl"), &actual).unwrap();
            let primary = decoded_frames::decode_current_frames(&actual, "canonical controller");
            assert_eq!(primary.len(), 1);
            assert!(primary[0].1.is_empty());
            let result = std::process::Command::new(crate::test_helpers::djxl_path())
                .arg(out.join("input.jxl"))
                .arg(out.join("native.pfm"))
                .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_SRG"])
                .output()
                .unwrap();
            std::fs::write(out.join("native.log"), &result.stderr).unwrap();
            assert!(result.status.success());
            let native = pfm::read_pfm_samples(&out.join("native.pfm"), (width, height));
            assert_eq!(primary[0].0.len(), rgb.len());
            assert!(
                primary[0]
                    .0
                    .iter()
                    .zip(&native)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 0.001)
            );
        }
    }
}

#[test]
fn request_rate_control_precompute_budget_error_propagates() {
    let mut encoder = crate::vardct::encoder::VarDctEncoder::new(1.0);
    encoder.budget = Some(crate::budget::MemoryBudget::new(1));
    let pixels = vec![0.25; 32 * 17 * 3];
    let error = match encoder.encode_request_rate_control(32, 17, &pixels, &[], &control(0), None) {
        Ok(_) => panic!("one-byte precompute budget unexpectedly admitted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("budget"), "{error}");
}
