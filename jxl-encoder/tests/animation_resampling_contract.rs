//! Explicit admission for the unimplemented animation spatial-resampling path.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, LossyConfig, PixelLayout};
#[path = "support/decoded_frames.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("animation-resampling-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|index| {
            let path = root.join(format!("run-{index}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{error}"),
            }
        })
        .unwrap()
}

#[test]
fn animation_spatial_resampling_requires_an_implemented_path() {
    let root = directory();
    let black = vec![0_u8; 33 * 17 * 3];
    let white = vec![255_u8; 33 * 17 * 3];
    let frames = [
        AnimationFrame::new(&black, 10),
        AnimationFrame::new(&white, 20),
    ];
    let mut accepted = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for factor in [0, 2, 4, 8] {
            for already in [false, true] {
                let config = LossyConfig::new(if factor == 0 { 12.0 } else { 1.0 })
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_auto_resampling(true)
                    .with_already_downsampled(already);
                let config = if factor == 0 {
                    config
                } else {
                    config.with_resampling(factor)
                };
                assert!(config.effective_resampling() > 1);
                for legacy in [false, true] {
                    let result = if legacy {
                        config.encode_animation(
                            33,
                            17,
                            PixelLayout::Rgb8,
                            &AnimationParams::default(),
                            &frames,
                        )
                    } else {
                        config
                            .encode_request(33, 17, PixelLayout::Rgb8)
                            .encode_animation(&AnimationParams::default(), &frames)
                    };
                    let label = format!("{policy}-{factor}-{already}-{legacy}");
                    match result {
                        Err(error) => {
                            assert!(error.to_string().contains("resampling"), "{label}: {error}")
                        }
                        Ok(bytes) => {
                            let image = jxl_oxide::JxlImage::builder()
                                .read(std::io::Cursor::new(&bytes))
                                .unwrap();
                            accepted.push(format!("{label}: {}x{}", image.width(), image.height()));
                            std::fs::write(root.join(format!("{label}.jxl")), bytes).unwrap();
                        }
                    }
                }
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "omitted animation spatial resampling: {accepted:?}; {}",
        root.display()
    );
}

#[test]
fn animation_factor_one_retains_packed_integer_frames() {
    let root = directory();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for layout in [PixelLayout::Rgb8, PixelLayout::Rgba16] {
            let (black, white) = if layout == PixelLayout::Rgb8 {
                (vec![0_u8; 33 * 17 * 3], vec![255_u8; 33 * 17 * 3])
            } else {
                (
                    [0_u16, 0, 0, 65535]
                        .repeat(33 * 17)
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect(),
                    [65535_u16; 4]
                        .repeat(33 * 17)
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect(),
                )
            };
            let frames = [
                AnimationFrame::new(&black, 10),
                AnimationFrame::new(&white, 20),
            ];
            let config = LossyConfig::new(1.0)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_auto_resampling(true)
                .with_resampling(1);
            let ordinary = config
                .encode_animation(33, 17, layout, &AnimationParams::default(), &frames)
                .unwrap();
            let bytes = config
                .with_already_downsampled(true)
                .encode_request(33, 17, layout)
                .encode_animation(&AnimationParams::default(), &frames)
                .unwrap();
            let label = format!("{policy}-{layout:?}");
            assert_eq!(ordinary, bytes, "{label}: factor-one bytes");
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap();
            assert_eq!((image.width(), image.height()), (33, 17));
            let rendered = decoded_extras::decode_current_frames(&bytes, &label);
            assert_eq!(rendered.len(), 2);
            for (index, (color, extras)) in rendered.iter().enumerate() {
                assert_eq!(color.len(), 33 * 17 * 3);
                assert!(
                    color
                        .iter()
                        .all(|v| v.is_finite() && (*v - index as f32).abs() < 0.005),
                    "{label}"
                );
                if layout == PixelLayout::Rgba16 {
                    assert_eq!(extras.len(), 1);
                    assert_eq!(extras[0].len(), 33 * 17);
                    assert!(extras[0].iter().all(|&v| v == 1.0));
                } else {
                    assert!(extras.is_empty());
                }
            }
            let directory = root.join(&label);
            std::fs::create_dir(&directory).unwrap();
            let input = directory.join("input.jxl");
            std::fs::write(&input, &bytes).unwrap();
            let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(input)
                .arg(directory.join("frame.pam"))
                .args(["--output_frames", "--num_threads=1", "--bits_per_sample=16"])
                .output()
                .unwrap();
            std::fs::write(directory.join("native.log"), &native.stderr).unwrap();
            assert!(
                native.status.success(),
                "{label}: {}",
                String::from_utf8_lossy(&native.stderr)
            );
            let mut outputs: Vec<_> = std::fs::read_dir(&directory)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "pam"))
                .collect();
            outputs.sort();
            assert_eq!(outputs.len(), 2);
            for (index, path) in outputs.iter().enumerate() {
                let image = image::open(path).unwrap().to_rgba16();
                assert_eq!(image.dimensions(), (33, 17));
                let expected = if index == 0 { 0 } else { 65535 };
                assert!(
                    image
                        .as_raw()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|p| p[3] == 65535
                            && p[..3].iter().all(|&v| v.abs_diff(expected) <= 328)),
                    "{label}: native samples"
                );
            }
        }
    }
}

#[test]
fn animation_resampling_default_policies_and_custom_permission() {
    use jxl_encoder::api::{EncoderImprovementsCustom, EncoderMode, EncoderStrategy};
    let root = directory();
    let pixels = vec![0_u8; 33 * 17 * 3];
    let frames = [
        AnimationFrame::new(&pixels, 10),
        AnimationFrame::new(&pixels, 20),
    ];
    let mut policies = encoder_policies::policies();
    for permission in [false, true] {
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            policies.push((
                format!("custom-auto-{permission}-{mode:?}"),
                EncoderStrategy::Custom(Box::new(EncoderImprovementsCustom {
                    auto_resample_libjxl_rule: permission,
                    ..Default::default()
                })),
                mode,
            ));
        }
    }
    for (policy, strategy, mode) in policies {
        let config = LossyConfig::new(12.0)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode);
        let factor = config.effective_resampling();
        let expected_factor =
            if policy.starts_with("libjxl") || policy.starts_with("custom-auto-true") {
                2
            } else {
                1
            };
        assert_eq!(
            factor, expected_factor,
            "{policy}: registered automatic permission"
        );
        let result = config
            .encode_request(33, 17, PixelLayout::Rgb8)
            .encode_animation(&AnimationParams::default(), &frames);
        if factor > 1 {
            let error = result.unwrap_err();
            assert!(
                error.to_string().contains("resampling"),
                "{policy}: {error}"
            );
            continue;
        }
        let bytes = result.unwrap();
        let forced = config
            .with_resampling(1)
            .encode_animation(
                33,
                17,
                PixelLayout::Rgb8,
                &AnimationParams::default(),
                &frames,
            )
            .unwrap();
        assert_eq!(bytes, forced, "{policy}: unchanged factor-one bytes");
        let rendered = decoded_extras::decode_current_frames(&bytes, &policy);
        assert_eq!(rendered.len(), 2);
        for (color, extras) in rendered {
            assert_eq!(color.len(), 33 * 17 * 3);
            assert!(color.iter().all(|v| v.is_finite() && v.abs() < 0.005));
            assert!(extras.is_empty());
        }
        let output = root.join(&policy);
        std::fs::create_dir(&output).unwrap();
        let input = output.join("input.jxl");
        std::fs::write(&input, &bytes).unwrap();
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(input)
            .arg(output.join("frame.pam"))
            .args(["--output_frames", "--num_threads=1"])
            .output()
            .unwrap();
        std::fs::write(output.join("native.log"), &native.stderr).unwrap();
        assert!(
            native.status.success(),
            "{policy}: {}",
            String::from_utf8_lossy(&native.stderr)
        );
        let outputs: Vec<_> = std::fs::read_dir(output)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "pam"))
            .collect();
        assert_eq!(outputs.len(), 2);
        for path in outputs {
            let decoded = image::open(path).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), (33, 17));
            assert!(decoded.as_raw().iter().all(|&v| v == 0));
        }
    }
}
