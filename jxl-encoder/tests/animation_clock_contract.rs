//! SMPTE clock order at public animation entries.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, LosslessConfig, LossyConfig, PixelLayout};
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("animation-clock-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
#[test]
fn animation_nonzero_timecodes_must_increase() {
    let root = directory();
    let black = vec![0_u8; 64 * 32 * 3];
    let white = vec![255_u8; 64 * 32 * 3];
    let mut journal = std::fs::File::create(root.join("timecodes.tsv")).unwrap();
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
        for lossless_path in [true, false] {
            for legacy in [true, false] {
                for second in [1, 2] {
                    let frames = [
                        AnimationFrame::new(&black, 10).with_timecode(2),
                        AnimationFrame::new(&white, 20).with_timecode(second),
                    ];
                    let result = if legacy {
                        if lossless_path {
                            lossless.encode_animation(
                                64,
                                32,
                                PixelLayout::Rgb8,
                                &AnimationParams::default(),
                                &frames,
                            )
                        } else {
                            lossy.encode_animation(
                                64,
                                32,
                                PixelLayout::Rgb8,
                                &AnimationParams::default(),
                                &frames,
                            )
                        }
                    } else {
                        let request = if lossless_path {
                            lossless.encode_request(64, 32, PixelLayout::Rgb8)
                        } else {
                            lossy.encode_request(64, 32, PixelLayout::Rgb8)
                        };
                        request.encode_animation(&AnimationParams::default(), &frames)
                    };
                    let label = format!("{policy}-{lossless_path}-{legacy}-{second}");
                    match result {
                        Err(error) => {
                            assert!(
                                error.to_string().contains("nonzero timecode"),
                                "{label}: {error}"
                            );
                            writeln!(journal, "{label}\terror {error}").unwrap()
                        }
                        Ok(bytes) => {
                            std::fs::write(root.join(format!("{label}.jxl")), bytes).unwrap();
                            writeln!(journal, "{label}\taccepted").unwrap();
                            failures.push(label);
                        }
                    };
                    journal.flush().unwrap();
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "accepted invalid clocks {failures:?}; {}",
        root.display()
    );
}

#[test]
fn animation_clock_history_controls_preserve_fields_and_pixels() {
    let root = directory();
    let black = vec![0_u8; 64 * 32 * 3];
    let white = vec![255_u8; 64 * 32 * 3];
    // (timecode, duration, ReferenceOnly, source sample). Hidden normal
    // clocks compare with displayed history but do not advance it.
    let cases = [
        vec![(2, 10, false, 0), (3, 20, false, 255)],
        vec![(2, 10, false, 0), (0, 10, false, 255), (3, 20, false, 0)],
        vec![(2, 10, false, 0), (100, 0, false, 255), (3, 20, false, 255)],
        vec![(2, 10, false, 0), (1, 77, true, 255), (3, 20, false, 255)],
        vec![
            (u32::MAX, 10, true, 255),
            (2, 10, false, 0),
            (3, 20, false, 255),
        ],
        vec![(u32::MAX, 0, false, 255), (1, 10, false, 0)],
        vec![(u32::MAX, 10, false, 0), (0, 20, false, 255)],
        vec![(0, 10, false, 0), (0, 20, false, 255)],
    ];
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
        for (case, specification) in cases.iter().enumerate() {
            let frames: Vec<_> = specification
                .iter()
                .map(|&(clock, duration, reference, sample)| {
                    let frame =
                        AnimationFrame::new(if sample == 0 { &black } else { &white }, duration)
                            .with_timecode(clock)
                            .with_reference_only(reference);
                    if reference {
                        frame.with_save_as_reference(2)
                    } else {
                        frame
                    }
                })
                .collect();
            let expected: Vec<u8> = specification
                .iter()
                .filter(|(_, duration, reference, _)| *duration != 0 && !reference)
                .map(|(_, _, _, sample)| *sample)
                .collect();
            for lossless_path in [true, false] {
                let label = format!("{policy}-{case}-{lossless_path}");
                let request = if lossless_path {
                    lossless.encode_request(64, 32, PixelLayout::Rgb8)
                } else {
                    lossy.encode_request(64, 32, PixelLayout::Rgb8)
                };
                let bytes = request
                    .encode_animation(&AnimationParams::default(), &frames)
                    .unwrap();
                let legacy = if lossless_path {
                    lossless.encode_animation(
                        64,
                        32,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                    )
                } else {
                    lossy.encode_animation(
                        64,
                        32,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                    )
                }
                .unwrap();
                assert_eq!(legacy, bytes, "{label}: entry bytes");
                let directory = root.join(&label);
                std::fs::create_dir(&directory).unwrap();
                let input = directory.join("input.jxl");
                std::fs::write(&input, &bytes).unwrap();
                let image = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                assert_eq!(image.num_loaded_frames(), specification.len(), "{label}");
                for (index, &(clock, duration, reference, _)) in specification.iter().enumerate() {
                    let header = image.frame(index).unwrap().header();
                    assert_eq!(
                        header.timecode,
                        if reference { 0 } else { clock },
                        "{label}/{index}"
                    );
                    assert_eq!(
                        header.duration,
                        if reference { 0 } else { duration },
                        "{label}/{index}"
                    );
                }
                let rendered = decoded_extras::decode_current_frames(&bytes, &label);
                assert_eq!(rendered.len(), expected.len(), "{label}");
                for ((color, extras), &sample) in rendered.iter().zip(&expected) {
                    assert_eq!(color.len(), 64 * 32 * 3);
                    assert!(extras.is_empty());
                    let expected = f32::from(sample) / 255.0;
                    if lossless_path {
                        assert!(
                            color.iter().all(|&v| v == expected),
                            "{label}: exact source samples expected {expected}; first {:?}, min/max {:?}",
                            &color[..12],
                            color.iter().fold(
                                (f32::INFINITY, f32::NEG_INFINITY),
                                |(lo, hi), &v| (lo.min(v), hi.max(v))
                            )
                        );
                    } else {
                        assert!(
                            color
                                .iter()
                                .all(|&v| v.is_finite() && (v - expected).abs() < 0.005),
                            "{label}"
                        );
                    }
                }
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(input)
                    .arg(directory.join("frame.ppm"))
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
                    .filter(|path| path.extension().is_some_and(|ext| ext == "ppm"))
                    .collect();
                outputs.sort();
                assert_eq!(outputs.len(), expected.len(), "{label}");
                for (path, &sample) in outputs.iter().zip(&expected) {
                    let decoded = image::open(path).unwrap().to_rgb16();
                    assert_eq!(decoded.dimensions(), (64, 32));
                    let expected = u16::from(sample) * 257;
                    if lossless_path {
                        assert!(
                            decoded.as_raw().iter().all(|&v| v == expected),
                            "{label}: native exact source"
                        );
                    } else {
                        assert!(
                            decoded
                                .as_raw()
                                .iter()
                                .all(|&v| v.abs_diff(expected) <= 328),
                            "{label}: native source"
                        );
                    }
                }
            }
        }
    }
}
