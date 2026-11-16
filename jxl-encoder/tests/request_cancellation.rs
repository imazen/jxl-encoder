//! Public request cancellation retains every observed Stop error.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy, ExtraChannel, SectionedTrees};
use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout, RctType};
use std::sync::atomic::{AtomicUsize, Ordering};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

struct StopAt {
    at: usize,
    polls: AtomicUsize,
}
impl enough::Stop for StopAt {
    #[track_caller]
    fn check(&self) -> Result<(), enough::StopReason> {
        if self.polls.fetch_add(1, Ordering::Relaxed) == self.at {
            Err(enough::StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn base(strategy: EncoderStrategy, mode: EncoderMode, group: u8) -> LosslessConfig {
    LosslessConfig::new()
        .with_effort(1)
        .with_strategy(strategy)
        .with_mode(mode)
        .with_threads(1)
        .with_modular_group_size(Some(group))
        .with_squeeze(false)
        .with_patches(false)
        .with_tree_learning(false)
        .with_modular_palette_colors(Some(0))
        .with_modular_channel_colors_group_percent(Some(0.0))
        .with_force_rct(Some(RctType(0)))
}

#[test]
fn shifted_request_propagates_every_observed_cancellation_and_preserves_destination() {
    use jxl_encoder::EncodeError;
    let pixels = vec![0; 259 * 17 * 3];
    let native = vec![173; 259_usize.div_ceil(8) * 17_usize.div_ceil(8)];
    let extras = [ExtraChannel::depth(&native).with_dim_shift(3)];
    for (name, strategy, mode) in [
        ("strict", EncoderStrategy::Libjxl, EncoderMode::Reference),
        (
            "experimental",
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ] {
        for sectioned in [SectionedTrees::Off, SectionedTrees::On] {
            let config = base(strategy.clone(), mode, 1)
                .with_effort(7)
                .with_tree_learning(true)
                .with_modular_predictor(Some(0))
                .with_sectioned_trees(sectioned)
                .with_ans(true);
            let complete = StopAt {
                at: usize::MAX,
                polls: AtomicUsize::new(0),
            };
            let baseline = config
                .encode_request(259, 17, PixelLayout::Rgb8)
                .with_extra_channels(&extras)
                .with_stop(&complete)
                .encode(&pixels)
                .unwrap();
            let polls = complete.polls.load(Ordering::Relaxed);
            assert!(polls > 2, "{name} {sectioned:?}: late checkpoints required");
            for at in 0..polls {
                let stop = StopAt {
                    at,
                    polls: AtomicUsize::new(0),
                };
                let mut destination = vec![1, 3, 5, 7];
                let result = config
                    .encode_request(259, 17, PixelLayout::Rgb8)
                    .with_extra_channels(&extras)
                    .with_stop(&stop)
                    .encode_into(&pixels, &mut destination);
                if stop.polls.load(Ordering::Relaxed) > at {
                    assert!(
                        matches!(&result, Err(e) if matches!(e.error(), EncodeError::Cancelled)),
                        "{name} {sectioned:?} at={at}: observed stop was lost: {result:?}"
                    );
                    assert_eq!(destination, [1, 3, 5, 7]);
                } else {
                    // A parallel task may finish with fewer polls than the baseline.
                    result.unwrap();
                    assert_eq!(&destination[..4], [1, 3, 5, 7]);
                    assert_eq!(&destination[4..], baseline);
                }
            }
        }
    }
}

#[test]
fn ordinary_lossless_and_lossy_requests_preserve_output_at_every_observed_stop() {
    let root = jxl_encoder::test_helpers::output_dir("request-cancellation");
    std::fs::create_dir_all(&root).unwrap();
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
    let pixels = vec![0; 32 * 17 * 3];
    for (name, strategy, mode) in encoder_policies::policies() {
        let lossless = base(strategy.clone(), mode, 1)
            .with_effort(7)
            .with_tree_learning(true)
            .with_modular_predictor(Some(0))
            .with_ans(true);
        let lossy = LossyConfig::new(1.0)
            .with_effort(3)
            .with_strategy(strategy)
            .with_mode(mode)
            .with_threads(1)
            .with_butteraugli_iters(0);
        for is_lossless in [true, false] {
            let request = || {
                if is_lossless {
                    lossless.encode_request(32, 17, PixelLayout::Rgb8)
                } else {
                    lossy.encode_request(32, 17, PixelLayout::Rgb8)
                }
            };
            let complete = StopAt {
                at: usize::MAX,
                polls: AtomicUsize::new(0),
            };
            let baseline = request().with_stop(&complete).encode(&pixels).unwrap();
            assert_eq!(
                baseline,
                request().encode(&pixels).unwrap(),
                "never-stopped still bytes"
            );
            let label = format!("{name}-lossless{is_lossless}");
            let (color, extras) = decoded_extras::decode_current_planes(&baseline, &label);
            assert!(extras.is_empty());
            assert_eq!(color.len(), pixels.len());
            let codes: Vec<u8> = color
                .iter()
                .map(|v| (*v * 255.0).round().clamp(0.0, 255.0) as u8)
                .collect();
            if is_lossless {
                assert_eq!(codes, pixels);
            } else {
                assert!(codes.iter().all(|v| *v <= 1), "{label}: black source");
            }
            let input = out.join(format!("{label}.jxl"));
            let output = out.join(format!("{label}.png"));
            std::fs::write(&input, &baseline).unwrap();
            let reference = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(&input)
                .arg(&output)
                .arg("--num_threads=1")
                .output()
                .unwrap();
            std::fs::write(out.join(format!("{label}.decode.log")), &reference.stderr).unwrap();
            assert!(reference.status.success(), "{label}: reference rendering");
            let codes = image::open(output).unwrap().to_rgb8().into_raw();
            assert_eq!(codes.len(), pixels.len());
            if is_lossless {
                assert_eq!(codes, pixels);
            } else {
                assert!(
                    codes.iter().all(|v| *v <= 1),
                    "{label}: reference black source"
                );
            }
            let polls = complete.polls.load(Ordering::Relaxed);
            assert!(polls > 2, "{label}: late checkpoints");
            for at in 0..polls {
                let stop = StopAt {
                    at,
                    polls: AtomicUsize::new(0),
                };
                let mut destination = vec![1, 3, 5, 7];
                let result = request()
                    .with_stop(&stop)
                    .encode_into(&pixels, &mut destination);
                if stop.polls.load(Ordering::Relaxed) > at {
                    assert!(
                        matches!(&result,Err(e) if matches!(e.error(),jxl_encoder::EncodeError::Cancelled)),
                        "{label} at={at}: {result:?}"
                    );
                    assert_eq!(destination, [1, 3, 5, 7]);
                } else {
                    result.unwrap();
                    assert_eq!(&destination[..4], [1, 3, 5, 7]);
                    assert_eq!(&destination[4..], baseline);
                }
            }
        }
    }
}

#[test]
fn animation_requests_retain_every_observed_stop() {
    use jxl_encoder::{AnimationFrame, AnimationParams};
    #[path = "support/pfm.rs"]
    mod pfm;
    let root = jxl_encoder::test_helpers::output_dir("animation-request-cancellation");
    std::fs::create_dir_all(&root).unwrap();
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
    let first = vec![0; 64 * 32 * 3];
    let second = vec![255; first.len()];
    let frames = [
        AnimationFrame::new(&first, 10),
        AnimationFrame::new(&second, 10),
    ];
    for (name, strategy, mode) in encoder_policies::policies() {
        let lossless = base(strategy.clone(), mode, 1)
            .with_effort(7)
            .with_tree_learning(true)
            .with_modular_predictor(Some(0))
            .with_ans(true)
            .with_auto_delta_frames(false);
        let lossy = LossyConfig::new(1.0)
            .with_effort(3)
            .with_strategy(strategy)
            .with_mode(mode)
            .with_threads(1)
            .with_butteraugli_iters(0)
            .with_auto_delta_frames(false);
        for is_lossless in [true, false] {
            let request = || {
                if is_lossless {
                    lossless.encode_request(64, 32, PixelLayout::Rgb8)
                } else {
                    lossy.encode_request(64, 32, PixelLayout::Rgb8)
                }
            };
            let complete = StopAt {
                at: usize::MAX,
                polls: AtomicUsize::new(0),
            };
            let baseline = request()
                .with_stop(&complete)
                .encode_animation(&AnimationParams::default(), &frames)
                .unwrap();
            assert_eq!(
                baseline,
                request()
                    .encode_animation(&AnimationParams::default(), &frames)
                    .unwrap(),
                "never-stopped animation bytes"
            );
            let label = format!("{name}-lossless{is_lossless}");
            let cell = out.join(&label);
            std::fs::create_dir(&cell).unwrap();
            std::fs::write(cell.join("input.jxl"), &baseline).unwrap();
            let decoded = decoded_extras::decode_current_frames(&baseline, &label);
            assert_eq!(decoded.len(), 2);
            for (index, (color, extras)) in decoded.into_iter().enumerate() {
                assert!(extras.is_empty());
                assert_eq!(color.len(), first.len());
                let expected = index as f32;
                if is_lossless {
                    assert!(color.iter().all(|v| v.to_bits() == expected.to_bits()));
                } else {
                    assert!(color.iter().all(|v| (*v - expected).abs() <= 1.0 / 255.0));
                }
            }
            let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(cell.join("input.jxl"))
                .arg(cell.join("frame.pfm"))
                .args(["--output_frames", "--num_threads=1"])
                .output()
                .unwrap();
            std::fs::write(cell.join("decode.log"), &result.stderr).unwrap();
            assert!(result.status.success(), "{label}: reference rendering");
            let mut reference: Vec<_> = std::fs::read_dir(&cell)
                .unwrap()
                .map(|p| p.unwrap().path())
                .filter(|p| p.extension().is_some_and(|e| e == "pfm"))
                .collect();
            reference.sort();
            assert_eq!(reference.len(), 2, "{label}: separate frame files");
            for (index, path) in reference.iter().enumerate() {
                let color = pfm::read_pfm_samples(path, (64, 32));
                let expected = index as f32;
                if is_lossless {
                    assert!(color.iter().all(|v| v.to_bits() == expected.to_bits()));
                } else {
                    assert!(color.iter().all(|v| (*v - expected).abs() <= 1.0 / 255.0));
                }
            }
            let polls = complete.polls.load(Ordering::Relaxed);
            assert!(polls > 2);
            for at in 0..polls {
                let stop = StopAt {
                    at,
                    polls: AtomicUsize::new(0),
                };
                let result = request()
                    .with_stop(&stop)
                    .encode_animation(&AnimationParams::default(), &frames);
                if stop.polls.load(Ordering::Relaxed) > at {
                    assert!(
                        matches!(&result,Err(e) if matches!(e.error(),jxl_encoder::EncodeError::Cancelled)),
                        "{label} animation at={at}: observed stop was lost: {result:?}"
                    );
                } else {
                    assert_eq!(result.unwrap(), baseline);
                }
            }
        }
    }
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn jpeg_stop_entries_propagate_every_observed_stop() {
    let jpeg = include_bytes!("fixtures/jbrd/base_a_444.jpg");
    let config = LosslessConfig::new().with_effort(3).with_threads(1);
    let root = jxl_encoder::test_helpers::output_dir("jpeg-request-cancellation");
    std::fs::create_dir_all(&root).unwrap();
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
    for container in [false, true] {
        let call = |stop: &dyn enough::Stop| {
            if container {
                config.encode_jpeg_transcode_with_stop(jpeg, stop)
            } else {
                config.encode_jpeg_transcode_codestream_with_stop(jpeg, stop)
            }
        };
        let complete = StopAt {
            at: usize::MAX,
            polls: AtomicUsize::new(0),
        };
        let baseline = call(&complete).unwrap();
        let plain = if container {
            config.encode_jpeg_transcode(jpeg)
        } else {
            config.encode_jpeg_transcode_codestream(jpeg)
        }
        .unwrap();
        assert_eq!(baseline, plain, "never-stopped JPEG bytes");
        let label = format!("container{container}");
        let (color, extras) = decoded_extras::decode_current_planes(&baseline, &label);
        assert!(!color.is_empty());
        assert!(extras.is_empty());
        let input = out.join(format!("{label}.jxl"));
        let output = out.join(format!("{label}.png"));
        std::fs::write(&input, &baseline).unwrap();
        let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(&input)
            .arg(&output)
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr).unwrap();
        assert!(result.status.success());
        assert_eq!(
            image::open(output).unwrap().to_rgb8().into_raw().len(),
            color.len()
        );
        if container {
            let rebuilt = out.join("rebuilt.jpg");
            let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(&input)
                .arg(&rebuilt)
                .output()
                .unwrap();
            std::fs::write(out.join("reconstruction.log"), &result.stderr).unwrap();
            assert!(result.status.success());
            assert_eq!(std::fs::read(rebuilt).unwrap(), jpeg);
        }
        let polls = complete.polls.load(Ordering::Relaxed);
        assert!(polls > 2);
        for at in 0..polls {
            let stop = StopAt {
                at,
                polls: AtomicUsize::new(0),
            };
            let result = call(&stop);
            if stop.polls.load(Ordering::Relaxed) > at {
                assert!(
                    matches!(&result,Err(e) if matches!(e.error(),jxl_encoder::EncodeError::Cancelled)),
                    "JPEG container={container} at={at}: observed stop lost: {result:?}"
                );
            } else {
                assert_eq!(result.unwrap(), baseline);
            }
        }
    }
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn low_level_jpeg_operations_retain_every_observed_stop() {
    use jxl_encoder::jpeg;
    let input = include_bytes!("fixtures/jbrd/base_a_444.jpg");
    let root = jxl_encoder::test_helpers::output_dir("low-level-jpeg-cancellation");
    std::fs::create_dir_all(&root).unwrap();
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
    for route in 0..4 {
        for scale in [1.0, 2.0] {
            let call = |stop: &dyn enough::Stop| -> Result<Vec<u8>, String> {
                match route {
                    0 => jpeg::read_jpeg(input, None, Some(stop))
                        .map_err(|e| format!("{e:?}"))
                        .and_then(|parsed| {
                            jpeg::encode_jpeg_to_jxl_with_effort(&parsed, 3)
                                .map_err(|e| format!("{e:?}"))
                        }),
                    1 => jpeg::encode_jpeg_recompress_codestream(
                        input,
                        scale,
                        0.1,
                        3,
                        None,
                        Some(stop),
                    )
                    .map_err(|e| format!("{e:?}")),
                    2 => jpeg::encode_jpeg_recompress_auto_codestream(
                        input,
                        scale,
                        3,
                        None,
                        Some(stop),
                    )
                    .map_err(|e| format!("{e:?}")),
                    _ => jpeg::encode_jpeg_recompress_planar_codestream(
                        input,
                        scale,
                        0.1,
                        scale,
                        0.1,
                        3,
                        None,
                        Some(stop),
                    )
                    .map_err(|e| format!("{e:?}")),
                }
            };
            let complete = StopAt {
                at: usize::MAX,
                polls: AtomicUsize::new(0),
            };
            let baseline = call(&complete).unwrap();
            let (color, _) = decoded_extras::decode_current_planes(&baseline, "low-level-jpeg");
            assert!(!color.is_empty());
            let label = format!("route{route}-scale{scale}");
            let encoded = out.join(format!("{label}.jxl"));
            let rendered = out.join(format!("{label}.png"));
            std::fs::write(&encoded, &baseline).unwrap();
            let reference = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(&encoded)
                .arg(&rendered)
                .arg("--num_threads=1")
                .output()
                .unwrap();
            std::fs::write(out.join(format!("{label}.decode.log")), &reference.stderr).unwrap();
            assert!(reference.status.success());
            assert_eq!(
                image::open(rendered).unwrap().to_rgb8().into_raw().len(),
                color.len()
            );
            let polls = complete.polls.load(Ordering::Relaxed);
            assert!(polls > 1);
            for at in 0..polls {
                let stop = StopAt {
                    at,
                    polls: AtomicUsize::new(0),
                };
                let result = call(&stop);
                if stop.polls.load(Ordering::Relaxed) > at {
                    assert_eq!(
                        result.unwrap_err(),
                        "Cancelled",
                        "route={route} scale={scale} at={at}"
                    );
                } else {
                    assert_eq!(result.unwrap(), baseline);
                }
            }
        }
    }
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn jpeg_parse_preserves_non_cancellation_decode_errors() {
    let input = include_bytes!("fixtures/jbrd/base_a_444.jpg");
    let sos = input.windows(2).position(|v| v == [0xff, 0xda]).unwrap();
    let mut missing_table = input.to_vec();
    missing_table[sos + 6] = 0x33; // absent DC/AC tables for the first component
    let error = jxl_encoder::jpeg::read_jpeg(&missing_table, None, None)
        .expect_err("a missing Huffman table must fail");
    assert!(
        matches!(error, jxl_encoder::jpeg::JpegError::Decode(_)),
        "{error:?}"
    );
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn jpeg_cancellation_keeps_explicit_limit_controls_byte_identical() {
    let input = include_bytes!("fixtures/jbrd/base_a_444.jpg");
    let plain = LosslessConfig::new().with_effort(3).with_threads(1);
    let limits = jxl_encoder::Limits::new().with_max_pixels(1_000_000);
    let limited = plain.clone().with_limits(&limits);
    for container in [false, true] {
        let expected = if container {
            plain.encode_jpeg_transcode(input)
        } else {
            plain.encode_jpeg_transcode_codestream(input)
        }
        .unwrap();
        let actual = if container {
            limited.encode_jpeg_transcode_with_stop(input, &enough::Unstoppable)
        } else {
            limited.encode_jpeg_transcode_codestream_with_stop(input, &enough::Unstoppable)
        }
        .unwrap();
        assert_eq!(actual, expected);
    }
}
