//! Annex M classification from actual serialized zero-duration frame areas.
#![cfg(feature = "__expert")]
use jxl_encoder::{
    AnimationFrame, AnimationParams, BlendMode, LosslessConfig, LossyConfig, PixelLayout,
};
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

fn level(bytes: &[u8]) -> u8 {
    if bytes.starts_with(&[0xff, 0x0a]) {
        return 5;
    }
    assert_eq!(&bytes[..12], b"\0\0\0\x0cJXL \r\n\x87\n");
    let mut position = 0;
    while position < bytes.len() {
        let size = u32::from_be_bytes(bytes[position..position + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && position + size <= bytes.len());
        if &bytes[position + 4..position + 8] == b"jxll" {
            assert_eq!(size, 9);
            return bytes[position + 8];
        }
        position += size;
    }
    5
}

fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("animation-zero-area-contract");
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

fn check(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let (width, height) = (4096_u32, 4096_u32);
    let black = vec![0_u8; width as usize * height as usize * 3];
    let white = vec![255_u8; black.len()];
    let gray = vec![128_u8; black.len()];
    let mut journal = std::fs::File::create(root.join("areas.tsv")).unwrap();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in policies {
        for lossless in [true, false] {
            for (zero_frames, crop) in [(16, false), (17, false), (17, true)] {
                let mut frames: Vec<_> = (0..zero_frames)
                    .map(|index| {
                        let pixels = if crop {
                            &white
                        } else if index % 2 == 1 {
                            &gray
                        } else {
                            &black
                        };
                        let frame = AnimationFrame::new(pixels, 0);
                        if crop {
                            frame
                        } else {
                            frame.with_blend_mode(BlendMode::Replace)
                        }
                    })
                    .collect();
                frames.push(AnimationFrame::new(&white, 10).with_blend_mode(BlendMode::Replace));
                let label = format!("{policy}-{lossless}-{zero_frames}-{crop}");
                let output = root.join(&label);
                std::fs::create_dir(&output).unwrap();
                writeln!(journal, "{label}\tstage=encoding").unwrap();
                journal.flush().unwrap();
                let bytes = if lossless {
                    LosslessConfig::new()
                        .with_effort(1)
                        .with_tree_learning(false)
                        .with_modular_predictor(Some(0))
                        .with_patches(false)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .encode_animation(
                            width,
                            height,
                            PixelLayout::Rgb8,
                            &AnimationParams::default(),
                            &frames,
                        )
                } else {
                    LossyConfig::new(1.0)
                        .with_effort(1)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_resampling(1)
                        .encode_animation(
                            width,
                            height,
                            PixelLayout::Rgb8,
                            &AnimationParams::default(),
                            &frames,
                        )
                }
                .unwrap();
                let input = output.join("input.jxl");
                std::fs::write(&input, &bytes).unwrap();
                writeln!(
                    journal,
                    "{label}\tstage=primary-render\tbytes={}",
                    bytes.len()
                )
                .unwrap();
                journal.flush().unwrap();
                let decoded = decoded_extras::decode_current_frames(&bytes, &label);
                assert_eq!(decoded.len(), 1, "{label}: coalesced displayed frame");
                let (color, extras) = &decoded[0];
                assert_eq!(color.len(), white.len());
                assert!(extras.is_empty());
                if lossless {
                    assert!(color.iter().all(|&v| v == 1.0), "{label}: exact source");
                } else {
                    assert!(
                        color
                            .iter()
                            .all(|&v| v.is_finite() && (v - 1.0).abs() < 0.005),
                        "{label}: source"
                    );
                }
                drop(decoded);
                let image = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                writeln!(
                    journal,
                    "{label}\tmetadata={:?}",
                    image.image_header().metadata
                )
                .unwrap();
                assert!(
                    image.image_header().metadata.modular_16bit_buffers,
                    "{label}: the independent modular-width flag must not mask this area control"
                );
                assert_eq!(image.num_loaded_frames(), frames.len());
                let mut accumulated = 0_u64;
                let mut maximum = 0_u64;
                for index in 0..image.num_loaded_frames() {
                    let header = image.frame(index).unwrap().header();
                    if header.duration == 0 {
                        accumulated += u64::from(header.width) * u64::from(header.height);
                        maximum = maximum.max(accumulated);
                    } else {
                        accumulated = 0;
                    }
                    writeln!(
                        journal,
                        "{label}\tframe={index}\t{}x{}\tduration={}\taccumulated={accumulated}",
                        header.width, header.height, header.duration
                    )
                    .unwrap();
                }
                let observed = level(&bytes);
                let required = if maximum > 1 << 28 { 10 } else { 5 };
                writeln!(
                    journal,
                    "{label}\tmaximum={maximum}\tlevel={observed}\trequired={required}"
                )
                .unwrap();
                journal.flush().unwrap();
                if observed != required {
                    failures.push(format!(
                        "{label}: actual maximum {maximum}, level {observed} instead of {required}"
                    ));
                }
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(input)
                    .arg(output.join("display.png"))
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(output.join("native.log"), &native.stderr).unwrap();
                assert!(
                    native.status.success(),
                    "{label}: {}",
                    String::from_utf8_lossy(&native.stderr)
                );
                let pixels = image::open(output.join("display.png")).unwrap().to_rgb8();
                assert_eq!(pixels.dimensions(), (width, height));
                assert!(
                    pixels.as_raw().iter().all(|&v| v == 255),
                    "{label}: native white"
                );
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

#[test]
#[ignore = "large resource probe; select explicitly through the frame-level just recipes"]
fn strict_zero_duration_level_uses_actual_frame_areas() {
    check(vec![encoder_policies::policies().remove(0)]);
}

#[test]
#[ignore = "large resource probe; select explicitly through the frame-level just recipes"]
fn all_offered_zero_duration_levels_use_actual_frame_areas() {
    check(encoder_policies::policies());
}

fn check_duration(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let pixels = [255_u8; 8 * 8 * 3];
    let mut journal = std::fs::File::create(root.join("durations.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut manifests = Vec::new();
    for (policy, strategy, mode) in policies {
        for lossless in [true, false] {
            for (numerator, denominator, duration, hidden, required) in [
                (120, 1, 1, false, 5),
                (121, 1, 1, false, 10),
                (240, 2, 1, false, 5),
                (241, 2, 1, false, 10),
                (240, 1, 2, false, 5),
                (241, 1, 2, false, 10),
                (1000, 1, 0, false, 5),
                (1000, 1, 1, true, 5),
                (1 << 30, 1024, u32::MAX, false, 5),
            ] {
                let params = AnimationParams {
                    tps_numerator: numerator,
                    tps_denominator: denominator,
                    ..Default::default()
                };
                let mut frames = vec![
                    AnimationFrame::new(&pixels, duration).with_blend_mode(BlendMode::Replace),
                ];
                if hidden {
                    frames[0] = AnimationFrame::new(&pixels, duration)
                        .with_reference_only(true)
                        .with_save_as_reference(2);
                    frames.push(
                        AnimationFrame::new(&pixels, 1000).with_blend_mode(BlendMode::Replace),
                    );
                }
                let label =
                    format!("{policy}-{lossless}-{numerator}-{denominator}-{duration}-{hidden}");
                let output = root.join(&label);
                std::fs::create_dir(&output).unwrap();
                let (bytes, legacy) = if lossless {
                    let cfg = LosslessConfig::new()
                        .with_effort(1)
                        .with_tree_learning(false)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    (
                        cfg.encode_request(8, 8, PixelLayout::Rgb8)
                            .encode_animation(&params, &frames)
                            .unwrap(),
                        cfg.encode_animation(8, 8, PixelLayout::Rgb8, &params, &frames)
                            .unwrap(),
                    )
                } else {
                    let cfg = LossyConfig::new(1.0)
                        .with_effort(1)
                        .with_resampling(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    (
                        cfg.encode_request(8, 8, PixelLayout::Rgb8)
                            .encode_animation(&params, &frames)
                            .unwrap(),
                        cfg.encode_animation(8, 8, PixelLayout::Rgb8, &params, &frames)
                            .unwrap(),
                    )
                };
                assert_eq!(bytes, legacy, "{label}: entries");
                let path = output.join("input.jxl");
                std::fs::write(&path, &bytes).unwrap();
                let image = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                assert!(
                    image.image_header().metadata.modular_16bit_buffers,
                    "{label}: width must not mask duration"
                );
                assert_eq!(image.num_loaded_frames(), frames.len());
                assert_eq!(
                    image.frame(0).unwrap().header().duration,
                    if hidden { 0 } else { duration }
                );
                if hidden {
                    assert_eq!(image.frame(1).unwrap().header().duration, 1000);
                }
                let observed = level(&bytes);
                writeln!(journal, "{label}\tlevel={observed}\trequired={required}").unwrap();
                journal.flush().unwrap();
                if observed != required {
                    failures.push(format!("{label}: level={observed}, required={required}"));
                }
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(&path)
                    .arg(output.join("output.png"))
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(output.join("native.log"), &native.stderr).unwrap();
                assert!(
                    native.status.success(),
                    "{label}: {}",
                    String::from_utf8_lossy(&native.stderr)
                );
                let pixels = image::open(output.join("output.png")).unwrap().to_rgb8();
                assert_eq!(pixels.dimensions(), (8, 8));
                assert!(pixels.as_raw().iter().all(|&v| v == 255));
                let primary = std::panic::catch_unwind(|| {
                    decoded_extras::decode_current_frames(&bytes, &label)
                });
                if let Ok(decoded) = primary {
                    assert_eq!(decoded.len(), 1);
                    assert_eq!(decoded[0].0.len(), pixels.len());
                    assert!(decoded[0].1.is_empty());
                    assert!(
                        decoded[0].0.iter().all(|&v| v.is_finite()
                            && if lossless {
                                v == 1.0
                            } else {
                                (v - 1.0).abs() < 0.005
                            }),
                        "{label}: source"
                    );
                } else {
                    failures.push(format!("{label}: primary decoder refused"));
                }
                manifests.push(serde_json::json!({"path":path,"frames":frames.len(),"reference":false,"blend":"Replace","tps_numerator":numerator,"tps_denominator":denominator}));
            }
        }
    }
    let manifest = root.join("streams.json");
    std::fs::write(&manifest, serde_json::to_vec(&manifests).unwrap()).unwrap();
    let parsed = std::process::Command::new("python3")
        .arg(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("scripts/check_explicit_blend_headers.py"),
        )
        .arg(manifest)
        .arg(root.join("headers.json"))
        .output()
        .unwrap();
    std::fs::write(
        root.join("headers.log"),
        [parsed.stdout, parsed.stderr].concat(),
    )
    .unwrap();
    assert!(parsed.status.success());
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        root.display()
    );
}
#[test]
fn strict_minimum_nonzero_duration_level() {
    check_duration(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_minimum_nonzero_duration_levels() {
    check_duration(encoder_policies::policies());
}
