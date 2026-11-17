//! Automatic crops must use a reference matching the full-canvas source model.
#![cfg(feature = "__expert")]
use jxl_encoder::{
    AnimationFrame, AnimationParams, BlendMode, LosslessConfig, LossyConfig, PixelLayout,
};
#[path = "support/decoded_frames.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("animation-crop-reference-contract");
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

fn check(lossless_path: bool, case: u8) {
    let root = directory();
    let mut envelopes = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(8, 8), (64, 32), (259, 17)] {
            for layout in [PixelLayout::Rgb8, PixelLayout::Rgba16] {
                let count = width as usize * height as usize;
                let (black, white) = if layout == PixelLayout::Rgb8 {
                    (vec![0_u8; count * 3], vec![255_u8; count * 3])
                } else {
                    (
                        [0_u16, 0, 0, 65535]
                            .repeat(count)
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect(),
                        [65535_u16; 4]
                            .repeat(count)
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect(),
                    )
                };
                let w = || AnimationFrame::new(&white, 10);
                let b = || AnimationFrame::new(&black, 10);
                let frames = match case {
                    0 => vec![w(), w()],
                    1 => vec![w().with_save_as_reference(2), w()],
                    2 => vec![w().with_save_as_reference(0), w()],
                    3 => vec![
                        w(),
                        b().with_reference_only(true).with_save_as_reference(1),
                        w(),
                    ],
                    4 => vec![
                        w(),
                        b().with_blend_mode(BlendMode::Add).with_blend_source(1),
                        b(),
                    ],
                    5 => vec![
                        w(),
                        b().with_reference_only(true).with_save_as_reference(2),
                        w().with_blend_source(2),
                    ],
                    6 => vec![
                        w(),
                        b().with_reference_only(true).with_save_as_reference(2),
                        w(),
                    ],
                    7 => vec![w().with_blend_mode(BlendMode::Replace), w()],
                    9 => vec![w(), w().with_blend_source(1)],
                    8 => vec![
                        b().with_save_as_reference(2),
                        AnimationFrame::new(&white, 0).with_save_as_reference(2),
                        w(),
                    ],
                    _ => unreachable!(),
                };
                let expected: &[u8] = match case {
                    4 => &[1, 1, 0],
                    8 => &[0, 1],
                    _ => &[1, 1],
                };
                let label = format!("{policy}-{case}-{lossless_path}-{width}x{height}-{layout:?}");
                let output = root.join(&label);
                std::fs::create_dir(&output).unwrap();
                let (bytes, legacy) = if lossless_path {
                    let config = LosslessConfig::new()
                        .with_effort(1)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    (
                        config
                            .encode_request(width, height, layout)
                            .encode_animation(&AnimationParams::default(), &frames)
                            .unwrap(),
                        config
                            .encode_animation(
                                width,
                                height,
                                layout,
                                &AnimationParams::default(),
                                &frames,
                            )
                            .unwrap(),
                    )
                } else {
                    let config = LossyConfig::new(1.0)
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_resampling(1);
                    (
                        config
                            .encode_request(width, height, layout)
                            .encode_animation(&AnimationParams::default(), &frames)
                            .unwrap(),
                        config
                            .encode_animation(
                                width,
                                height,
                                layout,
                                &AnimationParams::default(),
                                &frames,
                            )
                            .unwrap(),
                    )
                };
                assert_eq!(legacy, bytes, "{label}: entry bytes");
                let input = output.join("input.jxl");
                std::fs::write(&input, &bytes).unwrap();
                let rendered = decoded_extras::decode_current_frames(&bytes, &label);
                assert_eq!(rendered.len(), expected.len(), "{label}");
                let summaries = rendered
                    .iter()
                    .map(|(pixels, _)| {
                        format!(
                            "first {:?}, min/max {:?}\n",
                            &pixels[..3],
                            pixels.iter().fold(
                                (f32::INFINITY, f32::NEG_INFINITY),
                                |(lo, hi), &v| (lo.min(v), hi.max(v))
                            )
                        )
                    })
                    .collect::<String>();
                std::fs::write(output.join("rust-samples.txt"), summaries).unwrap();
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(input)
                    .arg(output.join("frame.pam"))
                    .args(["--output_frames", "--num_threads=1", "--bits_per_sample=16"])
                    .output()
                    .unwrap();
                std::fs::write(output.join("native.log"), &native.stderr).unwrap();
                assert!(
                    native.status.success(),
                    "{label}: {}",
                    String::from_utf8_lossy(&native.stderr)
                );
                let mut paths: Vec<_> = std::fs::read_dir(&output)
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .filter(|path| path.extension().is_some_and(|ext| ext == "pam"))
                    .collect();
                paths.sort();
                assert_eq!(paths.len(), expected.len(), "{label}");
                for (index, ((color, extras), &value)) in rendered.iter().zip(expected).enumerate()
                {
                    assert_eq!(color.len(), count * 3, "{label}");
                    if lossless_path {
                        assert!(
                            color.iter().all(|&v| v == f32::from(value)),
                            "{label}: Rust exact frame {index}"
                        );
                    } else {
                        assert!(
                            color
                                .iter()
                                .all(|&v| v.is_finite() && (v - f32::from(value)).abs() < 0.005),
                            "{label}: Rust frame {index}"
                        );
                    }
                    if layout == PixelLayout::Rgba16 {
                        assert_eq!(extras.len(), 1);
                        assert_eq!(extras[0].len(), count);
                        assert!(extras[0].iter().all(|&v| v == 1.0), "{label}: exact alpha");
                    } else {
                        assert!(extras.is_empty());
                    }
                    let decoded = image::open(&paths[index]).unwrap().to_rgba16();
                    assert_eq!(decoded.dimensions(), (width, height));
                    let sample = u16::from(value) * 65535;
                    let tolerance = if lossless_path { 0 } else { 328 };
                    assert!(
                        decoded
                            .as_raw()
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|p| p[3] == 65535
                                && p[..3].iter().all(|&v| v.abs_diff(sample) <= tolerance)),
                        "{label}: native frame {index}"
                    );
                }
                let expected_crop =
                    matches!(case, 0 | 6 | 7 | 9) && (lossless_path || width > 8 || height > 8);
                envelopes.push(serde_json::json!({"path": output.join("input.jxl"),
                    "frames": frames.len(), "reference": false, "partial_area": expected_crop}));
            }
        }
    }
    let manifest = root.join("envelopes.json");
    std::fs::write(&manifest, serde_json::to_vec_pretty(&envelopes).unwrap()).unwrap();
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../scripts/check_explicit_blend_headers.py");
    let parsed = std::process::Command::new("python3")
        .arg(script)
        .arg(manifest)
        .arg(root.join("observed-envelopes.json"))
        .output()
        .unwrap();
    std::fs::write(
        root.join("envelope.log"),
        [&parsed.stdout[..], &parsed.stderr[..]].concat(),
    )
    .unwrap();
    assert!(
        parsed.status.success(),
        "independent complete envelope/TOC checks: {}",
        String::from_utf8_lossy(&parsed.stderr)
    );
}

macro_rules! regression {
    ($name:ident, $lossless:expr, $case:expr) => {
        #[test]
        fn $name() {
            check($lossless, $case);
        }
    };
}
regression!(nondefault_slot_lossless, true, 1);
regression!(nondefault_slot_lossy, false, 1);
regression!(unsaved_canvas_lossless, true, 2);
regression!(unsaved_canvas_lossy, false, 2);
regression!(overwritten_reference_lossless, true, 3);
regression!(overwritten_reference_lossy, false, 3);
regression!(composed_canvas_lossless, true, 4);
regression!(composed_canvas_lossy, false, 4);
regression!(different_source_lossless, true, 5);
regression!(different_source_lossy, false, 5);
regression!(hidden_nondefault_lossless, true, 8);
regression!(hidden_nondefault_lossy, false, 8);
#[test]
fn valid_reference_controls_keep_cropping() {
    for lossless in [true, false] {
        for case in [0, 6, 7] {
            check(lossless, case);
        }
    }
}

#[test]
fn matching_explicit_source_keeps_cropping() {
    for lossless in [true, false] {
        check(lossless, 9);
    }
}
