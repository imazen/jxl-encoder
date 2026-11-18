//! CLI profile models, streaming admission and destination preservation.
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{EncoderMode, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout};
use std::process::Command;
#[path = "../../jxl-encoder/examples/distance_targeting_probe/decode.rs"]
mod primary;

#[test]
fn icc_models_match_library_output_and_refuse_mismatches_before_replacing_output() {
    let root = jxl_encoder::test_helpers::output_dir("icc-cli-contract");
    let directory = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir_all(&root).and_then(|()| std::fs::create_dir(&path)) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("create {}: {error}", path.display()),
            }
        })
        .unwrap();
    let profiles: Vec<Vec<u8>> =
        include_str!("../../jxl-encoder/src/headers/color_profiles_v012.tsv")
            .lines()
            .filter(|s| !s.starts_with('#'))
            .map(|s| {
                s.split_once('\t')
                    .unwrap()
                    .1
                    .as_bytes()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
                    .collect()
            })
            .collect();
    let mut failures = Vec::new();
    for gray in [false, true] {
        let input = directory.join(format!("source-{gray}.png"));
        let pixels = vec![64; 8 * 8 * if gray { 1 } else { 3 }];
        if gray {
            image::GrayImage::from_raw(8, 8, pixels.clone())
                .unwrap()
                .save(&input)
                .unwrap();
        } else {
            image::RgbImage::from_raw(8, 8, pixels.clone())
                .unwrap()
                .save(&input)
                .unwrap();
        }
        for matched in [true, false] {
            let profile = &profiles[usize::from(if matched { gray } else { !gray })];
            let icc = directory.join(format!("profile-{gray}-{matched}.icc"));
            std::fs::write(&icc, profile).unwrap();
            let metadata = ImageMetadata::new().with_icc_profile(profile);
            let layout = if gray {
                PixelLayout::Gray8
            } else {
                PixelLayout::Rgb8
            };
            for (spelling, strategy) in [
                ("libjxl", EncoderStrategy::Libjxl),
                ("libjxl-exact", EncoderStrategy::Libjxl),
                ("libjxl-strict", EncoderStrategy::Libjxl),
                ("lean-faster", EncoderStrategy::LeanFaster),
                ("zenjxl", EncoderStrategy::Zenjxl),
                ("aggressive", EncoderStrategy::Aggressive),
            ] {
                for experimental in [false, true] {
                    let mode = if experimental {
                        EncoderMode::Experimental
                    } else {
                        EncoderMode::Reference
                    };
                    for lossless in [false, true] {
                        let expected = if matched {
                            Some(if lossless {
                                LosslessConfig::new()
                                    .with_effort(1)
                                    .with_threads(1)
                                    .with_strategy(strategy.clone())
                                    .with_mode(mode)
                                    .encode_request(8, 8, layout)
                                    .with_metadata(&metadata)
                                    .encode(&pixels)
                                    .unwrap()
                            } else {
                                LossyConfig::new(1.0)
                                    .with_effort(1)
                                    .with_threads(1)
                                    .with_strategy(strategy.clone())
                                    .with_mode(mode)
                                    .encode_request(8, 8, layout)
                                    .with_metadata(&metadata)
                                    .encode(&pixels)
                                    .unwrap()
                            })
                        } else {
                            None
                        };
                        for route in ["bulk", "streaming-input", "streaming-output"] {
                            let label = format!(
                                "{gray}-{matched}-{spelling}-{experimental}-{lossless}-{route}"
                            );
                            let output = directory.join(format!("{label}.jxl"));
                            std::fs::write(&output, b"existing output").unwrap();
                            let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                            command
                                .arg(&input)
                                .arg(&output)
                                .args([
                                    "--effort",
                                    "1",
                                    "--threads",
                                    "1",
                                    "--strategy",
                                    spelling,
                                    "--icc",
                                ])
                                .arg(&icc);
                            if lossless {
                                command.arg("--lossless");
                            } else {
                                command.args(["--distance", "1"]);
                            }
                            if experimental {
                                command.arg("--experimental");
                            }
                            if route != "bulk" {
                                command.arg("--streaming-input");
                            }
                            if route == "streaming-output" {
                                command.arg("--streaming-output");
                            }
                            let result = command.output().unwrap();
                            std::fs::write(directory.join(format!("{label}.log")), &result.stderr)
                                .unwrap();
                            let bytes = std::fs::read(&output).unwrap();
                            if let Some(expected) = &expected {
                                assert!(
                                    result.status.success(),
                                    "{label}: {}",
                                    String::from_utf8_lossy(&result.stderr)
                                );
                                assert_eq!(&bytes, expected, "{label} must match library bytes");
                                let samples = primary::verify_jxl_rs(&bytes, 8, 8);
                                assert_eq!(samples.len(), pixels.len());
                                if lossless {
                                    assert!(
                                        samples.iter().all(|v| *v == 64.0 / 255.0),
                                        "{label}: exact source samples"
                                    );
                                }

                                let decoded = directory.join(format!("{label}.png"));
                                let result = Command::new(jxl_encoder::test_helpers::djxl_path())
                                    .arg(&output)
                                    .arg(&decoded)
                                    .arg(format!(
                                        "--orig_icc_out={}",
                                        directory.join(format!("{label}.icc")).display()
                                    ))
                                    .arg("--num_threads=1")
                                    .output()
                                    .unwrap();
                                assert!(
                                    result.status.success(),
                                    "{label}: {}",
                                    String::from_utf8_lossy(&result.stderr)
                                );
                                image::open(decoded).unwrap();
                                assert_eq!(
                                    std::fs::read(directory.join(format!("{label}.icc"))).unwrap(),
                                    *profile
                                );
                            } else {
                                if result.status.success()
                                    || !String::from_utf8_lossy(&result.stderr)
                                        .contains("ICC profile grayscale model")
                                {
                                    failures.push(format!(
                                        "{label}: wrong refusal/status {}",
                                        result.status
                                    ));
                                }
                                if bytes != b"existing output" {
                                    failures.push(format!(
                                        "{label}: replaced prior output with {} bytes",
                                        bytes.len()
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}: {failures:?}", directory.display());
}

#[test]
fn cmyk_icc_without_a_black_input_refuses_before_replacing_output() {
    let root = jxl_encoder::test_helpers::output_dir("cmyk-cli-contract");
    let directory = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let profile: Vec<u8> = include_str!("../../jxl-encoder/tests/fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect();
    let icc = directory.join("cmyk.icc");
    std::fs::write(&icc, profile).unwrap();
    for gray in [false, true] {
        let input = directory.join(format!("source-{gray}.png"));
        if gray {
            image::GrayImage::from_raw(8, 8, vec![64; 8 * 8])
                .unwrap()
                .save(&input)
                .unwrap();
        } else {
            image::RgbImage::from_raw(8, 8, vec![64; 8 * 8 * 3])
                .unwrap()
                .save(&input)
                .unwrap();
        }
        for spelling in [
            "libjxl",
            "libjxl-exact",
            "libjxl-strict",
            "lean-faster",
            "zenjxl",
            "aggressive",
        ] {
            for experimental in [false, true] {
                for lossless in [false, true] {
                    for route in ["bulk", "streaming-input", "streaming-output"] {
                        let label = format!("{gray}-{spelling}-{experimental}-{lossless}-{route}");
                        let output = directory.join(format!("{label}.jxl"));
                        std::fs::write(&output, b"existing output").unwrap();
                        let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                        command
                            .arg(&input)
                            .arg(&output)
                            .args([
                                "--effort",
                                "1",
                                "--threads",
                                "1",
                                "--strategy",
                                spelling,
                                "--icc",
                            ])
                            .arg(&icc);
                        if lossless {
                            command.arg("--lossless");
                        } else {
                            command.args(["--distance", "1"]);
                        }
                        if experimental {
                            command.arg("--experimental");
                        }
                        if route != "bulk" {
                            command.arg("--streaming-input");
                        }
                        if route == "streaming-output" {
                            command.arg("--streaming-output");
                        }
                        let result = command.output().unwrap();
                        std::fs::write(directory.join(format!("{label}.log")), &result.stderr)
                            .unwrap();
                        assert!(!result.status.success(), "{label}");
                        assert_eq!(
                            std::fs::read(&output).unwrap(),
                            b"existing output",
                            "{label}"
                        );
                        let error = String::from_utf8_lossy(&result.stderr);
                        assert!(
                            error.contains(if gray {
                                "ICC profile grayscale"
                            } else {
                                "CMYK ICC"
                            }),
                            "{label}: {error}"
                        );
                    }
                }
            }
        }
    }
}
