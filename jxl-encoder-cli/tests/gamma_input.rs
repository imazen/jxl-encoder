//! PNG gAMA forwarding through every CLI pixel entry.
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{EncoderMode, LosslessConfig, LossyConfig, PixelLayout};
use std::process::Command;
#[path = "../../jxl-encoder/tests/support/decoded_extras.rs"]
mod decoded_extras;
#[path = "../../jxl-encoder/tests/support/pfm.rs"]
mod pfm;

#[test]
fn png_gamma_white_and_refusal_match_library_entries() {
    let root = jxl_encoder::test_helpers::output_dir("gamma-cli-contract");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    for layout in [
        PixelLayout::Rgb8,
        PixelLayout::Rgb16,
        PixelLayout::Gray8,
        PixelLayout::Gray16,
    ] {
        let pixels = vec![255_u8; 32 * 17 * layout.bytes_per_pixel()];
        for gamma in [0.00012_f32, 0.005, 1.0] {
            let input = root.join(format!("{layout:?}-{:x}.png", gamma.to_bits()));
            let mut png = png::Encoder::new(std::fs::File::create(&input).unwrap(), 32, 17);
            png.set_color(if layout.is_grayscale() {
                png::ColorType::Grayscale
            } else {
                png::ColorType::Rgb
            });
            png.set_depth(if layout.is_16bit() {
                png::BitDepth::Sixteen
            } else {
                png::BitDepth::Eight
            });
            png.set_source_gamma(png::ScaledFloat::new(gamma));
            png.write_header()
                .unwrap()
                .write_image_data(&pixels)
                .unwrap();
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
                        let expected = if lossless {
                            LosslessConfig::new()
                                .with_effort(3)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                                .encode_request(32, 17, layout)
                                .with_source_gamma(gamma)
                                .encode(&pixels)
                        } else {
                            LossyConfig::new(0.01)
                                .with_effort(3)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                                .encode_request(32, 17, layout)
                                .with_source_gamma(gamma)
                                .encode(&pixels)
                        };
                        assert_eq!(expected.is_ok(), gamma >= 0.005);
                        for route in ["bulk", "streaming-input", "streaming-output"] {
                            let label = format!(
                                "{layout:?}-{:x}-{spelling}-{experimental}-{lossless}-{route}",
                                gamma.to_bits()
                            );
                            let out = root.join(&label);
                            std::fs::create_dir(&out).unwrap();
                            let target = out.join("output.jxl");
                            std::fs::write(&target, b"existing-output").unwrap();
                            let mut cmd = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                            cmd.arg(&input).arg(&target).args([
                                "--effort",
                                "3",
                                "--threads",
                                "1",
                                "--strategy",
                                spelling,
                            ]);
                            if experimental {
                                cmd.arg("--experimental");
                            }
                            if lossless {
                                cmd.arg("--lossless");
                            } else {
                                cmd.args(["--distance", "0.01"]);
                            }
                            if route != "bulk" {
                                cmd.arg("--streaming-input");
                            }
                            if route == "streaming-output" {
                                cmd.arg("--streaming-output");
                            }
                            let result = cmd.output().unwrap();
                            std::fs::write(
                                out.join("encode.log"),
                                [result.stdout, result.stderr].concat(),
                            )
                            .unwrap();
                            let bytes = std::fs::read(&target).unwrap();
                            match &expected {
                                Err(_) => {
                                    assert!(!result.status.success(), "{label}");
                                    assert_eq!(bytes, b"existing-output");
                                }
                                Ok(expected) => {
                                    assert!(result.status.success(), "{label}");
                                    assert_eq!(&bytes, expected, "{label}");
                                    let (color, extras) =
                                        decoded_extras::decode_current_planes(&bytes, &label);
                                    assert!(extras.is_empty());
                                    assert_eq!(
                                        color.len(),
                                        32 * 17 * if layout.is_grayscale() { 1 } else { 3 }
                                    );
                                    assert!(
                                        color
                                            .iter()
                                            .all(|v| v.is_finite() && (v - 1.0).abs() < 0.002),
                                        "{label} primary"
                                    );
                                    let decoded =
                                        Command::new(jxl_encoder::test_helpers::djxl_path())
                                            .arg(&target)
                                            .arg(out.join("native.pfm"))
                                            .args([
                                                "--num_threads=1",
                                                "--color_space=RGB_D65_SRG_Rel_Lin",
                                            ])
                                            .output()
                                            .unwrap();
                                    std::fs::write(
                                        out.join("decode.log"),
                                        [decoded.stdout, decoded.stderr].concat(),
                                    )
                                    .unwrap();
                                    assert!(decoded.status.success(), "{label}");
                                    let native =
                                        pfm::read_pfm_samples(&out.join("native.pfm"), (32, 17));
                                    assert!(
                                        native
                                            .iter()
                                            .all(|v| v.is_finite() && (v - 1.0).abs() < 0.002),
                                        "{label} native"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
