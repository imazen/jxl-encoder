//! The PNG16 CLI routes preserve the serialized modular-width level requirement.
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{EncoderMode, LossyConfig, PixelLayout};
use std::process::Command;
#[path = "../../jxl-encoder/tests/support/jxl_signature.rs"]
mod jxl_signature;
#[path = "../../jxl-encoder/tests/support/decoded_extras.rs"]
mod primary;

fn encoder_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
    if let Some(template) = std::env::var_os("LLVM_PROFILE_FILE") {
        let parent = std::path::Path::new(&template).parent().unwrap();
        command.env("LLVM_PROFILE_FILE", parent.join("width-cli-%1m.profraw"));
    }
    command
}

#[test]
fn png16_alpha_declares_level_ten_across_cli_modes_and_streaming_routes() {
    let root = jxl_encoder::test_helpers::output_dir("modular-width-cli");
    std::fs::create_dir_all(&root).unwrap();
    let out = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", path.display()),
            }
        })
        .unwrap();
    for alpha in [false, true] {
        for (width, height) in [(32_u32, 17_u32), (259, 17)] {
            let pixel: Vec<u16> = if alpha {
                vec![0x3333, 0x6666, 0x9999, 65534]
            } else {
                vec![0x3333, 0x6666, 0x9999]
            };
            let samples = pixel.repeat((width * height) as usize);
            let input = out.join(format!("source-{alpha}-{width}.png"));
            if alpha {
                image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(width, height, samples.clone())
                    .unwrap()
                    .save(&input)
                    .unwrap();
            } else {
                image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(width, height, samples.clone())
                    .unwrap()
                    .save(&input)
                    .unwrap();
            }
            let pixels: Vec<u8> = samples
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect();
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
                    let expected = LossyConfig::new(1.0)
                        .with_effort(1)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .encode(
                            &pixels,
                            width,
                            height,
                            if alpha {
                                PixelLayout::Rgba16
                            } else {
                                PixelLayout::Rgb16
                            },
                        )
                        .unwrap();
                    for route in ["bulk", "streaming-input", "streaming-output"] {
                        let label = format!("{alpha}-{width}-{spelling}-{experimental}-{route}");
                        let directory = out.join(&label);
                        std::fs::create_dir(&directory).unwrap();
                        let output = directory.join("output.jxl");
                        let mut command = encoder_command();
                        command.arg(&input).arg(&output).args([
                            "--effort",
                            "1",
                            "--threads",
                            "1",
                            "--distance",
                            "1",
                            "--strategy",
                            spelling,
                        ]);
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
                        std::fs::write(
                            directory.join("encode.log"),
                            [result.stdout, result.stderr].concat(),
                        )
                        .unwrap();
                        assert!(result.status.success(), "{label}");
                        let encoded = std::fs::read(&output).unwrap();
                        assert_eq!(encoded, expected, "{label}: library and CLI bytes");
                        jxl_signature::assert_signature(&encoded);
                        if alpha {
                            assert!(
                                encoded.starts_with(jxl_signature::LEVEL_TEN_PREFIX),
                                "{label}: Level10"
                            );
                        } else {
                            assert!(
                                encoded.starts_with(&[0xff, 0x0a]),
                                "{label}: ordinary Level5 RGB16"
                            );
                        }
                        let (color, extras) = primary::decode_current_planes(&encoded, &label);
                        assert_eq!(color.len(), (width * height * 3) as usize);
                        assert_eq!(extras.len(), usize::from(alpha));
                        if alpha {
                            assert!(
                                extras[0]
                                    .iter()
                                    .all(|sample| (sample * 65535.0).round() == 65534.0)
                            );
                        }
                        let native = Command::new(jxl_encoder::test_helpers::djxl_path())
                            .arg(&output)
                            .arg(directory.join("native.png"))
                            .args(["--num_threads=1", "--bits_per_sample=16"])
                            .output()
                            .unwrap();
                        std::fs::write(
                            directory.join("native.log"),
                            [native.stdout, native.stderr].concat(),
                        )
                        .unwrap();
                        assert!(native.status.success(), "{label}: native rendering");
                        let decoded = image::open(directory.join("native.png")).unwrap();
                        assert_eq!((decoded.width(), decoded.height()), (width, height));
                        assert_eq!(decoded.color().has_alpha(), alpha);
                        if alpha {
                            assert!(decoded.to_rgba16().pixels().all(|pixel| pixel[3] == 65534));
                        }
                    }
                }
            }
        }
    }
}
