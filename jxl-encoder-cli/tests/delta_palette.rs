//! Actual near-lossless CLI forwarding; quantized-pixel semantics have unit observers.
use jxl_encoder::api::{EncoderMode, EncoderStrategy};
use jxl_encoder::{LosslessConfig, PixelLayout};
use std::process::{Command, Stdio};
#[path = "../../jxl-encoder/tests/support/decoded_frames.rs"]
mod decoded_frames;
fn run_cli(command: &mut Command, label: &str) -> std::process::Output {
    let profile = std::env::var_os("LLVM_PROFILE_FILE")
        .map(std::path::PathBuf::from)
        .map(|p| p.with_file_name(format!("grid-child-{label}-%p-%m.profraw")));
    if let Some(p) = &profile {
        command.env("LLVM_PROFILE_FILE", p);
    }
    let invocation = format!("{command:?}");
    let child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let result = child.wait_with_output().unwrap();
    if let Some(p) = profile {
        std::fs::write(
            p.with_file_name(format!("grid-child-{label}.meta.txt")),
            format!(
                "{pid}\n{label}\n{}\n{}\n{invocation}\n",
                env!("CARGO_BIN_EXE_cjxl-rs"),
                result.status.code().unwrap()
            ),
        )
        .unwrap();
    }
    result
}
#[test]
fn delta_palette_all_cli_spellings_and_output_routes() {
    let root = jxl_encoder::test_helpers::output_dir("delta-palette-cli");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    for (width, height) in [(32u32, 32u32), (259, 17)] {
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let (xx, yy) = ((x % 32) as i32, (y % 32) as i32);
                    (if xx == 0 {
                        [128, 120, 136]
                    } else {
                        [
                            128 + 4 * xx - 4 * yy,
                            120 + 4 * xx - 3 * yy,
                            136 + 3 * xx - 4 * yy,
                        ]
                    })
                    .map(|v| u8::try_from(v).unwrap())
                })
            })
            .collect();
        let input = root.join(format!("{width}x{height}.png"));
        image::RgbImage::from_raw(width, height, pixels.clone())
            .unwrap()
            .save(&input)
            .unwrap();
        for (spelling, strategy) in [
            ("", EncoderStrategy::Zenjxl),
            ("libjxl", EncoderStrategy::Libjxl),
            ("libjxl-exact", EncoderStrategy::Libjxl),
            ("libjxl-strict", EncoderStrategy::Libjxl),
            ("lean-faster", EncoderStrategy::LeanFaster),
            ("zenjxl", EncoderStrategy::Zenjxl),
            ("aggressive", EncoderStrategy::Aggressive),
        ] {
            for experimental in [false, true] {
                let enabled: &[bool] = if spelling == "libjxl" {
                    &[false, true]
                } else {
                    &[true]
                };
                for &use_palette in enabled {
                    for use_ans in [false, true] {
                        let cfg = LosslessConfig::new()
                            .with_effort(7)
                            .with_threads(1)
                            .with_patches(false)
                            .with_squeeze(false)
                            .with_tree_learning(false)
                            .with_lz77(false)
                            .with_modular_predictor(Some(0))
                            .with_modular_palette_colors(Some(16))
                            .with_lossy_palette(use_palette)
                            .with_strategy(strategy.clone())
                            .with_mode(if experimental {
                                EncoderMode::Experimental
                            } else {
                                EncoderMode::Reference
                            });
                        let cfg = if use_ans { cfg } else { cfg.with_ans(false) };
                        let expected = cfg
                            .encode(&pixels, width, height, PixelLayout::Rgb8)
                            .unwrap();
                        let reference = decoded_frames::decode_current_frames(
                            &expected,
                            "palette CLI reference",
                        );
                        assert_eq!(reference.len(), 1);
                        assert!(reference[0].1.is_empty());
                        for route in ["buffered", "streaming-output", "streaming-input"] {
                            let label = format!(
                                "palette-{width}x{height}-{spelling}-{experimental}-{use_ans}-{use_palette}-{route}"
                            );
                            let output = root.join(format!("{label}.jxl"));
                            let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                            command.arg(&input).arg(&output).args([
                                "--lossless",
                                "--effort",
                                "7",
                                "--threads",
                                "1",
                                "--no-patches",
                                "--no-squeeze",
                                "--no-tree-learning",
                                "--no-lz77",
                                "--modular-predictor",
                                "0",
                                "--modular-palette-colors",
                                "16",
                            ]);
                            if use_palette {
                                command.arg("--lossy-palette");
                            }
                            if !spelling.is_empty() {
                                command.args(["--strategy", spelling]);
                            }
                            if experimental {
                                command.arg("--experimental");
                            }
                            if !use_ans {
                                command.arg("--no-ans");
                            }
                            if route != "buffered" {
                                command.arg(format!("--{route}"));
                            }
                            let result = run_cli(&mut command, &label);
                            std::fs::write(
                                root.join(format!("{label}.log")),
                                [result.stdout.as_slice(), result.stderr.as_slice()].concat(),
                            )
                            .unwrap();
                            assert!(
                                result.status.success(),
                                "{label}: {}",
                                String::from_utf8_lossy(&result.stderr)
                            );
                            let bytes = std::fs::read(&output).unwrap();
                            assert_eq!(bytes, expected, "{label}: exact CLI/API bytes");
                            let actual = decoded_frames::decode_current_frames(&bytes, &label);
                            assert_eq!(actual, reference);
                            let native_path = root.join(format!("{label}.png"));
                            let native = Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(&output)
                                .arg(&native_path)
                                .args(["--num_threads=1", "--bits_per_sample=8"])
                                .output()
                                .unwrap();
                            std::fs::write(
                                root.join(format!("{label}.native.log")),
                                [native.stdout.as_slice(), native.stderr.as_slice()].concat(),
                            )
                            .unwrap();
                            assert!(native.status.success());
                            let native = image::open(native_path).unwrap().to_rgb8().into_raw();
                            let codes: Vec<u8> = actual[0]
                                .0
                                .iter()
                                .map(|v| (*v * 255.0).round() as u8)
                                .collect();
                            assert_eq!(native, codes, "{label}: complete primary/native codes");
                        }
                    }
                }
            }
        }
    }
}

#[path = "../../jxl-encoder/tests/support/pfm.rs"]
mod pfm;
#[test]
fn delta_palette_all_cli_spellings_apng_animation() {
    use jxl_encoder::{AnimationFrame, AnimationParams};
    let root = jxl_encoder::test_helpers::output_dir("delta-palette-cli-animation");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    for (width, height) in [(32u32, 32u32), (259, 17)] {
        let first: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let (xx, yy) = ((x % 32) as i32, (y % 32) as i32);
                    (if xx == 0 {
                        [128, 120, 136]
                    } else {
                        [
                            128 + 4 * xx - 4 * yy,
                            120 + 4 * xx - 3 * yy,
                            136 + 3 * xx - 4 * yy,
                        ]
                    })
                    .map(|v| u8::try_from(v).unwrap())
                })
            })
            .collect();
        let second: Vec<u8> = first
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        let input = root.join(format!("{width}x{height}.png"));
        let mut encoder = png::Encoder::new(std::fs::File::create(&input).unwrap(), width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(2, 0).unwrap();
        encoder.set_frame_delay(1, 10).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&first).unwrap();
        writer.write_image_data(&second).unwrap();
        writer.finish().unwrap();
        let animation = AnimationParams {
            tps_numerator: 1000,
            tps_denominator: 1,
            num_loops: 0,
            premultiplied_alpha: false,
        };
        for (spelling, strategy) in [
            ("", EncoderStrategy::Zenjxl),
            ("libjxl", EncoderStrategy::Libjxl),
            ("libjxl-exact", EncoderStrategy::Libjxl),
            ("libjxl-strict", EncoderStrategy::Libjxl),
            ("lean-faster", EncoderStrategy::LeanFaster),
            ("zenjxl", EncoderStrategy::Zenjxl),
            ("aggressive", EncoderStrategy::Aggressive),
        ] {
            for experimental in [false, true] {
                let enabled: &[bool] = if spelling == "libjxl" {
                    &[false, true]
                } else {
                    &[true]
                };
                for &use_palette in enabled {
                    for use_ans in [false, true] {
                        let cfg = LosslessConfig::new()
                            .with_effort(7)
                            .with_threads(1)
                            .with_patches(false)
                            .with_squeeze(false)
                            .with_tree_learning(false)
                            .with_lz77(false)
                            .with_modular_predictor(Some(0))
                            .with_modular_palette_colors(Some(16))
                            .with_lossy_palette(use_palette)
                            .with_strategy(strategy.clone())
                            .with_mode(if experimental {
                                EncoderMode::Experimental
                            } else {
                                EncoderMode::Reference
                            });
                        let cfg = if use_ans { cfg } else { cfg.with_ans(false) };
                        let expected = cfg
                            .encode_animation(
                                width,
                                height,
                                PixelLayout::Rgb8,
                                &animation,
                                &[
                                    AnimationFrame::new(&first, 100),
                                    AnimationFrame::new(&second, 100),
                                ],
                            )
                            .unwrap();
                        let label = format!(
                            "palette-animation-{width}x{height}-{spelling}-{experimental}-{use_ans}-{use_palette}"
                        );
                        let out = root.join(&label);
                        std::fs::create_dir(&out).unwrap();
                        let path = out.join("input.jxl");
                        let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                        command.arg(&input).arg(&path).args([
                            "--lossless",
                            "--effort",
                            "7",
                            "--threads",
                            "1",
                            "--no-patches",
                            "--no-squeeze",
                            "--no-tree-learning",
                            "--no-lz77",
                            "--modular-predictor",
                            "0",
                            "--modular-palette-colors",
                            "16",
                        ]);
                        if use_palette {
                            command.arg("--lossy-palette");
                        }
                        if !spelling.is_empty() {
                            command.args(["--strategy", spelling]);
                        }
                        if experimental {
                            command.arg("--experimental");
                        }
                        if !use_ans {
                            command.arg("--no-ans");
                        }
                        let result = run_cli(&mut command, &label);
                        std::fs::write(
                            out.join("cli.log"),
                            [result.stdout.as_slice(), result.stderr.as_slice()].concat(),
                        )
                        .unwrap();
                        assert!(
                            result.status.success(),
                            "{}",
                            String::from_utf8_lossy(&result.stderr)
                        );
                        let bytes = std::fs::read(&path).unwrap();
                        assert_eq!(bytes, expected, "APNG CLI/API bytes");
                        let primary =
                            decoded_frames::decode_current_frames(&bytes, "APNG palette animation");
                        assert_eq!(primary.len(), 2);
                        let native = Command::new(jxl_encoder::test_helpers::djxl_path())
                            .arg(&path)
                            .arg(out.join("native.pfm"))
                            .args([
                                "--num_threads=1",
                                "--output_frames",
                                "--color_space=RGB_D65_SRG_Rel_SRG",
                            ])
                            .output()
                            .unwrap();
                        std::fs::write(
                            out.join("native.log"),
                            [native.stdout.as_slice(), native.stderr.as_slice()].concat(),
                        )
                        .unwrap();
                        assert!(
                            native.status.success(),
                            "{}",
                            String::from_utf8_lossy(&native.stderr)
                        );
                        let mut paths: Vec<_> = std::fs::read_dir(&out)
                            .unwrap()
                            .map(|p| p.unwrap().path())
                            .filter(|p| p.extension().is_some_and(|e| e == "pfm"))
                            .collect();
                        paths.sort();
                        assert_eq!(paths.len(), 2, "every native displayed frame retained");
                        for (frame, path) in primary.iter().zip(paths) {
                            assert!(frame.1.is_empty());
                            let native =
                                pfm::read_pfm_samples(&path, (width as usize, height as usize));
                            let codes: Vec<u8> =
                                native.iter().map(|v| (*v * 255.0).round() as u8).collect();
                            let primary: Vec<u8> =
                                frame.0.iter().map(|v| (*v * 255.0).round() as u8).collect();
                            assert_eq!(codes, primary, "complete APNG frame samples");
                        }
                    }
                }
            }
        }
    }
}
