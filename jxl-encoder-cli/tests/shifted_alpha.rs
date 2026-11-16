//! Exercise the CLI's alpha split and native-grid resampling entry point.
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{EncoderMode, LosslessConfig, PixelLayout};
use std::process::Command;
#[path = "../../jxl-encoder/tests/support/decoded_extras.rs"]
mod primary;

fn encoder_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
    // Child processes share one LLVM online-merge profile per binary.
    // Keeping %p here would create a full raw profile for every CLI cell.
    if let Some(template) = std::env::var_os("LLVM_PROFILE_FILE") {
        let template = std::path::Path::new(&template);
        let parent = template
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        command.env(
            "LLVM_PROFILE_FILE",
            parent.join("jxl-encoder-cli-%1m.profraw"),
        );
    }
    command
}

#[test]
fn downsampled_alpha_survives_odd_groups_in_every_cli_mode() {
    let root = jxl_encoder::test_helpers::output_dir("shifted-alpha-cli");
    std::fs::create_dir_all(&root).unwrap();
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{}: {e}", p.display()),
            }
        })
        .unwrap();
    let mut failures = Vec::new();
    for gray in [false, true] {
        for (width, height) in [(259, 17), (17, 259)] {
            let colors = if gray { 1 } else { 3 };
            let pixels: Vec<u8> = (0..width * height)
                .flat_map(|_| {
                    let mut p = vec![64; colors];
                    p.push(255);
                    p
                })
                .collect();
            let input = out.join(format!("source-{gray}-{width}.png"));
            if gray {
                image::GrayAlphaImage::from_raw(width, height, pixels.clone())
                    .unwrap()
                    .save(&input)
                    .unwrap();
            } else {
                image::RgbaImage::from_raw(width, height, pixels.clone())
                    .unwrap()
                    .save(&input)
                    .unwrap();
            }
            for factor in [2, 4, 8] {
                let native = vec![255; (width.div_ceil(factor) * height.div_ceil(factor)) as usize];
                let extra = [
                    jxl_encoder::api::ExtraChannel::from_alpha_buf(&native, false)
                        .with_dim_shift(factor.trailing_zeros()),
                ];
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
                        let expected = LosslessConfig::new()
                            .with_effort(1)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_fast_decode()
                            .encode_request(
                                width,
                                height,
                                if gray {
                                    PixelLayout::Gray8
                                } else {
                                    PixelLayout::Rgb8
                                },
                            )
                            .with_extra_channels(&extra)
                            .encode(&vec![64; width as usize * height as usize * colors])
                            .unwrap();
                        for route in ["bulk", "streaming-input", "streaming-output"] {
                            let label = format!(
                                "{gray}-{width}-{factor}-{spelling}-{experimental}-{route}"
                            );
                            let output = out.join(format!("{label}.jxl"));
                            let mut command = encoder_command();
                            command.arg(&input).arg(&output).args([
                                "--lossless",
                                "--effort",
                                "1",
                                "--threads",
                                "1",
                                "--fast-decode",
                                "--strategy",
                                spelling,
                                "--ec_resampling",
                                &factor.to_string(),
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
                            std::fs::write(out.join(format!("{label}.encode.log")), &result.stderr)
                                .unwrap();
                            assert!(
                                result.status.success(),
                                "{label}: {}",
                                String::from_utf8_lossy(&result.stderr)
                            );
                            let bytes = std::fs::read(&output).unwrap();
                            assert_eq!(bytes, expected, "{label}: library/CLI bytes");
                            let samples = std::panic::catch_unwind(|| {
                                primary::decode_current_planes(&bytes, &label)
                            });
                            match samples {
                                Ok((samples, planes)) => {
                                    assert_eq!(
                                        samples.len(),
                                        width as usize * height as usize * colors
                                    );
                                    assert_eq!(planes.len(), 1);
                                    assert_eq!(planes[0].len(), width as usize * height as usize);
                                    if !samples.iter().all(|v| *v == 64.0 / 255.0)
                                        || !planes[0].iter().all(|v| *v == 1.0)
                                    {
                                        failures.push(format!("{label}: Rust samples"));
                                    }
                                }
                                Err(_) => failures.push(format!("{label}: Rust rendering")),
                            }
                            let png = out.join(format!("{label}.png"));
                            let result = Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(&output)
                                .arg(&png)
                                .args(["--bits_per_sample=16", "--num_threads=1"])
                                .output()
                                .unwrap();
                            std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr)
                                .unwrap();
                            if result.status.success() {
                                let image = image::open(png).unwrap().to_rgba16();
                                assert_eq!(image.dimensions(), (width, height));
                                if !image
                                    .pixels()
                                    .all(|p| p.0 == [64 * 257, 64 * 257, 64 * 257, 65535])
                                {
                                    failures.push(format!("{label}: djxl samples"));
                                }
                            } else {
                                failures.push(format!("{label}: djxl rendering"));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}: {failures:?}", out.display());
}

#[test]
fn gray_alpha_png_layouts_preserve_eight_and_sixteen_bit_samples() {
    let root = jxl_encoder::test_helpers::output_dir("gray-alpha-cli");
    std::fs::create_dir_all(&root).unwrap();
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{}: {e}", p.display()),
            }
        })
        .unwrap();
    let (width, height) = (17_u32, 9_u32);
    for wide in [false, true] {
        let samples: Vec<u16> = (0..width * height)
            .flat_map(|i| {
                [
                    64 * 257,
                    if wide {
                        (i as u16) * 317 + 1
                    } else {
                        (i as u16 % 255 + 1) * 257
                    },
                ]
            })
            .collect();
        let input = out.join(format!("source-{wide}.png"));
        let (layout, pixels) = if wide {
            image::ImageBuffer::<image::LumaA<u16>, _>::from_raw(width, height, samples.clone())
                .unwrap()
                .save(&input)
                .unwrap();
            (
                PixelLayout::GrayAlpha16,
                samples
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<u8>>(),
            )
        } else {
            let pixels: Vec<u8> = samples.iter().map(|v| (v / 257) as u8).collect();
            image::GrayAlphaImage::from_raw(width, height, pixels.clone())
                .unwrap()
                .save(&input)
                .unwrap();
            (PixelLayout::GrayAlpha8, pixels)
        };
        for (spelling, strategy) in [
            ("libjxl", EncoderStrategy::Libjxl),
            ("libjxl-exact", EncoderStrategy::Libjxl),
            ("libjxl-strict", EncoderStrategy::Libjxl),
            ("zenjxl", EncoderStrategy::Zenjxl),
            ("aggressive", EncoderStrategy::Aggressive),
            ("lean-faster", EncoderStrategy::LeanFaster),
        ] {
            for experimental in [false, true] {
                let expected = LosslessConfig::new()
                    .with_effort(1)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_fast_decode()
                    .with_mode(if experimental {
                        EncoderMode::Experimental
                    } else {
                        EncoderMode::Reference
                    })
                    .encode(&pixels, width, height, layout)
                    .unwrap();
                for route in ["bulk", "streaming-input", "streaming-output"] {
                    let label = format!("{wide}-{spelling}-{experimental}-{route}");
                    let output = out.join(format!("{label}.jxl"));
                    let mut command = encoder_command();
                    command.arg(&input).arg(&output).args([
                        "--lossless",
                        "--effort",
                        "1",
                        "--threads",
                        "1",
                        "--fast-decode",
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
                    std::fs::write(out.join(format!("{label}.encode.log")), &result.stderr)
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{label}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    let bytes = std::fs::read(&output).unwrap();
                    assert_eq!(bytes, expected, "{label}: library bytes");
                    let (colors, extras) = primary::decode_current_planes(&bytes, &label);
                    assert_eq!(colors.len(), (width * height) as usize);
                    assert_eq!(extras.len(), 1);
                    assert!(
                        colors
                            .iter()
                            .all(|v| (*v * 65535.0).round() as u16 == 64 * 257),
                        "{label}: exact color codes"
                    );
                    assert_eq!(
                        extras[0]
                            .iter()
                            .map(|v| (*v * 65535.0).round() as u16)
                            .collect::<Vec<_>>(),
                        samples
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|p| p[1])
                            .collect::<Vec<_>>(),
                        "{label}: exact alpha codes"
                    );
                    let png = out.join(format!("{label}.png"));
                    let result = Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&output)
                        .arg(&png)
                        .args(["--bits_per_sample=16", "--num_threads=1"])
                        .output()
                        .unwrap();
                    std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr)
                        .unwrap();
                    assert!(result.status.success(), "{label}: djxl rendering");
                    assert_eq!(
                        image::open(png).unwrap().to_luma_alpha16().as_raw(),
                        &samples,
                        "{label}: djxl samples"
                    );
                }
            }
        }
    }
}
