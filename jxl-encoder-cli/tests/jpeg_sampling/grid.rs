//! Partial absolute grids through automatic, forced and coarsening CLI routes.
use super::*;
pub(super) fn native(input: &std::path::Path, output: &std::path::Path) {
    let r = Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(input)
        .arg(output)
        .args(["--bits_per_sample=16", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(
        output.with_file_name(format!(
            "{}.decode.log",
            output.file_name().unwrap().to_string_lossy()
        )),
        [r.stdout, r.stderr].concat(),
    )
    .unwrap();
    assert!(r.status.success());
}
pub(super) fn run_cli(command: &mut Command, label: &str) -> std::process::Output {
    let profile = std::env::var_os("LLVM_PROFILE_FILE")
        .map(std::path::PathBuf::from)
        .map(|p| p.with_file_name(format!("grid-child-{label}-%p-%m.profraw")));
    if let Some(path) = &profile {
        command.env("LLVM_PROFILE_FILE", path);
    }
    let invocation = format!("{command:?}");
    let child = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let output = child.wait_with_output().unwrap();
    if let Some(path) = profile {
        std::fs::write(
            path.with_file_name(format!("grid-child-{label}.meta.txt")),
            format!(
                "{pid}\n{label}\n{}\n{}\n{invocation}\n",
                env!("CARGO_BIN_EXE_cjxl-rs"),
                output.status.code().unwrap()
            ),
        )
        .unwrap();
    }
    output
}
#[test]
fn partial_common_sampling_grids_survive_cli_routes_and_efforts() {
    let root = jxl_encoder::test_helpers::output_dir("jpeg-cli-partial-grid");
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
    let text = include_str!(
        "../../../jxl-encoder/tests/fixtures/jbrd/scan_long-codes-nonzero-before.jpg.hex"
    );
    let original: Vec<u8> = text
        .trim()
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect();
    let mut sof = 2;
    while original[sof + 1] != 0xc0 {
        assert_eq!(original[sof], 255);
        sof += 2 + usize::from(u16::from_be_bytes([original[sof + 2], original[sof + 3]]));
    }
    for transpose in [false, true] {
        for sampling in [0x21, 0x12, 0x22] {
            let mut input = original.clone();
            input[sof + 11] = sampling;
            if transpose {
                input[sof + 5..sof + 7].copy_from_slice(&16u16.to_be_bytes());
                input[sof + 7..sof + 9].copy_from_slice(&8u16.to_be_bytes());
            }
            let source = directory.join(format!("{transpose}-{sampling}-reference.jpg"));
            std::fs::write(&source, &input).unwrap();
            let reference = source.with_extension("jxl");
            let r = Command::new(jxl_encoder::test_helpers::cjxl_path())
                .arg(&source)
                .arg(&reference)
                .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
                .output()
                .unwrap();
            std::fs::write(
                source.with_extension("encode.log"),
                [r.stdout, r.stderr].concat(),
            )
            .unwrap();
            assert!(r.status.success());
            let reference_png = source.with_extension("png");
            native(&reference, &reference_png);
            let original_pixels = image::open(reference_png).unwrap().to_rgb16();
            native(&reference, &source.with_extension("rebuilt.jpg"));
            assert_eq!(
                std::fs::read(source.with_extension("rebuilt.jpg")).unwrap(),
                input
            );
            for effort in [3, 11] {
                for route in ["auto", "forced", "coarsen-floor", "coarsen-two"] {
                    let coarsen = match route {
                        "coarsen-floor" => Some(1.0),
                        "coarsen-two" => Some(2.0),
                        _ => None,
                    };
                    let api = if let Some(scale) = coarsen {
                        jxl_encoder::jpeg::encode_jpeg_recompress_auto_codestream(
                            &input, scale, effort, None, None,
                        )
                        .unwrap()
                    } else {
                        jxl_encoder::LosslessConfig::new()
                            .with_effort(effort)
                            .encode_jpeg_transcode(&input)
                            .unwrap()
                    };
                    let expected =
                        decoded_frames::decode_current_frames(&api, "partial-grid API control");
                    let apipath =
                        directory.join(format!("{transpose}-{sampling}-{effort}-{route}-api.jxl"));
                    std::fs::write(&apipath, &api).unwrap();
                    native(&apipath, &apipath.with_extension("png"));
                    let expected_pixels = image::open(apipath.with_extension("png"))
                        .unwrap()
                        .to_rgb16();
                    if coarsen.is_none() || coarsen == Some(1.0) {
                        assert_eq!(expected_pixels, original_pixels);
                    }
                    let source = directory.join(format!(
                        "{transpose}-{sampling}-{effort}-{route}.{}",
                        if route == "forced" { "bin" } else { "jpg" }
                    ));
                    std::fs::write(&source, &input).unwrap();
                    for strategy in [
                        "",
                        "libjxl",
                        "libjxl-exact",
                        "libjxl-strict",
                        "lean-faster",
                        "zenjxl",
                        "aggressive",
                    ] {
                        for experimental in [false, true] {
                            let label = format!(
                                "{transpose}-{sampling}-{effort}-{route}-{strategy}-{experimental}"
                            );
                            let output = directory.join(format!("{label}.jxl"));
                            let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                            command
                                .arg(&source)
                                .arg(&output)
                                .args(["-e", &effort.to_string()]);
                            if route == "forced" {
                                command.arg("--lossless-jpeg");
                            }
                            if let Some(scale) = coarsen {
                                command.args(["--jpeg-coarsen", &scale.to_string()]);
                            }
                            if !strategy.is_empty() {
                                command.args(["--strategy", strategy]);
                            }
                            if experimental {
                                command.arg("--experimental");
                            }
                            let r = run_cli(&mut command, &label);
                            std::fs::write(
                                output.with_extension("cli.log"),
                                [r.stdout, r.stderr].concat(),
                            )
                            .unwrap();
                            assert!(r.status.success(), "{label}");
                            let bytes = std::fs::read(&output).unwrap();
                            assert_eq!(bytes, api, "{label}");
                            assert_eq!(
                                decoded_frames::decode_current_frames(&bytes, &label),
                                expected
                            );
                            let pixels = output.with_extension("png");
                            native(&output, &pixels);
                            assert_eq!(
                                image::open(pixels).unwrap().to_rgb16(),
                                expected_pixels,
                                "{label}: complete native samples"
                            );
                            if coarsen.is_none() {
                                let rebuilt = output.with_extension("jpg");
                                native(&output, &rebuilt);
                                assert_eq!(std::fs::read(rebuilt).unwrap(), input);
                            }
                        }
                    }
                }
            }
        }
    }
}
