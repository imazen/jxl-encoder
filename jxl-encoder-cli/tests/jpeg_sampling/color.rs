//! APP0 semantics and RGB sampling admission at the actual CLI boundary.
use super::grid::{native, run_cli};
use super::*;
fn insert_app0(input: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = input[..2].to_vec();
    out.extend([255, 0xe0]);
    out.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    out.extend(payload);
    out.extend(&input[2..]);
    out
}
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("jpeg-cli-color");
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
fn strip_adobe(input: &[u8]) -> Vec<u8> {
    let mut out = input[..2].to_vec();
    let mut at = 2;
    loop {
        assert_eq!(input[at], 255);
        let id = input[at + 1];
        let end = at + 2 + usize::from(u16::from_be_bytes([input[at + 2], input[at + 3]]));
        if id != 0xee {
            out.extend(&input[at..end]);
        }
        if id == 0xda {
            out.extend(&input[end..]);
            break;
        }
        at = end;
    }
    out
}
#[test]
fn unrelated_app0_and_rgb_sampling_cli_contract() {
    let root = directory();
    let fixture = include_bytes!("../../../jxl-encoder/tests/fixtures/jbrd/rgb_adobe.jpg");
    for adobe in [false, true] {
        let baseline = if adobe {
            fixture.to_vec()
        } else {
            strip_adobe(fixture)
        };
        for (payload_id, payload) in [b"".as_slice(), b"Private APP0 data"]
            .into_iter()
            .enumerate()
        {
            let input = insert_app0(&baseline, payload);
            for route in ["auto", "forced", "coarsen-floor", "coarsen-two"] {
                let encode = |data: &[u8]| {
                    if route.starts_with("coarsen") {
                        jxl_encoder::jpeg::encode_jpeg_recompress_auto_codestream(
                            data,
                            if route == "coarsen-two" { 2.0 } else { 1.0 },
                            3,
                            None,
                            None,
                        )
                        .unwrap()
                    } else {
                        jxl_encoder::LosslessConfig::new()
                            .with_effort(3)
                            .encode_jpeg_transcode(data)
                            .unwrap()
                    }
                };
                let expected = encode(&input);
                assert_eq!(
                    decoded_frames::decode_current_frames(&expected, "APP0 API"),
                    decoded_frames::decode_current_frames(&encode(&baseline), "baseline RGB API")
                );
                let apipath = root.join(format!("{adobe}-{payload_id}-{route}-api.jxl"));
                std::fs::write(&apipath, &expected).unwrap();
                native(&apipath, &apipath.with_extension("png"));
                let expected_pixels = image::open(apipath.with_extension("png"))
                    .unwrap()
                    .to_rgb16();
                let source = root.join(format!(
                    "{adobe}-{payload_id}-{route}.{}",
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
                        let label =
                            format!("color-{adobe}-{payload_id}-{route}-{strategy}-{experimental}");
                        let output = root.join(format!("{label}.jxl"));
                        let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                        command.arg(&source).arg(&output).args(["-e", "3"]);
                        if route == "forced" {
                            command.arg("--lossless-jpeg");
                        }
                        if route.starts_with("coarsen") {
                            command.args([
                                "--jpeg-coarsen",
                                if route == "coarsen-two" { "2" } else { "1" },
                            ]);
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
                        assert_eq!(bytes, expected, "{label}");
                        assert_eq!(
                            decoded_frames::decode_current_frames(&bytes, &label),
                            decoded_frames::decode_current_frames(&expected, "APP0 API")
                        );
                        native(&output, &output.with_extension("png"));
                        assert_eq!(
                            image::open(output.with_extension("png"))
                                .unwrap()
                                .to_rgb16(),
                            expected_pixels
                        );
                        if !route.starts_with("coarsen") {
                            native(&output, &output.with_extension("jpg"));
                            assert_eq!(std::fs::read(output.with_extension("jpg")).unwrap(), input);
                        }
                    }
                }
            }
        }
    }
    // This exact source is decoded independently by the API contract and
    // the caller-selected libjpeg-turbo source oracle. CLI image features do
    // not supply a second JPEG decoder.
    let input: Vec<u8> =
        include_str!("../../../jxl-encoder/tests/fixtures/jbrd/rgb_absolute22.jpg.hex")
            .trim()
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
    for app0 in [false, true] {
        let input = if app0 {
            insert_app0(&input, b"Private APP0 data")
        } else {
            input.clone()
        };
        for route in ["auto", "forced", "coarsen-floor", "coarsen-two"] {
            let source = root.join(format!(
                "refusal-{app0}-{route}.{}",
                if route == "forced" { "bin" } else { "jpg" }
            ));
            std::fs::write(&source, input.as_slice()).unwrap();
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
                    let label = format!("color-refusal-{app0}-{route}-{strategy}-{experimental}");
                    let output = root.join(format!("{label}.jxl"));
                    std::fs::write(&output, b"existing output").unwrap();
                    let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                    command.arg(&source).arg(&output).args(["-e", "3"]);
                    if route == "forced" {
                        command.arg("--lossless-jpeg");
                    }
                    if route.starts_with("coarsen") {
                        command.args([
                            "--jpeg-coarsen",
                            if route == "coarsen-two" { "2" } else { "1" },
                        ]);
                    }
                    if !strategy.is_empty() {
                        command.args(["--strategy", strategy]);
                    }
                    if experimental {
                        command.arg("--experimental");
                    }
                    let r = run_cli(&mut command, &label);
                    let diagnostic = String::from_utf8_lossy(&r.stderr);
                    assert!(!r.status.success(), "{label}");
                    assert!(
                        diagnostic.contains("RGB JPEG reconstruction requires 1x1 sampling"),
                        "{diagnostic}"
                    );
                    assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
                    std::fs::write(
                        output.with_extension("cli.log"),
                        [r.stdout, r.stderr].concat(),
                    )
                    .unwrap();
                }
            }
        }
    }
}
