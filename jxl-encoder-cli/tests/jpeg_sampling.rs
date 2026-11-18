//! Grayscale sampling through automatic and explicitly selected JPEG CLI input.
#![cfg(feature = "jpeg-reencoding")]
use std::process::Command;
#[path = "../../jxl-encoder/tests/support/decoded_frames.rs"]
mod decoded_frames;

#[test]
fn grayscale_sampling_survives_both_cli_jpeg_routes() {
    let root = jxl_encoder::test_helpers::output_dir("jpeg-cli-sampling");
    let directory = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let original = include_bytes!("../../jxl-encoder/tests/fixtures/jbrd/gray_a.jpg");
    let mut sof = 2;
    while original[sof + 1] != 0xc0 {
        assert_eq!(original[sof], 255);
        assert_ne!(original[sof + 1], 0xda);
        sof += 2 + usize::from(u16::from_be_bytes([original[sof + 2], original[sof + 3]]));
    }
    assert_eq!(original[sof + 9], 1);
    for sampling in [0x11, 0x21, 0x12, 0x22] {
        let mut input = original.to_vec();
        input[sof + 11] = sampling;
        let expected = jxl_encoder::LosslessConfig::new()
            .with_effort(3)
            .encode_jpeg_transcode(&input)
            .unwrap();
        let reference = decoded_frames::decode_current_frames(&expected, "CLI source control");
        for forced in [false, true] {
            let source = directory.join(format!(
                "{sampling}-{forced}.{}",
                if forced { "bin" } else { "jpg" }
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
                    let label = format!("{sampling}-{forced}-{strategy}-{experimental}");
                    let output = directory.join(format!("{label}.jxl"));
                    let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                    command.arg(&source).arg(&output).args(["-e", "3"]);
                    if !strategy.is_empty() {
                        command.args(["--strategy", strategy]);
                    }
                    if forced {
                        command.arg("--lossless-jpeg");
                    }
                    if experimental {
                        command.arg("--experimental");
                    }
                    let result = command.output().unwrap();
                    std::fs::write(
                        directory.join(format!("{label}.log")),
                        [&result.stdout[..], &result.stderr[..]].concat(),
                    )
                    .unwrap();
                    assert!(
                        result.status.success(),
                        "{label}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    let bytes = std::fs::read(&output).unwrap();
                    assert_eq!(bytes, expected, "{label} CLI/API bytes");
                    assert_eq!(
                        decoded_frames::decode_current_frames(&bytes, &label),
                        reference
                    );
                    let rebuilt = directory.join(format!("{label}.jpg"));
                    let native = Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&output)
                        .arg(&rebuilt)
                        .arg("--num_threads=1")
                        .output()
                        .unwrap();
                    std::fs::write(
                        directory.join(format!("{label}-native.log")),
                        [&native.stdout[..], &native.stderr[..]].concat(),
                    )
                    .unwrap();
                    assert!(native.status.success(), "{label} native reconstruction");
                    assert_eq!(
                        std::fs::read(&rebuilt).unwrap(),
                        input,
                        "{label} original JPEG"
                    );
                }
            }
        }
    }
}

#[path = "jpeg_sampling/metadata.rs"]
mod metadata;

#[path = "jpeg_sampling/dht.rs"]
mod dht;

#[path = "jpeg_sampling/grid.rs"]
mod grid;
