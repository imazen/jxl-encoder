//! Prefix-only APP1 through the CLI's two JPEG input routes.
use super::*;

#[test]
fn prefix_only_app1_metadata_keeps_original_jpeg_bytes() {
    let root = jxl_encoder::test_helpers::output_dir("jpeg-cli-metadata");
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
    for (name, prefix, base) in [
        (
            "Exif",
            b"Exif\0\0".as_slice(),
            include_bytes!("../../../jxl-encoder/tests/fixtures/jbrd/meta_d_exif_orient_big.jpg")
                .as_slice(),
        ),
        (
            "XMP",
            b"http://ns.adobe.com/xap/1.0/\0".as_slice(),
            include_bytes!("../../../jxl-encoder/tests/fixtures/jbrd/meta_a_xmp.jpg").as_slice(),
        ),
    ] {
        let mut input = base[..2].to_vec();
        input.extend([255, 0xe1]);
        input.extend(u16::try_from(2 + prefix.len()).unwrap().to_be_bytes());
        input.extend(prefix);
        input.extend(&base[2..]);
        let expected = jxl_encoder::LosslessConfig::new()
            .with_effort(3)
            .encode_jpeg_transcode(&input)
            .unwrap();
        let reference = decoded_frames::decode_current_frames(&expected, name);
        for forced in [false, true] {
            let source = root.join(format!(
                "{name}-{forced}.{}",
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
                    let label = format!("{name}-{forced}-{strategy}-{experimental}");
                    let output = root.join(format!("{label}.jxl"));
                    let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                    command.arg(&source).arg(&output).args(["-e", "3"]);
                    if forced {
                        command.arg("--lossless-jpeg");
                    }
                    if !strategy.is_empty() {
                        command.args(["--strategy", strategy]);
                    }
                    if experimental {
                        command.arg("--experimental");
                    }
                    let result = command.output().unwrap();
                    std::fs::write(
                        root.join(format!("{label}.log")),
                        [&result.stdout[..], &result.stderr[..]].concat(),
                    )
                    .unwrap();
                    assert!(
                        result.status.success(),
                        "{label}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    let bytes = std::fs::read(&output).unwrap();
                    assert_eq!(bytes, expected, "{label} CLI/API output");
                    assert_eq!(
                        decoded_frames::decode_current_frames(&bytes, &label),
                        reference
                    );
                    let reconstructed = root.join(format!("{label}.jpg"));
                    let native = Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&output)
                        .arg(&reconstructed)
                        .arg("--num_threads=1")
                        .output()
                        .unwrap();
                    std::fs::write(root.join(format!("{label}-native.log")), &native.stderr)
                        .unwrap();
                    assert!(native.status.success(), "{label}");
                    assert!(
                        std::fs::read(reconstructed).unwrap() == input,
                        "{label} exact original JPEG"
                    );
                }
            }
        }
    }
}
