//! Distinct empty and zero-symbol DHT records through both CLI JPEG routes.
use super::*;

#[test]
fn empty_and_zero_symbol_dht_keep_original_jpeg_bytes() {
    let root = jxl_encoder::test_helpers::output_dir("jpeg-cli-dht");
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
    for (name, input) in dht_inputs() {
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

fn dht_inputs() -> Vec<(&'static str, Vec<u8>)> {
    let base = include_bytes!("../../../jxl-encoder/tests/fixtures/jbrd/base_b_444.jpg");
    let mut at = 2;
    while base[at + 1] != 0xda {
        assert_eq!(base[at], 255);
        at += 2 + usize::from(u16::from_be_bytes([base[at + 2], base[at + 3]]));
    }
    let mut rows = Vec::new();
    for (name, payload) in [
        ("empty", Vec::new()),
        ("zero-dc3", {
            let mut v = vec![3];
            v.extend([0; 16]);
            v
        }),
    ] {
        let mut input = base[..at].to_vec();
        input.extend([255, 0xc4]);
        input.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        input.extend(payload);
        input.extend(&base[at..]);
        rows.push((name, input));
    }
    rows
}
