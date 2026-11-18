//! Independent tiny JPEG entropy controls for nonempty JBRD scan signals.
use super::*;

// Two zero-coefficient grayscale blocks. DC category zero has code 0;
// AC EOB has code 0 and ZRL code 10. Every entropy bit is explicit.
fn zero_blocks(progressive: bool, extra_zrl: bool, zero_padding: bool) -> Vec<u8> {
    fn marker(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
        out.extend([255, id]);
        out.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        out.extend(payload);
    }
    fn entropy(out: &mut Vec<u8>, bits: &[u8], zero_padding: bool) {
        let mut byte = 0;
        for i in 0..8 {
            byte = (byte << 1)
                | if i < bits.len() {
                    bits[i]
                } else {
                    u8::from(!zero_padding)
                };
        }
        out.push(byte);
        if byte == 255 {
            out.push(0);
        }
    }
    let mut out = vec![255, 0xd8];
    let mut quant = vec![0];
    quant.extend([1; 64]);
    marker(&mut out, 0xdb, &quant);
    marker(
        &mut out,
        if progressive { 0xc2 } else { 0xc0 },
        &[8, 0, 8, 0, 16, 1, 1, 0x11, 0],
    );
    let mut dht = vec![0, 1];
    dht.extend([0; 15]);
    dht.push(0);
    dht.extend([0x10, 1, 1]);
    dht.extend([0; 14]);
    dht.extend([0, 0xf0]);
    marker(&mut out, 0xc4, &dht);
    if progressive {
        marker(&mut out, 0xda, &[1, 1, 0, 0, 0, 0]);
        entropy(&mut out, &[0, 0], zero_padding);
        marker(&mut out, 0xda, &[1, 1, 0, 1, 63, 0]);
        entropy(
            &mut out,
            if extra_zrl { &[1, 0, 0, 0] } else { &[0, 0] },
            zero_padding,
        );
    } else {
        marker(&mut out, 0xda, &[1, 1, 0, 0, 63, 0]);
        entropy(
            &mut out,
            if extra_zrl {
                &[0, 1, 0, 0, 0, 0]
            } else {
                &[0, 0, 0, 0]
            },
            zero_padding,
        );
    }
    out.extend([255, 0xd9]);
    out
}

#[test]
fn populated_scan_signals_preserve_original_entropy_bytes() {
    let root = directory();
    let mut journal = std::fs::File::create(root.join("scan-signals.tsv")).unwrap();
    let mut failures = Vec::new();
    for progressive in [false, true] {
        for zrl in [false, true] {
            for padding in [false, true] {
                let label = format!("progressive-{progressive}-zrl-{zrl}-padding-{padding}");
                let input = zero_blocks(progressive, zrl, padding);
                let path = root.join(format!("{label}.jpg"));
                std::fs::write(&path, &input).unwrap();
                let independent = image::load_from_memory(&input).unwrap().to_luma8();
                assert_eq!(independent.dimensions(), (16, 8));
                assert!(independent.pixels().all(|p| p.0 == [128]));
                let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
                let resets = parsed
                    .scan_info
                    .iter()
                    .map(|s| s.reset_points.len())
                    .sum::<usize>();
                let extras = parsed
                    .scan_info
                    .iter()
                    .map(|s| s.extra_zero_runs.len())
                    .sum::<usize>();
                writeln!(
                    journal,
                    "{label}\tresets={resets}\textra_zero_runs={extras}\tpadding={}",
                    parsed.has_zero_padding_bit
                )
                .unwrap();
                journal.flush().unwrap();
                let reference = path.with_extension("reference.jxl");
                let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
                    .arg(&path)
                    .arg(&reference)
                    .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
                    .output()
                    .unwrap();
                std::fs::write(
                    reference.with_extension("encode.log"),
                    [&result.stdout[..], &result.stderr[..]].concat(),
                )
                .unwrap();
                assert!(result.status.success(), "native source admission {label}");
                native(
                    &reference,
                    &reference.with_extension("jpg"),
                    &reference.with_extension("rebuild.log"),
                );
                assert_eq!(
                    std::fs::read(reference.with_extension("jpg")).unwrap(),
                    input
                );
                let expected = decoded_frames::decode_current_frames(
                    &std::fs::read(&reference).unwrap(),
                    &label,
                );
                for (policy, strategy, mode) in encoder_policies::policies() {
                    let config = LosslessConfig::new()
                        .with_effort(3)
                        .with_strategy(strategy)
                        .with_mode(mode);
                    let bytes = config.encode_jpeg_transcode(&input).unwrap();
                    assert_eq!(
                        bytes,
                        jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3).unwrap()
                    );
                    assert_eq!(
                        decoded_frames::decode_current_frames(&bytes, &label),
                        expected
                    );
                    let out = root.join(format!("{label}-{policy}.jxl"));
                    std::fs::write(&out, &bytes).unwrap();
                    native(
                        &out,
                        &out.with_extension("png"),
                        &out.with_extension("render.log"),
                    );
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&out)
                        .arg(out.with_extension("jpg"))
                        .arg("--num_threads=1")
                        .output()
                        .unwrap();
                    std::fs::write(
                        out.with_extension("rebuild.log"),
                        [&result.stdout[..], &result.stderr[..]].concat(),
                    )
                    .unwrap();
                    let native_exact = result.status.success()
                        && std::fs::read(out.with_extension("jpg")).unwrap() == input;
                    let rust = zensim_decoder::reconstruct_jpeg(&bytes);
                    let rust_exact = matches!(&rust,Ok(Some(v))if v==&input);
                    if !native_exact || !rust_exact {
                        failures.push(format!(
                            "{label}-{policy} native={native_exact} rust={rust_exact}"
                        ));
                    }
                }
                // Reach assertions remain after the full before evidence is retained.
                if progressive && resets == 0 {
                    failures.push(format!("{label}: reset signal not observed"));
                }
                if zrl && extras == 0 {
                    failures.push(format!("{label}: extra-ZRL signal not observed"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        root.display()
    );
}

#[test]
fn native_nonzero_long_code_restart_and_edge_scans_preserve_samples_and_jpeg() {
    let cases = [
        (
            "nonzero-before-zrl",
            include_str!("../fixtures/jbrd/scan_nonzero-before-zrl.jpg.hex"),
        ),
        (
            "zrl-before-nonzero",
            include_str!("../fixtures/jbrd/scan_zrl-before-nonzero.jpg.hex"),
        ),
        (
            "three-zrl",
            include_str!("../fixtures/jbrd/scan_three-zrl.jpg.hex"),
        ),
        (
            "zrl-finishes-block",
            include_str!("../fixtures/jbrd/scan_zrl-finishes-block.jpg.hex"),
        ),
        (
            "long-codes-nonzero-before",
            include_str!("../fixtures/jbrd/scan_long-codes-nonzero-before.jpg.hex"),
        ),
        (
            "long-codes-zrl-before-nonzero",
            include_str!("../fixtures/jbrd/scan_long-codes-zrl-before-nonzero.jpg.hex"),
        ),
        (
            "gray-absolute22",
            include_str!("../fixtures/jbrd/scan_gray-absolute22.jpg.hex"),
        ),
        (
            "interleaved-last",
            include_str!("../fixtures/jbrd/scan_interleaved-last.jpg.hex"),
        ),
        (
            "interleaved-first-long",
            include_str!("../fixtures/jbrd/scan_interleaved-first-long.jpg.hex"),
        ),
        (
            "edge-padding-luma",
            include_str!("../fixtures/jbrd/scan_edge-padding-luma.jpg.hex"),
        ),
        (
            "edge-visible-chroma-long",
            include_str!("../fixtures/jbrd/scan_edge-visible-chroma-long.jpg.hex"),
        ),
        (
            "slow11-nonzero-before",
            include_str!("../fixtures/jbrd/scan_slow11-nonzero-before.jpg.hex"),
        ),
        (
            "slow11-zrl-before-nonzero",
            include_str!("../fixtures/jbrd/scan_slow11-zrl-before-nonzero.jpg.hex"),
        ),
    ];
    let root = directory();
    for (label, text) in cases {
        let input: Vec<u8> = text
            .trim()
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        let case = root.join(label);
        std::fs::create_dir(&case).unwrap();
        let source = case.join("source.jpg");
        std::fs::write(&source, &input).unwrap();
        let reference = case.join("reference.jxl");
        let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .arg(&source)
            .arg(&reference)
            .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
            .output()
            .unwrap();
        std::fs::write(
            case.join("reference-encode.log"),
            [result.stdout, result.stderr].concat(),
        )
        .unwrap();
        assert!(result.status.success());
        native(
            &reference,
            &case.join("reference.png"),
            &case.join("reference-render.log"),
        );
        native(
            &reference,
            &case.join("reference.jpg"),
            &case.join("reference-rebuild.log"),
        );
        assert_eq!(std::fs::read(case.join("reference.jpg")).unwrap(), input);
        let expected_native = image::open(case.join("reference.png")).unwrap().to_rgb16();
        let expected_primary = decoded_frames::decode_current_frames(
            &std::fs::read(&reference).unwrap(),
            &format!("native reference {label}"),
        );
        let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
        assert_eq!(parsed.scan_info.len(), 1);
        for (policy, strategy, mode) in encoder_policies::policies() {
            let config = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy)
                .with_mode(mode);
            let bytes = config.encode_jpeg_transcode(&input).unwrap();
            assert_eq!(
                bytes,
                config
                    .encode_jpeg_transcode_with_stop(&input, &enough::Unstoppable)
                    .unwrap()
            );
            assert_eq!(
                bytes,
                jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3).unwrap()
            );
            let path = case.join(format!("{policy}.jxl"));
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(
                decoded_frames::decode_current_frames(&bytes, &format!("encoder {label}")),
                expected_primary
            );
            native(
                &path,
                &path.with_extension("png"),
                &path.with_extension("render.log"),
            );
            assert_eq!(
                image::open(path.with_extension("png")).unwrap().to_rgb16(),
                expected_native,
                "{label}-{policy}"
            );
            native(
                &path,
                &path.with_extension("jpg"),
                &path.with_extension("rebuild.log"),
            );
            assert_eq!(
                std::fs::read(path.with_extension("jpg")).unwrap(),
                input,
                "{label}-{policy}"
            );
            assert_eq!(
                zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
                input,
                "{label}-{policy}"
            );
        }
    }
}
