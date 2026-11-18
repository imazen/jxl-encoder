//! RGB sampling admission controls.
use super::*;

fn rgb_subsampled() -> Vec<u8> {
    let mut input = include_bytes!("../fixtures/jbrd/base_a_420.jpg").to_vec();
    for (offset, marker, _) in header_segments(&input) {
        if marker == 0xc0 {
            assert_eq!(input[offset + 9], 3);
            for (i, id) in b"RGB".iter().enumerate() {
                input[offset + 10 + i * 3] = *id;
            }
        }
        if marker == 0xda {
            assert_eq!(input[offset + 4], 3);
            for (i, id) in b"RGB".iter().enumerate() {
                input[offset + 5 + i * 2] = *id;
            }
        }
    }
    let mut output = input[..2].to_vec();
    for (offset, marker, length) in header_segments(&input) {
        if marker != 0xe0 {
            output.extend_from_slice(&input[offset..offset + length]);
        }
        if marker == 0xda {
            output.extend_from_slice(&input[offset + length..]);
        }
    }
    output
}
#[test]
fn rgb_subsampling_is_refused_without_panicking() {
    let root = directory();
    let mut failures = Vec::new();
    for (label, input) in [
        ("relative420", rgb_subsampled()),
        ("absolute22", rgb_absolute22()),
        ("absolute12", rgb_absolute(1, 2)),
        ("absolute21", rgb_absolute(2, 1)),
    ] {
        let root = root.join(label);
        std::fs::create_dir(&root).unwrap();
        let independent = image::load_from_memory(&input).unwrap();
        std::fs::write(root.join("rgb-subsampled.jpg"), &input).unwrap();
        let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
        assert_eq!(
            (independent.width(), independent.height()),
            (parsed.width, parsed.height)
        );
        assert_eq!(parsed.components.len(), 3);
        assert_eq!(
            parsed.components[0].h_samp_factor,
            if label == "absolute12" { 1 } else { 2 }
        );
        assert_eq!(
            parsed.components[0].v_samp_factor,
            if label == "absolute21" { 1 } else { 2 }
        );
        assert_eq!(
            parsed.components[1].h_samp_factor,
            if label == "absolute22" || label == "absolute21" {
                2
            } else {
                1
            }
        );
        for (policy, strategy, mode) in encoder_policies::policies() {
            let config = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy)
                .with_mode(mode);
            for parsed_entry in [false, true] {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if parsed_entry {
                        jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3)
                            .map_err(|e| e.to_string())
                    } else {
                        config
                            .encode_jpeg_transcode(&input)
                            .map_err(|e| e.to_string())
                    }
                }));
                match result {
                    Ok(Err(error)) => {
                        assert!(
                            error.contains("RGB JPEG reconstruction requires 1x1 sampling"),
                            "{error}"
                        );
                        std::fs::write(
                            root.join(format!("{policy}-{parsed_entry}-refusal.txt")),
                            error,
                        )
                        .unwrap();
                    }
                    Ok(Ok(bytes)) => {
                        let path = root.join(format!("{policy}-{parsed_entry}-accepted.jxl"));
                        std::fs::write(&path, &bytes).unwrap();
                        let rust = std::panic::catch_unwind(|| {
                            decoded_frames::decode_current_frames(&bytes, "RGB subsampling before")
                        });
                        let native =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(&path)
                                .arg(path.with_extension("png"))
                                .args(["--num_threads=1", "--bits_per_sample=16"])
                                .output()
                                .unwrap();
                        std::fs::write(
                            path.with_extension("log"),
                            [&native.stdout[..], &native.stderr[..]].concat(),
                        )
                        .unwrap();
                        std::fs::write(
                            path.with_extension("verdict.txt"),
                            format!(
                                "primary_rendered={} native_rendered={}\n",
                                rust.is_ok(),
                                native.status.success()
                            ),
                        )
                        .unwrap();
                        failures.push(format!(
                            "{policy} parsed={parsed_entry}: accepted unrepresentable RGB sampling"
                        ));
                    }
                    Err(_) => failures.push(format!("{policy} parsed={parsed_entry}: panicked")),
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

fn rgb_absolute22() -> Vec<u8> {
    let expected: Vec<u8> = include_str!("../fixtures/jbrd/rgb_absolute22.jpg.hex")
        .trim()
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect();
    let source = rgb_absolute(2, 2);
    assert_eq!(source, expected);
    source
}
fn rgb_absolute(h: u8, v: u8) -> Vec<u8> {
    fn marker(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
        out.extend([255, id]);
        out.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        out.extend(payload);
    }
    let mut out = vec![255, 0xd8];
    let mut quant = vec![0];
    quant.extend([1; 64]);
    marker(&mut out, 0xdb, &quant);
    marker(
        &mut out,
        0xc0,
        &[
            8,
            0,
            8,
            0,
            16,
            3,
            b'R',
            h * 16 + v,
            0,
            b'G',
            h * 16 + v,
            0,
            b'B',
            h * 16 + v,
            0,
        ],
    );
    let mut tables = vec![0, 1];
    tables.extend([0; 15]);
    tables.push(0);
    tables.extend([0x10, 1]);
    tables.extend([0; 15]);
    tables.push(0);
    marker(&mut out, 0xc4, &tables);
    if h * v * 3 > 10 {
        // Separate component scans avoid an interleaved 12-block MCU.
        for id in b"RGB" {
            marker(&mut out, 0xda, &[1, *id, 0, 0, 63, 0]);
            out.push(0x0f); // Two visible DC0/EOB blocks; four one padding bits.
        }
    } else {
        marker(&mut out, 0xda, &[3, b'R', 0, b'G', 0, b'B', 0, 0, 63, 0]);
        let blocks = 16usize.div_ceil(8 * usize::from(h))
            * 8usize.div_ceil(8 * usize::from(v))
            * usize::from(h)
            * usize::from(v)
            * 3;
        let bits = blocks * 2;
        out.resize(out.len() + bits / 8, 0);
        if !bits.is_multiple_of(8) {
            out.push((1 << (8 - bits % 8)) - 1);
        }
    }
    out.extend([255, 0xd9]);
    out
}

#[test]
fn rgb_sampling_refusal_reaches_bare_recompression_and_nested_callers() {
    let root = directory();
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
    for (label, input) in [
        ("relative420", rgb_subsampled()),
        ("absolute22", rgb_absolute22()),
        ("absolute12", rgb_absolute(1, 2)),
        ("absolute21", rgb_absolute(2, 1)),
    ] {
        for app0 in [false, true] {
            let input = if app0 {
                let payload = b"Private APP0 data";
                let mut result = input[..2].to_vec();
                result.extend([255, 0xe0]);
                result.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
                result.extend(payload);
                result.extend(&input[2..]);
                result
            } else {
                input.clone()
            };
            let case = root.join(format!("{label}-app0-{app0}"));
            std::fs::create_dir(&case).unwrap();
            std::fs::write(case.join("original.jpg"), &input).unwrap();
            let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
            let mut secondary = input[..2].to_vec();
            secondary.extend([255, 0xe2]);
            secondary.extend(
                u16::try_from(2 + namespace.len() + metadata.len())
                    .unwrap()
                    .to_be_bytes(),
            );
            secondary.extend(namespace);
            secondary.extend(&metadata);
            secondary.extend(&input[2..]);
            let mut nested = include_bytes!("../fixtures/jbrd/base_a_420.jpg").to_vec();
            nested.extend(secondary);
            let refused = |result: Result<Vec<u8>, String>, route: &str| {
                let error = result.expect_err(route);
                assert!(
                    error.contains("RGB JPEG reconstruction requires 1x1 sampling"),
                    "{route}: {error}"
                );
                std::fs::write(case.join(format!("{route}.refusal.txt")), error).unwrap();
            };
            refused(
                jpeg::encode_jpeg_to_jxl(&parsed).map_err(|e| e.to_string()),
                "bare-default",
            );
            refused(
                jpeg::encode_jpeg_to_jxl_container(&parsed).map_err(|e| e.to_string()),
                "container-default",
            );
            for effort in [1, 3, 7, 8, 9, 11] {
                refused(
                    jpeg::encode_jpeg_to_jxl_with_effort(&parsed, effort)
                        .map_err(|e| e.to_string()),
                    &format!("bare-e{effort}"),
                );
                refused(
                    jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, effort)
                        .map_err(|e| e.to_string()),
                    &format!("container-e{effort}"),
                );
            }
            for (policy, strategy, mode) in encoder_policies::policies() {
                let config = LosslessConfig::new()
                    .with_effort(3)
                    .with_strategy(strategy)
                    .with_mode(mode);
                refused(
                    config
                        .encode_jpeg_transcode_codestream(&input)
                        .map_err(|e| e.to_string()),
                    &format!("{policy}-bare"),
                );
                refused(
                    config
                        .encode_jpeg_transcode_codestream_with_stop(&input, &enough::Unstoppable)
                        .map_err(|e| e.to_string()),
                    &format!("{policy}-bare-stop"),
                );
                refused(
                    config
                        .encode_jpeg_transcode_with_stop(&input, &enough::Unstoppable)
                        .map_err(|e| e.to_string()),
                    &format!("{policy}-stop"),
                );
                refused(
                    config
                        .encode_jpeg_transcode(&nested)
                        .map_err(|e| e.to_string()),
                    &format!("{policy}-nested"),
                );
                refused(
                    config
                        .encode_jpeg_transcode_with_stop(&nested, &enough::Unstoppable)
                        .map_err(|e| e.to_string()),
                    &format!("{policy}-nested-stop"),
                );
            }
            for scale in [1.0, 2.0] {
                refused(
                    jpeg::encode_jpeg_recompress_auto_codestream(
                        &input,
                        scale,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    )
                    .map_err(|e| e.to_string()),
                    &format!("auto-{scale}"),
                );
                refused(
                    jpeg::encode_jpeg_recompress_codestream(
                        &input,
                        scale,
                        0.5,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    )
                    .map_err(|e| e.to_string()),
                    &format!("explicit-{scale}"),
                );
                refused(
                    jpeg::encode_jpeg_recompress_planar_codestream(
                        &input,
                        scale,
                        0.5,
                        scale,
                        0.5,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    )
                    .map_err(|e| e.to_string()),
                    &format!("planar-{scale}"),
                );
            }
        }
    }
}
