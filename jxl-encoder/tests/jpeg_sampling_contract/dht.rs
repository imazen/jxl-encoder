//! Admission and exact reconstruction for the conditional JPEG review leads.
use super::*;

pub(super) fn before_sos(data: &[u8], marker: u8, payload: &[u8]) -> Vec<u8> {
    let offset = header_segments(data)
        .into_iter()
        .find(|(_, m, _)| *m == 0xda)
        .unwrap()
        .0;
    let mut result = data[..offset].to_vec();
    result.extend([255, marker]);
    result.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    result.extend(payload);
    result.extend_from_slice(&data[offset..]);
    result
}
fn check_unusual(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let mut journal = std::fs::File::create(root.join("unusual.tsv")).unwrap();
    let mut failures = Vec::new();
    let base = include_bytes!("../fixtures/jbrd/base_b_444.jpg");
    let mut zero_dc = vec![3u8];
    zero_dc.extend([0u8; 16]);
    let mut zero_ac = vec![0x13u8];
    zero_ac.extend([0u8; 16]);
    let cases = [
        ("ordinary", base.to_vec()),
        ("empty-dht", before_sos(base, 0xc4, &[])),
        ("dc3-zero-symbols", before_sos(base, 0xc4, &zero_dc)),
        ("ac3-zero-symbols", before_sos(base, 0xc4, &zero_ac)),
        (
            "progressive-restarts",
            include_bytes!("../fixtures/jbrd/prog_a_rst_444.jpg").to_vec(),
        ),
        (
            "baseline-restarts",
            include_bytes!("../fixtures/jbrd/base_a_rstblk_420.jpg").to_vec(),
        ),
    ];
    for (name, input) in cases {
        let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
        writeln!(journal,"{name}\tparsed=true\tscans={}\tresets={}\textra_zero_runs={}\tnondefault_padding={}\tpadding_bits={}",
                 parsed.scan_info.len(), parsed.scan_info.iter().map(|s|s.reset_points.len()).sum::<usize>(),
                 parsed.scan_info.iter().map(|s|s.extra_zero_runs.len()).sum::<usize>(),
                 parsed.has_zero_padding_bit, parsed.padding_bits.len()).unwrap();
        journal.flush().unwrap();
        let reference_path = root.join(format!("{name}-reference.jxl"));
        let input_path = root.join(format!("{name}-source.jpg"));
        std::fs::write(&input_path, &input).unwrap();
        let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .arg(&input_path)
            .arg(&reference_path)
            .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
            .output()
            .unwrap();
        std::fs::write(
            root.join(format!("{name}-reference.log")),
            [&result.stdout[..], &result.stderr[..]].concat(),
        )
        .unwrap();
        assert!(result.status.success());
        native(
            &reference_path,
            &root.join(format!("{name}-reference.png")),
            &root.join(format!("{name}-reference-render.log")),
        );
        let native_pixels = image::open(root.join(format!("{name}-reference.png")))
            .unwrap()
            .to_rgb16();
        let primary_pixels =
            decoded_frames::decode_current_frames(&std::fs::read(&reference_path).unwrap(), name);
        for (policy, strategy, mode) in &policies {
            let out = root.join(format!("{name}-{policy}"));
            std::fs::create_dir(&out).unwrap();
            std::fs::write(out.join("input.jpg"), &input).unwrap();
            let config = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy.clone())
                .with_mode(*mode);
            let bytes = config.encode_jpeg_transcode(&input).unwrap();
            assert_eq!(
                bytes,
                jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3).unwrap()
            );
            assert_eq!(
                bytes,
                config
                    .encode_jpeg_transcode_with_stop(&input, &enough::Unstoppable)
                    .unwrap()
            );
            let path = out.join("input.jxl");
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(
                decoded_frames::decode_current_frames(&bytes, name),
                primary_pixels
            );
            let bare = config.encode_jpeg_transcode_codestream(&input).unwrap();
            assert_eq!(
                bare,
                config
                    .encode_jpeg_transcode_codestream_with_stop(&input, &enough::Unstoppable)
                    .unwrap()
            );
            assert_eq!(
                bare,
                jpeg::encode_jpeg_to_jxl_with_effort(&parsed, 3).unwrap()
            );
            let default_bare = jpeg::encode_jpeg_to_jxl(&parsed).unwrap();
            assert_eq!(
                default_bare,
                jpeg::encode_jpeg_to_jxl_with_effort(&parsed, 7).unwrap()
            );
            assert_eq!(
                decoded_frames::decode_current_frames(&default_bare, name),
                primary_pixels
            );
            let default_container = jpeg::encode_jpeg_to_jxl_container(&parsed).unwrap();
            assert_eq!(
                default_container,
                jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 7).unwrap()
            );
            assert_eq!(
                zensim_decoder::reconstruct_jpeg(&default_container)
                    .unwrap()
                    .unwrap(),
                input
            );
            assert_eq!(
                decoded_frames::decode_current_frames(&bare, name),
                primary_pixels
            );
            let bare_path = out.join("bare.jxl");
            std::fs::write(&bare_path, &bare).unwrap();
            native(&bare_path, &out.join("bare.png"), &out.join("bare.log"));
            assert_eq!(
                image::open(out.join("bare.png")).unwrap().to_rgb16(),
                native_pixels
            );
            native(&path, &out.join("render.png"), &out.join("render.log"));
            assert_eq!(
                image::open(out.join("render.png")).unwrap().to_rgb16(),
                native_pixels
            );
            let rust = zensim_decoder::reconstruct_jpeg(&bytes);
            let rust_exact = matches!(&rust, Ok(Some(data)) if data == &input);
            if let Ok(Some(data)) = &rust {
                std::fs::write(out.join("rust.jpg"), data).unwrap();
            }
            let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(&path)
                .arg(out.join("native.jpg"))
                .arg("--num_threads=1")
                .output()
                .unwrap();
            std::fs::write(
                out.join("reconstruct.log"),
                [&result.stdout[..], &result.stderr[..]].concat(),
            )
            .unwrap();
            let native_exact =
                result.status.success() && std::fs::read(out.join("native.jpg")).unwrap() == input;
            writeln!(
                journal,
                "{name}\t{policy}\trust_exact={rust_exact}\tnative_exact={native_exact}"
            )
            .unwrap();
            journal.flush().unwrap();
            if !rust_exact || !native_exact {
                failures.push(format!(
                    "{name} {policy}: Rust={rust_exact}, native={native_exact}"
                ));
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
fn strict_unusual_huffman_and_scan_metadata_reconstructs() {
    check_unusual(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_unusual_huffman_and_scan_metadata_reconstructs() {
    check_unusual(encoder_policies::policies());
}
/// Free DC table zero by renaming the grayscale image's active DC table to one.
pub(super) fn empty_and_zero_dc0_pair() -> (Vec<u8>, Vec<u8>) {
    let mut base = include_bytes!("../fixtures/jbrd/gray_a.jpg").to_vec();
    let segments = header_segments(&base);
    let mut renamed = 0;
    for (offset, marker, length) in segments {
        if marker == 0xc4 {
            let mut at = offset + 4;
            while at < offset + length {
                let count = base[at + 1..at + 17]
                    .iter()
                    .map(|&v| usize::from(v))
                    .sum::<usize>();
                assert_ne!(base[at], 1, "DC1 must be unused before renaming");
                if base[at] == 0 {
                    base[at] = 1;
                    renamed += 1;
                }
                at += 17 + count;
            }
            assert_eq!(at, offset + length);
        }
        if marker == 0xda {
            assert_eq!(base[offset + 4], 1);
            assert_eq!(base[offset + 6], 0);
            base[offset + 6] = 0x10;
        }
    }
    assert_eq!(renamed, 1);
    let empty = before_sos(&base, 0xc4, &[]);
    let zero = before_sos(&base, 0xc4, &[0; 17]);
    assert_eq!(zero.len(), empty.len() + 17);
    (empty, zero)
}

#[test]
fn empty_segment_and_unused_zero_symbol_dc0_reconstruct_their_distinct_bytes() {
    let root = directory();
    let (empty, zero) = empty_and_zero_dc0_pair();
    let parsed_empty = jpeg::read_jpeg(&empty, None, None).unwrap();
    let parsed_zero = jpeg::read_jpeg(&zero, None, None).unwrap();
    std::fs::write(root.join("parsed-empty.txt"), format!("{parsed_empty:?}")).unwrap();
    std::fs::write(root.join("parsed-zero.txt"), format!("{parsed_zero:?}")).unwrap();
    std::fs::write(
        root.join("legacy-model-equality.txt"),
        format!(
            "{}\n",
            format!("{parsed_empty:?}") == format!("{parsed_zero:?}")
        ),
    )
    .unwrap();
    let mut failures = Vec::new();
    for (name, input) in [("empty", empty), ("zero", zero)] {
        let path = root.join(format!("{name}.jpg"));
        std::fs::write(&path, &input).unwrap();
        let reference_path = root.join(format!("{name}-reference.jxl"));
        let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .arg(&path)
            .arg(&reference_path)
            .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
            .output()
            .unwrap();
        std::fs::write(
            root.join(format!("{name}-reference-encode.log")),
            [&result.stdout[..], &result.stderr[..]].concat(),
        )
        .unwrap();
        assert!(
            result.status.success(),
            "{name} pinned native source admission"
        );
        native(
            &reference_path,
            &root.join(format!("{name}-reference.jpg")),
            &root.join(format!("{name}-reference-reconstruct.log")),
        );
        assert!(
            std::fs::read(root.join(format!("{name}-reference.jpg"))).unwrap() == input,
            "{name} native original control"
        );
        let reference =
            decoded_frames::decode_current_frames(&std::fs::read(&reference_path).unwrap(), name);
        for (policy, strategy, mode) in encoder_policies::policies() {
            let config = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy)
                .with_mode(mode);
            let bytes = config.encode_jpeg_transcode(&input).unwrap();
            let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
            assert_eq!(
                bytes,
                jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3).unwrap()
            );
            assert_eq!(
                bytes,
                config
                    .encode_jpeg_transcode_with_stop(&input, &enough::Unstoppable)
                    .unwrap()
            );
            assert_eq!(
                decoded_frames::decode_current_frames(&bytes, name),
                reference
            );
            let path = root.join(format!("{name}-{policy}.jxl"));
            std::fs::write(&path, &bytes).unwrap();
            native(
                &path,
                &path.with_extension("png"),
                &path.with_extension("render.log"),
            );
            native(
                &path,
                &path.with_extension("jpg"),
                &path.with_extension("reconstruct.log"),
            );
            let rust = zensim_decoder::reconstruct_jpeg(&bytes);
            let rust_exact = matches!(&rust,Ok(Some(v))if v==&input);
            let native_exact = std::fs::read(path.with_extension("jpg")).unwrap() == input;
            if !rust_exact || !native_exact {
                failures.push(format!(
                    "{name}-{policy} Rust={rust_exact} native={native_exact}"
                ));
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
fn parsed_dht_cloning_keeps_existing_field_edits_and_refuses_invalid_empty_segments() {
    let root = directory();
    let (empty, zero) = empty_and_zero_dc0_pair();
    let parsed_empty = jpeg::read_jpeg(&empty, None, None).unwrap();
    let mut parsed_zero = jpeg::read_jpeg(&zero, None, None).unwrap();
    // This is an external crate: the descriptor's type has no public name.
    // Existing inferred cloning and edits of its public fields still work.
    let mut cloned = parsed_zero.huffman_code[0].clone();
    cloned.id = 0;
    cloned.is_ac = false;
    cloned.is_last = true;
    cloned.counts = [0; 16];
    cloned.values.clear();
    let last = parsed_zero.huffman_code.len() - 1;
    assert_eq!(parsed_zero.huffman_code[last].id, 0);
    parsed_zero.huffman_code[last] = cloned;
    let bytes = jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed_zero, 3).unwrap();
    assert_eq!(
        zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
        zero
    );
    let path = root.join("cloned-zero.jxl");
    std::fs::write(&path, &bytes).unwrap();
    native(
        &path,
        &path.with_extension("jpg"),
        &path.with_extension("log"),
    );
    assert_eq!(std::fs::read(path.with_extension("jpg")).unwrap(), zero);
    let mut oversized = parsed_zero.clone();
    let hc = &mut oversized.huffman_code[last];
    hc.counts = [0; 16];
    hc.counts[15] = 256;
    hc.values = (0..=255).collect();
    let error = jpeg::encode_jbrd(&oversized).unwrap_err();
    assert!(error.to_string().contains("value 257"), "{error}");
    let last = parsed_empty.huffman_code.len() - 1;
    for variant in 0..5 {
        let mut parsed = parsed_empty.clone();
        let hc = &mut parsed.huffman_code[last];
        match variant {
            0 => hc.is_ac = true,
            1 => hc.id = 1,
            2 => hc.is_last = false,
            3 => hc.counts[3] = 1,
            4 => hc.values.push(0),
            _ => unreachable!(),
        }
        for direct in [false, true] {
            let result = std::panic::catch_unwind(|| {
                if direct {
                    jpeg::encode_jbrd(&parsed)
                } else {
                    jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3)
                }
            });
            assert!(
                matches!(result, Ok(Err(_))),
                "invalid empty segment {variant}, direct={direct}"
            );
        }
    }
}

#[test]
fn zero_dht_mixed_segments_and_effort_boundaries_reconstruct() {
    let root = directory();
    let base = include_bytes!("../fixtures/jbrd/base_b_444.jpg");
    let mut rows = Vec::new();
    for ids in [
        &[2u8][..],
        &[3][..],
        &[0x12][..],
        &[0x13][..],
        &[2, 0x12][..],
        &[0x13, 3][..],
    ] {
        let mut payload = Vec::new();
        for &id in ids {
            payload.push(id);
            payload.extend([0; 16]);
        }
        rows.push(before_sos(base, 0xc4, &payload));
    }
    rows.push(before_sos(&rows[0], 0xc4, &[]));
    for (i, input) in rows.iter().enumerate() {
        let parsed = jpeg::read_jpeg(input, None, None).unwrap();
        for effort in [1, 3, 7, 8, 9, 11] {
            let config = LosslessConfig::new()
                .with_effort(effort)
                .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl);
            let bytes = config.encode_jpeg_transcode(input).unwrap();
            assert_eq!(
                bytes,
                jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, effort).unwrap()
            );
            assert_eq!(
                zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
                *input
            );
            let path = root.join(format!("case-{i}-e{effort}.jxl"));
            std::fs::write(&path, &bytes).unwrap();
            decoded_frames::decode_current_frames(&bytes, "mixed tables");
            native(
                &path,
                &path.with_extension("png"),
                &path.with_extension("render.log"),
            );
            native(
                &path,
                &path.with_extension("jpg"),
                &path.with_extension("rebuild.log"),
            );
            assert_eq!(std::fs::read(path.with_extension("jpg")).unwrap(), *input);
        }
    }
}

#[test]
fn embedded_gain_map_dht_uses_the_shared_reconstruction_model() {
    let root = directory();
    let (empty, zero) = empty_and_zero_dc0_pair();
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
    for (name, gain) in [("empty", empty), ("zero", zero)] {
        let mut secondary = gain[..2].to_vec();
        secondary.extend([255, 0xe2]);
        secondary.extend(
            u16::try_from(2 + namespace.len() + metadata.len())
                .unwrap()
                .to_be_bytes(),
        );
        secondary.extend(namespace);
        secondary.extend(&metadata);
        secondary.extend(&gain[2..]);
        let mut original = include_bytes!("../fixtures/jbrd/base_b_444.jpg").to_vec();
        original.extend(&secondary);
        let bytes = LosslessConfig::new()
            .with_effort(3)
            .encode_jpeg_transcode(&original)
            .unwrap();
        assert_eq!(
            zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
            original
        );
        let path = root.join(format!("{name}.jxl"));
        std::fs::write(&path, &bytes).unwrap();
        decoded_frames::decode_current_frames(&bytes, name);
        native(
            &path,
            &path.with_extension("png"),
            &path.with_extension("render.log"),
        );
        native(
            &path,
            &path.with_extension("jpg"),
            &path.with_extension("rebuild.log"),
        );
        assert_eq!(std::fs::read(path.with_extension("jpg")).unwrap(), original);
        let mut at = 0;
        let mut found = None;
        while at < bytes.len() {
            let size = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            assert!(size >= 8 && size <= bytes.len() - at);
            if &bytes[at + 4..at + 8] == b"jhgm" {
                assert!(found.is_none());
                found = Some(&bytes[at + 8..at + size]);
            }
            at += size;
        }
        let bundle = found.unwrap();
        assert_eq!(bundle[0], 0);
        let count = usize::from(u16::from_be_bytes([bundle[1], bundle[2]]));
        assert_eq!(&bundle[3..3 + count], metadata);
        assert_eq!(&bundle[3 + count..8 + count], &[0; 5]);
        let stream = &bundle[8 + count..];
        decoded_frames::decode_current_frames(stream, name);
        let gain_path = root.join(format!("{name}-gain.jxl"));
        std::fs::write(&gain_path, stream).unwrap();
        native(
            &gain_path,
            &gain_path.with_extension("png"),
            &gain_path.with_extension("render.log"),
        );
    }
}

#[test]
fn malformed_dht_table_fields_return_errors() {
    for input in [
        &[255, 0xd8, 255, 0xc4][..],
        &[255, 0xd8, 255, 0xc4, 0, 5, 0][..],
    ] {
        let result = std::panic::catch_unwind(|| jpeg::read_jpeg(input, None, None));
        assert!(matches!(result, Ok(Err(_))), "truncated DHT {input:?}");
    }
    let base = include_bytes!("../fixtures/jbrd/base_b_444.jpg");
    let mut cases = vec![vec![0x20], vec![4], vec![0, 0]];
    for (counts, values) in [
        (
            {
                let mut c = [0; 16];
                c[15] = 13;
                c
            },
            vec![0; 13],
        ),
        (
            {
                let mut c = [0; 16];
                c[0] = 3;
                c
            },
            vec![0, 1, 2],
        ),
        (
            {
                let mut c = [0; 16];
                c[3] = 2;
                c
            },
            vec![0],
        ),
        (
            {
                let mut c = [0; 16];
                c[3] = 2;
                c
            },
            vec![0, 0],
        ),
        (
            {
                let mut c = [0; 16];
                c[3] = 1;
                c
            },
            vec![12],
        ),
    ] {
        let mut payload = vec![2];
        payload.extend(counts);
        payload.extend(values);
        cases.push(payload);
    }
    for payload in cases {
        let input = before_sos(base, 0xc4, &payload);
        let result = std::panic::catch_unwind(|| jpeg::read_jpeg(&input, None, None));
        assert!(matches!(result, Ok(Err(_))), "DHT payload {payload:?}");
    }
}
