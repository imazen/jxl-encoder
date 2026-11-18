//! Prefix-only APP1 remains opaque, including before genuine metadata.
use super::*;

fn insert_app1(input: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut output = input[..2].to_vec();
    output.extend([255, 0xe1]);
    output.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    output.extend(payload);
    output.extend_from_slice(&input[2..]);
    output
}
fn check_prefixes(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let base = include_bytes!("../fixtures/jbrd/base_b_444.jpg");
    assert!(
        !header_segments(base)
            .iter()
            .any(|(_, marker, _)| *marker == 0xe1)
    );
    let mut journal = std::fs::File::create(root.join("metadata.tsv")).unwrap();
    let mut failures = Vec::new();
    for (name, prefix) in [
        ("Exif", b"Exif\0\0".as_slice()),
        ("XMP", b"http://ns.adobe.com/xap/1.0/\0".as_slice()),
    ] {
        let valid_fixture: &[u8] = if name == "Exif" {
            include_bytes!("../fixtures/jbrd/meta_d_exif_orient_big.jpg")
        } else {
            include_bytes!("../fixtures/jbrd/meta_a_xmp.jpg")
        };
        for (placement, base) in [("absent", base.as_slice()), ("before-valid", valid_fixture)] {
            let original = jpeg::read_jpeg(base, None, None).unwrap();
            let input = insert_app1(base, prefix);
            assert_eq!(
                image::load_from_memory(&input).unwrap().to_rgb8(),
                image::load_from_memory(base).unwrap().to_rgb8()
            );
            let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
            for (component, before) in parsed.components.iter().zip(&original.components) {
                assert_eq!(component.coeffs, before.coeffs);
            }
            for (policy, strategy, mode) in &policies {
                let out = root.join(format!("{name}-{placement}-{policy}"));
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
                let expected = config.encode_jpeg_transcode(base).unwrap();
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
                assert_eq!(
                    decoded_frames::decode_current_frames(&bare, name),
                    decoded_frames::decode_current_frames(&expected, name)
                );
                let bare_path = out.join("bare.jxl");
                std::fs::write(&bare_path, &bare).unwrap();
                native(&bare_path, &out.join("bare.png"), &out.join("bare.log"));

                let path = out.join("input.jxl");
                std::fs::write(&path, &bytes).unwrap();
                let primary_same = decoded_frames::decode_current_frames(&bytes, name)
                    == decoded_frames::decode_current_frames(
                        &expected,
                        "unchanged coefficient baseline",
                    );
                native(&path, &out.join("render.png"), &out.join("render.log"));
                let expected_path = out.join("baseline.jxl");
                std::fs::write(&expected_path, &expected).unwrap();
                native(
                    &expected_path,
                    &out.join("baseline.png"),
                    &out.join("baseline-render.log"),
                );
                let native_same = image::open(out.join("render.png")).unwrap().to_rgb16()
                    == image::open(out.join("baseline.png")).unwrap().to_rgb16();
                assert_eq!(
                    image::open(out.join("bare.png")).unwrap().to_rgb16(),
                    image::open(out.join("baseline.png")).unwrap().to_rgb16()
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
                let native_exact = result.status.success()
                    && std::fs::read(out.join("native.jpg")).unwrap() == input;
                writeln!(journal,"{name}-{placement}\t{policy}\tprimary_same={primary_same}\tnative_same={native_same}\trust_exact={rust_exact}\tnative_exact={native_exact}\trust_result={:?}", rust.as_ref().map(|data| data.as_ref().map(Vec::len))).unwrap();
                journal.flush().unwrap();
                if !rust_exact || !native_exact || !primary_same || !native_same {
                    failures.push(format!(
                        "{name} {policy}: Rust={rust_exact}, native={native_exact}"
                    ));
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
fn strict_prefix_only_metadata_is_opaque_and_reconstructs() {
    check_prefixes(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_prefix_only_metadata_is_opaque_and_reconstructs() {
    check_prefixes(encoder_policies::policies());
}

#[test]
fn parsed_prefix_only_metadata_cannot_emit_an_unreconstructible_stream() {
    let root = directory();
    let base = include_bytes!("../fixtures/jbrd/base_b_444.jpg");
    for (name, prefix, fixture) in [
        (
            "Exif",
            b"Exif\0\0".as_slice(),
            include_bytes!("../fixtures/jbrd/meta_d_exif_orient_big.jpg").as_slice(),
        ),
        (
            "XMP",
            b"http://ns.adobe.com/xap/1.0/\0".as_slice(),
            include_bytes!("../fixtures/jbrd/meta_a_xmp.jpg").as_slice(),
        ),
    ] {
        let genuine = jpeg::read_jpeg(fixture, None, None).unwrap();
        let index = genuine
            .app_data
            .iter()
            .position(|data| data[3..].starts_with(prefix))
            .unwrap();
        let kind = genuine.app_marker_type[index];
        let input = insert_app1(base, prefix);
        let mut parsed = jpeg::read_jpeg(&input, None, None).unwrap();
        // The public parsed-data API lets a caller provide classifications
        // independently of the raw parser. The first APP marker is ours.
        assert_eq!(&parsed.app_data[0][3..], prefix);
        parsed.app_marker_type[0] = kind;
        let encoded =
            std::panic::catch_unwind(|| jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3))
                .expect("parsed metadata must return a typed result");
        match encoded {
            Err(error) => {
                std::fs::write(root.join(format!("{name}.refusal.txt")), error.to_string())
                    .unwrap();
            }
            Ok(bytes) => {
                let path = root.join(format!("{name}.jxl"));
                std::fs::write(&path, &bytes).unwrap();
                let frames = decoded_frames::decode_current_frames(&bytes, name);
                assert_eq!(frames.len(), 1);
                native(
                    &path,
                    &root.join(format!("{name}.png")),
                    &root.join(format!("{name}.render.log")),
                );
                assert_eq!(
                    zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
                    input
                );
                let output = root.join(format!("{name}.jpg"));
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(&path)
                    .arg(&output)
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(
                    root.join(format!("{name}.reconstruction.log")),
                    &native.stderr,
                )
                .unwrap();
                assert!(
                    native.status.success(),
                    "{name}: {}",
                    String::from_utf8_lossy(&native.stderr)
                );
                assert_eq!(std::fs::read(output).unwrap(), input);
            }
        }
    }
}
#[test]
fn parsed_metadata_classifications_are_checked_before_color_access() {
    let base = jpeg::read_jpeg(
        include_bytes!("../fixtures/jbrd/base_b_444.jpg"),
        None,
        None,
    )
    .unwrap();
    assert!(!base.app_data.is_empty());
    let root = directory();
    for name in [
        "count-short",
        "count-long",
        "wrong-marker",
        "short-data",
        "prefix-only",
        "wrong-prefix",
    ] {
        let mut parsed = base.clone();
        match name {
            "count-short" => {
                parsed.app_marker_type.pop();
            }
            "count-long" => parsed.app_marker_type.push(parsed.app_marker_type[0]),
            _ => {
                let genuine = jpeg::read_jpeg(
                    include_bytes!("../fixtures/jbrd/meta_d_exif_orient_big.jpg"),
                    None,
                    None,
                )
                .unwrap();
                let index = genuine
                    .app_data
                    .iter()
                    .position(|v| v[3..].starts_with(b"Exif\0\0"))
                    .unwrap();
                parsed.app_marker_type[0] = genuine.app_marker_type[index];
                parsed.app_data[0] = match name {
                    "wrong-marker" => b"\xe2\0\x09Exif\0\0I".to_vec(),
                    "short-data" => vec![0xe1],
                    "prefix-only" => b"\xe1\0\x08Exif\0\0".to_vec(),
                    "wrong-prefix" => b"\xe1\0\x09Other\0I".to_vec(),
                    _ => unreachable!(),
                };
            }
        }
        std::fs::write(root.join(format!("{name}.txt")), format!("{parsed:?}")).unwrap();
        for entry in ["bare", "container", "jbrd"] {
            let result = std::panic::catch_unwind(|| match entry {
                "bare" => jpeg::encode_jpeg_to_jxl_with_effort(&parsed, 3),
                "container" => jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3),
                "jbrd" => jpeg::encode_jbrd(&parsed),
                _ => unreachable!(),
            })
            .unwrap_or_else(|_| {
                panic!("{name} {entry} must reject before indexing color metadata")
            });
            let error = result.unwrap_err();
            assert!(
                error.to_string().contains(if name.starts_with("count-") {
                    "classification counts"
                } else {
                    "classified JPEG APP1"
                }),
                "{name} {entry}: {error}"
            );
        }
    }
}

#[test]
fn parsed_distinct_metadata_cannot_share_one_external_box() {
    let root = directory();
    for (name, prefix, fixture) in [
        (
            "Exif",
            b"Exif\0\0".as_slice(),
            include_bytes!("../fixtures/jbrd/meta_d_exif_orient_big.jpg").as_slice(),
        ),
        (
            "XMP",
            b"http://ns.adobe.com/xap/1.0/\0".as_slice(),
            include_bytes!("../fixtures/jbrd/meta_a_xmp.jpg").as_slice(),
        ),
    ] {
        let mut payload = prefix.to_vec();
        payload.push(b'I');
        let input = insert_app1(fixture, &payload);
        let mut parsed = jpeg::read_jpeg(&input, None, None).unwrap();
        let matching: Vec<usize> = parsed
            .app_data
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v[3..].starts_with(prefix).then_some(i))
            .collect();
        assert_eq!(matching.len(), 2);
        assert_ne!(parsed.app_data[matching[0]], parsed.app_data[matching[1]]);
        parsed.app_marker_type[matching[1]] = parsed.app_marker_type[matching[0]];
        let result =
            std::panic::catch_unwind(|| jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3))
                .expect("distinct classified metadata must not panic");
        match result {
            Err(error) => assert!(
                error.to_string().contains("classified JPEG APP1"),
                "{name}: {error}"
            ),
            Ok(bytes) => {
                let path = root.join(format!("{name}.jxl"));
                std::fs::write(&path, &bytes).unwrap();
                std::fs::write(root.join(format!("{name}.input.jpg")), &input).unwrap();
                assert_eq!(decoded_frames::decode_current_frames(&bytes, name).len(), 1);
                native(
                    &path,
                    &root.join(format!("{name}.png")),
                    &root.join(format!("{name}.render.log")),
                );
                let rust = zensim_decoder::reconstruct_jpeg(&bytes);
                std::fs::write(
                    root.join(format!("{name}.rust-result.txt")),
                    format!("{rust:?}"),
                )
                .unwrap();
                let rebuilt = root.join(format!("{name}.jpg"));
                let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(&path)
                    .arg(&rebuilt)
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(root.join(format!("{name}.native.log")), &result.stderr).unwrap();
                assert!(
                    result.status.success(),
                    "{name}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert_eq!(
                    std::fs::read(rebuilt).unwrap(),
                    input,
                    "{name} full original reconstruction"
                );
                assert_eq!(
                    rust.unwrap().unwrap(),
                    input,
                    "{name} Rust full original reconstruction"
                );
            }
        }
    }
}
#[test]
fn nonempty_metadata_and_repeated_raw_records_keep_original_bytes() {
    let root = directory();
    for (name, prefix, fixture) in [
        (
            "Exif",
            b"Exif\0\0".as_slice(),
            include_bytes!("../fixtures/jbrd/meta_d_exif_orient_big.jpg").as_slice(),
        ),
        (
            "XMP",
            b"http://ns.adobe.com/xap/1.0/\0".as_slice(),
            include_bytes!("../fixtures/jbrd/meta_a_xmp.jpg").as_slice(),
        ),
    ] {
        let genuine = jpeg::read_jpeg(fixture, None, None).unwrap();
        let index = genuine
            .app_data
            .iter()
            .position(|v| v[3..].starts_with(prefix))
            .unwrap();
        for identical in [false, true] {
            let input = if identical {
                insert_app1(fixture, &genuine.app_data[index][3..])
            } else {
                let mut p = prefix.to_vec();
                p.push(b'I');
                insert_app1(include_bytes!("../fixtures/jbrd/base_b_444.jpg"), &p)
            };
            let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
            if identical {
                let matching: Vec<usize> = parsed
                    .app_data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| v[3..].starts_with(prefix).then_some(i))
                    .collect();
                assert_eq!(matching.len(), 2);
                assert_eq!(parsed.app_data[matching[0]], parsed.app_data[matching[1]]);
                let mut forged = parsed.clone();
                forged.app_marker_type[matching[1]] = forged.app_marker_type[matching[0]];
                assert!(
                    jpeg::encode_jbrd(&forged)
                        .unwrap_err()
                        .to_string()
                        .contains("multiple classified JPEG APP1")
                );
                assert!(
                    jpeg::encode_jpeg_to_jxl_container_with_effort(&forged, 3)
                        .unwrap_err()
                        .to_string()
                        .contains("multiple classified JPEG APP1")
                );
            }
            let bytes = jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3).unwrap();
            let reference = LosslessConfig::new()
                .with_effort(3)
                .encode_jpeg_transcode(&input)
                .unwrap();
            assert_eq!(
                decoded_frames::decode_current_frames(&bytes, name),
                decoded_frames::decode_current_frames(&reference, name)
            );
            let out = root.join(format!("{name}-{identical}"));
            std::fs::create_dir(&out).unwrap();
            let path = out.join("output.jxl");
            std::fs::write(&path, &bytes).unwrap();
            native(&path, &out.join("render.png"), &out.join("render.log"));
            native(
                &path,
                &out.join("reconstructed.jpg"),
                &out.join("reconstruct.log"),
            );
            assert!(
                std::fs::read(out.join("reconstructed.jpg")).unwrap() == input,
                "{name}-{identical} native original bytes"
            );
            assert!(
                zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap() == input,
                "{name}-{identical} Rust original bytes"
            );
        }
    }
}
#[test]
fn empty_app_list_and_gain_map_secondary_prefixes_use_shared_admission() {
    let root = directory();
    let source = include_bytes!("../fixtures/jbrd/gray_a.jpg");
    let mut plain = source[..2].to_vec();
    for (offset, marker, length) in header_segments(source) {
        if !(0xe0..=0xef).contains(&marker) {
            plain.extend(&source[offset..offset + length]);
        }
        if marker == 0xda {
            plain.extend(&source[offset + length..]);
        }
    }
    let parsed = jpeg::read_jpeg(&plain, None, None).unwrap();
    assert!(parsed.app_data.is_empty());
    assert!(parsed.app_marker_type.is_empty());
    let config = LosslessConfig::new().with_effort(3);
    let bytes = config.encode_jpeg_transcode(&plain).unwrap();
    assert_eq!(
        bytes,
        jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3).unwrap()
    );
    assert!(zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap() == plain);
    let path = root.join("no-app.jxl");
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(
        decoded_frames::decode_current_frames(&bytes, "no APP").len(),
        1
    );
    native(
        &path,
        &root.join("no-app.png"),
        &root.join("no-app-render.log"),
    );
    native(
        &path,
        &root.join("no-app.jpg"),
        &root.join("no-app-reconstruct.log"),
    );
    assert!(std::fs::read(root.join("no-app.jpg")).unwrap() == plain);
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
    for (name, prefix) in [
        ("Exif", b"Exif\0\0".as_slice()),
        ("XMP", b"http://ns.adobe.com/xap/1.0/\0".as_slice()),
    ] {
        let gain = insert_app1(source, prefix);
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
        let mut original = plain.clone();
        original.extend(&secondary);
        let bytes = config.encode_jpeg_transcode(&original).unwrap();
        assert_eq!(
            bytes,
            config
                .encode_jpeg_transcode_with_stop(&original, &enough::Unstoppable)
                .unwrap()
        );
        assert!(zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap() == original);
        let path = root.join(format!("gain-{name}.jxl"));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(decoded_frames::decode_current_frames(&bytes, name).len(), 1);
        native(
            &path,
            &root.join(format!("gain-{name}.png")),
            &root.join(format!("gain-{name}-render.log")),
        );
        native(
            &path,
            &root.join(format!("gain-{name}.jpg")),
            &root.join(format!("gain-{name}-reconstruct.log")),
        );
        assert!(std::fs::read(root.join(format!("gain-{name}.jpg"))).unwrap() == original);
    }
}
