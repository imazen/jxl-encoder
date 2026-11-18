//! An unrelated APP0 does not change the meaning of explicit RGB samples.
use super::*;

fn without_adobe(input: &[u8]) -> Vec<u8> {
    let mut output = input[..2].to_vec();
    let segments = header_segments(input);
    for &(offset, marker, size) in &segments {
        if marker != 0xee {
            output.extend_from_slice(&input[offset..offset + size]);
        }
        if marker == 0xda {
            output.extend_from_slice(&input[offset + size..]);
        }
    }
    output
}
fn insert_app0(input: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut output = input[..2].to_vec();
    output.extend([255, 0xe0]);
    output.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    output.extend(payload);
    output.extend_from_slice(&input[2..]);
    output
}
fn reference(input: &std::path::Path, output: &std::path::Path) {
    let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
        .arg(input)
        .arg(output)
        .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
        .output()
        .unwrap();
    std::fs::write(
        output.with_extension("log"),
        [result.stdout, result.stderr].concat(),
    )
    .unwrap();
    assert!(result.status.success());
}
fn check_app0(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let original = include_bytes!("../fixtures/jbrd/rgb_adobe.jpg");
    let mut journal = std::fs::File::create(root.join("app0.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut envelopes = Vec::new();
    for adobe in [false, true] {
        let baseline = if adobe {
            original.to_vec()
        } else {
            without_adobe(original)
        };
        assert!(
            !header_segments(&baseline)
                .iter()
                .any(|(_, m, _)| *m == 0xe0)
        );
        for (payload_id, payload) in [
            b"".as_slice(),
            b"JFIF",
            b"JFIFx",
            b"JFXX\0extension",
            b"Private APP0 data",
        ]
        .into_iter()
        .enumerate()
        {
            let input = insert_app0(&baseline, payload);
            // An independent libjpeg-turbo source control checks every RGB8
            // sample (jpeg-app0-source-oracle). The image crate's JPEG parser
            // over-rejects the empty APP0 input before decoding its samples.
            // Here coefficient identity and explicit RGB signaling below bind
            // each mutation to the unchanged baseline pixel oracle.
            let case = root.join(format!("adobe-{adobe}-payload-{payload_id}"));
            std::fs::create_dir(&case).unwrap();
            let original_path = case.join("baseline.jpg");
            let input_path = case.join("unknown-app0.jpg");
            std::fs::write(&original_path, &baseline).unwrap();
            std::fs::write(&input_path, &input).unwrap();
            let original_parsed = jpeg::read_jpeg(&baseline, None, None).unwrap();
            let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
            assert_eq!(parsed.components.len(), 3);
            for (i, id) in b"RGB".iter().copied().enumerate() {
                assert_eq!(parsed.components[i].id, u32::from(id));
                assert_eq!(
                    parsed.components[i].coeffs,
                    original_parsed.components[i].coeffs
                );
            }
            let reference_path = case.join("reference.jxl");
            reference(&original_path, &reference_path);
            native(
                &reference_path,
                &case.join("reference.png"),
                &case.join("reference-render.log"),
            );
            let expected_native = image::open(case.join("reference.png")).unwrap().to_rgb16();
            let changed_reference_path = case.join("changed-reference.jxl");
            reference(&input_path, &changed_reference_path);
            native(
                &changed_reference_path,
                &case.join("changed-reference.png"),
                &case.join("changed-render.log"),
            );
            native(
                &changed_reference_path,
                &case.join("changed-reference.jpg"),
                &case.join("changed-rebuild.log"),
            );
            assert_eq!(
                std::fs::read(case.join("changed-reference.jpg")).unwrap(),
                input
            );
            writeln!(
                journal,
                "adobe={adobe}\tnative_reference_pixels_unchanged={}",
                image::open(case.join("changed-reference.png"))
                    .unwrap()
                    .to_rgb16()
                    == expected_native
            )
            .unwrap();
            journal.flush().unwrap();
            for (policy, strategy, mode) in &policies {
                let out = case.join(policy);
                std::fs::create_dir(&out).unwrap();
                let config = LosslessConfig::new()
                    .with_effort(3)
                    .with_strategy(strategy.clone())
                    .with_mode(*mode);
                let baseline_bytes = config.encode_jpeg_transcode(&baseline).unwrap();
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
                let before =
                    decoded_frames::decode_current_frames(&baseline_bytes, "explicit RGB baseline");
                let after = decoded_frames::decode_current_frames(&bytes, policy);
                let primary_same = before == after;
                let path = out.join("input.jxl");
                std::fs::write(&path, &bytes).unwrap();
                native(&path, &out.join("render.png"), &out.join("render.log"));
                let native_same =
                    image::open(out.join("render.png")).unwrap().to_rgb16() == expected_native;
                let rebuilt = zensim_decoder::reconstruct_jpeg(&bytes);
                let rust_exact = matches!(&rebuilt, Ok(Some(data)) if data == &input);
                if let Ok(Some(data)) = &rebuilt {
                    std::fs::write(out.join("rust.jpg"), data).unwrap();
                }
                native(&path, &out.join("native.jpg"), &out.join("reconstruct.log"));
                let native_exact = std::fs::read(out.join("native.jpg")).unwrap() == input;
                writeln!(journal,"adobe={adobe}\t{policy}\tprimary_unchanged={primary_same}\tnative_matches_rgb_reference={native_same}\trust_exact={rust_exact}\tnative_exact={native_exact}").unwrap();
                journal.flush().unwrap();
                if !primary_same || !native_same || !rust_exact || !native_exact {
                    failures.push(format!(
                    "adobe={adobe} {policy}: primary={primary_same}, native={native_same}, Rust exact={rust_exact}, native exact={native_exact}"
                ));
                }
                envelopes.push(serde_json::json!({"path":path,"sampling":[0,0,0],"ycbcr":0}));
            }
        }
    }
    observe_headers(&root, envelopes);
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        root.display()
    );
}
#[test]
fn strict_unrelated_app0_keeps_explicit_rgb_interpretation() {
    check_app0(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_unrelated_app0_keeps_explicit_rgb_interpretation() {
    check_app0(encoder_policies::policies());
}

#[test]
fn color_signals_keep_jfif_adobe_and_component_id_precedence() {
    let root = directory();
    let original = include_bytes!("../fixtures/jbrd/rgb_adobe.jpg");
    let rgb_ids = without_adobe(original);
    let adobe_offset = header_segments(original)
        .into_iter()
        .find(|r| r.1 == 0xee)
        .unwrap()
        .0;
    let mut adobe_one = original.to_vec();
    assert_eq!(&adobe_one[adobe_offset + 4..adobe_offset + 9], b"Adobe");
    adobe_one[adobe_offset + 15] = 1;
    let mut malformed_adobe = original.to_vec();
    malformed_adobe[adobe_offset + 4] = b'a';
    let mut fallback = rgb_ids.clone();
    for (offset, marker, _) in header_segments(&fallback) {
        if marker == 0xc0 {
            for i in 0..3 {
                fallback[offset + 10 + i * 3] = i as u8 + 1;
            }
        }
        if marker == 0xda {
            for i in 0..3 {
                fallback[offset + 5 + i * 2] = i as u8 + 1;
            }
        }
    }
    let payload = b"Private APP14 data";
    let baseline = without_adobe(original);
    let mut unknown_adobe = baseline[..2].to_vec();
    unknown_adobe.extend([255, 0xee]);
    unknown_adobe.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    unknown_adobe.extend(payload);
    unknown_adobe.extend(&baseline[2..]);
    let jfif = b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00";
    let mut envelopes = Vec::new();
    for (label, input, ycbcr) in [
        ("rgb-ids", rgb_ids, 0),
        ("adobe-zero", original.to_vec(), 0),
        ("adobe-one", adobe_one, 1),
        ("malformed-adobe-rgb-ids", malformed_adobe, 0),
        ("unknown-app14-rgb-ids", unknown_adobe, 0),
        ("default-ids", fallback, 1),
        ("jfif-over-adobe-zero", insert_app0(original, jfif), 1),
    ] {
        let case = root.join(label);
        std::fs::create_dir(&case).unwrap();
        let source = case.join("original.jpg");
        std::fs::write(&source, &input).unwrap();
        let control = case.join("reference.jxl");
        reference(&source, &control);
        native(
            &control,
            &case.join("reference.png"),
            &case.join("reference-render.log"),
        );
        let expected = image::open(case.join("reference.png")).unwrap().to_rgb16();
        let expected_primary =
            decoded_frames::decode_current_frames(&std::fs::read(&control).unwrap(), label);
        for (policy, strategy, mode) in encoder_policies::policies() {
            let bytes = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy)
                .with_mode(mode)
                .encode_jpeg_transcode(&input)
                .unwrap();
            let path = case.join(format!("{policy}.jxl"));
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(
                decoded_frames::decode_current_frames(&bytes, label),
                expected_primary
            );
            native(
                &path,
                &path.with_extension("png"),
                &path.with_extension("render.log"),
            );
            assert_eq!(
                image::open(path.with_extension("png")).unwrap().to_rgb16(),
                expected
            );
            native(
                &path,
                &path.with_extension("jpg"),
                &path.with_extension("rebuild.log"),
            );
            assert_eq!(std::fs::read(path.with_extension("jpg")).unwrap(), input);
            assert_eq!(
                zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
                input
            );
            envelopes.push(serde_json::json!({"path":path,"sampling":[0,0,0],"ycbcr":ycbcr}));
        }
    }
    observe_headers(&root, envelopes);
}

#[test]
fn unrelated_app0_bare_default_effort_and_recompression_callers() {
    let root = directory();
    let original = include_bytes!("../fixtures/jbrd/rgb_adobe.jpg");
    for adobe in [false, true] {
        let baseline = if adobe {
            original.to_vec()
        } else {
            without_adobe(original)
        };
        let input = insert_app0(&baseline, b"Private APP0 data");
        let case = root.join(format!("adobe-{adobe}"));
        std::fs::create_dir(&case).unwrap();
        let source = case.join("baseline.jpg");
        std::fs::write(&source, &baseline).unwrap();
        let control = case.join("reference.jxl");
        reference(&source, &control);
        native(
            &control,
            &case.join("reference.png"),
            &case.join("reference-render.log"),
        );
        let expected_native = image::open(case.join("reference.png")).unwrap().to_rgb16();
        let expected_primary =
            decoded_frames::decode_current_frames(&std::fs::read(&control).unwrap(), "RGB control");
        let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
        let mut variants = vec![
            (
                "bare-default".into(),
                jpeg::encode_jpeg_to_jxl(&parsed).unwrap(),
                false,
            ),
            (
                "container-default".into(),
                jpeg::encode_jpeg_to_jxl_container(&parsed).unwrap(),
                true,
            ),
        ];
        for effort in [1, 3, 7, 8, 9, 11] {
            variants.push((
                format!("bare-e{effort}"),
                jpeg::encode_jpeg_to_jxl_with_effort(&parsed, effort).unwrap(),
                false,
            ));
            variants.push((
                format!("container-e{effort}"),
                jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, effort).unwrap(),
                true,
            ));
        }
        for (policy, strategy, mode) in encoder_policies::policies() {
            let config = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy)
                .with_mode(mode);
            let bytes = config.encode_jpeg_transcode_codestream(&input).unwrap();
            assert_eq!(
                bytes,
                config
                    .encode_jpeg_transcode_codestream_with_stop(&input, &enough::Unstoppable)
                    .unwrap()
            );
            variants.push((policy, bytes, false));
        }
        for (label, bytes, container) in variants {
            assert_eq!(
                decoded_frames::decode_current_frames(&bytes, &label),
                expected_primary
            );
            let path = case.join(format!("{label}.jxl"));
            std::fs::write(&path, &bytes).unwrap();
            native(
                &path,
                &path.with_extension("png"),
                &path.with_extension("render.log"),
            );
            assert_eq!(
                image::open(path.with_extension("png")).unwrap().to_rgb16(),
                expected_native
            );
            if container {
                native(
                    &path,
                    &path.with_extension("jpg"),
                    &path.with_extension("rebuild.log"),
                );
                assert_eq!(std::fs::read(path.with_extension("jpg")).unwrap(), input);
                assert_eq!(
                    zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
                    input
                );
            }
        }
        for scale in [1.0, 2.0] {
            for label in ["auto", "explicit", "planar"] {
                let encode = |data: &[u8]| match label {
                    "auto" => jpeg::encode_jpeg_recompress_auto_codestream(
                        data,
                        scale,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    ),
                    "explicit" => jpeg::encode_jpeg_recompress_codestream(
                        data,
                        scale,
                        0.5,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    ),
                    "planar" => jpeg::encode_jpeg_recompress_planar_codestream(
                        data,
                        scale,
                        0.5,
                        scale,
                        0.5,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    ),
                    _ => unreachable!(),
                };
                let bytes = encode(&input).unwrap();
                let baseline_bytes = encode(&baseline).unwrap();
                assert_eq!(
                    bytes, baseline_bytes,
                    "{label}-{scale} source interpretation"
                );
                let path = case.join(format!("coarsen-{label}-{scale}.jxl"));
                std::fs::write(&path, &bytes).unwrap();
                native(
                    &path,
                    &path.with_extension("png"),
                    &path.with_extension("render.log"),
                );
                let primary = decoded_frames::decode_current_frames(&bytes, label);
                let native = image::open(path.with_extension("png")).unwrap().to_rgb16();
                assert_eq!(primary.len(), 1);
                assert_eq!(primary[0].0.len(), native.as_raw().len());
                for (&a, &b) in primary[0].0.iter().zip(native.as_raw()) {
                    assert!((a.clamp(0.0, 1.0) * 65535.0 - f32::from(b)).abs() <= 2.0);
                }
            }
        }
    }
}

#[test]
fn unrelated_app0_embedded_gain_map_preserves_rgb_samples_and_jpeg_tail() {
    let root = directory();
    let baseline = include_bytes!("../fixtures/jbrd/rgb_adobe.jpg");
    let original_path = root.join("baseline.jpg");
    std::fs::write(&original_path, baseline).unwrap();
    let reference_path = root.join("reference.jxl");
    reference(&original_path, &reference_path);
    native(
        &reference_path,
        &root.join("reference.png"),
        &root.join("reference-render.log"),
    );
    let expected_native = image::open(root.join("reference.png")).unwrap().to_rgb16();
    let expected_primary = decoded_frames::decode_current_frames(
        &std::fs::read(reference_path).unwrap(),
        "RGB gain control",
    );
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
    let input = insert_app0(baseline, b"Private APP0 data");
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
    let mut original = include_bytes!("../fixtures/jbrd/base_a_420.jpg").to_vec();
    original.extend(secondary);
    for (policy, strategy, mode) in encoder_policies::policies() {
        let config = LosslessConfig::new()
            .with_effort(3)
            .with_strategy(strategy)
            .with_mode(mode);
        let bytes = config.encode_jpeg_transcode(&original).unwrap();
        assert_eq!(
            bytes,
            config
                .encode_jpeg_transcode_with_stop(&original, &enough::Unstoppable)
                .unwrap()
        );
        let mut at = 0;
        let mut gain = None;
        while at < bytes.len() {
            let n = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            assert!(n >= 8 && at + n <= bytes.len());
            if &bytes[at + 4..at + 8] == b"jhgm" {
                assert!(gain.is_none());
                gain = Some(&bytes[at + 8..at + n]);
            }
            at += n;
        }
        let bundle = gain.unwrap();
        let count = usize::from(u16::from_be_bytes([bundle[1], bundle[2]]));
        assert_eq!(&bundle[3..3 + count], metadata);
        let gain = &bundle[8 + count..];
        assert_eq!(
            decoded_frames::decode_current_frames(gain, &policy),
            expected_primary
        );
        let gainpath = root.join(format!("{policy}-gain.jxl"));
        std::fs::write(&gainpath, gain).unwrap();
        native(
            &gainpath,
            &gainpath.with_extension("png"),
            &gainpath.with_extension("render.log"),
        );
        assert_eq!(
            image::open(gainpath.with_extension("png"))
                .unwrap()
                .to_rgb16(),
            expected_native
        );
        let path = root.join(format!("{policy}.jxl"));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(
            decoded_frames::decode_current_frames(&bytes, &policy).len(),
            1
        );
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
        assert_eq!(
            zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
            original
        );
    }
}
