//! One gain-map box, including its Brotli-wrapped representation.
#![cfg(all(
    feature = "hdr-gainmap",
    feature = "brotli-metadata",
    feature = "__expert"
))]
use jxl_encoder::{EncodeError, LosslessConfig, LossyConfig, PixelLayout};
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

// Exercise the approved fallible low-level entry alongside the typed bundle.
fn append_raw(input: &[u8], payload: &[u8]) -> Result<Vec<u8>, EncodeError> {
    jxl_encoder::append_gain_map_box(input, payload)
}

fn boxes(data: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    let mut offset = 0;
    let mut result = Vec::new();
    while offset < data.len() {
        let size32 = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap());
        let kind = data[offset + 4..offset + 8].try_into().unwrap();
        let (size, header) = match size32 {
            0 => (data.len() - offset, 8),
            1 => (
                u64::from_be_bytes(data[offset + 8..offset + 16].try_into().unwrap()) as usize,
                16,
            ),
            n => (n as usize, 8),
        };
        assert!(size >= header && offset + size <= data.len());
        result.push((kind, data[offset + header..offset + size].to_vec()));
        offset += size;
    }
    result
}
fn boxed(kind: [u8; 4], payload: &[u8], extended: bool) -> Vec<u8> {
    let mut data = Vec::new();
    if extended {
        data.extend(1_u32.to_be_bytes());
        data.extend(kind);
        data.extend((16 + payload.len() as u64).to_be_bytes());
    } else {
        data.extend((8 + payload.len() as u32).to_be_bytes());
        data.extend(kind);
    }
    data.extend(payload);
    data
}
fn compress(payload: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::new();
    {
        let params = brotli::enc::BrotliEncoderParams::default();
        let mut writer = brotli::CompressorWriter::with_params(&mut compressed, 4096, &params);
        writer.write_all(payload).unwrap();
        writer.flush().unwrap();
    }
    [b"jhgm".as_slice(), &compressed].concat()
}
fn maps(data: &[u8]) -> usize {
    boxes(data)
        .iter()
        .filter(|(kind, payload)| {
            kind == b"jhgm" || kind == b"brob" && payload.starts_with(b"jhgm")
        })
        .count()
}
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("gain-map-duplicate-contract");
    std::fs::create_dir_all(&root).unwrap();
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
fn native(directory: &std::path::Path, bytes: &[u8]) -> Vec<u16> {
    std::fs::create_dir(directory).unwrap();
    let input = directory.join("existing.jxl");
    std::fs::write(&input, bytes).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(directory.join("existing.png"))
        .args(["--num_threads=1", "--bits_per_sample=16"])
        .output()
        .unwrap();
    std::fs::write(directory.join("native.log"), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "{}: {}",
        directory.display(),
        String::from_utf8_lossy(&result.stderr)
    );
    let decoded = image::open(directory.join("existing.png"))
        .unwrap()
        .to_rgb16();
    assert_eq!(decoded.dimensions(), (16, 9));
    decoded.into_raw()
}
#[test]
fn duplicate_gain_map_append_preserves_the_existing_file() {
    let root = directory();
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let gain = LosslessConfig::new()
        .with_effort(1)
        .encode(&[42; 15], 5, 3, PixelLayout::Gray8)
        .unwrap();
    let bundle = jxl_encoder::hdr::GainMapBundle::new(metadata, gain);
    let payload = bundle.serialize().unwrap();
    let pixels = [64_u8, 96, 128].repeat(16 * 9);
    let mut accepted = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for lossless in [true, false] {
            let base = if lossless {
                LosslessConfig::new()
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .encode(&pixels, 16, 9, PixelLayout::Rgb8)
            } else {
                LossyConfig::new(1.0)
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .encode(&pixels, 16, 9, PixelLayout::Rgb8)
            }
            .unwrap();
            // A pre-existing container with ordinary compressed metadata is
            // a positive append input; a brob wrapper alone is not a gain map.
            let mut compressed_xml = compress(b"<x/>");
            compressed_xml[..4].copy_from_slice(b"xml ");
            let mut ordinary = jxl_encoder::wrap_in_container(&base, None, None);
            ordinary.extend(boxed(*b"brob", &compressed_xml, false));
            ordinary.extend(boxed(*b"test", b"jhgm", false));
            let positive = append_raw(&ordinary, &payload).unwrap();
            assert_eq!(maps(&positive), 1);
            assert_eq!(
                positive,
                jxl_encoder::hdr::append_gain_map_bundle(&ordinary, &bundle).unwrap()
            );
            let label = format!("{policy}-{lossless}-ordinary-metadata");
            assert_eq!(
                decoded_frames::decode_current_frames(&positive, &label),
                decoded_frames::decode_current_frames(&base, &label)
            );
            assert_eq!(
                native(&root.join(label), &positive),
                native(
                    &root.join(format!("{policy}-{lossless}-ordinary-base")),
                    &base
                )
            );
            let first = append_raw(&base, &payload).unwrap();
            assert_eq!(
                first,
                jxl_encoder::hdr::append_gain_map_bundle(&base, &bundle).unwrap()
            );
            assert_eq!(maps(&first), 1);
            assert_eq!(
                boxes(&first)
                    .into_iter()
                    .find(|(kind, _)| kind == b"jhgm")
                    .unwrap()
                    .1,
                payload
            );
            let mut variants = vec![("direct", first.clone())];
            let mut extended = Vec::new();
            for (kind, bytes) in boxes(&first) {
                extended.extend(boxed(kind, &bytes, kind == *b"jhgm"));
            }
            variants.push(("direct-extended", extended));
            for extended in [false, true] {
                let mut data = Vec::new();
                for (kind, bytes) in boxes(&first) {
                    if kind == *b"jhgm" {
                        data.extend(boxed(*b"brob", &compress(&bytes), extended));
                    } else {
                        data.extend(boxed(kind, &bytes, false));
                    }
                }
                variants.push((if extended { "brob-extended" } else { "brob" }, data));
            }
            for compressed in [false, true] {
                let mut prefix = Vec::new();
                for (kind, bytes) in boxes(&first) {
                    if kind != *b"jhgm" {
                        prefix.extend(boxed(kind, &bytes, false));
                    }
                }
                let mut tail = if compressed {
                    boxed(*b"brob", &compress(&payload), false)
                } else {
                    boxed(*b"jhgm", &payload, false)
                };
                tail[..4].copy_from_slice(&0_u32.to_be_bytes());
                prefix.extend(tail);
                variants.push((
                    if compressed {
                        "brob-to-eof"
                    } else {
                        "direct-to-eof"
                    },
                    prefix,
                ));
            }
            let expected_native = native(&root.join(format!("{policy}-{lossless}-base")), &base);
            let expected = decoded_frames::decode_current_frames(&base, &policy);
            assert_eq!(expected.len(), 1);
            for (shape, input) in variants {
                assert_eq!(maps(&input), 1);
                let output = root.join(format!("{policy}-{lossless}-{shape}"));
                std::fs::create_dir(&output).unwrap();
                std::fs::write(output.join("existing.jxl"), &input).unwrap();
                let decoded = decoded_frames::decode_current_frames(&input, &policy);
                assert_eq!(
                    decoded[0].0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected[0]
                        .0
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(native(&output.join("native"), &input), expected_native);
                for typed in [false, true] {
                    let saved = input.clone();
                    let result = if typed {
                        jxl_encoder::hdr::append_gain_map_bundle(&input, &bundle)
                    } else {
                        append_raw(&input, &payload)
                    };
                    assert_eq!(input, saved);
                    if let Err(error) = &result {
                        assert!(
                            matches!(error, EncodeError::InvalidInput { .. }),
                            "{policy} {shape}: {error:?}"
                        );
                        assert!(
                            error.to_string().contains("already contains a gain map"),
                            "{shape}: {error}"
                        );
                    }
                    if let Ok(bytes) = result {
                        let count = maps(&bytes);
                        std::fs::write(output.join(format!("accepted-{typed}.jxl")), bytes)
                            .unwrap();
                        accepted.push(format!(
                            "{policy} {lossless} {shape} typed={typed}: maps={count}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "{}; {}",
        accepted.join("\n"),
        root.display()
    );
}

#[test]
fn malformed_container_boundaries_fail_without_changing_input() {
    let base = LosslessConfig::new()
        .with_effort(1)
        .encode(&[64; 16 * 9 * 3], 16, 9, PixelLayout::Rgb8)
        .unwrap();
    let container = jxl_encoder::wrap_in_container(&base, None, None);
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let gain = LosslessConfig::new()
        .with_effort(1)
        .encode(&[42; 15], 5, 3, PixelLayout::Gray8)
        .unwrap();
    let mut bundle = jxl_encoder::hdr::GainMapBundle::new(metadata, gain);
    let payload = bundle.serialize().unwrap();
    let mut tails = vec![
        vec![0; 1],
        vec![0; 3],
        vec![0; 4],
        vec![0; 7],
        b"\0\0\0\x04free".to_vec(),
        b"\xff\xff\xff\xfffree".to_vec(),
    ];
    for length in 8..16 {
        let mut v = b"\0\0\0\x01free".to_vec();
        v.resize(length, 0);
        tails.push(v);
    }
    for size in [0_u64, 15, u64::MAX] {
        let mut v = b"\0\0\0\x01free".to_vec();
        v.extend(size.to_be_bytes());
        tails.push(v);
    }
    for tail in tails {
        let mut input = container.clone();
        input.extend(tail);
        let saved = input.clone();
        for result in [
            append_raw(&input, &payload),
            jxl_encoder::hdr::append_gain_map_bundle(&input, &bundle),
        ] {
            let error = result.unwrap_err();
            assert!(matches!(error, EncodeError::InvalidInput { .. }));
            assert!(
                error.to_string().contains("malformed container boxes"),
                "{error}"
            );
        }
        assert_eq!(input, saved);
    }
    bundle.jhgm_version = 1;
    assert!(
        jxl_encoder::hdr::append_gain_map_bundle(&container, &bundle).is_err(),
        "typed bundle serialization errors propagate"
    );
}
