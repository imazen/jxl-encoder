#![cfg(feature = "jpeg-reencoding")]

use super::jpeg_reencoding::decode_jxl_rs;
use jxl_encoder::{LosslessConfig, Unstoppable};
use std::path::Path;

const ISO_NAMESPACE: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";

fn gain_map_bundle(data: &[u8]) -> (&[u8], &[u8]) {
    let mut pos = 0;
    let mut found = None;
    while pos < data.len() {
        let size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && size <= data.len() - pos);
        if &data[pos + 4..pos + 8] == b"jhgm" {
            assert!(found.is_none());
            found = Some(&data[pos + 8..pos + size]);
        }
        pos += size;
    }
    let bundle = found.expect("JPEG gain map must be exposed as jhgm");
    assert_eq!(bundle[0], 0);
    let size = u16::from_be_bytes([bundle[1], bundle[2]]) as usize;
    let metadata = &bundle[3..3 + size];
    assert_eq!(&bundle[3 + size..8 + size], &[0; 5]);
    let codestream = &bundle[8 + size..];
    assert!(codestream.starts_with(&[0xff, 0x0a]));
    (metadata, codestream)
}

fn reference_decode(input: &Path, output: &Path, reconstruct: bool) {
    let mut command = std::process::Command::new(jxl_encoder::test_helpers::djxl_path());
    command.arg(input).arg(output).arg("--num_threads=1");
    if reconstruct {
        command.arg("--reconstruct_jpeg");
    }
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

/// The orientation a JXL file's or bare codestream's header signals, read with jxl-rs
/// without applying it.
fn header_orientation(data: &[u8]) -> String {
    use jxl::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};
    let mut input = data;
    let mut options = JxlDecoderOptions::default();
    options.adjust_orientation = false;
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                assert!(!input.is_empty(), "jxl-rs: truncated header");
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };
    format!("{:?}", decoder.basic_info().orientation)
}

/// `jpeg` with an Exif APP1 (after any JFIF APP0) whose IFD0 holds only `orientation`.
fn with_exif_orientation(jpeg: &[u8], orientation: u16) -> Vec<u8> {
    let mut exif = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
    exif.extend(orientation.to_le_bytes());
    exif.extend([0, 0, 0, 0, 0, 0]);
    let at = if jpeg[2..4] == [0xff, 0xe0] {
        4 + u16::from_be_bytes([jpeg[4], jpeg[5]]) as usize
    } else {
        2
    };
    let mut out = jpeg[..at].to_vec();
    out.extend([0xff, 0xe1]);
    out.extend(u16::try_from(2 + exif.len()).unwrap().to_be_bytes());
    out.extend(exif);
    out.extend_from_slice(&jpeg[at..]);
    out
}

/// The Exif orientation value for a jxl-rs orientation name.
fn exif_value(orientation: &str) -> u16 {
    match orientation {
        "Identity" => 1,
        "FlipHorizontal" => 2,
        "Rotate180" => 3,
        "FlipVertical" => 4,
        "Transpose" => 5,
        "Rotate90Cw" | "Rotate90CW" => 6,
        "AntiTranspose" => 7,
        "Rotate90Ccw" | "Rotate90CCW" => 8,
        other => panic!("orientation {other}"),
    }
}

fn verify(original: &[u8], dir: &Path) {
    verify_reference(original, dir, None);
}

fn verify_reference(original: &[u8], dir: &Path, explicit_reference: Option<(&[u8], u16)>) {
    std::fs::create_dir_all(dir).unwrap();
    let config = LosslessConfig::new().with_effort(3);
    let encoded = config.encode_jpeg_transcode(original).unwrap();
    std::fs::write(dir.join("output.jxl"), &encoded).unwrap();
    let (metadata, gain_map) = gain_map_bundle(&encoded);
    let images = zenjpeg::container::marker::find_jpeg_boundaries(original);
    assert_eq!(images.len(), 2);
    let secondary = &original[images[1].clone()];
    let source_metadata = zenjpeg::container::marker::iter(secondary)
        .find_map(|span| span.payload.strip_prefix(ISO_NAMESPACE))
        .unwrap();
    assert_eq!(
        metadata, source_metadata,
        "preserve exact ISO rational bytes"
    );
    // Secondary EXIF cannot override the primary's resolved orientation,
    // including Identity when primary EXIF is absent or invalid.
    let orientation = header_orientation(&encoded);
    if let Some((_, expected_tag)) = explicit_reference {
        assert_eq!(exif_value(&orientation), expected_tag);
    }
    assert_eq!(
        header_orientation(gain_map),
        orientation,
        "the gain-map codestream carries the primary's orientation"
    );
    let oriented_secondary = if let Some((reference, _)) = explicit_reference {
        reference.to_vec()
    } else {
        with_exif_orientation(secondary, exif_value(&orientation))
    };
    let standalone = config
        .encode_jpeg_transcode_codestream(&oriented_secondary)
        .unwrap();
    assert_eq!(gain_map, standalone);
    let gain_pixels = decode_jxl_rs(gain_map);
    let base_pixels = decode_jxl_rs(&encoded);
    assert_eq!(
        base_pixels,
        decode_jxl_rs(&config.encode_jpeg_transcode_codestream(original).unwrap())
    );
    assert_eq!(
        zensim_decoder::reconstruct_jpeg(&encoded).unwrap().unwrap(),
        original
    );
    std::fs::write(dir.join("gain-map.jxl"), gain_map).unwrap();
    reference_decode(&dir.join("gain-map.jxl"), &dir.join("gain-map.png"), false);
    reference_decode(&dir.join("output.jxl"), &dir.join("base.png"), false);
    reference_decode(
        &dir.join("output.jxl"),
        &dir.join("reconstructed.jpg"),
        true,
    );
    assert_eq!(
        std::fs::read(dir.join("reconstructed.jpg")).unwrap(),
        original
    );
    let gain = image::open(dir.join("gain-map.png")).unwrap();
    assert_eq!(
        (gain.width() as usize, gain.height() as usize),
        (gain_pixels.0, gain_pixels.1)
    );
    std::fs::write(dir.join("gain-map.jpg"), &oriented_secondary).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
        .arg(dir.join("gain-map.jpg"))
        .arg(dir.join("reference-gain-map.jxl"))
        .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    reference_decode(
        &dir.join("reference-gain-map.jxl"),
        &dir.join("reference-gain-map.png"),
        false,
    );
    let reference_gain = image::open(dir.join("reference-gain-map.png")).unwrap();
    let gain = gain.to_rgb16();
    let reference_gain = reference_gain.to_rgb16();
    assert_eq!(gain.dimensions(), reference_gain.dimensions());
    for (i, (&actual, &expected)) in gain
        .as_raw()
        .iter()
        .zip(reference_gain.as_raw())
        .enumerate()
    {
        assert_eq!(actual, expected, "reference gain-map sample {i}");
    }
    let base = image::open(dir.join("base.png")).unwrap();
    assert_eq!(
        (base.width() as usize, base.height() as usize),
        (base_pixels.0, base_pixels.1)
    );
}

#[test]
fn iso_gain_map_obeys_pixel_limit_and_preserves_unrelated_tails() {
    let primary = jpeg_crop(32, 32);
    let mut original = primary.clone();
    original.extend(with_iso_metadata(&jpeg_crop(259, 133)));
    let config = LosslessConfig::new()
        .with_effort(3)
        .with_limits(&jxl_encoder::Limits::default().with_max_pixels(4096));
    assert!(config.encode_jpeg_transcode(&primary).is_ok());
    let error = config.encode_jpeg_transcode(&original).unwrap_err();
    assert!(error.to_string().contains("pixel"), "{error}");

    let mut opaque_tail = primary.clone();
    opaque_tail.extend(b"opaque vendor data \xff\xd8\x00\xff\xd8");
    let encoded = config.encode_jpeg_transcode(&opaque_tail).unwrap();
    assert_eq!(
        zensim_decoder::reconstruct_jpeg(&encoded).unwrap().unwrap(),
        opaque_tail
    );
    assert_eq!(
        decode_jxl_rs(&encoded),
        decode_jxl_rs(&config.encode_jpeg_transcode(&primary).unwrap())
    );
}

fn jpeg_crop(w: u32, h: u32) -> Vec<u8> {
    let source = image::load_from_memory(include_bytes!("../images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let crop = image::imageops::crop_imm(&source, 0, 0, w, h).to_image();
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 90)
        .encode_image(&image::DynamicImage::ImageRgb8(crop))
        .unwrap();
    jpeg
}

fn with_iso_metadata(jpeg: &[u8]) -> Vec<u8> {
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let mut out = jpeg[..2].to_vec();
    out.extend([0xff, 0xe2]);
    out.extend(
        u16::try_from(2 + ISO_NAMESPACE.len() + metadata.len())
            .unwrap()
            .to_be_bytes(),
    );
    out.extend(ISO_NAMESPACE);
    out.extend(metadata);
    out.extend_from_slice(&jpeg[2..]);
    out
}

#[test]
fn iso_gain_map_survives_container_and_jpeg_reconstruction() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/jpeg-gainmap-validation");
    for (w, h) in [(64, 32), (259, 133)] {
        let primary = jpeg_crop(w, h);
        let secondary = with_iso_metadata(&jpeg_crop(w, h));
        let mut original = primary.clone();
        original.extend(b"padding before secondary");
        original.extend_from_slice(&secondary);
        original.extend(b"vendor trailer with incomplete SOI \xff\xd8");
        verify(&original, &root.join(format!("{w}x{h}")));
        let config = LosslessConfig::new().with_effort(3);
        let encoded = config.encode_jpeg_transcode(&original).unwrap();
        assert_eq!(
            encoded,
            config
                .encode_jpeg_transcode_with_stop(&original, &Unstoppable)
                .unwrap()
        );
        assert_eq!(
            encoded,
            config
                .clone()
                .with_limits(&jxl_encoder::Limits::default().with_fallible_alloc(true))
                .encode_jpeg_transcode(&original)
                .unwrap()
        );
        let tight = config
            .clone()
            .with_limits(&jxl_encoder::Limits::default().with_max_memory_bytes(1024));
        assert!(tight.encode_jpeg_transcode(&original).is_err());
        // The parsed-data entry point shares the same container implementation.
        let parsed = jxl_encoder::jpeg::read_jpeg(&original, None, None).unwrap();
        assert_eq!(
            encoded,
            jxl_encoder::jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3).unwrap()
        );
    }
}

/// A primary JPEG with Exif orientation 6 (a portrait phone photo stored landscape) and its
/// gain map, stored the same way: the jhgm gain map must display in the base's orientation.
#[test]
fn iso_gain_map_follows_primary_orientation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/jpeg-gainmap-validation");
    let primary = with_exif_orientation(&jpeg_crop(64, 32), 6);
    let secondary = with_iso_metadata(&jpeg_crop(64, 32));
    let mut original = primary.clone();
    original.extend_from_slice(&secondary);
    verify(&original, &root.join("orientation-6"));
    let encoded = LosslessConfig::new()
        .with_effort(3)
        .encode_jpeg_transcode(&original)
        .unwrap();
    let (_, gain_map) = gain_map_bundle(&encoded);
    assert_eq!(header_orientation(&encoded), header_orientation(gain_map));
    let base = decode_jxl_rs(&encoded);
    let gain = decode_jxl_rs(gain_map);
    assert_eq!((base.0, base.1), (32, 64), "the base displays rotated");
    assert_eq!(
        (gain.0, gain.1),
        (32, 64),
        "the gain map displays like the base"
    );
}

#[cfg(feature = "corpus-tests")]
#[test]
fn iso_gain_map_camera_corpus() {
    let root =
        std::env::var_os("JPEG_GAINMAP_CORPUS").expect("caller must set JPEG_GAINMAP_CORPUS");
    let output =
        std::env::var_os("JPEG_GAINMAP_ARTIFACTS").expect("caller must set JPEG_GAINMAP_ARTIFACTS");
    use sha2::{Digest, Sha256};
    let entries: Vec<_> = include_str!("../fixtures/jpeg_gainmap_corpus.tsv")
        .lines()
        .skip(1)
        .collect();
    assert_eq!(entries.len(), 33, "the registered Samsung UltraHDR corpus");
    for (i, entry) in entries.iter().enumerate() {
        let (sha, relative) = entry.split_once('\t').unwrap();
        let path = Path::new(&root).join(relative);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            sha,
            "{}",
            path.display()
        );
        eprintln!(
            "gain-map camera {}/{}: {}",
            i + 1,
            entries.len(),
            path.display()
        );
        verify(&bytes, &Path::new(&output).join(path.file_stem().unwrap()));
    }
}

#[derive(Clone, Copy, Debug)]
enum ExifCase {
    Absent,
    Value(u16),
    Truncated,
}

fn with_exif_case(jpeg: &[u8], case: ExifCase, big: bool) -> Vec<u8> {
    let tag = match case {
        ExifCase::Absent => return jpeg.to_vec(),
        ExifCase::Value(tag) => tag,
        ExifCase::Truncated => 1,
    };
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend(if big {
        b"MM\0\x2a\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01".as_slice()
    } else {
        b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".as_slice()
    });
    payload.extend(if big {
        tag.to_be_bytes()
    } else {
        tag.to_le_bytes()
    });
    payload.extend([0; 6]);
    if matches!(case, ExifCase::Truncated) {
        payload.truncate(22);
    }
    let mut output = jpeg[..2].to_vec();
    output.extend([0xff, 0xe1]);
    output.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    output.extend(payload);
    output.extend(&jpeg[2..]);
    output
}

fn with_xmp_orientation(jpeg: &[u8], tag: u16) -> Vec<u8> {
    let payload = [b"http://ns.adobe.com/xap/1.0/\0".as_slice(),
        format!("<x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:tiff='http://ns.adobe.com/tiff/1.0/'><tiff:Orientation>{tag}</tiff:Orientation></x:xmpmeta>").as_bytes()].concat();
    let mut output = jpeg[..2].to_vec();
    output.extend([0xff, 0xe1]);
    output.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    output.extend(payload);
    output.extend(&jpeg[2..]);
    output
}

/// The gain map must share primary orientation; its own EXIF is ignored.
/// https://developer.android.com/media/platform/hdr-image-format#gain_map_compression
#[test]
fn iso_gain_map_follows_primary_with_absent_or_invalid_exif() {
    let root = jxl_encoder::test_helpers::output_dir("gain-map-exif-fallback");
    let root = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", path.display()),
            }
        })
        .unwrap();
    let cases: Vec<_> = std::iter::once(ExifCase::Absent)
        .chain((0..=9).map(ExifCase::Value))
        .chain(std::iter::once(ExifCase::Truncated))
        .collect();
    let mut failures = Vec::new();
    let mut cells = 0;
    for (width, height) in [(64, 32), (259, 133)] {
        let jpeg = jpeg_crop(width, height);
        for &primary_case in &cases {
            let tag = match primary_case {
                ExifCase::Value(tag @ 1..=8) => tag,
                _ => 1,
            };
            let clean_reference = with_exif_orientation(&jpeg, tag);
            for &secondary_case in &cases {
                for primary_big in [false, true] {
                    for secondary_big in [false, true] {
                        let primary = with_xmp_orientation(
                            &with_exif_case(&jpeg, primary_case, primary_big),
                            if tag == 8 { 1 } else { 8 },
                        );
                        let secondary = with_iso_metadata(&with_xmp_orientation(
                            &with_exif_case(&jpeg, secondary_case, secondary_big),
                            if tag == 2 { 8 } else { 2 },
                        ));
                        let mut original = primary;
                        original.extend(secondary);
                        let label = format!(
                            "{width}x{height}-p{primary_case:?}-{primary_big}-s{secondary_case:?}-{secondary_big}"
                        );
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            verify_reference(
                                &original,
                                &root.join(&label),
                                Some((&clean_reference, tag)),
                            );
                        }));
                        if result.is_err() {
                            failures.push(label);
                        }
                        cells += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cells, 1152);
    assert!(
        failures.is_empty(),
        "{} of {cells} gain-map orientation cases failed: {failures:?}",
        failures.len()
    );
}

fn with_icc_profile(jpeg: &[u8], profile: &[u8]) -> Vec<u8> {
    let mut payload = b"ICC_PROFILE\0".to_vec();
    payload.extend([1, 1]);
    payload.extend(profile);
    let mut output = jpeg[..2].to_vec();
    output.extend([0xff, 0xe2]);
    output.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    output.extend(payload);
    output.extend(&jpeg[2..]);
    output
}

/// Gain-map ICC is not used; primary ICC and the original JPEG bytes remain.
/// https://developer.android.com/media/platform/hdr-image-format#gain_map_compression
#[test]
fn iso_gain_map_ignores_secondary_icc_and_preserves_primary_profile() {
    let root = jxl_encoder::test_helpers::output_dir("gain-map-icc-policy");
    let root = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", path.display()),
            }
        })
        .unwrap();
    let profile: &[u8] = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let config = LosslessConfig::new().with_effort(3);
    let mut failures = Vec::new();
    let mut cells = 0;
    for (width, height) in [(64, 32), (259, 133)] {
        let jpeg = jpeg_crop(width, height);
        for primary_case in [
            ExifCase::Absent,
            ExifCase::Value(0),
            ExifCase::Value(1),
            ExifCase::Value(6),
        ] {
            let tag = match primary_case {
                ExifCase::Value(tag @ 1..=8) => tag,
                _ => 1,
            };
            let clean_reference = with_exif_orientation(&jpeg, tag);
            for secondary_tag in [2, 8] {
                for primary_icc in [false, true] {
                    let mut primary = with_exif_case(&jpeg, primary_case, false);
                    if primary_icc {
                        primary = with_icc_profile(&primary, profile);
                    }
                    let secondary = with_iso_metadata(&with_icc_profile(
                        &with_exif_orientation(&jpeg, secondary_tag),
                        profile,
                    ));
                    let mut original = primary;
                    original.extend(secondary);
                    let label = format!(
                        "{width}x{height}-p{primary_case:?}-s{secondary_tag}-icc{primary_icc}"
                    );
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let encoded = config.encode_jpeg_transcode(&original).unwrap();
                        let base = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&encoded))
                            .unwrap();
                        assert_eq!(
                            base.original_icc(),
                            if primary_icc { Some(profile) } else { None }
                        );
                        let (_, gain) = gain_map_bundle(&encoded);
                        let gain = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(gain))
                            .unwrap();
                        assert!(
                            gain.original_icc().is_none(),
                            "gain-map profile must be ignored"
                        );
                        verify_reference(
                            &original,
                            &root.join(&label),
                            Some((&clean_reference, tag)),
                        );
                    }));
                    if result.is_err() {
                        failures.push(label);
                    }
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 32);
    assert!(
        failures.is_empty(),
        "{} of {cells} gain-map ICC cases failed: {failures:?}",
        failures.len()
    );
}
