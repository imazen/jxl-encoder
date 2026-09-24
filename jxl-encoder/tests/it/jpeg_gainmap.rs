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

fn verify(original: &[u8], dir: &Path) {
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
    let standalone = config.encode_jpeg_transcode_codestream(secondary).unwrap();
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
    std::fs::write(dir.join("gain-map.jpg"), secondary).unwrap();
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
