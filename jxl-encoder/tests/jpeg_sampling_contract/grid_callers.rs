//! Additional absolute-factor callers with a nonzero AC and trailing ZRL input.
use super::*;
type Frames = Vec<(Vec<f32>, Vec<Vec<f32>>)>;
type Rgb16Image = image::ImageBuffer<image::Rgb<u16>, Vec<u16>>;
fn source(h: u8, v: u8, transpose: bool) -> Vec<u8> {
    let text = include_str!("../fixtures/jbrd/scan_long-codes-nonzero-before.jpg.hex");
    let mut input: Vec<u8> = text
        .trim()
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect();
    let sof = header_segments(&input)
        .into_iter()
        .find(|r| r.1 == 0xc0)
        .unwrap()
        .0;
    assert_eq!(input[sof + 9], 1);
    input[sof + 11] = h * 16 + v;
    if transpose {
        input[sof + 5..sof + 7].copy_from_slice(&16u16.to_be_bytes());
        input[sof + 7..sof + 9].copy_from_slice(&8u16.to_be_bytes());
    }
    input
}
fn reference(root: &std::path::Path, input: &[u8]) -> (Frames, Rgb16Image) {
    let original = root.join("original.jpg");
    std::fs::write(&original, input).unwrap();
    let reference = root.join("reference.jxl");
    let r = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
        .arg(&original)
        .arg(&reference)
        .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
        .output()
        .unwrap();
    std::fs::write(
        root.join("reference-encode.log"),
        [r.stdout, r.stderr].concat(),
    )
    .unwrap();
    assert!(r.status.success());
    native(
        &reference,
        &root.join("reference.jpg"),
        &root.join("reference-rebuild.log"),
    );
    assert_eq!(std::fs::read(root.join("reference.jpg")).unwrap(), input);
    native(
        &reference,
        &root.join("reference.png"),
        &root.join("reference-render.log"),
    );
    (
        decoded_frames::decode_current_frames(
            &std::fs::read(reference).unwrap(),
            "native partial-grid source",
        ),
        image::open(root.join("reference.png")).unwrap().to_rgb16(),
    )
}
#[test]
fn partial_absolute_grid_bare_default_effort_and_recompression_callers() {
    let root = directory();
    for transpose in [false, true] {
        for (h, v) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            let case = root.join(format!("{transpose}-{h}-{v}"));
            std::fs::create_dir(&case).unwrap();
            let input = source(h, v, transpose);
            let (expected_primary, expected_native) = reference(&case, &input);
            let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
            assert!(!parsed.scan_info[0].extra_zero_runs.is_empty());
            let mut variants = vec![
                (
                    "bare-default".to_string(),
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
            }
            for (policy, strategy, mode) in encoder_policies::policies() {
                let config = LosslessConfig::new()
                    .with_effort(3)
                    .with_strategy(strategy)
                    .with_mode(mode);
                let raw = config.encode_jpeg_transcode(&input).unwrap();
                assert_eq!(
                    raw,
                    config
                        .encode_jpeg_transcode_with_stop(&input, &enough::Unstoppable)
                        .unwrap()
                );
                variants.push((format!("cancellable-{policy}"), raw, true));
                let bare = config.encode_jpeg_transcode_codestream(&input).unwrap();
                assert_eq!(
                    bare,
                    config
                        .encode_jpeg_transcode_codestream_with_stop(&input, &enough::Unstoppable)
                        .unwrap()
                );
                variants.push((format!("codestream-{policy}"), bare, false));
            }
            for (label, bytes, reconstruction) in variants {
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
                if reconstruction {
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
            let floor = LosslessConfig::new()
                .with_effort(3)
                .encode_jpeg_transcode_codestream(&input)
                .unwrap();
            for scale in [1.0, 2.0] {
                for (label, bytes) in [
                    (
                        "explicit",
                        jpeg::encode_jpeg_recompress_codestream(
                            &input,
                            scale,
                            0.5,
                            3,
                            None,
                            Some(&enough::Unstoppable),
                        )
                        .unwrap(),
                    ),
                    (
                        "auto",
                        jpeg::encode_jpeg_recompress_auto_codestream(
                            &input,
                            scale,
                            3,
                            None,
                            Some(&enough::Unstoppable),
                        )
                        .unwrap(),
                    ),
                    (
                        "planar",
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
                        .unwrap(),
                    ),
                ] {
                    if scale == 1.0 {
                        assert_eq!(bytes, floor, "{label} unchanged lossless floor");
                    }
                    let path = case.join(format!("recompress-{label}-{scale}.jxl"));
                    std::fs::write(&path, &bytes).unwrap();
                    let primary = decoded_frames::decode_current_frames(&bytes, label);
                    native(
                        &path,
                        &path.with_extension("png"),
                        &path.with_extension("render.log"),
                    );
                    let native = image::open(path.with_extension("png")).unwrap().to_luma16();
                    assert_eq!(primary.len(), 1);
                    assert_eq!(
                        native.dimensions(),
                        if transpose { (8, 16) } else { (16, 8) }
                    );
                    assert_eq!(primary[0].0.len(), 128);
                    for (&a, &b) in primary[0].0.iter().zip(native.as_raw()) {
                        assert!((a.clamp(0.0, 1.0) * 65535.0 - f32::from(b)).abs() <= 2.0);
                    }
                }
            }
        }
    }
}
#[test]
fn partial_absolute_grid_embedded_gain_map_keeps_nonzero_samples_and_tail() {
    let root = directory();
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
    for transpose in [false, true] {
        for (h, v) in [(2, 1), (1, 2), (2, 2)] {
            let case = root.join(format!("{transpose}-{h}-{v}"));
            std::fs::create_dir(&case).unwrap();
            let input = source(h, v, transpose);
            let (expected_primary, expected_native) = reference(&case, &input);
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
            original.extend(&secondary);
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
                let gainpath = case.join(format!("{policy}-gain.jxl"));
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
                let path = case.join(format!("{policy}.jxl"));
                std::fs::write(&path, &bytes).unwrap();
                native(
                    &path,
                    &path.with_extension("png"),
                    &path.with_extension("render.log"),
                );
                assert_eq!(
                    decoded_frames::decode_current_frames(&bytes, &policy).len(),
                    1
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
    }
}
