//! ICC tag-count admission, Part 1 E.4.4.
#![cfg(feature = "__expert")]
use jxl_encoder::{
    AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout,
};
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/icc_contract.rs"]
mod icc_contract;
use icc_contract::{new_directory, profiles, retain_and_decode_stream};

fn verify_opaque(
    bytes: &[u8],
    directory: &std::path::Path,
    label: &str,
    icc: &[u8],
    dims: (usize, usize),
    lossless: bool,
) {
    let (pixels, reference) = retain_and_decode_stream(directory, label, bytes, dims);
    let pixels = pixels.expect("primary decoder must render opaque ICC coding");
    assert!(reference, "reference must render {label}");
    assert_eq!(
        std::fs::read(directory.join(format!("{label}.icc"))).unwrap(),
        icc
    );
    assert_eq!(icc_contract::embedded_icc(bytes), icc);
    if lossless {
        assert!(pixels.iter().all(|v| (v * 255.0).round() as u8 == 64));
        let reference = image::open(directory.join(format!("{label}.pam")))
            .unwrap()
            .to_rgb8();
        assert!(reference.as_raw().iter().all(|v| *v == 64));
    }
}

#[test]
fn truncated_icc_tag_table_uses_opaque_prediction() {
    let profiles = profiles();
    let mut malformed = profiles[0][..128].to_vec();
    malformed[..4].copy_from_slice(&132_u32.to_be_bytes());
    malformed.extend_from_slice(&1_u32.to_be_bytes());
    let metadata = ImageMetadata::new().with_icc_profile(&malformed);
    let pixels = vec![64; 8 * 8 * 3];
    let directory = new_directory("tag-opaque");
    for (name, strategy, mode) in encoder_policies::policies() {
        let lossy = LossyConfig::new(1.0)
            .with_effort(1)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        let lossless = LosslessConfig::new()
            .with_effort(1)
            .with_strategy(strategy)
            .with_mode(mode);
        for animation in [false, true] {
            let a = lossy
                .encode_request(8, 8, PixelLayout::Rgb8)
                .with_metadata(&metadata);
            let b = lossless
                .encode_request(8, 8, PixelLayout::Rgb8)
                .with_metadata(&metadata);
            let frames = [AnimationFrame::new(&pixels, 10)];
            let a = if animation {
                a.encode_animation(&AnimationParams::default(), &frames)
            } else {
                a.encode(&pixels)
            };
            let b = if animation {
                b.encode_animation(&AnimationParams::default(), &frames)
            } else {
                b.encode(&pixels)
            };
            for (kind, bytes, exact) in
                [("lossy", a.unwrap(), false), ("lossless", b.unwrap(), true)]
            {
                let label = format!("{name}-{animation}-{kind}");
                verify_opaque(&bytes, &directory, &label, &malformed, (8, 8), exact);
            }
        }
        let mut stream = lossy
            .encoder(8, 8, PixelLayout::Rgb8)
            .unwrap()
            .with_icc_profile(&malformed);
        stream.push_rows(&pixels, 8).unwrap();
        let bytes = stream.finish().unwrap();
        verify_opaque(
            &bytes,
            &directory,
            &format!("{name}-lossy-stream"),
            &malformed,
            (8, 8),
            false,
        );
        let mut stream = lossless
            .encoder(8, 8, PixelLayout::Rgb8)
            .unwrap()
            .with_icc_profile(&malformed);
        stream.push_rows(&pixels, 8).unwrap();
        let bytes = stream.finish().unwrap();
        verify_opaque(
            &bytes,
            &directory,
            &format!("{name}-lossless-stream"),
            &malformed,
            (8, 8),
            true,
        );
        let mut destination = vec![9, 8, 7];
        lossy
            .encode_request(8, 8, PixelLayout::Rgb8)
            .with_metadata(&metadata)
            .encode_into(&pixels, &mut destination)
            .unwrap();
        assert_eq!(&destination[..3], &[9, 8, 7]);
        verify_opaque(
            &destination[3..],
            &directory,
            &format!("{name}-into"),
            &malformed,
            (8, 8),
            false,
        );
    }
}

#[test]
fn valid_rgb_gray_profiles_keep_exact_lossless_samples_and_streaming_bytes() {
    let profiles = profiles();
    let directory = new_directory("valid-profiles");
    for (name, strategy, mode) in encoder_policies::policies() {
        let config = LosslessConfig::new()
            .with_effort(1)
            .with_strategy(strategy)
            .with_mode(mode);
        for gray in [false, true] {
            let profile = &profiles[usize::from(gray)];
            let metadata = ImageMetadata::new().with_icc_profile(profile);
            for wide in [false, true] {
                let layout = match (gray, wide) {
                    (false, false) => PixelLayout::Rgb8,
                    (false, true) => PixelLayout::Rgb16,
                    (true, false) => PixelLayout::Gray8,
                    (true, true) => PixelLayout::Gray16,
                };
                for (width, height) in [(64_usize, 32_usize), (259, 17)] {
                    let channels = if gray { 1 } else { 3 };
                    let codes: Vec<u16> = (0..width * height)
                        .flat_map(|i| {
                            let code = if i % 7 < 3 {
                                0
                            } else if wide {
                                65535
                            } else {
                                255
                            };
                            std::iter::repeat_n(code, channels)
                        })
                        .collect();
                    let pixels: Vec<u8> = if wide {
                        codes.iter().flat_map(|v| v.to_ne_bytes()).collect()
                    } else {
                        codes.iter().map(|v| *v as u8).collect()
                    };
                    let packed = config
                        .encode_request(width as u32, height as u32, layout)
                        .with_metadata(&metadata)
                        .encode(&pixels)
                        .unwrap();
                    for route in ["packed", "strided", "streaming", "animation"] {
                        let bytes = match route {
                            "strided" => {
                                let row_bytes = width * layout.bytes_per_pixel();
                                let stride = row_bytes + 3;
                                let mut rows = vec![0xa5; stride * height];
                                for (src, dst) in pixels
                                    .chunks_exact(row_bytes)
                                    .zip(rows.chunks_exact_mut(stride))
                                {
                                    dst[..row_bytes].copy_from_slice(src);
                                }
                                let bytes = config
                                    .encode_request(width as u32, height as u32, layout)
                                    .with_metadata(&metadata)
                                    .with_row_stride(stride)
                                    .encode(&rows)
                                    .unwrap();
                                assert_eq!(bytes, packed);
                                bytes
                            }
                            "streaming" => {
                                let mut stream = config
                                    .encoder(width as u32, height as u32, layout)
                                    .unwrap()
                                    .with_icc_profile(profile);
                                for rows in pixels.chunks(width * layout.bytes_per_pixel() * 7) {
                                    stream
                                        .push_rows(
                                            rows,
                                            (rows.len() / (width * layout.bytes_per_pixel()))
                                                as u32,
                                        )
                                        .unwrap();
                                }
                                let bytes = stream.finish().unwrap();
                                assert_eq!(bytes, packed);
                                bytes
                            }
                            "animation" => config
                                .encode_request(width as u32, height as u32, layout)
                                .with_metadata(&metadata)
                                .encode_animation(
                                    &AnimationParams::default(),
                                    &[AnimationFrame::new(&pixels, 10)],
                                )
                                .unwrap(),
                            _ => packed.clone(),
                        };
                        let label = format!("{name}-{gray}-{wide}-{width}-{route}");
                        let path = directory.join(format!("{label}.jxl"));
                        std::fs::write(&path, &bytes).unwrap();
                        let samples = icc_contract::decode_pixels(&bytes, width, height);
                        assert_eq!(samples.len(), codes.len());
                        let max = if wide { 65535.0 } else { 255.0 };
                        let samples: Vec<u16> = samples
                            .into_iter()
                            .map(|v| {
                                assert!(v.is_finite());
                                (v * max).round() as u16
                            })
                            .collect();
                        assert_eq!(samples, codes, "{label} primary samples");
                        let result =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(&path)
                                .arg(directory.join(format!("{label}.png")))
                                .arg("--bits_per_sample=16")
                                .arg("--num_threads=1")
                                .output()
                                .unwrap();
                        assert!(
                            result.status.success(),
                            "{label}: {}",
                            String::from_utf8_lossy(&result.stderr)
                        );
                        let samples = image::open(directory.join(format!("{label}.png")))
                            .unwrap()
                            .to_rgb16()
                            .into_raw();
                        let expected: Vec<u16> = if gray {
                            codes
                                .iter()
                                .flat_map(|v| {
                                    std::iter::repeat_n(if wide { *v } else { *v * 257 }, 3)
                                })
                                .collect()
                        } else {
                            codes
                                .iter()
                                .map(|v| if wide { *v } else { *v * 257 })
                                .collect()
                        };
                        assert_eq!(samples, expected, "{label} djxl samples");
                        let image = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&bytes))
                            .unwrap();
                        assert_eq!(image.original_icc().unwrap(), profile);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "jpeg-reencoding")]
fn jpeg_with_icc_profile(jpeg: &[u8], profile: &[u8]) -> Vec<u8> {
    // Same APP2 constructor as it/jpeg_gainmap.rs.
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

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn raw_and_parsed_jpeg_preserve_opaque_icc_and_reconstruction() {
    use jxl_encoder::Unstoppable;
    let directory = new_directory("jpeg-tag-opaque");
    for (gray, source) in [
        (false, &include_bytes!("fixtures/jbrd/base_b_444.jpg")[..]),
        (true, &include_bytes!("fixtures/jbrd/gray_a.jpg")[..]),
    ] {
        let mut malformed = profiles()[usize::from(gray)][..128].to_vec();
        malformed[..4].copy_from_slice(&132_u32.to_be_bytes());
        malformed.extend_from_slice(&1_u32.to_be_bytes());
        let jpeg = jpeg_with_icc_profile(source, &malformed);
        let parsed = jxl_encoder::jpeg::read_jpeg(&jpeg, None, None).unwrap();
        let dims = (parsed.width as usize, parsed.height as usize);
        for effort in [1, 7, 9] {
            let config = LosslessConfig::new().with_effort(effort);
            for (route, bytes, container) in [
                (
                    "parsed-raw",
                    jxl_encoder::jpeg::encode_jpeg_to_jxl_with_effort(&parsed, effort).unwrap(),
                    false,
                ),
                (
                    "parsed-container",
                    jxl_encoder::jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, effort)
                        .unwrap(),
                    true,
                ),
                (
                    "raw-container",
                    config.encode_jpeg_transcode(&jpeg).unwrap(),
                    true,
                ),
                (
                    "raw-bare",
                    config.encode_jpeg_transcode_codestream(&jpeg).unwrap(),
                    false,
                ),
                (
                    "raw-container-stop",
                    config
                        .encode_jpeg_transcode_with_stop(&jpeg, &Unstoppable)
                        .unwrap(),
                    true,
                ),
                (
                    "raw-bare-stop",
                    config
                        .encode_jpeg_transcode_codestream_with_stop(&jpeg, &Unstoppable)
                        .unwrap(),
                    false,
                ),
            ] {
                let label = format!("{gray}-{effort}-{route}");
                verify_opaque(&bytes, &directory, &label, &malformed, dims, false);
                if container {
                    assert_eq!(
                        zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
                        jpeg
                    );
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(directory.join(format!("{label}.jxl")))
                        .arg(directory.join(format!("{label}.jpg")))
                        .arg("--reconstruct_jpeg")
                        .arg("--num_threads=1")
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    assert_eq!(
                        std::fs::read(directory.join(format!("{label}.jpg"))).unwrap(),
                        jpeg
                    );
                }
            }
        }
    }
}

#[cfg(all(feature = "jpeg-reencoding", feature = "chroma-subsampling"))]
#[test]
fn subsampled_requests_preserve_opaque_icc_tag_counts() {
    use jxl_encoder::ChromaSubsampling;
    let mut malformed = profiles()[0][..128].to_vec();
    malformed[..4].copy_from_slice(&132_u32.to_be_bytes());
    malformed.extend_from_slice(&1_u32.to_be_bytes());
    let metadata = ImageMetadata::new().with_icc_profile(&malformed);
    let pixels = [64_u8, 96, 128].repeat(8 * 8);
    for mode in [
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub440,
    ] {
        let result = LossyConfig::new(1.0)
            .with_chroma_subsampling(mode)
            .encode_request(8, 8, PixelLayout::Rgb8)
            .with_metadata(&metadata)
            .encode(&pixels);
        let directory = new_directory("subsampled-tag-opaque");
        verify_opaque(
            &result.unwrap(),
            &directory,
            "encoded",
            &malformed,
            (8, 8),
            false,
        );
    }
}

#[test]
fn opaque_tag_lists_preserve_generic_types_and_long_insertions() {
    use jxl_encoder::api::EncoderStrategy;
    let base = profiles();
    let mut payloads = vec![vec![], vec![0x55; 7], vec![0; 70_000]];
    let mut types = Vec::new();
    for keyword in [
        b"XYZ ", b"curv", b"para", b"sf32", b"text", b"mluc", b"mAB ", b"mBA ",
    ] {
        types.extend(keyword);
        types.extend([0; 4]);
        types.extend((0_u8..24).map(|v| v.wrapping_mul(17)));
    }
    payloads.push(types);
    let directory = new_directory("opaque-contents");
    for strategy in [EncoderStrategy::Libjxl, EncoderStrategy::Zenjxl] {
        for (cell, payload) in payloads.iter().enumerate() {
            let mut icc = base[0][..128].to_vec();
            icc.extend_from_slice(&u32::MAX.to_be_bytes());
            icc.extend(payload);
            let size = icc.len() as u32;
            icc[..4].copy_from_slice(&size.to_be_bytes());
            let metadata = ImageMetadata::new().with_icc_profile(&icc);
            let pixels = [64; 8 * 8 * 3];
            let lossy = LossyConfig::new(1.0)
                .with_effort(1)
                .with_strategy(strategy.clone());
            let lossless = LosslessConfig::new()
                .with_effort(1)
                .with_strategy(strategy.clone());
            for (kind, bytes, exact) in [
                (
                    "lossy",
                    lossy
                        .encode_request(8, 8, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .encode(&pixels)
                        .unwrap(),
                    false,
                ),
                (
                    "lossless",
                    lossless
                        .encode_request(8, 8, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .encode(&pixels)
                        .unwrap(),
                    true,
                ),
            ] {
                verify_opaque(
                    &bytes,
                    &directory,
                    &format!("{strategy:?}-{cell}-{kind}"),
                    &icc,
                    (8, 8),
                    exact,
                );
            }
        }
    }
}
