//! ICC tag-count admission, Part 1 E.4.4.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout};
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/icc_contract.rs"]
mod icc_contract;
use icc_contract::{profiles, new_directory, preserve_unexpected_stream};

#[test]
fn truncated_icc_tag_table_is_refused_before_output() {
    let profiles = profiles();
    let mut malformed = profiles[0][..128].to_vec();
    malformed[..4].copy_from_slice(&132_u32.to_be_bytes());
    malformed.extend_from_slice(&1_u32.to_be_bytes());
    let metadata = ImageMetadata::new().with_icc_profile(&malformed);
    let pixels = vec![64; 8 * 8 * 3];
    let directory = new_directory("tag-before");
    let mut accepted = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        let lossy = LossyConfig::new(1.0).with_effort(1).with_strategy(strategy.clone()).with_mode(mode);
        let lossless = LosslessConfig::new().with_effort(1).with_strategy(strategy).with_mode(mode);
        for animation in [false, true] {
            let a = lossy.encode_request(8, 8, PixelLayout::Rgb8).with_metadata(&metadata);
            let b = lossless.encode_request(8, 8, PixelLayout::Rgb8).with_metadata(&metadata);
            let frames = [AnimationFrame::new(&pixels, 10)];
            let a = if animation { a.encode_animation(&AnimationParams::default(), &frames) } else { a.encode(&pixels) };
            let b = if animation { b.encode_animation(&AnimationParams::default(), &frames) } else { b.encode(&pixels) };
            for (kind, result) in [("lossy", a), ("lossless", b)] {
                if let Ok(bytes) = result {
                    let label = format!("{name}-{animation}-{kind}");
                    preserve_unexpected_stream(&directory, &label, &bytes);
                    accepted.push(label);
                }
            }
        }
        let mut stream = lossy.encoder(8, 8, PixelLayout::Rgb8).unwrap().with_icc_profile(&malformed);
        stream.push_rows(&pixels, 8).unwrap();
        if let Ok(bytes) = stream.finish() {
            let label = format!("{name}-lossy-stream");
            preserve_unexpected_stream(&directory, &label, &bytes);
            accepted.push(label);
        }
        let mut stream = lossless.encoder(8, 8, PixelLayout::Rgb8).unwrap().with_icc_profile(&malformed);
        stream.push_rows(&pixels, 8).unwrap();
        if let Ok(bytes) = stream.finish() {
            let label = format!("{name}-lossless-stream");
            preserve_unexpected_stream(&directory, &label, &bytes);
            accepted.push(label);
        }
        let mut destination = vec![9, 8, 7];
        if lossy.encode_request(8, 8, PixelLayout::Rgb8).with_metadata(&metadata).encode_into(&pixels, &mut destination).is_err() {
            assert_eq!(destination, [9, 8, 7]);
        }
    }
    assert!(accepted.is_empty(), "accepted truncated ICC: {accepted:?}; streams at {}", directory.display());
}

#[test]
fn valid_rgb_gray_profiles_keep_exact_lossless_samples_and_streaming_bytes() {
    let profiles = profiles();
    let directory = new_directory("valid-profiles");
    for (name, strategy, mode) in encoder_policies::policies() {
        let config = LosslessConfig::new().with_effort(1).with_strategy(strategy).with_mode(mode);
        for gray in [false, true] {
            let profile = &profiles[usize::from(gray)];
            let metadata = ImageMetadata::new().with_icc_profile(profile);
            for wide in [false, true] {
                let layout = match (gray, wide) {
                    (false, false) => PixelLayout::Rgb8, (false, true) => PixelLayout::Rgb16,
                    (true, false) => PixelLayout::Gray8, (true, true) => PixelLayout::Gray16,
                };
                for (width, height) in [(64_usize, 32_usize), (259, 17)] {
                    let channels = if gray { 1 } else { 3 };
                    let codes: Vec<u16> = (0..width * height).flat_map(|i| {
                        let code = if i % 7 < 3 { 0 } else if wide { 65535 } else { 255 };
                        std::iter::repeat_n(code, channels)
                    }).collect();
                    let pixels: Vec<u8> = if wide { codes.iter().flat_map(|v| v.to_ne_bytes()).collect() }
                        else { codes.iter().map(|v| *v as u8).collect() };
                    let packed = config.encode_request(width as u32, height as u32, layout).with_metadata(&metadata).encode(&pixels).unwrap();
                    for route in ["packed", "strided", "streaming", "animation"] {
                        let bytes = match route {
                            "strided" => {
                                let row_bytes = width * layout.bytes_per_pixel();
                                let stride = row_bytes + 3;
                                let mut rows = vec![0xa5; stride * height];
                                for (src, dst) in pixels.chunks_exact(row_bytes).zip(rows.chunks_exact_mut(stride)) { dst[..row_bytes].copy_from_slice(src); }
                                let bytes = config.encode_request(width as u32, height as u32, layout).with_metadata(&metadata).with_row_stride(stride).encode(&rows).unwrap();
                                assert_eq!(bytes, packed); bytes
                            }
                            "streaming" => {
                                let mut stream = config.encoder(width as u32, height as u32, layout).unwrap().with_icc_profile(profile);
                                for rows in pixels.chunks(width * layout.bytes_per_pixel() * 7) {
                                    stream.push_rows(rows, (rows.len() / (width * layout.bytes_per_pixel())) as u32).unwrap();
                                }
                                let bytes = stream.finish().unwrap(); assert_eq!(bytes, packed); bytes
                            }
                            "animation" => config.encode_request(width as u32, height as u32, layout).with_metadata(&metadata)
                                .encode_animation(&AnimationParams::default(), &[AnimationFrame::new(&pixels, 10)]).unwrap(),
                            _ => packed.clone(),
                        };
                        let label = format!("{name}-{gray}-{wide}-{width}-{route}");
                        let path = directory.join(format!("{label}.jxl"));
                        std::fs::write(&path, &bytes).unwrap();
                        let samples = icc_contract::decode_pixels(&bytes, width, height);
                        assert_eq!(samples.len(), codes.len());
                        let max = if wide { 65535.0 } else { 255.0 };
                        let samples: Vec<u16> = samples.into_iter().map(|v| { assert!(v.is_finite()); (v * max).round() as u16 }).collect();
                        assert_eq!(samples, codes, "{label} primary samples");
                        let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path()).arg(&path)
                            .arg(directory.join(format!("{label}.png"))).arg("--bits_per_sample=16").arg("--num_threads=1").output().unwrap();
                        assert!(result.status.success(), "{label}: {}", String::from_utf8_lossy(&result.stderr));
                        let samples = image::open(directory.join(format!("{label}.png"))).unwrap().to_rgb16().into_raw();
                        let expected: Vec<u16> = if gray { codes.iter().flat_map(|v| std::iter::repeat_n(if wide { *v } else { *v * 257 }, 3)).collect() }
                            else { codes.iter().map(|v| if wide { *v } else { *v * 257 }).collect() };
                        assert_eq!(samples, expected, "{label} djxl samples");
                        let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&bytes)).unwrap();
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
    let mut payload = b"ICC_PROFILE\0".to_vec(); payload.extend([1, 1]); payload.extend(profile);
    let mut output = jpeg[..2].to_vec(); output.extend([0xff, 0xe2]);
    output.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes()); output.extend(payload); output.extend(&jpeg[2..]); output
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn raw_and_parsed_jpeg_refuse_truncated_icc_tag_counts() {
    for (gray, source) in [
        (false, &include_bytes!("fixtures/jbrd/base_b_444.jpg")[..]),
        (true, &include_bytes!("fixtures/jbrd/gray_a.jpg")[..]),
    ] {
        let mut malformed = profiles()[usize::from(gray)][..128].to_vec();
        malformed[..4].copy_from_slice(&132_u32.to_be_bytes()); malformed.extend_from_slice(&1_u32.to_be_bytes());
        let jpeg = jpeg_with_icc_profile(source, &malformed);
        let parsed = jxl_encoder::jpeg::read_jpeg(&jpeg, None, None).unwrap();
        for effort in [1, 7, 9] {
            assert!(jxl_encoder::jpeg::encode_jpeg_to_jxl_with_effort(&parsed, effort).is_err());
            assert!(jxl_encoder::jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, effort).is_err());
            let config = LosslessConfig::new().with_effort(effort);
            assert!(config.encode_jpeg_transcode(&jpeg).is_err());
            assert!(config.encode_jpeg_transcode_codestream(&jpeg).is_err());
        }
    }
}

#[cfg(all(feature = "jpeg-reencoding", feature = "chroma-subsampling"))]
#[test]
fn subsampled_requests_refuse_truncated_icc_tag_counts() {
    use jxl_encoder::ChromaSubsampling;
    let mut malformed = profiles()[0][..128].to_vec();
    malformed[..4].copy_from_slice(&132_u32.to_be_bytes()); malformed.extend_from_slice(&1_u32.to_be_bytes());
    let metadata = ImageMetadata::new().with_icc_profile(&malformed);
    let pixels = [64_u8, 96, 128].repeat(8 * 8);
    for mode in [ChromaSubsampling::Sub420, ChromaSubsampling::Sub422, ChromaSubsampling::Sub440] {
        let result = LossyConfig::new(1.0).with_chroma_subsampling(mode).encode_request(8, 8, PixelLayout::Rgb8).with_metadata(&metadata).encode(&pixels);
        let error = result.expect_err("invalid ICC count must fail in the shared writer");
        assert!(format!("{error:?}").contains("ICC tag count"), "{error:?}");
    }
}
