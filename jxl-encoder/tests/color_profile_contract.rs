//! ICC channel-model admission, Part 1 E.2.
#![cfg(feature = "__expert")]
use jxl_encoder::{
    AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout,
};
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/icc_contract.rs"]
mod icc_contract;
use icc_contract::{new_directory, profiles, retain_and_decode_stream};

#[test]
fn mismatched_icc_channel_models_are_refused() {
    let profiles = profiles();
    let directory = new_directory("model-before");
    let mut accepted = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in [
            PixelLayout::Rgb8,
            PixelLayout::Rgba8,
            PixelLayout::Rgb16,
            PixelLayout::Rgba16,
            PixelLayout::Gray8,
            PixelLayout::GrayAlpha8,
            PixelLayout::Gray16,
            PixelLayout::GrayAlpha16,
        ] {
            let gray = layout.is_grayscale();
            let pixels = vec![64; 8 * 8 * layout.bytes_per_pixel()];
            let metadata = ImageMetadata::new().with_icc_profile(&profiles[usize::from(!gray)]);
            let lossy = LossyConfig::new(1.0)
                .with_effort(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let lossless = LosslessConfig::new()
                .with_effort(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for animation in [false, true] {
                let a = lossy.encode_request(8, 8, layout).with_metadata(&metadata);
                let b = lossless
                    .encode_request(8, 8, layout)
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
                for (kind, result) in [("lossy", a), ("lossless", b)] {
                    if let Ok(bytes) = result {
                        let label = format!("{name}-{layout:?}-{animation}-{kind}");
                        retain_and_decode_stream(&directory, &label, &bytes, (8, 8));
                        accepted.push(label);
                    }
                }
            }
            let frames = [AnimationFrame::new(&pixels, 10)];
            assert!(
                lossy
                    .encode_animation_with_metadata(
                        8,
                        8,
                        layout,
                        &AnimationParams::default(),
                        &frames,
                        &metadata
                    )
                    .is_err()
            );
            assert!(
                lossless
                    .encode_animation_with_metadata(
                        8,
                        8,
                        layout,
                        &AnimationParams::default(),
                        &frames,
                        &metadata
                    )
                    .is_err()
            );
            for lossy_path in [false, true] {
                for destination in [false, true] {
                    let request = if lossy_path {
                        lossy.encode_request(8, 8, layout)
                    } else {
                        lossless.encode_request(8, 8, layout)
                    }
                    .with_metadata(&metadata);
                    let mut output = vec![0x42, 0x17];
                    let result = if destination {
                        request.encode_to(&pixels, &mut output)
                    } else {
                        request.encode_into(&pixels, &mut output)
                    };
                    assert!(result.is_err());
                    assert_eq!(output, [0x42, 0x17]);
                }
            }
            let mut stream = lossy
                .encoder(8, 8, layout)
                .unwrap()
                .with_icc_profile(metadata.icc_profile().unwrap());
            let result = stream.push_rows(&pixels, 8).and_then(|()| stream.finish());
            if let Ok(bytes) = result {
                let label = format!("{name}-{layout:?}-lossy-stream");
                retain_and_decode_stream(&directory, &label, &bytes, (8, 8));
                accepted.push(label);
            }
            let mut stream = lossless
                .encoder(8, 8, layout)
                .unwrap()
                .with_icc_profile(metadata.icc_profile().unwrap());
            let result = stream.push_rows(&pixels, 8).and_then(|()| stream.finish());
            if let Ok(bytes) = result {
                let label = format!("{name}-{layout:?}-lossless-stream");
                retain_and_decode_stream(&directory, &label, &bytes, (8, 8));
                accepted.push(label);
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted mismatched ICC models: {accepted:?}; streams at {}",
        directory.display()
    );
}

#[test]
fn canonicalization_keeps_icc_presence_and_channel_model_consistent() {
    use jxl_encoder::ColorEncoding;
    for (profile, color) in [
        (profiles().remove(0), ColorEncoding::srgb()),
        (
            zenpixels_convert::icc_profiles::DISPLAY_P3_V4.to_vec(),
            ColorEncoding::display_p3(),
        ),
    ] {
        let metadata = ImageMetadata::new().with_icc_profile(&profile);
        for (name, strategy, mode) in encoder_policies::policies() {
            let config = LossyConfig::new(1.0)
                .with_effort(1)
                .with_strategy(strategy)
                .with_mode(mode)
                .with_canonicalize_input(true);
            for layout in [PixelLayout::Rgba8, PixelLayout::Rgba16] {
                for (width, height) in [(8, 8), (259, 17)] {
                    let pixel = if layout == PixelLayout::Rgba8 {
                        vec![64_u8, 64, 64, 255]
                    } else {
                        [64_u16 * 257, 64 * 257, 64 * 257, 65535]
                            .into_iter()
                            .flat_map(u16::to_ne_bytes)
                            .collect()
                    };
                    let pixels = pixel.repeat(width * height);
                    let row_bytes = width * layout.bytes_per_pixel();
                    let stride = row_bytes + 7;
                    let mut strided = vec![0xa5; stride * height];
                    for y in 0..height {
                        strided[y * stride..y * stride + row_bytes]
                            .copy_from_slice(&pixels[y * row_bytes..(y + 1) * row_bytes]);
                    }
                    let mut expected = None;
                    for explicit_flag in [None, Some(false), Some(true)] {
                        let request = || {
                            let request = config
                                .encode_request(width as u32, height as u32, layout)
                                .with_metadata(&metadata);
                            if let Some(flag) = explicit_flag {
                                let mut color = color.clone();
                                color.want_icc = flag;
                                request.with_color_encoding(color)
                            } else {
                                request
                            }
                        };
                        let encoded = request().encode(&pixels).unwrap();
                        assert_eq!(
                            encoded,
                            request().with_row_stride(stride).encode(&strided).unwrap(),
                            "{name} strided canonicalization"
                        );
                        let directory = new_directory("canonical-model");
                        std::fs::write(directory.join("source.icc"), &profile).unwrap();
                        std::fs::write(directory.join("parameters.txt"), format!("policy={name} explicit_want_icc={explicit_flag:?} layout={layout:?} size={width}x{height} rgba=64,64,64,255 (16-bit replicated)\n")).unwrap();
                        let (decoded, djxl) = retain_and_decode_stream(
                            &directory,
                            "encoded",
                            &encoded,
                            (width, height),
                        );
                        let decoded =
                            decoded.expect("canonicalized output must render in primary decoder");
                        assert!(
                            djxl,
                            "{name}: reference rendering failed at {}",
                            directory.display()
                        );
                        let channels = decoded.len() / (width * height);
                        assert!(matches!(channels, 1 | 3));
                        let image = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&encoded))
                            .unwrap();
                        if let Some(profile) = image.original_icc() {
                            assert_eq!(
                                profile.get(16..20) == Some(b"GRAY"),
                                channels == 1,
                                "{name} canonical ICC model"
                            );
                            assert_eq!(icc_contract::embedded_icc(&encoded), profile);
                        }
                        if let Some(expected) = &expected {
                            assert_eq!(
                                &encoded, expected,
                                "{name} explicit selector must not change canonicalized output"
                            );
                        }
                        expected = Some(encoded);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn jpeg_icc_channel_model_mismatches_are_refused() {
    use jxl_encoder::Unstoppable;
    let profiles = profiles();
    let mut accepted = Vec::new();
    for (gray, source) in [
        (false, &include_bytes!("fixtures/jbrd/base_b_444.jpg")[..]),
        (true, &include_bytes!("fixtures/jbrd/gray_a.jpg")[..]),
    ] {
        let mut payload = b"ICC_PROFILE\0".to_vec();
        payload.extend([1, 1]);
        payload.extend(&profiles[usize::from(!gray)]);
        let mut jpeg = source[..2].to_vec();
        jpeg.extend([0xff, 0xe2]);
        jpeg.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        jpeg.extend(payload);
        jpeg.extend(&source[2..]);
        let parsed = jxl_encoder::jpeg::read_jpeg(&jpeg, None, None).unwrap();
        let directory = new_directory("jpeg-model-before");
        std::fs::write(directory.join("source.jpg"), &jpeg).unwrap();
        for effort in [1, 7, 9] {
            let config = LosslessConfig::new().with_effort(effort);
            for (route, result) in [
                (
                    "parsed-bare",
                    jxl_encoder::jpeg::encode_jpeg_to_jxl_with_effort(&parsed, effort),
                ),
                (
                    "parsed-container",
                    jxl_encoder::jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, effort),
                ),
            ] {
                if let Ok(bytes) = result {
                    std::fs::write(directory.join(format!("{route}-{effort}.jxl")), bytes).unwrap();
                    accepted.push(format!("gray={gray} effort={effort} {route}"));
                }
            }
            for (route, result) in [
                ("raw-container", config.encode_jpeg_transcode(&jpeg)),
                ("raw-bare", config.encode_jpeg_transcode_codestream(&jpeg)),
                (
                    "raw-container-stop",
                    config.encode_jpeg_transcode_with_stop(&jpeg, &Unstoppable),
                ),
                (
                    "raw-bare-stop",
                    config.encode_jpeg_transcode_codestream_with_stop(&jpeg, &Unstoppable),
                ),
            ] {
                if let Ok(bytes) = result {
                    std::fs::write(directory.join(format!("{route}-{effort}.jxl")), bytes).unwrap();
                    accepted.push(format!("gray={gray} effort={effort} {route}"));
                }
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted JPEG ICC model mismatch: {accepted:?}"
    );
}

#[cfg(all(feature = "jpeg-reencoding", feature = "chroma-subsampling"))]
#[test]
fn subsampled_requests_refuse_gray_profiles_for_rgb_samples() {
    use jxl_encoder::ChromaSubsampling;
    let gray = profiles().remove(1);
    let metadata = ImageMetadata::new().with_icc_profile(&gray);
    let pixels = [64_u8, 96, 128].repeat(8 * 8);
    for (_, strategy, mode) in encoder_policies::policies() {
        for sampling in [
            ChromaSubsampling::Sub420,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub440,
        ] {
            let error = LossyConfig::new(1.0)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_chroma_subsampling(sampling)
                .encode_request(8, 8, PixelLayout::Rgb8)
                .with_metadata(&metadata)
                .encode(&pixels)
                .unwrap_err();
            assert!(format!("{error:?}").contains("ICC profile grayscale model"));
        }
    }
}

#[test]
fn lossy_gray_profiles_match_animation_and_still_output_models() {
    let gray = profiles().remove(1);
    let metadata = ImageMetadata::new().with_icc_profile(&gray);
    let directory = new_directory("valid-lossy-gray");
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in [
            PixelLayout::Gray8,
            PixelLayout::GrayAlpha8,
            PixelLayout::Gray16,
            PixelLayout::GrayAlpha16,
        ] {
            for (width, height) in [(64, 32), (259, 17)] {
                let channels = layout.bytes_per_pixel() / if layout.is_16bit() { 2 } else { 1 };
                let codes: Vec<u16> = (0..width * height)
                    .flat_map(|i| {
                        let color = if i % 7 < 3 {
                            0
                        } else if layout.is_16bit() {
                            65535
                        } else {
                            255
                        };
                        let alpha = if layout.is_16bit() { 65535 } else { 255 };
                        [color, alpha].into_iter().take(channels)
                    })
                    .collect();
                let pixels: Vec<u8> = if layout.is_16bit() {
                    codes.iter().flat_map(|v| v.to_ne_bytes()).collect()
                } else {
                    codes.iter().map(|v| *v as u8).collect()
                };
                for effort in [1, 7, 8] {
                    let config = LossyConfig::new(1.0)
                        .with_effort(effort)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    for animation in [false, true] {
                        let request = config
                            .encode_request(width as u32, height as u32, layout)
                            .with_metadata(&metadata);
                        let frames = [AnimationFrame::new(&pixels, 10)];
                        let encoded = if animation {
                            request.encode_animation(&AnimationParams::default(), &frames)
                        } else {
                            request.encode(&pixels)
                        }
                        .unwrap();
                        let label = format!("{name}-{layout:?}-{width}-{effort}-{animation}");
                        let (decoded, djxl) =
                            retain_and_decode_stream(&directory, &label, &encoded, (width, height));
                        assert!(djxl, "{label}");
                        assert_eq!(
                            decoded.unwrap().len(),
                            width * height,
                            "{label} must report gray samples"
                        );
                        assert_eq!(icc_contract::embedded_icc(&encoded), gray);
                        let image = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&encoded))
                            .unwrap();
                        assert_eq!(image.original_icc().unwrap(), gray);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "__pre_quantized")]
#[test]
fn precomputed_and_prequantized_entry_points_check_final_icc_pairs() {
    use jxl_encoder::__pre_quantized::DistanceParams;
    use jxl_encoder::__pre_quantized::{EncoderPrecomputed, VarDctEncoder};
    let profiles = profiles();
    let directory = new_directory("precomputed-model");
    for gray in [false, true] {
        for (width, height) in [(32, 16), (259, 17)] {
            let mut encoder = VarDctEncoder::new(1.0);
            encoder.is_grayscale = gray;
            let linear = if gray {
                [0.2_f32; 3]
            } else {
                [0.1_f32, 0.2, 0.3]
            }
            .repeat(width * height);
            let precomputed = EncoderPrecomputed::compute(
                width,
                height,
                &linear,
                encoder.distance,
                encoder.cfl_enabled,
                encoder.ac_strategy_enabled,
                encoder.pixel_domain_loss,
                encoder.enable_noise,
                encoder.enable_denoise,
                encoder.enable_gaborish,
                encoder.force_strategy,
                &encoder.profile,
                encoder.color_encoding.as_ref(),
            )
            .unwrap();
            let qf = vec![1; precomputed.xsize_blocks * precomputed.ysize_blocks];
            let params = DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
            let mut transform_qf = qf.clone();
            let transform = encoder
                .transform_and_quantize_for_test(&precomputed, &mut transform_qf, &params)
                .unwrap();
            let alpha = vec![255; width * height];
            for (case, profile, want_icc, valid) in [
                ("absent", None, false, true),
                (
                    "present",
                    Some(profiles[usize::from(gray)].as_slice()),
                    false,
                    true,
                ),
                (
                    "wrong-model",
                    Some(profiles[usize::from(!gray)].as_slice()),
                    false,
                    false,
                ),
                ("missing", None, true, false),
            ] {
                encoder.icc_profile = profile.map(<[u8]>::to_vec);
                let mut color = if gray {
                    jxl_encoder::ColorEncoding::gray()
                } else {
                    jxl_encoder::ColorEncoding::srgb()
                };
                color.want_icc = want_icc;
                encoder.color_encoding = Some(color);
                for (route, result) in [
                    (
                        "precomputed",
                        encoder.encode_from_precomputed(&precomputed, &qf),
                    ),
                    (
                        "precomputed-alpha",
                        encoder.encode_from_precomputed_with_extras(
                            &precomputed,
                            &qf,
                            &[jxl_encoder::api::ExtraChannel::from_alpha_buf(
                                &alpha, false,
                            )],
                        ),
                    ),
                    (
                        "prequantized",
                        encoder.encode_from_pre_quantized_ac(
                            &precomputed,
                            &qf,
                            &transform.quant_dc,
                            &transform.quant_ac,
                            &transform.nzeros,
                            &transform.raw_nzeros,
                        ),
                    ),
                    (
                        "prequantized-alpha",
                        encoder.encode_from_pre_quantized_ac_with_extras(
                            &precomputed,
                            &qf,
                            &transform.quant_dc,
                            &transform.quant_ac,
                            &transform.nzeros,
                            &transform.raw_nzeros,
                            &[jxl_encoder::api::ExtraChannel::from_alpha_buf(
                                &alpha, false,
                            )],
                        ),
                    ),
                ] {
                    let label = format!("{gray}-{width}-{case}-{route}");
                    assert_eq!(result.is_ok(), valid, "{label}: {result:?}");
                    if valid {
                        let bytes = result.unwrap();
                        let (rust, djxl) =
                            retain_and_decode_stream(&directory, &label, &bytes, (width, height));
                        assert_eq!(
                            rust.unwrap().len(),
                            width * height * if gray { 1 } else { 3 }
                        );
                        assert!(djxl, "{label}");
                        if let Some(profile) = profile {
                            assert_eq!(icc_contract::embedded_icc(&bytes), profile);
                        }
                        if route.ends_with("alpha") {
                            assert!(
                                image::open(directory.join(format!("{label}.pam")))
                                    .unwrap()
                                    .to_rgba8()
                                    .pixels()
                                    .all(|p| p[3] == 255)
                            );
                        }
                    } else {
                        assert!(format!("{:?}", result.unwrap_err()).contains("InvalidInput"));
                    }
                }
            }
        }
    }
}

#[test]
fn streaming_profile_changes_recheck_input_and_preserve_destinations() {
    let profiles = profiles();
    for (_, strategy, mode) in encoder_policies::policies() {
        let lossy = LossyConfig::new(1.0)
            .with_effort(1)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        let lossless = LosslessConfig::new()
            .with_effort(1)
            .with_strategy(strategy)
            .with_mode(mode);
        for gray in [false, true] {
            let layout = if gray {
                PixelLayout::Gray8
            } else {
                PixelLayout::Rgb8
            };
            let row = vec![64; 8 * 4 * layout.bytes_per_pixel()];
            let wrong = &profiles[usize::from(!gray)];
            macro_rules! check {
                ($config:expr) => {{
                    let mut stream = $config
                        .encoder(8, 8, layout)
                        .unwrap()
                        .with_icc_profile(wrong);
                    assert!(stream.push_rows(&row, 4).is_err());
                    assert_eq!(stream.rows_pushed(), 0);
                    let mut stream = $config.encoder(8, 8, layout).unwrap();
                    stream.push_rows(&row, 4).unwrap();
                    stream = stream.with_icc_profile(wrong);
                    assert!(stream.push_rows(&row, 4).is_err());
                    assert_eq!(stream.rows_pushed(), 4);
                    let mut stream = $config.encoder(8, 8, layout).unwrap();
                    stream.push_rows(&row.repeat(2), 8).unwrap();
                    let stream = stream.with_icc_profile(wrong);
                    let mut output = vec![0x42, 0x17];
                    assert!(stream.finish_to(&mut output).is_err());
                    assert_eq!(output, [0x42, 0x17]);
                }};
            }
            check!(lossy);
            check!(lossless);
        }
    }
}

fn input_layouts() -> [PixelLayout; 26] {
    use PixelLayout::*;
    [
        Rgb8,
        Rgba8,
        Bgr8,
        Bgra8,
        Gray8,
        GrayAlpha8,
        Rgb16,
        Rgba16,
        Gray16,
        GrayAlpha16,
        RgbLinearF32,
        RgbaLinearF32,
        GrayLinearF32,
        GrayAlphaLinearF32,
        RgbLinearF16,
        RgbaLinearF16,
        GrayLinearF16,
        GrayAlphaLinearF16,
        RgbPqF32,
        RgbaPqF32,
        RgbHlgF32,
        RgbaHlgF32,
        RgbBt709F32,
        RgbaBt709F32,
        Cmyk8,
        Cmyk16,
    ]
}

#[test]
fn every_input_layout_refuses_opposite_icc_models_before_pixel_dispatch() {
    let profiles = profiles();
    for (_, strategy, mode) in encoder_policies::policies() {
        for layout in input_layouts() {
            let pixels = vec![0; 8 * 8 * layout.bytes_per_pixel()];
            let metadata = ImageMetadata::new()
                .with_icc_profile(&profiles[usize::from(!layout.is_grayscale())]);
            let lossy = LossyConfig::new(1.0)
                .with_effort(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let lossless = LosslessConfig::new()
                .with_effort(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for result in [
                lossy
                    .encode_request(8, 8, layout)
                    .with_metadata(&metadata)
                    .encode(&pixels),
                lossless
                    .encode_request(8, 8, layout)
                    .with_metadata(&metadata)
                    .encode(&pixels),
            ] {
                assert!(
                    format!("{:?}", result.unwrap_err()).contains("ICC profile grayscale model"),
                    "{layout:?}"
                );
            }
        }
    }
}

#[test]
fn matched_float_and_bgr_profile_models_keep_streaming_and_strided_bytes() {
    let profiles = profiles();
    let directory = new_directory("float-profile-models");
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in input_layouts().into_iter().filter(|layout| {
            layout.is_f16()
                || layout.is_f32()
                || matches!(layout, PixelLayout::Bgr8 | PixelLayout::Bgra8)
        }) {
            let pixels = vec![0; 9 * 7 * layout.bytes_per_pixel()];
            let profile = &profiles[usize::from(layout.is_grayscale())];
            let metadata = ImageMetadata::new().with_icc_profile(profile);
            let lossy = LossyConfig::new(1.0)
                .with_effort(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let lossless = LosslessConfig::new()
                .with_effort(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for lossless_path in [false, true] {
                let request = || {
                    if lossless_path {
                        lossless.encode_request(9, 7, layout)
                    } else {
                        lossy.encode_request(9, 7, layout)
                    }
                    .with_metadata(&metadata)
                };
                let result = request().encode(&pixels);
                if lossless_path
                    && matches!(
                        layout,
                        PixelLayout::RgbPqF32
                            | PixelLayout::RgbaPqF32
                            | PixelLayout::RgbHlgF32
                            | PixelLayout::RgbaHlgF32
                            | PixelLayout::RgbBt709F32
                            | PixelLayout::RgbaBt709F32
                    )
                {
                    assert!(
                        format!("{:?}", result.unwrap_err()).contains("UnsupportedPixelLayout")
                    );
                    continue;
                }
                let encoded = result.unwrap();
                let row_bytes = 9 * layout.bytes_per_pixel();
                let stride = row_bytes + 3;
                let mut padded = vec![0xff; stride * 7];
                for row in padded.chunks_exact_mut(stride) {
                    row[..row_bytes].fill(0);
                }
                assert_eq!(
                    request().with_row_stride(stride).encode(&padded).unwrap(),
                    encoded
                );
                let streaming = if lossless_path {
                    let mut stream = lossless
                        .encoder(9, 7, layout)
                        .unwrap()
                        .with_icc_profile(profile);
                    for rows in [3, 4] {
                        stream
                            .push_rows(&pixels[..rows * row_bytes], rows as u32)
                            .unwrap();
                    }
                    stream.finish().unwrap()
                } else {
                    let mut stream = lossy
                        .encoder(9, 7, layout)
                        .unwrap()
                        .with_icc_profile(profile);
                    for rows in [3, 4] {
                        stream
                            .push_rows(&pixels[..rows * row_bytes], rows as u32)
                            .unwrap();
                    }
                    stream.finish().unwrap()
                };
                assert_eq!(streaming, encoded);
                let label = format!("{name}-{layout:?}-{lossless_path}");
                let (rust, djxl) = icc_contract::retain_and_decode_stream_as(
                    &directory,
                    &label,
                    &encoded,
                    (9, 7),
                    layout.is_f16() || layout.is_f32(),
                );
                let samples = rust.unwrap();
                assert!(djxl, "{label}");
                if lossless_path {
                    assert!(samples.iter().all(|v| v.to_bits() == 0), "{label}");
                }
                if lossless_path {
                    let exact = if layout.is_f16() || layout.is_f32() {
                        icc_contract::read_pfm_samples(
                            &directory.join(format!("{label}.pfm")),
                            (9, 7),
                        )
                        .into_iter()
                        .all(|v| v.to_bits() == 0)
                    } else {
                        image::open(directory.join(format!("{label}.pam")))
                            .unwrap()
                            .to_rgba8()
                            .pixels()
                            .all(|p| p.0[..3] == [0, 0, 0])
                    };
                    assert!(exact, "{label}: exact reference source colors");
                }

                assert_eq!(icc_contract::embedded_icc(&encoded), *profile);
                assert_eq!(
                    std::fs::read(directory.join(format!("{label}.icc"))).unwrap(),
                    *profile
                );
            }
        }
    }
}
