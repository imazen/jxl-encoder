use jxl_encoder::{
    AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout,
};

#[test]
fn animation_metadata_embeds_the_supplied_icc_profile() {
    use zensim_decoder::api::{
        JxlColorProfile, JxlDecoder, JxlDecoderOptions, ProcessingResult, states,
    };
    let icc = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let metadata = ImageMetadata::default().with_icc_profile(icc);
    let pixels = [37, 101, 219].repeat(32 * 17);
    let frames = [
        AnimationFrame::new(&pixels, 10),
        AnimationFrame::new(&pixels, 20),
    ];
    for lossy in [false, true] {
        let encoded = if lossy {
            LossyConfig::new(1.0)
                .with_effort(1)
                .encode_animation_with_metadata(
                    32,
                    17,
                    PixelLayout::Rgb8,
                    &AnimationParams::default(),
                    &frames,
                    &metadata,
                )
        } else {
            LosslessConfig::new()
                .with_effort(1)
                .encode_animation_with_metadata(
                    32,
                    17,
                    PixelLayout::Rgb8,
                    &AnimationParams::default(),
                    &frames,
                    &metadata,
                )
        }
        .unwrap();
        let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let ProcessingResult::Complete { result: decoder } =
            decoder.process(&mut encoded.as_slice()).unwrap()
        else {
            panic!("complete encoded file needs more header input");
        };
        match decoder.embedded_color_profile() {
            JxlColorProfile::Icc(profile) => assert_eq!(profile.as_slice(), icc, "lossy={lossy}"),
            _ => panic!("animation discarded the ICC profile: lossy={lossy}"),
        }
        let oxide = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .unwrap();
        assert_eq!(oxide.original_icc(), Some(icc));
    }
}

fn decode_rgb_f32(encoded: &[u8], bits: u32) -> Vec<Vec<f32>> {
    decode_f32(encoded, bits, false)
}

fn decode_f32(encoded: &[u8], bits: u32, alpha: bool) -> Vec<Vec<f32>> {
    use zensim_decoder::api::{
        JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
    };
    let mut input = encoded;
    let init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let ProcessingResult::Complete {
        result: mut decoder,
    } = init.process(&mut input).unwrap()
    else {
        panic!("incomplete image header")
    };
    assert_eq!(decoder.basic_info().bit_depth.bits_per_sample(), bits);
    let (w, h) = decoder.basic_info().size;
    decoder.set_pixel_format(if alpha {
        JxlPixelFormat::rgba_f32(1)
    } else {
        JxlPixelFormat::rgb_f32(0)
    });
    let channels = if alpha { 4 } else { 3 };
    let mut frames = Vec::new();
    loop {
        let ProcessingResult::Complete { result: frame } = decoder.process(&mut input).unwrap()
        else {
            panic!("incomplete frame header")
        };
        let mut pixels = vec![0_u8; w * h * channels * 4];
        let mut buffers = [JxlOutputBuffer::new(&mut pixels, h, w * channels * 4)];
        let ProcessingResult::Complete { result: next } =
            frame.process(&mut input, &mut buffers).unwrap()
        else {
            panic!("incomplete frame")
        };
        decoder = next;
        frames.push(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect(),
        );
        if !decoder.has_more_frames() {
            break;
        }
    }
    frames
}

#[test]
fn lossless_animation_preserves_10_12_16_bit_samples_and_color_signaling() {
    use jxl_encoder::ColorEncoding;
    for (w, h) in [(17_u32, 13_u32), (511, 259)] {
        for bits in [10, 12, 16] {
            let max = (1_u32 << bits) - 1;
            let originals: Vec<Vec<u16>> = (0..2)
                .map(|f| {
                    (0..w * h * 3)
                        .map(|i| ((i * 73 + i / 17 * 113 + f * 29) & max) as u16)
                        .collect()
                })
                .collect();
            let bytes: Vec<Vec<u8>> = originals
                .iter()
                .map(|v| v.iter().flat_map(|s| s.to_ne_bytes()).collect())
                .collect();
            let frames = [
                AnimationFrame::new(&bytes[0], 1001),
                AnimationFrame::new(&bytes[1], 2002),
                AnimationFrame::new(&bytes[1], 1001),
            ];
            for (name, color) in [
                ("srgb", ColorEncoding::srgb()),
                ("linear", ColorEncoding::linear_srgb()),
                ("pq", ColorEncoding::bt2100_pq()),
                ("hlg", ColorEncoding::bt2100_hlg()),
            ] {
                let animation = AnimationParams {
                    tps_numerator: 30000,
                    tps_denominator: 1,
                    num_loops: 2,
                    ..Default::default()
                };
                let encoded = LosslessConfig::new()
                    .with_effort(1)
                    .encode_request(w, h, PixelLayout::Rgb16)
                    .with_bits_per_sample(bits)
                    .with_color_encoding(color.clone())
                    .encode_animation(&animation, &frames)
                    .unwrap();
                let decoded = decode_rgb_f32(&encoded, bits);
                assert_eq!(decoded.len(), 3);
                for (index, pixels) in decoded.iter().enumerate() {
                    for (&actual, &sample) in pixels.iter().zip(&originals[index.min(1)]) {
                        assert!(
                            (actual - f32::from(sample) / max as f32).abs() < 0.5 / max as f32,
                            "{w}x{h} {bits} {name} frame{index}: {actual} vs {sample}/{max}"
                        );
                    }
                }
                let image = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&encoded))
                    .unwrap();
                assert_eq!(image.num_loaded_keyframes(), 3);
                for i in 0..3 {
                    image.render_frame(i).unwrap();
                }
                if let Some(dir) = std::env::var_os("JXL_ANIMATION_ARTIFACTS") {
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir).unwrap();
                    std::fs::write(dir.join(format!("{w}x{h}-{bits}-{name}.jxl")), encoded)
                        .unwrap();
                }
            }
        }
    }
}

#[test]
fn animation_request_combines_metadata_limits_and_internal_cancellation() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct PollLimit(AtomicUsize);
    impl enough::Stop for PollLimit {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.0.fetch_add(1, Ordering::Relaxed) >= 8 {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }
    let data: Vec<_> = (0..511 * 259 * 3)
        .map(|i| ((i * 37 + i / 19 * 113) % 256) as u8)
        .collect();
    let frames = [
        AnimationFrame::new(&data, 10),
        AnimationFrame::new(&data, 20),
    ];
    let metadata =
        ImageMetadata::default().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let lossless = LosslessConfig::new().with_effort(1);
    let lossy = LossyConfig::new(1.0).with_effort(1);
    for cfg_lossy in [false, true] {
        let stop = PollLimit(AtomicUsize::new(0));
        let request = if cfg_lossy {
            lossy.encode_request(511, 259, PixelLayout::Rgb8)
        } else {
            lossless.encode_request(511, 259, PixelLayout::Rgb8)
        };
        let error = request
            .with_metadata(&metadata)
            .with_stop(&stop)
            .encode_animation(&AnimationParams::default(), &frames)
            .unwrap_err();
        assert!(
            matches!(error.error(), jxl_encoder::EncodeError::Cancelled),
            "{error:?}"
        );
        assert!(stop.0.load(Ordering::Relaxed) > 8);
        let limit = jxl_encoder::Limits::new().with_max_memory_bytes(1);
        let request = if cfg_lossy {
            lossy.encode_request(511, 259, PixelLayout::Rgb8)
        } else {
            lossless.encode_request(511, 259, PixelLayout::Rgb8)
        };
        assert!(matches!(
            request
                .with_metadata(&metadata)
                .with_limits(&limit)
                .encode_animation(&AnimationParams::default(), &frames)
                .unwrap_err()
                .error(),
            jxl_encoder::EncodeError::LimitExceeded { .. }
        ));
    }
}

#[test]
fn invalid_animation_clock_and_sample_precision_reject() {
    let cfg = LosslessConfig::new();
    let pixels = [0_u8; 12];
    let frames = [AnimationFrame::new(&pixels, 10)];
    for (num, den) in [(0, 1), (1, 0), (0, 0)] {
        assert!(
            cfg.encode_animation(
                2,
                2,
                PixelLayout::Rgb8,
                &AnimationParams {
                    tps_numerator: num,
                    tps_denominator: den,
                    ..Default::default()
                },
                &frames
            )
            .is_err()
        );
    }
    let pixels = [65535_u16; 12]
        .into_iter()
        .flat_map(u16::to_ne_bytes)
        .collect::<Vec<_>>();
    let frames = [AnimationFrame::new(&pixels, 10)];
    assert!(
        cfg.encode_request(2, 2, PixelLayout::Rgb16)
            .with_bits_per_sample(10)
            .encode_animation(&AnimationParams::default(), &frames)
            .is_err()
    );
}

#[test]
fn lossy_integer_animation_applies_declared_input_transfer_before_xyb() {
    use jxl_encoder::ColorEncoding;
    let values = [65_u8, 129, 193];
    let pixels = values.repeat(32 * 17);
    let frame = [AnimationFrame::new(&pixels, 10)];
    let cfg = LossyConfig::new(0.01).with_effort(1).with_gaborish(false);
    for (name, color) in [
        ("linear", ColorEncoding::linear_srgb()),
        ("gamma", ColorEncoding::with_gamma(1.0 / 2.2)),
        ("pq", ColorEncoding::bt2100_pq()),
        ("hlg", ColorEncoding::bt2100_hlg()),
    ] {
        let encoded = cfg
            .encode_request(32, 17, PixelLayout::Rgb8)
            .with_color_encoding(color)
            .encode_animation(&AnimationParams::default(), &frame)
            .unwrap();
        let decoded = decode_rgb_f32(&encoded, 8);
        for rgb in decoded[0].as_chunks::<3>().0.iter() {
            for (&actual, expected) in rgb.iter().zip(values) {
                assert!(
                    (actual - f32::from(expected) / 255.0).abs() < 0.005,
                    "{name}: {actual} vs {expected}/255"
                );
            }
        }
    }
}

#[test]
fn lossy_animation_preserves_full_precision_alpha() {
    for bits in [10, 12, 16] {
        let max = (1_u32 << bits) - 1;
        let samples: Vec<u16> = (0..17 * 13)
            .flat_map(|i| [max / 4, max / 2, max * 3 / 4, (i * 73) % max].map(|v| v as u16))
            .collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let frames = [
            AnimationFrame::new(&bytes, 10),
            AnimationFrame::new(&bytes, 20),
        ];
        let encoded = LossyConfig::new(0.1)
            .with_effort(1)
            .encode_request(17, 13, PixelLayout::Rgba16)
            .with_bits_per_sample(bits)
            .encode_animation(&AnimationParams::default(), &frames)
            .unwrap();
        for output in decode_f32(&encoded, bits, true) {
            for (actual, source) in output
                .as_chunks::<4>()
                .0
                .iter()
                .zip(samples.as_chunks::<4>().0.iter())
            {
                assert!(
                    (actual[3] - f32::from(source[3]) / max as f32).abs() < 0.5 / max as f32,
                    "{bits}-bit alpha: {} vs {}/{}",
                    actual[3],
                    source[3],
                    max
                );
            }
        }
    }
}
