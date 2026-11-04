//! Tone mapping validates the values that the decoder will actually read.
use jxl_encoder::{
    AnimationFrame, AnimationParams, ColorEncoding, ImageMetadata, LosslessConfig, LossyConfig,
    PixelLayout,
};

#[test]
fn minimum_luminance_cannot_exceed_an_implicit_peak() {
    let pixels = [64, 96, 128].repeat(16 * 9);
    let mut accepted = Vec::new();
    for lossy in [false, true] {
        for (name, color, lossy_peak) in [
            ("srgb", ColorEncoding::srgb(), 255.0),
            ("pq", ColorEncoding::bt2100_pq(), 10000.0),
            ("hlg", ColorEncoding::bt2100_hlg(), 1000.0),
        ] {
            let peak = if lossy { lossy_peak } else { 255.0 };
            let lossy_config = LossyConfig::new(0.1).with_effort(1);
            let lossless_config = LosslessConfig::new().with_effort(1);
            for from_metadata in [false, true] {
                for animation in [false, true] {
                    let meta = ImageMetadata::new().with_min_nits(peak + 1.0);
                    let request = if lossy {
                        lossy_config.encode_request(16, 9, PixelLayout::Rgb8)
                    } else {
                        lossless_config.encode_request(16, 9, PixelLayout::Rgb8)
                    };
                    let request = request.with_color_encoding(color.clone());
                    let request = if from_metadata {
                        request.with_metadata(&meta)
                    } else {
                        request.with_min_nits(peak + 1.0)
                    };
                    let result = if animation {
                        request.encode_animation(
                            &AnimationParams::default(),
                            &[
                                AnimationFrame::new(&pixels, 10),
                                AnimationFrame::new(&pixels, 20),
                            ],
                        )
                    } else {
                        request.encode(&pixels)
                    };
                    match result {
                        Ok(bytes) => {
                            let parsed =
                                jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(bytes));
                            accepted.push(format!("lossy={lossy}, {name}, metadata={from_metadata}, animation={animation}; decoder result: {:?}", parsed.err()));
                        }
                        Err(error) => assert!(error.to_string().contains("min_nits"), "{error}"),
                    }
                }
            }
            if !lossy {
                let mut stream = lossless_config
                    .encoder(16, 9, PixelLayout::Rgb8)
                    .unwrap()
                    .with_color_encoding(color.clone())
                    .with_min_nits(peak + 1.0);
                let result = stream.push_rows(&pixels, 9).and_then(|()| stream.finish());
                match result {
                    Ok(_) => accepted.push(format!("lossless stream, {name}")),
                    Err(error) => assert!(error.to_string().contains("min_nits"), "{error}"),
                }
            }
            let mut stream = lossy_config
                .encoder(16, 9, PixelLayout::Rgb8)
                .unwrap()
                .with_color_encoding(color)
                .with_min_nits(lossy_peak + 1.0);
            let result = stream.push_rows(&pixels, 9).and_then(|()| stream.finish());
            match result {
                Ok(_) => accepted.push(format!("lossy stream, {name}")),
                Err(error) => assert!(error.to_string().contains("min_nits"), "{error}"),
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted invalid tone mapping:\n{}",
        accepted.join("\n")
    );
}

#[test]
fn positive_peak_must_remain_positive_in_binary16() {
    let pixels = [64, 96, 128].repeat(16 * 9);
    let result = LosslessConfig::new()
        .with_effort(1)
        .encode_request(16, 9, PixelLayout::Rgb8)
        .with_intensity_target(1e-12)
        .encode(&pixels);
    assert!(result.is_err(), "positive peak was serialized as zero");
}

#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

#[test]
fn valid_defaults_and_effective_overrides_still_decode() {
    let pixels = [64, 96, 128].repeat(16 * 9);
    let root = jxl_encoder::test_helpers::output_dir("tone-mapping-validation");
    let directory = (0..)
        .find_map(|i| {
            let directory = root.join(format!("run-{i}"));
            match std::fs::create_dir(&directory) {
                Ok(()) => Some(directory),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", directory.display()),
            }
        })
        .unwrap();
    let mut cells = 0;
    for lossy in [false, true] {
        for (color, lossy_peak) in [
            (ColorEncoding::srgb(), 255.0),
            (ColorEncoding::bt2100_pq(), 10000.0),
            (ColorEncoding::bt2100_hlg(), 1000.0),
        ] {
            let peak = if lossy { lossy_peak } else { 255.0 };
            let lossy_config = LossyConfig::new(0.1).with_effort(1);
            let lossless_config = LosslessConfig::new().with_effort(1);
            for variant in 0..3 {
                for animation in [false, true] {
                    let metadata = match variant {
                        0 => ImageMetadata::new().with_min_nits(peak / 2.0),
                        1 => ImageMetadata::new().with_min_nits(peak + 1.0),
                        _ => ImageMetadata::new()
                            .with_intensity_target(peak / 4.0)
                            .with_min_nits(peak / 2.0),
                    };
                    let request = if lossy {
                        lossy_config.encode_request(16, 9, PixelLayout::Rgb8)
                    } else {
                        lossless_config.encode_request(16, 9, PixelLayout::Rgb8)
                    };
                    let request = request
                        .with_color_encoding(color.clone())
                        .with_metadata(&metadata);
                    let request = match variant {
                        0 => request,
                        1 => request.with_min_nits(peak / 2.0),
                        _ => request.with_intensity_target(peak),
                    };
                    let encoded = if animation {
                        request.encode_animation(
                            &AnimationParams::default(),
                            &[
                                AnimationFrame::new(&pixels, 10),
                                AnimationFrame::new(&pixels, 20),
                            ],
                        )
                    } else {
                        request.encode(&pixels)
                    }
                    .unwrap();
                    let image = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&encoded))
                        .unwrap();
                    let tone = &image.image_header().metadata.tone_mapping;
                    assert_eq!(tone.intensity_target, peak);
                    assert_eq!(tone.min_nits, peak / 2.0);
                    decode::verify_jxl_rs(&encoded, 16, 9);
                    let input = directory.join(format!("{cells}.jxl"));
                    let output = directory.join(format!("{cells}.png"));
                    std::fs::write(&input, encoded).unwrap();
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(input)
                        .arg(&output)
                        .args(["--num_threads=1", "--bits_per_sample=16"])
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    assert_eq!(
                        image::open(output).unwrap().to_rgb16().dimensions(),
                        (16, 9)
                    );
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 36);
    // The smallest positive binary16 value is valid and must not be
    // rejected by a normal-float minimum or an SDR/HDR heuristic.
    let minimum_peak = 1.0 / 16_777_216.0;
    let encoded = LosslessConfig::new()
        .with_effort(1)
        .encode_request(16, 9, PixelLayout::Rgb8)
        .with_intensity_target(minimum_peak)
        .encode(&pixels)
        .unwrap();
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&encoded))
        .unwrap();
    assert_eq!(
        image.image_header().metadata.tone_mapping.intensity_target,
        minimum_peak
    );
    decode::verify_jxl_rs(&encoded, 16, 9);
}

#[test]
fn implied_hdr_float_defaults_participate_in_validation() {
    let pixels: Vec<u8> = [0.4_f32, 0.5, 0.6]
        .repeat(16 * 9)
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let root = jxl_encoder::test_helpers::output_dir("tone-mapping-implied-hdr");
    let directory = (0..)
        .find_map(|i| {
            let directory = root.join(format!("run-{i}"));
            match std::fs::create_dir(&directory) {
                Ok(()) => Some(directory),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", directory.display()),
            }
        })
        .unwrap();
    for (layout, peak) in [
        (PixelLayout::RgbPqF32, 10000.0),
        (PixelLayout::RgbHlgF32, 1000.0),
    ] {
        let config = LossyConfig::new(0.1).with_effort(1);
        let rejected = config
            .encode_request(16, 9, layout)
            .with_min_nits(peak + 1.0)
            .encode(&pixels);
        assert!(rejected.is_err());
        let encoded = config
            .encode_request(16, 9, layout)
            .with_min_nits(peak / 2.0)
            .encode(&pixels)
            .unwrap();
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .unwrap();
        assert_eq!(
            image.image_header().metadata.tone_mapping.intensity_target,
            peak
        );
        decode::verify_jxl_rs(&encoded, 16, 9);
        let input = directory.join(format!("{layout:?}.jxl"));
        let output = directory.join(format!("{layout:?}.png"));
        std::fs::write(&input, encoded).unwrap();
        let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(input)
            .arg(&output)
            .args(["--num_threads=1", "--bits_per_sample=16"])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            image::open(output).unwrap().to_rgb16().dimensions(),
            (16, 9)
        );
    }
}
