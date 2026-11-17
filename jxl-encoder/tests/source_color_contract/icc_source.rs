//! Consumed source parameters remain meaningful when output carries an ICC.
use super::*;
use jxl_encoder::{AnimationFrame, AnimationParams, CIExy, CustomPrimaries, ImageMetadata};

#[test]
fn attached_icc_does_not_admit_invalid_consumed_source_parameters() {
    let root = directory();
    let metadata =
        ImageMetadata::new().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let mut bad_white = ColorEncoding::with_custom_white_point(CIExy::new(0.3, 0.0));
    bad_white.want_icc = true;
    let mut singular = ColorEncoding::with_custom_primaries(CustomPrimaries {
        red: CIExy::new(0.3, 0.3),
        green: CIExy::new(0.3, 0.3),
        blue: CIExy::new(0.3, 0.3),
    });
    singular.want_icc = true;
    let mut missing_white = ColorEncoding::srgb();
    missing_white.white_point = jxl_encoder::WhitePoint::Custom;
    let mut missing_primary = ColorEncoding::srgb();
    missing_primary.primaries = jxl_encoder::Primaries::Custom;
    let mut missing_primary_dci = missing_primary.clone();
    missing_primary_dci.white_point = jxl_encoder::WhitePoint::Dci;
    let mut failures = Vec::new();
    let mut journal = std::fs::File::create(root.join("icc-source.tsv")).unwrap();
    for (label, mut encoding) in [
        ("gamma-nan", ColorEncoding::with_gamma(f32::NAN)),
        ("gamma-zero", ColorEncoding::with_gamma(0.0)),
        ("white-zero-y", bad_white),
        ("singular-primary", singular),
        ("missing-white", missing_white),
        ("missing-primary", missing_primary),
        ("missing-primary-dci", missing_primary_dci),
    ] {
        encoding.want_icc = true;
        let pixels = [64_u8, 96, 128].repeat(16 * 17);
        for (policy, strategy, mode) in encoder_policies::policies() {
            let config = LossyConfig::new(1.0)
                .with_effort(1)
                .with_threads(1)
                .with_strategy(strategy)
                .with_mode(mode);
            for entry in ["still", "writer", "animation", "streaming"] {
                let mut dest = b"existing-output".to_vec();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let request = config
                        .encode_request(16, 17, PixelLayout::Rgb8)
                        .with_color_encoding(encoding.clone())
                        .with_metadata(&metadata);
                    match entry {
                        "still" => request.encode(&pixels).map(|_| ()),
                        "writer" => request.encode_to(&pixels, &mut dest).map(|_| ()),
                        "animation" => request
                            .encode_animation(
                                &AnimationParams::default(),
                                &[AnimationFrame::new(&pixels, 10)],
                            )
                            .map(|_| ()),
                        _ => {
                            let mut stream = config
                                .encoder(16, 17, PixelLayout::Rgb8)
                                .unwrap()
                                .with_color_encoding(encoding.clone())
                                .with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
                            let result = stream.push_rows(&pixels[..16 * 3], 1);
                            assert_eq!(stream.rows_pushed(), 0);
                            result
                        }
                    }
                }));
                writeln!(journal, "{label}\t{policy}\t{entry}\t{result:?}").unwrap();
                journal.flush().unwrap();
                if !matches!(result, Ok(Err(ref e)) if matches!(e.error(),
                    jxl_encoder::EncodeError::InvalidInput{message}
                    | jxl_encoder::EncodeError::InvalidConfig{message}
                    if message.contains("color encoding")))
                {
                    failures.push(format!("{label}-{policy}-{entry}: {result:?}"));
                }
                if dest != b"existing-output" {
                    std::fs::write(root.join(format!("{label}-{policy}-{entry}.jxl")), &dest)
                        .unwrap();
                    failures.push(format!("{label}-{policy}-{entry}: replaced output"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}

#[test]
fn attached_icc_preserves_omitted_and_overridden_source_fields() {
    use jxl_encoder::LosslessConfig;
    let root = directory();
    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let metadata = ImageMetadata::new().with_icc_profile(profile);
    for (policy, strategy, mode) in encoder_policies::policies() {
        for case in ["linear-float", "source-override", "lossless"] {
            let float = case == "linear-float";
            let layout = if float {
                PixelLayout::RgbLinearF32
            } else {
                PixelLayout::Rgb8
            };
            let pixels = if float {
                [0.5_f32, 0.2, 0.1]
                    .repeat(16 * 17)
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect::<Vec<_>>()
            } else {
                [64_u8, 96, 128].repeat(16 * 17)
            };
            let mut outputs = Vec::new();
            for ignored in [false, true] {
                let mut color = ColorEncoding::srgb();
                color.want_icc = true;
                if ignored {
                    color.gamma = Some(f32::NAN);
                }
                let bytes = if case == "lossless" {
                    let config = LosslessConfig::new()
                        .with_effort(1)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    let bytes = config
                        .encode_request(16, 17, layout)
                        .with_color_encoding(color.clone())
                        .with_metadata(&metadata)
                        .encode(&pixels)
                        .unwrap();
                    let mut stream = config
                        .encoder(16, 17, layout)
                        .unwrap()
                        .with_color_encoding(color)
                        .with_icc_profile(profile);
                    stream.push_rows(&pixels, 17).unwrap();
                    assert_eq!(bytes, stream.finish().unwrap());
                    bytes
                } else {
                    let config = LossyConfig::new(1.0)
                        .with_effort(1)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    let mut request = config
                        .encode_request(16, 17, layout)
                        .with_color_encoding(color.clone())
                        .with_metadata(&metadata);
                    if !float {
                        request = request.with_source_gamma(0.5);
                    }
                    let bytes = request.encode(&pixels).unwrap();
                    let mut stream = config
                        .encoder(16, 17, layout)
                        .unwrap()
                        .with_color_encoding(color)
                        .with_icc_profile(profile);
                    if !float {
                        stream = stream.with_source_gamma(0.5);
                    }
                    stream.push_rows(&pixels, 17).unwrap();
                    assert_eq!(bytes, stream.finish().unwrap());
                    bytes
                };
                let out = root.join(format!("{policy}-{case}-{ignored}"));
                std::fs::create_dir(&out).unwrap();
                for frames in render_case(&out, &bytes, 16, 17, false) {
                    assert_eq!(frames.len(), 1);
                    assert_eq!(frames[0].len(), 16 * 17 * 3);
                    assert!(frames[0].iter().all(|v| v.is_finite()));
                }
                let decoded = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(out.join("output.jxl"))
                    .arg(out.join("profile.pfm"))
                    .arg(format!(
                        "--orig_icc_out={}",
                        out.join("original.icc").display()
                    ))
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(out.join("profile.log"), &decoded.stderr).unwrap();
                assert!(decoded.status.success());
                assert_eq!(std::fs::read(out.join("original.icc")).unwrap(), profile);
                outputs.push(bytes);
            }
            assert_eq!(
                outputs[0], outputs[1],
                "{policy}-{case}: unused gamma changed bytes"
            );
        }
    }
}
