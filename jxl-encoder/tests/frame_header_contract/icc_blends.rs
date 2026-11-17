//! Original-profile ICC blending is independent of display-color conversion.
use super::*;

#[test]
fn original_icc_addition_preserves_profile_and_domain() {
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("caller-selected pinned original-ICC decoder required");
    let version = Command::new(&oracle).arg("--version").output().unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 12000"
    );
    let root = run_dir("original-icc-blends");
    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let metadata = jxl_encoder::ImageMetadata::new().with_icc_profile(profile);
    for (width, height) in [(16, 17), (259, 133)] {
        for alpha in [false, true] {
            let channels = if alpha { 4 } else { 3 };
            let pixel: &[u8] = if alpha {
                &[16, 32, 48, 128]
            } else {
                &[16, 32, 48]
            };
            let source = pixel.repeat((width * height) as usize);
            let mut doubled = source.clone();
            for pixel in doubled.chunks_exact_mut(channels) {
                for value in &mut pixel[..3] {
                    *value *= 2;
                }
            }
            let layout = if alpha {
                PixelLayout::Rgba8
            } else {
                PixelLayout::Rgb8
            };
            let still = LosslessConfig::new()
                .with_effort(1)
                .encode_request(width, height, layout)
                .with_metadata(&metadata)
                .encode(&doubled)
                .unwrap();
            let expected_primary =
                decoded_extras::decode_current_frames(&still, "doubled ICC still")
                    .pop()
                    .unwrap()
                    .0;
            for (policy, strategy, mode) in policies::policies() {
                for lossy in [false, true] {
                    let out = root.join(format!("{width}x{height}-a{alpha}-{policy}-lossy{lossy}"));
                    std::fs::create_dir(&out).unwrap();
                    let frames = [
                        AnimationFrame::new(&source, 10),
                        AnimationFrame::new(&source, 0)
                            .with_reference_only(true)
                            .with_save_as_reference(2),
                        AnimationFrame::new(&source, 20)
                            .with_blend_mode(BlendMode::Add)
                            .with_blend_source(2),
                    ];
                    let bytes = if lossy {
                        LossyConfig::new(0.1)
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_patches(false)
                            .encode_request(width, height, layout)
                            .with_metadata(&metadata)
                            .encode_animation(&AnimationParams::default(), &frames)
                    } else {
                        LosslessConfig::new()
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .encode_animation_with_metadata(
                                width,
                                height,
                                layout,
                                &AnimationParams::default(),
                                &frames,
                                &metadata,
                            )
                    }
                    .unwrap();
                    let path = out.join("output.jxl");
                    std::fs::write(&path, &bytes).unwrap();
                    use jxl::api::{
                        JxlColorProfile, JxlDecoder, JxlDecoderOptions, ProcessingResult, states,
                    };
                    let mut input = bytes.as_slice();
                    let ProcessingResult::Complete { result: header } =
                        JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default())
                            .process(&mut input)
                            .unwrap()
                    else {
                        panic!("incomplete ICC header");
                    };
                    let JxlColorProfile::Icc(embedded) = header.embedded_color_profile() else {
                        panic!("expected original ICC");
                    };
                    assert_eq!(embedded, profile);
                    let primary =
                        decoded_extras::decode_current_frames(&bytes, &out.display().to_string());
                    assert_eq!(primary.len(), 2);
                    if !lossy {
                        assert_eq!(
                            primary[1].0, expected_primary,
                            "lossless ICC composition vs doubled source"
                        );
                    }
                    let prefix = out.join("original");
                    let result = Command::new(&oracle)
                        .arg(&path)
                        .arg(&prefix)
                        .output()
                        .unwrap();
                    std::fs::write(out.join("native.log"), &result.stderr).unwrap();
                    assert!(
                        result.status.success(),
                        "{} original ICC render: {}",
                        out.display(),
                        String::from_utf8_lossy(&result.stderr)
                    );
                    assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), "2 frames");
                    let first = image::open(out.join("original-0.pam")).unwrap().to_rgba16();
                    let last = image::open(out.join("original-1.pam")).unwrap().to_rgba16();
                    assert_eq!(first.dimensions(), (width, height));
                    assert_eq!(last.dimensions(), (width, height));
                    for (index, (a, b)) in first.pixels().zip(last.pixels()).enumerate() {
                        for channel in 0..3 {
                            if !lossy {
                                assert_eq!(a[channel], u16::from(pixel[channel]) * 257);
                                assert_eq!(b[channel], u16::from(pixel[channel]) * 2 * 257);
                            } else {
                                assert!(
                                    (i32::from(b[channel]) - 2 * i32::from(a[channel])).abs()
                                        <= 264,
                                    "{} pixel{index} channel{channel} Add in original ICC space",
                                    out.display()
                                );
                            }
                        }
                        assert_eq!(a[3], if alpha { 128 * 257 } else { 65535 });
                        assert_eq!(b[3], a[3]);
                    }
                }
            }
        }
    }
}
