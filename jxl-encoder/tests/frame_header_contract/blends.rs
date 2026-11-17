//! Explicit buffers and reference storage domains, issue 144 H11.
use super::*;
use jxl_encoder::BlendMode;

#[path = "../support/decoded_extras.rs"]
mod decoded_extras;
#[path = "../support/pfm.rs"]
mod pfm;
use super::encoder_policies as policies;
#[cfg(feature = "corpus-tests")]
#[path = "../support/source_colors.rs"]
mod source_colors;

fn native_frames(path: &Path, width: usize, height: usize) -> Vec<Vec<f32>> {
    let output = Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg(path.with_extension("pfm"))
        .args([
            "--output_frames",
            "--num_threads=1",
            "--color_space=RGB_D65_SRG_Rel_SRG",
        ])
        .output()
        .unwrap();
    std::fs::write(path.with_extension("native.log"), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "{}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    (0..2)
        .map(|i| {
            pfm::read_pfm_samples(
                &path.with_file_name(format!(
                    "{}.pfm-{i}.pfm",
                    path.file_stem().unwrap().to_str().unwrap()
                )),
                (width, height),
            )
        })
        .collect()
}

fn verify_add(reference_only: bool, reference_slot: Option<u32>) {
    let dir = run_dir(if reference_only {
        "reference-color-add"
    } else {
        "explicit-full-add"
    });
    let (width, height) = (16_u32, 16_u32);
    let first = if reference_only { [8, 16, 24] } else { [64; 3] }.repeat(256);
    let hidden = [32, 48, 64].repeat(256);
    let addend = if reference_only {
        [64, 48, 32]
    } else {
        [64; 3]
    }
    .repeat(256);
    let expected = if reference_only { 96_u16 } else { 128 };
    let baseline = LosslessConfig::new()
        .with_effort(1)
        .encode(&first, width, height, PixelLayout::Rgb8)
        .unwrap();
    let (base_color, base_extra) =
        decoded_extras::decode_current_planes(&baseline, "explicit-blend still control");
    assert!(base_extra.is_empty());
    assert_eq!(base_color.len(), first.len());
    for (&actual, &source) in base_color.iter().zip(&first) {
        assert!((actual - f32::from(source) / 255.0).abs() < 0.000001);
    }
    let mut frames = vec![AnimationFrame::new(&first, 10)];
    if reference_only {
        let frame = AnimationFrame::new(&hidden, 0).with_reference_only(true);
        frames.push(match reference_slot {
            Some(slot) => frame.with_save_as_reference(slot),
            None => frame,
        });
    }
    frames.push(
        AnimationFrame::new(&addend, 20)
            .with_blend_mode(BlendMode::Add)
            .with_blend_source(if reference_only {
                reference_slot.unwrap_or(1)
            } else {
                1
            }),
    );
    let mut failures = Vec::new();
    for lossy in [false, true] {
        let bytes = if lossy {
            LossyConfig::new(0.01)
                .with_effort(3)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_gaborish(false)
                .with_epf_level(0)
                .with_patches(false)
                .encode_animation(
                    width,
                    height,
                    PixelLayout::Rgb8,
                    &AnimationParams::default(),
                    &frames,
                )
        } else {
            LosslessConfig::new()
                .with_effort(3)
                .with_strategy(EncoderStrategy::Libjxl)
                .encode_animation(
                    width,
                    height,
                    PixelLayout::Rgb8,
                    &AnimationParams::default(),
                    &frames,
                )
        }
        .unwrap();
        let path = dir.join(format!("lossy-{lossy}.jxl"));
        std::fs::write(&path, &bytes).unwrap();
        for (decoder, result) in [
            (
                "jxl-rs",
                std::panic::catch_unwind(|| {
                    let decoded = rust_frames(&bytes, 3);
                    assert_eq!(decoded.len(), 2);
                    assert_eq!(decoded[1].len(), 16 * 16 * 3);
                    for (i, &v) in decoded[1].iter().enumerate() {
                        assert!(v.is_finite());
                        let error = (v - f32::from(expected) / 255.0).abs();
                        assert!(
                            error < if lossy { 0.002 } else { 0.000001 },
                            "sample{i}: {v}, expected{expected}/255"
                        );
                    }
                }),
            ),
            (
                "djxl",
                std::panic::catch_unwind(|| {
                    let decoded = native_frames(&path, 16, 16);
                    assert_eq!(decoded.len(), 2);
                    assert_eq!(decoded[1].len(), 16 * 16 * 3);
                    for (i, &v) in decoded[1].iter().enumerate() {
                        assert!(v.is_finite());
                        assert!(
                            (v - f32::from(expected) / 255.0).abs()
                                < if lossy { 0.002 } else { 0.000001 },
                            "sample{i}: {v}, expected{expected}/255"
                        );
                    }
                }),
            ),
        ] {
            if result.is_err() {
                failures.push(format!("lossy{lossy}/{decoder}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{failures:?}; retained {}",
        dir.display()
    );
}

#[test]
fn explicit_add_uses_the_full_canvas() {
    verify_add(false, None);
}
#[test]
fn reference_only_color_add_uses_post_transform_samples() {
    verify_add(true, Some(2));
}

#[test]
fn explicit_blend_policy_mode_and_entry_matrix() {
    use jxl_encoder::ProgressiveMode;
    let root = run_dir("blend-headers");
    let mut header_rows = Vec::new();
    for (policy, strategy, mode) in policies::policies() {
        for (shape, (width, height)) in [(1_u32, 1_u32), (8, 8), (16, 16), (259, 133)]
            .into_iter()
            .enumerate()
        {
            for (blend_index, blend) in [
                BlendMode::Replace,
                BlendMode::Add,
                BlendMode::Blend,
                BlendMode::AlphaWeightedAdd,
                BlendMode::Mul,
            ]
            .into_iter()
            .enumerate()
            {
                // Identical buffers force the original erroneous crop; opaque
                // alpha makes all five color formulas independently checkable.
                let layout = if (shape + blend_index) % 2 == 0 {
                    PixelLayout::Rgb8
                } else {
                    PixelLayout::Rgba8
                };
                let mut pixel = vec![64_u8; 3];
                if layout.has_alpha() {
                    pixel.push(255);
                }
                let pixels = pixel.repeat((width * height) as usize);
                for reference in [false, true] {
                    let mut frames = vec![AnimationFrame::new(&pixels, 10)];
                    if reference {
                        frames.push(
                            AnimationFrame::new(&pixels, 0)
                                .with_reference_only(true)
                                .with_save_as_reference(2),
                        );
                    }
                    frames.push(
                        AnimationFrame::new(&pixels, 20)
                            .with_blend_mode(blend)
                            .with_blend_source(if reference { 2 } else { 1 }),
                    );
                    for lossy in [false, true] {
                        let dir = run_dir(&format!(
                            "blend-{policy}-{width}x{height}-{blend:?}-ref{reference}-lossy{lossy}"
                        ));
                        let path = dir.join("out.jxl");
                        let entry = (shape + blend_index) % 4;
                        let animation = AnimationParams {
                            premultiplied_alpha: shape % 2 == 1,
                            ..Default::default()
                        };
                        let bytes = if lossy {
                            let cfg = LossyConfig::new(0.01)
                                .with_effort(if shape == 0 { 1 } else { 3 })
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                                .with_progressive(
                                    [
                                        ProgressiveMode::Single,
                                        ProgressiveMode::QuantizedAcFullAc,
                                        ProgressiveMode::DcVlfLfAc,
                                    ][shape % 3],
                                )
                                .with_gaborish(false)
                                .with_epf_level(0)
                                .with_patches(false)
                                .with_auto_delta_frames(blend_index % 2 == 0);
                            match entry {
                                0 => cfg
                                    .encode_request(width, height, layout)
                                    .encode_animation(&animation, &frames),
                                1 => {
                                    cfg.encode_animation(width, height, layout, &animation, &frames)
                                }
                                2 => cfg.encode_animation_with_metadata(
                                    width,
                                    height,
                                    layout,
                                    &animation,
                                    &frames,
                                    &jxl_encoder::ImageMetadata::new().with_xmp(b"<xmpmeta/>"),
                                ),
                                _ => cfg.encode_animation_with_limits(
                                    width,
                                    height,
                                    layout,
                                    &animation,
                                    &frames,
                                    &jxl_encoder::Limits::default(),
                                ),
                            }
                        } else {
                            let cfg = LosslessConfig::new()
                                .with_effort(3)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                                .with_auto_delta_frames(blend_index % 2 == 0);
                            match entry {
                                0 => cfg
                                    .encode_request(width, height, layout)
                                    .encode_animation(&animation, &frames),
                                1 => {
                                    cfg.encode_animation(width, height, layout, &animation, &frames)
                                }
                                2 => cfg.encode_animation_with_metadata(
                                    width,
                                    height,
                                    layout,
                                    &animation,
                                    &frames,
                                    &jxl_encoder::ImageMetadata::new().with_xmp(b"<xmpmeta/>"),
                                ),
                                _ => cfg.encode_animation_with_limits(
                                    width,
                                    height,
                                    layout,
                                    &animation,
                                    &frames,
                                    &jxl_encoder::Limits::default(),
                                ),
                            }
                        }
                        .unwrap();
                        std::fs::write(&path, &bytes).unwrap();
                        let decoded = decoded_extras::decode_current_frames(
                            &bytes,
                            &dir.display().to_string(),
                        );
                        for (_, extras) in &decoded {
                            assert_eq!(extras.len(), usize::from(layout.has_alpha()));
                            for alpha in extras {
                                assert_eq!(alpha.len(), (width * height) as usize);
                                assert!(alpha.iter().all(|&a| a == 1.0));
                            }
                        }
                        let decoded: Vec<_> = decoded.into_iter().map(|(color, _)| color).collect();
                        let native = native_frames(&path, width as usize, height as usize);
                        let color = 64_f32 / 255.0;
                        let expected = match blend {
                            BlendMode::Add | BlendMode::AlphaWeightedAdd => color * 2.0,
                            BlendMode::Mul => color * color,
                            _ => color,
                        };
                        for (decoder, frames) in [("jxl-rs", decoded), ("djxl", native)] {
                            assert_eq!(frames.len(), 2);
                            let channels = 3;
                            assert_eq!(frames[1].len(), (width * height) as usize * channels);
                            for (i, &v) in frames[1].iter().enumerate() {
                                assert!(v.is_finite());
                                let target = expected;
                                assert!(
                                    (v - target).abs() < if lossy { 0.003 } else { 0.000001 },
                                    "{} {decoder} sample{i}:{v} target{target}",
                                    dir.display()
                                );
                            }
                        }
                        header_rows.push(serde_json::json!({"path":path,"reference":reference,"blend":format!("{blend:?}")}));
                    }
                }
            }
        }
    }
    let manifest = root.join("streams.json");
    std::fs::write(&manifest, serde_json::to_vec(&header_rows).unwrap()).unwrap();
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scripts/check_explicit_blend_headers.py");
    let status = Command::new("python3")
        .arg(script)
        .arg(&manifest)
        .arg(root.join("headers.json"))
        .status()
        .unwrap();
    assert!(
        status.success(),
        "independent complete frame sequence: {}",
        root.display()
    );
}

#[cfg(feature = "corpus-tests")]
#[test]
fn accepted_blend_layouts_keep_original_profile_addition() {
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("caller-selected v0.12 original-profile float oracle");
    let version = Command::new(&oracle).arg("--version").output().unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 12000"
    );
    let layouts = [
        PixelLayout::Rgb8,
        PixelLayout::Rgba8,
        PixelLayout::Bgr8,
        PixelLayout::Bgra8,
        PixelLayout::Gray8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgb16,
        PixelLayout::Rgba16,
        PixelLayout::Gray16,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbLinearF16,
        PixelLayout::RgbaLinearF16,
        PixelLayout::GrayLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbLinearF32,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayLinearF32,
        PixelLayout::GrayAlphaLinearF32,
        PixelLayout::RgbPqF32,
        PixelLayout::RgbaPqF32,
        PixelLayout::RgbHlgF32,
        PixelLayout::RgbaHlgF32,
        PixelLayout::RgbBt709F32,
        PixelLayout::RgbaBt709F32,
    ];
    let (width, height) = (16_u32, 17_u32);
    for layout in layouts {
        let channels =
            (if layout.is_grayscale() { 1 } else { 3 }) + usize::from(layout.has_alpha());
        let mut pixel = Vec::new();
        for lane in 0..channels {
            let alpha = layout.has_alpha() && lane == channels - 1;
            if layout.is_f16() {
                pixel.extend_from_slice(
                    &(if alpha {
                        0x3800_u16
                    } else {
                        [0x3400, 0x3000, 0x2c00][lane]
                    })
                    .to_ne_bytes(),
                );
            } else if layout.is_f32() {
                pixel.extend_from_slice(
                    &(if alpha {
                        0.5_f32
                    } else {
                        [0.25_f32, 0.125, 0.0625][lane]
                    })
                    .to_ne_bytes(),
                );
            } else if layout.is_16bit() {
                pixel.extend_from_slice(&(if alpha { 32768_u16 } else { 16448 }).to_ne_bytes());
            } else {
                pixel.push(if alpha { 127 } else { 64 });
            }
        }
        let check_alpha = |values: &[f32]| {
            assert_eq!(values.len(), (width * height) as usize);
            for &value in values {
                assert!(value.is_finite());
                if layout.is_f16() || layout.is_f32() {
                    assert_eq!(value.to_bits(), 0.5_f32.to_bits());
                } else if layout.is_16bit() {
                    assert_eq!((value * 65535.0).round() as u16, 32768);
                } else {
                    assert_eq!((value * 255.0).round() as u8, 127);
                }
            }
        };
        assert_eq!(pixel.len(), layout.bytes_per_pixel());
        let pixels = pixel.repeat((width * height) as usize);
        for (policy_index, (policy, strategy, mode)) in policies::policies().into_iter().enumerate()
        {
            for lossy in [false, true] {
                let dir = run_dir(&format!("typed-add-{layout:?}-{policy}-lossy{lossy}"));
                let frames = [
                    AnimationFrame::new(&pixels, 10),
                    AnimationFrame::new(&pixels, 0)
                        .with_reference_only(true)
                        .with_save_as_reference(2),
                    AnimationFrame::new(&pixels, 20)
                        .with_blend_mode(BlendMode::Add)
                        .with_blend_source(2),
                ];
                let animation = AnimationParams {
                    premultiplied_alpha: policy_index % 2 == 0,
                    ..Default::default()
                };
                let encoded =
                    if lossy {
                        let cfg = LossyConfig::new(0.01)
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_gaborish(false)
                            .with_epf_level(0)
                            .with_patches(false)
                            .with_auto_delta_frames(policy_index % 2 == 0);
                        if policy_index % 2 == 0 {
                            let request = cfg.encode_request(width, height, layout);
                            let request = match layout {
                                PixelLayout::RgbLinearF32 => {
                                    request.with_color_encoding(source_colors::encoding("p3-dci"))
                                }
                                PixelLayout::RgbaLinearF32 => request
                                    .with_color_encoding(source_colors::encoding("aces-ap0-dci")),
                                PixelLayout::RgbLinearF16 => request
                                    .with_color_encoding(source_colors::encoding("custom-p3-d50")),
                                PixelLayout::RgbaLinearF16 => {
                                    request.with_color_encoding(source_colors::encoding("srgb-e"))
                                }
                                _ => request,
                            };
                            request.encode_animation(&animation, &frames)
                        } else {
                            cfg.encode_animation(width, height, layout, &animation, &frames)
                        }
                    } else {
                        let cfg = LosslessConfig::new()
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_auto_delta_frames(policy_index % 2 == 0);
                        if policy_index % 2 == 0 {
                            let request = cfg.encode_request(width, height, layout);
                            let request = match layout {
                                PixelLayout::RgbLinearF32 => {
                                    request.with_color_encoding(source_colors::encoding("p3-dci"))
                                }
                                PixelLayout::RgbaLinearF32 => request
                                    .with_color_encoding(source_colors::encoding("aces-ap0-dci")),
                                PixelLayout::RgbLinearF16 => request
                                    .with_color_encoding(source_colors::encoding("custom-p3-d50")),
                                PixelLayout::RgbaLinearF16 => {
                                    request.with_color_encoding(source_colors::encoding("srgb-e"))
                                }
                                _ => request,
                            };
                            request.encode_animation(&animation, &frames)
                        } else {
                            cfg.encode_animation(width, height, layout, &animation, &frames)
                        }
                    };
                if !lossy && (layout.is_f16() || layout.is_f32()) {
                    assert!(
                        matches!(&encoded,Err(e) if matches!(e.error(),jxl_encoder::EncodeError::UnsupportedPixelLayout(actual) if *actual==layout)),
                        "explicit lossless animation layout refusal"
                    );
                    continue;
                }
                let bytes = encoded.unwrap();
                let input = dir.join("input.jxl");
                std::fs::write(&input, &bytes).unwrap();
                let primary =
                    decoded_extras::decode_current_frames(&bytes, &dir.display().to_string());
                assert_eq!(primary.len(), 2);
                for (_, extra) in &primary {
                    assert_eq!(extra.len(), usize::from(layout.has_alpha()));
                    for alpha in extra {
                        check_alpha(alpha);
                    }
                }
                let native = Command::new(&oracle)
                    .arg(&input)
                    .arg(dir.join("native"))
                    .arg("--source-fields-float")
                    .output()
                    .unwrap();
                std::fs::write(
                    dir.join("native.log"),
                    [native.stdout, native.stderr].concat(),
                )
                .unwrap();
                assert!(
                    native.status.success(),
                    "{} native float original fields",
                    dir.display()
                );
                if layout.has_alpha() {
                    for frame in 0..2 {
                        let alpha = pfm::read_pfm_samples(
                            &dir.join(format!("native-{frame}-ec0.pfm")),
                            (width as usize, height as usize),
                        );
                        check_alpha(&alpha);
                    }
                }
                let native: Vec<_> = (0..2)
                    .map(|frame| {
                        pfm::read_pfm_samples(
                            &dir.join(format!("native-{frame}.pfm")),
                            (width as usize, height as usize),
                        )
                    })
                    .collect();
                for (name, color) in [
                    (
                        "jxl-rs",
                        primary.into_iter().map(|(c, _)| c).collect::<Vec<_>>(),
                    ),
                    ("libjxl", native),
                ] {
                    let colors = if layout.is_grayscale() { 1 } else { 3 };
                    assert_eq!(color[0].len(), (width * height) as usize * colors);
                    assert_eq!(color[0].len(), color[1].len());
                    for (i, (&first, &last)) in color[0].iter().zip(&color[1]).enumerate() {
                        assert!(
                            first.is_finite() && last.is_finite() && first > 0.01,
                            "{} {name} source sample{i}:{first}",
                            dir.display()
                        );
                        if !lossy {
                            let (maximum, source) = if layout.is_16bit() {
                                (65535_f32, 16448_u32)
                            } else {
                                (255_f32, 64_u32)
                            };
                            assert_eq!(
                                (first * maximum).round() as u32,
                                source,
                                "{} {name} source sample{i}",
                                dir.display()
                            );
                            assert_eq!(
                                (last * maximum).round() as u32,
                                source * 2,
                                "{} {name} Add source code{i}",
                                dir.display()
                            );
                        }
                        assert!(
                            (last - 2.0 * first).abs() < if lossy { 0.004 } else { 0.000001 },
                            "{} {name} sample{i} first{first} last{last}",
                            dir.display()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn reference_only_default_and_boundary_slots_keep_color() {
    for slot in [None, Some(0), Some(3)] {
        verify_add(true, slot);
    }
}
#[cfg(feature = "corpus-tests")]
#[path = "patch_slots.rs"]
mod patch_slots;

#[cfg(feature = "corpus-tests")]
#[path = "icc_blends.rs"]
mod icc_blends;
