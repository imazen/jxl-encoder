use super::*;
use jxl_encoder::api::EncoderMode;
use jxl_encoder::{ImageMetadata, Limits};

use super::encoder_policies;
use encoder_policies::policies;

#[allow(clippy::too_many_arguments)]
fn encode(
    lossy: bool,
    strategy: EncoderStrategy,
    mode: EncoderMode,
    effort: u8,
    dims: (u32, u32),
    layout: PixelLayout,
    associated: bool,
    auto: bool,
    route: usize,
    frames: &[AnimationFrame<'_>],
) -> Vec<u8> {
    let (w, h) = dims;
    let animation = AnimationParams {
        premultiplied_alpha: associated,
        ..Default::default()
    };
    let metadata = ImageMetadata::default();
    let limits = Limits::default();
    if lossy {
        let cfg = LossyConfig::new(1.0)
            .with_effort(effort)
            .with_strategy(strategy)
            .with_mode(mode)
            .with_auto_delta_frames(auto);
        match route {
            0 => cfg.encode_animation(w, h, layout, &animation, frames),
            1 => cfg.encode_animation_with_metadata(w, h, layout, &animation, frames, &metadata),
            2 => cfg.encode_animation_with_limits(w, h, layout, &animation, frames, &limits),
            _ => cfg
                .encode_request(w, h, layout)
                .encode_animation(&animation, frames),
        }
        .unwrap()
    } else {
        let cfg = LosslessConfig::new()
            .with_effort(effort)
            .with_strategy(strategy)
            .with_mode(mode)
            .with_auto_delta_frames(auto);
        match route {
            0 => cfg.encode_animation(w, h, layout, &animation, frames),
            1 => cfg.encode_animation_with_metadata(w, h, layout, &animation, frames, &metadata),
            2 => cfg.encode_animation_with_limits(w, h, layout, &animation, frames, &limits),
            _ => cfg
                .encode_request(w, h, layout)
                .encode_animation(&animation, frames),
        }
        .unwrap()
    }
}

fn pixels(dims: (u32, u32), gray: bool, wide: bool, seed: u16) -> (Vec<u8>, Vec<u16>) {
    let mut bytes = Vec::new();
    let mut expected = Vec::new();
    for y in 0..dims.1 {
        for x in 0..dims.0 {
            let g = ((u32::from(seed) + x + y) % 70 + 30) as u16;
            let rgba = [
                g,
                if gray { g } else { g + 20 },
                if gray { g } else { g + 40 },
                193,
            ];
            let expanded = rgba.map(|v| if wide { v * 257 + 1 } else { v * 257 });
            expected.extend(expanded);
            let codes: Vec<_> = if gray {
                vec![expanded[0], expanded[3]]
            } else {
                expanded.to_vec()
            };
            for code in codes {
                if wide {
                    bytes.extend(code.to_ne_bytes());
                } else {
                    bytes.push((code / 257) as u8);
                }
            }
        }
    }
    (bytes, expected)
}

#[test]
fn reference_only_layouts_modes_and_entry_paths() {
    let dir = run_dir("reference-matrix");
    for (policy, strategy, mode) in policies() {
        for (cell, (dims, gray, wide, associated, effort)) in [
            ((64, 32), false, false, false, 1),
            ((64, 32), false, false, true, 3),
            ((64, 32), true, false, false, 3),
            ((64, 32), true, false, true, 7),
            ((64, 32), false, true, false, 3),
            ((64, 32), false, true, true, 7),
            ((64, 32), true, true, false, 7),
            ((64, 32), true, true, true, 1),
            ((259, 133), false, false, false, 3),
        ]
        .into_iter()
        .enumerate()
        {
            let layout = match (gray, wide) {
                (false, false) => PixelLayout::Rgba8,
                (false, true) => PixelLayout::Rgba16,
                (true, false) => PixelLayout::GrayAlpha8,
                (true, true) => PixelLayout::GrayAlpha16,
            };
            let (reference, _) = pixels(dims, gray, wide, 3);
            let (displayed, expected) = pixels(dims, gray, wide, 37);
            let frames = [
                AnimationFrame::new(&reference, 0)
                    .with_reference_only(true)
                    .with_save_as_reference(2)
                    .with_name("hidden"),
                AnimationFrame::new(&displayed, 10),
            ];
            for lossy in [false, true] {
                let route = cell % 4;
                let encoded = encode(
                    lossy,
                    strategy.clone(),
                    mode,
                    effort,
                    dims,
                    layout,
                    associated,
                    false,
                    route,
                    &frames,
                );
                let control = encode(
                    lossy,
                    strategy.clone(),
                    mode,
                    effort,
                    dims,
                    layout,
                    associated,
                    false,
                    route,
                    &frames[1..],
                );
                let stem = format!("{policy}-{cell}-{lossy}");
                let path = dir.join(format!("{stem}.jxl"));
                let control_path = dir.join(format!("{stem}-control.jxl"));
                std::fs::write(&path, &encoded).unwrap();
                std::fs::write(&control_path, &control).unwrap();
                let cpp_control = reference_pixels(&control_path);
                let cpp = reference_pixels(&path);
                let rust_control = rust_frames(&control, 4);
                let rust = rust_frames(&encoded, 4);
                assert_eq!(rust.len(), 1, "{stem}");
                assert_eq!(
                    rust, rust_control,
                    "hidden reference altered displayed pixels: {stem}"
                );
                assert_eq!(cpp.len(), 1, "{stem}");
                assert_eq!(
                    cpp, cpp_control,
                    "hidden reference altered djxl pixels: {stem}"
                );
                for (i, (&r, &c)) in rust[0].iter().zip(&cpp[0]).enumerate() {
                    assert!(r.is_finite(), "{stem} sample {i}");
                    if !lossy || i % 4 == 3 {
                        assert_eq!(
                            (r * 65535.0).round() as u16,
                            expected[i],
                            "Rust {stem} sample {i}"
                        );
                        assert_eq!(c, expected[i], "djxl {stem} sample {i}");
                    }
                }
                let image = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&encoded))
                    .unwrap();
                image.render_frame(0).unwrap();
                let hidden = image.frame(0).unwrap().header();
                assert_eq!(
                    hidden.frame_type,
                    jxl_oxide::frame::FrameType::ReferenceOnly
                );
                assert_eq!(hidden.save_as_reference, 2);
                assert!(hidden.save_before_ct);
                assert_eq!(image.num_loaded_keyframes(), 1);
            }
        }
    }
}

#[test]
fn covering_and_partial_identity_crops_all_modes() {
    let dir = run_dir("crop-matrix");
    for (policy, strategy, mode) in policies() {
        for (cell, dims) in [
            (1, 1),
            (1, 8),
            (8, 1),
            (7, 7),
            (8, 8),
            (9, 9),
            (16, 16),
            (259, 133),
        ]
        .into_iter()
        .enumerate()
        {
            let (pixels, expected) = pixels(dims, false, false, 37);
            let frames = [
                AnimationFrame::new(&pixels, 10),
                AnimationFrame::new(&pixels, 20),
            ];
            for lossy in [false, true] {
                for auto in [false, true] {
                    let encoded = encode(
                        lossy,
                        strategy.clone(),
                        mode,
                        3,
                        dims,
                        PixelLayout::Rgba8,
                        false,
                        auto,
                        cell % 4,
                        &frames,
                    );
                    let stem = format!("{policy}-{cell}-{lossy}-{auto}");
                    let path = dir.join(format!("{stem}.jxl"));
                    std::fs::write(&path, &encoded).unwrap();
                    let rust = rust_frames(&encoded, 4);
                    let cpp = reference_pixels(&path);
                    assert_eq!(rust.len(), 2, "{stem}");
                    assert_eq!(cpp.len(), 2, "{stem}");
                    for frame in 0..2 {
                        for (i, (&r, &c)) in rust[frame].iter().zip(&cpp[frame]).enumerate() {
                            assert!(r.is_finite(), "{stem}");
                            if !lossy || i % 4 == 3 {
                                assert_eq!(
                                    (r * 65535.0).round() as u16,
                                    expected[i],
                                    "Rust {stem} sample {i}"
                                );
                                assert_eq!(c, expected[i], "djxl {stem} sample {i}");
                            }
                        }
                    }
                    if auto {
                        assert_eq!(rust[0], rust[1], "identity delta changed samples: {stem}");
                        assert_eq!(
                            cpp[0], cpp[1],
                            "djxl identity delta changed samples: {stem}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn explicit_icc_flag_requires_payload_at_every_color_request_entry() {
    use jxl_encoder::ColorEncoding;
    let pixels = [64_u8, 96, 128].repeat(8 * 8);
    for (name, strategy, mode) in policies() {
        let mut ce = ColorEncoding::srgb();
        ce.want_icc = true;
        let lossy = LossyConfig::new(1.0)
            .with_effort(1)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        let lossless = LosslessConfig::new()
            .with_effort(1)
            .with_strategy(strategy)
            .with_mode(mode);
        for animation in [false, true] {
            let frames = [AnimationFrame::new(&pixels, 10)];
            let a = lossy
                .encode_request(8, 8, PixelLayout::Rgb8)
                .with_color_encoding(ce.clone());
            let b = lossless
                .encode_request(8, 8, PixelLayout::Rgb8)
                .with_color_encoding(ce.clone());
            let result_a = if animation {
                a.encode_animation(&AnimationParams::default(), &frames)
            } else {
                a.encode(&pixels)
            };
            let result_b = if animation {
                b.encode_animation(&AnimationParams::default(), &frames)
            } else {
                b.encode(&pixels)
            };
            assert!(
                result_a.is_err(),
                "{name} lossy animation={animation} accepted want_icc without profile"
            );
            assert!(
                result_b.is_err(),
                "{name} lossless animation={animation} accepted want_icc without profile"
            );
        }
        let mut stream = lossless
            .encoder(8, 8, PixelLayout::Rgb8)
            .unwrap()
            .with_color_encoding(ce.clone());
        stream.push_rows(&pixels, 8).unwrap();
        assert!(
            stream.finish().is_err(),
            "{name} lossless streaming accepted want_icc without profile"
        );
        let mut stream = lossy
            .encoder(8, 8, PixelLayout::Rgb8)
            .unwrap()
            .with_color_encoding(ce);
        stream.push_rows(&pixels, 8).unwrap();
        assert!(
            stream.finish().is_err(),
            "{name} lossy streaming accepted want_icc without profile"
        );
        // A present profile is accepted even when the explicit flag is unset:
        // attaching metadata selects ICC output automatically.
        let metadata =
            ImageMetadata::new().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
        for flag in [false, true] {
            let mut ce = ColorEncoding::display_p3();
            ce.want_icc = flag;
            for animation in [false, true] {
                let frames = [AnimationFrame::new(&pixels, 10)];
                let a = lossy
                    .encode_request(8, 8, PixelLayout::Rgb8)
                    .with_color_encoding(ce.clone())
                    .with_metadata(&metadata);
                let b = lossless
                    .encode_request(8, 8, PixelLayout::Rgb8)
                    .with_color_encoding(ce.clone())
                    .with_metadata(&metadata);
                for (kind, result) in [
                    (
                        "lossy",
                        if animation {
                            a.encode_animation(&AnimationParams::default(), &frames)
                        } else {
                            a.encode(&pixels)
                        },
                    ),
                    (
                        "lossless",
                        if animation {
                            b.encode_animation(&AnimationParams::default(), &frames)
                        } else {
                            b.encode(&pixels)
                        },
                    ),
                ] {
                    let bytes = result.unwrap();
                    let directory = run_dir("icc-present");
                    let path = directory.join(format!("{name}-{kind}-{flag}-{animation}.jxl"));
                    std::fs::write(&path, &bytes).unwrap();
                    assert_eq!(rust_frames(&bytes, 3).len(), 1);
                    render_reference(&path);
                    let image = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    assert_eq!(
                        image.original_icc().unwrap(),
                        metadata.icc_profile().unwrap()
                    );
                }
            }
            let mut stream = lossless
                .encoder(8, 8, PixelLayout::Rgb8)
                .unwrap()
                .with_color_encoding(ce.clone())
                .with_icc_profile(metadata.icc_profile().unwrap());
            stream.push_rows(&pixels, 8).unwrap();
            let expected = lossless
                .encode_request(8, 8, PixelLayout::Rgb8)
                .with_color_encoding(ce.clone())
                .with_metadata(&metadata)
                .encode(&pixels)
                .unwrap();
            assert_eq!(stream.finish().unwrap(), expected);
            let mut stream = lossy
                .encoder(8, 8, PixelLayout::Rgb8)
                .unwrap()
                .with_color_encoding(ce.clone())
                .with_icc_profile(metadata.icc_profile().unwrap());
            stream.push_rows(&pixels, 8).unwrap();
            let expected = lossy
                .encode_request(8, 8, PixelLayout::Rgb8)
                .with_color_encoding(ce)
                .with_metadata(&metadata)
                .encode(&pixels)
                .unwrap();
            assert_eq!(stream.finish().unwrap(), expected);
        }
    }
}
