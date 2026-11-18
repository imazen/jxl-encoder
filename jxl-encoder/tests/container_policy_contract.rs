//! Configured container policy through actual pixel output owners.
#![cfg(feature = "__expert")]
use jxl_encoder::{ContainerMode, LosslessConfig, LossyConfig, PixelLayout};
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/frame_level_helpers.rs"]
mod helpers;
#[path = "support/encoder_policies.rs"]
mod policies;

fn directory() -> std::path::PathBuf {
    static ROOT: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let root = ROOT.get_or_init(helpers::directory);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = root.join(format!("case-{id}"));
    std::fs::create_dir(&path).unwrap();
    path
}

fn check(bytes: &[u8], wrapped: bool, width: usize, label: &str) {
    check_with_domain(bytes, wrapped, width, label, false, None)
}
fn check_with_domain(
    bytes: &[u8],
    wrapped: bool,
    width: usize,
    label: &str,
    linear: bool,
    alpha: Option<u8>,
) {
    assert_eq!(!bytes.starts_with(&[255, 10]), wrapped, "{label}");
    let out = directory();
    std::fs::write(out.join("input.jxl"), bytes).unwrap();
    std::fs::write(out.join("case.txt"), label).unwrap();
    let mut frames = decoded_frames::decode_current_frames(bytes, label);
    std::fs::write(
        out.join("primary-first.txt"),
        format!("{:?}\n", &frames[0].0[..3]),
    )
    .unwrap();
    assert_eq!(frames.len(), 1);
    // The fixture's ICC is explicitly sRGB. The pinned primary chooses
    // linear-sRGB float output for XYB+ICC and preserves signal values for
    // non-XYB. Compare both decoders in the native PNG's sRGB signal domain.
    if linear {
        for value in &mut frames[0].0 {
            *value = if *value <= 0.0031308 {
                *value * 12.92
            } else {
                1.055 * value.powf(1.0 / 2.4) - 0.055
            };
        }
    }
    assert_eq!(frames[0].0.len(), width * 3);
    assert_eq!(frames[0].1.len(), usize::from(alpha.is_some()));
    if let Some(alpha) = alpha {
        assert_eq!(frames[0].1[0].len(), width);
        assert!(
            frames[0].1[0]
                .iter()
                .all(|v| v.is_finite() && (*v * 65535.0).round() as u16 == u16::from(alpha) * 257)
        );
    }
    assert!(
        frames[0]
            .0
            .iter()
            .all(|v| v.is_finite() && (*v - 64.0 / 255.0).abs() <= 0.01)
    );
    let input = out.join("input.jxl");
    std::fs::write(&input, bytes).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(input)
        .arg(out.join("native.png"))
        .args([
            "--num_threads=1",
            "--bits_per_sample=16",
            "--color_space=RGB_D65_SRG_Rel_SRG",
        ])
        .output()
        .unwrap();
    std::fs::write(out.join("native.log"), &result.stderr).unwrap();
    assert!(result.status.success(), "{label}");
    let native = image::open(out.join("native.png")).unwrap().to_rgb16();
    assert_eq!(native.dimensions(), (width as u32, 1));
    if let Some(alpha) = alpha {
        let rgba = image::open(out.join("native.png")).unwrap().to_rgba16();
        assert!(rgba.pixels().all(|p| p.0[3] == u16::from(alpha) * 257));
    }
    assert!(
        native
            .as_raw()
            .iter()
            .zip(&frames[0].0)
            .all(|(n, p)| (f32::from(*n) / 65535.0 - p).abs() <= 0.001)
    );
}

#[test]
fn configured_always_wraps_ordinary_requests() {
    for (name, strategy, mode) in policies::policies() {
        let input = [64_u8; 3].repeat(32);
        for lossless in [false, true] {
            let bytes = if lossless {
                LosslessConfig::new()
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_container_mode(ContainerMode::Always)
                    .encode_request(32, 1, PixelLayout::Rgb8)
                    .encode(&input)
                    .unwrap()
            } else {
                LossyConfig::new(1.0)
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_container_mode(ContainerMode::Always)
                    .encode_request(32, 1, PixelLayout::Rgb8)
                    .encode(&input)
                    .unwrap()
            };
            check(
                &bytes,
                true,
                32,
                &format!("{name} lossless{lossless} always"),
            );
        }
    }
}

#[test]
fn configured_never_rejects_level_ten_before_writing() {
    let input = [64_u8; 3].repeat(262145);
    for (name, strategy, mode) in policies::policies() {
        for lossless in [false, true] {
            let mut output = b"existing destination".to_vec();
            let result = if lossless {
                LosslessConfig::new()
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_container_mode(ContainerMode::Never)
                    .encode_request(262145, 1, PixelLayout::Rgb8)
                    .encode_to(&input, &mut output)
            } else {
                LossyConfig::new(1.0)
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_container_mode(ContainerMode::Never)
                    .encode_request(262145, 1, PixelLayout::Rgb8)
                    .encode_to(&input, &mut output)
            };
            assert!(
                result.is_err(),
                "{name} lossless{lossless} admitted Level10 without a container"
            );
            assert_eq!(output, b"existing destination");
        }
    }
}

#[test]
fn pinned_v012_container_controls() {
    use std::io::Write;
    for (width, mode, wrapped, level) in
        [(32, 0, false, 5), (32, 1, true, 5), (262145, 0, true, 10)]
    {
        let out = directory();
        let input = out.join("input.ppm");
        let mut file = std::fs::File::create(&input).unwrap();
        writeln!(file, "P6\n{width} 1\n255").unwrap();
        file.write_all(&[64_u8; 3].repeat(width)).unwrap();
        drop(file);
        let output = out.join("reference.jxl");
        let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .arg(input)
            .arg(&output)
            .args(["-e", "1", "-d", "1", "--num_threads=1"])
            .arg(format!("--container={mode}"))
            .output()
            .unwrap();
        std::fs::write(out.join("reference.log"), &result.stderr).unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let bytes = std::fs::read(output).unwrap();
        assert_eq!(helpers::level(&bytes), level);
        check(&bytes, wrapped, width, "pinned v0.12 container control");
    }
}

#[test]
fn configured_modes_reach_streaming_and_animation() {
    let pixels = [64_u8; 3].repeat(32);
    let exif = b"II\x2a\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let metadata = jxl_encoder::ImageMetadata::new()
        .with_exif(exif)
        .with_xmp(b"<x/>");
    let animation = jxl_encoder::AnimationParams::default();
    let frames = [jxl_encoder::AnimationFrame::new(&pixels, 10)];
    for (name, strategy, encoder_mode) in policies::policies() {
        for mode in [
            ContainerMode::Auto,
            ContainerMode::Always,
            ContainerMode::Never,
        ] {
            for with_metadata in [false, true] {
                let wrapped =
                    mode == ContainerMode::Always || (mode == ContainerMode::Auto && with_metadata);
                for lossless in [false, true] {
                    let (streaming, animated, legacy) = if lossless {
                        let config = LosslessConfig::new()
                            .with_effort(1)
                            .with_strategy(strategy.clone())
                            .with_mode(encoder_mode)
                            .with_container_mode(mode);
                        let mut stream = config.encoder(32, 1, PixelLayout::Rgb8).unwrap();
                        if with_metadata {
                            stream = stream.with_exif(exif).with_xmp(b"<x/>");
                        }
                        stream.push_rows(&pixels, 1).unwrap();
                        let request = config.encode_request(32, 1, PixelLayout::Rgb8);
                        let request = if with_metadata {
                            request.with_metadata(&metadata)
                        } else {
                            request
                        };
                        (
                            stream.finish().unwrap(),
                            request.encode_animation(&animation, &frames).unwrap(),
                            config.encode(&pixels, 32, 1, PixelLayout::Rgb8).unwrap(),
                        )
                    } else {
                        let config = LossyConfig::new(1.0)
                            .with_effort(1)
                            .with_strategy(strategy.clone())
                            .with_mode(encoder_mode)
                            .with_container_mode(mode);
                        let mut stream = config.encoder(32, 1, PixelLayout::Rgb8).unwrap();
                        if with_metadata {
                            stream = stream.with_exif(exif).with_xmp(b"<x/>");
                        }
                        stream.push_rows(&pixels, 1).unwrap();
                        let request = config.encode_request(32, 1, PixelLayout::Rgb8);
                        let request = if with_metadata {
                            request.with_metadata(&metadata)
                        } else {
                            request
                        };
                        (
                            stream.finish().unwrap(),
                            request.encode_animation(&animation, &frames).unwrap(),
                            config.encode(&pixels, 32, 1, PixelLayout::Rgb8).unwrap(),
                        )
                    };
                    for (label, bytes) in [("streaming", &streaming), ("animation", &animated)] {
                        check(
                            bytes,
                            wrapped,
                            32,
                            &format!(
                                "{name} {mode:?} metadata{with_metadata} lossless{lossless} {label}"
                            ),
                        );
                        if wrapped && with_metadata {
                            assert!(bytes.windows(4).any(|v| v == b"Exif"));
                            assert!(bytes.windows(4).any(|v| v == b"xml "));
                        }
                    }
                    check(
                        &legacy,
                        mode == ContainerMode::Always,
                        32,
                        "legacy config encode",
                    );
                }
            }
        }
    }
}

#[test]
fn configured_never_reaches_streaming_and_animation_level_ten_guards() {
    let pixels = [64_u8; 3].repeat(262145);
    let animation = jxl_encoder::AnimationParams::default();
    let frames = [jxl_encoder::AnimationFrame::new(&pixels, 10)];
    for (name, strategy, encoder_mode) in policies::policies() {
        for lossless in [false, true] {
            let (streaming, animated) = if lossless {
                let config = LosslessConfig::new()
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(encoder_mode)
                    .with_container_mode(ContainerMode::Never);
                let mut stream = config.encoder(262145, 1, PixelLayout::Rgb8).unwrap();
                stream.push_rows(&pixels, 1).unwrap();
                (
                    stream.finish(),
                    config.encode_animation(262145, 1, PixelLayout::Rgb8, &animation, &frames),
                )
            } else {
                let config = LossyConfig::new(1.0)
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(encoder_mode)
                    .with_container_mode(ContainerMode::Never);
                let mut stream = config.encoder(262145, 1, PixelLayout::Rgb8).unwrap();
                stream.push_rows(&pixels, 1).unwrap();
                (
                    stream.finish(),
                    config.encode_animation(262145, 1, PixelLayout::Rgb8, &animation, &frames),
                )
            };
            for result in [streaming, animated] {
                let error = result.unwrap_err();
                assert!(
                    error.to_string().contains("container mode Never"),
                    "{name} lossless{lossless}: {error}"
                );
            }
        }
    }
}

fn codestream(bytes: &[u8]) -> Vec<u8> {
    if bytes.starts_with(&[255, 10]) {
        return bytes.to_vec();
    }
    let mut output = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && offset + size <= bytes.len());
        match &bytes[offset + 4..offset + 8] {
            b"jxlc" => output.extend_from_slice(&bytes[offset + 8..offset + size]),
            b"jxlp" => {
                assert!(size >= 12);
                output.extend_from_slice(&bytes[offset + 12..offset + size]);
            }
            _ => {}
        }
        offset += size;
    }
    assert!(output.starts_with(&[255, 10]));
    output
}

#[test]
fn configured_never_keeps_icc_and_drops_only_container_metadata() {
    let pixels = [64_u8; 3].repeat(32);
    let hex = include_str!("../src/headers/color_profiles_v012.tsv")
        .lines()
        .find(|line| line.starts_with("0\t"))
        .unwrap()
        .split_once('\t')
        .unwrap()
        .1;
    let profile: Vec<u8> = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u8::from_str_radix(core::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect();
    let exif = b"II\x2a\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let metadata = jxl_encoder::ImageMetadata::new()
        .with_icc_profile(&profile)
        .with_exif(exif)
        .with_xmp(b"<x/>");
    for (name, strategy, encoder_mode) in policies::policies() {
        for lossless in [false, true] {
            let mut previous = None;
            for mode in [
                ContainerMode::Auto,
                ContainerMode::Always,
                ContainerMode::Never,
            ] {
                let bytes = if lossless {
                    LosslessConfig::new()
                        .with_effort(1)
                        .with_strategy(strategy.clone())
                        .with_mode(encoder_mode)
                        .with_container_mode(mode)
                        .encode_request(32, 1, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .encode(&pixels)
                        .unwrap()
                } else {
                    LossyConfig::new(1.0)
                        .with_effort(1)
                        .with_strategy(strategy.clone())
                        .with_mode(encoder_mode)
                        .with_container_mode(mode)
                        .encode_request(32, 1, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .encode(&pixels)
                        .unwrap()
                };
                check_with_domain(
                    &bytes,
                    mode != ContainerMode::Never,
                    32,
                    &format!("{name} lossless{lossless} {mode:?} ICC"),
                    !lossless,
                    None,
                );
                let parsed = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                assert_eq!(parsed.original_icc(), Some(profile.as_slice()));
                let stream = codestream(&bytes);
                if let Some(old) = &previous {
                    assert_eq!(
                        &stream, old,
                        "container policy changed codestream or embedded ICC"
                    );
                }
                previous = Some(stream);
            }
        }
    }
}

#[test]
fn configured_policy_reaches_every_pixel_output_wrapper() {
    use std::io::Write;
    let out = directory();
    let mut journal =
        std::io::BufWriter::new(std::fs::File::create(out.join("wrappers.tsv")).unwrap());
    macro_rules! exercise {
        ($config:expr, $pixels:expr, $width:expr, $name:expr, $lossless:expr, $mode:expr, $layout:expr) => {{
            let matrix_config = &$config;
            let pixels = &$pixels;
            let expected = matrix_config.encode(pixels, $width, 1, $layout);
            if let Err(error) = &expected {
                assert!(error.to_string().contains("container mode Never"));
            }
            for route in 0..7 {
                let prefix = b"existing destination";
                let mut sink = prefix.to_vec();
                let result = match route {
                    0 => matrix_config
                        .encode_request($width, 1, $layout)
                        .encode_with_stats(pixels)
                        .map(|mut r| {
                            sink.extend(r.take_data().unwrap());
                        }),
                    1 => matrix_config
                        .encode_request($width, 1, $layout)
                        .encode_into(pixels, &mut sink)
                        .map(|_| ()),
                    2 => matrix_config
                        .encode_request($width, 1, $layout)
                        .encode_to(pixels, &mut sink)
                        .map(|_| ()),
                    _ => {
                        let mut encoder = matrix_config.encoder($width, 1, $layout).unwrap();
                        encoder.push_rows(pixels, 1).unwrap();
                        match route {
                            3 => encoder.finish_with_stats().map(|mut r| {
                                sink.extend(r.take_data().unwrap());
                            }),
                            4 => encoder.finish_into(&mut sink).map(|_| ()),
                            5 => encoder.finish_to(&mut sink).map(|_| ()),
                            6 => {
                                let mut cursor = std::io::Cursor::new(sink);
                                cursor.set_position(prefix.len() as u64);
                                let result = encoder.finish_to_seekable(&mut cursor).map(|_| ());
                                sink = cursor.into_inner();
                                result
                            }
                            _ => unreachable!(),
                        }
                    }
                };
                match &expected {
                    Ok(bytes) => {
                        result.unwrap();
                        assert_eq!(&sink[..prefix.len()], prefix);
                        assert_eq!(&sink[prefix.len()..], bytes);
                    }
                    Err(_) => {
                        assert!(
                            result
                                .unwrap_err()
                                .to_string()
                                .contains("container mode Never")
                        );
                        assert_eq!(sink, prefix);
                    }
                }
            }
            let mut legacy = b"existing destination".to_vec();
            let result = matrix_config.encode_into(pixels, $width, 1, $layout, &mut legacy);
            match &expected {
                Ok(bytes) => {
                    result.unwrap();
                    assert_eq!(&legacy[..20], b"existing destination");
                    assert_eq!(&legacy[20..], bytes);
                }
                Err(_) => {
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("container mode Never")
                    );
                    assert_eq!(legacy, b"existing destination");
                }
            }
            writeln!(
                journal,
                "{}\t{}\t{}\t{:?}\t{:?}\tcomplete",
                $name, $lossless, $width, $mode, $layout
            )
            .unwrap();
            journal.flush().unwrap();
        }};
    }
    for (name, strategy, encoder_mode) in policies::policies() {
        for level_ten in [false, true] {
            let width = 32_u32;
            let layout = if level_ten {
                PixelLayout::RgbLinearF32
            } else {
                PixelLayout::Rgb8
            };
            let pixels: Vec<u8> = if level_ten {
                (0..width * 3)
                    .flat_map(|_| 0.25_f32.to_ne_bytes())
                    .collect()
            } else {
                [64_u8; 3].repeat(width as usize)
            };
            let modes = if !level_ten {
                &[
                    ContainerMode::Auto,
                    ContainerMode::Always,
                    ContainerMode::Never,
                ][..]
            } else {
                &[ContainerMode::Never][..]
            };
            for &mode in modes {
                let lossless = LosslessConfig::new()
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(encoder_mode)
                    .with_container_mode(mode);
                exercise!(lossless, pixels, width, name, true, mode, layout);
                let lossy = LossyConfig::new(1.0)
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(encoder_mode)
                    .with_container_mode(mode);
                exercise!(lossy, pixels, width, name, false, mode, layout);
            }
        }
    }
}

#[test]
fn configured_optional_boxes_and_animation_profile_branches() {
    let pixels = [64_u8, 64, 64, 137].repeat(32);
    let profile = include_str!("../src/headers/color_profiles_v012.tsv")
        .lines()
        .find(|l| l.starts_with("0\t"))
        .unwrap()
        .split_once('\t')
        .unwrap()
        .1;
    let profile: Vec<u8> = profile
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u8::from_str_radix(core::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect();
    let metadata = jxl_encoder::ImageMetadata::new()
        .with_icc_profile(&profile)
        .with_xmp(b"<x/>")
        .with_colr_payload(b"nclx")
        .with_hcdr_payload(b"hdr");
    let animation = jxl_encoder::AnimationParams::default();
    let frames = [jxl_encoder::AnimationFrame::new(&pixels, 10)];
    let cmyk_profile: Vec<u8> = include_str!("fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|hex| {
            (0..hex.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        })
        .collect();
    let cmyk_metadata = jxl_encoder::ImageMetadata::new().with_icc_profile(&cmyk_profile);
    let cmyk_pixels = [255_u8; 4].repeat(32);
    let cmyk_frames = [jxl_encoder::AnimationFrame::new(&cmyk_pixels, 10)];
    for (name, strategy, encoder_mode) in policies::policies() {
        for mode in [
            ContainerMode::Auto,
            ContainerMode::Always,
            ContainerMode::Never,
        ] {
            for lossless in [false, true] {
                let bytes = if lossless {
                    LosslessConfig::new()
                        .with_effort(1)
                        .with_strategy(strategy.clone())
                        .with_mode(encoder_mode)
                        .with_container_mode(mode)
                        .encode_animation_with_metadata(
                            32,
                            1,
                            PixelLayout::Rgba8,
                            &animation,
                            &frames,
                            &metadata,
                        )
                        .unwrap()
                } else {
                    LossyConfig::new(1.0)
                        .with_effort(1)
                        .with_strategy(strategy.clone())
                        .with_mode(encoder_mode)
                        .with_container_mode(mode)
                        .encode_animation_with_metadata(
                            32,
                            1,
                            PixelLayout::Rgba8,
                            &animation,
                            &frames,
                            &metadata,
                        )
                        .unwrap()
                };
                check_with_domain(
                    &bytes,
                    mode != ContainerMode::Never,
                    32,
                    &format!("{name} {mode:?} lossless{lossless} alpha/profile/optional boxes"),
                    !lossless,
                    Some(137),
                );
                let parsed = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                assert_eq!(parsed.original_icc(), Some(profile.as_slice()));
                if mode != ContainerMode::Never {
                    let mut offset = 0;
                    let mut seen = [false; 2];
                    while offset < bytes.len() {
                        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
                            as usize;
                        assert!(size >= 8 && offset + size <= bytes.len());
                        match &bytes[offset + 4..offset + 8] {
                            b"colr" => {
                                assert_eq!(&bytes[offset + 8..offset + size], b"nclx");
                                seen[0] = true;
                            }
                            b"hCdR" => {
                                assert_eq!(&bytes[offset + 8..offset + size], b"hdr");
                                seen[1] = true;
                            }
                            _ => {}
                        }
                        offset += size;
                    }
                    assert_eq!(seen, [true; 2]);
                }
            }
        }
        for lossless in [false, true] {
            let result = if lossless {
                LosslessConfig::new()
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(encoder_mode)
                    .with_container_mode(ContainerMode::Never)
                    .encode_animation_with_metadata(
                        32,
                        1,
                        PixelLayout::Cmyk8,
                        &animation,
                        &cmyk_frames,
                        &cmyk_metadata,
                    )
            } else {
                LossyConfig::new(1.0)
                    .with_effort(1)
                    .with_strategy(strategy.clone())
                    .with_mode(encoder_mode)
                    .with_container_mode(ContainerMode::Never)
                    .encode_animation_with_metadata(
                        32,
                        1,
                        PixelLayout::Cmyk8,
                        &animation,
                        &cmyk_frames,
                        &cmyk_metadata,
                    )
            };
            let error = result.unwrap_err();
            if lossless {
                // The lossless animation writer supports integer RGB/gray,
                // including alpha; it has no interleaved CMYK frame route.
                assert!(
                    matches!(
                        error.error(),
                        jxl_encoder::EncodeError::UnsupportedPixelLayout(PixelLayout::Cmyk8)
                    ),
                    "{name} lossless CMYK: {error}"
                );
            } else {
                assert!(
                    error.to_string().contains("container mode Never"),
                    "{name} lossy CMYK: {error}"
                );
            }
        }
    }
}
