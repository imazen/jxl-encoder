//! Independent composition of native-decoded CMYK crop layers.
#![cfg(all(feature = "__expert", feature = "std"))]
use jxl_encoder::{
    AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout,
    ProgressiveMode,
};
use std::{io::Write, path::Path};
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/decoded_frames_u16.rs"]
mod decoded_frames_u16;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;
fn profile() -> Vec<u8> {
    include_str!("fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect()
}
fn nonzero_black_profile() -> Vec<u8> {
    let mut profile = profile();
    let count = u32::from_be_bytes(profile[128..132].try_into().unwrap()) as usize;
    let mut found = false;
    for i in 0..count {
        let tag = profile[132 + i * 12..144 + i * 12].to_vec();
        if &tag[..4] != b"A2B0" {
            continue;
        }
        let offset = u32::from_be_bytes(tag[4..8].try_into().unwrap()) as usize;
        assert_eq!(&profile[offset..offset + 4], b"mft2");
        let (inputs, outputs, grid) = (
            usize::from(profile[offset + 8]),
            usize::from(profile[offset + 9]),
            usize::from(profile[offset + 10]),
        );
        assert_eq!((inputs, outputs, grid), (4, 3, 5));
        let entries = usize::from(u16::from_be_bytes(
            profile[offset + 48..offset + 50].try_into().unwrap(),
        ));
        let last = offset + 52 + inputs * entries * 2 + (grid.pow(inputs as u32) - 1) * outputs * 2;
        assert_eq!(&profile[last..last + 6], &[0_u8; 6]);
        for channel in 0..3 {
            profile[last + channel * 2..last + channel * 2 + 2]
                .copy_from_slice(&3277_u16.to_be_bytes());
        }
        found = true;
    }
    assert!(found);
    profile[84..100].fill(0);
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(&profile)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect::<String>();
    assert_eq!(
        hash,
        "668951805a35b3aebe32da9a32302b169811c3f20304a12af4a9271ca1b8b27b"
    );
    profile
}

fn compose(
    base: &[f32],
    layer: &[f32],
    canvas: (usize, usize),
    crop: (usize, usize, usize, usize),
    channels: usize,
    blend: u32,
) -> Vec<f32> {
    let (w, h) = canvas;
    let (x0, y0, cw, ch) = crop;
    assert!(x0 + cw <= w && y0 + ch <= h);
    assert_eq!(base.len(), w * h * channels);
    assert_eq!(layer.len(), cw * ch * channels);
    assert!(matches!(blend, 0 | 1));
    let mut result = base.to_vec();
    for y in 0..ch {
        for x in 0..cw {
            for c in 0..channels {
                let target = ((y + y0) * w + x + x0) * channels + c;
                let value = layer[(y * cw + x) * channels + c];
                result[target] = if blend == 0 {
                    value
                } else {
                    base[target] + value
                };
            }
        }
    }
    result
}
type NativeFrame = (Vec<f32>, Vec<f32>);

fn native_layers(
    out: &Path,
    bytes: &[u8],
    canvas: (usize, usize),
    oracle: &std::ffi::OsStr,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    native_layers_with_control_reference(out, bytes, canvas, oracle, None)
}
// A hidden ReferenceOnly image is not exposed by the native public decoder.
// A supplied seed is a separately decoded source control, not a pixel
// observation of the hidden frame in this stream.
fn native_layers_with_control_reference(
    out: &Path,
    bytes: &[u8],
    canvas: (usize, usize),
    oracle: &std::ffi::OsStr,
    control: Option<(usize, NativeFrame)>,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    native_layers_in_domain(out, bytes, canvas, oracle, control, "linear-layers")
}
fn native_layers_in_domain(
    out: &Path,
    bytes: &[u8],
    canvas: (usize, usize),
    oracle: &std::ffi::OsStr,
    control: Option<(usize, NativeFrame)>,
    domain: &str,
) -> Vec<NativeFrame> {
    std::fs::create_dir(out).unwrap();
    let input = out.join("animation.jxl");
    std::fs::write(&input, bytes).unwrap();
    let native = std::process::Command::new(oracle)
        .arg(&input)
        .arg(out.join("layer"))
        .arg(domain)
        .output()
        .unwrap();
    std::fs::write(out.join("headers.tsv"), &native.stdout).unwrap();
    std::fs::write(out.join("native.log"), &native.stderr).unwrap();
    assert!(
        native.status.success(),
        "{}: {}",
        out.display(),
        String::from_utf8_lossy(&native.stderr)
    );
    let mut refs: [(Vec<f32>, Vec<f32>); 4] = std::array::from_fn(|_| {
        (
            vec![0.0; canvas.0 * canvas.1 * 3],
            vec![0.0; canvas.0 * canvas.1],
        )
    });
    if let Some((slot, pixels)) = control {
        assert!((1..=3).contains(&slot));
        refs[slot] = pixels;
    }
    let mut frames = Vec::new();
    for line in String::from_utf8(native.stdout).unwrap().lines() {
        let row: Vec<i64> = line.split('\t').map(|v| v.parse().unwrap()).collect();
        assert_eq!(row.len(), 12);
        assert!(row[1] > 0);
        let i = row[0];
        let crop = (
            usize::try_from(row[3]).unwrap(),
            usize::try_from(row[4]).unwrap(),
            usize::try_from(row[5]).unwrap(),
            usize::try_from(row[6]).unwrap(),
        );
        let color = pfm::read_pfm_samples(&out.join(format!("layer-{i}.pfm")), (crop.2, crop.3));
        let black =
            pfm::read_pfm_samples(&out.join(format!("layer-{i}-black.pfm")), (crop.2, crop.3));
        assert!(color.iter().chain(&black).all(|v| v.is_finite()));
        let color = compose(
            &refs[usize::try_from(row[9]).unwrap()].0,
            &color,
            canvas,
            crop,
            3,
            row[7] as u32,
        );
        let black = compose(
            &refs[usize::try_from(row[10]).unwrap()].1,
            &black,
            canvas,
            crop,
            1,
            row[8] as u32,
        );
        if row[11] != 0 {
            refs[usize::try_from(row[11]).unwrap()] = (color.clone(), black.clone());
        }
        frames.push((color, black));
    }
    frames
}
#[test]
fn cmyk_cropped_and_identity_delta_frames_keep_black_and_color() {
    let root = jxl_encoder::test_helpers::output_dir("cmyk-animation-layers");
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native layer observer");
    let mut journal = std::fs::File::create(out.join("composition.tsv")).unwrap();
    for (profile_name, profile) in [
        ("standard", profile()),
        ("nonzero-black", nonzero_black_profile()),
    ] {
        let metadata = ImageMetadata::new().with_icc_profile(&profile);
        for (policy, strategy, mode) in encoder_policies::policies() {
            for (width, height) in [(32_usize, 17_usize), (259, 17)] {
                for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                    let wide = layout.is_16bit();
                    let scale = if wide { 257 } else { 1 };
                    let maximum = if wide { 65535.0 } else { 255.0 };
                    let swatch = [64_u8, 128, 192, 224];
                    let pixel: Vec<u8> = swatch
                        .into_iter()
                        .flat_map(|v| {
                            if wide {
                                (u16::from(v) * 257).to_ne_bytes().to_vec()
                            } else {
                                vec![v]
                            }
                        })
                        .collect();
                    let pixels = pixel.repeat(width * height);
                    for delta in [false, true] {
                        for progressive in [
                            ProgressiveMode::Single,
                            ProgressiveMode::QuantizedAcFullAc,
                            ProgressiveMode::DcVlfLfAc,
                        ] {
                            let lossy = LossyConfig::new(0.01)
                                .with_effort(3)
                                .with_threads(1)
                                .with_strategy(strategy.clone())
                                .with_mode(mode)
                                .with_auto_delta_frames(delta)
                                .with_progressive(progressive);
                            let request = lossy
                                .encode_request(width as u32, height as u32, layout)
                                .with_metadata(&metadata);
                            let bytes = request
                                .encode_animation(
                                    &AnimationParams::default(),
                                    &[
                                        AnimationFrame::new(&pixels, 10),
                                        AnimationFrame::new(&pixels, 20),
                                    ],
                                )
                                .unwrap();
                            let label = format!(
                                "{profile_name}-{policy}-{width}-{layout:?}-lossy-{delta}-{progressive:?}"
                            );
                            std::fs::write(out.join(format!("{label}.jxl")), &bytes).unwrap();
                            let frames = decoded_frames::decode_current_frames(&bytes, &label);
                            assert_eq!(frames.len(), 2);
                            let native =
                                native_layers(&out.join(&label), &bytes, (width, height), &oracle);
                            assert_eq!(native.len(), 2);
                            for (color, extras) in &frames {
                                assert_eq!(extras.len(), 1);
                                for (index, value) in extras[0].iter().enumerate() {
                                    assert!(value.is_finite());
                                    assert_eq!(
                                        (*value * maximum).round() as u16,
                                        224 * scale,
                                        "{label} Black pixel{index}"
                                    );
                                }
                                let expected: Vec<f32> =
                                    include_str!("fixtures/cmyk_lcms_swatches.hex")
                                        .lines()
                                        .nth(5)
                                        .unwrap()
                                        .split_whitespace()
                                        .map(|v| {
                                            f32::from_bits(u32::from_str_radix(v, 16).unwrap())
                                        })
                                        .collect();
                                let error = color
                                    .iter()
                                    .zip(expected.iter().cycle())
                                    .map(|(a, b)| (a - b).abs())
                                    .fold(0_f32, f32::max);
                                assert!(error <= 0.005, "{label} LCMS physical error {error}");
                            }
                            let error = frames[0]
                                .0
                                .iter()
                                .zip(&frames[1].0)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            assert!(
                                error <= 0.001,
                                "{label} identical source color changed {error}"
                            );
                            for (i, (color, black)) in native.into_iter().enumerate() {
                                assert!(
                                    black
                                        .iter()
                                        .all(|v| (*v * maximum).round() as u16 == 224 * scale),
                                    "{label} native Black"
                                );
                                assert_eq!(color.len(), frames[i].0.len());
                                let error = color
                                    .iter()
                                    .zip(&frames[i].0)
                                    .map(|(a, b)| (a - b).abs())
                                    .fold(0_f32, f32::max);
                                assert!(error <= 0.001, "{label} composed native/primary {error}");
                            }
                            writeln!(
                                journal,
                                "{label}\texact Black; independently composed color"
                            )
                            .unwrap();
                            journal.flush().unwrap();
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cmyk_lossless_animation_is_explicitly_unsupported() {
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    for (_, strategy, mode) in encoder_policies::policies() {
        for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
            let pixels = vec![64; 32 * 17 * layout.bytes_per_pixel()];
            let error = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .encode_request(32, 17, layout)
                .with_metadata(&metadata)
                .encode_animation(
                    &AnimationParams::default(),
                    &[AnimationFrame::new(&pixels, 10)],
                )
                .unwrap_err();
            assert!(
                format!("{error:?}").contains("UnsupportedPixelLayout"),
                "{error:?}"
            );
        }
    }
}

#[test]
fn cmyk_animation_limits_preserve_pixels_and_conversion_errors_propagate() {
    let root = jxl_encoder::test_helpers::output_dir("cmyk-animation-limits");
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("native layer observer");
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let mut opaque = vec![0_u8; 20];
    opaque[16..20].copy_from_slice(b"CMYK");
    let unavailable = ImageMetadata::new().with_icc_profile(&opaque);
    for (policy, strategy, mode) in encoder_policies::policies() {
        for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
            let wide = layout.is_16bit();
            let maximum = if wide { 65535.0 } else { 255.0 };
            let code = if wide { 0xe0e0 } else { 224 };
            let pixel: Vec<u8> = [64_u8, 128, 192, 224]
                .into_iter()
                .flat_map(|v| {
                    if wide {
                        (u16::from(v) * 257).to_ne_bytes().to_vec()
                    } else {
                        vec![v]
                    }
                })
                .collect();
            let pixels = pixel.repeat(32 * 17);
            let config = LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_auto_delta_frames(true);
            let frames = [
                AnimationFrame::new(&pixels, 10),
                AnimationFrame::new(&pixels, 20),
            ];
            let mut baseline = None;
            for fallible in [false, true] {
                let limits = jxl_encoder::Limits::default().with_fallible_alloc(fallible);
                let bytes = config
                    .encode_request(32, 17, layout)
                    .with_metadata(&metadata)
                    .with_limits(&limits)
                    .encode_animation(&AnimationParams::default(), &frames)
                    .unwrap();
                if let Some(control) = &baseline {
                    assert_eq!(&bytes, control);
                } else {
                    baseline = Some(bytes.clone());
                }
                let label = format!("{policy}-{layout:?}-{fallible}");
                let native = native_layers(&out.join(&label), &bytes, (32, 17), &oracle);
                let primary = decoded_frames::decode_current_frames(&bytes, &label);
                assert_eq!(native.len(), 2);
                assert_eq!(primary.len(), 2);
                for ((colors, black), (primary_color, primary_planes)) in
                    native.iter().zip(&primary)
                {
                    assert!(
                        black
                            .iter()
                            .chain(&primary_planes[0])
                            .all(|v| v.is_finite() && (*v * maximum).round() as u16 == code)
                    );
                    assert_eq!(colors.len(), primary_color.len());
                    assert!(
                        colors
                            .iter()
                            .zip(primary_color)
                            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 0.001)
                    );
                }
                let error = config
                    .encode_request(32, 17, layout)
                    .with_metadata(&unavailable)
                    .with_limits(&limits)
                    .encode_animation(&AnimationParams::default(), &frames)
                    .unwrap_err();
                assert!(error.to_string().contains("CMYK ICC"), "{label}: {error}");
                std::fs::write(out.join(format!("{label}-error.txt")), error.to_string()).unwrap();
            }
        }
    }
}

#[test]
fn cmyk_reference_only_and_edge_frames_preserve_displayed_black_and_light() {
    use jxl_encoder::BlendMode;
    let root = jxl_encoder::test_helpers::output_dir("cmyk-reference-edges");
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("native layer observer");
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let expected: Vec<f32> = include_str!("fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
        .collect();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [
            (1_usize, 1_usize),
            (1, 259),
            (259, 1),
            (17, 259),
            (257, 257),
        ] {
            for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                let wide = layout.is_16bit();
                let maximum = if wide { 65535.0 } else { 255.0 };
                let code = if wide { 0xe0e0 } else { 224 };
                let pixel: Vec<u8> = [64_u8, 128, 192, 224]
                    .into_iter()
                    .flat_map(|v| {
                        if wide {
                            (u16::from(v) * 257).to_ne_bytes().to_vec()
                        } else {
                            vec![v]
                        }
                    })
                    .collect();
                let pixels = pixel.repeat(width * height);
                for hidden in [false, true] {
                    for progressive in [ProgressiveMode::Single, ProgressiveMode::DcVlfLfAc] {
                        let config = LossyConfig::new(0.01)
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_progressive(progressive)
                            .with_auto_delta_frames(false);
                        let mut frames = Vec::new();
                        if hidden {
                            frames.push(
                                AnimationFrame::new(&pixels, 0)
                                    .with_reference_only(true)
                                    .with_save_as_reference(3),
                            );
                        }
                        frames.push(
                            AnimationFrame::new(&pixels, 10).with_blend_mode(BlendMode::Replace),
                        );
                        let bytes = config
                            .encode_request(width as u32, height as u32, layout)
                            .with_metadata(&metadata)
                            .encode_animation(&AnimationParams::default(), &frames)
                            .unwrap();
                        let label = format!(
                            "{policy}-{width}x{height}-{layout:?}-{hidden}-{progressive:?}"
                        );
                        let native =
                            native_layers(&out.join(&label), &bytes, (width, height), &oracle);
                        let primary = decoded_frames::decode_current_frames(&bytes, &label);
                        assert_eq!(native.len(), 1, "{label}");
                        assert_eq!(primary.len(), 1, "{label}");
                        for colors in [&native[0].0, &primary[0].0] {
                            assert_eq!(colors.len(), width * height * 3);
                            assert!(colors.iter().all(|v| v.is_finite()));
                            let error = colors
                                .iter()
                                .zip(expected.iter().cycle())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            assert!(error <= 0.005, "{label} physical error {error}");
                        }
                        for black in [&native[0].1, &primary[0].1[0]] {
                            assert_eq!(black.len(), width * height);
                            assert!(black.iter().all(|v|v.is_finite()&&(*v*maximum).round()as u16==code),"{label} exact Black");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cmyk_explicit_add_uses_the_hidden_color_reference_and_replaces_black() {
    use jxl_encoder::BlendMode;
    let root = jxl_encoder::test_helpers::output_dir("cmyk-explicit-add");
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("native layer observer");
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let physical: Vec<f32> = include_str!("fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
        .collect();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(16_usize, 16_usize), (17, 259), (257, 257)] {
            for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                let wide = layout.is_16bit();
                let maximum = if wide { 65535.0 } else { 255.0 };
                let code = if wide { 0xe0e0 } else { 224 };
                let pack = |pixel: [u8; 4]| {
                    pixel
                        .into_iter()
                        .flat_map(|v| {
                            if wide {
                                (u16::from(v) * 257).to_ne_bytes().to_vec()
                            } else {
                                vec![v]
                            }
                        })
                        .collect::<Vec<_>>()
                        .repeat(width * height)
                };
                let source = pack([64, 128, 192, 224]);
                let black = pack([255, 255, 255, 0]);
                for progressive in [ProgressiveMode::Single, ProgressiveMode::DcVlfLfAc] {
                    let config = LossyConfig::new(0.01)
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_progressive(progressive)
                        .with_auto_delta_frames(true);
                    let prefix = format!("{policy}-{width}x{height}-{layout:?}-{progressive:?}");
                    let control =
                        config
                            .encode_request(width as u32, height as u32, layout)
                            .with_metadata(&metadata)
                            .encode_animation(
                                &AnimationParams::default(),
                                &[AnimationFrame::new(&source, 10)
                                    .with_blend_mode(BlendMode::Replace)],
                            )
                            .unwrap();
                    let mut control_frames = native_layers_in_domain(
                        &out.join(format!("{prefix}-source-control")),
                        &control,
                        (width, height),
                        &oracle,
                        None,
                        "srgb-layers",
                    );
                    assert_eq!(control_frames.len(), 1);
                    let reference = control_frames.remove(0);
                    for hidden in [false, true] {
                        let mut frames = vec![
                            AnimationFrame::new(if hidden { &black } else { &source }, 10)
                                .with_blend_mode(BlendMode::Replace)
                                .with_save_as_reference(if hidden { 1 } else { 3 }),
                        ];
                        if hidden {
                            frames.push(
                                AnimationFrame::new(&source, 0)
                                    .with_reference_only(true)
                                    .with_save_as_reference(3),
                            );
                        }
                        frames.push(
                            AnimationFrame::new(&source, 20)
                                .with_blend_mode(BlendMode::Add)
                                .with_blend_source(3),
                        );
                        let bytes = config
                            .encode_request(width as u32, height as u32, layout)
                            .with_metadata(&metadata)
                            .encode_animation(&AnimationParams::default(), &frames)
                            .unwrap();
                        let label = format!("{prefix}-{hidden}");
                        let dir = out.join(&label);
                        let native = native_layers_in_domain(
                            &dir,
                            &bytes,
                            (width, height),
                            &oracle,
                            hidden.then(|| (3, reference.clone())),
                            "srgb-layers",
                        );
                        let primary = decoded_frames_u16::decode_current_frames_u16(&bytes, &label);
                        assert_eq!(native.len(), 2);
                        assert_eq!(primary.len(), 2);
                        if policy == "libjxl"
                            && width == 16
                            && !wide
                            && progressive == ProgressiveMode::Single
                        {
                            // Diagnostic only: the primary decoder selects linear sRGB
                            // for F32 ICC output. It must still blend in nonlinear sRGB
                            // (Part 1 E.2/F.2/L.2). Integer output above is the pixel
                            // oracle; retain the float disagreement independently.
                            let float = decoded_frames::decode_current_frames(&bytes, &label);
                            let mut report = String::from(
                                "sample\tnative_srgb_composite\texpected_linear\tobserved_linear\n",
                            );
                            let mut maximum = 0.0_f32;
                            for (i, (&signal, &actual)) in
                                native[1].0.iter().zip(&float[1].0).enumerate()
                            {
                                let expected = if signal <= 0.04045 {
                                    signal / 12.92
                                } else {
                                    ((signal + 0.055) / 1.055).powf(2.4)
                                };
                                maximum = maximum.max((actual - expected).abs());
                                if i < 3 {
                                    report.push_str(&format!(
                                        "{i}\t{signal:.10}\t{expected:.10}\t{actual:.10}\n"
                                    ));
                                }
                            }
                            report.push_str(&format!("max_error\t{maximum:.10}\n"));
                            std::fs::write(dir.join("float-output-diagnostic.tsv"), report)
                                .unwrap();
                        }
                        let headers = std::fs::read_to_string(dir.join("headers.tsv")).unwrap();
                        let last: Vec<i64> = headers
                            .lines()
                            .last()
                            .unwrap()
                            .split('\t')
                            .map(|v| v.parse().unwrap())
                            .collect();
                        assert_eq!(
                            (last[7], last[8], last[9]),
                            (1, 0, 3),
                            "{label} actual main Add / Black Replace / reference3"
                        );
                        assert_eq!(
                            (last[3], last[4], last[5], last[6]),
                            (0, 0, width as i64, height as i64),
                            "{label} full-canvas explicit Add"
                        );
                        for (colors, integer_output) in
                            [(&native[1].0, false), (&primary[1].0, true)]
                        {
                            assert_eq!(colors.len(), width * height * 3);
                            assert!(colors.iter().all(|v| v.is_finite()));
                            let error = colors
                                .iter()
                                .zip(physical.iter().cycle())
                                .map(|(a, b)| {
                                    let signal = if *b <= 0.0031308 {
                                        12.92 * b
                                    } else {
                                        1.055 * b.powf(1.0 / 2.4) - 0.055
                                    };
                                    let expected = 2.0 * signal.clamp(0.0, 1.0);
                                    let expected = if integer_output {
                                        expected.clamp(0.0, 1.0)
                                    } else {
                                        expected
                                    };
                                    (a - expected).abs()
                                })
                                .fold(0_f32, f32::max);
                            assert!(error <= 0.01, "{label} nonlinear sRGB Add error {error}");
                        }
                        assert!(
                            native[1]
                                .0
                                .iter()
                                .zip(&primary[1].0)
                                .all(|(a, b)| (a.clamp(0.0, 1.0) - b).abs() <= 0.001),
                            "{label} source-control/native-addend versus primary composite"
                        );
                        for values in [&native[1].1, &primary[1].1[0]] {
                            assert!(values.iter().all(|v|v.is_finite()&&(*v*maximum).round()as u16==code),"{label} explicit Black Replace");
                        }
                    }
                }
            }
        }
    }
}
