//! Independent composition of native-decoded CMYK crop layers.
#![cfg(all(feature = "__expert", feature = "std"))]
use jxl_encoder::{
    AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout,
    ProgressiveMode,
};
use std::{io::Write, path::Path};
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
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
fn native_layers(
    out: &Path,
    bytes: &[u8],
    canvas: (usize, usize),
    oracle: &std::ffi::OsStr,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    std::fs::create_dir(out).unwrap();
    let input = out.join("animation.jxl");
    std::fs::write(&input, bytes).unwrap();
    let native = std::process::Command::new(oracle)
        .arg(&input)
        .arg(out.join("layer"))
        .arg("linear-layers")
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
