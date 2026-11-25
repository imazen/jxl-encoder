//! Progressive preservation after required CMYK profile conversion.
#![cfg(all(
    feature = "__expert",
    any(
        feature = "std",
        all(feature = "__internals", feature = "__pre_quantized")
    )
))]
#[cfg(feature = "std")]
use jxl_encoder::{AnimationFrame, AnimationParams, BlendMode, ImageMetadata, PixelLayout};
use jxl_encoder::{LossyConfig, ProgressiveMode};
#[cfg(feature = "std")]
use std::io::Write;
use std::path::Path;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

type Planes = Vec<(Vec<u32>, Vec<u32>)>;
const SWATCHES: [[u8; 4]; 6] = [
    [255, 255, 255, 255],
    [0, 255, 255, 255],
    [255, 0, 255, 255],
    [255, 255, 0, 255],
    [255, 255, 255, 0],
    [64, 128, 192, 224],
];
fn swatch(x: usize, y: usize, frame: usize, solid: bool) -> usize {
    if solid {
        return if frame == 0 { 5 } else { 0 };
    }
    (x / 8 + y / 8 + frame) % SWATCHES.len()
}
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("progressive-cmyk-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn schedule(bytes: &[u8], frames: usize, progressive: ProgressiveMode) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(bytes))
        .unwrap();
    assert_eq!(image.num_loaded_frames(), frames);
    let wanted = match progressive {
        ProgressiveMode::Single => 1,
        ProgressiveMode::QuantizedAcFullAc => 2,
        ProgressiveMode::DcVlfLfAc => 3,
    };
    for frame in 0..frames {
        assert_eq!(
            image.frame(frame).unwrap().header().passes.num_passes,
            wanted
        );
    }
}
fn render(
    path: &Path,
    bytes: &[u8],
    dims: (usize, usize),
    frames: usize,
    flags: (bool, bool),
    oracle: &std::ffi::OsStr,
    physical: &[[f32; 3]],
) -> [Planes; 2] {
    let (wide, solid) = flags;
    std::fs::create_dir(path).unwrap();
    std::fs::write(path.join("input.jxl"), bytes).unwrap();
    let primary = decoded_frames::decode_current_frames(bytes, &path.display().to_string());
    assert_eq!(primary.len(), frames);
    let native = std::process::Command::new(oracle)
        .arg(path.join("input.jxl"))
        .arg(path.join("native"))
        .arg("linear")
        .output()
        .unwrap();
    std::fs::write(path.join("headers.tsv"), native.stdout).unwrap();
    std::fs::write(path.join("native.log"), native.stderr).unwrap();
    assert!(native.status.success(), "{}", path.display());
    let maximum = if wide { 65535.0 } else { 255.0 };
    let mut primary_bits = Vec::new();
    let mut native_bits = Vec::new();
    for (frame, (color, extras)) in primary.into_iter().enumerate() {
        assert!(matches!(extras.len(), 1 | 2), "Black with optional alpha");
        let black = extras.last().unwrap();
        let native_color = pfm::read_pfm_samples(&path.join(format!("native-{frame}.pfm")), dims);
        let native_black =
            pfm::read_pfm_samples(&path.join(format!("native-{frame}-black.pfm")), dims);
        assert_eq!(color.len(), dims.0 * dims.1 * 3);
        assert_eq!(native_color.len(), color.len());
        assert_eq!(black.len(), dims.0 * dims.1);
        assert_eq!(native_black.len(), black.len());
        for (name, values) in [("color", &color), ("black", black)] {
            std::fs::write(
                path.join(format!("primary-{frame}-{name}.f32le")),
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
        let mut errors = Vec::new();
        for values in [&color, &native_color] {
            assert!(values.iter().all(|v| v.is_finite()));
            let error = values
                .as_chunks::<3>()
                .0
                .iter()
                .enumerate()
                .map(|(i, pixel)| {
                    let expected = physical[swatch(i % dims.0, i / dims.0, frame, solid)];
                    pixel
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0_f32, f32::max)
                })
                .fold(0_f32, f32::max);
            errors.push(error);
        }
        std::fs::write(
            path.join(format!("physical-{frame}.json")),
            serde_json::to_vec(&errors).unwrap(),
        )
        .unwrap();
        assert!(
            !solid || errors.iter().all(|v| *v <= 0.005),
            "physical CMYK {} frame{frame} primary/native: {errors:?}",
            path.display()
        );
        // The established physical bound covers uniform swatches. Sharp swatch
        // boundaries retain error diagnostics; progressive preservation below
        // requires each decoder's exact Single result for every sample.
        assert!(
            color
                .iter()
                .zip(&native_color)
                .all(|(a, b)| (a - b).abs() <= 0.001)
        );
        for values in [black, &native_black] {
            for (i, &value) in values.iter().enumerate() {
                let code = u16::from(SWATCHES[swatch(i % dims.0, i / dims.0, frame, solid)][3])
                    * if wide { 257 } else { 1 };
                assert!(value.is_finite());
                assert_eq!((value * maximum).round() as u16, code, "exact Black");
            }
        }
        let mut native_extras = Vec::new();
        if extras.len() == 2 {
            let alpha =
                pfm::read_pfm_samples(&path.join(format!("native-{frame}-alpha.pfm")), dims);
            for values in [&extras[0], &alpha] {
                assert_eq!(values.len(), dims.0 * dims.1);
                assert!(values.iter().all(|v| v.is_finite()
                    && (v * maximum).round() as u16 == 137 * if wide { 257 } else { 1 }));
            }
            native_extras.extend(alpha.into_iter().map(f32::to_bits));
        }
        native_extras.extend(native_black.into_iter().map(f32::to_bits));
        primary_bits.push((
            color.into_iter().map(f32::to_bits).collect(),
            extras.iter().flatten().map(|v| v.to_bits()).collect(),
        ));
        native_bits.push((
            native_color.into_iter().map(f32::to_bits).collect(),
            native_extras,
        ));
    }
    [primary_bits, native_bits]
}
fn same_final(actual: &[Planes; 2], expected: &[Planes; 2], label: &str) {
    for (decoder, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.len(), expected.len(), "{label} frames");
        for (frame, ((ac, ab), (ec, eb))) in actual.iter().zip(expected).enumerate() {
            for (channel, a, e) in [("color", ac, ec), ("Black", ab, eb)] {
                assert_eq!(a.len(), e.len(), "{label} {channel} length");
                for (sample, (a, e)) in a.iter().zip(e).enumerate() {
                    assert_eq!(
                        a, e,
                        "{label} decoder{decoder} frame{frame} {channel} sample{sample}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "std")]
fn check(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let out = directory();
    let profile: Vec<u8> = include_str!("fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect();
    let physical: Vec<[f32; 3]> = include_str!("fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    assert_eq!(physical.len(), SWATCHES.len());
    std::fs::write(out.join("profile.icc"), &profile).unwrap();
    std::fs::write(
        out.join("physical.hex"),
        include_str!("fixtures/cmyk_lcms_swatches.hex"),
    )
    .unwrap();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE")
        .expect("caller must provide the pinned native CMYK observer");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl CMYK raw-frame oracle 13000"
    );
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    for (policy, strategy, mode) in policies {
        for (w, h) in [(1_usize, 17_usize), (32, 17), (259, 17), (17, 259)] {
            for solid in [true, false] {
                for wide in [false, true] {
                    let layout = if wide {
                        PixelLayout::Cmyk16
                    } else {
                        PixelLayout::Cmyk8
                    };
                    let source: Vec<Vec<u8>> = (0..2)
                        .map(|frame| {
                            (0..w * h)
                                .flat_map(|i| {
                                    SWATCHES[swatch(i % w, i / w, frame, solid)]
                                        .into_iter()
                                        .flat_map(|code| {
                                            if wide {
                                                (u16::from(code) * 257).to_ne_bytes().to_vec()
                                            } else {
                                                vec![code]
                                            }
                                        })
                                })
                                .collect()
                        })
                        .collect();
                    for (frame, pixels) in source.iter().enumerate() {
                        std::fs::write(
                            out.join(format!(
                                "source-{policy}-{w}x{h}-solid{solid}-wide{wide}-{frame}.bin"
                            )),
                            pixels,
                        )
                        .unwrap();
                    }
                    let mut baseline = [None, None];
                    for progressive in [
                        ProgressiveMode::Single,
                        ProgressiveMode::QuantizedAcFullAc,
                        ProgressiveMode::DcVlfLfAc,
                    ] {
                        let label =
                            format!("{policy}-{w}x{h}-solid{solid}-wide{wide}-{progressive:?}");
                        let config = LossyConfig::new(0.01)
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_force_strategy(Some(3))
                            .with_progressive(progressive)
                            .with_auto_delta_frames(false);
                        let request = || {
                            config
                                .encode_request(w as u32, h as u32, layout)
                                .with_metadata(&metadata)
                        };
                        let bytes = request().encode(&source[0]).unwrap();
                        schedule(&bytes, 1, progressive);
                        let stride = w * layout.bytes_per_pixel() + 11;
                        let mut padded = vec![0xa5; stride * h];
                        for (target, row) in padded
                            .chunks_exact_mut(stride)
                            .zip(source[0].chunks_exact(w * layout.bytes_per_pixel()))
                        {
                            target[..row.len()].copy_from_slice(row);
                        }
                        assert_eq!(
                            request().with_row_stride(stride).encode(&padded).unwrap(),
                            bytes
                        );
                        let mut writer = Vec::new();
                        request().encode_to(&source[0], &mut writer).unwrap();
                        assert_eq!(writer, bytes);
                        let actual = render(
                            &out.join(format!("{label}-still")),
                            &bytes,
                            (w, h),
                            1,
                            (wide, solid),
                            &oracle,
                            &physical,
                        );
                        if let Some(expected) = &baseline[0] {
                            same_final(&actual, expected, &format!("final still pixels {label}"));
                        } else {
                            baseline[0] = Some(actual);
                        }
                        let frames = [
                            AnimationFrame::new(&source[0], 10).with_blend_mode(BlendMode::Replace),
                            AnimationFrame::new(&source[1], 20).with_blend_mode(BlendMode::Replace),
                        ];
                        let params = AnimationParams::default();
                        let bytes = request().encode_animation(&params, &frames).unwrap();
                        schedule(&bytes, 2, progressive);
                        assert_eq!(
                            config
                                .encode_animation_with_metadata(
                                    w as u32, h as u32, layout, &params, &frames, &metadata
                                )
                                .unwrap(),
                            bytes
                        );
                        let actual = render(
                            &out.join(format!("{label}-animation")),
                            &bytes,
                            (w, h),
                            2,
                            (wide, solid),
                            &oracle,
                            &physical,
                        );
                        if let Some(expected) = &baseline[1] {
                            same_final(
                                &actual,
                                expected,
                                &format!("final displayed pixels {label}"),
                            );
                        } else {
                            baseline[1] = Some(actual);
                        }
                        writeln!(
                        journal,
                        "{label}\tuniform physical bound or sharp-edge diagnostics; exact Black; same final bits as Single"
                    )
                    .unwrap();
                        journal.flush().unwrap();
                    }
                }
            }
        }
    }
}
#[cfg(feature = "std")]
#[test]
fn strict_cmyk_progressive_preserves_converted_color_and_black() {
    check(vec![encoder_policies::policies().remove(0)]);
}
#[cfg(feature = "std")]
#[test]
fn all_offered_cmyk_progressive_preserves_converted_color_and_black() {
    check(encoder_policies::policies());
}

#[cfg(all(feature = "__internals", feature = "__pre_quantized"))]
#[path = "progressive_cmyk_contract/prepared.rs"]
mod prepared;

#[cfg(feature = "std")]
#[path = "progressive_cmyk_contract/separated.rs"]
mod separated;
