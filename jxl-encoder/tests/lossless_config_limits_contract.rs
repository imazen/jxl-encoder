//! Configured lossless limits follow requests and legacy entry points.
#![cfg(feature = "__expert")]
use jxl_encoder::{
    AnimationFrame, AnimationParams, EncodeError, ImageMetadata, Limits, LosslessConfig,
    PixelLayout,
};
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("lossless-config-limits-contract");
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

fn render(path: &std::path::Path, bytes: &[u8], dims: (usize, usize), expected: &[&[u8]]) {
    render_mode(path, bytes, dims, expected, true);
}

fn render_mode(
    path: &std::path::Path,
    bytes: &[u8],
    dims: (usize, usize),
    expected: &[&[u8]],
    exact: bool,
) {
    std::fs::create_dir_all(path).unwrap();
    std::fs::write(path.join("input.jxl"), bytes).unwrap();
    let primary = decoded_frames::decode_current_frames(bytes, "configured limits");
    assert_eq!(primary.len(), expected.len());
    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path.join("input.jxl"))
        .arg(path.join("native.pfm"))
        .args([
            "--num_threads=1",
            "--output_frames",
            "--color_space=RGB_D65_SRG_Rel_SRG",
        ])
        .output()
        .unwrap();
    std::fs::write(path.join("native.log"), &native.stderr).unwrap();
    assert!(native.status.success());
    for (index, (frame, expected)) in primary.iter().zip(expected).enumerate() {
        let name = if primary.len() == 1 {
            "native.pfm".into()
        } else {
            format!("native.pfm-{index}.pfm")
        };
        let native = pfm::read_pfm_samples(&path.join(name), dims);
        assert!(frame.1.is_empty());
        for decoded in [&frame.0, &native] {
            assert_eq!(decoded.len(), expected.len());
            assert!(decoded.iter().all(|v| v.is_finite()));
            let codes: Vec<u8> = decoded.iter().map(|v| (v * 255.0).round() as u8).collect();
            if exact {
                assert_eq!(&codes, expected, "{} frame{index}", path.display());
            } else {
                assert!(
                    decoded
                        .iter()
                        .zip(&frame.0)
                        .all(|(a, b)| (a - b).abs() <= 0.001),
                    "{} frame{index}: decoder agreement",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn inherited_lossless_limits_reach_still_and_animation() {
    let root = directory();
    let image = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            let data = image::imageops::crop_imm(&image, 333, 135, width as u32, height as u32)
                .to_image()
                .into_raw();
            let second: Vec<u8> = data.iter().map(|v| 255 - v).collect();
            let frames = [
                AnimationFrame::new(&data, 1),
                AnimationFrame::new(&second, 1),
            ];
            let base = LosslessConfig::new()
                .with_effort(1)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let tight = Limits::default().with_max_pixels((width * height - 1) as u64);
            let loose = Limits::default()
                .with_max_pixels((width * height) as u64)
                .with_max_quant_loop_iters(0);
            let config = base.clone().with_limits(&tight);
            let (w, h) = (width as u32, height as u32);
            let padded: Vec<u8> = data
                .chunks_exact(width * 3)
                .flat_map(|row| row.iter().copied().chain([0xff]))
                .collect();
            for entry in [
                "legacy",
                "request",
                "append",
                "writer",
                "strided",
                "animation",
                "legacy-animation",
                "metadata-animation",
            ] {
                let out = root.join(format!("{policy}-{width}x{height}-{entry}"));
                std::fs::create_dir(&out).unwrap();
                let mut sink = b"prior destination".to_vec();
                let prefix = sink.len();
                let result = match entry {
                    "legacy" => config.encode(&data, w, h, PixelLayout::Rgb8),
                    "append" => config
                        .encode_into(&data, w, h, PixelLayout::Rgb8, &mut sink)
                        .map(|()| sink[prefix..].to_vec()),
                    "writer" => config
                        .encode_request(w, h, PixelLayout::Rgb8)
                        .encode_to(&data, &mut sink)
                        .map(|_| sink[prefix..].to_vec()),
                    "strided" => config
                        .encode_request(w, h, PixelLayout::Rgb8)
                        .with_row_stride(width * 3 + 1)
                        .encode(&padded),
                    "animation" => config
                        .encode_request(w, h, PixelLayout::Rgb8)
                        .encode_animation(&AnimationParams::default(), &frames),
                    "metadata-animation" => config.encode_animation_with_metadata(
                        w,
                        h,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                        &ImageMetadata::new().with_xmp(b"<x/>"),
                    ),
                    "legacy-animation" => config.encode_animation(
                        w,
                        h,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                    ),
                    _ => config.encode_request(w, h, PixelLayout::Rgb8).encode(&data),
                };
                match result {
                    Err(error) => {
                        std::fs::write(out.join("error.txt"), error.to_string()).unwrap();
                        assert!(
                            matches!(error.error(), EncodeError::LimitExceeded { .. }),
                            "{error}"
                        );
                        assert_eq!(sink, b"prior destination");
                    }
                    Ok(bytes) => {
                        let expected: Vec<&[u8]> = if entry.contains("animation") {
                            vec![&data, &second]
                        } else {
                            vec![&data]
                        };
                        render(&out, &bytes, (width, height), &expected);
                        failures.push(format!(
                            "{policy}-{width}x{height}-{entry}: configured pixel cap was ignored"
                        ));
                    }
                }
            }
            // Explicit per-request/legacy limits retain their existing precedence.
            let override_bytes = config
                .encode_request(w, h, PixelLayout::Rgb8)
                .with_limits(&loose)
                .encode(&data)
                .unwrap();
            let plain = base.encode(&data, w, h, PixelLayout::Rgb8).unwrap();
            assert_eq!(override_bytes, plain);
            render(
                &root.join(format!("{policy}-{width}x{height}-override")),
                &override_bytes,
                (width, height),
                &[&data],
            );
            let animation = config
                .encode_animation_with_limits(
                    w,
                    h,
                    PixelLayout::Rgb8,
                    &AnimationParams::default(),
                    &frames,
                    &loose,
                )
                .unwrap();
            render(
                &root.join(format!("{policy}-{width}x{height}-animation-override")),
                &animation,
                (width, height),
                &[&data, &second],
            );
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[test]
fn animation_explicit_dimension_limits_reach_both_writers() {
    let root = directory();
    let image = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for lossless in [false, true] {
            for (width, height) in [(32_usize, 17_usize), (259, 17)] {
                let data = image::imageops::crop_imm(&image, 333, 135, width as u32, height as u32)
                    .to_image()
                    .into_raw();
                let frames = [AnimationFrame::new(&data, 1), AnimationFrame::new(&data, 1)];
                let (w, h) = (width as u32, height as u32);
                let lc = LosslessConfig::new()
                    .with_effort(1)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let vc = jxl_encoder::LossyConfig::new(0.1)
                    .with_effort(1)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                for (field, limits) in [
                    (
                        "width",
                        Limits::default().with_max_width((width - 1) as u64),
                    ),
                    (
                        "height",
                        Limits::default().with_max_height((height - 1) as u64),
                    ),
                    (
                        "pixels",
                        Limits::default().with_max_pixels((width * height - 1) as u64),
                    ),
                ] {
                    for legacy in [false, true] {
                        let out = root.join(format!(
                            "{policy}-{width}x{height}-lossless{lossless}-{field}-legacy{legacy}"
                        ));
                        std::fs::create_dir(&out).unwrap();
                        let result = match (lossless, legacy) {
                            (true, true) => lc.encode_animation_with_limits(
                                w,
                                h,
                                PixelLayout::Rgb8,
                                &AnimationParams::default(),
                                &frames,
                                &limits,
                            ),
                            (false, true) => vc.encode_animation_with_limits(
                                w,
                                h,
                                PixelLayout::Rgb8,
                                &AnimationParams::default(),
                                &frames,
                                &limits,
                            ),
                            (true, false) => lc
                                .encode_request(w, h, PixelLayout::Rgb8)
                                .with_limits(&limits)
                                .encode_animation(&AnimationParams::default(), &frames),
                            (false, false) => vc
                                .encode_request(w, h, PixelLayout::Rgb8)
                                .with_limits(&limits)
                                .encode_animation(&AnimationParams::default(), &frames),
                        };
                        match result {
                            Err(error) => {
                                std::fs::write(out.join("error.txt"), error.to_string()).unwrap();
                                assert!(
                                    matches!(error.error(), EncodeError::LimitExceeded { .. }),
                                    "{error}"
                                );
                            }
                            Ok(bytes) => {
                                render_mode(
                                    &out,
                                    &bytes,
                                    (width, height),
                                    &[&data, &data],
                                    lossless,
                                );
                                failures.push(format!(
                                    "{}: explicit animation dimension cap was ignored",
                                    out.display()
                                ));
                            }
                        }
                    }
                }
                let limits = Limits::default()
                    .with_max_width(width as u64)
                    .with_max_height(height as u64)
                    .with_max_pixels((width * height) as u64);
                let bytes = if lossless {
                    lc.encode_request(w, h, PixelLayout::Rgb8)
                        .with_limits(&limits)
                        .encode_animation(&AnimationParams::default(), &frames)
                } else {
                    vc.encode_request(w, h, PixelLayout::Rgb8)
                        .with_limits(&limits)
                        .encode_animation(&AnimationParams::default(), &frames)
                }
                .unwrap();
                let unlimited = if lossless {
                    lc.encode_animation(
                        w,
                        h,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                    )
                } else {
                    vc.encode_animation(
                        w,
                        h,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                    )
                }
                .unwrap();
                assert_eq!(bytes, unlimited, "valid dimension caps must preserve bytes");
                render_mode(
                    &root.join(format!(
                        "{policy}-{width}x{height}-lossless{lossless}-boundary"
                    )),
                    &bytes,
                    (width, height),
                    &[&data, &data],
                    lossless,
                );
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[cfg(feature = "butteraugli-loop")]
#[test]
fn animation_explicit_loop_limits_run_before_frame_work() {
    let root = directory();
    let image = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let data = image::imageops::crop_imm(&image, 333, 135, 32, 17)
        .to_image()
        .into_raw();
    let frames = [AnimationFrame::new(&data, 1), AnimationFrame::new(&data, 1)];
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        let config = jxl_encoder::LossyConfig::new(0.1)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode)
            .with_butteraugli_iters(2)
            .with_auto_delta_frames(false)
            .with_patches(false);
        for cap in [0, 1, 2] {
            let out = root.join(format!("{policy}-cap{cap}"));
            std::fs::create_dir(&out).unwrap();
            let trace = jxl_encoder::LoopTrace::new();
            let limits = Limits::default().with_max_quant_loop_iters(cap);
            let result = config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .with_limits(&limits)
                .with_loop_trace(&trace)
                .encode_animation(&AnimationParams::default(), &frames);
            let records = trace.take();
            std::fs::write(
                out.join("records.txt"),
                format!("configured=2\nlimit={cap}\nrecords={}\n", records.len()),
            )
            .unwrap();
            match result {
                Err(error) => {
                    std::fs::write(out.join("error.txt"), error.to_string()).unwrap();
                    assert!(
                        cap < 2 && matches!(error.error(), EncodeError::LimitExceeded { .. }),
                        "{policy} cap{cap}: {error}"
                    );
                    assert!(records.is_empty(), "rejection must precede loop/frame work");
                }
                Ok(bytes) => {
                    render_mode(&out, &bytes, (32, 17), &[&data, &data], false);
                    assert!(
                        !records.is_empty(),
                        "explicit loop must execute at the admitted boundary"
                    );
                    if cap < 2 {
                        failures.push(format!(
                            "{policy} cap{cap}: explicit animation loop cap was ignored"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[test]
fn animation_invalid_display_extents_are_rejected_before_frame_data() {
    // No buffer exists for these dimensions. Shared admission must refuse
    // before either animation writer attempts to inspect or allocate frames.
    let limits = Limits::default();
    let frames = [AnimationFrame::new(&[], 1)];
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(0, 17), (32, 0), (u32::MAX, 1)] {
            let lossless = LosslessConfig::new()
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let lossy = jxl_encoder::LossyConfig::new(1.0)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for result in [
                lossless
                    .encode_request(width, height, PixelLayout::Rgb8)
                    .with_limits(&limits)
                    .encode_animation(&AnimationParams::default(), &frames),
                lossy
                    .encode_request(width, height, PixelLayout::Rgb8)
                    .with_limits(&limits)
                    .encode_animation(&AnimationParams::default(), &frames),
            ] {
                let error = result.unwrap_err();
                assert!(
                    matches!(error.error(), EncodeError::InvalidInput { .. }),
                    "{policy}: {error}"
                );
                assert!(
                    error.to_string().contains("display dimensions"),
                    "{policy}: {error}"
                );
            }
        }
    }
}
