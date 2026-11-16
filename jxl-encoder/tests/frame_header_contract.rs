//! Production regressions for issue 144, Part 1 F.2.
#![cfg(feature = "__expert")]

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{AnimationFrame, AnimationParams, LosslessConfig, LossyConfig, PixelLayout};
use std::{path::Path, process::Command};

#[path = "frame_header_contract/matrix.rs"]
mod frame_header_matrix;

fn run_dir(label: &str) -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("frame-header-contract");
    for i in 0.. {
        let path = root.join(format!("{label}-{i}"));
        match std::fs::create_dir(&path) {
            Ok(()) => return path,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => panic!("{}: {e}", path.display()),
        }
    }
    unreachable!()
}

fn rust_frames(encoded: &[u8], channels: usize) -> Vec<Vec<f32>> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    let mut input = encoded;
    let init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let ProcessingResult::Complete {
        result: mut decoder,
    } = init.process(&mut input).unwrap()
    else {
        panic!("incomplete image header")
    };
    let (w, h) = decoder.basic_info().size;
    let format = JxlPixelFormat {
        color_type: if channels == 4 {
            JxlColorType::Rgba
        } else {
            JxlColorType::Rgb
        },
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; usize::from(channels == 4)],
    };
    decoder.set_pixel_format(format);
    let mut output = Vec::new();
    loop {
        let ProcessingResult::Complete { result: frame } = decoder.process(&mut input).unwrap()
        else {
            panic!("incomplete frame header")
        };
        let mut pixels = vec![0_u8; w * h * channels * 4];
        let mut buffers = [JxlOutputBuffer::new(&mut pixels, h, w * channels * 4)];
        let ProcessingResult::Complete { result: next } =
            frame.process(&mut input, &mut buffers).unwrap()
        else {
            panic!("incomplete frame pixels")
        };
        decoder = next;
        output.push(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect(),
        );
        if !decoder.has_more_frames() {
            break;
        }
    }
    output
}

fn render_reference(path: &Path) {
    let output = Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg(path.with_extension("png"))
        .args(["--bits_per_sample=16", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(path.with_extension("djxl.log"), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "{}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn reference_pixels(path: &Path) -> Vec<Vec<u16>> {
    let stem = format!("{}-frame", path.file_stem().unwrap().to_str().unwrap());
    let output = Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg(path.with_file_name(format!("{stem}.pam")))
        .args(["--bits_per_sample=16", "--num_threads=1", "--output_frames"])
        .output()
        .unwrap();
    std::fs::write(path.with_extension("djxl.log"), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "{}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let mut frames: Vec<_> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|entry| {
            entry.extension().is_some_and(|e| e == "pam")
                && entry
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with(&stem)
        })
        .collect();
    frames.sort();
    assert!(
        !frames.is_empty(),
        "djxl produced no frames for {}",
        path.display()
    );
    frames
        .into_iter()
        .map(|p| image::open(p).unwrap().to_rgba16().into_raw())
        .collect()
}

#[test]
fn reference_only_rgba_then_displayed_frame() {
    let dir = run_dir("reference-only");
    let reference = [23, 43, 63, 127].repeat(64 * 32);
    let displayed = [37, 101, 219, 193].repeat(64 * 32);
    let frames = [
        AnimationFrame::new(&reference, 0)
            .with_reference_only(true)
            .with_save_as_reference(2),
        AnimationFrame::new(&displayed, 10),
    ];
    let mut failures = Vec::new();
    for lossy in [false, true] {
        let encoded = if lossy {
            LossyConfig::new(1.0)
                .with_effort(3)
                .with_strategy(EncoderStrategy::Libjxl)
                .encode_animation(
                    64,
                    32,
                    PixelLayout::Rgba8,
                    &AnimationParams::default(),
                    &frames,
                )
        } else {
            LosslessConfig::new()
                .with_effort(3)
                .with_strategy(EncoderStrategy::Libjxl)
                .encode_animation(
                    64,
                    32,
                    PixelLayout::Rgba8,
                    &AnimationParams::default(),
                    &frames,
                )
        }
        .unwrap();
        let path = dir.join(format!("lossy-{lossy}.jxl"));
        std::fs::write(&path, &encoded).unwrap();
        for (name, result) in [
            ("djxl", std::panic::catch_unwind(|| render_reference(&path))),
            (
                "jxl-rs",
                std::panic::catch_unwind(|| {
                    let output = rust_frames(&encoded, 4);
                    assert_eq!(output.len(), 1);
                    for (i, &v) in output[0].iter().enumerate() {
                        assert!(v.is_finite());
                        if !lossy || i % 4 == 3 {
                            assert_eq!((v * 255.0).round() as u8, displayed[i], "sample {i}");
                        }
                    }
                }),
            ),
        ] {
            if result.is_err() {
                failures.push(format!("lossy={lossy} {name}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{failures:?}; artifacts {}",
        dir.display()
    );
}

#[test]
fn identical_eight_square_lossy_frames() {
    let dir = run_dir("covering-crop");
    let pixels = [37, 101, 219].repeat(8 * 8);
    let frames = [
        AnimationFrame::new(&pixels, 10),
        AnimationFrame::new(&pixels, 20),
    ];
    let encoded = LossyConfig::new(1.0)
        .with_effort(3)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_auto_delta_frames(false)
        .encode_animation(
            8,
            8,
            PixelLayout::Rgb8,
            &AnimationParams::default(),
            &frames,
        )
        .unwrap();
    let path = dir.join("identical.jxl");
    std::fs::write(&path, &encoded).unwrap();
    let reference = std::panic::catch_unwind(|| render_reference(&path));
    let rust = std::panic::catch_unwind(|| {
        let output = rust_frames(&encoded, 3);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], output[1]);
    });
    assert!(
        reference.is_ok() && rust.is_ok(),
        "artifacts {}",
        dir.display()
    );
}
