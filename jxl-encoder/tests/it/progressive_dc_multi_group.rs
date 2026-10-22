// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Progressive DC on images wider or taller than 2048 pixels.
//!
//! `with_progressive_dc(1|2)` writes the DC as a separate LF frame at 1/8
//! resolution. Once the image exceeds 2048 pixels on either axis that LF
//! frame spans more than one 256-pixel group and takes the multi-group
//! modular path. That path used to split the DC planes with the 1/8
//! quantizer-hint shift still attached to each channel, so every group
//! carried a region 8x too small on each axis and no decoder could read the
//! frame (jxl-rs "section too short", djxl "failed to decode", jxl-oxide
//! "reference frame failed to render").
//!
//! Each cell here decodes fully through jxl-rs, djxl v0.12 and jxl-oxide,
//! asserts the LF frame really is multi-group, and bounds the decoded error
//! against the source so a frame that decodes to the wrong DC also fails.

use jxl_encoder::{LossyConfig, PixelLayout};

/// Real-image content at any size: the frymire fixture, mirror-tiled.
fn tiled_frymire(width: u32, height: u32) -> Vec<u8> {
    let source = image::load_from_memory(include_bytes!("../images/frymire-srgb.png"))
        .expect("decode real screenshot fixture")
        .to_rgb8();
    let (sw, sh) = source.dimensions();
    let fold = |v: u32, n: u32| {
        let period = 2 * n;
        let m = v % period;
        if m < n { m } else { period - 1 - m }
    };
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            out.extend_from_slice(&source.get_pixel(fold(x, sw), fold(y, sh)).0);
        }
    }
    out
}

fn mean_abs_error(source: &[u8], decoded_rgb: &[f32]) -> f64 {
    assert_eq!(source.len(), decoded_rgb.len());
    let sum: f64 = source
        .iter()
        .zip(decoded_rgb)
        .map(|(&s, &d)| (s as f64 / 255.0 - (d as f64).clamp(0.0, 1.0)).abs())
        .sum();
    sum / source.len() as f64
}

fn decode_jxl_rs_rgb(data: &[u8]) -> (u32, u32, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => init = fallback,
            Err(e) => panic!("jxl-rs header: {e:?}"),
        }
    };
    let (width, height) = decoder.basic_info().size;
    let num_extras = decoder.basic_info().extra_channels.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; num_extras],
    });
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info: {e:?}"),
        }
    };
    let mut output = Image::<f32>::new((width * 3, height)).expect("alloc rgb f32");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * 3, height),
            })
            .into_raw(),
    )];
    loop {
        match frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs frame decode: {e:?}"),
        }
    }
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        pixels.extend_from_slice(output.row(y));
    }
    (width as u32, height as u32, pixels)
}

fn decode_djxl_rgb(data: &[u8], label: &str) -> (u32, u32, Vec<f32>) {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/progressive-dc-validation");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{label}.jxl"));
    let output = dir.join(format!("{label}.png"));
    std::fs::write(&input, data).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(
        result.status.success(),
        "djxl {label}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let decoded = image::open(&output).expect("read djxl PNG").to_rgb8();
    let (w, h) = decoded.dimensions();
    let pixels = decoded.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
    (w, h, pixels)
}

fn decode_jxl_oxide_rgb(data: &[u8]) -> (u32, u32, Vec<f32>) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide header");
    let lf = image.frame(0).expect("LF frame").header();
    assert_eq!(lf.frame_type as u8, 1, "first frame must be the LF frame");
    assert!(
        lf.num_groups() > 1,
        "LF frame must span several groups, got {}",
        lf.num_groups()
    );
    let render = image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render: {e:?}"));
    let width = image.width();
    let height = image.height();
    let framebuffer = render.image_all_channels();
    assert_eq!(framebuffer.channels(), 3);
    (width, height, framebuffer.buf().to_vec())
}

#[test]
fn progressive_dc_above_one_lf_group_decodes_in_three_decoders() {
    for (w, h) in [(2049u32, 300u32), (2400, 2100)] {
        let source = tiled_frymire(w, h);
        let baseline = LossyConfig::new(1.0)
            .with_effort(5)
            .encode(&source, w, h, PixelLayout::Rgb8)
            .expect("encode without progressive DC");
        let (_, _, base_pixels) = decode_jxl_rs_rgb(&baseline);
        let base_error = mean_abs_error(&source, &base_pixels);
        for level in [1u8, 2] {
            let label = format!("progdc{level}-{w}x{h}");
            let bytes = LossyConfig::new(1.0)
                .with_effort(5)
                .with_progressive_dc(level)
                .encode(&source, w, h, PixelLayout::Rgb8)
                .expect("encode with progressive DC");
            for (name, (dw, dh, pixels)) in [
                ("jxl-rs", decode_jxl_rs_rgb(&bytes)),
                ("djxl", decode_djxl_rgb(&bytes, &label)),
                ("jxl-oxide", decode_jxl_oxide_rgb(&bytes)),
            ] {
                assert_eq!((dw, dh), (w, h), "{name} {label} dimensions");
                let error = mean_abs_error(&source, &pixels);
                eprintln!(
                    "{label} {name}: mean abs error {error:.5} (no LF frame {base_error:.5})"
                );
                assert!(
                    error < base_error * 1.5 + 0.002,
                    "{name} {label}: error {error:.5} vs {base_error:.5} without progressive DC"
                );
            }
        }
    }
}
