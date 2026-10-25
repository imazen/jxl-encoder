// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy encodes code alpha losslessly, so 16-bit alpha must come back with
//! all 16 bits. It used to be quantized to 8 bits on the way in (found by the
//! conformance sweep: 65534 decoded as 65535, 3 as 0).

use jxl_encoder::{LossyConfig, PixelLayout};

/// Full jxl-rs decode; returns the last extra channel (alpha) scaled to
/// 0..=65535.
fn jxl_rs_alpha16(data: &[u8], label: &str) -> Vec<u16> {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => init = fallback,
            Err(e) => panic!("jxl-rs header {label}: {e:?}"),
        }
    };
    let (w, h) = decoder.basic_info().size;
    let format = decoder.current_pixel_format().clone();
    let nc = format.color_type.samples_per_pixel();
    let ne = format.extra_channel_format.len();
    assert!(ne >= 1, "{label}: no extra channel");
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); ne],
    });
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info {label}: {e:?}"),
        }
    };
    let mut color = Image::<f32>::new((w * nc, h)).unwrap();
    let mut extra: Vec<Image<f32>> = (0..ne)
        .map(|_| Image::<f32>::new((w, h)).unwrap())
        .collect();
    let mut bufs = vec![JxlOutputBuffer::from_image_rect_mut(
        color
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (w * nc, h),
            })
            .into_raw(),
    )];
    for e in &mut extra {
        bufs.push(JxlOutputBuffer::from_image_rect_mut(
            e.get_rect_mut(Rect {
                origin: (0, 0),
                size: (w, h),
            })
            .into_raw(),
        ));
    }
    loop {
        match frame.process(&mut input, &mut bufs) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs decode {label}: {e:?}"),
        }
    }
    drop(bufs);
    let alpha = &extra[ne - 1];
    (0..h)
        .flat_map(|y| {
            alpha.row(y)[..w]
                .iter()
                .map(|v| (v * 65535.0).round() as u16)
        })
        .collect()
}

/// djxl v0.12 to a 16-bit PNG; returns its alpha channel.
fn djxl_alpha16(data: &[u8], label: &str, gray: bool) -> Vec<u16> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/lossy-alpha16");
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
    let img = image::open(&output).expect("read djxl PNG");
    if gray {
        img.to_luma_alpha16().pixels().map(|p| p.0[1]).collect()
    } else {
        img.to_rgba16().pixels().map(|p| p.0[3]).collect()
    }
}

#[test]
fn lossy_16bit_alpha_keeps_all_bits() {
    for (w, h) in [(64usize, 32usize), (300, 260)] {
        let alpha: Vec<u16> = (0..w * h)
            .map(|i| ((i * 977 + i / w * 31) % 65536) as u16)
            .collect();
        for (layout, ch, gray) in [
            (PixelLayout::Rgba16, 4usize, false),
            (PixelLayout::GrayAlpha16, 2, true),
        ] {
            let mut px = Vec::with_capacity(w * h * ch * 2);
            for (i, a) in alpha.iter().enumerate() {
                for c in 0..ch - 1 {
                    let v = ((i % w) * 211 + (i / w) * 97 + c * 4099) % 65536;
                    px.extend_from_slice(&(v as u16).to_ne_bytes());
                }
                px.extend_from_slice(&a.to_ne_bytes());
            }
            for effort in [3u8, 7] {
                let label = format!("{layout:?}-{w}x{h}-e{effort}");
                let bytes = LossyConfig::new(1.0)
                    .with_effort(effort)
                    .encode(&px, w as u32, h as u32, layout)
                    .expect("lossy 16-bit alpha encode");
                assert_eq!(jxl_rs_alpha16(&bytes, &label), alpha, "jxl-rs {label}");
                assert_eq!(djxl_alpha16(&bytes, &label, gray), alpha, "djxl {label}");
            }
        }
    }
}
