// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-phase3-B1 — multi-decoder roundtrip on bitstreams produced by the
//! GPU butteraugli backend.
//!
//! Both jxl-rs (primary decoder) and jxl-oxide must decode the bitstream
//! cleanly. The bitstream is allowed to differ by tiny amounts from the
//! CPU-backend bitstream (the GPU's sRGB-u8 round-trip causes the
//! butteraugli score to drift by 0.02-0.05%, which moves the buttloop's
//! quant_field convergence to slightly different values) but the result
//! must remain a valid JXL stream that both decoders can render.
//!
//! Requires the `gpu-butteraugli` cargo feature AND CUDA at
//! `/usr/local/cuda`. The test is `#[ignore]` so a vanilla
//! `cargo test --features gpu-butteraugli` skips it; run via
//!   cargo test --release --features 'gpu-butteraugli butteraugli-loop parallel' \
//!     --test w44_phase3_b1_gpu_buttloop_roundtrip -- --ignored --nocapture

#![cfg(feature = "gpu-butteraugli")]

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::io::Cursor;

/// Tiny synthetic image — checkerboard with a soft gradient. Big enough to
/// fire the buttloop (e8+), small enough to keep the test fast.
fn synth_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let gx = (x as f32) / (w as f32);
            let gy = (y as f32) / (h as f32);
            let checker = ((x / 16) + (y / 16)) & 1;
            let base = (gx * 200.0 + gy * 55.0).clamp(0.0, 255.0) as u8;
            let v = if checker == 1 {
                base
            } else {
                base.saturating_sub(80)
            };
            out.push(v);
            out.push((v as u32 * 7 / 10) as u8);
            out.push((255 - v) / 2);
        }
    }
    out
}

fn decode_oxide(bytes: &[u8]) -> Result<(usize, usize), String> {
    let reader = Cursor::new(bytes);
    let img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .map_err(|e| format!("oxide read: {e}"))?;
    let r = img
        .render_frame(0)
        .map_err(|e| format!("oxide render: {e}"))?;
    let fb = r.image_all_channels();
    Ok((fb.width(), fb.height()))
}

/// Drive jxl-rs's streaming API enough to confirm the bitstream parses + a
/// frame renders. Pattern lifted from `buttloop_recon_parity.rs`.
fn decode_jxl_rs(bytes: &[u8]) -> Result<(usize, usize), String> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = bytes;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("jxl-rs: unexpected end of input during header".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs header: {e:?}")),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let channels = 3;

    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("jxl-rs: unexpected end of input before frame".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame info: {e:?}")),
        }
    };

    let mut output_image = Image::<f32>::new((width * channels, height))
        .map_err(|e| format!("jxl-rs alloc: {e:?}"))?;
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("jxl-rs: unexpected end of input during frame decode".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame: {e:?}")),
        }
    }
    Ok((width, height))
}

#[test]
#[ignore = "requires CUDA at /usr/local/cuda; run via cargo test --release --features 'gpu-butteraugli butteraugli-loop parallel' -- --ignored"]
fn gpu_backend_roundtrip_via_oxide_smoke() {
    // 256×256 is the smallest size that's worth running through the GPU
    // backend (smaller would test the construct-but-bypass path).
    let rgb = synth_rgb(256, 256);
    let cfg = LossyConfig::new(2.0)
        .with_effort(8) // e8 fires the buttloop
        .with_gpu_butteraugli(true)
        .with_threads(1);
    let bytes = cfg
        .encode(&rgb, 256, 256, PixelLayout::Rgb8)
        .expect("encode failed under GPU butteraugli");
    assert!(
        bytes.len() > 100,
        "encoded output too small: {}",
        bytes.len()
    );
    let (w, h) = decode_oxide(&bytes).expect("jxl-oxide failed to decode GPU-backend output");
    assert_eq!((w, h), (256, 256));
}

#[test]
#[ignore = "requires CUDA at /usr/local/cuda; run via cargo test --release --features 'gpu-butteraugli butteraugli-loop parallel' -- --ignored"]
fn gpu_backend_roundtrip_via_jxl_rs_smoke() {
    let rgb = synth_rgb(256, 256);
    let cfg = LossyConfig::new(2.0)
        .with_effort(8)
        .with_gpu_butteraugli(true)
        .with_threads(1);
    let bytes = cfg
        .encode(&rgb, 256, 256, PixelLayout::Rgb8)
        .expect("encode failed under GPU butteraugli");
    let (w, h) = decode_jxl_rs(&bytes).expect("jxl-rs failed to decode GPU-backend output");
    assert_eq!((w, h), (256, 256));
}

#[test]
#[ignore = "requires CUDA at /usr/local/cuda + corpus image"]
fn gpu_backend_roundtrip_terminal_e8_d4_via_oxide() {
    // Real screenshot from the corpus. Skip silently if the corpus
    // isn't present so the test is friendly to CI configurations that
    // didn't stage gb82-sc.
    let path = std::path::PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| "/home/lilith/work/codec-corpus".into()),
    )
    .join("gb82-sc/terminal.png");
    let img = match image::open(&path) {
        Ok(i) => i,
        Err(_) => {
            eprintln!("corpus image missing — skipping: {}", path.display());
            return;
        }
    };
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let cfg = LossyConfig::new(4.0)
        .with_effort(8)
        .with_gpu_butteraugli(true)
        .with_threads(1);
    let bytes = cfg
        .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let (dw, dh) = decode_oxide(&bytes).expect("jxl-oxide decode failed");
    assert_eq!((dw as u32, dh as u32), (w, h));
}
