// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-78 multi-decoder roundtrip: the gate-widened bitstreams must decode
//! cleanly via jxl-rs (primary) AND jxl-oxide. Tests the 3 affected EV
//! cells from the W44-78 sweep (1420710, 1044329, 2389166 at d=3.0 e7).
//!
//! Run with:
//!   cargo test --release -p jxl-encoder --features parallel \
//!     --test w44_78_decoder_roundtrip
//!
//! Panics if `~/work/codec-corpus/CID22-512/validation` is missing
//! (no graceful skip per repo policy).

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;

const CELLS: &[(&str, &str, f32)] = &[
    (
        "1420710",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        3.0,
    ),
    (
        "1044329",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1044329.png",
        3.0,
    ),
    (
        "2389166",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/2389166.png",
        3.0,
    ),
];

fn decode_oxide(bytes: &[u8]) -> Result<(u32, u32), String> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(bytes))
        .map_err(|e| format!("oxide read: {}", e))?;
    let w = img.width();
    let h = img.height();
    let _ = img
        .render_frame(0)
        .map_err(|e| format!("oxide render: {}", e))?;
    Ok((w, h))
}

fn decode_jxl_rs(bytes: &[u8]) -> Result<(usize, usize), String> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = bytes;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => return Err(format!("jxl-rs header: {:?}", e)),
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

    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame info: {:?}", e)),
        }
    };
    let mut output_image = Image::<f32>::new((width * channels, height))
        .map_err(|e| format!("output alloc: {:?}", e))?;
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];
    let _decoder = loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame: {:?}", e)),
        }
    };
    Ok((width, height))
}

/// Requires `~/work/codec-corpus/CID22-512/validation`. Per repo
/// "NO GRACEFUL SKIPS" rule, this test panics loudly if the corpus
/// is missing. CI environments must include codec-corpus, same as
/// every other roundtrip test in this crate.
#[test]
fn w44_78_widened_gate_decoders_clean() {
    let mut tested = 0;
    for &(name, path, distance) in CELLS {
        assert!(
            Path::new(path).exists(),
            "[w44-78] corpus file {} not found at {}",
            name,
            path
        );
        let img = image::open(path).expect("open png");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let bytes = LossyConfig::new(distance)
            .with_effort(7)
            .with_threads(4)
            .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
            .expect("encode");
        eprintln!(
            "[w44-78] {} d={:.1} -> {} bytes",
            name,
            distance,
            bytes.len()
        );

        let (ow, oh) = decode_oxide(&bytes)
            .unwrap_or_else(|e| panic!("[w44-78] {} oxide decode FAILED: {}", name, e));
        assert_eq!((ow, oh), (w, h), "oxide size mismatch on {}", name);

        let (rw, rh) = decode_jxl_rs(&bytes)
            .unwrap_or_else(|e| panic!("[w44-78] {} jxl-rs decode FAILED: {}", name, e));
        assert_eq!(
            (rw, rh),
            (w as usize, h as usize),
            "jxl-rs size mismatch on {}",
            name
        );

        tested += 1;
    }
    eprintln!("[w44-78] decoder roundtrip: {} tested", tested);
    assert_eq!(tested, CELLS.len(), "expected all 3 cells to pass");
}
