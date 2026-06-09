// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-164 Smart-Zenjxl chunk 1 multi-decoder roundtrip: bitstreams produced
//! with the auto-classifier ON (Zenjxl default) on GB82-SC screenshots at
//! e ∈ {5, 6} (where the dispatch actually fires patches enablement via
//! `adapt_to_image_content`) MUST decode cleanly via jxl-rs AND jxl-oxide.
//!
//! Photo cells are byte-identical to pre-W44-164 (auto-classifier short-
//! circuits via the Photo / Unknown classification → adapter is a no-op),
//! so they're already covered by the existing decoder tests; this file
//! focuses on the patches-enabled screenshot path.
//!
//! Run with:
//!   cargo test --release -p jxl-encoder --features parallel \
//!     --test w44_164_decoder_roundtrip
//!
//! Panics if the corpus is missing (no graceful skip per repo policy).

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;

/// (display_name, image_path, effort, distance) tuples — the chunk's
/// 3 spot-check screenshot cells where the auto-classifier flips
/// `patches` on at e ∈ {5, 6}.
const CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "codec_wiki_e5",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        5,
        1.0,
    ),
    (
        "imac_g3_e6",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        6,
        1.0,
    ),
    (
        "terminal_e5",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        5,
        1.0,
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

/// Requires the GB82-SC corpus. Per repo "NO GRACEFUL SKIPS" rule,
/// this test panics if the corpus is missing.
#[test]
fn w44_164_auto_classify_decoders_clean() {
    let mut tested = 0;
    for &(name, path, effort, distance) in CELLS {
        assert!(
            Path::new(path).exists(),
            "[w44-164] corpus file {} not found at {}",
            name,
            path
        );
        let img = image::open(path).expect("open png");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        // Default (Zenjxl) — auto-classifier ON, expected to flip
        // patches=true on this screenshot at e ∈ {5, 6}.
        let bytes = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(4)
            .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
            .expect("encode");
        eprintln!(
            "[w44-164] {} e{} d={:.1} -> {} bytes",
            name,
            effort,
            distance,
            bytes.len()
        );

        let (ow, oh) = decode_oxide(&bytes)
            .unwrap_or_else(|e| panic!("[w44-164] {} oxide decode FAILED: {}", name, e));
        assert_eq!((ow, oh), (w, h), "oxide size mismatch on {}", name);

        let (rw, rh) = decode_jxl_rs(&bytes)
            .unwrap_or_else(|e| panic!("[w44-164] {} jxl-rs decode FAILED: {}", name, e));
        assert_eq!(
            (rw, rh),
            (w as usize, h as usize),
            "jxl-rs size mismatch on {}",
            name
        );

        tested += 1;
    }
    eprintln!(
        "[w44-164] decoder roundtrip: {} of {} cells passed",
        tested,
        CELLS.len()
    );
    assert_eq!(tested, CELLS.len(), "expected all cells to pass");
}
