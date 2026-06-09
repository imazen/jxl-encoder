// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-205 multi-decoder roundtrip: bitstreams produced with the W44-205
//! extension (medium buckets 2 + 4 also disabled in `compute_custom_orders`)
//! on photo + screenshot cells from the Phase-1 probe MUST decode cleanly
//! via jxl-rs AND jxl-oxide.
//!
//! Per the W44-204 task spec: W44-205 only changes coefficient scan order
//! (Lehmer permutation header bytes), not the coefficient values themselves
//! — so decoded pixels are guaranteed bit-identical to the W44-201
//! baseline. This test is the smoke check that the bitstream the wire
//! emits IS spec-compliant on cells where the gate actually fires.
//!
//! Run with:
//!   cargo test --release -p jxl-encoder --features parallel \
//!     --test w44_205_decoder_roundtrip
//!
//! Panics if the corpus is missing (no graceful skip per repo policy).

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;

/// (display_name, image_path, effort, distance) tuples — 4 spot cells
/// chosen from the W44-205 Phase-1 probe: 3 Cluster-#1 photo LOSER
/// cells where the bucket-2/4 disable saves >1% bytes, and 1 SCRN
/// PROTECT cell where the gate has minimal effect (verifies it still
/// decodes cleanly on screenshots).
const CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "loser_3637739_d4",
        "CID22/CID22-512/validation/3637739.png",
        7,
        4.0,
    ),
    (
        "loser_297394_d5",
        "CID22/CID22-512/validation/297394.png",
        7,
        5.0,
    ),
    (
        "loser_7062219_d4",
        "CID22/CID22-512/validation/7062219.png",
        7,
        4.0,
    ),
    ("scrn_codec_wiki_d4", "gb82-sc/codec_wiki.png", 7, 4.0),
];

fn decode_oxide(bytes: &[u8]) -> Result<(u32, u32), String> {
    let img = jxl_oxide::JxlImage::builder()
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

/// Requires CID22 + GB82-SC corpora. Per repo "NO GRACEFUL SKIPS" rule,
/// this test panics if the corpus is missing.
#[test]
#[ignore = "needs codec-corpus (CODEC_CORPUS_DIR); nightly + local run with --include-ignored"]
fn w44_205_medium_buckets_decoders_clean() {
    let mut tested = 0;
    for &(name, path, effort, distance) in CELLS {
        let path = &crate::corpus_file(path);
        assert!(
            Path::new(path).exists(),
            "[w44-205] corpus file {} not found at {}",
            name,
            path
        );
        let img = image::open(path).expect("open png");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        // Default (Zenjxl) — W44-205 extension active, medium buckets
        // 2 + 4 skipped in addition to W44-201's large 3 + 6.
        let bytes = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(4)
            .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
            .expect("encode");
        eprintln!(
            "[w44-205] {} e{} d={:.1} -> {} bytes",
            name,
            effort,
            distance,
            bytes.len()
        );

        let (ow, oh) = decode_oxide(&bytes)
            .unwrap_or_else(|e| panic!("[w44-205] {} oxide decode FAILED: {}", name, e));
        assert_eq!((ow, oh), (w, h), "oxide size mismatch on {}", name);

        let (rw, rh) = decode_jxl_rs(&bytes)
            .unwrap_or_else(|e| panic!("[w44-205] {} jxl-rs decode FAILED: {}", name, e));
        assert_eq!(
            (rw, rh),
            (w as usize, h as usize),
            "jxl-rs size mismatch on {}",
            name
        );

        tested += 1;
    }
    eprintln!(
        "[w44-205] decoder roundtrip: {} of {} cells passed (jxl-rs + jxl-oxide)",
        tested,
        CELLS.len()
    );
    assert_eq!(tested, CELLS.len(), "expected all cells to pass");
}
