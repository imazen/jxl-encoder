// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-123 decoder check: encode codec_wiki d=3 with KEEP_DCT32 hint at
//! e5/e6/e7, write to /tmp, and decode via jxl-rs + jxl-oxide (mandatory
//! per CLAUDE.md decoder discipline). Caller runs `djxl` manually on the
//! output paths for the third decoder check.

use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use std::io::Cursor;
use std::path::PathBuf;

fn decode_jxl_rs(data: &[u8]) -> Result<(usize, usize), String> {
    use jxl::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder_init = decoder;
    let decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
            Err(e) => return Err(format!("jxl-rs header decode error: {e:?}")),
        }
    };
    let basic = decoder.basic_info().clone();
    let (w, h) = basic.size;
    Ok((w, h))
}

fn decode_jxl_oxide(bytes: &[u8]) -> Result<(usize, usize), String> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .map_err(|e| format!("jxl-oxide: failed to read: {e:?}"))?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img
        .render_frame(0)
        .map_err(|e| format!("jxl-oxide: failed to render: {e:?}"))?;
    let fb = render.image_all_channels();
    Ok((fb.width(), fb.height()))
}

fn main() {
    let path = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png");
    let img = image::open(&path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let rgb_u8: Vec<u8> = rgb.as_raw().clone();

    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);

    for effort in &[5u8, 6, 7] {
        let cfg_keep = LossyConfig::new(3.0)
            .with_effort(*effort)
            .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct32_keep_hint: Some(true), ..Default::default() });
        let bytes = cfg_keep
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&lim)
            .encode(&rgb_u8)
            .expect("encode");

        let out_path = format!("/tmp/w44_123_codec_wiki_e{}_keep.jxl", effort);
        std::fs::write(&out_path, &bytes).unwrap();

        let oxide = decode_jxl_oxide(&bytes);
        let rs = decode_jxl_rs(&bytes);

        println!(
            "e{} keep_dct32=true: {} bytes -> jxl-oxide: {:?}, jxl-rs: {:?}, file={}",
            effort,
            bytes.len(),
            oxide,
            rs,
            out_path
        );
    }

    println!("\nNow run djxl manually (the third decoder check):");
    for effort in &[5u8, 6, 7] {
        println!(
            "  djxl /tmp/w44_123_codec_wiki_e{}_keep.jxl /tmp/w44_123_codec_wiki_e{}_djxl.png",
            effort, effort
        );
    }
}
