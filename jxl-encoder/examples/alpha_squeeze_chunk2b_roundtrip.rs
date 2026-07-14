// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-2.b roundtrip check on the W13-4 audit images. Encodes each
//! with `with_alpha_squeeze(true)` at four distances, decodes through
//! jxl-rs, prints alpha-plane MAE per image.
//!
//! Run:
//!   cargo run --release -p jxl-encoder --example alpha_squeeze_chunk2b_roundtrip

use image::ImageReader;
use jxl_encoder::{LossyConfig, PixelLayout};

fn read_rgba8(path: &str) -> Option<(u32, u32, Vec<u8>)> {
    let img = ImageReader::open(path).ok()?.decode().ok()?;
    let rgba8 = img.to_rgba8();
    let (w, h) = rgba8.dimensions();
    Some((w, h, rgba8.into_raw()))
}

fn decode_jxl_rs_rgba8(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
            Err(e) => {
                eprintln!("jxl-rs header decode error: {e:?}");
                return None;
            }
        }
    };
    let basic = decoder.basic_info().clone();
    let (w, h) = basic.size;
    let num_extras = basic.extra_channels.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![None; num_extras],
    });
    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => {
                eprintln!("jxl-rs frame info error: {e:?}");
                return None;
            }
        }
    };
    let channels = 4usize;
    let mut img = Image::<u8>::new((w * channels, h)).ok()?;
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        img.get_rect_mut(Rect {
            origin: (0, 0),
            size: (w * channels, h),
        })
        .into_raw(),
    )];
    loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_frame = fallback,
            Err(e) => {
                eprintln!("jxl-rs frame decode error: {e:?}");
                return None;
            }
        }
    }
    let mut pixels = Vec::with_capacity(w * h * channels);
    for y in 0..h {
        pixels.extend_from_slice(img.row(y));
    }
    Some((w, h, pixels))
}

fn alpha_mae(decoded: &[u8], original: &[u8]) -> f32 {
    let mut sum = 0u32;
    let mut n = 0u32;
    for i in (3..decoded.len()).step_by(4) {
        sum += (decoded[i] as i32 - original[i] as i32).unsigned_abs();
        n += 1;
    }
    if n == 0 { 0.0 } else { sum as f32 / n as f32 }
}

fn main() {
    let images: &[(&str, &str)] = &[
        (
            "gradients_semitrans_ui",
            "/home/lilith/work/codec-corpus/imageflow/test_inputs/gradients.png",
        ),
        (
            "red_night_opaque",
            "/home/lilith/work/codec-corpus/imageflow/test_inputs/red-night.png",
        ),
        (
            "alpha_nonpremul_photo_mask",
            "/home/lilith/work/codec-corpus/jxl/reference/conformance/alpha_nonpremultiplied.png",
        ),
    ];
    println!("image\tw\th\talpha_d\tbytes_no_sq\tbytes_sq\tdelta_pct\talpha_mae_sq\tjxl_rs_ok");
    for (label, path) in images {
        let Some((w, h, rgba)) = read_rgba8(path) else {
            println!("{label}\t?\t?\t-\t-\t-\t-\t-\tREAD_ERR");
            continue;
        };
        for &ad in &[0.5f32, 1.0, 2.0, 5.0] {
            let no_sq = LossyConfig::new(1.0).with_alpha_distance(Some(ad)).encode(
                &rgba,
                w,
                h,
                PixelLayout::Rgba8,
            );
            let sq = LossyConfig::new(1.0)
                .with_alpha_distance(Some(ad))
                .with_alpha_squeeze(true)
                .encode(&rgba, w, h, PixelLayout::Rgba8);
            let (nb_len, sb_len, delta_pct, mae, ok) = match (no_sq, sq) {
                (Ok(nb), Ok(sb)) => {
                    let delta_pct = 100.0 * (sb.len() as f64 - nb.len() as f64) / nb.len() as f64;
                    let (mae, ok) = match decode_jxl_rs_rgba8(&sb) {
                        Some((dw, dh, dec)) => {
                            if dw as u32 == w && dh as u32 == h {
                                (alpha_mae(&dec, &rgba), "yes")
                            } else {
                                (-1.0, "DIM_MISMATCH")
                            }
                        }
                        None => (-1.0, "DEC_ERR"),
                    };
                    (
                        nb.len() as i64,
                        sb.len() as i64,
                        delta_pct,
                        mae,
                        ok.to_string(),
                    )
                }
                (_, Err(e)) => (
                    -1,
                    -1,
                    0.0,
                    -1.0,
                    format!(
                        "SQ_ENC_ERR:{}",
                        format!("{e:?}").chars().take(40).collect::<String>()
                    ),
                ),
                (Err(e), _) => (
                    -1,
                    -1,
                    0.0,
                    -1.0,
                    format!(
                        "BASE_ENC_ERR:{}",
                        format!("{e:?}").chars().take(40).collect::<String>()
                    ),
                ),
            };
            println!(
                "{label}\t{w}\t{h}\t{ad:.1}\t{nb_len}\t{sb_len}\t{delta_pct:+.2}\t{mae:.2}\t{ok}"
            );
        }
    }
}
