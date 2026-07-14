// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Score one or more pre-encoded .jxl files against a source PNG with the
//! CLAUDE.md-canonical metrics: decode via jxl-oxide in LINEAR sRGB, then Rust
//! butteraugli (smaller = better) + Rust fast-ssim2 (larger = better). Immune
//! to the butteraugli_main PNG-metadata bug. Used for A/B quality validation of
//! encoder changes where the two arms are produced out-of-band (e.g. a runtime
//! env toggle) and only the .jxl bytes are available here.
//!
//! usage:
//!   score_jxl_files <source.png> <label1>=<a.jxl> [<label2>=<b.jxl> ...]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use rgb::RGB;
use std::io::Cursor;

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 2,
        "usage: score_jxl_files <source.png> <label>=<file.jxl> [...]"
    );
    let src_path = &args[0];

    let src = image::open(src_path).expect("open source").to_rgb8();
    let (w, h) = (src.width() as usize, src.height() as usize);
    let src_raw = src.into_raw();

    // Reference in linear (butteraugli) + sRGB u8 (ssim2).
    let orig_linear: Vec<RGB<f32>> = src_raw
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    let orig_linear_img = Img::new(orig_linear, w, h);
    let orig_srgb: Vec<[u8; 3]> = src_raw.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let orig_srgb_img = Img::new(orig_srgb, w, h);
    let params = ButteraugliParams::default();

    println!(
        "{:<12} {:>10} {:>12} {:>10}",
        "label", "bytes", "butteraugli", "ssim2"
    );
    for arg in &args[1..] {
        let (label, path) = arg.split_once('=').expect("expected label=file.jxl");
        let bytes = std::fs::read(path).expect("read jxl");
        let (dw, dh, dec) = decode_jxl_linear(&bytes).expect("decode jxl");
        assert!(dw == w && dh == h, "{label}: decoded {dw}x{dh} != {w}x{h}");

        let dec_lin: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
        let dec_lin_img = Img::new(dec_lin, dw, dh);
        let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_lin_img.as_ref(), &params)
            .map(|r| r.score)
            .unwrap_or(f64::NAN);

        let dec_srgb: Vec<[u8; 3]> = dec
            .chunks(3)
            .map(|c| {
                [
                    linear_to_srgb_u8(c[0]),
                    linear_to_srgb_u8(c[1]),
                    linear_to_srgb_u8(c[2]),
                ]
            })
            .collect();
        let dec_srgb_img = Img::new(dec_srgb, dw, dh);
        let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
            .unwrap_or(f64::NAN);

        println!(
            "{label:<12} {:>10} {:>12.4} {:>10.4}",
            bytes.len(),
            bfly,
            ssim2
        );
    }
}
