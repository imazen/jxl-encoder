// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-96 baseline diff: encode every cell with default settings (W44-96
//! sub-discriminator active) and report the result so a follow-on script
//! can compare against the pre-W44-96 baseline (origin/main = W44-95
//! commit `85536ab8`).
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert parallel butteraugli-loop ssim2-loop' \
//!     --example w44_96_baseline_diff

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

// Cells of interest: ALL the cells from the dispatch_ab + extra
// FIXED-pass cells for broader coverage.
const CELLS: &[(&str, u8, f32, &str)] = &[
    // Already-FIXED cells we expect to stay byte-identical to baseline.
    // W44-29 fires + W44-96 sub-disc rejects (cells in REJECT_Z proxy set):
    ("2389166.png", 5, 3.0, "FIXED_BASELINE"),
    ("2389166.png", 5, 4.0, "FIXED_BASELINE"),
    ("2389166.png", 5, 5.0, "FIXED_BASELINE"),
    ("2389166.png", 5, 6.0, "FIXED_BASELINE"),
    ("2389166.png", 6, 4.0, "FIXED_BASELINE"),
    ("2389166.png", 6, 5.0, "FIXED_BASELINE"),
    ("2389166.png", 7, 4.0, "FIXED_BASELINE"),
    ("2389166.png", 7, 5.0, "FIXED_BASELINE"),
    ("3637739.png", 5, 3.0, "FIXED_BASELINE"),
    ("3637739.png", 5, 4.0, "FIXED_BASELINE"),
    ("3637739.png", 5, 5.0, "FIXED_BASELINE"),
    ("3637739.png", 5, 6.0, "FIXED_BASELINE"),
    ("3637739.png", 6, 4.0, "FIXED_BASELINE"),
    ("3637739.png", 6, 5.0, "FIXED_BASELINE"),
    ("3637739.png", 7, 4.0, "FIXED_BASELINE"),
    ("3637739.png", 7, 5.0, "FIXED_BASELINE"),
    ("1044329.png", 5, 3.0, "FIXED_BASELINE"),
    ("1044329.png", 5, 4.0, "FIXED_BASELINE"),
    ("1044329.png", 5, 6.0, "FIXED_BASELINE"),
    ("1044329.png", 6, 5.0, "FIXED_BASELINE"),
    ("1044329.png", 6, 6.0, "FIXED_BASELINE"),
    ("1044329.png", 7, 5.0, "FIXED_BASELINE"),
    ("7062219.png", 5, 5.0, "FIXED_BASELINE"),
    ("7062219.png", 6, 5.0, "FIXED_BASELINE"),
    ("7062219.png", 7, 5.0, "FIXED_BASELINE"),
    // WANT_Z cells (should differ from baseline = save bytes)
    ("1420710.png", 5, 5.0, "TARGET_Z"),
    ("1420710.png", 5, 6.0, "TARGET_Z"),
    ("1420710.png", 6, 5.0, "TARGET_Z"),
    ("1420710.png", 6, 6.0, "TARGET_Z"),
    ("1420710.png", 7, 5.0, "TARGET_Z"),
    ("1420710.png", 8, 5.0, "TARGET_Z"),
    ("1420710.png", 9, 5.0, "TARGET_Z"),
    ("1531677.png", 5, 5.0, "TARGET_Z"),
    ("1531677.png", 5, 6.0, "TARGET_Z"),
    ("1531677.png", 6, 5.0, "TARGET_Z"),
    ("1531677.png", 6, 6.0, "TARGET_Z"),
    ("1531677.png", 7, 5.0, "TARGET_Z"),
    // 1189261 W44-91 path (must not change from baseline)
    ("1189261.png", 7, 3.0, "FIXED_W91"),
    ("1189261.png", 7, 4.0, "FIXED_W91"),
    ("1189261.png", 7, 5.0, "FIXED_W91"),
    // 1418519 above all gates
    ("1418519.png", 7, 5.0, "FIXED_ABOVE_GATES"),
];

fn encode(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Vec<u8> {
    LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
}

fn srgb_u8_to_linear(rgb_u8: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let lin: Vec<RGB<f32>> = rgb_u8
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    Img::new(lin, w as usize, h as usize)
}

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
    let params = ButteraugliParams::default();
    println!("group\timage\teffort\tdistance\tbytes\tbfly\tssim2");

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    for &(image, effort, d, group) in CELLS.iter() {
        let path = PathBuf::from(CID22).join(image);
        let (w, h, raw, orig_linear_img, orig_srgb_img) =
            images_cache.entry(image.to_string()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let linear = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                (w, h, rgb, linear, srgb_img)
            });

        let bytes = encode(raw, *w, *h, effort, d);
        let mut bfly = f64::NAN;
        let mut ssim2 = f64::NAN;
        if let Some((dw, dh, dec)) = decode_jxl_linear(&bytes) {
            let dec_pixels: Vec<RGB<f32>> =
                dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
            let dec_linear_img = Img::new(dec_pixels, dw, dh);
            bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), &params)
                .map(|r| r.score as f64)
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
            ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
                .unwrap_or(f64::NAN);
        }
        println!(
            "{}\t{}\te{}\t{:.1}\t{}\t{:.4}\t{:.4}",
            group,
            image,
            effort,
            d,
            bytes.len(),
            bfly,
            ssim2
        );
    }
}
