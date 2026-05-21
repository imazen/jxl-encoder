// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-96 dispatch A/B: paired interleaved encode of the W44-96 sub-gate
//! target cells (closure goals) AND regression-defense cells (must stay
//! identical to current main / W44-95 baseline).
//!
//! Compares two A/B variants per cell:
//!   A (force_off):   `LossyConfig::with_high_d_photo_hint(Some(false))`
//!                    — disables both W44-29 AND the new W44-96 sub-gate.
//!   B (default):     `LossyConfig::new(d)` — auto W44-29 (with W44-96
//!                    sub-discriminator).
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert parallel butteraugli-loop ssim2-loop' \
//!     --example w44_96_dispatch_ab

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

// (image, effort, distance, group)
// Groups:
//   WANT_Z          — must close (variant Z fires, saves bytes)
//   W93_REGR_FIXED  — W44-91 1189261 + W44-78 1418519 cells (gate doesn't
//                     fire here because mask range/proxies don't match) —
//                     must stay byte-identical to baseline
//   W95_REGR_FIXED  — W44-95 measured FIXED→OPEN flips on Z; W44-96
//                     discriminator must reject them so they stay FIXED
//   FIXED_CTRL      — must stay byte-identical (W44-29 fires but variant
//                     Z must NOT fire — verifies the sub-discriminator
//                     rejects them)
const CELLS: &[(&str, u8, f32, &str)] = &[
    // ───── WANT_Z target cells (W44-95 measured wins) ─────
    ("1420710.png", 6, 5.0, "WANT_Z"),
    ("1420710.png", 6, 6.0, "WANT_Z"),
    ("1420710.png", 8, 5.0, "WANT_Z"),
    ("1420710.png", 9, 5.0, "WANT_Z"),
    ("1531677.png", 5, 6.0, "WANT_Z"),
    ("1531677.png", 6, 6.0, "WANT_Z"),
    // ───── W93 regression FIXED cells — mask>=50, W44-29 doesn't fire ─────
    ("1189261.png", 7, 3.0, "W93_REGR_FIXED"),
    ("1189261.png", 7, 4.0, "W93_REGR_FIXED"),
    ("1189261.png", 7, 5.0, "W93_REGR_FIXED"),
    ("1418519.png", 6, 5.0, "W93_REGR_FIXED"),
    ("1418519.png", 7, 5.0, "W93_REGR_FIXED"),
    ("1418519.png", 8, 5.0, "W93_REGR_FIXED"),
    // ───── W95 regression cells — variant Z REGRESSED in W44-95 ─────
    //   3637739 mask=75 — gate doesn't fire at all (verifies)
    //   2389166 mask=46 — W44-29 fires but W44-96 sub-disc must REJECT
    ("2389166.png", 7, 5.0, "W95_REGR_FIXED"),
    ("3637739.png", 5, 5.0, "W95_REGR_FIXED"),
    ("3637739.png", 7, 4.0, "W95_REGR_FIXED"),
    // ───── FIXED control cells where W44-29 fires but W44-96 must NOT ─────
    ("1420710.png", 6, 4.0, "FIXED_CTRL"),
    ("1531677.png", 5, 5.0, "FIXED_CTRL"),
    ("1531677.png", 7, 5.0, "FIXED_CTRL"),
    ("2389166.png", 5, 3.0, "FIXED_CTRL"),
    ("2389166.png", 5, 5.0, "FIXED_CTRL"),
    ("2389166.png", 6, 4.0, "FIXED_CTRL"),
    ("1044329.png", 5, 3.0, "FIXED_CTRL"),
    ("1044329.png", 6, 5.0, "FIXED_CTRL"),
    ("1044329.png", 7, 5.0, "FIXED_CTRL"),
    ("7062219.png", 6, 5.0, "FIXED_CTRL"),
    ("7062219.png", 7, 5.0, "FIXED_CTRL"),
];

fn encode(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32, force_off: bool) -> Vec<u8> {
    let mut cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    if force_off {
        cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides {
            high_d_photo_hint: Some(false),
            ..Default::default()
        });
    }
    cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode")
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
    eprintln!("# W44-96 dispatch A/B: variant Z sub-discriminator");
    let params = ButteraugliParams::default();
    println!(
        "group\timage\teffort\tdistance\tA_bytes\tB_bytes\tdelta_b\tdelta_pct\tA_bfly\tB_bfly\tdelta_bfly\tA_ssim2\tB_ssim2\tdelta_ssim2"
    );

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

        // Interleaved paired encode (A then B per cell).
        let a = encode(raw, *w, *h, effort, d, true);
        let b = encode(raw, *w, *h, effort, d, false);
        let a_n = a.len();
        let b_n = b.len();
        let delta_b = b_n as i64 - a_n as i64;
        let delta_pct = (delta_b as f64) / (a_n as f64) * 100.0;

        // Metrics on both encodes.
        let mut a_bfly = f64::NAN;
        let mut a_ssim2 = f64::NAN;
        if let Some((dw, dh, dec)) = decode_jxl_linear(&a) {
            let dec_pixels: Vec<RGB<f32>> =
                dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
            let dec_linear_img = Img::new(dec_pixels, dw, dh);
            a_bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), &params)
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
            a_ssim2 =
                fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
                    .unwrap_or(f64::NAN);
        }
        let mut b_bfly = f64::NAN;
        let mut b_ssim2 = f64::NAN;
        if let Some((dw, dh, dec)) = decode_jxl_linear(&b) {
            let dec_pixels: Vec<RGB<f32>> =
                dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
            let dec_linear_img = Img::new(dec_pixels, dw, dh);
            b_bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), &params)
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
            b_ssim2 =
                fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
                    .unwrap_or(f64::NAN);
        }
        println!(
            "{}\t{}\te{}\t{:.1}\t{}\t{}\t{:+}\t{:+.3}\t{:.4}\t{:.4}\t{:+.4}\t{:.4}\t{:.4}\t{:+.4}",
            group,
            image,
            effort,
            d,
            a_n,
            b_n,
            delta_b,
            delta_pct,
            a_bfly,
            b_bfly,
            b_bfly - a_bfly,
            a_ssim2,
            b_ssim2,
            b_ssim2 - a_ssim2
        );
    }
}
