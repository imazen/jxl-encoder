// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-140 default-flip verification. After promoting W44_140_EPF_SEED_FADE_MAX
//! to the const 1.5 production default, verify:
//!   - new_default == prior_C_fade15 (fade=1.5 explicit env-var)
//!   - fade_off (env=1.0, i.e. fade_max <= min_distance) == prior_B_main
//!     (no fade, full W44-117 at every d>=1.0)
//!
//! Runs 3 cells x 3 modes interleaved:
//!   - terminal e8 d=1.4 (the +1.014 win cell)
//!   - terminal e8 d=1.2 (the regression cell that fade closes)
//!   - terminal e8 d=4.0 (protection cell, must be unchanged)
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_140_default_check \
//!     --manifest-path jxl-encoder/Cargo.toml

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::PathBuf;

fn encode_shipped(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
    LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
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

fn main() {
    let path = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc/terminal.png");
    let img = image::open(&path).unwrap();
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    let lin = srgb_u8_to_linear(&rgb, w, h);
    let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);

    let cells = vec![
        ("terminal_d12_e8", 8u8, 1.2f32),
        ("terminal_d14_e8", 8u8, 1.4f32),
        ("terminal_d40_e8", 8u8, 4.0f32),
    ];

    let modes = vec![
        ("default_new", None::<&str>, None::<&str>),
        ("fade_off", None::<&str>, Some("1.0")), // disables blend (fade_max <= min_distance)
        ("legacy_uniform4", Some("1"), None),
    ];

    println!("cell\tmode\tbytes\tbfly\tssim2");
    for (name, effort, d) in &cells {
        for (mode_name, dis, fade) in &modes {
            unsafe {
                std::env::remove_var("JXL_W44_117_DISABLE");
                std::env::remove_var("JXL_W44_140_EPF_SEED_FADE_MAX");
                if let Some(v) = dis {
                    std::env::set_var("JXL_W44_117_DISABLE", v);
                }
                if let Some(v) = fade {
                    std::env::set_var("JXL_W44_140_EPF_SEED_FADE_MAX", v);
                }
            }
            let bitstream = encode_shipped(&rgb, w, h, *effort, *d).expect("encode");
            let bytes = bitstream.len();
            let (dw, dh, dec) = decode_jxl_linear(&bitstream).expect("decode");
            assert_eq!((dw, dh), (w as usize, h as usize));
            let dec_pixels: Vec<RGB<f32>> =
                dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
            let dec_lin = Img::new(dec_pixels, dw, dh);
            let bfly = butteraugli_linear(
                lin.as_ref(),
                dec_lin.as_ref(),
                &ButteraugliParams::default(),
            )
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
            let ssim2 = fast_ssim2::compute_ssimulacra2(srgb_img.as_ref(), dec_srgb_img.as_ref())
                .unwrap_or(f64::NAN);

            unsafe {
                std::env::remove_var("JXL_W44_117_DISABLE");
                std::env::remove_var("JXL_W44_140_EPF_SEED_FADE_MAX");
            }
            println!(
                "{}\t{}\t{}\t{:.4}\t{:.4}",
                name, mode_name, bytes, bfly, ssim2
            );
        }
    }
}
