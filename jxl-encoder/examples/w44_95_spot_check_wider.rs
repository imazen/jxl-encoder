// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-95 wider spot-check on additional mask < 50 photos NOT covered
//! by the W44-94 / W44-95 reproducer cell list.
//!
//! Targets: 2389166 (mask 46.24), 3637739 (mask 47.80), 1044329 (mask
//! 48.03) — all fall inside the W44-29 firing region (mask < 50) and so
//! their entropy_mul_table will swap to the new W44-95 values.
//!
//! Compares per-image cjxl vs jxl-encoder bytes at d=3..=6, e5..=e7.
//! Goal: verify no large unexpected regressions (>3 % vs cjxl) on photos
//! that W44-94 didn't probe.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_95_spot_check_wider

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const CJXL_BIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";

const CELLS: &[(&str, u8, f32)] = &[
    // 2389166 — mask 46.24, W44-29 fires
    ("2389166.png", 5, 3.0),
    ("2389166.png", 5, 4.0),
    ("2389166.png", 5, 5.0),
    ("2389166.png", 5, 6.0),
    ("2389166.png", 6, 4.0),
    ("2389166.png", 6, 5.0),
    ("2389166.png", 7, 4.0),
    ("2389166.png", 7, 5.0),
    // 3637739 — mask 47.80, W44-29 fires
    ("3637739.png", 5, 3.0),
    ("3637739.png", 5, 4.0),
    ("3637739.png", 5, 5.0),
    ("3637739.png", 5, 6.0),
    ("3637739.png", 6, 4.0),
    ("3637739.png", 6, 5.0),
    ("3637739.png", 7, 4.0),
    ("3637739.png", 7, 5.0),
    // 1044329 — mask 48.03 (W44-94 only probed e5 d=5 → -1.92pp; spread)
    ("1044329.png", 5, 3.0),
    ("1044329.png", 5, 4.0),
    ("1044329.png", 5, 6.0),
    ("1044329.png", 6, 5.0),
    ("1044329.png", 6, 6.0),
    ("1044329.png", 7, 5.0),
];

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

fn cjxl_size(src: &str, effort: u8, d: f32) -> Option<usize> {
    let tmp = format!(
        "/tmp/w44_95_spot_cjxl_{}_{}_{}.jxl",
        std::process::id(),
        effort,
        (d * 10.0) as u32
    );
    let out = Command::new(CJXL_BIN)
        .args(["-d", &d.to_string(), "-e", &effort.to_string(), src, &tmp])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sz = std::fs::metadata(&tmp).ok()?.len() as usize;
    let _ = std::fs::remove_file(&tmp);
    Some(sz)
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
    eprintln!("W44-95 wider spot-check (3 photos, mask < 50)");
    let params = ButteraugliParams::default();
    println!("image\teffort\tdistance\tcjxl_bytes\tshipped_bytes\tdelta_pct\tbfly\tssim2");

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let mut regressions = 0;
    let n_cells = CELLS.len();
    for (i, &(image, effort, d)) in CELLS.iter().enumerate() {
        eprintln!("[{}/{}] {} e{} d={}", i + 1, n_cells, image, effort, d);
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

        let cjxl_b = match cjxl_size(path.to_str().unwrap(), effort, d) {
            Some(s) => s,
            None => continue,
        };
        let shipped = match encode_shipped(raw, *w, *h, effort, d) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  encode failed: {}", e);
                continue;
            }
        };
        let shipped_bytes = shipped.len();
        let delta_pct = 100.0 * (shipped_bytes as f64 - cjxl_b as f64) / cjxl_b as f64;
        if delta_pct >= 3.0 {
            regressions += 1;
        }

        let (dw, dh, decoded_linear) = match decode_jxl_linear(&shipped) {
            Some(v) => v,
            None => continue,
        };
        if dw != *w as usize || dh != *h as usize {
            continue;
        }
        let dec_pixels: Vec<RGB<f32>> = decoded_linear
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let dec_linear_img = Img::new(dec_pixels, dw, dh);
        let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), &params)
            .map(|r| r.score as f64)
            .unwrap_or(f64::NAN);

        let dec_srgb: Vec<[u8; 3]> = decoded_linear
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
            "{}\te{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}",
            image, effort, d, cjxl_b, shipped_bytes, delta_pct, bfly, ssim2
        );
    }
    eprintln!();
    eprintln!(
        "Cells with shipped > cjxl + 3.0%: {} (acceptable: would-be-new-OPEN cells)",
        regressions
    );
}
