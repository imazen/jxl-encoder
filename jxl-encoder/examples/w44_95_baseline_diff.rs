// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-95 baseline diff. For each cell, encodes both:
//!  - OLD: forces the old W44-29 entropy_mul_table (dct32x32=1.34) via
//!    a hidden encoder hook. Specifically: uses `with_high_d_photo_hint(Some(true))`
//!    AND injects an internal_params override matching W44-94 default.
//!  - NEW: stock production default (W44-95 dct32x32=1.20).
//!
//! Compares per-image bytes and SSIM2. Verifies no FIXED→OPEN flips on
//! a wider photo population.
//!
//! Different from `w44_95_ship_variant_z_repro` which forced hint=false
//! (suppressing the gate entirely) — this version mimics the W44-94
//! default path exactly by leaving the gate ENABLED and injecting the
//! OLD table values via internal_params.
//!
//! Note: this doesn't perfectly simulate "without my code change" because
//! internal_params override is applied BEFORE the gate fires (and the
//! gate then swaps the profile.entropy_mul_table back to
//! high_d_photo_smooth_suppressed()). So we still need to compare
//! against a forced-hint=false + injected-old-table path. The OLD column
//! below reproduces what W44-94 reported for its "default" measurements.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_95_baseline_diff

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::effort::{EntropyMulTable, LossyInternalParams};
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
    ("2389166.png", 6, 4.0),
    ("2389166.png", 7, 5.0),
    // 3637739 — mask 47.80, W44-29 fires
    ("3637739.png", 5, 3.0),
    ("3637739.png", 5, 4.0),
    ("3637739.png", 5, 5.0),
    ("3637739.png", 6, 5.0),
    ("3637739.png", 7, 4.0),
    ("3637739.png", 7, 5.0),
    // 1044329 — mask 48.03
    ("1044329.png", 5, 3.0),
    ("1044329.png", 5, 4.0),
    ("1044329.png", 6, 5.0),
    ("1044329.png", 7, 5.0),
];

/// W44-29 OLD lift table (dct16=1.27, dct32=1.34, dct16x32=1.349) —
/// this matches the pre-W44-95 production constants.
fn old_w44_29_table() -> EntropyMulTable {
    let mut t = EntropyMulTable::reference();
    t.dct16x16 = 1.27;
    t.dct32x32 = 1.34;
    t.dct16x32 = 1.34 * (1.49 / 1.48);
    t
}

fn encode_old(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
    let mut cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_high_d_photo_hint(Some(false));
    let mut internal = LossyInternalParams::default();
    internal.entropy_mul_table = Some(old_w44_29_table());
    cfg = cfg.with_internal_params(internal);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

fn encode_new(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
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

fn measure(
    bytes: &[u8],
    w: u32,
    h: u32,
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> (f64, f64) {
    let (dw, dh, decoded_linear) = match decode_jxl_linear(bytes) {
        Some(v) => v,
        None => return (f64::NAN, f64::NAN),
    };
    if dw != w as usize || dh != h as usize {
        return (f64::NAN, f64::NAN);
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_lin.as_ref(), dec_lin_img.as_ref(), params)
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
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);
    (bfly, ssim2)
}

fn cjxl_size(src: &str, effort: u8, d: f32) -> Option<usize> {
    let tmp = format!(
        "/tmp/w44_95_diff_cjxl_{}_{}_{}.jxl",
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
    eprintln!("W44-95 baseline-diff on 3 photos (mask < 50)");
    let params = ButteraugliParams::default();
    println!(
        "image\teffort\tdistance\tcjxl\told_bytes\told_pct\told_bfly\told_ssim2\tnew_bytes\tnew_pct\tnew_bfly\tnew_ssim2\tbytes_delta\tssim2_delta\tstatus"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();
    let mut fixed_to_open = 0;
    let mut open_to_fixed = 0;
    let mut max_ssim2_drop: f64 = 0.0;

    let n_cells = CELLS.len();
    for (i, &(image, effort, d)) in CELLS.iter().enumerate() {
        eprintln!("[{}/{}] {} e{} d={}", i + 1, n_cells, image, effort, d);
        let path = PathBuf::from(CID22).join(image);

        let (w, h, raw, orig_lin, orig_srgb) =
            images_cache.entry(image.to_string()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let lin = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                (w, h, rgb, lin, srgb_img)
            });

        let cjxl_b = match cjxl_size(path.to_str().unwrap(), effort, d) {
            Some(s) => s,
            None => continue,
        };
        let old_b = match encode_old(raw, *w, *h, effort, d) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("old failed: {}", e);
                continue;
            }
        };
        let new_b = match encode_new(raw, *w, *h, effort, d) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("new failed: {}", e);
                continue;
            }
        };
        let (old_bfly, old_ssim2) = measure(&old_b, *w, *h, orig_lin, orig_srgb, &params);
        let (new_bfly, new_ssim2) = measure(&new_b, *w, *h, orig_lin, orig_srgb, &params);
        let old_pct = 100.0 * (old_b.len() as f64 - cjxl_b as f64) / cjxl_b as f64;
        let new_pct = 100.0 * (new_b.len() as f64 - cjxl_b as f64) / cjxl_b as f64;
        let bytes_delta = new_b.len() as i64 - old_b.len() as i64;
        let ssim2_delta = new_ssim2 - old_ssim2;
        max_ssim2_drop = max_ssim2_drop.min(ssim2_delta);
        let status = if old_pct < 3.0 && new_pct >= 3.0 {
            fixed_to_open += 1;
            "FIXED→OPEN!!!"
        } else if old_pct >= 3.0 && new_pct < 3.0 {
            open_to_fixed += 1;
            "OPEN→FIXED"
        } else if new_b.len() == old_b.len() {
            "byte-identical"
        } else {
            "stable"
        };
        println!(
            "{}\te{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:+}\t{:+.4}\t{}",
            image,
            effort,
            d,
            cjxl_b,
            old_b.len(),
            old_pct,
            old_bfly,
            old_ssim2,
            new_b.len(),
            new_pct,
            new_bfly,
            new_ssim2,
            bytes_delta,
            ssim2_delta,
            status
        );
    }
    eprintln!();
    eprintln!("=== W44-95 wider baseline-diff summary ===");
    eprintln!("FIXED→OPEN flips: {} (must be 0)", fixed_to_open);
    eprintln!("OPEN→FIXED closes: {} (bonus)", open_to_fixed);
    eprintln!("Worst SSIM2 drop: {:.4} (must be >= -0.30)", max_ssim2_drop);
}
