// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-118 Mode F validation — verify the screenshot-only W44-117 gate
//! across the W44-117 acceptance cell set + 1025469 + 30 spot FIXED
//! cells. Acceptance gates:
//!   * 1025469 e8/e9 d=4 SSIM2 regression ≤ -0.3 (was -0.85)
//!   * W44-117 wins preserved (terminal e8/e9 d=3 +0.66, d=4 +0.90)
//!   * Zero NEW FIXED→OPEN flips on 30+ spot cells
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_118_mode_f_validation \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_118_mode_f_validation_2026-05-20.tsv

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
const GB82SC: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// Same 32-cell set as w44_117_paired_ab.rs (the W44-117 acceptance set).
const CELLS: &[(&str, &str, u8, f32)] = &[
    // terminal e5..=e9 × d=2..6 (25 cells)
    ("terminal.png", "gb82-sc", 5, 2.0),
    ("terminal.png", "gb82-sc", 5, 3.0),
    ("terminal.png", "gb82-sc", 5, 4.0),
    ("terminal.png", "gb82-sc", 5, 5.0),
    ("terminal.png", "gb82-sc", 5, 6.0),
    ("terminal.png", "gb82-sc", 6, 2.0),
    ("terminal.png", "gb82-sc", 6, 3.0),
    ("terminal.png", "gb82-sc", 6, 4.0),
    ("terminal.png", "gb82-sc", 6, 5.0),
    ("terminal.png", "gb82-sc", 6, 6.0),
    ("terminal.png", "gb82-sc", 7, 2.0),
    ("terminal.png", "gb82-sc", 7, 3.0),
    ("terminal.png", "gb82-sc", 7, 4.0),
    ("terminal.png", "gb82-sc", 7, 5.0),
    ("terminal.png", "gb82-sc", 7, 6.0),
    ("terminal.png", "gb82-sc", 8, 2.0),
    ("terminal.png", "gb82-sc", 8, 3.0),
    ("terminal.png", "gb82-sc", 8, 4.0),
    ("terminal.png", "gb82-sc", 8, 5.0),
    ("terminal.png", "gb82-sc", 8, 6.0),
    ("terminal.png", "gb82-sc", 9, 2.0),
    ("terminal.png", "gb82-sc", 9, 3.0),
    ("terminal.png", "gb82-sc", 9, 4.0),
    ("terminal.png", "gb82-sc", 9, 5.0),
    ("terminal.png", "gb82-sc", 9, 6.0),
    // codec_wiki spot-check
    ("codec_wiki.png", "gb82-sc", 7, 3.0),
    ("codec_wiki.png", "gb82-sc", 7, 4.0),
    ("codec_wiki.png", "gb82-sc", 7, 5.0),
    // Photos
    ("1418519.png", "CID22", 8, 2.0),
    ("1418519.png", "CID22", 8, 3.0),
    ("1418519.png", "CID22", 8, 4.0),
    ("1418519.png", "CID22", 9, 2.0),
    ("1418519.png", "CID22", 9, 4.0),
    ("1025469.png", "CID22", 8, 2.0),
    ("1025469.png", "CID22", 8, 3.0),
    ("1025469.png", "CID22", 8, 4.0),
    ("1025469.png", "CID22", 9, 2.0),
    ("1025469.png", "CID22", 9, 4.0),
    ("1189261.png", "CID22", 8, 2.0),
    ("1189261.png", "CID22", 8, 4.0),
    ("1420710.png", "CID22", 8, 2.0),
    ("1420710.png", "CID22", 8, 4.0),
    ("1531677.png", "CID22", 8, 2.0),
    ("1531677.png", "CID22", 8, 4.0),
    // Extra photo spot checks (30+ cells gate)
    ("1418519.png", "CID22", 7, 1.0),
    ("1418519.png", "CID22", 7, 2.0),
    ("1418519.png", "CID22", 7, 3.0),
    ("1025469.png", "CID22", 7, 1.0),
    ("1025469.png", "CID22", 7, 2.0),
    ("1025469.png", "CID22", 7, 3.0),
    ("1025469.png", "CID22", 9, 3.0),
    ("1189261.png", "CID22", 7, 1.0),
    ("1189261.png", "CID22", 7, 4.0),
    ("1420710.png", "CID22", 7, 4.0),
    ("1531677.png", "CID22", 7, 4.0),
    // More screenshot spot checks (verify F=B for all screenshots)
    ("codec_wiki.png", "gb82-sc", 8, 4.0),
    ("imac_dark.png", "gb82-sc", 8, 4.0),
    ("imac_g3.png", "gb82-sc", 8, 4.0),
];

#[derive(Clone, Copy, Debug)]
enum Mode {
    A, // JXL_W44_117_DISABLE=1 (pre-W44-117 baseline)
    B, // default (W44-117 on, current main)
    F, // W44-117 gated on is_screenshot (proposed fix)
}

fn set_mode_env(mode: Mode) {
    unsafe {
        std::env::remove_var("JXL_W44_117_DISABLE");
        std::env::remove_var("JXL_W44_118_SCREENSHOT_ONLY");
    }
    match mode {
        Mode::A => unsafe { std::env::set_var("JXL_W44_117_DISABLE", "1") },
        Mode::B => {}
        Mode::F => unsafe { std::env::set_var("JXL_W44_118_SCREENSHOT_ONLY", "1") },
    }
}

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

#[derive(Clone, Copy, Default, Debug)]
struct Score {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
}

fn score_cell(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
) -> Option<Score> {
    let bitstream = match encode_shipped(rgb, w, h, effort, d) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  encode failed: {}", e);
            return None;
        }
    };
    let bytes = bitstream.len();
    let (dw, dh, decoded_linear) = decode_jxl_linear(&bitstream)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let params = ButteraugliParams::default();
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
    Some(Score {
        bytes,
        butteraugli: bfly,
        ssim2,
    })
}

fn main() {
    eprintln!("W44-118 Mode F validation: A=legacy / B=default / F=screen-only");
    eprintln!("Cells: {}", CELLS.len());

    println!(
        "image\teffort\tdistance\tA_bytes\tB_bytes\tF_bytes\tBA_pct\tFA_pct\tA_bfly\tB_bfly\tF_bfly\tA_ssim2\tB_ssim2\tF_ssim2\tBA_ssim2\tFA_ssim2\tFB_ssim2"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    let mut new_regressions = 0usize; // F vs B SSIM2 regressions > 0.3
    let mut regressions_fixed = 0usize; // B regressions that F restores

    for (i, &(image, corpus, effort, d)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} ({}) e{} d={}",
            i + 1,
            n_cells,
            image,
            corpus,
            effort,
            d,
        );

        let dir = match corpus {
            "CID22" => CID22,
            "gb82-sc" => GB82SC,
            _ => continue,
        };
        let path = PathBuf::from(dir).join(image);
        let cache_key = format!("{}/{}", corpus, image);
        let (w, h, raw, orig_linear_img, orig_srgb_img) =
            images_cache.entry(cache_key.clone()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let linear = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                (w, h, rgb, linear, srgb_img)
            });

        set_mode_env(Mode::A);
        let sa = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);
        set_mode_env(Mode::B);
        let sb = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);
        set_mode_env(Mode::F);
        let sf = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);

        if let (Some(a), Some(b), Some(f)) = (sa, sb, sf) {
            let ba_pct = 100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let fa_pct = 100.0 * (f.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let ba_s2 = b.ssim2 - a.ssim2;
            let fa_s2 = f.ssim2 - a.ssim2;
            let fb_s2 = f.ssim2 - b.ssim2;
            println!(
                "{}\te{}\t{}\t{}\t{}\t{}\t{:+.3}\t{:+.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.3}\t{:+.3}\t{:+.3}",
                image,
                effort,
                d,
                a.bytes,
                b.bytes,
                f.bytes,
                ba_pct,
                fa_pct,
                a.butteraugli,
                b.butteraugli,
                f.butteraugli,
                a.ssim2,
                b.ssim2,
                f.ssim2,
                ba_s2,
                fa_s2,
                fb_s2,
            );
            if fb_s2 < -0.3 {
                new_regressions += 1;
                eprintln!("  ⚠ F vs B regression: {:+.3}", fb_s2);
            }
            if ba_s2 < -0.3 && fa_s2 >= -0.1 {
                regressions_fixed += 1;
                eprintln!("  ✓ F fixed B regression: {:+.3} → {:+.3}", ba_s2, fa_s2);
            }
        } else {
            eprintln!("  one or more scores failed");
        }
    }

    eprintln!();
    eprintln!("=== W44-118 Mode F SUMMARY ===");
    eprintln!(
        "New F vs B regressions > 0.3 SSIM2: {} (gate: 0)",
        new_regressions
    );
    eprintln!("B regressions F restored: {}", regressions_fixed);
}
