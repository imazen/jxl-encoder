// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-118 bisection — identify the cause of the 1025469 e8/e9 d=4
//! SSIM2 -0.85 regression introduced by W44-117.
//!
//! Modes tested (sequential, single-image, with 1 warmup discard):
//!   A — JXL_W44_117_DISABLE=1                   (baseline: legacy uniform-4 EPF seed; pre-W44-117)
//!   B — default (W44-117 enabled)               (current main: regression)
//!   C — W44-117 enabled, content lifts disabled (via JXL_W44_118_DISABLE_LIFTS=1)
//!   D — W44-117 enabled, per-iter sharpness    (via JXL_W44_118_PER_ITER_SHARPNESS=1) [Option A]
//!
//! Cells: 1025469 e8 d={2,3,4}, e9 d={2,4} (regression cells + nearby control)
//! plus 3 spot-check cells from W44-117 wins (terminal e8/e9 d=3,4) to
//! verify the fix doesn't undo the W44-117 wins.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_118_bisect \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_118_bisect_2026-05-20.tsv

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

/// Cells: (image, corpus, effort, distance, role).
/// roles:
///  - "regress" — the W44-117 regression cluster
///  - "control" — same image other d (verify gradient)
///  - "spotwin" — W44-117 wins to verify modes C/D don't undo them
const CELLS: &[(&str, &str, u8, f32, &str)] = &[
    ("1025469.png", "CID22", 8, 2.0, "control"),
    ("1025469.png", "CID22", 8, 3.0, "control"),
    ("1025469.png", "CID22", 8, 4.0, "regress"),
    ("1025469.png", "CID22", 9, 2.0, "control"),
    ("1025469.png", "CID22", 9, 4.0, "regress"),
    // W44-117 wins to verify modes C/D don't undo them
    ("terminal.png", "gb82-sc", 8, 3.0, "spotwin"),
    ("terminal.png", "gb82-sc", 8, 4.0, "spotwin"),
    ("terminal.png", "gb82-sc", 9, 3.0, "spotwin"),
];

#[derive(Clone, Copy, Debug)]
enum Mode {
    A, // JXL_W44_117_DISABLE=1 (baseline)
    B, // default (W44-117 on, regression)
    C, // W44-117 on, content lifts disabled (W44_118_DISABLE_LIFTS) - moot
    D, // W44-117 on, per-iter sharpness (W44_118_PER_ITER_SHARPNESS)
    F, // W44-117 enabled only when is_screenshot (W44_118_SCREENSHOT_ONLY)
}

fn set_mode_env(mode: Mode) {
    // Clear all
    unsafe {
        std::env::remove_var("JXL_W44_117_DISABLE");
        std::env::remove_var("JXL_W44_118_DISABLE_LIFTS");
        std::env::remove_var("JXL_W44_118_PER_ITER_SHARPNESS");
        std::env::remove_var("JXL_W44_118_SCREENSHOT_ONLY");
    }
    // Set per mode
    match mode {
        Mode::A => unsafe { std::env::set_var("JXL_W44_117_DISABLE", "1") },
        Mode::B => {}
        Mode::C => unsafe { std::env::set_var("JXL_W44_118_DISABLE_LIFTS", "1") },
        Mode::D => unsafe { std::env::set_var("JXL_W44_118_PER_ITER_SHARPNESS", "1") },
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
    eprintln!("W44-118 bisection: identify cause of 1025469 e8/e9 d=4 -0.85 SSIM2 regression");
    eprintln!(
        "Cells: {} × 4 modes (A,B,C,D); interleaved A/B/C/D per cell",
        CELLS.len()
    );

    println!("image\teffort\tdistance\trole\tmode\tbytes\tbfly\tssim2");

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();

    for (i, &(image, corpus, effort, d, role)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} ({}) e{} d={} role={}",
            i + 1,
            n_cells,
            image,
            corpus,
            effort,
            d,
            role
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

        // For initial run, just test A and B to confirm regression.
        // Once C and D env hooks are wired, the full sweep runs.
        let want_all_modes = std::env::var("W44_118_ALL_MODES").is_ok();
        let modes: &[Mode] = if want_all_modes {
            &[Mode::A, Mode::B, Mode::C, Mode::D, Mode::F]
        } else {
            &[Mode::A, Mode::B, Mode::F]
        };
        for &mode in modes {
            set_mode_env(mode);
            let score = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);
            if let Some(s) = score {
                let mode_str = match mode {
                    Mode::A => "A_legacy",
                    Mode::B => "B_default",
                    Mode::C => "C_no_lifts",
                    Mode::D => "D_per_iter",
                    Mode::F => "F_screen_only",
                };
                println!(
                    "{}\te{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}",
                    image, effort, d, role, mode_str, s.bytes, s.butteraugli, s.ssim2,
                );
            } else {
                eprintln!("  mode {:?} score failed", mode);
            }
        }
        // Clear env between cells
        set_mode_env(Mode::B); // default
        unsafe { std::env::remove_var("JXL_W44_117_DISABLE") };
    }
}
