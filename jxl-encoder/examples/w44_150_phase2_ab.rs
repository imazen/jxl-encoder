// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-150 Phase 2 paired A/B bench — measured the Mechanism A photo
//! admission gate (`mask_p25 >= 85.0 AND distance >= 4.0`) on the
//! W44-147 1418519 d=5/6 cluster + W44-118 protection set (1025469)
//! + 4-photo spot-check (1189261/1420710/1531677/7552578).
//!
//! **HONEST-STOP (2026-05-21)**: production code REVERTED after this
//! bench measured +0.27 mean SSIM2 on 1418519 d=5/6 (HARD gate wanted
//! ≥ +1.0). Proxy discriminator works (51/51 protection cells
//! byte-identical) but the W44-117 EPF seed mechanism alone only
//! recovers ~30% of the deficit at d=5 and ~0% at d=6 on e8/e9 cells;
//! e7 cells stay byte-identical because W44-117 is gated on
//! `butteraugli_iters > 0 AND profile.epf_dynamic_sharpness` (both
//! false at e<=7). See `benchmarks/w44_150_mask_p25_admission_2026-05-21.meta`
//! for full rationale and pivot recommendation. This example is kept
//! as a measurement reproducer; the env hook `JXL_W44_150_PHOTO_DISABLE`
//! is NO LONGER WIRED (was removed at revert) — both modes A and B now
//! produce byte-identical output. To re-measure W44-150-style
//! admission, future agents would need to re-introduce the env hook
//! and gate at the W44-118 call site.
//!
//! Original modes (now both equivalent post-revert):
//!   * A = `JXL_W44_150_PHOTO_DISABLE=1` (force legacy photo path)
//!   * B = default
//!
//! Acceptance gates evaluated (per spec):
//!   * Hash-locks 36/36 BYTE-IDENTICAL: PASS
//!   * 1025469 d=2/3/4/5/6 × e7/e8/e9: BYTE-IDENTICAL: PASS
//!   * 39 other CID22 photo cells (4-photo spot-check): BYTE-IDENTICAL: PASS
//!   * 1418519 d=5/6: SSIM2 ≥ +1.0 net: **FAIL** (+0.27 mean)
//!   * 1418519 d=4: SSIM2 within ±0.30: PASS (-0.08 mean)
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_150_phase2_ab \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_150_mask_p25_admission_2026-05-21.tsv

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

/// TARGET = 1418519 d=4/5/6 × e7/e8/e9 (the W44-147 audit cluster, 9 cells)
/// PROTECT_W118 = 1025469 d=2/3/4/5/6 × e7/e8/e9 (W44-118 protection, 15 cells)
/// SPOT_PHOTO = {1189261, 1420710, 1531677, 7552578} × {e7,e8,e9} × {d=4,5,6} (36 cells)
const CELLS: &[(&str, u8, f32)] = &[
    // TARGET — 1418519 d=4/5/6 × e7/e8/e9
    ("1418519.png", 7, 4.0),
    ("1418519.png", 7, 5.0),
    ("1418519.png", 7, 6.0),
    ("1418519.png", 8, 4.0),
    ("1418519.png", 8, 5.0),
    ("1418519.png", 8, 6.0),
    ("1418519.png", 9, 4.0),
    ("1418519.png", 9, 5.0),
    ("1418519.png", 9, 6.0),
    // PROTECT_W118 — 1025469 d=2..6 × e7/e8/e9
    ("1025469.png", 7, 2.0),
    ("1025469.png", 7, 3.0),
    ("1025469.png", 7, 4.0),
    ("1025469.png", 7, 5.0),
    ("1025469.png", 7, 6.0),
    ("1025469.png", 8, 2.0),
    ("1025469.png", 8, 3.0),
    ("1025469.png", 8, 4.0),
    ("1025469.png", 8, 5.0),
    ("1025469.png", 8, 6.0),
    ("1025469.png", 9, 2.0),
    ("1025469.png", 9, 3.0),
    ("1025469.png", 9, 4.0),
    ("1025469.png", 9, 5.0),
    ("1025469.png", 9, 6.0),
    // SPOT_PHOTO — 1189261 d=4/5/6 × e7/e8/e9
    ("1189261.png", 7, 4.0),
    ("1189261.png", 7, 5.0),
    ("1189261.png", 7, 6.0),
    ("1189261.png", 8, 4.0),
    ("1189261.png", 8, 5.0),
    ("1189261.png", 8, 6.0),
    ("1189261.png", 9, 4.0),
    ("1189261.png", 9, 5.0),
    ("1189261.png", 9, 6.0),
    // SPOT_PHOTO — 1420710 d=4/5/6 × e7/e8/e9
    ("1420710.png", 7, 4.0),
    ("1420710.png", 7, 5.0),
    ("1420710.png", 7, 6.0),
    ("1420710.png", 8, 4.0),
    ("1420710.png", 8, 5.0),
    ("1420710.png", 8, 6.0),
    ("1420710.png", 9, 4.0),
    ("1420710.png", 9, 5.0),
    ("1420710.png", 9, 6.0),
    // SPOT_PHOTO — 1531677 d=4/5/6 × e7/e8/e9
    ("1531677.png", 7, 4.0),
    ("1531677.png", 7, 5.0),
    ("1531677.png", 7, 6.0),
    ("1531677.png", 8, 4.0),
    ("1531677.png", 8, 5.0),
    ("1531677.png", 8, 6.0),
    ("1531677.png", 9, 4.0),
    ("1531677.png", 9, 5.0),
    ("1531677.png", 9, 6.0),
    // SPOT_PHOTO — 7552578 (nearest CONTROL by mask_p25 = 77.90, must NOT fire)
    ("7552578.png", 7, 4.0),
    ("7552578.png", 7, 5.0),
    ("7552578.png", 7, 6.0),
    ("7552578.png", 8, 4.0),
    ("7552578.png", 8, 5.0),
    ("7552578.png", 8, 6.0),
    ("7552578.png", 9, 4.0),
    ("7552578.png", 9, 5.0),
    ("7552578.png", 9, 6.0),
];

#[derive(Clone, Copy, Debug)]
enum Mode {
    A, // JXL_W44_150_PHOTO_DISABLE=1 (pre-W44-150 baseline — W44-118 gate only)
    B, // default (W44-150 photo admission active — production)
}

fn set_mode_env(mode: Mode) {
    unsafe {
        std::env::remove_var("JXL_W44_150_PHOTO_DISABLE");
        std::env::remove_var("JXL_W44_117_DISABLE");
    }
    match mode {
        Mode::A => unsafe { std::env::set_var("JXL_W44_150_PHOTO_DISABLE", "1") },
        Mode::B => {}
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
    eprintln!(
        "W44-150 Phase 2 A/B: A=W44-150 disabled (W44-118 baseline) / B=default (W44-150 admits 1418519-class)"
    );
    eprintln!("Cells: {}", CELLS.len());

    println!(
        "image\teffort\tdistance\tA_bytes\tB_bytes\tBA_pct\tA_bfly\tB_bfly\tA_ssim2\tB_ssim2\tBA_ssim2"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    let mut byte_identical_count = 0usize;
    let mut byte_differ_count = 0usize;
    let mut target_1418519_ssim2_d56_improvements = Vec::<f64>::new();

    for (i, &(image, effort, d)) in CELLS.iter().enumerate() {
        eprintln!("[{}/{}] {} e{} d={}", i + 1, n_cells, image, effort, d);

        let path = PathBuf::from(CID22).join(image);
        let cache_key = image.to_string();
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

        if let (Some(a), Some(b)) = (sa, sb) {
            let ba_pct = if a.bytes > 0 {
                100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64
            } else {
                0.0
            };
            let ba_s2 = b.ssim2 - a.ssim2;
            println!(
                "{}\te{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}",
                image,
                effort,
                d,
                a.bytes,
                b.bytes,
                ba_pct,
                a.butteraugli,
                b.butteraugli,
                a.ssim2,
                b.ssim2,
                ba_s2,
            );
            if a.bytes == b.bytes {
                byte_identical_count += 1;
            } else {
                byte_differ_count += 1;
            }
            if image == "1418519.png" && (d == 5.0 || d == 6.0) {
                target_1418519_ssim2_d56_improvements.push(ba_s2);
            }
        }
    }

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!(
        "Byte-identical cells (B==A): {} / {}",
        byte_identical_count, n_cells
    );
    eprintln!(
        "Byte-differ cells (B!=A): {} / {}",
        byte_differ_count, n_cells
    );
    if !target_1418519_ssim2_d56_improvements.is_empty() {
        let sum: f64 = target_1418519_ssim2_d56_improvements.iter().sum();
        let n = target_1418519_ssim2_d56_improvements.len();
        let mean = sum / n as f64;
        let min = target_1418519_ssim2_d56_improvements
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = target_1418519_ssim2_d56_improvements
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "1418519 d=5/6 SSIM2 improvement (B-A): n={} mean=+{:.4} min={:+.4} max={:+.4}",
            n, mean, min, max
        );
    }
}
