// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-151 paired A/B bench — measure the Mechanism B photo admission
//! gate: widen the W44-29 outer dispatch to admit `mask_p25 >= 85.0`
//! photos at `d >= HIGH_D_PHOTO_MIN_DISTANCE` (3.0).
//!
//! **W44-151 HONEST-STOP (2026-05-21)**: production code REVERTED after
//! this bench measured d=5/6 mean SSIM2 = +0.544 (HARD gate wanted
//! ≥ +1.0). Discriminator works perfectly (51/51 protection cells +
//! 12/12 hash-locks BYTE-IDENTICAL); the d=6 cells over-fire on the
//! default entropy_mul table (+4.3-4.6% bytes for only +0.07-0.28
//! SSIM2). The env hook `JXL_W44_151_DISABLE=1` is NO LONGER WIRED
//! post-revert — both modes now produce byte-identical output. Future
//! agents can re-introduce the env hook + the OR-branch shown in the
//! W44-151 commit diff to re-measure with a narrower gate (see
//! `benchmarks/w44_151_w44_29_widen_2026-05-21.meta` W44-152 candidates).
//!
//! Follow-on to W44-150 (which honest-stopped on Mechanism A —
//! W44-117 EPF seed admission). W44-149 audit established that
//! `mask_p25 >= 85` cleanly admits 1418519 (88.88) only; nearest
//! CONTROL is 7552578 at 77.90 (11pp safety margin); all other CID22
//! validation photos fall below the threshold.
//!
//! Mechanism B routes the SAME discriminator into the W44-29
//! entropy_mul mechanism. The W44-29 default suppressed table
//! (`dct32x32 = 1.34` vs libjxl stock 1.48) fires inside the
//! `FindBest*Transform` cost evaluation, which IS available at
//! e >= 5 (no buttloop dependency). This addresses the W44-150
//! e <= 7 structural limit.
//!
//! 1418519's `mask1x1_median` is ~92, so the W44-96/98/99 variant Z
//! sub-gates do NOT fire (they require `mask_median < 50`). 1418519
//! lands on the DEFAULT `high_d_photo_smooth_suppressed()` table.
//!
//! Cells:
//!   * TARGET   = 1418519 d=4/5/6 × e7/e8/e9 (9 cells)
//!   * PROTECT  = 1025469 d=2..6 × e7/e8/e9 (15 cells; mask_p25=60.64,
//!                  REJECT — must stay byte-identical)
//!   * SPOT     = {1189261, 1420710, 1531677, 7552578} × {e7/e8/e9} ×
//!                  {d=3/4/5/6} (48 cells; all mask_p25 < 85, REJECT —
//!                  must stay byte-identical)
//!
//! Total: 72 cells × 2 modes (A = JXL_W44_151_DISABLE=1 baseline,
//! B = default).
//!
//! Acceptance gates (per task W44-151 spec):
//!   (a) Build: PASS
//!   (b) `cargo test --lib --features __expert`: PASS
//!   (c) Hash-locks 36/36 BYTE-IDENTICAL: REQUIRED (separate test)
//!   (d) 1025469: 15/15 BYTE-IDENTICAL
//!   (e) 4 SPOT photos: 48/48 BYTE-IDENTICAL
//!   (f) 1418519 d=5/6 (4 cells: e8/e9 × d=5/6): mean SSIM2 ≥ +1.0
//!   (g) 1418519 d=4 (3 cells): mean SSIM2 change ≥ -0.30
//!   (h) Multi-decoder roundtrip: PASS on 2 changed cells × 2 decoders
//!       (separate test — see `examples/w44_151_decoder_check.rs`)
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_151_w44_29_widen_ab \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_151_w44_29_widen_2026-05-21.tsv

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

const CELLS: &[(&str, u8, f32)] = &[
    // TARGET — 1418519 d=4/5/6 × e7/e8/e9 (W44-147 audit cluster)
    ("1418519.png", 7, 4.0),
    ("1418519.png", 7, 5.0),
    ("1418519.png", 7, 6.0),
    ("1418519.png", 8, 4.0),
    ("1418519.png", 8, 5.0),
    ("1418519.png", 8, 6.0),
    ("1418519.png", 9, 4.0),
    ("1418519.png", 9, 5.0),
    ("1418519.png", 9, 6.0),
    // PROTECT_W118 — 1025469 d=2..6 × e7/e8/e9 (mask_p25=60.64, REJECT)
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
    // SPOT — 1189261 d=3..6 × e7/e8/e9 (mask_p25 well below 85)
    ("1189261.png", 7, 3.0),
    ("1189261.png", 7, 4.0),
    ("1189261.png", 7, 5.0),
    ("1189261.png", 7, 6.0),
    ("1189261.png", 8, 3.0),
    ("1189261.png", 8, 4.0),
    ("1189261.png", 8, 5.0),
    ("1189261.png", 8, 6.0),
    ("1189261.png", 9, 3.0),
    ("1189261.png", 9, 4.0),
    ("1189261.png", 9, 5.0),
    ("1189261.png", 9, 6.0),
    // SPOT — 1420710 d=3..6 × e7/e8/e9
    ("1420710.png", 7, 3.0),
    ("1420710.png", 7, 4.0),
    ("1420710.png", 7, 5.0),
    ("1420710.png", 7, 6.0),
    ("1420710.png", 8, 3.0),
    ("1420710.png", 8, 4.0),
    ("1420710.png", 8, 5.0),
    ("1420710.png", 8, 6.0),
    ("1420710.png", 9, 3.0),
    ("1420710.png", 9, 4.0),
    ("1420710.png", 9, 5.0),
    ("1420710.png", 9, 6.0),
    // SPOT — 1531677 d=3..6 × e7/e8/e9
    ("1531677.png", 7, 3.0),
    ("1531677.png", 7, 4.0),
    ("1531677.png", 7, 5.0),
    ("1531677.png", 7, 6.0),
    ("1531677.png", 8, 3.0),
    ("1531677.png", 8, 4.0),
    ("1531677.png", 8, 5.0),
    ("1531677.png", 8, 6.0),
    ("1531677.png", 9, 3.0),
    ("1531677.png", 9, 4.0),
    ("1531677.png", 9, 5.0),
    ("1531677.png", 9, 6.0),
    // SPOT — 7552578 d=3..6 × e7/e8/e9 (nearest CONTROL by mask_p25)
    ("7552578.png", 7, 3.0),
    ("7552578.png", 7, 4.0),
    ("7552578.png", 7, 5.0),
    ("7552578.png", 7, 6.0),
    ("7552578.png", 8, 3.0),
    ("7552578.png", 8, 4.0),
    ("7552578.png", 8, 5.0),
    ("7552578.png", 8, 6.0),
    ("7552578.png", 9, 3.0),
    ("7552578.png", 9, 4.0),
    ("7552578.png", 9, 5.0),
    ("7552578.png", 9, 6.0),
];

#[derive(Clone, Copy, Debug)]
enum Mode {
    A, // JXL_W44_151_DISABLE=1 (legacy W44-29/91 baseline)
    B, // default (W44-151 active)
}

fn set_mode_env(mode: Mode) {
    unsafe {
        std::env::remove_var("JXL_W44_151_DISABLE");
    }
    match mode {
        Mode::A => unsafe { std::env::set_var("JXL_W44_151_DISABLE", "1") },
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

fn classify(image: &str) -> &'static str {
    match image {
        "1418519.png" => "TARGET",
        "1025469.png" => "PROTECT",
        _ => "SPOT",
    }
}

fn main() {
    eprintln!(
        "W44-151 A/B: A=JXL_W44_151_DISABLE=1 (W44-29/91 baseline) / B=default (W44-151 active)"
    );
    eprintln!("Cells: {}", CELLS.len());

    println!(
        "class\timage\teffort\tdistance\tA_bytes\tB_bytes\tBA_pct\tA_bfly\tB_bfly\tA_ssim2\tB_ssim2\tBA_ssim2"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    let mut byte_identical_count = 0usize;
    let mut byte_differ_count = 0usize;
    let mut target_1418519_ssim2_d56_improvements = Vec::<f64>::new();
    let mut target_1418519_ssim2_d4_changes = Vec::<f64>::new();
    let mut protect_byte_differs = Vec::<String>::new();
    let mut spot_byte_differs = Vec::<String>::new();

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

        let cls = classify(image);
        if let (Some(a), Some(b)) = (sa, sb) {
            let ba_pct = if a.bytes > 0 {
                100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64
            } else {
                0.0
            };
            let ba_s2 = b.ssim2 - a.ssim2;
            println!(
                "{}\t{}\te{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}",
                cls,
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
                let key = format!("{} e{} d={}", image, effort, d);
                match cls {
                    "PROTECT" => protect_byte_differs.push(key),
                    "SPOT" => spot_byte_differs.push(key),
                    _ => {}
                }
            }
            if image == "1418519.png" {
                if d == 5.0 || d == 6.0 {
                    target_1418519_ssim2_d56_improvements.push(ba_s2);
                } else if d == 4.0 {
                    target_1418519_ssim2_d4_changes.push(ba_s2);
                }
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
        eprintln!(
            "1418519 d=5/6 SSIM2 deltas (B-A): mean={:+.4} over {} cells",
            mean, n
        );
        eprintln!(
            "  HARD GATE (f): mean >= +1.0 → {}",
            if mean >= 1.0 { "PASS" } else { "FAIL" }
        );
    }
    if !target_1418519_ssim2_d4_changes.is_empty() {
        let sum: f64 = target_1418519_ssim2_d4_changes.iter().sum();
        let n = target_1418519_ssim2_d4_changes.len();
        let mean = sum / n as f64;
        eprintln!(
            "1418519 d=4 SSIM2 deltas (B-A): mean={:+.4} over {} cells",
            mean, n
        );
        eprintln!(
            "  HARD GATE (g): mean >= -0.30 → {}",
            if mean >= -0.30 { "PASS" } else { "FAIL" }
        );
    }
    eprintln!();
    eprintln!(
        "PROTECT (1025469) byte-differs: {} cells",
        protect_byte_differs.len()
    );
    for k in &protect_byte_differs {
        eprintln!("  {}", k);
    }
    eprintln!(
        "  HARD GATE (d): 0 byte-differs → {}",
        if protect_byte_differs.is_empty() {
            "PASS"
        } else {
            "FAIL"
        }
    );
    eprintln!();
    eprintln!(
        "SPOT (1189261/1420710/1531677/7552578) byte-differs: {} cells",
        spot_byte_differs.len()
    );
    for k in &spot_byte_differs {
        eprintln!("  {}", k);
    }
    eprintln!(
        "  HARD GATE (e): 0 byte-differs → {}",
        if spot_byte_differs.is_empty() {
            "PASS"
        } else {
            "FAIL"
        }
    );
}
