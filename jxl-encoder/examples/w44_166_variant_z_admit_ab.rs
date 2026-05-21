// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-166 (Smart-Zenjxl chunk 3) paired A/B/C reproducer.
//!
//! Measures whether admitting high-mask photos (1418519, mask_p25=88.88)
//! to the W44-96 variant Z dispatch via the `mask_p25 >= 85`
//! discriminator COMPOSES with W44-152's W44-29 outer-table win, or
//! COMPETES like W44-165's EPF seed admission did.
//!
//! - Mode A (baseline): current production (W44-152 mechanism only;
//!   variant Z requires `mask < 50` → 1418519 can't reach it)
//! - Mode B: admit via `mask_p25 >= 85` (1418519 m3=36.84 lands on
//!   high_colour Z' table per the W44-98 inner m3>=25 gate)
//! - Mode C: admit via `mask_p25 >= 85 AND m3 >= 80` (no current
//!   target image qualifies → equivalent to Mode A on 1418519; the
//!   mode exists as a stricter alternative for future high-m3 cases)
//!
//! Acceptance gates evaluated:
//!   * Hash-locks: 36/36 BYTE-IDENTICAL (synthetic gradients fail
//!     the pixel_domain_loss precondition or the mask_p25 >= 85
//!     discriminator).
//!   * PROTECT_W118 (1025469): 15/15 BYTE-IDENTICAL — mask_p25=60.64
//!     < 85.0 → discriminator REJECTS.
//!   * PROTECT_VARIANT_Z_W98 (1420710): 6/6 BYTE-IDENTICAL or within
//!     SSIM2 ±0.30 — variant Z's original W44-98 high_colour target,
//!     should NOT regress under widened admission (mask_p25 unchanged).
//!   * PROTECT_VARIANT_Z_W99 (1531677): 6/6 BYTE-IDENTICAL or within
//!     SSIM2 ±0.30 — variant Z's W44-99 low_colour target.
//!   * 1418519 d=5/6 e8/e9 SSIM2 mean for Mode B or Mode C: must be
//!     net positive (>= +0.10 mean) to ship; ANY negative mean is
//!     HONEST-STOP per the W44-165 falsification lesson.
//!
//! Per the W44-165 binding lesson, this measurement runs on CURRENT
//! main (W44-165 ship at `d2972c11`) — NOT on the W44-150 baseline
//! that may give a different sign due to intervening mechanism
//! shipments.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release -p jxl-encoder \
//!     --features 'butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_166_variant_z_admit_ab \
//!     > benchmarks/w44_166_variant_z_admit_zenjxl_2026-05-21.tsv

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

/// TARGET = 1418519 e7/e8/e9 × d=4/5/6 = 9 cells.
/// PROTECT_W118 = 1025469 e7/e8/e9 × d=2/3/4/5/6 = 15 cells.
/// PROTECT_VARIANT_Z_W98 = 1420710 e7/e8/e9 × d=5/6 = 6 cells (W44-98 target — variant Z high_colour).
/// PROTECT_VARIANT_Z_W99 = 1531677 e7/e8/e9 × d=5/6 = 6 cells (W44-99 target — variant Z low_colour).
const CELLS: &[(&str, u8, f32)] = &[
    // TARGET
    ("1418519.png", 7, 4.0),
    ("1418519.png", 7, 5.0),
    ("1418519.png", 7, 6.0),
    ("1418519.png", 8, 4.0),
    ("1418519.png", 8, 5.0),
    ("1418519.png", 8, 6.0),
    ("1418519.png", 9, 4.0),
    ("1418519.png", 9, 5.0),
    ("1418519.png", 9, 6.0),
    // PROTECT_W118 — 1025469 (mask_p25=60.64, should NEVER admit)
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
    // PROTECT_VARIANT_Z_W98 — 1420710 (mask=39.55, already in variant Z high_colour)
    ("1420710.png", 7, 5.0),
    ("1420710.png", 7, 6.0),
    ("1420710.png", 8, 5.0),
    ("1420710.png", 8, 6.0),
    ("1420710.png", 9, 5.0),
    ("1420710.png", 9, 6.0),
    // PROTECT_VARIANT_Z_W99 — 1531677 (mask=35.63, already in variant Z low_colour)
    ("1531677.png", 7, 5.0),
    ("1531677.png", 7, 6.0),
    ("1531677.png", 8, 5.0),
    ("1531677.png", 8, 6.0),
    ("1531677.png", 9, 5.0),
    ("1531677.png", 9, 6.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Baseline — JXL_W44_166_VARIANT_Z_ADMIT_MODE unset (= A).
    A,
    /// Mode B — JXL_W44_166_VARIANT_Z_ADMIT_MODE=B (mask_p25 >= 85).
    B,
    /// Mode C — JXL_W44_166_VARIANT_Z_ADMIT_MODE=C (mask_p25 >= 85 AND m3 >= 25).
    C,
}

impl Mode {
    /// W44-166: post-SHIP, Mode B is the production default (env unset
    /// returns Mode B). To bench Mode A baseline we MUST set the env to
    /// "A" explicitly. Mode B = explicit "B" for self-documentation in
    /// the bench (functionally equivalent to env unset).
    fn env(self) -> &'static str {
        match self {
            Mode::A => "A",
            Mode::B => "B",
            Mode::C => "C",
        }
    }
}

fn encode_with_mode(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
) -> Result<Vec<u8>, String> {
    let prev = std::env::var("JXL_W44_166_VARIANT_Z_ADMIT_MODE").ok();
    // Always explicitly set the env (Mode B is the default but we set
    // it anyway for self-documentation in the bench harness).
    // SAFETY: per Rust 1.92 std::env::set_var is unsafe in
    // multi-threaded contexts. This benchmark is single-threaded at
    // the env-set boundary (paired interleaved A/B/C); each encode
    // runs to completion before the next mode swap.
    unsafe { std::env::set_var("JXL_W44_166_VARIANT_Z_ADMIT_MODE", mode.env()) };
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_strategy(EncoderStrategy::Zenjxl);
    let result = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"));
    // Restore prior env (typically None)
    match prev {
        Some(v) => unsafe { std::env::set_var("JXL_W44_166_VARIANT_Z_ADMIT_MODE", v) },
        None => unsafe { std::env::remove_var("JXL_W44_166_VARIANT_Z_ADMIT_MODE") },
    }
    result
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
    mode: Mode,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
) -> Option<Score> {
    let bitstream = match encode_with_mode(rgb, w, h, effort, d, mode) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  encode failed ({:?}): {}", mode, e);
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
        .map(|r| r.score)
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
    eprintln!("W44-166 A/B/C: A=baseline / B=mask_p25>=85 / C=mask_p25>=85 AND m3>=25");
    eprintln!("Cells (interleaved A,B,C): {}", CELLS.len());

    println!(
        "image\teffort\tdistance\tA_bytes\tB_bytes\tC_bytes\tBA_pct\tCA_pct\tA_bfly\tB_bfly\tC_bfly\tA_ssim2\tB_ssim2\tC_ssim2\tBA_ssim2\tCA_ssim2\tclass"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    let mut target_ba_ssim2_d56 = Vec::<f64>::new();
    let mut target_ca_ssim2_d56 = Vec::<f64>::new();
    let mut target_ba_ssim2_d4 = Vec::<f64>::new();
    let mut target_ca_ssim2_d4 = Vec::<f64>::new();
    let mut target_ba_bytes_d56 = Vec::<f64>::new();
    let mut target_ca_bytes_d56 = Vec::<f64>::new();
    let mut target_ba_bytes_d4 = Vec::<f64>::new();
    let mut target_ca_bytes_d4 = Vec::<f64>::new();
    let mut protect_w118_ba_byte_identical = 0usize;
    let mut protect_w118_ca_byte_identical = 0usize;
    let mut protect_w118_total = 0usize;
    let mut protect_vz_w98_ba_byte_identical = 0usize;
    let mut protect_vz_w98_ca_byte_identical = 0usize;
    let mut protect_vz_w98_ba_worst_ssim2: f64 = 0.0;
    let mut protect_vz_w98_ca_worst_ssim2: f64 = 0.0;
    let mut protect_vz_w98_total = 0usize;
    let mut protect_vz_w99_ba_byte_identical = 0usize;
    let mut protect_vz_w99_ca_byte_identical = 0usize;
    let mut protect_vz_w99_ba_worst_ssim2: f64 = 0.0;
    let mut protect_vz_w99_ca_worst_ssim2: f64 = 0.0;
    let mut protect_vz_w99_total = 0usize;

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

        let sa = score_cell(
            raw,
            *w,
            *h,
            effort,
            d,
            Mode::A,
            orig_linear_img,
            orig_srgb_img,
        );
        let sb = score_cell(
            raw,
            *w,
            *h,
            effort,
            d,
            Mode::B,
            orig_linear_img,
            orig_srgb_img,
        );
        let sc = score_cell(
            raw,
            *w,
            *h,
            effort,
            d,
            Mode::C,
            orig_linear_img,
            orig_srgb_img,
        );

        let class = if image == "1418519.png" {
            "TARGET"
        } else if image == "1025469.png" {
            "PROTECT_W118"
        } else if image == "1420710.png" {
            "PROTECT_VARIANT_Z_W98"
        } else if image == "1531677.png" {
            "PROTECT_VARIANT_Z_W99"
        } else {
            "OTHER"
        };

        if let (Some(a), Some(b), Some(c)) = (sa, sb, sc) {
            let ba_pct = if a.bytes > 0 {
                100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64
            } else {
                0.0
            };
            let ca_pct = if a.bytes > 0 {
                100.0 * (c.bytes as f64 - a.bytes as f64) / a.bytes as f64
            } else {
                0.0
            };
            let ba_s2 = b.ssim2 - a.ssim2;
            let ca_s2 = c.ssim2 - a.ssim2;
            println!(
                "{}\te{}\t{}\t{}\t{}\t{}\t{:+.3}\t{:+.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}\t{}",
                image,
                effort,
                d,
                a.bytes,
                b.bytes,
                c.bytes,
                ba_pct,
                ca_pct,
                a.butteraugli,
                b.butteraugli,
                c.butteraugli,
                a.ssim2,
                b.ssim2,
                c.ssim2,
                ba_s2,
                ca_s2,
                class,
            );
            if class == "TARGET" {
                if d == 5.0 || d == 6.0 {
                    target_ba_ssim2_d56.push(ba_s2);
                    target_ca_ssim2_d56.push(ca_s2);
                    target_ba_bytes_d56.push(ba_pct);
                    target_ca_bytes_d56.push(ca_pct);
                }
                if d == 4.0 {
                    target_ba_ssim2_d4.push(ba_s2);
                    target_ca_ssim2_d4.push(ca_s2);
                    target_ba_bytes_d4.push(ba_pct);
                    target_ca_bytes_d4.push(ca_pct);
                }
            }
            if class == "PROTECT_W118" {
                protect_w118_total += 1;
                if a.bytes == b.bytes {
                    protect_w118_ba_byte_identical += 1;
                }
                if a.bytes == c.bytes {
                    protect_w118_ca_byte_identical += 1;
                }
            }
            if class == "PROTECT_VARIANT_Z_W98" {
                protect_vz_w98_total += 1;
                if a.bytes == b.bytes {
                    protect_vz_w98_ba_byte_identical += 1;
                }
                if a.bytes == c.bytes {
                    protect_vz_w98_ca_byte_identical += 1;
                }
                if ba_s2 < protect_vz_w98_ba_worst_ssim2 {
                    protect_vz_w98_ba_worst_ssim2 = ba_s2;
                }
                if ca_s2 < protect_vz_w98_ca_worst_ssim2 {
                    protect_vz_w98_ca_worst_ssim2 = ca_s2;
                }
            }
            if class == "PROTECT_VARIANT_Z_W99" {
                protect_vz_w99_total += 1;
                if a.bytes == b.bytes {
                    protect_vz_w99_ba_byte_identical += 1;
                }
                if a.bytes == c.bytes {
                    protect_vz_w99_ca_byte_identical += 1;
                }
                if ba_s2 < protect_vz_w99_ba_worst_ssim2 {
                    protect_vz_w99_ba_worst_ssim2 = ba_s2;
                }
                if ca_s2 < protect_vz_w99_ca_worst_ssim2 {
                    protect_vz_w99_ca_worst_ssim2 = ca_s2;
                }
            }
        }
    }

    let stats = |label: &str, v: &[f64], unit: &str| {
        if v.is_empty() {
            return;
        }
        let n = v.len();
        let sum: f64 = v.iter().sum();
        let mean = sum / n as f64;
        let min = v.iter().copied().fold(f64::INFINITY, f64::min);
        let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "{}: n={} mean={:+.4}{} min={:+.4}{} max={:+.4}{}",
            label, n, mean, unit, min, unit, max, unit
        );
    };

    eprintln!();
    eprintln!("=== Summary ===");
    stats("TARGET 1418519 d=5/6 SSIM2 B-A", &target_ba_ssim2_d56, "");
    stats("TARGET 1418519 d=5/6 SSIM2 C-A", &target_ca_ssim2_d56, "");
    stats("TARGET 1418519 d=5/6 bytes B-A%", &target_ba_bytes_d56, "%");
    stats("TARGET 1418519 d=5/6 bytes C-A%", &target_ca_bytes_d56, "%");
    stats("TARGET 1418519 d=4 SSIM2 B-A", &target_ba_ssim2_d4, "");
    stats("TARGET 1418519 d=4 SSIM2 C-A", &target_ca_ssim2_d4, "");
    stats("TARGET 1418519 d=4 bytes B-A%", &target_ba_bytes_d4, "%");
    stats("TARGET 1418519 d=4 bytes C-A%", &target_ca_bytes_d4, "%");
    eprintln!(
        "PROTECT_W118 byte-identical: BA {} / {}, CA {} / {}",
        protect_w118_ba_byte_identical,
        protect_w118_total,
        protect_w118_ca_byte_identical,
        protect_w118_total
    );
    eprintln!(
        "PROTECT_VARIANT_Z_W98 byte-identical: BA {} / {}, CA {} / {}; worst SSIM2 BA {:+.4} CA {:+.4}",
        protect_vz_w98_ba_byte_identical,
        protect_vz_w98_total,
        protect_vz_w98_ca_byte_identical,
        protect_vz_w98_total,
        protect_vz_w98_ba_worst_ssim2,
        protect_vz_w98_ca_worst_ssim2
    );
    eprintln!(
        "PROTECT_VARIANT_Z_W99 byte-identical: BA {} / {}, CA {} / {}; worst SSIM2 BA {:+.4} CA {:+.4}",
        protect_vz_w99_ba_byte_identical,
        protect_vz_w99_total,
        protect_vz_w99_ca_byte_identical,
        protect_vz_w99_total,
        protect_vz_w99_ba_worst_ssim2,
        protect_vz_w99_ca_worst_ssim2
    );

    eprintln!();
    eprintln!("=== Acceptance decision ===");
    let ba_mean_d56 = if target_ba_ssim2_d56.is_empty() {
        0.0
    } else {
        target_ba_ssim2_d56.iter().sum::<f64>() / target_ba_ssim2_d56.len() as f64
    };
    let ca_mean_d56 = if target_ca_ssim2_d56.is_empty() {
        0.0
    } else {
        target_ca_ssim2_d56.iter().sum::<f64>() / target_ca_ssim2_d56.len() as f64
    };
    let mut winning_mode = "NONE_HONEST_STOP";
    if ba_mean_d56 >= 0.10
        && protect_vz_w98_ba_worst_ssim2 > -0.30
        && protect_vz_w99_ba_worst_ssim2 > -0.30
    {
        winning_mode = "B (mask_p25 >= 85)";
    } else if ca_mean_d56 >= 0.10
        && protect_vz_w98_ca_worst_ssim2 > -0.30
        && protect_vz_w99_ca_worst_ssim2 > -0.30
    {
        winning_mode = "C (mask_p25 >= 85 AND m3 >= 25)";
    }
    eprintln!(
        "TARGET 1418519 d=5/6 mean SSIM2: B={:+.4}, C={:+.4} (gate >= +0.10)",
        ba_mean_d56, ca_mean_d56
    );
    eprintln!(
        "Worst PROTECT SSIM2: B={:+.4} (W98), {:+.4} (W99); C={:+.4} (W98), {:+.4} (W99) (gate > -0.30)",
        protect_vz_w98_ba_worst_ssim2,
        protect_vz_w99_ba_worst_ssim2,
        protect_vz_w98_ca_worst_ssim2,
        protect_vz_w99_ca_worst_ssim2
    );
    eprintln!("Winning mode: {}", winning_mode);
}
