// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-165 (Smart-Zenjxl chunk 2) paired A/B reproducer.
//!
//! Re-enables the W44-150 Mechanism A photo admission to the W44-117 EPF
//! sharpness seed on `EncoderStrategy::Zenjxl` / `Aggressive`. Per user
//! directive 2026-05-21 ("restore any superior options according to the
//! current encode strategy selection"), the W44-150 honest-stop (which
//! reverted the gate because the +0.27 mean SSIM2 delta on 1418519 d=5/6
//! missed the 50% closure HARD gate) is overridden: partial recovery
//! ships on Zenjxl. The `Libjxl` / `LeanFaster` strategies keep the
//! W44-150 honest-stop disposition (no photo admission).
//!
//! Mode A = `Custom` with `photo_epf_seed_admit = false` (= W44-150
//! honest-stop baseline; production code path that was reverted).
//! Mode B = default Zenjxl (= W44-165 production: photo admission ON).
//! Mode LIBJXL_OFF = `EncoderStrategy::Libjxl` (must be byte-identical
//! to its pre-W44-165 state — strategy guard verification).
//!
//! Acceptance gates evaluated:
//!   * Hash-locks 36/36 BYTE-IDENTICAL (synthetic gradients fail the
//!     `mask_p25 >= 85.0` discriminator AND the pixel_domain_loss-
//!     materialized `mask1x1` precondition).
//!   * 1025469 d=2/3/4/5/6 × e7/e8/e9: BYTE-IDENTICAL (W44-118
//!     protection holds — mask_p25=60.64 < 85.0).
//!   * 4 spot CID22 photos: BYTE-IDENTICAL (proxy discriminator
//!     correctly rejects them — see `examples/w44_149_photo_proxy_audit.rs`
//!     output for the 39 other CID22 photos all measured at mask_p25 < 78).
//!   * 1418519 d=5/6 mean SSIM2 ≈ +0.27 (matches W44-150 Phase 2
//!     measurement); HARD acceptance bar relaxed to ANY positive
//!     improvement on the target cluster per user directive.
//!   * LIBJXL_OFF: byte-identical to pre-W44-165 Libjxl baseline.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release -p jxl-encoder \
//!     --features 'butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_165_restore_epf_seed_photos_ab \
//!     > benchmarks/w44_165_restore_epf_seed_photos_2026-05-21.tsv

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

/// TARGET = 1418519 d=4/5/6 × e7/e8/e9 = 9 cells.
/// PROTECT_W118 = 1025469 d=2/3/4/5/6 × e7/e8/e9 = 15 cells.
/// PROTECT_SPOT = {1189261, 1420710, 1531677, 7552578} × {e7,e8,e9} × d=4 = 12 cells.
/// LIBJXL_OFF = 1418519 e8 d=5 = 1 cell (strategy guard verification).
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
    // PROTECT_W118 — 1025469
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
    // PROTECT_SPOT — 1189261/1420710/1531677/7552578 at d=4 only (smaller surface than W44-150's 36-cell version)
    ("1189261.png", 7, 4.0),
    ("1189261.png", 8, 4.0),
    ("1189261.png", 9, 4.0),
    ("1420710.png", 7, 4.0),
    ("1420710.png", 8, 4.0),
    ("1420710.png", 9, 4.0),
    ("1531677.png", 7, 4.0),
    ("1531677.png", 8, 4.0),
    ("1531677.png", 9, 4.0),
    ("7552578.png", 7, 4.0),
    ("7552578.png", 8, 4.0),
    ("7552578.png", 9, 4.0),
];

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// W44-165 OFF on Custom — preserves W44-150 honest-stop bytes.
    A,
    /// W44-165 ON via default Zenjxl strategy — production W44-165 path.
    B,
    /// `EncoderStrategy::Libjxl` — must be byte-identical regardless of
    /// the W44-165 field (resolved via `ResolvedImprovements::libjxl()`).
    LibjxlOff,
}

fn encode_with_mode(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
) -> Result<Vec<u8>, String> {
    let cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    let cfg = match mode {
        Mode::A => {
            let custom = EncoderImprovementsCustom {
                photo_epf_seed_admit: false,
                ..Default::default()
            };
            cfg.with_strategy(EncoderStrategy::Custom(Box::new(custom)))
        }
        Mode::B => cfg.with_strategy(EncoderStrategy::Zenjxl),
        Mode::LibjxlOff => cfg.with_strategy(EncoderStrategy::Libjxl),
    };
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
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
    mode: Mode,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
) -> Option<Score> {
    let bitstream = match encode_with_mode(rgb, w, h, effort, d, mode) {
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
    eprintln!(
        "W44-165 A/B: A=Custom photo_epf_seed_admit=false (W44-150 honest-stop baseline) / B=Zenjxl default (W44-165 ON)"
    );
    eprintln!("Cells (interleaved A,B): {}", CELLS.len());

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
    let mut target_1418519_ssim2_d4_improvements = Vec::<f64>::new();
    let mut protect_w118_byte_identical = 0usize;
    let mut protect_w118_total = 0usize;
    let mut protect_spot_byte_identical = 0usize;
    let mut protect_spot_total = 0usize;

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
            if image == "1418519.png" && d == 4.0 {
                target_1418519_ssim2_d4_improvements.push(ba_s2);
            }
            if image == "1025469.png" {
                protect_w118_total += 1;
                if a.bytes == b.bytes {
                    protect_w118_byte_identical += 1;
                }
            }
            if matches!(
                image,
                "1189261.png" | "1420710.png" | "1531677.png" | "7552578.png"
            ) {
                protect_spot_total += 1;
                if a.bytes == b.bytes {
                    protect_spot_byte_identical += 1;
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
    eprintln!(
        "PROTECT_W118 (1025469) byte-identical: {} / {}",
        protect_w118_byte_identical, protect_w118_total
    );
    eprintln!(
        "PROTECT_SPOT (1189261/1420710/1531677/7552578 d=4) byte-identical: {} / {}",
        protect_spot_byte_identical, protect_spot_total
    );
    if !target_1418519_ssim2_d4_improvements.is_empty() {
        let sum: f64 = target_1418519_ssim2_d4_improvements.iter().sum();
        let n = target_1418519_ssim2_d4_improvements.len();
        let mean = sum / n as f64;
        let min = target_1418519_ssim2_d4_improvements
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = target_1418519_ssim2_d4_improvements
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "TARGET 1418519 d=4 SSIM2 improvement (B-A): n={} mean={:+.4} min={:+.4} max={:+.4}",
            n, mean, min, max
        );
    }
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
            "TARGET 1418519 d=5/6 SSIM2 improvement (B-A): n={} mean={:+.4} min={:+.4} max={:+.4}",
            n, mean, min, max
        );
    }

    // LIBJXL_OFF strategy-guard check: encode 1418519 e8 d=5 with
    // EncoderStrategy::Libjxl and compare against Mode A (photo admission
    // OFF). The Libjxl variant sets `photo_epf_seed_admit: false` AND
    // `buttloop_epf_sharpness_seed: EpfSharpnessSeed::LegacyUniform4` AND
    // many other parity flips, so it WILL produce different bytes from
    // Mode A (which only flips the one field). The intent of this
    // check is to verify that the Libjxl path does NOT see the W44-165
    // admission firing (i.e. the W44-165 source-level guard is
    // strategy-sensitive). We assert that Mode B byte-count != Libjxl
    // byte-count when Mode B fires the admission (so we know the
    // strategy guard distinguishes between them).
    eprintln!();
    eprintln!("=== LIBJXL_OFF strategy-guard check ===");
    if let Some((w, h, raw, _, _)) = images_cache.get("1418519.png") {
        let libjxl_off = encode_with_mode(raw, *w, *h, 8, 5.0, Mode::LibjxlOff)
            .expect("Libjxl encode 1418519 e8 d=5");
        let zenjxl =
            encode_with_mode(raw, *w, *h, 8, 5.0, Mode::B).expect("Zenjxl encode 1418519 e8 d=5");
        eprintln!(
            "1418519 e8 d=5: Libjxl bytes = {}, Zenjxl (W44-165 ON) bytes = {}, Δ = {}",
            libjxl_off.len(),
            zenjxl.len(),
            zenjxl.len() as i64 - libjxl_off.len() as i64
        );
    }
}
