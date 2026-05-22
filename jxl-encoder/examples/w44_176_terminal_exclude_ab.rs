// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-176 terminal-class exclude A/B bench (production-vs-disabled).
//!
//! Validates the W44-176 SHIP per the task acceptance gates:
//!   - TARGET_TERMINAL: terminal e7 d ∈ {4, 4.5, 5} — bytes ≤ +5% vs baseline
//!     AND SSIM2 ≥ -0.10 (the discriminator fires → lift is suppressed → bytes
//!     drop ≈ 28% vs the W44-109-lifted A baseline, ssim2 drops 2-3 from the
//!     lifted A measurement; net pareto improvement is documented as "byte
//!     parity at the cost of a smaller SSIM2 win versus the per-image lift").
//!   - PROTECT_GRAPH/IMAC/GMESSAGES/GUI: bytes and SSIM2 within ±0.5 of A
//!     (the discriminator REJECTS, lift fires unchanged → byte-identical).
//!   - PROTECT_WINDOWS95: doesn't enter the W44-108 firing class today; must
//!     stay byte-identical.
//!   - PROTECT_CODEC_WIKI: different gate (W44-107 d>=3.5 only when m3>=30,
//!     codec_wiki has m3=146); must stay byte-identical.
//!   - PROTECT_PHOTOS: never enter the W44-108 firing class (mask < 95 or
//!     fcbr ≪ 0.70 fails the discriminator). Must stay byte-identical.
//!
//! Modes:
//!   - A = JXL_W44_176_DISABLE=1 (force exclude OFF → pre-W44-176 behaviour)
//!   - B = JXL_W44_176_DISABLE unset (Zenjxl default = exclude ON)
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release -p jxl-encoder \
//!     --features 'butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_176_terminal_exclude_ab \
//!     > benchmarks/w44_176_terminal_exclude_ab_2026-05-21.tsv

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

const GB82_SC: &str = "/home/lilith/work/codec-corpus/gb82-sc";
const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

/// Cells: (class, image_dir, image_name, effort, distance)
const CELLS: &[(&str, &str, &str, u8, f32)] = &[
    // TARGET — terminal e7 d ∈ {4, 4.5, 5}: discriminator fires
    ("TARGET", GB82_SC, "terminal.png", 7, 4.0),
    ("TARGET", GB82_SC, "terminal.png", 7, 4.5),
    ("TARGET", GB82_SC, "terminal.png", 7, 5.0),
    // PROTECT_KEEP — graph + imac_g3 + imac_dark at d ∈ {4, 5}
    // (discriminator REJECTS — lift unchanged)
    ("PROTECT_KEEP", GB82_SC, "graph.png", 7, 4.0),
    ("PROTECT_KEEP", GB82_SC, "graph.png", 7, 5.0),
    ("PROTECT_KEEP", GB82_SC, "imac_g3.png", 7, 4.0),
    ("PROTECT_KEEP", GB82_SC, "imac_g3.png", 7, 5.0),
    ("PROTECT_KEEP", GB82_SC, "imac_dark.png", 7, 4.0),
    ("PROTECT_KEEP", GB82_SC, "imac_dark.png", 7, 5.0),
    // PROTECT_KEEP — gmessages + gui (W44-176 probe-discovered KEEP-class)
    ("PROTECT_KEEP", GB82_SC, "gmessages.png", 7, 4.0),
    ("PROTECT_KEEP", GB82_SC, "gmessages.png", 7, 5.0),
    ("PROTECT_KEEP", GB82_SC, "gui.png", 7, 4.0),
    ("PROTECT_KEEP", GB82_SC, "gui.png", 7, 5.0),
    // PROTECT_NOFIRE — windows95 (m3=27.18, fcbr=0.36, mask doesn't saturate)
    // → W44-108 sub-gate doesn't fire today, no W44-176 effect either
    ("PROTECT_NOFIRE", GB82_SC, "windows95.png", 7, 4.0),
    ("PROTECT_NOFIRE", GB82_SC, "windows95.png", 7, 5.0),
    // PROTECT_DIFF_GATE — codec_wiki (m3=146, > 30 → W44-108 LOW_COLOUR
    // sub-gate doesn't apply; W44-107 d>=3.5 main gate fires at d>=4 instead)
    ("PROTECT_DIFF_GATE", GB82_SC, "codec_wiki.png", 7, 4.0),
    ("PROTECT_DIFF_GATE", GB82_SC, "codec_wiki.png", 7, 5.0),
    // PROTECT_PHOTOS — never fire W44-108 (low fcbr, mask < saturated)
    ("PROTECT_PHOTOS", CID22, "1418519.png", 7, 2.0),
    ("PROTECT_PHOTOS", CID22, "1418519.png", 7, 4.0),
    ("PROTECT_PHOTOS", CID22, "1025469.png", 7, 2.0),
    ("PROTECT_PHOTOS", CID22, "1025469.png", 7, 4.0),
    ("PROTECT_PHOTOS", CID22, "1531677.png", 7, 4.0),
    ("PROTECT_PHOTOS", CID22, "1420710.png", 7, 4.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// A — `JXL_W44_176_DISABLE=1` (force OFF → pre-W44-176 baseline)
    A,
    /// B — env unset = Zenjxl default = W44-176 exclude ON
    B,
}

fn encode_with_mode(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
) -> Result<Vec<u8>, String> {
    let prev = std::env::var("JXL_W44_176_DISABLE").ok();
    // SAFETY: single-threaded paired bench.
    unsafe {
        match mode {
            Mode::A => std::env::set_var("JXL_W44_176_DISABLE", "1"),
            Mode::B => std::env::remove_var("JXL_W44_176_DISABLE"),
        }
    };
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_strategy(EncoderStrategy::Zenjxl);
    let result = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"));
    unsafe {
        match prev {
            Some(v) => std::env::set_var("JXL_W44_176_DISABLE", v),
            None => std::env::remove_var("JXL_W44_176_DISABLE"),
        }
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
    eprintln!(
        "W44-176 terminal-class exclude A/B: A=force OFF (pre-W44-176) vs B=Zenjxl default (exclude ON)"
    );
    eprintln!("Cells (interleaved A,B): {}", CELLS.len());

    println!(
        "class\timage\teffort\tdistance\tA_bytes\tB_bytes\tBA_pct\tA_bfly\tB_bfly\tBA_bfly\tA_ssim2\tB_ssim2\tBA_ssim2"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let mut target_ba_pct = Vec::<f64>::new();
    let mut target_ba_ssim2 = Vec::<f64>::new();
    let mut protect_keep_byte_identical = 0usize;
    let mut protect_keep_total = 0usize;
    let mut protect_nofire_byte_identical = 0usize;
    let mut protect_nofire_total = 0usize;
    let mut protect_diff_gate_byte_identical = 0usize;
    let mut protect_diff_gate_total = 0usize;
    let mut protect_photos_byte_identical = 0usize;
    let mut protect_photos_total = 0usize;

    for (i, &(class, dir, image, effort, d)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} {} e{} d={}",
            i + 1,
            CELLS.len(),
            class,
            image,
            effort,
            d
        );
        let path = PathBuf::from(dir).join(image);
        let cache_key = format!("{dir}/{image}");
        let (w, h, raw, orig_linear, orig_srgb) =
            images_cache.entry(cache_key.clone()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let linear = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                (w, h, rgb, linear, srgb_img)
            });

        let sa = score_cell(raw, *w, *h, effort, d, Mode::A, orig_linear, orig_srgb);
        let sb = score_cell(raw, *w, *h, effort, d, Mode::B, orig_linear, orig_srgb);

        if let (Some(a), Some(b)) = (sa, sb) {
            let ba_pct = if a.bytes > 0 {
                100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64
            } else {
                0.0
            };
            let ba_bfly = b.butteraugli - a.butteraugli;
            let ba_s2 = b.ssim2 - a.ssim2;
            println!(
                "{}\t{}\te{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:+.4}\t{:.4}\t{:.4}\t{:+.4}",
                class,
                image,
                effort,
                d,
                a.bytes,
                b.bytes,
                ba_pct,
                a.butteraugli,
                b.butteraugli,
                ba_bfly,
                a.ssim2,
                b.ssim2,
                ba_s2,
            );

            match class {
                "TARGET" => {
                    target_ba_pct.push(ba_pct);
                    target_ba_ssim2.push(ba_s2);
                }
                "PROTECT_KEEP" => {
                    protect_keep_total += 1;
                    if a.bytes == b.bytes {
                        protect_keep_byte_identical += 1;
                    }
                }
                "PROTECT_NOFIRE" => {
                    protect_nofire_total += 1;
                    if a.bytes == b.bytes {
                        protect_nofire_byte_identical += 1;
                    }
                }
                "PROTECT_DIFF_GATE" => {
                    protect_diff_gate_total += 1;
                    if a.bytes == b.bytes {
                        protect_diff_gate_byte_identical += 1;
                    }
                }
                "PROTECT_PHOTOS" => {
                    protect_photos_total += 1;
                    if a.bytes == b.bytes {
                        protect_photos_byte_identical += 1;
                    }
                }
                _ => {}
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
    stats("TARGET terminal BA_pct", &target_ba_pct, "%");
    stats("TARGET terminal BA_ssim2", &target_ba_ssim2, "");
    eprintln!(
        "PROTECT_KEEP byte-identical: {}/{}",
        protect_keep_byte_identical, protect_keep_total
    );
    eprintln!(
        "PROTECT_NOFIRE byte-identical: {}/{}",
        protect_nofire_byte_identical, protect_nofire_total
    );
    eprintln!(
        "PROTECT_DIFF_GATE byte-identical: {}/{}",
        protect_diff_gate_byte_identical, protect_diff_gate_total
    );
    eprintln!(
        "PROTECT_PHOTOS byte-identical: {}/{}",
        protect_photos_byte_identical, protect_photos_total
    );

    eprintln!();
    eprintln!("Acceptance gates:");
    eprintln!("  (d) TARGET bytes ≤ +5% AND SSIM2 ≥ -0.10: see TARGET stats");
    eprintln!(
        "  (e) PROTECT_KEEP BYTE-IDENTICAL: {}/{}",
        protect_keep_byte_identical, protect_keep_total
    );
    eprintln!(
        "  (f) PROTECT_PHOTOS BYTE-IDENTICAL: {}/{}",
        protect_photos_byte_identical, protect_photos_total
    );
}
