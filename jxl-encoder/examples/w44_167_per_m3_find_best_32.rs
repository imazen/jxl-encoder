// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-167 (Smart-Zenjxl chunk 4) paired A/B/C/D reproducer.
//!
//! Measures whether the W44-94 honest-stopped find_best_32x32_transform
//! widening can ship under the W44-98/99 per-m3 sub-discriminator
//! pattern. W44-94 ruled out the GLOBAL OUTER widening because 1420710
//! (high-m3) benefitted but 1531677 (low-m3) regressed.
//!
//! This chunk tests whether the same lift, applied at the INNER variant
//! Z layer (where the m3 split already separates HC vs LC), can close
//! the 1420710 OPEN cluster without regressing 1531677.
//!
//! Modes:
//! - A (Baseline): JXL_W44_167_MODE unset (or =A). Byte-identical to
//!   pre-W44-167 main. Bench reference.
//! - B (GlobalLift): JXL_W44_167_MODE=B. Replay W44-94 X variant
//!   (dct16x32=1.40) on BOTH HC and LC variant Z tables. Hypothesis:
//!   stronger dct32x32 base (1.22 in variant Z vs 1.34 in OUTER) might
//!   change the W44-94 regression sign.
//! - C (HighM3Only): JXL_W44_167_MODE=C. Lift ONLY the HC table
//!   (m3>=25 → 1420710). LC stays at dct16x32=1.23.
//! - D (PerM3Split): JXL_W44_167_MODE=D. HC dct16x32=1.40, LC
//!   dct16x32=1.26 (mild lift), Z (none-of-the-above) dct16x32=1.22.
//!
//! Acceptance gates:
//! - (a) Build PASS
//! - (b) `cargo test --lib`: PASS
//! - (c) Hash-locks 36/36 BYTE-IDENTICAL with default Mode A
//! - (d) TARGET_1420710 d=5: ≥+0.10 SSIM2 mean AND bytes within ±3.0%
//! - (e) TARGET_1531677 d=5: NO SSIM2 regression > 0.30 on the chosen variant
//! - (f) PROTECT_W166_1418519: byte-identical OR SSIM2 ±0.10
//! - (g) PROTECT_W164_screenshots: BYTE-IDENTICAL (variant Z gate doesn't fire)
//! - (h) CONTROL_NOGATE: BYTE-IDENTICAL
//! - (i) EncoderStrategy::Libjxl: BYTE-IDENTICAL regardless of env
//! - (j) Multi-decoder PASS
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release -p jxl-encoder \
//!     --features 'butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_167_per_m3_find_best_32 \
//!     > benchmarks/w44_167_per_m3_find_best_32_2026-05-21.tsv

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

/// Cells:
/// - TARGET_1420710: 9 cells (e7/e8/e9 × d=4/5/6) — primary target
/// - TARGET_1531677: 9 cells (e7/e8/e9 × d=4/5/6) — must not regress
/// - PROTECT_W166_1418519: 9 cells (e7/e8/e9 × d=4/5/6) — W44-166 win must hold
/// - PROTECT_W164_screenshots: 3 screenshot cells (variant Z doesn't fire — must stay byte-identical)
/// - CONTROL_NOGATE: 1189261 + 1025469 at e7 d=4 — gate doesn't fire (mask>=50)
const CELLS: &[(&str, u8, f32)] = &[
    // TARGET_1420710 (mask=39.55, m3=32.93 → HC variant Z')
    ("1420710.png", 7, 4.0),
    ("1420710.png", 7, 5.0),
    ("1420710.png", 7, 6.0),
    ("1420710.png", 8, 4.0),
    ("1420710.png", 8, 5.0),
    ("1420710.png", 8, 6.0),
    ("1420710.png", 9, 4.0),
    ("1420710.png", 9, 5.0),
    ("1420710.png", 9, 6.0),
    // TARGET_1531677 (mask=35.63, m3=12.30 → LC variant Z'')
    ("1531677.png", 7, 4.0),
    ("1531677.png", 7, 5.0),
    ("1531677.png", 7, 6.0),
    ("1531677.png", 8, 4.0),
    ("1531677.png", 8, 5.0),
    ("1531677.png", 8, 6.0),
    ("1531677.png", 9, 4.0),
    ("1531677.png", 9, 5.0),
    ("1531677.png", 9, 6.0),
    // PROTECT_W166_1418519 (mask=92, m3=36.84 → admitted to HC via W44-166)
    ("1418519.png", 7, 4.0),
    ("1418519.png", 7, 5.0),
    ("1418519.png", 7, 6.0),
    ("1418519.png", 8, 4.0),
    ("1418519.png", 8, 5.0),
    ("1418519.png", 8, 6.0),
    ("1418519.png", 9, 4.0),
    ("1418519.png", 9, 5.0),
    ("1418519.png", 9, 6.0),
    // CONTROL_NOGATE (mask >= 50 — should NEVER touch the variant Z dispatch)
    ("1189261.png", 7, 4.0),
    ("1025469.png", 7, 4.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    A,
    B,
    C,
    D,
}

impl Mode {
    fn env(self) -> &'static str {
        match self {
            Mode::A => "A",
            Mode::B => "B",
            Mode::C => "C",
            Mode::D => "D",
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
    let prev = std::env::var("JXL_W44_167_MODE").ok();
    // SAFETY: single-threaded bench, paired interleaved.
    unsafe { std::env::set_var("JXL_W44_167_MODE", mode.env()) };
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_strategy(EncoderStrategy::Zenjxl);
    let result = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"));
    match prev {
        Some(v) => unsafe { std::env::set_var("JXL_W44_167_MODE", v) },
        None => unsafe { std::env::remove_var("JXL_W44_167_MODE") },
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

fn classify(image: &str) -> &'static str {
    match image {
        "1420710.png" => "TARGET_1420710",
        "1531677.png" => "TARGET_1531677",
        "1418519.png" => "PROTECT_W166_1418519",
        "1189261.png" => "CONTROL_NOGATE",
        "1025469.png" => "CONTROL_NOGATE",
        _ => "OTHER",
    }
}

fn main() {
    eprintln!("W44-167 A/B/C/D: A=baseline / B=globallift / C=high-m3-only / D=per-m3-split");
    eprintln!("Cells (interleaved A,B,C,D): {}", CELLS.len());

    println!(
        "image\teffort\tdistance\tA_bytes\tB_bytes\tC_bytes\tD_bytes\t\
         BA_pct\tCA_pct\tDA_pct\t\
         A_bfly\tB_bfly\tC_bfly\tD_bfly\t\
         A_ssim2\tB_ssim2\tC_ssim2\tD_ssim2\t\
         BA_ssim2\tCA_ssim2\tDA_ssim2\tclass"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    // Aggregates: (sum_ssim2, sum_bytes_pct, n) keyed by (class, d_x10, mode_diff)
    // Use u32 for distance (× 10) so the key is Ord.
    let mut agg: BTreeMap<(&'static str, u32, &'static str), (f64, f64, usize)> = BTreeMap::new();
    let mut byte_identical: BTreeMap<(&'static str, &'static str), (usize, usize)> =
        BTreeMap::new();

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
        let sd = score_cell(
            raw,
            *w,
            *h,
            effort,
            d,
            Mode::D,
            orig_linear_img,
            orig_srgb_img,
        );

        let class = classify(image);

        if let (Some(a), Some(b), Some(c), Some(dscore)) = (sa, sb, sc, sd) {
            let ba_pct = 100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let ca_pct = 100.0 * (c.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let da_pct = 100.0 * (dscore.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let ba_ss2 = b.ssim2 - a.ssim2;
            let ca_ss2 = c.ssim2 - a.ssim2;
            let da_ss2 = dscore.ssim2 - a.ssim2;
            println!(
                "{}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}\t{:+.4}\t{}",
                image,
                effort,
                d,
                a.bytes,
                b.bytes,
                c.bytes,
                dscore.bytes,
                ba_pct,
                ca_pct,
                da_pct,
                a.butteraugli,
                b.butteraugli,
                c.butteraugli,
                dscore.butteraugli,
                a.ssim2,
                b.ssim2,
                c.ssim2,
                dscore.ssim2,
                ba_ss2,
                ca_ss2,
                da_ss2,
                class
            );
            // Aggregate by (class, d, mode_label)
            let d_key = (d * 10.0).round() as u32;
            for (label, sdelta, byte_pct, score) in [
                ("B", ba_ss2, ba_pct, &b),
                ("C", ca_ss2, ca_pct, &c),
                ("D", da_ss2, da_pct, &dscore),
            ] {
                let key = (class, d_key, label);
                let entry = agg.entry(key).or_insert((0.0, 0.0, 0));
                entry.0 += sdelta;
                entry.1 += byte_pct;
                entry.2 += 1;
                let bi_key = (class, label);
                let bi_entry = byte_identical.entry(bi_key).or_insert((0, 0));
                bi_entry.1 += 1;
                if score.bytes == a.bytes {
                    bi_entry.0 += 1;
                }
            }
        }
    }

    eprintln!("\n=== W44-167 aggregates ===");
    eprintln!("by (class, d, mode):");
    for ((class, d_key, mode_lbl), (sum_ss2, sum_pct, n)) in &agg {
        let d = *d_key as f32 / 10.0;
        eprintln!(
            "  {:24} d={:.1} mode={} n={} mean_ΔSSIM2={:+.4} mean_Δbytes%={:+.3}",
            class,
            d,
            mode_lbl,
            n,
            sum_ss2 / *n as f64,
            sum_pct / *n as f64
        );
    }
    eprintln!("\nbyte-identical counts:");
    for ((class, mode_lbl), (bi, total)) in &byte_identical {
        eprintln!(
            "  {:24} mode={} byte-identical={}/{}",
            class, mode_lbl, bi, total
        );
    }
}
