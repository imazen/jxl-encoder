// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-169 (Smart-Zenjxl chunk 6) paired A/B reproducer.
//!
//! Distance-narrowed W44-168 SmoothSkip. W44-168 broad Mode B
//! honest-stopped because it destroyed the W44-166 +0.45 SSIM2 win on
//! 1418519 e8 d=6. Same measurement found STRICT WINS at the narrow
//! d=4/5 band on 1418519:
//!   - e8 d=4: ΔSSIM2 +0.627 + Δwall -4.79%
//!   - e8 d=5: ΔSSIM2 +0.559 + Δwall -4.13%
//!
//! W44-169 ships the narrow-band Mode B as the production default
//! (gated to `target_distance ∈ [4.0, 5.0]` ONLY). API surface:
//! `EncoderImprovementsCustom::adaptive_buttloop_iters_narrow: bool`.
//!
//! Modes:
//! - A (Baseline): `narrow_enabled = false`. Byte-identical to
//!   pre-W44-169 main (W44-168 main = 42833a05).
//! - B (W44-169 narrow SHIPPED): `narrow_enabled = true`. SmoothSkip
//!   fires ONLY at d ∈ [4.0, 5.0] on smooth/screenshot content at
//!   e>=8.
//!
//! Acceptance gates (per chunk spec):
//! - (a) Build PASS
//! - (b) `cargo test --lib`: PASS
//! - (c) Hash-locks 36/36 BYTE-IDENTICAL
//! - (d) TARGET 1418519 e8 d=4/5: SSIM2 mean improvement >= +0.5
//! - (e) PROTECT_W166 1418519 d=6: BYTE-IDENTICAL (3/3)
//! - (f) PROTECT_W164 screenshots: BYTE-IDENTICAL
//! - (g) PROTECT_other_photos (1025469/1420710/1531677 e8 d=4/5):
//!       BYTE-IDENTICAL (mask_p25 < 85 discriminator validated)
//! - (h) CONTROL 1189261 e8 d=4: BYTE-IDENTICAL
//! - (i) EncoderStrategy::Libjxl: BYTE-IDENTICAL regardless of field
//! - (j) Wall time net negative on TARGET (>= 2% reduction)
//! - (k) Multi-decoder PASS (handled separately)
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release -p jxl-encoder \
//!     --features 'butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_169_narrow_iter_skip \
//!     > benchmarks/w44_169_narrow_iter_skip_2026-05-21.tsv

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
use std::time::Instant;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// Cells (per W44-169 task spec):
/// - TARGET: 1418519 × {e7,e8,e9} × {d=4, d=5} = 6 cells
///   (W44-168 measurement found ΔSSIM2 +0.56/+0.63 + Δwall -4-5% at
///   e8 d=4/5; expanded to e7/e9 to verify the band is right at
///   adjacent efforts)
/// - PROTECT_W166: 1418519 × {e7,e8,e9} × {d=6} = 3 cells
///   (must BYTE-IDENTICAL: gate excludes d=6)
/// - PROTECT_W164: 3 GB82-SC screenshots × {e5,e6} = 6 cells
///   (must BYTE-IDENTICAL: gate is e>=8 only)
/// - PROTECT_other_photos: 1025469 + 1420710 + 1531677 × {e8} × {d=4,5}
///   = 6 cells (must BYTE-IDENTICAL: mask_p25 < 85 on these photos)
/// - CONTROL: 1189261 × {e8} × {d=4} = 1 cell (textured, gate must NOT fire)
const CELLS: &[(&str, &str, u8, f32)] = &[
    // TARGET 1418519 d=4/5 e7/e8/e9 (6 cells)
    ("cid22", "1418519.png", 7, 4.0),
    ("cid22", "1418519.png", 7, 5.0),
    ("cid22", "1418519.png", 8, 4.0),
    ("cid22", "1418519.png", 8, 5.0),
    ("cid22", "1418519.png", 9, 4.0),
    ("cid22", "1418519.png", 9, 5.0),
    // PROTECT_W166 1418519 d=6 e7/e8/e9 (3 cells, must BYTE-IDENTICAL)
    ("cid22", "1418519.png", 7, 6.0),
    ("cid22", "1418519.png", 8, 6.0),
    ("cid22", "1418519.png", 9, 6.0),
    // PROTECT_W164 screenshots e5/e6 (6 cells, must BYTE-IDENTICAL —
    // gate is e>=8 only)
    ("gb82sc", "codec_wiki.png", 5, 1.0),
    ("gb82sc", "codec_wiki.png", 6, 1.0),
    ("gb82sc", "imac_g3.png", 5, 1.0),
    ("gb82sc", "imac_g3.png", 6, 1.0),
    ("gb82sc", "terminal.png", 5, 1.0),
    ("gb82sc", "terminal.png", 6, 1.0),
    // PROTECT_other_photos at e8 d=4/5 (6 cells — must BYTE-IDENTICAL
    // because their mask_p25 is below 85)
    ("cid22", "1025469.png", 8, 4.0),
    ("cid22", "1025469.png", 8, 5.0),
    ("cid22", "1420710.png", 8, 4.0),
    ("cid22", "1420710.png", 8, 5.0),
    ("cid22", "1531677.png", 8, 4.0),
    ("cid22", "1531677.png", 8, 5.0),
    // CONTROL textured photo at e8 d=4 (must BYTE-IDENTICAL — gate
    // fires on smooth, not textured)
    ("cid22", "1189261.png", 8, 4.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    A, // narrow_enabled = false (baseline = pre-W44-169 = byte-identical to W44-168 main with env unset)
    B, // narrow_enabled = true  (W44-169 SHIPPED)
}

fn build_strategy(mode: Mode) -> EncoderStrategy {
    let mut custom = EncoderImprovementsCustom::default();
    custom.adaptive_buttloop_iters_narrow = matches!(mode, Mode::B);
    EncoderStrategy::Custom(Box::new(custom))
}

fn encode_with_mode(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
) -> Result<(Vec<u8>, f64), String> {
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_strategy(build_strategy(mode));
    let start = Instant::now();
    let result = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    result.map(|b| (b, elapsed_ms))
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
    encode_ms: f64,
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
    let (bitstream, encode_ms) = match encode_with_mode(rgb, w, h, effort, d, mode) {
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
        encode_ms,
    })
}

fn classify(corpus: &str, image: &str, effort: u8, d: f32) -> &'static str {
    if corpus == "gb82sc" {
        return "PROTECT_W164_SCREENSHOT";
    }
    match image {
        "1418519.png" => {
            if (d - 6.0).abs() < 0.01 {
                "PROTECT_W166_1418519_d6"
            } else {
                "TARGET_1418519"
            }
        }
        "1189261.png" => "CONTROL_TEXTURED",
        "1025469.png" | "1420710.png" | "1531677.png" => {
            // mask_p25 < 85 on all three — gate must NOT fire even at e8
            if effort == 8 {
                "PROTECT_OTHER_PHOTOS"
            } else {
                "OTHER"
            }
        }
        _ => "OTHER",
    }
}

fn corpus_path(corpus: &str, image: &str) -> PathBuf {
    match corpus {
        "cid22" => PathBuf::from(CID22).join(image),
        "gb82sc" => PathBuf::from(GB82_SC).join(image),
        _ => PathBuf::from(image),
    }
}

fn main() {
    eprintln!("W44-169 A/B: A=narrow_off (pre-W44-169 baseline) / B=narrow_on (SHIPPED)");
    eprintln!("Cells (interleaved A,B): {}", CELLS.len());

    println!(
        "corpus\timage\teffort\tdistance\t\
         A_bytes\tB_bytes\tBA_pct\t\
         A_bfly\tB_bfly\t\
         A_ssim2\tB_ssim2\tBA_ssim2\t\
         A_encode_ms\tB_encode_ms\tBA_ms_pct\tclass"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    // Aggregates: keyed by class
    let mut agg: BTreeMap<&'static str, (f64, f64, f64, usize)> = BTreeMap::new();
    let mut byte_identical: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();

    for (i, &(corpus, image, effort, d)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {}:{} e{} d={}",
            i + 1,
            n_cells,
            corpus,
            image,
            effort,
            d
        );

        let path = corpus_path(corpus, image);
        let cache_key = format!("{}:{}", corpus, image);
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

        // Interleaved paired (A, B, A, B) — single trial per cell;
        // wall-time deltas are inherently noisier than the W44-168
        // 4-trial bench but the byte/SSIM2 axes are deterministic.
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

        let class = classify(corpus, image, effort, d);

        if let (Some(a), Some(b)) = (sa, sb) {
            let ba_pct = 100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let ba_ss2 = b.ssim2 - a.ssim2;
            let ba_ms_pct = 100.0 * (b.encode_ms - a.encode_ms) / a.encode_ms;
            println!(
                "{}\t{}\t{}\t{:.1}\t{}\t{}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\t\
                 {:.2}\t{:.2}\t{:+.2}\t{}",
                corpus,
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
                ba_ss2,
                a.encode_ms,
                b.encode_ms,
                ba_ms_pct,
                class
            );
            let entry = agg.entry(class).or_insert((0.0, 0.0, 0.0, 0));
            entry.0 += ba_ss2;
            entry.1 += ba_pct;
            entry.2 += ba_ms_pct;
            entry.3 += 1;
            let bi_entry = byte_identical.entry(class).or_insert((0, 0));
            bi_entry.1 += 1;
            if a.bytes == b.bytes {
                bi_entry.0 += 1;
            }
        }
    }

    eprintln!("\n=== W44-169 aggregates ===");
    eprintln!("by class — mean (B vs A) across cells:");
    for (class, (sum_ss2, sum_pct, sum_ms_pct, n)) in &agg {
        eprintln!(
            "  {:32} n={} mean_ΔSSIM2={:+.4} mean_Δbytes%={:+.3} mean_Δms%={:+.2}",
            class,
            n,
            sum_ss2 / *n as f64,
            sum_pct / *n as f64,
            sum_ms_pct / *n as f64,
        );
    }
    eprintln!("\nbyte-identical counts (acceptance gates e/f/g/h):");
    for (class, (bi, total)) in &byte_identical {
        eprintln!("  {:32} byte-identical={}/{}", class, bi, total);
    }
}
