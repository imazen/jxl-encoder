// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-119 chain-disable A/B bench — paired comparison of the
//! W44-105/107/108/109 qac-scale chain ON vs OFF.
//!
//! Mode A: chain ON  (default — current main post-W44-118)
//! Mode B: chain OFF (sets `JXL_BUTTLOOP_INITIAL_QF_SCALE=1.0` AND
//!                    `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0`)
//!
//! Both env knobs read per-encode from `std::env::var` (see
//! `vardct/butteraugli_loop.rs:785` and
//! `vardct/butteraugli_loop.rs:388`). Setting both to 1.0 collapses
//! the chain to a no-op: buttloop seed-scale becomes 1.0× (W44-105
//! / W44-107 / W44-108 default off) and adaptive-quant pre-scale
//! becomes 1.0× (W44-109 default off).
//!
//! W44-117 EPF sharpness seed is INDEPENDENT of the chain and is
//! left ON (default) per W44-118 gate. The hypothesis under test is:
//! now that the buttloop measures EPF-accurate recon on screenshots,
//! does the chain still help — or has it become a redundant
//! correction stacked on top of an already-corrected mechanism?
//!
//! Cells (83 total):
//!   - terminal e5..=e9 × d ∈ {2, 3, 4, 5, 6}   (25 cells, screenshot canary)
//!   - codec_wiki e5..=e7 × d ∈ {3, 4, 5}       (9 cells, mid-distance)
//!   - imac_g3 e5..=e9 × d ∈ {2, 3, 4}          (15 cells, mixed)
//!   - 1025469 e7..=e9 × d ∈ {1, 2, 3, 4}       (8 cells, photo canary)
//!   - 1418519 e7..=e9 × d ∈ {1, 2, 3, 4}       (8 cells, photo)
//!   - 1189261 e7..=e9 × d ∈ {2, 3, 4}          (6 cells, photo)
//!   - 1420710 e7..=e9 × d ∈ {2, 3, 4}          (6 cells, photo)
//!   - 1531677 e7..=e9 × d ∈ {2, 3, 4}          (6 cells, photo)
//!
//! Interleaved (A, B, A, B, ...) to reduce thermal/turbo bias.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_119_chain_disable_ab \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_119_chain_disable_ab_2026-05-20.tsv

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

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

/// (image, corpus, effort, distance).
const CELLS: &[(&str, &str, u8, f32)] = &[
    // terminal e5..=e9 × d=2..6 (25 cells, screenshot canary)
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
    // codec_wiki e5..=e7 × d=3..5 (9 cells, mid-distance)
    ("codec_wiki.png", "gb82-sc", 5, 3.0),
    ("codec_wiki.png", "gb82-sc", 5, 4.0),
    ("codec_wiki.png", "gb82-sc", 5, 5.0),
    ("codec_wiki.png", "gb82-sc", 6, 3.0),
    ("codec_wiki.png", "gb82-sc", 6, 4.0),
    ("codec_wiki.png", "gb82-sc", 6, 5.0),
    ("codec_wiki.png", "gb82-sc", 7, 3.0),
    ("codec_wiki.png", "gb82-sc", 7, 4.0),
    ("codec_wiki.png", "gb82-sc", 7, 5.0),
    // imac_g3 e5..=e9 × d=2..4 (15 cells, mixed)
    ("imac_g3.png", "gb82-sc", 5, 2.0),
    ("imac_g3.png", "gb82-sc", 5, 3.0),
    ("imac_g3.png", "gb82-sc", 5, 4.0),
    ("imac_g3.png", "gb82-sc", 6, 2.0),
    ("imac_g3.png", "gb82-sc", 6, 3.0),
    ("imac_g3.png", "gb82-sc", 6, 4.0),
    ("imac_g3.png", "gb82-sc", 7, 2.0),
    ("imac_g3.png", "gb82-sc", 7, 3.0),
    ("imac_g3.png", "gb82-sc", 7, 4.0),
    ("imac_g3.png", "gb82-sc", 8, 2.0),
    ("imac_g3.png", "gb82-sc", 8, 3.0),
    ("imac_g3.png", "gb82-sc", 8, 4.0),
    ("imac_g3.png", "gb82-sc", 9, 2.0),
    ("imac_g3.png", "gb82-sc", 9, 3.0),
    ("imac_g3.png", "gb82-sc", 9, 4.0),
    // Photo canaries (W44-118 1025469 + others)
    ("1025469.png", "CID22", 7, 1.0),
    ("1025469.png", "CID22", 7, 2.0),
    ("1025469.png", "CID22", 7, 3.0),
    ("1025469.png", "CID22", 7, 4.0),
    ("1025469.png", "CID22", 8, 1.0),
    ("1025469.png", "CID22", 8, 2.0),
    ("1025469.png", "CID22", 8, 4.0),
    ("1025469.png", "CID22", 9, 4.0),
    ("1418519.png", "CID22", 7, 1.0),
    ("1418519.png", "CID22", 7, 2.0),
    ("1418519.png", "CID22", 7, 3.0),
    ("1418519.png", "CID22", 7, 4.0),
    ("1418519.png", "CID22", 8, 1.0),
    ("1418519.png", "CID22", 8, 2.0),
    ("1418519.png", "CID22", 8, 4.0),
    ("1418519.png", "CID22", 9, 4.0),
    ("1189261.png", "CID22", 7, 2.0),
    ("1189261.png", "CID22", 7, 3.0),
    ("1189261.png", "CID22", 7, 4.0),
    ("1189261.png", "CID22", 8, 2.0),
    ("1189261.png", "CID22", 8, 4.0),
    ("1189261.png", "CID22", 9, 4.0),
    ("1420710.png", "CID22", 7, 2.0),
    ("1420710.png", "CID22", 7, 3.0),
    ("1420710.png", "CID22", 7, 4.0),
    ("1420710.png", "CID22", 8, 2.0),
    ("1420710.png", "CID22", 8, 4.0),
    ("1420710.png", "CID22", 9, 4.0),
    ("1531677.png", "CID22", 7, 2.0),
    ("1531677.png", "CID22", 7, 3.0),
    ("1531677.png", "CID22", 7, 4.0),
    ("1531677.png", "CID22", 8, 2.0),
    ("1531677.png", "CID22", 8, 4.0),
    ("1531677.png", "CID22", 9, 4.0),
];

fn encode_default(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
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
    let bitstream = match encode_default(rgb, w, h, effort, d) {
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
    eprintln!("W44-119 chain-disable A/B");
    eprintln!("  Mode A = chain ON (current main, post-W44-118)");
    eprintln!("  Mode B = chain OFF (JXL_BUTTLOOP_INITIAL_QF_SCALE=1.0,");
    eprintln!("                      JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0)");
    eprintln!(
        "Cells: {} (interleaved A, B per cell to reduce thermal bias)",
        CELLS.len()
    );

    println!(
        "image\teffort\tdistance\tA_bytes\tB_bytes\tbytes_delta_pct\tA_bfly\tB_bfly\tbfly_delta_pct\tA_ssim2\tB_ssim2\tssim2_delta_abs"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    let mut regressions_ssim2_gt_0_3 = 0usize;
    let mut wins_ssim2_gt_0_3 = 0usize;
    let mut total_bytes_a = 0usize;
    let mut total_bytes_b = 0usize;
    let mut sum_ssim2_delta = 0f64;
    let mut sum_bfly_delta_pct = 0f64;
    let mut cells_with_data = 0;

    // Bucket counters per content class.
    let mut ssim2_terminal = 0f64;
    let mut n_terminal = 0;
    let mut ssim2_codec_wiki = 0f64;
    let mut n_codec_wiki = 0;
    let mut ssim2_imac_g3 = 0f64;
    let mut n_imac_g3 = 0;
    let mut ssim2_photo = 0f64;
    let mut n_photo = 0;
    let mut bytes_terminal_a = 0i64;
    let mut bytes_terminal_b = 0i64;
    let mut bytes_codec_wiki_a = 0i64;
    let mut bytes_codec_wiki_b = 0i64;
    let mut bytes_imac_g3_a = 0i64;
    let mut bytes_imac_g3_b = 0i64;
    let mut bytes_photo_a = 0i64;
    let mut bytes_photo_b = 0i64;

    for (i, &(image, corpus, effort, d)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} ({}) e{} d={}",
            i + 1,
            n_cells,
            image,
            corpus,
            effort,
            d
        );

        let dir = match corpus {
            "CID22" => CID22,
            "gb82-sc" => GB82SC,
            _ => {
                eprintln!("  unknown corpus: {}", corpus);
                continue;
            }
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

        // A = chain ON (default — make sure env is unset)
        // SAFETY: sequential A/B inside this single-threaded loop;
        // the encoder uses its own rayon pool internally but the
        // env state is set/unset between A and B in the same thread,
        // so no other thread observes a partially-modified env.
        unsafe {
            std::env::remove_var("JXL_BUTTLOOP_INITIAL_QF_SCALE");
            std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE");
        }
        let score_a = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);

        // SAFETY: same as above — A and B run sequentially in this
        // single thread; env mutations are bracketed around each
        // synchronous encode call.
        unsafe {
            std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "1.0");
            std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "1.0");
        }
        let score_b = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);
        // SAFETY: restore default state after B run completes.
        unsafe {
            std::env::remove_var("JXL_BUTTLOOP_INITIAL_QF_SCALE");
            std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE");
        }

        match (score_a, score_b) {
            (Some(a), Some(b)) => {
                let bytes_delta_pct = 100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64;
                let bfly_delta_pct = if a.butteraugli > 0.0 {
                    100.0 * (b.butteraugli - a.butteraugli) / a.butteraugli
                } else {
                    f64::NAN
                };
                let ssim2_delta_abs = b.ssim2 - a.ssim2;
                println!(
                    "{}\te{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:+.3}\t{:.4}\t{:.4}\t{:+.3}",
                    image,
                    effort,
                    d,
                    a.bytes,
                    b.bytes,
                    bytes_delta_pct,
                    a.butteraugli,
                    b.butteraugli,
                    bfly_delta_pct,
                    a.ssim2,
                    b.ssim2,
                    ssim2_delta_abs,
                );
                if ssim2_delta_abs < -0.3 {
                    regressions_ssim2_gt_0_3 += 1;
                    eprintln!("  ⚠ ssim2 regression (B-A): {:+.3}", ssim2_delta_abs);
                }
                if ssim2_delta_abs > 0.3 {
                    wins_ssim2_gt_0_3 += 1;
                }
                total_bytes_a += a.bytes;
                total_bytes_b += b.bytes;
                sum_ssim2_delta += ssim2_delta_abs;
                if !bfly_delta_pct.is_nan() {
                    sum_bfly_delta_pct += bfly_delta_pct;
                }
                cells_with_data += 1;

                // Bucket tally
                if image.starts_with("terminal") {
                    ssim2_terminal += ssim2_delta_abs;
                    n_terminal += 1;
                    bytes_terminal_a += a.bytes as i64;
                    bytes_terminal_b += b.bytes as i64;
                } else if image.starts_with("codec_wiki") {
                    ssim2_codec_wiki += ssim2_delta_abs;
                    n_codec_wiki += 1;
                    bytes_codec_wiki_a += a.bytes as i64;
                    bytes_codec_wiki_b += b.bytes as i64;
                } else if image.starts_with("imac_g3") {
                    ssim2_imac_g3 += ssim2_delta_abs;
                    n_imac_g3 += 1;
                    bytes_imac_g3_a += a.bytes as i64;
                    bytes_imac_g3_b += b.bytes as i64;
                } else {
                    ssim2_photo += ssim2_delta_abs;
                    n_photo += 1;
                    bytes_photo_a += a.bytes as i64;
                    bytes_photo_b += b.bytes as i64;
                }
            }
            _ => {
                eprintln!("  one or both scores failed; skipping");
            }
        }
    }

    eprintln!();
    eprintln!("=== W44-119 SUMMARY ===");
    eprintln!("Cells with data: {} / {}", cells_with_data, n_cells);
    if cells_with_data > 0 {
        let bytes_delta_pct =
            100.0 * (total_bytes_b as f64 - total_bytes_a as f64) / total_bytes_a as f64;
        let avg_ssim2_delta = sum_ssim2_delta / cells_with_data as f64;
        let avg_bfly_delta_pct = sum_bfly_delta_pct / cells_with_data as f64;
        eprintln!(
            "Total A bytes: {}  Total B bytes: {}  Delta: {:+.3}% (B vs A; +N = chain-off bigger)",
            total_bytes_a, total_bytes_b, bytes_delta_pct,
        );
        eprintln!(
            "Avg SSIM2 delta (B-A): {:+.4}  (+ = chain-off better)",
            avg_ssim2_delta,
        );
        eprintln!(
            "Avg bfly delta (B-A): {:+.3}%  (- = chain-off better, lower bfly is better)",
            avg_bfly_delta_pct,
        );
        eprintln!(
            "Cells with SSIM2 regression > 0.3 (chain-OFF worse than ON): {}",
            regressions_ssim2_gt_0_3,
        );
        eprintln!(
            "Cells with SSIM2 win > 0.3 (chain-OFF better than ON):      {}",
            wins_ssim2_gt_0_3,
        );
    }

    eprintln!();
    eprintln!("=== Per-cluster averages (SSIM2 delta B-A) ===");
    if n_terminal > 0 {
        let bytes_d =
            100.0 * (bytes_terminal_b - bytes_terminal_a) as f64 / bytes_terminal_a as f64;
        eprintln!(
            "  terminal   (n={}): avg ssim2 delta = {:+.3}, bytes delta = {:+.2}%",
            n_terminal,
            ssim2_terminal / n_terminal as f64,
            bytes_d,
        );
    }
    if n_codec_wiki > 0 {
        let bytes_d =
            100.0 * (bytes_codec_wiki_b - bytes_codec_wiki_a) as f64 / bytes_codec_wiki_a as f64;
        eprintln!(
            "  codec_wiki (n={}): avg ssim2 delta = {:+.3}, bytes delta = {:+.2}%",
            n_codec_wiki,
            ssim2_codec_wiki / n_codec_wiki as f64,
            bytes_d,
        );
    }
    if n_imac_g3 > 0 {
        let bytes_d = 100.0 * (bytes_imac_g3_b - bytes_imac_g3_a) as f64 / bytes_imac_g3_a as f64;
        eprintln!(
            "  imac_g3    (n={}): avg ssim2 delta = {:+.3}, bytes delta = {:+.2}%",
            n_imac_g3,
            ssim2_imac_g3 / n_imac_g3 as f64,
            bytes_d,
        );
    }
    if n_photo > 0 {
        let bytes_d = 100.0 * (bytes_photo_b - bytes_photo_a) as f64 / bytes_photo_a as f64;
        eprintln!(
            "  photo      (n={}): avg ssim2 delta = {:+.3}, bytes delta = {:+.2}%",
            n_photo,
            ssim2_photo / n_photo as f64,
            bytes_d,
        );
    }
}
