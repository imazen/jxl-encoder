// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-117 paired A/B bench — compares legacy uniform-4 EPF sharpness
//! seed (set via `JXL_W44_117_DISABLE=1`) against the new
//! `compute_epf_sharpness` seed (default) on the W44-105/107/108/109
//! screenshot-class qac-scale chain cells (terminal e5..e9 × d=2..6)
//! plus a few photo cells from the W44-90 / W44-95 cluster.
//!
//! Interleaved runs (A, B, A, B, ...) reduce thermal/turbo bias. Each
//! cell reports bytes / butteraugli / SSIM2 for both modes plus
//! per-cell deltas.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_117_paired_ab \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_117_paired_ab_2026-05-20.tsv

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

/// (image, effort, distance).
/// Coverage:
///   - terminal e5..=e9 × d ∈ {2, 3, 4, 5, 6}   (25 cells) — W44-105/107/108/109 territory
///   - codec_wiki e7 × d ∈ {3, 4, 5}            (3 cells) — W44-65 dct_suppress_hint
///   - 1418519 e8 × d ∈ {2, 4}                  (2 cells) — photo W44-95 territory
///   - 1025469 e8 × d ∈ {2, 4}                  (2 cells) — W44-116 reference
/// Total: 32 cells.
const CELLS: &[(&str, &str, u8, f32)] = &[
    // terminal e5..=e9 × d=2..6
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
    // codec_wiki spot-check (W44-65 territory)
    ("codec_wiki.png", "gb82-sc", 7, 3.0),
    ("codec_wiki.png", "gb82-sc", 7, 4.0),
    ("codec_wiki.png", "gb82-sc", 7, 5.0),
    // Photos — wider sweep at e8/e9 across distances to characterize
    // any qac-scale chain interaction (W44-105/107/108/109 was tuned
    // against the buttloop's optimistic measurement; the new fix gives
    // the buttloop a more accurate measurement which may shift the
    // converged quant_field on photo-class content).
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
];

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
    eprintln!("W44-117 paired A/B: legacy uniform-4 sharpness vs computed seed");
    eprintln!(
        "Cells: {} (interleaved A=DISABLE then B=ENABLE per cell)",
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
    let mut total_bytes_a = 0usize;
    let mut total_bytes_b = 0usize;
    let mut sum_ssim2_delta = 0f64;
    let mut sum_bfly_delta_pct = 0f64;
    let mut cells_with_data = 0;

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

        // A = legacy uniform-4 (force DISABLE)
        // SAFETY: single-threaded harness (encoder uses internal rayon
        // pool but this main loop is sequential between A and B). We
        // set/unset env var inline.
        unsafe { std::env::set_var("JXL_W44_117_DISABLE", "1") };
        let score_a = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);
        unsafe { std::env::remove_var("JXL_W44_117_DISABLE") };

        // B = W44-117 fix on (default)
        let score_b = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);

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
                    eprintln!("  ⚠ ssim2 regression: {:+.3}", ssim2_delta_abs);
                }
                total_bytes_a += a.bytes;
                total_bytes_b += b.bytes;
                sum_ssim2_delta += ssim2_delta_abs;
                if !bfly_delta_pct.is_nan() {
                    sum_bfly_delta_pct += bfly_delta_pct;
                }
                cells_with_data += 1;
            }
            _ => {
                eprintln!("  one or both scores failed; skipping");
            }
        }
    }

    eprintln!();
    eprintln!("=== W44-117 SUMMARY ===");
    eprintln!("Cells with data: {} / {}", cells_with_data, n_cells);
    if cells_with_data > 0 {
        eprintln!(
            "Total A bytes: {}  Total B bytes: {}  Delta: {:+.3}%",
            total_bytes_a,
            total_bytes_b,
            100.0 * (total_bytes_b as f64 - total_bytes_a as f64) / total_bytes_a as f64,
        );
        eprintln!(
            "Mean SSIM2 delta (B-A): {:+.4}",
            sum_ssim2_delta / cells_with_data as f64,
        );
        eprintln!(
            "Mean butteraugli delta pct (B-A)/A: {:+.3}%",
            sum_bfly_delta_pct / cells_with_data as f64,
        );
    }
    eprintln!(
        "Cells with SSIM2 regression < -0.3: {} (gate: 0)",
        regressions_ssim2_gt_0_3
    );
}
