// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-142 codec_wiki EPF seed suppression A/B — verifies that the
//! zenanalyze-driven sub-gate inside the W44-117/140 EPF seed admission
//! closes the codec_wiki d=1.2/1.6/1.8 SSIM2 regression cluster that
//! the W44-141 cjxl-parity ledger refresh surfaced as a follow-on to
//! W44-140 (`b8333091`), while preserving:
//!   - terminal d=1.0/1.2/1.4 W44-140 wins (terminal m3=13.85, rejected
//!     by the m3>=60 gate)
//!   - terminal d>=2 byte-identical (above suppression cap d<2.0)
//!   - codec_wiki d=3 W44-117/W44-124 wins (above suppression cap)
//!   - 4 photos at d=2/5 byte-identical (all CID22 photos have ed >= 0.16,
//!     rejected by ed<0.05 gate)
//!
//! Modes:
//!   A_pre_w44_142: JXL_W44_142_SUPPRESS_DISABLE=1 (current W44-140 main)
//!   B_w44_142:     no env vars (with W44-142 gate active)
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_142_codec_wiki_suppress_ab \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_142_codec_wiki_suppress_ab_2026-05-20.tsv

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

/// (image, corpus, effort, distance).
///
/// Coverage:
///   - codec_wiki e8/e9 × d ∈ {1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0} (14 cells)
///     - regression target = 1.2/1.6/1.8 (must close)
///     - boundary checks = 1.0/1.4/2.0 (must stay byte-identical or improve)
///     - protection = 3.0 (W44-124 + W44-117 win, must NOT regress)
///   - terminal e8/e9 × d ∈ {1.0, 1.2, 1.4, 1.5, 1.6, 2.0, 3.0, 4.0, 5.0} (18 cells)
///     - W44-140 wins protection = 1.0/1.2/1.4 (must preserve)
///     - W44-117 wins protection = 1.6/2.0/3.0/4.0/5.0 (must preserve)
///   - 2 CID22 photo cells (W44-118 gate verification — must be byte-identical)
const CELLS: &[(&str, &str, u8, f32)] = &[
    // === Primary regression cluster: codec_wiki e8/e9 d=1.2/1.6/1.8 ===
    ("codec_wiki.png", "gb82-sc", 8, 1.2),
    ("codec_wiki.png", "gb82-sc", 8, 1.6),
    ("codec_wiki.png", "gb82-sc", 8, 1.8),
    ("codec_wiki.png", "gb82-sc", 9, 1.2),
    ("codec_wiki.png", "gb82-sc", 9, 1.6),
    ("codec_wiki.png", "gb82-sc", 9, 1.8),
    // === codec_wiki boundary cells: 1.0 (W44-140 weight=0), 1.4, 2.0 ===
    ("codec_wiki.png", "gb82-sc", 8, 1.0),
    ("codec_wiki.png", "gb82-sc", 8, 1.4),
    ("codec_wiki.png", "gb82-sc", 8, 2.0),
    ("codec_wiki.png", "gb82-sc", 9, 1.0),
    ("codec_wiki.png", "gb82-sc", 9, 1.4),
    ("codec_wiki.png", "gb82-sc", 9, 2.0),
    // === codec_wiki d=3 protection (W44-117 + W44-124 win) ===
    ("codec_wiki.png", "gb82-sc", 8, 3.0),
    ("codec_wiki.png", "gb82-sc", 9, 3.0),
    // === terminal W44-140 cluster (must stay byte-identical) ===
    ("terminal.png", "gb82-sc", 8, 1.0),
    ("terminal.png", "gb82-sc", 8, 1.2),
    ("terminal.png", "gb82-sc", 8, 1.4),
    ("terminal.png", "gb82-sc", 8, 1.5),
    ("terminal.png", "gb82-sc", 8, 1.6),
    ("terminal.png", "gb82-sc", 9, 1.0),
    ("terminal.png", "gb82-sc", 9, 1.2),
    ("terminal.png", "gb82-sc", 9, 1.4),
    ("terminal.png", "gb82-sc", 9, 1.5),
    ("terminal.png", "gb82-sc", 9, 1.6),
    // === terminal W44-117 wins (must stay byte-identical) ===
    ("terminal.png", "gb82-sc", 8, 2.0),
    ("terminal.png", "gb82-sc", 8, 3.0),
    ("terminal.png", "gb82-sc", 8, 4.0),
    ("terminal.png", "gb82-sc", 8, 5.0),
    ("terminal.png", "gb82-sc", 9, 2.0),
    ("terminal.png", "gb82-sc", 9, 3.0),
    ("terminal.png", "gb82-sc", 9, 4.0),
    ("terminal.png", "gb82-sc", 9, 5.0),
    // === Photo protection (W44-118 gate — ed >= 0.16 rejects all) ===
    ("1418519.png", "CID22", 8, 2.0),
    ("1418519.png", "CID22", 8, 5.0),
    ("1025469.png", "CID22", 8, 2.0),
    ("1025469.png", "CID22", 8, 5.0),
];

#[derive(Clone, Copy, Debug)]
struct Mode {
    name: &'static str,
    suppress_disable: Option<&'static str>,
}

const MODES: &[Mode] = &[
    Mode {
        name: "A_pre_w44_142",
        suppress_disable: Some("1"),
    },
    Mode {
        name: "B_w44_142",
        suppress_disable: None,
    },
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

fn run_mode(
    mode: &Mode,
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
) -> Option<Score> {
    unsafe {
        std::env::remove_var("JXL_W44_142_SUPPRESS_DISABLE");
        if let Some(v) = mode.suppress_disable {
            std::env::set_var("JXL_W44_142_SUPPRESS_DISABLE", v);
        }
    }
    let score = score_cell(rgb, w, h, effort, d, orig_linear_img, orig_srgb_img);
    unsafe {
        std::env::remove_var("JXL_W44_142_SUPPRESS_DISABLE");
    }
    score
}

fn main() {
    eprintln!("W44-142 codec_wiki suppression A/B: A_pre_w44_142 vs B_w44_142");
    eprintln!(
        "Cells: {} ({} modes per cell, interleaved per cell)",
        CELLS.len(),
        MODES.len()
    );

    let mut header = String::from("image\teffort\tdistance");
    for m in MODES {
        header.push_str(&format!(
            "\t{}_bytes\t{}_bfly\t{}_ssim2",
            m.name, m.name, m.name
        ));
    }
    for m in &MODES[1..] {
        header.push_str(&format!(
            "\t{}_dB_pct\t{}_dBfly_pct\t{}_dSSIM2",
            m.name, m.name, m.name
        ));
    }
    println!("{}", header);

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    let start = std::time::Instant::now();

    for (i, &(name, corpus, effort, d)) in CELLS.iter().enumerate() {
        eprintln!("[{}/{}] {} e{} d={}", i + 1, n_cells, name, effort, d);
        let path = match corpus {
            "CID22" => PathBuf::from(format!("{}/{}", CID22, name)),
            "gb82-sc" => PathBuf::from(format!("{}/{}", GB82SC, name)),
            _ => panic!("unknown corpus: {}", corpus),
        };
        let cache_key = path.to_string_lossy().to_string();
        let entry = images_cache.entry(cache_key).or_insert_with(|| {
            let img = image::open(&path).unwrap_or_else(|e| {
                panic!("failed to open {}: {e}", path.display());
            });
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8().into_raw();
            let orig_linear = srgb_u8_to_linear(&rgb, w, h);
            let orig_srgb_pixels: Vec<[u8; 3]> =
                rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            let orig_srgb = Img::new(orig_srgb_pixels, w as usize, h as usize);
            (w, h, rgb, orig_linear, orig_srgb)
        });
        let (w, h, rgb, orig_linear, orig_srgb) = entry;
        let w = *w;
        let h = *h;

        let mut row_scores: Vec<Option<Score>> = Vec::with_capacity(MODES.len());
        for mode in MODES {
            let score = run_mode(mode, rgb, w, h, effort, d, orig_linear, orig_srgb);
            if let Some(s) = score {
                eprintln!(
                    "  {:>14}: bytes={:>7} bfly={:>6.3} ssim2={:>6.2}",
                    mode.name, s.bytes, s.butteraugli, s.ssim2
                );
            } else {
                eprintln!("  {:>14}: FAILED", mode.name);
            }
            row_scores.push(score);
        }

        let mut row = format!("{}\t{}\t{}", name, effort, d);
        for s in &row_scores {
            match s {
                Some(s) => row.push_str(&format!(
                    "\t{}\t{:.6}\t{:.4}",
                    s.bytes, s.butteraugli, s.ssim2
                )),
                None => row.push_str("\tNaN\tNaN\tNaN"),
            }
        }
        let baseline = row_scores[0];
        for s in row_scores.iter().skip(1) {
            match (s, baseline) {
                (Some(s), Some(b)) => {
                    let db_pct = 100.0 * (s.bytes as f64 - b.bytes as f64) / b.bytes as f64;
                    let dbfly_pct = if b.butteraugli > 0.0 {
                        100.0 * (s.butteraugli - b.butteraugli) / b.butteraugli
                    } else {
                        0.0
                    };
                    let dssim2 = s.ssim2 - b.ssim2;
                    row.push_str(&format!("\t{:.3}\t{:.3}\t{:.4}", db_pct, dbfly_pct, dssim2));
                }
                _ => row.push_str("\tNaN\tNaN\tNaN"),
            }
        }
        println!("{}", row);
    }

    eprintln!("\nDone in {:.1}s", start.elapsed().as_secs_f64());
}
