// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-145 per-block adaptive qac scaling A/B — measures whether routing
//! the W44-109 adaptive_quant qf seed scale through a per-block
//! mask1x1 lookup (instead of a uniform multiply) reduces the
//! +30 % byte overhead on the terminal d=4 e5/e6/e7 cluster while
//! preserving the SSIM2 gains from W44-109.
//!
//! Modes:
//!   A_pre_w44_145: `JXL_W44_145_PER_BLOCK_DISABLE=1` — uniform multiply,
//!                  equivalent to pre-W44-145 main behaviour (W44-109's
//!                  scale applied uniformly across every block).
//!   B_w44_145:     no env vars — per-block scaling active by default.
//!
//! Both modes leave all other knobs at production defaults so the
//! delta isolates the W44-145 mechanism.
//!
//! Coverage (35 cells):
//!   - PRIMARY: terminal e5/e6/e7 d=4 (3 cells) — must close bytes
//!     overhead from +33 % baseline toward +10-15 %, SSIM2 within ±0.30
//!   - ADJACENT distances: terminal e5/e6/e7 d=3.5/4.5 (6 cells) —
//!     bytes ±5 %, SSIM2 ±0.10
//!   - W44-105 OWNERSHIP: terminal e8/e9 d=4 (2 cells) — must be
//!     byte-identical (W44-145 doesn't fire at e>=8)
//!   - DIFFERENT CONTENT: codec_wiki d=3/d=4 (4 cells) — byte-identical
//!     (W44-109's m3 sub-gate excludes codec_wiki at d=3; d=4 fires
//!     and W44-145 may affect bytes — measured)
//!   - DIFFERENT DISTANCE: terminal d=1.0/d=2 (4 cells) —
//!     byte-identical (W44-109 doesn't fire below d=2 via m3 sub-gate;
//!     d=2 fires via m3<30 sub-gate)
//!   - PHOTOS: 8 CID22 cells — byte-identical (is_screenshot=false)
//!   - IMAC_G3 d=2/d=3 W44-109 wins (4 cells) — should reduce bytes
//!     overhead while preserving SSIM2
//!   - W44-105 (e8+) other distances: terminal e8 d=5/d=6 (4 cells) —
//!     byte-identical (e>=8, W44-145 inactive)
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_145_per_block_qac_ab \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_145_per_block_qac_ab_2026-05-21.tsv

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
const CELLS: &[(&str, &str, u8, f32)] = &[
    // === PRIMARY TARGET: terminal d=4 e5/e6/e7 (W44-144 cluster) ===
    ("terminal.png", "gb82-sc", 5, 4.0),
    ("terminal.png", "gb82-sc", 6, 4.0),
    ("terminal.png", "gb82-sc", 7, 4.0),
    // === ADJACENT distances ===
    ("terminal.png", "gb82-sc", 5, 3.5),
    ("terminal.png", "gb82-sc", 5, 4.5),
    ("terminal.png", "gb82-sc", 6, 3.5),
    ("terminal.png", "gb82-sc", 6, 4.5),
    ("terminal.png", "gb82-sc", 7, 3.5),
    ("terminal.png", "gb82-sc", 7, 4.5),
    // === W44-105 OWNERSHIP (W44-145 inactive at e>=8) ===
    ("terminal.png", "gb82-sc", 8, 4.0),
    ("terminal.png", "gb82-sc", 9, 4.0),
    ("terminal.png", "gb82-sc", 8, 5.0),
    ("terminal.png", "gb82-sc", 8, 6.0),
    ("terminal.png", "gb82-sc", 9, 5.0),
    ("terminal.png", "gb82-sc", 9, 6.0),
    // === codec_wiki: d=4 fires (mask>95), d=3 sub-gate rejects (m3>30) ===
    ("codec_wiki.png", "gb82-sc", 5, 4.0),
    ("codec_wiki.png", "gb82-sc", 6, 4.0),
    ("codec_wiki.png", "gb82-sc", 7, 4.0),
    ("codec_wiki.png", "gb82-sc", 8, 3.0),
    // === terminal d=1.0/d=2 (m3<30 sub-gate fires at d>=2 but not d=1) ===
    ("terminal.png", "gb82-sc", 5, 1.0),
    ("terminal.png", "gb82-sc", 5, 2.0),
    ("terminal.png", "gb82-sc", 6, 2.0),
    ("terminal.png", "gb82-sc", 7, 2.0),
    // === imac_g3 d=2/d=3 (W44-109 wins, W44-110 pareto trade) ===
    ("imac_g3.png", "gb82-sc", 5, 2.0),
    ("imac_g3.png", "gb82-sc", 6, 2.0),
    ("imac_g3.png", "gb82-sc", 5, 3.0),
    ("imac_g3.png", "gb82-sc", 6, 3.0),
    // === photo protection (is_screenshot=false, gate cannot fire) ===
    ("1418519.png", "CID22", 5, 4.0),
    ("1418519.png", "CID22", 7, 4.0),
    ("1418519.png", "CID22", 5, 2.0),
    ("1025469.png", "CID22", 5, 4.0),
    ("1025469.png", "CID22", 7, 4.0),
    ("1531677.png", "CID22", 5, 5.0),
    ("1420710.png", "CID22", 5, 5.0),
    ("1189261.png", "CID22", 7, 4.0),
];

#[derive(Clone, Copy, Debug)]
struct Mode {
    name: &'static str,
    per_block_disable: Option<&'static str>,
}

const MODES: &[Mode] = &[
    Mode {
        name: "A_pre_w44_145",
        per_block_disable: Some("1"),
    },
    Mode {
        name: "B_w44_145",
        per_block_disable: None,
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
        std::env::remove_var("JXL_W44_145_PER_BLOCK_DISABLE");
        if let Some(v) = mode.per_block_disable {
            std::env::set_var("JXL_W44_145_PER_BLOCK_DISABLE", v);
        }
    }
    let score = score_cell(rgb, w, h, effort, d, orig_linear_img, orig_srgb_img);
    unsafe {
        std::env::remove_var("JXL_W44_145_PER_BLOCK_DISABLE");
    }
    score
}

fn main() {
    eprintln!("W44-145 per-block adaptive qac A/B: A_pre_w44_145 vs B_w44_145");
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
