// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-140 EPF seed distance-fade bisection — probes whether a linear
//! per-block blend between the W44-117 sharpness map and uniform-4
//! closes the residual terminal e8/e9 d=1.0-1.6 SSIM2 oscillations
//! W44-120 documented as out-of-scope for pure threshold tightening.
//!
//! Target cluster (post-W44-120 main, vs A_legacy = uniform-4):
//!   d=0.8: 0.000 (W44-120 closes via threshold gate)
//!   d=1.0: +0.529 SSIM2 (W44-117 win)
//!   d=1.2: -0.726 SSIM2 (W44-117 regression, pre-existing oscillation)
//!   d=1.4: +0.685 SSIM2 (W44-117 win)
//!   d=1.5: -0.959 SSIM2 (W44-117 regression, pre-existing oscillation)
//!   d=2.0+: all wins (+0.0 to +0.9 SSIM2)
//!
//! Hypothesis: at d in [1.0, fade_max], blend with uniform-4 (weight
//! grows linearly from 0 at min_distance=1.0 to 1 at fade_max). If
//! the regression-at-d=1.2 is sensitive to seed magnitude, a softer
//! seed at low d may mute the regressions while preserving the wins
//! that come from the W44-117 mechanism in aggregate.
//!
//! Modes:
//!   A_legacy: JXL_W44_117_DISABLE=1 (pre-W44-117 baseline, uniform-4)
//!   B_main:   no env vars (current main = W44-120 threshold=1.0)
//!   C_fade15: JXL_W44_140_EPF_SEED_FADE_MAX=1.5 (fade 1.0..1.5)
//!   C_fade20: JXL_W44_140_EPF_SEED_FADE_MAX=2.0 (fade 1.0..2.0)
//!   C_fade30: JXL_W44_140_EPF_SEED_FADE_MAX=3.0 (fade 1.0..3.0)
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_140_distance_fade_bisect \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_140_distance_fade_bisect_2026-05-20.tsv

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
///   - terminal e8/e9 × d ∈ {0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 2.0, 3.0, 4.0, 5.0} (20 cells)
///     - target = 1.0..=1.6 cluster
///     - protection = 2.0..=5.0 (W44-117 wins must stay)
///     - d=0.8 anchor = below W44-120 threshold (uniform-4 in all modes)
///   - codec_wiki e8 d=3 (W44-107 protected — fades must not regress)
///   - imac_g3 e5 d=2 (W44-110 OPEN cell, byte-identical baseline)
///   - 2 photo cells (W44-118 gate verification — must be byte-identical)
const CELLS: &[(&str, &str, u8, f32)] = &[
    // === Primary cluster: terminal e8/e9 × d=0.8..1.6 ===
    ("terminal.png", "gb82-sc", 8, 0.8),
    ("terminal.png", "gb82-sc", 8, 1.0),
    ("terminal.png", "gb82-sc", 8, 1.2),
    ("terminal.png", "gb82-sc", 8, 1.4),
    ("terminal.png", "gb82-sc", 8, 1.5),
    ("terminal.png", "gb82-sc", 8, 1.6),
    ("terminal.png", "gb82-sc", 9, 0.8),
    ("terminal.png", "gb82-sc", 9, 1.0),
    ("terminal.png", "gb82-sc", 9, 1.2),
    ("terminal.png", "gb82-sc", 9, 1.4),
    ("terminal.png", "gb82-sc", 9, 1.5),
    ("terminal.png", "gb82-sc", 9, 1.6),
    // === W44-117 win protection: terminal e8/e9 × d=2..5 ===
    ("terminal.png", "gb82-sc", 8, 2.0),
    ("terminal.png", "gb82-sc", 8, 3.0),
    ("terminal.png", "gb82-sc", 8, 4.0),
    ("terminal.png", "gb82-sc", 8, 5.0),
    ("terminal.png", "gb82-sc", 9, 2.0),
    ("terminal.png", "gb82-sc", 9, 3.0),
    ("terminal.png", "gb82-sc", 9, 4.0),
    ("terminal.png", "gb82-sc", 9, 5.0),
    // === W44-107 protected screenshot cell (codec_wiki e8 d=3) — must NOT regress ===
    ("codec_wiki.png", "gb82-sc", 8, 3.0),
    // === Photo protection (W44-118 gate) ===
    ("1418519.png", "CID22", 8, 2.0),
    ("1025469.png", "CID22", 8, 4.0),
];

#[derive(Clone, Copy, Debug)]
struct Mode {
    name: &'static str,
    disable_w44_117: Option<&'static str>,
    fade_max: Option<&'static str>,
}

const MODES: &[Mode] = &[
    Mode {
        name: "A_legacy",
        disable_w44_117: Some("1"),
        fade_max: None,
    },
    Mode {
        name: "B_main",
        disable_w44_117: None,
        fade_max: None,
    },
    Mode {
        name: "C_fade15",
        disable_w44_117: None,
        fade_max: Some("1.5"),
    },
    Mode {
        name: "C_fade20",
        disable_w44_117: None,
        fade_max: Some("2.0"),
    },
    Mode {
        name: "C_fade30",
        disable_w44_117: None,
        fade_max: Some("3.0"),
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
        std::env::remove_var("JXL_W44_117_DISABLE");
        std::env::remove_var("JXL_W44_140_EPF_SEED_FADE_MAX");
        if let Some(v) = mode.disable_w44_117 {
            std::env::set_var("JXL_W44_117_DISABLE", v);
        }
        if let Some(v) = mode.fade_max {
            std::env::set_var("JXL_W44_140_EPF_SEED_FADE_MAX", v);
        }
    }
    let score = score_cell(rgb, w, h, effort, d, orig_linear_img, orig_srgb_img);
    unsafe {
        std::env::remove_var("JXL_W44_117_DISABLE");
        std::env::remove_var("JXL_W44_140_EPF_SEED_FADE_MAX");
    }
    score
}

fn main() {
    eprintln!("W44-140 distance-fade bisection: A_legacy + B_main + C_fade{{15,20,30}}");
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
                panic!("failed to open {}: {}", path.display(), e);
            });
            let (w, h) = img.dimensions();
            let rgb8 = img.to_rgb8().into_raw();
            let lin = srgb_u8_to_linear(&rgb8, w, h);
            let srgb_pixels: Vec<[u8; 3]> = rgb8.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
            (w, h, rgb8, lin, srgb_img)
        });
        let (w, h, rgb, lin, srgb_img) = entry.clone();

        let mut row = format!("{}\t{}\t{}", name, effort, d);
        let mut scores: Vec<Option<Score>> = Vec::with_capacity(MODES.len());

        for m in MODES {
            let s = run_mode(m, &rgb, w, h, effort, d, &lin, &srgb_img);
            if let Some(score) = s {
                row.push_str(&format!(
                    "\t{}\t{:.4}\t{:.4}",
                    score.bytes, score.butteraugli, score.ssim2
                ));
            } else {
                row.push_str("\tNA\tNA\tNA");
            }
            scores.push(s);
        }

        let baseline = scores[0];
        for s in &scores[1..] {
            match (baseline, s) {
                (Some(b), Some(v)) => {
                    let db_pct = (v.bytes as f64 - b.bytes as f64) / b.bytes as f64 * 100.0;
                    let dbfly_pct = (v.butteraugli - b.butteraugli) / b.butteraugli * 100.0;
                    let dssim2 = v.ssim2 - b.ssim2;
                    row.push_str(&format!(
                        "\t{:+.3}\t{:+.3}\t{:+.4}",
                        db_pct, dbfly_pct, dssim2
                    ));
                }
                _ => row.push_str("\tNA\tNA\tNA"),
            }
        }
        println!("{}", row);
    }
    eprintln!("Done in {:.1}s", start.elapsed().as_secs_f64());
}
