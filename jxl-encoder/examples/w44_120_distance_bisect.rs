// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-120 distance-gate bisection — sweeps the W44-117 EPF sharpness
//! seed activation threshold across 5 modes (A = legacy uniform-4 via
//! `JXL_W44_117_DISABLE=1`, then four W44-120 thresholds 0.8 / 1.0 /
//! 1.2 / 1.5 via `JXL_W44_120_EPF_SEED_MIN_DISTANCE=<f32>`).
//!
//! Targets the W44-119 ledger-refresh-surfaced regression: terminal
//! e8/e9 d=0.8 SSIM2 -1.87 (W44-117 over-correction at low distance).
//! Goal: find a distance cutoff that closes that regression while
//! preserving the W44-117/118 wins above the cutoff (terminal e8/e9
//! d=4 SSIM2 +0.90, d=3 +0.66, d=1.4 +0.69).
//!
//! Interleaved runs (A, B0.8, B1.0, B1.2, B1.5, A, ...) per cell to
//! reduce thermal/turbo bias. Each cell reports bytes / butteraugli /
//! SSIM2 for every mode plus per-mode deltas vs A (legacy baseline)
//! and vs B_default (the W44-117 default that surfaced the W44-119
//! regression).
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_120_distance_bisect \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_120_distance_bisect_2026-05-20.tsv

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
/// Coverage:
///   - terminal e8/e9 × d ∈ {0.5, 0.8, 1.0, 1.2, 1.4, 1.5, 2.0, 3.0,
///     4.0, 5.0, 6.0} (22 cells) — primary regression + win cluster
///   - GB82-SC screenshots × e8 × d=4 (10 cells) — preserve W44-117 wins
///   - 5 photo cells (e8 d=2 / d=4) — verify W44-118 gate still excludes
const CELLS: &[(&str, &str, u8, f32)] = &[
    // === Primary regression range (terminal e8/e9 × d=0.5..1.5) ===
    // The W44-119 regression cluster. We must close terminal d=0.8 SSIM2 -1.87.
    ("terminal.png", "gb82-sc", 8, 0.5),
    ("terminal.png", "gb82-sc", 8, 0.8),
    ("terminal.png", "gb82-sc", 8, 1.0),
    ("terminal.png", "gb82-sc", 8, 1.2),
    ("terminal.png", "gb82-sc", 8, 1.4),
    ("terminal.png", "gb82-sc", 8, 1.5),
    ("terminal.png", "gb82-sc", 9, 0.5),
    ("terminal.png", "gb82-sc", 9, 0.8),
    ("terminal.png", "gb82-sc", 9, 1.0),
    ("terminal.png", "gb82-sc", 9, 1.2),
    ("terminal.png", "gb82-sc", 9, 1.4),
    ("terminal.png", "gb82-sc", 9, 1.5),
    // === W44-117 win cluster (terminal e8/e9 × d=2..6) — must preserve ===
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
    // === Small GB82-SC screenshots × e8 d=4 — preserve W44-117 wins
    // for screenshot-class content other than terminal. Larger
    // screenshots (codec_wiki 2056×1606, imac_dark 1920×1200,
    // imac_g3 1920×1200) exceed the 2GB encoder memory budget at
    // e8 d=4 across all 5 modes equally (known limit per W44-118
    // memo — not a W44-120 regression), so we use the smaller
    // ones that fit. frymire.png is in gb82 (not gb82-sc).
    // imac_g3_strip is a strip variant we skip. ===
    ("windows95.png", "gb82-sc", 8, 4.0),
    ("imessage.png", "gb82-sc", 8, 4.0),
    ("gmessages.png", "gb82-sc", 8, 4.0),
    // === Photo cells — verify W44-118 gate still excludes ===
    ("1025469.png", "CID22", 8, 2.0),
    ("1025469.png", "CID22", 8, 4.0),
    ("1418519.png", "CID22", 8, 2.0),
    ("1418519.png", "CID22", 8, 4.0),
    ("1189261.png", "CID22", 8, 4.0),
];

/// W44-120 modes. A = legacy uniform-4 baseline (W44-117 forced off,
/// equivalent to pre-W44-117 main). B values are W44-120 distance
/// thresholds.
#[derive(Clone, Copy, Debug)]
struct Mode {
    name: &'static str,
    /// If `Some`, set `JXL_W44_117_DISABLE=<value>` before encoding.
    disable_w44_117: Option<&'static str>,
    /// If `Some`, set `JXL_W44_120_EPF_SEED_MIN_DISTANCE=<value>` before encoding.
    distance_threshold: Option<&'static str>,
}

const MODES: &[Mode] = &[
    // A: legacy uniform-4 (pre-W44-117 baseline). Reference.
    Mode {
        name: "A_legacy",
        disable_w44_117: Some("1"),
        distance_threshold: None,
    },
    // B0.8: W44-117 always on (matches pre-W44-120 main = W44-118 default).
    // The current production behaviour from W44-118 main. We want B1.0/B1.2/B1.5
    // to BEAT this on terminal d=0.8 without losing terminal d>=1.0 wins.
    Mode {
        name: "B0.8",
        disable_w44_117: None,
        distance_threshold: Some("0.8"),
    },
    // B1.0: candidate ship threshold.
    Mode {
        name: "B1.0",
        disable_w44_117: None,
        distance_threshold: Some("1.0"),
    },
    // B1.2: slightly tighter.
    Mode {
        name: "B1.2",
        disable_w44_117: None,
        distance_threshold: Some("1.2"),
    },
    // B1.5: most conservative. Should preserve d>=2 wins but lose
    // the d=1.0..1.4 cluster.
    Mode {
        name: "B1.5",
        disable_w44_117: None,
        distance_threshold: Some("1.5"),
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
    // SAFETY: single-threaded harness main loop (encoder uses internal
    // rayon pool but A/B is sequential).
    unsafe {
        std::env::remove_var("JXL_W44_117_DISABLE");
        std::env::remove_var("JXL_W44_120_EPF_SEED_MIN_DISTANCE");
        if let Some(v) = mode.disable_w44_117 {
            std::env::set_var("JXL_W44_117_DISABLE", v);
        }
        if let Some(v) = mode.distance_threshold {
            std::env::set_var("JXL_W44_120_EPF_SEED_MIN_DISTANCE", v);
        }
    }
    let score = score_cell(rgb, w, h, effort, d, orig_linear_img, orig_srgb_img);
    unsafe {
        std::env::remove_var("JXL_W44_117_DISABLE");
        std::env::remove_var("JXL_W44_120_EPF_SEED_MIN_DISTANCE");
    }
    score
}

fn main() {
    eprintln!("W44-120 distance-gate bisection: A_legacy + B0.8 + B1.0 + B1.2 + B1.5");
    eprintln!(
        "Cells: {} ({} modes per cell, interleaved)",
        CELLS.len(),
        MODES.len()
    );

    // Header — one bytes/bfly/ssim2 triple per mode.
    let mut header = String::from("image\teffort\tdistance");
    for m in MODES {
        header.push_str(&format!(
            "\t{}_bytes\t{}_bfly\t{}_ssim2",
            m.name, m.name, m.name
        ));
    }
    // Deltas: bytes %, bfly %, ssim2 abs, all vs A_legacy.
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
    // Stats per mode: (sum_bytes, sum_ssim2_delta_vs_A, sum_bfly_delta_pct_vs_A,
    //                  cells_with_data, count_ssim2_regression_gt_0_3_vs_A,
    //                  count_ssim2_regression_gt_0_3_vs_B08)
    let mut mode_stats: Vec<(usize, f64, f64, usize, usize, usize)> =
        vec![(0, 0.0, 0.0, 0, 0, 0); MODES.len()];

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

        if !images_cache.contains_key(&cache_key) {
            match image::open(&path) {
                Ok(img) => {
                    let (w, h) = img.dimensions();
                    let rgb = img.to_rgb8().into_raw();
                    let linear = srgb_u8_to_linear(&rgb, w, h);
                    let srgb_pixels: Vec<[u8; 3]> =
                        rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                    let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                    images_cache.insert(cache_key.clone(), (w, h, rgb, linear, srgb_img));
                }
                Err(e) => {
                    eprintln!("  decode png failed for {:?}: {} — skipping cell", path, e);
                    continue;
                }
            }
        }
        let (w, h, raw, orig_linear_img, orig_srgb_img) =
            images_cache.get(&cache_key).expect("just inserted");

        let mut scores: Vec<Option<Score>> = Vec::with_capacity(MODES.len());
        for m in MODES {
            scores.push(run_mode(
                m,
                raw,
                *w,
                *h,
                effort,
                d,
                orig_linear_img,
                orig_srgb_img,
            ));
        }

        // Reference for deltas: A_legacy (idx 0).
        let a = match &scores[0] {
            Some(s) => *s,
            None => {
                eprintln!("  A_legacy failed; skipping cell");
                continue;
            }
        };

        let mut row = format!("{}\te{}\t{}", image, effort, d);
        for s in &scores {
            match s {
                Some(s) => row.push_str(&format!(
                    "\t{}\t{:.4}\t{:.4}",
                    s.bytes, s.butteraugli, s.ssim2
                )),
                None => row.push_str("\tNA\tNA\tNA"),
            }
        }
        for (j, s) in scores.iter().enumerate().skip(1) {
            match s {
                Some(s) => {
                    let d_bytes = 100.0 * (s.bytes as f64 - a.bytes as f64) / a.bytes as f64;
                    let d_bfly = if a.butteraugli > 0.0 {
                        100.0 * (s.butteraugli - a.butteraugli) / a.butteraugli
                    } else {
                        f64::NAN
                    };
                    let d_ssim2 = s.ssim2 - a.ssim2;
                    row.push_str(&format!(
                        "\t{:+.3}\t{:+.3}\t{:+.3}",
                        d_bytes, d_bfly, d_ssim2
                    ));

                    // Accumulate per-mode stats.
                    mode_stats[j].0 += s.bytes;
                    mode_stats[j].1 += d_ssim2;
                    if !d_bfly.is_nan() {
                        mode_stats[j].2 += d_bfly;
                    }
                    mode_stats[j].3 += 1;
                    if d_ssim2 < -0.3 {
                        mode_stats[j].4 += 1;
                    }
                }
                None => row.push_str("\tNA\tNA\tNA"),
            }
        }
        // Also compute vs-B0.8 SSIM2 regressions (B0.8 is mode idx 1).
        // This tells us how each tighter threshold compares to the current
        // pre-W44-120 production default.
        if let Some(b08) = &scores[1] {
            for (j, s) in scores.iter().enumerate().skip(2) {
                if let Some(s) = s {
                    let d = s.ssim2 - b08.ssim2;
                    if d < -0.3 {
                        mode_stats[j].5 += 1;
                    }
                }
            }
        }
        // A_legacy bytes (mode 0) tracked separately.
        mode_stats[0].0 += a.bytes;
        mode_stats[0].3 += 1;

        println!("{}", row);
    }

    eprintln!();
    eprintln!("=== W44-120 BISECTION SUMMARY ===");
    eprintln!("(cells={}, modes={})", n_cells, MODES.len());
    eprintln!();
    eprintln!("Per-mode totals vs A_legacy:");
    eprintln!(
        "{:<10} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "mode", "bytes_total", "dB_pct", "mean_dB", "mean_dBfly", "mean_dSSIM2", "reg_vs_A"
    );
    let a_bytes_total = mode_stats[0].0;
    let a_cells = mode_stats[0].3;
    eprintln!(
        "{:<10} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        MODES[0].name,
        a_bytes_total,
        "(ref)",
        format!("{}", a_cells),
        "(ref)",
        "(ref)",
        "(ref)",
    );
    for (j, (bytes, sum_ssim2, sum_bfly, n, reg_vs_a, _reg_vs_b08)) in
        mode_stats.iter().enumerate().skip(1)
    {
        let bytes_delta_pct = 100.0 * (*bytes as f64 - a_bytes_total as f64) / a_bytes_total as f64;
        let mean_ssim2 = if *n > 0 { sum_ssim2 / *n as f64 } else { 0.0 };
        let mean_bfly = if *n > 0 { sum_bfly / *n as f64 } else { 0.0 };
        eprintln!(
            "{:<10} {:>12} {:>+10.3} {:>10} {:>+10.3} {:>+10.4} {:>10}",
            MODES[j].name, bytes, bytes_delta_pct, n, mean_bfly, mean_ssim2, reg_vs_a,
        );
    }
    eprintln!();
    eprintln!("Per-mode SSIM2 regressions vs B0.8 (current main):");
    for (j, (_bytes, _sum_ssim2, _sum_bfly, _n, _reg_vs_a, reg_vs_b08)) in
        mode_stats.iter().enumerate().skip(2)
    {
        eprintln!(
            "  {}: {} cells with SSIM2 < B0.8 by > 0.3",
            MODES[j].name, reg_vs_b08
        );
    }
    eprintln!();
    eprintln!("Acceptance gate: pick the lowest-threshold mode (favouring more W44-117 wins)");
    eprintln!("where terminal e8/e9 d=0.8 SSIM2 vs A_legacy >= -0.3 AND no new regressions");
    eprintln!("vs B0.8 (current main) on terminal d>=2 / GB82-SC / photos.");
}
