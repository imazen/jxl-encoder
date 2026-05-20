// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-123 A/B: clone of W44-122 harness retargeted to keep_dct32 lever.
//!
//! Per W44-122 honest-stop: admit-DCT64 closes only 22-52% of the codec_wiki
//! d=3 SSIM2 cluster AND introduces +29% butteraugli regression on terminal
//! e8/e9 + 11 other screen cells regress by -0.30 to -1.44 SSIM2.
//!
//! Per W44-122 "Recommended W44-123" + dispatch task: the narrower lever is
//! to re-enable `try_dct32 = true` (which allows `find_best_32x32_transform`
//! to run, evaluating DCT32X32 vs 4×DCT16X16) while KEEPING
//! `try_dct64 = false` (preserves W44-104 dead-letter ruling).
//!
//! Mechanism: W44-65 fires `try_dct64 = false`, W44-68 ALSO fires
//! `try_dct32 = false`. The new `LossyConfig::with_dct32_keep_hint(Some(true))`
//! API decouples the gates — caller asks for "drop DCT64 (W44-65) but KEEP
//! DCT32" so `find_best_32x32_transform` runs.
//!
//! Modes per cell:
//! - `baseline`: default (W44-65 + W44-68 fire together → both try_dct32 and try_dct64 off)
//! - `keep_dct32`: `LossyConfig::with_dct32_keep_hint(Some(true))`
//!   (try_dct32=true, try_dct64=false)
//!
//! Acceptance gates:
//! - codec_wiki d=3 SSIM2 improves by ≥+1.5 pts (W44-121 deficit -2.7 to -3.4)
//! - terminal d=4 SSIM2 preserved (W44-117/118/120)
//! - bytes regression ≤ +5%
//! - Zero NEW FIXED→OPEN flips on 30+ spot photo cells
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!     --features 'parallel butteraugli-loop ssim2-loop' \
//!     --example w44_123_keep_dct32_ab
//!
//! TSV: benchmarks/w44_123_keep_dct32_ab_2026-05-20.tsv

#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

// ── Cell sets ───────────────────────────────────────────────────────────────

const CODEC_WIKI_EFFORTS: &[u8] = &[5, 6, 7];
const CODEC_WIKI_DISTANCES: &[f32] = &[3.0];

const TERMINAL_EFFORTS: &[u8] = &[5, 6, 7, 8, 9];
const TERMINAL_DISTANCES: &[f32] = &[4.0];

const OTHER_SCREENS: &[&str] = &[
    "imac_g3.png",
    "imac_dark.png",
    "windows.png",
    "windows95.png",
    "imessage.png",
    "graph.png",
];
const SCREEN_EFFORTS: &[u8] = &[7];
const SCREEN_DISTANCES: &[f32] = &[2.0, 4.0, 6.0];

const PHOTO_SET: &[&str] = &[
    "1025469.png",
    "1418519.png",
    "1420710.png",
    "1531677.png",
    "1189261.png",
    "1044329.png",
    "2389166.png",
    "3637739.png",
];
const PHOTO_EFFORTS: &[u8] = &[7];
const PHOTO_DISTANCES: &[f32] = &[1.0, 3.0, 5.0];

const TRIALS: usize = 2;
const CORPUS_BASE: &str = "/home/lilith/work/codec-corpus";

#[derive(Copy, Clone, Debug)]
enum Mode {
    Baseline,  // default (W44-65 + W44-68 both fire)
    KeepDct32, // dct32_keep_hint = Some(true) (try_dct32 stays true)
}

impl Mode {
    #[allow(dead_code)]
    fn label(self) -> &'static str {
        match self {
            Mode::Baseline => "baseline",
            Mode::KeepDct32 => "keep_dct32",
        }
    }
    fn hint(self) -> Option<bool> {
        match self {
            Mode::Baseline => None,
            Mode::KeepDct32 => Some(true),
        }
    }
}

fn decode_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn encode_with_mode(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
) -> (Vec<u8>, f64) {
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct32_keep_hint: mode.hint(), ..Default::default() });
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    let t = Instant::now();
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&lim)
        .encode(rgb_u8)
        .expect("encode failed");
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn compute_metrics(
    bytes: &[u8],
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    bparams: &ButteraugliParams,
) -> (usize, f64, f64) {
    let (dw, dh, dec) = decode_linear(bytes).expect("decode failed");
    let dec_pix: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
    let dec_lin = Img::new(dec_pix, dw, dh);
    let bfly = butteraugli_linear(orig_lin.as_ref(), dec_lin.as_ref(), bparams)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);
    let dec_srgb: Vec<[u8; 3]> = dec
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), Img::new(dec_srgb, dw, dh).as_ref())
            .unwrap_or(f64::NAN);
    (bytes.len(), bfly, ssim2)
}

struct Cell {
    class: &'static str,
    label: String,
    effort: u8,
    distance: f32,
    rgb_u8: Vec<u8>,
    width: u32,
    height: u32,
    orig_lin: Img<Vec<RGB<f32>>>,
    orig_srgb: Img<Vec<[u8; 3]>>,
}

fn try_resolve(corpus: &PathBuf, name: &str) -> Option<PathBuf> {
    let p1 = corpus.join("CID22/CID22-512/validation").join(name);
    if p1.exists() {
        return Some(p1);
    }
    let p2 = corpus.join("gb82-sc").join(name);
    if p2.exists() {
        return Some(p2);
    }
    None
}

fn load_cell(
    class: &'static str,
    label: &str,
    corpus: &PathBuf,
    effort: u8,
    distance: f32,
) -> Option<Cell> {
    let path = try_resolve(corpus, label)?;
    let img = image::open(&path).ok()?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let rgb_u8: Vec<u8> = rgb.as_raw().clone();
    let linear: Vec<RGB<f32>> = rgb
        .pixels()
        .map(|p| {
            RGB::new(
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            )
        })
        .collect();
    let orig_lin = Img::new(linear, w as usize, h as usize);
    let srgb_arr: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    let orig_srgb = Img::new(srgb_arr, w as usize, h as usize);
    Some(Cell {
        class,
        label: label.to_string(),
        effort,
        distance,
        rgb_u8,
        width: w,
        height: h,
        orig_lin,
        orig_srgb,
    })
}

fn build_cell_list(corpus: &PathBuf) -> Vec<Cell> {
    let mut cells: Vec<Cell> = Vec::new();
    for &e in CODEC_WIKI_EFFORTS {
        for &d in CODEC_WIKI_DISTANCES {
            if let Some(c) = load_cell("CODEC_WIKI", "codec_wiki.png", corpus, e, d) {
                cells.push(c);
            }
        }
    }
    for &e in TERMINAL_EFFORTS {
        for &d in TERMINAL_DISTANCES {
            if let Some(c) = load_cell("TERMINAL", "terminal.png", corpus, e, d) {
                cells.push(c);
            }
        }
    }
    for name in OTHER_SCREENS {
        for &e in SCREEN_EFFORTS {
            for &d in SCREEN_DISTANCES {
                if let Some(c) = load_cell("SCREEN", name, corpus, e, d) {
                    cells.push(c);
                }
            }
        }
    }
    for name in PHOTO_SET {
        for &e in PHOTO_EFFORTS {
            for &d in PHOTO_DISTANCES {
                if let Some(c) = load_cell("PHOTO", name, corpus, e, d) {
                    cells.push(c);
                }
            }
        }
    }
    cells
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| String::from(CORPUS_BASE)),
    );
    let out_path = PathBuf::from("benchmarks/w44_123_keep_dct32_ab_2026-05-20.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/w44_123_keep_dct32_ab_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).unwrap();
    writeln!(
        out,
        "class\timage\teffort\tdistance\tbaseline_bytes\tkeep_dct32_bytes\tbaseline_bfly\tkeep_dct32_bfly\tbaseline_ssim2\tkeep_dct32_ssim2\tbaseline_ms\tkeep_dct32_ms\tbytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();
    let cells = build_cell_list(&corpus);
    eprintln!("Total cells: {}", cells.len());

    let mut codec_wiki_results: Vec<(u8, f32, f64, f64)> = Vec::new();
    let mut terminal_results: Vec<(u8, f32, f64, f64)> = Vec::new();
    let mut screen_regressions: Vec<(String, u8, f32, f64, f64)> = Vec::new();
    let mut photo_regressions: Vec<(String, u8, f32, f64, f64)> = Vec::new();
    let mut totals_bytes_delta = 0.0_f64;
    let mut totals_ssim2_delta = 0.0_f64;
    let mut count = 0;

    for (i, cell) in cells.iter().enumerate() {
        eprintln!(
            "[{:>3}/{}] {} {} e{} d={:.2}",
            i + 1,
            cells.len(),
            cell.class,
            cell.label,
            cell.effort,
            cell.distance
        );

        let mut b_bytes_all: Vec<usize> = Vec::with_capacity(TRIALS);
        let mut a_bytes_all: Vec<usize> = Vec::with_capacity(TRIALS);
        let mut b_ms_all: Vec<f64> = Vec::with_capacity(TRIALS);
        let mut a_ms_all: Vec<f64> = Vec::with_capacity(TRIALS);
        let mut last_b_bytes: Vec<u8> = Vec::new();
        let mut last_a_bytes: Vec<u8> = Vec::new();
        for t in 0..TRIALS {
            let (b1, m1) = encode_with_mode(
                &cell.rgb_u8,
                cell.width,
                cell.height,
                cell.effort,
                cell.distance,
                Mode::Baseline,
            );
            let (a1, m2) = encode_with_mode(
                &cell.rgb_u8,
                cell.width,
                cell.height,
                cell.effort,
                cell.distance,
                Mode::KeepDct32,
            );
            b_bytes_all.push(b1.len());
            a_bytes_all.push(a1.len());
            b_ms_all.push(m1);
            a_ms_all.push(m2);
            if t == TRIALS - 1 {
                last_b_bytes = b1;
                last_a_bytes = a1;
            }
        }
        let b_bytes_min = *b_bytes_all.iter().min().unwrap();
        let a_bytes_min = *a_bytes_all.iter().min().unwrap();
        let b_ms_min = b_ms_all.iter().cloned().fold(f64::INFINITY, f64::min);
        let a_ms_min = a_ms_all.iter().cloned().fold(f64::INFINITY, f64::min);

        let (_, b_bfly, b_ssim2) =
            compute_metrics(&last_b_bytes, &cell.orig_lin, &cell.orig_srgb, &bparams);
        let (_, a_bfly, a_ssim2) =
            compute_metrics(&last_a_bytes, &cell.orig_lin, &cell.orig_srgb, &bparams);

        let bytes_delta = (a_bytes_min as f64 - b_bytes_min as f64) / b_bytes_min as f64 * 100.0;
        let bfly_delta = if b_bfly > 0.0 {
            (a_bfly - b_bfly) / b_bfly * 100.0
        } else {
            0.0
        };
        let ssim2_delta = a_ssim2 - b_ssim2;

        writeln!(
            out,
            "{}\t{}\te{}\t{:.2}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.0}\t{:.0}\t{:.3}\t{:.3}\t{:.4}",
            cell.class,
            cell.label,
            cell.effort,
            cell.distance,
            b_bytes_min,
            a_bytes_min,
            b_bfly,
            a_bfly,
            b_ssim2,
            a_ssim2,
            b_ms_min,
            a_ms_min,
            bytes_delta,
            bfly_delta,
            ssim2_delta,
        )
        .unwrap();
        out.flush().ok();

        totals_bytes_delta += bytes_delta;
        totals_ssim2_delta += ssim2_delta;
        count += 1;

        if cell.class == "CODEC_WIKI" {
            codec_wiki_results.push((cell.effort, cell.distance, ssim2_delta, bytes_delta));
        } else if cell.class == "TERMINAL" {
            terminal_results.push((cell.effort, cell.distance, ssim2_delta, bytes_delta));
        } else if cell.class == "SCREEN" && (ssim2_delta < -0.3 || bytes_delta > 5.0) {
            screen_regressions.push((
                cell.label.clone(),
                cell.effort,
                cell.distance,
                ssim2_delta,
                bytes_delta,
            ));
        } else if cell.class == "PHOTO" && (ssim2_delta < -0.3 || bytes_delta > 5.0) {
            photo_regressions.push((
                cell.label.clone(),
                cell.effort,
                cell.distance,
                ssim2_delta,
                bytes_delta,
            ));
        }
    }

    drop(out);
    std::fs::copy(&staging, &out_path).ok();
    eprintln!(
        "\n=== TOTALS over {} cells ===\nbytes_delta_avg = {:.3}%\nssim2_delta_avg = {:.4}",
        count,
        totals_bytes_delta / count as f64,
        totals_ssim2_delta / count as f64
    );
    eprintln!("\n=== codec_wiki d=3 deltas (TARGET) ===");
    for (e, d, s, b) in &codec_wiki_results {
        eprintln!(
            "  codec_wiki e{} d={:.1}: ssim2_delta={:+.4}  bytes_delta={:+.2}%",
            e, d, s, b
        );
    }
    eprintln!("\n=== Terminal deltas (preservation gate) ===");
    for (e, d, s, b) in &terminal_results {
        eprintln!(
            "  terminal e{} d={:.1}: ssim2_delta={:+.4}  bytes_delta={:+.2}%",
            e, d, s, b
        );
    }
    eprintln!("\n=== Screen regressions (ssim2<-0.3 or bytes>+5%) ===");
    for (n, e, d, s, b) in &screen_regressions {
        eprintln!(
            "  {} e{} d={:.1}: ssim2_delta={:+.4} bytes_delta={:+.2}%",
            n, e, d, s, b
        );
    }
    eprintln!("\n=== Photo regressions (ssim2<-0.3 or bytes>+5%) ===");
    for (n, e, d, s, b) in &photo_regressions {
        eprintln!(
            "  {} e{} d={:.1}: ssim2_delta={:+.4} bytes_delta={:+.2}%",
            n, e, d, s, b
        );
    }
    eprintln!("\nTSV: {}", out_path.display());
}
