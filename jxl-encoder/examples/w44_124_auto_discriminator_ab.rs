// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-124 A/B: verify the auto-discriminator default-on behaviour.
//!
//! Three modes per cell:
//! - `baseline_off`: `with_dct32_keep_hint(Some(false))` — force-suppress
//!   (pre-W44-124 default = W44-68 behaviour, pre-W44-123 ALSO this).
//! - `auto_default`: `with_dct32_keep_hint(None)` — the new default that
//!   auto-discriminates via m3_colourfulness + edge_density.
//! - `keep_dct32`: `with_dct32_keep_hint(Some(true))` — force-keep
//!   (W44-123 measured opt-in mode).
//!
//! Expectations from W44-124 probe:
//! - codec_wiki: auto = keep_dct32 (FIRES); preserves W44-123 +1.40/+1.33/+0.90 wins
//! - terminal:   auto = baseline_off (m3=13.85 rejects); loses W44-123 +0.47 SSIM2 e8/e9
//! - 6 SCREEN regressions (graph/windows/imessage): auto = baseline_off; ZERO regression
//! - photos: auto = baseline_off; byte-identical
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!     --features 'parallel butteraugli-loop ssim2-loop' \
//!     --example w44_124_auto_discriminator_ab
//!
//! TSV: benchmarks/w44_124_auto_discriminator_ab_2026-05-20.tsv

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

// Target preservation: codec_wiki d=3 across e5/e6/e7 — must STAY a win
// (auto-discriminator must fire on codec_wiki).
const CODEC_WIKI_EFFORTS: &[u8] = &[5, 6, 7];
const CODEC_WIKI_DISTANCES: &[f32] = &[3.0];

// Terminal: W44-123 preserved win at e8/e9 d=4 (+0.47 SSIM2). Auto-
// discriminator will REJECT (m3=13.85 < 60). We accept losing this opt-in
// win in exchange for the codec_wiki default-on. Reported for visibility.
const TERMINAL_EFFORTS: &[u8] = &[5, 6, 7, 8, 9];
const TERMINAL_DISTANCES: &[f32] = &[4.0];

// 6 W44-123 regression screens — must be byte-identical under auto.
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

// Photo spot-check: must be byte-identical under auto (proxies will reject
// every CID22 photo per the W44-124 probe).
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
    BaselineOff, // with_dct32_keep_hint(Some(false)) — W44-68 force-suppress
    AutoDefault, // with_dct32_keep_hint(None) — W44-124 auto-discriminator
    KeepDct32,   // with_dct32_keep_hint(Some(true)) — W44-123 force-keep
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::BaselineOff => "baseline_off",
            Mode::AutoDefault => "auto_default",
            Mode::KeepDct32 => "keep_dct32",
        }
    }
    fn hint(self) -> Option<bool> {
        match self {
            Mode::BaselineOff => Some(false),
            Mode::AutoDefault => None,
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
    let out_path = PathBuf::from("benchmarks/w44_124_auto_discriminator_ab_2026-05-20.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/w44_124_auto_discriminator_ab_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).unwrap();
    writeln!(
        out,
        "class\timage\teffort\tdistance\tbaseline_off_bytes\tauto_default_bytes\tkeep_dct32_bytes\tbaseline_off_bfly\tauto_default_bfly\tkeep_dct32_bfly\tbaseline_off_ssim2\tauto_default_ssim2\tkeep_dct32_ssim2\tauto_vs_baseline_off_bytes_pct\tauto_vs_baseline_off_ssim2\tauto_vs_keep_dct32_bytes_pct\tauto_vs_keep_dct32_ssim2"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();
    let cells = build_cell_list(&corpus);
    eprintln!("Total cells: {}", cells.len());

    let mut codec_wiki_summary: Vec<(u8, f32, f64, f64)> = Vec::new();
    let mut terminal_summary: Vec<(u8, f32, f64, f64)> = Vec::new();
    let mut regression_summary: Vec<(String, u8, f32, f64, f64)> = Vec::new();
    let mut photo_summary: Vec<(String, u8, f32, f64, f64)> = Vec::new();

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

        // Interleaved 3-mode pairing
        let mut bo_bytes_all: Vec<usize> = Vec::with_capacity(TRIALS);
        let mut ad_bytes_all: Vec<usize> = Vec::with_capacity(TRIALS);
        let mut kd_bytes_all: Vec<usize> = Vec::with_capacity(TRIALS);
        let mut last_bo_bytes: Vec<u8> = Vec::new();
        let mut last_ad_bytes: Vec<u8> = Vec::new();
        let mut last_kd_bytes: Vec<u8> = Vec::new();
        for t in 0..TRIALS {
            let (bo, _) = encode_with_mode(
                &cell.rgb_u8,
                cell.width,
                cell.height,
                cell.effort,
                cell.distance,
                Mode::BaselineOff,
            );
            let (ad, _) = encode_with_mode(
                &cell.rgb_u8,
                cell.width,
                cell.height,
                cell.effort,
                cell.distance,
                Mode::AutoDefault,
            );
            let (kd, _) = encode_with_mode(
                &cell.rgb_u8,
                cell.width,
                cell.height,
                cell.effort,
                cell.distance,
                Mode::KeepDct32,
            );
            bo_bytes_all.push(bo.len());
            ad_bytes_all.push(ad.len());
            kd_bytes_all.push(kd.len());
            if t == TRIALS - 1 {
                last_bo_bytes = bo;
                last_ad_bytes = ad;
                last_kd_bytes = kd;
            }
        }
        let bo_min = *bo_bytes_all.iter().min().unwrap();
        let ad_min = *ad_bytes_all.iter().min().unwrap();
        let kd_min = *kd_bytes_all.iter().min().unwrap();

        let (_, bo_bfly, bo_ssim2) =
            compute_metrics(&last_bo_bytes, &cell.orig_lin, &cell.orig_srgb, &bparams);
        let (_, ad_bfly, ad_ssim2) =
            compute_metrics(&last_ad_bytes, &cell.orig_lin, &cell.orig_srgb, &bparams);
        let (_, kd_bfly, kd_ssim2) =
            compute_metrics(&last_kd_bytes, &cell.orig_lin, &cell.orig_srgb, &bparams);

        let auto_vs_bo_bytes = (ad_min as f64 - bo_min as f64) / bo_min as f64 * 100.0;
        let auto_vs_bo_ssim2 = ad_ssim2 - bo_ssim2;
        let auto_vs_kd_bytes = (ad_min as f64 - kd_min as f64) / kd_min as f64 * 100.0;
        let auto_vs_kd_ssim2 = ad_ssim2 - kd_ssim2;

        writeln!(
            out,
            "{}\t{}\te{}\t{:.2}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.4}\t{:.3}\t{:.4}",
            cell.class,
            cell.label,
            cell.effort,
            cell.distance,
            bo_min,
            ad_min,
            kd_min,
            bo_bfly,
            ad_bfly,
            kd_bfly,
            bo_ssim2,
            ad_ssim2,
            kd_ssim2,
            auto_vs_bo_bytes,
            auto_vs_bo_ssim2,
            auto_vs_kd_bytes,
            auto_vs_kd_ssim2,
        )
        .unwrap();
        out.flush().ok();

        match cell.class {
            "CODEC_WIKI" => codec_wiki_summary.push((
                cell.effort,
                cell.distance,
                auto_vs_bo_ssim2,
                auto_vs_bo_bytes,
            )),
            "TERMINAL" => terminal_summary.push((
                cell.effort,
                cell.distance,
                auto_vs_bo_ssim2,
                auto_vs_bo_bytes,
            )),
            "SCREEN" => regression_summary.push((
                cell.label.clone(),
                cell.effort,
                cell.distance,
                auto_vs_bo_ssim2,
                auto_vs_bo_bytes,
            )),
            "PHOTO" => photo_summary.push((
                cell.label.clone(),
                cell.effort,
                cell.distance,
                auto_vs_bo_ssim2,
                auto_vs_bo_bytes,
            )),
            _ => {}
        }
    }

    drop(out);
    std::fs::rename(&staging, &out_path).expect("atomic rename failed");
    eprintln!("Wrote {}", out_path.display());

    eprintln!("\n=== CODEC_WIKI summary (auto_vs_baseline_off) ===");
    for (e, d, ss2, b) in &codec_wiki_summary {
        eprintln!("  e{} d={:.1}  Δss2={:+.4}  Δbytes={:+.3}%", e, d, ss2, b);
    }

    eprintln!("\n=== TERMINAL summary (auto_vs_baseline_off) ===");
    for (e, d, ss2, b) in &terminal_summary {
        eprintln!("  e{} d={:.1}  Δss2={:+.4}  Δbytes={:+.3}%", e, d, ss2, b);
    }

    eprintln!("\n=== SCREEN regression check (auto_vs_baseline_off — must be ~0) ===");
    let mut bad = 0;
    for (img, e, d, ss2, b) in &regression_summary {
        let flag = if ss2.abs() > 0.01 || b.abs() > 0.05 {
            bad += 1;
            "BAD"
        } else {
            "OK "
        };
        eprintln!(
            "  {} {} e{} d={:.1}  Δss2={:+.4}  Δbytes={:+.3}%",
            flag, img, e, d, ss2, b
        );
    }
    eprintln!("SCREEN bad-cell count: {}", bad);

    eprintln!("\n=== PHOTO check (auto_vs_baseline_off — must be 0) ===");
    let mut photo_bad = 0;
    for (img, e, d, ss2, b) in &photo_summary {
        let flag = if ss2.abs() > 0.001 || b.abs() > 0.001 {
            photo_bad += 1;
            "BAD"
        } else {
            "OK "
        };
        eprintln!(
            "  {} {} e{} d={:.1}  Δss2={:+.4}  Δbytes={:+.3}%",
            flag, img, e, d, ss2, b
        );
    }
    eprintln!("PHOTO bad-cell count: {}", photo_bad);

    let _ = bad;
    let _ = photo_bad;
}

fn _label_unused(m: Mode) -> &'static str {
    m.label()
}
