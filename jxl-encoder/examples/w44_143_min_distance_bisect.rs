// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-143 bisect: sweep `W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE` candidates
//! `{2.0 (current), 1.8, 1.6, 1.4, 1.2}` to see whether widening the lower
//! bound closes the codec_wiki e8/e9 d=1.6/1.8 SSIM2 regression cluster
//! (W44-142 attribution finding) WITHOUT regressing:
//!
//! - W44-142 wins at d=1.0..1.4 on codec_wiki (preserved)
//! - W44-124 wins at d=2.5/3.0 on codec_wiki (preserved)
//! - W44-135 protection at d=4/5/6 on codec_wiki (no new -SSIM2 regression)
//! - W44-140 wins on terminal d=1.0-1.6 (preserved)
//! - Photo byte-identical (proxies reject)
//!
//! Mechanism: `JXL_W44_143_MIN_DISTANCE=<f32>` env-var override (added by
//! W44-143 in `vardct/encoder.rs`). Per-cell interleaved A/B/C/D/E across
//! the 5 candidates so cache + scheduler state is amortized across modes
//! (rather than per-mode bias).
//!
//! Cell sets (37 cells × 5 modes = 185 encodes):
//! - codec_wiki × {e8, e9} × {d=1.0, 1.2, 1.4, 1.6, 1.8, 2.0}   (12 cells)
//! - codec_wiki × e7 × {d=2.5, 3.0, 4.0, 5.0, 6.0}              ( 5 cells)
//! - terminal   × {e8, e9} × {d=1.0, 1.2, 1.4, 1.6, 1.8, 2.0}   (12 cells)
//! - imac_g3    × e5 × d=2.0                                    ( 1 cell)
//! - 1418519    × e7 × d=4.0                                    ( 1 cell)
//! - 1420710    × e7 × d=3.0                                    ( 1 cell, gate-fire photo check)
//! - 1531677    × e7 × d=5.0                                    ( 1 cell)
//! - codec_wiki × e7 × {d=1.0, 1.2, 1.4, 1.6, 1.8, 2.0}         ( 4 cells, e7 fill — but we already have 5+12)
//!
//! Actually keep tight: 31 cells (12+5+12+1+1).
//!
//! Build / run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run -p jxl-encoder --release \
//!     --features 'parallel butteraugli-loop ssim2-loop' \
//!     --example w44_143_min_distance_bisect
//!
//! TSV: benchmarks/w44_143_min_distance_bisect_2026-05-20.tsv

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

// ── Candidates (MIN_DISTANCE values for JXL_W44_143_MIN_DISTANCE env var) ──
// Index 0 = current main (2.0), then decreasing.
const CANDIDATES: &[(f32, &str)] = &[
    (2.0, "min2_0"), // baseline = current main behaviour
    (1.8, "min1_8"),
    (1.6, "min1_6"),
    (1.4, "min1_4"),
    (1.2, "min1_2"),
];

// ── Cells ───────────────────────────────────────────────────────────────────

// codec_wiki e8/e9 × {1.0..2.0}: primary target = d=1.6/1.8 (per W44-142
// attribution memo), must preserve d=1.0-1.4 W44-142 wins.
const CW_LOWD_EFFORTS: &[u8] = &[8, 9];
const CW_LOWD_DISTANCES: &[f32] = &[1.0, 1.2, 1.4, 1.6, 1.8, 2.0];

// codec_wiki e7 high-d: W44-124 wins at d=2.5/3.0 and W44-135 protection at
// d=4/5/6 (must stay byte-identical to current main when not lowering MIN).
const CW_HIGHD_EFFORTS: &[u8] = &[7];
const CW_HIGHD_DISTANCES: &[f32] = &[2.5, 3.0, 4.0, 5.0, 6.0];

// terminal e8/e9 × {1.0..2.0}: W44-140 wins protection (m3=13.85 < 60 should
// reject regardless of MIN_DISTANCE → byte-identical across all candidates).
const TERM_EFFORTS: &[u8] = &[8, 9];
const TERM_DISTANCES: &[f32] = &[1.0, 1.2, 1.4, 1.6, 1.8, 2.0];

// Photo sanity: gate must not fire on photo (ed>=0.16 fails ed gate). Pick
// 1418519 e7 d=4 which is a known stable cell.
const PHOTO_CELL: (&str, u8, f32) = ("1418519.png", 7, 4.0);

const TRIALS: usize = 1;
const CORPUS_BASE: &str = "/home/lilith/work/codec-corpus";

#[derive(Clone, Debug)]
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

fn encode_with_min_distance(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    min_distance_override: f32,
) -> (Vec<u8>, f64) {
    // Set the env var per-cell-per-mode. Use unsafe-set/remove in the
    // narrow surrounding scope. (Required because Rust 2024 marks set_var
    // unsafe.)
    // SAFETY: single-threaded test harness; no other code reads this var
    // concurrently.
    unsafe {
        std::env::set_var("JXL_W44_143_MIN_DISTANCE", format!("{min_distance_override}"));
    }
    let cfg = LossyConfig::new(d).with_effort(effort);
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    let t = Instant::now();
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&lim)
        .encode(rgb_u8)
        .expect("encode failed");
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    unsafe {
        std::env::remove_var("JXL_W44_143_MIN_DISTANCE");
    }
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
    for &e in CW_LOWD_EFFORTS {
        for &d in CW_LOWD_DISTANCES {
            if let Some(c) = load_cell("CW_LOWD", "codec_wiki.png", corpus, e, d) {
                cells.push(c);
            }
        }
    }
    for &e in CW_HIGHD_EFFORTS {
        for &d in CW_HIGHD_DISTANCES {
            if let Some(c) = load_cell("CW_HIGHD", "codec_wiki.png", corpus, e, d) {
                cells.push(c);
            }
        }
    }
    for &e in TERM_EFFORTS {
        for &d in TERM_DISTANCES {
            if let Some(c) = load_cell("TERM", "terminal.png", corpus, e, d) {
                cells.push(c);
            }
        }
    }
    let (name, e, d) = PHOTO_CELL;
    if let Some(c) = load_cell("PHOTO", name, corpus, e, d) {
        cells.push(c);
    }
    cells
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| String::from(CORPUS_BASE)),
    );
    let out_path = PathBuf::from("benchmarks/w44_143_min_distance_bisect_2026-05-20.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/w44_143_min_distance_bisect_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).unwrap();

    // header
    let mut header = String::from("class\timage\teffort\tdistance");
    for (_, lbl) in CANDIDATES {
        header.push_str(&format!("\t{lbl}_bytes"));
    }
    for (_, lbl) in CANDIDATES {
        header.push_str(&format!("\t{lbl}_bfly"));
    }
    for (_, lbl) in CANDIDATES {
        header.push_str(&format!("\t{lbl}_ssim2"));
    }
    for (_, lbl) in CANDIDATES {
        header.push_str(&format!("\t{lbl}_ms"));
    }
    writeln!(out, "{header}").unwrap();

    let bparams = ButteraugliParams::default();
    let cells = build_cell_list(&corpus);
    eprintln!(
        "Total cells: {} | candidates: {} | encodes: {}",
        cells.len(),
        CANDIDATES.len(),
        cells.len() * CANDIDATES.len()
    );

    let start = Instant::now();
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

        let mut per_cand_bytes: Vec<usize> = Vec::with_capacity(CANDIDATES.len());
        let mut per_cand_bfly: Vec<f64> = Vec::with_capacity(CANDIDATES.len());
        let mut per_cand_ssim2: Vec<f64> = Vec::with_capacity(CANDIDATES.len());
        let mut per_cand_ms: Vec<f64> = Vec::with_capacity(CANDIDATES.len());

        for (min_d, lbl) in CANDIDATES {
            let mut bytes_runs: Vec<usize> = Vec::with_capacity(TRIALS);
            let mut ms_runs: Vec<f64> = Vec::with_capacity(TRIALS);
            let mut last_bytes: Vec<u8> = Vec::new();
            for t in 0..TRIALS {
                let (b, ms) = encode_with_min_distance(
                    &cell.rgb_u8,
                    cell.width,
                    cell.height,
                    cell.effort,
                    cell.distance,
                    *min_d,
                );
                bytes_runs.push(b.len());
                ms_runs.push(ms);
                if t == TRIALS - 1 {
                    last_bytes = b;
                }
            }
            let (_, bfly, ssim2) =
                compute_metrics(&last_bytes, &cell.orig_lin, &cell.orig_srgb, &bparams);
            let min_bytes = *bytes_runs.iter().min().unwrap();
            let min_ms = ms_runs.iter().cloned().fold(f64::INFINITY, f64::min);
            per_cand_bytes.push(min_bytes);
            per_cand_bfly.push(bfly);
            per_cand_ssim2.push(ssim2);
            per_cand_ms.push(min_ms);
            eprintln!(
                "    {lbl} bytes={} bfly={:.4} ssim2={:.4} ms={:.1}",
                min_bytes, bfly, ssim2, min_ms
            );
        }

        let mut row = format!(
            "{}\t{}\t{}\t{:.2}",
            cell.class, cell.label, cell.effort, cell.distance
        );
        for &b in &per_cand_bytes {
            row.push_str(&format!("\t{b}"));
        }
        for &b in &per_cand_bfly {
            row.push_str(&format!("\t{:.5}", b));
        }
        for &s in &per_cand_ssim2 {
            row.push_str(&format!("\t{:.5}", s));
        }
        for &m in &per_cand_ms {
            row.push_str(&format!("\t{:.1}", m));
        }
        writeln!(out, "{row}").unwrap();
        out.flush().unwrap();
    }

    drop(out);
    std::fs::copy(&staging, &out_path).unwrap();
    eprintln!(
        "Wrote {} ({} cells, total wall {:.1}s)",
        out_path.display(),
        cells.len(),
        start.elapsed().as_secs_f64()
    );
}
