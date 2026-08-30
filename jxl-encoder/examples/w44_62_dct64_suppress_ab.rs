// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-62 A/B: probe whether forcing `try_dct64 = false` reduces bytes on the
//! 20 remaining OPEN cells in `cjxl_parity_ledger_2026-05-19_w44_61.tsv`.
//!
//! Hypothesis (from W44-31 strategy-distribution analysis): ours over-picks
//! DCT32/DCT64 vs cjxl on smooth photo high-d cells. W44-61 added more
//! DCT64X32 merges (closed several cells), but the residual may still be
//! DCT64-over-pick. If forcing DCT64 off REDUCES bytes on the OPEN cells,
//! a content-aware DCT64-gate (analog of high_d_photo_hint) is the lever.
//! If forcing DCT64 off REGRESSES bytes (or is neutral), the residual is
//! genuinely structural and the post-W44-61 ceiling holds.
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!       --features '__expert butteraugli-loop ssim2-loop parallel' \
//!       --example w44_62_dct64_suppress_ab

use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{LossyConfig, PixelLayout};

// 20 OPEN cells, sourced from cjxl_parity_ledger_2026-05-19_w44_61.tsv.
// (image, effort, distance, ours_baseline_bytes_w44_61).
const CELLS: &[(&str, u8, f32, usize)] = &[
    ("1189261.png", 5, 4.0, 23693),
    ("1189261.png", 8, 3.0, 28461),
    ("1189261.png", 9, 3.0, 28715),
    ("1420710.png", 5, 5.0, 24268),
    ("1420710.png", 5, 6.0, 21299),
    ("1420710.png", 6, 5.0, 24467),
    ("1420710.png", 6, 6.0, 21509),
    ("1420710.png", 7, 5.0, 24651),
    ("1420710.png", 7, 6.0, 21295),
    ("1420710.png", 8, 5.0, 22648),
    ("1420710.png", 9, 5.0, 22701),
    ("1531677.png", 5, 5.0, 21012),
    ("1531677.png", 5, 6.0, 18262),
    ("1531677.png", 6, 5.0, 21249),
    ("1531677.png", 6, 6.0, 18485),
    ("1531677.png", 8, 5.0, 19528),
    ("1531677.png", 9, 5.0, 19586),
    ("codec_wiki.png", 7, 4.0, 65830),
    ("codec_wiki.png", 7, 5.0, 59239),
    ("codec_wiki.png", 7, 6.0, 53936),
    // Other screenshots at e7 d=4/5/6 — currently FIXED (and BEATING cjxl).
    // Probe: does suppressing DCT64 regress these? If yes, the lever is
    // codec_wiki-specific and needs a tight discriminator.
    ("imac_g3.png", 7, 4.0, 0),
    ("imac_g3.png", 7, 5.0, 0),
    ("imac_g3.png", 7, 6.0, 0),
    ("terminal.png", 7, 4.0, 0),
    ("terminal.png", 7, 5.0, 0),
    ("terminal.png", 7, 6.0, 0),
];

const CORPUS_BASE: &str = "/home/lilith/work/codec-corpus";

fn corpus_path(name: &str) -> PathBuf {
    let cid22 = PathBuf::from(CORPUS_BASE)
        .join("CID22/CID22-512/validation")
        .join(name);
    if cid22.exists() {
        return cid22;
    }
    let gb82 = PathBuf::from(CORPUS_BASE).join("gb82-sc").join(name);
    if gb82.exists() {
        return gb82;
    }
    // Walk the corpus base for the file as a fallback.
    let mut stack = vec![PathBuf::from(CORPUS_BASE)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
                return p;
            }
        }
    }
    panic!("could not locate {} under {}", name, CORPUS_BASE);
}

fn load_png(path: &PathBuf) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("open {:?}: {}", path, e));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.as_raw().clone(), w, h)
}

fn encode_default(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> (Vec<u8>, f64) {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1);
    let start = Instant::now();
    let bytes = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode default");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn encode_no_dct64(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> (Vec<u8>, f64) {
    let mut params = LossyInternalParams::default();
    params.try_dct64 = Some(false);
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode no_dct64");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn encode_no_dct32_dct64(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> (Vec<u8>, f64) {
    let mut params = LossyInternalParams::default();
    params.try_dct32 = Some(false);
    params.try_dct64 = Some(false);
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode no_dct32_dct64");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn main() {
    println!(
        "W44-62 A/B: test `try_dct64=Some(false)` and `try_dct32+64=Some(false)` on 20 OPEN cells"
    );
    println!();
    println!(
        "cell                              bytes_A   bytes_B  ΔB_pct   bytes_C  ΔC_pct  ms_A   ms_B   ms_C"
    );
    println!(
        "───────────────────────────────  ────────  ────────  ──────  ────────  ──────  ─────  ─────  ─────"
    );
    let mut wins_b = 0usize;
    let mut regs_b = 0usize;
    let mut wins_c = 0usize;
    let mut regs_c = 0usize;
    let mut sum_a: i64 = 0;
    let mut sum_b: i64 = 0;
    let mut sum_c: i64 = 0;
    for &(image, effort, distance, _expected_baseline) in CELLS {
        let path = corpus_path(image);
        let (rgb, w, h) = load_png(&path);
        // 1 warmup + 1 measured for time-box (this is an exploratory A/B, not
        // a precision bench).
        let _ = encode_default(&rgb, w, h, distance, effort);
        let (a, ma) = encode_default(&rgb, w, h, distance, effort);
        let _ = encode_no_dct64(&rgb, w, h, distance, effort);
        let (b, mb) = encode_no_dct64(&rgb, w, h, distance, effort);
        let _ = encode_no_dct32_dct64(&rgb, w, h, distance, effort);
        let (c, mc) = encode_no_dct32_dct64(&rgb, w, h, distance, effort);
        let an = a.len() as i64;
        let bn = b.len() as i64;
        let cn = c.len() as i64;
        let pct_b = (bn - an) as f64 / an as f64 * 100.0;
        let pct_c = (cn - an) as f64 / an as f64 * 100.0;
        if bn < an {
            wins_b += 1;
        }
        if bn > an {
            regs_b += 1;
        }
        if cn < an {
            wins_c += 1;
        }
        if cn > an {
            regs_c += 1;
        }
        sum_a += an;
        sum_b += bn;
        sum_c += cn;
        let cell_label = format!("{} e{} d={}", image, effort, distance);
        println!(
            "{:<31}  {:>8}  {:>8}  {:+5.2}%  {:>8}  {:+5.2}%  {:>4.0}   {:>4.0}   {:>4.0}",
            cell_label, an, bn, pct_b, cn, pct_c, ma, mb, mc
        );
    }
    println!("───────────────────────────────  ────────  ────────  ──────  ────────  ──────");
    let tot_pct_b = (sum_b - sum_a) as f64 / sum_a as f64 * 100.0;
    let tot_pct_c = (sum_c - sum_a) as f64 / sum_a as f64 * 100.0;
    println!(
        "TOTAL                            {:>8}  {:>8}  {:+5.2}%  {:>8}  {:+5.2}%",
        sum_a, sum_b, tot_pct_b, sum_c, tot_pct_c
    );
    println!();
    println!(
        "B (try_dct64=false): wins={}, regs={}, neutral={}",
        wins_b,
        regs_b,
        CELLS.len() - wins_b - regs_b
    );
    println!(
        "C (try_dct32+64=false): wins={}, regs={}, neutral={}",
        wins_c,
        regs_c,
        CELLS.len() - wins_c - regs_c
    );
    println!();
    println!("Decision rule:");
    println!("  - If B has > 5 wins with no regression > 1%: actionable DCT64-gate lever exists.");
    println!(
        "  - If B is uniformly neutral or regressing: post-W44-61 structural ceiling confirmed."
    );
    println!("  - If C beats B significantly: lever extends to DCT32 as well.");
}
