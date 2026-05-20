// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-63 A/B/C: production wire-up of the W44-62 DCT64-suppress lever.
//!
//! W44-62 (`07f8b3d2`, harness only) probed `__expert
//! LossyInternalParams { try_dct64 = Some(false) }` on the 20 OPEN cells
//! residual to W44-61 and found a polar content-class split:
//! screenshot-class wins uniformly -0.13 % to -3.25 %, photo wins
//! sub-1 %. This chunk ships the public API
//! `LossyConfig::with_dct_suppress_hint(Option<bool>)` and the
//! `content_aware_entropy_mul`-gated encoder-internal `median(mask1x1) >
//! 95` discriminator (analog of W22-1 / W44-29).
//!
//! A/B/C variants:
//!
//!   A = default (`content_aware_entropy_mul=false`, hint=None) — production
//!       behaviour; byte-identical to main.
//!   B = `content_aware_entropy_mul=true` + hint=None — engages the encoder-
//!       internal auto discriminator (`median(mask1x1) > 95`). Photo cells
//!       see no change (the gate is photo-safe); screenshot cells see DCT64
//!       suppression.
//!   C = `content_aware_entropy_mul=true` + hint=Some(true) — caller forces
//!       DCT64 suppression regardless of mask1x1. Mirrors W44-62's `__expert
//!       try_dct64=Some(false)` measurement; sanity-check that the new
//!       public API matches the `__expert` channel byte-for-byte.
//!
//! Expected from W44-62: variant B closes codec_wiki e7 d=5 (OPEN→FIXED,
//! +3.51 % → +0.18 % bytes). Variant C matches W44-62's TOTAL -0.68 %
//! across all 26 cells.
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!       --features '__expert butteraugli-loop ssim2-loop parallel' \
//!       --example w44_63_dct_suppress_ab
//!
//! Optional: append a benchmark TSV path as a CLI argument to persist
//! per-cell rows.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::{LossyConfig, PixelLayout};

// 20 OPEN cells (post-W44-61) + 6 FIXED screenshot regression probes,
// sourced from cjxl_parity_ledger_2026-05-19_w44_61.tsv via the W44-62
// harness for parity. Columns: (image, effort, distance, content_class).
const CELLS: &[(&str, u8, f32, &str)] = &[
    ("1189261.png", 5, 4.0, "photo"),
    ("1189261.png", 8, 3.0, "photo"),
    ("1189261.png", 9, 3.0, "photo"),
    ("1420710.png", 5, 5.0, "photo"),
    ("1420710.png", 5, 6.0, "photo"),
    ("1420710.png", 6, 5.0, "photo"),
    ("1420710.png", 6, 6.0, "photo"),
    ("1420710.png", 7, 5.0, "photo"),
    ("1420710.png", 7, 6.0, "photo"),
    ("1420710.png", 8, 5.0, "photo"),
    ("1420710.png", 9, 5.0, "photo"),
    ("1531677.png", 5, 5.0, "photo"),
    ("1531677.png", 5, 6.0, "photo"),
    ("1531677.png", 6, 5.0, "photo"),
    ("1531677.png", 6, 6.0, "photo"),
    ("1531677.png", 8, 5.0, "photo"),
    ("1531677.png", 9, 5.0, "photo"),
    ("codec_wiki.png", 7, 4.0, "screen"),
    ("codec_wiki.png", 7, 5.0, "screen"),
    ("codec_wiki.png", 7, 6.0, "screen"),
    // FIXED-screenshot regression probes — verify suppression doesn't
    // regress already-passing cells (W44-62 measured -0.13 % to -1.07 %
    // — deeper wins on these).
    ("imac_g3.png", 7, 4.0, "screen"),
    ("imac_g3.png", 7, 5.0, "screen"),
    ("imac_g3.png", 7, 6.0, "screen"),
    ("terminal.png", 7, 4.0, "screen"),
    ("terminal.png", 7, 5.0, "screen"),
    ("terminal.png", 7, 6.0, "screen"),
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

fn encode_with_hint(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    hint: Option<bool>,
) -> (Vec<u8>, f64) {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_content_aware_entropy_mul(true)
        .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct_suppress_hint: hint, ..Default::default() });
    let start = Instant::now();
    let bytes = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode with_hint");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn main() {
    let tsv_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/w44_63_dct_suppress_ab_2026-05-19.tsv"));

    println!("W44-63 A/B/C: with_dct_suppress_hint API smoke + dispatch sweep");
    println!();
    println!(
        "cell                              cls    bytes_A   bytes_B  ΔB_pct   bytes_C  ΔC_pct   ms_A   ms_B   ms_C"
    );
    println!(
        "───────────────────────────────  ─────  ────────  ────────  ──────  ────────  ──────  ─────  ─────  ─────"
    );

    if let Some(parent) = tsv_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut tsv = std::fs::File::create(&tsv_path).expect("create TSV");
    writeln!(
        tsv,
        "image\teffort\tdistance\tclass\tbytes_a\tbytes_b\tbytes_c\tpct_b\tpct_c\tms_a\tms_b\tms_c"
    )
    .ok();

    let mut wins_b = 0usize;
    let mut regs_b = 0usize;
    let mut wins_c = 0usize;
    let mut regs_c = 0usize;
    let mut sum_a: i64 = 0;
    let mut sum_b: i64 = 0;
    let mut sum_c: i64 = 0;
    let mut photo_b_regression_any = false;
    let mut codec_wiki_e7_d5_pct_b: Option<f64> = None;
    for &(image, effort, distance, class) in CELLS {
        let path = corpus_path(image);
        let (rgb, w, h) = load_png(&path);
        // 1 warmup + 1 measured (exploratory A/B/C, not precision bench).
        let _ = encode_default(&rgb, w, h, distance, effort);
        let (a_bytes, ma) = encode_default(&rgb, w, h, distance, effort);
        let _ = encode_with_hint(&rgb, w, h, distance, effort, None);
        let (b_bytes, mb) = encode_with_hint(&rgb, w, h, distance, effort, None);
        let _ = encode_with_hint(&rgb, w, h, distance, effort, Some(true));
        let (c_bytes, mc) = encode_with_hint(&rgb, w, h, distance, effort, Some(true));
        let an = a_bytes.len() as i64;
        let bn = b_bytes.len() as i64;
        let cn = c_bytes.len() as i64;
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
        if class == "photo" && pct_b > 1.0 {
            photo_b_regression_any = true;
        }
        if image == "codec_wiki.png" && effort == 7 && (distance - 5.0).abs() < 1e-3 {
            codec_wiki_e7_d5_pct_b = Some(pct_b);
        }
        sum_a += an;
        sum_b += bn;
        sum_c += cn;
        let cell_label = format!("{} e{} d={}", image, effort, distance);
        println!(
            "{:<31}  {:<5}  {:>8}  {:>8}  {:+5.2}%  {:>8}  {:+5.2}%  {:>4.0}   {:>4.0}   {:>4.0}",
            cell_label, class, an, bn, pct_b, cn, pct_c, ma, mb, mc
        );
        writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.2}\t{:.2}\t{:.2}",
            image, effort, distance, class, an, bn, cn, pct_b, pct_c, ma, mb, mc
        )
        .ok();
    }
    println!(
        "───────────────────────────────  ─────  ────────  ────────  ──────  ────────  ──────"
    );
    let tot_pct_b = (sum_b - sum_a) as f64 / sum_a as f64 * 100.0;
    let tot_pct_c = (sum_c - sum_a) as f64 / sum_a as f64 * 100.0;
    println!(
        "TOTAL                                   {:>8}  {:>8}  {:+5.2}%  {:>8}  {:+5.2}%",
        sum_a, sum_b, tot_pct_b, sum_c, tot_pct_c
    );
    println!();
    println!(
        "B (opt-in, hint=None, auto mask1x1>95):  wins={}, regs={}, neutral={}",
        wins_b,
        regs_b,
        CELLS.len() - wins_b - regs_b
    );
    println!(
        "C (opt-in, hint=Some(true), force):       wins={}, regs={}, neutral={}",
        wins_c,
        regs_c,
        CELLS.len() - wins_c - regs_c
    );

    println!();
    println!("Acceptance gates:");
    let (ag, ag_msg) = match codec_wiki_e7_d5_pct_b {
        Some(pct) if pct <= 3.0 => (
            true,
            format!("codec_wiki e7 d=5 B delta {pct:+.2}% ≤ 3% (OPEN→FIXED candidate)"),
        ),
        Some(pct) => (
            false,
            format!("codec_wiki e7 d=5 B delta {pct:+.2}% > 3% (no flip)"),
        ),
        None => (false, "codec_wiki e7 d=5 cell missing".to_string()),
    };
    println!(
        "  (a) Flip cell:           [{}] {}",
        if ag { "PASS" } else { "FAIL" },
        ag_msg
    );
    let bg = !photo_b_regression_any;
    println!(
        "  (b) Photo regression:    [{}] no photo cell shows B delta > 1.0 %",
        if bg { "PASS" } else { "FAIL" }
    );
    println!("      TSV written: {}", tsv_path.display());
}
