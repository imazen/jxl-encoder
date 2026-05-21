// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-65 A/B: production default-on of the W44-63 dct_suppress_hint
//! auto-detection. Compares pre-W44-65 main behaviour (pinned via
//! `with_dct_suppress_hint(Some(false))`) against the new W44-65 default
//! (`with_dct_suppress_hint(None)` — auto-discriminator).
//!
//! Variants:
//!   A = pre-W44-65 main pinned (hint=Some(false), gate never fires) —
//!       reproduces the bitstream before this change.
//!   B = W44-65 default-on (hint=None) — auto-discriminator fires on
//!       screenshots, photos byte-identical.
//!
//! Expected from W44-63 results:
//!   - 3 codec_wiki cells (e7 d=4/5/6): -0.3 to -4 % bytes (OPEN → FIXED)
//!   - Already-FIXED screenshots (imac_g3, terminal at e7 d=4-6):
//!     -1 to -3 % bytes (deeper wins)
//!   - Photo cells: byte-identical (median < 95, gate doesn't fire)
//!   - windows95.png: byte-identical (median 81.49 < 95)
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!       --features 'parallel butteraugli-loop' \
//!       --example w44_65_default_on_ab \
//!       -- benchmarks/w44_65_default_on_ab_2026-05-19.tsv

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::{LossyConfig, PixelLayout};

// Mix of:
//  - 3 codec_wiki target cells (e7 d=4/5/6) expected to flip OPEN→FIXED
//  - 6 already-FIXED screenshot regression probes (imac_g3, terminal)
//  - 5 photo regression probes (photos should be byte-identical)
//  - windows95 (pixel-art regression probe — must stay byte-identical)
const CELLS: &[(&str, u8, f32, &str)] = &[
    // codec_wiki: target cells from W44-64 ledger refresh
    ("codec_wiki.png", 7, 4.0, "screen"),
    ("codec_wiki.png", 7, 5.0, "screen"),
    ("codec_wiki.png", 7, 6.0, "screen"),
    // Already-FIXED screenshots — verify deeper wins (regression check)
    ("imac_g3.png", 7, 4.0, "screen"),
    ("imac_g3.png", 7, 5.0, "screen"),
    ("imac_g3.png", 7, 6.0, "screen"),
    ("terminal.png", 7, 4.0, "screen"),
    ("terminal.png", 7, 5.0, "screen"),
    ("terminal.png", 7, 6.0, "screen"),
    // windows95: pixel-art regression check (must NOT regress)
    ("windows95.png", 7, 1.0, "pixel_art"),
    ("windows95.png", 7, 2.0, "pixel_art"),
    ("windows95.png", 7, 4.0, "pixel_art"),
    // Photo regression probes (must stay byte-identical at all distances)
    ("1189261.png", 5, 4.0, "photo"),
    ("1189261.png", 7, 3.0, "photo"),
    ("1418519.png", 7, 1.0, "photo"),
    ("1418519.png", 7, 3.0, "photo"),
    ("1418519.png", 7, 5.0, "photo"),
    ("1420710.png", 7, 5.0, "photo"),
    ("1420710.png", 7, 6.0, "photo"),
    ("1531677.png", 7, 5.0, "photo"),
    ("1531677.png", 7, 6.0, "photo"),
    ("1025469.png", 7, 3.0, "photo"),
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

fn encode(
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
        .with_strategy_overrides(jxl_encoder::api::StrategyOverrides {
            dct_suppress_hint: hint,
            ..Default::default()
        });
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn main() {
    let tsv_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/w44_65_default_on_ab_2026-05-19.tsv"));

    println!("W44-65 A/B: default-on dct_suppress_hint auto-detection");
    println!();
    println!(
        "cell                              cls          bytes_A   bytes_B   ΔB_pct   ms_A    ms_B"
    );
    println!(
        "─────────────────────────────  ─────────────  ────────  ────────  ───────  ──────  ──────"
    );

    if let Some(parent) = tsv_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut tsv = std::fs::File::create(&tsv_path).expect("create TSV");
    writeln!(
        tsv,
        "image\teffort\tdistance\tclass\tbytes_a\tbytes_b\tpct_b\tms_a\tms_b"
    )
    .ok();

    let mut wins_b = 0usize;
    let mut regs_b = 0usize;
    let mut sum_a: i64 = 0;
    let mut sum_b: i64 = 0;
    let mut photo_b_regression_any = false;
    let mut pixel_art_b_regression_any = false;
    let mut codec_wiki_e7_pcts: Vec<f64> = Vec::new();
    for &(image, effort, distance, class) in CELLS {
        let path = corpus_path(image);
        let (rgb, w, h) = load_png(&path);
        // 1 warmup + 1 measured (exploratory A/B, not precision bench).
        // Bytes should be deterministic; ms is exploratory.
        let _ = encode(&rgb, w, h, distance, effort, Some(false));
        let (a_bytes, ma) = encode(&rgb, w, h, distance, effort, Some(false));
        let _ = encode(&rgb, w, h, distance, effort, None);
        let (b_bytes, mb) = encode(&rgb, w, h, distance, effort, None);
        // Cross-verify: bytes are deterministic, re-encode A and assert
        // identical so the surprising windows95 e7 d=2 +1.13% isn't a
        // measurement artifact.
        let (a_bytes2, _) = encode(&rgb, w, h, distance, effort, Some(false));
        assert_eq!(
            a_bytes.len(),
            a_bytes2.len(),
            "A non-deterministic on {} e{} d={}",
            image,
            effort,
            distance
        );
        let an = a_bytes.len() as i64;
        let bn = b_bytes.len() as i64;
        let pct_b = (bn - an) as f64 / an as f64 * 100.0;

        sum_a += an;
        sum_b += bn;
        if pct_b < -0.05 {
            wins_b += 1;
        } else if pct_b > 0.05 {
            regs_b += 1;
            if class == "photo" {
                photo_b_regression_any = true;
            }
            if class == "pixel_art" {
                pixel_art_b_regression_any = true;
            }
        }
        if image == "codec_wiki.png" && effort == 7 {
            codec_wiki_e7_pcts.push(pct_b);
        }

        let cell = format!("{} e{} d={}", image, effort, distance);
        println!(
            "{:>30}  {:>13}  {:>8}  {:>8}  {:+7.4}  {:>6.1}  {:>6.1}",
            cell, class, an, bn, pct_b, ma, mb
        );
        writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.2}",
            image, effort, distance, class, an, bn, pct_b, ma, mb
        )
        .ok();
    }

    let total_pct = (sum_b - sum_a) as f64 / sum_a as f64 * 100.0;
    println!();
    println!("=== Summary ===");
    println!("Total bytes A (pre-W44-65 pinned): {}", sum_a);
    println!("Total bytes B (W44-65 default-on):  {}", sum_b);
    println!("Total Δ: {:+.4} %", total_pct);
    println!("Wins (Δ < -0.05 %): {}", wins_b);
    println!("Regressions (Δ > +0.05 %): {}", regs_b);
    println!(
        "Photo regression any cell? {}",
        if photo_b_regression_any {
            "YES — FAIL"
        } else {
            "no — PASS"
        }
    );
    println!(
        "Pixel-art (windows95) regression any cell? {}",
        if pixel_art_b_regression_any {
            "YES — FAIL"
        } else {
            "no — PASS"
        }
    );

    if !codec_wiki_e7_pcts.is_empty() {
        let flipped: usize = codec_wiki_e7_pcts.iter().filter(|p| **p < -0.05).count();
        println!(
            "codec_wiki e7 cells improving: {}/{} ({:.4}% to {:.4}%)",
            flipped,
            codec_wiki_e7_pcts.len(),
            codec_wiki_e7_pcts
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
            codec_wiki_e7_pcts
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
        );
    }

    println!();
    println!("TSV written: {}", tsv_path.display());
}
