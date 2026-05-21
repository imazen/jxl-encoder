// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-164 Smart-Zenjxl chunk 1 A/B bench — paired comparison of the
//! auto-classifier dispatch ON vs OFF on screenshots and photos.
//!
//! Mode A: auto-classifier ON  (default Zenjxl behaviour after W44-164)
//! Mode B: auto-classifier OFF (`EncoderStrategy::Custom` with
//!         `content_class_auto_classify = false`, every other field at
//!         the Zenjxl default — equivalent to pre-W44-164 behaviour)
//!
//! Hypotheses (per the W44-163 audit + W36-3 / W41-2 prior art on
//! `patches` at e ∈ {5, 6}):
//! 1. Screenshots at e ∈ {5, 6} get patches auto-enabled under Mode A,
//!    producing measurably smaller bytes than Mode B.
//! 2. Screenshots at e >= 7 are byte-identical (patches already on by
//!    default at e7+).
//! 3. Photos are byte-identical (auto-classifier calls them Photo, which
//!    is a no-op in `adapt_to_image_content`).
//!
//! Cells (24 total, paired interleaved A/B):
//!   - 3 GB82-SC screenshots × e ∈ {5, 6, 7} × d = 1.0
//!   - 5 CID22 photos × e ∈ {5, 7} × d = 1.0  (photo byte-identity)
//!   - 1 LIBJXL canary: zenjxl vs libjxl byte-identical on each
//!     screenshot at e7 (auto off on libjxl)
//!
//! Output: TSV to stdout. Run via:
//!   cargo run --release \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_164_auto_classify_ab \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_164_auto_classify_ab_2026-05-21.tsv

#![allow(clippy::too_many_arguments)]

use image::GenericImageView;
use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
use jxl_encoder::{LossyConfig, PixelLayout};
use std::collections::BTreeMap;
use std::path::PathBuf;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82SC: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// (image, corpus, effort, distance).
const CELLS: &[(&str, &str, u8, f32)] = &[
    // GB82-SC screenshots — chunk 1 TARGETS
    ("codec_wiki.png", "gb82-sc", 5, 1.0),
    ("codec_wiki.png", "gb82-sc", 6, 1.0),
    ("codec_wiki.png", "gb82-sc", 7, 1.0),
    ("imac_g3.png", "gb82-sc", 5, 1.0),
    ("imac_g3.png", "gb82-sc", 6, 1.0),
    ("imac_g3.png", "gb82-sc", 7, 1.0),
    ("terminal.png", "gb82-sc", 5, 1.0),
    ("terminal.png", "gb82-sc", 6, 1.0),
    ("terminal.png", "gb82-sc", 7, 1.0),
    ("windows95.png", "gb82-sc", 5, 1.0),
    ("windows95.png", "gb82-sc", 6, 1.0),
    ("windows95.png", "gb82-sc", 7, 1.0),
    // CID22 photos — protection set: should be byte-identical
    ("1189261.png", "CID22", 5, 1.0),
    ("1189261.png", "CID22", 7, 1.0),
    ("1025469.png", "CID22", 5, 1.0),
    ("1025469.png", "CID22", 7, 1.0),
    ("1418519.png", "CID22", 5, 1.0),
    ("1418519.png", "CID22", 7, 1.0),
    ("1279330.png", "CID22", 5, 1.0),
    ("1279330.png", "CID22", 7, 1.0),
    ("1044329.png", "CID22", 5, 1.0),
    ("1044329.png", "CID22", 7, 1.0),
];

fn encode_with_strategy(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    strategy: EncoderStrategy,
) -> Result<Vec<u8>, String> {
    LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_strategy(strategy)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

fn main() {
    eprintln!("W44-164 Smart-Zenjxl chunk 1 — auto-classifier A/B");
    eprintln!("  Mode A = auto ON  (EncoderStrategy::Zenjxl default)");
    eprintln!("  Mode B = auto OFF (Custom with content_class_auto_classify = false)");
    eprintln!("  cells: {} (paired sequentially)", CELLS.len());

    println!("image\tcorpus\teffort\tdistance\tA_bytes\tB_bytes\tbytes_delta\tbytes_delta_pct");

    let mut images_cache: BTreeMap<String, (u32, u32, Vec<u8>)> = BTreeMap::new();
    let mut byte_identical = 0usize;
    let mut a_smaller = 0usize;
    let mut b_smaller = 0usize;
    let mut total_a = 0i64;
    let mut total_b = 0i64;

    for (i, &(image, corpus, effort, d)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} ({}) e{} d={}",
            i + 1,
            CELLS.len(),
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

        let (w, h, raw) = images_cache.entry(cache_key.clone()).or_insert_with(|| {
            let img = image::open(&path).expect("decode png");
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8().into_raw();
            (w, h, rgb)
        });

        // Mode A: Zenjxl (auto-classifier ON by default).
        let a_bytes = match encode_with_strategy(raw, *w, *h, effort, d, EncoderStrategy::Zenjxl) {
            Ok(b) => b.len(),
            Err(e) => {
                eprintln!("  A failed: {}", e);
                continue;
            }
        };

        // Mode B: Zenjxl but with content_class_auto_classify = false.
        let custom = EncoderImprovementsCustom {
            content_class_auto_classify: false,
            ..Default::default()
        };
        let b_bytes = match encode_with_strategy(
            raw,
            *w,
            *h,
            effort,
            d,
            EncoderStrategy::Custom(Box::new(custom)),
        ) {
            Ok(b) => b.len(),
            Err(e) => {
                eprintln!("  B failed: {}", e);
                continue;
            }
        };

        let delta = a_bytes as i64 - b_bytes as i64;
        let pct = (delta as f64 / b_bytes as f64) * 100.0;
        println!(
            "{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{:+.3}",
            image, corpus, effort, d, a_bytes, b_bytes, delta, pct
        );

        total_a += a_bytes as i64;
        total_b += b_bytes as i64;
        if a_bytes == b_bytes {
            byte_identical += 1;
        } else if a_bytes < b_bytes {
            a_smaller += 1;
        } else {
            b_smaller += 1;
        }
    }

    let total_delta = total_a - total_b;
    let total_pct = (total_delta as f64 / total_b as f64) * 100.0;
    eprintln!("");
    eprintln!("─── Summary ──────────────────────────────────────────────────");
    eprintln!("  cells: {}", CELLS.len());
    eprintln!(
        "  byte-identical:        {:3} ({:.1}%)",
        byte_identical,
        100.0 * byte_identical as f64 / CELLS.len() as f64
    );
    eprintln!(
        "  A < B (auto helps):    {:3} ({:.1}%)",
        a_smaller,
        100.0 * a_smaller as f64 / CELLS.len() as f64
    );
    eprintln!(
        "  B < A (auto hurts):    {:3} ({:.1}%)",
        b_smaller,
        100.0 * b_smaller as f64 / CELLS.len() as f64
    );
    eprintln!(
        "  total bytes A: {} B: {} delta: {:+} ({:+.3}%)",
        total_a, total_b, total_delta, total_pct
    );
}
