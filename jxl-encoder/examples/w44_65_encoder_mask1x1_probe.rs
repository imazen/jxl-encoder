// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-65 mask1x1 probe — variant that measures via the actual encoder
//! pipeline (so values match `vardct::encoder::median_mask1x1` produced
//! during a real encode). The standalone `srgb_image_to_xyb` probe
//! diverges from the encoder by ~17 (windows95: 81.49 vs encoder
//! 99.06) because the encoder uses a precomputed sRGB→linear LUT and
//! a SIMD `linear_rgb_to_xyb_batch` path that produces slightly
//! different float values than the scalar `srgb_to_linear_value` +
//! `linear_rgb_to_xyb` path. For dispatch threshold tuning we need
//! the encoder's view, not the standalone view.
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!       --features '__pre_quantized debug-w44-65 parallel' \
//!       --example w44_65_encoder_mask1x1_probe
//!
//! Output is printed via the `debug-w44-65` feature inside
//! `vardct::encoder` — we just trigger one encode per image to dump
//! the mask1x1_median for that image.

use std::path::PathBuf;

use jxl_encoder::api::{LossyConfig, PixelLayout};

const CORPUS_BASE: &str = "/home/lilith/work/codec-corpus";

fn corpus_path(name: &str) -> Option<PathBuf> {
    let cid22 = PathBuf::from(CORPUS_BASE)
        .join("CID22/CID22-512/validation")
        .join(name);
    if cid22.exists() {
        return Some(cid22);
    }
    let gb82 = PathBuf::from(CORPUS_BASE).join("gb82-sc").join(name);
    if gb82.exists() {
        return Some(gb82);
    }
    None
}

fn probe(name: &str) {
    let Some(path) = corpus_path(name) else {
        println!("==== {}: NOT FOUND ====", name);
        return;
    };
    let img = image::open(&path).unwrap_or_else(|e| panic!("open {:?}: {}", path, e));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.as_raw().clone();
    eprintln!("==== {} ({}x{}) ====", name, w, h);
    // Single encode at e7 d=1 — debug-w44-65 will print the mask1x1
    // median for this image. We pin hint=Some(false) so the encode
    // path mirrors pre-W44-65 main exactly.
    let cfg = LossyConfig::new(1.0)
        .with_effort(7)
        .with_threads(1)
        .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct_suppress_hint: Some(false), ..Default::default() });
    let _ = cfg.encode(&raw, w, h, PixelLayout::Rgb8).expect("encode");
}

fn main() {
    eprintln!("W44-65 encoder mask1x1 probe — fires debug-w44-65 to stderr");
    println!("Run with: --features debug-w44-65 to see mask1x1_median per image");
    println!();
    for img in &[
        "codec_wiki.png",
        "imac_g3.png",
        "imac_dark.png",
        "terminal.png",
        "windows.png",
        "windows95.png",
        "imessage.png",
        "graph.png",
        "1189261.png",
        "1418519.png",
        "1420710.png",
        "1531677.png",
        "1025469.png",
    ] {
        probe(img);
    }
    // Also scan all CID22 validation photos
    println!("\nAll CID22 validation photos:");
    let validation =
        std::path::Path::new("/home/lilith/work/codec-corpus/CID22/CID22-512/validation");
    if let Ok(entries) = std::fs::read_dir(validation) {
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        for p in paths {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            probe(&name);
        }
    }
}
