// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-68 codec_wiki d=4 A/B: try additional suppressions on top of the
//! W44-65 default `try_dct64=false` to see if any close the +3.55 % cell.
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!       --features 'parallel __expert' \
//!       --example w44_68_ab
//!
//! Variants tested at d=4 on codec_wiki + regression check on
//! windows95 (the cell W44-65 protects via mask1x1 < 99.5 threshold) +
//! 2 photo cells.

use std::path::PathBuf;

use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_encoder::effort::LossyInternalParams;

const CORPUS_BASE: &str = "/home/lilith/work/codec-corpus";

fn corpus(subdir: &str, name: &str) -> PathBuf {
    PathBuf::from(CORPUS_BASE).join(subdir).join(name)
}

#[derive(Debug, Clone, Copy)]
struct Variant {
    name: &'static str,
    try_dct4x8_afv: Option<bool>,
    try_dct32: Option<bool>,
    try_dct64: Option<bool>,
}

const VARIANTS: &[Variant] = &[
    Variant {
        name: "baseline_W44_65_default",
        try_dct4x8_afv: None,
        try_dct32: None,
        try_dct64: None,
    },
    Variant {
        name: "suppress_dct32_and_dct64",
        try_dct4x8_afv: None,
        try_dct32: Some(false),
        try_dct64: Some(false),
    },
];

fn encode(path: &PathBuf, distance: f32, v: &Variant) -> usize {
    let img = image::open(path).unwrap_or_else(|e| panic!("open {:?}: {}", path, e));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.as_raw().clone();
    let mut params = LossyInternalParams::default();
    params.try_dct4x8_afv = v.try_dct4x8_afv;
    params.try_dct32 = v.try_dct32;
    params.try_dct64 = v.try_dct64;
    let mut cfg = LossyConfig::new(distance).with_effort(7).with_threads(1);
    cfg = cfg.with_internal_params(params);
    cfg.encode(&raw, w, h, PixelLayout::Rgb8)
        .expect("encode")
        .len()
}

fn report(image: &PathBuf, distance: f32) {
    eprintln!(
        "\n=== {} d={} ===",
        image.file_name().unwrap().to_string_lossy(),
        distance
    );
    let mut baseline = 0usize;
    for v in VARIANTS {
        let bytes = encode(image, distance, v);
        if baseline == 0 {
            baseline = bytes;
        }
        let delta_pct = 100.0 * (bytes as f64 - baseline as f64) / baseline as f64;
        eprintln!(
            "  {:35}  bytes={:7}  delta={:+.3}%",
            v.name, bytes, delta_pct
        );
    }
}

fn main() {
    eprintln!("W44-68 codec_wiki d=4 A/B (suppression strategy bisection)");

    // codec_wiki across all the cells (we want to verify no regression on the
    // ones already FIXED while the wedge cell d=4 flips).
    for d in &[0.5f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        report(&corpus("gb82-sc", "codec_wiki.png"), *d);
    }

    // Other screenshots that might also be affected (mask1x1 ≥ 99.5 set).
    for img in &[
        "terminal.png",
        "imac_g3.png",
        "imac_dark.png",
        "windows.png",
    ] {
        for d in &[1.0f32, 4.0] {
            report(&corpus("gb82-sc", img), *d);
        }
    }

    // windows95 across distances — DOES sit at 99.06 < 99.5 so it should NOT
    // be touched by the dispatcher, but we test the raw cost via expert
    // override to bound the worst-case regression IF the threshold gets
    // lowered.
    for d in &[2.0f32, 3.0, 4.0] {
        report(&corpus("gb82-sc", "windows95.png"), *d);
    }

    // Photo regression check (smooth content, NOT in dispatcher's class).
    for d in &[1.0f32, 4.0] {
        report(&corpus("CID22/CID22-512/validation", "1418519.png"), *d);
    }
}
