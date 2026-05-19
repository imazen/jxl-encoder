// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-68 codec_wiki d=4 mask1x1 probe — measures mask1x1_median at the
//! distances surrounding the wedge cell so we can compare with the
//! W44-65 threshold (>= 99.5).
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!       --features 'debug-w44-65 parallel' \
//!       --example w44_68_d4_probe

use std::path::PathBuf;

use jxl_encoder::api::{LossyConfig, PixelLayout};

const CORPUS_BASE: &str = "/home/lilith/work/codec-corpus";

fn gb82(name: &str) -> PathBuf {
    PathBuf::from(CORPUS_BASE).join("gb82-sc").join(name)
}

fn probe(name: &str, distance: f32) {
    let path = gb82(name);
    let img = image::open(&path).unwrap_or_else(|e| panic!("open {:?}: {}", path, e));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.as_raw().clone();
    eprintln!("==== {} d={:.2} ({}x{}) ====", name, distance, w, h);
    // hint=None so default-on W44-65 dispatch is exercised.
    let cfg = LossyConfig::new(distance).with_effort(7).with_threads(1);
    let _ = cfg.encode(&raw, w, h, PixelLayout::Rgb8).expect("encode");
}

fn main() {
    eprintln!("W44-68 codec_wiki d=4 probe");
    println!("Run with --features 'debug-w44-65 parallel'");
    println!();
    // Probe codec_wiki at the wedge distance and adjacent ones.
    for d in &[3.0f32, 4.0, 5.0, 6.0] {
        probe("codec_wiki.png", *d);
    }
    // windows95 regression-watch image
    for d in &[2.0f32, 3.0, 4.0] {
        probe("windows95.png", *d);
    }
}
