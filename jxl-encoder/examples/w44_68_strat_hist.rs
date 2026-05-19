// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-68 codec_wiki strategy histogram probe at d=3..6.

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
    eprintln!("\n==== {} d={:.2} ({}x{}) ====", name, distance, w, h);
    let cfg = LossyConfig::new(distance).with_effort(7).with_threads(1);
    let _ = cfg.encode(&raw, w, h, PixelLayout::Rgb8).expect("encode");
}

fn main() {
    eprintln!("W44-68 strategy histogram probe");
    for d in &[3.0f32, 4.0, 5.0, 6.0] {
        probe("codec_wiki.png", *d);
    }
}
