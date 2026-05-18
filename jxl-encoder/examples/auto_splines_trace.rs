// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Trace one image at one effort with auto_splines on. Used in
// chunk-4 investigation to characterise per-image gate behavior.
// Build with `--features debug-tokens` to see find_splines + gate output.
//
// Usage: cargo run --release --example auto_splines_trace --features 'std parallel butteraugli-loop debug-tokens' -- <path.png> <effort>

use image::ImageReader;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(args.get(1).expect("usage: trace <path> <effort>"));
    let effort: u8 = args.get(2).expect("effort").parse().expect("effort u8");
    let img = ImageReader::open(path).unwrap().decode().unwrap().to_rgb8();
    let (w, h, rgb) = (img.width(), img.height(), img.into_raw());
    eprintln!(
        "=== TRACE: {} effort={} ({}x{}) ===",
        path.display(),
        effort,
        w,
        h
    );
    let cfg = LossyConfig::new(1.0)
        .with_effort(effort)
        .with_auto_splines(true);
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&rgb)
        .unwrap();
    eprintln!("=== output: {} bytes ===", bytes.len());
}
