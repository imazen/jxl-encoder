// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Splines chunk-6 false-positive suppression bench (textured photos).
//
// W12-4 audit Top-5 #5: per-image suppression on textured photo content
// where the trial-encode gate might be fooled by Sobel edges that aren't
// true ridges. Reads PNG paths from CLI args and prints a TSV-style row
// per (image, distance, effort).
//
// Output columns:
//   label   distance   effort   w   h   off_bytes   on_bytes   delta_bytes   delta_pct

use std::path::Path;

use image::ImageReader;
use jxl_encoder::{LossyConfig, PixelLayout};

fn load_rgb8(path: &Path) -> (u32, u32, Vec<u8>) {
    let img = ImageReader::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
        .to_rgb8();
    (img.width(), img.height(), img.into_raw())
}

fn encode_bytes(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8, auto: bool) -> usize {
    let mut cfg = LossyConfig::new(distance).with_effort(effort);
    if auto {
        cfg = cfg.with_auto_splines(true);
    }
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(rgb)
        .unwrap_or_else(|e| panic!("encode: {e}"))
        .len()
}

fn main() {
    println!("label\tdistance\teffort\tw\th\toff_bytes\ton_bytes\tdelta_bytes\tdelta_pct");

    let efforts = [7u8, 8u8];
    let distances = [1.0f32, 2.0f32];

    for arg in std::env::args().skip(1) {
        let path = Path::new(&arg);
        let (w, h, rgb) = load_rgb8(path);
        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("anon")
            .to_string();
        for &distance in &distances {
            for &effort in &efforts {
                let off = encode_bytes(&rgb, w, h, distance, effort, false);
                let on = encode_bytes(&rgb, w, h, distance, effort, true);
                let delta = on as i64 - off as i64;
                let pct = 100.0 * delta as f64 / off as f64;
                println!(
                    "{label}\t{distance}\te{effort}\t{w}\t{h}\t{off}\t{on}\t{delta:+}\t{pct:+.3}"
                );
            }
        }
    }
}
