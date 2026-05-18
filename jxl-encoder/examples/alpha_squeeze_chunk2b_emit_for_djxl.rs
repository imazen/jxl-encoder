// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-2.b helper: writes chunk-2.b alpha-squeeze .jxl files for
//! each W13-4 audit image × 4 distances under `/tmp/chunk2b_*.jxl`
//! so the caller can `djxl` them for reference-decoder validation.
//!
//! Run:
//!   cargo run --release -p jxl-encoder --example alpha_squeeze_chunk2b_emit_for_djxl

use image::ImageReader;
use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let images: &[(&str, &str)] = &[
        (
            "gradients_semitrans_ui",
            "/home/lilith/work/codec-corpus/imageflow/test_inputs/gradients.png",
        ),
        (
            "red_night_opaque",
            "/home/lilith/work/codec-corpus/imageflow/test_inputs/red-night.png",
        ),
        (
            "alpha_nonpremul_photo_mask",
            "/home/lilith/work/codec-corpus/jxl/reference/conformance/alpha_nonpremultiplied.png",
        ),
    ];
    for (label, path) in images {
        let img = ImageReader::open(path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        let (w, h) = img.dimensions();
        let rgba = img.into_raw();
        for &ad in &[0.5f32, 1.0, 2.0, 5.0] {
            let bytes = LossyConfig::new(1.0)
                .with_alpha_distance(Some(ad))
                .with_alpha_squeeze(true)
                .encode(&rgba, w, h, PixelLayout::Rgba8)
                .expect("encode");
            let out_path = format!("/tmp/chunk2b_{label}_d{ad}.jxl");
            std::fs::write(&out_path, &bytes).unwrap();
            println!("{out_path}\t{}\tbytes", bytes.len());
        }
    }
}
