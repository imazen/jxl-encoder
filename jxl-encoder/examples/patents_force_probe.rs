// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GOAL_BEAT_CJXL patents-wedge probe (issue #74): cost-model vs
//! entropy attribution. Encodes the bilevel patent cell at e7 d0.5
//! with the default strategy search vs forced single strategies.
//! If forcing larger DCTs closes the 2x AC gap vs cjxl, the wedge is
//! cost-model-side (tiny-transform admission on bilevel); if not,
//! it's entropy-side (same mix coded worse).
//!
//! Run: cargo run --release -p jxl-encoder --example patents_force_probe -- <png>

use jxl_encoder::api::{LossyConfig, PixelLayout};

fn main() {
    let path = std::env::args().nth(1).expect("png path");
    let img = image::open(&path).expect("open");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.as_raw().clone();
    let cells: &[(&str, Option<u8>)] = &[
        ("default-search", None),
        ("force-DCT8", Some(0)),
        ("force-DCT16x16", Some(2)),
        ("force-DCT32x32", Some(3)),
        ("force-IDENTITY", Some(1)),
    ];
    for (label, fs) in cells {
        let data = LossyConfig::new(0.5)
            .with_effort(7)
            .with_force_strategy(*fs)
            .encode(&raw, w, h, PixelLayout::Rgb8)
            .expect("encode");
        println!("{:<16} {} bytes", label, data.len());
    }
}
