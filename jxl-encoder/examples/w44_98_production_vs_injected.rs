// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-98 production-vs-injected verification.
//!
//! After wiring the W44-98 dispatch in `vardct/encoder.rs`, running
//! the encoder on 1420710 cells at d>=4.5 with no overrides should
//! auto-fire variant Z' (high_colour=true), and produce bytes
//! IDENTICAL to the W44-98 ZA bisect bench (`dct16x32=1.30`) variant.
//!
//! On 1531677 cells, production should auto-fire variant Z (the
//! default suppressed_z table, dct16x32=1.208) — IDENTICAL to the
//! "default" column of the W44-98 bisect bench.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_98_production_vs_injected

#![allow(clippy::too_many_arguments)]

use image::GenericImageView;
use jxl_encoder::effort::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::PathBuf;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

fn encode_default(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
    LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

fn encode_injected_z_high_colour(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
) -> Result<Vec<u8>, String> {
    let mut cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides {
        high_d_photo_hint: Some(false),
        ..Default::default()
    });
    let mut internal = LossyInternalParams::default();
    internal.entropy_mul_table =
        Some(EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour());
    cfg = cfg.with_internal_params(internal);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

fn encode_injected_z(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
    let mut cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides {
        high_d_photo_hint: Some(false),
        ..Default::default()
    });
    let mut internal = LossyInternalParams::default();
    internal.entropy_mul_table = Some(EntropyMulTable::high_d_photo_smooth_suppressed_z());
    cfg = cfg.with_internal_params(internal);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

fn main() {
    println!(
        "class\timage\teffort\tdistance\tdefault_bytes\tinjected_bytes\tmatches\texpected_variant"
    );

    let cases = [
        // 1420710 — m3=32.93 ≥ 25.0 → should auto-fire variant Z'
        ("1420710.png", 5, 5.0, "Z_high_colour"),
        ("1420710.png", 5, 6.0, "Z_high_colour"),
        ("1420710.png", 7, 5.0, "Z_high_colour"),
        ("1420710.png", 6, 5.0, "Z_high_colour"),
        ("1420710.png", 8, 5.0, "Z_high_colour"),
        // 1531677 — m3=12.30 < 25.0 → should auto-fire variant Z (not Z')
        ("1531677.png", 5, 5.0, "Z"),
        ("1531677.png", 6, 5.0, "Z"),
        ("1531677.png", 8, 5.0, "Z"),
        // d<4.5 — no variant Z (uses W44-29 default suppressed)
        ("1420710.png", 5, 4.0, "default-suppressed"),
        ("1531677.png", 5, 4.0, "default-suppressed"),
    ];

    for &(image, effort, dist, expected) in &cases {
        let path = PathBuf::from(CID22).join(image);
        let img = image::open(&path).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();

        let default = encode_default(&rgb, w, h, effort, dist).unwrap();

        let injected = match expected {
            "Z_high_colour" => encode_injected_z_high_colour(&rgb, w, h, effort, dist).unwrap(),
            "Z" => encode_injected_z(&rgb, w, h, effort, dist).unwrap(),
            _ => Vec::new(), // skip injected comparison for d<4.5
        };

        let matches = if expected == "default-suppressed" {
            "skip".to_string()
        } else if default.len() == injected.len() && default == injected {
            "YES".to_string()
        } else {
            format!(
                "NO (default={}, injected={})",
                default.len(),
                injected.len()
            )
        };

        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            "case",
            image,
            effort,
            dist,
            default.len(),
            injected.len(),
            matches,
            expected
        );
    }
}
