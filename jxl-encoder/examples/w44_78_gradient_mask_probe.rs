// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-78 probe: print mask1x1_median for the 32x32 gradient used in
//! the `lossy_distance_3` hash-lock test, and a couple of other small
//! synthetic inputs. Used to verify hash-lock impact of widening the
//! W44-29 distance gate from 4.0 → 3.0.
//!
//! Build:
//!   cargo run -p jxl-encoder --release \
//!     --features '__pre_quantized debug-w44-65 parallel' \
//!     --example w44_78_gradient_mask_probe

use jxl_encoder::api::{LossyConfig, PixelLayout};

fn gradient_rgb_32x32() -> Vec<u8> {
    let (w, h) = (32, 32);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 128;
        }
    }
    out
}

fn gradient_rgb_13x17() -> Vec<u8> {
    let (w, h) = (13, 17);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = ((x * 255) / (w - 1).max(1)) as u8;
            out[i + 1] = ((y * 255) / (h - 1).max(1)) as u8;
            out[i + 2] = 128;
        }
    }
    out
}

fn rgb_48x48_noise() -> Vec<u8> {
    let (w, h) = (48, 48);
    let mut out = vec![0u8; w * h * 3];
    let mut seed: u32 = 0x12345678;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            out[i] = (seed & 0xff) as u8;
            out[i + 1] = ((seed >> 8) & 0xff) as u8;
            out[i + 2] = ((seed >> 16) & 0xff) as u8;
        }
    }
    out
}

fn probe(name: &str, rgb: &[u8], w: u32, h: u32, distance: f32) {
    eprintln!("==== {} ({}x{}) d={} ====", name, w, h, distance);
    let cfg = LossyConfig::new(distance)
        .with_effort(7)
        .with_threads(1)
        .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct_suppress_hint: Some(false), ..Default::default() })
        .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { high_d_photo_hint: Some(false), ..Default::default() });
    let _ = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
}

fn main() {
    eprintln!("W44-78 gradient mask probe — debug-w44-65 prints mask1x1_median");
    println!("Run with: --features debug-w44-65");
    println!();

    let g32 = gradient_rgb_32x32();
    let g13 = gradient_rgb_13x17();
    let noise48 = rgb_48x48_noise();

    // Test at d=1.0 (default), d=3.0 (lossy_distance_3 hash-lock), d=4.0
    for d in [1.0, 3.0, 4.0] {
        probe("gradient_32x32", &g32, 32, 32, d);
        probe("gradient_13x17", &g13, 13, 17, d);
        probe("noise_48x48", &noise48, 48, 48, d);
    }
}
