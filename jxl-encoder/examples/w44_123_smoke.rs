// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-123 smoke: verify `with_dct32_keep_hint(Some(true))` produces the
//! same bytes as the env-var `__JXL_W44_123_KEEP_DCT32=1` path that the
//! full A/B harness used (`benchmarks/w44_123_keep_dct32_ab_2026-05-20.tsv`).

use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use std::path::PathBuf;

fn main() {
    let path = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png");
    let img = image::open(&path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let rgb_u8: Vec<u8> = rgb.as_raw().clone();

    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);

    for effort in &[5u8, 6, 7] {
        let cfg_baseline = LossyConfig::new(3.0).with_effort(*effort);
        let b = cfg_baseline
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&lim)
            .encode(&rgb_u8)
            .unwrap();
        let cfg_keep = LossyConfig::new(3.0)
            .with_effort(*effort)
            .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct32_keep_hint: Some(true), ..Default::default() });
        let k = cfg_keep
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&lim)
            .encode(&rgb_u8)
            .unwrap();
        let cfg_explicit_off = LossyConfig::new(3.0)
            .with_effort(*effort)
            .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct32_keep_hint: Some(false), ..Default::default() });
        let o = cfg_explicit_off
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&lim)
            .encode(&rgb_u8)
            .unwrap();
        println!(
            "e{} baseline={} bytes, keep_dct32(Some(true))={} bytes (delta={:+.2}%), \
             keep_dct32(Some(false))={} bytes (delta={:+.2}% — must be 0)",
            effort,
            b.len(),
            k.len(),
            (k.len() as f64 - b.len() as f64) / b.len() as f64 * 100.0,
            o.len(),
            (o.len() as f64 - b.len() as f64) / b.len() as f64 * 100.0,
        );
    }
}
