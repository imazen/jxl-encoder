//! A/B comparison for EX-J13 — Adaptive Gaborish kernel strength.
//!
//! Encodes 5 mixed photos at d=1.0 e7 with and without
//! `LossyConfig::with_adaptive_gaborish(true)` and prints a TSV row per
//! image plus a totals row. The "DELTA_B" column is `adapt - fixed`
//! (negative = adaptive smaller).
//!
//! The 5-photo mix is deliberately mixed-content (smooth gradients vs.
//! high-frequency textures) so the per-tile contrast→mul mapping has a
//! chance to differ from the fixed mul=1.0 baseline.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --example adaptive_gaborish_ab \
//!       > benchmarks/adaptive_gaborish_ab_2026-05-18.tsv

use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    // Mixed-content set: textured CLIC photos + CID22 portrait/landscape
    // (which often have smooth skin/sky regions).
    let images: [(&str, String); 5] = [
        (
            "cid22_1025469.png",
            format!(
                "{}/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
                base
            ),
        ),
        (
            "cid22_1189261.png",
            format!(
                "{}/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
                base
            ),
        ),
        (
            "cid22_1418519.png",
            format!(
                "{}/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
                base
            ),
        ),
        (
            "clic_07b9f93f.png",
            format!(
                "{}/work/codec-corpus/clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
                base
            ),
        ),
        (
            "clic_100a02c2.png",
            format!(
                "{}/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png",
                base
            ),
        ),
    ];

    let distance: f32 = 1.0;
    let effort: u8 = 7;

    println!("image\twidth\theight\tdistance\teffort\tfixed_B\tadapt_B\tdelta_B\tdelta_pct");

    let mut total_fixed: u64 = 0;
    let mut total_adapt: u64 = 0;

    for (name, path) in &images {
        let Ok(img) = image::open(path) else {
            eprintln!("WARN: failed to open {path}");
            continue;
        };
        let img = img.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb = img.as_raw();

        let fixed = LossyConfig::new(distance)
            .with_effort(effort)
            .with_adaptive_gaborish(false)
            .encode(rgb, w, h, PixelLayout::Rgb8)
            .expect("fixed encode");

        let adapt = LossyConfig::new(distance)
            .with_effort(effort)
            .with_adaptive_gaborish(true)
            .encode(rgb, w, h, PixelLayout::Rgb8)
            .expect("adaptive encode");

        let delta = adapt.len() as i64 - fixed.len() as i64;
        let delta_pct = 100.0 * (delta as f64) / (fixed.len() as f64);
        println!(
            "{name}\t{w}\t{h}\t{distance:.2}\t{effort}\t{}\t{}\t{delta}\t{delta_pct:.3}",
            fixed.len(),
            adapt.len()
        );

        total_fixed += fixed.len() as u64;
        total_adapt += adapt.len() as u64;
    }

    let total_delta = total_adapt as i64 - total_fixed as i64;
    let total_pct = 100.0 * (total_delta as f64) / (total_fixed as f64);
    println!(
        "TOTAL\t-\t-\t{distance:.2}\t{effort}\t{total_fixed}\t{total_adapt}\t{total_delta}\t{total_pct:.3}"
    );
}
