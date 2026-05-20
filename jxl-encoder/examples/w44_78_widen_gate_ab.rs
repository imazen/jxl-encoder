// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-78 widen-W44-29-gate A/B sweep.
//!
//! Tests three candidate widenings of the W44-29 auto-fire gate
//! (default: `distance >= 4.0 AND mask1x1_median < 50.0`):
//!
//! - Variant A: `distance >= 3.0 AND mask1x1_median < 50.0`
//!   (lower distance gate; catches 1420710 d=3 + 1531677 d=3)
//! - Variant B: `distance >= 4.0 AND mask1x1_median < 80.0`
//!   (lift mask1x1 gate; catches 1189261 d=4-5 since its mask1x1=69)
//! - Variant C: `distance >= 3.0 AND mask1x1_median < 80.0`
//!   (both lifts; catches 1189261 d=3-5 + 1420710/1531677 d=3)
//! - Variant D: `distance >= 2.5 AND mask1x1_median < 95.0`
//!   (most aggressive — catches even more)
//!
//! Method: per cell, encode three times:
//!   * baseline = default LossyConfig (current gate)
//!   * variant  = LossyConfig with `with_high_d_photo_hint(Some(true))`
//!                IF the cell's (distance, mask1x1_median) would
//!                trigger under that variant's gate
//!   * cjxl ref via the cjxl binary
//!
//! Measures bytes-delta per cell and identifies regressions
//! (FIXED→OPEN flips) and savings (OPEN→FIXED flips).
//!
//! Build:
//!   cargo run -p jxl-encoder --release \
//!     --features '__pre_quantized debug-w44-65 parallel' \
//!     --example w44_78_widen_gate_ab

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82: &str = "/home/lilith/work/codec-corpus/gb82-sc";

// Distances to test. Includes d=2.5 to verify variant D won't hurt
// hash-locks at d<3, plus all FD-residual candidate d.
const DISTANCES: &[f32] = &[2.5, 3.0, 4.0, 5.0, 6.0];

// (distance_gate, mask1x1_gate, label)
const VARIANTS: &[(f32, f32, &str)] = &[
    (4.0, 50.0, "default (no widening)"),
    (3.0, 50.0, "A: dist>=3"),
    (4.0, 80.0, "B: mask<80"),
    (3.0, 80.0, "C: dist>=3 mask<80"),
    (2.5, 95.0, "D: dist>=2.5 mask<95"),
];

fn would_fire(dist: f32, mask: Option<f32>, dist_gate: f32, mask_gate: f32) -> bool {
    if dist < dist_gate {
        return false;
    }
    match mask {
        Some(m) => m < mask_gate,
        None => false,
    }
}

fn encode_with_variant(rgb: &[u8], w: u32, h: u32, d: f32, force_w44_29: bool) -> usize {
    let mut cfg = LossyConfig::new(d).with_effort(7).with_threads(8);
    if force_w44_29 {
        cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides { high_d_photo_hint: Some(true), ..Default::default() });
    } else {
        // Force OFF so we get a clean baseline that isn't accidentally
        // hit by the existing default gate.
        cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides { high_d_photo_hint: Some(false), ..Default::default() });
    }
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
        .len()
}

const CJXL_BIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";

fn cjxl_size(src: &Path, d: f32) -> Option<usize> {
    let tmp = format!(
        "/tmp/w44_78_cjxl_{}_{}.jxl",
        std::process::id(),
        (d * 10.0) as u32
    );
    let out = Command::new(CJXL_BIN)
        .args(["-d", &d.to_string(), "-e", "7", src.to_str()?, &tmp])
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!("cjxl failed for {:?} d={}: status={}", src, d, out.status);
        return None;
    }
    let sz = std::fs::metadata(&tmp).ok()?.len() as usize;
    let _ = std::fs::remove_file(&tmp);
    Some(sz)
}

/// Approximate the mask1x1_median per image. The encoder pipeline
/// computes this internally — we re-use the known values measured
/// via examples/w44_65_encoder_mask1x1_probe.
///
/// Anything not in this table will be probed empirically (the
/// `debug-w44-65` feature prints them).
fn known_mask_median(name: &str) -> Option<f32> {
    let table: &[(&str, f32)] = &[
        // CID22 photos
        ("1025469.png", 76.08),
        ("1044329.png", 48.03),
        ("1189261.png", 69.08),
        ("1279330.png", 83.76),
        ("1418519.png", 92.33),
        ("1420710.png", 39.55),
        ("1475938.png", 89.32),
        ("1531677.png", 35.63),
        ("1544947.png", 57.58),
        ("159550.png", 72.76),
        ("1624487.png", 79.69),
        ("162520.png", 64.49),
        ("164595.png", 71.88),
        ("2079234.png", 66.33),
        ("2190188.png", 55.83),
        ("225228.png", 77.05),
        ("2253934.png", 73.60),
        ("2389166.png", 46.24),
        ("2670327.png", 60.35),
        ("2736139.png", 59.84),
        ("2775196.png", 74.73),
        ("2887497.png", 88.18),
        ("2936831.png", 63.41),
        ("297394.png", 62.00),
        ("3156482.png", 69.42),
        ("3316926.png", 82.20),
        ("3637739.png", 47.80),
        // (a few more truncated — incomplete CID22 probe)
        // Screenshots
        ("codec_wiki.png", 100.01),
        ("imac_g3.png", 100.01),
        ("imac_dark.png", 100.01),
        ("terminal.png", 100.01),
        ("windows.png", 100.01),
        ("windows95.png", 99.06),
        ("imessage.png", 100.01),
        ("graph.png", 100.01),
    ];
    table.iter().find(|(n, _)| *n == name).map(|(_, m)| *m)
}

fn main() {
    eprintln!("W44-78 widen-gate A/B sweep starting");
    eprintln!(
        "Variants: {:?}",
        VARIANTS.iter().map(|v| v.2).collect::<Vec<_>>()
    );

    let mut all_images: Vec<(String, PathBuf, bool)> = Vec::new();

    // CID22 first (all 41)
    eprintln!("Reading CID22 dir: {}", CID22);
    if let Ok(entries) = std::fs::read_dir(CID22) {
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        for p in paths {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            all_images.push((name, p, false));
        }
    }

    // Screenshots (subset to avoid huge runtime)
    for name in &[
        "codec_wiki.png",
        "imac_g3.png",
        "terminal.png",
        "windows95.png",
        "graph.png",
    ] {
        let p = PathBuf::from(GB82).join(name);
        if p.exists() {
            all_images.push((name.to_string(), p, true));
        }
    }

    // header
    println!(
        "image\tclass\tdistance\tmask1x1\tcjxl_bytes\tdefault_bytes\t{}",
        VARIANTS
            .iter()
            .map(|v| format!("{}_bytes\t{}_delta_pct", v.2, v.2))
            .collect::<Vec<_>>()
            .join("\t")
    );

    let mut totals_per_variant: BTreeMap<String, (f64, i32)> = BTreeMap::new();
    let mut flips_per_variant: BTreeMap<String, (i32, i32)> = BTreeMap::new(); // (fixed_to_open, open_to_fixed)

    eprintln!("Found {} images total", all_images.len());
    let total = all_images.len();
    for (i, (name, path, is_screenshot)) in all_images.iter().enumerate() {
        eprintln!("[{}/{}] Processing: {}", i + 1, total, name);
        let img = match image::open(path) {
            Ok(i) => i,
            Err(_) => continue,
        };
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let raw = rgb.as_raw().clone();

        let mask = known_mask_median(name);

        for &d in DISTANCES {
            let cjxl_b = match cjxl_size(path, d) {
                Some(s) => s,
                None => continue,
            };
            // Default = current production gate (W44-29 auto, no override).
            // We use Some(false) for the "would not have fired anyway" baseline
            // and Some(true) for the variant-triggered case so we get a clean
            // diff.
            let default_fires = would_fire(d, mask, 4.0, 50.0);
            let default_b = encode_with_variant(&raw, w, h, d, default_fires);

            let mut variant_results: Vec<(String, usize)> = Vec::new();
            for &(dg, mg, lab) in VARIANTS {
                let fires = would_fire(d, mask, dg, mg);
                let b = if fires == default_fires {
                    // Same gate decision -> same bytes
                    default_b
                } else {
                    encode_with_variant(&raw, w, h, d, fires)
                };
                variant_results.push((lab.to_string(), b));

                // Track totals and flips
                let total = totals_per_variant
                    .entry(lab.to_string())
                    .or_insert((0.0, 0));
                total.0 += (b as i64 - default_b as i64) as f64;
                total.1 += 1;

                let flips = flips_per_variant.entry(lab.to_string()).or_insert((0, 0));
                // FIXED -> OPEN: was at or under cjxl, now over (by margin)
                let was_fixed = default_b <= cjxl_b * 1015 / 1000; // 1.5% slack
                let now_open = b > cjxl_b * 1015 / 1000;
                if was_fixed && now_open {
                    flips.0 += 1;
                }
                let was_open = default_b > cjxl_b * 1015 / 1000;
                let now_fixed = b <= cjxl_b * 1015 / 1000;
                if was_open && now_fixed {
                    flips.1 += 1;
                }
            }

            let class = if *is_screenshot { "screen" } else { "photo" };
            let mask_str = mask.map(|m| format!("{:.2}", m)).unwrap_or("?".to_string());
            let variant_cols: Vec<String> = variant_results
                .iter()
                .map(|(_, b)| {
                    let pct = (*b as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0;
                    format!("{}\t{:+.3}", b, pct)
                })
                .collect();
            println!(
                "{}\t{}\t{:.1}\t{}\t{}\t{}\t{}",
                name,
                class,
                d,
                mask_str,
                cjxl_b,
                default_b,
                variant_cols.join("\t")
            );
        }
    }

    eprintln!("\n=== W44-78 totals ===");
    for (lab, (total, n)) in &totals_per_variant {
        let flips = flips_per_variant.get(lab).cloned().unwrap_or((0, 0));
        eprintln!(
            "{:30}  Δbytes={:+8.0}B over {} cells   FIXED→OPEN: {}   OPEN→FIXED: {}",
            lab, total, n, flips.0, flips.1
        );
    }
}
