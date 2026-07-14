// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Characterize the beard-oil e7 distance non-monotonicity (task #12, #74).
//!
//! `9290_...beard-oil...1024x1536.png` (fcbr≈0.74, high flat-color) loses
//! +112 % bytes to cjxl at e7 d2.0 while d1.75 is fine — a distance-
//! monotonicity violation (higher distance ⇒ MORE bytes). This probe:
//!   1. Sweeps distance across the d2.0 boundary and prints bytes + the
//!      named AC-strategy histogram (`EncodeStats::strategy_counts`), so a
//!      transform that flips ON exactly at d2.0 is visible.
//!   2. Ablates each e7-gated `LossyInternalParams` knob at d2.0 to find
//!      which one, when disabled, removes the spike.
//!
//! usage:  beardoil_d2_probe <img.png>

use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_encoder::effort::LossyInternalParams;

/// (label, knob-tweak) pair for the d2.0 ablation table.
type Ablation = (&'static str, fn(&mut LossyInternalParams));

const NAMES: [&str; 19] = [
    "DCT8", "DCT16x8", "DCT8x16", "DCT16x16", "DCT32x32", "DCT4x8", "DCT8x4", "DCT4x4", "IDENTITY",
    "DCT2X2", "DCT32x16", "DCT16x32", "AFV0", "AFV1", "AFV2", "AFV3", "DCT64x64", "DCT64x32",
    "DCT32x64",
];

fn enc_stats(
    rgb: &[u8],
    w: u32,
    h: u32,
    d: f32,
    tweak: impl FnOnce(&mut LossyInternalParams),
) -> (usize, [u32; 19]) {
    let mut p = LossyInternalParams::default();
    tweak(&mut p);
    let res = LossyConfig::new(d)
        .with_effort(7)
        .with_internal_params(p)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode_with_stats(rgb)
        .expect("encode");
    let len = res.data().map(|d| d.len()).unwrap_or(0);
    (len, *res.stats().strategy_counts())
}

fn hist_str(h: &[u32; 19]) -> String {
    let total: u32 = h.iter().sum();
    let mut parts: Vec<(usize, u32)> = h
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, c)| *c > 0)
        .collect();
    parts.sort_by_key(|&(_, c)| core::cmp::Reverse(c));
    let top: Vec<String> = parts
        .iter()
        .take(6)
        .map(|(i, c)| {
            format!(
                "{}={}({:.0}%)",
                NAMES[*i],
                c,
                100.0 * *c as f64 / total.max(1) as f64
            )
        })
        .collect();
    format!("tot={total} {}", top.join(" "))
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: beardoil_d2_probe <img.png>");
    let img = image::open(&path).expect("open").to_rgb8();
    let (w, h) = (img.width(), img.height());
    let rgb = img.into_raw();
    println!("# {path} {w}x{h}");

    // --- 1. distance sweep across the d2.0 boundary ---
    println!("\n## distance sweep (production default, e7)");
    println!("{:>6}  {:>9}  strategy_histogram", "d", "bytes");
    let mut prev = 0usize;
    for &d in &[
        1.0_f32, 1.5, 1.75, 1.9, 1.99, 2.0, 2.01, 2.1, 2.25, 2.5, 3.0,
    ] {
        let (bytes, hist) = enc_stats(&rgb, w, h, d, |_| {});
        let delta = if prev > 0 {
            format!(
                "{:+.1}%",
                100.0 * (bytes as f64 - prev as f64) / prev as f64
            )
        } else {
            String::from("--")
        };
        println!("{d:>6.2}  {bytes:>9}  [{delta:>7}] {}", hist_str(&hist));
        prev = bytes;
    }

    // --- 2. knob ablation at d2.0 (the spike point) ---
    println!("\n## knob ablation at d2.0 (bytes, delta vs base, histogram)");
    let (base, base_h) = enc_stats(&rgb, w, h, 2.0, |_| {});
    println!(
        "{:<22} {:>9} {:>8}  {}",
        "base(default)",
        base,
        "--",
        hist_str(&base_h)
    );
    let ablations: &[Ablation] = &[
        ("try_dct64=false", |p| p.try_dct64 = Some(false)),
        ("try_dct32=false", |p| p.try_dct32 = Some(false)),
        ("try_dct16=false", |p| p.try_dct16 = Some(false)),
        ("try_dct4x8_afv=false", |p| p.try_dct4x8_afv = Some(false)),
        ("cfl_two_pass=false", |p| p.cfl_two_pass = Some(false)),
        ("cfl_keep_best=false", |p| p.cfl_keep_best = Some(false)),
        ("chromacity_adj=false", |p| {
            p.chromacity_adjustment = Some(false)
        }),
        ("fine_grained_step=1", |p| p.fine_grained_step = Some(1)),
        ("enh_clustering=false", |p| {
            p.enhanced_clustering_vardct = Some(false)
        }),
        ("non_aligned_eval=false", |p| {
            p.non_aligned_eval = Some(false)
        }),
    ];
    for (label, tweak) in ablations {
        let (bytes, hist) = enc_stats(&rgb, w, h, 2.0, *tweak);
        let delta = 100.0 * (bytes as f64 - base as f64) / base as f64;
        println!("{label:<22} {bytes:>9} {delta:>+7.1}%  {}", hist_str(&hist));
    }
}
