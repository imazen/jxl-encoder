//! W44-91 dispatch A/B: paired interleaved encode of TARGET + REGRESSION
//! cells with and without the zenanalyze-proxy auto-dispatch.
//!
//! Measures whether the new gate:
//!   1. Catches 1189261 at d∈{3,4,5} (OPEN → FIXED-band savings)
//!   2. Stays OFF at d=6 on 1189261 (avoid the W44-79 +560 B regression)
//!   3. Stays OFF on all 6 documented W44-78 regression-band images
//!      at all distances
//!   4. Doesn't disturb FIXED cells from W44-78 (mask<50 reference fires)
//!
//! Build:
//!   cargo run -p jxl-encoder --release \
//!     --features 'parallel' \
//!     --example w44_91_dispatch_ab

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82: &str = "/home/lilith/work/codec-corpus/gb82-sc";

// (corpus, class, image, expected_w44_91_fires_in_d_3_to_5)
const CELLS: &[(&str, &str, &str, bool)] = &[
    (CID22, "TARGET", "1189261.png", true),
    (CID22, "REGRESSION", "1025469.png", false),
    (CID22, "REGRESSION", "1624487.png", false),
    (CID22, "REGRESSION", "159550.png", false),
    (CID22, "REGRESSION", "2079234.png", false),
    (CID22, "REGRESSION", "2775196.png", false),
    (CID22, "REGRESSION", "297394.png", false),
    // mask<50 — W44-29 fires, NOT W44-91 (verifies no interference).
    (CID22, "W44_78_FIRES", "1420710.png", false),
    (CID22, "W44_78_FIRES", "1531677.png", false),
    (CID22, "W44_78_FIRES", "1044329.png", false),
    (CID22, "W44_78_FIRES", "2389166.png", false),
    (CID22, "W44_78_FIRES", "3637739.png", false),
    (CID22, "MASK_HIGH", "1418519.png", false),
    // gb82-sc screenshots — mask1x1 ≈ 100 (above 80 cap), W44-91 cannot
    // fire. Verifies the screenshot class stays on its existing W22-1
    // / W44-65 dispatch path.
    (GB82, "SCREENSHOT", "codec_wiki.png", false),
    (GB82, "SCREENSHOT", "imac_dark.png", false),
    (GB82, "SCREENSHOT", "terminal.png", false),
    (GB82, "SCREENSHOT", "windows95.png", false),
];
const DISTANCES: &[f32] = &[2.5, 3.0, 4.0, 5.0, 6.0];

fn encode(rgb: &[u8], w: u32, h: u32, d: f32, force_off: bool) -> usize {
    let mut cfg = LossyConfig::new(d).with_effort(7).with_threads(8);
    if force_off {
        // Suppress both W44-29 and W44-91 (they share the same hint).
        cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides { high_d_photo_hint: Some(false), ..Default::default() });
    }
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
        .len()
}

fn main() {
    println!("# W44-91 dispatch A/B sweep");
    println!("# baseline_off = LossyConfig::with_high_d_photo_hint(Some(false))");
    println!("# default      = LossyConfig::new(...) — auto W44-29 + W44-91 gates");
    println!("# delta = default - baseline_off; negative = saving");
    println!("class\timage\tdistance\tbaseline_off\tdefault\tdelta\tdelta_pct\tw44_91_expected");

    for (corpus, class, name, exp_fires) in CELLS {
        let p = Path::new(corpus).join(name);
        if !p.exists() {
            eprintln!("MISSING: {}", p.display());
            continue;
        }
        let img = image::open(&p).unwrap();
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let raw = rgb.into_raw();
        for &d in DISTANCES {
            // Interleave the two encodes (paired) per the W44-88 methodology
            // lesson: AB instead of A then all B.
            let a = encode(&raw, w, h, d, true);
            let b = encode(&raw, w, h, d, false);
            let delta = (b as i64) - (a as i64);
            let delta_pct = (delta as f64) / (a as f64) * 100.0;
            // Expected fire only when d in [3, 5] AND TARGET class
            let exp_in_band = *exp_fires && (d >= 3.0 - 0.01) && (d <= 5.0 + 0.01);
            println!(
                "{}\t{}\t{:.1}\t{}\t{}\t{}\t{:+.3}\t{}",
                class, name, d, a, b, delta, delta_pct, exp_in_band
            );
        }
    }
}
