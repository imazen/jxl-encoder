// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-221 Tier-2 knob expander wiring proof.
//!
//! Three assertions in one process (because `runtime::install` is
//! single-shot):
//!
//! 1. **Default round-trip**: encoding a screenshot at default
//!    `Tier2Knobs` produces bytes IDENTICAL to a fresh process
//!    encoding with NO override installed. Proves the expander's
//!    default mapping is byte-correct.
//! 2. **Non-default behaviour change**: installing the expansion of
//!    `Tier2Knobs` with `screenshot_quant_aggressiveness = 2.0`
//!    (max value; ~6.8× lift on p3 + p6) MUST change the encoded
//!    bytes. Proves the expander reaches the production consumer
//!    sites.
//! 3. **Multi-decoder roundtrip**: the non-default-knob output must
//!    decode cleanly through `jxl-oxide` (and djxl when available).
//!
//! Because `install` is single-shot per process, we run two test
//! functions in separate test binaries — Cargo gives us a per-`#[test]`
//! process when each `#[test]` lives in its own file and we use
//! `--test-threads=1`. Here we use a single file with two `#[test]`s
//! split by RUNTIME ORDER: the default-roundtrip test installs
//! `Tier2Knobs::default()` first, then proves bytes match the
//! pre-install baseline.
//!
//! The non-default-behaviour test goes in `_nondefault.rs` (a separate
//! file → separate test binary → fresh `install` slot).

#![cfg(all(feature = "tuning-override", feature = "butteraugli-loop"))]

use std::path::PathBuf;

use jxl_encoder::tuning::coupling::Tier2Knobs;
use jxl_encoder::tuning::runtime::{RuntimeTuning, install, is_loaded};
use jxl_encoder::{LossyConfig, PixelLayout};

/// Resolve the codec-corpus root. Mirrors `w44_213_runtime_tuning_wiring.rs`.
fn corpus_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("CODEC_CORPUS_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join("work/codec-corpus");
    if p.exists() { Some(p) } else { None }
}

/// Load `gb82-sc/terminal.png` 256×256 crop, or build synthetic
/// screenshot-class image (same pattern as `w44_213_runtime_tuning_wiring.rs`).
fn load_screenshot() -> (Vec<u8>, u32, u32) {
    if let Some(root) = corpus_root() {
        let path = root.join("gb82-sc/terminal.png");
        if path.exists()
            && let Ok(img) = image::open(&path)
        {
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            let (cw, ch) = (w.min(256), h.min(256));
            let mut out = Vec::with_capacity((cw * ch * 3) as usize);
            for y in 0..ch {
                for x in 0..cw {
                    let p = rgb.get_pixel(x, y);
                    out.extend_from_slice(&p.0);
                }
            }
            return (out, cw, ch);
        }
    }
    // Synthetic fallback: 256×256 white background with thin black
    // horizontal stripes (mask1x1 median > 99 → screenshot dispatch fires).
    let (w, h) = (256u32, 256u32);
    let mut pixels = vec![255u8; (w * h * 3) as usize];
    for y in [64u32, 192u32] {
        for x in 16..(w - 16) {
            let i = ((y * w + x) * 3) as usize;
            pixels[i] = 0;
            pixels[i + 1] = 0;
            pixels[i + 2] = 0;
        }
    }
    for x in [64u32, 128u32, 192u32] {
        for y in 16..(h - 16) {
            let i = ((y * w + x) * 3) as usize;
            pixels[i] = 0;
            pixels[i + 1] = 0;
            pixels[i + 2] = 0;
        }
    }
    (pixels, w, h)
}

fn encode_e8_d4(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let cfg = LossyConfig::new(4.0).with_effort(8);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed")
}

/// W44-221 default-knob round-trip: installing `Tier2Knobs::default()`
/// expansion must produce bytes byte-identical to no-install baseline.
#[test]
fn default_tier2_knobs_roundtrip_byte_identical() {
    assert!(
        !is_loaded(),
        "another test in this binary already installed a RuntimeTuning"
    );

    let (rgb, w, h) = load_screenshot();
    eprintln!("W44-221 default-roundtrip: encoding {w}x{h} screenshot");

    // Step 1: baseline encode (no install).
    let baseline_bytes = encode_e8_d4(&rgb, w, h);
    eprintln!("baseline_bytes (no install) = {}", baseline_bytes.len());

    // Step 2: expand default knobs → RuntimeTuning, install.
    let knobs = Tier2Knobs::default();
    let runtime = knobs.expand_to_runtime_tuning();

    // Verify expansion produces RuntimeTuning::default()
    let default = RuntimeTuning::default();
    assert_eq!(
        runtime.smart_zenjxl_photo_mask_p25_min,
        default.smart_zenjxl_photo_mask_p25_min
    );
    assert_eq!(
        runtime.screenshot_median_threshold,
        default.screenshot_median_threshold
    );
    assert_eq!(
        runtime.buttloop_default_screenshot_qf_seed_scale,
        default.buttloop_default_screenshot_qf_seed_scale
    );
    assert_eq!(
        runtime.buttloop_qf_seed_scale_min_distance,
        default.buttloop_qf_seed_scale_min_distance
    );
    assert_eq!(
        runtime.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
        default.adaptive_quant_screenshot_qf_seed_scale_e5_e6
    );
    assert_eq!(
        runtime.adaptive_quant_screenshot_qf_seed_scale_e7,
        default.adaptive_quant_screenshot_qf_seed_scale_e7
    );

    install(runtime).expect("install RuntimeTuning");
    assert!(is_loaded());

    // Step 3: encode with default expansion installed.
    let installed_bytes = encode_e8_d4(&rgb, w, h);
    eprintln!(
        "installed_bytes (default Tier2Knobs expansion) = {}",
        installed_bytes.len()
    );

    // The two encodes MUST be byte-for-byte identical. Anything else
    // means the expander's "default round-trip" contract is broken.
    assert_eq!(
        baseline_bytes,
        installed_bytes,
        "W44-221 default-roundtrip FAILED: installing Tier2Knobs::default() \
         changed encoded bytes (baseline={} installed={}). The expander \
         must round-trip to RuntimeTuning::default() byte-for-byte to \
         preserve the hash-lock contract.",
        baseline_bytes.len(),
        installed_bytes.len(),
    );
}
