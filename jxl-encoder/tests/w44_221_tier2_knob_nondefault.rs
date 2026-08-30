// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-221 Tier-2 knob wiring proof (non-default branch).
//!
//! Companion to `w44_221_tier2_knob_expander.rs`. Lives in a separate
//! file → separate test binary → fresh `runtime::install` slot.
//!
//! Asserts:
//! 1. Installing a non-default-knob expansion (max
//!    `screenshot_quant_aggressiveness`) DOES change encoded bytes.
//!    Proves the expander reaches the production consumer sites
//!    (`vardct/butteraugli_loop.rs` + `vardct/encoder.rs`).
//! 2. The resulting bytes decode cleanly via `jxl-oxide`.

#![cfg(all(feature = "tuning-override", feature = "butteraugli-loop"))]

use std::path::PathBuf;

use jxl_encoder::__test_exports::coupling::Tier2Knobs;
use jxl_encoder::tuning_runtime::{install, is_loaded};
use jxl_encoder::{LossyConfig, PixelLayout};

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

#[test]
fn nondefault_tier2_knobs_change_bytes_and_decode() {
    assert!(
        !is_loaded(),
        "another test in this binary already installed a RuntimeTuning"
    );

    let (rgb, w, h) = load_screenshot();
    eprintln!("W44-221 non-default: encoding {w}x{h} screenshot");

    // Step 1: baseline encode (no install).
    let baseline = encode_e8_d4(&rgb, w, h);
    eprintln!("baseline_bytes (no install) = {}", baseline.len());

    // Step 2: install MAX screenshot_quant_aggressiveness knob expansion.
    let knobs = Tier2Knobs {
        screenshot_quant_aggressiveness: 2.0,
        ..Default::default()
    };
    let runtime = knobs.expand_to_runtime_tuning();
    eprintln!("installed knobs: {:?}", knobs);
    eprintln!(
        "expanded RuntimeTuning: p3={}, p4={}, p5={}, p6={}",
        runtime.buttloop_default_screenshot_qf_seed_scale,
        runtime.buttloop_qf_seed_scale_min_distance,
        runtime.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
        runtime.adaptive_quant_screenshot_qf_seed_scale_e7,
    );
    install(runtime).expect("install RuntimeTuning");

    // Step 3: encode with non-default expansion installed.
    let modified = encode_e8_d4(&rgb, w, h);
    eprintln!("modified_bytes (a=2.0 knob) = {}", modified.len());

    // Bytes must differ — non-default p3 + p6 must wire through.
    let abs_delta = (modified.len() as isize - baseline.len() as isize).unsigned_abs();
    let pct_delta = 100.0 * (abs_delta as f64) / (baseline.len() as f64);
    eprintln!(
        "delta = {} bytes ({:.2}% of baseline)",
        abs_delta, pct_delta
    );

    assert_ne!(
        baseline.len(),
        modified.len(),
        "W44-221 wiring FAILED: bytes identical despite non-default knobs. \
         Expander not reaching consumer sites."
    );
    assert!(
        pct_delta > 0.1,
        "W44-221 wiring produced a suspiciously-small delta of {:.4}% \
         (baseline={} modified={}). Float-precision drift floor is ~0.001%; \
         a wired a=2.0 knob should produce ≥1% on screenshot-class content.",
        pct_delta,
        baseline.len(),
        modified.len(),
    );

    // Step 4: decode through jxl-oxide.
    let cursor = std::io::Cursor::new(&modified);
    let img = jxl_oxide::JxlImage::builder()
        .read(cursor)
        .expect("jxl-oxide read header");
    let _render = img.render_frame(0).expect("jxl-oxide render frame 0");
    let (out_w, out_h) = (img.width(), img.height());
    eprintln!("jxl-oxide decoded {}x{}", out_w, out_h);
    assert_eq!(out_w, w);
    assert_eq!(out_h, h);
}
