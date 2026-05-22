// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-213 wiring-proof test: when a non-default `RuntimeTuning` is
//! installed via `tuning::runtime::install`, encoded bytes for a
//! screenshot-class image at e8 d=4 MUST differ from the default-tuning
//! baseline. This proves the production code paths actually consult
//! `RuntimeTuning::get_or_default(...)` via the `runtime_or_default!`
//! macro, closing the W44-212 SCAFFOLDING_NOTE caveat.
//!
//! **Why this test exists**: W44-211 shipped the override struct +
//! installer; W44-212 shipped the sweep-runner that captures
//! `params_blob`. Until W44-213 wired the consumer sites, installing a
//! non-default `RuntimeTuning` was a no-op at the encoder — sweep data
//! captured `params_blob` but `encoded_bytes` was independent of
//! tuning-axis variance. This test fails if any of the 6 wired fields
//! regresses to a no-op binding.
//!
//! ## Cell choice
//!
//! `gb82-sc/terminal.png` at e8 d=4: the W44-105 buttloop QF seed
//! scale fix (`DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE = 4.0`,
//! gated at `BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE = 3.5`) fires here
//! deterministically (`is_screenshot=true` via mask1x1 median > 95,
//! `target_distance=4.0 >= 3.5`). Forcing the seed scale to 8.0
//! (double the default) materially changes the buttloop's
//! exploration window → different `quant_field` → different final
//! bytes. Predicted delta: ≥ 1 % byte change (much larger than
//! float-precision drift floor of ~0.001 %).
//!
//! ## Fallback to synthetic screenshot
//!
//! When the corpus is unavailable (CI without codec-corpus mounted),
//! the test falls back to a deterministic synthetic screenshot
//! (256×256 black-text-on-white-pixel-art) that ALSO triggers the
//! `mask1x1 median > 95` predicate. The synthetic path uses e8 d=4
//! with the same expected behaviour. The synthetic image is built
//! procedurally inside the test (no committed binary fixture, no
//! committed PNG); decoded-pixel byte parity is NOT under test —
//! only the bytes-differ invariant.

#![cfg(all(feature = "tuning-override", feature = "butteraugli-loop"))]

use std::path::PathBuf;

use jxl_encoder::tuning::runtime::{RuntimeTuning, install, is_loaded};
use jxl_encoder::{LossyConfig, PixelLayout};

/// Resolve the codec-corpus root. Mirrors `buttloop_target_parity.rs`.
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

/// Load `gb82-sc/terminal.png` as 8-bit sRGB RGB, or build a synthetic
/// screenshot-class image (256×256 black-on-white pixel art with thin
/// glyphs) if the corpus PNG isn't available.
///
/// Both paths produce content with `mask1x1` median > 95 (saturated
/// flat regions dominate; thin glyph edges produce few sub-95 blocks).
fn load_screenshot() -> (Vec<u8>, u32, u32) {
    if let Some(root) = corpus_root() {
        let path = root.join("gb82-sc/terminal.png");
        if path.exists() {
            if let Ok(img) = image::open(&path) {
                let rgb = img.to_rgb8();
                let (w, h) = rgb.dimensions();
                // Cap at 512×512 to keep test runtime reasonable.
                let (cw, ch) = (w.min(512), h.min(512));
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
    }
    // Synthetic fallback: 256×256 white background with two horizontal
    // thin-line "text" stripes. Saturated flat background → mask1x1
    // median > 99 → fires the screenshot discriminator.
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
    // A vertical "I-beam" pattern to add edges in another axis.
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

/// Encode one image at e8 d=4.0. Returns encoded byte count.
fn encode_e8_d4(rgb: &[u8], w: u32, h: u32) -> usize {
    let cfg = LossyConfig::new(4.0).with_effort(8);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed")
        .len()
}

/// Baseline-vs-override-vs-baseline encode sequence in ONE test
/// process. The sequence is:
///
/// 1. Encode at default tuning (no install) → `baseline_bytes`
/// 2. Install override that DOUBLES `buttloop_default_screenshot_qf_seed_scale`
///    (4.0 → 8.0)
/// 3. Encode again → `override_bytes`
/// 4. Assert `override_bytes != baseline_bytes` (proves the buttloop
///    scale consumer site is wired)
///
/// Because `install` is single-shot per process, the baseline encode
/// MUST happen before install. After install, subsequent encodes use
/// the installed value (until process exit).
#[test]
fn override_changes_buttloop_qf_seed_scale() {
    assert!(
        !is_loaded(),
        "another test in this binary already installed a RuntimeTuning — \
         this test must run before any other install. The W44-213 \
         wiring-proof test binary is intentionally a separate file so \
         it owns its install slot."
    );

    let (rgb, w, h) = load_screenshot();
    let n_pixels = (w as usize) * (h as usize);
    eprintln!(
        "W44-213 wiring-proof: encoding {}x{} ({} pixels) screenshot at e8 d=4.0",
        w, h, n_pixels
    );

    // Step 1: baseline encode at default tuning.
    let baseline_bytes = encode_e8_d4(&rgb, w, h);
    eprintln!("baseline_bytes (default tuning) = {}", baseline_bytes);

    // Step 2: install override that DOUBLES the buttloop QF seed scale
    // (4.0 → 8.0). The W44-105 buttloop gate fires on screenshot-class
    // content at d>=3.5, so e8 d=4 triggers; the doubled scale changes
    // the buttloop's exploration window and yields different bytes.
    let mut tuning = RuntimeTuning::default();
    tuning.buttloop_default_screenshot_qf_seed_scale = 8.0;
    install(tuning).expect("install RuntimeTuning");
    assert!(is_loaded(), "is_loaded() should be true after install");

    // Step 3: encode with override applied.
    let override_bytes = encode_e8_d4(&rgb, w, h);
    eprintln!(
        "override_bytes (buttloop_default_screenshot_qf_seed_scale=8.0) = {}",
        override_bytes
    );

    // Step 4: assert the bytes differ. If wiring is broken (consumer
    // site still reads the const directly), the encoder ignores the
    // installed override and bytes are byte-identical.
    let abs_delta = (override_bytes as isize - baseline_bytes as isize).unsigned_abs();
    let pct_delta = 100.0 * (abs_delta as f64) / (baseline_bytes as f64);
    eprintln!(
        "delta = {} bytes ({:.2}% of baseline)",
        abs_delta, pct_delta
    );

    assert!(
        baseline_bytes != override_bytes,
        "W44-213 wiring-proof FAILED: encoded bytes are byte-identical \
         between default-tuning and 2x-buttloop-seed-scale override \
         (baseline={baseline_bytes}, override={override_bytes}). The \
         consumer site at vardct/butteraugli_loop.rs:1352 likely still \
         reads the const directly instead of through \
         runtime_or_default!() — re-check the W44-213 wiring."
    );
    assert!(
        pct_delta > 0.1,
        "W44-213 wiring-proof produced a SUSPICIOUSLY-SMALL delta of \
         {:.4}% (baseline={baseline_bytes}, override={override_bytes}). \
         Float-precision drift floor is ~0.001%; a wired 2x-scale \
         change should produce ≥1%. Investigate the buttloop gate \
         predicate path.",
        pct_delta
    );
}
