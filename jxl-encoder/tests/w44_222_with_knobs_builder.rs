// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-222 `LossyConfig::with_knobs` builder wiring proof.
//!
//! Because `runtime::install` is single-shot per process, ALL assertions
//! that depend on installed/not-installed state must run sequentially in
//! ONE test function. `cargo test` parallelizes `#[test]`s by default;
//! splitting these into separate `#[test]` functions makes the order
//! non-deterministic and pollutes the OnceLock across tests.
//!
//! The single sequential test below covers all 4 assertions:
//!
//! 1. **Default knobs → no override installed → byte-identical to no-knobs**:
//!    encoding with `LossyConfig::default().with_knobs(Tier2Knobs::default())`
//!    produces bytes IDENTICAL to encoding without `.with_knobs()` at all.
//!    Proves the default-detection short-circuit in `encode_inner` correctly
//!    skips `install_or_check_idempotent` (preserves the no-override fast
//!    path → hash-lock invariant).
//!
//! 2. **Non-default knobs → bytes change AND install runtime tuning**.
//!
//! 3. **Idempotent re-install with SAME knobs succeeds (no error)**.
//!
//! 4. **MISMATCHED knobs return [`EncodeError::InvalidConfig`]** — the
//!    single-shot limitation is surfaced as a user-error, not a silent
//!    fallback.
//!
//! A SECOND test exercises only the getter API; no `runtime::install` is
//! reached (because no `LossyConfig::encode` is called).

#![cfg(feature = "tuning-override")]

use std::path::PathBuf;

use jxl_encoder::tuning::coupling::Tier2Knobs;
use jxl_encoder::tuning::runtime::is_loaded;
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

/// Load `gb82-sc/terminal.png` 256×256 crop, or build synthetic
/// screenshot-class image (mask1x1 median > 99 → screen dispatch fires).
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
    // horizontal stripes (mask1x1 median > 99 → screen dispatch fires).
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

fn encode(cfg: LossyConfig, rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed")
}

/// Single sequential test covering all 4 W44-222 wiring assertions.
/// Single-shot `runtime::install` means we MUST keep these in one test
/// to control ordering.
#[test]
fn w44_222_with_knobs_builder_full_sequence() {
    // ─── Assertion 1: Default knobs == no knobs (byte-identical) ───
    //
    // Pre-condition: no runtime tuning installed (this is the FIRST
    // encode in the test process).
    assert!(
        !is_loaded(),
        "runtime tuning should not be installed at the start of the W44-222 test process; \
         if this fails, another test in the same binary installed a tuning earlier"
    );

    let (rgb, w, h) = load_screenshot();

    // Encode A: no `.with_knobs()` at all → no install attempted.
    let cfg_no_knobs = LossyConfig::new(2.0).with_effort(5);
    let bytes_no_knobs = encode(cfg_no_knobs, &rgb, w, h);

    // Encode B: `.with_knobs(Tier2Knobs::default())` → the default-detection
    // short-circuit in encode_inner skips install_or_check_idempotent.
    let cfg_default_knobs = LossyConfig::new(2.0)
        .with_effort(5)
        .with_knobs(Tier2Knobs::default());
    let bytes_default_knobs = encode(cfg_default_knobs, &rgb, w, h);

    assert_eq!(
        bytes_no_knobs.len(),
        bytes_default_knobs.len(),
        "(1) default-knobs encode should be byte-length-identical to no-knobs encode"
    );
    assert_eq!(
        bytes_no_knobs, bytes_default_knobs,
        "(1) default-knobs encode should be byte-for-byte-identical to no-knobs encode"
    );
    assert!(
        !is_loaded(),
        "(1) default knobs should NOT install a runtime tuning \
         (preserves hash-lock fast path)"
    );

    // ─── Assertion 2: Non-default knobs change bytes AND install ───
    let cfg_b = LossyConfig::new(2.0).with_effort(5).with_knobs(Tier2Knobs {
        buttloop_aq_balance: 0.5,
        ..Default::default()
    });
    let bytes_b = encode(cfg_b, &rgb, w, h);

    assert!(
        is_loaded(),
        "(2) non-default knobs should install a runtime tuning"
    );

    // The byte difference vs the pre-install no-knobs baseline: with the
    // override now installed, encoding with no `.with_knobs()` ALSO picks
    // up the override (the production code path reads from
    // `runtime::get_or_default` which short-circuits to the installed
    // override). So we compare against the pre-install snapshot.
    assert_ne!(
        bytes_no_knobs, bytes_b,
        "(2) non-default knobs (buttloop_aq_balance=0.5) MUST change bytes \
         vs the pre-install no-knobs baseline"
    );

    // ─── Assertion 3: Idempotent re-install with SAME knobs succeeds ───
    let cfg_c = LossyConfig::new(2.0).with_effort(5).with_knobs(Tier2Knobs {
        buttloop_aq_balance: 0.5,
        ..Default::default()
    });
    let bytes_c = encode(cfg_c, &rgb, w, h);
    assert_eq!(
        bytes_b, bytes_c,
        "(3) re-encoding with the SAME knobs should produce identical bytes \
         (install_or_check_idempotent is a no-op when value matches)"
    );

    // ─── Assertion 4: MISMATCHED knobs return InvalidConfig ───
    let cfg_d = LossyConfig::new(2.0).with_effort(5).with_knobs(Tier2Knobs {
        buttloop_aq_balance: 0.5,
        smoothness_bias: 0.3, // different from the installed override
        ..Default::default()
    });
    let r = cfg_d.encode(&rgb, w, h, PixelLayout::Rgb8);
    assert!(
        r.is_err(),
        "(4) encode with mismatched knobs should return InvalidConfig"
    );
    let err = r.err().unwrap();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("single-shot")
            || msg.contains("with_knobs")
            || msg.contains("InvalidConfig")
            || msg.contains("InvalidConfig"),
        "(4) error message should mention with_knobs single-shot limitation; got: {}",
        msg
    );
}

/// Getter sanity: `LossyConfig::knobs()` returns what was set. No encode
/// is called → no runtime::install fires → safe to run in any order.
#[test]
fn knobs_getter_returns_what_was_set() {
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.knobs(), None, "default has no knobs");

    let k = Tier2Knobs {
        buttloop_aq_balance: 0.7,
        ..Default::default()
    };
    let cfg2 = cfg.with_knobs(k);
    assert_eq!(cfg2.knobs(), Some(k));
}
