// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-1 integration test for `LossyConfig::with_auto_splines`.
//!
//! This nails the chunk-1 contract:
//!
//!   - The API flag exists, builds, and is preserved across `with_effort`.
//!   - The encoder respects the gate (effort >= 7, no manual splines set,
//!     `auto_splines == true`) and calls into `find_splines`.
//!   - `find_splines` (the chunk-1 stub) returns `vec![]`, so the
//!     effective splines vector is empty and the encoder short-circuits
//!     the spline-subtract path → **bytes must equal the default-config
//!     bitstream byte-for-byte**.
//!
//! When chunk 2 lands a real detector the byte-identity assertion below
//! flips to a `bytes_with_auto != bytes_without_auto` test on power-line
//! imagery, plus a `bytes_on_photo == bytes_without_auto` no-regression
//! gate for natural photos where no thin features exist.

use jxl_encoder::{LossyConfig, PixelLayout};

/// A 128×64 RGB image with a high-contrast horizontal "wire" — exactly
/// the kind of shape the chunk-2 detector should latch onto. Chunk 1's
/// stub must ignore it, so bytes match the default-config encode.
fn make_test_image(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![80u8; w * h * 3]; // mid-grey background
    // Bright horizontal line at row h/2 — the shape a Hough/ridge
    // detector would pick out as a candidate spline.
    let y = h / 2;
    for x in 0..w {
        let i = (y * w + x) * 3;
        rgb[i] = 240;
        rgb[i + 1] = 240;
        rgb[i + 2] = 240;
    }
    rgb
}

#[test]
fn auto_splines_default_is_off() {
    // Sanity: a fresh config must have auto-splines disabled so the
    // default path stays hash-locked.
    let cfg = LossyConfig::new(1.0).with_effort(7);
    assert!(
        !cfg.auto_splines(),
        "default LossyConfig must not enable auto-splines"
    );
}

#[test]
fn auto_splines_preserved_across_with_effort() {
    // Setting `with_auto_splines(true)` must survive a `with_effort`
    // call — same contract as `with_splines` (api.rs preserves the
    // manual splines on with_effort; this mirrors it).
    let cfg = LossyConfig::new(1.0).with_auto_splines(true).with_effort(8);
    assert!(
        cfg.auto_splines(),
        "with_auto_splines must survive with_effort"
    );
}

#[test]
fn auto_splines_with_stub_is_byte_identical_to_default() {
    const W: usize = 128;
    const H: usize = 64;
    let rgb = make_test_image(W, H);

    // Default config: auto-splines OFF.
    let bytes_off = LossyConfig::new(1.0)
        .with_effort(7)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("default-config encode must succeed");

    // Same config, auto-splines ON. With the chunk-1 stub
    // (`find_splines` returns `vec![]`) the encoder MUST short-circuit
    // the spline path and produce byte-identical output. Any divergence
    // here means either (a) the stub is no longer empty (broke the
    // chunk-1 contract), or (b) the auto-splines branch is mutating
    // state along the way (an unintended side effect).
    let bytes_on = LossyConfig::new(1.0)
        .with_effort(7)
        .with_auto_splines(true)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("auto-splines-on encode must succeed");

    assert_eq!(
        bytes_off,
        bytes_on,
        "chunk-1 stub must produce byte-identical output to default \
         (off len={}, on len={})",
        bytes_off.len(),
        bytes_on.len()
    );
}

#[test]
fn auto_splines_below_effort_gate_is_byte_identical() {
    // Below the libjxl `speed_tier <= kSquirrel` gate (effort < 7),
    // auto-splines must NOT fire even when set. Defensive — the encoder
    // also enforces this internally.
    const W: usize = 128;
    const H: usize = 64;
    let rgb = make_test_image(W, H);

    let bytes_off = LossyConfig::new(1.0)
        .with_effort(5)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("effort-5 default encode must succeed");

    let bytes_on = LossyConfig::new(1.0)
        .with_effort(5)
        .with_auto_splines(true)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("effort-5 auto-splines-on encode must succeed");

    assert_eq!(
        bytes_off, bytes_on,
        "auto-splines below effort-7 gate must produce byte-identical \
         output to default"
    );
}
