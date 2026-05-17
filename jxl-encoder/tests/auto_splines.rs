// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Integration tests for `LossyConfig::with_auto_splines`.
//!
//! Chunk-2 contract (this commit replaces chunk-1's stub):
//!
//!   - The API flag exists, builds, and is preserved across `with_effort`.
//!   - The encoder respects the gate (effort >= 7, no manual splines set,
//!     `auto_splines == true`) and calls into `find_splines_at_distance`.
//!   - A high-contrast thin "power-line" synthetic image triggers the
//!     detector AND passes the cost-benefit gate → bytes differ vs the
//!     default-config encode.
//!   - A noise-like photo image either yields zero splines from the
//!     detector OR is rejected by the cost-benefit gate → bytes match
//!     the default-config encode (no photo regressions).
//!   - Below the effort gate (effort < 7), auto-splines is a no-op.

use jxl_encoder::{LossyConfig, PixelLayout};

/// RGB image with a single bright thin horizontal "wire" through the
/// middle row. Exactly the shape the chunk-2 detector targets;
/// callers pick a wide enough image (≥ ~1024px) that the path-length
/// driven cost-benefit gate admits the candidate.
fn make_power_line_image(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![80u8; w * h * 3];
    let y = h / 2;
    for x in 4..w - 4 {
        let i = (y * w + x) * 3;
        rgb[i] = 240;
        rgb[i + 1] = 240;
        rgb[i + 2] = 240;
    }
    rgb
}

/// Photo-like content: smoothly-varying brightness with low-amplitude
/// random noise. No ridges → detector should produce nothing, OR any
/// candidates it finds should be rejected by the cost gate. Either
/// way, bytes must match the default-config encode.
fn make_photo_like_image(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut seed = 0x9E3779B9u32; // golden-ratio splitmix-ish
    for y in 0..h {
        for x in 0..w {
            let ramp = (x * 200 / w + y * 50 / h) as i32;
            // xorshift LCG noise — deterministic, no rng dep.
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((seed >> 24) as i32) - 128;
            let v = (ramp + noise / 16).clamp(0, 255) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    rgb
}

#[test]
fn auto_splines_default_is_off() {
    let cfg = LossyConfig::new(1.0).with_effort(7);
    assert!(
        !cfg.auto_splines(),
        "default LossyConfig must not enable auto-splines"
    );
}

#[test]
fn auto_splines_preserved_across_with_effort() {
    let cfg = LossyConfig::new(1.0).with_auto_splines(true).with_effort(8);
    assert!(
        cfg.auto_splines(),
        "with_auto_splines must survive with_effort"
    );
}

/// Chunk-2: a power-line synthetic image triggers the detector AND
/// passes the cost gate. Bytes must differ vs the default-config
/// encode. (We don't assert strictly-smaller here because the spline
/// subtraction changes the VarDCT residual in a non-trivial way and
/// the encoded spline section has its own overhead — the contract is
/// "the detector ran and modified the encode".)
#[test]
fn auto_splines_power_line_changes_bitstream() {
    const W: usize = 1024;
    const H: usize = 256;
    let rgb = make_power_line_image(W, H);

    let bytes_off = LossyConfig::new(1.0)
        .with_effort(7)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("default-config encode must succeed");

    let bytes_on = LossyConfig::new(1.0)
        .with_effort(7)
        .with_auto_splines(true)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("auto-splines-on encode must succeed");

    assert_ne!(
        bytes_off,
        bytes_on,
        "chunk-2 detector must trigger on a long high-contrast thin \
         ridge; bytes_off len={} bytes_on len={} (if they're equal, \
         either the detector found nothing or the cost gate rejected \
         all candidates)",
        bytes_off.len(),
        bytes_on.len()
    );
}

/// Chunk-2 no-regression: photo-like content (smooth ramp + noise)
/// either produces zero spline candidates OR has all candidates
/// rejected by the cost-benefit gate. Bytes must match the
/// default-config encode.
#[test]
fn auto_splines_on_photo_is_byte_identical_to_default() {
    const W: usize = 256;
    const H: usize = 256;
    let rgb = make_photo_like_image(W, H);

    let bytes_off = LossyConfig::new(1.0)
        .with_effort(7)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("default-config photo encode must succeed");

    let bytes_on = LossyConfig::new(1.0)
        .with_effort(7)
        .with_auto_splines(true)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("auto-splines-on photo encode must succeed");

    assert_eq!(
        bytes_off,
        bytes_on,
        "photo-like content must not invent splines (cost gate must \
         reject all candidates); off len={} on len={}",
        bytes_off.len(),
        bytes_on.len()
    );
}

/// Below the libjxl `speed_tier <= kSquirrel` gate (effort < 7),
/// auto-splines must not fire even when set.
#[test]
fn auto_splines_below_effort_gate_is_byte_identical() {
    const W: usize = 1024;
    const H: usize = 256;
    let rgb = make_power_line_image(W, H);

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
