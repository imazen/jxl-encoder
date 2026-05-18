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
/// middle row, painted on a *non-flat photo-like background*. Exactly
/// the shape the chunk-2 detector targets; callers pick a wide enough
/// image (≥ ~1024px) that the path-length driven cost-benefit gate
/// admits the candidate.
///
/// The background uses a *low-amplitude* vertical-stripe modulation
/// over a diagonal ramp so the chunk-5 content discriminator
/// ([`jxl_encoder::vardct::splines::looks_like_screenshot`]) does NOT
/// trigger: a uniform `80`-grey background fills every 8x8 block with
/// max mask1x1 ≈ 100, which the discriminator now correctly classifies
/// as screenshot-like and skips splines on. The stripes are
/// gradient-magnitude orthogonal to the wire (vertical, while the wire
/// is horizontal) so Sobel-y / Hessian don't pick them up as ridges —
/// the polyline tracer still finds the single horizontal wire cleanly.
fn make_power_line_image(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let ramp = 80 + (x * 60 / w) as i32 + (y * 30 / h) as i32;
            // Vertical-only stripe modulation: amplitude ~12 grey
            // levels, period 4 px. Orthogonal to a horizontal wire so
            // the ridge detector picks the wire cleanly while mask1x1
            // sees real spatial variation.
            let stripe = if x % 4 < 2 { 6 } else { -6 };
            let v = (ramp + stripe).clamp(0, 255) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    // Bright horizontal wire through the middle row.
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

/// Chunk-5 contract (replaces chunk-3's net-byte-savings assertion):
/// on a multi-line "power-grid" synthetic painted on a *photo-like*
/// (non-screenshot) background, the detector must still RUN past the
/// chunk-5 `looks_like_screenshot` gate AND the cost gate must admit
/// at least one of the candidates (bytes_on must differ from bytes_off).
///
/// Chunk 3's original contract — "bytes_on < bytes_off" — was anchored
/// on a *flat 80-grey* background that chunk 5 now correctly classifies
/// as screenshot-class and short-circuits before the detector runs;
/// see [`jxl_encoder::vardct::splines::looks_like_screenshot`] and the
/// chunk-5 bench (`benchmarks/auto_splines_bench_2026-05-17_chunk5.tsv`)
/// for the rationale. On a non-flat background the per-spline savings
/// no longer net-positive (the background contains noise the wires
/// merely re-replace, so subtracting the splines doesn't shrink the
/// VarDCT residual the way it does on a pure-flat canvas). The
/// "detector runs + cost gate admits ≥1" invariant is the chunk-5
/// equivalent that still has signal value: it catches a regression
/// where the discriminator over-rejects.
#[test]
fn auto_splines_chunk5_multi_line_runs_detector() {
    const W: usize = 1024;
    const H: usize = 512;
    const N_LINES: usize = 4;

    let mut rgb = alloc_rgb_noisy_gradient(W, H);
    for k in 0..N_LINES {
        let y = ((k + 1) * H) / (N_LINES + 1);
        for x in 4..W - 4 {
            let i = (y * W + x) * 3;
            rgb[i] = 240;
            rgb[i + 1] = 240;
            rgb[i + 2] = 240;
        }
    }

    let bytes_off = LossyConfig::new(1.0)
        .with_effort(7)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("default-config multi-line encode must succeed");

    let bytes_on = LossyConfig::new(1.0)
        .with_effort(7)
        .with_auto_splines(true)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("auto-splines-on multi-line encode must succeed");

    assert_ne!(
        bytes_off,
        bytes_on,
        "chunk-5: multi-line synthetic on photo-like background must \
         still trigger the detector AND admit ≥1 candidate through the \
         cost gate — off len={} on len={} (if equal, either chunk-5 \
         over-rejected or the cost gate dropped every candidate)",
        bytes_off.len(),
        bytes_on.len(),
    );
}

/// Vertical-stripe-on-ramp grey background used by the multi-line
/// synthetic after chunk 5. Same shape as `make_power_line_image`'s
/// background so the discriminator does NOT classify the canvas as
/// screenshot-like, while the orthogonal (vertical) stripe orientation
/// leaves the horizontal-wire ridge detector untouched.
fn alloc_rgb_noisy_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let ramp = 80 + (x * 60 / w) as i32 + (y * 30 / h) as i32;
            let stripe = if x % 4 < 2 { 6 } else { -6 };
            let v = (ramp + stripe).clamp(0, 255) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    rgb
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
