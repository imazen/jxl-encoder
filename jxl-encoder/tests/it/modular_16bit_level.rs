// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! `modular_16_bit_buffer_sufficient` and the codestream level must agree —
//! and both must match libjxl.
//!
//! libjxl `VerifyLevelSettings` (`encode.cc:585`) makes `!modular_16_bit_
//! buffer_sufficient` its FIRST level-5 check and returns level 10. So a
//! stream that carries the field as `false` while signalling level 5 (by
//! omitting the `jxll` box) violates the level it declares.
//!
//! We shipped both halves of that violation for every 16-bit input:
//!
//! * **lossless 16-bit** — field `false` (correct: the modular streams really
//!   do carry the caller's 16-bit samples) but level 5. cjxl v0.12 on the same
//!   source emits a container with `jxll = 10`.
//! * **lossy 16-bit** — field `false` (WRONG: on the XYB path the modular
//!   sub-bitstreams carry DC and AC metadata, not the original samples, so
//!   16-bit buffers suffice regardless of input depth — libjxl's
//!   `!uses_original_profile ||` clause at `encode.cc:1330`) and level 5.
//!   cjxl writes the field as `true` and stays at level 5.
//!
//! Nothing caught either one because no decoder we test with enforces levels,
//! and the level was computed from dimensions/extra-channels/ICC only.
//!
//! These tests read the container structure directly rather than trusting a
//! decoder, for the same reason: a decoder that ignores levels cannot fail.

use jxl_encoder::api::{LosslessConfig, LossyConfig, PixelLayout};

/// 32x32 16-bit RGB ramp — deep enough to exceed the 12-bit threshold, small
/// enough to encode instantly.
fn ramp_rgb16(w: usize, h: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 6);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = ((x * 2003 + y * 6151 + c * 21841) % 65536) as u16;
                px.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    px
}

fn ramp_rgb8(w: usize, h: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                px.push(((x * 7 + y * 13 + c * 29) % 256) as u8);
            }
        }
    }
    px
}

/// Reported codestream level: `Some(n)` from a `jxll` box, or 5 when the
/// output is a bare codestream / a container without one.
fn signalled_level(data: &[u8]) -> u8 {
    if data.starts_with(&[0xff, 0x0a]) {
        return 5; // bare codestream ⇒ level 5 by definition
    }
    match data.windows(4).position(|w| w == b"jxll") {
        Some(i) => data[i + 4],
        None => 5,
    }
}

#[test]
fn lossless_16bit_signals_level_10_like_cjxl() {
    let px = ramp_rgb16(32, 32);
    let data = LosslessConfig::new()
        .encode_request(32, 32, PixelLayout::Rgb16)
        .encode(&px)
        .expect("encode");
    assert_eq!(
        signalled_level(&data),
        10,
        "16-bit lossless needs 32-bit modular buffers, so it cannot be level 5 \
         (libjxl encode.cc:585); cjxl v0.12 emits jxll = 10 on the same input"
    );
    assert!(
        !data.starts_with(&[0xff, 0x0a]),
        "level 10 requires a container so the jxll box can be carried"
    );
}

#[test]
fn lossless_8bit_stays_level_5_and_bare() {
    let px = ramp_rgb8(32, 32);
    let data = LosslessConfig::new()
        .encode_request(32, 32, PixelLayout::Rgb8)
        .encode(&px)
        .expect("encode");
    assert_eq!(
        signalled_level(&data),
        5,
        "8-bit fits 16-bit modular buffers"
    );
    assert!(
        data.starts_with(&[0xff, 0x0a]),
        "level 5 with no metadata must stay a bare codestream — this is the \
         guard that the level fix did not start wrapping everything"
    );
}

#[test]
fn lossy_16bit_stays_level_5_because_xyb_does_not_carry_original_samples() {
    let px = ramp_rgb16(32, 32);
    let data = LossyConfig::new(1.0)
        .encode_request(32, 32, PixelLayout::Rgb16)
        .encode(&px)
        .expect("encode");
    assert_eq!(
        signalled_level(&data),
        5,
        "on the XYB path the modular streams carry DC/AC metadata, not the \
         caller's 16-bit samples (libjxl encode.cc:1330 `!uses_original_profile`), \
         so 16-bit buffers suffice and level 5 stands — cjxl v0.12 agrees"
    );
    assert!(data.starts_with(&[0xff, 0x0a]));
}

// The header FIELD itself is pinned directly, as a unit test of the rule in
// `headers::file_header::ImageMetadata::modular_16bit_buffer_sufficient`
// (see `modular_16bit_rule` there). Poking a hardcoded bit offset from an
// integration test was tried and abandoned: the offset is fixture-dependent,
// so the test measured the layout rather than the field. The rule is the thing
// that must be right, and the three level assertions above are what prove it
// reaches the wire.
