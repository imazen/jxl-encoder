// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Behavioral coverage of `LossyConfig::with_non_finite_action(...)`:
// - Default (`Error`) returns `EncodeError::InvalidInput` on non-finite input.
// - Opt-in (`Sanitize`) replaces non-finite with 0.0 and produces a valid bitstream.
// - On clean input, both modes produce byte-identical output (sanitize is a no-op).
//
// Uses the `RgbLinearF32` pixel layout to feed non-finite values into XYB
// — this is the legitimate caller-bug-style way to surface non-finite at
// the boundary, since 8-bit/16-bit integer layouts can't express NaN/Inf.

use bytemuck::cast_slice;
use jxl_encoder::{EncodeError, LossyConfig, NonFiniteAction, PixelLayout};

fn make_clean_pixels(w: u32, h: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut pixels = vec![0.0f32; n * 3];
    for i in 0..n {
        pixels[i * 3] = (i as f32 % 256.0) / 256.0;
        pixels[i * 3 + 1] = ((i / 256) as f32 % 256.0) / 256.0;
        pixels[i * 3 + 2] = 0.5;
    }
    pixels
}

#[test]
fn default_action_is_error() {
    // Confirm the API contract: `Default::default() == NonFiniteAction::Error`.
    assert_eq!(NonFiniteAction::default(), NonFiniteAction::Error);
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.non_finite_action(), NonFiniteAction::Error);
}

#[test]
fn error_mode_rejects_nan_input() {
    let (w, h) = (64u32, 64u32);
    let mut pixels = make_clean_pixels(w, h);
    // Plant a NaN somewhere in the middle of the buffer.
    pixels[(w * h * 3 / 2) as usize] = f32::NAN;

    let result = LossyConfig::new(1.0)
        // Default mode is Error, but be explicit for the test.
        .with_non_finite_action(NonFiniteAction::Error)
        .encode(cast_slice(&pixels), w, h, PixelLayout::RgbLinearF32);

    let err = result.expect_err("expected NonFiniteAction::Error to reject NaN input");
    let inner: &EncodeError = err.as_ref();
    match inner {
        EncodeError::InvalidInput { message } => {
            assert!(
                message.contains("non-finite")
                    || message.contains("NaN")
                    || message.contains("XYB"),
                "expected non-finite error, got: {message}"
            );
        }
        other => panic!("expected InvalidInput, got: {other:?}"),
    }
}

#[test]
fn error_mode_rejects_inf_input() {
    let (w, h) = (64u32, 64u32);
    let mut pixels = make_clean_pixels(w, h);
    pixels[5] = f32::INFINITY;

    let result = LossyConfig::new(1.0)
        .with_non_finite_action(NonFiniteAction::Error)
        .encode(cast_slice(&pixels), w, h, PixelLayout::RgbLinearF32);

    assert!(matches!(
        result.as_ref().map_err(|e| e.as_ref()),
        Err(EncodeError::InvalidInput { .. })
    ));
}

#[test]
fn sanitize_mode_accepts_nan_input() {
    let (w, h) = (64u32, 64u32);
    let mut pixels = make_clean_pixels(w, h);
    pixels[(w * h * 3 / 2) as usize] = f32::NAN;
    pixels[5] = f32::INFINITY;
    pixels[7] = f32::NEG_INFINITY;

    let bytes = LossyConfig::new(1.0)
        .with_non_finite_action(NonFiniteAction::Sanitize)
        .encode(cast_slice(&pixels), w, h, PixelLayout::RgbLinearF32)
        .expect("Sanitize mode should accept and replace non-finite values");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");
}

#[test]
fn modes_agree_on_clean_input() {
    let (w, h) = (128u32, 128u32);
    let pixels = make_clean_pixels(w, h);

    let bytes_error = LossyConfig::new(1.0)
        .with_non_finite_action(NonFiniteAction::Error)
        .encode(cast_slice(&pixels), w, h, PixelLayout::RgbLinearF32)
        .expect("Error mode encodes clean input");
    let bytes_sanitize = LossyConfig::new(1.0)
        .with_non_finite_action(NonFiniteAction::Sanitize)
        .encode(cast_slice(&pixels), w, h, PixelLayout::RgbLinearF32)
        .expect("Sanitize mode encodes clean input");

    // Sanitize on clean input is a no-op — both modes must produce
    // byte-identical bitstreams.
    assert_eq!(
        bytes_error, bytes_sanitize,
        "Error and Sanitize modes diverged on clean input — sanitize \
         should be a no-op when no replacements are needed"
    );
}

#[test]
fn error_mode_does_not_fire_on_integer_input() {
    // 8-bit RGB inputs cannot express NaN, so the boundary check should
    // never trip. Just a smoke test confirming default behavior on the
    // common case.
    let (w, h) = (128u32, 128u32);
    let pixels = vec![128u8; (w * h * 3) as usize];
    let bytes = LossyConfig::new(1.0)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("default Error mode should not fire on 8-bit input");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
}
