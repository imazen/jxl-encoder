// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression tests for the streaming-encoder dimension validation gate (H1).
//!
//! Both `LossyConfig::encoder` and `LosslessConfig::encoder` previously accepted
//! `width = u32::MAX`, which the `write_size_u2s` 30-bit field would silently
//! truncate, producing a JXL header whose declared dimensions did not match the
//! encoded data. The shared `validate_dims` helper now rejects any dimension
//! greater than `MAX_JXL_DIM = 2^30` up front.

use jxl_encoder::{At, EncodeError, LosslessConfig, LossyConfig, PixelLayout};

const MAX_JXL_DIM: u32 = 1 << 30;

fn require_limit_exceeded<T>(result: Result<T, At<EncodeError>>, ctx: &str) {
    match result {
        Ok(_) => panic!("{ctx}: expected error, got Ok"),
        Err(at) => match at.error() {
            EncodeError::LimitExceeded { .. } => {}
            other => panic!("{ctx}: expected LimitExceeded, got {other:?}"),
        },
    }
}

fn require_invalid_input<T>(result: Result<T, At<EncodeError>>, ctx: &str) {
    match result {
        Ok(_) => panic!("{ctx}: expected error, got Ok"),
        Err(at) => match at.error() {
            EncodeError::InvalidInput { .. } => {}
            other => panic!("{ctx}: expected InvalidInput, got {other:?}"),
        },
    }
}

#[test]
fn streaming_lossy_rejects_width_above_spec_max() {
    let result = LossyConfig::new(1.0).encoder(MAX_JXL_DIM + 1, 16, PixelLayout::Rgb8);
    require_limit_exceeded(result, "lossy width=2^30+1");
}

#[test]
fn streaming_lossy_rejects_height_above_spec_max() {
    let result = LossyConfig::new(1.0).encoder(16, MAX_JXL_DIM + 1, PixelLayout::Rgb8);
    require_limit_exceeded(result, "lossy height=2^30+1");
}

#[test]
fn streaming_lossy_rejects_u32_max() {
    let result = LossyConfig::new(1.0).encoder(u32::MAX, 2, PixelLayout::Rgb8);
    require_limit_exceeded(result, "lossy width=u32::MAX");
}

#[test]
fn streaming_lossy_rejects_zero_dim() {
    let result = LossyConfig::new(1.0).encoder(0, 16, PixelLayout::Rgb8);
    require_invalid_input(result, "lossy width=0");
    let result = LossyConfig::new(1.0).encoder(16, 0, PixelLayout::Rgb8);
    require_invalid_input(result, "lossy height=0");
}

#[test]
fn streaming_lossless_rejects_width_above_spec_max() {
    let result = LosslessConfig::new().encoder(MAX_JXL_DIM + 1, 16, PixelLayout::Rgb8);
    require_limit_exceeded(result, "lossless width=2^30+1");
}

#[test]
fn streaming_lossless_rejects_height_above_spec_max() {
    let result = LosslessConfig::new().encoder(16, MAX_JXL_DIM + 1, PixelLayout::Rgb8);
    require_limit_exceeded(result, "lossless height=2^30+1");
}

#[test]
fn streaming_lossless_rejects_u32_max() {
    let result = LosslessConfig::new().encoder(u32::MAX, 2, PixelLayout::Rgb8);
    require_limit_exceeded(result, "lossless width=u32::MAX");
}

#[test]
fn streaming_lossless_rejects_zero_dim() {
    let result = LosslessConfig::new().encoder(0, 16, PixelLayout::Rgb8);
    require_invalid_input(result, "lossless width=0");
    let result = LosslessConfig::new().encoder(16, 0, PixelLayout::Rgb8);
    require_invalid_input(result, "lossless height=0");
}
