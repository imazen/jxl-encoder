// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression tests for jxl-encoder#79 — internal encode/validation error
//! origins must capture their *own* location, not the API boundary's.
//!
//! Before the fix the public `Result` was already `At<EncodeError>` and the
//! entry methods were `#[track_caller]`, but the internal error origins
//! constructed bare errors with no `at!()`, so a failure deep in the
//! encode/validation path surfaced with a trace that pointed only at the
//! outermost API method (or carried no frame at all). The private
//! orchestration + validation layer in `api.rs` now returns
//! `At<EncodeError>` and constructs its errors via `at!()`, so the trace's
//! origin frame is the real site (inside `api.rs`), and propagation adds the
//! entry-method frame on top.
//!
//! These tests assert two things on a forced error:
//!   1. the trace carries at least one location frame (it is no longer empty),
//!   2. at least one frame's file is in the encoder's api layer (`api.rs` or
//!      its `api/*.rs` submodules) — i.e. the origin was captured inside the
//!      library, not only at this test's call site.

use jxl_encoder::{At, EncodeError, LossyConfig, PixelLayout};

/// True when any frame in the trace was captured inside the encoder's api
/// layer — `api.rs` or its `api/*.rs` submodules (the origin/validation
/// layer) — as opposed to only this test file (the API caller).
fn has_frame_in_api_layer(at: &At<EncodeError>) -> bool {
    at.frames().any(|f| {
        f.location()
            .map(|loc| {
                // The api layer spans `api.rs` plus its `api/*.rs` submodules
                // (validate / ingest / animate / … extracted 2026-07); an
                // origin captured in any of them counts as "the api layer".
                let file = loc.file().replace('\\', "/");
                file.ends_with("src/api.rs") || file.contains("src/api/")
            })
            .unwrap_or(false)
    })
}

/// True when any frame was captured in this test file — used to confirm the
/// trace genuinely spans from the origin up to the caller, not that we are
/// merely reading the caller's location.
fn has_frame_in_this_test(at: &At<EncodeError>) -> bool {
    at.frames().any(|f| {
        f.location()
            .map(|loc| {
                loc.file()
                    .replace('\\', "/")
                    .ends_with("error_location_trace.rs")
            })
            .unwrap_or(false)
    })
}

fn dump(at: &At<EncodeError>) -> String {
    let mut s = format!("frame_count={}\n", at.frame_count());
    for (i, f) in at.frames().enumerate() {
        match f.location() {
            Some(loc) => s.push_str(&format!(
                "  [{i}] {}:{}:{}\n",
                loc.file(),
                loc.line(),
                loc.column()
            )),
            None => s.push_str(&format!("  [{i}] <no location>\n")),
        }
    }
    s
}

#[test]
fn pixel_buffer_mismatch_origin_is_captured_in_api() {
    // Wrong-size buffer: the origin is `validate_pixels` deep inside the
    // one-shot encode path, far below the `encode()` entry method.
    let too_small = [0u8; 8]; // 4x4 RGB8 needs 48 bytes
    let err: At<EncodeError> = LossyConfig::new(1.0)
        .encode_request(4, 4, PixelLayout::Rgb8)
        .encode(&too_small)
        .expect_err("4x4 RGB8 with an 8-byte buffer must fail");

    assert!(
        matches!(err.error(), EncodeError::InvalidInput { .. }),
        "expected InvalidInput, got {:?}",
        err.error()
    );
    // The whole point of #79: the trace is non-empty AND points at the
    // real origin site inside api.rs, not just this call site.
    assert!(
        err.frame_count() >= 1,
        "trace should carry at least one location frame; got:\n{}",
        dump(&err)
    );
    assert!(
        has_frame_in_api_layer(&err),
        "origin frame should be inside the encoder's api.rs (the real failure \
         site), not only this test's call site; got:\n{}",
        dump(&err)
    );
}

#[test]
fn deep_failure_trace_spans_origin_and_entry() {
    // Same forced failure, but assert the trace spans from the internal
    // origin up through the `#[track_caller]` entry method to this caller —
    // i.e. more than a single boundary frame.
    let too_small = [0u8; 8];
    let err: At<EncodeError> = LossyConfig::new(1.0)
        .encode_request(4, 4, PixelLayout::Rgb8)
        .encode(&too_small)
        .expect_err("must fail");

    assert!(
        has_frame_in_api_layer(&err),
        "missing the internal origin frame; got:\n{}",
        dump(&err)
    );
    assert!(
        has_frame_in_this_test(&err),
        "missing the caller frame (entry method should propagate the trace up \
         to this `encode()` call); got:\n{}",
        dump(&err)
    );
    // Origin + entry => at least two distinct frames.
    assert!(
        err.frame_count() >= 2,
        "deep failure should produce >= 2 frames (origin + entry), got {}:\n{}",
        err.frame_count(),
        dump(&err)
    );
}

#[test]
fn tone_mapping_origin_is_captured_in_api() {
    // A non-finite intensity_target is rejected by `validate_tone_mapping_full`,
    // a leaf validator called from `encode_inner` — another deep origin.
    let pixels = [0u8; 48]; // valid 4x4 RGB8
    let err: At<EncodeError> = LossyConfig::new(1.0)
        .encode_request(4, 4, PixelLayout::Rgb8)
        .with_intensity_target(f32::NAN)
        .encode(&pixels)
        .expect_err("NaN intensity_target must be rejected");

    assert!(
        matches!(err.error(), EncodeError::InvalidInput { .. }),
        "expected InvalidInput, got {:?}",
        err.error()
    );
    assert!(
        has_frame_in_api_layer(&err) && err.frame_count() >= 1,
        "tone-mapping validation origin should be captured in the api layer (api.rs or api/*.rs); got:\n{}",
        dump(&err)
    );
}
