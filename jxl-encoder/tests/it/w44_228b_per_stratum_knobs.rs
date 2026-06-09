// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-228b: per-stratum optimal `Tier2Knobs` lookup integration tests.
//!
//! Covers the OPT-IN API surface added in W44-228b:
//!
//! - [`Tier2Knobs::default_for_stratum`] — every `ContentStratum` variant
//!   returns a valid `Tier2Knobs` that round-trips through
//!   `expand_to_runtime_tuning()` and through `install_or_check_idempotent`
//!   against itself (idempotent).
//! - [`Tier2Knobs::auto_for_distance`] — for `ImageContentClass::Unknown`
//!   it returns the universal default knobs (the no-op path); for
//!   `Photo`/`Screenshot` it lands on the right stratum at exact band
//!   boundaries (`1.0`, `2.0`, `3.5` per the W44-217 / W44-228a convention).
//! - Encode/decode round-trip with `.with_knobs(auto_for_distance(...))`
//!   succeeds and decodes via jxl-oxide.
//! - Default path (no `.with_knobs(...)` at all) is byte-identical to
//!   `.with_knobs(Tier2Knobs::default())` — proves the opt-in API does
//!   NOT perturb the no-opt-in default path.
//!
//! Because `runtime::install` is single-shot per process, the tests that
//! depend on installed/not-installed state run inside ONE
//! `#[test]` function. The pure-data tests (no encode) run in their own
//! `#[test]` functions safely.

#![cfg(feature = "tuning-override")]

use jxl_encoder::effort::ImageContentClass;
use jxl_encoder::tuning::coupling::{ContentStratum, Tier2Knobs};
use jxl_encoder::{LossyConfig, PixelLayout};

// ─── Pure-data tests (no runtime::install reached — safe to parallelize) ───

/// (1) Every `ContentStratum` variant: `default_for_stratum(s)` produces a
/// `Tier2Knobs` whose `expand_to_runtime_tuning()` succeeds without panic.
/// Every expanded `RuntimeTuning` field is finite — no NaN propagation
/// from the table tuples through the ridge math.
///
/// Note: the "round-trip through `install_or_check_idempotent` against
/// itself" half of acceptance gate (j)/(1) is exercised by the combined
/// encode test below, which calls `LossyConfig::with_knobs(...)` →
/// `install_or_check_idempotent(...)` per-encode. We do NOT call install
/// from this pure-data test because `runtime::install` is single-shot
/// per process; a parallel-test ordering race would corrupt the
/// combined encode test that picks up the install path naturally.
#[test]
fn test_w44_228b_strata_round_trip_default() {
    let strata = [
        ContentStratum::ScreenVeryHigh,
        ContentStratum::ScreenHigh,
        ContentStratum::ScreenMid,
        ContentStratum::ScreenLow,
        ContentStratum::PhotoVeryHigh,
        ContentStratum::PhotoHigh,
        ContentStratum::PhotoMid,
        ContentStratum::PhotoLow,
    ];
    for s in strata {
        let k = Tier2Knobs::default_for_stratum(s);
        let rt = k.expand_to_runtime_tuning();
        assert!(
            rt.smart_zenjxl_photo_mask_p25_min.is_finite(),
            "{:?}: smart_zenjxl_photo_mask_p25_min not finite",
            s
        );
        assert!(
            rt.screenshot_median_threshold.is_finite(),
            "{:?}: screenshot_median_threshold not finite",
            s
        );
        assert!(
            rt.buttloop_default_screenshot_qf_seed_scale.is_finite(),
            "{:?}: buttloop_default_screenshot_qf_seed_scale not finite",
            s
        );
        assert!(
            rt.buttloop_qf_seed_scale_min_distance.is_finite(),
            "{:?}: buttloop_qf_seed_scale_min_distance not finite",
            s
        );
        assert!(
            rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6.is_finite(),
            "{:?}: adaptive_quant_screenshot_qf_seed_scale_e5_e6 not finite",
            s
        );
        assert!(
            rt.adaptive_quant_screenshot_qf_seed_scale_e7.is_finite(),
            "{:?}: adaptive_quant_screenshot_qf_seed_scale_e7 not finite",
            s
        );
        // Sanity: physical floors hold for every per-stratum tuple.
        assert!(rt.smart_zenjxl_photo_mask_p25_min >= 0.0);
        assert!(rt.screenshot_median_threshold >= 0.0);
        assert!(rt.buttloop_default_screenshot_qf_seed_scale >= 0.0);
        assert!(rt.buttloop_qf_seed_scale_min_distance >= 1.5);
        assert!(rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6 >= 0.0);
        assert!(rt.adaptive_quant_screenshot_qf_seed_scale_e7 >= 0.0);
    }
}

/// (2) `auto_for_distance(Unknown, d)` returns the universal default knobs
/// for every distance in {0.5, 2.0, 4.0, 6.0}. This is the no-op path
/// callers can use unconditionally (e.g. when the auto-classifier
/// returns `Unknown` for streaming / animation paths).
#[test]
fn test_w44_228b_unknown_class_uses_default() {
    for d in [0.5_f32, 2.0, 4.0, 6.0] {
        let k = Tier2Knobs::auto_for_distance(ImageContentClass::Unknown, d);
        assert_eq!(
            k,
            Tier2Knobs::default(),
            "auto_for_distance(Unknown, {}) should return default knobs",
            d
        );
    }
    // Other (mixed) class also has no per-stratum entry (W44-228a corpus
    // only covered Photo + Screenshot). Should fall back to default knobs.
    for d in [0.5_f32, 2.0, 4.0, 6.0] {
        let k = Tier2Knobs::auto_for_distance(ImageContentClass::Other, d);
        assert_eq!(
            k,
            Tier2Knobs::default(),
            "auto_for_distance(Other, {}) should fall back to default knobs",
            d
        );
    }
}

/// (3) Exact band boundaries: at `d = 1.0`, `d = 2.0`, `d = 3.5` the
/// `from_distance_band` half-open intervals produce the higher band.
/// Convention (W44-217 / W44-228a, see `ContentStratum` doc):
/// `low: d < 1.0`, `mid: [1.0, 2.0)`, `high: [2.0, 3.5)`, `very_high: d >= 3.5`.
///
/// This pins the binning convention; if a future re-derivation moves the
/// boundaries the test must be updated alongside the TSV.
#[test]
fn test_w44_228b_stratum_boundaries() {
    let s = |c, d| ContentStratum::from_distance_band(c, d);
    // Screenshot side
    assert_eq!(
        s(ImageContentClass::Screenshot, 0.999),
        Some(ContentStratum::ScreenLow)
    );
    assert_eq!(
        s(ImageContentClass::Screenshot, 1.0),
        Some(ContentStratum::ScreenMid)
    );
    assert_eq!(
        s(ImageContentClass::Screenshot, 1.999),
        Some(ContentStratum::ScreenMid)
    );
    assert_eq!(
        s(ImageContentClass::Screenshot, 2.0),
        Some(ContentStratum::ScreenHigh)
    );
    assert_eq!(
        s(ImageContentClass::Screenshot, 3.499),
        Some(ContentStratum::ScreenHigh)
    );
    assert_eq!(
        s(ImageContentClass::Screenshot, 3.5),
        Some(ContentStratum::ScreenVeryHigh)
    );
    assert_eq!(
        s(ImageContentClass::Screenshot, 10.0),
        Some(ContentStratum::ScreenVeryHigh)
    );
    // Photo side mirrors
    assert_eq!(
        s(ImageContentClass::Photo, 0.999),
        Some(ContentStratum::PhotoLow)
    );
    assert_eq!(
        s(ImageContentClass::Photo, 1.0),
        Some(ContentStratum::PhotoMid)
    );
    assert_eq!(
        s(ImageContentClass::Photo, 2.0),
        Some(ContentStratum::PhotoHigh)
    );
    assert_eq!(
        s(ImageContentClass::Photo, 3.5),
        Some(ContentStratum::PhotoVeryHigh)
    );
    // Unknown / Other → None
    assert_eq!(s(ImageContentClass::Unknown, 2.0), None);
    assert_eq!(s(ImageContentClass::Other, 2.0), None);
    // NaN / negative distance falls into `low` (clamped to 0)
    assert_eq!(
        s(ImageContentClass::Screenshot, f32::NAN),
        Some(ContentStratum::ScreenLow)
    );
    assert_eq!(
        s(ImageContentClass::Photo, -1.0),
        Some(ContentStratum::PhotoLow)
    );
}

// ─── Encode-driven tests (single-shot install — keep in ONE function) ───

/// Synthetic 64×64 photo-class gradient (low fcbr, smooth) — small
/// enough to keep encode fast, contains no patches / screenshot
/// patterns, falls into `Photo / mid` at d=2.0.
fn synth_photo_64() -> (Vec<u8>, u32, u32) {
    let (w, h) = (64u32, 64u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 4).min(255)) as u8;
            let g = ((y * 4).min(255)) as u8;
            let b = (((x + y) * 2).min(255)) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    (pixels, w, h)
}

fn encode(cfg: LossyConfig, rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed")
}

/// Combined test that satisfies both:
///
/// - `test_w44_228b_synthetic_fixture_opt_in_round_trip` (acceptance gate (4))
///   — encode a 64×64 synthetic gradient with
///   `.with_knobs(Tier2Knobs::auto_for_distance(Photo, 2.0))`; confirm
///   the byte-stream decodes via jxl-oxide.
/// - `test_w44_228b_default_path_byte_identical` (acceptance gate (5))
///   — confirm encoding with `.with_knobs(Tier2Knobs::default())`
///   produces byte-for-byte identical output to encoding without
///   `.with_knobs(...)` at all.
///
/// These MUST run in ONE `#[test]` function because both touch the
/// process-wide `runtime::install` OnceLock. We sequence them so the
/// no-opt-in baseline is captured BEFORE any install happens, then we
/// verify default knobs ≡ no knobs (acceptance (5)), then verify
/// `auto_for_distance` produces an encode that decodes cleanly
/// (acceptance (4)). Same single-shot pattern as
/// `tests/w44_222_with_knobs_builder.rs::w44_222_with_knobs_builder_full_sequence`.
///
/// Sibling-binary isolation: this test lives in its own integration test
/// binary (`tests/w44_228b_per_stratum_knobs.rs`). `cargo test` builds it
/// as a separate binary from `tests/w44_222_with_knobs_builder.rs`, so
/// the OnceLock state doesn't collide between the two test binaries.
#[test]
fn test_w44_228b_synthetic_fixture_opt_in_round_trip_and_default_byte_identical() {
    let (rgb, w, h) = synth_photo_64();

    // ── (5) Default path byte-identical: no `.with_knobs()` vs
    //       `.with_knobs(Tier2Knobs::default())` MUST produce the same
    //       bytes. Proves the opt-in API doesn't change the no-opt-in
    //       default path (preserves the hash-lock contract).
    //
    // Must run BEFORE the encode in (4) below, because (4) installs an
    // override that affects subsequent encodes (single-shot OnceLock).
    let cfg_no_knobs = LossyConfig::new(2.0).with_effort(5);
    let bytes_no_knobs = encode(cfg_no_knobs, &rgb, w, h);

    let cfg_default_knobs = LossyConfig::new(2.0)
        .with_effort(5)
        .with_knobs(Tier2Knobs::default());
    let bytes_default_knobs = encode(cfg_default_knobs, &rgb, w, h);

    assert_eq!(
        bytes_no_knobs.len(),
        bytes_default_knobs.len(),
        "(5) default-knobs encode should be byte-length-identical to no-knobs encode"
    );
    assert_eq!(
        bytes_no_knobs, bytes_default_knobs,
        "(5) default-knobs encode should be byte-for-byte-identical to no-knobs encode"
    );

    // ── (4) Opt-in synthetic-fixture round-trip: encode with
    //       `.with_knobs(Tier2Knobs::auto_for_distance(Photo, 2.0))`
    //       and confirm the bytes are non-empty AND decode cleanly via
    //       jxl-oxide. We deliberately do NOT assert byte-equality
    //       against the default — the whole point of the API is to
    //       CHANGE bytes (Photo/mid optimum has non-default values for
    //       k1/k3/k4/k5).
    let knobs_pm = Tier2Knobs::auto_for_distance(ImageContentClass::Photo, 2.0);
    // Sanity: knobs are non-default (otherwise we're not exercising the
    // install path).
    assert_ne!(
        knobs_pm,
        Tier2Knobs::default(),
        "Photo/mid optimum should differ from the universal default"
    );

    let cfg_optin = LossyConfig::new(2.0).with_effort(5).with_knobs(knobs_pm);
    let bytes_optin = encode(cfg_optin, &rgb, w, h);
    assert!(
        !bytes_optin.is_empty(),
        "opt-in encode should produce non-empty bytes"
    );

    // Multi-decoder roundtrip via jxl-oxide. djxl + jxl-rs are NOT
    // invoked from this test (no compiled-in binding); CI runs the
    // CLI roundtrip via the broader test harness.
    let _ = decode_oxide(&bytes_optin, w, h);
}

/// Helper: decode JXL bytes via jxl-oxide and confirm the decoded
/// dimensions match the encoded width/height.
fn decode_oxide(jxl_bytes: &[u8], expected_w: u32, expected_h: u32) -> Vec<u8> {
    let reader = std::io::Cursor::new(jxl_bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide read failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let header = image.image_header();
    assert_eq!(header.size.width, expected_w, "decoded width mismatch");
    assert_eq!(header.size.height, expected_h, "decoded height mismatch");
    let render = image.render_frame(0).expect("render_frame failed");
    let stream = render.stream();
    let channels = stream.channels() as usize;
    let mut out = vec![0.0_f32; (expected_w as usize) * (expected_h as usize) * channels];
    let mut s = stream;
    s.write_to_buffer(&mut out);
    // Spot-check: the buffer is the right length.
    assert_eq!(
        out.len(),
        (expected_w as usize) * (expected_h as usize) * channels
    );
    out.iter()
        .map(|f| (f.clamp(0.0, 1.0) * 255.0) as u8)
        .collect()
}
