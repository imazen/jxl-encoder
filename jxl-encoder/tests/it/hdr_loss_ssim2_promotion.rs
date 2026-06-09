//! Integration tests for W43-3 chunk 1 — promote `HdrLoss::Ssim2` to a
//! first-class variant alongside [`HdrLoss::Butteraugli`] and
//! [`HdrLoss::Vdp2`].
//!
//! The `ssim2-loop` cargo feature has wired
//! [`crate::vardct::encoder::VarDctEncoder::ssim2_refine_quant_field`]
//! internally for several releases. Chunk 1 exposes that path through
//! the public [`HdrLoss`] enum so callers can opt in via a single
//! [`LossyConfig::with_hdr_loss`] call — no fiddling with the legacy
//! `with_ssim2_iters` setter required.
//!
//! Verifies:
//! 1. `HdrLoss::Ssim2` is a first-class variant (Default-not, Copy,
//!    Debug, PartialEq, `as_str()`, `is_implemented()`).
//! 2. [`HdrLoss::Auto`] still resolves to [`HdrLoss::Butteraugli`] on
//!    SDR (no default flip — hash-lock corpus stays byte-identical).
//! 3. With the `ssim2-loop` cargo feature, encoding with
//!    `HdrLoss::Ssim2` at effort ≥ 8 (where buttloop iters > 0)
//!    completes and produces a valid JXL bitstream.
//! 4. Encoding with `HdrLoss::Ssim2` produces **different bytes** from
//!    `HdrLoss::Butteraugli` at the same distance / effort — proves the
//!    dispatch actually routes to the ssim2 loop.
//! 5. Without the `ssim2-loop` feature, selecting `HdrLoss::Ssim2`
//!    surfaces a typed `Error::NotImplemented` (no silent fallback).
//! 6. `with_effort()` preserves `hdr_loss = Ssim2` across re-application.
//!
//! Lives behind the `butteraugli-loop` cargo feature (the enum lives in
//! the same module). Tests 3 + 4 additionally require `ssim2-loop`.

#![cfg(feature = "butteraugli-loop")]

use jxl_encoder::{HdrLoss, LossyConfig, PixelLayout};

fn rgb8_buf(w: u32, h: u32) -> Vec<u8> {
    // Smooth gradient — API-wiring tests only. CLAUDE.md's
    // "No Synthetic-Only Quality Tests" rule applies to *quality*
    // validation, not dispatch plumbing; the proof-by-tests harness
    // for the metric itself lives in `just quality-compare`.
    (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect()
}

#[test]
fn ssim2_enum_surface() {
    // Copy / Clone / Debug / Eq derived as expected — same shape as the
    // existing Vdp2 test.
    let l = HdrLoss::Ssim2;
    let l2 = l;
    assert_eq!(l, l2);
    assert!(matches!(l, HdrLoss::Ssim2));
    assert_eq!(format!("{:?}", l), "Ssim2");
    assert_eq!(HdrLoss::Ssim2.as_str(), "ssim2");
    assert!(HdrLoss::Ssim2.is_implemented());
}

#[test]
fn ssim2_is_distinct_from_butteraugli_and_vdp2() {
    assert_ne!(HdrLoss::Ssim2, HdrLoss::Butteraugli);
    assert_ne!(HdrLoss::Ssim2, HdrLoss::Vdp2);
    assert_ne!(HdrLoss::Ssim2, HdrLoss::Auto);
}

#[test]
fn ssim2_passes_through_resolve() {
    // `HdrLoss::Ssim2`, like `Butteraugli` and `Vdp2`, is a "pin"
    // variant — `resolve()` returns it unchanged regardless of the
    // transfer function. Only `Auto` is mutated by `resolve()`.
    use jxl_encoder::TransferFunction;
    for tf in [
        None,
        Some(TransferFunction::Srgb),
        Some(TransferFunction::Bt709),
        Some(TransferFunction::Pq),
        Some(TransferFunction::Hlg),
    ] {
        assert_eq!(HdrLoss::Ssim2.resolve(tf), HdrLoss::Ssim2);
    }
}

#[test]
fn auto_still_resolves_sdr_to_butteraugli_no_default_flip() {
    // **Critical hash-lock invariant.** W43-3 chunk 1 explicitly does
    // NOT flip the default — `HdrLoss::Auto` on SDR content still
    // returns `Butteraugli`. The chunk-2 plan covers the A.9
    // decisive-rule eval that would justify a default flip; until
    // then, the 36/36 hash-lock corpus stays byte-identical.
    use jxl_encoder::TransferFunction;
    assert_eq!(HdrLoss::Auto.resolve(None), HdrLoss::Butteraugli);
    assert_eq!(
        HdrLoss::Auto.resolve(Some(TransferFunction::Srgb)),
        HdrLoss::Butteraugli
    );
    assert_eq!(
        HdrLoss::Auto.resolve(Some(TransferFunction::Bt709)),
        HdrLoss::Butteraugli
    );
    // SDR → never `Ssim2` (unless caller explicitly pins it).
    assert_ne!(HdrLoss::Auto.resolve(None), HdrLoss::Ssim2);
}

#[test]
fn ssim2_preserved_across_with_effort() {
    // Mirrors the `hdr_loss_preserved_across_with_effort` test for
    // `Vdp2`. A caller that sets Ssim2 and then re-applies effort must
    // keep Ssim2 (otherwise the chained-builder ergonomics break).
    let cfg = LossyConfig::new(1.0)
        .with_hdr_loss(HdrLoss::Ssim2)
        .with_effort(8);
    assert_eq!(cfg.hdr_loss(), HdrLoss::Ssim2);

    let cfg = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Ssim2);
    assert_eq!(cfg.hdr_loss(), HdrLoss::Ssim2);
}

#[test]
fn default_butteraugli_encode_unchanged_by_explicit_setter_at_e8() {
    // Hash-lock safety at effort 8 (where the buttloop runs):
    // explicit `HdrLoss::Butteraugli` must be byte-identical to the
    // implicit default. Covers the dispatch's "non-Ssim2" branch in
    // `vardct/encoder.rs` to make sure the W43-3 insertion didn't
    // regress the existing Butteraugli path.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let implicit = LossyConfig::new(1.0)
        .with_effort(8)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("implicit-default e8 encode must succeed");
    let explicit = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Butteraugli)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("explicit-Butteraugli e8 encode must succeed");

    assert_eq!(
        implicit, explicit,
        "explicit HdrLoss::Butteraugli at e8 must be byte-identical to the implicit default \
         (otherwise the W43-3 chunk 1 dispatch insertion regressed the existing path)"
    );
}

#[cfg(feature = "ssim2-loop")]
#[test]
fn ssim2_at_e8_completes() {
    // Smoke: with the `ssim2-loop` feature, encoding with
    // `HdrLoss::Ssim2` at effort 8 (where butteraugli_iters > 0)
    // completes and produces a valid JXL file.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let result = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Ssim2)
        .encode(&buf, w, h, PixelLayout::Rgb8);

    assert!(
        result.is_ok(),
        "HdrLoss::Ssim2 at e8 must complete with the ssim2-loop feature on; got {:?}",
        result.as_ref().err()
    );
    let bytes = result.unwrap();
    assert!(
        bytes.len() > 32,
        "Ssim2 encode produced suspiciously few bytes ({})",
        bytes.len()
    );
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "JXL signature missing");
}

#[cfg(feature = "ssim2-loop")]
#[test]
fn ssim2_bytes_differ_from_butteraugli_proves_dispatch_works() {
    // **The load-bearing chunk-1 test.** If the dispatch is wired
    // correctly, encoding with `HdrLoss::Ssim2` at effort 8 must
    // produce different bytes than encoding with `HdrLoss::Butteraugli`
    // at the same distance / effort. Equality would mean the dispatch
    // is silently falling through to the buttloop.
    //
    // Uses a non-trivial image (slightly more complex gradient + some
    // mid-gray variance) so the two loops diverge at the rounding-
    // sensitive boundaries.
    let w = 64u32;
    let h = 64u32;
    let buf: Vec<u8> = (0..(w * h * 3) as usize)
        .map(|i| {
            // Slightly more interesting than `i % 256` — adds a low-freq
            // wave so the two metrics see different per-block
            // distortion profiles.
            let x = (i / 3) % (w as usize);
            let y = (i / 3) / (w as usize);
            let chan = i % 3;
            let v = (x.wrapping_mul(3) ^ y.wrapping_mul(5) ^ chan.wrapping_mul(17)) & 0xFF;
            v as u8
        })
        .collect();

    let butteraugli_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Butteraugli)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("Butteraugli encode must succeed");
    let ssim2_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Ssim2)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("Ssim2 encode must succeed");

    assert!(
        butteraugli_bytes.len() > 32,
        "Butteraugli encode produced suspiciously few bytes"
    );
    assert!(
        ssim2_bytes.len() > 32,
        "Ssim2 encode produced suspiciously few bytes"
    );
    assert_ne!(
        butteraugli_bytes, ssim2_bytes,
        "HdrLoss::Ssim2 and HdrLoss::Butteraugli produced byte-identical output at e8 — \
         the W43-3 chunk 1 dispatch is silently falling through to the buttloop. \
         Check `vardct/encoder.rs` for the `take_ssim2_path` branch."
    );
}

#[cfg(feature = "ssim2-loop")]
#[test]
fn ssim2_at_e7_silently_unused_today() {
    // At effort 7 the default `butteraugli_iters` is 0, so the buttloop
    // never runs and the W43-3 dispatch never fires. This documents the
    // current chunk-1 behaviour — selecting `HdrLoss::Ssim2` at e7 is a
    // no-op, identical to `HdrLoss::Butteraugli` at e7 (no loop runs in
    // either case).
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let butteraugli_e7 = LossyConfig::new(1.0)
        .with_hdr_loss(HdrLoss::Butteraugli)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("Butteraugli at e7 must succeed");
    let ssim2_e7 = LossyConfig::new(1.0)
        .with_hdr_loss(HdrLoss::Ssim2)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("Ssim2 at e7 must succeed");

    assert_eq!(
        butteraugli_e7, ssim2_e7,
        "At e7 (buttloop disabled), HdrLoss::Ssim2 and HdrLoss::Butteraugli must be \
         byte-identical — the dispatch never fires. If this regressed, the chunk-1 path \
         is leaking into the no-buttloop encode."
    );
}

#[cfg(not(feature = "ssim2-loop"))]
#[test]
fn ssim2_without_feature_surfaces_typed_error_at_e8() {
    // Without the `ssim2-loop` cargo feature, selecting
    // `HdrLoss::Ssim2` at effort 8 (where the buttloop runs) must
    // surface a typed `Error::NotImplemented` instead of silently
    // falling back to butteraugli. See
    // `validate_loss` + `HdrMetricError::Ssim2FeatureDisabled`.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let result = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Ssim2)
        .encode(&buf, w, h, PixelLayout::Rgb8);

    assert!(
        result.is_err(),
        "HdrLoss::Ssim2 at e8 without `ssim2-loop` feature must error, not silently fall back"
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("ssim2-loop") || err.contains("Ssim2"),
        "Error message should mention the missing feature or the variant: {err}"
    );
}
