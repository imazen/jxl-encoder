//! Integration tests for EX-J11 chunk 2 — HDR-aware loss dispatch + VDP2-lite.
//!
//! Chunk 1 shipped the dispatch framework with a stub for [`HdrLoss::Vdp2`];
//! chunk 2 (this commit) lands the actual VDP2-lite maths so both
//! [`HdrLoss::Butteraugli`] (default) and [`HdrLoss::Vdp2`] run to completion.
//!
//! Verifies:
//! 1. `HdrLoss::Butteraugli` (default) produces byte-identical output to
//!    every encode that doesn't touch `with_hdr_loss` at all — the
//!    36/36 hash-lock safety net.
//! 2. `HdrLoss::Vdp2` is opt-in and now COMPLETES the encode (was a typed
//!    error in chunk 1; chunk 2 swaps in the multi-scale CSF pyramid).
//! 3. `with_effort()` preserves `hdr_loss` across re-application.
//! 4. The HdrLoss enum has the expected API surface (Default, Copy,
//!    Debug, PartialEq, `as_str`, `is_implemented`).
//!
//! Lives behind the `butteraugli-loop` cargo feature; no-op without it.

#![cfg(feature = "butteraugli-loop")]

use jxl_encoder::{HdrLoss, LossyConfig, PixelLayout};

fn rgb8_buf(w: u32, h: u32) -> Vec<u8> {
    // Smooth gradient — fine for an API-wiring test (synthetic content
    // is only banned for quality validation, not API plumbing per
    // CLAUDE.md "No Synthetic-Only Quality Tests").
    (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect()
}

#[test]
fn default_is_auto_chunk4() {
    // Chunk 4 flipped the default from `Butteraugli` to `Auto`.
    // `Auto` resolves to `Butteraugli` on SDR content at encode entry,
    // so hash-lock fixtures stay byte-identical (covered by
    // `default_butteraugli_encode_unchanged_by_explicit_setter`
    // below); the default only differs from the explicit
    // `Butteraugli` selection on PQ / HLG content (covered by
    // `auto_resolves_to_vdp2_on_pq` in the chunk-4 dispatch tests).
    assert_eq!(HdrLoss::default(), HdrLoss::Auto);
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.hdr_loss(), HdrLoss::Auto);
}

#[test]
fn hdr_loss_enum_surface() {
    // Copy / Clone / Debug / Eq derived as expected.
    let l = HdrLoss::Vdp2;
    let l2 = l;
    assert_eq!(l, l2);
    assert!(matches!(l, HdrLoss::Vdp2));
    assert_eq!(format!("{:?}", l), "Vdp2");
    assert_eq!(HdrLoss::Butteraugli.as_str(), "butteraugli");
    assert_eq!(HdrLoss::Vdp2.as_str(), "vdp2");
    // Chunk-2 invariant: both losses are implemented.
    assert!(HdrLoss::Butteraugli.is_implemented());
    assert!(HdrLoss::Vdp2.is_implemented());
}

#[test]
fn default_butteraugli_encode_unchanged_by_explicit_setter() {
    // Hash-lock safety: explicitly setting HdrLoss::Butteraugli must
    // produce byte-identical output to never touching `with_hdr_loss`
    // at all. The default path is byte-identical to every existing
    // release.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let implicit = LossyConfig::new(1.0)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("implicit-default encode must succeed");
    let explicit = LossyConfig::new(1.0)
        .with_hdr_loss(HdrLoss::Butteraugli)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("explicit-default encode must succeed");

    assert_eq!(
        implicit, explicit,
        "explicit HdrLoss::Butteraugli must be byte-identical to the implicit default \
         (otherwise hash-locks regress)"
    );
}

#[test]
fn vdp2_without_buttloop_iters_is_silently_unused_today() {
    // The Vdp2 stub only fires INSIDE the butteraugli loop. At
    // effort 7 the default `butteraugli_iters` is 0, so the loop
    // never runs and the stub never triggers. This documents the
    // current chunk-1 behaviour — a future chunk may want to surface
    // the validation earlier (e.g., at `LossyConfig::validate`) so
    // callers see the error even when the loop wouldn't run.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let result = LossyConfig::new(1.0) // effort 7, butteraugli_iters = 0
        .with_hdr_loss(HdrLoss::Vdp2)
        .encode(&buf, w, h, PixelLayout::Rgb8);

    // At e7 the loop never runs, so Vdp2 is silently ignored. Encode
    // succeeds — this is documented chunk-1 behaviour, not a bug.
    assert!(
        result.is_ok(),
        "Vdp2 at e7 (buttloop iters=0) is silently unused — chunk 1 only \
         dispatches inside the loop. Got: {:?}",
        result.as_ref().err()
    );
}

#[test]
fn vdp2_with_buttloop_iters_completes() {
    // Chunk-2: when the buttloop actually runs (effort 8+), Vdp2 now
    // completes the encode using the VDP2-lite metric. (Chunk 1 used
    // to surface a typed error here.)
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let result = LossyConfig::new(1.0)
        .with_effort(8) // effort 8: butteraugli_iters = 2 by default
        .with_hdr_loss(HdrLoss::Vdp2)
        .encode(&buf, w, h, PixelLayout::Rgb8);

    assert!(
        result.is_ok(),
        "Vdp2 at e8 must complete now that chunk 2 has landed; got {:?}",
        result.as_ref().err()
    );
    let bytes = result.unwrap();
    // JXL signature must be present (sanity: encode actually produced
    // a JXL file rather than e.g. an empty Vec).
    assert!(
        bytes.len() > 32,
        "VDP2 encode produced suspiciously few bytes ({})",
        bytes.len()
    );
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "JXL signature missing");
}

#[test]
fn vdp2_with_hdr_intensity_target_completes() {
    // Smoke test for the HDR path: encode at a 1000-nit intensity_target
    // and confirm the VDP2-lite metric handles the larger luminance range
    // without crashing or producing degenerate output. `with_intensity_target`
    // is exposed on `EncodeRequest`, not `LossyConfig`, so we go through
    // the request layer.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let cfg = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Vdp2);
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_intensity_target(1000.0)
        .encode(&buf);

    assert!(
        result.is_ok(),
        "Vdp2 at 1000-nit intensity_target must complete; got {:?}",
        result.as_ref().err()
    );
}

#[test]
fn hdr_loss_preserved_across_with_effort() {
    // Mirrors the `butteraugli_iters_explicit` preservation pattern.
    // A caller that sets Vdp2 and then re-applies effort must keep
    // Vdp2 (otherwise the chained-builder ergonomics break).
    let cfg = LossyConfig::new(1.0)
        .with_hdr_loss(HdrLoss::Vdp2)
        .with_effort(8);
    assert_eq!(cfg.hdr_loss(), HdrLoss::Vdp2);

    // And the reverse: starting with effort then setting loss.
    let cfg = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Vdp2);
    assert_eq!(cfg.hdr_loss(), HdrLoss::Vdp2);
}

#[test]
fn hdr_loss_butteraugli_with_buttloop_iters_works() {
    // Sanity check: at effort 8 with HdrLoss::Butteraugli (default),
    // the buttloop runs as before — no regression from the
    // dispatch insertion.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let result = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Butteraugli)
        .encode(&buf, w, h, PixelLayout::Rgb8);

    assert!(
        result.is_ok(),
        "HdrLoss::Butteraugli at e8 must complete the buttloop normally; got {:?}",
        result.err()
    );
}
