// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-1 integration tests for `Buffering`.
//!
//! This is the streaming refactor scaffolding chunk for jxl-encoder#11
//! (mirrors libjxl PRs #4634 + #4635 + #4637 + #4642 + #4728). The
//! enum + the `with_buffering` builders + the `--buffering` CLI flag
//! exist; **no dispatch is wired yet**. These tests pin:
//!
//! 1. `Buffering::Auto` is the default on both config types.
//! 2. Every variant survives a roundtrip through the builder + getter.
//! 3. `with_buffering` is order-independent with `with_effort` (the
//!    value carries across the effort-reset).
//! 4. `Buffering::from_i8` / `to_i8` match the libjxl `-1..=3`
//!    encoding exactly, with out-of-range values folding to `Auto`.
//! 5. `Buffering::resolve_for` mirrors libjxl's 2048² threshold:
//!    ≤ 2048×2048 → `FullBuffered`; larger → `BufferedOutput`. Non-
//!    `Auto` variants pass through unchanged.
//!
//! These tests intentionally do **not** assert anything about encoder
//! bytes — bitstream invariance under the `--buffering` knob is
//! covered by the existing `corpus_regression` / `hash_lock` tests
//! (which sweep their defaults with no `with_buffering` call and
//! therefore exercise `Buffering::Auto`). When chunks 2-7 land they
//! will add `tests/buffering_dispatch.rs` covering the actual
//! per-DC-group split paths.

use jxl_encoder::{Buffering, LosslessConfig, LossyConfig};

#[test]
fn buffering_default_is_auto_on_lossy() {
    assert_eq!(LossyConfig::new(1.0).buffering(), Buffering::Auto);
}

#[test]
fn buffering_default_is_auto_on_lossless() {
    assert_eq!(LosslessConfig::new().buffering(), Buffering::Auto);
}

#[test]
fn buffering_default_via_default_impl_is_auto() {
    // `LosslessConfig::default()` should also resolve to Auto so a
    // bare `LosslessConfig::default().buffering()` is predictable.
    assert_eq!(LosslessConfig::default().buffering(), Buffering::Auto);
}

#[test]
fn buffering_every_variant_roundtrips_on_lossy() {
    for variant in [
        Buffering::Auto,
        Buffering::FullBuffered,
        Buffering::Threshold2048,
        Buffering::BufferedOutput,
        Buffering::FullStreaming,
    ] {
        let cfg = LossyConfig::new(1.0).with_buffering(variant);
        assert_eq!(cfg.buffering(), variant);
    }
}

#[test]
fn buffering_every_variant_roundtrips_on_lossless() {
    for variant in [
        Buffering::Auto,
        Buffering::FullBuffered,
        Buffering::Threshold2048,
        Buffering::BufferedOutput,
        Buffering::FullStreaming,
    ] {
        let cfg = LosslessConfig::new().with_buffering(variant);
        assert_eq!(cfg.buffering(), variant);
    }
}

#[test]
fn buffering_carries_across_with_effort_on_lossy() {
    // The buffering knob is a caller preference, never effort-derived
    // — `with_effort` must not clobber an earlier `with_buffering`.
    let cfg = LossyConfig::new(1.0)
        .with_buffering(Buffering::BufferedOutput)
        .with_effort(5);
    assert_eq!(cfg.buffering(), Buffering::BufferedOutput);

    // Reverse order: same result.
    let cfg = LossyConfig::new(1.0)
        .with_effort(9)
        .with_buffering(Buffering::FullStreaming);
    assert_eq!(cfg.buffering(), Buffering::FullStreaming);

    // Auto stays Auto.
    let cfg = LossyConfig::new(1.0)
        .with_buffering(Buffering::Auto)
        .with_effort(3);
    assert_eq!(cfg.buffering(), Buffering::Auto);
}

#[test]
fn buffering_carries_across_with_effort_on_lossless() {
    let cfg = LosslessConfig::new()
        .with_buffering(Buffering::Threshold2048)
        .with_effort(9);
    assert_eq!(cfg.buffering(), Buffering::Threshold2048);

    let cfg = LosslessConfig::new()
        .with_effort(7)
        .with_buffering(Buffering::FullBuffered);
    assert_eq!(cfg.buffering(), Buffering::FullBuffered);
}

#[test]
fn buffering_from_i8_matches_libjxl_encoding() {
    assert_eq!(Buffering::from_i8(-1), Buffering::Auto);
    assert_eq!(Buffering::from_i8(0), Buffering::FullBuffered);
    assert_eq!(Buffering::from_i8(1), Buffering::Threshold2048);
    assert_eq!(Buffering::from_i8(2), Buffering::BufferedOutput);
    assert_eq!(Buffering::from_i8(3), Buffering::FullStreaming);

    // Out-of-range folds to Auto (matches libjxl's
    // JXL_ENC_FRAME_SETTING_BUFFERING defaulting behaviour).
    assert_eq!(Buffering::from_i8(-2), Buffering::Auto);
    assert_eq!(Buffering::from_i8(4), Buffering::Auto);
    assert_eq!(Buffering::from_i8(i8::MIN), Buffering::Auto);
    assert_eq!(Buffering::from_i8(i8::MAX), Buffering::Auto);
}

#[test]
fn buffering_to_i8_matches_libjxl_encoding() {
    assert_eq!(Buffering::Auto.to_i8(), -1);
    assert_eq!(Buffering::FullBuffered.to_i8(), 0);
    assert_eq!(Buffering::Threshold2048.to_i8(), 1);
    assert_eq!(Buffering::BufferedOutput.to_i8(), 2);
    assert_eq!(Buffering::FullStreaming.to_i8(), 3);
}

#[test]
fn buffering_i8_roundtrips_all_valid_values() {
    for v in -1i8..=3 {
        let parsed = Buffering::from_i8(v);
        assert_eq!(parsed.to_i8(), v, "from_i8({v}).to_i8() must roundtrip");
    }
}

#[test]
fn buffering_resolve_for_under_threshold_picks_full_buffered() {
    // libjxl `CanDoStreamingEncoding` threshold: 2048×2048 is exactly
    // one DC group. Anything ≤ that resolves to FullBuffered (no win
    // from streaming a single-DC-group image).
    assert_eq!(
        Buffering::Auto.resolve_for(1, 1),
        Buffering::FullBuffered,
        "1x1 fits in one DC group → FullBuffered"
    );
    assert_eq!(
        Buffering::Auto.resolve_for(2048, 2048),
        Buffering::FullBuffered,
        "exactly 2048² (one DC group) → FullBuffered"
    );
    // 1024×4096 = 2048² pixels — still one-DC-group's worth, even
    // though dimensions exceed 2048 on one axis.
    assert_eq!(
        Buffering::Auto.resolve_for(1024, 4096),
        Buffering::FullBuffered,
        "1024x4096 (≤ 2048² pixels) → FullBuffered"
    );
}

#[test]
fn buffering_resolve_for_above_threshold_picks_buffered_output() {
    assert_eq!(
        Buffering::Auto.resolve_for(2049, 2048),
        Buffering::BufferedOutput,
        "2049x2048 (> 2048² pixels) → BufferedOutput"
    );
    assert_eq!(
        Buffering::Auto.resolve_for(4096, 4096),
        Buffering::BufferedOutput,
        "4Kx4K (> 2048² pixels) → BufferedOutput"
    );
    assert_eq!(
        Buffering::Auto.resolve_for(8192, 8192),
        Buffering::BufferedOutput,
        "8Kx8K (> 2048² pixels) → BufferedOutput"
    );
}

#[test]
fn buffering_resolve_for_passes_through_concrete_variants() {
    // Non-Auto variants resolve to themselves regardless of size.
    for variant in [
        Buffering::FullBuffered,
        Buffering::Threshold2048,
        Buffering::BufferedOutput,
        Buffering::FullStreaming,
    ] {
        assert_eq!(
            variant.resolve_for(1, 1),
            variant,
            "{variant:?} should pass through unchanged on tiny image"
        );
        assert_eq!(
            variant.resolve_for(8192, 8192),
            variant,
            "{variant:?} should pass through unchanged on 8K image"
        );
    }
}

#[test]
fn buffering_resolve_for_handles_giant_dimensions_without_overflow() {
    // u32::MAX squared overflows u32 but the resolver uses u64
    // arithmetic with saturating_mul. Should not panic, and must
    // still resolve to BufferedOutput.
    assert_eq!(
        Buffering::Auto.resolve_for(u32::MAX, u32::MAX),
        Buffering::BufferedOutput,
        "saturating arithmetic must not panic on 4G x 4G dimensions"
    );
    assert_eq!(
        Buffering::Auto.resolve_for(u32::MAX, 1),
        Buffering::BufferedOutput,
        "4G x 1 (>> 2048² pixels) → BufferedOutput"
    );
}

#[test]
fn buffering_enum_is_copy_and_debug() {
    // Same shape as the rest of our config-shared knob enums
    // (`ContainerMode`, `PremultipliedAlphaMode`, ...). Useful for
    // sweep harnesses that want to iterate / format / compare values.
    fn assert_copy<T: Copy>() {}
    fn assert_debug<T: core::fmt::Debug>() {}
    fn assert_eq_trait<T: Eq>() {}
    assert_copy::<Buffering>();
    assert_debug::<Buffering>();
    assert_eq_trait::<Buffering>();
}
