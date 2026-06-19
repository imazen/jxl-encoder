// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression tests for issue #80: `with_effort()` silently discarded
//! every builder setting placed BEFORE it in the chain.
//!
//! `LossyConfig::with_effort` / `LosslessConfig::with_effort` rebuild the
//! config from the effort profile and used to preserve only a hand-picked
//! list of fields. So `LossyConfig::new(1.0).with_ans(false).with_effort(7)`
//! silently lost `with_ans(false)`.
//!
//! The fix mirrors the existing `auto_splines_explicit` / `patches_explicit`
//! / `tree_learning_user_set` touched-tracking pattern: each effort-derived
//! field with a public setter now has a private `*_explicit` flag that the
//! setter flips, and `with_effort` preserves the value when the flag is set.
//!
//! These tests assert BOTH directions of the chain produce the explicitly
//! set value (order-independence), and that the untouched common path keeps
//! adopting the effort-derived default unchanged (so existing hash-locks
//! stay byte-identical — proven separately by the hash-lock suite; here we
//! just pin the default-adoption behaviour at the API level).

use jxl_encoder::{LosslessConfig, LossyConfig, Lz77Method};

// ----------------------------------------------------------------------
// LossyConfig
// ----------------------------------------------------------------------

#[test]
fn lossy_with_ans_survives_with_effort_both_orders() {
    // setter BEFORE with_effort — the footgun #80 reported.
    assert!(
        !LossyConfig::new(1.0).with_ans(false).with_effort(7).ans(),
        "with_ans(false) before with_effort must be preserved (#80)"
    );
    // setter AFTER with_effort — already worked, must keep working.
    assert!(
        !LossyConfig::new(1.0).with_effort(7).with_ans(false).ans(),
        "with_ans(false) after with_effort must be preserved"
    );
    // Untouched common path: with_effort adopts the effort default (true).
    assert!(
        LossyConfig::new(1.0).with_effort(7).ans(),
        "untouched config must keep the effort-derived ans default"
    );
}

#[test]
fn lossy_with_gaborish_survives_with_effort_both_orders() {
    assert!(
        !LossyConfig::new(1.0)
            .with_gaborish(false)
            .with_effort(7)
            .gaborish(),
        "with_gaborish(false) before with_effort must be preserved (#80)"
    );
    assert!(
        !LossyConfig::new(1.0)
            .with_effort(7)
            .with_gaborish(false)
            .gaborish(),
        "with_gaborish(false) after with_effort must be preserved"
    );
    assert!(
        LossyConfig::new(1.0).with_effort(7).gaborish(),
        "untouched config must keep the effort-derived gaborish default"
    );
}

#[test]
fn lossy_with_error_diffusion_survives_with_effort_both_orders() {
    // Default error_diffusion is false; flip to true and check it sticks.
    assert!(
        LossyConfig::new(1.0)
            .with_error_diffusion(true)
            .with_effort(7)
            .error_diffusion(),
        "with_error_diffusion(true) before with_effort must be preserved (#80)"
    );
    assert!(
        LossyConfig::new(1.0)
            .with_effort(7)
            .with_error_diffusion(true)
            .error_diffusion(),
        "with_error_diffusion(true) after with_effort must be preserved"
    );
}

#[test]
fn lossy_with_pixel_domain_loss_survives_with_effort_both_orders() {
    assert!(
        !LossyConfig::new(1.0)
            .with_pixel_domain_loss(false)
            .with_effort(7)
            .pixel_domain_loss(),
        "with_pixel_domain_loss(false) before with_effort must be preserved (#80)"
    );
    assert!(
        !LossyConfig::new(1.0)
            .with_effort(7)
            .with_pixel_domain_loss(false)
            .pixel_domain_loss(),
        "with_pixel_domain_loss(false) after with_effort must be preserved"
    );
}

#[test]
fn lossy_with_lz77_survives_with_effort_both_orders() {
    // Pick effort 5 (lz77 default off) and flip on, so the explicit value
    // genuinely differs from the effort default at that effort.
    assert!(
        LossyConfig::new(1.0).with_lz77(true).with_effort(5).lz77(),
        "with_lz77(true) before with_effort must be preserved (#80)"
    );
    assert!(
        LossyConfig::new(1.0).with_effort(5).with_lz77(true).lz77(),
        "with_lz77(true) after with_effort must be preserved"
    );
    // And the disable direction at an effort where lz77 defaults on (e9).
    assert!(
        !LossyConfig::new(1.0).with_lz77(false).with_effort(9).lz77(),
        "with_lz77(false) before with_effort must be preserved (#80)"
    );
}

#[test]
fn lossy_with_lz77_method_survives_with_effort_both_orders() {
    assert_eq!(
        LossyConfig::new(1.0)
            .with_lz77_method(Lz77Method::Optimal)
            .with_effort(5)
            .lz77_method(),
        Lz77Method::Optimal,
        "with_lz77_method before with_effort must be preserved (#80)"
    );
    assert_eq!(
        LossyConfig::new(1.0)
            .with_effort(5)
            .with_lz77_method(Lz77Method::Optimal)
            .lz77_method(),
        Lz77Method::Optimal,
        "with_lz77_method after with_effort must be preserved"
    );
}

#[test]
fn lossy_perceptual_optimizations_pins_survive_with_effort() {
    // The convenience wrapper touches gaborish + pixel_domain_loss (+ patches);
    // #80 pins all of them so a following with_effort keeps them, matching the
    // pre-existing patches pin.
    let cfg = LossyConfig::new(1.0)
        .with_perceptual_optimizations(false)
        .with_effort(7);
    assert!(
        !cfg.gaborish(),
        "perceptual_optimizations(false) gaborish must survive with_effort"
    );
    assert!(
        !cfg.pixel_domain_loss(),
        "perceptual_optimizations(false) pixel_domain_loss must survive with_effort"
    );
    assert!(
        !cfg.patches(),
        "perceptual_optimizations(false) patches must survive with_effort"
    );
}

// ----------------------------------------------------------------------
// LosslessConfig
// ----------------------------------------------------------------------

#[test]
fn lossless_with_ans_survives_with_effort_both_orders() {
    assert!(
        !LosslessConfig::new().with_ans(false).with_effort(7).ans(),
        "with_ans(false) before with_effort must be preserved (#80)"
    );
    assert!(
        !LosslessConfig::new().with_effort(7).with_ans(false).ans(),
        "with_ans(false) after with_effort must be preserved"
    );
    assert!(
        LosslessConfig::new().with_effort(7).ans(),
        "untouched config must keep the effort-derived ans default"
    );
}

#[test]
fn lossless_with_patches_survives_with_effort_both_orders() {
    assert!(
        !LosslessConfig::new()
            .with_patches(false)
            .with_effort(7)
            .patches(),
        "with_patches(false) before with_effort must be preserved (#80)"
    );
    assert!(
        !LosslessConfig::new()
            .with_effort(7)
            .with_patches(false)
            .patches(),
        "with_patches(false) after with_effort must be preserved"
    );
    // patches default on at e7; off at e3. Untouched config tracks effort.
    assert!(
        LosslessConfig::new().with_effort(7).patches(),
        "untouched config must keep the effort-derived patches default (on at e7)"
    );
}

#[test]
fn lossless_with_tree_learning_survives_with_effort_both_orders() {
    // tree_learning_user_set existed but with_effort never preserved it (#80).
    // Enable tree learning at e5 (where it defaults off) and check it sticks.
    assert!(
        LosslessConfig::new()
            .with_tree_learning(true)
            .with_effort(5)
            .tree_learning(),
        "with_tree_learning(true) before with_effort must be preserved (#80)"
    );
    assert!(
        LosslessConfig::new()
            .with_effort(5)
            .with_tree_learning(true)
            .tree_learning(),
        "with_tree_learning(true) after with_effort must be preserved"
    );
    // Disable at e7 (where it defaults on).
    assert!(
        !LosslessConfig::new()
            .with_tree_learning(false)
            .with_effort(7)
            .tree_learning(),
        "with_tree_learning(false) before with_effort must be preserved (#80)"
    );
}

#[test]
fn lossless_with_lz77_survives_with_effort_both_orders() {
    // lz77 defaults off at e5, on at e7. Flip on at e5.
    assert!(
        LosslessConfig::new().with_lz77(true).with_effort(5).lz77(),
        "with_lz77(true) before with_effort must be preserved (#80)"
    );
    assert!(
        !LosslessConfig::new().with_lz77(false).with_effort(7).lz77(),
        "with_lz77(false) before with_effort must be preserved (#80)"
    );
}

#[test]
fn lossless_with_lz77_method_survives_with_effort_both_orders() {
    assert_eq!(
        LosslessConfig::new()
            .with_lz77_method(Lz77Method::Optimal)
            .with_effort(5)
            .lz77_method(),
        Lz77Method::Optimal,
        "with_lz77_method before with_effort must be preserved (#80)"
    );
    assert_eq!(
        LosslessConfig::new()
            .with_effort(5)
            .with_lz77_method(Lz77Method::Optimal)
            .lz77_method(),
        Lz77Method::Optimal,
        "with_lz77_method after with_effort must be preserved"
    );
}
