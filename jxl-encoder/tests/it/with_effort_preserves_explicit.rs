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

use jxl_encoder::{ContainerMode, LosslessConfig, LossyConfig, Lz77Method, RctType};

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

// ----------------------------------------------------------------------
// Issue #80 FOLLOW-UP: fixed-default caller-preference fields.
//
// PR #90 (above) covered the effort-DERIVED fields via `*_explicit`
// flags. These fields take a FIXED literal in `new_with_effort` /
// `with_effort_level` (not `profile.X`), so `with_effort` now preserves
// the caller's value UNCONDITIONALLY. Each test pins a value that
// differs from the construction default and asserts it survives
// `with_effort` in BOTH chain orders.
// ----------------------------------------------------------------------

// ---- LossyConfig fixed-default fields ----

#[test]
fn lossy_with_threads_survives_with_effort_both_orders() {
    // default threads = 0; pick 8.
    assert_eq!(
        LossyConfig::new(1.0)
            .with_threads(8)
            .with_effort(7)
            .threads(),
        8,
        "with_threads(8) before with_effort must be preserved (#80 follow-up)"
    );
    assert_eq!(
        LossyConfig::new(1.0)
            .with_effort(7)
            .with_threads(8)
            .threads(),
        8,
        "with_threads(8) after with_effort must be preserved"
    );
    // Untouched common path keeps the fixed default (0) — byte-identical.
    assert_eq!(
        LossyConfig::new(1.0).with_effort(7).threads(),
        0,
        "untouched config keeps the fixed threads default (0)"
    );
}

#[test]
fn lossy_with_resampling_survives_with_effort_both_orders() {
    // default resampling = 1; pick 2. resampling carries an `_explicit`
    // companion (gates auto-resample), so this uses the *_explicit pattern.
    assert_eq!(
        LossyConfig::new(1.0)
            .with_resampling(2)
            .with_effort(7)
            .resampling(),
        2,
        "with_resampling(2) before with_effort must be preserved (#80 follow-up)"
    );
    assert_eq!(
        LossyConfig::new(1.0)
            .with_effort(7)
            .with_resampling(2)
            .resampling(),
        2,
        "with_resampling(2) after with_effort must be preserved"
    );
    // The `_explicit` flag itself must survive too: an explicit
    // `with_resampling(1)` suppresses auto-resample at d>=10, so the
    // EFFECTIVE factor must stay 1 (not auto-bumped to 2) after with_effort.
    assert_eq!(
        LossyConfig::new(12.0)
            .with_resampling(1)
            .with_effort(7)
            .effective_resampling(),
        1,
        "explicit with_resampling(1) must keep suppressing auto-resample at d>=10 after with_effort (#80 follow-up)"
    );
    // Sanity: WITHOUT the explicit pin, auto-resample still fires at d>=10.
    assert_eq!(
        LossyConfig::new(12.0).with_effort(7).effective_resampling(),
        2,
        "auto-resample default must still engage at d>=10 when resampling is untouched"
    );
}

#[test]
fn lossy_with_dot_detection_survives_with_effort_both_orders() {
    // default dot_detection = true; flip to false.
    assert!(
        !LossyConfig::new(1.0)
            .with_dot_detection(false)
            .with_effort(7)
            .dot_detection(),
        "with_dot_detection(false) before with_effort must be preserved (#80 follow-up)"
    );
    assert!(
        !LossyConfig::new(1.0)
            .with_effort(7)
            .with_dot_detection(false)
            .dot_detection(),
        "with_dot_detection(false) after with_effort must be preserved"
    );
    assert!(
        LossyConfig::new(1.0).with_effort(7).dot_detection(),
        "untouched config keeps the fixed dot_detection default (true)"
    );
}

// NOTE: `simplify_invisible` (lossy `with_simplify_invisible` / lossless
// `with_keep_invisible`) has no public getter, only a private flag. Its
// `with_effort` preservation is proven by the in-crate unit tests in
// `src/api.rs` (`with_effort_simplify_invisible_*`), which CAN reach the
// `#[cfg(test)]` accessor — an external integration crate cannot.

#[test]
fn lossy_with_lf_frame_survives_with_effort_both_orders() {
    // default lf_frame = false; flip to true.
    assert!(
        LossyConfig::new(1.0)
            .with_lf_frame(true)
            .with_effort(7)
            .lf_frame(),
        "with_lf_frame(true) before with_effort must be preserved (#80 follow-up)"
    );
    assert!(
        LossyConfig::new(1.0)
            .with_effort(7)
            .with_lf_frame(true)
            .lf_frame(),
        "with_lf_frame(true) after with_effort must be preserved"
    );
    assert!(
        !LossyConfig::new(1.0).with_effort(7).lf_frame(),
        "untouched config keeps the fixed lf_frame default (false)"
    );
}

// ---- LosslessConfig fixed-default fields ----

#[test]
fn lossless_with_lossy_palette_survives_with_effort_both_orders() {
    // default lossy_palette = false; flip to true.
    assert!(
        LosslessConfig::new()
            .with_lossy_palette(true)
            .with_effort(7)
            .lossy_palette(),
        "with_lossy_palette(true) before with_effort must be preserved (#80 follow-up)"
    );
    assert!(
        LosslessConfig::new()
            .with_effort(7)
            .with_lossy_palette(true)
            .lossy_palette(),
        "with_lossy_palette(true) after with_effort must be preserved"
    );
    assert!(
        !LosslessConfig::new().with_effort(7).lossy_palette(),
        "untouched config keeps the fixed lossy_palette default (false)"
    );
}

#[test]
fn lossless_with_modular_predictor_survives_with_effort_both_orders() {
    // default modular_predictor = None; pick Some(5).
    assert_eq!(
        LosslessConfig::new()
            .with_modular_predictor(Some(5))
            .with_effort(7)
            .modular_predictor(),
        Some(5),
        "with_modular_predictor before with_effort must be preserved (#80 follow-up)"
    );
    assert_eq!(
        LosslessConfig::new()
            .with_effort(7)
            .with_modular_predictor(Some(5))
            .modular_predictor(),
        Some(5),
        "with_modular_predictor after with_effort must be preserved"
    );
    assert_eq!(
        LosslessConfig::new().with_effort(7).modular_predictor(),
        None,
        "untouched config keeps the fixed modular_predictor default (None)"
    );
}

#[test]
fn lossless_with_force_rct_survives_with_effort_both_orders() {
    // default forced_rct = None; force YCoCg (RctType inner = 6). RctType
    // has no PartialEq, so compare the public `.0` byte.
    assert_eq!(
        LosslessConfig::new()
            .with_force_rct(Some(RctType::YCOCG))
            .with_effort(7)
            .force_rct()
            .map(|r| r.0),
        Some(RctType::YCOCG.0),
        "with_force_rct before with_effort must be preserved (#80 follow-up)"
    );
    assert_eq!(
        LosslessConfig::new()
            .with_effort(7)
            .with_force_rct(Some(RctType::YCOCG))
            .force_rct()
            .map(|r| r.0),
        Some(RctType::YCOCG.0),
        "with_force_rct after with_effort must be preserved"
    );
    assert!(
        LosslessConfig::new().with_effort(7).force_rct().is_none(),
        "untouched config keeps the fixed forced_rct default (None)"
    );
}

#[test]
fn lossless_with_container_mode_survives_with_effort_both_orders() {
    // default container_mode = Auto; pick Always.
    assert!(
        matches!(
            LosslessConfig::new()
                .with_container_mode(ContainerMode::Always)
                .with_effort(7)
                .container_mode(),
            ContainerMode::Always
        ),
        "with_container_mode before with_effort must be preserved (#80 follow-up)"
    );
    assert!(
        matches!(
            LosslessConfig::new()
                .with_effort(7)
                .with_container_mode(ContainerMode::Always)
                .container_mode(),
            ContainerMode::Always
        ),
        "with_container_mode after with_effort must be preserved"
    );
    assert!(
        matches!(
            LosslessConfig::new().with_effort(7).container_mode(),
            ContainerMode::Auto
        ),
        "untouched config keeps the fixed container_mode default (Auto)"
    );
}

#[test]
fn lossless_with_threads_survives_with_effort_both_orders() {
    assert_eq!(
        LosslessConfig::new()
            .with_threads(8)
            .with_effort(7)
            .threads(),
        8,
        "with_threads(8) before with_effort must be preserved (#80 follow-up)"
    );
    assert_eq!(
        LosslessConfig::new()
            .with_effort(7)
            .with_threads(8)
            .threads(),
        8,
        "with_threads(8) after with_effort must be preserved"
    );
    assert_eq!(
        LosslessConfig::new().with_effort(7).threads(),
        0,
        "untouched config keeps the fixed threads default (0)"
    );
}
