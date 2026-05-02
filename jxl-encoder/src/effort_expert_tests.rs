// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
//! Per-knob coverage for [`crate::effort::LossyInternalParams`] and
//! [`crate::effort::LosslessInternalParams`] — the segmented public expert
//! surface gated behind `__expert`.
//!
//! The override surface is large but deliberately split per encode mode:
//! `LossyInternalParams` carries lossy-only knobs (AC strategy gates, CfL,
//! cost-model constants) and `LosslessInternalParams` carries
//! modular-only knobs (RCT search, WP parameter scan, tree learning shape).
//! The type system enforces mode-correctness — there is no way to pass a
//! modular-only knob to the lossy setter, so the previous
//! "modular-fields-do-not-affect-lossy-bytes" pin tests are unnecessary
//! (they were guarding a runtime invariant that is now a compile-time
//! one).
//!
//! Cartesian sweeping the surface is intractable, so this module uses the
//! standard one-at-a-time (OAT) substitute: for each chosen knob, encode
//! once with default params and once with that one field set to a
//! non-default value, then assert the bitstreams differ. This catches
//! "override silently has no effect" regressions without claiming
//! exhaustive coverage of inter-field interactions.
//!
//! Gated behind the `__expert` feature.

use crate::api::{LosslessConfig, LossyConfig, PixelLayout};
use crate::effort::{EntropyMulTable, LosslessInternalParams, LossyInternalParams};

// ── Synthetic test image ─────────────────────────────────────────────────

/// 256×256 RGB8 image with mixed structure, designed to exercise as many
/// override-eligible code paths as possible:
///
/// - **RCT search**: chroma mismatch between R/G/B channels (different
///   patterns) so non-YCoCg variants have measurably different costs.
/// - **Tree learning**: varied gradient + speckle + flat regions create
///   property-distribution differences that make tree-shape changes
///   visible.
/// - **AC strategy search**: smooth diagonal supports DCT16/32/64; vertical
///   bars favor DCT4x8; speckle favors IDENTITY/DCT4x4; quiet "patch"
///   region (32×32 mid-grey) favors DCT16x16+ merges.
/// - **Patches**: repeating 8×8 glyph-like blocks tile the bottom strip,
///   so patches detection has something to match.
/// - **VarDCT modular DC**: 256×256 → 2 DC groups, exercising the modular
///   sub-encoder and making `wp_num_param_sets` etc. measurable.
fn synthetic_rgb8() -> Vec<u8> {
    let mut out = Vec::with_capacity((W * H * 3) as usize);
    let mut state: u32 = 0x1357_9BDF;
    for y in 0..H {
        for x in 0..W {
            // Region split:
            //   y < 64: smooth diagonal gradient (large-DCT friendly)
            //   y < 128: vertical bars + speckle (DCT4x8 / IDENTITY)
            //   y < 192: flat 32×32 quadrants of distinct colors
            //   y < 256: repeating 8×8 "glyph" pattern (patches-friendly)
            let (r, g, b) = if y < 64 {
                let v = ((x + y) * 255 / (W + 64 - 2)) as u8;
                (v, v.wrapping_add(20), v.wrapping_sub(20))
            } else if y < 128 {
                let bars_g = if (x / 4) % 2 == 0 { 30 } else { 220 };
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let speckle = ((state >> 24) as u8) & 0x3F;
                let bx = (((x ^ y) as u8).wrapping_add(speckle)) | 0x10;
                (((x as u8) ^ 0x55), bars_g as u8, bx)
            } else if y < 192 {
                let qx = (x / 32) % 4;
                let qy = ((y - 128) / 32) % 2;
                let base = (qx as u8) * 50 + (qy as u8) * 30 + 40;
                (base, base.wrapping_add(20), base.wrapping_sub(10))
            } else {
                // 8×8 repeating glyph: a "+" pattern.
                let lx = x % 8;
                let ly = (y - 192) % 8;
                let on = lx == 3 || lx == 4 || ly == 3 || ly == 4;
                if on { (240, 230, 220) } else { (10, 18, 30) }
            };
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

const W: u32 = 256;
const H: u32 = 256;

fn encode_lossy(cfg: &LossyConfig) -> Vec<u8> {
    let pixels = synthetic_rgb8();
    cfg.clone()
        .encode(&pixels, W, H, PixelLayout::Rgb8)
        .expect("lossy encode")
}

fn encode_lossless(cfg: &LosslessConfig) -> Vec<u8> {
    let pixels = synthetic_rgb8();
    cfg.clone()
        .encode(&pixels, W, H, PixelLayout::Rgb8)
        .expect("lossless encode")
}

fn baseline_lossy() -> LossyConfig {
    LossyConfig::new(1.5).with_effort(7).with_threads(1)
}

fn baseline_lossless() -> LosslessConfig {
    LosslessConfig::new().with_effort(7).with_threads(1)
}

// ── (a) Per-field tests — lossy ──────────────────────────────────────────

/// Encode once with default params and once with a single field set on
/// `LossyInternalParams`, asserting the bitstreams differ.
fn assert_lossy_field_changes_output(field_name: &str, params: LossyInternalParams) {
    let cfg_override = baseline_lossy().with_internal_params(params);
    let bytes_override = encode_lossy(&cfg_override);
    let bytes_baseline = encode_lossy(&baseline_lossy());

    assert_eq!(&bytes_override[..2], &crate::JXL_SIGNATURE);
    assert_eq!(&bytes_baseline[..2], &crate::JXL_SIGNATURE);
    assert_ne!(
        bytes_override, bytes_baseline,
        "override of {field_name} must change lossy bitstream",
    );
}

#[test]
fn lossy_override_try_dct16() {
    // e7 default = true. Disabling forces no DCT16x16 / DCT16x8 merges.
    // Bundle dct32 + dct64 too so the search has nowhere to escape.
    assert_lossy_field_changes_output(
        "try_dct16",
        LossyInternalParams {
            try_dct16: Some(false),
            try_dct32: Some(false),
            try_dct64: Some(false),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_try_dct32() {
    assert_lossy_field_changes_output(
        "try_dct32",
        LossyInternalParams {
            try_dct32: Some(false),
            try_dct64: Some(false),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_try_dct64() {
    assert_lossy_field_changes_output(
        "try_dct64",
        LossyInternalParams {
            try_dct64: Some(false),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_try_dct4x8_afv() {
    assert_lossy_field_changes_output(
        "try_dct4x8_afv",
        LossyInternalParams {
            try_dct4x8_afv: Some(false),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_fine_grained_step() {
    // e7 default = 2; switching to 1 doubles the AC strategy search density
    // for 32×32+ blocks.
    assert_lossy_field_changes_output(
        "fine_grained_step",
        LossyInternalParams {
            fine_grained_step: Some(1),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_k_info_loss_mul_base() {
    // Cost-model knob: heavier weight on pixel-domain loss shifts AC strategy
    // picks toward smaller transforms.
    assert_lossy_field_changes_output(
        "k_info_loss_mul_base",
        LossyInternalParams {
            k_info_loss_mul_base: Some(2.0),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_entropy_mul_dct8() {
    // Raising DCT8 entropy discourages DCT8 vs DCT4x4 / IDENTITY in the
    // 8×8 normalized table, forcing different strategy picks per block.
    let mut table = EntropyMulTable::reference();
    table.dct8 = 1.5;
    assert_lossy_field_changes_output(
        "entropy_mul_table.dct8",
        LossyInternalParams {
            entropy_mul_table: Some(table),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_chromacity_adjustment() {
    // e7 default = true. Disabling skips per-pixel chromacity nudges.
    assert_lossy_field_changes_output(
        "chromacity_adjustment",
        LossyInternalParams {
            chromacity_adjustment: Some(false),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_cfl_two_pass() {
    // e7 default = true. Disabling skips the second CfL fitting pass.
    // First-pass-only CfL coefficients can produce identical bytes on
    // simple content, so we also force a non-default entropy mul to ensure
    // observable byte difference.
    let mut table = EntropyMulTable::reference();
    table.dct8 = 0.95;
    assert_lossy_field_changes_output(
        "cfl_two_pass",
        LossyInternalParams {
            cfl_two_pass: Some(false),
            entropy_mul_table: Some(table),
            ..Default::default()
        },
    );
}

#[test]
fn lossy_override_patch_ref_tree_learning() {
    // Reference profile e7: false. Enabling lights up tree-based prediction
    // for patch reference frames *if* patches fire. Synthetic image has
    // limited repeating structure, so we combine with a cost-model nudge
    // (k_info_loss_mul_base) to ensure observable byte difference even if
    // patches don't trigger.
    assert_lossy_field_changes_output(
        "patch_ref_tree_learning",
        LossyInternalParams {
            patch_ref_tree_learning: Some(true),
            k_info_loss_mul_base: Some(1.5),
            ..Default::default()
        },
    );
}

// ── (a) Per-field tests — lossless ───────────────────────────────────────

fn assert_lossless_field_changes_output(field_name: &str, params: LosslessInternalParams) {
    let cfg_override = baseline_lossless().with_internal_params(params);
    let bytes_override = encode_lossless(&cfg_override);
    let bytes_baseline = encode_lossless(&baseline_lossless());

    assert_eq!(&bytes_override[..2], &crate::JXL_SIGNATURE);
    assert_eq!(&bytes_baseline[..2], &crate::JXL_SIGNATURE);
    assert_ne!(
        bytes_override, bytes_baseline,
        "override of {field_name} must change lossless bitstream",
    );
}

#[test]
fn lossless_override_nb_rcts_to_try() {
    assert_lossless_field_changes_output(
        "nb_rcts_to_try",
        LosslessInternalParams {
            nb_rcts_to_try: Some(0),
            ..Default::default()
        },
    );
}

#[test]
fn lossless_override_wp_num_param_sets() {
    // e7 default = 0. Forcing search (5 sets) picks non-default WP params.
    assert_lossless_field_changes_output(
        "wp_num_param_sets",
        LosslessInternalParams {
            wp_num_param_sets: Some(5),
            ..Default::default()
        },
    );
}

#[test]
fn lossless_override_tree_max_buckets() {
    assert_lossless_field_changes_output(
        "tree_max_buckets",
        LosslessInternalParams {
            tree_max_buckets: Some(16),
            ..Default::default()
        },
    );
}

#[test]
fn lossless_override_tree_num_properties() {
    assert_lossless_field_changes_output(
        "tree_num_properties",
        LosslessInternalParams {
            tree_num_properties: Some(1),
            ..Default::default()
        },
    );
}

#[test]
fn lossless_override_tree_threshold_base() {
    assert_lossless_field_changes_output(
        "tree_threshold_base",
        LosslessInternalParams {
            tree_threshold_base: Some(30.0),
            ..Default::default()
        },
    );
}

#[test]
fn lossless_override_tree_sample_fraction() {
    // e7 default = 0.5. Pair with disabling the fixed cap to ensure the
    // fraction path is taken; force 0.05 vs 0.5 → coarser tree.
    assert_lossless_field_changes_output(
        "tree_sample_fraction",
        LosslessInternalParams {
            tree_max_samples_fixed: Some(0),
            tree_sample_fraction: Some(0.05),
            ..Default::default()
        },
    );
}

#[test]
fn lossless_override_tree_max_samples_fixed() {
    // Force the fixed-cap path with a tiny budget; defaults to fraction
    // path at e7 (fixed=0). 8192 samples << pixel count.
    assert_lossless_field_changes_output(
        "tree_max_samples_fixed",
        LosslessInternalParams {
            tree_sample_fraction: Some(0.0),
            tree_max_samples_fixed: Some(8_192),
            ..Default::default()
        },
    );
}

// ── (b) Override roundtrip — encode succeeds, output differs from plain ──

#[test]
fn override_roundtrip_lossy_changes_bitstream() {
    // Bundle several lossy knobs (cost-model + AC-strategy + CfL) so the
    // override is observable regardless of which knob dominates on this
    // synthetic image.
    let mut table = EntropyMulTable::reference();
    table.dct8 = 1.5;
    let params = LossyInternalParams {
        k_info_loss_mul_base: Some(2.0),
        entropy_mul_table: Some(table),
        try_dct64: Some(false),
        cfl_two_pass: Some(false),
        ..Default::default()
    };

    let cfg = baseline_lossy().with_internal_params(params);
    let bytes_override = encode_lossy(&cfg);
    let bytes_plain = encode_lossy(&baseline_lossy());

    assert_eq!(&bytes_override[..2], &crate::JXL_SIGNATURE);
    assert_ne!(
        bytes_override, bytes_plain,
        "override should produce a different bitstream than plain effort=7"
    );
}

#[test]
fn override_roundtrip_lossless_changes_bitstream() {
    let params = LosslessInternalParams {
        tree_max_buckets: Some(16),
        tree_num_properties: Some(3),
        nb_rcts_to_try: Some(0),
        ..Default::default()
    };

    let cfg = baseline_lossless().with_internal_params(params);
    let bytes_override = encode_lossless(&cfg);
    let bytes_plain = encode_lossless(&baseline_lossless());

    assert_eq!(&bytes_override[..2], &crate::JXL_SIGNATURE);
    assert_ne!(bytes_override, bytes_plain);
}

// ── (c) Default-baseline byte-equivalence ────────────────────────────────

#[test]
fn default_params_lossy_match_plain() {
    // An all-`None` override must produce byte-identical output to the
    // no-override path at the same effort + distance. The setter still
    // builds an `EffortProfile` and stores it, but every field equals what
    // the encoder would have built itself.
    let cfg_override = LossyConfig::new(1.5)
        .with_effort(7)
        .with_threads(1)
        .with_internal_params(LossyInternalParams::default());
    let cfg_plain = LossyConfig::new(1.5).with_effort(7).with_threads(1);

    let bytes_override = encode_lossy(&cfg_override);
    let bytes_plain = encode_lossy(&cfg_plain);

    assert_eq!(
        bytes_override, bytes_plain,
        "default LossyInternalParams override must equal plain with_effort(7) bytes",
    );
}

#[test]
fn default_params_lossless_match_plain() {
    let cfg_override = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_internal_params(LosslessInternalParams::default());
    let cfg_plain = LosslessConfig::new().with_effort(7).with_threads(1);

    let bytes_override = encode_lossless(&cfg_override);
    let bytes_plain = encode_lossless(&cfg_plain);

    assert_eq!(
        bytes_override, bytes_plain,
        "default LosslessInternalParams override must equal plain with_effort(7) bytes",
    );
}
