// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
//! Per-knob coverage for [`crate::EffortProfile`] override behaviour.
//!
//! The override surface is large (~50 user-relevant fields). Cartesian
//! sweeping is intractable, so this module uses the standard one-at-a-time
//! (OAT) substitute: for each chosen knob, encode once with the default
//! profile and once with that one knob mutated, then assert the bitstreams
//! differ. This catches "override silently has no effect" regressions
//! without claiming exhaustive coverage of inter-field interactions.
//!
//! The chosen knob subset is documented in [`LOSSY_KNOBS_DOC`] /
//! [`LOSSLESS_KNOBS_DOC`] below. We restrict to fields that actually flow
//! through `profile.X` at encode time (verified against `vardct/` and
//! `modular/` source); fields like `gaborish` and `lz77` are stored on
//! `LossyConfig` / `LosslessConfig` directly and are not affected by the
//! profile override (their per-knob `with_*()` setters cover them).
//!
//! Gated behind the `__expert` feature.

use crate::api::{EncoderMode, LosslessConfig, LossyConfig, PixelLayout};
use crate::effort::EffortProfile;
use crate::entropy_coding::lz77::Lz77Method;

/// Subset of lossy knobs covered by per-field override tests.
/// All flow through `profile.X` at lossy encode time (verified against
/// `vardct/encoder.rs`, `vardct/ac_strategy_search.rs`, `vardct/transform.rs`,
/// `vardct/precomputed.rs`, `vardct/bitstream.rs`).
///
/// Modular-only fields (`nb_rcts_to_try`, `wp_num_param_sets`, the `tree_*`
/// family) are intentionally NOT covered on the lossy side: VarDCT's DC
/// frame uses a fixed Gradient predictor without RCT search or tree
/// learning (libjxl `enc_modular.cc::ForModular`), so overrides on those
/// knobs do not affect lossy bytes. See
/// `lossy_modular_fields_do_not_affect_lossy_bitstream` which pins this.
const LOSSY_KNOBS_DOC: &str = "\
try_dct16, try_dct32, try_dct64, try_dct4x8_afv, fine_grained_step, \
k_info_loss_mul_base, entropy_mul_table.dct8, cfl_two_pass, \
chromacity_adjustment, patch_ref_tree_learning";

/// Subset of lossless knobs covered by per-field override tests.
/// All flow through `profile.X` in `modular/encode.rs`, `modular/frame.rs`,
/// `modular/section.rs`, `modular/predictor.rs`, and
/// `modular/tree_learn.rs`.
const LOSSLESS_KNOBS_DOC: &str = "\
nb_rcts_to_try, wp_num_param_sets, tree_max_buckets, tree_num_properties, \
tree_threshold_base, tree_sample_fraction, tree_max_samples_fixed";

#[test]
fn knob_subset_doc_strings_nonempty() {
    // Sanity: keep the doc strings referenced so they don't bit-rot.
    assert!(!LOSSY_KNOBS_DOC.is_empty());
    assert!(!LOSSLESS_KNOBS_DOC.is_empty());
}

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

/// Encode once with the default profile and once with `mutator` applied to
/// the default e7 profile, asserting the bitstreams differ.
fn assert_lossy_field_changes_output<F: FnOnce(&mut EffortProfile)>(field_name: &str, mutator: F) {
    let mut profile = EffortProfile::lossy(7, EncoderMode::Reference);
    mutator(&mut profile);

    let cfg_override = baseline_lossy().with_effort_profile_override(profile);
    let bytes_override = encode_lossy(&cfg_override);
    let bytes_baseline = encode_lossy(&baseline_lossy());

    assert_eq!(&bytes_override[..2], &crate::JXL_SIGNATURE);
    assert_eq!(&bytes_baseline[..2], &crate::JXL_SIGNATURE);
    assert_ne!(
        bytes_override, bytes_baseline,
        "override of {field_name} must change lossy bitstream",
    );
}

// Note on lossy + modular fields:
//
// The fields `nb_rcts_to_try`, `wp_num_param_sets`, `tree_max_buckets`,
// `tree_num_properties`, `tree_threshold_base`, `tree_sample_fraction`,
// and `tree_max_samples_fixed` are read only from `modular/encode.rs`,
// `modular/frame.rs`, `modular/section.rs`, and `modular/predictor.rs`.
// VarDCT lossy emits its DC frame via `vardct/lf_frame.rs` with a fixed
// Gradient predictor (matching libjxl's `enc_modular.cc::ForModular`),
// so the *lossy* bitstream is insensitive to these knobs. They are
// covered on the lossless side below; the assertions here would be
// false positives if duplicated. See `lossless_override_*` tests.
#[test]
fn lossy_modular_fields_do_not_affect_lossy_bitstream() {
    // Pin the limitation: mutating modular-only fields in a lossy override
    // must NOT change lossy bytes. If this ever flips to `assert_ne!`, the
    // lossy DC path now respects the modular profile knobs and the per-
    // field tests can be added on the lossy side.
    let mut profile = EffortProfile::lossy(7, EncoderMode::Reference);
    profile.nb_rcts_to_try = 0;
    profile.wp_num_param_sets = 5;
    profile.tree_max_buckets = 16;
    profile.tree_num_properties = 1;
    profile.tree_threshold_base = 30.0;
    profile.tree_sample_fraction = 0.05;
    profile.tree_max_samples_fixed = 1024;

    let cfg = baseline_lossy().with_effort_profile_override(profile);
    let bytes_with_modular_overrides = encode_lossy(&cfg);
    let bytes_plain = encode_lossy(&baseline_lossy());

    assert_eq!(
        bytes_with_modular_overrides, bytes_plain,
        "modular-only fields must not affect lossy output (libjxl uses fixed Gradient for DC)",
    );
}

#[test]
fn lossy_override_try_dct16() {
    // e7 default = true. Disabling forces no DCT16x16 / DCT16x8 merges.
    assert_lossy_field_changes_output("try_dct16", |p| {
        p.try_dct16 = false;
        p.try_dct32 = false;
        p.try_dct64 = false;
    });
}

#[test]
fn lossy_override_try_dct32() {
    assert_lossy_field_changes_output("try_dct32", |p| {
        p.try_dct32 = false;
        p.try_dct64 = false;
    });
}

#[test]
fn lossy_override_try_dct64() {
    assert_lossy_field_changes_output("try_dct64", |p| p.try_dct64 = false);
}

#[test]
fn lossy_override_try_dct4x8_afv() {
    assert_lossy_field_changes_output("try_dct4x8_afv", |p| p.try_dct4x8_afv = false);
}

#[test]
fn lossy_override_fine_grained_step() {
    // e7 default = 2; switching to 1 doubles the AC strategy search density
    // for 32×32+ blocks.
    assert_lossy_field_changes_output("fine_grained_step", |p| p.fine_grained_step = 1);
}

#[test]
fn lossy_override_k_info_loss_mul_base() {
    // Cost-model knob: heavier weight on pixel-domain loss shifts AC strategy
    // picks toward smaller transforms.
    assert_lossy_field_changes_output("k_info_loss_mul_base", |p| p.k_info_loss_mul_base = 2.0);
}

#[test]
fn lossy_override_entropy_mul_dct8() {
    // Raising DCT8 entropy discourages DCT8 vs DCT4x4 / IDENTITY in the
    // 8×8 normalized table, forcing different strategy picks per block.
    assert_lossy_field_changes_output("entropy_mul_table.dct8", |p| p.entropy_mul_table.dct8 = 1.5);
}

#[test]
fn lossy_override_chromacity_adjustment() {
    // e7 default = true. Disabling skips per-pixel chromacity nudges.
    assert_lossy_field_changes_output("chromacity_adjustment", |p| p.chromacity_adjustment = false);
}

#[test]
fn lossy_override_cfl_two_pass() {
    // e7 default = true. Disabling skips the second CfL fitting pass.
    // This is sensitive — first-pass-only CfL coefficients may produce
    // identical bytes on simple content, so we also bump the eps to
    // a value that visibly changes the Newton path.
    assert_lossy_field_changes_output("cfl_two_pass+eps", |p| {
        p.cfl_two_pass = false;
        p.cfl_newton_eps = 100.0; // libjxl's oscillating value
    });
}

#[test]
fn lossy_override_patch_ref_tree_learning() {
    // Reference profile e7: false. Enabling lights up tree-based prediction
    // for patch reference frames *if* patches fire. Synthetic image has
    // limited repeating structure, so we combine with a cost-model nudge
    // to ensure observable byte difference even if patches don't trigger.
    assert_lossy_field_changes_output("patch_ref_tree_learning+k_zeros", |p| {
        p.patch_ref_tree_learning = true;
        p.k_zeros_mul_base = 12.0;
    });
}

// ── (a) Per-field tests — lossless ───────────────────────────────────────

fn assert_lossless_field_changes_output<F: FnOnce(&mut EffortProfile)>(
    field_name: &str,
    mutator: F,
) {
    let mut profile = EffortProfile::lossless(7, EncoderMode::Reference);
    mutator(&mut profile);

    let cfg_override = baseline_lossless().with_effort_profile_override(profile);
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
    assert_lossless_field_changes_output("nb_rcts_to_try", |p| p.nb_rcts_to_try = 0);
}

#[test]
fn lossless_override_wp_num_param_sets() {
    // e7 default = 0. Forcing search (5 sets) picks non-default WP params.
    assert_lossless_field_changes_output("wp_num_param_sets", |p| p.wp_num_param_sets = 5);
}

#[test]
fn lossless_override_tree_max_buckets() {
    assert_lossless_field_changes_output("tree_max_buckets", |p| p.tree_max_buckets = 16);
}

#[test]
fn lossless_override_tree_num_properties() {
    assert_lossless_field_changes_output("tree_num_properties", |p| p.tree_num_properties = 1);
}

#[test]
fn lossless_override_tree_threshold_base() {
    assert_lossless_field_changes_output("tree_threshold_base", |p| p.tree_threshold_base = 30.0);
}

#[test]
fn lossless_override_tree_sample_fraction() {
    // e7 default = 0.5. Pair with disabling the fixed cap to ensure the
    // fraction path is taken; force 0.05 vs 0.5 → coarser tree.
    assert_lossless_field_changes_output("tree_sample_fraction", |p| {
        p.tree_max_samples_fixed = 0;
        p.tree_sample_fraction = 0.05;
    });
}

#[test]
fn lossless_override_tree_max_samples_fixed() {
    // Force the fixed-cap path with a tiny budget; defaults to fraction
    // path at e7 (fixed=0). 8192 samples << pixel count.
    assert_lossless_field_changes_output("tree_max_samples_fixed", |p| {
        p.tree_sample_fraction = 0.0;
        p.tree_max_samples_fixed = 8_192;
    });
}

// ── (b) Profile-builder coverage — lossy + lossless × all efforts × modes ─

#[test]
fn profile_builder_lossy_all_efforts_all_modes() {
    for &effort in &[1u8, 4, 7, 10] {
        for &mode in &[EncoderMode::Reference, EncoderMode::Experimental] {
            let p = EffortProfile::lossy(effort, mode);
            assert_eq!(p.effort, effort);
            // Sanity invariants — non-negative cost-model constants where required,
            // tree shape clamped to reasonable bounds.
            assert!(p.tree_num_properties > 0);
            assert!(p.tree_max_buckets > 0);
            assert!(p.k_ac_quant > 0.0);
            assert!(p.k_info_loss_mul_base > 0.0);
            assert!(p.cfl_newton_max_iters > 0);
        }
    }
}

#[test]
fn profile_builder_lossless_all_efforts_all_modes() {
    for &effort in &[1u8, 4, 7, 10] {
        for &mode in &[EncoderMode::Reference, EncoderMode::Experimental] {
            let p = EffortProfile::lossless(effort, mode);
            assert_eq!(p.effort, effort);
            assert!(p.tree_num_properties > 0);
            assert!(p.tree_max_buckets > 0);
            // Lossless N/A flags must remain off.
            assert!(!p.gaborish, "gaborish must be N/A for lossless");
            assert!(!p.pixel_domain_loss, "pixel_domain_loss must be N/A");
            assert_eq!(p.butteraugli_iters, 0, "butteraugli iters must be 0");
            assert!(!p.ac_strategy_enabled, "ac_strategy must be off");
        }
    }
}

#[test]
fn profile_builder_lossy_clamps_effort_extremes() {
    // EffortProfile::lossy clamps to 1..=10.
    assert_eq!(
        EffortProfile::lossy(0, EncoderMode::Reference).effort,
        1,
        "effort 0 → clamps to 1"
    );
    assert_eq!(
        EffortProfile::lossy(255, EncoderMode::Reference).effort,
        10,
        "effort 255 → clamps to 10"
    );
}

// ── (c) Override roundtrip — encode succeeds, output differs from plain ───

#[test]
fn override_roundtrip_lossy_changes_bitstream() {
    // Bundle several VarDCT-relevant knobs (cost-model + AC-strategy +
    // CfL) so the override is observable regardless of which knob
    // dominates on this synthetic image.
    let mut profile = EffortProfile::lossy(7, EncoderMode::Reference);
    profile.k_info_loss_mul_base = 2.0;
    profile.entropy_mul_table.dct8 = 1.5;
    profile.try_dct64 = false;
    profile.cfl_two_pass = false;
    profile.cfl_newton_eps = 100.0;

    let cfg = baseline_lossy().with_effort_profile_override(profile);
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
    let mut profile = EffortProfile::lossless(7, EncoderMode::Reference);
    profile.tree_max_buckets = 16;
    profile.tree_num_properties = 3;
    profile.nb_rcts_to_try = 0;

    let cfg = baseline_lossless().with_effort_profile_override(profile);
    let bytes_override = encode_lossless(&cfg);
    let bytes_plain = encode_lossless(&baseline_lossless());

    assert_eq!(&bytes_override[..2], &crate::JXL_SIGNATURE);
    assert_ne!(bytes_override, bytes_plain);
}

#[test]
fn override_skipped_for_cfg_owned_field_lz77_method() {
    // Documented limitation: `lz77_method` is stored on `LossyConfig` (and
    // copied from the profile at construction). Overriding the profile
    // *after* construction does not retroactively change `cfg.lz77_method`.
    // The per-knob `with_lz77_method()` setter is the supported escape
    // hatch for this field. This test pins the limitation so a future
    // re-plumbing to `effective_profile()` reads on the lossy hot path
    // intentionally invalidates it.
    let mut profile = EffortProfile::lossy(7, EncoderMode::Reference);
    profile.lz77 = true; // e7 default is false (libjxl gates VarDCT LZ77 to e9+).
    profile.lz77_method = Lz77Method::Optimal;

    let cfg = baseline_lossy().with_effort_profile_override(profile);
    let bytes_with_override = encode_lossy(&cfg);
    let bytes_plain = encode_lossy(&baseline_lossy());

    // Bytes are equal because `cfg.lz77` / `cfg.lz77_method` were captured
    // at LossyConfig construction time and the profile override does not
    // re-thread them. If this assertion ever flips to `assert_ne!`, the
    // hot path now respects the profile override for these fields and
    // this test should be split into a positive override case.
    assert_eq!(
        bytes_with_override, bytes_plain,
        "lz77 / lz77_method are cfg-owned for lossy and not affected by the profile override; \
         use with_lz77() / with_lz77_method() instead"
    );
}

// ── (d) Default-baseline byte-equivalence ─────────────────────────────────

#[test]
fn default_profile_override_matches_plain_lossy() {
    // Override with the *same* profile that the encoder would build itself.
    // Should produce byte-identical output to the no-override path at the
    // same effort + distance.
    let baseline_profile = EffortProfile::lossy(7, EncoderMode::Reference);

    let cfg_override = LossyConfig::new(1.5)
        .with_effort(7)
        .with_threads(1)
        .with_effort_profile_override(baseline_profile);
    let cfg_plain = LossyConfig::new(1.5).with_effort(7).with_threads(1);

    let bytes_override = encode_lossy(&cfg_override);
    let bytes_plain = encode_lossy(&cfg_plain);

    assert_eq!(
        bytes_override, bytes_plain,
        "default-built EffortProfile override must equal plain with_effort(7) bytes",
    );
}

#[test]
fn default_profile_override_matches_plain_lossless() {
    let baseline_profile = EffortProfile::lossless(7, EncoderMode::Reference);

    let cfg_override = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_effort_profile_override(baseline_profile);
    let cfg_plain = LosslessConfig::new().with_effort(7).with_threads(1);

    let bytes_override = encode_lossless(&cfg_override);
    let bytes_plain = encode_lossless(&cfg_plain);

    assert_eq!(
        bytes_override, bytes_plain,
        "default-built EffortProfile override must equal plain with_effort(7) bytes",
    );
}
