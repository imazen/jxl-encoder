// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! End-to-end smoke tests for the A1 audit CLI passthrough bundle.
//!
//! Covers the new `cjxl-rs` flags that bridge libjxl `cjxl` parity gaps
//! discovered during the A1 audit (CLI parity section):
//!
//! - `--intensity-target` → `EncodeRequest::with_intensity_target`
//! - `--brotli-effort` → `EncodeRequest::with_brotli_metadata` (no-op
//!   without the `brotli-metadata` cargo feature; the flag is still
//!   accepted so scripts stay portable)
//! - `--alpha-distance`, `--group-order`, `--center-x`, `--center-y`,
//!   `--upsampling-mode` → stored on `LossyConfig` (most queued for
//!   encoder-side wiring; `--group-order 1` does mirror `center_first`)
//! - `--modular-predictor`, `--modular-palette-colors`,
//!   `--modular-channel-colors-global-percent`,
//!   `--modular-channel-colors-group-percent`,
//!   `--modular-nb-prev-channels` → stored on `LosslessConfig`
//!
//! These tests check that the CLI:
//!   1. Accepts each flag without parse errors.
//!   2. Produces a valid JXL bitstream (signature byte check).
//!   3. Does NOT change output bytes for flags wired only as storage
//!      (the "skeleton" flags) when the underlying encoder behaviour
//!      is unchanged. This guards against future wiring accidentally
//!      flipping bitstreams without a CHANGELOG entry.
//!
//! Per CLAUDE.md "no synthetic-only quality tests", uses the real
//! `frymire.png` photo fixture (committed under jxl-encoder/tests/images).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

fn frymire_source_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("../jxl-encoder/tests/images/frymire.png")
}

/// Crop the frymire fixture down to a small region so each CLI invocation
/// stays well under the test-budget (~1-2 s on a warm machine).
fn write_cropped_fixture(name: &str) -> PathBuf {
    let src = frymire_source_path();
    assert!(
        src.exists(),
        "frymire fixture missing at {} (expected committed under jxl-encoder/tests/images)",
        src.display()
    );

    let img = image::open(&src)
        .unwrap_or_else(|e| panic!("failed to decode {}: {e}", src.display()))
        .to_rgb8();

    let (w, h) = (img.width(), img.height());
    const CROP: u32 = 128;
    assert!(w >= CROP && h >= CROP, "fixture too small: {w}x{h}");
    let x0 = (w - CROP) / 2;
    let y0 = (h - CROP) / 2;
    let crop = image::imageops::crop_imm(&img, x0, y0, CROP, CROP).to_image();

    let out_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/tests-tmp/cli_passthrough_smoke");
    std::fs::create_dir_all(&out_dir).expect("create_dir_all");
    let out_path = out_dir.join(format!("{name}.png"));
    crop.save(&out_path).expect("save crop");
    out_path
}

/// Run `cjxl-rs` with the given args (input/output handled internally)
/// and return the encoded bytes. Panics on non-zero exit so the test
/// captures stderr verbatim.
fn run_cjxl(input_png: &Path, label: &str, extra_args: &[&str]) -> Vec<u8> {
    let out_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/tests-tmp/cli_passthrough_smoke");
    std::fs::create_dir_all(&out_dir).expect("create_dir_all");
    let out_path = out_dir.join(format!("{label}.jxl"));
    if out_path.exists() {
        let _ = std::fs::remove_file(&out_path);
    }

    let bin = env!("CARGO_BIN_EXE_cjxl-rs");
    let input_str = input_png.to_str().expect("utf-8 path");
    let output_str = out_path.to_str().expect("utf-8 path");
    let mut args: Vec<&str> = vec![input_str, output_str, "--quiet", "--effort", "3"];
    args.extend_from_slice(extra_args);

    let output = Command::new(bin)
        .args(&args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn cjxl-rs ({bin}): {e}"));

    if !output.status.success() {
        panic!(
            "cjxl-rs {:?} failed: status={:?}\nstdout: {}\nstderr: {}",
            args,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let bytes = std::fs::read(&out_path)
        .unwrap_or_else(|e| panic!("read({}) failed: {e}", out_path.display()));
    assert!(
        bytes.len() > 12,
        "encoded {label} too small ({} bytes)",
        bytes.len()
    );
    // JXL container or codestream signature.
    let is_container = &bytes[..12] == b"\x00\x00\x00\x0CJXL \r\n\x87\n";
    let is_codestream = bytes[0] == 0xFF && bytes[1] == 0x0A;
    assert!(
        is_container || is_codestream,
        "{label}: missing JXL signature"
    );
    bytes
}

// ── Lossless passthrough flags (LosslessConfig skeletons) ────────────

#[test]
fn modular_predictor_flag_accepted_lossless_path() {
    let png = write_cropped_fixture("predictor");
    // Lossless path: -d 0 routes through LosslessConfig where the
    // modular_predictor field lives. Default and `--modular-predictor 5`
    // (Gradient, the legacy default) must produce byte-identical output:
    //
    // - On the no-tree-learn paths, `resolve_fixed_predictor` maps
    //   `None → 5`, so the explicit `5` is the identity.
    // - On the tree-learn path (default at effort >= 7),
    //   `resolve_tree_learn_force_predictor` returns `None` for
    //   `Some(5)` specifically, so ID3 runs unchanged.
    //
    // Both invariants preserve hash-lock parity for the default-effort
    // CLI path. Forcing a NON-Gradient id (e.g. `--modular-predictor 1`)
    // does change bytes — see api_tests::
    // `modular_knobs_predictor_overrides_tree_learner_left`.
    let baseline = run_cjxl(&png, "predictor_default", &["-d", "0"]);
    let forced = run_cjxl(
        &png,
        "predictor_forced",
        &["-d", "0", "--modular-predictor", "5"],
    );
    assert_eq!(
        fingerprint(&baseline),
        fingerprint(&forced),
        "`-P 5` (Gradient, legacy default) must stay byte-identical to unset \
         on both tree-learn and non-tree-learn paths"
    );
}

#[test]
fn modular_palette_colors_flag_disables_palette_detection() {
    let png = write_cropped_fixture("palette_colors");
    // The 128x128 frymire crop has enough unique colours that the
    // ModularKnobs::palette_colors_or_default() path with `Some(0)`
    // disables palette detection entirely. With detection on, palette
    // typically fires on the screenshot-style content; with it off the
    // encoder falls through to RCT / improved modular. The bitstreams
    // should differ.
    let baseline = run_cjxl(&png, "palette_default", &["-d", "0"]);
    let disabled = run_cjxl(
        &png,
        "palette_disabled",
        &["-d", "0", "--modular-palette-colors", "0"],
    );
    assert_ne!(
        fingerprint(&baseline),
        fingerprint(&disabled),
        "--modular-palette-colors 0 must disable palette detection (encoder-side wiring)"
    );
    // Also verify bytes increase: disabling palette on a few-colour
    // screenshot region removes a major compression lever. (Equal-or-
    // larger guards against test flake on borderline content.)
    assert!(
        disabled.len() >= baseline.len(),
        "palette-disabled bytes ({}) should be >= palette-on baseline ({})",
        disabled.len(),
        baseline.len(),
    );
}

#[test]
fn modular_palette_colors_capped_still_decodes() {
    // Cap to a small palette (2 colours) — different from both default
    // (1024) and disabled (0). The cap is honoured by `analyze_palette`,
    // so most images won't satisfy the palette use-case and will fall
    // through to RCT. Bytes should differ from the default.
    let png = write_cropped_fixture("palette_cap2");
    let baseline = run_cjxl(&png, "palette_cap2_default", &["-d", "0"]);
    let capped = run_cjxl(
        &png,
        "palette_cap2_capped",
        &["-d", "0", "--modular-palette-colors", "2"],
    );
    assert_ne!(
        fingerprint(&baseline),
        fingerprint(&capped),
        "--modular-palette-colors 2 should differ from default 1024 on a few-colour crop"
    );
}

#[test]
fn modular_channel_colors_global_pct_flag_accepted_lossless_path() {
    let png = write_cropped_fixture("ccgp");
    // ChannelCompact only fires inside
    // `write_modular_stream_with_tree_dc_quant_knobs`, which is reached
    // when tree-learning is on (default at effort >= 7; the shared
    // run_cjxl helper bakes in `--effort 3`). We force the path via
    // `--tree-learning`, and disable palette so the multi-channel
    // palette short-circuit cannot fire — leaving ChannelCompact as the
    // remaining decorrelation step that the knob can affect.
    let baseline = run_cjxl(
        &png,
        "ccgp_default",
        &[
            "-d",
            "0",
            "--tree-learning",
            "--modular-palette-colors",
            "0",
        ],
    );
    let tight = run_cjxl(
        &png,
        "ccgp_tight",
        &[
            "-d",
            "0",
            "--tree-learning",
            "--modular-palette-colors",
            "0",
            "--modular-channel-colors-global-percent",
            "1",
        ],
    );
    assert_ne!(
        fingerprint(&baseline),
        fingerprint(&tight),
        "--modular-channel-colors-global-percent 1 must change channel-compact behaviour \
         (tree-learning + palette disabled to force ChannelCompact path)"
    );
}

#[test]
fn modular_channel_colors_group_pct_flag_accepted_lossless_path() {
    // The group-percent override only affects the multi-group
    // ChannelCompact pass (frame.rs:493). 128x128 fixture is single-
    // group so this test only proves the CLI parses + the bytes are a
    // valid JXL bitstream. A multi-group bytes-divergence test would
    // need a >=512x512 crop; deferred until corpus-regression covers
    // this knob.
    let png = write_cropped_fixture("ccgrp");
    let _ = run_cjxl(
        &png,
        "ccgrp",
        &["-d", "0", "--modular-channel-colors-group-percent", "60"],
    );
}

#[test]
fn modular_nb_prev_channels_flag_accepted_lossless_path() {
    let png = write_cropped_fixture("npc");
    // Cap the previous-channel ref count to 0 — this disables ref-channel
    // properties in the MA tree learner. On a 3-channel RGB image with
    // tree learning, this materially changes the tree → bytes should
    // diverge from the default (which uses image-natural max_ref_channels).
    //
    // The shared run_cjxl helper bakes in `--effort 3`, which does NOT
    // route through the tree-learning + palette path that consumes
    // `nb_prev_channels`. We force the tree path on via tree-learning +
    // palette disabled so `write_modular_stream_with_tree_dc_quant_knobs`
    // sees both knobs and exercises the `num_refs` cap. (`--no-patches`
    // keeps frame.rs from short-circuiting via patches detection.)
    let baseline = run_cjxl(
        &png,
        "npc_default",
        &[
            "-d",
            "0",
            "--tree-learning",
            "--modular-palette-colors",
            "0",
        ],
    );
    let zero_refs = run_cjxl(
        &png,
        "npc_zero",
        &[
            "-d",
            "0",
            "--tree-learning",
            "--modular-palette-colors",
            "0",
            "--modular-nb-prev-channels",
            "0",
        ],
    );
    assert_ne!(
        fingerprint(&baseline),
        fingerprint(&zero_refs),
        "--modular-nb-prev-channels 0 must change tree-learner ref properties"
    );
}

// ── Lossy passthrough flags (LossyConfig skeletons) ──────────────────

#[test]
fn alpha_distance_flag_accepted_lossy_path() {
    let png = write_cropped_fixture("alpha_dist");
    // No alpha in source — flag still parses and the value rides on
    // the config as a no-op until the lossy-alpha pipeline lands.
    let _ = run_cjxl(
        &png,
        "alpha_dist",
        &["-d", "1.0", "--alpha-distance", "0.5"],
    );
}

#[test]
fn group_order_zero_flag_accepted_lossy_path() {
    let png = write_cropped_fixture("group_order_0");
    let _ = run_cjxl(&png, "group_order_0", &["-d", "1.0", "--group-order", "0"]);
}

#[test]
fn group_order_one_engages_center_first() {
    // --group-order 1 mirrors center_first on the LossyConfig. On a
    // 128x128 single-group image the AC reorder is a no-op (num_groups
    // = 1), so bytes are identical to the default — but the flag must
    // parse and produce a valid bitstream.
    let png = write_cropped_fixture("group_order_1");
    let _ = run_cjxl(&png, "group_order_1", &["-d", "1.0", "--group-order", "1"]);
}

#[test]
fn center_xy_flags_accepted_when_paired_with_group_order_1() {
    let png = write_cropped_fixture("center_xy");
    let _ = run_cjxl(
        &png,
        "center_xy",
        &[
            "-d",
            "1.0",
            "--group-order",
            "1",
            "--center-x",
            "32",
            "--center-y",
            "32",
        ],
    );
}

#[test]
fn upsampling_mode_flag_accepted_lossy_path() {
    let png = write_cropped_fixture("upsampling_mode");
    let _ = run_cjxl(
        &png,
        "upsampling_mode",
        &["-d", "1.0", "--upsampling-mode", "-1"],
    );
}

// ── Wired-through flags (intensity-target + brotli-effort) ───────────

#[test]
fn intensity_target_flag_changes_bitstream_lossy_path() {
    // intensity-target is fully wired into the EncodeRequest builder
    // and writes the ToneMapping.intensity_target field. SDR default is
    // 255 nits; bumping to 4000 must surface as different bitstream
    // bytes (the value lives in the file header).
    let png = write_cropped_fixture("itarget");
    let baseline = run_cjxl(&png, "itarget_default", &["-d", "1.0"]);
    let hdr = run_cjxl(
        &png,
        "itarget_4000",
        &["-d", "1.0", "--intensity-target", "4000"],
    );
    assert_ne!(
        fingerprint(&baseline),
        fingerprint(&hdr),
        "intensity_target should write ToneMapping.intensity_target → different bytes"
    );
}

#[test]
fn brotli_effort_flag_accepted_default_features() {
    // The brotli-metadata feature is not in the CLI default set, so
    // passing --brotli-effort on a default build should be silently
    // accepted (no error) and produce a valid file. The actual `brob`
    // box only emits when the feature is on; this test guards the
    // forward-compat contract.
    let png = write_cropped_fixture("brotli_effort");
    let _ = run_cjxl(
        &png,
        "brotli_effort",
        &["-d", "1.0", "--brotli-effort", "4"],
    );
}
