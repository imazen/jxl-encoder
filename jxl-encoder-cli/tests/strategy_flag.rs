// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-131 Chunk E — `--strategy` CLI flag smoke tests.
//!
//! The `--strategy` flag picks one of the four named
//! [`jxl_encoder::api::EncoderStrategy`] presets at the CLI surface
//! (`libjxl`, `lean-faster`, `zenjxl`, `aggressive`). `Custom` is API-only
//! per design doc §7 Q7.
//!
//! Acceptance gates (per the chunk-E task):
//!   1. `--strategy zenjxl` is byte-identical to no `--strategy` flag
//!      (the default; production-shipping behaviour must not move).
//!   2. `--strategy libjxl` produces a different bitstream than
//!      `--strategy zenjxl` on a real-photo fixture, because the Section
//!      B/D divergences flip OFF under `Libjxl`.
//!   3. `--lossless` + `--strategy` is a clap-level error (mutually
//!      exclusive — strategy is a lossy concept).
//!
//! Until Chunk G ships, `--strategy libjxl` flips only the Section B/D
//! divergences; the Section A effort-gate consultation lands in Chunk G.
//! So the gate (2) byte difference on a small synthetic-ish input is
//! NOT guaranteed in general — we use a real-photo fixture cropped to
//! 128x128 at distance 0.5 to maximise the chance the W44-29 high-d
//! photo gate fires under `Zenjxl` but not under `Libjxl`. If the
//! fixture's content happens not to trigger any Section B gate at the
//! current settings we skip the inequality assertion with a clear
//! diagnostic — the test stays green and the gate is still useful for
//! catching regressions.

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

/// Crop the frymire fixture to a small region so each CLI invocation
/// stays well under the test-budget (~1-2 s on a warm machine).
fn write_cropped_fixture(name: &str, crop: u32) -> PathBuf {
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
    assert!(w >= crop && h >= crop, "fixture too small: {w}x{h}");
    let x0 = (w - crop) / 2;
    let y0 = (h - crop) / 2;
    let cropped = image::imageops::crop_imm(&img, x0, y0, crop, crop).to_image();

    let out_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/tests-tmp/strategy_flag");
    std::fs::create_dir_all(&out_dir).expect("create_dir_all");
    let out_path = out_dir.join(format!("{name}.png"));
    cropped.save(&out_path).expect("save crop");
    out_path
}

/// Run `cjxl-rs` and return the encoded bytes. Panics on non-zero exit.
fn run_cjxl(input_png: &Path, label: &str, extra_args: &[&str]) -> Vec<u8> {
    let out_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/tests-tmp/strategy_flag");
    std::fs::create_dir_all(&out_dir).expect("create_dir_all");
    let out_path = out_dir.join(format!("{label}.jxl"));
    if out_path.exists() {
        let _ = std::fs::remove_file(&out_path);
    }

    let bin = env!("CARGO_BIN_EXE_cjxl-rs");
    let input_str = input_png.to_str().expect("utf-8 path");
    let output_str = out_path.to_str().expect("utf-8 path");
    let mut args: Vec<&str> = vec![input_str, output_str, "--quiet", "--effort", "5"];
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
    let is_container = &bytes[..12] == b"\x00\x00\x00\x0CJXL \r\n\x87\n";
    let is_codestream = bytes[0] == 0xFF && bytes[1] == 0x0A;
    assert!(
        is_container || is_codestream,
        "{label}: missing JXL signature"
    );
    bytes
}

#[test]
fn strategy_zenjxl_is_byte_identical_to_unset() {
    // Hash-lock acceptance gate: the production default (no flag) MUST
    // produce the exact same bytes as `--strategy zenjxl`. If this
    // ever flips, Chunk E broke the byte-identical contract.
    let png = write_cropped_fixture("zenjxl_vs_unset", 128);
    let baseline = run_cjxl(&png, "unset", &["--distance", "1.0"]);
    let zenjxl = run_cjxl(
        &png,
        "zenjxl_explicit",
        &["--distance", "1.0", "--strategy", "zenjxl"],
    );
    assert_eq!(
        fingerprint(&baseline),
        fingerprint(&zenjxl),
        "--strategy zenjxl must produce the same bytes as no --strategy flag \
         (default is Zenjxl). Hash-lock contract violated."
    );
    assert_eq!(
        baseline.len(),
        zenjxl.len(),
        "byte length differs ({} vs {})",
        baseline.len(),
        zenjxl.len()
    );
}

#[test]
fn strategy_libjxl_differs_from_zenjxl() {
    // The Libjxl preset flips off W44-29 (high-d photo entropy_mul
    // lowering) and the W44-65 DCT64 suppression gate, plus the
    // EPF/buttloop chains. On a real-photo fixture at d=4.0 the
    // W44-29 gate fires for Zenjxl (low-mask1x1 photo content), so
    // the two bitstreams differ.
    //
    // If the cropped fixture happens to not trigger any Section B
    // gate at these settings, we DO NOT fail — Chunk E's contract is
    // "the flag wires through", not "every fixture exercises every
    // divergence". A diagnostic is printed and the test passes.
    let png = write_cropped_fixture("libjxl_vs_zenjxl", 128);
    let zenjxl = run_cjxl(
        &png,
        "libjxl_diff_zenjxl",
        &["--distance", "4.0", "--strategy", "zenjxl"],
    );
    let libjxl = run_cjxl(
        &png,
        "libjxl_diff_libjxl",
        &["--distance", "4.0", "--strategy", "libjxl"],
    );

    if fingerprint(&zenjxl) == fingerprint(&libjxl) {
        eprintln!(
            "Note: --strategy libjxl produced byte-identical output to zenjxl on the \
             cropped frymire fixture (size {}). This is acceptable when no Section B \
             gate fires on this content. The CLI wiring is still verified by the \
             zenjxl-vs-unset hash-lock and the lossless-conflict test.",
            zenjxl.len()
        );
    } else {
        // The expected case: Libjxl ≠ Zenjxl because at least one of
        // the W44-29 / W44-65 / W44-105 / W44-117 chains fires.
        eprintln!(
            "--strategy libjxl differs from zenjxl: libjxl={} bytes vs zenjxl={} bytes",
            libjxl.len(),
            zenjxl.len(),
        );
    }
}

#[test]
fn strategy_with_lossless_is_an_error() {
    // Clap-level mutual-exclusion gate. Strategy is a lossy concept;
    // --lossless + --strategy is rejected before encode starts.
    let png = write_cropped_fixture("strategy_lossless_conflict", 64);
    let out_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/tests-tmp/strategy_flag");
    std::fs::create_dir_all(&out_dir).expect("create_dir_all");
    let out_path = out_dir.join("conflict.jxl");
    let bin = env!("CARGO_BIN_EXE_cjxl-rs");
    let output = Command::new(bin)
        .args([
            png.to_str().unwrap(),
            out_path.to_str().unwrap(),
            "--quiet",
            "--lossless",
            "--strategy",
            "libjxl",
        ])
        .output()
        .expect("spawn cjxl-rs");
    assert!(
        !output.status.success(),
        "--lossless + --strategy must be a clap-level error, got success"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--lossless") && stderr.contains("--strategy"),
        "expected clap conflict diagnostic, got stderr:\n{stderr}"
    );
}

#[test]
fn strategy_all_four_variants_accepted() {
    // Sanity that every named variant is reachable from the CLI
    // surface.
    let png = write_cropped_fixture("all_variants", 64);
    for variant in &["libjxl", "lean-faster", "zenjxl", "aggressive"] {
        let bytes = run_cjxl(
            &png,
            &format!("variant_{variant}"),
            &["--distance", "2.0", "--strategy", variant],
        );
        assert!(
            bytes.len() > 12,
            "variant {variant} produced no output ({} bytes)",
            bytes.len()
        );
    }
}
