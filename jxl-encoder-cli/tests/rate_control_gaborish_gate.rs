// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression test for the `--rate-control` gaborish gate at distance > 0.5.
//!
//! Exercises the gate added in commit `f41d59c` (jxl-encoder-cli/src/main.rs:818-822):
//!
//! ```ignore
//! tiny.enable_gaborish = if args.no_gaborish {
//!     false
//! } else {
//!     args.effort >= 3 && distance > 0.5
//! };
//! ```
//!
//! Mirrors libjxl's `enc_frame.cc:281` gate (and `LossyConfig::encode` at
//! `jxl-encoder/src/api.rs:3842`). Before f41d59c, the rate-control CLI path
//! enabled gaborish unconditionally at effort >= 3, so `--rate-control -d 0.4`
//! and `--rate-control -d 0.4 --no-gaborish` produced different bitstreams.
//! With the gate, `-d 0.4` already turns gaborish off internally, so adding
//! `--no-gaborish` is a no-op at d <= 0.5 — the two encodes are byte-identical.
//!
//! ## Why this assertion (and not "sizes differ across d=0.4 vs d=0.6")
//!
//! A naive "different distance → different size" check passes regardless of
//! the gate, because the distance scalar dominates the resulting file size at
//! BOTH `gaborish_on` and `gaborish_off` configurations. The truly
//! discriminating signal is whether `--rate-control -d 0.4` produces the same
//! bitstream as `--rate-control -d 0.4 --no-gaborish`:
//!
//! | Path                                 | gate ON       | gate OFF (pre-f41d59c) |
//! |--------------------------------------|---------------|------------------------|
//! | `-d 0.4` (default)                   | gaborish OFF  | gaborish ON            |
//! | `-d 0.4 --no-gaborish`               | gaborish OFF  | gaborish OFF           |
//! | byte-equal?                          | YES           | NO                     |
//!
//! So the assertion `bytes(d=0.4 default) == bytes(d=0.4 --no-gaborish)` is
//! TRUE under the f41d59c gate and FALSE without it. The d=0.6 control case
//! is also exercised — at d > 0.5 the default keeps gaborish on regardless,
//! so the same comparison should produce DIFFERENT bytes (gaborish ON vs OFF)
//! both with AND without the gate. d=0.6 inequality is therefore not a
//! discriminating signal; it's only included as a sanity guard against
//! accidentally making `--no-gaborish` a no-op for all distances.
//!
//! This test invokes the actual `cjxl-rs` binary (via Cargo's
//! `CARGO_BIN_EXE_cjxl-rs`) so any regression of the gate in `main.rs` is
//! caught. In-process mirroring of the gate logic would only test the test,
//! not the CLI.

#![cfg(feature = "rate-control")]

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Cheap content fingerprint for use in assertion messages — keeps failure
/// output to a single line rather than dumping two ~64 KB byte vectors.
/// Not cryptographic; collisions are vanishingly unlikely between the two
/// JXL bitstreams produced by adjacent CLI invocations.
fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Locate the `frymire.png` real-photo fixture.
///
/// `frymire.png` lives in the sibling `jxl-encoder` crate's tests/images dir
/// (committed alongside the encoder for quality regression tests). Per the
/// project CLAUDE.md "no synthetic-only quality tests" rule we use a real
/// photo rather than a gradient — the gate is purely on `distance > 0.5`,
/// so any real image proves the gate flipped behavior.
fn frymire_source_path() -> PathBuf {
    // CARGO_MANIFEST_DIR is jxl-encoder-cli/. The fixture is one level up
    // in the sibling crate's tests/images.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("../jxl-encoder/tests/images/frymire.png")
}

/// Crop the source PNG to a 256x256 region centered on the image and
/// write it to `target/tests-tmp/rate_control_gaborish_gate/<name>.png`.
///
/// 256x256 is a real-photo crop (not a synthetic image): pixel values come
/// from the frymire test fixture. The crop keeps each end-to-end CLI invocation
/// well under the < 5 s budget the task constraints require.
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
    const CROP: u32 = 256;
    assert!(
        w >= CROP && h >= CROP,
        "fixture {} is too small: {}x{} (need >= {}x{})",
        src.display(),
        w,
        h,
        CROP,
        CROP
    );
    let x0 = (w - CROP) / 2;
    let y0 = (h - CROP) / 2;
    let crop = image::imageops::crop_imm(&img, x0, y0, CROP, CROP).to_image();

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/tests-tmp/rate_control_gaborish_gate");
    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("create_dir_all({}) failed: {e}", out_dir.display()));
    let out_path = out_dir.join(format!("{name}.png"));
    crop.save(&out_path)
        .unwrap_or_else(|e| panic!("save({}) failed: {e}", out_path.display()));
    out_path
}

/// Invoke the freshly-built `cjxl-rs` binary at the given distance and
/// optional `--no-gaborish` flag, returning the encoded bytes.
///
/// `CARGO_BIN_EXE_cjxl-rs` is set by Cargo for integration tests in the
/// same package as the binary — it always points at the up-to-date build.
fn encode_with_rate_control(
    input_png: &Path,
    distance: f32,
    no_gaborish: bool,
    label: &str,
) -> Vec<u8> {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/tests-tmp/rate_control_gaborish_gate");
    std::fs::create_dir_all(&out_dir).expect("create out_dir");
    let out_path = out_dir.join(format!("{label}.jxl"));
    if out_path.exists() {
        let _ = std::fs::remove_file(&out_path);
    }

    let bin = env!("CARGO_BIN_EXE_cjxl-rs");
    let distance_str = format!("{distance}");
    let mut args: Vec<&str> = vec![
        input_png.to_str().expect("utf-8 input path"),
        out_path.to_str().expect("utf-8 output path"),
        "--rate-control",
        "--distance",
        &distance_str,
        "--quiet",
        // Effort 7 is the CLI default and satisfies effort >= 3 (the other
        // half of the gate) so the only varying inputs are `distance` and
        // `--no-gaborish`.
        "--effort",
        "7",
        // RC iteration count: 1 keeps the single encode pass deterministic
        // and well under the 5 s test budget. The gate is checked on the
        // FIRST encode so iteration count doesn't change the gate outcome.
        "--rc-iterations",
        "1",
    ];
    if no_gaborish {
        args.push("--no-gaborish");
    }

    let output = Command::new(bin)
        .args(&args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn cjxl-rs ({bin}): {e}"));

    if !output.status.success() {
        panic!(
            "cjxl-rs --rate-control --distance {distance}{} failed: status={:?}\n\
             stdout: {}\n\
             stderr: {}",
            if no_gaborish { " --no-gaborish" } else { "" },
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let bytes = std::fs::read(&out_path)
        .unwrap_or_else(|e| panic!("read({}) failed: {e}", out_path.display()));
    assert!(
        bytes.len() > 12,
        "encoded {} too small ({} bytes)",
        label,
        bytes.len()
    );

    // JXL signature sanity check: 0xFF 0x0A.
    assert_eq!(bytes[0], 0xFF, "{label}: missing JXL signature byte 0");
    assert_eq!(bytes[1], 0x0A, "{label}: missing JXL signature byte 1");

    bytes
}

/// Proves the `--rate-control` gaborish gate at `distance > 0.5` actually
/// flips encoder behavior at the boundary.
///
/// See module docs for the discrimination matrix. Briefly:
///
/// - At d=0.4 the gate forces gaborish OFF, so `--rate-control -d 0.4` and
///   `--rate-control -d 0.4 --no-gaborish` MUST produce identical bytes.
///   Without the gate, `-d 0.4` keeps gaborish ON while `--no-gaborish`
///   forces it OFF, so the two bitstreams diverge.
/// - At d=0.6 the gate keeps gaborish ON, so the two encodes MUST differ
///   (one with gaborish, one without) — sanity guard that `--no-gaborish`
///   isn't a silent no-op above the gate threshold.
///
/// If this assertion fails, the gate has been removed or its boundary moved.
/// Verify against `enc_frame.cc:281` and `api.rs:3842` before "fixing" the
/// test.
#[test]
fn rate_control_gaborish_gate_flips_at_distance_0_5() {
    let input = write_cropped_fixture("frymire_crop");

    // Below the gate (d=0.4): gaborish should already be off, so adding
    // `--no-gaborish` must be a no-op (byte-identical bitstreams).
    let bytes_d04_default = encode_with_rate_control(&input, 0.4, false, "d04_default");
    let bytes_d04_no_gab = encode_with_rate_control(&input, 0.4, true, "d04_no_gab");

    // Above the gate (d=0.6): gaborish stays on by default, so adding
    // `--no-gaborish` must change the bitstream.
    let bytes_d06_default = encode_with_rate_control(&input, 0.6, false, "d06_default");
    let bytes_d06_no_gab = encode_with_rate_control(&input, 0.6, true, "d06_no_gab");

    let fp_d04_default = fingerprint(&bytes_d04_default);
    let fp_d04_no_gab = fingerprint(&bytes_d04_no_gab);
    let fp_d06_default = fingerprint(&bytes_d06_default);
    let fp_d06_no_gab = fingerprint(&bytes_d06_no_gab);

    eprintln!(
        "rate_control_gaborish_gate (bytes / fingerprint):\n\
         \t d=0.4 default        = {:>7} / 0x{:016x}\n\
         \t d=0.4 --no-gaborish  = {:>7} / 0x{:016x}\n\
         \t d=0.6 default        = {:>7} / 0x{:016x}\n\
         \t d=0.6 --no-gaborish  = {:>7} / 0x{:016x}",
        bytes_d04_default.len(),
        fp_d04_default,
        bytes_d04_no_gab.len(),
        fp_d04_no_gab,
        bytes_d06_default.len(),
        fp_d06_default,
        bytes_d06_no_gab.len(),
        fp_d06_no_gab,
    );

    // PRIMARY: Below the gate, default and --no-gaborish MUST be byte-equal.
    // This is the discriminating assertion: it is FALSE iff the gate has
    // been removed or pushed above d=0.4.
    //
    // Compare on (length, fingerprint) rather than the raw Vec<u8>s so a
    // failure message is one line rather than two ~64 KB byte dumps.
    assert_eq!(
        (bytes_d04_default.len(), fp_d04_default),
        (bytes_d04_no_gab.len(), fp_d04_no_gab),
        "rate-control at d=0.4: default ({} B / 0x{:016x}) ≠ --no-gaborish ({} B / 0x{:016x}). \
         The gaborish gate (jxl-encoder-cli/src/main.rs:818-822, introduced \
         in f41d59c) should force gaborish OFF at d <= 0.5, making `--no-gaborish` \
         a no-op. If they differ, the gate was removed or its threshold raised \
         above 0.4. Reference: libjxl enc_frame.cc:281, jxl-encoder api.rs:3842.",
        bytes_d04_default.len(),
        fp_d04_default,
        bytes_d04_no_gab.len(),
        fp_d04_no_gab,
    );

    // SANITY: Above the gate, --no-gaborish MUST change the bitstream — guards
    // against a regression that makes --no-gaborish a silent no-op everywhere.
    assert_ne!(
        (bytes_d06_default.len(), fp_d06_default),
        (bytes_d06_no_gab.len(), fp_d06_no_gab),
        "rate-control at d=0.6: default ({} B / 0x{:016x}) == --no-gaborish ({} B / 0x{:016x}). \
         Above the gate threshold gaborish is on by default, so `--no-gaborish` \
         must change the encoded bitstream. If the bytes match, `--no-gaborish` \
         is being ignored or the gate threshold has been lowered below 0.6.",
        bytes_d06_default.len(),
        fp_d06_default,
        bytes_d06_no_gab.len(),
        fp_d06_no_gab,
    );
}
