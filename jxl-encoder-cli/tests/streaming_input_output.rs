// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! End-to-end smoke tests for the `--streaming-input` /
//! `--streaming-output` CLI flags (A1 audit cleanup chunk 3).
//!
//! The streaming flags route the encode through the
//! [`jxl_encoder::LossyEncoder`] / [`jxl_encoder::LosslessEncoder`]
//! `push_rows()` + `finish()` / `finish_to()` surface instead of the
//! bulk one-shot `EncodeRequest::encode()` path. The bitstreams MUST
//! be byte-identical to the bulk path on the eligible subset (lossy
//! VarDCT without rate-control/progressive/lf-frame and lossless
//! modular without lossy-palette).
//!
//! Mirrors `cli_passthrough_smoke.rs`'s test pattern: real `frymire.png`
//! crop, run `cjxl-rs` as a subprocess, hash the encoded bytes.

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
    // 192-row crop forces ≥ 3 chunks at STREAM_CHUNK_ROWS=64 so the loop
    // boundary in `encode_lossy_streaming` actually iterates.
    const CROP_W: u32 = 128;
    const CROP_H: u32 = 192;
    assert!(w >= CROP_W && h >= CROP_H, "fixture too small: {w}x{h}");
    let x0 = (w - CROP_W) / 2;
    let y0 = (h - CROP_H) / 2;
    let crop = image::imageops::crop_imm(&img, x0, y0, CROP_W, CROP_H).to_image();

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/tests-tmp/streaming_input_output");
    std::fs::create_dir_all(&out_dir).expect("create_dir_all");
    let out_path = out_dir.join(format!("{name}.png"));
    crop.save(&out_path).expect("save crop");
    out_path
}

/// Run `cjxl-rs` with the given args and return the encoded bytes.
fn run_cjxl(input_png: &Path, label: &str, extra_args: &[&str]) -> Vec<u8> {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/tests-tmp/streaming_input_output");
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
    let is_container = &bytes[..12] == b"\x00\x00\x00\x0CJXL \r\n\x87\n";
    let is_codestream = bytes[0] == 0xFF && bytes[1] == 0x0A;
    assert!(
        is_container || is_codestream,
        "{label}: missing JXL signature"
    );
    bytes
}

/// `--streaming-input` on a lossy VarDCT encode must produce a byte-
/// identical bitstream to the bulk one-shot path. The streaming
/// encoder shares the same `LossyConfig` so the dispatch is identical;
/// only the input feed differs (`push_rows()` vs `EncodeRequest::encode()`
/// which calls a one-shot internal equivalent).
#[test]
fn streaming_input_lossy_byte_identical_to_bulk() {
    let png = write_cropped_fixture("lossy");
    let bulk = run_cjxl(&png, "lossy_bulk", &["-d", "1.0"]);
    let streaming = run_cjxl(&png, "lossy_streaming", &["-d", "1.0", "--streaming-input"]);
    assert_eq!(
        fingerprint(&bulk),
        fingerprint(&streaming),
        "lossy streaming-input must match bulk byte-for-byte \
         (bulk={} bytes, streaming={} bytes)",
        bulk.len(),
        streaming.len()
    );
}

/// `--streaming-input` on a lossless modular encode must produce a
/// byte-identical bitstream to the bulk one-shot path.
#[test]
fn streaming_input_lossless_byte_identical_to_bulk() {
    let png = write_cropped_fixture("lossless");
    let bulk = run_cjxl(&png, "lossless_bulk", &["-d", "0"]);
    let streaming = run_cjxl(
        &png,
        "lossless_streaming",
        &["-d", "0", "--streaming-input"],
    );
    assert_eq!(
        fingerprint(&bulk),
        fingerprint(&streaming),
        "lossless streaming-input must match bulk byte-for-byte \
         (bulk={} bytes, streaming={} bytes)",
        bulk.len(),
        streaming.len()
    );
}

/// `--streaming-output` combined with `--streaming-input` writes the
/// bytes via `LossyEncoder::finish_to(BufWriter<File>)` instead of
/// buffering a `Vec<u8>`. The output file must still be byte-identical
/// to the in-memory streaming path.
#[test]
fn streaming_output_lossy_writes_same_bytes_as_in_memory() {
    let png = write_cropped_fixture("stream_out_lossy");
    let in_memory = run_cjxl(
        &png,
        "lossy_stream_in_memory",
        &["-d", "1.0", "--streaming-input"],
    );
    let direct = run_cjxl(
        &png,
        "lossy_stream_direct",
        &["-d", "1.0", "--streaming-input", "--streaming-output"],
    );
    assert_eq!(
        fingerprint(&in_memory),
        fingerprint(&direct),
        "lossy --streaming-output must produce the same bytes as the \
         in-memory streaming finish() path \
         (in_memory={} bytes, direct={} bytes)",
        in_memory.len(),
        direct.len()
    );
}

/// `--streaming-output` combined with `--streaming-input` on the
/// lossless path must produce the same bytes as the in-memory streaming
/// path. Covers [`LosslessEncoder::finish_to`].
#[test]
fn streaming_output_lossless_writes_same_bytes_as_in_memory() {
    let png = write_cropped_fixture("stream_out_lossless");
    let in_memory = run_cjxl(
        &png,
        "lossless_stream_in_memory",
        &["-d", "0", "--streaming-input"],
    );
    let direct = run_cjxl(
        &png,
        "lossless_stream_direct",
        &["-d", "0", "--streaming-input", "--streaming-output"],
    );
    assert_eq!(
        fingerprint(&in_memory),
        fingerprint(&direct),
        "lossless --streaming-output must produce the same bytes as the \
         in-memory streaming finish() path \
         (in_memory={} bytes, direct={} bytes)",
        in_memory.len(),
        direct.len()
    );
}

/// `--streaming-input` paired with `--progressive` must fall back to
/// the bulk path with a warning (progressive isn't supported on the
/// streaming encoder yet). Output should still be a valid bitstream
/// and match the bulk `--progressive` output.
#[test]
fn streaming_input_progressive_falls_back_to_bulk() {
    let png = write_cropped_fixture("stream_progressive_fallback");
    let bulk_progressive = run_cjxl(&png, "bulk_progressive", &["-d", "1.0", "--progressive"]);
    let streaming_progressive = run_cjxl(
        &png,
        "stream_progressive",
        &["-d", "1.0", "--progressive", "--streaming-input"],
    );
    assert_eq!(
        fingerprint(&bulk_progressive),
        fingerprint(&streaming_progressive),
        "--streaming-input + --progressive must fall back to bulk path \
         and produce identical bytes (bulk={} bytes, streaming={} bytes)",
        bulk_progressive.len(),
        streaming_progressive.len()
    );
}
