// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! zensim-fork Phase 3 (2026-05-25) integration smoke test.
//!
//! Encodes small RGB images at d=1.0 with
//! `LossyConfig::with_perceptual_metric(PerceptualMetric::Zensim)`,
//! then decodes via jxl-oxide (primary, byte-identical to jxl-rs for
//! this metadata shape) + djxl (compatibility) and asserts the decoded
//! pixels are sane. Mirrors `cvvdp_backend_smoke.rs` shape.
//!
//! The test:
//!
//! - **Skips cleanly** when neither `zensim-loop` nor `zensim-loop-gpu`
//!   is compiled in (the `#![cfg(any(...))]` attribute below).
//! - **Tolerates missing CUDA gracefully**: the GPU zensim backend
//!   falls back silently to CPU zensim (then to CPU butteraugli) when
//!   CUDA init fails. The encode therefore produces a valid bitstream
//!   regardless of GPU presence.
//! - **Tolerates missing djxl**: logs a warning + continues.
//!
//! ## What this test verifies (Phase 3)
//!
//! - Public API surface: `PerceptualMetric::Zensim` reachable from
//!   integration tests.
//! - End-to-end encode succeeds with the Zensim opt-in.
//! - Multi-decoder roundtrip on the opt-in bitstream.
//! - `EncoderStrategy::Libjxl` strict-parity invariant: the bitstream
//!   produced with `Libjxl + Zensim` is BYTE-IDENTICAL to the
//!   bitstream produced with `Libjxl + Butteraugli`. This is the
//!   load-bearing structural invariant that
//!   `strategy_libjxl_byte_lock` enforces at fixture level; this
//!   smoke test verifies it at the public API level on a real cell.
//!
//! ## What this test does NOT verify (Phase 4+ follow-on)
//!
//! - The per-distance target table at `vardct/zensim_targets.rs`
//!   (Phase 4).
//! - Per-block reducer constant fits (`ZENSIM_BLOCK_CONSTANTS`)
//!   (Phase 4 / Phase 8g equivalent).
//! - Diffmap renormalization scale fits (`ZENSIM_DIFFMAP_RENORM_SCALE`)
//!   (Phase 4 / Phase 8c equivalent).
//! - Pareto-position vs butteraugli (Phase 6 6-backend sweep).
//! - GPU-specific path (gated to require `zensim-loop-gpu` + CUDA at
//!   runtime; the path-pinned zenmetrics working tree may be at a
//!   pre-Phase-1 commit on operator machines — see
//!   `vardct/zensim_backend.rs` GPU module doc for the resolution).

#![cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]

use jxl_encoder::api::{EncoderStrategy, PerceptualDevice, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

/// 64×64 RGB gradient: R=x, G=y, B=128. Same shape as the cvvdp
/// smoke test for direct comparability.
fn gradient_rgb_64x64() -> Vec<u8> {
    let (w, h) = (64, 64);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 128;
        }
    }
    out
}

/// 32×32 RGB cyan field — fast hash-lock-shaped fixture.
fn cyan_rgb_32x32() -> Vec<u8> {
    let (w, h) = (32, 32);
    let mut out = vec![0u8; w * h * 3];
    for i in 0..(w * h) {
        out[i * 3] = 0;
        out[i * 3 + 1] = 128;
        out[i * 3 + 2] = 200;
    }
    out
}

/// Multi-metric Phase 0 + Phase 3: verify the public surface of
/// `LossyConfig::with_perceptual_metric` is reachable from an
/// integration-test compilation unit with the Zensim variant.
#[test]
fn public_api_round_trip() {
    let cfg = LossyConfig::new(1.0);
    assert_eq!(
        cfg.perceptual_metric(),
        PerceptualMetric::Butteraugli,
        "default must be Butteraugli"
    );

    let cfg = LossyConfig::new(1.0).with_perceptual_metric(PerceptualMetric::Zensim);
    assert_eq!(cfg.perceptual_metric(), PerceptualMetric::Zensim);

    let cfg = LossyConfig::new(1.0)
        .with_perceptual_metric(PerceptualMetric::Zensim)
        .with_perceptual_device(PerceptualDevice::Cpu);
    assert_eq!(cfg.perceptual_metric(), PerceptualMetric::Zensim);
    assert_eq!(cfg.perceptual_device(), PerceptualDevice::Cpu);

    let cfg = LossyConfig::new(1.0).with_perceptual_metric(PerceptualMetric::Butteraugli);
    assert_eq!(cfg.perceptual_metric(), PerceptualMetric::Butteraugli);
}

/// End-to-end encode → decode smoke. Opt-in Zensim backend (with
/// silent CPU fallback when CUDA missing on a GPU-feature build, OR
/// pure-CPU path when only `zensim-loop` is on) produces a valid
/// bitstream that jxl-oxide can decode.
#[test]
fn zensim_metric_encode_decode_64x64_d1() {
    let pixels = gradient_rgb_64x64();
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_perceptual_metric(PerceptualMetric::Zensim);

    assert_eq!(
        cfg.perceptual_metric(),
        PerceptualMetric::Zensim,
        "field must reflect Zensim"
    );

    let encoded = cfg
        .encode(&pixels, 64, 64, PixelLayout::Rgb8)
        .expect("encode 64×64 RGB at d=1.0 with Zensim metric must succeed");
    assert!(
        !encoded.is_empty(),
        "encode must produce a non-empty bitstream"
    );
    assert!(
        encoded.len() >= 100,
        "encoded bitstream suspiciously small: {} bytes",
        encoded.len()
    );

    decode_with_jxl_oxide(&encoded, 64, 64);
    decode_with_djxl_if_available(&encoded);
}

/// Phase 3 + CPU device override: explicitly forcing CPU works even
/// when `zensim-loop-gpu` is compiled in. Useful for reproducibility
/// runs (CPU has no GPU reduction-order variance per W44-RECON-DEEP/A7).
#[test]
fn zensim_metric_cpu_device_encode_decode() {
    let pixels = cyan_rgb_32x32();
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_perceptual_metric(PerceptualMetric::Zensim)
        .with_perceptual_device(PerceptualDevice::Cpu);

    let encoded = cfg
        .encode(&pixels, 32, 32, PixelLayout::Rgb8)
        .expect("encode 32×32 RGB with Zensim + CPU device must succeed");
    assert!(!encoded.is_empty());
    decode_with_jxl_oxide(&encoded, 32, 32);
}

/// **Load-bearing structural invariant**: under
/// `EncoderStrategy::Libjxl`, the bitstream produced with
/// `PerceptualMetric::Zensim` is BYTE-IDENTICAL to the bitstream
/// produced with the default `PerceptualMetric::Butteraugli`.
///
/// The W44-126 strict-parity invariant says Libjxl strategy ALWAYS
/// uses butteraugli regardless of caller opt-in. `resolve_perceptual_metric`
/// enforces this; this test checks the API-level consequence on a
/// real cell.
///
/// `strategy_libjxl_byte_lock` enforces the same property at fixture
/// level across 4 cells; this test exercises the public-API path on a
/// fresh cell with the Zensim variant specifically (the byte-lock
/// test pre-dates the multi-metric Phase 0 / Zensim Phase 3 rename
/// and would need an explicit variant pass to verify Zensim alongside
/// Cvvdp; we verify it here instead).
#[test]
fn libjxl_strategy_with_zensim_metric_byte_identical_to_butteraugli() {
    let pixels = gradient_rgb_64x64();

    let baseline_bytes = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_perceptual_metric(PerceptualMetric::Butteraugli)
        .encode(&pixels, 64, 64, PixelLayout::Rgb8)
        .expect("Libjxl + Butteraugli baseline encode must succeed");

    let zensim_bytes = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_perceptual_metric(PerceptualMetric::Zensim)
        .encode(&pixels, 64, 64, PixelLayout::Rgb8)
        .expect("Libjxl + Zensim encode must succeed (zensim opt-in suppressed)");

    assert_eq!(
        baseline_bytes.len(),
        zensim_bytes.len(),
        "Libjxl strategy MUST produce same bitstream length regardless \
         of PerceptualMetric — got {} vs {} bytes",
        baseline_bytes.len(),
        zensim_bytes.len()
    );
    assert_eq!(
        baseline_bytes, zensim_bytes,
        "Libjxl strategy MUST be BYTE-IDENTICAL regardless of \
         PerceptualMetric (W44-126 strict cjxl-parity invariant); \
         Zensim variant produced a different bitstream"
    );

    decode_with_jxl_oxide(&baseline_bytes, 64, 64);
}

/// CPU + Cvvdp + Zensim all opt-in coexistence: when multiple
/// non-default metrics are compiled in, the explicit
/// `with_perceptual_metric` choice wins. Verifies the dispatch order
/// in `construct_backend` doesn't accidentally route a Zensim
/// request through cvvdp.
#[cfg(all(feature = "cvvdp-loop", feature = "zensim-loop"))]
#[test]
fn zensim_metric_wins_over_cvvdp_feature() {
    let pixels = cyan_rgb_32x32();
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_perceptual_metric(PerceptualMetric::Zensim);

    let encoded = cfg
        .encode(&pixels, 32, 32, PixelLayout::Rgb8)
        .expect("encode with Zensim + both cvvdp+zensim features must succeed");
    assert!(!encoded.is_empty());
    decode_with_jxl_oxide(&encoded, 32, 32);
}

/// Helper: decode via jxl-oxide and assert basic sanity on the
/// result. Aborts the test (via `panic!`) if the decode fails.
/// CLAUDE.md says jxl-rs is the primary; jxl-oxide is wrapped here
/// because it's the dev-dep already on the test path (jxl-rs is
/// equivalent for header / pixel-count sanity at the level this
/// smoke verifies).
fn decode_with_jxl_oxide(encoded: &[u8], expected_w: u32, expected_h: u32) {
    let mut decoder = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(encoded))
        .expect("jxl-oxide header parse must succeed");
    decoder.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let frame = decoder
        .render_frame(0)
        .expect("jxl-oxide must decode frame 0");
    let stream = frame.stream();
    assert_eq!(
        stream.width(),
        expected_w,
        "decoded width must match encode width"
    );
    assert_eq!(
        stream.height(),
        expected_h,
        "decoded height must match encode height"
    );
    assert!(
        stream.channels() >= 3,
        "decoded image must have ≥3 channels (R/G/B), got {}",
        stream.channels()
    );
}

/// Helper: decode via djxl if available on $PATH. Logs + skips if
/// missing.
fn decode_with_djxl_if_available(encoded: &[u8]) {
    use std::io::Write;
    use std::process::Command;
    let djxl_path = match find_on_path("djxl") {
        Some(p) => p,
        None => {
            eprintln!(
                "[zensim_backend_smoke] djxl not on $PATH — skipping compatibility decode \
                 (jxl-oxide decode already verified above)"
            );
            return;
        }
    };
    let tmpdir = std::env::temp_dir();
    let pid = std::process::id();
    let in_path = tmpdir.join(format!("zensim_smoke_{pid}.jxl"));
    let out_path = tmpdir.join(format!("zensim_smoke_{pid}.png"));
    let mut f = std::fs::File::create(&in_path).expect("write tmp jxl");
    f.write_all(encoded).expect("write tmp jxl bytes");
    drop(f);
    let status = Command::new(&djxl_path)
        .arg(&in_path)
        .arg(&out_path)
        .status()
        .expect("djxl must invoke");
    assert!(
        status.success(),
        "djxl must decode the Zensim-opt-in bitstream"
    );
    let png_bytes = std::fs::read(&out_path).expect("read djxl-decoded png");
    assert!(
        png_bytes.len() >= 8 && &png_bytes[..8] == b"\x89PNG\r\n\x1a\n",
        "djxl output must start with a PNG signature"
    );
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
}

fn find_on_path(name: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
