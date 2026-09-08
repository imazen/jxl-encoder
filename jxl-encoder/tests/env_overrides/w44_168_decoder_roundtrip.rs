// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-168 (Smart-Zenjxl chunk 5) integration tests.
//!
//! These tests verify the content-aware butteraugli_iters dispatch:
//! 1. `EncoderStrategy::Libjxl` produces byte-identical output regardless
//!    of `JXL_W44_168_MODE` env var (because `adaptive_buttloop_iters =
//!    false` on Libjxl — the gate cannot promote iters).
//! 2. The W44-168 gate does NOT affect images where the proxies don't
//!    fire (e.g. `1189261.png` at e=7 with smooth-only thresholds).
//! 3. Explicit opt-in modes fully decode on a real photo with jxl-rs,
//!    djxl v0.12 and jxl-oxide. Named presets leave this gate disabled.

#![cfg(all(feature = "butteraugli-loop", feature = "ssim2-loop"))]

#[path = "../../examples/distance_targeting_probe/decode.rs"]
mod full_decode;

use image::GenericImageView;
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};

fn load_image(name: &str) -> (u32, u32, Vec<u8>) {
    let path = crate::corpus_file(&format!("CID22/CID22-512/validation/{name}"));
    let img = image::open(&path).expect("decode png");
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    (w, h, rgb)
}

fn encode_with_env(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    strategy: EncoderStrategy,
    mode_env: Option<&str>,
) -> Vec<u8> {
    let prev = std::env::var("JXL_W44_168_MODE").ok();
    // SAFETY: tests are single-threaded; we save+restore.
    match mode_env {
        // SAFETY: every caller holds crate::env_serial() across this helper.
        Some(v) => unsafe { std::env::set_var("JXL_W44_168_MODE", v) },
        // SAFETY: every caller holds crate::env_serial() across this helper.
        None => unsafe { std::env::remove_var("JXL_W44_168_MODE") },
    }
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(2)
        .with_strategy(strategy);
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode ok");
    match prev {
        // SAFETY: every caller holds crate::env_serial() across this helper.
        Some(v) => unsafe { std::env::set_var("JXL_W44_168_MODE", v) },
        // SAFETY: every caller holds crate::env_serial() across this helper.
        None => unsafe { std::env::remove_var("JXL_W44_168_MODE") },
    }
    full_decode::verify_jxl_rs(&bytes, w as usize, h as usize);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/w44-168-decode")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mode.jxl");
    std::fs::write(&path, &bytes).unwrap();
    let decoded = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .args(["--disable_output", "--num_threads=1"])
        .output()
        .unwrap();
    assert!(
        decoded.status.success(),
        "djxl: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    bytes
}

/// `EncoderStrategy::Libjxl` MUST produce byte-identical output regardless
/// of the `JXL_W44_168_MODE` env var (the gate is disabled when
/// `adaptive_buttloop_iters = false`).
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_168_libjxl_strategy_byte_identical_regardless_of_env() {
    let _env_serial = crate::env_serial();
    let (w, h, rgb) = load_image("1418519.png");
    // Pick an effort/distance where the buttloop normally fires (e8 at d=5):
    // - Mode A: iters=2 (e8 baseline)
    // - Mode B: would decrement to 1 if `adaptive_buttloop_iters=true`
    // - On Libjxl, gate is off so all modes must produce the same iters=2
    let a = encode_with_env(&rgb, w, h, 8, 5.0, EncoderStrategy::Libjxl, Some("A"));
    let b = encode_with_env(&rgb, w, h, 8, 5.0, EncoderStrategy::Libjxl, Some("B"));
    let c = encode_with_env(&rgb, w, h, 8, 5.0, EncoderStrategy::Libjxl, Some("C"));
    let d = encode_with_env(&rgb, w, h, 8, 5.0, EncoderStrategy::Libjxl, Some("D"));
    assert_eq!(
        a, b,
        "Libjxl strategy must be byte-identical with Mode B vs Mode A"
    );
    assert_eq!(
        a, c,
        "Libjxl strategy must be byte-identical with Mode C vs Mode A"
    );
    assert_eq!(
        a, d,
        "Libjxl strategy must be byte-identical with Mode D vs Mode A"
    );
}

/// `1189261.png` at e=8 with Mode B: discriminator IS the question —
/// we just verify all modes decode cleanly (proves the dispatch chain
/// doesn't produce broken bitstreams). Byte-equality isn't asserted
/// because the discriminator value depends on the image content.
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_168_custom_all_modes_decode_clean() {
    let _env_serial = crate::env_serial();
    use std::io::Cursor;
    let (w, h, rgb) = load_image("1418519.png");
    for mode in &["A", "B", "C", "D"] {
        let bytes = encode_with_env(&rgb, w, h, 8, 5.0, legacy_iteration_strategy(), Some(mode));
        let reader = Cursor::new(&bytes);
        let mut img = jxl_oxide::JxlImage::builder()
            .read(reader)
            .unwrap_or_else(|e| panic!("jxl-oxide read failed for mode {}: {:?}", mode, e));
        img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let _render = img
            .render_frame(0)
            .unwrap_or_else(|e| panic!("jxl-oxide render failed for mode {}: {:?}", mode, e));
    }
}

/// `1189261.png` at e=7 with Mode B (SmoothSkip): does NOT fire because
/// Mode B is `e >= 8` only. Must be byte-identical to Mode A baseline
/// regardless of whether the image looks smooth.
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_168_mode_b_does_not_fire_at_e7() {
    let _env_serial = crate::env_serial();
    let (w, h, rgb) = load_image("1189261.png");
    let a = encode_with_env(&rgb, w, h, 7, 4.0, legacy_iteration_strategy(), Some("A"));
    let b = encode_with_env(&rgb, w, h, 7, 4.0, legacy_iteration_strategy(), Some("B"));
    assert_eq!(
        a, b,
        "Mode B (SmoothSkip) is e>=8 only — e7 cells must be byte-identical to baseline"
    );
}

fn legacy_iteration_strategy() -> EncoderStrategy {
    EncoderStrategy::Custom(Box::new(jxl_encoder::api::EncoderImprovementsCustom {
        adaptive_buttloop_iters: true,
        ..Default::default()
    }))
}
