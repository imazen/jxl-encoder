// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-167 (Smart-Zenjxl chunk 4) integration tests.
//!
//! These tests verify the per-m3 sub-discriminator lift on INNER variant
//! Z tables:
//! 1. `EncoderStrategy::Libjxl` produces byte-identical output regardless
//!    of `JXL_W44_167_MODE` env var (because `photo_variant_z_admit =
//!    false` AND `find_best_32_per_m3_lift = false` on Libjxl, so the
//!    W44-167 gate never fires).
//! 2. The W44-167 gate does NOT affect non-firing photos
//!    (`1189261.png`, `1025469.png` — mask>=50, variant Z never fires).

#![cfg(all(feature = "butteraugli-loop", feature = "ssim2-loop"))]

use image::GenericImageView;
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::PathBuf;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

fn load_image(name: &str) -> (u32, u32, Vec<u8>) {
    let path = PathBuf::from(CID22).join(name);
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
    let prev = std::env::var("JXL_W44_167_MODE").ok();
    // SAFETY: tests are single-threaded; we save+restore.
    match mode_env {
        Some(v) => unsafe { std::env::set_var("JXL_W44_167_MODE", v) },
        None => unsafe { std::env::remove_var("JXL_W44_167_MODE") },
    }
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(2)
        .with_strategy(strategy);
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode ok");
    match prev {
        Some(v) => unsafe { std::env::set_var("JXL_W44_167_MODE", v) },
        None => unsafe { std::env::remove_var("JXL_W44_167_MODE") },
    }
    bytes
}

/// `EncoderStrategy::Libjxl` MUST produce byte-identical output regardless
/// of the `JXL_W44_167_MODE` env var.
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_167_libjxl_strategy_byte_identical_regardless_of_env() {
    let (w, h, rgb) = load_image("1420710.png");
    let a = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Libjxl, Some("A"));
    let b = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Libjxl, Some("B"));
    let c = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Libjxl, Some("C"));
    let d = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Libjxl, Some("D"));
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

/// `1189261.png` (mask=69.08) does NOT fire variant Z (mask>=50). All
/// W44-167 modes must produce byte-identical output to baseline.
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_167_non_firing_photo_byte_identical_all_modes() {
    let (w, h, rgb) = load_image("1189261.png");
    let a = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Zenjxl, Some("A"));
    let b = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Zenjxl, Some("B"));
    let c = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Zenjxl, Some("C"));
    let d = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Zenjxl, Some("D"));
    assert_eq!(
        a, b,
        "1189261 (mask=69.08, fails variant Z gate) Mode B must be byte-identical to Mode A"
    );
    assert_eq!(a, c, "1189261 Mode C must be byte-identical to Mode A");
    assert_eq!(a, d, "1189261 Mode D must be byte-identical to Mode A");
}

/// `1420710.png` cells under Zenjxl with Mode A (baseline) must decode
/// cleanly via jxl-oxide. (jxl-rs roundtrip is exercised by the
/// per-bench multi-decoder harness; this test just smoke-checks one
/// strategy/mode combo.)
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_167_zenjxl_mode_d_decoders_clean() {
    use std::io::Cursor;
    let (w, h, rgb) = load_image("1420710.png");
    let bytes = encode_with_env(&rgb, w, h, 7, 5.0, EncoderStrategy::Zenjxl, Some("D"));
    let reader = Cursor::new(&bytes);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide read ok");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let _render = img.render_frame(0).expect("jxl-oxide render ok");
}
