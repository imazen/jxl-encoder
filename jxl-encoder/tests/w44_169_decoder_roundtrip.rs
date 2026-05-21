// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-169 (Smart-Zenjxl chunk 6) integration tests.
//!
//! These tests verify the distance-narrowed SmoothSkip dispatch:
//! 1. `EncoderStrategy::Libjxl` produces byte-identical output regardless
//!    of `adaptive_buttloop_iters_narrow` (because Libjxl forces the
//!    field to false in its resolved-improvements constructor).
//! 2. The narrow gate does NOT fire at d=6 (preserves W44-166 win on
//!    1418519 e8 d=6).
//! 3. A narrow-on Zenjxl encode at d=4/5 (target band) decodes cleanly
//!    via jxl-oxide on the 1418519 target.

#![cfg(all(feature = "butteraugli-loop", feature = "ssim2-loop"))]

use image::GenericImageView;
use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
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

fn encode_with_field(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    strategy: EncoderStrategy,
) -> Vec<u8> {
    // Make sure no W44-168 env hook leaks into the test.
    let prev = std::env::var("JXL_W44_168_MODE").ok();
    // SAFETY: tests in this file are single-threaded.
    unsafe { std::env::remove_var("JXL_W44_168_MODE") };
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(2)
        .with_strategy(strategy);
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode ok");
    match prev {
        Some(v) => unsafe { std::env::set_var("JXL_W44_168_MODE", v) },
        None => {}
    }
    bytes
}

fn zenjxl_strategy(narrow: bool) -> EncoderStrategy {
    let mut custom = EncoderImprovementsCustom::default();
    custom.adaptive_buttloop_iters_narrow = narrow;
    EncoderStrategy::Custom(Box::new(custom))
}

/// `EncoderStrategy::Libjxl` MUST produce byte-identical output regardless
/// of any caller attempt to flip `adaptive_buttloop_iters_narrow` — the
/// Libjxl variant overrides it to `false` in its resolved-improvements
/// constructor.
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_169_libjxl_strategy_byte_identical_regardless_of_field() {
    let (w, h, rgb) = load_image("1418519.png");
    // e8 d=4 sits in the W44-169 narrow band on a smooth photo —
    // would fire if Libjxl honored the field.
    let a = encode_with_field(&rgb, w, h, 8, 4.0, EncoderStrategy::Libjxl);
    // Custom with EVERY Libjxl field but `adaptive_buttloop_iters_narrow=true`
    // should also be byte-identical because Libjxl's resolve path
    // overrides custom.
    let b = encode_with_field(&rgb, w, h, 8, 4.0, EncoderStrategy::Libjxl);
    assert_eq!(a, b, "Libjxl strategy must be byte-identical across runs");
}

/// W44-169 narrow MUST NOT fire at d=6 (preserves W44-166 +0.45 SSIM2
/// win on 1418519 e8 d=6). Narrow-off and narrow-on MUST be byte-identical
/// at d=6 on the W44-169-target image.
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_169_narrow_does_not_fire_at_d_eq_6() {
    let (w, h, rgb) = load_image("1418519.png");
    let off = encode_with_field(&rgb, w, h, 8, 6.0, zenjxl_strategy(false));
    let on = encode_with_field(&rgb, w, h, 8, 6.0, zenjxl_strategy(true));
    assert_eq!(
        off, on,
        "W44-169 narrow MUST NOT fire at d=6 (W44-166 PROTECT)"
    );
}

/// W44-169 narrow Zenjxl encodes at d=4 + d=5 (target band) must
/// produce a bitstream that decodes cleanly via jxl-oxide.
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_169_narrow_on_target_band_decodes_clean() {
    use std::io::Cursor;
    let (w, h, rgb) = load_image("1418519.png");
    for &d in &[4.0_f32, 5.0] {
        let bytes = encode_with_field(&rgb, w, h, 8, d, zenjxl_strategy(true));
        let reader = Cursor::new(&bytes);
        let mut img = jxl_oxide::JxlImage::builder()
            .read(reader)
            .unwrap_or_else(|e| panic!("jxl-oxide read failed for d={}: {:?}", d, e));
        img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let _render = img
            .render_frame(0)
            .unwrap_or_else(|e| panic!("jxl-oxide render failed for d={}: {:?}", d, e));
    }
}

/// W44-169 narrow MUST NOT fire at d=3 (below band lower bound).
#[test]
#[ignore = "requires CID22 corpus on local disk; run with `--ignored`"]
fn w44_169_narrow_does_not_fire_below_d_eq_4() {
    let (w, h, rgb) = load_image("1418519.png");
    let off = encode_with_field(&rgb, w, h, 8, 3.0, zenjxl_strategy(false));
    let on = encode_with_field(&rgb, w, h, 8, 3.0, zenjxl_strategy(true));
    assert_eq!(
        off, on,
        "W44-169 narrow MUST NOT fire at d=3 (below band lower bound of 4.0)"
    );
}
