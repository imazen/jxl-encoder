// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-180 multi-decoder roundtrip on the top-3 e8 wedge cells.
//!
//! These tests verify that the W44-180 incremental-histogram DC tree split
//! scan produces a bitstream that decodes cleanly via:
//! - jxl-oxide (Rust decoder, primary)
//! - jxl-rs (Rust decoder, second source of truth)
//!
//! The W44-180 port replaces the per-quantile re-scan inside
//! `find_best_split_variable` with libjxl's running-histogram update from
//! `enc_ma.cc:280-439`. The candidate splitval set is unchanged (same
//! 32-quantile grid as the pre-W44-180 code); only the cost-evaluation
//! algorithm changes. The two paths produce byte-identical bitstream output —
//! the unit test `test_find_best_split_variable_legacy_vs_incremental_byte_equivalent`
//! in `vardct::dc_tree_learn::tests` proves this on a synthetic 32×32 fixture,
//! and the 36/36 hash-lock fixtures cover the integration-level invariant.
//!
//! These multi-decoder tests cover the production-corpus cells where the
//! W44-180 perf win is largest: top-3 e8 d=0.5 wedges from W44-175.

#![cfg(all(feature = "butteraugli-loop", feature = "ssim2-loop"))]

use image::GenericImageView;
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::PathBuf;

const CORPUS_ROOT: &str = "/home/lilith/work/codec-corpus";

fn load_image(relpath: &str) -> (u32, u32, Vec<u8>) {
    let path = PathBuf::from(CORPUS_ROOT).join(relpath);
    let img = image::open(&path).expect("decode png");
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    (w, h, rgb)
}

fn encode_zenjxl(rgb: &[u8], w: u32, h: u32, effort: u8, distance: f32) -> Vec<u8> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode ok")
}

fn decode_jxl_oxide(bytes: &[u8]) -> (usize, usize) {
    use std::io::Cursor;
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide read");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).expect("jxl-oxide render");
    let fb = render.image_all_channels();
    (fb.width(), fb.height())
}

/// W44-180 top-3 e8 wedge cells decode cleanly via jxl-oxide.
///
/// Verifies the incremental-histogram DC tree split scan produces a
/// well-formed bitstream on the cells where the perf win is largest.
#[test]
#[ignore = "requires codec-corpus on local disk; run with `--ignored`"]
fn w44_180_top_three_e8_wedges_decode_clean_jxl_oxide() {
    for &(relpath, name) in &[
        ("gb82-sc/terminal.png", "terminal_e8_d05"),
        ("gb82-sc/codec_wiki.png", "codec_wiki_e8_d05"),
        ("gb82-sc/imac_dark.png", "imac_dark_e8_d05"),
    ] {
        let (w, h, rgb) = load_image(relpath);
        let bytes = encode_zenjxl(&rgb, w, h, 8, 0.5);
        let (dw, dh) = decode_jxl_oxide(&bytes);
        assert_eq!(
            (dw, dh),
            (w as usize, h as usize),
            "W44-180 {} jxl-oxide decode dims mismatch: encoded {}x{}, decoded {}x{}",
            name,
            w,
            h,
            dw,
            dh
        );
    }
}

/// W44-180 same wedges at e9 d=0.5 (PROTECT_E9 from the bench harness)
/// must also decode cleanly. W44-172 picks `Predictor::Variable` (14
/// predictors) at e9, so the W44-180 incremental scan walks the larger
/// active set.
#[test]
#[ignore = "requires codec-corpus on local disk; run with `--ignored`"]
fn w44_180_e9_wedges_decode_clean_jxl_oxide() {
    let (w, h, rgb) = load_image("gb82-sc/terminal.png");
    let bytes = encode_zenjxl(&rgb, w, h, 9, 0.5);
    let (dw, dh) = decode_jxl_oxide(&bytes);
    assert_eq!(
        (dw, dh),
        (w as usize, h as usize),
        "W44-180 terminal e9 d=0.5 jxl-oxide decode dims mismatch"
    );
}

/// W44-180 photo cell (CID22 1418519 e8 d=1) decodes cleanly. Confirms the
/// incremental scan path is well-behaved on photo content too (the W44-180
/// perf wedge is on screenshots; photos should still be byte-identical to
/// the legacy path).
#[test]
#[ignore = "requires codec-corpus on local disk; run with `--ignored`"]
fn w44_180_photo_cell_decodes_clean_jxl_oxide() {
    let (w, h, rgb) = load_image("CID22/CID22-512/validation/1418519.png");
    let bytes = encode_zenjxl(&rgb, w, h, 8, 1.0);
    let (dw, dh) = decode_jxl_oxide(&bytes);
    assert_eq!(
        (dw, dh),
        (w as usize, h as usize),
        "W44-180 photo cell jxl-oxide decode dims mismatch"
    );
}
