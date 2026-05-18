// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-3 dispatch tests for `Buffering` (jxl-encoder#11).
//!
//! Chunk 3 lands the per-DC-group precompute loop driver
//! ([`fill_dc_group_state_whole_image`] now iterates
//! [`compute_dc_group`] over real `DC_GROUP_DIM`-aligned regions).
//! Every `Buffering` variant routes through this same per-region loop
//! — the chunk-3 invariant is that output bytes are bit-identical
//! across all variants.
//!
//! These tests pin that invariant on real images (including a
//! deliberately multi-DC-group input to exercise the loop driver with
//! more than one iteration), so future chunks (4/5) can replace the
//! per-region implementation step by step without breaking the
//! variant-equivalence contract.

use jxl_encoder::{Buffering, LosslessConfig, LossyConfig, PixelLayout};

const VARIANTS: [Buffering; 5] = [
    Buffering::Auto,
    Buffering::FullBuffered,
    Buffering::Threshold2048,
    Buffering::BufferedOutput,
    Buffering::FullStreaming,
];

fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / (w - 1).max(1)) as u8;
            out[i + 1] = (y * 255 / (h - 1).max(1)) as u8;
            // Bake in some content variation so AC strategy / CfL
            // aren't trivial — important for the per-DC-group path to
            // exercise non-default codepaths.
            out[i + 2] = ((x ^ y) & 0xff) as u8;
        }
    }
    out
}

#[test]
fn lossy_buffering_variants_byte_identical_single_dc_group() {
    // 256×256 = one PASS group, well within one DC group (2048²).
    let w = 256u32;
    let h = 256u32;
    let pixels = gradient_rgb(w, h);
    let baseline = LossyConfig::new(1.0)
        .with_buffering(Buffering::FullBuffered)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("FullBuffered encode failed");

    for variant in VARIANTS {
        let bytes = LossyConfig::new(1.0)
            .with_buffering(variant)
            .encode(&pixels, w, h, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{variant:?} encode failed: {e:?}"));
        assert_eq!(
            bytes,
            baseline,
            "lossy single-DC-group: {variant:?} differs from FullBuffered \
             ({} vs {} bytes)",
            bytes.len(),
            baseline.len()
        );
    }
}

#[test]
fn lossy_buffering_variants_byte_identical_multi_dc_group() {
    // 2560×2560 > 2048×2048 → multiple DC groups, exercises the loop
    // driver with at least 2 iterations in each dimension.
    //
    // libjxl `--buffering 2` is the default for images this size;
    // chunk 3 must keep `BufferedOutput` byte-identical to
    // `FullBuffered` (chunk 5 will allow a small compression delta
    // when actual buffered-output streaming lands).
    let w = 2560u32;
    let h = 2560u32;
    let pixels = gradient_rgb(w, h);
    let baseline = LossyConfig::new(2.0)
        .with_buffering(Buffering::FullBuffered)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("FullBuffered encode failed");

    for variant in VARIANTS {
        let bytes = LossyConfig::new(2.0)
            .with_buffering(variant)
            .encode(&pixels, w, h, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{variant:?} encode failed: {e:?}"));
        assert_eq!(
            bytes,
            baseline,
            "lossy multi-DC-group: {variant:?} differs from FullBuffered \
             ({} vs {} bytes)",
            bytes.len(),
            baseline.len()
        );
    }
}

#[test]
fn lossless_buffering_variants_byte_identical_small() {
    let w = 64u32;
    let h = 64u32;
    let pixels = gradient_rgb(w, h);
    let baseline = LosslessConfig::new()
        .with_buffering(Buffering::FullBuffered)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("FullBuffered encode failed");

    for variant in VARIANTS {
        let bytes = LosslessConfig::new()
            .with_buffering(variant)
            .encode(&pixels, w, h, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{variant:?} encode failed: {e:?}"));
        assert_eq!(
            bytes,
            baseline,
            "lossless small: {variant:?} differs from FullBuffered \
             ({} vs {} bytes)",
            bytes.len(),
            baseline.len()
        );
    }
}

#[test]
fn lossy_per_region_loop_runs_for_multi_dc_group_input() {
    // Smoke test that the per-DC-group loop in
    // `fill_dc_group_state_whole_image` doesn't panic, OOB, or
    // hash-mismatch when iterated across multiple DC groups. A 2049×16
    // image is the minimum case where `xsize_dc_groups > 1` — it
    // triggers the per-region loop with 2 iterations in the X
    // direction, exercising the assembly path that `copy_region_from`
    // and the CfL slice writeback take.
    let w = 2049u32;
    let h = 16u32;
    let pixels = gradient_rgb(w, h);
    let bytes = LossyConfig::new(1.0)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    assert!(!bytes.is_empty(), "non-trivial input produced empty output");
    // Header sniff — every JXL codestream starts with 0xFF 0x0A.
    assert_eq!(
        &bytes[..2],
        &[0xff, 0x0a],
        "missing JXL codestream signature"
    );
}
