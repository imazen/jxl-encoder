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

// ── Chunk 6 (#11): Buffering-driven dispatch + WritableSeek ──────────────

/// `Buffering::BufferedOutput` at >2048² engages the per-region
/// precompute path via the chunk-5 helpers when going through
/// [`VarDctEncoder::encode_with_rate_control_config`] (which is the
/// only consumer of [`compute_with_budget_and_buffering`]). The default
/// `LossyConfig::encode()` goes through `encode_inner` (inline
/// precompute, no dispatch) and won't hit the routing — chunk 7 will
/// reshape that path.
///
/// Acceptance contract (chunk 6):
///
/// - **FP-drift-free pair**: `FullBuffered` / `Threshold2048` /
///   `Auto`-resolving-to-`FullBuffered` (≤2048² images) produce
///   byte-identical bitstreams via the whole-image precompute. We
///   assert this on a **256×256** input (well below the 2048²
///   threshold, so `Auto` resolves to `FullBuffered`).
/// - **per-region pair**: `BufferedOutput` / `FullStreaming` /
///   `Auto`-resolving-to-`BufferedOutput` (>2048² images) all engage
///   the per-region precompute path. The chunk-5 commit documents
///   that this path has a bounded FP drift (≤256 ULPs at per-function
///   unit-test level, ~0.1% bytes-level divergence in the worst case).
///   We assert size is within 1% of the whole-image baseline — that's
///   the rd-regression-budget contract — but do NOT require
///   byte-identity (a real chunk-7 follow-on would tighten the
///   per-region kernel to bit-exact and remove this allowance).
#[test]
#[cfg(feature = "rate-control")]
fn rate_control_buffering_dispatch_routes_correctly() {
    use jxl_encoder::vardct::{RateControlConfig, VarDctEncoder};

    // sRGB → linear converter used by the VarDctEncoder API.
    let to_lin = |c: u8| -> f32 {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };

    // ── Sub-threshold (256²): all variants byte-identical ────────────
    let (w_s, h_s) = (256usize, 256usize);
    let pixels_s_u8 = gradient_rgb(w_s as u32, h_s as u32);
    let linear_s: Vec<f32> = pixels_s_u8.iter().map(|&b| to_lin(b)).collect();
    let cfg = RateControlConfig::default();

    let mut enc_s = VarDctEncoder::new(2.0);
    enc_s.buffering = Buffering::FullBuffered;
    let (baseline_s, _) = enc_s
        .encode_with_rate_control_config(w_s, h_s, &linear_s, &cfg)
        .expect("sub-threshold FullBuffered rate-control encode failed");

    for variant in VARIANTS {
        let mut enc2 = VarDctEncoder::new(2.0);
        enc2.buffering = variant;
        let (bytes, _) = enc2
            .encode_with_rate_control_config(w_s, h_s, &linear_s, &cfg)
            .unwrap_or_else(|e| panic!("sub-threshold {variant:?} encode failed: {e:?}"));
        assert_eq!(
            bytes, baseline_s,
            "sub-threshold (256²): {variant:?} differs from FullBuffered \
             — Auto should resolve to FullBuffered at this size and all \
             explicit variants go through compute_with_budget_and_buffering's \
             whole-image path",
        );
    }

    // ── Super-threshold (2560²): per-region path engaged ─────────────
    // Whole-image (FullBuffered, Threshold2048 below 2048² but here
    // resolves to per-region because 2560 > 2048; actually Threshold2048
    // at 2560² IS routed to per-region too per resolve_for semantics
    // — but our chunk-6 routing only flips on the resolved variant
    // being BufferedOutput or FullStreaming, NOT Threshold2048. So
    // Threshold2048 keeps the whole-image path at any size today, which
    // matches libjxl post-`032d39a` (level 1 == buffer everything ≤
    // 2048² else stream input; we treat the "stream input" bit as a
    // chunk-7 concern). The byte sizes below confirm the routing.
    let (w_l, h_l) = (2560usize, 2560usize);
    let pixels_l_u8 = gradient_rgb(w_l as u32, h_l as u32);
    let linear_l: Vec<f32> = pixels_l_u8.iter().map(|&b| to_lin(b)).collect();

    let mut enc_l = VarDctEncoder::new(2.0);
    enc_l.buffering = Buffering::FullBuffered;
    let (baseline_l, _) = enc_l
        .encode_with_rate_control_config(w_l, h_l, &linear_l, &cfg)
        .expect("super-threshold FullBuffered encode failed");
    assert!(!baseline_l.is_empty(), "baseline encode produced no bytes");

    let mut enc_thresh = VarDctEncoder::new(2.0);
    enc_thresh.buffering = Buffering::Threshold2048;
    let (bytes_thresh, _) = enc_thresh
        .encode_with_rate_control_config(w_l, h_l, &linear_l, &cfg)
        .expect("super-threshold Threshold2048 encode failed");
    assert_eq!(
        bytes_thresh, baseline_l,
        "super-threshold Threshold2048 must stay on the whole-image path \
         (chunk 6 only routes BufferedOutput/FullStreaming to per-region; \
         Threshold2048 still stays whole-image even >2048² until libjxl-\
         parity streaming-input lands in chunk 7+)"
    );

    // BufferedOutput + FullStreaming + Auto(>2048²) → per-region path
    for variant in [
        Buffering::BufferedOutput,
        Buffering::FullStreaming,
        Buffering::Auto,
    ] {
        let mut enc2 = VarDctEncoder::new(2.0);
        enc2.buffering = variant;
        let (bytes, _) = enc2
            .encode_with_rate_control_config(w_l, h_l, &linear_l, &cfg)
            .unwrap_or_else(|e| panic!("super-threshold {variant:?} encode failed: {e:?}"));
        assert!(!bytes.is_empty(), "{variant:?} produced no bytes");
        assert_eq!(
            &bytes[..2],
            &[0xff, 0x0a],
            "{variant:?} missing JXL codestream signature"
        );
        // chunk-5 FP-drift envelope: per-region path produces size
        // within 1% of whole-image baseline.
        let delta =
            (bytes.len() as i64 - baseline_l.len() as i64).abs() as f64 / baseline_l.len() as f64;
        assert!(
            delta < 0.01,
            "super-threshold {variant:?}: per-region size deviates {:.3}% \
             from FullBuffered baseline ({} vs {} bytes) — exceeds chunk-5 \
             FP-drift envelope (<1%)",
            delta * 100.0,
            bytes.len(),
            baseline_l.len()
        );
    }
}

/// `LossyEncoder::finish_to_seekable` round-trips bytes identically to
/// `finish()` (chunk-6: the seek capability is plumbed but the buffered-
/// output bytes are still produced in memory + written in one pass).
#[test]
fn finish_to_seekable_round_trips_identically_lossy() {
    use std::io::Cursor;

    let w = 64u32;
    let h = 64u32;
    let pixels = gradient_rgb(w, h);

    let mut enc_a = LossyConfig::new(1.0)
        .encoder(w, h, PixelLayout::Rgb8)
        .expect("encoder a failed");
    enc_a.push_rows(&pixels, h).expect("push_rows a failed");
    let baseline = enc_a.finish().expect("finish failed");

    let mut enc_b = LossyConfig::new(1.0)
        .encoder(w, h, PixelLayout::Rgb8)
        .expect("encoder b failed");
    enc_b.push_rows(&pixels, h).expect("push_rows b failed");
    let mut cursor = Cursor::new(Vec::<u8>::new());
    let _stats = enc_b
        .finish_to_seekable(&mut cursor)
        .expect("finish_to_seekable failed");
    assert_eq!(
        cursor.into_inner(),
        baseline,
        "finish_to_seekable output differs from finish()"
    );
}

/// `LosslessEncoder::finish_to_seekable` round-trips bytes identically
/// to `finish()`.
#[test]
fn finish_to_seekable_round_trips_identically_lossless() {
    use std::io::Cursor;

    let w = 64u32;
    let h = 64u32;
    let pixels = gradient_rgb(w, h);

    let mut enc_a = LosslessConfig::new()
        .encoder(w, h, PixelLayout::Rgb8)
        .expect("encoder a failed");
    enc_a.push_rows(&pixels, h).expect("push_rows a failed");
    let baseline = enc_a.finish().expect("finish failed");

    let mut enc_b = LosslessConfig::new()
        .encoder(w, h, PixelLayout::Rgb8)
        .expect("encoder b failed");
    enc_b.push_rows(&pixels, h).expect("push_rows b failed");
    let mut cursor = Cursor::new(Vec::<u8>::new());
    let _stats = enc_b
        .finish_to_seekable(&mut cursor)
        .expect("finish_to_seekable failed");
    assert_eq!(
        cursor.into_inner(),
        baseline,
        "finish_to_seekable output differs from finish()"
    );
}

/// libjxl `6553831` fix: the non-streaming TOC writer MUST emit
/// `permuted_toc=0` explicitly (the bug was that the bit was silently
/// absent because the streaming path always wrote `1`). Our
/// [`write_toc`] helper writes `0` unconditionally via
/// `writer.write(1, 0)`, but we add an invariant test here so that
/// future chunk-7 wireup of the `Buffering::FullStreaming` path doesn't
/// accidentally regress the chunk-6 `Buffering::BufferedOutput` `0`-bit
/// emit.
///
/// We pin this by asserting the encoded bitstreams under
/// `BufferedOutput` and `FullBuffered` are byte-identical on a
/// multi-DC-group image — which means both paths wrote the same TOC
/// permutation bit (in our chunk-6 case, both `0`). When chunk 7 lands
/// the level-3 streaming path, that path WILL write `1` and produce
/// a different byte sequence; the chunk-7 commit will need to update
/// this test to expect divergence for `FullStreaming` only.
#[test]
fn permuted_toc_zero_invariant_for_buffered_output() {
    let w = 2560u32;
    let h = 2560u32;
    let pixels = gradient_rgb(w, h);
    let full_buffered = LossyConfig::new(2.0)
        .with_buffering(Buffering::FullBuffered)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("FullBuffered encode failed");
    let buffered_output = LossyConfig::new(2.0)
        .with_buffering(Buffering::BufferedOutput)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("BufferedOutput encode failed");
    assert_eq!(
        full_buffered, buffered_output,
        "permuted_toc=0 invariant: BufferedOutput must produce identical \
         bytes to FullBuffered (chunk 6 — chunk 7 lifts this when level-3 \
         streaming output lands with permuted_toc=1)"
    );
}
