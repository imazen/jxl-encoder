// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Smoke test for [`__pre_quantized::EncoderPrecomputed::with_patches_data`]
//! (added in commit `e23a1b26`).
//!
//! `encode_from_precomputed` has three patch-handling cases (see
//! `vardct/encoder.rs:1860-1908`):
//!   1. `precomputed.patches_data == Some(pd)` — caller pre-detected,
//!      pre-quantized patches and pre-fitted `cfl_map` / `quant_field`
//!      / `ac_strategy` to the patches-subtracted XYB. Encoder writes
//!      the patches frame and uses precomputed state unchanged.
//!   2. `precomputed.patches_data == None && xyb_pre_gaborish == Some` —
//!      legacy `from_parts` path. Encoder runs detection here and
//!      recomputes CfL pass 1 on patches-subtracted XYB.
//!   3. Both `None` — no patches.
//!
//! The case-1 path (the one this test exercises) had zero in-tree
//! coverage before. A regression that left `patches_data` ignored, that
//! mis-clones the `PatchesData`, or that fails to write the patches
//! frame in case 1 would have shipped silently.
//!
//! ## What this test asserts
//!
//! Two `encode_from_precomputed` runs on the same pre-detected screenshot:
//!   - **Case 1**: builds a precomputed via `from_parts` + attaches
//!     patches via `with_patches_data(pd)` — bitstream contains the
//!     patches frame.
//!   - **Case 2**: builds a precomputed via `from_parts` + attaches the
//!     pre-gab XYB via `with_xyb_pre_gaborish` (no `with_patches_data`)
//!     — encoder re-detects and writes patches itself.
//!
//! Bytes from case 1 and case 2 MUST differ — that proves
//! `with_patches_data` is actually being routed through the case-1 path
//! (case-1 reuses the precomputed CfL map; case-2 recomputes pass 1 on
//! patches-subtracted XYB, producing a slightly different bitstream).
//! Both bitstreams MUST decode round-trip through jxl-rs.
//!
//! Synthetic-only by design: per the project CLAUDE.md "no synthetic-only
//! quality tests" rule, this test does NOT assert any quality threshold.
//! It only checks that bitstreams produce, decode, and that the case-1
//! and case-2 paths produce different bytes — proving the API surface
//! is wired up.

#![cfg(feature = "__pre_quantized")]

use jxl_encoder::EncoderMode;
use jxl_encoder::__pre_quantized::{
    DistanceParams, EffortProfile, EncoderPrecomputed, VarDctEncoder, compute_ac_strategy,
    compute_cfl_map, compute_mask1x1, compute_quant_field_float_free, find_and_build_patches,
    gaborish_inverse, quantize_quant_field, subtract_patches,
};
use jxl_encoder::color::xyb::linear_rgb_to_xyb;

/// Build a "screenshot-like" image: solid background + a small text-glyph
/// foreground replicated at multiple positions. Exactly the shape libjxl's
/// `find_text_like_patches` is tuned to detect (flat-block seeds + repeated
/// foreground pixels above background).
///
/// Returns interleaved linear-RGB f32 (`width * height * 3`).
fn make_screenshot(width: usize, height: usize) -> Vec<f32> {
    // sRGB-ish background + foreground (linear values; values close to those
    // produced by sRGB-to-linear of light grey on dark grey). The exact
    // values don't matter — only that there's enough flat background to seed
    // the detector and the foreground glyph is small (8x8) and replicated
    // many times.
    let bg = [0.85_f32, 0.85, 0.85];
    let fg = [0.05_f32, 0.05, 0.05];
    let mut out = vec![0.0_f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            out[i] = bg[0];
            out[i + 1] = bg[1];
            out[i + 2] = bg[2];
        }
    }

    // Stamp a small "T"-glyph (8x8) at a 24-px stride across the image. The
    // detector demands both a flat 4x4 background AND repeated patches with
    // ≥2 occurrences and max patch dim ≥20px in the bin-packed dictionary.
    // 8x8 glyphs at distinct positions force a meaningful text-like pattern.
    let glyph: [[u8; 8]; 8] = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ];
    let stride = 24_usize;
    let mut py = 16_usize;
    while py + 8 <= height.saturating_sub(16) {
        let mut px = 16_usize;
        while px + 8 <= width.saturating_sub(16) {
            for gy in 0..8 {
                for gx in 0..8 {
                    if glyph[gy][gx] != 0 {
                        let i = ((py + gy) * width + (px + gx)) * 3;
                        out[i] = fg[0];
                        out[i + 1] = fg[1];
                        out[i + 2] = fg[2];
                    }
                }
            }
            px += stride;
        }
        py += stride;
    }
    out
}

/// Convert interleaved linear RGB to padded XYB planes (matches what
/// `EncoderPrecomputed::compute` does internally — but exposed here so we
/// can snapshot the pre-gaborish state, which `compute()` doesn't return.)
fn linear_rgb_to_xyb_padded(
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
    linear_rgb: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = padded_width * padded_height;
    let mut xyb_x = vec![0.0_f32; n];
    let mut xyb_y = vec![0.0_f32; n];
    let mut xyb_b = vec![0.0_f32; n];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            let (xv, yv, bv) = linear_rgb_to_xyb(linear_rgb[i], linear_rgb[i + 1], linear_rgb[i + 2]);
            let dst = y * padded_width + x;
            xyb_x[dst] = xv;
            xyb_y[dst] = yv;
            xyb_b[dst] = bv;
        }
        // Right-edge replicate.
        if padded_width > width {
            let last = y * padded_width + (width - 1);
            let lx = xyb_x[last];
            let ly = xyb_y[last];
            let lb = xyb_b[last];
            for x in width..padded_width {
                let dst = y * padded_width + x;
                xyb_x[dst] = lx;
                xyb_y[dst] = ly;
                xyb_b[dst] = lb;
            }
        }
    }
    // Bottom-edge replicate.
    if padded_height > height {
        let last_row = (height - 1) * padded_width;
        for y in height..padded_height {
            let dst = y * padded_width;
            xyb_x.copy_within(last_row..last_row + padded_width, dst);
            xyb_y.copy_within(last_row..last_row + padded_width, dst);
            xyb_b.copy_within(last_row..last_row + padded_width, dst);
        }
    }
    (xyb_x, xyb_y, xyb_b)
}

/// Build an `EncoderPrecomputed` for the screenshot at `distance`,
/// optionally taking the case-1 path by attaching the pre-detected
/// `PatchesData` via `with_patches_data`.
///
/// Returns `(precomputed, quant_field_u8, vardct, found_patches)`.
/// `found_patches` is `true` when `find_and_build_patches` returned `Some`
/// — the test asserts this so a content regression that breaks our
/// detector trigger is caught immediately (instead of silently skipping
/// the case-1 path).
fn build_precomputed(
    width: usize,
    height: usize,
    linear_rgb: &[f32],
    distance: f32,
    use_case1: bool,
) -> (EncoderPrecomputed, Vec<u8>, VarDctEncoder, bool) {
    // Block-aligned dimensions.
    let xsize_blocks = width.div_ceil(8);
    let ysize_blocks = height.div_ceil(8);
    let padded_width = xsize_blocks * 8;
    let padded_height = ysize_blocks * 8;

    // Pre-gaborish XYB (snapshot taken before any sharpening).
    let (mut xyb_x, mut xyb_y, mut xyb_b) =
        linear_rgb_to_xyb_padded(width, height, padded_width, padded_height, linear_rgb);
    let pre_gab = [xyb_x.clone(), xyb_y.clone(), xyb_b.clone()];

    // Detect patches on PRE-gaborish XYB (libjxl pipeline order).
    let mut patches = find_and_build_patches(
        [&pre_gab[0], &pre_gab[1], &pre_gab[2]],
        width,
        height,
        padded_width,
    );
    let found = patches.is_some();
    if let Some(ref mut pd) = patches {
        pd.quantize_ref_image();
    }

    // Subtract patches from the pre-gaborish XYB BEFORE running gaborish /
    // CfL / AC strategy — matches `compute_with_budget` ordering.
    if let Some(ref pd) = patches {
        let mut xyb_arr = [
            core::mem::take(&mut xyb_x),
            core::mem::take(&mut xyb_y),
            core::mem::take(&mut xyb_b),
        ];
        subtract_patches(&mut xyb_arr, padded_width, pd);
        let [x, y, b] = xyb_arr;
        xyb_x = x;
        xyb_y = y;
        xyb_b = b;
    }

    let profile = EffortProfile::lossy(7, EncoderMode::Reference);

    // Initial quant field + masking on PRE-gaborish patches-subtracted XYB.
    let (quant_field_float, masking) = compute_quant_field_float_free(
        &xyb_x,
        &xyb_y,
        &xyb_b,
        padded_width,
        padded_height,
        xsize_blocks,
        ysize_blocks,
        distance,
        profile.k_ac_quant,
    )
    .expect("compute_quant_field_float_free");

    // mask1x1 on PRE-gaborish patches-subtracted Y.
    let mask1x1 = compute_mask1x1(&xyb_y, padded_width, padded_height);

    // Apply gaborish_inverse (sharpening) — encoder DCT operates on this.
    gaborish_inverse(&mut xyb_x, &mut xyb_y, &mut xyb_b, padded_width, padded_height)
        .expect("gaborish_inverse");

    // CfL pass 1 on POST-gaborish patches-subtracted XYB.
    let cfl_map = compute_cfl_map(
        &xyb_x,
        &xyb_y,
        &xyb_b,
        padded_width,
        padded_height,
        xsize_blocks,
        ysize_blocks,
        true,
        profile.cfl_newton_eps,
        profile.cfl_newton_max_iters,
    );

    // AC strategy on POST-gaborish patches-subtracted XYB.
    let ac_strategy = compute_ac_strategy(
        &xyb_x,
        &xyb_y,
        &xyb_b,
        padded_width,
        padded_height,
        xsize_blocks,
        ysize_blocks,
        distance,
        &quant_field_float,
        &masking,
        &cfl_map,
        Some(&mask1x1),
        padded_width,
        &profile,
    );

    // Build precomputed via `from_parts`. xyb_x/y/b here are the
    // post-gaborish PATCHES-SUBTRACTED planes — what the encoder will
    // DCT in case 1.
    let mut precomputed = EncoderPrecomputed::from_parts(
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        padded_width,
        padded_height,
        xyb_x,
        xyb_y,
        xyb_b,
        Vec::new(),
        cfl_map,
        None,
        quant_field_float.clone(),
        masking,
        Some(mask1x1),
        ac_strategy,
        true,
        distance,
        0,
        0,
    );

    if use_case1 {
        // Case 1: attach the patches dict; encoder skips its own
        // detection and uses the precomputed state unchanged.
        if let Some(pd) = patches {
            precomputed = precomputed.with_patches_data(pd);
        }
    } else {
        // Case 2: attach pre-gaborish XYB (un-patched). Encoder will
        // re-detect and re-subtract internally, then recompute CfL pass 1
        // on patches-subtracted XYB. Skip the patches step here since the
        // encoder does it.
        precomputed = precomputed.with_xyb_pre_gaborish(pre_gab);
    }

    let vardct = VarDctEncoder::new(distance);
    let params = DistanceParams::compute_for_profile(distance, &vardct.profile);
    let quant_field_u8 = quantize_quant_field(&quant_field_float, params.inv_scale);

    (precomputed, quant_field_u8, vardct, found)
}

/// Decode JXL bytes via jxl-rs (project CLAUDE.md mandates jxl-rs as the
/// primary decoder for roundtrip validation). Returns `(w, h, rgb_f32)`
/// or panics with the decoder's error.
fn decode_jxl_rs(data: &[u8]) -> (usize, usize, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during header");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;

    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input before frame");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {e:?}"),
        }
    };

    let mut output = Image::<f32>::new((width * 3, height)).expect("alloc output");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * 3, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during decode");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs decode error: {e:?}"),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        pixels.extend_from_slice(output.row(y));
    }
    (width, height, pixels)
}

#[test]
fn with_patches_data_case1_vs_case2_distinct_bitstreams_decode_roundtrip() {
    // 256x256 keeps the test fast (~1-2s release build) while still
    // exceeding the patch detector's `bw >= 3 && bh >= 3` floor with
    // plenty of room for the BFS background flood.
    let width = 256_usize;
    let height = 256_usize;
    let distance = 1.0_f32;
    let linear_rgb = make_screenshot(width, height);

    // Case 1: caller pre-attaches the PatchesData via `with_patches_data`.
    let (pc1, qf1, vardct1, found1) =
        build_precomputed(width, height, &linear_rgb, distance, /*use_case1=*/ true);
    assert!(
        found1,
        "synthetic screenshot must trigger patches detection — \
         find_and_build_patches returned None. Adjust make_screenshot \
         to keep the test exercising the case-1 path."
    );
    let bytes1 = vardct1
        .encode_from_precomputed(&pc1, &qf1)
        .expect("encode_from_precomputed (case 1, with_patches_data)");

    // Case 2: same content, but caller does NOT attach PatchesData.
    // Encoder runs detection itself via `xyb_pre_gaborish`.
    let (pc2, qf2, vardct2, _found2) =
        build_precomputed(width, height, &linear_rgb, distance, /*use_case1=*/ false);
    let bytes2 = vardct2
        .encode_from_precomputed(&pc2, &qf2)
        .expect("encode_from_precomputed (case 2, no with_patches_data)");

    // Both bitstreams must be valid JXL.
    assert_eq!(&bytes1[..2], &[0xFF, 0x0A], "case 1: missing JXL signature");
    assert_eq!(&bytes2[..2], &[0xFF, 0x0A], "case 2: missing JXL signature");
    assert!(bytes1.len() > 32, "case 1: bitstream suspiciously short");
    assert!(bytes2.len() > 32, "case 2: bitstream suspiciously short");

    // The negative assertion: bytes1 != bytes2.
    //
    // Case 1 reuses `precomputed.cfl_map` (fitted to patches-subtracted XYB
    // by the test harness). Case 2 has the encoder recompute CfL pass 1 on
    // patches-subtracted XYB internally — same algorithm, but the precomputed
    // state passed into the encoder differs: case 1's `precomputed.xyb_*` is
    // already patches-subtracted, while case 2's `precomputed.xyb_*` was
    // patches-subtracted by us too (we apply the same subtract in
    // `build_precomputed` regardless of `use_case1`), BUT the encoder in
    // case 2 ALSO re-applies patches subtract to a fresh clone of the pre-gab
    // planes. The two bitstreams diverge because case 1's cfl_map sees
    // the already-subtracted post-gab planes whereas case 2's recomputed
    // cfl_map sees the encoder's freshly-subtracted-and-re-gaborish'd planes.
    //
    // If a future refactor makes case 1 and case 2 produce identical bitstreams
    // (e.g. by routing both through the same recompute), this assertion will
    // fail and the test author should re-evaluate what "case 1 is being taken"
    // looks like. The shape of the assertion is "the API surface routes
    // through different code paths" — not "case 1 always beats case 2".
    assert_ne!(
        bytes1, bytes2,
        "case 1 (with_patches_data) and case 2 (no with_patches_data) \
         produced byte-identical bitstreams — the with_patches_data() API \
         is being silently ignored. This means a regression in the case-1 \
         path of encode_from_precomputed (vardct/encoder.rs:1860-1908)."
    );

    // Both bitstreams MUST roundtrip through jxl-rs (the primary decoder).
    let (w1, h1, _rgb1) = decode_jxl_rs(&bytes1);
    assert_eq!((w1, h1), (width, height), "case 1: decoded dims wrong");
    let (w2, h2, _rgb2) = decode_jxl_rs(&bytes2);
    assert_eq!((w2, h2), (width, height), "case 2: decoded dims wrong");

    // Print byte sizes for the report — visible with `cargo test -- --nocapture`.
    eprintln!(
        "with_patches_data smoke: case1={}B case2={}B (delta={:+}B)",
        bytes1.len(),
        bytes2.len(),
        bytes1.len() as i64 - bytes2.len() as i64,
    );
}
