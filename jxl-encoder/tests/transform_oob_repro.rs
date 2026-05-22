// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression for an OOB panic at `vardct/transform.rs:544` (DCT64x64 DC
//! extraction).
//!
//! The user's repro is the GB82-SC `codec_wiki.png` screenshot resized to
//! 12 MP at distance 0.5 / 1.0, encoded through the
//! `jxl-encoder-gpu::perf_bitstream_u8_vs_f32` example:
//!
//! ```text
//! thread '<unnamed>' panicked at jxl-encoder/src/vardct/transform.rs:544:37:
//! index out of bounds: the len is 1024 but the index is 1048
//! ```
//!
//! ## Root cause
//!
//! The CPU encoder's group transform pipeline assumes every multi-block
//! AC strategy fits inside a single 32×32-block pass-group. Per-tile AC
//! strategy search satisfies that invariant naturally because tile size
//! (8 blocks) divides group size (32 blocks). However, downstream callers
//! of `__pre_quantized::EncoderPrecomputed::from_parts` (notably the GPU
//! encoder's strat-search injector) can supply an `AcStrategyMap` that
//! marks a large transform — DCT64×64, DCT64×32, DCT32×64, DCT32×32 etc.
//! — at a position whose coverage straddles a group boundary.
//!
//! When the per-group transform code later iterates the group's blocks
//! and writes the strategy's DC values, the local index
//! `(by - yoff + iy) * width + (bx - xoff + ix)` falls outside the
//! group's `quant_dc[c]` (length `width * height`).
//!
//! The release build's silent OOB shows up at line 544 (DCT64x64) because
//! that's the largest transform and the easiest to land in a panicking
//! position; the same bug exists for any multi-block strategy crossing a
//! group boundary.
//!
//! ## Fix
//!
//! `AcStrategyMap::set` now silently skips writes whose multi-block
//! coverage would cross a group or image boundary, so the touched
//! blocks keep whatever existing value they had (`DCT8` from
//! `new_dct8`). The in-tree per-tile search already respects the
//! invariant, so this is a no-op for first-party callers; for external
//! producers (jxl-encoder-gpu's pre-quantized injector) it converts a
//! panic into a correct (if mildly less compact) bitstream. A
//! `debug_assert!` is retained so internal callers still surface the
//! invariant violation at the source in debug builds.

#[allow(unused_imports)]
use jxl_encoder::{LossyConfig, PixelLayout};

#[cfg(all(feature = "__pre_quantized", not(debug_assertions)))]
use jxl_encoder::__pre_quantized::AcStrategyMap;

/// Repro using the actual screenshot from the user's report. Ignored by
/// default because the file is corpus-only; run with `--ignored` locally.
#[test]
#[ignore]
fn transform_oob_repro_codec_wiki_12mp_d05_e7() {
    let path = "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png";
    let mut img = image::open(path).expect("open codec_wiki.png").to_rgb8();
    let (sw, sh) = img.dimensions();
    let cur_mp = sw as f32 * sh as f32 / 1e6;
    let scale = (12.0 / cur_mp).sqrt();
    let nw = ((sw as f32 * scale).round() as u32).max(8);
    let nh = ((sh as f32 * scale).round() as u32).max(8);
    let filter = if nw * nh < (sw * sh) {
        image::imageops::FilterType::Lanczos3
    } else {
        image::imageops::FilterType::Triangle
    };
    img = image::imageops::resize(&img, nw, nh, filter);
    let (w, h) = img.dimensions();
    let pixels = img.into_raw();
    let bytes = LossyConfig::new(0.5)
        .with_effort(7)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("encode codec_wiki @ 12MP d=0.5 e=7 must not panic");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
}

/// Direct repro: hand-craft an `AcStrategyMap` with a DCT64×64 placed so
/// it crosses a 32×32-block pass-group boundary, then drive the encoder
/// through `EncoderPrecomputed::from_parts` (the same downstream entry
/// point jxl-encoder-gpu uses). Before the fix this panics at
/// `transform.rs:544` with `index out of bounds: the len is 1024 but the
/// index is 1048`.
///
/// Release-only: in debug builds `AcStrategyMap::set` keeps a
/// `debug_assert!` so first-party callers (`compute_ac_strategy`) catch
/// the invariant violation at the source. The runtime safety net only
/// fires in release builds, and that's exactly what we exercise here.
#[cfg(all(feature = "__pre_quantized", not(debug_assertions)))]
#[test]
fn dct64x64_at_group_boundary_does_not_panic() {
    use jxl_encoder::__pre_quantized::{
        DistanceParams, EncoderPrecomputed, VarDctEncoder, compute_cfl_map,
        compute_quant_field_float_free, quantize_quant_field,
    };

    // 64×64-block grid (= 2×2 pass-groups, each 32×32 blocks). At least
    // 2 groups in each dimension so we can land DCT64x64 across the
    // group boundary at bx=24, by=24 (which spans 24..32 and crosses
    // into the 32..40 block coords if the strategy were honored).
    //
    // Picture pixel size: 64×64 blocks × 8 px/block = 512×512 px.
    let xsize_blocks = 64usize;
    let ysize_blocks = 64usize;
    let width = 512u32;
    let height = 512u32;
    let padded_width = 512usize;
    let padded_height = 512usize;

    // Smooth gradient so any large transform looks attractive.
    let n = padded_width * padded_height;
    let mut linear_rgb = vec![0.0f32; n * 3];
    for y in 0..padded_height {
        for x in 0..padded_width {
            let r = (x as f32 / padded_width as f32) * 0.5 + 0.25;
            let g = (y as f32 / padded_height as f32) * 0.5 + 0.25;
            let b = ((x + y) as f32 / (padded_width + padded_height) as f32) * 0.5 + 0.25;
            linear_rgb[(y * padded_width + x) * 3] = r;
            linear_rgb[(y * padded_width + x) * 3 + 1] = g;
            linear_rgb[(y * padded_width + x) * 3 + 2] = b;
        }
    }
    // Convert to XYB planes (independently per pixel).
    let mut xyb_x = vec![0.0f32; n];
    let mut xyb_y = vec![0.0f32; n];
    let mut xyb_b = vec![0.0f32; n];
    for i in 0..n {
        let r = linear_rgb[i * 3];
        let g = linear_rgb[i * 3 + 1];
        let b = linear_rgb[i * 3 + 2];
        // Cheap stand-in for the true linear-RGB→XYB transform; the
        // encoder will treat these as XYB inputs but precise correctness
        // doesn't matter — we only need the panic path to fire.
        xyb_x[i] = (r - g) * 0.5;
        xyb_y[i] = (r + g) * 0.5;
        xyb_b[i] = b - 0.25 * (r + g);
    }
    // Hand-build an AcStrategyMap whose only non-DCT8 entry is a
    // DCT64×64 that crosses the (32, 32)-aligned group boundary.
    // RAW_STRATEGY_DCT64X64 = 16 (matches the const in
    // vardct/ac_strategy.rs).
    let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
    // bx=24, by=24 → coverage spans 24..32 in BOTH dims (within a single
    // group). To cross a group boundary, place the strategy starting at
    // bx=25, by=25 (spans 25..33 → crosses x=32, y=32). This is the
    // exact pre-condition that triggers the OOB.
    ac_strategy.set(25, 25, 16 /* RAW_STRATEGY_DCT64X64 */);

    // Compute supporting structures (CfL, quant field, masking) the same
    // way the GPU encoder does.
    let cfl_map = compute_cfl_map(
        &xyb_x,
        &xyb_y,
        &xyb_b,
        padded_width,
        padded_height,
        xsize_blocks,
        ysize_blocks,
        true,
        1e-3,
        10,
        false, // W44-184: default-path Newton (libjxl_parity off)
    );
    let distance = 1.0_f32;
    let (quant_field_float, masking) = compute_quant_field_float_free(
        &xyb_x,
        &xyb_y,
        &xyb_b,
        padded_width,
        padded_height,
        xsize_blocks,
        ysize_blocks,
        distance,
        0.765, // K_AC_QUANT
    )
    .expect("quant field");

    let precomputed = EncoderPrecomputed::from_parts(
        width as usize,
        height as usize,
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
        None,
        ac_strategy,
        true,
        distance,
        0,
        0,
    );

    let vardct = VarDctEncoder::new(distance);
    let params = DistanceParams::compute_for_profile(distance, &vardct.profile);
    let quant_field_u8 = quantize_quant_field(&quant_field_float, params.inv_scale);

    let bytes = vardct
        .encode_from_precomputed(&precomputed, &quant_field_u8)
        .expect("encode_from_precomputed must not panic on cross-group strategy");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");
}
