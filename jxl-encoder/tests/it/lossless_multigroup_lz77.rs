// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression gate for per-section LZ77 on the lossless multi-group path
//! (issue #69 item 1).
//!
//! Before the fix, the tree-learned multi-group path deliberately dropped
//! `use_lz77`/`lz77_method`: the global ANS code would have advertised
//! LZ77 symbols the per-group sections never emitted (histogram
//! mismatch), so every lossless image over one group (256×256) encoded
//! with no LZ77 at all — the documented e7-RLE / e8-Greedy / e9-Optimal
//! schedule was aspirational. The fix mirrors the squeeze multi-group
//! path: each section's token stream is transformed independently with
//! `dist_multiplier = max(section channel widths)` (the decoder creates a
//! fresh LZ77 state per section), the global histogram is built over the
//! transformed streams, and the per-group writers re-apply the identical
//! deterministic transform at write time.
//!
//! Content: a 512×512 RGB image (2×2 groups) tiled from a 64-px-wide
//! per-pixel LCG noise pattern: tokens the tree cannot predict away
//! (expensive symbols, non-degenerate histograms — the ANS-test
//! discipline) recurring at tile distance, which e8 greedy / e9 optimal
//! backrefs monetize. e7 RLE correctly declines under the libjxl 20 %
//! cost gate (the learned tree turns value-runs into near-free zero
//! residuals, so distance-1 runs of expensive tokens don't exist) — the
//! e7 leg asserts no-regression + roundtrip instead. Real-content
//! coverage comes from the 43-pick k-means bench set run with
//! `scripts/bench_lossless_ab.py --decode-verify` (djxl) — this in-tree
//! gate covers the bitstream contract with jxl-rs (primary) + jxl-oxide.

use jxl_encoder::{LosslessConfig, PixelLayout};

/// A `tile`×`tile` pattern of 8×8 constant-colour blocks (LCG colours),
/// repeated across the image — pixel-art-like content. In-row runs of 8
/// give the e7 RLE method period-1 token runs; the tile repetition gives
/// e8 greedy / e9 optimal long backward matches; 64 distinct LCG colours
/// per tile keep the token histograms non-degenerate (the ANS-test
/// discipline: no gradients). `block == 0` selects per-pixel LCG noise
/// (no runs, no repeats when `tile == width`) for the identity test.
fn blocky_tiled_rgb8(width: usize, height: usize, tile: usize, block: usize) -> Vec<u8> {
    let mut state: u32 = 42;
    let mut lcg = move || {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        ((state >> 16) & 0xff) as u8
    };
    let mut tile_px = vec![0u8; tile * tile * 3];
    if block == 0 {
        for b in tile_px.iter_mut() {
            *b = lcg();
        }
    } else {
        for by in (0..tile).step_by(block) {
            for bx in (0..tile).step_by(block) {
                let (r, g, b) = (lcg(), lcg(), lcg());
                for y in by..(by + block).min(tile) {
                    for x in bx..(bx + block).min(tile) {
                        let i = (y * tile + x) * 3;
                        tile_px[i] = r;
                        tile_px[i + 1] = g;
                        tile_px[i + 2] = b;
                    }
                }
            }
        }
    }
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let src = ((y % tile) * tile + (x % tile)) * 3;
            let dst = (y * width + x) * 3;
            out[dst..dst + 3].copy_from_slice(&tile_px[src..src + 3]);
        }
    }
    out
}

/// Decode with jxl-rs (the primary roundtrip decoder) as RGB8.
fn decode_jxl_rs_rgb8(data: &[u8]) -> (usize, usize, Vec<u8>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder_init = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let num_extras = basic_info.extra_channels.len();

    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![None; num_extras],
    });

    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info error: {e:?}"),
        }
    };

    let channels = 3;
    let mut output_image = Image::<u8>::new((width * channels, height)).expect("alloc rgb8 buffer");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_frame = fallback,
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width, height, pixels)
}

/// Decode with jxl-oxide (secondary validation): the bitstream must parse
/// and render with the right dimensions. Exact-pixel duty is jxl-rs's
/// (above) in-process and djxl's via the bench-set CLI verification.
fn decode_jxl_oxide_ok(data: &[u8]) -> (usize, usize) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide rejected the bitstream");
    let header = image.image_header();
    let (w, h) = (header.size.width, header.size.height);
    image.render_frame(0).expect("jxl-oxide render failed");
    (w as usize, h as usize)
}

fn roundtrip_at_effort(pixels: &[u8], w: usize, h: usize, effort: u8) -> Vec<u8> {
    let encoded = LosslessConfig::new()
        .with_effort(effort)
        .encode(pixels, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("e{effort}: encode failed: {e}"));

    let (dw, dh, decoded) = decode_jxl_rs_rgb8(&encoded);
    assert_eq!((dw, dh), (w, h), "e{effort}: dims");
    assert_eq!(decoded, pixels, "e{effort}: jxl-rs pixels differ");

    assert_eq!(
        decode_jxl_oxide_ok(&encoded),
        (w, h),
        "e{effort}: jxl-oxide"
    );

    encoded
}

/// Per-section LZ77 fires on the multi-group tree-learned path and every
/// decoder reproduces the pixels exactly at each schedule effort.
#[test]
fn multigroup_lz77_fires_and_roundtrips_e7_e8_e9() {
    let (w, h) = (512usize, 512usize);
    // Per-pixel LCG noise in a 64-wide tile, repeated across the image:
    // expensive tokens (the tree cannot predict noise) recurring at tile
    // distance — exactly what greedy (e8) / optimal (e9) backrefs feed on.
    let pixels = blocky_tiled_rgb8(w, h, 64, 0);

    for effort in [7u8, 8, 9] {
        let with_lz77 = roundtrip_at_effort(&pixels, w, h, effort);

        let without = LosslessConfig::new()
            .with_effort(effort)
            .with_lz77(false)
            .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("e{effort}: no-lz77 encode failed: {e}"));

        if effort >= 8 {
            // The lz77 axis must be LIVE (issue #69: it was silently
            // dropped). Greedy/optimal find the tile-distance matches and
            // clear the libjxl 20 % benefit floor on this content.
            assert!(
                with_lz77.len() < without.len(),
                "e{effort}: LZ77 did not fire on tile-repeated content \
                 (with: {} B, without: {} B)",
                with_lz77.len(),
                without.len(),
            );
        } else {
            // e7 = RLE: needs runs of identical *expensive* tokens at
            // distance 1. The learned tree predicts value-runs to ~zero
            // residuals (cheap symbols), so RLE correctly declines under
            // the same cost gate libjxl uses — never a regression.
            assert!(
                with_lz77.len() <= without.len(),
                "e7: LZ77-RLE regressed bytes (with: {} B, without: {} B)",
                with_lz77.len(),
                without.len(),
            );
        }

        // The no-LZ77 arm must roundtrip too (it is the pre-#69 layout).
        let (_, _, decoded) = decode_jxl_rs_rgb8(&without);
        assert_eq!(decoded, pixels, "e{effort}: no-lz77 jxl-rs pixels differ");
    }
}

/// Photo-like content (pure LCG noise, no repetition) must stay
/// byte-identical with LZ77 on vs off: no section clears the 20 % benefit
/// floor, so the transform is the identity and the header stays
/// lz77.enabled=0 — the pre-#69 bitstream.
#[test]
fn multigroup_lz77_is_identity_on_noise() {
    let (w, h) = (512usize, 512usize);
    // tile == width, block == 0 → per-pixel noise, no repetition anywhere.
    let pixels = blocky_tiled_rgb8(w, h, 512, 0);

    let on = LosslessConfig::new()
        .with_effort(7)
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
        .expect("encode (lz77 default-on) failed");
    let off = LosslessConfig::new()
        .with_effort(7)
        .with_lz77(false)
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
        .expect("encode (lz77 off) failed");
    assert_eq!(
        on, off,
        "noise content must not pay any LZ77 overhead (identity transform)"
    );
}
