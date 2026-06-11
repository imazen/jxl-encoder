// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression gate for the full-image palette transform on the lossless
//! multi-group path (issue #69 item 2).
//!
//! Before the fix, palette detection only existed on the single-group
//! paths: a 2-colour 512×512 image (2×2 groups) encoded with no palette
//! transform at all, and `LosslessConfig::with_modular_palette_colors`
//! was a stored-but-unconsumed knob above one group. The fix runs the
//! same `analyze_palette` detection the single-group tree path uses,
//! BEFORE ChannelCompact/RCT (indices are nominal), keeps the palette
//! meta-channel whole in the global section (the ChannelCompact meta
//! mechanism), and splits the index channel per-group.
//!
//! Content is few-colour blocky tiles (LCG-picked colours): exactly the
//! plot/pixel-art class palettes exist for, with non-degenerate index
//! histograms. Real-content coverage comes from the 43-pick bench set
//! A/B; this in-tree gate covers the bitstream contract with jxl-rs
//! (primary) + jxl-oxide.

use jxl_encoder::{LosslessConfig, PixelLayout};

/// Blocky few-colour content: 8×8 blocks, each one of `n_colors` LCG
/// colours, tiled across the image. Distinct colour count stays well
/// under MAX_PALETTE_COLORS so detection must engage.
fn blocky_palette_rgb(width: usize, height: usize, n_colors: usize, bpp: usize) -> Vec<u8> {
    let mut state: u32 = 7;
    let mut lcg = move || {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        (state >> 16) as u8
    };
    let palette: Vec<[u8; 3]> = (0..n_colors).map(|_| [lcg(), lcg(), lcg()]).collect();
    let mut out = vec![0u8; width * height * bpp];
    for by in 0..height.div_ceil(8) {
        for bx in 0..width.div_ceil(8) {
            // Deterministic but non-periodic colour pick.
            let c = palette[(bx * 31 + by * 17 + (bx * by) % 7) % n_colors];
            for y in (by * 8)..((by * 8) + 8).min(height) {
                for x in (bx * 8)..((bx * 8) + 8).min(width) {
                    let i = (y * width + x) * bpp;
                    out[i..i + 3].copy_from_slice(&c);
                    if bpp == 4 {
                        // Few-valued alpha so the extra channel is live
                        // but doesn't explode the colour-tuple count
                        // (palette covers RGB only; alpha stays extra).
                        out[i + 3] = if (bx + by) % 3 == 0 { 200 } else { 255 };
                    }
                }
            }
        }
    }
    out
}

/// Decode with jxl-rs (the primary roundtrip decoder) as RGB8/RGBA8.
fn decode_jxl_rs(data: &[u8], channels: usize) -> (usize, usize, Vec<u8>) {
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
        color_type: if channels == 4 {
            JxlColorType::Rgba
        } else {
            JxlColorType::Rgb
        },
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

    let mut output_image =
        Image::<u8>::new((width * channels, height)).expect("alloc decode buffer");
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

/// jxl-oxide secondary validation: parses + renders with right dims.
fn decode_jxl_oxide_ok(data: &[u8]) -> (usize, usize) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide rejected the bitstream");
    let header = image.image_header();
    let (w, h) = (header.size.width, header.size.height);
    image.render_frame(0).expect("jxl-oxide render failed");
    (w as usize, h as usize)
}

/// Full-image palette engages on the multi-group path, beats the
/// palette-disabled arm, and both arms roundtrip pixel-exact at the
/// schedule efforts (e7 default + e9 ref-properties config).
#[test]
fn multigroup_palette_fires_and_roundtrips_rgb() {
    let (w, h) = (512usize, 512usize);
    let pixels = blocky_palette_rgb(w, h, 17, 3);

    for effort in [7u8, 9] {
        let with_pal = LosslessConfig::new()
            .with_effort(effort)
            .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("e{effort}: encode failed: {e}"));
        let without = LosslessConfig::new()
            .with_effort(effort)
            .with_modular_palette_colors(Some(0))
            .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("e{effort}: no-palette encode failed: {e}"));

        // The knob axis must be LIVE and the transform must pay on
        // 17-colour content (issue #69: it was silently absent).
        assert!(
            with_pal.len() < without.len(),
            "e{effort}: palette did not engage on 17-colour multi-group content \
             (with: {} B, without: {} B)",
            with_pal.len(),
            without.len(),
        );

        for (label, bytes) in [("palette", &with_pal), ("no-palette", &without)] {
            let (dw, dh, decoded) = decode_jxl_rs(bytes, 3);
            assert_eq!((dw, dh), (w, h), "e{effort} {label}: dims");
            assert_eq!(decoded, pixels, "e{effort} {label}: jxl-rs pixels differ");
            assert_eq!(
                decode_jxl_oxide_ok(bytes),
                (w, h),
                "e{effort} {label}: jxl-oxide"
            );
        }
    }
}

/// RGBA: palette covers the RGB colour channels; alpha stays an extra
/// channel split per-group alongside the index channel.
#[test]
fn multigroup_palette_rgba_roundtrips() {
    let (w, h) = (512usize, 512usize);
    let pixels = blocky_palette_rgb(w, h, 17, 4);

    let with_pal = LosslessConfig::new()
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgba8)
        .expect("rgba encode failed");
    let without = LosslessConfig::new()
        .with_modular_palette_colors(Some(0))
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgba8)
        .expect("rgba no-palette encode failed");
    assert!(
        with_pal.len() < without.len(),
        "palette did not engage on RGBA multi-group content (with: {} B, without: {} B)",
        with_pal.len(),
        without.len(),
    );

    let (dw, dh, decoded) = decode_jxl_rs(&with_pal, 4);
    assert_eq!((dw, dh), (w, h), "rgba dims");
    assert_eq!(decoded, pixels, "rgba jxl-rs pixels differ");
    assert_eq!(decode_jxl_oxide_ok(&with_pal), (w, h), "rgba jxl-oxide");
}

/// Photo-like content (per-pixel noise: ~every tuple distinct) must not
/// engage the palette — bytes identical with the knob on either setting.
#[test]
fn multigroup_palette_declines_on_noise() {
    let (w, h) = (512usize, 512usize);
    let mut state: u32 = 99;
    let mut lcg = move || {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        (state >> 16) as u8
    };
    let pixels: Vec<u8> = (0..w * h * 3).map(|_| lcg()).collect();

    let on = LosslessConfig::new()
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
        .expect("encode failed");
    let off = LosslessConfig::new()
        .with_modular_palette_colors(Some(0))
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
        .expect("no-palette encode failed");
    assert_eq!(
        on, off,
        "noise must not pay any palette overhead (detection declines)"
    );
}

/// Grayscale bilevel (document-scan class): single-channel palette is
/// ChannelCompact's domain on the multi-group path, and for nc=1 the two
/// transforms are bitstream-identical — so the full-palette knob arm and
/// the default arm must produce the SAME bytes (ChannelCompact covers
/// it either way), and they must roundtrip.
#[test]
fn multigroup_palette_gray_bilevel_covered_by_channel_compact() {
    let (w, h) = (512usize, 512usize);
    let mut pixels = vec![255u8; w * h];
    // text-ish strokes: every 7th row carries runs of black
    for y in (0..h).step_by(7) {
        for x in 0..w {
            if (x / 5 + y / 7) % 3 != 0 {
                pixels[y * w + x] = 0;
            }
        }
    }
    let with_pal = LosslessConfig::new()
        .encode(&pixels, w as u32, h as u32, PixelLayout::Gray8)
        .expect("gray encode failed");
    let without = LosslessConfig::new()
        .with_modular_palette_colors(Some(0))
        .encode(&pixels, w as u32, h as u32, PixelLayout::Gray8)
        .expect("gray no-palette encode failed");
    assert_eq!(
        with_pal, without,
        "gray multi-group must be byte-identical with the palette knob on \
         either setting: nc=1 is ChannelCompact's job and the transforms \
         are bitstream-identical"
    );
    let (dw, dh, _decoded) = decode_jxl_rs(&with_pal, 3);
    assert_eq!((dw, dh), (w, h), "gray dims");
    assert_eq!(decode_jxl_oxide_ok(&with_pal), (w, h), "gray jxl-oxide");
}
