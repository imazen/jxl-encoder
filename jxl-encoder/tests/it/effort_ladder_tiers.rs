// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Behavioural coverage for the post-shift extended effort tiers
//! (issue #45): the e11+ lossless TectonicPlate config trial and the
//! e10 iterative 2× downsampler, beyond the byte pins in
//! `hash_lock_features.rs`. Every decoded stream goes through BOTH
//! jxl-rs (primary) and jxl-oxide (secondary) per the repo decoder
//! policy.

use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};

/// 320×96 crop of the 17-colour blocky pattern (same generator as the
/// hash-lock fixture) — palette-friendly multi-group content where the
/// TectonicPlate schedule finds a materially better config.
fn blocky17_rgb_320x96() -> (Vec<u8>, u32, u32) {
    let (fw, fh) = (512usize, 512usize);
    let mut seed = 7777u64;
    let mut lcg = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 56) as u8
    };
    let palette: Vec<[u8; 3]> = (0..17).map(|_| [lcg(), lcg(), lcg()]).collect();
    let mut full = vec![0u8; fw * fh * 3];
    for by in 0..fh / 8 {
        for bx in 0..fw / 8 {
            let c = palette[(bx * 31 + by * 17 + (bx * by) % 7) % 17];
            for y in (by * 8)..(by * 8 + 8) {
                for x in (bx * 8)..(bx * 8 + 8) {
                    let i = (y * fw + x) * 3;
                    full[i..i + 3].copy_from_slice(&c);
                }
            }
        }
    }
    let (w, h) = (320usize, 96usize);
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        px[y * w * 3..(y + 1) * w * 3].copy_from_slice(&full[y * fw * 3..y * fw * 3 + w * 3]);
    }
    (px, w as u32, h as u32)
}

/// Decode through jxl-rs (primary) and jxl-oxide (secondary); assert
/// both agree on dimensions and render non-degenerate output.
fn decode_both(name: &str, data: &[u8], w: u32, h: u32) {
    // jxl-rs (primary decoder; typed-state streaming API).
    {
        use jxl::api::{
            JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
            JxlPixelFormat, ProcessingResult, states,
        };
        use jxl::image::{Image, Rect};

        let mut input = data;
        let mut decoder_init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder_init.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
                Err(e) => panic!("{name}: jxl-rs header decode error: {e:?}"),
            }
        };
        let basic_info = decoder.basic_info().clone();
        let (dw, dh) = basic_info.size;
        assert_eq!((dw as u32, dh as u32), (w, h), "{name}: jxl-rs dims");
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
                Err(e) => panic!("{name}: jxl-rs frame info error: {e:?}"),
            }
        };
        let channels = 3usize;
        let mut output_image = Image::<u8>::new((dw * channels, dh)).expect("alloc jxl-rs output");
        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            output_image
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (dw * channels, dh),
                })
                .into_raw(),
        )];
        loop {
            match decoder_frame.process(&mut input, &mut buffers) {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_frame = fallback,
                Err(e) => panic!("{name}: jxl-rs frame decode error: {e:?}"),
            }
        }
        let non_zero = (0..dh).any(|y| output_image.row(y).iter().any(|&b| b != 0));
        assert!(non_zero, "{name}: jxl-rs decoded all-zero output");
    }
    // jxl-oxide (secondary).
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .unwrap_or_else(|e| panic!("{name}: jxl-oxide parse failed: {e:?}"));
    let hdr = image.image_header();
    assert_eq!(
        (hdr.size.width, hdr.size.height),
        (w, h),
        "{name}: jxl-oxide dims"
    );
    image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("{name}: jxl-oxide render failed: {e:?}"));
}

/// The e11 TectonicPlate trial finds a materially smaller config than
/// the e10 default on palette-friendly blocky content, and the stream
/// decodes in both reference Rust decoders. (The schedule does NOT
/// guarantee e11 ≤ e10 on arbitrary content — the libjxl trial list
/// omits the untouched default config — but on this procedural fixture
/// the win is structural and pinned by the hash-lock cell too.)
#[test]
fn tectonic_e11_beats_e10_default_on_blocky() {
    let (px, w, h) = blocky17_rgb_320x96();
    let e10 = LosslessConfig::new()
        .with_effort(10)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    let e11 = LosslessConfig::new()
        .with_effort(11)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    decode_both("tectonic_e11_blocky", &e11, w, h);
    assert!(
        e11.len() < e10.len(),
        "e11 trial schedule must beat the e10 default on blocky content: \
         e11 {} vs e10 {} bytes",
        e11.len(),
        e10.len()
    );
}

/// Caller-explicit knobs are pinned across the trial schedule (caller
/// intent wins, unlike libjxl's clobber): pinning the palette OFF at
/// e11 must forgo the palette-driven win the unpinned schedule finds.
#[test]
fn tectonic_e11_pins_explicit_palette_off() {
    let (px, w, h) = blocky17_rgb_320x96();
    let unpinned = LosslessConfig::new()
        .with_effort(11)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    let pinned = LosslessConfig::new()
        .with_effort(11)
        .with_modular_palette_colors(Some(0))
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    decode_both("tectonic_e11_palette_pinned", &pinned, w, h);
    assert!(
        pinned.len() > unpinned.len(),
        "palette pinned OFF must forgo the palette win ({} !> {})",
        pinned.len(),
        unpinned.len()
    );
}

/// Lossless e10 diverges from e9 exactly the way libjxl e10 does: the
/// MA-tree split threshold drops from 89 to 75
/// (`75 + 14 × speed_tier`, kGlacier = 0 — libjxl `enc_modular.cc:536`,
/// mirrored by `EffortProfile::tree_threshold_base`), admitting more
/// splits. The stream must differ from e9 and decode everywhere. e12
/// additionally engages the trial + 16-seed machinery.
#[test]
fn lossless_e10_kglacier_threshold_and_e12_decode() {
    let (px, w, h) = blocky17_rgb_320x96();
    let e9 = LosslessConfig::new()
        .with_effort(9)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    let e10 = LosslessConfig::new()
        .with_effort(10)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    decode_both("lossless_e10_blocky", &e10, w, h);
    assert_ne!(
        e9, e10,
        "lossless e10 must engage kGlacier's lower tree-split threshold (75 vs e9's 89)"
    );
    let e12 = LosslessConfig::new()
        .with_effort(12)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    decode_both("tectonic_e12_blocky", &e12, w, h);
}

/// The e10 iterative downsampler under `with_resampling(2)` produces a
/// different (and decodable) stream from the e9 sharper kernel; both
/// decode to the full pre-downsample dimensions in both decoders.
#[test]
fn resampling2_iterative_vs_sharper_streams() {
    // Structured gradient+edge content (the downsamplers only differ on
    // structure; a flat fixture would tie).
    let (w, h) = (320u32, 128u32);
    let mut px = vec![0u8; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = (y * w as usize + x) * 3;
            let g = ((x * 255) / w as usize) as u8;
            let e = if (x / 24 + y / 16) % 2 == 0 { 200 } else { 30 };
            px[i] = g;
            px[i + 1] = e;
            px[i + 2] = g / 2 + e / 2;
        }
    }
    let e9 = LossyConfig::new(1.0)
        .with_effort(9)
        .with_resampling(2)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    let e10 = LossyConfig::new(1.0)
        .with_effort(10)
        .with_resampling(2)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .unwrap();
    decode_both("r2_sharper_e9", &e9, w, h);
    decode_both("r2_iterative_e10", &e10, w, h);
    assert_ne!(
        e9, e10,
        "e10 must dispatch the iterative downsampler (different stream from e9's sharper)"
    );
}
