// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression test for multi-group modular sub-bitstreams whose section has
//! no decodable channels.
//!
//! ## The bug, encoder side
//!
//! Two trigger configurations exist:
//!
//! 1. **Multi-group VarDCT frame with an extra channel (alpha)** whose
//!    dimensions exceed `group_dim` on at least one axis. The alpha plane
//!    is deferred to PassGroups, so the global LfGlobal modular section
//!    carries only the GroupHeader.
//!
//! 2. **Multi-group patches reference frame** whose color channels exceed
//!    `group_dim`. All channels are deferred to PassGroups, so the
//!    LfGlobal modular section carries the global tree + histogram +
//!    GroupHeader but no per-pixel symbols.
//!
//! Pre-fix, `jxl-encoder` ended both sections without the 32-bit ANS
//! initial state. libjxl is bug-compatible by *always* emitting that 32
//! bits via `WriteTokens` (even with zero tokens) — its `Decoder::begin()`
//! reads them unconditionally before checking buffer dims. The fix mirrors
//! libjxl: emit the 32 bits even when no tokens follow in this section.
//!
//! ## The bug, decoder side
//!
//! `jxl-rs` and `djxl` (libjxl) short-circuit before reading the ANS
//! state when no channels are decodable in this section
//! (`is_empty` / `num_chans == 0` early-returns), so they decode our
//! pre-fix bitstreams correctly. **Stock jxl-oxide 0.12.5 does not** —
//! it calls `Decoder::begin()` unconditionally, hits EOF, and rejects
//! the file. `imazen/jxl-oxide@fd4e2c3` adds the matching skip on the
//! decoder side; this encoder-side fix removes the need for that
//! workaround for files we produce.
//!
//! ## What this test asserts
//!
//! Encodes a 512×512 RGBA image at distance 1.0, effort 7 (multi-group:
//! 2×2 groups of 256×256, alpha deferred to PassGroups). Decodes via
//! jxl-rs and via in-process `jxl-oxide` (the workspace pins the imazen
//! fork). Both must succeed and reproduce the alpha plane exactly.
//!
//! Stock jxl-oxide 0.12.5 was verified to decode the output successfully
//! after the fix; it cannot be exercised in this in-process test because
//! the workspace `[patch.crates-io]` redirects the `jxl-oxide` crate to
//! the imazen fork (kept as defense-in-depth for bitstreams from third
//! parties). See the cited commit for the fork's decoder-side fix.

use jxl_encoder::{LossyConfig, PixelLayout};

/// Build an RGBA buffer with a non-trivial alpha pattern, so a regression
/// that fills alpha with a constant or drops it entirely is caught
/// immediately.
fn make_rgba_buffer(width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            // Gradients across the image — RGB channels carry real
            // content so the VarDCT path isn't degenerate.
            out[idx] = ((x * 255) / width.max(1)) as u8;
            out[idx + 1] = ((y * 255) / height.max(1)) as u8;
            out[idx + 2] = (((x + y) * 255) / (width + height).max(1)) as u8;
            // Diamond alpha pattern + per-pixel modulation so no two
            // pixels share an alpha value.
            let cx = width as i32 / 2;
            let cy = height as i32 / 2;
            let dx = (x as i32 - cx).abs();
            let dy = (y as i32 - cy).abs();
            let dist = dx + dy;
            let radius = (width.min(height) / 2) as i32;
            let base = if dist > radius {
                0
            } else {
                ((radius - dist).clamp(0, 255) as u8).saturating_mul(8)
            };
            let modulation = ((x ^ y) & 0x07) as u8;
            out[idx + 3] = base.saturating_add(modulation);
        }
    }
    out
}

/// Decode a JXL bitstream as RGBA8 with jxl-rs. Returns
/// `(width, height, num_extras, rgba_bytes)`.
fn decode_jxl_rs_rgba8(data: &[u8]) -> (u32, u32, usize, Vec<u8>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let num_extras = basic_info.extra_channels.len();

    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![None; num_extras],
    });

    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {e:?}"),
        }
    };

    let channels = 4;
    let mut output_image =
        Image::<u8>::new((width * channels, height)).expect("alloc rgba8 buffer");
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
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => panic!(
                "jxl-rs frame decode error (likely empty-modular-section EOF regression): \
                 {e:?}"
            ),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width as u32, height as u32, num_extras, pixels)
}

/// Decode a JXL bitstream with jxl-oxide and assert it succeeds.
/// The workspace `[patch.crates-io]` redirects `jxl-oxide` to the
/// imazen fork (which carries `fd4e2c3`'s decoder-side workaround for
/// the same bug). Stock 0.12.5 was verified separately by running the
/// installed `jxl-oxide` CLI binary against the same fixture.
fn decode_jxl_oxide(data: &[u8]) -> (u32, u32) {
    use jxl_oxide::JxlImage;

    let image = JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide: header parse must succeed");
    let header = image.image_header();
    let width = header.size.width;
    let height = header.size.height;

    let _frame = image.render_frame(0).unwrap_or_else(|e| {
        panic!(
            "jxl-oxide: render_frame failed — likely the empty-modular-section EOF \
             regression (missing 32-bit ANS state in an LfGlobal modular sub-bitstream \
             whose section has no decodable channels). Error: {e}",
        )
    });

    (width, height)
}

/// Trigger 1: multi-group VarDCT with alpha extra channel deferred to
/// PassGroups. 512×512 RGBA at d=1.0 e=7 produces a 2×2 group layout
/// where the alpha plane exceeds `group_dim` (256), so the global
/// LfGlobal modular section is empty (no decodable channels). The fix
/// emits a 32-bit ANS initial state for that empty section so pre-fix
/// jxl-oxide and any other decoder that doesn't short-circuit on
/// `is_empty` / `num_chans == 0` accepts the file.
#[test]
fn multigroup_vardct_alpha_roundtrips_when_global_section_is_empty() {
    const W: usize = 512;
    const H: usize = 512;

    let rgba = make_rgba_buffer(W, H);

    let bytes = LossyConfig::new(1.0)
        .with_effort(7)
        .encode_request(W as u32, H as u32, PixelLayout::Rgba8)
        .encode(&rgba)
        .expect("encode 512x512 RGBA at d=1.0 e=7 must succeed");

    assert!(
        bytes.len() > 32,
        "encoded bytes too small ({}); something went wrong",
        bytes.len(),
    );
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");

    // jxl-rs roundtrip — must report 1 extra channel and reproduce
    // alpha. jxl-rs handles the empty-section case via its
    // `decode_modular_subbitstream` `is_empty` early-return, so it
    // would tolerate the pre-fix bitstream. Kept as a basic
    // roundtrip-correctness gate.
    let (jxlrs_w, jxlrs_h, jxlrs_extras, rgba_out) = decode_jxl_rs_rgba8(&bytes);
    assert_eq!(jxlrs_w, W as u32);
    assert_eq!(jxlrs_h, H as u32);
    assert_eq!(
        jxlrs_extras, 1,
        "jxl-rs: expected exactly one extra channel (alpha), got {jxlrs_extras}",
    );
    assert_eq!(rgba_out.len(), W * H * 4);

    // Alpha is encoded losslessly as a modular sub-bitstream — must
    // match the input pixel-for-pixel. Detects regressions that drop
    // alpha or scramble it.
    let mut decoded_alpha = vec![0u8; W * H];
    let mut input_alpha = vec![0u8; W * H];
    for i in 0..(W * H) {
        decoded_alpha[i] = rgba_out[i * 4 + 3];
        input_alpha[i] = rgba[i * 4 + 3];
    }
    assert_eq!(
        decoded_alpha, input_alpha,
        "jxl-rs: decoded alpha plane does not match input — modular extras path is broken",
    );

    // jxl-oxide roundtrip — exercises the exact decoder path that
    // pre-fix would EOF on. The in-process build is the imazen fork
    // (which has the workaround), so this won't catch a regression on
    // its own. Stock 0.12.5 was manually verified.
    let (ox_w, ox_h) = decode_jxl_oxide(&bytes);
    assert_eq!(ox_w, W as u32);
    assert_eq!(ox_h, H as u32);
}

/// Trigger 2: multi-group patches reference frame with all channels
/// deferred to PassGroups. A 2940×1912 screenshot at d=1.0 e=7 detects
/// text-like patches, packs them into a >256×256 reference frame, and
/// writes a multi-group reference frame whose LfGlobal modular section
/// has no decodable channels in section 0 (channels deferred to
/// PassGroups). The fix emits the 32-bit ANS initial state for that
/// section so jxl-oxide pre-fix decodes the file successfully.
///
/// `#[ignore]` because the fixture isn't shipped in-tree; the test runs
/// when the codec-corpus path is present. Run via:
/// `cargo test -p jxl-encoder --test empty_modular_section_roundtrip \
///   multigroup_patches_ref_frame -- --ignored --nocapture`.
#[test]
#[ignore = "requires ~/work/codec-corpus/gb82-sc/imac_g3.png fixture"]
fn multigroup_patches_ref_frame_roundtrips_when_global_section_is_empty() {
    use std::path::PathBuf;

    let home = std::env::var("HOME").expect("HOME must be set");
    let path = PathBuf::from(home).join("work/codec-corpus/gb82-sc/imac_g3.png");
    if !path.exists() {
        eprintln!("Skipping: fixture not found at {}", path.display());
        return;
    }

    // Decode PNG to RGBA8 (codec-corpus screenshots are RGB but we
    // round-trip as RGB here — the patches reference frame trigger
    // comes from the screenshot content, not alpha).
    let img = image::open(&path).expect("decode PNG fixture").to_rgb8();
    let (w, h) = (img.width(), img.height());
    let pixels: Vec<u8> = img.into_raw();

    let bytes = LossyConfig::new(1.0)
        .with_effort(7)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&pixels)
        .expect("encode 2940x1912 screenshot at d=1.0 e=7 must succeed");

    assert_eq!(&bytes[..2], &[0xFF, 0x0A]);

    // jxl-rs roundtrip — handles empty sections correctly via
    // `is_empty` short-circuit even pre-fix.
    let (jxlrs_w, jxlrs_h, _extras, _rgba_out) = decode_jxl_rs_rgba8(&bytes);
    assert_eq!(jxlrs_w, w);
    assert_eq!(jxlrs_h, h);

    // jxl-oxide roundtrip — exercises the patches-reference-frame
    // LfGlobal-empty-section path that pre-fix EOFs on.
    let (ox_w, ox_h) = decode_jxl_oxide(&bytes);
    assert_eq!(ox_w, w);
    assert_eq!(ox_h, h);
}
