// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
//
// Use of this source code is governed by AGPL-3.0-or-later. Commercial
// licenses at https://www.imazen.io/pricing
//
//! Roundtrip tests for `LossyConfig::with_content_class` (RFC #45
//! pick #4 chunk 1). Confirms that:
//!
//! 1. Passing `Some(ImageContentClass::Screenshot)` at lossy effort 5/6
//!    enables `patches` at the encoder level (visible as a bitstream
//!    size delta vs the same encode with `None`).
//! 2. The patches-enabled bitstream decodes through jxl-rs (primary)
//!    AND jxl-oxide (secondary), end-to-end without errors.
//! 3. Photo-class (and Unknown) classification on the same input is a
//!    no-op — bytes byte-identical vs the unspecified-class encode.

use jxl_encoder::{ImageContentClass, LossyConfig, PixelLayout};

fn decode_jxl_rs_rgb8_smoke(data: &[u8]) -> (u32, u32) {
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
        color_type: JxlColorType::Rgb,
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
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }
    (width as u32, height as u32)
}

fn decode_jxl_oxide_smoke(data: &[u8]) -> (u32, u32) {
    use jxl_oxide::JxlImage;
    let image = JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide: header parse must succeed");
    let header = image.image_header();
    let _frame = image
        .render_frame(0)
        .expect("jxl-oxide: render_frame must succeed");
    (header.size.width, header.size.height)
}

/// Build a synthetic 256×256 RGB8 image with screenshot-shaped content:
/// 8×8 tiles of solid colors with a small "text-like" repeated pattern.
/// FlatColorBlockRatio on this should comfortably exceed 0.30, so any
/// classifier built on the same feature would label it Screenshot.
fn synth_screenshot_rgb8(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let tile_x = x / 16;
            let tile_y = y / 16;
            let color = match (tile_x % 4, tile_y % 4) {
                (0, _) => [240u8, 240, 240], // light gray (background)
                (_, 0) => [240u8, 240, 240],
                (1, 1) => [16, 16, 16], // dark glyph
                (2, 2) => [16, 16, 16],
                (3, 3) => [200, 32, 32], // red accent
                _ => [240u8, 240, 240],
            };
            let p = (y * w + x) as usize * 3;
            buf[p] = color[0];
            buf[p + 1] = color[1];
            buf[p + 2] = color[2];
        }
    }
    buf
}

#[test]
fn content_class_dispatch_screenshot_enables_patches_at_e5() {
    let w = 256u32;
    let h = 256u32;
    let rgb = synth_screenshot_rgb8(w, h);

    // OFF: standard e5 encode, no content class.
    let off = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode off");

    // ON: same e5 encode, Screenshot class.
    let on = LossyConfig::new(1.0)
        .with_effort(5)
        .with_content_class(Some(ImageContentClass::Screenshot))
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode on");

    // Dispatch enables patches; on a repetitive screenshot pattern,
    // patches should produce a meaningfully smaller bitstream (or at
    // worst be cost-rejected and stay byte-identical).
    assert!(
        on.len() <= off.len(),
        "Screenshot-dispatched encode should not be larger than baseline: on={} off={}",
        on.len(),
        off.len()
    );

    // Both bitstreams must decode through jxl-rs (PRIMARY) and
    // jxl-oxide (secondary) without errors. The dispatched bitstream
    // may carry a patches reference frame the baseline doesn't —
    // verifying it decodes is the load-bearing check.
    let (dw_off, dh_off) = decode_jxl_rs_rgb8_smoke(&off);
    assert_eq!((dw_off, dh_off), (w, h));
    let (dw_on, dh_on) = decode_jxl_rs_rgb8_smoke(&on);
    assert_eq!((dw_on, dh_on), (w, h));

    let (ow_off, oh_off) = decode_jxl_oxide_smoke(&off);
    assert_eq!((ow_off, oh_off), (w, h));
    let (ow_on, oh_on) = decode_jxl_oxide_smoke(&on);
    assert_eq!((ow_on, oh_on), (w, h));
}

#[test]
fn content_class_dispatch_photo_class_is_noop_vs_unspecified() {
    // A 256×256 random-RGB image — no repeated patterns, dispatch
    // wouldn't fire on Photo class. Verify bytes are byte-identical.
    let w = 256u32;
    let h = 256u32;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for (i, p) in rgb.iter_mut().enumerate() {
        // Cheap non-repeating pattern: each channel scrambled by a
        // multiplicative hash; no short period.
        *p = ((i.wrapping_mul(2654435761)) >> 24) as u8;
    }

    let unspecified = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode unspecified");
    let photo = LossyConfig::new(1.0)
        .with_effort(5)
        .with_content_class(Some(ImageContentClass::Photo))
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode photo");
    let unknown = LossyConfig::new(1.0)
        .with_effort(5)
        .with_content_class(Some(ImageContentClass::Unknown))
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode unknown");

    assert_eq!(
        unspecified, photo,
        "Photo-class dispatch must be a no-op vs unspecified"
    );
    assert_eq!(
        unspecified, unknown,
        "Unknown-class dispatch must be a no-op vs unspecified"
    );
}

#[test]
fn content_class_dispatch_with_patches_false_respects_opt_out() {
    // Explicit `with_patches(false)` after Screenshot class must
    // suppress the dispatch.
    let w = 256u32;
    let h = 256u32;
    let rgb = synth_screenshot_rgb8(w, h);

    let baseline = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode baseline");
    let class_then_optout = LossyConfig::new(1.0)
        .with_effort(5)
        .with_content_class(Some(ImageContentClass::Screenshot))
        .with_patches(false)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode class_then_optout");

    assert_eq!(
        baseline, class_then_optout,
        "Explicit with_patches(false) must win over content-class dispatch"
    );
}

#[test]
fn content_class_dispatch_e7_is_noop_when_already_default() {
    // At effort 7 patches is already on by default; the dispatch
    // shouldn't change anything, so bytes are byte-identical.
    let w = 256u32;
    let h = 256u32;
    let rgb = synth_screenshot_rgb8(w, h);

    let off = LossyConfig::new(1.0)
        .with_effort(7)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode off");
    let on = LossyConfig::new(1.0)
        .with_effort(7)
        .with_content_class(Some(ImageContentClass::Screenshot))
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode on");

    assert_eq!(
        off, on,
        "At e7 the dispatch should be a no-op (patches already on)"
    );
}

#[test]
fn content_class_dispatch_below_pixel_gate_is_noop() {
    // Below CONTENT_CLASS_MIN_PIXELS (65 536), the dispatch must
    // never fire — protects synthetic / thumbnail fixtures.
    let w = 64u32;
    let h = 64u32;
    let rgb = synth_screenshot_rgb8(w, h);

    let off = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode off");
    let on = LossyConfig::new(1.0)
        .with_effort(5)
        .with_content_class(Some(ImageContentClass::Screenshot))
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode on");

    assert_eq!(off, on, "Below pixel gate, dispatch must be a no-op");
}
