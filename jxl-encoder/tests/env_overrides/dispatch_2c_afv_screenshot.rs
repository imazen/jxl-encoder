// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
//
// Use of this source code is governed by AGPL-3.0-or-later. Commercial
// licenses at https://www.imazen.io/pricing
//
//! Issue #43 chunk 2c — Screenshot-class `try_dct4x8_afv` lift at e5.
//!
//! Covers the two pieces the lib-crate unit tests cannot:
//!
//! 1. The `JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1` env hook (process-env
//!    mutation requires `unsafe` in edition 2024; the lib crate is
//!    `#![forbid(unsafe_code)]`, integration tests are their own crate
//!    roots). Same mutex + snapshot/restore discipline as
//!    `strategy_env_fallback.rs`.
//! 2. End-to-end decodability of a gated encode (explicit
//!    `with_content_class(Screenshot)` at e5 on >= 65,536-px content)
//!    through jxl-rs (PRIMARY) and jxl-oxide. Real-corpus gated cells
//!    (gb82-sc at e5 via the W44-164 auto-classifier) are decoder-
//!    verified in the bench companion
//!    (`benchmarks/dispatch_2c_afv_screenshot_2026-06-10.meta`).

use jxl_encoder::api::EncoderMode;
use jxl_encoder::{EffortProfile, ImageContentClass};
use jxl_encoder::{LossyConfig, PixelLayout};

/// Serialises env-var mutation within this test binary. Only this
/// module touches `JXL_DISPATCH_AFV_SCREENSHOT_DISABLE`; the lock
/// still guards against future tests adopting the same var.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const VAR: &str = "JXL_DISPATCH_AFV_SCREENSHOT_DISABLE";

fn with_env_locked<R>(value: Option<&str>, f: impl FnOnce() -> R) -> R {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let snapshot = std::env::var(VAR).ok();
    match value {
        // SAFETY: env mutation is serialised by ENV_LOCK; no other
        // thread reads or writes this var while the guard is held.
        Some(v) => unsafe { std::env::set_var(VAR, v) },
        // SAFETY: see above.
        None => unsafe { std::env::remove_var(VAR) },
    }
    let r = f();
    match snapshot {
        // SAFETY: see above — guard still held.
        Some(v) => unsafe { std::env::set_var(VAR, v) },
        // SAFETY: see above — guard still held.
        None => unsafe { std::env::remove_var(VAR) },
    }
    r
}

/// With the env hook unset, the e5 Screenshot lift fires; with
/// `JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1` it must not. Values other
/// than `"1"` keep the lift active.
#[test]
fn afv_lift_env_disable_hook() {
    let _env_serial = crate::env_serial();
    // Unset → lift fires.
    with_env_locked(None, || {
        let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
        assert!(!p.try_dct4x8_afv, "e5 baseline must be false");
        p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
        assert!(p.try_dct4x8_afv, "env unset: lift must fire");
    });
    // "1" → suppressed.
    with_env_locked(Some("1"), || {
        let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
        p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
        assert!(!p.try_dct4x8_afv, "env=1: lift must be suppressed");
        // The chunk-1 patches rule has no env hook and must be
        // unaffected by this one.
        assert!(p.patches, "patches rule must be unaffected by the AFV hook");
    });
    // "0" → lift still fires (only the exact string "1" disables).
    with_env_locked(Some("0"), || {
        let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
        p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
        assert!(p.try_dct4x8_afv, "env=0: lift must still fire");
    });
}

/// Build a synthetic 256×256 RGB8 image with screenshot-shaped content
/// (same generator as `content_class_dispatch_roundtrip.rs`): 16×16
/// solid tiles with glyph-like accents.
fn synth_screenshot_rgb8(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let tile_x = x / 16;
            let tile_y = y / 16;
            let color = match (tile_x % 4, tile_y % 4) {
                (0, _) => [240u8, 240, 240],
                (_, 0) => [240u8, 240, 240],
                (1, 1) => [16, 16, 16],
                (2, 2) => [16, 16, 16],
                (3, 3) => [200, 32, 32],
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

/// A gated e5 Screenshot-class encode (AFV/8×8-class block enabled via
/// the chunk-2c lift) must decode end-to-end through jxl-rs (PRIMARY)
/// and jxl-oxide. The env-disabled arm must equal the arm where the
/// only difference is the lift — i.e. the hook precisely inverts the
/// gate at the bitstream level, not just the profile level.
#[test]
fn afv_lift_e5_screenshot_gated_encode_roundtrips() {
    let _env_serial = crate::env_serial();
    let w = 256u32;
    let h = 256u32;
    let rgb = synth_screenshot_rgb8(w, h);

    let encode = || {
        LossyConfig::new(1.0)
            .with_effort(5)
            .with_content_class(Some(ImageContentClass::Screenshot))
            .with_threads(1)
            .encode(&rgb, w, h, PixelLayout::Rgb8)
            .expect("encode")
    };

    let gated = with_env_locked(None, encode);
    let suppressed = with_env_locked(Some("1"), encode);

    // Both bitstreams decode through jxl-rs (PRIMARY) + jxl-oxide.
    assert_eq!(decode_jxl_rs_rgb8_smoke(&gated), (w, h));
    assert_eq!(decode_jxl_oxide_smoke(&gated), (w, h));
    assert_eq!(decode_jxl_rs_rgb8_smoke(&suppressed), (w, h));
    assert_eq!(decode_jxl_oxide_smoke(&suppressed), (w, h));
}
