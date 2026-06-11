// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-176 multi-decoder roundtrip test for the terminal-class exclude.
//!
//! Verifies:
//! 1. The W44-176 production output (Zenjxl strategy with the W44-176
//!    exclude ON) decodes cleanly via jxl-rs AND jxl-oxide on terminal
//!    e7 d=4/5 (the cells where the discriminator fires and the W44-109
//!    lift is suppressed).
//! 2. The Libjxl strategy is byte-identical regardless of W44-176 (the
//!    `terminal_class_exclude` flag is `false` on Libjxl + `adaptive_quant_qf_seed:
//!    Off` already suppresses the helper before W44-176 logic runs).
//! 3. `JXL_W44_176_DISABLE=1` (Mode A — force OFF) produces bitstreams
//!    that ALSO decode cleanly (regression guard against the env hook).
//!
//! Run with:
//!   cargo test --release -p jxl-encoder --features parallel \
//!     --test w44_176_decoder_roundtrip

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};

const CELLS: &[(&str, &str, u8, f32)] = &[
    ("terminal_e7_d4", "gb82-sc/terminal.png", 7, 4.0),
    ("terminal_e7_d5", "gb82-sc/terminal.png", 7, 5.0),
];

fn decode_oxide(bytes: &[u8]) -> Result<(u32, u32), String> {
    let img = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(bytes))
        .map_err(|e| format!("oxide read: {}", e))?;
    let w = img.width();
    let h = img.height();
    let _ = img
        .render_frame(0)
        .map_err(|e| format!("oxide render: {}", e))?;
    Ok((w, h))
}

fn decode_jxl_rs(bytes: &[u8]) -> Result<(usize, usize), String> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = bytes;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => return Err(format!("jxl-rs header: {:?}", e)),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let channels = 3;
    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame info: {:?}", e)),
        }
    };
    let mut output_image = Image::<f32>::new((width * channels, height))
        .map_err(|e| format!("output alloc: {:?}", e))?;
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];
    let _decoder = loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame: {:?}", e)),
        }
    };
    Ok((width, height))
}

#[test]
#[ignore = "needs codec-corpus (CODEC_CORPUS_DIR); nightly + local run with --include-ignored"]
fn w44_176_zenjxl_default_decodes_cleanly() {
    let _env_serial = crate::env_serial();
    for &(cell_name, path, effort, distance) in CELLS {
        let path = &crate::corpus_file(path);
        let img = match image::open(path) {
            Ok(i) => i,
            Err(_) => panic!("W44-176 corpus missing: {}", path),
        };
        let rgb_img = img.to_rgb8();
        let (w, h) = (rgb_img.width(), rgb_img.height());
        let rgb = rgb_img.into_raw();
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Zenjxl);
        let bytes = cfg
            .encode(&rgb, w, h, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{cell_name}: Zenjxl encode failed: {e:?}"));
        eprintln!("[{cell_name}] Zenjxl bytes={}", bytes.len());

        let (ow, oh) = decode_oxide(&bytes)
            .unwrap_or_else(|e| panic!("{cell_name}: jxl-oxide decode failed: {e}"));
        assert_eq!(ow, w, "{cell_name}: jxl-oxide width mismatch");
        assert_eq!(oh, h, "{cell_name}: jxl-oxide height mismatch");

        let (rw, rh) = decode_jxl_rs(&bytes)
            .unwrap_or_else(|e| panic!("{cell_name}: jxl-rs decode failed: {e}"));
        assert_eq!(rw, w as usize, "{cell_name}: jxl-rs width mismatch");
        assert_eq!(rh, h as usize, "{cell_name}: jxl-rs height mismatch");
    }
}

#[test]
#[ignore = "needs codec-corpus (CODEC_CORPUS_DIR); nightly + local run with --include-ignored"]
fn w44_176_libjxl_strategy_decodes_cleanly() {
    let _env_serial = crate::env_serial();
    // Libjxl strategy disables `adaptive_quant_qf_seed` via the
    // `AdaptiveQuantQfSeedPolicy::Off` setting; the helper short-
    // circuits before evaluating the W44-176 discriminator. This
    // test verifies the resulting bitstream decodes cleanly (the
    // Libjxl hash-lock test in `tests/strategy_libjxl_hash_locks.rs`
    // covers the byte-identical-to-pre-W44-176 invariant via its
    // pinned-fixture comparison).
    for &(cell_name, path, effort, distance) in CELLS {
        let path = &crate::corpus_file(path);
        let img = match image::open(path) {
            Ok(i) => i,
            Err(_) => panic!("W44-176 corpus missing: {}", path),
        };
        let rgb_img = img.to_rgb8();
        let (w, h) = (rgb_img.width(), rgb_img.height());
        let rgb = rgb_img.into_raw();
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl);
        let bytes = cfg
            .encode(&rgb, w, h, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{cell_name}: Libjxl encode failed: {e:?}"));
        eprintln!("[{cell_name}] Libjxl bytes={}", bytes.len());

        let (ow, oh) = decode_oxide(&bytes)
            .unwrap_or_else(|e| panic!("{cell_name}: jxl-oxide decode failed: {e}"));
        assert_eq!(ow, w);
        assert_eq!(oh, h);

        let (rw, rh) = decode_jxl_rs(&bytes)
            .unwrap_or_else(|e| panic!("{cell_name}: jxl-rs decode failed: {e}"));
        assert_eq!(rw, w as usize);
        assert_eq!(rh, h as usize);
    }
}

/// Regression guard for the `JXL_W44_176_DISABLE=1` env hook (Mode A —
/// force exclude OFF). Verifies the resulting bitstream still decodes
/// cleanly when the W44-176 exclude is bypassed.
#[test]
#[ignore = "needs codec-corpus (CODEC_CORPUS_DIR); nightly + local run with --include-ignored"]
fn w44_176_env_disable_decodes_cleanly() {
    let _env_serial = crate::env_serial();
    // SAFETY: single-threaded test runner (`#[test]` runs sequentially
    // unless `--test-threads` set; this test toggles env once and
    // restores).
    let prev = std::env::var("JXL_W44_176_DISABLE").ok();
    unsafe {
        std::env::set_var("JXL_W44_176_DISABLE", "1");
    }

    let (cell_name, path, effort, distance) = CELLS[0];
    let path = &crate::corpus_file(path);
    let img = match image::open(path) {
        Ok(i) => i,
        Err(_) => {
            // Restore env before panicking
            unsafe {
                match prev {
                    Some(v) => std::env::set_var("JXL_W44_176_DISABLE", v),
                    None => std::env::remove_var("JXL_W44_176_DISABLE"),
                }
            }
            panic!("W44-176 corpus missing: {}", path);
        }
    };
    let rgb_img = img.to_rgb8();
    let (w, h) = (rgb_img.width(), rgb_img.height());
    let rgb = rgb_img.into_raw();
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(EncoderStrategy::Zenjxl);
    let bytes = cfg
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("Zenjxl encode with W44-176 DISABLE=1");

    // Restore env immediately after the encode so test pollution stays
    // local even if a later assertion panics.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("JXL_W44_176_DISABLE", v),
            None => std::env::remove_var("JXL_W44_176_DISABLE"),
        }
    }

    eprintln!(
        "[{cell_name}] Zenjxl(W44-176-disabled) bytes={}",
        bytes.len()
    );

    let (ow, oh) = decode_oxide(&bytes).expect("jxl-oxide decode");
    assert_eq!(ow, w);
    assert_eq!(oh, h);

    let (rw, rh) = decode_jxl_rs(&bytes).expect("jxl-rs decode");
    assert_eq!(rw, w as usize);
    assert_eq!(rh, h as usize);
}
