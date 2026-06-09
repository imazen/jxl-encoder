// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-166 Smart-Zenjxl chunk 3 (B1 ship) multi-decoder roundtrip:
//! bitstreams produced with the W44-166 photo admission ON (Mode B,
//! Zenjxl default) on 1418519 d=5/6 e8 MUST decode cleanly via jxl-rs
//! AND jxl-oxide.
//!
//! Per W44-166 bench `benchmarks/w44_166_variant_z_admit_zenjxl_2026-05-21.tsv`:
//! - 1418519 e8 d=5: -3.19% bytes, -0.23 SSIM2 (within ±0.30 budget)
//! - 1418519 e8 d=6: -1.72% bytes, +0.45 SSIM2 (strong win)
//!
//! These cells use the W44-96 variant Z high_colour Z' table newly
//! admitted via mask_p25 >= 85. The mechanism produces a different
//! AC strategy distribution → must verify decoders parse the resulting
//! bitstream cleanly.
//!
//! Run with:
//!   cargo test --release -p jxl-encoder --features parallel \
//!     --test w44_166_decoder_roundtrip
//!
//! Panics if the corpus is missing (no graceful skip per repo policy).

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

const CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "1418519_e8_d5",
        "CID22/CID22-512/validation/1418519.png",
        8,
        5.0,
    ),
    (
        "1418519_e8_d6",
        "CID22/CID22-512/validation/1418519.png",
        8,
        6.0,
    ),
    (
        "1418519_e9_d6",
        "CID22/CID22-512/validation/1418519.png",
        9,
        6.0,
    ),
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

/// W44-166 LIBJXL strategy guard: encoding 1418519 e8 d=5 with
/// `EncoderStrategy::Libjxl` MUST produce byte-identical output
/// regardless of the `JXL_W44_166_VARIANT_Z_ADMIT_MODE` env var
/// (because `photo_variant_z_admit = false` on Libjxl disables the
/// gate entirely). Single-threaded test serialises env var access.
#[test]
#[ignore = "needs codec-corpus (CODEC_CORPUS_DIR); nightly + local run with --include-ignored"]
fn w44_166_libjxl_strategy_byte_identical_regardless_of_env() {
    let path = "CID22/CID22-512/validation/1418519.png";
    let path = &crate::corpus_file(path);
    assert!(
        Path::new(path).exists(),
        "[w44-166] corpus file 1418519.png not found at {}",
        path
    );
    let img = image::open(path).expect("open png");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    // SAFETY: set/remove env var. Test crate runs each #[test] in its
    // own thread but cargo test by default runs tests in parallel.
    // Single-mutation pattern guarded by --test-threads=1 in CI is the
    // canonical pattern for env-mutation tests; here we self-protect by
    // doing both encodes back-to-back inside ONE test.
    unsafe { std::env::remove_var("JXL_W44_166_VARIANT_Z_ADMIT_MODE") };
    let bytes_unset = LossyConfig::new(5.0)
        .with_effort(8)
        .with_threads(4)
        .with_strategy(EncoderStrategy::Libjxl)
        .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("encode unset");
    unsafe { std::env::set_var("JXL_W44_166_VARIANT_Z_ADMIT_MODE", "B") };
    let bytes_b = LossyConfig::new(5.0)
        .with_effort(8)
        .with_threads(4)
        .with_strategy(EncoderStrategy::Libjxl)
        .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("encode B");
    unsafe { std::env::remove_var("JXL_W44_166_VARIANT_Z_ADMIT_MODE") };
    eprintln!(
        "[w44-166] Libjxl 1418519 e8 d=5: env-unset bytes = {}, env=B bytes = {}",
        bytes_unset.len(),
        bytes_b.len()
    );
    assert_eq!(
        bytes_unset, bytes_b,
        "Libjxl strategy MUST be byte-identical regardless of W44-166 env"
    );
}

/// Requires the CID22-512/validation corpus. Per repo "NO GRACEFUL
/// SKIPS" rule, this test panics if the corpus is missing.
#[test]
#[ignore = "needs codec-corpus (CODEC_CORPUS_DIR); nightly + local run with --include-ignored"]
fn w44_166_variant_z_photo_admit_decoders_clean() {
    for &(name, path, effort, distance) in CELLS {
        let path = &crate::corpus_file(path);
        assert!(
            Path::new(path).exists(),
            "[w44-166] corpus file {} not found at {}",
            name,
            path
        );
        let img = image::open(path).expect("open png");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        // Default (Zenjxl) — W44-166 Mode B fires the photo admission
        // on this 1418519 input (mask_p25=88.88, m3=36.84).
        let bytes = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(4)
            .with_strategy(EncoderStrategy::Zenjxl)
            .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
            .expect("encode");
        eprintln!(
            "[w44-166] {} e{} d={:.1} -> {} bytes",
            name,
            effort,
            distance,
            bytes.len()
        );

        let (ow, oh) = decode_oxide(&bytes)
            .unwrap_or_else(|e| panic!("[w44-166] {} oxide decode FAILED: {}", name, e));
        assert_eq!((ow, oh), (w, h), "oxide size mismatch on {}", name);

        let (rw, rh) = decode_jxl_rs(&bytes)
            .unwrap_or_else(|e| panic!("[w44-166] {} jxl-rs decode FAILED: {}", name, e));
        assert_eq!(
            (rw, rh),
            (w as usize, h as usize),
            "jxl-rs size mismatch on {}",
            name
        );
        eprintln!(
            "[w44-166] {} e{} d={:.1} oxide={:?} jxl-rs={:?} OK",
            name,
            effort,
            distance,
            (ow, oh),
            (rw, rh)
        );
    }
}
