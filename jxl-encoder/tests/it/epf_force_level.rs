// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! `LossyConfig::with_epf_level` — `--epf -1..3` override.
//!
//! Mirrors libjxl `enc_frame.cc:284-285` (`cparams.epf != -1` overrides the
//! distance-derived `LoopFilter.epf_iters`). At `d=1.0` the default
//! distance-based selection yields `epf_iters=1`; we sanity-check that
//! forcing levels 0..=3 produces decodable output and that the byte streams
//! genuinely differ from the default.

use jxl_encoder::{LossyConfig, PixelLayout};

/// Minimal jxl-rs decode wrapper — width/height only, just to confirm the
/// forced bitstream is structurally valid. (Pixel-level EPF behaviour
/// differences are exercised by the per-block sharpness unit tests in
/// `src/vardct/epf.rs`; here we only need to prove the override reached
/// the bitstream.)
fn decode_dims(data: &[u8]) -> (usize, usize) {
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
            Err(e) => panic!("jxl-rs header decode error: {:?}", e),
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

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input before frame");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {:?}", e),
        }
    };

    let mut output_image =
        Image::<f32>::new((width * channels, height)).expect("alloc output buffer");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
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
            Err(e) => panic!("jxl-rs decode error: {:?}", e),
        }
    }

    (width, height)
}

/// 128×128 mixed-gradient test image. Smooth horizontal ramp plus a
/// diagonal stripe — enough block boundaries that EPF on/off produces
/// different reconstruction errors, but small enough to keep the test
/// cheap.
fn mixed_gradient_128() -> Vec<u8> {
    let w: usize = 128;
    let h: usize = 128;
    let mut buf = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let stripe = ((x + y) % 32) as u8;
            let ramp = (x * 2) as u8;
            buf.push(ramp);
            buf.push(stripe);
            buf.push(255 - ramp);
        }
    }
    buf
}

fn encode_with_epf(data: &[u8], w: u32, h: u32, level: i8) -> Vec<u8> {
    LossyConfig::new(1.0)
        .with_effort(5)
        .with_epf_level(level)
        .encode(data, w, h, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("encode failed at epf_level={level}: {e}"))
}

#[test]
fn epf_force_level_default_roundtrips() {
    let data = mixed_gradient_128();
    let encoded = encode_with_epf(&data, 128, 128, -1);
    assert_eq!(&encoded[..2], &[0xFF, 0x0A]);
    let (dw, dh) = decode_dims(&encoded);
    assert_eq!((dw, dh), (128, 128));
}

#[test]
fn epf_force_level_changes_bitstream() {
    // d=1.0 + default profile → distance-derived epf_iters=1.
    // Forcing 0 and 3 must yield distinct byte streams from the default
    // and from each other; otherwise the override isn't actually wired
    // into the encoder pipeline.
    let data = mixed_gradient_128();
    let bytes_auto = encode_with_epf(&data, 128, 128, -1);
    let bytes_off = encode_with_epf(&data, 128, 128, 0);
    let bytes_max = encode_with_epf(&data, 128, 128, 3);

    assert_ne!(
        bytes_auto, bytes_off,
        "epf=-1 (auto) and epf=0 (forced off) must produce different bitstreams \
         (epf=0 disables both the filter and the per-block sharpness search)"
    );
    assert_ne!(
        bytes_auto, bytes_max,
        "epf=-1 (auto, 1 iter at d=1.0) and epf=3 (forced 3 iters) must produce \
         different bitstreams (epf_iters in the frame header differs, and the \
         sharpness map is recomputed for a heavier filter)"
    );
    assert_ne!(
        bytes_off, bytes_max,
        "epf=0 (off) and epf=3 (max) must produce different bitstreams"
    );

    // Each forced level must still round-trip through jxl-rs.
    for (label, bytes) in [
        ("auto", &bytes_auto),
        ("off", &bytes_off),
        ("max", &bytes_max),
    ] {
        assert_eq!(&bytes[..2], &[0xFF, 0x0A], "bad JXL signature at {label}");
        let (dw, dh) = decode_dims(bytes);
        assert_eq!((dw, dh), (128, 128), "decode dims mismatch at {label}");
    }
}

#[test]
fn epf_force_each_level_roundtrips() {
    // Every -1..=3 level must produce a decodable bitstream.
    let data = mixed_gradient_128();
    for level in [-1i8, 0, 1, 2, 3] {
        let encoded = encode_with_epf(&data, 128, 128, level);
        assert_eq!(&encoded[..2], &[0xFF, 0x0A], "bad signature at epf={level}");
        let (dw, dh) = decode_dims(&encoded);
        assert_eq!((dw, dh), (128, 128), "decode mismatch at epf={level}");
    }
}
