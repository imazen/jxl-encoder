// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Roundtrip tests for the four IEEE 754 binary16 (f16) input pixel
//! layouts added under A1 audit "Pixel formats / extras":
//!
//! - [`PixelLayout::RgbLinearF16`]   (6 bytes/pixel — 3 × u16 native-endian)
//! - [`PixelLayout::RgbaLinearF16`]  (8 bytes/pixel — 4 × u16 native-endian)
//! - [`PixelLayout::GrayLinearF16`]  (2 bytes/pixel — 1 × u16 native-endian)
//! - [`PixelLayout::GrayAlphaLinearF16`] (4 bytes/pixel — 2 × u16 native-endian)
//!
//! Storage convention matches the corresponding f32 linear layouts —
//! values are in linear light, channels interleaved, with the encoder
//! expanding f16 → f32 internally before XYB conversion. The
//! `BitDepth::float16()` signaling has been present for some time; this
//! file is the missing input *wiring's* verification.
//!
//! Encoding path: lossy (matches the f32 linear path — the lossless
//! modular path does not accept linear-float layouts in this crate;
//! see the `UnsupportedPixelLayout` branch at `api.rs:4308`). The
//! `LossyConfig::with_gaborish(false)` toggle mirrors the existing
//! `test_lossy_rgba_linear_f32` smoke test so the comparison is
//! apples-to-apples.
//!
//! Decoder path: encode via the public API, decode via both `jxl-rs`
//! (primary, per CLAUDE.md) and `jxl-oxide` (secondary), then compare
//! decoded pixels against the f16-quantized expected values. The
//! tolerance accommodates the lossy quantization at distance 2.0 (we
//! only assert that decode succeeds and channel values land in
//! `[0, 1.5]` — the wiring contract, not bit-exact pixel parity).

use jxl_encoder::api::{LossyConfig, PixelLayout};

/// Tolerance for the f16 → encode → decode → f32 channel comparison.
///
/// The encoding is lossy at distance 0.5 ("perceptually lossless");
/// f16 input adds at most ~1e-3 of f16-quantization noise on top.
/// We use 0.07 on the linear [0, 1] range — this is much larger
/// than the encoder's f16→f32 expansion error (~1e-3 worst case)
/// but tight enough to catch wiring bugs (which would produce >0.5
/// errors, NaN, or all-zero output).
const F16_TOL: f32 = 0.07;

/// Lossy butteraugli distance used for the f16 wiring roundtrip. At
/// d=0.5 the per-pixel quantization is small enough that the f16
/// quantum dominates the noise floor on a 16x16 synthetic image —
/// any pixel-level deviation beyond [`F16_TOL`] points at a wiring
/// bug, not at codec quality.
const LOSSY_DIST: f32 = 0.5;

/// Test-side f32 → f16 packer. Mirrors the encoder-private
/// `f16::f32_to_f16_bits` (see `jxl-encoder/src/f16.rs`) for the
/// representable range only — values must be finite, |v| ≤ 65504, and
/// the inputs in this test are all small powers-of-two fractions which
/// are exact in f16 (so the roundtrip-after-quantize equality holds
/// without invoking the encoder's truncation logic). The duplication
/// keeps the `f16` module `pub(crate)` so we don't widen the public
/// API just for a test fixture.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exp == 0 && mantissa == 0 {
        return (sign << 15) as u16;
    }
    assert!(exp != 0xFF, "test f16 inputs must not be Inf/NaN");
    let new_exp = exp - 127 + 15;
    assert!(new_exp < 31, "test f16 inputs must fit (got {value})");
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let m = mantissa | 0x80_0000;
        let shift = 1 - new_exp;
        let half = (m >> (13 + shift)) as u16;
        return ((sign << 15) as u16) | half;
    }
    let half_mantissa = (mantissa >> 13) as u16;
    let half_exp = (new_exp as u16) << 10;
    ((sign << 15) as u16) | half_exp | half_mantissa
}

/// Inverse of [`f32_to_f16_bits`]; computes the value the decoder will
/// reconstruct so the test can compare against "what got encoded",
/// not "what the caller passed in".
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mantissa;
        let mut e: i32 = -1;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    // Normal — Inf/NaN unused in tests.
    let f32_exp = (exp as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// f32 → f16 → f32 roundtrip — the value the decoder will see.
fn f16_roundtrip(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

/// Pack an [f32] slice as native-endian u16 f16 bits — the storage
/// shape callers pass to the encoder for the f16 input layouts.
fn pack_f16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_f16_bits(v).to_ne_bytes());
    }
    out
}

/// Build a synthetic 16x16 linear-RGB image as f32, with values that
/// roundtrip exactly through f16 (powers-of-two fractions). Returns
/// `(rgb_f32, alpha_f32)` — interleaved 3-channel + interleaved alpha.
fn synth_rgb_f32() -> (Vec<f32>, Vec<f32>) {
    let mut rgb = Vec::with_capacity(16 * 16 * 3);
    let mut alpha = Vec::with_capacity(16 * 16);
    for y in 0..16 {
        for x in 0..16 {
            // Halves of 1/16 — exact f16 representation.
            let r = (x as f32) / 16.0;
            let g = (y as f32) / 16.0;
            let b = ((x + y) as f32) / 32.0;
            let a = 1.0 - ((x ^ y) as f32) / 32.0;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
            alpha.push(a);
        }
    }
    (rgb, alpha)
}

/// Decode `jxl` via jxl-rs and return the decoded RGB f32 framebuffer
/// (extras discarded, RGB requested explicitly so the value comparison
/// can ignore alpha synthesis differences between encoders).
///
/// jxl-rs is the primary roundtrip decoder per project CLAUDE.md.
fn decode_jxl_rs_rgb_f32(jxl: &[u8]) -> (u32, u32, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = jxl;
    let opts = JxlDecoderOptions::default();
    let init = JxlDecoder::<states::Initialized>::new(opts);

    let mut decoder_init = init;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };
    let basic_info = decoder.basic_info().clone();
    let (w, h) = basic_info.size;
    let num_extra = basic_info.extra_channels.len();
    let rgb_channels = 3;

    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); num_extra],
    });

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info error: {e:?}"),
        }
    };

    let mut output_image = Image::<f32>::new((w * rgb_channels, h)).expect("alloc rgb f32 buffer");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (w * rgb_channels, h),
            })
            .into_raw(),
    )];
    let mut extra_images: Vec<Image<f32>> = (0..num_extra)
        .map(|_| Image::<f32>::new((w, h)).expect("alloc extra"))
        .collect();
    for extra in &mut extra_images {
        buffers.push(JxlOutputBuffer::from_image_rect_mut(
            extra
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (w, h),
                })
                .into_raw(),
        ));
    }

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }

    let mut pixels = Vec::with_capacity(w * h * rgb_channels);
    for y in 0..h {
        pixels.extend_from_slice(output_image.row(y));
    }
    (w as u32, h as u32, pixels)
}

/// Decode `jxl` via jxl-oxide in linear-sRGB f32 (per project CLAUDE.md
/// — metric-safe decode path that does NOT double-apply sRGB gamma).
fn decode_jxl_oxide_rgba_f32(jxl: &[u8]) -> (u32, u32, usize, Vec<f32>) {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(jxl))
        .expect("jxl-oxide read");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).expect("jxl-oxide render");
    let fb = render.image_all_channels();
    let w = fb.width() as u32;
    let h = fb.height() as u32;
    let ch = fb.channels();
    let buf = fb.buf().to_vec();
    (w, h, ch, buf)
}

#[test]
fn rgb_linear_f16_roundtrips_through_jxl_rs_and_jxl_oxide() {
    let (rgb, _alpha) = synth_rgb_f32();
    let f16_bytes = pack_f16(&rgb);

    // Encode lossy from f16 RGB input.
    let jxl = LossyConfig::new(LOSSY_DIST)
        .with_gaborish(false)
        .encode_request(16, 16, PixelLayout::RgbLinearF16)
        .encode(&f16_bytes)
        .expect("lossy RgbLinearF16 must encode");
    assert_eq!(&jxl[..2], &[0xFF, 0x0A], "must be a bare JXL codestream");

    // jxl-rs (primary).
    let (w_rs, h_rs, dec_rgb) = decode_jxl_rs_rgb_f32(&jxl);
    assert_eq!((w_rs, h_rs), (16, 16));
    let mut max_diff = 0f32;
    for i in 0..(16 * 16) {
        for c in 0..3 {
            let want = f16_roundtrip(rgb[i * 3 + c]);
            let got = dec_rgb[i * 3 + c];
            let d = (got - want).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d <= F16_TOL,
                "jxl-rs: pixel {i} ch {c}: got {got} want {want} diff {d}"
            );
        }
    }
    eprintln!("RgbLinearF16 jxl-rs max channel diff: {max_diff}");

    // jxl-oxide (secondary).
    let (w, h, ch, decoded) = decode_jxl_oxide_rgba_f32(&jxl);
    assert_eq!((w, h), (16, 16));
    let mut max_diff = 0f32;
    for i in 0..(16 * 16) {
        for c in 0..3 {
            let want = f16_roundtrip(rgb[i * 3 + c]);
            let got = decoded[i * ch + c];
            let d = (got - want).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d <= F16_TOL,
                "jxl-oxide: pixel {i} ch {c}: got {got} want {want} diff {d}"
            );
        }
    }
    eprintln!("RgbLinearF16 jxl-oxide max channel diff: {max_diff}");
}

#[test]
fn rgba_linear_f16_roundtrips_through_jxl_oxide() {
    let (rgb, alpha) = synth_rgb_f32();
    // Interleave RGBA as f32, then pack to f16.
    let mut rgba_f32 = Vec::with_capacity(16 * 16 * 4);
    for i in 0..(16 * 16) {
        rgba_f32.push(rgb[i * 3]);
        rgba_f32.push(rgb[i * 3 + 1]);
        rgba_f32.push(rgb[i * 3 + 2]);
        rgba_f32.push(alpha[i]);
    }
    let f16_bytes = pack_f16(&rgba_f32);

    let jxl = LossyConfig::new(LOSSY_DIST)
        .with_gaborish(false)
        .encode_request(16, 16, PixelLayout::RgbaLinearF16)
        .encode(&f16_bytes)
        .expect("lossy RgbaLinearF16 must encode");

    // jxl-rs (primary) — RGB only, but file must carry one extra channel
    // (alpha) per the basic_info round-trip.
    let (w_rs, h_rs, dec_rgb) = decode_jxl_rs_rgb_f32(&jxl);
    assert_eq!((w_rs, h_rs), (16, 16));
    let mut max_rgb_diff_rs = 0f32;
    for i in 0..(16 * 16) {
        for c in 0..3 {
            let want = f16_roundtrip(rgb[i * 3 + c]);
            let got = dec_rgb[i * 3 + c];
            let d = (got - want).abs();
            if d > max_rgb_diff_rs {
                max_rgb_diff_rs = d;
            }
            assert!(d <= F16_TOL, "jxl-rs RGBA px {i} ch {c}: diff {d}");
        }
    }
    eprintln!("RgbaLinearF16 jxl-rs max RGB diff: {max_rgb_diff_rs}");

    let (w, h, ch, decoded) = decode_jxl_oxide_rgba_f32(&jxl);
    assert_eq!((w, h), (16, 16));
    assert!(ch >= 4, "expected 4 channels, got {ch}");

    let mut max_rgb_diff = 0f32;
    let mut max_alpha_diff = 0f32;
    for i in 0..(16 * 16) {
        for c in 0..3 {
            let want = f16_roundtrip(rgb[i * 3 + c]);
            let got = decoded[i * ch + c];
            let d = (got - want).abs();
            if d > max_rgb_diff {
                max_rgb_diff = d;
            }
            assert!(d <= F16_TOL, "RGBA RGB ch {c} pixel {i}: diff {d}");
        }
        // Alpha is round-tripped through u8 (extract_alpha_f16 → u8),
        // so tolerance there is ~1/255 ≈ 4e-3 plus the f16 quantum.
        let want_a = f16_roundtrip(alpha[i]);
        let got_a = decoded[i * ch + 3];
        let d = (got_a - want_a).abs();
        if d > max_alpha_diff {
            max_alpha_diff = d;
        }
        assert!(d <= F16_TOL + 1.0 / 255.0, "alpha pixel {i}: diff {d}");
    }
    eprintln!(
        "RgbaLinearF16 jxl-oxide max RGB diff {max_rgb_diff} max alpha diff {max_alpha_diff}"
    );
}

#[test]
fn gray_linear_f16_roundtrips_through_jxl_oxide() {
    // Build a 16x16 gray f32 image with values that roundtrip through f16.
    let mut gray_f32 = Vec::with_capacity(16 * 16);
    for y in 0..16 {
        for x in 0..16 {
            gray_f32.push(((x ^ y) as f32) / 32.0);
        }
    }
    let f16_bytes = pack_f16(&gray_f32);

    let jxl = LossyConfig::new(LOSSY_DIST)
        .with_gaborish(false)
        .encode_request(16, 16, PixelLayout::GrayLinearF16)
        .encode(&f16_bytes)
        .expect("lossy GrayLinearF16 must encode");

    let (w, h, ch, decoded) = decode_jxl_oxide_rgba_f32(&jxl);
    assert_eq!((w, h), (16, 16));
    // jxl-oxide may decode grayscale as a single-channel framebuffer
    // OR expand to RGBA depending on color encoding negotiation. Both
    // paths must agree on the gray value.
    let mut max_diff = 0f32;
    for i in 0..(16 * 16) {
        let want = f16_roundtrip(gray_f32[i]);
        let got = decoded[i * ch]; // first channel (gray or R, equal)
        let d = (got - want).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= F16_TOL,
            "GrayLinearF16 px {i}: got {got} want {want} diff {d}"
        );
    }
    eprintln!("GrayLinearF16 jxl-oxide max diff: {max_diff}");
}

#[test]
fn gray_alpha_linear_f16_roundtrips_through_jxl_oxide() {
    let mut interleaved = Vec::with_capacity(16 * 16 * 2);
    for y in 0..16 {
        for x in 0..16 {
            let gray = ((x ^ y) as f32) / 32.0;
            let alpha = 1.0 - (x as f32) / 32.0;
            interleaved.push(gray);
            interleaved.push(alpha);
        }
    }
    let f16_bytes = pack_f16(&interleaved);

    let jxl = LossyConfig::new(LOSSY_DIST)
        .with_gaborish(false)
        .encode_request(16, 16, PixelLayout::GrayAlphaLinearF16)
        .encode(&f16_bytes)
        .expect("lossy GrayAlphaLinearF16 must encode");

    let (w, h, ch, decoded) = decode_jxl_oxide_rgba_f32(&jxl);
    assert_eq!((w, h), (16, 16));
    assert!(ch >= 2, "expected >= 2 channels, got {ch}");

    let mut max_gray_diff = 0f32;
    let mut max_alpha_diff = 0f32;
    for i in 0..(16 * 16) {
        let want_gray = f16_roundtrip(interleaved[i * 2]);
        let want_alpha = f16_roundtrip(interleaved[i * 2 + 1]);
        // Gray decodes to the first 1 or 3 channels depending on the
        // negotiated color encoding; alpha sits at the last channel.
        let got_gray = decoded[i * ch];
        let got_alpha = decoded[i * ch + ch - 1];
        let dg = (got_gray - want_gray).abs();
        let da = (got_alpha - want_alpha).abs();
        if dg > max_gray_diff {
            max_gray_diff = dg;
        }
        if da > max_alpha_diff {
            max_alpha_diff = da;
        }
        assert!(dg <= F16_TOL, "px {i} gray: diff {dg}");
        // Alpha rides through u8 again.
        assert!(da <= F16_TOL + 1.0 / 255.0, "px {i} alpha: diff {da}");
    }
    eprintln!("GrayAlphaLinearF16 jxl-oxide max gray {max_gray_diff} max alpha {max_alpha_diff}");
}

#[test]
fn rgb_linear_f16_metadata_helpers_are_consistent() {
    // Sanity: every metadata helper on PixelLayout returns the same
    // answer for f16 as for the corresponding f32 layout (except
    // is_f32 / is_f16 / bytes_per_pixel).
    assert!(PixelLayout::RgbLinearF16.is_linear());
    assert!(PixelLayout::RgbaLinearF16.is_linear());
    assert!(PixelLayout::GrayLinearF16.is_linear());
    assert!(PixelLayout::GrayAlphaLinearF16.is_linear());

    assert!(PixelLayout::RgbLinearF16.is_f16());
    assert!(!PixelLayout::RgbLinearF16.is_f32());
    assert!(!PixelLayout::RgbLinearF32.is_f16());

    assert!(!PixelLayout::RgbLinearF16.has_alpha());
    assert!(PixelLayout::RgbaLinearF16.has_alpha());
    assert!(!PixelLayout::GrayLinearF16.has_alpha());
    assert!(PixelLayout::GrayAlphaLinearF16.has_alpha());

    assert!(!PixelLayout::RgbLinearF16.is_grayscale());
    assert!(!PixelLayout::RgbaLinearF16.is_grayscale());
    assert!(PixelLayout::GrayLinearF16.is_grayscale());
    assert!(PixelLayout::GrayAlphaLinearF16.is_grayscale());

    assert_eq!(PixelLayout::RgbLinearF16.bytes_per_pixel(), 6);
    assert_eq!(PixelLayout::RgbaLinearF16.bytes_per_pixel(), 8);
    assert_eq!(PixelLayout::GrayLinearF16.bytes_per_pixel(), 2);
    assert_eq!(PixelLayout::GrayAlphaLinearF16.bytes_per_pixel(), 4);
}
