// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Roundtrip test for `Primaries::Custom` + `CustomPrimaries` signaling.
//!
//! Encodes a tiny image with custom RGB primaries (close to DCI-P3 but
//! slightly off so the values cannot have been confused with a built-in
//! enum), then decodes via jxl-rs (primary) and jxl-oxide (secondary) and
//! asserts the decoded color encoding carries the same primaries within the
//! fixed-point quantization tolerance.
//!
//! Wire format: each xy coordinate is stored as a signed 32-bit integer
//! `round(value * 1_000_000)` (see `headers/color_encoding.rs:43-44`), so
//! the round-trip error per coordinate is at most `0.5e-6` in xy space.
//! We assert with a generous `2e-6` tolerance.

use jxl_encoder::{CIExy, ColorEncoding, CustomPrimaries, Primaries};
use jxl_encoder::{LosslessConfig, PixelLayout};

/// Tolerance for round-trip of CIE xy coordinates through the JXL bitstream
/// fixed-point encoding (`round(value * 1e6)`).
const XY_TOLERANCE: f32 = 2.0e-6;

/// Custom primaries used for the round-trip — close to DCI-P3 but each
/// coordinate is nudged so the values can't have accidentally hit the
/// built-in `Primaries::P3` enum on the decode side.
fn slightly_off_p3() -> CustomPrimaries {
    CustomPrimaries {
        // DCI-P3 reference: red (0.680, 0.320), green (0.265, 0.690), blue (0.150, 0.060)
        red: CIExy::new(0.681_234, 0.319_876),
        green: CIExy::new(0.264_321, 0.690_555),
        blue: CIExy::new(0.151_111, 0.060_222),
    }
}

/// Build a 32×32 RGB8 image with a simple gradient.
fn build_rgb8_image(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 8) as u8);
            pixels.push((y * 8) as u8);
            pixels.push(((x ^ y) * 4) as u8);
        }
    }
    pixels
}

/// Encode a small image with `ColorEncoding` set to custom primaries.
///
/// Goes through `LosslessConfig::encode_request(...).with_color_encoding(...)`
/// because `with_color_encoding` lives on `EncodeRequest`, not on
/// `LosslessConfig` itself.
fn encode_with_custom_primaries(ce: ColorEncoding) -> Vec<u8> {
    let w = 32u32;
    let h = 32u32;
    let pixels = build_rgb8_image(w, h);

    let cfg = LosslessConfig::new();
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .with_color_encoding(ce)
        .encode(&pixels)
        .expect("encode with custom primaries should succeed")
}

/// Decode the codestream via jxl-rs and pull out the embedded color profile.
fn jxl_rs_decoded_primaries(data: &[u8]) -> jxl::api::JxlColorEncoding {
    use jxl::api::{JxlColorProfile, JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder_init = JxlDecoder::<states::Initialized>::new(options);

    let decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };

    match decoder.embedded_color_profile() {
        JxlColorProfile::Simple(enc) => enc.clone(),
        JxlColorProfile::Icc(_) => {
            panic!("jxl-rs reported an ICC profile but we asked for an enum encoding")
        }
    }
}

/// Decode via jxl-oxide and pull out the enum color encoding from the
/// embedded image header.
fn jxl_oxide_decoded_primaries(data: &[u8]) -> jxl_oxide::color::Primaries {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide read");

    let header = image.image_header();
    match &header.metadata.colour_encoding {
        jxl_oxide::color::ColourEncoding::Enum(enc) => enc.primaries,
        jxl_oxide::color::ColourEncoding::IccProfile(_) => {
            panic!("jxl-oxide reported ICC profile; expected enum encoding")
        }
    }
}

#[test]
fn lossless_custom_primaries_roundtrip_jxl_rs() {
    use jxl::api::{JxlColorEncoding, JxlPrimaries};

    let cp = slightly_off_p3();
    let ce = ColorEncoding::with_custom_primaries(cp);
    assert_eq!(ce.primaries, Primaries::Custom);

    let jxl = encode_with_custom_primaries(ce);
    assert_eq!(
        &jxl[..2],
        &[0xFF, 0x0A],
        "expected bare codestream signature"
    );

    let decoded = jxl_rs_decoded_primaries(&jxl);
    let JxlColorEncoding::RgbColorSpace { primaries, .. } = decoded else {
        panic!("expected RGB color space, got {decoded:?}");
    };

    let JxlPrimaries::Chromaticities {
        rx,
        ry,
        gx,
        gy,
        bx,
        by,
    } = primaries
    else {
        panic!(
            "expected custom Chromaticities, got {primaries:?} — decoder may have collapsed to a built-in enum"
        );
    };

    assert!(
        (rx - cp.red.x as f32).abs() < XY_TOLERANCE,
        "red.x: got {rx}, want {} (tol {XY_TOLERANCE})",
        cp.red.x
    );
    assert!(
        (ry - cp.red.y as f32).abs() < XY_TOLERANCE,
        "red.y: got {ry}, want {} (tol {XY_TOLERANCE})",
        cp.red.y
    );
    assert!(
        (gx - cp.green.x as f32).abs() < XY_TOLERANCE,
        "green.x: got {gx}, want {} (tol {XY_TOLERANCE})",
        cp.green.x
    );
    assert!(
        (gy - cp.green.y as f32).abs() < XY_TOLERANCE,
        "green.y: got {gy}, want {} (tol {XY_TOLERANCE})",
        cp.green.y
    );
    assert!(
        (bx - cp.blue.x as f32).abs() < XY_TOLERANCE,
        "blue.x: got {bx}, want {} (tol {XY_TOLERANCE})",
        cp.blue.x
    );
    assert!(
        (by - cp.blue.y as f32).abs() < XY_TOLERANCE,
        "blue.y: got {by}, want {} (tol {XY_TOLERANCE})",
        cp.blue.y
    );
}

#[test]
fn lossless_custom_primaries_roundtrip_jxl_oxide() {
    let cp = slightly_off_p3();
    let ce = ColorEncoding::with_custom_primaries(cp);

    let jxl = encode_with_custom_primaries(ce);

    let prim = jxl_oxide_decoded_primaries(&jxl);
    let jxl_oxide::color::Primaries::Custom { red, green, blue } = prim else {
        panic!(
            "expected jxl_oxide::color::Primaries::Custom, got {prim:?} — decoder may have collapsed to a built-in enum"
        );
    };

    let [rx, ry] = red.as_float();
    let [gx, gy] = green.as_float();
    let [bx, by] = blue.as_float();

    assert!(
        (rx - cp.red.x as f32).abs() < XY_TOLERANCE,
        "red.x {rx} vs {}",
        cp.red.x
    );
    assert!(
        (ry - cp.red.y as f32).abs() < XY_TOLERANCE,
        "red.y {ry} vs {}",
        cp.red.y
    );
    assert!(
        (gx - cp.green.x as f32).abs() < XY_TOLERANCE,
        "green.x {gx} vs {}",
        cp.green.x
    );
    assert!(
        (gy - cp.green.y as f32).abs() < XY_TOLERANCE,
        "green.y {gy} vs {}",
        cp.green.y
    );
    assert!(
        (bx - cp.blue.x as f32).abs() < XY_TOLERANCE,
        "blue.x {bx} vs {}",
        cp.blue.x
    );
    assert!(
        (by - cp.blue.y as f32).abs() < XY_TOLERANCE,
        "blue.y {by} vs {}",
        cp.blue.y
    );
}

/// Helper used by the `--ignored` djxl-dump test below. Writes the
/// codestream that the other two tests verify to a known path so it can
/// be fed to `djxl` (reference C++ decoder) manually.
#[test]
#[ignore = "writes /tmp/custom_primaries_test.jxl for manual djxl verification"]
fn write_custom_primaries_codestream_for_djxl() {
    let cp = slightly_off_p3();
    let ce = ColorEncoding::with_custom_primaries(cp);
    let jxl = encode_with_custom_primaries(ce);
    let path = "/tmp/custom_primaries_test.jxl";
    std::fs::write(path, &jxl).expect("write jxl");
    eprintln!("wrote {path} ({} bytes)", jxl.len());
}

/// Negative-control: encoding with built-in `Primaries::P3` must NOT
/// surface as `Primaries::Custom` on the decode side. Guards against an
/// encoder bug where the custom-primaries write path silently runs for
/// non-Custom primaries.
#[test]
fn lossless_builtin_p3_primaries_not_signaled_as_custom() {
    use jxl::api::{JxlColorEncoding, JxlPrimaries};

    let ce = ColorEncoding::display_p3();
    assert_eq!(ce.primaries, Primaries::P3);

    let jxl = encode_with_custom_primaries(ce);
    let decoded = jxl_rs_decoded_primaries(&jxl);
    let JxlColorEncoding::RgbColorSpace { primaries, .. } = decoded else {
        panic!("expected RGB color space, got {decoded:?}");
    };
    assert!(
        matches!(primaries, JxlPrimaries::P3),
        "Display P3 should decode as JxlPrimaries::P3, got {primaries:?}"
    );
}
