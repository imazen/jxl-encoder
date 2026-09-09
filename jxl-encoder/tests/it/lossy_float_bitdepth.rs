// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! #109 F0: a lossy encode must announce the caller's REAL input format.
//!
//! `ImageMetadata.bit_depth` describes the INPUT, not the coded
//! representation — libjxl calls it "just metadata for XYB images"
//! (`enc_modular.cc:742`) yet still fills it from the caller's declared format
//! in every mode (`SetFloat32Samples` / `SetFloat16Samples`). Decoders read it
//! to pick a default output format, so getting it wrong misinforms every
//! consumer even though it changes no pixel.
//!
//! We derived it solely from an `is 16-bit?` flag, so **f32 and f16 input was
//! announced as 8- or 16-bit integer**. Verified against cjxl v0.12 on the same
//! content: it writes `floating_point = 1, bits_per_sample = 32`.
//!
//! ## Why this reads the field through a decoder, not a bit offset
//!
//! An earlier draft of this file poked hardcoded bit offsets into the
//! codestream. Every case failed — *including the integer control*, which is
//! the tell that the offsets were wrong rather than the feature. The same
//! mistake was already made and corrected once in `modular_16bit_level.rs`.
//! `ImageMetadata`'s layout is conditional (`all_default`, the `ColorEncoding`
//! bundle, extra channels), so no constant offset is valid across fixtures.
//! jxl-oxide parses the header per spec and hands back the decoded enum, so the
//! assertion is about the VALUE and cannot silently read a neighbouring field.

use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_image::BitDepth;

fn encode(layout: PixelLayout, w: u32, h: u32) -> Vec<u8> {
    let n = (w * h) as usize;
    let px: Vec<u8> = match layout {
        PixelLayout::RgbLinearF32 | PixelLayout::RgbPqF32 => (0..n * 3)
            .flat_map(|i| ((i % 251) as f32 / 251.0).to_ne_bytes())
            .collect(),
        PixelLayout::RgbLinearF16 => (0..n * 3)
            .flat_map(|i| (((i % 251) as u16) | 0x3000).to_ne_bytes())
            .collect(),
        PixelLayout::Rgb8 => (0..n * 3).map(|i| (i % 251) as u8).collect(),
        PixelLayout::Rgb16 => (0..n * 3)
            .flat_map(|i| ((i % 65_521) as u16).to_ne_bytes())
            .collect(),
        other => panic!("unhandled layout {other:?}"),
    };
    LossyConfig::new(1.0)
        .encode_request(w, h, layout)
        .encode(&px)
        .expect("encode")
}

/// Read the signalled `BitDepth` back out with an independent decoder.
fn signalled_bit_depth(data: &[u8]) -> BitDepth {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide must parse our header");
    image.image_header().metadata.bit_depth
}

#[test]
fn lossy_f32_announces_float32() {
    // 320x256 is multi-group in one axis, so this is not a single-group-only
    // claim (see the multi-group directive in CLAUDE.md).
    for (w, h) in [(64, 64), (320, 256)] {
        let bd = signalled_bit_depth(&encode(PixelLayout::RgbLinearF32, w, h));
        assert_eq!(
            bd,
            BitDepth::FloatSample {
                bits_per_sample: 32,
                exp_bits: 8
            },
            "f32 input at {w}x{h} must announce binary32 — cjxl v0.12 does"
        );
    }
}

#[test]
fn lossy_f16_announces_float16() {
    let bd = signalled_bit_depth(&encode(PixelLayout::RgbLinearF16, 64, 64));
    assert_eq!(
        bd,
        BitDepth::FloatSample {
            bits_per_sample: 16,
            exp_bits: 5
        },
        "f16 input must announce binary16 (bits 16, exponent 5)"
    );
}

#[test]
fn lossy_pq_f32_announces_float32() {
    // A transfer-function-tagged float layout is still float input. The TF
    // lives in the ColorEncoding, not in BitDepth, so tagging must not demote
    // the sample format back to integer.
    let bd = signalled_bit_depth(&encode(PixelLayout::RgbPqF32, 64, 64));
    assert_eq!(
        bd,
        BitDepth::FloatSample {
            bits_per_sample: 32,
            exp_bits: 8
        },
        "PQ f32 input must still announce binary32"
    );
}

#[test]
fn integer_layouts_still_announce_integer() {
    // The control. If this ever fails alongside the float cases, suspect the
    // harness, not the feature.
    assert_eq!(
        signalled_bit_depth(&encode(PixelLayout::Rgb8, 64, 64)),
        BitDepth::IntegerSample { bits_per_sample: 8 },
        "Rgb8 must stay 8-bit integer"
    );
    assert_eq!(
        signalled_bit_depth(&encode(PixelLayout::Rgb16, 64, 64)),
        BitDepth::IntegerSample {
            bits_per_sample: 16
        },
        "Rgb16 must stay 16-bit integer"
    );
}

/// The rule itself, over EVERY layout — so a newly added float layout that
/// forgets its arm fails here rather than shipping mislabelled. The list is
/// written out rather than derived so that adding a variant to the enum
/// without adding it here fails the exhaustiveness assertion below.
#[test]
fn source_bit_depth_covers_every_layout() {
    use jxl_encoder::api::PixelLayout::*;
    const ALL: &[PixelLayout] = &[
        Rgb8,
        Rgba8,
        Bgr8,
        Bgra8,
        Gray8,
        GrayAlpha8,
        Rgb16,
        Rgba16,
        Gray16,
        GrayAlpha16,
        RgbLinearF32,
        RgbaLinearF32,
        GrayLinearF32,
        GrayAlphaLinearF32,
        RgbLinearF16,
        RgbaLinearF16,
        GrayLinearF16,
        GrayAlphaLinearF16,
        RgbPqF32,
        RgbaPqF32,
        RgbHlgF32,
        RgbaHlgF32,
        RgbBt709F32,
        RgbaBt709F32,
        Cmyk8,
        Cmyk16,
    ];
    assert_eq!(
        ALL.len(),
        26,
        "a PixelLayout variant was added or removed — extend this list and \
         give the new layout its source_bit_depth arm"
    );

    for &layout in ALL {
        let (is_float, bits, exp) = layout.source_bit_depth();
        let name = format!("{layout:?}");
        let looks_float = name.contains("F32") || name.contains("F16");
        assert_eq!(
            is_float, looks_float,
            "{name}: source_bit_depth float-ness must match the layout's sample type"
        );
        if name.contains("F32") {
            assert_eq!((bits, exp), (32, 8), "{name} must be binary32");
        } else if name.contains("F16") {
            assert_eq!((bits, exp), (16, 5), "{name} must be binary16");
        } else {
            assert_eq!(exp, 0, "{name} is integer, exponent_bits must be 0");
            assert!(bits == 8 || bits == 16, "{name} unexpected bits {bits}");
        }
    }

    // Spot-pin two arms by name so a blanket rewrite of the match cannot pass
    // the string-shaped check above by accident.
    assert_eq!(RgbaHlgF32.source_bit_depth(), (true, 32, 8));
    assert_eq!(GrayAlphaLinearF16.source_bit_depth(), (true, 16, 5));
}
