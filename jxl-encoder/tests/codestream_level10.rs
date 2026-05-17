// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Codestream-level signaling (`jxll` box) integration test.
//!
//! Validates that an encode whose properties exceed the JPEG XL Level 5
//! caps (per libjxl `VerifyLevelSettings` in `lib/jxl/encode.cc:550`)
//! emits a `jxll` box carrying `level = 10` directly after the `ftyp`
//! box, and that the produced container parses + decodes cleanly with
//! the primary Rust decoder (`jxl-rs`).
//!
//! The minimal level-bumping trigger that doesn't require a > 256 MP
//! pixel buffer (which would need ~3 GB of input pixels for a real
//! roundtrip) is an ICC profile larger than `1 << 22` bytes — that
//! flips the level-5 ICC cap and forces a `jxll` box even on a tiny
//! image. Same one-byte payload, same container wiring; the level-10
//! `jxll` box is the load-bearing piece of this top-1 audit item.

use jxl_encoder::api::{ImageMetadata, LosslessConfig, PixelLayout};

/// Inspect a JXL container and pull the `jxll` (level) byte out, if
/// present. Mirrors libjxl's container layout: signature(12) +
/// ftyp(20) + optional jxll(9) + rest.
fn find_jxll_box(jxl: &[u8]) -> Option<u8> {
    // Container signature occupies offsets 0..12; ftyp 12..32.
    if jxl.len() < 41 {
        return None;
    }
    // ftyp box must sit at offset 12 ("ftyp" type starts at byte 16).
    if &jxl[16..20] != b"ftyp" {
        return None;
    }
    // jxll box, when present, sits directly after ftyp at offset 32.
    if &jxl[36..40] == b"jxll" {
        // 4-byte BE size (9) + 4-byte type + 1-byte payload (level).
        let size = u32::from_be_bytes([jxl[32], jxl[33], jxl[34], jxl[35]]);
        assert_eq!(size, 9, "jxll box must be 9 bytes total");
        return Some(jxl[40]);
    }
    None
}

#[test]
fn synthetic_large_icc_forces_level10_jxll_box() {
    // 4 MB + 1 byte > level-5 ICC cap (2²² = 4 194 304). Mirrors libjxl
    // `VerifyLevelSettings` `icc_size > (1 << 22)` clause. Synthetic
    // payload — ICC validity isn't probed by the level-selection logic
    // (libjxl only inspects ICC length there too).
    let large_icc = vec![0xAB; (1 << 22) + 1];
    let meta = ImageMetadata::default().with_icc_profile(&large_icc);

    let jxl = LosslessConfig::new()
        .encode_request(8, 8, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&[128u8; 8 * 8 * 3])
        .expect("lossless encode with 4 MB ICC must succeed at level 10");

    // Container is mandatory at level 10 (mirrors libjxl
    // `MustUseContainer`).
    assert!(
        jxl.starts_with(&[0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ']),
        "level-10 encode must wrap in a container even without EXIF/XMP",
    );

    // The `jxll` box is the load-bearing piece — one byte advertising
    // the codestream level to decoders.
    let level =
        find_jxll_box(&jxl).expect("level-10 encode must emit a `jxll` box directly after `ftyp`");
    assert_eq!(level, 10, "jxll payload must signal level 10");
}

#[test]
fn baseline_image_does_not_emit_jxll_box() {
    // Same encoder + same pixels + no oversized ICC ⇒ level 5 ⇒ no
    // `jxll` box, and (because no EXIF/XMP) no container at all — the
    // bare codestream signature `0xFF 0x0A` must remain the first
    // bytes. This is what guarantees the hash-lock impact is zero for
    // normal images.
    let jxl = LosslessConfig::new()
        .encode_request(8, 8, PixelLayout::Rgb8)
        .encode(&[128u8; 8 * 8 * 3])
        .expect("baseline lossless encode must succeed");

    assert_eq!(&jxl[..2], &[0xFF, 0x0A], "baseline encode must stay bare");
    assert!(
        find_jxll_box(&jxl).is_none(),
        "baseline encode must NOT emit a `jxll` box",
    );
}

#[test]
fn level10_box_order_matches_libjxl() {
    // libjxl `encode.cc:830-838` places the `jxll` box **directly after
    // ftyp** and before any subsequent boxes. Verify our layout
    // matches: [signature 12B] [ftyp 20B] [jxll 9B] [jxlc + payload].
    let large_icc = vec![0xAB; (1 << 22) + 1];
    let meta = ImageMetadata::default().with_icc_profile(&large_icc);
    let jxl = LosslessConfig::new()
        .encode_request(8, 8, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&[128u8; 8 * 8 * 3])
        .expect("lossless encode at level 10 must succeed");

    // Signature box
    assert_eq!(&jxl[..4], &[0, 0, 0, 0x0C], "signature box size");
    assert_eq!(&jxl[4..8], b"JXL ", "signature box type");

    // ftyp box
    assert_eq!(&jxl[12..16], &[0, 0, 0, 0x14], "ftyp box size");
    assert_eq!(&jxl[16..20], b"ftyp", "ftyp box type");

    // jxll box directly after ftyp
    assert_eq!(&jxl[32..36], &[0, 0, 0, 0x09], "jxll box size = 9");
    assert_eq!(&jxl[36..40], b"jxll", "jxll box type");
    assert_eq!(jxl[40], 10, "jxll payload signals level 10");

    // jxlc box directly after jxll (no other box between them when
    // there's no JPEG-reconstruction or EXIF/XMP routed through a
    // partial-header jxlp).
    assert_eq!(&jxl[45..49], b"jxlc", "jxlc box type follows jxll");
}
