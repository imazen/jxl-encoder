// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Verifies the `colr` (alternative colour descriptor, ISO/IEC 14496-12) and
//! `hCdR` (HDR content description) container boxes round-trip through the
//! one-shot [`jxl_encoder::api::EncodeRequest`] path. These are pass-through
//! ISOBMFF boxes — JPEG XL spec clause 5 requires decoders to ignore boxes
//! with unrecognised types, so emitting them never alters decoded pixels.
//!
//! No decoder verification: these boxes are container-only and live alongside
//! the codestream, not inside it. The tests here scan the output bytes for
//! the FourCC + payload, mirroring the `container::tests::*` invariant tests
//! but exercising the full `ImageMetadata` → `wrap_metadata_container` →
//! append-aux-box wiring.

use jxl_encoder::api::{ImageMetadata, LosslessConfig, PixelLayout};
use jxl_encoder::container::colr_nclx_payload;

/// Build a 16×16 solid-red RGB8 image. Small enough that encode + scan is
/// fast; big enough to exercise the lossless single-group path end-to-end.
fn red_16x16_rgb8() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(16 * 16 * 3);
    for _ in 0..(16 * 16) {
        pixels.extend_from_slice(&[200, 32, 32]);
    }
    pixels
}

/// Find the first occurrence of `needle` in `haystack`, returning its offset.
fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[test]
fn colr_box_appears_in_one_shot_lossless_output() {
    let pixels = red_16x16_rgb8();
    let colr = colr_nclx_payload(9, 16, 9, true); // BT.2100 PQ, full range

    let cfg = LosslessConfig::new();
    let meta = ImageMetadata::new().with_colr_payload(&colr);
    let request = cfg
        .encode_request(16, 16, PixelLayout::Rgb8)
        .with_metadata(&meta);
    let bytes = request.encode(&pixels).expect("encode must succeed");

    // The 4-byte FourCC "colr" must appear in the output.
    let colr_pos = find_subseq(&bytes, b"colr").expect("colr FourCC must be in the container");
    // The 8-byte box header (size + type) sits ahead of our payload. The
    // payload's first 4 bytes are `nclx` since we used the helper.
    assert_eq!(&bytes[colr_pos + 4..colr_pos + 8], b"nclx");
    // CICP fields immediately follow (3 × u16 BE).
    assert_eq!(&bytes[colr_pos + 8..colr_pos + 10], &[0x00, 0x09]); // primaries 9
    assert_eq!(&bytes[colr_pos + 10..colr_pos + 12], &[0x00, 0x10]); // transfer 16
    assert_eq!(&bytes[colr_pos + 12..colr_pos + 14], &[0x00, 0x09]); // matrix 9
    assert_eq!(bytes[colr_pos + 14], 0x80); // full-range MSB set
}

#[test]
fn hcdr_box_appears_in_one_shot_lossless_output() {
    let pixels = red_16x16_rgb8();
    // Synthetic 28-byte HDR metadata blob (mastering display volume +
    // MaxCLL/MaxFALL shape — the encoder doesn't interpret).
    let hcdr: [u8; 28] = [
        0x80, 0x00, 0x33, 0x33, 0x1F, 0x40, 0xC3, 0x50, 0x21, 0x34, 0x08, 0x6A, 0x3D, 0x13, 0x40,
        0x42, 0x00, 0x98, 0x96, 0x80, 0x00, 0x00, 0x00, 0x32, 0x03, 0xE8, 0x01, 0x90,
    ];

    let cfg = LosslessConfig::new();
    let meta = ImageMetadata::new().with_hcdr_payload(&hcdr);
    let request = cfg
        .encode_request(16, 16, PixelLayout::Rgb8)
        .with_metadata(&meta);
    let bytes = request.encode(&pixels).expect("encode must succeed");

    let hcdr_pos = find_subseq(&bytes, b"hCdR").expect("hCdR FourCC must be in the container");
    // Payload of `hcdr` length immediately follows the 8-byte box header.
    assert_eq!(&bytes[hcdr_pos + 4..hcdr_pos + 4 + hcdr.len()], &hcdr);
}

#[test]
fn both_boxes_present_when_attached_together() {
    let pixels = red_16x16_rgb8();
    let colr = colr_nclx_payload(1, 13, 1, false); // sRGB studio-swing
    let hcdr = b"\xDE\xAD\xBE\xEF\x12\x34\x56\x78";

    let cfg = LosslessConfig::new();
    let meta = ImageMetadata::new()
        .with_colr_payload(&colr)
        .with_hcdr_payload(hcdr);
    let request = cfg
        .encode_request(16, 16, PixelLayout::Rgb8)
        .with_metadata(&meta);
    let bytes = request.encode(&pixels).expect("encode must succeed");

    let colr_pos = find_subseq(&bytes, b"colr").expect("colr FourCC must be present");
    let hcdr_pos = find_subseq(&bytes, b"hCdR").expect("hCdR FourCC must be present");
    // Both boxes are appended after wrap; colr is written first, hCdR
    // second (matches the append order in `EncodeRequest::encode_inner`).
    assert!(
        colr_pos < hcdr_pos,
        "colr must precede hCdR in the byte stream"
    );
}

#[test]
fn no_boxes_emitted_when_neither_field_is_set() {
    // Pin the hash-lock invariant: opt-in fields default to None and
    // produce byte-identical output to the no-metadata path.
    let pixels = red_16x16_rgb8();

    let cfg = LosslessConfig::new();
    let baseline = cfg
        .encode_request(16, 16, PixelLayout::Rgb8)
        .encode(&pixels)
        .expect("baseline encode must succeed");

    // Attaching empty ImageMetadata must not change a single byte.
    let meta = ImageMetadata::new();
    let with_empty_meta = cfg
        .encode_request(16, 16, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&pixels)
        .expect("encode with empty metadata must succeed");

    assert_eq!(
        baseline, with_empty_meta,
        "ImageMetadata::new() without colr/hcdr must produce byte-identical output"
    );
    // And specifically: no `colr` / `hCdR` FourCC anywhere.
    assert!(
        find_subseq(&baseline, b"colr").is_none(),
        "baseline must not contain colr"
    );
    assert!(
        find_subseq(&baseline, b"hCdR").is_none(),
        "baseline must not contain hCdR"
    );
}
