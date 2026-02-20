// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Comprehensive hash-locked tests for every encoder feature combination.
//!
//! These tests hash the **file header** and **frame+data** portions separately,
//! so file header refactoring (Phase 2+) can be validated independently of
//! frame-level changes.
//!
//! The split point is the first byte-aligned boundary after the file header
//! (signature + SizeHeader + ImageMetadata + CustomTransformData + zero_pad).
//!
//! If a hash changes, it means the bitstream changed. Update only after verifying
//! the new output decodes correctly with djxl, jxl-rs, and jxl-oxide.

use jxl_encoder::bit_writer::BitWriter;
use jxl_encoder::headers::color_encoding::{ColorEncoding, RenderingIntent};
use jxl_encoder::headers::extra_channels::ExtraChannelInfo;
use jxl_encoder::headers::file_header::{BitDepth, FileHeader, ImageMetadata};
use jxl_encoder::{LosslessConfig, LossyConfig, Lz77Method, PixelLayout};

// ── Feature-gated helpers ────────────────────────────────────────────────────

trait NoButterflyExt {
    fn no_butteraugli(self) -> Self;
}

impl NoButterflyExt for LossyConfig {
    fn no_butteraugli(self) -> Self {
        #[cfg(feature = "butteraugli-loop")]
        {
            self.with_butteraugli_iters(0)
        }
        #[cfg(not(feature = "butteraugli-loop"))]
        {
            self
        }
    }
}

// ── Hashing helpers ─────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash — deterministic across all platforms and Rust versions.
/// (DefaultHasher uses SipHash with version-dependent seeding.)
fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Measure the file header byte length by encoding the header alone.
///
/// We build the same FileHeader the encoder would, write it, and measure.
/// This works for both lossy and lossless because Phase 1 unified the header.
fn measure_file_header_len(
    width: u32,
    height: u32,
    xyb_encoded: bool,
    has_alpha: bool,
    is_gray: bool,
    bit_depth_16: bool,
) -> usize {
    let bit_depth = if bit_depth_16 {
        BitDepth::uint16()
    } else {
        BitDepth::uint8()
    };

    let mut color_encoding = if is_gray {
        ColorEncoding::gray()
    } else {
        ColorEncoding::srgb()
    };
    if xyb_encoded {
        color_encoding.rendering_intent = RenderingIntent::Relative;
    }

    let extra_channels = if has_alpha {
        vec![ExtraChannelInfo::alpha()]
    } else {
        Vec::new()
    };

    let file_header = FileHeader {
        width,
        height,
        metadata: ImageMetadata {
            bit_depth,
            color_encoding,
            extra_channels,
            xyb_encoded,
            ..ImageMetadata::default()
        },
    };

    let mut writer = BitWriter::new();
    file_header.write(&mut writer).unwrap();
    writer.zero_pad_to_byte();
    writer.finish_with_padding().len()
}

/// Split encoded JXL bytes into (file_header, frame_and_data).
fn split_header_frame(
    data: &[u8],
    width: u32,
    height: u32,
    xyb_encoded: bool,
    has_alpha: bool,
    is_gray: bool,
    bit_depth_16: bool,
) -> (&[u8], &[u8]) {
    let header_len =
        measure_file_header_len(width, height, xyb_encoded, has_alpha, is_gray, bit_depth_16);
    assert!(
        header_len <= data.len(),
        "header_len {} > data.len() {}",
        header_len,
        data.len()
    );
    data.split_at(header_len)
}

/// Hash both header and frame portions, returning (header_hash, frame_hash).
fn hash_split(
    data: &[u8],
    width: u32,
    height: u32,
    xyb_encoded: bool,
    has_alpha: bool,
    is_gray: bool,
    bit_depth_16: bool,
) -> (u64, u64) {
    let (header, frame) = split_header_frame(
        data,
        width,
        height,
        xyb_encoded,
        has_alpha,
        is_gray,
        bit_depth_16,
    );
    (hash_bytes(header), hash_bytes(frame))
}

#[allow(clippy::too_many_arguments)]
fn assert_hashes(
    name: &str,
    data: &[u8],
    width: u32,
    height: u32,
    xyb_encoded: bool,
    has_alpha: bool,
    is_gray: bool,
    bit_depth_16: bool,
    expected_header: u64,
    expected_frame: u64,
    expected_size: usize,
) {
    let (hdr, frm) = hash_split(
        data,
        width,
        height,
        xyb_encoded,
        has_alpha,
        is_gray,
        bit_depth_16,
    );
    assert_eq!(
        data.len(),
        expected_size,
        "{name}: SIZE mismatch: got {}, expected {expected_size}",
        data.len(),
    );
    assert_eq!(
        hdr,
        expected_header,
        "{name}: HEADER hash mismatch: got {hdr:#018x}, expected {expected_header:#018x} \
         (total_size={}, header_len={})",
        data.len(),
        measure_file_header_len(width, height, xyb_encoded, has_alpha, is_gray, bit_depth_16),
    );
    assert_eq!(
        frm,
        expected_frame,
        "{name}: FRAME hash mismatch: got {frm:#018x}, expected {expected_frame:#018x} \
         (total_size={})",
        data.len(),
    );
}

// ── Synthetic image generators ──────────────────────────────────────────────

/// 32x32 RGB gradient (R=x, G=y, B=0.5) as sRGB u8.
fn gradient_rgb_32x32() -> Vec<u8> {
    let (w, h) = (32, 32);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 128;
        }
    }
    out
}

/// 32x32 RGBA gradient with varying alpha.
fn gradient_rgba_32x32() -> Vec<u8> {
    let (w, h) = (32, 32);
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 128;
            out[i + 3] = ((x + y) * 255 / (w + h - 2)) as u8;
        }
    }
    out
}

/// 32x32 grayscale gradient.
fn gradient_gray_32x32() -> Vec<u8> {
    let (w, h) = (32, 32);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = ((x + y) * 255 / (w + h - 2)) as u8;
        }
    }
    out
}

/// 48x48 RGB noise (deterministic PRNG).
fn noise_rgb_48x48() -> Vec<u8> {
    let (w, h) = (48, 48);
    let mut out = vec![0u8; w * h * 3];
    let mut seed = 42u64;
    for val in &mut out {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *val = (seed >> 56) as u8;
    }
    out
}

/// 13x17 RGB noise (non-multiple-of-8 dimensions).
fn noise_rgb_13x17() -> Vec<u8> {
    let (w, h) = (13, 17);
    let mut out = vec![0u8; w * h * 3];
    let mut seed = 99u64;
    for val in &mut out {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *val = (seed >> 56) as u8;
    }
    out
}

/// 32x32 checkerboard (8x8 tiles, two gray levels).
fn checkerboard_rgb_32x32() -> Vec<u8> {
    let (w, h) = (32, 32);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let val = if ((x / 8) + (y / 8)) % 2 == 0 {
                200
            } else {
                55
            };
            out[i] = val;
            out[i + 1] = val;
            out[i + 2] = val;
        }
    }
    out
}

/// 32x32 RGB16 gradient (native endian).
fn gradient_rgb16_32x32() -> Vec<u8> {
    let (w, h) = (32, 32);
    let mut out = vec![0u8; w * h * 3 * 2];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 6;
            let r = (x * 65535 / (w - 1)) as u16;
            let g = (y * 65535 / (h - 1)) as u16;
            let b = 32768u16;
            out[i..i + 2].copy_from_slice(&r.to_ne_bytes());
            out[i + 2..i + 4].copy_from_slice(&g.to_ne_bytes());
            out[i + 4..i + 6].copy_from_slice(&b.to_ne_bytes());
        }
    }
    out
}

// ── Lossy (VarDCT) feature tests ────────────────────────────────────────────

#[test]
fn lossy_defaults_rgb_32x32() {
    let data = LossyConfig::new(1.0)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_defaults_rgb_32x32",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0xe02094e538f50b78, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        314,                // lossy_defaults_rgb_32x32
    );
}

#[test]
fn lossy_defaults_rgb_48x48_noise() {
    let data = LossyConfig::new(1.0)
        .encode(&noise_rgb_48x48(), 48, 48, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_defaults_rgb_48x48_noise",
        &data,
        48,
        48,
        true,
        false,
        false,
        false,
        0xb81c2f58aebac4b3,
        0x7feacc8b6b3ce9bc, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        3605,               // lossy_defaults_rgb_48x48_noise
    );
}

#[test]
fn lossy_defaults_rgb_13x17() {
    let data = LossyConfig::new(1.0)
        .encode(&noise_rgb_13x17(), 13, 17, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_defaults_rgb_13x17",
        &data,
        13,
        17,
        true,
        false,
        false,
        false,
        0x3333c10727f60b90,
        0xc8cb8d880fb19ee8, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        662,                // lossy_defaults_rgb_13x17
    );
}

#[test]
fn lossy_rgba_32x32() {
    let data = LossyConfig::new(1.0)
        .encode(&gradient_rgba_32x32(), 32, 32, PixelLayout::Rgba8)
        .unwrap();
    assert_hashes(
        "lossy_rgba_32x32",
        &data,
        32,
        32,
        true,
        true,
        false,
        false,
        0xe058fd017b3de453,
        0x1809962c08814051, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        751,                // lossy_rgba_32x32
    );
}

#[test]
fn lossy_rgb16_32x32() {
    let data = LossyConfig::new(1.0)
        .encode(&gradient_rgb16_32x32(), 32, 32, PixelLayout::Rgb16)
        .unwrap();
    assert_hashes(
        "lossy_rgb16_32x32",
        &data,
        32,
        32,
        true,
        false,
        false,
        true,
        0xe37a0d041fe39334,
        0xa64e35edf0214a59, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        318,                // lossy_rgb16_32x32
    );
}

#[test]
fn lossy_no_ans_huffman() {
    let data = LossyConfig::new(1.0)
        .with_ans(false)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_no_ans_huffman",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x27a15f51e75cd931, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        264,                // lossy_no_ans_huffman
    );
}

#[test]
fn lossy_no_gaborish() {
    let data = LossyConfig::new(1.0)
        .with_gaborish(false)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_no_gaborish",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x3c4ccd8db2a57fcc, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        327,                // lossy_no_gaborish
    );
}

#[test]
fn lossy_with_noise() {
    let data = LossyConfig::new(1.0)
        .with_noise(true)
        .encode(&noise_rgb_48x48(), 48, 48, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_with_noise",
        &data,
        48,
        48,
        true,
        false,
        false,
        false,
        0xb81c2f58aebac4b3,
        0x7feacc8b6b3ce9bc, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        3605,               // lossy_with_noise
    );
}

#[test]
fn lossy_no_error_diffusion() {
    let data = LossyConfig::new(1.0)
        .with_error_diffusion(false)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_no_error_diffusion",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x158690341794f656, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        316,                // lossy_no_error_diffusion
    );
}

#[test]
fn lossy_no_pixel_domain_loss() {
    let data = LossyConfig::new(1.0)
        .with_pixel_domain_loss(false)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_no_pixel_domain_loss",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0xe02094e538f50b78, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        314,                // lossy_no_pixel_domain_loss
    );
}

#[test]
fn lossy_no_butteraugli() {
    let data = LossyConfig::new(1.0)
        .no_butteraugli()
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_no_butteraugli",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0xe02094e538f50b78, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        314,                // lossy_no_butteraugli
    );
}

#[test]
fn lossy_force_dct8() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(0)) // DCT8
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_force_dct8",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x4e7311085222db5f, // Updated: CfL Newton convergence fallback to LS
        412,                // lossy_force_dct8
    );
}

#[test]
fn lossy_force_dct16x16() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(4)) // DCT16x16
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_force_dct16x16",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0xe02094e538f50b78, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        314,                // lossy_force_dct16x16
    );
}

#[test]
fn lossy_force_identity() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(8)) // IDENTITY (raw strategy code)
        .no_butteraugli()
        .encode(&checkerboard_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_force_identity",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x02ef2487cf4355ed, // Updated: CfL Newton convergence fallback to LS
        570,                // lossy_force_identity
    );
}

#[test]
fn lossy_force_dct2x2() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(9)) // DCT2x2 (raw strategy code)
        .no_butteraugli()
        .encode(&checkerboard_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_force_dct2x2",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x16fa005a7adbd25c, // Updated: CfL Newton convergence fallback to LS
        519,                // lossy_force_dct2x2
    );
}

#[test]
fn lossy_force_dct4x4() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(7)) // DCT4x4 (raw strategy code)
        .no_butteraugli()
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_force_dct4x4",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x95c976d16b9e1cb3, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        526,                // lossy_force_dct4x4
    );
}

#[test]
fn lossy_force_afv0() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(12)) // AFV0 (raw strategy code)
        .no_butteraugli()
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_force_afv0",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0xadd208f5f46de37f, // Updated: CfL Newton convergence fallback to LS
        537,                // lossy_force_afv0
    );
}

#[test]
fn lossy_with_lz77_greedy() {
    let data = LossyConfig::new(1.0)
        .with_lz77(true)
        .with_lz77_method(Lz77Method::Greedy)
        .encode(&noise_rgb_48x48(), 48, 48, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_with_lz77_greedy",
        &data,
        48,
        48,
        true,
        false,
        false,
        false,
        0xb81c2f58aebac4b3,
        0x7feacc8b6b3ce9bc, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        3605,               // lossy_with_lz77_greedy
    );
}

#[test]
fn lossy_with_lz77_rle() {
    let data = LossyConfig::new(1.0)
        .with_lz77(true)
        .with_lz77_method(Lz77Method::Rle)
        .encode(&noise_rgb_48x48(), 48, 48, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_with_lz77_rle",
        &data,
        48,
        48,
        true,
        false,
        false,
        false,
        0xb81c2f58aebac4b3,
        0x7feacc8b6b3ce9bc, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        3605,               // lossy_with_lz77_rle
    );
}

#[test]
fn lossy_distance_05() {
    let data = LossyConfig::new(0.5)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_distance_05",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x9ae13e498fa23554, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        366,                // lossy_distance_05 (gab OFF → larger)
    );
}

#[test]
fn lossy_distance_3() {
    let data = LossyConfig::new(3.0)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_distance_3",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x8e0f6107c74bc24f, // Updated: CfL Newton convergence fallback to LS
        269,                // lossy_distance_3
    );
}

#[test]
fn lossy_all_off() {
    // Minimal lossy: no gaborish, no noise, no error diffusion, no pixel domain,
    // no butteraugli, no LZ77, Huffman, forced DCT8
    let data = LossyConfig::new(1.0)
        .with_ans(false)
        .with_gaborish(false)
        .with_noise(false)
        .with_error_diffusion(false)
        .with_pixel_domain_loss(false)
        .no_butteraugli()
        .with_lz77(false)
        .with_force_strategy(Some(0))
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_all_off",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0x952a3afb53eba2b4, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        392,                // lossy_all_off
    );
}

#[test]
fn lossy_bgr8() {
    let data = LossyConfig::new(1.0)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Bgr8)
        .unwrap();
    assert_hashes(
        "lossy_bgr8",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
        0x3ba5403031c1499f,
        0xb9a9172abc3fbffb, // Updated: CfL LS-seeded Newton (eps=1) + skip Newton in pass 1
        311,                // lossy_bgr8
    );
}

// ── Lossless (Modular) feature tests ────────────────────────────────────────

#[test]
fn lossless_defaults_rgb_32x32() {
    let data = LosslessConfig::new()
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_defaults_rgb_32x32",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x6ecc5b958aadf8d3, // Updated: enhanced clustering + total_pixel scaling for tree learning
        132,                // lossless_defaults_rgb_32x32
    );
}

#[test]
fn lossless_defaults_rgb_48x48_noise() {
    let data = LosslessConfig::new()
        .encode(&noise_rgb_48x48(), 48, 48, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_defaults_rgb_48x48_noise",
        &data,
        48,
        48,
        false,
        false,
        false,
        false,
        0x4610698f0cf81821,
        0x16d5504f8c104725, // Updated: fixed rct_type U32 offset (BitsOffset(6,10) not 18)
        6978,               // lossless_defaults_rgb_48x48_noise
    );
}

#[test]
fn lossless_defaults_rgb_13x17() {
    let data = LosslessConfig::new()
        .encode(&noise_rgb_13x17(), 13, 17, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_defaults_rgb_13x17",
        &data,
        13,
        17,
        false,
        false,
        false,
        false,
        0x492393a4c3f29174,
        0xbed6cafa23cebd77, // Updated: enhanced clustering + total_pixel scaling for tree learning
        797,                // lossless_defaults_rgb_13x17
    );
}

#[test]
fn lossless_rgba_32x32() {
    let data = LosslessConfig::new()
        .encode(&gradient_rgba_32x32(), 32, 32, PixelLayout::Rgba8)
        .unwrap();
    assert_hashes(
        "lossless_rgba_32x32",
        &data,
        32,
        32,
        false,
        true,
        false,
        false,
        0x68b6bf8f2098c6eb,
        0xbae35472cbaf6dd8, // Updated: enhanced clustering + total_pixel scaling for tree learning
        148,                // lossless_rgba_32x32
    );
}

#[test]
fn lossless_gray_32x32() {
    let data = LosslessConfig::new()
        .encode(&gradient_gray_32x32(), 32, 32, PixelLayout::Gray8)
        .unwrap();
    assert_hashes(
        "lossless_gray_32x32",
        &data,
        32,
        32,
        false,
        false,
        true,
        false,
        0xd7858d308a0845e5,
        0xb3eb8879c469235d, // Updated: squeeze disabled by default
        451,                // lossless_gray_32x32
    );
}

#[test]
fn lossless_no_ans_huffman() {
    let data = LosslessConfig::new()
        .with_ans(false)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_no_ans_huffman",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x72f288c2ad0c943d, // Updated: squeeze disabled by default
        621,                // lossless_no_ans_huffman
    );
}

#[test]
fn lossless_with_tree_learning() {
    // Now redundant with defaults (tree learning is default-on at effort 7),
    // but kept for backwards compatibility testing.
    let data = LosslessConfig::new()
        .with_tree_learning(true)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_with_tree_learning",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x6ecc5b958aadf8d3, // Updated: enhanced clustering + total_pixel scaling for tree learning
        132,                // lossless_with_tree_learning
    );
}

#[test]
fn lossless_with_squeeze() {
    let data = LosslessConfig::new()
        .with_squeeze(true)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_with_squeeze",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x48e4346586752ae2, // Updated: enhanced clustering + total_pixel scaling for tree learning
        324,                // lossless_with_squeeze
    );
}

#[test]
fn lossless_with_lz77_greedy() {
    // Note: tree learning (default-on at effort 7) takes priority over LZ77,
    // so this produces the same output as defaults. LZ77 is silently ignored.
    let data = LosslessConfig::new()
        .with_lz77(true)
        .with_lz77_method(Lz77Method::Greedy)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_with_lz77_greedy",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x6ecc5b958aadf8d3, // Updated: enhanced clustering + total_pixel scaling for tree learning
        132,                // lossless_with_lz77_greedy
    );
}

#[test]
fn lossless_with_lz77_rle() {
    // Note: tree learning (default-on at effort 7) takes priority over LZ77,
    // so this produces the same output as defaults. LZ77 is silently ignored.
    let data = LosslessConfig::new()
        .with_lz77(true)
        .with_lz77_method(Lz77Method::Rle)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_with_lz77_rle",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x6ecc5b958aadf8d3, // Updated: enhanced clustering + total_pixel scaling for tree learning
        132,                // lossless_with_lz77_rle
    );
}

#[test]
fn lossless_tree_learning_and_squeeze() {
    let data = LosslessConfig::new()
        .with_tree_learning(true)
        .with_squeeze(true)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_tree_learning_and_squeeze",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x48e4346586752ae2, // Updated: enhanced clustering + total_pixel scaling for tree learning
        324,                // lossless_tree_learning_and_squeeze
    );
}

#[test]
fn lossless_all_off() {
    // Minimal lossless: Huffman, no tree learning, no squeeze, no LZ77
    let data = LosslessConfig::new()
        .with_ans(false)
        .with_tree_learning(false)
        .with_squeeze(false)
        .with_lz77(false)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_all_off",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x72f288c2ad0c943d, // Updated: RCT (YCoCg) now applied to non-squeeze paths
        621,                // lossless_all_off
    );
}

#[test]
fn lossless_bgr8() {
    let data = LosslessConfig::new()
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Bgr8)
        .unwrap();
    assert_hashes(
        "lossless_bgr8",
        &data,
        32,
        32,
        false,
        false,
        false,
        false,
        0x6a695d8f2209b4e5,
        0x18c57fb704f2a20a, // Updated: enhanced clustering + total_pixel scaling for tree learning
        132,                // lossless_bgr8
    );
}
