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
//! Expected hash values are stored in `hash_lock_expected.txt` (sidecar file).
//! To regenerate after an intentional encoding change:
//!
//!   rm -f jxl_encoder/tests/hash_lock_expected.txt
//!   UPDATE_HASHES=1 cargo test --test hash_lock_features -- --test-threads=1
//!
//! Then verify the new output decodes correctly with djxl, jxl-rs, and jxl-oxide.

use jxl_encoder::bit_writer::BitWriter;
use jxl_encoder::headers::color_encoding::{ColorEncoding, RenderingIntent};
use jxl_encoder::headers::extra_channels::ExtraChannelInfo;
use jxl_encoder::headers::file_header::{BitDepth, FileHeader, ImageMetadata};
use jxl_encoder::{LosslessConfig, LossyConfig, Lz77Method, PixelLayout};

use std::collections::HashMap;
use std::sync::OnceLock;

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

// ── Sidecar file helpers ──────────────────────────────────────────────────────

const SIDECAR_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/hash_lock_expected.txt");

struct ExpectedHash {
    size: usize,
    hdr_hash: u64,
    frm_hash: u64,
}

static EXPECTED: OnceLock<HashMap<String, ExpectedHash>> = OnceLock::new();

fn load_expected() -> &'static HashMap<String, ExpectedHash> {
    EXPECTED.get_or_init(|| {
        let content = match std::fs::read_to_string(SIDECAR_PATH) {
            Ok(c) => c,
            Err(_) => return HashMap::new(),
        };
        let mut map = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 4 {
                let name = parts[0].to_string();
                let size: usize = parts[1].parse().unwrap();
                let hdr: u64 = u64::from_str_radix(parts[2].trim_start_matches("0x"), 16).unwrap();
                let frm: u64 = u64::from_str_radix(parts[3].trim_start_matches("0x"), 16).unwrap();
                map.insert(
                    name,
                    ExpectedHash {
                        size,
                        hdr_hash: hdr,
                        frm_hash: frm,
                    },
                );
            }
        }
        map
    })
}

fn is_update_mode() -> bool {
    std::env::var("UPDATE_HASHES").is_ok()
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
    let header_len =
        measure_file_header_len(width, height, xyb_encoded, has_alpha, is_gray, bit_depth_16);
    assert!(
        header_len <= data.len(),
        "header_len {} > data.len() {}",
        header_len,
        data.len()
    );
    let (header, frame) = data.split_at(header_len);
    (hash_bytes(header), hash_bytes(frame))
}

/// Assert or update hash expectations.
///
/// In normal mode: reads expected values from sidecar, asserts match.
/// With `UPDATE_HASHES=1`: appends computed values to sidecar, skips assertions.
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

    if is_update_mode() {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(SIDECAR_PATH)
            .unwrap();
        writeln!(f, "{name} {} {hdr:#018x} {frm:#018x}", data.len()).unwrap();
        return;
    }

    let expected = load_expected();
    let exp = expected.get(name).unwrap_or_else(|| {
        panic!(
            "{name}: no expected hashes in sidecar. Run:\n  \
             rm -f jxl_encoder/tests/hash_lock_expected.txt && \
             UPDATE_HASHES=1 cargo test --test hash_lock_features -- --test-threads=1"
        );
    });

    assert_eq!(
        data.len(),
        exp.size,
        "{name}: SIZE mismatch: got {}, expected {}",
        data.len(),
        exp.size,
    );
    assert_eq!(
        hdr,
        exp.hdr_hash,
        "{name}: HEADER hash mismatch: got {hdr:#018x}, expected {:#018x} \
         (total_size={}, header_len={})",
        exp.hdr_hash,
        data.len(),
        measure_file_header_len(width, height, xyb_encoded, has_alpha, is_gray, bit_depth_16),
    );
    assert_eq!(
        frm,
        exp.frm_hash,
        "{name}: FRAME hash mismatch: got {frm:#018x}, expected {:#018x} (total_size={})",
        exp.frm_hash,
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
    );
}

#[test]
fn lossy_rgba_32x32() {
    let data = LossyConfig::new(1.0)
        .encode(&gradient_rgba_32x32(), 32, 32, PixelLayout::Rgba8)
        .unwrap();
    assert_hashes("lossy_rgba_32x32", &data, 32, 32, true, true, false, false);
}

#[test]
fn lossy_rgb16_32x32() {
    let data = LossyConfig::new(1.0)
        .encode(&gradient_rgb16_32x32(), 32, 32, PixelLayout::Rgb16)
        .unwrap();
    assert_hashes("lossy_rgb16_32x32", &data, 32, 32, true, false, false, true);
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
    );
}

#[test]
fn lossy_with_noise() {
    let data = LossyConfig::new(1.0)
        .with_noise(true)
        .encode(&noise_rgb_48x48(), 48, 48, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes("lossy_with_noise", &data, 48, 48, true, false, false, false);
}

#[test]
fn lossy_with_error_diffusion() {
    // Error diffusion is off by default (libjxl accepts param but never uses it).
    // This test verifies the opt-in ED path produces different (not necessarily better) output.
    let data = LossyConfig::new(1.0)
        .with_error_diffusion(true)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_with_error_diffusion",
        &data,
        32,
        32,
        true,
        false,
        false,
        false,
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
    );
}

#[test]
fn lossy_force_dct8() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(0))
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes("lossy_force_dct8", &data, 32, 32, true, false, false, false);
}

#[test]
fn lossy_force_dct16x16() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(4))
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
    );
}

#[test]
fn lossy_force_identity() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(8))
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
    );
}

#[test]
fn lossy_force_dct2x2() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(9))
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
    );
}

#[test]
fn lossy_force_dct4x4() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(7))
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
    );
}

#[test]
fn lossy_force_afv0() {
    let data = LossyConfig::new(1.0)
        .with_force_strategy(Some(12))
        .no_butteraugli()
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes("lossy_force_afv0", &data, 32, 32, true, false, false, false);
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
    );
}

#[test]
fn lossy_distance_3() {
    let data = LossyConfig::new(3.0)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes("lossy_distance_3", &data, 32, 32, true, false, false, false);
}

#[test]
fn lossy_all_off() {
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
    assert_hashes("lossy_all_off", &data, 32, 32, true, false, false, false);
}

#[test]
fn lossy_bgr8() {
    let data = LossyConfig::new(1.0)
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Bgr8)
        .unwrap();
    assert_hashes("lossy_bgr8", &data, 32, 32, true, false, false, false);
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
    );
}

#[test]
fn lossless_with_tree_learning() {
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
    );
}

#[test]
fn lossless_with_lz77_greedy() {
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
    );
}

#[test]
fn lossless_with_lz77_rle() {
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
    );
}

#[test]
fn lossless_all_off() {
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
    );
}

#[test]
fn lossless_bgr8() {
    let data = LosslessConfig::new()
        .encode(&gradient_rgb_32x32(), 32, 32, PixelLayout::Bgr8)
        .unwrap();
    assert_hashes("lossless_bgr8", &data, 32, 32, false, false, false, false);
}
