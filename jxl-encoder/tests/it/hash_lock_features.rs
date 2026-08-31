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

use jxl_encoder::api::SectionedTrees;
use jxl_encoder::test_helpers::measure_file_header_len;
use jxl_encoder::{ColorEncoding, LosslessConfig, LossyConfig, Lz77Method, PixelLayout};

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

static EXPECTED: OnceLock<HashMap<String, Vec<ExpectedHash>>> = OnceLock::new();

fn load_expected() -> &'static HashMap<String, Vec<ExpectedHash>> {
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
                map.entry(name).or_insert_with(Vec::new).push(ExpectedHash {
                    size,
                    hdr_hash: hdr,
                    frm_hash: frm,
                });
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

// `measure_file_header_len` moved to `jxl_encoder::test_helpers` in #76
// (0.4.0 narrowing): this suite compiles under default features, and the
// header/bit-writer internals it re-serialized are now `pub(crate)`.

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

    // A hash lock alone blesses whatever bytes the encoder produced — the
    // zenjpeg locked-values suite green-lit undecodable output for months
    // that way (sweep issue #97). Decode EVERY cell in-process with
    // jxl-oxide, in both check and update modes, so neither a regression
    // nor a regen can ever lock undecodable bytes.
    {
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(data))
            .unwrap_or_else(|e| panic!("{name}: jxl-oxide header parse failed: {e:?}"));
        let header = image.image_header();
        assert_eq!(
            (header.size.width, header.size.height),
            (width, height),
            "{name}: decoded dimensions"
        );
        image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("{name}: jxl-oxide render failed: {e:?}"));
    }

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
    // Per-arch override: `name@<arch>` lines in the sidecar take
    // precedence over the bare name. Used ONLY where a platform's libm
    // produces a different (equally valid) encode — e.g. enhanced
    // clustering's f32 log2 hits a near-tie merge decision that flips
    // between glibc x86_64 and aarch64 (first seen on the 512x512
    // tiled-noise e8 cell: identical tree + identical LZ77 stream, 34 vs
    // 35 histograms). Override lines are added MANUALLY with the values
    // measured on that platform — regen (UPDATE_HASHES) always writes
    // bare names from the canonical x86_64 dev box. The lock stays
    // byte-exact on every platform; the divergence is documented, not
    // tolerated away.
    let arch_key = format!("{name}@{}", std::env::consts::ARCH);
    let mut variants: Vec<&ExpectedHash> = Vec::new();
    if let Some(v) = expected.get(&arch_key) {
        variants.extend(v.iter());
    }
    if let Some(v) = expected.get(name) {
        variants.extend(v.iter());
    }
    if variants.is_empty() {
        panic!(
            "{name}: no expected hashes in sidecar. Run:\n  \
             rm -f jxl_encoder/tests/hash_lock_expected.txt && \
             UPDATE_HASHES=1 cargo test --test hash_lock_features -- --test-threads=1"
        );
    }

    // Any-of exact match: the output must equal one RECORDED fingerprint.
    // Multiple lines with the same name are enumerated platform variants
    // (libm near-ties differ across glibc versions / ISAs — issue #70);
    // each is measured on its platform and added MANUALLY. Drift to an
    // unrecorded fingerprint still fails, and the panic prints the got-
    // fingerprint in sidecar-line format so a new platform's variant can
    // be lifted directly from its CI failure log.
    let matched = variants
        .iter()
        .any(|e| e.size == data.len() && e.hdr_hash == hdr && e.frm_hash == frm);
    if !matched {
        let known: Vec<String> = variants
            .iter()
            .map(|e| {
                format!(
                    "(size {}, hdr {:#018x}, frm {:#018x})",
                    e.size, e.hdr_hash, e.frm_hash
                )
            })
            .collect();
        panic!(
            "{name}: output matches NO recorded fingerprint.\n  got sidecar line: \
             {name} {} {hdr:#018x} {frm:#018x}\n  known variants: {}\n  (header_len={})",
            data.len(),
            known.join(", "),
            measure_file_header_len(width, height, xyb_encoded, has_alpha, is_gray, bit_depth_16),
        );
    }
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

/// 512x512 RGB noise (deterministic PRNG) — MULTI-GROUP (2x2 groups of
/// 256x256). Locks the multi-group lossless layout end-to-end: learned
/// tree + per-group sections + (since #69 item 1) the per-section LZ77
/// evaluation, which declines on noise (lz77.enabled=0 layout). The
/// single-group fixtures above never exercise any of that path — #68's
/// e9 desync and the #69 LZ77 drop both lived in exactly this blind
/// spot. Procedural like every other lock fixture: zero committed bytes.
fn noise_rgb_512x512() -> Vec<u8> {
    let (w, h) = (512, 512);
    let mut out = vec![0u8; w * h * 3];
    let mut seed = 1337u64;
    for val in &mut out {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *val = (seed >> 56) as u8;
    }
    out
}

/// 1024x1024 "photo-like" RGB: four octaves of deterministic value noise
/// (hashed integer lattice, bilinear between lattice points), six octaves
/// down to a 2-px lattice plus per-pixel grain, and a couple of hard edges. Unlike `noise_rgb_512x512` (which is i.i.d. per byte and
/// therefore structureless), this has multi-scale spatial correlation, so
/// the modular tree learner grows a DEEP tree over it — which is exactly
/// what the sectioned probe-tree selector needs in order to fire.
///
/// It exists because the two 512x512 sectioned lock cells do NOT trip the
/// selector (their probe trees fall under `SECTIONED_PROBE_MIN_LEAVES`), so
/// without this cell the content-adaptive path would ship with ZERO
/// byte-lock coverage — the same lock-coverage gap that let the
/// `resampling > 1` bugs survive (see CLAUDE.md, 2026-08-30).
fn photoish_rgb_1024x1024() -> Vec<u8> {
    let (w, h) = (1024usize, 1024usize);
    let hash = |x: i64, y: i64, c: i64| -> f32 {
        let mut v = (x.wrapping_mul(0x27d4_eb2d)
            ^ y.wrapping_mul(0x1656_67b1)
            ^ c.wrapping_mul(0x9e37_79b9)) as u64;
        v ^= v >> 33;
        v = v.wrapping_mul(0xff51_afd7_ed55_8ccd);
        v ^= v >> 29;
        ((v >> 40) as u32 as f32) / 16_777_216.0
    };
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3usize {
                let mut acc = 0.0f32;
                let mut amp = 0.5f32;
                for oct in 0..6u32 {
                    let step = 1usize << (6 - oct); // 64, 32, 16, 8, 4, 2
                    let (gx, gy) = (x / step, y / step);
                    let (fx, fy) = (
                        (x % step) as f32 / step as f32,
                        (y % step) as f32 / step as f32,
                    );
                    let c0 = c as i64 * 7 + oct as i64 * 131;
                    let v00 = hash(gx as i64, gy as i64, c0);
                    let v10 = hash(gx as i64 + 1, gy as i64, c0);
                    let v01 = hash(gx as i64, gy as i64 + 1, c0);
                    let v11 = hash(gx as i64 + 1, gy as i64 + 1, c0);
                    let top = v00 + (v10 - v00) * fx;
                    let bot = v01 + (v11 - v01) * fx;
                    acc += amp * (top + (bot - top) * fy);
                    amp *= 0.55;
                }
                // Two hard edges so the learner sees discontinuities as well
                // as texture.
                if x > 700 && y < 300 {
                    acc = acc * 0.35 + 0.6;
                }
                if (x as i64 - y as i64).abs() < 6 {
                    acc = 1.0 - acc;
                }
                // Per-pixel grain: without it the field is too smooth and the
                // probe tree stays shallow (measured: 191 leaves, under the
                // trust floor), so the cell would not cover the adaptive path.
                acc += (hash(x as i64, y as i64, c as i64 * 977 + 31) - 0.5) * 0.08;
                out[(y * w + x) * 3 + c] = (acc.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    out
}

/// 512x512 RGBA: 17-colour blocky RGB + a few-valued alpha pattern —
/// MULTI-GROUP with an extra channel (alpha defers to PassGroups) AND the
/// full-image palette engaging on the colour channels. Locks the
/// palette+alpha multi-group layout.
fn blocky17_rgba_512x512() -> Vec<u8> {
    let rgb = blocky17_rgb_512x512();
    let (w, h) = (512usize, 512usize);
    let mut out = vec![0u8; w * h * 4];
    for i in 0..w * h {
        out[i * 4..i * 4 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
        out[i * 4 + 3] = if (i / 8 + i / (w * 8)) % 3 == 0 {
            200
        } else {
            255
        };
    }
    out
}

/// 512x512 bilevel grayscale (text-like strokes) — MULTI-GROUP
/// single-channel: locks the ChannelCompact (nc=1 palette) multi-group
/// layout that document scans hit.
fn bilevel_gray_512x512() -> Vec<u8> {
    let (w, h) = (512usize, 512usize);
    let mut out = vec![255u8; w * h];
    for y in (0..h).step_by(7) {
        for x in 0..w {
            if (x / 5 + y / 7) % 3 != 0 {
                out[y * w + x] = 0;
            }
        }
    }
    out
}

/// 512x512 RGB of 8x8 blocks in 17 PRNG colours — MULTI-GROUP content
/// where the full-image palette transform (issue #69 item 2) engages:
/// locks the palette-meta-in-global + per-group index-channel layout
/// that no other fixture reaches.
fn blocky17_rgb_512x512() -> Vec<u8> {
    let (w, h) = (512usize, 512usize);
    let mut seed = 7777u64;
    let mut lcg = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 56) as u8
    };
    let palette: Vec<[u8; 3]> = (0..17).map(|_| [lcg(), lcg(), lcg()]).collect();
    let mut out = vec![0u8; w * h * 3];
    for by in 0..h / 8 {
        for bx in 0..w / 8 {
            let c = palette[(bx * 31 + by * 17 + (bx * by) % 7) % 17];
            for y in (by * 8)..(by * 8 + 8) {
                for x in (bx * 8)..(bx * 8 + 8) {
                    let i = (y * w + x) * 3;
                    out[i..i + 3].copy_from_slice(&c);
                }
            }
        }
    }
    out
}

/// 512x512 RGB built from a 64-px-wide PRNG noise tile repeated across
/// the image — MULTI-GROUP content where per-section LZ77 (e8 greedy)
/// clears the libjxl 20 % benefit floor. Locks the lz77.enabled=1
/// multi-group layout (num_contexts + 1 entropy code, per-section
/// dist_multiplier) that no other fixture reaches.
fn tiled_noise_rgb_512x512() -> Vec<u8> {
    let (w, h, tile) = (512usize, 512usize, 64usize);
    let mut tile_px = vec![0u8; tile * tile * 3];
    let mut seed = 4242u64;
    for val in &mut tile_px {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *val = (seed >> 56) as u8;
    }
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let src = ((y % tile) * tile + (x % tile)) * 3;
            let dst = (y * w + x) * 3;
            out[dst..dst + 3].copy_from_slice(&tile_px[src..src + 3]);
        }
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

/// Multi-group lossless at the e7 default: 2x2 groups, learned tree,
/// per-section LZ77 evaluated (declines on noise -> lz77.enabled=0).
#[test]
fn lossless_mg_rgb_512x512_noise_e7() {
    let data = LosslessConfig::new()
        // Pinned: this cell locks GLOBAL-tree bytes; since the 2026-08-19
        // Auto policy, an unpinned e7 encode under a parallel multi-thread
        // build (e.g. workspace feature unification) resolves to Sectioned.
        .with_sectioned_trees(SectionedTrees::Off)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_512x512_noise_e7",
        &data,
        512,
        512,
        false,
        false,
        false,
        false,
    );
}

/// Sectioned-trees twin of the e7 noise cell: locks the PER-GROUP-trees
/// engine's bytes (mode pinned On). Sectioned output is a pure function
/// of (pixels, options) — per-group learn/encode is group-local and the
/// merge is index-ordered — so the lock holds at any thread count.
#[test]
fn lossless_mg_rgb_512x512_noise_e7_sectioned() {
    let data = LosslessConfig::new()
        .with_sectioned_trees(SectionedTrees::On)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_512x512_noise_e7_sectioned",
        &data,
        512,
        512,
        false,
        false,
        false,
        false,
    );
}

/// The ONLY lock cell that exercises the content-adaptive sectioned
/// predictor selector (#99 item 1, 2026-08-31). 4x4 groups of multi-scale
/// texture: the whole-image probe tree clears
/// `SECTIONED_PROBE_MIN_LEAVES`, so the per-group learns run over the
/// probe's leaf-share-selected set instead of the fixed root-cost K.
///
/// If a future change makes the probe untrusted here (a bigger
/// `MIN_LEAVES`, a smaller `MAX_NODES`, a different probe density) this
/// cell would silently fall back to the fixed-K path and stop testing what
/// it was added for. The NON-VACUITY check lives in the env-overrides
/// binary (`env_overrides::sectioned_probe_selector::
/// photoish_1024_actually_trips_the_probe_selector`) because it needs
/// `JXL_SECTIONED_TREE_PREDICTORS=off` and `it` must not mutate env.
/// `JXL_PROBE_COSTS=1` prints the chosen set.
#[test]
fn lossless_mg_rgb_1024x1024_photoish_e7_sectioned() {
    let px = photoish_rgb_1024x1024();
    let data = LosslessConfig::new()
        .with_sectioned_trees(SectionedTrees::On)
        .encode(&px, 1024, 1024, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_1024x1024_photoish_e7_sectioned",
        &data,
        1024,
        1024,
        false,
        false,
        false,
        false,
    );
}

/// 8-bit multi-group lossless at e5: the GOAL_BEAT_CJXL wedge-1
/// budgeted tree-learn lift fires (8-bit + e5, scoreboard 2026-06-12)
/// — locks the lifted-path bytes AND its budget caps. Same fixture as
/// the e7 cell so the e5-lift vs e7-full-tree relationship stays
/// observable in the sidecar.
#[test]
fn lossless_mg_rgb_512x512_noise_e5() {
    let data = LosslessConfig::new()
        .with_effort(5)
        // Pinned global — see lossless_mg_rgb_512x512_noise_e7.
        .with_sectioned_trees(SectionedTrees::Off)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_512x512_noise_e5",
        &data,
        512,
        512,
        false,
        false,
        false,
        false,
    );
}

/// Multi-group lossless at e8 on tile-repeated noise: per-section LZ77
/// FIRES (greedy backrefs clear the 20 % floor) — locks the
/// lz77.enabled=1 multi-group layout (#69 item 1).
#[test]
fn lossless_mg_rgb_512x512_tiled_e8() {
    let data = LosslessConfig::new()
        .with_effort(8)
        .encode(&tiled_noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_512x512_tiled_e8",
        &data,
        512,
        512,
        false,
        false,
        false,
        false,
    );
}

/// 16-bit multi-group lossless at e5: the issue-#72 budgeted tree-learn
/// lift fires (Rgb16 + e5) — locks lift bytes AND the 16-bit
/// multi-group layout.
#[test]
fn lossless_mg_rgb16_512x512_ramp_e5() {
    let data = LosslessConfig::new()
        .with_effort(5)
        // Pinned global — see lossless_mg_rgb_512x512_noise_e7.
        .with_sectioned_trees(SectionedTrees::Off)
        .encode(&ramp_noise_rgb16_512x512(), 512, 512, PixelLayout::Rgb16)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb16_512x512_ramp_e5",
        &data,
        512,
        512,
        false,
        false,
        false,
        true,
    );
}

/// Multi-group lossless on 17-colour blocky content: the full-image
/// palette transform engages (issue #69 item 2) — locks the
/// palette + index multi-group layout.
#[test]
fn lossless_mg_rgb_512x512_palette_e7() {
    let data = LosslessConfig::new()
        // Pinned global — see lossless_mg_rgb_512x512_noise_e7. Since
        // 2026-08-28 the sectioned writer covers palette/ChannelCompact
        // content too, so an unpinned e7 encode under a parallel build
        // resolves Auto -> Sectioned here as well (it fell back before).
        .with_sectioned_trees(SectionedTrees::Off)
        .encode(&blocky17_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_512x512_palette_e7",
        &data,
        512,
        512,
        false,
        false,
        false,
        false,
    );
}

/// Multi-group LOSSY VarDCT at the e5 default path (2x2 DC groups of
/// AC groups; dispatch, CfL, per-group AC streams) — no lossy multi-group
/// cell existed before 2026-06-11 (all lossy fixtures were <=48x48).
#[test]
fn lossy_mg_rgb_512x512_noise_e5_d1() {
    let data = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_e5_d1",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

/// Multi-group LOSSY VarDCT at e7 (cfl_two_pass, patches default-on,
/// content-adaptive block context map — the production web path).
#[test]
fn lossy_mg_rgb_512x512_noise_e7_d1() {
    let data = LossyConfig::new(1.0)
        .with_effort(7)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_e7_d1",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

/// Sectioned-trees twin of the palette cell: locks the stream-0
/// meta-channel layout the sectioned writer emits since 2026-08-28 (the
/// palette meta channel coded in LfGlobal with its own learned tree,
/// index channel per-group local trees). Mode pinned On; thread-invariant
/// like the noise twin.
#[test]
fn lossless_mg_rgb_512x512_palette_e7_sectioned() {
    let data = LosslessConfig::new()
        .with_sectioned_trees(SectionedTrees::On)
        .encode(&blocky17_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_512x512_palette_e7_sectioned",
        &data,
        512,
        512,
        false,
        false,
        false,
        false,
    );
}

/// Multi-group lossless RGBA: alpha extra channel splits per-group while
/// the full-image palette engages on the RGB channels.
#[test]
fn lossless_mg_rgba_512x512_blocky_e7() {
    let data = LosslessConfig::new()
        // Pinned global — see lossless_mg_rgb_512x512_noise_e7. Since
        // 2026-08-28 the sectioned writer covers palette/ChannelCompact
        // content too, so an unpinned e7 encode under a parallel build
        // resolves Auto -> Sectioned here as well (it fell back before).
        .with_sectioned_trees(SectionedTrees::Off)
        .encode(&blocky17_rgba_512x512(), 512, 512, PixelLayout::Rgba8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgba_512x512_blocky_e7",
        &data,
        512,
        512,
        false,
        true,
        false,
        false,
    );
}

/// Multi-group lossless single-channel (document-scan class): bilevel
/// gray through ChannelCompact on the multi-group path.
#[test]
fn lossless_mg_gray_512x512_bilevel_e7() {
    let data = LosslessConfig::new()
        // Pinned global — see lossless_mg_rgb_512x512_noise_e7. Since
        // 2026-08-28 the sectioned writer covers palette/ChannelCompact
        // content too, so an unpinned e7 encode under a parallel build
        // resolves Auto -> Sectioned here as well (it fell back before).
        .with_sectioned_trees(SectionedTrees::Off)
        .encode(&bilevel_gray_512x512(), 512, 512, PixelLayout::Gray8)
        .unwrap();
    assert_hashes(
        "lossless_mg_gray_512x512_bilevel_e7",
        &data,
        512,
        512,
        false,
        false,
        true,
        false,
    );
}

/// Multi-group LOSSY 16-bit PQ at e7: locks the #74 HDR QuantizeWP
/// dispatch (resolved TF ∈ {Pq, Hlg} && effort ≤ 7 enables the
/// libjxl nl_dc DC shaping on the HDR path only — SDR cells are
/// structurally unchanged, which the rest of this suite pins).
#[test]
fn lossy_mg_rgb16_pq_512x512_ramp_e7_d1() {
    let data = LossyConfig::new(1.0)
        .with_effort(7)
        .encode_request(512, 512, PixelLayout::Rgb16)
        .with_color_encoding(ColorEncoding::bt2100_pq())
        .encode(&ramp_noise_rgb16_512x512())
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb16_pq_512x512_ramp_e7_d1",
        &data,
        512,
        512,
        true,
        false,
        false,
        true,
    );
}

/// Multi-group LOSSY gray at e7 (document-scan class): locks the
/// issue-#75 CfL zero-target guard — gray input has an identically-zero
/// XYB X plane; pre-fix the e7+ Newton fit fabricated ytox (±17) and
/// the encoder coded −ytox·Y as the X residual (~2× AC bytes). This
/// cell pins X-channel silence on the gray lossy path.
#[test]
fn lossy_mg_gray_512x512_bilevel_e7_d1() {
    let data = LossyConfig::new(1.0)
        .with_effort(7)
        .encode(&bilevel_gray_512x512(), 512, 512, PixelLayout::Gray8)
        .unwrap();
    assert_hashes(
        "lossy_mg_gray_512x512_bilevel_e7_d1",
        &data,
        512,
        512,
        true,
        false,
        true,
        false,
    );
}

/// 512x512 Rgb16 — MULTI-GROUP 16-bit: smooth vertical PQ-ish ramps
/// with per-pixel LCG noise in the low byte (the HDR-photo signal shape
/// that motivated issue #72). Locks the 16-bit e5 budgeted tree-learn
/// lift (and the 16-bit multi-group layout generally — no 16-bit
/// fixture existed before).
fn ramp_noise_rgb16_512x512() -> Vec<u8> {
    let (w, h) = (512usize, 512usize);
    let mut seed = 31337u64;
    let mut lcg = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 56) as u16
    };
    let mut out = Vec::with_capacity(w * h * 6);
    for y in 0..h {
        let base = ((y * 60000) / h) as u16;
        for x in 0..w {
            let r = base.saturating_add(lcg());
            let g = base
                .saturating_add((x / 2) as u16 & 0xff)
                .saturating_add(lcg());
            let b = base.saturating_add(lcg() / 2);
            for v in [r, g, b] {
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    out
}

/// Multi-group lossy at e10 with `resampling = 2`: pins the
/// `DownsampleImage2_Iterative` port (issue #45 ladder shift — the
/// iterative kernel is THE lossy e10 differentiator beyond
/// `fine_grained_step`; e ≤ 9 keeps the sharper kernel). The frame is
/// coded at 256² with `upsampling = 2` signaled; decoded dims stay 512².
#[test]
fn lossy_mg_rgb_512x512_noise_r2_e10() {
    let data = LossyConfig::new(1.0)
        .with_effort(10)
        .with_resampling(2)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_r2_e10",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

/// Multi-group lossy at e3/e7/e9 with `resampling = 2`: pins the
/// **e ≤ 9** sharper-kernel 2× downsample path (`DownsampleImage2_Sharper`
/// port). Added 2026-08-30 with the #45 T1 colour-domain fix — before
/// that commit NO lock covered any e ≤ 9 resampling cell, which is
/// exactly why the linear-RGB-vs-opsin domain mismatch survived (the
/// e10 cell above covers only the iterative kernel). Three efforts
/// because the surrounding encode differs per tier while the downsampler
/// does not: a future change that moves only one of them is a signal.
/// Frames are coded at 256² with `upsampling = 2` signaled; decoded dims
/// stay 512².
#[test]
fn lossy_mg_rgb_512x512_noise_r2_e3() {
    let data = LossyConfig::new(1.0)
        .with_effort(3)
        .with_resampling(2)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_r2_e3",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

#[test]
fn lossy_mg_rgb_512x512_noise_r2_e7() {
    let data = LossyConfig::new(1.0)
        .with_effort(7)
        .with_resampling(2)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_r2_e7",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

#[test]
fn lossy_mg_rgb_512x512_noise_r2_e9() {
    let data = LossyConfig::new(1.0)
        .with_effort(9)
        .with_resampling(2)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_r2_e9",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

/// Multi-group lossy with `resampling = 4` / `8`: pins the **box**
/// downsample path, which is effort-independent (libjxl
/// `enc_frame.cc:761-762` → `DownsampleImage(*opsin, upsampling)` for
/// every factor other than 2) and was likewise uncovered before
/// 2026-08-30. Frames are coded at 128² / 64² with `upsampling = 4` / `8`
/// signaled. e7 is the default tier; the box kernel is shared by every
/// other effort including e10, so one cell per factor pins it.
#[test]
fn lossy_mg_rgb_512x512_noise_r4_e7() {
    let data = LossyConfig::new(1.0)
        .with_effort(7)
        .with_resampling(4)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_r4_e7",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

#[test]
fn lossy_mg_rgb_512x512_noise_r8_e7() {
    let data = LossyConfig::new(1.0)
        .with_effort(7)
        .with_resampling(8)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_r8_e7",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

/// Multi-group lossy RGBA at `resampling = 2`: the colour planes go
/// through the opsin-domain sharper 2×, while **alpha** takes the plain
/// `box_downsample_alpha_u8` path — libjxl downsamples extra channels
/// with `DownsampleImage(ec, ec_resampling)` on the raw channel, no
/// colour transform (`enc_frame.cc:1650-1654`). This cell pins that
/// split, and pins that the frame header signals `ec_upsampling = 2`
/// alongside `upsampling = 2`: before 2026-08-30 it wrote `1` there and
/// the stream was undecodable, which is why the cell could not be locked
/// when the rest of its family was.
#[test]
fn lossy_mg_rgba_512x512_blocky_r2_e7() {
    let data = LossyConfig::new(1.0)
        .with_effort(7)
        .with_resampling(2)
        .encode(&blocky17_rgba_512x512(), 512, 512, PixelLayout::Rgba8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgba_512x512_blocky_r2_e7",
        &data,
        512,
        512,
        true,
        true,
        false,
        false,
    );
}

/// Multi-group lossy at e11: the extended-tier buttloop (8 iters) +
/// 2-seed multi-seed search — the pre-shift "e10" machinery at its
/// post-shift number (issue #45). First lock coverage for the extended
/// lossy tiers.
#[test]
fn lossy_mg_rgb_512x512_noise_e11_d1() {
    let data = LossyConfig::new(1.0)
        .with_effort(11)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossy_mg_rgb_512x512_noise_e11_d1",
        &data,
        512,
        512,
        true,
        false,
        false,
        false,
    );
}

/// Multi-group lossless at e11: the TectonicPlate per-image config
/// trial (issue #45 — probe pair, branch, ~20 unique trial encodes at
/// e10, winner re-encoded at the e11 profile). 17-colour blocky content
/// keeps every palette/channel-compact axis live; the 320×96 canvas
/// (2 horizontal modular groups) keeps the ~22-encode schedule cheap
/// enough for CI (a 512² variant measured 27 s locally). e12/e13 share
/// this machinery (bigger multi-seed finals only) and are exercised
/// with in-process decode by the effort-ladder bench.
#[test]
fn lossless_mg_rgb_320x96_blocky17_e11() {
    let (w, h) = (320usize, 96usize);
    let full = blocky17_rgb_512x512();
    // Top-left crop of the 512² blocky pattern (same 17-colour palette).
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        px[y * w * 3..(y + 1) * w * 3].copy_from_slice(&full[y * 512 * 3..y * 512 * 3 + w * 3]);
    }
    let data = LosslessConfig::new()
        .with_effort(11)
        .encode(&px, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_320x96_blocky17_e11",
        &data,
        w as u32,
        h as u32,
        false,
        false,
        false,
        false,
    );
}

/// Multi-group lossless at e9: ref-properties offered to the learner
/// (slots 0..2), the exact configuration where #68's two desyncs hid.
#[test]
fn lossless_mg_rgb_512x512_noise_e9() {
    let data = LosslessConfig::new()
        .with_effort(9)
        .encode(&noise_rgb_512x512(), 512, 512, PixelLayout::Rgb8)
        .unwrap();
    assert_hashes(
        "lossless_mg_rgb_512x512_noise_e9",
        &data,
        512,
        512,
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
