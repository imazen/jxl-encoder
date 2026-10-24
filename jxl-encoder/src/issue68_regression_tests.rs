// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #68 regression gate (backported to 0.3.x): e9+ multi-group
//! lossless used to emit bitstreams no decoder accepts. Two independent
//! causes: the ref-property stride cut mid 4-slot group, and tree
//! property 1 (`group_id`) using ad-hoc ids instead of the spec stream id.
//! Both tests decode with jxl-rs (exact samples) and djxl (`DJXL_PATH`,
//! libjxl v0.12).

use crate::api::{LosslessConfig, PixelLayout};

/// glibc-LCG noise, `(state >> 16) & 0xFF`: compressible-but-noisy content
/// that reproduces the stride cause at 512^2.
fn noise_rgb8(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    let mut state = seed;
    for _ in 0..(w * h * 3) {
        state = (1103515245u64.wrapping_mul(state).wrapping_add(12345)) % (1 << 31);
        out.push(((state >> 16) & 0xFF) as u8);
    }
    out
}

fn assert_decodes_exactly(jxl: &[u8], px: &[u8], w: u32, h: u32, what: &str) {
    // Reference decoder (set DJXL_PATH to a libjxl v0.12 djxl). The
    // crates.io jxl-oxide 0.12 used by this branch's tests mis-reads some
    // multi-group modular frames that jxl-rs and djxl both accept.
    let r = crate::test_helpers::decode_with_djxl(jxl)
        .unwrap_or_else(|e| panic!("{what}: djxl: {e:?}"));
    assert_eq!((r.width, r.height), (w as usize, h as usize), "{what}: djxl dims");
    let d = crate::test_helpers::decode_with_jxl_rs(jxl)
        .unwrap_or_else(|e| panic!("{what}: jxl-rs: {e:?}"));
    assert_eq!((d.width, d.height), (w as usize, h as usize), "{what}");
    for (i, &want) in px.iter().enumerate() {
        let got = (d.pixels[i / 3 * d.channels + i % 3] * 255.0).round();
        assert_eq!(got, want as f32, "{what}: jxl-rs sample {i}");
    }
}

fn encode(effort: u8, px: &[u8], w: u32, h: u32) -> Vec<u8> {
    LosslessConfig::new()
        .with_effort(effort)
        .with_threads(1)
        .encode(px, w, h, PixelLayout::Rgb8)
        .expect("encode")
}

#[test]
fn e9_lossless_decodes_on_noise() {
    let (w, h) = (512u32, 512u32);
    let px = noise_rgb8(w, h, 42);
    assert_decodes_exactly(&encode(8, &px, w, h), &px, w, h, "e8 noise");
    assert_decodes_exactly(&encode(9, &px, w, h), &px, w, h, "e9 noise (#68)");
}

/// Each 256^2 group gets a distinct character so the e9 learner splits on
/// group_id; under the old ids this desynced every decoder.
#[test]
fn e9_lossless_decodes_on_group_distinct_content() {
    let (w, h) = (512u32, 512u32);
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    let mut state: u64 = 7;
    for y in 0..h {
        for x in 0..w {
            let rgb: [u8; 3] = match (x / 256, y / 256) {
                (0, 0) => {
                    let v = ((x + y) / 4) as u8;
                    [v, v.wrapping_add(13), v.wrapping_sub(9)]
                }
                (1, 0) => {
                    let mut n = [0u8; 3];
                    for c in &mut n {
                        state = (1103515245u64.wrapping_mul(state).wrapping_add(12345)) % (1 << 31);
                        *c = ((state >> 16) & 0xFF) as u8;
                    }
                    n
                }
                (0, 1) => {
                    let v = if (x / 8) % 2 == 0 { 40 } else { 210 };
                    [v, v, (x % 251) as u8]
                }
                _ => {
                    let v = if ((x / 4) + (y / 4)) % 2 == 0 { 25 } else { 230 };
                    [v, (y % 247) as u8, v]
                }
            };
            px.extend_from_slice(&rgb);
        }
    }
    assert_decodes_exactly(&encode(9, &px, w, h), &px, w, h, "e9 group-distinct (#68)");
    assert_decodes_exactly(&encode(8, &px, w, h), &px, w, h, "e8 group-distinct");
}
