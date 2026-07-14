// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #68 regression gate: e9+ lossless used to emit bitstreams no
//! decoder accepts (`SectionTooShort` / entropy-stream EOF in
//! zenjxl-decoder, jxl-oxide AND djxl) on content whose learned tree
//! split on a mid-group reference-channel property.
//!
//! Root cause: `collect_residuals_with_tree*` sized the per-pixel
//! property record at `max_tree_prop + 1`, and the ref-property writer
//! fills 4-slot groups behind a `base + 3 < prop_stride` guard — so a
//! tree whose maximum property id landed mid-group (id % 4 != 3, only
//! reachable at e9+ where the learner may split on slots 0..2) had its
//! ENTIRE top ref-group skipped: the encode-side walk read zeros where
//! every decoder computes real values → context desync → EOF. The fix
//! rounds the stride up to whole ref-groups.
//!
//! e≤8 trees only split on slot 3 (group always complete), which is
//! why this was e9+-only, and why the fix is byte-neutral for e≤8.

use crate::api::{EncoderMode, LosslessConfig, PixelLayout};
use crate::effort::EffortProfile;

/// Deterministic RGB noise — byte-for-byte the zenjpeg-bench-utils
/// `generate_noise` generator (glibc LCG, `(state >> 16) & 0xFF`),
/// because zenjxl's sweep_validate proved `generate_noise(512,512,42)`
/// reproduces #68. NOTE: `(state >> 16) & 0xFF` of a glibc LCG is NOT
/// uniform white noise — the low-ish bits carry LCG structure, which is
/// exactly the kind of compressible-but-noisy content that trips the
/// bug (pure uniform noise encodes near-raw and does NOT reproduce).
fn noise_rgb8(w: u32, h: u32, seed: u64) -> Vec<u8> {
    const A: u64 = 1103515245;
    const C: u64 = 12345;
    const M: u64 = 1 << 31;
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    let mut state = seed;
    for _ in 0..(w * h * 3) {
        state = (A.wrapping_mul(state).wrapping_add(C)) % M;
        out.push(((state >> 16) & 0xFF) as u8);
    }
    out
}

fn decode_ok(jxl: &[u8]) -> Result<(), String> {
    let cursor = std::io::Cursor::new(jxl);
    let img = jxl_oxide::JxlImage::builder()
        .read(cursor)
        .map_err(|e| format!("read: {e:?}"))?;
    img.render_frame(0).map_err(|e| format!("render: {e:?}"))?;
    Ok(())
}

fn encode_lossless_e(effort: u8, px: &[u8], w: u32, h: u32) -> Vec<u8> {
    LosslessConfig::new()
        .with_effort(effort)
        .with_threads(1)
        .encode(px, w, h, PixelLayout::Rgb8)
        .expect("encode")
}

/// Print the complete e8-vs-e9 profile diff so the bisect set is
/// enumerated from reality, not from memory. Run with `--nocapture`.
#[test]
fn dump_e8_vs_e9_profile_diff() {
    let p8 = EffortProfile::lossless(8, EncoderMode::Reference);
    let p9 = EffortProfile::lossless(9, EncoderMode::Reference);
    let d8 = format!("{p8:#?}");
    let d9 = format!("{p9:#?}");
    println!("--- e8 vs e9 lossless profile field diff ---");
    for (l8, l9) in d8.lines().zip(d9.lines()) {
        if l8 != l9 {
            println!("e8: {l8}\ne9: {l9}");
        }
    }
    println!("--- end diff ---");
    assert_ne!(d8, d9, "e8 and e9 lossless profiles must differ");
}

/// Baseline sanity: e8 must decode (it does today); e9 is the #68
/// repro. THIS IS THE REGRESSION GATE — it asserts the fixed behavior
/// and fails until #68 is fixed. 512² is the smallest repro size
/// (256² noise decodes fine at e9 — the failure is size-gated).
#[test]
fn e9_lossless_decodes_on_noise() {
    let (w, h) = (512u32, 512u32);
    let px = noise_rgb8(w, h, 42);
    let e8 = encode_lossless_e(8, &px, w, h);
    decode_ok(&e8).expect("e8 lossless must decode");
    let e9 = encode_lossless_e(9, &px, w, h);
    decode_ok(&e9).unwrap_or_else(|e| panic!("e9 lossless must decode (#68): {e}"));
}

/// #68 second cause: property-1 (`group_id`) values must be the spec
/// stream ids the decoder evaluates. Only e9+ trees may split on
/// group_id, and only multi-group images make it non-constant — this
/// 512² content gives each 256² group a distinct character (smooth
/// gradient / LCG noise / bars / checker) so group_id is strongly
/// predictive and the learner reliably splits on it. Under the old
/// ad-hoc ids (`meta_offset + group_idx` vs the decoder's
/// `1 + 3*num_lf_groups + 17 + g`) this desynced and failed to decode.
#[test]
fn e9_lossless_decodes_on_group_distinct_content() {
    let (w, h) = (512u32, 512u32);
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    let mut state: u64 = 7;
    for y in 0..h {
        for x in 0..w {
            let (gx, gy) = (x / 256, y / 256);
            let rgb: [u8; 3] = match (gx, gy) {
                (0, 0) => {
                    // Smooth diagonal gradient.
                    let v = ((x + y) / 4) as u8;
                    [v, v.wrapping_add(13), v.wrapping_sub(9)]
                }
                (1, 0) => {
                    // glibc-LCG noise (compressible-but-noisy).
                    let mut n = [0u8; 3];
                    for c in &mut n {
                        state = (1103515245u64.wrapping_mul(state).wrapping_add(12345)) % (1 << 31);
                        *c = ((state >> 16) & 0xFF) as u8;
                    }
                    n
                }
                (0, 1) => {
                    // Vertical bars.
                    let v = if (x / 8) % 2 == 0 { 40 } else { 210 };
                    [v, v, (x % 251) as u8]
                }
                _ => {
                    // 4px checker.
                    let v = if ((x / 4) + (y / 4)) % 2 == 0 {
                        25
                    } else {
                        230
                    };
                    [v, (y % 247) as u8, v]
                }
            };
            px.extend_from_slice(&rgb);
        }
    }
    let e9 = encode_lossless_e(9, &px, w, h);
    decode_ok(&e9)
        .unwrap_or_else(|e| panic!("e9 multi-group lossless must decode (#68 cause 2): {e}"));
    let e8 = encode_lossless_e(8, &px, w, h);
    decode_ok(&e8).expect("e8 must decode");
}
