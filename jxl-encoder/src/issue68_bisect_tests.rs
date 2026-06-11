// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #68 bisect harness: e9+ lossless emits bitstreams no decoder
//! accepts on photographic/noisy content (`SectionTooShort` /
//! entropy-stream EOF), while e7/e8 round-trip exactly.
//!
//! Strategy: enumerate the e8→e9 [`EffortProfile`] field deltas, then
//! flip each one individually onto the working e8 profile (and back off
//! the broken e9 profile) and decode-verify with jxl-oxide in-process.
//! The field whose flip transfers the failure is the culprit.
//!
//! Once the bug is fixed, `e9_lossless_decodes_on_noise` stays as the
//! regression gate.

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
    let mut img = jxl_oxide::JxlImage::builder()
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

/// Bisect round 2 over the remaining e8→e9 deltas: threshold (the only
/// LosslessInternalParams-coverable field my zenjxl-side bisect missed —
/// "e9+thresh89" there was a no-op alias of e9's own default) and
/// lz77_method via the public setter (with a bytes-differ plumb check so
/// a silently-unconsumed setter can't fake a rule-out again).
#[test]
fn bisect_remaining_deltas_512() {
    let (w, h) = (512u32, 512u32);
    let px = noise_rgb8(w, h, 42);

    let enc = |effort: u8,
               thresh: Option<f32>,
               method: Option<crate::entropy_coding::lz77::Lz77Method>| {
        let mut cfg = LosslessConfig::new().with_effort(effort).with_threads(1);
        if let Some(t) = thresh {
            let mut p = crate::effort::LosslessInternalParams::default();
            p.tree_threshold_base = Some(t);
            cfg = cfg.with_internal_params(p);
        }
        if let Some(m) = method {
            cfg = cfg.with_lz77_method(m);
        }
        cfg.encode(&px, w, h, PixelLayout::Rgb8).expect("encode")
    };

    let e9_def = enc(9, None, None);
    println!(
        "e9 default          : {} bytes, decode={:?}",
        e9_def.len(),
        decode_ok(&e9_def).err()
    );
    let e9_t103 = enc(9, Some(103.0), None);
    println!(
        "e9 + thresh103 (e8) : {} bytes (differs={}), decode={:?}",
        e9_t103.len(),
        e9_t103 != e9_def,
        decode_ok(&e9_t103).err()
    );
    let e8_def = enc(8, None, None);
    let e8_t89 = enc(8, Some(89.0), None);
    println!(
        "e8 + thresh89 (e9)  : {} bytes (differs-from-e8={}), decode={:?}",
        e8_t89.len(),
        e8_t89 != e8_def,
        decode_ok(&e8_t89).err()
    );
    let e9_greedy = enc(
        9,
        None,
        Some(crate::entropy_coding::lz77::Lz77Method::Greedy),
    );
    println!(
        "e9 + lz77 Greedy    : {} bytes (differs={}), decode={:?}",
        e9_greedy.len(),
        e9_greedy != e9_def,
        decode_ok(&e9_greedy).err()
    );
    let e8_optimal = enc(
        8,
        None,
        Some(crate::entropy_coding::lz77::Lz77Method::Optimal),
    );
    println!(
        "e8 + lz77 Optimal   : {} bytes (differs-from-e8={}), decode={:?}",
        e8_optimal.len(),
        e8_optimal != e8_def,
        decode_ok(&e8_optimal).err()
    );
}
