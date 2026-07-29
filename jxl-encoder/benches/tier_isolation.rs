// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at
// https://www.imazen.io/pricing
//
//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! `jxl-encoder-simd` carries the XYB transform, DCT8/DCT64, Gaborish 5x5, EPF,
//! chroma-from-luma and pixel-loss kernels, every one of them
//! `#[magetypes(... neon ...)]`. The benches in this crate are microbenches of
//! specific tree-learning strategies; none of them compares the encoder against
//! itself with SIMD disabled, so a kernel slower than its own scalar fallback
//! is invisible. (The same gap in linear-srgb was hiding a real regression.)
//!
//! Both lossy (VarDCT) and lossless (modular) are measured: elsewhere in this
//! sweep the two modes behaved very differently — zenjxl-decoder gets 1.7-1.8x
//! on VarDCT and ~1.0x on modular, because modular is predictor/entropy serial.
//!
//! Run: `cargo bench -p jxl-encoder --bench tier_isolation --features _dev`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};
use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

/// Noise + patches. A gradient would give degenerate DCT coefficients and
/// understate exactly the transform kernels this is measuring.
fn make_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut state = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = ((state >> 24) as u8).wrapping_add(patch.wrapping_mul(40));
            rgb[i + 1] = ((state >> 16) as u8).wrapping_add(patch.wrapping_mul(80));
            rgb[i + 2] = ((state >> 8) as u8).wrapping_add(patch.wrapping_mul(120));
        }
    }
    rgb
}

fn bench_tiers(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, build with --features _dev). \
             Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    let (w, h) = (512usize, 512usize);
    let rgb = make_rgb(w, h);

    // Lossy / VarDCT — the DCT, XYB and Gaborish kernels.
    let rgb_l: &'static [u8] = Box::leak(rgb.clone().into_boxed_slice());
    suite.compare("encode_lossy_d1_512", |g| {
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || set_simd(simd)).run(move |_| {
                    LossyConfig::new(1.0)
                        .with_effort(5)
                        .encode(rgb_l, w as u32, h as u32, PixelLayout::Rgb8)
                        .unwrap()
                })
            });
        }
    });

    // Lossless / modular — predictor + entropy, expected to gain much less.
    suite.compare("encode_lossless_512", |g| {
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || set_simd(simd)).run(move |_| {
                    LosslessConfig::new()
                        .with_effort(5)
                        .encode(rgb_l, w as u32, h as u32, PixelLayout::Rgb8)
                        .unwrap()
                })
            });
        }
    });
    set_simd(true);
}

zenbench::main!(bench_tiers);
