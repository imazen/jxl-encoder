//! `JXL_TREE_PRUNE_PREDICTORS` override coverage (#96 per-group predictor
//! pruning). The sectioned writer prunes to K=8 by default; the env forces
//! any K (>= 14 disables). Env-mutating — lives in the env-overrides
//! binary; every test takes `env_serial()`.

use crate::env_serial;
use jxl_encoder::api::SectionedTrees;
use jxl_encoder::{LosslessConfig, PixelLayout};

/// Multi-group procedural fixture (512x512 RGB — >256px so the sectioned
/// per-group path actually engages, per the multi-group rule). Gradients +
/// pseudo-noise so per-group trees are non-trivial.
fn prng_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    let mut state: u32 = 0x9e37_79b9;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = (state >> 24) as u8;
            out.push(((x * 255 / w) as u8).wrapping_add(n & 0x1f));
            out.push(((y * 255 / h) as u8) ^ (n & 0x0f));
            out.push((((x + y) * 128 / (w + h)) as u8).wrapping_add(n >> 3));
        }
    }
    out
}

fn encode_sectioned(pixels: &[u8], w: u32, h: u32) -> Vec<u8> {
    LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_sectioned_trees(SectionedTrees::On)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("sectioned lossless encode")
}

/// Decode and return tightly-packed RGB8 (the decoder emits RGBA).
fn decode_pixels(bytes: &[u8], w: usize, h: usize) -> Vec<u8> {
    let decoded = zenjxl_decoder::decode(bytes).expect("zenjxl-decoder decode");
    assert_eq!((decoded.width, decoded.height), (w, h), "decoded dims");
    assert_eq!(decoded.channels, 4, "expected RGBA output");
    decoded
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect()
}

#[test]
fn prune_k2_still_roundtrips_pixel_exact() {
    let _guard = env_serial();
    let (w, h) = (512usize, 512usize);
    let pixels = prng_rgb(w, h);
    // Hardest prune the knob allows: Weighted + 1 static predictor.
    // SAFETY: env_serial() guarantees exclusive env access in this binary.
    unsafe { std::env::set_var("JXL_TREE_PRUNE_PREDICTORS", "2") };
    let bytes = encode_sectioned(&pixels, w as u32, h as u32);
    // SAFETY: env_serial() guarantees exclusive env access in this binary.
    unsafe { std::env::remove_var("JXL_TREE_PRUNE_PREDICTORS") };
    assert_eq!(
        decode_pixels(&bytes, w, h),
        pixels,
        "K=2 sectioned roundtrip must be pixel-exact"
    );
}

#[test]
fn prune_env_matches_default_and_full_set_roundtrips() {
    let _guard = env_serial();
    let (w, h) = (512usize, 512usize);
    let pixels = prng_rgb(w, h);
    // SAFETY: env_serial() guarantees exclusive env access in this binary.
    unsafe { std::env::remove_var("JXL_TREE_PRUNE_PREDICTORS") };
    let default_bytes = encode_sectioned(&pixels, w as u32, h as u32);
    // SAFETY: env_serial() guarantees exclusive env access in this binary.
    unsafe { std::env::set_var("JXL_TREE_PRUNE_PREDICTORS", "8") };
    let explicit_k8 = encode_sectioned(&pixels, w as u32, h as u32);
    // SAFETY: env_serial() guarantees exclusive env access in this binary.
    unsafe { std::env::set_var("JXL_TREE_PRUNE_PREDICTORS", "14") };
    let full_set = encode_sectioned(&pixels, w as u32, h as u32);
    // SAFETY: env_serial() guarantees exclusive env access in this binary.
    unsafe { std::env::remove_var("JXL_TREE_PRUNE_PREDICTORS") };
    // The unset default IS K=8: explicit 8 must be byte-identical.
    assert_eq!(default_bytes, explicit_k8, "env K=8 == unset default");
    // K>=14 disables pruning; the full-set stream must roundtrip too.
    assert_eq!(
        decode_pixels(&full_set, w, h),
        pixels,
        "K=14 (disabled) sectioned roundtrip must be pixel-exact"
    );
}
