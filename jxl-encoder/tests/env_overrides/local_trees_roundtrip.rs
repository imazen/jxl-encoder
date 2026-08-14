//! Sectioned-tree lossless mode (imazen/jxl-encoder#96,
//! `JXL_LOSSLESS_LOCAL_TREES=1`): every PassGroup stream carries its own MA
//! tree (`use_global_tree = false`) learned from that group's samples only.
//! This test proves, in one process:
//!
//! 1. The mode ENGAGES on a multi-group image (bitstream differs from the
//!    global-tree encode).
//! 2. Both bitstreams decode PIXEL-EXACT via zenjxl-decoder.
//! 3. Clearing the env restores byte-identical global-tree output (the
//!    default path is untouched).
//!
//! Lives in the env-overrides binary because it mutates process env
//! (`env_serial` serializes with every other test here).

use jxl_encoder::{LosslessConfig, PixelLayout};

fn prng_rgb(w: usize, h: usize) -> Vec<u8> {
    // Small-period procedural content with real structure (gradients +
    // pseudo-noise) so trees are non-trivial in every group.
    let mut out = Vec::with_capacity(w * h * 3);
    let mut state: u32 = 0x1234_5678;
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

fn encode(pixels: &[u8], w: u32, h: u32) -> Vec<u8> {
    LosslessConfig::new()
        .with_effort(7)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("lossless encode")
}

/// Decode and return tightly-packed RGB8 (the decoder emits RGBA).
fn decode_pixels(bytes: &[u8], w: usize, h: usize) -> Vec<u8> {
    let decoded = zenjxl_decoder::decode(bytes).expect("zenjxl-decoder decode");
    assert_eq!((decoded.width, decoded.height), (w, h), "decoded dims");
    assert_eq!(decoded.channels, 4, "expected RGBA output");
    decoded
        .data
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect()
}

#[test]
fn local_trees_mode_roundtrips_and_default_is_untouched() {
    let _guard = crate::env_serial();
    // Multi-group: 512x512 = 2x2 groups at the default 256 group size.
    let (w, h) = (512usize, 512usize);
    let pixels = prng_rgb(w, h);

    // SAFETY: env mutation is serialized by `env_serial` (this binary's
    // process-wide mutex); no other thread reads env concurrently.
    unsafe { std::env::remove_var("JXL_LOSSLESS_LOCAL_TREES") };
    let global_a = encode(&pixels, w as u32, h as u32);

    // SAFETY: as above — serialized by `env_serial`.
    unsafe { std::env::set_var("JXL_LOSSLESS_LOCAL_TREES", "1") };
    let local = encode(&pixels, w as u32, h as u32);
    // SAFETY: as above — serialized by `env_serial`.
    unsafe { std::env::remove_var("JXL_LOSSLESS_LOCAL_TREES") };
    let global_b = encode(&pixels, w as u32, h as u32);

    assert_eq!(
        global_a, global_b,
        "default path must be byte-identical before/after toggling the mode"
    );
    assert_ne!(
        global_a, local,
        "the mode must actually engage (bitstreams should differ)"
    );

    let dec_global = decode_pixels(&global_a, w, h);
    let dec_local = decode_pixels(&local, w, h);
    assert_eq!(dec_global, pixels, "global-tree roundtrip must be exact");
    assert_eq!(dec_local, pixels, "local-tree roundtrip must be exact");
}
