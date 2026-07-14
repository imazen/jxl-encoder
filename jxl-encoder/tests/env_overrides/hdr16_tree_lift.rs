// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #72: the 16-bit e5/e6 budgeted tree-learning lift.
//!
//! Lives in the `env_overrides` binary because the opt-out contract is
//! the runtime `JXL_NO_16BIT_TREE_LIFT` hook (mutated below via
//! `env_serial()`); the dispatch-sensitive readers in `it` must never
//! share a process with these mutations.

use jxl_encoder::{LosslessConfig, PixelLayout};

fn ramp_noise_rgb16(w: usize, h: usize) -> Vec<u8> {
    let mut seed = 777u64;
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

fn encode_e5(pixels: &[u8], w: u32, h: u32) -> Vec<u8> {
    LosslessConfig::new()
        .with_effort(5)
        .with_threads(1)
        .encode(pixels, w, h, PixelLayout::Rgb16)
        .expect("encode")
}

/// The lift fires on 16-bit multi-group at e5 (bytes differ from the
/// env-disabled arm and are SMALLER), the env hook disables it, an
/// explicit `with_tree_learning(false)` ALSO disables it (user-set
/// suppression), and the lifted bitstream roundtrips via jxl-oxide.
#[test]
fn lift_fires_and_is_suppressable() {
    let _env_serial = crate::env_serial();
    let (w, h) = (512usize, 512usize);
    let pixels = ramp_noise_rgb16(w, h);

    // SAFETY: this test holds env_serial(); no other thread touches the environment.
    unsafe { std::env::remove_var("JXL_NO_16BIT_TREE_LIFT") };
    let lifted = encode_e5(&pixels, w as u32, h as u32);

    // SAFETY: this test holds env_serial(); no other thread touches the environment.
    unsafe { std::env::set_var("JXL_NO_16BIT_TREE_LIFT", "1") };
    let disabled = encode_e5(&pixels, w as u32, h as u32);
    // SAFETY: this test holds env_serial(); no other thread touches the environment.
    unsafe { std::env::remove_var("JXL_NO_16BIT_TREE_LIFT") };

    assert!(
        lifted.len() < disabled.len(),
        "lift must fire and shrink 16-bit e5 output (lifted {} vs disabled {})",
        lifted.len(),
        disabled.len(),
    );

    // Explicit user choice suppresses the lift even with the env unset.
    let user_off = LosslessConfig::new()
        .with_effort(5)
        .with_threads(1)
        .with_tree_learning(false)
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb16)
        .expect("encode user-off");
    assert_eq!(
        user_off, disabled,
        "with_tree_learning(false) must match the env-disabled arm"
    );

    // 8-bit input is untouched by the gate: byte-identical both arms.
    let pixels8: Vec<u8> = pixels
        .chunks_exact(2)
        .map(|c| (u16::from_ne_bytes([c[0], c[1]]) >> 8) as u8)
        .collect();
    // SAFETY: this test holds env_serial(); no other thread touches the environment.
    unsafe { std::env::set_var("JXL_NO_16BIT_TREE_LIFT", "1") };
    let eight_a = LosslessConfig::new()
        .with_effort(5)
        .with_threads(1)
        .encode(&pixels8, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();
    // SAFETY: this test holds env_serial(); no other thread touches the environment.
    unsafe { std::env::remove_var("JXL_NO_16BIT_TREE_LIFT") };
    let eight_b = LosslessConfig::new()
        .with_effort(5)
        .with_threads(1)
        .encode(&pixels8, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();
    assert_eq!(
        eight_a, eight_b,
        "8-bit e5 must be unaffected by the lift/env"
    );

    // Roundtrip the lifted bitstream (secondary decoder in-process; the
    // jxl-rs + djxl legs run via the bench-set CLI verification).
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&lifted[..]))
        .expect("jxl-oxide parse");
    image.render_frame(0).expect("jxl-oxide render");
}
