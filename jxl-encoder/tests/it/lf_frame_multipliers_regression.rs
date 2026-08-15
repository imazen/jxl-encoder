//! Regression: the lossy-modular multipliers path
//! (`compute_best_tree_with_multipliers`, reached via
//! `LossyConfig::with_lf_frame(true)`) reads `samples.props[0/1]` for its
//! forced static splits AFTER `pre_quantize` — whose per-wave raw-column
//! free (51a0b473) emptied them, panicking out-of-bounds. The wave-free
//! now retains the caller-named static axes on that path. This encode
//! panicked before the fix; it must produce a valid multi-group stream.

use jxl_encoder::api::{LossyConfig, PixelLayout};

#[test]
fn lossy_lf_frame_multipliers_path_encodes() {
    let (w, h) = (512u32, 512u32);
    let mut px = vec![0u8; (w * h * 3) as usize];
    let mut state = 0x1234_5678u32;
    for (i, b) in px.iter_mut().enumerate() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let x = (i / 3) % w as usize;
        *b = ((x * 255 / w as usize) as u8).wrapping_add((state >> 28) as u8);
    }
    let bytes = LossyConfig::new(2.0)
        .with_effort(5)
        .with_threads(1)
        .with_lf_frame(true)
        .encode(&px, w, h, PixelLayout::Rgb8)
        .expect("lf_frame lossy encode must not fail");
    assert!(bytes.len() > 100, "implausibly small stream");
}
