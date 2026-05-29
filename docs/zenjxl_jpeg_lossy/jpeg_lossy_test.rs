//! Integration tests for the lossy JPEG → JXL recompression closed loop
//! (`zenjxl::jpeg_lossy`). Proves the in-process encode → decode → score loop:
//! zenjxl can coarsen a JPEG (jxl-encoder PreserveJxl), decode the result
//! (zenjxl-decoder), and drive a quality target with a caller-supplied scorer.
//!
//! Run: cargo test -p zenjxl --features jpeg-lossy --test jpeg_lossy
#![cfg(feature = "jpeg-lossy")]

use zenjxl::jpeg_lossy::{recompress_jpeg_coarsen, recompress_jpeg_lossy_relative};

// A tiny real-photo baseline JPEG (96x96, 3-component, ~3.8 KB).
const TINY_JPEG: &[u8] = include_bytes!("fixtures/tiny.jpg");

/// Mean squared error over tightly-packed RGB8 (lower = better quality).
fn mse(a: &[u8], b: &[u8], _w: u32, _h: u32) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f32::MAX;
    }
    let mut s = 0f64;
    for i in 0..n {
        let d = a[i] as f64 - b[i] as f64;
        s += d * d;
    }
    (s / n as f64) as f32
}

/// Decode a bare codestream to RGB8 + dims via the public decode API.
fn decode_dims(cs: &[u8]) -> (u32, u32) {
    let out = zenjxl::decode(cs, None, &[zenpixels::PixelDescriptor::RGB8])
        .expect("decode recompressed output");
    (out.info.width, out.info.height)
}

#[test]
fn coarsen_is_monotone_and_decodes() {
    let lossless = recompress_jpeg_coarsen(TINY_JPEG, 1.0, 5).expect("scale 1.0");
    let coarse = recompress_jpeg_coarsen(TINY_JPEG, 3.0, 5).expect("scale 3.0");
    // both decode to the source dimensions
    assert_eq!(decode_dims(&lossless), (96, 96));
    assert_eq!(decode_dims(&coarse), (96, 96));
    // coarsening shrinks the codestream
    assert!(
        coarse.len() < lossless.len(),
        "scale 3.0 ({}) must be smaller than lossless ({})",
        coarse.len(),
        lossless.len()
    );
}

#[test]
fn relative_loop_looser_target_is_smaller() {
    // MSE: lower is better, so higher_is_better = false.
    // Loose target (MSE <= 300) allows more coarsening than strict (MSE <= 30).
    let strict = recompress_jpeg_lossy_relative(TINY_JPEG, 30.0, false, &mse, 5)
        .expect("strict target");
    let loose = recompress_jpeg_lossy_relative(TINY_JPEG, 300.0, false, &mse, 5)
        .expect("loose target");
    assert_eq!(decode_dims(&strict), (96, 96));
    assert_eq!(decode_dims(&loose), (96, 96));
    assert!(
        loose.len() <= strict.len(),
        "looser target ({}) must be <= stricter target ({})",
        loose.len(),
        strict.len()
    );
}

#[test]
fn unreachable_target_returns_lossless_floor() {
    // An impossible target (MSE <= 0 = pixel-exact) can't be met by coarsening;
    // the loop must fall back to the lossless transcode (the floor), not error.
    let out = recompress_jpeg_lossy_relative(TINY_JPEG, 0.0, false, &mse, 5)
        .expect("unreachable target falls back to lossless");
    let lossless = recompress_jpeg_coarsen(TINY_JPEG, 1.0, 5).expect("lossless");
    assert_eq!(out.len(), lossless.len(), "unreachable -> lossless floor");
    assert_eq!(decode_dims(&out), (96, 96));
}
