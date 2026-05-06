// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Repro for an effort=9 encoder hang on synthetic high-frequency images.
//!
//! `LossyConfig::new(0.5).with_effort(9)` on a 1024x1024 grayscale 8-pixel
//! checker pattern hangs indefinitely (single thread, ~100% CPU, no progress
//! after 60+ seconds). The same image at effort=7 finishes in ~170ms, and
//! libjxl C reference (cjxl 0.10.3) at effort=9 finishes in ~2 seconds.
//!
//! This test is `#[ignore]`d because it currently hangs. Remove the ignore
//! once the bug is fixed; the test enforces a 5-second wallclock budget for
//! the encode (generous: libjxl C does it in ~2s, e7 does it in <1s).
//!
//! Tracking issue: https://github.com/imazen/jxl-encoder/issues/27

use jxl_encoder::{LossyConfig, PixelLayout};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Generate a 1024x1024 grayscale 8-pixel checker pattern as RGB8 (R=G=B).
///
/// Tile size = 8 px; cells alternate 0/255 in row+col parity. This is the
/// same shape as `synthetic/checker_4_1024x1024.png` from the zentrain
/// corpus; we generate it inline so the test has no corpus dependency.
fn make_checker_rgb8(width: usize, height: usize, tile: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let v = if ((x / tile) + (y / tile)) % 2 == 0 {
                0u8
            } else {
                255u8
            };
            out.push(v);
            out.push(v);
            out.push(v);
        }
    }
    out
}

/// Run `f` on a worker thread; return its output if it finishes within
/// `timeout`, otherwise return `None`. The worker thread is leaked on
/// timeout (we cannot safely kill a Rust thread mid-encode), but the
/// test process exits shortly after, so the leak is bounded.
fn run_with_timeout<F, T>(timeout: Duration, f: F) -> Option<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(timeout).ok()
}

#[test]
#[ignore = "currently hangs indefinitely; see issue tracker"]
fn e9_checker_pattern_does_not_hang() {
    const W: usize = 1024;
    const H: usize = 1024;
    const TILE: usize = 8;
    const BUDGET: Duration = Duration::from_secs(5);

    let pixels = make_checker_rgb8(W, H, TILE);
    let input_len = pixels.len();

    let result = run_with_timeout(BUDGET, move || {
        LossyConfig::new(0.5)
            .with_effort(9)
            .encode(&pixels, W as u32, H as u32, PixelLayout::Rgb8)
    });

    let encoded = result.expect("encode did not complete within 5s budget (hang)");
    let encoded = encoded.expect("encode returned an error");

    assert_eq!(
        &encoded[..2],
        &[0xFF, 0x0A],
        "missing JXL signature in output"
    );
    assert!(
        encoded.len() < input_len,
        "output suspiciously large: {} bytes for {}x{} input",
        encoded.len(),
        W,
        H
    );
}
