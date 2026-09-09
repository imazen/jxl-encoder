//! Probe (#95 review): is `encode_planar_int`'s memory admission sized for the
//! REAL input width, and does it run before the image is materialised?
//!
//! Read from source first, measured here:
//!  1. `encode_planar_int` calls `ModularImage::from_planar_int` -- which
//!     materialises every channel -- and only then builds the encoder and sets
//!     `input_admitted = true` by hand, skipping the push-time admission added
//!     by the 2026-09-08 streaming fix.
//!  2. `finish_inner`'s preflight is sized with `self.layout.bytes_per_pixel()`,
//!     and this path deliberately BORROWS a 16-bit layout (Rgb16 = 6 B/px)
//!     while the real input is u32 planes (12 B/px) plus an i32 ModularImage.
//!     `api.rs` already guards the tree-learning lift against exactly this
//!     borrowed-layout hazard; the memory estimate is not guarded.
//!  3. `LosslessConfig` exposes no `with_limits`, so a caller cannot tighten
//!     this path at all (lossless limits arrive via `EncodeRequest`).
//!
//! Run under `/usr/bin/time -l` and compare peak RSS with the printed estimate.
use jxl_encoder::api::{LosslessConfig, PixelLayout};

fn main() {
    let (w, h) = (2048u32, 2048u32);
    let n = (w * h) as usize;
    let plane: Vec<u32> = (0..n)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761)) >> 1)
        .collect();
    let planes: Vec<&[u32]> = vec![&plane, &plane, &plane];

    let real_input = n as u64 * 3 * 4; // u32 planes actually supplied
    let cfg = LosslessConfig::new().with_effort(5);

    // What the admission sizes against: the BORROWED 16-bit layout.
    let est_borrowed = cfg.estimate_peak_memory_bytes(w, h, PixelLayout::Rgb16);
    println!(
        "real input planes      : {:>10} B ({} MB)",
        real_input,
        real_input / 1_048_576
    );
    println!(
        "admission's input term : {:>10} B (Rgb16 = {} B/px)",
        n as u64 * PixelLayout::Rgb16.bytes_per_pixel() as u64,
        PixelLayout::Rgb16.bytes_per_pixel()
    );
    println!("estimate_peak (Rgb16)  : {:?}", est_borrowed);

    let t = std::time::Instant::now();
    let out = cfg
        .encode_planar_int(w, h, &planes, 31, false, false)
        .expect("planar encode");
    println!("encoded {} bytes in {:?}", out.len(), t.elapsed());
}
