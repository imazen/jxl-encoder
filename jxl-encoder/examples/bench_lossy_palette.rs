//! Microbenchmark for `apply_lossy_palette`.
//!
//! Encodes synthetic palette-friendly 256x256 RGB images (single-group, where
//! lossy palette engages) repeatedly and prints per-iteration wall time.
//! Used to verify refactor of `quant_rows` / `quant_out` from `Vec<Vec<...>>`
//! to a single flat allocation does not regress.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release --example bench_lossy_palette

use jxl_encoder::api::{LosslessConfig, PixelLayout};
use std::time::Instant;

fn make_palette_image(seed: u32, w: u32, h: u32, num_colors: usize) -> Vec<u8> {
    // Deterministic palette-like image. Spatial blocks of pseudo-random
    // colors with low-amplitude noise to exercise the delta path.
    let palette: Vec<[u8; 3]> = (0..num_colors)
        .map(|i| {
            let s = seed
                .wrapping_mul(2654435761)
                .wrapping_add(i as u32 * 0x9E3779B1);
            [s as u8, (s >> 8) as u8, (s >> 16) as u8]
        })
        .collect();
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            // 8x8 spatial blocks of one palette colour.
            let bx = x / 8;
            let by = y / 8;
            let idx = (bx
                .wrapping_mul(31)
                .wrapping_add(by.wrapping_mul(17))
                .wrapping_add(seed)) as usize
                % palette.len();
            let c = palette[idx];
            // 3-bit noise to create a non-trivial delta distribution.
            let n = ((x.wrapping_mul(7) ^ y.wrapping_mul(13)) & 7) as i16 - 3;
            for &ch in c.iter() {
                out.push((ch as i16 + n).clamp(0, 255) as u8);
            }
        }
    }
    out
}

fn time_one(name: &str, pixels: &[u8], w: u32, h: u32, iters: usize) -> u128 {
    let cfg = LosslessConfig::new()
        .with_lossy_palette(true)
        .with_ans(true);

    // Warm-up to populate caches and JITted code paths.
    let _ = cfg.encode(pixels, w, h, PixelLayout::Rgb8).expect("warmup");

    // Interleaved-style: just measure many iterations back-to-back.
    let t0 = Instant::now();
    let mut total_bytes = 0usize;
    for _ in 0..iters {
        let out = cfg.encode(pixels, w, h, PixelLayout::Rgb8).expect("encode");
        total_bytes += out.len();
    }
    let dt = t0.elapsed();
    let per_ns = dt.as_nanos() / iters as u128;
    eprintln!(
        "{name:<28} iters={iters:>3} per_iter={per_ns:>10} ns  total_bytes={total_bytes} ({:.1}KB/iter)",
        total_bytes as f64 / iters as f64 / 1024.0
    );
    per_ns
}

fn main() {
    // Multiple distinct images to avoid measuring a degenerate case.
    let cases: Vec<(&str, u32, u32, usize, u32)> = vec![
        ("256x256 / 8 colors / s=1", 256, 256, 8, 1),
        ("256x256 / 16 colors / s=2", 256, 256, 16, 2),
        ("256x256 / 32 colors / s=3", 256, 256, 32, 3),
        ("256x256 / 64 colors / s=4", 256, 256, 64, 4),
    ];
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    println!("# bench_lossy_palette  iters_per_case={iters}");
    let mut total = 0u128;
    for (name, w, h, k, s) in cases {
        let img = make_palette_image(s, w, h, k);
        total += time_one(name, &img, w, h, iters);
    }
    println!("# sum-of-medians (ns) = {}", total);
}
