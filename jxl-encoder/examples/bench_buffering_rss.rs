//! Peak-RSS bench: encode a 4096×4096 photo with FullBuffered vs
//! BufferedOutput vs FullStreaming. Chunk-3 expectation: all variants
//! route through the same per-region loop, so peak RSS should be
//! identical (within measurement noise). Chunk-5 will introduce the
//! actual memory savings on BufferedOutput/FullStreaming.
//!
//! Run with: cargo run --release -p jxl-encoder --example bench_buffering_rss [variant]

use jxl_encoder::{Buffering, LossyConfig, PixelLayout};

fn main() {
    let variant_str = std::env::args().nth(1).unwrap_or_else(|| "full".into());
    let variant = match variant_str.as_str() {
        "full" | "0" => Buffering::FullBuffered,
        "buffered" | "2" => Buffering::BufferedOutput,
        "streaming" | "3" => Buffering::FullStreaming,
        "auto" | "-1" => Buffering::Auto,
        _ => panic!("unknown variant: {variant_str}"),
    };
    // 3072x3072 (= 9.4 MP, 2x2 = 4 DC groups) is the default. Default
    // memory budget is 2 GiB; 4096² hits the cap on the lossy path.
    // Override with `./bench_buffering_rss <variant> <w> <h>`.
    let w: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3072);
    let h: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3072);
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    // Cheap fbm-ish content: not flat, not random. Mixes some structure
    // so the AC strategy / CfL paths see real work.
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let v1 = ((x * 17 + y * 11) & 0xff) as u8;
            let v2 = ((x ^ y) & 0xff) as u8;
            let v3 = (((x as u64).wrapping_mul(31) ^ (y as u64).wrapping_mul(7)) & 0xff) as u8;
            pixels[i] = v1;
            pixels[i + 1] = v2;
            pixels[i + 2] = v3;
        }
    }
    let t0 = std::time::Instant::now();
    let bytes = LossyConfig::new(1.0)
        .with_buffering(variant)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let dt = t0.elapsed();
    eprintln!(
        "variant={variant:?} bytes={} took={:.2}s",
        bytes.len(),
        dt.as_secs_f64()
    );
}
