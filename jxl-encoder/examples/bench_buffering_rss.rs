//! Peak-RSS bench: encode a NxN photo with FullBuffered vs
//! BufferedOutput vs FullStreaming. Chunk-8b (#11) expectation: all
//! variants still route through the same whole-image XYB plane
//! buffers (the `XybRegionSource` trait + walker land the call-site
//! seam but no streaming source is wired yet). Chunk-8c will
//! materialise the streaming source + drop per-region buffers, at
//! which point this bench should show a peak-RSS drop on
//! BufferedOutput/FullStreaming.
//!
//! Run with: cargo run --release -p jxl-encoder --example bench_buffering_rss [variant] [w] [h] [mb_cap]

use jxl_encoder::{Buffering, Limits, LossyConfig, PixelLayout};

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
    // memory budget is 2 GiB; 4096² hits the cap on the lossy path so
    // pass an mb_cap (>=4096) to use the bigger image.
    // Override with `./bench_buffering_rss <variant> <w> <h> [mb_cap]`.
    let w: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3072);
    let h: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3072);
    let mb_cap: Option<u64> = std::env::args().nth(4).and_then(|s| s.parse().ok());
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
    let config = LossyConfig::new(1.0).with_buffering(variant);
    let t0 = std::time::Instant::now();
    let bytes = if let Some(mb) = mb_cap {
        let limits = Limits::new().with_max_memory_bytes(mb * 1024 * 1024);
        config
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&limits)
            .encode(&pixels)
            .expect("encode failed")
    } else {
        config
            .encode(&pixels, w, h, PixelLayout::Rgb8)
            .expect("encode failed")
    };
    let dt = t0.elapsed();
    eprintln!(
        "variant={variant:?} bytes={} took={:.2}s",
        bytes.len(),
        dt.as_secs_f64()
    );
}
