/// Minimal encode benchmark for WASM/AArch64 speed testing.
/// Build: cargo build --example wasm_bench --target wasm32-wasip1 --release --no-default-features
/// Run:   wasmtime ./target/wasm32-wasip1/release/examples/wasm_bench.wasm
use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};

fn main() {
    let width = 256u32;
    let height = 256u32;
    let bpp = 3;

    // Generate a gradient test image
    let mut pixels = vec![0u8; (width * height) as usize * bpp];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) as usize) * bpp;
            pixels[idx] = (x & 255) as u8;
            pixels[idx + 1] = (y & 255) as u8;
            pixels[idx + 2] = ((x + y) & 255) as u8;
        }
    }

    // Lossless encode
    let t0 = std::time::Instant::now();
    let lossless = LosslessConfig::new()
        .encode(&pixels, width, height, PixelLayout::Rgb8)
        .expect("lossless encode failed");
    let t_lossless = t0.elapsed();

    // Lossy encode
    let t1 = std::time::Instant::now();
    let lossy = LossyConfig::new(1.0)
        .encode(&pixels, width, height, PixelLayout::Rgb8)
        .expect("lossy encode failed");
    let t_lossy = t1.elapsed();

    eprintln!(
        "256x256 lossless: {} bytes in {:.1}ms",
        lossless.len(),
        t_lossless.as_secs_f64() * 1000.0
    );
    eprintln!(
        "256x256 lossy d=1.0: {} bytes in {:.1}ms",
        lossy.len(),
        t_lossy.as_secs_f64() * 1000.0
    );

    // Larger encode for better timing
    let width2 = 1024u32;
    let height2 = 1024u32;
    let mut pixels2 = vec![0u8; (width2 * height2) as usize * bpp];
    for y in 0..height2 {
        for x in 0..width2 {
            let idx = ((y * width2 + x) as usize) * bpp;
            pixels2[idx] = (x & 255) as u8;
            pixels2[idx + 1] = (y & 255) as u8;
            pixels2[idx + 2] = ((x.wrapping_mul(y)) & 255) as u8;
        }
    }

    let t2 = std::time::Instant::now();
    let lossless2 = LosslessConfig::new()
        .encode(&pixels2, width2, height2, PixelLayout::Rgb8)
        .expect("lossless 1024 encode failed");
    let t_lossless2 = t2.elapsed();

    let t3 = std::time::Instant::now();
    let lossy2 = LossyConfig::new(1.0)
        .encode(&pixels2, width2, height2, PixelLayout::Rgb8)
        .expect("lossy 1024 encode failed");
    let t_lossy2 = t3.elapsed();

    eprintln!(
        "1024x1024 lossless: {} bytes in {:.1}ms",
        lossless2.len(),
        t_lossless2.as_secs_f64() * 1000.0
    );
    eprintln!(
        "1024x1024 lossy d=1.0: {} bytes in {:.1}ms",
        lossy2.len(),
        t_lossy2.as_secs_f64() * 1000.0
    );
}
