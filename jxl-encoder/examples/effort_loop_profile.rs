//! Keeps one (image, size, effort, distance) encode running in a tight loop so
//! a sampling profiler (`sample <pid>` on macOS, `perf record` on Linux) has
//! something to attach to. Prints nothing until it stops.
//!
//! Exists because `benchmarks/ladder_vs_cjxl_2026-09-10.*` put lossy e3 at
//! 1.40x cjxl v0.12 at 1024^2 threads=1 and 1.45x at 2048^2 — the worst band on
//! the ladder — and the phase timers only localise it as far as `build_codes`
//! (10-14 ms of a 28-35 ms encode). Going finer needs a profiler.
//!
//! Run: `effort_loop_profile <png> <size> <effort> <distance> <seconds>`
use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::time::{Duration, Instant};

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .expect("usage: <png> <size> <effort> <distance> <secs>");
    let want: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let effort: u8 = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let d: f32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let secs: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(20);

    let rgb = image::open(&path).expect("open").to_rgb8();
    let n = want.min(rgb.width()).min(rgb.height());
    let (x0, y0) = ((rgb.width() - n) / 2, (rgb.height() - n) / 2);
    let src: Vec<u8> = image::imageops::crop_imm(&rgb, x0, y0, n, n)
        .to_image()
        .as_raw()
        .clone();

    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut iters = 0u64;
    let mut bytes = 0usize;
    while Instant::now() < deadline {
        let enc = LossyConfig::new(d)
            .with_effort(effort)
            .encode_request(n, n, PixelLayout::Rgb8)
            .encode(&src)
            .expect("encode");
        bytes = enc.len();
        iters += 1;
    }
    eprintln!("{n}x{n} e{effort} d{d}: {iters} encodes, {bytes} B");
}
