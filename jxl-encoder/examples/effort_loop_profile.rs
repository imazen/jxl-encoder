//! Keeps one (image, size, effort, distance) encode running in a tight loop so
//! a sampling profiler (`sample <pid>` on macOS, `perf record` on Linux) has
//! something to attach to. Prints nothing until it stops.
//!
//! Exists because `benchmarks/ladder_vs_cjxl_2026-09-10.*` put lossy e3 at
//! 1.40x cjxl v0.12 at 1024^2 threads=1 and 1.45x at 2048^2 — the worst band on
//! the ladder — and the phase timers only localise it as far as `build_codes`
//! (10-14 ms of a 28-35 ms encode). Going finer needs a profiler.
//!
//! Run: `effort_loop_profile <png> <size> <effort> <distance> <seconds> [threads]`
//!
//! `threads` defaults to 1. It is a real argument rather than `RAYON_NUM_THREADS`
//! because the encoder takes its thread count from `LossyConfig::with_threads`
//! and ignores the environment — a 2026-09-10 run that set only
//! `RAYON_NUM_THREADS` measured the SAME single-threaded encode twice and
//! reported it as a threads=1 vs threads=8 comparison.
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
    let threads: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    // `parallel` is NOT a jxl-encoder default feature — only the CLI turns it
    // on — so a plain `cargo run --example` build encodes single-threaded no
    // matter what `with_threads` says. Refuse rather than silently report two
    // identical single-threaded runs as a threads=1 vs threads=8 comparison,
    // which is exactly what happened on 2026-09-10.
    #[cfg(not(feature = "parallel"))]
    if threads > 1 {
        eprintln!(
            "refusing threads={threads}: this binary was built WITHOUT the \
             `parallel` feature, so the encode would run single-threaded and \
             the numbers would be a lie. Rebuild with \
             `--features parallel,__env_var_diagnostics`."
        );
        std::process::exit(2);
    }

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
            .with_threads(threads)
            .encode_request(n, n, PixelLayout::Rgb8)
            .encode(&src)
            .expect("encode");
        bytes = enc.len();
        iters += 1;
    }
    eprintln!(
        "{n}x{n} e{effort} d{d} t{threads}: {iters} encodes, {bytes} B, {:.1} ms/encode",
        secs as f64 * 1000.0 / iters as f64
    );
}
