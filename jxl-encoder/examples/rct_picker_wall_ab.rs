//! Wall-clock A/B for the RCT smart picker (chunk 1 investigation).
//!
//! On a small batch of CLIC 1024-px images, measures wall-clock of:
//!   - baseline: nb_rcts_to_try=7 (current e7 default)
//!   - always-10: with_force_rct(Some(RctType(10)))
//!   - 2-trial picker: pick best of {RCT 10, RCT 40} (2-trial fixed-pair sim)
//!
//! The smart picker uses a *fixed pair* (RCT-10 + RCT-40) which approximates
//! the RF top-2 prediction without the zenanalyze runtime cost. This isolates
//! the wall-clock benefit of trial reduction from the zenanalyze overhead.
//!
//! Uses the existing public `with_force_rct` API which dispatches through
//! `nb_rcts_to_try=0` semantics (skip search). Reports paired stats per image.

use jxl_encoder::RctType;
use jxl_encoder::api::{LosslessConfig, PixelLayout};
use std::path::Path;
use std::time::Instant;

const IMAGES: &[&str] = &[
    "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    "/home/lilith/work/codec-corpus/clic2025-1024/0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png",
    "/home/lilith/work/codec-corpus/clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
];

const ITERS: u32 = 3;

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap();
    let rgb = img.to_rgb8();
    (rgb.as_raw().clone(), rgb.width(), rgb.height())
}

fn bench(name: &str, rgb: &[u8], w: u32, h: u32, force: Option<RctType>) -> (usize, f64) {
    let mut times = Vec::new();
    let mut bytes = 0usize;
    for _ in 0..ITERS {
        let cfg = LosslessConfig::new().with_effort(7).with_force_rct(force);
        let start = Instant::now();
        let out = cfg.encode(rgb, w, h, PixelLayout::Rgb8).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        bytes = out.len();
    }
    let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "  {:<30} bytes={:>9}  best={:>7.0}ms  mean={:>7.0}ms",
        name, bytes, best, mean
    );
    (bytes, best)
}

fn main() {
    println!(
        "RCT picker wall-clock A/B (8-thread rayon, ITERS={})",
        ITERS
    );
    for path in IMAGES {
        let (rgb, w, h) = load_png(Path::new(path));
        let fname = Path::new(path).file_name().unwrap().to_str().unwrap();
        println!("\n== {} ({}x{}) ==", &fname[..16], w, h);

        let (b_default, t_default) = bench("nb_rcts=7 (default)", &rgb, w, h, None);
        let (b_force10, t_force10) = bench("force RCT-10", &rgb, w, h, Some(RctType(10)));
        // Note: there is no public 2-trial API; we just compare 7-trial vs 1-trial.
        // The smart-picker wall cost would be (1 zenanalyze pass + 2 trials).

        println!("  --");
        println!(
            "  force-10 vs default: bytes {:+.2}% wall {:+.1}%",
            100.0 * (b_force10 as f64 - b_default as f64) / b_default as f64,
            100.0 * (t_force10 - t_default) / t_default,
        );
    }
}
