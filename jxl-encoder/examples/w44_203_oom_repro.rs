//! W44-203 OOM repro for a single (image, effort, distance, strategy).
//!
//! Usage: --image NAME --effort N --distance F --strategy {zenjxl|libjxl}
//!        [--corpus PATH]

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::PathBuf;
use std::time::Instant;

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn resolve_image(name: &str) -> PathBuf {
    // Best-effort lookup against the W44-170 corpus manifest layout.
    let candidates = [
        format!("CID22/CID22-512/validation/{name}.png"),
        format!("gb82-sc/{name}.png"),
        format!("CLIC2025-1024/{name}.png"),
    ];
    let base = corpus_dir();
    for c in &candidates {
        let p = base.join(c);
        if p.exists() {
            return p;
        }
    }
    panic!("could not find image {name} under {}", base.display());
}

fn main() {
    let mut image: Option<String> = None;
    let mut effort: u8 = 5;
    let mut distance: f32 = 1.0;
    let mut strategy_str: String = "zenjxl".into();
    let mut threads: usize = 1;
    let mut rayon_threads: Option<usize> = None;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--image" => image = Some(it.next().expect("--image VALUE")),
            "--effort" => effort = it.next().unwrap().parse().unwrap(),
            "--distance" => distance = it.next().unwrap().parse().unwrap(),
            "--strategy" => strategy_str = it.next().unwrap(),
            "--threads" => threads = it.next().unwrap().parse().unwrap(),
            "--rayon-threads" => rayon_threads = Some(it.next().unwrap().parse().unwrap()),
            other => panic!("unknown arg: {other}"),
        }
    }

    if let Some(rt) = rayon_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(rt)
            .build_global()
            .ok();
        eprintln!("[w44-203] rayon global pool = {rt}");
    }

    let image = image.expect("--image required");
    let strategy = match strategy_str.as_str() {
        "zenjxl" => EncoderStrategy::Zenjxl,
        "libjxl" => EncoderStrategy::Libjxl,
        other => panic!("unknown strategy: {other}"),
    };

    let png_path = resolve_image(&image);
    eprintln!("[w44-203] image={} ({})", image, png_path.display());
    eprintln!(
        "[w44-203] effort={effort} distance={distance} strategy={strategy_str} threads={threads}"
    );

    // Load PNG → linear sRGB u8 with the same pipeline as the sweep example.
    let img = image::open(&png_path).expect("failed to load PNG");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let rgb_bytes = rgb.into_raw();
    eprintln!(
        "[w44-203] dim={}x{} = {:.2} MP",
        w,
        h,
        (w as f64 * h as f64) / 1.0e6
    );

    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(strategy)
        .with_threads(threads);

    eprintln!("[w44-203] encoding ...");
    let start = Instant::now();
    let result = cfg.encode(&rgb_bytes, w, h, PixelLayout::Rgb8);
    let elapsed = start.elapsed();
    match result {
        Ok(bytes) => {
            eprintln!(
                "[w44-203] OK in {:?} → {} bytes ({:.3} bpp)",
                elapsed,
                bytes.len(),
                (bytes.len() as f64 * 8.0) / (w as f64 * h as f64)
            );
        }
        Err(e) => {
            eprintln!("[w44-203] FAILED in {:?}: {}", elapsed, e);
            std::process::exit(1);
        }
    }
}
