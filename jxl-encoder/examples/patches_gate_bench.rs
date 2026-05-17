//! Bench harness for the patches cost-effectiveness gate.
//!
//! Encodes 5 screenshots + 5 photos at d∈{0.5, 1.0, 2.0, 4.0} and prints
//! bytes per cell. Used to compare main vs the gate fix; run twice
//! (before/after) and diff the bytes.
//!
//! Usage: cargo run --release -p jxl-encoder --example patches_gate_bench
use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    // 5 screenshots from gb82-sc (mix of low-d-regression cases)
    let screenshots = [
        (
            "windows95.png",
            format!("{}/work/codec-corpus/gb82-sc/windows95.png", base),
        ),
        (
            "terminal.png",
            format!("{}/work/codec-corpus/gb82-sc/terminal.png", base),
        ),
        (
            "codec_wiki.png",
            format!("{}/work/codec-corpus/gb82-sc/codec_wiki.png", base),
        ),
        (
            "imac_g3.png",
            format!("{}/work/codec-corpus/gb82-sc/imac_g3.png", base),
        ),
        (
            "windows.png",
            format!("{}/work/codec-corpus/gb82-sc/windows.png", base),
        ),
    ];

    // 5 photos from CID22 validation (the canonical "patches must not regress" set)
    let cid = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let photos = [
        ("1025469.png", format!("{}/1025469.png", cid)),
        ("1044329.png", format!("{}/1044329.png", cid)),
        ("1189261.png", format!("{}/1189261.png", cid)),
        ("1279330.png", format!("{}/1279330.png", cid)),
        ("1418519.png", format!("{}/1418519.png", cid)),
    ];

    let distances = [0.5_f32, 1.0, 2.0, 4.0];

    println!("class\timage\twidth\theight\tdistance\tbytes");

    for (label, set) in [("screenshot", &screenshots[..]), ("photo", &photos[..])] {
        for (name, path) in set {
            let Ok(img) = image::open(path) else {
                eprintln!("WARN: failed to open {}", path);
                continue;
            };
            let img = img.to_rgb8();
            let (w, h) = (img.width(), img.height());
            let rgb = img.as_raw();
            for &d in &distances {
                let bytes = LossyConfig::new(d)
                    .encode(rgb, w, h, PixelLayout::Rgb8)
                    .expect("encode failed");
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{}",
                    label,
                    name,
                    w,
                    h,
                    d,
                    bytes.len()
                );
            }
        }
    }
}
