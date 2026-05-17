//! A/B comparison of `PatchesData::is_cost_effective` in
//! `EncoderMode::Experimental` — the path where the per-set savings
//! constant actually fires (RFC#45 chunk 4 follow-up).
//!
//! The default `EncoderMode::Reference` always admits patches and runs
//! the per-patch gate, so the `is_cost_effective` constant change is
//! invisible there (verified: `patches_gate_bench` is byte-identical
//! across W5-5 / chunk-4 builds).
//!
//! In `EncoderMode::Experimental` the per-set gate IS the only patches
//! gate. The pre-chunk-4 `C = 0.3` constant rejects every winning case
//! where ref-frame overhead exceeded ~3-4 KB; the chunk-4 `C = 1.0`
//! constant matches the empirical median bytes-per-pixel savings, so
//! the gate should now admit the winning cases.
//!
//! Output: bytes for `Experimental(patches default)` vs
//! `Experimental(patches forced-off)` to show that recalibration
//! re-activates patches admission on the Experimental path.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --example patches_gate_experimental_ab \
//!       > benchmarks/patches_gate_experimental_ab_2026-05-17.tsv

use jxl_encoder::{EncoderMode, LossyConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

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

    let cid = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let photos = [
        ("1025469.png", format!("{}/1025469.png", cid)),
        ("1044329.png", format!("{}/1044329.png", cid)),
        ("1189261.png", format!("{}/1189261.png", cid)),
        ("1279330.png", format!("{}/1279330.png", cid)),
        ("1418519.png", format!("{}/1418519.png", cid)),
    ];

    let distances = [0.5_f32, 1.0, 2.0, 4.0];

    println!("class\timage\twidth\theight\tdistance\texp_with_patches\texp_no_patches\tdelta_B");

    for (label, set) in [("screenshot", &screenshots[..]), ("photo", &photos[..])] {
        for (name, path) in set {
            let Ok(img) = image::open(path) else {
                eprintln!("WARN: failed to open {path}");
                continue;
            };
            let img = img.to_rgb8();
            let (w, h) = (img.width(), img.height());
            let rgb = img.as_raw();
            for &d in &distances {
                let exp_with = LossyConfig::new(d)
                    .with_mode(EncoderMode::Experimental)
                    .encode(rgb, w, h, PixelLayout::Rgb8)
                    .expect("encode failed");
                let exp_without = LossyConfig::new(d)
                    .with_mode(EncoderMode::Experimental)
                    .with_patches(false)
                    .encode(rgb, w, h, PixelLayout::Rgb8)
                    .expect("encode failed");
                let delta = exp_without.len() as i64 - exp_with.len() as i64;
                println!(
                    "{label}\t{name}\t{w}\t{h}\t{d:.2}\t{}\t{}\t{delta}",
                    exp_with.len(),
                    exp_without.len()
                );
            }
        }
    }
}
