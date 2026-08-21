//! Score decoded PNG pairs with Rust fast-ssim2 — decoder-independent
//! quality attribution (W44-232 follow-on, 2026-08-21).
//!
//! Unlike `score_jxl_files` (which decodes .jxl via jxl-oxide in-process),
//! this takes ALREADY-DECODED PNGs, so the decoder is held constant by the
//! caller (e.g. both files decoded with djxl). Used to separate "the
//! bitstream is worse" from "the in-process decode path scores worse".
//!
//! usage:
//!   ssim2_png_pair <source.png> <label1>=<decoded1.png> [<label2>=<b.png> ...]

use fast_ssim2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};

fn load_rgb_f32(path: &str) -> (Vec<[f32; 3]>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("load {path}: {e}"))
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let px = img
        .pixels()
        .map(|p| {
            [
                p.0[0] as f32 / 255.0,
                p.0[1] as f32 / 255.0,
                p.0[2] as f32 / 255.0,
            ]
        })
        .collect();
    (px, w, h)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: ssim2_png_pair <source.png> <label>=<decoded.png> ...");
        std::process::exit(2);
    }
    let (src, w, h) = load_rgb_f32(&args[0]);
    for spec in &args[1..] {
        let (label, path) = spec.split_once('=').expect("label=path");
        let (dec, dw, dh) = load_rgb_f32(path);
        assert_eq!((w, h), (dw, dh), "{label}: dimension mismatch");
        let source = Rgb::new(
            src.clone(),
            w,
            h,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .expect("source Rgb");
        let distorted = Rgb::new(dec, w, h, TransferCharacteristic::SRGB, ColorPrimaries::BT709)
            .expect("distorted Rgb");
        let ssim2 = compute_frame_ssimulacra2(source, distorted).expect("ssim2");
        println!("{label}\tssim2={ssim2:.4}");
    }
}
