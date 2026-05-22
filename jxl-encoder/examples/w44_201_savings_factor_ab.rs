//! W44-201 A/B: try a more conservative savings factor in the
//! cost-benefit gate, instead of (or in addition to) the bucket disable.
//!
//! The current gate uses `total_savings_bits = (nzeros_custom - nzeros_natural) * max_count`,
//! assuming 1 bit saved per extra trailing zero per block. The empirical
//! AC encoding cost per trailing zero is closer to 0.3-0.5 bits (run-length
//! encoding overhead).
//!
//! This bench tries a savings_factor in {1.0 (default), 0.7, 0.5, 0.3}
//! and measures byte impact across photos + screenshots. Lower factor
//! means the gate is MORE conservative (rejects more custom orders).

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn encode(pixels: &[u8], w: u32, h: u32, distance: f32, savings_factor: Option<f32>) -> usize {
    unsafe {
        match savings_factor {
            Some(f) => std::env::set_var("JXL_W44_201_SAVINGS_FACTOR", format!("{}", f)),
            None => std::env::remove_var("JXL_W44_201_SAVINGS_FACTOR"),
        }
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(7)
        .with_strategy(EncoderStrategy::Zenjxl);
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("encode");
    buf.len()
}

fn main() {
    let cells: Vec<(&str, &str, f32)> = vec![
        ("cid22_3637739_d4", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png", 4.0),
        ("cid22_3637739_d2", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png", 2.0),
        ("cid22_3637739_d6", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png", 6.0),
        ("cid22_1418519_d4", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png", 4.0),
        ("cid22_1420710_d4", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png", 4.0),
        ("cid22_1420710_d6", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png", 6.0),
        ("cid22_1531677_d4", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png", 4.0),
        ("cid22_1189261_d4", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png", 4.0),
        ("cid22_1025469_d4", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png", 4.0),
        ("cid22_2389166_d4", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/2389166.png", 4.0),
        ("gb82_codec_wiki_d2", "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png", 2.0),
        ("gb82_imac_g3_d2", "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png", 2.0),
        ("gb82_terminal_d2", "/home/lilith/work/codec-corpus/gb82-sc/terminal.png", 2.0),
        ("gb82_windows95_d2", "/home/lilith/work/codec-corpus/gb82-sc/windows95.png", 2.0),
    ];

    print!("label");
    for f in &[1.0_f32, 0.7, 0.5, 0.3] {
        print!("\tf={}", f);
    }
    println!();

    let mut sums = [0i64; 4];
    let factors = [1.0_f32, 0.7, 0.5, 0.3];
    for (label, path, d) in &cells {
        let Some((pixels, w, h)) = load_png(Path::new(path)) else { continue };
        print!("{}", label);
        for (i, &f) in factors.iter().enumerate() {
            let bytes = encode(&pixels, w, h, *d, Some(f));
            sums[i] += bytes as i64;
            print!("\t{}", bytes);
        }
        println!();
    }
    print!("TOTAL");
    for s in &sums {
        print!("\t{}", s);
    }
    println!();
    print!("delta_vs_f=1.0");
    for &s in &sums[1..] {
        let d = s - sums[0];
        print!("\t{:+} ({:+.2}%)", d, 100.0 * d as f64 / sums[0] as f64);
    }
    println!();
}
