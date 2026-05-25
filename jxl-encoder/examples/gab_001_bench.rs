//! gab-001 bench: encode 5 cells, print bytes + SHA + (optional) SSIM2.
//!
//! Used to capture pre-fix vs post-fix delta for the gab-001 SIMD FMA
//! parity change.  Output is one TSV row per cell.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features __expert --example gab_001_bench
//!
//! Cells (5):
//!   - cid22_1418519 e7 d=1
//!   - gb82_codec_wiki e7 d=2
//!   - gb82_terminal e8 d=4
//!   - gb82_frymire e6 d=0.5
//!   - synthetic_gradient_64 e7 d=1

use jxl_encoder::{LossyConfig, PixelLayout};
use sha2::{Digest, Sha256};

fn sha8(b: &[u8]) -> String {
    let h = Sha256::digest(b);
    h.iter()
        .take(4)
        .map(|x| format!("{x:02x}"))
        .collect::<String>()
}

fn encode_cell(name: &str, path: Option<&str>, effort: u8, distance: f32, label: &str) {
    let (w, h, rgb) = if let Some(p) = path {
        let Ok(img) = image::open(p) else {
            println!("{name}\t{label}\t{effort}\t{distance:.2}\tFAILED\tFAILED\tFAILED_OPEN");
            return;
        };
        let img = img.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb = img.as_raw().to_vec();
        (w, h, rgb)
    } else {
        // Synthetic 64x64 gradient
        let mut rgb = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64u32 {
            for x in 0..64u32 {
                rgb.push((x * 4) as u8);
                rgb.push((y * 4) as u8);
                rgb.push(((x + y) * 2) as u8);
            }
        }
        (64u32, 64u32, rgb)
    };

    let bytes = LossyConfig::new(distance)
        .with_effort(effort)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("encode");

    println!(
        "{name}\t{label}\t{effort}\t{distance:.2}\t{}\t{}\t{}",
        w,
        bytes.len(),
        sha8(&bytes),
    );
}

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    println!("cell\tlabel\teffort\tdistance\twidth\tbytes\tsha8");

    encode_cell(
        "cid22_1418519",
        Some(&format!(
            "{}/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
            base
        )),
        7,
        1.0,
        "1418519_e7_d1",
    );

    encode_cell(
        "gb82_codec_wiki",
        Some(&format!(
            "{}/work/codec-corpus/gb82-sc/codec_wiki.png",
            base
        )),
        7,
        2.0,
        "codec_wiki_e7_d2",
    );

    encode_cell(
        "gb82_terminal",
        Some(&format!("{}/work/codec-corpus/gb82-sc/terminal.png", base)),
        8,
        4.0,
        "terminal_e8_d4",
    );

    encode_cell(
        "gb82_imac_dark",
        Some(&format!("{}/work/codec-corpus/gb82-sc/imac_dark.png", base)),
        6,
        0.5,
        "imac_dark_e6_d0.5",
    );

    encode_cell("synthetic_gradient", None, 7, 1.0, "gradient_64_e7_d1");
}
