//! W44-35 acceptance check.
//!
//! Compares default-encoding bytes on:
//!   - 5 W44-34 1418519 cells (expect FIXED — match cjxl or beat)
//!   - 4 screenshots × e7 × d ∈ {3, 4, 5, 6} (expect NO regression vs pre-W44-35)
//!   - F-D residual cells (1531677 + 1420710 at d=5/6/etc — expect not WORSE)
//!
//! Build:
//!   cargo build -p jxl-encoder --release \
//!       --features 'parallel butteraugli-loop ssim2-loop' \
//!       --example w44_35_acceptance_check
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!       --features 'parallel butteraugli-loop ssim2-loop' \
//!       --example w44_35_acceptance_check

use std::path::PathBuf;

use jxl_encoder::api::{LossyConfig, PixelLayout};

/// (label, path, effort, distance, expected behaviour).
const CELLS: &[(&str, &str, u8, f32, &str)] = &[
    // W44-34 1418519 cells — should ALL get the -5 to -7% win
    (
        "1418519/e6/d1.0",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        6,
        1.0,
        "expected -5.86%",
    ),
    (
        "1418519/e6/d1.2",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        6,
        1.2,
        "expected -5.60%",
    ),
    (
        "1418519/e6/d1.6",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        6,
        1.6,
        "expected -6.80%",
    ),
    (
        "1418519/e7/d1.2",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        7,
        1.2,
        "expected -5.59%",
    ),
    (
        "1418519/e7/d1.6",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        7,
        1.6,
        "expected -6.78%",
    ),
    // Screenshots: each at e7 × d ∈ {3, 4, 5, 6} — gate doesn't fire (d >= 2.0)
    // so smoothness hint is irrelevant; bytes should be IDENTICAL to pre-W44-35.
    (
        "terminal/e7/d3",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        7,
        3.0,
        "expected: ±0%",
    ),
    (
        "terminal/e7/d4",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        7,
        4.0,
        "expected: ±0%",
    ),
    (
        "terminal/e7/d5",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        7,
        5.0,
        "expected: ±0%",
    ),
    (
        "terminal/e7/d6",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        7,
        6.0,
        "expected: ±0%",
    ),
    (
        "codec_wiki/e7/d3",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        7,
        3.0,
        "expected: ±0%",
    ),
    (
        "codec_wiki/e7/d4",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        7,
        4.0,
        "expected: ±0%",
    ),
    (
        "codec_wiki/e7/d5",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        7,
        5.0,
        "expected: ±0%",
    ),
    (
        "codec_wiki/e7/d6",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        7,
        6.0,
        "expected: ±0%",
    ),
    (
        "imac_g3/e7/d3",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        7,
        3.0,
        "expected: ±0%",
    ),
    (
        "imac_g3/e7/d4",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        7,
        4.0,
        "expected: ±0%",
    ),
    (
        "imac_g3/e7/d5",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        7,
        5.0,
        "expected: ±0%",
    ),
    (
        "imac_g3/e7/d6",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        7,
        6.0,
        "expected: ±0%",
    ),
    (
        "windows95/e7/d3",
        "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
        7,
        3.0,
        "expected: ±0%",
    ),
    (
        "windows95/e7/d4",
        "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
        7,
        4.0,
        "expected: ±0%",
    ),
    (
        "windows95/e7/d5",
        "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
        7,
        5.0,
        "expected: ±0%",
    ),
    (
        "windows95/e7/d6",
        "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
        7,
        6.0,
        "expected: ±0%",
    ),
    // F-D residual cells (1531677 + 1420710 at d=5, 6) — gate doesn't fire
    // (d >= 2.0); should stay IDENTICAL to pre-W44-35.
    (
        "1531677/e7/d5",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        7,
        5.0,
        "expected: ±0%",
    ),
    (
        "1531677/e7/d6",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        7,
        6.0,
        "expected: ±0%",
    ),
    (
        "1420710/e7/d5",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        7,
        5.0,
        "expected: ±0%",
    ),
    (
        "1420710/e7/d6",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        7,
        6.0,
        "expected: ±0%",
    ),
    // Other small photos that the discriminator should NOT admit (pixels<500k
    // AND distance<2.0, but classifier rejects them) — should stay
    // byte-identical to pre-W44-35 (try_dct64 stays gated off).
    (
        "1531677/e7/d1.2",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        7,
        1.2,
        "expected: ±0%",
    ),
    (
        "1420710/e7/d1.2",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        7,
        1.2,
        "expected: ±0%",
    ),
    (
        "1189261/e7/d1.2",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
        7,
        1.2,
        "expected: ±0%",
    ),
    (
        "1025469/e7/d1.2",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
        7,
        1.2,
        "expected: ±0%",
    ),
];

fn load_png(path: &PathBuf) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.as_raw().clone(), w, h)
}

fn encode(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> usize {
    LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed")
        .len()
}

fn main() {
    let cell = "cell";
    let wlbl = "w";
    let hlbl = "h";
    let bytes_lbl = "bytes";
    let note = "note";
    println!("{cell:<24} {wlbl:>5}x{hlbl:<5} {bytes_lbl:>8} {note}");
    println!("{}", "-".repeat(60));
    for &(label, path_str, effort, distance, expect) in CELLS {
        let path = PathBuf::from(path_str);
        let (rgb, w, h) = load_png(&path);
        let bytes = encode(&rgb, w, h, distance, effort);
        println!("{label:<24} {w:>5}x{h:<5} {bytes:>8} {expect}");
    }
}
