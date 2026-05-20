//! W44-96 hot-path mask1x1_median probe — invoke production encoder
//! exactly as the dispatch site sees it, so we know which gate fires.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features 'parallel debug-w44-65' \
//!     --example w44_96_mask_probe

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

fn main() {
    // Test cells of interest (image, effort, distance).
    let cells: &[(&str, u8, f32)] = &[
        ("1420710.png", 6, 5.0),
        ("1420710.png", 6, 6.0),
        ("1531677.png", 5, 6.0),
        ("1531677.png", 6, 6.0),
        ("2389166.png", 6, 4.0),
        ("2389166.png", 7, 5.0),
        ("3637739.png", 5, 5.0),
        ("3637739.png", 7, 4.0),
        ("3637739.png", 7, 5.0),
        ("1044329.png", 5, 5.0),
        ("1044329.png", 7, 5.0),
        ("1189261.png", 7, 4.0),
        ("1418519.png", 7, 5.0),
    ];

    eprintln!("# W44-96 hot-path mask1x1_median per (image, effort, distance)");
    eprintln!("# Watch for 'W44-65 dbg: distance=... mask1x1_median=Some(X) ...' lines");
    for (name, eff, d) in cells {
        let path = Path::new(CID22).join(name);
        let img = image::open(&path).expect("open");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let raw = rgb.into_raw();
        eprintln!("=== ENCODE: {} e{} d={}", name, eff, d);
        let _bytes = LossyConfig::new(*d)
            .with_effort(*eff)
            .with_threads(8)
            .encode(&raw, w, h, PixelLayout::Rgb8)
            .expect("encode");
    }
}
