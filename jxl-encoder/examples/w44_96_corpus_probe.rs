//! W44-96 corpus-wide probe: identify ALL CID22 photos where the
//! W44-29 gate fires (mask1x1_median < 50) and report their proxy
//! values so we can verify the WANT_Z discriminator stays clean
//! across the corpus.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features 'parallel debug-w44-65' \
//!     --example w44_96_corpus_probe

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

fn main() {
    // Iterate every PNG in the validation set; encode at e7 d=5 to make
    // the W44-65 debug print fire once per image.
    let dir = Path::new(CID22);
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("read CID22")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    paths.sort();

    eprintln!("# W44-96 corpus probe — mask1x1_median for all CID22 validation images at e7 d=5");
    for path in &paths {
        let img = image::open(path).expect("open");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let raw = rgb.into_raw();
        eprintln!(
            "=== ENCODE: {}",
            path.file_name().unwrap().to_string_lossy()
        );
        let _bytes = LossyConfig::new(5.0)
            .with_effort(7)
            .with_threads(8)
            .encode(&raw, w, h, PixelLayout::Rgb8)
            .expect("encode");
    }
}
