//! Emit one Lloyd-Max-encoded .jxl file for djxl + jxl-rs sanity testing.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features '__expert parallel-tree-learning' \
//!     --example lloyd_max_emit_jxl -- <input.png> <output.jxl>

use jxl_encoder::LosslessInternalParams;
use jxl_encoder::api::{LosslessConfig, PixelLayout};

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect("usage: <input.png> <output.jxl>");
    let output = args.next().expect("usage: <input.png> <output.jxl>");

    let img = image::open(&input).expect("read input").to_rgb8();
    let (w, h) = img.dimensions();

    let mut params = LosslessInternalParams::default();
    params.lloyd_max_buckets = Some(true);
    let cfg = LosslessConfig::new()
        .with_effort(7)
        .with_threads(8)
        .with_internal_params(params);

    let bytes = cfg
        .encode(img.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("lloyd_max lossless encode");

    std::fs::write(&output, &bytes).expect("write output");
    eprintln!(
        "wrote {} bytes to {} ({}x{} RGB, Lloyd-Max bucket boundaries on props 4/5/15)",
        bytes.len(),
        output,
        w,
        h
    );
}
