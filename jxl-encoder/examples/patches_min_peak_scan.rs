//! W41-1 follow-up: drive every gb82-sc screenshot through the lossy
//! pipeline at all 6 distances {0.5, 1.0, 2.0, 3.0, 4.0, 5.0} × effort 7.
//!
//! Purpose: byte-emission test bed for the next-chunk follow-on to the
//! W41-1 audit (issue #52). Pair the byte counts against cjxl or against
//! a temporary in-tree `eprintln!` in `vardct::encoder::encode_inner` (in
//! the patches-detection block) to derive per-image / per-distance patch
//! admission stats. The shipped recording from 2026-05-18 (proving the
//! patch set is identical between `min_peak=1` and `min_peak=2` on every
//! wedge image except `windows95.png`) is at
//! `benchmarks/patches_min_peak_admission_2026-05-19.txt`.
//!
//! Run: cargo run --release -p jxl-encoder --example patches_min_peak_scan

use image::GenericImageView;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc");
    let files = [
        "codec_wiki.png",
        "gmessages.png",
        "graph.png",
        "gui.png",
        "imac_dark.png",
        "imac_g3.png",
        "imac_g3_strip.png",
        "imessage.png",
        "terminal.png",
        "windows95.png",
        "windows.png",
    ];
    let distances: Vec<f32> = vec![0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
    for f in files.iter() {
        let path = dir.join(f);
        let img = match image::open(&path) {
            Ok(i) => i,
            Err(_) => continue,
        };
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();
        for &d in &distances {
            let cfg = LossyConfig::new(d).with_effort(7);
            match cfg.encode(&rgb, w, h, PixelLayout::Rgb8) {
                Ok(bytes) => println!("{}\td={:.2}\t{} bytes", f, d, bytes.len()),
                Err(e) => eprintln!("{} d={:.2} ERR: {:?}", f, d, e),
            }
        }
    }
}
