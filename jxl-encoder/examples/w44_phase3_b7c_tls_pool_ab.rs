//! W44-phase3-B7c paired wall-time bench: butteraugli TLS pool vs Mutex pool.
//!
//! Measures wall-clock time of `LossyConfig`-driven encode-with-buttloop on
//! 4 photo cells at effort 9, 8-thread (default rayon pool). The bench runs
//! N iters per cell, prints per-cell median + delta vs the reference.
//!
//! The B7c change is INSIDE butteraugli (the `BufferPool` internals) and
//! has no encoder-side env hook — there's nothing in jxl-encoder to flip.
//! So this bench takes BOTH wall samples from the SAME binary, then a
//! companion script (`run_w44_phase3_b7c_ab.sh`) builds the binary against
//! butteraugli BEFORE and AFTER B7c and runs each side N times, writing
//! a paired TSV.
//!
//! Output (one binary side): TSV per-iter
//!   `cell\teffort\tdistance\titer\twall_us`
//!
//! Run (single-side):
//!
//! ```text
//! cargo run -p jxl-encoder --release --features '__expert butteraugli-loop parallel' \
//!     --example w44_phase3_b7c_tls_pool_ab -- <out.tsv>
//! ```

use std::path::PathBuf;
use std::time::Instant;

use image::ImageReader;
use jxl_encoder::api::{LossyConfig, PixelLayout};

/// Four 1024² photos drawn from CLIC 2025. Chosen for content diversity
/// (smooth, edge-heavy, textured) so the wall-time number isn't a
/// one-cell artifact. Cells were eyeballed; if any are missing they
/// are skipped (with a warning) rather than failing the bench.
const PHOTO_CELLS: &[&str] = &[
    "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png",
    "/home/lilith/work/codec-corpus/clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png",
    "/home/lilith/work/codec-corpus/clic2025-1024/11f2b039b293758398b1a7a8afa64bb2.png",
];

const TIME_ITERS: usize = 6;
const DISTANCE: f32 = 1.0;
const EFFORT: u8 = 9;

fn encode_once(rgb: &[u8], width: u32, height: u32, distance: f32, effort: u8) -> u128 {
    let cfg = LossyConfig::new(distance).with_effort(effort);
    let t0 = Instant::now();
    let _bytes = cfg
        .encode(rgb, width, height, PixelLayout::Rgb8)
        .expect("encode");
    t0.elapsed().as_micros()
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/w44_phase3_b7c_tls_pool_ab.tsv".to_string());

    let label = std::env::var("B7C_LABEL").unwrap_or_else(|_| "UNLABELED".to_string());

    let mut lines: Vec<String> = vec!["cell\teffort\tdistance\titer\twall_us\tlabel".to_string()];

    for cell in PHOTO_CELLS {
        let path = PathBuf::from(cell);
        if !path.exists() {
            eprintln!("SKIP {cell} (not found)");
            continue;
        }
        let img = ImageReader::open(&path)
            .expect("open")
            .decode()
            .expect("decode")
            .to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb = img.into_raw();

        // Warm once
        let _ = encode_once(&rgb, w, h, DISTANCE, EFFORT);

        let mut samples = Vec::with_capacity(TIME_ITERS);
        for i in 0..TIME_ITERS {
            let us = encode_once(&rgb, w, h, DISTANCE, EFFORT);
            samples.push(us);
            lines.push(format!("{cell}\t{EFFORT}\t{DISTANCE}\t{i}\t{us}\t{label}"));
        }
        samples.sort_unstable();
        let median = samples[samples.len() / 2];
        lines.push(format!(
            "{cell}\t{EFFORT}\t{DISTANCE}\tMEDIAN\t{median}\t{label}"
        ));
        eprintln!("{label} {cell} {EFFORT} d{DISTANCE}: median {median} us");
    }

    std::fs::write(&out_path, lines.join("\n") + "\n").expect("write");
    eprintln!(
        "wrote {} lines to {} (label={label})",
        lines.len() - 1,
        out_path
    );
}
