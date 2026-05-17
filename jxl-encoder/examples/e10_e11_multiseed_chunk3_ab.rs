//! Paired A/B/C bench for RFC#45 chunk 3: broader seed variance for
//! multi-seed lossless tree learning at e10/e11.
//!
//! Chunk 2 (`d4f2e282`) wired up the multi-seed dispatch but only varied
//! gather start_offset — on 3 CID22 photos the canonical seed always won.
//! Chunk 3 adds split_threshold jitter, property-order rotation, and
//! per-seed stride perturbation to broaden the candidate space.
//!
//! Per-cell A/B/C (lossless):
//!   A = e9  (baseline; libjxl kTortoise, single-seed tree learning)
//!   B = e10 (2-seed pick on the global modular tree)
//!   C = e11 (4-seed pick)
//!
//! Acceptance gate: ≥ 30-40 % of (image, effort >= 10) cells must satisfy
//! `bytes(seed_pick) < bytes(e9)` for chunk 3 to be worth shipping.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel' \
//!     --example e10_e11_multiseed_chunk3_ab \
//!     > benchmarks/e10_e11_multiseed_chunk3_ab_$(date +%Y-%m-%d).tsv
//!
//! Environment:
//!   SAMPLES=2         (default: 2; per-cell sample count, paired)
//!   THREADS=8         (default: 8)
//!   IMAGES="a.png,b.png,c.png"  (default: 5 CID22-512 photos)
//!   CORPUS_DIR        (default: /home/lilith/work/codec-corpus)

use jxl_encoder::api::{LosslessConfig, PixelLayout};
use sha2::Digest;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_IMAGES: &[&str] = &[
    "CID22/CID22-512/validation/1025469.png",
    "CID22/CID22-512/validation/1044329.png",
    "CID22/CID22-512/validation/1189261.png",
    "CID22/CID22-512/validation/1279330.png",
    "CID22/CID22-512/validation/1418519.png",
];

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_images(corpus_dir: &Path) -> Vec<PathBuf> {
    if let Ok(s) = std::env::var("IMAGES") {
        return s
            .split(',')
            .map(|t| corpus_dir.join(t.trim()))
            .collect::<Vec<_>>();
    }
    DEFAULT_IMAGES.iter().map(|p| corpus_dir.join(p)).collect()
}

fn load_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

fn short_sha(bytes: &[u8]) -> String {
    let mut h = sha2::Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    format!("{:02x}{:02x}{:02x}{:02x}", d[0], d[1], d[2], d[3])
}

fn encode_at_effort(
    pixels: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    threads: usize,
) -> (Vec<u8>, std::time::Duration) {
    let cfg = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(threads);
    let t0 = Instant::now();
    let bytes = cfg
        .encode(pixels, w, h, PixelLayout::Rgb8)
        .expect("lossless encode");
    (bytes, t0.elapsed())
}

fn main() {
    let samples = parse_usize("SAMPLES", 2);
    let threads = parse_usize("THREADS", 8);
    let corpus_dir = std::env::var("CORPUS_DIR")
        .unwrap_or_else(|_| "/home/lilith/work/codec-corpus".to_string());
    let corpus_dir = PathBuf::from(corpus_dir);
    let images = parse_images(&corpus_dir);

    // TSV header
    println!(
        "# RFC#45 chunk 3: broader seed variance A/B/C\n\
         # samples={samples}, threads={threads}\n\
         # commit_pending\n\
         image\teffort\tsample\tbytes\tsha\twall_ms"
    );

    let efforts: [u8; 3] = [9, 10, 11];

    for img_path in &images {
        let Some((pixels, w, h)) = load_rgb(img_path) else {
            eprintln!("skip (load failed): {}", img_path.display());
            continue;
        };
        let img_name = img_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());
        eprintln!("encoding {img_name} ({w}×{h}, {} bytes RGB)", pixels.len());

        // Sample-major interleave: (A,B,C) per sample.
        for sample_idx in 0..samples {
            for &effort in &efforts {
                let (bytes, dt) = encode_at_effort(&pixels, w, h, effort, threads);
                println!(
                    "{img_name}\te{effort}\t{sample_idx}\t{}\t{}\t{:.2}",
                    bytes.len(),
                    short_sha(&bytes),
                    dt.as_secs_f64() * 1000.0,
                );
            }
        }
    }
}
