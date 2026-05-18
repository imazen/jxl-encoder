//! Paired A/B bench for RFC#45 chunk 5: seed-slot split + budget expansion
//! at e11.
//!
//! W8-3-r2 (`ef5c1d11`) honest finding: chunk 4 regressed vs chunk 3 at
//! e11 (+0.39% bytes on 5 CID22-512 photos) because a fixed 4-seed budget
//! cycles through *different* 4 trees rather than *more*. Chunk-3's
//! threshold-jitter / property-rotation / stride perturbations hit better
//! minima on 2/5 images than chunk-4's recombined set.
//!
//! Chunk 5 (this commit) addresses that with two changes:
//!
//!   1. `tree_learn_seeds_for(11)`: 4 → 8 seeds.
//!   2. Seed-slot split:
//!      - seeds 0..=3: chunk-3-only (chunk-4 helpers no-op so
//!        `derive_seeded_sample_fraction` returns None and
//!        `derive_seeded_predictor_order` returns the canonical order).
//!      - seeds 4..=7: chunk-4 dimensions
//!        (`derive_seeded_sample_fraction` cycles through Some(0.40) /
//!        Some(0.60) / Some(0.70) / None and
//!        `derive_seeded_predictor_order` cycles through the four
//!        permutations of `CANDIDATE_PREDICTORS`).
//!
//! Per-cell A/B (lossless):
//!   A = e9  (baseline; libjxl kTortoise, single-seed)
//!   B = e10 (2-seed; unchanged from chunk 3/4)
//!   C = e11 (8-seed; chunk-5 split)
//!
//! Expectation: e10 byte-identical to chunk 3/4 (still 2 seeds, both
//! reserved for chunk-3-only perturbations). e11 strictly ≥ chunk-3
//! wins because seeds 0..=3 cover the chunk-3 candidate space; the
//! additional 4 chunk-4 candidates can only improve (token-cost picker
//! keeps the cheapest).
//!
//! Usage (run from workspace root):
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel parallel-tree-learning' \
//!     --example e10_e11_multiseed_chunk5_ab \
//!     > benchmarks/e10_e11_multiseed_chunk5_ab_$(date +%Y-%m-%d).tsv
//!
//! Note: `jxl-encoder/benchmarks/*.tsv` is gitignored — archive the
//! TSV + .meta sidecar to the workspace-root `benchmarks/` directory.
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

    println!(
        "# RFC#45 chunk 5: seed-slot split + e11 budget expansion A/B/C\n\
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
        eprintln!("encoding {img_name} ({w}x{h}, {} bytes RGB)", pixels.len());

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
