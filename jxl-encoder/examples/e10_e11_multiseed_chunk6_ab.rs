//! Paired A/B bench for RFC#45 chunk 6: seed budget 8 → 16 at e11 with two
//! new variance dimensions (split-bucket-count + properties-slice
//! truncation).
//!
//! Chunk 5 (`2b2ce912`) raised e11 from 4 → 8 seeds and split chunk-3
//! perturbations (seeds 0..=3) from chunk-4 dimensions (seeds 4..=7),
//! producing −0.46% bytes vs chunk 4 / strict win over chunk 3 on the
//! 5-image CID22-512 paired bench.
//!
//! Chunk 6 (this commit) extends the same seed-slot pattern to two new
//! variance dimensions:
//!
//! 1. `tree_learn_seeds_for(11)`: 8 → 16 seeds.
//! 2. Seed-slot layout:
//!    - seeds 0..=3 chunk-3 perturbations (split_threshold jitter,
//!      property-order rotation, per-seed stride).
//!    - seeds 4..=7 chunk-4 dimensions on top of chunk-3
//!      (sample-fraction override + predictor-order shuffle).
//!    - seeds 8..=11 chunk-6 dim A — `max_property_values` ∈
//!      {64, 128, 192, canonical 256}. Coarser bucket grids in
//!      `find_best_split`'s value quantization can land on different
//!      (and sometimes cheaper) discrete thresholds than the
//!      256-bucket grid.
//!    - seeds 12..=15 chunk-6 dim B — `properties` slice truncation ∈
//!      {8, 10, 12, canonical 14+}. Forces the greedy ID3 builder to
//!      pick among fewer properties — a structural-regularization
//!      fallback that can outperform the full-property tree when
//!      canonical over-fits late-tier properties (e.g., the
//!      WPMaxError family at indices 10-15 chasing bucket noise on
//!      smooth content).
//!
//! Why these two dimensions (and not max_tree_depth / property_gain
//! threshold per the original chunk-6 brief): TreeLearningParams has no
//! `max_tree_depth` knob — the greedy builder grows until no split
//! clears `split_threshold` or `max_nodes` is hit. The acceptance-
//! threshold dimension is already exercised by chunk-3's
//! split_threshold-jitter. `max_property_values` (split granularity) and
//! `properties.len()` (set size) are the next two structurally
//! orthogonal knobs that change which trees the greedy builder can
//! reach without recombining chunk-3's existing perturbations.
//!
//! Per-cell A/B (lossless):
//!   A = e9  (baseline; libjxl kTortoise, single-seed)
//!   B = e10 (2-seed; unchanged from chunks 3-5)
//!   C = e11 (16-seed; chunk-6 split)
//!
//! Expectation: e10 byte-identical to chunks 3-5 (still 2 seeds, both
//! reserved for chunk-3-only perturbations). e11 strictly ≥ chunk-5
//! wins because seeds 0..=7 cover the same chunk-3/4/5 candidate space;
//! the additional 8 chunk-6 candidates can only improve (token-cost
//! picker keeps the cheapest).
//!
//! Usage (run from workspace root):
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel parallel-tree-learning' \
//!     --example e10_e11_multiseed_chunk6_ab \
//!     > benchmarks/e10_e11_multiseed_chunk6_ab_$(date +%Y-%m-%d).tsv
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
        "# RFC#45 chunk 6: seed-budget 8 → 16 + 2 new dimensions A/B/C\n\
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
