//! Paired A/B/C bench for RFC#45 chunk 4: two extra variance dimensions
//! (sample-fraction jitter + predictor-order shuffle) on top of chunk 3.
//!
//! Chunk 3 (`a8fbd360`) widened the per-seed candidate space with three
//! perturbations (split_threshold jitter, property-order rotation,
//! per-seed stride). Chunk 4 adds:
//!
//!   1. **Sample-fraction jitter** — `tree_sample_fraction ∈ {0.40, 0.60,
//!      0.70}` for seeds 1..=3 (seed 0 keeps the canonical profile
//!      fraction). Wider density-of-sample variation than chunk 3's
//!      `derive_seeded_stride`, which only ±1'd the baseline stride.
//!   2. **Predictor evaluation order shuffle** — per-seed permutation of
//!      `CANDIDATE_PREDICTORS` (4 deterministic orderings). Greedy ID3's
//!      lowest-index tie-break flips on ties, surfacing trees with
//!      different leaf predictors.
//!
//! Per-cell A/B/C (lossless):
//!   A = e9  (baseline; libjxl kTortoise, single-seed tree learning)
//!   B = e10 (2-seed pick on the global modular tree)
//!   C = e11 (4-seed pick)
//!
//! Expectation: chunk 4 improves over chunk 3 because each seed now
//! varies in 5 dimensions instead of 3 — more diverse candidate trees
//! within the same 2 / 4 seed budget.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel parallel-tree-learning' \
//!     --example e10_e11_multiseed_chunk4_ab \
//!     > benchmarks/e10_e11_multiseed_chunk4_ab_$(date +%Y-%m-%d).tsv
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
        "# RFC#45 chunk 4: sample-fraction jitter + predictor-order shuffle A/B/C\n\
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
