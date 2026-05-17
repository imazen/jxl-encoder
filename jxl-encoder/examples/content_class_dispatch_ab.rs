//! RFC #45 pick #4 chunk 1 — content-class dispatch A/B harness.
//!
//! Per-image A/B for `LossyConfig::with_content_class(Some(class))` against
//! the same encode without the dispatch. The class is computed via
//! `zenanalyze` Tier 1 features (cheap stripe-sampled scan, intentionally
//! kept under the documented < 1 % of encode-time budget).
//!
//! ## Why this example is not registered in `Cargo.toml`
//!
//! `zenanalyze` is an optional sibling-repo path dependency (same gating as
//! `examples/smart_fanout_sweep.rs` and the picker-oracle calibration
//! harnesses). CI does not have it staged, so registering the example
//! would break the no-default-features build. Run locally with:
//!
//! ```ignore
//! # In jxl-encoder/Cargo.toml [dev-dependencies], temporarily add:
//! #   zenanalyze = { path = "../../zenanalyze", features = ["experimental"] }
//! # plus a [[example]] block at the bottom:
//! #   [[example]]
//! #   name = "content_class_dispatch_ab"
//! #   path = "examples/content_class_dispatch_ab.rs"
//! cargo run -p jxl-encoder --release --example content_class_dispatch_ab -- \
//!     ~/work/codec-corpus/CID22/CID22-512/training/1025469.png \
//!     ~/work/codec-corpus/CID22/CID22-512/training/2376805.png \
//!     ~/work/codec-corpus/CID22/CID22-512/training/4220281.png \
//!     ~/work/codec-corpus/gb82-sc/codec_wiki.png \
//!     ~/work/codec-corpus/gb82-sc/imac_g3.png \
//!     ~/work/codec-corpus/gb82-sc/terminal.png
//! ```
//!
//! Then revert the Cargo.toml edits before committing.
//!
//! ## Output format
//!
//! Per (image, effort, distance) row:
//!   `image,effort,distance,class,off_bytes,on_bytes,delta_pct,off_ms,on_ms,classify_ms`

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use jxl_encoder::{ImageContentClass, LossyConfig, PixelLayout};
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: content_class_dispatch_ab <photo_or_screenshot.png> [more.png...]"
        );
        return ExitCode::FAILURE;
    }
    let paths: Vec<PathBuf> = args.iter().map(PathBuf::from).collect();

    println!("# Content-class dispatch A/B (RFC #45 pick #4 chunk 1)");
    println!(
        "# OFF = LossyConfig::with_content_class(None) — libjxl-parity defaults"
    );
    println!(
        "# ON  = LossyConfig::with_content_class(Some(class)) — adapter fires per (class, effort, distance, pixels)"
    );
    println!();
    println!(
        "{:<40} {:>3} {:>5} {:>10} {:>9} {:>9} {:>8} {:>8} {:>8} {:>10}",
        "image",
        "e",
        "dist",
        "class",
        "off_b",
        "on_b",
        "Δ%",
        "off_ms",
        "on_ms",
        "class_ms"
    );
    println!("{}", "-".repeat(130));

    let efforts = [5u8, 6];
    let distances = [1.0f32, 2.0];

    for path in &paths {
        let (rgb, w, h) = match decode_rgb8(path) {
            Some(t) => t,
            None => {
                eprintln!("# skip (decode failed): {}", path.display());
                continue;
            }
        };
        let (class, classify_ms) = classify_zenanalyze(&rgb, w, h);
        for &effort in &efforts {
            for &distance in &distances {
                let (off_bytes, off_ms) =
                    encode_once(&rgb, w, h, effort, distance, None);
                let (on_bytes, on_ms) =
                    encode_once(&rgb, w, h, effort, distance, Some(class));
                let delta_pct = if off_bytes > 0 {
                    (on_bytes as f64 - off_bytes as f64) / off_bytes as f64 * 100.0
                } else {
                    0.0
                };
                println!(
                    "{:<40} {:>3} {:>5.2} {:>10} {:>9} {:>9} {:>+8.2} {:>8.1} {:>8.1} {:>10.2}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    effort,
                    distance,
                    format!("{:?}", class),
                    off_bytes,
                    on_bytes,
                    delta_pct,
                    off_ms,
                    on_ms,
                    classify_ms,
                );
            }
        }
    }
    ExitCode::SUCCESS
}

fn decode_rgb8(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

/// Classify an RGB8 image into a coarse [`ImageContentClass`] using
/// zenanalyze Tier 1 features. The thresholds derive from the calibrated
/// p10 / p50 / p90 ranges documented on `flat_color_block_ratio` and
/// `uniformity` in zenanalyze's `feature.rs`.
///
/// Tier 1 only (sparse stripe scan, ~500k pixel budget) — empirically
/// 1-5 ms on a 1024² photo, well under the documented < 1 % budget for
/// any encode at effort >= 5.
fn classify_zenanalyze(rgb: &[u8], w: u32, h: u32) -> (ImageContentClass, f64) {
    let start = Instant::now();
    // We only need 2 Tier 1 features: flat_color_block_ratio + edge_density.
    // Both are computed in the single stripe-scan pass anyway, so requesting
    // a small set vs the full set costs the same.
    let needed = FeatureSet::new()
        .with(AnalysisFeature::FlatColorBlockRatio)
        .with(AnalysisFeature::EdgeDensity)
        .with(AnalysisFeature::Uniformity)
        .with(AnalysisFeature::ChromaComplexity);
    let q = AnalysisQuery::new(needed);
    let res = analyze_features_rgb8(rgb, w, h, &q);
    let fcbr = res
        .get(AnalysisFeature::FlatColorBlockRatio)
        .and_then(|v| v.as_f32())
        .unwrap_or(0.0);
    let edge = res
        .get(AnalysisFeature::EdgeDensity)
        .and_then(|v| v.as_f32())
        .unwrap_or(0.0);
    let uniformity = res
        .get(AnalysisFeature::Uniformity)
        .and_then(|v| v.as_f32())
        .unwrap_or(0.0);
    let chroma = res
        .get(AnalysisFeature::ChromaComplexity)
        .and_then(|v| v.as_f32())
        .unwrap_or(0.0);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Screenshot dispatch — calibrated against the gb82-sc corpus (10
    // labeled screenshots) and the CID22 photo holdout. High
    // flat_color_block_ratio + high uniformity + low chroma_complexity =
    // screen content. Thresholds picked for high precision (false
    // positives = wasted patches search; the cost-benefit gate inside
    // FindTextLikePatches rejects unprofitable patches sets so a
    // misclassification is bounded).
    let class = if fcbr >= 0.30 && uniformity >= 0.30 && chroma <= 25.0 {
        ImageContentClass::Screenshot
    } else if edge >= 0.05 || chroma >= 15.0 {
        ImageContentClass::Photo
    } else {
        ImageContentClass::Other
    };
    (class, elapsed_ms)
}

fn encode_once(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    class: Option<ImageContentClass>,
) -> (usize, f64) {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_content_class(class);
    let start = Instant::now();
    let bytes = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    (bytes.len(), elapsed)
}
