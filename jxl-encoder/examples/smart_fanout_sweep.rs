//! Smart tree-fanout dispatch investigation (chunk 1 of issue #42-style work).
//!
//! Hypothesis: optimal `tree_parallel_max_depth` + `tree_parallel_floor` depend
//! on image content (tree size + parallelism efficiency), not just bundled effort.
//! Per-effort constants leave wall-clock on the table on the "wrong" image.
//!
//! Method:
//!   For each (image, effort, (depth, floor)) cell — encode N times, take `min`.
//!   Then per image: extract zenanalyze features (Tier 1 + Tier 3) and correlate
//!   with the optimal-fanout pick.
//!
//! Output: TSV with columns:
//!   image, effort, depth, floor, threads, iter, time_ms, bytes, mp
//! Plus a features TSV: image, mp, ${FEATURE_NAME}*
//!
//! Cells (default grid):
//!   - depth ∈ {3, 4, 5, 6}
//!   - floor ∈ {4096, 8192, 16384, 32768}
//! → 16 cells, plus baseline (effort default).
//!
//! Usage:
//!   cargo run --release --no-default-features \
//!       --features 'std parallel-tree-learning __expert' \
//!       --example smart_fanout_sweep -- \
//!       --output /tmp/smart_fanout_sweep.tsv \
//!       --features-output /tmp/smart_fanout_features.tsv
//!
//! Env:
//!   SAMPLES=5        (samples per cell — take min)
//!   THREADS=8        (rayon threads)
//!   EFFORTS=7,8,9

use jxl_encoder::api::{LosslessConfig, PixelLayout};
use jxl_encoder::LosslessInternalParams;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

// ─────────────────────────────────────────────────────────────────────
// Image set: photos (CID22 small, CLIC medium / large), screenshots
// (gb82-sc), and a flat / gradient outlier.
// ─────────────────────────────────────────────────────────────────────
const IMAGES: &[(&str, &str, &str)] = &[
    // (label, content_class, path)
    // 3 profile images per CLAUDE.md e8/e9 cliff plan
    (
        "small_0.26MP",
        "photo",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png",
    ),
    (
        "medium_1.05MP",
        "photo",
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    ),
    (
        "large_4.19MP",
        "photo",
        "/home/lilith/work/codec-corpus/clic2025/final-test/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png",
    ),
    // Additional 1024×1024 photos for the medium-range correlation fit.
    (
        "photo_clic_a10ae",
        "photo",
        "/home/lilith/work/codec-corpus/clic2025-1024/a10ae819a2d24873079c93346200cab3.png",
    ),
    (
        "photo_clic_2a242",
        "photo",
        "/home/lilith/work/codec-corpus/clic2025-1024/2a2420f929cd47122eae364a2bc27710.png",
    ),
    (
        "photo_clic_afe36",
        "photo",
        "/home/lilith/work/codec-corpus/clic2025-1024/afe3676bbfcf7183d697ff7b5d5bd45b.png",
    ),
    // Screenshots — patches+text+UI content. Native sizes (large + small).
    (
        "screen_codec_wiki",
        "screen",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
    ),
    (
        "screen_terminal",
        "screen",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
    ),
];

// (depth, floor) cells to sweep — keep root_threshold at effort default.
// Drop depth=3 (rarely wins on real images) and {8192, 32768} floor cells
// per smoke test — keep {4, 5, 6} × {4096, 16384} = 6 cells.
const DEPTH_GRID: &[u32] = &[4, 5, 6];
const FLOOR_GRID: &[usize] = &[4_096, 16_384];

fn parse_args() -> (PathBuf, PathBuf) {
    let mut out = PathBuf::from("/tmp/smart_fanout_sweep.tsv");
    let mut feats = PathBuf::from("/tmp/smart_fanout_features.tsv");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--features-output" => {
                feats = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            _ => i += 1,
        }
    }
    (out, feats)
}

fn parse_efforts() -> Vec<u8> {
    std::env::var("EFFORTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u8>().ok())
                .collect()
        })
        .filter(|v: &Vec<u8>| !v.is_empty())
        .unwrap_or_else(|| vec![7, 8, 9])
}

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn load_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

fn encode_with_fanout(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    depth: u32,
    floor: usize,
    threads: usize,
) -> (usize, f64) {
    let mut params = LosslessInternalParams::default();
    params.tree_parallel_max_depth = Some(depth);
    params.tree_parallel_floor = Some(floor);
    let cfg = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(threads)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes.len(), ms)
}

fn feature_query() -> AnalysisQuery {
    // Cheap-but-discriminative set:
    //   Tier 1: Variance, EdgeDensity, ChromaComplexity, Uniformity,
    //           FlatColorBlockRatio, CbSharpness, CrSharpness
    //   Tier 1 experimental: LaplacianVariance, VarianceSpread, Colourfulness
    //   Tier 3: HighFreqEnergyRatio, LumaHistogramEntropy
    //   Tier 3 experimental: DctCompressibilityY, DctCompressibilityUV,
    //                        PatchFraction, GradientFraction
    //   Palette: DistinctColorBins
    let s = FeatureSet::new()
        .with(AnalysisFeature::Variance)
        .with(AnalysisFeature::EdgeDensity)
        .with(AnalysisFeature::ChromaComplexity)
        .with(AnalysisFeature::Uniformity)
        .with(AnalysisFeature::FlatColorBlockRatio)
        .with(AnalysisFeature::CbSharpness)
        .with(AnalysisFeature::CrSharpness)
        .with(AnalysisFeature::HighFreqEnergyRatio)
        .with(AnalysisFeature::LumaHistogramEntropy)
        .with(AnalysisFeature::DistinctColorBins)
        .with(AnalysisFeature::LaplacianVariance)
        .with(AnalysisFeature::VarianceSpread)
        .with(AnalysisFeature::Colourfulness)
        .with(AnalysisFeature::DctCompressibilityY)
        .with(AnalysisFeature::DctCompressibilityUV)
        .with(AnalysisFeature::PatchFraction)
        .with(AnalysisFeature::GradientFraction);
    AnalysisQuery::new(s)
}

fn feature_value_str(results: &zenanalyze::feature::AnalysisResults, f: AnalysisFeature) -> String {
    if let Some(v) = results.get_f32(f) {
        format!("{v:.6}")
    } else if let Some(v) = results.get(f) {
        match v {
            zenanalyze::feature::FeatureValue::F32(x) => format!("{x:.6}"),
            zenanalyze::feature::FeatureValue::U32(x) => format!("{x}"),
            zenanalyze::feature::FeatureValue::U64(x) => format!("{x}"),
            zenanalyze::feature::FeatureValue::Bool(b) => format!("{}", b as u8),
            _ => String::new(),
        }
    } else {
        String::new()
    }
}

fn refresh_marker(activity: &str) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| {
            let s = d.as_secs();
            let (h, m, sec) = ((s / 3600) % 24, (s / 60) % 60, s % 60);
            // Crude UTC ISO-8601 — good enough for marker freshness.
            let days_since_epoch = s / 86400;
            // Days→Y/M/D via a 1970 epoch math (approximate; only the
            // timestamp granularity matters for the marker contract).
            let _ = days_since_epoch;
            format!("1970-01-01T{h:02}:{m:02}:{sec:02}Z+{days_since_epoch}d")
        })
        .unwrap_or_else(|_| "?".into());
    let _ = std::fs::write(
        "/home/lilith/work/zen/jxl-encoder/.workongoing",
        format!("{ts} claude-zenanalyze-tree-pred sweep: {activity}\n"),
    );
}

fn main() {
    let (out_path, feats_path) = parse_args();
    let samples = parse_usize("SAMPLES", 5);
    let threads = parse_usize("THREADS", 8);
    let efforts = parse_efforts();

    // Capture feature column order once.
    let feature_cols: Vec<AnalysisFeature> = feature_query().features().into_iter().collect();

    eprintln!(
        "# smart_fanout_sweep: {} images × {} efforts × {} depths × {} floors\n# samples/cell={} threads={} efforts={:?}",
        IMAGES.len(),
        efforts.len(),
        DEPTH_GRID.len(),
        FLOOR_GRID.len(),
        samples,
        threads,
        efforts,
    );

    // Open output files.
    let mut out = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&out_path)
        .expect("open output");
    writeln!(
        out,
        "image\tclass\teffort\tdepth\tfloor\tthreads\titer\ttime_ms\tbytes\tmp"
    )
    .unwrap();

    let mut feats = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&feats_path)
        .expect("open features output");
    let mut feat_header = String::from("image\tclass\twidth\theight\tmp");
    for f in &feature_cols {
        feat_header.push('\t');
        feat_header.push_str(f.name());
    }
    writeln!(feats, "{feat_header}").unwrap();

    for (label, class, path) in IMAGES {
        eprintln!("# loading {label} ({path})");
        let (rgb, w, h) = match load_rgb(path) {
            Some(t) => t,
            None => {
                eprintln!("# SKIP {label}: load failed");
                continue;
            }
        };
        let mp = (w as f64) * (h as f64) / 1_000_000.0;
        refresh_marker(&format!("analyze {label}"));

        // --- Per-image feature extraction (one pass) ---
        let analysis = analyze_features_rgb8(&rgb, w, h, &feature_query());
        let mut feat_row = format!("{label}\t{class}\t{w}\t{h}\t{mp:.4}");
        for f in &feature_cols {
            feat_row.push('\t');
            feat_row.push_str(&feature_value_str(&analysis, *f));
        }
        writeln!(feats, "{feat_row}").unwrap();
        feats.flush().unwrap();

        // --- Per-(effort, depth, floor) timing ---
        for &effort in &efforts {
            // Warmup once per (image, effort) to settle FS / shared lib cache.
            let _ = encode_with_fanout(&rgb, w, h, effort, 4, 16_384, threads);

            for &depth in DEPTH_GRID {
                for &floor in FLOOR_GRID {
                    refresh_marker(&format!("{label} e{effort} d{depth} f{floor}"));
                    for iter in 1..=samples {
                        let (bytes, ms) =
                            encode_with_fanout(&rgb, w, h, effort, depth, floor, threads);
                        writeln!(
                            out,
                            "{label}\t{class}\t{effort}\t{depth}\t{floor}\t{threads}\t{iter}\t{ms:.3}\t{bytes}\t{mp:.4}"
                        )
                        .unwrap();
                        out.flush().unwrap();
                    }
                }
            }
        }
    }
    eprintln!(
        "# done. output={} features={}",
        out_path.display(),
        feats_path.display()
    );
}
