//! Per-image RCT sweep for the smart-RCT-picker investigation.
//!
//! For each image in the manifest, encodes lossless at effort 7 with
//! each of the 7 candidate RCT variants forced via
//! `with_force_rct(Some(RctType(N)))`. Records bytes + encode_ms per
//! (image, rct). Also extracts the full zenanalyze SUPPORTED feature set.
//!
//! Output TSV columns:
//!   image_sha, content_class, width, height,
//!   bytes_rct_0, bytes_rct_6, bytes_rct_5, bytes_rct_10,
//!   bytes_rct_26, bytes_rct_40, bytes_rct_12,
//!   best_rct, best_bytes, ms_total,
//!   feat_<name>... (one column per zenanalyze SUPPORTED feature)
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!     --features 'std parallel __expert' \
//!     --example rct_per_image_sweep -- \
//!       --manifest /home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv \
//!       --output benchmarks/rct_per_image_<DATE>.tsv \
//!       [--max-images N] [--max-side N]

use jxl_encoder::RctType;
use jxl_encoder::api::{LosslessConfig, PixelLayout};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};

/// Matches modular/encode.rs RCT_CANDIDATES (effort 7 = 7 candidates).
const RCT_CANDIDATES: &[u8] = &[0, 6, 5, 10, 26, 40, 12];

#[derive(Debug, Clone)]
struct Entry {
    sha256: String,
    split: String,
    content_class: String,
    path: PathBuf,
}

fn parse_manifest(path: &std::path::Path) -> Vec<Entry> {
    let f = BufReader::new(File::open(path).expect("manifest open"));
    let mut out = Vec::new();
    let mut header_parsed = false;
    let mut col_sha = 0;
    let mut col_split = 1;
    let mut col_class = 2;
    let mut col_path = 5;
    for line in f.lines() {
        let line = line.unwrap();
        let cols: Vec<&str> = line.split('\t').collect();
        if !header_parsed {
            for (i, c) in cols.iter().enumerate() {
                match *c {
                    "sha256" => col_sha = i,
                    "split" => col_split = i,
                    "content_class" => col_class = i,
                    "path" => col_path = i,
                    _ => {}
                }
            }
            header_parsed = true;
            continue;
        }
        if cols.len() <= col_path {
            continue;
        }
        out.push(Entry {
            sha256: cols[col_sha].to_string(),
            split: cols[col_split].to_string(),
            content_class: cols[col_class].to_string(),
            path: PathBuf::from(cols[col_path]),
        });
    }
    out
}

fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn maybe_downsize(rgb: &[u8], w: u32, h: u32, max_side: u32) -> (Vec<u8>, u32, u32) {
    if max_side == 0 || w.max(h) <= max_side {
        return (rgb.to_vec(), w, h);
    }
    let scale = max_side as f32 / w.max(h) as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(w, h, rgb.to_vec())
        .expect("rgb8 buf");
    let resized = image::imageops::resize(&buf, nw, nh, image::imageops::FilterType::Lanczos3);
    (resized.into_raw(), nw, nh)
}

fn encode_with_forced_rct(rgb: &[u8], w: u32, h: u32, rct: RctType) -> Option<(usize, f64)> {
    let cfg = LosslessConfig::new()
        .with_effort(7)
        .with_force_rct(Some(rct))
        .with_threads(1);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes.len(), ms))
}

fn feature_columns() -> Vec<AnalysisFeature> {
    FeatureSet::SUPPORTED.iter().collect()
}

fn feature_value_str(r: &zenanalyze::feature::AnalysisResults, f: AnalysisFeature) -> String {
    match r.get(f) {
        Some(FeatureValue::F32(v)) => format!("{:.6}", v),
        Some(FeatureValue::U32(v)) => format!("{}", v),
        Some(FeatureValue::Bool(v)) => format!("{}", v as u8),
        Some(_) => "nan".to_string(),
        None => "nan".to_string(),
    }
}

struct Args {
    manifest: PathBuf,
    output: PathBuf,
    max_images: usize,
    max_side: u32,
}

fn parse_args() -> Args {
    let mut manifest =
        PathBuf::from("/home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv");
    let mut output = PathBuf::from("benchmarks/rct_per_image_2026-05-17.tsv");
    let mut max_images = usize::MAX;
    let mut max_side = 0u32;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--manifest" => {
                manifest = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--output" => {
                output = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--max-images" => {
                max_images = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--max-side" => {
                max_side = argv[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {}", other);
                i += 1;
            }
        }
    }
    Args {
        manifest,
        output,
        max_images,
        max_side,
    }
}

fn main() {
    let args = parse_args();
    let entries: Vec<Entry> = parse_manifest(&args.manifest)
        .into_iter()
        .take(args.max_images)
        .collect();
    eprintln!(
        "Sweeping {} images, output={}",
        entries.len(),
        args.output.display()
    );

    let cols = feature_columns();
    let out = Mutex::new(File::create(&args.output).expect("create output"));
    {
        let mut f = out.lock().unwrap();
        write!(f, "image_sha\tsplit\tcontent_class\twidth\theight").ok();
        for r in RCT_CANDIDATES {
            write!(f, "\tbytes_rct_{}", r).ok();
        }
        write!(
            f,
            "\tbest_rct\tbest_bytes\tworst_bytes\tymin_over_ymax\ttotal_ms"
        )
        .ok();
        for c in &cols {
            write!(f, "\tfeat_{}", c.name()).ok();
        }
        writeln!(f).ok();
    }

    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let started = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);

    entries.par_iter().for_each(|entry| {
        let Some((rgb_native, w_native, h_native)) = load_png(&entry.path) else {
            eprintln!("  skip (load failed): {}", entry.path.display());
            return;
        };
        let (rgb_owned, w, h) = maybe_downsize(&rgb_native, w_native, h_native, args.max_side);
        let rgb = rgb_owned.as_slice();

        // Encode with each forced RCT.
        let mut bytes_per_rct: Vec<Option<(usize, f64)>> = Vec::with_capacity(RCT_CANDIDATES.len());
        let t0 = Instant::now();
        for &r in RCT_CANDIDATES {
            bytes_per_rct.push(encode_with_forced_rct(rgb, w, h, RctType(r)));
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Identify best (smallest) RCT among successful trials.
        let mut best_idx = 0usize;
        let mut best_bytes = usize::MAX;
        let mut worst_bytes = 0usize;
        for (i, b) in bytes_per_rct.iter().enumerate() {
            if let Some((nb, _)) = b {
                if *nb < best_bytes {
                    best_bytes = *nb;
                    best_idx = i;
                }
                if *nb > worst_bytes {
                    worst_bytes = *nb;
                }
            }
        }
        let best_rct = RCT_CANDIDATES[best_idx];
        let ratio = if worst_bytes > 0 {
            best_bytes as f64 / worst_bytes as f64
        } else {
            1.0
        };

        // Extract features
        let analysis = analyze_features_rgb8(rgb, w, h, &query);

        {
            let mut f = out.lock().unwrap();
            write!(
                f,
                "{}\t{}\t{}\t{}\t{}",
                entry.sha256, entry.split, entry.content_class, w, h
            )
            .ok();
            for b in &bytes_per_rct {
                match b {
                    Some((n, _)) => write!(f, "\t{}", n).ok(),
                    None => write!(f, "\tnan").ok(),
                };
            }
            write!(
                f,
                "\t{}\t{}\t{}\t{:.4}\t{:.1}",
                best_rct, best_bytes, worst_bytes, ratio, total_ms
            )
            .ok();
            for c in &cols {
                write!(f, "\t{}", feature_value_str(&analysis, *c)).ok();
            }
            writeln!(f).ok();
            f.flush().ok();
        }
        let n = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        eprintln!(
            "[{}/{}] {} {}x{} best_rct={} ratio={:.3} {:.0}ms",
            n,
            entries.len(),
            &entry.sha256[..10],
            w,
            h,
            best_rct,
            ratio,
            total_ms
        );
    });

    eprintln!(
        "Done in {:.1}s. Output: {}",
        started.elapsed().as_secs_f64(),
        args.output.display()
    );
}
