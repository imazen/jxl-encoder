//! Pareto calibration sweep for the zenjxl lossless picker (issue #24).
//!
//! Sweeps four axes per the global benchmark-discipline rule:
//!   - **size**: tiny (64), small (256), medium (1024), large (native)
//!   - **config**: effort × {squeeze, tree_learning, patches} bool axes
//!     × LZ77 method {Rle, Greedy, Optimal at e>=7}
//!   - **content**: photo + screenshot + mixed via the picker-train corpus
//!
//! Per (image, size, config) we emit one TSV row capturing
//! `bytes + encode_ms`. Per-image zenanalyze features are captured separately.
//!
//! v1 limitation: only public LosslessConfig knobs are swept. Underlying
//! `nb_rcts_to_try`, `wp_num_param_sets`, `tree_max_buckets`,
//! `tree_num_properties`, `tree_sample_fraction` are bundled into the
//! effort knob. Decoupling those requires API additions tracked in #24.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!     --features 'std parallel' \
//!     --example lossless_pareto_calibrate -- \
//!       --manifest /home/lilith/work/codec-corpus/picker-train/manifest.tsv \
//!       --split train \
//!       --output benchmarks/lossless_pareto_<DATE>.tsv \
//!       --features-output benchmarks/lossless_pareto_features_<DATE>.tsv \
//!       [--max-images N] [--sizes 64,256,1024,native] [--threads N]
//!       [--features-only] [--smoke]
//!
//! Output is appended row-by-row so a partial run is still useful.

use jxl_encoder::api::{LosslessConfig, Lz77Method, PixelLayout};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

// ---------------------------------------------------------------------
// Sweep grid
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct ConfigSpec {
    id: u32,
    name: &'static str,
    effort: u8,
    squeeze: bool,
    tree_learning: bool,
    patches: bool,
    lz77_method: Option<Lz77Method>, // None = use default-by-effort
}

const fn cfg(id: u32, name: &'static str, effort: u8, sq: bool, tl: bool, pa: bool, lz: Option<Lz77Method>) -> ConfigSpec {
    ConfigSpec { id, name, effort, squeeze: sq, tree_learning: tl, patches: pa, lz77_method: lz }
}

/// 76 configs covering the picker's outer search space at public-API
/// granularity. Inner knobs (nb_rcts_to_try, wp_num_param_sets,
/// tree_max_buckets/num_properties/sample_fraction) are bundled into
/// `effort` for v1. Two-stage sweep follows once underlying knobs are
/// exposed.
const CONFIGS: &[ConfigSpec] = &[
    // === Effort 5 (Hare) — defaults at this effort have squeeze off,
    // tree_learning off, patches on, lz77=Rle. Vary squeeze and patches.
    cfg(0, "e5_default",                            5, false, false, true,  None),
    cfg(1, "e5_no_patches",                         5, false, false, false, None),
    cfg(2, "e5_squeeze",                            5, true,  false, true,  None),
    cfg(3, "e5_squeeze_no_patches",                 5, true,  false, false, None),

    // === Effort 7 (Squirrel) — defaults: tree_learning on, patches on, lz77=Rle.
    // Varying squeeze + tree_learning + patches + lz77 method.
    cfg(10, "e7_default",                           7, false, true,  true,  None),
    cfg(11, "e7_lz77_greedy",                       7, false, true,  true,  Some(Lz77Method::Greedy)),
    cfg(12, "e7_lz77_optimal",                      7, false, true,  true,  Some(Lz77Method::Optimal)),
    cfg(13, "e7_no_tree",                           7, false, false, true,  None),
    cfg(14, "e7_no_patches",                        7, false, true,  false, None),
    cfg(15, "e7_no_tree_no_patches",                7, false, false, false, None),
    cfg(16, "e7_squeeze",                           7, true,  true,  true,  None),
    cfg(17, "e7_squeeze_lz77_greedy",               7, true,  true,  true,  Some(Lz77Method::Greedy)),
    cfg(18, "e7_squeeze_lz77_optimal",              7, true,  true,  true,  Some(Lz77Method::Optimal)),
    cfg(19, "e7_squeeze_no_tree",                   7, true,  false, true,  None),
    cfg(20, "e7_squeeze_no_patches",                7, true,  true,  false, None),

    // === Effort 8 (Kitten) — defaults: tree_learning on, patches on, lz77=Greedy.
    cfg(30, "e8_default",                           8, false, true,  true,  None),
    cfg(31, "e8_lz77_rle",                          8, false, true,  true,  Some(Lz77Method::Rle)),
    cfg(32, "e8_lz77_optimal",                      8, false, true,  true,  Some(Lz77Method::Optimal)),
    cfg(33, "e8_no_tree",                           8, false, false, true,  None),
    cfg(34, "e8_no_patches",                        8, false, true,  false, None),
    cfg(35, "e8_no_tree_no_patches",                8, false, false, false, None),
    cfg(36, "e8_squeeze",                           8, true,  true,  true,  None),
    cfg(37, "e8_squeeze_lz77_rle",                  8, true,  true,  true,  Some(Lz77Method::Rle)),
    cfg(38, "e8_squeeze_lz77_optimal",              8, true,  true,  true,  Some(Lz77Method::Optimal)),
    cfg(39, "e8_squeeze_no_tree",                   8, true,  false, true,  None),
    cfg(40, "e8_squeeze_no_patches",                8, true,  true,  false, None),

    // === Effort 9 (Tortoise) — defaults: tree_learning on, patches on, lz77=Optimal.
    cfg(50, "e9_default",                           9, false, true,  true,  None),
    cfg(51, "e9_lz77_rle",                          9, false, true,  true,  Some(Lz77Method::Rle)),
    cfg(52, "e9_lz77_greedy",                       9, false, true,  true,  Some(Lz77Method::Greedy)),
    cfg(53, "e9_no_tree",                           9, false, false, true,  None),
    cfg(54, "e9_no_patches",                        9, false, true,  false, None),
    cfg(55, "e9_no_tree_no_patches",                9, false, false, false, None),
    cfg(56, "e9_squeeze",                           9, true,  true,  true,  None),
    cfg(57, "e9_squeeze_lz77_rle",                  9, true,  true,  true,  Some(Lz77Method::Rle)),
    cfg(58, "e9_squeeze_lz77_greedy",               9, true,  true,  true,  Some(Lz77Method::Greedy)),
    cfg(59, "e9_squeeze_no_tree",                   9, true,  false, true,  None),
    cfg(60, "e9_squeeze_no_patches",                9, true,  true,  false, None),

    // === Anchors: e3 (fast baseline), e10 (max effort)
    cfg(70, "e3_default",                           3, false, false, false, None),
    cfg(71, "e10_default",                         10, false, true,  true,  None),
];

/// Default size axis. `0` means "use native dimensions" (large).
const DEFAULT_SIZES: &[u32] = &[64, 256, 1024, 0];

// ---------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------

struct Args {
    manifest: PathBuf,
    split: String,
    sizes: Vec<u32>,
    output: PathBuf,
    features_output: PathBuf,
    max_images: usize,
    threads: usize,
    features_only: bool,
    smoke: bool,
}

fn parse_args() -> Args {
    let mut manifest = PathBuf::from("/home/lilith/work/codec-corpus/picker-train/manifest.tsv");
    let mut split = "train".to_string();
    let mut sizes: Vec<u32> = Vec::new();
    let mut max_images = usize::MAX;
    let mut threads = 0;
    let mut features_only = false;
    let mut smoke = false;
    let date = chrono_today();
    let mut output = PathBuf::from(format!("benchmarks/lossless_pareto_{date}.tsv"));
    let mut features_output =
        PathBuf::from(format!("benchmarks/lossless_pareto_features_{date}.tsv"));

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--manifest" => manifest = PathBuf::from(it.next().unwrap()),
            "--split" => split = it.next().unwrap(),
            "--sizes" => {
                let s = it.next().unwrap();
                for tok in s.split(',') {
                    if tok == "native" {
                        sizes.push(0);
                    } else {
                        sizes.push(tok.parse().expect("size must be uint or 'native'"));
                    }
                }
            }
            "--output" => output = PathBuf::from(it.next().unwrap()),
            "--features-output" => features_output = PathBuf::from(it.next().unwrap()),
            "--max-images" => max_images = it.next().unwrap().parse().expect("max-images uint"),
            "--threads" => threads = it.next().unwrap().parse().expect("threads uint"),
            "--features-only" => features_only = true,
            "--smoke" => {
                smoke = true;
                max_images = max_images.min(5);
                if sizes.is_empty() {
                    sizes = vec![256];
                }
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if sizes.is_empty() {
        sizes = DEFAULT_SIZES.to_vec();
    }
    Args {
        manifest,
        split,
        sizes,
        output,
        features_output,
        max_images,
        threads,
        features_only,
        smoke,
    }
}

fn chrono_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = (secs / 86400) as i64;
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ---------------------------------------------------------------------
// Manifest loading
// ---------------------------------------------------------------------

#[derive(Clone)]
struct ManifestEntry {
    sha256: String,
    split: String,
    content_class: String,
    path: PathBuf,
}

fn load_manifest(path: &std::path::Path, split_filter: &str) -> Vec<ManifestEntry> {
    let txt = std::fs::read_to_string(path).expect("read manifest");
    let mut out = Vec::new();
    for (i, line) in txt.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 6 {
            continue;
        }
        if !split_filter.is_empty() && cols[1] != split_filter {
            continue;
        }
        out.push(ManifestEntry {
            sha256: cols[0].to_string(),
            split: cols[1].to_string(),
            content_class: cols[2].to_string(),
            path: PathBuf::from(cols[5]),
        });
    }
    out
}

// ---------------------------------------------------------------------
// Image loading + resize
// ---------------------------------------------------------------------

fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = match image::open(path) {
        Ok(i) => i,
        Err(_) => return None,
    };
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn resize_to(rgb: &[u8], w: u32, h: u32, target_max: u32) -> (Vec<u8>, u32, u32) {
    if target_max == 0 || (w.max(h) <= target_max) {
        return (rgb.to_vec(), w, h);
    }
    let scale = target_max as f32 / w.max(h) as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(w, h, rgb.to_vec())
        .expect("rgb8 buffer");
    let resized =
        image::imageops::resize(&buf, new_w, new_h, image::imageops::FilterType::Lanczos3);
    (resized.into_raw(), new_w, new_h)
}

// ---------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------

fn build_encoder(spec: ConfigSpec) -> LosslessConfig {
    let mut cfg = LosslessConfig::new()
        .with_effort(spec.effort)
        .with_squeeze(spec.squeeze)
        .with_tree_learning(spec.tree_learning)
        .with_patches(spec.patches)
        .with_threads(1); // outer rayon parallelism — disable inner
    if let Some(method) = spec.lz77_method {
        cfg = cfg.with_lz77(true).with_lz77_method(method);
    }
    cfg
}

fn encode_one(rgb: &[u8], w: u32, h: u32, spec: ConfigSpec) -> Option<(usize, f64)> {
    let cfg = build_encoder(spec);
    let start = Instant::now();
    let bytes = match cfg.encode(rgb, w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(_) => return None,
    };
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes.len(), encode_ms))
}

// ---------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------

fn feature_columns() -> Vec<AnalysisFeature> {
    FeatureSet::SUPPORTED.iter().collect()
}

fn feature_value_str(
    analysis: &zenanalyze::feature::AnalysisResults,
    f: AnalysisFeature,
) -> String {
    if let Some(v) = analysis.get_f32(f) {
        format!("{v:.6}")
    } else if let Some(v) = analysis.get(f) {
        match v {
            zenanalyze::feature::FeatureValue::F32(x) => format!("{x:.6}"),
            zenanalyze::feature::FeatureValue::U32(x) => format!("{x}"),
            zenanalyze::feature::FeatureValue::Bool(b) => format!("{}", b as u8),
            _ => String::new(),
        }
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

fn main() {
    let args = parse_args();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let entries = load_manifest(&args.manifest, &args.split);
    let n_images = entries.len().min(args.max_images);
    let entries: Vec<ManifestEntry> = entries.into_iter().take(n_images).collect();

    let cells = entries.len() * args.sizes.len() * CONFIGS.len();
    eprintln!(
        "[lossless_pareto_calibrate] {} images × {} sizes × {} configs = {} encodes ({})",
        entries.len(),
        args.sizes.len(),
        CONFIGS.len(),
        cells,
        if args.features_only { "features-only" } else { "full sweep" },
    );
    eprintln!("[lossless_pareto_calibrate] manifest: {} ({})", args.manifest.display(), args.split);
    eprintln!("[lossless_pareto_calibrate] output:   {}", args.output.display());
    eprintln!("[lossless_pareto_calibrate] features: {}", args.features_output.display());

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let main_file: Option<Mutex<std::fs::File>> = if args.features_only {
        None
    } else {
        let is_new = !args.output.exists();
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output)
            .expect("open output");
        let f = Mutex::new(f);
        if is_new {
            let mut g = f.lock().unwrap();
            writeln!(
                g,
                "image_sha\tsplit\tcontent_class\tsize_class\twidth\theight\tconfig_id\tconfig_name\teffort\tsqueeze\ttree_learning\tpatches\tlz77_method\tbytes\tencode_ms"
            )
            .ok();
        }
        Some(f)
    };

    if let Some(parent) = args.features_output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let feat_is_new = !args.features_output.exists();
    let feat_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.features_output)
        .expect("open features output");
    let feat_file = Mutex::new(feat_file);
    let cols = feature_columns();
    if feat_is_new {
        let mut f = feat_file.lock().unwrap();
        write!(f, "image_sha\tsplit\tcontent_class\tsize_class\twidth\theight").ok();
        for c in &cols {
            write!(f, "\tfeat_{}", c.name()).ok();
        }
        writeln!(f).ok();
    }

    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let started = Instant::now();
    let unit_count = entries.len() * args.sizes.len();
    let done = std::sync::atomic::AtomicUsize::new(0);

    let work_units: Vec<(ManifestEntry, u32)> = entries
        .iter()
        .flat_map(|e| args.sizes.iter().map(move |&sz| (e.clone(), sz)))
        .collect();

    work_units.par_iter().for_each(|(entry, target_size)| {
        let target_size = *target_size;
        let (rgb_native, w_native, h_native) = match load_png(&entry.path) {
            Some(t) => t,
            None => {
                eprintln!("  skip (load failed): {}", entry.path.display());
                return;
            }
        };
        let (rgb, w, h) = resize_to(&rgb_native, w_native, h_native, target_size);
        let size_class = match target_size {
            64 => "tiny",
            256 => "small",
            1024 => "medium",
            0 => "large",
            _ => "custom",
        };

        // Per-(image, size) features.
        let analysis = analyze_features_rgb8(&rgb, w, h, &query);
        {
            let mut f = feat_file.lock().unwrap();
            write!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}",
                entry.sha256, entry.split, entry.content_class, size_class, w, h
            )
            .ok();
            for c in &cols {
                write!(f, "\t{}", feature_value_str(&analysis, *c)).ok();
            }
            writeln!(f).ok();
            f.flush().ok();
        }

        if let Some(main_file) = main_file.as_ref() {
            for spec in CONFIGS {
                let row = encode_one(&rgb, w, h, *spec);
                let lz77_str = match spec.lz77_method {
                    None => "default",
                    Some(Lz77Method::Rle) => "rle",
                    Some(Lz77Method::Greedy) => "greedy",
                    Some(Lz77Method::Optimal) => "optimal",
                };
                let mut f = main_file.lock().unwrap();
                match row {
                    Some((bytes, encode_ms)) => {
                        writeln!(
                            f,
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}",
                            entry.sha256,
                            entry.split,
                            entry.content_class,
                            size_class,
                            w,
                            h,
                            spec.id,
                            spec.name,
                            spec.effort,
                            spec.squeeze as u8,
                            spec.tree_learning as u8,
                            spec.patches as u8,
                            lz77_str,
                            bytes,
                            encode_ms,
                        )
                        .ok();
                    }
                    None => {
                        writeln!(
                            f,
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\t",
                            entry.sha256,
                            entry.split,
                            entry.content_class,
                            size_class,
                            w,
                            h,
                            spec.id,
                            spec.name,
                            spec.effort,
                            spec.squeeze as u8,
                            spec.tree_learning as u8,
                            spec.patches as u8,
                            lz77_str,
                        )
                        .ok();
                    }
                }
            }
            main_file.lock().unwrap().flush().ok();
        }

        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n % 4 == 0 || n == unit_count {
            let dt = started.elapsed().as_secs_f64();
            let rate = n as f64 / dt;
            let eta = (unit_count - n) as f64 / rate;
            eprintln!(
                "  progress: {}/{}  ({:.2}/sec, ETA {:.0}s = {:.1}h)",
                n,
                unit_count,
                rate,
                eta,
                eta / 3600.0,
            );
        }
    });

    eprintln!(
        "[lossless_pareto_calibrate] done in {:.0}s ({:.2}h){}",
        started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64() / 3600.0,
        if args.smoke { " [smoke]" } else { "" },
    );
}
