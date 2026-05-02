//! Pareto calibration sweep for the zenjxl lossless picker (issue #24).
//!
//! Sweeps INDEPENDENT internal knobs via `LosslessConfig::with_internal_params`.
//! The picker is going to *replace* the bundled-effort axis, so the oracle
//! must expose each underlying knob independently — not the bundled effort.
//!
//! Cells (categorical, 16):
//!   - lz77_method ∈ {None, Rle, Greedy, Optimal}    (4)
//!   - use_squeeze ∈ {false, true}                   (2)
//!   - use_patches ∈ {false, true}                   (2)
//!
//! Per-cell scalar samples (continuous, dense random):
//!   - nb_rcts_to_try         ∈ {0, 4, 7, 9, 19}
//!   - wp_num_param_sets      ∈ {0, 2, 5}
//!   - tree_max_buckets       ∈ {16, 32, 48, 64, 96, 128}   (192/256 dropped per >10s rule)
//!   - tree_num_properties    ∈ {3, 5, 7, 10, 13, 16}
//!   - tree_sample_fraction   ∈ {0.10, 0.20, 0.35, 0.50, 0.65}
//!
//! Per-cell sample plan: 25 random scalar tuples (with deterministic RNG
//! seed = hash(image_sha, size, cell_id) so reruns produce identical data).
//! 16 cells × 25 samples = 400 configs per (image, size).
//!
//! Plus 16 anchor configs holding (mid-scalar) per cell to give the picker
//! a stable reference point per cell.
//!
//! Per row: bytes + encode_ms + all knob values, joined to per-(image,size)
//! features TSV. Pareto extraction + scalar regression labels happen at
//! training time per zenanalyze#43 (time_budgeted objective).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!     --features 'std parallel' \
//!     --example lossless_pareto_calibrate -- \
//!       --manifest /home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv \
//!       --output benchmarks/lossless_pareto_<DATE>.tsv \
//!       --features-output benchmarks/lossless_pareto_features_<DATE>.tsv \
//!       [--samples-per-cell N] [--max-images N] [--sizes 64,256,1024,native]
//!       [--features-only] [--smoke]

use jxl_encoder::LosslessInternalParams;
use jxl_encoder::api::{LosslessConfig, Lz77Method, PixelLayout};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

// ---------------------------------------------------------------------
// Scalar grids
// ---------------------------------------------------------------------

const NB_RCTS_GRID: &[u8] = &[0, 4, 7, 9, 19];
const WP_PARAM_GRID: &[u8] = &[0, 2, 5];
// 192 + 256 dropped per >10s rule: 256 catastrophic (avg 661s @ small),
// 192 borderline (medium 4.7s, large 3.9s with n=9 — biased toward easy
// units; native CLIC photos extrapolate to 10-20s at 192 buckets).
// Picker can recommend up to 128 — well-covered range matching libjxl's
// e8 default (kKitten=128).
const TREE_MAX_BUCKETS_GRID: &[u16] = &[16, 32, 48, 64, 96, 128];
const TREE_NUM_PROPS_GRID: &[u8] = &[3, 5, 7, 10, 13, 16];
const TREE_SAMPLE_FRACTION_GRID: &[f32] = &[0.10, 0.20, 0.35, 0.50, 0.65];

// ---------------------------------------------------------------------
// Categorical cell axes
// ---------------------------------------------------------------------

const LZ77_AXES: &[(u8, &str, Option<Lz77Method>)] = &[
    (0, "none", None),
    (1, "rle", Some(Lz77Method::Rle)),
    (2, "greedy", Some(Lz77Method::Greedy)),
    (3, "optimal", Some(Lz77Method::Optimal)),
];

#[derive(Clone, Copy, Debug)]
struct CellSpec {
    cell_id: u8,
    lz77_label: &'static str,
    lz77_method: Option<Lz77Method>,
    squeeze: bool,
    patches: bool,
}

fn enumerate_cells() -> Vec<CellSpec> {
    let mut out = Vec::new();
    let mut id = 0u8;
    for &(_lz_id, lz_label, lz_method) in LZ77_AXES {
        for &squeeze in &[false, true] {
            for &patches in &[false, true] {
                out.push(CellSpec {
                    cell_id: id,
                    lz77_label: lz_label,
                    lz77_method: lz_method,
                    squeeze,
                    patches,
                });
                id += 1;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------
// Config (one row per encode)
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct RowConfig {
    cell: CellSpec,
    nb_rcts_to_try: u8,
    wp_num_param_sets: u8,
    tree_max_buckets: u16,
    tree_num_properties: u8,
    tree_sample_fraction: f32,
    /// Anchor=0 means a deterministic mid-scalar reference; anchor=N means
    /// the N-th random sample for this (image, size, cell).
    sample_idx: u32,
}

/// Anchor scalar values placed at one row per cell.
fn anchor_scalars() -> (u8, u8, u16, u8, f32) {
    // mid-of-grid for each (matches roughly e7 defaults).
    (7, 2, 96, 7, 0.50)
}

/// Random scalar tuple deterministic from (image_sha, size_class, cell_id, sample_idx).
fn sample_scalars(
    image_sha: &str,
    size_class: &str,
    cell_id: u8,
    sample_idx: u32,
) -> (u8, u8, u16, u8, f32) {
    let mut hasher = DefaultHasher::new();
    image_sha.hash(&mut hasher);
    size_class.hash(&mut hasher);
    cell_id.hash(&mut hasher);
    sample_idx.hash(&mut hasher);
    let seed = hasher.finish();
    let mut r = fastrand::Rng::with_seed(seed);
    (
        NB_RCTS_GRID[r.usize(0..NB_RCTS_GRID.len())],
        WP_PARAM_GRID[r.usize(0..WP_PARAM_GRID.len())],
        TREE_MAX_BUCKETS_GRID[r.usize(0..TREE_MAX_BUCKETS_GRID.len())],
        TREE_NUM_PROPS_GRID[r.usize(0..TREE_NUM_PROPS_GRID.len())],
        TREE_SAMPLE_FRACTION_GRID[r.usize(0..TREE_SAMPLE_FRACTION_GRID.len())],
    )
}

// ---------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------

struct Args {
    manifest: PathBuf,
    split: String,
    sizes: Vec<u32>,
    output: PathBuf,
    features_output: PathBuf,
    samples_per_cell: u32,
    max_images: usize,
    threads: usize,
    features_only: bool,
    smoke: bool,
}

fn parse_args() -> Args {
    let mut manifest =
        PathBuf::from("/home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv");
    let mut split = "".to_string(); // empty = all splits
    let mut sizes: Vec<u32> = Vec::new();
    let mut samples_per_cell = 25u32;
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
            "--samples-per-cell" => samples_per_cell = it.next().unwrap().parse().expect("uint"),
            "--output" => output = PathBuf::from(it.next().unwrap()),
            "--features-output" => features_output = PathBuf::from(it.next().unwrap()),
            "--max-images" => max_images = it.next().unwrap().parse().expect("max-images uint"),
            "--threads" => threads = it.next().unwrap().parse().expect("threads uint"),
            "--features-only" => features_only = true,
            "--smoke" => {
                smoke = true;
                max_images = max_images.min(2);
                samples_per_cell = samples_per_cell.min(3);
                if sizes.is_empty() {
                    sizes = vec![256];
                }
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if sizes.is_empty() {
        sizes = vec![64, 256, 1024, 0];
    }
    Args {
        manifest,
        split,
        sizes,
        output,
        features_output,
        samples_per_cell,
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
// Manifest
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
            continue;
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
// Image IO
// ---------------------------------------------------------------------

fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
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
// Encoder construction with custom LosslessInternalParams
// ---------------------------------------------------------------------

/// Build a LosslessConfig from the row's scalar values, applying overrides
/// via [`LosslessInternalParams`]. We start from `with_effort(7)` (a sane
/// midpoint for the bundled fields we don't sweep) and feed only the
/// sweep-controlled fields through the segmented public surface. Cell
/// categoricals (`lz77_method`, `patches`, `squeeze`) ride on the existing
/// per-knob public setters because they're not part of the internal-param
/// surface.
fn build_encoder(rc: &RowConfig) -> LosslessConfig {
    let params = LosslessInternalParams {
        nb_rcts_to_try: Some(rc.nb_rcts_to_try),
        wp_num_param_sets: Some(rc.wp_num_param_sets),
        tree_max_buckets: Some(rc.tree_max_buckets),
        tree_num_properties: Some(rc.tree_num_properties),
        tree_sample_fraction: Some(rc.tree_sample_fraction),
        // Use fraction-based sampling (clear the fixed cap).
        tree_max_samples_fixed: Some(0),
        ..Default::default()
    };

    // Build the LosslessConfig: apply the effort first so the
    // internal-params builder snapshots the right effort-derived defaults
    // before applying overrides; squeeze + patches + lz77_method ride on
    // the per-knob public setters.
    let mut cfg = LosslessConfig::new()
        .with_effort(7)
        .with_internal_params(params)
        .with_squeeze(rc.cell.squeeze)
        .with_patches(rc.cell.patches)
        .with_threads(1);

    if let Some(m) = rc.cell.lz77_method {
        cfg = cfg.with_lz77(true).with_lz77_method(m);
    } else {
        cfg = cfg.with_lz77(false);
    }
    cfg
}

fn encode_one(rgb: &[u8], w: u32, h: u32, rc: &RowConfig) -> Option<(usize, f64)> {
    let cfg = build_encoder(rc);
    let start = Instant::now();
    let bytes = match cfg.encode(rgb, w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(_) => return None,
    };
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes.len(), encode_ms))
}

// ---------------------------------------------------------------------
// Features
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
    let cells = enumerate_cells();

    // Each (image, size, cell) generates 1 anchor + samples_per_cell random rows.
    let rows_per_image_size = (cells.len() as u32) * (1 + args.samples_per_cell);
    let total_encodes = (entries.len() as u32) * (args.sizes.len() as u32) * rows_per_image_size;
    eprintln!(
        "[lossless_pareto_calibrate] {} images × {} sizes × {} cells × (1+{}) samples = {} encodes ({})",
        entries.len(),
        args.sizes.len(),
        cells.len(),
        args.samples_per_cell,
        total_encodes,
        if args.features_only {
            "features-only"
        } else {
            "full sweep"
        },
    );
    eprintln!(
        "[lossless_pareto_calibrate] manifest: {} (split={})",
        args.manifest.display(),
        if args.split.is_empty() {
            "<all>"
        } else {
            &args.split
        }
    );
    eprintln!(
        "[lossless_pareto_calibrate] output:   {}",
        args.output.display()
    );
    eprintln!(
        "[lossless_pareto_calibrate] features: {}",
        args.features_output.display()
    );

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
                "image_sha\tsplit\tcontent_class\tsize_class\twidth\theight\tcell_id\tlz77_method\tsqueeze\tpatches\tnb_rcts_to_try\twp_num_param_sets\ttree_max_buckets\ttree_num_properties\ttree_sample_fraction\tsample_idx\tbytes\tencode_ms"
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
        write!(
            f,
            "image_sha\tsplit\tcontent_class\tsize_class\twidth\theight"
        )
        .ok();
        for c in &cols {
            write!(f, "\tfeat_{}", c.name()).ok();
        }
        writeln!(f).ok();
    }

    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let started = Instant::now();
    let unit_count = entries.len() * args.sizes.len();
    let done = std::sync::atomic::AtomicUsize::new(0);

    // Outer parallelism on entries (not (entry, size) pairs) so each
    // image's PNG decodes once instead of once per size. Features are
    // computed by resizing directly from the native buffer for each
    // size (same pixels as the original per-pair version — no
    // cascading approximation). Saves the 4× redundant PNG decode and
    // shrinks features-only wall-clock by ~50%; encode-dominated full
    // sweeps see a smaller relative gain since encoding dwarfs IO.
    entries.par_iter().for_each(|entry| {
        let (rgb_native, w_native, h_native) = match load_png(&entry.path) {
            Some(t) => t,
            None => {
                eprintln!("  skip (load failed): {}", entry.path.display());
                return;
            }
        };

        for &target_size in &args.sizes {
            // Resize from native every time — bit-exact same as the
            // original per-pair version. The win is the load skip,
            // not the resize chain.
            let (rgb_owned, w, h) =
                resize_to(&rgb_native, w_native, h_native, target_size);
            let rgb = rgb_owned.as_slice();
            let size_class = match target_size {
                64 => "tiny",
                256 => "small",
                1024 => "medium",
                0 => "large",
                _ => "custom",
            };

            let analysis = analyze_features_rgb8(rgb, w, h, &query);
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
                let (a_rcts, a_wp, a_buckets, a_props, a_frac) = anchor_scalars();
                for cell in &cells {
                    // Build all configs for this cell: 1 anchor + N samples.
                    let mut row_cfgs: Vec<RowConfig> = Vec::with_capacity(1 + args.samples_per_cell as usize);
                    row_cfgs.push(RowConfig {
                        cell: *cell,
                        nb_rcts_to_try: a_rcts,
                        wp_num_param_sets: a_wp,
                        tree_max_buckets: a_buckets,
                        tree_num_properties: a_props,
                        tree_sample_fraction: a_frac,
                        sample_idx: 0,
                    });
                    for s in 1..=args.samples_per_cell {
                        let (r, wpr, b, p, f) = sample_scalars(&entry.sha256, size_class, cell.cell_id, s);
                        row_cfgs.push(RowConfig {
                            cell: *cell,
                            nb_rcts_to_try: r,
                            wp_num_param_sets: wpr,
                            tree_max_buckets: b,
                            tree_num_properties: p,
                            tree_sample_fraction: f,
                            sample_idx: s,
                        });
                    }

                    for rc in &row_cfgs {
                        let row = encode_one(rgb, w, h, rc);
                        let mut f = main_file.lock().unwrap();
                        match row {
                            Some((bytes, encode_ms)) => {
                                writeln!(
                                    f,
                                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{}\t{:.3}",
                                    entry.sha256,
                                    entry.split,
                                    entry.content_class,
                                    size_class,
                                    w,
                                    h,
                                    rc.cell.cell_id,
                                    rc.cell.lz77_label,
                                    rc.cell.squeeze as u8,
                                    rc.cell.patches as u8,
                                    rc.nb_rcts_to_try,
                                    rc.wp_num_param_sets,
                                    rc.tree_max_buckets,
                                    rc.tree_num_properties,
                                    rc.tree_sample_fraction,
                                    rc.sample_idx,
                                    bytes,
                                    encode_ms,
                                )
                                .ok();
                            }
                            None => {
                                writeln!(
                                    f,
                                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t\t",
                                    entry.sha256,
                                    entry.split,
                                    entry.content_class,
                                    size_class,
                                    w,
                                    h,
                                    rc.cell.cell_id,
                                    rc.cell.lz77_label,
                                    rc.cell.squeeze as u8,
                                    rc.cell.patches as u8,
                                    rc.nb_rcts_to_try,
                                    rc.wp_num_param_sets,
                                    rc.tree_max_buckets,
                                    rc.tree_num_properties,
                                    rc.tree_sample_fraction,
                                    rc.sample_idx,
                                )
                                .ok();
                            }
                        }
                    }
                    main_file.lock().unwrap().flush().ok();
                }
            }

            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 4 == 0 || n == unit_count {
                let dt = started.elapsed().as_secs_f64();
                let rate = n as f64 / dt;
                let eta = (unit_count - n) as f64 / rate;
                eprintln!(
                    "  progress: {}/{}  ({:.2}/sec, ETA {:.0}s = {:.1}h)",
                    n, unit_count, rate, eta, eta / 3600.0,
                );
            }
        }
    });

    eprintln!(
        "[lossless_pareto_calibrate] done in {:.0}s ({:.2}h){}",
        started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64() / 3600.0,
        if args.smoke { " [smoke]" } else { "" },
    );
}
