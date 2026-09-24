//! Pareto calibration sweep for the zenjxl lossless picker (issue #24).
//!
//! Sweeps INDEPENDENT internal knobs via `LosslessConfig::with_internal_params`.
//! The picker is going to *replace* the bundled-effort axis, so the oracle
//! must expose each underlying knob independently — not the bundled effort.
//!
//! Base cells (categorical, 16), repeated for each requested WP mode:
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
//! 16 cells × 25 samples = 400 sampled configs per (image, size, WP mode).
//!
//! Plus 16 anchor configs holding (mid-scalar) per cell to give the picker
//! a stable reference point per cell.
//!
//! Per row: bytes + encode_ms + all knob values, joined to per-(image,size)
//! features TSV. This tool collects oracle candidates; it does not train or
//! qualify a picker or establish a time-budgeted training objective.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert parallel learned-admission' \
//!     --example lossless_pareto_calibrate -- \
//!       --manifest /home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv \
//!       --output benchmarks/lossless_pareto_<DATE>.tsv \
//!       --features-output benchmarks/lossless_pareto_features_<DATE>.tsv \
//!       [--samples-per-cell N] [--max-images N] [--sizes 64,256,1024,native]
//!       [--features-only] [--smoke] [--wp-modes search,0,1,2,3,4]
//!
//! Outputs are new files only. Encoded bytes are retained by SHA256 in
//! `<output>.artifacts`; its `_MANIFEST.json` records the build and input plan.
//! Build with `JXL_BENCH_COMMIT` set to the full source commit. Worker count
//! defaults to one; timings with concurrent workers are not isolated latency.

use jxl_encoder::LosslessInternalParams;
use jxl_encoder::api::{LosslessConfig, Lz77Method, PixelLayout};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
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
// Retain the historical candidate grid. This harness does not qualify a
// bucket limit or extrapolate runtime to unsampled sizes.
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
    forced_wp_mode: Option<u8>,
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
                    forced_wp_mode: None,
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
    wp_modes: Vec<Option<u8>>,
}

fn parse_args() -> Args {
    let mut manifest =
        PathBuf::from("/home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv");
    let mut split = "".to_string(); // empty = all splits
    let mut sizes: Vec<u32> = Vec::new();
    let mut samples_per_cell = 25u32;
    let mut max_images = usize::MAX;
    let mut threads = 1;
    let mut features_only = false;
    let mut smoke = false;
    let mut wp_modes = vec![None];
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
            "--wp-modes" => {
                wp_modes = it
                    .next()
                    .expect("--wp-modes needs a list")
                    .split(',')
                    .map(|v| {
                        if v == "search" {
                            None
                        } else {
                            let mode: u8 = v.parse().expect("WP mode must be search or 0..=4");
                            assert!(mode <= 4, "WP mode must be 0..=4");
                            Some(mode)
                        }
                    })
                    .collect();
                for (i, mode) in wp_modes.iter().enumerate() {
                    assert!(!wp_modes[..i].contains(mode), "duplicate WP mode");
                }
            }
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
    assert!(threads > 0, "--threads must be positive");
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
        wp_modes,
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
            panic!("manifest line {} has fewer than six columns", i + 1);
        }
        if !split_filter.is_empty() && cols[1] != split_filter {
            continue;
        }
        assert!(
            cols[0].len() == 64 && cols[0].bytes().all(|b| b.is_ascii_hexdigit()),
            "manifest line {} needs a source-file SHA256",
            i + 1
        );
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

fn load_png(path: &std::path::Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|err| panic!("load {}: {err}", path.display()));
    let rgb = img.to_rgb8();
    (rgb.as_raw().clone(), rgb.width(), rgb.height())
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
    let mut params = LosslessInternalParams::default();
    params.nb_rcts_to_try = Some(rc.nb_rcts_to_try);
    params.wp_num_param_sets = Some(rc.wp_num_param_sets);
    params.forced_wp_mode = rc.cell.forced_wp_mode;
    params.tree_max_buckets = Some(rc.tree_max_buckets);
    params.tree_num_properties = Some(rc.tree_num_properties);
    params.tree_sample_fraction = Some(rc.tree_sample_fraction);
    // Use fraction-based sampling (clear the fixed cap).
    params.tree_max_samples_fixed = Some(0);

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

fn encode_one(
    rgb: &[u8],
    w: u32,
    h: u32,
    rc: &RowConfig,
) -> Result<(Vec<u8>, f64), jxl_encoder::At<jxl_encoder::EncodeError>> {
    let cfg = build_encoder(rc);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8)?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((bytes, encode_ms))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn persist_encode(dir: &std::path::Path, encoded: &[u8]) -> std::io::Result<String> {
    // Worker threads may produce identical outputs. Keep verification and creation
    // in one critical section; persistence is outside the encode timer.
    static WRITER: Mutex<()> = Mutex::new(());
    let _guard = WRITER.lock().unwrap();
    let sha = sha256_hex(encoded);
    let path = dir.join(format!("{sha}.jxl"));
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => file.write_all(encoded)?,
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            if std::fs::read(&path)? != encoded {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("existing artifact differs: {}", path.display()),
                ));
            }
        }
        Err(err) => return Err(err),
    }
    Ok(sha)
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
    if let Some(v) = analysis.get(f) {
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

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

fn main() {
    let args = parse_args();
    let Some(build_commit) = option_env!("JXL_BENCH_COMMIT") else {
        eprintln!("build with JXL_BENCH_COMMIT=<full source commit>");
        std::process::exit(2);
    };
    assert!(
        build_commit.len() == 40 && build_commit.bytes().all(|b| b.is_ascii_hexdigit()),
        "JXL_BENCH_COMMIT must be a full commit hash"
    );
    let artifacts = args.output.with_extension("artifacts");
    if !args.features_only {
        std::fs::create_dir_all(&artifacts).expect("create artifact directory");
    }

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("build worker pool");
    }

    let entries = load_manifest(&args.manifest, &args.split);
    let n_images = entries.len().min(args.max_images);
    let entries: Vec<ManifestEntry> = entries.into_iter().take(n_images).collect();
    assert!(!entries.is_empty(), "manifest/filter selected no images");
    let mut cells = Vec::new();
    for mode in &args.wp_modes {
        for mut cell in enumerate_cells() {
            cell.cell_id = cells.len() as u8;
            cell.forced_wp_mode = *mode;
            cells.push(cell);
        }
    }
    let cols = feature_columns();
    let metadata = serde_json::json!({
        "schema": "lossless-picker-oracle-v3",
        "build_commit": build_commit,
        "binary_sha256": sha256_hex(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
        "manifest": args.manifest,
        "manifest_sha256": sha256_hex(&std::fs::read(&args.manifest).unwrap()),
        "split": args.split,
        "sizes": args.sizes,
        "images": entries.len(),
        "samples_per_cell": args.samples_per_cell,
        "worker_threads": args.threads,
        "encoder_threads": 1,
        "wp_modes": args.wp_modes,
        "features_only": args.features_only,
        "features": cols.iter().map(|c| c.name()).collect::<Vec<_>>(),
        "pixel_input": "PNG converted to packed RGB8; native or Lanczos3 downsample",
        "date": chrono_today(),
    });
    let metadata_path = if args.features_only {
        args.features_output.with_extension("meta.json")
    } else {
        artifacts.join("_MANIFEST.json")
    };
    if let Some(parent) = metadata_path.parent() {
        std::fs::create_dir_all(parent).expect("create metadata parent");
    }
    let metadata_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(metadata_path)
        .expect("create new provenance file");
    serde_json::to_writer_pretty(metadata_file, &metadata).expect("write provenance");

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
        std::fs::create_dir_all(parent).expect("complete output operation");
    }
    let main_file: Option<Mutex<std::fs::File>> = if args.features_only {
        None
    } else {
        let is_new = !args.output.exists();
        let f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)
            .expect("open output");
        let f = Mutex::new(f);
        if is_new {
            let mut g = f.lock().unwrap();
            writeln!(
                g,
                "image_sha\tsplit\tcontent_class\tsize_class\twidth\theight\tcell_id\tlz77_method\tsqueeze\tpatches\tnb_rcts_to_try\twp_num_param_sets\ttree_max_buckets\ttree_num_properties\ttree_sample_fraction\tsample_idx\tbytes\tencode_ms\tencoded_sha256\tforced_wp_mode"
            )
            .expect("complete output operation");
        }
        Some(f)
    };

    if let Some(parent) = args.features_output.parent() {
        std::fs::create_dir_all(parent).expect("complete output operation");
    }
    let feat_is_new = !args.features_output.exists();
    let feat_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.features_output)
        .expect("open features output");
    let feat_file = Mutex::new(feat_file);
    if feat_is_new {
        let mut f = feat_file.lock().unwrap();
        write!(
            f,
            "image_sha\tsplit\tcontent_class\tsize_class\twidth\theight"
        )
        .expect("complete output operation");
        for c in &cols {
            write!(f, "\tfeat_{}", c.name()).expect("complete output operation");
        }
        writeln!(f).expect("complete output operation");
    }

    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let started = Instant::now();
    let unit_count = entries.len() * args.sizes.len();
    let done = std::sync::atomic::AtomicUsize::new(0);

    // Decode each source once; resize every size directly from the native pixels.
    entries.par_iter().for_each(|entry| {
        let source = std::fs::read(&entry.path).expect("read source for hash verification");
        assert_eq!(sha256_hex(&source), entry.sha256.to_ascii_lowercase(),
            "source-file hash mismatch: {}", entry.path.display());
        drop(source);
        let (rgb_native, w_native, h_native) = load_png(&entry.path);

        for &target_size in &args.sizes {
            // Resize from native every time — bit-exact same as the
            // original per-pair version. The win is the load skip,
            // not the resize chain.
            let (rgb_owned, w, h) =
                resize_to(&rgb_native, w_native, h_native, target_size);
            let rgb = rgb_owned.as_slice();
            let size_class = match target_size {
                64 => "tiny".to_owned(),
                256 => "small".to_owned(),
                1024 => "medium".to_owned(),
                0 => "large".to_owned(),
                other => format!("max{other}"),
            };

            let analysis = analyze_features_rgb8(rgb, w, h, &query);
            {
                let mut f = feat_file.lock().unwrap();
                write!(
                    f,
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    entry.sha256, entry.split, entry.content_class, size_class, w, h
                )
                .expect("complete output operation");
                for c in &cols {
                    write!(f, "\t{}", feature_value_str(&analysis, *c)).expect("complete output operation");
                }
                writeln!(f).expect("complete output operation");
                f.flush().expect("complete output operation");
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
                        let (r, wpr, b, p, f) = sample_scalars(&entry.sha256, &size_class, cell.cell_id, s);
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
                        let (encoded, encode_ms) = encode_one(rgb, w, h, rc).unwrap_or_else(|err| {
                            panic!("encode {} cell {} sample {}: {err}", entry.path.display(), rc.cell.cell_id, rc.sample_idx)
                        });
                        let encoded_sha256 = persist_encode(&artifacts, &encoded).expect("persist encoded artifact");
                        let mut f = main_file.lock().unwrap();
                        writeln!(
                            f,
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{}\t{:.3}\t{}\t{}",
                            entry.sha256, entry.split, entry.content_class, size_class,
                            w, h, rc.cell.cell_id, rc.cell.lz77_label,
                            rc.cell.squeeze as u8, rc.cell.patches as u8,
                            rc.nb_rcts_to_try, rc.wp_num_param_sets, rc.tree_max_buckets,
                            rc.tree_num_properties, rc.tree_sample_fraction, rc.sample_idx,
                            encoded.len(), encode_ms, encoded_sha256,
                            rc.cell.forced_wp_mode.map(|m| m.to_string()).unwrap_or_default(),
                        ).expect("write encoded row");

                    }
                    main_file.lock().unwrap().flush().expect("complete output operation");
                    eprintln!("  persisted {} size {} cell {}", entry.sha256, target_size, cell.cell_id);
                }
            }

            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(4) || n == unit_count {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_real_encodes_are_exact_in_both_decoders() {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .unwrap();
        let dir = PathBuf::from(home)
            .join("tmp")
            .join(format!("lossless-oracle-artifacts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/images/frymire-srgb.png"
        ))
        .unwrap();
        for (w, h) in [(64, 64), (259, 133)] {
            let rgb = source.crop_imm(0, 0, w, h).to_rgb8();
            let (r, wp, buckets, props, fraction) = anchor_scalars();
            let rc = RowConfig {
                cell: enumerate_cells()[0],
                nb_rcts_to_try: r,
                wp_num_param_sets: wp,
                tree_max_buckets: buckets,
                tree_num_properties: props,
                tree_sample_fraction: fraction,
                sample_idx: 0,
            };
            let (encoded, _) = encode_one(rgb.as_raw(), w, h, &rc).unwrap();
            let sha = persist_encode(&dir, &encoded).unwrap();
            assert_eq!(persist_encode(&dir, &encoded).unwrap(), sha);
            let path = dir.join(format!("{sha}.jxl"));
            let persisted = std::fs::read(&path).unwrap();
            assert_eq!(persisted, encoded);
            let decoded = zenjxl_decoder::decode(&persisted).unwrap();
            assert_eq!(
                (decoded.width, decoded.height, decoded.channels),
                (w as usize, h as usize, 4)
            );
            let pixels: Vec<_> = decoded
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            assert_eq!(pixels, *rgb.as_raw());
            let png = dir.join(format!("{sha}.png"));
            let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(&path)
                .arg(&png)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(image::open(png).unwrap().to_rgb8(), rgb);
        }
    }

    #[test]
    fn artifact_errors_are_reported_without_overwriting() {
        let dir = PathBuf::from(
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .unwrap(),
        )
        .join("tmp")
        .join(format!("lossless-oracle-refusals-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bytes = b"artifact refusal control";
        let sha = sha256_hex(bytes);
        let path = dir.join(format!("{sha}.jxl"));
        std::fs::write(&path, b"existing content").unwrap();
        assert_eq!(
            persist_encode(&dir, bytes).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"existing content");
        assert!(persist_encode(&path, bytes).is_err());
        // An IO error must not poison the shared writer or prevent later output.
        assert!(persist_encode(&dir, b"next output").is_ok());
    }
}
