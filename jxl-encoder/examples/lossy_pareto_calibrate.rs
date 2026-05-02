//! Pareto calibration sweep for the zenjxl lossy picker.
//!
//! Sweeps INDEPENDENT internal knobs via `LossyConfig::with_internal_params`,
//! at single-shot quantization (butteraugli_iters = 0 always). The decision
//! "should I run the butteraugli loop on this image?" is a separate-stage
//! picker question; the underlying first-shot quantization tuning is what
//! we want this oracle to characterize.
//!
//! Cells (categorical, 8):
//!   - ac_strategy_intensity ∈ {compact, full}     (2)
//!     - compact: try_dct64=false, fine_grained_step=2 (e7-style)
//!     - full:    try_dct64=true,  fine_grained_step=1 (e9-style)
//!   - enhanced_clustering_vardct ∈ {off, on}      (2)
//!   - gaborish ∈ {off, on}                        (2)
//!   - patches ∈ {off, on}                         (2)
//!
//! Per-cell scalar samples (continuous, dense random):
//!   - k_info_loss_mul_base   ∈ [1.0, 1.5]   (cost-model: pixel-domain loss weight)
//!   - k_ac_quant             ∈ [0.65, 0.85] (AC quantization threshold)
//!   - entropy_mul_dct8       ∈ [0.70, 0.95] (DCT8 favor in AC strategy selection)
//!
//! Distance axis: 9 distances {0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}.
//!
//! Per row: bytes + encode_ms + butteraugli + ssim2, all knob values, joined to
//! per-(image, size) features. Pareto extraction + scalar regression labels
//! happen at training time per zenanalyze#43 (time_budgeted objective).
//!
//! Per CLAUDE.md: jxl-oxide decode in srgb_linear; butteraugli on linear f32;
//! SSIM2 on decoded-linear→sRGB u8.

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rayon::prelude::*;
use rgb::RGB;
use std::fs::OpenOptions;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

// ---------------------------------------------------------------------
// Distance grid + scalar bands
// ---------------------------------------------------------------------

const DEFAULT_DISTANCES: &[f32] = &[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0];

/// Discrete value grids for stratified random sampling of scalar knobs.
const K_INFO_LOSS_MUL_GRID: &[f32] = &[1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
const K_AC_QUANT_GRID: &[f32] = &[0.65, 0.70, 0.75, 0.80, 0.85];
const ENTROPY_MUL_DCT8_GRID: &[f32] = &[0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

// ---------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum AcIntensity {
    Compact, // e7-style: try_dct64=false, fine_grained_step=2
    Full,    // e9-style: try_dct64=true, fine_grained_step=1
}

#[derive(Clone, Copy, Debug)]
struct CellSpec {
    cell_id: u8,
    ac_intensity: AcIntensity,
    enhanced_clustering: bool,
    gaborish: bool,
    patches: bool,
}

fn enumerate_cells() -> Vec<CellSpec> {
    let mut out = Vec::new();
    let mut id = 0u8;
    for &ac in &[AcIntensity::Compact, AcIntensity::Full] {
        for &ec in &[false, true] {
            for &gab in &[false, true] {
                for &pa in &[false, true] {
                    out.push(CellSpec {
                        cell_id: id,
                        ac_intensity: ac,
                        enhanced_clustering: ec,
                        gaborish: gab,
                        patches: pa,
                    });
                    id += 1;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------
// Row config
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct RowConfig {
    cell: CellSpec,
    distance: f32,
    k_info_loss_mul: f32,
    k_ac_quant: f32,
    entropy_mul_dct8: f32,
    sample_idx: u32,
}

fn anchor_scalars() -> (f32, f32, f32) {
    // Match e7 reference defaults.
    (1.2, 0.765, 0.8)
}

fn sample_scalars(
    image_sha: &str,
    size_class: &str,
    cell_id: u8,
    distance: f32,
    sample_idx: u32,
) -> (f32, f32, f32) {
    let mut hasher = DefaultHasher::new();
    image_sha.hash(&mut hasher);
    size_class.hash(&mut hasher);
    cell_id.hash(&mut hasher);
    distance.to_bits().hash(&mut hasher);
    sample_idx.hash(&mut hasher);
    let seed = hasher.finish();
    let mut r = fastrand::Rng::with_seed(seed);
    (
        K_INFO_LOSS_MUL_GRID[r.usize(0..K_INFO_LOSS_MUL_GRID.len())],
        K_AC_QUANT_GRID[r.usize(0..K_AC_QUANT_GRID.len())],
        ENTROPY_MUL_DCT8_GRID[r.usize(0..ENTROPY_MUL_DCT8_GRID.len())],
    )
}

// ---------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------

struct Args {
    manifest: PathBuf,
    split: String,
    sizes: Vec<u32>,
    distances: Vec<f32>,
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
    let mut split = "".to_string();
    let mut sizes: Vec<u32> = Vec::new();
    let mut distances: Vec<f32> = Vec::new();
    let mut samples_per_cell = 10u32;
    let mut max_images = usize::MAX;
    let mut threads = 0;
    let mut features_only = false;
    let mut smoke = false;
    let date = chrono_today();
    let mut output = PathBuf::from(format!("benchmarks/lossy_pareto_{date}.tsv"));
    let mut features_output = PathBuf::from(format!("benchmarks/lossy_pareto_features_{date}.tsv"));

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
            "--distances" => {
                let s = it.next().unwrap();
                for tok in s.split(',') {
                    distances.push(tok.parse().expect("float"));
                }
            }
            "--samples-per-cell" => samples_per_cell = it.next().unwrap().parse().expect("uint"),
            "--output" => output = PathBuf::from(it.next().unwrap()),
            "--features-output" => features_output = PathBuf::from(it.next().unwrap()),
            "--max-images" => max_images = it.next().unwrap().parse().expect("uint"),
            "--threads" => threads = it.next().unwrap().parse().expect("uint"),
            "--features-only" => features_only = true,
            "--smoke" => {
                smoke = true;
                max_images = max_images.min(2);
                samples_per_cell = samples_per_cell.min(2);
                if sizes.is_empty() {
                    sizes = vec![256];
                }
                if distances.is_empty() {
                    distances = vec![0.5, 1.0, 2.0];
                }
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if sizes.is_empty() {
        sizes = vec![64, 256, 1024, 0];
    }
    if distances.is_empty() {
        distances = DEFAULT_DISTANCES.to_vec();
    }
    Args {
        manifest,
        split,
        sizes,
        distances,
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
// Image IO + linear conversion
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

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn srgb_pixels_to_linear(rgb: &[u8]) -> Vec<RGB<f32>> {
    rgb.chunks(3)
        .map(|p| {
            RGB::new(
                srgb_to_linear_f32(p[0]),
                srgb_to_linear_f32(p[1]),
                srgb_to_linear_f32(p[2]),
            )
        })
        .collect()
}

fn srgb_pixels_to_arr3(rgb: &[u8]) -> Vec<[u8; 3]> {
    rgb.chunks(3).map(|p| [p[0], p[1], p[2]]).collect()
}

// ---------------------------------------------------------------------
// Encoder construction with custom LossyInternalParams
// ---------------------------------------------------------------------

/// Build a LossyConfig from the row's scalar values + cell, applying
/// overrides via [`LossyInternalParams`]. Starts from `with_effort(7)`
/// (sane midpoint), applies cell gates (AC strategy intensity,
/// enhanced_clustering) through the segmented public surface, and overrides
/// the swept scalar fields. `gaborish` / `patches` / `butteraugli_iters`
/// ride on the existing per-knob public setters because they're not part
/// of the internal-param surface.
fn build_encoder(rc: &RowConfig) -> LossyConfig {
    // AC strategy intensity gate: bundles try_dct64 + fine_grained_step.
    let (try_dct64, fine_grained_step) = match rc.cell.ac_intensity {
        AcIntensity::Compact => (false, 2u8), // e7-style
        AcIntensity::Full => (true, 1u8),     // e9-style
    };

    let mut entropy_mul = jxl_encoder::EntropyMulTable::reference();
    entropy_mul.dct8 = rc.entropy_mul_dct8;

    let params = LossyInternalParams {
        try_dct64: Some(try_dct64),
        fine_grained_step: Some(fine_grained_step),
        try_dct32: Some(true),
        try_dct16: Some(true),
        try_dct4x8_afv: Some(true),
        non_aligned_eval: Some(true),
        enhanced_clustering_vardct: Some(rc.cell.enhanced_clustering),
        k_info_loss_mul_base: Some(rc.k_info_loss_mul),
        k_ac_quant: Some(rc.k_ac_quant),
        entropy_mul_table: Some(entropy_mul),
        ..Default::default()
    };

    // Apply effort first so the internal-params builder snapshots the right
    // effort-derived defaults before our overrides land. gaborish / patches /
    // butteraugli_iters use per-knob public setters (not part of the
    // internal-param surface). Single-shot quantization (iter=0) per design.
    LossyConfig::new(rc.distance)
        .with_effort(7)
        .with_internal_params(params)
        .with_gaborish(rc.cell.gaborish)
        .with_patches(rc.cell.patches)
        .with_butteraugli_iters(0)
        .with_threads(1)
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn encode_and_score(
    rgb: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
    rc: &RowConfig,
) -> Option<(usize, f64, f64, f64)> {
    let cfg = build_encoder(rc);
    let start = Instant::now();
    let bytes = match cfg.encode(rgb, w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(_) => return None,
    };
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    let (dw, dh, dec_lin) = decode_jxl_linear(&bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }

    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let bfly = match butteraugli_linear(
        orig_linear.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    ) {
        Ok(r) => r.score as f64,
        Err(_) => return None,
    };

    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = match fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()) {
        Ok(s) => s,
        Err(_) => return None,
    };

    Some((bytes.len(), encode_ms, bfly, ssim2))
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

    let rows_per_image_size_distance = (cells.len() as u32) * (1 + args.samples_per_cell);
    let total_encodes = (entries.len() as u32)
        * (args.sizes.len() as u32)
        * (args.distances.len() as u32)
        * rows_per_image_size_distance;
    eprintln!(
        "[lossy_pareto_calibrate] {} images × {} sizes × {} distances × {} cells × (1+{}) samples = {} encodes ({})",
        entries.len(),
        args.sizes.len(),
        args.distances.len(),
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
        "[lossy_pareto_calibrate] manifest: {} (split={})",
        args.manifest.display(),
        if args.split.is_empty() {
            "<all>"
        } else {
            &args.split
        }
    );
    eprintln!(
        "[lossy_pareto_calibrate] output:   {}",
        args.output.display()
    );
    eprintln!(
        "[lossy_pareto_calibrate] features: {}",
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
                "image_sha\tsplit\tcontent_class\tsize_class\twidth\theight\tdistance\tcell_id\tac_intensity\tenhanced_clustering\tgaborish\tpatches\tk_info_loss_mul\tk_ac_quant\tentropy_mul_dct8\tsample_idx\tbytes\tencode_ms\tbutteraugli\tssim2"
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
    // image's PNG decodes once instead of once per size. Resize from
    // native each time keeps the features bit-exact identical to the
    // original per-pair version.
    entries.par_iter().for_each(|entry| {
        let (rgb_native, w_native, h_native) = match load_png(&entry.path) {
            Some(t) => t,
            None => {
                eprintln!("  skip (load failed): {}", entry.path.display());
                return;
            }
        };

        for &target_size in &args.sizes {
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
                let orig_linear_pixels = srgb_pixels_to_linear(rgb);
                let orig_srgb_arrs = srgb_pixels_to_arr3(rgb);
                let orig_lin_img: Img<Vec<RGB<f32>>> =
                    Img::new(orig_linear_pixels, w as usize, h as usize);
                let orig_srgb_img: Img<Vec<[u8; 3]>> =
                    Img::new(orig_srgb_arrs, w as usize, h as usize);

                let (a_info, a_aq, a_dct8) = anchor_scalars();

                for &distance in &args.distances {
                    for cell in &cells {
                        let mut row_cfgs: Vec<RowConfig> = Vec::with_capacity(1 + args.samples_per_cell as usize);
                        row_cfgs.push(RowConfig {
                            cell: *cell,
                            distance,
                            k_info_loss_mul: a_info,
                            k_ac_quant: a_aq,
                            entropy_mul_dct8: a_dct8,
                            sample_idx: 0,
                        });
                        for s in 1..=args.samples_per_cell {
                            let (i, q, d) = sample_scalars(&entry.sha256, size_class, cell.cell_id, distance, s);
                            row_cfgs.push(RowConfig {
                                cell: *cell,
                                distance,
                                k_info_loss_mul: i,
                                k_ac_quant: q,
                                entropy_mul_dct8: d,
                                sample_idx: s,
                            });
                        }

                        for rc in &row_cfgs {
                            let row = encode_and_score(rgb, &orig_lin_img, &orig_srgb_img, w, h, rc);
                            let ac_str = match rc.cell.ac_intensity {
                                AcIntensity::Compact => "compact",
                                AcIntensity::Full => "full",
                            };
                            let mut f = main_file.lock().unwrap();
                            match row {
                                Some((bytes, encode_ms, bfly, ssim2)) => {
                                    writeln!(
                                        f,
                                        "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.3}\t{:.4}\t{:.3}",
                                        entry.sha256,
                                        entry.split,
                                        entry.content_class,
                                        size_class,
                                        w,
                                        h,
                                        rc.distance,
                                        rc.cell.cell_id,
                                        ac_str,
                                        rc.cell.enhanced_clustering as u8,
                                        rc.cell.gaborish as u8,
                                        rc.cell.patches as u8,
                                        rc.k_info_loss_mul,
                                        rc.k_ac_quant,
                                        rc.entropy_mul_dct8,
                                        rc.sample_idx,
                                        bytes,
                                        encode_ms,
                                        bfly,
                                        ssim2,
                                    )
                                    .ok();
                                }
                                None => {
                                    writeln!(
                                        f,
                                        "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t\t\t\t",
                                        entry.sha256,
                                        entry.split,
                                        entry.content_class,
                                        size_class,
                                        w,
                                        h,
                                        rc.distance,
                                        rc.cell.cell_id,
                                        ac_str,
                                        rc.cell.enhanced_clustering as u8,
                                        rc.cell.gaborish as u8,
                                        rc.cell.patches as u8,
                                        rc.k_info_loss_mul,
                                        rc.k_ac_quant,
                                        rc.entropy_mul_dct8,
                                        rc.sample_idx,
                                    )
                                    .ok();
                                }
                            }
                        }
                        main_file.lock().unwrap().flush().ok();
                    }
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
        "[lossy_pareto_calibrate] done in {:.0}s ({:.2}h){}",
        started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64() / 3600.0,
        if args.smoke { " [smoke]" } else { "" },
    );
}
