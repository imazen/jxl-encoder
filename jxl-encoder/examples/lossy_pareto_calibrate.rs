//! Pareto calibration sweep for the zenjxl lossy picker.
//!
//! Sweeps four axes per the global benchmark-discipline rule:
//!   - **size**: tiny (64), small (256), medium (1024), large (native)
//!   - **distance**: 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
//!     (dense at low quality per web-focused-codec rule)
//!   - **config**: effort × butteraugli_iters override × cost-model knobs
//!   - **content**: photo + screenshot + mixed via the picker-train corpus
//!
//! Per (image, size, distance, config) we emit one TSV row capturing
//! `bytes + encode_ms + butteraugli + ssim2`. The picker's "is the
//! butteraugli loop worth it?" decision is trained on the bytes/quality
//! delta between e9_loop=4 and e9_loop=0 across (image, distance) cells.
//!
//! Per CLAUDE.md: jxl-oxide decode in linear sRGB to avoid PNG-metadata
//! butteraugli inflation. NEVER use butteraugli_main CLI; always Rust
//! butteraugli on linear f32.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!     --features 'std parallel butteraugli-loop' \
//!     --example lossy_pareto_calibrate -- \
//!       --manifest /home/lilith/work/codec-corpus/picker-train/manifest.tsv \
//!       --split train \
//!       --output benchmarks/lossy_pareto_<DATE>.tsv \
//!       --features-output benchmarks/lossy_pareto_features_<DATE>.tsv \
//!       [--max-images N] [--sizes 64,256,1024,native] [--threads N]
//!       [--features-only] [--smoke]

use butteraugli::{butteraugli_linear, ButteraugliParams};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rayon::prelude::*;
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
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
    /// `Some(n)` → override butteraugli_iters; `None` → use effort default.
    butteraugli_iters: Option<u32>,
    gaborish: Option<bool>,
    pixel_domain_loss: Option<bool>,
    patches: Option<bool>,
}

const fn cfg(
    id: u32,
    name: &'static str,
    effort: u8,
    bi: Option<u32>,
    gab: Option<bool>,
    pdl: Option<bool>,
    pa: Option<bool>,
) -> ConfigSpec {
    ConfigSpec {
        id,
        name,
        effort,
        butteraugli_iters: bi,
        gaborish: gab,
        pixel_domain_loss: pdl,
        patches: pa,
    }
}

/// Distance grid — dense at low-q regime per web-codec rule
/// (q5–q60 same density as q60–q100). 9 distances spanning the
/// production range.
const DISTANCES: &[f32] = &[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0];

/// Configs spanning the picker's outer search at each distance.
/// The picker's main short-circuit opportunity: detect when the
/// butteraugli loop pays off vs when first-shot quant is fine.
const CONFIGS: &[ConfigSpec] = &[
    // Effort baselines
    cfg(0, "e5_baseline",     5, None,    None, None, None),
    cfg(1, "e7_default",      7, None,    None, None, None),

    // Effort 8 — default loop=2 vs no-loop
    cfg(10, "e8_default",     8, None,    None, None, None),
    cfg(11, "e8_no_loop",     8, Some(0), None, None, None),

    // Effort 9 — default loop=4, scan loop counts
    cfg(20, "e9_default",     9, None,    None, None, None),
    cfg(21, "e9_loop_2",      9, Some(2), None, None, None),
    cfg(22, "e9_loop_1",      9, Some(1), None, None, None),
    cfg(23, "e9_no_loop",     9, Some(0), None, None, None),

    // Effort 9 cost-model knobs at default loop count (4 iters).
    // Used to disentangle "loop" from "knob" effects on quality.
    cfg(30, "e9_no_gaborish",       9, None, Some(false), None,        None),
    cfg(31, "e9_no_pixel_loss",     9, None, None,        Some(false), None),
    cfg(32, "e9_no_patches",        9, None, None,        None,        Some(false)),

    // Effort 10 anchor (max effort)
    cfg(40, "e10_default",   10, None,    None, None, None),
];

/// Default size axis. `0` means native dims.
const DEFAULT_SIZES: &[u32] = &[64, 256, 1024, 0];

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
    max_images: usize,
    threads: usize,
    features_only: bool,
    smoke: bool,
}

fn parse_args() -> Args {
    let mut manifest = PathBuf::from("/home/lilith/work/codec-corpus/picker-train/manifest.tsv");
    let mut split = "train".to_string();
    let mut sizes: Vec<u32> = Vec::new();
    let mut distances: Vec<f32> = Vec::new();
    let mut max_images = usize::MAX;
    let mut threads = 0;
    let mut features_only = false;
    let mut smoke = false;
    let date = chrono_today();
    let mut output = PathBuf::from(format!("benchmarks/lossy_pareto_{date}.tsv"));
    let mut features_output =
        PathBuf::from(format!("benchmarks/lossy_pareto_features_{date}.tsv"));

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
                    distances.push(tok.parse().expect("distance must be float"));
                }
            }
            "--output" => output = PathBuf::from(it.next().unwrap()),
            "--features-output" => features_output = PathBuf::from(it.next().unwrap()),
            "--max-images" => max_images = it.next().unwrap().parse().expect("max-images uint"),
            "--threads" => threads = it.next().unwrap().parse().expect("threads uint"),
            "--features-only" => features_only = true,
            "--smoke" => {
                smoke = true;
                max_images = max_images.min(3);
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
        sizes = DEFAULT_SIZES.to_vec();
    }
    if distances.is_empty() {
        distances = DISTANCES.to_vec();
    }
    Args {
        manifest,
        split,
        sizes,
        distances,
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
// Manifest loading (shared shape with lossless harness)
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
// Image loading + resize + reference precomputation
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
// Encode + decode + score
// ---------------------------------------------------------------------

fn build_encoder(spec: ConfigSpec, distance: f32) -> LossyConfig {
    let mut cfg = LossyConfig::new(distance)
        .with_effort(spec.effort)
        .with_threads(1);
    if let Some(n) = spec.butteraugli_iters {
        cfg = cfg.with_butteraugli_iters(n);
    }
    if let Some(g) = spec.gaborish {
        cfg = cfg.with_gaborish(g);
    }
    if let Some(p) = spec.pixel_domain_loss {
        cfg = cfg.with_pixel_domain_loss(p);
    }
    if let Some(p) = spec.patches {
        cfg = cfg.with_patches(p);
    }
    cfg
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

/// Returns (bytes, encode_ms, butteraugli, ssim2).
fn encode_and_score(
    rgb: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
    spec: ConfigSpec,
    distance: f32,
) -> Option<(usize, f64, f64, f64)> {
    let cfg = build_encoder(spec, distance);
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

    // Butteraugli on linear f32.
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

    // SSIM2 on sRGB u8 (decoded linear -> sRGB).
    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| [linear_to_srgb_u8(c[0]), linear_to_srgb_u8(c[1]), linear_to_srgb_u8(c[2])])
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

    let cells = entries.len() * args.sizes.len() * args.distances.len() * CONFIGS.len();
    eprintln!(
        "[lossy_pareto_calibrate] {} images × {} sizes × {} distances × {} configs = {} encodes ({})",
        entries.len(),
        args.sizes.len(),
        args.distances.len(),
        CONFIGS.len(),
        cells,
        if args.features_only { "features-only" } else { "full sweep" },
    );
    eprintln!("[lossy_pareto_calibrate] manifest: {} ({})", args.manifest.display(), args.split);
    eprintln!("[lossy_pareto_calibrate] output:   {}", args.output.display());
    eprintln!("[lossy_pareto_calibrate] features: {}", args.features_output.display());

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
                "image_sha\tsplit\tcontent_class\tsize_class\twidth\theight\tdistance\tconfig_id\tconfig_name\teffort\tbutteraugli_iters\tgaborish\tpixel_domain_loss\tpatches\tbytes\tencode_ms\tbutteraugli\tssim2"
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
            // Pre-compute reference: linear-light pixels for butteraugli,
            // [u8;3] arrays for SSIM2.
            let orig_linear_pixels = srgb_pixels_to_linear(&rgb);
            let orig_srgb_arrs = srgb_pixels_to_arr3(&rgb);
            let orig_lin_img: Img<Vec<RGB<f32>>> =
                Img::new(orig_linear_pixels, w as usize, h as usize);
            let orig_srgb_img: Img<Vec<[u8; 3]>> =
                Img::new(orig_srgb_arrs, w as usize, h as usize);

            for &distance in &args.distances {
                for spec in CONFIGS {
                    let row = encode_and_score(
                        &rgb,
                        &orig_lin_img,
                        &orig_srgb_img,
                        w,
                        h,
                        *spec,
                        distance,
                    );
                    let bi_str = match spec.butteraugli_iters {
                        Some(n) => n.to_string(),
                        None => "default".to_string(),
                    };
                    let opt_bool = |b: Option<bool>| match b {
                        Some(true) => "1".to_string(),
                        Some(false) => "0".to_string(),
                        None => "default".to_string(),
                    };
                    let mut f = main_file.lock().unwrap();
                    match row {
                        Some((bytes, encode_ms, bfly, ssim2)) => {
                            writeln!(
                                f,
                                "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.4}\t{:.3}",
                                entry.sha256,
                                entry.split,
                                entry.content_class,
                                size_class,
                                w,
                                h,
                                distance,
                                spec.id,
                                spec.name,
                                spec.effort,
                                bi_str,
                                opt_bool(spec.gaborish),
                                opt_bool(spec.pixel_domain_loss),
                                opt_bool(spec.patches),
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
                                "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\t\t\t",
                                entry.sha256,
                                entry.split,
                                entry.content_class,
                                size_class,
                                w,
                                h,
                                distance,
                                spec.id,
                                spec.name,
                                spec.effort,
                                bi_str,
                                opt_bool(spec.gaborish),
                                opt_bool(spec.pixel_domain_loss),
                                opt_bool(spec.patches),
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
    });

    eprintln!(
        "[lossy_pareto_calibrate] done in {:.0}s ({:.2}h){}",
        started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64() / 3600.0,
        if args.smoke { " [smoke]" } else { "" },
    );
}
