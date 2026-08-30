//! W44-35 calibration sweep — find a smoothness discriminator
//! that admits 1418519-class (smooth 512×512 photos) into DCT64
//! eval at mid-distances while keeping screenshots / pixel-art / texture
//! gated off.
//!
//! Tests `try_dct64 = true` vs default `false` (gated by
//! `adapt_to_image_lossy`) on a stratified set of small images,
//! emitting zenanalyze Tier-1 features per image so we can read off
//! the discriminator threshold from the TSV.
//!
//! Build:
//!   cargo build -p jxl-encoder --release \
//!       --features 'parallel butteraugli-loop ssim2-loop __expert' \
//!       --example dct64_smart_dispatch_calibrate
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!       --features 'parallel butteraugli-loop ssim2-loop __expert' \
//!       --example dct64_smart_dispatch_calibrate \
//!       -- benchmarks/w44_35_dct64_smart_dispatch_calibrate.tsv

use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_encoder::LossyInternalParams;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

// 6 photos (mostly smooth, some textured) + 4 screenshots / pixel-art.
// Photos are CID22-512 (512×512 = 262_144 px, well under 500k gate).
// Screenshots vary in size but several are small enough to trigger the gate.
const SOURCES: &[(&str, &str, &str)] = &[
    // (label, class, path)
    (
        "1418519",
        "smooth_photo",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
    ),
    (
        "1531677",
        "photo",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
    ),
    (
        "1420710",
        "photo",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
    ),
    (
        "1189261",
        "photo",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
    ),
    (
        "1025469",
        "photo",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
    ),
    (
        "7256805",
        "textured_photo",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png",
    ),
    // Screenshots — small enough to trigger the gate (we crop to 512×512 if larger)
    (
        "terminal",
        "screenshot",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
    ),
    (
        "codec_wiki",
        "screenshot",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
    ),
    (
        "imac_g3",
        "screenshot",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
    ),
    (
        "windows95",
        "pixel_art",
        "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
    ),
];

const DISTANCES: &[f32] = &[1.0, 1.2, 1.6, 2.0];
const EFFORTS: &[u8] = &[6, 7];
// Crop dimension for too-large source images: 512×512 (= 262_144 px), same
// as 1418519. Keeps us under the 500k gate so try_dct64=false is the default.
const CROP_DIM: u32 = 512;

fn load_and_crop(path: &PathBuf) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("open {:?}: {}", path, e));
    let mut rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w > CROP_DIM || h > CROP_DIM {
        let crop_w = w.min(CROP_DIM);
        let crop_h = h.min(CROP_DIM);
        let x0 = (w - crop_w) / 2;
        let y0 = (h - crop_h) / 2;
        let cropped = image::imageops::crop(&mut rgb, x0, y0, crop_w, crop_h).to_image();
        let pixels = cropped.as_raw().clone();
        (pixels, crop_w, crop_h)
    } else {
        (rgb.as_raw().clone(), w, h)
    }
}

fn classify(rgb: &[u8], w: u32, h: u32) -> Features {
    let needed = FeatureSet::new()
        .with(AnalysisFeature::HighFreqEnergyRatio)
        .with(AnalysisFeature::FlatColorBlockRatio)
        .with(AnalysisFeature::EdgeDensity)
        .with(AnalysisFeature::Variance)
        .with(AnalysisFeature::Uniformity);
    let q = AnalysisQuery::new(needed);
    let t0 = Instant::now();
    let res = analyze_features_rgb8(rgb, w, h, &q);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Features {
        high_freq_energy_ratio: res
            .get(AnalysisFeature::HighFreqEnergyRatio)
            .and_then(|v| v.as_f32())
            .unwrap_or(f32::NAN),
        flat_color_block_ratio: res
            .get(AnalysisFeature::FlatColorBlockRatio)
            .and_then(|v| v.as_f32())
            .unwrap_or(f32::NAN),
        edge_density: res
            .get(AnalysisFeature::EdgeDensity)
            .and_then(|v| v.as_f32())
            .unwrap_or(f32::NAN),
        variance: res
            .get(AnalysisFeature::Variance)
            .and_then(|v| v.as_f32())
            .unwrap_or(f32::NAN),
        uniformity: res
            .get(AnalysisFeature::Uniformity)
            .and_then(|v| v.as_f32())
            .unwrap_or(f32::NAN),
        classify_ms: elapsed_ms,
    }
}

#[derive(Debug, Clone, Copy)]
struct Features {
    high_freq_energy_ratio: f32,
    flat_color_block_ratio: f32,
    edge_density: f32,
    variance: f32,
    uniformity: f32,
    classify_ms: f64,
}

fn encode(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    force_dct64: bool,
) -> (usize, f64) {
    let cfg = if force_dct64 {
        let mut params = LossyInternalParams::default();
        params.try_dct64 = Some(true);
        LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(1)
            .with_internal_params(params)
    } else {
        LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(1)
    };
    let start = Instant::now();
    let bytes = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes.len(), ms)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = args
        .into_iter()
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/w44_35_dct64_smart_dispatch_calibrate.tsv"));

    let mut tsv = String::new();
    tsv.push_str(
        "image\tclass\twidth\theight\tpixels\teffort\tdistance\t\
         bytes_default\tbytes_force\tdelta_bytes\tdelta_pct\t\
         ms_default\tms_force\t\
         hf_energy_ratio\tflat_color_block_ratio\tedge_density\tvariance\tuniformity\tclassify_ms\n",
    );

    println!(
        "{:<14} {:<16} {:>5}x{:<5} {:<2} {:<4} {:>8} {:>8} {:>8} {:>7} {:>6} {:>6} {:>6} {:>6}",
        "image",
        "class",
        "w",
        "h",
        "e",
        "d",
        "bytes_A",
        "bytes_B",
        "Δ_bytes",
        "Δ_pct",
        "hfer",
        "fcbr",
        "edge",
        "var",
    );

    for &(label, class, path_str) in SOURCES {
        let path = PathBuf::from(path_str);
        let (rgb, w, h) = load_and_crop(&path);
        let pixels = (w as u64) * (h as u64);
        let feats = classify(&rgb, w, h);
        for &effort in EFFORTS {
            for &d in DISTANCES {
                // 2 measured per variant, take min wall-time
                let mut a_bytes = 0usize;
                let mut b_bytes = 0usize;
                let mut a_ms = f64::INFINITY;
                let mut b_ms = f64::INFINITY;
                for _ in 0..2 {
                    let (ba, ma) = encode(&rgb, w, h, d, effort, false);
                    let (bb, mb) = encode(&rgb, w, h, d, effort, true);
                    a_bytes = ba;
                    b_bytes = bb;
                    a_ms = a_ms.min(ma);
                    b_ms = b_ms.min(mb);
                }
                let delta = b_bytes as i64 - a_bytes as i64;
                let pct = (delta as f64) / (a_bytes as f64) * 100.0;
                println!(
                    "{:<14} {:<16} {:>5}x{:<5} {:<2} {:<4} {:>8} {:>8} {:>+8} {:>+6.2}% {:>6.3} {:>6.3} {:>6.3} {:>6.1}",
                    label,
                    class,
                    w,
                    h,
                    effort,
                    d,
                    a_bytes,
                    b_bytes,
                    delta,
                    pct,
                    feats.high_freq_energy_ratio,
                    feats.flat_color_block_ratio,
                    feats.edge_density,
                    feats.variance,
                );
                tsv.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{:.3}\t{:.6}\t{:.6}\t{:.6}\t{:.3}\t{:.6}\t{:.3}\n",
                    label, class, w, h, pixels, effort, d,
                    a_bytes, b_bytes, delta, pct, a_ms, b_ms,
                    feats.high_freq_energy_ratio,
                    feats.flat_color_block_ratio,
                    feats.edge_density,
                    feats.variance,
                    feats.uniformity,
                    feats.classify_ms,
                ));
            }
        }
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&out_path, tsv).expect("write tsv");
    println!();
    println!("TSV written to {}", out_path.display());
}
