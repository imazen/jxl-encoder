//! W44-35 chunk 2 — validate cheap RGB-based smoothness proxies
//! against zenanalyze ground-truth features.
//!
//! For each test image, compute:
//!   - zenanalyze: edge_density, flat_color_block_ratio,
//!                 high_freq_energy_ratio (ground truth)
//!   - proxy_edge_density: mean abs luma diff between adjacent
//!                         4x-downsampled samples
//!   - proxy_flat_ratio: fraction of 8x8 blocks where variance < 200
//!
//! Goal: find proxies that, when paired with simple thresholds,
//! reproduce the W44-35 calibration discriminator (admit DCT64
//! for 1418519 + 7256805, skip everything else).
//!
//! Build:
//!   cargo build -p jxl-encoder --release \
//!       --features '__expert' \
//!       --example dct64_smart_dispatch_proxy_calibrate
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!       --features '__expert' \
//!       --example dct64_smart_dispatch_proxy_calibrate

use std::path::PathBuf;
use std::time::Instant;

use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

const SOURCES: &[(&str, &str, &str)] = &[
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

const CROP_DIM: u32 = 512;

fn load_and_crop(path: &PathBuf) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap();
    let mut rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w > CROP_DIM || h > CROP_DIM {
        let crop_w = w.min(CROP_DIM);
        let crop_h = h.min(CROP_DIM);
        let x0 = (w - crop_w) / 2;
        let y0 = (h - crop_h) / 2;
        let cropped = image::imageops::crop(&mut rgb, x0, y0, crop_w, crop_h).to_image();
        (cropped.as_raw().clone(), crop_w, crop_h)
    } else {
        (rgb.as_raw().clone(), w, h)
    }
}

/// Mean absolute horizontal + vertical luma diff on a 4x-downsampled
/// (every 4th row × every 4th column) integer luma plane.
/// Higher value = more edges = less smooth.
fn proxy_edge_density(rgb: &[u8], w: u32, h: u32) -> (f32, f64) {
    let t0 = Instant::now();
    let w = w as usize;
    let h = h as usize;
    let stride = w * 3;
    // Downsample stride
    let ds = 4;
    let dw = w / ds;
    let dh = h / ds;
    if dw < 2 || dh < 2 {
        return (0.0, 0.0);
    }
    // Build luma plane (integer, scaled by 4 to avoid fp)
    let mut luma = vec![0u16; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let x = dx * ds;
            let y = dy * ds;
            let i = y * stride + x * 3;
            let r = rgb[i] as u16;
            let g = rgb[i + 1] as u16;
            let b = rgb[i + 2] as u16;
            // Approximate luma: (R + 2G + B) / 4 ; range 0..255
            luma[dy * dw + dx] = (r + 2 * g + b) >> 2;
        }
    }
    // Mean abs horizontal + vertical diff
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    for y in 0..dh {
        for x in 1..dw {
            let a = luma[y * dw + x - 1] as i32;
            let b = luma[y * dw + x] as i32;
            sum += (a - b).unsigned_abs() as u64;
            count += 1;
        }
    }
    for y in 1..dh {
        for x in 0..dw {
            let a = luma[(y - 1) * dw + x] as i32;
            let b = luma[y * dw + x] as i32;
            sum += (a - b).unsigned_abs() as u64;
            count += 1;
        }
    }
    let mean_diff = (sum as f32) / (count.max(1) as f32);
    // Normalise to roughly [0, 1] like zenanalyze edge_density.
    // (Empirical scaling: divide by 64 -> photos with mean_diff 25 -> 0.39, similar to zenanalyze)
    let proxy = mean_diff / 64.0;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (proxy, ms)
}

/// Proxy for high-frequency energy ratio.
/// Mean abs second-derivative on 4x-downsampled luma divided by max-abs.
/// Roughly correlates with zenanalyze HighFreqEnergyRatio.
fn proxy_hf_energy(rgb: &[u8], w: u32, h: u32) -> (f32, f64) {
    let t0 = Instant::now();
    let w = w as usize;
    let h = h as usize;
    let stride = w * 3;
    let ds = 4;
    let dw = w / ds;
    let dh = h / ds;
    if dw < 3 || dh < 3 {
        return (0.0, 0.0);
    }
    let mut luma = vec![0i32; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let x = dx * ds;
            let y = dy * ds;
            let i = y * stride + x * 3;
            let r = rgb[i] as i32;
            let g = rgb[i + 1] as i32;
            let b = rgb[i + 2] as i32;
            luma[dy * dw + dx] = (r + 2 * g + b) >> 2; // 0..255
        }
    }
    // Mean abs 2nd-derivative (proxy for HF energy)
    let mut sum2: u64 = 0;
    let mut count2: u64 = 0;
    let mut sum1: u64 = 0;
    let mut count1: u64 = 0;
    for y in 0..dh {
        for x in 1..dw - 1 {
            let l = luma[y * dw + x - 1];
            let c = luma[y * dw + x];
            let r = luma[y * dw + x + 1];
            sum2 += (2 * c - l - r).unsigned_abs() as u64;
            count2 += 1;
            sum1 += (r - l).unsigned_abs() as u64;
            count1 += 1;
        }
    }
    for y in 1..dh - 1 {
        for x in 0..dw {
            let t = luma[(y - 1) * dw + x];
            let c = luma[y * dw + x];
            let b = luma[(y + 1) * dw + x];
            sum2 += (2 * c - t - b).unsigned_abs() as u64;
            count2 += 1;
            sum1 += (b - t).unsigned_abs() as u64;
            count1 += 1;
        }
    }
    let mean_d2 = (sum2 as f32) / (count2.max(1) as f32);
    let mean_d1 = (sum1 as f32) / (count1.max(1) as f32) + 0.001;
    // Ratio of 2nd to 1st derivative magnitude as HF proxy
    let ratio = mean_d2 / mean_d1;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (ratio, ms)
}

/// Fraction of 8x8 blocks where intra-block luma variance < THRESHOLD.
/// Higher value = more flat regions = smoother / screenshot-like.
fn proxy_flat_block_ratio(rgb: &[u8], w: u32, h: u32, var_threshold: f32) -> (f32, f64) {
    let t0 = Instant::now();
    let w = w as usize;
    let h = h as usize;
    let stride = w * 3;
    let block = 8usize;
    let bw = w / block;
    let bh = h / block;
    if bw == 0 || bh == 0 {
        return (0.0, 0.0);
    }
    let mut flat = 0u64;
    let mut total = 0u64;
    for by in 0..bh {
        for bx in 0..bw {
            let mut sum: u32 = 0;
            let mut sum_sq: u32 = 0;
            for py in 0..block {
                for px in 0..block {
                    let x = bx * block + px;
                    let y = by * block + py;
                    let i = y * stride + x * 3;
                    let r = rgb[i] as u32;
                    let g = rgb[i + 1] as u32;
                    let b = rgb[i + 2] as u32;
                    let luma = (r + 2 * g + b) >> 2; // 0..255
                    sum += luma;
                    sum_sq += luma * luma;
                }
            }
            let n = (block * block) as u32;
            let mean = sum as f32 / n as f32;
            let var = (sum_sq as f32 / n as f32) - mean * mean;
            if var < var_threshold {
                flat += 1;
            }
            total += 1;
        }
    }
    let ratio = (flat as f32) / (total.max(1) as f32);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (ratio, ms)
}

fn zenanalyze_features(rgb: &[u8], w: u32, h: u32) -> (f32, f32, f32) {
    let q = AnalysisQuery::new(
        FeatureSet::new()
            .with(AnalysisFeature::EdgeDensity)
            .with(AnalysisFeature::FlatColorBlockRatio)
            .with(AnalysisFeature::HighFreqEnergyRatio),
    );
    let res = analyze_features_rgb8(rgb, w, h, &q);
    let edge = res
        .get(AnalysisFeature::EdgeDensity)
        .and_then(|v| v.as_f32())
        .unwrap_or(f32::NAN);
    let fcbr = res
        .get(AnalysisFeature::FlatColorBlockRatio)
        .and_then(|v| v.as_f32())
        .unwrap_or(f32::NAN);
    let hfer = res
        .get(AnalysisFeature::HighFreqEnergyRatio)
        .and_then(|v| v.as_f32())
        .unwrap_or(f32::NAN);
    (edge, fcbr, hfer)
}

fn main() {
    println!(
        "{:<14} {:<16} | {:>10} {:>10} {:>10} | {:>10} {:>8} {:>10} {:>8}",
        "image", "class", "za_edge", "za_fcbr", "za_hfer", "proxy_edge", "ms", "proxy_flat", "ms",
    );
    println!("{}", "─".repeat(120));

    println!(
        "{:<14} {:<16} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10}",
        "image", "class", "za_edge", "za_fcbr", "za_hfer", "proxy_edge", "flat_v=5", "proxy_hf",
    );
    for &(label, class, path_str) in SOURCES {
        let path = PathBuf::from(path_str);
        let (rgb, w, h) = load_and_crop(&path);
        let (za_edge, za_fcbr, za_hfer) = zenanalyze_features(&rgb, w, h);
        let (proxy_e, _ms_e) = proxy_edge_density(&rgb, w, h);
        let (flat5, _) = proxy_flat_block_ratio(&rgb, w, h, 5.0);
        let (proxy_h, _) = proxy_hf_energy(&rgb, w, h);
        println!(
            "{:<14} {:<16} | {:>10.3} {:>10.3} {:>10.3} | {:>10.4} {:>10.4} {:>10.4}",
            label, class, za_edge, za_fcbr, za_hfer, proxy_e, flat5, proxy_h,
        );
    }
}
