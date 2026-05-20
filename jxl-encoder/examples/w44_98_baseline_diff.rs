// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-98 baseline-diff: production W44-98 (m3 sub-discriminator inside
//! variant Z) vs an INJECTED baseline that forces the W44-96 default
//! variant Z table (`high_d_photo_smooth_suppressed_z()`) on every cell
//! that would have gated, simulating origin/main behaviour.
//!
//! For W44-29-firing cells that DON'T pass W44-96 gates the baseline
//! and production paths are identical (both hit
//! `high_d_photo_smooth_suppressed()`).
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_98_baseline_diff

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::effort::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const CJXL_BIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";

type Cell = (&'static str, u8, f32, &'static str);

const CELLS: &[Cell] = &[
    // 7 W44-97 OPEN cells
    ("1420710.png", 5, 5.0, "OPEN"),
    ("1420710.png", 5, 6.0, "OPEN"),
    ("1420710.png", 7, 5.0, "OPEN"),
    ("1531677.png", 5, 5.0, "OPEN"),
    ("1531677.png", 6, 5.0, "OPEN"),
    ("1531677.png", 8, 5.0, "OPEN"),
    ("1531677.png", 9, 5.0, "OPEN"),
    // 3 W44-95 regressors
    ("2389166.png", 7, 5.0, "W95_REGR"),
    ("3637739.png", 5, 5.0, "W95_REGR"),
    ("3637739.png", 7, 4.0, "W95_REGR"),
    // 6 W93_REGR
    ("1189261.png", 7, 3.0, "W93_REGR"),
    ("1189261.png", 7, 4.0, "W93_REGR"),
    ("1189261.png", 7, 5.0, "W93_REGR"),
    ("1418519.png", 6, 5.0, "W93_REGR"),
    ("1418519.png", 7, 5.0, "W93_REGR"),
    ("1418519.png", 8, 5.0, "W93_REGR"),
    // W44-96 FIXED cells (regression check)
    ("1420710.png", 6, 5.0, "W96_FIXED"),
    ("1420710.png", 6, 6.0, "W96_FIXED"),
    ("1420710.png", 8, 5.0, "W96_FIXED"),
    ("1420710.png", 9, 5.0, "W96_FIXED"),
    ("1531677.png", 5, 6.0, "W96_FIXED"),
    ("1531677.png", 6, 6.0, "W96_FIXED"),
    // W44-29 FIXED cells (regression check, d<4.5)
    ("1420710.png", 5, 4.0, "SPOT_FIXED"),
    ("1420710.png", 6, 4.0, "SPOT_FIXED"),
    ("1531677.png", 5, 4.0, "SPOT_FIXED"),
    ("1531677.png", 6, 4.0, "SPOT_FIXED"),
    ("1025469.png", 5, 5.0, "SPOT_FIXED"),
    ("1025469.png", 6, 5.0, "SPOT_FIXED"),
    ("1044329.png", 5, 5.0, "SPOT_FIXED"),
];

#[derive(Debug, Clone, Copy)]
struct ImageProxy {
    mask_med: f32,
    edge_density: f32,
    fcbr: f32,
}

fn known_proxies(image: &str) -> Option<ImageProxy> {
    match image {
        "1420710.png" => Some(ImageProxy {
            mask_med: 39.549,
            edge_density: 0.9298,
            fcbr: 0.0,
        }),
        "1531677.png" => Some(ImageProxy {
            mask_med: 35.634,
            edge_density: 0.8766,
            fcbr: 0.0,
        }),
        "2389166.png" => Some(ImageProxy {
            mask_med: 46.241,
            edge_density: 0.4409,
            fcbr: 0.13354,
        }),
        "1044329.png" => Some(ImageProxy {
            mask_med: 48.029,
            edge_density: 0.5486,
            fcbr: 0.12158,
        }),
        "3637739.png" => Some(ImageProxy {
            mask_med: 75.827,
            edge_density: 0.2962,
            fcbr: 0.09155,
        }),
        "1189261.png" => Some(ImageProxy {
            mask_med: 69.078,
            edge_density: 0.4895,
            fcbr: 0.00342,
        }),
        "1418519.png" => Some(ImageProxy {
            mask_med: 92.331,
            edge_density: 0.1637,
            fcbr: 0.09839,
        }),
        "1025469.png" => Some(ImageProxy {
            mask_med: 76.08,
            edge_density: 0.3,
            fcbr: 0.05,
        }),
        _ => None,
    }
}

fn w44_96_would_fire(d: f32, p: ImageProxy) -> bool {
    d >= 4.5 && p.mask_med < 50.0 && p.edge_density >= 0.7 && p.fcbr < 0.01
}

/// Production-default encode (uses new W44-98 dispatch).
fn encode_production(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Vec<u8> {
    LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed")
}

/// Baseline-injected encode (forces W44-96 variant Z table on cells that
/// would have fired W44-96 — simulating origin/main behaviour where the
/// m3 sub-discriminator does not exist).
fn encode_baseline(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32, p: ImageProxy) -> Vec<u8> {
    let mut cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    if w44_96_would_fire(d, p) {
        // Force variant Z (the pre-W44-98 default in this gate region).
        cfg = cfg.with_high_d_photo_hint(Some(false));
        let mut internal = LossyInternalParams::default();
        internal.entropy_mul_table = Some(EntropyMulTable::high_d_photo_smooth_suppressed_z());
        cfg = cfg.with_internal_params(internal);
    }
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode failed")
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

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn measure_metrics(
    bytes: &[u8],
    w: u32,
    h: u32,
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> (f64, f64) {
    let (dw, dh, decoded) = match decode_jxl_linear(bytes) {
        Some(v) => v,
        None => return (f64::NAN, f64::NAN),
    };
    if dw != w as usize || dh != h as usize {
        return (f64::NAN, f64::NAN);
    }
    let dec_pixels: Vec<RGB<f32>> = decoded
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear.as_ref(), dec_linear.as_ref(), params)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);
    let dec_srgb: Vec<[u8; 3]> = decoded
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);
    (bfly, ssim2)
}

fn cjxl_size(src: &str, effort: u8, d: f32) -> Option<usize> {
    let tmp = format!(
        "/tmp/w44_98_baseline_cjxl_{}_{}_{}.jxl",
        std::process::id(),
        effort,
        (d * 10.0) as u32
    );
    let out = Command::new(CJXL_BIN)
        .args(["-d", &d.to_string(), "-e", &effort.to_string(), src, &tmp])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sz = std::fs::metadata(&tmp).ok()?.len() as usize;
    let _ = std::fs::remove_file(&tmp);
    Some(sz)
}

fn srgb_u8_to_linear(rgb_u8: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let lin: Vec<RGB<f32>> = rgb_u8
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    Img::new(lin, w as usize, h as usize)
}

fn main() {
    let params = ButteraugliParams::default();
    println!(
        "class\timage\teffort\tdistance\tcjxl_bytes\tbaseline_bytes\tprod_bytes\tdelta_bytes\tbaseline_pct\tprod_pct\tbaseline_ssim2\tprod_ssim2\tdelta_ssim2\tbaseline_bfly\tprod_bfly\tgate_fires"
    );
    let mut closes = 0;
    let mut regressions = 0;
    let mut worst_ssim2_delta = 0.0f64;
    let mut byte_delta_total = 0i64;
    let n = CELLS.len();
    for (i, &(image, effort, dist, class)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} e{} d{} ({})",
            i + 1,
            n,
            image,
            effort,
            dist,
            class
        );
        let path = PathBuf::from(CID22).join(image);
        let img = image::open(&path).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();
        let orig_linear = srgb_u8_to_linear(&rgb, w, h);
        let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let orig_srgb = Img::new(srgb_pixels, w as usize, h as usize);

        let cjxl_b = cjxl_size(path.to_str().unwrap(), effort, dist).unwrap_or(0);
        let p = known_proxies(image).unwrap();
        let gate = w44_96_would_fire(dist, p);

        let baseline = encode_baseline(&rgb, w, h, effort, dist, p);
        let prod = encode_production(&rgb, w, h, effort, dist);

        let (b_bfly, b_ssim2) = measure_metrics(&baseline, w, h, &orig_linear, &orig_srgb, &params);
        let (p_bfly, p_ssim2) = measure_metrics(&prod, w, h, &orig_linear, &orig_srgb, &params);

        let baseline_pct = if cjxl_b > 0 {
            (baseline.len() as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0
        } else {
            0.0
        };
        let prod_pct = if cjxl_b > 0 {
            (prod.len() as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0
        } else {
            0.0
        };
        let delta_bytes = prod.len() as i64 - baseline.len() as i64;
        let delta_ssim2 = p_ssim2 - b_ssim2;
        byte_delta_total += delta_bytes;

        // FIXED threshold: +3.0%
        let baseline_fixed = baseline_pct <= 3.0;
        let prod_fixed = prod_pct <= 3.0;
        if !baseline_fixed && prod_fixed {
            closes += 1;
        }
        if baseline_fixed && !prod_fixed {
            regressions += 1;
        }
        if delta_ssim2 < worst_ssim2_delta {
            worst_ssim2_delta = delta_ssim2;
        }

        println!(
            "{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{:+.3}\t{:+.3}\t{:.4}\t{:.4}\t{:+.4}\t{:.4}\t{:.4}\t{}",
            class,
            image,
            effort,
            dist,
            cjxl_b,
            baseline.len(),
            prod.len(),
            delta_bytes,
            baseline_pct,
            prod_pct,
            b_ssim2,
            p_ssim2,
            delta_ssim2,
            b_bfly,
            p_bfly,
            if gate { "Z" } else { "-" }
        );
    }
    eprintln!("\n=== W44-98 baseline diff aggregate ===");
    eprintln!("Total cells: {}", n);
    eprintln!("OPEN→FIXED closes: {}", closes);
    eprintln!("FIXED→OPEN regressions: {}", regressions);
    eprintln!("Worst ssim2 delta vs baseline: {:+.4}", worst_ssim2_delta);
    eprintln!("Total byte delta: {:+}B over {} cells", byte_delta_total, n);
}
