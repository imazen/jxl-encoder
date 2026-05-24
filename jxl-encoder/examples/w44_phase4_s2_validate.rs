// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-PHASE4-S2-validate: production-cell verification of W44-PHASE4-S2-refit
//! (commit `2991bb27`) per-stratum Tier2Knobs lookup.
//!
//! **Hypothesis** (from W44-PHASE4-S2-refit memo): the refit lookup
//! actually shifts encoded bytes on production-size cells from each
//! of the 8 strata. Hash-locks alone (synthetic ≤48² fixtures) can't
//! confirm this — the Tier-2 auto-discriminator doesn't engage on
//! tiny images.
//!
//! **Mechanism** (single-process per cell to dodge the OnceLock
//! `runtime::install_or_check_idempotent`):
//! - Mode A: `Tier2Knobs::default()` (production default; expansion
//!   → `RuntimeTuning::default()` → no install fires → byte-identical
//!   to the no-knobs encode path).
//! - Mode B: `Tier2Knobs::auto_for_distance(class, distance)` using
//!   the on-disk W44-PHASE4-S2 lookup. Non-default tuple →
//!   `RuntimeTuning` is installed → encoder takes the override path.
//!
//! For each invocation, this binary encodes a SINGLE (image, effort,
//! distance, mode, class) tuple and emits ONE tab-separated row to
//! stdout (or appends to `--output`). The harness shell loop invokes
//! this binary 16 times (8 cells × 2 modes), aggregates the TSV.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo build -p jxl-encoder --release \
//!     --features '__expert tuning-override parallel butteraugli-loop ssim2-loop' \
//!     --example w44_phase4_s2_validate
//!
//! Invoke (single cell):
//!   target/release/examples/w44_phase4_s2_validate \
//!     --image <path> --effort 8 --distance 4.0 --mode B \
//!     --class screen --stratum-name screen/very_high \
//!     --append benchmarks/w44_phase4_s2_validate_2026-05-24.tsv

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, Limits, LossyConfig, PixelLayout};
use jxl_encoder::effort::ImageContentClass;
use jxl_encoder::tuning::coupling::Tier2Knobs;
use rgb::RGB;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Number of timing trials per encode. Best wall reported; bytes /
/// metrics are deterministic so first trial wins.
const TRIALS: usize = 3;

fn parse_arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
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

fn compute_metrics(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> (f64, f64) {
    if let Some((dw, dh, dec)) = decode_jxl_linear(bytes) {
        let dec_pixels: Vec<RGB<f32>> = dec
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let dec_linear_img = Img::new(dec_pixels, dw, dh);
        let bfly = butteraugli_linear(orig_linear.as_ref(), dec_linear_img.as_ref(), params)
            .map(|r| r.score as f64)
            .unwrap_or(f64::NAN);
        let dec_srgb: Vec<[u8; 3]> = dec
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
    } else {
        (f64::NAN, f64::NAN)
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let image_path = parse_arg(&args, "--image").expect("--image <path>");
    let effort: u8 = parse_arg(&args, "--effort")
        .expect("--effort <0..9>")
        .parse()
        .expect("effort int");
    let distance: f32 = parse_arg(&args, "--distance")
        .expect("--distance <f32>")
        .parse()
        .expect("distance f32");
    let mode = parse_arg(&args, "--mode").expect("--mode <A|B>");
    let class_str = parse_arg(&args, "--class").expect("--class <screen|photo>");
    let stratum_name = parse_arg(&args, "--stratum-name").unwrap_or("unknown");
    let append_path = parse_arg(&args, "--append");
    let print_header = args.iter().any(|a| a == "--print-header");

    let class = match class_str {
        "screen" | "Screenshot" | "screenshot" => ImageContentClass::Screenshot,
        "photo" | "Photo" => ImageContentClass::Photo,
        other => panic!("unknown --class {}", other),
    };

    let knobs = match mode {
        "A" | "default" => Tier2Knobs::default(),
        "B" | "auto" => Tier2Knobs::auto_for_distance(class, distance),
        other => panic!("unknown --mode {}; expected A or B", other),
    };

    // Load image.
    let path = PathBuf::from(image_path);
    let img = image::open(&path).expect("decode png");
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    let linear = srgb_u8_to_linear(&rgb, w, h);
    let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);

    // Encode (best-of-TRIALS for wall, first encode for bytes).
    let mut first_bytes: Option<Vec<u8>> = None;
    let mut ms_best: u128 = u128::MAX;
    for _ in 0..TRIALS {
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_threads(8)
            .with_knobs(knobs);
        let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024 * 1024);
        let t0 = Instant::now();
        let bytes = cfg
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&limits)
            .encode(&rgb)
            .expect("encode");
        let ms = t0.elapsed().as_millis();
        if first_bytes.is_none() {
            first_bytes = Some(bytes);
        }
        ms_best = ms_best.min(ms);
    }
    let bytes = first_bytes.expect("at least 1 trial");
    let n_bytes = bytes.len();
    let sha256 = {
        let mut h = Sha256::new();
        h.update(&bytes);
        let digest = h.finalize();
        let mut s = String::with_capacity(digest.len() * 2);
        for b in digest.iter() {
            use std::fmt::Write;
            write!(&mut s, "{:02x}", b).unwrap();
        }
        s
    };

    let params = ButteraugliParams::default();
    let (bfly, ssim2) = compute_metrics(&bytes, &linear, &srgb_img, &params);

    // image_basename for compact TSV.
    let image_basename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let header = "stratum\timage\teffort\tdistance\tmode\tclass\tbytes\tbfly\tssim2\tencode_ms\tsha256_8";
    let sha8 = &sha256[..16];
    let row = format!(
        "{stratum}\t{img}\te{eff}\t{dist:.2}\t{mode}\t{cls}\t{bytes}\t{bfly:.5}\t{ssim2:.5}\t{ms}\t{sha8}",
        stratum = stratum_name,
        img = image_basename,
        eff = effort,
        dist = distance,
        mode = mode,
        cls = class_str,
        bytes = n_bytes,
        bfly = bfly,
        ssim2 = ssim2,
        ms = ms_best,
        sha8 = sha8,
    );

    if print_header {
        println!("{}", header);
    }
    println!("{}", row);

    if let Some(p) = append_path {
        let needs_header = !std::path::Path::new(p).exists();
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .expect("open append");
        if needs_header {
            writeln!(f, "{}", header).unwrap();
        }
        writeln!(f, "{}", row).unwrap();
    }
}
