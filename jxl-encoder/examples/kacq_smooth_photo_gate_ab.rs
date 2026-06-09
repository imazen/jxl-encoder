// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #25 follow-on B: smooth-photo `k_ac_quant = 0.65` gate —
//! probe + paired A/B bench.
//!
//! Two subcommands:
//!
//! `--probe` — print the PRODUCTION [`ZenanalyzeProxies`] (the exact
//! struct the encoder consumes: m3_colourfulness, flat_color_block_ratio,
//! edge_density, luma_var) for the candidate/rejected/screenshot image
//! set, so discriminator thresholds are derived from the same numbers
//! the gate would see at runtime.
//!
//! `--bench --output <tsv>` — paired-interleaved A/B over the protocol
//! grid (images × efforts {5,7,8} × distances {0.5,1,2,3,4,5}):
//!   A = k_ac_quant 0.765 (libjxl default)
//!   B = k_ac_quant 0.65  (candidate smooth-photo value)
//! via `LossyInternalParams` (`__expert`), `EncoderStrategy::Zenjxl`.
//! Per cell: bytes, butteraugli, ssim2, encode_ms, sha256 prefix.
//! SAMPLES=1 — the encoder is deterministic per (cell, mode); the
//! 2026-05-25 retry proved byte-identical output across 6 samples.
//!
//! Build:
//!   cargo build -p jxl-encoder --release \
//!     --features '__expert __pre_quantized parallel butteraugli-loop ssim2-loop' \
//!     --example kacq_smooth_photo_gate_ab

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::__pre_quantized::ZenanalyzeProxies;
use jxl_encoder::api::{EncoderStrategy, Limits, LossyConfig, PixelLayout};
use jxl_encoder::effort::LossyInternalParams;
use rgb::RGB;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// (class, name, path). Classes are hypothesis labels from W44-96 probe
/// data, NOT the gate predicate (the predicate is derived from this
/// run's measurement):
///   PHOTO_LOWEDGE   — smooth-masking, LOW Sobel edge_density photos
///                     (the issue follow-on B "smooth photo" candidates)
///   PHOTO_TEXTURED  — smooth-masking but HIGH edge_density photos
///   PHOTO_DETAILED  — high-masking detailed photo (1418519, mask=92)
///   SCREENSHOT      — gb82-sc screens (rejected class for THIS gate;
///                     screens already have their own W44 lift stack)
struct ImageSpec {
    class: &'static str,
    name: &'static str,
    path: &'static str,
}

/// Bench set: every image gets the full (effort × distance) grid.
const BENCH_IMAGES: &[ImageSpec] = &[
    // Smooth low-edge photos (candidate admitted class per follow-on B)
    ImageSpec {
        class: "PHOTO_LOWEDGE",
        name: "2389166",
        path: "2389166.png",
    },
    ImageSpec {
        class: "PHOTO_LOWEDGE",
        name: "1044329",
        path: "1044329.png",
    },
    ImageSpec {
        class: "PHOTO_LOWEDGE",
        name: "7062219",
        path: "7062219.png",
    },
    // Smooth textured photos (catastrophic in the 2026-05-25 sweep)
    ImageSpec {
        class: "PHOTO_TEXTURED",
        name: "1531677",
        path: "1531677.png",
    },
    ImageSpec {
        class: "PHOTO_TEXTURED",
        name: "1420710",
        path: "1420710.png",
    },
    // Mid/detailed photos (1025469 catastrophic at d=4; 1418519 borderline)
    ImageSpec {
        class: "PHOTO_MID",
        name: "1025469",
        path: "1025469.png",
    },
    ImageSpec {
        class: "PHOTO_DETAILED",
        name: "1418519",
        path: "1418519.png",
    },
    // Screenshots (4+ per protocol)
    ImageSpec {
        class: "SCREENSHOT",
        name: "codec_wiki",
        path: "codec_wiki.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "terminal",
        path: "terminal.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "imac_g3",
        path: "imac_g3.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "windows95",
        path: "windows95.png",
    },
];

/// Probe-only extras for proxy-distribution context (false-fire risk on
/// the wider CID22 photo population).
const PROBE_EXTRAS: &[ImageSpec] = &[
    ImageSpec {
        class: "PHOTO_CONTEXT",
        name: "1189261",
        path: "1189261.png",
    },
    ImageSpec {
        class: "PHOTO_CONTEXT",
        name: "3637739",
        path: "3637739.png",
    },
    ImageSpec {
        class: "PHOTO_CONTEXT",
        name: "297394",
        path: "297394.png",
    },
    ImageSpec {
        class: "PHOTO_CONTEXT",
        name: "2775196",
        path: "2775196.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "graph",
        path: "graph.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "imac_dark",
        path: "imac_dark.png",
    },
];

const EFFORTS: &[u8] = &[5, 7, 8];
const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 3.0, 4.0, 5.0];

fn full_path(spec: &ImageSpec) -> PathBuf {
    if spec.class == "SCREENSHOT" {
        PathBuf::from(GB82).join(spec.path)
    } else {
        PathBuf::from(CID22).join(spec.path)
    }
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
        let dec_pixels: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
        let dec_linear_img = Img::new(dec_pixels, dw, dh);
        let bfly = butteraugli_linear(orig_linear.as_ref(), dec_linear_img.as_ref(), params)
            .map(|r| r.score)
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

fn encode_once(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    k_ac_quant: Option<f32>,
) -> (Vec<u8>, u128) {
    let mut cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_threads(8);
    if let Some(k) = k_ac_quant {
        let mut params = LossyInternalParams::default();
        params.k_ac_quant = Some(k);
        cfg = cfg.with_internal_params(params);
    }
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024 * 1024);
    let t0 = Instant::now();
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(rgb)
        .expect("encode");
    let ms = t0.elapsed().as_millis();
    (bytes, ms)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest.iter() {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

fn run_probe() {
    println!(
        "# kacq probe — PRODUCTION ZenanalyzeProxies (the exact values the gate would consume)"
    );
    println!("class\timage\tw\th\tm3_colourfulness\tfcbr\tedge_density\tluma_var");
    for spec in BENCH_IMAGES.iter().chain(PROBE_EXTRAS.iter()) {
        let p = full_path(spec);
        if !p.exists() {
            eprintln!("MISSING: {}", p.display());
            continue;
        }
        let img = image::open(&p).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();
        let prox = ZenanalyzeProxies::compute_srgb_u8(&rgb, w as usize, h as usize, 3, 0, 1, 2);
        println!(
            "{}\t{}\t{}\t{}\t{:.3}\t{:.5}\t{:.5}\t{:.1}",
            spec.class,
            spec.name,
            w,
            h,
            prox.m3_colourfulness,
            prox.flat_color_block_ratio,
            prox.edge_density,
            prox.luma_var,
        );
    }
}

/// Bench modes:
/// - "explore": A = internal_params k=0.765, B = internal_params k=0.65.
///   Pure RD measurement of the lever; no production gate involved.
/// - "gate": A = production default with env `JXL_KACQ_SMOOTH_GATE_DISABLE=1`
///   set by the CALLER before launching, B = production default. Used to
///   verify the shipped gate's firing set (only valid after the gate
///   lands; both modes run in the same process so the env must be set
///   per-process — the wrapper script runs two processes).
fn run_bench(output_path: &str) {
    let mut out = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(output_path)
        .expect("open output");
    writeln!(
        out,
        "class\timage\teffort\tdistance\tmode\tk_ac_quant\tbytes\tbfly\tssim2\tencode_ms\tsha256_8"
    )
    .unwrap();

    let bp = ButteraugliParams::default();

    for spec in BENCH_IMAGES {
        let p = full_path(spec);
        eprintln!("[image] {} ({})", spec.name, p.display());
        let img = image::open(&p).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();
        let linear = srgb_u8_to_linear(&rgb, w, h);
        let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);

        for &effort in EFFORTS {
            for &distance in DISTANCES {
                // Interleaved A then B (paired; encoder is deterministic).
                for (mode, k) in &[("A", 0.765_f32), ("B", 0.65_f32)] {
                    let (bytes, ms) = encode_once(&rgb, w, h, effort, distance, Some(*k));
                    let (bfly, ssim2) = compute_metrics(&bytes, &linear, &srgb_img, &bp);
                    let sha = sha256_hex(&bytes);
                    let row = format!(
                        "{}\t{}\t{}\t{:.1}\t{}\t{:.4}\t{}\t{:.5}\t{:.5}\t{}\t{}",
                        spec.class,
                        spec.name,
                        effort,
                        distance,
                        mode,
                        k,
                        bytes.len(),
                        bfly,
                        ssim2,
                        ms,
                        &sha[..16],
                    );
                    println!("{}", row);
                    writeln!(out, "{}", row).unwrap();
                    out.flush().unwrap();
                }
            }
        }
    }
    eprintln!("[done] wrote {}", output_path);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--probe") {
        run_probe();
        return;
    }
    if args.iter().any(|a| a == "--bench") {
        let output_path = args
            .iter()
            .position(|a| a == "--output")
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
            .expect("--output <tsv>");
        run_bench(output_path);
        return;
    }
    eprintln!("usage: kacq_smooth_photo_gate_ab --probe | --bench --output <tsv>");
    std::process::exit(2);
}
