// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #25 retry: K_AC_QUANT (`k_ac_quant`) A/B vs default 0.765 → picker
//! oracle's claimed 0.65 win on `EncoderStrategy::Zenjxl`.
//!
//! Verifies whether K_AC_QUANT is scaling-only (Hypothesis B from the first
//! agent's intermediate description: would make A/B byte-identical because
//! `global_scale` consumes the inverse) OR a real RD lever (Hypothesis A from
//! the picker oracle finding: cells produce different bytes/quality).
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo build -p jxl-encoder --release \
//!     --features '__expert parallel butteraugli-loop ssim2-loop' \
//!     --example issue_25_k_ac_quant_retry
//!
//! Per cell, encodes paired-interleaved A (0.765) and B (0.65) for SAMPLES
//! iterations and emits one TSV row per (cell, mode, sample) tuple.

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{EncoderStrategy, Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const SAMPLES: usize = 6;

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

struct CellSpec<'a> {
    name: &'a str,
    image_path: &'a str,
    effort: u8,
    distance: f32,
}

const CELLS: &[CellSpec<'static>] = &[
    CellSpec {
        name: "1418519_e7_d1",
        image_path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        effort: 7,
        distance: 1.0,
    },
    CellSpec {
        name: "codec_wiki_e7_d2",
        image_path: "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        effort: 7,
        distance: 2.0,
    },
    CellSpec {
        name: "terminal_e8_d4",
        image_path: "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        effort: 8,
        distance: 4.0,
    },
    CellSpec {
        name: "1531677_e9_d5",
        image_path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        effort: 9,
        distance: 5.0,
    },
];

fn encode_once(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    k_ac_quant: f32,
) -> (Vec<u8>, u128) {
    let mut params = LossyInternalParams::default();
    params.k_ac_quant = Some(k_ac_quant);
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_threads(8)
        .with_internal_params(params);
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let output_path = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .expect("--output <tsv>");

    let mut out = OpenOptions::new()
        .create(true)
        .append(false)
        .truncate(true)
        .write(true)
        .open(output_path)
        .expect("open output");
    writeln!(
        out,
        "cell\tsample\tmode\tk_ac_quant\tbytes\tbfly\tssim2\tencode_ms\tsha256_8"
    )
    .unwrap();

    let bp = ButteraugliParams::default();

    for cell in CELLS {
        eprintln!("[cell] {} → loading {}", cell.name, cell.image_path);
        let img = image::open(PathBuf::from(cell.image_path)).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();
        let linear = srgb_u8_to_linear(&rgb, w, h);
        let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);

        for sample in 0..SAMPLES {
            // Interleave A then B per sample to share thermal/system state.
            for (mode, k) in &[("A", 0.765_f32), ("B", 0.65_f32)] {
                let (bytes, ms) = encode_once(&rgb, w, h, cell.effort, cell.distance, *k);
                let (bfly, ssim2) = compute_metrics(&bytes, &linear, &srgb_img, &bp);
                let sha = sha256_hex(&bytes);
                let row = format!(
                    "{cell}\t{sample}\t{mode}\t{k:.4}\t{bytes}\t{bfly:.5}\t{ssim2:.5}\t{ms}\t{sha8}",
                    cell = cell.name,
                    sample = sample,
                    mode = mode,
                    k = k,
                    bytes = bytes.len(),
                    bfly = bfly,
                    ssim2 = ssim2,
                    ms = ms,
                    sha8 = &sha[..16],
                );
                println!("{}", row);
                writeln!(out, "{}", row).unwrap();
                out.flush().unwrap();
            }
        }
    }
    eprintln!("[done] wrote {}", output_path);
}
