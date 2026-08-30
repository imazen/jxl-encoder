// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-228c1 SHIP-cell validation: paired encode-decode bench on the
//! W44-105 SHIP cells comparing `Tier2Knobs::default()` (A) vs
//! `Tier2Knobs::auto_for_distance(Screenshot, distance)` (B).
//!
//! **Why this exists** (per `docs/TIER_2_KNOBS.md` §W44-228c gate +
//! `memory/phase_b_tier2_complete_2026-05-23.md`): every W44-228a
//! per-stratum optimum sets `screenshot_quant_aggressiveness = 0`,
//! which DISABLES the W44-105 buttloop screen-seed lift. W44-105 was a
//! measured SHIP that fixed real SSIM2 wins on terminal / imac_g3 /
//! codec_wiki text at d=4..6, e8+. The W44-228 surface "reroutes" via
//! the other 4 knobs on the W44-219 corpus — but that corpus DID NOT
//! INCLUDE the W44-105 SHIP cells. This bench measures DIRECTLY whether
//! the reroute holds up on those specific cells, resolving belief #18
//! pending-contradiction.
//!
//! **Gate criteria** (from `TIER_2_KNOBS.md` §W44-228c):
//!   (1) Mean Δssim2 across 12 cells: ≤ 0.1 regression
//!   (2) Mean Δbytes_pct across 12 cells: ≤ +1.5%
//!   (3) Per-cell worst-case Δssim2 not worse than −0.5
//!   (4) Per-cell worst-case Δbytes_pct not worse than +5%
//! PASS = all 4. If PASS → W44-228c2 follows (20-anchor bootstrap).
//! If FAIL → W44-228b opt-in API stands as final state.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert parallel butteraugli-loop ssim2-loop' \
//!     --example w44_228c1_ship_cell_validation \
//!     -- --output benchmarks/w44_228c1_ship_cell_validation_2026-05-23.tsv
//!
//! Idempotent: re-running overwrites the TSV in place.

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::__test_exports::coupling::Tier2Knobs;
use jxl_encoder::ImageContentClass;
use jxl_encoder::api::{EncoderStrategy, Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::env;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const GB82_SC: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// W44-105 SHIP cells: 3 images × 4 (effort, distance) pairs = 12 cells.
///
/// The W44-105 fix (`bc994a21`, May 2026) introduced a 4× buttloop
/// screen-seed scale at distance ≥ 3.5 that closed measured SSIM2
/// regressions on these specific cells. See CLAUDE.md "W44-105 buttloop
/// quant field seed scale" investigation note for the full per-cell
/// before/after table.
const SHIP_CELLS: &[(&str, u8, f32)] = &[
    ("terminal.png", 8, 4.0),
    ("terminal.png", 8, 5.0),
    ("terminal.png", 9, 4.0),
    ("terminal.png", 9, 5.0),
    ("imac_g3.png", 8, 4.0),
    ("imac_g3.png", 8, 5.0),
    ("imac_g3.png", 9, 4.0),
    ("imac_g3.png", 9, 5.0),
    ("codec_wiki.png", 8, 4.0),
    ("codec_wiki.png", 8, 5.0),
    ("codec_wiki.png", 9, 4.0),
    ("codec_wiki.png", 9, 5.0),
];

/// Number of timing trials per (cell, mode). Best-of-N reported for
/// encode_ms; bytes/metrics are deterministic so first trial wins.
const TRIALS: usize = 3;

/// One encode run.
///
/// `use_auto_knobs = false` → `Tier2Knobs::default()` (production
/// default — produces byte-identical encode to pre-W44-222 main).
/// `use_auto_knobs = true`  → `Tier2Knobs::auto_for_distance(Screenshot, d)`.
///
/// Both modes pin `EncoderStrategy::Zenjxl` explicitly so the dispatch
/// path is identical except for the knob values.
///
/// **Memory budget**: bumped to 8 GiB (vs default 2 GiB) because W44-93
/// documented imac_g3 e9 d=3 OOMing at the 2 GiB cap with try_dct64
/// widening; W44-228c1 spans imac_g3 at e8/e9 d=4-5 where DCT64
/// evaluation infrastructure × 4 buttloop iters × multi-MP image easily
/// exceeds the default cap. 8 GiB is comfortable on the 128 GB host.
fn encode_once(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    use_auto_knobs: bool,
) -> (Vec<u8>, u128) {
    let knobs = if use_auto_knobs {
        Tier2Knobs::auto_for_distance(ImageContentClass::Screenshot, distance)
    } else {
        Tier2Knobs::default()
    };
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
        .encode(rgb)
        .expect("encode");
    let elapsed_ms = t0.elapsed().as_millis();
    (bytes, elapsed_ms)
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
    eprintln!("# W44-228c1 SHIP-cell validation — Tier2Knobs::default() vs auto_for_distance");
    eprintln!(
        "# {} W44-105 SHIP cells × 2 modes × {} timing trials",
        SHIP_CELLS.len(),
        TRIALS
    );
    eprintln!("# Phase ordering: ALL mode-A encodes FIRST (no override installed) → ");
    eprintln!(
        "#                 ALL mode-B encodes SECOND (single OnceLock install — all SHIP cells"
    );
    eprintln!(
        "#                 fall in ScreenVeryHigh stratum so the same knob tuple covers them all)."
    );
    eprintln!(
        "# This sequencing is REQUIRED — `runtime::install` uses a OnceLock that affects every"
    );
    eprintln!(
        "# encode in the process regardless of `cfg.tier2_knobs`. Interleaving A/B in the same"
    );
    eprintln!("# process would silently corrupt mode-A measurements after the first B encode.");

    // Output destination: --output <path> or stdout (TSV is also echoed to stdout).
    let args: Vec<String> = env::args().collect();
    let out_path = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let mut out_file = out_path.as_ref().map(|p| {
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(p)
            .expect("open output TSV")
    });

    let header = "image\teffort\tdistance\tbytes_A\tbytes_B\tdelta_bytes\tdelta_bytes_pct\tbfly_A\tbfly_B\tdelta_bfly\tssim2_A\tssim2_B\tdelta_ssim2\tencode_ms_A\tencode_ms_B";
    println!("{}", header);
    if let Some(f) = out_file.as_mut() {
        writeln!(f, "{}", header).unwrap();
    }

    let params = ButteraugliParams::default();
    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    // Per-cell storage for both phases.
    #[derive(Default)]
    struct CellResult {
        bytes: usize,
        bfly: f64,
        ssim2: f64,
        ms_best: u128,
    }
    let mut a_results: Vec<CellResult> = Vec::with_capacity(SHIP_CELLS.len());
    let mut b_results: Vec<CellResult> = Vec::with_capacity(SHIP_CELLS.len());

    // ────── Phase 1: ALL mode-A encodes (default knobs) ──────
    eprintln!("\n# Phase 1 / mode A: Tier2Knobs::default() (production)");
    for (idx, &(image, effort, distance)) in SHIP_CELLS.iter().enumerate() {
        let path = PathBuf::from(GB82_SC).join(image);
        let (w, h, raw, orig_linear_img, orig_srgb_img) =
            images_cache.entry(image.to_string()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let linear = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                (w, h, rgb, linear, srgb_img)
            });

        let mut first_bytes: Option<Vec<u8>> = None;
        let mut ms_best = u128::MAX;
        for _ in 0..TRIALS {
            let (bytes, ms) = encode_once(raw, *w, *h, effort, distance, false);
            if first_bytes.is_none() {
                first_bytes = Some(bytes);
            }
            ms_best = ms_best.min(ms);
        }
        let bytes = first_bytes.expect("at least 1 trial");
        let (bfly, ssim2) = compute_metrics(&bytes, orig_linear_img, orig_srgb_img, &params);
        let n = bytes.len();
        a_results.push(CellResult {
            bytes: n,
            bfly,
            ssim2,
            ms_best,
        });
        eprintln!(
            "#   [A {:>2}/{}] {} e{} d={:.1}: {} B, bfly={:.4}, ssim2={:.4}, {} ms",
            idx + 1,
            SHIP_CELLS.len(),
            image,
            effort,
            distance,
            n,
            bfly,
            ssim2,
            ms_best
        );
    }

    // ────── Phase 2: ALL mode-B encodes (auto knobs — single OnceLock install) ──────
    eprintln!(
        "\n# Phase 2 / mode B: Tier2Knobs::auto_for_distance(Screenshot, d) — single OnceLock install"
    );
    for (idx, &(image, effort, distance)) in SHIP_CELLS.iter().enumerate() {
        let (w, h, raw, orig_linear_img, orig_srgb_img) = images_cache
            .get(image)
            .expect("Phase-1 already populated cache");

        let mut first_bytes: Option<Vec<u8>> = None;
        let mut ms_best = u128::MAX;
        for _ in 0..TRIALS {
            let (bytes, ms) = encode_once(raw, *w, *h, effort, distance, true);
            if first_bytes.is_none() {
                first_bytes = Some(bytes);
            }
            ms_best = ms_best.min(ms);
        }
        let bytes = first_bytes.expect("at least 1 trial");
        let (bfly, ssim2) = compute_metrics(&bytes, orig_linear_img, orig_srgb_img, &params);
        let n = bytes.len();
        b_results.push(CellResult {
            bytes: n,
            bfly,
            ssim2,
            ms_best,
        });
        eprintln!(
            "#   [B {:>2}/{}] {} e{} d={:.1}: {} B, bfly={:.4}, ssim2={:.4}, {} ms",
            idx + 1,
            SHIP_CELLS.len(),
            image,
            effort,
            distance,
            n,
            bfly,
            ssim2,
            ms_best
        );
    }

    // ────── Compute per-cell deltas + emit TSV + 4-gate decision ──────
    let mut delta_ssim2_vec: Vec<f64> = Vec::new();
    let mut delta_bytes_pct_vec: Vec<f64> = Vec::new();
    let mut worst_cell_ssim2: (&str, u8, f32, f64) = ("", 0, 0.0, f64::INFINITY);
    let mut worst_cell_bytes: (&str, u8, f32, f64) = ("", 0, 0.0, f64::NEG_INFINITY);

    for (idx, &(image, effort, distance)) in SHIP_CELLS.iter().enumerate() {
        let a = &a_results[idx];
        let b = &b_results[idx];
        let delta_b = b.bytes as i64 - a.bytes as i64;
        let delta_pct = (delta_b as f64) / (a.bytes as f64) * 100.0;
        let delta_bfly = b.bfly - a.bfly;
        let delta_ssim2 = b.ssim2 - a.ssim2;

        if delta_ssim2 < worst_cell_ssim2.3 {
            worst_cell_ssim2 = (image, effort, distance, delta_ssim2);
        }
        if delta_pct > worst_cell_bytes.3 {
            worst_cell_bytes = (image, effort, distance, delta_pct);
        }
        delta_ssim2_vec.push(delta_ssim2);
        delta_bytes_pct_vec.push(delta_pct);

        let row = format!(
            "{}\te{}\t{:.1}\t{}\t{}\t{:+}\t{:+.3}\t{:.4}\t{:.4}\t{:+.4}\t{:.4}\t{:.4}\t{:+.4}\t{}\t{}",
            image,
            effort,
            distance,
            a.bytes,
            b.bytes,
            delta_b,
            delta_pct,
            a.bfly,
            b.bfly,
            delta_bfly,
            a.ssim2,
            b.ssim2,
            delta_ssim2,
            a.ms_best,
            b.ms_best,
        );
        println!("{}", row);
        if let Some(f) = out_file.as_mut() {
            writeln!(f, "{}", row).unwrap();
        }
    }

    let mean_ssim2: f64 = delta_ssim2_vec.iter().sum::<f64>() / delta_ssim2_vec.len() as f64;
    let mean_bytes_pct: f64 =
        delta_bytes_pct_vec.iter().sum::<f64>() / delta_bytes_pct_vec.len() as f64;

    let gate1_pass = mean_ssim2 >= -0.1; // ≤ 0.1 regression
    let gate2_pass = mean_bytes_pct <= 1.5; // ≤ +1.5%
    let gate3_pass = worst_cell_ssim2.3 >= -0.5; // worst not worse than -0.5
    let gate4_pass = worst_cell_bytes.3 <= 5.0; // worst not worse than +5%

    eprintln!("\n# W44-228c gate decision");
    eprintln!(
        "# (1) mean Δssim2 = {:+.4}  [target ≥ -0.1]  → {}",
        mean_ssim2,
        if gate1_pass { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "# (2) mean Δbytes_pct = {:+.4}%  [target ≤ +1.5%]  → {}",
        mean_bytes_pct,
        if gate2_pass { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "# (3) worst Δssim2 = {:+.4} on {} e{} d={:.1}  [target ≥ -0.5]  → {}",
        worst_cell_ssim2.3,
        worst_cell_ssim2.0,
        worst_cell_ssim2.1,
        worst_cell_ssim2.2,
        if gate3_pass { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "# (4) worst Δbytes_pct = {:+.4}% on {} e{} d={:.1}  [target ≤ +5.0%]  → {}",
        worst_cell_bytes.3,
        worst_cell_bytes.0,
        worst_cell_bytes.1,
        worst_cell_bytes.2,
        if gate4_pass { "PASS" } else { "FAIL" }
    );
    let overall_pass = gate1_pass && gate2_pass && gate3_pass && gate4_pass;
    eprintln!(
        "# OVERALL: {} → recommendation: {}",
        if overall_pass { "PASS" } else { "FAIL" },
        if overall_pass {
            "SHIP-W44-228c2 (20-anchor bootstrap re-measurement)"
        } else {
            "KEEP-OPT-IN-ONLY (W44-228b state stands as final)"
        }
    );
}
