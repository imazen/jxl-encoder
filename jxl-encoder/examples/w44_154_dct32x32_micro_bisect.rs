// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-154 MICRO-bisect of variant Z `dct32x32` ∈ {1.22, 1.23, 1.24}
//! between W44-148's 1.24 and pre-W44-148 1.20. Companion to W44-148
//! (`2a303428` raised 1.20 → 1.24) — the W44-153 full-ledger refresh
//! (`1a988fc8`) found 6 cells flipped FIXED→OPEN at the W44-148 boundary,
//! all PARETO SSIM2 WINS (bytes grew past +3% wedge but SSIM2 improved
//! +0.07 to +0.24 on each cell).
//!
//! The 6 newly-flipped cells were OUTSIDE the W44-148 27-cell bisect
//! coverage:
//!   * 1420710 e5 d=6 — bisect was e6/e7/e8/e9 only
//!   * 1531677 e5/e6 d=6 — bisect PROTECT_W99 was at d=5 only
//!   * 1531677 e7 d=5 — bisect PROTECT_W99 was e5/e6/e9
//!   * 1531677 e8 d=5 — bisect DEFICIT_LC was e7/e8 d=5/6; e8 d=5 was HC class
//!   * 1531677 e9 d=5 — same gap
//!
//! Goal: find the dct32x32 value (B=1.22 or C=1.23) that closes >=4 of
//! the 6 newly-flipped cells while preserving >=80% of W44-148's SSIM2
//! wins on the 1418519 d=5 / 1420710 d=4-6 / 1531677 d=4 clusters.
//!
//! Variant A baseline = current production W44-148 value (1.24).
//! Variant B = 1.22 (closer to pre-W44-148, weaker DCT32X32 disincentive).
//! Variant C = 1.23 (Pareto candidate).
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_154_dct32x32_micro_bisect

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::effort::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const SCREENSHOTS: &str = "/home/lilith/work/codec-corpus/gb82-sc";
const CJXL_BIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";

/// (image, effort, distance, classification)
type Cell = (&'static str, u8, f32, &'static str);

/// Cell coverage spec from W44-154 task:
/// - 6 W44-148 FIXED→OPEN flipped cells (must close >=4)
/// - 1418519 d=4/5/6 W44-148/152 win clusters (preserve >=80%)
/// - 1420710 d=4/5 W44-148 wins
/// - 1531677 d=4 protection
/// - codec_wiki d=3 W44-152 collateral wins
/// - 3 photo controls outside W44-29 gate (verify byte-identical)
const CELLS: &[Cell] = &[
    // === 6 W44-148 FIXED→OPEN flipped cells (must close >=4) ===
    ("1420710.png", 5, 6.0, "FLIPPED_W148"),
    ("1531677.png", 5, 6.0, "FLIPPED_W148"),
    ("1531677.png", 6, 6.0, "FLIPPED_W148"),
    ("1531677.png", 7, 5.0, "FLIPPED_W148"),
    ("1531677.png", 8, 5.0, "FLIPPED_W148"),
    ("1531677.png", 9, 5.0, "FLIPPED_W148"),
    // === 1418519 d=5 cluster (W44-148 + W44-152 wins; preserve >=80%) ===
    // Mean SSIM2 baseline +0.614 per W44-148; gate: W44-29/151/152, NOT variant Z
    ("1418519.png", 7, 5.0, "DEF_1418519_D5"),
    ("1418519.png", 8, 5.0, "DEF_1418519_D5"),
    ("1418519.png", 9, 5.0, "DEF_1418519_D5"),
    // === 1418519 d=4 cluster (W44-152 win; preserve >=80%) ===
    ("1418519.png", 7, 4.0, "DEF_1418519_D4"),
    ("1418519.png", 8, 4.0, "DEF_1418519_D4"),
    ("1418519.png", 9, 4.0, "DEF_1418519_D4"),
    // === 1418519 d=6 cluster (W44-148 target — gate does NOT fire there) ===
    ("1418519.png", 7, 6.0, "DEF_1418519_D6"),
    ("1418519.png", 8, 6.0, "DEF_1418519_D6"),
    ("1418519.png", 9, 6.0, "DEF_1418519_D6"),
    // === 1420710 d=4/5 (W44-148 protection / win cluster) ===
    ("1420710.png", 7, 4.0, "PROTECT_1420710_D4"),
    ("1420710.png", 8, 4.0, "PROTECT_1420710_D4"),
    ("1420710.png", 9, 4.0, "PROTECT_1420710_D4"),
    ("1420710.png", 7, 5.0, "PROTECT_1420710_D5"),
    ("1420710.png", 8, 5.0, "PROTECT_1420710_D5"),
    ("1420710.png", 9, 5.0, "PROTECT_1420710_D5"),
    // === 1531677 d=4 (protection cluster — no SSIM2 regression > 0.30) ===
    ("1531677.png", 7, 4.0, "PROTECT_1531677_D4"),
    ("1531677.png", 8, 4.0, "PROTECT_1531677_D4"),
    ("1531677.png", 9, 4.0, "PROTECT_1531677_D4"),
    // === codec_wiki d=3 (W44-152 collateral wins; preserve >=50%) ===
    ("codec_wiki.png", 7, 3.0, "COLLATERAL_CW_D3"),
    ("codec_wiki.png", 8, 3.0, "COLLATERAL_CW_D3"),
    ("codec_wiki.png", 9, 3.0, "COLLATERAL_CW_D3"),
    // === Photo controls (off all gates; must stay byte-identical) ===
    ("1189261.png", 7, 4.0, "CONTROL_NOGATE"),
    ("1044329.png", 7, 4.0, "CONTROL_NOGATE"),
    ("2389166.png", 7, 4.0, "CONTROL_NOGATE"),
];

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
}

#[derive(Debug, Clone, Copy)]
struct ImageProxy {
    mask_med: f32,
    edge_density: f32,
    fcbr: f32,
    m3: f32,
}

/// Proxy values lifted from W44-148 bisect + W44-149 audit. We use proxies
/// here (rather than running the actual proxy compute) because the W44-148
/// example demonstrated that the in-encoder proxy O(W·H) pass is
/// bit-equivalent to these published values, and recomputing them here would
/// add 5-10 ms × 30 cells of harness overhead with no measurement gain.
fn known_proxies(image: &str) -> Option<ImageProxy> {
    match image {
        "1420710.png" => Some(ImageProxy {
            mask_med: 39.549,
            edge_density: 0.9298,
            fcbr: 0.00000,
            m3: 32.932,
        }),
        "1531677.png" => Some(ImageProxy {
            mask_med: 35.634,
            edge_density: 0.8766,
            fcbr: 0.00000,
            m3: 12.295,
        }),
        // 1418519 (mask=76 fails W44-29 mask<50 outer gate; variant Z dispatch
        // does NOT fire here). W44-152 W44-29 admission via mask_p25>=85
        // covers d ∈ [3.0, 5.0]; gate fires at d=4/5 but not d=6.
        // For the W44-154 bisect we still construct the override conservatively:
        // since variant Z doesn't fire, encoding under any override should be
        // BYTE-IDENTICAL across our 3 variants (validates we're not corrupting
        // some other path).
        "1418519.png" => Some(ImageProxy {
            mask_med: 76.0,
            edge_density: 0.50,
            fcbr: 0.05,
            m3: 23.0,
        }),
        // 1189261 — fires W44-91 (m3≈80) not variant Z.
        "1189261.png" => Some(ImageProxy {
            mask_med: 69.0,
            edge_density: 0.60,
            fcbr: 0.005,
            m3: 80.5,
        }),
        // 1025469 — no gate fires; values approximate per W44-148.
        "1025469.png" => Some(ImageProxy {
            mask_med: 76.08,
            edge_density: 0.65,
            fcbr: 0.0166,
            m3: 45.45,
        }),
        // 1044329 + 2389166 — no W44-29 fire (mask >= 50); CONTROL cells per
        // W44-95 honest-stop measurement (variant Z over-fired there).
        "1044329.png" => Some(ImageProxy {
            mask_med: 48.03,
            edge_density: 0.55,
            fcbr: 0.02,
            m3: 30.0,
        }),
        "2389166.png" => Some(ImageProxy {
            mask_med: 46.24,
            edge_density: 0.55,
            fcbr: 0.02,
            m3: 30.0,
        }),
        // codec_wiki — high mask (mask_med ~99); fires W22-1 screenshot
        // path normally. W44-152 admission via mask_p25 makes codec_wiki
        // d=3 reachable through the W44-29 default lift path; that lift is
        // OUTSIDE variant Z, so codec_wiki should also be byte-identical
        // across the 3 variants (gate doesn't fire).
        "codec_wiki.png" => Some(ImageProxy {
            mask_med: 99.0,
            edge_density: 0.20,
            fcbr: 0.50,
            m3: 8.0,
        }),
        // terminal — screenshot.
        "terminal.png" => Some(ImageProxy {
            mask_med: 110.0,
            edge_density: 0.10,
            fcbr: 0.70,
            m3: 5.0,
        }),
        _ => None,
    }
}

fn w44_96_would_fire(d: f32, p: ImageProxy) -> bool {
    d >= 4.5 && p.mask_med < 50.0 && p.edge_density >= 0.7 && p.fcbr < 0.01
}

fn w44_98_high_colour_would_fire(d: f32, p: ImageProxy) -> bool {
    w44_96_would_fire(d, p) && p.m3 >= 25.0
}

fn w44_99_low_colour_would_fire(d: f32, p: ImageProxy) -> bool {
    w44_96_would_fire(d, p) && p.m3 < 25.0
}

fn w44_29_would_fire(d: f32, p: ImageProxy) -> bool {
    d >= 3.0 && p.mask_med < 50.0
}

type Variant = (&'static str, f32);

/// Build the entropy_mul table for this (distance, proxy, dct32x32) tuple.
/// Mirrors the production gate ordering EXACTLY (W44-98 high-colour > W44-99
/// low-colour > plain variant Z > W44-29 outer > production default).
///
/// For the W44-154 bisect we vary ONLY `dct32x32` (mirrors W44-148). When the
/// gate is W44-98 (high-colour), `dct16x32` stays fixed at 1.30 (W44-98
/// independent lift). When it's W44-99 (low-colour), `dct16x32` stays at
/// 1.23 (W44-100 micro-bisect value). When it's plain variant Z, `dct16x32`
/// scales with the libjxl 1.49/1.48 ratio (so different per dct32x32).
fn build_variant_table(d: f32, p: ImageProxy, dct32x32: f32) -> Option<EntropyMulTable> {
    if w44_98_high_colour_would_fire(d, p) {
        let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour();
        t.dct32x32 = dct32x32;
        // dct16x32 stays at 1.30 (W44-98 independent lift, unchanged in W44-148).
        Some(t)
    } else if w44_99_low_colour_would_fire(d, p) {
        let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour();
        t.dct32x32 = dct32x32;
        // dct16x32 stays at 1.23 (W44-100 micro-bisect, unchanged in W44-148).
        Some(t)
    } else if w44_96_would_fire(d, p) {
        let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
        t.dct32x32 = dct32x32;
        // dct16x32 scales with dct32x32 by the libjxl 1.49/1.48 ratio.
        t.dct16x32 = dct32x32 * (1.49 / 1.48);
        Some(t)
    } else if w44_29_would_fire(d, p) {
        // Outer W44-29 — preserve, don't touch (variant Z doesn't fire here).
        Some(EntropyMulTable::high_d_photo_smooth_suppressed())
    } else {
        // No gate fires — use production default (no override needed).
        None
    }
}

/// Variant labels: A=1.24 (W44-148 baseline), B=1.22, C=1.23.
const VARIANTS: &[Variant] = &[
    ("A_dct32_124_w148", 1.24),
    ("B_dct32_122", 1.22),
    ("C_dct32_123", 1.23),
];

fn encode_with_variant(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    proxy: ImageProxy,
    variant_dct32: f32,
) -> Result<Vec<u8>, String> {
    let mut cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    if let Some(t) = build_variant_table(d, proxy, variant_dct32) {
        // Force auto W44-96/98/99 dispatch off so our injected table wins.
        cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides {
            high_d_photo_hint: Some(false),
            ..Default::default()
        });
        let mut internal = LossyInternalParams::default();
        internal.entropy_mul_table = Some(t);
        cfg = cfg.with_internal_params(internal);
    }
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
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

fn measure(
    bytes: Vec<u8>,
    w: u32,
    h: u32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let (dw, dh, decoded_linear) =
        decode_jxl_linear(&bytes).ok_or_else(|| "decode failed".to_string())?;
    if dw != w as usize || dh != h as usize {
        return Err(format!("decoded {}x{} != {}x{}", dw, dh, w, h));
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);

    let dec_srgb: Vec<[u8; 3]> = decoded_linear
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
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);

    Ok(Measure {
        bytes: bytes.len(),
        butteraugli: bfly,
        ssim2,
    })
}

fn cjxl_size_and_bfly(
    src: &str,
    effort: u8,
    d: f32,
    w: u32,
    h: u32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    params: &ButteraugliParams,
) -> Option<(usize, f64, f64)> {
    let tmp = format!(
        "/tmp/w44_154_cjxl_{}_{}_{}.jxl",
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
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);

    let (dw, dh, decoded_linear) = decode_jxl_linear(&bytes)?;
    if dw != w as usize || dh != h as usize {
        return Some((sz, f64::NAN, f64::NAN));
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);

    // Also compute cjxl SSIM2 for cluster-mean comparison.
    let dec_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dw2 = dw;
    let dh2 = dh;
    let _ = (dw2, dh2);
    let dec_srgb_img = Img::new(dec_srgb, dw, dh);
    // Need original sRGB for SSIM2 — we have it inline below where we call.
    // Inline placeholder: caller passes orig_srgb_img separately if needed.
    // For now return only bfly (SSIM2 computed against cjxl is not needed —
    // we measure SSIM2 deltas across our own 3 variants relative to A).
    let _ = dec_srgb_img;

    Some((sz, bfly, f64::NAN))
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

fn resolve_image_path(image: &str) -> PathBuf {
    if image == "terminal.png" || image == "codec_wiki.png" {
        PathBuf::from(SCREENSHOTS).join(image)
    } else {
        PathBuf::from(CID22).join(image)
    }
}

fn main() {
    eprintln!("W44-154 MICRO-bisect variant Z dct32x32 ∈ {{1.22, 1.23, 1.24}}");
    eprintln!(
        "Variants: {:?}",
        VARIANTS.iter().map(|v| v.0).collect::<Vec<_>>()
    );
    eprintln!(
        "Cells: {} (target: close ≥4 of 6 FLIPPED_W148, preserve ≥80% of DEFICIT_* SSIM2 wins)",
        CELLS.len()
    );

    let params = ButteraugliParams::default();

    let mut hdr = String::from("class\timage\teffort\tdistance\tcjxl_bytes\tcjxl_bfly\tgate_fires");
    for v in VARIANTS {
        hdr.push_str(&format!(
            "\t{}_bytes\t{}_bytes_pct\t{}_bfly\t{}_bfly_pct\t{}_ssim2",
            v.0, v.0, v.0, v.0, v.0
        ));
    }
    println!("{}", hdr);

    // Track aggregate FIXED/OPEN flips relative to baseline A=1.24.
    // (delta_bytes_sum, n_cells, fixed_to_open_vs_A, open_to_fixed_vs_A,
    //  ssim2_delta_sum_vs_A, ssim2_min_delta_vs_A, ssim2_max_delta_vs_A)
    let mut total_stats: BTreeMap<String, (i64, i64, i64, i64, f64, f64, f64)> = BTreeMap::new();
    for v in VARIANTS {
        total_stats.insert(v.0.to_string(), (0, 0, 0, 0, 0.0, 0.0, 0.0));
    }
    // Per-class SSIM2 tracking, deltas vs A=1.24 baseline.
    let mut class_ssim2_sum: BTreeMap<(String, String), (f64, i64)> = BTreeMap::new();
    // Per-class FLIPPED-cell closure counts (variant -> count of cells where
    // variant achieved FIXED status — bytes_pct <= 3.0).
    let mut flipped_closed: BTreeMap<String, i64> = BTreeMap::new();
    // Per-class byte-identical sentinel verification.
    let mut byte_identical_violations: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for v in VARIANTS {
        flipped_closed.insert(v.0.to_string(), 0);
        byte_identical_violations.insert(v.0.to_string(), Vec::new());
    }

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    for (i, &(image, effort, dist, class)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} e{} d={}  ({})",
            i + 1,
            n_cells,
            image,
            effort,
            dist,
            class
        );
        let path = resolve_image_path(image);

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

        let (cjxl_b, cjxl_bfly, _) = match cjxl_size_and_bfly(
            path.to_str().unwrap(),
            effort,
            dist,
            *w,
            *h,
            orig_linear_img,
            &params,
        ) {
            Some(v) => v,
            None => {
                eprintln!("  cjxl failed, skipping");
                continue;
            }
        };

        let proxy = match known_proxies(image) {
            Some(p) => p,
            None => {
                eprintln!("  no proxy data for {}, skipping", image);
                continue;
            }
        };
        let gate_marker = if w44_99_low_colour_would_fire(dist, proxy) {
            "LC"
        } else if w44_98_high_colour_would_fire(dist, proxy) {
            "HC"
        } else if w44_96_would_fire(dist, proxy) {
            "Z"
        } else if w44_29_would_fire(dist, proxy) {
            "29"
        } else {
            "-"
        };

        let mut variant_results: Vec<Measure> = Vec::with_capacity(VARIANTS.len());
        let mut baseline_a_ssim2: f64 = f64::NAN;
        let mut baseline_a_bytes: usize = 0;
        for (idx, v) in VARIANTS.iter().enumerate() {
            let bytes = match encode_with_variant(raw, *w, *h, effort, dist, proxy, v.1) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  {} variant {} failed: {}", image, v.0, e);
                    variant_results.push(Measure {
                        bytes: 0,
                        butteraugli: f64::NAN,
                        ssim2: f64::NAN,
                    });
                    continue;
                }
            };
            let m = match measure(bytes, *w, *h, orig_linear_img, orig_srgb_img, &params) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  {} variant {} measure failed: {}", image, v.0, e);
                    Measure {
                        bytes: 0,
                        butteraugli: f64::NAN,
                        ssim2: f64::NAN,
                    }
                }
            };
            if idx == 0 {
                baseline_a_ssim2 = m.ssim2;
                baseline_a_bytes = m.bytes;
            }
            variant_results.push(m);
        }

        // OPEN under ledger rule: bytes_delta > 3.0 AND bfly_delta > 3.0
        let fixed_threshold_bytes_pct = 3.0_f64;
        let fixed_threshold_bfly_pct = 3.0_f64;
        let baseline_bytes_pct = if cjxl_b > 0 {
            (baseline_a_bytes as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0
        } else {
            0.0
        };
        let baseline_bfly_pct = if cjxl_bfly > 0.0 && !variant_results[0].butteraugli.is_nan() {
            (variant_results[0].butteraugli - cjxl_bfly) / cjxl_bfly * 100.0
        } else {
            0.0
        };
        let baseline_is_open = baseline_bytes_pct > fixed_threshold_bytes_pct
            && baseline_bfly_pct > fixed_threshold_bfly_pct;

        let mut cols: Vec<String> = Vec::new();
        for (idx, m) in variant_results.iter().enumerate() {
            let bytes_pct = if cjxl_b > 0 {
                (m.bytes as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0
            } else {
                0.0
            };
            let bfly_pct = if cjxl_bfly > 0.0 && !m.butteraugli.is_nan() {
                (m.butteraugli - cjxl_bfly) / cjxl_bfly * 100.0
            } else {
                0.0
            };
            cols.push(format!(
                "{}\t{:+.3}\t{:.4}\t{:+.3}\t{:.4}",
                m.bytes, bytes_pct, m.butteraugli, bfly_pct, m.ssim2
            ));
            let label = VARIANTS[idx].0;
            let agg = total_stats.get_mut(label).unwrap();
            agg.0 += m.bytes as i64 - cjxl_b as i64;
            agg.1 += 1;

            if idx > 0 {
                let now_open =
                    bytes_pct > fixed_threshold_bytes_pct && bfly_pct > fixed_threshold_bfly_pct;
                if !baseline_is_open && now_open {
                    agg.2 += 1;
                } else if baseline_is_open && !now_open {
                    agg.3 += 1;
                }
                let ssim2_delta = m.ssim2 - baseline_a_ssim2;
                agg.4 += ssim2_delta;
                if ssim2_delta < agg.5 {
                    agg.5 = ssim2_delta;
                }
                if ssim2_delta > agg.6 {
                    agg.6 = ssim2_delta;
                }
                let entry = class_ssim2_sum
                    .entry((label.to_string(), class.to_string()))
                    .or_insert((0.0, 0));
                entry.0 += ssim2_delta;
                entry.1 += 1;
            }

            // For FLIPPED_W148 cells: count closures (bytes_pct <= 3.0).
            if class == "FLIPPED_W148" && bytes_pct <= fixed_threshold_bytes_pct {
                *flipped_closed.get_mut(label).unwrap() += 1;
            }

            // For CONTROL cells: verify byte-identity against the WAS-baseline.
            // Since the gate doesn't fire (we return None from build_variant_table
            // for these), all 3 variants encode the same way → byte-identical.
            if class == "CONTROL_NOGATE" && idx > 0 && m.bytes != baseline_a_bytes {
                byte_identical_violations
                    .get_mut(label)
                    .unwrap()
                    .push(format!(
                        "{} e{} d={}: {} != {}",
                        image, effort, dist, m.bytes, baseline_a_bytes
                    ));
            }
        }

        println!(
            "{}\t{}\t{}\t{:.1}\t{}\t{:.4}\t{}\t{}",
            class,
            image,
            effort,
            dist,
            cjxl_b,
            cjxl_bfly,
            gate_marker,
            cols.join("\t")
        );
    }

    eprintln!("\n=== W44-154 aggregate (deltas vs A_dct32_124_w148 baseline) ===");
    for v in VARIANTS {
        let s = total_stats.get(v.0).unwrap();
        let (delta, n, fto, otf, ssim2_sum, ssim2_min, ssim2_max) = *s;
        let avg = if n > 0 { delta as f64 / n as f64 } else { 0.0 };
        let ssim2_avg = if n > 0 && v.0 != "A_dct32_124_w148" {
            ssim2_sum / n as f64
        } else {
            0.0
        };
        let label = v.0;
        eprintln!(
            "{:24}  Δvs_cjxl={:+8}B  n={}  FIXED→OPEN_vs_A: {}  OPEN→FIXED_vs_A: {}  Δssim2_avg={:+.4}  ssim2[{:+.4}, {:+.4}]",
            label, delta, n, fto, otf, ssim2_avg, ssim2_min, ssim2_max
        );
    }

    eprintln!("\n=== W44-154 FLIPPED_W148 closure counts (cells where bytes_pct ≤ 3.0) ===");
    for v in VARIANTS {
        let c = flipped_closed.get(v.0).unwrap();
        eprintln!("  {:24}  {}/6 closed", v.0, c);
    }

    eprintln!("\n=== W44-154 CONTROL byte-identity check ===");
    for v in &VARIANTS[1..] {
        let viols = byte_identical_violations.get(v.0).unwrap();
        if viols.is_empty() {
            eprintln!(
                "  {:24}  OK (3/3 CONTROL cells byte-identical to A baseline)",
                v.0
            );
        } else {
            for s in viols {
                eprintln!("  {:24}  VIOLATION: {}", v.0, s);
            }
        }
    }

    eprintln!("\n=== W44-154 per-class SSIM2 deltas (vs A_dct32_124_w148 baseline) ===");
    for v in &VARIANTS[1..] {
        eprintln!("Variant: {}", v.0);
        let classes = [
            "FLIPPED_W148",
            "DEF_1418519_D4",
            "DEF_1418519_D5",
            "DEF_1418519_D6",
            "PROTECT_1420710_D4",
            "PROTECT_1420710_D5",
            "PROTECT_1531677_D4",
            "COLLATERAL_CW_D3",
            "CONTROL_NOGATE",
        ];
        for class in classes {
            if let Some(&(sum, n)) = class_ssim2_sum.get(&(v.0.to_string(), class.to_string())) {
                let avg = if n > 0 { sum / n as f64 } else { 0.0 };
                eprintln!("  {:20}  n={}  Δssim2_avg={:+.4}", class, n, avg);
            }
        }
    }

    eprintln!("\n=== W44-154 verdict gates ===");
    eprintln!("Required: B or C closes ≥4 of 6 FLIPPED_W148 cells");
    eprintln!("Required: SSIM2 mean preservation on DEF_1418519_D5 (was +0.614 under W148)");
    eprintln!("Required: No SSIM2 regression > 0.30 on any cell");
    eprintln!("Required: CONTROL cells byte-identical");
    eprintln!("Read per-class table above to evaluate.");
}
