// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-156 distance-aware variant Z dispatch A/B bisect.
//!
//! Tests two threshold values for splitting variant Z dispatch on
//! `target_distance`:
//!
//! - **A = baseline** (no d-high split; equivalent to W44-154 ship —
//!   dct32x32 = 1.22 at every distance)
//! - **B = threshold 5.5** (d > 5.5 → dct32x32 = 1.20; d ≤ 5.5 → 1.22)
//! - **C = threshold 5.0** (d > 5.0 → dct32x32 = 1.20; d ≤ 5.0 → 1.22)
//!
//! Goal: close 1420710 e5 d=6 (W44-155-diagnosed OPEN cell) without
//! regressing the 1531677 d=6 protection cluster (already at SSIM2
//! -0.247 threshold post-W44-154).
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_156_distance_aware_variant_z_ab
//!
//! We use direct table injection via `LossyInternalParams`
//! (same as W44-154) — this lets us toggle the d-high split per-cell
//! without process-wide env vars. The runtime env hook
//! `__JXL_W44_156_THRESHOLD` is for ad-hoc inspection; the bench drives
//! the choice deterministically.

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

/// 20-cell coverage per W44-156 task:
/// - TARGET: 1420710 e5 d=6 (the cell to close)
/// - PROTECT_d6: 1531677 d=6 × e7/e8/e9 (already at -0.247 SSIM2)
/// - PROTECT_1420710_d=4/5: 1420710 d=4/5 × e7/e8/e9 (W44-148/154 wins)
/// - PROTECT_1531677_d=5: 1531677 d=5 × e7/e8/e9 (W44-154 wins)
/// - PROTECT_1418519: 1418519 d=5/6 × e8/e9 (W44-148/152 wins; gate
///   doesn't fire — must stay byte-identical)
/// - PROTECT_codec_wiki_d3: codec_wiki d=3 × e8 (W44-152 collateral —
///   gate doesn't fire, must stay byte-identical)
/// - CONTROL: 1189261 d=4 e7, 1044329 d=5 e7 (off-gate, byte-identical)
const CELLS: &[Cell] = &[
    // === TARGET: the cell to close ===
    ("1420710.png", 5, 6.0, "TARGET"),
    // === PROTECT_d6: 1531677 d=6 cluster (-0.247 SSIM2 already) ===
    ("1531677.png", 7, 6.0, "PROTECT_1531677_d6"),
    ("1531677.png", 8, 6.0, "PROTECT_1531677_d6"),
    ("1531677.png", 9, 6.0, "PROTECT_1531677_d6"),
    // === PROTECT_1420710_d=4/5 (W44-148/154 wins) ===
    ("1420710.png", 7, 4.0, "PROTECT_1420710_d45"),
    ("1420710.png", 8, 4.0, "PROTECT_1420710_d45"),
    ("1420710.png", 9, 4.0, "PROTECT_1420710_d45"),
    ("1420710.png", 7, 5.0, "PROTECT_1420710_d45"),
    ("1420710.png", 8, 5.0, "PROTECT_1420710_d45"),
    ("1420710.png", 9, 5.0, "PROTECT_1420710_d45"),
    // === PROTECT_1531677_d=5 (W44-154 wins) ===
    ("1531677.png", 7, 5.0, "PROTECT_1531677_d5"),
    ("1531677.png", 8, 5.0, "PROTECT_1531677_d5"),
    ("1531677.png", 9, 5.0, "PROTECT_1531677_d5"),
    // === PROTECT_1418519 (W44-148/152 wins; variant Z doesn't fire) ===
    ("1418519.png", 8, 5.0, "PROTECT_1418519"),
    ("1418519.png", 9, 5.0, "PROTECT_1418519"),
    ("1418519.png", 8, 6.0, "PROTECT_1418519"),
    ("1418519.png", 9, 6.0, "PROTECT_1418519"),
    // === PROTECT_codec_wiki_d3 (W44-152 collateral; variant Z doesn't fire) ===
    ("codec_wiki.png", 8, 3.0, "PROTECT_codec_wiki_d3"),
    // === CONTROL: off all variant Z gates ===
    ("1189261.png", 7, 4.0, "CONTROL_NOGATE"),
    ("1044329.png", 7, 5.0, "CONTROL_NOGATE"),
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

/// Proxy values lifted from W44-148/154 bisect + W44-149 audit. Hot-cached
/// per-image to avoid per-encode O(W·H) recompute (the production gate
/// does compute these in the encoder pipeline; for the bench harness we
/// inline known values so the A/B can isolate the table change).
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
        // 1418519: mask=76 fails W44-29 mask<50 outer gate. Variant Z
        // dispatch does NOT fire here.
        "1418519.png" => Some(ImageProxy {
            mask_med: 76.0,
            edge_density: 0.50,
            fcbr: 0.05,
            m3: 23.0,
        }),
        // 1189261: fires W44-91 path, NOT variant Z.
        "1189261.png" => Some(ImageProxy {
            mask_med: 69.0,
            edge_density: 0.60,
            fcbr: 0.005,
            m3: 80.5,
        }),
        // 1044329: mask=48 narrowly fires W44-29 but fails W44-96
        // edge_density >= 0.7 gate (variant Z does NOT fire).
        "1044329.png" => Some(ImageProxy {
            mask_med: 48.03,
            edge_density: 0.55,
            fcbr: 0.02,
            m3: 30.0,
        }),
        // codec_wiki: screenshot path (W22-1 not Z).
        "codec_wiki.png" => Some(ImageProxy {
            mask_med: 99.0,
            edge_density: 0.20,
            fcbr: 0.50,
            m3: 8.0,
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

#[derive(Debug, Clone, Copy)]
struct Variant {
    label: &'static str,
    /// `None` = no d-high split (baseline / W44-154); `Some(t)` = split
    /// at distance `t`.
    threshold: Option<f32>,
}

const VARIANTS: &[Variant] = &[
    Variant {
        label: "A_baseline_no_split",
        threshold: None,
    },
    Variant {
        label: "B_threshold_5p5",
        threshold: Some(5.5),
    },
    Variant {
        label: "C_threshold_5p0",
        threshold: Some(5.0),
    },
];

/// Build the entropy_mul table for this (distance, proxy, threshold)
/// tuple. Mirrors the production gate ordering INCLUDING the W44-156
/// d-high split (when `threshold` is `Some(t)` and `d > t`).
fn build_variant_table(d: f32, p: ImageProxy, threshold: Option<f32>) -> Option<EntropyMulTable> {
    let d_high = match threshold {
        Some(t) => d > t,
        None => false,
    };
    if w44_98_high_colour_would_fire(d, p) {
        Some(if d_high {
            EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour_d_high()
        } else {
            EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour()
        })
    } else if w44_99_low_colour_would_fire(d, p) {
        Some(if d_high {
            EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour_d_high()
        } else {
            EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour()
        })
    } else if w44_96_would_fire(d, p) {
        Some(if d_high {
            EntropyMulTable::high_d_photo_smooth_suppressed_z_d_high()
        } else {
            EntropyMulTable::high_d_photo_smooth_suppressed_z()
        })
    } else if w44_29_would_fire(d, p) {
        // Outer W44-29 — preserve, don't touch (variant Z doesn't fire here).
        // Returning the default suppressed table forces injection (matches
        // the W44-154 harness pattern so all variants go through the same
        // override path; the result is byte-identical because all three
        // variants pick the SAME default table when variant Z doesn't fire).
        Some(EntropyMulTable::high_d_photo_smooth_suppressed())
    } else {
        // No gate fires — use production default (no override needed).
        None
    }
}

fn encode_with_variant(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    proxy: ImageProxy,
    threshold: Option<f32>,
) -> Result<Vec<u8>, String> {
    let mut cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    if let Some(t) = build_variant_table(d, proxy, threshold) {
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
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Option<(usize, f64, f64)> {
    let tmp = format!(
        "/tmp/w44_156_cjxl_{}_{}_{}.jxl",
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

    Some((sz, bfly, ssim2))
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
    eprintln!("W44-156 distance-aware variant Z A/B (threshold ∈ {{none, 5.5, 5.0}})");
    eprintln!(
        "Variants: {:?}",
        VARIANTS.iter().map(|v| v.label).collect::<Vec<_>>()
    );
    eprintln!(
        "Cells: {} (target: close TARGET, preserve PROTECT_d6 SSIM2 budget)",
        CELLS.len()
    );

    let params = ButteraugliParams::default();

    let mut hdr = String::from(
        "class\timage\teffort\tdistance\tcjxl_bytes\tcjxl_bfly\tcjxl_ssim2\tgate_fires",
    );
    for v in VARIANTS {
        hdr.push_str(&format!(
            "\t{}_bytes\t{}_bytes_pct\t{}_bfly\t{}_bfly_pct\t{}_ssim2",
            v.label, v.label, v.label, v.label, v.label
        ));
    }
    println!("{}", hdr);

    // Track aggregate FIXED/OPEN flips relative to baseline A.
    // (delta_bytes_sum, n_cells, fixed_to_open_vs_A, open_to_fixed_vs_A,
    //  ssim2_delta_sum_vs_A, ssim2_min_delta_vs_A, ssim2_max_delta_vs_A)
    let mut total_stats: BTreeMap<String, (i64, i64, i64, i64, f64, f64, f64)> = BTreeMap::new();
    for v in VARIANTS {
        total_stats.insert(v.label.to_string(), (0, 0, 0, 0, 0.0, 0.0, 0.0));
    }
    let mut class_ssim2_sum: BTreeMap<(String, String), (f64, i64)> = BTreeMap::new();
    let mut class_bytes_sum: BTreeMap<(String, String), (i64, i64)> = BTreeMap::new();
    let mut byte_identical_violations: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut target_closure: BTreeMap<String, Option<(bool, f64, f64)>> = BTreeMap::new();
    for v in VARIANTS {
        byte_identical_violations.insert(v.label.to_string(), Vec::new());
        target_closure.insert(v.label.to_string(), None);
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

        let (cjxl_b, cjxl_bfly, cjxl_ssim2) = match cjxl_size_and_bfly(
            path.to_str().unwrap(),
            effort,
            dist,
            *w,
            *h,
            orig_linear_img,
            orig_srgb_img,
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
            let bytes = match encode_with_variant(raw, *w, *h, effort, dist, proxy, v.threshold) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  {} variant {} failed: {}", image, v.label, e);
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
                    eprintln!("  {} variant {} measure failed: {}", image, v.label, e);
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
            let label = VARIANTS[idx].label;
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
                let entry_b = class_bytes_sum
                    .entry((label.to_string(), class.to_string()))
                    .or_insert((0, 0));
                entry_b.0 += m.bytes as i64 - baseline_a_bytes as i64;
                entry_b.1 += 1;
            }

            // For CONTROL / PROTECT_1418519 / PROTECT_codec_wiki_d3:
            // variant Z does NOT fire (the build_variant_table function
            // returns the SAME default suppressed table OR None for these
            // cells, regardless of threshold), so all variants should be
            // byte-identical.
            let must_be_byte_identical = matches!(
                class,
                "CONTROL_NOGATE" | "PROTECT_1418519" | "PROTECT_codec_wiki_d3"
            );
            if must_be_byte_identical && idx > 0 && m.bytes != baseline_a_bytes {
                byte_identical_violations
                    .get_mut(label)
                    .unwrap()
                    .push(format!(
                        "{} e{} d={}: {} != {}",
                        image, effort, dist, m.bytes, baseline_a_bytes
                    ));
            }

            // For TARGET cell: track closure (bytes_pct <= 3.0 OR bfly_pct <= 3.0).
            if class == "TARGET" {
                let closed =
                    bytes_pct <= fixed_threshold_bytes_pct || bfly_pct <= fixed_threshold_bfly_pct;
                target_closure.insert(label.to_string(), Some((closed, bytes_pct, bfly_pct)));
            }
        }

        println!(
            "{}\t{}\t{}\t{:.1}\t{}\t{:.4}\t{:.4}\t{}\t{}",
            class,
            image,
            effort,
            dist,
            cjxl_b,
            cjxl_bfly,
            cjxl_ssim2,
            gate_marker,
            cols.join("\t")
        );
    }

    eprintln!("\n=== W44-156 aggregate (deltas vs A_baseline_no_split) ===");
    for v in VARIANTS {
        let s = total_stats.get(v.label).unwrap();
        let (delta, n, fto, otf, ssim2_sum, ssim2_min, ssim2_max) = *s;
        let _avg = if n > 0 { delta as f64 / n as f64 } else { 0.0 };
        let ssim2_avg = if n > 0 && v.label != "A_baseline_no_split" {
            ssim2_sum / n as f64
        } else {
            0.0
        };
        eprintln!(
            "{:24}  Δvs_cjxl={:+8}B  n={}  FIXED→OPEN_vs_A: {}  OPEN→FIXED_vs_A: {}  Δssim2_avg={:+.4}  ssim2[{:+.4}, {:+.4}]",
            v.label, delta, n, fto, otf, ssim2_avg, ssim2_min, ssim2_max
        );
    }

    eprintln!("\n=== W44-156 TARGET (1420710 e5 d=6) closure status ===");
    for v in VARIANTS {
        match target_closure.get(v.label).unwrap() {
            Some((closed, bp, bfp)) => {
                eprintln!(
                    "  {:24}  closed={}  bytes_pct={:+.3}  bfly_pct={:+.3}",
                    v.label, closed, bp, bfp
                );
            }
            None => eprintln!("  {:24}  not measured", v.label),
        }
    }

    eprintln!("\n=== W44-156 byte-identity checks (must-be-byte-identical classes) ===");
    for v in &VARIANTS[1..] {
        let viols = byte_identical_violations.get(v.label).unwrap();
        if viols.is_empty() {
            eprintln!(
                "  {:24}  OK (all PROTECT_1418519 + PROTECT_codec_wiki_d3 + CONTROL_NOGATE cells byte-identical to A)",
                v.label
            );
        } else {
            for s in viols {
                eprintln!("  {:24}  VIOLATION: {}", v.label, s);
            }
        }
    }

    eprintln!("\n=== W44-156 per-class deltas (vs A_baseline_no_split) ===");
    for v in &VARIANTS[1..] {
        eprintln!("Variant: {}", v.label);
        let classes = [
            "TARGET",
            "PROTECT_1531677_d6",
            "PROTECT_1420710_d45",
            "PROTECT_1531677_d5",
            "PROTECT_1418519",
            "PROTECT_codec_wiki_d3",
            "CONTROL_NOGATE",
        ];
        for class in classes {
            let ss = class_ssim2_sum.get(&(v.label.to_string(), class.to_string()));
            let bb = class_bytes_sum.get(&(v.label.to_string(), class.to_string()));
            if let (Some(&(ssum, sn)), Some(&(bsum, bn))) = (ss, bb) {
                let savg = if sn > 0 { ssum / sn as f64 } else { 0.0 };
                let bavg = if bn > 0 { bsum as f64 / bn as f64 } else { 0.0 };
                eprintln!(
                    "  {:24}  n={}  Δssim2_avg={:+.4}  Δbytes_avg={:+.1}B",
                    class, sn, savg, bavg
                );
            }
        }
    }

    eprintln!("\n=== W44-156 verdict gates ===");
    eprintln!("Required: B or C closes 1420710 e5 d=6 (bytes_pct <= 3.0)");
    eprintln!(
        "Required: PROTECT_1531677_d6 SSIM2 mean preservation (no regression beyond -0.30 vs cjxl absolute)"
    );
    eprintln!(
        "Required: PROTECT_1420710_d45 / PROTECT_1531677_d5 SSIM2 mean preservation (>=90% of W44-154 levels)"
    );
    eprintln!("Required: PROTECT_1418519 + PROTECT_codec_wiki_d3 BYTE-IDENTICAL");
    eprintln!("Required: CONTROL_NOGATE BYTE-IDENTICAL");
    eprintln!("Read per-class tables above to evaluate.");
}
