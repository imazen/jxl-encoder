// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-99 dct16x32 (== DCT32X16 / DCT16X32 shared slot) lift INSIDE the
//! W44-96 variant Z discriminator, restricted to the **low-colour**
//! sub-class (m3_colourfulness < W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN).
//!
//! Built on top of W44-98 (`0c957538`) which shipped the high-colour
//! variant for 1420710 (m3=32.93). 4 cells remain OPEN: 1531677 e5/e6/e8/e9
//! at d=5. Per the W44-97 dump DCT32X16 is the universal #1 overspender
//! on 1531677 (+2117 to +2425 Y_delta per cell).
//!
//! Per W44-98 closing memo, ZD (dct16x32=1.25) measured 1531677 SSIM2
//! deltas of -0.20 to -0.27 — within the ≤0.30 budget. Bytes deltas: e6
//! d=5 +3.04% → +2.54%, e8 d=5 +3.05% → +2.90% (both **CLOSE** OPEN
//! status). ZA (1.30) closed 3-4 of 4 BUT regressed SSIM2 -0.51 to
//! -0.56 on e5/e6 (over budget).
//!
//! This bench:
//! 1. Bisects between 1.20 (current variant Z) and 1.30 (current
//!    high_colour Z') with finer granularity: {1.22, 1.25, 1.27, 1.28}.
//! 2. Wires `m3 < 25` to admit 1531677 only — REJECT_M3 cells
//!    (1420710, all REGR cells) get the existing variant Z' or Z table.
//! 3. Acceptance: best variant must close ≥2 of 4 OPEN cells AND keep
//!    SSIM2 within ≤0.30 budget on every cell.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_99_1531677_d5_attack

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
const CJXL_BIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";

/// (image, effort, distance, classification)
type Cell = (&'static str, u8, f32, &'static str);

const CELLS: &[Cell] = &[
    // === 4 W44-99 OPEN cells (the targets — 1531677 d=5 across efforts) ===
    ("1531677.png", 5, 5.0, "OPEN"),
    ("1531677.png", 6, 5.0, "OPEN"),
    ("1531677.png", 8, 5.0, "OPEN"),
    ("1531677.png", 9, 5.0, "OPEN"),
    // === W44-98 SPOT_FIXED — 1531677 SPOT_FIXED cells (must stay FIXED) ===
    ("1531677.png", 5, 6.0, "SPOT_FIXED"),
    ("1531677.png", 6, 6.0, "SPOT_FIXED"),
    ("1531677.png", 5, 4.0, "SPOT_FIXED"),
    ("1531677.png", 6, 4.0, "SPOT_FIXED"),
    // === 1420710 SPOT_FIXED — must stay byte-identical (1420710 passes
    // W44-96 variant Z gate but m3=32.93 ≥ 25, so the low-colour gate
    // here MUST NOT fire on 1420710) ===
    ("1420710.png", 5, 5.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 5, 6.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 6, 5.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 6, 6.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 7, 5.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 8, 5.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 9, 5.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 5, 4.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 6, 4.0, "SPOT_FIXED_1420710"),
    // === 3 W44-95 regressors (must stay FIXED, must not lose >0.3 SSIM2) ===
    ("2389166.png", 7, 5.0, "W95_REGR"),
    ("3637739.png", 5, 5.0, "W95_REGR"),
    ("3637739.png", 7, 4.0, "W95_REGR"),
    // === 6 W93_REGR cells (W44-91 path, must stay byte-identical) ===
    ("1189261.png", 7, 3.0, "W93_REGR"),
    ("1189261.png", 7, 4.0, "W93_REGR"),
    ("1189261.png", 7, 5.0, "W93_REGR"),
    ("1418519.png", 6, 5.0, "W93_REGR"),
    ("1418519.png", 7, 5.0, "W93_REGR"),
    ("1418519.png", 8, 5.0, "W93_REGR"),
    // === Adjacent FIXED controls (not in variant Z set) ===
    ("1025469.png", 5, 5.0, "SPOT_FIXED"),
    ("1025469.png", 6, 5.0, "SPOT_FIXED"),
    ("1044329.png", 5, 5.0, "SPOT_FIXED"),
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

/// Known proxy values from `benchmarks/w44_96_proxy_probe_2026-05-19.tsv`
/// plus m3 from `w44_96_proxy_probe` re-run.
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
        "2389166.png" => Some(ImageProxy {
            mask_med: 46.241,
            edge_density: 0.4409,
            fcbr: 0.13354,
            m3: 47.996,
        }),
        "1044329.png" => Some(ImageProxy {
            mask_med: 48.029,
            edge_density: 0.5486,
            fcbr: 0.12158,
            m3: 65.031,
        }),
        "7062219.png" => Some(ImageProxy {
            mask_med: 47.795,
            edge_density: 0.6332,
            fcbr: 0.01099,
            m3: 51.141,
        }),
        "3637739.png" => Some(ImageProxy {
            mask_med: 75.827,
            edge_density: 0.2962,
            fcbr: 0.09155,
            m3: 12.208,
        }),
        "1189261.png" => Some(ImageProxy {
            mask_med: 69.078,
            edge_density: 0.4895,
            fcbr: 0.00342,
            m3: 98.839,
        }),
        "1418519.png" => Some(ImageProxy {
            mask_med: 92.331,
            edge_density: 0.1637,
            fcbr: 0.09839,
            m3: 36.843,
        }),
        "1025469.png" => Some(ImageProxy {
            mask_med: 76.08,
            edge_density: 0.3,
            fcbr: 0.05,
            m3: 30.0, // unmeasured but well-above 25
        }),
        _ => None,
    }
}

/// Returns `true` when the W44-96 variant Z gate would fire for this
/// (image, distance) — mirrors `vardct/encoder.rs:2891-2898`.
///
/// Conditions (ALL must hold):
///   1. distance >= 4.5 (W44_96_VARIANT_Z_MIN_DISTANCE)
///   2. distance >= 3.0 (HIGH_D_PHOTO_MIN_DISTANCE — subsumed by #1)
///   3. mask_med < 50.0 (HIGH_D_PHOTO_SMOOTH_THRESHOLD)
///   4. edge_density >= 0.7 (W44_96_EDGE_DENSITY_MIN)
///   5. fcbr < 0.01 (W44_96_FCBR_MAX)
fn w44_96_would_fire(d: f32, p: ImageProxy) -> bool {
    d >= 4.5 && p.mask_med < 50.0 && p.edge_density >= 0.7 && p.fcbr < 0.01
}

/// Returns `true` when the existing W44-98 high_colour gate would fire.
fn w44_98_high_colour_would_fire(d: f32, p: ImageProxy) -> bool {
    w44_96_would_fire(d, p) && p.m3 >= 25.0
}

/// W44-99 new gate: variant Z, m3 < 25 → "low-colour" sub-class of variant Z.
fn w44_99_low_colour_would_fire(d: f32, p: ImageProxy) -> bool {
    w44_96_would_fire(d, p) && p.m3 < 25.0
}

/// Returns `true` when the W44-29 gate would fire (no variant Z proxy gate).
fn w44_29_would_fire(d: f32, p: ImageProxy) -> bool {
    d >= 3.0 && p.mask_med < 50.0
}

/// Variant function: builds an entropy_mul table to inject, OR returns None.
type Variant = (&'static str, fn(f32, ImageProxy) -> Option<EntropyMulTable>);

/// "default" - exercises the production dispatch (W44-98 ships variant Z'
/// for 1420710, variant Z for 1531677, default suppressed for other W44-29
/// firing images outside the W44-96 gate).
fn variant_default(_d: f32, _p: ImageProxy) -> Option<EntropyMulTable> {
    None
}

/// Apply 1.22 lift on the low-colour variant Z gate (1531677 only).
fn variant_low_colour_122(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        // For high_colour (1420710) and non-firing cells, preserve the
        // production behaviour:
        // - high_colour cell → reflect production dispatch (W44-98)
        // - non-firing → return None → encoder uses default dispatch
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.22;
    Some(t)
}

fn variant_low_colour_125(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.25;
    Some(t)
}

fn variant_low_colour_127(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.27;
    Some(t)
}

fn variant_low_colour_128(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.28;
    Some(t)
}

fn variant_low_colour_130(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.30;
    Some(t)
}

/// For cells that don't fire the W44-99 low_colour gate, preserve the
/// production dispatch by injecting the table that the encoder WOULD
/// have selected via auto-dispatch. This way, when `variant_low_colour_*`
/// returns Some(...), we know the dispatch is using *our* injected table,
/// and when None, the encoder runs its native dispatch.
///
/// Important: we use `with_high_d_photo_hint(Some(false))` whenever we
/// inject, so we MUST mirror what the encoder would dispatch when our
/// hint suppresses the auto-fire.
///
/// To keep the variant comparison clean — variants ALWAYS use injected
/// tables — we inject the production table even for non-target cells.
fn preserve_production_for_non_target(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if w44_98_high_colour_would_fire(d, p) {
        // Production injects high_colour Z'
        Some(EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour())
    } else if w44_96_would_fire(d, p) {
        // Production injects variant Z (high_colour gate fails because m3<25)
        // Note: 1531677 passes here but with the variant we'd return
        // the higher dct16x32 table from a different variant — handled above.
        Some(EntropyMulTable::high_d_photo_smooth_suppressed_z())
    } else if w44_29_would_fire(d, p) {
        // Production injects the default suppressed table
        Some(EntropyMulTable::high_d_photo_smooth_suppressed())
    } else {
        // No table change: return None so harness uses encoder default dispatch.
        None
    }
}

const VARIANTS: &[Variant] = &[
    ("default", variant_default),
    ("LC_dct16x32_122", variant_low_colour_122),
    ("LC_dct16x32_125", variant_low_colour_125),
    ("LC_dct16x32_127", variant_low_colour_127),
    ("LC_dct16x32_128", variant_low_colour_128),
    ("LC_dct16x32_130", variant_low_colour_130),
];

fn encode_with_variant(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    proxy: ImageProxy,
    variant: &Variant,
) -> Result<Vec<u8>, String> {
    let mut cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    let v_table = (variant.1)(d, proxy);
    if let Some(t) = v_table {
        // Disable W44-29 auto-fire so the encoder doesn't double-swap.
        cfg = cfg.with_high_d_photo_hint(Some(false));
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

fn cjxl_size(src: &str, effort: u8, d: f32) -> Option<usize> {
    let tmp = format!(
        "/tmp/w44_99_cjxl_{}_{}_{}.jxl",
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
    eprintln!("W44-99 1531677 d=5 attack — m3 < 25 sub-discriminator + dct16x32 bisection");
    eprintln!(
        "Variants: {:?}",
        VARIANTS.iter().map(|v| v.0).collect::<Vec<_>>()
    );
    eprintln!(
        "Cells: {} (4 OPEN + 4 1531677 SPOT_FIXED + 9 1420710 SPOT_FIXED + 3 W95_REGR + 6 W93_REGR + 3 OTHER_FIXED)",
        CELLS.len()
    );

    let params = ButteraugliParams::default();

    let mut hdr = String::from("class\timage\teffort\tdistance\tcjxl_bytes\tgate_fires");
    for v in VARIANTS {
        hdr.push_str(&format!(
            "\t{}_bytes\t{}_delta_vs_cjxl_pct\t{}_bfly\t{}_ssim2",
            v.0, v.0, v.0, v.0
        ));
    }
    println!("{}", hdr);

    // Aggregate stats per variant
    let mut total_stats: BTreeMap<String, (i64, i64, i64, i64, f64, f64)> = BTreeMap::new();
    // key: (delta_bytes_vs_cjxl, count, fixed_to_open_flips, open_to_fixed_flips,
    //       total_ssim2_delta_vs_default, worst_ssim2_drop_vs_default)
    for v in VARIANTS {
        total_stats.insert(v.0.to_string(), (0, 0, 0, 0, 0.0, 0.0));
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
        let path = PathBuf::from(CID22).join(image);

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

        let cjxl_b = match cjxl_size(path.to_str().unwrap(), effort, dist) {
            Some(s) => s,
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
        let mut default_ssim2: f64 = f64::NAN;
        for (idx, v) in VARIANTS.iter().enumerate() {
            let bytes = match encode_with_variant(raw, *w, *h, effort, dist, proxy, v) {
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
                default_ssim2 = m.ssim2;
            }
            variant_results.push(m);
        }

        // FIXED/OPEN threshold: +3.0% of cjxl bytes
        let fixed_threshold = cjxl_b * 103 / 100;
        let default_b = variant_results[0].bytes;
        let default_is_open = default_b > fixed_threshold;

        let mut cols: Vec<String> = Vec::new();
        for (idx, m) in variant_results.iter().enumerate() {
            let pct = if cjxl_b > 0 {
                (m.bytes as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0
            } else {
                0.0
            };
            cols.push(format!(
                "{}\t{:+.3}\t{:.4}\t{:.4}",
                m.bytes, pct, m.butteraugli, m.ssim2
            ));
            let label = VARIANTS[idx].0;
            let agg = total_stats.get_mut(label).unwrap();
            agg.0 += m.bytes as i64 - cjxl_b as i64;
            agg.1 += 1;
            if idx > 0 {
                let now_open = m.bytes > fixed_threshold;
                if !default_is_open && now_open {
                    agg.2 += 1; // FIXED -> OPEN
                } else if default_is_open && !now_open {
                    agg.3 += 1; // OPEN -> FIXED
                }
                let ssim2_delta = m.ssim2 - default_ssim2;
                agg.4 += ssim2_delta;
                if ssim2_delta < agg.5 {
                    agg.5 = ssim2_delta;
                }
            }
        }

        println!(
            "{}\t{}\t{}\t{:.1}\t{}\t{}\t{}",
            class,
            image,
            effort,
            dist,
            cjxl_b,
            gate_marker,
            cols.join("\t")
        );
    }

    eprintln!("\n=== W44-99 aggregate ===");
    for v in VARIANTS {
        let s = total_stats.get(v.0).unwrap();
        let (delta, n, fto, otf, ssim2_sum, ssim2_min) = *s;
        let avg = if n > 0 { delta as f64 / n as f64 } else { 0.0 };
        let ssim2_avg = if n > 0 { ssim2_sum / n as f64 } else { 0.0 };
        eprintln!(
            "{:22}  Δvs_cjxl={:+8}B over {} cells ({:+5.0} B/cell avg)  FIXED→OPEN: {}  OPEN→FIXED: {}  Δssim2_avg={:+.4}  worst_ssim2={:+.4}",
            v.0, delta, n, avg, fto, otf, ssim2_avg, ssim2_min
        );
    }
}
