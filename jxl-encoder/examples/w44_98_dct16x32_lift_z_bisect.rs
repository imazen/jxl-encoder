// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-98 dct16x32 / dct32x16 lift INSIDE the W44-96 variant Z discriminator.
//!
//! Per W44-97 per-strategy dump on the 7 OPEN cells remaining after
//! W44-96, **DCT32X16 is the universal #1 overspender** (+10017 Y_delta
//! total, max +2425 on 1531677 e6 d5). DCT16X32 is #2 (+2465). Both
//! share the `dct16x32` entropy_mul slot in [`EntropyMulTable`]
//! (`ac_strategy.rs:713`: `RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 =>
//! table.dct16x32`), so lifting that single value affects both transforms.
//!
//! The W44-96 variant Z currently scales `dct16x32` with `dct32x32`
//! (1.49 / 1.48 ratio, so 1.20 → 1.208). This bench bisects breaking
//! that ratio — keep `dct32x32=1.20` (preserved to honour the W44-95
//! SSIM2 budget) and raise `dct16x32` independently. Per W44-94
//! variant X already measured `dct16x32=1.40` globally and tanked
//! 1531677 SSIM2 by -0.30 to -0.74. With the narrower W44-96 gate
//! (auto-only, d>=4.5, mask<50 AND `edge_density>=0.7` AND `fcbr<0.01`)
//! the regression may not reproduce; if it does, bisect smaller lifts
//! (1.25, 1.30).
//!
//! All variants ONLY inject when the W44-96 gate would fire — for any
//! cell whose mask/edge_density/fcbr does not match, the variant returns
//! `None` and the default encoder dispatch runs (yielding the same bytes
//! as the baseline).
//!
//! Cells: 7 W44-97 OPEN + 3 W44-95 regressors + 6 W93_REGR + 8 controls.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_98_dct16x32_lift_z_bisect

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
    // === 7 W44-97 OPEN cells (post-W44-96) ===
    ("1420710.png", 5, 5.0, "OPEN"),
    ("1420710.png", 5, 6.0, "OPEN"),
    ("1420710.png", 7, 5.0, "OPEN"),
    ("1531677.png", 5, 5.0, "OPEN"),
    ("1531677.png", 6, 5.0, "OPEN"),
    ("1531677.png", 8, 5.0, "OPEN"),
    ("1531677.png", 9, 5.0, "OPEN"),
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
    // === SPOT_FIXED controls (W44-96 close cells - must stay FIXED) ===
    ("1420710.png", 6, 5.0, "SPOT_FIXED"),
    ("1420710.png", 6, 6.0, "SPOT_FIXED"),
    ("1420710.png", 8, 5.0, "SPOT_FIXED"),
    ("1420710.png", 9, 5.0, "SPOT_FIXED"),
    ("1531677.png", 5, 6.0, "SPOT_FIXED"),
    ("1531677.png", 6, 6.0, "SPOT_FIXED"),
    // === Variant-Z-firing-but-different-cells controls (W44-29 load-bearing) ===
    ("1420710.png", 5, 4.0, "SPOT_FIXED"),
    ("1420710.png", 6, 4.0, "SPOT_FIXED"),
    ("1531677.png", 5, 4.0, "SPOT_FIXED"),
    ("1531677.png", 6, 4.0, "SPOT_FIXED"),
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
}

/// Known proxy values from `benchmarks/w44_96_proxy_probe_2026-05-19.tsv`.
/// Returning None means we don't have data → variant won't fire (safe fallback).
fn known_proxies(image: &str) -> Option<ImageProxy> {
    match image {
        "1420710.png" => Some(ImageProxy {
            mask_med: 39.549,
            edge_density: 0.9298,
            fcbr: 0.00000,
        }),
        "1531677.png" => Some(ImageProxy {
            mask_med: 35.634,
            edge_density: 0.8766,
            fcbr: 0.00000,
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
        "7062219.png" => Some(ImageProxy {
            mask_med: 47.795,
            edge_density: 0.6332,
            fcbr: 0.01099,
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
            edge_density: 0.3, // approximate; not in proxy probe TSV
            fcbr: 0.05,
        }),
        _ => None,
    }
}

/// Returns `true` when the W44-96 variant Z gate would fire for this
/// (image, distance) — mirrors `vardct/encoder.rs:2867-2874`.
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

/// Returns `true` when the W44-29 gate would fire (no variant Z proxy gate).
/// Used to verify the "default" path produces variant Z naturally — when
/// W44-29 fires AND proxies pass AND d>=4.5, the encoder will auto-pick
/// variant Z and our None-returning variants get the same bytes as that
/// natural dispatch (the table is identical to what the encoder would build).
fn w44_29_would_fire(d: f32, p: ImageProxy) -> bool {
    d >= 3.0 && p.mask_med < 50.0
}

/// Variant function: builds an entropy_mul table to inject via
/// `internal_params`, OR returns None to use encoder default dispatch.
/// When Some, the harness sets `with_high_d_photo_hint(Some(false))`
/// to disable auto-fire then injects the table.
type Variant = (&'static str, fn(f32, ImageProxy) -> Option<EntropyMulTable>);

fn variant_default(_d: f32, _p: ImageProxy) -> Option<EntropyMulTable> {
    None
}

/// ZA: variant Z (dct32x32=1.20) with dct16x32 lifted to 1.30
/// (libjxl reference is 1.49; W44-29 default is 1.34*1.49/1.48=1.349;
/// current variant Z is 1.20*1.49/1.48=1.208).
fn variant_za_dct16x32_130(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_96_would_fire(d, p) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.30;
    Some(t)
}

/// ZB: variant Z (dct32x32=1.20) with dct16x32 lifted to 1.40
/// (matches W44-94 variant X's lift value, but inside the much
/// narrower W44-96 gate instead of the global W44-29 d>=5 gate).
fn variant_zb_dct16x32_140(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_96_would_fire(d, p) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.40;
    Some(t)
}

/// ZC: variant Z (dct32x32=1.20) with dct16x32 lifted to 1.49
/// (libjxl reference). This is the most aggressive variant.
fn variant_zc_dct16x32_149(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_96_would_fire(d, p) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.49;
    Some(t)
}

/// ZD: variant Z (dct32x32=1.20) with dct16x32=1.25 (most conservative).
fn variant_zd_dct16x32_125(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_96_would_fire(d, p) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.25;
    Some(t)
}

const VARIANTS: &[Variant] = &[
    ("default", variant_default),
    ("ZD_dct16x32_125", variant_zd_dct16x32_125),
    ("ZA_dct16x32_130", variant_za_dct16x32_130),
    ("ZB_dct16x32_140", variant_zb_dct16x32_140),
    ("ZC_dct16x32_149", variant_zc_dct16x32_149),
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
        // Inject our table via internal_params.
        cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides { high_d_photo_hint: Some(false), ..Default::default() });
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
        "/tmp/w44_98_cjxl_{}_{}_{}.jxl",
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
    eprintln!("W44-98 dct16x32 lift inside W44-96 variant Z A/B sweep");
    eprintln!(
        "Variants: {:?}",
        VARIANTS.iter().map(|v| v.0).collect::<Vec<_>>()
    );
    eprintln!(
        "Cells: {} (7 OPEN + 3 W95_REGR + 6 W93_REGR + 14 SPOT_FIXED)",
        CELLS.len()
    );

    let params = ButteraugliParams::default();

    // Header
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
    //       total_ssim2_delta_vs_default, max_ssim2_drop_vs_default)
    for v in VARIANTS {
        total_stats.insert(v.0.to_string(), (0, 0, 0, 0, 0.0, 0.0));
    }

    // Cache decoded sources per image
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
        let gate_fires = w44_96_would_fire(dist, proxy);
        let w44_29_fires = w44_29_would_fire(dist, proxy);

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
            // Update aggregate
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

        let gate_marker = if gate_fires {
            "Z"
        } else if w44_29_fires {
            "29"
        } else {
            "-"
        };

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

    eprintln!("\n=== W44-98 aggregate ===");
    for v in VARIANTS {
        let s = total_stats.get(v.0).unwrap();
        let (delta, n, fto, otf, ssim2_sum, ssim2_min) = *s;
        let avg = if n > 0 { delta as f64 / n as f64 } else { 0.0 };
        let ssim2_avg = if n > 0 { ssim2_sum / n as f64 } else { 0.0 };
        eprintln!(
            "{:20}  Δvs_cjxl={:+8}B over {} cells ({:+5.0} B/cell avg)  FIXED→OPEN: {}  OPEN→FIXED: {}  Δssim2_avg={:+.4}  worst_ssim2={:+.4}",
            v.0, delta, n, avg, fto, otf, ssim2_avg, ssim2_min
        );
    }
}
