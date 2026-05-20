// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-94 widen W44-77/W44-29 find_best_32x32 tightening A/B sweep.
//!
//! Tests stronger tightening of the W44-29 swap to
//! [`EntropyMulTable::high_d_photo_smooth_suppressed`] (currently
//! dct32x32=1.34, dct16x32=1.349, dct16x16=1.27).  The W44-77 sweep showed
//! no uniform dct16x32 value beats 1.349 at d<=4, so the widening direction
//! must be either:
//!   (a) stronger dct32x32 push (e.g. 1.27 to match dct16x16)
//!   (b) per-distance — different dct16x32 at d>=5 vs d=3-4
//!   (c) both
//!
//! W44-93 honest-stopped on widening try_dct64 because of SSIM2 collateral
//! on photos.  This sweep MEASURES SSIM2 + butteraugli to avoid the same
//! failure mode.
//!
//! Cells: 13 OPEN cells + 5 W44-93-regressed FIXED cells + 13 adjacent
//! FIXED cells = 31 total.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_94_find_best_32_widen

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
    // === 13 W44-92 OPEN cells ===
    ("1420710.png", 5, 5.0, "OPEN"),
    ("1420710.png", 5, 6.0, "OPEN"),
    ("1420710.png", 6, 5.0, "OPEN"),
    ("1420710.png", 6, 6.0, "OPEN"),
    ("1420710.png", 7, 5.0, "OPEN"),
    ("1420710.png", 8, 5.0, "OPEN"),
    ("1420710.png", 9, 5.0, "OPEN"),
    ("1531677.png", 5, 5.0, "OPEN"),
    ("1531677.png", 5, 6.0, "OPEN"),
    ("1531677.png", 6, 5.0, "OPEN"),
    ("1531677.png", 6, 6.0, "OPEN"),
    ("1531677.png", 8, 5.0, "OPEN"),
    ("1531677.png", 9, 5.0, "OPEN"),
    // === 5 W44-93-regressed FIXED cells (must not regress now) ===
    ("1189261.png", 5, 6.0, "W93_REGR"),
    ("1189261.png", 6, 6.0, "W93_REGR"),
    ("1418519.png", 5, 5.0, "W93_REGR"),
    ("1418519.png", 5, 6.0, "W93_REGR"),
    ("1418519.png", 6, 5.0, "W93_REGR"),
    ("1418519.png", 6, 6.0, "W93_REGR"),
    // === SPOT_FIXED — controls ===
    ("1418519.png", 7, 5.0, "SPOT_FIXED"),
    ("1418519.png", 7, 6.0, "SPOT_FIXED"),
    ("1420710.png", 5, 4.0, "SPOT_FIXED"), // W44-29's LOAD-BEARING cell
    ("1420710.png", 6, 4.0, "SPOT_FIXED"),
    ("1531677.png", 5, 4.0, "SPOT_FIXED"),
    ("1531677.png", 6, 4.0, "SPOT_FIXED"),
    ("1189261.png", 5, 4.0, "SPOT_FIXED"),
    ("1189261.png", 6, 4.0, "SPOT_FIXED"),
    ("1189261.png", 7, 5.0, "SPOT_FIXED"),
    ("1189261.png", 7, 6.0, "SPOT_FIXED"),
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

/// Variant function: (distance, mask_median) → optional override table.
/// `None` means default (no override; W44-29 auto runs as usual).
/// Implementations MUST gate themselves on `mask < 50` for cells
/// where the existing W44-29 gate would also fire — otherwise we
/// over-apply and regress non-target cells.
type Variant = (&'static str, fn(f32, Option<f32>) -> Option<EntropyMulTable>);

fn variant_default(_d: f32, _m: Option<f32>) -> Option<EntropyMulTable> {
    None
}

/// `true` when the W44-29 auto gate would fire (distance + mask).
/// Mirrors `vardct/encoder.rs:2684` (`HIGH_D_PHOTO_MIN_DISTANCE = 3.0`,
/// `HIGH_D_PHOTO_SMOOTH_THRESHOLD = 50.0`).
fn w44_29_would_fire(d: f32, mask: Option<f32>) -> bool {
    d >= 3.0 && mask.is_some_and(|m| m < 50.0)
}

/// W: stronger dct32x32 only where W44-29 would fire.
fn variant_w(d: f32, m: Option<f32>) -> Option<EntropyMulTable> {
    if !w44_29_would_fire(d, m) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed();
    t.dct32x32 = 1.27;
    t.dct16x32 = 1.27 * (1.49 / 1.48);
    Some(t)
}

/// X: per-distance dct16x32 only where W44-29 would fire.
fn variant_x(d: f32, m: Option<f32>) -> Option<EntropyMulTable> {
    if !w44_29_would_fire(d, m) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed();
    if d >= 5.0 {
        t.dct16x32 = 1.40;
    }
    Some(t)
}

/// Y: W + X combined only where W44-29 would fire.
fn variant_y(d: f32, m: Option<f32>) -> Option<EntropyMulTable> {
    if !w44_29_would_fire(d, m) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed();
    t.dct32x32 = 1.27;
    if d >= 5.0 {
        t.dct16x32 = 1.40;
    } else {
        t.dct16x32 = 1.27 * (1.49 / 1.48);
    }
    Some(t)
}

/// Z: even stronger dct32x32, only where W44-29 would fire.
fn variant_z(d: f32, m: Option<f32>) -> Option<EntropyMulTable> {
    if !w44_29_would_fire(d, m) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed();
    t.dct32x32 = 1.20;
    t.dct16x32 = 1.20 * (1.49 / 1.48);
    Some(t)
}

/// XN: per-distance dct16x32 at d>=6 only (narrower than X)
fn variant_xn(d: f32, m: Option<f32>) -> Option<EntropyMulTable> {
    if !w44_29_would_fire(d, m) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed();
    if d >= 6.0 {
        t.dct16x32 = 1.40;
    }
    Some(t)
}

/// WX_d6: W + X combined, X only at d>=6 (preserves 1531677 d=5 SSIM2)
fn variant_wx_d6(d: f32, m: Option<f32>) -> Option<EntropyMulTable> {
    if !w44_29_would_fire(d, m) {
        return None;
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed();
    t.dct32x32 = 1.27;
    if d >= 6.0 {
        t.dct16x32 = 1.40;
    } else {
        t.dct16x32 = 1.27 * (1.49 / 1.48);
    }
    Some(t)
}

const VARIANTS: &[Variant] = &[
    ("default", variant_default),
    ("W_dct32_127", variant_w),
    ("X_dct16x32_per_d", variant_x),
    ("Y_combined", variant_y),
    ("Z_dct32_120", variant_z),
    ("XN_dct16x32_d6", variant_xn),
    ("WX_d6", variant_wx_d6),
];

/// Per-image mask1x1 median (encoder-pipeline measured, from W44-78 table).
fn known_mask_median(image: &str) -> Option<f32> {
    match image {
        "1420710.png" => Some(39.55),
        "1531677.png" => Some(35.63),
        "1189261.png" => Some(69.08),
        "1418519.png" => Some(92.33),
        "1025469.png" => Some(76.08),
        "1044329.png" => Some(48.03),
        _ => None,
    }
}

fn encode_with_variant(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mask: Option<f32>,
    variant: &Variant,
) -> Result<Vec<u8>, String> {
    let mut cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    let v_table = (variant.1)(d, mask);
    if let Some(t) = v_table {
        // We claim the W44-29 firing position; disable auto-fire so
        // the encoder doesn't double-swap, and inject our stronger
        // table via internal_params.
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
        "/tmp/w44_94_cjxl_{}_{}_{}.jxl",
        std::process::id(),
        effort,
        (d * 10.0) as u32
    );
    let out = Command::new(CJXL_BIN)
        .args([
            "-d",
            &d.to_string(),
            "-e",
            &effort.to_string(),
            src,
            &tmp,
        ])
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
        .map(|c| RGB::new(srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])))
        .collect();
    Img::new(lin, w as usize, h as usize)
}

fn main() {
    eprintln!("W44-94 widen W44-77/W44-29 tightening A/B sweep");
    eprintln!("Variants: {:?}", VARIANTS.iter().map(|v| v.0).collect::<Vec<_>>());
    eprintln!("Cells: {} (13 OPEN + 6 W93_REGR + 13 SPOT_FIXED)", CELLS.len());

    let params = ButteraugliParams::default();

    // Header
    let mut hdr = String::from("class\timage\teffort\tdistance\tcjxl_bytes");
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
    let mut images_cache: BTreeMap<String, (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>)> =
        BTreeMap::new();

    let n_cells = CELLS.len();
    for (i, &(image, effort, dist, class)) in CELLS.iter().enumerate() {
        eprintln!("[{}/{}] {} e{} d={}  ({})", i + 1, n_cells, image, effort, dist, class);
        let path = PathBuf::from(CID22).join(image);

        let (w, h, raw, orig_linear_img, orig_srgb_img) =
            images_cache.entry(image.to_string()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let linear = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> =
                    rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
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

        let mask = known_mask_median(image);
        let mut variant_results: Vec<Measure> = Vec::with_capacity(VARIANTS.len());
        let mut default_ssim2: f64 = f64::NAN;
        for (idx, v) in VARIANTS.iter().enumerate() {
            let bytes = match encode_with_variant(raw, *w, *h, effort, dist, mask, v) {
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

        // FIXED/OPEN threshold: +3.0% of cjxl bytes (matches gate from task)
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

        println!(
            "{}\t{}\t{}\t{:.1}\t{}\t{}",
            class,
            image,
            effort,
            dist,
            cjxl_b,
            cols.join("\t")
        );
    }

    eprintln!("\n=== W44-94 aggregate ===");
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
