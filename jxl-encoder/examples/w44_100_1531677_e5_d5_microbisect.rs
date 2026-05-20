// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-100 micro-bisect of dct16x32 lift values {1.23, 1.24, 1.25} on the
//! last OPEN cell (1531677 e5 d=5, at +3.090% under W44-99 LC_1.22).
//!
//! W44-99 (`cb63f216`) bisected {1.22, 1.25, 1.27, 1.28, 1.30} and picked
//! 1.22 because LC_125 had a worst-cell SSIM2 delta of -0.2874 (close to
//! the 0.30 budget), while LC_122 had only -0.0100. But LC_125 *would*
//! close 1531677 e5 d=5 at +3.001% (right at the 3.0% threshold; ledger
//! rule `bytes_delta > 3.0` makes that still OPEN by 0.001 pp).
//!
//! Two unexplored candidates in the LC_122 → LC_125 window: 1.23 and 1.24.
//! Worth a targeted try before pivoting to the multi-day butteraugli-loop
//! e7 promotion plan. The W44-99 LC_127 data shows the cost-model is
//! non-monotonic in this region (1.27 had MORE bytes than 1.22), so this
//! micro-bisect is the right shape.
//!
//! Acceptance gates (per task):
//!   - 1531677 e5 d=5 closes (bytes_delta ≤ 3.0% AND/OR bfly_delta ≤ 3.0%)
//!   - Zero FIXED→OPEN regressions on any of 8 1531677 cells
//!   - Worst SSIM2 regression ≤ 0.30 across all firing cells
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_100_1531677_e5_d5_microbisect

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
    // === 4 OPEN cells: the target (1531677 d=5 across efforts) ===
    ("1531677.png", 5, 5.0, "OPEN"),
    ("1531677.png", 6, 5.0, "OPEN"),
    ("1531677.png", 8, 5.0, "OPEN"),
    ("1531677.png", 9, 5.0, "OPEN"),
    // === 4 1531677 SPOT_FIXED (must stay FIXED) ===
    ("1531677.png", 5, 6.0, "SPOT_FIXED"),
    ("1531677.png", 6, 6.0, "SPOT_FIXED"),
    ("1531677.png", 5, 4.0, "SPOT_FIXED"),
    ("1531677.png", 6, 4.0, "SPOT_FIXED"),
    // === 1420710 SPOT_FIXED (HC dispatch — must stay byte-identical) ===
    ("1420710.png", 5, 5.0, "SPOT_FIXED_1420710"),
    ("1420710.png", 6, 5.0, "SPOT_FIXED_1420710"),
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

type Variant = (&'static str, fn(f32, ImageProxy) -> Option<EntropyMulTable>);

fn variant_default(_d: f32, _p: ImageProxy) -> Option<EntropyMulTable> {
    None
}

/// Production W44-99: dct16x32 = 1.22 on LC.
fn variant_lc_122(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.22;
    Some(t)
}

fn variant_lc_123(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.23;
    Some(t)
}

fn variant_lc_124(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.24;
    Some(t)
}

fn variant_lc_125(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if !w44_99_low_colour_would_fire(d, p) {
        return preserve_production_for_non_target(d, p);
    }
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
    t.dct16x32 = 1.25;
    Some(t)
}

fn preserve_production_for_non_target(d: f32, p: ImageProxy) -> Option<EntropyMulTable> {
    if w44_98_high_colour_would_fire(d, p) {
        Some(EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour())
    } else if w44_96_would_fire(d, p) {
        Some(EntropyMulTable::high_d_photo_smooth_suppressed_z())
    } else if w44_29_would_fire(d, p) {
        Some(EntropyMulTable::high_d_photo_smooth_suppressed())
    } else {
        None
    }
}

const VARIANTS: &[Variant] = &[
    ("default", variant_default),
    ("LC_dct16x32_122", variant_lc_122),
    ("LC_dct16x32_123", variant_lc_123),
    ("LC_dct16x32_124", variant_lc_124),
    ("LC_dct16x32_125", variant_lc_125),
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

fn cjxl_size_and_bfly(
    src: &str,
    effort: u8,
    d: f32,
    w: u32,
    h: u32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    params: &ButteraugliParams,
) -> Option<(usize, f64)> {
    let tmp = format!(
        "/tmp/w44_100_cjxl_{}_{}_{}.jxl",
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

    // Decode cjxl output to compute butteraugli
    let (dw, dh, decoded_linear) = decode_jxl_linear(&bytes)?;
    if dw != w as usize || dh != h as usize {
        return Some((sz, f64::NAN));
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);
    Some((sz, bfly))
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
    eprintln!("W44-100 1531677 e5 d=5 micro-bisect — LC dct16x32 {{1.22, 1.23, 1.24, 1.25}}");
    eprintln!(
        "Variants: {:?}",
        VARIANTS.iter().map(|v| v.0).collect::<Vec<_>>()
    );
    eprintln!("Cells: {}", CELLS.len());

    let params = ButteraugliParams::default();

    let mut hdr = String::from("class\timage\teffort\tdistance\tcjxl_bytes\tcjxl_bfly\tgate_fires");
    for v in VARIANTS {
        hdr.push_str(&format!(
            "\t{}_bytes\t{}_bytes_pct\t{}_bfly\t{}_bfly_pct\t{}_ssim2",
            v.0, v.0, v.0, v.0, v.0
        ));
    }
    println!("{}", hdr);

    let mut total_stats: BTreeMap<String, (i64, i64, i64, i64, f64, f64)> = BTreeMap::new();
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

        let (cjxl_b, cjxl_bfly) = match cjxl_size_and_bfly(
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

        // OPEN under ledger rule: bytes_delta > 3.0 AND (bfly_delta > 3.0 OR ssim2_delta < -1.0)
        // We compute the bytes-only gate (the relevant gate after LC lifts hit bfly).
        let fixed_threshold_bytes_pct = 3.0_f64;
        let fixed_threshold_bfly_pct = 3.0_f64;
        let default_bytes_pct = if cjxl_b > 0 {
            (variant_results[0].bytes as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0
        } else {
            0.0
        };
        let default_bfly_pct = if cjxl_bfly > 0.0 {
            (variant_results[0].butteraugli - cjxl_bfly) / cjxl_bfly * 100.0
        } else {
            0.0
        };
        let default_is_open = default_bytes_pct > fixed_threshold_bytes_pct
            && default_bfly_pct > fixed_threshold_bfly_pct;

        let mut cols: Vec<String> = Vec::new();
        for (idx, m) in variant_results.iter().enumerate() {
            let bytes_pct = if cjxl_b > 0 {
                (m.bytes as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0
            } else {
                0.0
            };
            let bfly_pct = if cjxl_bfly > 0.0 {
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

    eprintln!("\n=== W44-100 aggregate ===");
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
