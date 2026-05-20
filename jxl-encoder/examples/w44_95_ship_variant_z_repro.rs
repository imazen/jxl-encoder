// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-95 ship-variant-Z reproducer.
//!
//! Verifies that variant Z's 6 OPEN closures from the W44-94 A/B sweep
//! reproduce on current main after editing
//! [`EntropyMulTable::high_d_photo_smooth_suppressed`] from
//! `dct32x32=1.34` → `dct32x32=1.20` (variant Z values).
//!
//! Compares two configurations on the same 19 cells (13 OPEN + 6 W93_REGR):
//!  - **baseline**: forces the OLD W44-29 values (dct32x32=1.34,
//!    dct16x32=1.349, dct16x16=1.27) via `LossyInternalParams` override
//!    + `with_high_d_photo_hint(Some(false))` to suppress the auto-gate
//!    (otherwise the new production default would override the injected
//!    table). This matches W44-94's "default" column.
//!  - **shipped**: stock production default — the auto-gate naturally
//!    fires and applies the NEW table values (dct32x32=1.20).
//!
//! Acceptance:
//!  - 6 OPEN cells with `default ≥ +3.0 % bytes vs cjxl` close (`<+3.0 %`).
//!  - Zero FIXED→OPEN flips on W93_REGR cells (they are mask≥50, gate
//!    never fires on either side, so they MUST be byte-identical).
//!  - SSIM2 regression ≤ 0.3 on every cell.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_95_ship_variant_z_repro

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
    // === 13 W44-92 OPEN cells (Z closes 6 of these per W44-94) ===
    ("1420710.png", 5, 5.0, "OPEN"),
    ("1420710.png", 5, 6.0, "OPEN"),
    ("1420710.png", 6, 5.0, "OPEN"), // Z closes
    ("1420710.png", 6, 6.0, "OPEN"), // Z closes
    ("1420710.png", 7, 5.0, "OPEN"),
    ("1420710.png", 8, 5.0, "OPEN"), // Z closes
    ("1420710.png", 9, 5.0, "OPEN"), // Z closes
    ("1531677.png", 5, 5.0, "OPEN"),
    ("1531677.png", 5, 6.0, "OPEN"), // Z closes
    ("1531677.png", 6, 5.0, "OPEN"),
    ("1531677.png", 6, 6.0, "OPEN"), // Z closes
    ("1531677.png", 8, 5.0, "OPEN"),
    ("1531677.png", 9, 5.0, "OPEN"),
    // === 6 W44-93-regressed FIXED cells (must stay byte-identical) ===
    ("1189261.png", 5, 6.0, "W93_REGR"),
    ("1189261.png", 6, 6.0, "W93_REGR"),
    ("1418519.png", 5, 5.0, "W93_REGR"),
    ("1418519.png", 5, 6.0, "W93_REGR"),
    ("1418519.png", 6, 5.0, "W93_REGR"),
    ("1418519.png", 6, 6.0, "W93_REGR"),
];

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
}

/// The OLD W44-29 values shipped before W44-95.
fn old_w44_29_table() -> EntropyMulTable {
    let mut t = EntropyMulTable::reference();
    t.dct16x16 = 1.27;
    t.dct32x32 = 1.34;
    t.dct16x32 = 1.34 * (1.49 / 1.48);
    t
}

fn encode_baseline(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
    // Force the OLD W44-29 dct32x32=1.34 values via internal_params,
    // disabling the auto-gate so the new production default doesn't override it.
    let mut cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { high_d_photo_hint: Some(false), ..Default::default() });
    let mut internal = LossyInternalParams::default();
    internal.entropy_mul_table = Some(old_w44_29_table());
    cfg = cfg.with_internal_params(internal);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

fn encode_shipped(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
    // Stock production default — auto-gate fires and uses the new
    // `high_d_photo_smooth_suppressed()` table (W44-95 variant Z values).
    LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .encode(rgb, w, h, PixelLayout::Rgb8)
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
        "/tmp/w44_95_cjxl_{}_{}_{}.jxl",
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
    eprintln!("W44-95 ship-variant-Z reproducer");
    eprintln!("Cells: {} (13 OPEN + 6 W93_REGR)", CELLS.len());
    eprintln!("baseline = old W44-29 (dct32x32=1.34) forced via internal_params");
    eprintln!("shipped  = new W44-95 production default (dct32x32=1.20, variant Z)");

    let params = ButteraugliParams::default();

    println!(
        "class\timage\teffort\tdistance\tcjxl_bytes\tbaseline_bytes\tbaseline_pct\tbaseline_bfly\tbaseline_ssim2\tshipped_bytes\tshipped_pct\tshipped_bfly\tshipped_ssim2\tbytes_delta\tssim2_delta\tclosed_status"
    );

    // Aggregates
    let mut open_closed = 0;
    let mut w93_byte_identical = 0;
    let mut w93_byte_diff = 0;
    let mut max_ssim2_drop: f64 = 0.0;
    let mut total_baseline_bytes: i64 = 0;
    let mut total_shipped_bytes: i64 = 0;

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

        let baseline_bytes = match encode_baseline(raw, *w, *h, effort, dist) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  baseline encode failed: {}", e);
                continue;
            }
        };
        let shipped_bytes = match encode_shipped(raw, *w, *h, effort, dist) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  shipped encode failed: {}", e);
                continue;
            }
        };

        let baseline_m = match measure(
            baseline_bytes,
            *w,
            *h,
            orig_linear_img,
            orig_srgb_img,
            &params,
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  baseline measure failed: {}", e);
                continue;
            }
        };
        let shipped_m = match measure(
            shipped_bytes,
            *w,
            *h,
            orig_linear_img,
            orig_srgb_img,
            &params,
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  shipped measure failed: {}", e);
                continue;
            }
        };

        let baseline_pct = 100.0 * (baseline_m.bytes as f64 - cjxl_b as f64) / cjxl_b as f64;
        let shipped_pct = 100.0 * (shipped_m.bytes as f64 - cjxl_b as f64) / cjxl_b as f64;
        let bytes_delta = shipped_m.bytes as i64 - baseline_m.bytes as i64;
        let ssim2_delta = shipped_m.ssim2 - baseline_m.ssim2;
        max_ssim2_drop = max_ssim2_drop.min(ssim2_delta);
        total_baseline_bytes += baseline_m.bytes as i64;
        total_shipped_bytes += shipped_m.bytes as i64;

        let status = if class == "OPEN" {
            if baseline_pct >= 3.0 && shipped_pct < 3.0 {
                open_closed += 1;
                "OPEN→FIXED"
            } else if baseline_pct < 3.0 && shipped_pct >= 3.0 {
                "FIXED→OPEN!!!"
            } else if baseline_pct >= 3.0 {
                "still OPEN"
            } else {
                "already FIXED"
            }
        } else if class == "W93_REGR" {
            if baseline_m.bytes == shipped_m.bytes {
                w93_byte_identical += 1;
                "byte-identical"
            } else {
                w93_byte_diff += 1;
                "BYTES DIFFER"
            }
        } else {
            "?"
        };

        println!(
            "{}\t{}\te{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:+}\t{:+.4}\t{}",
            class,
            image,
            effort,
            dist,
            cjxl_b,
            baseline_m.bytes,
            baseline_pct,
            baseline_m.butteraugli,
            baseline_m.ssim2,
            shipped_m.bytes,
            shipped_pct,
            shipped_m.butteraugli,
            shipped_m.ssim2,
            bytes_delta,
            ssim2_delta,
            status
        );
    }

    eprintln!();
    eprintln!("=== W44-95 RESULT SUMMARY ===");
    eprintln!("OPEN cells closed (Z reproduces wins): {}", open_closed);
    eprintln!(
        "W93_REGR byte-identical: {} (must be 6 for PASS)",
        w93_byte_identical
    );
    eprintln!("W93_REGR byte-diff: {} (must be 0 for PASS)", w93_byte_diff);
    eprintln!(
        "Worst SSIM2 drop: {:.4} (must be >= -0.30 for PASS)",
        max_ssim2_drop
    );
    eprintln!(
        "Total bytes: baseline {} → shipped {} (Δ {:+})",
        total_baseline_bytes,
        total_shipped_bytes,
        total_shipped_bytes - total_baseline_bytes
    );
}
