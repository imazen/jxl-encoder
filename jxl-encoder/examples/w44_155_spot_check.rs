// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-155 spot-check: verify W44-154 cluster preservation in current
//! production main on 33 cells:
//!   * 1418519 d=4/5/6 × e7/e8/e9 — W44-148/W44-152 wins, expected to be
//!     preserved (W44-154 ships variant Z dct32x32=1.22, which doesn't
//!     fire on 1418519 anyway).
//!   * codec_wiki d=3 × e7/e8/e9 — W44-152 collateral wins, expected to
//!     be preserved (gate doesn't fire variant Z there either).
//!   * 1420710 d=4/5/6 × e7/e8/e9 — TARGET (5/6 cells expected to close
//!     under W44-154 micro-bisect).
//!   * 1531677 d=4/5/6 × e7/e8/e9 — verify W44-148/154 wins survive.
//!   * 3 photo controls off-gate — 1189261/1044329/2389166 at e7 d=4
//!     (must be byte-identical to W44-153 baseline).
//!
//! Approach: ONE encode per cell with current production main, compute
//! bytes + butteraugli + ssim2 vs cjxl, write per-cell row to TSV,
//! compare against W44-153 ledger row (image, effort, distance match).
//!
//! NO production code change. Measurement-only.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_155_spot_check

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const SCREENSHOTS: &str = "/home/lilith/work/codec-corpus/gb82-sc";
const CJXL_BIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";

/// (image, effort, distance, class)
type Cell = (&'static str, u8, f32, &'static str);

const CELLS: &[Cell] = &[
    // 1418519 d=4/5/6 × e7/e8/e9 (W44-148/W44-152 wins — preserve)
    ("1418519.png", 7, 4.0, "WIN_1418519"),
    ("1418519.png", 8, 4.0, "WIN_1418519"),
    ("1418519.png", 9, 4.0, "WIN_1418519"),
    ("1418519.png", 7, 5.0, "WIN_1418519"),
    ("1418519.png", 8, 5.0, "WIN_1418519"),
    ("1418519.png", 9, 5.0, "WIN_1418519"),
    ("1418519.png", 7, 6.0, "WIN_1418519"),
    ("1418519.png", 8, 6.0, "WIN_1418519"),
    ("1418519.png", 9, 6.0, "WIN_1418519"),
    // codec_wiki d=3 × e7/e8/e9 (W44-152 collateral wins — preserve)
    ("codec_wiki.png", 7, 3.0, "WIN_CW_D3"),
    ("codec_wiki.png", 8, 3.0, "WIN_CW_D3"),
    ("codec_wiki.png", 9, 3.0, "WIN_CW_D3"),
    // 1420710 d=4/5/6 × e7/e8/e9 (TARGET — 5/6 cells expected closed under W44-154)
    ("1420710.png", 7, 4.0, "TARGET_1420710"),
    ("1420710.png", 8, 4.0, "TARGET_1420710"),
    ("1420710.png", 9, 4.0, "TARGET_1420710"),
    ("1420710.png", 7, 5.0, "TARGET_1420710"),
    ("1420710.png", 8, 5.0, "TARGET_1420710"),
    ("1420710.png", 9, 5.0, "TARGET_1420710"),
    ("1420710.png", 7, 6.0, "TARGET_1420710"),
    ("1420710.png", 8, 6.0, "TARGET_1420710"),
    ("1420710.png", 9, 6.0, "TARGET_1420710"),
    // 1531677 d=4/5/6 × e7/e8/e9 (W44-148 wins — preserve)
    ("1531677.png", 7, 4.0, "WIN_1531677"),
    ("1531677.png", 8, 4.0, "WIN_1531677"),
    ("1531677.png", 9, 4.0, "WIN_1531677"),
    ("1531677.png", 7, 5.0, "WIN_1531677"),
    ("1531677.png", 8, 5.0, "WIN_1531677"),
    ("1531677.png", 9, 5.0, "WIN_1531677"),
    ("1531677.png", 7, 6.0, "WIN_1531677"),
    ("1531677.png", 8, 6.0, "WIN_1531677"),
    ("1531677.png", 9, 6.0, "WIN_1531677"),
    // Photo controls off-gate (must stay byte-identical to W44-153 baseline)
    ("1189261.png", 7, 4.0, "CONTROL_NOGATE"),
    ("1044329.png", 7, 4.0, "CONTROL_NOGATE"),
    ("2389166.png", 7, 4.0, "CONTROL_NOGATE"),
];

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

fn encode_production(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<(Vec<u8>, f64), String> {
    let cfg = LossyConfig::new(d).with_effort(effort).with_threads(8);
    let start = Instant::now();
    let bytes = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((bytes, ms))
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

fn measure_bytes(
    bytes: Vec<u8>,
    encode_ms: f64,
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
        encode_ms,
    })
}

fn cjxl_size_and_bfly_and_ssim2(
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
        "/tmp/w44_155_cjxl_{}_{}_{}.jxl",
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
    let out_dir = PathBuf::from("/home/lilith/work/zen/jxl-encoder/benchmarks");
    std::fs::create_dir_all(&out_dir).ok();
    let tsv_path = out_dir.join("w44_155_spot_check_2026-05-21.tsv");

    eprintln!("W44-155 spot-check: {} cells", CELLS.len());
    eprintln!("TSV out: {}", tsv_path.display());

    let mut tsv = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tsv_path)
        .expect("open tsv");
    writeln!(
        tsv,
        "class\timage\teffort\tdistance\tours_bytes\tcjxl_bytes\tbytes_delta_pct\tours_bfly\tcjxl_bfly\tbfly_delta_pct\tours_ssim2\tcjxl_ssim2\tssim2_delta_abs\tours_encode_ms"
    )
    .expect("write header");

    let params = ButteraugliParams::default();
    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    // Aggregate stats per class
    let mut class_stats: BTreeMap<&'static str, (f64, f64, f64, i64)> = BTreeMap::new();
    let byte_identical_violations: Vec<String> = Vec::new();
    // Track per-class baseline (CONTROL cells must match prior measurement;
    // for non-control, we just print observed values).
    // The CONTROL baseline is the cjxl_bytes value from W44-153 ledger;
    // we'll spot-check whether our bytes match the prior W44-153 row in
    // the meta-analysis phase, not here.

    for (i, &(image, effort, dist, class)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} e{} d={}  ({})",
            i + 1,
            CELLS.len(),
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

        let (cjxl_b, cjxl_bfly, cjxl_ssim2) = match cjxl_size_and_bfly_and_ssim2(
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

        let (ours_bytes_vec, ours_ms) = match encode_production(raw, *w, *h, effort, dist) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  ours encode failed: {}", e);
                continue;
            }
        };
        let ours_bytes_n = ours_bytes_vec.len();
        let m = match measure_bytes(
            ours_bytes_vec,
            ours_ms,
            *w,
            *h,
            orig_linear_img,
            orig_srgb_img,
            &params,
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  ours measure failed: {}", e);
                continue;
            }
        };

        let bytes_pct = (m.bytes as f64 - cjxl_b as f64) / cjxl_b as f64 * 100.0;
        let bfly_pct = if cjxl_bfly > 0.0 {
            (m.butteraugli - cjxl_bfly) / cjxl_bfly * 100.0
        } else {
            0.0
        };
        let ssim2_abs = m.ssim2 - cjxl_ssim2;

        writeln!(
            tsv,
            "{}\t{}\t{}\t{:.4}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:+.3}\t{:.4}\t{:.4}\t{:+.4}\t{:.2}",
            class,
            image,
            effort,
            dist,
            m.bytes,
            cjxl_b,
            bytes_pct,
            m.butteraugli,
            cjxl_bfly,
            bfly_pct,
            m.ssim2,
            cjxl_ssim2,
            ssim2_abs,
            m.encode_ms
        )
        .expect("write row");
        tsv.flush().ok();

        let entry = class_stats.entry(class).or_insert((0.0, 0.0, 0.0, 0));
        entry.0 += bytes_pct;
        entry.1 += bfly_pct;
        entry.2 += ssim2_abs;
        entry.3 += 1;

        eprintln!(
            "  bytes={} ({:+.2}%) bfly={:.3} ({:+.2}%) ssim2={:.3} ({:+.3}) ms={:.0}",
            m.bytes, bytes_pct, m.butteraugli, bfly_pct, m.ssim2, ssim2_abs, m.encode_ms
        );
        let _ = ours_bytes_n;

        // refresh marker every cell
        if let Ok(mut f) = std::fs::File::create("/home/lilith/work/zen/jxl-encoder/.workongoing") {
            use std::io::Write;
            let _ = writeln!(
                f,
                "{} claude-w44-155 W44-155 spot-check cell {}/{}",
                chrono_iso_now(),
                i + 1,
                CELLS.len()
            );
        }
    }

    eprintln!("\n=== W44-155 per-class summary (mean per cell) ===");
    for (class, (b, bf, s, n)) in &class_stats {
        if *n == 0 {
            continue;
        }
        let nf = *n as f64;
        eprintln!(
            "  {:<20}  n={}  Δbytes={:+.2}%  Δbfly={:+.2}%  Δssim2={:+.3}",
            class, n, b / nf, bf / nf, s / nf
        );
    }

    if !byte_identical_violations.is_empty() {
        eprintln!("\n=== Byte-identity VIOLATIONS (vs W44-153 baseline expected) ===");
        for s in &byte_identical_violations {
            eprintln!("  {}", s);
        }
    }

    eprintln!("\nTSV: {}", tsv_path.display());
}

fn chrono_iso_now() -> String {
    // Lightweight ISO-8601 UTC formatter without chrono dep.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Convert to YYYY-MM-DDTHH:MM:SSZ. Crude impl — good enough for marker file.
    let days = secs / 86400;
    let secs_today = secs % 86400;
    let h = secs_today / 3600;
    let m = (secs_today % 3600) / 60;
    let s = secs_today % 60;
    // Days since 1970-01-01 → Y-M-D. Use simple algorithm (correct for 2026).
    let (y, mo, d) = days_to_ymd(days as i64);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
}

fn days_to_ymd(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let mo = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d)
}
