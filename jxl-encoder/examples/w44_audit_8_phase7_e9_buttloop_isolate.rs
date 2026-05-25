//! W44-AUDIT-8 Phase 7 — isolate the buttloop as the e9 cliff source.
//!
//! After the env-bisect ruled out every W44-* discriminator hook (all 20
//! variants produced byte-identical SSIM2 at clic_22ea12 e9 d=4), the
//! remaining structural difference between e7 (SSIM2 +0.94 vs cjxl after
//! Phase 5) and e9 (SSIM2 -3.54 vs cjxl) is `butteraugli_iters`:
//! `effort.rs:1332`
//!
//!   0..=7 => 0,    // no buttloop
//!   8     => 2,
//!   9     => 4,    // e9 = libjxl kTortoise default
//!
//! Test: pin butteraugli_iters=0 at e9 and observe whether SSIM2 closes.
//! If yes → the buttloop is converging to a worse local optimum than the
//! e7 pre-buttloop seed; the deficit is buttloop convergence, not the
//! pre-buttloop pipeline.
//!
//! Cells:
//!   - clic_22ea12 e9 d=4: PRIMARY (worst cluster cell)
//!   - clic_097cb4 e9 d=4: secondary (-2.44 SSIM2)
//!   - clic_100a02 e9 d=4: tertiary (-2.41 SSIM2)
//!
//! Variants per cell:
//!   - BASELINE: production e9 (4 buttloop iters)
//!   - ITERS_0: with_butteraugli_iters(0) — disables the loop entirely
//!   - ITERS_2: with_butteraugli_iters(2) — half the iter count
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_8_phase7_e9_buttloop_isolate

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "clic_22ea12",
        "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
        9,
        4.0,
    ),
    (
        "clic_097cb4",
        "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png",
        9,
        4.0,
    ),
    (
        "clic_100a02",
        "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png",
        9,
        4.0,
    ),
];

/// (label, butteraugli_iters_override)
/// `None` = default (production behaviour).
const VARIANTS: &[(&str, Option<u32>)] = &[
    ("BASELINE_ITERS_4", None),
    ("ITERS_0_DISABLE_LOOP", Some(0)),
    ("ITERS_2_HALF_LOOP", Some(2)),
];

fn srgb_u8_to_linear_f32(x: u8) -> f32 {
    let x = x as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
fn linear_to_srgb_u8(x: f32) -> u8 {
    let x = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (x * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("open png");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.into_raw(), w, h)
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

fn score(
    bytes: &[u8],
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let n = dw * dh;
    let ch = if n > 0 { dec_lin.len() / n } else { 0 };
    if ch < 3 {
        return None;
    }
    let dec_p: Vec<RGB<f32>> = dec_lin
        .chunks_exact(ch)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img = Img::new(dec_p, dw, dh);
    let bfly = butteraugli_linear(
        orig_lin.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks_exact(ch)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn encode_zenjxl(
    pixels: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    iters_override: Option<u32>,
) -> Option<(Vec<u8>, u128)> {
    let mut cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    if let Some(iters) = iters_override {
        cfg = cfg.with_butteraugli_iters(iters);
    }
    let t0 = Instant::now();
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .ok()?;
    Some((buf, t0.elapsed().as_millis()))
}

fn encode_cjxl(src: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_p7iso_{}_{}_{}_{}.jxl",
        src.file_stem()?.to_string_lossy(),
        effort,
        (distance * 1000.0) as u32,
        std::process::id()
    ));
    let t0 = Instant::now();
    let s = Command::new(cjxl)
        .arg(src)
        .arg(&tmp)
        .arg("-e")
        .arg(format!("{}", effort))
        .arg("-d")
        .arg(format!("{}", distance))
        .arg("--quiet")
        .status()
        .ok()?;
    let ms = t0.elapsed().as_millis();
    if !s.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some((bytes, ms))
}

fn main() {
    println!("# W44-AUDIT-8 Phase 7 e9 buttloop isolation");
    println!(
        "image_id\teffort\tdistance\tvariant\tours_bytes\tours_bfly\tours_ssim2\tours_ms\tcjxl_bytes\tcjxl_bfly\tcjxl_ssim2\tdbytes_pct\tdbfly\tdssim2"
    );
    for &(id, path, effort, distance) in CELLS {
        let pathobj = Path::new(path);
        if !pathobj.exists() {
            eprintln!("SKIP (missing): {}", path);
            continue;
        }
        let (pixels, w, h) = load_png(pathobj);
        let lin: Vec<RGB<f32>> = pixels
            .chunks_exact(3)
            .map(|c| {
                RGB::new(
                    srgb_u8_to_linear_f32(c[0]),
                    srgb_u8_to_linear_f32(c[1]),
                    srgb_u8_to_linear_f32(c[2]),
                )
            })
            .collect();
        let srgb: Vec<[u8; 3]> = pixels.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let lin_img = Img::new(lin, w as usize, h as usize);
        let srgb_img = Img::new(srgb, w as usize, h as usize);

        let (cjxl_buf, _cjxl_ms) = match encode_cjxl(pathobj, effort, distance) {
            Some(x) => x,
            None => {
                eprintln!("cjxl encode failed {} e{} d={}", id, effort, distance);
                continue;
            }
        };
        let (cjxl_bfly, cjxl_ssim2) =
            score(&cjxl_buf, &lin_img, &srgb_img, w, h).unwrap_or((f64::NAN, f64::NAN));
        let cjxl_bytes = cjxl_buf.len();

        for &(label, iters_override) in VARIANTS {
            let (ours_buf, ours_ms) =
                match encode_zenjxl(&pixels, w, h, effort, distance, iters_override) {
                    Some(x) => x,
                    None => {
                        eprintln!(
                            "ours encode failed {} e{} d={} variant={}",
                            id, effort, distance, label
                        );
                        continue;
                    }
                };
            let (ours_bfly, ours_ssim2) =
                score(&ours_buf, &lin_img, &srgb_img, w, h).unwrap_or((f64::NAN, f64::NAN));
            let ours_bytes = ours_buf.len();
            let dbytes_pct = 100.0 * (ours_bytes as f64 - cjxl_bytes as f64) / cjxl_bytes as f64;
            let dbfly = ours_bfly - cjxl_bfly;
            let dssim2 = ours_ssim2 - cjxl_ssim2;
            println!(
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:+.3}\t{:+.4}\t{:+.4}",
                id,
                effort,
                distance,
                label,
                ours_bytes,
                ours_bfly,
                ours_ssim2,
                ours_ms,
                cjxl_bytes,
                cjxl_bfly,
                cjxl_ssim2,
                dbytes_pct,
                dbfly,
                dssim2
            );
        }
    }
}
