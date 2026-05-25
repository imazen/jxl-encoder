//! W44-AUDIT-8 Phase 6 — 2-cell bisect of `QuantizeWP` shape.
//!
//! Tests: clic_22ea12 e7 d=4 (Phase 5 closed to +0.94 SSIM2 vs cjxl)
//! and clic_22ea12 e9 d=4 (Phase 5 untouched — extra_dc_precision=0 at e≥8).
//!
//! Phase 5 set `extra_dc_precision = 1` at e≤7 only (scale-only fix).
//! Phase 6 layers libjxl's QuantizeWP shape (WP-relative residual +
//! 0.62 deadzone + snap-to-even) on TOP of Phase 5, at the same
//! `effort ≤ 7` gate.
//!
//! Acceptance:
//!   - e7 cell: SSIM2 within ±0.5 of Phase 5 (additive or neutral)
//!     AND bytes within +5 % of Phase 5.
//!   - e9 cell: structurally unchanged (Phase 6 gate also OFF at e≥8).
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_8_phase6_bisect

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const CELLS: &[(&str, &str, u8, f32, &str)] = &[
    (
        "clic_22ea12",
        "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
        7,
        4.0,
        "PRIMARY_E7",
    ),
    (
        "clic_22ea12",
        "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
        9,
        4.0,
        "PRIMARY_E9",
    ),
    // Additional PROTECT_E8 spot check (must stay structurally byte-identical
    // at e=8 since both Phase 5 gate AND Phase 6 gate fire at e≤7 only).
    (
        "terminal",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        8,
        4.0,
        "PROTECT_E8",
    ),
    (
        "codec_wiki",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        8,
        4.0,
        "PROTECT_E8",
    ),
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
) -> Option<(Vec<u8>, u128)> {
    // Phase 6 default-flip HONEST-STOPPED — force the QuantizeWP gate ON
    // via the documented opt-in path (env hook respected by both
    // primary callers + this bench example).
    //
    // SAFETY: this env is process-local; bench runs serially.
    // SAFETY: the example is single-threaded for env access.
    unsafe {
        std::env::set_var("JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP", "1");
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    let t0 = Instant::now();
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .ok()?;
    unsafe {
        std::env::remove_var("JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP");
    }
    Some((buf, t0.elapsed().as_millis()))
}

fn encode_cjxl(src: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_p6_{}_{}_{}_{}.jxl",
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
    println!("# W44-AUDIT-8 Phase 6 bisect");
    println!(
        "image_id\trole\teffort\tdistance\tours_bytes\tours_bfly\tours_ssim2\tcjxl_bytes\tcjxl_bfly\tcjxl_ssim2\tdbytes_pct\tdbfly\tdssim2"
    );
    for &(id, path, effort, distance, role) in CELLS {
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

        let (ours_buf, _ours_ms) = match encode_zenjxl(&pixels, w, h, effort, distance) {
            Some(x) => x,
            None => {
                eprintln!("ours encode failed {} e{} d={}", id, effort, distance);
                continue;
            }
        };
        let (ours_bfly, ours_ssim2) =
            score(&ours_buf, &lin_img, &srgb_img, w, h).unwrap_or((f64::NAN, f64::NAN));
        let ours_bytes = ours_buf.len();

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

        let dbytes_pct = 100.0 * (ours_bytes as f64 - cjxl_bytes as f64) / cjxl_bytes as f64;
        let dbfly = ours_bfly - cjxl_bfly;
        let dssim2 = ours_ssim2 - cjxl_ssim2;
        println!(
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:+.3}%\t{:+.4}\t{:+.4}",
            id,
            role,
            effort,
            distance,
            ours_bytes,
            ours_bfly,
            ours_ssim2,
            cjxl_bytes,
            cjxl_bfly,
            cjxl_ssim2,
            dbytes_pct,
            dbfly,
            dssim2
        );
    }
}
