//! W44-AUDIT-8 Phase 7 — env-disable bisect on the worst e9 SSIM2 deficit cell.
//!
//! Phase 5 (`extra_dc_precision=1` at e≤7) recovered +0.187 mean SSIM2 on the
//! 30-cell cluster but is structurally OFF at e≥8 (libjxl `nl_dc =
//! speed_tier < kFalcon` parity). e9 cluster cells remain at -1.0 to -3.5
//! SSIM2 vs cjxl.
//!
//! Worst e9 cell: clic_22ea12 e9 d=4 at -3.54 SSIM2 (Phase-5 post-state).
//! At e8 the buttloop fires (4 iterations); at e9 it fires (4 iters too).
//! Hypothesis to test via env hooks: any of W44-117 (EPF seed in buttloop),
//! W44-120 (EPF distance gate), W44-156 (variant Z distance-aware lift),
//! W44-166 (variant Z admit), W44-167 (variant Z Mode D split),
//! W44-176 (buttloop iter-count), AUDIT-5-P3 (Mode D), AUDIT-6 (M3 gate),
//! W44-201 (coeff-order savings), or AUDIT-8-P6 (QuantizeWP shape OPT-IN)
//! moves SSIM2 meaningfully when toggled.
//!
//! Each row of the output corresponds to a single env-hook variant.
//! Baseline is the default-build production behaviour.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_8_phase7_e9_env_bisect

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

/// (env_var, value_to_set, label)
/// Empty value means "do not set" — used for the BASELINE row.
const ENV_VARIANTS: &[(&str, &str, &str)] = &[
    ("", "", "BASELINE"),
    ("JXL_W44_117_DISABLE", "1", "W44_117_DISABLE_EPF_SEED"),
    (
        "JXL_W44_120_EPF_SEED_MIN_DISTANCE",
        "99",
        "W44_120_EPF_NEVER",
    ),
    (
        "JXL_W44_AUDIT_5_P3_DISABLE",
        "1",
        "AUDIT_5_P3_DISABLE_MODE_D",
    ),
    ("JXL_W44_AUDIT_6_DISABLE", "1", "AUDIT_6_DISABLE_M3_GATE"),
    ("JXL_W44_156_DISABLE", "1", "W44_156_DISABLE_DIST_VARZ"),
    (
        "JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP",
        "1",
        "AUDIT_8_P6_FORCE_WP",
    ),
    ("JXL_W44_176_DISABLE", "1", "W44_176_DISABLE_BUTTLOOP_ITER"),
    ("JXL_W44_142_SUPPRESS_DISABLE", "1", "W44_142_DISABLE"),
    ("JXL_W44_152_DISABLE", "1", "W44_152_DISABLE"),
    ("JXL_W44_151_DISABLE", "1", "W44_151_DISABLE"),
    (
        "JXL_W44_166_VARIANT_Z_ADMIT_MODE",
        "off",
        "W44_166_VARIANT_Z_OFF",
    ),
    ("JXL_W44_167_MODE", "off", "W44_167_MODE_OFF"),
    ("JXL_W44_168_MODE", "off", "W44_168_MODE_OFF"),
    (
        "JXL_W44_171_FORCE_TRIAL_ALL_EFFORTS",
        "1",
        "W44_171_TRIAL_ALL",
    ),
    (
        "JXL_W44_172_FORCE_VARIABLE_AT_E8",
        "1",
        "W44_172_VARIABLE_AT_E8",
    ),
    (
        "__JXL_W44_57_FORCE_VARIABLE",
        "1",
        "W44_57_FORCE_VARIABLE_DC_TREE",
    ),
    (
        "__JXL_W44_57_FORCE_FIXED",
        "1",
        "W44_57_FORCE_FIXED_DC_TREE",
    ),
    (
        "JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS",
        "1",
        "W44_201_LEGACY_LARGE_BUCKETS",
    ),
    (
        "JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS",
        "1",
        "W44_205_LEGACY_MEDIUM_BUCKETS",
    ),
];

const CELLS: &[(&str, &str, u8, f32)] = &[(
    "clic_22ea12",
    "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
    9,
    4.0,
)];

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
    env_var: &str,
    env_val: &str,
) -> Option<(Vec<u8>, u128)> {
    if !env_var.is_empty() {
        // SAFETY: process-local, bench is single-threaded.
        unsafe {
            std::env::set_var(env_var, env_val);
        }
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    let t0 = Instant::now();
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .ok()?;
    if !env_var.is_empty() {
        unsafe {
            std::env::remove_var(env_var);
        }
    }
    Some((buf, t0.elapsed().as_millis()))
}

fn encode_cjxl(src: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_p7_{}_{}_{}_{}.jxl",
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
    println!("# W44-AUDIT-8 Phase 7 e9 env-bisect (worst e9 deficit cell)");
    println!(
        "image_id\teffort\tdistance\tvariant\tours_bytes\tours_bfly\tours_ssim2\tcjxl_bytes\tcjxl_bfly\tcjxl_ssim2\tdbytes_pct\tdbfly\tdssim2"
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

        for &(env_var, env_val, label) in ENV_VARIANTS {
            let (ours_buf, _ours_ms) =
                match encode_zenjxl(&pixels, w, h, effort, distance, env_var, env_val) {
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
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:+.3}\t{:+.4}\t{:+.4}",
                id,
                effort,
                distance,
                label,
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
}
