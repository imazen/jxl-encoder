//! W44-29 paired baseline-vs-hint validation for the
//! `LossyConfig::with_high_d_photo_hint` gate.
//!
//! Cell coverage:
//! - **F-D residual photos** (e=5, d∈{4,5,6}): expect default (None=auto)
//!   to FIRE on smooth-photo content and close 2-5 of 5 cells.
//! - **Screenshots** (e=7, d∈{3,4,5,6} × {imac_g3, terminal}): expect
//!   default (None=auto) to NOT fire (median(mask1x1) > SMOOTH_THRESHOLD)
//!   and produce byte-identical output.
//! - **Photo control** (d=1.0): expect default to NOT fire (d < 4.0) and
//!   produce byte-identical output.
//!
//! Compares 3 modes per cell: `baseline` (no hint, on `main@origin`),
//! `auto` (default None — current branch, gate decides), `forced_on`
//! (Some(true), proves the lower-table is what's doing the work),
//! `forced_off` (Some(false), proves the suppression bypass works).
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_29_paired_ab

#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::PathBuf;

/// F-D residual photo cells (per W44-28 honest-stop + W44-1 baseline).
/// Expected behaviour: AUTO fires (smooth-photo, d >= 4) → bytes drop.
const FD_CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "cid22/1531677",
        "CID22/CID22-512/validation/1531677.png",
        5,
        4.0,
    ),
    (
        "cid22/1531677",
        "CID22/CID22-512/validation/1531677.png",
        5,
        5.0,
    ),
    (
        "cid22/1531677",
        "CID22/CID22-512/validation/1531677.png",
        5,
        6.0,
    ),
    (
        "cid22/1420710",
        "CID22/CID22-512/validation/1420710.png",
        5,
        5.0,
    ),
    (
        "cid22/1420710",
        "CID22/CID22-512/validation/1420710.png",
        5,
        6.0,
    ),
];

/// Screenshot cells (per W44-28 regression panel).
/// Expected behaviour: AUTO does NOT fire (mask1x1 > SMOOTH_THRESHOLD) →
/// byte-identical to baseline.
const SCREEN_CELLS: &[(&str, &str, u8, f32)] = &[
    ("gb82/imac_g3", "gb82-sc/imac_g3.png", 7, 3.0),
    ("gb82/imac_g3", "gb82-sc/imac_g3.png", 7, 4.0),
    ("gb82/imac_g3", "gb82-sc/imac_g3.png", 7, 5.0),
    ("gb82/imac_g3", "gb82-sc/imac_g3.png", 7, 6.0),
    ("gb82/terminal", "gb82-sc/terminal.png", 7, 3.0),
    ("gb82/terminal", "gb82-sc/terminal.png", 7, 4.0),
    ("gb82/terminal", "gb82-sc/terminal.png", 7, 5.0),
    ("gb82/terminal", "gb82-sc/terminal.png", 7, 6.0),
];

/// Photo control at d=1.0: AUTO does NOT fire (d < 4.0) → byte-identical.
const PHOTO_CONTROL_CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "cid22/1531677",
        "CID22/CID22-512/validation/1531677.png",
        7,
        1.0,
    ),
    (
        "cid22/1420710",
        "CID22/CID22-512/validation/1420710.png",
        7,
        1.0,
    ),
];

#[derive(Copy, Clone, Debug)]
enum Mode {
    Baseline,
    Auto,
    ForcedOn,
    ForcedOff,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Baseline => "baseline",
            Mode::Auto => "auto",
            Mode::ForcedOn => "forced_on",
            Mode::ForcedOff => "forced_off",
        }
    }
    fn hint(self) -> Option<Option<bool>> {
        match self {
            Mode::Baseline => None,               // do not call with_high_d_photo_hint
            Mode::Auto => Some(None),             // default (None=auto)
            Mode::ForcedOn => Some(Some(true)),   // force-fire
            Mode::ForcedOff => Some(Some(false)), // suppress
        }
    }
}

fn decode_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn encode_with_mode(rgb_u8: &[u8], w: u32, h: u32, effort: u8, d: f32, mode: Mode) -> Vec<u8> {
    let cfg = LossyConfig::new(d).with_effort(effort);
    let cfg = match mode.hint() {
        Some(h) => cfg.with_high_d_photo_hint(h),
        None => cfg,
    };
    cfg.encode(rgb_u8, w, h, PixelLayout::Rgb8).unwrap()
}

fn compute_metrics(
    bytes: &[u8],
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    bparams: &ButteraugliParams,
) -> (usize, f64, f64) {
    let (dw, dh, dec) = decode_linear(bytes).unwrap();
    let dec_pix: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
    let dec_lin = Img::new(dec_pix, dw, dh);
    let bfly = butteraugli_linear(orig_lin.as_ref(), dec_lin.as_ref(), bparams)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);
    let dec_srgb: Vec<[u8; 3]> = dec
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), Img::new(dec_srgb, dw, dh).as_ref())
            .unwrap_or(f64::NAN);
    (bytes.len(), bfly, ssim2)
}

fn run_class(
    out: &mut impl Write,
    class: &str,
    cells: &[(&str, &str, u8, f32)],
    corpus: &PathBuf,
    bparams: &ButteraugliParams,
) {
    eprintln!("\n=== {} ===", class);
    for &(label, rel, effort, d) in cells {
        let path = corpus.join(rel);
        if !path.exists() {
            eprintln!("MISS {}", path.display());
            continue;
        }
        let img = image::open(&path).unwrap();
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8();
        let rgb_u8: &[u8] = rgb.as_raw();
        let linear: Vec<RGB<f32>> = rgb
            .pixels()
            .map(|p| {
                RGB::new(
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                )
            })
            .collect();
        let orig_lin = Img::new(linear, w as usize, h as usize);
        let srgb_arr: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let orig_srgb = Img::new(srgb_arr, w as usize, h as usize);

        let modes = [Mode::Baseline, Mode::Auto, Mode::ForcedOn, Mode::ForcedOff];
        let mut results: Vec<(usize, f64, f64)> = Vec::with_capacity(modes.len());
        for &m in &modes {
            let bytes = encode_with_mode(rgb_u8, w, h, effort, d, m);
            let metrics = compute_metrics(&bytes, &orig_lin, &orig_srgb, bparams);
            results.push(metrics);
        }
        let (bb, bbfly, bssim2) = results[0];
        eprintln!(
            "{:<20} e={} d={:.1}  baseline: {} B  bfly={:.4}  ssim2={:.4}",
            label, effort, d, bb, bbfly, bssim2
        );
        for (i, &m) in modes.iter().enumerate() {
            let (b, bf, s) = results[i];
            let bd_pct = (b as f64 - bb as f64) / bb as f64 * 100.0;
            let bfd_pct = (bf - bbfly) / bbfly.max(1e-9) * 100.0;
            let sd_abs = s - bssim2;
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}\t{:+.4}",
                class,
                label,
                effort,
                d,
                m.label(),
                b,
                bb,
                bf,
                bbfly,
                bd_pct,
                bfd_pct,
                sd_abs
            )
            .unwrap();
            out.flush().ok();
            if !matches!(m, Mode::Baseline) {
                let identical = if b == bb { " IDENTICAL" } else { "" };
                eprintln!(
                    "  {:>10}: {} B  Δbytes={:+.3}% Δbfly={:+.3}% Δssim2={:+.4}{}",
                    m.label(),
                    b,
                    bd_pct,
                    bfd_pct,
                    sd_abs,
                    identical
                );
            }
        }
    }
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );
    let out_path = PathBuf::from("benchmarks/w44_29_paired_ab_2026-05-19.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let staging = PathBuf::from(format!("/tmp/w44_29_paired_ab_{}.tsv", std::process::id()));
    let mut out = std::fs::File::create(&staging).unwrap();
    writeln!(
        out,
        "class\timage\teffort\tdistance\tmode\tbytes\tbase_bytes\tbfly\tbase_bfly\tbytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs"
    )
    .unwrap();
    let bparams = ButteraugliParams::default();

    run_class(&mut out, "FD_PHOTO", FD_CELLS, &corpus, &bparams);
    run_class(&mut out, "SCREENSHOT", SCREEN_CELLS, &corpus, &bparams);
    run_class(
        &mut out,
        "PHOTO_CONTROL",
        PHOTO_CONTROL_CELLS,
        &corpus,
        &bparams,
    );

    drop(out);
    std::fs::rename(&staging, &out_path).unwrap();
    eprintln!("\nTSV: {}", out_path.display());
}
