//! W44-28 follow-on: test alternate top-5 candidates from the main sweep
//! against the screenshot regression gate. The primary winner (dct16=1.34
//! dct32=1.20) closed -8.72 % bytes on F-D but caused +36.5 % butteraugli
//! regression on imac_g3 d=4 (a screenshot at e=7). Check whether any of
//! the next-best candidates (dct16=1.27 dct32=1.20, dct16=1.20 dct32=1.20)
//! avoid the screenshot regression.
//!
//! Set W44_28_DCT16 / W44_28_DCT32 env vars to override the tested tuple.
//!
//! Run:
//!   W44_28_DCT16=1.27 W44_28_DCT32=1.20 \
//!     cargo run -p jxl-encoder --release \
//!       --features '__expert butteraugli-loop ssim2-loop parallel' \
//!       --example w44_28_candidate_check

#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::effort::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const SCREEN_LABELS: &[(&str, &str)] = &[
    ("gb82/terminal", "gb82-sc/terminal.png"),
    ("gb82/codec_wiki", "gb82-sc/codec_wiki.png"),
    ("gb82/imac_g3", "gb82-sc/imac_g3.png"),
];
const SCREEN_DISTANCES: &[f32] = &[3.0, 4.0, 5.0, 6.0];

const FD_CELLS: &[(&str, &str, f32)] = &[
    (
        "cid22/1531677",
        "CID22/CID22-512/validation/1531677.png",
        4.0,
    ),
    (
        "cid22/1420710",
        "CID22/CID22-512/validation/1420710.png",
        6.0,
    ),
];

fn build_table(dct16: f32, dct32: f32) -> EntropyMulTable {
    let mut t = EntropyMulTable::reference();
    t.dct16x16 = dct16;
    t.dct32x32 = dct32;
    t.dct16x32 = dct32 * (1.49 / 1.48);
    t
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

fn main() {
    let dct16: f32 = std::env::var("W44_28_DCT16")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.27);
    let dct32: f32 = std::env::var("W44_28_DCT32")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.20);

    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );

    let out_path = PathBuf::from(format!(
        "benchmarks/w44_28_candidate_dct16_{:.2}_dct32_{:.2}_2026-05-19.tsv",
        dct16, dct32
    ));
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/w44_28_candidate_check_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).expect("create staging");
    writeln!(
        out,
        "stage\timage\tclass\teffort\tdistance\tdct16\tdct32\tbaseline_bytes\tlifted_bytes\tbaseline_bfly\tlifted_bfly\tbaseline_ssim2\tlifted_ssim2\tbytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();

    eprintln!(
        "=== W44-28 candidate check: dct16={:.3} dct32={:.3} ===",
        dct16, dct32
    );

    // F-D residual cells (e=5) — sanity-check that F-D wins still hold.
    eprintln!("\n--- F-D residual cells (e=5) ---");
    for &(label, rel, d) in FD_CELLS {
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

        let t0 = Instant::now();
        let base_bytes = LossyConfig::new(d)
            .with_effort(5)
            .encode(rgb_u8, w, h, PixelLayout::Rgb8)
            .unwrap();
        let _base_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut p = LossyInternalParams::default();
        p.entropy_mul_table = Some(build_table(dct16, dct32));
        let t1 = Instant::now();
        let lift_bytes = LossyConfig::new(d)
            .with_effort(5)
            .with_internal_params(p)
            .encode(rgb_u8, w, h, PixelLayout::Rgb8)
            .unwrap();
        let _lift_ms = t1.elapsed().as_secs_f64() * 1000.0;

        for (tag, b) in [("baseline", &base_bytes), ("lifted", &lift_bytes)] {
            let (dw, dh, dec) = decode_linear(b).unwrap();
            let dec_pix: Vec<RGB<f32>> =
                dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
            let dec_lin = Img::new(dec_pix, dw, dh);
            let bfly = butteraugli_linear(orig_lin.as_ref(), dec_lin.as_ref(), &bparams)
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
            let dec_srgb_img = Img::new(dec_srgb, dw, dh);
            let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref())
                .unwrap_or(f64::NAN);
            eprintln!(
                "  {} {} d={}: {} bytes  bfly={:.4}  ssim2={:.4}",
                label,
                tag,
                d,
                b.len(),
                bfly,
                ssim2
            );
            if tag == "lifted" {
                let _ = (bfly, ssim2);
            }
        }
        // Compute deltas and write
        let (dw, dh, dec_b) = decode_linear(&base_bytes).unwrap();
        let dec_b_pix: Vec<RGB<f32>> = dec_b
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let dec_b_lin = Img::new(dec_b_pix, dw, dh);
        let b_bfly = butteraugli_linear(orig_lin.as_ref(), dec_b_lin.as_ref(), &bparams)
            .map(|r| r.score as f64)
            .unwrap();
        let dec_b_srgb: Vec<[u8; 3]> = dec_b
            .chunks(3)
            .map(|c| {
                [
                    linear_to_srgb_u8(c[0]),
                    linear_to_srgb_u8(c[1]),
                    linear_to_srgb_u8(c[2]),
                ]
            })
            .collect();
        let b_ssim2 = fast_ssim2::compute_ssimulacra2(
            orig_srgb.as_ref(),
            Img::new(dec_b_srgb, dw, dh).as_ref(),
        )
        .unwrap();

        let (_dw, _dh, dec_l) = decode_linear(&lift_bytes).unwrap();
        let dec_l_pix: Vec<RGB<f32>> = dec_l
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let dec_l_lin = Img::new(dec_l_pix, dw, dh);
        let l_bfly = butteraugli_linear(orig_lin.as_ref(), dec_l_lin.as_ref(), &bparams)
            .map(|r| r.score as f64)
            .unwrap();
        let dec_l_srgb: Vec<[u8; 3]> = dec_l
            .chunks(3)
            .map(|c| {
                [
                    linear_to_srgb_u8(c[0]),
                    linear_to_srgb_u8(c[1]),
                    linear_to_srgb_u8(c[2]),
                ]
            })
            .collect();
        let l_ssim2 = fast_ssim2::compute_ssimulacra2(
            orig_srgb.as_ref(),
            Img::new(dec_l_srgb, dw, dh).as_ref(),
        )
        .unwrap();

        let bd =
            (lift_bytes.len() as f64 - base_bytes.len() as f64) / base_bytes.len() as f64 * 100.0;
        let bfd = (l_bfly - b_bfly) / b_bfly.max(1e-9) * 100.0;
        let sd = l_ssim2 - b_ssim2;
        writeln!(out, "FD\t{}\tphoto\t5\t{:.3}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.3}\t{:+.3}\t{:+.4}",
            label, d, dct16, dct32, base_bytes.len(), lift_bytes.len(), b_bfly, l_bfly, b_ssim2, l_ssim2, bd, bfd, sd).unwrap();
        out.flush().unwrap();
        eprintln!(
            "  → Δbytes={:+.2}% Δbfly={:+.2}% Δssim2={:+.4}",
            bd, bfd, sd
        );
    }

    eprintln!("\n--- Screenshot regression check (e=7) ---");
    for &(slabel, srel) in SCREEN_LABELS {
        let spath = corpus.join(srel);
        if !spath.exists() {
            eprintln!("MISS");
            continue;
        }
        let img = image::open(&spath).unwrap();
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
        eprintln!("--- {} ({}x{}) ---", slabel, w, h);
        for &d in SCREEN_DISTANCES {
            let base = LossyConfig::new(d)
                .with_effort(7)
                .encode(rgb_u8, w, h, PixelLayout::Rgb8)
                .unwrap();
            let mut p = LossyInternalParams::default();
            p.entropy_mul_table = Some(build_table(dct16, dct32));
            let lift = LossyConfig::new(d)
                .with_effort(7)
                .with_internal_params(p)
                .encode(rgb_u8, w, h, PixelLayout::Rgb8)
                .unwrap();
            let (dw, dh, dec_b) = decode_linear(&base).unwrap();
            let dec_b_pix: Vec<RGB<f32>> = dec_b
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let b_bfly = butteraugli_linear(
                orig_lin.as_ref(),
                Img::new(dec_b_pix, dw, dh).as_ref(),
                &bparams,
            )
            .map(|r| r.score as f64)
            .unwrap();
            let dec_b_srgb: Vec<[u8; 3]> = dec_b
                .chunks(3)
                .map(|c| {
                    [
                        linear_to_srgb_u8(c[0]),
                        linear_to_srgb_u8(c[1]),
                        linear_to_srgb_u8(c[2]),
                    ]
                })
                .collect();
            let b_ssim2 = fast_ssim2::compute_ssimulacra2(
                orig_srgb.as_ref(),
                Img::new(dec_b_srgb, dw, dh).as_ref(),
            )
            .unwrap();
            let (_dw2, _dh2, dec_l) = decode_linear(&lift).unwrap();
            let dec_l_pix: Vec<RGB<f32>> = dec_l
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let l_bfly = butteraugli_linear(
                orig_lin.as_ref(),
                Img::new(dec_l_pix, dw, dh).as_ref(),
                &bparams,
            )
            .map(|r| r.score as f64)
            .unwrap();
            let dec_l_srgb: Vec<[u8; 3]> = dec_l
                .chunks(3)
                .map(|c| {
                    [
                        linear_to_srgb_u8(c[0]),
                        linear_to_srgb_u8(c[1]),
                        linear_to_srgb_u8(c[2]),
                    ]
                })
                .collect();
            let l_ssim2 = fast_ssim2::compute_ssimulacra2(
                orig_srgb.as_ref(),
                Img::new(dec_l_srgb, dw, dh).as_ref(),
            )
            .unwrap();

            let bd = (lift.len() as f64 - base.len() as f64) / base.len() as f64 * 100.0;
            let bfd = (l_bfly - b_bfly) / b_bfly.max(1e-9) * 100.0;
            let sd = l_ssim2 - b_ssim2;
            writeln!(out, "SCR\t{}\tscreen\t7\t{:.3}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.3}\t{:+.3}\t{:+.4}",
                slabel, d, dct16, dct32, base.len(), lift.len(), b_bfly, l_bfly, b_ssim2, l_ssim2, bd, bfd, sd).unwrap();
            out.flush().unwrap();
            eprintln!(
                "  d={d}: Δbytes={:+.3}% Δbfly={:+.3}% Δssim2={:+.4}  (base bfly={:.4} → lift bfly={:.4})",
                bd, bfd, sd, b_bfly, l_bfly
            );
        }
    }

    drop(out);
    std::fs::rename(&staging, &out_path).expect("atomic mv");
    eprintln!("\nTSV: {}", out_path.display());
}
