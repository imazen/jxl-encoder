//! W44-102 finish-up bench for the 5 windows screenshot cells that
//! were missed when the main bench was prematurely terminated.
//! Appends to the canonical TSV.

#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use jxl_encoder::effort::LossyInternalParams;
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const CELLS: &[(&str, &str, u8, f32)] = &[
    ("gb82/windows", "gb82-sc/windows.png", 5, 1.0),
    ("gb82/windows", "gb82-sc/windows.png", 5, 3.0),
    ("gb82/windows", "gb82-sc/windows.png", 6, 0.5),
    ("gb82/windows", "gb82-sc/windows.png", 6, 1.0),
    ("gb82/windows", "gb82-sc/windows.png", 6, 3.0),
];

const TRIALS: usize = 5;

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

fn encode(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32, two_pass: bool) -> (Vec<u8>, f64) {
    let mut params = LossyInternalParams::default();
    params.cfl_two_pass = Some(two_pass);
    let cfg = LossyConfig::new(d).with_effort(effort).with_internal_params(params);
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    let t = Instant::now();
    let bytes = cfg.encode_request(w, h, PixelLayout::Rgb8).with_limits(&lim).encode(rgb).unwrap();
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn metrics(
    bytes: &[u8],
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    bp: &ButteraugliParams,
) -> (f64, f64) {
    let (dw, dh, dec) = decode_linear(bytes).unwrap();
    let dec_pix: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
    let dec_lin = Img::new(dec_pix, dw, dh);
    let bfly = butteraugli_linear(orig_lin.as_ref(), dec_lin.as_ref(), bp)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);
    let dec_srgb: Vec<[u8; 3]> = dec
        .chunks(3)
        .map(|c| [linear_to_srgb_u8(c[0]), linear_to_srgb_u8(c[1]), linear_to_srgb_u8(c[2])])
        .collect();
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), Img::new(dec_srgb, dw, dh).as_ref())
        .unwrap_or(f64::NAN);
    (bfly, ssim2)
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| "/home/lilith/work/codec-corpus".into()),
    );
    let out_path = PathBuf::from("benchmarks/w44_102_cfl_two_pass_e5_2026-05-19.tsv");
    let mut out = std::fs::OpenOptions::new().append(true).open(&out_path).unwrap();
    let bp = ButteraugliParams::default();

    for &(label, rel, effort, d) in CELLS {
        eprintln!("running {} e{} d={:.2}", label, effort, d);
        let path = corpus.join(rel);
        let img = image::open(&path).unwrap();
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8();
        let rgb_u8: Vec<u8> = rgb.as_raw().clone();
        let linear: Vec<RGB<f32>> = rgb
            .pixels()
            .map(|p| RGB::new(srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2])))
            .collect();
        let orig_lin = Img::new(linear, w as usize, h as usize);
        let srgb_arr: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let orig_srgb = Img::new(srgb_arr, w as usize, h as usize);

        let mut b_bytes_min = usize::MAX;
        let mut w_bytes_min = usize::MAX;
        let mut b_ms_min = f64::INFINITY;
        let mut w_ms_min = f64::INFINITY;
        let mut last_b = Vec::new();
        let mut last_w = Vec::new();
        for t in 0..TRIALS {
            let (b1, m1) = encode(&rgb_u8, w, h, effort, d, false);
            let (w1, n1) = encode(&rgb_u8, w, h, effort, d, true);
            b_bytes_min = b_bytes_min.min(b1.len());
            w_bytes_min = w_bytes_min.min(w1.len());
            b_ms_min = b_ms_min.min(m1);
            w_ms_min = w_ms_min.min(n1);
            if t == TRIALS - 1 {
                last_b = b1;
                last_w = w1;
            }
        }
        let (b_bfly, b_ssim2) = metrics(&last_b, &orig_lin, &orig_srgb, &bp);
        let (w_bfly, w_ssim2) = metrics(&last_w, &orig_lin, &orig_srgb, &bp);
        let bd = (w_bytes_min as f64 - b_bytes_min as f64) / b_bytes_min as f64 * 100.0;
        let fd = (w_bfly - b_bfly) / b_bfly.max(1e-9) * 100.0;
        let sd = w_ssim2 - b_ssim2;
        let md = (w_ms_min - b_ms_min) / b_ms_min.max(1e-9) * 100.0;
        writeln!(
            out,
            "SCREEN\t{}\t{}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.3}\t{:+.4}\t{:+.4}\t{:+.4}\t{:+.3}",
            label, effort, d, b_bytes_min, w_bytes_min, b_bfly, w_bfly, b_ssim2, w_ssim2,
            b_ms_min, w_ms_min, bd, fd, sd, md
        ).unwrap();
        out.flush().ok();
        eprintln!(
            "  baseline {}B bfly={:.4} ssim2={:.4} {:.0}ms | widened {}B bfly={:.4} ssim2={:.4} {:.0}ms | Δb={:+.2}% Δbfly={:+.2}% Δssim2={:+.3} Δms={:+.1}%",
            b_bytes_min, b_bfly, b_ssim2, b_ms_min, w_bytes_min, w_bfly, w_ssim2, w_ms_min,
            bd, fd, sd, md
        );
    }
    eprintln!("done. appended 5 rows to {}", out_path.display());
}
