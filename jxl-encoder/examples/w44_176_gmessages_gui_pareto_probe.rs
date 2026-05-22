// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-176 pareto-direction probe for gmessages / gui (extra screens in the
//! W44-108 firing class that the proxy bisect predicate `chroma_blk_ratio < 0.18
//! AND chroma < 0.16` would catch alongside terminal).
//!
//! Goal: determine whether the W44-109 lift IS NET-NEGATIVE on these images
//! (terminal-like) or NET-POSITIVE (graph/imac_g3/imac_dark-like) BEFORE we
//! ship a discriminator that might wrongly exclude them.
//!
//! Method: paired A=baseline (W44-109 lift ON) vs B=`JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0`
//! (lift OFF) at e7 d∈{4,5}. Same env-var lever the W44-174 diagnosis used.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release -p jxl-encoder \
//!     --features 'butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_176_gmessages_gui_pareto_probe

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::PathBuf;

const GB82_SC: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// Cells: image, effort, distance. Probe at the W44-108 firing band (e7 d=4,5,6).
/// Include terminal as a known REGRESSION reference, and graph/imac_g3 as known WIN refs.
const CELLS: &[(&str, u8, f32)] = &[
    // SUSPECTED-REGRESSION (caught by `cblk<0.18 AND chroma<0.16` predicate)
    ("gmessages.png", 7, 4.0),
    ("gmessages.png", 7, 5.0),
    ("gui.png", 7, 4.0),
    ("gui.png", 7, 5.0),
    // REFERENCE-REGRESSION (caught by predicate AND known net-negative per W44-174)
    ("terminal.png", 7, 4.0),
    ("terminal.png", 7, 5.0),
    // REFERENCE-WIN (NOT caught by predicate, known net-positive per W44-174)
    ("graph.png", 7, 4.0),
    ("graph.png", 7, 5.0),
    ("imac_g3.png", 7, 4.0),
    ("imac_g3.png", 7, 5.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Baseline — env unset = W44-109 lift fires.
    A,
    /// B — env=1.0 disables W44-109 lift.
    B,
}

fn encode_with_mode(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
) -> Result<Vec<u8>, String> {
    let prev = std::env::var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE").ok();
    // SAFETY: single-threaded paired bench.
    unsafe {
        match mode {
            Mode::A => std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE"),
            Mode::B => std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "1.0"),
        }
    };
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_strategy(EncoderStrategy::Zenjxl);
    let result = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"));
    unsafe {
        match prev {
            Some(v) => std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", v),
            None => std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE"),
        }
    }
    result
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

#[derive(Clone, Copy, Default, Debug)]
struct Score {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
}

fn score_cell(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
) -> Option<Score> {
    let bitstream = match encode_with_mode(rgb, w, h, effort, d, mode) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  encode failed ({:?}): {}", mode, e);
            return None;
        }
    };
    let bytes = bitstream.len();
    let (dw, dh, decoded_linear) = decode_jxl_linear(&bitstream)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let params = ButteraugliParams::default();
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), &params)
        .map(|r| r.score)
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
    Some(Score {
        bytes,
        butteraugli: bfly,
        ssim2,
    })
}

fn main() {
    eprintln!("W44-176 pareto-direction probe: A=baseline (W44-109 lift on) vs B=lift off");
    eprintln!(
        "Question: are gmessages/gui NET-NEGATIVE (like terminal) or NET-POSITIVE (like graph)?"
    );
    eprintln!("Cells (interleaved A,B): {}", CELLS.len());

    println!(
        "image\teffort\tdistance\tA_bytes\tB_bytes\tBA_pct\tA_bfly\tB_bfly\tBA_bfly\tA_ssim2\tB_ssim2\tBA_ssim2"
    );

    use std::collections::BTreeMap;
    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    for (i, &(image, effort, d)) in CELLS.iter().enumerate() {
        eprintln!("[{}/{}] {} e{} d={}", i + 1, CELLS.len(), image, effort, d);
        let path = PathBuf::from(GB82_SC).join(image);
        let cache_key = image.to_string();
        let (w, h, raw, orig_linear, orig_srgb) =
            images_cache.entry(cache_key.clone()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let linear = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                (w, h, rgb, linear, srgb_img)
            });

        let sa = score_cell(raw, *w, *h, effort, d, Mode::A, orig_linear, orig_srgb);
        let sb = score_cell(raw, *w, *h, effort, d, Mode::B, orig_linear, orig_srgb);

        if let (Some(a), Some(b)) = (sa, sb) {
            let ba_pct = if a.bytes > 0 {
                100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64
            } else {
                0.0
            };
            let ba_bfly = b.butteraugli - a.butteraugli;
            let ba_s2 = b.ssim2 - a.ssim2;
            println!(
                "{}\te{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:+.4}\t{:.4}\t{:.4}\t{:+.4}",
                image,
                effort,
                d,
                a.bytes,
                b.bytes,
                ba_pct,
                a.butteraugli,
                b.butteraugli,
                ba_bfly,
                a.ssim2,
                b.ssim2,
                ba_s2,
            );
        }
    }

    eprintln!();
    eprintln!("=== Interpretation ===");
    eprintln!("BA_pct < 0 → disabling W44-109 lift reduces bytes (lift was over-allocating)");
    eprintln!("BA_ssim2 > 0 → disabling W44-109 lift IMPROVES SSIM2 (lift was hurting quality)");
    eprintln!("BA_ssim2 < 0 → disabling W44-109 lift HURTS SSIM2 (lift was buying real quality)");
    eprintln!();
    eprintln!(
        "Net-NEGATIVE pareto (terminal-class, want EXCLUDE): BA_pct < 0 AND BA_ssim2 >= -0.10"
    );
    eprintln!(
        "Net-POSITIVE pareto (graph-class, want KEEP): BA_ssim2 < -1.0 (lift buys real SSIM2)"
    );
}
