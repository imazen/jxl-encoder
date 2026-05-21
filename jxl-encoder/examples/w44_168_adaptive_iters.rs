// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-168 (Smart-Zenjxl chunk 5) paired A/B/C/D reproducer.
//!
//! Measures whether content-aware `butteraugli_iters` (SmoothSkip /
//! TexturedExtend / Combined) saves wall time on smooth content at
//! e>=8 AND extends e7 quality on textured content without breaking
//! the protections cluster.
//!
//! Modes (gated via `JXL_W44_168_MODE`):
//! - A (Baseline): env unset (or =A). Byte-identical to pre-W44-168
//!   fixed-per-effort schedule. Bench reference.
//! - B (SmoothSkip): JXL_W44_168_MODE=B. At e>=8 on smooth/screenshot
//!   content (`mask_p25 >= 85` OR `mask1x1_median > 95`), iters - 1
//!   saturating at 1. Saves ~30% wall time at e8 (was iters=2 → now 1).
//! - C (TexturedExtend): JXL_W44_168_MODE=C. At e==7 on textured
//!   content (`edge_density >= 0.5`), iters 0 → 2. Bridges textured
//!   e7 toward e8 quality.
//! - D (Combined): JXL_W44_168_MODE=D. Both B and C.
//!
//! Acceptance gates:
//! - (a) Build PASS
//! - (b) `cargo test --lib`: PASS
//! - (c) Hash-locks 36/36 BYTE-IDENTICAL with default Mode A
//! - (d) TARGET_SMOOTH_PHOTO: wall time mean reduction >=5% AND SSIM2
//!       mean change within ±0.10
//! - (e) TARGET_TEXTURED: SSIM2 mean improvement >= +0.1 (Mode C/D)
//!       OR byte-identical (Mode B)
//! - (f) PROTECT_W164: BYTE-IDENTICAL (Mode B only fires at e>=8)
//! - (g) PROTECT_W166: BYTE-IDENTICAL OR SSIM2 within ±0.10
//! - (h) CONTROL: BYTE-IDENTICAL
//! - (i) EncoderStrategy::Libjxl: BYTE-IDENTICAL
//! - (j) Multi-decoder PASS
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release -p jxl-encoder \
//!     --features 'butteraugli-loop ssim2-loop parallel __expert' \
//!     --example w44_168_adaptive_iters \
//!     > benchmarks/w44_168_adaptive_iters_2026-05-21.tsv

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// Cells:
/// - TARGET_SMOOTH_PHOTO: 1418519 + 7552578 × e{7,8,9} × d{3,4,5} = 18 cells
///   (smooth, mask_p25 high → Mode B SmoothSkip should fire at e>=8;
///   Mode C TexturedExtend should NOT fire because edge_density < 0.5)
/// - TARGET_TEXTURED: 1189261 + 1420710 × e{7} × d{1,1.5,2} = 6 cells
///   (textured, edge_density >= 0.5 → Mode C TexturedExtend should
///   fire at e7 only)
/// - PROTECT_W164_screenshots: codec_wiki, imac_g3, terminal × e{5,6}
///   = 6 cells (auto-classify Screenshot — Mode B only fires at e>=8;
///   must be BYTE-IDENTICAL)
/// - PROTECT_W166_1418519: 1418519 × e{8,9} × d{5,6} = 4 cells
///   (W44-166 variant Z fire — Mode B fires here; PROTECT means SSIM2
///   within ±0.10)
/// - CONTROL: 1025469 × e{7,8} × d{4} = 2 cells
const CELLS: &[(&str, &str, u8, f32)] = &[
    // TARGET_SMOOTH_PHOTO (mask_p25 high; smooth → Mode B fires at e8+)
    ("cid22", "1418519.png", 7, 3.0),
    ("cid22", "1418519.png", 7, 4.0),
    ("cid22", "1418519.png", 7, 5.0),
    ("cid22", "1418519.png", 8, 3.0),
    ("cid22", "1418519.png", 8, 4.0),
    ("cid22", "1418519.png", 8, 5.0),
    ("cid22", "1418519.png", 9, 3.0),
    ("cid22", "1418519.png", 9, 4.0),
    ("cid22", "1418519.png", 9, 5.0),
    ("cid22", "7552578.png", 7, 3.0),
    ("cid22", "7552578.png", 7, 4.0),
    ("cid22", "7552578.png", 7, 5.0),
    ("cid22", "7552578.png", 8, 3.0),
    ("cid22", "7552578.png", 8, 4.0),
    ("cid22", "7552578.png", 8, 5.0),
    ("cid22", "7552578.png", 9, 3.0),
    ("cid22", "7552578.png", 9, 4.0),
    ("cid22", "7552578.png", 9, 5.0),
    // TARGET_TEXTURED (edge_density high; textured → Mode C fires at e7)
    ("cid22", "1189261.png", 7, 1.0),
    ("cid22", "1189261.png", 7, 1.5),
    ("cid22", "1189261.png", 7, 2.0),
    ("cid22", "1420710.png", 7, 1.0),
    ("cid22", "1420710.png", 7, 1.5),
    ("cid22", "1420710.png", 7, 2.0),
    // PROTECT_W164_screenshots (auto-classify Screenshot; Mode B at
    // e>=8 fires — but these are at e5/e6 so the gate doesn't apply)
    ("gb82sc", "codec_wiki.png", 5, 1.0),
    ("gb82sc", "codec_wiki.png", 6, 1.0),
    ("gb82sc", "imac_g3.png", 5, 1.0),
    ("gb82sc", "imac_g3.png", 6, 1.0),
    ("gb82sc", "terminal.png", 5, 1.0),
    ("gb82sc", "terminal.png", 6, 1.0),
    // PROTECT_W166_1418519 (W44-166 variant Z; Mode B fires here)
    ("cid22", "1418519.png", 8, 5.0), // dup with TARGET; keep for protection accounting
    ("cid22", "1418519.png", 8, 6.0),
    ("cid22", "1418519.png", 9, 5.0), // dup; same as above
    ("cid22", "1418519.png", 9, 6.0),
    // CONTROL (no gate fires — Mode B at e7 baseline=0 already)
    ("cid22", "1025469.png", 7, 4.0),
    ("cid22", "1025469.png", 8, 4.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    A,
    B,
    C,
    D,
}

impl Mode {
    fn env(self) -> &'static str {
        match self {
            Mode::A => "A",
            Mode::B => "B",
            Mode::C => "C",
            Mode::D => "D",
        }
    }
}

fn encode_with_mode(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
) -> Result<(Vec<u8>, f64), String> {
    let prev = std::env::var("JXL_W44_168_MODE").ok();
    // SAFETY: single-threaded bench, paired interleaved.
    unsafe { std::env::set_var("JXL_W44_168_MODE", mode.env()) };
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .with_strategy(EncoderStrategy::Zenjxl);
    let start = Instant::now();
    let result = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    match prev {
        Some(v) => unsafe { std::env::set_var("JXL_W44_168_MODE", v) },
        None => unsafe { std::env::remove_var("JXL_W44_168_MODE") },
    }
    result.map(|b| (b, elapsed_ms))
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
    encode_ms: f64,
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
    let (bitstream, encode_ms) = match encode_with_mode(rgb, w, h, effort, d, mode) {
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
        encode_ms,
    })
}

fn classify(corpus: &str, image: &str, effort: u8, d: f32) -> &'static str {
    // Note: 1418519 e8/e9 d=5 appears in BOTH TARGET_SMOOTH_PHOTO (full
    // sweep) AND PROTECT_W166 (specific cluster). The first occurrence
    // (TARGET_SMOOTH_PHOTO) wins by iteration order; the PROTECT_W166
    // duplicate doesn't change accounting because Mode B affects both.
    if corpus == "gb82sc" {
        "PROTECT_W164_SCREENSHOT"
    } else {
        match image {
            "1418519.png" => {
                if (effort == 8 || effort == 9) && (d - 6.0).abs() < 0.01 {
                    "PROTECT_W166_1418519_d6"
                } else if (effort == 8 || effort == 9) && (d - 5.0).abs() < 0.01 {
                    "TARGET_SMOOTH_PHOTO_or_PROTECT_W166"
                } else {
                    "TARGET_SMOOTH_PHOTO"
                }
            }
            "7552578.png" => "TARGET_SMOOTH_PHOTO",
            "1189261.png" => "TARGET_TEXTURED",
            "1420710.png" => "TARGET_TEXTURED",
            "1025469.png" => "CONTROL",
            _ => "OTHER",
        }
    }
}

fn corpus_path(corpus: &str, image: &str) -> PathBuf {
    match corpus {
        "cid22" => PathBuf::from(CID22).join(image),
        "gb82sc" => PathBuf::from(GB82_SC).join(image),
        _ => PathBuf::from(image),
    }
}

fn main() {
    eprintln!("W44-168 A/B/C/D: A=baseline / B=SmoothSkip / C=TexturedExtend / D=Combined");
    eprintln!("Cells (interleaved A,B,C,D): {}", CELLS.len());

    println!(
        "corpus\timage\teffort\tdistance\t\
         A_bytes\tB_bytes\tC_bytes\tD_bytes\t\
         BA_pct\tCA_pct\tDA_pct\t\
         A_bfly\tB_bfly\tC_bfly\tD_bfly\t\
         A_ssim2\tB_ssim2\tC_ssim2\tD_ssim2\t\
         BA_ssim2\tCA_ssim2\tDA_ssim2\t\
         A_encode_ms\tB_encode_ms\tC_encode_ms\tD_encode_ms\t\
         BA_ms_pct\tCA_ms_pct\tDA_ms_pct\tclass"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    // Aggregates: keyed by (class, mode_diff)
    // value: (sum_ssim2_diff, sum_bytes_pct, sum_ms_pct, n)
    let mut agg: BTreeMap<(&'static str, &'static str), (f64, f64, f64, usize)> = BTreeMap::new();
    let mut byte_identical: BTreeMap<(&'static str, &'static str), (usize, usize)> =
        BTreeMap::new();

    for (i, &(corpus, image, effort, d)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {}:{} e{} d={}",
            i + 1,
            n_cells,
            corpus,
            image,
            effort,
            d
        );

        let path = corpus_path(corpus, image);
        let cache_key = format!("{}:{}", corpus, image);
        let (w, h, raw, orig_linear_img, orig_srgb_img) =
            images_cache.entry(cache_key.clone()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let linear = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                (w, h, rgb, linear, srgb_img)
            });

        let sa = score_cell(
            raw,
            *w,
            *h,
            effort,
            d,
            Mode::A,
            orig_linear_img,
            orig_srgb_img,
        );
        let sb = score_cell(
            raw,
            *w,
            *h,
            effort,
            d,
            Mode::B,
            orig_linear_img,
            orig_srgb_img,
        );
        let sc = score_cell(
            raw,
            *w,
            *h,
            effort,
            d,
            Mode::C,
            orig_linear_img,
            orig_srgb_img,
        );
        let sd = score_cell(
            raw,
            *w,
            *h,
            effort,
            d,
            Mode::D,
            orig_linear_img,
            orig_srgb_img,
        );

        let class = classify(corpus, image, effort, d);

        if let (Some(a), Some(b), Some(c), Some(dscore)) = (sa, sb, sc, sd) {
            let ba_pct = 100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let ca_pct = 100.0 * (c.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let da_pct = 100.0 * (dscore.bytes as f64 - a.bytes as f64) / a.bytes as f64;
            let ba_ss2 = b.ssim2 - a.ssim2;
            let ca_ss2 = c.ssim2 - a.ssim2;
            let da_ss2 = dscore.ssim2 - a.ssim2;
            let ba_ms_pct = 100.0 * (b.encode_ms - a.encode_ms) / a.encode_ms;
            let ca_ms_pct = 100.0 * (c.encode_ms - a.encode_ms) / a.encode_ms;
            let da_ms_pct = 100.0 * (dscore.encode_ms - a.encode_ms) / a.encode_ms;
            println!(
                "{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t\
                 {:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t\
                 {:+.4}\t{:+.4}\t{:+.4}\t\
                 {:.2}\t{:.2}\t{:.2}\t{:.2}\t{:+.2}\t{:+.2}\t{:+.2}\t{}",
                corpus,
                image,
                effort,
                d,
                a.bytes,
                b.bytes,
                c.bytes,
                dscore.bytes,
                ba_pct,
                ca_pct,
                da_pct,
                a.butteraugli,
                b.butteraugli,
                c.butteraugli,
                dscore.butteraugli,
                a.ssim2,
                b.ssim2,
                c.ssim2,
                dscore.ssim2,
                ba_ss2,
                ca_ss2,
                da_ss2,
                a.encode_ms,
                b.encode_ms,
                c.encode_ms,
                dscore.encode_ms,
                ba_ms_pct,
                ca_ms_pct,
                da_ms_pct,
                class
            );
            // Aggregate by (class, mode_label)
            for (label, sdelta, byte_pct, ms_pct, score) in [
                ("B", ba_ss2, ba_pct, ba_ms_pct, &b),
                ("C", ca_ss2, ca_pct, ca_ms_pct, &c),
                ("D", da_ss2, da_pct, da_ms_pct, &dscore),
            ] {
                let key = (class, label);
                let entry = agg.entry(key).or_insert((0.0, 0.0, 0.0, 0));
                entry.0 += sdelta;
                entry.1 += byte_pct;
                entry.2 += ms_pct;
                entry.3 += 1;
                let bi_entry = byte_identical.entry((class, label)).or_insert((0, 0));
                bi_entry.1 += 1;
                if score.bytes == a.bytes {
                    bi_entry.0 += 1;
                }
            }
        }
    }

    eprintln!("\n=== W44-168 aggregates ===");
    eprintln!("by (class, mode) — mean across cells:");
    for ((class, mode_lbl), (sum_ss2, sum_pct, sum_ms_pct, n)) in &agg {
        eprintln!(
            "  {:38} mode={} n={} mean_ΔSSIM2={:+.4} mean_Δbytes%={:+.3} mean_Δms%={:+.2}",
            class,
            mode_lbl,
            n,
            sum_ss2 / *n as f64,
            sum_pct / *n as f64,
            sum_ms_pct / *n as f64,
        );
    }
    eprintln!("\nbyte-identical counts:");
    for ((class, mode_lbl), (bi, total)) in &byte_identical {
        eprintln!(
            "  {:38} mode={} byte-identical={}/{}",
            class, mode_lbl, bi, total
        );
    }
}
