// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #61 A/B (post-W44-73 retry of W44-84 honest-stop): does widening
//! `compute_block_ctx_map`'s small-image gate from `tot < 1024 * distance`
//! (libjxl parity) to a distance-independent threshold close more F-D
//! cells now that the ANS+LZ77 context-map writer (W44-73) has landed?
//!
//! Background: W44-81 audit #2a proposed widening this gate to fire on
//! F-D 512×512 photo cells (1420710 / 1531677 / 1189261 at d ≥ 5,
//! tot=4096 which falls below 1024 * d=5 = 5120). W44-84 measured the
//! widening and honest-stopped: only -71.8B mean win on 4 F-D OPEN
//! cells (target -100B), 0 OPEN→FIXED flips. The binding constraint
//! cited was the legacy Huffman+MTF context-map writer
//! (`write_context_map_nonsimple`).
//!
//! Since W44-84, W44-73 landed and `write_context_map_nonsimple` now
//! compares Huffman+MTF vs ANS+LZ77 and picks the cheaper. This AB
//! retries W44-84's bisect with that writer in place.
//!
//! Modes (selected via `JXL_ISSUE_61_WIDEN_THRESHOLD`):
//! - `baseline`   (unset)  → libjxl parity: `tot < 1024 * distance`
//! - `widenA`     ("A")    → `tot < 1024` (distance-independent)
//! - `widenB`     ("B")    → `tot < 512 * distance` (half-scale)
//!
//! All three modes share identical encoder config and source pixels;
//! only the gate threshold differs.
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release \
//!     --features 'parallel butteraugli-loop ssim2-loop __expert' \
//!     --example issue_61_block_ctx_map_widen_ab
//!
//! TSV: benchmarks/issue_61_block_ctx_map_widen_2026-05-25.tsv

#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::PathBuf;

// ── Cells ───────────────────────────────────────────────────────────────────
// Per W44-84 memo: F-D OPEN cluster at e7 d ∈ {1, 2, 3, 4, 5, 6}.
// Adaptive path fires at d ≤ 4 (tot=4096 ≥ 1024*d for d ≤ 4); falls back to
// default for d ≥ 5. Widening exercises the d ≥ 5 cells.
//
// + Spot screen+hash-lock-style probe cells to verify no flips elsewhere.

#[derive(Clone)]
struct CellSpec {
    class: &'static str,
    label: &'static str,
    effort: u8,
    distance: f32,
}

const CELLS: &[CellSpec] = &[
    // F-D photo cluster (the target — predicted wins)
    CellSpec {
        class: "FD_OPEN",
        label: "1420710.png",
        effort: 7,
        distance: 5.0,
    },
    CellSpec {
        class: "FD_OPEN",
        label: "1420710.png",
        effort: 7,
        distance: 6.0,
    },
    CellSpec {
        class: "FD_OPEN",
        label: "1531677.png",
        effort: 7,
        distance: 5.0,
    },
    CellSpec {
        class: "FD_OPEN",
        label: "1531677.png",
        effort: 7,
        distance: 6.0,
    },
    CellSpec {
        class: "FD_OPEN",
        label: "1189261.png",
        effort: 7,
        distance: 4.0,
    },
    // Photo controls (avoid FIXED→OPEN flips)
    CellSpec {
        class: "PHOTO_CTRL",
        label: "1189261.png",
        effort: 7,
        distance: 5.0,
    },
    CellSpec {
        class: "PHOTO_CTRL",
        label: "1189261.png",
        effort: 7,
        distance: 6.0,
    },
    CellSpec {
        class: "PHOTO_CTRL",
        label: "1025469.png",
        effort: 7,
        distance: 5.0,
    },
    CellSpec {
        class: "PHOTO_CTRL",
        label: "1025469.png",
        effort: 7,
        distance: 6.0,
    },
    CellSpec {
        class: "PHOTO_CTRL",
        label: "1418519.png",
        effort: 7,
        distance: 5.0,
    },
    CellSpec {
        class: "PHOTO_CTRL",
        label: "1418519.png",
        effort: 7,
        distance: 6.0,
    },
    // Lower-distance photo (gate fires for d ≥ 1 under widening, vs d ≥ 4 in baseline)
    CellSpec {
        class: "PHOTO_LOWD",
        label: "1420710.png",
        effort: 7,
        distance: 1.0,
    },
    CellSpec {
        class: "PHOTO_LOWD",
        label: "1420710.png",
        effort: 7,
        distance: 2.0,
    },
    CellSpec {
        class: "PHOTO_LOWD",
        label: "1420710.png",
        effort: 7,
        distance: 3.0,
    },
    // Screen control (mask_p25 high; W44-29 doesn't fire)
    CellSpec {
        class: "SCREEN",
        label: "codec_wiki.png",
        effort: 7,
        distance: 4.0,
    },
    CellSpec {
        class: "SCREEN",
        label: "terminal.png",
        effort: 7,
        distance: 4.0,
    },
];

const CORPUS_BASE: &str = "/home/lilith/work/codec-corpus";

#[derive(Copy, Clone, Debug)]
enum Mode {
    Baseline, // libjxl parity (env unset)
    WidenA,   // tot < 1024
    WidenB,   // tot < 512 * distance
}

impl Mode {
    #[allow(dead_code)]
    fn label(self) -> &'static str {
        match self {
            Mode::Baseline => "baseline",
            Mode::WidenA => "widenA",
            Mode::WidenB => "widenB",
        }
    }
    fn apply(self) {
        unsafe {
            std::env::remove_var("JXL_ISSUE_61_WIDEN_THRESHOLD");
            match self {
                Mode::Baseline => {}
                Mode::WidenA => std::env::set_var("JXL_ISSUE_61_WIDEN_THRESHOLD", "A"),
                Mode::WidenB => std::env::set_var("JXL_ISSUE_61_WIDEN_THRESHOLD", "B"),
            }
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

fn encode(pixels: &[u8], w: u32, h: u32, d: f32, e: u8, mode: Mode) -> Vec<u8> {
    mode.apply();
    let cfg = LossyConfig::new(d).with_effort(e);
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&lim)
        .encode(pixels)
        .expect("encode failed")
}

fn compute_metrics(
    bytes: &[u8],
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    bparams: &ButteraugliParams,
) -> (usize, f64, f64) {
    let (dw, dh, dec) = decode_linear(bytes).expect("decode failed");
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

fn try_resolve(corpus: &PathBuf, name: &str) -> Option<PathBuf> {
    for sub in [
        "CID22/CID22-512/validation",
        "CID22/CID22-512/training",
        "gb82-sc",
    ] {
        let p = corpus.join(sub).join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

struct LoadedImage {
    width: u32,
    height: u32,
    rgb_u8: Vec<u8>,
    orig_lin: Img<Vec<RGB<f32>>>,
    orig_srgb: Img<Vec<[u8; 3]>>,
}

fn load_image(corpus: &PathBuf, label: &str) -> Option<LoadedImage> {
    let path = try_resolve(corpus, label)?;
    let img = image::open(&path).ok()?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let rgb_u8: Vec<u8> = rgb.as_raw().clone();
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
    Some(LoadedImage {
        width: w,
        height: h,
        rgb_u8,
        orig_lin,
        orig_srgb,
    })
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| String::from(CORPUS_BASE)),
    );
    let out_path = PathBuf::from("benchmarks/issue_61_block_ctx_map_widen_2026-05-25.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/issue_61_block_ctx_map_widen_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).unwrap();
    writeln!(
        out,
        "class\timage\teffort\tdistance\tbaseline_bytes\twidenA_bytes\twidenB_bytes\tbaseline_bfly\twidenA_bfly\twidenB_bfly\tbaseline_ssim2\twidenA_ssim2\twidenB_ssim2\tA_delta_pct\tB_delta_pct\tA_ssim2_delta\tB_ssim2_delta"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();
    eprintln!("Total cells: {}", CELLS.len());

    // Per-class accumulators
    let mut fd_a_bytes_deltas: Vec<f64> = Vec::new();
    let mut fd_b_bytes_deltas: Vec<f64> = Vec::new();
    let mut photo_ctrl_a_bytes_deltas: Vec<f64> = Vec::new();
    let mut photo_lowd_a_bytes_deltas: Vec<f64> = Vec::new();
    let mut screen_a_bytes_deltas: Vec<f64> = Vec::new();
    let mut worst_ssim2_regression: f64 = 0.0;
    let mut worst_bytes_regression_pct: f64 = 0.0;

    let mut last_image: Option<(String, LoadedImage)> = None;

    for (i, c) in CELLS.iter().enumerate() {
        eprintln!(
            "[{:>3}/{}] {} {} e{} d={:.2}",
            i + 1,
            CELLS.len(),
            c.class,
            c.label,
            c.effort,
            c.distance
        );

        // Cache-load image once per (label) — many cells per same image
        if last_image
            .as_ref()
            .map(|(l, _)| l.as_str() != c.label)
            .unwrap_or(true)
        {
            last_image = load_image(&corpus, c.label).map(|li| (c.label.to_string(), li));
        }
        let li = match &last_image {
            Some((_, li)) => li,
            None => {
                eprintln!("  -- IMAGE MISSING: {}", c.label);
                continue;
            }
        };

        let b_bytes = encode(
            &li.rgb_u8,
            li.width,
            li.height,
            c.distance,
            c.effort,
            Mode::Baseline,
        );
        let a_bytes = encode(
            &li.rgb_u8,
            li.width,
            li.height,
            c.distance,
            c.effort,
            Mode::WidenA,
        );
        let bb_bytes = encode(
            &li.rgb_u8,
            li.width,
            li.height,
            c.distance,
            c.effort,
            Mode::WidenB,
        );

        let (b_len, b_bfly, b_ssim2) =
            compute_metrics(&b_bytes, &li.orig_lin, &li.orig_srgb, &bparams);
        let (a_len, a_bfly, a_ssim2) =
            compute_metrics(&a_bytes, &li.orig_lin, &li.orig_srgb, &bparams);
        let (bb_len, bb_bfly, bb_ssim2) =
            compute_metrics(&bb_bytes, &li.orig_lin, &li.orig_srgb, &bparams);

        let a_delta = (a_len as f64 - b_len as f64) / b_len as f64 * 100.0;
        let bb_delta = (bb_len as f64 - b_len as f64) / b_len as f64 * 100.0;
        let a_ssim2_delta = a_ssim2 - b_ssim2;
        let bb_ssim2_delta = bb_ssim2 - b_ssim2;

        writeln!(
            out,
            "{}\t{}\te{}\t{:.2}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.3}\t{:.4}\t{:.4}",
            c.class, c.label, c.effort, c.distance,
            b_len, a_len, bb_len,
            b_bfly, a_bfly, bb_bfly,
            b_ssim2, a_ssim2, bb_ssim2,
            a_delta, bb_delta, a_ssim2_delta, bb_ssim2_delta
        )
        .unwrap();
        out.flush().ok();

        match c.class {
            "FD_OPEN" => {
                fd_a_bytes_deltas.push(a_delta);
                fd_b_bytes_deltas.push(bb_delta);
            }
            "PHOTO_CTRL" => photo_ctrl_a_bytes_deltas.push(a_delta),
            "PHOTO_LOWD" => photo_lowd_a_bytes_deltas.push(a_delta),
            "SCREEN" => screen_a_bytes_deltas.push(a_delta),
            _ => {}
        }
        if a_ssim2_delta < worst_ssim2_regression {
            worst_ssim2_regression = a_ssim2_delta;
        }
        if a_delta > worst_bytes_regression_pct {
            worst_bytes_regression_pct = a_delta;
        }
    }
    drop(out);
    std::fs::rename(&staging, &out_path).unwrap();
    eprintln!("Wrote {}", out_path.display());

    // Print summary
    fn mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
    eprintln!("\n=== Issue #61 Widen-A Summary (vs libjxl-parity baseline) ===");
    eprintln!(
        "FD_OPEN n={}: mean A bytes delta = {:.3}%, mean B bytes delta = {:.3}%",
        fd_a_bytes_deltas.len(),
        mean(&fd_a_bytes_deltas),
        mean(&fd_b_bytes_deltas)
    );
    eprintln!(
        "PHOTO_CTRL n={}: mean A bytes delta = {:.3}%",
        photo_ctrl_a_bytes_deltas.len(),
        mean(&photo_ctrl_a_bytes_deltas)
    );
    eprintln!(
        "PHOTO_LOWD n={}: mean A bytes delta = {:.3}%",
        photo_lowd_a_bytes_deltas.len(),
        mean(&photo_lowd_a_bytes_deltas)
    );
    eprintln!(
        "SCREEN n={}: mean A bytes delta = {:.3}%",
        screen_a_bytes_deltas.len(),
        mean(&screen_a_bytes_deltas)
    );
    eprintln!(
        "WORST A SSIM2 regression: {:.4}    WORST A bytes regression: +{:.3}%",
        worst_ssim2_regression, worst_bytes_regression_pct
    );
}
