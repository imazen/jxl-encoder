//! W44-102 paired A/B (RULED OUT): widening `cfl_two_pass` effort
//! gate from `effort >= 7` to `effort >= 5` (libjxl parity at `kHare`)
//! produces zero meaningful bfly improvement on the W44-101 wedge
//! cells and 2 SSIM2 regressions exceeding the -0.3 acceptance gate.
//! Gate retained at `effort >= 7`. This bench is preserved as the
//! reproducer for the ruling. See `effort.rs:1027-1042` for the
//! in-source RULED-OUT comment.
//!
//! Background: W44-101 audit (`w44_101_source_diff_audit_3_cells_2026-05-19.md`)
//! found `effort.rs:1027` set `cfl_two_pass: effort >= 7`, but libjxl
//! `enc_heuristics.cc:1190` gates Pass-2 at `speed_tier <= kHare`
//! (= effort >= 5). At e5/e6, libjxl runs Pass-2 with `fast=true`
//! (least-squares); at e7+ it uses Newton. Our `cfl_newton: effort >= 7`
//! gate already matches that split — only `cfl_two_pass` was tight.
//!
//! Modes per cell:
//! - `baseline`: pin `cfl_two_pass = Some(false)` (= pre-W44-102 behaviour
//!   at e5/e6; at e7+ this is a regression vs default).
//! - `widened` : pin `cfl_two_pass = Some(true)` (= new default at e5+).
//!
//! Reports per cell: bytes, bfly, ssim2, encode_ms. Bytes-delta /
//! bfly-delta / ssim2-delta computed widened relative to baseline.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features 'parallel butteraugli-loop ssim2-loop __expert' \
//!     --example w44_102_cfl_two_pass_ab
//!
//! TSV: benchmarks/w44_102_cfl_two_pass_e5_2026-05-19.tsv

#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

// ── Cell sets ───────────────────────────────────────────────────────────────

/// Named bfly wedge cells from W44-101 finding + W44-100 ledger.
const WEDGE_CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "cid22/1420710",
        "CID22/CID22-512/validation/1420710.png",
        6,
        5.0,
    ),
    (
        "cid22/1025469",
        "CID22/CID22-512/validation/1025469.png",
        6,
        4.0,
    ),
    ("gb82/codec_wiki", "gb82-sc/codec_wiki.png", 6, 0.2),
    (
        "cid22/1418519",
        "CID22/CID22-512/validation/1418519.png",
        6,
        6.0,
    ),
];

const CID22_PHOTOS: &[(&str, &str)] = &[
    ("cid22/1025469", "CID22/CID22-512/validation/1025469.png"),
    ("cid22/1044329", "CID22/CID22-512/validation/1044329.png"),
    ("cid22/1189261", "CID22/CID22-512/validation/1189261.png"),
    ("cid22/1418519", "CID22/CID22-512/validation/1418519.png"),
    ("cid22/1420710", "CID22/CID22-512/validation/1420710.png"),
    ("cid22/1531677", "CID22/CID22-512/validation/1531677.png"),
    ("cid22/2389166", "CID22/CID22-512/validation/2389166.png"),
    ("cid22/3637739", "CID22/CID22-512/validation/3637739.png"),
];

const SCREENSHOTS: &[(&str, &str)] = &[
    ("gb82/codec_wiki", "gb82-sc/codec_wiki.png"),
    ("gb82/imac_dark", "gb82-sc/imac_dark.png"),
    ("gb82/imac_g3", "gb82-sc/imac_g3.png"),
    ("gb82/terminal", "gb82-sc/terminal.png"),
    ("gb82/windows", "gb82-sc/windows.png"),
];

const PHOTO_EFFORTS: &[u8] = &[5, 6];
const PHOTO_DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

const SCREEN_EFFORTS: &[u8] = &[5, 6];
const SCREEN_DISTANCES: &[f32] = &[0.5, 1.0, 3.0];

const TRIALS: usize = 5;

#[derive(Copy, Clone, Debug)]
enum Mode {
    Baseline, // cfl_two_pass = Some(false)
    Widened,  // cfl_two_pass = Some(true)
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Baseline => "baseline",
            Mode::Widened => "widened",
        }
    }
    fn cfl_two_pass(self) -> Option<bool> {
        match self {
            Mode::Baseline => Some(false),
            Mode::Widened => Some(true),
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

fn encode_with_mode(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    mode: Mode,
) -> (Vec<u8>, f64) {
    let mut params = LossyInternalParams::default();
    params.cfl_two_pass = mode.cfl_two_pass();
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_internal_params(params);
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    let t = Instant::now();
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&lim)
        .encode(rgb_u8)
        .expect("encode failed");
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
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

struct Cell {
    class: &'static str,
    label: String,
    effort: u8,
    distance: f32,
    rgb_u8: Vec<u8>,
    width: u32,
    height: u32,
    orig_lin: Img<Vec<RGB<f32>>>,
    orig_srgb: Img<Vec<[u8; 3]>>,
}

fn load_cell(
    class: &'static str,
    label: &str,
    rel: &str,
    corpus: &PathBuf,
    effort: u8,
    distance: f32,
) -> Option<Cell> {
    let path = corpus.join(rel);
    if !path.exists() {
        eprintln!("MISS {}", path.display());
        return None;
    }
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
    Some(Cell {
        class,
        label: label.to_string(),
        effort,
        distance,
        rgb_u8,
        width: w,
        height: h,
        orig_lin,
        orig_srgb,
    })
}

fn build_cell_list(corpus: &PathBuf) -> Vec<Cell> {
    let mut cells: Vec<Cell> = Vec::new();
    // Named wedge cells first (priority)
    for &(label, rel, effort, distance) in WEDGE_CELLS {
        if let Some(c) = load_cell("WEDGE", label, rel, corpus, effort, distance) {
            cells.push(c);
        }
    }
    // CID22 sweep × e5/e6 × d∈{0.5,1,2,3,4,5,6}
    for &(label, rel) in CID22_PHOTOS {
        for &e in PHOTO_EFFORTS {
            for &d in PHOTO_DISTANCES {
                // Skip if already covered by WEDGE list
                let dup = WEDGE_CELLS
                    .iter()
                    .any(|&(wl, _, we, wd)| wl == label && we == e && (wd - d).abs() < 1e-6);
                if dup {
                    continue;
                }
                if let Some(c) = load_cell("PHOTO", label, rel, corpus, e, d) {
                    cells.push(c);
                }
            }
        }
    }
    // GB82-SC × e5/e6 × d∈{0.5,1,3}
    for &(label, rel) in SCREENSHOTS {
        for &e in SCREEN_EFFORTS {
            for &d in SCREEN_DISTANCES {
                let dup = WEDGE_CELLS
                    .iter()
                    .any(|&(wl, _, we, wd)| wl == label && we == e && (wd - d).abs() < 1e-6);
                if dup {
                    continue;
                }
                if let Some(c) = load_cell("SCREEN", label, rel, corpus, e, d) {
                    cells.push(c);
                }
            }
        }
    }
    cells
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );
    let out_path = PathBuf::from("benchmarks/w44_102_cfl_two_pass_e5_2026-05-19.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/w44_102_cfl_two_pass_ab_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).unwrap();
    writeln!(
        out,
        "class\timage\teffort\tdistance\tbaseline_bytes\twidened_bytes\tbaseline_bfly\twidened_bfly\tbaseline_ssim2\twidened_ssim2\tbaseline_ms_min\twidened_ms_min\tbytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs\tms_delta_pct"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();
    let cells = build_cell_list(&corpus);
    eprintln!("Total cells: {}", cells.len());

    let mut totals_bytes_delta = 0.0_f64;
    let mut totals_bfly_delta = 0.0_f64;
    let mut totals_ssim2_delta = 0.0_f64;
    let mut totals_ms_delta = 0.0_f64;
    let mut wedge_bfly_improvements: Vec<(String, f32, f64, f64)> = Vec::new();
    let mut count = 0;
    let mut fixed_regressions: Vec<(String, u8, f32, f64, f64, f64)> = Vec::new();

    for (i, cell) in cells.iter().enumerate() {
        eprintln!(
            "[{:>3}/{}] {} {} e{} d={:.2}",
            i + 1,
            cells.len(),
            cell.class,
            cell.label,
            cell.effort,
            cell.distance
        );

        // INTERLEAVED A/B: alternate baseline/widened across trials to kill
        // thermal/turbo bias. Each trial encodes both modes back-to-back.
        let mut b_bytes_all: Vec<usize> = Vec::with_capacity(TRIALS);
        let mut w_bytes_all: Vec<usize> = Vec::with_capacity(TRIALS);
        let mut b_ms_all: Vec<f64> = Vec::with_capacity(TRIALS);
        let mut w_ms_all: Vec<f64> = Vec::with_capacity(TRIALS);
        let mut last_b_bytes: Vec<u8> = Vec::new();
        let mut last_w_bytes: Vec<u8> = Vec::new();
        for t in 0..TRIALS {
            let (b1, m1) = encode_with_mode(
                &cell.rgb_u8,
                cell.width,
                cell.height,
                cell.effort,
                cell.distance,
                Mode::Baseline,
            );
            let (w1, n1) = encode_with_mode(
                &cell.rgb_u8,
                cell.width,
                cell.height,
                cell.effort,
                cell.distance,
                Mode::Widened,
            );
            b_bytes_all.push(b1.len());
            w_bytes_all.push(w1.len());
            b_ms_all.push(m1);
            w_ms_all.push(n1);
            if t == TRIALS - 1 {
                last_b_bytes = b1;
                last_w_bytes = w1;
            }
        }
        let b_min_ms = b_ms_all.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_min_ms = w_ms_all.iter().cloned().fold(f64::INFINITY, f64::min);
        let b_min_bytes = *b_bytes_all.iter().min().unwrap();
        let w_min_bytes = *w_bytes_all.iter().min().unwrap();

        // Bytes are deterministic — sanity check
        for &x in &b_bytes_all[1..] {
            debug_assert_eq!(x, b_bytes_all[0]);
        }
        for &x in &w_bytes_all[1..] {
            debug_assert_eq!(x, w_bytes_all[0]);
        }

        // Quality metrics: compute once on the last encoded bytes (deterministic).
        let (_, b_bfly, b_ssim2) =
            compute_metrics(&last_b_bytes, &cell.orig_lin, &cell.orig_srgb, &bparams);
        let (_, w_bfly, w_ssim2) =
            compute_metrics(&last_w_bytes, &cell.orig_lin, &cell.orig_srgb, &bparams);

        let bytes_d = (w_min_bytes as f64 - b_min_bytes as f64) / b_min_bytes as f64 * 100.0;
        let bfly_d = (w_bfly - b_bfly) / b_bfly.max(1e-9) * 100.0;
        let ssim2_d = w_ssim2 - b_ssim2;
        let ms_d = (w_min_ms - b_min_ms) / b_min_ms.max(1e-9) * 100.0;

        writeln!(
            out,
            "{}\t{}\t{}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.3}\t{:+.4}\t{:+.4}\t{:+.4}\t{:+.3}",
            cell.class, cell.label, cell.effort, cell.distance,
            b_min_bytes, w_min_bytes,
            b_bfly, w_bfly,
            b_ssim2, w_ssim2,
            b_min_ms, w_min_ms,
            bytes_d, bfly_d, ssim2_d, ms_d
        ).unwrap();
        out.flush().ok();

        eprintln!(
            "  baseline: {} B  bfly={:.4}  ssim2={:.4}  {:.1}ms  |  widened: {} B  bfly={:.4}  ssim2={:.4}  {:.1}ms  |  Δbytes={:+.2}%  Δbfly={:+.2}%  Δssim2={:+.3}  Δms={:+.1}%",
            b_min_bytes,
            b_bfly,
            b_ssim2,
            b_min_ms,
            w_min_bytes,
            w_bfly,
            w_ssim2,
            w_min_ms,
            bytes_d,
            bfly_d,
            ssim2_d,
            ms_d
        );

        totals_bytes_delta += bytes_d;
        totals_bfly_delta += bfly_d;
        totals_ssim2_delta += ssim2_d;
        totals_ms_delta += ms_d;
        count += 1;

        // Wedge improvements tracking
        if cell.class == "WEDGE" {
            wedge_bfly_improvements.push((cell.label.clone(), cell.distance, b_bfly, w_bfly));
        }
        // Acceptance gates: bytes regression > +3% on any cell, ssim2 regression > 0.3
        if bytes_d > 3.0 || ssim2_d < -0.3 {
            fixed_regressions.push((
                format!("{} e{} d={:.2}", cell.label, cell.effort, cell.distance),
                cell.effort,
                cell.distance,
                bytes_d,
                ssim2_d,
                bfly_d,
            ));
        }
    }

    drop(out);
    std::fs::rename(&staging, &out_path).unwrap();

    eprintln!("\n========================");
    eprintln!("Summary across {} cells:", count);
    eprintln!(
        "  avg bytes delta:  {:+.3}%",
        totals_bytes_delta / count as f64
    );
    eprintln!(
        "  avg bfly delta:   {:+.3}%",
        totals_bfly_delta / count as f64
    );
    eprintln!(
        "  avg ssim2 delta:  {:+.4}",
        totals_ssim2_delta / count as f64
    );
    eprintln!(
        "  avg ms delta:     {:+.3}%",
        totals_ms_delta / count as f64
    );

    eprintln!("\nWedge cell bfly improvements:");
    for (label, d, b_bfly, w_bfly) in &wedge_bfly_improvements {
        let imp_pct = (b_bfly - w_bfly) / b_bfly.max(1e-9) * 100.0;
        eprintln!(
            "  {} d={:.1}: baseline bfly={:.4}, widened bfly={:.4}  ({:+.2}% improvement)",
            label, d, b_bfly, w_bfly, imp_pct
        );
    }

    if !fixed_regressions.is_empty() {
        eprintln!("\nCells exceeding regression gates (bytes>+3% or ssim2<-0.3):");
        for (lab, e, d, bd, sd, fd) in &fixed_regressions {
            eprintln!(
                "  {} (e{} d={:.2}): bytes={:+.2}%, ssim2={:+.3}, bfly={:+.2}%",
                lab, e, d, bd, sd, fd
            );
        }
    } else {
        eprintln!("\nNo cells exceeded regression gates.");
    }

    eprintln!("\nTSV: {}", out_path.display());
}
