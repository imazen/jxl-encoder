// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Zenjxl regression gate — locks W44-202 baseline numbers as a CI invariant.
//!
//! ## Why this exists
//!
//! After the W44-86 → W44-202 cjxl-parity arc closed at 0 OPEN cells on the
//! 595-cell parity ledger, Zenjxl's bytes/quality/wall-time profile is the
//! highest-EV thing in the encoder to keep nailed down. Every tuning chunk
//! that touches a Section A / B / C / D divergence row can drift Zenjxl
//! without changing Libjxl byte-locks (which already gate the "we still
//! mirror cjxl exactly" axis). This test guards the *other* axis: "Zenjxl
//! still wins over cjxl by the W44-202 margin."
//!
//! Failure mode on drift: the test PRINTS per-cell diffs (current vs locked
//! baseline) and the aggregate movement, so the failure message tells you
//! exactly which cells regressed and by how much. No mystery hashes.
//!
//! ## Design (Option A: per-cell + aggregate)
//!
//! 12 representative cells from the 1978-cell W44-202 baseline. Picked for:
//! - **Speed**: ~10s total wall time on the 7950X reference. Heavy screen
//!   cells (codec_wiki 2560×1664, imac_dark 4032×1908) are excluded because
//!   they cost 5–20 s/cell at e8+; smaller screens (terminal 800×600,
//!   windows95 320×200, graph 1280×800) carry the screenshot coverage.
//! - **Effort spread**: 3 cells at e5, 2 at e6, 3 at e7, 2 at e8, 2 at e9.
//!   Covers the W44-171 e5/e6/e7 DC tree wedge, the W44-105/107 e8/e9
//!   buttloop seed gate, and the standard e7 default path.
//! - **Content mix**: 8 CID22 photos (the validation gold-standard split),
//!   3 small screenshots, 1 CLIC2025-1024 photo.
//! - **Distance spread**: d=0.5/1.0/2.0/3.0 — covers the high-d photo
//!   discriminator (W44-91/96/98/99) and the low-d screenshot lift (W22-1).
//!
//! ## Slack tolerances (per-cell)
//!
//! - `bytes_pct_slack: +1.5 pp` (cell must stay within +1.5 pp of locked
//!   `delta_bytes_pct`). At -0.537 % baseline this means must stay ≤ +0.963 %.
//! - `ssim2_slack: -0.30` (cell SSIM2 must stay within 0.30 of locked
//!   `ours_ssim2`). Matches the W44-100 / W44-101 / W44-107 acceptance gate
//!   used throughout cjxl-parity work.
//! - `bfly_pct_slack: +5.0 pp` (looser than bytes because butteraugli is
//!   noisier and we accept small bfly drift in exchange for SSIM2 gains).
//! - `ms_pct_slack: +75 pp` (wall-time is noisy under load; aggregate
//!   is tighter).
//!
//! ## Aggregate gates (over 12 cells)
//!
//! - mean `delta_bytes_pct` must stay ≤ `BASELINE_MEAN_BYTES_PCT + 1.5`
//! - mean `delta_ssim2` must stay ≥ `BASELINE_MEAN_DELTA_SSIM2 - 0.18`
//! - mean `delta_ms_pct` must stay ≤ `BASELINE_MEAN_MS_PCT + 50.0`
//!
//! ## How to run
//!
//! ```bash
//! # Local run (requires CODEC_CORPUS_DIR pointing at the corpus root):
//! CODEC_CORPUS_DIR=/home/lilith/work/codec-corpus \
//!   cargo test \
//!   --release \
//!   -p jxl-encoder \
//!   --features "parallel butteraugli-loop ssim2-loop zenjxl-regression-gate" \
//!   --test zenjxl_regression_gate \
//!   -- --include-ignored --nocapture
//! ```
//!
//! ## How to rebaseline
//!
//! When the user (and only the user) decides the locked W44-202 numbers
//! are stale and a new baseline should be adopted:
//!
//! 1. Run the W44-170 sweep harness to produce a fresh per-strategy TSV.
//! 2. Re-extract the 12 cells listed in [`LOCKED_CELLS`] from that TSV.
//! 3. Update [`LOCKED_CELLS`] in this file with the new numbers AND
//!    update [`BASELINE_*`] aggregate constants.
//! 4. Bump [`BASELINE_COMMIT_SHA`] + [`BASELINE_DATE`] and add a comment
//!    explaining why the rebaseline happened.
//!
//! This test is INTENTIONALLY brittle — every drift must be explained, not
//! silently absorbed.
//!
//! ## Provenance
//!
//! - Baseline TSV: `benchmarks/cjxl_step025_w44_202_zenjxl_2026-05-22.tsv`
//!   (1978 OK cells × W44-202 commit `56c9bd80`).
//! - Baseline aggregate (full 1978 cells, not just these 12):
//!   - mean `delta_bytes_pct`: -5.88 %
//!   - mean `delta_ssim2`:     +0.484
//!   - mean `delta_bfly_pct`:  -6.68 %
//!   - mean `delta_ms_pct`:    +7.8 %  (median: -15.1 %)
//! - The 12-cell subset has a different aggregate (its own baseline below)
//!   because it intentionally over-samples hard cells (CID22 + screens at
//!   d≥2.0). The 12-cell subset's own locked aggregates are what this
//!   test asserts; the 1978-cell aggregates are documented for context.

#![cfg(all(
    feature = "zenjxl-regression-gate",
    feature = "butteraugli-loop",
    feature = "ssim2-loop",
    feature = "parallel"
))]

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ── Provenance ─────────────────────────────────────────────────────────────

/// The W44-202 baseline was measured at this commit. Source: meta header at
/// `benchmarks/cjxl_step025_w44_202_2026-05-22.meta` (parent_commit field).
// Rebaselined 2026-06-11 (user-approved after bisect): the gate was BORN
// red — it landed 2026-06-09 (b2475e9b) carrying May-22 numbers, but
// commit bbaa12a9 (2026-05-24, W44-AUDIT-8 Phase 5: extra_dc_precision=1
// at effort <= 7, libjxl nl_dc parity) had already moved every e<=7 cell:
// 2x DC precision costs bytes (+1.9..+7.8pp on these cells) and buys the
// CLIC/photo SSIM2-deficit cluster back (aggregate delta_ssim2
// -0.287 -> +0.293). Clean-archive byte bisect over 235 commits put the
// ENTIRE 1418519 e6 d=2 jump (14,266 -> 15,434 B) in that one commit;
// -1 B residual across the rest of the window. cjxl reference values are
// unchanged (same cjxl 0.12.0 run). base_ours_ms values are NOT
// refreshed (debug-only, no assertion; measured-under-load walls are
// not anchor-grade per benchmarks/REVISIT_QUEUE_2026-06-11.md).
const BASELINE_COMMIT_SHA: &str = "420b6a9aa9d5a8a74359c04596b03836cd1c6583";
const BASELINE_DATE: &str = "2026-06-11";

// ── Slack tolerances ───────────────────────────────────────────────────────

const PER_CELL_BYTES_PCT_SLACK: f64 = 1.5;
const PER_CELL_SSIM2_SLACK: f64 = 0.30;
const PER_CELL_BFLY_PCT_SLACK: f64 = 5.0;
const PER_CELL_MS_PCT_SLACK: f64 = 75.0;

const AGG_BYTES_PCT_SLACK: f64 = 1.5;
const AGG_SSIM2_SLACK: f64 = 0.18;
const AGG_MS_PCT_SLACK: f64 = 50.0;

// ── Locked baseline (12 cells, from W44-202 zenjxl TSV) ────────────────────

/// One row from `benchmarks/cjxl_step025_w44_202_zenjxl_2026-05-22.tsv`.
///
/// Field semantics match the sweep harness:
/// - `delta_bytes_pct = (ours - cjxl) / cjxl * 100`
/// - `delta_ssim2     =  ours - cjxl`
/// - `delta_bfly_pct  = (ours - cjxl) / cjxl * 100`
/// - `delta_ms_pct    = (ours - cjxl) / cjxl * 100`
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // `class` is for human-readable provenance, `base_ours_ms` is logged in diagnostic output for reference even when no assertion uses it
struct LockedCell {
    /// Image stem (per `benchmarks/corpora/w44_170_varied_corpus.tsv`).
    name: &'static str,
    /// Subdirectory under `CODEC_CORPUS_DIR` — kept for provenance / debug
    /// output; not used by the encode call.
    class: &'static str,
    /// Path under `CODEC_CORPUS_DIR/<class_dir>/...`.
    relative_path: &'static str,
    effort: u8,
    distance: f32,
    base_ours_bytes: u64,
    base_cjxl_bytes: u64,
    base_ours_ssim2: f64,
    base_cjxl_ssim2: f64,
    base_ours_bfly: f64,
    base_cjxl_bfly: f64,
    /// Locked wall time from W44-202; not asserted (wall time is noisy
    /// under load — we only gate on `delta_ms_pct`). Kept for debug printout.
    base_ours_ms: f64,
    base_cjxl_ms: f64,
    base_delta_bytes_pct: f64,
    base_delta_ssim2: f64,
    base_delta_bfly_pct: f64,
    base_delta_ms_pct: f64,
}

/// 12-cell locked subset. Numbers extracted directly from
/// `benchmarks/cjxl_step025_w44_202_zenjxl_2026-05-22.tsv` at
/// [`BASELINE_COMMIT_SHA`].
const LOCKED_CELLS: &[LockedCell] = &[
    // ── e5 (lowest effort with full divergence stack) ──────────────────
    LockedCell {
        name: "1025469",
        class: "cid22",
        relative_path: "CID22/CID22-512/validation/1025469.png",
        effort: 5,
        distance: 1.0,
        base_ours_bytes: 38783,
        base_cjxl_bytes: 38010,
        base_ours_ssim2: 88.4938,
        base_cjxl_ssim2: 88.1120,
        base_ours_bfly: 1.402900,
        base_cjxl_bfly: 1.408793,
        base_ours_ms: 135.905,
        base_cjxl_ms: 304.971,
        base_delta_bytes_pct: 2.034,
        base_delta_ssim2: 0.3818,
        base_delta_bfly_pct: -0.419,
        base_delta_ms_pct: -88.61,
    },
    LockedCell {
        name: "1189261",
        class: "cid22",
        relative_path: "CID22/CID22-512/validation/1189261.png",
        effort: 5,
        distance: 0.5,
        base_ours_bytes: 106025,
        base_cjxl_bytes: 104752,
        base_ours_ssim2: 91.9682,
        base_cjxl_ssim2: 91.6778,
        base_ours_bfly: 0.805500,
        base_cjxl_bfly: 0.822214,
        base_ours_ms: 225.011,
        base_cjxl_ms: 472.802,
        base_delta_bytes_pct: 1.215,
        base_delta_ssim2: 0.2904,
        base_delta_bfly_pct: -2.033,
        base_delta_ms_pct: -92.42,
    },
    LockedCell {
        name: "terminal",
        class: "screenshot",
        relative_path: "gb82-sc/terminal.png",
        effort: 5,
        distance: 1.0,
        base_ours_bytes: 51693,
        base_cjxl_bytes: 133671,
        base_ours_ssim2: 94.3520,
        base_cjxl_ssim2: 92.2658,
        base_ours_bfly: 0.745000,
        base_cjxl_bfly: 0.755738,
        base_ours_ms: 1646.3,
        base_cjxl_ms: 1014.0,
        base_delta_bytes_pct: -61.328,
        base_delta_ssim2: 2.0862,
        base_delta_bfly_pct: -1.424,
        base_delta_ms_pct: -73.23,
    },
    // ── e6 ────────────────────────────────────────────────────────────
    LockedCell {
        name: "1418519",
        class: "cid22",
        relative_path: "CID22/CID22-512/validation/1418519.png",
        effort: 6,
        distance: 2.0,
        base_ours_bytes: 15433,
        base_cjxl_bytes: 14404,
        base_ours_ssim2: 83.5071,
        base_cjxl_ssim2: 81.9194,
        base_ours_bfly: 2.147500,
        base_cjxl_bfly: 2.150974,
        base_ours_ms: 313.1,
        base_cjxl_ms: 616.6,
        base_delta_bytes_pct: 7.144,
        base_delta_ssim2: 1.5877,
        base_delta_bfly_pct: -0.163,
        base_delta_ms_pct: -92.80,
    },
    LockedCell {
        name: "graph",
        class: "screenshot",
        relative_path: "gb82-sc/graph.png",
        effort: 6,
        distance: 0.5,
        base_ours_bytes: 28876,
        base_cjxl_bytes: 27993,
        base_ours_ssim2: 95.2080,
        base_cjxl_ssim2: 95.0728,
        base_ours_bfly: 0.346800,
        base_cjxl_bfly: 0.389995,
        base_ours_ms: 429.3,
        base_cjxl_ms: 369.2,
        base_delta_bytes_pct: 3.154,
        base_delta_ssim2: 0.1352,
        base_delta_bfly_pct: -11.076,
        base_delta_ms_pct: -87.58,
    },
    // ── e7 (default-ish effort, biggest sample) ────────────────────────
    LockedCell {
        name: "1531677",
        class: "cid22",
        relative_path: "CID22/CID22-512/validation/1531677.png",
        effort: 7,
        distance: 3.0,
        base_ours_bytes: 31438,
        base_cjxl_bytes: 32698,
        base_ours_ssim2: 65.3129,
        base_cjxl_ssim2: 65.2641,
        base_ours_bfly: 3.361900,
        base_cjxl_bfly: 3.261528,
        base_ours_ms: 478.2,
        base_cjxl_ms: 528.1,
        base_delta_bytes_pct: -3.853,
        base_delta_ssim2: 0.0488,
        base_delta_bfly_pct: 3.078,
        base_delta_ms_pct: -89.88,
    },
    LockedCell {
        name: "1420710",
        class: "cid22",
        relative_path: "CID22/CID22-512/validation/1420710.png",
        effort: 7,
        distance: 2.0,
        base_ours_bytes: 56814,
        base_cjxl_bytes: 55869,
        base_ours_ssim2: 78.4071,
        base_cjxl_ssim2: 77.9000,
        base_ours_bfly: 2.347900,
        base_cjxl_bfly: 2.336734,
        base_ours_ms: 329.8,
        base_cjxl_ms: 402.2,
        base_delta_bytes_pct: 1.691,
        base_delta_ssim2: 0.5071,
        base_delta_bfly_pct: 0.477,
        base_delta_ms_pct: -87.69,
    },
    LockedCell {
        name: "windows95",
        class: "screenshot",
        relative_path: "gb82-sc/windows95.png",
        effort: 7,
        distance: 0.5,
        base_ours_bytes: 49124,
        base_cjxl_bytes: 45869,
        base_ours_ssim2: 94.0592,
        base_cjxl_ssim2: 93.9788,
        base_ours_bfly: 0.554800,
        base_cjxl_bfly: 0.748572,
        base_ours_ms: 388.5,
        base_cjxl_ms: 553.6,
        base_delta_bytes_pct: 7.096,
        base_delta_ssim2: 0.0804,
        base_delta_bfly_pct: -25.884,
        base_delta_ms_pct: -89.61,
    },
    // ── e8 (buttloop on) ──────────────────────────────────────────────
    LockedCell {
        name: "1420710",
        class: "cid22",
        relative_path: "CID22/CID22-512/validation/1420710.png",
        effort: 8,
        distance: 2.0,
        base_ours_bytes: 51657,
        base_cjxl_bytes: 52481,
        base_ours_ssim2: 76.5142,
        base_cjxl_ssim2: 77.0121,
        base_ours_bfly: 2.343100,
        base_cjxl_bfly: 2.495016,
        base_ours_ms: 1024.9,
        base_cjxl_ms: 1745.3,
        base_delta_bytes_pct: -1.570,
        base_delta_ssim2: -0.4979,
        base_delta_bfly_pct: -6.088,
        base_delta_ms_pct: -91.70,
    },
    LockedCell {
        name: "windows95",
        class: "screenshot",
        relative_path: "gb82-sc/windows95.png",
        effort: 8,
        distance: 0.5,
        base_ours_bytes: 45909,
        base_cjxl_bytes: 53534,
        base_ours_ssim2: 93.7581,
        base_cjxl_ssim2: 94.1815,
        base_ours_bfly: 0.551800,
        base_cjxl_bfly: 0.623475,
        base_ours_ms: 1550.4,
        base_cjxl_ms: 1709.5,
        base_delta_bytes_pct: -14.243,
        base_delta_ssim2: -0.4234,
        base_delta_bfly_pct: -11.494,
        base_delta_ms_pct: -91.74,
    },
    // ── e9 (max effort) ───────────────────────────────────────────────
    LockedCell {
        name: "1475938",
        class: "cid22",
        relative_path: "CID22/CID22-512/validation/1475938.png",
        effort: 9,
        distance: 1.0,
        base_ours_bytes: 32099,
        base_cjxl_bytes: 32355,
        base_ours_ssim2: 87.6275,
        base_cjxl_ssim2: 87.9583,
        base_ours_bfly: 1.171100,
        base_cjxl_bfly: 1.101691,
        base_ours_ms: 1437.5,
        base_cjxl_ms: 2608.7,
        base_delta_bytes_pct: -0.791,
        base_delta_ssim2: -0.3308,
        base_delta_bfly_pct: 6.303,
        base_delta_ms_pct: -90.94,
    },
    LockedCell {
        name: "1025469",
        class: "cid22",
        relative_path: "CID22/CID22-512/validation/1025469.png",
        effort: 9,
        distance: 2.0,
        base_ours_bytes: 20203,
        base_cjxl_bytes: 21008,
        base_ours_ssim2: 78.5006,
        base_cjxl_ssim2: 78.8558,
        base_ours_bfly: 2.073100,
        base_cjxl_bfly: 2.075254,
        base_ours_ms: 1792.3,
        base_cjxl_ms: 2570.6,
        base_delta_bytes_pct: -3.832,
        base_delta_ssim2: -0.3552,
        base_delta_bfly_pct: -0.102,
        base_delta_ms_pct: -90.84,
    },
];

/// Aggregate over the 12 locked cells (mean of per-cell deltas).
/// Computed offline from [`LOCKED_CELLS`] (verified at test runtime).
const BASELINE_MEAN_BYTES_PCT: f64 = -5.274;
const BASELINE_MEAN_DELTA_SSIM2: f64 = 0.2925;
const BASELINE_MEAN_DELTA_BFLY_PCT: f64 = -4.069;
const BASELINE_MEAN_MS_PCT: f64 = -88.92;

// ── Corpus + I/O ───────────────────────────────────────────────────────────

fn corpus_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| "/home/lilith/work/codec-corpus".to_string()),
    )
}

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| {
        panic!(
            "failed to read corpus PNG `{}`: {e}. \
             Set CODEC_CORPUS_DIR to the corpus root (default \
             /home/lilith/work/codec-corpus). The Imazen corpus lives at \
             https://github.com/imazen/codec-corpus.",
            path.display()
        )
    });
    let rgb = img.to_rgb8();
    (rgb.as_raw().clone(), rgb.width(), rgb.height())
}

// ── Color helpers (mirror w44_170_cjxl_step025_sweep.rs) ───────────────────

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
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

fn rgb_to_linear_img(rgb: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let pixels: Vec<RGB<f32>> = rgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear_f32(c[0]),
                srgb_to_linear_f32(c[1]),
                srgb_to_linear_f32(c[2]),
            )
        })
        .collect();
    Img::new(pixels, w as usize, h as usize)
}

fn rgb_to_srgb_arr3(rgb: &[u8], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w as usize, h as usize)
}

// ── Encode + decode ────────────────────────────────────────────────────────

fn encode_zenjxl(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> (Vec<u8>, f64) {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_threads(1);
    let start = Instant::now();
    let bytes = cfg
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("zenjxl encode failed");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn decode_jxl_linear(bytes: &[u8]) -> (usize, usize, Vec<f32>) {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide failed to parse encoded bytes");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img
        .render_frame(0)
        .expect("jxl-oxide failed to render frame 0");
    let fb = render.image_all_channels();
    (fb.width(), fb.height(), fb.buf().to_vec())
}

fn score_jxl(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> (f64, f64) {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes);
    assert_eq!(
        (dw, dh),
        (w as usize, h as usize),
        "decoded dimensions mismatch source"
    );
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(
        orig_linear.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .expect("butteraugli compute failed")
    .score as f64;

    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref())
        .expect("ssim2 compute failed");

    (bfly, ssim2)
}

// ── Test harness ───────────────────────────────────────────────────────────

#[allow(dead_code)] // `cur_ms` is logged in the per-cell printout but the assertion only uses `cur_delta_ms_pct`
struct CellResult {
    locked: LockedCell,
    cur_bytes: u64,
    cur_ssim2: f64,
    cur_bfly: f64,
    cur_ms: f64,
    cur_delta_bytes_pct: f64,
    cur_delta_ssim2: f64,
    cur_delta_bfly_pct: f64,
    cur_delta_ms_pct: f64,
}

fn run_cell(locked: &LockedCell) -> CellResult {
    let png_path = corpus_dir().join(locked.relative_path);
    let (rgb, w, h) = load_png(&png_path);
    let orig_lin = rgb_to_linear_img(&rgb, w, h);
    let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);
    let (bytes, ms) = encode_zenjxl(&rgb, w, h, locked.distance, locked.effort);
    let (bfly, ssim2) = score_jxl(&bytes, &orig_lin, &orig_srgb, w, h);
    let cur_bytes = bytes.len() as u64;
    let cjxl_b = locked.base_cjxl_bytes as f64;
    let cjxl_s = locked.base_cjxl_ssim2;
    let cjxl_bf = locked.base_cjxl_bfly;
    let cjxl_ms = locked.base_cjxl_ms;
    CellResult {
        locked: *locked,
        cur_bytes,
        cur_ssim2: ssim2,
        cur_bfly: bfly,
        cur_ms: ms,
        cur_delta_bytes_pct: (cur_bytes as f64 - cjxl_b) / cjxl_b * 100.0,
        cur_delta_ssim2: ssim2 - cjxl_s,
        cur_delta_bfly_pct: (bfly - cjxl_bf) / cjxl_bf * 100.0,
        cur_delta_ms_pct: (ms - cjxl_ms) / cjxl_ms * 100.0,
    }
}

fn format_cell_diff(r: &CellResult) -> String {
    let l = &r.locked;
    let cell_tag = format!("{} e{} d={}", l.name, l.effort, l.distance);
    format!(
        "  {cell_tag}\n\
         {:<8}lock cur diff   slack\n\
         {:<8}{} {} {:+}\n\
         {:<8}{:.4} {:.4} {:+.4}     ±{}\n\
         {:<8}{:.4} {:.4} {:+.4}\n\
         {:<8}{:+.3}% {:+.3}% {:+.3}pp      ±{}pp\n\
         {:<8}{:+.4} {:+.4} {:+.4}    ±{}\n\
         {:<8}{:+.3}% {:+.3}% {:+.3}pp      ±{}pp\n\
         {:<8}{:+.2}% {:+.2}% {:+.2}pp       ±{}pp",
        "",
        "bytes:",
        l.base_ours_bytes,
        r.cur_bytes,
        (r.cur_bytes as i64) - (l.base_ours_bytes as i64),
        "ssim2:",
        l.base_ours_ssim2,
        r.cur_ssim2,
        r.cur_ssim2 - l.base_ours_ssim2,
        PER_CELL_SSIM2_SLACK,
        "bfly:",
        l.base_ours_bfly,
        r.cur_bfly,
        r.cur_bfly - l.base_ours_bfly,
        "Δbytes:",
        l.base_delta_bytes_pct,
        r.cur_delta_bytes_pct,
        r.cur_delta_bytes_pct - l.base_delta_bytes_pct,
        PER_CELL_BYTES_PCT_SLACK,
        "Δssim2:",
        l.base_delta_ssim2,
        r.cur_delta_ssim2,
        r.cur_delta_ssim2 - l.base_delta_ssim2,
        PER_CELL_SSIM2_SLACK,
        "Δbfly:",
        l.base_delta_bfly_pct,
        r.cur_delta_bfly_pct,
        r.cur_delta_bfly_pct - l.base_delta_bfly_pct,
        PER_CELL_BFLY_PCT_SLACK,
        "Δms:",
        l.base_delta_ms_pct,
        r.cur_delta_ms_pct,
        r.cur_delta_ms_pct - l.base_delta_ms_pct,
        PER_CELL_MS_PCT_SLACK,
    )
}

#[test]
#[ignore = "regression gate; run via --include-ignored when CODEC_CORPUS_DIR is set"]
fn zenjxl_regression_gate_w44_202_baseline() {
    eprintln!(
        "Zenjxl regression gate — baseline {} (commit {}…), {} cells\n",
        BASELINE_DATE,
        &BASELINE_COMMIT_SHA[..8],
        LOCKED_CELLS.len()
    );

    let mut results: Vec<CellResult> = Vec::with_capacity(LOCKED_CELLS.len());
    let mut per_cell_failures: Vec<(LockedCell, Vec<String>)> = Vec::new();

    let total_start = Instant::now();
    for cell in LOCKED_CELLS {
        let r = run_cell(cell);
        let mut failures: Vec<String> = Vec::new();

        let bytes_drift = r.cur_delta_bytes_pct - cell.base_delta_bytes_pct;
        if bytes_drift > PER_CELL_BYTES_PCT_SLACK {
            failures.push(format!(
                "delta_bytes_pct drifted +{:.3}pp (slack +{:.3}pp)",
                bytes_drift, PER_CELL_BYTES_PCT_SLACK
            ));
        }

        let ssim2_drift = r.cur_ssim2 - cell.base_ours_ssim2;
        if ssim2_drift < -PER_CELL_SSIM2_SLACK {
            failures.push(format!(
                "ours_ssim2 dropped {:+.4} (slack -{:.3})",
                ssim2_drift, PER_CELL_SSIM2_SLACK
            ));
        }

        let bfly_drift = r.cur_delta_bfly_pct - cell.base_delta_bfly_pct;
        if bfly_drift > PER_CELL_BFLY_PCT_SLACK {
            failures.push(format!(
                "delta_bfly_pct drifted +{:.3}pp (slack +{:.3}pp)",
                bfly_drift, PER_CELL_BFLY_PCT_SLACK
            ));
        }

        let ms_drift = r.cur_delta_ms_pct - cell.base_delta_ms_pct;
        if ms_drift > PER_CELL_MS_PCT_SLACK {
            failures.push(format!(
                "delta_ms_pct drifted +{:.3}pp (slack +{:.3}pp)",
                ms_drift, PER_CELL_MS_PCT_SLACK
            ));
        }

        eprintln!("{}", format_cell_diff(&r));
        if !failures.is_empty() {
            eprintln!("    !! FAILURES: {}", failures.join("; "));
            per_cell_failures.push((*cell, failures));
        }
        eprintln!();
        results.push(r);
    }

    let total_secs = total_start.elapsed().as_secs_f64();
    eprintln!("Total cell runtime: {:.1}s\n", total_secs);

    // Aggregate stats
    let n = results.len() as f64;
    let mean_bytes_pct: f64 = results.iter().map(|r| r.cur_delta_bytes_pct).sum::<f64>() / n;
    let mean_ssim2: f64 = results.iter().map(|r| r.cur_delta_ssim2).sum::<f64>() / n;
    let mean_bfly_pct: f64 = results.iter().map(|r| r.cur_delta_bfly_pct).sum::<f64>() / n;
    let mean_ms_pct: f64 = results.iter().map(|r| r.cur_delta_ms_pct).sum::<f64>() / n;

    eprintln!(
        "Aggregate over {} cells:\n  delta_bytes_pct: locked={:+.3}, current={:+.3}, drift={:+.3}pp (slack +{}pp)\n  delta_ssim2:     locked={:+.4}, current={:+.4}, drift={:+.4} (slack -{})\n  delta_bfly_pct:  locked={:+.3}, current={:+.3}, drift={:+.3}pp (no aggregate gate)\n  delta_ms_pct:    locked={:+.2}, current={:+.2}, drift={:+.2}pp (slack +{}pp)\n",
        results.len(),
        BASELINE_MEAN_BYTES_PCT,
        mean_bytes_pct,
        mean_bytes_pct - BASELINE_MEAN_BYTES_PCT,
        AGG_BYTES_PCT_SLACK,
        BASELINE_MEAN_DELTA_SSIM2,
        mean_ssim2,
        mean_ssim2 - BASELINE_MEAN_DELTA_SSIM2,
        AGG_SSIM2_SLACK,
        BASELINE_MEAN_DELTA_BFLY_PCT,
        mean_bfly_pct,
        mean_bfly_pct - BASELINE_MEAN_DELTA_BFLY_PCT,
        BASELINE_MEAN_MS_PCT,
        mean_ms_pct,
        mean_ms_pct - BASELINE_MEAN_MS_PCT,
        AGG_MS_PCT_SLACK,
    );

    // Aggregate gates
    let mut agg_failures: Vec<String> = Vec::new();
    if mean_bytes_pct - BASELINE_MEAN_BYTES_PCT > AGG_BYTES_PCT_SLACK {
        agg_failures.push(format!(
            "mean delta_bytes_pct drifted +{:.3}pp (slack +{:.1}pp)",
            mean_bytes_pct - BASELINE_MEAN_BYTES_PCT,
            AGG_BYTES_PCT_SLACK
        ));
    }
    if mean_ssim2 - BASELINE_MEAN_DELTA_SSIM2 < -AGG_SSIM2_SLACK {
        agg_failures.push(format!(
            "mean delta_ssim2 dropped {:+.4} (slack -{:.3})",
            mean_ssim2 - BASELINE_MEAN_DELTA_SSIM2,
            AGG_SSIM2_SLACK
        ));
    }
    if mean_ms_pct - BASELINE_MEAN_MS_PCT > AGG_MS_PCT_SLACK {
        agg_failures.push(format!(
            "mean delta_ms_pct drifted +{:.3}pp (slack +{:.1}pp)",
            mean_ms_pct - BASELINE_MEAN_MS_PCT,
            AGG_MS_PCT_SLACK
        ));
    }

    if !per_cell_failures.is_empty() || !agg_failures.is_empty() {
        eprintln!("=== ZENJXL REGRESSION GATE FAILED ===");
        for (cell, fails) in &per_cell_failures {
            eprintln!(
                "  cell {} e{} d={}: {}",
                cell.name,
                cell.effort,
                cell.distance,
                fails.join("; ")
            );
        }
        for fail in &agg_failures {
            eprintln!("  AGG: {}", fail);
        }
        panic!(
            "Zenjxl regression gate failed: {} per-cell + {} aggregate violations.\n\
             Locked baseline: {} (commit {}…).\n\
             If the regression is INTENTIONAL, update LOCKED_CELLS + BASELINE_* constants \
             AND bump BASELINE_COMMIT_SHA/BASELINE_DATE. See module doc for rebaseline steps.",
            per_cell_failures.len(),
            agg_failures.len(),
            BASELINE_DATE,
            &BASELINE_COMMIT_SHA[..8],
        );
    }

    eprintln!("=== ZENJXL REGRESSION GATE PASSED ({:.1}s) ===", total_secs);
}

// ── Sanity test: locked aggregate constants stay in sync with LOCKED_CELLS ─

/// Recomputes the BASELINE_MEAN_* constants from LOCKED_CELLS and asserts
/// they match. This is the test that catches "someone added a cell but
/// forgot to update the aggregate" — runs cheaply with `cargo test`
/// (no encode/decode work).
#[test]
fn locked_aggregate_constants_match_cells() {
    let n = LOCKED_CELLS.len() as f64;
    let mean_bytes_pct: f64 = LOCKED_CELLS
        .iter()
        .map(|c| c.base_delta_bytes_pct)
        .sum::<f64>()
        / n;
    let mean_ssim2: f64 = LOCKED_CELLS.iter().map(|c| c.base_delta_ssim2).sum::<f64>() / n;
    let mean_bfly_pct: f64 = LOCKED_CELLS
        .iter()
        .map(|c| c.base_delta_bfly_pct)
        .sum::<f64>()
        / n;
    let mean_ms_pct: f64 = LOCKED_CELLS
        .iter()
        .map(|c| c.base_delta_ms_pct)
        .sum::<f64>()
        / n;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    assert!(
        approx_eq(mean_bytes_pct, BASELINE_MEAN_BYTES_PCT, 0.01),
        "BASELINE_MEAN_BYTES_PCT={:.3} but LOCKED_CELLS mean is {:.3}",
        BASELINE_MEAN_BYTES_PCT,
        mean_bytes_pct
    );
    assert!(
        approx_eq(mean_ssim2, BASELINE_MEAN_DELTA_SSIM2, 0.001),
        "BASELINE_MEAN_DELTA_SSIM2={:.4} but LOCKED_CELLS mean is {:.4}",
        BASELINE_MEAN_DELTA_SSIM2,
        mean_ssim2
    );
    assert!(
        approx_eq(mean_bfly_pct, BASELINE_MEAN_DELTA_BFLY_PCT, 0.01),
        "BASELINE_MEAN_DELTA_BFLY_PCT={:.3} but LOCKED_CELLS mean is {:.3}",
        BASELINE_MEAN_DELTA_BFLY_PCT,
        mean_bfly_pct
    );
    assert!(
        approx_eq(mean_ms_pct, BASELINE_MEAN_MS_PCT, 0.01),
        "BASELINE_MEAN_MS_PCT={:.3} but LOCKED_CELLS mean is {:.3}",
        BASELINE_MEAN_MS_PCT,
        mean_ms_pct
    );
}
