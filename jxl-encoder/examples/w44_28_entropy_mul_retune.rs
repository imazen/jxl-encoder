//! W44-28 entropy_mul_table re-tune at d>4 for F-D wedge.
//!
//! Goal: per the W44-27 investigation, AdjustQuantBlockAC D-heuristic fires
//! on 79.8 % of DCT8 blocks at the F-D residual cells (1531677 d=4,
//! 1420710 d=6). Every excess DCT8 pick where cjxl picks DCT32X32 inflates
//! quant by +1 → +6.8 % bytes on these cells. Hypothesis: lowering
//! `entropy_mul[DCT16X16]` and `entropy_mul[DCT32X32]` makes large
//! transforms cheaper, so the AC strategy selector picks them over DCT8
//! on smooth regions where cjxl already does.
//!
//! Sweep grid:
//!   DCT16X16 ∈ {1.20, 1.27, 1.34 (baseline), 1.41, 1.48}
//!   DCT32X32 ∈ {1.20, 1.34, 1.48 (baseline), 1.62, 1.76}
//!   DCT16X32 / DCT32X16: scale with DCT32X32 (× 1.49/1.48 ratio)
//!   DCT64X32 / DCT64X64: hold at 2.25 baseline (rare at 512×512)
//!   = 25 cells per (image, distance), × 2 (image, distance) cells = 50 encodes
//!
//! Screenshot regression check: 3 screenshots × 1 winning tuple
//!   at e=7 d∈{3,4,5,6} (12 baseline + 12 lifted) — only run if there's
//!   a winner in stage A.
//!
//! Each lifted measurement is paired against a baseline cached per
//! (image, distance) — encoder is deterministic.
//!
//! Run:
//!   cargo run -p jxl-encoder --release --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_28_entropy_mul_retune [-- <out.tsv>]
//!
//! Default output: `benchmarks/w44_28_entropy_mul_retune_2026-05-19.tsv`.

#![allow(
    clippy::too_many_arguments,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::unnecessary_cast
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const EFFORT: u8 = 5;

// F-D residual cells (chosen as 2-worst by bytes_delta from W44-1 ledger).
// Cell 1: 1531677.png d=4.0 (+6.843% bytes vs cjxl, OPEN)
// Cell 2: 1420710.png d=6.0 (+6.839% bytes vs cjxl, OPEN)
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

// Screenshot regression check cells (only if winner found in stage A).
// Distances span the patches+W44-8 window where screenshots are sensitive.
const SCREEN_LABELS: &[(&str, &str)] = &[
    ("gb82/terminal", "gb82-sc/terminal.png"),
    ("gb82/codec_wiki", "gb82-sc/codec_wiki.png"),
    ("gb82/imac_g3", "gb82-sc/imac_g3.png"),
];
const SCREEN_DISTANCES: &[f32] = &[3.0, 4.0, 5.0, 6.0];

const DCT16X16_SWEEP: &[f32] = &[1.20, 1.27, 1.34, 1.41, 1.48];
const DCT32X32_SWEEP: &[f32] = &[1.20, 1.34, 1.48, 1.62, 1.76];

fn build_entropy_table(dct16: f32, dct32: f32) -> EntropyMulTable {
    let mut t = EntropyMulTable::reference();
    t.dct16x16 = dct16;
    t.dct32x32 = dct32;
    // Scale DCT16X32/DCT32X16 with DCT32X32 (preserve the libjxl 1.49/1.48 ratio).
    t.dct16x32 = dct32 * (1.49 / 1.48);
    t
}

fn encode_baseline(rgb_u8: &[u8], w: u32, h: u32, d: f32) -> Result<Vec<u8>, String> {
    LossyConfig::new(d)
        .with_effort(EFFORT)
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("baseline encode failed: {e:?}"))
}

fn encode_lifted(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    dct16: f32,
    dct32: f32,
) -> Result<Vec<u8>, String> {
    let table = build_entropy_table(dct16, dct32);
    let mut params = LossyInternalParams::default();
    params.entropy_mul_table = Some(table);
    LossyConfig::new(d)
        .with_effort(EFFORT)
        .with_internal_params(params)
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("lifted encode failed: {e:?}"))
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

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

fn measure_bytes_bfly_ssim2(
    bytes: Vec<u8>,
    w: u32,
    h: u32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
    encode_ms: f64,
) -> Result<Measure, String> {
    let (dw, dh, decoded_linear) =
        decode_jxl_linear(&bytes).ok_or_else(|| "decode failed".to_string())?;
    if dw != w as usize || dh != h as usize {
        return Err(format!("decoded {}x{} != {}x{}", dw, dh, w, h));
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score as f64)
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

    Ok(Measure {
        bytes: bytes.len(),
        butteraugli: bfly,
        ssim2,
        encode_ms,
    })
}

fn measure_baseline(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let t0 = Instant::now();
    let bytes = encode_baseline(rgb_u8, w, h, d)?;
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    measure_bytes_bfly_ssim2(
        bytes,
        w,
        h,
        orig_linear_img,
        orig_srgb_img,
        params,
        encode_ms,
    )
}

fn measure_lifted(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    dct16: f32,
    dct32: f32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let t0 = Instant::now();
    let bytes = encode_lifted(rgb_u8, w, h, d, dct16, dct32)?;
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    measure_bytes_bfly_ssim2(
        bytes,
        w,
        h,
        orig_linear_img,
        orig_srgb_img,
        params,
        encode_ms,
    )
}

fn load_linear_and_srgb(
    path: &PathBuf,
) -> Option<(Vec<u8>, u32, u32, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>)> {
    let img = image::open(path).ok()?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let rgb_u8: Vec<u8> = rgb.as_raw().clone();
    let linear_rgb: Vec<RGB<f32>> = rgb
        .pixels()
        .map(|p| {
            RGB::new(
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            )
        })
        .collect();
    let orig_linear = Img::new(linear_rgb, w as usize, h as usize);
    let srgb_arr3: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    let orig_srgb = Img::new(srgb_arr3, w as usize, h as usize);
    Some((rgb_u8, w, h, orig_linear, orig_srgb))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = args
        .into_iter()
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/w44_28_entropy_mul_retune_2026-05-19.tsv"));

    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/w44_28_entropy_mul_retune_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).expect("create staging tsv");
    writeln!(
        out,
        "stage\timage\tclass\teffort\tdistance\tdct16x16\tdct32x32\tdct16x32\t\
         baseline_bytes\tlifted_bytes\tbaseline_bfly\tlifted_bfly\tbaseline_ssim2\tlifted_ssim2\t\
         bytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs\tbaseline_ms\tlifted_ms"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();

    // Cache best lifted tuple per cell for stage B decisions.
    // key = (image_label, effort, distance_key * 1000)
    let mut best_per_cell: HashMap<(String, u8, u32), (f32, f32, f64, f64, f64)> = HashMap::new();

    // ─── Stage A: F-D residual sweep ────────────────────────────────────
    eprintln!("=== Stage A: F-D residual sweep ===");
    eprintln!(
        "  DCT16X16 ∈ {:?}, DCT32X32 ∈ {:?}, effort={}",
        DCT16X16_SWEEP, DCT32X32_SWEEP, EFFORT
    );
    eprintln!("  DCT16X32/DCT32X16 scale = DCT32X32 × (1.49 / 1.48), DCT64* held at 2.25");

    for &(label, rel_path, d) in FD_CELLS {
        let path = corpus.join(rel_path);
        if !path.exists() {
            eprintln!("MISS {}", path.display());
            continue;
        }
        let (rgb_u8, w, h, orig_linear, orig_srgb) = match load_linear_and_srgb(&path) {
            Some(v) => v,
            None => {
                eprintln!("load failed {}", path.display());
                continue;
            }
        };
        eprintln!("--- {} ({}x{}) e={} d={} ---", label, w, h, EFFORT, d);

        // Baseline once.
        let baseline = match measure_baseline(&rgb_u8, w, h, d, &orig_linear, &orig_srgb, &bparams)
        {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  baseline failed: {e}");
                continue;
            }
        };
        eprintln!(
            "  baseline: {}B bfly={:.4} ssim2={:.4} {:.0}ms",
            baseline.bytes, baseline.butteraugli, baseline.ssim2, baseline.encode_ms
        );

        let mut best_score = f64::INFINITY;
        for &dct16 in DCT16X16_SWEEP {
            for &dct32 in DCT32X32_SWEEP {
                let lifted = match measure_lifted(
                    &rgb_u8,
                    w,
                    h,
                    d,
                    dct16,
                    dct32,
                    &orig_linear,
                    &orig_srgb,
                    &bparams,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("  dct16={dct16} dct32={dct32}: {e}");
                        continue;
                    }
                };
                let dct16x32 = dct32 * (1.49 / 1.48);
                let bytes_delta_pct =
                    (lifted.bytes as f64 - baseline.bytes as f64) / baseline.bytes as f64 * 100.0;
                let bfly_delta_pct = (lifted.butteraugli - baseline.butteraugli)
                    / baseline.butteraugli.max(1e-9)
                    * 100.0;
                let ssim2_delta_abs = lifted.ssim2 - baseline.ssim2;
                writeln!(
                    out,
                    "A\t{}\tphoto\t{}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.3}\t{:+.3}\t{:+.4}\t{:.2}\t{:.2}",
                    label,
                    EFFORT,
                    d,
                    dct16,
                    dct32,
                    dct16x32,
                    baseline.bytes,
                    lifted.bytes,
                    baseline.butteraugli,
                    lifted.butteraugli,
                    baseline.ssim2,
                    lifted.ssim2,
                    bytes_delta_pct,
                    bfly_delta_pct,
                    ssim2_delta_abs,
                    baseline.encode_ms,
                    lifted.encode_ms
                )
                .unwrap();
                out.flush().unwrap();
                eprintln!(
                    "    dct16={:.2} dct32={:.2} → Δbytes={:+.2}% Δbfly={:+.2}% Δssim2={:+.3} ({}B)",
                    dct16, dct32, bytes_delta_pct, bfly_delta_pct, ssim2_delta_abs, lifted.bytes
                );

                // Score: bytes delta dominates, penalize bfly/ssim2 regressions hard.
                // Cell-internal selection: any tuple with bytes_delta <= 0 AND
                // bfly_delta_pct <= +5% AND ssim2_delta_abs >= -0.3 is valid.
                let valid =
                    bytes_delta_pct < 0.0 && bfly_delta_pct <= 5.0 && ssim2_delta_abs >= -0.3;
                if valid && bytes_delta_pct < best_score {
                    best_score = bytes_delta_pct;
                    let dkey = (d * 1000.0).round() as u32;
                    best_per_cell.insert(
                        (label.to_string(), EFFORT, dkey),
                        (
                            dct16,
                            dct32,
                            bytes_delta_pct,
                            bfly_delta_pct,
                            ssim2_delta_abs,
                        ),
                    );
                }
            }
        }
    }

    // ─── Pareto analysis: pick a single tuple winning on BOTH F-D cells ──
    // Approach: find tuples that pass the per-cell validity check on both
    // cells, then pick the one with the best aggregate bytes_delta_pct sum.
    // If no tuple wins both, that's a honest-stop signal.
    eprintln!("\n=== Stage A Pareto analysis ===");
    if best_per_cell.len() < FD_CELLS.len() {
        eprintln!(
            "  per-cell winners: {} (need {})",
            best_per_cell.len(),
            FD_CELLS.len()
        );
        for ((lbl, e, dk), (d16, d32, bd, bf, sd)) in &best_per_cell {
            eprintln!(
                "    {} e{} d={:.3}: dct16={:.2} dct32={:.2} bytes={:+.2}% bfly={:+.2}% ssim2={:+.3}",
                lbl,
                e,
                *dk as f32 / 1000.0,
                d16,
                d32,
                bd,
                bf,
                sd
            );
        }
    }

    // For each (dct16, dct32) tuple, sum per-cell bytes_delta_pct.
    // Only count tuples that are VALID on every F-D cell.
    // (sum_bytes_delta_pct, sum_bfly_delta_pct, sum_ssim2_delta_abs, n_cells_valid)
    type TupleAgg = (f64, f64, f64, u32);
    let mut tuple_scores: HashMap<(u32, u32), TupleAgg> = HashMap::new();

    // Re-read TSV staging to do it cleanly — instead, gather from a second pass.
    // For simplicity here: just re-run measure_lifted? No — that'd double encode time.
    // Instead, parse the staging TSV we just wrote.
    drop(out);
    let tsv_text = std::fs::read_to_string(&staging).expect("read staging");
    let mut lines = tsv_text.lines();
    let _header = lines.next();
    for line in lines {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 19 || cols[0] != "A" {
            continue;
        }
        let dct16: f32 = cols[5].parse().unwrap_or(0.0);
        let dct32: f32 = cols[6].parse().unwrap_or(0.0);
        let bytes_delta_pct: f64 = cols[14].parse().unwrap_or(0.0);
        let bfly_delta_pct: f64 = cols[15].parse().unwrap_or(0.0);
        let ssim2_delta_abs: f64 = cols[16].parse().unwrap_or(0.0);
        let key = (
            (dct16 * 1000.0).round() as u32,
            (dct32 * 1000.0).round() as u32,
        );
        // Validity gate
        let valid = bfly_delta_pct <= 5.0 && ssim2_delta_abs >= -0.3;
        let entry = tuple_scores.entry(key).or_insert((0.0, 0.0, 0.0, 0));
        if valid {
            entry.0 += bytes_delta_pct;
            entry.1 += bfly_delta_pct;
            entry.2 += ssim2_delta_abs;
            entry.3 += 1;
        }
    }

    // Print top-5 candidate tuples by aggregate bytes_delta_pct.
    // ((dct16_x1000, dct32_x1000), sum_bytes, sum_bfly, sum_ssim2, n_cells)
    type TupleRow = ((u32, u32), f64, f64, f64, u32);
    let mut tuples: Vec<TupleRow> = tuple_scores
        .into_iter()
        .filter(|(_, (_, _, _, n))| *n == FD_CELLS.len() as u32) // valid on every cell
        .map(|((d16, d32), (bd, bf, sd, n))| ((d16, d32), bd, bf, sd, n))
        .collect();
    tuples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    eprintln!(
        "  Top-5 candidates (valid on all {} F-D cells, sum-ordered by bytes_delta_pct):",
        FD_CELLS.len()
    );
    for ((d16, d32), bd, bf, sd, _) in tuples.iter().take(5) {
        eprintln!(
            "    dct16={:.3} dct32={:.3} sum_bytes={:+.3}% sum_bfly={:+.3}% sum_ssim2={:+.4}",
            *d16 as f32 / 1000.0,
            *d32 as f32 / 1000.0,
            bd,
            bf,
            sd
        );
    }

    let winner_tuple = tuples.first().copied();

    // Re-open out file in append mode for stage B.
    let mut out = std::fs::OpenOptions::new()
        .append(true)
        .open(&staging)
        .expect("re-open staging");

    // ─── Stage B: screenshot regression check ────────────────────────────
    if let Some(((d16, d32), agg_bytes, agg_bfly, agg_ssim2, _)) = winner_tuple {
        let dct16 = d16 as f32 / 1000.0;
        let dct32 = d32 as f32 / 1000.0;
        eprintln!(
            "\n=== Stage B: screenshot regression check for winner dct16={:.3} dct32={:.3} (agg bytes={:+.3}% bfly={:+.3}% ssim2={:+.4}) ===",
            dct16, dct32, agg_bytes, agg_bfly, agg_ssim2
        );
        for &(slabel, srel) in SCREEN_LABELS {
            let spath = corpus.join(srel);
            if !spath.exists() {
                eprintln!("MISS {}", spath.display());
                continue;
            }
            let (rgb_u8, w, h, orig_linear, orig_srgb) = match load_linear_and_srgb(&spath) {
                Some(v) => v,
                None => {
                    eprintln!("load failed {}", spath.display());
                    continue;
                }
            };
            eprintln!("--- {} ({}x{}) e=7 ---", slabel, w, h);
            for &d in SCREEN_DISTANCES {
                // Screenshots run at e=7 (per the W44-8 / patches window).
                let cfg_base = LossyConfig::new(d).with_effort(7);
                let t0 = Instant::now();
                let bytes_b = match cfg_base.encode(&rgb_u8, w, h, PixelLayout::Rgb8) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("  baseline e=7 d={d}: {e:?}");
                        continue;
                    }
                };
                let base_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let baseline = match measure_bytes_bfly_ssim2(
                    bytes_b,
                    w,
                    h,
                    &orig_linear,
                    &orig_srgb,
                    &bparams,
                    base_ms,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("  baseline measure: {e}");
                        continue;
                    }
                };

                let mut params = LossyInternalParams::default();
                params.entropy_mul_table = Some(build_entropy_table(dct16, dct32));
                let cfg_lift = LossyConfig::new(d)
                    .with_effort(7)
                    .with_internal_params(params);
                let t1 = Instant::now();
                let bytes_l = match cfg_lift.encode(&rgb_u8, w, h, PixelLayout::Rgb8) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("  lifted e=7 d={d}: {e:?}");
                        continue;
                    }
                };
                let lift_ms = t1.elapsed().as_secs_f64() * 1000.0;
                let lifted = match measure_bytes_bfly_ssim2(
                    bytes_l,
                    w,
                    h,
                    &orig_linear,
                    &orig_srgb,
                    &bparams,
                    lift_ms,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("  lifted measure: {e}");
                        continue;
                    }
                };
                let bytes_delta_pct =
                    (lifted.bytes as f64 - baseline.bytes as f64) / baseline.bytes as f64 * 100.0;
                let bfly_delta_pct = (lifted.butteraugli - baseline.butteraugli)
                    / baseline.butteraugli.max(1e-9)
                    * 100.0;
                let ssim2_delta_abs = lifted.ssim2 - baseline.ssim2;
                writeln!(
                    out,
                    "B\t{}\tscreen\t7\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.3}\t{:+.3}\t{:+.4}\t{:.2}\t{:.2}",
                    slabel,
                    d,
                    dct16,
                    dct32,
                    dct32 * (1.49 / 1.48),
                    baseline.bytes,
                    lifted.bytes,
                    baseline.butteraugli,
                    lifted.butteraugli,
                    baseline.ssim2,
                    lifted.ssim2,
                    bytes_delta_pct,
                    bfly_delta_pct,
                    ssim2_delta_abs,
                    baseline.encode_ms,
                    lifted.encode_ms
                )
                .unwrap();
                out.flush().unwrap();
                eprintln!(
                    "  d={d}: Δbytes={:+.3}% Δbfly={:+.3}% Δssim2={:+.4} ({}B vs {}B)",
                    bytes_delta_pct, bfly_delta_pct, ssim2_delta_abs, lifted.bytes, baseline.bytes
                );
            }
        }
    } else {
        eprintln!("\n=== Stage B SKIPPED: no winning tuple in stage A ===");
        eprintln!(
            "  (no tuple satisfied bfly_delta_pct<=+5% AND ssim2_delta_abs>=-0.3 AND bytes_delta_pct<0 on every F-D cell)"
        );
    }

    drop(out);
    std::fs::rename(&staging, &out_path).expect("atomic mv staging → out_path");
    eprintln!("\nTSV: {}", out_path.display());
}
