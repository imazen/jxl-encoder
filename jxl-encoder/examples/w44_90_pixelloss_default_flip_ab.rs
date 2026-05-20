//! W44-90 — paired interleaved A/B/C bench for the PixelLossDispatch
//! default-flip decision (W38-2 follow-on).
//!
//! Goal: decide whether `PixelLossDispatch::Auto` (or
//! `AlwaysSinglePass`/`AlwaysOff` on a gated subset) can become the
//! default at e5 without regressing RD on any cell.
//!
//! Acceptance gates (per W44-90 task spec):
//!   - bytes Δ vs AlwaysOn baseline: ≤ +5% on every cell
//!   - butteraugli Δ: ≤ +3% on every cell
//!   - ssim2 Δ: ≥ -1.5 points on every cell
//! AND median perf win on smooth-photo cells (where Auto fires) ≥ 5 ms.
//!
//! Methodology: paired interleaved A/B/C with 5 trials per cell, ALL
//! variants run back-to-back within each trial to amortise cold-pool /
//! thermal noise (W44-89 lesson). Default RUNS_PER_CELL = 5.
//!
//! Variants:
//!   A = `AlwaysOn`      — baseline (current default)
//!   B = `AlwaysOff`     — full-force skip mask1x1 (W38-2 "AlwaysSinglePass")
//!   C = `Auto`          — content-aware gate (median(mask1x1) > 80)
//!
//! Corpus (wider than W38-2's 8 images):
//!   - 8 CID22-512 photos
//!   - 10 GB82-SC screenshots
//!   - 4 mid-class photos
//!
//! Effort: e5 only (per W38-2 finding that pixel_domain_loss matters
//! most at e5).
//! Distances: {0.5, 1.0, 2.0, 3.0, 5.0}.
//!
//! Reproducer:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_90_pixelloss_default_flip_ab -- \
//!     --out benchmarks/w44_90_pixelloss_default_flip_2026-05-19.tsv

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout, PixelLossDispatch};
use rgb::RGB;
use std::fs;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Convert linear light value to sRGB u8 using the correct sRGB
/// transfer function (NOT gamma 2.2). Matches
/// `tests/quality_compare.rs::linear_to_srgb_u8`.
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 3.0, 5.0];
const EFFORT: u8 = 5;
const RUNS_PER_CELL: usize = 5;

fn dispatch_label(d: PixelLossDispatch) -> &'static str {
    match d {
        PixelLossDispatch::AlwaysOn => "always_on",
        PixelLossDispatch::Auto => "auto",
        PixelLossDispatch::AlwaysOff => "always_off",
    }
}

/// Interleaved order: A, B, C, A, B, C, ... within each trial.
fn all_dispatches() -> [PixelLossDispatch; 3] {
    [
        PixelLossDispatch::AlwaysOn,
        PixelLossDispatch::AlwaysOff,
        PixelLossDispatch::Auto,
    ]
}

struct Source {
    label: &'static str,
    class: &'static str,
    path: PathBuf,
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

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

fn measure_cell(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    effort: u8,
    dispatch: PixelLossDispatch,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_strategy(jxl_encoder::api::EncoderStrategy::Custom(Box::new(jxl_encoder::api::EncoderImprovementsCustom { pixel_loss_dispatch: dispatch, ..Default::default() })));

    let t0 = Instant::now();
    let bytes = cfg
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))?;
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (dw, dh, decoded_linear) =
        decode_jxl_linear(&bytes).ok_or_else(|| "decode failed".to_string())?;
    if dw != w as usize || dh != h as usize {
        return Err(format!("decoded {}x{} ≠ {}x{}", dw, dh, w, h));
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);

    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score)
        .unwrap_or(f64::NAN);

    // SSIM2 wants sRGB u8 inputs. Convert decoded linear → sRGB u8.
    let dec_srgb_pixels: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb_pixels, dw, dh);
    let s2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);

    Ok(Measure {
        bytes: bytes.len(),
        butteraugli: bfly,
        ssim2: s2,
        encode_ms,
    })
}

fn median_f64(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn median_u64(mut v: Vec<u64>) -> u64 {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out_path = PathBuf::from("benchmarks/w44_90_pixelloss_default_flip_2026-05-19.tsv");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--out" && i + 1 < args.len() {
            out_path = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let cid_dir = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let scrn_dir = format!("{}/work/codec-corpus/gb82-sc", base);

    let cid_photo_names: &[&str] = &[
        "1025469.png",
        "1044329.png",
        "1189261.png",
        "1279330.png",
        "1418519.png",
        "1420710.png",
        "1531677.png",
        "2389166.png",
    ];
    let cid_mid_names: &[&str] = &["3637739.png", "1624487.png", "1544947.png", "1475938.png"];
    let scrn_names: &[&str] = &[
        "codec_wiki.png",
        "imac_g3.png",
        "imac_dark.png",
        "terminal.png",
        "windows.png",
        "windows95.png",
        "gmessages.png",
        "imessage.png",
        "graph.png",
        "gui.png",
    ];

    let mut sources: Vec<Source> = Vec::new();
    for n in cid_photo_names {
        sources.push(Source {
            label: n,
            class: "photo",
            path: PathBuf::from(format!("{}/{}", cid_dir, n)),
        });
    }
    for n in cid_mid_names {
        sources.push(Source {
            label: n,
            class: "mid",
            path: PathBuf::from(format!("{}/{}", cid_dir, n)),
        });
    }
    for n in scrn_names {
        sources.push(Source {
            label: n,
            class: "screen",
            path: PathBuf::from(format!("{}/{}", scrn_dir, n)),
        });
    }

    // Filter to images that exist on disk
    sources.retain(|s| s.path.exists());
    if sources.is_empty() {
        eprintln!(
            "No source images found on disk under {} or {}",
            cid_dir, scrn_dir
        );
        std::process::exit(1);
    }

    let dispatches = all_dispatches();
    let total_cells = sources.len() * DISTANCES.len() * dispatches.len();
    eprintln!(
        "Sweep: {} images × 1 effort (e{}) × {} distances × {} dispatches × {} trials (interleaved) = {} encodes",
        sources.len(),
        EFFORT,
        DISTANCES.len(),
        dispatches.len(),
        RUNS_PER_CELL,
        total_cells * RUNS_PER_CELL
    );

    // Per-cell aggregate header.
    let header = "image\tclass\teffort\tdistance\tdispatch\twidth\theight\tbytes_med\tbutteraugli_med\tssim2_med\tencode_ms_med\tencode_ms_min\tencode_ms_max\tencode_ms_p25\tencode_ms_p75\tvariance_pct";

    let tmp = format!(
        "/tmp/w44_90_pixelloss_ab_{}.tsv.partial",
        std::process::id()
    );
    {
        let mut f = fs::File::create(&tmp).expect("create tmp");
        writeln!(f, "{}", header).unwrap();
    }

    // Also dump raw per-trial rows for diagnostic / variance analysis.
    let raw_path = out_path.with_extension("raw.tsv");
    let raw_tmp = format!(
        "/tmp/w44_90_pixelloss_ab_raw_{}.tsv.partial",
        std::process::id()
    );
    {
        let mut f = fs::File::create(&raw_tmp).expect("create raw tmp");
        writeln!(
            f,
            "image\tclass\teffort\tdistance\tdispatch\ttrial\twidth\theight\tbytes\tbutteraugli\tssim2\tencode_ms"
        )
        .unwrap();
    }

    let params = ButteraugliParams::default();
    let t_start = Instant::now();
    let mut row_count = 0usize;

    for src in &sources {
        let img = match image::open(&src.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("skip {}: {e}", src.path.display());
                continue;
            }
        };
        let (w, h) = (img.width(), img.height());
        let rgb = img.to_rgb8();
        let rgb_bytes = rgb.as_raw().clone();

        // Build linear reference (sRGB → linear) for butteraugli.
        let lin_pixels: Vec<RGB<f32>> = rgb_bytes
            .chunks(3)
            .map(|c| {
                RGB::new(
                    srgb_to_linear(c[0]),
                    srgb_to_linear(c[1]),
                    srgb_to_linear(c[2]),
                )
            })
            .collect();
        let orig_linear_img = Img::new(lin_pixels, w as usize, h as usize);
        // sRGB u8 reference for SSIM2.
        let srgb_pixels: Vec<[u8; 3]> = rgb_bytes.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let orig_srgb_img = Img::new(srgb_pixels, w as usize, h as usize);

        for &d in DISTANCES {
            // INTERLEAVED: for each trial, run ALL 3 dispatch variants
            // back-to-back, then move to next trial. This is the W44-89
            // methodology that the W38-2 ab harness lacked.
            let mut per_dispatch_runs: Vec<Vec<Measure>> = vec![Vec::new(); dispatches.len()];

            for trial in 0..RUNS_PER_CELL {
                for (di, &dispatch) in dispatches.iter().enumerate() {
                    match measure_cell(
                        &rgb_bytes,
                        w,
                        h,
                        d,
                        EFFORT,
                        dispatch,
                        &orig_linear_img,
                        &orig_srgb_img,
                        &params,
                    ) {
                        Ok(m) => {
                            per_dispatch_runs[di].push(m);
                            // Stream raw row.
                            let raw_row = format!(
                                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.3}",
                                src.label,
                                src.class,
                                EFFORT,
                                d,
                                dispatch_label(dispatch),
                                trial,
                                w,
                                h,
                                m.bytes,
                                m.butteraugli,
                                m.ssim2,
                                m.encode_ms,
                            );
                            let mut rf =
                                fs::OpenOptions::new().append(true).open(&raw_tmp).unwrap();
                            writeln!(rf, "{}", raw_row).unwrap();
                        }
                        Err(e) => eprintln!(
                            "  err {} d={} dispatch={} trial={}: {e}",
                            src.label,
                            d,
                            dispatch_label(dispatch),
                            trial
                        ),
                    }
                }
                eprint!(".");
            }
            eprintln!();

            for (di, &dispatch) in dispatches.iter().enumerate() {
                let runs = &per_dispatch_runs[di];
                if runs.is_empty() {
                    continue;
                }

                let bytes = median_u64(runs.iter().map(|r| r.bytes as u64).collect::<Vec<_>>());
                let bfly = median_f64(runs.iter().map(|r| r.butteraugli).collect::<Vec<_>>());
                let s2 = median_f64(runs.iter().map(|r| r.ssim2).collect::<Vec<_>>());
                let ms_med = median_f64(runs.iter().map(|r| r.encode_ms).collect::<Vec<_>>());
                let ms_min = runs
                    .iter()
                    .map(|r| r.encode_ms)
                    .fold(f64::INFINITY, f64::min);
                let ms_max = runs
                    .iter()
                    .map(|r| r.encode_ms)
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut sorted: Vec<f64> = runs.iter().map(|r| r.encode_ms).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p25 = sorted[sorted.len() / 4];
                let p75 = sorted[(3 * sorted.len()) / 4];
                let variance_pct = if ms_med > 0.0 {
                    100.0 * (ms_max - ms_min) / ms_med
                } else {
                    0.0
                };

                let row = format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
                    src.label,
                    src.class,
                    EFFORT,
                    d,
                    dispatch_label(dispatch),
                    w,
                    h,
                    bytes,
                    bfly,
                    s2,
                    ms_med,
                    ms_min,
                    ms_max,
                    p25,
                    p75,
                    variance_pct,
                );
                row_count += 1;
                let mut f = fs::OpenOptions::new().append(true).open(&tmp).unwrap();
                writeln!(f, "{}", row).unwrap();
            }
            eprintln!(
                "[{}] {} d={} done ({} dispatches × {} trials)",
                src.class,
                src.label,
                d,
                dispatches.len(),
                RUNS_PER_CELL
            );
        }
    }

    let dur = t_start.elapsed();
    eprintln!(
        "Sweep done in {:.1}s, wrote {} aggregate rows",
        dur.as_secs_f64(),
        row_count
    );

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::rename(&tmp, &out_path).expect("rename tmp → out");
    fs::rename(&raw_tmp, &raw_path).expect("rename raw tmp → raw");
    eprintln!("Wrote {}", out_path.display());
    eprintln!("Wrote {}", raw_path.display());
}
