//! W38-2 #3.1 A/B bench: literal GPU port of distance-aware buttloop
//! split (CPU port of GPU commit `d75bf7c`).
//!
//! **PRE** explicitly sets libjxl defaults at every regime
//! (`cur_pow=0.2`, `max_increase=100.0`), matching the
//! production CPU defaults baked by the port.
//!
//! **POST** explicitly sets the GPU-tuned LOW values
//! (`cur_pow=0.5`, `max_increase=1.3`) at d<2.0 and keeps
//! libjxl defaults at HIGH (d>=2.0). Demonstrates what shipping
//! the literal port as default-on would look like.
//!
//! The PRE/POST sweep at `benchmarks/buttloop_distance_split_port_*.tsv`
//! is the empirical basis for keeping LOW defaults libjxl-faithful in
//! production: POST regresses bfly +4-13 % on screenshots and +1-8 % on
//! photos at d<2.0 (HIGH cells are byte-identical, expected). The
//! atomic overrides remain useful as a sweep harness for future
//! CPU-specific tuning.
//!
//! Grid:
//! - 3 screenshots × {d=0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0} × {e8, e9}
//! - 3 photos      × {d=0.5, 1.0, 1.5, 3.0, 4.0, 5.0}      × {e8, e9}
//!
//! Two modes (PRE / POST) per cell. PRE forces libjxl defaults at all
//! distances (matches the production CPU defaults baked by the port);
//! POST forces the literal GPU LOW tuning at d<2.0.
//!
//! Metric capture: jxl-oxide `srgb_linear` decode + Rust `butteraugli_linear`
//!     + `fast_ssim2::compute_ssimulacra2` (CLAUDE.md compliant — no
//!     `butteraugli_main`, no PNG metadata bug).
//!
//! Output:
//!   benchmarks/buttloop_distance_split_port_<UTC>.tsv  (per-cell paired)
//!   benchmarks/buttloop_distance_split_port_<UTC>.meta (provenance)
//!
//! Reproducer:
//!   cargo run -p jxl-encoder --release --example buttloop_distance_split_ab

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

// Bring atomic-override knobs in scope. They live behind a
// `#[doc(hidden)] pub mod __buttloop_overrides` re-export.
use jxl_encoder::vardct::__buttloop_overrides::{
    CUR_POW_X1000_HIGH, CUR_POW_X1000_LOW, DISTANCE_SPLIT_X1000, MAX_INCREASE_X1000_HIGH,
    MAX_INCREASE_X1000_LOW,
};

const SCREENSHOTS: &[&str] = &["terminal.png", "codec_wiki.png", "imac_g3.png"];
const PHOTOS: &[&str] = &["1025469.png", "1418519.png", "1531677.png"];

// Distance grid covers BOTH regimes:
// - HIGH (d>=2.0): the WF3 wedge; port is a NO-OP here by design (CPU
//   was already at libjxl defaults at HIGH). Documents the boundary
//   condition that the literal GPU port does not reach the WF3 wedge.
// - LOW (d<2.0): the actual scope of the port; PRE vs POST should
//   diverge here.
const SCREENSHOT_DIST: &[f32] = &[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
const PHOTO_DIST: &[f32] = &[0.5, 1.0, 1.5, 3.0, 4.0, 5.0];
const EFFORTS: &[u8] = &[8, 9];

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

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

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
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

fn score(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
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
    .ok()?
    .score as f64;
    // SSIM2 via fast_ssim2: needs sRGB u8 [u8;3].
    let dec_srgb_pixels: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb_pixels, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn set_mode_pre() {
    // PRE = libjxl defaults at all distances: cur_pow=0.2,
    // max_increase=100.0 (≈ no cap). Matches the production CPU
    // defaults baked by the port.
    CUR_POW_X1000_LOW.store(200, Ordering::Relaxed); // 0.200
    CUR_POW_X1000_HIGH.store(200, Ordering::Relaxed); // 0.200
    MAX_INCREASE_X1000_LOW.store(100_000, Ordering::Relaxed); // 100.0
    MAX_INCREASE_X1000_HIGH.store(100_000, Ordering::Relaxed);
    DISTANCE_SPLIT_X1000.store(2000, Ordering::Relaxed);
}

fn set_mode_post() {
    // POST = literal GPU port: LOW (d<2.0) uses GPU-tuned values
    // (cur_pow=0.5, max_increase=1.3); HIGH (d>=2.0) stays at libjxl
    // defaults. Demonstrates what shipping the literal port as
    // default-on would look like on CPU.
    CUR_POW_X1000_LOW.store(500, Ordering::Relaxed); // 0.500
    CUR_POW_X1000_HIGH.store(200, Ordering::Relaxed); // 0.200
    MAX_INCREASE_X1000_LOW.store(1300, Ordering::Relaxed); // 1.300
    MAX_INCREASE_X1000_HIGH.store(100_000, Ordering::Relaxed); // 100.0
    DISTANCE_SPLIT_X1000.store(2000, Ordering::Relaxed);
}

/// Reset all overrides to the production defaults (libjxl-faithful).
/// Called at end of run so a re-invocation starts from clean state.
fn reset_overrides() {
    CUR_POW_X1000_LOW.store(i32::MIN, Ordering::Relaxed);
    CUR_POW_X1000_HIGH.store(i32::MIN, Ordering::Relaxed);
    MAX_INCREASE_X1000_LOW.store(i32::MIN, Ordering::Relaxed);
    MAX_INCREASE_X1000_HIGH.store(i32::MIN, Ordering::Relaxed);
    DISTANCE_SPLIT_X1000.store(2000, Ordering::Relaxed);
}

#[derive(Clone)]
struct Row {
    image: String,
    class: &'static str,
    width: u32,
    height: u32,
    mode: &'static str, // "pre" | "post"
    effort: u8,
    distance: f32,
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

fn row_header() -> &'static str {
    "image\tclass\twidth\theight\tmode\teffort\tdistance\tbytes\tbutteraugli\tssim2\tencode_ms\n"
}

fn row_tsv(r: &Row) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.2}\n",
        r.image,
        r.class,
        r.width,
        r.height,
        r.mode,
        r.effort,
        r.distance,
        r.bytes,
        r.butteraugli,
        r.ssim2,
        r.encode_ms,
    )
}

#[derive(Clone)]
struct Cell {
    image: String,
    class: &'static str,
    path: PathBuf,
    distance: f32,
    effort: u8,
}

fn enumerate_cells() -> Vec<Cell> {
    let corpus = corpus_dir();
    let cid_dir = corpus.join("CID22/CID22-512/validation");
    let sc_dir = corpus.join("gb82-sc");
    let mut cells = Vec::new();
    for name in SCREENSHOTS {
        let p = sc_dir.join(name);
        if !p.exists() {
            eprintln!("skipping missing screenshot {}", p.display());
            continue;
        }
        for &d in SCREENSHOT_DIST {
            for &e in EFFORTS {
                cells.push(Cell {
                    image: name.to_string(),
                    class: "screenshot",
                    path: p.clone(),
                    distance: d,
                    effort: e,
                });
            }
        }
    }
    for name in PHOTOS {
        let p = cid_dir.join(name);
        if !p.exists() {
            eprintln!("skipping missing photo {}", p.display());
            continue;
        }
        for &d in PHOTO_DIST {
            for &e in EFFORTS {
                cells.push(Cell {
                    image: name.to_string(),
                    class: "photo",
                    path: p.clone(),
                    distance: d,
                    effort: e,
                });
            }
        }
    }
    cells
}

fn encode_and_score(
    rgb: &[u8],
    w: u32,
    h: u32,
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    d: f32,
    e: u8,
) -> Option<(usize, f64, f64, f64)> {
    let cfg = LossyConfig::new(d).with_effort(e);
    // 8 GB cap so imac_g3 (2560x1664) at e9 doesn't trip the 2 GB
    // default mid-buttloop. Bench-only.
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    let t0 = Instant::now();
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&lim)
        .encode(rgb)
        .map_err(|err| eprintln!("encode failed: {err:?}"))
        .ok()?;
    let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (bfly, ssim2) = score(&bytes, orig_lin, orig_srgb, w, h)?;
    Some((bytes.len(), bfly, ssim2, enc_ms))
}

fn main() {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let utc_label = format!("{now_secs}");
    let out_tmp = PathBuf::from(format!("/tmp/buttloop_distance_split_port_{utc_label}.tsv"));
    let out_final = PathBuf::from(format!(
        "benchmarks/buttloop_distance_split_port_{utc_label}.tsv"
    ));
    let meta_final = out_final.with_extension("meta");

    let cells = enumerate_cells();
    eprintln!(
        "cells planned: {} (×2 modes = {} encodes)",
        cells.len(),
        cells.len() * 2
    );

    // Write tmp, then atomic-mv to repo at end (per failure-mode memory).
    let mut tmp = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&out_tmp)
        .expect("open tmp");
    tmp.write_all(row_header().as_bytes()).unwrap();

    let n_cells = cells.len();
    let mut rows: Vec<Row> = Vec::with_capacity(n_cells * 2);
    let mut failed: Vec<String> = Vec::new();

    for (idx, c) in cells.iter().enumerate() {
        let (rgb, w, h) = match load_png(&c.path) {
            Some(t) => t,
            None => {
                eprintln!("load failed: {}", c.path.display());
                failed.push(format!("load:{}", c.image));
                continue;
            }
        };
        let orig_lin = rgb_to_linear_img(&rgb, w, h);
        let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);

        for &mode in &["pre", "post"] {
            if mode == "pre" {
                set_mode_pre();
            } else {
                set_mode_post();
            }
            match encode_and_score(&rgb, w, h, &orig_lin, &orig_srgb, c.distance, c.effort) {
                Some((bytes, bfly, ssim2, enc_ms)) => {
                    let r = Row {
                        image: c.image.clone(),
                        class: c.class,
                        width: w,
                        height: h,
                        mode,
                        effort: c.effort,
                        distance: c.distance,
                        bytes,
                        butteraugli: bfly,
                        ssim2,
                        encode_ms: enc_ms,
                    };
                    tmp.write_all(row_tsv(&r).as_bytes()).unwrap();
                    tmp.flush().unwrap();
                    rows.push(r);
                }
                None => {
                    failed.push(format!(
                        "{}:{}:e{}:d{:.2}:{}",
                        c.class, c.image, c.effort, c.distance, mode
                    ));
                }
            }
        }

        if idx % 4 == 0 {
            // Refresh workongoing marker per CLAUDE.md.
            let _ = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!(
                    "date -u +%Y-%m-%dT%H:%M:%SZ | xargs -I {{}} printf '%s claude-buttloop-distance-split bench cell {}/{}\\n' {{}} > .workongoing",
                    idx + 1, n_cells
                ))
                .status();
            eprintln!(
                "[{}/{}] {} {} e{} d={:.2}",
                idx + 1,
                n_cells,
                c.class,
                c.image,
                c.effort,
                c.distance
            );
        }
    }

    // Reset overrides to production defaults before exit.
    reset_overrides();

    drop(tmp);

    // Atomic move of tmp → final, then write meta sidecar.
    std::fs::create_dir_all(out_final.parent().unwrap()).ok();
    std::fs::rename(&out_tmp, &out_final).expect("atomic mv tmp -> repo");

    // Print summary aggregates (paired pre/post deltas by class × distance × effort).
    let mut summary = String::new();
    summary.push_str("\n=== Paired aggregates (post - pre) ===\n");
    summary.push_str(&format!(
        "{:<12} {:<6} {:<6} {:<6} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}\n",
        "class",
        "effort",
        "dist",
        "n",
        "d_bytes%",
        "d_bfly%",
        "d_ssim2",
        "pre_bytes",
        "post_bytes",
        "pre_bfly",
    ));

    // (class, effort, distance_x100) keyed paired (PRE, POST) rows.
    type AggKey = (&'static str, u8, u32);
    type PairBucket = Vec<(Row, Row)>;
    let mut class_d_e: std::collections::BTreeMap<AggKey, PairBucket> =
        std::collections::BTreeMap::new();
    // Pair pre/post within (image, class, effort, distance).
    use std::collections::BTreeMap;
    let mut keyed: BTreeMap<(&'static str, String, u8, u32, &'static str), Row> = BTreeMap::new();
    for r in &rows {
        keyed.insert(
            (
                r.class,
                r.image.clone(),
                r.effort,
                (r.distance * 100.0) as u32,
                r.mode,
            ),
            r.clone(),
        );
    }
    for r_pre in rows.iter().filter(|r| r.mode == "pre") {
        let key = (
            r_pre.class,
            r_pre.image.clone(),
            r_pre.effort,
            (r_pre.distance * 100.0) as u32,
            "post",
        );
        if let Some(r_post) = keyed.get(&key) {
            class_d_e
                .entry((r_pre.class, r_pre.effort, (r_pre.distance * 100.0) as u32))
                .or_default()
                .push((r_pre.clone(), r_post.clone()));
        }
    }
    for ((class, effort, dx100), pairs) in &class_d_e {
        let n = pairs.len();
        let mut sum_db = 0.0f64;
        let mut sum_dbf = 0.0f64;
        let mut sum_ds = 0.0f64;
        let mut sum_pre_b = 0.0f64;
        let mut sum_post_b = 0.0f64;
        let mut sum_pre_bf = 0.0f64;
        for (pre, post) in pairs {
            let db = (post.bytes as f64 - pre.bytes as f64) / pre.bytes as f64 * 100.0;
            let dbf = (post.butteraugli - pre.butteraugli) / pre.butteraugli * 100.0;
            let ds = post.ssim2 - pre.ssim2;
            sum_db += db;
            sum_dbf += dbf;
            sum_ds += ds;
            sum_pre_b += pre.bytes as f64;
            sum_post_b += post.bytes as f64;
            sum_pre_bf += pre.butteraugli;
        }
        let d = *dx100 as f32 / 100.0;
        summary.push_str(&format!(
            "{:<12} {:<6} {:<6.2} {:<6} {:>+8.2} {:>+8.2} {:>+8.3} {:>10.0} {:>10.0} {:>10.4}\n",
            class,
            effort,
            d,
            n,
            sum_db / n as f64,
            sum_dbf / n as f64,
            sum_ds / n as f64,
            sum_pre_b / n as f64,
            sum_post_b / n as f64,
            sum_pre_bf / n as f64,
        ));
    }
    if !failed.is_empty() {
        summary.push_str(&format!("\nFAILED ({} cells):\n", failed.len()));
        for f in &failed {
            summary.push_str(&format!("  {f}\n"));
        }
    }

    eprintln!("{}", summary);

    // Write .meta sidecar with provenance.
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());
    let mut meta = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&meta_final)
        .expect("meta open");
    writeln!(meta, "# W38-2 #3.1 distance-aware buttloop split A/B").unwrap();
    writeln!(meta, "# Generated: {utc_label}").unwrap();
    writeln!(meta, "# Host: {host}").unwrap();
    writeln!(
        meta,
        "# CPU port of GPU commit d75bf7c (memory `buttloop_rd_gap_2026-05-14.md`)"
    )
    .unwrap();
    writeln!(
        meta,
        "# Cells: 3 screenshots × {{d=0.5,1.0,1.5,2.0,3.0,4.0,5.0}} × {{e8,e9}}"
    )
    .unwrap();
    writeln!(
        meta,
        "#      + 3 photos      × {{d=0.5,1.0,1.5,3.0,4.0,5.0}}     × {{e8,e9}}"
    )
    .unwrap();
    writeln!(
        meta,
        "# PRE mode = libjxl defaults at all distances (cur_pow=0.2, no cap) — matches production CPU defaults"
    )
    .unwrap();
    writeln!(
        meta,
        "# POST mode = literal GPU port (LOW d<2.0 → cur_pow=0.5, max_increase=1.3; HIGH d>=2.0 → libjxl defaults)"
    )
    .unwrap();
    writeln!(meta, "# Decoder: jxl-oxide srgb_linear (metadata-immune).").unwrap();
    writeln!(
        meta,
        "# Metric: Rust butteraugli_linear + fast_ssim2::compute_ssimulacra2."
    )
    .unwrap();
    writeln!(meta, "# Total rows: {}", rows.len()).unwrap();
    writeln!(meta, "# Failed cells: {}", failed.len()).unwrap();
    writeln!(meta, "#").unwrap();
    write!(meta, "{summary}").unwrap();
    drop(meta);

    eprintln!("wrote {}", out_final.display());
    eprintln!("wrote {}", meta_final.display());
}
