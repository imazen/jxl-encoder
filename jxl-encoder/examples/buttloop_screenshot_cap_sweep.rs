//! W39-2 (WF3 fix): screenshot-class HIGH-regime `max_increase` cap
//! sweep for the butteraugli quantization loop.
//!
//! **Goal**: identify a `max_increase` cap that fixes the W38-2 audit
//! WF3 wedge (e8/e9 buttloop over-compresses screenshots at d≥2.0:
//! bfly +9-19 %, ssim2 -2 to -5 vs cjxl) without regressing photos.
//!
//! **Setup**:
//! - Override `MAX_INCREASE_X1000_HIGH_SCREENSHOT` to one of
//!   `{1.3, 1.5, 1.8, 2.0, ∞=100.0}` per cap-arm.
//! - Three screenshots × {d=2.0, 3.0, 4.0, 5.0} × {e8, e9} × 5 caps = 120 cells.
//! - Three photos      × same dist × eff × 5 caps                   =  72 cells.
//!   (Photos shouldn't fire the gate — verified separately.)
//!
//! The screenshot gate is computed inside the encoder using
//! `median(mask1x1) > SCREENSHOT_MEDIAN_THRESHOLD (=95.0)`. The
//! same classifier is already used by `splines::looks_like_screenshot`
//! and the W22-1 entropy_mul content-aware dispatch.
//!
//! **Decision criteria** for shipping a cap as default:
//!   - Screen class at d≥2.0 e8/e9: bfly Δ ≤ -3 % AND ssim2 Δ ≥ +1
//!     AND bytes Δ within ±2 % vs the no-cap baseline.
//!   - Photo class: bit-identical (gate doesn't fire). Verify.
//!
//! **Metrics**: jxl-oxide `srgb_linear` decode + Rust `butteraugli_linear`
//! + `fast_ssim2::compute_ssimulacra2` — metadata-immune (no
//!   `butteraugli_main`).
//!
//! **Output**:
//!   benchmarks/buttloop_screenshot_cap_sweep_<UTC>.tsv  (per-cell)
//!   benchmarks/buttloop_screenshot_cap_sweep_<UTC>.meta (provenance)
//!
//! **Reproducer**:
//!   cargo run -p jxl-encoder --release --features 'std parallel butteraugli-loop' \
//!     --example buttloop_screenshot_cap_sweep

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

// Atomic overrides — see vardct/mod.rs::__buttloop_overrides.
use jxl_encoder::vardct::__buttloop_overrides::{
    MAX_INCREASE_X1000_HIGH, MAX_INCREASE_X1000_HIGH_SCREENSHOT, MAX_INCREASE_X1000_LOW,
};

const SCREENSHOTS: &[&str] = &["terminal.png", "codec_wiki.png", "imac_g3.png"];
const PHOTOS: &[&str] = &["1025469.png", "1418519.png", "1531677.png"];

// HIGH regime only (d>=2.0). This is the WF3 wedge.
const DISTANCES: &[f32] = &[2.0, 3.0, 4.0, 5.0];
const EFFORTS: &[u8] = &[8, 9];

/// Cap values to sweep on the screenshot HIGH slot. `100.0` is the
/// libjxl "no cap" baseline (matches `DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT`
/// and `DEFAULT_MAX_INCREASE_HIGH` — the pre-W39-2 behaviour).
const CAPS: &[f64] = &[1.3, 1.5, 1.8, 2.0, 100.0];

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

/// Reset all overrides to production defaults.
fn reset_overrides() {
    MAX_INCREASE_X1000_LOW.store(i32::MIN, Ordering::Relaxed);
    MAX_INCREASE_X1000_HIGH.store(i32::MIN, Ordering::Relaxed);
    MAX_INCREASE_X1000_HIGH_SCREENSHOT.store(i32::MIN, Ordering::Relaxed);
}

/// Override the screenshot HIGH cap to `cap`. `100.0` (no cap) is the
/// baseline arm — encoded directly so the override path is identical
/// across all arms (rules out "override-present vs override-absent"
/// confounders).
fn set_screenshot_cap(cap: f64) {
    let x1000 = (cap * 1000.0).round() as i32;
    MAX_INCREASE_X1000_HIGH_SCREENSHOT.store(x1000, Ordering::Relaxed);
}

#[derive(Clone)]
struct Row {
    image: String,
    class: &'static str,
    width: u32,
    height: u32,
    cap: f64,
    effort: u8,
    distance: f32,
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

fn row_header() -> &'static str {
    "image\tclass\twidth\theight\tcap\teffort\tdistance\tbytes\tbutteraugli\tssim2\tencode_ms\n"
}

fn row_tsv(r: &Row) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.2}\n",
        r.image,
        r.class,
        r.width,
        r.height,
        r.cap,
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
        for &d in DISTANCES {
            for &e in EFFORTS {
                cells.push(Cell {
                    image: (*name).to_string(),
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
        for &d in DISTANCES {
            for &e in EFFORTS {
                cells.push(Cell {
                    image: (*name).to_string(),
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
    let out_tmp = PathBuf::from(format!(
        "/tmp/buttloop_screenshot_cap_sweep_{utc_label}.tsv"
    ));
    let out_final = PathBuf::from(format!(
        "benchmarks/buttloop_screenshot_cap_sweep_{utc_label}.tsv"
    ));
    let meta_final = out_final.with_extension("meta");

    let cells = enumerate_cells();
    eprintln!(
        "cells planned: {} (×{} caps = {} encodes)",
        cells.len(),
        CAPS.len(),
        cells.len() * CAPS.len()
    );

    // Write tmp, then atomic-mv to repo at end.
    let mut tmp = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&out_tmp)
        .expect("open tmp");
    tmp.write_all(row_header().as_bytes()).unwrap();

    let n_cells = cells.len();
    let mut rows: Vec<Row> = Vec::with_capacity(n_cells * CAPS.len());
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

        for &cap in CAPS {
            set_screenshot_cap(cap);
            match encode_and_score(&rgb, w, h, &orig_lin, &orig_srgb, c.distance, c.effort) {
                Some((bytes, bfly, ssim2, enc_ms)) => {
                    let r = Row {
                        image: c.image.clone(),
                        class: c.class,
                        width: w,
                        height: h,
                        cap,
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
                        "{}:{}:e{}:d{:.2}:cap{:.2}",
                        c.class, c.image, c.effort, c.distance, cap
                    ));
                }
            }
        }

        if idx % 4 == 0 {
            // Refresh workongoing marker (CLAUDE.md ≤2 min cadence).
            let _ = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!(
                    "date -u +%Y-%m-%dT%H:%M:%SZ | xargs -I {{}} printf '%s claude-screenshot-buttloop-cap bench cell {}/{}\\n' {{}} > .workongoing",
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

    // Reset overrides before exit so a re-invocation starts clean.
    reset_overrides();
    drop(tmp);

    // Atomic mv tmp → repo.
    std::fs::create_dir_all(out_final.parent().unwrap()).ok();
    std::fs::rename(&out_tmp, &out_final).expect("atomic mv tmp -> repo");

    // ===== Summary aggregates =====
    let mut summary = String::new();
    summary.push_str("\n=== Per-cap aggregates (vs cap=100.0 baseline) ===\n");
    summary.push_str(&format!(
        "{:<12} {:<6} {:<6} {:<6} {:<6} {:>8} {:>8} {:>8} {:>10} {:>10}\n",
        "class",
        "effort",
        "dist",
        "cap",
        "n",
        "d_bytes%",
        "d_bfly%",
        "d_ssim2",
        "base_bytes",
        "base_bfly",
    ));

    // Index rows by (image, class, effort, distance, cap).
    use std::collections::BTreeMap;
    type Key = (
        &'static str, // class
        String,       // image
        u8,           // effort
        u32,          // distance_x100
        u32,          // cap_x100
    );
    let mut keyed: BTreeMap<Key, Row> = BTreeMap::new();
    for r in &rows {
        keyed.insert(
            (
                r.class,
                r.image.clone(),
                r.effort,
                (r.distance * 100.0) as u32,
                (r.cap * 100.0) as u32,
            ),
            r.clone(),
        );
    }

    // Aggregate by (class, effort, distance, cap) vs baseline cap=100.0.
    type AggKey = (&'static str, u8, u32, u32);
    let mut agg: BTreeMap<AggKey, (u32, f64, f64, f64, f64, f64)> = BTreeMap::new();
    let baseline_cap_x100 = (100.0_f64 * 100.0) as u32;
    for r in &rows {
        let baseline_key: Key = (
            r.class,
            r.image.clone(),
            r.effort,
            (r.distance * 100.0) as u32,
            baseline_cap_x100,
        );
        let Some(base) = keyed.get(&baseline_key) else {
            continue;
        };
        let db = (r.bytes as f64 - base.bytes as f64) / base.bytes as f64 * 100.0;
        let dbf = (r.butteraugli - base.butteraugli) / base.butteraugli * 100.0;
        let ds = r.ssim2 - base.ssim2;
        let key: AggKey = (
            r.class,
            r.effort,
            (r.distance * 100.0) as u32,
            (r.cap * 100.0) as u32,
        );
        let e = agg.entry(key).or_insert((0, 0.0, 0.0, 0.0, 0.0, 0.0));
        e.0 += 1;
        e.1 += db;
        e.2 += dbf;
        e.3 += ds;
        e.4 += base.bytes as f64;
        e.5 += base.butteraugli;
    }
    for ((class, effort, dx100, cx100), (n, sb, sbf, sds, base_b, base_bf)) in &agg {
        let d = *dx100 as f32 / 100.0;
        let cap = *cx100 as f32 / 100.0;
        summary.push_str(&format!(
            "{:<12} {:<6} {:<6.2} {:<6.2} {:<6} {:>+8.2} {:>+8.2} {:>+8.3} {:>10.0} {:>10.4}\n",
            class,
            effort,
            d,
            cap,
            n,
            sb / *n as f64,
            sbf / *n as f64,
            sds / *n as f64,
            base_b / *n as f64,
            base_bf / *n as f64,
        ));
    }

    // Photo bit-identity verification: cap arms should byte-match the
    // baseline cap=100.0 arm on photo class (the screenshot gate
    // doesn't fire on photo input). Any drift here is a bug.
    summary.push_str("\n=== Photo-class bit-identity check (vs cap=100.0) ===\n");
    let mut photo_drift = 0usize;
    let mut photo_checked = 0usize;
    for r in rows.iter().filter(|r| r.class == "photo") {
        let baseline_key: Key = (
            r.class,
            r.image.clone(),
            r.effort,
            (r.distance * 100.0) as u32,
            baseline_cap_x100,
        );
        let Some(base) = keyed.get(&baseline_key) else {
            continue;
        };
        photo_checked += 1;
        if r.bytes != base.bytes {
            photo_drift += 1;
            summary.push_str(&format!(
                "PHOTO DRIFT: {} e{} d={:.2} cap={:.2}: bytes={} vs base={}\n",
                r.image, r.effort, r.distance, r.cap, r.bytes, base.bytes
            ));
        }
    }
    summary.push_str(&format!(
        "photo cells checked: {photo_checked} drift: {photo_drift}\n"
    ));

    if !failed.is_empty() {
        summary.push_str(&format!("\nFAILED ({} cells):\n", failed.len()));
        for f in &failed {
            summary.push_str(&format!("  {f}\n"));
        }
    }

    eprintln!("{summary}");

    // Write .meta sidecar.
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());
    let mut meta = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&meta_final)
        .expect("meta open");
    writeln!(
        meta,
        "# W39-2 (WF3 fix): screenshot-class HIGH-regime max_increase cap sweep"
    )
    .unwrap();
    writeln!(meta, "# Generated: {utc_label}").unwrap();
    writeln!(meta, "# Host: {host}").unwrap();
    writeln!(
        meta,
        "# Follow-on to W39-1 (3ecd397b) — the scaffolding shipped the atomic"
    )
    .unwrap();
    writeln!(
        meta,
        "# infrastructure; this chunk wires content-class dispatch on top."
    )
    .unwrap();
    writeln!(
        meta,
        "# Cells: 3 screenshots × {{d=2.0,3.0,4.0,5.0}} × {{e8,e9}}"
    )
    .unwrap();
    writeln!(
        meta,
        "#      + 3 photos      × {{d=2.0,3.0,4.0,5.0}} × {{e8,e9}}"
    )
    .unwrap();
    writeln!(
        meta,
        "# Caps swept: {CAPS:?} (100.0 = libjxl no-cap baseline)"
    )
    .unwrap();
    writeln!(
        meta,
        "# Classifier: median(mask1x1) > 95.0 (matches splines::looks_like_screenshot"
    )
    .unwrap();
    writeln!(
        meta,
        "#             and encoder.rs::CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD)"
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
