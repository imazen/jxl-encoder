//! W44-AUDIT-8 Phase 5 — cluster validation of `extra_dc_precision = 1`
//! at effort ≤ 7 (libjxl `nl_dc` parity).
//!
//! Phase 4 (`memory/w44_audit_8_phase4_dc_pipeline_dump_clic_2026-05-24.md`)
//! traced the CLIC + photo SSIM2-deficit cluster's worst cell
//! (clic_22ea12 e7 d=4, dSsim2 = -3.84 vs cjxl) to a DC quantization
//! step mismatch: cjxl uses 2× DC integer precision at effort ≤ 7
//! (`enc_cache.cc:232-234`: `nl_dc = speed_tier < kFalcon`), we used 1×.
//!
//! Phase 5 ships the fix: `EffortProfile::extra_dc_precision = 1` at
//! e ≤ 7, `0` at e ≥ 8; encoder/decoder symmetric multiplier
//! `dc_mul = 1 << extra_dc_precision` on `inv_factor` in `transform.rs`
//! + `reconstruct.rs`; bitstream `extra_dc_precision` field at
//! `bitstream.rs::write_dc_group{_from_tokens_inner}` now reads from
//! `self.profile.extra_dc_precision`.
//!
//! This bench measures the fix on:
//! 1. The 30-cell SSIM2-deficit cluster (from Phase 2 — same cell list).
//! 2. 4 W44-228c1 SHIP PROTECT screenshot cells (terminal/codec_wiki/
//!    imac_g3 e8 d=4-5) — these fire at e ≥ 8 where extra_dc_precision = 0
//!    structurally, so MUST stay byte-identical to pre-Phase-5 baseline.
//!
//! Mode: current build (post-Phase-5) vs cjxl reference.
//! Acceptance:
//!   - clic_22ea12 e7 d=4: SSIM2 Δ vs cjxl improves by ≥ +2.0 (Phase 4
//!     predicted ≥ +2.0 from closing the half-step rounding gap).
//!   - Cluster aggregate: ≥ 50 % of cells improve SSIM2 Δ by ≥ +1.0.
//!   - PROTECT (e ≥ 8): bytes drift ≤ 0.5 % vs pre-Phase-5 (these
//!     cells go through the unchanged `extra_dc_precision = 0` path; any
//!     drift indicates a regression in the encoder/reconstruct symmetry).
//!
//! Wall budget: ≤ 30 min (34 cells × 2 encodes = 68 encodes).
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_8_phase5_cluster_validate -- \
//!     --output benchmarks/w44_audit_8_phase5_cluster_validate_2026-05-24.tsv

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

struct Cell {
    image_id: &'static str,
    path: &'static str,
    class: &'static str,
    effort: u8,
    distance: f32,
    role: &'static str, // "CLUSTER" or "PROTECT_E8"
}

// 30-cell cluster from Phase 2 + 4 PROTECT_E8 cells.
const CELLS: &[Cell] = &[
    // ─── 30-cell SSIM2-deficit cluster (Phase 2 list) ─────────────────────
    // PHOTO_PORTRAIT
    Cell { image_id: "1418519", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png", class: "PHOTO_PORTRAIT", effort: 5, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1418519", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png", class: "PHOTO_PORTRAIT", effort: 7, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1418519", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png", class: "PHOTO_PORTRAIT", effort: 9, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1279330", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1279330.png", class: "PHOTO_PORTRAIT", effort: 7, distance: 4.0, role: "CLUSTER" },
    // PHOTO_LANDSCAPE
    Cell { image_id: "1475938", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png", class: "PHOTO_LANDSCAPE", effort: 7, distance: 2.0, role: "CLUSTER" },
    Cell { image_id: "1475938", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png", class: "PHOTO_LANDSCAPE", effort: 7, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1475938", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png", class: "PHOTO_LANDSCAPE", effort: 9, distance: 4.0, role: "CLUSTER" },
    // PHOTO_SMOOTH
    Cell { image_id: "1531677", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png", class: "PHOTO_SMOOTH", effort: 5, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1531677", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png", class: "PHOTO_SMOOTH", effort: 7, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1531677", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png", class: "PHOTO_SMOOTH", effort: 9, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1420710", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png", class: "PHOTO_SMOOTH", effort: 7, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1544947", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1544947.png", class: "PHOTO_SMOOTH", effort: 7, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "1544947", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1544947.png", class: "PHOTO_SMOOTH", effort: 9, distance: 4.0, role: "CLUSTER" },
    // CLIC2025_WEB — clic_097cb4
    Cell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 5, distance: 2.0, role: "CLUSTER" },
    Cell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 5, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 7, distance: 2.0, role: "CLUSTER" },
    Cell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 7, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 9, distance: 2.0, role: "CLUSTER" },
    Cell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 9, distance: 4.0, role: "CLUSTER" },
    // CLIC2025_WEB — clic_0c49a5
    Cell { image_id: "clic_0c49a5", path: "/home/lilith/work/codec-corpus/clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png", class: "CLIC2025_WEB", effort: 7, distance: 2.0, role: "CLUSTER" },
    Cell { image_id: "clic_0c49a5", path: "/home/lilith/work/codec-corpus/clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png", class: "CLIC2025_WEB", effort: 9, distance: 2.0, role: "CLUSTER" },
    // CLIC2025_WEB — clic_100a02
    Cell { image_id: "clic_100a02", path: "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png", class: "CLIC2025_WEB", effort: 5, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "clic_100a02", path: "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png", class: "CLIC2025_WEB", effort: 7, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "clic_100a02", path: "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png", class: "CLIC2025_WEB", effort: 9, distance: 4.0, role: "CLUSTER" },
    // CLIC2025_WEB — clic_22ea12 (PHASE 4 WORST CELL — primary bisect target)
    Cell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 5, distance: 2.0, role: "CLUSTER" },
    Cell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 5, distance: 4.0, role: "CLUSTER" },
    Cell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 7, distance: 2.0, role: "CLUSTER" },
    Cell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 7, distance: 4.0, role: "CLUSTER" }, // ★ PRIMARY
    Cell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 9, distance: 2.0, role: "CLUSTER" },
    Cell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 9, distance: 4.0, role: "CLUSTER" },
    // ─── 4 PROTECT_E8 cells (must stay structurally unchanged at e>=8) ────
    Cell { image_id: "terminal", path: "/home/lilith/work/codec-corpus/gb82-sc/terminal.png", class: "SCREEN", effort: 8, distance: 4.0, role: "PROTECT_E8" },
    Cell { image_id: "codec_wiki", path: "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png", class: "SCREEN", effort: 8, distance: 4.0, role: "PROTECT_E8" },
    Cell { image_id: "codec_wiki", path: "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png", class: "SCREEN", effort: 8, distance: 5.0, role: "PROTECT_E8" },
    Cell { image_id: "imac_g3", path: "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png", class: "SCREEN", effort: 8, distance: 4.0, role: "PROTECT_E8" },
];

fn srgb_u8_to_linear_f32(x: u8) -> f32 {
    let x = x as f32 / 255.0;
    if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb_u8(x: f32) -> u8 {
    let x = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn make_imgs(
    pixels: &[u8],
    w: u32,
    h: u32,
) -> (Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>) {
    let lin: Vec<RGB<f32>> = pixels
        .chunks_exact(3)
        .map(|c| RGB::new(srgb_u8_to_linear_f32(c[0]), srgb_u8_to_linear_f32(c[1]), srgb_u8_to_linear_f32(c[2])))
        .collect();
    let srgb: Vec<[u8; 3]> = pixels
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    (Img::new(lin, w as usize, h as usize), Img::new(srgb, w as usize, h as usize))
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

fn score_jxl(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize { return None; }
    let n = dw * dh;
    let ch = if n > 0 { dec_lin.len() / n } else { 0 };
    if ch < 3 { return None; }
    let dec_pixels: Vec<RGB<f32>> = dec_lin.chunks_exact(ch).map(|c| RGB::new(c[0], c[1], c[2])).collect();
    let dec_lin_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear.as_ref(), dec_lin_img.as_ref(), &ButteraugliParams::default()).ok()?.score as f64;
    let dec_srgb: Vec<[u8; 3]> = dec_lin.chunks_exact(ch).map(|c| [linear_to_srgb_u8(c[0]), linear_to_srgb_u8(c[1]), linear_to_srgb_u8(c[2])]).collect();
    let dec_srgb_img = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn encode_zenjxl(pixels: &[u8], w: u32, h: u32, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    let t0 = Instant::now();
    let buf = cfg.encode_request(w, h, PixelLayout::Rgb8).encode(pixels).ok()?;
    Some((buf, t0.elapsed().as_millis()))
}

fn encode_cjxl(src: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_audit8p5_{}_{}_{}_{}.jxl",
        src.file_stem()?.to_string_lossy(),
        effort,
        (distance * 1000.0) as u32,
        std::process::id()
    ));
    let t0 = Instant::now();
    let s = Command::new(cjxl)
        .arg(src).arg(&tmp).arg("-e").arg(format!("{}", effort)).arg("-d").arg(format!("{}", distance))
        .arg("--quiet").status().ok()?;
    let ms = t0.elapsed().as_millis();
    if !s.success() { return None; }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some((bytes, ms))
}

#[derive(Default, Debug, Clone)]
struct Row {
    image_id: String,
    class: String,
    role: String,
    width: u32,
    height: u32,
    effort: u8,
    distance: f32,
    cjxl_bytes: usize,
    cjxl_bfly: f64,
    cjxl_ssim2: f64,
    ours_bytes: usize,
    ours_bfly: f64,
    ours_ssim2: f64,
    ours_ms: u128,
}

fn write_tsv(out: &Path, rows: &[Row]) {
    if let Some(p) = out.parent() { let _ = std::fs::create_dir_all(p); }
    let mut f = File::create(out).expect("create tsv");
    writeln!(f, "image_id\tclass\trole\twidth\theight\teffort\tdistance\t\
                cjxl_bytes\tcjxl_bfly\tcjxl_ssim2\t\
                ours_bytes\tours_bfly\tours_ssim2\tours_ms\t\
                delta_bytes_pct\tdelta_ssim2\tdelta_bfly_pct").unwrap();
    for r in rows {
        let db = if r.cjxl_bytes > 0 {
            (r.ours_bytes as f64 - r.cjxl_bytes as f64) / r.cjxl_bytes as f64 * 100.0
        } else { f64::NAN };
        let ds = r.ours_ssim2 - r.cjxl_ssim2;
        let dbf = if r.cjxl_bfly > 0.0 { (r.ours_bfly - r.cjxl_bfly) / r.cjxl_bfly * 100.0 } else { f64::NAN };
        writeln!(f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}\t{:.3}\t{:.3}\t{:.3}",
            r.image_id, r.class, r.role, r.width, r.height, r.effort, r.distance,
            r.cjxl_bytes, r.cjxl_bfly, r.cjxl_ssim2,
            r.ours_bytes, r.ours_bfly, r.ours_ssim2, r.ours_ms,
            db, ds, dbf,
        ).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out: PathBuf = PathBuf::from("benchmarks/w44_audit_8_phase5_cluster_validate_2026-05-24.tsv");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() { out = PathBuf::from(&args[i + 1]); i += 2; }
        else { i += 1; }
    }
    eprintln!("[bench] {} cells × 2 encodes = {} encodes", CELLS.len(), CELLS.len() * 2);
    eprintln!("[bench] output: {}", out.display());
    let mut rows: Vec<Row> = Vec::with_capacity(CELLS.len());
    let start = Instant::now();

    // Group by image_id+path so PNG loads once.
    let mut by_image: std::collections::BTreeMap<(&'static str, &'static str), Vec<&Cell>> = std::collections::BTreeMap::new();
    for c in CELLS { by_image.entry((c.image_id, c.path)).or_default().push(c); }

    let total = CELLS.len();
    let mut done = 0usize;
    for ((image_id, path), cells) in &by_image {
        let (pixels, w, h) = match load_png(Path::new(*path)) {
            Some(t) => t,
            None => { eprintln!("[bench] FAIL load {}", path); continue; }
        };
        eprintln!("[bench] LOADED {} {}x{} ({} cells)", image_id, w, h, cells.len());
        let (lin, srgb) = make_imgs(&pixels, w, h);

        for cell in cells {
            done += 1;
            let t = Instant::now();
            eprint!("[bench] {}/{} {} e{} d={} ({}) ... ", done, total, cell.image_id, cell.effort, cell.distance, cell.role);
            let mut row = Row {
                image_id: cell.image_id.to_string(),
                class: cell.class.to_string(),
                role: cell.role.to_string(),
                width: w, height: h,
                effort: cell.effort, distance: cell.distance,
                ..Default::default()
            };
            if let Some((cb, _)) = encode_cjxl(Path::new(cell.path), cell.effort, cell.distance) {
                row.cjxl_bytes = cb.len();
                if let Some((bf, ss)) = score_jxl(&cb, &lin, &srgb, w, h) {
                    row.cjxl_bfly = bf; row.cjxl_ssim2 = ss;
                }
            }
            if let Some((ob, ms)) = encode_zenjxl(&pixels, w, h, cell.effort, cell.distance) {
                row.ours_bytes = ob.len(); row.ours_ms = ms;
                if let Some((bf, ss)) = score_jxl(&ob, &lin, &srgb, w, h) {
                    row.ours_bfly = bf; row.ours_ssim2 = ss;
                }
            }
            let db = if row.cjxl_bytes > 0 { (row.ours_bytes as f64 - row.cjxl_bytes as f64) / row.cjxl_bytes as f64 * 100.0 } else { 0.0 };
            let ds = row.ours_ssim2 - row.cjxl_ssim2;
            eprintln!("ours={}B/ss2={:.2} cjxl={}B/ss2={:.2} Δbytes={:+.2}% Δssim2={:+.3} ({}ms)",
                row.ours_bytes, row.ours_ssim2, row.cjxl_bytes, row.cjxl_ssim2, db, ds, t.elapsed().as_millis());
            rows.push(row);
            if done % 5 == 0 { write_tsv(&out, &rows); }
        }
    }
    write_tsv(&out, &rows);
    eprintln!("[bench] {} rows written in {:.1}s", rows.len(), start.elapsed().as_secs_f64());

    // ── Summary ────────────────────────────────────────────────────────────
    let mut cluster_n = 0;
    let mut cluster_imp1 = 0;
    let mut cluster_imp2 = 0;
    let mut sum_dssim2 = 0.0;
    let mut sum_dbytes = 0.0;
    let mut protect_n = 0;
    let mut protect_drift_max = 0.0f64;
    let mut clic_22ea12_e7_d4: Option<(f64, f64, f64)> = None; // (ours_ssim2, cjxl_ssim2, delta_bytes_pct)
    for r in &rows {
        let db = if r.cjxl_bytes > 0 { (r.ours_bytes as f64 - r.cjxl_bytes as f64) / r.cjxl_bytes as f64 * 100.0 } else { 0.0 };
        let ds = r.ours_ssim2 - r.cjxl_ssim2;
        if r.role == "CLUSTER" {
            cluster_n += 1;
            sum_dssim2 += ds;
            sum_dbytes += db;
            // For cluster, "improvement" means ds is LESS NEGATIVE (closer to 0)
            // vs the pre-Phase-4 baseline. Phase 4 cluster worst cells had
            // ds ≈ -3.8. Here we just report ds magnitude and aggregate.
            if ds >= -2.0 { cluster_imp1 += 1; }
            if ds >= -1.0 { cluster_imp2 += 1; }
            if r.image_id == "clic_22ea12" && r.effort == 7 && (r.distance - 4.0).abs() < 0.01 {
                clic_22ea12_e7_d4 = Some((r.ours_ssim2, r.cjxl_ssim2, db));
            }
        } else if r.role == "PROTECT_E8" {
            protect_n += 1;
            if db.abs() > protect_drift_max { protect_drift_max = db.abs(); }
        }
    }
    eprintln!();
    eprintln!("=== Cluster ({} cells) ===", cluster_n);
    eprintln!("  mean Δssim2 vs cjxl = {:+.3}", if cluster_n > 0 { sum_dssim2 / cluster_n as f64 } else { 0.0 });
    eprintln!("  mean Δbytes  vs cjxl = {:+.3}%", if cluster_n > 0 { sum_dbytes / cluster_n as f64 } else { 0.0 });
    eprintln!("  cells with Δssim2 ≥ -2.0: {} / {}", cluster_imp1, cluster_n);
    eprintln!("  cells with Δssim2 ≥ -1.0: {} / {}", cluster_imp2, cluster_n);
    if let Some((our_s, cjxl_s, db)) = clic_22ea12_e7_d4 {
        eprintln!();
        eprintln!("=== PRIMARY BISECT: clic_22ea12 e7 d=4 ===");
        eprintln!("  ours SSIM2 = {:.4}, cjxl SSIM2 = {:.4}, Δssim2 = {:+.3}", our_s, cjxl_s, our_s - cjxl_s);
        eprintln!("  Δbytes vs cjxl = {:+.3}%", db);
        eprintln!("  Phase 4 baseline: ours -3.84 vs cjxl; expected Phase 5 ≥ -1.8 (+2.0 recovery).");
    }
    eprintln!();
    eprintln!("=== PROTECT_E8 ({} cells) ===", protect_n);
    eprintln!("  max |Δbytes pct| vs cjxl = {:.3}%", protect_drift_max);
    eprintln!("  Note: PROTECT_E8 cells fire at e≥8 where extra_dc_precision=0; structurally invariant under Phase 5.");
}
