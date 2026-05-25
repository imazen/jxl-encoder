//! W44-AUDIT-8 Phase 2 — Mode C bench on the 30-cell photo SSIM2-deficit cluster.
//!
//! Tests whether enabling CfL `cfl_newton_libjxl_math_with_ls_warm_start = true`
//! (Mode C from AUDIT-5 P2) closes the SSIM2 wedge observed by AUDIT-7 wider-
//! corpus across the 30-cell PHOTO/CLIC cluster (dSsim2 ≤ -1.0 vs cjxl).
//!
//! Mode A: `EncoderStrategy::Zenjxl` default (cfl_newton_libjxl_math_with_ls_warm_start = false).
//! Mode C: `EncoderStrategy::Custom(EncoderImprovementsCustom::default())` with the
//!   `cfl_newton_libjxl_math_with_ls_warm_start` field flipped to `true`.
//!   Default = Zenjxl preset; flipping one field is the cleanest single-
//!   process A/B without env var contamination.
//!
//! Cell list extracted from `benchmarks/w44_audit_8_phase1_cluster_chars_2026-05-24.tsv`:
//! 30 (image, effort, distance) tuples spanning 13 unique images.
//!
//! Wall budget: ≤ 30 min (30 cells × 2 modes × 1 cjxl run = 90 encodes).
//!
//! Acceptance per Phase 2 Step 3 classification:
//!   - WIN   : Mode C ssim2 ≥ Mode A + 0.5 AND |Δbytes| ≤ 2%
//!   - LOSS  : Mode C ssim2 ≤ Mode A - 0.5 OR Δbytes > 2%
//!   - NEUTRAL: otherwise (|Δssim2| < 0.5 AND |Δbytes| < 0.5%)
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_8_phase2_mode_c_on_cluster -- \
//!     --output benchmarks/w44_audit_8_phase2_mode_c_on_cluster_2026-05-24.tsv

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{
    EncoderImprovementsCustom, EncoderStrategy, LossyConfig, PixelLayout,
};
use rgb::RGB;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

// ── 30-cell cluster: (image_id, path, class, effort, distance, subclass) ────
//
// Extracted from benchmarks/w44_audit_8_phase1_cluster_chars_2026-05-24.tsv.
struct ClusterCell {
    image_id: &'static str,
    path: &'static str,
    class: &'static str,
    effort: u8,
    distance: f32,
    subclass: &'static str,
}

const CLUSTER: &[ClusterCell] = &[
    // PHOTO_PORTRAIT
    ClusterCell { image_id: "1418519", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png", class: "PHOTO_PORTRAIT", effort: 5, distance: 4.0, subclass: "A_LibjxlBetter" },
    ClusterCell { image_id: "1418519", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png", class: "PHOTO_PORTRAIT", effort: 7, distance: 4.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "1418519", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png", class: "PHOTO_PORTRAIT", effort: 9, distance: 4.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "1279330", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1279330.png", class: "PHOTO_PORTRAIT", effort: 7, distance: 4.0, subclass: "B_BothFail" },
    // PHOTO_LANDSCAPE
    ClusterCell { image_id: "1475938", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png", class: "PHOTO_LANDSCAPE", effort: 7, distance: 2.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "1475938", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png", class: "PHOTO_LANDSCAPE", effort: 7, distance: 4.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "1475938", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png", class: "PHOTO_LANDSCAPE", effort: 9, distance: 4.0, subclass: "B_BothFail" },
    // PHOTO_SMOOTH
    ClusterCell { image_id: "1531677", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png", class: "PHOTO_SMOOTH", effort: 5, distance: 4.0, subclass: "A_LibjxlBetter" },
    ClusterCell { image_id: "1531677", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png", class: "PHOTO_SMOOTH", effort: 7, distance: 4.0, subclass: "C_LibjxlWORSE" },
    ClusterCell { image_id: "1531677", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png", class: "PHOTO_SMOOTH", effort: 9, distance: 4.0, subclass: "C_LibjxlWORSE" },
    ClusterCell { image_id: "1420710", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png", class: "PHOTO_SMOOTH", effort: 7, distance: 4.0, subclass: "C_LibjxlWORSE" },
    ClusterCell { image_id: "1544947", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1544947.png", class: "PHOTO_SMOOTH", effort: 7, distance: 4.0, subclass: "C_LibjxlWORSE" },
    ClusterCell { image_id: "1544947", path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1544947.png", class: "PHOTO_SMOOTH", effort: 9, distance: 4.0, subclass: "C_LibjxlWORSE" },
    // CLIC2025_WEB — clic_097cb4
    ClusterCell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 5, distance: 2.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 5, distance: 4.0, subclass: "A_LibjxlBetter" },
    ClusterCell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 7, distance: 2.0, subclass: "C_LibjxlWORSE" },
    ClusterCell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 7, distance: 4.0, subclass: "C_LibjxlWORSE" },
    ClusterCell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 9, distance: 2.0, subclass: "C_LibjxlWORSE" },
    ClusterCell { image_id: "clic_097cb4", path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png", class: "CLIC2025_WEB", effort: 9, distance: 4.0, subclass: "C_LibjxlWORSE" },
    // CLIC2025_WEB — clic_0c49a5
    ClusterCell { image_id: "clic_0c49a5", path: "/home/lilith/work/codec-corpus/clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png", class: "CLIC2025_WEB", effort: 7, distance: 2.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "clic_0c49a5", path: "/home/lilith/work/codec-corpus/clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png", class: "CLIC2025_WEB", effort: 9, distance: 2.0, subclass: "B_BothFail" },
    // CLIC2025_WEB — clic_100a02
    ClusterCell { image_id: "clic_100a02", path: "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png", class: "CLIC2025_WEB", effort: 5, distance: 4.0, subclass: "A_LibjxlBetter" },
    ClusterCell { image_id: "clic_100a02", path: "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png", class: "CLIC2025_WEB", effort: 7, distance: 4.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "clic_100a02", path: "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png", class: "CLIC2025_WEB", effort: 9, distance: 4.0, subclass: "B_BothFail" },
    // CLIC2025_WEB — clic_22ea12
    ClusterCell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 5, distance: 2.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 5, distance: 4.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 7, distance: 2.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 7, distance: 4.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 9, distance: 2.0, subclass: "B_BothFail" },
    ClusterCell { image_id: "clic_22ea12", path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png", class: "CLIC2025_WEB", effort: 9, distance: 4.0, subclass: "B_BothFail" },
];

// ── Helpers (shared with AUDIT-7) ───────────────────────────────────────────

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
        .map(|c| {
            RGB::new(
                srgb_u8_to_linear_f32(c[0]),
                srgb_u8_to_linear_f32(c[1]),
                srgb_u8_to_linear_f32(c[2]),
            )
        })
        .collect();
    let srgb: Vec<[u8; 3]> = pixels
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let lin_img = Img::new(lin, w as usize, h as usize);
    let srgb_img = Img::new(srgb, w as usize, h as usize);
    (lin_img, srgb_img)
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
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let n_pixels = dw * dh;
    let channels = if n_pixels > 0 { dec_lin.len() / n_pixels } else { 0 };
    if channels < 3 {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks_exact(channels)
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

    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks_exact(channels)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;

    Some((bfly, ssim2))
}

fn encode_with_strategy(
    pixels: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    strategy: EncoderStrategy,
) -> Option<(Vec<u8>, u128)> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(strategy);
    let t0 = Instant::now();
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .ok()?;
    let ms = t0.elapsed().as_millis();
    Some((buf, ms))
}

fn encode_cjxl(src_path: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_audit8p2_{}_{}_{}_{}.jxl",
        src_path.file_stem()?.to_string_lossy(),
        effort,
        (distance * 1000.0) as u32,
        std::process::id()
    ));
    let t0 = Instant::now();
    let status = Command::new(cjxl)
        .arg(src_path)
        .arg(&tmp)
        .arg("-e")
        .arg(format!("{}", effort))
        .arg("-d")
        .arg(format!("{}", distance))
        .arg("--quiet")
        .status()
        .ok()?;
    let ms = t0.elapsed().as_millis();
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some((bytes, ms))
}

#[derive(Default, Debug, Clone)]
struct Row {
    image_id: String,
    class: String,
    subclass: String,
    width: u32,
    height: u32,
    effort: u8,
    distance: f32,
    cjxl_bytes: usize,
    cjxl_bfly: f64,
    cjxl_ssim2: f64,
    mode_a_bytes: usize,
    mode_a_bfly: f64,
    mode_a_ssim2: f64,
    mode_a_ms: u128,
    mode_c_bytes: usize,
    mode_c_bfly: f64,
    mode_c_ssim2: f64,
    mode_c_ms: u128,
}

fn classify(delta_ssim2: f64, delta_bytes_pct: f64) -> &'static str {
    // WIN  : Mode C >= Mode A + 0.5 SSIM2 AND |Δbytes| ≤ 2%
    // LOSS : Mode C <= Mode A - 0.5 SSIM2  OR  Δbytes > 2%
    // NEUTRAL: otherwise
    if delta_ssim2 >= 0.5 && delta_bytes_pct.abs() <= 2.0 {
        "WIN"
    } else if delta_ssim2 <= -0.5 || delta_bytes_pct > 2.0 {
        "LOSS"
    } else {
        "NEUTRAL"
    }
}

fn write_tsv(out_path: &Path, rows: &[Row]) {
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = match File::create(out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[bench] FAIL create TSV: {}", e);
            return;
        }
    };
    writeln!(
        f,
        "image_id\tclass\tsubclass\twidth\theight\teffort\tdistance\t\
         cjxl_bytes\tcjxl_bfly\tcjxl_ssim2\t\
         mode_a_bytes\tmode_a_bfly\tmode_a_ssim2\tmode_a_ms\t\
         mode_c_bytes\tmode_c_bfly\tmode_c_ssim2\tmode_c_ms\t\
         mode_a_dBytes_pct\tmode_a_dSsim2\t\
         mode_c_dBytes_pct\tmode_c_dSsim2\t\
         delta_c_minus_a_bytes_pct\tdelta_c_minus_a_ssim2\tdelta_c_minus_a_bfly\tverdict"
    )
    .unwrap();
    for r in rows {
        let mode_a_dbytes = if r.cjxl_bytes > 0 {
            (r.mode_a_bytes as f64 - r.cjxl_bytes as f64) / r.cjxl_bytes as f64 * 100.0
        } else { f64::NAN };
        let mode_a_dssim2 = r.mode_a_ssim2 - r.cjxl_ssim2;
        let mode_c_dbytes = if r.cjxl_bytes > 0 {
            (r.mode_c_bytes as f64 - r.cjxl_bytes as f64) / r.cjxl_bytes as f64 * 100.0
        } else { f64::NAN };
        let mode_c_dssim2 = r.mode_c_ssim2 - r.cjxl_ssim2;
        let delta_c_minus_a_bytes = if r.mode_a_bytes > 0 {
            (r.mode_c_bytes as f64 - r.mode_a_bytes as f64) / r.mode_a_bytes as f64 * 100.0
        } else { f64::NAN };
        let delta_c_minus_a_ssim2 = r.mode_c_ssim2 - r.mode_a_ssim2;
        let delta_c_minus_a_bfly = r.mode_c_bfly - r.mode_a_bfly;
        let verdict = classify(delta_c_minus_a_ssim2, delta_c_minus_a_bytes);
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
             {}\t{:.4}\t{:.4}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {:.3}\t{:.3}\t\
             {:.3}\t{:.3}\t\
             {:.3}\t{:.3}\t{:.4}\t{}",
            r.image_id, r.class, r.subclass, r.width, r.height, r.effort, r.distance,
            r.cjxl_bytes, r.cjxl_bfly, r.cjxl_ssim2,
            r.mode_a_bytes, r.mode_a_bfly, r.mode_a_ssim2, r.mode_a_ms,
            r.mode_c_bytes, r.mode_c_bfly, r.mode_c_ssim2, r.mode_c_ms,
            mode_a_dbytes, mode_a_dssim2,
            mode_c_dbytes, mode_c_dssim2,
            delta_c_minus_a_bytes, delta_c_minus_a_ssim2, delta_c_minus_a_bfly,
            verdict,
        )
        .unwrap();
    }
}

fn make_mode_c_strategy() -> EncoderStrategy {
    // Start from Zenjxl default, flip the single Mode C field.
    let mut custom = EncoderImprovementsCustom::default();
    custom.cfl_newton_libjxl_math_with_ls_warm_start = true;
    EncoderStrategy::Custom(Box::new(custom))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out_path: PathBuf = PathBuf::from(
        "benchmarks/w44_audit_8_phase2_mode_c_on_cluster_2026-05-24.tsv",
    );
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            out_path = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    eprintln!("[bench] output: {}", out_path.display());
    eprintln!("[bench] {} cluster cells × Mode A + Mode C + cjxl = {} encodes",
        CLUSTER.len(), CLUSTER.len() * 3);

    let total = CLUSTER.len();
    let mut rows: Vec<Row> = Vec::with_capacity(total);
    let bench_start = Instant::now();

    // Group by image so we only load each PNG once.
    let mut cells_by_image: std::collections::BTreeMap<&'static str, Vec<&ClusterCell>> =
        std::collections::BTreeMap::new();
    for c in CLUSTER {
        cells_by_image.entry(c.image_id).or_default().push(c);
    }

    let mut done = 0;
    for (image_id, cells) in &cells_by_image {
        let (pixels, w, h) = match load_png(Path::new(cells[0].path)) {
            Some(t) => t,
            None => {
                eprintln!("[bench] FAIL: load {}", cells[0].path);
                continue;
            }
        };
        eprintln!("[bench] LOADED {} {}x{} ({} cluster cells)", image_id, w, h, cells.len());
        let (lin_img, srgb_img) = make_imgs(&pixels, w, h);

        for cell in cells {
            done += 1;
            let t_cell = Instant::now();
            eprint!(
                "[bench] {}/{} {} e{} d={} ({}) ... ",
                done, total, cell.image_id, cell.effort, cell.distance, cell.subclass
            );

            let mut row = Row {
                image_id: cell.image_id.to_string(),
                class: cell.class.to_string(),
                subclass: cell.subclass.to_string(),
                width: w,
                height: h,
                effort: cell.effort,
                distance: cell.distance,
                ..Default::default()
            };

            // cjxl reference
            if let Some((c_bytes, _c_ms)) =
                encode_cjxl(Path::new(cell.path), cell.effort, cell.distance)
            {
                row.cjxl_bytes = c_bytes.len();
                if let Some((bfly, ss)) = score_jxl(&c_bytes, &lin_img, &srgb_img, w, h) {
                    row.cjxl_bfly = bfly;
                    row.cjxl_ssim2 = ss;
                }
            }

            // Mode A: default Zenjxl
            if let Some((a_bytes, a_ms)) = encode_with_strategy(
                &pixels, w, h, cell.effort, cell.distance, EncoderStrategy::Zenjxl,
            ) {
                row.mode_a_bytes = a_bytes.len();
                row.mode_a_ms = a_ms;
                if let Some((bfly, ss)) = score_jxl(&a_bytes, &lin_img, &srgb_img, w, h) {
                    row.mode_a_bfly = bfly;
                    row.mode_a_ssim2 = ss;
                }
            }

            // Mode C: Zenjxl default + cfl_newton_libjxl_math_with_ls_warm_start = true
            if let Some((c_bytes, c_ms)) = encode_with_strategy(
                &pixels, w, h, cell.effort, cell.distance, make_mode_c_strategy(),
            ) {
                row.mode_c_bytes = c_bytes.len();
                row.mode_c_ms = c_ms;
                if let Some((bfly, ss)) = score_jxl(&c_bytes, &lin_img, &srgb_img, w, h) {
                    row.mode_c_bfly = bfly;
                    row.mode_c_ssim2 = ss;
                }
            }

            let dc_a_bytes = if row.mode_a_bytes > 0 {
                (row.mode_c_bytes as f64 - row.mode_a_bytes as f64)
                    / row.mode_a_bytes as f64 * 100.0
            } else { f64::NAN };
            let dc_a_ssim2 = row.mode_c_ssim2 - row.mode_a_ssim2;
            let verdict = classify(dc_a_ssim2, dc_a_bytes);
            let elapsed_ms = t_cell.elapsed().as_millis();
            eprintln!(
                "A={}B/{:.2}/{:.2} C={}B/{:.2}/{:.2} (ΔbytesC-A={:+.2}% Δssim2C-A={:+.3}) [{}] ({}ms)",
                row.mode_a_bytes, row.mode_a_bfly, row.mode_a_ssim2,
                row.mode_c_bytes, row.mode_c_bfly, row.mode_c_ssim2,
                dc_a_bytes, dc_a_ssim2, verdict,
                elapsed_ms,
            );
            rows.push(row);

            // Incremental flush every 5 cells so partial results survive aborts.
            if done % 5 == 0 {
                write_tsv(&out_path, &rows);
            }
        }
    }

    write_tsv(&out_path, &rows);
    eprintln!(
        "[bench] {} rows written to {} in {:.1}s",
        rows.len(),
        out_path.display(),
        bench_start.elapsed().as_secs_f64()
    );

    // Summary by subclass
    eprintln!();
    eprintln!("=== Subclass × Verdict summary ===");
    let subclasses = ["A_LibjxlBetter", "B_BothFail", "C_LibjxlWORSE"];
    let verdicts = ["WIN", "NEUTRAL", "LOSS"];
    for s in &subclasses {
        let mut counts = [0; 3];
        for r in &rows {
            if r.subclass == *s {
                let dc_a_bytes = if r.mode_a_bytes > 0 {
                    (r.mode_c_bytes as f64 - r.mode_a_bytes as f64)
                        / r.mode_a_bytes as f64 * 100.0
                } else { 0.0 };
                let v = classify(r.mode_c_ssim2 - r.mode_a_ssim2, dc_a_bytes);
                for (i, vlabel) in verdicts.iter().enumerate() {
                    if v == *vlabel {
                        counts[i] += 1;
                        break;
                    }
                }
            }
        }
        let total_s: usize = counts.iter().sum();
        eprintln!(
            "  {:16}: WIN={:>2} NEUTRAL={:>2} LOSS={:>2} (total={})",
            s, counts[0], counts[1], counts[2], total_s
        );
    }

    let mut total_counts = [0usize; 3];
    let mut worst_delta_ssim2 = 0.0f64;
    let mut worst_cell = String::new();
    for r in &rows {
        let dc_a_bytes = if r.mode_a_bytes > 0 {
            (r.mode_c_bytes as f64 - r.mode_a_bytes as f64)
                / r.mode_a_bytes as f64 * 100.0
        } else { 0.0 };
        let dc_a_ssim2 = r.mode_c_ssim2 - r.mode_a_ssim2;
        let v = classify(dc_a_ssim2, dc_a_bytes);
        for (i, vlabel) in verdicts.iter().enumerate() {
            if v == *vlabel { total_counts[i] += 1; break; }
        }
        if dc_a_ssim2 < worst_delta_ssim2 {
            worst_delta_ssim2 = dc_a_ssim2;
            worst_cell = format!("{} e{} d={}", r.image_id, r.effort, r.distance);
        }
    }
    eprintln!();
    eprintln!("=== Overall (30 cells) ===");
    eprintln!("  WIN={} NEUTRAL={} LOSS={}", total_counts[0], total_counts[1], total_counts[2]);
    eprintln!("  Worst Mode C effect: {:+.3} SSIM2 on {}", worst_delta_ssim2, worst_cell);
}
