//! cjxl parity bench — W44-AUDIT-1 snapshot 2026-05-24.
//!
//! Headline measurement of zenjxl-vs-cjxl after the W44-158→W44-205
//! tuning arc + S2-refit-c2 + Phase-3 B1-B7d perf work. The prior
//! committed parity ledger (`benchmarks/cjxl_parity_ledger_2026-05-19_*.tsv`)
//! is stale: ~50 commits have landed since, including W44-164 auto-classify,
//! W44-171 DC-tree gate, W44-172 Predictor::Best, W44-193 strategy_def,
//! W44-201/205 coeff_orders, S2-refit-c2.
//!
//! Goals:
//!   1. Concrete zenjxl-vs-cjxl deltas on 4 representative images, 3
//!      efforts, 3 distances, BOTH EncoderStrategy::Zenjxl (default,
//!      all wins on) AND EncoderStrategy::Libjxl (strict-parity gate).
//!   2. Single TSV the audit can publish as 3 markdown tables (bytes,
//!      SSIM2, butteraugli) + an aggregate row.
//!
//! Cells: 4 images × 3 distances × 3 efforts × 2 zenjxl strategies = 72
//!        cells + 36 cjxl cells (one per image×distance×effort) = 108.
//!
//! Wall budget on a 7950X (release, parallel + butteraugli-loop +
//! ssim2-loop): ~30-90 minutes — within the 60 min audit budget.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example cjxl_parity_2026_05_24 -- \
//!     --output benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.tsv

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

// ── Bench matrix ────────────────────────────────────────────────────────────

const EFFORTS: &[u8] = &[5, 7, 9];
const DISTANCES: &[f32] = &[0.5, 2.0, 4.0];

struct Cell {
    image_id: &'static str,
    path: &'static str,
    class: &'static str,
}

const CELLS: &[Cell] = &[
    Cell {
        image_id: "codec_wiki",
        path: "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        class: "SCREEN",
    },
    Cell {
        image_id: "1418519",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        class: "PHOTO",
    },
    Cell {
        image_id: "1025469",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
        class: "PHOTO",
    },
    Cell {
        image_id: "1531677",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        class: "PHOTO_SMOOTH",
    },
];

// ── Helpers ─────────────────────────────────────────────────────────────────

fn srgb_u8_to_linear_f32(x: u8) -> f32 {
    let x = x as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
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

fn make_imgs(pixels: &[u8], w: u32, h: u32) -> (Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>) {
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
    let srgb: Vec<[u8; 3]> = pixels.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
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
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks_exact(3)
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
        .chunks_exact(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;

    Some((bfly, ssim2))
}

fn encode_zenjxl(
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
        "cjxl_parity_{}_{}_{}_{}.jxl",
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

// ── Main ────────────────────────────────────────────────────────────────────

#[derive(Default, Debug, Clone)]
struct Row {
    image_id: String,
    class: String,
    width: u32,
    height: u32,
    effort: u8,
    distance: f32,
    cjxl_bytes: usize,
    cjxl_bfly: f64,
    cjxl_ssim2: f64,
    cjxl_encode_ms: u128,
    zenjxl_bytes: usize,
    zenjxl_bfly: f64,
    zenjxl_ssim2: f64,
    zenjxl_encode_ms: u128,
    libjxl_strat_bytes: usize,
    libjxl_strat_bfly: f64,
    libjxl_strat_ssim2: f64,
    libjxl_strat_encode_ms: u128,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out_path: PathBuf =
        PathBuf::from("benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.tsv");
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

    let total_cells = CELLS.len() * EFFORTS.len() * DISTANCES.len();
    let mut rows: Vec<Row> = Vec::with_capacity(total_cells);
    let bench_start = Instant::now();
    let mut done = 0;

    for cell in CELLS {
        let (pixels, w, h) = match load_png(Path::new(cell.path)) {
            Some(t) => t,
            None => {
                eprintln!("[bench] FAIL: load {}", cell.path);
                continue;
            }
        };
        let (lin_img, srgb_img) = make_imgs(&pixels, w, h);

        for &effort in EFFORTS {
            for &distance in DISTANCES {
                done += 1;
                let t_cell = Instant::now();
                eprint!(
                    "[bench] {}/{} {} e{} d={} ... ",
                    done, total_cells, cell.image_id, effort, distance
                );

                let mut row = Row {
                    image_id: cell.image_id.to_string(),
                    class: cell.class.to_string(),
                    width: w,
                    height: h,
                    effort,
                    distance,
                    ..Default::default()
                };

                // cjxl
                if let Some((c_bytes, c_ms)) = encode_cjxl(Path::new(cell.path), effort, distance) {
                    row.cjxl_bytes = c_bytes.len();
                    row.cjxl_encode_ms = c_ms;
                    if let Some((bfly, ss)) = score_jxl(&c_bytes, &lin_img, &srgb_img, w, h) {
                        row.cjxl_bfly = bfly;
                        row.cjxl_ssim2 = ss;
                    }
                }

                // zenjxl (default strategy = Zenjxl)
                if let Some((z_bytes, z_ms)) =
                    encode_zenjxl(&pixels, w, h, effort, distance, EncoderStrategy::Zenjxl)
                {
                    row.zenjxl_bytes = z_bytes.len();
                    row.zenjxl_encode_ms = z_ms;
                    if let Some((bfly, ss)) = score_jxl(&z_bytes, &lin_img, &srgb_img, w, h) {
                        row.zenjxl_bfly = bfly;
                        row.zenjxl_ssim2 = ss;
                    }
                }

                // EncoderStrategy::Libjxl (strict cjxl-parity gate)
                if let Some((l_bytes, l_ms)) =
                    encode_zenjxl(&pixels, w, h, effort, distance, EncoderStrategy::Libjxl)
                {
                    row.libjxl_strat_bytes = l_bytes.len();
                    row.libjxl_strat_encode_ms = l_ms;
                    if let Some((bfly, ss)) = score_jxl(&l_bytes, &lin_img, &srgb_img, w, h) {
                        row.libjxl_strat_bfly = bfly;
                        row.libjxl_strat_ssim2 = ss;
                    }
                }

                let elapsed_ms = t_cell.elapsed().as_millis();
                eprintln!(
                    "z={}B/{:.3}/{:.2} l={}B/{:.3}/{:.2} c={}B/{:.3}/{:.2} ({}ms)",
                    row.zenjxl_bytes,
                    row.zenjxl_bfly,
                    row.zenjxl_ssim2,
                    row.libjxl_strat_bytes,
                    row.libjxl_strat_bfly,
                    row.libjxl_strat_ssim2,
                    row.cjxl_bytes,
                    row.cjxl_bfly,
                    row.cjxl_ssim2,
                    elapsed_ms
                );
                rows.push(row);
            }
        }
    }

    // Write TSV
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = File::create(&out_path).expect("create TSV");
    writeln!(
        f,
        "image_id\tclass\twidth\theight\teffort\tdistance\t\
         cjxl_bytes\tcjxl_bfly\tcjxl_ssim2\tcjxl_encode_ms\t\
         zenjxl_bytes\tzenjxl_bfly\tzenjxl_ssim2\tzenjxl_encode_ms\t\
         libjxl_strat_bytes\tlibjxl_strat_bfly\tlibjxl_strat_ssim2\tlibjxl_strat_encode_ms\t\
         zenjxl_dBytes_pct\tzenjxl_dSsim2\tzenjxl_dBfly\t\
         libjxl_dBytes_pct\tlibjxl_dSsim2\tlibjxl_dBfly"
    )
    .unwrap();
    for r in &rows {
        let z_dbytes = if r.cjxl_bytes > 0 {
            (r.zenjxl_bytes as f64 - r.cjxl_bytes as f64) / r.cjxl_bytes as f64 * 100.0
        } else {
            f64::NAN
        };
        let z_dssim2 = r.zenjxl_ssim2 - r.cjxl_ssim2;
        let z_dbfly = r.zenjxl_bfly - r.cjxl_bfly;
        let l_dbytes = if r.cjxl_bytes > 0 {
            (r.libjxl_strat_bytes as f64 - r.cjxl_bytes as f64) / r.cjxl_bytes as f64 * 100.0
        } else {
            f64::NAN
        };
        let l_dssim2 = r.libjxl_strat_ssim2 - r.cjxl_ssim2;
        let l_dbfly = r.libjxl_strat_bfly - r.cjxl_bfly;
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {:.3}\t{:.3}\t{:.4}\t\
             {:.3}\t{:.3}\t{:.4}",
            r.image_id,
            r.class,
            r.width,
            r.height,
            r.effort,
            r.distance,
            r.cjxl_bytes,
            r.cjxl_bfly,
            r.cjxl_ssim2,
            r.cjxl_encode_ms,
            r.zenjxl_bytes,
            r.zenjxl_bfly,
            r.zenjxl_ssim2,
            r.zenjxl_encode_ms,
            r.libjxl_strat_bytes,
            r.libjxl_strat_bfly,
            r.libjxl_strat_ssim2,
            r.libjxl_strat_encode_ms,
            z_dbytes,
            z_dssim2,
            z_dbfly,
            l_dbytes,
            l_dssim2,
            l_dbfly
        )
        .unwrap();
    }
    eprintln!(
        "[bench] {} rows written to {} in {:.1}s",
        rows.len(),
        out_path.display(),
        bench_start.elapsed().as_secs_f64()
    );
}
