//! W44-AUDIT-6 Phase 1 — single-cell bisection on codec_wiki e7 d=4.
//!
//! Purpose: verify the new `high_colour_class_exclude` gate (Section B,
//! `m3_colourfulness >= 80.0` on `ZenanalyzeProxies`) actually fires on
//! codec_wiki and produces the predicted Pareto improvement:
//!   - bytes within +5% of cjxl (vs +44.03% baseline per W44-AUDIT-4)
//!   - SSIM2 within -0.5 of cjxl 84.86
//!
//! Mode layout (3 cells × 4 modes per cell):
//!   A: baseline    — Zenjxl WITHOUT the AUDIT-6 exclude (env disable)
//!                    Reproduces pre-AUDIT-6 +44% wedge.
//!   B: shipped     — Zenjxl WITH the AUDIT-6 exclude (production default)
//!                    Should suppress lift on codec_wiki (M3=145.73 >= 80).
//!   C: no-W109 ref — Zenjxl with W44-109 fully disabled (env JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0)
//!                    Sanity check that AUDIT-6 lands at the SAME numbers
//!                    as the W44-109-disabled baseline (proves the exclude
//!                    works exactly like turning W44-109 off on this cell).
//!   D: cjxl ref    — cjxl v0.12.0 at same effort/distance.
//!
//! Cells:
//!   codec_wiki e7 d=4    (the AUDIT-4 target wedge)
//!   codec_wiki e7 d=3    (below W44-109 gate min_distance=3.5, A==B sanity)
//!   codec_wiki e7 d=5    (above W44-109 gate, more aggressive lift)
//!
//! Acceptance gate (Phase 1):
//!   On e7 d=4, Mode B vs Mode D (cjxl):
//!     bytes within +5%, SSIM2 within -0.5 (cjxl - B)
//!   On e7 d=3, Mode B == Mode A (gate doesn't fire, byte-identical)
//!
//! Output: benchmarks/w44_audit_6_phase1_bisect_2026-05-24.tsv + meta
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_6_phase1_bisect

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const IMAGE_PATH: &str = "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png";
const IMAGE_ID: &str = "codec_wiki";

/// (effort, distance, expect_gate_to_fire_in_mode_B)
const CELLS: &[(u8, f32, bool)] = &[
    (7, 3.0, false), // below W44-109 min_distance, A==B expected
    (7, 4.0, true),  // primary target wedge cell
    (7, 5.0, true),  // above gate
];

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
    (
        Img::new(lin, w as usize, h as usize),
        Img::new(srgb, w as usize, h as usize),
    )
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

/// Mode selects which env vars to set / which strategy to use:
///   "A_baseline_no_audit6"     → Zenjxl, JXL_W44_AUDIT_6_DISABLE=1
///   "B_shipped"                → Zenjxl, no env (production default — audit-6 ON)
///   "C_no_w109"                → Zenjxl, JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0
fn encode_ours(
    pixels: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    mode: &str,
) -> Option<(Vec<u8>, u128)> {
    // SAFETY: harness is sequential.
    unsafe {
        std::env::remove_var("JXL_W44_AUDIT_6_DISABLE");
        std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE");
        match mode {
            "A_baseline_no_audit6" => {
                std::env::set_var("JXL_W44_AUDIT_6_DISABLE", "1");
            }
            "B_shipped" => {} // production default
            "C_no_w109" => {
                std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "1.0");
            }
            _ => unreachable!("unknown mode: {mode}"),
        }
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    let t0 = Instant::now();
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .ok()?;
    let ms = t0.elapsed().as_millis();
    unsafe {
        std::env::remove_var("JXL_W44_AUDIT_6_DISABLE");
        std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE");
    }
    Some((buf, ms))
}

fn encode_cjxl(src_path: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_audit6_{}_{}_{}_{}.jxl",
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

fn main() {
    let out_path: PathBuf = PathBuf::from("benchmarks/w44_audit_6_phase1_bisect_2026-05-24.tsv");

    let (pixels, w, h) = load_png(Path::new(IMAGE_PATH)).expect("load codec_wiki");
    eprintln!(
        "[bench W44-AUDIT-6 P1] codec_wiki {}×{} = {:.2} MP",
        w,
        h,
        (w as f64 * h as f64) / 1_000_000.0
    );
    let (lin_img, srgb_img) = make_imgs(&pixels, w, h);

    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = File::create(&out_path).expect("create TSV");
    writeln!(
        f,
        "image_id\teffort\tdistance\tmode\tbytes\tbfly\tssim2\tms\t\
         delta_bytes_pct_vs_cjxl\tdelta_ssim2_vs_cjxl"
    )
    .unwrap();

    let bench_start = Instant::now();
    for &(effort, dist, gate_expected) in CELLS {
        eprintln!(
            "\n[cell] codec_wiki e{} d={} (gate_fires_in_B={})",
            effort, dist, gate_expected
        );

        // D first (cjxl) so we have a reference
        let (cjxl_b, cjxl_ms) = encode_cjxl(Path::new(IMAGE_PATH), effort, dist).unwrap();
        let (cjxl_bfly, cjxl_ssim2) =
            score_jxl(&cjxl_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
        let cjxl_bytes = cjxl_b.len();
        eprintln!(
            "  D=cjxl      {} B   bfly={:.3}  ssim2={:.2}  ({} ms)",
            cjxl_bytes, cjxl_bfly, cjxl_ssim2, cjxl_ms
        );
        writeln!(
            f,
            "{}\t{}\t{}\tD_cjxl\t{}\t{:.4}\t{:.4}\t{}\t0\t0",
            IMAGE_ID, effort, dist, cjxl_bytes, cjxl_bfly, cjxl_ssim2, cjxl_ms
        )
        .unwrap();

        for mode in ["A_baseline_no_audit6", "B_shipped", "C_no_w109"] {
            let (buf, ms) = encode_ours(&pixels, w, h, effort, dist, mode).unwrap();
            let (bfly, ssim2) = score_jxl(&buf, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
            let dbytes = (buf.len() as f64 - cjxl_bytes as f64) / cjxl_bytes as f64 * 100.0;
            let dssim2 = ssim2 - cjxl_ssim2;
            eprintln!(
                "  {:<24}  {} B   bfly={:.3}  ssim2={:.2}  ({} ms)   Δb={:+.2}%  Δss2={:+.2}",
                mode,
                buf.len(),
                bfly,
                ssim2,
                ms,
                dbytes,
                dssim2,
            );
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}",
                IMAGE_ID,
                effort,
                dist,
                mode,
                buf.len(),
                bfly,
                ssim2,
                ms,
                dbytes,
                dssim2,
            )
            .unwrap();
        }
    }
    let total_s = bench_start.elapsed().as_secs_f64();
    eprintln!(
        "\n[bench W44-AUDIT-6 P1] done in {:.1}s; TSV: {}",
        total_s,
        out_path.display()
    );
}
