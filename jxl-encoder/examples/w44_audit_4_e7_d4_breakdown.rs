//! W44-AUDIT-4 — diagnose the codec_wiki e7 d=4 +44% bytes wedge.
//!
//! The W44-AUDIT-1/3 benches surfaced:
//!   - Zenjxl  e7 d=4: 90,323 B vs cjxl 62,710 B = +44.03% bytes (SSIM2 +0.03)
//!   - Libjxl  e7 d=4: 70,217 B vs cjxl 62,710 B = +11.97% bytes (SSIM2 -5.51)
//!
//! The wedge decomposes as:
//!   ~ +12pp "structural" (Libjxl-strategy also +12% — a cjxl optimization
//!     we don't have at e7 even in cjxl-parity mode)
//!   ~ +32pp "Zenjxl-only" (the W44-105 buttloop seed × W44-94/96/98/167
//!     variant Z lift stack overshooting at e7 d=4)
//!
//! BUT: at e7 our `butteraugli_iters = 0` (effort.rs profile), so the
//! W44-105 *buttloop* gate cannot fire. The remaining suspect for the
//! Zenjxl-only +32pp wedge at e7 is **W44-109** — the same screenshot
//! seed-scale lift wired in the adaptive_quant pre-buttloop path, gated
//! at `effort ∈ [5, 7]` with per-effort scales `e5/e6 = 2.0` and
//! `e7 = 3.0`. (`vardct/butteraugli_loop.rs:693`.)
//!
//! This harness:
//!   1. Encodes codec_wiki at 5 cells (e7 d=3, e7 d=4, e7 d=5, e8 d=4,
//!      e9 d=4) with Zenjxl + Libjxl + cjxl. Per-cell bytes/SSIM2/bfly.
//!   2. Adds a 4th column "Zenjxl_no_109" (`JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0`)
//!      to isolate the W44-109 share of the wedge at e5/e6/e7.
//!   3. Adds a 5th column "Zenjxl_no_109_no_105" (also
//!      `JXL_BUTTLOOP_INITIAL_QF_SCALE=1.0`) for the e8/e9 cells.
//!
//! Output: benchmarks/w44_audit_4_e7_d4_breakdown_2026-05-24.tsv + meta
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_4_e7_d4_breakdown

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

/// (effort, distance, can_test_w44_109, can_test_w44_105)
const CELLS: &[(u8, f32, bool, bool)] = &[
    (7, 3.0, true, false),
    (7, 4.0, true, false),
    (7, 5.0, true, false),
    (8, 4.0, false, true),
    (9, 4.0, false, true),
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
    let dec_pixels: Vec<RGB<f32>> =
        dec_lin.chunks_exact(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
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
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;

    Some((bfly, ssim2))
}

fn encode_ours(
    pixels: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    strategy: EncoderStrategy,
    disable_w44_109: bool,
    disable_w44_105: bool,
) -> Option<(Vec<u8>, u128)> {
    // SAFETY: harness is sequential.
    unsafe {
        if disable_w44_109 {
            std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "1.0");
        } else {
            std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE");
        }
        if disable_w44_105 {
            std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "1.0");
        } else {
            std::env::remove_var("JXL_BUTTLOOP_INITIAL_QF_SCALE");
        }
    }
    let cfg = LossyConfig::new(distance).with_effort(effort).with_strategy(strategy);
    let t0 = Instant::now();
    let buf = cfg.encode_request(w, h, PixelLayout::Rgb8).encode(pixels).ok()?;
    let ms = t0.elapsed().as_millis();
    unsafe {
        std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE");
        std::env::remove_var("JXL_BUTTLOOP_INITIAL_QF_SCALE");
    }
    Some((buf, ms))
}

fn encode_cjxl(src_path: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_audit4_{}_{}_{}_{}.jxl",
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
    let out_path: PathBuf =
        PathBuf::from("benchmarks/w44_audit_4_e7_d4_breakdown_2026-05-24.tsv");

    let (pixels, w, h) = load_png(Path::new(IMAGE_PATH)).expect("load codec_wiki");
    eprintln!(
        "[bench W44-AUDIT-4] codec_wiki {}×{} = {:.2} MP",
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
        "image_id\teffort\tdistance\t\
         cjxl_B\tcjxl_bfly\tcjxl_ssim2\tcjxl_ms\t\
         libjxl_B\tlibjxl_bfly\tlibjxl_ssim2\tlibjxl_ms\t\
         zenjxl_B\tzenjxl_bfly\tzenjxl_ssim2\tzenjxl_ms\t\
         zenjxl_no109_B\tzenjxl_no109_bfly\tzenjxl_no109_ssim2\t\
         zenjxl_no109_no105_B\tzenjxl_no109_no105_bfly\tzenjxl_no109_no105_ssim2\t\
         libjxl_dpct\tzenjxl_dpct\tzenjxl_no109_dpct\tzenjxl_no109_no105_dpct"
    )
    .unwrap();

    let bench_start = Instant::now();
    for &(effort, dist, can109, can105) in CELLS {
        let t_cell = Instant::now();
        eprint!(
            "[bench] {} e{} d={} ...\n",
            IMAGE_ID, effort, dist
        );

        let (cjxl_b, cjxl_ms) = encode_cjxl(Path::new(IMAGE_PATH), effort, dist).unwrap();
        let (cjxl_bfly, cjxl_ssim2) =
            score_jxl(&cjxl_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
        eprintln!(
            "  cjxl   = {:7} B  bfly={:.3} ssim2={:.2}  ({} ms)",
            cjxl_b.len(),
            cjxl_bfly,
            cjxl_ssim2,
            cjxl_ms
        );

        let (lib_b, lib_ms) =
            encode_ours(&pixels, w, h, effort, dist, EncoderStrategy::Libjxl, false, false)
                .unwrap();
        let (lib_bfly, lib_ssim2) =
            score_jxl(&lib_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
        eprintln!(
            "  libjxl = {:7} B  bfly={:.3} ssim2={:.2}  ({} ms)  Δ={:+.2}% vs cjxl",
            lib_b.len(),
            lib_bfly,
            lib_ssim2,
            lib_ms,
            (lib_b.len() as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0,
        );

        let (zen_b, zen_ms) =
            encode_ours(&pixels, w, h, effort, dist, EncoderStrategy::Zenjxl, false, false)
                .unwrap();
        let (zen_bfly, zen_ssim2) =
            score_jxl(&zen_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
        eprintln!(
            "  zenjxl = {:7} B  bfly={:.3} ssim2={:.2}  ({} ms)  Δ={:+.2}% vs cjxl",
            zen_b.len(),
            zen_bfly,
            zen_ssim2,
            zen_ms,
            (zen_b.len() as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0,
        );

        let (no109_b, no109_bfly, no109_ssim2) = if can109 {
            let (b, _) = encode_ours(&pixels, w, h, effort, dist, EncoderStrategy::Zenjxl, true, false).unwrap();
            let (bf, ss) = score_jxl(&b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
            eprintln!(
                "  zenjxl_no109 = {:7} B  bfly={:.3} ssim2={:.2}  Δ={:+.2}% vs cjxl",
                b.len(),
                bf,
                ss,
                (b.len() as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0,
            );
            (b.len(), bf, ss)
        } else {
            (0usize, 0.0, 0.0)
        };

        let (no_b, no_bfly, no_ss) = if can105 {
            let (b, _) = encode_ours(&pixels, w, h, effort, dist, EncoderStrategy::Zenjxl, false, true).unwrap();
            let (bf, ss) = score_jxl(&b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
            eprintln!(
                "  zenjxl_no105 = {:7} B  bfly={:.3} ssim2={:.2}  Δ={:+.2}% vs cjxl",
                b.len(),
                bf,
                ss,
                (b.len() as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0,
            );
            (b.len(), bf, ss)
        } else {
            (0usize, 0.0, 0.0)
        };

        let libpct =
            (lib_b.len() as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0;
        let zenpct =
            (zen_b.len() as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0;
        let no109pct = if no109_b > 0 {
            (no109_b as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0
        } else {
            f64::NAN
        };
        let no_pct = if no_b > 0 {
            (no_b as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0
        } else {
            f64::NAN
        };

        writeln!(
            f,
            "{}\t{}\t{}\t\
             {}\t{:.3}\t{:.2}\t{}\t\
             {}\t{:.3}\t{:.2}\t{}\t\
             {}\t{:.3}\t{:.2}\t{}\t\
             {}\t{:.3}\t{:.2}\t\
             {}\t{:.3}\t{:.2}\t\
             {:.2}\t{:.2}\t{:.2}\t{:.2}",
            IMAGE_ID,
            effort,
            dist,
            cjxl_b.len(),
            cjxl_bfly,
            cjxl_ssim2,
            cjxl_ms,
            lib_b.len(),
            lib_bfly,
            lib_ssim2,
            lib_ms,
            zen_b.len(),
            zen_bfly,
            zen_ssim2,
            zen_ms,
            no109_b,
            no109_bfly,
            no109_ssim2,
            no_b,
            no_bfly,
            no_ss,
            libpct,
            zenpct,
            no109pct,
            no_pct,
        )
        .unwrap();
        eprintln!("  cell took {:.1}s", t_cell.elapsed().as_secs_f64());
    }

    drop(f);
    eprintln!(
        "[bench W44-AUDIT-4] total {:.1}s, output {}",
        bench_start.elapsed().as_secs_f64(),
        out_path.display()
    );
}
