//! W44-AUDIT-3 — focused 3-cell re-validation of codec_wiki d=4 post-OOM-fix.
//!
//! The W44-AUDIT-1 bench (`benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.tsv`)
//! recorded:
//!   - codec_wiki e5 d=4: zenjxl 74641 / cjxl 72551 = +2.88% (RAN)
//!   - codec_wiki e7 d=4: zenjxl 90323 / cjxl 62710 = +44.03% (RAN, but +44% wedge)
//!   - codec_wiki e9 d=4: OOM on both strategies (FAIL)
//!
//! W44-AUDIT-2 (`887cac54`) fixed the OOM (EPF budget-accounting leak).
//! This harness re-runs JUST the 3 codec_wiki d=4 cells (e5/e7/e9) with
//! full bytes + butteraugli + SSIM2 scoring so the published parity tables
//! can be updated with real values for the previously-FAIL e9 row.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_3_revalidate -- \
//!     --output benchmarks/cjxl_parity_2026-05-24_post_w44_audit_2_partial.tsv

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
const CLASS: &str = "SCREEN";
const DISTANCE: f32 = 4.0;
const EFFORTS: &[u8] = &[5, 7, 9];

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
        "cjxl_audit3_{}_{}_{}_{}.jxl",
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
    let args: Vec<String> = std::env::args().collect();
    let mut out_path: PathBuf =
        PathBuf::from("benchmarks/cjxl_parity_2026-05-24_post_w44_audit_2_partial.tsv");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            out_path = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    eprintln!("[bench W44-AUDIT-3] output: {}", out_path.display());

    let (pixels, w, h) = match load_png(Path::new(IMAGE_PATH)) {
        Some(t) => t,
        None => {
            eprintln!("FAIL: cannot load {}", IMAGE_PATH);
            std::process::exit(1);
        }
    };
    eprintln!(
        "[bench] loaded {} ({}×{} = {:.2} MP)",
        IMAGE_ID,
        w,
        h,
        (w as f64 * h as f64) / 1_000_000.0
    );
    let (lin_img, srgb_img) = make_imgs(&pixels, w, h);

    let bench_start = Instant::now();
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

    for &effort in EFFORTS {
        let t_cell = Instant::now();
        eprint!("[bench] {} e{} d={} ... ", IMAGE_ID, effort, DISTANCE);

        let mut cjxl_bytes = 0usize;
        let mut cjxl_bfly = 0.0f64;
        let mut cjxl_ssim2 = 0.0f64;
        let mut cjxl_ms = 0u128;
        let mut zen_bytes = 0usize;
        let mut zen_bfly = 0.0f64;
        let mut zen_ssim2 = 0.0f64;
        let mut zen_ms = 0u128;
        let mut lib_bytes = 0usize;
        let mut lib_bfly = 0.0f64;
        let mut lib_ssim2 = 0.0f64;
        let mut lib_ms = 0u128;

        if let Some((b, ms)) = encode_cjxl(Path::new(IMAGE_PATH), effort, DISTANCE) {
            cjxl_bytes = b.len();
            cjxl_ms = ms;
            if let Some((bf, ss)) = score_jxl(&b, &lin_img, &srgb_img, w, h) {
                cjxl_bfly = bf;
                cjxl_ssim2 = ss;
            }
        }

        if let Some((b, ms)) =
            encode_zenjxl(&pixels, w, h, effort, DISTANCE, EncoderStrategy::Zenjxl)
        {
            zen_bytes = b.len();
            zen_ms = ms;
            if let Some((bf, ss)) = score_jxl(&b, &lin_img, &srgb_img, w, h) {
                zen_bfly = bf;
                zen_ssim2 = ss;
            }
        }

        if let Some((b, ms)) =
            encode_zenjxl(&pixels, w, h, effort, DISTANCE, EncoderStrategy::Libjxl)
        {
            lib_bytes = b.len();
            lib_ms = ms;
            if let Some((bf, ss)) = score_jxl(&b, &lin_img, &srgb_img, w, h) {
                lib_bfly = bf;
                lib_ssim2 = ss;
            }
        }

        let elapsed_ms = t_cell.elapsed().as_millis();
        eprintln!(
            "z={}B/{:.3}/{:.2} l={}B/{:.3}/{:.2} c={}B/{:.3}/{:.2} ({}ms)",
            zen_bytes,
            zen_bfly,
            zen_ssim2,
            lib_bytes,
            lib_bfly,
            lib_ssim2,
            cjxl_bytes,
            cjxl_bfly,
            cjxl_ssim2,
            elapsed_ms
        );

        let z_dbytes = if cjxl_bytes > 0 {
            (zen_bytes as f64 - cjxl_bytes as f64) / cjxl_bytes as f64 * 100.0
        } else {
            f64::NAN
        };
        let z_dssim2 = zen_ssim2 - cjxl_ssim2;
        let z_dbfly = zen_bfly - cjxl_bfly;
        let l_dbytes = if cjxl_bytes > 0 {
            (lib_bytes as f64 - cjxl_bytes as f64) / cjxl_bytes as f64 * 100.0
        } else {
            f64::NAN
        };
        let l_dssim2 = lib_ssim2 - cjxl_ssim2;
        let l_dbfly = lib_bfly - cjxl_bfly;

        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {}\t{:.4}\t{:.4}\t{}\t\
             {:.3}\t{:.3}\t{:.4}\t\
             {:.3}\t{:.3}\t{:.4}",
            IMAGE_ID,
            CLASS,
            w,
            h,
            effort,
            DISTANCE,
            cjxl_bytes,
            cjxl_bfly,
            cjxl_ssim2,
            cjxl_ms,
            zen_bytes,
            zen_bfly,
            zen_ssim2,
            zen_ms,
            lib_bytes,
            lib_bfly,
            lib_ssim2,
            lib_ms,
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
        "[bench W44-AUDIT-3] 3 rows written to {} in {:.1}s",
        out_path.display(),
        bench_start.elapsed().as_secs_f64()
    );
}
