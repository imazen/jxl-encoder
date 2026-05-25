//! SA-G Fix B smoke: clic_22ea12 e9 d=4 under EncoderStrategy::Libjxl vs cjxl.
//!
//! Measures bytes + butteraugli + SSIM2 for our libjxl-parity encoder vs cjxl
//! after applying SA-G Fix B (Highway-order reduction in CfL Newton). Also
//! re-runs on Zenjxl strategy to verify byte-identity vs the parent commit's
//! Zenjxl baseline (regression check).
//!
//! Pre-fix SA-G measurement on clic_22ea12 e9 d=4:
//!   Strategy::Libjxl bytes: +2.19% vs cjxl
//!   Strategy::Libjxl SSIM2: -3.87 vs cjxl
//!   cmap_x at one tile: -0.119048 vs cjxl 0.000000 (snap-to-zero divergence)
//!
//! Acceptance target: bytes within 1pp of cjxl, SSIM2 within 0.5 of cjxl.
//!
//! Usage:
//!   cargo run --release \
//!     --features 'parallel butteraugli-loop ssim2-loop __internals __expert' \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     --example sa_g_fix_b_smoke

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::{Img, ImgVec};
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

const EFFORT: u8 = 9;
const DISTANCE: f32 = 4.0;
const CELL_PATH: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png";

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL_BIN") {
        return PathBuf::from(p);
    }
    PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
}
fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}
fn decode_jxl_linear(bytes: &[u8]) -> Option<(u32, u32, Vec<f32>)> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width() as u32, fb.height() as u32, fb.buf().to_vec()))
}
fn rgb_u8_to_linear_img(rgb: &[u8], w: u32, h: u32) -> ImgVec<RGB<f32>> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(RGB {
            r: srgb_to_linear_f32(rgb[i * 3]),
            g: srgb_to_linear_f32(rgb[i * 3 + 1]),
            b: srgb_to_linear_f32(rgb[i * 3 + 2]),
        });
    }
    Img::new(out, w as usize, h as usize)
}
fn linear_planar_to_img(lin: &[f32], w: u32, h: u32) -> ImgVec<RGB<f32>> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(RGB {
            r: lin[i * 3].clamp(0.0, 1.0),
            g: lin[i * 3 + 1].clamp(0.0, 1.0),
            b: lin[i * 3 + 2].clamp(0.0, 1.0),
        });
    }
    Img::new(out, w as usize, h as usize)
}
fn linear_planar_to_srgb_arr3(lin: &[f32], w: u32, h: u32) -> ImgVec<[u8; 3]> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let conv = |c: f32| -> u8 {
            let c = c.clamp(0.0, 1.0);
            let s = if c <= 0.003_130_8 {
                12.92 * c
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0 + 0.5) as u8
        };
        out.push([conv(lin[i * 3]), conv(lin[i * 3 + 1]), conv(lin[i * 3 + 2])]);
    }
    Img::new(out, w as usize, h as usize)
}
fn rgb_u8_to_arr3_img(rgb: &[u8], w: u32, h: u32) -> ImgVec<[u8; 3]> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]]);
    }
    Img::new(out, w as usize, h as usize)
}
fn encode_with_strategy(rgb: &[u8], w: u32, h: u32, strategy: EncoderStrategy) -> Vec<u8> {
    LossyConfig::new(DISTANCE)
        .with_effort(EFFORT)
        .with_threads(1)
        .with_strategy(strategy)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
}
fn encode_cjxl(src_png: &Path) -> Option<Vec<u8>> {
    let stem = src_png.file_stem()?.to_string_lossy().into_owned();
    let out_path =
        std::env::temp_dir().join(format!("sa_g_fix_b_cjxl_{stem}_e{EFFORT}_d{DISTANCE}.jxl"));
    let status = Command::new(cjxl_bin())
        .arg(src_png)
        .arg(&out_path)
        .arg("-d")
        .arg(format!("{DISTANCE}"))
        .arg("-e")
        .arg(format!("{EFFORT}"))
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&out_path).ok()?;
    let _ = std::fs::remove_file(&out_path);
    Some(bytes)
}
fn ssim2_metric(src: &ImgVec<[u8; 3]>, dec: &ImgVec<[u8; 3]>) -> f32 {
    fast_ssim2::compute_ssimulacra2(src.as_ref(), dec.as_ref()).unwrap_or(0.0) as f32
}
fn bfly_metric(src: &ImgVec<RGB<f32>>, dec: &ImgVec<RGB<f32>>) -> f32 {
    let p = ButteraugliParams::default();
    butteraugli_linear(src.as_ref(), dec.as_ref(), &p)
        .map(|s| s.score as f32)
        .unwrap_or(99.0)
}

fn main() {
    let path = Path::new(CELL_PATH);
    let (rgb, w, h) = load_png(path).expect("load_png");
    let src_lin = rgb_u8_to_linear_img(&rgb, w, h);
    let src_srgb = rgb_u8_to_arr3_img(&rgb, w, h);

    println!(
        "# SA-G Fix B smoke: clic_22ea12 e{EFFORT} d={DISTANCE} ({}x{})",
        w, h
    );
    println!(
        "strategy\tbytes\tssim2\tbutteraugli\tcjxl_bytes\tcjxl_ssim2\tcjxl_bfly\tdelta_bytes_pct\tdelta_ssim2\tdelta_bfly"
    );

    let cjxl_bytes = encode_cjxl(path).expect("cjxl encode failed");
    let (cw, ch, cdec) = decode_jxl_linear(&cjxl_bytes).expect("cjxl decode failed");
    assert_eq!((cw, ch), (w, h));
    let cjxl_lin = linear_planar_to_img(&cdec, w, h);
    let cjxl_srgb = linear_planar_to_srgb_arr3(&cdec, w, h);
    let cjxl_ssim2 = ssim2_metric(&src_srgb, &cjxl_srgb);
    let cjxl_bfly = bfly_metric(&src_lin, &cjxl_lin);

    for (label, strategy) in &[
        ("Libjxl", EncoderStrategy::Libjxl),
        ("Zenjxl", EncoderStrategy::Zenjxl),
    ] {
        let our_bytes = encode_with_strategy(&rgb, w, h, strategy.clone());
        if std::env::var("SA_G_DUMP_JXL").is_ok() {
            std::fs::write(format!("/tmp/sa_g_fix_b_{label}.jxl"), &our_bytes).ok();
        }
        let (dw, dh, dec) = decode_jxl_linear(&our_bytes).expect("decode our jxl");
        assert_eq!((dw, dh), (w, h));
        let our_lin = linear_planar_to_img(&dec, w, h);
        let our_srgb = linear_planar_to_srgb_arr3(&dec, w, h);
        let our_ssim2 = ssim2_metric(&src_srgb, &our_srgb);
        let our_bfly = bfly_metric(&src_lin, &our_lin);
        let db =
            (our_bytes.len() as f32 - cjxl_bytes.len() as f32) / cjxl_bytes.len() as f32 * 100.0;
        let ds = our_ssim2 - cjxl_ssim2;
        let dbf = our_bfly - cjxl_bfly;
        println!(
            "{}\t{}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:.3}\t{:.3}\t{:.3}",
            label,
            our_bytes.len(),
            our_ssim2,
            our_bfly,
            cjxl_bytes.len(),
            cjxl_ssim2,
            cjxl_bfly,
            db,
            ds,
            dbf
        );
    }
}
