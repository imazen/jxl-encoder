//! W44-178 Phase 1.5 — verify the W44-173 region SSIM2 finding.
//!
//! Phase 1 (w44_178_recon_diff) showed our decoded output and cjxl's
//! decoded output differ by max 0.0025 in linear RGB in the right
//! column (where DCT64X64 dominates 100%). Yet W44-173 reported the
//! right column SSIM2 is -7 worse than cjxl. Both can't be true unless
//! SSIM2 amplifies sub-millivolt noise — let me verify.
//!
//! This example reproduces the W44-173 per-region SSIM2 numerically
//! AND adds: SSIM2(ours, cjxl) per region (which should be ~99 if
//! ours/cjxl really are nearly identical in the right column).
//!
//! Run:
//!   cargo run --release --features 'parallel butteraugli-loop ssim2-loop'
//!     --manifest-path jxl-encoder/Cargo.toml --example w44_178_region_ssim2

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

const SRC_PNG: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png";
const EFFORT: u8 = 7;
const DISTANCE: f32 = 5.0;

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
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

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("load png");
    let rgb = img.to_rgb8();
    (rgb.as_raw().clone(), rgb.width(), rgb.height())
}

/// Decode bitstream into linear-RGB f32 planar via jxl-oxide.
fn decode_jxl_linear(bytes: &[u8]) -> (usize, usize, Vec<f32>) {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).expect("read");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).expect("render");
    let fb = render.image_all_channels();
    (fb.width(), fb.height(), fb.buf().to_vec())
}

fn encode_ours(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    LossyConfig::new(DISTANCE)
        .with_effort(EFFORT)
        .with_threads(1)
        .with_strategy(EncoderStrategy::Zenjxl)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode ours")
}

fn encode_cjxl(src_png: &Path) -> Vec<u8> {
    let tmp = std::env::temp_dir().join("w44_178_region_ssim2_cjxl.jxl");
    let bin = cjxl_bin();
    let status = Command::new(&bin)
        .arg(src_png)
        .arg(&tmp)
        .args(["-e", &EFFORT.to_string()])
        .args(["-d", &DISTANCE.to_string()])
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .status()
        .expect("cjxl");
    assert!(status.success());
    let bytes = std::fs::read(&tmp).expect("read jxl");
    let _ = std::fs::remove_file(&tmp);
    bytes
}

fn extract_lin(
    full: &[f32],
    width: usize,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
) -> Img<Vec<RGB<f32>>> {
    let mut out = Vec::with_capacity(rw * rh);
    for y in y0..(y0 + rh) {
        for x in x0..(x0 + rw) {
            let i = (y * width + x) * 3;
            out.push(RGB::new(full[i], full[i + 1], full[i + 2]));
        }
    }
    Img::new(out, rw, rh)
}

fn linear_planar_to_srgb_packed(lin: &[f32], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height * 3);
    for px in lin.chunks(3) {
        out.push(linear_to_srgb_u8(px[0]));
        out.push(linear_to_srgb_u8(px[1]));
        out.push(linear_to_srgb_u8(px[2]));
    }
    out
}

fn region_srgb(
    src_srgb: &[u8],
    width: usize,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
) -> Img<Vec<[u8; 3]>> {
    let mut out = Vec::with_capacity(rw * rh);
    for y in y0..(y0 + rh) {
        for x in x0..(x0 + rw) {
            let i = (y * width + x) * 3;
            out.push([src_srgb[i], src_srgb[i + 1], src_srgb[i + 2]]);
        }
    }
    Img::new(out, rw, rh)
}

fn main() {
    let (src_rgb, w, h) = load_png(Path::new(SRC_PNG));
    let width = w as usize;
    let height = h as usize;
    eprintln!("Source: {}×{}", width, height);

    eprintln!("[encode ours] ...");
    let ours_bytes = encode_ours(&src_rgb, w, h);
    eprintln!("  ours {} B", ours_bytes.len());

    eprintln!("[encode cjxl] ...");
    let cjxl_bytes = encode_cjxl(Path::new(SRC_PNG));
    eprintln!("  cjxl {} B", cjxl_bytes.len());

    eprintln!("[decode both] ...");
    let (dw, dh, ours_lin) = decode_jxl_linear(&ours_bytes);
    assert_eq!(dw, width);
    assert_eq!(dh, height);
    let (cw, ch, cjxl_lin) = decode_jxl_linear(&cjxl_bytes);
    assert_eq!(cw, width);
    assert_eq!(ch, height);

    // Source as both linear (for butteraugli) and sRGB u8 (for ssim2).
    let orig_lin_img = rgb_to_linear_img(&src_rgb, w, h);
    let orig_srgb_img = rgb_to_srgb_arr3(&src_rgb, w, h);

    // Encoded outputs need conversion to sRGB u8 for ssim2.
    let ours_srgb_packed = linear_planar_to_srgb_packed(&ours_lin, width, height);
    let cjxl_srgb_packed = linear_planar_to_srgb_packed(&cjxl_lin, width, height);

    // Match W44-173 region sizing: (dw / 3) & !31.
    let region_w = (width / 3) & !31;
    let region_h = (height / 3) & !31;

    eprintln!(
        "\nregion sizing: {}×{} (W44-173-compatible)",
        region_w, region_h
    );
    eprintln!(
        "\n{:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "region",
        "ours-src",
        "cjxl-src",
        "delta-ssim2",
        "ours-cjxl-ssim2",
        "ours-cjxl-bfly",
        "ours-cjxl-maxabs"
    );
    let mut total_delta = 0.0;
    for ry in 0..3 {
        for rx in 0..3 {
            let x0 = rx * region_w;
            let y0 = ry * region_h;
            let rw = if rx == 2 { width - x0 } else { region_w };
            let rh = if ry == 2 { height - y0 } else { region_h };

            let orig_lin_reg = {
                let mut out = Vec::with_capacity(rw * rh);
                let buf = orig_lin_img.buf();
                let stride = orig_lin_img.width();
                for y in y0..(y0 + rh) {
                    for x in x0..(x0 + rw) {
                        out.push(buf[y * stride + x]);
                    }
                }
                Img::new(out, rw, rh)
            };
            let orig_srgb_reg = {
                let mut out = Vec::with_capacity(rw * rh);
                let buf = orig_srgb_img.buf();
                let stride = orig_srgb_img.width();
                for y in y0..(y0 + rh) {
                    for x in x0..(x0 + rw) {
                        out.push(buf[y * stride + x]);
                    }
                }
                Img::new(out, rw, rh)
            };

            let ours_lin_reg = extract_lin(&ours_lin, width, x0, y0, rw, rh);
            let cjxl_lin_reg = extract_lin(&cjxl_lin, width, x0, y0, rw, rh);

            let ours_srgb_reg = region_srgb(&ours_srgb_packed, width, x0, y0, rw, rh);
            let cjxl_srgb_reg = region_srgb(&cjxl_srgb_packed, width, x0, y0, rw, rh);

            let ours_ssim2 =
                fast_ssim2::compute_ssimulacra2(orig_srgb_reg.as_ref(), ours_srgb_reg.as_ref())
                    .unwrap();
            let cjxl_ssim2 =
                fast_ssim2::compute_ssimulacra2(orig_srgb_reg.as_ref(), cjxl_srgb_reg.as_ref())
                    .unwrap();
            let ours_cjxl_ssim2 =
                fast_ssim2::compute_ssimulacra2(ours_srgb_reg.as_ref(), cjxl_srgb_reg.as_ref())
                    .unwrap();
            let ours_cjxl_bfly = butteraugli_linear(
                ours_lin_reg.as_ref(),
                cjxl_lin_reg.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|s| s.score as f64)
            .unwrap_or(0.0);

            let mut max_abs = 0.0_f32;
            for (a, b) in ours_lin_reg.buf().iter().zip(cjxl_lin_reg.buf().iter()) {
                max_abs = max_abs
                    .max((a.r - b.r).abs())
                    .max((a.g - b.g).abs())
                    .max((a.b - b.b).abs());
            }

            println!(
                "r[{},{}]  {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                ry,
                rx,
                ours_ssim2,
                cjxl_ssim2,
                ours_ssim2 - cjxl_ssim2,
                ours_cjxl_ssim2,
                ours_cjxl_bfly,
                max_abs
            );
            total_delta += ours_ssim2 - cjxl_ssim2;
            let _ = orig_lin_reg;
        }
    }
    println!("\nTotal SSIM2 delta (ours - cjxl): {:.4}", total_delta);
}
