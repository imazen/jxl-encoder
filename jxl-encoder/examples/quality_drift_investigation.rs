//! Quality-targeting drift investigation: cjxl-rs vs cjxl at the SAME --distance.
//!
//! Question: at the same `--distance` setting, does cjxl-rs produce the same
//! perceptual quality (butteraugli / ssim2) as libjxl's cjxl? If our quality
//! is consistently worse for the same `-d`, then we have calibration drift —
//! a UX regression where users get different quality from `--distance 1.0`
//! than they expect.
//!
//! Methodology:
//! - 3 source images (smooth photo, detailed photo, screenshot)
//! - 4 distances: 0.5, 1.0, 2.0, 4.0
//! - Both encoders, fresh encode in this run
//! - Decode both with jxl-oxide in **linear sRGB** (per CLAUDE.md jxl-oxide rule)
//! - Measure butteraugli (Rust crate) + ssim2 (fast-ssim2 crate)
//! - Avoids butteraugli_main / PNG metadata bug (CLAUDE.md "PNG metadata causes
//!   bogus butteraugli scores")
//!
//! Run with:
//!   cargo run --example quality_drift_investigation --release \
//!     --manifest-path jxl-encoder/Cargo.toml
use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use rgb::RGB;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 4.0];

struct Source {
    label: &'static str,
    class: &'static str,
    path: PathBuf,
}

fn cjxl_path() -> String {
    std::env::var("CJXL_PATH")
        .unwrap_or_else(|_| String::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl"))
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

/// Encode via the public `LossyConfig` API path (mirrors what end users get).
/// This applies the gaborish gate (`distance > 0.5`) and other effort-aware
/// profile defaults; a direct `VarDctEncoder::new` call would NOT.
fn encode_with_cjxl_rs(rgb_u8: &[u8], w: u32, h: u32, d: f32, effort: u8) -> Result<Vec<u8>, String> {
    use jxl_encoder::{LossyConfig, PixelLayout};
    let cfg = LossyConfig::new(d).with_effort(effort);
    cfg.encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("cjxl-rs encode failed: {e:?}"))
}

fn encode_with_cjxl(src_path: &Path, d: f32, effort: u8) -> Result<Vec<u8>, String> {
    // Use a tmpfile per (image,d) so concurrent runs don't clobber.
    let stem = src_path.file_stem().unwrap().to_string_lossy();
    let tmp = std::env::temp_dir().join(format!("qdi_{}_d{}_e{}.jxl", stem, d, effort));
    let _ = std::fs::remove_file(&tmp);
    let status = Command::new(cjxl_path())
        .arg(src_path)
        .arg(&tmp)
        .args(["-d", &format!("{}", d)])
        .args(["-e", &format!("{}", effort)])
        .arg("--quiet")
        .status()
        .map_err(|e| format!("spawn cjxl: {e}"))?;
    if !status.success() {
        return Err(format!("cjxl exit {:?}", status.code()));
    }
    let bytes = std::fs::read(&tmp).map_err(|e| format!("read cjxl out: {e}"))?;
    let _ = std::fs::remove_file(&tmp);
    Ok(bytes)
}

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
}

fn measure(
    decoded_linear: &[f32],
    w: usize,
    h: usize,
    bytes_len: usize,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Measure {
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, w, h);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score)
        .unwrap_or(f64::NAN);
    let dec_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb, w, h);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);
    Measure {
        bytes: bytes_len,
        butteraugli: bfly,
        ssim2,
    }
}

fn main() {
    let corpus = std::env::var("CODEC_CORPUS_DIR")
        .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus"));
    let corpus = PathBuf::from(corpus);

    let sources = vec![
        Source {
            label: "clic2025/02809272 (smooth-photo, 1024x1024)",
            class: "smooth_photo",
            path: corpus.join(
                "clic2025/final-test/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
            ),
        },
        Source {
            label: "cid22/1025469 (detailed-photo, 512x512)",
            class: "detailed_photo",
            path: corpus.join("CID22/CID22-512/validation/1025469.png"),
        },
        Source {
            label: "gb82-sc/graph (screenshot, 796x481)",
            class: "screenshot",
            path: corpus.join("gb82-sc/graph.png"),
        },
    ];

    let effort_cjxl: u8 = std::env::var("CJXL_EFFORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let efforts_to_test: Vec<u8> = std::env::var("CJXL_RS_EFFORTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect::<Vec<u8>>()
        })
        .unwrap_or_else(|| vec![7]);

    println!(
        "# Quality drift investigation: cjxl-rs efforts={:?} vs cjxl (e{}) at SAME --distance",
        efforts_to_test, effort_cjxl
    );
    println!("# Both decoded with jxl-oxide in linear sRGB; metrics: Rust butteraugli + Rust ssim2");
    println!("# Cjxl version: {}", cjxl_version_string());
    println!(
        "image\tclass\tdistance\tencoder\tbytes\tbutteraugli\tssim2\tbfly_ratio_us_over_cjxl\tsize_ratio_us_over_cjxl\tssim2_delta_us_minus_cjxl"
    );

    for src in &sources {
        if !src.path.exists() {
            eprintln!("WARN: {} missing; skip", src.path.display());
            continue;
        }
        eprintln!("\n=== {} ===", src.label);
        let img = match image::open(&src.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  open fail: {e}");
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let (w, h) = (w as usize, h as usize);
        let rgb = img.to_rgb8();
        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                ]
            })
            .collect();
        let orig_pixels: Vec<RGB<f32>> = linear_rgb
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let orig_linear_img = Img::new(orig_pixels, w, h);
        let orig_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let orig_srgb_img = Img::new(orig_srgb, w, h);
        let params = ButteraugliParams::default();

        for &d in DISTANCES {
            // cjxl reference encode (one per d)
            let cj_bytes = match encode_with_cjxl(&src.path, d, effort_cjxl) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  d={}: cjxl failed: {e}", d);
                    continue;
                }
            };
            let cj_dec = decode_jxl_linear(&cj_bytes);
            let (cj_w, cj_h, cj_lin) = match cj_dec {
                Some(t) => t,
                None => {
                    eprintln!("  d={}: cjxl decode fail", d);
                    continue;
                }
            };
            assert_eq!(cj_w, w);
            assert_eq!(cj_h, h);
            let cj_m = measure(
                &cj_lin,
                w,
                h,
                cj_bytes.len(),
                &orig_linear_img,
                &orig_srgb_img,
                &params,
            );
            println!(
                "{}\t{}\t{}\tcjxl-e{}\t{}\t{:.6}\t{:.4}\t1.0000\t1.0000\t0.0000",
                src.label, src.class, d, effort_cjxl, cj_m.bytes, cj_m.butteraugli, cj_m.ssim2
            );

            for &eff in &efforts_to_test {
                // cjxl-rs encode at this effort (via public LossyConfig API)
                let rs_bytes = match encode_with_cjxl_rs(rgb.as_raw(), w as u32, h as u32, d, eff) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("  d={} eff={}: cjxl-rs failed: {e}", d, eff);
                        continue;
                    }
                };
                let rs_dec = decode_jxl_linear(&rs_bytes);
                let (rs_w, rs_h, rs_lin) = match rs_dec {
                    Some(t) => t,
                    None => {
                        eprintln!("  d={} eff={}: cjxl-rs decode fail", d, eff);
                        continue;
                    }
                };
                assert_eq!(rs_w, w);
                assert_eq!(rs_h, h);
                let rs_m = measure(
                    &rs_lin,
                    w,
                    h,
                    rs_bytes.len(),
                    &orig_linear_img,
                    &orig_srgb_img,
                    &params,
                );
                let bfly_ratio = if cj_m.butteraugli > 0.0 {
                    rs_m.butteraugli / cj_m.butteraugli
                } else {
                    f64::NAN
                };
                let size_ratio = if cj_m.bytes > 0 {
                    rs_m.bytes as f64 / cj_m.bytes as f64
                } else {
                    f64::NAN
                };
                let ssim2_delta = rs_m.ssim2 - cj_m.ssim2;
                eprintln!(
                    "  d={:.2} cjxl-rs e{}: bfly={:.4} ssim2={:.3} bytes={}  |  cjxl-e{} bfly={:.4} ssim2={:.3} bytes={}  | bfly_ratio={:.3} size_ratio={:.3} ssim2_delta={:.3}",
                    d, eff, rs_m.butteraugli, rs_m.ssim2, rs_m.bytes,
                    effort_cjxl, cj_m.butteraugli, cj_m.ssim2, cj_m.bytes,
                    bfly_ratio, size_ratio, ssim2_delta
                );
                println!(
                    "{}\t{}\t{}\tcjxl-rs-e{}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                    src.label, src.class, d, eff, rs_m.bytes, rs_m.butteraugli, rs_m.ssim2,
                    bfly_ratio, size_ratio, ssim2_delta
                );
            }
        }
    }
}

fn cjxl_version_string() -> String {
    Command::new(cjxl_path())
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok().or_else(|| String::from_utf8(o.stderr).ok()))
        .map(|s| s.lines().next().unwrap_or("?").to_string())
        .unwrap_or_else(|| String::from("?"))
}
