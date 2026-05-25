//! W44-AUDIT-5 Phase 3: 3-mode A/B/D bisect on codec_wiki e7 d=4 + 2 photo cells.
//!
//! Compares three CfL Newton modes on the same input × distance × effort:
//!  - **Mode A** (current Zenjxl): LS warm-start + LS-only refinement.
//!  - **Mode B** (`EncoderStrategy::Libjxl`): bit-exact libjxl Newton
//!    (eps=100, iters=20, `x=0` start, no LS fallback) — ALWAYS on EVERY image.
//!  - **Mode D** (Phase 3 NEW): Zenjxl with `cfl_pass1_screenshot_x0_start =
//!    true`. Per-image content-class router: when the ZenanalyzeProxies
//!    indicate `m3_colourfulness >= 80` (mixed-content screenshot like
//!    codec_wiki), the CfL Pass-1 + Pass-2 calls flip `libjxl_parity = true`
//!    for that image only — matching cjxl behaviour on screenshots while
//!    preserving the LS warm-start path for photos.
//!
//! Single-process-per-mode invocation pattern to avoid OnceLock state
//! collisions. Usage:
//!   MODE=A cargo run --release --example w44_audit_5_phase3_mode_bisect ...
//!   MODE=B cargo run ...
//!   MODE=D cargo run ...
//!
//! Outputs one TSV row per (cell, mode) to stdout — orchestrator
//! concatenates the three runs into a single bench artifact.
//!
//! Acceptance criteria (Phase 3 Step 3):
//!  - Mode D codec_wiki e7 d=4 SSIM2 ≥ Mode B − 0.3
//!  - Mode D codec_wiki e7 d=4 bytes within +5% of Mode A
//!  - Mode D 1418519 + 1531677 cells stay byte-identical OR within
//!    ±1.0 SSIM2 AND ±2% bytes of Mode A
//!
//! Run:
//!   cargo run --release \
//!     --features 'parallel butteraugli-loop ssim2-loop __internals __expert' \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     --example w44_audit_5_phase3_mode_bisect

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::{Img, ImgVec};
use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

const EFFORT: u8 = 7;
const DISTANCE: f32 = 4.0;

struct Cell {
    short: &'static str,
    path: &'static str,
    class: &'static str,
}

const CELLS: &[Cell] = &[
    Cell {
        short: "gb82_codec_wiki",
        path: "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        class: "screenshot",
    },
    Cell {
        short: "cid22_1418519",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        class: "photo",
    },
    Cell {
        short: "cid22_1531677",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        class: "photo",
    },
];

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
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
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
        out.push([
            linear_to_srgb_u8(lin[i * 3]),
            linear_to_srgb_u8(lin[i * 3 + 1]),
            linear_to_srgb_u8(lin[i * 3 + 2]),
        ]);
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
    let out_path = std::env::temp_dir().join(format!("w44_audit_5_p3_cjxl_{stem}_d{DISTANCE}.jxl"));
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

fn ssim2_metric(src: &ImgVec<[u8; 3]>, decoded: &ImgVec<[u8; 3]>) -> f32 {
    fast_ssim2::compute_ssimulacra2(src.as_ref(), decoded.as_ref()).unwrap_or(0.0) as f32
}
fn butteraugli_metric(src: &ImgVec<RGB<f32>>, decoded: &ImgVec<RGB<f32>>) -> f32 {
    let params = ButteraugliParams::default();
    butteraugli_linear(src.as_ref(), decoded.as_ref(), &params)
        .map(|s| s.score as f32)
        .unwrap_or(99.0)
}

fn main() {
    let mode = std::env::var("MODE").unwrap_or_else(|_| "D".to_string());
    let (strategy, mode_name): (EncoderStrategy, &str) = match mode.as_str() {
        "A" => (EncoderStrategy::Zenjxl, "A_zenjxl_baseline_ls_warm_start"),
        "B" => (EncoderStrategy::Libjxl, "B_libjxl_strategy_bit_exact"),
        "D" => {
            // Mode D: Zenjxl with the Phase 3 per-image route opt-in.
            // EncoderImprovementsCustom::default() carries Zenjxl values;
            // we flip ONLY `cfl_pass1_screenshot_x0_start = true`.
            let mut custom = EncoderImprovementsCustom::default();
            custom.cfl_pass1_screenshot_x0_start = true;
            (
                EncoderStrategy::Custom(Box::new(custom)),
                "D_zenjxl_phase3_per_image_m3_x0_route",
            )
        }
        other => panic!("MODE must be A | B | D, got {other}"),
    };

    eprintln!("[w44-audit-5-p3] MODE={mode} ({mode_name}) effort={EFFORT} distance={DISTANCE}");

    // TSV header (printed only once when MODE=A so orchestrator gets a single header).
    if mode == "A" {
        println!(
            "cell\tclass\tw\th\teffort\tdistance\tmode\tmode_name\tbytes\tssim2\tbutteraugli\tcjxl_bytes\tcjxl_ssim2\tcjxl_butteraugli\tdelta_bytes_pct_vs_cjxl\tdelta_ssim2_vs_cjxl"
        );
    }

    for cell in CELLS {
        let path = Path::new(cell.path);
        let (rgb, w, h) = match load_png(path) {
            Some(v) => v,
            None => {
                eprintln!("  skip (load failed): {}", cell.short);
                continue;
            }
        };
        let src_lin = rgb_u8_to_linear_img(&rgb, w, h);
        let src_srgb = rgb_u8_to_arr3_img(&rgb, w, h);

        let our_bytes = encode_with_strategy(&rgb, w, h, strategy.clone());
        let cjxl_bytes = match encode_cjxl(path) {
            Some(b) => b,
            None => {
                eprintln!("  cjxl encode failed for {}", cell.short);
                continue;
            }
        };

        let (dw_o, dh_o, dec_o) = decode_jxl_linear(&our_bytes)
            .unwrap_or_else(|| panic!("our decode failed for {}", cell.short));
        let (dw_c, dh_c, dec_c) = decode_jxl_linear(&cjxl_bytes)
            .unwrap_or_else(|| panic!("cjxl decode failed for {}", cell.short));
        assert_eq!((dw_o, dh_o), (w, h));
        assert_eq!((dw_c, dh_c), (w, h));

        let our_lin = linear_planar_to_img(&dec_o, w, h);
        let cjxl_lin = linear_planar_to_img(&dec_c, w, h);
        let our_srgb = linear_planar_to_srgb_arr3(&dec_o, w, h);
        let cjxl_srgb = linear_planar_to_srgb_arr3(&dec_c, w, h);

        let our_ssim2 = ssim2_metric(&src_srgb, &our_srgb);
        let cjxl_ssim2 = ssim2_metric(&src_srgb, &cjxl_srgb);
        let our_bfly = butteraugli_metric(&src_lin, &our_lin);
        let cjxl_bfly = butteraugli_metric(&src_lin, &cjxl_lin);

        let delta_bytes_pct =
            (our_bytes.len() as f32 - cjxl_bytes.len() as f32) / cjxl_bytes.len() as f32 * 100.0;
        let delta_ssim2 = our_ssim2 - cjxl_ssim2;

        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            cell.short,
            cell.class,
            w,
            h,
            EFFORT,
            DISTANCE,
            mode,
            mode_name,
            our_bytes.len(),
            our_ssim2,
            our_bfly,
            cjxl_bytes.len(),
            cjxl_ssim2,
            cjxl_bfly,
            delta_bytes_pct,
            delta_ssim2,
        );
    }
}
