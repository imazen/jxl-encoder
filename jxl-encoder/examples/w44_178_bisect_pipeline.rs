//! W44-178 Phase 2 — bisect the recon-pipeline by toggling gaborish and
//! EPF. If toggling either gaborish or EPF closes the SSIM2 deficit on
//! the right-column DCT64X64 region (or otherwise shifts where the
//! deficit lives), that pinpoints the stage to investigate.
//!
//! Phase 1.5 found: r[0,2]/r[1,2]/r[2,2] (right column) ours-cjxl
//! max-abs ≈ 0.0064-0.0080 linear RGB, yet ours-source SSIM2 is -7
//! worse than cjxl-source. This Phase 2 attempts to localize the
//! sub-pixel shift to a specific pipeline stage.
//!
//! IMPORTANT: this varies OUR encoder; we always compare against cjxl
//! at default settings.
//!
//! Cells tested (all e7 d=5 on clic_097cb426):
//!   gab=on,  epf=auto  (= default Zenjxl)
//!   gab=off, epf=auto
//!   gab=on,  epf=0     (disable EPF)
//!   gab=off, epf=0
//!
//! Output: stdout + benchmarks/w44_178_bisect_pipeline_2026-05-21.{tsv,meta}

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
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
fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn load_png(p: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(p).expect("open png");
    let rgb = img.to_rgb8();
    (rgb.as_raw().clone(), rgb.width(), rgb.height())
}

fn decode_linear(bytes: &[u8]) -> (usize, usize, Vec<f32>) {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).expect("read");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).expect("render");
    let fb = render.image_all_channels();
    (fb.width(), fb.height(), fb.buf().to_vec())
}

fn encode_ours(rgb: &[u8], w: u32, h: u32, gab: bool, epf_level: i8) -> Vec<u8> {
    LossyConfig::new(DISTANCE)
        .with_effort(EFFORT)
        .with_threads(1)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_gaborish(gab)
        .with_epf_level(epf_level)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode ours")
}

fn encode_cjxl(src_png: &Path) -> Vec<u8> {
    let tmp = std::env::temp_dir().join("w44_178_bisect_cjxl.jxl");
    let st = Command::new(cjxl_bin())
        .arg(src_png)
        .arg(&tmp)
        .args(["-e", &EFFORT.to_string()])
        .args(["-d", &DISTANCE.to_string()])
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .status()
        .expect("cjxl");
    assert!(st.success());
    let b = std::fs::read(&tmp).expect("read");
    let _ = std::fs::remove_file(&tmp);
    b
}

fn rgb_img(rgb: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
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
fn srgb_packed(rgb: &[u8], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w as usize, h as usize)
}
fn lin_planar_to_srgb_packed(lin: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(lin.len());
    for px in lin.chunks(3) {
        out.push(linear_to_srgb_u8(px[0]));
        out.push(linear_to_srgb_u8(px[1]));
        out.push(linear_to_srgb_u8(px[2]));
    }
    out
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
fn extract_srgb(
    packed: &[u8],
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
            out.push([packed[i], packed[i + 1], packed[i + 2]]);
        }
    }
    Img::new(out, rw, rh)
}

#[derive(Debug, Clone, Copy)]
struct CellSummary {
    bytes: usize,
    ssim2_global: f64,
    ssim2_right: f64,          // right column average (r[*,2])
    ssim2_right_max_diff: f64, // worst region in right col
    bfly_right_avg: f64,
    max_abs_right: f64, // max-abs in right column vs cjxl decoded
}

fn measure_cell(
    bytes: &[u8],
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    cjxl_lin: &[f32],
    cjxl_srgb_packed: &[u8],
    width: usize,
    height: usize,
) -> CellSummary {
    let (dw, dh, lin) = decode_linear(bytes);
    assert_eq!(dw, width);
    assert_eq!(dh, height);
    let srgb_packed = lin_planar_to_srgb_packed(&lin);

    let region_w = (width / 3) & !31;
    let region_h = (height / 3) & !31;

    // Global SSIM2 — use full image regions for proper computation
    let global_ssim2 = {
        let dec_srgb = srgb_packed_to_img(&srgb_packed, width, height);
        fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb.as_ref()).unwrap()
    };

    // Right column ssim2
    let mut right_ssim2 = [0.0; 3];
    let mut right_bfly = [0.0; 3];
    let mut max_abs_right = 0.0_f64;
    for ry in 0..3 {
        let x0 = 2 * region_w;
        let y0 = ry * region_h;
        let rw = width - x0;
        let rh = if ry == 2 { height - y0 } else { region_h };
        let dec_lin_reg = extract_lin(&lin, width, x0, y0, rw, rh);
        let dec_srgb_reg = extract_srgb(&srgb_packed, width, x0, y0, rw, rh);
        let orig_lin_reg = extract_lin_from_img(orig_lin, x0, y0, rw, rh);
        let orig_srgb_reg = extract_srgb_from_img(orig_srgb, x0, y0, rw, rh);

        right_ssim2[ry] =
            fast_ssim2::compute_ssimulacra2(orig_srgb_reg.as_ref(), dec_srgb_reg.as_ref()).unwrap();
        right_bfly[ry] = butteraugli_linear(
            orig_lin_reg.as_ref(),
            dec_lin_reg.as_ref(),
            &ButteraugliParams::default(),
        )
        .map(|s| s.score as f64)
        .unwrap_or(0.0);

        // Diff vs cjxl decoded for this region
        for y in y0..(y0 + rh) {
            for x in x0..(x0 + rw) {
                let i = (y * width + x) * 3;
                for ch in 0..3 {
                    let d = (lin[i + ch] - cjxl_lin[i + ch]).abs() as f64;
                    if d > max_abs_right {
                        max_abs_right = d;
                    }
                }
            }
        }
        let _ = (cjxl_srgb_packed,);
    }

    let r_avg = right_ssim2.iter().sum::<f64>() / 3.0;
    // max deficit within right col vs cjxl ssim2 reference (computed at cjxl baseline below)
    let r_min = right_ssim2.iter().cloned().fold(f64::INFINITY, f64::min);
    let r_max = right_ssim2
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    CellSummary {
        bytes: bytes.len(),
        ssim2_global: global_ssim2,
        ssim2_right: r_avg,
        ssim2_right_max_diff: r_max - r_min,
        bfly_right_avg: right_bfly.iter().sum::<f64>() / 3.0,
        max_abs_right,
    }
}

fn srgb_packed_to_img(packed: &[u8], w: usize, h: usize) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = packed.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w, h)
}

fn extract_lin_from_img(
    img: &Img<Vec<RGB<f32>>>,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
) -> Img<Vec<RGB<f32>>> {
    let stride = img.width();
    let buf = img.buf();
    let mut out = Vec::with_capacity(rw * rh);
    for y in y0..(y0 + rh) {
        for x in x0..(x0 + rw) {
            out.push(buf[y * stride + x]);
        }
    }
    Img::new(out, rw, rh)
}
fn extract_srgb_from_img(
    img: &Img<Vec<[u8; 3]>>,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
) -> Img<Vec<[u8; 3]>> {
    let stride = img.width();
    let buf = img.buf();
    let mut out = Vec::with_capacity(rw * rh);
    for y in y0..(y0 + rh) {
        for x in x0..(x0 + rw) {
            out.push(buf[y * stride + x]);
        }
    }
    Img::new(out, rw, rh)
}

fn main() {
    let (src_rgb, w, h) = load_png(Path::new(SRC_PNG));
    let width = w as usize;
    let height = h as usize;
    let orig_lin = rgb_img(&src_rgb, w, h);
    let orig_srgb = srgb_packed(&src_rgb, w, h);

    // cjxl baseline
    eprintln!("[encode cjxl baseline] ...");
    let cjxl_bytes = encode_cjxl(Path::new(SRC_PNG));
    let (_, _, cjxl_lin) = decode_linear(&cjxl_bytes);
    let cjxl_srgb_packed = lin_planar_to_srgb_packed(&cjxl_lin);
    let cjxl_summary = measure_cell(
        &cjxl_bytes,
        &orig_lin,
        &orig_srgb,
        &cjxl_lin,
        &cjxl_srgb_packed,
        width,
        height,
    );
    eprintln!("cjxl baseline:");
    eprintln!(
        "  bytes={}  global_ssim2={:.4}  right_ssim2={:.4}",
        cjxl_summary.bytes, cjxl_summary.ssim2_global, cjxl_summary.ssim2_right
    );

    let cells: &[(&str, bool, i8)] = &[
        ("default(gab=ON,epf=auto)", true, -1), // -1 = default (i.e. auto)
        ("gab=OFF,epf=auto", false, -1),
        ("gab=ON,epf=0", true, 0),
        ("gab=OFF,epf=0", false, 0),
    ];

    let bench_dir = PathBuf::from("benchmarks");
    let _ = std::fs::create_dir_all(&bench_dir);
    let tsv_path = bench_dir.join("w44_178_bisect_pipeline_2026-05-21.tsv");
    let meta_path = bench_dir.join("w44_178_bisect_pipeline_2026-05-21.meta");

    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tsv_path)
        .unwrap();
    writeln!(
        f,
        "# W44-178 Phase 2 — bisect pipeline on clic_097cb426 e7 d=5"
    )
    .unwrap();
    writeln!(f, "# cell\tbytes\tglobal_ssim2\tright_ssim2_avg\tright_ssim2_deficit_vs_cjxl\tright_bfly_avg\tmax_abs_right_vs_cjxl").unwrap();
    writeln!(
        f,
        "cjxl_baseline\t{}\t{:.4}\t{:.4}\t0.0\t{:.4}\t0.0",
        cjxl_summary.bytes,
        cjxl_summary.ssim2_global,
        cjxl_summary.ssim2_right,
        cjxl_summary.bfly_right_avg
    )
    .unwrap();

    println!(
        "\n{:<28} {:>8} {:>12} {:>14} {:>16} {:>14} {:>18}",
        "cell",
        "bytes",
        "global_ssim2",
        "right_ssim2",
        "right_deficit",
        "right_bfly",
        "max_abs_vs_cjxl"
    );

    for (name, gab, epf) in cells {
        let bytes = if *epf < 0 {
            // Don't call with_epf_level - use default
            LossyConfig::new(DISTANCE)
                .with_effort(EFFORT)
                .with_threads(1)
                .with_strategy(EncoderStrategy::Zenjxl)
                .with_gaborish(*gab)
                .encode(&src_rgb, w, h, PixelLayout::Rgb8)
                .expect("encode")
        } else {
            encode_ours(&src_rgb, w, h, *gab, *epf)
        };
        let s = measure_cell(
            &bytes,
            &orig_lin,
            &orig_srgb,
            &cjxl_lin,
            &cjxl_srgb_packed,
            width,
            height,
        );
        let deficit = s.ssim2_right - cjxl_summary.ssim2_right;
        println!(
            "{:<28} {:>8} {:>12.4} {:>14.4} {:>16.4} {:>14.4} {:>18.4}",
            name,
            s.bytes,
            s.ssim2_global,
            s.ssim2_right,
            deficit,
            s.bfly_right_avg,
            s.max_abs_right
        );
        writeln!(
            f,
            "{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            name,
            s.bytes,
            s.ssim2_global,
            s.ssim2_right,
            deficit,
            s.bfly_right_avg,
            s.max_abs_right
        )
        .unwrap();
    }

    {
        let mut m = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&meta_path)
            .unwrap();
        writeln!(
            m,
            "W44-178 Phase 2 — bisect gaborish + EPF on clic_097cb426 e7 d=5"
        )
        .unwrap();
        writeln!(m, "Date: 2026-05-21").unwrap();
        writeln!(m).unwrap();
        writeln!(
            m,
            "Hypothesis: the right-column DCT64X64 SSIM2 deficit (W44-173)"
        )
        .unwrap();
        writeln!(
            m,
            "is driven by a small sub-pixel shift introduced by either"
        )
        .unwrap();
        writeln!(
            m,
            "gaborish reconstruction or EPF filter selection. If toggling"
        )
        .unwrap();
        writeln!(
            m,
            "either stage closes/shifts the deficit, that is the bug site."
        )
        .unwrap();
        writeln!(m).unwrap();
        writeln!(m, "Reads: cjxl baseline + 4 cells (gab × epf cross).").unwrap();
        writeln!(m, "Output: {}", tsv_path.display()).unwrap();
    }
    eprintln!("\nWrote: {}", tsv_path.display());
}
