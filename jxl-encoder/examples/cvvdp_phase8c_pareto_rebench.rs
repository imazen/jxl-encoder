// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 8c (2026-05-25) Pareto re-bench.
//!
//! Re-runs a small 5-fixture × 4-distance subset of the Phase 6
//! tracking sweep with cvvdp + Phase 8c diffmap renormalization (scale
//! 0.018) active. Encodes each cell with both B (butteraugli baseline)
//! and C_GPU_v2 (cvvdp + renorm) backends, decodes via jxl-oxide, and
//! scores with butteraugli + cvvdp-gpu + SSIMULACRA2. Output is appended
//! to the Phase 6 tracking TSV format with a NEW backend tag
//! `C_GPU_v2` so the Phase 8a diagnosis script can compare the
//! pre-renorm `C_GPU` rows against the post-renorm `C_GPU_v2` rows
//! side by side.
//!
//! Each (image, distance) cell is encoded with:
//!   - B: butteraugli loop (Phase 6 baseline; identical to existing
//!     B rows for our 5 fixtures).
//!   - C_GPU_v2: cvvdp loop + Phase 8c renorm.
//!
//! Effort 8 (buttloop fires). The cvvdp path uses
//! `CVVDP_DIFFMAP_RENORM_SCALE = 0.018` by default; the env
//! `JXL_CVVDP_DIFFMAP_RENORM_SCALE=<float>` overrides for sweeps.
//!
//! Output schema matches Phase 6 (12 cols) so the
//! `scripts/cvvdp_pareto_diagnosis.py` script can ingest the new rows
//! directly.

#![cfg(all(
    feature = "cvvdp-loop",
    feature = "butteraugli-loop",
    feature = "ssim2-loop"
))]

use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;

const CID22_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

const FIXTURES: &[(&str, &str)] = &[
    ("CID22", "1025469.png"),
    ("CID22", "1418519.png"),
    ("CID22", "1189261.png"),
    ("GB82-SC", "terminal.png"),
    ("GB82-SC", "imac_g3.png"),
];

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 3.0];
const EFFORT: u8 = 8;

fn load_source(corpus: &str, name: &str) -> (Vec<u8>, u32, u32) {
    let p = match corpus {
        "CID22" => std::path::Path::new(CID22_DIR).join(name),
        "GB82-SC" => std::path::Path::new(GB82_SC_DIR).join(name),
        _ => panic!("unknown corpus {corpus}"),
    };
    let img = image::open(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.as_raw().clone(), w, h)
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    let w = fb.width();
    let h = fb.height();
    let buf: Vec<f32> = fb.buf().iter().copied().collect();
    Some((w, h, buf))
}

fn encode_cell(
    pixels: &[u8],
    w: u32,
    h: u32,
    d: f32,
    cvvdp_opt_in: bool,
) -> Result<(Vec<u8>, f64), Box<dyn std::error::Error>> {
    let cfg = LossyConfig::new(d)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_effort(EFFORT)
        .with_cvvdp_loop(if cvvdp_opt_in { Some(true) } else { None });
    let t = Instant::now();
    let encoded = cfg.encode(pixels, w, h, PixelLayout::Rgb8)?;
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok((encoded, wall_ms))
}

fn score_cell(
    src_rgb: &[u8],
    w: u32,
    h: u32,
    encoded: &[u8],
) -> (Option<f64>, Option<f64>, Option<f64>) {
    let (dec_w, dec_h, dec_linear_f32) = match decode_jxl_linear(encoded) {
        Some(x) => x,
        None => return (None, None, None),
    };
    if dec_w != w as usize || dec_h != h as usize {
        return (None, None, None);
    }
    // Convert source sRGB-u8 to linear-f32 packed (matching jxl-oxide output layout).
    let n_pix = (w as usize) * (h as usize);
    let mut src_linear = vec![0.0_f32; n_pix * 3];
    for i in 0..n_pix {
        for c in 0..3 {
            src_linear[i * 3 + c] = srgb_to_linear(src_rgb[i * 3 + c]);
        }
    }

    // butteraugli (CPU) — wrap into ImgRef<RGB<f32>>.
    let src_lin_pixels: Vec<RGB<f32>> = src_linear
        .chunks_exact(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_pixels: Vec<RGB<f32>> = dec_linear_f32
        .chunks_exact(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let src_img = Img::new(src_lin_pixels, w as usize, h as usize);
    let dec_img = Img::new(dec_lin_pixels, w as usize, h as usize);
    let butter_cpu = butteraugli_linear(
        src_img.as_ref(),
        dec_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()
    .map(|r| r.score as f64);

    // SSIM2 / cvvdp scoring needs sRGB-u8 of the DECODED image.
    let mut dec_srgb = vec![0_u8; n_pix * 3];
    for i in 0..n_pix {
        for c in 0..3 {
            let v = dec_linear_f32[i * 3 + c].clamp(0.0, 1.0);
            let s = if v <= 0.003_130_8 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            dec_srgb[i * 3 + c] = (s * 255.0).round() as u8;
        }
    }

    // cvvdp-gpu via compute_srgb_u8 (matches Phase 6 baseline scoring).
    let cvvdp_gpu_score = {
        use cvvdp_gpu::CvvdpOpaque;
        use cvvdp_gpu::params::CvvdpParams;
        let inner = std::panic::catch_unwind(|| {
            CvvdpOpaque::new(
                cvvdp_gpu::opaque::Backend::Cuda,
                w,
                h,
                CvvdpParams::default(),
            )
        });
        match inner {
            Ok(Ok(mut c)) => {
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    c.compute_srgb_u8(src_rgb, &dec_srgb)
                }));
                match res {
                    Ok(Ok(s)) => Some(s.value as f64),
                    _ => None,
                }
            }
            _ => None,
        }
    };

    // SSIMULACRA2 wants Img<[u8; 3]>.
    let src_srgb_pixels: Vec<[u8; 3]> = src_rgb
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let dec_srgb_pixels: Vec<[u8; 3]> = dec_srgb
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let src_srgb_img = Img::new(src_srgb_pixels, w as usize, h as usize);
    let dec_srgb_img = Img::new(dec_srgb_pixels, w as usize, h as usize);
    let ssim2 = fast_ssim2::compute_ssimulacra2(src_srgb_img.as_ref(), dec_srgb_img.as_ref()).ok();

    (butter_cpu, cvvdp_gpu_score, ssim2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut out_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--out" && i + 1 < args.len() {
            out_path = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else {
            eprintln!("Unknown arg: {}", args[i]);
            i += 1;
        }
    }
    let out_path =
        out_path.unwrap_or_else(|| PathBuf::from("benchmarks/cvvdp_phase8c_pareto_2026-05-25.tsv"));

    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&out_path)?;
    writeln!(
        f,
        "image\tcorpus\teffort\tdistance\tbackend\tbytes\twall_ms\tscore_butter_cpu\tscore_butter_gpu\tscore_cvvdp_gpu\tscore_cvvdp_cpu\tscore_ssim2\tnotes"
    )?;

    for (corpus, name) in FIXTURES {
        eprintln!("[p8c_rebench] loading {corpus}/{name}");
        let (pixels, w, h) = load_source(corpus, name);
        for &d in DISTANCES {
            // Pre-Phase 8c baseline: backend B (butteraugli).
            let (b_bytes, b_wall) = encode_cell(&pixels, w, h, d, false)?;
            let (b_butter, b_cvvdp, b_ssim2) = score_cell(&pixels, w, h, &b_bytes);
            writeln!(
                f,
                "{name}\t{corpus}\t{}\t{d:.2}\tB\t{}\t{:.3}\t{}\tNA\t{}\tNA\t{}\tphase8c_rebench",
                EFFORT,
                b_bytes.len(),
                b_wall,
                b_butter.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                b_cvvdp.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                b_ssim2.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
            )?;

            // Post-Phase 8c: backend C_GPU_v2 (cvvdp + renorm 0.660).
            let (c_bytes, c_wall) = encode_cell(&pixels, w, h, d, true)?;
            let (c_butter, c_cvvdp, c_ssim2) = score_cell(&pixels, w, h, &c_bytes);
            writeln!(
                f,
                "{name}\t{corpus}\t{}\t{d:.2}\tC_GPU_v2\t{}\t{:.3}\t{}\tNA\t{}\tNA\t{}\tphase8c_rebench renorm=active",
                EFFORT,
                c_bytes.len(),
                c_wall,
                c_butter.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c_cvvdp.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c_ssim2.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
            )?;
            f.sync_all().ok();

            let dpct = 100.0 * (c_bytes.len() as f64 - b_bytes.len() as f64) / b_bytes.len() as f64;
            let dcvvdp = match (b_cvvdp, c_cvvdp) {
                (Some(b), Some(c)) => format!("{:+.4}", c - b),
                _ => "NA".into(),
            };
            eprintln!(
                "  {corpus}/{name} d={d:.2}: B={} bytes ({:.0}ms), C_GPU_v2={} bytes ({:.0}ms), \
                 Δbytes%={dpct:.2}, Δcvvdp={dcvvdp}",
                b_bytes.len(),
                b_wall,
                c_bytes.len(),
                c_wall,
            );
        }
    }
    drop(f);
    eprintln!("\n[p8c_rebench] output: {}", out_path.display());
    Ok(())
}
