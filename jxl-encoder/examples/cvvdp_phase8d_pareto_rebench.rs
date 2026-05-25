// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 8d (2026-05-25) Pareto re-bench with bytes-tighten
//! exit pass.
//!
//! Mirror of `cvvdp_phase8c_pareto_rebench.rs` (which captured the
//! Phase 8c diffmap-renormalization-only baseline at 60% Pareto-front).
//! Phase 8d adds the post-convergence bytes-tighten exit pass on top of
//! 8c's renorm.
//!
//! Each (image, distance) cell is encoded with TWO backends:
//!   - B: butteraugli loop (baseline; identical to Phase 6/8c B rows).
//!   - C_GPU_v3: cvvdp loop + Phase 8c renorm + Phase 8d bytes-tighten.
//!
//! The backend tag `C_GPU_v3` is NEW — the Pareto-diagnosis script's
//! aggregation is per-backend so `C_GPU_v3` is independently
//! computable from `C_GPU` (pre-8c) and `C_GPU_v2` (post-8c, no
//! tighten). Compose all 3 TSVs for an end-to-end Phase 8 progression.
//!
//! Effort 8 (buttloop fires). The cvvdp path uses
//! `CVVDP_DIFFMAP_RENORM_SCALE = 0.018` (Phase 8c production constant)
//! and the Phase 8d tighten pass is default-on inside the
//! `cvvdp-loop-tighten` cargo feature (which this example requires).

#![cfg(all(
    feature = "cvvdp-loop",
    feature = "cvvdp-loop-tighten",
    feature = "butteraugli-loop",
    feature = "ssim2-loop"
))]

use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, Limits};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;

const CID22_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

// Phase 8c fixture list — same 5 fixtures so the C_GPU vs C_GPU_v2 vs
// C_GPU_v3 progression is comparable across phases.
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

/// Modes:
///   - `Baseline` → butteraugli loop
///   - `CvvdpNoTighten` → cvvdp loop, Phase 8c renorm only, no tighten
///   - `CvvdpFull` → cvvdp loop, Phase 8c renorm + Phase 8d tighten
#[derive(Clone, Copy, Debug)]
enum Mode {
    Baseline,
    CvvdpNoTighten,
    CvvdpFull,
}

fn encode_cell(
    pixels: &[u8],
    w: u32,
    h: u32,
    d: f32,
    mode: Mode,
) -> Result<(Vec<u8>, f64), Box<dyn std::error::Error>> {
    let mut cfg = LossyConfig::new(d)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_effort(EFFORT);
    match mode {
        Mode::Baseline => {
            // butteraugli loop, no cvvdp anything.
        }
        Mode::CvvdpNoTighten => {
            cfg = cfg
                .with_cvvdp_loop(Some(true))
                .with_cvvdp_bytes_tighten(Some(false));
        }
        Mode::CvvdpFull => {
            cfg = cfg
                .with_cvvdp_loop(Some(true))
                .with_cvvdp_bytes_tighten(Some(true));
        }
    }
    // Phase 8d v3 (2026-05-25): bump the memory budget for large
    // screenshots. imac_g3 d=2 with the tighten pass active exceeded
    // the 2 GiB default; the tighten pass allocates fresh
    // `reconstruct_xyb` planes per probe iter, so a 12 MP-class
    // screenshot easily blows past the default cap with 5 probe iters.
    // Lifting to 8 GiB is plenty for the test corpus.
    let limits = Limits::default().with_max_memory_bytes(8 * 1024 * 1024 * 1024);
    let t = Instant::now();
    let encoded = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(pixels)?;
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
    let n_pix = (w as usize) * (h as usize);
    let mut src_linear = vec![0.0_f32; n_pix * 3];
    for i in 0..n_pix {
        for c in 0..3 {
            src_linear[i * 3 + c] = srgb_to_linear(src_rgb[i * 3 + c]);
        }
    }

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
        out_path.unwrap_or_else(|| PathBuf::from("benchmarks/cvvdp_phase8d_pareto_2026-05-25.tsv"));

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
        eprintln!("[p8d_rebench] loading {corpus}/{name}");
        let (pixels, w, h) = load_source(corpus, name);
        for &d in DISTANCES {
            // Baseline B (butteraugli).
            let (b_bytes, b_wall) = encode_cell(&pixels, w, h, d, Mode::Baseline)?;
            let (b_butter, b_cvvdp, b_ssim2) = score_cell(&pixels, w, h, &b_bytes);
            writeln!(
                f,
                "{name}\t{corpus}\t{}\t{d:.2}\tB\t{}\t{:.3}\t{}\tNA\t{}\tNA\t{}\tphase8d_rebench",
                EFFORT,
                b_bytes.len(),
                b_wall,
                b_butter.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                b_cvvdp.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                b_ssim2.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
            )?;

            // C_GPU_v2: Phase 8c renorm, no tighten (Phase 8c reference).
            let (c2_bytes, c2_wall) = encode_cell(&pixels, w, h, d, Mode::CvvdpNoTighten)?;
            let (c2_butter, c2_cvvdp, c2_ssim2) = score_cell(&pixels, w, h, &c2_bytes);
            writeln!(
                f,
                "{name}\t{corpus}\t{}\t{d:.2}\tC_GPU_v2\t{}\t{:.3}\t{}\tNA\t{}\tNA\t{}\tphase8d_rebench renorm=active tighten=off",
                EFFORT,
                c2_bytes.len(),
                c2_wall,
                c2_butter.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c2_cvvdp.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c2_ssim2.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
            )?;

            // C_GPU_v3: Phase 8c renorm + Phase 8d tighten (this chunk).
            let (c3_bytes, c3_wall) = encode_cell(&pixels, w, h, d, Mode::CvvdpFull)?;
            let (c3_butter, c3_cvvdp, c3_ssim2) = score_cell(&pixels, w, h, &c3_bytes);
            writeln!(
                f,
                "{name}\t{corpus}\t{}\t{d:.2}\tC_GPU_v3\t{}\t{:.3}\t{}\tNA\t{}\tNA\t{}\tphase8d_rebench renorm=active tighten=active",
                EFFORT,
                c3_bytes.len(),
                c3_wall,
                c3_butter.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c3_cvvdp.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c3_ssim2.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
            )?;
            f.sync_all().ok();

            // Per-cell summary: bytes deltas + wall hit.
            let dpct_v3_vs_v2 =
                100.0 * (c3_bytes.len() as f64 - c2_bytes.len() as f64) / c2_bytes.len() as f64;
            let dpct_v3_vs_b =
                100.0 * (c3_bytes.len() as f64 - b_bytes.len() as f64) / b_bytes.len() as f64;
            let wall_hit_pct = 100.0 * (c3_wall - c2_wall) / c2_wall;
            let dcvvdp_v3_vs_v2 = match (c2_cvvdp, c3_cvvdp) {
                (Some(c2), Some(c3)) => format!("{:+.4}", c3 - c2),
                _ => "NA".into(),
            };
            eprintln!(
                "  {corpus}/{name} d={d:.2}: B={} ({:.0}ms) C_GPU_v2={} ({:.0}ms) C_GPU_v3={} ({:.0}ms) | \
                 Δbytes_v3_vs_v2={dpct_v3_vs_v2:+.2}% Δbytes_v3_vs_B={dpct_v3_vs_b:+.2}% \
                 wall_hit={wall_hit_pct:+.1}% Δcvvdp_v3_vs_v2={dcvvdp_v3_vs_v2}",
                b_bytes.len(),
                b_wall,
                c2_bytes.len(),
                c2_wall,
                c3_bytes.len(),
                c3_wall,
            );
        }
    }
    drop(f);
    eprintln!("\n[p8d_rebench] output: {}", out_path.display());
    eprintln!(
        "[p8d_rebench] analyze: python3 scripts/cvvdp_pareto_diagnosis.py {}",
        out_path.display()
    );
    Ok(())
}
