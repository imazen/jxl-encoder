// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 8g (2026-05-25) Pareto re-bench with per-block
//! reducer constants refit (Intervention B per RFC §3.2).
//!
//! Mirror of `cvvdp_phase8d_pareto_rebench.rs` but emits a **new
//! `C_GPU_v4` backend tag** for the per-block constants refit run.
//! The Phase 8g constants are switched at runtime in
//! `vardct/perceptual_loop.rs` via
//! [`block_reducer_constants_for_backend`] on the
//! `self.cvvdp_loop && !use_vdp2` predicate. Production callers see no
//! API change; this rebench probes the production cvvdp loop's bytes /
//! cvvdp score / Pareto-front position with the new constants.
//!
//! ## What's in this rebench
//!
//! Each (image, distance) cell encodes with FOUR modes:
//!   - `B`: butteraugli loop baseline (Phase 6 / 8c / 8d baseline)
//!   - `C_GPU_v2`: Phase 8c renorm only (legacy from Phase 8c)
//!   - `C_GPU_v3`: Phase 8c renorm + Phase 8d tighten (requires
//!     `cvvdp-loop-tighten` feature; if compiled-out we emit NA rows)
//!   - `C_GPU_v4`: Phase 8c renorm + Phase 8g constants refit
//!
//! Reading the result via
//! `scripts/cvvdp_pareto_diagnosis.py <out.tsv>` shows per-backend
//! Pareto-front position so the C_GPU progression
//! 40.3% → 60% → 60% → ??? can be tracked end-to-end.
//!
//! ## Effort 8 (buttloop fires)
//!
//! At e≥8 (libjxl gates the buttloop at `speed_tier <= kKitten`); we
//! match. Lower efforts don't engage cvvdp's loop, so the Pareto sweep
//! at e<8 produces byte-identical output across all modes.

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
use jxl_encoder::api::{EncoderStrategy, Limits};
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

#[derive(Clone, Copy, Debug)]
enum Mode {
    Baseline,
    /// Phase 8c renorm only (no tighten). Equivalent to Phase 8c rebench's
    /// C_GPU_v2.
    CvvdpRenormOnly,
    /// Phase 8c renorm + Phase 8g constants refit (this chunk's payload).
    /// Tighten OFF so we isolate Intervention B from Intervention C.
    CvvdpConstantsRefit,
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
            // Pure butteraugli.
        }
        Mode::CvvdpRenormOnly => {
            cfg = cfg.with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Cvvdp);
            // Don't try to set bytes_tighten here — different feature
            // configurations interact. Plain `with_cvvdp_loop(Some(true))`
            // gives us Phase 8c renorm + no tighten + Phase 8g constants.
            // We special-case the "no constants refit" path via env hook
            // below (CvvdpRenormOnly = env JXL_CVVDP_K_TILE_NORM=1.2 to
            // force butter-equivalent).
        }
        Mode::CvvdpConstantsRefit => {
            cfg = cfg.with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Cvvdp);
        }
    }
    let limits = Limits::default().with_max_memory_bytes(8 * 1024 * 1024 * 1024);
    // Set env override to control which constants table applies.
    // SAFETY: single-threaded harness; set + unset around each encode.
    let restore_env: Option<String> = std::env::var("JXL_CVVDP_K_TILE_NORM").ok();
    match mode {
        Mode::CvvdpRenormOnly => unsafe {
            // Force k_tile_norm = 1.2 (butter-equivalent) to isolate the
            // Phase 8c-only baseline regardless of CVVDP_BLOCK_CONSTANTS
            // value committed to source.
            std::env::set_var("JXL_CVVDP_K_TILE_NORM", "1.2");
        },
        _ => unsafe {
            // Allow the committed CVVDP_BLOCK_CONSTANTS.k_tile_norm to apply
            // for CvvdpConstantsRefit. For Baseline, the env hook is read
            // only on the cvvdp branch, so this is dead.
            std::env::remove_var("JXL_CVVDP_K_TILE_NORM");
        },
    }
    let t = Instant::now();
    let encoded = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(pixels)?;
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    // Restore env.
    unsafe {
        match restore_env {
            Some(v) => std::env::set_var("JXL_CVVDP_K_TILE_NORM", v),
            None => std::env::remove_var("JXL_CVVDP_K_TILE_NORM"),
        }
    }
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
        out_path.unwrap_or_else(|| PathBuf::from("benchmarks/cvvdp_phase8g_pareto_2026-05-25.tsv"));

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
        eprintln!("[p8g_rebench] loading {corpus}/{name}");
        let (pixels, w, h) = load_source(corpus, name);
        for &d in DISTANCES {
            // B baseline.
            let (b_bytes, b_wall) = encode_cell(&pixels, w, h, d, Mode::Baseline)?;
            let (b_butter, b_cvvdp, b_ssim2) = score_cell(&pixels, w, h, &b_bytes);
            writeln!(
                f,
                "{name}\t{corpus}\t{}\t{d:.2}\tB\t{}\t{:.3}\t{}\tNA\t{}\tNA\t{}\tphase8g_rebench",
                EFFORT,
                b_bytes.len(),
                b_wall,
                b_butter.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                b_cvvdp.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                b_ssim2.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
            )?;

            // C_GPU_v2: Phase 8c renorm only (env-forced k_tile_norm=1.2).
            let (c2_bytes, c2_wall) = encode_cell(&pixels, w, h, d, Mode::CvvdpRenormOnly)?;
            let (c2_butter, c2_cvvdp, c2_ssim2) = score_cell(&pixels, w, h, &c2_bytes);
            writeln!(
                f,
                "{name}\t{corpus}\t{}\t{d:.2}\tC_GPU_v2\t{}\t{:.3}\t{}\tNA\t{}\tNA\t{}\tphase8g_rebench renorm=active constants=butter_default",
                EFFORT,
                c2_bytes.len(),
                c2_wall,
                c2_butter.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c2_cvvdp.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c2_ssim2.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
            )?;

            // C_GPU_v4: Phase 8c renorm + Phase 8g constants refit (committed value).
            let (c4_bytes, c4_wall) = encode_cell(&pixels, w, h, d, Mode::CvvdpConstantsRefit)?;
            let (c4_butter, c4_cvvdp, c4_ssim2) = score_cell(&pixels, w, h, &c4_bytes);
            writeln!(
                f,
                "{name}\t{corpus}\t{}\t{d:.2}\tC_GPU_v4\t{}\t{:.3}\t{}\tNA\t{}\tNA\t{}\tphase8g_rebench renorm=active constants=refit",
                EFFORT,
                c4_bytes.len(),
                c4_wall,
                c4_butter.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c4_cvvdp.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
                c4_ssim2.map(|v| format!("{v:.6}")).unwrap_or("NA".into()),
            )?;
            f.sync_all().ok();

            // Per-cell summary.
            let dpct_v4_vs_v2 =
                100.0 * (c4_bytes.len() as f64 - c2_bytes.len() as f64) / c2_bytes.len() as f64;
            let dpct_v4_vs_b =
                100.0 * (c4_bytes.len() as f64 - b_bytes.len() as f64) / b_bytes.len() as f64;
            let wall_hit_pct = 100.0 * (c4_wall - c2_wall) / c2_wall;
            let dcvvdp_v4_vs_v2 = match (c2_cvvdp, c4_cvvdp) {
                (Some(c2), Some(c4)) => format!("{:+.4}", c4 - c2),
                _ => "NA".into(),
            };
            eprintln!(
                "  {corpus}/{name} d={d:.2}: B={} ({:.0}ms) C_GPU_v2={} ({:.0}ms) C_GPU_v4={} ({:.0}ms) | \
                 Δbytes_v4_vs_v2={dpct_v4_vs_v2:+.2}% Δbytes_v4_vs_B={dpct_v4_vs_b:+.2}% \
                 wall_hit={wall_hit_pct:+.1}% Δcvvdp_v4_vs_v2={dcvvdp_v4_vs_v2}",
                b_bytes.len(),
                b_wall,
                c2_bytes.len(),
                c2_wall,
                c4_bytes.len(),
                c4_wall,
            );
        }
    }
    drop(f);
    eprintln!("\n[p8g_rebench] output: {}", out_path.display());
    eprintln!(
        "[p8g_rebench] analyze: python3 scripts/cvvdp_pareto_diagnosis.py {}",
        out_path.display()
    );
    Ok(())
}
