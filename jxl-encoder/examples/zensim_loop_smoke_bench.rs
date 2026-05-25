// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! zensim-fork Phase 4 (2026-05-25) 20-cell smoke bench — produces the
//! `benchmarks/zensim_loop_smoke_<DATE>.tsv` referenced in the Phase 4
//! brief Step 6.
//!
//! For each of 5 fixtures × 4 distances {0.5, 1.0, 2.0, 3.0}:
//! - Encode with `PerceptualMetric::Butteraugli` (default) → `backend = B`
//! - Encode with `PerceptualMetric::Zensim + PerceptualDevice::Cpu` →
//!   `backend = Z_CPU`
//!
//! Schema mirrors the master tracking TSV
//! (`benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`) so future
//! agents can merge the rows into the master tracking file when
//! `score_zensim_{cpu,gpu}` columns are wired into the scoring scaffold.
//!
//! The Z_GPU backend is intentionally NOT exercised here: the
//! `zensim-loop-gpu` feature requires CUDA at runtime and the Phase 4
//! deliverable scope is `Z_CPU initially; Z_GPU optional`. Phase 6's
//! 6-backend tracking sweep is where the GPU backend gets the full
//! coverage.
//!
//! Run via:
//! ```bash
//! cargo run --release -p jxl-encoder \
//!   --features "__expert butteraugli-loop zensim-loop ssim2-loop parallel" \
//!   --example zensim_loop_smoke_bench -- \
//!   --out benchmarks/zensim_loop_smoke_2026-05-25.tsv
//! ```

#![cfg(feature = "zensim-loop")]

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::{EncoderStrategy, PerceptualDevice, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

fn gradient_rgb_64x64() -> Vec<u8> {
    let (w, h) = (64usize, 64usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 128;
        }
    }
    out
}

fn diagonal_stripes_128x128() -> Vec<u8> {
    let (w, h) = (128usize, 128usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let stripe = ((x + y) / 8) & 1;
            if stripe == 0 {
                out[i] = (x as u8).wrapping_mul(2);
                out[i + 1] = (y as u8).wrapping_mul(2);
                out[i + 2] = 64;
            } else {
                out[i] = 200;
                out[i + 1] = ((x ^ y) as u8).wrapping_mul(3);
                out[i + 2] = 200;
            }
        }
    }
    out
}

fn noisy_photo_96x96() -> Vec<u8> {
    let (w, h) = (96usize, 96usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let cx = w as i32 / 2;
            let cy = h as i32 / 2;
            let r = (((x as i32 - cx).pow(2) + (y as i32 - cy).pow(2)) as f32).sqrt()
                / ((cx.pow(2) + cy.pow(2)) as f32).sqrt();
            let base = 128.0 + 100.0 * (1.0 - r);
            let noise = (((x * 13) ^ (y * 7)) & 0x1f) as f32 - 15.5;
            let v = (base + noise).clamp(0.0, 255.0) as u8;
            out[i] = v;
            out[i + 1] = v.saturating_sub(8);
            out[i + 2] = v.saturating_add(8);
        }
    }
    out
}

fn flat_grey_32x32() -> Vec<u8> {
    let (w, h) = (32usize, 32usize);
    vec![128u8; w * h * 3]
}

fn checker_64x64() -> Vec<u8> {
    let (w, h) = (64usize, 64usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let on = ((x / 8) ^ (y / 8)) & 1 == 0;
            let v = if on { 200 } else { 50 };
            out[i] = v;
            out[i + 1] = v;
            out[i + 2] = v;
        }
    }
    out
}

struct Fixture {
    name: &'static str,
    corpus: &'static str,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    layout: PixelLayout,
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            name: "gradient_64",
            corpus: "synthetic",
            pixels: gradient_rgb_64x64(),
            width: 64,
            height: 64,
            layout: PixelLayout::Rgb8,
        },
        Fixture {
            name: "diagonal_128",
            corpus: "synthetic",
            pixels: diagonal_stripes_128x128(),
            width: 128,
            height: 128,
            layout: PixelLayout::Rgb8,
        },
        Fixture {
            name: "noisy_96",
            corpus: "synthetic",
            pixels: noisy_photo_96x96(),
            width: 96,
            height: 96,
            layout: PixelLayout::Rgb8,
        },
        Fixture {
            name: "flat_grey_32",
            corpus: "synthetic",
            pixels: flat_grey_32x32(),
            width: 32,
            height: 32,
            layout: PixelLayout::Rgb8,
        },
        Fixture {
            name: "checker_64",
            corpus: "synthetic",
            pixels: checker_64x64(),
            width: 64,
            height: 64,
            layout: PixelLayout::Rgb8,
        },
    ]
}

fn encode_cell(
    pixels: &[u8],
    w: u32,
    h: u32,
    layout: PixelLayout,
    d: f32,
    use_zensim: bool,
    effort: u8,
) -> Result<(Vec<u8>, f64), Box<dyn std::error::Error + Send + Sync>> {
    let cfg = LossyConfig::new(d)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_effort(effort)
        .with_perceptual_metric(if use_zensim {
            PerceptualMetric::Zensim
        } else {
            PerceptualMetric::Butteraugli
        })
        // Force CPU when using zensim so the bench doesn't require CUDA.
        .with_perceptual_device(if use_zensim {
            PerceptualDevice::Cpu
        } else {
            PerceptualDevice::Auto
        });
    let t = Instant::now();
    let encoded = cfg.encode(pixels, w, h, layout)?;
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok((encoded, wall_ms))
}

fn git_short_sha() -> String {
    if let Ok(out) = std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return s;
        }
    }
    if let Ok(out) = std::process::Command::new("jj")
        .args(["log", "-n", "1", "--no-graph", "-T", "commit_id.short()"])
        .output()
        && out.status.success()
    {
        return String::from_utf8_lossy(&out.stdout).trim().to_string();
    }
    "unknown".to_string()
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
        out_path.unwrap_or_else(|| PathBuf::from("benchmarks/zensim_loop_smoke_2026-05-25.tsv"));

    let sha = git_short_sha();
    eprintln!("[zensim_loop_smoke_bench] encoder_sha={sha}");
    eprintln!("[zensim_loop_smoke_bench] output: {}", out_path.display());

    let mut f = File::create(&out_path)?;
    writeln!(
        f,
        "image\tcorpus\teffort\tdistance\tbackend\tbytes\twall_ms\tscore_butter_cpu\tscore_butter_gpu\tscore_cvvdp_gpu\tscore_cvvdp_cpu\tscore_zensim_cpu\tscore_zensim_gpu\tscore_ssim2\tnotes"
    )?;

    // 5 fixtures × 4 distances = 20 cells × 2 backends (B + Z_CPU) = 40 rows.
    let distances = [0.5_f32, 1.0, 2.0, 3.0];
    // Effort 8 ≥ kKitten — the buttloop ACTUALLY runs at this effort.
    let effort = 8;
    let mut row_count = 0;
    let mut z_cpu_rows = 0;
    for fx in fixtures() {
        let Fixture {
            name,
            corpus,
            pixels,
            width: w,
            height: h,
            layout,
        } = fx;
        for &d in &distances {
            // B = butteraugli baseline.
            let (b_bytes, b_wall) = encode_cell(&pixels, w, h, layout, d, false, effort)?;
            writeln!(
                f,
                "{name}\t{corpus}\t{effort}\t{d:.2}\tB\t{}\t{:.3}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tencoder_sha={sha} zensim_phase4_smoke",
                b_bytes.len(),
                b_wall
            )?;
            row_count += 1;

            // Z_CPU = zensim opt-in (CPU backend).
            let (z_bytes, z_wall) = encode_cell(&pixels, w, h, layout, d, true, effort)?;
            writeln!(
                f,
                "{name}\t{corpus}\t{effort}\t{d:.2}\tZ_CPU\t{}\t{:.3}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tencoder_sha={sha} zensim_phase4_smoke metric=Zensim device=Cpu",
                z_bytes.len(),
                z_wall
            )?;
            row_count += 1;
            z_cpu_rows += 1;

            eprintln!(
                "  cell={name} d={d:.2}: B={} bytes ({:.1}ms), Z_CPU={} bytes ({:.1}ms), Δ%={:.2}",
                b_bytes.len(),
                b_wall,
                z_bytes.len(),
                z_wall,
                100.0 * (z_bytes.len() as f64 - b_bytes.len() as f64) / b_bytes.len() as f64
            );
        }
    }
    drop(f);
    eprintln!(
        "[zensim_loop_smoke_bench] wrote {row_count} rows ({z_cpu_rows} Z_CPU + {} B) → {}",
        row_count - z_cpu_rows,
        out_path.display()
    );
    Ok(())
}
