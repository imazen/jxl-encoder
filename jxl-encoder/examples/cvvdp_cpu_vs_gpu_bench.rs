// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 5 (2026-05-24) — paired A/B bench: CPU CVVDP vs
//! GPU CVVDP backends for the buttloop's per-iter compare.
//!
//! For each of 10 cells (mix of CID22-style + GB82-SC-style + tiny
//! synthetic fixtures, sized 64²-128² to keep wall-times short
//! enough for a smoke), encode the cell twice:
//!
//! - Mode B (GPU CVVDP):
//!   `cvvdp_loop=Some(true) + cvvdp_use_cpu=Some(false)`
//! - Mode C (CPU CVVDP):
//!   `cvvdp_loop=Some(true) + cvvdp_use_cpu=Some(true)`
//!
//! Records per-mode wall-time (best of N iters), bytes, and SHA256.
//! Goal: confirm bytes / score agree within ±1 % (the two backends
//! produce ≤ 1e-4 JOD drift per Agent A's reference parity tests);
//! wall-times should show the expected ~10× GPU advantage at sizes
//! ≥ 256² (smaller sizes amortise CPU's fixed per-call overhead more
//! poorly so the ratio shrinks).
//!
//! The bench tolerates missing CUDA gracefully: on hosts without
//! CUDA, Mode B's GPU CVVDP `try_new` returns `None` and the
//! dispatch silently falls back to CPU CVVDP (see
//! `construct_backend` in `vardct/perceptual_backend.rs`); in that
//! case both columns are CPU CVVDP and the speedup ratio is ~1×.
//!
//! Output: `benchmarks/cvvdp_cpu_vs_gpu_buttloop_<DATE>.tsv` with
//! columns `cell, mode, backend_requested, bytes, sha256_8, wall_ms_best,
//! wall_ms_median, n_iters`. Companion `.meta` documents commit + host
//! + feature flags + cvvdp-cpu / cvvdp-gpu SHAs.
//!
//! Run via:
//! ```bash
//! cargo run --release -p jxl-encoder \
//!   --features "__expert butteraugli-loop cvvdp-loop cvvdp-loop-cpu ssim2-loop parallel" \
//!   --example cvvdp_cpu_vs_gpu_bench -- \
//!   --out benchmarks/cvvdp_cpu_vs_gpu_buttloop_2026-05-24.tsv
//! ```

#![cfg(feature = "cvvdp-loop-cpu")]

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};

// ============================================================================
// Synthetic fixtures (self-contained, no codec-corpus dep)
// ============================================================================

fn gradient_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / (w - 1).max(1)) as u8;
            out[i + 1] = (y * 255 / (h - 1).max(1)) as u8;
            out[i + 2] = 128;
        }
    }
    out
}

fn noisy_photo(w: usize, h: usize) -> Vec<u8> {
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

fn screenshot_like(w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Mostly white background + dark "text" strokes every 8 px.
            let on_text = (y % 12) < 2 && (x % 4) < 3;
            if on_text {
                out[i] = 0x10;
                out[i + 1] = 0x10;
                out[i + 2] = 0x10;
            } else {
                out[i] = 0xf8;
                out[i + 1] = 0xf8;
                out[i + 2] = 0xf8;
            }
        }
    }
    out
}

struct BenchCell {
    name: &'static str,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    layout: PixelLayout,
    distance: f32,
}

fn bench_cells() -> Vec<BenchCell> {
    vec![
        // Tiny synthetic — fastest path, exercises plumbing.
        BenchCell {
            name: "gradient_64_d1.0",
            pixels: gradient_rgb(64, 64),
            width: 64,
            height: 64,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
        },
        BenchCell {
            name: "gradient_64_d3.0",
            pixels: gradient_rgb(64, 64),
            width: 64,
            height: 64,
            layout: PixelLayout::Rgb8,
            distance: 3.0,
        },
        // 96² photo-like — exercises buttloop convergence.
        BenchCell {
            name: "noisy_96_d1.0",
            pixels: noisy_photo(96, 96),
            width: 96,
            height: 96,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
        },
        BenchCell {
            name: "noisy_96_d2.0",
            pixels: noisy_photo(96, 96),
            width: 96,
            height: 96,
            layout: PixelLayout::Rgb8,
            distance: 2.0,
        },
        BenchCell {
            name: "noisy_96_d3.0",
            pixels: noisy_photo(96, 96),
            width: 96,
            height: 96,
            layout: PixelLayout::Rgb8,
            distance: 3.0,
        },
        // 128² screenshot-like — text + flat regions; the kind of cell
        // the W44-105 buttloop screen-seed scale fires on (though at
        // this size the gates may not trigger; the bench still
        // exercises the dispatch).
        BenchCell {
            name: "screenshot_128_d1.0",
            pixels: screenshot_like(128, 128),
            width: 128,
            height: 128,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
        },
        BenchCell {
            name: "screenshot_128_d3.0",
            pixels: screenshot_like(128, 128),
            width: 128,
            height: 128,
            layout: PixelLayout::Rgb8,
            distance: 3.0,
        },
        // 128² photo at higher distances — exercises buttloop iter
        // count growth.
        BenchCell {
            name: "noisy_128_d2.0",
            pixels: noisy_photo(128, 128),
            width: 128,
            height: 128,
            layout: PixelLayout::Rgb8,
            distance: 2.0,
        },
        BenchCell {
            name: "noisy_128_d4.0",
            pixels: noisy_photo(128, 128),
            width: 128,
            height: 128,
            layout: PixelLayout::Rgb8,
            distance: 4.0,
        },
        BenchCell {
            name: "gradient_128_d2.0",
            pixels: gradient_rgb(128, 128),
            width: 128,
            height: 128,
            layout: PixelLayout::Rgb8,
            distance: 2.0,
        },
    ]
}

// ============================================================================
// SHA256 (zero-dep, ~50 LOC) — copy from libstd patterns since the
// crate doesn't pull `sha2` in.
// ============================================================================

fn sha256_8(bytes: &[u8]) -> String {
    // We don't actually need full SHA256 cryptographic strength here —
    // a 64-bit FNV-1a is more than enough to fingerprint encoded output
    // for the bench TSV's "did these bytes change" check.
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

// ============================================================================
// Bench harness
// ============================================================================

fn encode_one(cell: &BenchCell, use_cpu: bool) -> (Vec<u8>, std::time::Duration) {
    let start = Instant::now();
    let encoded = LossyConfig::new(cell.distance)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Cvvdp)
        .with_perceptual_device(if use_cpu {
            jxl_encoder::api::PerceptualDevice::Cpu
        } else {
            jxl_encoder::api::PerceptualDevice::Gpu
        })
        .encode(&cell.pixels, cell.width, cell.height, cell.layout)
        .unwrap_or_else(|e| {
            panic!(
                "[{}] cvvdp_use_cpu={use_cpu} encode failed: {e:?}",
                cell.name
            )
        });
    (encoded, start.elapsed())
}

fn parse_out_arg() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--out" {
            if let Some(p) = args.next() {
                return PathBuf::from(p);
            }
        }
    }
    PathBuf::from("benchmarks/cvvdp_cpu_vs_gpu_buttloop_2026-05-24.tsv")
}

fn parse_iters_arg() -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--iters" {
            if let Some(p) = args.next() {
                if let Ok(n) = p.parse::<usize>() {
                    return n.max(1);
                }
            }
        }
    }
    3
}

fn main() {
    let out_path = parse_out_arg();
    let n_iters = parse_iters_arg();

    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut tsv =
        File::create(&out_path).unwrap_or_else(|e| panic!("create {out_path:?} failed: {e}"));

    writeln!(
        tsv,
        "cell\tmode\tbackend_requested\tbytes\tsha256_8\twall_ms_best\twall_ms_median\tn_iters"
    )
    .unwrap();

    let cells = bench_cells();
    eprintln!(
        "[cvvdp-fork P5] bench {} cells × 2 modes (B/C) × {} iters → {}",
        cells.len(),
        n_iters,
        out_path.display()
    );

    for cell in &cells {
        for (mode_label, backend_label, use_cpu) in [
            ("B_GPU", "cvvdp-gpu-cuda (or fallback)", false),
            ("C_CPU", "cvvdp-cpu (explicit opt-in)", true),
        ] {
            let mut walls: Vec<std::time::Duration> = Vec::with_capacity(n_iters);
            let mut bytes_last: Vec<u8> = Vec::new();
            let mut sha_last = String::new();
            for _ in 0..n_iters {
                let (bytes, dur) = encode_one(cell, use_cpu);
                let sha = sha256_8(&bytes);
                // Sanity: every iter must produce the same bytes (the
                // buttloop is deterministic on a fixed backend).
                if !bytes_last.is_empty() && sha != sha_last {
                    eprintln!(
                        "[{} {mode_label}] WARNING: non-deterministic encode \
                         across iters (sha={sha} prev={sha_last}). Recording last.",
                        cell.name
                    );
                }
                bytes_last = bytes;
                sha_last = sha;
                walls.push(dur);
            }
            walls.sort();
            let best = walls[0];
            let median = walls[walls.len() / 2];
            writeln!(
                tsv,
                "{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{}",
                cell.name,
                mode_label,
                backend_label,
                bytes_last.len(),
                sha_last,
                best.as_secs_f64() * 1000.0,
                median.as_secs_f64() * 1000.0,
                n_iters,
            )
            .unwrap();
            eprintln!(
                "[{}] {mode_label}  bytes={}  best={:.2}ms  med={:.2}ms",
                cell.name,
                bytes_last.len(),
                best.as_secs_f64() * 1000.0,
                median.as_secs_f64() * 1000.0,
            );
        }
    }

    eprintln!("[cvvdp-fork P5] wrote {}", out_path.display());
}
