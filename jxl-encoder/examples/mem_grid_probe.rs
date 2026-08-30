// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Peak-memory measurement probe over large corpus PPMs (P6), one encode
//! per process, for the size × effort × threads memory-recalibration grid
//! (`benchmarks/jxl_encode_mem_threads_2026-08-01.tsv`).
//!
//! Differs from `jxl-encoder-cli/examples/mem_probe.rs` (the 2026-06
//! calibration harness, PNG-based, kept for `scripts/mem_peak_calibrate.py`)
//! in three ways this grid needs:
//! - reads raw **PPM P6** directly (no `image` crate double-buffer — the
//!   108 MP corpus file is 324 MB; the input buffer must be materialized
//!   exactly once so process peak RSS ≈ input + encoder working set),
//! - attaches an explicit [`jxl_encoder::Limits`] budget (`max` =
//!   `u64::MAX`) so the pre-flight/budget can't reject or clamp the encode
//!   and the TRUE unconstrained peak is observable,
//! - prints the encoder's own `MemoryBudget` peak
//!   ([`jxl_encoder::EncodeStats::budget_peak_bytes`]) next to the process
//!   `VmHWM`, so `RSS − budget_peak` (the unguarded allocation mass) is a
//!   one-row join.
//!
//! Usage:
//!   mem_grid_probe <img.ppm> <lossy|lossless> <effort> <distance> <threads> [budget_bytes|max|default]
//!
//! `threads` is passed to `with_threads` (0 = ambient rayon pool, n ≥ 1 =
//! dedicated n-thread pool). `budget` defaults to `max`; `default` attaches
//! no Limits (the production path-aware soft caps apply).
//!
//! Prints one parseable line; on encode error, prints the line with
//! `ok=0 err=…` and exits 3 (so a driver records rejections as data).

use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

fn vmhwm_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

/// Minimal PPM P6 reader (8-bit, maxval 255). Header comments supported.
fn read_ppm_p6(path: &str) -> (u32, u32, Vec<u8>) {
    let f = File::open(path).expect("open ppm");
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 2];
    r.read_exact(&mut magic).expect("read magic");
    assert_eq!(&magic, b"P6", "not a PPM P6 file");
    // Parse width, height, maxval: integers separated by whitespace, with
    // '#'-to-newline comments. The single whitespace byte terminating
    // maxval is the last header byte; raw RGB data follows immediately.
    let mut vals = [0u64; 3];
    let mut n = 0usize;
    let mut cur: Option<u64> = None;
    let mut in_comment = false;
    while n < 3 {
        let mut b = [0u8; 1];
        r.read_exact(&mut b).expect("read header");
        let c = b[0];
        if in_comment {
            if c == b'\n' {
                in_comment = false;
            }
            continue;
        }
        match c {
            b'#' => in_comment = true,
            b'0'..=b'9' => cur = Some(cur.unwrap_or(0) * 10 + u64::from(c - b'0')),
            _ => {
                if let Some(v) = cur.take() {
                    vals[n] = v;
                    n += 1;
                }
            }
        }
    }
    let (w, h, maxval) = (vals[0] as u32, vals[1] as u32, vals[2]);
    assert_eq!(maxval, 255, "only 8-bit PPMs supported");
    let mut pixels = vec![0u8; w as usize * h as usize * 3];
    r.read_exact(&mut pixels).expect("read pixel data");
    (w, h, pixels)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!(
            "usage: mem_grid_probe <img.ppm> <lossy|lossless> <effort> <distance> <threads> \
             [budget_bytes|max|default]"
        );
        std::process::exit(2);
    }
    let (path, mode, effort, distance, threads) = (
        &a[1],
        &a[2],
        a[3].parse::<u8>().expect("effort"),
        a[4].parse::<f32>().expect("distance"),
        a[5].parse::<usize>().expect("threads"),
    );
    let budget_arg = a.get(6).map(String::as_str).unwrap_or("max");

    use jxl_encoder::{Limits, LosslessConfig, LossyConfig, PixelLayout};
    let (w, h, pixels) = read_ppm_p6(path);
    let is_lossless = mode == "lossless";

    let limits = match budget_arg {
        "default" => None,
        "max" => Some(Limits::new().with_max_memory_bytes(u64::MAX)),
        s => Some(Limits::new().with_max_memory_bytes(s.parse().expect("budget bytes"))),
    };

    // Model prediction for the same cell (thread-aware), in the same row.
    let est =
        jxl_encoder::estimate_encode_threaded(w, h, 3, false, is_lossless, effort, threads.max(1));
    let (est_typ, est_max) = est
        .map(|e| (e.peak_memory_bytes / 1024, e.peak_memory_bytes_max / 1024))
        .unwrap_or((0, 0));

    // Configs live in main's scope: `encode_request` borrows the config.
    let lossless_cfg = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(threads);
    let lossy_cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads);

    let baseline = vmhwm_kb();
    let t0 = Instant::now();
    let result = {
        let req = if is_lossless {
            lossless_cfg.encode_request(w, h, PixelLayout::Rgb8)
        } else {
            lossy_cfg.encode_request(w, h, PixelLayout::Rgb8)
        };
        let req = match limits.as_ref() {
            Some(l) => req.with_limits(l),
            None => req,
        };
        req.encode_with_stats(&pixels)
    };
    let wall = t0.elapsed();
    let peak = vmhwm_kb();

    match result {
        Ok(res) => {
            let stats = res.stats();
            println!(
                "w={} h={} mode={} effort={} distance={} threads={} budget={} \
                 vmhwm_base_kb={} vmhwm_peak_kb={} delta_kb={} budget_peak_kb={} \
                 wall_ms={:.1} out_bytes={} est_typ_kb={} est_max_kb={} ok=1",
                w,
                h,
                mode,
                effort,
                distance,
                threads,
                budget_arg,
                baseline,
                peak,
                peak.saturating_sub(baseline),
                stats.budget_peak_bytes() / 1024,
                wall.as_secs_f64() * 1000.0,
                stats.output_size(),
                est_typ,
                est_max,
            );
        }
        Err(e) => {
            println!(
                "w={} h={} mode={} effort={} distance={} threads={} budget={} \
                 vmhwm_base_kb={} vmhwm_peak_kb={} delta_kb={} budget_peak_kb=0 \
                 wall_ms={:.1} out_bytes=0 est_typ_kb={} est_max_kb={} ok=0 err={:?}",
                w,
                h,
                mode,
                effort,
                distance,
                threads,
                budget_arg,
                baseline,
                peak,
                peak.saturating_sub(baseline),
                wall.as_secs_f64() * 1000.0,
                est_typ,
                est_max,
                e.to_string(),
            );
            std::process::exit(3);
        }
    }
}
