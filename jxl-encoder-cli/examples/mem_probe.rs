//! Minimal library memory probe for `scripts/mem_peak_calibrate.py`.
//!
//! Loads a PNG, then measures the encoder's MARGINAL peak working set —
//! the `VmHWM` high-water delta across the `encode()` call only, so the
//! binary's static footprint and the input-buffer load (both present
//! before the encode) cancel out. That delta is what
//! `estimate_peak_memory_bytes` should predict (the encoder's own
//! allocations on top of the caller-provided pixels), unlike the CLI
//! whole-process RSS which is inflated by a ~126 MB binary/decode floor.
//!
//! Usage: mem_probe <png> <lossy|lossless> <effort> <distance> <8|16> [rgb|rgba]
//! Prints: `delta_kb=<n> peak_kb=<n> bytes=<encoded_len>`
//!
//! The optional 6th arg selects the channel layout. `rgba` builds a
//! 4-channel buffer whose alpha plane is the source's GREEN channel — a
//! deterministic, high-entropy (≈ worst-case) alpha, since the calibration
//! corpus is all-opaque. That measures the conservative extra working set
//! the encoder spends on an alpha extra-channel (modular alpha alongside
//! VarDCT, or the 4th channel in lossless), which is what a memory cap
//! should budget for.

use std::fs;

fn vmhwm_kb() -> u64 {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: mem_probe <png> <lossy|lossless> <effort> <distance> <8|16>");
        std::process::exit(2);
    }
    let (path, mode, effort, distance, depth) = (
        &a[1],
        &a[2],
        a[3].parse::<u8>().unwrap(),
        a[4].parse::<f32>().unwrap(),
        a[5].parse::<u8>().unwrap(),
    );
    let alpha = a.get(6).map(String::as_str).unwrap_or("rgb");

    use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};
    let img = image::open(path).expect("open png");
    let (w, h) = (img.width(), img.height());

    // Materialize the caller-provided pixel buffer BEFORE the baseline so it
    // is part of the load floor, not the measured encode delta. For `rgba`
    // the alpha plane is the green channel (deterministic high-entropy alpha).
    let (pixels, layout): (Vec<u8>, PixelLayout) = match (depth, alpha) {
        (16, "rgba") => {
            let buf = img.to_rgba16();
            let raw = buf.as_raw(); // RGBA interleaved
            let mut bytes = Vec::with_capacity(raw.len() * 2);
            for px in raw.chunks_exact(4) {
                let g = px[1];
                bytes.extend_from_slice(&px[0].to_ne_bytes());
                bytes.extend_from_slice(&px[1].to_ne_bytes());
                bytes.extend_from_slice(&px[2].to_ne_bytes());
                bytes.extend_from_slice(&g.to_ne_bytes()); // alpha := green
            }
            (bytes, PixelLayout::Rgba16)
        }
        (16, _) => {
            let buf = img.to_rgb16();
            let mut bytes = Vec::with_capacity(buf.as_raw().len() * 2);
            for &v in buf.as_raw() {
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
            (bytes, PixelLayout::Rgb16)
        }
        (_, "rgba") => {
            let mut buf = img.to_rgba8().into_raw();
            for px in buf.chunks_exact_mut(4) {
                px[3] = px[1]; // alpha := green
            }
            (buf, PixelLayout::Rgba8)
        }
        _ => (img.to_rgb8().into_raw(), PixelLayout::Rgb8),
    };

    let baseline = vmhwm_kb();
    let encoded = if mode == "lossless" {
        LosslessConfig::new()
            .with_effort(effort)
            .encode_request(w, h, layout)
            .encode(&pixels)
    } else {
        LossyConfig::new(distance)
            .with_effort(effort)
            .encode_request(w, h, layout)
            .encode(&pixels)
    };
    let peak = vmhwm_kb();
    let len = encoded.map(|d| d.len()).unwrap_or(0);
    println!(
        "delta_kb={} peak_kb={} bytes={}",
        peak.saturating_sub(baseline),
        peak,
        len
    );
}
