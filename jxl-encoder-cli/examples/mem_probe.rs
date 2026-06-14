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
//! Usage: mem_probe <png> <lossy|lossless> <effort> <distance> <8|16>
//! Prints: `delta_kb=<n> peak_kb=<n> bytes=<encoded_len>`

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

    use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};
    let img = image::open(path).expect("open png");
    let (w, h) = (img.width(), img.height());

    // Materialize the caller-provided pixel buffer BEFORE the baseline so it
    // is part of the load floor, not the measured encode delta.
    let (pixels, layout): (Vec<u8>, PixelLayout) = if depth == 16 {
        let buf = img.to_rgb16();
        let mut bytes = Vec::with_capacity(buf.as_raw().len() * 2);
        for &v in buf.as_raw() {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        (bytes, PixelLayout::Rgb16)
    } else {
        (img.to_rgb8().into_raw(), PixelLayout::Rgb8)
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
