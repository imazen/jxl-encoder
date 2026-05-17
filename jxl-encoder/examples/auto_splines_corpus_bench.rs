// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Auto-splines default-on bench: A/B compares `auto_splines off` vs
// `auto_splines on` across a curated corpus subset at e8 + e9. Reads
// PNG paths from CLI args (paths are absolute or relative to cwd) and
// prints a TSV-style row per (image, effort).
//
// Used for the "flip default-on at e8+" gate decision: photos must not
// regress, multi-line / power-line synthetics must net-save.

use std::path::Path;

use image::ImageReader;
use jxl_encoder::{LossyConfig, PixelLayout};

fn load_rgb8(path: &Path) -> (u32, u32, Vec<u8>) {
    let img = ImageReader::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
        .to_rgb8();
    (img.width(), img.height(), img.into_raw())
}

fn encode_bytes(rgb: &[u8], w: u32, h: u32, effort: u8, auto: bool) -> usize {
    let mut cfg = LossyConfig::new(1.0).with_effort(effort);
    if auto {
        cfg = cfg.with_auto_splines(true);
    }
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(rgb)
        .unwrap_or_else(|e| panic!("encode: {e}"))
        .len()
}

fn make_power_line_image(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![80u8; w * h * 3];
    let y = h / 2;
    for x in 4..w - 4 {
        let i = (y * w + x) * 3;
        rgb[i] = 240;
        rgb[i + 1] = 240;
        rgb[i + 2] = 240;
    }
    rgb
}

fn make_multi_line_image(w: usize, h: usize, n_lines: usize) -> Vec<u8> {
    let mut rgb = vec![80u8; w * h * 3];
    for k in 0..n_lines {
        let y = ((k + 1) * h) / (n_lines + 1);
        for x in 4..w - 4 {
            let i = (y * w + x) * 3;
            rgb[i] = 240;
            rgb[i + 1] = 240;
            rgb[i + 2] = 240;
        }
    }
    rgb
}

fn main() {
    println!("label\teffort\tw\th\toff_bytes\ton_bytes\tdelta_bytes\tdelta_pct");

    // Synthetic power-line / multi-line baselines (kept verbatim from
    // splines_chunk3_bench so the numbers are directly comparable).
    let synthetics: Vec<(String, u32, u32, Vec<u8>)> = vec![
        (
            "synth_1line_1024x256".into(),
            1024,
            256,
            make_power_line_image(1024, 256),
        ),
        (
            "synth_4lines_1024x512".into(),
            1024,
            512,
            make_multi_line_image(1024, 512, 4),
        ),
        (
            "synth_8lines_2048x1024".into(),
            2048,
            1024,
            make_multi_line_image(2048, 1024, 8),
        ),
    ];

    for (label, w, h, rgb) in synthetics {
        for effort in [7u8, 8u8, 9u8] {
            let off = encode_bytes(&rgb, w, h, effort, false);
            let on = encode_bytes(&rgb, w, h, effort, true);
            let delta = on as i64 - off as i64;
            let pct = 100.0 * delta as f64 / off as f64;
            println!("{label}\te{effort}\t{w}\t{h}\t{off}\t{on}\t{delta:+}\t{pct:+.3}");
        }
    }

    // Real images passed on the CLI.
    for arg in std::env::args().skip(1) {
        let path = Path::new(&arg);
        let (w, h, rgb) = load_rgb8(path);
        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("anon")
            .to_string();
        for effort in [7u8, 8u8, 9u8] {
            let off = encode_bytes(&rgb, w, h, effort, false);
            let on = encode_bytes(&rgb, w, h, effort, true);
            let delta = on as i64 - off as i64;
            let pct = 100.0 * delta as f64 / off as f64;
            println!("{label}\te{effort}\t{w}\t{h}\t{off}\t{on}\t{delta:+}\t{pct:+.3}");
        }
    }
}
