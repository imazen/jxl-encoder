// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Splines chunk-7 default-on re-bench, follow-on to chunk 5 (ddc02a02) +
// chunk 6 (d77c589d). With the chunk-6 bbox-span gate in place, re-asks the
// "should we flip default-on at e8+" question on a wider corpus:
//
//   * 10 CID22-512 photos at d=1.0 e8 — must stay byte-identical
//     (chunk-6 already verified 42-of-42 byte-identical at e7/e8, this is
//     a confirmation slice for the chunk-7 audit row).
//
//   * 5 power-line synthetics with photo-realistic backgrounds
//     (the integration-test image shape; the discriminator does NOT fire
//     on these so they hit the trial-encode + span gate). Counts as the
//     "real-content thin-feature" proxy.
//
//   * 3 hand-picked mixed photos with plausible thin features
//     (pexels photos passed via CLI args).
//
// Each cell reports off_bytes / on_bytes / delta_bytes / delta_pct at
// distance=1.0, effort=8 with default features (butteraugli-loop on).

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

fn encode_bytes(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8, auto: bool) -> usize {
    let mut cfg = LossyConfig::new(distance).with_effort(effort);
    if auto {
        cfg = cfg.with_auto_splines(true);
    }
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(rgb)
        .unwrap_or_else(|e| panic!("encode: {e}"))
        .len()
}

/// Power-line on a photo-realistic background. Matches the
/// `tests/auto_splines.rs::make_power_line_image` shape so the chunk-5
/// screenshot discriminator does NOT fire (stripes + diagonal ramp give
/// mask1x1 < SCREENSHOT_MEDIAN_MASK_THRESHOLD), forcing the candidate
/// through the span + trial-encode gate.
fn make_realistic_power_line_image(w: usize, h: usize, n_wires: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let ramp = 80 + (x * 60 / w) as i32 + (y * 30 / h) as i32;
            // Vertical stripes ortho to horizontal wires.
            let stripe = if x % 4 < 2 { 6 } else { -6 };
            let v = (ramp + stripe).clamp(0, 255) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    // n_wires bright horizontal wires spanning the full image width.
    for k in 0..n_wires {
        let y = ((k + 1) * h) / (n_wires + 1);
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
    println!(
        "category\tlabel\tdistance\teffort\tw\th\toff_bytes\ton_bytes\tdelta_bytes\tdelta_pct"
    );

    let distance = 1.0f32;
    let effort = 8u8;

    // ---- 5 power-line synthetics on photo-realistic backgrounds.
    // Vary widths/lengths per the chunk-7 task. Discriminator does NOT
    // fire (chunk-3 test image shape), so the chunk-6 span gate is the
    // gatekeeper. Width must be full image to clear span = 1.0× long dim.
    let synthetics: Vec<(String, u32, u32, Vec<u8>)> = vec![
        (
            "realistic_1wire_1024x256".into(),
            1024,
            256,
            make_realistic_power_line_image(1024, 256, 1),
        ),
        (
            "realistic_2wires_1024x256".into(),
            1024,
            256,
            make_realistic_power_line_image(1024, 256, 2),
        ),
        (
            "realistic_4wires_1024x512".into(),
            1024,
            512,
            make_realistic_power_line_image(1024, 512, 4),
        ),
        (
            "realistic_4wires_2048x512".into(),
            2048,
            512,
            make_realistic_power_line_image(2048, 512, 4),
        ),
        (
            "realistic_8wires_2048x1024".into(),
            2048,
            1024,
            make_realistic_power_line_image(2048, 1024, 8),
        ),
    ];
    for (label, w, h, rgb) in synthetics {
        let off = encode_bytes(&rgb, w, h, distance, effort, false);
        let on = encode_bytes(&rgb, w, h, distance, effort, true);
        let delta = on as i64 - off as i64;
        let pct = 100.0 * delta as f64 / off as f64;
        println!(
            "synthetic\t{label}\t{distance}\te{effort}\t{w}\t{h}\t{off}\t{on}\t{delta:+}\t{pct:+.3}"
        );
    }

    // ---- Real CID22 photos (10 expected) + mixed thin-feature photos
    // (3 expected) passed on the CLI. The caller prefixes the
    // image path with `<category>:` to set the column (e.g.
    // `photo:/path/to/1001682.png` or `mixed:/path/to/pexels-xyz.png`).
    for arg in std::env::args().skip(1) {
        let (category, path_str) = match arg.split_once(':') {
            Some((c, p)) => (c.to_string(), p.to_string()),
            None => ("photo".to_string(), arg),
        };
        let path = Path::new(&path_str);
        let (w, h, rgb) = load_rgb8(path);
        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("anon")
            .to_string();
        let off = encode_bytes(&rgb, w, h, distance, effort, false);
        let on = encode_bytes(&rgb, w, h, distance, effort, true);
        let delta = on as i64 - off as i64;
        let pct = 100.0 * delta as f64 / off as f64;
        println!(
            "{category}\t{label}\t{distance}\te{effort}\t{w}\t{h}\t{off}\t{on}\t{delta:+}\t{pct:+.3}"
        );
    }
}
