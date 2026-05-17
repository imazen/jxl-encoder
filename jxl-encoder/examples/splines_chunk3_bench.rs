// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Splines chunk-3 power-line baseline measurement.
//
// Reproduces the test in `tests/auto_splines.rs` and prints byte counts
// for: default off vs auto-splines on, plus a photo-like image.

use jxl_encoder::{LossyConfig, PixelLayout};

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

fn make_photo_like_image(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut seed = 0x9E3779B9u32;
    for y in 0..h {
        for x in 0..w {
            let ramp = (x * 200 / w + y * 50 / h) as i32;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((seed >> 24) as i32) - 128;
            let v = (ramp + noise / 16).clamp(0, 255) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    rgb
}

fn encode(rgb: &[u8], w: u32, h: u32, auto: bool) -> usize {
    let mut cfg = LossyConfig::new(1.0).with_effort(7);
    if auto {
        cfg = cfg.with_auto_splines(true);
    }
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(rgb)
        .expect("encode")
        .len()
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
    let (w, h) = (1024usize, 256usize);
    let pl = make_power_line_image(w, h);
    let off = encode(&pl, w as u32, h as u32, false);
    let on = encode(&pl, w as u32, h as u32, true);
    let delta = on as i64 - off as i64;
    println!(
        "power_line  {}x{} 1line:   off={} on={} delta={:+} (negative = chunk-3 wins)",
        w, h, off, on, delta
    );

    // Wider canvases amortize the per-image fixed splines-section header
    // and tend to expose realised savings more clearly.
    for &(ww, hh, n) in &[
        (2048usize, 512usize, 1usize),
        (1024usize, 512usize, 4usize),
        (2048usize, 1024usize, 8usize),
    ] {
        let img = make_multi_line_image(ww, hh, n);
        let off = encode(&img, ww as u32, hh as u32, false);
        let on = encode(&img, ww as u32, hh as u32, true);
        let delta = on as i64 - off as i64;
        println!(
            "power_line  {}x{} {}line: off={} on={} delta={:+}",
            ww, hh, n, off, on, delta
        );
    }

    let (pw, ph) = (256usize, 256usize);
    let photo = make_photo_like_image(pw, ph);
    let off_p = encode(&photo, pw as u32, ph as u32, false);
    let on_p = encode(&photo, pw as u32, ph as u32, true);
    let delta_p = on_p as i64 - off_p as i64;
    println!(
        "photo       {}x{}: off={} on={} delta={:+} (must be 0)",
        pw, ph, off_p, on_p, delta_p
    );
}
