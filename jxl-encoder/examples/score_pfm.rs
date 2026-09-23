// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Score a PFM dump (or an .f32 planar-linear recon from the JXL_AQDBG_DUMP
//! hook) against a source PNG with the Rust butteraugli metric, using the
//! same `intensity_target = 80` the buttloop uses for SDR sRGB input.
//!
//! usage:
//!   score_pfm <source.png> <label>=<file> [...]
//!   where <file> is .pfm (sRGB-gamma display encoding, as dumped by the
//!   instrumented cjxl AQDBG hook) or .f32 (planar linear RGB with a
//!   [w,h,3] i32 header, as dumped by our JXL_AQDBG_DUMP hook).

#![cfg(feature = "butteraugli-loop")]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use rgb::RGB;

fn read_pfm(path: &str) -> (usize, usize, Vec<RGB<f32>>) {
    let d = std::fs::read(path).expect("read pfm");
    // Header: "PF\n<w> <h>\n<scale>\n" then w*h*3 little-endian f32,
    // rows bottom-up.
    let mut pos = 0usize;
    let mut line = || {
        let end = d[pos..].iter().position(|&b| b == b'\n').unwrap() + pos;
        let s = std::str::from_utf8(&d[pos..end])
            .unwrap()
            .trim()
            .to_string();
        pos = end + 1;
        s
    };
    assert_eq!(line(), "PF");
    let dims: Vec<usize> = line().split(' ').map(|t| t.parse().unwrap()).collect();
    let (w, h) = (dims[0], dims[1]);
    let scale: f32 = line().parse().unwrap();
    assert!(scale < 0.0, "expected little-endian PFM");
    let mut img = vec![RGB::new(0.0f32, 0.0, 0.0); w * h];
    for y in (0..h).rev() {
        for x in 0..w {
            for c in 0..3 {
                let v = f32::from_le_bytes([d[pos], d[pos + 1], d[pos + 2], d[pos + 3]]);
                pos += 4;
                let p = &mut img[y * w + x];
                match c {
                    0 => p.r = v,
                    1 => p.g = v,
                    _ => p.b = v,
                }
            }
        }
    }
    (w, h, img)
}

fn read_f32_planar(path: &str) -> (usize, usize, Vec<RGB<f32>>) {
    let d = std::fs::read(path).expect("read f32");
    let w = i32::from_le_bytes([d[0], d[1], d[2], d[3]]) as usize;
    let h = i32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize;
    let c = i32::from_le_bytes([d[8], d[9], d[10], d[11]]) as usize;
    assert_eq!(c, 3, "expected 3 planar channels");
    let n = w * h;
    let mut img = vec![RGB::new(0.0f32, 0.0, 0.0); n];
    let mut off = 12;
    for ch in 0..3 {
        for i in 0..n {
            let v = f32::from_le_bytes([d[off], d[off + 1], d[off + 2], d[off + 3]]);
            off += 4;
            let p = &mut img[i];
            match ch {
                0 => p.r = v,
                1 => p.g = v,
                _ => p.b = v,
            }
        }
    }
    (w, h, img)
}

fn srgb_gamma_to_linear(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 2,
        "usage: score_pfm <source.png> <label>=<file.pfm|.f32> [--srgb|--linear] [...]"
    );
    let src = image::open(&args[0]).expect("open source").to_rgb8();
    let (w, h) = (src.width() as usize, src.height() as usize);
    let orig_linear: Vec<RGB<f32>> = src
        .as_raw()
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    let orig_img = Img::new(orig_linear, w, h);
    let params = ButteraugliParams::default().with_intensity_target(80.0);

    for arg in &args[1..] {
        let (label, spec) = arg.split_once('=').expect("expected label=file");
        let (path, encoding) = match spec.split_once(':') {
            Some((p, enc)) => (p, enc),
            None => (spec, ""),
        };
        let (dw, dh, mut img) = if path.ends_with(".pfm") {
            read_pfm(path)
        } else {
            read_f32_planar(path)
        };
        assert!(dw == w && dh == h, "{label}: {dw}x{dh} != {w}x{h}");
        let is_linear = encoding == "linear" || (encoding.is_empty() && path.ends_with(".f32"));
        if !is_linear {
            for p in img.iter_mut() {
                p.r = srgb_gamma_to_linear(p.r);
                p.g = srgb_gamma_to_linear(p.g);
                p.b = srgb_gamma_to_linear(p.b);
            }
        }
        let img_ref = Img::new(img, w, h);
        let score = butteraugli_linear(orig_img.as_ref(), img_ref.as_ref(), &params)
            .map(|r| r.score)
            .unwrap_or(f64::NAN);
        println!("{label:<24} butteraugli={score:.6}");
    }
}
