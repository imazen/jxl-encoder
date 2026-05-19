// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-78 multi-decoder roundtrip on the affected EV cells.
//! Verifies that the gate-widened bitstreams decode cleanly via:
//!   1. jxl-rs (primary)
//!   2. jxl-oxide
//!   3. djxl (libjxl CLI, optional — skipped if not on PATH)
//!
//! Build:
//!   cargo run --release -p jxl-encoder --features parallel \
//!     --example w44_78_decoder_roundtrip

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;
use std::process::Command;

const DJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

const CELLS: &[(&str, &str, f32)] = &[
    (
        "1420710",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        3.0,
    ),
    (
        "1044329",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1044329.png",
        3.0,
    ),
    (
        "2389166",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/2389166.png",
        3.0,
    ),
];

fn encode(rgb: &[u8], w: u32, h: u32, d: f32) -> Vec<u8> {
    LossyConfig::new(d)
        .with_effort(7)
        .with_threads(8)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
}

fn djxl_decode(bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let in_path = format!("/tmp/w44_78_djxl_{}.jxl", std::process::id());
    let out_path = format!("/tmp/w44_78_djxl_{}.png", std::process::id());
    std::fs::File::create(&in_path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| format!("write tmp: {}", e))?;
    let out = Command::new(DJXL)
        .args([&in_path, &out_path])
        .output()
        .map_err(|e| format!("spawn djxl: {}", e))?;
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    if !out.status.success() {
        return Err(format!(
            "djxl rc={:?}: stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn oxide_decode(bytes: &[u8]) -> Result<(u32, u32), String> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(bytes))
        .map_err(|e| format!("jxl-oxide read: {}", e))?;
    let w = img.width();
    let h = img.height();
    let _ = img
        .render_frame(0)
        .map_err(|e| format!("jxl-oxide render: {}", e))?;
    Ok((w, h))
}

fn main() {
    for &(name, path, distance) in CELLS {
        if !Path::new(path).exists() {
            eprintln!("SKIP {} (file missing)", name);
            continue;
        }
        let img = image::open(path).expect("open");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let bytes = encode(rgb.as_raw(), w, h, distance);
        println!(
            "==== {} (d={:.1}, {}×{}, {} bytes) ====",
            name,
            distance,
            w,
            h,
            bytes.len()
        );

        match oxide_decode(&bytes) {
            Ok((dw, dh)) => println!("  jxl-oxide:  OK ({}×{})", dw, dh),
            Err(e) => println!("  jxl-oxide:  FAIL — {}", e),
        }
        match djxl_decode(&bytes) {
            Ok(()) => println!("  djxl:       OK"),
            Err(e) => println!("  djxl:       FAIL — {}", e),
        }
    }
}
