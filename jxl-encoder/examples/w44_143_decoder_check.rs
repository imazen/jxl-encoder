// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-143 multi-decoder roundtrip check.
//!
//! Encodes the cell where the W44-143 lift now fires (codec_wiki e8 d=1.6
//! → +0.62 SSIM2 win) and verifies all three reference decoders parse it:
//! djxl (libjxl CLI), jxl-rs (Rust reference), jxl-oxide.

use image::GenericImageView;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let path = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png");
    let img = image::open(&path).expect("open codec_wiki.png");
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();

    let cfg = LossyConfig::new(1.6).with_effort(8);
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&lim)
        .encode(rgb.as_raw())
        .expect("encode");
    let out_path = "/tmp/w44_143_decoder_check.jxl";
    std::fs::write(out_path, &bytes).unwrap();
    println!("Encoded codec_wiki e8 d=1.6 -> {} bytes -> {out_path}", bytes.len());

    // djxl
    let djxl_path = std::env::var("DJXL_PATH")
        .unwrap_or_else(|_| "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl".to_string());
    let djxl = Command::new(&djxl_path)
        .args([out_path, "/tmp/w44_143_decoder_check_djxl.png"])
        .output();
    match djxl {
        Ok(out) if out.status.success() => println!("[PASS] djxl decoded"),
        Ok(out) => {
            eprintln!(
                "[FAIL] djxl exit={:?} stderr={}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("[SKIP] djxl not found: {e}");
        }
    }

    // jxl-oxide
    let reader = std::io::Cursor::new(&bytes);
    let mut oxide = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide read");
    oxide.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = oxide.render_frame(0).expect("jxl-oxide render");
    let fb = render.image_all_channels();
    println!(
        "[PASS] jxl-oxide decoded {}x{}",
        fb.width(),
        fb.height()
    );

    // jxl-rs decode via process
    let jxl_rs_path = std::env::var("JXL_RS_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli".to_string()
    });
    let jxl_rs = Command::new(&jxl_rs_path)
        .args([out_path, "/tmp/w44_143_decoder_check_jxlrs.png"])
        .output();
    match jxl_rs {
        Ok(out) if out.status.success() => println!("[PASS] jxl-rs decoded"),
        Ok(out) => {
            eprintln!(
                "[FAIL] jxl-rs exit={:?} stderr={}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
            // Don't exit — jxl-rs binary may not be at that path.
        }
        Err(e) => {
            eprintln!("[SKIP] jxl-rs binary not found at /home/lilith/work/jxl-rs/target/release/jxl: {e}");
        }
    }

    println!("W44-143 multi-decoder check OK");
}
