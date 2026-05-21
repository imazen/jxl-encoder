// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-120 decoder roundtrip — verify the distance-gated W44-117 EPF
//! sharpness seed (W44-120 production fix) emits files that decode
//! cleanly with both djxl + jxl-rs (via jxl-oxide).
//!
//! 3 cells:
//! - terminal e8 d=0.8 (the W44-119-surfaced regression cell, now
//!   restored to legacy uniform-4 behaviour by the W44-120 d<1.0
//!   gate).
//! - terminal e8 d=4 (W44-117 win cell, still benefits from the seed
//!   because d >= W44_120_EPF_SEED_MIN_DISTANCE).
//! - 1025469 e8 d=4 (photo, no W44-117/118 effect — W44-118
//!   is_screenshot gate keeps the seed off regardless of distance).
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop parallel' \
//!     --example w44_120_decoder_check \
//!     --manifest-path jxl-encoder/Cargo.toml

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;
use std::process::Command;

const CJXL_DECODER: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

fn check_one(corpus_sub: &str, image_name: &str, effort: u32, d: f32) -> bool {
    let corpus = std::env::var("CORPUS_ROOT")
        .unwrap_or_else(|_| format!("{}/work/codec-corpus", std::env::var("HOME").unwrap()));
    let path = if corpus_sub == "CID22" {
        Path::new(&corpus)
            .join("CID22/CID22-512/validation")
            .join(image_name)
    } else {
        Path::new(&corpus).join(corpus_sub).join(image_name)
    };
    let img = image::open(&path).expect("open image");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.into_raw();
    let cfg = LossyConfig::new(d)
        .with_effort(effort as u8)
        .with_threads(8);
    let jxl = cfg
        .encode(&raw, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let stem = image_name.trim_end_matches(".png");
    let out = format!("/tmp/w44_120_{}_e{}_d{}.jxl", stem, effort, d);
    std::fs::write(&out, &jxl).unwrap();
    println!(
        "# {} e{} d={} → {} bytes written to {}",
        image_name,
        effort,
        d,
        jxl.len(),
        out
    );

    let mut ok = true;

    // jxl-oxide (jxl-rs front-end)
    let reader = std::io::Cursor::new(&jxl);
    match jxl_oxide::JxlImage::builder().read(reader) {
        Ok(mut image) => {
            image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            match image.render_frame(0) {
                Ok(_) => println!("# {} e{} d={} jxl-oxide: OK", image_name, effort, d),
                Err(e) => {
                    println!(
                        "# {} e{} d={} jxl-oxide RENDER FAIL: {:?}",
                        image_name, effort, d, e
                    );
                    ok = false;
                }
            }
        }
        Err(e) => {
            println!(
                "# {} e{} d={} jxl-oxide READ FAIL: {:?}",
                image_name, effort, d, e
            );
            ok = false;
        }
    }

    // djxl
    let pfm = format!("/tmp/w44_120_{}_e{}_d{}.pfm", stem, effort, d);
    let status = Command::new(CJXL_DECODER)
        .arg(&out)
        .arg(&pfm)
        .status()
        .expect("djxl spawn");
    if status.success() {
        println!("# {} e{} d={} djxl: OK", image_name, effort, d);
    } else {
        println!(
            "# {} e{} d={} djxl FAIL: status={:?}",
            image_name, effort, d, status
        );
        ok = false;
    }
    ok
}

fn main() {
    let mut pass = 0;
    let mut fail = 0;
    for (corpus, image, effort, d) in [
        // W44-119-surfaced regression cell — W44-120 d<1.0 gate keeps
        // legacy uniform-4 seed (no W44-117 fire). Verify bitstream is
        // still valid.
        ("gb82-sc", "terminal.png", 8, 0.8),
        // W44-117 win cell — seed still fires (d >= 1.0 W44-120 gate).
        ("gb82-sc", "terminal.png", 8, 4.0),
        // Photo — W44-118 is_screenshot gate keeps the seed off
        // regardless of distance (mask < 95).
        ("CID22", "1025469.png", 8, 4.0),
    ] {
        if check_one(corpus, image, effort, d) {
            pass += 1;
        } else {
            fail += 1;
        }
    }
    println!(
        "\n=== W44-120 decoder check: {} pass, {} fail ===",
        pass, fail
    );
    std::process::exit(if fail > 0 { 1 } else { 0 });
}
