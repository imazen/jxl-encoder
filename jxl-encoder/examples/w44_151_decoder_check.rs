// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-151 decoder roundtrip — verify the W44-29 widening (Mechanism B
//! photo admission via mask_p25 >= 85) emits files that BOTH jxl-rs
//! (via jxl-oxide front-end) AND djxl can decode cleanly.
//!
//! Acceptance gate (h): 2 changed cells × 2 decoders → 4 checks PASS.
//! Coverage: 1418519 e8 d=5 + 1418519 e9 d=6 (the W44-147 audit cluster's
//! two highest-distance cells, where the W44-150 audit projected the
//! largest SSIM2 recovery via this Mechanism B path).

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;
use std::process::Command;

const CJXL_DECODER: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

fn check_one(image_name: &str, effort: u32, d: f32) -> bool {
    let corpus = std::env::var("CORPUS_ROOT")
        .unwrap_or_else(|_| format!("{}/work/codec-corpus", std::env::var("HOME").unwrap()));
    let path = Path::new(&corpus)
        .join("CID22/CID22-512/validation")
        .join(image_name);
    let img = image::open(&path).expect("open image");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.into_raw();
    let cfg = LossyConfig::new(d).with_effort(effort as u8).with_threads(8);
    let jxl = cfg
        .encode(&raw, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let stem = image_name.trim_end_matches(".png");
    let out = format!("/tmp/w44_151_{}_e{}_d{}.jxl", stem, effort, d);
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

    // jxl-oxide (jxl-rs front-end shared parser core)
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

    // djxl (libjxl reference decoder)
    let pfm = format!("/tmp/w44_151_{}_e{}_d{}.pfm", stem, effort, d);
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
    // Two changed cells × two decoders (jxl-oxide + djxl) — gate (h).
    for (image, effort, d) in [("1418519.png", 8, 5.0), ("1418519.png", 9, 6.0)] {
        if check_one(image, effort, d) {
            pass += 1;
        } else {
            fail += 1;
        }
    }
    println!(
        "\n=== W44-151 decoder check: {} pass, {} fail ===",
        pass, fail
    );
    std::process::exit(if fail > 0 { 1 } else { 0 });
}
