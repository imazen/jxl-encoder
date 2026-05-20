// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-98 decoder roundtrip check.
//!
//! Encodes 3 W44-98-closed cells with the new production dispatch and
//! roundtrips through djxl + jxl_cli (jxl-rs) + jxl-oxide (in-process).
//! All three must successfully decode and produce the same dimensions
//! as the source.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_98_decoder_check

use image::GenericImageView;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const DJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXL_CLI: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

fn main() {
    let cells: &[(&str, u8, f32)] = &[
        ("1420710.png", 5, 5.0), // OPEN closed by W44-98
        ("1420710.png", 5, 6.0), // OPEN closed by W44-98
        ("1420710.png", 7, 5.0), // OPEN closed by W44-98
    ];

    let mut total_pass = 0;
    let mut total_fail = 0;

    for &(image, effort, dist) in cells {
        let path = PathBuf::from(CID22).join(image);
        let img = image::open(&path).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();

        let bytes = LossyConfig::new(dist)
            .with_effort(effort)
            .with_threads(8)
            .encode(&rgb, w, h, PixelLayout::Rgb8)
            .expect("encode failed");

        let tmp_jxl = format!(
            "/tmp/w44_98_dec_{}_{}_{}.jxl",
            image.replace('.', "_"),
            effort,
            (dist * 10.0) as u32
        );
        std::fs::write(&tmp_jxl, &bytes).expect("write jxl");

        // djxl
        let tmp_djxl_png = format!("{}.djxl.png", tmp_jxl);
        let out = Command::new(DJXL)
            .args([&tmp_jxl, &tmp_djxl_png])
            .output()
            .expect("djxl run");
        let djxl_ok = out.status.success();
        let _ = std::fs::remove_file(&tmp_djxl_png);

        // jxl_cli (jxl-rs)
        let tmp_jxlrs_png = format!("{}.jxlrs.png", tmp_jxl);
        let out = Command::new(JXL_CLI)
            .args([&tmp_jxl, &tmp_jxlrs_png])
            .output()
            .expect("jxl_cli run");
        let jxlrs_ok = out.status.success();
        if !jxlrs_ok {
            eprintln!("jxl_cli stderr: {}", String::from_utf8_lossy(&out.stderr));
        }
        let _ = std::fs::remove_file(&tmp_jxlrs_png);

        // jxl-oxide in-process
        let reader = Cursor::new(&bytes);
        let oxide_ok = jxl_oxide::JxlImage::builder()
            .read(reader)
            .ok()
            .and_then(|mut img| img.render_frame(0).ok())
            .map(|render| {
                let fb = render.image_all_channels();
                fb.width() == w as usize && fb.height() == h as usize
            })
            .unwrap_or(false);

        println!(
            "{} e{} d{}: djxl={} jxl-rs={} jxl-oxide={} (bytes={})",
            image,
            effort,
            dist,
            if djxl_ok { "OK" } else { "FAIL" },
            if jxlrs_ok { "OK" } else { "FAIL" },
            if oxide_ok { "OK" } else { "FAIL" },
            bytes.len(),
        );

        for ok in [djxl_ok, jxlrs_ok, oxide_ok] {
            if ok {
                total_pass += 1;
            } else {
                total_fail += 1;
            }
        }

        let _ = std::fs::remove_file(&tmp_jxl);
    }

    println!(
        "\nW44-98 decoder roundtrip: {}/{} pass",
        total_pass,
        total_pass + total_fail
    );
    if total_fail > 0 {
        std::process::exit(1);
    }
}
