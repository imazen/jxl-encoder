// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-102 decoder roundtrip check (bitstream-parity sanity).
//!
//! Encodes 5 W44-102 cells with `__expert` override
//! `cfl_two_pass = Some(true)` at e5/e6 (where the production default
//! is `cfl_two_pass: effort >= 7`; W44-102 RULED OUT widening to
//! e5+). Roundtrips through djxl + jxl-rs + jxl-oxide to confirm the
//! widened-gate code path produces a decoder-compatible bitstream.
//! Bytes/quality NOT measured — see `w44_102_cfl_two_pass_ab` for the
//! ruling-out bench.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_102_decoder_check

use image::GenericImageView;
use jxl_encoder::effort::LossyInternalParams;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::PathBuf;
use std::process::Command;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82: &str = "/home/lilith/work/codec-corpus/gb82-sc";
const DJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXL_CLI: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

fn main() {
    let cells: &[(&str, &str, u8, f32, &str)] = &[
        (CID22, "1420710.png", 6, 5.0, "W44-101 wedge (priority)"),
        (CID22, "1025469.png", 6, 4.0, "W44-101 wedge"),
        (GB82, "codec_wiki.png", 6, 0.2, "W44-101 wedge"),
        (CID22, "1418519.png", 6, 6.0, "W44-101 wedge"),
        (CID22, "1044329.png", 5, 0.5, "Best bfly improvement cell"),
    ];

    let mut total_pass = 0;
    let mut total_fail = 0;

    for &(dir, image, effort, dist, label) in cells {
        let path = PathBuf::from(dir).join(image);
        let img = image::open(&path).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();

        // Use __expert override `cfl_two_pass = Some(true)` so the
        // bitstream IS exercising the widened-gate code path even
        // though the production default at e5/e6 stays at
        // `cfl_two_pass: effort >= 7` (W44-102 RULED OUT).
        let mut params = LossyInternalParams::default();
        params.cfl_two_pass = Some(true);
        let bytes = LossyConfig::new(dist)
            .with_effort(effort)
            .with_threads(8)
            .with_internal_params(params)
            .encode(&rgb, w, h, PixelLayout::Rgb8)
            .expect("encode failed");

        let tmp_jxl = format!(
            "/tmp/w44_102_dec_{}_{}_{}.jxl",
            image.replace('.', "_"),
            effort,
            (dist * 10.0) as u32
        );
        std::fs::write(&tmp_jxl, &bytes).expect("write jxl");

        // djxl
        let tmp_djxl_png = format!("{}.djxl.png", tmp_jxl);
        let djxl = Command::new(DJXL)
            .args([&tmp_jxl, &tmp_djxl_png])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        // jxl-rs (jxl_cli)
        let tmp_jxlrs_png = format!("{}.jxlrs.png", tmp_jxl);
        let jxlrs = Command::new(JXL_CLI)
            .args([&tmp_jxl, &tmp_jxlrs_png])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        // jxl-oxide (in-process)
        let oxide = std::io::Cursor::new(bytes.clone());
        let oxide_ok = match jxl_oxide::JxlImage::builder().read(oxide) {
            Ok(img) => img.render_frame(0).is_ok(),
            Err(_) => false,
        };

        let mark = |b: bool| if b { "OK" } else { "FAIL" };
        println!(
            "{} e{} d{}: djxl={} jxl-rs={} jxl-oxide={} (bytes={}) [{}]",
            image,
            effort,
            dist,
            mark(djxl),
            mark(jxlrs),
            mark(oxide_ok),
            bytes.len(),
            label
        );

        for b in [djxl, jxlrs, oxide_ok] {
            if b {
                total_pass += 1;
            } else {
                total_fail += 1;
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&tmp_jxl);
        let _ = std::fs::remove_file(&tmp_djxl_png);
        let _ = std::fs::remove_file(&tmp_jxlrs_png);
    }

    println!(
        "\nW44-102 decoder roundtrip: {}/{} pass",
        total_pass,
        total_pass + total_fail
    );
    if total_fail > 0 {
        std::process::exit(1);
    }
}
