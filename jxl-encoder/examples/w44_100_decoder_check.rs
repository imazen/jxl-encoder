// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-100 decoder roundtrip check.
//!
//! Encodes 1531677 e5 d=5 (the cell closed by W44-100) with the new
//! production dispatch (dct16x32 = 1.23 on LC) and roundtrips through
//! djxl + jxl_cli (jxl-rs) + jxl-oxide. Also checks that 1531677 e6/e8/e9
//! d=5 still roundtrip cleanly with the bumped value, plus 1420710 e5 d=5
//! (HC dispatch — must stay byte-identical to W44-98).
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_100_decoder_check

use image::GenericImageView;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::PathBuf;
use std::process::Command;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const DJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXL_CLI: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

fn main() {
    let cells: &[(&str, u8, f32, &str)] = &[
        ("1531677.png", 5, 5.0, "OPEN closed by W44-100"),
        ("1531677.png", 6, 5.0, "still LC, bumped to 1.23"),
        ("1531677.png", 8, 5.0, "still LC, bumped to 1.23"),
        ("1531677.png", 9, 5.0, "still LC, bumped to 1.23"),
        ("1420710.png", 5, 5.0, "HC unchanged from W44-98"),
    ];

    let mut total_pass = 0;
    let mut total_fail = 0;

    for &(image, effort, dist, label) in cells {
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
            "/tmp/w44_100_dec_{}_{}_{}.jxl",
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
        "\nW44-100 decoder roundtrip: {}/{} pass",
        total_pass,
        total_pass + total_fail
    );
    if total_fail > 0 {
        std::process::exit(1);
    }
}
