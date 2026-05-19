//! W44-35 multi-decoder roundtrip check.
//!
//! Encodes 1418519.png at the 5 ledger cells with the default API
//! (W44-35 dispatch active) and decodes each through jxl-oxide to
//! confirm the bitstream is spec-valid. djxl / jxl-rs check via a
//! separate `just rd-regression` invocation.
//!
//! Build:
//!   cargo build -p jxl-encoder --release \
//!       --features 'parallel' \
//!       --example w44_35_decoder_check
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!       --features 'parallel' \
//!       --example w44_35_decoder_check

use std::io::Cursor;
use std::path::PathBuf;

use jxl_encoder::api::{LossyConfig, PixelLayout};

const CELLS: &[(u8, f32, &str)] = &[
    (6, 1.0, "e6_d1.0"),
    (6, 1.2, "e6_d1.2"),
    (6, 1.6, "e6_d1.6"),
    (7, 1.2, "e7_d1.2"),
    (7, 1.6, "e7_d1.6"),
];

fn main() {
    let path =
        PathBuf::from("/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png");
    let img = image::open(&path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let rgb_bytes = rgb.as_raw().clone();

    let mut all_ok = true;
    println!(
        "{:<10} {:>8} {:>8} {:>10}",
        "cell", "bytes", "dec_w", "dec_h"
    );
    for &(effort, distance, label) in CELLS {
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(1);
        let bytes = cfg
            .encode(&rgb_bytes, w, h, PixelLayout::Rgb8)
            .expect("encode");
        // Decode via jxl-oxide
        let reader = Cursor::new(&bytes);
        let mut img = jxl_oxide::JxlImage::builder()
            .read(reader)
            .unwrap_or_else(|e| panic!("oxide read {label}: {e}"));
        img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = img
            .render_frame(0)
            .unwrap_or_else(|e| panic!("oxide render {label}: {e}"));
        let fb = render.image_all_channels();
        let dec_w = fb.width() as u32;
        let dec_h = fb.height() as u32;
        let ok = dec_w == w && dec_h == h;
        if !ok {
            all_ok = false;
        }
        println!(
            "{:<10} {:>8} {:>8} {:>10} {}",
            label,
            bytes.len(),
            dec_w,
            dec_h,
            if ok { "OK" } else { "FAIL" }
        );
    }
    if all_ok {
        println!("\nAll 5 cells decode-OK via jxl-oxide");
    } else {
        eprintln!("\nDECODE FAILURE detected");
        std::process::exit(1);
    }
}
