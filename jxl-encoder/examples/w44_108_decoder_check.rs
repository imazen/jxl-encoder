//! W44-108 decoder roundtrip — verify the recovered W44-105 wins
//! (terminal e8 d=2..3, imac_g3 e8 d=3, terminal e9 d=2.5) decode
//! cleanly via both jxl-oxide (jxl-rs) and djxl.
//!
//! Acceptance gate (e): multi-decoder roundtrip via djxl + jxl-rs on
//! 3 recovered cells. We test 4 (3 + 1 spare) to be safe.

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;
use std::process::Command;

const CJXL_DECODER: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

fn check_one(image_name: &str, effort: u32, d: f32) {
    let corpus = std::env::var("CORPUS_ROOT")
        .unwrap_or_else(|_| format!("{}/work/codec-corpus", std::env::var("HOME").unwrap()));
    let path = Path::new(&corpus).join("gb82-sc").join(image_name);
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
    let out = format!("/tmp/w44_108_{}_e{}_d{}.jxl", stem, effort, d);
    std::fs::write(&out, &jxl).unwrap();
    println!(
        "# {} e{} d={} → {} bytes written to {}",
        image_name,
        effort,
        d,
        jxl.len(),
        out
    );

    // jxl-oxide (jxl-rs front-end)
    let reader = std::io::Cursor::new(&jxl);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide read");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let _ = image.render_frame(0).expect("jxl-oxide render");
    println!("# {} e{} d={} jxl-oxide: OK", image_name, effort, d);

    // djxl
    let pfm = format!("/tmp/w44_108_{}_e{}_d{}.pfm", stem, effort, d);
    let status = Command::new(CJXL_DECODER)
        .arg(&out)
        .arg(&pfm)
        .status()
        .expect("djxl spawn");
    if status.success() {
        println!("# {} e{} d={} djxl: OK", image_name, effort, d);
    } else {
        eprintln!("# {} e{} d={} djxl: FAILED", image_name, effort, d);
        std::process::exit(1);
    }
}

fn main() {
    // 4 recovered cells (3 required by acceptance gate + 1 spare to
    // cover both effort variants and codec_wiki rejection-still-decodes).
    check_one("terminal.png", 8, 3.0);
    check_one("imac_g3.png", 8, 3.0);
    check_one("terminal.png", 9, 2.5);
    // codec_wiki at d=3 — gate DOES NOT fire here; verifies the
    // suppression path still produces a decodable file.
    check_one("codec_wiki.png", 8, 3.0);
    println!("# W44-108 decoder roundtrip: ALL PASS");
}
