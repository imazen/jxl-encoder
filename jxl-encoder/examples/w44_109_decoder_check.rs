//! W44-109 decoder roundtrip — verify the new low-effort
//! screenshot-class qf-pre-scale path (e5/e6/e7 d>=3.5 OR W44-108
//! sub-gate at d∈[2.0, 3.5) with low m3) emits files that both
//! jxl-oxide (jxl-rs front-end) and djxl can decode cleanly.
//!
//! Acceptance gate (d): multi-decoder roundtrip via djxl + jxl-rs on
//! 3 terminal cells. We test 3 cells covering the gate's effort sweep
//! (e5/e6/e7) at d=4.0.

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
    let out = format!("/tmp/w44_109_{}_e{}_d{}.jxl", stem, effort, d);
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
    let pfm = format!("/tmp/w44_109_{}_e{}_d{}.pfm", stem, effort, d);
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
    // 3 low-effort terminal cells (the primary W44-109 target). The
    // gate fires at all three (terminal mask1x1 median ≫ 95, d=4 ≥ 3.5).
    check_one("terminal.png", 5, 4.0);
    check_one("terminal.png", 6, 4.0);
    check_one("terminal.png", 7, 4.0);
    println!("# W44-109 decoder roundtrip: ALL PASS");
}
