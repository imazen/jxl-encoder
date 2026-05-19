//! W44-91 decoder roundtrip — verify the auto-fired W44-91 lift produces
//! valid JXL that both jxl-rs (via jxl-oxide) and djxl can decode on
//! 1189261 at d=3, 4, 5 (the three cells where the new dispatch fires).

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;
use std::process::Command;

const CJXL_DECODER: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

fn check_one(d: f32) {
    let path = Path::new("/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png");
    let img = image::open(path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.into_raw();
    // Default config (no hint) — the W44-91 auto gate should fire.
    let cfg = LossyConfig::new(d).with_effort(7).with_threads(1);
    let jxl = cfg
        .encode(&raw, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let out = format!("/tmp/w44_91_1189261_d{}_auto.jxl", d);
    std::fs::write(&out, &jxl).unwrap();
    println!("# d={} → {} bytes written to {}", d, jxl.len(), out);

    // jxl-oxide
    let reader = std::io::Cursor::new(&jxl);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide read");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let _ = image.render_frame(0).expect("jxl-oxide render");
    println!("# d={} jxl-oxide: OK", d);

    // djxl
    let pfm = format!("/tmp/w44_91_1189261_d{}_auto.pfm", d);
    let status = Command::new(CJXL_DECODER)
        .arg(&out)
        .arg(&pfm)
        .status()
        .expect("djxl spawn");
    if status.success() {
        println!("# d={} djxl: OK", d);
    } else {
        eprintln!("# d={} djxl: FAILED", d);
        std::process::exit(1);
    }
}

fn main() {
    // All 3 distances where the W44-91 auto-fire is active.
    for &d in &[3.0_f32, 4.0, 5.0] {
        check_one(d);
    }
    println!("# W44-91 decoder roundtrip: ALL PASS");
}
