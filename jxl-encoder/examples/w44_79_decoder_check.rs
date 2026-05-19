//! W44-79 decoder roundtrip — verify hint=Some(true) produces a valid JXL
//! that both jxl-rs and djxl can decode.

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;
use std::process::Command;

fn main() {
    let path = Path::new("/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png");
    let img = image::open(path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.into_raw();
    let cfg = LossyConfig::new(4.0)
        .with_effort(7)
        .with_threads(1)
        .with_high_d_photo_hint(Some(true));
    let jxl = cfg.encode(&raw, w, h, PixelLayout::Rgb8).expect("encode");
    let out = "/tmp/w44_79_1189261_d4_hint.jxl";
    std::fs::write(out, &jxl).unwrap();
    println!("# Wrote {} bytes to {}", jxl.len(), out);

    // jxl-rs via jxl-oxide (already a workspace dep)
    let bytes = std::fs::read(out).unwrap();
    let reader = std::io::Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide read");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let _ = image.render_frame(0).expect("jxl-oxide render");
    println!("# jxl-oxide: OK");

    // djxl
    let pfm = "/tmp/w44_79_1189261_d4_hint.pfm";
    let status = Command::new("/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl")
        .arg(out)
        .arg(pfm)
        .status()
        .expect("djxl spawn");
    if status.success() {
        println!("# djxl: OK");
    } else {
        eprintln!("# djxl: FAILED");
        std::process::exit(1);
    }
}
