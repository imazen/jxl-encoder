//! W44-124 decoder roundtrip — verify the auto-fired W44-124 dispatch
//! produces valid JXL that both jxl-rs (via jxl-oxide) and djxl can
//! decode on codec_wiki at d=3 (the cell where the new auto-discriminator
//! fires).

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;
use std::process::Command;

const CJXL_DECODER: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

fn check_one(effort: u8, d: f32) {
    let path = Path::new("/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png");
    let img = image::open(path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.into_raw();
    // Default config (no hint) — the W44-124 auto-discriminator should fire
    // (codec_wiki m3=145.7, ed=0.0396 → both gates pass).
    let cfg = LossyConfig::new(d).with_effort(effort).with_threads(1);
    let jxl = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&raw)
        .expect("encode failed");
    let out = format!("/tmp/w44_124_codec_wiki_e{}_d{}_auto.jxl", effort, d);
    std::fs::write(&out, &jxl).unwrap();
    println!(
        "# e{} d={} → {} bytes written to {}",
        effort,
        d,
        jxl.len(),
        out
    );

    // jxl-oxide
    let reader = std::io::Cursor::new(&jxl);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide read");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let _ = image.render_frame(0).expect("jxl-oxide render");
    println!("# e{} d={} jxl-oxide: OK", effort, d);

    // djxl
    let pfm = format!("/tmp/w44_124_codec_wiki_e{}_d{}_auto.pfm", effort, d);
    let status = Command::new(CJXL_DECODER)
        .arg(&out)
        .arg(&pfm)
        .status()
        .expect("djxl spawn");
    if status.success() {
        println!("# e{} d={} djxl: OK", effort, d);
    } else {
        eprintln!("# e{} d={} djxl: FAILED", effort, d);
        std::process::exit(1);
    }
}

fn main() {
    // 3 cells where the W44-124 auto-discriminator fires (codec_wiki e5/e6/e7 d=3).
    for &(e, d) in &[(5u8, 3.0_f32), (6, 3.0), (7, 3.0)] {
        check_one(e, d);
    }
    println!("# W44-124 decoder roundtrip: ALL PASS");
}
