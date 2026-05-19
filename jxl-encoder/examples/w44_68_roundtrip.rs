// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-68 codec_wiki d=4 roundtrip: encode + decode with jxl-rs (primary) and
//! verify djxl can also decode the same bytes externally.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use jxl_encoder::api::{LossyConfig, PixelLayout};

fn corpus(name: &str) -> PathBuf {
    PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc").join(name)
}

fn encode_and_save(name: &str, d: f32, out: &PathBuf) -> usize {
    let path = corpus(name);
    let img = image::open(&path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.as_raw().clone();
    let cfg = LossyConfig::new(d).with_effort(7).with_threads(1);
    let bytes = cfg.encode(&raw, w, h, PixelLayout::Rgb8).expect("encode");
    let mut f = std::fs::File::create(out).unwrap();
    f.write_all(&bytes).unwrap();
    bytes.len()
}

fn decode_jxlrs(jxl_path: &PathBuf) -> Result<(), String> {
    let out_png = PathBuf::from("/tmp/w44-68-jxlrs.png");
    let r = Command::new("/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli")
        .args([jxl_path.to_str().unwrap(), out_png.to_str().unwrap()])
        .output()
        .map_err(|e| format!("spawn jxl_cli: {}", e))?;
    if !r.status.success() {
        return Err(format!(
            "jxl_cli failed: {}",
            String::from_utf8_lossy(&r.stderr)
        ));
    }
    Ok(())
}

fn decode_djxl(jxl_path: &PathBuf) -> Result<(), String> {
    let out_png = PathBuf::from("/tmp/w44-68-djxl.png");
    let r = Command::new("/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl")
        .args([jxl_path.to_str().unwrap(), out_png.to_str().unwrap()])
        .output()
        .map_err(|e| format!("spawn djxl: {}", e))?;
    if !r.status.success() {
        return Err(format!(
            "djxl failed: {}",
            String::from_utf8_lossy(&r.stderr)
        ));
    }
    Ok(())
}

fn main() {
    for (name, d) in &[
        ("codec_wiki.png", 4.0f32),
        ("codec_wiki.png", 3.0),
        ("terminal.png", 4.0),
        ("imac_g3.png", 4.0),
    ] {
        let out = PathBuf::from(format!(
            "/tmp/w44-68-{}-d{}.jxl",
            name.trim_end_matches(".png"),
            d
        ));
        let bytes = encode_and_save(name, *d, &out);
        eprint!("{} d={}: bytes={} ", name, d, bytes);
        match decode_jxlrs(&out) {
            Ok(()) => eprint!("jxl-rs=OK "),
            Err(e) => eprint!("jxl-rs=FAIL[{}] ", e),
        }
        match decode_djxl(&out) {
            Ok(()) => eprintln!("djxl=OK"),
            Err(e) => eprintln!("djxl=FAIL[{}]", e),
        }
    }
}
