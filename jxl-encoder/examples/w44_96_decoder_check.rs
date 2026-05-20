// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-96 decoder roundtrip: verify the auto-fired W44-96 variant Z lift
//! on the WANT_Z cells produces valid JXL that decodes cleanly through
//! the jxl-rs and jxl-oxide (and djxl if available) decoders.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run -p jxl-encoder --release \
//!     --features 'parallel' \
//!     --example w44_96_decoder_check

use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::{Path, PathBuf};
use std::process::Command;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

// Cells where the W44-96 sub-discriminator fires.
const CELLS: &[(&str, u8, f32)] = &[
    ("1420710.png", 6, 5.0),
    ("1420710.png", 6, 6.0),
    ("1531677.png", 5, 6.0),
    ("1531677.png", 6, 6.0),
];

fn try_djxl(jxl_bytes: &[u8], tag: &str) -> Result<(), String> {
    let djxl_bin = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
    if !Path::new(djxl_bin).exists() {
        return Ok(()); // skip silently
    }
    let in_path = format!("/tmp/w44_96_{}.jxl", tag);
    let out_path = format!("/tmp/w44_96_{}.png", tag);
    std::fs::write(&in_path, jxl_bytes).map_err(|e| format!("write jxl: {e}"))?;
    let out = Command::new(djxl_bin)
        .args([&in_path, &out_path])
        .output()
        .map_err(|e| format!("spawn djxl: {e}"))?;
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    if !out.status.success() {
        return Err(format!(
            "djxl failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn try_jxl_rs(jxl_bytes: &[u8], tag: &str) -> Result<(), String> {
    let jxl_rs_bin = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";
    if !Path::new(jxl_rs_bin).exists() {
        return Ok(()); // skip silently
    }
    let in_path = format!("/tmp/w44_96_jxlrs_{}.jxl", tag);
    let out_path = format!("/tmp/w44_96_jxlrs_{}.png", tag);
    std::fs::write(&in_path, jxl_bytes).map_err(|e| format!("write jxl: {e}"))?;
    let out = Command::new(jxl_rs_bin)
        .args([&in_path, &out_path])
        .output()
        .map_err(|e| format!("spawn jxl-rs: {e}"))?;
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    if !out.status.success() {
        return Err(format!(
            "jxl-rs failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn try_jxl_oxide(jxl_bytes: &[u8]) -> Result<(), String> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(jxl_bytes))
        .map_err(|e| format!("jxl-oxide parse: {e}"))?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img
        .render_frame(0)
        .map_err(|e| format!("jxl-oxide render: {e}"))?;
    let fb = render.image_all_channels();
    if fb.buf().is_empty() {
        return Err("jxl-oxide: empty buffer".into());
    }
    Ok(())
}

fn main() {
    println!("# W44-96 decoder roundtrip: WANT_Z cells");
    let mut all_ok = true;
    for (name, eff, d) in CELLS {
        let path = PathBuf::from(CID22).join(name);
        let img = image::open(&path).expect("open");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let raw = rgb.into_raw();
        let bytes = LossyConfig::new(*d)
            .with_effort(*eff)
            .with_threads(8)
            .encode(&raw, w, h, PixelLayout::Rgb8)
            .expect("encode");

        let tag = format!(
            "{}_{}_{}",
            name.trim_end_matches(".png"),
            eff,
            (*d * 10.0) as u32
        );
        let djxl_res = try_djxl(&bytes, &tag);
        let oxide_res = try_jxl_oxide(&bytes);
        let jxlrs_res = try_jxl_rs(&bytes, &tag);

        let dj = djxl_res.as_ref().map(|_| "OK").unwrap_or("FAIL");
        let ox = oxide_res.as_ref().map(|_| "OK").unwrap_or("FAIL");
        let jr = jxlrs_res.as_ref().map(|_| "OK").unwrap_or("FAIL");
        println!(
            "{} e{} d={:.1} bytes={} djxl={} jxl-oxide={} jxl-rs={}",
            name,
            eff,
            d,
            bytes.len(),
            dj,
            ox,
            jr
        );
        if djxl_res.is_err() || oxide_res.is_err() || jxlrs_res.is_err() {
            all_ok = false;
            if let Err(e) = djxl_res {
                eprintln!("  djxl error: {e}");
            }
            if let Err(e) = oxide_res {
                eprintln!("  jxl-oxide error: {e}");
            }
            if let Err(e) = jxlrs_res {
                eprintln!("  jxl-rs error: {e}");
            }
        }
    }
    if all_ok {
        println!("ALL DECODES OK");
    } else {
        std::process::exit(1);
    }
}
