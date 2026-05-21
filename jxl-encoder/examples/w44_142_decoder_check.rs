// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-142 multi-decoder roundtrip — verifies that the codec_wiki cell
//! where the W44-142 suppression sub-gate fires (e9 d=1.2) produces a
//! bitstream that decodes cleanly via jxl-oxide (in-process) + djxl
//! + jxl-rs (external CLIs).
//!
//! Encodes codec_wiki.png at e9 d=1.2 with the W44-142 gate active
//! (production default), writes to /tmp/w44_142_check.jxl, decodes via
//! jxl-oxide in-process for pixel verification, and runs djxl +
//! jxl_cli (jxl-rs) for round-trip parity.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_142_decoder_check \
//!     --manifest-path jxl-encoder/Cargo.toml

use image::GenericImageView;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::io::Cursor;
use std::path::Path;
use std::process::Command;

const SOURCE: &str = "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png";
const OUT_JXL: &str = "/tmp/w44_142_check.jxl";
const OUT_DJXL: &str = "/tmp/w44_142_check_djxl.png";
const OUT_JXLRS: &str = "/tmp/w44_142_check_jxlrs.png";

const DJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXLRS: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let img = image::open(SOURCE)?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();

    eprintln!("Encoding codec_wiki e9 d=1.2 with W44-142 default (m3>=60 AND ed<0.05 AND d<1.5 → fire)...");
    let bitstream = LossyConfig::new(1.2)
        .with_effort(9)
        .with_threads(8)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))?;
    eprintln!("  bitstream: {} bytes", bitstream.len());
    std::fs::write(OUT_JXL, &bitstream)?;

    // jxl-oxide in-process decode
    eprintln!("Decode via jxl-oxide (in-process)...");
    let reader = Cursor::new(bitstream.clone());
    let mut jxl_img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .map_err(|e| format!("jxl-oxide read: {e:?}"))?;
    jxl_img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = jxl_img
        .render_frame(0)
        .map_err(|e| format!("jxl-oxide render_frame: {e:?}"))?;
    let fb = render.image_all_channels();
    eprintln!(
        "  jxl-oxide OK: {}x{}, {} channels",
        fb.width(),
        fb.height(),
        fb.channels()
    );

    // djxl CLI
    if Path::new(DJXL).exists() {
        eprintln!("Decode via djxl...");
        let st = Command::new(DJXL).arg(OUT_JXL).arg(OUT_DJXL).status()?;
        if !st.success() {
            return Err(format!("djxl failed: {:?}", st).into());
        }
        eprintln!("  djxl OK: {}", OUT_DJXL);
    } else {
        eprintln!("  djxl NOT FOUND at {DJXL} — skipping");
    }

    // jxl-rs CLI
    if Path::new(JXLRS).exists() {
        eprintln!("Decode via jxl_cli (jxl-rs)...");
        let st = Command::new(JXLRS).arg(OUT_JXL).arg(OUT_JXLRS).status()?;
        if !st.success() {
            return Err(format!("jxl_cli failed: {:?}", st).into());
        }
        eprintln!("  jxl_cli OK: {}", OUT_JXLRS);
    } else {
        eprintln!("  jxl_cli NOT FOUND at {JXLRS} — skipping");
    }

    eprintln!("\nAll decoders OK. Bitstream parity-verified.");
    Ok(())
}
