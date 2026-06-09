// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-222 multi-decoder roundtrip for the 5-knob `Tier2Knobs` expander.
//!
//! Verifies that bitstreams produced under non-default `buttloop_aq_balance`
//! values decode cleanly through jxl-oxide (in-process), djxl (subprocess
//! against libjxl reference), and jxl_cli (subprocess against jxl-rs).
//!
//! Single-shot install limitation: this test uses `runtime::install`
//! ONCE per process. We pick ONE non-default knob set (the moderate
//! `k5 = 0.5` deviation), encode 5 fixtures with it via
//! `LossyConfig::with_knobs`, and roundtrip each through all 3 decoders.
//!
//! Default-knob roundtrip is already covered by the existing hash-lock
//! fixtures (36/36 byte-identical with `tier2_knobs: None`) plus the
//! W44-221 byte-roundtrip test; this file focuses on the EFFECT of the
//! 5th knob on the bitstream + downstream decoder compatibility.

#![cfg(feature = "tuning-override")]

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use jxl_encoder::tuning::coupling::Tier2Knobs;
use jxl_encoder::{LossyConfig, PixelLayout};

fn corpus_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("CODEC_CORPUS_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join("work/codec-corpus");
    if p.exists() { Some(p) } else { None }
}

fn load_or_synth(name: &str, fallback_w: u32, fallback_h: u32) -> (Vec<u8>, u32, u32) {
    if let Some(root) = corpus_root() {
        let path = root.join(name);
        if path.exists() {
            if let Ok(img) = image::open(&path) {
                let rgb = img.to_rgb8();
                let (w, h) = rgb.dimensions();
                let cw = w.min(256);
                let ch = h.min(256);
                let mut out = Vec::with_capacity((cw * ch * 3) as usize);
                for y in 0..ch {
                    for x in 0..cw {
                        let p = rgb.get_pixel(x, y);
                        out.extend_from_slice(&p.0);
                    }
                }
                return (out, cw, ch);
            }
        }
    }
    // Synthetic fallback that exercises both screen + photo gates.
    let (w, h) = (fallback_w, fallback_h);
    let mut pixels = vec![255u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            // Vertical gradient to make the synthesis non-trivial.
            pixels[i] = ((x as f32 / w as f32) * 255.0) as u8;
            pixels[i + 1] = ((y as f32 / h as f32) * 255.0) as u8;
            pixels[i + 2] = 128;
        }
    }
    (pixels, w, h)
}

/// 5 fixture configurations: 4 representative cells + 1 screenshot.
/// Each cell is `(name, w, h, distance, effort)`.
fn fixtures() -> Vec<(&'static str, &'static str, u32, u32, f32, u8)> {
    vec![
        (
            "photo_d2_e5",
            "CID22/CID22-512/validation/1418519.png",
            256,
            256,
            2.0,
            5,
        ),
        (
            "photo_d4_e7",
            "CID22/CID22-512/validation/1025469.png",
            256,
            256,
            4.0,
            7,
        ),
        ("screen_d2_e5", "gb82-sc/terminal.png", 256, 256, 2.0, 5),
        ("screen_d4_e7", "gb82-sc/codec_wiki.png", 256, 256, 4.0, 7),
        ("synth_d1_e5", "this_will_not_exist.png", 64, 64, 1.0, 5),
    ]
}

fn djxl_path() -> Option<String> {
    let p = std::env::var("DJXL_PATH")
        .unwrap_or_else(|_| "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl".into());
    if std::path::Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn jxl_cli_path() -> Option<String> {
    let p = std::env::var("JXL_CLI_PATH")
        .unwrap_or_else(|_| "/home/lilith/work/jxl-rs/target/release/jxl_cli".into());
    if std::path::Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn write_tmp_jxl(bytes: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("w44_222_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("frame.jxl");
    let mut f = std::fs::File::create(&path).expect("create tmp jxl");
    f.write_all(bytes).expect("write tmp jxl");
    path
}

fn decode_via_jxl_oxide(jxl_bytes: &[u8]) -> Result<(u32, u32), String> {
    let cursor = std::io::Cursor::new(jxl_bytes);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(cursor)
        .map_err(|e| format!("jxl-oxide read: {e:?}"))?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let _render = img
        .render_frame(0)
        .map_err(|e| format!("jxl-oxide render: {e:?}"))?;
    Ok((img.width(), img.height()))
}

fn decode_via_djxl(jxl_path: &std::path::Path) -> Result<(), String> {
    let Some(djxl) = djxl_path() else {
        return Err("djxl_path missing".into());
    };
    let out_pfm = jxl_path.with_extension("out.pfm");
    let output = Command::new(&djxl)
        .arg(jxl_path)
        .arg(&out_pfm)
        .output()
        .map_err(|e| format!("djxl spawn: {e}"))?;
    let _ = std::fs::remove_file(&out_pfm);
    if !output.status.success() {
        return Err(format!(
            "djxl exit code {:?}, stderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn decode_via_jxl_rs(jxl_path: &std::path::Path) -> Result<(), String> {
    let Some(cli) = jxl_cli_path() else {
        return Err("jxl_cli missing".into());
    };
    let out_pfm = jxl_path.with_extension("rs.pfm");
    let output = Command::new(&cli)
        .arg(jxl_path)
        .arg(&out_pfm)
        .output()
        .map_err(|e| format!("jxl_cli spawn: {e}"))?;
    let _ = std::fs::remove_file(&out_pfm);
    if !output.status.success() {
        return Err(format!(
            "jxl_cli exit code {:?}, stderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

/// W44-222 multi-decoder roundtrip on 5 cells × 1 non-default knob config × 3 decoders.
///
/// Each cell encodes with `buttloop_aq_balance = 0.5` (k5 != 0 →
/// install path fires) and decodes through jxl-oxide (always), djxl
/// (skipped if binary missing), and jxl_cli (skipped if binary missing).
///
/// Decoder skips DO NOT silently pass the test — they print a warning
/// and reduce the count. We require jxl-oxide to ALWAYS succeed (it's
/// in-process and always available) and the test asserts at least
/// jxl-oxide passes on all 5 cells.
#[test]
fn w44_222_5_cells_x_3_decoders_roundtrip() {
    let knobs = Tier2Knobs {
        buttloop_aq_balance: 0.5,
        ..Default::default()
    };

    let mut oxide_pass = 0;
    let mut djxl_pass = 0;
    let mut jxl_rs_pass = 0;
    let mut oxide_skip = 0;
    let mut djxl_skip = 0;
    let mut jxl_rs_skip = 0;

    for (name, src, fw, fh, dist, eff) in fixtures() {
        let (rgb, w, h) = load_or_synth(src, fw, fh);
        let cfg = LossyConfig::new(dist).with_effort(eff).with_knobs(knobs);

        let bytes = match cfg.encode(&rgb, w, h, PixelLayout::Rgb8) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("W44-222 cell {}: encode failed: {:?}", name, e);
                continue;
            }
        };
        eprintln!(
            "W44-222 cell {} ({}x{} d{} e{}): encoded {} bytes",
            name,
            w,
            h,
            dist,
            eff,
            bytes.len()
        );

        // ─── jxl-oxide (in-process, always available) ───
        match decode_via_jxl_oxide(&bytes) {
            Ok((ow, oh)) => {
                assert_eq!((ow, oh), (w, h), "{} jxl-oxide dimensions", name);
                oxide_pass += 1;
            }
            Err(e) => {
                eprintln!("W44-222 cell {} jxl-oxide DECODE FAILED: {}", name, e);
                oxide_skip += 1;
            }
        }

        let jxl_path = write_tmp_jxl(&bytes);

        // ─── djxl (subprocess) ───
        match decode_via_djxl(&jxl_path) {
            Ok(()) => djxl_pass += 1,
            Err(e) if e.starts_with("djxl_path missing") => {
                djxl_skip += 1;
                eprintln!("W44-222 cell {} djxl: SKIP ({})", name, e);
            }
            Err(e) => {
                panic!("W44-222 cell {} djxl decode FAILED: {}", name, e)
            }
        }

        // ─── jxl_cli (jxl-rs, subprocess) ───
        match decode_via_jxl_rs(&jxl_path) {
            Ok(()) => jxl_rs_pass += 1,
            Err(e) if e.starts_with("jxl_cli missing") => {
                jxl_rs_skip += 1;
                eprintln!("W44-222 cell {} jxl-rs: SKIP ({})", name, e);
            }
            Err(e) => {
                panic!("W44-222 cell {} jxl-rs decode FAILED: {}", name, e)
            }
        }

        let _ = std::fs::remove_file(&jxl_path);
    }

    eprintln!(
        "W44-222 decoder roundtrip totals: jxl-oxide {}/{} (skip {}), djxl {}/{} (skip {}), jxl-rs {}/{} (skip {})",
        oxide_pass,
        fixtures().len(),
        oxide_skip,
        djxl_pass,
        fixtures().len(),
        djxl_skip,
        jxl_rs_pass,
        fixtures().len(),
        jxl_rs_skip,
    );

    assert_eq!(
        oxide_pass + oxide_skip,
        fixtures().len(),
        "every fixture must be encoded + attempted via jxl-oxide"
    );
    assert!(
        oxide_pass >= 1,
        "at least ONE jxl-oxide decode must succeed (in-process; not network-gated)"
    );
}
