// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-2 byte report for `with_alpha_squeeze(true)` (W14-4 follow-on).
//!
//! Sweeps the three W13-4 audit images (commit `a160deb7`) at four
//! alpha distances and prints a TSV row per (image, distance) showing
//! the chunk-1 framework's no-squeeze baseline byte count vs the
//! chunk-2 squeeze pipeline. Multi-group images (>256×256) hit the
//! chunk-2.b NotImplemented gate and report `SQUEEZE_ERR`;
//! gradients_semitrans_ui (256×128) is single-group and exercises the
//! end-to-end new pipeline.
//!
//! Run:
//!   cargo run --release -p jxl-encoder --example alpha_squeeze_chunk2_bytes

use image::ImageReader;
use jxl_encoder::{LossyConfig, PixelLayout};

fn read_rgba8(path: &str) -> Option<(u32, u32, Vec<u8>)> {
    let img = ImageReader::open(path).ok()?.decode().ok()?;
    let rgba8 = img.to_rgba8();
    let (w, h) = rgba8.dimensions();
    Some((w, h, rgba8.into_raw()))
}

fn try_encode(buf: &[u8], w: u32, h: u32, ad: f32, squeeze: bool) -> Result<Vec<u8>, String> {
    let mut cfg = LossyConfig::new(1.0).with_alpha_distance(Some(ad));
    if squeeze {
        cfg = cfg.with_alpha_squeeze(true);
    }
    cfg.encode(buf, w, h, PixelLayout::Rgba8)
        .map_err(|e| format!("{e:?}"))
}

fn short_err(e: &str) -> String {
    let line = e.lines().next().unwrap_or("");
    line.chars().take(80).collect()
}

fn main() {
    let images: &[(&str, &str)] = &[
        (
            "gradients_semitrans_ui",
            "/home/lilith/work/codec-corpus/imageflow/test_inputs/gradients.png",
        ),
        (
            "red_night_opaque",
            "/home/lilith/work/codec-corpus/imageflow/test_inputs/red-night.png",
        ),
        (
            "alpha_nonpremul_photo_mask",
            "/home/lilith/work/codec-corpus/jxl/reference/conformance/alpha_nonpremultiplied.png",
        ),
    ];
    println!(
        "image\twidth\theight\talpha_distance\tbytes_no_squeeze\tbytes_squeeze\tdelta_bytes\tdelta_pct\tsqueeze_result"
    );
    for (label, path) in images {
        let Some((w, h, rgba)) = read_rgba8(path) else {
            println!("{label}\t?\t?\t-\t-\t-\t-\t-\tREAD_ERR:{path}");
            continue;
        };
        for &ad in &[0.5f32, 1.0, 2.0, 5.0] {
            let no_sq = try_encode(&rgba, w, h, ad, false);
            let sq = try_encode(&rgba, w, h, ad, true);
            match (&no_sq, &sq) {
                (Ok(nb), Ok(sb)) => {
                    let delta = sb.len() as i64 - nb.len() as i64;
                    let pct = 100.0 * delta as f64 / nb.len() as f64;
                    println!(
                        "{label}\t{w}\t{h}\t{ad:.1}\t{}\t{}\t{delta:+}\t{pct:+.2}\tOK",
                        nb.len(),
                        sb.len()
                    );
                }
                (Ok(nb), Err(e)) => {
                    println!(
                        "{label}\t{w}\t{h}\t{ad:.1}\t{}\t-\t-\t-\tSQUEEZE_ERR:{}",
                        nb.len(),
                        short_err(e)
                    );
                }
                (Err(e), _) => {
                    println!(
                        "{label}\t{w}\t{h}\t{ad:.1}\t-\t-\t-\t-\tBASE_ERR:{}",
                        short_err(e)
                    );
                }
            }
        }
    }
}
