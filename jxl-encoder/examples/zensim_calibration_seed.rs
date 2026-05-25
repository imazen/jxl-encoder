// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! zensim-fork Phase 4 (2026-05-25) calibration-seed generator.
//!
//! Mirrors the methodology of `scripts/cvvdp_calibration_seed.py` but
//! for zensim: encode a small held-out corpus with the butteraugli-
//! default buttloop at each distance in
//! `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}`, decode via jxl-oxide, score
//! with `zensim::Zensim` (native `[0, 100]` higher-is-better), and
//! emit a TSV that `scripts/zensim_calibration_seed.py` can consume.
//!
//! The pre-existing tracking TSV
//! (`benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`) does NOT
//! carry a `score_zensim` column — the cvvdp Phase 6 sweep ran before
//! the zensim metric scaffold was wired through. This example is the
//! small post-hoc scoring pass the Phase 4 brief calls for ("read
//! 10-20 butteraugli-encoded JXL outputs at distances `{0.5, 1.0, 1.5,
//! 2.0, 3.0, 4.0, 5.0}`, decode via jxl-oxide, score with zensim").
//!
//! ## Sample size
//!
//! The brief allows 10-20 cells per distance. We pick 3 images at each
//! distance × 7 distances = 21 cells total. Three sources:
//! - 2 CID22 validation photos (1418519, 1025469)
//! - 1 GB82-SC screenshot (codec_wiki)
//!
//! This is intentionally small — the seed values are a Phase 4
//! STARTING POINT, not a finished calibration. Phase 6 produces the
//! full 6-backend tracking sweep including `score_zensim_{cpu,gpu}`;
//! Phase 8-zensim (conditional) refits per RFC §3.2 Intervention A if
//! Pareto < 85%.
//!
//! ## Run via
//!
//! ```bash
//! cargo run --release -p jxl-encoder \
//!   --features "__expert butteraugli-loop zensim-loop ssim2-loop parallel" \
//!   --example zensim_calibration_seed -- \
//!   --output benchmarks/zensim_calibration_seed_2026-05-25.tsv
//! ```
//!
//! Then:
//!
//! ```bash
//! python3 scripts/zensim_calibration_seed.py \
//!   benchmarks/zensim_calibration_seed_2026-05-25.tsv \
//!   > benchmarks/zensim_calibration_seed_2026-05-25.txt
//! ```

#![cfg(feature = "zensim-loop")]

use std::fs::File;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};

const CID22_VAL_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// One image cell to score.
struct ImageCell {
    name: &'static str,
    corpus: &'static str,
    abs_path: String,
}

fn load_image_rgb8(
    abs_path: &str,
) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error + Send + Sync>> {
    let img = image::open(abs_path)?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    Ok((img.into_raw(), w, h))
}

fn encode_butteraugli_default(
    pixels: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let cfg = LossyConfig::new(distance)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_effort(effort);
    Ok(cfg.encode(pixels, w, h, PixelLayout::Rgb8)?)
}

/// Decode the JXL bytes through jxl-oxide and return tight-stride sRGB
/// u8 pixels (suitable for `zensim::RgbSlice`).
fn decode_jxl_srgb_u8(
    encoded: &[u8],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut decoder = jxl_oxide::JxlImage::builder().read(Cursor::new(encoded))?;
    // For zensim's RgbSlice we want sRGB encoded u8, NOT linear. The
    // default jxl-oxide output is the file-declared encoding which is
    // already sRGB for our encoder (we signal TransferFunction::Srgb).
    let frame = decoder.render_frame(0)?;
    let stream = frame.stream();
    let dec_w = stream.width();
    let dec_h = stream.height();
    if dec_w != w || dec_h != h {
        return Err(format!(
            "decoded dims mismatch: expected {}×{}, got {}×{}",
            w, h, dec_w, dec_h
        )
        .into());
    }
    let ch = stream.channels() as usize;
    let mut pixels_f32: Vec<f32> = vec![0.0; (dec_w as usize) * (dec_h as usize) * ch];
    let mut stream_mut = stream;
    let _ = stream_mut.write_to_buffer(&mut pixels_f32);

    // jxl-oxide returns float in `[0, 1]` (clamped). Convert to u8 and
    // emit only the first 3 channels (drop alpha if present).
    let n_px = (dec_w as usize) * (dec_h as usize);
    let mut rgb = Vec::with_capacity(n_px * 3);
    for i in 0..n_px {
        let r = (pixels_f32[i * ch].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = if ch >= 2 {
            (pixels_f32[i * ch + 1].clamp(0.0, 1.0) * 255.0).round() as u8
        } else {
            r
        };
        let b = if ch >= 3 {
            (pixels_f32[i * ch + 2].clamp(0.0, 1.0) * 255.0).round() as u8
        } else {
            r
        };
        rgb.extend_from_slice(&[r, g, b]);
    }
    Ok(rgb)
}

/// Score `(source_rgb8, decoded_rgb8)` with zensim and return the native
/// `[0, 100]` higher-is-better score.
fn zensim_score(
    source_rgb: &[u8],
    decoded_rgb: &[u8],
    w: u32,
    h: u32,
) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    use zensim::{RgbSlice, Zensim, ZensimProfile};

    // Pack as &[[u8; 3]] for RgbSlice.
    let source_chunks: &[[u8; 3]] = bytemuck::cast_slice(&source_rgb[..(w * h * 3) as usize]);
    let dist_chunks: &[[u8; 3]] = bytemuck::cast_slice(&decoded_rgb[..(w * h * 3) as usize]);

    let source = RgbSlice::new(source_chunks, w as usize, h as usize);
    let dist = RgbSlice::new(dist_chunks, w as usize, h as usize);

    let scorer = Zensim::new(ZensimProfile::PreviewV0_2);
    let res = scorer.compute(&source, &dist)?;
    Ok(res.score() as f64)
}

fn cells() -> Vec<ImageCell> {
    vec![
        ImageCell {
            name: "1418519",
            corpus: "CID22",
            abs_path: format!("{CID22_VAL_DIR}/1418519.png"),
        },
        ImageCell {
            name: "1025469",
            corpus: "CID22",
            abs_path: format!("{CID22_VAL_DIR}/1025469.png"),
        },
        ImageCell {
            name: "codec_wiki",
            corpus: "gb82-sc",
            abs_path: format!("{GB82_SC_DIR}/codec_wiki.png"),
        },
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let mut output_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            output_path = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else {
            eprintln!("Unknown arg: {}", args[i]);
            i += 1;
        }
    }
    let output_path = output_path
        .unwrap_or_else(|| PathBuf::from("benchmarks/zensim_calibration_seed_2026-05-25.tsv"));

    eprintln!(
        "[zensim_calibration_seed] output: {}",
        output_path.display()
    );

    let distances = [0.5_f32, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
    // Use effort 8 to ensure the buttloop actually runs (gated at
    // speed_tier <= kKitten = effort >= 8 in libjxl).
    let effort = 8;

    let mut f = File::create(&output_path)?;
    writeln!(
        f,
        "image\tcorpus\teffort\tdistance\tbackend\tbytes\tencode_ms\tscore_zensim_native"
    )?;

    let cells = cells();
    let total = cells.len() * distances.len();
    let mut idx = 0;
    for cell in &cells {
        eprintln!("[zensim_calibration_seed] loading {}", cell.abs_path);
        let (pixels, w, h) = load_image_rgb8(&cell.abs_path)?;
        eprintln!(
            "[zensim_calibration_seed]   {} pixels = {} ({} × {})",
            cell.name,
            pixels.len(),
            w,
            h
        );
        for &d in &distances {
            idx += 1;
            let t = Instant::now();
            let encoded = encode_butteraugli_default(&pixels, w, h, d, effort)?;
            let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
            let decoded = decode_jxl_srgb_u8(&encoded, w, h)?;
            let score = zensim_score(&pixels, &decoded, w, h)?;
            writeln!(
                f,
                "{name}\t{corpus}\t{effort}\t{d:.2}\tB\t{bytes}\t{encode_ms:.2}\t{score:.4}",
                name = cell.name,
                corpus = cell.corpus,
                effort = effort,
                d = d,
                bytes = encoded.len(),
                encode_ms = encode_ms,
                score = score,
            )?;
            eprintln!(
                "  [{idx}/{total}] {name} d={d:.2} bytes={bytes} encode={encode_ms:.1}ms zensim={score:.3}",
                idx = idx,
                total = total,
                name = cell.name,
                d = d,
                bytes = encoded.len(),
                encode_ms = encode_ms,
                score = score,
            );
        }
    }
    drop(f);
    eprintln!(
        "[zensim_calibration_seed] wrote {idx} rows to {}",
        output_path.display()
    );
    Ok(())
}
