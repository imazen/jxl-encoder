// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Demonstrate the libjxl-parity knobs that landed in the
//! 2026-05-12 session. Synthesises a small RGB image (no corpus
//! dependency), encodes it with each knob set, prints file sizes,
//! and verifies each output decodes via jxl-oxide.
//!
//! Run with: `cargo run --release --example libjxl_parity_knobs`
//!
//! The knobs demonstrated mirror libjxl's `cparams.*` fields that
//! `cjxl` exposes via command-line flags:
//!
//! - `with_photon_noise_iso(Some(iso))` ↔ `--photon_noise=ISO`
//! - `with_manual_noise_lut(Some(lut))` ↔ `cparams.manual_noise`
//! - `with_original_distance(Some(d))` ↔ `--original_butteraugli_distance`
//! - `with_quant_ac_rescale(Some(r))` ↔ `--quant_ac_rescale`
//! - `with_already_downsampled(true)` ↔ `cparams.already_downsampled`
//! - `with_force_rct(Some(rct))` ↔ `--colorspace` (lossless)
//! - `with_tree_learning_sample_fraction(f)` ↔ effort-7 cliff (#23)
//! - `estimate_peak_memory_bytes(...)` (no libjxl equivalent;
//!   capacity-planning helper)

use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout, RctType};
use std::io::Cursor;

fn main() {
    let w = 256u32;
    let h = 256u32;
    let pixels = synth_rgb_image(w, h);

    println!("Source: {w}x{h} RGB8 ({} bytes)\n", pixels.len());

    // ── Lossy: capacity planning ──
    let est = LossyConfig::new(1.0)
        .estimate_peak_memory_bytes(w, h, PixelLayout::Rgb8)
        .expect("estimate");
    println!(
        "LossyConfig::estimate_peak_memory_bytes: ~{} MB working set",
        est / (1024 * 1024)
    );

    // ── Lossy: baseline ──
    let baseline = encode_lossy(&pixels, w, h, |c| c);
    println!(
        "\nlossy d=2.0 e5 baseline:                  {} bytes",
        baseline
    );

    // ── Lossy: photon noise ──
    let with_photon = encode_lossy(&pixels, w, h, |c| c.with_photon_noise_iso(Some(3200.0)));
    println!(
        "lossy + with_photon_noise_iso(3200):       {} bytes",
        with_photon
    );

    // ── Lossy: manual noise LUT ──
    let manual_lut = [0.05f32, 0.10, 0.15, 0.20, 0.15, 0.10, 0.05, 0.0];
    let with_manual = encode_lossy(&pixels, w, h, |c| c.with_manual_noise_lut(Some(manual_lut)));
    println!(
        "lossy + with_manual_noise_lut(...):        {} bytes",
        with_manual
    );

    // ── Lossy: original distance (re-encode pipeline) ──
    let with_orig = encode_lossy(&pixels, w, h, |c| c.with_original_distance(Some(5.0)));
    println!(
        "lossy + with_original_distance(5.0):       {} bytes",
        with_orig
    );

    // ── Lossy: quant_ac_rescale (finer AC quant) ──
    let with_rescale = encode_lossy(&pixels, w, h, |c| c.with_quant_ac_rescale(Some(0.85)));
    println!(
        "lossy + with_quant_ac_rescale(0.85):       {} bytes",
        with_rescale
    );

    // ── Lossy: already_downsampled (skip internal downsample) ──
    // Caller passes a 256x256 image but tells the encoder to declare
    // it as 512x512 with 2x upsampling. The decoded image will be
    // 512x512 (file header), even though our input was 256x256.
    let with_already = encode_lossy(&pixels, w, h, |c| {
        c.with_resampling(2).with_already_downsampled(true)
    });
    println!(
        "lossy + with_already_downsampled+resamp2:  {} bytes",
        with_already
    );
    verify_dims(
        &encode_lossy_full(&pixels, w, h, |c| {
            c.with_resampling(2).with_already_downsampled(true)
        }),
        w * 2,
        h * 2,
    );

    // ── Lossless: baseline ──
    let lossless_baseline = encode_lossless(&pixels, w, h, |c| c);
    println!(
        "\nlossless e7 baseline:                     {} bytes",
        lossless_baseline
    );

    // ── Lossless: forced YCoCg RCT ──
    let with_force_rct = encode_lossless(&pixels, w, h, |c| c.with_force_rct(Some(RctType::YCOCG)));
    println!(
        "lossless + with_force_rct(YCOCG):          {} bytes",
        with_force_rct
    );

    // ── Lossless: tree-learning lite (refs #23) ──
    let with_tree_lite = encode_lossless(&pixels, w, h, |c| {
        c.with_tree_learning_sample_fraction(0.15)
    });
    println!(
        "lossless + tree_learning_sample_frac(0.15): {} bytes",
        with_tree_lite
    );

    println!("\nAll outputs verified to decode via jxl-oxide.");
}

/// Build a synthetic 256x256 RGB image with structure (smooth ramps,
/// a bright circle, sparse bright dots) so the encoder has something
/// non-trivial to compress.
fn synth_rgb_image(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            // Diagonal gradient in R, horizontal in G, vertical in B
            pixels[idx] = ((x + y) * 255 / (w + h)) as u8;
            pixels[idx + 1] = (x * 255 / w) as u8;
            pixels[idx + 2] = (y * 255 / h) as u8;
            // Bright circle in the middle
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt();
            if r < 40.0 {
                pixels[idx] = 240;
                pixels[idx + 1] = 240;
                pixels[idx + 2] = 240;
            }
        }
    }
    // Sparse bright dots.
    for &(dx, dy) in &[(40, 40), (200, 60), (60, 200), (200, 200)] {
        let idx = ((dy * w + dx) * 3) as usize;
        pixels[idx] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 0;
    }
    pixels
}

fn encode_lossy(
    pixels: &[u8],
    w: u32,
    h: u32,
    customize: impl FnOnce(LossyConfig) -> LossyConfig,
) -> usize {
    let bytes = encode_lossy_full(pixels, w, h, customize);
    let len = bytes.len();
    verify_decode(&bytes);
    len
}

fn encode_lossy_full(
    pixels: &[u8],
    w: u32,
    h: u32,
    customize: impl FnOnce(LossyConfig) -> LossyConfig,
) -> Vec<u8> {
    let cfg = customize(LossyConfig::new(2.0).with_effort(5));
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("lossy encode failed")
}

fn encode_lossless(
    pixels: &[u8],
    w: u32,
    h: u32,
    customize: impl FnOnce(LosslessConfig) -> LosslessConfig,
) -> usize {
    let cfg = customize(LosslessConfig::new().with_effort(7));
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("lossless encode failed");
    let len = bytes.len();
    verify_decode(&bytes);
    len
}

fn verify_decode(bytes: &[u8]) {
    let _image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .expect("jxl-oxide parse")
        .render_frame(0)
        .expect("jxl-oxide render");
}

fn verify_dims(bytes: &[u8], expected_w: u32, expected_h: u32) {
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .expect("jxl-oxide parse");
    assert_eq!(image.width(), expected_w);
    assert_eq!(image.height(), expected_h);
}
