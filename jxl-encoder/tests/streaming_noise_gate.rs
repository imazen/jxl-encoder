// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression tests for the streaming `LossyEncoder` photon-noise /
//! manual-noise-LUT silent-drop gate (A1 audit top-10 #2, 2026-05-17).
//!
//! Pre-fix the streaming `LossyConfig::encoder` → `LossyEncoder::finish`
//! path in `api.rs::finish_inner` only wired `enc.photon_noise_iso`;
//! `cfg.manual_noise_lut`, `cfg.quant_ac_rescale`, and
//! `cfg.original_distance` were silently dropped. The one-shot
//! `EncodeRequest::encode_lossy` (api.rs:4531) and animation
//! `encode_animation_lossy` (api.rs:6892) paths wired all fields
//! correctly; the streaming path was the divergent gate.
//!
//! These tests pair the streaming encoder against the no-noise default —
//! if the noise field is wired the outputs differ (noise header is
//! emitted), if it's dropped the outputs are byte-identical (the bug).

use jxl_encoder::{LossyConfig, PixelLayout};

/// 64x64 solid mid-grey RGB — enough to force a noise header without
/// pulling in any image fixtures.
fn solid_grey(w: u32, h: u32) -> Vec<u8> {
    vec![128u8; (w as usize) * (h as usize) * 3]
}

fn encode_streaming(cfg: LossyConfig, w: u32, h: u32, pixels: &[u8]) -> Vec<u8> {
    let mut enc = cfg
        .encoder(w, h, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("encoder() failed: {e:?}"));
    let stride = (w as usize) * 3;
    for row in pixels.chunks_exact(stride) {
        enc.push_rows(row, 1)
            .unwrap_or_else(|e| panic!("push_rows failed: {e:?}"));
    }
    enc.finish()
        .unwrap_or_else(|e| panic!("finish() failed: {e:?}"))
}

#[test]
fn streaming_lossy_photon_noise_iso_emits_noise_header() {
    let (w, h) = (64u32, 64u32);
    let pixels = solid_grey(w, h);

    let no_noise = encode_streaming(LossyConfig::new(1.0), w, h, &pixels);
    let with_photon = encode_streaming(
        LossyConfig::new(1.0).with_photon_noise_iso(Some(800.0)),
        w,
        h,
        &pixels,
    );

    eprintln!(
        "[streaming photon-noise] no_noise={} with_photon_iso800={} delta={}",
        no_noise.len(),
        with_photon.len(),
        with_photon.len() as i64 - no_noise.len() as i64,
    );

    assert_ne!(
        no_noise, with_photon,
        "with_photon_noise_iso(Some(800.0)) produced byte-identical output to \
         no-noise streaming encode. The photon-noise header was not emitted — \
         streaming finish_inner silently dropped the photon_noise_iso setting."
    );
}

#[test]
fn streaming_lossy_manual_noise_lut_emits_noise_header() {
    let (w, h) = (64u32, 64u32);
    let pixels = solid_grey(w, h);

    let no_noise = encode_streaming(LossyConfig::new(1.0), w, h, &pixels);
    let lut = [0.0_f32, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35];
    let with_manual = encode_streaming(
        LossyConfig::new(1.0).with_manual_noise_lut(Some(lut)),
        w,
        h,
        &pixels,
    );

    eprintln!(
        "[streaming manual-noise-lut] no_noise={} with_manual={} delta={}",
        no_noise.len(),
        with_manual.len(),
        with_manual.len() as i64 - no_noise.len() as i64,
    );

    assert_ne!(
        no_noise, with_manual,
        "with_manual_noise_lut produced byte-identical output to no-noise \
         streaming encode. The noise header was not emitted — streaming \
         finish_inner silently dropped the manual_noise_lut setting."
    );
}

#[test]
fn streaming_lossy_quant_ac_rescale_changes_output() {
    let (w, h) = (64u32, 64u32);
    // Use a non-uniform image — solid grey produces tiny bitstreams where
    // the rescale multiplier may round to the same global_scale.
    let mut pixels = vec![0u8; (w as usize) * (h as usize) * 3];
    for (idx, b) in pixels.iter_mut().enumerate() {
        *b = ((idx * 13) & 0xFF) as u8;
    }

    let baseline = encode_streaming(LossyConfig::new(1.0), w, h, &pixels);
    let rescaled = encode_streaming(
        LossyConfig::new(1.0).with_quant_ac_rescale(Some(0.5)),
        w,
        h,
        &pixels,
    );

    eprintln!(
        "[streaming quant_ac_rescale] baseline={} rescale=0.5={} delta={}",
        baseline.len(),
        rescaled.len(),
        rescaled.len() as i64 - baseline.len() as i64,
    );

    assert_ne!(
        baseline, rescaled,
        "with_quant_ac_rescale(Some(0.5)) produced byte-identical output to \
         the default streaming encode. The rescale multiplier was not wired \
         through — streaming finish_inner silently dropped quant_ac_rescale."
    );
}
