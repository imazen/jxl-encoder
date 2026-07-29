// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! C3b smoke test (zensim task #67): `JXL_ZENSIM_MODEL_MAP=attr` runs the
//! fused score+attribution steering path end-to-end and produces sane
//! per-tile steering values.
//!
//! Lives in its own integration-test binary so the env mutation cannot race
//! another test (each integration test file is a separate process, and this
//! file contains exactly one test).

#![cfg(feature = "zensim-loop")]

use jxl_encoder::api::{EncoderStrategy, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

#[test]
fn model_map_attr_steers_with_fused_map() {
    let (w, h) = (128u32, 128u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40u8 } else { 0 };
            pixels.extend_from_slice(&[
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ]);
        }
    }

    let probe = std::env::temp_dir().join(format!(
        "zensim_attr_smoke_probe_{}.tsv",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&probe);
    // SAFETY (edition-2024 set_var): this integration-test binary contains
    // exactly one test — no concurrent env readers in this process.
    unsafe {
        std::env::set_var("JXL_ZENSIM_RD_PROFILE", "b"); // embedded 372 profile
        std::env::set_var("JXL_ZENSIM_MODEL_MAP", "attr");
        std::env::set_var("JXL_ZENSIM_ATTR_PROBE", &probe);
    }

    let encoded = LossyConfig::new(1.5)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_effort(6)
        .with_perceptual_metric(PerceptualMetric::Zensim)
        .with_butteraugli_iters(0)
        .with_zensim_iters(3)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("attr-steered encode");
    assert!(!encoded.is_empty(), "empty bitstream");

    // The probe must show >= 1 attr-steered iteration (iterations 2+) with
    // finite, non-degenerate tile-dist stats.
    let probe_txt = std::fs::read_to_string(&probe).expect("attr probe written");
    let lines: Vec<&str> = probe_txt.lines().collect();
    assert!(
        !lines.is_empty(),
        "attr steering never engaged (no probe lines)"
    );
    for l in &lines {
        let f: Vec<f64> = l
            .split('\t')
            .skip(1)
            .filter_map(|x| x.parse().ok())
            .collect();
        assert_eq!(f.len(), 3, "probe line malformed: {l}");
        let (mn, mx, mean) = (f[0], f[1], f[2]);
        assert!(
            mn.is_finite() && mx.is_finite() && mean.is_finite(),
            "non-finite steering stats: {l}"
        );
        assert!(mx > 0.0, "tile steering all-zero: {l}");
        assert!(mn >= 0.0, "negative tile dist after anchor blend: {l}");
        assert!(mx >= mean && mean >= mn, "inconsistent stats: {l}");
    }
    let _ = std::fs::remove_file(&probe);
}
