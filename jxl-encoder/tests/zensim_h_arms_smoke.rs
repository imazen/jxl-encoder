// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! #69 smoke: the H1/H2/H3 loop-steering arms run end-to-end and produce
//! sane per-tile SIGNED steering fields. Own integration binary (single
//! test fn, sequential arms) so env mutation cannot race.

#![cfg(feature = "zensim-loop")]

use jxl_encoder::api::{EncoderStrategy, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

#[test]
fn h_arms_steer_with_signed_fields() {
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
    // SAFETY: edition-2024 set_var — single test fn in this binary; arms run
    // sequentially with no concurrent env readers.
    unsafe {
        std::env::set_var("JXL_ZENSIM_RD_PROFILE", "b");
    }
    for arm in ["h1-signed", "h2-ctrl", "h3-mag"] {
        let probe =
            std::env::temp_dir().join(format!("zensim_h_smoke_{arm}_{}.tsv", std::process::id()));
        let _ = std::fs::remove_file(&probe);
        // SAFETY: edition-2024 set_var — same single-test sequential contract
        // as above (no concurrent env readers in this binary).
        unsafe {
            std::env::set_var("JXL_ZENSIM_MODEL_MAP", arm);
            std::env::set_var("JXL_ZENSIM_ATTR_PROBE", &probe);
        }
        let encoded = LossyConfig::new(1.5)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_effort(6)
            .with_perceptual_metric(PerceptualMetric::Zensim)
            .with_butteraugli_iters(0)
            .with_zensim_iters(3)
            .encode(&pixels, w, h, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("{arm} encode: {e:?}"));
        assert!(!encoded.is_empty(), "{arm}: empty bitstream");
        let probe_txt = std::fs::read_to_string(&probe)
            .unwrap_or_else(|e| panic!("{arm}: probe not written: {e}"));
        let lines: Vec<&str> = probe_txt.lines().collect();
        assert!(!lines.is_empty(), "{arm}: steering never engaged");
        for l in &lines {
            let f: Vec<f64> = l
                .split('\t')
                .skip(1)
                .filter_map(|x| x.parse().ok())
                .collect();
            assert_eq!(f.len(), 3, "{arm}: malformed probe line {l}");
            assert!(
                f.iter().all(|v| v.is_finite()),
                "{arm}: non-finite steering stats {l}"
            );
            // Signed field: on textured content the tile field must not be
            // a constant (max > min) — degenerate steering would mean the
            // arm silently steers nothing.
            assert!(f[1] > f[0], "{arm}: degenerate constant tile field {l}");
        }
        let _ = std::fs::remove_file(&probe);
    }
    // September 8 complete-candidate route. This fixture is shipped D,
    // SHA256 cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6.
    // Synthetic pixels exercise wiring only; real-image RD lives in the harness.
    let model_bytes = include_bytes!("support/zensim_d_byid_2026-09-06.bin");
    let dir = std::env::temp_dir().join(format!("zensim_candidate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bake = dir.join("model.bin");
    let probe = dir.join("map.tsv");
    std::fs::write(&bake, model_bytes).unwrap();
    let encode_candidate = |gain: &str| {
        // SAFETY: the only test in this binary; no concurrent encodes/readers.
        unsafe {
            std::env::set_var("JXL_ZENSIM_RD_PROFILE", format!("bake:{}", bake.display()));
            std::env::set_var("JXL_ZENSIM_MODEL_MAP", "h3-mag");
            std::env::set_var("ZENSIM_H3_GAIN", gain);
            std::env::set_var("JXL_ZENSIM_ATTR_PROBE", &probe);
            std::env::remove_var("JXL_ZENSIM_TARGET_SCORE");
        }
        LossyConfig::new(1.5)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_effort(6)
            .with_perceptual_metric(PerceptualMetric::Zensim)
            .with_butteraugli_iters(0)
            .with_zensim_iters(2)
            .encode(&pixels, w, h, PixelLayout::Rgb8)
    };
    let neutral = encode_candidate("0").expect("candidate neutral encode");
    std::fs::write(&probe, "").unwrap();
    let active = encode_candidate("10").expect("candidate active encode");
    assert_ne!(active, neutral, "candidate map never changed emitted bytes");
    let active_pixels = decode::verify_jxl_rs(&active, w as usize, h as usize);
    let neutral_pixels = decode::verify_jxl_rs(&neutral, w as usize, h as usize);
    assert_ne!(
        active_pixels, neutral_pixels,
        "map changed only container bytes"
    );
    let trace = std::fs::read_to_string(&probe).unwrap();
    assert_eq!(
        trace.lines().count(),
        3,
        "fresh map must cover each reconstruction"
    );
    for line in trace.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 4, "existing trace contract changed");
        let values: Vec<f64> = fields[1..].iter().map(|s| s.parse().unwrap()).collect();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values[1] > values[0], "candidate map is constant");
    }
    // A process-global first-bake cache would wrongly accept the overwritten
    // path and reproduce the prior output. Loading per encode must see failure.
    std::fs::write(&bake, b"invalid model").unwrap();
    assert!(
        encode_candidate("10").is_err(),
        "stale model cache hid changed bytes"
    );
    std::fs::write(&bake, model_bytes).unwrap();
    assert_eq!(
        encode_candidate("10").unwrap(),
        active,
        "per-image state leaked"
    );
    // SAFETY: sequential test has finished all encodes.
    unsafe {
        for name in [
            "JXL_ZENSIM_RD_PROFILE",
            "JXL_ZENSIM_MODEL_MAP",
            "ZENSIM_H3_GAIN",
            "JXL_ZENSIM_ATTR_PROBE",
        ] {
            std::env::remove_var(name);
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
