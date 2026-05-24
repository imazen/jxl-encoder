// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-PHASE3-B5b — multi-decoder roundtrip on bitstreams produced by the
//! GPU butteraugli backend with the iter-0 divergence detector ENABLED.
//!
//! Smoke gates that the detector path:
//!   (a) Doesn't break encoding on the gpu-cuda-DETECTOR codepath.
//!   (b) Produces a bitstream that jxl-oxide AND jxl-rs decode cleanly.
//!   (c) Increments the b5b_counters' `run_count` (≥1) for the cell.
//!   (d) Did NOT trip on this synthetic 256×256 fixture (the detector
//!       only triggers on rare cumulative-drift cells; synthetic
//!       checkerboards aren't divergent).
//!
//! Requires the `gpu-butteraugli` cargo feature AND CUDA at
//! `/usr/local/cuda`. The test is `#[ignore]` so a vanilla
//! `cargo test --features gpu-butteraugli` skips it; run via
//!   cargo test --release \
//!     --features 'gpu-butteraugli butteraugli-loop parallel __expert' \
//!     --test w44_phase3_b5b_divergence_detector_roundtrip -- \
//!     --ignored --nocapture

#![cfg(feature = "gpu-butteraugli")]

use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_encoder::vardct::__b5b_counters as b5b_counters;
use std::io::Cursor;

/// Tiny synthetic checkerboard image. 256×256 is the smallest size worth
/// running through the GPU backend (smaller would test the construct-but-
/// bypass path).
fn synth_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let gx = (x as f32) / (w as f32);
            let gy = (y as f32) / (h as f32);
            let checker = ((x / 16) + (y / 16)) & 1;
            let base = (gx * 200.0 + gy * 55.0).clamp(0.0, 255.0) as u8;
            let v = if checker == 1 {
                base
            } else {
                base.saturating_sub(80)
            };
            out.push(v);
            out.push((v as u32 * 7 / 10) as u8);
            out.push((255 - v) / 2);
        }
    }
    out
}

fn decode_oxide(bytes: &[u8]) -> Result<(usize, usize), String> {
    let reader = Cursor::new(bytes);
    let img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .map_err(|e| format!("oxide read: {e}"))?;
    let r = img
        .render_frame(0)
        .map_err(|e| format!("oxide render: {e}"))?;
    let fb = r.image_all_channels();
    Ok((fb.width(), fb.height()))
}

fn decode_jxl_rs(bytes: &[u8]) -> Result<(usize, usize), String> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = bytes;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("jxl-rs: unexpected end of input during header".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs header: {e:?}")),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let channels = 3;

    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("jxl-rs: unexpected end of input before frame".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame info: {e:?}")),
        }
    };

    let mut output_image = Image::<f32>::new((width * channels, height))
        .map_err(|e| format!("jxl-rs alloc: {e:?}"))?;
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("jxl-rs: unexpected end of input during frame decode".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame: {e:?}")),
        }
    }
    Ok((width, height))
}

#[test]
#[ignore = "requires CUDA at /usr/local/cuda; run via cargo test --release --features 'gpu-butteraugli butteraugli-loop parallel __expert' -- --ignored"]
fn b5b_detector_path_roundtrips_via_oxide() {
    // Reset counters BEFORE setting the env so the snapshot reflects
    // only this encode.
    b5b_counters::reset();
    // SAFETY: tests run single-threaded by default; env mutation is
    // for the duration of the test only. We clear the env after.
    unsafe { std::env::set_var("JXL_W44_PHASE3_B5B_DETECTOR", "1") };

    let rgb = synth_rgb(256, 256);
    let cfg = LossyConfig::new(2.0)
        .with_effort(8) // e8 fires the buttloop
        .with_gpu_butteraugli(true)
        .with_threads(1);
    let bytes = cfg
        .encode(&rgb, 256, 256, PixelLayout::Rgb8)
        .expect("encode failed under GPU butteraugli + detector");

    unsafe { std::env::remove_var("JXL_W44_PHASE3_B5B_DETECTOR") };

    let snap = b5b_counters::snapshot();
    eprintln!(
        "[B5b roundtrip oxide] run_count={} fallback_count={} \
         div_pct_max={:.6} div_pct_sum={:.6}",
        snap.run_count, snap.fallback_count, snap.divergence_pct_max, snap.divergence_pct_sum,
    );

    // The detector should have run at least once (we set env, GPU active,
    // shadow CPU built, iter-0 detector fired).
    assert!(
        snap.run_count >= 1,
        "detector run_count should be ≥ 1; got {}",
        snap.run_count
    );
    // On a synthetic checkerboard the detector should NOT trip (synthetic
    // fixtures don't exhibit the cumulative-drift mechanism). If this
    // assertion ever fires, the detector's threshold has been lowered to
    // a value that catches synthetic inputs, which means it would also
    // trigger on the production 36 non-divergent cells (false-positive
    // regression).
    assert_eq!(
        snap.fallback_count, 0,
        "detector should NOT trip on synthetic 256×256 checkerboard; \
         got fallback_count={}",
        snap.fallback_count
    );

    assert!(
        bytes.len() > 100,
        "encoded output too small: {}",
        bytes.len()
    );
    let (w, h) = decode_oxide(&bytes).expect("jxl-oxide failed to decode B5b output");
    assert_eq!((w, h), (256, 256));
}

#[test]
#[ignore = "requires CUDA at /usr/local/cuda; run via cargo test --release --features 'gpu-butteraugli butteraugli-loop parallel __expert' -- --ignored"]
fn b5b_detector_path_roundtrips_via_jxl_rs() {
    b5b_counters::reset();
    unsafe { std::env::set_var("JXL_W44_PHASE3_B5B_DETECTOR", "1") };

    let rgb = synth_rgb(256, 256);
    let cfg = LossyConfig::new(2.0)
        .with_effort(8)
        .with_gpu_butteraugli(true)
        .with_threads(1);
    let bytes = cfg
        .encode(&rgb, 256, 256, PixelLayout::Rgb8)
        .expect("encode failed under GPU butteraugli + detector");

    unsafe { std::env::remove_var("JXL_W44_PHASE3_B5B_DETECTOR") };

    let snap = b5b_counters::snapshot();
    assert!(snap.run_count >= 1, "detector run_count should be ≥ 1");

    let (w, h) = decode_jxl_rs(&bytes).expect("jxl-rs failed to decode B5b output");
    assert_eq!((w, h), (256, 256));
}

// NOTE: a "dormant without env" test was removed — it was racy because
// the process-global `b5b_counters` atomics are shared across all tests
// in a binary, and `cargo test` runs tests in parallel by default. Other
// tests in this file mutate the env var around encode calls, which means
// a concurrent "dormant" test would observe the OTHER tests' detector
// invocations as `run_count > 0`. The default-OFF path is instead
// covered by:
//   - `cpu_backend_divergence_status_always_none` (CPU backend ignores
//     the env var entirely)
//   - `b5b_counters_reset_zero_state` (reset clears counters)
//   - `b5b_counters_record_round_trip` (record + snapshot round-trip)
// — all in the in-source `vardct::perceptual_backend::tests` module,
// which is GPU-feature-gated but doesn't invoke the encoder at all so
// is not subject to the env-var race.
