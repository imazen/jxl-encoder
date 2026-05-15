// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Layer-1 invariant test for the quality-drift investigation
//! (memory/quality_drift_investigation_2026-05-15.md).
//!
//! The butteraugli quantization loop (effort >= 8) iteratively refines the
//! per-block quant_field by measuring butteraugli on its own internal
//! reconstruction: `reconstruct_xyb → gab_smooth → EPF → add_patches →
//! add_splines → xyb_to_linear_rgb_planar`. The hypothesis is that this
//! internal recon DIVERGES from what the user-facing decoder produces from
//! the SHIPPED bitstream — so the loop converges to a target image the
//! decoder never delivers, explaining the e8 quality-targeting drift
//! identified in the investigation (overshoot at low d, undershoot at
//! high d).
//!
//! This test:
//!   1. Encodes a real CID22 photo at d=2.0 e8 (buttloop ON, 2 iters by default).
//!   2. Captures the buttloop's INTERNAL recon at the final iteration via the
//!      `__internal_recon_hook` feature gate.
//!   3. Decodes the SAME shipped bitstream via jxl-rs (primary decoder per
//!      CLAUDE.md).
//!   4. Linearizes jxl-rs's sRGB f32 output to match the internal recon's
//!      linear RGB color space.
//!   5. Compares per-pixel: per-channel max-abs-diff and mean-abs-diff.
//!   6. Asserts max-abs-diff < 1e-3 in linear sRGB. If this FAILS, the
//!      buttloop is targeting an image the decoder cannot deliver — that
//!      divergence IS the drift's smoking gun.
//!
//! Per CLAUDE.md "NEVER relax test expectations": the threshold is
//! intentionally tight. If the test fails on first run, the failure is
//! the bug, not the test. The test is `#[ignore]` so CI passes; remove
//! the ignore once the divergence is fixed (or repurpose it as a regression
//! gate for the fix).

#![cfg(all(feature = "__internal_recon_hook", feature = "butteraugli-loop"))]

use std::path::PathBuf;

use jxl_encoder::vardct::__recon_hook;
use jxl_encoder::{LossyConfig, PixelLayout};

/// Convert sRGB normalized [0,1] f32 to linear light using the sRGB transfer
/// function (matches `butteraugli::srgb_to_linear` and the existing
/// `srgb_to_linear_val` helpers in `tests/llf_invariants.rs`).
fn srgb_to_linear_val(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Decode a JXL bitstream with jxl-rs (the PRIMARY decoder per CLAUDE.md).
/// jxl-rs returns f32 in the file's signaled color space — for our encoder
/// that's sRGB (with sRGB transfer function applied). The caller is expected
/// to linearize with `srgb_to_linear_val` if linear-light comparison is needed.
fn decode_jxl_rs(data: &[u8]) -> (usize, usize, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;

    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during header");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {:?}", e),
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
                    panic!("jxl-rs: unexpected end of input before frame");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info decode error: {:?}", e),
        }
    };

    let mut output_image = Image::<f32>::new((width * channels, height))
        .expect("jxl-rs: failed to create output buffer");

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
                    panic!("jxl-rs: unexpected end of input during frame decode");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame decode error: {:?}", e),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }

    (width, height, pixels)
}

/// Source PNG path for the drift Layer-1 test. Defaults to the same CID22
/// detailed-photo image used by the drift investigation
/// (`memory/quality_drift_investigation_2026-05-15.md`). Override via env var
/// `BUTTLOOP_RECON_PARITY_IMAGE` for ad-hoc bisection on other content.
fn source_image_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BUTTLOOP_RECON_PARITY_IMAGE") {
        return Some(PathBuf::from(p));
    }
    // Match the drift investigation's "detailed photo": cid22/1025469 (512x512).
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join("work/codec-corpus/CID22/CID22-512/validation/1025469.png");
    if p.exists() { Some(p) } else { None }
}

/// Layer-1 drift invariant: the buttloop's internal reconstruction (the image
/// it measures butteraugli against on its final iteration) MUST match what
/// jxl-rs decodes from the shipped bitstream. Divergence above the tight
/// threshold pinpoints the e8 quality-targeting drift root cause.
///
/// Distance: 2.0 (matches the drift investigation's mid-distance band where
/// the e8 buttloop still actively refines but the bitstream divergence is
/// observable).
///
/// **CURRENTLY EXPECTED TO FAIL** — that failure IS the drift's smoking gun.
/// The threshold is intentionally tight per CLAUDE.md "NEVER relax test
/// expectations". When the buttloop is fixed (e.g. the loop measures on a
/// real encode→decode roundtrip instead of an internal recon, or the internal
/// recon is corrected to match the decoder's pipeline byte-for-byte), this
/// test will start passing and the `#[ignore]` can be removed.
#[test]
#[ignore = "intentional - drift root cause; see memory/quality_drift_investigation_2026-05-15.md"]
fn buttloop_internal_recon_matches_jxl_rs_decode() {
    let src_path = source_image_path().unwrap_or_else(|| {
        panic!(
            "test source image not found. Set BUTTLOOP_RECON_PARITY_IMAGE=/path/to/photo.png \
             or stage CID22 at $HOME/work/codec-corpus/CID22/CID22-512/validation/1025469.png. \
             Synthetic images are forbidden per CLAUDE.md \"No Synthetic-Only Quality Tests\"."
        )
    });

    let img = image::open(&src_path)
        .unwrap_or_else(|e| panic!("failed to open {}: {}", src_path.display(), e));
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb_u8 = img.to_rgb8();

    // Mirror the public-API path used by `LossyConfig::encode(...)` exactly:
    // `LossyConfig` does the sRGB-u8 → linear-f32 conversion internally for
    // PixelLayout::Rgb8, so we just hand it the sRGB u8 buffer.
    let pixels: Vec<u8> = rgb_u8.into_raw();

    // Distance 2.0, effort 8. Effort 8 enables the butteraugli loop with 2
    // iterations by default (per `EffortProfile::lossy(8, ...)`), which is
    // exactly what the drift investigation exercised.
    let cfg = LossyConfig::new(2.0).with_effort(8);

    // Drain any prior recon (paranoia — mutex is process-global).
    let _ = __recon_hook::take_last();
    __recon_hook::set_capture_enabled(true);

    let bitstream = cfg
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
        .expect("encode failed");

    __recon_hook::set_capture_enabled(false);

    let recon = __recon_hook::take_last().expect(
        "buttloop did not capture an internal recon — verify (a) effort 8 has \
         butteraugli_iters > 0 in EffortProfile::lossy and (b) the encoder \
         routes through encoder.rs::encode_inner (not animation/precomputed). \
         If this assertion fires, the test source path or the encode hook is \
         the bug, not the drift.",
    );

    assert_eq!(recon.width, w, "recon width mismatch");
    assert_eq!(recon.height, h, "recon height mismatch");
    assert_eq!(
        recon.iter, recon.iters,
        "captured recon must be from the FINAL iteration"
    );

    // Decode the SHIPPED bitstream with jxl-rs (primary decoder).
    let (dec_w, dec_h, jxl_rs_pixels) = decode_jxl_rs(&bitstream);
    assert_eq!(dec_w, w, "jxl-rs decoded width mismatch");
    assert_eq!(dec_h, h, "jxl-rs decoded height mismatch");

    // jxl-rs returns f32 in the file's signaled color space (sRGB for our
    // encoder). Linearize to match the internal recon's linear-light RGB.
    let n_pixels = w * h;
    let mut max_abs_per_ch = [0.0f64; 3];
    let mut sum_abs_per_ch = [0.0f64; 3];
    let mut sum_sq_per_ch = [0.0f64; 3];

    let recon_planes = [&recon.r, &recon.g, &recon.b];
    for i in 0..n_pixels {
        // jxl-rs interleaved [R0,G0,B0,R1,G1,B1,...]
        let dec_r_srgb = jxl_rs_pixels[i * 3];
        let dec_g_srgb = jxl_rs_pixels[i * 3 + 1];
        let dec_b_srgb = jxl_rs_pixels[i * 3 + 2];
        let dec_linear = [
            srgb_to_linear_val(dec_r_srgb.clamp(0.0, 1.0)),
            srgb_to_linear_val(dec_g_srgb.clamp(0.0, 1.0)),
            srgb_to_linear_val(dec_b_srgb.clamp(0.0, 1.0)),
        ];
        for c in 0..3 {
            // Buttloop recon is unclamped linear-light; clamp before diff so
            // out-of-gamut floats don't pollute the metric (jxl-rs clamps to
            // its display range, the encoder doesn't).
            let recon_c = recon_planes[c][i].clamp(0.0, 1.0);
            let d = (dec_linear[c] - recon_c).abs() as f64;
            if d > max_abs_per_ch[c] {
                max_abs_per_ch[c] = d;
            }
            sum_abs_per_ch[c] += d;
            sum_sq_per_ch[c] += d * d;
        }
    }

    let mean_abs_per_ch: [f64; 3] = [
        sum_abs_per_ch[0] / n_pixels as f64,
        sum_abs_per_ch[1] / n_pixels as f64,
        sum_abs_per_ch[2] / n_pixels as f64,
    ];
    let rms_per_ch: [f64; 3] = [
        (sum_sq_per_ch[0] / n_pixels as f64).sqrt(),
        (sum_sq_per_ch[1] / n_pixels as f64).sqrt(),
        (sum_sq_per_ch[2] / n_pixels as f64).sqrt(),
    ];

    let overall_max_abs = max_abs_per_ch[0]
        .max(max_abs_per_ch[1])
        .max(max_abs_per_ch[2]);
    let overall_mean_abs = (mean_abs_per_ch[0] + mean_abs_per_ch[1] + mean_abs_per_ch[2]) / 3.0;

    eprintln!(
        "buttloop_recon_parity: image={} ({}x{}) d=2.0 e8 buttloop iters={}",
        src_path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        w,
        h,
        recon.iters
    );
    eprintln!(
        "  per-channel max-abs-diff (linear RGB): R={:.6} G={:.6} B={:.6}",
        max_abs_per_ch[0], max_abs_per_ch[1], max_abs_per_ch[2]
    );
    eprintln!(
        "  per-channel mean-abs-diff (linear RGB): R={:.6} G={:.6} B={:.6}",
        mean_abs_per_ch[0], mean_abs_per_ch[1], mean_abs_per_ch[2]
    );
    eprintln!(
        "  per-channel RMS-diff (linear RGB):      R={:.6} G={:.6} B={:.6}",
        rms_per_ch[0], rms_per_ch[1], rms_per_ch[2]
    );
    eprintln!(
        "  overall max-abs={:.6}  overall mean-abs={:.6}",
        overall_max_abs, overall_mean_abs
    );

    // Tight threshold: in pure linear sRGB on [0,1], 1e-3 is ~0.1% of the
    // dynamic range. If the buttloop's recon and the decoder's output diverge
    // more than this, the loop is targeting an image the decoder cannot
    // produce. This threshold is intentionally tight per CLAUDE.md "NEVER
    // relax test expectations". If it fails, that's the drift bug.
    const MAX_ABS_THRESHOLD: f64 = 1.0e-3;
    assert!(
        overall_max_abs < MAX_ABS_THRESHOLD,
        "buttloop INTERNAL recon diverges from jxl-rs decode of the shipped bitstream by \
         max-abs={:.6} (threshold {:.0e}). This is the e8 quality-drift smoking gun: \
         the buttloop converges to a target the user-facing decoder cannot deliver. \
         Per-channel: max R={:.6} G={:.6} B={:.6}, mean R={:.6} G={:.6} B={:.6}. \
         See memory/quality_drift_investigation_2026-05-15.md.",
        overall_max_abs,
        MAX_ABS_THRESHOLD,
        max_abs_per_ch[0],
        max_abs_per_ch[1],
        max_abs_per_ch[2],
        mean_abs_per_ch[0],
        mean_abs_per_ch[1],
        mean_abs_per_ch[2],
    );
}
