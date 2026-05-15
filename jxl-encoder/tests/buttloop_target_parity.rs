// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Layer-2 invariant test for the quality-drift investigation
//! (memory/quality_drift_investigation_2026-05-15.md).
//!
//! Companion to the Layer-1 test (`buttloop_recon_parity.rs`, commit
//! `f73765f`). Where Layer-1 proves the buttloop's INTERNAL reconstruction
//! diverges from `jxl-rs`'s decode of the SHIPPED bitstream, this Layer-2
//! invariant measures the user-observable consequence: at effort 8/9, the
//! butteraugli of (encode → decode → compare with original) should be close
//! to the `--distance` parameter the user requested, because libjxl's
//! `--distance` is calibrated so that "distance N" means
//! "max butteraugli ≈ N" in the limit.
//!
//! Spec (from quality_drift_investigation Chunk 2):
//!   - 3 photos × 4 distances = 12 cells
//!   - Encode at e8 via the public `LossyConfig::distance(d).effort(8)`
//!     path (buttloop ON by default at effort >= 8)
//!   - Decode with `jxl-rs` (PRIMARY decoder per CLAUDE.md)
//!   - Linearize jxl-rs's sRGB f32 output, run Rust butteraugli on
//!     (original linear) vs (decoded linear)
//!   - Assert measured butteraugli <= `distance * 1.10`
//!
//! The test FAILS today on cells where the e8 buttloop undershoots the
//! distance target (the "+30% bytes" / "1.30× cjxl bfly" pattern observed in
//! the drift investigation at high d). It passes on cells where the buttloop
//! actually delivers the requested quality (or better). This is the
//! Layer-2 regression target — when Chunk 3's fix lands, all 12 cells should
//! pass and the `#[ignore]` can be removed.
//!
//! Per CLAUDE.md "NEVER relax test expectations": the +10% tolerance is the
//! plan's documented bound, NOT a knob to widen until the test passes. If
//! the test fails, the failure IS the drift, not the test threshold.
//!
//! Per CLAUDE.md "No Synthetic-Only Quality Tests": all 3 source images are
//! real photos / screenshots from `~/work/codec-corpus/` — the same set the
//! drift investigation harness uses (smooth-photo: clic2025/02809272 1024×1024,
//! detailed-photo: cid22/1025469 512×512, screenshot: gb82-sc/graph 796×481).

#![cfg(feature = "butteraugli-loop")]

use std::path::PathBuf;

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;

/// sRGB normalized [0,1] f32 → linear-light. Matches the helper used by
/// `quality_compare.rs` and `buttloop_recon_parity.rs`. Used to linearize
/// jxl-rs's decoded sRGB f32 output before handing to `butteraugli_linear`.
fn srgb_to_linear_val(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Decode a JXL bitstream with jxl-rs (the PRIMARY decoder per CLAUDE.md
/// "Decoder Testing Priority"). Returns interleaved sRGB f32 RGB
/// (the file's signaled color space — our encoder declares
/// `TransferFunction::Srgb`). Caller linearizes with `srgb_to_linear_val`.
///
/// This mirrors the decoder loop in `buttloop_recon_parity.rs` (the Layer-1
/// test from commit f73765f). Pulled in directly here rather than factored
/// into a shared helper to avoid a `tests/common/mod.rs` cycle that would
/// drag the `__internal_recon_hook` feature onto this test.
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

/// One source image to sweep distances against. Same 3-image set as the
/// drift investigation harness (`examples/quality_drift_investigation.rs`).
struct DriftSource {
    label: &'static str,
    rel_path: &'static str,
}

const DRIFT_SOURCES: &[DriftSource] = &[
    DriftSource {
        label: "smooth_photo (clic2025/02809272 1024x1024)",
        rel_path: "clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    },
    DriftSource {
        label: "detailed_photo (cid22/1025469 512x512)",
        rel_path: "CID22/CID22-512/validation/1025469.png",
    },
    DriftSource {
        label: "screenshot (gb82-sc/graph 796x481)",
        rel_path: "gb82-sc/graph.png",
    },
];

/// Distances to sweep (matches the drift investigation's grid).
const DRIFT_DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 4.0];

/// Effort 8 = buttloop ON with 2 iterations (per `EffortProfile::lossy(8, ...)`).
/// Effort 8 specifically is what the drift investigation table characterized;
/// e9 has 4 iterations and would be a separate sweep. Keeping this test to a
/// single effort (8) is intentional — the cell count is already 12, and e8 is
/// the lowest-effort buttloop tier so it's the most-tested in production.
const DRIFT_EFFORT: u8 = 8;

/// Per the spec: "the EXPECTED measured score should be approximately equal
/// to `distance` ... assert measured score <= `distance * 1.10`". The +10%
/// bound is single-sided (under-quality / target-undershoot fails; the loop
/// happily delivering BETTER quality than asked is not flagged here, even
/// though it's also a form of drift — that direction wastes bits but doesn't
/// surprise the user with worse-than-asked perceptual quality).
const TARGET_UPPER_RATIO: f64 = 1.10;

/// Resolve the codec-corpus root. Order:
///   1. `CODEC_CORPUS_DIR` env var (matches the drift example harness).
///   2. `$HOME/work/codec-corpus` (default layout per project CLAUDE.md).
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

/// Encode → decode → measure butteraugli for one (image, distance) cell.
/// Returns the measured Rust-butteraugli score in linear sRGB.
fn measure_one(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    params: &ButteraugliParams,
    distance: f32,
) -> Result<(usize, f64), String> {
    let cfg = LossyConfig::new(distance).with_effort(DRIFT_EFFORT);
    let bitstream = cfg
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))?;

    let (dec_w, dec_h, dec_srgb_f32) = decode_jxl_rs(&bitstream);
    if dec_w != w as usize || dec_h != h as usize {
        return Err(format!(
            "jxl-rs decoded {}x{} but encoded {}x{}",
            dec_w, dec_h, w, h
        ));
    }
    let n_pixels = (w as usize) * (h as usize);

    // jxl-rs returns sRGB f32 (file declares Srgb TF). Linearize to match
    // the original linear-light buffer that goes into butteraugli_linear.
    // Clamp before linearization: jxl-rs may emit slightly-out-of-range
    // floats near saturation, and the sRGB→linear formula is only valid
    // on [0,1].
    let mut dec_pixels = Vec::with_capacity(n_pixels);
    for i in 0..n_pixels {
        let r = srgb_to_linear_val(dec_srgb_f32[i * 3].clamp(0.0, 1.0));
        let g = srgb_to_linear_val(dec_srgb_f32[i * 3 + 1].clamp(0.0, 1.0));
        let b = srgb_to_linear_val(dec_srgb_f32[i * 3 + 2].clamp(0.0, 1.0));
        dec_pixels.push(RGB::new(r, g, b));
    }
    let dec_linear_img = Img::new(dec_pixels, w as usize, h as usize);

    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map_err(|e| format!("butteraugli failed: {e:?}"))?
        .score;

    Ok((bitstream.len(), bfly))
}

/// One row of the printed result table.
struct DriftRow {
    image: &'static str,
    distance: f32,
    target: f32,
    measured: f64,
    ratio: f64,
    bytes: usize,
    exceeds_bound: bool,
}

/// Layer-2 drift invariant: measured butteraugli of the encode→decode
/// roundtrip must be within +10% of the requested `--distance` for all
/// (image, distance) cells at effort 8.
///
/// **CURRENTLY EXPECTED TO FAIL** on the cells where the e8 buttloop
/// undershoots quality at high distance (per the drift investigation's
/// e8 table: detailed photo @ d=4.0 had ratio 1.227, screenshot @ d=4.0
/// had ratio 1.298 vs cjxl). The failure pinpoints which cells the
/// Layer-1-identified divergence translates into a user-visible quality
/// shortfall. When Chunk 3's fix lands, the assertion at the end should
/// pass and the `#[ignore]` can be removed (or repurposed as a regression
/// gate).
///
/// Per CLAUDE.md "NEVER relax test expectations": the +10% tolerance is
/// the documented plan bound, NOT a knob.
#[test]
#[ignore = "intentional - drift Chunk 2 invariant; see quality_drift_investigation"]
fn buttloop_target_butteraugli_within_10pct_of_distance() {
    let corpus = corpus_root().unwrap_or_else(|| {
        panic!(
            "codec-corpus not found. Set CODEC_CORPUS_DIR=/path or stage at \
             $HOME/work/codec-corpus. Synthetic images are forbidden per CLAUDE.md \
             \"No Synthetic-Only Quality Tests\"."
        )
    });

    let params = ButteraugliParams::default();
    let mut rows: Vec<DriftRow> = Vec::with_capacity(DRIFT_SOURCES.len() * DRIFT_DISTANCES.len());

    for src in DRIFT_SOURCES {
        let path = corpus.join(src.rel_path);
        if !path.exists() {
            panic!(
                "drift source missing: {} (under codec-corpus root {}). \
                 The drift investigation's 3-image set (smooth/detailed/screenshot) \
                 is required — substituting other content would change the cells \
                 the drift table characterized.",
                path.display(),
                corpus.display(),
            );
        }
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("failed to open {}: {}", path.display(), e));
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8();
        let rgb_raw: Vec<u8> = rgb.as_raw().clone();

        // Build the original-linear-RGB Img once per image (shared across
        // all 4 distances). butteraugli_linear takes an immutable ref so we
        // can reuse it.
        let orig_linear: Vec<RGB<f32>> = rgb
            .pixels()
            .map(|p| {
                RGB::new(
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                )
            })
            .collect();
        let orig_linear_img = Img::new(orig_linear, w as usize, h as usize);

        for &d in DRIFT_DISTANCES {
            let (bytes, measured) = match measure_one(&rgb_raw, w, h, &orig_linear_img, &params, d)
            {
                Ok(v) => v,
                Err(e) => panic!("{} d={}: measurement failed: {}", src.label, d, e),
            };
            let ratio = measured / d as f64;
            let exceeds = ratio > TARGET_UPPER_RATIO;
            rows.push(DriftRow {
                image: src.label,
                distance: d,
                target: d,
                measured,
                ratio,
                bytes,
                exceeds_bound: exceeds,
            });
        }
    }

    // Print the per-cell table for human inspection. eprintln! lands on
    // stderr so `--nocapture` shows it even on PASS.
    eprintln!(
        "\nLayer-2 buttloop target parity (e{}, decode=jxl-rs, butteraugli=Rust crate):",
        DRIFT_EFFORT
    );
    eprintln!(
        "{:<48} {:>5} {:>7} {:>9} {:>7} {:>9} Bound",
        "Image", "Dist", "Target", "Measured", "Ratio", "Bytes"
    );
    for r in &rows {
        eprintln!(
            "{:<48} {:>5.2} {:>7.2} {:>9.4} {:>7.3} {:>9} {}",
            r.image,
            r.distance,
            r.target,
            r.measured,
            r.ratio,
            r.bytes,
            if r.exceeds_bound { "FAIL" } else { "ok  " },
        );
    }

    let n_failing: usize = rows.iter().filter(|r| r.exceeds_bound).count();
    eprintln!(
        "\nCells exceeding {:.0}% upper bound: {}/{}",
        (TARGET_UPPER_RATIO - 1.0) * 100.0,
        n_failing,
        rows.len(),
    );

    // The actual assertion. ANY cell over the +10% bound is a drift
    // regression target; print the full table first so a CI-style failure
    // is debuggable.
    if n_failing > 0 {
        let summary: String = rows
            .iter()
            .filter(|r| r.exceeds_bound)
            .map(|r| {
                format!(
                    "  {} d={}: measured {:.4} > target * {:.2} = {:.4} (ratio {:.3})",
                    r.image,
                    r.distance,
                    r.measured,
                    TARGET_UPPER_RATIO,
                    r.target as f64 * TARGET_UPPER_RATIO,
                    r.ratio,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "Layer-2 drift invariant FAILED: {} of {} (image, distance) cells produced measured \
             butteraugli > distance * {:.2} at effort 8.\n\
             This is the user-observable consequence of the Layer-1 buttloop-recon-vs-decoder \
             divergence (see commit f73765f and memory/quality_drift_investigation_2026-05-15.md): \
             the buttloop converges on a target the decoder cannot deliver, so the SHIPPED \
             bitstream has worse perceptual quality than the user requested via --distance.\n\
             Failing cells:\n{}",
            n_failing,
            rows.len(),
            TARGET_UPPER_RATIO,
            summary,
        );
    }
}
