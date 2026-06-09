// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! zensim-fork Phase 4 (2026-05-25) integration smoke test.
//!
//! Mirrors `tests/cvvdp_loop_smoke.rs` but exercises the zensim
//! `PerceptualMetric` opt-in. Phase 4 plumbed the zensim signal
//! through the buttloop body proper (see
//! `docs/RFC_ZENSIM_FORK_PLAN.md` §6); this test exercises the
//! resulting behavioural matrix:
//!
//! 1. **`PerceptualMetric::Butteraugli` byte-identical to default** —
//!    the explicit-opt-out path must produce the same bytes as the
//!    default. Guards against the new
//!    `effective_metric_target_distance` plumbing accidentally
//!    perturbing the butteraugli path.
//!
//! 2. **`EncoderStrategy::Libjxl` byte-identical regardless of
//!    `PerceptualMetric`** — the strict cjxl-parity invariant
//!    (`LossyConfig::resolve_perceptual_metric` forces Butteraugli
//!    for Libjxl). This is the structural test that ensures the W44
//!    byte-lock infrastructure stays intact even when the zensim
//!    opt-in is used carelessly.
//!
//! 3. **`PerceptualMetric::Zensim` decodes via jxl-oxide** — the
//!    encoder produces a well-formed JXL bitstream when the zensim
//!    backend actually fires; pixels are sane (no NaN/Inf, dims
//!    match).
//!
//! ## Test gating
//!
//! - The test is gated `#![cfg(feature = "zensim-loop")]` — skipped at
//!   default features.
//! - The `metric_zensim_encodes_and_decodes` test is `#[ignore]`-d so
//!   `cargo test` doesn't run it by default. zensim CPU works without
//!   a GPU, but the encode passes through the full buttloop and is
//!   slow (~5-30s per 64×64 cell on small fixtures, much longer for
//!   the larger ones).
//!
//! ## What this test deliberately does NOT verify
//!
//! - That the zensim loop converges to a DIFFERENT bitstream from
//!   butteraugli (Phase 6 tracking sweep is the tool for that).
//! - The zensim-direction calibration table's specific values
//!   (Phase 6 sweep + RFC §5.4 decision rule is where the targets get
//!   re-validated against real data).
//! - End-to-end SSIM2 / zensim scores on the output (those live in
//!   the benchmark TSVs, not the per-PR smoke).

#![cfg(feature = "zensim-loop")]

use jxl_encoder::api::{EncoderStrategy, PerceptualDevice, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

// =============================================================================
// Fixture corpora — synthetic miniatures (mirror cvvdp_loop_smoke.rs).
// =============================================================================

fn gradient_rgb_64x64() -> Vec<u8> {
    let (w, h) = (64usize, 64usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 128;
        }
    }
    out
}

fn diagonal_stripes_128x128() -> Vec<u8> {
    let (w, h) = (128usize, 128usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let stripe = ((x + y) / 8) & 1;
            if stripe == 0 {
                out[i] = (x as u8).wrapping_mul(2);
                out[i + 1] = (y as u8).wrapping_mul(2);
                out[i + 2] = 64;
            } else {
                out[i] = 200;
                out[i + 1] = ((x ^ y) as u8).wrapping_mul(3);
                out[i + 2] = 200;
            }
        }
    }
    out
}

fn noisy_photo_96x96() -> Vec<u8> {
    let (w, h) = (96usize, 96usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let cx = w as i32 / 2;
            let cy = h as i32 / 2;
            let r = (((x as i32 - cx).pow(2) + (y as i32 - cy).pow(2)) as f32).sqrt()
                / ((cx.pow(2) + cy.pow(2)) as f32).sqrt();
            let base = 128.0 + 100.0 * (1.0 - r);
            let noise = (((x * 13) ^ (y * 7)) & 0x1f) as f32 - 15.5;
            let v = (base + noise).clamp(0.0, 255.0) as u8;
            out[i] = v;
            out[i + 1] = v.saturating_sub(8);
            out[i + 2] = v.saturating_add(8);
        }
    }
    out
}

/// One smoke-test cell: fixture name + pixels + dims + layout + distance.
struct SmokeCell {
    name: &'static str,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    layout: PixelLayout,
    distance: f32,
}

/// 5 cells per the Phase 4 brief Step 5.
fn smoke_cells() -> Vec<SmokeCell> {
    vec![
        SmokeCell {
            name: "gradient_64_d1.0",
            pixels: gradient_rgb_64x64(),
            width: 64,
            height: 64,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
        },
        SmokeCell {
            name: "gradient_64_d2.0",
            pixels: gradient_rgb_64x64(),
            width: 64,
            height: 64,
            layout: PixelLayout::Rgb8,
            distance: 2.0,
        },
        SmokeCell {
            name: "diagonal_128_d1.5",
            pixels: diagonal_stripes_128x128(),
            width: 128,
            height: 128,
            layout: PixelLayout::Rgb8,
            distance: 1.5,
        },
        SmokeCell {
            name: "noisy_96_d1.0",
            pixels: noisy_photo_96x96(),
            width: 96,
            height: 96,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
        },
        SmokeCell {
            name: "noisy_96_d3.0",
            pixels: noisy_photo_96x96(),
            width: 96,
            height: 96,
            layout: PixelLayout::Rgb8,
            distance: 3.0,
        },
    ]
}

// =============================================================================
// (1) Invariant: PerceptualMetric::Butteraugli byte-identical to default
// =============================================================================

/// `PerceptualMetric::Butteraugli` must produce the SAME bytes as the
/// default (no `with_perceptual_metric` call). Verifies the
/// explicit-opt-out path doesn't accidentally perturb the butteraugli
/// path through the new `effective_metric_target_distance` plumbing.
#[test]
fn metric_butteraugli_byte_identical_to_default() {
    for cell in smoke_cells() {
        let SmokeCell {
            name,
            pixels,
            width: w,
            height: h,
            layout,
            distance: d,
        } = cell;
        let default_bytes = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] default encode failed: {e:?}"));

        let opt_out_bytes = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_perceptual_metric(PerceptualMetric::Butteraugli)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] explicit Butteraugli encode failed: {e:?}"));

        assert_eq!(
            default_bytes,
            opt_out_bytes,
            "[{name}] PerceptualMetric::Butteraugli MUST be byte-identical to default — \
             default={} bytes, opt_out={} bytes",
            default_bytes.len(),
            opt_out_bytes.len()
        );
    }
}

/// The implicit default (no `with_perceptual_metric` call) round-trips
/// through `with_perceptual_metric(PerceptualMetric::Butteraugli)` —
/// belt-and-suspenders against accidental default-state drift.
#[test]
fn metric_default_byte_identical_to_butteraugli_auto() {
    for cell in smoke_cells() {
        let SmokeCell {
            name,
            pixels,
            width: w,
            height: h,
            layout,
            distance: d,
        } = cell;
        let default_bytes = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] default encode failed: {e:?}"));

        let explicit_default = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_perceptual_metric(PerceptualMetric::Butteraugli)
            .with_perceptual_device(PerceptualDevice::Auto)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] explicit Butteraugli+Auto encode failed: {e:?}"));

        assert_eq!(
            default_bytes, explicit_default,
            "[{name}] explicit Butteraugli+Auto MUST be byte-identical to default",
        );
    }
}

// =============================================================================
// (2) Invariant: EncoderStrategy::Libjxl byte-identical regardless of metric
// =============================================================================

/// The Libjxl invariant: with `EncoderStrategy::Libjxl`,
/// `LossyConfig::resolve_perceptual_metric()` forces Butteraugli
/// regardless of the field value. The encoded bytes MUST match the
/// default Libjxl encode (no `with_perceptual_metric` call).
#[test]
fn libjxl_strategy_byte_identical_regardless_of_zensim_metric() {
    for cell in smoke_cells() {
        let SmokeCell {
            name,
            pixels,
            width: w,
            height: h,
            layout,
            distance: d,
        } = cell;
        let libjxl_default = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Libjxl)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] Libjxl default encode failed: {e:?}"));

        // Exercise every metric × device combination on Libjxl. The
        // strict cjxl-parity invariant (`resolve_perceptual_metric`
        // short-circuits to Butteraugli) MUST keep the bytes
        // identical to the default Libjxl encode regardless of the
        // explicit caller selection.
        for metric in [PerceptualMetric::Butteraugli, PerceptualMetric::Zensim] {
            for device in [
                PerceptualDevice::Auto,
                PerceptualDevice::Cpu,
                PerceptualDevice::Gpu,
            ] {
                let libjxl_with_opt = LossyConfig::new(d)
                    .with_strategy(EncoderStrategy::Libjxl)
                    .with_perceptual_metric(metric)
                    .with_perceptual_device(device)
                    .encode(&pixels, w, h, layout)
                    .unwrap_or_else(|e| {
                        panic!("[{name}] Libjxl + {metric:?}/{device:?} encode failed: {e:?}")
                    });

                assert_eq!(
                    libjxl_default, libjxl_with_opt,
                    "[{name}] EncoderStrategy::Libjxl with metric={metric:?} \
                     device={device:?} MUST be byte-identical to default Libjxl \
                     — strict cjxl-parity invariant violated",
                );
            }
        }
    }
}

// =============================================================================
// (3) Invariant: PerceptualMetric::Zensim encodes + decodes via jxl-oxide
// =============================================================================

/// `PerceptualMetric::Zensim` on a non-Libjxl strategy must produce a
/// well-formed JXL bitstream that jxl-oxide can decode without errors.
/// The decoded pixel buffer must be NaN/Inf-free and have the correct
/// dimensions.
///
/// **Gated `#[ignore]`** — the zensim CPU backend works without a GPU,
/// but the encode passes through the full buttloop which is slow
/// (~5-30s per cell at effort 8). On the smoke cells above (small
/// synthetic fixtures) it terminates in reasonable time; we still
/// gate to keep `cargo test` fast at default settings. Run
/// explicitly with:
///
/// ```bash
/// cargo test \
///   --features "__expert butteraugli-loop zensim-loop ssim2-loop parallel" \
///   --test zensim_loop_smoke \
///   -- --ignored metric_zensim_encodes_and_decodes
/// ```
#[test]
#[ignore = "slow (CPU zensim buttloop) — run explicitly with --ignored"]
fn metric_zensim_encodes_and_decodes() {
    for cell in smoke_cells() {
        let SmokeCell {
            name,
            pixels,
            width: w,
            height: h,
            layout,
            distance: d,
        } = cell;
        let encoded = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_perceptual_metric(PerceptualMetric::Zensim)
            // Force CPU so the test doesn't need CUDA.
            .with_perceptual_device(PerceptualDevice::Cpu)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] Zensim metric encode failed: {e:?}"));

        assert!(
            !encoded.is_empty(),
            "[{name}] encoded bitstream must be non-empty"
        );
        assert!(
            encoded.len() >= 100,
            "[{name}] encoded bitstream suspiciously small: {} bytes",
            encoded.len()
        );

        // Decode via jxl-oxide.
        let mut decoder = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .unwrap_or_else(|e| panic!("[{name}] jxl-oxide parse failed: {e:?}"));
        decoder.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let frame = decoder
            .render_frame(0)
            .unwrap_or_else(|e| panic!("[{name}] jxl-oxide render_frame failed: {e:?}"));
        let stream = frame.stream();

        assert_eq!(
            stream.width(),
            w,
            "[{name}] decoded width mismatch: expected {} got {}",
            w,
            stream.width()
        );
        assert_eq!(
            stream.height(),
            h,
            "[{name}] decoded height mismatch: expected {} got {}",
            h,
            stream.height()
        );
        assert!(
            stream.channels() >= 3,
            "[{name}] decoded image must have ≥3 channels, got {}",
            stream.channels()
        );

        let mut pixels_out: Vec<f32> =
            vec![0.0; (w as usize) * (h as usize) * (stream.channels() as usize)];
        let mut stream = stream;
        let _ = stream.write_to_buffer(&mut pixels_out);
        for (i, v) in pixels_out.iter().enumerate() {
            assert!(v.is_finite(), "[{name}] decoded pixel[{i}] non-finite: {v}");
        }
    }
}

// =============================================================================
// Public API smoke (also runs without GPU — duplicates zensim_backend_smoke
// coverage but pins to the Phase 4 invariants the brief calls out).
// =============================================================================

/// Multi-metric Phase 0 (RFC #3): `LossyConfig::with_perceptual_metric`
/// / `perceptual_metric` round-trip via the public API surface for
/// the new `Zensim` variant. Belt-and-suspenders against accidental
/// `pub(crate)` regressions on the setter / getter.
#[test]
fn public_api_round_trip_zensim_phase4() {
    let cfg = LossyConfig::new(1.0);
    assert_eq!(
        cfg.perceptual_metric(),
        PerceptualMetric::Butteraugli,
        "default must be Butteraugli"
    );

    for m in [
        PerceptualMetric::Butteraugli,
        PerceptualMetric::Cvvdp,
        PerceptualMetric::Zensim,
    ] {
        let cfg = LossyConfig::new(1.0).with_perceptual_metric(m);
        assert_eq!(
            cfg.perceptual_metric(),
            m,
            "with_perceptual_metric({m:?}) round-trip via perceptual_metric() getter"
        );
    }
}
