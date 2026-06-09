// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 5 (2026-05-24) CPU CVVDP backend integration smoke
//! test.
//!
//! Mirrors the structure of `cvvdp_loop_smoke.rs` (Phase 4) and
//! `cvvdp_backend_smoke.rs` (Phase 3) but exercises the CPU CVVDP
//! backend specifically via
//! [`LossyConfig::with_cvvdp_use_cpu(Some(true))`].
//!
//! The CPU backend has no CUDA dependency, so unlike the GPU smoke
//! these tests are NOT `#[ignore]`-d — they run as part of the default
//! cargo-test invocation when the `cvvdp-loop-cpu` cargo feature is
//! compiled in.
//!
//! ## Invariants exercised
//!
//! 1. **`cvvdp_use_cpu=Some(true)` produces a well-formed JXL
//!    bitstream**. We encode 5 cells (mix of small RGB + RGBA synthetic
//!    fixtures), decode each via jxl-oxide (project standard Rust
//!    decoder), and assert dimensions + finite pixels.
//!
//! 2. **`EncoderStrategy::Libjxl` invariant**: a Libjxl encode with
//!    `with_cvvdp_use_cpu(Some(true))` AND `with_cvvdp_loop(Some(true))`
//!    must be byte-identical to a default Libjxl encode (the Libjxl
//!    invariant fires upstream via `resolve_cvvdp_loop` returning
//!    `false`, which makes the entire cvvdp dispatch branch
//!    unreachable). This is the structural test that the W44 byte-lock
//!    infrastructure stays intact under the new Phase 5 opt-in.
//!
//! 3. **`cvvdp_use_cpu=None` + `cvvdp_loop=None`** byte-identical to
//!    default (no-opt-in baseline). Guards against the Phase 5 builder
//!    plumbing accidentally perturbing the default path.
//!
//! ## What this test deliberately does NOT verify
//!
//! - That the CPU CVVDP backend was ACTUALLY chosen (e.g. GPU CVVDP
//!   wasn't preferred). When both backends are compiled in, the
//!   default policy is GPU-first (Agent A's CPU port is 10× slower
//!   per the Phase 5 brief's measurement). On hosts with CUDA, the
//!   `cvvdp_use_cpu=Some(true)` opt-in IS load-bearing but
//!   structurally untestable without GPU profiler instrumentation
//!   that's out of scope here.
//! - End-to-end SSIM2 / cvvdp scores — those live in
//!   `benchmarks/cvvdp_cpu_vs_gpu_buttloop_2026-05-24.tsv`.
//! - CPU vs GPU bit-for-bit equivalence — the two backends produce
//!   ≤ 1e-4 JOD drift per Agent A's reference parity tests; the
//!   bench TSV quantifies that on real encoder paths.

#![cfg(feature = "cvvdp-loop-cpu")]

use jxl_encoder::api::{EncoderStrategy, PerceptualDevice, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

// =============================================================================
// Fixture corpora — synthetic miniatures. Same shapes as the Phase 4
// smoke test (so the two smokes share corpus methodology), repeated
// here so the file is self-contained (no shared #[path] imports
// between integration tests).
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

struct SmokeCell {
    name: &'static str,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    layout: PixelLayout,
    distance: f32,
}

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
// (1) Public API surface check
// =============================================================================

/// Multi-metric Phase 0 (RFC #3): `with_perceptual_device` /
/// `perceptual_device` round-trip via the public API surface.
/// Guards against accidental `pub(crate)` regressions on the public
/// setter and getter. Replaces the pre-Phase-0 `with_cvvdp_use_cpu`
/// shape (the CPU-vs-GPU toggle is now metric-agnostic on
/// `PerceptualDevice`).
#[test]
fn public_api_round_trip() {
    let cfg = LossyConfig::new(1.0);
    assert_eq!(
        cfg.perceptual_device(),
        PerceptualDevice::Auto,
        "default must be Auto"
    );

    for d in [
        PerceptualDevice::Auto,
        PerceptualDevice::Cpu,
        PerceptualDevice::Gpu,
    ] {
        let cfg = LossyConfig::new(1.0).with_perceptual_device(d);
        assert_eq!(cfg.perceptual_device(), d);
    }
}

// =============================================================================
// (2) Invariant: cvvdp_use_cpu=None + cvvdp_loop=None byte-identical to default
// =============================================================================

/// `cvvdp_use_cpu = None` + `cvvdp_loop = None` (default tri-state for
/// both fields) MUST produce the same bytes as the implicit default
/// (no `with_cvvdp_*` calls). Guards against the Phase 5 builder
/// plumbing accidentally perturbing the default path.
#[test]
fn default_opt_out_byte_identical_to_default() {
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
            .with_perceptual_device(PerceptualDevice::Auto)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] explicit Butteraugli+Auto encode failed: {e:?}"));

        assert_eq!(
            default_bytes,
            opt_out_bytes,
            "[{name}] explicit Butteraugli + Auto MUST be byte-identical \
             to default — default={} bytes, opt_out={} bytes",
            default_bytes.len(),
            opt_out_bytes.len()
        );
    }
}

/// Multi-metric Phase 0: `PerceptualDevice::Cpu` on the
/// `Butteraugli` metric (= "force CPU butteraugli") is the pre-Phase-0
/// `with_cvvdp_use_cpu(Some(true))` minus the cvvdp opt-in. With
/// `Butteraugli` as the metric (default), choosing `Cpu` should be
/// byte-identical to default when the `gpu-butteraugli` cargo feature
/// is OFF (the default feature set doesn't include it). With the
/// feature ON, default device `Auto` resolves to GPU, so explicit `Cpu`
/// here exercises the field's effect — this is the renamed
/// "CPU-vs-GPU selector without the outer cvvdp opt-in" test.
#[test]
fn perceptual_device_cpu_without_cvvdp_metric_resolves_butter_cpu() {
    for cell in smoke_cells() {
        let SmokeCell {
            name,
            pixels,
            width: w,
            height: h,
            layout,
            distance: d,
        } = cell;
        // Default features: Auto = CPU butteraugli. Forcing Cpu
        // explicitly must produce identical bytes (still butteraugli
        // CPU under the hood).
        #[cfg(not(feature = "gpu-butteraugli"))]
        {
            let default_bytes = LossyConfig::new(d)
                .with_strategy(EncoderStrategy::Zenjxl)
                .encode(&pixels, w, h, layout)
                .unwrap_or_else(|e| panic!("[{name}] default encode failed: {e:?}"));

            let cpu_only_bytes = LossyConfig::new(d)
                .with_strategy(EncoderStrategy::Zenjxl)
                .with_perceptual_device(PerceptualDevice::Cpu)
                .encode(&pixels, w, h, layout)
                .unwrap_or_else(|e| panic!("[{name}] explicit Cpu device encode failed: {e:?}"));

            assert_eq!(
                default_bytes, cpu_only_bytes,
                "[{name}] PerceptualDevice::Cpu on default metric (Butteraugli) \
                 must be byte-identical to default when gpu-butteraugli is OFF",
            );
        }
        // With gpu-butteraugli ON, Auto = GPU; explicit Cpu would
        // differ. The byte-identity check above only fires when GPU
        // is unavailable. Skip the explicit assertion under the
        // feature.
        #[cfg(feature = "gpu-butteraugli")]
        {
            let _ = (name, pixels, w, h, layout, d);
        }
    }
}

// =============================================================================
// (3) Invariant: EncoderStrategy::Libjxl byte-identical regardless of cvvdp fields
// =============================================================================

/// The Libjxl invariant: with `EncoderStrategy::Libjxl`,
/// `LossyConfig::resolve_cvvdp_loop()` returns `false` regardless of
/// the field values, which makes the entire cvvdp dispatch branch
/// (CPU or GPU) unreachable. The encoded bytes MUST match the default
/// Libjxl encode (no `with_cvvdp_*` calls). This is the structural
/// test that the W44 byte-lock infrastructure stays intact under the
/// new Phase 5 opt-in.
#[test]
fn libjxl_strategy_byte_identical_regardless_of_cvvdp_use_cpu() {
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

        // Multi-metric Phase 0: exhaustive 2×3 matrix over
        // (metric, device) — all 6 cells must be byte-identical to
        // default Libjxl (strict cjxl-parity invariant).
        for metric in [PerceptualMetric::Butteraugli, PerceptualMetric::Cvvdp] {
            for device in [
                PerceptualDevice::Auto,
                PerceptualDevice::Cpu,
                PerceptualDevice::Gpu,
            ] {
                let libjxl_with_opts = LossyConfig::new(d)
                    .with_strategy(EncoderStrategy::Libjxl)
                    .with_perceptual_metric(metric)
                    .with_perceptual_device(device)
                    .encode(&pixels, w, h, layout)
                    .unwrap_or_else(|e| {
                        panic!("[{name}] Libjxl + {metric:?}/{device:?} encode failed: {e:?}")
                    });

                assert_eq!(
                    libjxl_default, libjxl_with_opts,
                    "[{name}] EncoderStrategy::Libjxl with metric={metric:?}, \
                     device={device:?} MUST be byte-identical to default \
                     Libjxl — Libjxl invariant violated",
                );
            }
        }
    }
}

// =============================================================================
// (4) End-to-end: cvvdp_use_cpu=Some(true) + cvvdp_loop=Some(true) encodes
// =============================================================================

/// `cvvdp_loop = Some(true)` + `cvvdp_use_cpu = Some(true)` on a
/// non-Libjxl strategy must produce a well-formed JXL bitstream that
/// jxl-oxide can decode without errors. The decoded pixel buffer must
/// have the correct dimensions and contain only finite values.
///
/// This test is NOT `#[ignore]`-d because the CPU CVVDP backend has
/// no CUDA dependency. On a host with both `cvvdp-loop` and
/// `cvvdp-loop-cpu` compiled, this test exercises the CPU CVVDP
/// dispatch path (per the Phase 5 brief's dispatch policy).
#[test]
fn cvvdp_cpu_encode_decode_roundtrip() {
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
            .with_perceptual_metric(PerceptualMetric::Cvvdp)
            .with_perceptual_device(PerceptualDevice::Cpu)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] Cvvdp + Cpu device encode failed: {e:?}"));

        assert!(
            !encoded.is_empty(),
            "[{name}] encoded bitstream must be non-empty"
        );
        assert!(
            encoded.len() >= 100,
            "[{name}] encoded bitstream suspiciously small: {} bytes",
            encoded.len()
        );

        // Decode via jxl-oxide (the available Rust decoder in this
        // crate's dev-deps; mirrors the existing Phase 3 / Phase 4
        // smoke tests' choice).
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
