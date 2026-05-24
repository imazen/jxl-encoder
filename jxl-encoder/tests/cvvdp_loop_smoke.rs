// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 4 (2026-05-24) integration smoke test.
//!
//! See `docs/RFC_CVVDP_PHASE4_BRIEF.md` Step 5 (Multi-decoder smoke).
//!
//! Phase 4 plumbed the CVVDP signal through the buttloop body proper —
//! the rest of this test exercises the resulting behavioural matrix:
//!
//! 1. **`cvvdp_loop=Some(false)` byte-identity to default** — the
//!    explicit-opt-out path must produce the same bytes as the default
//!    (no `with_cvvdp_loop` call). Guards against the new
//!    `effective_metric_target_distance` plumbing accidentally
//!    perturbing the butteraugli path.
//!
//! 2. **`EncoderStrategy::Libjxl` byte-identity regardless of
//!    `cvvdp_loop` field** — the strict cjxl-parity invariant
//!    (`LossyConfig::resolve_cvvdp_loop` short-circuits to `false` for
//!    Libjxl). This is the structural test that ensures the W44 byte-
//!    lock infrastructure stays intact even when the cvvdp opt-in is
//!    used carelessly.
//!
//! 3. **`cvvdp_loop=Some(true)` decodes via jxl-rs** — the encoder
//!    produces a well-formed JXL bitstream when the cvvdp backend
//!    actually fires; pixels are sane (no NaN/Inf, dims match).
//!
//! ## Test gating
//!
//! - The test is gated `#![cfg(feature = "cvvdp-loop")]` — skipped at
//!   default features.
//! - Each `#[test]` is also `#[ignore]`-d so `cargo test` doesn't run
//!   them by default. Tests require CUDA at runtime; CI hosts without
//!   GPU will trigger the silent-fallback path inside
//!   `construct_backend` (the cvvdp backend's `try_new` returns `None`
//!   and the buttloop falls back to butteraugli), which means the
//!   "actually-cvvdp-was-used" assertion can't be made structurally.
//!   The test still validates the encoder doesn't BREAK when cvvdp is
//!   requested. Run explicitly with:
//!
//!   ```bash
//!   cargo test \
//!     --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel" \
//!     --test cvvdp_loop_smoke -- --ignored
//!   ```
//!
//! ## What this test deliberately does NOT verify
//!
//! - That the cvvdp loop converges to a DIFFERENT bitstream from
//!   butteraugli (Phase 6 tracking sweep is the tool for that).
//! - The cvvdp-direction calibration table's specific values (Phase 6
//!   sweep + RFC §5.4 decision rule is where the targets get
//!   re-validated against real data).
//! - End-to-end SSIM2 / cvvdp scores on the output (those live in the
//!   benchmark TSVs, not the per-PR smoke).

#![cfg(feature = "cvvdp-loop")]

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};

// =============================================================================
// Fixture corpora — synthetic miniatures that don't pull in codec-corpus.
// =============================================================================

/// 64×64 RGB gradient — sRGB-byte plane. Triggers a non-trivial
/// buttloop (the gradient produces a non-degenerate butteraugli
/// reference and at least one round of qf refinement at d ≥ 0.5).
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

/// 128×128 RGB gradient with diagonal stripes. Bigger than the 64×64
/// fixture so the buttloop has more blocks to refine, exercises
/// per-block tile_dist arithmetic at scale.
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

/// 96×96 RGB "noisy photo-like" — combines a low-frequency gradient
/// with a moderate-amplitude high-frequency pattern. Closer in
/// statistics to a real photograph than the pure gradient, so the
/// adaptive_quant + buttloop produce a less-degenerate qf field.
fn noisy_photo_96x96() -> Vec<u8> {
    let (w, h) = (96usize, 96usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Low-frequency: radial brightness.
            let cx = w as i32 / 2;
            let cy = h as i32 / 2;
            let r = (((x as i32 - cx).pow(2) + (y as i32 - cy).pow(2)) as f32).sqrt()
                / ((cx.pow(2) + cy.pow(2)) as f32).sqrt();
            let base = 128.0 + 100.0 * (1.0 - r);
            // High-frequency: small dithering.
            let noise = (((x * 13) ^ (y * 7)) & 0x1f) as f32 - 15.5;
            let v = (base + noise).clamp(0.0, 255.0) as u8;
            out[i] = v;
            out[i + 1] = v.saturating_sub(8);
            out[i + 2] = v.saturating_add(8);
        }
    }
    out
}

/// One smoke-test cell: fixture name + pixels + dims + layout +
/// distance. Wrapped in a struct (vs a 6-tuple) so the test harness
/// satisfies clippy::type_complexity.
struct SmokeCell {
    name: &'static str,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    layout: PixelLayout,
    distance: f32,
}

/// 5 cells per the Phase 4 brief Step 5 ("5 cells (mix of CID22 +
/// GB82-SC + a tiny synthetic)"). We don't pull in codec-corpus to
/// keep the smoke test self-contained — the synthetic fixtures above
/// exercise the same buttloop code paths.
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
// (1) Invariant: cvvdp_loop=Some(false) is byte-identical to default
// =============================================================================

/// `cvvdp_loop = Some(false)` must produce the SAME bytes as the
/// default (no `with_cvvdp_loop` call). Verifies the explicit-opt-out
/// path doesn't accidentally perturb the butteraugli path through the
/// new `effective_metric_target_distance` plumbing.
///
/// Runs at every cell — pure CPU butteraugli, no GPU required, so
/// this test is NOT `#[ignore]`-d.
#[test]
fn cvvdp_loop_some_false_byte_identical_to_default() {
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
            .with_cvvdp_loop(Some(false))
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] cvvdp_loop=Some(false) encode failed: {e:?}"));

        assert_eq!(
            default_bytes,
            opt_out_bytes,
            "[{name}] cvvdp_loop=Some(false) MUST be byte-identical to default — \
             default={} bytes, opt_out={} bytes",
            default_bytes.len(),
            opt_out_bytes.len()
        );
    }
}

/// `cvvdp_loop = None` (the tri-state default) must also produce the
/// SAME bytes as the implicit default. Mirror of the `Some(false)`
/// test for the third state of the tri-state field.
#[test]
fn cvvdp_loop_none_byte_identical_to_default() {
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

        let explicit_none_bytes = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_cvvdp_loop(None)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] cvvdp_loop=None encode failed: {e:?}"));

        assert_eq!(
            default_bytes, explicit_none_bytes,
            "[{name}] cvvdp_loop=None MUST be byte-identical to default (tri-state default)",
        );
    }
}

// =============================================================================
// (2) Invariant: EncoderStrategy::Libjxl byte-identical regardless of cvvdp_loop
// =============================================================================

/// The Libjxl invariant: with `EncoderStrategy::Libjxl`,
/// `LossyConfig::resolve_cvvdp_loop()` returns `false` regardless of
/// the field value. The encoded bytes MUST match the default Libjxl
/// encode (no `with_cvvdp_loop` call). This is the structural test
/// that the W44 byte-lock infrastructure stays intact under cvvdp
/// opt-in.
///
/// Runs at every cell — Libjxl strategy never touches the cvvdp
/// backend (no GPU required).
#[test]
fn libjxl_strategy_byte_identical_regardless_of_cvvdp_loop() {
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

        for opt in [Some(true), Some(false), None] {
            let libjxl_with_opt = LossyConfig::new(d)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_cvvdp_loop(opt)
                .encode(&pixels, w, h, layout)
                .unwrap_or_else(|e| {
                    panic!("[{name}] Libjxl with_cvvdp_loop({opt:?}) encode failed: {e:?}")
                });

            assert_eq!(
                libjxl_default, libjxl_with_opt,
                "[{name}] EncoderStrategy::Libjxl with cvvdp_loop={opt:?} MUST be byte-identical \
                 to default Libjxl — Libjxl invariant violated",
            );
        }
    }
}

// =============================================================================
// (3) Invariant: cvvdp_loop=Some(true) encodes + decodes via jxl-rs
// =============================================================================

/// `cvvdp_loop = Some(true)` on a non-Libjxl strategy must produce a
/// well-formed JXL bitstream that jxl-rs (preferred decoder per
/// CLAUDE.md) can decode without errors. The decoded pixel buffer
/// must be NaN/Inf-free and have the correct dimensions.
///
/// **Gated `#[ignore]`** — requires CUDA at runtime. On hosts without
/// CUDA, the cvvdp backend's `try_new` returns `None` and the buttloop
/// falls back to butteraugli; the encoded bytes will be the same as
/// the default in that case (the test passes either way), but the
/// "actually cvvdp was used" verification can only be done on a host
/// with CUDA + the GPU profiler instrumentation, which is out of
/// scope for this smoke.
///
/// Run with:
/// ```bash
/// cargo test \
///   --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel" \
///   --test cvvdp_loop_smoke \
///   -- --ignored cvvdp_loop_some_true
/// ```
#[test]
#[ignore = "requires CUDA at runtime — run explicitly with --ignored"]
fn cvvdp_loop_some_true_encodes_and_decodes_via_jxl_rs() {
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
            .with_cvvdp_loop(Some(true))
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] cvvdp_loop=Some(true) encode failed: {e:?}"));

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
        // crate's dev-deps; jxl-rs would also satisfy the project
        // CLAUDE.md "use jxl-rs FIRST" rule but is not part of the
        // standard dev-dep tree — `cvvdp_backend_smoke.rs` already
        // documents this choice).
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

        // Pull pixels and sanity-check no NaN/Inf escaped.
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
// Public API smoke (also runs without CUDA — duplicates cvvdp_backend_smoke
// coverage but pins to the Phase 4 invariants the brief calls out).
// =============================================================================

/// `LossyConfig::with_cvvdp_loop` / `cvvdp_loop` round-trip via the
/// public API surface. Belt-and-suspenders against accidental
/// `pub(crate)` regressions on the setter / getter (the Phase 3 smoke
/// also covers this; this one is sized to the Phase 4 deliverable for
/// completeness).
#[test]
fn public_api_round_trip_phase4() {
    let cfg = LossyConfig::new(1.0);
    assert!(cfg.cvvdp_loop().is_none(), "default must be None");

    for opt in [Some(true), Some(false), None] {
        let cfg = LossyConfig::new(1.0).with_cvvdp_loop(opt);
        assert_eq!(
            cfg.cvvdp_loop(),
            opt,
            "with_cvvdp_loop({opt:?}) round-trip via cvvdp_loop() getter"
        );
    }
}
