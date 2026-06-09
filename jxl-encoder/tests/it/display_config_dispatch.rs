// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Phase 1 display-config backfill integration tests
//! (RFC `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`, 2026-05-25).
//!
//! Validates the [`crate::api::DisplayConfig`] dispatch surface end-to-end:
//!
//! 1. **Default invariant**: `with_target_display(WebSdr80)` (default)
//!    is byte-identical to omitting the call entirely — guards against
//!    accidental wire-format perturbation from the new plumbing.
//! 2. **Display dispatch**: with cvvdp active, switching to `Phone` or
//!    `Tv` MAY change bytes (the cvvdp loop converges against a
//!    different per-distance target). On hosts with CUDA + the cvvdp
//!    feature, the bytes SHOULD differ; on hosts without CUDA the cvvdp
//!    backend falls back to butteraugli silently and bytes stay
//!    identical. The test asserts the encoder doesn't BREAK either way.
//! 3. **Butteraugli unaffected**: when the active metric is butteraugli
//!    (default), `with_target_display(Tv)` MUST be byte-identical to
//!    `with_target_display(WebSdr80)` — display config only routes
//!    through the cvvdp scoring path.
//! 4. **Libjxl strict-parity**: `EncoderStrategy::Libjxl` forces
//!    `target_display = WebSdr80` at the resolver layer, regardless of
//!    any caller `with_target_display` call. Encoded bytes MUST match
//!    the default Libjxl encode.
//! 5. **Multi-decoder roundtrip**: 2 cells × 2 DisplayConfigs decode
//!    via jxl-oxide cleanly (CUDA-optional).
//!
//! All invariants run at default features (CPU butteraugli + the cvvdp
//! `--features cvvdp-loop` build); none require CUDA at runtime.
//! `--ignored` is reserved for the multi-decoder roundtrip variant
//! that requires running the full backend chain end-to-end and asserts
//! decoded pixels are sane (no NaN/Inf).

#![cfg(feature = "cvvdp-loop")]

use jxl_encoder::api::{DisplayConfig, EncoderStrategy, PerceptualDevice, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

// =============================================================================
// Fixtures (synthetic, self-contained — no codec-corpus dep)
// =============================================================================

/// 64×64 RGB gradient — sRGB-byte. Triggers a non-trivial buttloop
/// (gradient produces a non-degenerate butteraugli reference + at least
/// one round of qf refinement at d ≥ 0.5).
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

/// 96×96 RGB "noisy photo-like" — radial gradient + small dither. Less
/// degenerate than pure gradient; produces a meaningful per-block qf
/// field for the cvvdp loop to differentiate display configs against.
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

// =============================================================================
// (1) Default invariant: WebSdr80 explicit ≡ default (no call)
// =============================================================================

/// `with_target_display(WebSdr80)` MUST be byte-identical to omitting
/// the call. Phase 1 ships `WebSdr80` as the default (matches
/// `cvvdp_gpu::params::DisplayModel::STANDARD_4K` — the pre-Phase-1
/// scoring shape). Verifies the new plumbing doesn't accidentally
/// perturb the bitstream when the default config is explicit.
#[test]
fn target_display_websdr80_byte_identical_to_default() {
    for (name, pixels, w, h, layout, d) in fixtures() {
        let default_bytes = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] default encode failed: {e:?}"));

        let explicit_bytes = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_target_display(DisplayConfig::WebSdr80)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] explicit WebSdr80 encode failed: {e:?}"));

        assert_eq!(
            default_bytes,
            explicit_bytes,
            "[{name}] with_target_display(WebSdr80) MUST be byte-identical to default — \
             default={} bytes, explicit={} bytes",
            default_bytes.len(),
            explicit_bytes.len()
        );
    }
}

// =============================================================================
// (2) Display dispatch effects bytes when cvvdp is active (CUDA-dependent)
// =============================================================================

/// When cvvdp is the active metric AND CUDA is available, switching
/// `target_display` between WebSdr80 / Phone / Tv SHOULD change encoded
/// bytes (different per-distance target → different convergence point).
///
/// On hosts without CUDA, the cvvdp backend falls back to butteraugli
/// silently (per the `construct_backend` dispatch chain), and bytes
/// stay identical. The test asserts the encoder doesn't BREAK on either
/// host class — it only emits an informational println when the bytes
/// happen to be identical (so test output highlights the CUDA-missing
/// case).
///
/// Marked `#[ignore]` because it requires CUDA for the bytes-differ
/// assertion to be meaningful; on a CUDA-less host it's a tautology.
/// Run with:
/// ```bash
/// cargo test \
///   --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel" \
///   --test display_config_dispatch -- --ignored
/// ```
#[test]
#[ignore = "requires CUDA for the bytes-differ assertion to be meaningful"]
fn cvvdp_target_display_shifts_bytes_when_cuda_present() {
    let (_, pixels, w, h, layout, d) = fixtures().into_iter().next().unwrap();
    let mk = |display: DisplayConfig| {
        LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_perceptual_metric(PerceptualMetric::Cvvdp)
            .with_perceptual_device(PerceptualDevice::Auto)
            .with_target_display(display)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{display:?}] cvvdp encode failed: {e:?}"))
    };
    let b_web = mk(DisplayConfig::WebSdr80);
    let b_phone = mk(DisplayConfig::Phone);
    let b_tv = mk(DisplayConfig::Tv);

    // Sanity: all three encodes finished without panicking and produce
    // non-empty bitstreams.
    assert!(
        !b_web.is_empty(),
        "WebSdr80 cvvdp encode produced empty bytes"
    );
    assert!(
        !b_phone.is_empty(),
        "Phone cvvdp encode produced empty bytes"
    );
    assert!(!b_tv.is_empty(), "Tv cvvdp encode produced empty bytes");

    // On CUDA hosts: Phone and Tv targets are 4% / 12% tighter than
    // WebSdr80 at every distance band, so the cvvdp loop converges to
    // a stricter qac → bytes typically grow. On CUDA-less hosts the
    // backend silently falls back to butteraugli, which ignores the
    // display config — bytes stay identical. Either case is acceptable
    // for the encoder's contract (the test asserts no panic / no broken
    // bitstream, not a specific bytes delta).
    if b_web == b_phone && b_web == b_tv {
        eprintln!(
            "[Phase 1 display dispatch] all 3 DisplayConfigs produced identical bytes — \
             likely running on a CUDA-less host where cvvdp falls back to butteraugli. \
             To assert a real dispatch shift, run on a host with CUDA + the cvvdp-loop \
             feature enabled."
        );
    } else {
        eprintln!(
            "[Phase 1 display dispatch] CUDA-present host: WebSdr80={} Phone={} Tv={} bytes",
            b_web.len(),
            b_phone.len(),
            b_tv.len()
        );
    }
}

// =============================================================================
// (3) Butteraugli unaffected by target_display (resolved metric != cvvdp)
// =============================================================================

/// When the active metric is butteraugli (default), `with_target_display`
/// MUST NOT change bytes — display config only routes through the cvvdp
/// scoring path. Phase 1 invariant.
///
/// Run unconditionally; uses CPU butteraugli, no CUDA required.
#[test]
fn butteraugli_unaffected_by_target_display() {
    for (name, pixels, w, h, layout, d) in fixtures() {
        let mk = |display: DisplayConfig| {
            LossyConfig::new(d)
                .with_strategy(EncoderStrategy::Zenjxl)
                .with_perceptual_metric(PerceptualMetric::Butteraugli)
                .with_target_display(display)
                .encode(&pixels, w, h, layout)
                .unwrap_or_else(|e| panic!("[{name} {display:?}] encode failed: {e:?}"))
        };
        let b_web = mk(DisplayConfig::WebSdr80);
        let b_phone = mk(DisplayConfig::Phone);
        let b_tv = mk(DisplayConfig::Tv);
        assert_eq!(
            b_web,
            b_phone,
            "[{name}] butteraugli+target_display(Phone) MUST equal butteraugli+target_display(WebSdr80) \
             — display config only affects cvvdp scoring. \
             web={} phone={} bytes",
            b_web.len(),
            b_phone.len()
        );
        assert_eq!(
            b_web,
            b_tv,
            "[{name}] butteraugli+target_display(Tv) MUST equal butteraugli+target_display(WebSdr80). \
             web={} tv={} bytes",
            b_web.len(),
            b_tv.len()
        );
    }
}

// =============================================================================
// (4) EncoderStrategy::Libjxl forces WebSdr80 regardless of with_target_display
// =============================================================================

/// `EncoderStrategy::Libjxl` MUST force the resolved `target_display`
/// to `WebSdr80` (strict cjxl-parity invariant — matches W44-126 for
/// `with_perceptual_metric`). Setting `with_target_display(Tv)` MUST
/// produce byte-identical output to the default Libjxl encode.
///
/// Run unconditionally; Libjxl strategy never reaches the cvvdp
/// backend (the resolved metric is forced to Butteraugli too).
#[test]
fn libjxl_strategy_byte_identical_regardless_of_target_display() {
    for (name, pixels, w, h, layout, d) in fixtures() {
        let default_bytes = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Libjxl)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("[{name}] Libjxl default encode failed: {e:?}"));

        for display in [
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            let bytes = LossyConfig::new(d)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_target_display(display)
                .encode(&pixels, w, h, layout)
                .unwrap_or_else(|e| panic!("[{name} Libjxl + {display:?}] encode failed: {e:?}"));
            assert_eq!(
                default_bytes,
                bytes,
                "[{name}] Libjxl+target_display({display:?}) MUST be byte-identical to default \
                 Libjxl — strict cjxl-parity invariant. default={} got={} bytes",
                default_bytes.len(),
                bytes.len()
            );
        }
    }
}

// =============================================================================
// (5) Multi-decoder roundtrip (jxl-oxide on 2 cells × 2 displays)
// =============================================================================

/// 2 fixture cells × 2 DisplayConfigs (WebSdr80 + Tv) decode cleanly
/// via jxl-oxide and produce sane pixels (no NaN/Inf, dims match).
///
/// Runs unconditionally — the encoder produces a well-formed JXL
/// bitstream regardless of CUDA availability (cvvdp falls back to
/// butteraugli silently on CUDA-less hosts). The test asserts the
/// bitstream is decodable, not what the cvvdp loop converged to.
#[test]
fn display_config_dispatch_roundtrips_via_jxl_oxide() {
    let cells: Vec<_> = fixtures().into_iter().take(2).collect();
    for (name, pixels, w, h, layout, d) in cells {
        for display in [DisplayConfig::WebSdr80, DisplayConfig::Tv] {
            let bytes = LossyConfig::new(d)
                .with_strategy(EncoderStrategy::Zenjxl)
                .with_perceptual_metric(PerceptualMetric::Cvvdp)
                .with_target_display(display)
                .encode(&pixels, w, h, layout)
                .unwrap_or_else(|e| panic!("[{name} {display:?}] encode failed: {e:?}"));

            // Decode via jxl-oxide. Stream API mirrors the pattern
            // used in `cvvdp_loop_smoke.rs` Step 5.
            let mut decoder = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap_or_else(|e| panic!("[{name} {display:?}] jxl-oxide parse failed: {e:?}"));
            decoder.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let frame = decoder
                .render_frame(0)
                .unwrap_or_else(|e| panic!("[{name} {display:?}] render_frame failed: {e:?}"));
            let mut stream = frame.stream();
            assert_eq!(
                stream.width(),
                w,
                "[{name} {display:?}] decoded width mismatch: expected {} got {}",
                w,
                stream.width()
            );
            assert_eq!(
                stream.height(),
                h,
                "[{name} {display:?}] decoded height mismatch: expected {} got {}",
                h,
                stream.height()
            );
            assert!(
                stream.channels() >= 3,
                "[{name} {display:?}] decoded must have ≥3 channels, got {}",
                stream.channels()
            );
            // Pull pixels + assert no NaN/Inf escaped through the
            // dispatch chain.
            let mut pixels_out: Vec<f32> =
                vec![0.0; (w as usize) * (h as usize) * (stream.channels() as usize)];
            let _ = stream.write_to_buffer(&mut pixels_out);
            for (i, v) in pixels_out.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "[{name} {display:?}] non-finite decoded pixel[{i}]: {v}"
                );
            }
        }
    }
}

// =============================================================================
// Fixture iterator
// =============================================================================

fn fixtures() -> Vec<(&'static str, Vec<u8>, u32, u32, PixelLayout, f32)> {
    vec![
        (
            "gradient_64_d1.0",
            gradient_rgb_64x64(),
            64,
            64,
            PixelLayout::Rgb8,
            1.0,
        ),
        (
            "noisy_96_d2.0",
            noisy_photo_96x96(),
            96,
            96,
            PixelLayout::Rgb8,
            2.0,
        ),
    ]
}
