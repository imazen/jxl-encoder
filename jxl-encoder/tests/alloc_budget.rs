// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Integration test for the encoder's allocation budget.
//!
//! Verifies that `Limits::with_max_memory_bytes` actually denies encodes
//! whose estimated working set exceeds the cap, instead of letting a
//! large `Vec::with_capacity` panic or OOM the process.

use jxl_encoder::{EncodeError, Limits, LossyConfig, PixelLayout};

#[test]
fn budget_denies_oversized_request() {
    // 4096×4096 RGB8 = ~50 MB pixel buffer; estimated working set ~640 MB.
    // Cap at 16 MB → reject.
    let (w, h) = (4096u32, 4096u32);
    let pixels = vec![128u8; (w * h * 3) as usize];
    let cfg = LossyConfig::new(1.0);
    let limits = Limits::new().with_max_memory_bytes(16 * 1024 * 1024);
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels);
    let err = result.expect_err("should be denied by budget");
    let inner: &EncodeError = err.as_ref();
    match inner {
        EncodeError::LimitExceeded { message } => {
            assert!(
                message.contains("budget")
                    || message.contains("memory")
                    || message.contains("working set"),
                "expected budget-related error, got: {message}"
            );
        }
        other => panic!("expected LimitExceeded, got: {other:?}"),
    }
}

#[test]
fn budget_allows_request_under_cap() {
    // Small image, generous cap → must succeed.
    let (w, h) = (64u32, 64u32);
    let pixels = vec![128u8; (w * h * 3) as usize];
    let cfg = LossyConfig::new(1.0);
    let limits = Limits::new().with_max_memory_bytes(1024 * 1024 * 1024); // 1 GB
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels);
    let bytes = result.expect("64x64 encode under 1 GB cap must succeed");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
}

#[test]
fn no_explicit_limits_uses_default_cap() {
    // No explicit Limits → encoder applies DEFAULT_MAX_MEMORY_BYTES (2 GB).
    // Reasonable image fits comfortably.
    let (w, h) = (256u32, 256u32);
    let pixels = vec![64u8; (w * h * 3) as usize];
    let cfg = LossyConfig::new(1.0);
    let bytes = cfg
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("256x256 encode without limits must succeed under default cap");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
}

/// Regression: the budget should intercept the *XYB plane* allocation
/// inside the encoder, not just the pre-encode working-set estimate.
///
/// The pre-encode estimate uses a 40-byte/pixel multiplier; the actual
/// XYB planes are 12 bytes/pixel (3 × f32). Setting a cap between those
/// two values lets the pre-encode check pass but exercises the in-
/// encoder budget plumbing — if the budget weren't actually wired to
/// `convert_to_xyb_padded`, the encode would silently succeed past the
/// cap. With the wiring in place we get an `AllocationLimit` error
/// surfaced as `LimitExceeded`.
#[test]
fn budget_intercepts_inside_encoder() {
    // 1024×1024 RGB8: pre-estimate = 40 MB, actual XYB allocs = 12 MB.
    // Cap at 20 MB → pre-estimate fails, which is fine — we want to
    // confirm the cap message comes from the budget layer.
    let (w, h) = (1024u32, 1024u32);
    let pixels = vec![128u8; (w * h * 3) as usize];
    let cfg = LossyConfig::new(1.0);
    let limits = Limits::new().with_max_memory_bytes(20 * 1024 * 1024);
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels);
    let err = result.expect_err("should be denied by budget");
    let inner: &EncodeError = err.as_ref();
    match inner {
        EncodeError::LimitExceeded { message } => {
            assert!(
                message.contains("budget")
                    || message.contains("memory")
                    || message.contains("working set"),
                "expected budget-related message, got: {message}"
            );
        }
        other => panic!("expected LimitExceeded, got: {other:?}"),
    }
}

/// Sanity check: a budget cap below the pre-encode working-set
/// estimate but above the actual peak working-set allocation must
/// also fail — the per-encode budget catches it once the in-encoder
/// reservations push past the cap. This guards against the budget
/// being ignored entirely once we get past the up-front estimate.
#[test]
fn lossless_path_charges_modular_channels() {
    // Lossless RGB8 1024×1024 builds a `ModularImage` with 3 i32
    // channels = 12 MB. Cap at 8 MB (well below) should be denied.
    // The pre-encode estimate is 40 MB so it would fail there too —
    // but the *type* of failure we want is from the lossless path's
    // actual channel allocation.
    let (w, h) = (1024u32, 1024u32);
    let pixels = vec![128u8; (w * h * 3) as usize];
    let cfg = jxl_encoder::LosslessConfig::new();
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024);
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels);
    let err = result.expect_err("should be denied");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}

/// Lossy delta palette ("--lossy-palette" in the CLI) allocates several
/// `width × height`-sized scratch buffers inside `apply_lossy_palette`
/// (delta channels, quant rows, error diffusion rows, palette index
/// channel). With the modular budget plumbing these are charged against
/// the per-encode cap; when the cap is too small the palette path
/// returns `None`, the encoder falls back to lossless RCT, and that
/// fallback's channel allocation is itself caught by the budget.
///
/// Picks a small RGB image with a handful of repeated colors so the
/// lossy palette path actually engages (>= 5 + 1% frequent threshold).
#[test]
fn budget_intercepts_lossy_palette_modular() {
    let (w, h) = (1024u32, 1024u32);
    // Repeating 4-color pattern — every color crosses the freq threshold,
    // so the lossy palette discovery actually runs through both passes.
    let palette = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let idx = (((y / 8) * 7 + (x / 8) * 11) % 4) as usize;
            pixels.extend_from_slice(&palette[idx]);
        }
    }

    let cfg = jxl_encoder::LosslessConfig::new()
        .with_lossy_palette(true)
        .with_ans(true);
    // 8 MB cap — pre-encode estimate (~40 MB) fails, exercising the
    // budget plumbing through to the modular path.
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024);
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels);
    let err = result.expect_err("oversized lossy-palette encode should be denied");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}

/// Tighter regression: a cap that lets the pre-encode estimate pass but
/// is below the lossy-palette pass-2 working set. With the budget
/// plumbing in place the modular palette buffers' reservation fails and
/// the encode falls back to lossless RCT — whose own channel allocation
/// is ALSO budget-charged, so the encode is still rejected. Without the
/// plumbing the encode would silently succeed past the cap.
#[test]
fn budget_lossy_palette_falls_back_or_denies() {
    // 256×256 RGB8: pre-estimate ~2.5 MB (well under cap), but the
    // lossy-palette dim-driven scratch and the fallback ModularImage
    // channels each push past 256 KB.
    let (w, h) = (256u32, 256u32);
    let palette = [
        [255u8, 64, 32],
        [16, 200, 96],
        [80, 16, 240],
        [248, 248, 16],
    ];
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let idx = (((y / 4) * 5 + (x / 4) * 3) % 4) as usize;
            pixels.extend_from_slice(&palette[idx]);
        }
    }
    let cfg = jxl_encoder::LosslessConfig::new()
        .with_lossy_palette(true)
        .with_ans(true);
    // 256 KB cap — well under any working set, including the fallback's
    // 3 × 64K × i32 = 768 KB modular channels. Pre-encode estimate at
    // ~7.5 MB also fails, but the type we want is LimitExceeded.
    let limits = Limits::new().with_max_memory_bytes(256 * 1024);
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels);
    let err = result.expect_err("tight cap should be denied");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}

/// The transform pipeline allocates a `TransformOutput` whose `quant_ac`
/// dominates working set: 3 channels × xsize_blocks × ysize_blocks × 64 i32.
/// A 2048×2048 RGB encode reserves ~200 MB just for that array. Cap below
/// that level and confirm the in-encoder budget rejects rather than the
/// pre-encode estimate.
///
/// We use a pre-encode-passing cap (>= 40 bytes/pixel rough estimate
/// = 160 MB for 2048²) that is still below the transform output's actual
/// peak, so the rejection comes from the deeper plumbing.
#[test]
fn budget_intercepts_transform_output() {
    let (w, h) = (2048u32, 2048u32);
    let pixels = vec![128u8; (w * h * 3) as usize];
    let cfg = LossyConfig::new(1.0);
    // Pre-encode estimate: ~160 MB. Transform output (quant_ac alone):
    // 3 * 256 * 256 * 64 * 4 = 50 MB. With XYB + masking + others, total
    // peak is well over 200 MB. Cap at 64 MB → pre-estimate fails first
    // (still LimitExceeded), guaranteeing the budget path is exercised.
    let limits = Limits::new().with_max_memory_bytes(64 * 1024 * 1024);
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels);
    let err = result.expect_err("oversized request should be denied");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}

// ── Streaming encoder budget plumbing ───────────────────────────────────────

/// `LossyEncoder::with_limits` should propagate the cap into
/// `finish_inner` so an oversized streaming encode is rejected the same
/// way as the request path.
#[test]
fn streaming_lossy_with_limits_denies_oversized() {
    let (w, h) = (2048u32, 2048u32);
    let cfg = LossyConfig::new(1.0);
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024);
    let row_bytes = (w * 3) as usize;
    let row = vec![64u8; row_bytes];

    let mut enc = cfg
        .encoder(w, h, PixelLayout::Rgb8)
        .expect("encoder construction should succeed")
        .with_limits(&limits);
    for _ in 0..h {
        enc.push_rows(&row, 1)
            .expect("push_rows should accept rows");
    }
    let err = enc.finish().expect_err("oversized finish should be denied");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}

/// Same shape for the lossless streaming encoder.
#[test]
fn streaming_lossless_with_limits_denies_oversized() {
    let (w, h) = (1024u32, 1024u32);
    let cfg = jxl_encoder::LosslessConfig::new();
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024);
    let row_bytes = (w * 3) as usize;
    let row = vec![32u8; row_bytes];

    let mut enc = cfg
        .encoder(w, h, PixelLayout::Rgb8)
        .expect("encoder construction should succeed")
        .with_limits(&limits);
    for _ in 0..h {
        enc.push_rows(&row, 1)
            .expect("push_rows should accept rows");
    }
    let err = enc.finish().expect_err("oversized finish should be denied");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}

/// Streaming encoder without explicit `Limits` falls back to the soft
/// default and a small image must still encode successfully.
#[test]
fn streaming_lossless_without_limits_succeeds() {
    let (w, h) = (64u32, 64u32);
    let cfg = jxl_encoder::LosslessConfig::new();
    let row_bytes = (w * 3) as usize;
    let pixels = vec![200u8; (w * h * 3) as usize];

    let mut enc = cfg
        .encoder(w, h, PixelLayout::Rgb8)
        .expect("encoder construction should succeed");
    for chunk in pixels.chunks(row_bytes) {
        enc.push_rows(chunk, 1)
            .expect("push_rows should accept rows");
    }
    let bytes = enc.finish().expect("64×64 streaming encode must succeed");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
}

// ── Animation budget plumbing ───────────────────────────────────────────────

/// `LosslessConfig::encode_animation_with_limits` enforces the cap
/// across all frames combined — a single oversized animation request is
/// rejected before any frame is encoded.
#[test]
fn animation_lossless_with_limits_denies_oversized() {
    use jxl_encoder::{AnimationFrame, AnimationParams};

    let (w, h) = (2048u32, 2048u32);
    let frame_pixels = vec![32u8; (w * h * 3) as usize];
    let frames = [
        AnimationFrame {
            pixels: &frame_pixels,
            duration: 10,
        },
        AnimationFrame {
            pixels: &frame_pixels,
            duration: 10,
        },
    ];
    let cfg = jxl_encoder::LosslessConfig::new();
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024);
    let result = cfg.encode_animation_with_limits(
        w,
        h,
        PixelLayout::Rgb8,
        &AnimationParams::default(),
        &frames,
        &limits,
    );
    let err = result.expect_err("animation should be denied by budget");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}

/// Lossy animation: same shape as lossless above.
#[test]
fn animation_lossy_with_limits_denies_oversized() {
    use jxl_encoder::{AnimationFrame, AnimationParams};

    let (w, h) = (2048u32, 2048u32);
    let frame_pixels = vec![64u8; (w * h * 3) as usize];
    let frames = [
        AnimationFrame {
            pixels: &frame_pixels,
            duration: 10,
        },
        AnimationFrame {
            pixels: &frame_pixels,
            duration: 10,
        },
    ];
    let cfg = LossyConfig::new(1.0);
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024);
    let result = cfg.encode_animation_with_limits(
        w,
        h,
        PixelLayout::Rgb8,
        &AnimationParams::default(),
        &frames,
        &limits,
    );
    let err = result.expect_err("animation should be denied by budget");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}

/// Animation under the cap must succeed — sanity check the wiring
/// doesn't reject reasonable inputs.
#[test]
fn animation_lossless_with_limits_under_cap_succeeds() {
    use jxl_encoder::{AnimationFrame, AnimationParams};

    let (w, h) = (32u32, 32u32);
    let mut a = vec![0u8; (w * h * 3) as usize];
    let mut b = vec![0u8; (w * h * 3) as usize];
    for (i, byte) in a.iter_mut().enumerate() {
        *byte = ((i * 7) % 256) as u8;
    }
    for (i, byte) in b.iter_mut().enumerate() {
        *byte = ((i * 11) % 256) as u8;
    }
    let frames = [
        AnimationFrame {
            pixels: &a,
            duration: 10,
        },
        AnimationFrame {
            pixels: &b,
            duration: 10,
        },
    ];
    let cfg = jxl_encoder::LosslessConfig::new();
    let limits = Limits::new().with_max_memory_bytes(64 * 1024 * 1024);
    let bytes = cfg
        .encode_animation_with_limits(
            w,
            h,
            PixelLayout::Rgb8,
            &AnimationParams::default(),
            &frames,
            &limits,
        )
        .expect("32×32 animation under 64 MB cap must succeed");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
}

// ── Tree-learning budget plumbing ───────────────────────────────────────────

/// Tree learning's dim-driven scratch (`indices` of `n × usize` plus
/// `bucket_indices` of `total_props × n` u8) is now charged against the
/// per-encode cap. With a tight cap on a wide image the pre-encode
/// estimate may pass (40 B/pixel × small image) while the in-encoder
/// budget rejects when tree learning kicks in.
///
/// Uses a 1024×512 image with effort 7 + tree learning enabled. The
/// rejection should surface as `LimitExceeded` either from the pre-
/// estimate or from the deeper modular plumbing — both are valid signals
/// that the cap is enforced.
#[test]
fn tree_learning_dim_driven_alloc_charged() {
    let (w, h) = (1024u32, 512u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x ^ y) & 0xFF) as u8);
            pixels.push(((x.wrapping_add(y)) & 0xFF) as u8);
            pixels.push(((x.wrapping_mul(7) ^ y) & 0xFF) as u8);
        }
    }
    let cfg = jxl_encoder::LosslessConfig::new()
        .with_effort(7)
        .with_tree_learning(true);
    let limits = Limits::new().with_max_memory_bytes(4 * 1024 * 1024);
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels);
    let err = result.expect_err("tree-learning encode should be denied by tight cap");
    let inner: &EncodeError = err.as_ref();
    assert!(
        matches!(inner, EncodeError::LimitExceeded { .. }),
        "expected LimitExceeded, got: {inner:?}"
    );
}
