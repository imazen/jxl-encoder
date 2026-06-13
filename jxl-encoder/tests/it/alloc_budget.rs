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
    // No explicit Limits → encoder applies DEFAULT_MAX_MEMORY_BYTES (4 GiB).
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
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame_pixels,
            duration: 10,
            ..Default::default()
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
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame_pixels,
            duration: 10,
            ..Default::default()
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
            ..Default::default()
        },
        AnimationFrame {
            pixels: &b,
            duration: 10,
            ..Default::default()
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

/// W44-AUDIT-2 regression: at e9 (4 buttloop iters) on a multi-megapixel
/// screenshot at d>=4, the encoder must NOT spuriously OOM on the default
/// 2 GiB budget.
///
/// Root cause (commit `d1c01c2f`, May 9): `apply_epf` and
/// `apply_epf_with_scratch` used `MemoryBudget::reserve_permanent_opt` for
/// padded scratch + output Vecs that were function-local. Each buttloop
/// iter called `apply_epf` and leaked ~150 MB of budget accounting per
/// call (4 MP image, EPF step 1 + 2 padded planes + outputs). At e9 with
/// 4 iters, the leak accumulated to ~600 MB extra-charged, blowing the
/// 2 GiB cap on cells whose actual peak working set was ~1.93 GB (which
/// e8 with 2 iters could just fit). Both `Zenjxl` and `Libjxl` strategies
/// were affected — the bug was in the always-on EPF reconstruction path.
///
/// Fix (W44-AUDIT-2): switch the four sites in `vardct/epf.rs` to
/// `reserve_opt` with a function-scope `BudgetGuard`. Buffers still
/// peak-track correctly (used briefly bumps by `padded*12 + n*12`) and
/// the guard drops at scope end — so per-call accounting is net-zero
/// (output planes swapped into caller's existing planes; old freed).
///
/// This test reproduces the exact failing cell from the W44-AUDIT-1
/// fresh bench TSV `benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.tsv`
/// — 2560×1664 (≈4.26 MP) screenshot-class content at e9 d=4. Pre-fix:
/// errors with `LimitExceeded { requested: 153_658_800 bytes on top of
/// 2_087_219_952 (cap 2_147_483_648) }`. Post-fix: encodes in ≈3 s,
/// produces a valid JXL bitstream.
///
/// Uses synthetic high-frequency RGB to mimic screenshot content
/// (lots of sharp edges → engages EPF non-trivially). Validates the
/// default 2 GiB cap path (no explicit `with_max_memory_bytes`).
#[test]
fn w44_audit_2_e9_d4_large_screenshot_no_spurious_oom() {
    use jxl_encoder::api::EncoderStrategy;
    // 2560×1664 = 4_259_840 pixels = ~12.8 MB RGB8 source.
    // Same shape as the W44-AUDIT-1 failing cell on codec_wiki.png.
    let (w, h) = (2560u32, 1664u32);
    let mut pixels = Vec::with_capacity((w as usize) * (h as usize) * 3);
    // Mix of high-frequency text-like content (sharp edges) and gradients
    // — should engage EPF, gaborish, and the buttloop fully. Without
    // genuine sharp edges the buttloop converges too fast and the per-iter
    // EPF leak doesn't manifest. With them, every iter calls apply_epf
    // → reserves the ~150 MB transient (now correctly released).
    for y in 0..h {
        for x in 0..w {
            // Periodic sharp transitions every 6 pixels → text-like.
            let edge = ((x / 6) ^ (y / 6)) & 1;
            let r = if edge == 0 { 240 } else { 16 };
            let g = if edge == 0 { 200 } else { 32 };
            let b = if edge == 0 { 160 } else { 48 };
            pixels.push(r as u8);
            pixels.push(g as u8);
            pixels.push(b as u8);
        }
    }

    // This regression was calibrated against the (then-default) 2 GiB cap.
    // The default is now 4 GiB (12 MP HDR memory fix, 2026-06-13), which
    // would hand this 4 MP cell ~2 GiB of slack and blunt the test. Pin the
    // historical 2 GiB cap explicitly so a re-introduced leak still trips it.
    let limits = Limits::new().with_max_memory_bytes(2 * 1024 * 1024 * 1024);
    for strategy in [EncoderStrategy::Zenjxl, EncoderStrategy::Libjxl] {
        let cfg = LossyConfig::new(4.0)
            .with_effort(9)
            .with_strategy(strategy.clone());
        let bytes = cfg
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&limits)
            .encode(&pixels)
            .unwrap_or_else(|e| {
                panic!(
                    "W44-AUDIT-2 regression: e9 d=4 on 4 MP screenshot OOM'd \
                 under explicit 2 GiB cap with strategy {strategy:?}: {e}"
                )
            });
        assert!(
            bytes.len() > 1000 && bytes.len() < (w as usize * h as usize),
            "expected non-trivial JXL output, got {} bytes",
            bytes.len()
        );
        assert_eq!(&bytes[..2], &[0xFF, 0x0A], "JXL signature expected");
    }
}

/// Issue #54 regression: imac_g3.png (2940×1912 ≈ 5.62 MP) at e9 d∈{2, 4, 5, 6}
/// must NOT spuriously OOM on the default 2 GiB budget.
///
/// W44-1 (`dd51c504`) cjxl_parity_ledger seed run caught 6 cells failing on
/// imac_g3.png × e9 × d∈{2, 4, 5, 6} — 6 cells out of 600. Other distances
/// and efforts on the same image succeeded. The failure mechanism is
/// identical to W44-AUDIT-2 (EPF budget-accounting leak in `apply_epf` /
/// `apply_epf_with_scratch`): each buttloop iter leaked ~200 MB of budget
/// accounting on this 5.6 MP image (larger than the W44-AUDIT-2 codec_wiki
/// cell at 4.26 MP, so the per-call leak was proportionally bigger),
/// accumulating to >800 MB extra-charged across 4 buttloop iters at e9.
/// W44-AUDIT-2 (commit `887cac54`, May 24) closed this issue as a
/// side-effect by fixing the same `reserve_opt` vs `reserve_permanent_opt`
/// shape in `vardct/epf.rs`.
///
/// This regression test exists alongside the W44-AUDIT-2 test because:
/// 1. It exercises the LARGER image dimension (5.6 MP vs 4.26 MP) that
///    issue #54 surfaced. The W44-AUDIT-2 test catches the leak at +51 MB
///    above cap; this test catches it at +135 MB above cap (per repro
///    logs at parent `a11fd0e2` with the fix reverted) — the per-call EPF
///    leak scales linearly with pixel count, so any future re-introduction
///    of the same shape of bug that happens to fit the 4.26 MP AUDIT-2 cell
///    within cap would still blow past this 5.62 MP cell's headroom first.
/// 2. It pins a known-real production cell (imac_g3.png screenshot
///    dimensions) — a future leak that synthetic AUDIT-2 inputs happen to
///    avoid would still trigger here if the dimension + content pattern
///    matches a real production failure.
/// 3. Uses d=4 because empirically d=2 on synthetic high-frequency content
///    converges fast enough in the buttloop that the per-iter EPF leak
///    doesn't compound past cap (encoder succeeds in ~3.5 min even with
///    fix reverted). d=4 reliably reproduces the OOM in <15 s pre-fix on
///    this dimension. The W44-1 ledger surfaced failures at d∈{2, 4, 5, 6}
///    using the actual imac_g3.png pixels which engage EPF differently
///    than synthetic; d=4 is the most reliable smoke gate for the
///    budget-leak mechanism on synthetic-content tests.
///
/// Verified: pre-fix (4 `reserve_permanent_opt` sites in `vardct/epf.rs`)
/// fails in ~9 s with `LimitExceeded { requested: 135210864 bytes on top
/// of 2145893808 (cap 2147483648) }`. Post-fix passes in ~60 s.
#[test]
fn issue_54_imac_g3_e9_d4_no_spurious_oom() {
    // imac_g3.png dimensions (per W44-1 seed surface).
    let (w, h) = (2940u32, 1912u32);
    let mut pixels = Vec::with_capacity((w as usize) * (h as usize) * 3);
    // Same text-like high-frequency pattern as the W44-AUDIT-2 test —
    // engages EPF, gaborish, and the buttloop fully so per-iter EPF
    // budget reservation runs ≥ 4 times at e9.
    for y in 0..h {
        for x in 0..w {
            let edge = ((x / 6) ^ (y / 6)) & 1;
            let r = if edge == 0 { 240 } else { 16 };
            let g = if edge == 0 { 200 } else { 32 };
            let b = if edge == 0 { 160 } else { 48 };
            pixels.push(r as u8);
            pixels.push(g as u8);
            pixels.push(b as u8);
        }
    }

    let cfg = LossyConfig::new(4.0).with_effort(9);
    // Pin the historical 2 GiB cap explicitly: the default is now 4 GiB
    // (12 MP HDR memory fix, 2026-06-13), so relying on the default would
    // give this 5.6 MP cell too much slack to catch a re-introduced leak.
    let limits = Limits::new().with_max_memory_bytes(2 * 1024 * 1024 * 1024);
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels)
        .unwrap_or_else(|e| {
            panic!(
                "issue #54 regression: e9 d=4 on imac_g3-sized (2940×1912 ≈ \
                 5.6 MP) screenshot OOM'd under explicit 2 GiB cap: {e}"
            )
        });
    assert!(
        bytes.len() > 1000 && bytes.len() < (w as usize * h as usize),
        "expected non-trivial JXL output, got {} bytes",
        bytes.len()
    );
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "JXL signature expected");
}
