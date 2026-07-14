// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Unit tests for the public API surface (`super` = `crate::api`).
//! Extracted verbatim from api.rs on 2026-07-14 (no test-body changes).

use super::*;

// ─── ec_resampling helper (A1 audit "Pixel formats / extras") ──

/// Box filter at factor=1 is a passthrough.
#[test]
fn test_downsample_channel_u8_factor1_is_passthrough() {
    let src = [10u8, 20, 30, 40, 50, 60, 70, 80, 90];
    let got = downsample_channel_u8(&src, 3, 3, 1);
    assert_eq!(got, src);
}

/// 2×2 box filter on a 4×4 uniform region averages each quadrant.
#[test]
fn test_downsample_channel_u8_factor2_uniform_quadrants() {
    // 4×4 image where each 2×2 quadrant has a distinct value.
    let src: [u8; 16] = [
        10, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40,
    ];
    let got = downsample_channel_u8(&src, 4, 4, 2);
    assert_eq!(got, vec![10, 20, 30, 40]);
}

/// Partial edge cells (image dim not divisible by factor) only average
/// in-bounds samples — matches libjxl `DoDownsampleImage`.
#[test]
fn test_downsample_channel_u8_factor2_partial_edges() {
    // 3×3 image, downsample by 2 → 2×2 output. Output cell (1, 1)
    // averages only the single in-bounds sample at (2, 2).
    let src: [u8; 9] = [10, 20, 30, 40, 50, 60, 70, 80, 90];
    let got = downsample_channel_u8(&src, 3, 3, 2);
    // (0,0): avg of (10,20,40,50) = 120/4 = 30
    // (0,1): avg of (30,60) = 90/2 = 45
    // (1,0): avg of (70,80) = 150/2 = 75
    // (1,1): avg of (90) = 90
    assert_eq!(got, vec![30, 45, 75, 90]);
}

/// Factor=4 over an 8×8 image with a vertical gradient.
#[test]
fn test_downsample_channel_u8_factor4_8x8() {
    // 8 rows × 8 cols, value = row*16 → [0, 16, 32, 48, 64, 80, 96, 112].
    let mut src = vec![0u8; 64];
    for y in 0..8 {
        for x in 0..8 {
            src[y * 8 + x] = (y * 16) as u8;
        }
    }
    let got = downsample_channel_u8(&src, 8, 8, 4);
    // 4×4 box top-left: rows 0..4 → values [0,16,32,48], mean = 24.
    // Bottom-left: rows 4..8 → [64,80,96,112], mean = 88.
    assert_eq!(got, vec![24, 24, 88, 88]);
}

/// Dimensions output match libjxl's `DivCeil(d, factor)`.
#[test]
fn test_downsample_channel_u8_output_dims() {
    let src = vec![42u8; 13 * 17];
    // 13 div_ceil 4 = 4; 17 div_ceil 4 = 5 → 20 samples.
    let got = downsample_channel_u8(&src, 13, 17, 4);
    assert_eq!(got.len(), 20);
    // Uniform input must produce uniform output.
    assert!(got.iter().all(|&v| v == 42));
}

/// factor=0 returns empty (defensive — caller usually validates).
#[test]
fn test_downsample_channel_u8_factor0_returns_empty() {
    let src = [1u8, 2, 3, 4];
    let got = downsample_channel_u8(&src, 2, 2, 0);
    assert!(got.is_empty());
}

// ─── PQ EOTF (closes PQ portion of #17) ───────────────────────

/// Spot-check the PQ EOTF against published reference points
/// (BT.2100 Table 4 / ST 2084 inverse). Tolerance is 1e-3 because
/// the encoder uses fast_powf instead of std::powf — accuracy
/// is in the same neighborhood as libjxl's PQ implementation.
#[test]
fn test_pq_to_linear_f_reference_points() {
    // EOTF[0] = 0
    assert!(pq_to_linear_f(0.0).abs() < 1e-3);
    // EOTF[1] = 1.0 (peak luminance / 10000 nits = full scale)
    let one = pq_to_linear_f(1.0);
    assert!(
        (one - 1.0).abs() < 1e-3,
        "PQ(1.0) should be 1.0 (peak); got {one}",
    );
    // EOTF[0.5081] ≈ 0.01 (= 100 nits / 10000) — the SDR diffuse
    // white reference. Per BT.2100 the encoded value for 100 nits
    // is ~0.508. Tolerance loosened a touch because fast_powf
    // diverges from std::powf in the middle of the range.
    let mid = pq_to_linear_f(0.5081);
    assert!(
        (mid - 0.01).abs() < 5e-3,
        "PQ(0.5081) should be ≈0.01 (100 nits); got {mid}",
    );
    // Monotonic
    let a = pq_to_linear_f(0.25);
    let b = pq_to_linear_f(0.5);
    let c = pq_to_linear_f(0.75);
    assert!(a < b && b < c, "PQ should be monotone; got {a}, {b}, {c}");
}

#[test]
fn test_pq_to_linear_f_clamps_negative() {
    // Negative input clamps to 0 → output ~0 (avoids NaN from
    // x.powf(non-int) on negative x). fast_powf can produce a
    // tiny negative result from rounding; both must be safe to
    // feed into the encoder (no NaN/Inf).
    let v = pq_to_linear_f(-0.1);
    assert!(v.is_finite(), "PQ(-0.1) should be finite; got {v}");
    assert!(v.abs() < 1e-3, "PQ(-0.1) should clamp to ~0; got {v}");
}

/// BT.709 inverse OETF reference points (Rec. BT.709-6).
/// - BT709(0) = 0
/// - BT709(0.081) = 0.018 (toe/shoulder boundary)
/// - BT709(1) = 1.0
/// - Monotonic
#[test]
fn test_bt709_to_linear_f_reference_points() {
    assert!(bt709_to_linear_f(0.0).abs() < 1e-6);
    // Boundary: encoded 0.081 → linear 0.018 (= 0.081 / 4.5).
    let boundary = bt709_to_linear_f(0.081);
    assert!(
        (boundary - 0.018).abs() < 1e-5,
        "BT.709(0.081) should be ≈0.018; got {boundary}",
    );
    let one = bt709_to_linear_f(1.0);
    assert!(
        (one - 1.0).abs() < 1e-3,
        "BT.709(1.0) should be 1.0; got {one}",
    );
    let a = bt709_to_linear_f(0.25);
    let b = bt709_to_linear_f(0.5);
    let c = bt709_to_linear_f(0.75);
    assert!(
        a < b && b < c,
        "BT.709 should be monotone; got {a}, {b}, {c}"
    );
}

#[test]
fn test_bt709_to_linear_f_clamps_negative() {
    let v = bt709_to_linear_f(-0.1);
    assert!(v.is_finite());
    assert!(
        (0.0..1e-3).contains(&v),
        "BT.709(-0.1) should clamp to ~0; got {v}"
    );
}

/// Reference points for HLG inverse OETF (BT.2100).
/// - HLG(0) = 0
/// - HLG(0.5) = 0.25 / 3 = 0.083333... (boundary of toe / shoulder)
/// - HLG(1) = 1.0 (peak signal → peak scene-light)
/// - Monotonic
#[test]
fn test_hlg_to_linear_f_reference_points() {
    assert!(hlg_to_linear_f(0.0).abs() < 1e-6);
    let half = hlg_to_linear_f(0.5);
    assert!(
        (half - (0.25 / 3.0)).abs() < 1e-5,
        "HLG(0.5) should be 0.0833...; got {half}",
    );
    let one = hlg_to_linear_f(1.0);
    assert!(
        (one - 1.0).abs() < 1e-3,
        "HLG(1.0) should be 1.0 (peak); got {one}",
    );
    let a = hlg_to_linear_f(0.25);
    let b = hlg_to_linear_f(0.5);
    let c = hlg_to_linear_f(0.75);
    assert!(a < b && b < c, "HLG should be monotone; got {a}, {b}, {c}");
}

#[test]
fn test_hlg_to_linear_f_clamps_negative() {
    let v = hlg_to_linear_f(-0.1);
    assert!(v.is_finite());
    assert!(
        (0.0..1e-3).contains(&v),
        "HLG(-0.1) should clamp to ~0; got {v}"
    );
}

#[test]
fn test_pq_u16_to_linear_f32_uses_pq_eotf() {
    // 16-bit PQ value 65535 should give linear ≈1.0.
    let pixels_u16: Vec<u16> = vec![65535, 65535, 65535];
    let bytes: &[u8] = bytemuck::cast_slice(&pixels_u16);
    let linear = pq_u16_to_linear_f32(bytes, 3, 65535.0);
    for v in &linear {
        assert!((v - 1.0).abs() < 1e-3, "PQ(1.0) should be ≈1.0; got {v}");
    }
    // 16-bit PQ value 0 should give 0.
    let pixels0: Vec<u16> = vec![0, 0, 0];
    let bytes0: &[u8] = bytemuck::cast_slice(&pixels0);
    let linear0 = pq_u16_to_linear_f32(bytes0, 3, 65535.0);
    for v in &linear0 {
        assert!(v.abs() < 1e-6, "PQ(0) should be 0; got {v}");
    }
}

/// Audit item #3: `effective_profile_for_image` must drop
/// `tree_max_buckets` from 256 → 192 ONLY at the (pixels >= 4 MP,
/// effort >= 9) cell. Every other cell must keep the effort-only
/// default so hash-locks stay stable.
#[test]
fn test_effective_profile_for_image_tree_max_buckets_dispatch() {
    // e9 + large: dispatch fires.
    let cfg = LosslessConfig::new().with_effort(9);
    let p = cfg.effective_profile_for_image(4_194_304);
    assert_eq!(
        p.tree_max_buckets,
        crate::effort::LARGE_E9_TREE_MAX_BUCKETS,
        "e9 large: buckets must drop to 192"
    );

    // e9 + medium (< 4 MP): no dispatch.
    let p = cfg.effective_profile_for_image(1_048_576);
    assert_eq!(p.tree_max_buckets, 256, "e9 medium: buckets stay 256");

    // e7 + large: no dispatch (effort gate).
    let cfg = LosslessConfig::new().with_effort(7);
    let p = cfg.effective_profile_for_image(8_000_000);
    assert_eq!(
        p.tree_max_buckets, 96,
        "e7 large: buckets stay 96 (default)"
    );

    // e10 + large: dispatch fires (effort >= 9).
    let cfg = LosslessConfig::new().with_effort(10);
    let p = cfg.effective_profile_for_image(8_000_000);
    assert_eq!(
        p.tree_max_buckets,
        crate::effort::LARGE_E9_TREE_MAX_BUCKETS,
        "e10 large: buckets drop to 192"
    );
}

/// When the caller has supplied an explicit `__expert`
/// profile_override (e.g. a sweep harness pinning a specific
/// `tree_max_buckets`), the always-on dispatch must NOT silently
/// stomp it.
#[cfg(feature = "__expert")]
#[test]
fn test_effective_profile_for_image_respects_internal_params_override() {
    let params = crate::effort::LosslessInternalParams {
        tree_max_buckets: Some(128),
        ..Default::default()
    };
    let cfg = LosslessConfig::new()
        .with_effort(9)
        .with_internal_params(params);
    let p = cfg.effective_profile_for_image(8_000_000);
    // Override wins — dispatch did not fire.
    assert_eq!(
        p.tree_max_buckets, 128,
        "sweep override must survive the dispatch"
    );
}

/// Chunk 1 VarDCT AC dispatch (`adapt_to_image_lossy`): drop
/// `try_dct64` to `false` ONLY when the image is small (< 500_000
/// pixels) AND distance is low (< 2.0). Every other cell keeps the
/// effort-only default so corpus_regression bytes stay stable.
#[test]
fn test_lossy_effective_profile_for_image_dct64_dispatch() {
    // small + low-d at effort 7: dispatch fires (try_dct64 → false).
    let cfg = LossyConfig::new(1.0).with_effort(7);
    let p = cfg.effective_profile_for_image(256 * 256);
    assert!(
        !p.try_dct64,
        "small_0.07MP + d=1.0 + e7: try_dct64 must drop to false"
    );

    // small_0.26MP (512×512) + d=1.0 + e7: still small + low-d.
    let cfg = LossyConfig::new(1.0).with_effort(7);
    let p = cfg.effective_profile_for_image(512 * 512);
    assert!(
        !p.try_dct64,
        "small_0.26MP + d=1.0 + e7: try_dct64 must drop to false"
    );

    // medium (1 MP) + d=1.0: no dispatch (pixel-count gate).
    let cfg = LossyConfig::new(1.0).with_effort(7);
    let p = cfg.effective_profile_for_image(1024 * 1024);
    assert!(
        p.try_dct64,
        "medium_1.0MP: try_dct64 stays true (pixel gate excludes ≥500k)"
    );

    // small + d=2.0: no dispatch (distance gate is strict <).
    let cfg = LossyConfig::new(2.0).with_effort(7);
    let p = cfg.effective_profile_for_image(256 * 256);
    assert!(
        p.try_dct64,
        "small + d=2.0: try_dct64 stays true (distance gate is strict <2.0)"
    );

    // small + d=5.0: no dispatch (distance gate).
    let cfg = LossyConfig::new(5.0).with_effort(7);
    let p = cfg.effective_profile_for_image(256 * 256);
    assert!(p.try_dct64, "small + d=5.0: try_dct64 stays true");

    // small + low-d + effort 5: no dispatch (effort < 7 means
    // try_dct64 is already false in the default profile — adapter
    // is a no-op, no false-flip-to-true).
    let cfg = LossyConfig::new(1.0).with_effort(5);
    let p = cfg.effective_profile_for_image(256 * 256);
    assert!(
        !p.try_dct64,
        "small + d=1.0 + e5: try_dct64 already false at effort < 7"
    );

    // large + low-d at e7: no dispatch (pixel gate).
    let cfg = LossyConfig::new(0.5).with_effort(7);
    let p = cfg.effective_profile_for_image(4_194_304);
    assert!(p.try_dct64, "large_4MP + d=0.5 + e7: try_dct64 stays true");
}

/// W44-35 smooth-photo DCT64 admission gate: when the smoothness
/// auto-detector (or caller hint) is `true`, suppress the
/// `adapt_to_image_lossy` `try_dct64 -> false` flip even on the
/// gated cell.
#[test]
fn test_lossy_effective_profile_for_image_smooth_photo_admission() {
    // Baseline: small + low-d + e7 with `smooth_photo=false` keeps
    // the gated behaviour (try_dct64 = false). Matches the pre-W44-35
    // result asserted in the existing dct64_dispatch test.
    let cfg = LossyConfig::new(1.0).with_effort(7);
    let p = cfg.effective_profile_for_image_with_smoothness(512 * 512, false);
    assert!(
        !p.try_dct64,
        "smooth_photo=false on gated cell: try_dct64 stays false"
    );

    // Auto detector returns `true` (input classified smooth) →
    // the dispatch must restore try_dct64=true so the encoder
    // evaluates DCT64-class transforms. This is the W44-34 fix.
    let cfg = LossyConfig::new(1.0).with_effort(7);
    let p = cfg.effective_profile_for_image_with_smoothness(512 * 512, true);
    assert!(
        p.try_dct64,
        "smooth_photo=true on gated cell: try_dct64 restored to true (W44-35)"
    );

    // Caller hint Some(true) wins over auto detector value false
    // (W44-130 Chunk D: hint moved into `StrategyOverrides`).
    let cfg = LossyConfig::new(1.0)
        .with_effort(7)
        .with_strategy_overrides(StrategyOverrides {
            smooth_photo_dct64_hint: Some(true),
            ..Default::default()
        });
    let p = cfg.effective_profile_for_image_with_smoothness(512 * 512, false);
    assert!(
        p.try_dct64,
        "explicit hint Some(true) wins over auto=false: try_dct64=true"
    );

    // Caller hint Some(false) wins over auto detector value true
    // (W44-130 Chunk D: hint moved into `StrategyOverrides`).
    let cfg = LossyConfig::new(1.0)
        .with_effort(7)
        .with_strategy_overrides(StrategyOverrides {
            smooth_photo_dct64_hint: Some(false),
            ..Default::default()
        });
    let p = cfg.effective_profile_for_image_with_smoothness(512 * 512, true);
    assert!(
        !p.try_dct64,
        "explicit hint Some(false) wins over auto=true: try_dct64=false"
    );

    // Outside the gate envelope (medium image), smoothness signal is
    // irrelevant — try_dct64 stays at the effort default (true at e7).
    let cfg = LossyConfig::new(1.0).with_effort(7);
    let p = cfg.effective_profile_for_image_with_smoothness(1024 * 1024, false);
    assert!(
        p.try_dct64,
        "medium image: try_dct64 stays true regardless of smoothness"
    );

    // At e6 the baseline try_dct64 is false (effort gate); on the
    // small + low-d cell the smooth-photo hint admits it (forced
    // true) so the encoder evaluates DCT64-class transforms.
    // Closes the 1418519 e6 cells (W44-34 forensics).
    let cfg = LossyConfig::new(1.2).with_effort(6);
    let p_default = cfg.effective_profile_for_image_with_smoothness(512 * 512, false);
    assert!(
        !p_default.try_dct64,
        "e6 baseline: try_dct64 stays false (pre-W44-35 behaviour)"
    );
    let p_smooth = cfg.effective_profile_for_image_with_smoothness(512 * 512, true);
    assert!(
        p_smooth.try_dct64,
        "e6 + smooth_photo=true on gated cell: try_dct64 admitted (W44-35)"
    );
}

/// W44-35 auto detector: smooth photo (low edge, low HF, low solid
/// fill) returns `true`; textured / screen-content / large images
/// return `false`.
#[test]
fn test_detect_smooth_photo_for_dct64() {
    // Large input (>= 500_000 px): short-circuits to false even on
    // smooth content (the gate it informs doesn't fire above 500k).
    let large_smooth = vec![128u8; 800 * 800 * 3];
    assert!(!detect_smooth_photo_for_dct64_from_layout(
        &large_smooth,
        800,
        800,
        PixelLayout::Rgb8,
    ));

    // Flat solid mid-gray (variance=0 everywhere) on a 512×512:
    // proxy_flat is 1.0 (all blocks solid) → rejected as
    // screenshot-like.
    let solid = vec![128u8; 512 * 512 * 3];
    assert!(!detect_smooth_photo_for_dct64_from_layout(
        &solid,
        512,
        512,
        PixelLayout::Rgb8,
    ));

    // Smooth low-frequency texture (photo-like): low edge density,
    // moderate flat ratio, low HF — should classify as smooth photo.
    // Built from a slow sinusoidal modulation so per-block variance
    // sits in the "smooth gradient" band (var > 5 → not "solid")
    // but the wavelength is long enough that proxy_edge and
    // proxy_hf both stay below the admission thresholds.
    let mut smooth = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            // Slow sinusoid in both axes, mean=128, amp~80.
            let fx = (x as f32) * 0.02; // ~32px wavelength
            let fy = (y as f32) * 0.02;
            let v = (128.0 + 80.0 * fx.sin() * fy.cos()).clamp(0.0, 255.0) as u8;
            let i = (y * 256 + x) * 3;
            smooth[i] = v;
            smooth[i + 1] = v;
            smooth[i + 2] = v;
        }
    }
    assert!(
        detect_smooth_photo_for_dct64_from_layout(&smooth, 256, 256, PixelLayout::Rgb8),
        "low-frequency sinusoidal texture should classify as smooth photo"
    );

    // Coarse high-contrast checkerboard (8×8 cells): high edge
    // density, screen-content-like — rejected. Cell size 8 is past
    // the 4× downsample Nyquist so edges survive into the proxy.
    let mut checker = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let v = if on { 255u8 } else { 0 };
            let i = (y * 256 + x) * 3;
            checker[i] = v;
            checker[i + 1] = v;
            checker[i + 2] = v;
        }
    }
    assert!(
        !detect_smooth_photo_for_dct64_from_layout(&checker, 256, 256, PixelLayout::Rgb8),
        "coarse high-contrast checkerboard must NOT classify as smooth photo"
    );

    // Non-u8 layouts return false (auto detector skipped — caller
    // can still set Some(true) via the hint API).
    let f32_data = vec![0u8; 256 * 256 * 4 * 4]; // float pixels
    assert!(!detect_smooth_photo_for_dct64_from_layout(
        &f32_data,
        256,
        256,
        PixelLayout::RgbaLinearF32,
    ));
}

/// W44-164: the auto-classifier discriminator predicate. Exercises
/// the pure `classify_from_proxies` function with synthesised
/// proxy values that bracket the threshold (no allocator / no
/// O(W·H) scan).
#[test]
fn test_w44_164_classify_from_proxies_screenshot() {
    use crate::effort::ImageContentClass;
    use crate::vardct::encoder::ZenanalyzeProxies;
    // gb82-sc gmessages (real corpus): fcbr=0.907, m3=10.67
    let p = ZenanalyzeProxies {
        m3_colourfulness: 10.67,
        flat_color_block_ratio: 0.907,
        edge_density: 0.021,
        luma_var: 0.0,
    };
    assert_eq!(classify_from_proxies(&p), ImageContentClass::Screenshot);
    // gb82-sc windows95 (real corpus, outlier): fcbr=0.360. Above
    // the W44_164 threshold (0.35) → Screenshot.
    let p = ZenanalyzeProxies {
        m3_colourfulness: 27.19,
        flat_color_block_ratio: 0.360,
        edge_density: 0.268,
        luma_var: 0.0,
    };
    assert_eq!(classify_from_proxies(&p), ImageContentClass::Screenshot);
    // gb82-sc imac_g3 (real corpus): fcbr=0.709, m3=15.32
    let p = ZenanalyzeProxies {
        m3_colourfulness: 15.32,
        flat_color_block_ratio: 0.709,
        edge_density: 0.079,
        luma_var: 0.0,
    };
    assert_eq!(classify_from_proxies(&p), ImageContentClass::Screenshot);
}

#[test]
fn test_w44_164_classify_from_proxies_photo() {
    use crate::effort::ImageContentClass;
    use crate::vardct::encoder::ZenanalyzeProxies;
    // 1189261 (W44-91 TARGET, real corpus): fcbr=0.0034, m3=98.84
    let p = ZenanalyzeProxies {
        m3_colourfulness: 98.84,
        flat_color_block_ratio: 0.0034,
        edge_density: 0.633,
        luma_var: 0.0,
    };
    assert_eq!(classify_from_proxies(&p), ImageContentClass::Photo);
    // 1025469 (W44-91 REGRESSION cell, real corpus): fcbr=0.0166,
    // m3=45.45 — must classify as Photo (NOT Screenshot), proving
    // the auto-classifier does not misfire on the W44-91 regression
    // band.
    let p = ZenanalyzeProxies {
        m3_colourfulness: 45.45,
        flat_color_block_ratio: 0.0166,
        edge_density: 0.300,
        luma_var: 0.0,
    };
    assert_eq!(classify_from_proxies(&p), ImageContentClass::Photo);
    // 297394 (high-colour photo, fcbr=0.0957) — just below the
    // 0.10 photo ceiling, fcbr still in photo range, m3 high.
    let p = ZenanalyzeProxies {
        m3_colourfulness: 103.70,
        flat_color_block_ratio: 0.0957,
        edge_density: 0.300,
        luma_var: 0.0,
    };
    assert_eq!(classify_from_proxies(&p), ImageContentClass::Photo);
}

#[test]
fn test_w44_164_classify_from_proxies_deadband() {
    use crate::effort::ImageContentClass;
    use crate::vardct::encoder::ZenanalyzeProxies;
    // Synthetic deadband (fcbr ∈ [0.10, 0.35)) — no fcbr value
    // currently observed in either corpus sits here, but the
    // classifier must short-circuit defensively. The deadband
    // exists so future inputs in the gap don't get misclassified.
    for fcbr in [0.10_f32, 0.15, 0.20, 0.25, 0.30, 0.34] {
        let p = ZenanalyzeProxies {
            m3_colourfulness: 50.0,
            flat_color_block_ratio: fcbr,
            edge_density: 0.2,
            luma_var: 0.0,
        };
        assert_eq!(
            classify_from_proxies(&p),
            ImageContentClass::Unknown,
            "fcbr={fcbr} in deadband must classify as Unknown",
        );
    }
    // Near-grayscale low-m3 content (fcbr below photo ceiling but
    // m3 below the photo floor) → Unknown (no class adapter fires
    // → byte-identical to pre-W44-164).
    let p = ZenanalyzeProxies {
        m3_colourfulness: 2.0,
        flat_color_block_ratio: 0.05,
        edge_density: 0.1,
        luma_var: 0.0,
    };
    assert_eq!(classify_from_proxies(&p), ImageContentClass::Unknown);
}

/// W44-164 auto-classifier entry point: short-circuits below the
/// `CONTENT_CLASS_MIN_PIXELS` (= 65,536 px) gate so the per-encode
/// O(W·H) scan is skipped on hash-lock-sized fixtures (largest
/// 48×48 = 2,304 px). The pixel gate exists in BOTH the helper
/// AND `adapt_to_image_content` so the dispatch is doubly
/// short-circuited — defence in depth.
#[test]
fn test_w44_164_auto_classify_pixel_gate() {
    // 48×48 RGB (largest hash-lock fixture): below 65,536 → None.
    let small = vec![200u8; 48 * 48 * 3];
    let result = auto_classify_content_class_from_layout(&small, 48, 48, PixelLayout::Rgb8);
    assert_eq!(
        result, None,
        "auto-classifier MUST short-circuit below CONTENT_CLASS_MIN_PIXELS \
             so hash-lock fixtures stay byte-identical"
    );
    // 256×256 (= 65,536 px): exactly at threshold → computes.
    // Construct fcbr-screenshot-like input: solid mid-gray (range=0
    // on every block, fcbr=1.0).
    let solid = vec![128u8; 256 * 256 * 3];
    let result = auto_classify_content_class_from_layout(&solid, 256, 256, PixelLayout::Rgb8);
    // Solid mid-gray: fcbr=1.0 (every block range=0), m3=0 →
    // Screenshot via fcbr branch.
    assert_eq!(
        result,
        Some(crate::effort::ImageContentClass::Screenshot),
        "solid 256×256 (fcbr=1.0) must classify as Screenshot"
    );
}

/// W44-164: layout dispatch — non-u8-sRGB layouts return None
/// (the proxy scan only knows BT.601-shaped layouts).
#[test]
fn test_w44_164_auto_classify_layout_dispatch() {
    // 16-bit and float layouts return None — auto-classifier
    // unavailable; caller can still set `with_content_class(Some)`
    // explicitly. Use 256×256 (above pixel threshold) so the
    // layout gate is the only thing rejecting.
    let f32_data = vec![0u8; 256 * 256 * 4 * 4];
    assert_eq!(
        auto_classify_content_class_from_layout(&f32_data, 256, 256, PixelLayout::RgbaLinearF32),
        None,
    );
    let u16_data = vec![0u8; 256 * 256 * 3 * 2];
    assert_eq!(
        auto_classify_content_class_from_layout(&u16_data, 256, 256, PixelLayout::Rgb16),
        None,
    );
    let gray_data = vec![0u8; 256 * 256];
    assert_eq!(
        auto_classify_content_class_from_layout(&gray_data, 256, 256, PixelLayout::Gray8),
        None,
    );
}

/// W44-164: ResolvedImprovements.content_class_auto_classify
/// defaults per strategy.
#[test]
fn test_w44_164_resolved_default_per_strategy() {
    let zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert!(
        zenjxl.content_class_auto_classify,
        "Zenjxl must enable the auto-classifier"
    );
    let aggressive = EncoderStrategy::Aggressive.resolve(&StrategyOverrides::default());
    assert!(
        aggressive.content_class_auto_classify,
        "Aggressive must enable the auto-classifier"
    );
    let libjxl = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());
    assert!(
        !libjxl.content_class_auto_classify,
        "Libjxl must disable the auto-classifier (strict parity)"
    );
    let lean = EncoderStrategy::LeanFaster.resolve(&StrategyOverrides::default());
    assert!(
        !lean.content_class_auto_classify,
        "LeanFaster must disable the auto-classifier (skip heavy per-image gates)"
    );
    // Custom inherits whatever the user set on EncoderImprovementsCustom.
    let mut custom = EncoderImprovementsCustom::default();
    assert!(
        custom.content_class_auto_classify,
        "EncoderImprovementsCustom::default() matches Zenjxl"
    );
    custom.content_class_auto_classify = false;
    let resolved = EncoderStrategy::Custom(Box::new(custom)).resolve(&StrategyOverrides::default());
    assert!(
        !resolved.content_class_auto_classify,
        "Custom with field set false must propagate"
    );
}

/// W44-165: ResolvedImprovements.photo_epf_seed_admit defaults
/// per strategy. Zenjxl + Aggressive enable; Libjxl + LeanFaster
/// disable.
#[test]
fn test_w44_165_photo_epf_seed_admit_default_per_strategy() {
    let zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert!(
        zenjxl.photo_epf_seed_admit,
        "Zenjxl must enable W44-165 photo EPF seed admission"
    );
    let aggressive = EncoderStrategy::Aggressive.resolve(&StrategyOverrides::default());
    assert!(
        aggressive.photo_epf_seed_admit,
        "Aggressive must enable W44-165 photo EPF seed admission"
    );
    let libjxl = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());
    assert!(
        !libjxl.photo_epf_seed_admit,
        "Libjxl must disable W44-165 (strict parity — W44-150 honest-stop disposition)"
    );
    let lean = EncoderStrategy::LeanFaster.resolve(&StrategyOverrides::default());
    assert!(
        !lean.photo_epf_seed_admit,
        "LeanFaster must disable W44-165 (skip per-image mask percentile cost)"
    );
    // Custom inherits whatever the user set on EncoderImprovementsCustom.
    let mut custom = EncoderImprovementsCustom::default();
    assert!(
        custom.photo_epf_seed_admit,
        "EncoderImprovementsCustom::default() matches Zenjxl"
    );
    custom.photo_epf_seed_admit = false;
    let resolved = EncoderStrategy::Custom(Box::new(custom)).resolve(&StrategyOverrides::default());
    assert!(
        !resolved.photo_epf_seed_admit,
        "Custom with field set false must propagate"
    );
}

/// W44-166: ResolvedImprovements.photo_variant_z_admit defaults
/// per strategy. Zenjxl + Aggressive enable; Libjxl + LeanFaster
/// disable. Mirrors the W44-165 photo_epf_seed_admit pattern.
#[test]
fn test_w44_166_photo_variant_z_admit_default_per_strategy() {
    let zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert!(
        zenjxl.photo_variant_z_admit,
        "Zenjxl must enable W44-166 photo variant Z admission"
    );
    let aggressive = EncoderStrategy::Aggressive.resolve(&StrategyOverrides::default());
    assert!(
        aggressive.photo_variant_z_admit,
        "Aggressive must enable W44-166 photo variant Z admission"
    );
    let libjxl = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());
    assert!(
        !libjxl.photo_variant_z_admit,
        "Libjxl must disable W44-166 (strict parity — W44-148 DO-NOT \
             '1418519 OUTSIDE variant Z reach' disposition)"
    );
    let lean = EncoderStrategy::LeanFaster.resolve(&StrategyOverrides::default());
    assert!(
        !lean.photo_variant_z_admit,
        "LeanFaster must disable W44-166 (skip per-image mask percentile cost)"
    );
    // Custom inherits whatever the user set on EncoderImprovementsCustom.
    let mut custom = EncoderImprovementsCustom::default();
    assert!(
        custom.photo_variant_z_admit,
        "EncoderImprovementsCustom::default() matches Zenjxl"
    );
    custom.photo_variant_z_admit = false;
    let resolved = EncoderStrategy::Custom(Box::new(custom)).resolve(&StrategyOverrides::default());
    assert!(
        !resolved.photo_variant_z_admit,
        "Custom with field set false must propagate"
    );
}

/// W44-167: ResolvedImprovements.find_best_32_per_m3_lift defaults
/// per strategy. Zenjxl + Aggressive enable; Libjxl + LeanFaster
/// disable. Mirrors the W44-166 photo_variant_z_admit pattern.
#[test]
fn test_w44_167_find_best_32_per_m3_lift_default_per_strategy() {
    let zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert!(
        zenjxl.find_best_32_per_m3_lift,
        "Zenjxl must enable W44-167 per-m3 lift"
    );
    let aggressive = EncoderStrategy::Aggressive.resolve(&StrategyOverrides::default());
    assert!(
        aggressive.find_best_32_per_m3_lift,
        "Aggressive must enable W44-167 per-m3 lift"
    );
    let libjxl = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());
    assert!(
        !libjxl.find_best_32_per_m3_lift,
        "Libjxl must disable W44-167 (strict parity — W44-94 honest-stop)"
    );
    let lean = EncoderStrategy::LeanFaster.resolve(&StrategyOverrides::default());
    assert!(
        !lean.find_best_32_per_m3_lift,
        "LeanFaster must disable W44-167 (skip per-image proxy gate cost)"
    );
    // Custom inherits whatever the user set on EncoderImprovementsCustom.
    let mut custom = EncoderImprovementsCustom::default();
    assert!(
        custom.find_best_32_per_m3_lift,
        "EncoderImprovementsCustom::default() matches Zenjxl"
    );
    custom.find_best_32_per_m3_lift = false;
    let resolved = EncoderStrategy::Custom(Box::new(custom)).resolve(&StrategyOverrides::default());
    assert!(
        !resolved.find_best_32_per_m3_lift,
        "Custom with field set false must propagate"
    );
}

/// W44-168 (Smart-Zenjxl chunk 5): `adaptive_buttloop_iters` defaults
/// per strategy. Mirrors the W44-166/167 per-strategy test pattern.
#[test]
fn test_w44_168_adaptive_buttloop_iters_default_per_strategy() {
    let zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert!(
        zenjxl.adaptive_buttloop_iters,
        "Zenjxl must enable W44-168 adaptive buttloop iters"
    );
    let aggressive = EncoderStrategy::Aggressive.resolve(&StrategyOverrides::default());
    assert!(
        aggressive.adaptive_buttloop_iters,
        "Aggressive must enable W44-168 adaptive buttloop iters"
    );
    let libjxl = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());
    assert!(
        !libjxl.adaptive_buttloop_iters,
        "Libjxl must disable W44-168 (strict per-effort iter parity)"
    );
    let lean = EncoderStrategy::LeanFaster.resolve(&StrategyOverrides::default());
    assert!(
        !lean.adaptive_buttloop_iters,
        "LeanFaster must disable W44-168 (skip per-image proxy gate cost)"
    );
    // Custom inherits whatever the user set on EncoderImprovementsCustom.
    let mut custom = EncoderImprovementsCustom::default();
    assert!(
        custom.adaptive_buttloop_iters,
        "EncoderImprovementsCustom::default() matches Zenjxl"
    );
    custom.adaptive_buttloop_iters = false;
    let resolved = EncoderStrategy::Custom(Box::new(custom)).resolve(&StrategyOverrides::default());
    assert!(
        !resolved.adaptive_buttloop_iters,
        "Custom with field set false must propagate"
    );
}

/// W44-169 (Smart-Zenjxl chunk 6): `adaptive_buttloop_iters_narrow`
/// defaults per strategy. Mirrors the W44-168 per-strategy test.
#[test]
fn test_w44_169_adaptive_buttloop_iters_narrow_default_per_strategy() {
    let zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert!(
        zenjxl.adaptive_buttloop_iters_narrow,
        "Zenjxl must enable W44-169 narrow SmoothSkip (production SHIPPED)"
    );
    let aggressive = EncoderStrategy::Aggressive.resolve(&StrategyOverrides::default());
    assert!(
        aggressive.adaptive_buttloop_iters_narrow,
        "Aggressive must enable W44-169 narrow SmoothSkip"
    );
    let libjxl = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());
    assert!(
        !libjxl.adaptive_buttloop_iters_narrow,
        "Libjxl must disable W44-169 (strict per-effort iter parity)"
    );
    let lean = EncoderStrategy::LeanFaster.resolve(&StrategyOverrides::default());
    assert!(
        !lean.adaptive_buttloop_iters_narrow,
        "LeanFaster must disable W44-169 (skip per-image proxy gate cost)"
    );
    let mut custom = EncoderImprovementsCustom::default();
    assert!(
        custom.adaptive_buttloop_iters_narrow,
        "EncoderImprovementsCustom::default() matches Zenjxl"
    );
    custom.adaptive_buttloop_iters_narrow = false;
    let resolved = EncoderStrategy::Custom(Box::new(custom)).resolve(&StrategyOverrides::default());
    assert!(
        !resolved.adaptive_buttloop_iters_narrow,
        "Custom with field set false must propagate"
    );
}

/// W44-164 explicit `with_content_class(Some(...))` ALWAYS wins
/// over the auto-classifier even on Zenjxl.
#[test]
fn test_w44_164_explicit_content_class_wins() {
    use crate::effort::ImageContentClass;
    // Build a 256×256 sRGB Photo-class input (smooth gradient,
    // low fcbr). Auto-classifier should call this Photo; the
    // explicit override (Screenshot) must win.
    let mut photo = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let i = (y * 256 + x) * 3;
            let r = (x as u8).wrapping_add(50);
            let g = ((x + y) as u8 / 2).wrapping_add(80);
            let b = (y as u8).wrapping_add(110);
            photo[i] = r;
            photo[i + 1] = g;
            photo[i + 2] = b;
        }
    }
    // Sanity-check the input: classifier sees this as Photo (or
    // Unknown if fcbr or m3 falls outside the bands; in either
    // case NOT Screenshot — we just need to verify the override
    // path).
    let auto = auto_classify_content_class_from_layout(&photo, 256, 256, PixelLayout::Rgb8);
    assert_ne!(
        auto,
        Some(ImageContentClass::Screenshot),
        "synthetic gradient should not auto-classify as Screenshot"
    );

    // Build a Zenjxl LossyConfig + explicit Screenshot override.
    // At e5, baseline (no class) → patches=false. Auto-classifier
    // saying Photo → patches stays false. Explicit override
    // Screenshot → patches=true (per `adapt_to_image_content`).
    let cfg = LossyConfig::new(1.0)
        .with_effort(5)
        .with_content_class(Some(ImageContentClass::Screenshot));
    let p = cfg.effective_profile_for_image_with_smoothness_and_class(
        256 * 256,
        false,
        auto, // Photo or Unknown, NEVER Screenshot
    );
    assert!(
        p.patches,
        "explicit with_content_class(Screenshot) must enable patches at e5"
    );

    // Now flip the override OFF (Some(Photo)) but auto says
    // Screenshot — explicit Photo still wins → no patches.
    let cfg = LossyConfig::new(1.0)
        .with_effort(5)
        .with_content_class(Some(ImageContentClass::Photo));
    let p = cfg.effective_profile_for_image_with_smoothness_and_class(
        256 * 256,
        false,
        Some(ImageContentClass::Screenshot),
    );
    assert!(
        !p.patches,
        "explicit with_content_class(Photo) must keep patches off at e5"
    );
}

/// W44-164: auto-classifier fires on Zenjxl + Aggressive when the
/// caller leaves content_class unset.
#[test]
fn test_w44_164_auto_fires_on_zenjxl() {
    use crate::effort::ImageContentClass;
    // Zenjxl + no caller class + auto-classifier says Screenshot:
    // patches flips on at e5.
    let cfg = LossyConfig::new(1.0).with_effort(5);
    assert_eq!(cfg.content_class, None);
    let p = cfg.effective_profile_for_image_with_smoothness_and_class(
        256 * 256,
        false,
        Some(ImageContentClass::Screenshot),
    );
    assert!(
        p.patches,
        "Zenjxl + auto-classifier=Screenshot must enable patches at e5"
    );
    // Same scenario but on Libjxl: auto-classifier DOES NOT fire,
    // patches stays off (matching libjxl behaviour).
    let cfg = LossyConfig::new(1.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::Libjxl);
    let p = cfg.effective_profile_for_image_with_smoothness_and_class(
        256 * 256,
        false,
        Some(ImageContentClass::Screenshot),
    );
    assert!(
        !p.patches,
        "Libjxl must NOT auto-classify (strict parity), patches stays off"
    );
    // LeanFaster: same as Libjxl on this axis (heavy gates dropped).
    let cfg = LossyConfig::new(1.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::LeanFaster);
    let p = cfg.effective_profile_for_image_with_smoothness_and_class(
        256 * 256,
        false,
        Some(ImageContentClass::Screenshot),
    );
    assert!(
        !p.patches,
        "LeanFaster must NOT auto-classify, patches stays off"
    );
    // Aggressive: auto-classifier fires (same as Zenjxl).
    let cfg = LossyConfig::new(1.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::Aggressive);
    let p = cfg.effective_profile_for_image_with_smoothness_and_class(
        256 * 256,
        false,
        Some(ImageContentClass::Screenshot),
    );
    assert!(p.patches, "Aggressive must auto-classify same as Zenjxl");
}

/// `__expert` sweep override pinning `try_dct64=Some(true)` must
/// survive the per-image dispatch — mirrors the lossless override-
/// respecting behaviour.
#[cfg(feature = "__expert")]
#[test]
fn test_lossy_effective_profile_for_image_respects_internal_params_override() {
    let params = crate::effort::LossyInternalParams {
        try_dct64: Some(true),
        ..Default::default()
    };
    let cfg = LossyConfig::new(1.0)
        .with_effort(7)
        .with_internal_params(params);
    let p = cfg.effective_profile_for_image(256 * 256);
    // Override wins — dispatch did not fire.
    assert!(
        p.try_dct64,
        "sweep override try_dct64=Some(true) must survive the dispatch"
    );
}

#[test]
fn test_lossless_config_builder_and_getters() {
    let cfg = LosslessConfig::new()
        .with_effort(5)
        .with_ans(false)
        .with_squeeze(true)
        .with_tree_learning(true);
    assert_eq!(cfg.effort(), 5);
    assert!(!cfg.ans());
    assert!(cfg.squeeze());
    assert!(cfg.tree_learning());
}

#[test]
fn test_lossy_config_builder_and_getters() {
    let cfg = LossyConfig::new(2.0)
        .with_effort(3)
        .with_gaborish(false)
        .with_noise(true);
    assert_eq!(cfg.distance(), 2.0);
    assert_eq!(cfg.effort(), 3);
    assert!(!cfg.gaborish());
    assert!(cfg.noise());
}

#[test]
fn test_lossy_config_epf_level_default_and_override() {
    // Default is -1 (encoder chooses).
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.epf_level(), -1);

    // Forced levels round-trip 0..=3.
    for level in [0i8, 1, 2, 3] {
        let cfg = LossyConfig::new(1.0).with_epf_level(level);
        assert_eq!(cfg.epf_level(), level);
    }

    // Values outside the libjxl `-1..=3` band are clamped.
    assert_eq!(LossyConfig::new(1.0).with_epf_level(-5).epf_level(), -1);
    assert_eq!(LossyConfig::new(1.0).with_epf_level(7).epf_level(), 3);
}

#[test]
fn test_pixel_layout_helpers() {
    assert_eq!(PixelLayout::Rgb8.bytes_per_pixel(), 3);
    assert_eq!(PixelLayout::Rgba8.bytes_per_pixel(), 4);
    assert_eq!(PixelLayout::Bgr8.bytes_per_pixel(), 3);
    assert_eq!(PixelLayout::Bgra8.bytes_per_pixel(), 4);
    assert_eq!(PixelLayout::Gray8.bytes_per_pixel(), 1);
    assert_eq!(PixelLayout::GrayAlpha8.bytes_per_pixel(), 2);
    assert_eq!(PixelLayout::Rgb16.bytes_per_pixel(), 6);
    assert_eq!(PixelLayout::Rgba16.bytes_per_pixel(), 8);
    assert_eq!(PixelLayout::Gray16.bytes_per_pixel(), 2);
    assert_eq!(PixelLayout::GrayAlpha16.bytes_per_pixel(), 4);
    assert_eq!(PixelLayout::RgbLinearF32.bytes_per_pixel(), 12);
    assert_eq!(PixelLayout::RgbaLinearF32.bytes_per_pixel(), 16);
    assert_eq!(PixelLayout::GrayLinearF32.bytes_per_pixel(), 4);
    assert_eq!(PixelLayout::GrayAlphaLinearF32.bytes_per_pixel(), 8);
    // Linear
    assert!(!PixelLayout::Rgb8.is_linear());
    assert!(PixelLayout::RgbLinearF32.is_linear());
    assert!(PixelLayout::RgbaLinearF32.is_linear());
    assert!(PixelLayout::GrayLinearF32.is_linear());
    assert!(PixelLayout::GrayAlphaLinearF32.is_linear());
    assert!(!PixelLayout::Rgb16.is_linear());
    // Alpha
    assert!(!PixelLayout::Rgb8.has_alpha());
    assert!(PixelLayout::Rgba8.has_alpha());
    assert!(PixelLayout::Bgra8.has_alpha());
    assert!(PixelLayout::GrayAlpha8.has_alpha());
    assert!(PixelLayout::Rgba16.has_alpha());
    assert!(PixelLayout::GrayAlpha16.has_alpha());
    assert!(PixelLayout::RgbaLinearF32.has_alpha());
    assert!(PixelLayout::GrayAlphaLinearF32.has_alpha());
    assert!(!PixelLayout::Rgb16.has_alpha());
    assert!(!PixelLayout::RgbLinearF32.has_alpha());
    // 16-bit
    assert!(PixelLayout::Rgb16.is_16bit());
    assert!(PixelLayout::Rgba16.is_16bit());
    assert!(PixelLayout::Gray16.is_16bit());
    assert!(PixelLayout::GrayAlpha16.is_16bit());
    assert!(!PixelLayout::Rgb8.is_16bit());
    assert!(!PixelLayout::RgbLinearF32.is_16bit());
    // f32
    assert!(PixelLayout::RgbLinearF32.is_f32());
    assert!(PixelLayout::RgbaLinearF32.is_f32());
    assert!(PixelLayout::GrayLinearF32.is_f32());
    assert!(PixelLayout::GrayAlphaLinearF32.is_f32());
    assert!(!PixelLayout::Rgb8.is_f32());
    assert!(!PixelLayout::Rgb16.is_f32());
    // Grayscale
    assert!(PixelLayout::Gray8.is_grayscale());
    assert!(PixelLayout::GrayAlpha8.is_grayscale());
    assert!(PixelLayout::Gray16.is_grayscale());
    assert!(PixelLayout::GrayAlpha16.is_grayscale());
    assert!(PixelLayout::GrayLinearF32.is_grayscale());
    assert!(PixelLayout::GrayAlphaLinearF32.is_grayscale());
    assert!(!PixelLayout::Rgb16.is_grayscale());
    assert!(!PixelLayout::RgbLinearF32.is_grayscale());
    // CMYK (#58) — 4-byte/8-byte, no alpha, not grayscale,
    // is_cmyk() flagged, Cmyk16 is also 16-bit.
    assert_eq!(PixelLayout::Cmyk8.bytes_per_pixel(), 4);
    assert_eq!(PixelLayout::Cmyk16.bytes_per_pixel(), 8);
    assert!(PixelLayout::Cmyk8.is_cmyk());
    assert!(PixelLayout::Cmyk16.is_cmyk());
    assert!(!PixelLayout::Rgb8.is_cmyk());
    assert!(!PixelLayout::Rgba8.is_cmyk());
    assert!(!PixelLayout::Cmyk8.has_alpha());
    assert!(!PixelLayout::Cmyk16.has_alpha());
    assert!(!PixelLayout::Cmyk8.is_grayscale());
    assert!(PixelLayout::Cmyk16.is_16bit());
    assert!(!PixelLayout::Cmyk8.is_16bit());
}

#[test]
fn test_quality_to_distance() {
    assert!(Quality::Distance(1.0).to_distance().unwrap() == 1.0);
    assert!(Quality::Distance(-1.0).to_distance().is_err());
    assert!(Quality::Percent(100).to_distance().is_err()); // lossless invalid for lossy
    assert!(Quality::Percent(90).to_distance().unwrap() == 1.0);
}

#[test]
fn test_pixel_validation() {
    let cfg = LosslessConfig::new();
    let req = cfg.encode_request(2, 2, PixelLayout::Rgb8);
    assert!(req.validate_pixels(&[0u8; 12]).is_ok());
}

#[test]
fn test_pixel_validation_wrong_size() {
    let cfg = LosslessConfig::new();
    let req = cfg.encode_request(2, 2, PixelLayout::Rgb8);
    assert!(req.validate_pixels(&[0u8; 11]).is_err());
}

#[test]
fn test_limits_check() {
    let limits = Limits::new().with_max_width(100);
    let cfg = LosslessConfig::new();
    let req = cfg
        .encode_request(200, 100, PixelLayout::Rgb8)
        .with_limits(&limits);
    assert!(req.check_limits().is_err());
}

#[test]
fn test_lossless_encode_rgb8_small() {
    // 4x4 red image
    let pixels = [255u8, 0, 0].repeat(16);
    let result = LosslessConfig::new()
        .encode_request(4, 4, PixelLayout::Rgb8)
        .encode(&pixels);
    assert!(result.is_ok());
    let jxl = result.unwrap();
    assert_eq!(&jxl[..2], &[0xFF, 0x0A]); // JXL signature
}

#[test]
fn test_lossy_encode_rgb8_small() {
    // 8x8 gradient
    let mut pixels = Vec::with_capacity(8 * 8 * 3);
    for y in 0..8u8 {
        for x in 0..8u8 {
            pixels.push(x * 32);
            pixels.push(y * 32);
            pixels.push(128);
        }
    }
    let result = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(8, 8, PixelLayout::Rgb8)
        .encode(&pixels);
    assert!(result.is_ok());
    let jxl = result.unwrap();
    assert_eq!(&jxl[..2], &[0xFF, 0x0A]);
}

#[test]
fn test_fluent_lossless() {
    let pixels = vec![128u8; 4 * 4 * 3];
    let result = LosslessConfig::new().encode(&pixels, 4, 4, PixelLayout::Rgb8);
    assert!(result.is_ok());
}

#[test]
fn test_lossy_gray8() {
    // Grayscale input → RGB expansion → VarDCT (XYB)
    let pixels = vec![128u8; 8 * 8];
    let result = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(8, 8, PixelLayout::Gray8)
        .encode(&pixels);
    assert!(result.is_ok(), "lossy Gray8 should encode: {result:?}");
}

#[test]
fn test_lossy_gray_alpha8() {
    let pixels: Vec<u8> = (0..8 * 8).flat_map(|_| [128u8, 255]).collect();
    let result = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(8, 8, PixelLayout::GrayAlpha8)
        .encode(&pixels);
    assert!(result.is_ok(), "lossy GrayAlpha8 should encode: {result:?}");
}

#[test]
fn test_lossy_gray16() {
    let pixels_u16: Vec<u16> = (0..8 * 8).map(|_| 32768u16).collect();
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);
    let result = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(8, 8, PixelLayout::Gray16)
        .encode(pixels);
    assert!(result.is_ok(), "lossy Gray16 should encode: {result:?}");
}

#[test]
fn test_lossy_rgba_linear_f32() {
    let pixels_f32: Vec<f32> = (0..8 * 8).flat_map(|_| [0.5f32, 0.3, 0.7, 1.0]).collect();
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_f32);
    let result = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(8, 8, PixelLayout::RgbaLinearF32)
        .encode(pixels);
    assert!(
        result.is_ok(),
        "lossy RgbaLinearF32 should encode: {result:?}"
    );
}

#[test]
fn test_lossy_gray_linear_f32() {
    let pixels_f32: Vec<f32> = (0..8 * 8).map(|_| 0.5f32).collect();
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_f32);
    let result = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(8, 8, PixelLayout::GrayLinearF32)
        .encode(pixels);
    assert!(
        result.is_ok(),
        "lossy GrayLinearF32 should encode: {result:?}"
    );
}

#[test]
fn test_lossless_grayalpha8() {
    let pixels: Vec<u8> = (0..8 * 8).flat_map(|_| [200u8, 255]).collect();
    let result = LosslessConfig::new().encode(&pixels, 8, 8, PixelLayout::GrayAlpha8);
    assert!(
        result.is_ok(),
        "lossless GrayAlpha8 should encode: {result:?}"
    );
}

#[test]
fn test_lossless_grayalpha16() {
    let pixels_u16: Vec<u16> = (0..8 * 8).flat_map(|_| [32768u16, 65535]).collect();
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);
    let result = LosslessConfig::new().encode(pixels, 8, 8, PixelLayout::GrayAlpha16);
    assert!(
        result.is_ok(),
        "lossless GrayAlpha16 should encode: {result:?}"
    );
}

#[test]
fn test_bgra_lossless() {
    // 4x4 red image in BGRA (B=0, G=0, R=255, A=255)
    let pixels = [0u8, 0, 255, 255].repeat(16);
    let result = LosslessConfig::new().encode(&pixels, 4, 4, PixelLayout::Bgra8);
    assert!(result.is_ok());
    let jxl = result.unwrap();
    assert_eq!(&jxl[..2], &[0xFF, 0x0A]);
}

#[test]
fn test_lossy_alpha_encodes() {
    // Lossy+alpha: VarDCT RGB + modular alpha extra channel
    let pixels = [255u8, 0, 0, 255].repeat(64);
    let result =
        LossyConfig::new(2.0)
            .with_gaborish(false)
            .encode(&pixels, 8, 8, PixelLayout::Bgra8);
    assert!(
        result.is_ok(),
        "BGRA lossy encode failed: {:?}",
        result.err()
    );

    let result2 = LossyConfig::new(2.0).encode(&pixels, 8, 8, PixelLayout::Rgba8);
    assert!(
        result2.is_ok(),
        "RGBA lossy encode failed: {:?}",
        result2.err()
    );
}

#[test]
fn test_stop_cancellation() {
    use enough::Unstoppable;
    // Unstoppable should not cancel
    let pixels = vec![128u8; 4 * 4 * 3];
    let cfg = LosslessConfig::new();
    let result = cfg
        .encode_request(4, 4, PixelLayout::Rgb8)
        .with_stop(&Unstoppable)
        .encode(&pixels);
    assert!(result.is_ok());
}

#[test]
fn test_stop_cancels_lossy_multigroup() {
    // A 512x512 image is multi-group (2x2 AC groups + a DC group), so the
    // per-group cancellation checkpoint in the VarDCT entropy phase runs.
    // An always-cancelling Stop must abort the encode with `Cancelled`.
    let (w, h) = (512u32, 512u32);
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
    }

    struct AlwaysCancel;
    impl enough::Stop for AlwaysCancel {
        fn check(&self) -> core::result::Result<(), enough::StopReason> {
            Err(enough::StopReason::Cancelled)
        }
    }

    let cancelled = LossyConfig::new(1.0)
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_stop(&AlwaysCancel)
        .encode(&pixels);
    assert!(
        matches!(&cancelled, Err(e) if matches!(e.error(), EncodeError::Cancelled)),
        "expected EncodeError::Cancelled, got {cancelled:?}"
    );

    // Unstoppable on the same input must still succeed — the checkpoint is
    // a no-op on the success path.
    let ok = LossyConfig::new(1.0)
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_stop(&enough::Unstoppable)
        .encode(&pixels);
    assert!(ok.is_ok(), "Unstoppable lossy encode failed: {ok:?}");
}

/// Issue #77 cross-codec cancellation: the effort-8 butteraugli quantization
/// loop polls `Stop` per iteration, and the VarDCT encode entry polls before
/// any work. An always-cancelling token aborts an e8 encode with `Cancelled`;
/// `Unstoppable` runs every poll (entry + per-buttloop-iteration) and is
/// byte-identical to the no-stop path — proving the buttloop poll is a no-op.
#[test]
fn test_stop_cancels_lossy_e8_buttloop() {
    // 128x128 single-group: small enough to keep the debug-mode e8 butteraugli
    // loop fast, large enough that the loop actually runs (and polls).
    let (w, h) = (128u32, 128u32);
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i.wrapping_mul(2_654_435_761) >> 11) as u8;
    }

    struct AlwaysCancel;
    impl enough::Stop for AlwaysCancel {
        fn check(&self) -> core::result::Result<(), enough::StopReason> {
            Err(enough::StopReason::Cancelled)
        }
    }

    let cancelled = LossyConfig::new(1.0)
        .with_effort(8)
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_stop(&AlwaysCancel)
        .encode(&pixels);
    assert!(
        matches!(&cancelled, Err(e) if matches!(e.error(), EncodeError::Cancelled)),
        "expected EncodeError::Cancelled at e8, got {cancelled:?}"
    );

    // Unstoppable runs the buttloop polls and matches the no-stop output.
    let with_uns = LossyConfig::new(1.0)
        .with_effort(8)
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_stop(&enough::Unstoppable)
        .encode(&pixels)
        .expect("Unstoppable e8 encode should succeed");
    let plain = LossyConfig::new(1.0)
        .with_effort(8)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&pixels)
        .expect("plain e8 encode should succeed");
    assert_eq!(with_uns, plain, "Unstoppable e8 diverged from no-stop path");
}

/// Issue #77 cross-codec cancellation: the modular (lossless) multi-group
/// path polls `Stop` at the encode boundary and before the heavy per-group
/// parallel encode (after tree learning). An always-cancelling token aborts a
/// multi-group lossless encode with `Cancelled`; `Unstoppable` is byte-
/// identical to the no-stop path.
#[test]
fn test_stop_cancels_lossless_multigroup() {
    // 512x512 → multi-group modular; the default effort runs tree learning
    // + per-group parallel encoding (where the polls live).
    let (w, h) = (512u32, 512u32);
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
    }

    struct AlwaysCancel;
    impl enough::Stop for AlwaysCancel {
        fn check(&self) -> core::result::Result<(), enough::StopReason> {
            Err(enough::StopReason::Cancelled)
        }
    }

    let cancelled = LosslessConfig::new()
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_stop(&AlwaysCancel)
        .encode(&pixels);
    assert!(
        matches!(&cancelled, Err(e) if matches!(e.error(), EncodeError::Cancelled)),
        "expected EncodeError::Cancelled for lossless multi-group, got {cancelled:?}"
    );

    let with_uns = LosslessConfig::new()
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_stop(&enough::Unstoppable)
        .encode(&pixels)
        .expect("Unstoppable lossless encode should succeed");
    let plain = LosslessConfig::new()
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&pixels)
        .expect("plain lossless encode should succeed");
    assert_eq!(
        with_uns, plain,
        "Unstoppable lossless multi-group diverged from no-stop path"
    );
}

/// User directive (2026-06-17): the runtime fallible-alloc toggle
/// (`Limits::with_fallible_alloc`) is wired through the STANDARD lossy +
/// lossless encodes, not just JPEG transcode. The dimension-driven output /
/// group-writer / quant / XYB-plane / modular-channel buffers pick `vec!`
/// (fast) vs `try_reserve` (graceful OOM) from the budget policy, so the
/// toggle changes only the allocation *mechanism* — a successful encode is
/// byte-identical in both modes — and both modes still honour the budget.
#[test]
fn test_standard_encode_fallible_alloc_toggle() {
    let (w, h) = (256u32, 256u32);
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
    }

    let inf = Limits::new().with_fallible_alloc(false);
    let fal = Limits::new().with_fallible_alloc(true);

    // Lossy: fallible vs infallible → byte-identical output.
    let lossy_inf = LossyConfig::new(1.0)
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&inf)
        .encode(&pixels)
        .expect("lossy infallible encode");
    let lossy_fal = LossyConfig::new(1.0)
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&fal)
        .encode(&pixels)
        .expect("lossy fallible encode");
    assert_eq!(
        lossy_inf, lossy_fal,
        "lossy fallible toggle changed output bytes"
    );

    // Lossless: fallible vs infallible → byte-identical output (exercises
    // the modular channel-data allocation path).
    let ll_inf = LosslessConfig::new()
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&inf)
        .encode(&pixels)
        .expect("lossless infallible encode");
    let ll_fal = LosslessConfig::new()
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&fal)
        .encode(&pixels)
        .expect("lossless fallible encode");
    assert_eq!(
        ll_inf, ll_fal,
        "lossless fallible toggle changed output bytes"
    );

    // Both modes honour the budget: a 1 KiB cap rejects lossy + lossless
    // either way (the dimension-driven buffers can't fit — the reservation
    // fails before allocation regardless of the fallible policy).
    for fallible in [false, true] {
        let tight = Limits::new()
            .with_fallible_alloc(fallible)
            .with_max_memory_bytes(1024);
        assert!(
            LossyConfig::new(1.0)
                .encode_request(w, h, PixelLayout::Rgb8)
                .with_limits(&tight)
                .encode(&pixels)
                .is_err(),
            "lossy + 1 KiB cap (fallible={fallible}) must reject"
        );
        assert!(
            LosslessConfig::new()
                .encode_request(w, h, PixelLayout::Rgb8)
                .with_limits(&tight)
                .encode(&pixels)
                .is_err(),
            "lossless + 1 KiB cap (fallible={fallible}) must reject"
        );
    }
}

/// Issue #77 item 2: the JPEG-transcode path is cancellable via the
/// `*_with_stop` entry points (the formerly-dead `Stop` plumbing). A
/// cancelling token aborts with `Cancelled`; an `Unstoppable` token is
/// byte-identical to the non-stop path — proving the per-phase polls
/// (decode + per-group + pre-entropy) are harmless no-ops on success.
#[cfg(feature = "jpeg-reencoding")]
#[test]
fn test_jpeg_transcode_cancellation() {
    use enough::{Stop, StopReason, Unstoppable};

    // A committed baseline 4:4:4 JPEG — the transcode path's supported shape.
    let data = include_bytes!("../tests/fixtures/jbrd/base_a_444.jpg");
    let cfg = LosslessConfig::new();

    struct AlwaysCancel;
    impl Stop for AlwaysCancel {
        fn check(&self) -> core::result::Result<(), StopReason> {
            Err(StopReason::Cancelled)
        }
    }

    // A cancelling Stop aborts both transcode entry points with
    // `Cancelled` (mapped from `JpegError::Cancelled` / `Error::Cancelled`).
    let cancelled = cfg.encode_jpeg_transcode_with_stop(data, &AlwaysCancel);
    assert!(
        matches!(&cancelled, Err(e) if matches!(e.error(), EncodeError::Cancelled)),
        "expected EncodeError::Cancelled, got {cancelled:?}"
    );
    let cancelled_cs = cfg.encode_jpeg_transcode_codestream_with_stop(data, &AlwaysCancel);
    assert!(
        matches!(&cancelled_cs, Err(e) if matches!(e.error(), EncodeError::Cancelled)),
        "codestream: expected EncodeError::Cancelled, got {cancelled_cs:?}"
    );

    // Unstoppable runs every poll (decode + per-group + pre-entropy),
    // each returning Ok → byte-identical to the non-stop entry points.
    let with_uns = cfg
        .encode_jpeg_transcode_with_stop(data, &Unstoppable)
        .expect("Unstoppable transcode should succeed");
    let plain = cfg
        .encode_jpeg_transcode(data)
        .expect("plain transcode should succeed");
    assert_eq!(
        with_uns, plain,
        "Unstoppable diverged from the no-stop path"
    );

    let with_uns_cs = cfg
        .encode_jpeg_transcode_codestream_with_stop(data, &Unstoppable)
        .expect("Unstoppable codestream transcode should succeed");
    let plain_cs = cfg
        .encode_jpeg_transcode_codestream(data)
        .expect("plain codestream transcode should succeed");
    assert_eq!(with_uns_cs, plain_cs, "codestream Unstoppable diverged");
}

/// Issue #77 item 1 (full): the JPEG-transcode path is bounded by a
/// per-encode `MemoryBudget` built from `Limits::max_memory_bytes`,
/// threaded through the decode coefficient buffers AND the encode working
/// set. A tight cap is rejected with `LimitExceeded`; `u64::MAX` opts out.
#[cfg(feature = "jpeg-reencoding")]
#[test]
fn test_jpeg_transcode_memory_budget() {
    let data = include_bytes!("../tests/fixtures/jbrd/base_a_444.jpg");

    // Default cap (8 GiB lossless default) transcodes the small fixture.
    assert!(
        LosslessConfig::new().encode_jpeg_transcode(data).is_ok(),
        "default-budget transcode should succeed"
    );

    // A 1 KiB cap can't fit the coefficient buffers / encode working set —
    // rejected with LimitExceeded (proves the budget is threaded through
    // both the decode and encode phases).
    let tight = Limits::new().with_max_memory_bytes(1024);
    let r = LosslessConfig::new()
        .with_limits(&tight)
        .encode_jpeg_transcode(data);
    assert!(
        matches!(&r, Err(e) if matches!(e.error(), EncodeError::LimitExceeded { .. })),
        "expected LimitExceeded under a 1 KiB cap, got {r:?}"
    );
    let r_cs = LosslessConfig::new()
        .with_limits(&tight)
        .encode_jpeg_transcode_codestream(data);
    assert!(
        matches!(&r_cs, Err(e) if matches!(e.error(), EncodeError::LimitExceeded { .. })),
        "codestream: expected LimitExceeded under a 1 KiB cap, got {r_cs:?}"
    );

    // `u64::MAX` opts out of the cap → succeeds.
    let unbounded = Limits::new().with_max_memory_bytes(u64::MAX);
    assert!(
        LosslessConfig::new()
            .with_limits(&unbounded)
            .encode_jpeg_transcode(data)
            .is_ok(),
        "u64::MAX cap should transcode fine"
    );
}

/// User directive (2026-06-17): fallible-vs-infallible allocation is a
/// **runtime** toggle (`Limits::with_fallible_alloc`) — `vec!` (fast,
/// calloc) vs `try_reserve` (graceful OOM). The toggle changes the
/// allocation *mechanism*, not the bytes, so a successful transcode is
/// byte-identical in both modes; the difference only surfaces as a clean
/// error (vs abort) on a genuine OOM, which can't be provoked without
/// exhausting memory. Both modes still honour the `MemoryBudget`.
#[cfg(feature = "jpeg-reencoding")]
#[test]
fn test_jpeg_transcode_fallible_alloc_toggle() {
    let data = include_bytes!("../tests/fixtures/jbrd/base_a_444.jpg");

    let infallible = Limits::new().with_fallible_alloc(false);
    let fallible = Limits::new().with_fallible_alloc(true);

    let out_infallible = LosslessConfig::new()
        .with_limits(&infallible)
        .encode_jpeg_transcode(data)
        .expect("infallible-alloc transcode should succeed");
    let out_fallible = LosslessConfig::new()
        .with_limits(&fallible)
        .encode_jpeg_transcode(data)
        .expect("fallible-alloc transcode should succeed");
    assert_eq!(
        out_infallible, out_fallible,
        "fallible toggle changed output bytes (must be alloc-mechanism only)"
    );

    // Both modes still honour the budget: a 1 KiB cap rejects either way
    // (the byte reservation fails before the allocation in both modes).
    let tight_inf = Limits::new()
        .with_fallible_alloc(false)
        .with_max_memory_bytes(1024);
    let tight_fal = Limits::new()
        .with_fallible_alloc(true)
        .with_max_memory_bytes(1024);
    assert!(
        LosslessConfig::new()
            .with_limits(&tight_inf)
            .encode_jpeg_transcode(data)
            .is_err(),
        "infallible + 1 KiB cap must reject"
    );
    assert!(
        LosslessConfig::new()
            .with_limits(&tight_fal)
            .encode_jpeg_transcode(data)
            .is_err(),
        "fallible + 1 KiB cap must reject"
    );
}

/// Issue #77 follow-ups: the PreserveJxl `encode_jpeg_recompress_*` free
/// functions honour `Limits` (memory budget + pixel cap) and `Stop`.
#[cfg(feature = "jpeg-reencoding")]
#[test]
fn test_jpeg_recompress_limits_and_stop() {
    use enough::{Stop, StopReason, Unstoppable};

    let data = include_bytes!("../tests/fixtures/jbrd/base_a_444.jpg");

    // Default (no limits / unstoppable): the lossless floor succeeds.
    assert!(
        crate::jpeg::encode_jpeg_recompress_auto_codestream(data, 1.0, 7, None, None).is_ok(),
        "default recompress should succeed"
    );

    // A tight memory cap rejects.
    let tight = Limits::new().with_max_memory_bytes(1024);
    let r = crate::jpeg::encode_jpeg_recompress_auto_codestream(data, 2.0, 7, Some(&tight), None);
    assert!(
        r.is_err(),
        "expected a memory-limit rejection under a 1 KiB cap"
    );

    // A cancelling Stop aborts with `Error::Cancelled`.
    struct AlwaysCancel;
    impl Stop for AlwaysCancel {
        fn check(&self) -> core::result::Result<(), StopReason> {
            Err(StopReason::Cancelled)
        }
    }
    let c = crate::jpeg::encode_jpeg_recompress_auto_codestream(
        data,
        2.0,
        7,
        None,
        Some(&AlwaysCancel),
    );
    assert!(
        matches!(c, Err(crate::error::Error::Cancelled)),
        "expected Error::Cancelled, got {c:?}"
    );

    // Unstoppable + default cap succeeds.
    assert!(
        crate::jpeg::encode_jpeg_recompress_auto_codestream(data, 2.0, 7, None, Some(&Unstoppable))
            .is_ok(),
        "Unstoppable recompress should succeed"
    );
}

#[test]
fn test_lossy_palette_encode() {
    // 16x16 RGB image with 4 colors + slight noise
    let colors = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
    let mut pixels = Vec::with_capacity(16 * 16 * 3);
    for y in 0..16u8 {
        for x in 0..16u8 {
            let ci = ((y / 4) * 4 + x / 4) as usize % 4;
            let noise = ((x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13))) % 5) as i16 - 2;
            for &channel in &colors[ci][..3] {
                let v = (channel as i16 + noise).clamp(0, 255) as u8;
                pixels.push(v);
            }
        }
    }
    let cfg = LosslessConfig::new()
        .with_lossy_palette(true)
        .with_ans(true);
    let result = cfg.encode(&pixels, 16, 16, PixelLayout::Rgb8);
    assert!(
        result.is_ok(),
        "lossy palette encode failed: {:?}",
        result.err()
    );
    let jxl = result.unwrap();
    assert_eq!(&jxl[..2], &[0xFF, 0x0A], "JXL signature");

    // Verify jxl-oxide can parse and decode it
    let cursor = std::io::Cursor::new(&jxl);
    let reader = std::io::BufReader::new(cursor);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide parse");
    assert!(
        image.width() > 0,
        "decoded image should have non-zero width"
    );
}

#[test]
fn test_lossy_palette_multi_group() {
    // 300x300 RGB image with ~20 dominant colors + noise (>256x256 = multi-group)
    let colors = [
        [255u8, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 128, 128],
        [64, 64, 64],
    ];
    let mut pixels = Vec::with_capacity(300 * 300 * 3);
    for y in 0..300u32 {
        for x in 0..300u32 {
            let ci = ((y / 40) * 8 + x / 40) as usize % colors.len();
            let noise = ((x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13))) % 7) as i16 - 3;
            for &channel in &colors[ci][..3] {
                let v = (channel as i16 + noise).clamp(0, 255) as u8;
                pixels.push(v);
            }
        }
    }

    // Encode with lossy palette + ANS (multi-group)
    let cfg = LosslessConfig::new()
        .with_lossy_palette(true)
        .with_ans(true);
    let jxl = cfg
        .encode(&pixels, 300, 300, PixelLayout::Rgb8)
        .expect("lossy palette multi-group encode");
    assert_eq!(&jxl[..2], &[0xFF, 0x0A], "JXL signature");
    assert!(jxl.len() < 300 * 300 * 3, "should compress");

    // Save to disk for inspection
    let out = crate::test_helpers::output_dir("lossy_palette");
    let jxl_out = out.join("lossy_palette_multi.jxl");
    let png_out = out.join("lossy_palette_multi.png");
    std::fs::write(&jxl_out, &jxl).ok();
    eprintln!(
        "LOSSY_PALETTE_MULTI test: encoded {} bytes ({}x{})",
        jxl.len(),
        300,
        300
    );

    // Try djxl decode first for better error messages
    let djxl_result = std::process::Command::new("djxl")
        .args([jxl_out.to_str().unwrap(), png_out.to_str().unwrap()])
        .output();
    if let Ok(output) = djxl_result {
        eprintln!(
            "djxl: status={}, stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Verify jxl-rs can decode it
    let decoded = crate::test_helpers::decode_with_jxl_rs(&jxl).expect("jxl-rs decode failed");
    assert_eq!(decoded.width, 300);
    assert_eq!(decoded.height, 300);
    assert_eq!(decoded.channels, 3);

    // Verify lossy quality: each pixel should be within 50 of original (delta palette error)
    // decoded.pixels is f32 in [0.0, 1.0] — convert to u8 for comparison
    let mut max_error = 0i32;
    let mut error_pos = (0, 0, 0);
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
        let dec_u8 = (dec * 255.0).round().clamp(0.0, 255.0) as u8;
        let diff = (orig as i32 - dec_u8 as i32).abs();
        if diff > max_error {
            max_error = diff;
            let pixel = i / 3;
            error_pos = (pixel % 300, pixel / 300, i % 3);
        }
    }
    let err_idx = error_pos.1 * 300 * 3 + error_pos.0 * 3 + error_pos.2;
    let dec_u8 = (decoded.pixels[err_idx] * 255.0).round().clamp(0.0, 255.0) as u8;
    eprintln!(
        "max_error={} at ({},{}) ch={}, orig={} decoded={}",
        max_error, error_pos.0, error_pos.1, error_pos.2, pixels[err_idx], dec_u8,
    );
    assert!(
        max_error <= 80,
        "lossy palette max error {} too large (expected <= 80)",
        max_error
    );
}

#[test]
fn test_palette_256_colors_regression() {
    // Regression test for palette+ANS checksum mismatch with many unique colors.
    // Root cause was u2S bit width bug in write_palette_transform (fixed Feb 17, 2026):
    // nb_colors selectors 1-2 used 11/14 bits instead of 10/12 bits. Triggered when
    // nb_colors >= 256 (selector 1). Two test cases:
    //
    // 1. 32x32 with 256 unique colors via standard API (passes 50% heuristic)
    // 2. 16x16 with 256 unique colors via internal API (bypasses heuristic)
    use crate::modular::channel::{Channel, ModularImage};
    use crate::modular::encode::write_modular_stream_with_palette;

    // Test 1: 32x32 through standard API (256 colors, each used 4x)
    let mut pixels = Vec::with_capacity(32 * 32 * 3);
    for i in 0..1024u32 {
        let idx = (i / 4) as u8;
        pixels.push(idx);
        pixels.push(((idx as u32 * 7 + 13) & 0xFF) as u8);
        pixels.push(((idx as u32 * 31 + 97) & 0xFF) as u8);
    }
    let cfg = LosslessConfig::new().with_ans(true);
    let jxl = cfg
        .encode(&pixels, 32, 32, PixelLayout::Rgb8)
        .expect("palette 256-colors encode");
    let decoded = crate::test_helpers::decode_with_jxl_rs(&jxl).expect("jxl-rs decode failed");
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
        let dec_u8 = (dec * 255.0).round().clamp(0.0, 255.0) as u8;
        assert_eq!(
            orig, dec_u8,
            "32x32: mismatch at byte {}: orig={} decoded={}",
            i, orig, dec_u8
        );
    }

    // Test 2: 16x16 via internal API (bypasses 50% heuristic)
    let mut channels = Vec::new();
    for c in 0..3 {
        let mut ch = Channel::new(16, 16).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let idx = y * 16 + x;
                let val = match c {
                    0 => idx as i32,
                    1 => ((idx * 3 + 17) & 0xFF) as i32,
                    2 => (255 - idx) as i32,
                    _ => 0,
                };
                ch.set(x, y, val);
            }
        }
        channels.push(ch);
    }
    let image = ModularImage {
        channels,
        bit_depth: 8,
        is_grayscale: false,
        has_alpha: false,
    };
    let mut writer = crate::bit_writer::BitWriter::new();
    write_modular_stream_with_palette(&image, &mut writer, true, 0, 3)
        .expect("palette encode with 256 unique colors must not fail");
}

#[test]
fn test_16bit_tree_learning() {
    // Test multiple 16-bit scenarios that previously failed
    for &(w, h, layout, label) in &[
        (32u32, 32u32, PixelLayout::Rgb16, "32x32 RGB16"),
        (8, 8, PixelLayout::Rgba16, "8x8 RGBA16"),
        (8, 8, PixelLayout::Rgb16, "8x8 RGB16"),
        (16, 16, PixelLayout::Gray16, "16x16 Gray16"),
    ] {
        let nc = layout.bytes_per_pixel()
            / if layout.is_16bit() {
                2
            } else if layout.is_f32() {
                4
            } else {
                1
            };
        let mut pixels = vec![0u16; (w * h) as usize * nc];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) as usize) * nc;
                pixels[idx] = (x * 2048) as u16;
                if nc >= 2 {
                    pixels[idx + 1] = (y * 2048) as u16;
                }
                if nc >= 3 {
                    pixels[idx + 2] = ((x + y) * 1024) as u16;
                }
                if nc >= 4 {
                    pixels[idx + 3] = 65535; // opaque alpha
                }
            }
        }
        let bytes: Vec<u8> = pixels.iter().flat_map(|v| v.to_ne_bytes()).collect();

        let cfg = LosslessConfig::new().with_effort(7).with_ans(true);
        let jxl = cfg
            .encode(&bytes, w, h, layout)
            .unwrap_or_else(|e| panic!("{}: encode failed: {}", label, e));

        let decoded = crate::test_helpers::decode_with_jxl_rs(&jxl)
            .unwrap_or_else(|e| panic!("{}: jxl-rs decode failed: {}", label, e));
        assert_eq!(decoded.width, w as usize, "{}: width", label);
        assert_eq!(decoded.height, h as usize, "{}: height", label);

        let scale = 65535.0;
        let mut mismatches = 0;
        for (i, (&orig, &dec_f)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
            let dec = (dec_f * scale).round().clamp(0.0, scale) as u16;
            if orig != dec && mismatches < 3 {
                eprintln!("{}: mismatch[{}]: orig={} dec={}", label, i, orig, dec);
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "{}: {} mismatches", label, mismatches);
        eprintln!("{}: PASS ({} bytes)", label, jxl.len());
    }
}

#[test]
fn test_srgb_lut_matches_powf() {
    for i in 0u16..256 {
        let lut_val = SRGB_U8_TO_LINEAR[i as usize];
        let fast_val = srgb_to_linear_f(i as f32 / 255.0);
        let diff = (lut_val - fast_val).abs();
        // LUT uses f64 exact powf, srgb_to_linear_f uses fast_powf (~3e-5 relative error)
        let tol = fast_val.abs() * 5e-5 + 1e-7;
        assert!(
            diff <= tol,
            "sRGB LUT mismatch at {i}: LUT={lut_val}, fast={fast_val}, diff={diff}"
        );
    }
}

#[test]
fn test_quality_to_distance_f32_mapping() {
    // Verify the piecewise mapping at key points.
    assert_eq!(quality_to_distance(100.0), 0.0);
    assert_eq!(quality_to_distance(90.0), 1.0); // visually lossless
    assert_eq!(quality_to_distance(80.0), 1.5);
    assert_eq!(quality_to_distance(70.0), 2.0);
    assert_eq!(quality_to_distance(50.0), 4.0);
    assert_eq!(quality_to_distance(0.0), 9.0);
    // Clamped above 100
    assert_eq!(quality_to_distance(110.0), 0.0);
}

#[test]
fn test_calibrated_jxl_quality() {
    // Boundary: below table minimum clamps to first entry's output.
    assert_eq!(calibrated_jxl_quality(0.0), 5.0);
    // Boundary: above table maximum clamps to last entry's output.
    assert_eq!(calibrated_jxl_quality(100.0), 93.8);
    // Exact table entry.
    assert_eq!(calibrated_jxl_quality(90.0), 84.2);
    // Interpolated mid-point between (50, 48.5) and (55, 51.9).
    let mid = calibrated_jxl_quality(52.5);
    let expected = 48.5 + 0.5 * (51.9 - 48.5);
    assert!(
        (mid - expected).abs() < 0.01,
        "expected {expected}, got {mid}"
    );
}

#[test]
fn test_interp_quality_edge_cases() {
    let table = &[(10.0f32, 20.0f32), (20.0, 40.0), (30.0, 60.0)];
    // Below table
    assert_eq!(interp_quality(table, 5.0), 20.0);
    // Above table
    assert_eq!(interp_quality(table, 35.0), 60.0);
    // Exact match
    assert_eq!(interp_quality(table, 20.0), 40.0);
    // Midpoint
    assert!((interp_quality(table, 15.0) - 30.0).abs() < 0.001);
}

// -----------------------------------------------------------------
// Internal-params override (__expert) — segmented Lossy / Lossless
// -----------------------------------------------------------------

#[cfg(feature = "__expert")]
mod internal_params {
    use super::*;
    use crate::effort::{LosslessInternalParams, LossyInternalParams};

    // Pseudo-random RGB image — large enough + complex enough to exercise
    // RCT search, WP, and tree-learning splits so different param
    // settings produce different bitstreams.
    fn pseudo_random_rgb8(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity((w * h * 3) as usize);
        let mut state: u32 = 0xDEAD_BEEF;
        for _ in 0..(w * h) {
            let r = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state = r;
            let g = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state = g;
            let b = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state = b;
            out.push((r >> 24) as u8);
            out.push((g >> 24) as u8);
            out.push((b >> 24) as u8);
        }
        out
    }

    #[test]
    fn lossless_internal_params_changes_bitstream() {
        // Tighten tree learning + skip RCT search to push bytes off the
        // e7 default.
        let params = LosslessInternalParams {
            tree_max_buckets: Some(16),
            tree_num_properties: Some(3),
            nb_rcts_to_try: Some(0),
            ..Default::default()
        };

        let cfg_override = LosslessConfig::new()
            .with_effort(7)
            .with_internal_params(params)
            .with_threads(1);
        let cfg_default = LosslessConfig::new().with_effort(7).with_threads(1);

        let pixels = pseudo_random_rgb8(64, 64);
        let bytes_a = cfg_override
            .encode(&pixels, 64, 64, PixelLayout::Rgb8)
            .expect("override encode");
        let bytes_b = cfg_default
            .encode(&pixels, 64, 64, PixelLayout::Rgb8)
            .expect("default encode");

        assert_eq!(&bytes_a[..2], &crate::JXL_SIGNATURE);
        assert_eq!(&bytes_b[..2], &crate::JXL_SIGNATURE);
        assert_ne!(
            bytes_a, bytes_b,
            "internal_params override should produce different bitstream"
        );
    }

    #[test]
    fn lossy_internal_params_changes_bitstream() {
        let mut entropy = crate::effort::EntropyMulTable::reference();
        entropy.dct8 = 0.95;
        let params = LossyInternalParams {
            try_dct16: Some(false),
            try_dct32: Some(false),
            try_dct64: Some(false),
            try_dct4x8_afv: Some(false),
            k_info_loss_mul_base: Some(1.5),
            entropy_mul_table: Some(entropy),
            ..Default::default()
        };

        let cfg_override = LossyConfig::new(2.0)
            .with_effort(7)
            .with_internal_params(params)
            .with_threads(1);
        let cfg_default = LossyConfig::new(2.0).with_effort(7).with_threads(1);

        let pixels = pseudo_random_rgb8(64, 64);
        let bytes_a = cfg_override
            .encode(&pixels, 64, 64, PixelLayout::Rgb8)
            .expect("override encode");
        let bytes_b = cfg_default
            .encode(&pixels, 64, 64, PixelLayout::Rgb8)
            .expect("default encode");

        assert_eq!(&bytes_a[..2], &crate::JXL_SIGNATURE);
        assert_eq!(&bytes_b[..2], &crate::JXL_SIGNATURE);
        assert_ne!(
            bytes_a, bytes_b,
            "internal_params override should produce different bitstream"
        );
    }

    #[test]
    fn lossless_internal_params_persist_across_with_effort() {
        // Issue #80: `with_internal_params` is now order-independent vs
        // `with_effort`. The override is stored sparsely and resolved
        // lazily against the FINAL effort, so it (a) takes effect and
        // (b) is byte-identical regardless of whether it was applied
        // before or after `with_effort`. `use_ans: Some(false)` forces
        // Huffman — a guaranteed-visible delta vs the ANS default.
        //
        // (Pre-#80 this test pinned `tree_max_buckets: Some(16)` and
        // passed only *because of* the order bug: the eager
        // `profile_override` captured effort=7 set before
        // `with_effort(9)`, so the override config encoded at e7 ≠
        // e9-plain. With the bug fixed it correctly resolves at e9,
        // where 16 buckets happen not to change this 64×64 encode — so
        // the assertion is now order-independence + a visible knob.)
        let params = LosslessInternalParams {
            nb_rcts_to_try: Some(1),
            ..Default::default()
        };
        let before = LosslessConfig::new()
            .with_internal_params(params.clone())
            .with_effort(9)
            .with_threads(1);
        let after = LosslessConfig::new()
            .with_effort(9)
            .with_internal_params(params.clone())
            .with_threads(1);
        let plain = LosslessConfig::new().with_effort(9).with_threads(1);

        // (1) The override reaches the resolved profile regardless of
        //     builder order (the #80 order-independence invariant).
        assert_eq!(before.effective_profile().nb_rcts_to_try, 1);
        assert_eq!(after.effective_profile().nb_rcts_to_try, 1);
        // (2) ...and it actually overrides the e9 schedule default.
        assert_ne!(plain.effective_profile().nb_rcts_to_try, 1);
        // (3) ...and the encode is byte-identical regardless of order.
        let pixels = pseudo_random_rgb8(64, 64);
        let enc = |cfg: &LosslessConfig| {
            cfg.clone()
                .encode(&pixels, 64, 64, PixelLayout::Rgb8)
                .expect("encode")
        };
        assert_eq!(
            enc(&before),
            enc(&after),
            "with_internal_params is order-independent vs with_effort (#80)"
        );
    }
}

// ─── Shared knob enums (libjxl `cjxl` parity) ─────────────────

#[test]
fn test_container_mode_default_auto() {
    assert_eq!(LossyConfig::new(1.0).container_mode(), ContainerMode::Auto);
    assert_eq!(LosslessConfig::new().container_mode(), ContainerMode::Auto);
}

#[test]
fn test_container_mode_round_trip() {
    let cfg = LossyConfig::new(1.0).with_container_mode(ContainerMode::Always);
    assert_eq!(cfg.container_mode(), ContainerMode::Always);
    let cfg = cfg.with_container_mode(ContainerMode::Never);
    assert_eq!(cfg.container_mode(), ContainerMode::Never);
}

#[test]
fn test_faster_decoding_clamp() {
    // Out-of-range values clamp to MAX_FASTER_DECODING.
    let cfg = LossyConfig::new(1.0).with_faster_decoding(99);
    assert_eq!(cfg.faster_decoding(), MAX_FASTER_DECODING);
    // 0 is the default (no speed bias).
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.faster_decoding(), 0);
    // In-range values pass through.
    for tier in 0..=MAX_FASTER_DECODING {
        assert_eq!(
            LossyConfig::new(1.0)
                .with_faster_decoding(tier)
                .faster_decoding(),
            tier,
        );
        assert_eq!(
            LosslessConfig::new()
                .with_faster_decoding(tier)
                .faster_decoding(),
            tier,
        );
    }
}

#[test]
fn test_faster_decoding_lossless_effective_getters() {
    // Tier 0: all getters return the stored field values.
    let cfg = LosslessConfig::new();
    let stored_lz77 = cfg.lz77();
    let stored_tree = cfg.tree_learning();
    let stored_patches = cfg.patches();
    assert_eq!(cfg.effective_lz77(), stored_lz77);
    assert_eq!(cfg.effective_tree_learning(), stored_tree);
    assert_eq!(cfg.effective_patches(), stored_patches);
    assert_eq!(cfg.effective_modular_group_size_shift(), None);

    // Tier 1: LZ77 off. Tree-learning + patches unchanged.
    let cfg = LosslessConfig::new().with_faster_decoding(1);
    assert!(!cfg.effective_lz77(), "tier 1 disables LZ77");
    assert_eq!(cfg.effective_tree_learning(), stored_tree);
    assert_eq!(cfg.effective_patches(), stored_patches);
    assert_eq!(cfg.effective_modular_group_size_shift(), None);

    // Tier 2: + group_size_shift = 0 + patches off.
    let cfg = LosslessConfig::new().with_faster_decoding(2);
    assert!(!cfg.effective_lz77());
    assert_eq!(cfg.effective_tree_learning(), stored_tree);
    assert!(!cfg.effective_patches(), "tier 2 disables patches");
    assert_eq!(cfg.effective_modular_group_size_shift(), Some(0));

    // Tier 4: + tree_learning off.
    let cfg = LosslessConfig::new().with_faster_decoding(4);
    assert!(!cfg.effective_lz77());
    assert!(
        !cfg.effective_tree_learning(),
        "tier 4 disables tree learning"
    );
    assert!(!cfg.effective_patches());
    assert_eq!(cfg.effective_modular_group_size_shift(), Some(0));

    // Explicit `with_modular_group_size` overrides the tier-2 default.
    let cfg = LosslessConfig::new()
        .with_faster_decoding(2)
        .with_modular_group_size(Some(2));
    assert_eq!(
        cfg.effective_modular_group_size_shift(),
        Some(2),
        "explicit modular_group_size wins over tier-2 default"
    );
}

#[test]
fn test_faster_decoding_lossy_effective_getters() {
    // Tier 0: getters return stored field values.
    let cfg = LossyConfig::new(1.0);
    let stored_lz77 = cfg.lz77();
    let stored_patches = cfg.patches();
    let stored_gab = cfg.gaborish();
    assert_eq!(cfg.effective_lz77(), stored_lz77);
    assert_eq!(cfg.effective_patches(), stored_patches);
    assert_eq!(cfg.effective_gaborish(), stored_gab);

    // Tier 1: LZ77 off.
    let cfg = LossyConfig::new(1.0).with_faster_decoding(1);
    assert!(!cfg.effective_lz77());
    assert_eq!(cfg.effective_patches(), stored_patches);
    assert_eq!(cfg.effective_gaborish(), stored_gab);

    // Tier 2: + patches off.
    let cfg = LossyConfig::new(1.0).with_faster_decoding(2);
    assert!(!cfg.effective_lz77());
    assert!(!cfg.effective_patches());
    assert_eq!(cfg.effective_gaborish(), stored_gab);

    // Tier 4: + gaborish forced off.
    let cfg = LossyConfig::new(1.0).with_faster_decoding(4);
    assert!(!cfg.effective_lz77());
    assert!(!cfg.effective_patches());
    assert!(!cfg.effective_gaborish(), "tier 4 disables gaborish");
}

#[test]
fn test_faster_decoding_lossless_roundtrip_levels_0_2_4() {
    // Encode a deterministic synthetic RGB image at faster_decoding
    // levels 0, 2, 4 and verify:
    //   (a) every level produces a valid jxl-rs roundtrip,
    //   (b) bytes grow monotonically as the tier rises (libjxl
    //       semantics — higher tier = simpler bitstream = larger
    //       file at the same effort).
    const W: u32 = 96;
    const H: u32 = 96;
    let mut pixels = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            // Mix of smooth gradients and high-frequency content so the
            // tier-1/2/4 disables (LZ77, group_size, tree learning) all
            // see something interesting to bias on. Pure noise would
            // make tier-X bytes uncomfortably close to incompressible.
            let r = ((x.wrapping_mul(3) ^ y.wrapping_mul(5)) & 0xFF) as u8;
            let g = ((x + y) & 0xFF) as u8;
            let b = ((x.wrapping_mul(17) ^ (y.wrapping_mul(11))) & 0xFF) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    let encode = |tier: u8| -> Vec<u8> {
        LosslessConfig::new()
            .with_effort(7)
            .with_faster_decoding(tier)
            .encode(&pixels, W, H, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("encode tier={} failed: {:?}", tier, e))
    };

    let bytes0 = encode(0);
    let bytes2 = encode(2);
    let bytes4 = encode(4);

    // (a) all three roundtrip via jxl-rs and reproduce input bit-exact.
    for (tier, bytes) in [(0, &bytes0), (2, &bytes2), (4, &bytes4)] {
        let decoded = crate::test_helpers::decode_with_jxl_rs(bytes)
            .unwrap_or_else(|e| panic!("jxl-rs decode tier={} failed: {:?}", tier, e));
        assert_eq!(decoded.width, W as usize, "tier {} width", tier);
        assert_eq!(decoded.height, H as usize, "tier {} height", tier);
        assert_eq!(decoded.channels, 3, "tier {} channels", tier);
        // Lossless: pixels must match exactly.
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
            let dec_u8 = (dec * 255.0).round().clamp(0.0, 255.0) as u8;
            assert_eq!(
                orig, dec_u8,
                "tier {}: pixel mismatch at byte {}: orig={} decoded={}",
                tier, i, orig, dec_u8,
            );
        }
    }

    // (b) bytes grow with tier. Higher tier = simpler bitstream =
    // larger file (the decode-speed tradeoff).
    eprintln!(
        "faster_decoding lossless bytes: t0={} t2={} t4={}",
        bytes0.len(),
        bytes2.len(),
        bytes4.len(),
    );
    assert!(
        bytes2.len() >= bytes0.len(),
        "tier 2 ({} B) should be >= tier 0 ({} B)",
        bytes2.len(),
        bytes0.len()
    );
    assert!(
        bytes4.len() >= bytes2.len(),
        "tier 4 ({} B) should be >= tier 2 ({} B)",
        bytes4.len(),
        bytes2.len()
    );
}

#[test]
fn test_faster_decoding_lossy_roundtrip_levels_0_2_4() {
    // Lossy analog: encode at d=1.0, e7 with faster_decoding 0/2/4
    // and verify jxl-rs decodes each + records the byte counts.
    // We do NOT assert byte monotonicity on lossy — quality drift
    // from disabling gaborish (tier 4) can occasionally produce
    // smaller files via different AC strategy selection. The hard
    // requirement is "all tiers decode".
    const W: u32 = 96;
    const H: u32 = 96;
    let mut pixels = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            let r = ((x.wrapping_mul(3) ^ y.wrapping_mul(5)) & 0xFF) as u8;
            let g = ((x + y) & 0xFF) as u8;
            let b = ((x.wrapping_mul(17) ^ (y.wrapping_mul(11))) & 0xFF) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    let encode = |tier: u8| -> Vec<u8> {
        LossyConfig::new(1.0)
            .with_effort(7)
            .with_faster_decoding(tier)
            .encode(&pixels, W, H, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("lossy encode tier={} failed: {:?}", tier, e))
    };

    let bytes0 = encode(0);
    let bytes2 = encode(2);
    let bytes4 = encode(4);

    eprintln!(
        "faster_decoding lossy bytes: t0={} t2={} t4={}",
        bytes0.len(),
        bytes2.len(),
        bytes4.len(),
    );

    for (tier, bytes) in [(0, &bytes0), (2, &bytes2), (4, &bytes4)] {
        let decoded = crate::test_helpers::decode_with_jxl_rs(bytes)
            .unwrap_or_else(|e| panic!("jxl-rs decode tier={} failed: {:?}", tier, e));
        assert_eq!(decoded.width, W as usize, "tier {} width", tier);
        assert_eq!(decoded.height, H as usize, "tier {} height", tier);
    }
}

#[test]
fn test_faster_decoding_profile_apply() {
    use crate::effort::EffortProfile;

    let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
    let base_lz77 = p.lz77;
    let base_custom_orders = p.custom_orders;
    let base_enhanced = p.enhanced_clustering_vardct;
    let base_gaborish = p.gaborish;
    let base_try_dct32 = p.try_dct32;
    let base_tree_learning = p.tree_learning;
    let base_patches = p.patches;
    let base_threshold = p.tree_threshold_base;

    // Tier 0: no-op.
    let mut p0 = p.clone();
    p0.apply_faster_decoding(0);
    assert_eq!(p0.lz77, base_lz77);
    assert_eq!(p0.custom_orders, base_custom_orders);

    // Tier 1.
    let mut p1 = p.clone();
    p1.apply_faster_decoding(1);
    assert!(!p1.lz77);
    assert_eq!(p1.enhanced_clustering_vardct, base_enhanced);
    assert_eq!(p1.custom_orders, base_custom_orders);

    // Tier 2.
    let mut p2 = p.clone();
    p2.apply_faster_decoding(2);
    assert!(!p2.lz77);
    assert!(!p2.enhanced_clustering_vardct);
    assert_eq!(p2.custom_orders, base_custom_orders);

    // Tier 3: + custom_orders off + threshold raised.
    let mut p3 = p.clone();
    p3.apply_faster_decoding(3);
    assert!(!p3.custom_orders);
    assert!(p3.tree_threshold_base > base_threshold);

    // Tier 4: + tree_learning off + patches off + gaborish off + no DCT32.
    p.apply_faster_decoding(4);
    assert!(!p.tree_learning);
    assert!(!p.patches);
    assert!(!p.gaborish);
    assert!(!p.try_dct32);
    assert!(!p.try_dct64);
    // Sanity: base values were on at effort 7.
    let _ = (
        base_gaborish,
        base_try_dct32,
        base_tree_learning,
        base_patches,
    );
}

#[test]
fn test_progressive_dc_clamp_and_lf_frame_implication() {
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.progressive_dc(), 0);
    assert!(!cfg.lf_frame());

    // level 1 implies lf_frame=true.
    let cfg = LossyConfig::new(1.0).with_progressive_dc(1);
    assert_eq!(cfg.progressive_dc(), 1);
    assert!(cfg.lf_frame(), "progressive_dc>=1 should imply lf_frame");

    // level 2 also implies lf_frame=true.
    let cfg = LossyConfig::new(1.0).with_progressive_dc(2);
    assert_eq!(cfg.progressive_dc(), 2);
    assert!(cfg.lf_frame());

    // Out-of-range clamps to MAX_PROGRESSIVE_DC.
    let cfg = LossyConfig::new(1.0).with_progressive_dc(255);
    assert_eq!(cfg.progressive_dc(), MAX_PROGRESSIVE_DC);
}

#[test]
fn test_premultiplied_alpha_mode_from_i8() {
    assert_eq!(
        PremultipliedAlphaMode::from_i8(-1),
        PremultipliedAlphaMode::Auto
    );
    assert_eq!(
        PremultipliedAlphaMode::from_i8(-127),
        PremultipliedAlphaMode::Auto
    );
    assert_eq!(
        PremultipliedAlphaMode::from_i8(0),
        PremultipliedAlphaMode::Off
    );
    assert_eq!(
        PremultipliedAlphaMode::from_i8(1),
        PremultipliedAlphaMode::On
    );
    assert_eq!(
        PremultipliedAlphaMode::from_i8(127),
        PremultipliedAlphaMode::On
    );
}

#[test]
fn test_premultiplied_alpha_mode_builder_round_trip() {
    let cfg = LossyConfig::new(1.0);
    {
        let req = cfg.encode_request(8, 8, PixelLayout::Rgba8);
        // Default: Off (matches the boolean `false` default).
        assert_eq!(req.premultiplied_alpha_mode(), PremultipliedAlphaMode::Off);
    }

    {
        let req = cfg
            .encode_request(8, 8, PixelLayout::Rgba8)
            .with_premultiplied_alpha_mode(PremultipliedAlphaMode::On);
        assert_eq!(req.premultiplied_alpha_mode(), PremultipliedAlphaMode::On);
    }

    {
        let req = cfg
            .encode_request(8, 8, PixelLayout::Rgba8)
            .with_premultiplied_alpha_mode(PremultipliedAlphaMode::Auto);
        assert_eq!(req.premultiplied_alpha_mode(), PremultipliedAlphaMode::Auto);
    }
}

// ─── EncoderStrategy (W44-127 Chunk A) ─────────────────────────────

/// Default is `Zenjxl` — production shipping behaviour.
/// See `docs/COMPATIBILITY_MODES.md` §4.1 + §7 Q1.
#[test]
fn test_encoder_strategy_default_is_zenjxl() {
    assert_eq!(EncoderStrategy::default(), EncoderStrategy::Zenjxl);
}

/// `EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default())`
/// returns the strict-libjxl-parity bundle: every Section B policy
/// is at the disabled / ForceAllow / ForceSkip variant, every
/// Section A `EffortGate` is at `EffortGate::Libjxl`,
/// `block_ctx_map_15_cluster == true`, and perf dispatches are at
/// their `Default`.
#[test]
fn test_resolve_libjxl_field_values() {
    let resolved = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());

    // Section B content-aware gates: all disabled / force-allow / force-skip
    assert_eq!(
        resolved.screenshot_entropy_mul,
        ScreenshotEntropyMulPolicy::Disabled
    );
    assert_eq!(
        resolved.high_d_photo_entropy_mul,
        HighDPhotoEntropyMulPolicy::Disabled
    );
    assert_eq!(resolved.dct64_search_policy, Dct64SearchPolicy::ForceAllow);
    assert_eq!(
        resolved.dct32_search_policy,
        Dct32SearchPolicy::FollowDct64Suppression
    );
    assert_eq!(
        resolved.smooth_photo_dct64_admission,
        SmoothPhotoDct64Policy::ForceSkip
    );
    assert_eq!(resolved.buttloop_qf_seed, ButtloopQfSeedPolicy::Off);
    assert_eq!(
        resolved.adaptive_quant_qf_seed,
        AdaptiveQuantQfSeedPolicy::Off
    );
    assert_eq!(
        resolved.buttloop_epf_sharpness_seed,
        EpfSharpnessSeed::LegacyUniform4
    );

    // Section A effort-gate flips: every gate at Libjxl
    assert_eq!(resolved.cfl_two_pass_min_effort, EffortGate::Libjxl);
    assert_eq!(resolved.try_dct64_min_effort, EffortGate::Libjxl);
    assert_eq!(
        resolved.epf_dynamic_sharpness_min_effort,
        EffortGate::Libjxl
    );

    // Section D KNOWN-BUG re-enable
    assert!(resolved.block_ctx_map_15_cluster);

    // W44-184: Section C CfL Newton libjxl-parity flip
    assert!(
        resolved.cfl_newton_libjxl_parity,
        "Libjxl strategy must set cfl_newton_libjxl_parity = true"
    );

    // W44-197: Section C CfL Pass-2 LS-only at e=5/6 flip
    assert!(
        resolved.cfl_pass2_ls_at_low_effort,
        "Libjxl strategy must set cfl_pass2_ls_at_low_effort = true (W44-197)"
    );

    // Perf dispatches: at Default (orthogonal to libjxl byte parity)
    assert_eq!(resolved.epf_dispatch, EpfDispatch::default());
    assert_eq!(resolved.pixel_loss_dispatch, PixelLossDispatch::default());
    assert_eq!(
        resolved.single_pass_entropy_dispatch,
        SinglePassEntropyDispatch::default()
    );
    assert_eq!(resolved.patches_dispatch, PatchesDispatch::default());
}

/// W44-184: Zenjxl / LeanFaster / Aggressive / Custom-default must
/// resolve `cfl_newton_libjxl_parity = false` so hash-locks stay
/// byte-identical and the W44-183 default-path regression doesn't
/// fire. ONLY `Libjxl` flips this bit.
#[test]
fn test_resolve_cfl_newton_libjxl_parity_only_libjxl() {
    assert!(
        !EncoderStrategy::Zenjxl
            .resolve(&StrategyOverrides::default())
            .cfl_newton_libjxl_parity,
        "Zenjxl must NOT set cfl_newton_libjxl_parity"
    );
    assert!(
        !EncoderStrategy::LeanFaster
            .resolve(&StrategyOverrides::default())
            .cfl_newton_libjxl_parity,
        "LeanFaster must NOT set cfl_newton_libjxl_parity (W44-183 \
             regression would fire on its Zenjxl-tuned cost model)"
    );
    assert!(
        !EncoderStrategy::Aggressive
            .resolve(&StrategyOverrides::default())
            .cfl_newton_libjxl_parity,
        "Aggressive must NOT set cfl_newton_libjxl_parity"
    );
    assert!(
        !EncoderStrategy::Custom(Box::default())
            .resolve(&StrategyOverrides::default())
            .cfl_newton_libjxl_parity,
        "Custom(default) must NOT set cfl_newton_libjxl_parity"
    );
    assert!(
        EncoderStrategy::Libjxl
            .resolve(&StrategyOverrides::default())
            .cfl_newton_libjxl_parity,
        "Libjxl MUST set cfl_newton_libjxl_parity"
    );
}

/// W44-197: Zenjxl / LeanFaster / Aggressive / Custom-default must
/// resolve `cfl_pass2_ls_at_low_effort = false` so the W44-29..W44-172
/// downstream cost-model calibration (which assumes no Pass-2 at e=5/6
/// on the default path) stays intact. ONLY `Libjxl` flips this bit.
#[test]
fn test_resolve_cfl_pass2_ls_at_low_effort_only_libjxl() {
    assert!(
        !EncoderStrategy::Zenjxl
            .resolve(&StrategyOverrides::default())
            .cfl_pass2_ls_at_low_effort,
        "Zenjxl must NOT set cfl_pass2_ls_at_low_effort"
    );
    assert!(
        !EncoderStrategy::LeanFaster
            .resolve(&StrategyOverrides::default())
            .cfl_pass2_ls_at_low_effort,
        "LeanFaster must NOT set cfl_pass2_ls_at_low_effort"
    );
    assert!(
        !EncoderStrategy::Aggressive
            .resolve(&StrategyOverrides::default())
            .cfl_pass2_ls_at_low_effort,
        "Aggressive must NOT set cfl_pass2_ls_at_low_effort"
    );
    assert!(
        !EncoderStrategy::Custom(Box::default())
            .resolve(&StrategyOverrides::default())
            .cfl_pass2_ls_at_low_effort,
        "Custom(default) must NOT set cfl_pass2_ls_at_low_effort"
    );
    assert!(
        EncoderStrategy::Libjxl
            .resolve(&StrategyOverrides::default())
            .cfl_pass2_ls_at_low_effort,
        "Libjxl MUST set cfl_pass2_ls_at_low_effort (W44-197 Candidate B)"
    );
}

/// `EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default())`
/// equals `ResolvedImprovements::default()` — every field at its
/// enum's `#[default]`.
#[test]
fn test_resolve_zenjxl_field_values() {
    let resolved = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert_eq!(resolved, ResolvedImprovements::default());
}

/// `EncoderStrategy::LeanFaster.resolve(...)`:
/// `high_d_photo_entropy_mul` is `Auto` (kept — cheap), all
/// screenshot-class is `Disabled` / `ForceAllow` / `ForceSkip`,
/// perf dispatches are at `Default`.
#[test]
fn test_resolve_lean_faster_field_values() {
    let resolved = EncoderStrategy::LeanFaster.resolve(&StrategyOverrides::default());

    // Photo-class entropy-mul lowering KEPT (Auto) — cheap table swaps
    assert_eq!(
        resolved.high_d_photo_entropy_mul,
        HighDPhotoEntropyMulPolicy::Auto
    );

    // Screenshot-class / heavy gates: all disabled
    assert_eq!(
        resolved.screenshot_entropy_mul,
        ScreenshotEntropyMulPolicy::Disabled
    );
    assert_eq!(resolved.dct64_search_policy, Dct64SearchPolicy::ForceAllow);
    assert_eq!(
        resolved.dct32_search_policy,
        Dct32SearchPolicy::FollowDct64Suppression
    );
    assert_eq!(
        resolved.smooth_photo_dct64_admission,
        SmoothPhotoDct64Policy::ForceSkip
    );
    assert_eq!(resolved.buttloop_qf_seed, ButtloopQfSeedPolicy::Off);
    assert_eq!(
        resolved.adaptive_quant_qf_seed,
        AdaptiveQuantQfSeedPolicy::Off
    );
    assert_eq!(
        resolved.buttloop_epf_sharpness_seed,
        EpfSharpnessSeed::LegacyUniform4
    );

    // Section A effort gates: OURS (not libjxl) — keeps our
    // speed-conscious gating
    assert_eq!(resolved.cfl_two_pass_min_effort, EffortGate::Ours);
    assert_eq!(resolved.try_dct64_min_effort, EffortGate::Ours);
    assert_eq!(resolved.epf_dynamic_sharpness_min_effort, EffortGate::Ours);

    // Section D KNOWN-BUG: not re-enabled
    assert!(!resolved.block_ctx_map_15_cluster);

    // Perf dispatches: at Default
    assert_eq!(resolved.epf_dispatch, EpfDispatch::default());
    assert_eq!(resolved.pixel_loss_dispatch, PixelLossDispatch::default());
    assert_eq!(
        resolved.single_pass_entropy_dispatch,
        SinglePassEntropyDispatch::default()
    );
    assert_eq!(resolved.patches_dispatch, PatchesDispatch::default());
}

/// Per `docs/COMPATIBILITY_MODES.md` §4.4 + §7 Q1 note:
/// `EncoderStrategy::Aggressive` is currently equivalent to
/// `EncoderStrategy::Zenjxl` after W44-124's auto-discriminator
/// obsoleted the previous "Aggressive flips W44-123 globally"
/// behaviour.
#[test]
fn test_resolve_aggressive_equals_zenjxl() {
    let aggressive = EncoderStrategy::Aggressive.resolve(&StrategyOverrides::default());
    let zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert_eq!(aggressive, zenjxl);
}

/// `Custom(Box::new(EncoderImprovementsCustom { dct64_search_policy:
/// ForceSuppress, ..Default::default() }))` round-trips through
/// resolve — the resolved struct exposes the same field values the
/// caller put in `Custom`.
#[test]
fn test_resolve_custom_round_trip() {
    let custom = EncoderImprovementsCustom {
        dct64_search_policy: Dct64SearchPolicy::ForceSuppress,
        dct32_search_policy: Dct32SearchPolicy::KeepWhenDct64Suppressed,
        buttloop_qf_seed: ButtloopQfSeedPolicy::ForceScale(2.5),
        adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::AutoScaleCustom {
            e5_e6: 1.5,
            e7: 2.0,
        },
        buttloop_epf_sharpness_seed: EpfSharpnessSeed::AutoW44_117 { min_distance: 2.0 },
        cfl_two_pass_min_effort: EffortGate::AtLeast(6),
        try_dct64_min_effort: EffortGate::Off,
        block_ctx_map_15_cluster: true,
        ..Default::default()
    };
    let strategy = EncoderStrategy::Custom(Box::new(custom.clone()));
    let resolved = strategy.resolve(&StrategyOverrides::default());

    assert_eq!(resolved.dct64_search_policy, custom.dct64_search_policy);
    assert_eq!(resolved.dct32_search_policy, custom.dct32_search_policy);
    assert_eq!(resolved.buttloop_qf_seed, custom.buttloop_qf_seed);
    assert_eq!(
        resolved.adaptive_quant_qf_seed,
        custom.adaptive_quant_qf_seed
    );
    assert_eq!(
        resolved.buttloop_epf_sharpness_seed,
        custom.buttloop_epf_sharpness_seed
    );
    assert_eq!(
        resolved.cfl_two_pass_min_effort,
        custom.cfl_two_pass_min_effort
    );
    assert_eq!(resolved.try_dct64_min_effort, custom.try_dct64_min_effort);
    assert_eq!(
        resolved.block_ctx_map_15_cluster,
        custom.block_ctx_map_15_cluster
    );

    // Fields left at Default should be at the
    // EncoderImprovementsCustom::default() value (= Zenjxl
    // baseline). Note `screenshot_entropy_mul` defaults to
    // `Disabled` (NOT `Auto`) per W44-130 Chunk D — Zenjxl
    // preserves the pre-Chunk-D default-off W22-1 lift.
    assert_eq!(
        resolved.screenshot_entropy_mul,
        ScreenshotEntropyMulPolicy::Disabled
    );
    assert_eq!(
        resolved.epf_dynamic_sharpness_min_effort,
        EffortGate::default()
    );
}

/// `StrategyOverrides` field-by-field precedence over the resolved
/// preset. `Some(...)` overrides; `None` is a no-op.
#[test]
fn test_strategy_overrides_precedence() {
    // Start from Libjxl (every screenshot gate Disabled) then
    // override two fields and confirm only those two flip.
    let overrides = StrategyOverrides {
        dct_suppress_hint: Some(true),
        dct32_keep_hint: Some(true),
        ..Default::default()
    };
    let resolved = EncoderStrategy::Libjxl.resolve(&overrides);

    // Overridden fields:
    assert_eq!(
        resolved.dct64_search_policy,
        Dct64SearchPolicy::ForceSuppress
    );
    assert_eq!(
        resolved.dct32_search_policy,
        Dct32SearchPolicy::KeepWhenDct64Suppressed
    );

    // Un-overridden fields stay at Libjxl values:
    assert_eq!(
        resolved.screenshot_entropy_mul,
        ScreenshotEntropyMulPolicy::Disabled
    );
    assert_eq!(
        resolved.buttloop_epf_sharpness_seed,
        EpfSharpnessSeed::LegacyUniform4
    );
    assert!(resolved.block_ctx_map_15_cluster);
}

/// Default impls for every nested policy match the documented
/// "production shipping" picks.
#[test]
fn test_policy_defaults() {
    assert_eq!(
        ScreenshotEntropyMulPolicy::default(),
        ScreenshotEntropyMulPolicy::Auto
    );
    assert_eq!(
        HighDPhotoEntropyMulPolicy::default(),
        HighDPhotoEntropyMulPolicy::Auto
    );
    assert_eq!(Dct64SearchPolicy::default(), Dct64SearchPolicy::Auto);
    assert_eq!(
        Dct32SearchPolicy::default(),
        Dct32SearchPolicy::FollowDct64Suppression
    );
    assert_eq!(
        SmoothPhotoDct64Policy::default(),
        SmoothPhotoDct64Policy::Auto
    );
    assert_eq!(
        ButtloopQfSeedPolicy::default(),
        ButtloopQfSeedPolicy::AutoScale4
    );
    assert_eq!(
        AdaptiveQuantQfSeedPolicy::default(),
        AdaptiveQuantQfSeedPolicy::AutoScalePerEffort
    );
    assert_eq!(
        EpfSharpnessSeed::default(),
        EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 }
    );
    assert_eq!(EffortGate::default(), EffortGate::Ours);
}

/// `EncoderImprovementsCustom::default()` ≡
/// `ResolvedImprovements::default()` field-by-field — Custom with
/// all defaults resolves to Zenjxl.
#[test]
fn test_custom_default_equals_zenjxl_resolved() {
    let custom_strategy = EncoderStrategy::Custom(Box::<EncoderImprovementsCustom>::default());
    let resolved_custom = custom_strategy.resolve(&StrategyOverrides::default());
    let resolved_zenjxl = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert_eq!(resolved_custom, resolved_zenjxl);
}

// ── W44-128 Chunk B tests ────────────────────────────────────

/// Default [`LossyConfig`] returns [`EncoderStrategy::Zenjxl`]
/// from [`LossyConfig::strategy`]. Equivalent to never calling
/// [`LossyConfig::with_strategy`].
#[test]
fn test_lossy_config_default_strategy_is_zenjxl() {
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.strategy(), &EncoderStrategy::Zenjxl);
}

/// [`LossyConfig::with_strategy`] roundtrips through
/// [`LossyConfig::strategy`] for every named variant.
#[test]
fn test_with_strategy_setter_roundtrip() {
    for variant in [
        EncoderStrategy::Libjxl,
        EncoderStrategy::LeanFaster,
        EncoderStrategy::Zenjxl,
        EncoderStrategy::Aggressive,
    ] {
        let cfg = LossyConfig::new(1.0).with_strategy(variant.clone());
        assert_eq!(cfg.strategy(), &variant);
    }
    // Custom variant carries a payload; equality is structural.
    let custom_inner = EncoderImprovementsCustom {
        dct64_search_policy: Dct64SearchPolicy::ForceSuppress,
        ..Default::default()
    };
    let custom = EncoderStrategy::Custom(Box::new(custom_inner.clone()));
    let cfg = LossyConfig::new(1.0).with_strategy(custom.clone());
    assert_eq!(cfg.strategy(), &custom);
}

/// Override precedence (W44-128 / Chunk B contract, updated for
/// W44-130 / Chunk D — `with_*_hint(Option<bool>)` setters
/// deleted; per-field overrides now flow via
/// `with_strategy_overrides(StrategyOverrides { ... })`):
///
/// 1. `with_strategy(Libjxl).with_strategy_overrides(...)`:
///    `Libjxl` resolves `dct64_search_policy = ForceAllow`. The
///    `Some(false)` override also maps to `ForceAllow` — the two
///    agree, so resolution returns `ForceAllow`. Demonstrates the
///    override path's no-op behaviour when caller and preset
///    agree.
///
/// 2. `with_strategy(Custom { dct64=ForceSuppress, .. })`
///    `.with_strategy_overrides(...)`:
///    Custom asks for `ForceSuppress`, but the override
///    rewrites it to `ForceAllow`. Demonstrates that overrides
///    WIN over the preset (mirrors the
///    `with_perceptual_optimizations(false).with_gaborish(true)`
///    precedence pattern).
#[test]
fn test_with_strategy_libjxl_then_hint_override() {
    // Case 1: Libjxl + Some(false) override → both say ForceAllow.
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_strategy_overrides(StrategyOverrides {
            dct_suppress_hint: Some(false),
            ..Default::default()
        });
    let resolved = cfg.resolve_improvements();
    assert_eq!(
        resolved.dct64_search_policy,
        Dct64SearchPolicy::ForceAllow,
        "Libjxl base + Some(false) override should both agree on ForceAllow"
    );

    // Case 2: Custom asks for ForceSuppress, but a `Some(false)`
    // override rewrites the resolved policy to ForceAllow.
    // Overrides WIN over the preset.
    let custom_inner = EncoderImprovementsCustom {
        dct64_search_policy: Dct64SearchPolicy::ForceSuppress,
        ..Default::default()
    };
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Custom(Box::new(custom_inner)))
        .with_strategy_overrides(StrategyOverrides {
            dct_suppress_hint: Some(false),
            ..Default::default()
        });
    let resolved = cfg.resolve_improvements();
    assert_eq!(
        resolved.dct64_search_policy,
        Dct64SearchPolicy::ForceAllow,
        "Some(false) override should rewrite Custom(ForceSuppress) to ForceAllow"
    );
}

/// `LossyConfig::resolve_improvements()` at the default strategy
/// (Zenjxl) with no hints set must equal `Zenjxl` resolved
/// directly — proving the resolution helper doesn't smuggle in
/// extra state from `LossyConfig`.
#[test]
fn test_resolve_improvements_default_equals_zenjxl_resolved() {
    let cfg = LossyConfig::new(1.0);
    let from_cfg = cfg.resolve_improvements();
    let direct = EncoderStrategy::Zenjxl.resolve(&StrategyOverrides::default());
    assert_eq!(from_cfg, direct);
}

/// `with_effort` preserves the caller's `with_strategy` choice.
/// Effort-derived fields regenerate; the strategy bundle does not.
#[test]
fn test_with_strategy_preserved_across_with_effort() {
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_effort(8);
    assert_eq!(cfg.strategy(), &EncoderStrategy::Libjxl);
    assert_eq!(cfg.effort(), 8);
}

/// `resolve_improvements` propagates all five `StrategyOverrides`
/// fields correctly. Starting from `Libjxl` (every relevant
/// policy at `Disabled` / `ForceAllow` / `ForceSkip`), set every
/// hint and confirm each one re-maps the matching policy field.
/// (W44-130 Chunk D: hints moved into `StrategyOverrides`.)
#[test]
fn test_resolve_improvements_propagates_all_hints() {
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_strategy_overrides(StrategyOverrides {
            screenshot_lift_hint: Some(true),
            high_d_photo_hint: Some(true),
            smooth_photo_dct64_hint: Some(true),
            dct_suppress_hint: Some(true),
            dct32_keep_hint: Some(true),
        });
    let resolved = cfg.resolve_improvements();
    assert_eq!(
        resolved.screenshot_entropy_mul,
        ScreenshotEntropyMulPolicy::ForceOn,
        "screenshot_lift_hint(Some(true)) maps to ForceOn"
    );
    assert_eq!(
        resolved.high_d_photo_entropy_mul,
        HighDPhotoEntropyMulPolicy::ForceOn,
        "high_d_photo_hint(Some(true)) maps to ForceOn"
    );
    assert_eq!(
        resolved.smooth_photo_dct64_admission,
        SmoothPhotoDct64Policy::ForceAdmit,
        "smooth_photo_dct64_hint(Some(true)) maps to ForceAdmit"
    );
    assert_eq!(
        resolved.dct64_search_policy,
        Dct64SearchPolicy::ForceSuppress,
        "dct_suppress_hint(Some(true)) maps to ForceSuppress"
    );
    assert_eq!(
        resolved.dct32_search_policy,
        Dct32SearchPolicy::KeepWhenDct64Suppressed,
        "dct32_keep_hint(Some(true)) maps to KeepWhenDct64Suppressed"
    );
    // Un-overridden Libjxl fields stay at Libjxl values.
    assert_eq!(
        resolved.buttloop_epf_sharpness_seed,
        EpfSharpnessSeed::LegacyUniform4
    );
    assert!(resolved.block_ctx_map_15_cluster);
}

/// W44-130 Chunk D — `with_strategy_overrides` setter round-trips:
/// the setter stores the struct verbatim and the getter returns
/// a reference to it. Default is all-`None` (no overrides applied).
#[test]
fn test_with_strategy_overrides_setter_roundtrip() {
    // Default: empty overrides, all None.
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.strategy_overrides(), &StrategyOverrides::default());

    // Set + read back: every field preserved exactly.
    let overrides = StrategyOverrides {
        screenshot_lift_hint: Some(true),
        high_d_photo_hint: Some(false),
        smooth_photo_dct64_hint: Some(true),
        dct_suppress_hint: Some(false),
        dct32_keep_hint: Some(true),
    };
    let cfg = LossyConfig::new(1.0).with_strategy_overrides(overrides.clone());
    assert_eq!(cfg.strategy_overrides(), &overrides);

    // Resolved policy reflects every override (Libjxl preset →
    // every override maps to Force*; the un-set buttloop fields
    // stay at Libjxl values, confirming the overrides don't
    // leak past their five named fields).
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_strategy_overrides(overrides);
    let resolved = cfg.resolve_improvements();
    assert_eq!(
        resolved.screenshot_entropy_mul,
        ScreenshotEntropyMulPolicy::ForceOn
    );
    assert_eq!(
        resolved.high_d_photo_entropy_mul,
        HighDPhotoEntropyMulPolicy::ForceOff
    );
    assert_eq!(
        resolved.smooth_photo_dct64_admission,
        SmoothPhotoDct64Policy::ForceAdmit
    );
    assert_eq!(resolved.dct64_search_policy, Dct64SearchPolicy::ForceAllow);
    assert_eq!(
        resolved.dct32_search_policy,
        Dct32SearchPolicy::KeepWhenDct64Suppressed
    );
    // Un-overridden Libjxl-baseline fields preserved.
    assert_eq!(
        resolved.buttloop_epf_sharpness_seed,
        EpfSharpnessSeed::LegacyUniform4
    );
}

// ── W44-132 Chunk F (env-var-MUTATING tests) ────────────────
//
// Tests that mutate process env-vars live in
// `tests/strategy_env_fallback.rs` (integration test, can opt
// out of `#![forbid(unsafe_code)]` for the `unsafe { env::
// set_var(...) }` calls Rust 2024 requires). The library code
// itself just READS env-vars (safe) inside
// `apply_env_var_fallbacks` — only the test suite needs the
// mutating path.
//
// Pure unit tests below cover the no-env-var case (default
// pass-through) and the explicit-caller-wins-over-env case
// without needing to mutate the process environment.

/// With NO env vars set, the resolved policy stays at the
/// strategy preset's default value (bit-identical to pre-Chunk-F
/// resolved values when no env-var is set). This is the
/// production-default case — exercises the fallback function's
/// "field equals default but env-var unset" code path.
#[test]
fn test_w44_132_env_fallback_pure_no_env_default_passthrough() {
    // NOTE: this test does NOT mutate env vars; it reads
    // whatever the runner inherited. The cjxl-rs CI sets no
    // JXL_* env vars, so the production hash-lock test (which
    // also runs unset) is the binding gate.
    //
    // What this test verifies: when the resolved field
    // (post-overrides) equals `Default::default()`, the fallback
    // function's match-on-default check works correctly without
    // running into the actual env-var lookup path (the parent
    // `if r.field == Default::default()` short-circuits the
    // env-var read if the policy was caller-overridden — but
    // when it WASN'T overridden, the env-var path is taken
    // safely).
    //
    // The mutating tests live in
    // `tests/strategy_env_fallback.rs` for the env-on cases.

    // Use Libjxl which sets every promoted field to a NON-default
    // value (Off / Off / LegacyUniform4) so the fallback's
    // default-check short-circuits the env read entirely.
    let resolved = EncoderStrategy::Libjxl.resolve(&StrategyOverrides::default());
    assert_eq!(resolved.buttloop_qf_seed, ButtloopQfSeedPolicy::Off);
    assert_eq!(
        resolved.adaptive_quant_qf_seed,
        AdaptiveQuantQfSeedPolicy::Off
    );
    assert_eq!(
        resolved.buttloop_epf_sharpness_seed,
        EpfSharpnessSeed::LegacyUniform4
    );
}

/// Caller's explicit `EncoderStrategy::Custom(...)` value sets
/// the field to a non-default value, which short-circuits the
/// env-var fallback's `field == default` check. This test does
/// not need to mutate env vars to verify the precedence rule —
/// any non-default field value disqualifies the env-var path.
#[test]
fn test_w44_132_env_fallback_pure_custom_non_default_short_circuits() {
    // ForceScale is structurally non-default (`AutoScale4` is
    // default); fallback `if r.field == default` is false, so
    // the env-var read is skipped entirely regardless of what
    // any JXL_* env var is set to in the test runner's env.
    let custom = EncoderImprovementsCustom {
        buttloop_qf_seed: ButtloopQfSeedPolicy::ForceScale(5.0),
        adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::AutoScaleCustom {
            e5_e6: 1.5,
            e7: 2.5,
        },
        buttloop_epf_sharpness_seed: EpfSharpnessSeed::AutoW44_117 { min_distance: 3.0 },
        ..Default::default()
    };
    let strategy = EncoderStrategy::Custom(Box::new(custom.clone()));
    let resolved = strategy.resolve(&StrategyOverrides::default());
    assert_eq!(resolved.buttloop_qf_seed, custom.buttloop_qf_seed);
    assert_eq!(
        resolved.adaptive_quant_qf_seed,
        custom.adaptive_quant_qf_seed
    );
    assert_eq!(
        resolved.buttloop_epf_sharpness_seed,
        custom.buttloop_epf_sharpness_seed
    );
}

/// Multi-metric Phase 0 (RFC #3, 2026-05-25): `LossyConfig`
/// defaults to [`PerceptualMetric::Butteraugli`] +
/// [`PerceptualDevice::Auto`]. Builder symmetry: every variant
/// of either setter round-trips through the getter unchanged.
#[cfg(feature = "butteraugli-loop")]
#[test]
fn test_perceptual_metric_default_and_setter_roundtrip() {
    let cfg = LossyConfig::new(1.0);
    assert_eq!(
        cfg.perceptual_metric(),
        PerceptualMetric::Butteraugli,
        "default metric must be Butteraugli"
    );
    assert_eq!(
        cfg.perceptual_device(),
        PerceptualDevice::Auto,
        "default device must be Auto"
    );
    assert_eq!(
        cfg.perceptual_target_score(),
        None,
        "default target_score must be None"
    );

    // Each setter round-trips.
    for m in [PerceptualMetric::Butteraugli, PerceptualMetric::Cvvdp] {
        let cfg = LossyConfig::new(1.0).with_perceptual_metric(m);
        assert_eq!(cfg.perceptual_metric(), m);
    }
    for d in [
        PerceptualDevice::Auto,
        PerceptualDevice::Cpu,
        PerceptualDevice::Gpu,
    ] {
        let cfg = LossyConfig::new(1.0).with_perceptual_device(d);
        assert_eq!(cfg.perceptual_device(), d);
    }
    let cfg = LossyConfig::new(1.0).with_perceptual_target_score(Some(0.05));
    assert_eq!(cfg.perceptual_target_score(), Some(0.05));
    let cfg = cfg.with_perceptual_target_score(None);
    assert_eq!(cfg.perceptual_target_score(), None);
}

/// Phase 1 display-config backfill (2026-05-25, RFC
/// `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`): the `with_target_display`
/// setter + `target_display` getter round-trip via the public API.
/// Default is `WebSdr80`; explicit set to Phone / Tv pins the field.
#[test]
fn test_target_display_default_and_round_trip() {
    // Default ≡ WebSdr80 (preserves pre-Phase-1 behaviour).
    let cfg = LossyConfig::new(1.0);
    assert_eq!(
        cfg.target_display(),
        DisplayConfig::WebSdr80,
        "default target_display must be WebSdr80 for backwards compat"
    );
    for d in [
        DisplayConfig::WebSdr80,
        DisplayConfig::Phone,
        DisplayConfig::Tv,
    ] {
        let cfg = LossyConfig::new(1.0).with_target_display(d);
        assert_eq!(
            cfg.target_display(),
            d,
            "field value must reflect the explicit setter"
        );
    }
}

/// Phase 1 display-config backfill: `EncoderStrategy::Libjxl`
/// MUST force `resolve_target_display()` to `WebSdr80` regardless
/// of the field value (strict cjxl-parity invariant — mirrors
/// W44-126 for `with_perceptual_metric`).
#[test]
fn test_resolve_target_display_libjxl_short_circuit() {
    // Libjxl + explicit Tv setter: field reflects the setter, but
    // resolver returns WebSdr80.
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_target_display(DisplayConfig::Tv);
    assert_eq!(
        cfg.target_display(),
        DisplayConfig::Tv,
        "field value must reflect the explicit setter"
    );
    assert_eq!(
        cfg.resolve_target_display(),
        DisplayConfig::WebSdr80,
        "Libjxl strategy MUST force resolved target_display to WebSdr80 \
             — strict cjxl-parity invariant"
    );
    // Zenjxl + Phone: resolver returns Phone (no short-circuit).
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_target_display(DisplayConfig::Phone);
    assert_eq!(cfg.resolve_target_display(), DisplayConfig::Phone);
}

/// Multi-metric Phase 0 (RFC #3 §1.3, 2026-05-25): the resolver
/// applies the EncoderStrategy::Libjxl strict-parity short-circuit
/// + the per-metric cargo-feature gate.
#[cfg(feature = "butteraugli-loop")]
#[test]
fn test_resolve_perceptual_metric_libjxl_short_circuit() {
    // Libjxl strategy: explicit Cvvdp still resolves to Butteraugli
    // (strict cjxl-parity invariant).
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_perceptual_metric(PerceptualMetric::Cvvdp);
    assert_eq!(
        cfg.perceptual_metric(),
        PerceptualMetric::Cvvdp,
        "field value must reflect the explicit setter"
    );
    assert_eq!(
        cfg.resolve_perceptual_metric(),
        PerceptualMetric::Butteraugli,
        "EncoderStrategy::Libjxl must force resolve_perceptual_metric() \
             back to Butteraugli regardless of field value"
    );

    // Zenjxl strategy: resolver honors the field, modulo
    // cargo-feature gate.
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_perceptual_metric(PerceptualMetric::Cvvdp);
    let resolved = cfg.resolve_perceptual_metric();
    #[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
    assert_eq!(
        resolved,
        PerceptualMetric::Cvvdp,
        "EncoderStrategy::Zenjxl + cvvdp feature compiled must honor Cvvdp"
    );
    #[cfg(not(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu")))]
    assert_eq!(
        resolved,
        PerceptualMetric::Butteraugli,
        "Without a cvvdp cargo feature, the resolver silently \
             falls back to Butteraugli"
    );

    // Default (Butteraugli) always resolves to Butteraugli.
    let cfg = LossyConfig::new(1.0);
    assert_eq!(
        cfg.resolve_perceptual_metric(),
        PerceptualMetric::Butteraugli,
    );
}

/// Multi-metric Phase 0: `resolve_perceptual_device` is a
/// pass-through; the construct-backend dispatch consumes it via
/// `resolve_perceptual_metric_selection`. Smoke for the bundling.
///
/// Updated in Phase 1 of RFC
/// `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md` (2026-05-26):
/// `target_score` now routes through
/// [`LossyConfig::resolve_perceptual_target_score`], which forces
/// `None` for `EncoderStrategy::Libjxl` (matches the W44-126
/// strict-parity invariant). Pre-Phase-1 the field passed through
/// raw; that was safe because the backend dispatch discarded the
/// field at the `let _ = selection.target_score;` line in
/// `vardct/perceptual_backend.rs::construct_backend`. Phase 1
/// connects the wire, so the resolver MUST gate it here.
#[cfg(feature = "butteraugli-loop")]
#[test]
fn test_resolve_perceptual_metric_selection_bundles_metric_and_device() {
    // Default — Butteraugli + Auto + None.
    let cfg = LossyConfig::new(1.0);
    let sel = cfg.resolve_perceptual_metric_selection();
    assert_eq!(sel.metric, PerceptualMetric::Butteraugli);
    assert_eq!(sel.device, PerceptualDevice::Auto);
    assert_eq!(sel.target_score, None);

    // Libjxl-forced Butteraugli + explicit device still flows the
    // device through (it's a no-op for the forced backend, but the
    // selection struct is the carrier). Phase 1: target_score
    // MUST drop to None under Libjxl strict-parity.
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_perceptual_metric(PerceptualMetric::Cvvdp)
        .with_perceptual_device(PerceptualDevice::Cpu)
        .with_perceptual_target_score(Some(0.05));
    let sel = cfg.resolve_perceptual_metric_selection();
    assert_eq!(
        sel.metric,
        PerceptualMetric::Butteraugli,
        "Libjxl forces metric to Butteraugli in the bundle"
    );
    assert_eq!(sel.device, PerceptualDevice::Cpu);
    assert_eq!(
        sel.target_score, None,
        "EncoderStrategy::Libjxl MUST force target_score to None \
             (W44-126 strict-parity + RFC `RFC_BUTTERAUGLI_TARGET_SYMMETRY.md` §10)"
    );

    // Phase 1 short-circuit: with target_score set BUT a non-Libjxl
    // strategy, the value MUST pass through.
    let cfg = LossyConfig::new(1.0).with_perceptual_target_score(Some(1.2245));
    let sel = cfg.resolve_perceptual_metric_selection();
    assert_eq!(sel.target_score, Some(1.2245));

    // Phase 1 NaN/Inf/non-positive guard: bogus inputs drop to
    // None at the resolver layer (defense-in-depth — the buttloop
    // dispatch also guards).
    for bogus in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
        let cfg = LossyConfig::new(1.0).with_perceptual_target_score(Some(bogus));
        let sel = cfg.resolve_perceptual_metric_selection();
        assert_eq!(
            sel.target_score, None,
            "bogus target_score {bogus} MUST drop to None at the resolver"
        );
    }
}

/// Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
/// (2026-05-26): the strict-parity short-circuit at
/// `resolve_perceptual_target_score` forces None for Libjxl.
/// Mirrors `test_resolve_perceptual_metric_libjxl_short_circuit`
/// + `test_resolve_target_display_libjxl_short_circuit`.
#[cfg(feature = "butteraugli-loop")]
#[test]
fn test_resolve_perceptual_target_score_libjxl_short_circuit() {
    // Phase 1: Libjxl forces None regardless of caller field.
    let cfg = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_perceptual_target_score(Some(1.2245));
    assert_eq!(
        cfg.resolve_perceptual_target_score(),
        None,
        "EncoderStrategy::Libjxl MUST force resolve_perceptual_target_score() to None"
    );
    // Non-Libjxl strategies pass the caller field through.
    let cfg = LossyConfig::new(1.0).with_perceptual_target_score(Some(0.7223));
    assert_eq!(cfg.resolve_perceptual_target_score(), Some(0.7223));
    // Default is None.
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.resolve_perceptual_target_score(), None);
}

/// Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
/// (2026-05-26): NaN / Inf / non-positive caller inputs drop to
/// None at the resolver to guard the buttloop dispatch arithmetic.
/// Defense-in-depth — the metric-side lookups
/// (`butteraugli_targets.rs` + the cvvdp/zensim arms in
/// `perceptual_loop.rs`) ALSO guard.
#[cfg(feature = "butteraugli-loop")]
#[test]
fn test_resolve_perceptual_target_score_sanitises_bogus_inputs() {
    for bogus in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
        let cfg = LossyConfig::new(1.0).with_perceptual_target_score(Some(bogus));
        assert_eq!(
            cfg.resolve_perceptual_target_score(),
            None,
            "bogus target_score {bogus} MUST drop to None"
        );
    }
    // Sanity: positive finite values pass through.
    for good in [0.1_f32, 0.7223, 1.2245, 2.1936, 5.0] {
        let cfg = LossyConfig::new(1.0).with_perceptual_target_score(Some(good));
        assert_eq!(
            cfg.resolve_perceptual_target_score(),
            Some(good),
            "good target_score {good} MUST pass through"
        );
    }
}
