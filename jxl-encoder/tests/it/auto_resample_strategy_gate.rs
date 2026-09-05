//! Issue #101 follow-up: libjxl's automatic 2× resampling switch
//! (`enc_frame.cc:108-114`: `distance >= 10` → `resampling = 2`, internal
//! distance `d * 0.25 + 0.25`) is a strategy gate, `auto_resample_libjxl_rule`.
//!
//! Measured 2026-09-05 on 20 real images × e5/e8
//! (`benchmarks/auto_resample_monotonicity_2026-09-05.analysis.md`): the
//! switch never produced the cheaper regime at matched butteraugli at
//! d = 10 — photos paid +12 %/+30 % bytes for slightly worse butteraugli,
//! graphics lost 9–26 butteraugli / 30–85 SSIM2 — and won only at d ≥ 17 on
//! 6/40 cells by ≤ 14 %. cjxl v0.11.1 shows the same at its own d ≥ 20
//! switch. So the zen strategies keep ONE regime at every distance (no
//! structural byte or quality discontinuity on a distance ladder), while
//! `EncoderStrategy::Libjxl` keeps the rule for byte parity, and the caller
//! pin `with_auto_resampling(bool)` wins over either strategy default.
//!
//! These cells pin the contract by byte identity (the same encode paths are
//! decoder-validated in `resampling_odd_dims*.rs`): the default output at
//! d ≥ 10 must equal the auto-off output; the opt-in output must equal the
//! explicit `with_resampling(2)` encode at the remapped distance; the Libjxl
//! strategy must still switch. 300×200 = multi-group (two 256-wide groups).

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};

/// Procedural fixture: gradient + xorshift noise (DC and AC energy).
fn fixture(w: u32, h: u32) -> Vec<u8> {
    let mut state = 0x2545_F491_u32 ^ (w << 16) ^ h;
    let mut px = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h {
        for x in 0..w {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state & 0x1F) as u8;
            px.push(((x * 255) / w.max(1)) as u8 ^ noise);
            px.push(((y * 255) / h.max(1)) as u8);
            px.push(((x ^ y) & 0xFF) as u8);
        }
    }
    px
}

fn encode(cfg: LossyConfig, px: &[u8], w: u32, h: u32) -> Vec<u8> {
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(px)
        .expect("encode")
}

/// Default (Zenjxl) never switches regime: at every distance across and
/// beyond the libjxl threshold the output is byte-identical to auto-off,
/// and differs from the opted-in (switched) encode so the check is not
/// vacuous.
#[test]
fn issue_101_default_strategy_keeps_one_regime_across_the_libjxl_threshold() {
    let (w, h) = (300u32, 200u32);
    let px = fixture(w, h);
    for &d in &[9.9f32, 10.0, 10.5, 12.0, 25.0] {
        let cfg = LossyConfig::new(d).with_effort(5);
        assert!(
            !cfg.auto_resampling(),
            "d={d}: zen default must not enable the rule"
        );
        assert_eq!(cfg.effective_resampling(), 1, "d={d}: one regime");
        assert_eq!(
            cfg.effective_distance(),
            d,
            "d={d}: distance passes through"
        );
        let default = encode(cfg.clone(), &px, w, h);
        let off = encode(cfg.clone().with_auto_resampling(false), &px, w, h);
        assert_eq!(
            default, off,
            "d={d}: default must equal auto-off byte for byte"
        );
        if d >= 10.0 {
            let on = encode(cfg.with_auto_resampling(true), &px, w, h);
            assert_ne!(
                default, on,
                "d={d}: the opted-in rule must change the stream (non-vacuous)"
            );
        }
    }
}

/// The opted-in rule is exactly libjxl's: 2× at internal distance
/// `d * 0.25 + 0.25`, byte-identical to the explicit `with_resampling(2)`
/// encode at that distance.
#[test]
fn issue_101_opt_in_matches_explicit_res2_at_remapped_distance() {
    let (w, h) = (300u32, 200u32);
    let px = fixture(w, h);
    for &d in &[10.0f32, 15.0, 25.0] {
        let remapped = d * 0.25 + 0.25;
        let on = encode(
            LossyConfig::new(d)
                .with_effort(5)
                .with_auto_resampling(true),
            &px,
            w,
            h,
        );
        let explicit = encode(
            LossyConfig::new(remapped).with_effort(5).with_resampling(2),
            &px,
            w,
            h,
        );
        assert_eq!(
            on, explicit,
            "d={d}: opt-in must equal with_resampling(2) at {remapped}"
        );
    }
}

/// `EncoderStrategy::Libjxl` keeps libjxl's switch (byte parity): at d=10
/// it equals the strategy's explicit 2× encode at 2.75 — with the ORIGINAL
/// distance (10) pinned, because under `x_qm_scale_from_original_distance`
/// (a Libjxl-only gate mirroring `enc_frame.cc:676`) the auto path encodes
/// at 2.75 but derives `x_qm_scale` from the requested 10, as cjxl does —
/// and differs from its own auto-off encode; the caller pin still wins.
#[test]
fn issue_101_libjxl_strategy_keeps_the_switch() {
    let (w, h) = (300u32, 200u32);
    let px = fixture(w, h);
    let base = LossyConfig::new(10.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::Libjxl);
    assert!(base.auto_resampling());
    assert_eq!(base.effective_resampling(), 2);
    let default = encode(base.clone(), &px, w, h);
    let explicit = encode(
        LossyConfig::new(2.75)
            .with_effort(5)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_resampling(2)
            .with_original_distance(Some(10.0)),
        &px,
        w,
        h,
    );
    assert_eq!(
        default, explicit,
        "Libjxl default at d=10 must be the 2x encode at 2.75 with original distance 10"
    );
    let pinned_off = encode(base.with_auto_resampling(false), &px, w, h);
    assert_ne!(
        default, pinned_off,
        "with_auto_resampling(false) must win over the Libjxl default"
    );
}
