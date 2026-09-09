// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Wall-clock regression gates.
//!
//! # Why these exist
//!
//! imazen/jxl-encoder#115 was a ~1550x pathology on shipped paths — a fixed
//! compaction threshold that turned a distinct-value collector quadratic — and
//! **every existing gate was blind to it**. Hash locks, byte locks and the
//! drift suite all compare *bytes*, and the bug changed no bytes at all. It
//! survived until someone happened to time an encode.
//!
//! # What they assert, and why not milliseconds
//!
//! Absolute timings are a property of the machine, not of the code: they vary
//! with core count, thermal state, and whatever else is running. A threshold in
//! milliseconds is either so loose it catches nothing or so tight it fails on a
//! busy laptop.
//!
//! So these gates assert **shape, not speed** — how cost GROWS with input size.
//! Doubling the pixels should roughly double the work; a quadratic regression
//! quadruples it. That ratio is dimensionless, so it cancels out machine speed
//! almost entirely, which is what makes a timing test tolerable in a normal
//! `cargo test` run.
//!
//! # Skipped under CI
//!
//! `CI=true` skips them. Shared CI runners are exactly where timing noise is
//! worst — neighbours, throttling, oversubscription — and a flaky gate that
//! people learn to ignore is worse than no gate. These run on developer
//! machines, where a regression is caught by the person who introduced it.
//!
//! # The honest limitation
//!
//! **A ladder of small images would NOT have caught #115.** That bug only
//! engaged once a property's distinct set approached 65536, which needs more
//! samples than a small image provides — at 256x256 it did not reproduce at
//! all. Size-ladder gates catch pathologies whose onset is below the top of the
//! ladder; they cannot catch one whose onset is above it.
//!
//! That is why there are two gates, not one: the encoder ladder here for
//! general superlinearity, and a direct complexity assertion on the collector
//! itself (`modular::tree_learn::tests::distinct_collector_*`), which reaches
//! the degenerate regime cheaply because it needs no encoding. Neither is
//! sufficient alone.

use std::time::{Duration, Instant};

use jxl_encoder::api::{LosslessConfig, PixelLayout};

/// CI runners are too noisy for ratio timing; see the module docs.
fn skip_under_ci() -> bool {
    matches!(
        std::env::var("CI").as_deref(),
        Ok("true") | Ok("1") | Ok("TRUE")
    )
}

/// Smallest measurement we will draw a conclusion from. Below this the clock
/// resolution and per-encode fixed costs dominate the signal, and a ratio
/// computed from noise is worse than no ratio.
const MIN_MEANINGFUL: Duration = Duration::from_millis(2);

/// Min of `reps` — the least noisy estimator of "how fast can this go" on a
/// machine that is also doing other things. Interleaving is not needed here
/// because each arm is a different input size rather than a different code
/// path, so there is no A/B ordering bias to cancel.
fn best_of<F: FnMut() -> usize>(reps: usize, mut f: F) -> (Duration, usize) {
    let mut best = Duration::MAX;
    let mut out = 0;
    for _ in 0..reps {
        let t = Instant::now();
        out = f();
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    (best, out)
}

/// Content whose property columns carry many distinct values — the shape that
/// stresses tree learning — while staying smooth enough to be realistic.
fn planar_ramp(w: usize, h: usize, bits: u32) -> Vec<u32> {
    let max = (1u32 << bits) - 1;
    (0..w * h)
        .map(|i| {
            let x = (i % w) as u64;
            let y = (i / w) as u64;
            (((x * 65_537 + y * 4_099) & u64::from(max)) as u32).min(max)
        })
        .collect()
}

/// Assert that quadrupling the pixel count does not quadruple the *rate*.
///
/// Linear work gives ~4x for 4x the pixels; `n log n` gives ~4.3x. A quadratic
/// regression gives 16x in theory and far more in practice (#115 was ~1550x on
/// its worst cell). The bound is set at **10x** — comfortably above anything
/// well-behaved, comfortably below anything quadratic. It is deliberately not
/// tight: this gate exists to catch catastrophes, and a gate that fires on a
/// 20 % drift would be turned off within a week.
fn assert_subquadratic(label: &str, small: Duration, large: Duration, pixel_ratio: f64) {
    let ratio = large.as_secs_f64() / small.as_secs_f64();
    // Bound is overridable so the gate's own sensitivity can be demonstrated:
    // `WALL_CLOCK_GATE_BOUND=1.0 cargo test ... wall_clock_gates` must FAIL.
    // A gate nobody has ever seen fail is indistinguishable from one that
    // cannot fail.
    let bound: f64 = std::env::var("WALL_CLOCK_GATE_BOUND")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10.0);
    // Report what was measured rather than passing silently — a green gate
    // should still leave evidence of what it observed.
    eprintln!(
        "wall_clock_gates: {label}: {pixel_ratio}x pixels -> {ratio:.2}x time          (small={small:?} large={large:?}, bound {bound}x)"
    );
    assert!(
        ratio < bound,
        "{label}: {pixel_ratio}x the pixels cost {ratio:.1}x the time \
         (bound {bound}x). Linear would be ~{pixel_ratio}x and n log n a little \
         more; a ratio near or above the bound means some phase went \
         superlinear. small={small:?} large={large:?}"
    );
}

#[test]
fn lossless_scales_subquadratically_with_pixels() {
    if skip_under_ci() {
        eprintln!("wall_clock_gates: skipped (CI=true)");
        return;
    }
    let cfg = LosslessConfig::new();
    // 4x pixels between the two, both small enough to keep the whole gate well
    // under a second on a normal machine.
    let (small_w, small_h) = (128usize, 128usize);
    let (large_w, large_h) = (256usize, 256usize);

    let s = planar_ramp(small_w, small_h, 17);
    let l = planar_ramp(large_w, large_h, 17);

    let (t_small, _) = best_of(3, || {
        cfg.encode_planar_int(small_w as u32, small_h as u32, &[&s], 17, true, false)
            .expect("small encode")
            .len()
    });
    let (t_large, _) = best_of(3, || {
        cfg.encode_planar_int(large_w as u32, large_h as u32, &[&l], 17, true, false)
            .expect("large encode")
            .len()
    });

    if t_small < MIN_MEANINGFUL {
        eprintln!(
            "wall_clock_gates: base measurement {t_small:?} below the {MIN_MEANINGFUL:?} \
             floor — machine too fast for this ladder to say anything; skipping"
        );
        return;
    }
    assert_subquadratic("lossless planar 17-bit", t_small, t_large, 4.0);
}

#[test]
fn lossless_16bit_layout_scales_subquadratically() {
    if skip_under_ci() {
        eprintln!("wall_clock_gates: skipped (CI=true)");
        return;
    }
    // The shipped Rgb16 path — the one #115 actually degraded. Three channels
    // with different slopes so the reference-channel delta properties, which
    // span 17 bits, carry many distinct values.
    let build = |w: usize, h: usize| -> Vec<u8> {
        let mut px = Vec::with_capacity(w * h * 6);
        for i in 0..w * h {
            let x = (i % w) as u64;
            let y = (i / w) as u64;
            let base = x * 61 + y * 37;
            for k in [0u64, 3, 7] {
                let v = ((base * (k + 1) / 2 + k * 9_311) % 65_536) as u16;
                px.extend_from_slice(&v.to_ne_bytes());
            }
        }
        px
    };
    let cfg = LosslessConfig::new();
    let s = build(128, 128);
    let l = build(256, 256);

    let (t_small, _) = best_of(3, || {
        cfg.encode_request(128, 128, PixelLayout::Rgb16)
            .encode(&s)
            .expect("small")
            .len()
    });
    let (t_large, _) = best_of(3, || {
        cfg.encode_request(256, 256, PixelLayout::Rgb16)
            .encode(&l)
            .expect("large")
            .len()
    });

    if t_small < MIN_MEANINGFUL {
        eprintln!("wall_clock_gates: base {t_small:?} below floor; skipping");
        return;
    }
    assert_subquadratic("lossless Rgb16", t_small, t_large, 4.0);
}

#[test]
fn lossy_scales_subquadratically_with_pixels() {
    if skip_under_ci() {
        eprintln!("wall_clock_gates: skipped (CI=true)");
        return;
    }
    use jxl_encoder::api::LossyConfig;
    let build = |w: usize, h: usize| -> Vec<u8> {
        let mut px = Vec::with_capacity(w * h * 3);
        for i in 0..w * h {
            let x = (i % w) as u32;
            let y = (i / w) as u32;
            px.push(((x * 7 + y * 3) % 256) as u8);
            px.push(((x * 3 + y * 11) % 256) as u8);
            px.push(((x ^ y) % 256) as u8);
        }
        px
    };
    let cfg = LossyConfig::new(1.0);
    let s = build(128, 128);
    let l = build(256, 256);

    let (t_small, _) = best_of(3, || {
        cfg.encode_request(128, 128, PixelLayout::Rgb8)
            .encode(&s)
            .expect("small")
            .len()
    });
    let (t_large, _) = best_of(3, || {
        cfg.encode_request(256, 256, PixelLayout::Rgb8)
            .encode(&l)
            .expect("large")
            .len()
    });

    if t_small < MIN_MEANINGFUL {
        eprintln!("wall_clock_gates: base {t_small:?} below floor; skipping");
        return;
    }
    assert_subquadratic("lossy VarDCT", t_small, t_large, 4.0);
}
