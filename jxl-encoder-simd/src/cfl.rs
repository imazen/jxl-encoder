// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated chroma-from-luma (CfL) dot product computation.
//!
//! `find_best_multiplier`: least-squares fitting of integer CfL coefficient.
//! Inner loop is a dual dot product (sum_aa, sum_ab) over up to 4096 elements.

const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;

/// Bias towards zero and quantize to i8 (libjxl enc_chroma_from_luma.cc:176-183).
///
/// Small CfL factors (within ±2.6 of zero) are snapped to zero to reduce
/// oscillations in the CfL map. Larger factors are shifted towards zero by 2.6.
#[inline(always)]
fn bias_and_quantize(x: f32) -> i8 {
    const TOWARDS_ZERO: f32 = 2.6;
    let biased = if x >= TOWARDS_ZERO {
        x - TOWARDS_ZERO
    } else if x <= -TOWARDS_ZERO {
        x + TOWARDS_ZERO
    } else {
        0.0
    };
    crate::scalarmath::round_f32(biased).clamp(-128.0, 127.0) as i8
}

/// Newton's method constants.
///
/// **W44-183 INTENTIONAL DIVERGENCE FROM LIBJXL AT THE DEFAULT PATH** (do not
/// "fix" without measurement):
///
/// libjxl `enc_chroma_from_luma.cc:152-159` uses `eps = 100`, `max_iters = 20`,
/// starts Newton at `x = 0`, and uses the last `x` regardless of convergence.
/// At the default (`libjxl_parity = false`) path we diverge on ALL FOUR
/// (`eps = 1.0`, `max_iters = 10`, start at LS warm-start `ls_x`, fallback to
/// `ls_x` on non-convergence).
///
/// The W44-183 spot-check (`benchmarks/w44_183_cfl_newton_spotcheck_2026-05-22.{pre,post}.tsv`)
/// measured the bit-exact libjxl port at the default path: 0/27 SSIM2 wins,
/// 25/27 SSIM2 regressions (worst -13.02 SSIM2 / +26% bytes on
/// `cid22_1531677` e7 d=5), mean +7.82% bytes. libjxl-parity Newton produces
/// the libjxl-correct ytox/ytob multipliers — but the rest of our encoder
/// (W44-29..W44-172 entropy_mul tables, AC strategy gates, buttloop seeds,
/// adaptive_quant) has been incrementally tuned against the effective-LS
/// baseline produced by these parameters. Correcting Newton to libjxl parity
/// at the default path throws off the downstream calibration.
///
/// W44-182 (`3c2026b2`) identified this as a SUSPECTED root cause for the
/// `clic_097cb426` right-column SSIM2 deficit (-7.0). W44-183 falsified that:
/// the port makes the right-column cmap BYTE-EXACT to cjxl, but the
/// right-column SSIM2 stays at -7.0/-7.7/-7.4 — CfL Newton was NOT the cause.
///
/// **The previous docstring here ("libjxl uses eps=100 which causes oscillation")
/// was empirically FALSE**: cjxl's Newton with eps=100 converges fine (134-186/256
/// nonzero tiles on `clic_097cb426`). The right framing is "OUR pipeline is
/// calibrated against this divergence at the default path; reverting it would
/// require re-calibrating W44-29..W44-172 from scratch." See
/// `docs/LIBJXL_DIVERGENCES.md` Section C + Section F.
///
/// **W44-184 SALVAGE — `libjxl_parity = true` path**: the four-parameter
/// libjxl port IS wired, but ONLY when the caller selects
/// [`crate::api::EncoderStrategy::Libjxl`] (the all-divergence libjxl-parity
/// preset). Under that strategy the rest of the cost model is also flipped
/// to libjxl-parity defaults (Section A effort gates widen, content-aware
/// gates disable, etc.) so the +7.82% SSIM2 regression observed at the
/// default-path port does not apply — the downstream calibration is
/// uniformly absent. See `EncoderImprovementsCustom::cfl_newton_libjxl_parity`
/// and `EffortProfile::apply_section_c_cfl_newton_libjxl_parity` for the
/// dispatch, plus the W44-184 commit memo for measurements.
///
/// `NEWTON_EPS_DEFAULT` and `NEWTON_MAX_ITERS_DEFAULT` are defaults for the
/// `libjxl_parity = false` path; callers can override via function parameters
/// to tune CfL fitting precision vs. convergence. These defaults are IGNORED
/// when `libjxl_parity = true` — the SIMD code unconditionally uses
/// `NEWTON_EPS_LIBJXL` (100) and `NEWTON_MAX_ITERS_LIBJXL` (20) on that path,
/// matching libjxl `enc_chroma_from_luma.cc:152-167` bit-exactly.
pub const NEWTON_EPS_DEFAULT: f32 = 1.0;
pub const NEWTON_MAX_ITERS_DEFAULT: usize = 10;
/// libjxl-bit-exact parameters used when `libjxl_parity=true`
/// (`enc_chroma_from_luma.cc:152-167`).
pub const NEWTON_EPS_LIBJXL: f32 = 100.0;
pub const NEWTON_MAX_ITERS_LIBJXL: usize = 20;
const NEWTON_CLAMP: f32 = 20.0;
const NEWTON_COEFF: f32 = 1.0 / 3.0;
const NEWTON_THRES: f32 = 100.0;
const NEWTON_STABILIZER: f32 = 0.85;
const NEWTON_CONVERGENCE: f32 = 3e-3;

/// Find the best integer CfL multiplier via regularized least-squares.
///
/// Computes: `x = -sum_ab / (sum_aa + num * distance_mul * 0.5)`
/// where `sum_aa = sum(a_i^2)`, `sum_ab = sum(a_i * b_i)`,
/// `a_i = values_m[i] / 84`, `b_i = base * values_m[i] - values_s[i]`.
pub fn find_best_multiplier(
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return find_best_multiplier_avx2(token, values_m, values_s, num, base, distance_mul);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return find_best_multiplier_neon(token, values_m, values_s, num, base, distance_mul);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            return find_best_multiplier_wasm128(
                token,
                values_m,
                values_s,
                num,
                base,
                distance_mul,
            );
        }
    }

    find_best_multiplier_scalar(values_m, values_s, num, base, distance_mul)
}

pub fn find_best_multiplier_scalar(
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    if num == 0 {
        return 0;
    }
    let mut sum_aa = 0.0_f32;
    let mut sum_ab = 0.0_f32;
    for i in 0..num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
    }
    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    bias_and_quantize(x)
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn find_best_multiplier_avx2(
    token: archmage::X64V3Token,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    use magetypes::simd::f32x8;

    if num == 0 {
        return 0;
    }

    let inv_cf = f32x8::splat(token, K_INV_COLOR_FACTOR);
    let base_v = f32x8::splat(token, base);
    let mut acc_aa = f32x8::splat(token, 0.0);
    let mut acc_ab = f32x8::splat(token, 0.0);

    let simd_end = num & !7;
    let mut i = 0;
    while i < simd_end {
        let m = crate::load_f32x8(token, values_m, i);
        let s = crate::load_f32x8(token, values_s, i);
        let a = inv_cf * m;
        let b = base_v * m - s;
        acc_aa = a.mul_add(a, acc_aa);
        acc_ab = a.mul_add(b, acc_ab);
        i += 8;
    }

    // Horizontal reduction
    let aa_arr: [f32; 8] = acc_aa.into();
    let ab_arr: [f32; 8] = acc_ab.into();
    let mut sum_aa: f32 = aa_arr.iter().sum();
    let mut sum_ab: f32 = ab_arr.iter().sum();

    // Scalar remainder
    while i < num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
        i += 1;
    }

    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    bias_and_quantize(x)
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn find_best_multiplier_neon(
    token: archmage::NeonToken,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    use magetypes::simd::f32x4;

    if num == 0 {
        return 0;
    }

    let inv_cf = f32x4::splat(token, K_INV_COLOR_FACTOR);
    let base_v = f32x4::splat(token, base);
    let mut acc_aa = f32x4::splat(token, 0.0);
    let mut acc_ab = f32x4::splat(token, 0.0);

    let simd_end = num & !3;
    let mut i = 0;
    while i < simd_end {
        let m = f32x4::from_slice(token, &values_m[i..]);
        let s = f32x4::from_slice(token, &values_s[i..]);
        let a = inv_cf * m;
        let b = base_v * m - s;
        acc_aa = a.mul_add(a, acc_aa);
        acc_ab = a.mul_add(b, acc_ab);
        i += 4;
    }

    let aa_arr: [f32; 4] = acc_aa.into();
    let ab_arr: [f32; 4] = acc_ab.into();
    let mut sum_aa: f32 = aa_arr.iter().sum();
    let mut sum_ab: f32 = ab_arr.iter().sum();

    while i < num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
        i += 1;
    }

    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    bias_and_quantize(x)
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn find_best_multiplier_wasm128(
    token: archmage::Wasm128Token,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    use magetypes::simd::f32x4;

    if num == 0 {
        return 0;
    }

    let inv_cf = f32x4::splat(token, K_INV_COLOR_FACTOR);
    let base_v = f32x4::splat(token, base);
    let mut acc_aa = f32x4::splat(token, 0.0);
    let mut acc_ab = f32x4::splat(token, 0.0);

    let simd_end = num & !3;
    let mut i = 0;
    while i < simd_end {
        let m = f32x4::from_slice(token, &values_m[i..]);
        let s = f32x4::from_slice(token, &values_s[i..]);
        let a = inv_cf * m;
        let b = base_v * m - s;
        acc_aa = a.mul_add(a, acc_aa);
        acc_ab = a.mul_add(b, acc_ab);
        i += 4;
    }

    let aa_arr: [f32; 4] = acc_aa.into();
    let ab_arr: [f32; 4] = acc_ab.into();
    let mut sum_aa: f32 = aa_arr.iter().sum();
    let mut sum_ab: f32 = ab_arr.iter().sum();

    while i < num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
        i += 1;
    }

    let x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);
    bias_and_quantize(x)
}

/// Find the best integer CfL multiplier via Newton's method with perceptual cost.
///
/// Uses the cost function: `1/3 * sum((|ax+b| + 1)^2 - 1) + distance_mul * x^2 * num`
/// where `a = values_m[i] / 84`, `b = base * values_m[i] - values_s[i]`.
///
/// Newton iterations use central finite differences for the second derivative.
/// Large residuals (|ax+b| >= 100) are clipped (ignored) for robustness.
///
/// libjxl uses this at effort >= 7 (speed_tier <= kSquirrel).
/// At effort 5-6, the fast (least-squares) path is used instead.
///
/// **W44-184**: when `libjxl_parity == true`, ignores `eps` / `max_iters`
/// and runs the libjxl-bit-exact Newton: `eps = NEWTON_EPS_LIBJXL` (100),
/// `max_iters = NEWTON_MAX_ITERS_LIBJXL` (20), starting `x = 0`, NO LS
/// fallback (uses last `x` even on non-convergence). Mirrors libjxl
/// `enc_chroma_from_luma.cc:152-167`.
/// When `false` (default), uses the W44-183-shipped behaviour: `eps` and
/// `max_iters` as supplied, starting `x = ls_x` (warm-start), LS fallback
/// on non-convergence. This default is calibrated against by
/// W44-29..W44-172 downstream cost-model tuning; only flip `libjxl_parity`
/// when running under `EncoderStrategy::Libjxl` where the rest of the
/// pipeline is also strict-libjxl-parity.
pub fn find_best_multiplier_newton(
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
    eps: f32,
    max_iters: usize,
    libjxl_parity: bool,
) -> i8 {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return find_best_multiplier_newton_avx2(
                token,
                values_m,
                values_s,
                num,
                base,
                distance_mul,
                eps,
                max_iters,
                libjxl_parity,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return find_best_multiplier_newton_neon(
                token,
                values_m,
                values_s,
                num,
                base,
                distance_mul,
                eps,
                max_iters,
                libjxl_parity,
            );
        }
    }

    find_best_multiplier_newton_scalar(
        values_m,
        values_s,
        num,
        base,
        distance_mul,
        eps,
        max_iters,
        libjxl_parity,
    )
}

/// Scalar Newton's method for CfL multiplier.
///
/// **`libjxl_parity = false` (default)**: seeds Newton from the
/// least-squares solution (warm start) so it begins near the optimum.
/// With `eps = NEWTON_EPS_DEFAULT` (1.0), Newton tries to refine the LS
/// solution toward the smoothed-L1 optimum in a few iterations. Falls
/// back to LS if Newton doesn't converge. This is the W44-183-shipped
/// default that the downstream cost model (W44-29..W44-172) is calibrated
/// against. In practice `eps = 1` produces noisy local second derivatives
/// → Newton frequently fails to converge → the encoder effectively emits
/// LS-warm-start values on most tiles. The downstream cost model is tuned
/// for THAT effective baseline (W44-183 measurement: porting libjxl's
/// parameters here regresses our pipeline by 0.25-13 SSIM2 and +7.82%
/// bytes; see `NEWTON_EPS_DEFAULT` docstring for the full narrative).
///
/// **`libjxl_parity = true` (W44-184)**: bit-exact mirror of libjxl
/// `enc_chroma_from_luma.cc:152-167`. Ignores supplied `eps` /
/// `max_iters`; uses `NEWTON_EPS_LIBJXL` (100), `NEWTON_MAX_ITERS_LIBJXL`
/// (20), starting `x = 0`, NO fallback (writes whatever `x` is at loop
/// exit regardless of convergence). Wired only under
/// [`crate::api::EncoderStrategy::Libjxl`] where the W44-29..W44-172
/// downstream calibration is also turned off (Section A effort gates
/// widen, content-aware gates disable) so the +7.82% SSIM2 regression
/// observed at the default-path port does not apply.
pub fn find_best_multiplier_newton_scalar(
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
    eps: f32,
    max_iters: usize,
    libjxl_parity: bool,
) -> i8 {
    if num == 0 {
        return 0;
    }

    // Compute LS solution (always used at `libjxl_parity=false` as both
    // start and fallback). At `libjxl_parity=true` it's not the start
    // but we still need it as a stable reference value in case `num` is
    // tiny — libjxl itself uses it implicitly because `x` starts at 0
    // and the regularizer keeps the loop near the LS minimum.
    let mut sum_aa = 0.0_f32;
    let mut sum_ab = 0.0_f32;
    for i in 0..num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
    }
    let ls_x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);

    // W44-184: at libjxl_parity, override the loop parameters and start.
    let (eps, max_iters, mut x) = if libjxl_parity {
        (NEWTON_EPS_LIBJXL, NEWTON_MAX_ITERS_LIBJXL, 0.0_f32)
    } else {
        (eps, max_iters, ls_x)
    };
    let coeffx2 = NEWTON_COEFF * 2.0;
    let mut converged = false;

    for _ in 0..max_iters {
        let mut fd = 2.0 * distance_mul * num as f32 * x;
        let mut fd_pe = 2.0 * distance_mul * num as f32 * (x + eps);
        let mut fd_me = 2.0 * distance_mul * num as f32 * (x - eps);

        for i in 0..num {
            let a = K_INV_COLOR_FACTOR * values_m[i];
            let b = base * values_m[i] - values_s[i];

            let v = a * x + b;
            let vpe = a * (x + eps) + b;
            let vme = a * (x - eps) + b;

            let av = v.abs();
            let avpe = vpe.abs();
            let avme = vme.abs();

            let acoeffx2 = coeffx2 * a;

            let mut d = acoeffx2 * (av + 1.0);
            let mut dpe = acoeffx2 * (avpe + 1.0);
            let mut dme = acoeffx2 * (avme + 1.0);

            // Sign flip for negative residuals
            if v < 0.0 {
                d = -d;
            }
            if vpe < 0.0 {
                dpe = -dpe;
            }
            if vme < 0.0 {
                dme = -dme;
            }

            // Threshold clipping: ignore large residuals
            if av < NEWTON_THRES {
                fd += d;
            }
            if avpe < NEWTON_THRES {
                fd_pe += dpe;
            }
            if avme < NEWTON_THRES {
                fd_me += dme;
            }
        }

        // Second derivative via central difference
        let ddf = (fd_pe - fd_me) / (2.0 * eps);
        let step = fd / (ddf + NEWTON_STABILIZER);
        x -= step.clamp(-NEWTON_CLAMP, NEWTON_CLAMP);

        if step.abs() < NEWTON_CONVERGENCE {
            converged = true;
            break;
        }
    }

    // W44-184: at libjxl_parity, write whatever `x` is at loop exit
    // (libjxl `enc_chroma_from_luma.cc:167` has no else branch — `x`
    // is used regardless of convergence). At default (`libjxl_parity =
    // false`), fall back to LS on non-convergence (W44-183 finding:
    // most tiles never converge at eps=1, so the encoder effectively
    // emits LS solutions — the downstream calibration is tuned for that).
    if libjxl_parity || converged {
        bias_and_quantize(x)
    } else {
        bias_and_quantize(ls_x)
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[archmage::arcane]
pub fn find_best_multiplier_newton_avx2(
    token: archmage::X64V3Token,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
    eps: f32,
    max_iters: usize,
    libjxl_parity: bool,
) -> i8 {
    use magetypes::simd::f32x8;

    if num == 0 {
        return 0;
    }

    let inv_cf = f32x8::splat(token, K_INV_COLOR_FACTOR);
    let base_v = f32x8::splat(token, base);
    let coeffx2_v = f32x8::splat(token, NEWTON_COEFF * 2.0);
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::splat(token, 0.0);
    let thres_v = f32x8::splat(token, NEWTON_THRES);

    let simd_end = num & !7;

    // Compute LS solution as starting point for Newton (reuses SIMD loop).
    let mut acc_aa = f32x8::splat(token, 0.0);
    let mut acc_ab = f32x8::splat(token, 0.0);
    let mut i = 0;
    while i < simd_end {
        let m = crate::load_f32x8(token, values_m, i);
        let s = crate::load_f32x8(token, values_s, i);
        let a = inv_cf * m;
        let b = base_v * m - s;
        acc_aa = a.mul_add(a, acc_aa);
        acc_ab = a.mul_add(b, acc_ab);
        i += 8;
    }
    let mut sum_aa: f32 = acc_aa.reduce_add();
    let mut sum_ab: f32 = acc_ab.reduce_add();
    while i < num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
        i += 1;
    }
    let ls_x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);

    // W44-184: at libjxl_parity, override loop params + start (see
    // scalar variant for the rationale + libjxl source ref).
    let (eps, max_iters, mut x) = if libjxl_parity {
        (NEWTON_EPS_LIBJXL, NEWTON_MAX_ITERS_LIBJXL, 0.0_f32)
    } else {
        (eps, max_iters, ls_x)
    };
    let mut converged = false;

    for _ in 0..max_iters {
        let x_v = f32x8::splat(token, x);
        let xpe_v = f32x8::splat(token, x + eps);
        let xme_v = f32x8::splat(token, x - eps);

        let mut acc_fd = f32x8::splat(token, 0.0);
        let mut acc_fdpe = f32x8::splat(token, 0.0);
        let mut acc_fdme = f32x8::splat(token, 0.0);

        let mut i = 0;
        while i < simd_end {
            let m = f32x8::from_slice(token, &values_m[i..]);
            let s = f32x8::from_slice(token, &values_s[i..]);
            let a = inv_cf * m;
            let b = base_v * m - s;

            let v = a.mul_add(x_v, b);
            let vpe = a.mul_add(xpe_v, b);
            let vme = a.mul_add(xme_v, b);

            let av = v.abs();
            let avpe = vpe.abs();
            let avme = vme.abs();

            let acoeffx2 = coeffx2_v * a;
            let d_unsigned = acoeffx2 * (av + one);
            let dpe_unsigned = acoeffx2 * (avpe + one);
            let dme_unsigned = acoeffx2 * (avme + one);

            // Sign flip: if v < 0 then -d else d (matches libjxl IfThenElse)
            let neg_v = v.simd_lt(zero);
            let neg_vpe = vpe.simd_lt(zero);
            let neg_vme = vme.simd_lt(zero);
            let d = f32x8::blend(neg_v, zero - d_unsigned, d_unsigned);
            let dpe = f32x8::blend(neg_vpe, zero - dpe_unsigned, dpe_unsigned);
            let dme = f32x8::blend(neg_vme, zero - dme_unsigned, dme_unsigned);

            // Threshold: zero out when |v| >= THRES
            let above = av.simd_ge(thres_v);
            let above_pe = avpe.simd_ge(thres_v);
            let above_me = avme.simd_ge(thres_v);
            acc_fd += f32x8::blend(above, zero, d);
            acc_fdpe += f32x8::blend(above_pe, zero, dpe);
            acc_fdme += f32x8::blend(above_me, zero, dme);

            i += 8;
        }

        let mut fd: f32 = 2.0 * distance_mul * num as f32 * x + acc_fd.reduce_add();
        let mut fd_pe: f32 = 2.0 * distance_mul * num as f32 * (x + eps) + acc_fdpe.reduce_add();
        let mut fd_me: f32 = 2.0 * distance_mul * num as f32 * (x - eps) + acc_fdme.reduce_add();

        // Scalar tail
        while i < num {
            let a = K_INV_COLOR_FACTOR * values_m[i];
            let b = base * values_m[i] - values_s[i];
            let v = a * x + b;
            let vpe = a * (x + eps) + b;
            let vme = a * (x - eps) + b;
            let av = v.abs();
            let avpe = vpe.abs();
            let avme = vme.abs();
            let acoeffx2 = NEWTON_COEFF * 2.0 * a;
            let mut d = acoeffx2 * (av + 1.0);
            let mut dpe = acoeffx2 * (avpe + 1.0);
            let mut dme = acoeffx2 * (avme + 1.0);
            if v < 0.0 {
                d = -d;
            }
            if vpe < 0.0 {
                dpe = -dpe;
            }
            if vme < 0.0 {
                dme = -dme;
            }
            if av < NEWTON_THRES {
                fd += d;
            }
            if avpe < NEWTON_THRES {
                fd_pe += dpe;
            }
            if avme < NEWTON_THRES {
                fd_me += dme;
            }
            i += 1;
        }

        let ddf = (fd_pe - fd_me) / (2.0 * eps);
        let step = fd / (ddf + NEWTON_STABILIZER);
        x -= step.clamp(-NEWTON_CLAMP, NEWTON_CLAMP);

        if step.abs() < NEWTON_CONVERGENCE {
            converged = true;
            break;
        }
    }

    // W44-184: at libjxl_parity, use whatever `x` is at loop exit (no
    // fallback). At default, fall back to LS on non-convergence (the
    // W44-183-shipped behaviour the downstream calibration is tuned
    // against).
    if libjxl_parity || converged {
        bias_and_quantize(x)
    } else {
        bias_and_quantize(ls_x)
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[archmage::arcane]
pub fn find_best_multiplier_newton_neon(
    token: archmage::NeonToken,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
    eps: f32,
    max_iters: usize,
    libjxl_parity: bool,
) -> i8 {
    use magetypes::simd::f32x4;

    if num == 0 {
        return 0;
    }

    let inv_cf = f32x4::splat(token, K_INV_COLOR_FACTOR);
    let base_v = f32x4::splat(token, base);
    let coeffx2_v = f32x4::splat(token, NEWTON_COEFF * 2.0);
    let one = f32x4::splat(token, 1.0);
    let zero = f32x4::splat(token, 0.0);
    let thres_v = f32x4::splat(token, NEWTON_THRES);

    let simd_end = num & !3;

    // Compute LS solution as starting point for Newton (reuses SIMD loop).
    let mut acc_aa = f32x4::splat(token, 0.0);
    let mut acc_ab = f32x4::splat(token, 0.0);
    let mut i = 0;
    while i < simd_end {
        let m = f32x4::from_slice(token, &values_m[i..]);
        let s = f32x4::from_slice(token, &values_s[i..]);
        let a = inv_cf * m;
        let b = base_v * m - s;
        acc_aa = a.mul_add(a, acc_aa);
        acc_ab = a.mul_add(b, acc_ab);
        i += 4;
    }
    let mut sum_aa: f32 = acc_aa.reduce_add();
    let mut sum_ab: f32 = acc_ab.reduce_add();
    while i < num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        sum_aa += a * a;
        sum_ab += a * b;
        i += 1;
    }
    let ls_x = -sum_ab / (sum_aa + num as f32 * distance_mul * 0.5);

    // W44-184: at libjxl_parity, override loop params + start (see
    // scalar variant for the rationale + libjxl source ref).
    let (eps, max_iters, mut x) = if libjxl_parity {
        (NEWTON_EPS_LIBJXL, NEWTON_MAX_ITERS_LIBJXL, 0.0_f32)
    } else {
        (eps, max_iters, ls_x)
    };
    let mut converged = false;

    for _ in 0..max_iters {
        let x_v = f32x4::splat(token, x);
        let xpe_v = f32x4::splat(token, x + eps);
        let xme_v = f32x4::splat(token, x - eps);

        let mut acc_fd = f32x4::splat(token, 0.0);
        let mut acc_fdpe = f32x4::splat(token, 0.0);
        let mut acc_fdme = f32x4::splat(token, 0.0);

        let mut i = 0;
        while i < simd_end {
            let m = f32x4::from_slice(token, &values_m[i..]);
            let s = f32x4::from_slice(token, &values_s[i..]);
            let a = inv_cf * m;
            let b = base_v * m - s;

            let v = a.mul_add(x_v, b);
            let vpe = a.mul_add(xpe_v, b);
            let vme = a.mul_add(xme_v, b);

            let av = v.abs();
            let avpe = vpe.abs();
            let avme = vme.abs();

            let acoeffx2 = coeffx2_v * a;
            let d_unsigned = acoeffx2 * (av + one);
            let dpe_unsigned = acoeffx2 * (avpe + one);
            let dme_unsigned = acoeffx2 * (avme + one);

            let neg_v = v.simd_lt(zero);
            let neg_vpe = vpe.simd_lt(zero);
            let neg_vme = vme.simd_lt(zero);
            let d = f32x4::blend(neg_v, zero - d_unsigned, d_unsigned);
            let dpe = f32x4::blend(neg_vpe, zero - dpe_unsigned, dpe_unsigned);
            let dme = f32x4::blend(neg_vme, zero - dme_unsigned, dme_unsigned);

            let above = av.simd_ge(thres_v);
            let above_pe = avpe.simd_ge(thres_v);
            let above_me = avme.simd_ge(thres_v);
            acc_fd += f32x4::blend(above, zero, d);
            acc_fdpe += f32x4::blend(above_pe, zero, dpe);
            acc_fdme += f32x4::blend(above_me, zero, dme);

            i += 4;
        }

        let mut fd: f32 = 2.0 * distance_mul * num as f32 * x + acc_fd.reduce_add();
        let mut fd_pe: f32 = 2.0 * distance_mul * num as f32 * (x + eps) + acc_fdpe.reduce_add();
        let mut fd_me: f32 = 2.0 * distance_mul * num as f32 * (x - eps) + acc_fdme.reduce_add();

        while i < num {
            let a = K_INV_COLOR_FACTOR * values_m[i];
            let b = base * values_m[i] - values_s[i];
            let v = a * x + b;
            let vpe = a * (x + eps) + b;
            let vme = a * (x - eps) + b;
            let av = v.abs();
            let avpe = vpe.abs();
            let avme = vme.abs();
            let acoeffx2 = NEWTON_COEFF * 2.0 * a;
            let mut d = acoeffx2 * (av + 1.0);
            let mut dpe = acoeffx2 * (avpe + 1.0);
            let mut dme = acoeffx2 * (avme + 1.0);
            if v < 0.0 {
                d = -d;
            }
            if vpe < 0.0 {
                dpe = -dpe;
            }
            if vme < 0.0 {
                dme = -dme;
            }
            if av < NEWTON_THRES {
                fd += d;
            }
            if avpe < NEWTON_THRES {
                fd_pe += dpe;
            }
            if avme < NEWTON_THRES {
                fd_me += dme;
            }
            i += 1;
        }

        let ddf = (fd_pe - fd_me) / (2.0 * eps);
        let step = fd / (ddf + NEWTON_STABILIZER);
        x -= step.clamp(-NEWTON_CLAMP, NEWTON_CLAMP);

        if step.abs() < NEWTON_CONVERGENCE {
            converged = true;
            break;
        }
    }

    // W44-184: at libjxl_parity, use whatever `x` is at loop exit (no
    // fallback). At default, fall back to LS on non-convergence.
    if libjxl_parity || converged {
        bias_and_quantize(x)
    } else {
        bias_and_quantize(ls_x)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn test_find_best_multiplier_scalar_vs_dispatch() {
        let num = 256;
        let values_m: alloc::vec::Vec<f32> = (0..num).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let values_s: alloc::vec::Vec<f32> =
            (0..num).map(|i| (i as f32 - 128.0) * 0.05 + 0.3).collect();

        let ref0 = find_best_multiplier_scalar(&values_m, &values_s, num, 0.0, 1e-3);
        let ref1 = find_best_multiplier_scalar(&values_m, &values_s, num, 1.0, 1e-3);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let test0 = find_best_multiplier(&values_m, &values_s, num, 0.0, 1e-3);
                assert_eq!(
                    ref0, test0,
                    "base=0.0: scalar={ref0} dispatch={test0} [{perm}]"
                );

                let test1 = find_best_multiplier(&values_m, &values_s, num, 1.0, 1e-3);
                assert_eq!(
                    ref1, test1,
                    "base=1.0: scalar={ref1} dispatch={test1} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_find_best_multiplier_empty() {
        assert_eq!(find_best_multiplier(&[], &[], 0, 0.0, 1e-3), 0);
    }

    #[test]
    fn test_find_best_multiplier_correlated() {
        let factor = 42.0_f32;
        let m: alloc::vec::Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 10.0).collect();
        let s: alloc::vec::Vec<f32> = m.iter().map(|&v| factor / 84.0 * v).collect();
        let result = find_best_multiplier(&m, &s, 64, 0.0, 1e-3);
        // Optimization produces ~42.0, towards_zero bias subtracts 2.6 → ~39.4 → rounds to 39
        let expected = (factor - 2.6).round() as i8;
        assert!(
            (result as f32 - expected as f32).abs() < 2.0,
            "Expected ~{expected} (factor {factor} - 2.6 bias), got {result}"
        );
    }

    #[test]
    fn test_newton_empty() {
        assert_eq!(
            find_best_multiplier_newton(
                &[],
                &[],
                0,
                0.0,
                1e-9,
                NEWTON_EPS_DEFAULT,
                NEWTON_MAX_ITERS_DEFAULT,
                false,
            ),
            0,
        );
        // W44-184: libjxl_parity also handles empty input safely.
        assert_eq!(
            find_best_multiplier_newton(&[], &[], 0, 0.0, 1e-9, 0.0, 0, true),
            0,
        );
    }

    #[test]
    fn test_newton_scalar_vs_dispatch() {
        let num = 256;
        let values_m: alloc::vec::Vec<f32> = (0..num).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let values_s: alloc::vec::Vec<f32> =
            (0..num).map(|i| (i as f32 - 128.0) * 0.05 + 0.3).collect();

        let eps = NEWTON_EPS_DEFAULT;
        let iters = NEWTON_MAX_ITERS_DEFAULT;

        let ref0 = find_best_multiplier_newton_scalar(
            &values_m, &values_s, num, 0.0, 1e-9, eps, iters, false,
        );
        let ref1 = find_best_multiplier_newton_scalar(
            &values_m, &values_s, num, 1.0, 1e-9, eps, iters, false,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let test0 = find_best_multiplier_newton(
                    &values_m, &values_s, num, 0.0, 1e-9, eps, iters, false,
                );
                assert_eq!(
                    ref0, test0,
                    "newton base=0.0: scalar={ref0} dispatch={test0} [{perm}]"
                );

                let test1 = find_best_multiplier_newton(
                    &values_m, &values_s, num, 1.0, 1e-9, eps, iters, false,
                );
                assert_eq!(
                    ref1, test1,
                    "newton base=1.0: scalar={ref1} dispatch={test1} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    /// W44-184: scalar vs dispatched SIMD agreement on the
    /// `libjxl_parity=true` path. Verifies all three SIMD variants
    /// (scalar / AVX2 / NEON) consume the new bool and produce the
    /// same i8 result given identical inputs.
    #[test]
    fn test_newton_libjxl_parity_scalar_vs_dispatch() {
        let num = 256;
        let values_m: alloc::vec::Vec<f32> = (0..num).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let values_s: alloc::vec::Vec<f32> =
            (0..num).map(|i| (i as f32 - 128.0) * 0.05 + 0.3).collect();

        let ref0 = find_best_multiplier_newton_scalar(
            &values_m, &values_s, num, 0.0, 1e-9, 999.0, 99, true,
        );
        let ref1 = find_best_multiplier_newton_scalar(
            &values_m, &values_s, num, 1.0, 1e-9, 999.0, 99, true,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let test0 = find_best_multiplier_newton(
                    &values_m, &values_s, num, 0.0, 1e-9, 999.0, 99, true,
                );
                assert_eq!(
                    ref0, test0,
                    "newton libjxl_parity base=0.0: scalar={ref0} dispatch={test0} [{perm}]"
                );

                let test1 = find_best_multiplier_newton(
                    &values_m, &values_s, num, 1.0, 1e-9, 999.0, 99, true,
                );
                assert_eq!(
                    ref1, test1,
                    "newton libjxl_parity base=1.0: scalar={ref1} dispatch={test1} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    /// W44-184: confirm `libjxl_parity=true` ignores the supplied
    /// `eps` and `max_iters` arguments. Two calls with identical inputs
    /// but wildly different `eps`/`max_iters` must produce the same
    /// quantized i8 — the supplied params are dead args on this path.
    #[test]
    fn test_newton_libjxl_parity_ignores_args() {
        let num = 256;
        let values_m: alloc::vec::Vec<f32> = (0..num).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let values_s: alloc::vec::Vec<f32> =
            (0..num).map(|i| (i as f32 - 128.0) * 0.05 + 0.3).collect();

        let a =
            find_best_multiplier_newton_scalar(&values_m, &values_s, num, 0.0, 1e-9, 1.0, 5, true);
        let b = find_best_multiplier_newton_scalar(
            &values_m, &values_s, num, 0.0, 1e-9, 999.0, 999, true,
        );
        assert_eq!(
            a, b,
            "libjxl_parity must ignore eps/max_iters args (a={a} b={b})",
        );
    }

    #[test]
    fn test_newton_correlated() {
        // With a strong correlation, Newton should find a similar result to least-squares
        let factor = 42.0_f32;
        let m: alloc::vec::Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 10.0).collect();
        let s: alloc::vec::Vec<f32> = m.iter().map(|&v| factor / 84.0 * v).collect();

        let ls_result = find_best_multiplier(&m, &s, 64, 0.0, 1e-9);
        let eps = NEWTON_EPS_DEFAULT;
        let iters = NEWTON_MAX_ITERS_DEFAULT;
        let newton_result = find_best_multiplier_newton(&m, &s, 64, 0.0, 1e-9, eps, iters, false);

        // Both should be in the right ballpark (Newton uses perceptual cost, not MSE)
        let expected = (factor - 2.6).round() as i8;
        assert!(
            (newton_result as f32 - expected as f32).abs() <= 3.0,
            "Newton expected ~{expected}, got {newton_result}"
        );
        // Newton and LS should agree within ±3 for well-conditioned data
        assert!(
            (newton_result as i16 - ls_result as i16).unsigned_abs() <= 3,
            "Newton={newton_result} vs LS={ls_result} differ by more than 3"
        );
    }
}
