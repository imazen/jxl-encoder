// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated coefficient processing for entropy estimation.
//!
//! The inner coefficient loop of `estimate_entropy_full` is the single biggest
//! encoder hotspot (~7.5% CPU). This kernel vectorizes the per-coefficient math:
//!   val = (block_c[i] - block_y[i] * cmap_factor) / weights[i] * quant
//!   rval = round(val)
//!   entropy_sum += sqrt(|rval|) * cost_delta
//!   nzeros += (rval != 0)
//!
//! **magetypes-consolidated** (W43-2 chunk-7, final): the three per-arch
//! variants (AVX2 / NEON / WASM128) previously copy-pasted in this file
//! collapse to one `#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]`
//! body (`entropy_coeffs_impl`). The macro generates one
//! `#[arcane]`-wrapped variant per listed tier:
//!   - `entropy_coeffs_impl_v4` (x86_64 AVX-512 native 256-bit f32x8,
//!     opt-in via the `avx512` cargo feature)
//!   - `entropy_coeffs_impl_v3` (x86_64 AVX2, native 256-bit f32x8)
//!   - `entropy_coeffs_impl_neon` (aarch64, 2× f32x4 polyfill of f32x8)
//!   - `entropy_coeffs_impl_wasm128` (wasm32, 2× f32x4 polyfill of f32x8)
//!   - `entropy_coeffs_impl_scalar` (portable scalar fallback)
//!
//! Pure-f32 kernel: `v4` tier compiles cleanly here (no `f64x4` is required
//! anywhere in the body — all five accumulators are `f32x8`). Contrast with
//! W43-2 chunk-5's `pixel_domain_loss` body which DID require `f64x4`
//! (8th-power norm) and was capped at `v3`.
//!
//! **FMA-reduction-order preservation.** The five-accumulator layout is the
//! exact structure the prior hand-written AVX2 / NEON / WASM bodies used,
//! which had IDENTICAL FMA op order across all three (lane width was the
//! only difference). The macro-generated variants inherit that order
//! unchanged. Per W44-9 (Sub-chunk B in flight), even subtle FMA-order
//! perturbations affect AC-strategy selection and risk bitstream divergence
//! — so the body below is a 1:1 collapse of the prior arch variants, not
//! a re-derivation.

/// Results from vectorized entropy coefficient processing.
#[derive(Debug, Clone, Copy)]
pub struct EntropyCoeffResult {
    /// Sum of sqrt(|round(val)|) * cost_delta for all coefficients.
    pub entropy_sum: f32,
    /// Count of non-zero quantized coefficients.
    pub nzeros_sum: f32,
    /// Sum of |val - round(val)| (coefficient-domain mode only).
    pub info_loss_sum: f32,
    /// Sum of (val - round(val))^2 (coefficient-domain mode only).
    pub info_loss2_sum: f32,
}

impl EntropyCoeffResult {
    pub const ZERO: Self = Self {
        entropy_sum: 0.0,
        nzeros_sum: 0.0,
        info_loss_sum: 0.0,
        info_loss2_sum: 0.0,
    };
}

use archmage::prelude::*;

/// Vectorized entropy coefficient processing.
///
/// For each coefficient `i` in 0..n:
///   `val = (block_c[i] - block_y[i] * cmap_factor) * inv_weights[i] * quant`
///   rval = round(val)
///   entropy_sum += sqrt(|rval|) * k_cost_delta
///   nzeros += (rval != 0)
///
/// `inv_weights` contains precomputed reciprocals (1/quant_weight) to replace
/// per-coefficient SIMD division with multiplication.
///
/// In pixel-domain mode: writes `error_coeffs[i] = weights[i] * (val - rval)`
/// In coefficient-domain mode: accumulates info_loss stats and k_cost2 penalty.
///
/// Dispatch through `incant!` — picks the best magetypes-generated variant
/// (`_v4` AVX-512 with the `avx512` cargo feature, else `_v3` AVX2, else
/// `_neon`, else `_wasm128`, else `_scalar`) at runtime.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn entropy_estimate_coeffs(
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    inv_weights: &[f32],
    n: usize,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
    pixel_domain: bool,
    error_coeffs: &mut [f32],
) -> EntropyCoeffResult {
    incant!(
        entropy_coeffs_impl(
            block_c,
            block_y,
            weights,
            inv_weights,
            n,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            pixel_domain,
            error_coeffs,
        ),
        [v4, v3, neon, wasm128, scalar]
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_scalar(
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    inv_weights: &[f32],
    n: usize,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
    pixel_domain: bool,
    error_coeffs: &mut [f32],
) -> EntropyCoeffResult {
    let mut entropy_sum = 0.0f32;
    let mut nzeros_sum = 0.0f32;
    let mut info_loss_sum = 0.0f32;
    let mut info_loss2_sum = 0.0f32;

    for i in 0..n {
        let val_in = block_c[i];
        let val_y = block_y[i] * cmap_factor;
        let val = (val_in - val_y) * inv_weights[i] * quant;
        // W44-9ef2819 follow-on (SIMD-parity entropy-001): libjxl
        // `enc_ac_strategy.cc::EstimateEntropy` uses Highway `Round` which
        // is IEEE 754 round-to-nearest-ties-to-even. Rust's `f32::round`
        // (and our `round_f32` helper) rounds ties AWAY from zero, biasing
        // toward more non-zero rounded values and making downstream
        // heuristics fire less consistently. The SIMD path on line 251
        // uses `_mm256_round_ps(ROUND_TO_NEAREST_INT)` which IS ties-to-even
        // — so before this fix scalar and SIMD diverged on the 0.5/-0.5 edge.
        // The W44-9ef2819 sweep fixed 3 sites but missed this one;
        // surfaced by the SIMD-vs-scalar parity infrastructure (eedc1877+
        // fb871c83), entry `entropy-001` in
        // docs/SIMD_PARITY_KNOWN_DIVERGENCES.md.
        let rval = crate::scalarmath::round_ties_even_f32(val);
        let diff = val - rval;

        if pixel_domain {
            error_coeffs[i] = weights[i] * diff;
        }

        let q = rval.abs();
        entropy_sum = crate::scalarmath::mul_add_f32(
            crate::scalarmath::sqrt_f32(q),
            k_cost_delta,
            entropy_sum,
        );
        if q != 0.0 {
            nzeros_sum += 1.0;
        }

        if !pixel_domain {
            let diff_abs = diff.abs();
            info_loss_sum += diff_abs;
            info_loss2_sum = crate::scalarmath::mul_add_f32(diff_abs, diff_abs, info_loss2_sum);
            if q >= 1.5 {
                entropy_sum += k_cost2;
            }
        }
    }

    EntropyCoeffResult {
        entropy_sum,
        nzeros_sum,
        info_loss_sum,
        info_loss2_sum,
    }
}

// ============================================================================
// magetypes-consolidated SIMD implementation
// ============================================================================
//
// Single body, one source of truth. The `#[magetypes(...)]` macro generates
// one `#[arcane]`-wrapped variant per listed tier:
//   - `entropy_coeffs_impl_v4` (x86_64 AVX-512 native 256-bit f32x8,
//     opt-in via the `avx512` cargo feature)
//   - `entropy_coeffs_impl_v3` (x86_64 AVX2, native 256-bit f32x8)
//   - `entropy_coeffs_impl_neon` (aarch64, 2× f32x4 polyfill of f32x8)
//   - `entropy_coeffs_impl_wasm128` (wasm32, 2× f32x4 polyfill of f32x8)
//   - `entropy_coeffs_impl_scalar` (portable scalar fallback)
//
// **FMA-reduction-order preservation (load-bearing).** The pre-consolidation
// AVX2 / NEON / WASM bodies had IDENTICAL per-lane FMA op order (only the
// lane width differed: 8 vs 4 vs 4). The five accumulators
// (`entropy_acc`, `nzeros_acc`, `info_loss_acc`, `info_loss2_acc`,
// `cost2_acc`) are independent and stay independent — they are NOT
// reordered, fused, or grouped differently in this consolidation. The
// `q.sqrt().mul_add(cost_delta_v, entropy_acc)` and
// `diff_abs.mul_add(diff_abs, info_loss2_acc)` chains preserve libjxl's
// `MulAdd(x, y, acc)` order bit-for-bit. The final
// `reduce_add()` on each accumulator + scalar add of `cost2_acc.reduce_add()`
// (only when `!pixel_domain`) matches the prior bodies' tail.
//
// Per W44-9 Sub-chunk B (in flight investigating fused DCT8 entropy
// FMA-order sensitivity) we explicitly DO NOT re-group these accumulators
// — even subtle reduction-order perturbations can affect AC-strategy
// selection and risk bitstream divergence downstream.

#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_impl(
    token: Token,
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    inv_weights: &[f32],
    n: usize,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
    pixel_domain: bool,
    error_coeffs: &mut [f32],
) -> EntropyCoeffResult {
    let cmap_v = f32x8::splat(token, cmap_factor);
    let quant_v = f32x8::splat(token, quant);
    let cost_delta_v = f32x8::splat(token, k_cost_delta);
    let cost2_v = f32x8::splat(token, k_cost2);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let thr_1_5 = f32x8::splat(token, 1.5);

    let mut entropy_acc = f32x8::zero(token);
    let mut nzeros_acc = f32x8::zero(token);
    let mut info_loss_acc = f32x8::zero(token);
    let mut info_loss2_acc = f32x8::zero(token);
    let mut cost2_acc = f32x8::zero(token);

    let chunks = n / 8;
    let simd_n = chunks * 8;
    let block_c_s = &block_c[..simd_n];
    let block_y_s = &block_y[..simd_n];
    let weights_s = &weights[..simd_n];
    let inv_weights_s = &inv_weights[..simd_n];
    for chunk in 0..chunks {
        let base = chunk * 8;

        let bc = f32x8::from_slice(token, &block_c_s[base..]);
        let by_v = f32x8::from_slice(token, &block_y_s[base..]);
        let w = f32x8::from_slice(token, &weights_s[base..]);
        let iw = f32x8::from_slice(token, &inv_weights_s[base..]);

        // val = (block_c - block_y * cmap_factor) * inv_weights * quant
        let adjusted = bc - by_v * cmap_v;
        let val = adjusted * iw * quant_v;

        let rval = val.round();
        let diff = val - rval;

        // Write error coefficients for pixel-domain loss
        if pixel_domain {
            let err = w * diff;
            let out: &mut [f32; 8] = (&mut error_coeffs[base..base + 8]).try_into().unwrap();
            err.store(out);
        }

        // Entropy accumulation: entropy += sqrt(|rval|) * cost_delta
        let q = rval.abs();
        entropy_acc = q.sqrt().mul_add(cost_delta_v, entropy_acc);

        // nzeros: count non-zero rounded values
        let nz_mask = q.simd_ne(zero);
        nzeros_acc += f32x8::blend(nz_mask, one, zero);

        // Coefficient-domain statistics
        if !pixel_domain {
            let diff_abs = diff.abs();
            info_loss_acc += diff_abs;
            info_loss2_acc = diff_abs.mul_add(diff_abs, info_loss2_acc);

            // q >= 1.5 penalty
            let ge_mask = q.simd_ge(thr_1_5);
            cost2_acc += f32x8::blend(ge_mask, cost2_v, zero);
        }
    }

    // Handle remainder with scalar fallback (skip when n is multiple of 8)
    let start = chunks * 8;
    let remainder = if start < n {
        entropy_coeffs_scalar(
            &block_c[start..n],
            &block_y[start..n],
            &weights[start..n],
            &inv_weights[start..n],
            n - start,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            pixel_domain,
            &mut error_coeffs[start..n],
        )
    } else {
        EntropyCoeffResult::ZERO
    };

    let mut entropy_sum = entropy_acc.reduce_add() + remainder.entropy_sum;
    if !pixel_domain {
        entropy_sum += cost2_acc.reduce_add();
    }

    EntropyCoeffResult {
        entropy_sum,
        nzeros_sum: nzeros_acc.reduce_add() + remainder.nzeros_sum,
        info_loss_sum: info_loss_acc.reduce_add() + remainder.info_loss_sum,
        info_loss2_sum: info_loss2_acc.reduce_add() + remainder.info_loss2_sum,
    }
}

// ============================================================================
// Backwards-compat suffixed re-exports
// ============================================================================
//
// Pre-consolidation callers spelled the variants `entropy_coeffs_avx2` /
// `_neon` / `_wasm128`. magetypes' tier names are `_v3` (AVX2) / `_neon` /
// `_wasm128`. Re-export under the historical names so the external API
// (including the per-arch `pub use` re-exports in `lib.rs`) stays stable.

#[cfg(target_arch = "x86_64")]
pub use entropy_coeffs_impl_v3 as entropy_coeffs_avx2;

#[cfg(target_arch = "aarch64")]
pub use entropy_coeffs_impl_neon as entropy_coeffs_neon;

#[cfg(target_arch = "wasm32")]
pub use entropy_coeffs_impl_wasm128 as entropy_coeffs_wasm128;

// ============================================================================
// Shannon entropy computation (P6: histogram entropy for clustering)
// ============================================================================

// fast_log2f polynomial coefficients (shared with mask1x1, adaptive_quant).
// Used by arch-gated Shannon entropy functions and scalar fallback.
const LOG2_P0: f32 = -1.850_383_3e-6;
const LOG2_P1: f32 = 1.428_716;
const LOG2_P2: f32 = 0.742_458_7;
const LOG2_Q0: f32 = 0.990_328_14;
const LOG2_Q1: f32 = 1.009_671_9;
const LOG2_Q2: f32 = 0.174_093_43;

/// Fast log2 approximation. Max relative error ~3e-7. Input must be > 0.
///
/// Uses integer bit manipulation on f32 with a Padé approximant for the
/// fractional part. Matches libjxl's `FastLog2f` from `fast_math-inl.h`.
#[inline(always)]
pub fn fast_log2f(x: f32) -> f32 {
    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits.wrapping_sub(0x3f2a_aaab_u32 as i32);
    let exp_shifted = exp_bits >> 23;
    let mantissa = f32::from_bits((x_bits.wrapping_sub(exp_shifted << 23)) as u32);
    let exp_val = exp_shifted as f32;
    let frac = mantissa - 1.0;
    let num = LOG2_P0 + frac * (LOG2_P1 + frac * LOG2_P2);
    let den = LOG2_Q0 + frac * (LOG2_Q1 + frac * LOG2_Q2);
    num / den + exp_val
}

/// Fast base-2 exponentiation. Max relative error ~3e-7.
///
/// Matches libjxl's `FastPow2f` from `fast_math-inl.h` (line 72).
/// Uses integer bit manipulation for the integer exponent part and a (3,3)
/// rational polynomial for the fractional part.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn fast_pow2f(x: f32) -> f32 {
    let floorx = crate::scalarmath::floor_f32(x);
    // Integer part → IEEE 754 exponent via bit shift
    let exp = f32::from_bits(((floorx as i32 + 127) << 23) as u32);
    let frac = x - floorx;
    // (3,3) rational polynomial for 2^frac, frac in [0, 1)
    // Coefficients from libjxl fast_math-inl.h — must match exactly.
    // Numerator: Horner form
    let mut num = frac + 1.01749063e+01;
    num = num * frac + 4.88687798e+01;
    num = num * frac + 9.85506591e+01;
    num *= exp;
    // Denominator: Horner form
    let mut den = frac * 2.10242958e-01 + (-2.22328856e-02);
    den = den * frac + (-1.94414990e+01);
    den = den * frac + 9.85506633e+01;
    num / den
}

/// Fast power function: `base^exponent`. Max relative error ~3e-5.
///
/// Matches libjxl's `FastPowf` from `fast_math-inl.h` (line 90).
/// Computes `2^(log2(base) * exponent)` using [`fast_log2f`] and [`fast_pow2f`].
/// Input `base` must be > 0.
#[inline(always)]
pub fn fast_powf(base: f32, exponent: f32) -> f32 {
    fast_pow2f(fast_log2f(base) * exponent)
}

/// Shannon entropy (bits) of a histogram of i32 counts.
///
/// CANONICAL ARCH-STABLE KERNEL (2026-08-17): identical bit pattern on
/// every tier — 0-anchored 8-element blocks with the same lane formula
/// (magetypes polyfills `f32x8` lane-pure on 4-wide arches), one virtual
/// 8-lane accumulator (`acc_id = i & 7`), a FIXED scalar combine tree,
/// and an identical scalar f64-free tail. The pre-2026-08-17 hand-written
/// kernels grouped by native register width (AVX2 8 vs NEON 4), which made
/// ANS clustering decisions flip between x86_64 and aarch64 on near-ties.
/// Changing the block width, mapping, combine tree, or tail split changes
/// encoded bytes on every arch — regenerate hash-locks deliberately.
#[inline]
pub fn shannon_entropy_bits(counts: &[i32], total_count: usize) -> f32 {
    if total_count == 0 {
        return 0.0;
    }
    incant!(
        shannon_entropy_impl(counts, total_count),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Scalar Shannon entropy using fast_log2f (historical per-element
/// formula; NOT bit-equal to the canonical kernel's accumulation).
#[inline]
pub fn shannon_entropy_scalar(counts: &[i32], total_count: usize) -> f32 {
    let inv_total = 1.0 / total_count as f32;
    let total_f = total_count as f32;
    let mut entropy = 0.0f32;

    for &count in counts {
        if count > 0 {
            let c = count as f32;
            if c != total_f {
                entropy -= c * fast_log2f(c * inv_total);
            }
        }
    }

    entropy
}

#[magetypes(define(f32x8, i32x8), v4, v3, neon, wasm128, scalar)]
pub fn shannon_entropy_impl(token: Token, counts: &[i32], total_count: usize) -> f32 {
    let inv_total_v = f32x8::splat(token, 1.0 / total_count as f32);
    let total_f_v = f32x8::splat(token, total_count as f32);
    let zero_f = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let mut acc = f32x8::zero(token);

    // fast_log2f constants
    let offset = i32x8::splat(token, 0x3f2a_aaab_u32 as i32);
    let p0 = f32x8::splat(token, LOG2_P0);
    let p1 = f32x8::splat(token, LOG2_P1);
    let p2 = f32x8::splat(token, LOG2_P2);
    let q0 = f32x8::splat(token, LOG2_Q0);
    let q1 = f32x8::splat(token, LOG2_Q1);
    let q2 = f32x8::splat(token, LOG2_Q2);

    let chunks = counts.len() / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let c_i = i32x8::from_slice(token, &counts[base..]);
        let c_f = c_i.to_f32x8();

        let nonzero_mask = c_f.simd_gt(zero_f);
        let not_total_mask = c_f.simd_ne(total_f_v);
        let nz_float = f32x8::blend(nonzero_mask, one, zero_f);
        let nt_float = f32x8::blend(not_total_mask, one, zero_f);
        let valid_mask = nz_float * nt_float;

        let safe_c = f32x8::blend(nonzero_mask, c_f, one);
        let prob = safe_c * inv_total_v;
        let log2_prob = {
            let x_bits: i32x8 = prob.bitcast_i32x8();
            let exp_bits = x_bits - offset;
            let exp_shifted = exp_bits.shr_arithmetic::<23>();
            let mantissa_bits = x_bits - exp_shifted.shl::<23>();
            let mantissa = mantissa_bits.bitcast_f32x8();
            let exp_val = exp_shifted.to_f32x8();
            let frac = mantissa - one;
            let num = frac.mul_add(p2, p1).mul_add(frac, p0);
            let den = frac.mul_add(q2, q1).mul_add(frac, q0);
            num / den + exp_val
        };

        let contribution = c_f * log2_prob * valid_mask;
        acc -= contribution;
    }

    // Canonical combine: FIXED scalar tree over the 8 virtual lanes.
    let mut lanes = [0.0f32; 8];
    acc.store(&mut lanes);
    let s4 = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    let simd_sum = (s4[0] + s4[2]) + (s4[1] + s4[3]);

    // Scalar tail — identical formula on every tier.
    let mut scalar_sum = 0.0f32;
    let inv_total = 1.0 / total_count as f32;
    let total_f = total_count as f32;
    for &count in &counts[chunks * 8..] {
        if count > 0 {
            let c = count as f32;
            if c != total_f {
                scalar_sum -= c * fast_log2f(c * inv_total);
            }
        }
    }

    simd_sum + scalar_sum
}

// ============================================================================
// Tree-learning estimate_bits (probability-floor cost for find_best_split)
// ============================================================================

/// Estimate the coded-bit cost of a symbol histogram (`counts`, `total`).
///
/// CANONICAL ARCH-STABLE KERNEL (2026-08-17): every tier — AVX-512, AVX2,
/// NEON, WASM128, scalar — computes the identical bit pattern:
///
/// * elements are processed in 0-anchored 8-element blocks with the SAME
///   f32 lane formula on every tier (magetypes polyfills `f32x8` as
///   lane-pure `f32x4` pairs on 4-wide arches);
/// * each element feeds virtual accumulator `i & 15` (blocks alternate
///   between two `f32x8` accumulators), so the accumulation grouping no
///   longer depends on the native register width — the pre-2026-08-17
///   hand-written kernels grouped by native width (AVX2 `i & 15` vs NEON
///   `i & 7`), which made x86_64 and aarch64 disagree in the low bits and
///   flipped near-tie tree-learner decisions (3 hash-lock fixtures);
/// * the 16 virtual accumulators combine through one FIXED scalar tree;
/// * the `< 8` tail uses the identical scalar f64 formula on every tier.
///
/// Any change to the block width, accumulator mapping, combine tree, or
/// tail split changes ENCODED BYTES on every arch at once — regenerate the
/// hash-lock sidecar deliberately, never as a side effect.
#[inline]
pub fn estimate_bits_u32(counts: &[u32], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }
    incant!(
        estimate_bits_u32_impl(counts, total),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Scalar reference for [`estimate_bits_u32`]'s historical per-element
/// formula. Sums in iteration order using f64 accumulation. NOT bit-equal
/// to the canonical kernel (different accumulator structure); kept for
/// tests and as documentation of the underlying formula.
#[inline]
pub fn estimate_bits_scalar_f64(counts: &[u32], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let min_prob = 1.0 / 4096.0;
    let mut bits = 0.0;
    for &c in counts {
        if c > 0 {
            let p = (c as f64 / total_f).max(min_prob);
            bits -= c as f64 * fast_log2f(p as f32) as f64;
        }
    }
    bits
}

#[magetypes(define(f32x8, i32x8, u32x8), v4, v3, neon, wasm128, scalar)]
pub fn estimate_bits_u32_impl(token: Token, counts: &[u32], total: u32) -> f64 {
    let total_f = total as f32;
    let inv_total = f32x8::splat(token, 1.0 / total_f);
    let min_prob = f32x8::splat(token, 1.0 / 4096.0);
    let zero_f = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);

    // fast_log2f constants
    let offset = i32x8::splat(token, 0x3f2a_aaab_u32 as i32);
    let p0 = f32x8::splat(token, LOG2_P0);
    let p1 = f32x8::splat(token, LOG2_P1);
    let p2 = f32x8::splat(token, LOG2_P2);
    let q0 = f32x8::splat(token, LOG2_Q0);
    let q1 = f32x8::splat(token, LOG2_Q1);
    let q2 = f32x8::splat(token, LOG2_Q2);

    // fast_log2f (log2 via integer bit manipulation) is inlined at each
    // use site below — lane-pure, so every tier computes identical
    // per-element values.

    let mut acc_a = f32x8::zero(token);
    let mut acc_b = f32x8::zero(token);
    let blocks = counts.len() / 8;
    let mut blk = 0usize;
    while blk + 1 < blocks {
        let base_a = blk * 8;
        let base_b = base_a + 8;
        let ca = u32x8::from_slice(token, &counts[base_a..base_a + 8])
            .bitcast_i32x8()
            .to_f32x8();
        let cb = u32x8::from_slice(token, &counts[base_b..base_b + 8])
            .bitcast_i32x8()
            .to_f32x8();
        let nza = ca.simd_gt(zero_f);
        let nzb = cb.simd_gt(zero_f);
        let sa = f32x8::blend(nza, ca, one);
        let sb = f32x8::blend(nzb, cb, one);
        let pa = (sa * inv_total).max(min_prob);
        let pb = (sb * inv_total).max(min_prob);
        let la = {
            let x_bits: i32x8 = pa.bitcast_i32x8();
            let exp_bits = x_bits - offset;
            let exp_shifted = exp_bits.shr_arithmetic::<23>();
            let mantissa_bits = x_bits - exp_shifted.shl::<23>();
            let mantissa = mantissa_bits.bitcast_f32x8();
            let exp_val = exp_shifted.to_f32x8();
            let frac = mantissa - one;
            let num = frac.mul_add(p2, p1).mul_add(frac, p0);
            let den = frac.mul_add(q2, q1).mul_add(frac, q0);
            num / den + exp_val
        };
        let lb = {
            let x_bits: i32x8 = pb.bitcast_i32x8();
            let exp_bits = x_bits - offset;
            let exp_shifted = exp_bits.shr_arithmetic::<23>();
            let mantissa_bits = x_bits - exp_shifted.shl::<23>();
            let mantissa = mantissa_bits.bitcast_f32x8();
            let exp_val = exp_shifted.to_f32x8();
            let frac = mantissa - one;
            let num = frac.mul_add(p2, p1).mul_add(frac, p0);
            let den = frac.mul_add(q2, q1).mul_add(frac, q0);
            num / den + exp_val
        };
        acc_a -= ca * la;
        acc_b -= cb * lb;
        blk += 2;
    }
    if blk < blocks {
        // Trailing lone block has an even index — virtual accs 0-7 (acc_a).
        let base = blk * 8;
        let c = u32x8::from_slice(token, &counts[base..base + 8])
            .bitcast_i32x8()
            .to_f32x8();
        let nz = c.simd_gt(zero_f);
        let sc = f32x8::blend(nz, c, one);
        let pc = (sc * inv_total).max(min_prob);
        let lc = {
            let x_bits: i32x8 = pc.bitcast_i32x8();
            let exp_bits = x_bits - offset;
            let exp_shifted = exp_bits.shr_arithmetic::<23>();
            let mantissa_bits = x_bits - exp_shifted.shl::<23>();
            let mantissa = mantissa_bits.bitcast_f32x8();
            let exp_val = exp_shifted.to_f32x8();
            let frac = mantissa - one;
            let num = frac.mul_add(p2, p1).mul_add(frac, p0);
            let den = frac.mul_add(q2, q1).mul_add(frac, q0);
            num / den + exp_val
        };
        acc_a -= c * lc;
    }

    // Canonical combine: virtual acc j pairs with j+8, then a fixed scalar
    // tree — identical on every tier.
    let mut arr_a = [0.0f32; 8];
    let mut arr_b = [0.0f32; 8];
    acc_a.store(&mut arr_a);
    acc_b.store(&mut arr_b);
    let mut s8 = [0.0f32; 8];
    for j in 0..8 {
        s8[j] = arr_a[j] + arr_b[j];
    }
    let s4 = [s8[0] + s8[4], s8[1] + s8[5], s8[2] + s8[6], s8[3] + s8[7]];
    let acc_f32 = (s4[0] + s4[2]) + (s4[1] + s4[3]);
    let mut bits = acc_f32 as f64;

    // Scalar f64 tail (< 8 entries) — identical formula on every tier.
    let total_f64 = total as f64;
    let min_prob_f64 = 1.0 / 4096.0;
    for &c in &counts[blocks * 8..] {
        if c > 0 {
            let p = (c as f64 / total_f64).max(min_prob_f64);
            bits -= c as f64 * fast_log2f(p as f32) as f64;
        }
    }

    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    extern crate std;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Verify SIMD matches scalar for pixel-domain mode.
    #[test]
    fn test_entropy_coeffs_pixel_domain() {
        let n = 64;
        let block_c: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 - 20.0) * 0.1).collect();
        let block_y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 - 15.0) * 0.1).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.01 + (i as f32) * 0.005).collect();
        let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

        let cmap_factor = 0.15f32;
        let quant = 3.5f32;
        let k_cost_delta = 5.335f32;
        let k_cost2 = 4.463f32;

        // Reference: scalar
        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            n,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            true,
            &mut error_ref,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut error_simd = vec![0.0f32; n];
                let simd_result = entropy_estimate_coeffs(
                    &block_c,
                    &block_y,
                    &weights,
                    &inv_weights,
                    n,
                    cmap_factor,
                    quant,
                    k_cost_delta,
                    k_cost2,
                    true,
                    &mut error_simd,
                );
                let rel_eps = 0.005;
                let entropy_rel = (simd_result.entropy_sum - ref_result.entropy_sum).abs()
                    / ref_result.entropy_sum.abs();
                assert!(
                    entropy_rel < rel_eps,
                    "entropy_sum rel_err={entropy_rel:.4} [{perm}]"
                );
                let nz_rel = (simd_result.nzeros_sum - ref_result.nzeros_sum).abs()
                    / ref_result.nzeros_sum.abs().max(1.0);
                assert!(nz_rel < 0.05, "nzeros_sum rel_err={nz_rel:.4} [{perm}]");
                let mut max_err = 0.0f32;
                for i in 0..n {
                    max_err = max_err.max((error_simd[i] - error_ref[i]).abs());
                }
                assert!(
                    max_err < 0.5,
                    "Error coeffs max diff: {max_err:.2e} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    /// Verify SIMD matches scalar for coefficient-domain mode.
    #[test]
    fn test_entropy_coeffs_coeff_domain() {
        let n = 64;
        let block_c: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3 - 40.0) * 0.05).collect();
        let block_y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.9 - 30.0) * 0.05).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.02 + (i as f32) * 0.003).collect();
        let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

        let cmap_factor = 0.0f32;
        let quant = 5.0f32;
        let k_cost_delta = 5.335f32;
        let k_cost2 = 4.463f32;

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            n,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
            false,
            &mut error_ref,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut error_simd = vec![0.0f32; n];
                let simd_result = entropy_estimate_coeffs(
                    &block_c,
                    &block_y,
                    &weights,
                    &inv_weights,
                    n,
                    cmap_factor,
                    quant,
                    k_cost_delta,
                    k_cost2,
                    false,
                    &mut error_simd,
                );
                let rel_eps = 0.005;
                let entropy_rel = (simd_result.entropy_sum - ref_result.entropy_sum).abs()
                    / ref_result.entropy_sum.abs();
                assert!(entropy_rel < rel_eps, "entropy_sum [{perm}]");
                let nz_rel = (simd_result.nzeros_sum - ref_result.nzeros_sum).abs()
                    / ref_result.nzeros_sum.abs().max(1.0);
                assert!(nz_rel < 0.05, "nzeros_sum [{perm}]");
                let il_rel = (simd_result.info_loss_sum - ref_result.info_loss_sum).abs()
                    / ref_result.info_loss_sum.abs().max(1.0);
                assert!(il_rel < rel_eps, "info_loss_sum [{perm}]");
                let il2_rel = (simd_result.info_loss2_sum - ref_result.info_loss2_sum).abs()
                    / ref_result.info_loss2_sum.abs().max(1.0);
                assert!(il2_rel < rel_eps, "info_loss2_sum [{perm}]");
            },
        );
        std::eprintln!("{report}");
    }

    /// Test with non-multiple-of-8 sizes (remainder handling).
    #[test]
    fn test_entropy_coeffs_remainder() {
        let n = 67;
        let block_c: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let block_y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.08 - 2.5).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.01 + (i as f32) * 0.002).collect();
        let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            n,
            0.2,
            4.0,
            5.335,
            4.463,
            true,
            &mut error_ref,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut error_simd = vec![0.0f32; n];
                let simd_result = entropy_estimate_coeffs(
                    &block_c,
                    &block_y,
                    &weights,
                    &inv_weights,
                    n,
                    0.2,
                    4.0,
                    5.335,
                    4.463,
                    true,
                    &mut error_simd,
                );
                let rel_eps = 0.005;
                let entropy_rel = (simd_result.entropy_sum - ref_result.entropy_sum).abs()
                    / ref_result.entropy_sum.abs().max(1.0);
                assert!(entropy_rel < rel_eps, "entropy_sum [{perm}]");
                let nz_rel = (simd_result.nzeros_sum - ref_result.nzeros_sum).abs()
                    / ref_result.nzeros_sum.abs().max(1.0);
                assert!(nz_rel < 0.05, "nzeros_sum [{perm}]");
                let max_err = error_simd
                    .iter()
                    .zip(error_ref.iter())
                    .take(n)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 0.01,
                    "Error coeffs max diff: {max_err:.2e} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    /// Test with large blocks (DCT64x64 = 4096 coefficients).
    #[test]
    fn test_entropy_coeffs_large_block() {
        let n = 4096;
        let block_c: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() * 5.0).collect();
        let block_y: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos() * 4.0).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.005 + (i as f32) * 0.001).collect();
        let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

        let mut error_ref = vec![0.0f32; n];
        let ref_result = entropy_coeffs_scalar(
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            n,
            0.1,
            2.0,
            5.335,
            4.463,
            true,
            &mut error_ref,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut error_simd = vec![0.0f32; n];
                let simd_result = entropy_estimate_coeffs(
                    &block_c,
                    &block_y,
                    &weights,
                    &inv_weights,
                    n,
                    0.1,
                    2.0,
                    5.335,
                    4.463,
                    true,
                    &mut error_simd,
                );

                // Large block: use relative tolerance
                let rel_eps = 0.005;
                let entropy_rel = (simd_result.entropy_sum - ref_result.entropy_sum).abs()
                    / ref_result.entropy_sum.abs();
                assert!(
                    entropy_rel < rel_eps,
                    "entropy_sum: SIMD={}, ref={}, rel_err={:.4}% [{perm}]",
                    simd_result.entropy_sum,
                    ref_result.entropy_sum,
                    entropy_rel * 100.0
                );

                let max_err = error_simd
                    .iter()
                    .zip(error_ref.iter())
                    .take(n)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 1e-3,
                    "Error coeffs max diff: {:.2e} [{perm}]",
                    max_err
                );
            },
        );
        std::eprintln!("{report}");
    }

    // =====================================================================
    // Shannon entropy tests
    // =====================================================================

    /// Reference Shannon entropy using f32::log2 (not fast_log2f).
    fn reference_shannon_entropy(counts: &[i32], total_count: usize) -> f32 {
        if total_count == 0 {
            return 0.0;
        }
        let inv_total = 1.0 / total_count as f32;
        let total_f = total_count as f32;
        let mut entropy = 0.0f32;
        for &count in counts {
            if count > 0 {
                let c = count as f32;
                if c != total_f {
                    entropy -= c * (c * inv_total).log2();
                }
            }
        }
        entropy
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution: entropy = n * log2(n) bits total
        let counts = [100i32, 100, 100, 100, 0, 0, 0, 0];
        let total = 400;
        let ref_ent = reference_shannon_entropy(&counts, total);
        let simd_ent = shannon_entropy_bits(&counts, total);
        let scalar_ent = shannon_entropy_scalar(&counts, total);

        // Expected: 400 * log2(4) = 800
        assert!((ref_ent - 800.0).abs() < 0.1, "ref = {ref_ent}");
        assert!(
            (simd_ent - ref_ent).abs() < 0.5,
            "simd={simd_ent} ref={ref_ent}"
        );
        assert!(
            (scalar_ent - ref_ent).abs() < 0.5,
            "scalar={scalar_ent} ref={ref_ent}"
        );
    }

    #[test]
    fn test_shannon_entropy_single_symbol() {
        // All counts in one symbol: entropy = 0
        let counts = [1000i32, 0, 0, 0, 0, 0, 0, 0];
        let total = 1000;
        let ent = shannon_entropy_bits(&counts, total);
        assert!(ent.abs() < 0.01, "entropy should be 0, got {ent}");
    }

    #[test]
    fn test_shannon_entropy_realistic_histogram() {
        // Realistic distribution like AC coefficient magnitudes
        let mut counts = alloc::vec![0i32; 64];
        counts[0] = 5000; // lots of zeros (but treated as symbol 0)
        counts[1] = 2000;
        counts[2] = 1000;
        counts[3] = 500;
        counts[4] = 200;
        counts[5] = 100;
        counts[6] = 50;
        counts[7] = 20;
        let total: usize = counts.iter().map(|&c| c as usize).sum();

        let ref_ent = reference_shannon_entropy(&counts, total);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let simd_ent = shannon_entropy_bits(&counts, total);
                let rel_err = (simd_ent - ref_ent).abs() / ref_ent.abs().max(1.0);
                assert!(
                    rel_err < 0.001,
                    "Shannon entropy: simd={simd_ent}, ref={ref_ent}, rel_err={rel_err:.4} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_shannon_entropy_large_alphabet() {
        // Large alphabet (256 symbols) with Zipf-like distribution
        let mut counts = alloc::vec![0i32; 256];
        let mut total = 0usize;
        for (i, count) in counts.iter_mut().enumerate() {
            *count = 10000 / (i as i32 + 1);
            total += *count as usize;
        }

        let ref_ent = reference_shannon_entropy(&counts, total);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let simd_ent = shannon_entropy_bits(&counts, total);
                let rel_err = (simd_ent - ref_ent).abs() / ref_ent.abs().max(1.0);
                assert!(
                    rel_err < 0.001,
                    "Large alphabet: simd={simd_ent}, ref={ref_ent}, rel_err={rel_err:.4} [{perm}]"
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_shannon_entropy_empty() {
        let counts = [0i32; 8];
        let ent = shannon_entropy_bits(&counts, 0);
        assert_eq!(ent, 0.0);
    }

    /// Verify SIMD `estimate_bits_u32` matches scalar within tight FP tolerance.
    ///
    /// The probability-floor `max(p, 1/4096)` formulation is exercised across:
    /// - uniform histograms (no floor active)
    /// - sparse histograms (floor very active)
    /// - realistic photo-like Geometric distributions
    /// - the exact node-shape used in `find_best_split` (256-entry padded buf).
    #[test]
    fn test_estimate_bits_simd_matches_scalar_uniform() {
        let counts = [100u32; 8];
        let total = 800;
        let ref_v = estimate_bits_scalar_f64(&counts, total);
        let simd_v = estimate_bits_u32(&counts, total);
        let rel = ((simd_v - ref_v).abs() / ref_v.abs()).max(0.0);
        assert!(
            rel < 1e-5,
            "uniform: scalar={ref_v}, simd={simd_v}, rel={rel}"
        );
    }

    #[test]
    fn test_estimate_bits_simd_matches_scalar_sparse() {
        // Most zeros — exercises the c > 0 branch in scalar vs the
        // zero-masked SIMD multiply.
        let mut counts = vec![0u32; 256];
        counts[3] = 700;
        counts[17] = 50;
        counts[19] = 2;
        counts[200] = 1;
        let total = 753;
        let ref_v = estimate_bits_scalar_f64(&counts, total);
        let simd_v = estimate_bits_u32(&counts, total);
        let rel = ((simd_v - ref_v).abs() / ref_v.abs()).max(0.0);
        assert!(
            rel < 1e-4,
            "sparse: scalar={ref_v}, simd={simd_v}, rel={rel}"
        );
    }

    #[test]
    fn test_estimate_bits_simd_matches_scalar_floor_active() {
        // Floor triggers when c/total < 1/4096, i.e., c < total/4096.
        // total=10_000, c=1 → 1/10000 < 1/4096 → floor applies.
        let mut counts = vec![0u32; 64];
        for i in 0..32 {
            counts[i] = 1; // each below floor
        }
        counts[32] = 10_000 - 32;
        let total: u32 = counts.iter().sum();
        let ref_v = estimate_bits_scalar_f64(&counts, total);
        let simd_v = estimate_bits_u32(&counts, total);
        let rel = ((simd_v - ref_v).abs() / ref_v.abs()).max(0.0);
        assert!(
            rel < 1e-4,
            "floor-active: scalar={ref_v}, simd={simd_v}, rel={rel}"
        );
    }

    #[test]
    fn test_estimate_bits_simd_matches_scalar_geometric() {
        // Geometric-decay distribution (photo residual shape)
        let mut counts = vec![0u32; 256];
        let mut x = 50_000u32;
        for i in 0..200 {
            counts[i] = x;
            x = x * 7 / 10; // λ ≈ 0.7
            if x == 0 {
                break;
            }
        }
        let total: u32 = counts.iter().sum();
        let ref_v = estimate_bits_scalar_f64(&counts, total);
        let simd_v = estimate_bits_u32(&counts, total);
        let rel = ((simd_v - ref_v).abs() / ref_v.abs()).max(0.0);
        assert!(
            rel < 1e-4,
            "geometric: scalar={ref_v}, simd={simd_v}, rel={rel}"
        );
    }

    #[test]
    fn test_estimate_bits_simd_matches_scalar_padded256() {
        // Exact shape used by find_best_split: HISTO_PADDED = 256, many
        // zero gaps from sparse token distribution after node split.
        let mut counts = vec![0u32; 256];
        // Cluster near low tokens (typical after Hybrid encoding of
        // residual values close to predictor)
        let weights = [4321, 2114, 998, 543, 287, 121, 88, 41, 19, 11, 7, 3, 1];
        for (i, &w) in weights.iter().enumerate() {
            counts[i * 2] = w;
        }
        let total: u32 = counts.iter().sum();
        let ref_v = estimate_bits_scalar_f64(&counts, total);
        let simd_v = estimate_bits_u32(&counts, total);
        let rel = ((simd_v - ref_v).abs() / ref_v.abs()).max(1e-12);
        assert!(
            rel < 1e-4,
            "padded256: scalar={ref_v}, simd={simd_v}, rel={rel}"
        );
    }

    #[test]
    fn test_estimate_bits_simd_handles_tail() {
        // 13 nonzero entries: 8 in chunks of 8, then 5 in tail
        let counts = [50u32, 30, 20, 10, 8, 6, 4, 2, 100, 90, 80, 70, 60];
        let total: u32 = counts.iter().sum();
        let ref_v = estimate_bits_scalar_f64(&counts, total);
        let simd_v = estimate_bits_u32(&counts, total);
        let rel = ((simd_v - ref_v).abs() / ref_v.abs()).max(0.0);
        assert!(rel < 1e-4, "tail: scalar={ref_v}, simd={simd_v}, rel={rel}");
    }

    #[test]
    fn test_estimate_bits_total_zero() {
        let counts = [0u32; 256];
        assert_eq!(estimate_bits_u32(&counts, 0), 0.0);
        assert_eq!(estimate_bits_scalar_f64(&counts, 0), 0.0);
    }

    #[test]
    fn test_fast_pow2f_accuracy() {
        // Test exact powers of 2
        assert!((fast_pow2f(0.0) - 1.0).abs() < 1e-5);
        assert!((fast_pow2f(1.0) - 2.0).abs() < 1e-4);
        assert!((fast_pow2f(3.0) - 8.0).abs() < 1e-3);
        assert!((fast_pow2f(-1.0) - 0.5).abs() < 1e-5);

        // Test fractional exponents
        let val = fast_pow2f(0.5);
        let expected = core::f32::consts::SQRT_2;
        assert!(
            (val - expected).abs() / expected < 5e-7,
            "2^0.5: got {val}, expected {expected}"
        );
    }

    #[test]
    fn test_fast_powf_accuracy() {
        // Test basic powers
        let val = fast_powf(2.0, 3.0);
        assert!(
            (val - 8.0).abs() / 8.0 < 5e-5,
            "2^3: got {val}, expected 8.0"
        );

        // Test sRGB TF: (0.5)^2.4
        let base = 0.5f32;
        let exact = base.powf(2.4);
        let fast = fast_powf(base, 2.4);
        assert!(
            (fast - exact).abs() / exact < 5e-5,
            "0.5^2.4: got {fast}, expected {exact}"
        );

        // Test ratio^K_POW (the compute_scaled_constants case)
        let ratio = 1.5f32;
        let exact = ratio.powf(0.337);
        let fast = fast_powf(ratio, 0.337);
        assert!(
            (fast - exact).abs() / exact < 5e-5,
            "1.5^0.337: got {fast}, expected {exact}"
        );
    }
}

#[cfg(test)]
mod expanded_coverage {
    use super::*;
    use crate::test_helpers::*;
    use alloc::format;
    use alloc::vec::Vec;

    /// entropy_estimate_coeffs across edge battery (n=64 representative block).
    /// Both pixel_domain modes (true/false).
    ///
    /// Originally `#[ignore]`d as `entropy-001`: scalar used `f32::round`
    /// (ties AWAY from zero) while SIMD uses `_mm256_round_ps ROUND_TO_NEAREST_INT`
    /// (ties to even). Closed by switching scalar to `round_ties_even_f32`,
    /// matching libjxl `enc_ac_strategy.cc::EstimateEntropy` which uses
    /// Highway `Round` (IEEE 754 ties-to-even). W44-9ef2819 fixed 3 sites
    /// and missed this one; the SIMD-parity harness (eedc1877+fb871c83)
    /// caught it.
    #[test]
    fn entropy_coeffs_scalar_vs_dispatch_edge_battery() {
        for case in f32_edge_battery(64) {
            // Skip empty + `large_pos` (1e20 inputs): the 8-lane SIMD
            // accumulator and the scalar 1-element accumulator produce
            // sub-ULP-different f32 sums at e12 magnitude (~6e5 absolute
            // diff at 3.3e12 sum). Mirrors the filter pattern in
            // quantize-001/dct64/idct32 — documented as expected
            // ordering noise on out-of-domain inputs.
            if case.data.is_empty() || case.label.starts_with("large_pos") {
                continue;
            }
            let block_c = case.data.clone();
            // Use a different seeded distribution for block_y so cmap_factor*y is nontrivial.
            let block_y = gen_f32(case.data.len() as u64 ^ 0x1234, 64, 1.0);
            let weights: Vec<f32> = (0..64).map(|i| 0.5 + (i as f32) * 0.1).collect();
            let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();

            for &(cmap, k_cost2, pd) in &[(0.0_f32, 0.0_f32, true), (0.35, 0.5, false)] {
                let mut ref_err = alloc::vec![0.0_f32; 64];
                let ref_res = entropy_coeffs_scalar(
                    &block_c,
                    &block_y,
                    &weights,
                    &inv_weights,
                    64,
                    cmap,
                    2.5,
                    5.335,
                    k_cost2,
                    pd,
                    &mut ref_err,
                );

                run_dispatch_parity(|perm| {
                    let mut act_err = alloc::vec![0.0_f32; 64];
                    let act_res = entropy_estimate_coeffs(
                        &block_c,
                        &block_y,
                        &weights,
                        &inv_weights,
                        64,
                        cmap,
                        2.5,
                        5.335,
                        k_cost2,
                        pd,
                        &mut act_err,
                    );
                    if pd {
                        assert_f32_slice_close_ulps_abs(
                            &ref_err,
                            &act_err,
                            32,
                            1e-4,
                            perm,
                            &format!("err::pd={pd}::{}", case.label),
                        );
                    }
                    let e_diff = (ref_res.entropy_sum - act_res.entropy_sum).abs();
                    assert!(
                        e_diff < 1e-2,
                        "entropy diverged ({}): ref={} act={} diff={} perm={perm}",
                        case.label,
                        ref_res.entropy_sum,
                        act_res.entropy_sum,
                        e_diff
                    );
                    let nz_diff = (ref_res.nzeros_sum - act_res.nzeros_sum).abs();
                    assert!(
                        nz_diff < 1.0,
                        "nzeros diverged ({}): ref={} act={} perm={perm}",
                        case.label,
                        ref_res.nzeros_sum,
                        act_res.nzeros_sum
                    );
                });
            }
        }
    }

    /// shannon_entropy_bits across multiple count-distribution patterns.
    #[test]
    fn shannon_entropy_scalar_vs_dispatch_distributions() {
        let cases: alloc::vec::Vec<(alloc::string::String, alloc::vec::Vec<i32>)> = alloc::vec![
            (
                alloc::string::String::from("uniform_16"),
                alloc::vec![100_i32; 16]
            ),
            (alloc::string::String::from("skewed_one_big"), {
                let mut v = alloc::vec![1_i32; 32];
                v[0] = 1000;
                v
            }),
            (
                alloc::string::String::from("all_zero_one"),
                alloc::vec![1_i32; 256]
            ),
            (alloc::string::String::from("alternating_zero"), {
                let mut v = alloc::vec![0_i32; 64];
                for i in (0..64).step_by(2) {
                    v[i] = 10;
                }
                v
            }),
            (alloc::string::String::from("single_nonzero"), {
                let mut v = alloc::vec![0_i32; 64];
                v[7] = 500;
                v
            }),
        ];
        for (label, counts) in &cases {
            let total: usize = counts.iter().map(|&c| c as usize).sum();
            if total == 0 {
                continue;
            }
            let ref_bits = shannon_entropy_scalar(counts, total);
            run_dispatch_parity(|perm| {
                let act_bits = shannon_entropy_bits(counts, total);
                let diff = (ref_bits - act_bits).abs();
                assert!(
                    diff < 1e-2,
                    "shannon_entropy diverged ({label}): ref={ref_bits} act={act_bits} \
                     diff={diff} perm={perm}"
                );
            });
        }
    }
}
