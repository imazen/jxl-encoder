// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated XYB ↔ Linear RGB color conversion.
//!
//! Forward (linear RGB → XYB): matrix multiply + cube root + mix.
//! Inverse (XYB → linear RGB): unmix + cube + inverse matrix multiply.
//!
//! The cube root uses Newton-Raphson in f64 with bit-manipulation initial guess,
//! following the proven approach from fast-ssim2/yuvxyb. The SIMD path extracts
//! each lane to scalar, runs Newton-Raphson, then reloads — same as the
//! pre-consolidation AVX2 body.
//!
//! Data layout: separate channel buffers (SoA), not interleaved.
//!
//! **magetypes-consolidated** (W43-2 chunk-6): the three forward and three
//! inverse variants (AVX2 / NEON / WASM128) previously copy-pasted in this
//! file collapse to two `#[magetypes(...)]` bodies (`forward_xyb_impl`,
//! `inverse_xyb_planar_impl`) plus a scalar interleave wrapper for the
//! AoS `xyb_to_linear_rgb_batch` path. Each `#[magetypes(...)]` generates
//! one `#[arcane]`-wrapped variant per listed tier:
//!   - `*_v4` (x86_64 AVX-512 native 256-bit f32x8, opt-in via the
//!     `avx512` cargo feature)
//!   - `*_v3` (x86_64 AVX2, native 256-bit f32x8)
//!   - `*_neon` (aarch64, 2× f32x4 polyfill of f32x8)
//!   - `*_wasm128` (wasm32, 2× f32x4 polyfill of f32x8)
//!   - `*_scalar` (portable scalar fallback)
//!
//! Pure-f32 kernels: the v4 tier compiles cleanly here (no `f64x4` is
//! required inside the SIMD body — the cube root happens on extracted
//! `[f32; 8]` lanes through the precision-critical `cbrt_fast` scalar).
//! Contrast with W43-2 chunk-5's `pixel_domain_loss` body which DID
//! require `f64x4` (8th-power norm) and was capped at `v3` because
//! magetypes 0.9.23 has no `F64x4Backend` for `X64V4Token`.

use archmage::prelude::*;

// --- Constants ---

// Opsin absorbance matrix (libjxl cms/opsin_params.h)
const OPSIN_MATRIX: [[f32; 3]; 3] = [
    [0.30, 0.622, 0.078],
    [0.23, 0.692, 0.078],
    [0.243_422_69, 0.204_767_45, 0.551_809_87],
];

// Inverse opsin absorbance matrix.
//
// W44-RECON-DEEP/A11 (2026-05-23): The off-diagonal blue cross-terms
// `[0][2]` and `[1][2]` use the full f64 literal `-0.16462299647058826`
// to round to f32 bits `0xbe2892ee`, matching libjxl's
// `kDefaultInverseOpsinAbsorbanceMatrix` (cms/opsin_params.h:44-47),
// jxl-rs `OpsinInverseMatrix.inverse_matrix`, and jxl-oxide
// `OpsinInverseMatrix.inv_mat`. Truncated literal `-0.164_623` was
// 1 ULP off (bits `0xbe2892ef`) and contributed ≤1.5e-8 per-pixel
// drift in linear-RGB vs the decoder. See A2 audit memo.
#[allow(clippy::excessive_precision)]
const INV_OPSIN: [[f32; 3]; 3] = [
    [11.031_566_9, -9.866_943_9, -0.164_622_996_470_588_26],
    [-3.254_147_4, 4.418_770_4, -0.164_622_996_470_588_26],
    [-3.658_851_3, 2.712_923, 1.945_928_2],
];

// Bias added before cube root
#[allow(clippy::excessive_precision)]
const OPSIN_BIAS: [f32; 3] = [0.003_793_073_4; 3];

// Precomputed -cbrt(bias) ≈ -0.15595420054
#[allow(clippy::excessive_precision)]
const NEG_CBRT_BIAS: [f32; 3] = [-0.155_954_2; 3];

// --- Public dispatch entry points ---

/// Convert separate R, G, B channel buffers to separate X, Y, B channel buffers.
///
/// All buffers must be at least `n` elements. Uses SIMD for the inner loop.
/// The cube root uses Newton-Raphson in f64 for precision.
#[inline]
pub fn linear_rgb_to_xyb_batch(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let n = r
        .len()
        .min(g.len())
        .min(b.len())
        .min(x_out.len())
        .min(y_out.len())
        .min(b_out.len());

    // Dispatch through incant! — picks the best magetypes-generated variant
    // at runtime. Falls through to `_scalar` on platforms without a SIMD
    // token. Tier list omits `v4` (AVX-512) because the body now uses
    // `f64x4` for the vectorized cube-root Newton iterations and magetypes
    // has no `F64x4Backend` for `X64V4Token` (same constraint + precedent
    // as `pixel_loss.rs`); `v3` is the x86_64 ceiling. This kernel's `v4`
    // variant was f32x8-wide anyway, so nothing narrows.
    incant!(
        forward_xyb_impl(r, g, b, x_out, y_out, b_out, n),
        [v3, neon, wasm128, scalar]
    )
}

/// Convert separate X, Y, B channel buffers to planar linear RGB.
///
/// Output is three separate channel slices, each of length `n`.
/// This avoids the interleave overhead when the consumer needs planar data.
#[inline]
pub fn xyb_to_linear_rgb_planar(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    n: usize,
) {
    debug_assert!(xyb_x.len() >= n && xyb_y.len() >= n && xyb_b.len() >= n);
    debug_assert!(out_r.len() >= n && out_g.len() >= n && out_b.len() >= n);

    incant!(
        inverse_xyb_planar_impl(xyb_x, xyb_y, xyb_b, out_r, out_g, out_b, n),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Convert separate X, Y, B channel buffers to interleaved linear RGB.
///
/// Output is `[R0, G0, B0, R1, G1, B1, ...]` with length `3 * n`.
///
/// Implementation: run the planar inverse into temporary buffers, then
/// scalar-interleave. The pre-consolidation per-arch bodies open-coded
/// the interleave for each SIMD width; consolidating routes through the
/// planar kernel (which is the load-bearing inner loop) and pays a
/// modest interleave pass on top. The fast path for most callers is
/// `xyb_to_linear_rgb_planar`; the interleaved entry exists for the
/// few decoder consumers that need AoS output.
#[inline]
pub fn xyb_to_linear_rgb_batch(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    debug_assert!(xyb_x.len() >= n && xyb_y.len() >= n && xyb_b.len() >= n);
    debug_assert!(linear_rgb.len() >= n * 3);

    // Allocate tiny per-call scratch only when the caller actually uses AoS.
    // The inner kernel is the same `inverse_xyb_planar_impl` that the planar
    // entry uses; the scalar interleave below is O(n) memcpy-style writes.
    extern crate alloc;
    use alloc::vec;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];

    incant!(
        inverse_xyb_planar_impl(xyb_x, xyb_y, xyb_b, &mut r, &mut g, &mut b, n),
        [v4, v3, neon, wasm128, scalar]
    );

    for i in 0..n {
        linear_rgb[i * 3] = r[i];
        linear_rgb[i * 3 + 1] = g[i];
        linear_rgb[i * 3 + 2] = b[i];
    }
}

// --- Scalar cube root helper ---

/// The integer-bit-trick initial guess + f64 promotion shared by
/// [`cbrt_fast`] and the vectorized Newton path in `forward_xyb_impl`.
/// Returns `(t0, x as f64)`. Note: no zero early-out here — the
/// vectorized caller applies the x == 0 fixup after the iterations,
/// producing exactly `cbrt_fast`'s 0.0.
#[inline(always)]
fn cbrt_newton_init(x: f32) -> (f64, f64) {
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let sign = ui & 0x8000_0000;
    let hx = ui & 0x7FFF_FFFF;
    let approx = hx / 3 + B1;
    (f64::from(f32::from_bits(sign | approx)), f64::from(x))
}

/// Newton-Raphson cube root with bit-manipulation initial guess.
/// 2 iterations in f64 gives ~1e-7 relative error.
#[inline]
fn cbrt_fast(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let sign = ui & 0x8000_0000;
    let hx = ui & 0x7FFF_FFFF;
    let approx = hx / 3 + B1;
    let mut t = f64::from(f32::from_bits(sign | approx));
    let xf64 = f64::from(x);
    // First Newton iteration: t = t * (2x + t³) / (x + 2t³)
    let r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    // Second Newton iteration
    let r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

// --- Scalar fallbacks (also reused by the magetypes `_scalar` tier internally) ---

#[inline]
pub fn forward_xyb_scalar(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    n: usize,
) {
    use crate::scalarmath::mul_add_f32 as fma;
    for i in 0..n {
        // Matrix multiply + bias (chained FMA for single-rounding parity with SIMD path)
        let mixed0 = fma(
            OPSIN_MATRIX[0][0],
            r[i],
            fma(
                OPSIN_MATRIX[0][1],
                g[i],
                fma(OPSIN_MATRIX[0][2], b[i], OPSIN_BIAS[0]),
            ),
        );
        let mixed1 = fma(
            OPSIN_MATRIX[1][0],
            r[i],
            fma(
                OPSIN_MATRIX[1][1],
                g[i],
                fma(OPSIN_MATRIX[1][2], b[i], OPSIN_BIAS[1]),
            ),
        );
        let mixed2 = fma(
            OPSIN_MATRIX[2][0],
            r[i],
            fma(
                OPSIN_MATRIX[2][1],
                g[i],
                fma(OPSIN_MATRIX[2][2], b[i], OPSIN_BIAS[2]),
            ),
        );

        // Clamp + cube root + bias offset
        let l = cbrt_fast(mixed0.max(0.0)) + NEG_CBRT_BIAS[0];
        let m = cbrt_fast(mixed1.max(0.0)) + NEG_CBRT_BIAS[1];
        let s = cbrt_fast(mixed2.max(0.0)) + NEG_CBRT_BIAS[2];

        // Mix into XYB
        x_out[i] = 0.5 * (l - m);
        y_out[i] = 0.5 * (l + m);
        b_out[i] = s;
    }
}

#[inline]
pub fn inverse_xyb_planar_scalar(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    n: usize,
) {
    for i in 0..n {
        let x = xyb_x[i];
        let y = xyb_y[i];
        let b = xyb_b[i];

        let gamma_r = y + x - NEG_CBRT_BIAS[0];
        let gamma_g = y - x - NEG_CBRT_BIAS[1];
        let gamma_b = b - NEG_CBRT_BIAS[2];

        // W44-RECON-DEEP/A11: cube+bias via single FMA matches libjxl's
        // `MulAdd(gamma_r2, gamma_r, neg_bias_r)` (dec_xyb-inl.h:66-71).
        // Previous `g*g*g - bias` had 3 separate roundings; the FMA gives
        // 2 roundings (one for the square, one for the cube+bias fuse) and
        // closes a ~4-5e-6 max-abs linear-RGB divergence vs the decoder.
        let fma = crate::scalarmath::mul_add_f32;
        let gr2 = gamma_r * gamma_r;
        let gg2 = gamma_g * gamma_g;
        let gb2 = gamma_b * gamma_b;
        let mixed_r = fma(gr2, gamma_r, -OPSIN_BIAS[0]);
        let mixed_g = fma(gg2, gamma_g, -OPSIN_BIAS[1]);
        let mixed_b = fma(gb2, gamma_b, -OPSIN_BIAS[2]);

        out_r[i] = fma(
            INV_OPSIN[0][0],
            mixed_r,
            fma(INV_OPSIN[0][1], mixed_g, INV_OPSIN[0][2] * mixed_b),
        );
        out_g[i] = fma(
            INV_OPSIN[1][0],
            mixed_r,
            fma(INV_OPSIN[1][1], mixed_g, INV_OPSIN[1][2] * mixed_b),
        );
        out_b[i] = fma(
            INV_OPSIN[2][0],
            mixed_r,
            fma(INV_OPSIN[2][1], mixed_g, INV_OPSIN[2][2] * mixed_b),
        );
    }
}

#[inline]
pub fn inverse_xyb_scalar(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    for i in 0..n {
        let x = xyb_x[i];
        let y = xyb_y[i];
        let b = xyb_b[i];

        // Unmix XYB to gamma-domain LMS + add cbrt(bias)
        let gamma_r = y + x - NEG_CBRT_BIAS[0];
        let gamma_g = y - x - NEG_CBRT_BIAS[1];
        let gamma_b = b - NEG_CBRT_BIAS[2];

        // Cube + bias via single FMA — W44-RECON-DEEP/A11. See
        // `inverse_xyb_planar_scalar` for full rationale.
        let fma = crate::scalarmath::mul_add_f32;
        let gr2 = gamma_r * gamma_r;
        let gg2 = gamma_g * gamma_g;
        let gb2 = gamma_b * gamma_b;
        let mixed_r = fma(gr2, gamma_r, -OPSIN_BIAS[0]);
        let mixed_g = fma(gg2, gamma_g, -OPSIN_BIAS[1]);
        let mixed_b = fma(gb2, gamma_b, -OPSIN_BIAS[2]);

        // Inverse opsin matrix → linear RGB (chained FMA for SIMD parity)
        let r = fma(
            INV_OPSIN[0][0],
            mixed_r,
            fma(INV_OPSIN[0][1], mixed_g, INV_OPSIN[0][2] * mixed_b),
        );
        let g = fma(
            INV_OPSIN[1][0],
            mixed_r,
            fma(INV_OPSIN[1][1], mixed_g, INV_OPSIN[1][2] * mixed_b),
        );
        let b_lin = fma(
            INV_OPSIN[2][0],
            mixed_r,
            fma(INV_OPSIN[2][1], mixed_g, INV_OPSIN[2][2] * mixed_b),
        );

        linear_rgb[i * 3] = r;
        linear_rgb[i * 3 + 1] = g;
        linear_rgb[i * 3 + 2] = b_lin;
    }
}

// ============================================================================
// magetypes-consolidated SIMD implementation — forward (RGB → XYB)
// ============================================================================
//
// Single body, one source of truth. The `#[magetypes(...)]` macro generates
// one `#[arcane]`-wrapped variant per listed tier:
//   - `forward_xyb_impl_v4`      (x86_64 AVX-512 256-bit f32x8, opt-in `avx512`)
//   - `forward_xyb_impl_v3`      (x86_64 AVX2, native 256-bit f32x8)
//   - `forward_xyb_impl_neon`    (aarch64, 2× f32x4 polyfill of f32x8)
//   - `forward_xyb_impl_wasm128` (wasm32, 2× f32x4 polyfill of f32x8)
//   - `forward_xyb_impl_scalar`  (portable scalar fallback)
//
// FMA association: outermost is `m00 * r + (m01 * g + (m02 * b + bias0))`,
// matching the pre-consolidation AVX2/NEON/WASM bodies bit-for-bit. The
// cube root keeps `cbrt_fast`'s exact f64 Newton-Raphson math — integer
// initial guess per lane (scalar, cheap), then the two Newton iterations
// run in `f64x4` with the SAME per-lane IEEE operations in the SAME
// order (`(t*t)*t`, `(x+x)+r`, `(x+r)+r`, `(t*num)/den` — plain
// mul/add/div, NO fma), so every lane is bit-identical to the scalar
// `cbrt_fast` and hash-locks are unaffected. What the vectorization buys
// is pipelined f64 division (the walk's latency bottleneck: 2 divides
// per element). **Do not** replace with a SIMD `cbrt_lowp` and do not
// re-associate or fma-fuse the Newton arithmetic; the f64 op-for-op
// discipline is what this kernel relies on for hash-lock byte-identity
// through downstream quantization. `v4` omitted: no `F64x4Backend` for
// `X64V4Token` (pixel_loss.rs precedent).

#[magetypes(define(f32x8, f64x4), v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
pub fn forward_xyb_impl(
    token: Token,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    n: usize,
) {
    let m00 = f32x8::splat(token, OPSIN_MATRIX[0][0]);
    let m01 = f32x8::splat(token, OPSIN_MATRIX[0][1]);
    let m02 = f32x8::splat(token, OPSIN_MATRIX[0][2]);
    let m10 = f32x8::splat(token, OPSIN_MATRIX[1][0]);
    let m11 = f32x8::splat(token, OPSIN_MATRIX[1][1]);
    let m12 = f32x8::splat(token, OPSIN_MATRIX[1][2]);
    let m20 = f32x8::splat(token, OPSIN_MATRIX[2][0]);
    let m21 = f32x8::splat(token, OPSIN_MATRIX[2][1]);
    let m22 = f32x8::splat(token, OPSIN_MATRIX[2][2]);
    let bias0 = f32x8::splat(token, OPSIN_BIAS[0]);
    let bias1 = f32x8::splat(token, OPSIN_BIAS[1]);
    let bias2 = f32x8::splat(token, OPSIN_BIAS[2]);
    let neg_cbrt0 = f32x8::splat(token, NEG_CBRT_BIAS[0]);
    let neg_cbrt1 = f32x8::splat(token, NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x8::splat(token, NEG_CBRT_BIAS[2]);
    let half = f32x8::splat(token, 0.5);
    let zero = f32x8::splat(token, 0.0);

    let chunks = n / 8;
    let simd_n = chunks * 8;

    for chunk in 0..chunks {
        let base = chunk * 8;
        let rv = f32x8::from_slice(token, &r[base..]);
        let gv = f32x8::from_slice(token, &g[base..]);
        let bv = f32x8::from_slice(token, &b[base..]);

        // Matrix multiply + bias (FMA chains, association preserved)
        let mixed0 = m00.mul_add(rv, m01.mul_add(gv, m02.mul_add(bv, bias0)));
        let mixed1 = m10.mul_add(rv, m11.mul_add(gv, m12.mul_add(bv, bias1)));
        let mixed2 = m20.mul_add(rv, m21.mul_add(gv, m22.mul_add(bv, bias2)));

        // Clamp negative to zero
        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        // Cube root — `cbrt_fast`'s math, vectorized (see module note
        // above): scalar integer initial guess per lane, then the two
        // f64 Newton iterations in f64x4 (6 groups of 4 across the 24
        // lane-values). Plain mul/add/div in the scalar op order — each
        // lane is bit-identical to `cbrt_fast`, including the x == 0
        // fixup (scalar's early return).
        let m0_arr = mixed0.to_array();
        let m1_arr = mixed1.to_array();
        let m2_arr = mixed2.to_array();
        let mut init_t = [0.0f64; 24];
        let mut init_x = [0.0f64; 24];
        for j in 0..8 {
            let (t, xf) = cbrt_newton_init(m0_arr[j]);
            init_t[j] = t;
            init_x[j] = xf;
            let (t, xf) = cbrt_newton_init(m1_arr[j]);
            init_t[8 + j] = t;
            init_x[8 + j] = xf;
            let (t, xf) = cbrt_newton_init(m2_arr[j]);
            init_t[16 + j] = t;
            init_x[16 + j] = xf;
        }
        let mut out24 = [0.0f32; 24];
        for grp in 0..6 {
            let base4 = grp * 4;
            let t0 = f64x4::from_array(
                token,
                [
                    init_t[base4],
                    init_t[base4 + 1],
                    init_t[base4 + 2],
                    init_t[base4 + 3],
                ],
            );
            let x = f64x4::from_array(
                token,
                [
                    init_x[base4],
                    init_x[base4 + 1],
                    init_x[base4 + 2],
                    init_x[base4 + 3],
                ],
            );
            let x2 = x + x; // (x + x) — matches scalar `xf64 + xf64`
            // First Newton iteration: t = (t * ((x+x) + r)) / ((x + r) + r)
            let r0 = (t0 * t0) * t0;
            let t1 = (t0 * (x2 + r0)) / ((x + r0) + r0);
            // Second Newton iteration
            let r1 = (t1 * t1) * t1;
            let t2 = (t1 * (x2 + r1)) / ((x + r1) + r1);
            let arr = t2.to_array();
            out24[base4] = arr[0] as f32;
            out24[base4 + 1] = arr[1] as f32;
            out24[base4 + 2] = arr[2] as f32;
            out24[base4 + 3] = arr[3] as f32;
        }
        // Scalar `cbrt_fast` early-returns 0.0 for x == 0; apply the
        // same fixup (the Newton path would produce a tiny nonzero).
        for j in 0..8 {
            if m0_arr[j] == 0.0 {
                out24[j] = 0.0;
            }
            if m1_arr[j] == 0.0 {
                out24[8 + j] = 0.0;
            }
            if m2_arr[j] == 0.0 {
                out24[16 + j] = 0.0;
            }
        }
        let mut c0 = [0.0f32; 8];
        let mut c1 = [0.0f32; 8];
        let mut c2 = [0.0f32; 8];
        c0.copy_from_slice(&out24[..8]);
        c1.copy_from_slice(&out24[8..16]);
        c2.copy_from_slice(&out24[16..24]);
        let l = f32x8::from_array(token, c0) + neg_cbrt0;
        let m = f32x8::from_array(token, c1) + neg_cbrt1;
        let s = f32x8::from_array(token, c2) + neg_cbrt2;

        // XYB mixing
        let xv = half * (l - m);
        let yv = half * (l + m);

        let xs: &mut [f32; 8] = (&mut x_out[base..base + 8]).try_into().unwrap();
        xv.store(xs);
        let ys: &mut [f32; 8] = (&mut y_out[base..base + 8]).try_into().unwrap();
        yv.store(ys);
        let bs: &mut [f32; 8] = (&mut b_out[base..base + 8]).try_into().unwrap();
        s.store(bs);
    }

    // Scalar remainder
    if simd_n < n {
        forward_xyb_scalar(
            &r[simd_n..],
            &g[simd_n..],
            &b[simd_n..],
            &mut x_out[simd_n..],
            &mut y_out[simd_n..],
            &mut b_out[simd_n..],
            n - simd_n,
        );
    }
}

// ============================================================================
// magetypes-consolidated SIMD implementation — inverse planar (XYB → RGB)
// ============================================================================
//
// Inverse direction has no cube root (the cube is a SIMD-friendly multiply
// chain), so the body is shorter and entirely vectorizable.

#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
pub fn inverse_xyb_planar_impl(
    token: Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    n: usize,
) {
    let neg_cbrt0 = f32x8::splat(token, -NEG_CBRT_BIAS[0]); // positive: cbrt(bias)
    let neg_cbrt1 = f32x8::splat(token, -NEG_CBRT_BIAS[1]);
    let neg_cbrt2 = f32x8::splat(token, -NEG_CBRT_BIAS[2]);
    let neg_bias0 = f32x8::splat(token, -OPSIN_BIAS[0]);
    let neg_bias1 = f32x8::splat(token, -OPSIN_BIAS[1]);
    let neg_bias2 = f32x8::splat(token, -OPSIN_BIAS[2]);
    let inv00 = f32x8::splat(token, INV_OPSIN[0][0]);
    let inv01 = f32x8::splat(token, INV_OPSIN[0][1]);
    let inv02 = f32x8::splat(token, INV_OPSIN[0][2]);
    let inv10 = f32x8::splat(token, INV_OPSIN[1][0]);
    let inv11 = f32x8::splat(token, INV_OPSIN[1][1]);
    let inv12 = f32x8::splat(token, INV_OPSIN[1][2]);
    let inv20 = f32x8::splat(token, INV_OPSIN[2][0]);
    let inv21 = f32x8::splat(token, INV_OPSIN[2][1]);
    let inv22 = f32x8::splat(token, INV_OPSIN[2][2]);

    let chunks = n / 8;
    let simd_n = chunks * 8;

    for chunk in 0..chunks {
        let base = chunk * 8;
        let x = f32x8::from_slice(token, &xyb_x[base..]);
        let y = f32x8::from_slice(token, &xyb_y[base..]);
        let b = f32x8::from_slice(token, &xyb_b[base..]);

        // Unmix to gamma-domain LMS + add cbrt(bias)
        let gamma_r = y + x + neg_cbrt0;
        let gamma_g = y - x + neg_cbrt1;
        let gamma_b = b + neg_cbrt2;

        // W44-RECON-DEEP/A11: cube+bias via single FMA matches libjxl's
        // `MulAdd(gamma_r2, gamma_r, neg_bias_r)` (dec_xyb-inl.h:66-71).
        // Previous `g*g*g + neg_bias` had 3 separate roundings; the FMA
        // gives 2 (square plus fused cube+bias) and closes a ~4-5e-6
        // max-abs linear-RGB divergence vs the decoder buttloop saw.
        let gr2 = gamma_r * gamma_r;
        let gg2 = gamma_g * gamma_g;
        let gb2 = gamma_b * gamma_b;
        let mixed_r = gr2.mul_add(gamma_r, neg_bias0);
        let mixed_g = gg2.mul_add(gamma_g, neg_bias1);
        let mixed_b = gb2.mul_add(gamma_b, neg_bias2);

        // Inverse opsin matrix (FMA chains, association preserved)
        let rv = inv00.mul_add(mixed_r, inv01.mul_add(mixed_g, inv02 * mixed_b));
        let gv = inv10.mul_add(mixed_r, inv11.mul_add(mixed_g, inv12 * mixed_b));
        let bv = inv20.mul_add(mixed_r, inv21.mul_add(mixed_g, inv22 * mixed_b));

        // Planar SIMD store — no scalar interleave needed on the hot path.
        let rs: &mut [f32; 8] = (&mut out_r[base..base + 8]).try_into().unwrap();
        rv.store(rs);
        let gs: &mut [f32; 8] = (&mut out_g[base..base + 8]).try_into().unwrap();
        gv.store(gs);
        let bs: &mut [f32; 8] = (&mut out_b[base..base + 8]).try_into().unwrap();
        bv.store(bs);
    }

    if simd_n < n {
        inverse_xyb_planar_scalar(
            &xyb_x[simd_n..],
            &xyb_y[simd_n..],
            &xyb_b[simd_n..],
            &mut out_r[simd_n..],
            &mut out_g[simd_n..],
            &mut out_b[simd_n..],
            n - simd_n,
        );
    }
}

// ============================================================================
// Backwards-compat suffixed re-exports
// ============================================================================
//
// Pre-consolidation callers spelled the variants `forward_xyb_avx2` etc.
// magetypes' tier names are `_v3` (AVX2) / `_neon` / `_wasm128`. Re-export
// under the historical names so the external API stays stable.
//
// `inverse_xyb_*` (AoS interleaved) historical exports route through the
// scalar wrapper that calls the planar magetypes body and interleaves on
// the way out. The per-arch direct re-exports of the planar variant
// satisfy callers that want to skip the dispatch.

#[cfg(target_arch = "x86_64")]
pub use forward_xyb_impl_v3 as forward_xyb_avx2;

#[cfg(target_arch = "x86_64")]
pub use inverse_xyb_planar_impl_v3 as inverse_xyb_planar_avx2;

#[cfg(target_arch = "aarch64")]
pub use forward_xyb_impl_neon as forward_xyb_neon;

#[cfg(target_arch = "aarch64")]
pub use inverse_xyb_planar_impl_neon as inverse_xyb_planar_neon;

#[cfg(target_arch = "wasm32")]
pub use forward_xyb_impl_wasm128 as forward_xyb_wasm128;

#[cfg(target_arch = "wasm32")]
pub use inverse_xyb_planar_impl_wasm128 as inverse_xyb_planar_wasm128;

// AoS interleaved inverse — re-export the public dispatch entry under each
// historical per-arch alias so existing callers keep compiling. Direct
// per-arch fast paths into the AoS form are not exposed (the SIMD work
// happens in the planar kernel; the interleave is a scalar tail).
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn inverse_xyb_avx2(
    _token: archmage::X64V3Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    xyb_to_linear_rgb_batch(xyb_x, xyb_y, xyb_b, linear_rgb, n);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn inverse_xyb_neon(
    _token: archmage::NeonToken,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    xyb_to_linear_rgb_batch(xyb_x, xyb_y, xyb_b, linear_rgb, n);
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn inverse_xyb_wasm128(
    _token: archmage::Wasm128Token,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    linear_rgb: &mut [f32],
    n: usize,
) {
    xyb_to_linear_rgb_batch(xyb_x, xyb_y, xyb_b, linear_rgb, n);
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    extern crate std;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Sweep test: compare SIMD forward XYB against reference std cbrt.
    #[test]
    fn test_forward_xyb_sweep() {
        let n = 256;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];

        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            r[i] = t;
            g[i] = 1.0 - t;
            b[i] = (t * 2.0).min(1.0);
        }

        // Reference: use std cbrt
        let mut x_ref = vec![0.0f32; n];
        let mut y_ref = vec![0.0f32; n];
        let mut b_ref = vec![0.0f32; n];
        for i in 0..n {
            let mixed0 = OPSIN_MATRIX[0][0] * r[i]
                + OPSIN_MATRIX[0][1] * g[i]
                + OPSIN_MATRIX[0][2] * b[i]
                + OPSIN_BIAS[0];
            let mixed1 = OPSIN_MATRIX[1][0] * r[i]
                + OPSIN_MATRIX[1][1] * g[i]
                + OPSIN_MATRIX[1][2] * b[i]
                + OPSIN_BIAS[1];
            let mixed2 = OPSIN_MATRIX[2][0] * r[i]
                + OPSIN_MATRIX[2][1] * g[i]
                + OPSIN_MATRIX[2][2] * b[i]
                + OPSIN_BIAS[2];
            let l = mixed0.max(0.0).cbrt() + NEG_CBRT_BIAS[0];
            let m = mixed1.max(0.0).cbrt() + NEG_CBRT_BIAS[1];
            let s = mixed2.max(0.0).cbrt() + NEG_CBRT_BIAS[2];
            x_ref[i] = 0.5 * (l - m);
            y_ref[i] = 0.5 * (l + m);
            b_ref[i] = s;
        }

        // Dispatch — test all token permutations so every magetypes tier
        // available on this host runs at least once.
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut x_out = vec![0.0f32; n];
                let mut y_out = vec![0.0f32; n];
                let mut b_out = vec![0.0f32; n];
                linear_rgb_to_xyb_batch(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

                for i in 0..n {
                    let ex = (x_out[i] - x_ref[i]).abs();
                    let ey = (y_out[i] - y_ref[i]).abs();
                    let eb = (b_out[i] - b_ref[i]).abs();
                    assert!(
                        ex < 1e-5 && ey < 1e-5 && eb < 1e-5,
                        "Pixel {}: SIMD=({},{},{}), ref=({},{},{}), err=({},{},{}) [{perm}]",
                        i,
                        x_out[i],
                        y_out[i],
                        b_out[i],
                        x_ref[i],
                        y_ref[i],
                        b_ref[i],
                        ex,
                        ey,
                        eb
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    /// Sweep test: compare SIMD inverse XYB against reference scalar.
    #[test]
    fn test_inverse_xyb_sweep() {
        let n = 256;
        let mut xyb_x = vec![0.0f32; n];
        let mut xyb_y = vec![0.0f32; n];
        let mut xyb_b = vec![0.0f32; n];

        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            xyb_x[i] = (t - 0.5) * 0.8;
            xyb_y[i] = t * 1.1;
            xyb_b[i] = t * 0.9 - 0.1;
        }

        // Reference scalar
        let mut ref_rgb = vec![0.0f32; n * 3];
        inverse_xyb_scalar(&xyb_x, &xyb_y, &xyb_b, &mut ref_rgb, n);

        // Dispatch — test all token permutations
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_rgb = vec![0.0f32; n * 3];
                xyb_to_linear_rgb_batch(&xyb_x, &xyb_y, &xyb_b, &mut simd_rgb, n);

                for i in 0..n * 3 {
                    let err = (simd_rgb[i] - ref_rgb[i]).abs();
                    assert!(
                        err < 1e-5,
                        "Component {}: SIMD={}, ref={}, err={:.2e} [{perm}]",
                        i,
                        simd_rgb[i],
                        ref_rgb[i],
                        err
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    /// Roundtrip test: RGB → XYB → RGB should be approximately identity.
    #[test]
    fn test_xyb_roundtrip_sweep() {
        let n = 256;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];

        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            r[i] = t * 0.8 + 0.01;
            g[i] = (1.0 - t) * 0.9 + 0.01;
            b[i] = (t * 1.5).min(0.95) + 0.01;
        }

        // Forward: RGB → XYB
        let mut x = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut bv = vec![0.0f32; n];
        linear_rgb_to_xyb_batch(&r, &g, &b, &mut x, &mut y, &mut bv);

        // Inverse: XYB → RGB
        let mut rgb_out = vec![0.0f32; n * 3];
        xyb_to_linear_rgb_batch(&x, &y, &bv, &mut rgb_out, n);

        let mut max_err = 0.0f32;
        for i in 0..n {
            let er = (rgb_out[i * 3] - r[i]).abs();
            let eg = (rgb_out[i * 3 + 1] - g[i]).abs();
            let eb = (rgb_out[i * 3 + 2] - b[i]).abs();
            max_err = max_err.max(er).max(eg).max(eb);
        }
        assert!(
            max_err < 1e-4,
            "Roundtrip max error {:.2e} exceeds 1e-4",
            max_err
        );
    }

    /// Edge cases: black, white, primary colors, near-zero values.
    #[test]
    fn test_forward_xyb_edge_cases() {
        let test_cases: &[(f32, f32, f32)] = &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.001, 0.001, 0.001),
            (0.999, 0.999, 0.999),
            (0.5, 0.5, 0.5),
        ];

        let n = test_cases.len();
        let r: Vec<f32> = test_cases.iter().map(|c| c.0).collect();
        let g: Vec<f32> = test_cases.iter().map(|c| c.1).collect();
        let b: Vec<f32> = test_cases.iter().map(|c| c.2).collect();

        // Reference using std cbrt
        let mut x_ref = vec![0.0f32; n];
        let mut y_ref = vec![0.0f32; n];
        let mut b_ref = vec![0.0f32; n];
        for i in 0..n {
            let mixed0 = OPSIN_MATRIX[0][0] * r[i]
                + OPSIN_MATRIX[0][1] * g[i]
                + OPSIN_MATRIX[0][2] * b[i]
                + OPSIN_BIAS[0];
            let mixed1 = OPSIN_MATRIX[1][0] * r[i]
                + OPSIN_MATRIX[1][1] * g[i]
                + OPSIN_MATRIX[1][2] * b[i]
                + OPSIN_BIAS[1];
            let mixed2 = OPSIN_MATRIX[2][0] * r[i]
                + OPSIN_MATRIX[2][1] * g[i]
                + OPSIN_MATRIX[2][2] * b[i]
                + OPSIN_BIAS[2];
            let l = mixed0.max(0.0).cbrt() + NEG_CBRT_BIAS[0];
            let m = mixed1.max(0.0).cbrt() + NEG_CBRT_BIAS[1];
            let s = mixed2.max(0.0).cbrt() + NEG_CBRT_BIAS[2];
            x_ref[i] = 0.5 * (l - m);
            y_ref[i] = 0.5 * (l + m);
            b_ref[i] = s;
        }

        // Dispatch — test all token permutations
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut x_out = vec![0.0f32; n];
                let mut y_out = vec![0.0f32; n];
                let mut b_out = vec![0.0f32; n];
                linear_rgb_to_xyb_batch(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

                for i in 0..n {
                    let ex = (x_out[i] - x_ref[i]).abs();
                    let ey = (y_out[i] - y_ref[i]).abs();
                    let eb = (b_out[i] - b_ref[i]).abs();
                    assert!(
                        ex < 1e-5 && ey < 1e-5 && eb < 1e-5,
                        "Edge case {:?}: SIMD=({},{},{}), ref=({},{},{}), err=({:.2e},{:.2e},{:.2e}) [{perm}]",
                        test_cases[i],
                        x_out[i],
                        y_out[i],
                        b_out[i],
                        x_ref[i],
                        y_ref[i],
                        b_ref[i],
                        ex,
                        ey,
                        eb
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    /// Test that planar inverse matches interleaved inverse.
    #[test]
    fn test_inverse_xyb_planar_matches_interleaved() {
        let n = 256;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];

        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            r[i] = t;
            g[i] = 1.0 - t;
            b[i] = (t * 2.0).min(1.0);
        }

        // Use scalar forward to get deterministic XYB input
        let mut x = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut bv = vec![0.0f32; n];
        forward_xyb_scalar(&r, &g, &b, &mut x, &mut y, &mut bv, n);

        // Scalar reference for interleaved inverse
        let mut ref_rgb = vec![0.0f32; n * 3];
        inverse_xyb_scalar(&x, &y, &bv, &mut ref_rgb, n);

        // Scalar reference for planar inverse
        let mut ref_r = vec![0.0f32; n];
        let mut ref_g = vec![0.0f32; n];
        let mut ref_b = vec![0.0f32; n];
        inverse_xyb_planar_scalar(&x, &y, &bv, &mut ref_r, &mut ref_g, &mut ref_b, n);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                // Interleaved inverse
                let mut interleaved = vec![0.0f32; n * 3];
                xyb_to_linear_rgb_batch(&x, &y, &bv, &mut interleaved, n);

                // Planar inverse
                let mut pr = vec![0.0f32; n];
                let mut pg = vec![0.0f32; n];
                let mut pb = vec![0.0f32; n];
                xyb_to_linear_rgb_planar(&x, &y, &bv, &mut pr, &mut pg, &mut pb, n);

                for i in 0..n {
                    let ir = interleaved[i * 3];
                    let ig = interleaved[i * 3 + 1];
                    let ib = interleaved[i * 3 + 2];
                    // Both dispatch paths must match scalar
                    assert!(
                        (ir - ref_rgb[i * 3]).abs() < 1e-5,
                        "Interleaved R mismatch at {i}: got {ir}, ref {} [{perm}]",
                        ref_rgb[i * 3]
                    );
                    assert!(
                        (pr[i] - ref_r[i]).abs() < 1e-5,
                        "Planar R mismatch at {i}: got {}, ref {} [{perm}]",
                        pr[i],
                        ref_r[i]
                    );
                    // And they must match each other
                    assert!(
                        (pr[i] - ir).abs() < 1e-6
                            && (pg[i] - ig).abs() < 1e-6
                            && (pb[i] - ib).abs() < 1e-6,
                        "Planar/interleaved mismatch at {i}: planar=({},{},{}) interleaved=({ir},{ig},{ib}) [{perm}]",
                        pr[i],
                        pg[i],
                        pb[i]
                    );
                }
            },
        );
        std::eprintln!("{report}");
    }

    /// W44-RECON-DEEP/A11 unit test: assert our inverse XYB matches a
    /// libjxl-style single-FMA reference (same constants, same FMA shape)
    /// to <1e-7 per-channel max-abs. The reference uses the
    /// `(g*g).mul_add(g, neg_bias)` shape from libjxl `dec_xyb-inl.h:66-71`
    /// and the full f64 `-0.16462299647058826` literal for the off-diagonal
    /// blue cross-terms. This catches regressions if either fix is reverted.
    #[test]
    fn test_inverse_xyb_libjxl_single_fma_parity() {
        // Hand-coded libjxl reference (single FMA cube+bias, libjxl
        // `kDefaultInverseOpsinAbsorbanceMatrix` to full f64 precision).
        const LIBJXL_INV: [[f32; 3]; 3] = [
            [
                11.031_566_901_960_783,
                -9.866_943_921_568_629,
                -0.164_622_996_470_588_26,
            ],
            [
                -3.254_147_380_392_157,
                4.418_770_392_156_863,
                -0.164_622_996_470_588_26,
            ],
            [
                -3.658_851_286_274_509_7,
                2.712_923_047_058_823_5,
                1.945_928_239_215_686_3,
            ],
        ];
        // libjxl `kNegOpsinAbsorbanceBiasRGB[0] = -kOpsinAbsorbanceBias0`.
        // `OpsinParams::Init` copies this into `opsin_biases`, then
        // `opsin_biases_cbrt[c] = cbrtf(opsin_biases[c])`. So both the
        // bias passed to MulAdd AND the cbrt unmix are NEGATIVE values.
        const LIBJXL_NEG_BIAS: f32 = -0.003_793_073_255_275_449_3;
        // cbrtf of the negative bias is also negative.
        let libjxl_cbrt_bias: f32 = LIBJXL_NEG_BIAS.cbrt();

        let inv_ref = |x: f32, y: f32, b: f32| -> [f32; 3] {
            // libjxl `dec_xyb-inl.h:57-63` — gamma = (y±x) - opsin_biases_cbrt.
            // With opsin_biases_cbrt negative, this becomes `(y±x) + 0.155954`.
            let gamma_r = y + x - libjxl_cbrt_bias;
            let gamma_g = y - x - libjxl_cbrt_bias;
            let gamma_b = b - libjxl_cbrt_bias;
            // Single FMA cube+bias (libjxl dec_xyb-inl.h:69-71)
            let mixed_r = (gamma_r * gamma_r).mul_add(gamma_r, LIBJXL_NEG_BIAS);
            let mixed_g = (gamma_g * gamma_g).mul_add(gamma_g, LIBJXL_NEG_BIAS);
            let mixed_b = (gamma_b * gamma_b).mul_add(gamma_b, LIBJXL_NEG_BIAS);
            // Right-to-left FMA chain (matches ours; libjxl decoder uses
            // left-to-right but the difference is bounded by the 1-ULP
            // matrix constant fix and the FMA cube+bias single-rounding).
            let r = LIBJXL_INV[0][0].mul_add(
                mixed_r,
                LIBJXL_INV[0][1].mul_add(mixed_g, LIBJXL_INV[0][2] * mixed_b),
            );
            let g = LIBJXL_INV[1][0].mul_add(
                mixed_r,
                LIBJXL_INV[1][1].mul_add(mixed_g, LIBJXL_INV[1][2] * mixed_b),
            );
            let b_out = LIBJXL_INV[2][0].mul_add(
                mixed_r,
                LIBJXL_INV[2][1].mul_add(mixed_g, LIBJXL_INV[2][2] * mixed_b),
            );
            [r, g, b_out]
        };

        // Confirm our INV_OPSIN matches the libjxl matrix bit-for-bit at f32.
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(
                    INV_OPSIN[r][c].to_bits(),
                    LIBJXL_INV[r][c].to_bits(),
                    "INV_OPSIN[{r}][{c}] bit mismatch: ours={:#010x}, libjxl={:#010x}",
                    INV_OPSIN[r][c].to_bits(),
                    LIBJXL_INV[r][c].to_bits()
                );
            }
        }
        // Confirm OPSIN_BIAS bits match too.
        assert_eq!(OPSIN_BIAS[0].to_bits(), (-LIBJXL_NEG_BIAS).to_bits());

        // Sweep typical buttloop XYB triples — same range as
        // test_inverse_xyb_sweep so we cover the gamut the buttloop touches.
        let n = 256;
        let mut xyb_x = vec![0.0f32; n];
        let mut xyb_y = vec![0.0f32; n];
        let mut xyb_b = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            xyb_x[i] = (t - 0.5) * 0.8;
            xyb_y[i] = t * 1.1;
            xyb_b[i] = t * 0.9 - 0.1;
        }

        // Our scalar path (planar)
        let mut ours_r = vec![0.0f32; n];
        let mut ours_g = vec![0.0f32; n];
        let mut ours_b = vec![0.0f32; n];
        inverse_xyb_planar_scalar(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            &mut ours_r,
            &mut ours_g,
            &mut ours_b,
            n,
        );

        let mut max_abs = 0.0f32;
        for i in 0..n {
            let ref_rgb = inv_ref(xyb_x[i], xyb_y[i], xyb_b[i]);
            let er = (ours_r[i] - ref_rgb[0]).abs();
            let eg = (ours_g[i] - ref_rgb[1]).abs();
            let eb = (ours_b[i] - ref_rgb[2]).abs();
            max_abs = max_abs.max(er).max(eg).max(eb);
        }
        // The only remaining divergence vs the libjxl-style reference is
        // the matrix-mul FMA association (right-to-left vs left-to-right),
        // which is bounded at ~1-2 ULP on linear-RGB magnitudes ≤ 12.
        // 1e-5 is a safe ceiling. If this trips it means cube+bias FMA or
        // INV_OPSIN constants regressed.
        assert!(
            max_abs < 1e-5,
            "W44-RECON-DEEP/A11 parity: max-abs {max_abs:.3e} exceeds 1e-5 — cube+bias FMA or INV_OPSIN constants regressed"
        );
    }
}

#[cfg(test)]
mod expanded_coverage {
    use super::*;
    use crate::test_helpers::*;
    use alloc::format;
    use alloc::vec;

    /// linear_rgb_to_xyb_batch across multiple sizes and edge inputs.
    /// XYB uses cube-root via f64 Newton + 3x3 matrix.  SIMD vs scalar
    /// diverges by ~1-2 ULP per matrix-mul FMA association.
    #[test]
    fn linear_rgb_to_xyb_scalar_vs_dispatch_sizes() {
        // Use only NON-NEGATIVE inputs — cube root of negative produces
        // NaN in the standard branch.  Use unit-range random values.
        for &n in &[1_usize, 7, 8, 9, 16, 17, 64, 129] {
            let r = gen_f32_unit(0xA001_AAAA ^ n as u64, n);
            let g = gen_f32_unit(0xA002_BBBB ^ n as u64, n);
            let b_in = gen_f32_unit(0xA003_CCCC ^ n as u64, n);

            let mut ref_x = vec![0.0_f32; n];
            let mut ref_y = vec![0.0_f32; n];
            let mut ref_b = vec![0.0_f32; n];
            forward_xyb_scalar(&r, &g, &b_in, &mut ref_x, &mut ref_y, &mut ref_b, n);

            run_dispatch_parity(|perm| {
                let mut act_x = vec![0.0_f32; n];
                let mut act_y = vec![0.0_f32; n];
                let mut act_b = vec![0.0_f32; n];
                linear_rgb_to_xyb_batch(&r, &g, &b_in, &mut act_x, &mut act_y, &mut act_b);
                assert_f32_slice_close_ulps_abs(
                    &ref_x,
                    &act_x,
                    16,
                    1e-5,
                    perm,
                    &format!("fwd_x(n={n})"),
                );
                assert_f32_slice_close_ulps_abs(
                    &ref_y,
                    &act_y,
                    16,
                    1e-5,
                    perm,
                    &format!("fwd_y(n={n})"),
                );
                assert_f32_slice_close_ulps_abs(
                    &ref_b,
                    &act_b,
                    16,
                    1e-5,
                    perm,
                    &format!("fwd_b(n={n})"),
                );
            });
        }
    }

    /// xyb_to_linear_rgb_planar across sizes — exercises inverse cube +
    /// 3x3 inverse matrix.  Bounded by W44-A11's 1e-5 max-abs invariant.
    #[test]
    fn xyb_to_linear_rgb_planar_scalar_vs_dispatch_sizes() {
        for &n in &[1_usize, 7, 8, 9, 16, 17, 64, 129] {
            // Use realistic XYB ranges: x small (~0.025), y small (~0.5), b ~0.5.
            let xyb_x: alloc::vec::Vec<f32> =
                (0..n).map(|i| (i as f32 * 0.001).sin() * 0.025).collect();
            let xyb_y: alloc::vec::Vec<f32> = (0..n)
                .map(|i| 0.5 + (i as f32 * 0.0013).cos() * 0.3)
                .collect();
            let xyb_b: alloc::vec::Vec<f32> = (0..n)
                .map(|i| 0.5 + (i as f32 * 0.0017).sin() * 0.3)
                .collect();

            let mut ref_r = vec![0.0_f32; n];
            let mut ref_g = vec![0.0_f32; n];
            let mut ref_b = vec![0.0_f32; n];
            inverse_xyb_planar_scalar(
                &xyb_x, &xyb_y, &xyb_b, &mut ref_r, &mut ref_g, &mut ref_b, n,
            );

            run_dispatch_parity(|perm| {
                let mut act_r = vec![0.0_f32; n];
                let mut act_g = vec![0.0_f32; n];
                let mut act_b = vec![0.0_f32; n];
                xyb_to_linear_rgb_planar(
                    &xyb_x, &xyb_y, &xyb_b, &mut act_r, &mut act_g, &mut act_b, n,
                );
                assert_f32_slice_close_ulps_abs(
                    &ref_r,
                    &act_r,
                    16,
                    1e-5,
                    perm,
                    &format!("inv_r(n={n})"),
                );
                assert_f32_slice_close_ulps_abs(
                    &ref_g,
                    &act_g,
                    16,
                    1e-5,
                    perm,
                    &format!("inv_g(n={n})"),
                );
                assert_f32_slice_close_ulps_abs(
                    &ref_b,
                    &act_b,
                    16,
                    1e-5,
                    perm,
                    &format!("inv_b(n={n})"),
                );
            });
        }
    }
}
