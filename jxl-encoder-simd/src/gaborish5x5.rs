// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Gaborish inverse: 5x5 symmetric sharpening kernel with SIMD acceleration.
//!
//! The kernel has 6 weight classes:
//! ```text
//!   D  L  R  L  D
//!   L  d  r  d  L
//!   R  r  c  r  R
//!   L  d  r  d  L
//!   D  L  R  L  D
//! ```
//! where c=center, r=orthogonal(1), d=diagonal(√2), R=orthogonal(2),
//! L=knight's move, D=corner(2√2).
//!
//! **magetypes-consolidated** (W43-2 chunk-4): one `#[magetypes(...)]` body
//! generates every per-arch SIMD variant. The body operates on `f32x8`
//! generics; on backends without native 256-bit registers (NEON, WASM128,
//! scalar) magetypes polyfills `f32x8` to two `f32x4` ops. On x86_64 with
//! AVX-512 (`v4` tier, opt-in via the `avx512` cargo feature) the same body
//! lowers to 512-bit `__m512` ops.
//!
//! Bonus: the pre-consolidation crate had **no WASM SIMD body** (the
//! dispatcher fell through to scalar on wasm32 because no
//! `gaborish_5x5_wasm128` existed). The consolidated path generates a
//! wasm128 variant automatically — that's the "free WASM SIMD" win the
//! W43-2 audit memo predicted.

use archmage::prelude::*;

/// Apply the 5x5 gaborish inverse kernel to a single channel.
///
/// `data` is modified in place. `scratch` is used as temporary input copy.
/// Both must be at least `width * height` elements.
///
/// The 6 weights (wc, wr, wd, w_big_r, wl, w_big_d) should already be
/// normalized (sum to 1.0).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_channel(
    data: &mut [f32],
    scratch: &mut [f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    let n = width * height;
    debug_assert!(data.len() >= n);
    debug_assert!(scratch.len() >= n);

    // Copy input to scratch
    scratch[..n].copy_from_slice(&data[..n]);

    // Dispatch through incant! — picks the best magetypes-generated variant
    // at runtime. Falls through to _scalar on platforms without a SIMD token.
    incant!(gaborish_5x5_impl(
        data, scratch, width, height, wc, wr, wd, w_big_r, wl, w_big_d
    ));
}

// ============================================================================
// Scalar fallback
// ============================================================================

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_scalar(
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    let px = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, (width - 1) as isize) as usize;
        let cy = y.clamp(0, (height - 1) as isize) as usize;
        input[cy * width + cx]
    };

    for y in 0..height {
        let iy = y as isize;
        for x in 0..width {
            let ix = x as isize;

            let mut val = wc * px(ix, iy);

            // r: 4 orthogonal neighbors at distance 1
            val += wr * (px(ix - 1, iy) + px(ix + 1, iy) + px(ix, iy - 1) + px(ix, iy + 1));

            // d: 4 diagonal neighbors at distance sqrt(2)
            val += wd
                * (px(ix - 1, iy - 1)
                    + px(ix + 1, iy - 1)
                    + px(ix - 1, iy + 1)
                    + px(ix + 1, iy + 1));

            // R: 4 orthogonal neighbors at distance 2
            val += w_big_r * (px(ix - 2, iy) + px(ix + 2, iy) + px(ix, iy - 2) + px(ix, iy + 2));

            // L: 8 knight's move neighbors
            val += wl
                * (px(ix - 2, iy - 1)
                    + px(ix - 2, iy + 1)
                    + px(ix + 2, iy - 1)
                    + px(ix + 2, iy + 1)
                    + px(ix - 1, iy - 2)
                    + px(ix + 1, iy - 2)
                    + px(ix - 1, iy + 2)
                    + px(ix + 1, iy + 2));

            // D: 4 corner neighbors at distance 2*sqrt(2)
            val += w_big_d
                * (px(ix - 2, iy - 2)
                    + px(ix + 2, iy - 2)
                    + px(ix - 2, iy + 2)
                    + px(ix + 2, iy + 2));

            output[y * width + x] = val;
        }
    }
}

// ============================================================================
// magetypes-consolidated SIMD implementation
// ============================================================================
//
// Single body, one source of truth. The `#[magetypes(...)]` macro generates
// one `#[arcane]`-wrapped variant per listed tier:
//   - `gaborish_5x5_impl_v4`      (x86_64 AVX-512, opt-in via the `avx512`
//                                  cargo feature on jxl-encoder-simd)
//   - `gaborish_5x5_impl_v3`      (x86_64 AVX2, native 256-bit f32x8)
//   - `gaborish_5x5_impl_neon`    (aarch64, 2x f32x4 polyfill of f32x8)
//   - `gaborish_5x5_impl_wasm128` (wasm32, 2x f32x4 polyfill of f32x8 —
//                                  NEW: the pre-consolidation crate had no
//                                  WASM SIMD body and fell through to scalar)
//   - `gaborish_5x5_impl_scalar`  (portable scalar fallback)
//
// The `define(f32x8)` clause injects a `f32x8` type alias substituting
// `Token` for the concrete token at each tier.

#[magetypes(define(f32x8), v4, v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_impl(
    token: Token,
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    // For images too small for SIMD interior we need x-2..x+10 in-range and a
    // 5-row neighborhood. Min interior pixel count = 8 (one SIMD chunk),
    // so width must accommodate x=2..(width-10) with width-10 > 2, i.e. width >= 13.
    if width < 13 || height < 5 {
        gaborish_5x5_scalar(
            output, input, width, height, wc, wr, wd, w_big_r, wl, w_big_d,
        );
        return;
    }

    let wc_v = f32x8::splat(token, wc);
    let wr_v = f32x8::splat(token, wr);
    let wd_v = f32x8::splat(token, wd);
    let w_big_r_v = f32x8::splat(token, w_big_r);
    let wl_v = f32x8::splat(token, wl);
    let w_big_d_v = f32x8::splat(token, w_big_d);

    // Scalar helpers for border pixels. Captured by-reference of `input` etc.
    let px = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, (width - 1) as isize) as usize;
        let cy = y.clamp(0, (height - 1) as isize) as usize;
        input[cy * width + cx]
    };

    let scalar_pixel = |ix: isize, iy: isize| -> f32 {
        let mut val = wc * px(ix, iy);
        val += wr * (px(ix - 1, iy) + px(ix + 1, iy) + px(ix, iy - 1) + px(ix, iy + 1));
        val += wd
            * (px(ix - 1, iy - 1) + px(ix + 1, iy - 1) + px(ix - 1, iy + 1) + px(ix + 1, iy + 1));
        val += w_big_r * (px(ix - 2, iy) + px(ix + 2, iy) + px(ix, iy - 2) + px(ix, iy + 2));
        val += wl
            * (px(ix - 2, iy - 1)
                + px(ix - 2, iy + 1)
                + px(ix + 2, iy - 1)
                + px(ix + 2, iy + 1)
                + px(ix - 1, iy - 2)
                + px(ix + 1, iy - 2)
                + px(ix - 1, iy + 2)
                + px(ix + 1, iy + 2));
        val += w_big_d
            * (px(ix - 2, iy - 2) + px(ix + 2, iy - 2) + px(ix - 2, iy + 2) + px(ix + 2, iy + 2));
        val
    };

    for y in 0..height {
        let iy = y as isize;

        // Border rows (y < 2 or y >= height-2): all scalar
        if y < 2 || y >= height - 2 {
            for x in 0..width {
                output[y * width + x] = scalar_pixel(x as isize, iy);
            }
            continue;
        }

        // Interior row: scalar left border (x < 2)
        for x in 0..2 {
            output[y * width + x] = scalar_pixel(x as isize, iy);
        }

        // Pre-slice rows to help compiler eliminate bounds checks.
        let r_m2 = (y - 2) * width;
        let r_m1 = (y - 1) * width;
        let r_0 = y * width;
        let r_p1 = (y + 1) * width;
        let r_p2 = (y + 2) * width;

        // SIMD interior: loads access x-2..x+10, so need x + 10 <= width.
        let simd_end = if width >= 12 { width - 10 } else { 2 };
        let mut x = 2;

        while x < simd_end {
            // Center
            let center = f32x8::from_slice(token, &input[r_0 + x..]);

            // r: 4 orthogonal at distance 1
            let left1 = f32x8::from_slice(token, &input[r_0 + x - 1..]);
            let right1 = f32x8::from_slice(token, &input[r_0 + x + 1..]);
            let top1 = f32x8::from_slice(token, &input[r_m1 + x..]);
            let bot1 = f32x8::from_slice(token, &input[r_p1 + x..]);
            let r_sum = left1 + right1 + top1 + bot1;

            // d: 4 diagonal at distance sqrt(2)
            let tl1 = f32x8::from_slice(token, &input[r_m1 + x - 1..]);
            let tr1 = f32x8::from_slice(token, &input[r_m1 + x + 1..]);
            let bl1 = f32x8::from_slice(token, &input[r_p1 + x - 1..]);
            let br1 = f32x8::from_slice(token, &input[r_p1 + x + 1..]);
            let d_sum = tl1 + tr1 + bl1 + br1;

            // R: 4 orthogonal at distance 2
            let left2 = f32x8::from_slice(token, &input[r_0 + x - 2..]);
            let right2 = f32x8::from_slice(token, &input[r_0 + x + 2..]);
            let top2 = f32x8::from_slice(token, &input[r_m2 + x..]);
            let bot2 = f32x8::from_slice(token, &input[r_p2 + x..]);
            let big_r_sum = left2 + right2 + top2 + bot2;

            // L: 8 knight's move neighbors
            let l_a = f32x8::from_slice(token, &input[r_m1 + x - 2..]);
            let l_b = f32x8::from_slice(token, &input[r_p1 + x - 2..]);
            let l_c = f32x8::from_slice(token, &input[r_m1 + x + 2..]);
            let l_d = f32x8::from_slice(token, &input[r_p1 + x + 2..]);
            let l_e = f32x8::from_slice(token, &input[r_m2 + x - 1..]);
            let l_f = f32x8::from_slice(token, &input[r_m2 + x + 1..]);
            let l_g = f32x8::from_slice(token, &input[r_p2 + x - 1..]);
            let l_h = f32x8::from_slice(token, &input[r_p2 + x + 1..]);
            let l_sum = l_a + l_b + l_c + l_d + l_e + l_f + l_g + l_h;

            // D: 4 corner at distance 2*sqrt(2)
            let tl2 = f32x8::from_slice(token, &input[r_m2 + x - 2..]);
            let tr2 = f32x8::from_slice(token, &input[r_m2 + x + 2..]);
            let bl2 = f32x8::from_slice(token, &input[r_p2 + x - 2..]);
            let br2 = f32x8::from_slice(token, &input[r_p2 + x + 2..]);
            let big_d_sum = tl2 + tr2 + bl2 + br2;

            // Combine with FMA chains:
            // result = wc*center + wr*r_sum + wd*d_sum + w_big_r*big_r_sum
            //        + wl*l_sum + w_big_d*big_d_sum
            //
            // FMA association matches the prior hand-written AVX2/NEON bodies
            // bit-for-bit: outermost is `wc*center + (...)`, innermost is
            // `w_big_d * big_d_sum` (plain multiply, no FMA on the tail).
            let result = wc_v.mul_add(
                center,
                wr_v.mul_add(
                    r_sum,
                    wd_v.mul_add(
                        d_sum,
                        w_big_r_v.mul_add(big_r_sum, wl_v.mul_add(l_sum, w_big_d_v * big_d_sum)),
                    ),
                ),
            );

            let out_arr: &mut [f32; 8] = (&mut output[r_0 + x..r_0 + x + 8]).try_into().unwrap();
            result.store(out_arr);

            x += 8;
        }

        // Scalar right border + remainder
        while x < width {
            output[y * width + x] = scalar_pixel(x as isize, iy);
            x += 1;
        }
    }
}

/// The `scalar` tier, hand-written — and NOT a delegation to
/// `gaborish_5x5_scalar`, which is the interesting part.
///
/// The generated `_scalar` tier disagreed with the AVX2 / AVX-512 / NEON tiers
/// by up to **49 ULP**, because magetypes' scalar backend implements `mul_add`
/// as `a * b + c` (`simd/impls/scalar.rs`) rather than fusing it. On a host
/// that cannot summon a vector token — pre-AVX2 x86_64, i686, any architecture
/// magetypes has no backend for — this kernel therefore produced different
/// output, and it is on the byte-affecting path
/// (`jxl-encoder/src/vardct/gaborish.rs` calls `gaborish_5x5_channel`).
///
/// **Delegating to `gaborish_5x5_scalar` would make it worse, not better.**
/// That function accumulates left to right (`val = wc*c; val += wr*r; ...`)
/// while the vector interior uses a nested FMA chain
/// (`wc*c + (wr*r + (wd*d + (wR*R + (wl*L + wD*D))))`), and the two differ by
/// up to 40 ULP on the same inputs. That difference is deliberate and already
/// shipped: the vector body itself uses the left-to-right form for BORDER
/// pixels (`scalar_pixel`) and the nested chain for interior ones. So the
/// scalar tier has to reproduce the vector body's structure exactly — border
/// rows and columns left-to-right, interior in the nested chain — and differ
/// from it only in walking one pixel at a time instead of eight. Per lane the
/// arithmetic is then identical, so the bits are.
///
/// Found 2026-09-10 by the cross-kernel audit that the forward-XYB fix called
/// for; the third kernel of five to have this, after `forward_xyb_impl` and
/// `inverse_xyb_planar_impl` (both of which COULD just delegate). Pinned by
/// `gaborish_5x5_dispatch_is_bit_identical_across_tiers`.
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_impl_scalar(
    _token: archmage::ScalarToken,
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    use crate::scalarmath::mul_add_f32 as fma;

    // Same early-out as the vector body, delegating to the same function.
    if width < 13 || height < 5 {
        gaborish_5x5_scalar(
            output, input, width, height, wc, wr, wd, w_big_r, wl, w_big_d,
        );
        return;
    }

    let px = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, (width - 1) as isize) as usize;
        let cy = y.clamp(0, (height - 1) as isize) as usize;
        input[cy * width + cx]
    };

    // Byte-for-byte the vector body's `scalar_pixel`: left-to-right accumulation.
    let scalar_pixel = |ix: isize, iy: isize| -> f32 {
        let mut val = wc * px(ix, iy);
        val += wr * (px(ix - 1, iy) + px(ix + 1, iy) + px(ix, iy - 1) + px(ix, iy + 1));
        val += wd
            * (px(ix - 1, iy - 1) + px(ix + 1, iy - 1) + px(ix - 1, iy + 1) + px(ix + 1, iy + 1));
        val += w_big_r * (px(ix - 2, iy) + px(ix + 2, iy) + px(ix, iy - 2) + px(ix, iy + 2));
        val += wl
            * (px(ix - 2, iy - 1)
                + px(ix - 2, iy + 1)
                + px(ix + 2, iy - 1)
                + px(ix + 2, iy + 1)
                + px(ix - 1, iy - 2)
                + px(ix + 1, iy - 2)
                + px(ix - 1, iy + 2)
                + px(ix + 1, iy + 2));
        val += w_big_d
            * (px(ix - 2, iy - 2) + px(ix + 2, iy - 2) + px(ix - 2, iy + 2) + px(ix + 2, iy + 2));
        val
    };

    for y in 0..height {
        let iy = y as isize;

        if y < 2 || y >= height - 2 {
            for x in 0..width {
                output[y * width + x] = scalar_pixel(x as isize, iy);
            }
            continue;
        }

        for x in 0..2 {
            output[y * width + x] = scalar_pixel(x as isize, iy);
        }

        let r_m2 = (y - 2) * width;
        let r_m1 = (y - 1) * width;
        let r_0 = y * width;
        let r_p1 = (y + 1) * width;
        let r_p2 = (y + 2) * width;

        let simd_end = if width >= 12 { width - 10 } else { 2 };
        let mut x = 2;

        // The iteration structure is the vector body's, verbatim: chunks of
        // eight starting at 2 while the START is below `simd_end`, so the last
        // chunk may run past it. Reproducing the loop rather than computing an
        // end index is what keeps the boundary exactly where the vector tiers
        // put it.
        while x < simd_end {
            for i in 0..8 {
                let cx = x + i;
                let r_sum =
                    input[r_0 + cx - 1] + input[r_0 + cx + 1] + input[r_m1 + cx] + input[r_p1 + cx];
                let d_sum = input[r_m1 + cx - 1]
                    + input[r_m1 + cx + 1]
                    + input[r_p1 + cx - 1]
                    + input[r_p1 + cx + 1];
                let big_r_sum =
                    input[r_0 + cx - 2] + input[r_0 + cx + 2] + input[r_m2 + cx] + input[r_p2 + cx];
                let l_sum = input[r_m1 + cx - 2]
                    + input[r_p1 + cx - 2]
                    + input[r_m1 + cx + 2]
                    + input[r_p1 + cx + 2]
                    + input[r_m2 + cx - 1]
                    + input[r_m2 + cx + 1]
                    + input[r_p2 + cx - 1]
                    + input[r_p2 + cx + 1];
                let big_d_sum = input[r_m2 + cx - 2]
                    + input[r_m2 + cx + 2]
                    + input[r_p2 + cx - 2]
                    + input[r_p2 + cx + 2];

                // Same nesting as the vector combine: outermost `wc * center`,
                // innermost a plain multiply on the `w_big_d` tail.
                output[r_0 + cx] = fma(
                    wc,
                    input[r_0 + cx],
                    fma(
                        wr,
                        r_sum,
                        fma(
                            wd,
                            d_sum,
                            fma(w_big_r, big_r_sum, fma(wl, l_sum, w_big_d * big_d_sum)),
                        ),
                    ),
                );
            }
            x += 8;
        }

        while x < width {
            output[y * width + x] = scalar_pixel(x as isize, iy);
            x += 1;
        }
    }
}

// ============================================================================
// libjxl `Symmetric5` parity variant (strict-Libjxl strategy only)
// ============================================================================
//
// Bit-exact port of libjxl `convolve_symmetric5.cc`. The shipping kernel
// above diverges from the reference in three ways, all reproduced here:
//
// 1. **Border semantics** — libjxl wraps out-of-range coordinates with
//    `Mirror` (`x<0 → -x-1`, `x>=size → 2*size-1-x`, image_ops.h), i.e.
//    edge-pixel-ONCE reflection: read `-2` → `1`, `size` → `size-1`.
//    Ours clamps (edge replication): `-2` → `0`. Differing reads are
//    confined to the 2-pixel border ring.
// 2. **Accumulation order** — libjxl evaluates each of the five kernel
//    rows as a horizontal 1×5 weighted sum
//    `wx2*(in_m2+in_p2) + (wx1*(in_m1+in_p1) + wx0*in_00)`
//    (convolve_symmetric5.cc:66-69 — explicit `Mul`/`Add`, no FMA), then
//    combines `sum0 = WS(0) + WS(-2) + WS(-1)`, `sum1 = WS(+2) + WS(+1)`,
//    `out = sum0 + sum1`. Ours sums each distance class left-to-right
//    and FMA-chains the classes — different rounding on every pixel.
// 3. **Row weights** — libjxl's row triples are `(c, r, R)` on row 0,
//    `(R, L, D)` on rows ±2, `(r, d, L)` on rows ±1 (the `w0/w1/w4/w5/w8`
//    fields of `WeightsSymmetric5`). Passing our per-distance weights
//    through this mapping gives identical coverage.
//
// The libjxl `Symmetric5Row` driver splits each row into a scalar border
// (first `Lanes` + trailing pixels) and a vector interior, but both paths
// compute the same function — the split is purely an implementation
// detail. A single scalar loop reproduces it exactly.

/// Mirror-wrap a coordinate into `[0, size)` (libjxl `image_ops.h::Mirror`):
/// the mirror sits *outside* the edge pixel, so `-1 → 0`, `-2 → 1`,
/// `size → size-1`, `size+1 → size-2`. The kernel radius is 2, so at most
/// one reflection ever applies.
#[inline(always)]
fn mirror_coord(x: isize, size: usize) -> usize {
    debug_assert!(size != 0);
    let size = size as isize;
    let mut x = x;
    while !(0..size).contains(&x) {
        x = if x < 0 { -x - 1 } else { 2 * size - 1 - x };
    }
    x as usize
}

/// libjxl `WeightedSumBorder` — one horizontal 1×5 weighted sum at
/// `(ix, iy)` with weights `[wx2 wx1 wx0 wx1 wx2]` and mirror wrap on
/// both axes:
/// `wx2*(in_m2+in_p2) + (wx1*(in_m1+in_p1) + wx0*in_00)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn weighted_sum_border(
    input: &[f32],
    ix: isize,
    iy: isize,
    width: usize,
    height: usize,
    wx0: f32,
    wx1: f32,
    wx2: f32,
) -> f32 {
    let row = mirror_coord(iy, height) * width;
    let in_m2 = input[row + mirror_coord(ix - 2, width)];
    let in_p2 = input[row + mirror_coord(ix + 2, width)];
    let in_m1 = input[row + mirror_coord(ix - 1, width)];
    let in_p1 = input[row + mirror_coord(ix + 1, width)];
    let in_00 = input[row + ix as usize];
    let sum_2 = wx2 * (in_m2 + in_p2);
    let sum_1 = wx1 * (in_m1 + in_p1);
    let sum_0 = wx0 * in_00;
    sum_2 + (sum_1 + sum_0)
}

/// libjxl-bit-exact 5×5 gaborish inverse on one channel.
///
/// `data` is modified in place; `scratch` (≥ `width * height`) holds the
/// input copy. Same weight semantics as [`gaborish_5x5_channel`] —
/// `(wc, wr, wd, w_big_r, wl, w_big_d)` — but computed libjxl-style by
/// the caller (f32 `normalize` / `normalize_mul` chain, see
/// `vardct/gaborish.rs::compute_weights_libjxl`).
///
/// Scalar-only: the strict-Libjxl strategy trades wall-time for
/// byte-parity here; a SIMD port must preserve the exact accumulation
/// order above (no FMA fusion).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_channel_libjxl(
    data: &mut [f32],
    scratch: &mut [f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    let n = width * height;
    debug_assert!(data.len() >= n);
    debug_assert!(scratch.len() >= n);

    scratch[..n].copy_from_slice(&data[..n]);
    let input = &scratch[..n];

    for y in 0..height {
        let iy = y as isize;
        for x in 0..width {
            let ix = x as isize;
            // Row triples (wx0, wx1, wx2) per libjxl Symmetric5Border:
            //   row  0 : (c, r, R)   — center, orth-1, orth-2
            //   rows ±2: (R, L, D)   — orth-2, knight, corner
            //   rows ±1: (r, d, L)   — orth-1, diagonal, knight
            let mut sum0 = weighted_sum_border(input, ix, iy, width, height, wc, wr, w_big_r);
            sum0 += weighted_sum_border(input, ix, iy - 2, width, height, w_big_r, wl, w_big_d);
            let mut sum1 =
                weighted_sum_border(input, ix, iy + 2, width, height, w_big_r, wl, w_big_d);
            sum0 += weighted_sum_border(input, ix, iy - 1, width, height, wr, wd, wl);
            sum1 += weighted_sum_border(input, ix, iy + 1, width, height, wr, wd, wl);
            data[y * width + x] = sum0 + sum1;
        }
    }
}

// ============================================================================
// Backwards-compat suffixed re-exports
// ============================================================================
//
// Older callers spelled the variants `gaborish_5x5_avx2` / `_neon`.
// magetypes' tier names are `_v3` (AVX2) / `_neon` / `_wasm128`. Re-export
// under the historical names so external API stays stable.

#[cfg(target_arch = "x86_64")]
pub use gaborish_5x5_impl_v3 as gaborish_5x5_avx2;

#[cfg(target_arch = "aarch64")]
pub use gaborish_5x5_impl_neon as gaborish_5x5_neon;

// wasm128 is NEW (no pre-consolidation alias existed). Export under the
// magetypes-generated name so wasm32 callers can reach it directly if needed.
#[cfg(target_arch = "wasm32")]
pub use gaborish_5x5_impl_wasm128 as gaborish_5x5_wasm128;

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    extern crate std;
    use alloc::vec;

    /// libjxl gaborish weights (default `gab=1` mode, normalized to sum=1):
    /// from `lib/jxl/enc_gaborish.cc` — used by the tests below to exercise
    /// the kernel under realistic weights.
    fn default_weights() -> (f32, f32, f32, f32, f32, f32) {
        // Y channel weights — the test only needs a representative tuple, not
        // bit-exact libjxl values; what matters is the FMA chain is non-trivial.
        let wc = 1.0_f32;
        let wr = 0.115_416_72;
        let wd = 0.061_359_57;
        let w_big_r = 0.026_375_18;
        let wl = 0.005_125_56;
        let w_big_d = 0.001_660_99;
        let sum = wc + 4.0 * wr + 4.0 * wd + 4.0 * w_big_r + 8.0 * wl + 4.0 * w_big_d;
        (
            wc / sum,
            wr / sum,
            wd / sum,
            w_big_r / sum,
            wl / sum,
            w_big_d / sum,
        )
    }

    fn synthetic_plane(width: usize, height: usize) -> alloc::vec::Vec<f32> {
        let mut buf = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 7 + y * 13 + x * y) % 1000) as f32 / 1000.0;
                buf[y * width + x] = v * 0.5;
            }
        }
        buf
    }

    #[test]
    fn test_gaborish_5x5_simd_matches_scalar() {
        let (wc, wr, wd, w_big_r, wl, w_big_d) = default_weights();
        let width = 128;
        let height = 64;
        let input = synthetic_plane(width, height);

        // Scalar reference: read from `input`, write into a fresh buffer.
        let mut scalar_out = vec![0.0f32; width * height];
        gaborish_5x5_scalar(
            &mut scalar_out,
            &input,
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );

        // Dispatch — test all token permutations so every magetypes tier
        // available on this host runs at least once.
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut data = input.clone();
                let mut scratch = vec![0.0f32; width * height];
                gaborish_5x5_channel(
                    &mut data,
                    &mut scratch,
                    width,
                    height,
                    wc,
                    wr,
                    wd,
                    w_big_r,
                    wl,
                    w_big_d,
                );

                let mut max_abs = 0.0f32;
                for i in 0..width * height {
                    let diff = (data[i] - scalar_out[i]).abs();
                    max_abs = max_abs.max(diff);
                }

                // Same FMA chain at every tier; deltas should be on the
                // order of a few ulp (mul_add lowers to vfmadd on AVX2/NEON
                // and to mul+add on backends without FMA).
                assert!(
                    max_abs < 1e-4,
                    "SIMD vs scalar max_abs = {max_abs} [{perm}]",
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_gaborish_5x5_small_images_safe() {
        // Tiny images (below the SIMD threshold) must still run via scalar.
        let (wc, wr, wd, w_big_r, wl, w_big_d) = default_weights();
        for (w, h) in [(1, 1), (4, 4), (12, 4), (13, 4), (13, 5), (8, 8)] {
            let input = vec![0.25f32; w * h];
            let mut data = input.clone();
            let mut scratch = vec![0.0f32; w * h];
            gaborish_5x5_channel(
                &mut data,
                &mut scratch,
                w,
                h,
                wc,
                wr,
                wd,
                w_big_r,
                wl,
                w_big_d,
            );
            for v in &data {
                assert!(v.is_finite(), "{w}x{h}: produced non-finite value {v}");
            }
        }
    }

    #[test]
    fn test_gaborish_5x5_non_multiple_of_8_width() {
        // Width not a multiple of 8 — exercises the scalar right-edge remainder.
        let (wc, wr, wd, w_big_r, wl, w_big_d) = default_weights();
        let width = 37;
        let height = 19;
        let mut input = vec![0.0f32; width * height];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as f32 * 0.001).sin().abs() * 0.3;
        }

        let mut scalar_out = vec![0.0f32; width * height];
        gaborish_5x5_scalar(
            &mut scalar_out,
            &input,
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );

        let mut data = input.clone();
        let mut scratch = vec![0.0f32; width * height];
        gaborish_5x5_channel(
            &mut data,
            &mut scratch,
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );

        let mut max_abs = 0.0f32;
        for i in 0..width * height {
            let diff = (data[i] - scalar_out[i]).abs();
            max_abs = max_abs.max(diff);
        }
        assert!(max_abs < 1e-4, "non-mul-of-8 max_abs = {max_abs}");
    }

    /// Bit-exact golden for `gaborish_5x5_channel_libjxl` (strict-Libjxl
    /// parity kernel). Expected values were produced by the verified
    /// scalar transcription of libjxl `convolve_symmetric5.cc`
    /// `Symmetric5Border`/`Symmetric5Interior` — mirror borders, the
    /// `wx2*(m2+p2) + (wx1*(m1+p1) + wx0*c)` row grouping, and the
    /// `sum0 + sum1` row accumulation order are all load-bearing; any
    /// "optimization" that changes rounding (FMA fusion, reordered
    /// accumulation, different border fill) must fail this test.
    ///
    /// Weights are the libjxl `mul = 1.0` values from
    /// `enc_gaborish.cc::GaborishInverse` (f32 chain, 2026-09-17).
    #[test]
    fn test_gaborish_5x5_libjxl_golden() {
        const W: usize = 8;
        const H: usize = 6;
        let mut data = vec![0.0f32; W * H];
        for y in 0..H {
            for x in 0..W {
                data[y * W + x] = ((x * 7 + y * 13 + x * y) % 97) as f32 / 97.0 - 0.3;
            }
        }
        // mul=1.0 libjxl weights (enc_gaborish.cc, f32 chain):
        // normalize = 1/(1+4*(kG0+kG1+kG2+kG4+2*kG3)) ≈ 1.7951821.
        let (wc, wr, wd, w_big_r, wl, w_big_d) = (
            1.795_182_1_f32,
            -0.170_467_18,
            -0.073_659_42,
            0.024_611_956,
            0.011_687_005,
            -0.002_654_906_4,
        );
        let mut scratch = vec![0.0f32; W * H];
        gaborish_5x5_channel_libjxl(
            &mut data,
            &mut scratch,
            W,
            H,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );

        #[rustfmt::skip]
        const EXPECTED: [f32; W * H] = [
            -0.33458868, -0.24873106, -0.18139957, -0.11094986,
            -0.040500242, 0.02994942, 0.09728098, 0.18313858,
            -0.1739439, -0.073853396, 0.0055684447, 0.08848265,
            0.17405173, 0.24527884, 0.30008882, 0.3884922,
            -0.047396444, 0.064784355, 0.15360822, 0.24904658,
            0.3184561, 0.4602871, 0.7078911, 0.8963862,
            0.0849089, 0.21220908, 0.2992153, 0.366009,
            0.50646234, 0.9531444, -0.6696852, -0.3862596,
            0.21145628, 0.3391598, 0.5092274, 0.7434328,
            1.0771476, -0.54044515, -0.09141564, 0.094367445,
            0.3721011, 0.4867707, 0.8510828, -0.65723765,
            -0.22305709, 0.00023879576, 0.15778476, 0.28803408,
        ];
        for (i, (&got, &want)) in data.iter().zip(EXPECTED.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "pixel {i} (x={}, y={}): {got:?} != {want:?}",
                i % W,
                i / W,
            );
        }
    }
}

#[cfg(test)]
mod expanded_coverage {
    use super::*;
    use crate::test_helpers::*;
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    // Normalized 5x5 gaborish weights (must sum to 1.0).
    const WC: f32 = 0.685_3;
    const WR: f32 = 0.034_5;
    const WD: f32 = 0.018_7;
    const W_BIG_R: f32 = 0.012_2;
    const WL: f32 = 0.009_1;
    const W_BIG_D: f32 = 0.004_8;

    /// Sweep image sizes including kernel-boundary cases.
    /// Every tier of `gaborish_5x5_impl` must produce BIT-IDENTICAL output.
    ///
    /// This compares tier against TIER, not tier against
    /// `gaborish_5x5_scalar` — and that distinction is the whole point here.
    /// The shipped kernel deliberately uses two different summation orders:
    /// left-to-right for border pixels (`scalar_pixel`) and a nested FMA chain
    /// for the interior. `gaborish_5x5_scalar` uses the left-to-right form
    /// everywhere, so it differs from the interior by up to 40 ULP by design,
    /// and `gaborish_5x5_scalar_vs_dispatch_sizes` below tolerates that with a
    /// 16-ULP + 1e-4 bound. What must NOT differ is one tier from another,
    /// because `hash_lock_expected.txt` is one committed sidecar and this
    /// kernel is on the byte-affecting path.
    ///
    /// It caught the generated `_scalar` tier diverging by up to 49 ULP
    /// (magetypes' scalar `mul_add` is unfused); see
    /// `gaborish_5x5_impl_scalar`.
    ///
    /// Skipped on wasm32 for the reason in `docs/SIMD_PARITY_KNOWN_DIVERGENCES.md`
    /// (xyb-001): WASM SIMD has no FMA instruction, so its tier cannot agree
    /// with a fused one.
    #[test]
    #[cfg_attr(
        target_arch = "wasm32",
        ignore = "FIXME(SIMD-parity): xyb-001 — WASM SIMD has no FMA instruction; see docs/SIMD_PARITY_KNOWN_DIVERGENCES.md"
    )]
    fn gaborish_5x5_dispatch_is_bit_identical_across_tiers() {
        // Sizes that straddle the width<13/height<5 early-out, the 2-pixel
        // borders and the 8-wide interior chunking.
        for &(w, h) in &[
            (5_usize, 5_usize),
            (12, 5),
            (13, 5),
            (16, 16),
            (17, 17),
            (23, 9),
            (24, 9),
            (25, 9),
            (32, 16),
            (33, 17),
            (64, 32),
        ] {
            let n = w * h;
            let input = gen_f32_unit(0xC0FF_EE00 ^ n as u64, n);
            let mut first: Option<alloc::vec::Vec<f32>> = None;
            run_dispatch_parity(|perm| {
                let mut out = input.clone();
                let mut scratch = vec![0.0_f32; n];
                gaborish_5x5_channel(
                    &mut out,
                    &mut scratch,
                    w,
                    h,
                    WC,
                    WR,
                    WD,
                    W_BIG_R,
                    WL,
                    W_BIG_D,
                );
                match &first {
                    None => first = Some(out),
                    Some(base) => {
                        for i in 0..n {
                            assert_eq!(
                                base[i].to_bits(),
                                out[i].to_bits(),
                                "{perm}: {w}x{h} lane {i}: this tier gives {:?} where the first \
                                 permutation gave {:?} — the tiers must agree bitwise or \
                                 hash_lock_expected.txt cannot hold across architectures",
                                out[i],
                                base[i]
                            );
                        }
                    }
                }
            });
        }
    }

    #[test]
    fn gaborish_5x5_scalar_vs_dispatch_sizes() {
        let cases: &[(usize, usize)] = &[
            (1, 1),
            (5, 5),
            (8, 8),
            (9, 9),
            (16, 16),
            (17, 17),
            (32, 16),
            (33, 17),
            (64, 32),
        ];
        for &(w, h) in cases {
            let n = w * h;
            let input: Vec<f32> = gen_f32(0x6A60_F00D ^ ((w as u64) << 32) ^ h as u64, n, 5.0);

            let mut ref_out = input.clone();
            let mut ref_scratch = vec![0.0_f32; n];
            ref_scratch.copy_from_slice(&ref_out);
            gaborish_5x5_scalar(
                &mut ref_out,
                &ref_scratch,
                w,
                h,
                WC,
                WR,
                WD,
                W_BIG_R,
                WL,
                W_BIG_D,
            );

            run_dispatch_parity(|perm| {
                let mut act_out = input.clone();
                let mut act_scratch = vec![0.0_f32; n];
                gaborish_5x5_channel(
                    &mut act_out,
                    &mut act_scratch,
                    w,
                    h,
                    WC,
                    WR,
                    WD,
                    W_BIG_R,
                    WL,
                    W_BIG_D,
                );
                assert_f32_slice_close_ulps_abs(
                    &ref_out,
                    &act_out,
                    16,
                    1e-4,
                    perm,
                    &format!("gab5x5({w}x{h})"),
                );
            });
        }
    }
}
