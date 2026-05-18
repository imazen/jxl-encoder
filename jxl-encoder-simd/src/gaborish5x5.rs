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

#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
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
}
