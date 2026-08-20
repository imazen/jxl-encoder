// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated pixel-domain loss computation for AC strategy estimation.
//!
//! Computes the 8th-power-norm of masked pixel errors:
//!   channel_loss = sum_over_pixels( ((mask[px] + offset) * error[px])^8 )
//!
//! Inner multiply is in f32; the squaring is done in f64 for precision:
//!   m2 = (masked * masked) as f64
//!   m4 = m2 * m2
//!   m8 = m4 * m4
//!
//! **magetypes-consolidated** (W43-2 chunk-5): one `#[magetypes(...)]` body
//! generates every per-arch SIMD variant. The body operates on `f32x8` +
//! `f64x4` generics; on backends without native 256-bit registers (NEON,
//! WASM128, scalar) magetypes polyfills both to 2× narrower ops.
//!
//! The pre-consolidation crate had three nearly-identical hand-written
//! bodies (AVX2 / NEON / WASM128) plus a scalar fallback. The consolidated
//! path collapses them to one source of truth.
//!
//! **AVX-512 (`v4`) note:** magetypes 0.9.23 does not implement
//! `F64x4Backend` for `X64V4Token` / `X64V4xToken` — the natural f64 width
//! on AVX-512 is `f64x8` (one 512-bit register). Selecting `v4` here
//! would force a polyfill that doesn't exist. The `v3` AVX2 path is the
//! ceiling on x86_64; revisit when magetypes gains a v4 f64x4 polyfill
//! or rewrite around `f64x8` if a future kernel needs 8-wide f64.

use archmage::prelude::*;

/// Compute pixel-domain loss for one channel of a block.
///
/// For each pixel: channel_loss += ((mask_val + mask_offset) * error_val)^8
///
/// The inner multiply is in f32; then squared three times — first squaring
/// promotes to f64 for precision, matching libjxl's `EstimateEntropy`.
///
/// `pixel_error`: error values, row-major, `block_width * block_height` elements
/// `mask`: full mask1x1 buffer (stride = `mask_stride`)
/// `mask_row_base`: `pixel_y * mask_stride + pixel_x` (start of this block in mask)
/// `mask_offset`: channel-specific offset added to mask values
/// `block_width`: pixels per row (always multiple of 8)
/// `block_height`: number of rows
///
/// Returns the channel loss as f64.
#[inline]
pub fn pixel_domain_loss(
    pixel_error: &[f32],
    mask: &[f32],
    mask_row_base: usize,
    mask_stride: usize,
    mask_offset: f32,
    block_width: usize,
    block_height: usize,
) -> f64 {
    debug_assert!(
        block_width.is_multiple_of(8),
        "block_width must be multiple of 8"
    );
    debug_assert!(pixel_error.len() >= block_width * block_height);

    // Dispatch through incant! — picks the best magetypes-generated variant
    // at runtime. Falls through to _scalar on platforms without a SIMD token.
    //
    // Explicit tier list omits `v4` (AVX-512) because magetypes 0.9.23 has
    // no `F64x4Backend` for `X64V4Token`; the kernel uses `f64x4` so the
    // ceiling on x86_64 is `v3` (AVX2). Listing tiers explicitly also
    // silences the v0.9.9 `incant!` deprecation warning about implicit
    // `scalar`.
    incant!(
        pixel_domain_loss_impl(
            pixel_error,
            mask,
            mask_row_base,
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        ),
        [v4, v3, neon, wasm128, scalar]
    )
}

// ============================================================================
// Scalar fallback
// ============================================================================

/// Scalar emulation of the canonical kernel — 8 virtual f32 lanes
/// (lane = dx & 7) + the same fixed combine tree, so it is bit-identical
/// to every SIMD tier (used as the parity-test reference).
#[inline]
pub fn pixel_domain_loss_scalar(
    pixel_error: &[f32],
    mask: &[f32],
    mask_row_base: usize,
    mask_stride: usize,
    mask_offset: f32,
    block_width: usize,
    block_height: usize,
) -> f64 {
    let mut lanes = [0.0f32; 8];
    for py in 0..block_height {
        let mask_row_start = mask_row_base + py * mask_stride;
        let error_row_start = py * block_width;
        let mask_row = &mask[mask_row_start..mask_row_start + block_width];
        let error_row = &pixel_error[error_row_start..error_row_start + block_width];
        let (mask_chunks, _) = mask_row.as_chunks::<8>();
        let (error_chunks, _) = error_row.as_chunks::<8>();
        for (chunk_i, (mask_chunk, error_chunk)) in mask_chunks.iter().zip(error_chunks).enumerate()
        {
            let _ = chunk_i;
            for j in 0..8 {
                let masked = (mask_chunk[j] + mask_offset) * error_chunk[j];
                let m2 = masked * masked;
                let m4 = m2 * m2;
                let m8 = m4 * m4;
                lanes[j] += m8;
            }
        }
    }
    let s4 = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    let total = (s4[0] + s4[2]) + (s4[1] + s4[3]);
    total as f64
}

// ============================================================================
// magetypes-consolidated SIMD implementation
// ============================================================================
//
// Single body, one source of truth. The `#[magetypes(...)]` macro generates
// one `#[arcane]`-wrapped variant per listed tier:
//   - `pixel_domain_loss_impl_v3`      (x86_64 AVX2, native 256-bit
//                                       f32x8 + f64x4)
//   - `pixel_domain_loss_impl_neon`    (aarch64, 2× f32x4 polyfill of
//                                       f32x8 and 2× f64x2 polyfill of
//                                       f64x4)
//   - `pixel_domain_loss_impl_wasm128` (wasm32, same polyfill shape as
//                                       NEON)
//   - `pixel_domain_loss_impl_scalar`  (portable scalar fallback)
//
// `define(f32x8, f64x4)` injects type aliases substituted per tier. The
// body promotes f32→f64 via the array round-trip `f32x8::to_array()` →
// `[f64; 4]` literals → `f64x4::from_array(...)`; on AVX2 LLVM fuses
// the store-load into a single `vcvtps2pd` pair (matching the previous
// hand-written intrinsic path bit-for-bit) and on smaller backends it
// lowers to whatever the f64x2 polyfill expects.
//
// Accumulator structure (`acc_lo` / `acc_hi`) preserves the per-half
// accumulation grouping of the pre-consolidation AVX2 body — lanes 0-3
// sum into `acc_lo`, lanes 4-7 into `acc_hi`, then `reduce_add()`
// on each followed by a final `+`. The 8th-power computation order
// (m2 = masked·masked, m4 = m2·m2, m8 = m4·m4) is the manual
// `x²·x²·x²` chain, **not** a single `powi(8)` — preserving libjxl's
// rounding behaviour exactly.

/// Canonical arch-stable pixel-domain loss (2026-08-18): PURE f32 like
/// libjxl `EstimateEntropy` (enc_ac_strategy.cc masku loop — masku*err,
/// then three squarings to the 8th power, f32 accumulate). The previous
/// body promoted to f64x4 after m² — BOTH a libjxl divergence (they never
/// leave f32) and the hottest cost in the per-candidate estimate (f64
/// converts + twice the accumulate width; perf-annotate showed the
/// cvtps2pd/mulpd chains inside the find_best_16x16 bucket).
///
/// Determinism contract (same as the canonical entropy kernels): ONE
/// f32x8 accumulator (lane = dx & 7 — row chunks are 8-aligned since
/// block_width is a multiple of 8), lane-pure ops on every tier
/// (magetypes polyfills 4-wide arches as f32x4 pairs), and a FIXED
/// scalar combine tree — bit-identical across x86_64/aarch64/wasm/scalar
/// by construction. Changing the accumulator structure or tree changes
/// encoded bytes on every arch at once.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub fn pixel_domain_loss_impl(
    token: Token,
    pixel_error: &[f32],
    mask: &[f32],
    mask_row_base: usize,
    mask_stride: usize,
    mask_offset: f32,
    block_width: usize,
    block_height: usize,
) -> f64 {
    let offset_v = f32x8::splat(token, mask_offset);
    let mut acc = f32x8::zero(token);

    for py in 0..block_height {
        let mask_row_start = mask_row_base + py * mask_stride;
        let error_row_start = py * block_width;
        // Pre-slice rows so the compiler can prove SIMD loads are in-bounds.
        let mask_row = &mask[mask_row_start..mask_row_start + block_width];
        let error_row = &pixel_error[error_row_start..error_row_start + block_width];

        let (mask_chunks, _) = mask_row.as_chunks::<8>();
        let (error_chunks, _) = error_row.as_chunks::<8>();
        for (mask_chunk, error_chunk) in mask_chunks.iter().zip(error_chunks) {
            let mask_v = f32x8::from_slice(token, mask_chunk);
            let error_v = f32x8::from_slice(token, error_chunk);

            // libjxl order: masku = mask + offset; in = masku * err;
            // in = in*in (x3) -> masked^8, all f32.
            let masked = (mask_v + offset_v) * error_v;
            let m2 = masked * masked;
            let m4 = m2 * m2;
            let m8 = m4 * m4;
            acc += m8;
        }
    }

    // Canonical combine: FIXED scalar tree over the 8 virtual lanes,
    // promoted to f64 only at the very end (the caller's channel-mul
    // stays f64).
    let mut lanes = [0.0f32; 8];
    acc.store(&mut lanes);
    let s4 = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    let total = (s4[0] + s4[2]) + (s4[1] + s4[3]);
    total as f64
}

// ============================================================================
// Backwards-compat suffixed re-exports
// ============================================================================
//
// Older callers spelled the variants `pixel_domain_loss_avx2` / `_neon` /
// `_wasm128`. magetypes' tier names are `_v3` (AVX2) / `_neon` / `_wasm128`.
// Re-export under the historical names so external API stays stable.

#[cfg(target_arch = "x86_64")]
pub use pixel_domain_loss_impl_v3 as pixel_domain_loss_avx2;

#[cfg(target_arch = "aarch64")]
pub use pixel_domain_loss_impl_neon as pixel_domain_loss_neon;

#[cfg(target_arch = "wasm32")]
pub use pixel_domain_loss_impl_wasm128 as pixel_domain_loss_wasm128;

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use alloc::vec;

    #[test]
    fn test_pixel_domain_loss_uniform() {
        let block_width = 8;
        let block_height = 8;
        let mask_stride = 16;

        let pixel_error = vec![1.0f32; block_width * block_height];
        let mask = vec![0.5f32; mask_stride * 16];
        let mask_offset = 0.5f32;

        let result = pixel_domain_loss(
            &pixel_error,
            &mask,
            0,
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );

        // masked = (0.5 + 0.5) * 1.0 = 1.0
        // m2 = 1.0, m4 = 1.0, m8 = 1.0
        // 64 pixels × 1.0 = 64.0
        assert!(
            (result - 64.0).abs() < 1e-6,
            "Expected 64.0, got {}",
            result
        );
    }

    #[test]
    fn test_pixel_domain_loss_matches_scalar() {
        let block_width = 16;
        let block_height = 8;
        let mask_stride = 32;

        // Use varied data
        let mut pixel_error = vec![0.0f32; block_width * block_height];
        let mut mask = vec![0.0f32; mask_stride * 16];
        for (i, val) in pixel_error.iter_mut().enumerate() {
            *val = (i as f32 * 0.1 + 0.5) * if i % 3 == 0 { -1.0 } else { 1.0 };
        }
        for (i, val) in mask.iter_mut().enumerate() {
            *val = (i as f32 * 0.01 + 0.3).sin().abs();
        }
        let mask_offset = 0.7f32;

        let scalar_result = pixel_domain_loss_scalar(
            &pixel_error,
            &mask,
            0,
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );

        // Test all token permutations so every magetypes tier available on
        // this host runs at least once.
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let simd_result = pixel_domain_loss(
                    &pixel_error,
                    &mask,
                    0,
                    mask_stride,
                    mask_offset,
                    block_width,
                    block_height,
                );

                let rel_err = ((simd_result - scalar_result) / scalar_result.max(1e-20)).abs();
                assert!(
                    rel_err < 1e-6,
                    "SIMD ({}) vs scalar ({}) relative error {} too large [{perm}]",
                    simd_result,
                    scalar_result,
                    rel_err
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_pixel_domain_loss_16x16() {
        let block_width = 16;
        let block_height = 16;
        let mask_stride = 32;

        let pixel_error = vec![0.5f32; block_width * block_height];
        let mask = vec![1.0f32; mask_stride * 32];
        let mask_offset = 0.0f32;

        let result = pixel_domain_loss(
            &pixel_error,
            &mask,
            0,
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );

        // masked = 1.0 * 0.5 = 0.5
        // m8 = 0.5^8 = 1/256 = 0.00390625
        // 256 pixels × 0.00390625 = 1.0
        assert!((result - 1.0).abs() < 1e-6, "Expected 1.0, got {}", result);
    }
}

#[cfg(test)]
mod expanded_coverage {
    use super::*;
    use crate::test_helpers::*;

    /// Sweep block dimensions (width must be multiple of 8) + edge-value
    /// inputs.  The kernel computes loss = sum((mask+offset)^2 * err^2)^8
    /// summed; FMA association makes bit-exactness impossible but small
    /// relative tolerance catches structural bugs.
    #[test]
    fn pixel_domain_loss_scalar_vs_dispatch_block_sizes() {
        let cases: &[(usize, usize)] = &[
            (8, 8),   // 1 SIMD chunk
            (16, 8),  // 2 chunks wide
            (8, 16),  // 2 chunks tall
            (32, 32), // 4x4 chunks
            (8, 64),  // tall + multiple rows
            (64, 8),  // wide + single row
        ];
        let mask_offset = 0.5_f32;
        for &(bw, bh) in cases {
            let n = bw * bh;
            let pixel_error: alloc::vec::Vec<f32> =
                gen_f32(0xE220_1234 ^ ((bw as u64) << 32) ^ bh as u64, n, 0.5);
            // Pad mask to a stride larger than bw to exercise mask_stride
            // != block_width path.
            let mask_stride = bw + 16;
            let mask_h = bh + 4;
            let mask: alloc::vec::Vec<f32> = gen_f32_unit(
                0xFACE_5555 ^ ((bw as u64) << 32) ^ bh as u64,
                mask_stride * mask_h,
            );
            let mask_row_base = 2 * mask_stride; // offset into mask

            let ref_loss = pixel_domain_loss_scalar(
                &pixel_error,
                &mask,
                mask_row_base,
                mask_stride,
                mask_offset,
                bw,
                bh,
            );

            run_dispatch_parity(|perm| {
                let act_loss = pixel_domain_loss(
                    &pixel_error,
                    &mask,
                    mask_row_base,
                    mask_stride,
                    mask_offset,
                    bw,
                    bh,
                );
                // 8th-power sums are extremely sensitive to ULP noise; compare
                // by relative error.
                let rel_err = if ref_loss.abs() > 1e-20 {
                    ((act_loss - ref_loss).abs() / ref_loss.abs()) as f32
                } else {
                    (act_loss - ref_loss).abs() as f32
                };
                assert!(
                    rel_err < 1e-3,
                    "pixel_loss rel divergence ({bw}x{bh}): ref={} act={} rel_err={} perm={perm}",
                    ref_loss,
                    act_loss,
                    rel_err
                );
            });
        }
    }

    /// Zero error → zero loss.  Critical short-circuit invariant.
    #[test]
    fn pixel_domain_loss_zero_error_yields_zero() {
        let bw = 16;
        let bh = 16;
        let pixel_error = alloc::vec![0.0_f32; bw * bh];
        let mask = alloc::vec![1.0_f32; bw * bh];
        let ref_loss = pixel_domain_loss_scalar(&pixel_error, &mask, 0, bw, 0.5, bw, bh);
        assert_eq!(ref_loss, 0.0);
        run_dispatch_parity(|perm| {
            let act_loss = pixel_domain_loss(&pixel_error, &mask, 0, bw, 0.5, bw, bh);
            assert_eq!(act_loss, 0.0, "perm={perm}");
        });
    }
}
