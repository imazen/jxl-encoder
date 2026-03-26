// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Fused DCT8 + entropy estimation kernel.
//!
//! For DCT8 (the most frequently evaluated strategy), this kernel fuses pixel
//! extraction, forward DCT, DC zeroing, CfL decorrelation, quantization, and
//! entropy accumulation into a single pass. On AVX2, the DCT output stays in
//! YMM registers — never written to scratch memory — eliminating ~128 writes
//! and ~192 reads per block (384 writes + 576 reads per 3-channel evaluation).
//!
//! Only used in pixel-domain loss mode (default-on). Coefficient-domain mode
//! falls back to the separate DCT + entropy path.

use crate::entropy::EntropyCoeffResult;

/// Fused DCT8 + entropy estimation.
///
/// Loads 8x8 pixel block from strided plane, performs forward DCT, zeros DC,
/// and computes entropy estimation with CfL decorrelation — all in one pass.
///
/// `inv_weights` contains precomputed reciprocals (1/quant_weight) to replace
/// per-coefficient SIMD division with multiplication.
///
/// For Y channel (`cmap_factor == 0.0`): stores DCT output to `dct_output`.
/// For X/B channels: reads Y DCT from `y_dct` for CfL decorrelation.
///
/// Only computes pixel-domain statistics (no info_loss/info_loss2).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn fused_dct8_entropy(
    plane: &[f32],
    plane_stride: usize,
    bx: usize,
    by: usize,
    y_dct: &[f32; 64],
    weights: &[f32; 64],
    inv_weights: &[f32; 64],
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    error_coeffs: &mut [f32; 64],
    dct_output: Option<&mut [f32; 64]>,
) -> EntropyCoeffResult {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return fused_dct8_entropy_avx2(
                token,
                plane,
                plane_stride,
                bx,
                by,
                y_dct,
                weights,
                inv_weights,
                cmap_factor,
                quant,
                k_cost_delta,
                error_coeffs,
                dct_output,
            );
        }
    }

    fused_dct8_entropy_fallback(
        plane,
        plane_stride,
        bx,
        by,
        y_dct,
        weights,
        inv_weights,
        cmap_factor,
        quant,
        k_cost_delta,
        error_coeffs,
        dct_output,
    )
}

/// Fallback: extract + separate DCT + entropy using dispatching functions.
///
/// On NEON/WASM machines, the DCT and entropy kernels each pick their own
/// best SIMD path. On scalar-only machines, everything is scalar.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn fused_dct8_entropy_fallback(
    plane: &[f32],
    plane_stride: usize,
    bx: usize,
    by: usize,
    y_dct: &[f32; 64],
    weights: &[f32; 64],
    inv_weights: &[f32; 64],
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    error_coeffs: &mut [f32; 64],
    dct_output: Option<&mut [f32; 64]>,
) -> EntropyCoeffResult {
    // Extract 8x8 from strided plane
    let mut input = crate::scratch_buf::<64>();
    let x0 = bx * 8;
    for dy in 0..8 {
        let src = (by * 8 + dy) * plane_stride + x0;
        input[dy * 8..dy * 8 + 8].copy_from_slice(&plane[src..src + 8]);
    }

    // Forward DCT (dispatches to NEON/WASM/scalar internally)
    let mut dct = crate::scratch_buf::<64>();
    crate::dct_8x8(&input, &mut dct);

    // Zero DC coefficient
    dct[0] = 0.0;

    // Store DCT output if requested (Y channel for CfL reference)
    if let Some(out) = dct_output {
        out.copy_from_slice(&dct);
    }

    // Entropy estimation (dispatches to NEON/WASM/scalar internally)
    crate::entropy_estimate_coeffs(
        &dct,
        y_dct,
        weights,
        inv_weights,
        64,
        cmap_factor,
        quant,
        k_cost_delta,
        0.0, // k_cost2 unused in pixel-domain mode
        true,
        error_coeffs,
    )
}

// ============================================================================
// x86_64 AVX2 fused implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
pub fn fused_dct8_entropy_avx2(
    token: archmage::X64V3Token,
    plane: &[f32],
    plane_stride: usize,
    bx: usize,
    by: usize,
    y_dct: &[f32; 64],
    weights: &[f32; 64],
    inv_weights: &[f32; 64],
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    error_coeffs: &mut [f32; 64],
    dct_output: Option<&mut [f32; 64]>,
) -> EntropyCoeffResult {
    use magetypes::simd::f32x8;

    // Load 8 rows from strided plane (eliminates extract_block_8x8 copy)
    let x0 = bx * 8;
    let base = by * 8 * plane_stride + x0;
    let r0 = crate::load_f32x8(token, plane, base);
    let r1 = crate::load_f32x8(token, plane, base + plane_stride);
    let r2 = crate::load_f32x8(token, plane, base + 2 * plane_stride);
    let r3 = crate::load_f32x8(token, plane, base + 3 * plane_stride);
    let r4 = crate::load_f32x8(token, plane, base + 4 * plane_stride);
    let r5 = crate::load_f32x8(token, plane, base + 5 * plane_stride);
    let r6 = crate::load_f32x8(token, plane, base + 6 * plane_stride);
    let r7 = crate::load_f32x8(token, plane, base + 7 * plane_stride);

    // Column DCT
    let (r0, r1, r2, r3, r4, r5, r6, r7) =
        crate::dct8::vectorized_dct1d_8(token, r0, r1, r2, r3, r4, r5, r6, r7);

    // Scale by 1/8
    let scale = f32x8::splat(token, 0.125);
    let r0 = r0 * scale;
    let r1 = r1 * scale;
    let r2 = r2 * scale;
    let r3 = r3 * scale;
    let r4 = r4 * scale;
    let r5 = r5 * scale;
    let r6 = r6 * scale;
    let r7 = r7 * scale;

    // Transpose
    let (r0, r1, r2, r3, r4, r5, r6, r7) =
        crate::dct8::transpose_8x8_regs(token, r0, r1, r2, r3, r4, r5, r6, r7);

    // Row DCT
    let (r0, r1, r2, r3, r4, r5, r6, r7) =
        crate::dct8::vectorized_dct1d_8(token, r0, r1, r2, r3, r4, r5, r6, r7);

    // Scale by 1/8
    let r0 = r0 * scale;
    let r1 = r1 * scale;
    let r2 = r2 * scale;
    let r3 = r3 * scale;
    let r4 = r4 * scale;
    let r5 = r5 * scale;
    let r6 = r6 * scale;
    let r7 = r7 * scale;

    // Zero DC: element [0] of r0 in transposed layout. Multiply by mask [0,1,1,1,1,1,1,1].
    let dc_mask = f32x8::from_array(token, [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let r0 = r0 * dc_mask;

    // Store DCT output if requested (Y channel for CfL reference)
    if let Some(out) = dct_output {
        crate::store_f32x8(out, 0, r0);
        crate::store_f32x8(out, 8, r1);
        crate::store_f32x8(out, 16, r2);
        crate::store_f32x8(out, 24, r3);
        crate::store_f32x8(out, 32, r4);
        crate::store_f32x8(out, 40, r5);
        crate::store_f32x8(out, 48, r6);
        crate::store_f32x8(out, 56, r7);
    }

    // Row-by-row entropy estimation from DCT registers.
    // Each row is one f32x8 register — no memory round-trip for DCT coefficients.
    let cmap_v = f32x8::splat(token, cmap_factor);
    let quant_v = f32x8::splat(token, quant);
    let cost_delta_v = f32x8::splat(token, k_cost_delta);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);

    let mut entropy_acc = f32x8::zero(token);
    let mut nzeros_acc = f32x8::zero(token);

    // Process one row of DCT output: quantize, accumulate entropy, write error coefficients.
    macro_rules! process_row {
        ($dct_row:expr, $row_idx:expr) => {{
            let base = $row_idx * 8;
            let w = crate::load_f32x8(token, weights, base);
            let iw = crate::load_f32x8(token, inv_weights, base);
            let y = crate::load_f32x8(token, y_dct, base);

            // val = (dct_c - y * cmap_factor) * inv_weight * quant
            let adjusted = $dct_row - y * cmap_v;
            let val = adjusted * iw * quant_v;

            let rval = val.round();
            let diff = val - rval;

            // Write error coefficients: weight * (val - round(val))
            let err = w * diff;
            crate::store_f32x8(error_coeffs, base, err);

            // Entropy: sqrt(|round(val)|) * cost_delta
            let q = rval.abs();
            entropy_acc = q.sqrt().mul_add(cost_delta_v, entropy_acc);

            // Count non-zero quantized coefficients
            let nz_mask = q.simd_ne(zero);
            nzeros_acc += f32x8::blend(nz_mask, one, zero);
        }};
    }

    process_row!(r0, 0);
    process_row!(r1, 1);
    process_row!(r2, 2);
    process_row!(r3, 3);
    process_row!(r4, 4);
    process_row!(r5, 5);
    process_row!(r6, 6);
    process_row!(r7, 7);

    EntropyCoeffResult {
        entropy_sum: entropy_acc.reduce_add(),
        nzeros_sum: nzeros_acc.reduce_add(),
        info_loss_sum: 0.0,
        info_loss2_sum: 0.0,
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    extern crate std;
    use super::*;

    /// Verify fused kernel produces identical results to separate DCT + entropy path.
    #[test]
    fn test_fused_matches_separate() {
        // Synthetic 16x16 plane (padded beyond 8x8 for stride)
        let stride = 16;
        let mut plane = alloc::vec![0.0f32; stride * 16];
        for y in 0..8 {
            for x in 0..8 {
                plane[y * stride + x] = ((y * 8 + x) as f32 * 0.37 + 1.5).cos();
            }
        }

        let y_dct = [0.0f32; 64]; // Y channel: no CfL
        let mut weights = [1.0f32; 64];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = 0.5 + (i as f32 * 0.1).sin().abs() * 2.0;
        }
        weights[0] = 1.0; // DC weight (will be effectively zeroed)
        let mut inv_weights = [0.0f32; 64];
        for (i, iw) in inv_weights.iter_mut().enumerate() {
            *iw = 1.0 / weights[i];
        }

        let cmap_factor = 0.0f32;
        let quant = 3.0f32;
        let k_cost_delta = 5.335;

        // Fused path
        let mut fused_error = [0.0f32; 64];
        let mut fused_dct_out = [0.0f32; 64];
        let fused_result = fused_dct8_entropy(
            &plane,
            stride,
            0,
            0,
            &y_dct,
            &weights,
            &inv_weights,
            cmap_factor,
            quant,
            k_cost_delta,
            &mut fused_error,
            Some(&mut fused_dct_out),
        );

        // Separate path: extract + DCT + zero DC + entropy
        let mut input = [0.0f32; 64];
        for dy in 0..8 {
            input[dy * 8..dy * 8 + 8].copy_from_slice(&plane[dy * stride..dy * stride + 8]);
        }
        let mut sep_dct = [0.0f32; 64];
        crate::dct_8x8(&input, &mut sep_dct);
        sep_dct[0] = 0.0;

        let mut sep_error = [0.0f32; 64];
        let sep_result = crate::entropy_estimate_coeffs(
            &sep_dct,
            &y_dct,
            &weights,
            &inv_weights,
            64,
            cmap_factor,
            quant,
            k_cost_delta,
            0.0,
            true,
            &mut sep_error,
        );

        // DCT output must match
        for i in 0..64 {
            assert!(
                (fused_dct_out[i] - sep_dct[i]).abs() < 1e-6,
                "DCT mismatch at {i}: fused={} sep={}",
                fused_dct_out[i],
                sep_dct[i]
            );
        }

        // Error coefficients must match
        for i in 0..64 {
            assert!(
                (fused_error[i] - sep_error[i]).abs() < 1e-5,
                "Error coeff mismatch at {i}: fused={} sep={}",
                fused_error[i],
                sep_error[i]
            );
        }

        // Entropy results must match
        assert!(
            (fused_result.entropy_sum - sep_result.entropy_sum).abs() < 1e-4,
            "Entropy mismatch: fused={} sep={}",
            fused_result.entropy_sum,
            sep_result.entropy_sum
        );
        assert!(
            (fused_result.nzeros_sum - sep_result.nzeros_sum).abs() < 1e-6,
            "Nzeros mismatch: fused={} sep={}",
            fused_result.nzeros_sum,
            sep_result.nzeros_sum
        );
    }

    /// Verify fused kernel with CfL (non-zero cmap_factor) matches separate path.
    #[test]
    fn test_fused_cfl_matches_separate() {
        let stride = 16;
        let mut plane = alloc::vec![0.0f32; stride * 16];
        for y in 0..8 {
            for x in 0..8 {
                plane[y * stride + x] = ((y * 3 + x * 7) as f32 * 0.13).sin();
            }
        }

        // Pre-computed Y channel DCT (simulating Y-first processing)
        let mut y_dct = [0.0f32; 64];
        {
            let mut y_plane = alloc::vec![0.0f32; stride * 16];
            for y in 0..8 {
                for x in 0..8 {
                    y_plane[y * stride + x] = ((y + x) as f32 * 0.5).cos();
                }
            }
            let mut y_input = [0.0f32; 64];
            for dy in 0..8 {
                y_input[dy * 8..dy * 8 + 8].copy_from_slice(&y_plane[dy * stride..dy * stride + 8]);
            }
            crate::dct_8x8(&y_input, &mut y_dct);
            y_dct[0] = 0.0;
        }

        let mut weights = [1.0f32; 64];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = 0.3 + (i as f32 * 0.2).cos().abs() * 3.0;
        }
        let mut inv_weights = [0.0f32; 64];
        for (i, iw) in inv_weights.iter_mut().enumerate() {
            *iw = 1.0 / weights[i];
        }

        let cmap_factor = 0.35f32;
        let quant = 2.5f32;
        let k_cost_delta = 5.335;

        // Fused path
        let mut fused_error = [0.0f32; 64];
        let fused_result = fused_dct8_entropy(
            &plane,
            stride,
            0,
            0,
            &y_dct,
            &weights,
            &inv_weights,
            cmap_factor,
            quant,
            k_cost_delta,
            &mut fused_error,
            None,
        );

        // Separate path
        let mut input = [0.0f32; 64];
        for dy in 0..8 {
            input[dy * 8..dy * 8 + 8].copy_from_slice(&plane[dy * stride..dy * stride + 8]);
        }
        let mut sep_dct = [0.0f32; 64];
        crate::dct_8x8(&input, &mut sep_dct);
        sep_dct[0] = 0.0;

        let mut sep_error = [0.0f32; 64];
        let sep_result = crate::entropy_estimate_coeffs(
            &sep_dct,
            &y_dct,
            &weights,
            &inv_weights,
            64,
            cmap_factor,
            quant,
            k_cost_delta,
            0.0,
            true,
            &mut sep_error,
        );

        for i in 0..64 {
            assert!(
                (fused_error[i] - sep_error[i]).abs() < 1e-5,
                "CfL error coeff mismatch at {i}: fused={} sep={}",
                fused_error[i],
                sep_error[i]
            );
        }

        assert!(
            (fused_result.entropy_sum - sep_result.entropy_sum).abs() < 1e-4,
            "CfL entropy mismatch: fused={} sep={}",
            fused_result.entropy_sum,
            sep_result.entropy_sum
        );
    }
}
