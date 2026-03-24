// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! SIMD-accelerated AC coefficient quantization.
//!
//! Two kernels:
//! - `quantize_block_dct8`: Fixed 64-element DCT8 path (~4% of encoder CPU)
//! - `quantize_block_large`: Generic path for DCT16+ blocks (128–4096 coefficients)
//!
//! Both use dead-zone thresholding: coefficients below a per-quadrant threshold
//! are zeroed. SIMD processes 8 (AVX2) or 4 (NEON/WASM) coefficients at a time.

/// Quantize a DCT8 block (64 coefficients) with dead-zone thresholding.
///
/// For each coefficient `i` (except DC at index 0):
///   `val = dct_coeffs[i] / weights[i] * qac_qm`
///   if `|val| < threshold[quadrant]`: output 0
///   else: output round(val) as i32
///
/// DC (index 0) is always set to 0 (handled separately by LLF coding).
///
/// `thresholds` are the 4 quadrant thresholds:
///   `[0]` = top-left (y<4, x<4),
///   `[1]` = top-right (y<4, x>=4),
///   `[2]` = bottom-left (y>=4, x<4),
///   `[3]` = bottom-right (y>=4, x>=4)
#[inline]
pub fn quantize_block_dct8(
    dct_coeffs: &[f32; 64],
    weights: &[f32; 64],
    qac_qm: f32,
    thresholds: &[f32; 4],
    output: &mut [i32; 64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            quantize_dct8_avx2(token, dct_coeffs, weights, qac_qm, thresholds, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            quantize_dct8_neon(token, dct_coeffs, weights, qac_qm, thresholds, output);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            quantize_dct8_wasm128(token, dct_coeffs, weights, qac_qm, thresholds, output);
            return;
        }
    }

    quantize_dct8_scalar(dct_coeffs, weights, qac_qm, thresholds, output);
}

#[inline]
pub fn quantize_dct8_scalar(
    dct_coeffs: &[f32; 64],
    weights: &[f32; 64],
    qac_qm: f32,
    thresholds: &[f32; 4],
    output: &mut [i32; 64],
) {
    output[0] = 0; // DC
    for idx in 1..64 {
        let y = idx / 8;
        let x = idx % 8;
        let thr_idx = (if y >= 4 { 2 } else { 0 }) + (if x >= 4 { 1 } else { 0 });
        let val = dct_coeffs[idx] * (1.0 / weights[idx]) * qac_qm;
        output[idx] = if val.abs() < thresholds[thr_idx] {
            0
        } else {
            val.round_ties_even() as i32
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
pub fn quantize_dct8_avx2(
    token: archmage::X64V3Token,
    dct_coeffs: &[f32; 64],
    weights: &[f32; 64],
    qac_qm: f32,
    thresholds: &[f32; 4],
    output: &mut [i32; 64],
) {
    use magetypes::simd::f32x8;

    let qac_qm_v = f32x8::splat(token, qac_qm);
    let zero_f = f32x8::zero(token);

    // Pre-build threshold vectors for each row group:
    // Rows 0-3: [t[0], t[0], t[0], t[0], t[1], t[1], t[1], t[1]]
    // Rows 4-7: [t[2], t[2], t[2], t[2], t[3], t[3], t[3], t[3]]
    let thr_top = f32x8::from_array(
        token,
        [
            thresholds[0],
            thresholds[0],
            thresholds[0],
            thresholds[0],
            thresholds[1],
            thresholds[1],
            thresholds[1],
            thresholds[1],
        ],
    );
    let thr_bot = f32x8::from_array(
        token,
        [
            thresholds[2],
            thresholds[2],
            thresholds[2],
            thresholds[2],
            thresholds[3],
            thresholds[3],
            thresholds[3],
            thresholds[3],
        ],
    );

    // Process 8 chunks of 8 elements (one row each)
    for chunk in 0..8 {
        let base = chunk * 8;
        let coeffs = f32x8::from_slice(token, &dct_coeffs[base..]);
        let w = f32x8::from_slice(token, &weights[base..]);
        let thr = if chunk < 4 { thr_top } else { thr_bot };

        // val = coeffs / weights * qac_qm
        let val = coeffs / w * qac_qm_v;

        // Dead-zone thresholding: if |val| < thr, output 0
        let abs_val = val.abs();
        let mask = abs_val.simd_ge(thr); // all-ones where |val| >= threshold

        // Round and select (0 where below threshold)
        let rounded = val.round();
        let result = f32x8::blend(mask, rounded, zero_f);

        // Convert to i32 (truncate — result is already at integer values)
        let result_i32 = result.to_i32x8();
        result_i32.store((&mut output[base..base + 8]).try_into().unwrap());
    }

    // DC is always 0 (overwrite whatever SIMD produced for index 0)
    output[0] = 0;
}

// --- aarch64 NEON implementation ---

#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
pub fn quantize_dct8_neon(
    token: archmage::NeonToken,
    dct_coeffs: &[f32; 64],
    weights: &[f32; 64],
    qac_qm: f32,
    thresholds: &[f32; 4],
    output: &mut [i32; 64],
) {
    use magetypes::simd::f32x4;

    let qac_qm_v = f32x4::splat(token, qac_qm);
    let zero_f = f32x4::zero(token);

    // With f32x4 (4 elements = half a row), each chunk has a uniform threshold:
    // row 0-3 lo (cols 0-3): thresholds[0]
    // row 0-3 hi (cols 4-7): thresholds[1]
    // row 4-7 lo (cols 0-3): thresholds[2]
    // row 4-7 hi (cols 4-7): thresholds[3]
    let thr = [
        f32x4::splat(token, thresholds[0]),
        f32x4::splat(token, thresholds[1]),
        f32x4::splat(token, thresholds[2]),
        f32x4::splat(token, thresholds[3]),
    ];

    // Process 16 chunks of 4 elements (2 per row, 8 rows)
    for row in 0..8 {
        let thr_row = if row < 4 { 0 } else { 2 };
        for half in 0..2usize {
            let base = row * 8 + half * 4;
            let coeffs = f32x4::from_slice(token, &dct_coeffs[base..]);
            let w = f32x4::from_slice(token, &weights[base..]);
            let t = thr[thr_row + half];

            let val = coeffs / w * qac_qm_v;
            let abs_val = val.abs();
            let mask = abs_val.simd_ge(t);
            let rounded = val.round();
            let result = f32x4::blend(mask, rounded, zero_f);
            let result_i32 = result.to_i32x4();
            result_i32.store((&mut output[base..base + 4]).try_into().unwrap());
        }
    }

    output[0] = 0;
}

// --- wasm32 SIMD128 implementation ---

#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
pub fn quantize_dct8_wasm128(
    token: archmage::Wasm128Token,
    dct_coeffs: &[f32; 64],
    weights: &[f32; 64],
    qac_qm: f32,
    thresholds: &[f32; 4],
    output: &mut [i32; 64],
) {
    use magetypes::simd::f32x4;

    let qac_qm_v = f32x4::splat(token, qac_qm);
    let zero_f = f32x4::zero(token);

    let thr = [
        f32x4::splat(token, thresholds[0]),
        f32x4::splat(token, thresholds[1]),
        f32x4::splat(token, thresholds[2]),
        f32x4::splat(token, thresholds[3]),
    ];

    // Process 16 chunks of 4 elements (2 per row, 8 rows)
    for row in 0..8 {
        let thr_row = if row < 4 { 0 } else { 2 };
        for half in 0..2usize {
            let base = row * 8 + half * 4;
            let coeffs = f32x4::from_slice(token, &dct_coeffs[base..]);
            let w = f32x4::from_slice(token, &weights[base..]);
            let t = thr[thr_row + half];

            let val = coeffs / w * qac_qm_v;
            let abs_val = val.abs();
            let mask = abs_val.simd_ge(t);
            let rounded = val.round();
            let result = f32x4::blend(mask, rounded, zero_f);
            let result_i32 = result.to_i32x4();
            result_i32.store((&mut output[base..base + 4]).try_into().unwrap());
        }
    }

    output[0] = 0;
}

// ============================================================================
// Generic large-block quantization (DCT16+)
// ============================================================================

/// Quantize AC coefficients for a large block (DCT16+) to a flat output buffer.
///
/// For each coefficient at position (y, x) in the grid:
///   val = dct_coeffs[y*grid_width + x] / weights[y*grid_width + x] * qac_qm
///   if y < llf_y && x < llf_x: output 0 (LLF handled separately)
///   elif `|val| < threshold[quadrant]`: output 0
///   else: output round_ties_even(val) as i32
///
/// `grid_width` MUST be a multiple of 8.
/// `thresholds[0..4]` map to quadrants: [top-left, top-right, bottom-left, bottom-right]
/// where the split is at grid_height/2 and grid_width/2.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn quantize_block_large(
    dct_coeffs: &[f32],
    weights: &[f32],
    qac_qm: f32,
    thresholds: &[f32; 4],
    grid_width: usize,
    grid_height: usize,
    llf_x: usize,
    llf_y: usize,
    output: &mut [i32],
) {
    debug_assert_eq!(grid_width % 8, 0, "grid_width must be a multiple of 8");
    let size = grid_width * grid_height;
    debug_assert!(dct_coeffs.len() >= size);
    debug_assert!(weights.len() >= size);
    debug_assert!(output.len() >= size);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            quantize_large_avx2(
                token,
                dct_coeffs,
                weights,
                qac_qm,
                thresholds,
                grid_width,
                grid_height,
                llf_x,
                llf_y,
                output,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            quantize_large_neon(
                token,
                dct_coeffs,
                weights,
                qac_qm,
                thresholds,
                grid_width,
                grid_height,
                llf_x,
                llf_y,
                output,
            );
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            quantize_large_wasm128(
                token,
                dct_coeffs,
                weights,
                qac_qm,
                thresholds,
                grid_width,
                grid_height,
                llf_x,
                llf_y,
                output,
            );
            return;
        }
    }

    quantize_large_scalar(
        dct_coeffs,
        weights,
        qac_qm,
        thresholds,
        grid_width,
        grid_height,
        llf_x,
        llf_y,
        output,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn quantize_large_scalar(
    dct_coeffs: &[f32],
    weights: &[f32],
    qac_qm: f32,
    thresholds: &[f32; 4],
    grid_width: usize,
    grid_height: usize,
    llf_x: usize,
    llf_y: usize,
    output: &mut [i32],
) {
    let half_h = grid_height / 2;
    let half_w = grid_width / 2;
    let size = grid_width * grid_height;

    for idx in 0..size {
        let y = idx / grid_width;
        let x = idx % grid_width;

        // LLF positions are handled separately
        if y < llf_y && x < llf_x {
            output[idx] = 0;
            continue;
        }

        let thr_idx = (if y >= half_h { 2 } else { 0 }) + (if x >= half_w { 1 } else { 0 });
        let val = dct_coeffs[idx] * (1.0 / weights[idx]) * qac_qm;
        output[idx] = if val.abs() < thresholds[thr_idx] {
            0
        } else {
            val.round_ties_even() as i32
        };
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[inline]
#[archmage::arcane]
pub fn quantize_large_avx2(
    token: archmage::X64V3Token,
    dct_coeffs: &[f32],
    weights: &[f32],
    qac_qm: f32,
    thresholds: &[f32; 4],
    grid_width: usize,
    grid_height: usize,
    llf_x: usize,
    llf_y: usize,
    output: &mut [i32],
) {
    use magetypes::simd::f32x8;

    let qac_v = f32x8::splat(token, qac_qm);
    let zero_f = f32x8::zero(token);

    let half_h = grid_height / 2;
    let half_w = grid_width / 2;
    let chunks_per_row = grid_width / 8;

    // Pre-build threshold splats for each quadrant
    let thr_splat = [
        f32x8::splat(token, thresholds[0]),
        f32x8::splat(token, thresholds[1]),
        f32x8::splat(token, thresholds[2]),
        f32x8::splat(token, thresholds[3]),
    ];

    // Pre-slice to help bounds check elimination
    let coeffs = &dct_coeffs[..grid_width * grid_height];
    let wts = &weights[..grid_width * grid_height];
    let out = &mut output[..grid_width * grid_height];

    for y in 0..grid_height {
        let row_thr_base = if y >= half_h { 2 } else { 0 };
        let row_off = y * grid_width;

        for chunk in 0..chunks_per_row {
            let x_base = chunk * 8;
            let base = row_off + x_base;
            let thr_idx = row_thr_base + if x_base >= half_w { 1 } else { 0 };

            let c = crate::load_f32x8(token, coeffs, base);
            let w = crate::load_f32x8(token, wts, base);
            let thr = thr_splat[thr_idx];

            // val = coeff / weight * qac_qm
            let val = c / w * qac_v;

            // Dead-zone thresholding
            let abs_val = val.abs();
            let mask = abs_val.simd_ge(thr);
            let rounded = val.round();
            let result = f32x8::blend(mask, rounded, zero_f);

            let result_i32 = result.to_i32x8();
            result_i32.store((&mut out[base..base + 8]).try_into().unwrap());
        }
    }

    // Zero out LLF positions
    for y in 0..llf_y {
        for x in 0..llf_x {
            out[y * grid_width + x] = 0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "aarch64")]
#[inline]
#[archmage::arcane]
pub fn quantize_large_neon(
    token: archmage::NeonToken,
    dct_coeffs: &[f32],
    weights: &[f32],
    qac_qm: f32,
    thresholds: &[f32; 4],
    grid_width: usize,
    grid_height: usize,
    llf_x: usize,
    llf_y: usize,
    output: &mut [i32],
) {
    use magetypes::simd::f32x4;

    let qac_v = f32x4::splat(token, qac_qm);
    let zero_f = f32x4::zero(token);

    let half_h = grid_height / 2;
    let half_w = grid_width / 2;

    let thr_splat = [
        f32x4::splat(token, thresholds[0]),
        f32x4::splat(token, thresholds[1]),
        f32x4::splat(token, thresholds[2]),
        f32x4::splat(token, thresholds[3]),
    ];

    let coeffs = &dct_coeffs[..grid_width * grid_height];
    let wts = &weights[..grid_width * grid_height];
    let out = &mut output[..grid_width * grid_height];

    for y in 0..grid_height {
        let row_thr_base = if y >= half_h { 2 } else { 0 };
        let row_off = y * grid_width;

        // Process in 4-wide chunks
        let chunks_per_row = grid_width / 4;
        for chunk in 0..chunks_per_row {
            let x_base = chunk * 4;
            let base = row_off + x_base;
            let thr_idx = row_thr_base + if x_base >= half_w { 1 } else { 0 };

            let c = f32x4::from_slice(token, &coeffs[base..]);
            let w = f32x4::from_slice(token, &wts[base..]);
            let thr = thr_splat[thr_idx];

            let val = c / w * qac_v;
            let abs_val = val.abs();
            let mask = abs_val.simd_ge(thr);
            let rounded = val.round();
            let result = f32x4::blend(mask, rounded, zero_f);

            let result_i32 = result.to_i32x4();
            result_i32.store((&mut out[base..base + 4]).try_into().unwrap());
        }
    }

    // Zero out LLF positions
    for y in 0..llf_y {
        for x in 0..llf_x {
            out[y * grid_width + x] = 0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "wasm32")]
#[inline]
#[archmage::arcane]
pub fn quantize_large_wasm128(
    token: archmage::Wasm128Token,
    dct_coeffs: &[f32],
    weights: &[f32],
    qac_qm: f32,
    thresholds: &[f32; 4],
    grid_width: usize,
    grid_height: usize,
    llf_x: usize,
    llf_y: usize,
    output: &mut [i32],
) {
    use magetypes::simd::f32x4;

    let qac_v = f32x4::splat(token, qac_qm);
    let zero_f = f32x4::zero(token);

    let half_h = grid_height / 2;
    let half_w = grid_width / 2;

    let thr_splat = [
        f32x4::splat(token, thresholds[0]),
        f32x4::splat(token, thresholds[1]),
        f32x4::splat(token, thresholds[2]),
        f32x4::splat(token, thresholds[3]),
    ];

    let coeffs = &dct_coeffs[..grid_width * grid_height];
    let wts = &weights[..grid_width * grid_height];
    let out = &mut output[..grid_width * grid_height];

    for y in 0..grid_height {
        let row_thr_base = if y >= half_h { 2 } else { 0 };
        let row_off = y * grid_width;

        let chunks_per_row = grid_width / 4;
        for chunk in 0..chunks_per_row {
            let x_base = chunk * 4;
            let base = row_off + x_base;
            let thr_idx = row_thr_base + if x_base >= half_w { 1 } else { 0 };

            let c = f32x4::from_slice(token, &coeffs[base..]);
            let w = f32x4::from_slice(token, &wts[base..]);
            let thr = thr_splat[thr_idx];

            let val = c / w * qac_v;
            let abs_val = val.abs();
            let mask = abs_val.simd_ge(thr);
            let rounded = val.round();
            let result = f32x4::blend(mask, rounded, zero_f);

            let result_i32 = result.to_i32x4();
            result_i32.store((&mut out[base..base + 4]).try_into().unwrap());
        }
    }

    // Zero out LLF positions
    for y in 0..llf_y {
        for x in 0..llf_x {
            out[y * grid_width + x] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    extern crate std;

    #[test]
    fn test_quantize_dct8_matches_scalar() {
        // Realistic DCT8 coefficients
        let mut coeffs = [0.0f32; 64];
        let mut weights = [0.0f32; 64];
        for i in 0..64 {
            coeffs[i] = ((i as f32) * 1.7 - 50.0) * 0.3;
            weights[i] = 0.01 + (i as f32) * 0.005;
        }

        let thresholds = [0.56f32, 0.62, 0.62, 0.62];
        let qac_qm = 3.5f32;

        let mut ref_out = [0i32; 64];
        quantize_dct8_scalar(&coeffs, &weights, qac_qm, &thresholds, &mut ref_out);

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = [0i32; 64];
                quantize_block_dct8(&coeffs, &weights, qac_qm, &thresholds, &mut simd_out);

                // DC must be 0
                assert_eq!(simd_out[0], 0, "DC must be 0 [{perm}]");
                assert_eq!(ref_out[0], 0, "DC must be 0 (ref) [{perm}]");

                // Compare all AC coefficients — may differ by 1 at rounding boundaries
                let mut max_diff = 0i32;
                let mut diff_count = 0;
                for i in 1..64 {
                    let diff = (simd_out[i] - ref_out[i]).abs();
                    if diff > 0 {
                        diff_count += 1;
                    }
                    max_diff = max_diff.max(diff);
                }
                assert!(
                    max_diff <= 1,
                    "Max quantization diff: {} (at most 1 due to FP rounding boundary) [{perm}]",
                    max_diff
                );
                // Allow up to ~5% of coefficients to differ by 1 at rounding boundaries
                assert!(
                    diff_count <= 3,
                    "Too many differing coefficients: {}/63 [{perm}]",
                    diff_count
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_quantize_dct8_all_zeros() {
        let coeffs = [0.0f32; 64];
        let weights = [1.0f32; 64];
        let thresholds = [0.5f32; 4];
        let mut output = [99i32; 64]; // fill with non-zero to verify

        quantize_block_dct8(&coeffs, &weights, 1.0, &thresholds, &mut output);

        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 0, "Index {} should be 0", i);
        }
    }

    #[test]
    fn test_quantize_dct8_large_coeffs() {
        // Large coefficients should all survive thresholding
        let mut coeffs = [100.0f32; 64];
        coeffs[0] = 0.0; // DC doesn't matter
        let weights = [1.0f32; 64];
        let thresholds = [0.5f32; 4];

        let mut output = [0i32; 64];
        quantize_block_dct8(&coeffs, &weights, 1.0, &thresholds, &mut output);

        assert_eq!(output[0], 0, "DC must be 0");
        for (i, &val) in output.iter().enumerate().skip(1) {
            assert_eq!(val, 100, "Index {} should be 100", i);
        }
    }

    // =====================================================================
    // Large-block quantize tests
    // =====================================================================

    #[test]
    fn test_quantize_large_dct16x16_matches_scalar() {
        let grid_w = 16;
        let grid_h = 16;
        let size = grid_w * grid_h;
        let llf_x = 2; // cx for DCT16x16
        let llf_y = 2; // cy for DCT16x16

        let mut coeffs = alloc::vec![0.0f32; size];
        let mut weights = alloc::vec![0.0f32; size];
        for i in 0..size {
            coeffs[i] = ((i as f32) * 0.37 - 40.0) * 0.5;
            weights[i] = 0.01 + (i as f32) * 0.002;
        }

        let thresholds = [0.56f32, 0.62, 0.62, 0.62];
        let qac_qm = 4.2f32;

        let mut ref_out = alloc::vec![0i32; size];
        quantize_large_scalar(
            &coeffs,
            &weights,
            qac_qm,
            &thresholds,
            grid_w,
            grid_h,
            llf_x,
            llf_y,
            &mut ref_out,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = alloc::vec![0i32; size];
                quantize_block_large(
                    &coeffs,
                    &weights,
                    qac_qm,
                    &thresholds,
                    grid_w,
                    grid_h,
                    llf_x,
                    llf_y,
                    &mut simd_out,
                );

                // LLF positions must be 0
                for y in 0..llf_y {
                    for x in 0..llf_x {
                        assert_eq!(
                            simd_out[y * grid_w + x],
                            0,
                            "LLF ({},{}) must be 0 [{perm}]",
                            y,
                            x
                        );
                    }
                }

                let mut max_diff = 0i32;
                let mut diff_count = 0;
                for i in 0..size {
                    let diff = (simd_out[i] - ref_out[i]).abs();
                    if diff > 0 {
                        diff_count += 1;
                    }
                    max_diff = max_diff.max(diff);
                }
                assert!(
                    max_diff <= 1,
                    "Max diff: {} (at most 1 due to FP rounding) [{perm}]",
                    max_diff
                );
                let tolerance = size / 20; // 5%
                assert!(
                    diff_count <= tolerance,
                    "Too many diffs: {}/{} [{perm}]",
                    diff_count,
                    size
                );
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_quantize_large_dct32x32_matches_scalar() {
        let grid_w = 32;
        let grid_h = 32;
        let size = grid_w * grid_h;
        let llf_x = 4;
        let llf_y = 4;

        let mut coeffs = alloc::vec![0.0f32; size];
        let mut weights = alloc::vec![0.0f32; size];
        for i in 0..size {
            coeffs[i] = ((i as f32) * 0.19 - 80.0) * 0.3;
            weights[i] = 0.005 + (i as f32) * 0.001;
        }

        let thresholds = [0.54f32, 0.60, 0.58, 0.62];
        let qac_qm = 5.0f32;

        let mut ref_out = alloc::vec![0i32; size];
        quantize_large_scalar(
            &coeffs,
            &weights,
            qac_qm,
            &thresholds,
            grid_w,
            grid_h,
            llf_x,
            llf_y,
            &mut ref_out,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = alloc::vec![0i32; size];
                quantize_block_large(
                    &coeffs,
                    &weights,
                    qac_qm,
                    &thresholds,
                    grid_w,
                    grid_h,
                    llf_x,
                    llf_y,
                    &mut simd_out,
                );

                for y in 0..llf_y {
                    for x in 0..llf_x {
                        assert_eq!(simd_out[y * grid_w + x], 0, "LLF ({},{}) [{perm}]", y, x);
                    }
                }

                let mut max_diff = 0i32;
                for i in 0..size {
                    let diff = (simd_out[i] - ref_out[i]).abs();
                    max_diff = max_diff.max(diff);
                }
                assert!(max_diff <= 1, "Max diff: {} [{perm}]", max_diff);
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_quantize_large_dct64x64_matches_scalar() {
        let grid_w = 64;
        let grid_h = 64;
        let size = grid_w * grid_h;
        let llf_x = 8;
        let llf_y = 8;

        let mut coeffs = alloc::vec![0.0f32; size];
        let mut weights = alloc::vec![0.0f32; size];
        for i in 0..size {
            coeffs[i] = ((i as f32) * 0.07 - 120.0) * 0.2;
            weights[i] = 0.002 + (i as f32) * 0.0005;
        }

        let thresholds = [0.56f32, 0.62, 0.62, 0.62];
        let qac_qm = 3.0f32;

        let mut ref_out = alloc::vec![0i32; size];
        quantize_large_scalar(
            &coeffs,
            &weights,
            qac_qm,
            &thresholds,
            grid_w,
            grid_h,
            llf_x,
            llf_y,
            &mut ref_out,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = alloc::vec![0i32; size];
                quantize_block_large(
                    &coeffs,
                    &weights,
                    qac_qm,
                    &thresholds,
                    grid_w,
                    grid_h,
                    llf_x,
                    llf_y,
                    &mut simd_out,
                );

                for y in 0..llf_y {
                    for x in 0..llf_x {
                        assert_eq!(simd_out[y * grid_w + x], 0, "LLF ({},{}) [{perm}]", y, x);
                    }
                }

                let mut max_diff = 0i32;
                for i in 0..size {
                    let diff = (simd_out[i] - ref_out[i]).abs();
                    max_diff = max_diff.max(diff);
                }
                assert!(max_diff <= 1, "Max diff: {} [{perm}]", max_diff);
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_quantize_large_nonsquare_16x8() {
        let grid_w = 16;
        let grid_h = 8;
        let size = grid_w * grid_h;
        let llf_x = 2;
        let llf_y = 1;

        let mut coeffs = alloc::vec![0.0f32; size];
        let mut weights = alloc::vec![0.0f32; size];
        for i in 0..size {
            coeffs[i] = ((i as f32) * 0.53 - 30.0) * 0.8;
            weights[i] = 0.02 + (i as f32) * 0.004;
        }

        let thresholds = [0.56f32, 0.62, 0.62, 0.62];
        let qac_qm = 2.5f32;

        let mut ref_out = alloc::vec![0i32; size];
        quantize_large_scalar(
            &coeffs,
            &weights,
            qac_qm,
            &thresholds,
            grid_w,
            grid_h,
            llf_x,
            llf_y,
            &mut ref_out,
        );

        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |perm| {
                let mut simd_out = alloc::vec![0i32; size];
                quantize_block_large(
                    &coeffs,
                    &weights,
                    qac_qm,
                    &thresholds,
                    grid_w,
                    grid_h,
                    llf_x,
                    llf_y,
                    &mut simd_out,
                );

                let mut max_diff = 0i32;
                for i in 0..size {
                    let diff = (simd_out[i] - ref_out[i]).abs();
                    max_diff = max_diff.max(diff);
                }
                assert!(max_diff <= 1, "Max diff: {} [{perm}]", max_diff);
            },
        );
        std::eprintln!("{report}");
    }

    #[test]
    fn test_quantize_large_all_zeros() {
        let grid_w = 16;
        let grid_h = 16;
        let size = grid_w * grid_h;

        let coeffs = alloc::vec![0.0f32; size];
        let weights = alloc::vec![1.0f32; size];
        let thresholds = [0.5f32; 4];
        let mut output = alloc::vec![99i32; size];

        quantize_block_large(
            &coeffs,
            &weights,
            1.0,
            &thresholds,
            grid_w,
            grid_h,
            2,
            2,
            &mut output,
        );

        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 0, "Index {} should be 0", i);
        }
    }

    #[test]
    fn test_quantize_large_llf_zeroed() {
        // Verify LLF positions are zeroed even with large coefficients
        let grid_w = 32;
        let grid_h = 32;
        let size = grid_w * grid_h;
        let llf_x = 4;
        let llf_y = 4;

        let coeffs = alloc::vec![100.0f32; size];
        let weights = alloc::vec![1.0f32; size];
        let thresholds = [0.1f32; 4]; // low threshold so everything survives
        let mut output = alloc::vec![0i32; size];

        quantize_block_large(
            &coeffs,
            &weights,
            1.0,
            &thresholds,
            grid_w,
            grid_h,
            llf_x,
            llf_y,
            &mut output,
        );

        for y in 0..llf_y {
            for x in 0..llf_x {
                assert_eq!(output[y * grid_w + x], 0, "LLF ({},{}) must be 0", y, x);
            }
        }
        // Non-LLF should be 100
        assert_eq!(output[llf_x], 100, "First non-LLF position should be 100");
        assert_eq!(
            output[llf_y * grid_w],
            100,
            "First non-LLF row should be 100"
        );
    }
}
