// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD-accelerated AC coefficient quantization for DCT8 blocks.
//!
//! The quantize_ac_block inner loop is ~4% of encoder CPU. For DCT8 (the most
//! common strategy), the output is a contiguous [i32; 64] with simple
//! threshold + round per coefficient. This kernel vectorizes 8 coefficients
//! at a time with dead-zone thresholding.

/// Quantize a DCT8 block (64 coefficients) with dead-zone thresholding.
///
/// For each coefficient i (except DC at index 0):
///   val = dct_coeffs[i] / weights[i] * qac_qm
///   if |val| < threshold[quadrant]: output 0
///   else: output round(val) as i32
///
/// DC (index 0) is always set to 0 (handled separately by LLF coding).
///
/// `thresholds` are the 4 quadrant thresholds:
///   [0] = top-left (y<4, x<4)
///   [1] = top-right (y<4, x>=4)
///   [2] = bottom-left (y>=4, x<4)
///   [3] = bottom-right (y>=4, x>=4)
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

    quantize_dct8_scalar(dct_coeffs, weights, qac_qm, thresholds, output);
}

fn quantize_dct8_scalar(
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
            val.round() as i32
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn quantize_dct8_avx2(
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
#[archmage::arcane]
fn quantize_dct8_neon(
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

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;

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

        let mut simd_out = [0i32; 64];
        quantize_block_dct8(&coeffs, &weights, qac_qm, &thresholds, &mut simd_out);

        // DC must be 0
        assert_eq!(simd_out[0], 0, "DC must be 0");
        assert_eq!(ref_out[0], 0, "DC must be 0 (ref)");

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
            "Max quantization diff: {} (at most 1 due to FP rounding boundary)",
            max_diff
        );
        // Allow up to ~5% of coefficients to differ by 1 at rounding boundaries
        assert!(
            diff_count <= 3,
            "Too many differing coefficients: {}/63",
            diff_count
        );
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
        for i in 1..64 {
            assert_eq!(output[i], 100, "Index {} should be 100", i);
        }
    }
}
