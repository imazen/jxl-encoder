// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Forward DCT implementations.
//!
//! These implement the Type-II DCT (DCT-II) used in JPEG and JPEG XL.
//! The transforms are normalized to match the JPEG XL specification.

use std::f32::consts::PI;

/// Computes the 2x2 forward DCT.
pub fn dct2(input: &[f32; 4], output: &mut [f32; 4]) {
    // 2x2 DCT is essentially Hadamard transform with scaling
    let a = input[0] + input[1] + input[2] + input[3];
    let b = input[0] - input[1] + input[2] - input[3];
    let c = input[0] + input[1] - input[2] - input[3];
    let d = input[0] - input[1] - input[2] + input[3];

    output[0] = a * 0.5;
    output[1] = b * 0.5;
    output[2] = c * 0.5;
    output[3] = d * 0.5;
}

/// Computes the 4x4 forward DCT.
pub fn dct4(input: &[f32; 16], output: &mut [f32; 16]) {
    // Temporary buffer for row transforms
    let mut temp = [0.0f32; 16];

    // Transform rows
    for row in 0..4 {
        let row_start = row * 4;
        dct4_1d(
            &input[row_start..row_start + 4],
            &mut temp[row_start..row_start + 4],
        );
    }

    // Transform columns
    for col in 0..4 {
        let col_input: [f32; 4] = [temp[col], temp[col + 4], temp[col + 8], temp[col + 12]];
        let mut col_output = [0.0f32; 4];
        dct4_1d(&col_input, &mut col_output);
        output[col] = col_output[0];
        output[col + 4] = col_output[1];
        output[col + 8] = col_output[2];
        output[col + 12] = col_output[3];
    }
}

/// 1D 4-point DCT.
fn dct4_1d(input: &[f32], output: &mut [f32]) {
    let c1 = (PI / 8.0).cos();
    let c2 = (PI / 4.0).cos();
    let c3 = (3.0 * PI / 8.0).cos();

    let s0 = input[0] + input[3];
    let s1 = input[1] + input[2];
    let s2 = input[0] - input[3];
    let s3 = input[1] - input[2];

    output[0] = 0.5 * (s0 + s1);
    output[2] = 0.5 * c2 * (s0 - s1);
    output[1] = 0.5 * (c1 * s2 + c3 * s3);
    output[3] = 0.5 * (c3 * s2 - c1 * s3);
}

/// Computes the 8x8 forward DCT.
pub fn dct8(input: &[f32; 64], output: &mut [f32; 64]) {
    let mut temp = [0.0f32; 64];

    // Transform rows
    for row in 0..8 {
        let row_start = row * 8;
        dct8_1d(
            &input[row_start..row_start + 8],
            &mut temp[row_start..row_start + 8],
        );
    }

    // Transform columns
    for col in 0..8 {
        let col_input: Vec<f32> = (0..8).map(|row| temp[row * 8 + col]).collect();
        let mut col_output = [0.0f32; 8];
        dct8_1d(&col_input, &mut col_output);
        for row in 0..8 {
            output[row * 8 + col] = col_output[row];
        }
    }
}

/// 1D 8-point DCT using the standard DCT-II formula.
fn dct8_1d(input: &[f32], output: &mut [f32]) {
    let n = 8usize;
    let scale = (2.0 / n as f32).sqrt();

    for (k, out) in output.iter_mut().enumerate().take(n) {
        let mut sum = 0.0f32;
        for (i, &inp) in input.iter().enumerate().take(n) {
            let angle = PI * ((2 * i + 1) * k) as f32 / (2 * n) as f32;
            sum += inp * angle.cos();
        }
        let ck = if k == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
        *out = scale * ck * sum;
    }
}

/// Computes the 16x16 forward DCT.
pub fn dct16(input: &[f32; 256], output: &mut [f32; 256]) {
    let mut temp = [0.0f32; 256];

    // Transform rows
    for row in 0..16 {
        let row_start = row * 16;
        dct_1d_generic(
            &input[row_start..row_start + 16],
            &mut temp[row_start..row_start + 16],
            16,
        );
    }

    // Transform columns
    for col in 0..16 {
        let col_input: Vec<f32> = (0..16).map(|row| temp[row * 16 + col]).collect();
        let mut col_output = vec![0.0f32; 16];
        dct_1d_generic(&col_input, &mut col_output, 16);
        for row in 0..16 {
            output[row * 16 + col] = col_output[row];
        }
    }
}

/// Computes the 32x32 forward DCT.
pub fn dct32(input: &[f32; 1024], output: &mut [f32; 1024]) {
    let mut temp = [0.0f32; 1024];

    // Transform rows
    for row in 0..32 {
        let row_start = row * 32;
        dct_1d_generic(
            &input[row_start..row_start + 32],
            &mut temp[row_start..row_start + 32],
            32,
        );
    }

    // Transform columns
    for col in 0..32 {
        let col_input: Vec<f32> = (0..32).map(|row| temp[row * 32 + col]).collect();
        let mut col_output = vec![0.0f32; 32];
        dct_1d_generic(&col_input, &mut col_output, 32);
        for row in 0..32 {
            output[row * 32 + col] = col_output[row];
        }
    }
}

/// Generic 1D DCT-II implementation.
fn dct_1d_generic(input: &[f32], output: &mut [f32], n: usize) {
    let scale = (2.0 / n as f32).sqrt();

    for (k, out) in output.iter_mut().enumerate().take(n) {
        let mut sum = 0.0f32;
        for (i, &inp) in input.iter().enumerate().take(n) {
            let angle = PI * ((2 * i + 1) * k) as f32 / (2 * n) as f32;
            sum += inp * angle.cos();
        }
        let ck = if k == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
        *out = scale * ck * sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct2_dc() {
        // Constant input should produce DC only
        let input = [1.0, 1.0, 1.0, 1.0];
        let mut output = [0.0f32; 4];
        dct2(&input, &mut output);

        // DC coefficient should be 2.0 (sum of inputs * 0.5)
        assert!((output[0] - 2.0).abs() < 1e-6);
        // AC coefficients should be zero
        assert!(output[1].abs() < 1e-6);
        assert!(output[2].abs() < 1e-6);
        assert!(output[3].abs() < 1e-6);
    }

    #[test]
    fn test_dct8_dc() {
        // Constant input should produce DC only
        let input = [1.0f32; 64];
        let mut output = [0.0f32; 64];
        dct8(&input, &mut output);

        // DC coefficient should be non-zero
        assert!(output[0].abs() > 0.1);
        // High-frequency coefficients should be near zero
        assert!(output[63].abs() < 1e-5);
    }

    #[test]
    fn test_dct8_energy_preservation() {
        // Parseval's theorem: energy should be preserved
        let input: [f32; 64] = std::array::from_fn(|i| (i as f32).sin());
        let mut output = [0.0f32; 64];
        dct8(&input, &mut output);

        let input_energy: f32 = input.iter().map(|x| x * x).sum();
        let output_energy: f32 = output.iter().map(|x| x * x).sum();

        // Energy should be approximately preserved (within numerical precision)
        assert!((input_energy - output_energy).abs() < 0.1);
    }
}
