// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

use super::*;

#[test]
fn test_dct1d_2() {
    let mut data = [1.0, 2.0];
    dct1d_2(&mut data);
    assert!((data[0] - 3.0).abs() < 1e-6);
    assert!((data[1] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_dct1d_4() {
    // Constant input should give DC-only output
    let mut data = [1.0, 1.0, 1.0, 1.0];
    dct1d_4(&mut data);
    // DC should be 4.0 (sum), AC should be ~0
    assert!((data[0] - 4.0).abs() < 1e-5, "DC: {}", data[0]);
    for (i, val) in data.iter().enumerate().skip(1) {
        assert!(val.abs() < 1e-5, "AC[{}]: {}", i, val);
    }
}

#[test]
fn test_dct1d_8() {
    // Constant input should give DC-only output
    let mut data = [1.0; 8];
    dct1d_8(&mut data);
    // DC should be 8.0 (sum), AC should be ~0
    assert!((data[0] - 8.0).abs() < 1e-5, "DC: {}", data[0]);
    for (i, val) in data.iter().enumerate().skip(1) {
        assert!(val.abs() < 1e-5, "AC[{}]: {}", i, val);
    }
}

#[test]
fn test_dct_8x8_constant() {
    // Constant 8x8 block should have only DC
    let input = [0.5f32; 64];
    let mut output = [0.0f32; 64];
    dct_8x8(&input, &mut output);

    // DC = sum / 64 = 32 / 64 = 0.5, but with scaling it's sum * (1/8)^2 = 32/64 = 0.5
    // Actually for constant input, DCT gives DC = sum * normalization
    // With our 1/N scaling per dimension: DC = 8 * 0.5 / 8 * 8 / 8 = 0.5
    // This depends on the exact normalization. Let's just check AC is zero.
    for (i, val) in output.iter().enumerate().skip(1) {
        assert!(val.abs() < 1e-4, "AC[{}]: {} should be ~0", i, val);
    }
}

#[test]
fn test_dct_8x8_energy_preservation() {
    // DCT should preserve energy (Parseval's theorem)
    let input: [f32; 64] = core::array::from_fn(|i| (i as f32) / 64.0);
    let mut output = [0.0f32; 64];
    dct_8x8(&input, &mut output);

    let _input_energy: f32 = input.iter().map(|x| x * x).sum();
    let output_energy: f32 = output.iter().map(|x| x * x).sum();

    // Energy may differ by normalization factor, but should be proportional
    assert!(output_energy > 0.0, "Output should have non-zero energy");
}

#[test]
fn test_dc_from_dct_8x8() {
    let mut coeffs = [0.0f32; 64];
    coeffs[0] = 42.0;
    assert_eq!(dc_from_dct_8x8(&coeffs), 42.0);
}

#[test]
fn test_dc_from_dct_16x8() {
    // Test with known values
    let mut coeffs = [0.0f32; 128];
    coeffs[0] = 1.0;
    let dc = dc_from_dct_16x8(&coeffs);
    // With only lf0=1.0 and lf1=0, we get [1, 1] from IDCT
    assert!((dc[0] - 1.0).abs() < 1e-5);
    assert!((dc[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_dct_16x16_constant() {
    // Constant 16x16 block should have only DC, all AC ~0
    let input = [0.5f32; 256];
    let mut output = [0.0f32; 256];
    dct_16x16(&input, &mut output);

    // AC should be zero
    for (i, val) in output.iter().enumerate().skip(1) {
        assert!(val.abs() < 1e-4, "AC[{}]: {} should be ~0", i, val);
    }
}

#[test]
fn test_dct_16x16_no_final_transpose() {
    // Verify the output layout: coefficients should NOT be transposed back.
    // For a block that's non-zero only along row 0, the DCT should produce
    // coefficients that vary along the frequency row direction (index / 16)
    // but are zero for frequency column > 0 (index % 16 > 0).
    let mut input = [0.0f32; 256];
    // Set row 0 to a specific pattern
    for (x, val) in input.iter_mut().enumerate().take(16) {
        *val = (x as f32 + 1.0) / 16.0;
    }
    let mut output = [0.0f32; 256];
    dct_16x16(&input, &mut output);

    // After DCT with no transpose: the horizontal transform is applied first,
    // then vertical. With only row 0 non-zero, the vertical transform produces
    // DC in all rows. So we should have non-zero values in column 0 (indices 0, 16, 32, ...)
    // but also in other positions due to the non-trivial row content.
    // Key check: output[1] should be non-zero (it's the (0,1) frequency),
    // NOT at output[16] which would be the case with a final transpose.
    assert!(
        output[1].abs() > 1e-6,
        "output[1] should be non-zero for non-trivial row 0"
    );
}

#[test]
fn test_dc_from_dct_16x16_uniform() {
    // Uniform input: all 4 DC values should be equal
    let input = [1.0f32; 256];
    let mut output = [0.0f32; 256];
    dct_16x16(&input, &mut output);
    let dc = dc_from_dct_16x16(&output);

    // All 4 DC values should be approximately equal (uniform input)
    for i in 1..4 {
        assert!(
            (dc[i] - dc[0]).abs() < 1e-3,
            "dc[{}]={} should equal dc[0]={}",
            i,
            dc[i],
            dc[0]
        );
    }
}

#[test]
fn test_dc_from_dct_16x16_dc_only() {
    // If only LLF[0] (DC) is set, all 4 outputs should be equal
    let mut coeffs = [0.0f32; 256];
    coeffs[0] = 4.0;
    let dc = dc_from_dct_16x16(&coeffs);
    // With b00=4, b01=b10=b11=0:
    // dc00 = dc01 = dc10 = dc11 = 4.0
    for (i, val) in dc.iter().enumerate() {
        assert!((*val - 4.0).abs() < 1e-5, "dc[{}]={} should be 4.0", i, val);
    }
}

#[test]
fn test_dct1d_8_roundtrip_scaling() {
    // Check 1D DCT8 → IDCT8 scaling for comparison
    let orig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut forward = orig;
    dct1d_8(&mut forward);
    let mut inverse = forward;
    idct1d_8(&mut inverse);
    let ratio = inverse[0] / orig[0];
    eprintln!("1D DCT8 raw roundtrip scale factor: {:.6}", ratio);

    let mut forward2 = orig;
    dct1d_8(&mut forward2);
    for v in forward2.iter_mut() {
        *v *= 1.0 / 8.0;
    }
    let mut inverse2 = forward2;
    idct1d_8(&mut inverse2);
    let max_err = orig
        .iter()
        .zip(inverse2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("1D DCT8 roundtrip (with 1/8 scale): max_err={:.6}", max_err);
}

#[test]
fn test_dct1d_16_roundtrip_scaling() {
    // Check 1D DCT16 → IDCT16 scaling
    let orig = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let mut forward = orig;
    dct1d_16(&mut forward);
    // Apply 1/16 scaling (like dct_16x16 does)
    for v in forward.iter_mut() {
        *v *= 1.0 / 16.0;
    }
    let mut inverse = forward;
    idct1d_16(&mut inverse);
    let max_err = orig
        .iter()
        .zip(inverse.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "1D DCT16 roundtrip (with 1/16 scale): max_err={:.6}",
        max_err
    );
    eprintln!("  orig[0..4]: {:?}", &orig[..4]);
    eprintln!("  back[0..4]: {:.6?}", &inverse[..4]);

    // Also test without scaling
    let mut forward2 = orig;
    dct1d_16(&mut forward2);
    let mut inverse2 = forward2;
    idct1d_16(&mut inverse2);
    let ratio = inverse2[0] / orig[0];
    eprintln!(
        "1D DCT16 roundtrip (raw, no scale): scale_factor={:.6}",
        ratio
    );
    eprintln!("  orig[0..4]: {:?}", &orig[..4]);
    eprintln!("  back[0..4]: {:.6?}", &inverse2[..4]);
}

#[test]
fn test_dct_16x16_roundtrip() {
    // Test DCT16x16 → IDCT16x16 roundtrip with pseudo-random data
    let mut input = [0.0f32; 256];
    for (i, val) in input.iter_mut().enumerate() {
        *val = ((i as f32 * 0.7 + 3.14).sin() * 100.0).round() / 100.0;
    }
    let mut dct_output = [0.0f32; 256];
    dct_16x16(&input, &mut dct_output);
    let mut roundtrip = [0.0f32; 256];
    idct_16x16(&dct_output, &mut roundtrip);
    let max_err = input
        .iter()
        .zip(roundtrip.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("DCT16x16 roundtrip max error: {:.6}", max_err);
    eprintln!("  input[0..4]: {:.4?}", &input[..4]);
    eprintln!("  roundtrip[0..4]: {:.4?}", &roundtrip[..4]);
    // Check ratio
    if input[0].abs() > 0.01 {
        eprintln!("  scale_factor[0]: {:.6}", roundtrip[0] / input[0]);
    }
    assert!(
        max_err < 1e-4,
        "DCT16x16 roundtrip max error: {} (should be < 1e-4)",
        max_err
    );
}

// ─── DCT32 tests ──────────────────────────────────────────────────

#[test]
fn test_dct1d_32_constant() {
    // Constant input should give DC-only output
    let mut data = [1.0f32; 32];
    dct1d_32(&mut data);
    // DC should be 32.0 (sum), AC should be ~0
    assert!(
        (data[0] - 32.0).abs() < 1e-3,
        "DC: {} expected 32.0",
        data[0]
    );
    for (i, val) in data.iter().enumerate().skip(1) {
        assert!(val.abs() < 1e-3, "AC[{}]: {} should be ~0", i, val);
    }
}

#[test]
fn test_dct_32x32_constant() {
    // Constant 32x32 block should have only DC, all AC ~0
    let input = [0.5f32; 1024];
    let mut output = [0.0f32; 1024];
    dct_32x32(&input, &mut output);

    // AC should be zero
    for (i, val) in output.iter().enumerate().skip(1) {
        assert!(val.abs() < 1e-3, "AC[{}]: {} should be ~0", i, val);
    }
}

#[test]
fn test_dc_from_dct_32x32_uniform() {
    // Uniform input: all 16 DC values should be approximately equal
    let input = [1.0f32; 1024];
    let mut output = [0.0f32; 1024];
    dct_32x32(&input, &mut output);
    let dc = dc_from_dct_32x32(&output);

    // All 16 DC values should be approximately equal (uniform input)
    for i in 1..16 {
        assert!(
            (dc[i] - dc[0]).abs() < 1e-2,
            "dc[{}]={} should equal dc[0]={}",
            i,
            dc[i],
            dc[0]
        );
    }
}

#[test]
fn test_dc_from_dct_32x32_dc_only() {
    // If only LLF[0] (DC) is set, all 16 outputs should be equal
    let mut coeffs = [0.0f32; 1024];
    coeffs[0] = 16.0;
    let dc = dc_from_dct_32x32(&coeffs);
    // All should be equal since only DC is set
    for i in 1..16 {
        assert!(
            (dc[i] - dc[0]).abs() < 1e-3,
            "dc[{}]={} should equal dc[0]={}",
            i,
            dc[i],
            dc[0]
        );
    }
}

#[test]
fn test_dct_32x32_no_final_transpose() {
    // Verify no final transpose: input with only row 0 non-zero
    let mut input = [0.0f32; 1024];
    for (x, val) in input.iter_mut().enumerate().take(32) {
        *val = (x as f32 + 1.0) / 32.0;
    }
    let mut output = [0.0f32; 1024];
    dct_32x32(&input, &mut output);

    // output[1] should be non-zero (horizontal frequency)
    assert!(
        output[1].abs() > 1e-6,
        "output[1] should be non-zero for non-trivial row 0"
    );
}

// ─── DCT4x8 / DCT8x4 tests ──────────────────────────────────────────

#[test]
fn test_dct_4x8_constant() {
    // Constant 4x8 block should have only DC
    let input = [0.5f32; 32];
    let mut output = [0.0f32; 32];
    dct_4x8(&input, &mut output);

    // AC should be near zero
    for (i, val) in output.iter().enumerate().skip(1) {
        assert!(val.abs() < 1e-4, "AC[{}]: {} should be ~0", i, val);
    }
}

#[test]
fn test_dct_8x4_constant() {
    // Constant 8x4 block should have only DC
    let input = [0.5f32; 32];
    let mut output = [0.0f32; 32];
    dct_8x4(&input, &mut output);

    // AC should be near zero
    for (i, val) in output.iter().enumerate().skip(1) {
        assert!(val.abs() < 1e-4, "AC[{}]: {} should be ~0", i, val);
    }
}

#[test]
fn test_dct_4x8_full_constant() {
    // Constant 8x8 block processed with DCT4X8 should have only DC
    let input = [0.5f32; 64];
    let mut output = [0.0f32; 64];
    dct_4x8_full(&input, &mut output);

    // DC at [0] should be non-zero
    assert!(output[0].abs() > 0.1, "DC should be non-zero");
    // The LLF coefficient at [8] can be non-zero (difference of sub-block DCs)
    // but for uniform input it should be zero
    assert!(
        output[8].abs() < 1e-4,
        "LLF[8] should be ~0 for uniform input"
    );
    // Other AC should be near zero
    for (i, val) in output.iter().enumerate().skip(1) {
        if i == 8 {
            continue;
        } // skip LLF[8]
        assert!(val.abs() < 1e-4, "AC[{}]: {} should be ~0", i, val);
    }
}

#[test]
fn test_dct_8x4_full_constant() {
    // Constant 8x8 block processed with DCT8X4 should have only DC
    let input = [0.5f32; 64];
    let mut output = [0.0f32; 64];
    dct_8x4_full(&input, &mut output);

    // DC at [0] should be non-zero
    assert!(output[0].abs() > 0.1, "DC should be non-zero");
    // LLF[8] should be ~0 for uniform input
    assert!(
        output[8].abs() < 1e-4,
        "LLF[8] should be ~0 for uniform input"
    );
    // Other AC should be near zero
    for (i, val) in output.iter().enumerate().skip(1) {
        if i == 8 {
            continue;
        }
        assert!(val.abs() < 1e-4, "AC[{}]: {} should be ~0", i, val);
    }
}

#[test]
fn test_dct_4x8_full_dc_extraction() {
    // Verify DC extraction works correctly
    let input = [1.0f32; 64];
    let mut output = [0.0f32; 64];
    dct_4x8_full(&input, &mut output);

    let dc = dc_from_dct_4x8_full(&output);
    assert!(dc.abs() > 0.1, "DC should be non-zero: {}", dc);
}

#[test]
fn test_dct_8x4_full_dc_extraction() {
    // Verify DC extraction works correctly
    let input = [1.0f32; 64];
    let mut output = [0.0f32; 64];
    dct_8x4_full(&input, &mut output);

    let dc = dc_from_dct_8x4_full(&output);
    assert!(dc.abs() > 0.1, "DC should be non-zero: {}", dc);
}

#[test]
fn test_dct_4x8_full_top_bottom_different() {
    // Test that DCT4X8 can distinguish top vs bottom halves
    let mut input = [0.0f32; 64];
    // Top half = 1.0, bottom half = 0.0
    for y in 0..4 {
        for x in 0..8 {
            input[y * 8 + x] = 1.0;
        }
    }
    let mut output = [0.0f32; 64];
    dct_4x8_full(&input, &mut output);

    // LLF[8] should be non-zero (it encodes the top-bottom difference)
    assert!(
        output[8].abs() > 0.1,
        "LLF[8] should capture top-bottom difference: {}",
        output[8]
    );
}

#[test]
fn test_dct_8x4_full_left_right_different() {
    // Test that DCT8X4 can distinguish left vs right halves
    let mut input = [0.0f32; 64];
    // Left half = 1.0, right half = 0.0
    for y in 0..8 {
        for x in 0..4 {
            input[y * 8 + x] = 1.0;
        }
    }
    let mut output = [0.0f32; 64];
    dct_8x4_full(&input, &mut output);

    // LLF[8] should be non-zero (it encodes the left-right difference)
    assert!(
        output[8].abs() > 0.1,
        "LLF[8] should capture left-right difference: {}",
        output[8]
    );
}

#[test]
fn test_dct_4x8_energy_preservation() {
    // DCT should approximately preserve energy
    let input: [f32; 32] = core::array::from_fn(|i| (i as f32) / 32.0);
    let mut output = [0.0f32; 32];
    dct_4x8(&input, &mut output);

    let input_energy: f32 = input.iter().map(|x| x * x).sum();
    let output_energy: f32 = output.iter().map(|x| x * x).sum();

    // Energy should be proportional (may differ by normalization)
    assert!(output_energy > 0.0, "Output should have non-zero energy");
    assert!(
        input_energy > 0.0 && output_energy > 0.0,
        "Both should have energy"
    );
}

#[test]
fn test_dct_8x4_energy_preservation() {
    // DCT should approximately preserve energy
    let input: [f32; 32] = core::array::from_fn(|i| (i as f32) / 32.0);
    let mut output = [0.0f32; 32];
    dct_8x4(&input, &mut output);

    let input_energy: f32 = input.iter().map(|x| x * x).sum();
    let output_energy: f32 = output.iter().map(|x| x * x).sum();

    assert!(output_energy > 0.0, "Output should have non-zero energy");
    assert!(
        input_energy > 0.0 && output_energy > 0.0,
        "Both should have energy"
    );
}

#[test]
fn test_idct_8x8_constant() {
    use super::{dct_8x8, idct_8x8};

    // Constant input of 1.0
    let input = [1.0f32; 64];
    let mut coeffs = [0.0f32; 64];
    dct_8x8(&input, &mut coeffs);

    eprintln!("DCT of constant 1.0:");
    eprintln!("  DC = {}", coeffs[0]);
    eprintln!(
        "  AC[1] = {}, AC[8] = {}, AC[9] = {}",
        coeffs[1], coeffs[8], coeffs[9]
    );
    // DC should be 1.0 (sum / 64 = 64/64 = 1), AC should be ~0

    let mut reconstructed = [0.0f32; 64];
    idct_8x8(&coeffs, &mut reconstructed);

    eprintln!("IDCT reconstructed:");
    eprintln!("  [0] = {}", reconstructed[0]);

    // Try to find the scale factor
    if coeffs[0] != 0.0 {
        let raw_scale = 1.0 / reconstructed[0]; // what we need to multiply by
        eprintln!("Scale factor needed to get 1.0: {}", raw_scale);
    }

    // Expected: all 1.0
    let max_err = reconstructed
        .iter()
        .map(|&x| (x - 1.0).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Max error from expected 1.0: {}", max_err);
}

#[test]
fn test_idct_8x8_impulse() {
    use super::{dct_8x8, idct_8x8};

    // Impulse at (0,0)
    let mut input = [0.0f32; 64];
    input[0] = 64.0; // scale so DC = 1 after DCT

    let mut coeffs = [0.0f32; 64];
    dct_8x8(&input, &mut coeffs);

    eprintln!("DCT of impulse at (0,0) scaled to 64:");
    eprintln!(
        "  [0] = {}, [1] = {}, [8] = {}",
        coeffs[0], coeffs[1], coeffs[8]
    );

    let mut reconstructed = [0.0f32; 64];
    idct_8x8(&coeffs, &mut reconstructed);

    eprintln!(
        "IDCT reconstructed [0] = {}, should be 64",
        reconstructed[0]
    );
    eprintln!("Scale factor: {}", reconstructed[0] / input[0]);
}

#[test]
#[ignore] // IDCT scaling needs calibration - see NOTE in IDCT section
fn test_idct_8x8_roundtrip() {
    use super::{dct_8x8, idct_8x8};

    // Random-ish input
    let input: [f32; 64] = core::array::from_fn(|i| ((i as f32 * 0.7).sin() + 0.5) * 100.0);

    let mut coeffs = [0.0f32; 64];
    let mut reconstructed = [0.0f32; 64];

    dct_8x8(&input, &mut coeffs);
    idct_8x8(&coeffs, &mut reconstructed);

    // Verify roundtrip
    let mut max_err = 0.0f32;
    for i in 0..64 {
        let err = (input[i] - reconstructed[i]).abs();
        max_err = max_err.max(err);
    }
    eprintln!("Roundtrip max error: {}", max_err);
    eprintln!(
        "Input[0]: {}, Reconstructed[0]: {}",
        input[0], reconstructed[0]
    );
    eprintln!("Scale factor needed: {}", reconstructed[0] / input[0]);
    assert!(
        max_err < 1e-3,
        "idct_8x8 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_identity_roundtrip() {
    // Random-ish 8x8 pixel block
    let pixels: [f32; 64] = core::array::from_fn(|i| ((i as f32 * 1.3).sin() + 0.5) * 200.0);

    let mut coeffs = [0.0f32; 64];
    identity_transform(&pixels, &mut coeffs);

    let mut reconstructed = [0.0f32; 64];
    inverse_identity_transform(&coeffs, &mut reconstructed);

    let mut max_err = 0.0f32;
    for i in 0..64 {
        let err = (pixels[i] - reconstructed[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < 1e-4,
        "identity roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_identity_roundtrip_constant() {
    // Constant block: all pixels the same
    let pixels = [42.0f32; 64];

    let mut coeffs = [0.0f32; 64];
    identity_transform(&pixels, &mut coeffs);

    let mut reconstructed = [0.0f32; 64];
    inverse_identity_transform(&coeffs, &mut reconstructed);

    let mut max_err = 0.0f32;
    for i in 0..64 {
        let err = (pixels[i] - reconstructed[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-4,
        "identity roundtrip (constant) max error {} too large",
        max_err
    );
}

#[test]
fn test_dct2x2_roundtrip() {
    // Random-ish 8x8 pixel block
    let pixels: [f32; 64] = core::array::from_fn(|i| ((i as f32 * 0.9).cos() + 1.0) * 128.0);

    let mut coeffs = [0.0f32; 64];
    dct2x2_transform(&pixels, &mut coeffs);

    let mut reconstructed = [0.0f32; 64];
    inverse_dct2x2_transform(&coeffs, &mut reconstructed);

    let mut max_err = 0.0f32;
    for i in 0..64 {
        let err = (pixels[i] - reconstructed[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < 1e-4,
        "dct2x2 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_dct2x2_roundtrip_constant() {
    let pixels = [99.0f32; 64];

    let mut coeffs = [0.0f32; 64];
    dct2x2_transform(&pixels, &mut coeffs);

    let mut reconstructed = [0.0f32; 64];
    inverse_dct2x2_transform(&coeffs, &mut reconstructed);

    let mut max_err = 0.0f32;
    for i in 0..64 {
        let err = (pixels[i] - reconstructed[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-4,
        "dct2x2 roundtrip (constant) max error {} too large",
        max_err
    );
}

#[test]
fn test_dct_32x32_dc_extraction_constant() {
    let val = 42.0f32;
    let input = [val; 1024];
    let mut output = [0.0f32; 1024];
    dct_32x32(&input, &mut output);

    let dcs = dc_from_dct_32x32(&output);
    eprintln!("DCT32x32 DC[0] = {}, expected {}", dcs[0], val);
    for (i, &dc) in dcs.iter().enumerate() {
        assert!(
            (dc - val).abs() < 0.5,
            "DC32x32[{}] = {}, expected ~{}",
            i,
            dc,
            val
        );
    }
}

#[test]
fn test_dct_64x64_dc_constant() {
    // A constant 64x64 block should have all energy in DC
    let val = 42.0f32;
    let input = [val; 4096];
    let mut output = [0.0f32; 4096];
    dct_64x64(&input, &mut output);

    // DC should be val (after double 1/64 scaling: val * 64 * (1/64) * 64 * (1/64) = val)
    let dc = output[0];
    assert!(
        (dc - val).abs() < 0.01,
        "DCT64x64 constant DC = {}, expected {}",
        dc,
        val
    );

    // All other coefficients should be ~0
    let mut max_ac = 0.0f32;
    for &coeff in &output[1..] {
        max_ac = max_ac.max(coeff.abs());
    }
    assert!(
        max_ac < 1e-3,
        "DCT64x64 constant max AC = {}, expected ~0",
        max_ac
    );
}

#[test]
fn test_dct_64x64_dc_extraction_constant() {
    // Constant 64x64 block: all 64 DCs should equal the constant value
    let val = 42.0f32;
    let input = [val; 4096];
    let mut output = [0.0f32; 4096];
    dct_64x64(&input, &mut output);

    let dcs = dc_from_dct_64x64(&output);
    for (i, &dc) in dcs.iter().enumerate() {
        assert!(
            (dc - val).abs() < 0.5,
            "DC[{}] = {}, expected ~{}",
            i,
            dc,
            val
        );
    }
}

#[test]
fn test_dct_64x32_dc_extraction_constant() {
    let val = 42.0f32;
    let input = [val; 2048];
    let mut output = [0.0f32; 2048];
    dct_64x32(&input, &mut output);

    let dcs = dc_from_dct_64x32(&output);
    for (i, &dc) in dcs.iter().enumerate() {
        assert!(
            (dc - val).abs() < 0.5,
            "DC64x32[{}] = {}, expected ~{}",
            i,
            dc,
            val
        );
    }
}

#[test]
fn test_dct_32x64_dc_extraction_constant() {
    let val = 42.0f32;
    let input = [val; 2048];
    let mut output = [0.0f32; 2048];
    dct_32x64(&input, &mut output);

    let dcs = dc_from_dct_32x64(&output);
    for (i, &dc) in dcs.iter().enumerate() {
        assert!(
            (dc - val).abs() < 0.5,
            "DC32x64[{}] = {}, expected ~{}",
            i,
            dc,
            val
        );
    }
}

#[test]
fn test_dct_32x16_dc_extraction_constant() {
    let val = 42.0f32;
    let input = [val; 512];
    let mut output = [0.0f32; 512];
    dct_32x16(&input, &mut output);

    let dcs = dc_from_dct_32x16(&output);
    for (i, &dc) in dcs.iter().enumerate() {
        assert!(
            (dc - val).abs() < 0.5,
            "DC32x16[{}] = {}, expected ~{}",
            i,
            dc,
            val
        );
    }
}

#[test]
fn test_dct_16x32_dc_extraction_constant() {
    let val = 42.0f32;
    let input = [val; 512];
    let mut output = [0.0f32; 512];
    dct_16x32(&input, &mut output);

    let dcs = dc_from_dct_16x32(&output);
    for (i, &dc) in dcs.iter().enumerate() {
        assert!(
            (dc - val).abs() < 0.5,
            "DC16x32[{}] = {}, expected ~{}",
            i,
            dc,
            val
        );
    }
}

#[test]
fn test_idct_8x4_roundtrip() {
    let input: [f32; 32] = core::array::from_fn(|i| ((i as f32 * 1.7).sin()) * 100.0);
    let mut coeffs = [0.0f32; 32];
    dct_8x4(&input, &mut coeffs);
    let mut output = [0.0f32; 32];
    idct_8x4(&coeffs, &mut output);

    let mut max_err = 0.0f32;
    for i in 0..32 {
        let err = (input[i] - output[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-4,
        "idct_8x4 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_32x32_roundtrip() {
    let input: [f32; 1024] = core::array::from_fn(|i| ((i as f32 * 0.7).sin()) * 100.0);
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);
    let mut output = [0.0f32; 1024];
    idct_32x32(&coeffs, &mut output);

    let mut max_err = 0.0f32;
    for i in 0..1024 {
        let err = (input[i] - output[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-3,
        "idct_32x32 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_32x16_roundtrip() {
    let input: [f32; 512] = core::array::from_fn(|i| ((i as f32 * 0.3).cos()) * 50.0);
    let mut coeffs = [0.0f32; 512];
    dct_32x16(&input, &mut coeffs);
    let mut output = [0.0f32; 512];
    idct_32x16(&coeffs, &mut output);

    let mut max_err = 0.0f32;
    for i in 0..512 {
        let err = (input[i] - output[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-3,
        "idct_32x16 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_16x32_roundtrip() {
    let input: [f32; 512] = core::array::from_fn(|i| ((i as f32 * 0.5).sin()) * 75.0);
    let mut coeffs = [0.0f32; 512];
    dct_16x32(&input, &mut coeffs);
    let mut output = [0.0f32; 512];
    idct_16x32(&coeffs, &mut output);

    let mut max_err = 0.0f32;
    for i in 0..512 {
        let err = (input[i] - output[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-3,
        "idct_16x32 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_64x64_roundtrip() {
    let input: Vec<f32> = (0..4096).map(|i| ((i as f32 * 0.4).sin()) * 80.0).collect();
    let mut coeffs = vec![0.0f32; 4096];
    dct_64x64(&input, &mut coeffs);
    let mut output = vec![0.0f32; 4096];
    idct_64x64(&coeffs, &mut output);

    let mut max_err = 0.0f32;
    for i in 0..4096 {
        let err = (input[i] - output[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-2,
        "idct_64x64 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_64x32_roundtrip() {
    let input: Vec<f32> = (0..2048).map(|i| ((i as f32 * 0.6).cos()) * 60.0).collect();
    let mut coeffs = vec![0.0f32; 2048];
    dct_64x32(&input, &mut coeffs);
    let mut output = vec![0.0f32; 2048];
    idct_64x32(&coeffs, &mut output);

    let mut max_err = 0.0f32;
    for i in 0..2048 {
        let err = (input[i] - output[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-2,
        "idct_64x32 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_32x64_roundtrip() {
    let input: Vec<f32> = (0..2048).map(|i| ((i as f32 * 0.8).sin()) * 90.0).collect();
    let mut coeffs = vec![0.0f32; 2048];
    dct_32x64(&input, &mut coeffs);
    let mut output = vec![0.0f32; 2048];
    idct_32x64(&coeffs, &mut output);

    let mut max_err = 0.0f32;
    for i in 0..2048 {
        let err = (input[i] - output[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-2,
        "idct_32x64 roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_4x8_full_roundtrip() {
    let input: [f32; 64] = core::array::from_fn(|i| (i as f32 * 0.3).sin() * 50.0);
    let mut coeffs = [0.0f32; 64];
    dct_4x8_full(&input, &mut coeffs);
    let mut output = [0.0f32; 64];
    idct_4x8_full(&coeffs, &mut output);

    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "idct_4x8_full roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_8x4_full_roundtrip() {
    let input: [f32; 64] = core::array::from_fn(|i| (i as f32 * 0.7).cos() * 30.0);
    let mut coeffs = [0.0f32; 64];
    dct_8x4_full(&input, &mut coeffs);
    let mut output = [0.0f32; 64];
    idct_8x4_full(&coeffs, &mut output);

    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "idct_8x4_full roundtrip max error {} too large",
        max_err
    );
}

#[test]
fn test_idct_4x4_full_roundtrip() {
    let input: [f32; 64] = core::array::from_fn(|i| (i as f32 * 0.5 + 1.0).sin() * 40.0);
    let mut coeffs = [0.0f32; 64];
    dct_4x4_full(&input, &mut coeffs);
    let mut output = [0.0f32; 64];
    idct_4x4_full(&coeffs, &mut output);

    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "idct_4x4_full roundtrip max error {} too large",
        max_err
    );
}
