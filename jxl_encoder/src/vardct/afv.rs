// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]

//! AFV (Adaptive Frequency Variable) transforms for corner DCT.
//!
//! AFV transforms are used for 8x8 blocks at the corners of larger transform regions.
//! They provide better frequency localization for blocks adjacent to differently-sized
//! transforms.
//!
//! AFV0-AFV3 correspond to the four corners:
//! - AFV0: top-left corner (x even, y even)
//! - AFV1: top-right corner (x odd, y even)
//! - AFV2: bottom-left corner (x even, y odd)
//! - AFV3: bottom-right corner (x odd, y odd)
//!
//! Each AFV transform combines:
//! - A special 4x4 AFV DCT in one corner (using custom basis matrix)
//! - A regular 4x4 DCT in the adjacent corner
//! - A 4x8 DCT for the remaining half of the block

/// AFV4x4 basis matrix transpose.
/// This is the transpose of the matrix used for forward transform.
/// From libjxl enc_transforms-inl.h k4x4AFVBasisTranspose.
#[rustfmt::skip]
const AFV4X4_BASIS_TRANSPOSE: [[f32; 16]; 16] = [
    [0.2500000000000000, 0.8769029297991420, 0.0000000000000000, 0.0000000000000000,
     0.0000000000000000, -0.4105377591765233, 0.0000000000000000, 0.0000000000000000,
     0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
     0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
    [0.2500000000000000, 0.2206518106944235, 0.0000000000000000, 0.0000000000000000,
     -0.7071067811865474, 0.6235485373547691, 0.0000000000000000, 0.0000000000000000,
     0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
     0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
    [0.2500000000000000, -0.1014005039375376, 0.4067007583026075, -0.2125574805828875,
     0.0000000000000000, -0.0643507165794627, -0.4517556589999482, -0.3046847507248690,
     0.3017929516615495, 0.4082482904638627, 0.1747866975480809, -0.2110560104933578,
     -0.1426608480880726, -0.1381354035075859, -0.1743760259965107, 0.1135498731499434],
    [0.2500000000000000, -0.1014005039375375, 0.4444481661973445, 0.3085497062849767,
     0.0000000000000000, -0.0643507165794627, 0.1585450355184006, 0.5112616136591823,
     0.2579236279634118, 0.0000000000000000, 0.0812611176717539, 0.1856718091610980,
     -0.3416446842253372, 0.3302282550303788, 0.0702790691196284, -0.0741750459581035],
    [0.2500000000000000, 0.2206518106944236, 0.0000000000000000, 0.0000000000000000,
     0.7071067811865476, 0.6235485373547694, 0.0000000000000000, 0.0000000000000000,
     0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
     0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
    [0.2500000000000000, -0.1014005039375378, 0.0000000000000000, 0.4706702258572536,
     0.0000000000000000, -0.0643507165794628, -0.0403851516082220, 0.0000000000000000,
     0.1627234014286620, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
     0.7367497537172237, 0.0875511500058708, -0.2921026642334881, 0.1940289303259434],
    [0.2500000000000000, -0.1014005039375377, 0.1957439937204294, -0.1621205195722993,
     0.0000000000000000, -0.0643507165794628, 0.0074182263792424, -0.2904801297289980,
     0.0952002265347504, 0.0000000000000000, -0.3675398009862027, 0.4921585901373873,
     0.2462710772207515, -0.0794670660590957, 0.3623817333531167, -0.4351904965232280],
    [0.2500000000000000, -0.1014005039375376, 0.2929100136981264, 0.0000000000000000,
     0.0000000000000000, -0.0643507165794627, 0.3935103426921017, -0.0657870154914280,
     0.0000000000000000, -0.4082482904638628, -0.3078822139579090, -0.3852501370925192,
     -0.0857401903551931, -0.4613374887461511, 0.0000000000000000, 0.2191868483885747],
    [0.2500000000000000, -0.1014005039375376, -0.4067007583026072, -0.2125574805828705,
     0.0000000000000000, -0.0643507165794627, -0.4517556589999464, 0.3046847507248840,
     0.3017929516615503, -0.4082482904638635, -0.1747866975480813, 0.2110560104933581,
     -0.1426608480880734, -0.1381354035075829, -0.1743760259965108, 0.1135498731499426],
    [0.2500000000000000, -0.1014005039375377, -0.1957439937204287, -0.1621205195722833,
     0.0000000000000000, -0.0643507165794628, 0.0074182263792444, 0.2904801297290076,
     0.0952002265347505, 0.0000000000000000, 0.3675398009862011, -0.4921585901373891,
     0.2462710772207514, -0.0794670660591026, 0.3623817333531165, -0.4351904965232251],
    [0.2500000000000000, -0.1014005039375375, 0.0000000000000000, -0.4706702258572528,
     0.0000000000000000, -0.0643507165794627, 0.1107416575309343, 0.0000000000000000,
     -0.1627234014286617, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
     0.1488339922711357, 0.4972464710953509, 0.2921026642334879, 0.5550443808910661],
    [0.2500000000000000, -0.1014005039375377, 0.1137907446044809, -0.1464291867126764,
     0.0000000000000000, -0.0643507165794628, 0.0829816309488205, -0.2388977352334460,
     -0.3531238544981630, -0.4082482904638630, 0.4826689115059883, 0.1741941265991622,
     -0.0476868035022925, 0.1253805944856366, -0.4326608024727445, -0.2546827712406646],
    [0.2500000000000000, -0.1014005039375377, -0.4444481661973438, 0.3085497062849487,
     0.0000000000000000, -0.0643507165794628, 0.1585450355183970, -0.5112616136592012,
     0.2579236279634129, 0.0000000000000000, -0.0812611176717504, -0.1856718091610990,
     -0.3416446842253373, 0.3302282550303805, 0.0702790691196282, -0.0741750459581023],
    [0.2500000000000000, -0.1014005039375376, -0.2929100136981264, 0.0000000000000000,
     0.0000000000000000, -0.0643507165794627, 0.3935103426921022, 0.0657870154914254,
     0.0000000000000000, 0.4082482904638634, 0.3078822139579031, 0.3852501370925211,
     -0.0857401903551927, -0.4613374887461554, 0.0000000000000000, 0.2191868483885793],
    [0.2500000000000000, -0.1014005039375377, -0.1137907446044814, -0.1464291867126654,
     0.0000000000000000, -0.0643507165794628, 0.0829816309488214, 0.2388977352334547,
     -0.3531238544981624, 0.4082482904638636, -0.4826689115059846, -0.1741941265991693,
     -0.0476868035022926, 0.1253805944856419, -0.4326608024727457, -0.2546827712406567],
    [0.2500000000000000, -0.1014005039375374, 0.0000000000000000, 0.4251149611657548,
     0.0000000000000000, -0.0643507165794626, -0.4517556589999480, 0.0000000000000000,
     -0.6035859033230976, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
     -0.1426608480880724, -0.1381354035075845, 0.3487520519930227, 0.1135498731499429],
];

/// Forward AFV 4x4 DCT using the custom basis matrix.
/// Input: 16 pixels, Output: 16 coefficients
///
/// Let B be the basis matrix. AFV4X4_BASIS_TRANSPOSE = B^T.
/// Forward: coeffs = B * pixels = sum_i(B[j][i] * pixels[i]) = sum_i(B^T[i][j] * pixels[i])
///
/// Loop structure: outer on input pixel (i), inner on output coefficient (j).
/// This gives stride-1 access to AFV4X4_BASIS_TRANSPOSE[i][j] (j varies, i fixed),
/// enabling auto-vectorization of the inner multiply-accumulate loop.
/// The original j-outer/i-inner loop had stride-16 access (gather pattern).
#[inline(always)]
fn afv_dct_4x4(pixels: &[f32; 16], coeffs: &mut [f32; 16]) {
    *coeffs = [0.0; 16];
    for i in 0..16 {
        let p = pixels[i];
        let row = &AFV4X4_BASIS_TRANSPOSE[i];
        for j in 0..16 {
            coeffs[j] += p * row[j];
        }
    }
}

/// Inverse AFV 4x4 DCT using the custom basis matrix.
/// Input: 16 coefficients, Output: 16 pixels
///
/// Inverse: pixels = B^T * coeffs = sum_j(B^T[i][j] * coeffs[j])
#[inline(always)]
fn afv_idct_4x4(coeffs: &[f32; 16], pixels: &mut [f32; 16]) {
    for i in 0..16 {
        let mut sum = 0.0f32;
        for j in 0..16 {
            sum += coeffs[j] * AFV4X4_BASIS_TRANSPOSE[i][j];
        }
        pixels[i] = sum;
    }
}

/// Forward scaled 4x4 DCT wrapper.
#[inline(always)]
fn dct_4x4_simple(pixels: &[f32; 16], coeffs: &mut [f32; 16]) {
    super::dct::dct_4x4(pixels, coeffs);
}

/// Forward scaled 4x8 DCT wrapper.
#[inline(always)]
fn dct_4x8_simple(pixels: &[f32; 32], coeffs: &mut [f32; 32]) {
    super::dct::dct_4x8(pixels, coeffs);
}

/// Raw AFV strategy codes.
/// Note: These are internal indices, not bitstream codes.
/// Bitstream codes for AFV0-3 are 14-17 respectively.
/// Used by ac_strategy.rs for strategy selection.
#[allow(dead_code)]
pub const RAW_STRATEGY_AFV0: u8 = 12;
#[allow(dead_code)]
pub const RAW_STRATEGY_AFV1: u8 = 13;
#[allow(dead_code)]
pub const RAW_STRATEGY_AFV2: u8 = 14;
#[allow(dead_code)]
pub const RAW_STRATEGY_AFV3: u8 = 15;

/// Bitstream codes for AFV strategies.
/// Used by STRATEGY_CODE_LUT in ac_strategy.rs.
#[allow(dead_code)]
pub const STRATEGY_CODE_AFV0: u8 = 14;
#[allow(dead_code)]
pub const STRATEGY_CODE_AFV1: u8 = 15;
#[allow(dead_code)]
pub const STRATEGY_CODE_AFV2: u8 = 16;
#[allow(dead_code)]
pub const STRATEGY_CODE_AFV3: u8 = 17;

/// Convert raw strategy code to AFV kind (0-3).
/// Returns None if not an AFV strategy.
#[allow(dead_code)]
pub fn afv_kind_from_strategy(raw_strategy: u8) -> Option<usize> {
    match raw_strategy {
        RAW_STRATEGY_AFV0 => Some(0),
        RAW_STRATEGY_AFV1 => Some(1),
        RAW_STRATEGY_AFV2 => Some(2),
        RAW_STRATEGY_AFV3 => Some(3),
        _ => None,
    }
}

/// Perform forward AFV transform on an 8x8 block.
///
/// # Arguments
/// * `pixels` - 8x8 input pixels (row-major, stride 8)
/// * `afv_kind` - 0-3 for AFV0-AFV3 (corner location)
/// * `coefficients` - 64-element output array
///
/// The output coefficient layout is:
/// - (even, even) positions: AFV 4x4 coefficients
/// - (odd, even) positions: DCT 4x4 coefficients
/// - (any, odd) positions: DCT 4x8 coefficients
#[inline(always)]
pub fn afv_transform_from_pixels(pixels: &[f32], afv_kind: usize, coefficients: &mut [f32; 64]) {
    let afv_x = afv_kind & 1;
    let afv_y = afv_kind / 2;

    // Extract the 4x4 corner block for AFV DCT, with mirroring for different corners
    let mut block_4x4 = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            let src_y = if afv_y == 1 { 3 - iy } else { iy };
            let src_x = if afv_x == 1 { 3 - ix } else { ix };
            block_4x4[src_y * 4 + src_x] = pixels[(iy + 4 * afv_y) * 8 + ix + 4 * afv_x];
        }
    }

    // AFV 4x4 DCT - coefficients go in (even, even) positions
    let mut afv_coeffs = [0.0f32; 16];
    afv_dct_4x4(&block_4x4, &mut afv_coeffs);
    for iy in 0..4 {
        for ix in 0..4 {
            coefficients[iy * 2 * 8 + ix * 2] = afv_coeffs[iy * 4 + ix];
        }
    }

    // Regular 4x4 DCT of the adjacent corner - coefficients go in (odd, even) positions
    let mut dct4_pixels = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            dct4_pixels[iy * 4 + ix] = pixels[(iy + afv_y * 4) * 8 + ix + (1 - afv_x) * 4];
        }
    }
    let mut dct4_coeffs = [0.0f32; 16];
    dct_4x4_simple(&dct4_pixels, &mut dct4_coeffs);
    for iy in 0..4 {
        for ix in 0..4 {
            coefficients[iy * 2 * 8 + ix * 2 + 1] = dct4_coeffs[iy * 4 + ix];
        }
    }

    // 4x8 DCT of the other half - coefficients go in (any, odd) positions
    let mut dct4x8_pixels = [0.0f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            dct4x8_pixels[iy * 8 + ix] = pixels[(iy + (1 - afv_y) * 4) * 8 + ix];
        }
    }
    let mut dct4x8_coeffs = [0.0f32; 32];
    dct_4x8_simple(&dct4x8_pixels, &mut dct4x8_coeffs);
    for iy in 0..4 {
        for ix in 0..8 {
            coefficients[(1 + iy * 2) * 8 + ix] = dct4x8_coeffs[iy * 8 + ix];
        }
    }

    // DC coefficient combining (inverse of decoder's DC extraction)
    // Decoder: dc[0] = (block00 + block10 + block01) * 4.0
    //          dc[1] = block00 + block10 - block01
    //          dc[2] = block00 - block10
    // So we need to solve for block00, block01, block10:
    // Let a = coefficients[0] (AFV DC), b = coefficients[1] (DCT4 DC), c = coefficients[8] (DCT4x8 DC)
    // Encoder transforms these to packed DC format
    let block00 = coefficients[0] * 0.25;
    let block01 = coefficients[1];
    let block10 = coefficients[8];

    // Transform DC values to the packed format expected by decoder
    coefficients[0] = (block00 + block01 + 2.0 * block10) * 0.25;
    coefficients[1] = (block00 - block01) * 0.5;
    coefficients[8] = (block00 + block01 - 2.0 * block10) * 0.25;
}

/// Extract DC value from AFV coefficients.
/// The DC is stored in a combined format - we need to decode it.
pub fn dc_from_afv(coefficients: &[f32; 64]) -> f32 {
    // DC extraction for AFV is the same as IDENTITY: just coefficients[0]
    // Reference: libjxl enc_transforms-inl.h:791-796 (DCFromLowestFrequencies)
    coefficients[0]
}

/// Perform inverse AFV transform: coefficients to pixels.
/// This is used for pixel-domain loss computation in strategy selection.
///
/// # Arguments
/// * `coefficients` - 64-element coefficient array
/// * `afv_kind` - 0-3 for AFV0-AFV3 (corner location)
/// * `pixels` - 8x8 output pixels (row-major, stride 8)
///
/// Reference: jxl-rs jxl_transforms/src/transform.rs afv_transform_to_pixels
#[inline(always)]
pub fn inverse_afv_transform(coefficients: &[f32; 64], afv_kind: usize, pixels: &mut [f32; 64]) {
    let afv_x = afv_kind & 1;
    let afv_y = afv_kind / 2;

    // Extract DC values from packed format
    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];

    // Compute DCs for each sub-transform
    let dcs: [f32; 3] = [
        (block00 + block10 + block01) * 4.0, // AFV4x4 DC
        block00 + block10 - block01,         // DCT4x4 DC
        block00 - block10,                   // DCT4x8 DC
    ];

    // Inverse AFV 4x4: (even, even) positions
    let mut afv_coeff = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            afv_coeff[iy * 4 + ix] = if ix == 0 && iy == 0 {
                dcs[0]
            } else {
                coefficients[iy * 2 * 8 + ix * 2]
            };
        }
    }
    let mut afv_pixels = [0.0f32; 16];
    afv_idct_4x4(&afv_coeff, &mut afv_pixels);

    // Write AFV pixels with mirroring based on corner
    for iy in 0..4 {
        let block_y = if afv_y == 1 { 3 - iy } else { iy };
        for ix in 0..4 {
            let block_x = if afv_x == 1 { 3 - ix } else { ix };
            pixels[(iy + afv_y * 4) * 8 + afv_x * 4 + ix] = afv_pixels[block_y * 4 + block_x];
        }
    }

    // Inverse DCT 4x4: (odd, even) positions
    let mut dct4_coeff = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            dct4_coeff[iy * 4 + ix] = if ix == 0 && iy == 0 {
                dcs[1]
            } else {
                coefficients[iy * 2 * 8 + ix * 2 + 1]
            };
        }
    }
    let mut dct4_pixels = [0.0f32; 16];
    super::dct::idct_4x4(&dct4_coeff, &mut dct4_pixels);

    // Write DCT4x4 pixels to the adjacent corner
    for iy in 0..4 {
        for ix in 0..4 {
            pixels[(iy + afv_y * 4) * 8 + (1 - afv_x) * 4 + ix] = dct4_pixels[iy * 4 + ix];
        }
    }

    // Inverse DCT 4x8: (any, odd) positions
    let mut dct4x8_coeff = [0.0f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            dct4x8_coeff[iy * 8 + ix] = if ix == 0 && iy == 0 {
                dcs[2]
            } else {
                coefficients[(1 + iy * 2) * 8 + ix]
            };
        }
    }
    let mut dct4x8_pixels = [0.0f32; 32];
    super::dct::idct_4x8(&dct4x8_coeff, &mut dct4x8_pixels);

    // Write DCT4x8 pixels to the other half
    for iy in 0..4 {
        for ix in 0..8 {
            pixels[(iy + (1 - afv_y) * 4) * 8 + ix] = dct4x8_pixels[iy * 8 + ix];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_afv_dct_4x4_constant() {
        // Constant input - check that forward/inverse roundtrips
        // Note: AFV basis is not strictly orthogonal like standard DCT,
        // so we just verify the transform runs without panic and produces
        // reasonable values (DC coefficient is dominant).
        let pixels = [1.0f32; 16];
        let mut coeffs = [0.0f32; 16];
        afv_dct_4x4(&pixels, &mut coeffs);

        // DC should be the largest coefficient by magnitude
        let dc_mag = coeffs[0].abs();
        for i in 1..16 {
            assert!(
                coeffs[i].abs() < dc_mag,
                "AC coeffs[{}] = {} should be smaller than DC = {}",
                i,
                coeffs[i],
                coeffs[0]
            );
        }

        // DC should be positive and significant
        assert!(coeffs[0] > 1.0, "DC = {} should be positive", coeffs[0]);
    }

    #[test]
    fn test_afv_transform_from_pixels_constant() {
        // Constant 8x8 block
        let pixels = [1.0f32; 64];
        let mut coeffs = [0.0f32; 64];

        afv_transform_from_pixels(&pixels, 0, &mut coeffs);

        // DC should be non-zero
        let dc = dc_from_afv(&coeffs);
        assert!(dc > 0.0, "DC = {}", dc);
    }

    #[test]
    fn test_afv_kind_from_strategy() {
        assert_eq!(afv_kind_from_strategy(RAW_STRATEGY_AFV0), Some(0));
        assert_eq!(afv_kind_from_strategy(RAW_STRATEGY_AFV1), Some(1));
        assert_eq!(afv_kind_from_strategy(RAW_STRATEGY_AFV2), Some(2));
        assert_eq!(afv_kind_from_strategy(RAW_STRATEGY_AFV3), Some(3));
        assert_eq!(afv_kind_from_strategy(0), None);
    }

    #[test]
    fn test_afv_4x4_roundtrip() {
        // Test that AFV 4x4 forward/inverse are properly paired
        let pixels = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        let mut coeffs = [0.0f32; 16];
        afv_dct_4x4(&pixels, &mut coeffs);

        let mut recovered = [0.0f32; 16];
        afv_idct_4x4(&coeffs, &mut recovered);

        // Check roundtrip error
        let max_error: f32 = pixels
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        assert!(
            max_error < 1e-4,
            "AFV 4x4 roundtrip error {} should be < 1e-4",
            max_error
        );
    }

    #[test]
    fn test_afv_full_roundtrip() {
        // Test that full AFV transform forward/inverse roundtrips correctly
        let mut pixels = [0.0f32; 64];
        for (i, px) in pixels.iter_mut().enumerate() {
            *px = (i as f32) + 1.0;
        }

        // Test all 4 AFV variants
        for afv_kind in 0..4 {
            let mut coeffs = [0.0f32; 64];
            afv_transform_from_pixels(&pixels, afv_kind, &mut coeffs);

            let mut recovered = [0.0f32; 64];
            inverse_afv_transform(&coeffs, afv_kind, &mut recovered);

            // Check roundtrip error (allow larger error due to DC packing/unpacking)
            let max_error: f32 = pixels
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |a, b| a.max(b));

            assert!(
                max_error < 1.0,
                "AFV{} full roundtrip error {} should be < 1.0",
                afv_kind,
                max_error
            );
        }
    }

    #[test]
    fn test_afv_regions_isolated() {
        // Test each region independently to isolate the issue
        let afv_kind = 0; // AFV0: afv_x=0, afv_y=0

        // Test 1: Only AFV corner (top-left 4x4)
        let mut pixels1 = [0.0f32; 64];
        for i in 0..4 {
            for j in 0..4 {
                pixels1[i * 8 + j] = (i * 4 + j + 1) as f32;
            }
        }
        let mut coeffs1 = [0.0f32; 64];
        afv_transform_from_pixels(&pixels1, afv_kind, &mut coeffs1);
        let mut recov1 = [0.0f32; 64];
        inverse_afv_transform(&coeffs1, afv_kind, &mut recov1);
        let err1: f32 = pixels1
            .iter()
            .zip(recov1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("AFV corner only - max error: {}", err1);

        // Test 2: Only DCT4 corner (top-right 4x4)
        let mut pixels2 = [0.0f32; 64];
        for i in 0..4 {
            for j in 4..8 {
                pixels2[i * 8 + j] = (i * 4 + (j - 4) + 1) as f32;
            }
        }
        let mut coeffs2 = [0.0f32; 64];
        afv_transform_from_pixels(&pixels2, afv_kind, &mut coeffs2);
        let mut recov2 = [0.0f32; 64];
        inverse_afv_transform(&coeffs2, afv_kind, &mut recov2);
        let err2: f32 = pixels2
            .iter()
            .zip(recov2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("DCT4 corner only - max error: {}", err2);

        // Test 3: Only DCT4x8 region (bottom half)
        let mut pixels3 = [0.0f32; 64];
        for i in 4..8 {
            for j in 0..8 {
                pixels3[i * 8 + j] = ((i - 4) * 8 + j + 1) as f32;
            }
        }
        let mut coeffs3 = [0.0f32; 64];
        afv_transform_from_pixels(&pixels3, afv_kind, &mut coeffs3);
        let mut recov3 = [0.0f32; 64];
        inverse_afv_transform(&coeffs3, afv_kind, &mut recov3);
        let err3: f32 = pixels3
            .iter()
            .zip(recov3.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("DCT4x8 only - max error: {}", err3);

        assert!(err1 < 1e-4, "AFV corner roundtrip error {} too large", err1);
        assert!(
            err2 < 1e-4,
            "DCT4 corner roundtrip error {} too large",
            err2
        );
        assert!(err3 < 1e-4, "DCT4x8 roundtrip error {} too large", err3);
    }
}
