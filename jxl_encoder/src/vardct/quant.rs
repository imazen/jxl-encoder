// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Quantization weights and matrices for the encoder.
//!
//! All weights are generated parametrically from libjxl's default band parameters
//! (quant_weights.cc). This matches what the decoder expects when the frame header
//! signals `all_default=true`.

// Ported float constants from C++ - exact values are intentional for parity.
#![allow(clippy::excessive_precision)]

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use once_cell::race::OnceBox;

/// Number of valid AC strategies.
/// 0 = DCT8 (8x8), 1 = DCT16X8, 2 = DCT8X16, 3 = DCT16X16, 4 = DCT32X32,
/// 5 = DCT4X8, 6 = DCT8X4, 7 = DCT4X4, 8 = IDENTITY, 9 = DCT2X2,
/// 10 = DCT32X16, 11 = DCT16X32, 12 = AFV0, 13 = AFV1, 14 = AFV2, 15 = AFV3,
/// 16 = DCT64X64, 17 = DCT64X32, 18 = DCT32X64
pub const NUM_VALID_STRATEGIES: usize = 19;

/// Inverse DC quantization constants per channel (X, Y, B).
/// These are the denominators for DC quantization.
pub const INV_DC_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];

/// DC quantization constants per channel (X, Y, B).
/// DC_QUANT[c] = 1.0 / INV_DC_QUANT[c]
#[allow(dead_code)]
pub const DC_QUANT: [f32; 3] = [
    1.0 / 4096.0, // X channel
    1.0 / 512.0,  // Y channel
    1.0 / 256.0,  // B channel
];

// =============================================================================
// Parametric weight generation
// =============================================================================

/// Band multiplier: converts distance_bands parameter to multiplicative factor.
/// Matches libjxl `Mult()` in quant_weights.cc.
fn band_mult(v: f64) -> f64 {
    if v > 0.0 { 1.0 + v } else { 1.0 / (1.0 - v) }
}

/// Interpolate in log-space between band values.
/// Matches libjxl `Interpolate()` / `InterpolateVec()`.
fn interpolate_band(pos: f64, bands: &[f64]) -> f64 {
    let len = bands.len();
    if len == 1 {
        return bands[0];
    }
    let idx = (pos as usize).min(len - 2);
    let frac = pos - idx as f64;
    let a = bands[idx];
    let b = bands[idx + 1];
    // a * pow(b/a, frac) = exp(log(a) + frac * log(b/a))
    a * (b / a).powf(frac)
}

/// Generate quantization weights for a ROWS x COLS DCT transform using the parametric formula.
///
/// Matches libjxl `GetQuantWeights(ROWS, COLS, ...)` in quant_weights.cc.
/// Band params are `[initial, mult1, mult2, ...]` where `bands[0] = initial`
/// and `bands[i] = bands[i-1] * Mult(params[i])`.
///
/// Returns `3 * rows * cols` floats: X channel, then Y, then B.
/// These are **quant** weights (1/dequant), matching what the encoder needs.
fn generate_dct_quant_weights_rect(
    rows: usize,
    cols: usize,
    band_params: &[&[f64]; 3],
    num_bands: usize,
) -> Vec<f32> {
    let num = rows * cols;
    let total = 3 * num;
    let mut out = vec![0.0f32; total];

    let sqrt2 = core::f64::consts::SQRT_2;
    let scale = (num_bands as f64 - 1.0) / (sqrt2 + 1e-6);
    let rcpcol = scale / (cols as f64 - 1.0);
    let rcprow = scale / (rows as f64 - 1.0);

    for c in 0..3 {
        // Build band values from parameters
        let params = band_params[c];
        let mut bands = vec![0.0f64; num_bands];
        bands[0] = params[0];
        for i in 1..num_bands {
            bands[i] = bands[i - 1] * band_mult(params[i]);
        }

        for y in 0..rows {
            let dy = y as f64 * rcprow;
            let dy2 = dy * dy;
            for x in 0..cols {
                let dx = x as f64 * rcpcol;
                let scaled_distance = (dx * dx + dy2).sqrt();
                let dequant_weight = interpolate_band(scaled_distance, &bands);
                let quant_weight = 1.0 / dequant_weight;
                out[c * num + y * cols + x] = quant_weight as f32;
            }
        }
    }

    out
}

// =============================================================================
// Band parameters from libjxl quant_weights.cc (default library values)
// =============================================================================

/// DCT8 band parameters from libjxl quant_weights.cc:535-561.
/// 6 distance bands per channel.
const DCT8_PARAMS: [[f64; 6]; 3] = [
    // X channel
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
    // Y channel
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
    // B channel
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
];

/// DCT16x16 band parameters from libjxl quant_weights.cc:647-676.
/// 7 distance bands per channel.
const DCT16X16_PARAMS: [[f64; 7]; 3] = [
    // X channel
    [
        8996.8725711814115328,
        -1.3000777393353804,
        -0.49424529824571225,
        -0.439093774457103443,
        -0.6350101832695744,
        -0.90177264050827612,
        -1.6162099239887414,
    ],
    // Y channel
    [
        3191.48366296844234752,
        -0.67424582104194355,
        -0.80745813428471001,
        -0.44925837484843441,
        -0.35865440981033403,
        -0.31322389111877305,
        -0.37615025315725483,
    ],
    // B channel
    [
        1157.50408145487200256,
        -2.0531423165804414,
        -1.4,
        -0.50687130033378396,
        -0.42708730624733904,
        -1.4856834539296244,
        -4.9209142884401604,
    ],
];

/// DCT16x8 (8 rows x 16 cols) band parameters from libjxl quant_weights.cc:716-745.
/// 7 distance bands per channel.
///
/// Note: libjxl names this "DCT8X16" in the QuantTable enum (column-major naming),
/// but the GetQuantWeights call uses (ROWS=8, COLS=16).
const DCT16X8_PARAMS: [[f64; 7]; 3] = [
    // X channel
    [7240.7734393502, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
    // Y channel
    [1448.15468787004, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
    // B channel
    [506.854140754517, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
];

/// DCT32x32 band parameters from libjxl quant_weights.cc:680-712.
/// 8 distance bands per channel.
const DCT32X32_BAND_PARAMS: [[f64; 8]; 3] = [
    // X channel
    [
        15718.40830982518931456,
        -1.025,
        -0.98,
        -0.9012,
        -0.4,
        -0.48819395464,
        -0.421064,
        -0.27,
    ],
    // Y channel
    [
        7305.7636810695983104,
        -0.8041958212306401,
        -0.7633036457487539,
        -0.55660379990111464,
        -0.49785304658857626,
        -0.43699592683512467,
        -0.40180866526242109,
        -0.27321683125358037,
    ],
    // B channel
    [
        3803.53173721215041536,
        -3.060733579805728,
        -2.0413270132490346,
        -2.0235650159727417,
        -0.5495389509954993,
        -0.4,
        -0.4,
        -0.3,
    ],
];

/// DCT16X32 band parameters from jxl-rs quant_weights.rs:561-590 (Dct16x32 case).
/// 8 distance bands per channel. Used for both DCT32X16 and DCT16X32.
const DCT16X32_BAND_PARAMS: [[f64; 8]; 3] = [
    // X channel
    [
        13844.97076442300573,
        -0.97113799999999995,
        -0.658,
        -0.42026,
        -0.22712,
        -0.2206,
        -0.226,
        -0.6,
    ],
    // Y channel
    [
        4798.964084220744293,
        -0.61125308982767057,
        -0.83770786552491361,
        -0.79014862079498627,
        -0.2692727459704829,
        -0.38272769465388551,
        -0.22924222653091453,
        -0.20719098826199578,
    ],
    // B channel
    [
        1807.236946760964614,
        -1.2,
        -1.2,
        -0.7,
        -0.7,
        -0.7,
        -0.4,
        -0.5,
    ],
];

/// DCT64X64 band parameters from libjxl quant_weights.cc:899-931.
/// 8 distance bands per channel.
const DCT64X64_BAND_PARAMS: [[f64; 8]; 3] = [
    // X channel (0.9 * 26629.073922049845 = 23966.16653...)
    [
        23966.16652984486,
        -1.025,
        -0.78,
        -0.65012,
        -0.19041574084286472,
        -0.20819395464,
        -0.421064,
        -0.32733845535848671,
    ],
    // Y channel (0.9 * 9311.3238710010046 = 8380.19148...)
    [
        8380.191483900904,
        -0.3041958212306401,
        -0.3633036457487539,
        -0.35660379990111464,
        -0.3443074455424403,
        -0.33699592683512467,
        -0.30180866526242109,
        -0.27321683125358037,
    ],
    // B channel (0.9 * 4992.2486445538634 = 4493.02378...)
    [
        4493.02378009847706,
        -1.2,
        -1.2,
        -0.8,
        -0.7,
        -0.7,
        -0.4,
        -0.5,
    ],
];

/// DCT32X64/DCT64X32 band parameters from libjxl quant_weights.cc:935-968.
/// 8 distance bands per channel. Used for both DCT64X32 and DCT32X64.
const DCT32X64_BAND_PARAMS: [[f64; 8]; 3] = [
    // X channel (0.65 * 23629.073922049845 = 15358.898...)
    [
        15358.898049332399,
        -1.025,
        -0.78,
        -0.65012,
        -0.19041574084286472,
        -0.20819395464,
        -0.421064,
        -0.32733845535848671,
    ],
    // Y channel (0.65 * 8611.3238710010046 = 5597.360...)
    [
        5597.36051615065299,
        -0.3041958212306401,
        -0.3633036457487539,
        -0.35660379990111464,
        -0.3443074455424403,
        -0.33699592683512467,
        -0.30180866526242109,
        -0.27321683125358037,
    ],
    // B channel (0.65 * 4492.2486445538634 = 2919.961...)
    [2919.961618960011, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5],
];

/// DCT4X8 band parameters from jxl-oxide dequant.rs:44-48.
/// 4 distance bands per channel.
const DCT4X8_BAND_PARAMS: [[f64; 4]; 3] = [
    // X channel
    [2198.0505, -0.96269625, -0.7619425, -0.65511405],
    // Y channel
    [764.36554, -0.926302, -0.967523, -0.2784529],
    // B channel
    [527.10754, -1.4594386, -1.4500821, -1.5843723],
];

/// DCT4X4 band parameters from jxl-oxide dequant.rs:49-53.
/// 4 distance bands per channel.
const DCT4_BAND_PARAMS: [[f64; 4]; 3] = [
    // X channel
    [2200.0, 0.0, 0.0, 0.0],
    // Y channel
    [392.0, 0.0, 0.0, 0.0],
    // B channel
    [112.0, -0.25, -0.25, -0.5],
];

/// DCT4X4 LLF multiplier parameters from jxl-oxide dequant.rs:257-277.
/// params[0] is used for LLF positions 1 and 8.
/// params[1] is used for LLF position 9.
const DCT4_LLF_PARAMS: [[f64; 2]; 3] = [
    // X channel
    [1.0, 1.0],
    // Y channel
    [1.0, 1.0],
    // B channel
    [1.0, 1.0],
];

/// AFV weight parameters from jxl-rs quant_weights.rs and libjxl quant_weights.cc.
/// Per-channel: [dc_tend0, dc_tend1, corner0, corner1, corner2, band0, band1, band2, band3]
/// - dc_tend0, dc_tend1: weights for positions (0,1) and (1,0)
/// - corner0, corner1, corner2: weights for positions (0,2), (2,0), (2,2)
/// - band0-3: distance bands for interpolating other AFV positions
const AFV_WEIGHTS: [[f64; 9]; 3] = [
    // X channel
    [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
    // Y channel
    [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
    // B channel
    [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
];

/// Frequency lookup table for AFV interpolation.
/// From libjxl quant_weights.cc kFreqs array.
const AFV_FREQS: [f64; 16] = [
    0.0,                // (0,0) - not used
    0.0,                // (1,0) - not used
    0.8517778890324296, // (2,0)
    5.37778436506804,   // (3,0)
    0.0,                // (0,1) - not used
    0.0,                // (1,1) - not used
    4.734747904497923,  // (2,1)
    5.449245381693219,  // (3,1)
    1.6598270267479331, // (0,2)
    4.0,                // (1,2)
    7.275749096817861,  // (2,2)
    10.423227632456525, // (3,2)
    2.662932286148962,  // (0,3)
    7.630657783650829,  // (1,3)
    8.962388608184032,  // (2,3)
    12.97166202570235,  // (3,3)
];

// =============================================================================
// Lazily-generated weight tables
// =============================================================================

/// DCT8 quantization weights (192 floats: 64 per channel).
/// Generated from libjxl's default DCT8 band parameters.
static QUANT_WEIGHTS_DCT8: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct8() -> &'static [f32] {
    QUANT_WEIGHTS_DCT8.get_or_init(|| {
        Box::new(generate_dct_quant_weights_rect(
            8,
            8,
            &[&DCT8_PARAMS[0], &DCT8_PARAMS[1], &DCT8_PARAMS[2]],
            6,
        ))
    })
}

/// DCT16x16 quantization weights (768 floats: 256 per channel).
/// Generated from libjxl's default DCT16x16 band parameters.
static QUANT_WEIGHTS_DCT16X16: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct16x16() -> &'static [f32] {
    QUANT_WEIGHTS_DCT16X16.get_or_init(|| {
        Box::new(generate_dct_quant_weights_rect(
            16,
            16,
            &[
                &DCT16X16_PARAMS[0],
                &DCT16X16_PARAMS[1],
                &DCT16X16_PARAMS[2],
            ],
            7,
        ))
    })
}

/// DCT16x8 quantization weights (384 floats: 128 per channel).
/// Generated from libjxl's default DCT8X16 band parameters.
/// GetQuantWeights(ROWS=8, COLS=16, ...) produces 8x16 = 128 values per channel.
static QUANT_WEIGHTS_DCT16X8: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct16x8() -> &'static [f32] {
    QUANT_WEIGHTS_DCT16X8.get_or_init(|| {
        Box::new(generate_dct_quant_weights_rect(
            8,
            16,
            &[&DCT16X8_PARAMS[0], &DCT16X8_PARAMS[1], &DCT16X8_PARAMS[2]],
            7,
        ))
    })
}

/// DCT32x32 quantization weights (3072 floats: 1024 per channel).
static QUANT_WEIGHTS_DCT32X32: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct32x32() -> &'static [f32] {
    QUANT_WEIGHTS_DCT32X32.get_or_init(|| {
        Box::new(generate_dct_quant_weights_rect(
            32,
            32,
            &[
                &DCT32X32_BAND_PARAMS[0],
                &DCT32X32_BAND_PARAMS[1],
                &DCT32X32_BAND_PARAMS[2],
            ],
            8,
        ))
    })
}

/// DCT16x32/DCT32x16 quantization weights (1536 floats: 512 per channel).
/// Used for both DCT32X16 (raw strategy 10) and DCT16X32 (raw strategy 11).
static QUANT_WEIGHTS_DCT16X32: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct16x32() -> &'static [f32] {
    QUANT_WEIGHTS_DCT16X32.get_or_init(|| {
        Box::new(generate_dct_quant_weights_rect(
            16,
            32,
            &[
                &DCT16X32_BAND_PARAMS[0],
                &DCT16X32_BAND_PARAMS[1],
                &DCT16X32_BAND_PARAMS[2],
            ],
            8,
        ))
    })
}

/// DCT64x64 quantization weights (12288 floats: 4096 per channel).
static QUANT_WEIGHTS_DCT64X64: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct64x64() -> &'static [f32] {
    QUANT_WEIGHTS_DCT64X64.get_or_init(|| {
        Box::new(generate_dct_quant_weights_rect(
            64,
            64,
            &[
                &DCT64X64_BAND_PARAMS[0],
                &DCT64X64_BAND_PARAMS[1],
                &DCT64X64_BAND_PARAMS[2],
            ],
            8,
        ))
    })
}

/// DCT32x64/DCT64x32 quantization weights (6144 floats: 2048 per channel).
/// Used for both DCT64X32 (raw strategy 17) and DCT32X64 (raw strategy 18).
static QUANT_WEIGHTS_DCT32X64: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct32x64() -> &'static [f32] {
    QUANT_WEIGHTS_DCT32X64.get_or_init(|| {
        Box::new(generate_dct_quant_weights_rect(
            32,
            64,
            &[
                &DCT32X64_BAND_PARAMS[0],
                &DCT32X64_BAND_PARAMS[1],
                &DCT32X64_BAND_PARAMS[2],
            ],
            8,
        ))
    })
}

/// Generate DCT4X8 quantization weights using parametric formula.
/// Matches jxl-oxide's dequant.rs:279-294.
///
/// Process:
/// 1. Generate 8x4 weight matrix using parametric bands
/// 2. Duplicate each row to get 8x8 (matching interleaved coefficient layout)
/// 3. Reciprocate to match encoder convention (encoder uses 1/weight)
fn generate_dct4x8_weights() -> Vec<f32> {
    let mut weights = Vec::with_capacity(192);
    let sqrt2 = core::f64::consts::SQRT_2;

    for params in &DCT4X8_BAND_PARAMS {
        // Build bands from parameters
        let mut bands = vec![params[0]];
        let mut last = params[0];
        for &v in &params[1..] {
            last *= band_mult(v);
            bands.push(last);
        }

        // Generate 8x4 matrix (width=8, height=4)
        let width = 8usize;
        let height = 4usize;
        let mut mat_8x4 = vec![0.0f64; width * height];

        for y in 0..height {
            let dy = y as f64 / (height - 1).max(1) as f64;
            for x in 0..width {
                let dx = x as f64 / (width - 1).max(1) as f64;
                let distance = (dx * dx + dy * dy).sqrt();
                let scaled = distance * (bands.len() - 1) as f64 / (sqrt2 + 1e-6);
                let weight = interpolate_band(scaled, &bands);
                mat_8x4[y * width + x] = weight;
            }
        }

        // Duplicate rows to get 8x8 matrix (matching interleaved layout)
        // Original rows: [row0, row1, row2, row3]
        // Duplicated:    [row0, row0, row1, row1, row2, row2, row3, row3]
        // Also reciprocate to match encoder convention (encoder multiplies by 1/weight)
        for row in 0..height {
            // First copy of row (at position row*2)
            for x in 0..width {
                // Reciprocate: parametric generates dequant weights, we need quant weights
                weights.push((1.0 / mat_8x4[row * width + x]) as f32);
            }
            // Second copy of row (at position row*2+1)
            for x in 0..width {
                weights.push((1.0 / mat_8x4[row * width + x]) as f32);
            }
        }
    }

    weights
}

/// DCT4X8 quantization weights (192 floats: 64 per channel).
/// Parametric formula matching jxl-oxide decoder, row-duplicated from 8x4 base.
static QUANT_WEIGHTS_DCT4X8: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct4x8() -> &'static [f32] {
    QUANT_WEIGHTS_DCT4X8.get_or_init(|| Box::new(generate_dct4x8_weights()))
}

/// DCT8X4 quantization weights (192 floats: 64 per channel).
/// Same parametric formula as DCT4X8.
static QUANT_WEIGHTS_DCT8X4: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct8x4() -> &'static [f32] {
    QUANT_WEIGHTS_DCT8X4.get_or_init(|| Box::new(generate_dct4x8_weights()))
}

/// Generate DCT4X4 quantization weights using parametric formula.
/// Matches jxl-oxide's dequant.rs:257-277 (Dct4 case).
///
/// Process:
/// 1. Generate 4x4 weight matrix using parametric bands
/// 2. Replicate each weight to a 2x2 region in the 8x8 output
/// 3. Apply LLF divisors to positions 1, 8, 9
/// 4. Reciprocate to match encoder convention (encoder uses 1/weight)
fn generate_dct4x4_weights() -> Vec<f32> {
    let mut weights = Vec::with_capacity(192);
    let sqrt2 = core::f64::consts::SQRT_2;

    for (c, params) in DCT4_BAND_PARAMS.iter().enumerate() {
        // Build bands from parameters
        let mut bands = vec![params[0]];
        let mut last = params[0];
        for &v in &params[1..] {
            last *= band_mult(v);
            bands.push(last);
        }

        // Generate 4x4 base matrix
        let size = 4usize;
        let mut mat_4x4 = vec![0.0f64; size * size];

        for y in 0..size {
            let dy = y as f64 / (size - 1).max(1) as f64;
            for x in 0..size {
                let dx = x as f64 / (size - 1).max(1) as f64;
                let distance = (dx * dx + dy * dy).sqrt();
                let scaled = distance * (bands.len() - 1) as f64 / (sqrt2 + 1e-6);
                let weight = interpolate_band(scaled, &bands);
                mat_4x4[y * size + x] = weight;
            }
        }

        // Build 8x8 output by replicating each 4x4 weight to a 2x2 region
        // Layout: mat_4x4[y*4+x] maps to output positions:
        //   [y*16 + x*2], [y*16 + x*2 + 1], [(y*2+1)*8 + x*2], [(y*2+1)*8 + x*2 + 1]
        let mut channel_weights = vec![0.0f64; 64];
        for y in 0..4 {
            for x in 0..4 {
                let w = mat_4x4[y * 4 + x];
                // Top-left of 2x2
                channel_weights[y * 16 + x * 2] = w;
                // Top-right of 2x2
                channel_weights[y * 16 + x * 2 + 1] = w;
                // Bottom-left of 2x2
                channel_weights[(y * 2 + 1) * 8 + x * 2] = w;
                // Bottom-right of 2x2
                channel_weights[(y * 2 + 1) * 8 + x * 2 + 1] = w;
            }
        }

        // Apply LLF divisors (jxl-oxide divides positions 1, 8 by params[0], position 9 by params[1])
        channel_weights[1] /= DCT4_LLF_PARAMS[c][0];
        channel_weights[8] /= DCT4_LLF_PARAMS[c][0];
        channel_weights[9] /= DCT4_LLF_PARAMS[c][1];

        // Reciprocate: parametric generates dequant weights, we need quant weights
        for w in &channel_weights {
            weights.push((1.0 / w) as f32);
        }
    }

    weights
}

/// DCT4X4 quantization weights (192 floats: 64 per channel).
/// Parametric formula matching jxl-oxide decoder, 2x2-replicated from 4x4 base.
static QUANT_WEIGHTS_DCT4X4: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct4x4() -> &'static [f32] {
    QUANT_WEIGHTS_DCT4X4.get_or_init(|| Box::new(generate_dct4x4_weights()))
}

// =============================================================================
// IDENTITY weights (quant_weights.cc:80-90, 564-579)
// =============================================================================

/// IDENTITY quantization weights from libjxl defaults.
/// 3 weights per channel: [DC, AC_pos1_pos8, AC_pos9].
/// All other positions use the DC weight.
const IDENTITY_WEIGHTS: [[f32; 3]; 3] = [
    [280.0, 3160.0, 3160.0], // X channel
    [60.0, 864.0, 864.0],    // Y channel
    [18.0, 200.0, 200.0],    // B channel
];

/// Generate 64-position IDENTITY quant weights for all 3 channels (192 floats).
/// These are **quant** weights (1/dequant), matching the encoder convention.
fn generate_identity_weights() -> Vec<f32> {
    let mut weights = vec![0.0f32; 3 * 64];
    for (c, ch_weights) in IDENTITY_WEIGHTS.iter().enumerate() {
        let start = c * 64;
        let dequant0 = ch_weights[0];
        let dequant1 = ch_weights[1];
        let dequant2 = ch_weights[2];
        // Fill all positions with DC weight (1/dequant)
        for w in &mut weights[start..start + 64] {
            *w = 1.0 / dequant0;
        }
        // Override specific positions
        weights[start + 1] = 1.0 / dequant1;
        weights[start + 8] = 1.0 / dequant1;
        weights[start + 9] = 1.0 / dequant2;
    }
    weights
}

static QUANT_WEIGHTS_IDENTITY: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_identity() -> &'static [f32] {
    QUANT_WEIGHTS_IDENTITY.get_or_init(|| Box::new(generate_identity_weights()))
}

// =============================================================================
// DCT2X2 weights (quant_weights.cc:48-77, 583-607)
// =============================================================================

/// DCT2X2 dequantization weights from libjxl defaults.
/// 6 weights per channel for hierarchical frequency bands.
const DCT2_WEIGHTS: [[f32; 6]; 3] = [
    [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0], // X channel
    [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],    // Y channel
    [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],       // B channel
];

/// Generate 64-position DCT2X2 quant weights for all 3 channels (192 floats).
/// Matches libjxl GetQuantWeightsDCT2() mapping from 6 band weights to 64 positions.
fn generate_dct2x2_weights() -> Vec<f32> {
    let mut weights = vec![0.0f32; 3 * 64];
    for (c, band_weights) in DCT2_WEIGHTS.iter().enumerate() {
        let start = c * 64;
        let w = band_weights;
        // Position 0 (DC) is handled separately by the encoder via INV_DC_QUANT.
        // Mark with 0xBAD sentinel matching libjxl (quant = 1/0xBAD ≈ 0.00002).
        weights[start] = 1.0 / 0xBAD as f32;
        // Positions 1, 8 → band 0
        weights[start + 1] = 1.0 / w[0];
        weights[start + 8] = 1.0 / w[0];
        // Position 9 → band 1
        weights[start + 9] = 1.0 / w[1];
        // 2x2 regions: positions (y*8+x) for y∈{0,1}, x∈{2,3} and y∈{2,3}, x∈{0,1} → band 2
        for y in 0..2usize {
            for x in 0..2usize {
                weights[start + y * 8 + x + 2] = 1.0 / w[2];
                weights[start + (y + 2) * 8 + x] = 1.0 / w[2];
            }
        }
        // Positions (y+2)*8 + (x+2) for y∈{0,1}, x∈{0,1} → band 3
        for y in 0..2usize {
            for x in 0..2usize {
                weights[start + (y + 2) * 8 + x + 2] = 1.0 / w[3];
            }
        }
        // Positions y*8+x+4 and (y+4)*8+x for y∈{0..4}, x∈{0..4} → band 4
        for y in 0..4usize {
            for x in 0..4usize {
                weights[start + y * 8 + x + 4] = 1.0 / w[4];
                weights[start + (y + 4) * 8 + x] = 1.0 / w[4];
            }
        }
        // Bottom-right 4x4: (y+4)*8 + (x+4) for y∈{0..4}, x∈{0..4} → band 5
        for y in 0..4usize {
            for x in 0..4usize {
                weights[start + (y + 4) * 8 + x + 4] = 1.0 / w[5];
            }
        }
    }
    weights
}

static QUANT_WEIGHTS_DCT2X2: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_dct2x2() -> &'static [f32] {
    QUANT_WEIGHTS_DCT2X2.get_or_init(|| Box::new(generate_dct2x2_weights()))
}

// =============================================================================
// AFV weights
// =============================================================================

/// Generate AFV quantization weights.
/// AFV (Adaptive Frequency Variable) is a hybrid transform combining:
/// - AFV-specific 4x4 weights in even rows/even columns
/// - DCT 4x4 weights in even rows/odd columns
/// - DCT 4x8 weights in odd rows
///
/// Matches jxl-rs quant_weights.rs and libjxl quant_weights.cc.
fn generate_afv_weights() -> Vec<f32> {
    let mut weights = vec![0.0f32; 192]; // 64 per channel

    // First generate the DCT4x8 and DCT4x4 weights we'll need
    let weights4x8 = generate_dct4x8_weights();
    let weights4x4 = generate_dct4x4_weights();

    const LO: f64 = 0.8517778890324296;
    const HI: f64 = 12.97166202570235 - LO + 1e-6;

    for (c, afv) in AFV_WEIGHTS.iter().enumerate() {
        let start = c * 64;

        // Build AFV distance bands
        let mut bands = [0.0f64; 4];
        bands[0] = afv[5]; // band0
        for i in 1..4 {
            bands[i] = bands[i - 1] * band_mult(afv[5 + i]);
        }

        // Position (0,0) is DC - use band interpolation at distance 0
        // DC weight = 1.0 / bands[0] (similar to other DCT strategies)
        weights[start] = (1.0 / bands[0]) as f32;

        // Positions (0,1) and (1,0): DC tendency weights
        weights[start + 1] = (1.0 / afv[0]) as f32; // position (0,1)
        weights[start + 8] = (1.0 / afv[1]) as f32; // position (1,0)

        // AFV corner positions
        weights[start + 2] = (1.0 / afv[2]) as f32; // position (0,2)
        weights[start + 16] = (1.0 / afv[3]) as f32; // position (2,0)
        weights[start + 18] = (1.0 / afv[4]) as f32; // position (2,2)

        // Other AFV positions (even rows, even columns, x>=2 or y>=2)
        for y in 0..4usize {
            for x in 0..4usize {
                if x < 2 && y < 2 {
                    continue; // Already handled above
                }
                let freq = AFV_FREQS[y * 4 + x];
                let val = interpolate_band((freq - LO) / HI * 3.0, &bands);
                weights[start + (2 * y) * 8 + (2 * x)] = (1.0 / val) as f32;
            }
        }

        // DCT 4x8 weights in odd rows (except position 8 which is DC tendency)
        for y in 0..4usize {
            for x in 0..8usize {
                if x == 0 && y == 0 {
                    continue; // Position (0,1) is DC tendency, already set
                }
                let idx4x8 = c * 64 + y * 16 + x; // Row-duplicated layout: base row y at duplicated row 2*y
                weights[start + (2 * y + 1) * 8 + x] = weights4x8[idx4x8];
            }
        }

        // DCT 4x4 weights in even rows, odd columns (except position 1 which is DC tendency)
        for y in 0..4usize {
            for x in 0..4usize {
                if x == 0 && y == 0 {
                    continue; // Position (1,0) would be at odd col, but (0,1) is already handled
                }
                let idx4x4 = c * 64 + y * 16 + x * 2; // DCT4x4 weights are in replicated format
                weights[start + (2 * y) * 8 + (2 * x + 1)] = weights4x4[idx4x4];
            }
        }
    }

    weights
}

/// AFV quantization weights (192 floats: 64 per channel).
/// All AFV variants (AFV0-AFV3) share the same weights.
static QUANT_WEIGHTS_AFV: OnceBox<Vec<f32>> = OnceBox::new();
fn quant_weights_afv() -> &'static [f32] {
    QUANT_WEIGHTS_AFV.get_or_init(|| Box::new(generate_afv_weights()))
}

/// Per-channel weight count for each strategy.
pub(super) const WEIGHT_SIZES: [usize; NUM_VALID_STRATEGIES] = [
    64, 128, 128, 256, 1024, 64, 64, 64, 64, 64, 512, 512, 64, 64, 64, 64, 4096, 2048, 2048,
];

/// Get the full quant weight table for a strategy (all 3 channels concatenated).
#[inline]
pub(super) fn quant_weights_full(strategy: usize) -> &'static [f32] {
    match strategy {
        0 => quant_weights_dct8(),
        1 | 2 => quant_weights_dct16x8(),
        3 => quant_weights_dct16x16(),
        4 => quant_weights_dct32x32(),
        5 => quant_weights_dct4x8(),
        6 => quant_weights_dct8x4(),
        7 => quant_weights_dct4x4(),
        8 => quant_weights_identity(),
        9 => quant_weights_dct2x2(),
        10 | 11 => quant_weights_dct16x32(),
        12..=15 => quant_weights_afv(),
        16 => quant_weights_dct64x64(),
        17 | 18 => quant_weights_dct32x64(),
        _ => unreachable!("Invalid strategy: {}", strategy),
    }
}

/// Get the quantization weight table for a given strategy and channel.
#[inline]
pub fn quant_weights(strategy: usize, channel: usize) -> &'static [f32] {
    debug_assert!(strategy < NUM_VALID_STRATEGIES);
    debug_assert!(channel < 3);
    let per_ch = WEIGHT_SIZES[strategy];
    let offset = channel * per_ch;
    &quant_weights_full(strategy)[offset..offset + per_ch]
}

/// Get the inverse quantization weight (1/weight) for a coefficient.
///
/// This is used during encoding to multiply coefficients before quantization.
#[inline]
#[allow(dead_code)]
pub fn inv_quant_weight(strategy: usize, channel: usize, coeff_idx: usize) -> f32 {
    let weights = quant_weights(strategy, channel);
    debug_assert!(coeff_idx < weights.len());
    1.0 / weights[coeff_idx]
}

// =============================================================================
// Precomputed dequant weight tables (reciprocals of quant_weights)
// =============================================================================

/// Generate reciprocals (1/w) for every element of a quant_weights table.
fn generate_dequant_weights(strategy: usize) -> Vec<f32> {
    let per_ch = WEIGHT_SIZES[strategy];
    let mut out = Vec::with_capacity(3 * per_ch);
    for c in 0..3 {
        let w = quant_weights(strategy, c);
        for &v in w {
            out.push(1.0 / v);
        }
    }
    out
}

static DEQUANT_WEIGHTS_DCT8: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT16X8: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT16X16: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT32X32: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT4X8: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT8X4: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT4X4: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_IDENTITY: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT2X2: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT16X32: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_AFV: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT64X64: OnceBox<Vec<f32>> = OnceBox::new();
static DEQUANT_WEIGHTS_DCT32X64: OnceBox<Vec<f32>> = OnceBox::new();

/// Get the full dequant weight table for a strategy (all 3 channels concatenated).
#[inline]
pub(super) fn dequant_weights_full(strategy: usize) -> &'static [f32] {
    match strategy {
        0 => DEQUANT_WEIGHTS_DCT8.get_or_init(|| Box::new(generate_dequant_weights(0))),
        1 | 2 => DEQUANT_WEIGHTS_DCT16X8.get_or_init(|| Box::new(generate_dequant_weights(1))),
        3 => DEQUANT_WEIGHTS_DCT16X16.get_or_init(|| Box::new(generate_dequant_weights(3))),
        4 => DEQUANT_WEIGHTS_DCT32X32.get_or_init(|| Box::new(generate_dequant_weights(4))),
        5 => DEQUANT_WEIGHTS_DCT4X8.get_or_init(|| Box::new(generate_dequant_weights(5))),
        6 => DEQUANT_WEIGHTS_DCT8X4.get_or_init(|| Box::new(generate_dequant_weights(6))),
        7 => DEQUANT_WEIGHTS_DCT4X4.get_or_init(|| Box::new(generate_dequant_weights(7))),
        8 => DEQUANT_WEIGHTS_IDENTITY.get_or_init(|| Box::new(generate_dequant_weights(8))),
        9 => DEQUANT_WEIGHTS_DCT2X2.get_or_init(|| Box::new(generate_dequant_weights(9))),
        10 | 11 => DEQUANT_WEIGHTS_DCT16X32.get_or_init(|| Box::new(generate_dequant_weights(10))),
        12..=15 => DEQUANT_WEIGHTS_AFV.get_or_init(|| Box::new(generate_dequant_weights(12))),
        16 => DEQUANT_WEIGHTS_DCT64X64.get_or_init(|| Box::new(generate_dequant_weights(16))),
        17 | 18 => DEQUANT_WEIGHTS_DCT32X64.get_or_init(|| Box::new(generate_dequant_weights(17))),
        _ => unreachable!("Invalid strategy: {}", strategy),
    }
}

/// Get the precomputed dequantization weight table (1/quant_weight) for a given
/// strategy and channel.
#[inline]
pub fn dequant_weights(strategy: usize, channel: usize) -> &'static [f32] {
    debug_assert!(strategy < NUM_VALID_STRATEGIES);
    debug_assert!(channel < 3);
    let per_ch = WEIGHT_SIZES[strategy];
    let offset = channel * per_ch;
    &dequant_weights_full(strategy)[offset..offset + per_ch]
}

/// Quantize a single coefficient.
///
/// # Arguments
/// * `coeff` - The DCT coefficient to quantize
/// * `strategy` - AC strategy (0=DCT8, 1=DCT16X8, 2=DCT8X16, etc.)
/// * `channel` - Channel index (0=X, 1=Y, 2=B)
/// * `coeff_idx` - Index of coefficient in the block
/// * `global_scale` - Global quantization scale factor
///
/// # Returns
/// Quantized integer coefficient
#[inline]
#[allow(dead_code)]
pub fn quantize_coeff(
    coeff: f32,
    strategy: usize,
    channel: usize,
    coeff_idx: usize,
    global_scale: f32,
) -> i32 {
    let weight = quant_weights(strategy, channel)[coeff_idx];
    let q = coeff * global_scale / weight;
    q.round() as i32
}

/// Dequantize a single coefficient.
///
/// # Arguments
/// * `qcoeff` - The quantized coefficient
/// * `strategy` - AC strategy (0=DCT8, 1=DCT16X8, 2=DCT8X16, etc.)
/// * `channel` - Channel index (0=X, 1=Y, 2=B)
/// * `coeff_idx` - Index of coefficient in the block
/// * `inv_global_scale` - Inverse global quantization scale factor (1/global_scale)
///
/// # Returns
/// Dequantized float coefficient
#[inline]
#[allow(dead_code)]
pub fn dequantize_coeff(
    qcoeff: i32,
    strategy: usize,
    channel: usize,
    coeff_idx: usize,
    inv_global_scale: f32,
) -> f32 {
    let weight = quant_weights(strategy, channel)[coeff_idx];
    (qcoeff as f32) * weight * inv_global_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_sizes() {
        // DCT8: 64 per channel, 192 total
        assert_eq!(quant_weights_dct8().len(), 192);
        // DCT16X8: 128 per channel, 384 total
        assert_eq!(quant_weights_dct16x8().len(), 384);
        // DCT16X16: 256 per channel, 768 total
        assert_eq!(quant_weights_dct16x16().len(), 768);
        // DCT32X32: 1024 per channel, 3072 total
        assert_eq!(quant_weights_dct32x32().len(), 3072);
    }

    #[test]
    fn test_dc_quant_inverse() {
        for c in 0..3 {
            let product = DC_QUANT[c] * INV_DC_QUANT[c];
            assert!(
                (product - 1.0).abs() < 1e-6,
                "DC_QUANT[{}] * INV_DC_QUANT[{}] = {} != 1.0",
                c,
                c,
                product
            );
        }
    }

    #[test]
    fn test_quant_weights_access() {
        // DCT8 should have 64 coefficients per channel
        for c in 0..3 {
            assert_eq!(quant_weights(0, c).len(), 64);
        }

        // DCT16X8 and DCT8X16 should have 128 coefficients per channel
        for strategy in 1..3 {
            for c in 0..3 {
                assert_eq!(quant_weights(strategy, c).len(), 128);
            }
        }

        // DCT16X16 should have 256 coefficients per channel
        for c in 0..3 {
            assert_eq!(quant_weights(3, c).len(), 256);
        }

        // DCT32X32 should have 1024 coefficients per channel
        for c in 0..3 {
            assert_eq!(quant_weights(4, c).len(), 1024);
        }

        // DCT4X8, DCT8X4, DCT4X4 should have 64 coefficients per channel
        for strategy in 5..8 {
            for c in 0..3 {
                assert_eq!(quant_weights(strategy, c).len(), 64);
            }
        }
    }

    #[test]
    fn test_all_weights_positive() {
        let strategies = [
            (0, "DCT8", 64),
            (1, "DCT16X8", 128),
            (2, "DCT8X16", 128),
            (3, "DCT16X16", 256),
            (4, "DCT32X32", 1024),
            (5, "DCT4X8", 64),
            (6, "DCT8X4", 64),
            (7, "DCT4X4", 64),
        ];

        for &(strat, name, expected_len) in &strategies {
            for c in 0..3 {
                let w = quant_weights(strat, c);
                assert_eq!(w.len(), expected_len, "{} ch={} wrong length", name, c);
                for (i, &val) in w.iter().enumerate() {
                    assert!(
                        val > 0.0,
                        "{} weight[ch={}, {}] = {} should be positive",
                        name,
                        c,
                        i,
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_dc_smallest_weight() {
        // DC weight (position 0) should be the SMALLEST for each channel.
        // Quant weights are inverse of dequant weights. DC has highest dequant
        // weight (preserve DC best), so lowest quant weight.
        let strategies = [(0, "DCT8"), (3, "DCT16X16"), (4, "DCT32X32")];

        for &(strat, name) in &strategies {
            for c in 0..3 {
                let w = quant_weights(strat, c);
                let dc = w[0];
                for (i, &val) in w.iter().enumerate().skip(1) {
                    assert!(
                        val >= dc * 0.99, // allow tiny floating point margin
                        "{} weight[ch={}, {}] = {} is less than DC = {}",
                        name,
                        c,
                        i,
                        val,
                        dc
                    );
                }
            }
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let global_scale = 1.0;
        let inv_scale = 1.0;

        // Test with a few coefficients
        let test_values = [1.0f32, -1.0, 100.0, -100.0, 0.001, -0.001];

        for &val in &test_values {
            let q = quantize_coeff(val, 0, 0, 0, global_scale);
            let dq = dequantize_coeff(q, 0, 0, 0, inv_scale);

            // Should be close to original after roundtrip (within quantization error)
            let weight = quant_weights(0, 0)[0];
            let expected_error = weight / 2.0; // Max quantization error is half a step
            assert!(
                (dq - val).abs() <= expected_error + 1e-6,
                "Roundtrip error too large: {} -> {} -> {}, weight={}",
                val,
                q,
                dq,
                weight
            );
        }
    }

    #[test]
    fn test_weight_ranges() {
        // All parametric weights should be in a reasonable range
        for strat in 0..NUM_VALID_STRATEGIES {
            for c in 0..3 {
                let w = quant_weights(strat, c);
                let min_weight = w.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_weight = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                assert!(
                    min_weight > 1e-7,
                    "strat={} ch={}: min weight {} too small",
                    strat,
                    c,
                    min_weight
                );
                assert!(
                    max_weight < 1.0,
                    "strat={} ch={}: max weight {} too large",
                    strat,
                    c,
                    max_weight
                );
            }
        }
    }

    /// Print weight statistics per strategy/channel for diagnostics.
    /// Use `cargo test -p jxl_encoder --lib test_dct4x8_position8_weight -- --nocapture` to see output.
    #[test]
    fn test_dct4x8_position8_weight() {
        // Check DCT4X8 weights, especially position 8 (DC difference)
        let channels = ["X", "Y", "B"];
        for (c, ch_name) in channels.iter().enumerate() {
            let w = quant_weights(5, c); // 5 = DCT4X8
            eprintln!(
                "DCT4X8 {}: pos0={:.6}, pos8={:.6}, ratio={:.6}",
                ch_name,
                w[0],
                w[8],
                w[0] / w[8]
            );

            // Compare with DCT8
            let dct8_w = quant_weights(0, c);
            eprintln!(
                "  DCT8 {}: pos0={:.6}, pos8={:.6}",
                ch_name, dct8_w[0], dct8_w[8]
            );
        }

        // DCT4X8 weights should be in similar range to DCT8 weights
        // (they use similar parametric formulas, just different params)
        for c in 0..3 {
            let dct4x8 = quant_weights(5, c);
            let dct8 = quant_weights(0, c);
            // Position 8 in DCT4X8 should be similar magnitude to position 0 (both are DC-like)
            let ratio = dct4x8[8] / dct8[0];
            assert!(
                (0.1..10.0).contains(&ratio),
                "DCT4X8[8] / DCT8[0] ratio for channel {} is out of range: {} ({}:{})",
                c,
                ratio,
                dct4x8[8],
                dct8[0]
            );
        }
    }

    /// Compare dequant weights at equivalent frequencies between DCT8 and DCT16x16.
    /// DCT16 position (2y, 2x) corresponds to DCT8 position (y, x).
    #[test]
    fn test_dct16_vs_dct8_equivalent_frequencies() {
        let channels = ["X", "Y", "B"];
        for (c, ch_name) in channels.iter().enumerate() {
            let w8 = quant_weights(0, c);
            let w16 = quant_weights(3, c);

            eprintln!(
                "=== Channel {} equivalent frequency dequant weights ===",
                ch_name
            );
            eprintln!(
                "{:>10} {:>12} {:>12} {:>8}",
                "DCT8(y,x)", "DCT8_dequant", "DCT16_dequant", "ratio"
            );

            for y8 in 0..8 {
                for x8 in 0..8 {
                    let idx8 = y8 * 8 + x8;
                    let y16 = y8 * 2;
                    let x16 = x8 * 2;
                    let idx16 = y16 * 16 + x16;

                    let dequant8 = 1.0 / w8[idx8];
                    let dequant16 = 1.0 / w16[idx16];
                    let ratio = dequant16 / dequant8;

                    if y8 < 4 && x8 < 4 {
                        eprintln!(
                            "  ({},{})     {:>12.2} {:>12.2} {:>8.3}",
                            y8, x8, dequant8, dequant16, ratio
                        );
                    }
                }
            }

            // Summary: average ratio for all 64 equivalent positions
            let mut total_ratio = 0.0f64;
            for y8 in 0..8 {
                for x8 in 0..8 {
                    let dequant8 = 1.0 / w8[y8 * 8 + x8] as f64;
                    let dequant16 = 1.0 / w16[y8 * 2 * 16 + x8 * 2] as f64;
                    total_ratio += dequant16 / dequant8;
                }
            }
            eprintln!(
                "  Average dequant16/dequant8 ratio: {:.4}",
                total_ratio / 64.0
            );
        }

        // Also show the unique frequencies that DCT16 has but DCT8 doesn't
        // (odd positions like (1,0), (0,1), etc.)
        eprintln!("\n=== DCT16 extra frequencies (Y channel) ===");
        let w16 = quant_weights(3, 1);
        let mut extra_dequant_sum = 0.0f64;
        let mut extra_count = 0;
        for y in 0..16 {
            for x in 0..16 {
                if y % 2 != 0 || x % 2 != 0 {
                    let dequant = 1.0 / w16[y * 16 + x] as f64;
                    extra_dequant_sum += dequant;
                    extra_count += 1;
                }
            }
        }
        let w8 = quant_weights(0, 1);
        let mut base_dequant_sum = 0.0f64;
        for y in 0..8 {
            for x in 0..8 {
                base_dequant_sum += 1.0 / w8[y * 8 + x] as f64;
            }
        }
        eprintln!(
            "DCT8 avg dequant: {:.2}, DCT16 extra freq avg dequant: {:.2}",
            base_dequant_sum / 64.0,
            extra_dequant_sum / extra_count as f64
        );
    }

    #[test]
    fn test_weight_stats_per_strategy() {
        let strategies = [
            (0, "DCT8"),
            (1, "DCT16x8"),
            (2, "DCT8x16"),
            (3, "DCT16x16"),
            (4, "DCT32x32"),
            (5, "DCT4X8"),
            (6, "DCT8X4"),
        ];
        let channels = ["X", "Y", "B"];

        for &(strat, name) in &strategies {
            for (c, ch_name) in channels.iter().enumerate() {
                let w = quant_weights(strat, c);
                let min_w = w.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_w = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean_w: f32 = w.iter().sum::<f32>() / w.len() as f32;
                // 1/weight is the quantization multiplier. Larger 1/weight = more quantization.
                let max_inv = 1.0 / min_w;
                let min_inv = 1.0 / max_w;
                eprintln!(
                    "{:>8} ch={}: {} coeffs, weight range [{:.6}, {:.6}], mean={:.6}, inv range [{:.1}, {:.1}]",
                    name,
                    ch_name,
                    w.len(),
                    min_w,
                    max_w,
                    mean_w,
                    min_inv,
                    max_inv
                );
            }
        }
    }
}

#[cfg(test)]
mod weight_debug_tests {
    use super::*;

    #[test]
    fn test_print_dct8_weights() {
        let w = quant_weights(0, 0); // DCT8, X channel
        println!("DCT8 X channel quant_weights()[0..8]:");
        for (i, &wi) in w.iter().enumerate().take(8) {
            println!("  [{}] = {:.6e}  (reciprocal = {:.6e})", i, wi, 1.0 / wi);
        }
        let w_y = quant_weights(0, 1); // DCT8, Y channel
        println!("DCT8 Y channel quant_weights()[0..8]:");
        for (i, &wi) in w_y.iter().enumerate().take(8) {
            println!("  [{}] = {:.6e}  (reciprocal = {:.6e})", i, wi, 1.0 / wi);
        }
    }
}
