// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Quantization weights and matrices for the tiny encoder.
//!
//! Ported from libjxl-tiny quant_weights.cc/h.

use super::common::DCT_BLOCK_SIZE;

/// Number of valid AC strategies in libjxl-tiny.
/// 0 = DCT (8x8), 1 = DCT16X8, 2 = DCT8X16
pub const NUM_VALID_STRATEGIES: usize = 3;

/// Inverse DC quantization constants per channel (X, Y, B).
/// These are the denominators for DC quantization.
pub const INV_DC_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];

/// DC quantization constants per channel (X, Y, B).
/// DC_QUANT[c] = 1.0 / INV_DC_QUANT[c]
pub const DC_QUANT: [f32; 3] = [
    1.0 / 4096.0, // X channel
    1.0 / 512.0,  // Y channel
    1.0 / 256.0,  // B channel
];

/// Total table size: 9 blocks of 64 coefficients each.
pub const TOTAL_TABLE_SIZE: usize = 9 * DCT_BLOCK_SIZE;

/// Size in 8x8 blocks for each (strategy, channel) combination.
/// Index = strategy * 3 + channel
/// Strategies: 0=DCT8, 1=DCT16X8, 2=DCT8X16
/// Channels: 0=X, 1=Y, 2=B
#[rustfmt::skip]
pub const TABLE_SIZE_IN_BLOCKS: [usize; 9] = [
    1, 1, 1,  // DCT8: X, Y, B
    2, 2, 2,  // DCT16X8: X, Y, B
    2, 2, 2,  // DCT8X16: X, Y, B
];

/// Offset in 8x8 blocks for each (strategy, channel) combination.
/// Index = strategy * 3 + channel
#[rustfmt::skip]
pub const TABLE_OFFSET_IN_BLOCKS: [usize; 9] = [
    0, 1, 2,  // DCT8: X, Y, B
    3, 5, 7,  // DCT16X8: X, Y, B
    3, 5, 7,  // DCT8X16: X, Y, B (shares tables with DCT16X8)
];

/// Pre-computed quantization weights.
/// 576 floats = 9 blocks * 64 coefficients per block.
///
/// Layout:
/// - Block 0 (offset 0): DCT8 X channel (64 coeffs)
/// - Block 1 (offset 64): DCT8 Y channel (64 coeffs)
/// - Block 2 (offset 128): DCT8 B channel (64 coeffs)
/// - Blocks 3-4 (offset 192): DCT16X8/DCT8X16 X channel (128 coeffs)
/// - Blocks 5-6 (offset 320): DCT16X8/DCT8X16 Y channel (128 coeffs)
/// - Blocks 7-8 (offset 448): DCT16X8/DCT8X16 B channel (128 coeffs)
#[rustfmt::skip]
pub const QUANT_WEIGHTS: [f32; TOTAL_TABLE_SIZE] = [
    // Copied directly from libjxl-tiny quant_weights.cc
    3.1746033e-04, 3.1746057e-04, 3.1854658e-04, 3.7755401e-04, 4.4749113e-04,
    5.3038419e-04, 6.2863121e-04, 7.4507861e-04, 3.1746057e-04, 3.1746062e-04,
    3.3158599e-04, 3.8811122e-04, 4.5695182e-04, 5.3938502e-04, 6.3753547e-04,
    7.5413194e-04, 3.1854658e-04, 3.3158599e-04, 3.6670428e-04, 4.1847790e-04,
    4.8487642e-04, 5.6626293e-04, 6.6427846e-04, 7.8140449e-04, 3.7755401e-04,
    3.8811122e-04, 4.1847790e-04, 4.6632939e-04, 5.3038419e-04, 6.1082945e-04,
    7.0903177e-04, 8.2727504e-04, 4.4749113e-04, 4.5695182e-04, 4.8487642e-04,
    5.3038419e-04, 5.9302151e-04, 6.7320757e-04, 7.7229418e-04, 9.4286882e-04,
    5.3038419e-04, 5.3938502e-04, 5.6626293e-04, 6.1082945e-04, 6.7320757e-04,
    7.5413194e-04, 8.5507357e-04, 1.2723245e-03, 6.2863121e-04, 6.3753547e-04,
    6.6427846e-04, 7.0903177e-04, 7.7229418e-04, 8.5507357e-04, 1.1923184e-03,
    1.7919940e-03, 7.4507861e-04, 7.5413194e-04, 7.8140449e-04, 8.2727504e-04,
    9.4286882e-04, 1.2723245e-03, 1.7919940e-03, 2.6133191e-03, 1.7857145e-03,
    1.7857157e-03, 1.7904768e-03, 2.0441783e-03, 2.3338278e-03, 2.6645192e-03,
    3.0420676e-03, 3.4731133e-03, 1.7857157e-03, 1.7857160e-03, 1.8473724e-03,
    2.0886122e-03, 2.3722122e-03, 2.6997121e-03, 3.0756146e-03, 3.5059757e-03,
    1.7904768e-03, 1.8473724e-03, 1.9982266e-03, 2.2149722e-03, 2.4845072e-03,
    2.8040458e-03, 3.1757555e-03, 3.6044519e-03, 2.0441783e-03, 2.0886122e-03,
    2.2149722e-03, 2.4100873e-03, 2.6645192e-03, 2.9746795e-03, 3.3413812e-03,
    3.7683977e-03, 2.3338278e-03, 2.3722122e-03, 2.4845072e-03, 2.6645192e-03,
    2.9068382e-03, 3.2089923e-03, 3.5716419e-03, 3.9980840e-03, 2.6645192e-03,
    2.6997121e-03, 2.8040458e-03, 2.9746795e-03, 3.2089923e-03, 3.5059757e-03,
    3.8667743e-03, 4.2947000e-03, 3.0420676e-03, 3.0756146e-03, 3.1757555e-03,
    3.3413812e-03, 3.5716419e-03, 3.8667743e-03, 4.2286036e-03, 4.6607289e-03,
    3.4731133e-03, 3.5059757e-03, 3.6044519e-03, 3.7683977e-03, 3.9980840e-03,
    4.2947000e-03, 4.6607289e-03, 5.1001739e-03, 1.9531252e-03, 3.4018266e-03,
    5.9007513e-03, 8.3743408e-03, 1.1718751e-02, 1.1718759e-02, 1.1968765e-02,
    1.6986061e-02, 3.4018266e-03, 4.2808522e-03, 6.4091417e-03, 8.8638803e-03,
    1.1718752e-02, 1.1718759e-02, 1.2320629e-02, 1.7413978e-02, 5.9007513e-03,
    6.4091417e-03, 7.8861341e-03, 1.0351914e-02, 1.1718754e-02, 1.1718762e-02,
    1.3408982e-02, 1.8736197e-02, 8.3743408e-03, 8.8638803e-03, 1.0351914e-02,
    1.1718752e-02, 1.1718759e-02, 1.1718766e-02, 1.5336527e-02, 2.1072537e-02,
    1.1718751e-02, 1.1718752e-02, 1.1718754e-02, 1.1718759e-02, 1.1718764e-02,
    1.3782934e-02, 1.8288977e-02, 2.5368163e-02, 1.1718759e-02, 1.1718759e-02,
    1.1718762e-02, 1.1718766e-02, 1.3782934e-02, 1.7413978e-02, 2.2557227e-02,
    3.4232263e-02, 1.1968765e-02, 1.2320629e-02, 1.3408982e-02, 1.5336527e-02,
    1.8288977e-02, 2.2557227e-02, 3.2079678e-02, 4.8214123e-02, 1.6986061e-02,
    1.7413978e-02, 1.8736197e-02, 2.1072537e-02, 2.5368163e-02, 3.4232263e-02,
    4.8214123e-02, 7.0312120e-02, 1.3810680e-04, 1.6047071e-04, 1.8645605e-04,
    2.1664926e-04, 2.5173181e-04, 2.9249521e-04, 3.3985957e-04, 3.9489369e-04,
    4.1871337e-04, 4.4087201e-04, 4.6420316e-04, 4.8876996e-04, 5.1463587e-04,
    5.4187077e-04, 5.7054684e-04, 6.0074159e-04, 1.9049694e-04, 1.9694651e-04,
    2.1442315e-04, 2.4016941e-04, 2.7289384e-04, 3.1245520e-04, 3.5932945e-04,
    4.0429863e-04, 4.2484730e-04, 4.4662904e-04, 4.6966935e-04, 4.9400958e-04,
    5.1969837e-04, 5.4679497e-04, 5.7536521e-04, 6.0547784e-04, 2.6276117e-04,
    2.6734054e-04, 2.8085473e-04, 3.0283103e-04, 3.3291124e-04, 3.7106971e-04,
    4.0540050e-04, 4.2322354e-04, 4.4259510e-04, 4.6344541e-04, 4.8574654e-04,
    5.0949713e-04, 5.3471868e-04, 5.6144706e-04, 5.8973121e-04, 6.1962701e-04,
    3.6243827e-04, 3.6666830e-04, 3.7935356e-04, 3.9960333e-04, 4.0956112e-04,
    4.2183659e-04, 4.3620329e-04, 4.5248115e-04, 4.7053859e-04, 4.9028790e-04,
    5.1167433e-04, 5.3467450e-04, 5.5928703e-04, 5.8552966e-04, 6.1343284e-04,
    6.4304151e-04, 4.3123538e-04, 4.3253010e-04, 4.3638589e-04, 4.4272337e-04,
    4.5142765e-04, 4.6236772e-04, 4.7541404e-04, 4.9045112e-04, 5.0738233e-04,
    5.2613625e-04, 5.4666388e-04, 5.6893763e-04, 5.9295003e-04, 6.1870721e-04,
    6.4623309e-04, 6.7556335e-04, 4.8162136e-04, 4.8277923e-04, 4.8623976e-04,
    4.9196521e-04, 4.9989921e-04, 5.0997391e-04, 5.2211789e-04, 5.3626322e-04,
    5.5235048e-04, 5.7033246e-04, 5.9017621e-04, 6.1186077e-04, 6.3538179e-04,
    6.6074729e-04, 6.8797835e-04, 7.5214857e-04, 5.3789350e-04, 5.3897168e-04,
    5.4219965e-04, 5.4755906e-04, 5.5502116e-04, 5.6455150e-04, 5.7611306e-04,
    5.8966759e-04, 6.0518348e-04, 6.2263483e-04, 6.4200454e-04, 6.6328526e-04,
    6.8647927e-04, 7.3936180e-04, 8.0337300e-04, 8.7534159e-04, 6.0074159e-04,
    6.0177385e-04, 6.0486794e-04, 6.1001495e-04, 6.1720144e-04, 6.2641077e-04,
    6.3762552e-04, 6.5082888e-04, 6.6600717e-04, 6.8315107e-04, 7.1794738e-04,
    7.6673855e-04, 8.2213664e-04, 8.8475435e-04, 9.5529074e-04, 1.0345384e-03,
    6.9053401e-04, 7.7444571e-04, 8.6855399e-04, 9.7409816e-04, 1.0924696e-03,
    1.2252233e-03, 1.3741088e-03, 1.5410866e-03, 1.7283577e-03, 1.9383827e-03,
    2.1739292e-03, 2.3783136e-03, 2.5041751e-03, 2.6366978e-03, 2.7762330e-03,
    2.9231580e-03, 8.8290084e-04, 9.0565201e-04, 9.6644071e-04, 1.0539154e-03,
    1.1619731e-03, 1.2886107e-03, 1.4338633e-03, 1.5988132e-03, 1.7851711e-03,
    1.9951246e-03, 2.2312698e-03, 2.4038092e-03, 2.5288085e-03, 2.6606584e-03,
    2.7996788e-03, 2.9462043e-03, 1.1288587e-03, 1.1438611e-03, 1.1877866e-03,
    1.2581701e-03, 1.3525900e-03, 1.4695247e-03, 1.6085195e-03, 1.7700332e-03,
    1.9552717e-03, 2.1660449e-03, 2.3636019e-03, 2.4791702e-03, 2.6018962e-03,
    2.7319542e-03, 2.8695827e-03, 3.0150530e-03, 1.4433329e-03, 1.4561869e-03,
    1.4945272e-03, 1.5578135e-03, 1.6454635e-03, 1.7571596e-03, 1.8930284e-03,
    2.0537286e-03, 2.2404641e-03, 2.3856999e-03, 2.4897642e-03, 2.6016813e-03,
    2.7214440e-03, 2.8491386e-03, 2.9849128e-03, 3.1289863e-03, 1.8454153e-03,
    1.8577600e-03, 1.8947916e-03, 1.9565322e-03, 2.0431101e-03, 2.1548597e-03,
    2.2924175e-03, 2.3864941e-03, 2.4688798e-03, 2.5601350e-03, 2.6600207e-03,
    2.7684029e-03, 2.8852450e-03, 3.0105773e-03, 3.1445161e-03, 3.2872346e-03,
    2.3435291e-03, 2.3491632e-03, 2.3660017e-03, 2.3938618e-03, 2.4324679e-03,
    2.4814904e-03, 2.5405819e-03, 2.6094120e-03, 2.6876912e-03, 2.7751899e-03,
    2.8717481e-03, 2.9772632e-03, 3.0917146e-03, 3.2151414e-03, 3.3476453e-03,
    3.4893905e-03, 2.6173447e-03, 2.6225911e-03, 2.6382981e-03, 2.6643763e-03,
    2.7006865e-03, 2.7470603e-03, 2.8033180e-03, 2.8692731e-03, 2.9447721e-03,
    3.0296890e-03, 3.1239407e-03, 3.2274905e-03, 3.3403505e-03, 3.4625907e-03,
    3.5943130e-03, 3.7356857e-03, 2.9231580e-03, 2.9281813e-03, 2.9432368e-03,
    2.9682817e-03, 3.0032503e-03, 3.0480621e-03, 3.1026325e-03, 3.1668788e-03,
    3.2407353e-03, 3.3241559e-03, 3.4171303e-03, 3.5196650e-03, 3.6318223e-03,
    3.7536959e-03, 3.8854245e-03, 4.0271855e-03, 1.9729543e-03, 2.5272998e-03,
    3.2374004e-03, 4.1470206e-03, 4.8498721e-03, 5.1065302e-03, 5.3767711e-03,
    5.6613120e-03, 6.3208523e-03, 7.0889443e-03, 7.9503711e-03, 8.9164926e-03,
    9.9999988e-03, 1.1215170e-02, 1.2578006e-02, 1.5967883e-02, 3.3539708e-03,
    3.5433732e-03, 4.0769530e-03, 4.7721486e-03, 4.9862624e-03, 5.2236770e-03,
    5.4806760e-03, 5.8470899e-03, 6.5286267e-03, 7.2964565e-03, 8.1600742e-03,
    9.1304630e-03, 1.0220082e-02, 1.1443083e-02, 1.2854187e-02, 1.6610704e-02,
    4.9218577e-03, 4.9511627e-03, 5.0357706e-03, 5.1678251e-03, 5.3387447e-03,
    5.5415547e-03, 5.8825868e-03, 6.4732647e-03, 7.1507092e-03, 7.9215374e-03,
    8.7942975e-03, 9.7792931e-03, 1.0888627e-02, 1.2136214e-02, 1.4550334e-02,
    1.8655479e-02, 5.4969229e-03, 5.5188821e-03, 5.5837538e-03, 5.6971479e-03,
    6.0176966e-03, 6.4261849e-03, 6.9230762e-03, 7.5107799e-03, 8.1936987e-03,
    8.9781955e-03, 9.8724691e-03, 1.0886625e-02, 1.2032620e-02, 1.4036770e-02,
    1.7736901e-02, 2.2478340e-02, 6.7489492e-03, 6.7940955e-03, 6.9295247e-03,
    7.1553192e-03, 7.4719470e-03, 7.8806318e-03, 8.3836997e-03, 8.9848433e-03,
    9.6892491e-03, 1.0503774e-02, 1.1436986e-02, 1.2499244e-02, 1.4953869e-02,
    1.8516723e-02, 2.3044668e-02, 2.8803868e-02, 8.6290650e-03, 8.6752698e-03,
    8.8141672e-03, 9.0466458e-03, 9.3743140e-03, 9.7996546e-03, 1.0326200e-02,
    1.0958694e-02, 1.1703256e-02, 1.2567491e-02, 1.4605597e-02, 1.7509630e-02,
    2.1164555e-02, 2.5766177e-02, 3.1564441e-02, 4.4291977e-02, 1.1032925e-02,
    1.1082166e-02, 1.1230315e-02, 1.1478675e-02, 1.1829470e-02, 1.2285955e-02,
    1.2938387e-02, 1.4542448e-02, 1.6570158e-02, 1.9115077e-02, 2.2296766e-02,
    2.6267400e-02, 3.1220267e-02, 4.1523870e-02, 5.6756895e-02, 7.8389272e-02,
    1.5967883e-02, 1.6106272e-02, 1.6526788e-02, 1.7245775e-02, 1.8291343e-02,
    1.9704822e-02, 2.1542856e-02, 2.3880199e-02, 2.6813647e-02, 3.0466938e-02,
    3.7175436e-02, 4.7613274e-02, 6.1909460e-02, 8.1609353e-02, 1.0892317e-01,
    1.4702357e-01,
];

/// Get the quantization weight table for a given strategy and channel.
///
/// # Arguments
/// * `strategy` - AC strategy (0=DCT8, 1=DCT16X8, 2=DCT8X16)
/// * `channel` - Channel index (0=X, 1=Y, 2=B)
///
/// # Returns
/// Slice of quantization weights for the strategy/channel combination.
#[inline]
pub fn quant_weights(strategy: usize, channel: usize) -> &'static [f32] {
    debug_assert!(strategy < NUM_VALID_STRATEGIES);
    debug_assert!(channel < 3);

    let idx = strategy * 3 + channel;
    let offset = TABLE_OFFSET_IN_BLOCKS[idx] * DCT_BLOCK_SIZE;
    let size = TABLE_SIZE_IN_BLOCKS[idx] * DCT_BLOCK_SIZE;

    &QUANT_WEIGHTS[offset..offset + size]
}

/// Get the inverse quantization weight (1/weight) for a coefficient.
///
/// This is used during encoding to multiply coefficients before quantization.
#[inline]
pub fn inv_quant_weight(strategy: usize, channel: usize, coeff_idx: usize) -> f32 {
    let weights = quant_weights(strategy, channel);
    debug_assert!(coeff_idx < weights.len());
    1.0 / weights[coeff_idx]
}

/// Quantize a single coefficient.
///
/// # Arguments
/// * `coeff` - The DCT coefficient to quantize
/// * `strategy` - AC strategy (0=DCT8, 1=DCT16X8, 2=DCT8X16)
/// * `channel` - Channel index (0=X, 1=Y, 2=B)
/// * `coeff_idx` - Index of coefficient in the block
/// * `global_scale` - Global quantization scale factor
///
/// # Returns
/// Quantized integer coefficient
#[inline]
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
/// * `strategy` - AC strategy (0=DCT8, 1=DCT16X8, 2=DCT8X16)
/// * `channel` - Channel index (0=X, 1=Y, 2=B)
/// * `coeff_idx` - Index of coefficient in the block
/// * `inv_global_scale` - Inverse global quantization scale factor (1/global_scale)
///
/// # Returns
/// Dequantized float coefficient
#[inline]
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
        // Verify total size matches
        assert_eq!(QUANT_WEIGHTS.len(), TOTAL_TABLE_SIZE);
        assert_eq!(TOTAL_TABLE_SIZE, 9 * 64);
        assert_eq!(TOTAL_TABLE_SIZE, 576);
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
    }

    #[test]
    fn test_quant_weights_positive() {
        // All weights should be positive
        for &w in &QUANT_WEIGHTS {
            assert!(w > 0.0, "Quantization weight {} should be positive", w);
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
            let weight = QUANT_WEIGHTS[0];
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
        // Weights should be in reasonable range (from examining the data)
        let min_weight = QUANT_WEIGHTS.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_weight = QUANT_WEIGHTS
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            min_weight > 1e-5,
            "Min weight {} too small",
            min_weight
        );
        assert!(max_weight < 1.0, "Max weight {} too large", max_weight);
    }
}
