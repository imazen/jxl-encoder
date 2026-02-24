// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Gaborish inverse pre-filter for the encoder.
//!
//! Applies a 5x5 symmetric sharpening kernel to XYB channels before DCT.
//! The decoder applies a 3x3 Gabor-like blur; this encoder-side inverse
//! compensates, reducing blocking artifacts and improving rate-distortion.
//!
//! Ported from libjxl `lib/jxl/enc_gaborish.cc`.

/// Butteraugli-optimized 5x5 symmetric kernel weights.
///
/// These are NOT the mathematical inverse of the decoder's 3x3 blur — they
/// were optimized by butteraugli for favorable rate-distortion tradeoffs.
///
/// Kernel layout (lower-right quadrant):
/// ```text
///   c  r  R
///   r  d  L
///   R  L  D
/// ```
/// where:
///   r = kGaborish[0] (orthogonal distance 1)
///   d = kGaborish[1] (diagonal distance sqrt(2))
///   R = kGaborish[2] (orthogonal distance 2)
///   L = kGaborish[3] (knight's move distance)
///   D = kGaborish[4] (corner distance 2*sqrt(2))
const K_GABORISH: [f64; 5] = [
    -0.09495815671340026,   // [0] r: orthogonal dist 1
    -0.041031725066768575,  // [1] d: diagonal dist sqrt(2)
    0.013710004822696948,   // [2] R: orthogonal dist 2
    0.006510206083837737,   // [3] L: knight's move
    -0.0014789063378272242, // [4] D: corner dist 2*sqrt(2)
];

/// Compute normalized weights for one channel.
///
/// Returns `(center_weight, r, d, big_r, l, big_d)` all as f32.
fn compute_weights(mul: f64) -> (f32, f32, f32, f32, f32, f32) {
    let sum = 1.0
        + mul
            * 4.0
            * (K_GABORISH[0] + K_GABORISH[1] + K_GABORISH[2] + K_GABORISH[4] + 2.0 * K_GABORISH[3]);
    let sum = if sum < 1e-5 { 1e-5 } else { sum };
    let normalize = 1.0 / sum;
    let normalize_mul = mul * normalize;

    (
        normalize as f32,                       // center
        (normalize_mul * K_GABORISH[0]) as f32, // r
        (normalize_mul * K_GABORISH[1]) as f32, // d
        (normalize_mul * K_GABORISH[2]) as f32, // R
        (normalize_mul * K_GABORISH[3]) as f32, // L
        (normalize_mul * K_GABORISH[4]) as f32, // D
    )
}

/// Apply the gaborish inverse (5x5 sharpening) to one channel in-place.
///
/// Uses a scratch buffer to avoid reading already-modified values.
/// Boundary handling: clamp coordinates to [0, dim-1] (edge replication).
/// Dispatches to SIMD-accelerated implementation via jxl_simd.
fn apply_channel(data: &mut [f32], scratch: &mut [f32], width: usize, height: usize, mul: f64) {
    let (wc, wr, wd, w_big_r, wl, w_big_d) = compute_weights(mul);
    jxl_simd::gaborish_5x5_channel(
        data, scratch, width, height, wc, wr, wd, w_big_r, wl, w_big_d,
    );
}

/// Apply gaborish inverse sharpening to all three XYB channels.
///
/// This should be called AFTER noise estimation/denoising and BEFORE
/// adaptive quantization, matching the libjxl pipeline order.
///
/// Uses `mul=[1.0, 1.0, 1.0]` for all channels (libjxl VarDCT default).
pub fn gaborish_inverse(
    xyb_x: &mut [f32],
    xyb_y: &mut [f32],
    xyb_b: &mut [f32],
    width: usize,
    height: usize,
) {
    // Reuse one scratch buffer across all 3 channels to avoid 3 allocations
    let mut scratch = jxl_simd::vec_f32_dirty(width * height);

    // mul=1.0 for all channels, matching libjxl enc_heuristics.cc line 1137-1140
    apply_channel(xyb_x, &mut scratch, width, height, 1.0);
    apply_channel(xyb_y, &mut scratch, width, height, 1.0);
    apply_channel(xyb_b, &mut scratch, width, height, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_normalization() {
        // With mul=1.0, the weights should sum to 1.0
        let (wc, wr, wd, w_big_r, wl, w_big_d) = compute_weights(1.0);
        // center: 1 weight
        // r: 4 weights
        // d: 4 weights
        // R: 4 weights
        // L: 8 weights
        // D: 4 weights
        let sum = wc + 4.0 * wr + 4.0 * wd + 4.0 * w_big_r + 8.0 * wl + 4.0 * w_big_d;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Kernel weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_uniform_image_preserved() {
        // A constant-value image should be unchanged after gaborish inverse
        let width = 16;
        let height = 16;
        let value = 0.5f32;
        let mut data = vec![value; width * height];
        let mut scratch = vec![0.0f32; width * height];
        apply_channel(&mut data, &mut scratch, width, height, 1.0);

        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - value).abs() < 1e-5,
                "Pixel {} changed from {} to {} on uniform image",
                i,
                value,
                v
            );
        }
    }

    #[test]
    fn test_sharpening_effect() {
        // A bright center pixel surrounded by dark pixels should get brighter
        // (sharpening increases contrast)
        let width = 8;
        let height = 8;
        let mut data = vec![0.0f32; width * height];
        // Set center pixel bright
        data[4 * width + 4] = 1.0;
        let original_center = data[4 * width + 4];

        let mut scratch = vec![0.0f32; width * height];
        apply_channel(&mut data, &mut scratch, width, height, 1.0);

        // Center should still be the brightest (sharpening increases it relative to neighbors)
        let new_center = data[4 * width + 4];
        // The center weight is > 1.0 (normalizing with negative neighbor weights),
        // so the center pixel should increase
        assert!(
            new_center > original_center,
            "Sharpening should increase isolated bright pixel: {} -> {}",
            original_center,
            new_center
        );

        // Neighbors should become negative (ringing from sharpening)
        let neighbor = data[4 * width + 3];
        assert!(
            neighbor < 0.0,
            "Sharpening should create negative ringing at neighbors: got {}",
            neighbor
        );
    }
}
