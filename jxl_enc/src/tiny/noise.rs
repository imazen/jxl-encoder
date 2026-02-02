// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Noise estimation and encoding for JPEG XL.
//!
//! Ported from libjxl `enc_noise.cc` and `enc_optimize.h`.
//!
//! The encoder estimates noise parameters from the XYB image by:
//! 1. Dividing the image into 8×8 patches
//! 2. Computing SAD (Sum of Absolute Differences) to identify flat patches
//! 3. For flat patches, computing mean intensity and noise level via Laplacian filter
//! 4. Fitting an 8-point LUT using Scaled Conjugate Gradient optimization
//!
//! The resulting LUT is encoded as 8 × 10-bit values in the bitstream.
//! The decoder uses these parameters to synthesize noise during rendering.

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Number of points in the noise LUT.
pub const NUM_NOISE_POINTS: usize = 8;

/// Precision for noise parameter encoding (10 bits → 1024).
const NOISE_PRECISION: f32 = 1024.0;

/// Maximum encodable noise value.
const NOISE_LUT_MAX: f32 = 1023.4999 / NOISE_PRECISION;

/// Noise parameters: an 8-point lookup table mapping intensity to noise level.
#[derive(Debug, Clone)]
pub struct NoiseParams {
    /// LUT mapping intensity to noise level. Index is derived from pixel intensity.
    pub lut: [f32; NUM_NOISE_POINTS],
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            lut: [0.0; NUM_NOISE_POINTS],
        }
    }
}

impl NoiseParams {
    /// Returns true if any noise parameter is significant (> 1e-3).
    pub fn has_any(&self) -> bool {
        self.lut.iter().any(|&v| v.abs() > 1e-3)
    }

    /// Clear all noise parameters to zero.
    fn clear(&mut self) {
        self.lut = [0.0; NUM_NOISE_POINTS];
    }
}

/// Write noise parameters to the bitstream.
///
/// Each of the 8 LUT values is encoded as 10 bits: `round(value * 1024)`.
pub fn write_noise_params(params: &NoiseParams, writer: &mut BitWriter) -> Result<()> {
    for &val in &params.lut {
        let quantized = (val * NOISE_PRECISION).round() as u64;
        debug_assert!(
            quantized < 1024,
            "noise param {} too large (quantized={})",
            val,
            quantized
        );
        writer.write(10, quantized.min(1023))?;
    }
    Ok(())
}

// ── Noise estimation ──

/// A single noise measurement: intensity and noise level at a flat patch.
struct NoiseLevel {
    intensity: f32,
    noise_level: f32,
}

/// Compute the SAD (Sum of Absolute Differences) score for one 8×8 patch.
///
/// Uses a sliding 3×4 sub-block compared against a center reference patch
/// at offset (2,2). Returns the average of the smallest half of SAD values
/// (ROAD-like robust estimator).
fn get_score_sad(
    xyb_x: &[f32],
    xyb_y: &[f32],
    width: usize,
    x: usize,
    y: usize,
    block_size: usize,
) -> f32 {
    let small_bl_size_x = 3;
    let small_bl_size_y = 4;
    let num_sad = (block_size - small_bl_size_x) * (block_size - small_bl_size_y);
    let offset = 2;

    let mut sad = Vec::with_capacity(num_sad);

    for y_bl in 0..(block_size - small_bl_size_y) {
        for x_bl in 0..(block_size - small_bl_size_x) {
            let mut sad_sum = 0.0f32;
            for cy in 0..small_bl_size_y {
                for cx in 0..small_bl_size_x {
                    let wnd_idx = (y + y_bl + cy) * width + (x + x_bl + cx);
                    let center_idx = (y + offset + cy) * width + (x + offset + cx);
                    let wnd = 0.5 * (xyb_y[wnd_idx] + xyb_x[wnd_idx]);
                    let center = 0.5 * (xyb_y[center_idx] + xyb_x[center_idx]);
                    sad_sum += (center - wnd).abs();
                }
            }
            sad.push(sad_sum);
        }
    }

    let k_samples = num_sad / 2;
    sad.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let total: f32 = sad[..k_samples].iter().sum();
    total / k_samples as f32
}

/// Simple histogram with 256 bins for SAD score analysis.
struct SadHistogram {
    bins: [u32; 256],
}

impl SadHistogram {
    fn new() -> Self {
        Self { bins: [0; 256] }
    }

    fn increment(&mut self, x: f32) {
        let idx = (x as usize).min(255);
        self.bins[idx] += 1;
    }

    /// Find the mode (bin with highest count).
    fn mode(&self) -> usize {
        let mut max_idx = 0;
        for i in 1..256 {
            if self.bins[i] > self.bins[max_idx] {
                max_idx = i;
            }
        }
        max_idx
    }
}

/// Compute SAD scores for all 8×8 patches in the image.
fn get_sad_scores(
    xyb_x: &[f32],
    xyb_y: &[f32],
    width: usize,
    height: usize,
    block_s: usize,
) -> (Vec<f32>, SadHistogram) {
    let num_bin = 256;
    let patches_x = width / block_s;
    let patches_y = height / block_s;
    let mut scores = Vec::with_capacity(patches_x * patches_y);
    let mut histogram = SadHistogram::new();

    for y in (0..height).step_by(block_s) {
        if y + block_s > height {
            break;
        }
        for x in (0..width).step_by(block_s) {
            if x + block_s > width {
                break;
            }
            let sad = get_score_sad(xyb_x, xyb_y, width, x, y, block_s);
            scores.push(sad);
            histogram.increment(sad * num_bin as f32);
        }
    }

    (scores, histogram)
}

/// Get the SAD threshold from histogram. Patches below this threshold are "flat".
fn get_sad_threshold(histogram: &SadHistogram) -> f32 {
    let mode = histogram.mode();
    mode as f32 / 256.0
}

/// Compute intensity index and fractional part for LUT interpolation.
fn index_and_frac(x: f32) -> (usize, f32) {
    let k_scale_numerator = (NUM_NOISE_POINTS - 2) as f32;
    let scaled_x = (x * k_scale_numerator).max(0.0);
    let floor_x = scaled_x.floor();
    let frac_x = scaled_x - floor_x;
    if scaled_x >= k_scale_numerator + 1.0 {
        (k_scale_numerator as usize, 1.0)
    } else {
        (floor_x as usize, frac_x)
    }
}

/// Compute noise levels for all flat patches in the image.
///
/// For each patch below the SAD threshold, computes:
/// - Mean intensity: average of 0.5*(X+Y) over the patch
/// - Noise level: average absolute Laplacian response
fn get_noise_levels(
    xyb_x: &[f32],
    xyb_y: &[f32],
    width: usize,
    height: usize,
    sad_scores: &[f32],
    threshold: f32,
    block_s: usize,
) -> Vec<NoiseLevel> {
    let mut noise_levels = Vec::new();

    // 3×3 Laplacian filter
    let lapl_filter: [[f32; 3]; 3] = [
        [-0.25, -1.0, -0.25],
        [-1.0, 5.0, -1.0],
        [-0.25, -1.0, -0.25],
    ];

    let mut patch_index = 0;

    for y in (0..height).step_by(block_s) {
        if y + block_s > height {
            break;
        }
        for x in (0..width).step_by(block_s) {
            if x + block_s > width {
                break;
            }
            if sad_scores[patch_index] <= threshold {
                // Mean intensity over the patch
                let mut mean_int = 0.0f32;
                for y_bl in 0..block_s {
                    for x_bl in 0..block_s {
                        let idx = (y + y_bl) * width + (x + x_bl);
                        mean_int += 0.5 * (xyb_y[idx] + xyb_x[idx]);
                    }
                }
                mean_int /= (block_s * block_s) as f32;

                // Noise level via Laplacian filter with mirror boundary
                let mut noise_level = 0.0f32;
                let mut count = 0usize;
                for y_bl in 0..block_s {
                    for x_bl in 0..block_s {
                        let mut filtered = 0.0f32;
                        for y_f in -1i32..=1 {
                            for x_f in -1i32..=1 {
                                // Mirror boundary within the patch
                                let sy = if (y_bl as i32 + y_f) >= 0
                                    && (y_bl as i32 + y_f) < block_s as i32
                                {
                                    y_bl as i32 + y_f
                                } else {
                                    y_bl as i32 - y_f
                                };
                                let sx = if (x_bl as i32 + x_f) >= 0
                                    && (x_bl as i32 + x_f) < block_s as i32
                                {
                                    x_bl as i32 + x_f
                                } else {
                                    x_bl as i32 - x_f
                                };
                                let idx = (y + sy as usize) * width + (x + sx as usize);
                                filtered += 0.5
                                    * (xyb_y[idx] + xyb_x[idx])
                                    * lapl_filter[(y_f + 1) as usize][(x_f + 1) as usize];
                            }
                        }
                        noise_level += filtered.abs();
                        count += 1;
                    }
                }
                noise_level /= count as f32;

                noise_levels.push(NoiseLevel {
                    intensity: mean_int,
                    noise_level,
                });
            }
            patch_index += 1;
        }
    }

    noise_levels
}

// ── Scaled Conjugate Gradient optimizer ──
// Ported from libjxl `enc_optimize.h`.

/// Fixed-size array for optimization (8 elements for noise LUT).
type OptArray = [f64; NUM_NOISE_POINTS];

fn arr_add(a: &OptArray, b: &OptArray) -> OptArray {
    let mut r = [0.0; NUM_NOISE_POINTS];
    for i in 0..NUM_NOISE_POINTS {
        r[i] = a[i] + b[i];
    }
    r
}

fn arr_sub(a: &OptArray, b: &OptArray) -> OptArray {
    let mut r = [0.0; NUM_NOISE_POINTS];
    for i in 0..NUM_NOISE_POINTS {
        r[i] = a[i] - b[i];
    }
    r
}

fn arr_scale(s: f64, a: &OptArray) -> OptArray {
    let mut r = [0.0; NUM_NOISE_POINTS];
    for i in 0..NUM_NOISE_POINTS {
        r[i] = s * a[i];
    }
    r
}

fn arr_dot(a: &OptArray, b: &OptArray) -> f64 {
    let mut r = 0.0;
    for i in 0..NUM_NOISE_POINTS {
        r += a[i] * b[i];
    }
    r
}

/// Loss function for noise parameter optimization.
///
/// loss = sum asym * (F(x) - nl)^2 + kReg * num_points * sum (w[i] - w[i+1])^2
/// where asym = 1 if F(x) < nl, kAsym if F(x) > nl.
struct NoiseLossFunction {
    nl: Vec<NoiseLevel>,
}

impl NoiseLossFunction {
    fn new(nl: Vec<NoiseLevel>) -> Self {
        Self { nl }
    }

    /// Compute loss and negative gradient at point w.
    fn compute(&self, w: &OptArray, df: &mut OptArray, skip_regularization: bool) -> f64 {
        const K_REG: f64 = 0.005;
        const K_ASYM: f64 = 1.1;

        let mut loss = 0.0;
        *df = [0.0; NUM_NOISE_POINTS];

        for sample in &self.nl {
            let (pos, frac) = index_and_frac(sample.intensity);
            debug_assert!(pos < NUM_NOISE_POINTS - 1);
            let low = w[pos];
            let hi = w[pos + 1];
            let val = low * (1.0 - frac as f64) + hi * frac as f64;
            let dist = val - sample.noise_level as f64;
            if dist > 0.0 {
                loss += K_ASYM * dist * dist;
                df[pos] -= K_ASYM * (1.0 - frac as f64) * dist;
                df[pos + 1] -= K_ASYM * frac as f64 * dist;
            } else {
                loss += dist * dist;
                df[pos] -= (1.0 - frac as f64) * dist;
                df[pos + 1] -= frac as f64 * dist;
            }
        }

        if skip_regularization {
            return loss;
        }

        let n = self.nl.len() as f64;
        for i in 0..(NUM_NOISE_POINTS - 1) {
            let diff = w[i] - w[i + 1];
            loss += K_REG * n * diff * diff;
            df[i] -= K_REG * diff * n;
            df[i + 1] += K_REG * diff * n;
        }

        loss
    }
}

/// Scaled Conjugate Gradient optimization.
///
/// Ported from libjxl `OptimizeWithScaledConjugateGradientMethod` (enc_optimize.h).
/// Minimizes the loss function starting from `w0`.
///
/// The control flow exactly matches the C++ reference: when `success` is false,
/// the `m`/`psq`/`s`/`t` variables retain their values from the previous iteration.
fn optimize_scg(
    loss_fn: &NoiseLossFunction,
    w0: &OptArray,
    precision: f64,
    max_iters: usize,
) -> OptArray {
    let n = NUM_NOISE_POINTS;
    let rsq_threshold = precision * precision;
    let sigma0 = 0.0001f64;
    let l_min = 1.0e-15f64;
    let l_max = 1.0e15f64;

    let mut w = *w0;
    let mut r = [0.0; NUM_NOISE_POINTS];
    let mut rt = [0.0; NUM_NOISE_POINTS];
    let mut e;

    let mut fw = loss_fn.compute(&w, &mut r, false);
    let mut _rsq = arr_dot(&r, &r);
    e = r;
    let mut p = r;
    let mut l = 1.0f64;
    let mut success = true;
    let mut n_success = 0usize;
    let mut k = 0usize;

    // These persist across iterations (C++ declares them outside the loop)
    let mut m = 0.0f64;
    let mut psq = 0.0f64;
    #[allow(unused_assignments)]
    let mut s;
    let mut t = 0.0f64;

    while k < max_iters {
        k += 1;

        if success {
            m = -arr_dot(&p, &r);
            if m >= 0.0 {
                p = r;
                m = -arr_dot(&p, &r);
            }
            psq = arr_dot(&p, &p);
            s = sigma0 / psq.sqrt();
            let w_plus_sp = arr_add(&w, &arr_scale(s, &p));
            loss_fn.compute(&w_plus_sp, &mut rt, false);
            t = arr_dot(&p, &arr_sub(&r, &rt)) / s;
        }

        let mut d = t + l * psq;
        if d <= 0.0 {
            d = l * psq;
            l -= t / psq;
        }

        let a = -m / d;
        let wp = arr_add(&w, &arr_scale(a, &p));
        let fp = loss_fn.compute(&wp, &mut rt, false);

        let big_d = 2.0 * (fp - fw) / (a * m);
        if big_d >= 0.0 {
            success = true;
            n_success += 1;
            w = wp;
        } else {
            success = false;
        }

        if success {
            e = r;
            r = rt;
            _rsq = arr_dot(&r, &r);
            fw = fp;
            if _rsq <= rsq_threshold {
                break;
            }
        }

        if big_d < 0.25 {
            l = (4.0 * l).min(l_max);
        } else if big_d > 0.75 {
            l = (0.25 * l).max(l_min);
        }

        if n_success.is_multiple_of(n) {
            p = r;
            l = 1.0;
        } else if success {
            let b = arr_dot(&arr_sub(&e, &r), &r) / m;
            p = arr_add(&arr_scale(b, &p), &r);
        }
    }

    w
}

/// Optimize noise parameters from collected noise level measurements.
fn optimize_noise_params(noise_levels: &[NoiseLevel], params: &mut NoiseParams, mul: f32) {
    const MAX_ERROR: f64 = 1e-3;
    const PRECISION: f64 = 1e-8;
    const MAX_ITER: usize = 40;

    let avg: f32 =
        noise_levels.iter().map(|nl| nl.noise_level).sum::<f32>() / noise_levels.len() as f32;

    let loss_fn = NoiseLossFunction::new(
        noise_levels
            .iter()
            .map(|nl| NoiseLevel {
                intensity: nl.intensity,
                noise_level: nl.noise_level,
            })
            .collect(),
    );

    let mut w = [avg as f64; NUM_NOISE_POINTS];
    w = optimize_scg(&loss_fn, &w, PRECISION, MAX_ITER);

    // Clamp to codestream limits, apply quality multiplier
    for v in w.iter_mut() {
        *v = (*v * mul as f64).clamp(0.0, NOISE_LUT_MAX as f64);
    }

    // Check approximation quality
    let mut unused = [0.0; NUM_NOISE_POINTS];
    let loss = loss_fn.compute(&w, &mut unused, true) / noise_levels.len() as f64;

    if loss > MAX_ERROR {
        // Approximation too poor: no noise
        params.clear();
        return;
    }

    for (i, &val) in w.iter().enumerate() {
        params.lut[i] = val as f32;
    }
}

/// Estimate noise parameters from an XYB image.
///
/// Returns `Some(NoiseParams)` if noise is detected, `None` if the image is too
/// textured or has no detectable noise.
///
/// The `quality_coef` is derived from the encoding distance:
/// - At d=1.0: quality_coef ≈ 0.25
/// - Ramps to 1.0 at d≥1.6
///
/// This matches libjxl's approach of reducing noise synthesis at lower distances.
pub fn estimate_noise_params(
    xyb_x: &[f32],
    xyb_y: &[f32],
    _xyb_b: &[f32],
    width: usize,
    height: usize,
    quality_coef: f32,
) -> Option<NoiseParams> {
    let block_s = 8;

    // Need at least one full block
    if width < block_s || height < block_s {
        return None;
    }

    let (sad_scores, sad_histogram) = get_sad_scores(xyb_x, xyb_y, width, height, block_s);

    let sad_threshold = get_sad_threshold(&sad_histogram);

    // If threshold is too large, image has strong texture that would fool the model.
    // If zero or negative, no flat patches found.
    if sad_threshold > 0.15 || sad_threshold <= 0.0 {
        return None;
    }

    let noise_levels = get_noise_levels(
        xyb_x,
        xyb_y,
        width,
        height,
        &sad_scores,
        sad_threshold,
        block_s,
    );

    if noise_levels.is_empty() {
        return None;
    }

    let mut params = NoiseParams::default();
    optimize_noise_params(&noise_levels, &mut params, quality_coef * 1.4);

    if params.has_any() { Some(params) } else { None }
}

/// Compute the quality coefficient for noise synthesis from encoding distance.
///
/// Matches libjxl's `enc_frame.cc` lines 696-709 exactly:
/// - d < 1.0: quality_coef = 1.0 (full noise)
/// - d = 1.0: quality_coef = 0.25
/// - d in (1.0, 1.6): linear ramp from 0.25 to 1.0
/// - d >= 1.6: quality_coef = 1.0 (full noise)
pub fn noise_quality_coef(distance: f32) -> f32 {
    const RAMPUP_START: f32 = 1.0;
    const RAMPUP_RANGE: f32 = 0.6;
    const LEVEL_AT_START: f32 = 0.25;

    let rampup = (distance - RAMPUP_START) / RAMPUP_RANGE;
    if rampup < 0.0 {
        // Below d=1.0: full noise (matches kNoiseRampupStart = 1.0)
        1.0
    } else if rampup < 1.0 {
        // Ramp from 0.25 to 1.0
        LEVEL_AT_START + (1.0 - LEVEL_AT_START) * rampup
    } else {
        // Above d=1.6: full noise
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_params_write_roundtrip() {
        let params = NoiseParams {
            lut: [0.1, 0.2, 0.3, 0.15, 0.05, 0.25, 0.35, 0.0],
        };

        let mut writer = BitWriter::new();
        write_noise_params(&params, &mut writer).unwrap();

        // Should be exactly 80 bits (8 × 10)
        assert_eq!(writer.bits_written(), 80);

        // Verify the values can round-trip through quantization
        let data = writer.finish();
        for (i, &val) in params.lut.iter().enumerate() {
            let quantized = (val * NOISE_PRECISION).round() as u32;
            let reconstructed = quantized as f32 / NOISE_PRECISION;
            let diff = (val - reconstructed).abs();
            assert!(
                diff < 1.0 / NOISE_PRECISION + 1e-6,
                "LUT[{}]: original={}, reconstructed={}, diff={}",
                i,
                val,
                reconstructed,
                diff,
            );
        }

        // Also verify by reading the bits back
        // Each 10-bit value in little-endian bitstream
        let mut bit_pos = 0;
        for (i, &val) in params.lut.iter().enumerate() {
            let expected_quantized = (val * NOISE_PRECISION).round() as u32;
            // Read 10 bits from the byte stream
            let mut read_val = 0u32;
            for b in 0..10 {
                let byte_idx = (bit_pos + b) / 8;
                let bit_idx = (bit_pos + b) % 8;
                if byte_idx < data.len() {
                    read_val |= (((data[byte_idx] >> bit_idx) & 1) as u32) << b;
                }
            }
            assert_eq!(
                read_val, expected_quantized,
                "LUT[{}]: expected bits {}, got {}",
                i, expected_quantized, read_val,
            );
            bit_pos += 10;
        }
    }

    #[test]
    fn test_noise_params_has_any() {
        let zero = NoiseParams::default();
        assert!(!zero.has_any());

        let nonzero = NoiseParams {
            lut: [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        };
        assert!(nonzero.has_any());
    }

    #[test]
    fn test_noise_params_clamp() {
        // Values should be clamped to [0, NOISE_LUT_MAX]
        let params = NoiseParams {
            lut: [NOISE_LUT_MAX; NUM_NOISE_POINTS],
        };
        let mut writer = BitWriter::new();
        write_noise_params(&params, &mut writer).unwrap();
        // Should not panic
    }

    #[test]
    fn test_index_and_frac() {
        let (idx, frac) = index_and_frac(0.0);
        assert_eq!(idx, 0);
        assert!((frac - 0.0).abs() < 1e-6);

        let (idx, frac) = index_and_frac(0.5);
        assert_eq!(idx, 3);
        assert!((frac - 0.0).abs() < 1e-6);

        let (idx, frac) = index_and_frac(1.0);
        assert_eq!(idx, 6);
        assert!((frac - 0.0).abs() < 1e-6);

        // Beyond range: clamped
        let (idx, frac) = index_and_frac(2.0);
        assert_eq!(idx, 6);
        assert!((frac - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scg_optimizer_quadratic() {
        // Verify the noise loss function converges on uniform data.
        // We use NoiseLossFunction directly with uniform noise levels and
        // check that the SCG optimizer finds the correct constant solution.
        let noise_levels: Vec<NoiseLevel> = (0..100)
            .map(|i| NoiseLevel {
                intensity: i as f32 / 100.0,
                noise_level: 0.1, // uniform noise
            })
            .collect();

        let loss_fn = NoiseLossFunction::new(noise_levels);
        let w0 = [0.5; NUM_NOISE_POINTS]; // Start far from solution
        let result = optimize_scg(&loss_fn, &w0, 1e-8, 40);

        // All values should converge near 0.1
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 0.1).abs() < 0.05,
                "SCG result[{}] = {}, expected ~0.1",
                i,
                v,
            );
        }
    }

    #[test]
    fn test_noise_quality_coef() {
        // Matches libjxl enc_frame.cc:696-709 exactly

        // At d=1.0, coef should be 0.25 (start of ramp)
        let coef = noise_quality_coef(1.0);
        assert!((coef - 0.25).abs() < 1e-6);

        // At d=1.6, coef should be 1.0 (end of ramp)
        let coef = noise_quality_coef(1.6);
        assert!((coef - 1.0).abs() < 1e-6);

        // At d=2.0, coef should be 1.0 (saturated)
        let coef = noise_quality_coef(2.0);
        assert!((coef - 1.0).abs() < 1e-6);

        // At d=0.5, coef should be 1.0 (below ramp start → full noise)
        let coef = noise_quality_coef(0.5);
        assert!((coef - 1.0).abs() < 1e-6);

        // At d=1.3, coef = 0.25 + 0.75 * 0.5 = 0.625
        let coef = noise_quality_coef(1.3);
        assert!((coef - 0.625).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_noise_too_small() {
        // Image smaller than one block → None
        let result = estimate_noise_params(&[0.0; 4], &[0.0; 4], &[0.0; 4], 2, 2, 1.0);
        assert!(result.is_none());
    }
}
