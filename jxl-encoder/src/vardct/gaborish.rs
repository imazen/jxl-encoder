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

// ---------------------------------------------------------------------------
// EX-J13 — Adaptive Gaborish (encoder-only).
//
// The decoder always applies a fixed 3x3 Gabor-like blur. So any encoder-side
// adaptivity must be pre-baked into the post-Gab samples we hand the DCT — we
// cannot signal a per-region multiplier.
//
// The adaptive path runs the same SIMD 5x5 convolution as the fixed path but
// with a per-tile `mul` derived from local Laplacian contrast. The mapping
// is biased BELOW the libjxl baseline of `mul = 1.0`:
//   - high-contrast tiles (edges, text)    → mul = 1.0 (libjxl-faithful)
//   - low-contrast tiles  (sky, gradients) → mul ≈ 0.8 (gentler sharpening)
// Pushing `mul > 1.0` over-sharpens natural content and blows up AC
// coefficient energy with no perceptual win the decoder's fixed 3x3 inverse
// blur can recover; the byte-rate gain lives in *reducing* the kernel's
// reach on smooth regions where sharpening is pure noise.
//
// The tile-by-tile implementation overlaps tiles by 2 pixels (kernel radius)
// and uses the SIMD whole-channel routine on each padded tile, then copies
// the inner 16x16 result back into `data`. This keeps the boundary-clamping
// and SIMD fast paths intact while letting `mul` vary spatially.
// ---------------------------------------------------------------------------

/// Tile size used for the adaptive contrast lookup. Matches the AC block
/// alignment (BLOCK_DIM * 2) so per-tile `mul` shifts at the same granularity
/// as AC strategy decisions.
const ADAPTIVE_TILE: usize = 16;

/// Kernel radius (5x5 kernel → radius 2).
const ADAPTIVE_RADIUS: usize = 2;

/// Map a tile's mean absolute Laplacian to a per-tile multiplier.
///
/// The Laplacian magnitude scales with channel intensity (~0..1 for Y in XYB).
/// The mapping is **biased below the libjxl baseline of `mul = 1.0`**:
///   - At `contrast <= LOW` (smooth gradients, sky) → `MIN_MUL = 0.8`
///   - At `contrast >= HIGH` (text edges, high-frequency texture) → `MAX_MUL = 1.0`
///
/// Bias rationale: the libjxl-baseline kernel has *negative* side-weights, so
/// `mul = 1.0` already does the spec-default sharpening. Pushing `mul > 1.0`
/// (over-sharpening) blows up AC coefficient energy on natural images and
/// hurts file size with no perceptual win that the fixed 3x3 decoder blur
/// can recover. The "0.3-0.5 % BD-rate" target in EX-J13 lives in *reducing*
/// the kernel's reach on smooth regions where the sharpening is pure noise.
/// Edges/text keep the libjxl-faithful `mul = 1.0`.
fn tile_contrast_to_mul(contrast: f32) -> f64 {
    const LOW: f32 = 0.0;
    const HIGH: f32 = 0.05;
    const MIN_MUL: f64 = 0.8;
    const MAX_MUL: f64 = 1.0;
    let t = ((contrast - LOW) / (HIGH - LOW)).clamp(0.0, 1.0) as f64;
    MIN_MUL + t * (MAX_MUL - MIN_MUL)
}

/// Mean absolute Laplacian over a `[ty .. ty + h] × [tx .. tx + w]` window.
///
/// Edge replication for samples outside `[0, width) × [0, height)`. The
/// Laplacian is the simple 4-neighbour form `|4*c - n - s - e - w|`, summed
/// and normalized over the window.
fn tile_mean_abs_laplacian(
    data: &[f32],
    width: usize,
    height: usize,
    tx: usize,
    ty: usize,
    tw: usize,
    th: usize,
) -> f32 {
    debug_assert!(tx < width && ty < height);
    let mut sum = 0.0f32;
    let mut n = 0u32;
    let x_end = (tx + tw).min(width);
    let y_end = (ty + th).min(height);
    for y in ty..y_end {
        let y_n = y.saturating_sub(1);
        let y_s = (y + 1).min(height - 1);
        for x in tx..x_end {
            let x_w = x.saturating_sub(1);
            let x_e = (x + 1).min(width - 1);
            let c = data[y * width + x];
            let nn = data[y_n * width + x];
            let ss = data[y_s * width + x];
            let ww = data[y * width + x_w];
            let ee = data[y * width + x_e];
            sum += (4.0 * c - nn - ss - ww - ee).abs();
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// Apply adaptive 5x5 gaborish to one channel in-place.
///
/// Computes per-tile `mul` from local Laplacian contrast on `data`, then runs
/// the SIMD kernel on `data` tile-by-tile. Each tile is processed against an
/// `ADAPTIVE_RADIUS`-padded view that re-runs the whole-channel SIMD routine
/// — this is slightly wasteful per-tile but keeps the SIMD fast path
/// (jxl_simd::gaborish_5x5_channel) as the only convolution backend, so the
/// adaptive path inherits all of the same boundary handling.
///
/// The contrast is sampled on the PRE-filter `data`, which matches the
/// libjxl-style "decide once, then sharpen" pattern and avoids feedback
/// between the multiplier and the kernel output.
fn apply_channel_adaptive(data: &mut [f32], scratch: &mut [f32], width: usize, height: usize) {
    // First snapshot the original samples — we need them after each tile
    // overwrites the corresponding slice of `data` so subsequent tiles still
    // sample the unfiltered signal for contrast estimation.
    //
    // (Allocation: one channel-sized vec. Adaptive path is opt-in so callers
    // accept the extra memory pressure.)
    let original: alloc::vec::Vec<f32> = data.to_vec();

    let tiles_x = width.div_ceil(ADAPTIVE_TILE);
    let tiles_y = height.div_ceil(ADAPTIVE_TILE);
    // Padded tile size (interior + 2*radius border for kernel reach).
    let pad_tile_w = ADAPTIVE_TILE + 2 * ADAPTIVE_RADIUS;
    let pad_tile_h = ADAPTIVE_TILE + 2 * ADAPTIVE_RADIUS;
    let pad_len = pad_tile_w * pad_tile_h;

    // Reusable per-tile buffers.
    let mut tile_in: alloc::vec::Vec<f32> = alloc::vec![0.0f32; pad_len];
    let mut tile_scratch: alloc::vec::Vec<f32> = alloc::vec![0.0f32; pad_len];

    for ty_idx in 0..tiles_y {
        for tx_idx in 0..tiles_x {
            let tx = tx_idx * ADAPTIVE_TILE;
            let ty = ty_idx * ADAPTIVE_TILE;
            let tw = ADAPTIVE_TILE.min(width - tx);
            let th = ADAPTIVE_TILE.min(height - ty);

            let contrast = tile_mean_abs_laplacian(&original, width, height, tx, ty, tw, th);
            let mul = tile_contrast_to_mul(contrast);

            // Build the padded tile from `original` with edge replication.
            for py in 0..pad_tile_h {
                let src_y = (ty as isize + py as isize - ADAPTIVE_RADIUS as isize)
                    .clamp(0, height as isize - 1) as usize;
                for px in 0..pad_tile_w {
                    let src_x = (tx as isize + px as isize - ADAPTIVE_RADIUS as isize)
                        .clamp(0, width as isize - 1) as usize;
                    tile_in[py * pad_tile_w + px] = original[src_y * width + src_x];
                }
            }

            apply_channel(&mut tile_in, &mut tile_scratch, pad_tile_w, pad_tile_h, mul);

            // Copy the interior `tw x th` of the filtered tile back to `data`.
            for iy in 0..th {
                let src_off = (iy + ADAPTIVE_RADIUS) * pad_tile_w + ADAPTIVE_RADIUS;
                let dst_off = (ty + iy) * width + tx;
                data[dst_off..dst_off + tw].copy_from_slice(&tile_in[src_off..src_off + tw]);
            }
        }
    }

    // Mark the named scratch parameter as used to match the non-adaptive
    // signature; the adaptive path manages its own per-tile scratch above.
    let _ = scratch;
}

/// Apply gaborish inverse sharpening to all three XYB channels.
///
/// This should be called AFTER noise estimation/denoising and BEFORE
/// adaptive quantization, matching the libjxl pipeline order.
///
/// Uses `mul=[1.0, 1.0, 1.0]` for all channels (libjxl VarDCT default).
///
/// Kept around for the `__pre_quantized` re-export
/// (`crate::__pre_quantized::gaborish_inverse`) and for the
/// `with_patches_data` integration test, both of which want the unconditional
/// libjxl-faithful fixed kernel. The crate-internal VarDCT pipeline calls
/// [`gaborish_inverse_maybe_adaptive`] directly.
#[allow(dead_code)] // Used through __pre_quantized re-export and integration tests.
pub fn gaborish_inverse(
    xyb_x: &mut [f32],
    xyb_y: &mut [f32],
    xyb_b: &mut [f32],
    width: usize,
    height: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<()> {
    gaborish_inverse_maybe_adaptive(xyb_x, xyb_y, xyb_b, width, height, false, budget)
}

/// Apply gaborish inverse to all three XYB channels with optional per-tile
/// adaptive kernel strength (EX-J13).
///
/// When `adaptive` is `false`, equivalent to [`gaborish_inverse`]: every tile
/// receives `mul = 1.0` and the SIMD whole-channel kernel runs once per
/// channel.
///
/// When `adaptive` is `true`, each 16x16 tile gets a `mul ∈ [0.8, 1.2]`
/// derived from its local Laplacian contrast — sharper on edges/text, gentler
/// on smooth regions. **Encoder-only**: the decoder always applies the fixed
/// 3x3 inverse Gabor blur, so the adaptive multiplier must be baked into the
/// post-Gab samples before they reach the DCT.
///
/// Notes:
///   - The Y (luma) channel drives almost all of the perceptual benefit and
///     bit-cost dynamics. Adaptive is currently applied to the Y channel
///     only; X and B keep `mul = 1.0`. This matches the principle in the
///     `compute_mask1x1` pipeline (mask is also driven by Y intensity) and
///     keeps chroma drift minimal.
///   - The adaptive path is heavier than the fixed path (one channel-sized
///     snapshot plus per-tile work). The flag is off by default; callers opt
///     in via `LossyConfig::with_adaptive_gaborish(true)`.
pub fn gaborish_inverse_maybe_adaptive(
    xyb_x: &mut [f32],
    xyb_y: &mut [f32],
    xyb_b: &mut [f32],
    width: usize,
    height: usize,
    adaptive: bool,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<()> {
    // mul=1.0 for all channels, matching libjxl enc_heuristics.cc line 1137-1140.
    //
    // Channels are independent: apply_channel mutates its own input slice using
    // its own scratch. With `parallel`, run all 3 concurrently via rayon::join.
    // Serial fallback reuses one scratch buffer across channels for allocation
    // economy.
    let n = width
        .checked_mul(height)
        .ok_or(crate::error::Error::DimensionOverflow {
            width,
            height,
            channels: 1,
        })?;
    #[cfg(feature = "parallel")]
    {
        // Three concurrent scratch buffers, one per channel — accounted as a
        // transient guard. Released as soon as rayon::join returns.
        // Adaptive Y also allocates one extra channel-sized snapshot; reserve
        // proactively so the budget gate sees consistent peak usage.
        let extra = if adaptive { 1 } else { 0 };
        let _g = crate::budget::MemoryBudget::reserve_opt(
            budget,
            (n as u64).saturating_mul(4 * (3 + extra)),
        )?;
        let (((), ()), ()) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut scratch = jxl_simd::vec_f32_dirty(n);
                        apply_channel(xyb_x, &mut scratch, width, height, 1.0);
                    },
                    || {
                        let mut scratch = jxl_simd::vec_f32_dirty(n);
                        if adaptive {
                            apply_channel_adaptive(xyb_y, &mut scratch, width, height);
                        } else {
                            apply_channel(xyb_y, &mut scratch, width, height, 1.0);
                        }
                    },
                )
            },
            || {
                let mut scratch = jxl_simd::vec_f32_dirty(n);
                apply_channel(xyb_b, &mut scratch, width, height, 1.0);
            },
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        // Reuse one scratch buffer across all 3 channels to avoid 3 allocations.
        let extra = if adaptive { 1 } else { 0 };
        let _g = crate::budget::MemoryBudget::reserve_opt(
            budget,
            (n as u64).saturating_mul(4 * (1 + extra)),
        )?;
        let mut scratch = jxl_simd::vec_f32_dirty(n);
        apply_channel(xyb_x, &mut scratch, width, height, 1.0);
        if adaptive {
            apply_channel_adaptive(xyb_y, &mut scratch, width, height);
        } else {
            apply_channel(xyb_y, &mut scratch, width, height, 1.0);
        }
        apply_channel(xyb_b, &mut scratch, width, height, 1.0);
    }
    Ok(())
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

    // -----------------------------------------------------------------------
    // EX-J13 — Adaptive Gaborish tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_mul_low_contrast() {
        // Flat tile → mul ≈ 0.8 (lower bound).
        assert!((tile_contrast_to_mul(0.0) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_mul_high_contrast() {
        // Saturating contrast → mul ≈ 1.0 (libjxl baseline upper bound).
        assert!((tile_contrast_to_mul(1.0) - 1.0).abs() < 1e-9);
        assert!((tile_contrast_to_mul(0.05) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_mul_never_exceeds_baseline() {
        // EX-J13 contract: adaptive `mul` is always <= the libjxl baseline
        // `mul = 1.0`. Pushing above 1.0 would over-sharpen on edges and
        // blow up AC coefficient energy with no perceptual win the decoder
        // can recover.
        let pts: [f32; 9] = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 10.0];
        for &c in &pts {
            let m = tile_contrast_to_mul(c);
            assert!(m <= 1.0 + 1e-9, "mul({}) = {} > 1.0", c, m);
            assert!(m >= 0.8 - 1e-9, "mul({}) = {} < 0.8", c, m);
        }
    }

    #[test]
    fn test_adaptive_mul_monotonic() {
        // Multiplier must be non-decreasing in contrast.
        let pts: [f32; 6] = [0.0, 0.005, 0.01, 0.02, 0.04, 0.1];
        let mut prev = tile_contrast_to_mul(pts[0]);
        for &c in &pts[1..] {
            let m = tile_contrast_to_mul(c);
            assert!(
                m >= prev - 1e-9,
                "non-monotonic at {} ({} -> {})",
                c,
                prev,
                m
            );
            prev = m;
        }
    }

    #[test]
    fn test_adaptive_uniform_image_preserved() {
        // A constant-value image should still be (essentially) unchanged under
        // the adaptive path — contrast is 0 everywhere → mul=0.8 → kernel is
        // normalized → output ≈ input.
        let width = 32;
        let height = 32;
        let value = 0.42f32;
        let mut data = alloc::vec![value; width * height];
        let mut scratch = alloc::vec![0.0f32; width * height];
        apply_channel_adaptive(&mut data, &mut scratch, width, height);
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - value).abs() < 1e-5,
                "adaptive: pixel {} drifted from {} to {}",
                i,
                value,
                v
            );
        }
    }

    #[test]
    fn test_adaptive_low_contrast_softens() {
        // A nearly-flat tile (tiny noise on top of a constant background)
        // should pick `mul < 1.0` so the post-filter sample drift toward
        // the neighborhood mean is SMALLER than the fixed `mul = 1.0`
        // path. The fixed kernel has a `wc > 1.0` center weight and
        // negative side weights — both shrink in magnitude when `mul`
        // shrinks, so the output stays closer to the input on smooth
        // tiles. (Effectively: the adaptive path is a gentler
        // pre-sharpener on smooth regions, which is the byte-rate win.)
        let width = 16;
        let height = 16;
        // Smooth gradient with a tiny perturbation — Laplacian is small.
        let mut fixed: alloc::vec::Vec<f32> = (0..width * height)
            .map(|i| 0.5 + (i as f32) * 1e-5)
            .collect();
        let mut adapt = fixed.clone();
        let original = fixed.clone();

        let mut scratch_a = alloc::vec![0.0f32; width * height];
        apply_channel(&mut fixed, &mut scratch_a, width, height, 1.0);
        let mut scratch_b = alloc::vec![0.0f32; width * height];
        apply_channel_adaptive(&mut adapt, &mut scratch_b, width, height);

        // Sum of |adapt - original| should be <= sum of |fixed - original|.
        let drift_fixed: f64 = fixed
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (*a - *b).abs() as f64)
            .sum();
        let drift_adapt: f64 = adapt
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (*a - *b).abs() as f64)
            .sum();
        assert!(
            drift_adapt <= drift_fixed + 1e-6,
            "adaptive should not perturb a near-flat tile more than fixed: \
             drift_fixed={} drift_adapt={}",
            drift_fixed,
            drift_adapt
        );
    }

    #[test]
    fn test_adaptive_pipeline_three_channels() {
        // End-to-end smoke for the three-channel adaptive entry point.
        let width = 32;
        let height = 32;
        let mut x: alloc::vec::Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let mut y: alloc::vec::Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.01).cos())
            .collect();
        let mut b: alloc::vec::Vec<f32> = (0..width * height)
            .map(|i| ((i % 7) as f32) * 0.05)
            .collect();
        gaborish_inverse_maybe_adaptive(
            &mut x, &mut y, &mut b, width, height, /* adaptive */ true, None,
        )
        .expect("adaptive gaborish should succeed");
        // Values should remain finite.
        for v in x.iter().chain(y.iter()).chain(b.iter()) {
            assert!(v.is_finite(), "adaptive gaborish produced non-finite value");
        }
    }
}
