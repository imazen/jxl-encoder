// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-tile quality measurement from butteraugli diffmap.
//!
//! Computes per-tile (8x8 block) quality metrics from a butteraugli distance map,
//! matching libjxl's `TileDistMap` approach for iterative rate control.

use super::ac_strategy::AcStrategyMap;
use super::common::*;

/// Per-tile distance map computed from butteraugli diffmap.
///
/// Each entry corresponds to one 8x8 block and represents the 16th-norm
/// of butteraugli distances within that block. This matches libjxl's
/// approach for computing per-block quality during rate control.
pub struct TileDistMap {
    /// Per-block distances (16th-norm of butteraugli values).
    pub distances: Vec<f32>,
    /// Number of blocks in x direction.
    pub xsize_blocks: usize,
    /// Number of blocks in y direction.
    pub ysize_blocks: usize,
}

impl TileDistMap {
    /// Create a new tile distance map from a butteraugli diffmap.
    ///
    /// The diffmap should be width × height pixels with one f32 per pixel
    /// representing the local butteraugli distance.
    ///
    /// For each 8x8 block, computes the 16th-norm of the pixel distances:
    /// `block_dist = (sum(pixel_dist^16) / count)^(1/16)`
    ///
    /// This is more sensitive to outliers than RMS, helping to catch
    /// isolated bad pixels that affect perceptual quality.
    pub fn from_diffmap(
        diffmap: &[f32],
        width: usize,
        height: usize,
        _ac_strategy: &AcStrategyMap,
    ) -> Self {
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let mut distances = vec![0.0f32; xsize_blocks * ysize_blocks];

        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let block_y_start = by * BLOCK_DIM;
                let block_x_start = bx * BLOCK_DIM;

                let mut sum_pow = 0.0f32;
                let mut count = 0usize;

                for py in 0..BLOCK_DIM {
                    let y = block_y_start + py;
                    if y >= height {
                        continue;
                    }

                    for px in 0..BLOCK_DIM {
                        let x = block_x_start + px;
                        if x >= width {
                            continue;
                        }

                        let pixel_dist = diffmap[y * width + x];
                        let clamped = pixel_dist.clamp(0.0, 100.0);
                        // x^16 = ((x^2)^2)^2)^2 — avoids expensive powf
                        let v2 = clamped * clamped;
                        let v4 = v2 * v2;
                        let v8 = v4 * v4;
                        let v16 = v8 * v8;
                        sum_pow += v16;
                        count += 1;
                    }
                }

                let block_dist = if count > 0 {
                    // x^(1/16) = sqrt(sqrt(sqrt(sqrt(x))))
                    (sum_pow / count as f32).sqrt().sqrt().sqrt().sqrt()
                } else {
                    0.0
                };

                distances[by * xsize_blocks + bx] = block_dist;
            }
        }

        Self {
            distances,
            xsize_blocks,
            ysize_blocks,
        }
    }

    /// Get the distance for a specific block.
    #[inline]
    pub fn get(&self, bx: usize, by: usize) -> f32 {
        self.distances[by * self.xsize_blocks + bx]
    }

    /// Get the maximum distance across all blocks.
    #[allow(dead_code)] // Used for debug logging
    pub fn max(&self) -> f32 {
        self.distances.iter().copied().fold(0.0f32, |a, b| a.max(b))
    }

    /// Get the 95th percentile distance (ignoring worst 5% of blocks).
    ///
    /// This is useful for convergence checks since a few outlier blocks
    /// shouldn't prevent declaring convergence.
    pub fn percentile_95(&self) -> f32 {
        if self.distances.is_empty() {
            return 0.0;
        }

        let mut sorted = self.distances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (sorted.len() as f32 * 0.95).ceil() as usize;
        let idx = idx.saturating_sub(1).min(sorted.len() - 1);
        sorted[idx]
    }

    /// Get the mean distance across all blocks.
    #[allow(dead_code)] // Used for debug logging
    pub fn mean(&self) -> f32 {
        if self.distances.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.distances.iter().sum();
        sum / self.distances.len() as f32
    }

    /// Count blocks that exceed the target distance.
    #[allow(dead_code)] // Used for debug logging
    pub fn count_exceeding(&self, target: f32) -> usize {
        self.distances.iter().filter(|&&d| d > target).count()
    }

    /// Get the fraction of blocks exceeding the target.
    #[allow(dead_code)] // Used for debug logging
    pub fn fraction_exceeding(&self, target: f32) -> f32 {
        if self.distances.is_empty() {
            return 0.0;
        }
        self.count_exceeding(target) as f32 / self.distances.len() as f32
    }
}

/// Compute butteraugli diffmap between two linear RGB images.
///
/// Returns a per-pixel distance map where each value represents the
/// local butteraugli distance at that pixel.
///
/// Both images must be the same size and in linear RGB format.
pub fn compute_butteraugli_diffmap(
    original: &[f32],
    decoded: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    // Use the butteraugli crate's linear comparison function
    // Both images should be in linear RGB (not sRGB)
    use butteraugli::{ButteraugliParams, butteraugli_linear};
    use imgref::Img;
    use rgb::RGB;

    // Convert flat arrays to RGB arrays
    let orig_rgb: Vec<RGB<f32>> = original
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();

    let decoded_rgb: Vec<RGB<f32>> = decoded
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();

    let orig_img = Img::new(orig_rgb.as_slice(), width, height);
    let decoded_img = Img::new(decoded_rgb.as_slice(), width, height);

    // Create params with diffmap computation enabled
    let params = ButteraugliParams::new().with_compute_diffmap(true);

    // butteraugli_linear returns Result<ButteraugliResult, ButteraugliError>
    match butteraugli_linear(orig_img, decoded_img, &params) {
        Ok(result) => {
            if let Some(diffmap_img) = result.diffmap {
                // Extract the buffer from ImgVec<f32>
                diffmap_img.into_buf()
            } else {
                // Fallback: create uniform diffmap from score
                vec![result.score as f32; width * height]
            }
        }
        Err(_) => {
            // On error, return a high-distance diffmap
            vec![10.0; width * height]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_distmap_uniform() {
        // Create a uniform diffmap
        let width = 32;
        let height = 32;
        let diffmap = vec![1.5f32; width * height];

        let ac_strategy = AcStrategyMap::new_dct8(div_ceil(width, 8), div_ceil(height, 8));
        let tile_dist = TileDistMap::from_diffmap(&diffmap, width, height, &ac_strategy);

        // With uniform values, each block should have the same distance
        for d in &tile_dist.distances {
            assert!((*d - 1.5).abs() < 0.01, "Expected ~1.5, got {}", d);
        }

        assert!((tile_dist.max() - 1.5).abs() < 0.01);
        assert!((tile_dist.mean() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_tile_distmap_outlier_sensitivity() {
        // 16th-norm should be sensitive to outliers
        let width = 8;
        let height = 8;
        let mut diffmap = vec![1.0f32; width * height];

        // Add one outlier
        diffmap[0] = 10.0;

        let ac_strategy = AcStrategyMap::new_dct8(1, 1);
        let tile_dist = TileDistMap::from_diffmap(&diffmap, width, height, &ac_strategy);

        // With 16th-norm, the outlier should dominate
        // (1^16 * 63 + 10^16) / 64 = (63 + 1e16) / 64 ≈ 1.56e14
        // (1.56e14)^(1/16) ≈ 7.5
        let dist = tile_dist.get(0, 0);
        assert!(
            dist > 5.0,
            "Expected > 5.0, got {} (outlier should dominate)",
            dist
        );
        assert!(dist < 10.0, "Expected < 10.0, got {}", dist);
    }
}
