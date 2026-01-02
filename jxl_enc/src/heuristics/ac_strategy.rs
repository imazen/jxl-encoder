//! AC Strategy selection heuristics for VarDCT encoding.
//!
//! Selects optimal DCT transform size for each block based on image content.
//! Smooth areas benefit from larger DCTs (better compression), while
//! detailed areas need smaller DCTs (preserve high frequencies).

use crate::BLOCK_DIM;
use crate::vardct::AcStrategy;

/// Map of AC strategies for each 8x8 block in the image.
#[derive(Clone)]
pub struct AcStrategyMap {
    /// Strategy for each block, stored in row-major order.
    strategies: Vec<AcStrategy>,
    /// Number of blocks in X direction.
    pub blocks_x: usize,
    /// Number of blocks in Y direction.
    pub blocks_y: usize,
}

impl AcStrategyMap {
    /// Create a new map with all DCT8 (default strategy).
    pub fn new_dct8(blocks_x: usize, blocks_y: usize) -> Self {
        Self {
            strategies: vec![AcStrategy::Dct8x8; blocks_x * blocks_y],
            blocks_x,
            blocks_y,
        }
    }

    /// Get strategy at (bx, by).
    pub fn get(&self, bx: usize, by: usize) -> AcStrategy {
        self.strategies[by * self.blocks_x + bx]
    }

    /// Set strategy at (bx, by).
    pub fn set(&mut self, bx: usize, by: usize, strategy: AcStrategy) {
        self.strategies[by * self.blocks_x + bx] = strategy;
    }

    /// Check if all blocks use DCT8.
    pub fn is_all_dct8(&self) -> bool {
        self.strategies.iter().all(|s| *s == AcStrategy::Dct8x8)
    }

    /// Count blocks using each strategy type.
    pub fn strategy_counts(&self) -> [(AcStrategy, usize); 3] {
        let mut dct8_count = 0;
        let mut dct16_count = 0;
        let mut dct32_count = 0;

        for &s in &self.strategies {
            match s {
                AcStrategy::Dct8x8 => dct8_count += 1,
                AcStrategy::Dct16x16 => dct16_count += 1,
                AcStrategy::Dct32x32 => dct32_count += 1,
                _ => {}
            }
        }

        [
            (AcStrategy::Dct8x8, dct8_count),
            (AcStrategy::Dct16x16, dct16_count),
            (AcStrategy::Dct32x32, dct32_count),
        ]
    }

    /// Iterate over all strategies in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &AcStrategy> {
        self.strategies.iter()
    }
}

/// Heuristic level for AC strategy selection.
#[derive(Clone, Copy, Debug, Default)]
pub enum HeuristicLevel {
    /// Use only DCT8 (JPEG-like, fastest)
    #[default]
    Dct8Only,
    /// Use DCT8/DCT16/DCT32 based on variance (moderate quality improvement)
    VarianceBased,
}

/// Compute variance of an 8x8 block.
///
/// Returns sum of squared differences from the mean.
fn block_variance(plane: &[f32], width: usize, height: usize, bx: usize, by: usize) -> f32 {
    let start_x = bx * BLOCK_DIM;
    let start_y = by * BLOCK_DIM;

    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0;

    for dy in 0..BLOCK_DIM {
        let y = (start_y + dy).min(height - 1);
        for dx in 0..BLOCK_DIM {
            let x = (start_x + dx).min(width - 1);
            let v = plane[y * width + x];
            sum += v;
            sum_sq += v * v;
            count += 1;
        }
    }

    let mean = sum / count as f32;
    sum_sq / count as f32 - mean * mean
}

/// Image dimensions for variance computation.
struct ImageDims {
    width: usize,
    height: usize,
    blocks_x: usize,
    blocks_y: usize,
}

/// Compute average variance over a region of blocks.
///
/// Returns the average variance across all 8x8 blocks in the region.
fn region_variance(
    plane: &[f32],
    dims: &ImageDims,
    start_bx: usize,
    start_by: usize,
    region_size: usize,
) -> f32 {
    let mut total_var = 0.0f32;
    let mut count = 0;

    for dy in 0..region_size {
        let by = start_by + dy;
        if by >= dims.blocks_y {
            continue;
        }
        for dx in 0..region_size {
            let bx = start_bx + dx;
            if bx >= dims.blocks_x {
                continue;
            }
            total_var += block_variance(plane, dims.width, dims.height, bx, by);
            count += 1;
        }
    }

    if count > 0 {
        total_var / count as f32
    } else {
        0.0
    }
}

/// Select AC strategies based on image content.
///
/// # Arguments
/// * `y_plane` - Y (luma) channel data in row-major order
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `level` - Heuristic level (complexity/quality tradeoff)
///
/// # Returns
/// AC strategy map for the image.
pub fn select_ac_strategies(
    y_plane: &[f32],
    width: usize,
    height: usize,
    level: HeuristicLevel,
) -> AcStrategyMap {
    let blocks_x = width.div_ceil(BLOCK_DIM);
    let blocks_y = height.div_ceil(BLOCK_DIM);

    match level {
        HeuristicLevel::Dct8Only => AcStrategyMap::new_dct8(blocks_x, blocks_y),
        HeuristicLevel::VarianceBased => {
            select_variance_based(y_plane, width, height, blocks_x, blocks_y)
        }
    }
}

/// Variance thresholds for DCT size selection.
/// Lower variance = smoother = larger DCT beneficial.
const VARIANCE_THRESHOLD_DCT32: f32 = 0.001; // Very smooth
const VARIANCE_THRESHOLD_DCT16: f32 = 0.01; // Moderately smooth

/// Select strategies based on local variance.
fn select_variance_based(
    y_plane: &[f32],
    width: usize,
    height: usize,
    blocks_x: usize,
    blocks_y: usize,
) -> AcStrategyMap {
    let mut map = AcStrategyMap::new_dct8(blocks_x, blocks_y);
    let dims = ImageDims {
        width,
        height,
        blocks_x,
        blocks_y,
    };

    // First pass: try to assign DCT32 (4x4 block regions)
    // DCT32 covers 4x4 = 16 8x8 blocks
    let regions_x = blocks_x / 4;
    let regions_y = blocks_y / 4;

    for ry in 0..regions_y {
        for rx in 0..regions_x {
            let start_bx = rx * 4;
            let start_by = ry * 4;

            let var = region_variance(y_plane, &dims, start_bx, start_by, 4);

            if var < VARIANCE_THRESHOLD_DCT32 {
                // Mark top-left block as DCT32, others will be skipped
                // For simplicity, mark all 16 blocks
                // (proper implementation marks only corner and handles coverage)
                for dy in 0..4 {
                    for dx in 0..4 {
                        map.set(start_bx + dx, start_by + dy, AcStrategy::Dct32x32);
                    }
                }
            }
        }
    }

    // Second pass: try DCT16 (2x2 block regions) for remaining DCT8 areas
    let regions_x = blocks_x / 2;
    let regions_y = blocks_y / 2;

    for ry in 0..regions_y {
        for rx in 0..regions_x {
            let start_bx = rx * 2;
            let start_by = ry * 2;

            // Skip if already covered by DCT32
            if map.get(start_bx, start_by) == AcStrategy::Dct32x32 {
                continue;
            }

            let var = region_variance(y_plane, &dims, start_bx, start_by, 2);

            if var < VARIANCE_THRESHOLD_DCT16 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        map.set(start_bx + dx, start_by + dy, AcStrategy::Dct16x16);
                    }
                }
            }
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac_strategy_map_new() {
        let map = AcStrategyMap::new_dct8(4, 4);
        assert_eq!(map.blocks_x, 4);
        assert_eq!(map.blocks_y, 4);
        assert!(map.is_all_dct8());
    }

    #[test]
    fn test_ac_strategy_map_set_get() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        map.set(1, 2, AcStrategy::Dct16x16);
        assert_eq!(map.get(1, 2), AcStrategy::Dct16x16);
        assert_eq!(map.get(0, 0), AcStrategy::Dct8x8);
        assert!(!map.is_all_dct8());
    }

    #[test]
    fn test_block_variance_uniform() {
        // Uniform block should have zero variance
        let plane = vec![0.5f32; 64];
        let var = block_variance(&plane, 8, 8, 0, 0);
        assert!(
            var.abs() < 1e-6,
            "Uniform block variance should be ~0, got {}",
            var
        );
    }

    #[test]
    fn test_block_variance_gradient() {
        // Gradient has non-zero variance
        let plane: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let var = block_variance(&plane, 8, 8, 0, 0);
        assert!(var > 0.0, "Gradient block should have positive variance");
    }

    #[test]
    fn test_select_dct8_only() {
        let plane = vec![0.5f32; 256]; // 16x16 = 2x2 blocks
        let map = select_ac_strategies(&plane, 16, 16, HeuristicLevel::Dct8Only);
        assert_eq!(map.blocks_x, 2);
        assert_eq!(map.blocks_y, 2);
        assert!(map.is_all_dct8());
    }

    #[test]
    fn test_select_variance_based_uniform() {
        // 32x32 uniform image = 4x4 blocks, low variance
        let plane = vec![0.5f32; 32 * 32];
        let map = select_ac_strategies(&plane, 32, 32, HeuristicLevel::VarianceBased);
        assert_eq!(map.blocks_x, 4);
        assert_eq!(map.blocks_y, 4);

        // Low variance should select larger DCTs
        let counts = map.strategy_counts();
        let dct8_count = counts
            .iter()
            .find(|(s, _)| *s == AcStrategy::Dct8x8)
            .unwrap()
            .1;
        // Should have at least some non-DCT8 blocks for uniform image
        assert!(
            dct8_count < 16,
            "Uniform 4x4 block grid should have some non-DCT8 blocks, got {} DCT8",
            dct8_count
        );
    }

    #[test]
    fn test_select_variance_based_noise() {
        // 32x32 noisy image = high variance
        let plane: Vec<f32> = (0..32 * 32)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let map = select_ac_strategies(&plane, 32, 32, HeuristicLevel::VarianceBased);

        // High variance should keep DCT8
        let counts = map.strategy_counts();
        let dct8_count = counts
            .iter()
            .find(|(s, _)| *s == AcStrategy::Dct8x8)
            .unwrap()
            .1;
        assert!(dct8_count > 0, "Noisy image should have some DCT8 blocks");
    }

    #[test]
    fn test_strategy_counts() {
        let mut map = AcStrategyMap::new_dct8(4, 4);
        map.set(0, 0, AcStrategy::Dct16x16);
        map.set(1, 0, AcStrategy::Dct16x16);
        map.set(0, 1, AcStrategy::Dct16x16);
        map.set(1, 1, AcStrategy::Dct16x16);
        map.set(2, 2, AcStrategy::Dct32x32);

        let counts = map.strategy_counts();
        let dct8 = counts
            .iter()
            .find(|(s, _)| *s == AcStrategy::Dct8x8)
            .unwrap()
            .1;
        let dct16 = counts
            .iter()
            .find(|(s, _)| *s == AcStrategy::Dct16x16)
            .unwrap()
            .1;
        let dct32 = counts
            .iter()
            .find(|(s, _)| *s == AcStrategy::Dct32x32)
            .unwrap()
            .1;

        assert_eq!(dct8, 11); // 16 - 4 - 1 = 11
        assert_eq!(dct16, 4);
        assert_eq!(dct32, 1);
    }
}
