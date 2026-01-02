//! Adaptive quantization field for VarDCT encoding.
//!
//! Adjusts per-block quantization based on local image characteristics.
//! Smooth areas tolerate more quantization, while detailed areas need
//! finer quantization to preserve quality.

use crate::BLOCK_DIM;

/// Per-block quantization field.
pub struct QuantField {
    /// Quant values per block (0-255), stored in row-major order.
    values: Vec<u8>,
    /// Number of blocks in X direction.
    pub blocks_x: usize,
    /// Number of blocks in Y direction.
    pub blocks_y: usize,
    /// Base quant value (from distance parameter).
    pub base_quant: u8,
}

impl QuantField {
    /// Create a uniform quant field (same value for all blocks).
    pub fn uniform(blocks_x: usize, blocks_y: usize, base_quant: u8) -> Self {
        Self {
            values: vec![base_quant; blocks_x * blocks_y],
            blocks_x,
            blocks_y,
            base_quant,
        }
    }

    /// Compute adaptive quant field from image data.
    ///
    /// # Arguments
    /// * `y_plane` - Y (luma) channel data in row-major order
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `base_quant` - Base quantization value from distance parameter
    /// * `strength` - Adaptation strength (0.0 = uniform, 1.0 = full adaptation)
    pub fn compute_adaptive(
        y_plane: &[f32],
        width: usize,
        height: usize,
        base_quant: u8,
        strength: f32,
    ) -> Self {
        let blocks_x = width.div_ceil(BLOCK_DIM);
        let blocks_y = height.div_ceil(BLOCK_DIM);
        let num_blocks = blocks_x * blocks_y;

        let mut values = vec![base_quant; num_blocks];

        // Compute per-block variance
        let variances: Vec<f32> = (0..num_blocks)
            .map(|i| {
                let bx = i % blocks_x;
                let by = i / blocks_x;
                block_variance(y_plane, width, height, bx, by)
            })
            .collect();

        // Find variance range for normalization
        let min_var = variances.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_var = variances.iter().cloned().fold(0.0f32, f32::max);
        let var_range = (max_var - min_var).max(1e-6);

        // Adjust quant based on normalized variance
        // Higher variance = more detail = lower quant (higher quality)
        // Lower variance = smoother = higher quant (more compression)
        for (i, &var) in variances.iter().enumerate() {
            let normalized = (var - min_var) / var_range; // 0.0 = smooth, 1.0 = detailed

            // Compute adjustment factor
            // Smooth areas: increase quant by up to 50%
            // Detailed areas: decrease quant by up to 25%
            let factor = if normalized < 0.5 {
                // Smooth: boost quant (more compression)
                1.0 + strength * (1.0 - 2.0 * normalized) * 0.5
            } else {
                // Detailed: reduce quant (preserve quality)
                1.0 - strength * (2.0 * normalized - 1.0) * 0.25
            };

            let adjusted = (base_quant as f32 * factor).round().clamp(1.0, 255.0) as u8;
            values[i] = adjusted;
        }

        Self {
            values,
            blocks_x,
            blocks_y,
            base_quant,
        }
    }

    /// Get quant value at block position.
    pub fn get(&self, bx: usize, by: usize) -> u8 {
        self.values[by * self.blocks_x + bx]
    }

    /// Set quant value at block position.
    pub fn set(&mut self, bx: usize, by: usize, value: u8) {
        self.values[by * self.blocks_x + bx] = value;
    }

    /// Get all quant values as a slice.
    pub fn values(&self) -> &[u8] {
        &self.values
    }

    /// Check if the field is uniform.
    pub fn is_uniform(&self) -> bool {
        self.values.iter().all(|&v| v == self.base_quant)
    }

    /// Get statistics about the quant field.
    pub fn stats(&self) -> QuantFieldStats {
        let min = *self.values.iter().min().unwrap_or(&self.base_quant);
        let max = *self.values.iter().max().unwrap_or(&self.base_quant);
        let sum: u64 = self.values.iter().map(|&v| v as u64).sum();
        let avg = sum as f32 / self.values.len().max(1) as f32;

        QuantFieldStats { min, max, avg }
    }
}

/// Statistics about a quant field.
#[derive(Clone, Copy, Debug)]
pub struct QuantFieldStats {
    pub min: u8,
    pub max: u8,
    pub avg: f32,
}

/// Compute variance of an 8x8 block.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_field() {
        let field = QuantField::uniform(4, 4, 64);
        assert_eq!(field.blocks_x, 4);
        assert_eq!(field.blocks_y, 4);
        assert!(field.is_uniform());
        assert_eq!(field.get(0, 0), 64);
        assert_eq!(field.get(3, 3), 64);
    }

    #[test]
    fn test_adaptive_field_flat_image() {
        // Flat image should have uniform quant
        let width = 32;
        let height = 32;
        let y_plane = vec![0.5f32; width * height];

        let field = QuantField::compute_adaptive(&y_plane, width, height, 64, 1.0);

        // All blocks have same variance, so should be uniform or nearly uniform
        let stats = field.stats();
        assert!(
            (stats.max - stats.min) <= 1,
            "Flat image should have uniform quant"
        );
    }

    #[test]
    fn test_adaptive_field_varied_image() {
        // Image with one smooth and one textured region
        let width = 16;
        let height = 16;
        let mut y_plane = vec![0.5f32; width * height];

        // Make left half smooth, right half textured
        for y in 0..height {
            for x in width / 2..width {
                y_plane[y * width + x] = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }

        let field = QuantField::compute_adaptive(&y_plane, width, height, 64, 1.0);

        // Left side (smooth) should have higher quant, right side (textured) lower
        let smooth_quant = field.get(0, 0);
        let textured_quant = field.get(1, 0);

        assert!(
            smooth_quant >= textured_quant,
            "Smooth area ({}) should have >= quant than textured ({})",
            smooth_quant,
            textured_quant
        );
    }

    #[test]
    fn test_stats() {
        let mut field = QuantField::uniform(4, 4, 64);
        field.set(0, 0, 32);
        field.set(3, 3, 96);

        let stats = field.stats();
        assert_eq!(stats.min, 32);
        assert_eq!(stats.max, 96);
        assert!(!field.is_uniform());
    }
}
