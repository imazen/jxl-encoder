//! Chroma-from-Luma (CfL) correlation for VarDCT encoding.
//!
//! Computes linear correlation between Y (luma) and X/B (chroma) channels
//! to improve compression. The residuals (X - factor*Y, B - factor*Y)
//! typically have lower entropy than raw X/B values.

use crate::BLOCK_DIM;

/// Color tile dimension in pixels (8 blocks = 64 pixels).
pub const COLOR_TILE_DIM: usize = 64;

/// Color tile dimension in 8x8 blocks.
pub const COLOR_TILE_DIM_IN_BLOCKS: usize = COLOR_TILE_DIM / BLOCK_DIM;

/// Default color factor (denominator for CfL ratios).
pub const DEFAULT_COLOR_FACTOR: u32 = 84;

/// Default Y-to-B ratio from JXL spec (kYToBRatio).
pub const DEFAULT_YTOB_RATIO: f32 = 1.0;

/// CfL correlation parameters for a single tile.
#[derive(Clone, Copy, Debug, Default)]
pub struct TileCorrelation {
    /// Y-to-X factor (quantized, range -128 to 127).
    pub ytox_factor: i8,
    /// Y-to-B factor (quantized, range -128 to 127).
    pub ytob_factor: i8,
}

/// CfL correlation map for the entire image.
pub struct ColorCorrelationMap {
    /// Per-tile correlation factors.
    tiles: Vec<TileCorrelation>,
    /// Number of tiles in X direction.
    pub tiles_x: usize,
    /// Number of tiles in Y direction.
    pub tiles_y: usize,
    /// Color factor (denominator, typically 84).
    pub color_factor: u32,
    /// DC correlation for X channel.
    pub ytox_dc: i32,
    /// DC correlation for B channel.
    pub ytob_dc: i32,
    /// Base correlation for X (typically 0).
    pub base_correlation_x: f32,
    /// Base correlation for B (typically 1.0).
    pub base_correlation_b: f32,
}

impl ColorCorrelationMap {
    /// Create a new color correlation map with default (no) correlation.
    pub fn new_default(width: usize, height: usize) -> Self {
        let tiles_x = width.div_ceil(COLOR_TILE_DIM);
        let tiles_y = height.div_ceil(COLOR_TILE_DIM);
        let num_tiles = tiles_x * tiles_y;

        Self {
            tiles: vec![TileCorrelation::default(); num_tiles],
            tiles_x,
            tiles_y,
            color_factor: DEFAULT_COLOR_FACTOR,
            ytox_dc: 0,
            ytob_dc: 0,
            base_correlation_x: 0.0,
            base_correlation_b: DEFAULT_YTOB_RATIO,
        }
    }

    /// Compute CfL correlation from XYB image data.
    ///
    /// # Arguments
    /// * `xyb_data` - Interleaved XYB data (X, Y, B per pixel)
    /// * `width` - Image width
    /// * `height` - Image height
    pub fn compute_from_xyb(xyb_data: &[f32], width: usize, height: usize) -> Self {
        let tiles_x = width.div_ceil(COLOR_TILE_DIM);
        let tiles_y = height.div_ceil(COLOR_TILE_DIM);
        let num_tiles = tiles_x * tiles_y;

        let mut tiles = vec![TileCorrelation::default(); num_tiles];

        // Compute per-tile correlation
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let tile_idx = ty * tiles_x + tx;
                tiles[tile_idx] = compute_tile_correlation(xyb_data, width, height, tx, ty);
            }
        }

        // Compute DC correlation (average across all tiles)
        let (ytox_dc, ytob_dc) = compute_dc_correlation(&tiles);

        Self {
            tiles,
            tiles_x,
            tiles_y,
            color_factor: DEFAULT_COLOR_FACTOR,
            ytox_dc,
            ytob_dc,
            base_correlation_x: 0.0,
            base_correlation_b: DEFAULT_YTOB_RATIO,
        }
    }

    /// Get tile correlation at tile coordinates.
    pub fn get(&self, tx: usize, ty: usize) -> TileCorrelation {
        self.tiles[ty * self.tiles_x + tx]
    }

    /// Get Y-to-X ratio for a given factor.
    pub fn ytox_ratio(&self, factor: i32) -> f32 {
        self.base_correlation_x + factor as f32 / self.color_factor as f32
    }

    /// Get Y-to-B ratio for a given factor.
    pub fn ytob_ratio(&self, factor: i32) -> f32 {
        self.base_correlation_b + factor as f32 / self.color_factor as f32
    }

    /// Check if this is the default (no) correlation.
    pub fn is_default(&self) -> bool {
        self.ytox_dc == 0
            && self.ytob_dc == 0
            && self.color_factor == DEFAULT_COLOR_FACTOR
            && self
                .tiles
                .iter()
                .all(|t| t.ytox_factor == 0 && t.ytob_factor == 0)
    }
}

/// Compute correlation factors for a single tile.
fn compute_tile_correlation(
    xyb_data: &[f32],
    width: usize,
    height: usize,
    tx: usize,
    ty: usize,
) -> TileCorrelation {
    let start_x = tx * COLOR_TILE_DIM;
    let start_y = ty * COLOR_TILE_DIM;
    let end_x = (start_x + COLOR_TILE_DIM).min(width);
    let end_y = (start_y + COLOR_TILE_DIM).min(height);

    // Compute linear regression: X = a*Y + b, B = c*Y + d
    // We want to find 'a' and 'c' (slopes)
    let mut sum_y = 0.0f64;
    let mut sum_y2 = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_by = 0.0f64;
    let mut count = 0.0f64;

    for py in start_y..end_y {
        for px in start_x..end_x {
            let idx = (py * width + px) * 3;
            let x = xyb_data[idx] as f64;
            let y = xyb_data[idx + 1] as f64;
            let b = xyb_data[idx + 2] as f64;

            sum_y += y;
            sum_y2 += y * y;
            sum_x += x;
            sum_xy += x * y;
            sum_b += b;
            sum_by += b * y;
            count += 1.0;
        }
    }

    if count < 1.0 {
        return TileCorrelation::default();
    }

    // Linear regression slope: (n*sum_xy - sum_x*sum_y) / (n*sum_y2 - sum_y^2)
    let denom = count * sum_y2 - sum_y * sum_y;

    let ytox_slope = if denom.abs() > 1e-10 {
        (count * sum_xy - sum_x * sum_y) / denom
    } else {
        0.0
    };

    let ytob_slope = if denom.abs() > 1e-10 {
        (count * sum_by - sum_b * sum_y) / denom
    } else {
        DEFAULT_YTOB_RATIO as f64
    };

    // Convert to quantized factors
    // factor = (slope - base_correlation) * color_factor
    let ytox_factor = (ytox_slope * DEFAULT_COLOR_FACTOR as f64)
        .round()
        .clamp(-127.0, 127.0) as i8;

    // For B channel, subtract base correlation (1.0)
    let ytob_factor = ((ytob_slope - DEFAULT_YTOB_RATIO as f64) * DEFAULT_COLOR_FACTOR as f64)
        .round()
        .clamp(-127.0, 127.0) as i8;

    TileCorrelation {
        ytox_factor,
        ytob_factor,
    }
}

/// Compute DC correlation from tile factors.
fn compute_dc_correlation(tiles: &[TileCorrelation]) -> (i32, i32) {
    if tiles.is_empty() {
        return (0, 0);
    }

    // Average the tile factors for DC
    let sum_x: i32 = tiles.iter().map(|t| t.ytox_factor as i32).sum();
    let sum_b: i32 = tiles.iter().map(|t| t.ytob_factor as i32).sum();
    let count = tiles.len() as i32;

    (sum_x / count, sum_b / count)
}

/// Apply CfL decorrelation to XYB coefficients in-place.
///
/// Transforms X and B channels to residuals:
/// X' = X - ratio_x * Y
/// B' = B - ratio_b * Y
///
/// This should be called on DC coefficients or per-tile AC coefficients.
pub fn apply_cfl_decorrelation(
    xyb_coeffs: &mut [f32],
    y_coeffs: &[f32],
    ytox_ratio: f32,
    ytob_ratio: f32,
) {
    // xyb_coeffs is interleaved (X, Y, B) per position
    // y_coeffs is just Y values
    debug_assert_eq!(xyb_coeffs.len(), y_coeffs.len() * 3);

    for i in 0..y_coeffs.len() {
        let y = y_coeffs[i];
        xyb_coeffs[i * 3] -= ytox_ratio * y; // X' = X - ratio_x * Y
        // Y stays the same
        xyb_coeffs[i * 3 + 2] -= ytob_ratio * y; // B' = B - ratio_b * Y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_correlation() {
        let cmap = ColorCorrelationMap::new_default(64, 64);
        assert_eq!(cmap.tiles_x, 1);
        assert_eq!(cmap.tiles_y, 1);
        assert!(cmap.is_default());
    }

    #[test]
    fn test_correlation_from_flat_image() {
        // Flat gray image - no correlation needed
        let width = 64;
        let height = 64;
        let xyb_data: Vec<f32> = (0..width * height * 3).map(|_| 0.5).collect();

        let cmap = ColorCorrelationMap::compute_from_xyb(&xyb_data, width, height);
        assert_eq!(cmap.tiles_x, 1);
        assert_eq!(cmap.tiles_y, 1);

        // For a flat image, slopes should be ~0
        let tile = cmap.get(0, 0);
        assert!(
            tile.ytox_factor.abs() < 10,
            "ytox_factor should be small for flat image: {}",
            tile.ytox_factor
        );
    }

    #[test]
    fn test_correlation_from_correlated_image() {
        // Image where X = 0.5 * Y (strong correlation)
        // and B = 1.3 * Y (base_correlation_b = 1.0, so factor ≈ +25)
        let width = 64;
        let height = 64;
        let mut xyb_data = vec![0.0f32; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let luma = (x + y) as f32 / 128.0 + 0.1; // Y channel (avoid zero)
                xyb_data[idx] = 0.5 * luma; // X = 0.5 * Y
                xyb_data[idx + 1] = luma; // Y
                xyb_data[idx + 2] = 1.3 * luma; // B = 1.3 * Y (no intercept)
            }
        }

        let cmap = ColorCorrelationMap::compute_from_xyb(&xyb_data, width, height);
        let tile = cmap.get(0, 0);

        // X has slope 0.5, factor = 0.5 * 84 ≈ 42
        assert!(
            (tile.ytox_factor as i32 - 42).abs() < 5,
            "ytox_factor should be ~42, got {}",
            tile.ytox_factor
        );

        // B has slope 1.3, base is 1.0, so factor = (1.3 - 1.0) * 84 ≈ 25
        assert!(
            (tile.ytob_factor as i32 - 25).abs() < 5,
            "ytob_factor should be ~25, got {}",
            tile.ytob_factor
        );
    }

    #[test]
    fn test_decorrelation() {
        let mut xyb = vec![0.5, 1.0, 1.5]; // X=0.5, Y=1.0, B=1.5
        let y = vec![1.0];

        apply_cfl_decorrelation(&mut xyb, &y, 0.5, 1.0);

        // X' = 0.5 - 0.5*1.0 = 0.0
        // B' = 1.5 - 1.0*1.0 = 0.5
        assert!((xyb[0] - 0.0).abs() < 0.001);
        assert!((xyb[1] - 1.0).abs() < 0.001); // Y unchanged
        assert!((xyb[2] - 0.5).abs() < 0.001);
    }
}
