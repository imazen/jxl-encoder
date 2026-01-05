//! Quantization weights and dequantization matrices for VarDCT encoding.
//!
//! JPEG XL uses perceptually-weighted quantization matrices for each transform type.
//! There are 17 distinct quantization tables (some transforms share tables).
//!
//! For encoding, we primarily need:
//! - DC quantization factors per channel (X, Y, B)
//! - The ability to signal "use library defaults" (most common case)
//! - Eventually: custom matrix support

use super::ac_strategy::AcStrategy;
use crate::bit_writer::BitWriter;

/// Number of distinct quantization tables.
pub const NUM_QUANT_TABLES: usize = 17;

/// Inverse LF (DC) quantization factors per channel [X, Y, B].
/// These are the denominators: DC_quant = raw_DC / INV_LF_QUANT[c]
pub const INV_LF_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];

/// LF (DC) quantization factors per channel [X, Y, B].
/// DC_quant = raw_DC * LF_QUANT[c]
pub const LF_QUANT: [f32; 3] = [
    1.0 / INV_LF_QUANT[0], // X: 1/4096
    1.0 / INV_LF_QUANT[1], // Y: 1/512
    1.0 / INV_LF_QUANT[2], // B: 1/256
];

/// Quantization table index.
/// Maps the 27 transform types to 17 distinct quantization tables.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantTable {
    Dct8 = 0,
    Identity = 1,
    Dct2 = 2,
    Dct4 = 3,
    Dct16 = 4,
    Dct32 = 5,
    Dct8x16 = 6,
    Dct8x32 = 7,
    Dct16x32 = 8,
    Dct4x8 = 9,
    Afv = 10,
    Dct64 = 11,
    Dct32x64 = 12,
    Dct128 = 13,
    Dct64x128 = 14,
    Dct256 = 15,
    Dct128x256 = 16,
}

impl QuantTable {
    /// Number of quant table types.
    pub const CARDINALITY: usize = NUM_QUANT_TABLES;

    /// All quant table values.
    pub const ALL: [QuantTable; NUM_QUANT_TABLES] = [
        QuantTable::Dct8,
        QuantTable::Identity,
        QuantTable::Dct2,
        QuantTable::Dct4,
        QuantTable::Dct16,
        QuantTable::Dct32,
        QuantTable::Dct8x16,
        QuantTable::Dct8x32,
        QuantTable::Dct16x32,
        QuantTable::Dct4x8,
        QuantTable::Afv,
        QuantTable::Dct64,
        QuantTable::Dct32x64,
        QuantTable::Dct128,
        QuantTable::Dct64x128,
        QuantTable::Dct256,
        QuantTable::Dct128x256,
    ];

    /// Get the quant table for an AC strategy.
    pub fn from_ac_strategy(strategy: AcStrategy) -> QuantTable {
        QuantTable::ALL[strategy.dequant_matrix_index()]
    }

    /// Required size in X (number of 8x8 blocks).
    pub fn required_size_x(self) -> usize {
        const LUT: [usize; NUM_QUANT_TABLES] =
            [1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 8, 4, 16, 8, 32, 16];
        LUT[self as usize]
    }

    /// Required size in Y (number of 8x8 blocks).
    pub fn required_size_y(self) -> usize {
        const LUT: [usize; NUM_QUANT_TABLES] =
            [1, 1, 1, 1, 2, 4, 2, 4, 4, 1, 1, 8, 8, 16, 16, 32, 32];
        LUT[self as usize]
    }

    /// Total number of coefficients in this quant table.
    pub fn num_coefficients(self) -> usize {
        self.required_size_x() * self.required_size_y() * 64
    }
}

/// Quantization encoding mode.
#[derive(Clone, Debug, Default)]
pub enum QuantEncoding {
    /// Use library defaults (most common).
    #[default]
    Library,
    // Other modes (Identity, Dct2, Dct4, Dct4x8, Afv, Dct, Raw) can be added later
}

/// Dequantization matrices for VarDCT encoding.
///
/// For encoding, we start with Library mode (default matrices).
/// Custom matrices can be added later for advanced use cases.
#[derive(Clone, Debug)]
pub struct DequantMatrices {
    /// Encoding mode for each of the 17 tables.
    /// Currently unused but reserved for custom matrix support.
    #[allow(dead_code)]
    encodings: [QuantEncoding; NUM_QUANT_TABLES],
    /// Whether all tables use default Library mode.
    all_default: bool,
}

impl Default for DequantMatrices {
    fn default() -> Self {
        Self::new_library()
    }
}

impl DequantMatrices {
    /// Create with all library defaults.
    pub fn new_library() -> Self {
        Self {
            encodings: std::array::from_fn(|_| QuantEncoding::Library),
            all_default: true,
        }
    }

    /// Check if all tables use library defaults.
    pub fn is_all_default(&self) -> bool {
        self.all_default
    }

    /// Write the dequant matrices header to the bitstream.
    ///
    /// For Library mode, this just writes a single bit (all_default=1).
    pub fn write(&self, writer: &mut BitWriter) {
        if self.all_default {
            // all_default = true
            let _ = writer.write(1, 1);
        } else {
            // all_default = false, then encode each table
            let _ = writer.write(1, 0);
            // TODO: Implement per-table encoding when needed
            unimplemented!("Custom dequant matrices not yet supported");
        }
    }
}

/// LF channel dequantization factors.
///
/// These scale the DC coefficients per channel.
#[derive(Clone, Debug)]
pub struct LfQuantFactors {
    /// Quantization factors [X, Y, B].
    pub quant_factors: [f32; 3],
    /// Inverse quantization factors [X, Y, B].
    pub inv_quant_factors: [f32; 3],
}

impl Default for LfQuantFactors {
    fn default() -> Self {
        Self {
            quant_factors: LF_QUANT,
            inv_quant_factors: INV_LF_QUANT,
        }
    }
}

impl LfQuantFactors {
    /// Create with default LF quant factors.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the quantization factor for a channel.
    pub fn quant(&self, channel: usize) -> f32 {
        self.quant_factors[channel]
    }

    /// Get the inverse quantization factor for a channel.
    pub fn inv_quant(&self, channel: usize) -> f32 {
        self.inv_quant_factors[channel]
    }
}

// =============================================================================
// DCT8 Perceptual Quantization Weights
// =============================================================================

/// DCT8 library default distance bands per channel [X, Y, B].
/// These define the perceptual weighting for each coefficient position.
const DCT8_DISTANCE_BANDS: [[f32; 6]; 3] = [
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0], // X channel
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],  // Y channel
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],  // B channel
];

/// Number of distance bands for DCT8.
const DCT8_NUM_BANDS: usize = 6;

/// Minimum valid weight value.
const ALMOST_ZERO: f32 = 1e-8;

/// sqrt(2) for distance normalization.
const SQRT_2: f32 = std::f32::consts::SQRT_2;

/// Convert differential band parameter to multiplier.
///
/// If v > 0: returns 1 + v (grows by v proportion)
/// If v <= 0: returns 1 / (1 - v) (shrinks by -v proportion)
#[inline]
fn band_mult(v: f32) -> f32 {
    if v > 0.0 { 1.0 + v } else { 1.0 / (1.0 - v) }
}

/// Interpolate in log-space between array values.
///
/// Performs exponential interpolation: a * (b/a)^frac
#[inline]
fn interpolate_vec(scaled_pos: f32, bands: &[f32]) -> f32 {
    let idx = scaled_pos as usize;
    let frac = scaled_pos - idx as f32;
    let a = bands[idx];
    let b = bands[idx + 1];
    // a * (b/a)^frac = exp(frac * ln(b/a) + ln(a))
    a * (b / a).powf(frac)
}

/// Generate DCT8 quantization weights for all 3 channels.
///
/// Returns a 3*64 = 192 element array with weights for each position.
/// Layout: [X0..X63, Y0..Y63, B0..B63] where index i is at row i/8, col i%8.
pub fn generate_dct8_weights() -> [f32; 3 * 64] {
    let mut weights = [0.0f32; 3 * 64];

    for c in 0..3 {
        // Expand differential parameters into absolute band values
        let mut bands = [0.0f32; DCT8_NUM_BANDS];
        bands[0] = DCT8_DISTANCE_BANDS[c][0];

        for i in 1..DCT8_NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(DCT8_DISTANCE_BANDS[c][i]);
        }

        // Compute spatial scaling factors
        // Maps (0,0) to 0 and (7,7) to (num_bands-1) / sqrt(2)
        let scale = (DCT8_NUM_BANDS - 1) as f32 / (SQRT_2 + 1e-6);
        let rcpcol = scale / 7.0; // 8 columns, max index 7
        let rcprow = scale / 7.0; // 8 rows, max index 7

        // For each pixel position, compute weight based on distance from DC
        for y in 0..8 {
            let dy = y as f32 * rcprow;
            let dy2 = dy * dy;

            for x in 0..8 {
                let dx = x as f32 * rcpcol;
                let scaled_distance = (dx * dx + dy2).sqrt();

                // Clamp to valid interpolation range
                let scaled_distance = scaled_distance.min((DCT8_NUM_BANDS - 1) as f32 - 0.001);

                let weight = interpolate_vec(scaled_distance, &bands);
                weights[c * 64 + y * 8 + x] = weight.max(ALMOST_ZERO);
            }
        }
    }

    weights
}

/// Get the inverse dequant matrix for DCT8 encoding.
///
/// For encoding, we use the perceptual weights directly.
/// In libjxl, InvDequantMatrix() returns the weights themselves (not 1/weight).
pub fn get_dct8_inv_dequant() -> [f32; 64] {
    let weights = generate_dct8_weights();
    let mut inv_dequant = [0.0f32; 64];

    // Use Y channel weights (index 1) as it's the luminance channel
    // This is a simplification - proper implementation would weight all 3 channels
    for i in 0..64 {
        let weight = weights[64 + i]; // Y channel
        inv_dequant[i] = weight; // Use weight directly for quantization
    }

    inv_dequant
}

/// Get per-channel inverse dequant matrices for DCT8 encoding.
///
/// Returns [X_inv_dequant, Y_inv_dequant, B_inv_dequant].
///
/// In libjxl terminology:
/// - "InvDequantMatrix" (used for quantization/encoding) contains the weights themselves
/// - "DequantMatrix" (used for dequantization/decoding) contains 1/weight
///
/// So for ENCODING, we return the weights directly.
pub fn get_dct8_inv_dequant_per_channel() -> [[f32; 64]; 3] {
    let weights = generate_dct8_weights();
    let mut result = [[0.0f32; 64]; 3];

    for c in 0..3 {
        for i in 0..64 {
            let weight = weights[c * 64 + i];
            // Use weight directly for quantization (encoding).
            // libjxl stores this as "inv_table" which is used by InvDequantMatrix().
            result[c][i] = weight;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_band_mult() {
        // Positive values: 1 + v
        assert!((band_mult(0.5) - 1.5).abs() < 1e-6);
        assert!((band_mult(1.0) - 2.0).abs() < 1e-6);

        // Negative values: 1 / (1 - v)
        assert!((band_mult(-0.5) - 2.0 / 3.0).abs() < 1e-6);
        assert!((band_mult(-1.0) - 0.5).abs() < 1e-6);

        // Zero: edge case
        assert!((band_mult(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_dct8_weights() {
        let weights = generate_dct8_weights();

        // Check array size
        assert_eq!(weights.len(), 3 * 64);

        // DC coefficient (position 0,0) should have highest weight per channel
        for c in 0..3 {
            let dc_weight = weights[c * 64];
            // DC weight should be close to the first band value
            assert!(
                dc_weight > 100.0,
                "DC weight for channel {} = {}",
                c,
                dc_weight
            );

            // High frequency corners should have lower weight
            let hf_weight = weights[c * 64 + 63]; // position (7,7)
            assert!(
                hf_weight < dc_weight,
                "HF weight {} should be < DC weight {} for channel {}",
                hf_weight,
                dc_weight,
                c
            );
        }

        // Y channel (c=1) DC should be ~560
        let y_dc = weights[64];
        assert!(
            (y_dc - 560.0).abs() < 1.0,
            "Y channel DC weight = {}, expected ~560",
            y_dc
        );
    }

    #[test]
    fn test_get_dct8_inv_dequant() {
        let inv_dequant = get_dct8_inv_dequant();

        // Check array size
        assert_eq!(inv_dequant.len(), 64);

        // DC coefficient should have highest weight (perceptually most important)
        // HF coefficient should have lower weight (perceptually less important)
        let dc_weight = inv_dequant[0];
        let hf_weight = inv_dequant[63];

        // DC weight > HF weight (DC is more important)
        assert!(
            dc_weight > hf_weight,
            "DC weight {} should be > HF weight {}",
            dc_weight,
            hf_weight
        );

        // Weights should be reasonable (Y channel: DC~560, HF~196)
        assert!(dc_weight > 100.0, "DC weight should be > 100");
        assert!(hf_weight > 100.0, "HF weight should be > 100");
    }

    #[test]
    fn test_get_dct8_inv_dequant_per_channel() {
        let inv_dequant = get_dct8_inv_dequant_per_channel();

        // Check array sizes
        assert_eq!(inv_dequant.len(), 3);
        for c in 0..3 {
            assert_eq!(inv_dequant[c].len(), 64);
        }

        // Each channel should have DC weight > HF weight
        // (DC is perceptually more important)
        for c in 0..3 {
            let dc_weight = inv_dequant[c][0];
            let hf_weight = inv_dequant[c][63];
            assert!(
                dc_weight > hf_weight,
                "Channel {} DC weight {} should be > HF weight {}",
                c,
                dc_weight,
                hf_weight
            );

            // Weights should be reasonable (> 100)
            assert!(dc_weight > 100.0, "Channel {} DC weight should be > 100", c);
        }
    }

    #[test]
    fn test_lf_quant_values() {
        assert!((LF_QUANT[0] - 1.0 / 4096.0).abs() < 1e-10);
        assert!((LF_QUANT[1] - 1.0 / 512.0).abs() < 1e-10);
        assert!((LF_QUANT[2] - 1.0 / 256.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv_lf_quant_reciprocal() {
        for i in 0..3 {
            assert!((LF_QUANT[i] * INV_LF_QUANT[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_quant_table_from_ac_strategy() {
        assert_eq!(
            QuantTable::from_ac_strategy(AcStrategy::Dct8x8),
            QuantTable::Dct8
        );
        assert_eq!(
            QuantTable::from_ac_strategy(AcStrategy::Identity),
            QuantTable::Identity
        );
        assert_eq!(
            QuantTable::from_ac_strategy(AcStrategy::Dct16x16),
            QuantTable::Dct16
        );
        // Rectangular transforms share tables
        assert_eq!(
            QuantTable::from_ac_strategy(AcStrategy::Dct16x8),
            QuantTable::from_ac_strategy(AcStrategy::Dct8x16)
        );
    }

    #[test]
    fn test_quant_table_sizes() {
        // DCT8: 1x1 blocks = 64 coefficients
        assert_eq!(QuantTable::Dct8.num_coefficients(), 64);
        // DCT16: 2x2 blocks = 256 coefficients
        assert_eq!(QuantTable::Dct16.num_coefficients(), 256);
        // DCT32: 4x4 blocks = 1024 coefficients
        assert_eq!(QuantTable::Dct32.num_coefficients(), 1024);
        // DCT256: 32x32 blocks = 65536 coefficients
        assert_eq!(QuantTable::Dct256.num_coefficients(), 65536);
    }

    #[test]
    fn test_dequant_matrices_default() {
        let dm = DequantMatrices::default();
        assert!(dm.is_all_default());
    }

    #[test]
    fn test_lf_quant_factors() {
        let lf = LfQuantFactors::new();
        assert_eq!(lf.quant(0), LF_QUANT[0]);
        assert_eq!(lf.quant(1), LF_QUANT[1]);
        assert_eq!(lf.quant(2), LF_QUANT[2]);
    }
}
