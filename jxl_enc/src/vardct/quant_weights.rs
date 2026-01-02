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

#[cfg(test)]
mod tests {
    use super::*;

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
