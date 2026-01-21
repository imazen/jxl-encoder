//! AC Strategy (transform type) definitions for VarDCT encoding.
//!
//! AC Strategy determines which DCT transform size is used for each block.
//! JPEG XL supports 27 different transform types from 2x2 to 256x256.

use crate::BLOCK_DIM;

/// Maximum number of 8x8 coefficient blocks covered by any transform.
pub const MAX_COEFF_BLOCKS: usize = 32;

/// Maximum block dimension in pixels (256 for DCT256x256).
pub const MAX_BLOCK_DIM: usize = BLOCK_DIM * MAX_COEFF_BLOCKS;

/// Number of distinct transform types.
pub const NUM_TRANSFORM_TYPES: usize = 27;

/// Number of coefficient orders (one per unique block shape).
pub const NUM_ORDERS: usize = 13;

/// AC Strategy (transform type) for VarDCT encoding.
///
/// Each variant represents a different DCT transform size and shape.
/// The enum values match the JPEG XL specification.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AcStrategy {
    /// Standard 8x8 DCT (default, like JPEG)
    #[default]
    Dct8x8 = 0,
    /// Identity transform (no DCT, encode pixels directly). Also called "Hornuss".
    Identity = 1,
    /// 2x2 DCT (Hadamard-like)
    Dct2x2 = 2,
    /// 4x4 DCT
    Dct4x4 = 3,
    /// 16x16 DCT (covers 2x2 blocks)
    Dct16x16 = 4,
    /// 32x32 DCT (covers 4x4 blocks)
    Dct32x32 = 5,
    /// 16x8 DCT (rectangular)
    Dct16x8 = 6,
    /// 8x16 DCT (rectangular)
    Dct8x16 = 7,
    /// 32x8 DCT (rectangular)
    Dct32x8 = 8,
    /// 8x32 DCT (rectangular)
    Dct8x32 = 9,
    /// 32x16 DCT (rectangular)
    Dct32x16 = 10,
    /// 16x32 DCT (rectangular)
    Dct16x32 = 11,
    /// 4x8 DCT
    Dct4x8 = 12,
    /// 8x4 DCT
    Dct8x4 = 13,
    /// Corner-DCT variant 0 (AFV = Adaptive Frequency-Varying)
    Afv0 = 14,
    /// Corner-DCT variant 1
    Afv1 = 15,
    /// Corner-DCT variant 2
    Afv2 = 16,
    /// Corner-DCT variant 3
    Afv3 = 17,
    /// 64x64 DCT (covers 8x8 blocks)
    Dct64x64 = 18,
    /// 64x32 DCT (rectangular)
    Dct64x32 = 19,
    /// 32x64 DCT (rectangular)
    Dct32x64 = 20,
    /// 128x128 DCT (covers 16x16 blocks)
    Dct128x128 = 21,
    /// 128x64 DCT (rectangular)
    Dct128x64 = 22,
    /// 64x128 DCT (rectangular)
    Dct64x128 = 23,
    /// 256x256 DCT (covers 32x32 blocks)
    Dct256x256 = 24,
    /// 256x128 DCT (rectangular)
    Dct256x128 = 25,
    /// 128x256 DCT (rectangular)
    Dct128x256 = 26,
}

impl AcStrategy {
    /// All AC strategy values in order.
    pub const ALL: [AcStrategy; NUM_TRANSFORM_TYPES] = [
        AcStrategy::Dct8x8,
        AcStrategy::Identity,
        AcStrategy::Dct2x2,
        AcStrategy::Dct4x4,
        AcStrategy::Dct16x16,
        AcStrategy::Dct32x32,
        AcStrategy::Dct16x8,
        AcStrategy::Dct8x16,
        AcStrategy::Dct32x8,
        AcStrategy::Dct8x32,
        AcStrategy::Dct32x16,
        AcStrategy::Dct16x32,
        AcStrategy::Dct4x8,
        AcStrategy::Dct8x4,
        AcStrategy::Afv0,
        AcStrategy::Afv1,
        AcStrategy::Afv2,
        AcStrategy::Afv3,
        AcStrategy::Dct64x64,
        AcStrategy::Dct64x32,
        AcStrategy::Dct32x64,
        AcStrategy::Dct128x128,
        AcStrategy::Dct128x64,
        AcStrategy::Dct64x128,
        AcStrategy::Dct256x256,
        AcStrategy::Dct256x128,
        AcStrategy::Dct128x256,
    ];

    /// Create from raw index, returns None if out of range.
    pub fn from_index(idx: usize) -> Option<AcStrategy> {
        AcStrategy::ALL.get(idx).copied()
    }

    /// Get the raw index value.
    pub fn index(self) -> usize {
        self as usize
    }

    /// Number of 8x8 coefficient blocks covered in X direction.
    pub fn covered_blocks_x(self) -> u32 {
        const LUT: [u32; NUM_TRANSFORM_TYPES] = [
            1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32,
        ];
        LUT[self as usize]
    }

    /// Number of 8x8 coefficient blocks covered in Y direction.
    pub fn covered_blocks_y(self) -> u32 {
        const LUT: [u32; NUM_TRANSFORM_TYPES] = [
            1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16,
        ];
        LUT[self as usize]
    }

    /// Transform width in pixels.
    pub fn width(self) -> usize {
        self.covered_blocks_x() as usize * BLOCK_DIM
    }

    /// Transform height in pixels.
    pub fn height(self) -> usize {
        self.covered_blocks_y() as usize * BLOCK_DIM
    }

    /// Total number of coefficients for this transform.
    pub fn num_coefficients(self) -> usize {
        self.width() * self.height()
    }

    /// Block shape ID (used for coefficient ordering).
    /// Transforms with the same shape share coefficient orders.
    pub fn block_shape_id(self) -> u32 {
        const LUT: [u32; NUM_TRANSFORM_TYPES] = [
            0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
        ];
        LUT[self as usize]
    }

    /// Order ID for coefficient scan order selection.
    /// Maps transform types to the 13 distinct coefficient orders.
    pub fn order_id(self) -> usize {
        const LUT: [usize; NUM_TRANSFORM_TYPES] = [
            0,  // Dct8x8
            1,  // Identity
            1,  // Dct2x2
            1,  // Dct4x4
            2,  // Dct16x16
            3,  // Dct32x32
            4,  // Dct16x8
            4,  // Dct8x16
            5,  // Dct32x8
            5,  // Dct8x32
            6,  // Dct32x16
            6,  // Dct16x32
            1,  // Dct4x8
            1,  // Dct8x4
            1,  // Afv0
            1,  // Afv1
            1,  // Afv2
            1,  // Afv3
            7,  // Dct64x64
            8,  // Dct64x32
            8,  // Dct32x64
            9,  // Dct128x128
            10, // Dct128x64
            10, // Dct64x128
            11, // Dct256x256
            12, // Dct256x128
            12, // Dct128x256
        ];
        LUT[self as usize]
    }

    /// Number of 8x8 blocks covered as (width, height) tuple.
    pub fn covered_blocks(self) -> (usize, usize) {
        (
            self.covered_blocks_x() as usize,
            self.covered_blocks_y() as usize,
        )
    }

    /// Log2 of covered blocks (total area in 8x8 units).
    pub fn log2_covered_blocks(self) -> u32 {
        let blocks = self.covered_blocks_x() * self.covered_blocks_y();
        blocks.trailing_zeros()
    }

    /// Whether this transform uses rectangular (non-square) blocks.
    pub fn is_rectangular(self) -> bool {
        self.covered_blocks_x() != self.covered_blocks_y()
    }

    /// Whether this is an AFV (corner-DCT) transform.
    pub fn is_afv(self) -> bool {
        matches!(
            self,
            AcStrategy::Afv0 | AcStrategy::Afv1 | AcStrategy::Afv2 | AcStrategy::Afv3
        )
    }

    /// Whether this transform needs transpose for encoding.
    /// Applies to transforms where width > height.
    pub fn needs_transpose(self) -> bool {
        self.covered_blocks_x() < self.covered_blocks_y()
    }

    /// Dequant matrix parameter index for this transform.
    /// Used to look up quantization tables.
    pub fn dequant_matrix_index(self) -> usize {
        const LUT: [usize; NUM_TRANSFORM_TYPES] = [
            0,  // Dct8x8
            1,  // Identity
            2,  // Dct2x2
            3,  // Dct4x4
            4,  // Dct16x16
            5,  // Dct32x32
            6,  // Dct16x8
            6,  // Dct8x16
            7,  // Dct32x8
            7,  // Dct8x32
            8,  // Dct32x16
            8,  // Dct16x32
            9,  // Dct4x8
            9,  // Dct8x4
            10, // Afv0
            10, // Afv1
            10, // Afv2
            10, // Afv3
            11, // Dct64x64
            12, // Dct64x32
            12, // Dct32x64
            13, // Dct128x128
            14, // Dct128x64
            14, // Dct64x128
            15, // Dct256x256
            16, // Dct256x128
            16, // Dct128x256
        ];
        LUT[self as usize]
    }
}

/// Transform types that have their own coefficient order.
/// These are the "primary" transforms; other transforms share orders with these.
pub const ORDER_TRANSFORM_TYPES: [AcStrategy; NUM_ORDERS] = [
    AcStrategy::Dct8x8,
    AcStrategy::Identity,
    AcStrategy::Dct16x16,
    AcStrategy::Dct32x32,
    AcStrategy::Dct8x16,
    AcStrategy::Dct8x32,
    AcStrategy::Dct16x32,
    AcStrategy::Dct64x64,
    AcStrategy::Dct32x64,
    AcStrategy::Dct128x128,
    AcStrategy::Dct64x128,
    AcStrategy::Dct256x256,
    AcStrategy::Dct128x256,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_variants_in_order() {
        for (i, &strategy) in AcStrategy::ALL.iter().enumerate() {
            assert_eq!(strategy.index(), i, "Index mismatch for {:?}", strategy);
        }
    }

    #[test]
    fn test_from_index() {
        for i in 0..NUM_TRANSFORM_TYPES {
            let strategy = AcStrategy::from_index(i);
            assert!(strategy.is_some(), "Missing strategy for index {}", i);
            assert_eq!(strategy.unwrap().index(), i);
        }
        assert!(AcStrategy::from_index(NUM_TRANSFORM_TYPES).is_none());
    }

    #[test]
    fn test_covered_blocks_dct8x8() {
        let dct8 = AcStrategy::Dct8x8;
        assert_eq!(dct8.covered_blocks_x(), 1);
        assert_eq!(dct8.covered_blocks_y(), 1);
        assert_eq!(dct8.width(), 8);
        assert_eq!(dct8.height(), 8);
        assert_eq!(dct8.num_coefficients(), 64);
    }

    #[test]
    fn test_covered_blocks_dct256() {
        let dct256 = AcStrategy::Dct256x256;
        assert_eq!(dct256.covered_blocks_x(), 32);
        assert_eq!(dct256.covered_blocks_y(), 32);
        assert_eq!(dct256.width(), 256);
        assert_eq!(dct256.height(), 256);
        assert_eq!(dct256.num_coefficients(), 65536);
    }

    #[test]
    fn test_rectangular_detection() {
        assert!(!AcStrategy::Dct8x8.is_rectangular());
        assert!(!AcStrategy::Dct16x16.is_rectangular());
        assert!(AcStrategy::Dct16x8.is_rectangular());
        assert!(AcStrategy::Dct8x16.is_rectangular());
        assert!(AcStrategy::Dct32x8.is_rectangular());
    }

    #[test]
    fn test_afv_detection() {
        assert!(!AcStrategy::Dct8x8.is_afv());
        assert!(!AcStrategy::Identity.is_afv());
        assert!(AcStrategy::Afv0.is_afv());
        assert!(AcStrategy::Afv1.is_afv());
        assert!(AcStrategy::Afv2.is_afv());
        assert!(AcStrategy::Afv3.is_afv());
    }

    #[test]
    fn test_log2_covered_blocks() {
        assert_eq!(AcStrategy::Dct8x8.log2_covered_blocks(), 0); // 1 block
        assert_eq!(AcStrategy::Dct16x16.log2_covered_blocks(), 2); // 4 blocks
        assert_eq!(AcStrategy::Dct32x32.log2_covered_blocks(), 4); // 16 blocks
        assert_eq!(AcStrategy::Dct256x256.log2_covered_blocks(), 10); // 1024 blocks
    }

    #[test]
    fn test_order_ids_valid() {
        for &strategy in &AcStrategy::ALL {
            let order_id = strategy.order_id();
            assert!(
                order_id < NUM_ORDERS,
                "Order ID {} out of range for {:?}",
                order_id,
                strategy
            );
        }
    }

    #[test]
    fn test_dequant_matrix_indices_valid() {
        for &strategy in &AcStrategy::ALL {
            let idx = strategy.dequant_matrix_index();
            // There are 17 unique dequant matrices (indices 0-16)
            assert!(
                idx < 17,
                "Dequant index {} out of range for {:?}",
                idx,
                strategy
            );
        }
    }

    #[test]
    fn test_order_transform_types() {
        for (i, &strategy) in ORDER_TRANSFORM_TYPES.iter().enumerate() {
            assert_eq!(
                strategy.order_id(),
                i,
                "ORDER_TRANSFORM_TYPES[{}] = {:?} has wrong order_id",
                i,
                strategy
            );
        }
    }
}
