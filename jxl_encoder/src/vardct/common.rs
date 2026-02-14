// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Common constants and helper functions for the VarDCT encoder.
//!
//! These are ported from libjxl-tiny and will be used as encoding is implemented.

#![allow(dead_code)]

/// Block dimension (8 pixels).
pub const BLOCK_DIM: usize = 8;

/// DCT block size (64 coefficients).
pub const DCT_BLOCK_SIZE: usize = BLOCK_DIM * BLOCK_DIM;

/// Group dimension in pixels (256x256).
pub const GROUP_DIM: usize = 256;

/// Group dimension in blocks (32x32).
pub const GROUP_DIM_IN_BLOCKS: usize = GROUP_DIM / BLOCK_DIM;

/// DC group dimension (8 groups = 2048 pixels).
pub const DC_GROUP_DIM: usize = GROUP_DIM * BLOCK_DIM;

/// DC group dimension in blocks (256 blocks).
pub const DC_GROUP_DIM_IN_BLOCKS: usize = DC_GROUP_DIM / BLOCK_DIM;

/// Tile dimension for chroma-from-luma (64 pixels when enabled).
pub const TILE_DIM: usize = 64;

/// Tile dimension in blocks.
pub const TILE_DIM_IN_BLOCKS: usize = TILE_DIM / BLOCK_DIM;

/// Group dimension in tiles.
pub const GROUP_DIM_IN_TILES: usize = GROUP_DIM / TILE_DIM;

/// Bits per byte.
pub const BITS_PER_BYTE: usize = 8;

/// Divide and round up.
#[inline]
pub const fn div_ceil(a: usize, b: usize) -> usize {
    // Using a.div_ceil(b) is not const-stable yet, so we use this pattern
    // Note: Rust 1.93+ has const div_ceil but we keep this for compatibility
    #[allow(clippy::manual_div_ceil)]
    {
        (a + b - 1) / b
    }
}

/// Round up to a multiple.
#[inline]
pub const fn round_up_to(what: usize, align: usize) -> usize {
    div_ceil(what, align) * align
}

/// Clamp a value to a range.
#[inline]
pub fn clamp<T: PartialOrd>(val: T, low: T, hi: T) -> T {
    if val < low {
        low
    } else if val > hi {
        hi
    } else {
        val
    }
}

/// Encode signed integer as unsigned (zig-zag encoding).
/// Encodes non-negative (X) into (2 * X), negative (-X) into (2 * X - 1).
#[inline]
pub const fn pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ (((!(value as u32)) >> 31).wrapping_sub(1))
}

/// Ceiling log2 of a non-zero value.
#[inline]
pub const fn ceil_log2_nonzero(n: usize) -> u32 {
    if n <= 1 {
        0
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

/// Floor log2 of a non-zero value.
#[inline]
pub const fn floor_log2_nonzero(n: u32) -> u32 {
    31 - n.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_signed() {
        assert_eq!(pack_signed(0), 0);
        assert_eq!(pack_signed(1), 2);
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(2), 4);
        assert_eq!(pack_signed(-2), 3);
        assert_eq!(pack_signed(3), 6);
        assert_eq!(pack_signed(-3), 5);
    }

    #[test]
    fn test_ceil_log2_nonzero() {
        assert_eq!(ceil_log2_nonzero(1), 0);
        assert_eq!(ceil_log2_nonzero(2), 1);
        assert_eq!(ceil_log2_nonzero(3), 2);
        assert_eq!(ceil_log2_nonzero(4), 2);
        assert_eq!(ceil_log2_nonzero(5), 3);
        assert_eq!(ceil_log2_nonzero(8), 3);
        assert_eq!(ceil_log2_nonzero(9), 4);
    }

    #[test]
    fn test_floor_log2_nonzero() {
        assert_eq!(floor_log2_nonzero(1), 0);
        assert_eq!(floor_log2_nonzero(2), 1);
        assert_eq!(floor_log2_nonzero(3), 1);
        assert_eq!(floor_log2_nonzero(4), 2);
        assert_eq!(floor_log2_nonzero(7), 2);
        assert_eq!(floor_log2_nonzero(8), 3);
        assert_eq!(floor_log2_nonzero(16), 4);
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(0, 8), 0);
        assert_eq!(div_ceil(1, 8), 1);
        assert_eq!(div_ceil(8, 8), 1);
        assert_eq!(div_ceil(9, 8), 2);
        assert_eq!(div_ceil(16, 8), 2);
        assert_eq!(div_ceil(256, 8), 32);
    }
}
