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

/// Horizontal shift for each jpeg_upsampling mode.
/// Mode 0: no subsampling, 1: 4:2:0, 2: 4:2:2, 3: 4:4:0
pub const JPEG_UPSAMPLING_H_SHIFT: [usize; 4] = [0, 1, 1, 0];

/// Vertical shift for each jpeg_upsampling mode.
/// Mode 0: no subsampling, 1: 4:2:0, 2: 4:2:2, 3: 4:4:0
pub const JPEG_UPSAMPLING_V_SHIFT: [usize; 4] = [0, 1, 0, 1];

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

/// Return an uninitialized `[f32; N]` buffer.
///
/// With the `unsafe-performance` feature, this skips the memset zero-fill and
/// returns memory with indeterminate contents via [`core::mem::MaybeUninit`].
/// **Every caller MUST write all `N` positions before reading any of them.**
///
/// Without the feature (default), this returns `[0.0f32; N]`.
#[cfg(feature = "unsafe-performance")]
#[allow(unsafe_code, clippy::uninit_assumed_init)]
#[inline(always)]
pub fn uninit_buf<const N: usize>() -> [f32; N] {
    // SAFETY: All call sites write every element via extract_block_* / DCT /
    // IDCT before any read.  f32 has no trap representations on IEEE 754
    // hardware, and LLVM treats the bytes as "undef" which is exactly what we
    // want — the dead-store memset is eliminated.
    unsafe { core::mem::MaybeUninit::<[f32; N]>::uninit().assume_init() }
}

/// Return a zero-initialized `[f32; N]` buffer (safe default path).
#[cfg(not(feature = "unsafe-performance"))]
#[inline(always)]
pub fn uninit_buf<const N: usize>() -> [f32; N] {
    [0.0f32; N]
}

/// Convert `&slice[offset..offset+N]` to `&[f32; N]` without bounds checking.
///
/// With `unsafe-performance`, skips the slice range check and `try_into` length check.
/// **Caller MUST ensure `offset + N <= slice.len()`.**
///
/// Without the feature, falls back to `slice[offset..offset+N].try_into().unwrap()`.
#[cfg(feature = "unsafe-performance")]
#[allow(unsafe_code)]
#[inline(always)]
pub fn as_array_ref<const N: usize>(slice: &[f32], offset: usize) -> &[f32; N] {
    debug_assert!(offset + N <= slice.len());
    // SAFETY: caller guarantees offset + N <= slice.len()
    unsafe { &*(slice.as_ptr().add(offset) as *const [f32; N]) }
}

#[cfg(not(feature = "unsafe-performance"))]
#[inline(always)]
pub fn as_array_ref<const N: usize>(slice: &[f32], offset: usize) -> &[f32; N] {
    slice[offset..offset + N].try_into().unwrap()
}

/// Convert `&mut slice[offset..offset+N]` to `&mut [f32; N]` without bounds checking.
///
/// With `unsafe-performance`, skips the slice range check and `try_into` length check.
/// **Caller MUST ensure `offset + N <= slice.len()`.**
///
/// Without the feature, falls back to `(&mut slice[offset..offset+N]).try_into().unwrap()`.
#[cfg(feature = "unsafe-performance")]
#[allow(unsafe_code)]
#[inline(always)]
pub fn as_array_mut<const N: usize>(slice: &mut [f32], offset: usize) -> &mut [f32; N] {
    debug_assert!(offset + N <= slice.len());
    // SAFETY: caller guarantees offset + N <= slice.len()
    unsafe { &mut *(slice.as_mut_ptr().add(offset) as *mut [f32; N]) }
}

#[cfg(not(feature = "unsafe-performance"))]
#[inline(always)]
pub fn as_array_mut<const N: usize>(slice: &mut [f32], offset: usize) -> &mut [f32; N] {
    (&mut slice[offset..offset + N]).try_into().unwrap()
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
