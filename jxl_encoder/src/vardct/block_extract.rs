// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Block pixel extraction from image planes.
//!
//! Extracts rectangular pixel blocks from padded image buffers.
//! Uses `copy_from_slice` for row-wide copies (single bounds check per row).

use super::common::BLOCK_DIM;

/// Extract an 8×8 pixel block from a plane.
///
/// The buffer must be padded to at least (by*8+8) rows and (bx*8+8) columns
/// with edge-replicated values, so no bounds checking is needed.
#[inline(always)]
pub(super) fn extract_block_8x8(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 64],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..8 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 8..dy * 8 + 8].copy_from_slice(&plane[src..src + 8]);
    }
}

/// Extract an 8×16 pixel block (1 wide × 2 tall) for DCT16x8.
/// Layout: 16 rows × 8 cols, row-major.
///
/// The buffer must be padded to at least (by*8+16) rows and (bx*8+8) columns.
#[inline(always)]
pub(super) fn extract_block_8x16(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 128],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..16 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 8..dy * 8 + 8].copy_from_slice(&plane[src..src + 8]);
    }
}

/// Extract a 16×8 pixel block (2 wide × 1 tall) for DCT8x16.
/// Layout: 8 rows × 16 cols, row-major.
///
/// The buffer must be padded to at least (by*8+8) rows and (bx*8+16) columns.
#[inline(always)]
pub(super) fn extract_block_16x8(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 128],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..8 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 16..dy * 16 + 16].copy_from_slice(&plane[src..src + 16]);
    }
}

/// Extract a 16×16 pixel block (2 wide × 2 tall) for DCT16x16.
/// Layout: 16 rows × 16 cols, row-major.
///
/// The buffer must be padded to at least (by*8+16) rows and (bx*8+16) columns.
#[inline(always)]
pub(super) fn extract_block_16x16(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 256],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..16 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 16..dy * 16 + 16].copy_from_slice(&plane[src..src + 16]);
    }
}

/// Extract a 32×32 pixel block (4 wide × 4 tall) for DCT32x32.
/// Layout: 32 rows × 32 cols, row-major.
///
/// The buffer must be padded to at least (by*8+32) rows and (bx*8+32) columns.
#[inline(always)]
pub(super) fn extract_block_32x32(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 1024],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..32 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 32..dy * 32 + 32].copy_from_slice(&plane[src..src + 32]);
    }
}

/// Extract a 32×16 pixel block (2 wide × 4 tall) for DCT32x16.
/// Layout: 32 rows × 16 cols, row-major.
///
/// The buffer must be padded to at least (by*8+32) rows and (bx*8+16) columns.
#[inline(always)]
pub(super) fn extract_block_32x16(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 512],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..32 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 16..dy * 16 + 16].copy_from_slice(&plane[src..src + 16]);
    }
}

/// Extract a 16×32 pixel block (4 wide × 2 tall) for DCT16x32.
/// Layout: 16 rows × 32 cols, row-major.
///
/// The buffer must be padded to at least (by*8+16) rows and (bx*8+32) columns.
#[inline(always)]
pub(super) fn extract_block_16x32(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 512],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..16 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 32..dy * 32 + 32].copy_from_slice(&plane[src..src + 32]);
    }
}

/// Extract a 64×64 pixel block (8 wide × 8 tall) for DCT64x64.
/// Layout: 64 rows × 64 cols, row-major.
#[inline(always)]
pub(super) fn extract_block_64x64(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 4096],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..64 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 64..dy * 64 + 64].copy_from_slice(&plane[src..src + 64]);
    }
}

/// Extract a 64×32 pixel block (4 wide × 8 tall) for DCT64x32.
/// Layout: 64 rows × 32 cols, row-major.
#[inline(always)]
pub(super) fn extract_block_64x32(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 2048],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..64 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 32..dy * 32 + 32].copy_from_slice(&plane[src..src + 32]);
    }
}

/// Extract a 32×64 pixel block (8 wide × 4 tall) for DCT32x64.
/// Layout: 32 rows × 64 cols, row-major.
#[inline(always)]
pub(super) fn extract_block_32x64(
    plane: &[f32],
    stride: usize,
    bx: usize,
    by: usize,
    out: &mut [f32; 2048],
) {
    let x0 = bx * BLOCK_DIM;
    for dy in 0..32 {
        let src = (by * BLOCK_DIM + dy) * stride + x0;
        out[dy * 64..dy * 64 + 64].copy_from_slice(&plane[src..src + 64]);
    }
}
